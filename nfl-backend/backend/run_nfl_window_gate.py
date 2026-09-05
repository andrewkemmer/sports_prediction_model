"""NFL window gate — W2014 / W2016 through the MLB-aligned within-run
incumbent gate (nfl_moneyline.adopt_decision, fully-within-run as of the
2026-09-02 revision).

Background: the feature program concluded the market-free 12-pool is the
ceiling at the current 1,960-game decided frame, and the window-extension
ablation (32c338e) measured W2014/W2016 with the HARNESS geometry
(arm-vs-W2019 relative ECE). This harness measures the same candidates
through the PRODUCTION gate instead, under its FULLY within-run baseline:

  POOLED  — the production-config 12-pool re-fit in the candidate's OWN
            fold loop, on each fold's strictly-prior training portion
            restricted to the PRODUCTION window (INCUMBENT_MIN_SEASON=2019;
            same seed, same folds, same A/B walk state by construction).
  SEALED  — one production-config re-fit on all pre-2025 rows of the
            current pull (restricted to 2019+), scored on sealed-2025 with
            the candidate's shared Platt map.

ADOPT = within TOLERANCE of that within-run incumbent on BOTH pooled and
sealed for ALL THREE metrics (TOL_LL 0.012, TOL_AUC 0.016, ECE_TOL 0.01),
each of the six conditions BLOCKING — nothing else, and NO advisory verdict
mode (the within-run baseline always exists). The persisted served bundle
(seeded on origin/main as ensemble_latest.joblib per 9f88206, guarded load
per the 2026-09-02 revision) is a DIAGNOSTIC CROSS-CHECK only: it is
re-scored on sealed with its own stored weights + Platt map and compared to
the within-run sealed incumbent so cross-pull drift becomes visible — it
never enters the verdict.

Candidates (same 12-pool, market-free, identical walk-forward geometry —
2021-2024 pooled folds, sealed-2025 holdout, same seed):
  W2016 — warmup 2015, core 2016-2025 (train 2016..2024)
  W2014 — warmup 2013, core 2014-2025 (train 2014..2024)

NO production config change ships from this harness: FEATURE_COLUMNS /
DEFAULT_SEASONS are untouched regardless of outcome (wiring the window is a
separate decision).

Usage (network + nflreadpy needed for the raw pull):
    python3 run_nfl_window_gate.py                      # both arms + record
    python3 run_nfl_window_gate.py --arms W2016
    python3 run_nfl_window_gate.py --features <features.csv>
    python3 run_nfl_window_gate.py --no-incumbent       # skip bundle cross-check
    python3 run_nfl_window_gate.py --no-record
Artifact: data_delivery/nfl_window_gate_<sha>.json (examined before any
commit; the evidence record is committed with the harness per convention).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from run_feature_winpct_ablation import DEPLOYED_12
from run_tier1_ablation import _frame_sha256
import nfl_moneyline as M  # the production gate module — shared constants

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"

# Candidate windows — CORE starts at the boundary; the warmup year is B-1.
BOUNDARIES: dict[str, int] = {"W2016": 2016, "W2014": 2014}
SEALED_SEASON = 2025           # sealed hold-out — constant across candidates
TRAIN_END = SEALED_SEASON - 1  # train window is [B .. 2024]
VAL_SEASONS = [2021, 2022, 2023, 2024]  # pooled-OOF weeks — constant
# The within-run incumbent's training rows are restricted to the PRODUCTION
# window (production TRAIN_SEASONS start) so a wider-window candidate is
# compared against the model it would replace, trained in the SAME run on
# the SAME folds + pull.
INCUMBENT_MIN_SEASON = 2019

# The gate's relative tolerances, SHARED with the production gate (single
# source of truth — run tests pin the identity): logloss, AUC, ECE.
ECE_TOL = M.ECE_TOL
TOL_LL = M.TOL_LL
TOL_AUC = M.TOL_AUC


# ---------------------------------------------------------------------------
# Raw pull + decided frame + feature build (mirrors run_nfl_window_ablation)
# ---------------------------------------------------------------------------
def pull_raw(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """nflreadpy schedule + column-trimmed pbp for ``seasons`` (pandas)."""
    import nflreadpy
    from nfl_features import TIER1_NEEDS
    sched = nflreadpy.load_schedules(seasons).to_pandas()
    pbp = nflreadpy.load_pbp(seasons)
    keep = [c for c in (("game_id", "play_id", "posteam", "yards_gained",
                         "epa", "qb_epa", "game_seconds_remaining")
                        + TIER1_NEEDS) if c in pbp.columns]
    pbp = pbp.select(keep).to_pandas()
    return sched, pbp


def build_decided_frame(sched: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Canonical decided frame over the pull window (nfl_game_frame rules)."""
    from nfl_game_frame import aggregate_game_frame, canonical_decided_frame
    return canonical_decided_frame(aggregate_game_frame(sched, pbp))


def arm_features(feats: pd.DataFrame) -> list[str]:
    """The served 12-pool, kept only where the frame carries the column."""
    return [c for c in DEPLOYED_12 if c in feats.columns]


def load_arm_features(boundary: int) -> pd.DataFrame:
    """Build the feature frame for one candidate window (warmup B-1 + core
    B..2025) — end-to-end on the candidate's window, exactly like the
    production build path."""
    from nfl_features import build_features
    seasons = list(range(boundary - 1, SEALED_SEASON + 1))
    sched, pbp = pull_raw(seasons)
    decided = build_decided_frame(sched, pbp)
    feats = build_features(decided, sched, pbp)
    if "home_win" not in feats.columns:
        feats["home_win"] = (feats["home_score"] > feats["away_score"]).astype(int)
    return feats


# ---------------------------------------------------------------------------
# Within-run incumbent helpers
# ---------------------------------------------------------------------------
def incumbent_predict(sld: pd.DataFrame, bundle: dict,
                      inc_features: list[str]) -> np.ndarray:
    """Blend the bundle's members with its stored adaptive weights and its
    stored Platt map — the probabilities the board actually served.

    A PURE function of (features, bundle): the predictions never depend on
    the target column, which is the within-run-isolation property the tests
    pin (corrupting sealed outcomes leaves predictions byte-identical; only
    the scored metrics change). Used for the DIAGNOSTIC bundle cross-check
    only — never the verdict.
    """
    _, members, _ = M.ensemble_predict(bundle["models"], sld,
                                       features=inc_features)
    iw = M._member_weights(list(members),
                           adaptive=bundle.get("adaptive_weights"))
    ib = np.zeros(len(sld))
    for name, p in members.items():
        ib += iw[name] * np.asarray(p, dtype=float)
    if bundle.get("platt") is not None:
        ib = M.platt_predict(ib, bundle.get("platt"))
    return ib


def _bundle_crosscheck(sld: pd.DataFrame, bundle: dict | None,
                       within_run_sealed: dict) -> dict | None:
    """Diagnostic-only: score the persisted served bundle on the sealed rows
    of this pull and compare it to the within-run sealed incumbent.

    The bundle NEVER enters the verdict — the within-run incumbent is the
    gate baseline for BOTH views. Unusable bundle -> None (never misleads).
    """
    if bundle is None:
        return None
    inc_features = [f for f in (bundle.get("features") or [])
                    if f in sld.columns]
    need = len(bundle.get("features") or [])
    if not (bundle.get("models") and need and len(inc_features) == need):
        logger.warning("bundle cross-check skipped: %d/%d features present",
                       len(inc_features), need)
        return None
    try:
        ics = incumbent_predict(sld, bundle, inc_features)
        b_val = {
            "logloss": round(M.logloss(sld[M.TARGET], ics), 4),
            "auc": round(M.auc(sld[M.TARGET], ics), 4),
            "ece": round(M.ece(sld[M.TARGET], ics), 4),
        }
        base = {k: within_run_sealed.get(k) for k in ("logloss", "auc", "ece")}

        def _drift(key: str) -> float | None:
            b = base.get(key)
            return None if b is None else round(b_val[key] - b, 4)

        return {
            "sealed": b_val,
            "within_run_sealed": base,
            "drift_vs_within_run": {
                "logloss": _drift("logloss"),
                "auc": _drift("auc"),
                "ece": _drift("ece"),
            },
            "note": "diagnostic cross-check only — the bundle is NOT the "
                    "gate baseline; divergence here is cross-pull drift or "
                    "config change",
            "metadata": bundle.get("metadata"),
            "features": bundle.get("features"),
        }
    except Exception as e:  # noqa: BLE001 — diagnostic never crashes the run
        logger.warning("bundle cross-check predict failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Step 2 — window-parameterized walk-forward WITH the within-run incumbent
# ---------------------------------------------------------------------------
def run_walk_forward_gate(feats: pd.DataFrame,
                          features: list[str],
                          train_seasons: list[int],
                          sealed_season: int = SEALED_SEASON,
                          incumbent_bundle: dict | None = None,
                          load_default_bundle: bool = True,
                          incumbent_min_season: int | None =
                          INCUMBENT_MIN_SEASON) -> dict:
    """Prequential walk-forward with a configurable training window, a twin
    of ``nfl_moneyline.run_walk_forward`` that INCLUDES the fully within-run
    incumbent baseline (pooled fold-local + sealed pre-2025 re-fit) exactly
    as production does (2026-09-02 revision).

    The ONLY differences from production are that ``train_seasons`` /
    ``sealed_season`` are arguments instead of module constants (the
    ``generate_weekly_folds`` val window stays 2021-2024), the history/
    calibration record emitters are omitted (pure gate measurement), and the
    within-run incumbent re-trains on each fold's training slice restricted
    to ``incumbent_min_season`` (the production window) so a wider-window
    candidate is compared against the production-config model in the SAME
    run. The persisted bundle (injected or auto-loaded via the guarded
    production loader) is a DIAGNOSTIC cross-check only — never a verdict
    baseline; there is no advisory verdict mode.
    """
    from nfl_run_engine_legacy_windows import generate_weekly_folds
    from nfl_moneyline import (TARGET, _adaptive_blend, _elo_logistic_p,
                               _score_member_table, _valid_rows, adopt_decision,
                               auc, compute_adaptive_weights, compute_metrics,
                               ece, ensemble_predict,
                               load_ensemble, logloss, platt_fit,
                               platt_predict, train_ensemble)

    preq_all = feats[feats["season"].isin(train_seasons)].copy()
    sealed = feats[feats["season"] == sealed_season].copy()

    Xcol = [f for f in features if f in feats.columns]
    if not Xcol:
        raise ValueError("no model features present in the frame")

    preq = preq_all[_valid_rows(preq_all, Xcol)].copy()
    sld = sealed[_valid_rows(sealed, Xcol)].copy()
    folds = generate_weekly_folds(preq)

    # ---- BUNDLE (DIAGNOSTIC cross-check only — never the verdict) --------
    bundle = incumbent_bundle
    if bundle is None and load_default_bundle:
        bundle = load_ensemble()          # guarded production loader

    inc_raw_pool: list[np.ndarray] = []
    inc_y_pool: list[np.ndarray] = []   # the incumbent's OWN prior-fold OOF y
    inc_pool_cal: list[np.ndarray] = []

    order_actual, order_raw, order_elo, ws_list = [], [], [], []
    oof_members: dict[str, list[float]] = {}
    oof_members_cal: dict[str, list[float]] = {}
    cal_pool, raw_pool, elo_pool, y_pool = [], [], [], []

    for f in folds:
        tr, va = f["train"], f["val"]
        yva = va[TARGET].to_numpy(dtype=float)
        try:
            models, _mets = train_ensemble(tr, va, features=Xcol)
        except Exception as e:  # noqa: BLE001 — fold-level skip, as production
            logger.warning("fold %s ensemble failed: %s", f["week_start"], e)
            continue
        blend, member_probs, _wts = ensemble_predict(models, va, features=Xcol)
        elo_p = _elo_logistic_p(tr, va, Xcol)

        lr = None
        if y_pool:
            lr = platt_fit(np.concatenate(raw_pool),
                           np.concatenate(y_pool).astype(int))
            cal_p = platt_predict(blend, lr)
        else:
            cal_p = blend.copy()
        for name, p in member_probs.items():
            p_arr = np.asarray(p, dtype=float)
            oof_members.setdefault(name, []).extend(p_arr.tolist())
            pc = platt_predict(p_arr, lr) if lr is not None else p_arr
            oof_members_cal.setdefault(name, []).extend(pc.tolist())

        order_actual.append(yva)
        order_raw.append(blend)
        order_elo.append(elo_p)
        ws_list.append(f["week_start"])
        cal_pool.append(cal_p)
        raw_pool.append(blend)
        elo_pool.append(elo_p)
        y_pool.append(yva)

        # ---- Within-run POOLED incumbent --------------------------------
        # Production-config 12-pool re-fit on THIS fold's strictly-prior
        # training portion restricted to the production window — same seed,
        # same fold loop, same A/B walk state as the candidate. For a
        # production-window candidate this is byte-identical to the
        # candidate's own fold model (RANDOM_SEED determinism).
        inc_tr = tr
        if incumbent_min_season is not None and "season" in tr.columns:
            inc_tr = tr[tr["season"] >= incumbent_min_season]
        if len(inc_tr) == 0:
            # degenerate (production window empty on this fold) — fall back
            # to the candidate's own fold model, documented, never fabricated
            inc_cal = cal_p
        else:
            inc_models, _ = train_ensemble(inc_tr, va, features=Xcol)
            inc_blend, _, _ = ensemble_predict(inc_models, va, features=Xcol)
            lr_inc = None
            if inc_raw_pool:   # strictly-EARLIER folds only (candidate pattern)
                lr_inc = platt_fit(np.concatenate(inc_raw_pool),
                                   np.concatenate(inc_y_pool).astype(int))
                inc_cal = platt_predict(inc_blend, lr_inc)
            else:
                inc_cal = inc_blend.copy()
            inc_raw_pool.append(inc_blend)
            inc_y_pool.append(yva)
        inc_pool_cal.append(inc_cal)

    if not y_pool:
        raise RuntimeError("no folds produced ensemble predictions")

    y_po = np.concatenate(y_pool)
    raw_po = np.concatenate(raw_pool)
    cal_po = np.concatenate(cal_pool)
    elo_po = np.concatenate(elo_pool)

    const_p = preq[TARGET].mean()

    pooled = {
        "n": int(len(y_po)),
        "fold_count": len(folds),
        "constant_home_edge": {
            "proba": round(float(const_p), 4),
            "logloss": round(logloss(y_po, np.full_like(y_po, const_p)), 4),
            "auc": round(auc(y_po, np.full_like(y_po, const_p)), 4),
        },
        "elo_logistic": {
            "logloss": round(logloss(y_po, elo_po), 4),
            "auc": round(auc(y_po, elo_po), 4),
        },
        "model_raw": {
            "logloss": round(logloss(y_po, raw_po), 4),
            "auc": round(auc(y_po, raw_po), 4),
        },
        "model_platt": {
            "logloss": round(logloss(y_po, cal_po), 4),
            "auc": round(auc(y_po, cal_po), 4),
            "ece": round(ece(y_po, cal_po), 4),
        },
    }

    # Within-run incumbent POOLED arm (same OOF games as the candidate)
    incumbent_pooled = None
    if inc_pool_cal:
        inc_cal_all = np.concatenate(inc_pool_cal)
        if len(inc_cal_all) == len(y_po):
            incumbent_pooled = {
                "logloss": round(logloss(y_po, inc_cal_all), 4),
                "auc": round(auc(y_po, inc_cal_all), 4),
                "ece": round(ece(y_po, inc_cal_all), 4),
            }
    if incumbent_pooled is None:  # defensive — folds succeeded implies present
        incumbent_pooled = dict(pooled["model_platt"])

    adaptive = compute_adaptive_weights(oof_members, y_po)
    members_table = {}
    for name in sorted(set(oof_members)):
        raw_p = np.asarray(oof_members[name], dtype=float)
        entry = {"weight": float(adaptive.get(name, 0.0))}
        if len(raw_p) == len(y_po):
            m = compute_metrics(y_po, raw_p)
            entry.update({k: m[k] for k in ("logloss", "auc", "ece", "brier")})
        if len(oof_members_cal.get(name, [])) == len(y_po):
            mc = compute_metrics(y_po, np.asarray(oof_members_cal[name], dtype=float))
            entry.update({"logloss_calibrated": mc["logloss"],
                          "auc_calibrated": mc["auc"],
                          "ece_calibrated": mc["ece"]})
        members_table[name] = entry

    # ---- SEALED 2025 ----
    models_sealed, _ = train_ensemble(preq, None, features=Xcol)
    sealed_raw, sealed_members, _w = ensemble_predict(models_sealed, sld,
                                                      features=Xcol)
    sealed_elo = _elo_logistic_p(preq, sld, Xcol)

    oof_adaptive_blend = _adaptive_blend(oof_members, adaptive, len(y_po))
    platt_sealed = platt_fit(oof_adaptive_blend, y_po.astype(int))
    sealed_cal = platt_predict(sealed_raw, platt_sealed)
    const_sealed = preq[TARGET].mean()
    sealed_members_table = _score_member_table(sld[TARGET].to_numpy(),
                                               sealed_members)

    sealed = {
        "n": int(len(sld)),
        "constant_home_edge": {
            "proba": round(float(const_sealed), 4),
            "logloss": round(logloss(sld[TARGET], np.full(len(sld), const_sealed)), 4),
            "auc": round(auc(sld[TARGET], np.full(len(sld), const_sealed)), 4),
        },
        "elo_logistic": {
            "logloss": round(logloss(sld[TARGET], sealed_elo), 4),
            "auc": round(auc(sld[TARGET], sealed_elo), 4),
        },
        "model_raw": {
            "logloss": round(logloss(sld[TARGET], sealed_raw), 4),
            "auc": round(auc(sld[TARGET], sealed_raw), 4),
        },
        "model_platt": {
            "logloss": round(logloss(sld[TARGET], sealed_cal), 4),
            "auc": round(auc(sld[TARGET], sealed_cal), 4),
            "ece": round(ece(sld[TARGET], sealed_cal), 4),
        },
    }

    # ---- Within-run incumbent SEALED arm --------------------------------
    # Production-config 12-pool re-fit ONCE on all pre-2025 rows of the
    # current pull restricted to the production window (strictly prior to
    # sealed), scored on the same 2025 rows and calibrated with the SAME
    # sealed Platt map as the candidate (shared calibration machinery — any
    # difference is model-level, not Platt-fit noise).
    preq_inc = preq
    if incumbent_min_season is not None and "season" in preq.columns:
        preq_inc = preq[preq["season"] >= incumbent_min_season]
    if len(preq_inc) == 0:
        preq_inc = preq  # degenerate — documented, never fabricated
    models_inc_sealed, _ = train_ensemble(preq_inc, None, features=Xcol)
    sealed_inc_raw, _, _ = ensemble_predict(models_inc_sealed, sld,
                                            features=Xcol)
    sealed_inc_cal = platt_predict(sealed_inc_raw, platt_sealed)
    incumbent_sealed = {
        "logloss": round(logloss(sld[TARGET], sealed_inc_cal), 4),
        "auc": round(auc(sld[TARGET], sealed_inc_cal), 4),
        "ece": round(ece(sld[TARGET], sealed_inc_cal), 4),
    }
    pooled["incumbent"] = incumbent_pooled
    sealed["incumbent"] = incumbent_sealed

    # ---- BUNDLE DIAGNOSTIC CROSS-CHECK (demoted — informational only) ----
    bundle_crosscheck = _bundle_crosscheck(sld, bundle, incumbent_sealed)

    # The within-run incumbent ALWAYS exists — the verdict is the six
    # tolerance legs vs it (the bundle is diagnostic-only, never gating).
    verdict = M.adopt_decision(pooled, sealed, incumbent={
        "pooled_model_platt": incumbent_pooled,
        "sealed_model_platt": incumbent_sealed})

    inc_geometry = ("fold-local pooled re-fit (production-window restriction "
                    "%s) + within-run pre-2025 sealed re-fit; the persisted "
                    "bundle is a diagnostic cross-check only"
                    % (incumbent_min_season
                       if incumbent_min_season is not None else "none"))

    return {
        "fold_geometry": {
            "train_seasons": train_seasons,
            "val_seasons": VAL_SEASONS,
            "sealed_season": sealed_season,
            "fold_count": len(folds),
            "pooled_oof_games": int(len(y_po)),
            "sealed_games": int(len(sld)),
            "preq_weeks": [str(f["week_start"].date()) for f in folds],
        },
        "pooled_preq_2021_2024": pooled,
        "sealed_2025": sealed,
        "adaptive_weights": adaptive,
        "members": members_table,
        "members_sealed": sealed_members_table,
        "incumbent_within_run": {
            "pooled": incumbent_pooled,
            "sealed": incumbent_sealed,
            "geometry": inc_geometry,
        },
        "bundle_crosscheck": bundle_crosscheck,
        "verdict": verdict,
        "_deployed": {"features": Xcol},
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _platt(rec: dict) -> dict:
    return {k: rec.get(k) for k in ("logloss", "auc", "ece")}


def _arm_cell(rec: dict, key: str) -> dict:
    return _platt(rec[key].get("model_platt") or {})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=None,
                    help="pre-built features CSV (skips the nflreadpy pull; "
                         "arms slice the frame at each boundary)")
    ap.add_argument("--arms", nargs="*", choices=sorted(BOUNDARIES),
                    default=None, help="candidates to run (default: both)")
    ap.add_argument("--no-incumbent", action="store_true",
                    help="skip the bundle diagnostic cross-check (fresh-clone "
                         "simulation); the within-run verdict is unaffected")
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON record")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    bundle = None
    if args.no_incumbent:
        print("[warn] --no-incumbent: bundle cross-check skipped (fresh-clone "
              "simulation) — the within-run incumbent verdict is unaffected")
    else:
        bundle = M.load_ensemble()
        if bundle is None:
            print("[warn] no valid served bundle on this checkout — bundle "
                  "cross-check unavailable; within-run verdict unaffected")
        else:
            meta = (bundle.get("metadata") or {})
            print(f"[bundle] {M.ENSEMBLE_FILE}: "
                  f"{len(bundle.get('features') or [])} features, created "
                  f"{meta.get('created_utc', '?')} — DIAGNOSTIC cross-check "
                  "only (never a verdict baseline)")

    todo = [n for n in BOUNDARIES if args.arms is None or n in args.arms]
    feats_by_arm: dict[str, pd.DataFrame] = {}
    for name in todo:
        if args.features:
            base = pd.read_csv(args.features)
            base["gameday"] = pd.to_datetime(base["gameday"])
            if "home_win" not in base.columns:
                base["home_win"] = (base["home_score"]
                                    > base["away_score"]).astype(int)
            feats_by_arm[name] = base[base["season"] >= BOUNDARIES[name]].copy()
        else:
            b = BOUNDARIES[name]
            print(f"\n=== building feature frame for {name} "
                  f"(warmup {b - 1}, core {b}-{SEALED_SEASON}) ===")
            feats_by_arm[name] = load_arm_features(b)

    results = {}
    for name in todo:
        feats = feats_by_arm[name]
        cols = arm_features(feats)
        train = list(range(BOUNDARIES[name], TRAIN_END + 1))
        print(f"\n=== running within-run incumbent gate arm {name} "
              f"(train {train[0]}-{train[-1]}, sealed {SEALED_SEASON}) ===")
        print(f"  decided rows: {len(feats)} | model cols: {len(cols)} | "
              f"frame sha: {_frame_sha256(feats)}")
        results[name] = run_walk_forward_gate(feats, cols, train,
                                              incumbent_bundle=bundle)

    print(f"\n=== NFL window gate — fully within-run incumbent "
          f"(TOL_LL {TOL_LL}, TOL_AUC {TOL_AUC}, ECE_TOL {ECE_TOL}; "
          f"incumbent window >= {INCUMBENT_MIN_SEASON}) ===")
    print("arm        sealed_ll  sealed_auc  sealed_ece  pooled_ll  "
          "pooled_auc  pooled_ece")
    for name in todo:
        rec = results[name]
        s = rec["sealed_2025"]
        p = rec["pooled_preq_2021_2024"]
        lab = name
        print(f"{lab:6s} {s['model_platt']['logloss']:10.4f} "
              f"{s['model_platt']['auc']:10.4f} {s['model_platt']['ece']:10.4f}"
              f"  {p['model_platt']['logloss']:9.4f} "
              f"{p['model_platt']['auc']:9.4f} "
              f"{p['model_platt']['ece']:9.4f}")
        wr = rec["incumbent_within_run"]["sealed"]
        wp = rec["incumbent_within_run"]["pooled"]
        print(f"   with-run  {wr['logloss']:10.4f} {wr['auc']:10.4f} "
              f"{wr['ece']:10.4f}  {wp['logloss']:9.4f} "
              f"{wp['auc']:9.4f} {wp['ece']:9.4f}")
        bc = rec.get("bundle_crosscheck")
        if bc is not None:
            bs = bc["sealed"]
            d = bc["drift_vs_within_run"]
            print(f"   bundle    {bs['logloss']:10.4f} {bs['auc']:10.4f} "
                  f"{bs['ece']:10.4f}  (drift ll {d['logloss']} auc "
                  f"{d['auc']} ece {d['ece']} — diagnostic only)")

    for name in todo:
        v = results[name]["verdict"]
        verdict_txt = "ADOPT" if v["adopt"] else "DON'T ADOPT"
        reason_txt = "; ".join(v["reasons"]) if v["reasons"] else "no reasons"
        print(f"\nVERDICT ({name}): {verdict_txt} | {reason_txt}")
        print(f"  ece_mode {v['ece_mode']} | ll_ok_pooled {v['ll_ok_pooled']} | "
              f"auc_ok_pooled {v['auc_ok_pooled']} | ece_ok_pooled "
              f"{v['ece_ok_pooled']}")
        print(f"  ll_ok_sealed {v['ll_ok_sealed']} | auc_ok_sealed "
              f"{v['auc_ok_sealed']} | ece_ok_sealed {v['ece_ok_sealed']} | "
              f"tol {v['tol']}")

    if args.no_record:
        return 0

    first_fh = _frame_sha256(feats_by_arm[todo[0]])
    inc_meta = (bundle.get("metadata") or {}) if bundle is not None else None
    record = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "method": ("fully within-run incumbent gate (nfl_moneyline."
                   "adopt_decision, third revision): baseline = production-"
                   "config 12-pool trained WITHIN this run — pooled as a "
                   "fold-local re-fit on each fold's strictly-prior training "
                   "slice (restricted to production window >= %d) in the "
                   "candidate's own fold loop, sealed as one re-fit on all "
                   "pre-2025 rows of the current pull (restricted to >= %d); "
                   "adopt = ll_ok and auc_ok and ece_ok on BOTH pooled and "
                   "sealed (ll_ok = cand <= inc + TOL_LL, auc_ok = cand >= "
                   "inc - TOL_AUC, ece_ok = cand <= inc + ECE_TOL) — each "
                   "blocking, nothing else, NO advisory verdict mode; the "
                   "persisted bundle is a diagnostic cross-check only"
                   % (INCUMBENT_MIN_SEASON, INCUMBENT_MIN_SEASON)),
        "frame_sha256": first_fh,  # first arm's frame — per-arm shas live in arms[]
        "tol": {"ll": TOL_LL, "auc": TOL_AUC, "ece": ECE_TOL},
        "candidates": BOUNDARIES,
        "incumbent": {
            "baseline": ("within-run (production window >= %d, same pull, "
                         "same folds, same seed)" % INCUMBENT_MIN_SEASON),
            "bundle": M.ENSEMBLE_FILE,
            "bundle_role": "diagnostic cross-check only — never a verdict "
                           "baseline",
            "bundle_features": (bundle.get("features") if bundle is not None
                                else None),
            "bundle_metadata": inc_meta,
        },
        "arms": {},
        "verdicts": {},
    }
    for name in todo:
        rec = results[name]
        record["arms"][name] = {
            "boundary": BOUNDARIES[name],
            "features": rec["_deployed"]["features"],
            "decided_rows": int(len(feats_by_arm[name])),
            "frame_sha256": _frame_sha256(feats_by_arm[name]),
            "fold_geometry": rec["fold_geometry"],
            "sealed_2025": rec["sealed_2025"],
            "pooled_preq_2021_2024": rec["pooled_preq_2021_2024"],
            "incumbent_within_run": rec["incumbent_within_run"],
            "bundle_crosscheck": rec.get("bundle_crosscheck"),
            "members": {m: dict(v) for m, v in (rec.get("members") or {}).items()},
            "members_sealed": {m: dict(v) for m, v in
                               (rec.get("members_sealed") or {}).items()},
        }
        v = rec["verdict"]
        record["verdicts"][name] = {
            "adopt": v["adopt"],
            "ece_mode": v["ece_mode"],
            "ll_ok_pooled": v["ll_ok_pooled"],
            "auc_ok_pooled": v["auc_ok_pooled"],
            "ece_ok_pooled": v["ece_ok_pooled"],
            "ll_ok_sealed": v["ll_ok_sealed"],
            "auc_ok_sealed": v["auc_ok_sealed"],
            "ece_ok_sealed": v["ece_ok_sealed"],
            "tol": v["tol"],
            # informational table rows only — NOT part of the verdict
            "sealed_beats_elo": v["sealed_beats_elo"],
            "sealed_beats_constant": v["sealed_beats_constant"],
            "reasons": v["reasons"],
        }
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"nfl_window_gate_{first_fh}.json"
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())