"""NFL margin engine C0-vs-C1 ablation (Phase-1 spec, record-only).

C0: 12-pool baseline (FEATURE_COLUMNS minus the is_home anchor — the
    moneyline's served view)
C1: 12-pool + pt_margin_diff (OOF margin predictions from nfl_margin_engine)

Uses the moneyline's own walk-forward geometry (generate_weekly_folds,
train_ensemble, ensemble_predict) for the six-leg gate. pt_margin_diff is
computed OUT-OF-FOLD to prevent leakage — each fold's margin comes from a
model trained strictly before that fold's validation week, and the moneyline
ensemble that consumes it is fit per-fold on the SAME pre-fold data only
(READ-ONLY feature producer; never retrained on moneyline val windows).

Honest bar (baked in): MLB's increment was small at n≈6,500; at NFL
n≈1,107 a similar relative effect likely sits under TOL/3 → probable
data-lever, not worth-having. Not forced: if it fails only on the pooled
magnitude bar but directions agree, the record says RE_TEST_CANDIDATE
(re-test when scored n grows) — never null.

Geometry: C0 = 12-pool; C1 = 12-pool + pt_margin_diff. Same 88 folds /
pooled 2021–24 / sealed 2025, same six-leg gate (tolerance_verdict, the ONE
shared rule), worth-having per TOL/3 + pooled/sealed agreement.

Usage:
    cd nfl-backend && python3 backend/run_nfl_margin_ablation.py [--features <csv>]
Artifact: data_delivery/nfl_margin_engine_<frame_sha>.json (uncommitted;
review before any commit — record-only, no wiring).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nfl_features import FEATURE_COLUMNS, build_features
from nfl_margin_engine import MARGIN_COL, MARGIN_FEATURES, oof_margins, refit_margins
from nfl_moneyline import (
    TARGET,
    TRAIN_SEASONS,
    SEALED_SEASON,
    generate_weekly_folds,
    train_ensemble,
    ensemble_predict,
    platt_fit,
    platt_predict,
    _valid_rows,
    compute_metrics,
    ECE_TOL,
    TOL_LL,
    TOL_AUC,
    tolerance_verdict,
)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY = ROOT_DIR / "data_delivery"
DECIDED_FRAME = DATA_DELIVERY / "nfl_game_level_features.csv"


def _frame_sha256(df: pd.DataFrame) -> str:
    """Content hash of the feature frame (row-sorted) — tier-1 convention."""
    h = hashlib.sha256()
    sorted_df = df.sort_values("game_id").reset_index(drop=True)
    h.update(sorted_df.to_csv(index=False).encode("utf-8"))
    return h.hexdigest()[:12]


def load_features(features_csv: str | None = None) -> pd.DataFrame:
    """Load the decided feature frame with all 12-pool columns.

    Prefers an explicit ``--features`` CSV; else the decided frame built via
    nflreadpy schedule + PBP (cached season parquets under /tmp when
    present), cached to /tmp keyed by frame sha so re-runs skip the pull.
    """
    if features_csv and Path(features_csv).exists():
        df = pd.read_csv(features_csv)
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    else:
        df = pd.read_csv(DECIDED_FRAME)
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")

    missing = [f for f in MARGIN_FEATURES if f not in df.columns]
    if missing:
        print(f"  Missing columns: {missing} — building features from nflreadpy...")
        cache = Path(f"/tmp/nfl_features_{_frame_sha256(df)}.parquet")
        if cache.exists():
            df = pd.read_parquet(cache)
            df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
            missing = [f for f in MARGIN_FEATURES if f not in df.columns]
        if missing:
            import nflreadpy
            seasons = list(range(2018, 2026))
            schedule = nflreadpy.load_schedules(seasons).to_pandas()
            pbp_parts = []
            for yr in range(2019, 2026):
                pf = Path(f"/tmp/nfl_pbp_{yr}.parquet")
                if pf.exists():
                    pbp_parts.append(pd.read_parquet(pf))
                else:
                    part = nflreadpy.load_pbp([yr])
                    if hasattr(part, "to_pandas"):
                        part = part.to_pandas()
                    part.to_parquet(pf)
                    pbp_parts.append(part)
            pbp = pd.concat(pbp_parts, ignore_index=True)
            decided = df.copy()
            decided["gameday"] = pd.to_datetime(decided["gameday"])
            df = build_features(decided, schedule=schedule, pbp=pbp)
            df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache)
            except Exception as e:  # noqa: BLE001
                print(f"  [WARN] feature cache write failed: {e}")
            still_missing = [f for f in MARGIN_FEATURES if f not in df.columns]
            if still_missing:
                raise RuntimeError(f"Still missing after build: {still_missing}")

    if TARGET not in df.columns and "home_score" in df.columns \
            and "away_score" in df.columns:
        df[TARGET] = (df["home_score"] > df["away_score"]).astype(int)
    return df


def compute_oof_margins(feats: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute OOF pt_margin_diff on the moneyline's own folds (2021–24)."""
    preq = feats[feats["season"].isin(TRAIN_SEASONS)].copy()
    preq_valid = preq[_valid_rows(preq, MARGIN_FEATURES)].copy()
    folds = generate_weekly_folds(preq_valid)

    margins, rounds, n_uncov = oof_margins(folds, MARGIN_FEATURES, feats)
    logger.info("margin engine: OOF %d games, %d uncovered, rounds=%s",
                len(margins), n_uncov, rounds)
    return margins, rounds


def attach_margins(feats: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    """Attach OOF pt_margin_diff to the feature frame (left merge on game_id).

    Mirror of MLB's _attach_oof_run_margins: a pure keyed left join — rows
    with no OOF margin (pre-2021 warmup games, never in any fold's val
    window) stay NaN for tree-native routing / train-median imputation;
    never zero-filled.
    """
    out = feats.copy()
    if MARGIN_COL in out.columns:
        out = out.drop(columns=[MARGIN_COL])
    out = out.merge(margins[["game_id", MARGIN_COL]], on="game_id", how="left")
    coverage = out[MARGIN_COL].notna().mean() * 100
    logger.info("margin attach: %.1f%% coverage (%d/%d games)",
                coverage, out[MARGIN_COL].notna().sum(), len(out))
    return out


def walk_arm(feats: pd.DataFrame, features: list[str], label: str,
             margin_rounds: dict | None = None) -> dict:
    """Run one walk-forward arm using the moneyline machinery.

    Returns pooled + sealed leg metrics. When the margin feature is in the
    arm (C1), the sealed frame's margins come from a fit-only refit on all
    pre-2025 data at the median fold round count (never trained on 2025).
    """
    preq_all = feats[feats["season"].isin(TRAIN_SEASONS)].copy()
    sealed = feats[feats["season"] == SEALED_SEASON].copy()

    Xcol = [f for f in features if f in feats.columns]
    # Row filter = BASE features + target only. For C1 the margin column may
    # be NaN on pre-2021 warmup rows — those rows MUST stay in the folds so
    # the ensemble trains on the same games as C0; the moneyline's own
    # imputation (trees route NaN natively; logistic/MLP/RF get train-fold
    # medians) handles the missing margin, per the Phase-1 imputation
    # discipline. Never drop rows just because a feature is NaN.
    base_valid = [f for f in Xcol if f != MARGIN_COL]
    preq = preq_all[_valid_rows(preq_all, base_valid)].copy()
    sld = sealed[_valid_rows(sealed, base_valid)].copy()

    # Sealed margin refill (C1 only): fit on 2019–2024, predict 2025.
    if MARGIN_COL in Xcol and sld[MARGIN_COL].isna().any() and margin_rounds is not None:
        n_rounds = int(margin_rounds.get("home", 30))
        margin_refit = refit_margins(preq, sld, n_rounds, base_valid)
        sld = sld.drop(columns=[MARGIN_COL], errors="ignore")
        sld = sld.merge(margin_refit[["game_id", MARGIN_COL]],
                        on="game_id", how="left")
        logger.info("%s sealed margin refill: %d/%d rows",
                    label, int(sld[MARGIN_COL].notna().sum()), len(sld))

    folds = generate_weekly_folds(preq)

    # ---- Fold loop (pooled OOF 2021–24) ----
    raw_pool, y_pool, cal_pool = [], [], []

    for f in folds:
        tr, va = f["train"], f["val"]
        yva = va[TARGET].to_numpy(dtype=float)
        try:
            models, _mets = train_ensemble(tr, va, features=Xcol)
        except Exception as e:  # noqa: BLE001
            logger.warning("fold %s failed: %s", f["week_start"], e)
            continue
        blend, _member_probs, _wts = ensemble_predict(models, va, features=Xcol)

        # Prequential Platt twin (fit on prior folds' OOF only).
        if y_pool:
            lr = platt_fit(np.concatenate(raw_pool),
                           np.concatenate(y_pool).astype(int))
            cal_pool.append(platt_predict(blend, lr))
        else:
            cal_pool.append(blend.copy())

        raw_pool.append(blend)
        y_pool.append(yva)

    if not raw_pool:
        return {"error": f"no folds scored for {label}"}

    y_all = np.concatenate(y_pool)
    raw_all = np.concatenate(raw_pool)
    cal_all = np.concatenate(cal_pool)

    # ---- Sealed evaluation (fit 2019–24, predict 2025) ----
    sealed_result = {}
    if len(sld) > 0:
        try:
            models, _ = train_ensemble(preq, sld, features=Xcol)
            sealed_raw, _, _ = ensemble_predict(models, sld, features=Xcol)
            lr = platt_fit(raw_all, y_all.astype(int))
            sealed_cal = platt_predict(sealed_raw, lr)
            sealed_result = _legs(sld[TARGET].to_numpy(dtype=float),
                                  sealed_raw, sealed_cal, "_sealed")
        except Exception as e:  # noqa: BLE001
            logger.warning("%s sealed eval failed: %s", label, e)
            import traceback
            traceback.print_exc()

    pooled = _legs(y_all, raw_all, cal_all, "_pooled")
    return {"pooled": pooled, "sealed": sealed_result}


def _legs(y: np.ndarray, raw: np.ndarray, cal: np.ndarray,
          suffix: str) -> dict:
    """ll / auc / ece via nfl_moneyline.compute_metrics — the EXACT metric
    convention the production gate uses (ECE_BINS=10), so the record's
    numbers are directly comparable to the moneyline artifacts."""
    m_raw = compute_metrics(y, raw)
    m_cal = compute_metrics(y, cal)
    return {
        f"ll{suffix}": m_raw["logloss"],
        f"ll_cal{suffix}": m_cal["logloss"],
        f"auc{suffix}": m_raw["auc"],
        f"auc_cal{suffix}": m_cal["auc"],
        f"ece_raw{suffix}": m_raw["ece"],
        f"ece_cal{suffix}": m_cal["ece"],
        f"n{suffix}": int(len(y)),
    }


def _six_leg_gate(c0: dict, c1: dict) -> dict:
    """The ONE shared gate: tolerance_verdict (TOL_LL/TOL_AUC/ECE_TOL), pooled
    AND sealed, each condition blocking. Baseline = the arm's own C0."""
    def _view(arm: dict, view: str) -> dict:
        d = arm.get(view, {})
        return {"logloss": d.get(f"ll_cal_{view}"),
                "auc": d.get(f"auc_cal_{view}"),
                "ece": d.get(f"ece_cal_{view}")}

    gate = tolerance_verdict(pooled_cand=_view(c1, "pooled"),
                             pooled_base=_view(c0, "pooled"),
                             sealed_cand=_view(c1, "sealed"),
                             sealed_base=_view(c0, "sealed"),
                             baseline_name="C0")

    _metric_key = {"logloss": "ll_cal", "auc": "auc_cal", "ece": "ece_cal"}
    _tol = {"logloss": TOL_LL, "auc": TOL_AUC, "ece": ECE_TOL}
    legs = {}
    for view in ("pooled", "sealed"):
        for metric, op_key in (("logloss", "ll"), ("auc", "auc"), ("ece", "ece")):
            key = f"{op_key}_ok_{view}"
            c = c1.get(view, {}).get(f"{_metric_key[metric]}_{view}")
            b = c0.get(view, {}).get(f"{_metric_key[metric]}_{view}")
            legs[key] = {
                "cand": c, "base": b,
                "delta": round(c - b, 4) if c is not None and b is not None else None,
                "tol": _tol[metric], "pass": bool(gate[key]),
            }
    return {"verdict": "ADOPT" if gate["adopt"] else "DON'T_ADOPT",
            "adopt": gate["adopt"], "legs": legs,
            "reasons": gate["reasons"], "delta": gate["delta"]}


def _worth_having(c0: dict, c1: dict) -> dict:
    """Worth-having: all six deltas beyond TOL/3 in the right direction
    (pooled + sealed × ll/auc/ece), direction-agreed.

    Verdicts:
      WORTH_HAVING   — every delta beyond TOL/3 and in the right direction.
      RE_TEST_CANDIDATE — all six deltas in the right direction but NOT all
        beyond TOL/3 (spec: re-test when scored n grows — never null).
      NOT_WORTH_HAVING — otherwise (direction disagreement or regression).
    """
    checks = {}
    for view in ("pooled", "sealed"):
        for metric, tol, direction in [
            (f"ll_cal_{view}", TOL_LL / 3, -1),
            (f"auc_cal_{view}", TOL_AUC / 3, 1),
            (f"ece_cal_{view}", ECE_TOL / 3, -1),
        ]:
            c = c1.get(view, {}).get(metric)
            b = c0.get(view, {}).get(metric)
            delta = None
            if c is not None and b is not None:
                delta = c - b
            beyond = bool(delta is not None and abs(delta) > tol)
            right_dir = bool(delta is not None and delta != 0
                             and (delta * direction) > 0)
            checks[f"{view}_{metric}"] = {
                "delta": round(delta, 4) if delta is not None else None,
                "tol_3": round(tol, 4), "beyond_tol": beyond,
                "right_direction": right_dir,
            }

    all_right = all(c["right_direction"] for c in checks.values())
    all_beyond = all(c["beyond_tol"] for c in checks.values())
    if all_right and all_beyond:
        verdict = "WORTH_HAVING"
    elif all_right:
        verdict = "RE_TEST_CANDIDATE"
    else:
        verdict = "NOT_WORTH_HAVING"
    return {"checks": checks, "verdict": verdict}


def _margin_mae_crps(feats_m: pd.DataFrame, label: str) -> dict:
    """MAE/CRPS of the margin model on covered rows (point forecast ⇒
    CRPS == MAE for a degenerate prediction)."""
    covered = feats_m[feats_m[MARGIN_COL].notna()].copy()
    actual = covered["home_score"].astype(float) - covered["away_score"].astype(float)
    err = (covered[MARGIN_COL] - actual).abs()
    return {
        "label": label,
        "n": int(len(covered)),
        "mae": round(float(err.mean()), 3),
        "crps": round(float(err.mean()), 3),  # degenerate point forecast
        "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=None,
                    help="pre-built features CSV (skips the nflreadpy pull)")
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON record")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    print("=" * 70)
    print("  NFL Margin Engine C0-vs-C1 Ablation (Phase-1, record-only)")
    print("=" * 70)

    t0 = time.time()
    feats = load_features(args.features)
    print(f"Loaded: {len(feats)} games, {feats['season'].min()}-{feats['season'].max()} "
          f"| frame_sha256={_frame_sha256(feats)}")

    # Step 1: OOF margins on the moneyline's own folds
    print("\n[1] Computing OOF pt_margin_diff...")
    margins, margin_rounds = compute_oof_margins(feats)

    # Step 2: Attach margins
    print("\n[2] Attaching margins...")
    feats_m = attach_margins(feats, margins)

    # Step 3/4: C0 then C1 walks (same 88-fold geometry, sealed 2025)
    print("\n[3] Walking C0 (12-pool baseline)...")
    c0_result = walk_arm(feats, MARGIN_FEATURES, "C0")
    print("\n[4] Walking C1 (12-pool + pt_margin_diff)...")
    c1_result = walk_arm(feats_m, MARGIN_FEATURES + [MARGIN_COL], "C1",
                         margin_rounds=margin_rounds)

    # Step 5: six-leg gate
    print("\n[5] Six-leg gate...")
    gate = _six_leg_gate(c0_result, c1_result)

    # Step 6: worth-having check (TOL/3 + pooled/sealed agreement)
    print("\n[6] Worth-having check...")
    worth = _worth_having(c0_result, c1_result)

    # Margin-model self-eval (the deliverable's MAE/CRPS + coverage)
    oof_score = _margin_mae_crps(feats_m, "oof 2021-24")
    sealed_score = _margin_mae_crps(
        _sealed_margins_frame(feats, margins, margin_rounds), "sealed 2025")
    pre_sealed = int((feats["season"] < SEALED_SEASON).sum())
    n_oof = int(margins["game_id"].nunique())
    coverage = {
        "n_total": int(len(feats)),
        "n_pre_sealed_2019_24": pre_sealed,
        "n_sealed_2025": int(len(feats) - pre_sealed),
        "n_oof_margins": n_oof,
        "pct_covered_pre_sealed": round(n_oof / max(pre_sealed, 1) * 100, 1),
        "n_uncovered_pre_sealed": int(pre_sealed - n_oof),
        "n_sealed_refill": int(sealed_score["n"]),
        "pct_covered_sealed": round(sealed_score["n"] / max(len(feats) - pre_sealed, 1) * 100, 1),
        "imputation": ("uncovered = 2019-20 warmup (never in a fold's val window) "
                        "+ playoff weeks with <5 val games → NaN → tree-native "
                        "routing / train-median for logistic+MLP; sealed 2025 = "
                        "fit-only refill at median rounds (100% of valid rows)"),
    }

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    print("\n=== RESULTS ===")
    for arm_name, arm in [("C0", c0_result), ("C1", c1_result)]:
        p, s = arm.get("pooled", {}), arm.get("sealed", {})
        print(f"\n{arm_name}: pooled n={p.get('n_pooled')} "
              f"ll_cal={p.get('ll_cal_pooled')} auc_cal={p.get('auc_cal_pooled')} "
              f"ece_cal={p.get('ece_cal_pooled')} | sealed n={s.get('n_sealed')} "
              f"ll_cal={s.get('ll_cal_sealed')} auc_cal={s.get('auc_cal_sealed')} "
              f"ece_cal={s.get('ece_cal_sealed')}")
    print(f"\nMargin model: oof MAE/CRPS={oof_score['mae']} (n={oof_score['n']}) | "
          f"sealed MAE/CRPS={sealed_score['mae']} (n={sealed_score['n']})")
    print(f"\nGate verdict: {gate['verdict']}")
    print(f"Worth-having: {worth['verdict']}")

    frame_sha = hashlib.sha256(DECIDED_FRAME.read_bytes()).hexdigest()[:12]
    record = {
        "record": "nfl_margin_engine_ablation",
        "frame_sha": frame_sha,
        "frame_sha256": _frame_sha256(feats),
        "written_utc": pd.Timestamp.now("UTC").isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "geometry": {
            "seasons": sorted(feats["season"].unique().tolist()),
            "train_seasons": TRAIN_SEASONS,
            "sealed_season": SEALED_SEASON,
            "n_folds": len(generate_weekly_folds(
                feats[feats["season"].isin(TRAIN_SEASONS)])),
            "c0": MARGIN_FEATURES,
            "c1": MARGIN_FEATURES + [MARGIN_COL],
        },
        "coverage": coverage,
        "margin_model": {
            "oof": oof_score,
            "sealed": sealed_score,
            "median_rounds": margin_rounds,
            "features": MARGIN_FEATURES,
        },
        "c0": c0_result,
        "c1": c1_result,
        "gate": gate,
        "worth_having": worth,
        "feature_columns_untouched": True,
        "adopted": False,  # record-only; no wiring
        "judgment_calls": {
            "phase1_view_12pool": "12-pool as-is (cheapest clean first answer; "
                                  "per-side offense/defense EWM splits are Phase 2)",
            "single_margin_regression": "not per-side Poisson — NFL margins are "
                                        "near-normal; per-side matters only for "
                                        "spread/total pricing later",
            "imputation": "uncovered rows (pre-2021 warmup) → NaN → tree-native "
                          "routing / train-median for logistic+MLP; sealed = fit "
                          "2019-24 at median rounds; slate = fit all 2019-25",
        },
    }

    if not args.no_record:
        record_path = DATA_DELIVERY / f"nfl_margin_engine_{frame_sha}.json"
        record_path.write_text(json.dumps(record, indent=2, default=str))
        print(f"\nRecord: {record_path.name}")
    else:
        print("\n[--no-record] record skipped")
    return 0


def _sealed_margins_frame(feats: pd.DataFrame, margins: pd.DataFrame,
                          rounds: dict) -> pd.DataFrame:
    """Sealed margins (fit 2019–24 at median rounds, predict 2025), merged
    back onto the 2025 rows for the MAE/CRPS evaluation."""
    preq = feats[feats["season"].isin(TRAIN_SEASONS)].copy()
    sld = feats[feats["season"] == SEALED_SEASON].copy()
    preq = preq[_valid_rows(preq, MARGIN_FEATURES)].copy()
    sld = sld[_valid_rows(sld, MARGIN_FEATURES)].copy()
    n_rounds = int(rounds.get("home", 30))
    refit = refit_margins(preq, sld, n_rounds, MARGIN_FEATURES)
    out = sld.merge(refit[["game_id", MARGIN_COL]], on="game_id", how="left")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    raise SystemExit(main())