"""NFL window-extension ablation — does MORE decided history help the
moneyline ensemble on the same sealed-2025 hold-out?

Background: the feature program concluded the market-free 12-pool is the
ceiling at the current 1,960-game decided frame (2019-2025) — the constraint
is sample size, not architecture. Extending the decided window is the
identified growth lever; 2016 is the likely sweet spot because it roughly
doubles the decided frame AND unlocks Tier-5 (player-level participation
data starts 2016). Market data is NEVER an input (policy).

Arms (same 12-feature pool, same fold geometry as the production gate):
  W2019 — current window: warmup 2018, core 2019-2025 (production baseline)
  W2016 — warmup 2015, core 2016-2025
  W2014 — warmup 2013, core 2014-2025

Fold geometry is IDENTICAL across arms: pooled OOF = the same prequential
2021-2024 weeks, sealed = 2025. Only the training breadth before each fold
differs — the fairest possible comparison of window length.

MANDATED ORDER: the survey runs FIRST (or via --survey-only) and reports
per-boundary nflreadpy coverage (schedule decided rate, pbp games, epa /
qb_epa / pace non-null, venue fields). A boundary only becomes an arm if
its coverage clears the floors (decided >= 0.98, epa >= 0.90, qb_epa >= 0.85,
pace >= 0.90, stadium/roof/gametime >= 0.95) — the harness never runs an arm
the survey rejects.

Adoption rule — the SAME gate as Tier-1/2/3/4 (run_tier1_ablation.adopt_verdict),
applied to W2019 (baseline) vs each wider arm: the wider window must beat
W2019 on SEALED logloss AND AUC without degrading ECE-cal; pooled logs
corroborate. If W2016 beats W2019 -> recommend extending (also unlocks
Tier-5). If not -> era non-stationarity is real; stay at 2019. If W2014
loses to W2016 -> the era floor is 2016.

NO production config change ships from this harness: FEATURE_COLUMNS /
DEFAULT_SEASONS are untouched regardless of outcome (adoption is a separate
decision).

Usage (network + nflreadpy needed for the raw pull):
    python3 run_nfl_window_ablation.py              # survey + arms + record
    python3 run_nfl_window_ablation.py --survey-only
    python3 run_nfl_window_ablation.py --arms W2019 W2016
    python3 run_nfl_window_ablation.py --features <features.csv>
    python3 run_nfl_window_ablation.py --no-record
Artifact: data_delivery/nfl_window_ablation_<sha>.json (examined before any
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
from run_tier1_ablation import (MEMBER_NAMES, _frame_sha256, _member_metrics,
                                adopt_verdict)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"

# Boundary season per arm (the CORE starts here; the warmup year is B-1).
BOUNDARIES: dict[str, int] = {"W2019": 2019, "W2016": 2016, "W2014": 2014}
SEALED_SEASON = 2025          # sealed hold-out — constant across arms
TRAIN_END = SEALED_SEASON - 1  # train window is [B .. 2024]
VAL_SEASONS = [2021, 2022, 2023, 2024]  # pooled-OOF weeks — constant across arms

# Survey coverage floors (empirically grounded on the 2019-2025 window,
# which clears all of these by wide margins).
FLOORS = {
    "decided_rate": 0.98, "pbp_games_rate": 0.95, "epa_pct": 0.90,
    "qb_epa_pct": 0.85, "pace_pct": 0.90, "stadium_pct": 0.95,
    "roof_pct": 0.95, "gametime_pct": 0.95,
}

# The 12 features mapped to the survey metric that gates their buildability.
FEATURE_SOURCE = {
    "elo_diff": "decided_rate", "win_pct_diff": "decided_rate",
    "rest_days_diff": "decided_rate", "rest_short_diff": "decided_rate",
    "ewm_net_pts_diff": "decided_rate", "ewm_ypp_diff": "decided_rate",
    "pace_plays_min_diff": "pace_pct", "is_dome_home": "roof_pct",
    "div_game": "decided_rate", "travel_miles_diff": "stadium_pct",
    "altitude_home": "stadium_pct", "prime_time": "gametime_pct",
}


# ---------------------------------------------------------------------------
# Raw pull (mirrors nfl_features._load_raw, plus play_id for the game frame)
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


# ---------------------------------------------------------------------------
# Step 1 — survey (mandated before any arm is committed to)
# ---------------------------------------------------------------------------
def survey_season(season: int,
                  sched: pd.DataFrame,
                  pbp: pd.DataFrame) -> dict:
    """One per-season coverage row for the survey table."""
    sc = sched[sched["season"] == season]
    pb = pbp[pbp["season"] == season] if "season" in pbp.columns else pbp
    decided = sc[sc[["away_score", "home_score"]].notna().all(axis=1)]
    sched_games = int(len(sc))
    decided_games = int(len(decided))
    def pct(col, frame):
        n = float(frame[col].notna().mean()) if len(frame) else float("nan")
        return round(100.0 * n, 1) if n == n else float("nan")
    pbp_games = int(pb["game_id"].nunique()) if len(pb) else 0
    return {
        "season": season,
        "sched_games": sched_games,
        "decided_games": decided_games,
        "decided_rate": round(decided_games / sched_games, 3) if sched_games else float("nan"),
        "pbp_games": pbp_games,
        "pbp_rows": int(len(pb)),
        "epa_pct": pct("epa", pb),
        "qb_epa_pct": pct("qb_epa", pb),
        "pace_pct": pct("game_seconds_remaining", pb),
        "stadium_pct": pct("stadium", sc),
        "roof_pct": pct("roof", sc),
        "gametime_pct": pct("gametime", sc),
    }


def survey_table(rows: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Per-season survey rows -> (per-season frame, per-boundary aggregates)."""
    per_season = pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
    agg = {}
    for name, b in BOUNDARIES.items():
        core = per_season[(per_season["season"] >= b)
                          & (per_season["season"] <= SEALED_SEASON)]
        warm = per_season[per_season["season"] == b - 1]
        if core.empty:
            agg[name] = None
            continue
        def wmean(col):
            v = core[col].dropna()
            return round(float(v.mean()), 3) if len(v) else float("nan")
        games = int(core["sched_games"].sum())
        decided = int(core["decided_games"].sum())
        pbp_games = int(core["pbp_games"].sum())
        agg[name] = {
            "boundary": b, "warmup": b - 1,
            "core_seasons": [int(s) for s in core["season"]],
            "sched_games": games,
            "decided_games": decided,
            "decided_rate": wmean("decided_rate"),
            "pbp_games": pbp_games,
            "pbp_games_rate": round(pbp_games / games, 3) if games else float("nan"),
            "pbp_rows": int(core["pbp_rows"].sum()),
            "epa_pct": wmean("epa_pct"),
            "qb_epa_pct": wmean("qb_epa_pct"),
            "pace_pct": wmean("pace_pct"),
            "stadium_pct": wmean("stadium_pct"),
            "roof_pct": wmean("roof_pct"),
            "gametime_pct": wmean("gametime_pct"),
            "warmup_decided": int(warm["decided_games"].sum())
            if not warm.empty else 0,
        }
    return per_season, agg


def survey_verdict(a: dict | None) -> tuple[bool, list[str]]:
    """Boundary passes if every floor metric clears its threshold."""
    if a is None:
        return False, ["no data"]
    reasons = []
    ok = True
    for key, floor in FLOORS.items():
        v = a.get(key)
        if v is None or v != v:  # NaN
            ok = False
            reasons.append(f"{key}: absent")
        elif v < floor:
            ok = False
            reasons.append(f"{key}: {v} < {floor}")
    return ok, reasons


def _print_survey(per_season: pd.DataFrame, agg: dict[str, dict]) -> None:
    show = ["season", "sched_games", "decided_games", "decided_rate",
            "pbp_games", "epa_pct", "qb_epa_pct", "pace_pct",
            "stadium_pct", "roof_pct", "gametime_pct"]
    print("=== survey: per-season nflreadpy coverage 2009-2025 ===")
    print(per_season[show].to_string(index=False))
    print("\n=== survey: per-boundary aggregates ===")
    rows = []
    for name, a in agg.items():
        if a is None:
            rows.append({"arm": name, "boundary": BOUNDARIES[name],
                         "decided": None, "decided_rate": None,
                         "epa_pct": None, "qb_epa_pct": None,
                         "pace_pct": None, "stadium_pct": None,
                         "verdict": "NO DATA"})
            continue
        ok, reasons = survey_verdict(a)
        rows.append({
            "arm": name, "boundary": a["boundary"], "decided": a["decided_games"],
            "decided_rate": a["decided_rate"], "epa_pct": a["epa_pct"],
            "qb_epa_pct": a["qb_epa_pct"], "pace_pct": a["pace_pct"],
            "stadium_pct": a["stadium_pct"],
            "verdict": "PASS" if ok else "REJECT: " + "; ".join(reasons),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))


# ---------------------------------------------------------------------------
# Step 2 — window-parameterized walk-forward (clone of the production gate)
# ---------------------------------------------------------------------------
def run_walk_forward_window(feats: pd.DataFrame,
                            features: list[str],
                            train_seasons: list[int],
                            sealed_season: int = SEALED_SEASON) -> dict:
    """Prequential walk-forward with a configurable training window.

    A parameterized twin of ``nfl_moneyline.run_walk_forward``: identical
    fold machinery (``generate_weekly_folds`` — val weeks stay 2021-2024),
    identical ensemble/adaptive-blend/Platt pipeline, and the same return
    shape. The ONLY difference from production is that ``train_seasons`` and
    ``sealed_season`` are arguments instead of module constants.
    """
    from nfl_moneyline import (TARGET, META_COLS, _adaptive_blend,
                               _elo_logistic_p, _member_weights,
                               _score_member_table, _valid_rows, auc, ece,
                               compute_adaptive_weights, compute_metrics,
                               ensemble_predict, generate_weekly_folds,
                               logloss, platt_fit, platt_predict,
                               train_ensemble)

    preq_all = feats[feats["season"].isin(train_seasons)].copy()
    sealed = feats[feats["season"] == sealed_season].copy()

    Xcol = [f for f in features if f in feats.columns]
    if not Xcol:
        raise ValueError("no model features present in the frame")

    preq = preq_all[_valid_rows(preq_all, Xcol)].copy()
    sld = sealed[_valid_rows(sealed, Xcol)].copy()
    folds = generate_weekly_folds(preq)

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
        "_deployed": {"features": Xcol},
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def load_arm_features(boundary: int) -> pd.DataFrame:
    """Build the feature frame for one arm window (warmup B-1 + core B..2025)."""
    from nfl_features import build_features
    seasons = list(range(boundary - 1, SEALED_SEASON + 1))
    sched, pbp = pull_raw(seasons)
    decided = build_decided_frame(sched, pbp)
    feats = build_features(decided, sched, pbp)
    if "home_win" not in feats.columns:
        feats["home_win"] = (feats["home_score"] > feats["away_score"]).astype(int)
    return feats


def _platt(rec: dict) -> dict:
    return {k: rec.get(k) for k in ("logloss", "auc", "ece")}


def run_survey() -> tuple[pd.DataFrame, dict]:
    """Pull 2009-2025 per-season (processed one season at a time to keep peak
    memory low) and return (per_season, per_boundary) survey tables."""
    import nflreadpy
    from nfl_features import TIER1_NEEDS
    seasons_all = [2009] + list(range(2010, SEALED_SEASON + 1))
    sched = nflreadpy.load_schedules(seasons_all).to_pandas()
    rows = []
    for s in seasons_all:
        pbp_s = nflreadpy.load_pbp([s])
        keep = [c for c in (("game_id", "season", "epa", "qb_epa",
                             "game_seconds_remaining") + TIER1_NEEDS)
                if c in pbp_s.columns]
        pbp_s = pbp_s.select(keep).to_pandas()
        sched_s = sched[sched["season"] == s]
        rows.append(survey_season(s, sched_s, pbp_s))
        del pbp_s
    return survey_table(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=None,
                    help="pre-built features CSV (skips the nflreadpy pull; "
                         "survey still runs for the record)")
    ap.add_argument("--arms", nargs="*", choices=sorted(BOUNDARIES),
                    default=None, help="arms to run (default: survey-gated set)")
    ap.add_argument("--survey-only", action="store_true",
                    help="run the coverage survey and stop (no arms)")
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON record")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    print("=== STEP 1: window survey (verifies before committing to arms) ===")
    per_season, agg = pd.DataFrame(), {}
    if args.features:
        print("[skip] survey (--features offline path — survey needs the "
              "nflreadpy pull; arms run at the caller's discretion)")
    else:
        per_season, agg = run_survey()
        _print_survey(per_season, agg)

    if args.survey_only:
        return 0

    # Survey-gated arms: a boundary becomes an arm ONLY if it passes its
    # floors — --arms can restrict the set but never bypass the survey gate.
    if agg:
        passes = {name: survey_verdict(agg[name])[0] for name in BOUNDARIES}
    else:
        passes = {name: True for name in BOUNDARIES}
    passes["W2019"] = True  # production baseline always runs
    for name, ok in passes.items():
        if not ok:
            print(f"\n[skip] {name}: survey floors not met — arm not run "
                  f"({'; '.join(survey_verdict(agg[name])[1])})")
    todo = [n for n in BOUNDARIES
            if passes[n] and (args.arms is None or n in args.arms)]

    feats_by_arm: dict[str, pd.DataFrame] = {}
    if args.features:
        base = pd.read_csv(args.features)
        base["gameday"] = pd.to_datetime(base["gameday"])
        if "home_win" not in base.columns:
            base["home_win"] = (base["home_score"] > base["away_score"]).astype(int)
        for name in todo:
            b = BOUNDARIES[name]
            feats_by_arm[name] = base[base["season"] >= b].copy()
    else:
        for name in todo:
            b = BOUNDARIES[name]
            print(f"\n=== building feature frame for {name} (core {b}-2025) ===")
            feats_by_arm[name] = load_arm_features(b)

    results = {}
    for name in todo:
        b = BOUNDARIES[name]
        feats = feats_by_arm[name]
        cols = arm_features(feats)
        train = list(range(b, TRAIN_END + 1))
        print(f"\n=== running walk-forward arm {name} "
              f"(train {train[0]}-{train[-1]}, sealed {SEALED_SEASON}) ===")
        print(f"  decided rows in frame: {len(feats)} | model cols: "
              f"{len(cols)} | frame sha: {_frame_sha256(feats)}")
        results[name] = run_walk_forward_window(feats, cols, train)

    def _arm_row(name: str) -> str:
        rec = results[name]
        s = rec["sealed_2025"]["model_platt"]
        p = rec["pooled_preq_2021_2024"]["model_platt"]
        return (f"{name:6s} {s['logloss']}  {s['auc']}  "
                f"{s['ece']}  {p['logloss']}")

    print("\n=== window-extension ablation (sealed-2025 hold-out) ===")
    print("arm     sealed_ll  sealed_auc  sealed_ece  pooled_ll"
          "   decided_games")
    for name in todo:
        rec = results[name]
        fh = _frame_sha256(feats_by_arm[name])
        print(f"{_arm_row(name)}  {rec['fold_geometry']['pooled_oof_games']}"
              f" (oof)")
        print(f"        decided {len(feats_by_arm[name])} | frame {fh}")

    member_pooled = {n: _member_metrics(results[n], "members") for n in todo}
    member_sealed = {n: _member_metrics(results[n], "members_sealed")
                     for n in todo}

    def _member_rows(blk: dict[str, dict]) -> None:
        print(f"{'member':12s}" + "".join(f"{n:>16s}" for n in todo))
        for m in MEMBER_NAMES:
            cells = []
            for n in todo:
                e = blk[n].get(m) or {}
                cells.append(f"{e.get('logloss', '--')}/{e.get('auc', '--')}")
            print(f"{m:12s}" + "".join(f"{c:>16s}" for c in cells))

    print("\n=== per-member pooled OOF (logloss/auc) ===")
    _member_rows(member_pooled)
    print("\n=== per-member sealed 2025 (logloss/auc) ===")
    _member_rows(member_sealed)

    # ---- era-shift verdict lines --------------------------------------
    if "W2016" in todo and "W2019" in todo:
        v16 = adopt_verdict(
            _platt(results["W2019"]["sealed_2025"]["model_platt"]),
            _platt(results["W2016"]["sealed_2025"]["model_platt"]),
            _platt(results["W2019"]["pooled_preq_2021_2024"]["model_platt"]),
            _platt(results["W2016"]["pooled_preq_2021_2024"]["model_platt"]))
        print("\nVERDICT (W2016 vs W2019):",
              "ADOPT — the wider window beats 2019 on sealed logloss AND AUC"
              " -> extend the core window to 2016 (also unlocks Tier-5 "
              "participation data)"
              if v16["adopt"] else
              "DON'T ADOPT — W2016 does not beat W2019 on sealed logloss AND "
              "AUC -> era non-stationarity is real; stay at 2019",
              "|", " | ".join(v16["reason"]))
    if "W2014" in todo and "W2016" in todo:
        v14 = adopt_verdict(
            _platt(results["W2016"]["sealed_2025"]["model_platt"]),
            _platt(results["W2014"]["sealed_2025"]["model_platt"]),
            _platt(results["W2016"]["pooled_preq_2021_2024"]["model_platt"]),
            _platt(results["W2014"]["pooled_preq_2021_2024"]["model_platt"]))
        print("VERDICT (W2014 vs W2016):",
              "ADOPT — going back to 2014 still helps -> extend further"
              if v14["adopt"] else
              "DON'T ADOPT — W2014 loses to W2016 -> the era floor is 2016",
              "|", " | ".join(v14["reason"]))

    if args.no_record:
        return 0

    baseline_fh = _frame_sha256(feats_by_arm["W2019"]) if "W2019" in todo \
        else _frame_sha256(next(iter(feats_by_arm.values())))
    record = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "frame_sha256": baseline_fh,
        "baseline": "deployed market-free 12-pool (FEATURE_COLUMNS minus "
                    "is_home); arm comparison on the same sealed-2025 hold-out",
        "survey": {
            "floors": FLOORS,
            "per_season": per_season.to_dict("records"),
            "per_boundary": agg,
            "feature_source_map": FEATURE_SOURCE,
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
            "sealed_model_platt": _platt(rec["sealed_2025"]["model_platt"]),
            "pooled_model_platt": _platt(rec["pooled_preq_2021_2024"]["model_platt"]),
            "members": {m: dict(v) for m, v in (rec.get("members") or {}).items()},
            "members_sealed": {m: dict(v) for m, v in
                               (rec.get("members_sealed") or {}).items()},
        }
    if "W2016" in todo and "W2019" in todo:
        record["verdicts"]["w2016_vs_w2019"] = adopt_verdict(
            _platt(results["W2019"]["sealed_2025"]["model_platt"]),
            _platt(results["W2016"]["sealed_2025"]["model_platt"]),
            _platt(results["W2019"]["pooled_preq_2021_2024"]["model_platt"]),
            _platt(results["W2016"]["pooled_preq_2021_2024"]["model_platt"]))
    if "W2014" in todo and "W2016" in todo:
        record["verdicts"]["w2014_vs_w2016"] = adopt_verdict(
            _platt(results["W2016"]["sealed_2025"]["model_platt"]),
            _platt(results["W2014"]["sealed_2025"]["model_platt"]),
            _platt(results["W2016"]["pooled_preq_2021_2024"]["model_platt"]),
            _platt(results["W2014"]["pooled_preq_2021_2024"]["model_platt"]))
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"nfl_window_ablation_{baseline_fh}.json"
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())