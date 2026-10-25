"""Pitching+ replacement ablation (A vs B, production protocol, pipeline-independent).

A = current production serving width (active_moneyline_feature_cols()).
B = A with the 6 quality/interaction families swapped to Pitching+,
    bullpen_pitches_3d triplet RETAINED, wind 1 col -> 3-col triplet (+2 net):

  sp_xfip_5g_{diff,home,away}
    -> sp_pitchingplus_5g_{diff,home,away}
  pitcher_regression_indicator_{diff,home,away}
    -> pitcher_regression_indicator_pitchingplus_{diff,home,away}
  wind_advantage_flyball_factor (single)
    -> wind_advantage_flyball_factor_pitchingplus_{diff,home,away}
  bullpen_kbb_10g_{diff,home,away}
    -> bullpen_pitchingplus_10g_{diff,home,away}
  bullpen_kbb_3g_{diff,home,away}
    -> bullpen_pitchingplus_3g_{diff,home,away}
  bullpen_meltdown_risk_{diff,home,away}
    -> bullpen_meltdown_risk_pitchingplus_{diff,home,away}

Pipeline independence: this script loads the decided frame itself,
re-derives diffs via features.add_diff_features, and writes its own
incremental JSON after every fold checkpoint. It never reads the
pipeline's predictions_history / run-diagnostics OOF; the pipeline is
not expected to store the per-fold cumulative series.

Coverage gate: if the B levels are absent from the frame (stale
game_level_features.csv pre-dating the 2026-10-24 trial), B would score
as all-NaN. The script fails LOUD with a per-column coverage table
instead of producing misleading numbers. Pass --allow-all-nan only for
harness smoke tests.

Metrics per arm: pooled OOF AUC, logloss, Brier, ECE (training.
compute_metrics, same 10-bin ECE as production headlines) on the
GRADING population (grades_pooled), plus the deployed adaptive blend
weights. Incremental history pools grades rows up to each fold.

Run with cwd=mlb-backend/backend:
  python run_pitchingplus_replacement_ablation.py --help
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

import features
import training
from config import DATA_DELIVERY_DIR
from frames import get_decided_frame

# A is the FULL production baseline (no removals). B swaps these 16
# incumbent cols (5 families x3 + the wind single) for the 18 Pitching+
# cols below (net +2: wind 1 col -> pitchingplus triplet).
B_REMOVE = [
    "sp_xfip_5g_diff", "sp_xfip_5g_home", "sp_xfip_5g_away",
    "pitcher_regression_indicator_diff",
    "pitcher_regression_indicator_home",
    "pitcher_regression_indicator_away",
    "wind_advantage_flyball_factor",
    "bullpen_kbb_10g_diff", "bullpen_kbb_10g_home", "bullpen_kbb_10g_away",
    "bullpen_kbb_3g_diff", "bullpen_kbb_3g_home", "bullpen_kbb_3g_away",
    "bullpen_meltdown_risk_diff",
    "bullpen_meltdown_risk_home",
    "bullpen_meltdown_risk_away",
]

B_ADD = [
    "sp_pitchingplus_5g_diff", "sp_pitchingplus_5g_home", "sp_pitchingplus_5g_away",
    "pitcher_regression_indicator_pitchingplus_diff",
    "pitcher_regression_indicator_pitchingplus_home",
    "pitcher_regression_indicator_pitchingplus_away",
    "wind_advantage_flyball_factor_pitchingplus_diff",
    "wind_advantage_flyball_factor_pitchingplus_home",
    "wind_advantage_flyball_factor_pitchingplus_away",
    "bullpen_pitchingplus_10g_diff",
    "bullpen_pitchingplus_10g_home",
    "bullpen_pitchingplus_10g_away",
    "bullpen_pitchingplus_3g_diff",
    "bullpen_pitchingplus_3g_home",
    "bullpen_pitchingplus_3g_away",
    "bullpen_meltdown_risk_pitchingplus_diff",
    "bullpen_meltdown_risk_pitchingplus_home",
    "bullpen_meltdown_risk_pitchingplus_away",
]

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        v = o.item()
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    return o


def coverage(frame: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    out = {}
    for c in cols:
        if c not in frame.columns:
            out[c] = 0.0
        else:
            s = pd.to_numeric(frame[c], errors="coerce")
            out[c] = float(s.notna().mean())
    return out


def incremental_history(combined: pd.DataFrame) -> list[dict]:
    """Cumulative pooled grading metrics up to each fold (incremental series)."""
    if combined.empty or "fold_idx" not in combined.columns:
        return []
    grades = (
        combined["grades_pooled"].astype(bool).to_numpy()
        if "grades_pooled" in combined.columns
        else np.ones(len(combined), dtype=bool)
    )
    y = combined["home_win"].to_numpy(dtype=float)
    p = combined["home_win_prob_model"].to_numpy(dtype=float)
    folds = sorted(combined["fold_idx"].unique().tolist())
    hist = []
    for f in folds:
        m = (combined["fold_idx"].to_numpy() <= f) & grades
        if m.sum() < 2:
            continue
        try:
            met = training.compute_metrics(y[m], p[m])
        except Exception:
            continue
        wcols = {}
        for w in ("weight_xgboost", "weight_lightgbm", "weight_elasticnet"):
            if w in combined.columns:
                vals = pd.to_numeric(combined[w], errors="coerce").to_numpy()
                fm = (combined["fold_idx"].to_numpy() == f) & grades
                vv = vals[fm]
                vv = vv[np.isfinite(vv)]
                wcols[w] = float(vv[0]) if len(vv) else None
        hist.append(clean({
            "fold_idx": int(f),
            "n_grades_cum": int(m.sum()),
            "val_end": str(combined.loc[combined.fold_idx == f, "game_date"].max())
            if "game_date" in combined.columns else None,
            **met,
            **wcols,
        }))
    return hist


def run_arm(frame: pd.DataFrame, cols: list[str], label: str,
            min_train_days: int, max_eval_folds: int) -> dict:
    from frames import get_decided_frame as _gdf
    training.set_feature_subset(cols)
    try:
        assert list(training.active_moneyline_feature_cols()) == cols, \
            "subset canonical order drift"
        snap = _gdf(frame)
        best, pooled, combined = training.walk_forward_evaluate(
            frame,
            min_train_days=min_train_days,
            max_eval_folds=max_eval_folds,
            decided_snapshot=snap,
        )
    finally:
        training.reset_feature_subset()
    weights = {}
    # Deployed weights live in the OOF weight_* columns (last fold earned)
    # plus the ensemble-info roster.
    for w in ("weight_xgboost", "weight_lightgbm", "weight_elasticnet"):
        if w in combined.columns and len(combined):
            v = pd.to_numeric(combined[w], errors="coerce").dropna()
            weights[w] = float(v.iloc[-1]) if len(v) else None
    try:
        info = training.last_ensemble_info()
    except Exception:
        info = []
    return clean({
        "label": label,
        "n_cols": len(cols),
        "pooled": pooled,
        "weights_last_fold": weights,
        "ensemble_info": info,
        "n_oof_rows": int(len(combined)),
        "history": incremental_history(combined),
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default=str(DATA_DELIVERY_DIR / "game_level_features.csv"))
    ap.add_argument("--out", default="../../scratch/mlb_pitchingplus_replacement_ab.json")
    ap.add_argument("--min-train-days", type=int, default=30)
    ap.add_argument("--max-eval-folds", type=int, default=0)
    ap.add_argument("--allow-all-nan", action="store_true",
                    help="run even if every B column is missing/all-NaN (smoke only)")
    args = ap.parse_args()

    fp = Path(args.frame)
    log(f"loading frame {fp}")
    if not fp.exists():
        # Parquet twin (fresh builds write parquet beside the run cache).
        alt = fp.with_suffix(".parquet")
        if alt.exists():
            fp = alt
        else:
            print(f"frame not found: {args.frame}", flush=True)
            return 2
    if fp.suffix == ".parquet":
        frame = pd.read_parquet(fp)
    else:
        frame = pd.read_csv(fp, low_memory=False)
    log(f"frame rows={len(frame)} cols={len(frame.columns)}")

    # Re-derive diffs locally (idempotent): never trusts the pipeline's
    # stored diffs for the trial columns.
    try:
        frame = features.add_diff_features(frame)
        log("add_diff_features re-derived")
    except Exception as exc:
        log(f"add_diff_features failed ({exc}); continuing on stored diffs")

    try:
        decided = get_decided_frame(frame)
    except Exception as exc:
        print(f"no decided games: {exc}", flush=True)
        return 2
    log(f"decided rows={len(decided)} "
        f"{decided['game_date'].min()} -> {decided['game_date'].max()}")

    base = list(training.active_moneyline_feature_cols())
    log(f"production serving width={len(base)}")
    missing_b = [c for c in B_REMOVE if c not in base]
    if missing_b:
        print(f"B_REMOVE not in serving width: {missing_b}", flush=True)
        return 2
    # A is the FULL baseline; B replaces the 16 with the 18.
    a_cols = list(base)
    b_cols = [c for c in base if c not in set(B_REMOVE)] + \
        [c for c in B_ADD if c not in base]
    # Canonical pool order (same contract as set_feature_subset).
    pool = list(training.KNOWN_FEATURE_COLS)
    unknown = [c for c in b_cols if c not in pool]
    if unknown:
        print(f"B cols outside known pool: {unknown}", flush=True)
        return 2
    keep = set(b_cols)
    b_cols = [c for c in pool if c in keep]
    keep_a = set(a_cols)
    a_cols = [c for c in pool if c in keep_a]
    log(f"A width={len(a_cols)} (full baseline); "
        f"B width={len(b_cols)} (removed {len(B_REMOVE)}, added {len(B_ADD)}, "
        f"net +{len(b_cols) - len(base)})")

    cov_a = coverage(frame, B_REMOVE + ["bullpen_pitches_3d_diff",
                                        "bullpen_pitches_3d_home",
                                        "bullpen_pitches_3d_away"])
    cov_b = coverage(frame, B_ADD)
    log("coverage A-remove (incumbent, expect high): "
        + ", ".join(f"{k}={v:.2f}" for k, v in cov_a.items()))
    log("coverage B-add (pitchingplus, must be high to trust): "
        + ", ".join(f"{k}={v:.2f}" for k, v in cov_b.items()))
    core = ["sp_pitchingplus_5g_diff", "bullpen_pitchingplus_10g_diff",
            "bullpen_pitchingplus_3g_diff"]
    core_cov = [cov_b.get(c, 0.0) for c in core]
    # Dome-only wind zeros (apply_indoor_neutral_fills writes 0 on closed
    # roofs even when the pp input is missing) must not count as signal:
    # the SP/bullpen quality levels have to exist in the frame.
    if (max(core_cov) < 0.30) and not args.allow_all_nan:
        out = clean({
            "record": "mlb_pitchingplus_replacement_ab",
            "date": date.today().isoformat(),
            "status": "blocked_missing_pitchingplus",
            "message": ("Core Pitching+ quality levels are missing/all-NaN "
                        f"(sp/bullpen diffs coverage {core_cov}). Dome-only "
                        "wind zeros do not count. Rebuild the frame from "
                        "pitches.parquet (fresh build_features + "
                        "add_diff_features) so the 6 pp levels exist, then "
                        "rerun. No walk-forward ran."),
            "a_cols": a_cols, "b_cols": b_cols,
            "coverage_a": cov_a, "coverage_b": cov_b,
        })
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        log(f"BLOCKED: wrote {args.out}")
        return 3

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict = clean({
        "record": "mlb_pitchingplus_replacement_ab",
        "date": date.today().isoformat(),
        "frame": str(fp),
        "n_decided": int(len(decided)),
        "min_train_days": args.min_train_days,
        "max_eval_folds": args.max_eval_folds,
        "a_cols": a_cols, "b_cols": b_cols,
        "coverage_a": cov_a, "coverage_b": cov_b,
        "arms": {},
    })
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for label, cols in (("A_production", a_cols), ("B_pitchingplus", b_cols)):
        log(f"ARM {label}: {len(cols)} cols, walking...")
        arm = run_arm(frame, cols, label, args.min_train_days, args.max_eval_folds)
        result["arms"][label] = arm
        out_path.write_text(json.dumps(clean(result), indent=2), encoding="utf-8")
        log(f"ARM {label} pooled={arm['pooled']} weights={arm['weights_last_fold']}")
        log(f"incremental checkpoint wrote {out_path} "
            f"({len(arm['history'])} fold rows)")

    # Paired deltas on the identical grading population are computed by the
    # caller from the two OOF frames; the headline here is pooled + history.
    print(json.dumps(clean(result), indent=2))
    log(f"done: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
