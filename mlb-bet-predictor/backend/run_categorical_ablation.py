"""Full-harness ablation: venue + starter-ID native categoricals (LGB/XGB).

WITH vs WITHOUT on identical folds — standalone, uncommitted, read-only over
the model. The only code change under test is TREE_CATEGORICAL_COLS:
  * WITH    = the shipped 5-column set (teams + venue_id + home/away
              starter_cat_id) — production training.py routing.
  * WITHOUT = the prior team-only pair (patched at runtime, the exact
              pre-change behavior).

Harness (production path, untouched):
  * matrix = tune_lightgbm_optuna.prepare_data -> enriched 65-col tune frame
    (run_margin_diff via training._attach_oof_run_margins, run engine
    READ-ONLY) + sealed 21-day holdout (last 284 games). The /tmp
    enriched-margin cache is reused when the data-hash matches; otherwise the
    run-engine derivation runs once and is cached (never skipped, never faked).
  * per arm: training.walk_forward_evaluate(tune) — all 5 members blend with
    adaptive OOF-earned weights re-earned per run; prequential Platt on OOF
    per the production path. Variant isolation is automatic (walk_forward_
    evaluate clears the adaptive weights / calibrator at start and end).
  * sealed holdout: blend with the final-fit models + pooled-OOF Platt map.
  * WITHOUT arm patches training.TREE_CATEGORICAL_COLS to the team-only pair
    for BOTH the fit and the holdout predict (the 2-categorical routing), so
    the measurement is exactly the pre-change model.

Gate (task): ADOPT only if WITH beats WITHOUT on the sealed holdout (logloss
AND AUC) without degrading ECE-cal AND does not lose pooled OOF. Otherwise
DON'T ADOPT and record the numbers.

Record: data_delivery/categorical_ablation_<hash>.json, written after EACH
arm so a mid-run interruption keeps completed rows. Nothing is committed and
config.py / training.py are never modified by this script.

Usage:
    python run_categorical_ablation.py --arms WITHOUT,WITH
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

_BACKEND = Path(__file__).resolve().parent
if str(_BACKEND.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND.parent))
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import training  # noqa: E402
from config import (  # noqa: E402
    DATA_DELIVERY_DIR,
    LIGHTGBM_PARAMS,
    RETRAIN_CADENCE_DAYS,
    MIN_VAL_FOLD_GAMES,
)
from tune_lightgbm_optuna import prepare_data, _sha256_file  # noqa: E402

EPS = 1e-7
MARGIN_COL = "run_margin_diff"
HOLDOUT_DAYS = 21
CSV = DATA_DELIVERY_DIR / "game_level_features.csv"


def data_hash(folds: list[dict]) -> str:
    """Same key as tune_lightgbm_optuna.prepare_data's /tmp cache."""
    key = hashlib.sha256()
    key.update(_sha256_file(CSV).encode())
    key.update(json.dumps([str(s["val_start"]) for s in folds]).encode())
    return key.hexdigest()[:16]


def _skip_margin_attach(games, splits, min_val_games, max_eval_folds,
                        retrain_cadence_days, min_train_days):
    """Reuse the already-enriched frame instead of re-deriving margins.

    prepare_data attached the OOF margins with the SAME machinery
    (training._attach_oof_run_margins, same folds, deterministic seed) and
    cached the result under /tmp — re-deriving per arm would re-run the whole
    run-engine path for identical values. Skipped only when the frame already
    carries the margin column at high coverage; otherwise the real derivation
    runs.
    """
    cov = float(games[MARGIN_COL].notna().mean()) if MARGIN_COL in games.columns else 0.0
    if cov > 0.90:
        regen = training._regenerate_splits(
            games, splits, min_val_games, retrain_cadence_days,
            max_eval_folds, min_train_days)
        print(f"  [margin] enriched frame reused (coverage {100 * cov:.1f}%) — "
              "skipping run-engine re-derivation", flush=True)
        return games, regen
    return training._attach_oof_run_margins(
        games, splits, min_val_games, max_eval_folds,
        retrain_cadence_days, min_train_days)


def run_variant(tune: pd.DataFrame, hold: pd.DataFrame,
                with_categoricals: bool, tag: str) -> dict:
    """Full harness for one arm. Returns the result row."""
    print(f"\n===== arm {tag}: TREE_CATEGORICAL_COLS = "
          f"{'5 (teams+venue+starters, NOT adopted)' if with_categoricals "
          f"else '2 (teams only, adopted)'} "
          f"=====", flush=True)

    import contextlib
    stack = contextlib.ExitStack()
    stack.enter_context(patch.object(training, "_attach_oof_run_margins",
                                     side_effect=_skip_margin_attach))
    if with_categoricals:
        # The measured (not adopted) 5-col set — teams + venue + starters.
        stack.enter_context(patch.object(
            training, "TREE_CATEGORICAL_COLS",
            list(training.FULL_TREE_CATEGORICAL_COLS)))
    else:
        # The adopted team-only pair (the live default; explicit for clarity).
        stack.enter_context(patch.object(training, "TREE_CATEGORICAL_COLS",
                                         list(training.RF_TREE_CATEGORICAL_COLS)))
    with stack:
        best_models, pooled, combined = training.walk_forward_evaluate(tune)

        n_oof = len(combined) if combined is not None and len(combined) else 0

        # ---- sealed holdout: deployed-bundle behavior (final models + map)
        proba, _member_probs, _wts = training.ensemble_predict(best_models, hold)
        proba = np.clip(np.asarray(proba, dtype=float), EPS, 1 - EPS)
        cal = training.get_last_calibrator()
        proba_cal = np.clip(np.asarray(training.apply_platt(proba, cal), dtype=float),
                            EPS, 1 - EPS)
        hold_y = hold["home_win"].to_numpy(float)
        m_raw = training.compute_metrics(hold_y, proba)
        m_cal = training.compute_metrics(hold_y, proba_cal)

    info = {e["name"]: e for e in training.last_ensemble_info()}
    weights = {name: float(e.get("weight", 0.0)) for name, e in info.items()}

    row = {
        "arm": tag,
        "with_categoricals": bool(with_categoricals),
        "n_categorical_cols": len(training.TREE_CATEGORICAL_COLS),
        "n_oof_games": int(n_oof),
        "holdout_n": int(len(hold_y)),
        "holdout_range": [str(hold["game_date"].min().date()),
                          str(hold["game_date"].max().date())],
        "member_blend_weights": {k: round(v, 4) for k, v in weights.items()},
        "oof": {
            "logloss": pooled.get("logloss"),
            "auc": pooled.get("auc"),
            "brier": pooled.get("brier"),
            "ece": pooled.get("ece"),
            "logloss_calibrated": pooled.get("logloss_calibrated"),
            "ece_calibrated": pooled.get("ece_calibrated"),
        },
        "holdout": {
            "logloss": m_raw["logloss"],
            "auc": m_raw["auc"],
            "brier": m_raw["brier"],
            "ece": m_raw["ece"],
            "logloss_calibrated": m_cal["logloss"],
            "ece_calibrated": m_cal["ece"],
        },
    }
    print(f"  OOF blend   : logloss={row['oof']['logloss']}  "
          f"auc={row['oof']['auc']}  ece_cal={row['oof']['ece_calibrated']}  "
          f"(n={n_oof})")
    print(f"  HOLD blend  : logloss={row['holdout']['logloss']}  "
          f"auc={row['holdout']['auc']}  ece_cal="
          f"{row['holdout']['ece_calibrated']}  (n={len(hold_y)})")
    print(f"  weights     : {row['member_blend_weights']}")
    return row


def decide(without: dict, with_: dict) -> dict:
    """Holdout gate (the task's rule): WITH wins only if it beats WITHOUT on
    the sealed holdout logloss AND AUC, without degrading ECE-cal, and does
    not lose pooled OOF."""
    wo_h, wi_h = without["holdout"], with_["holdout"]
    wo_o, wi_o = without["oof"], with_["oof"]
    hold_gain = (wi_h["logloss"] < wo_h["logloss"]
                 and wi_h["auc"] > wo_h["auc"])
    cal_ok = wi_h["ece_calibrated"] <= wo_h["ece_calibrated"]
    pooled_ok = (wi_o["logloss"] <= wo_o["logloss"]
                 and wi_o["auc"] >= wo_o["auc"])
    return {
        "verdict": "ADOPT" if (hold_gain and cal_ok and pooled_ok) else "DON'T ADOPT",
        "holdout_gain": bool(hold_gain),
        "holdout_cal_ok": bool(cal_ok),
        "pooled_ok": bool(pooled_ok),
        "delta_holdout_logloss": round(wi_h["logloss"] - wo_h["logloss"], 5),
        "delta_holdout_auc": round(wi_h["auc"] - wo_h["auc"], 4),
        "delta_holdout_ece_cal": round(
            wi_h["ece_calibrated"] - wo_h["ece_calibrated"], 4),
        "delta_pooled_logloss": round(wi_o["logloss"] - wo_o["logloss"], 5),
        "delta_pooled_auc": round(wi_o["auc"] - wo_o["auc"], 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="WITHOUT,WITH",
                    help="comma-separated arms, e.g. WITHOUT,WITH")
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp"))
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    print("loading matrix + margins (prepare_data; /tmp cache reused on "
          "hash match) ...", flush=True)
    data = prepare_data(HOLDOUT_DAYS, CSV, cache_dir=args.cache_dir)
    tune, hold = data["tune"], data["hold"]
    h = data_hash(data["folds"])
    print(f"data-hash: {h} | tune={len(tune)} hold={len(hold)} "
          f"folds={len(data['folds'])} | margin coverage tune="
          f"{100 * tune[MARGIN_COL].notna().mean():.1f}%")

    record_path = DATA_DELIVERY_DIR / f"categorical_ablation_{h}.json"
    record = {"data_hash": h, "csv": str(CSV), "holdout_days": HOLDOUT_DAYS,
              "categorical_set": {
                  "WITH": list(training.FULL_TREE_CATEGORICAL_COLS),
                  "WITHOUT": list(training.RF_TREE_CATEGORICAL_COLS)},
              "results": {}}
    if record_path.exists():
        try:
            record = json.loads(record_path.read_text())
        except json.JSONDecodeError:
            record = {"data_hash": h, "csv": str(CSV),
                      "holdout_days": HOLDOUT_DAYS,
                      "categorical_set": record["categorical_set"],
                      "results": {}}

    for arm in arms:
        if arm in record["results"]:
            print(f"  arm {arm} already recorded — skipping")
            continue
        row = run_variant(tune, hold, arm == "WITH", arm)
        record["results"][arm] = row
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True))
        print(f"  -> record updated: {record_path}")

    res = record["results"]
    if "WITHOUT" in res and "WITH" in res:
        gate = decide(res["WITHOUT"], res["WITH"])
        record["gate"] = gate
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True))
        print("\n================= GATE (WITH vs WITHOUT) =================")
        print(f"sealed holdout: Δlogloss={gate['delta_holdout_logloss']:+.5f} "
              f"ΔAUC={gate['delta_holdout_auc']:+.4f} "
              f"ΔECE-cal={gate['delta_holdout_ece_cal']:+.4f}")
        print(f"  holdout logloss AND AUC better : {gate['holdout_gain']}")
        print(f"  holdout ECE-cal not degraded   : {gate['holdout_cal_ok']}")
        print(f"  pooled OOF not lost            : {gate['pooled_ok']}")
        print(f"→ {gate['verdict']}")
    else:
        print("\n(gate requires both WITHOUT and WITH arms — run "
              "--arms WITHOUT first, then --arms WITH)")


if __name__ == "__main__":
    main()
