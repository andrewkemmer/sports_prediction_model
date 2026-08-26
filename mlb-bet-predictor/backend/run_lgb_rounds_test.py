"""Full-harness test: LIGHTGBM_PARAMS at the shipped 50 rounds vs lower
round counts (17r primary, 15r/20r brackets) — standalone, uncommitted,
read-only over the model.

Why: verify_lgb_winner.py showed the current LGB config is over-trained —
at its tail-ES optimum (~17 rounds) raw sealed-holdout ECE improved
0.0736 -> 0.0548 and logloss 0.6829 -> 0.6823 at the MEMBER level. This
script measures whether that gain survives the full 5-member blend.

Harness (production path, untouched):
  * matrix = tune_lightgbm_optuna.prepare_data -> enriched 65-col tune
    frame (run_margin_diff via training._attach_oof_run_margins, run
    engine READ-ONLY) + sealed 21-day holdout (last 284 games,
    2026-08-05 -> 2026-08-25). The /tmp enriched-margin cache is reused
    when the data-hash matches (identical machinery + fold geometry,
    deterministic) — the run-engine derivation is NOT re-run.
  * per variant: training.walk_forward_evaluate(tune) with
    training.LIGHTGBM_PARAMS patched to n_estimators=<rounds> only. All 5
    members blend with adaptive OOF-earned weights re-earned per run;
    prequential Platt calibration on OOF per the production path.
  * sealed holdout: blend with the final-fit models (best_models) + the
    pooled-OOF Platt map, exactly like the deployed bundle.
  * variant isolation is automatic: walk_forward_evaluate clears the
    adaptive weights / calibrator at start and end.

Gate (task): adopt 17r only if it beats 50r on the sealed holdout
(logloss AND AUC) without degrading ECE-cal AND does not lose pooled OOF.
Otherwise keep 50 rounds and record the result.

Record: data_delivery/lgb_rounds_<hash>.json, written after EACH variant
so a mid-run interruption keeps completed rows. Nothing is committed and
config.py / training.py / LIGHTGBM_PARAMS are never modified.

Usage:
    python run_lgb_rounds_test.py --variants 50        # control first
    python run_lgb_rounds_test.py --variants 17,15,20  # then the rest
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
    cached the result under /tmp — re-deriving per variant would re-run the
    whole run-engine path (~30-60 min) for identical values. Skipped only
    when the frame already carries the margin column at high coverage;
    otherwise the real derivation runs.
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


def run_variant(tune: pd.DataFrame, hold: pd.DataFrame, rounds: int,
                tag: str) -> dict:
    """Full harness for one LGB round count. Returns the result row."""
    params = dict(LIGHTGBM_PARAMS)
    params["n_estimators"] = int(rounds)
    print(f"\n===== variant {tag}: n_estimators={rounds} "
          f"(all other LIGHTGBM_PARAMS verbatim) =====", flush=True)

    with patch.object(training, "LIGHTGBM_PARAMS", params), \
         patch.object(training, "_attach_oof_run_margins",
                      side_effect=_skip_margin_attach):
        best_models, pooled, combined = training.walk_forward_evaluate(tune)

    n_oof = len(combined) if combined is not None and len(combined) else 0

    # ---- sealed holdout: deployed-bundle behavior (final models + map) ----
    proba, _member_probs, _wts = training.ensemble_predict(best_models, hold)
    proba = np.clip(np.asarray(proba, dtype=float), EPS, 1 - EPS)
    cal = training.get_last_calibrator()
    proba_cal = np.clip(np.asarray(training.apply_platt(proba, cal), dtype=float),
                        EPS, 1 - EPS)
    hold_y = hold["home_win"].to_numpy(float)
    m_raw = training.compute_metrics(hold_y, proba)
    m_cal = training.compute_metrics(hold_y, proba_cal)

    info = {e["name"]: e for e in training.last_ensemble_info()}
    lgb_w = float(info.get("lightgbm", {}).get("weight", 0.0))

    row = {
        "variant": tag,
        "rounds": int(rounds),
        "n_oof_games": int(n_oof),
        "holdout_n": int(len(hold_y)),
        "holdout_range": [str(hold["game_date"].min().date()),
                          str(hold["game_date"].max().date())],
        "lgb_blend_weight": round(lgb_w, 4),
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
    print(f"  LGB weight  : {lgb_w:.1%}  |  final calibrator applied: "
          f"{cal is not None}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", required=True,
                    help="comma-separated n_estimators values, e.g. 50,17")
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp"))
    args = ap.parse_args()
    variants = [int(v) for v in args.variants.split(",") if v.strip()]

    print("loading matrix + margins (prepare_data; /tmp cache reused on "
          "hash match) ...", flush=True)
    data = prepare_data(HOLDOUT_DAYS, CSV, cache_dir=args.cache_dir)
    tune, hold = data["tune"], data["hold"]
    h = data_hash(data["folds"])
    print(f"data-hash: {h} | tune={len(tune)} hold={len(hold)} "
          f"folds={len(data['folds'])} | margin coverage tune="
          f"{100 * tune[MARGIN_COL].notna().mean():.1f}%")

    record_path = DATA_DELIVERY_DIR / f"lgb_rounds_{h}.json"
    record = {"data_hash": h, "csv": str(CSV), "holdout_days": HOLDOUT_DAYS,
              "config": "LIGHTGBM_PARAMS verbatim, n_estimators varied only",
              "results": {}}
    if record_path.exists():
        try:
            record = json.loads(record_path.read_text())
        except json.JSONDecodeError:
            record = {"data_hash": h, "csv": str(CSV),
                      "holdout_days": HOLDOUT_DAYS,
                      "config": "LIGHTGBM_PARAMS verbatim, n_estimators varied only",
                      "results": {}}

    for r in variants:
        tag = f"{r}r"
        if tag in record["results"]:
            print(f"  variant {tag} already recorded — skipping")
            continue
        row = run_variant(tune, hold, r, tag)
        record["results"][tag] = row
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True))
        print(f"  -> record updated: {record_path}")

    res = record["results"]
    print("\n================= PER-VARIANT TABLE =================")
    header = (f"{'variant':<8}{'OOF ll':>9}{'OOF auc':>9}{'OOF ece_cal':>12}"
              f"{'HOLD ll':>9}{'HOLD auc':>9}{'HOLD ece_cal':>12}"
              f"{'LGB wt':>8}")
    print(header)
    for tag, row in res.items():
        o, m = row["oof"], row["holdout"]
        print(f"{tag:<8}{o['logloss']:>9}{o['auc']:>9}"
              f"{o['ece_calibrated']:>12}{m['logloss']:>9}{m['auc']:>9}"
              f"{m['ece_calibrated']:>12}{row['lgb_blend_weight']:>7.1%}")

    # ---- gate: 17r vs 50r ----
    if "50r" in res and "17r" in res:
        c, v = res["50r"], res["17r"]
        hold_gain = (v["holdout"]["logloss"] < c["holdout"]["logloss"]
                     and v["holdout"]["auc"] > c["holdout"]["auc"])
        cal_ok = v["holdout"]["ece_calibrated"] <= c["holdout"]["ece_calibrated"]
        pooled_ok = (v["oof"]["logloss"] <= c["oof"]["logloss"]
                     and v["oof"]["auc"] >= c["oof"]["auc"])
        verdict = "ADOPT 17r" if (hold_gain and cal_ok and pooled_ok) \
            else "KEEP 50 rounds"
        print("\n================= GATE (17r vs 50r) =================")
        print(f"sealed holdout: Δlogloss={v['holdout']['logloss'] - c['holdout']['logloss']:+.5f} "
              f"ΔAUC={v['holdout']['auc'] - c['holdout']['auc']:+.4f} "
              f"ΔECE-cal={v['holdout']['ece_calibrated'] - c['holdout']['ece_calibrated']:+.4f}")
        print(f"  holdout logloss AND AUC better : {hold_gain}")
        print(f"  holdout ECE-cal not degraded   : {cal_ok}")
        print(f"  pooled OOF not lost            : {pooled_ok}")
        print(f"→ {verdict}")
    else:
        print("\n(gate requires both 50r and 17r — run --variants 50 first, "
              "then --variants 17)")


if __name__ == "__main__":
    main()
