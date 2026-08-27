"""Full-harness gate: moneyline CALIBRATION_MODE platt vs identity, at the
BLEND level — standalone, run-engine READ-ONLY.

Why blend-level: the moneyline applies prequential Platt on pooled OOF and the
adaptive ensemble re-earns weights per run, so removing the map can change the
blend weights, not just member probabilities (the 17-round LGB precedent proved
member-level gains can invert at the blend). This harness measures the flip at
exactly the level that ships.

Harness (production path, untouched):
  * matrix = tune_lightgbm_optuna.prepare_data -> enriched 65-col tune frame
    (run_margin_diff attached, run engine READ-ONLY, /tmp cache reused on
    hash match) + sealed 21-day holdout (284 games, 2026-08-05 -> 2026-08-25).
  * fixed folds: 45 (identical for both arms).
  * per mode: calibration.set_calibration_mode(mode) then
    training.walk_forward_evaluate(tune) with _attach_oof_run_margins patched
    to reuse the enriched frame (no re-derivation). walk_forward_evaluate
    clears adaptive weights / calibrator at start and end -> variant isolation.
  * sealed holdout: blend with the final-fit models (best_models) + the
    pooled-OOF moneyline map (gated by mode), exactly like the deployed bundle.
  * reported per arm: pooled OOF logloss/AUC/ECE (raw + calibrated), sealed
    holdout logloss/AUC/ECE (raw + calibrated), adaptive blend weights.

Gate (task): ADOPT identity only if, vs platt:
  * sealed holdout logloss AND AUC not degraded (AUC should tie — rank
    preserving),
  * sealed holdout ECE-cal improved (expect ~0.0557 -> ~0.038),
  * pooled OOF ECE-cal improved (expect ~0.0151 -> ~0.0108) WITHOUT pooled
    OOF logloss degrading.
Otherwise keep platt. Verdict stated flatly.

Record: data_delivery/calibration_flip_<date>.json written after EACH variant
so a mid-run interruption keeps completed rows. Nothing is committed by the
harness; the default in config.py stays "platt" until the gate passes.

Usage:
    python run_calibration_flip_test.py --modes platt        # control first
    python run_calibration_flip_test.py --modes identity     # then the challenger
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

_BACKEND = Path(__file__).resolve().parent
if str(_BACKEND.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND.parent))
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import calibration  # noqa: E402
import training  # noqa: E402
from config import DATA_DELIVERY_DIR, RETRAIN_CADENCE_DAYS  # noqa: E402
from tune_lightgbm_optuna import prepare_data  # noqa: E402

EPS = 1e-7
MARGIN_COL = "run_margin_diff"
HOLDOUT_DAYS = 21
CSV = DATA_DELIVERY_DIR / "game_level_features.csv"


def _skip_margin_attach(games, splits, min_val_games, max_eval_folds,
                        retrain_cadence_days, min_train_days):
    """Reuse the already-enriched tune frame instead of re-deriving margins.

    prepare_data attached the OOF margins with the SAME machinery and cached
    the result — re-deriving per arm would re-run the whole run-engine path
    (~30-60 min) for identical values. Skipped only when the frame already
    carries the margin column at high coverage; otherwise the real derivation
    runs (both arms must be identical either way).
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


def run_variant(tune: pd.DataFrame, hold: pd.DataFrame, mode: str,
                tag: str) -> dict:
    """Full harness for one calibration mode. Returns the result row."""
    calibration.set_calibration_mode(mode)
    print(f"\n===== variant {tag}: CALIBRATION_MODE={mode} =====", flush=True)

    with patch.object(training, "_attach_oof_run_margins",
                      side_effect=_skip_margin_attach):
        best_models, pooled, combined = training.walk_forward_evaluate(tune)

    n_oof = len(combined) if combined is not None and len(combined) else 0

    # ---- sealed holdout: deployed-bundle behavior (final models + map) ----
    proba, _member_probs, _wts = training.ensemble_predict(best_models, hold)
    proba = np.clip(np.asarray(proba, dtype=float), EPS, 1 - EPS)
    cal = training.get_last_calibrator()
    proba_cal = np.clip(np.asarray(calibration.moneyline_apply(proba, cal),
                                   dtype=float), EPS, 1 - EPS)
    hold_y = hold["home_win"].to_numpy(float)
    m_raw = training.compute_metrics(hold_y, proba)
    m_cal = training.compute_metrics(hold_y, proba_cal)

    info = {e["name"]: e for e in training.last_ensemble_info()}
    weights = {name: float(e.get("weight", 0.0)) for name, e in info.items()}

    # Full-frame pooled OOF (tune folds + sealed holdout) — matches the
    # monitor's pooled set, which includes the sealed-window games as OOF
    # predictions of the final folds. The gate's "pooled OOF" legs use this.
    oof_incl = None
    if combined is not None and len(combined):
        full_y = np.concatenate([combined["home_win"].to_numpy(float), hold_y])
        full_raw = np.concatenate([
            combined["home_win_prob_model"].to_numpy(float), proba])
        full_cal = np.concatenate([
            combined["home_win_prob_model_calibrated"].to_numpy(float),
            proba_cal])
        mp = training.compute_metrics(full_y, full_raw)
        mc = training.compute_metrics(full_y, full_cal)
        oof_incl = {**mp,
                    "logloss_calibrated": mc["logloss"],
                    "brier_calibrated": mc["brier"],
                    "ece_calibrated": mc["ece"],
                    "n": int(len(full_y))}

    row = {
        "oof_incl_sealed": oof_incl,
        "mode": mode,
        "n_oof_games": int(n_oof),
        "holdout_n": int(len(hold_y)),
        "holdout_range": [str(hold["game_date"].min().date()),
                          str(hold["game_date"].max().date())],
        "adaptive_blend_weights": weights,
        "final_calibrator_applied": cal is not None,
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
          f"auc={row['oof']['auc']}  ece={row['oof']['ece']}  "
          f"ece_cal={row['oof']['ece_calibrated']}  (n={n_oof})")
    if oof_incl:
        print(f"  OOF+sealed  : logloss={oof_incl['logloss']}  "
              f"ece={oof_incl['ece']}  ece_cal={oof_incl['ece_calibrated']}  "
              f"(n={oof_incl['n']})")
    print(f"  HOLD blend  : logloss={row['holdout']['logloss']}  "
          f"auc={row['holdout']['auc']}  ece={row['holdout']['ece']}  "
          f"ece_cal={row['holdout']['ece_calibrated']}  (n={len(hold_y)})")
    print(f"  adaptive weights: {weights}")
    return row


def gate(platt: dict, ident: dict) -> dict:
    """Verdict rule from the task — stated flatly, no hedging."""
    p_h, i_h = platt["holdout"], ident["holdout"]
    p_o = platt.get("oof_incl_sealed") or platt["oof"]
    i_o = ident.get("oof_incl_sealed") or ident["oof"]

    hold_ll_ok = i_h["logloss"] <= p_h["logloss"]
    hold_auc_ok = i_h["auc"] >= p_h["auc"]
    hold_ece_cal_ok = i_h["ece_calibrated"] < p_h["ece_calibrated"]
    oof_ece_cal_ok = i_o["ece_calibrated"] < p_o["ece_calibrated"]
    oof_ll_ok = i_o["logloss"] <= p_o["logloss"]

    adopt = (hold_ll_ok and hold_auc_ok and hold_ece_cal_ok
             and oof_ece_cal_ok and oof_ll_ok)
    return {
        "verdict": "ADOPT identity" if adopt else "DON'T ADOPT (keep platt)",
        "checks": {
            "sealed_holdout_logloss_not_degraded": hold_ll_ok,
            "sealed_holdout_auc_not_degraded": hold_auc_ok,
            "sealed_holdout_ece_cal_improved": hold_ece_cal_ok,
            "pooled_oof_ece_cal_improved": oof_ece_cal_ok,
            "pooled_oof_logloss_not_degraded": oof_ll_ok,
        },
        "deltas_identity_minus_platt": {
            "holdout_logloss": round(i_h["logloss"] - p_h["logloss"], 6),
            "holdout_auc": round(i_h["auc"] - p_h["auc"], 6),
            "holdout_ece_cal": round(i_h["ece_calibrated"] - p_h["ece_calibrated"], 6),
            "oof_logloss": round(i_o["logloss"] - p_o["logloss"], 6),
            "oof_ece_cal": round(i_o["ece_calibrated"] - p_o["ece_calibrated"], 6),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes", required=True,
                    help="comma-separated modes, e.g. platt,identity")
    ap.add_argument("--target-date", type=str, default=None,
                    help="Pipeline target date (YYYY-MM-DD); defaults to today")
    args = ap.parse_args()
    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in ("platt", "identity"):
            raise SystemExit(f"unknown mode {m!r}")

    print("loading matrix + margins (prepare_data; /tmp cache reused on "
          "hash match) ...", flush=True)
    data = prepare_data(HOLDOUT_DAYS, CSV)
    tune, hold = data["tune"], data["hold"]
    print(f"tune={len(tune)} hold={len(hold)} folds={len(data['folds'])} | "
          f"margin coverage tune={100 * tune[MARGIN_COL].notna().mean():.1f}%")

    target = args.target_date or date.today().isoformat()
    stamp = target.replace("-", "")
    record_path = DATA_DELIVERY_DIR / f"calibration_flip_{stamp}.json"
    record = {"csv": str(CSV), "holdout_days": HOLDOUT_DAYS,
              "folds": len(data["folds"]),
              "config": "CALIBRATION_MODE switch only; all training params verbatim",
              "results": {}}
    if record_path.exists():
        try:
            record = json.loads(record_path.read_text())
        except json.JSONDecodeError:
            pass

    for mode in modes:
        tag = mode
        if tag in record["results"]:
            print(f"  variant {tag} already recorded — skipping")
            continue
        row = run_variant(tune, hold, mode, tag)
        record["results"][tag] = row
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True))
        print(f"  -> record updated: {record_path}")

    res = record["results"]
    if "platt" in res and "identity" in res:
        g = gate(res["platt"], res["identity"])
        record["gate"] = g
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True))
        print("\n================= GATE (identity vs platt) =================")
        for k, v in g["checks"].items():
            print(f"  {k:<44}: {v}")
        print(f"→ {g['verdict']}")
    else:
        print("\n(both arms required for the gate; ran:", list(res), ")")


if __name__ == "__main__":
    main()
