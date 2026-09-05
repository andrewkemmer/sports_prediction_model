"""NFL per-side mean-bias calibration + joint-layer chain re-run (record-only).

Adds a leak-free LINEAR recalibration of the per-side means (prediction-layer
transform) and re-runs the full joint chain through the EXISTING engine
entrypoints — ``nfl_per_side_engine.py`` and ``nfl_joint_engine.py`` are NOT
modified. FEATURE_COLUMNS / 12-pool / moneyline / daily pipeline untouched.

Pipeline (identical geometry to the prior records — same 88 folds, pooled OOF
2021-24 n=1,091, sealed 2025 n=285):
1. Diagnostics (no retraining): per-side mean OOF residual by season and by
   prediction decile; OLS actual ~ a*pred + b with CIs; advisory
   classification (offset / slope tilt / curvature / time trend); sealed
   2025 bias (does the −1.49 away bias persist?).
2. Recalibration: per side OLS actual ~ a*pred + b on POOLED OOF ONLY;
   pred_cal = b + a*pred applied to pooled AND sealed 2025 (never refit on
   sealed — fit_on marker + season guard). Bias before/after, pooled + sealed.
3. Chain re-run: residuals recomputed from calibrated preds; joint params
   (family / sigma / rho) refit on the recalibrated pooled table via
   ``nfl_joint_engine.fit_joint_params`` (sigma re-estimation is in scope —
   the old sigma was fit on biased residuals); per-game joint PMFs rebuilt
   via ``build_joint_pmfs``; gates G1-G5 re-run AND the totals ECE seam check
   (prior 0.138, hot top bin). Before-figures read from the joint record on
   main (nfl_joint_<sha>.json, commit 4c69cdb) — deterministic, so the
   stored numbers ARE the before numbers.
4. Record nfl_mean_bias_calibration_<sha>.json + explicit statement on
   whether the construction change (side-anchored away features) remains
   warranted.

Gates (same rules as the joint step-1 runner):
  G1 — per-side distributional CRPS >= 5% better than same-sample
       climatological on pooled OOF (both legs); sealed = external check.
  G2 — mean calibrated tie mass == empirical rate within +-0.2pp.
  G3 — post-IPF row/col sums == marginals to 1e-9.
  G4 — derived-ML coherence vs the C0 anchor (reported, not hard-gated).
  G5 — determinism: two identical builds byte-identical.

Exit criteria (if met, the market layer is queued next):
  G1 passes BOTH legs, per-side pooled bias < 0.1 pts, totals ECE improves
  (recorded), G5 passes. If the away leg still fails, the next lever is the
  sigma/climatology layer, NOT the means.

Usage:
    cd nfl-backend && python3 backend/run_nfl_mean_bias_calibration.py
        [--features <csv>] [--no-record] [--skip-seam]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nfl_bias_calibration import (
    ACTUAL_COLS, CAL_PRED, CAL_RESID, PRED_COLS, RESID_COLS,
    apply_calibration, diagnose, engine_table, fit_calibration,
)
from nfl_joint_engine import build_joint_pmfs, fit_joint_params, \
    load_residual_artifact
from nfl_run_engine_legacy_windows import SEALED_SEASON, TRAIN_SEASONS
from nfl_moneyline import compute_metrics
from nfl_per_side_engine import SIDE_TARGETS
from run_nfl_margin_ablation import _frame_sha256, load_features
from run_nfl_joint import (C0_ANCHOR, _crps_vs_climatology, _sealed_predictions,
                           _seam_check)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY = ROOT_DIR / "data_delivery"
DECIDED_FRAME = DATA_DELIVERY / "nfl_game_level_features.csv"
ARTIFACT = DATA_DELIVERY / "nfl_per_side_oof_residuals_3e8c8a510f04.csv"

# Prior-record provenance for the before-figures (deterministic chain ⇒ the
# stored numbers ARE the before numbers on identical code paths).
JOINT_RECORD_NAME = "nfl_joint_3e8c8a510f04.json"

SCOPE_PIN = (
    "Step adds a leak-free LINEAR recalibration of the per-side means "
    "(prediction-layer transform: per-side OLS actual ~ a*pred + b fit on "
    "pooled OOF only; pred_cal = b + a*pred applied to pooled AND sealed "
    "2025) and re-runs the joint chain through the EXISTING engine "
    "entrypoints — no engine edits, no wiring. Sigma re-estimation is in "
    "scope because the old sigma was fit on biased residuals. Market "
    "pricing/calibration paths, artifact emitters, and wire-in remain later "
    "phases; future market-layer wiring consumes the stored params JSON."
)

# OLS expectation for away sigma after removing the ~1.49 offset (recorded as
# the pinned re-estimate check): sqrt(9.0632^2 - 1.49^2) ~ 8.94.
AWAY_SIGMA_BEFORE = 9.0632
AWAY_BIAS_BEFORE = 1.49


def _frame_sha() -> str:
    return hashlib.sha256(DECIDED_FRAME.read_bytes()).hexdigest()[:12]


def _side_bias(pred: pd.Series, actual: pd.Series) -> dict:
    r = actual.to_numpy(float) - pred.to_numpy(float)
    return {"n": int(len(r)), "mean_resid": round(float(r.mean()), 4),
            "rmse": round(float(np.sqrt(np.mean(r ** 2))), 4),
            "mae": round(float(np.abs(r).mean()), 4)}


def _mae(df: pd.DataFrame, pred_col: str, actual_col: str) -> float:
    r = df[actual_col].to_numpy(float) - df[pred_col].to_numpy(float)
    return round(float(np.abs(r).mean()), 4)


def _before_from_joint_record() -> dict[str, Any]:
    """Read the joint step-1 record (main, commit 4c69cdb) for the
    before-figures — the chain is deterministic, so stored == recomputed."""
    rec_path = DATA_DELIVERY / JOINT_RECORD_NAME
    if not rec_path.exists():
        raise RuntimeError(f"prior joint record missing: {rec_path}")
    r = json.loads(rec_path.read_text())
    crps = r["crps_vs_climatology"]
    seam = r.get("data_seam") or {}
    return {
        "provenance": f"{JOINT_RECORD_NAME} on main (commit 4c69cdb)",
        "sigma": r["sigma_curve"],
        "rho": r["rho"],
        "tie": r["tie"],
        "crps_pooled": crps["pooled"],
        "crps_sealed": crps["sealed"],
        "totals_ece": seam.get("totals", {}).get("ece") if seam.get("ok")
                      else None,
        "covers_ece": seam.get("covers", {}).get("ece") if seam.get("ok")
                      else None,
        "derived_ml_metrics": r["gates"]["g4"].get("metrics"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=None,
                    help="pre-built features CSV (skips the nflreadpy pull)")
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON record")
    ap.add_argument("--skip-seam", action="store_true",
                    help="skip the nflreadpy seam check (totals ECE after "
                         "then unavailable)")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    warnings.filterwarnings("ignore",
                            message="The argument 'eval_set' is deprecated")

    print("=" * 70)
    print("  NFL Per-Side Mean-Bias Calibration + Joint Chain Re-run "
          "(record-only)")
    print("=" * 70)
    t0 = time.time()
    frame_sha = _frame_sha()
    print(f"frame_sha256={frame_sha}")

    # ---- 0. Inputs: artifact (loud guard) + decided frame + sealed mu's ----
    art = load_residual_artifact(ARTIFACT)
    feats = load_features(args.features)
    art = art.merge(feats[["game_id", "home_score", "away_score", "season"]],
                    on="game_id", how="left")
    if art["home_score"].isna().any():
        raise RuntimeError("artifact join left NaN actuals — "
                           "frame/artifact mismatch")
    rounds = {"home": int(art["best_iter_home"].median()),
              "away": int(art["best_iter_away"].median())}
    sealed_raw = _sealed_predictions(feats, rounds)
    print(f"Pooled OOF n={len(art)} (2021-24) | sealed {SEALED_SEASON} "
          f"n={len(sealed_raw)} | rounds={rounds}")

    # ---- 1. Step-1 diagnostics (raw, no retraining) ----
    print("\n[1] Diagnostics (raw pooled-OOF residuals)...")
    diag = diagnose(art)
    for side in PRED_COLS:
        d = diag[side]
        print(f"  {side}: bias={d['stats']['mean_resid']} rmse="
              f"{d['stats']['rmse']} | OLS a={d['ols_actual_on_pred']['a']} "
              f"b={d['ols_actual_on_pred']['b']} | labels="
              f"{d['classification']['labels'] or 'none'}")
        seas = ", ".join(f"{s['season']}:{s['mean_resid']}"
                         for s in d["by_season"])
        print(f"    by season: {seas}")
    sealed_bias = {side: _side_bias(sealed_raw[PRED_COLS[side]],
                                    sealed_raw[ACTUAL_COLS[side]])
                   for side in PRED_COLS}
    for side in PRED_COLS:
        print(f"  sealed {SEALED_SEASON} {side} bias: "
              f"{sealed_bias[side]['mean_resid']} (n="
              f"{sealed_bias[side]['n']})")

    # ---- 2. Recalibration (pooled OOF ONLY; apply to pooled + sealed) ----
    print("\n[2] Recalibration (per-side OLS actual ~ a*pred + b, pooled "
          "OOF only)...")
    cal = fit_calibration(art, sealed_season=SEALED_SEASON)
    for side in PRED_COLS:
        m = cal[side]
        print(f"  {side}: a={m['a']} CI[{m['a_ci_low']},{m['a_ci_high']}] "
              f"b={m['b']} CI[{m['b_ci_low']},{m['b_ci_high']}] r2={m['r2']}")
    art_cal = apply_calibration(art, cal)
    sealed_cal = apply_calibration(sealed_raw, cal)
    bias_ba = {}
    for side in PRED_COLS:
        bias_ba[side] = {
            "pooled_before": round(float(art[RESID_COLS[side]].mean()), 4),
            "pooled_after": round(float(art_cal[CAL_RESID[side]].mean()), 4),
            "sealed_before": sealed_bias[side]["mean_resid"],
            "sealed_after": round(float(sealed_cal[CAL_RESID[side]].mean()), 4),
            "pooled_mae_before": _mae(art, PRED_COLS[side],
                                      ACTUAL_COLS[side]),
            "pooled_mae_after": _mae(art_cal, CAL_PRED[side],
                                     ACTUAL_COLS[side]),
            "sealed_mae_before": _mae(sealed_raw, PRED_COLS[side],
                                      ACTUAL_COLS[side]),
            "sealed_mae_after": _mae(sealed_cal, CAL_PRED[side],
                                     ACTUAL_COLS[side]),
        }
        print(f"  {side}: pooled bias {bias_ba[side]['pooled_before']} -> "
              f"{bias_ba[side]['pooled_after']} | sealed bias "
              f"{bias_ba[side]['sealed_before']} -> "
              f"{bias_ba[side]['sealed_after']}")

    # ---- 3. Chain re-run through the EXISTING joint engine ----
    print("\n[3] Joint chain re-run (engine entrypoints unmodified)...")
    eng_pooled = engine_table(art_cal)
    eng_sealed = engine_table(sealed_cal)
    params_cal = fit_joint_params(eng_pooled)
    print(f"  joint params after: family={params_cal['family']} "
          f"sigma_h={params_cal['sigma_h']} sigma_a={params_cal['sigma_a']} "
          f"rho={params_cal['rho']} CI={params_cal['rho_ci']}")

    n_ties = int((art["home_score"] == art["away_score"]).sum())
    p_tie = n_ties / len(art)
    print(f"  final-tie base: {n_ties}/{len(art)} = {p_tie:.5f}")

    pooled_pmfs, pooled_sum = build_joint_pmfs(eng_pooled, params_cal, p_tie)
    pooled_derived = pooled_sum["derived"].copy()
    pooled_derived = pooled_derived.merge(
        eng_pooled[["game_id", "home_score", "away_score"]],
        on="game_id", how="left")
    sealed_pmfs, sealed_sum = build_joint_pmfs(eng_sealed, params_cal, p_tie)
    sealed_derived = sealed_sum["derived"].copy()
    sealed_derived = sealed_derived.merge(
        eng_sealed[["game_id", "home_score", "away_score"]],
        on="game_id", how="left")
    print(f"  joint PMFs: pooled n={len(pooled_pmfs)} sealed "
          f"n={len(sealed_pmfs)} | calibrated tie mean="
          f"{pooled_sum['summary']['d_calibrated_mean']}")

    # Gates.
    crps = _crps_vs_climatology({"pooled": eng_pooled, "sealed": eng_sealed},
                                params_cal)
    g1_pooled = all(crps["pooled"][f"ratio_{s}"] <= 0.95
                    for s in ("home", "away"))
    print(f"  G1 pooled: home {crps['pooled']['improvement_pct_home']}% "
          f"away {crps['pooled']['improvement_pct_away']}% → {g1_pooled}")
    d_cal = float(np.mean([np.trace(p) for p in pooled_pmfs]))
    g2_delta_pp = (d_cal - p_tie) * 100
    g2 = abs(g2_delta_pp) <= 0.2
    g3_err = pooled_sum["summary"]["max_marginal_err_post_ipf"]
    g3 = g3_err is not None and g3_err <= 1e-9
    y_ml = (pooled_derived["home_score"] > pooled_derived["away_score"])
    ml_metrics = compute_metrics(y_ml.astype(float).to_numpy(),
                                 pooled_derived["derived_ml"].to_numpy(float))
    g4_flag = ml_metrics["logloss"] - C0_ANCHOR["logloss"] > 0.02
    _pmfs2, sum2 = build_joint_pmfs(eng_pooled, params_cal, p_tie)
    det = (pooled_sum["derived"].to_csv(index=False)
           == sum2["derived"].to_csv(index=False))
    print(f"  G2 tie Δ={g2_delta_pp:+.4f}pp → {g2} | G3 marg err {g3_err} → "
          f"{g3} | G4 derived-ML {ml_metrics} (flag={g4_flag}) | G5 → {det}")

    seam = {} if args.skip_seam else _seam_check(feats, pooled_pmfs,
                                                 pooled_derived)
    if seam:
        totals_ece_after = (seam.get("totals") or {}).get("ece") \
            if seam.get("ok") else None
        covers_ece_after = (seam.get("covers") or {}).get("ece") \
            if seam.get("ok") else None
        print(f"  seam: ok={seam.get('ok')} totals ECE={totals_ece_after} "
              f"covers ECE={covers_ece_after}")
    else:
        totals_ece_after = covers_ece_after = None

    # ---- before/after assembly ----
    before = _before_from_joint_record()
    sigma_ba = {
        "home": {"before": before["sigma"]["home"]["sigma0"],
                 "after": params_cal["sigma_h"]["sigma0"]},
        "away": {"before": before["sigma"]["away"]["sigma0"],
                 "after": params_cal["sigma_a"]["sigma0"]},
    }
    totals_ece_before = before["totals_ece"]
    covers_ece_before = before["covers_ece"]
    print(f"\n  sigma before/after: home {sigma_ba['home']} away "
          f"{sigma_ba['away']} (expect away ~ "
          f"{np.sqrt(AWAY_SIGMA_BEFORE ** 2 - AWAY_BIAS_BEFORE ** 2):.2f})")
    print(f"  totals ECE: {totals_ece_before} -> {totals_ece_after}")

    # ---- exit criteria ----
    bias_ok = all(abs(bias_ba[s]["pooled_after"]) < 0.1 for s in PRED_COLS)
    ece_improved = (totals_ece_before is not None
                    and totals_ece_after is not None
                    and totals_ece_after < totals_ece_before)
    exit_criteria = {
        "g1_both_legs": bool(g1_pooled),
        "pooled_bias_lt_0_1": bool(bias_ok),
        "totals_ece_improved": bool(ece_improved),
        "g5_pass": bool(det),
        "all_met": bool(g1_pooled and bias_ok and ece_improved and det),
    }
    print(f"\n  exit criteria: {exit_criteria}")

    # Construction-change statement (Step 4 content).
    worst_a = max(abs(cal[s]["a"] - 1.0) for s in PRED_COLS)
    if worst_a > 0.15:
        flag_text = (
            "EXCEEDS 0.15 — side-anchored away features remain a warranted "
            "construction-change candidate for the future view-expansion "
            "work; recorded, not a blocker here")
    else:
        flag_text = (
            "within 0.15 — no construction-change trigger from the linear "
            "slope; the mean bias is an offset-type miss, addressable at "
            "the prediction layer")
    construction_statement = (
        f"|a - 1| max = {round(worst_a, 3)} — {flag_text}. "
        "The away bias (−1.49 pooled, sign +/− per side) is the diagnosed "
        "defect; this step's linear map corrects it leak-free at the "
        "prediction layer.")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    record = {
        "record": "nfl_mean_bias_calibration",
        "frame_sha": frame_sha,
        "frame_sha256": _frame_sha256(feats),
        "written_utc": pd.Timestamp.now("UTC").isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "scope": SCOPE_PIN,
        "geometry": {
            "seasons": sorted(feats["season"].unique().tolist()),
            "train_seasons": TRAIN_SEASONS,
            "sealed_season": SEALED_SEASON,
            "n_folds": 88,
            "pooled_oof_n": int(len(art)),
            "sealed_n": int(len(sealed_raw)),
            "view": "12-pool per-side PIT (SIDE_FEATURES, unchanged)",
            "engines_modified": False,
            "note": "per-side + joint engine modules untouched; "
                    "prediction-layer transform only",
        },
        "step1_diagnostics": {
            "pooled_oof_n": int(len(art)),
            "per_side": diag,
            "sealed_2025_bias": sealed_bias,
            "sealed_bias_question": (
                "does the −1.49 away OOF bias persist on sealed 2025? — "
                "see sealed_2025_bias.away.mean_resid (fit-only refill, "
                "never refit on sealed)"),
        },
        "step2_calibration": {
            "method": cal["method"],
            "fit_on": cal["fit_on"],
            "sealed_season": cal["sealed_season"],
            "leak_guards": ("fit_calibration raises on any season >= sealed "
                            "in the fit input; apply_calibration refuses "
                            "cal dicts whose fit_on != pooled_oof"),
            "home": cal["home"],
            "away": cal["away"],
            "bias_and_mae_before_after": bias_ba,
            "pooled_calibration_is_in_sample_note": (
                "the linear map (2 params/side) is fit on the pooled rows "
                "being scored → pooled bias-after ~ 0 is by construction; "
                "sealed 2025 is the honest external check"),
        },
        "step3_chain_rerun": {
            "joint_params_after": {
                "family": params_cal["family"],
                "sigma_h": params_cal["sigma_h"],
                "sigma_a": params_cal["sigma_a"],
                "rho": params_cal["rho"],
                "rho_ci": params_cal["rho_ci"],
                "rho_n": params_cal["rho_n"],
                "fit_on": params_cal["fit_on"],
            },
            "ll_table_after": params_cal["ll_table"],
            "sigma_before_after": sigma_ba,
            "sigma_note": ("old sigma was fit on biased residuals — "
                           "re-estimation in scope; away expectation "
                           f"sqrt({AWAY_SIGMA_BEFORE}^2 − "
                           f"{AWAY_BIAS_BEFORE}^2) ≈ "
                           f"{np.sqrt(AWAY_SIGMA_BEFORE**2 - AWAY_BIAS_BEFORE**2):.2f}"),
            "tie": {"p_tie_empirical": round(p_tie, 5), "n_ties_pooled":
                    n_ties,
                    "d_raw_mean_after": pooled_sum["summary"]["d_raw_mean"],
                    "d_calibrated_mean_after":
                        pooled_sum["summary"]["d_calibrated_mean"]},
            "crps_vs_climatology_after": crps,
            "totals_ece_before_after": {
                "before": totals_ece_before,
                "after": totals_ece_after,
                "covers_ece_before": covers_ece_before,
                "covers_ece_after": covers_ece_after,
                "before_provenance": before["provenance"],
                "hot_top_bin_prior": (
                    "prior totals top bin pred 0.83 vs actual 0.55 (joint "
                    "record) — see seam.totals.bins after"),
            },
            "seam": seam if seam else {"skipped": True},
            "gates": {
                "g1": {"pass": bool(g1_pooled), "pooled": crps["pooled"],
                       "sealed": crps["sealed"],
                       "rule": "per-side distributional CRPS >= 5% better "
                               "than same-sample climatological on pooled "
                               "OOF (sealed = external check)"},
                "g2": {"pass": bool(g2), "delta_pp": round(g2_delta_pp, 4),
                       "rule": "mean calibrated tie mass == empirical rate "
                               "within +-0.2pp"},
                "g3": {"pass": bool(g3), "max_err": g3_err,
                       "rule": "post-IPF row/col sums == marginals to 1e-9"},
                "g4": {"metrics": ml_metrics, "c0_anchor": C0_ANCHOR,
                       "flag": bool(g4_flag),
                       "flag_rule": "flagged (coherency note, not failure) "
                                    "if derived-ML logloss trails C0 by "
                                    "> 0.02"},
                "g5": {"pass": bool(det),
                       "method": "two identical builds byte-identical "
                                 "(derived tables CSV-equal)"},
            },
            "g1_before_after_summary": {
                "home_improvement_pct": [
                    before["crps_pooled"]["improvement_pct_home"],
                    crps["pooled"]["improvement_pct_home"]],
                "away_improvement_pct": [
                    before["crps_pooled"]["improvement_pct_away"],
                    crps["pooled"]["improvement_pct_away"]],
            },
        },
        "before_reference": {
            "source": before["provenance"],
            "sigma": before["sigma"],
            "rho": before["rho"],
            "tie": before["tie"],
            "crps_pooled": before["crps_pooled"],
            "crps_sealed": before["crps_sealed"],
            "totals_ece": before["totals_ece"],
            "covers_ece": before["covers_ece"],
            "derived_ml_metrics": before["derived_ml_metrics"],
        },
        "exit_criteria": exit_criteria,
        "construction_change_statement": construction_statement,
        "next_lever_if_away_still_fails": (
            "sigma/climatology layer, NOT means — recorded per the task "
            "spec exit criteria"),
        "feature_columns_untouched": True,
        "judgment_calls": {
            "linear_over_isotonic": ("n=1,091 is thin; linear OLS chosen; "
                                     "isotonic only if post-linear "
                                     "diagnostics show clear curvature "
                                     "(see step1 per-side curvature_swing_pts)"),
            "per_side_independent_maps": ("home a/b and away a/b fitted "
                                          "separately, never pooled"),
            "prediction_layer_only": ("calibration is a prediction-layer "
                                      "transform; engine modules untouched; "
                                      "future market-layer wiring consumes "
                                      "the stored params JSON"),
            "sigma_reestimation_in_scope": ("old sigma fit on biased "
                                            "residuals — re-estimated from "
                                            "calibrated residuals"),
        },
    }
    if not args.no_record:
        record_path = DATA_DELIVERY / f"nfl_mean_bias_calibration_{frame_sha}.json"
        record_path.write_text(json.dumps(record, indent=2, default=str))
        print(f"\nRecord: {record_path.name}")
    else:
        print("\n[--no-record] record skipped")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    raise SystemExit(main())
