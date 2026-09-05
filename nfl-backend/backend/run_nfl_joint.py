"""NFL per-side joint layer — step-1 runner (record-only).

Builds the correlated per-game joint PMF over NFL final integer scores from
the step-1 per-side outputs (commit 688d417): residual artifact
(nfl_per_side_oof_residuals_<sha>.csv) + fit-only sealed predictions.
Record-only — no wiring; FEATURE_COLUMNS / 12-pool / moneyline / daily
pipeline untouched.

Pipeline (identical geometry to the per-side / margin engines):
1. Load the residual artifact; FAIL LOUDLY on missing cols / empty / wrong
   counts. Sealed 2025 mu's come from the same fit-only refill at the
   artifact's per-side median rounds (never refit on sealed).
2. Per-side sigma(mu) curves (power law vs constant RMSE) + marginal-family
   candidates (DN vs NB) selected by pooled-OOF integer LL (DN tiebreak
   within 10 LL units; NB does not auto-win).
3. Global rho on pooled-OOF standardized pairs + Fisher's-z CI.
4. Per-game joint PMFs on grid 0..75 via the Gaussian copula; row-mass
   check < 1e-4; IPF tie calibration to the pooled empirical final-tie rate
   (constant base — covariate model data-limited, not attempted).
5. Derived probabilities per game: margin PMF, total PMF, P(cover -L),
   P(over U), tie mass, derived ML = P(H>A)/(1 - P_tie).
6. Data-seam check EARLY: nflverse spread_line/total_line coverage on the
   pooled rows. >=90% → calibrate covers/totals vs actual offered lines
   (reliability tables); else fall back to integer-grid thresholds and
   record the gap prominently.
7. Record JSON: family choice, sigma params, rho + CI, D_raw vs calibrated
   tie mass, CRPS + calibration tables, data-seam note, gates G1-G5.

Gates:
  G1 — distributional CRPS per side (grid CDFs) on pooled OOF beats the
       same-sample climatological CRPS by >= 5% relative (sealed = external
       check).
  G2 — mean calibrated tie mass == empirical final-tie rate within +-0.2pp.
  G3 — post-IPF row/col sums == per-side marginal PMFs to 1e-9.
  G4 — derived-ML calibration reported honestly, not hard-gated (logloss /
       AUC / ECE, 10 bins, pooled OOF decided-only), anchored vs the clean
       C0 incumbent (pooled ll 0.6312 / auc 0.6923 / ece 0.0346). Flag if
       derived ML trails by > 0.02 logloss — coherency note, not failure.
  G5 — determinism: two identical runs byte-identical.

Usage:
    cd nfl-backend && python3 backend/run_nfl_joint.py [--features <csv>]
        [--no-record] [--skip-seam]
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

from nfl_joint_engine import (
    build_joint_pmfs, cover_prob, crps_discrete, fit_joint_params,
    load_residual_artifact, marginal_pmf, over_prob, sigma_callable,
    margin_pmf_from_joint, total_pmf_from_joint,
)
from nfl_run_engine_legacy_windows import (SEALED_SEASON, TRAIN_SEASONS,
                                           VAL_SEASONS)
from nfl_moneyline import (_valid_rows, compute_metrics)
from nfl_per_side_engine import SIDE_FEATURES, refit_per_side
from run_nfl_margin_ablation import _frame_sha256, load_features

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY = ROOT_DIR / "data_delivery"
DECIDED_FRAME = DATA_DELIVERY / "nfl_game_level_features.csv"

SCOPE_PIN = (
    "Step-1 delivers the correlated per-side joint PMF plus validated "
    "derived probabilities. Market pricing/calibration paths, artifact "
    "emitters, and wire-in are later phases. Marginal family and sigma "
    "chosen empirically on pooled OOF; rho is a global scalar. Tie handling "
    "uses an IPF-calibrated final-tie diagonal because tied-regulation games "
    "resolve in OT; regulation-score/OT modeling is a deferred improvement."
)

# G4 anchor: clean C0 incumbent (served 12-pool moneyline, pooled OOF).
C0_ANCHOR = {"logloss": 0.6312, "auc": 0.6923, "ece": 0.0346}
G4_LL_FLAG = 0.02

# Integer-grid reference thresholds (reported alongside seam calibration).
COVER_THRESHOLDS = [3.5, 6.5, 10.5]
OVER_THRESHOLDS = [42.5, 47.5, 52.5]

ARTIFACT = DATA_DELIVERY / "nfl_per_side_oof_residuals_3e8c8a510f04.csv"


def _frame_sha() -> str:
    return hashlib.sha256(DECIDED_FRAME.read_bytes()).hexdigest()[:12]


def _climatological_pmf(scores: np.ndarray) -> np.ndarray:
    """Same-sample empirical marginal PMF on the grid — the climatological
    forecast for G1 (integer-grid analog of sigma/sqrt(pi))."""
    v = np.clip(np.asarray(scores, dtype=float), 0, 75)
    pmf = np.bincount(v.astype(int), minlength=76).astype(float)
    return pmf / pmf.sum()


def _side_model_pmfs(sub: pd.DataFrame, side: str,
                     params: dict[str, Any]) -> list[np.ndarray]:
    """Per-game marginal PMFs under the chosen family + sigma curve."""
    col = "pred_home" if side == "home" else "pred_away"
    sig_fn = sigma_callable(params["sigma_h" if side == "home" else "sigma_a"])
    return [marginal_pmf(float(m), sig_fn(float(m)), params["family"])
            for m in sub[col].to_numpy(float)]


def _crps_vs_climatology(pred_frames: dict[str, pd.DataFrame],
                         params: dict[str, Any]) -> dict[str, Any]:
    """Per-side distributional CRPS (winning marginal PMFs) vs the same-sample
    climatological empirical marginal, per split (pooled gate / sealed check)."""
    out: dict[str, Any] = {}
    for split, sub in pred_frames.items():
        s: dict[str, Any] = {"n": int(len(sub))}
        for side, col in (("home", "home_score"), ("away", "away_score")):
            y = sub[col].to_numpy(float)
            clim_pmf = _climatological_pmf(y)
            clim = float(np.mean([crps_discrete(clim_pmf, a) for a in y]))
            model = float(np.mean(
                [crps_discrete(p, a) for p, a in
                 zip(_side_model_pmfs(sub, side, params), y)]))
            s[f"crps_{side}"] = round(model, 4)
            s[f"climatological_crps_{side}"] = round(clim, 4)
            s[f"ratio_{side}"] = round(model / clim, 4)
            s[f"improvement_pct_{side}"] = round((1 - model / clim) * 100, 2)
        out[split] = s
    return out


def _sealed_predictions(feats: pd.DataFrame, rounds: dict[str, int]
                        ) -> pd.DataFrame:
    """Fit-only refill: fit 2019-24 at the artifact's per-side median rounds,
    predict 2025 (mirror of the per-side runner's _sealed_eval)."""
    preq = feats[feats["season"].isin(TRAIN_SEASONS)].copy()
    preq_valid = preq[_valid_rows(preq, SIDE_FEATURES)].copy()
    sld = feats[feats["season"] == SEALED_SEASON].copy()
    sld_valid = sld[_valid_rows(sld, SIDE_FEATURES)].copy()
    refit = refit_per_side(preq_valid, sld_valid, rounds, SIDE_FEATURES,
                           family="lgb")
    return sld_valid.merge(refit, on="game_id", how="left")


def _reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Decile reliability table + ECE for a binary calibration check."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 20:
        return {"n": int(len(y)), "ece": None, "bins": []}
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0 + 1e-12
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        bins.append({
            "bin": f"[{lo:.2f},{hi:.2f})", "n": int(m.sum()),
            "pred_mean": round(float(p[m].mean()), 4),
            "actual_rate": round(float(y[m].mean()), 4),
        })
    ece = float(np.mean([abs(b["pred_mean"] - b["actual_rate"])
                         for b in bins]))
    return {"n": int(len(y)), "ece": round(ece, 4), "bins": bins}


def _seam_check(feats: pd.DataFrame, pooled_pmfs: np.ndarray,
                pooled_derived: pd.DataFrame) -> dict[str, Any]:
    """nflverse spread_line/total_line coverage + offered-line calibration.

    Coverage >= 90% → reliability tables for derived covers
    (margin > spread_line) and derived overs (total > total_line) at each
    game's ACTUAL offered line. Absence never blocks — the gap is recorded
    prominently.
    """
    out: dict[str, Any] = {"path": "nflreadpy.load_schedules", "ok": False}
    try:
        import nflreadpy
        sch = nflreadpy.load_schedules([2019, 2020, 2021, 2022, 2023, 2024,
                                        2025])
        if hasattr(sch, "to_pandas"):
            sch = sch.to_pandas()
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        out["note"] = ("schedule pull failed → integer-grid thresholds only; "
                       "seam calibration SKIPPED (gap recorded prominently)")
        return out

    ids = list(pooled_derived["game_id"])
    pmf_by_id = {gid: pm for gid, pm in zip(ids, pooled_pmfs)}
    margin_by_id = {gid: margin_pmf_from_joint(pmf_by_id[gid])
                    for gid in ids}
    total_by_id = {gid: total_pmf_from_joint(pmf_by_id[gid])
                   for gid in ids}

    m = feats[feats["game_id"].isin(set(ids))].copy()
    m = m.merge(sch[["game_id", "spread_line", "total_line"]], on="game_id",
                how="left")
    m["margin"] = m["home_score"] - m["away_score"]
    m["total"] = m["home_score"] + m["away_score"]
    cov_s = float(m["spread_line"].notna().mean())
    cov_t = float(m["total_line"].notna().mean())
    out["spread_line_coverage_pct"] = round(cov_s * 100, 1)
    out["total_line_coverage_pct"] = round(cov_t * 100, 1)
    out["sign_convention"] = (
        "positive spread_line = home favored "
        f"(corr(spread_line, margin) = {round(m['spread_line'].corr(m['margin']), 3)} "
        "on the same rows); home covers iff margin > spread_line; "
        "over iff total > total_line")

    if cov_s < 0.90 or cov_t < 0.90:
        out["ok"] = False
        out["note"] = (f"coverage below 90% (spread {cov_s*100:.1f}% / total "
                       f"{cov_t*100:.1f}%) → integer-grid thresholds only; "
                       "seam calibration SKIPPED (gap recorded prominently)")
        return out

    out["ok"] = True
    # Covers at each game's actual offered line.
    sub = m.dropna(subset=["spread_line"])
    p_cover = np.array([cover_prob(margin_by_id[gid], float(sl))
                        for gid, sl in zip(sub["game_id"], sub["spread_line"])])
    y_cover = (sub["margin"] > sub["spread_line"]).astype(float).to_numpy()
    out["covers"] = _reliability(y_cover, p_cover)
    out["covers"]["definition"] = ("derived P(margin > spread_line) vs actual "
                                   "margin > spread_line (home perspective)")
    out["covers"]["empirical_home_ats_rate"] = round(float(y_cover.mean()), 4)
    # Totals at each game's actual offered line.
    sub_t = m.dropna(subset=["total_line"])
    p_over = np.array([over_prob(total_by_id[gid], float(tl))
                       for gid, tl in zip(sub_t["game_id"], sub_t["total_line"])])
    y_over = (sub_t["total"] > sub_t["total_line"]).astype(float).to_numpy()
    out["totals"] = _reliability(y_over, p_over)
    out["totals"]["definition"] = ("derived P(total > total_line) vs actual "
                                   "total > total_line")
    out["totals"]["empirical_over_rate"] = round(float(y_over.mean()), 4)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=None,
                    help="pre-built features CSV (skips the nflreadpy pull)")
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON record")
    ap.add_argument("--skip-seam", action="store_true",
                    help="skip the nflreadpy seam check (integer thresholds only)")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    warnings.filterwarnings("ignore",
                            message="The argument 'eval_set' is deprecated")

    print("=" * 70)
    print("  NFL Per-Side Joint Layer — Step 1 (record-only)")
    print("=" * 70)
    t0 = time.time()
    frame_sha = _frame_sha()
    print(f"frame_sha256={frame_sha}")

    # ---- 1. Inputs: artifact (loud guard) + decided frame ----
    art = load_residual_artifact(ARTIFACT)
    feats = load_features(args.features)
    print(f"Artifact: {len(art)} pooled-OOF rows "
          f"(guard: cols/empty/dup/n all OK)")
    art = art.merge(feats[["game_id", "home_score", "away_score", "season"]],
                    on="game_id", how="left")
    if art["home_score"].isna().any():
        raise RuntimeError("artifact join left NaN actuals — "
                           "frame/artifact mismatch")

    # Sealed mu's: same fit-only refill at the artifact's median rounds.
    rounds = {"home": int(art["best_iter_home"].median()),
              "away": int(art["best_iter_away"].median())}
    sealed = _sealed_predictions(feats, rounds)
    print(f"Sealed {SEALED_SEASON}: {len(sealed)} games "
          f"(fit-only refill, rounds={rounds})")

    # ---- 2/3. Pooled-OOF params: family, sigma, rho ----
    params = fit_joint_params(art)
    print(f"Family: {params['family']} | sigma_h={params['sigma_h']['spec']} "
          f"sigma_a={params['sigma_a']['spec']} | rho={params['rho']} "
          f"CI={params['rho_ci']}")
    for k, v in params["ll_table"].items():
        print(f"  LL {k}: {v['ll_total']} (home {v['ll_home']} / away {v['ll_away']})")

    # ---- Empirical final-tie rate on pooled OOF rows (constant base) ----
    n_ties = int((art["home_score"] == art["away_score"]).sum())
    p_tie = n_ties / len(art)
    print(f"Final-tie base rate: {n_ties}/{len(art)} = {p_tie:.4f}")

    # ---- 4/5. Per-game joint PMFs + derived probabilities ----
    pooled_pmfs, pooled_sum = build_joint_pmfs(art, params, p_tie)
    pooled_derived = pooled_sum["derived"].copy()
    pooled_derived = pooled_derived.merge(
        art[["game_id", "home_score", "away_score"]], on="game_id", how="left")
    sealed_pmfs, sealed_sum = build_joint_pmfs(sealed, params, p_tie)
    sealed_derived = sealed_sum["derived"].copy()
    sealed_derived = sealed_derived.merge(
        sealed[["game_id", "home_score", "away_score"]], on="game_id",
        how="left")
    print(f"Joint PMFs: pooled n={len(pooled_pmfs)} sealed n={len(sealed_pmfs)} "
          f"| D_raw mean={pooled_sum['summary']['d_raw_mean']} "
          f"calibrated={pooled_sum['summary']['d_calibrated_mean']}")

    # ---- 6. Data-seam check (EARLY by design) ----
    seam = {} if args.skip_seam else _seam_check(feats, pooled_pmfs,
                                                 pooled_derived)
    if seam:
        print(f"Seam: spread cov={seam.get('spread_line_coverage_pct')}% "
              f"total cov={seam.get('total_line_coverage_pct')}% "
              f"ok={seam.get('ok')}")

    # ---- G1: distributional CRPS vs climatology ----
    crps = _crps_vs_climatology({"pooled": art, "sealed": sealed}, params)
    g1_pooled = all(
        crps["pooled"][f"ratio_{s}"] <= 0.95 for s in ("home", "away"))
    print(f"G1 pooled: home {crps['pooled']['crps_home']} vs clim "
          f"{crps['pooled']['climatological_crps_home']} "
          f"({crps['pooled']['improvement_pct_home']}%) | away "
          f"{crps['pooled']['crps_away']} vs "
          f"{crps['pooled']['climatological_crps_away']} "
          f"({crps['pooled']['improvement_pct_away']}%) → {g1_pooled}")

    # ---- G2: mean calibrated tie mass vs base rate ----
    d_cal = float(np.mean([np.trace(p) for p in pooled_pmfs]))
    g2_delta_pp = (d_cal - p_tie) * 100
    g2 = abs(g2_delta_pp) <= 0.2
    print(f"G2: calibrated tie mean {d_cal:.6f} vs base {p_tie:.6f} "
          f"(Δ {g2_delta_pp:+.4f}pp) → {g2}")

    # ---- G3: post-IPF marginals ----
    g3_err = pooled_sum["summary"]["max_marginal_err_post_ipf"]
    g3 = g3_err is not None and g3_err <= 1e-9
    print(f"G3: max post-IPF marginal err {g3_err} → {g3}")

    # ---- G4: derived-ML calibration (honest, not hard-gated) ----
    y_ml = (pooled_derived["home_score"] > pooled_derived["away_score"])
    ml_metrics = compute_metrics(y_ml.astype(float).to_numpy(),
                                 pooled_derived["derived_ml"].to_numpy(float))
    g4_flag = ml_metrics["logloss"] - C0_ANCHOR["logloss"] > G4_LL_FLAG
    print(f"G4: derived-ML {ml_metrics} vs C0 anchor {C0_ANCHOR} "
          f"(trails-by->0.02 flag: {g4_flag})")

    # ---- G5: determinism (two identical builds byte-identical) ----
    _pmfs2, sum2 = build_joint_pmfs(art, params, p_tie)
    det = (pooled_sum["derived"].to_csv(index=False)
           == sum2["derived"].to_csv(index=False))
    print(f"G5: two identical runs byte-identical → {det}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    record = {
        "record": "nfl_joint_layer_step1",
        "frame_sha": frame_sha,
        "frame_sha256": _frame_sha256(feats),
        "written_utc": pd.Timestamp.now("UTC").isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "scope": SCOPE_PIN,
        "geometry": {
            "seasons": sorted(feats["season"].unique().tolist()),
            "train_seasons": TRAIN_SEASONS,
            "val_seasons": VAL_SEASONS,
            "sealed_season": SEALED_SEASON,
            "n_folds": 88,
            "grid": f"integer scores 0..{params['grid_max']} "
                    "(upper tail absorbed)",
            "targets": "full-game final scores incl. OT (how markets settle)",
            "view": SIDE_FEATURES,
        },
        "input_artifact": {
            "path": str(ARTIFACT.relative_to(ROOT_DIR)),
            "n_rows": int(len(art)),
            "guard": "load_residual_artifact raises RuntimeError on missing "
                     "cols / empty / duplicate game_ids / wrong row count",
            "sealed": {"method": "fit-only refill at artifact median rounds",
                       "rounds": rounds, "n": int(len(sealed))},
        },
        "marginal_family": {
            "chosen": params["family"],
            "selection": "pooled-OOF integer log-likelihood; DN default "
                         "within 10 LL units (NB does not auto-win)",
            "ll_table": params["ll_table"],
        },
        "sigma_curve": {
            "home": params["sigma_h"],
            "away": params["sigma_a"],
            "note": "per-side power law sigma(mu)=sigma0*mu^q vs constant "
                    "RMSE, winner by per-side pooled-OOF integer LL under "
                    "the chosen family; clamp [1.0, 15.0]",
        },
        "rho": {"rho": params["rho"], "ci": params["rho_ci"],
                "n": params["rho_n"],
                "note": "global scalar on pooled-OOF standardized pairs "
                        "z=resid/sigma(mu); applied unchanged to sealed — "
                        "never refit on sealed"},
        "tie": {
            "p_tie_empirical": round(p_tie, 5),
            "n_ties_pooled": n_ties,
            "d_raw_mean": pooled_sum["summary"]["d_raw_mean"],
            "d_calibrated_mean": pooled_sum["summary"]["d_calibrated_mean"],
            "method": "IPF-calibrated final-tie diagonal (constant base; "
                      "covariate model data-limited at ~5-10 positives — not "
                      "attempted); excess mass shifts to near-diagonal cells",
        },
        "crps_vs_climatology": crps,
        "gates": {
            "g1": {"pass": bool(g1_pooled),
                   "rule": "per-side distributional CRPS >= 5% better than "
                           "same-sample climatological on pooled OOF "
                           "(sealed = external check)"},
            "g2": {"pass": bool(g2), "delta_pp": round(g2_delta_pp, 4),
                   "rule": "mean calibrated tie mass == empirical rate "
                           "within +-0.2pp"},
            "g3": {"pass": bool(g3), "max_err": g3_err,
                   "rule": "post-IPF row/col sums == marginals to 1e-9"},
            "g4": {"metrics": ml_metrics, "c0_anchor": C0_ANCHOR,
                   "flag": bool(g4_flag),
                   "flag_rule": "flagged (coherency note, not failure) if "
                                "derived-ML logloss trails C0 by > 0.02",
                   "note": "derived ML = P(H>A)/(1-P_tie), raw "
                           "(uncalibrated) vs the Platt-calibrated 12-pool "
                           "ensemble; comparison is framing, not a gate"},
            "g5": {"pass": bool(det),
                   "method": "two identical builds byte-identical (derived "
                             "tables CSV-equal)"},
        },
        "data_seam": seam if seam else {"skipped": True},
        "derived_reference": {
            "pooled_n": int(len(pooled_derived)),
            "mean_derived_ml": round(
                float(pooled_derived["derived_ml"].mean()), 4),
            "mean_p_home_win": round(
                float(pooled_derived["p_home_win"].mean()), 4),
            "mean_p_tie": round(float(pooled_derived["p_tie"].mean()), 5),
            "cover_thresholds": {f"-{L}": _pooled_cover(pooled_pmfs, L)
                                 for L in COVER_THRESHOLDS},
            "over_thresholds": {str(U): _pooled_over(pooled_pmfs, U)
                                for U in OVER_THRESHOLDS},
        },
        "sealed_external": {
            "n": int(len(sealed_derived)),
            "mean_derived_ml": round(
                float(sealed_derived["derived_ml"].mean()), 4),
            "d_raw_mean": sealed_sum["summary"]["d_raw_mean"],
            "d_calibrated_mean": sealed_sum["summary"]["d_calibrated_mean"],
            "crps": crps["sealed"],
        },
        "coverage": {
            "n_total": int(len(feats)),
            "n_pooled_oof_priced": int(len(pooled_derived)),
            "n_sealed_2025_priced": int(len(sealed_derived)),
            "n_uncovered_pre_sealed": int(
                (feats["season"] < SEALED_SEASON).sum() - len(pooled_derived)),
            "imputation": "uncovered = 2019-20 warmup + small playoff folds "
                          "(absent from the step-1 artifact → no joint PMF); "
                          "sealed 2025 = fit-only refill (100% priced)",
        },
        "feature_columns_untouched": True,
        "judgment_calls": {
            "family": "NB tested but not default; discrete-normal wins on "
                      "near-equal OOF LL for grid simplicity. Data decides.",
            "tie_diagonal": "calibrated statistically at the joint level "
                            "(IPF) rather than fixed via regulation + OT "
                            "sub-models — acceptable record-only milestone; "
                            "flagged as the known limitation",
            "rho": "one global scalar, not feature- or game-state-dependent",
            "weather": "intentionally absent; plausible channel is the "
                       "sigma(mu) curve (variance, not the moneyline mean); "
                       "a weather sigma-arm is a clean follow-on because the "
                       "sigma seam accepts covariates later",
        },
    }
    if not args.no_record:
        record_path = DATA_DELIVERY / f"nfl_joint_{frame_sha}.json"
        record_path.write_text(json.dumps(record, indent=2, default=str))
        print(f"\nRecord: {record_path.name}")
    else:
        print("\n[--no-record] record skipped")
    return 0


def _pooled_cover(pmfs: np.ndarray, L: float) -> float:
    """Mean P(margin > L) over the pooled calibrated joints."""
    return round(float(np.mean(
        [cover_prob(margin_pmf_from_joint(J), L) for J in pmfs])), 4)


def _pooled_over(pmfs: np.ndarray, U: float) -> float:
    return round(float(np.mean(
        [over_prob(total_pmf_from_joint(J), U) for J in pmfs])), 4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    raise SystemExit(main())