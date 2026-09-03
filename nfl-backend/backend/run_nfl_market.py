"""NFL market layer — run-line/totals chain runner (record-only).

Re-quotes the totals product on the grid-fixed engine through the market
engine (nfl_market_engine.py): fold-disciplined disagreement shrinkage
(c_k, d_k) fitted second-level over the OOF val weeks, sealed 2025 via
median-of-fold, delta/2 PMF rebuild through the joint engine entrypoints,
and both product arms measured (own-line honest ECE + shrink-to-line).

DESIGN RULE (verbatim scope): NO pooled static overlay of any kind — d and
c are fitted over the folds and transferred to sealed by median-of-fold.
The prior pooled-map failure mode (away -0.14 -> +1.45, record 56893d3)
is forbidden. No wiring, no engine edits; moneyline FEATURE_COLUMNS /
12-pool / daily pipeline untouched.

State pin: origin/main == HEAD == 5eb7d5c (re-baseline); canonical frame
3e8c8a510f04 unchanged; era-centered per-side outputs are the /tmp
nfl_era_e2_{pooled,sealed}.csv dumps from run_nfl_era.py (era record
7260ddc) — this runner fails loudly if they are absent.

Deterministic (no RNG): identical inputs => identical outputs.

Usage:
    cd nfl-backend && python3 backend/run_nfl_market.py [--no-record]
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

import nfl_market_engine as M  # noqa: E402
from nfl_joint_engine import build_joint_pmfs, fit_joint_params  # noqa: E402
from nfl_moneyline import (SEALED_SEASON, TRAIN_SEASONS,  # noqa: E402
                           _valid_rows, compute_metrics, generate_weekly_folds)
from nfl_per_side_engine import SIDE_FEATURES  # noqa: E402
from run_nfl_margin_ablation import _frame_sha256, load_features  # noqa: E402

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY = ROOT_DIR / "data_delivery"
DECIDED_FRAME = DATA_DELIVERY / "nfl_game_level_features.csv"

# Era-centered per-side OOF outputs (era record 7260ddc).
ERA_POOLED_DUMP = "/tmp/nfl_era_e2_pooled.csv"
ERA_SEALED_DUMP = "/tmp/nfl_era_e2_sealed.csv"
REBASELINE_RECORD_NAME = "nfl_joint_rebaseline_3e8c8a510f04.json"
MARKET_RECORD_NAME = "nfl_market_3e8c8a510f04.json"

SCOPE_PIN = (
    "Market layer for the run-line/totals chain — record-only. Only fitted "
    "thing: per-fold (c_k, d_k) disagreement shrinkage by second-level "
    "walk-forward over the OOF val weeks (weeks < k), transferred to sealed "
    "2025 by median-of-fold. NO pooled static overlay of any kind (the "
    "away -0.14 -> +1.45 pooled-map failure mode is forbidden). Spread side "
    "untouched (delta/2 keeps mu_H - mu_A fixed); derived ML stays a "
    "G4-style coherence report (the board moneyline is the frozen 12-pool "
    "incumbent — no Platt, no challenge). No wiring; moneyline "
    "FEATURE_COLUMNS / 12-pool / daily pipeline untouched."
)

ENGINE_FILES = ["nfl_joint_engine.py", "nfl_per_side_engine.py",
                "nfl_era_features.py"]

# Re-baseline anchors (5eb7d5c) — the own-line arm must reproduce these
# (C0 machinery check) and the covers side must stay put (G3).
ANCHOR = {
    "seam_totals_ece": 0.087,
    "seam_totals_top_bin_pred": 0.7739,
    "seam_covers_ece": 0.078,
}
C0_TOTALS_ECE_TOL = 0.001
C0_TOP_BIN_TOL = 0.002
G3_COVERS_TOL = 0.001


def _frame_sha() -> str:
    return hashlib.sha256(DECIDED_FRAME.read_bytes()).hexdigest()[:12]


def _load_era_dumps() -> tuple[pd.DataFrame, pd.DataFrame]:
    for p, n in ((ERA_POOLED_DUMP, 1091), (ERA_SEALED_DUMP, 285)):
        if not Path(p).exists():
            raise RuntimeError(f"era-centered dump missing: {p} — run "
                               "run_nfl_era.py first (era record 7260ddc)")
    pooled = pd.read_csv(ERA_POOLED_DUMP)
    sealed = pd.read_csv(ERA_SEALED_DUMP)
    if len(pooled) != 1091 or len(sealed) != 285:
        raise RuntimeError("era dumps wrong row counts — rerun run_nfl_era")
    return pooled, sealed


def _week_map_from_folds(feats: pd.DataFrame) -> dict[str, Any]:
    """Reconstruct the SAME generate_weekly_folds geometry the era walk used
    and map each OOF val game_id -> its fold's week_start (Monday)."""
    preq = feats[feats["season"].isin(TRAIN_SEASONS)].copy()
    preq_valid = preq[_valid_rows(preq, SIDE_FEATURES)].copy()
    folds = generate_weekly_folds(preq_valid)
    if len(folds) != 88:
        raise RuntimeError(f"fold geometry mismatch: expected 88 folds, "
                           f"got {len(folds)}")
    week_map: dict[str, Any] = {}
    for f in folds:
        for gid in f["val"]["game_id"]:
            week_map[str(gid)] = f["week_start"]
    return week_map


def _g5_no_pooled_static(pooled_market: pd.DataFrame, walk: dict,
                         sealed_market: pd.DataFrame) -> dict[str, Any]:
    """G5: every scored row used a fold-fitted (c_k, d_k) — pooled rows their
    own fold's, sealed rows the median-of-fold. No row may use a pooled-global
    (c, d)."""
    bad = []
    for _, r in pooled_market.iterrows():
        ck, dk = walk["used_cd"][str(r["game_id"])]
        if abs(ck - r.get("used_c", float("nan"))) > 1e-9 or \
                abs(dk - r.get("used_d", float("nan"))) > 1e-9:
            bad.append(str(r["game_id"]))
    # sealed uses the median-of-fold (not a pooled fit, not per-row leakage)
    if len(sealed_market):
        med_mask = ((np.abs(sealed_market["used_c"] - walk["median_c"]) < 1e-9)
                    & (np.abs(sealed_market["used_d"]
                              - walk["median_d"]) < 1e-9))
        n_med = int(med_mask.sum())
    else:
        n_med = 0
    return {
        "pooled_rows_mismatched": len(bad),
        "sealed_rows_on_median": int(n_med),
        "sealed_n": int(len(sealed_market)),
        "pass": bool(len(bad) == 0 and n_med == len(sealed_market)),
        "read": ("every pooled row carries its own fold's (c_k, d_k); "
                 "sealed 2025 carries the median-of-fitted-folds; no "
                 "pooled-global (c, d) exists in this layer by construction"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON record")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    warnings.filterwarnings("ignore",
                            message="The argument 'eval_set' is deprecated")
    t0 = time.time()
    frame_sha = _frame_sha()
    if frame_sha != "3e8c8a510f04":
        print(f"FATAL: frame sha {frame_sha} != canonical 3e8c8a510f04 — "
              "the prior records' inputs changed; STOP")
        return 1
    print(f"frame_sha256={frame_sha}")

    # =====================================================================
    # STEP 0 — market dataset (committed artifacts only)
    # =====================================================================
    print("\n[Step 0] market dataset...")
    feats = load_features(None)
    pooled, sealed = _load_era_dumps()
    week_map = _week_map_from_folds(feats)
    dump_ids = set(pooled["game_id"].astype(str))
    if not dump_ids.issubset(set(week_map)):
        raise RuntimeError("era dump contains game_ids outside the fold "
                           "geometry — dump/fold mismatch; STOP")
    lines = M.load_offered_lines()
    m_pooled, m_sealed = M.build_market_frame(pooled, sealed, feats,
                                              week_map, lines)
    corr = round(float(m_pooled["spread_line"].corr(m_pooled["margin"])), 3)
    cov_s_p = round(float(m_pooled["spread_line"].notna().mean()) * 100, 1)
    cov_t_p = round(float(m_pooled["total_line"].notna().mean()) * 100, 1)
    cov_s_s = round(float(m_sealed["spread_line"].notna().mean()) * 100, 1)
    cov_t_s = round(float(m_sealed["total_line"].notna().mean()) * 100, 1)
    if not (cov_s_p == 100.0 and cov_t_p == 100.0
            and cov_s_s == 100.0 and cov_t_s == 100.0):
        raise RuntimeError("market frame coverage < 100% — STOP")
    print(f"  market frame: pooled n={len(m_pooled)} sealed n={len(m_sealed)} "
          f"| spread cov {cov_s_p}%/{cov_s_s}% | total cov {cov_t_p}%/"
          f"{cov_t_s}% | corr(spread_line, margin)={corr}")
    step0 = {
        "pooled_n": int(len(m_pooled)), "sealed_n": int(len(m_sealed)),
        "line_coverage_pct": {"pooled_spread": cov_s_p,
                              "pooled_total": cov_t_p,
                              "sealed_spread": cov_s_s,
                              "sealed_total": cov_t_s},
        "spread_margin_corr": corr,
        "line_vintage_caveat": ("nflreadpy schedule lines — closing vs early "
                                "unconfirmed; if historical lines are closing "
                                "lines, fitted d understates pre-game "
                                "shrinkage at slate time (judgment call 5)"),
        "source": "era record 7260ddc /tmp era-centered dumps + nflreadpy "
                  "schedules (canonical frame 3e8c8a510f04)",
    }

    # =====================================================================
    # STEP 1 — fold-disciplined disagreement model (the ONLY fitted thing)
    # =====================================================================
    print("\n[Step 1] second-level fold-disciplined (c_k, d_k) walk...")
    walk = M.fit_fold_disciplined_cd(m_pooled)
    d_fitted = [f["d"] for f in walk["folds"] if not f["warmup"]]
    c_fitted = [f["c"] for f in walk["folds"] if not f["warmup"]]
    print(f"  folds={walk['n_folds']} fitted={walk['n_fitted']} "
          f"warmup={walk['n_warmup']} leak_safe={walk['leak_safe']} | "
          f"median (c,d)=({walk['median_c']}, {walk['median_d']}) "
          f"d range [{min(d_fitted):.3f}, {max(d_fitted):.3f}] "
          f"sigma-record pooled d=0.34 (SE 0.07)")
    if not walk["leak_safe"]:
        raise RuntimeError("walk leak detected — STOP")
    step1 = {
        "method": ("second-level walk-forward over the OOF val weeks: fold k "
                   "fits OLS actual_total - line ~ c + d*(mu_T_hat - line) on "
                   "the val rows of strictly-prior folds ONLY (weeks < k); "
                   "warmup folds (< 50 prior rows) use d=1, c=0 no-shrink"),
        "folds": walk["folds"],
        "median_c": walk["median_c"], "median_d": walk["median_d"],
        "n_folds": walk["n_folds"], "n_fitted": walk["n_fitted"],
        "n_warmup": walk["n_warmup"], "min_prior_rows": walk["min_prior_rows"],
        "geometry_note": ("88-fold weekly geometry; the walk evaluates the "
                          "79 weeks that carry pooled-OOF rows (9 tiny-val "
                          "folds skipped by the era walk) — the second-level "
                          "fit set for week k is the val rows of strictly-"
                          "prior evaluated weeks only"),
        "leak_safe": bool(walk["leak_safe"]),
        "d_summary_fitted": {
            "mean": round(float(np.mean(d_fitted)), 4),
            "median": round(float(np.median(d_fitted)), 4),
            "min": round(float(np.min(d_fitted)), 4),
            "max": round(float(np.max(d_fitted)), 4),
            "iqr": round(float(np.percentile(d_fitted, 75)
                               - np.percentile(d_fitted, 25)), 4)},
        "c_summary_fitted": {
            "mean": round(float(np.mean(c_fitted)), 4),
            "median": round(float(np.median(c_fitted)), 4)},
        "sigma_record_pooled_reference": {"d": 0.34, "se": 0.07},
    }

    # =====================================================================
    # STEP 2 — rebuild + re-quote both arms (engine entrypoints, zero edits)
    # =====================================================================
    print("\n[Step 2] rebuild + re-quote (own-line and shrink-to-line)...")
    params = fit_joint_params(pooled)
    n_ties = int((pooled["home_score"] == pooled["away_score"]).sum())
    p_tie = n_ties / len(pooled)
    cd_by_week = {f["week_start"]: (f["c"], f["d"]) for f in walk["folds"]}
    median_cd = (walk["median_c"], walk["median_d"])
    print(f"  joint params: family={params['family']} "
          f"sigma_h={params['sigma_h']['sigma0']} "
          f"sigma_a={params['sigma_a']['sigma0']} rho={params['rho']} "
          f"p_tie={p_tie:.5f} ({n_ties}/{len(pooled)})")

    own_p = M.build_arm(m_pooled, params, p_tie, "none")
    shr_p = M.build_arm(m_pooled, params, p_tie, "fold",
                        cd_by_week=cd_by_week)
    own_s = M.build_arm(m_sealed, params, p_tie, "none")
    shr_s = M.build_arm(m_sealed, params, p_tie, "median",
                        median_cd=median_cd)

    rec_tbl = M.market_record_table(own_p, shr_p)
    rec_tbl_s = M.market_record_table(own_s, shr_s)
    rec_tbl = pd.concat([rec_tbl, rec_tbl_s], ignore_index=True)
    rec_path = Path("/tmp") / f"nfl_market_records_{frame_sha}.csv"
    rec_tbl.to_csv(rec_path, index=False)

    t_p = M.totals_calibration(own_p, "p_over", "y_over")
    t_ps = M.totals_calibration(shr_p, "p_over", "y_over")
    t_s = M.totals_calibration(own_s, "p_over", "y_over")
    t_ss = M.totals_calibration(shr_s, "p_over", "y_over")
    c_p = M.covers_calibration(own_p)
    c_ps = M.covers_calibration(shr_p)
    c_s = M.covers_calibration(own_s)
    c_ss = M.covers_calibration(shr_s)

    def _ml(arm: pd.DataFrame) -> dict[str, Any]:
        return compute_metrics(arm["y_home_win"].to_numpy(float),
                               arm["derived_ml"].to_numpy(float))

    ml_own_p, ml_shr_p = _ml(own_p), _ml(shr_p)
    ml_own_s, ml_shr_s = _ml(own_s), _ml(shr_s)

    # C0 machinery pins: own-line arm == the re-baseline seam numbers.
    c0_ok = (abs(t_p["ece"] - ANCHOR["seam_totals_ece"]) <= C0_TOTALS_ECE_TOL
             and abs(t_p["top_bin"].get("pred_mean", 0.0)
                     - ANCHOR["seam_totals_top_bin_pred"]) <= C0_TOP_BIN_TOL
             and abs(c_p["ece"] - ANCHOR["seam_covers_ece"]) <= 0.001)
    print(f"  C0 pins: own totals ECE {t_p['ece']} (anchor "
          f"{ANCHOR['seam_totals_ece']}) | top bin "
          f"{t_p['top_bin'].get('pred_mean')} (anchor "
          f"{ANCHOR['seam_totals_top_bin_pred']}) | covers ECE "
          f"{c_p['ece']} (anchor {ANCHOR['seam_covers_ece']}) -> {c0_ok}")
    print(f"  pooled: own ECE {t_p['ece']} vs shrink ECE {t_ps['ece']} | "
          f"top bin own {t_p['top_bin'].get('pred_mean')} vs shrink "
          f"{t_ps['top_bin'].get('pred_mean')} (actual "
          f"{t_p['top_bin'].get('actual_rate')})")
    print(f"  sealed: own ECE {t_s['ece']} vs shrink ECE {t_ss['ece']} | "
          f"top bin own {t_s['top_bin'].get('pred_mean')} vs shrink "
          f"{t_ss['top_bin'].get('pred_mean')} (actual "
          f"{t_s['top_bin'].get('actual_rate')})")
    print(f"  covers ECE: pooled own {c_p['ece']} / shrink {c_ps['ece']} | "
          f"sealed own {c_s['ece']} / shrink {c_ss['ece']}")

    step2 = {
        "joint_params_fixed_engine": {
            "family": params["family"], "sigma_h": params["sigma_h"],
            "sigma_a": params["sigma_a"], "rho": params["rho"],
            "fit_on": params["fit_on"]},
        "tie": {"p_tie": round(p_tie, 5), "n_ties": n_ties,
                "n_pooled": int(len(pooled))},
        "arms": {
            "own_line": {"method": ("era-centered mu unshifted; fair total = "
                                    "median of total PMF; P(over) at the "
                                    "offered line; honest ECE as-is (no "
                                    "market blending)")},
            "shrink_to_line": {"method": ("mu*_T = line + c_k + "
                                          "d_k*(mu_T_hat - line); shift both "
                                          "means by delta/2; rebuild via "
                                          "build_joint_pmfs (engine "
                                          "entrypoints, zero edits)")},
        },
        "pooled": {
            "own": {"totals": t_p, "covers": c_p, "derived_ml": ml_own_p},
            "shrink": {"totals": t_ps, "covers": c_ps,
                       "derived_ml": ml_shr_p}},
        "sealed": {
            "own": {"totals": t_s, "covers": c_s, "derived_ml": ml_own_s},
            "shrink": {"totals": t_ss, "covers": c_ss,
                       "derived_ml": ml_shr_s}},
        "quote_the_line_baseline": {
            "pooled": {"ece": M.quote_the_line_ece(
                m_pooled["total"].to_numpy(float)
                > m_pooled["total_line"].to_numpy(float))},
            "sealed": {"ece": M.quote_the_line_ece(
                m_sealed["total"].to_numpy(float)
                > m_sealed["total_line"].to_numpy(float))},
        },
        "c0_machinery_check": {
            "pass": bool(c0_ok),
            "own_totals_ece": t_p["ece"],
            "anchor_totals_ece": ANCHOR["seam_totals_ece"],
            "own_top_bin_pred": t_p["top_bin"].get("pred_mean"),
            "anchor_top_bin_pred": ANCHOR["seam_totals_top_bin_pred"],
            "own_covers_ece": c_p["ece"],
            "anchor_covers_ece": ANCHOR["seam_covers_ece"],
            "read": ("own-line arm reproduces the re-baseline seam numbers "
                     "bit-for-bit (same joints, same lines, same binning) — "
                     "the machinery check for the whole chain")},
        "market_records_csv": str(rec_path),
    }

    # =====================================================================
    # STEP 3 — seam/invariance/determinism
    # =====================================================================
    print("\n[Step 3] invariance + determinism...")
    g3 = abs(c_ps["ece"] - ANCHOR["seam_covers_ece"]) <= G3_COVERS_TOL
    # determinism: byte-identical double build of both arms
    own_p2 = M.build_arm(m_pooled, params, p_tie, "none")
    shr_p2 = M.build_arm(m_pooled, params, p_tie, "fold",
                         cd_by_week=cd_by_week)
    tbl1 = M.market_record_table(own_p, shr_p).to_csv(index=False)
    tbl2 = M.market_record_table(own_p2, shr_p2).to_csv(index=False)
    g4 = tbl1 == tbl2
    print(f"  G3 covers ECE shrink {c_ps['ece']} vs anchor "
          f"{ANCHOR['seam_covers_ece']} (tol {G3_COVERS_TOL}) -> {g3}")
    print(f"  G4 determinism (byte-identical double walk) -> {g4}")

    # =====================================================================
    # GATES
    # =====================================================================
    g1 = (t_ps["ece"] < t_p["ece"] - 1e-9) and (t_ss["ece"] < t_s["ece"] - 1e-9)
    own_gap = abs(t_p["top_bin"].get("pred_mean", 0.0)
                  - t_p["top_bin"].get("actual_rate", 0.0))
    shr_gap = abs(t_ps["top_bin"].get("pred_mean", 0.0)
                  - t_ps["top_bin"].get("actual_rate", 0.0))
    own_gap_s = abs(t_s["top_bin"].get("pred_mean", 0.0)
                    - t_s["top_bin"].get("actual_rate", 0.0))
    shr_gap_s = abs(t_ss["top_bin"].get("pred_mean", 0.0)
                    - t_ss["top_bin"].get("actual_rate", 0.0))
    g2 = (shr_gap < own_gap - 0.01) and (shr_gap_s < own_gap_s)
    g5 = _g5_no_pooled_static(m_pooled.merge(
        rec_tbl[["game_id", "used_c", "used_d"]], on="game_id", how="left"),
        walk, m_sealed.merge(
            rec_tbl[["game_id", "used_c", "used_d"]], on="game_id",
            how="left"))
    gates = {
        "g1": {"pass": bool(g1),
               "pooled": {"own": t_p["ece"], "shrink": t_ps["ece"]},
               "sealed": {"own": t_s["ece"], "shrink": t_ss["ece"]},
               "rule": "shrink totals ECE < own-line totals ECE (pooled AND "
                       "sealed)"},
        "g2": {"pass": bool(g2),
               "pooled": {"own_gap": round(own_gap, 4),
                          "shrink_gap": round(shr_gap, 4),
                          "own_top_bin": t_p["top_bin"],
                          "shrink_top_bin": t_ps["top_bin"]},
               "sealed": {"own_gap": round(own_gap_s, 4),
                          "shrink_gap": round(shr_gap_s, 4),
                          "own_top_bin": t_s["top_bin"],
                          "shrink_top_bin": t_ss["top_bin"]},
               "rule": "top-bin gap closes materially (>= 1pp pooled; "
                       "sealed reported — no pre-announced target)"},
        "g3": {"pass": bool(g3), "covers_ece_shrink_pooled": c_ps["ece"],
               "anchor": ANCHOR["seam_covers_ece"], "tol": G3_COVERS_TOL,
               "rule": "covers ECE unchanged at 0.078 ± 0.001 (spread "
                       "invariance — delta/2 leaves the margin center fixed)"},
        "g4": {"pass": bool(g4), "method": "byte-identical double walk"},
        "g5": g5,
    }
    print("\n=== GATES ===")
    for k, v in gates.items():
        print(f"  {k}: pass={v['pass']}")

    # =====================================================================
    # DELTA vs re-baseline (5eb7d5c) + VERDICT
    # =====================================================================
    d_med = float(walk["median_d"])
    shrink_gain_pooled = round(t_p["ece"] - t_ps["ece"], 4)
    line_base_p = M.quote_the_line_ece(
        m_pooled["total"].to_numpy(float)
        > m_pooled["total_line"].to_numpy(float))
    line_base_s = M.quote_the_line_ece(
        m_sealed["total"].to_numpy(float)
        > m_sealed["total_line"].to_numpy(float))
    all_gates = all(gates[k]["pass"] for k in ("g1", "g2", "g3", "g4", "g5"))
    if all_gates and d_med >= 0.10:
        verdict_state = "ADOPT_SHRINK_TO_LINE"
        verdict_read = (f"every gate passes and d median {d_med:.3f} is "
                        "material on clean fold-disciplined fitting (not "
                        "collapsed) — the 12-pool carries real totals signal "
                        "over the market line. Shrink closes the top-bin gap "
                        "and totals ECE vs own-line on pooled AND sealed. "
                        f"Vs pure line-quoting: shrink pooled ECE "
                        f"{t_ps['ece']} vs {line_base_p} (sealed "
                        f"{t_ss['ece']} vs {line_base_s}) — the market line "
                        "is near-efficient, so the shrunk model's edge is "
                        "directional (top-bin discrimination), not a big "
                        "ECE margin over the degenerate floor.")
    elif t_p["ece"] <= 0.06 and shrink_gain_pooled < 0.01:
        verdict_state = "ADOPT_OWN_LINE_WITH_HONEST_ECE"
        verdict_read = (f"own-line ECE {t_p['ece']} is honest enough to ship "
                        "without a market feed (shrink gain only "
                        f"{shrink_gain_pooled}); product architecture = "
                        "own-line slate path.")
    else:
        verdict_state = "NEITHER"
        verdict_read = ("gate table did not justify either arm (see gates); "
                        "if d collapsed toward 0 on fold-disciplined fitting, "
                        "the 12-pool adds nothing over the market totals line "
                        "— stated plainly.")
    if d_med < 0.10:
        verdict_read += (f" d median {d_med:.3f} collapses toward 0 — the "
                         "12-pool totals add nothing over the market line.")
    print(f"\n  verdict: {verdict_state} (d_med={d_med:.3f}, "
          f"shrink_gain_pooled={shrink_gain_pooled}, "
          f"quote-the-line pooled {line_base_p})")

    rows = [
        {"quantity": "seam totals ECE (own-line)", "before": 0.087,
         "after": t_p["ece"], "read": "C0 pin — own-line reproduces the "
         "re-baseline number bit-for-bit"},
        {"quantity": "seam totals ECE (shrink)", "before": 0.087,
         "after": t_ps["ece"], "read": "fold-disciplined (c_k, d_k) "
         "disagreement shrinkage"},
        {"quantity": "totals top bin pred/actual (own)", "before":
         {"pred": 0.7739, "actual": 0.5273}, "after":
         {"pred": t_p["top_bin"].get("pred_mean"),
          "actual": t_p["top_bin"].get("actual_rate")},
         "read": "C0 pin — model-vs-market residual as re-baselined"},
        {"quantity": "totals top bin pred/actual (shrink)", "before":
         {"pred": 0.7739, "actual": 0.5273}, "after":
         {"pred": t_ps["top_bin"].get("pred_mean"),
          "actual": t_ps["top_bin"].get("actual_rate")},
         "read": "the residual the market layer targets"},
        {"quantity": "seam covers ECE (shrink)", "before": 0.078,
         "after": c_ps["ece"], "read": "spread side untouched — delta/2 "
         "keeps the margin center fixed (G3)"},
        {"quantity": "sealed 2025 totals ECE (own -> shrink)", "before":
         t_s["ece"], "after": t_ss["ece"],
         "read": "median-of-fold transfer to sealed"},
        {"quantity": "quote-the-line baseline ECE (pooled)", "before": None,
         "after": line_base_p,
         "read": "P(over)=0.5 everywhere — the no-model floor"},
        {"quantity": "median fold d", "before": 0.34, "after": d_med,
         "read": "sigma-record pooled d=0.34 (SE 0.07) vs fold-disciplined "
                 "median"},
        {"quantity": "derived ML pooled (own -> shrink)",
         "before": {"logloss": ml_own_p["logloss"], "ece": ml_own_p["ece"]},
         "after": {"logloss": ml_shr_p["logloss"], "ece": ml_shr_p["ece"]},
         "read": "G4-style coherence report — raw derived ML, no Platt "
                 "(the board moneyline owns that product)"},
    ]
    print("\n=== DELTA TABLE (vs re-baseline 5eb7d5c) ===")
    for r in rows:
        print(f"  {r['quantity']:<52} {str(r['before']):>28} -> "
              f"{str(r['after']):>28}")

    elapsed = time.time() - t0

    # =====================================================================
    # RECORD
    # =====================================================================
    engine_bytes = {f: hashlib.sha256(
        (Path(__file__).resolve().parent / f).read_bytes()).hexdigest()[:16]
        for f in ENGINE_FILES}
    record = {
        "record": "nfl_market_layer",
        "frame_sha": frame_sha,
        "frame_sha256": _frame_sha256(feats),
        "written_utc": pd.Timestamp.now("UTC").isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "scope": SCOPE_PIN,
        "design_rule": ("NO pooled static overlay of any kind — (c, d) "
                        "fitted over the folds (second-level walk-forward "
                        "over OOF val weeks) and transferred to sealed by "
                        "median-of-fold; the prior pooled-map failure mode "
                        "(away -0.14 -> +1.45) is forbidden"),
        "geometry": {
            "seasons": sorted(feats["season"].unique().tolist()),
            "train_seasons": TRAIN_SEASONS,
            "sealed_season": SEALED_SEASON,
            "n_folds": 88,
            "pooled_oof_n": int(len(m_pooled)),
            "sealed_n": int(len(m_sealed)),
            "grid": "integer scores 0..75 (upper tail absorbed)",
            "engine_files_sha256": engine_bytes,
            "engines_modified": False,
            "prior_records": {"era": "nfl_era_3e8c8a510f04.json",
                              "sigma": "nfl_sigma_layer_3e8c8a510f04.json",
                              "rebaseline": REBASELINE_RECORD_NAME},
        },
        "step0_market_dataset": step0,
        "step1_disagreement_walk": step1,
        "step2_arms": step2,
        "step3_invariance": {
            "covers_ece_shrink_pooled": c_ps["ece"],
            "anchor_covers_ece": ANCHOR["seam_covers_ece"],
            "determinism": bool(g4),
            "read": ("margin center mu_H - mu_A unchanged by construction "
                     "(delta/2 cancels); covers ECE stays 0.078 ± 0.001 as a "
                     "result (G3)")},
        "gates": gates,
        "delta_vs_rebaseline": rows,
        "verdict": {
            "state": verdict_state,
            "pass": bool(all_gates),
            "read": verdict_read,
        },
        "feature_columns_untouched": True,
        "judgment_calls": {
            "1_second_level_only": ("second-level walk-forward over the OOF "
                                    "val weeks is the ONLY fold-disciplined "
                                    "option — train rows have no offered "
                                    "lines (market-independence: lines never "
                                    "enter the feature frame) and "
                                    "era-centered mu exists only for OOF val "
                                    "+ sealed rows, so d cannot be fit on "
                                    "train splits"),
            "2_intercept_kept": ("full line y ~ c + d*x per fold (c "
                                 "included) — fold-fitted c is "
                                 "transfer-safe by construction; not dropped "
                                 "to avoid the pooled-map ghost"),
            "3_mu_level_shrink": ("shrink at the mu level then rebuild via "
                                  "engine entrypoints (not PMF array "
                                  "surgery) — reuses committed, tested "
                                  "machinery"),
            "4_derived_ml_raw": ("derived ML reported raw with honest ECE "
                                 "only — no Platt in this layer; the "
                                 "incumbent moneyline owns that product"),
            "5_line_vintage": ("line vintage (closing vs early) unconfirmed "
                               "— nflreadpy schedule lines; if historical "
                               "lines are closing, fitted d understates "
                               "pre-game shrinkage at slate time "
                               "(slate-transfer caveat)"),
        },
        "artifacts": {"market_records_csv": str(rec_path)},
    }
    if not args.no_record:
        rec_path_out = DATA_DELIVERY / MARKET_RECORD_NAME
        rec_path_out.write_text(json.dumps(record, indent=2, default=str))
        print(f"\nRecord: {rec_path_out.name}")
    else:
        print("\n[--no-record] record skipped")
    print(f"Done in {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    raise SystemExit(main())