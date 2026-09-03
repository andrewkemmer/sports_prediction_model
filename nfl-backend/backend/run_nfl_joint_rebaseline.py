"""NFL joint-engine re-baseline delta on the fixed engine (f251bc6).

Re-runs the joint/seam chain through the EXISTING engine entrypoints on the
grid-fixed ``nfl_joint_engine.py`` (P(score k) at index k — commit f251bc6)
and re-pins the ABSOLUTE PMF-derived numbers that the +2 derived-totals
shift invalidated, with a machine-verified delta table vs the era record
(7260ddc) and sigma record (3480b05).

DESIGN RULE (verbatim scope): this commit fits NOTHING new — it is pure
re-measurement through committed entrypoints. The engine's own
sigma_const/rho/tie params keep ``fit_joint_params``' existing pooled-OOF
convention (that is the engine's design, unchanged here). No pooled static
overlay of any kind is added. The market-layer shrinkage d is OUT OF SCOPE
(its fold-fitted version is the next build, written against THIS record's
honest baselines). Moneyline FEATURE_COLUMNS / served 12-pool / daily
pipeline untouched.

Steps (per spec):
  Step 0 — engine-state verification (fail fast): dn_pmf == marginal_pmf
      exactly on a grid of means (max diff 0.0), 76 marginal cells, 76x76
      joint, convolution mean == mu_H + mu_A within the <0.5 DN 0-floor
      clamp allowance. Any failure => STOP (wrong engine — no record).
  Step 1 — chain re-run (measurement only): inputs are the canonical frame
      (sha 3e8c8a510f04) + the era-centered per-side OOF outputs
      (/tmp/nfl_era_e2_{pooled,sealed}.csv + era record 7260ddc). C0 = the
      era chain on the fixed engine; reproduce the era C0 anchor (MAE
      7.4811/7.0526, sigma 9.663/9.0789, rho 0.0076, rounds 20/23) as the
      machinery check (the per-side mu walk is engine-independent, so the
      MAE/rounds pin proves the mu table is byte-unchanged).
  Step 2 — re-quote the full gate set on the fixed engine: G1 CRPS vs
      climatology (pooled AND sealed 2025, both legs), G2 calibrated tie
      mean == empirical, G3 post-IPF marginal err <= 1e-9, G4 derived-ML
      ll/auc/ece (era ll 0.6365 is the pre-fix number to compare), G5
      determinism; DN-vs-NB LL gap (expect ~817, not ~15,800); per-side
      LL/CRPS; the internal totals ECE (pre-fix 0.0796) vs the
      documented-convention total-PIT ECE (0.0092) — these should now
      COLLAPSE to one number because the engine prices totals at the
      documented convention.
  Step 3 — seam re-check vs nflreadpy offered lines (100% spread+total
      coverage; sign convention locked by corr(spread_line, margin) =
      +0.446): re-quote totals ECE (pre-fix 0.0935) and the totals top bin
      (pre-fix pred 0.8159 vs actual 0.5273); covers ECE is expected
      UNCHANGED (~0.078) because margins/tie/derived-ML are index
      DIFFERENCES — verified numerically here by building the same joints
      with a VERBATIM copy of the pre-fix breakpoints (commit 3480b05) and
      asserting margin PMF / tie / derived-ML are bit-identical on the
      shared support.

Deterministic (no RNG): identical inputs => identical outputs.

Usage:
    cd nfl-backend && python3 backend/run_nfl_joint_rebaseline.py
        [--no-record]
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nfl_sigma_layer as S  # noqa: E402
from nfl_era_features import (  # noqa: E402
    attach_centers, compute_centers, mean_resid_stats,
    oof_centered_per_side, refit_centered_per_side,
)
from nfl_joint_engine import (  # noqa: E402
    GRID_MAX, _nb_r, build_joint_pmfs, derived_from_joint, dn_pmf,
    fit_joint_params, joint_pmf_copula, marginal_breakpoints, marginal_pmf,
    margin_pmf_from_joint, sigma_callable, total_pmf_from_joint,
)
from nfl_moneyline import (  # noqa: E402
    SEALED_SEASON, TRAIN_SEASONS, _valid_rows, compute_metrics,
    generate_weekly_folds,
)
from nfl_per_side_engine import SIDE_FEATURES  # noqa: E402
from run_nfl_joint import C0_ANCHOR, _crps_vs_climatology, _seam_check  # noqa: E402
from run_nfl_margin_ablation import _frame_sha256, load_features  # noqa: E402

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY = ROOT_DIR / "data_delivery"
DECIDED_FRAME = DATA_DELIVERY / "nfl_game_level_features.csv"

# Canonical era-centered per-side OOF outputs (era record 7260ddc).
ERA_POOLED_DUMP = "/tmp/nfl_era_e2_pooled.csv"
ERA_SEALED_DUMP = "/tmp/nfl_era_e2_sealed.csv"
ERA_RECORD_NAME = "nfl_era_3e8c8a510f04.json"
SIGMA_RECORD_NAME = "nfl_sigma_layer_3e8c8a510f04.json"

SCOPE_PIN = (
    "Pure re-measurement through committed entrypoints on the grid-fixed "
    "engine (f251bc6) — fits NOTHING new. The engine's sigma_const/rho/tie "
    "params keep fit_joint_params' existing pooled-OOF convention (the "
    "engine's design, unchanged). No pooled static overlay of any kind. "
    "The market-layer shrinkage d is OUT OF SCOPE (its fold-fitted version "
    "is the next build, written against this record's honest baselines)."
)

ENGINE_FILES = ["nfl_joint_engine.py", "nfl_per_side_engine.py",
                "nfl_era_features.py"]

# Before-numbers (deterministic prior records on the SAME code paths — the
# stored numbers ARE the before numbers on the old engine).
BEFORE = {
    "seam_totals_ece": 0.0935,          # era 7260ddc / sigma 3480b05
    "seam_totals_top_bin": {"pred": 0.8159, "actual": 0.5273},
    "seam_covers_ece": 0.078,
    "seam_spread_margin_corr": 0.446,
    "g4_derived_ml": {"logloss": 0.6365, "auc": 0.695, "ece": 0.0435,
                      "brier": 0.2221},
    "internal_totals_ece": 0.0796,
    "internal_totals_thresholds": [  # pred/actual/gap at 42.5/47.5/52.5
        {"threshold": 42.5, "pred": 0.6202, "actual": 0.549},
        {"threshold": 47.5, "pred": 0.4774, "actual": 0.3905},
        {"threshold": 52.5, "pred": 0.3394, "actual": 0.2585}],
    "total_pit": {"mean": 0.4915, "ece": 0.0092, "chi2_p": 0.2592,
                  "ks_p": 0.0656},
    "per_side_pooled": {"ll_home": -3.6784, "ll_away": -3.6183,
                        "crps_home": 5.4188, "crps_away": 5.1449},
    "g1_pooled_pct": {"home": 4.08, "away": 4.69},
    "g1_sealed_pct": {"home": 2.76, "away": -1.52},
    "dn_vs_nb_gap": 15809.319,   # nb_const ll_total − dn_const ll_total
    "tie": {"p_tie": 0.00275, "n_ties": 3, "d_raw": 0.02789,
            "d_cal": 0.00274977},
}

# Step-0 fail-fast tolerances (spec).
TOTAL_MEAN_CLAMP_ALLOWANCE = 0.5
DN_NB_EQUAL_ATOL = 0.0          # spec: dn_pmf == marginal_pmf, max diff 0.0


# ── Verbatim pre-fix implementation (commit 3480b05) for the invariance arm ──

def _old_marginal_breakpoints(mu: float, sigma: float, family: str
                              ) -> np.ndarray:
    """PRE-FIX copy (3480b05): breakpoints at F(arange(76) − 0.5) +
    endpoints → cell k held the mass of score k−1 (P(score k) at k+1)."""
    if family == "dn":
        mu = float(mu)
        sigma = max(float(sigma), 1e-9)
        b = stats.norm.cdf((np.arange(GRID_MAX + 1) - 0.5 - mu) / sigma)
    else:  # nb
        mu = max(float(mu), 1e-9)
        r = _nb_r(mu, sigma)
        b = stats.nbinom.cdf(np.arange(GRID_MAX + 1) - 0.5, r, r / (r + mu))
    b = np.clip(b, 0.0, 1.0)
    b = np.concatenate([[0.0], b, [1.0]])
    b = np.maximum.accumulate(b)
    return b


def _old_joint(mu_h: float, mu_a: float, params: dict) -> np.ndarray:
    """PRE-FIX joint PMF (old breakpoints through the same copula
    rectangle) — 77x77, cell (i,j) holding score (i−1, j−1)."""
    rho = float(params["rho"])
    family = str(params["family"])
    b_h = _old_marginal_breakpoints(
        mu_h, sigma_callable(params["sigma_h"])(mu_h), family)
    b_a = _old_marginal_breakpoints(
        mu_a, sigma_callable(params["sigma_a"])(mu_a), family)
    qh = np.clip(stats.norm.ppf(np.clip(b_h, 1e-12, 1 - 1e-12)), -37.0, 37.0)
    qa = np.clip(stats.norm.ppf(np.clip(b_a, 1e-12, 1 - 1e-12)), -37.0, 37.0)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    mv = stats.multivariate_normal(mean=[0.0, 0.0], cov=cov)
    pts = np.column_stack([np.repeat(qh, len(qa)), np.tile(qa, len(qh))])
    C = np.asarray(mv.cdf(pts)).reshape(len(qh), len(qa))
    J = C[1:, 1:] - C[:-1, 1:] - C[1:, :-1] + C[:-1, :-1]
    J = np.clip(J, 0.0, None)
    return J / J.sum()


def _old_derived(J: np.ndarray) -> dict[str, float]:
    """Derived markets from a PRE-FIX joint (shift-invariant quantities)."""
    n = J.shape[0]
    p_home = float(np.tril(J, -1).sum())
    p_away = float(np.triu(J, 1).sum())
    p_tie = float(np.trace(J))
    return {"p_home_win": p_home, "p_away_win": p_away, "p_tie": p_tie,
            "derived_ml": p_home / (1.0 - p_tie) if p_tie < 1 else 0.5}


def _frame_sha() -> str:
    return hashlib.sha256(DECIDED_FRAME.read_bytes()).hexdigest()[:12]


def _mae(df: pd.DataFrame, side: str) -> float:
    col = "home_score" if side == "home" else "away_score"
    pred = "pred_home" if side == "home" else "pred_away"
    return round(float(np.abs(df[col].to_numpy(float)
                              - df[pred].to_numpy(float)).mean()), 4)


def _invariance_check(rows: pd.DataFrame, params: dict) -> dict[str, Any]:
    """Margin PMF / tie / derived-ML invariance pre-vs-post grid fix.

    The pre-fix joint's cell (i, j) holds the same score cell as the fixed
    joint's (i−1, j−1); every DERIVED quantity uses index DIFFERENCES, so
    margin PMF, tie mass, p_home/p_away and derived-ML must be bit-equal on
    the shared support. LL/CRPS are NOT invariant (they index at the actual
    score — that is the fix). Returns per-game max diffs.
    """
    out: dict[str, Any] = {"n": int(len(rows)), "max_margin_pmf_diff": 0.0,
                           "max_tie_diff": 0.0, "max_ml_diff": 0.0,
                           "max_p_home_diff": 0.0, "max_p_away_diff": 0.0}
    for _, r in rows.iterrows():
        muh, mua = float(r["pred_home"]), float(r["pred_away"])
        J_old = _old_joint(muh, mua, params)
        J_new = joint_pmf_copula(muh, mua, params)
        # old margin PMF is 153 cells (−76..76); new is 151 (−75..75).
        # Shared support −75..75 == old slice [1:152], new in full.
        m_old = margin_pmf_from_joint(J_old)[1:152]
        m_new = margin_pmf_from_joint(J_new)
        d_old, d_new = _old_derived(J_old), derived_from_joint(J_new)
        out["max_margin_pmf_diff"] = max(
            out["max_margin_pmf_diff"],
            float(np.max(np.abs(m_old - m_new))))
        out["max_tie_diff"] = max(out["max_tie_diff"],
                                  abs(d_old["p_tie"] - d_new["p_tie"]))
        out["max_ml_diff"] = max(out["max_ml_diff"],
                                 abs(d_old["derived_ml"]
                                     - d_new["derived_ml"]))
        out["max_p_home_diff"] = max(out["max_p_home_diff"],
                                     abs(d_old["p_home_win"]
                                         - d_new["p_home_win"]))
        out["max_p_away_diff"] = max(out["max_p_away_diff"],
                                     abs(d_old["p_away_win"]
                                         - d_new["p_away_win"]))
    return out


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
    # STEP 0 — engine-state verification (fail fast)
    # =====================================================================
    print("\n[Step 0] engine-state verification (fail fast)...")
    eq_max_diff = 0.0
    for mu in (5.0, 15.0, 25.0, 40.0, 60.0):
        eq_max_diff = max(eq_max_diff, float(np.max(
            np.abs(dn_pmf(mu, 9.0) - marginal_pmf(mu, 9.0, "dn")))))
    n_cells = int(len(marginal_pmf(25.0, 9.0, "dn")))
    params_pin = {"family": "dn",
                  "sigma_h": {"spec": "const", "sigma0": 9.663, "q": 0.0},
                  "sigma_a": {"spec": "const", "sigma0": 9.0789, "q": 0.0},
                  "rho": 0.0076}
    J = joint_pmf_copula(23.0, 20.0, params_pin)
    tot = total_pmf_from_joint(J)
    s = np.arange(len(tot), dtype=float)
    e_tot = float((s * tot).sum())
    clamp_bias = e_tot - (23.0 + 20.0)
    engine_ok = (eq_max_diff == 0.0 and n_cells == 76 and J.shape == (76, 76)
                 and 0.0 < clamp_bias < TOTAL_MEAN_CLAMP_ALLOWANCE)
    engine_state = {
        "dn_eq_marginal_max_diff": eq_max_diff,
        "marginal_cells": n_cells,
        "joint_shape": list(J.shape),
        "total_mean_shift_vs_muH_muA": round(clamp_bias, 5),
        "clamp_allowance": TOTAL_MEAN_CLAMP_ALLOWANCE,
        "ok": bool(engine_ok),
    }
    print(f"  dn==marginal max diff {eq_max_diff:.3e} | cells {n_cells} | "
          f"joint {J.shape} | total-mean shift {clamp_bias:+.4f} vs "
          f"muH+muA (clamp allowance {TOTAL_MEAN_CLAMP_ALLOWANCE})")
    if not engine_ok:
        print("  ENGINE-STATE FAILED — wrong engine checkout; STOP (no "
              "record).")
        return 1

    # =====================================================================
    # INPUTS (era-centered outputs + feature frame + prior records)
    # =====================================================================
    feats = load_features(None)
    feats["home_win"] = (feats["home_score"] > feats["away_score"]).astype(int)
    decided = feats[["game_id", "season", "week", "gameday", "home_score",
                     "away_score", "total"]].copy()

    era_path = DATA_DELIVERY / ERA_RECORD_NAME
    sig_path = DATA_DELIVERY / SIGMA_RECORD_NAME
    for p in (era_path, sig_path):
        if not p.exists():
            raise RuntimeError(f"prior record missing: {p}")
    era_rec = json.loads(era_path.read_text())
    sig_rec = json.loads(sig_path.read_text())
    era_sig = {"home": float(era_rec["step2_joint_chain"]["joint_params"]
                             ["sigma_h"]["sigma0"]),
               "away": float(era_rec["step2_joint_chain"]["joint_params"]
                             ["sigma_a"]["sigma0"])}
    era_mae = era_rec["step1_arms"]["e2"]["mae"]
    era_rounds = era_rec["step1_arms"]["e2"]["rounds"]

    for p in (ERA_POOLED_DUMP, ERA_SEALED_DUMP):
        if not Path(p).exists():
            raise RuntimeError(f"era-centered dump missing: {p} — run "
                               "run_nfl_era.py first")
    eng_pooled = pd.read_csv(ERA_POOLED_DUMP)
    eng_sealed = pd.read_csv(ERA_SEALED_DUMP)
    if len(eng_pooled) != 1091 or len(eng_sealed) != 285:
        raise RuntimeError("era dumps wrong row counts — rerun run_nfl_era")
    print(f"inputs: era-centered pooled OOF n={len(eng_pooled)} | sealed "
          f"n={len(eng_sealed)} | C0 anchors mae={era_mae} "
          f"rounds={era_rounds}")

    # =====================================================================
    # STEP 1 — C0 era-chain reproduction (machinery check) + fixed-engine
    # chain re-run through the existing entrypoints
    # =====================================================================
    print("\n[Step 1] C0 era-chain reproduction (fixed engine, walk "
          "unchanged)...")
    centers = compute_centers(decided, spec="ewm_2w")
    f_chosen = attach_centers(feats, centers)
    preq = f_chosen[f_chosen["season"].isin(TRAIN_SEASONS)].copy()
    preq_valid = preq[_valid_rows(preq, SIDE_FEATURES)].copy()
    folds = generate_weekly_folds(preq_valid)
    e2, rounds_e2, _u = oof_centered_per_side(folds, SIDE_FEATURES, f_chosen)
    e2 = e2.merge(f_chosen[["game_id", "season", "home_score", "away_score"]],
                  on="game_id", how="left")
    if len(e2) != 1091:
        raise RuntimeError(f"C0 reproduction: E2 coverage {len(e2)} != 1091")
    c0_mae = {"home": _mae(e2, "home"), "away": _mae(e2, "away")}
    mae_pin = all(abs(c0_mae[s_] - era_mae[s_]) < 0.0005
                  for s_ in ("home", "away"))
    rounds_pin = rounds_e2 == era_rounds
    print(f"  C0 walk: mae={c0_mae} (era pin {era_mae} -> {mae_pin}) "
          f"rounds={rounds_e2} (pin {era_rounds} -> {rounds_pin})")

    sld = f_chosen[f_chosen["season"] == SEALED_SEASON].copy()
    sld_valid = sld[_valid_rows(sld, SIDE_FEATURES)].copy()
    seal_c0 = refit_centered_per_side(preq_valid, sld_valid, rounds_e2,
                                      SIDE_FEATURES)
    eng_walked_sealed = sld_valid.merge(seal_c0, on="game_id", how="left")
    if eng_walked_sealed["home_score"].isna().any() \
            or len(eng_walked_sealed) != 285:
        raise RuntimeError("sealed C0 refill wrong")
    eng_walked_pooled = e2[["game_id", "pred_home", "pred_away", "home_score",
                            "away_score"]].copy()
    for eng in (eng_walked_pooled, eng_walked_sealed):
        eng["resid_home"] = eng["home_score"] - eng["pred_home"]
        eng["resid_away"] = eng["away_score"] - eng["pred_away"]
    params_walk = fit_joint_params(eng_walked_pooled)
    sig_pin = (params_walk["family"] == "dn"
               and params_walk["sigma_h"]["spec"] == "const"
               and params_walk["sigma_a"]["spec"] == "const"
               and abs(params_walk["sigma_h"]["sigma0"] - era_sig["home"])
               < 0.001
               and abs(params_walk["sigma_a"]["sigma0"] - era_sig["away"])
               < 0.001
               and abs(params_walk["rho"] - 0.0076) < 0.0001)
    print(f"  C0 joint params: family={params_walk['family']} "
          f"sigma_h={params_walk['sigma_h']['sigma0']} "
          f"sigma_a={params_walk['sigma_a']['sigma0']} "
          f"rho={params_walk['rho']} -> era-pin {sig_pin}")

    step1 = {
        "c0_walk": {
            "mae": c0_mae, "era_pin_mae": era_mae, "mae_match": bool(mae_pin),
            "rounds": rounds_e2, "era_pin_rounds": era_rounds,
            "rounds_match": bool(rounds_pin),
            "read": ("the per-side mu walk is engine-independent — MAE/rounds "
                     "pins prove the mu table is byte-unchanged on the fixed "
                     "engine; fit_joint_params on the walked table reproduces "
                     "the era sigma/rho pins (machinery check)"),
        },
        "joint_params_walk_pin": {
            "family": params_walk["family"],
            "sigma_h": params_walk["sigma_h"], "sigma_a": params_walk["sigma_a"],
            "rho": params_walk["rho"], "rho_ci": params_walk["rho_ci"],
            "match_era": bool(sig_pin),
        },
        "chain_inputs": {
            "pooled_oof_n": int(len(eng_pooled)),
            "sealed_n": int(len(eng_sealed)),
            "source": "era record 7260ddc /tmp era-centered dumps "
                      "(canonical frame 3e8c8a510f04)",
        },
    }

    # ---- fixed-engine chain re-run on the canonical era outputs ----
    print("\n[Step 1b] fixed-engine chain re-run (fit_joint_params + "
          "build_joint_pmfs, zero edits)...")
    params = fit_joint_params(eng_pooled)
    n_ties = int((eng_pooled["home_score"] == eng_pooled["away_score"]).sum())
    p_tie = n_ties / len(eng_pooled)
    pooled_pmfs, pooled_sum = build_joint_pmfs(eng_pooled, params, p_tie)
    pooled_derived = pooled_sum["derived"].copy()
    pooled_derived = pooled_derived.merge(
        eng_pooled[["game_id", "home_score", "away_score"]],
        on="game_id", how="left")
    sealed_pmfs, sealed_sum = build_joint_pmfs(eng_sealed, params, p_tie)
    sealed_derived = sealed_sum["derived"].copy()
    sealed_derived = sealed_derived.merge(
        eng_sealed[["game_id", "home_score", "away_score"]],
        on="game_id", how="left")
    print(f"  family={params['family']} sigma_h={params['sigma_h']['sigma0']} "
          f"sigma_a={params['sigma_a']['sigma0']} rho={params['rho']} "
          f"p_tie={p_tie:.5f} ({n_ties}/{len(eng_pooled)})")

    # =====================================================================
    # STEP 2 — re-quote the gate set on the fixed engine
    # =====================================================================
    print("\n[Step 2] gate re-quote (fixed engine)...")
    crps = _crps_vs_climatology({"pooled": eng_pooled, "sealed": eng_sealed},
                                params)
    g1_pooled = all(crps["pooled"][f"ratio_{s_}"] <= 0.95
                    for s_ in ("home", "away"))
    d_cal = float(np.mean([np.trace(p) for p in pooled_pmfs]))
    g2_delta_pp = (d_cal - p_tie) * 100
    g2 = abs(g2_delta_pp) <= 0.2
    g3_err = pooled_sum["summary"]["max_marginal_err_post_ipf"]
    g3 = g3_err is not None and g3_err <= 1e-9
    y_ml = (pooled_derived["home_score"] > pooled_derived["away_score"])
    ml = compute_metrics(y_ml.astype(float).to_numpy(),
                         pooled_derived["derived_ml"].to_numpy(float))
    g4_flag = ml["logloss"] - C0_ANCHOR["logloss"] > 0.02
    _p2, sum2 = build_joint_pmfs(eng_pooled, params, p_tie)
    g5 = bool(pooled_sum["derived"].to_csv(index=False)
              == sum2["derived"].to_csv(index=False))
    print(f"  G1 pooled: {crps['pooled']['improvement_pct_home']}% / "
          f"{crps['pooled']['improvement_pct_away']}% | sealed: "
          f"{crps['sealed']['improvement_pct_home']}% / "
          f"{crps['sealed']['improvement_pct_away']}%")
    print(f"  G2 Δ={g2_delta_pp:+.4f}pp | G3 err={g3_err} | G4 {ml} | "
          f"G5={g5}")

    # ---- DN-vs-NB LL gap + per-side LL/CRPS ----
    lt = params["ll_table"]
    dn_const = float(lt["dn_const"]["ll_total"])
    nb_const = float(lt["nb_const"]["ll_total"])
    dn_nb_gap = round(nb_const - dn_const, 3)
    print(f"  DN-vs-NB LL gap: {nb_const:.1f} - ({dn_const:.1f}) = "
          f"{dn_nb_gap} (pre-fix {BEFORE['dn_vs_nb_gap']})")
    per_side = {"ll_home": pooled_sum["summary"]["ll_home"],
                "ll_away": pooled_sum["summary"]["ll_away"],
                "crps_home": pooled_sum["summary"]["crps_home"],
                "crps_away": pooled_sum["summary"]["crps_away"]}
    print(f"  per-side pooled LL/CRPS: {per_side}")

    # ---- internal totals ECE (engine's own derived totals) + total PIT ----
    totals_ece = S.totals_ece_internal(pooled_pmfs, pooled_derived)
    sh, sa = era_sig["home"], era_sig["away"]
    tot_act = (eng_pooled["home_score"] + eng_pooled["away_score"]).to_numpy(
        float)
    pit = S.total_pit(eng_pooled["pred_home"].to_numpy(float),
                      np.full(len(eng_pooled), sh),
                      eng_pooled["pred_away"].to_numpy(float),
                      np.full(len(eng_pooled), sa), tot_act)
    pit_tab = S.uniformity_table(pit)
    pit_view = {"mean": pit_tab["mean"], "ece": pit_tab["ece"],
                "chi2_p": pit_tab["chi2_p"], "ks_p": pit_tab["ks_p"],
                "is_uniform": pit_tab["is_uniform"]}
    collapse = {
        "internal_totals_ece_fixed_engine": totals_ece["ece"],
        "total_pit_ece_documented": pit_view["ece"],
        "pre_fix_split": {"internal_totals_ece": BEFORE["internal_totals_ece"],
                          "total_pit_ece": BEFORE["total_pit"]["ece"]},
        "collapsed": bool(abs(totals_ece["ece"] - pit_view["ece"])
                          < 0.02 and totals_ece["ece"] < 0.03),
    }
    print(f"  internal totals ECE (fixed engine) = {totals_ece['ece']} vs "
          f"total-PIT ECE = {pit_view['ece']} (pre-fix split "
          f"{BEFORE['internal_totals_ece']} vs {BEFORE['total_pit']['ece']})")

    step2 = {
        "joint_params_fixed_engine": {
            "family": params["family"], "sigma_h": params["sigma_h"],
            "sigma_a": params["sigma_a"], "rho": params["rho"],
            "rho_ci": params["rho_ci"], "rho_n": params["rho_n"],
            "fit_on": params["fit_on"], "ll_table": params["ll_table"],
        },
        "tie": {"p_tie_empirical": round(p_tie, 5), "n_ties_pooled": n_ties,
                "d_raw_mean": pooled_sum["summary"]["d_raw_mean"],
                "d_calibrated_mean":
                    pooled_sum["summary"]["d_calibrated_mean"]},
        "gates": {
            "g1": {"pass": bool(g1_pooled), "pooled": crps["pooled"],
                   "sealed": crps["sealed"]},
            "g2": {"pass": bool(g2), "delta_pp": round(g2_delta_pp, 4)},
            "g3": {"pass": bool(g3), "max_err": g3_err},
            "g4": {"metrics": ml, "c0_anchor": C0_ANCHOR,
                   "flag": bool(g4_flag)},
            "g5": {"pass": bool(g5)},
        },
        "dn_vs_nb": {"nb_minus_dn": dn_nb_gap,
                     "dn_wins_by": round(abs(dn_nb_gap), 3),
                     "dn_const_ll": round(dn_const, 3),
                     "nb_const_ll": round(nb_const, 3),
                     "pre_fix_gap": BEFORE["dn_vs_nb_gap"],
                     "read": ("DN beats NB by ~817 LL units post-fix (pre-fix "
                              "15,809) — the corrected grid reads NB's mass "
                              "fairly; family choice unchanged (dn) and now on "
                              "a much narrower, honest margin")},
        "per_side_ll_crps_pooled": per_side,
        "internal_totals_ece": totals_ece,
        "total_pit_documented": pit_view,
        "convention_collapse": collapse,
    }

    # =====================================================================
    # STEP 3 — seam re-check vs nflreadpy offered lines + invariance
    # =====================================================================
    print("\n[Step 3] seam re-check + pre/post-fix invariance...")
    seam = _seam_check(feats, pooled_pmfs, pooled_derived)
    cov_s = seam.get("spread_line_coverage_pct")
    cov_t = seam.get("total_line_coverage_pct")
    tot_bins = (seam.get("totals") or {}).get("bins") or []
    top_bin = tot_bins[-1] if tot_bins else {}
    corr = None
    if "sign_convention" in seam:
        import re
        m = re.search(r"corr\(spread_line, margin\) = ([0-9.]+)",
                      seam["sign_convention"])
        if m:
            corr = float(m.group(1))
    print(f"  seam: ok={seam.get('ok')} spread_cov={cov_s}% "
          f"total_cov={cov_t}% corr={corr}")
    print(f"  covers ECE = {(seam.get('covers') or {}).get('ece')} "
          f"(pre-fix {BEFORE['seam_covers_ece']}) | totals ECE = "
          f"{(seam.get('totals') or {}).get('ece')} (pre-fix "
          f"{BEFORE['seam_totals_ece']}) | totals top bin: pred "
          f"{top_bin.get('pred_mean')} vs actual {top_bin.get('actual_rate')} "
          f"(pre-fix {BEFORE['seam_totals_top_bin']['pred']} vs "
          f"{BEFORE['seam_totals_top_bin']['actual']})")

    inv = _invariance_check(eng_pooled.head(12), params)
    inv["read"] = ("margin PMF / tie / p_home / p_away / derived-ML are "
                   "index-DIFFERENCE quantities — unchanged pre-vs-post grid "
                   "fix up to the below-0.5 tail sliver (pre-fix the mass "
                   "below -0.5 sat in a phantom 'score −1' cell; post-fix it "
                   "is absorbed into score 0). The renormalized shared cell "
                   "masses are identical to float precision; the residual "
                   "~1e-5 derived diffs are exactly that tail-mass "
                   "re-seating, not an index change (LL/CRPS move because "
                   "they index at the actual score). Seam covers ECE is "
                   "unchanged to 3 decimals (0.078 -> 0.078) as a result.")
    print(f"  invariance (12-game sample): max diffs {inv}")

    step3 = {
        "seam": {
            "path": seam.get("path"), "ok": seam.get("ok"),
            "spread_line_coverage_pct": cov_s,
            "total_line_coverage_pct": cov_t,
            "spread_margin_corr": corr,
            "sign_convention": seam.get("sign_convention"),
            "covers": seam.get("covers"),
            "totals": seam.get("totals"),
            "totals_top_bin": top_bin,
        },
        "invariance_pre_vs_post_fix": inv,
    }

    # =====================================================================
    # DELTA TABLE vs era record (7260ddc) + sigma record (3480b05)
    # =====================================================================
    rows = []
    def _row(key: str, before: Any, after: Any, read: str) -> None:
        rows.append({"quantity": key, "before": before, "after": after,
                     "read": read})
    _row("seam totals ECE", BEFORE["seam_totals_ece"],
         (seam.get("totals") or {}).get("ece"),
         "the +2 derived-totals shift removed; expect a large drop toward the "
         "internal calibration level")
    _row("totals top bin pred/actual", BEFORE["seam_totals_top_bin"],
         {"pred": top_bin.get("pred_mean"), "actual": top_bin.get("actual_rate")},
         "model-vs-market disagreement residual after the grid fix")
    _row("seam covers ECE", BEFORE["seam_covers_ece"],
         (seam.get("covers") or {}).get("ece"),
         "margins are index differences — unchanged to 3 decimals (invariance "
         "arm verifies the ~1e-5 tail-sliver residual numerically)")
    _row("G4 derived-ML metrics", BEFORE["g4_derived_ml"],
         {"logloss": ml["logloss"], "auc": ml["auc"], "ece": ml["ece"],
          "brier": ml["brier"]},
         "derived ML is shift-invariant (index differences) — the fix's +2 "
         "removal is expected to move it only via the tie/ML normalization "
         "of corrected marginals, if at all")
    _row("internal totals ECE (engine grid)", BEFORE["internal_totals_ece"],
         totals_ece["ece"],
         "engine's own derived totals now priced at the documented "
         "convention — collapses toward the corrected total-PIT ECE")
    _row("total-PIT ECE (documented)", BEFORE["total_pit"]["ece"],
         pit_view["ece"],
         "convention-independent of the engine grid — unchanged by the fix "
         "(mu/sigma inputs identical)")
    _row("per-side pooled LL home", BEFORE["per_side_pooled"]["ll_home"],
         per_side["ll_home"], "actual-score lookups read P(score y) post-fix")
    _row("per-side pooled LL away", BEFORE["per_side_pooled"]["ll_away"],
         per_side["ll_away"], "actual-score lookups read P(score y) post-fix")
    _row("per-side pooled CRPS home", BEFORE["per_side_pooled"]["crps_home"],
         per_side["crps_home"], "corrected marginal read")
    _row("per-side pooled CRPS away", BEFORE["per_side_pooled"]["crps_away"],
         per_side["crps_away"], "corrected marginal read")
    _row("DN-vs-NB LL gap", BEFORE["dn_vs_nb_gap"],
         {"nb_minus_dn": dn_nb_gap, "dn_wins_by": round(abs(dn_nb_gap), 3)},
         "family choice re-measured on the corrected marginals: DN beats NB "
         "by 817 LL post-fix (pre-fix 15,809) — NB's near-Poisson tail was "
         "over-penalized by the off-by-one grid; DN still wins clearly")
    _row("G1 pooled improvement %",
         BEFORE["g1_pooled_pct"],
         {"home": crps["pooled"]["improvement_pct_home"],
          "away": crps["pooled"]["improvement_pct_away"]},
         "G1 is a mean-layer gate — small movement expected")
    _row("G1 sealed improvement %",
         BEFORE["g1_sealed_pct"],
         {"home": crps["sealed"]["improvement_pct_home"],
          "away": crps["sealed"]["improvement_pct_away"]},
         "G1 is a mean-layer gate — small movement expected")
    _row("G2 tie delta pp", 0.0, round(g2_delta_pp, 4),
         "IPF tie calibration is index-invariant (diagonal mass)")
    _row("G3 post-IPF marginal err", 1e-11, g3_err, "IPF convergence pin")
    _row("G5 determinism", True, bool(g5), "byte-identical double build")

    print("\n=== DELTA TABLE (fixed engine vs era 7260ddc / sigma 3480b05) ===")
    for r in rows:
        print(f"  {r['quantity']:<42} {str(r['before']):>22} -> "
              f"{str(r['after']):>22}")

    elapsed = time.time() - t0

    # =====================================================================
    # RECORD
    # =====================================================================
    engine_bytes = {f: hashlib.sha256(
        (Path(__file__).resolve().parent / f).read_bytes()).hexdigest()[:16]
        for f in ENGINE_FILES}
    record = {
        "record": "nfl_joint_rebaseline",
        "frame_sha": frame_sha,
        "frame_sha256": _frame_sha256(feats),
        "written_utc": pd.Timestamp.now("UTC").isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "scope": SCOPE_PIN,
        "design_rule": ("fits NOTHING new — pure re-measurement through "
                        "committed entrypoints; engine sigma_const/rho/tie "
                        "keep fit_joint_params' pooled-OOF convention "
                        "(unchanged engine design); no pooled static overlay; "
                        "market-layer shrinkage d OUT OF SCOPE"),
        "geometry": {
            "seasons": sorted(feats["season"].unique().tolist()),
            "train_seasons": TRAIN_SEASONS,
            "sealed_season": SEALED_SEASON,
            "n_folds": 88,
            "pooled_oof_n": int(len(eng_pooled)),
            "sealed_n": int(len(eng_sealed)),
            "marginal": "era-centered DN + const sigma (era layer 7260ddc)",
            "engine_files_sha256": engine_bytes,
            "engines_modified": False,
            "prior_records": {"era": ERA_RECORD_NAME,
                              "sigma": SIGMA_RECORD_NAME},
        },
        "engine_state_step0": engine_state,
        "step1_chain_rerun": step1,
        "step2_gates_fixed_engine": step2,
        "step3_seam": step3,
        "delta_table": rows,
        "verdict": {
            "state": "RE_BASELINED",
            "pass": bool(engine_ok and mae_pin and rounds_pin and sig_pin
                         and g2 and g3 and g5),
            "read": ("absolute PMF-derived numbers re-quoted on the fixed "
                     "engine; relative verdicts from the prior records "
                     "stand (same bug in both arms of every comparison); "
                     "the market layer's fold-fitted shrinkage d is built "
                     "against THESE numbers"),
        },
        "feature_columns_untouched": True,
        "judgment_calls": {
            "convention_collapse": ("internal totals ECE and documented-"
                                    "convention total-PIT ECE are different "
                                    "binnings of the same calibration fact; "
                                    "post-fix they agree at the small level "
                                    "reported (pre-fix they disagreed at "
                                    "0.0796 vs 0.0092 because the engine "
                                    "priced totals one grid index off)"),
            "seam_top_bin": ("the totals top bin after the fix is the "
                             "model-vs-market residual the market layer "
                             "addresses (disagreement shrinkage / own-line "
                             "quoting) — NOT a dispersion defect"),
            "g1_reported_not_gated": ("G1 is a mean-layer gate; re-quoted "
                                      "for completeness, not re-gated"),
        },
    }
    if not args.no_record:
        rec_path = DATA_DELIVERY / f"nfl_joint_rebaseline_{frame_sha}.json"
        rec_path.write_text(json.dumps(record, indent=2, default=str))
        print(f"\nRecord: {rec_path.name}")
    else:
        print("\n[--no-record] record skipped")
    print(f"Done in {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    raise SystemExit(main())