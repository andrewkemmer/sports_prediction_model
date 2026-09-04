"""Probe: binary moneyline serving axis — hinge-map arm test (READ-ONLY).

Follow-on to nfl_binary_calibration_3e8c8a510f04.json (e3aeece, DO_NOT_REFIT
for the pchip spline). Tests the hypothesis that the pchip +0.0129 pooled
logloss regression was MAP OVERFIT (88 per-fold flexible splines -> fold
variance; logloss is convex — prediction variance costs even when
mean-honest), and that the calibration curve KINKS near raw ~0.70-0.72:
identity (or near) below ~0.70, regularized monotone sharpening above.

Phase 1 (read-only diagnostic; no model change):
  1.1 per-band pooled logloss delta (in-band raw>0.70 vs out-of-band
      raw<=0.70) for current-Platt vs pchip vs raw — nested protocol
  1.2 fold-stability: spread across the per-fold map fits at fixed raw
      inputs 0.50/0.68/0.75/0.85 (Jensen variance cost)
  1.3 raw ECE computed directly (the gate never measured it): pooled +
      sealed overall + 70-80 band for RAW (identity)
  1.4 hinge-point scan h in 0.66..0.74 (0.01 steps) — nested pooled
      cal-logloss + band ECE per h; the chosen h feeds R2

Phase 2 (arm table + gated serving-axis selection):
  R0 = current global Platt (bit-consistency: a/b to 3e-5, sealed
       ll 0.6249 / ECE 0.0745)
  R1 = RAW (identity map) — ECE reported directly, no nan
  R2 = two-segment hinge: identity below h; above h a regularized
       1-parameter logistic segment anchored at h (slope shrunk toward
       1, monotone by construction, continuous at h)
  All maps fit nested strictly-earlier per fold on pre-holdout rows for
  the POOLED legs; SEALED legs use the deployed protocol (fit on ALL
  pooled rows, score sealed-2025) — the production semantics that
  reproduce the published map. Gate legs per spec; verdict routed as
  ADOPT_R1 / ADOPT_R2 / KEEP_PLATT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import probe_binary_calibration as P   # shared helpers (read-only module)

DD = Path(__file__).resolve().parent.parent / "data_delivery"
DATE = P.DATE

HINGE_LO, HINGE_HI = 0.66, 0.74
FIXED_INPUTS = (0.50, 0.68, 0.75, 0.85)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _hinge_apply(p: np.ndarray, a: float, h: float) -> np.ndarray:
    """Two-segment hinge map value at p: identity below h; logistic
    segment anchored at h with slope a above. Continuous at h by
    construction (sigma(logit(h)) == h), monotone in p for a > 0."""
    p = np.asarray(p, dtype=float)
    out = p.copy()
    hi = p > h
    x = _logit(p[hi]) - _logit(np.float64(h))
    out[hi] = 1.0 / (1.0 + np.exp(-(_logit(np.float64(h)) + a * x)))
    return out


def _fit_hinge(p_hi: np.ndarray, y_hi: np.ndarray, h: float) -> float:
    """Fit the hinge slope a on the above-h data only (logistic MLE on
    x = logit(p) - logit(h), no intercept), shrunk toward 1 (identity) and
    floored at 1 (sharpening, never softening). Returns a (1.0 = identity
    when the above-h pool cannot support a fit)."""
    y_hi = np.asarray(y_hi, dtype=int)
    if len(y_hi) < 20 or len(np.unique(y_hi)) < 2:
        return 1.0
    from sklearn.linear_model import LogisticRegression
    x = (_logit(np.asarray(p_hi, dtype=float)) - _logit(np.float64(h))
         ).reshape(-1, 1)
    lr = LogisticRegression(C=1e3, fit_intercept=False, max_iter=1000)
    lr.fit(x, y_hi)
    a = float(lr.coef_[0][0])
    a_reg = 1.0 + 0.75 * (a - 1.0)     # shrink toward identity (fold variance)
    return max(a_reg, 1.0)


def _sealed_protocol_arms(pool: pd.DataFrame, seal: pd.DataFrame, h: float):
    """Fit each arm on ALL pooled rows (deployed protocol) -> sealed eval.
    Returns dict arm -> {cal vector on sealed}."""
    y_p = pool["home_win"].to_numpy(float)
    raw_p = pool["raw"].to_numpy(float)
    raw_s = seal["raw"].to_numpy(float)
    ma = P._fit_platt(raw_p, y_p)
    a_hi = raw_p > h
    a_h = _fit_hinge(raw_p[a_hi], y_p[a_hi], h)
    return {
        "R0_platt": P._apply(ma, raw_s),
        "R1_raw": raw_s.copy(),
        "R2_hinge": _hinge_apply(raw_s, a_h, h),
    }


def main() -> int:
    hist = pd.read_csv(DD / f"nfl_predictions_history_{DATE}.csv")
    hdf = hist.copy()
    hdf["raw"] = hdf["home_win_prob_model"]
    hdf["cal"] = hdf["home_win_prob_model_calibrated"]
    hdf["home_win"] = (hdf["home_score"] > hdf["away_score"]).astype(float)
    hdf["gameday"] = pd.to_datetime(hdf["game_date"])
    hdf["week_start"] = P._week_start(hdf["game_date"])
    pool = hdf[hdf["season"] <= 2024].copy()
    pool = pool.sort_values("gameday").reset_index(drop=True)
    seal = hdf[hdf["season"] == 2025].copy()
    seal = seal.sort_values("gameday").reset_index(drop=True)

    out: dict = {"universe": {
        "pooled_pre_holdout_rows": int(len(pool)),
        "sealed_rows": int(len(seal)),
        "pooled_fold_weeks": int(pool["week_start"].nunique()),
        "published_platt": {"a": 1.276336, "b": 0.121988},
    }}

    y_all = pool["home_win"].to_numpy(float)
    raw_all = pool["raw"].to_numpy(float)
    weeks = pool["week_start"]
    uniq_weeks = pd.unique(weeks)

    # ---- nested evaluator: per-fold maps fit on strictly-earlier folds ---
    def _nested(arm_name: str, h: float | None = None):
        cal = np.zeros(len(pool))
        mappers = {}
        for w in uniq_weeks:
            f_mask = (weeks == w).to_numpy()
            prior = (weeks < w).to_numpy()
            if arm_name == "R1_raw":
                mapper = None
            elif prior.sum() >= 10:
                if arm_name == "R0_platt":
                    mapper = P._fit_platt(raw_all[prior], y_all[prior])
                elif arm_name == "R2_hinge":
                    hi = raw_all[prior] > h
                    mapper = _fit_hinge(raw_all[prior][hi], y_all[prior][hi], h)
                else:  # pchip (diagnostic reference from e3aeece)
                    mapper = P._fit_pchip(raw_all[prior], y_all[prior])
            else:
                mapper = None
            if arm_name == "R2_hinge" and mapper is not None:
                cal[f_mask] = _hinge_apply(raw_all[f_mask], mapper, h)
            elif mapper is not None:
                cal[f_mask] = P._apply(mapper, raw_all[f_mask])
            else:
                cal[f_mask] = raw_all[f_mask]
            mappers[str(pd.Timestamp(w).date())] = mapper
        return cal, mappers

    def _ll_split(mask: np.ndarray, p: np.ndarray) -> float:
        return P._ll(y_all[mask], p[mask]) if mask.sum() else float("nan")

    # ---- Phase 1.1 per-band pooled logloss delta (nested) -----------------
    cal_a, _ = _nested("R0_platt")
    cal_c, _ = _nested("pchip")
    raw_v = raw_all.copy()
    in_mask = raw_all > 0.70
    out_mask = ~in_mask
    out["phase1_per_band_logloss"] = {
        "in_band_raw_gt_0_70": {
            "n": int(in_mask.sum()),
            "raw_identity": round(_ll_split(in_mask, raw_v), 4),
            "platt_nested": round(_ll_split(in_mask, cal_a), 4),
            "pchip_nested": round(_ll_split(in_mask, cal_c), 4),
        },
        "out_of_band_raw_le_0_70": {
            "n": int(out_mask.sum()),
            "raw_identity": round(_ll_split(out_mask, raw_v), 4),
            "platt_nested": round(_ll_split(out_mask, cal_a), 4),
            "pchip_nested": round(_ll_split(out_mask, cal_c), 4),
        },
        "pooled_total": {
            "n": int(len(pool)),
            "raw_identity": round(P._ll(y_all, raw_v), 4),
            "platt_nested": round(P._ll(y_all, cal_a), 4),
            "pchip_nested": round(P._ll(y_all, cal_c), 4),
        },
        "hypothesis_label": ("if the pchip regression concentrates "
                             "OUT-OF-BAND -> spline wiggle/per-fold "
                             "instability (overfit); if IN-BAND -> genuine "
                             "within-band discrimination loss"),
    }

    # ---- Phase 1.2 fold-stability at fixed raw inputs ---------------------
    _, mappers_a = _nested("R0_platt")
    _, mappers_c = _nested("pchip")
    stab = {}
    for fam, mappers in (("platt_nested", mappers_a), ("pchip_nested", mappers_c)):
        rows = {}
        for x in FIXED_INPUTS:
            vals = []
            for m in mappers.values():
                if m is None:
                    continue
                v = float(P._apply(m, np.array([x]))[0])
                vals.append(v)
            rows[f"input_{x:.2f}"] = {
                "n_fits": len(vals),
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }
        stab[fam] = rows
    out["phase1_fold_stability"] = {
        "fixed_inputs": list(FIXED_INPUTS),
        "identity_spread": "0 by definition (R1 is a constant map at every input)",
        "families": stab,
        "finding_note": ("high fold-to-fold spread at a CONSTANT raw input is "
                         "the Jensen variance cost: logloss is convex, so "
                         "per-fold prediction variance costs logloss even "
                         "when the map is mean-honest"),
    }

    # ---- Phase 1.3 raw ECE direct (no nan) --------------------------------
    y_seal = seal["home_win"].to_numpy(float)
    raw_seal = seal["raw"].to_numpy(float)
    cal_seal = seal["cal"].to_numpy(float)

    def _band_row(y, p, lo=0.70, hi=0.80):
        m = (p >= lo) & (p < hi)
        return {"n": int(m.sum()),
                "band_ece": round(P.band_ece(y, p, lo, hi), 4) if m.sum() else None}

    out["phase1_raw_ece_direct"] = {
        "pooled": {
            "n": int(len(pool)),
            "overall_ece": round(P._band_ece_overall(y_all, raw_v), 4),
            "band_70_80": _band_row(y_all, raw_v),
            "note": "identity map — ECE computed directly; the sealed gate "
                    "only scored calibrated axes, so this baseline was never "
                    "measured"},
        "sealed": {
            "n": int(len(seal)),
            "overall_ece": round(P._band_ece_overall(y_seal, raw_seal), 4),
            "band_70_80": _band_row(y_seal, raw_seal),
            "reference_platt_sealed": {
                "overall_ece": round(P._band_ece_overall(y_seal, cal_seal), 4),
                "band_70_80": _band_row(y_seal, cal_seal)}},
    }

    # ---- Phase 1.4 hinge-point scan (pre-holdout, nested) -----------------
    scan = []
    hs = np.round(np.arange(HINGE_LO, HINGE_HI + 1e-9, 0.01), 2)
    for h in hs:
        cal_h, _ = _nested("R2_hinge", float(h))
        scan.append({
            "h": round(float(h), 2),
            "pooled_logloss": round(P._ll(y_all, cal_h), 4),
            "band_70_80_ece": round(P.band_ece(y_all, cal_h, 0.70, 0.80), 4),
        })
    best_h = min(scan, key=lambda r: r["pooled_logloss"])["h"]
    out["phase1_hinge_scan"] = {
        "surface": scan,
        "chosen_h": best_h,
        "chosen_criterion": "argmin pooled nested cal-logloss on pre-holdout; "
                            "band ECE reported alongside",
    }

    # ---- Phase 2 arm table ------------------------------------------------
    h = best_h
    cal_r0, _ = _nested("R0_platt")
    cal_r2, _ = _nested("R2_hinge", h)
    cal_r1 = raw_all.copy()

    def _ll2(p: np.ndarray) -> float:
        return P._ll(y_all, p)

    pooled_table = {
        "R0_platt": {"logloss": round(_ll2(cal_r0), 4),
                     "ece": round(P._band_ece_overall(y_all, cal_r0), 4)},
        "R1_raw": {"logloss": round(_ll2(cal_r1), 4),
                   "ece": round(P._band_ece_overall(y_all, cal_r1), 4)},
        "R2_hinge": {"logloss": round(_ll2(cal_r2), 4),
                     "ece": round(P._band_ece_overall(y_all, cal_r2), 4)},
    }
    # AUC contract (monotone arms are rank-invariant on pooled)
    from nfl_moneyline import auc
    auc_raw = auc(y_all, raw_all)
    aucs = {"R0": round(auc(y_all, cal_r0), 6),
            "R1": round(auc_raw, 6),
            "R2": round(auc(y_all, cal_r2), 6)}

    # sealed deployed protocol
    s_arms = _sealed_protocol_arms(pool, seal, h)
    sealed_table = {}
    for name, cal_s in s_arms.items():
        d = {"logloss": round(P._ll(y_seal, cal_s), 4),
             "ece": round(P._band_ece_overall(y_seal, cal_s), 4),
             "bands": {}}
        for lo, hi in P.BANDS:
            d["bands"][f"{lo:.0%}-{hi:.0%}"] = _band_row(y_seal, cal_s, lo, hi)
        if name == "R0_platt":
            d["bit_consistency"] = {"published_sealed_ll": 0.6249,
                                    "published_sealed_ece": 0.0745}
        sealed_table[name] = d

    # S1/S2 splits (sealed chronological halves; S2 arbiter)
    half = len(seal) // 2
    splits = {}
    for label, sl in (("S1_first", seal.iloc[:half]), ("S2_second", seal.iloc[half:])):
        ys = sl["home_win"].to_numpy(float)
        raws = sl["raw"].to_numpy(float)
        ma = P._fit_platt(raw_all, y_all)
        a_hi = raw_all > h
        a_h = _fit_hinge(raw_all[a_hi], y_all[a_hi], h)
        splits[label] = {
            "n": int(len(sl)),
            "R0": {"ll": round(P._ll(ys, P._apply(ma, raws)), 4),
                   "ece": round(P._band_ece_overall(ys, P._apply(ma, raws)), 4)},
            "R1": {"ll": round(P._ll(ys, raws), 4),
                   "ece": round(P._band_ece_overall(ys, raws), 4)},
            "R2": {"ll": round(P._ll(ys, _hinge_apply(raws, a_h, h)), 4),
                   "ece": round(P._band_ece_overall(ys, _hinge_apply(raws, a_h, h)), 4)},
        }

    # ---- gate legs ---------------------------------------------------------
    r0ll, r1ll, r2ll = (pooled_table[k]["logloss"] for k in
                        ("R0_platt", "R1_raw", "R2_hinge"))
    r0ece, r1ece, r2ece = (pooled_table[k]["ece"] for k in
                           ("R0_platt", "R1_raw", "R2_hinge"))
    b_r0 = sealed_table["R0_platt"]["bands"]["70%-80%"]["band_ece"]
    b_r1 = sealed_table["R1_raw"]["bands"]["70%-80%"]["band_ece"]
    b_r2 = sealed_table["R2_hinge"]["bands"]["70%-80%"]["band_ece"]

    def _adj_reg(name: str) -> float:
        s = sealed_table[name]["bands"]
        return max((s["60%-70%"]["band_ece"] or 0.0)
                   - (sealed_table["R0_platt"]["bands"]["60%-70%"]["band_ece"] or 0.0),
                   (s["80%-101%"]["band_ece"] or 0.0)
                   - (sealed_table["R0_platt"]["bands"]["80%-101%"]["band_ece"] or 0.0))

    legs = {
        "R1_raw": {
            "pooled_ll_no_regression_beyond_0_001_vs_R0": {
                "r0": r0ll, "arm": r1ll, "delta": round(r1ll - r0ll, 4),
                "pass": (r1ll - r0ll) <= 0.001},
            "pooled_ece_improves": {"r0": r0ece, "arm": r1ece,
                                    "pass": r1ece <= r0ece,
                                    "target_lt_0_02_reached": r1ece < 0.02},
            "sealed_70_80_band_lt_0_03": {"r0": b_r0, "arm": b_r1,
                                          "pass": (b_r1 is not None) and b_r1 < 0.03},
            "sealed_adjacent_no_regression_gt_0_01": {"delta": round(_adj_reg("R1_raw"), 4),
                                                      "pass": _adj_reg("R1_raw") <= 0.01},
            "auc_flat_within_0_001": {"r0_auc": aucs["R0"], "arm_auc": aucs["R1"],
                                      "pass": abs(aucs["R1"] - aucs["R0"]) <= 0.001},
            "sealed_s2_arbiter": {
                "s2_ece": splits["S2_second"]["R1"]["ece"],
                "s2_ece_r0": splits["S2_second"]["R0"]["ece"],
                "s2_ll": splits["S2_second"]["R1"]["ll"],
                "s2_ll_r0": splits["S2_second"]["R0"]["ll"],
                "pass": (splits["S2_second"]["R1"]["ece"]
                         <= splits["S2_second"]["R0"]["ece"] + 0.01)
                        and (splits["S2_second"]["R1"]["ll"]
                             <= splits["S2_second"]["R0"]["ll"] + 0.001)},
            "worth_having": {"pass": (r1ll < r0ll - 0.001)
                                     or (r1ece < r0ece - 0.005)
                                     or ((b_r1 or 1.0) < (b_r0 or 0.0) - 0.005)},
        },
        "R2_hinge": {
            "pooled_ll_no_regression_beyond_0_001_vs_R0": {
                "r0": r0ll, "arm": r2ll, "delta": round(r2ll - r0ll, 4),
                "pass": (r2ll - r0ll) <= 0.001},
            "pooled_ece_improves": {"r0": r0ece, "arm": r2ece,
                                    "pass": r2ece <= r0ece,
                                    "target_lt_0_02_reached": r2ece < 0.02},
            "sealed_70_80_band_lt_0_03": {"r0": b_r0, "arm": b_r2,
                                          "pass": (b_r2 is not None) and b_r2 < 0.03},
            "sealed_adjacent_no_regression_gt_0_01": {"delta": round(_adj_reg("R2_hinge"), 4),
                                                      "pass": _adj_reg("R2_hinge") <= 0.01},
            "auc_flat_within_0_001": {"r0_auc": aucs["R0"], "arm_auc": aucs["R2"],
                                      "pass": abs(aucs["R2"] - aucs["R0"]) <= 0.001},
            "sealed_s2_arbiter": {
                "s2_ece": splits["S2_second"]["R2"]["ece"],
                "s2_ece_r0": splits["S2_second"]["R0"]["ece"],
                "s2_ll": splits["S2_second"]["R2"]["ll"],
                "s2_ll_r0": splits["S2_second"]["R0"]["ll"],
                "pass": (splits["S2_second"]["R2"]["ece"]
                         <= splits["S2_second"]["R0"]["ece"] + 0.01)
                        and (splits["S2_second"]["R2"]["ll"]
                             <= splits["S2_second"]["R0"]["ll"] + 0.001)},
            "worth_having": {"pass": (r2ll < r0ll - 0.001)
                                     or (r2ece < r0ece - 0.005)
                                     or ((b_r2 or 1.0) < (b_r0 or 0.0) - 0.005)},
        },
    }

    def _route(arm: str) -> bool:
        return all(v["pass"] for v in legs[arm].values())

    r1_ok, r2_ok = _route("R1_raw"), _route("R2_hinge")
    if r1_ok and r2_ok:
        verdict = "ADOPT_R2" if r2ll < r1ll else "ADOPT_R1"
    elif r2_ok:
        verdict = "ADOPT_R2"
    elif r1_ok:
        verdict = "ADOPT_R1"
    else:
        verdict = "KEEP_PLATT"
    out["verdict_routing_rule"] = ("single passer adopted; if both pass, the "
                                   "better (lower) pooled nested cal-logloss "
                                   "wins; neither -> KEEP_PLATT")
    out["phase2"] = {
        "hinge_h": h,
        "pooled_nested": pooled_table,
        "pooled_auc": aucs,
        "sealed_deployed": sealed_table,
        "sealed_s1_s2": splits,
        "legs": legs,
        "R1_passes_all": r1_ok,
        "R2_passes_all": r2_ok,
        "verdict": verdict,
    }

    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
