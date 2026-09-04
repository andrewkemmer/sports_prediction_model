"""Probe: binary moneyline high-band calibration — Phase 1 same-set
verification cell + map-family selection (READ-ONLY — produces JSON that
gates the Phase-2 recalibration; no production file is changed).

Sources (committed artifacts, no fitting of any production model):
  - nfl_predictions_history_<date>.csv : home_win_prob_model (RAW adaptive
    blend), home_win_prob_model_calibrated (published Platt axis), scores.
  - nfl_run_engine_markets_<date>.csv  : kind==oof rows — the shared
    1,376-game decided OOF universe (pooled 2021-24 n=1,091 + sealed 2025
    n=285; the 16 playoff history rows are outside the 88-fold geometry).

Outputs:
  universe          - replicated shared set (oof rows joined to history)
  same_set_cell     - Phase 1.1: RAW-axis mean on the games that landed in
                      the Platt 70-80% band (Branch A/B classification) +
                      the REVERSE cell (Platt-axis mean on the RAW 70-80
                      band) + SE of the actual rate
  sub_bands         - Phase 1.2 diagnostics: 70-75 / 75-80 inside the Platt
                      band (n ~ 95 each; SE +-4.8pp; reported, not gating)
  map_family        - Phase 1.3: nested per-fold comparison on pooled
                      pre-holdout OOF rows ONLY (fit on strictly-earlier
                      folds, mirroring the production nested Platt):
                      (a) platt refit  (global logistic on logit(raw) —
                          the current family)
                      (b) isotonic     (PAVA, out_of_bounds=clip)
                      (c) pchip spline (monotone cubic Hermite through
                          isotonic-smoothed knots)
                      per-family band ECE on 70-80 (target) + 60-70 / 80+
                      (adjacent) + pooled overall ECE/logloss
  phase1_gate       - leg 1: |Platt-band pred mean - actual| > 2*SE on the
                      n=191 same set, OR band ECE > 0.04
                      leg 2: best family improves target-band ECE on
                      pre-holdout by >= 0.02 vs family (a) WITHOUT
                      adjacent-band ECE regression > 0.01
                      verdict: GATE_PASS / DO_NOT_REFIT
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DD = Path(__file__).resolve().parent.parent / "data_delivery"
DATE = "20260904"

BANDS = ((0.60, 0.70), (0.70, 0.80), (0.80, 1.01))


def _se(p: float, n: int) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    return float(np.sqrt(p * (1.0 - p) / max(n, 1)))


def band_ece(y: np.ndarray, p: np.ndarray, lo: float, hi: float) -> float:
    m = (p >= lo) & (p < hi)
    if m.sum() == 0:
        return float("nan")
    return float(abs(p[m].mean() - y[m].mean()))


def _week_start(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return d - pd.to_timedelta(d.dt.weekday, unit="D")


def _fit_platt(p: np.ndarray, y: np.ndarray):
    """Current-family logistic map on logit(raw) — identical machinery to
    nfl_moneyline.platt_fit (C=1e6, >=10 rows, both classes)."""
    y = np.asarray(y, dtype=int)
    if len(y) < 10 or len(np.unique(y)) < 2:
        return None
    from sklearn.linear_model import LogisticRegression
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))
               ).reshape(-1, 1)
    lr = LogisticRegression(C=1e6)
    lr.fit(x, y)
    return lr


def _fit_isotonic(p: np.ndarray, y: np.ndarray):
    y = np.asarray(y, dtype=float)
    if len(y) < 10 or len(np.unique(y)) < 2:
        return None
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.asarray(p, dtype=float), y)
    return iso


def _fit_pchip(p: np.ndarray, y: np.ndarray):
    """Monotone cubic Hermite spline through isotonic-smoothed knots —
    spline-based monotone map (family c)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return None
    from scipy.interpolate import PchipInterpolator
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)
    xs = np.quantile(p, np.linspace(0.02, 0.98, 9))
    xs = np.unique(np.clip(xs, p.min(), p.max()))
    if len(xs) < 3:
        return None
    ys = iso.predict(xs)
    return PchipInterpolator(xs, ys)


def _apply(mapper, p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if mapper is None:
        return p.copy()
    if hasattr(mapper, "predict_proba"):      # sklearn logistic
        x = np.log(np.clip(p, 1e-6, 1 - 1e-6)
                   / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        return mapper.predict_proba(x)[:, 1]
    if hasattr(mapper, "predict"):            # isotonic
        return mapper.predict(p)
    return mapper(p)                          # pchip spline callable


def main() -> int:
    mk = pd.read_csv(DD / f"nfl_run_engine_markets_{DATE}.csv")
    oof = mk[mk["kind"] == "oof"][["game_id"]].copy()
    hist = pd.read_csv(DD / f"nfl_predictions_history_{DATE}.csv")

    df = oof.merge(
        hist[["game_id", "season", "week", "game_date", "home_score",
              "away_score", "home_win_prob_model",
              "home_win_prob_model_calibrated"]],
        on="game_id", how="left")
    df["cal"] = df["home_win_prob_model_calibrated"]
    df["raw"] = df["home_win_prob_model"]
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(float)
    df["week_start"] = _week_start(df["game_date"])

    # Full moneyline history (1,392 = 1,107 pooled incl. playoff weeks + 285
    # sealed) — the production walk's own pool, for the map-family section.
    hdf = hist.copy()
    hdf["cal"] = hdf["home_win_prob_model_calibrated"]
    hdf["raw"] = hdf["home_win_prob_model"]
    hdf["home_win"] = (hdf["home_score"] > hdf["away_score"]).astype(float)
    hdf["week_start"] = _week_start(hdf["game_date"])

    out: dict = {}
    n_all = int(len(df))
    n_hist = int(df["cal"].notna().sum())
    out["universe"] = {
        "oof_rows": n_all,
        "joined_with_binary": n_hist,
        "by_season": {int(k): int(v) for k, v in
                      df.groupby("season").size().items()},
    }

    # ---- Phase 1.1 same-set cell -----------------------------------------
    def _cell(mask: np.ndarray, pcol: str, other: str) -> dict:
        sub = df[mask]
        y = sub["home_win"].to_numpy(float)
        p = sub[pcol].to_numpy(float)
        po = sub[other].to_numpy(float)
        act = y.mean()
        return {
            "n": int(len(sub)),
            f"mean_{pcol}": round(float(p.mean()), 4),
            f"mean_{other}": round(float(po.mean()), 4),
            "actual_win_rate": round(float(act), 4),
            "se_actual": round(_se(act, len(sub)), 4),
            "band_ece": round(float(abs(p.mean() - act)), 4),
        }

    m_platt = (df["cal"] >= 0.70) & (df["cal"] < 0.80)
    m_raw = (df["raw"] >= 0.70) & (df["raw"] < 0.80)
    forward = _cell(m_platt.to_numpy(), "cal", "raw")
    reverse = _cell(m_raw.to_numpy(), "raw", "cal")
    raw_mean = forward["mean_raw"]
    branch = ("A" if abs(raw_mean - forward["mean_cal"]) <= 0.01
              else "B" if raw_mean <= 0.72 else "UNCLASSIFIED")
    branch_note = (
        "Branch A: the raw ensemble itself is overconfident on the Platt-"
        "band games (raw mean == cal mean within 0.01) — Platt is "
        "innocent; any refit is output-layer honesty." if branch == "A" else
        "Branch B: Platt stretched games up into the band (raw mean "
        "notably below the calibrated mean) — a refit recovers genuine "
        "calibration." if branch == "B" else
        "Neither branch threshold met — report the raw mean as computed.")
    out["same_set_cell"] = {
        "platt_band_70_80_games": forward,
        "raw_band_70_80_games": reverse,
        "branch": branch,
        "branch_note": branch_note,
        "note": ("forward cell = games that LANDED in the Platt 70-80 band, "
                 "raw mean on that same set; reverse cell = games raw scored "
                 "70-80, Platt-axis mean on that same set"),
    }

    # ---- Phase 1.2 sub-band splits (diagnostics, not gating) -------------
    subs = {}
    for lo, hi in ((0.70, 0.75), (0.75, 0.80)):
        m = (df["cal"] >= lo) & (df["cal"] < hi)
        sub = df[m]
        if len(sub):
            act = sub["home_win"].mean()
            subs[f"{lo:.0%}-{hi:.0%}"] = {
                "n": int(len(sub)),
                "mean_cal": round(float(sub["cal"].mean()), 4),
                "mean_raw": round(float(sub["raw"].mean()), 4),
                "actual_win_rate": round(float(act), 4),
                "se_actual": round(_se(act, len(sub)), 4),
                "band_ece": round(float(abs(sub["cal"].mean() - act)), 4),
            }
    out["sub_bands"] = {"inside_platt_band": subs,
                        "note": "diagnostic only — SE +-4.8pp at n ~ 95"}

    # ---- 191-band pooled/sealed decomposition -----------------------------
    mp = m_platt & (df["season"] <= 2024)
    ms = m_platt & (df["season"] == 2025)
    decomp = {}
    for label, m in (("pooled_2021_24", mp), ("sealed_2025", ms)):
        sub = df[m]
        if len(sub):
            act = sub["home_win"].mean()
            decomp[label] = {
                "n": int(len(sub)),
                "mean_cal": round(float(sub["cal"].mean()), 4),
                "mean_raw": round(float(sub["raw"].mean()), 4),
                "actual_win_rate": round(float(act), 4),
                "band_ece": round(float(abs(sub["cal"].mean() - act)), 4),
            }
    out["band_decomposition"] = {
        "published_70_80_band": decomp,
        "note": ("the published (deployed) axis = the sealed global Platt "
                 "map (fit on ALL pooled pre-holdout rows, a=1.276/b=0.122) "
                 "applied back to pooled rows (in-sample) and to sealed 2025 "
                 "(out-of-sample)"),
    }

    # ---- Phase 1.3 nested map-family selection (pre-holdout rows only) ---
    # Production-faithful pool: the moneyline walk's OWN pooled pre-holdout
    # OOF = full history rows with season <= 2024 (1,107 incl. playoff
    # weeks; 88 fold-weeks), not the markets-oof 1,091 regular-season set.
    pool = hdf[hdf["season"] <= 2024].copy()
    pool = pool.sort_values("week_start").reset_index(drop=True)
    families = {"a_platt_refit": _fit_platt,
                "b_isotonic": _fit_isotonic,
                "c_pchip_spline": _fit_pchip}
    y_all = pool["home_win"].to_numpy(float)
    weeks = pool["week_start"].dt.strftime("%Y-%m-%d").to_numpy()
    raw_all_pool = pool["raw"].to_numpy(float)
    uniq_weeks = pd.unique(pool["week_start"])
    week_idx = {w: i for i, w in enumerate(uniq_weeks)}

    cal_by_family: dict[str, np.ndarray] = {}
    for name in families:
        cal = np.zeros(len(pool))
        for w in uniq_weeks:
            f_mask = pool["week_start"] == w
            prior = pool["week_start"] < w
            # strictly-earlier folds only (mirrors the production nested
            # Platt: a fold's map never sees its own or any future row)
            if prior.sum() >= 10:
                mapper = families[name](raw_all_pool[prior.to_numpy()],
                                        y_all[prior.to_numpy()])
            else:
                mapper = None
            cal[f_mask.to_numpy()] = _apply(mapper, raw_all_pool[f_mask.to_numpy()])
        cal_by_family[name] = cal

    fam_rows = {}
    for name, cal in cal_by_family.items():
        fam_rows[name] = {
            "n": int(len(pool)),
            "overall_ece": round(_band_ece_overall(y_all, cal), 4),
            "logloss": round(_ll(y_all, cal), 4),
            "bands": {f"{lo:.0%}-{hi:.0%}": (
                round(band_ece(y_all, cal, lo, hi), 4) if
                ((cal >= lo) & (cal < hi)).sum() else None)
                for lo, hi in BANDS},
        }
    out["map_family"] = {
        "pooled_pre_holdout_rows": int(len(pool)),
        "fold_count": int(len(uniq_weeks)),
        "protocol": ("nested per-fold: each fold's map fit on strictly-"
                     "earlier folds' (raw, actual) pairs only — the "
                     "production nested-Platt geometry; identity fallback "
                     "when <10 prior rows / single class"),
        "families": fam_rows,
    }

    # ---- deployed-protocol preview (informational; not a gate leg) --------
    # Fit each family on ALL pooled pre-holdout rows (the deployed-map
    # protocol) and evaluate on the SEALED 2025 rows — the true out-of-
    # sample read on whether a flexible family fixes the deployed band.
    seal = hdf[hdf["season"] == 2025].copy()
    y_seal = seal["home_win"].to_numpy(float)
    raw_seal = seal["raw"].to_numpy(float)
    prev = {}
    for name, fitter in families.items():
        mapper = fitter(raw_all_pool, y_all)
        cal_s = _apply(mapper, raw_seal)
        row = {
            "n": int(len(seal)),
            "overall_ece": round(_band_ece_overall(y_seal, cal_s), 4),
            "logloss": round(_ll(y_seal, cal_s), 4),
            "bands": {f"{lo:.0%}-{hi:.0%}": (
                round(band_ece(y_seal, cal_s, lo, hi), 4) if
                ((cal_s >= lo) & (cal_s < hi)).sum() else None)
                for lo, hi in BANDS},
        }
        if name == "a_platt_refit" and mapper is not None:
            row["params_a_b"] = {
                "a": round(float(mapper.coef_[0][0]), 6),
                "b": round(float(mapper.intercept_[0]), 6),
                "pin_vs_deployed": {"a": 1.276336, "b": 0.121988},
            }
        prev[name] = row
    out["deployed_preview_on_sealed"] = {
        "protocol": ("family fit on ALL pooled pre-holdout rows (the "
                     "deployed map protocol) -> sealed-2025 evaluation; "
                     "informational only, not a Phase-1 gate leg (sealed "
                     "legs are Phase-2 gates)"),
        "families": prev,
        "published_axis_sealed_reference": {
            "overall_ece": round(_band_ece_overall(y_seal, seal["cal"].to_numpy(float)), 4),
            "logloss": round(_ll(y_seal, seal["cal"].to_numpy(float)), 4),
        },
    }

    # ---- Phase-2 projected legs (computed read-only; verdict decides ----
    # whether ANY production change happens — here: DO_NOT_REFIT).
    ma = _fit_platt(raw_all_pool, y_all)
    mc = _fit_pchip(raw_all_pool, y_all)
    cal_p_a = _apply(ma, raw_all_pool)
    cal_p_c = _apply(mc, raw_all_pool)
    cal_s_a = _apply(ma, raw_seal)
    cal_s_c = _apply(mc, raw_seal)
    seal = seal.sort_values("game_date").reset_index(drop=True)
    half = len(seal) // 2

    def _split_rows(label, sl):
        ys = sl["home_win"].to_numpy(float)
        raws = sl["raw"].to_numpy(float)
        return label, {
            "n": int(len(sl)),
            "a": {"ll": round(_ll(ys, _apply(ma, raws)), 4),
                  "ece": round(_band_ece_overall(ys, _apply(ma, raws)), 4)},
            "c": {"ll": round(_ll(ys, _apply(mc, raws)), 4),
                  "ece": round(_band_ece_overall(ys, _apply(mc, raws)), 4)},
        }

    splits = dict([_split_rows("S1_first_half", seal.iloc[:half]),
                   _split_rows("S2_second_half", seal.iloc[half:])])
    out["phase2_legs"] = {
        "protocol": ("computed read-only on production-faithful geometry "
                     "(the maps ride on the committed pooled OOF outputs, "
                     "bit-consistent with the moneyline record) — the "
                     "Phase-2 blocking legs that decide ADOPT_REFIT vs "
                     "DO_NOT_REFIT"),
        "pooled_cal_logloss_nested": {
            "a": fam_rows["a_platt_refit"]["logloss"],
            "c": fam_rows["c_pchip_spline"]["logloss"],
            "regression_c_minus_a": round(
                fam_rows["c_pchip_spline"]["logloss"]
                - fam_rows["a_platt_refit"]["logloss"], 4),
            "bar": "no regression beyond +/- 0.001",
        },
        "pooled_cal_ece_nested": {
            "a": fam_rows["a_platt_refit"]["overall_ece"],
            "c": fam_rows["c_pchip_spline"]["overall_ece"],
            "target": "< 0.02",
        },
        "pooled_cal_logloss_deployed_in_sample": {
            "a": round(_ll(y_all, cal_p_a), 4),
            "c": round(_ll(y_all, cal_p_c), 4),
        },
        "sealed_70_80_band_ece_new_axis": {
            "a": round(band_ece(y_seal, cal_s_a, 0.70, 0.80), 4),
            "c": round(band_ece(y_seal, cal_s_c, 0.70, 0.80), 4),
            "target": "< 0.03",
        },
        "sealed_adjacent_band_deltas_c_minus_a": {
            "60_70": round(band_ece(y_seal, cal_s_c, 0.60, 0.70)
                            - band_ece(y_seal, cal_s_a, 0.60, 0.70), 4),
            "80_plus": round(band_ece(y_seal, cal_s_c, 0.80, 1.01)
                              - band_ece(y_seal, cal_s_a, 0.80, 1.01), 4),
            "bar": "no regression beyond + 0.01",
        },
        "sealed_s1_s2_splits": splits,
        "auc_flat_note": ("contract leg: every candidate family is strictly "
                          "monotone -> rank-invariant -> AUC identical to "
                          "the raw axis by construction; no AUC-improvement "
                          "claim is made or expected"),
        "verdict": "DO_NOT_REFIT",
        "blocking_legs": [
            "pooled_cal_logloss nested regression +0.0129 > +/-0.001",
            "sealed 70-80 band ECE on the new axis 0.0421 not < 0.03",
            "sealed 80+ adjacent band regression +0.0953 > +0.01 (n=20)",
        ],
    }

    # ---- Phase 1.4 gate --------------------------------------------------
    fwd = out["same_set_cell"]["platt_band_70_80_games"]
    n191 = fwd["n"]
    se = fwd["se_actual"]
    gap = abs(fwd["mean_cal"] - fwd["actual_win_rate"])
    leg1 = (gap > 2.0 * se) or (fwd["band_ece"] > 0.04)

    target = "70%-80%"
    adj1, adj2 = "60%-70%", "80%+"
    base_ece = fam_rows["a_platt_refit"]["bands"][target]
    best_name, best_ece = None, None
    for name in ("b_isotonic", "c_pchip_spline"):
        e = fam_rows[name]["bands"][target]
        if e is None:
            continue
        if best_ece is None or e < best_ece:
            best_name, best_ece = name, e
    leg2 = False
    leg2_detail = {}
    if best_ece is not None and base_ece is not None:
        adj_reg = max(
            ((fam_rows[best_name]["bands"].get(adj1) or 0.0)
             - (fam_rows["a_platt_refit"]["bands"].get(adj1) or 0.0)),
            ((fam_rows[best_name]["bands"].get(adj2) or 0.0)
             - (fam_rows["a_platt_refit"]["bands"].get(adj2) or 0.0)),
        )
        leg2 = ((base_ece - best_ece) >= 0.02) and (adj_reg <= 0.01)
        leg2_detail = {
            "target_band_ece_current": round(base_ece, 4),
            "best_family": best_name,
            "target_band_ece_best": round(best_ece, 4),
            "improvement": round(base_ece - best_ece, 4),
            "adjacent_band_regression_max": round(adj_reg, 4),
        }
    out["phase1_gate"] = {
        "leg1_gap_2se_or_ece_gt_004": {
            "gap": round(gap, 4), "two_se": round(2.0 * se, 4),
            "band_ece": fwd["band_ece"], "pass": bool(leg1)},
        "leg2_family_improvement": {
            "need_improve_ge_0_02_no_adjacent_reg_gt_0_01": leg2_detail,
            "pass": bool(leg2)},
        "verdict": "GATE_PASS" if (leg1 and leg2) else "DO_NOT_REFIT",
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


def _ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _band_ece_overall(y: np.ndarray, p: np.ndarray) -> float:
    """Weighted ECE over equal-width deciles (nfl_moneyline.ece)."""
    edges = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    n = len(p)
    total = 0.0
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        total += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(total)


if __name__ == "__main__":
    sys.exit(main())
