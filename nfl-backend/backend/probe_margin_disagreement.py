"""Probe: binary-vs-derived ML disagreement on the shared decided OOF
universe (READ-ONLY — produces JSON for the margin-audit record).

Sources (both committed artifacts, no fitting of any production model):
  - nfl_run_engine_markets_<date>.csv  kind==oof rows: pred_home/pred_away,
    p_home_win_derived, home_score/away_score, margin, total, frame_view.
  - nfl_predictions_history_<date>.csv : home_win_prob_model (RAW ensemble),
    home_win_prob_model_calibrated (published Platt axis), scores.

Outputs:
  coverage          - join coverage both directions on the 1,376 OOF universe
  disagreement      - binary - derived: mean/median/quantiles, by favorite
                      band / road favorite / totals band
  calibration       - actual win rate per confidence band (binary calibrated
                      axis, binary raw, run-engine derived)
  sextile_response  - actual vs run-engine-pred vs binary-implied margin
                      spread across sextiles of the top quality features
  gap_regression    - (binary - derived) on the 12-pool (stdz, OLS): R2 + coefs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DD = Path(__file__).resolve().parent.parent / "data_delivery"
DATE = "20260904"

SIDE_FEATURES = ["elo_diff", "win_pct_diff", "rest_days_diff", "is_dome_home",
                 "ewm_net_pts_diff", "ewm_ypp_diff", "pace_plays_min_diff",
                 "rest_short_diff", "div_game", "travel_miles_diff",
                 "altitude_home", "prime_time"]

PINNED_SIGMA_MARGIN = 13.2   # sqrt(9.663^2 + 9.0789^2 - 2*0.0076*9.663*9.0789)


def _norm_ppf(p: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(float(np.clip(p, 1e-6, 1 - 1e-6))))


def main() -> int:
    mk = pd.read_csv(DD / f"nfl_run_engine_markets_{DATE}.csv")
    oof = mk[mk["kind"] == "oof"].copy()
    hist = pd.read_csv(DD / f"nfl_predictions_history_{DATE}.csv")

    oof = oof.rename(columns={"pred_home": "re_pred_home",
                              "pred_away": "re_pred_away"})
    df = oof.merge(
        hist[["game_id", "home_win_prob_model",
              "home_win_prob_model_calibrated"]],
        on="game_id", how="left")
    # The markets CSV prices rows but does not carry the 12-pool feature
    # columns — join them from the canonical decided feature frame
    # (load_features hits the /tmp cache; read-only, no fitting).
    from run_nfl_margin_ablation import load_features
    feats = load_features(None)[["game_id"] + SIDE_FEATURES]
    df = df.merge(feats, on="game_id", how="left")

    out: dict = {}
    n_oof = len(oof)
    n_joined = int(df["home_win_prob_model_calibrated"].notna().sum())
    out["coverage"] = {
        "run_engine_oof_rows": n_oof,
        "oof_with_binary_calibrated": n_joined,
        "oof_without_binary": int((n_oof - n_joined)),
        "binary_history_rows": int(len(hist)),
        "binary_rows_not_in_oof": int(
            (~hist["game_id"].isin(set(oof["game_id"]))).sum()),
    }

    df["binary"] = df["home_win_prob_model_calibrated"]
    df["binary_raw"] = df["home_win_prob_model"]
    df["derived"] = df["p_home_win_derived"]
    df["gap"] = df["binary"] - df["derived"]
    df["re_margin"] = df["re_pred_home"] - df["re_pred_away"]
    df["binary_implied_margin"] = df["binary"].map(
        lambda p: _norm_ppf(p) * PINNED_SIGMA_MARGIN)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(float)

    g = df["gap"]
    out["disagreement"] = {
        "n": int(g.notna().sum()),
        "mean_binary_minus_derived": round(float(g.mean()), 4),
        "median": round(float(g.median()), 4),
        "q05": round(float(g.quantile(0.05)), 4),
        "q95": round(float(g.quantile(0.95)), 4),
        "mean_abs_gap": round(float(g.abs().mean()), 4),
        "pct_gap_gt_5pt": round(float((g.abs() > 0.05).mean()), 4),
        "pct_gap_gt_10pt": round(float((g.abs() > 0.10).mean()), 4),
        "binary_higher": round(float((g > 0).mean()), 4),
        "derived_higher": round(float((g < 0).mean()), 4),
        "run_engine_mean_margin": round(float(df["re_margin"].mean()), 3),
        "binary_implied_mean_margin": round(
            float(df["binary_implied_margin"].mean()), 3),
        "sigma_convention": f"DN margin sigma {PINNED_SIGMA_MARGIN}",
    }

    # --- concentration by favorite band (derived ML), road favorite, totals
    bands = {}
    edges = [0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (df["derived"] >= lo) & (df["derived"] < hi)
        if m.sum() == 0:
            continue
        sub = df[m]
        bands[f"{lo:.0%}-{hi:.0%}"] = {
            "n": int(len(sub)),
            "mean_binary": round(float(sub["binary"].mean()), 4),
            "mean_derived": round(float(sub["derived"].mean()), 4),
            "mean_gap": round(float(sub["gap"].mean()), 4),
        }
    out["by_derived_band"] = bands

    road = df[df["re_margin"] < 0]
    home_fav = df[df["re_margin"] >= 0]
    out["favorite_side"] = {
        "home_fav_n": int(len(home_fav)),
        "home_fav_mean_gap": round(float(home_fav["gap"].mean()), 4),
        "road_fav_n": int(len(road)),
        "road_fav_mean_gap": round(float(road["gap"].mean()), 4),
    }

    t_bands = {}
    for lo, hi in ((0, 40), (40, 46), (46, 52), (52, 100)):
        m = (df["total"] >= lo) & (df["total"] < hi)
        if m.sum() == 0:
            continue
        sub = df[m]
        t_bands[f"{lo}-{hi}"] = {
            "n": int(len(sub)),
            "mean_gap": round(float(sub["gap"].mean()), 4),
            "mean_binary": round(float(sub["binary"].mean()), 4),
        }
    out["by_total_band"] = t_bands

    # --- calibration by confidence band (actual win rate)
    def _cal_rows(pcol: str) -> dict:
        rows = {}
        for lo, hi in ((0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)):
            m = (df[pcol] >= lo) & (df[pcol] < hi)
            if m.sum() == 0:
                continue
            sub = df[m]
            rows[f"{lo:.0%}-{hi:.0%}"] = {
                "n": int(len(sub)),
                "mean_pred": round(float(sub[pcol].mean()), 4),
                "actual_win_rate": round(float(sub["home_win"].mean()), 4),
                "ece_band": round(float(abs(
                    sub[pcol].mean() - sub["home_win"].mean())), 4),
            }
        return rows

    out["calibration"] = {
        "binary_platt": _cal_rows("binary"),
        "binary_raw": _cal_rows("binary_raw"),
        "run_engine_derived": _cal_rows("derived"),
        "note": ("home-win convention: pred is P(home wins); actual = "
                 "home_score > away_score (ties excluded, 0.275% mass)"),
    }

    # --- sextile response (actual vs run-engine pred vs binary-implied
    # margin) across the top quality features — MLB 39c865e mirror
    sext = {}
    for feat in ("elo_diff", "ewm_net_pts_diff", "win_pct_diff",
                 "ewm_ypp_diff"):
        if feat not in df.columns:
            continue
        try:
            df["_q"] = pd.qcut(df[feat], 6, labels=False, duplicates="drop")
        except ValueError:
            continue
        rows = []
        for q in sorted(df["_q"].dropna().unique()):
            sub = df[df["_q"] == q]
            rows.append({
                "sextile": int(q),
                "n": int(len(sub)),
                "actual_margin": round(float(sub["margin"].mean()), 3),
                "re_pred_margin": round(float(sub["re_margin"].mean()), 3),
                "binary_implied_margin": round(
                    float(sub["binary_implied_margin"].mean()), 3),
            })
        act = rows[0]["actual_margin"] - rows[-1]["actual_margin"]
        rep = rows[0]["re_pred_margin"] - rows[-1]["re_pred_margin"]
        bim = (rows[0]["binary_implied_margin"]
               - rows[-1]["binary_implied_margin"])
        sext[feat] = {
            "rows": rows,
            "actual_spread": round(act, 3),
            "re_pred_spread": round(rep, 3),
            "binary_implied_spread": round(bim, 3),
            "recovery_pct_re": round(100.0 * rep / act, 1) if act else None,
            "recovery_pct_binary": round(100.0 * bim / act, 1) if act else None,
        }
    out["sextile_response"] = sext

    # --- gap attribution: OLS of (binary - derived) on the 12-pool
    reg_df = df[["gap"] + SIDE_FEATURES].dropna()
    X = reg_df[SIDE_FEATURES].to_numpy(float)
    y = reg_df["gap"].to_numpy(float)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Xs = np.column_stack([np.ones(len(Xs)), Xs])
    beta, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    resid = y - Xs @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot else None
    out["gap_regression"] = {
        "n": int(len(reg_df)),
        "r2": round(r2, 4) if r2 is not None else None,
        "intercept": round(float(beta[0]), 4),
        "coefs_stdz": {f: round(float(b), 4)
                       for f, b in zip(SIDE_FEATURES, beta[1:])},
        "top_abs": sorted(
            ((f, round(abs(float(b)), 4))
             for f, b in zip(SIDE_FEATURES, beta[1:])),
            key=lambda t: -t[1])[:5],
        "note": "standardized OLS; R2 = share of the binary-derived gap the "
                "12-pool explains on the shared OOF",
    }

    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())