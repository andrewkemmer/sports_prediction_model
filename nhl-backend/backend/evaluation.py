"""OOF evaluation for the production moneyline, run-line, and totals models.

Structural mirror of the NFL evaluation.py. Every metric is computed on
out-of-fold predictions only. No full-history correlations, no in-sample
importance, no future outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from backend import config
    from backend import distributions as dist_mod
except ImportError:
    import config
    import distributions as dist_mod


# ---------------------------------------------------------------------------
# Binary (moneyline) metrics
# ---------------------------------------------------------------------------
def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-width bins."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return np.nan
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = len(p)
    err = 0.0
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        err += m.sum() / total * abs(p[m].mean() - y[m].mean())
    return float(err)


def binary_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) < 2 or len(np.unique(y)) < 2:
        return {"n": int(len(p)), "auc": np.nan, "logloss": np.nan,
                "brier": np.nan, "ece": np.nan}
    return {
        "n": int(len(p)),
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(p, y),
    }


def calibration_buckets(p: np.ndarray, y: np.ndarray,
                        n_bins: int = 10) -> list[dict]:
    """MLB-shaped reliability buckets (frontend presentation contract).

    Matches the MLB implementation exactly so the shared Calibration page
    renders all three sports through one code path with identical
    presentation:

    * FAVORED-team perspective: each game contributes ONE point at
      ``max(p, 1-p)`` ∈ [0.5, 1], labeled by whether the favorite won.
    * Bucket labels use the MLB en-dash convention (e.g. ``"50–60%"``).

    Values remain the NHL pipeline's own pooled OOF statistics.
    ``gap`` = mean_predicted − mean_actual (>0 = overconfident).
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    fav_prob = np.maximum(p, 1.0 - p)
    fav_won = np.where(p >= 0.5, y, 1.0 - y)
    half = max(n_bins // 2, 1)
    bin_edges = np.linspace(0.5, 1.0, half + 1)
    rows = []
    for i in range(len(bin_edges) - 1):
        m = (fav_prob >= bin_edges[i]) & (fav_prob < bin_edges[i + 1])
        if i == len(bin_edges) - 2:  # include 1.0 in the top bucket
            m |= fav_prob == bin_edges[i + 1]
        if not m.any():
            continue
        mean_pred, mean_actual = float(fav_prob[m].mean()), float(fav_won[m].mean())
        rows.append({
            "bucket": f"{bin_edges[i] * 100:.0f}–{bin_edges[i + 1] * 100:.0f}%",
            "mean_predicted": round(mean_pred, 4),
            "mean_actual": round(mean_actual, 4),
            "count": int(m.sum()),
            "gap": round(mean_pred - mean_actual, 4),
        })
    return rows


def calibration_buckets_pair(p_raw: np.ndarray, p_calibrated: np.ndarray,
                              y: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Build raw and calibrated reliability views from the same games.

    Bucket membership is defined by the raw favored probability. The
    calibrated value is oriented to that same favored side, rather than
    re-binning the games after calibration.
    """
    raw = np.asarray(p_raw, dtype=float)
    calibrated = np.asarray(p_calibrated, dtype=float)
    y = np.asarray(y, dtype=float)
    if not (len(raw) == len(calibrated) == len(y)):
        raise ValueError("raw, calibrated, and target arrays must have equal length")
    ok = np.isfinite(raw) & np.isfinite(calibrated) & np.isfinite(y)
    raw, calibrated, y = raw[ok], calibrated[ok], y[ok]
    raw_fav = np.maximum(raw, 1.0 - raw)
    cal_fav = np.where(raw >= 0.5, calibrated, 1.0 - calibrated)
    fav_won = np.where(raw >= 0.5, y, 1.0 - y)

    half = max(n_bins // 2, 1)
    edges = np.linspace(0.5, 1.0, half + 1)
    rows = []
    for i in range(len(edges) - 1):
        mask = (raw_fav >= edges[i]) & (raw_fav < edges[i + 1])
        if i == len(edges) - 2:
            mask |= raw_fav == edges[i + 1]
        if not mask.any():
            continue
        mean_raw = float(raw_fav[mask].mean())
        mean_cal = float(cal_fav[mask].mean())
        mean_actual = float(fav_won[mask].mean())
        rows.append({
            "bucket": f"{edges[i] * 100:.0f}–{edges[i + 1] * 100:.0f}%",
            "mean_predicted": round(mean_cal, 4),
            "mean_calibrated": round(mean_cal, 4),
            "mean_predicted_raw": round(mean_raw, 4),
            "mean_actual": round(mean_actual, 4),
            "count": int(mask.sum()),
            "gap": round(mean_raw - mean_actual, 4),
            "gap_calibrated": round(mean_cal - mean_actual, 4),
        })
    return rows


# ---------------------------------------------------------------------------
# Distributional metrics (run line + totals)
# ---------------------------------------------------------------------------
def nb_distribution_metrics(oof: pd.DataFrame, params: dict,
                             n_draws: int = 2000) -> dict:
    """Raw NB/Monte-Carlo OOF metrics at each game's model fair line."""
    if oof is None or not len(oof):
        return {"run_line": {"n": 0}, "totals": {"n": 0}}
    sim = dist_mod.simulate_distributions(
        oof["mu_h"].to_numpy(float), oof["mu_a"].to_numpy(float),
        float(params.get("alpha_home", 0.0)),
        float(params.get("alpha_away", 0.0)), n_draws=n_draws)
    margin = oof["margin"].to_numpy(float)
    total = oof["total"].to_numpy(float)
    p_cover = sim["p_cover_fair"].to_numpy(float)
    p_over = sim["p_over_fair"].to_numpy(float)
    fair_s = sim["fair_spread"].to_numpy(float)
    fair_t = sim["fair_total"].to_numpy(float)
    ok_m = np.isfinite(p_cover) & (margin != fair_s)
    ok_t = np.isfinite(p_over) & (total != fair_t)

    def _metrics(p, y):
        if not len(p):
            return {"n": 0, "n_pushes": 0, "logscore": np.nan,
                    "brier": np.nan, "ece_cover": np.nan}
        y = np.asarray(y, float)
        p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
        return {"n": int(len(y)), "n_pushes": 0,
                "logscore": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                "brier": float(np.mean((p - y) ** 2)),
                "ece_cover": ece(p, y)}

    return {
        "run_line": dict(_metrics(p_cover[ok_m], (margin[ok_m] > fair_s[ok_m]).astype(float)),
                          n_pushes=int((margin == fair_s).sum())),
        "totals": dict(_metrics(p_over[ok_t], (total[ok_t] > fair_t[ok_t]).astype(float)),
                        n_pushes=int((total == fair_t).sum())),
    }
