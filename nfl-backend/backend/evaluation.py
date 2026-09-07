"""OOF evaluation for the production moneyline, run-line, and totals models.

Every metric is computed on out-of-fold predictions only. No full-history
correlations, no in-sample importance, no future outcomes.
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
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        rows.append({
            "bucket": f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}",
            "n": int(m.sum()),
            "mean_pred": float(p[m].mean()),
            "mean_actual": float(y[m].mean()),
        })
    return rows


# ---------------------------------------------------------------------------
# Distributional metrics (run line + totals)
# ---------------------------------------------------------------------------
def distribution_metrics(oof: pd.DataFrame,
                         sigma_margin: float, sigma_total: float) -> dict:
    """Probabilistic scoring + calibration for the joint distribution model.

    Run line: log score + Brier of the home-cover outcome at the fair
    spread; calibration ECE of P(cover). Totals: log score + Brier of the
    over outcome at the fair total; ECE of P(over). Pushes excluded from
    the 2-way populations (reported separately), matching the frontend's
    whole-number push convention.
    """
    out: dict = {}
    ok_m = np.isfinite(oof["mu_h"].to_numpy()) & np.isfinite(oof["mu_a"].to_numpy())
    sub = oof[ok_m].reset_index(drop=True)
    if not len(sub):
        return {"run_line": {"n": 0}, "totals": {"n": 0}}

    sup_m = dist_mod.MARGIN_SUPPORT
    sup_t = dist_mod.TOTAL_SUPPORT
    y_margin = sub["margin"].to_numpy(float)
    y_total = sub["total"].to_numpy(float)

    # Run line at the fair spread (per-game PMF)
    ls_m, br_m, n_push_m = [], [], 0
    p_cover_all, cover_all = [], []
    for i, r in enumerate(sub.itertuples(index=False)):
        pmf = dist_mod.discrete_normal_pmf(
            r.mu_h - r.mu_a, sigma_margin, sup_m)
        fair = dist_mod._pmf_median(pmf, sup_m)
        if not np.isfinite(fair):
            continue
        p_cover = dist_mod.margin_cdf_above(pmf, sup_m, fair)
        p_push = dist_mod.margin_pmf_at(pmf, sup_m, fair)
        y_cover = 1.0 if y_margin[i] > fair else 0.0
        if y_margin[i] == fair:
            n_push_m += 1
            continue
        eps = 1e-12
        yv = 1.0 if y_margin[i] > fair else 0.0
        ls_m.append(-(yv * np.log(max(p_cover, eps))
                      + (1 - yv) * np.log(max(1 - p_cover, eps))))
        br_m.append((p_cover - yv) ** 2)
        p_cover_all.append(p_cover)
        cover_all.append(yv)
    rl = {
        "n": len(ls_m), "n_pushes": n_push_m,
        "logscore": float(np.mean(ls_m)) if ls_m else np.nan,
        "brier": float(np.mean(br_m)) if br_m else np.nan,
        "ece_cover": ece(np.array(p_cover_all), np.array(cover_all)) if ls_m else np.nan,
    }

    # Totals at the fair total
    ls_t, br_t, n_push_t = [], [], 0
    p_over_all, over_all = [], []
    for i, r in enumerate(sub.itertuples(index=False)):
        pmf = dist_mod.discrete_normal_pmf(
            r.mu_h + r.mu_a, sigma_total, sup_t)
        fair = dist_mod._pmf_median(pmf, sup_t)
        if not np.isfinite(fair):
            continue
        p_over, p_push, _pu = dist_mod.total_probabilities(pmf, sup_t, fair)
        if y_total[i] == fair:
            n_push_t += 1
            continue
        yv = 1.0 if y_total[i] > fair else 0.0
        eps = 1e-12
        ls_t.append(-(yv * np.log(max(p_over, eps))
                      + (1 - yv) * np.log(max(1 - p_over, eps))))
        br_t.append((p_over - yv) ** 2)
        p_over_all.append(p_over)
        over_all.append(yv)
    tt = {
        "n": len(ls_t), "n_pushes": n_push_t,
        "logscore": float(np.mean(ls_t)) if ls_t else np.nan,
        "brier": float(np.mean(br_t)) if br_t else np.nan,
        "ece_over": ece(np.array(p_over_all), np.array(over_all)) if ls_t else np.nan,
    }
    return {"run_line": rl, "totals": tt}


def margin_calibration_table(oof: pd.DataFrame, sigma_margin: float,
                             n_bins: int = 8) -> list[dict]:
    """Predicted-vs-observed P(home covers the fair spread) by mu_margin bin."""
    rows = []
    ok = np.isfinite(oof["mu_h"].to_numpy()) & np.isfinite(oof["mu_a"].to_numpy())
    sub = oof[ok].reset_index(drop=True)
    if not len(sub):
        return rows
    sub = sub.assign(mu_margin=sub["mu_h"] - sub["mu_a"])
    sup = dist_mod.MARGIN_SUPPORT
    bins = pd.qcut(sub["mu_margin"], n_bins, duplicates="drop")
    for b, grp in sub.groupby(bins, observed=True):
        p_c, y_c = [], []
        for r in grp.itertuples(index=False):
            pmf = dist_mod.discrete_normal_pmf(r.mu_margin, sigma_margin, sup)
            fair = dist_mod._pmf_median(pmf, sup)
            if not np.isfinite(fair):
                continue
            if r.margin == fair:
                continue  # push excluded from the 2-way population
            p_c.append(dist_mod.margin_cdf_above(pmf, sup, fair))
            y_c.append(1.0 if r.margin > fair else 0.0)
        if not p_c:
            continue
        rows.append({
            "mu_margin_bin": str(b), "n": len(p_c),
            "mean_pred_cover": float(np.mean(p_c)),
            "obs_cover_rate": float(np.mean(y_c)),
        })
    return rows


def total_calibration_table(oof: pd.DataFrame, sigma_total: float,
                            n_bins: int = 8) -> list[dict]:
    """Predicted-vs-observed P(total > fair total) by mu_total bin."""
    rows = []
    ok = np.isfinite(oof["mu_h"].to_numpy()) & np.isfinite(oof["mu_a"].to_numpy())
    sub = oof[ok].reset_index(drop=True)
    if not len(sub):
        return rows
    sub = sub.assign(mu_total=sub["mu_h"] + sub["mu_a"])
    sup = dist_mod.TOTAL_SUPPORT
    bins = pd.qcut(sub["mu_total"], n_bins, duplicates="drop")
    for b, grp in sub.groupby(bins, observed=True):
        p_o, y_o = [], []
        for r in grp.itertuples(index=False):
            pmf = dist_mod.discrete_normal_pmf(r.mu_total, sigma_total, sup)
            fair = dist_mod._pmf_median(pmf, sup)
            if not np.isfinite(fair):
                continue
            if r.total == fair:
                continue
            p_o.append(dist_mod.total_probabilities(pmf, sup, fair)[0])
            y_o.append(1.0 if r.total > fair else 0.0)
        if not p_o:
            continue
        rows.append({
            "mu_total_bin": str(b), "n": len(p_o),
            "mean_pred_over": float(np.mean(p_o)),
            "obs_over_rate": float(np.mean(y_o)),
        })
    return rows
