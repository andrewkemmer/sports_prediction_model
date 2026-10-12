"""OOF evaluation helpers for the NBA production pipeline."""
from __future__ import annotations
import numpy as np
import pandas as pd
try:
    from backend import distributions as dist_mod
except ImportError:
    import distributions as dist_mod

def ece(p, y, n_bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float); ok = np.isfinite(p) & np.isfinite(y); p, y = p[ok], y[ok]
    if not len(p): return np.nan
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1); total = len(p); err = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any(): err += m.sum() / total * abs(float(p[m].mean() - y[m].mean()))
    return float(err)

def binary_metrics(p, y):
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    p, y = np.asarray(p, float), np.asarray(y, float); ok = np.isfinite(p) & np.isfinite(y); p, y = p[ok], y[ok]
    if len(p) < 2 or len(np.unique(y)) < 2: return {"n": int(len(p)), "auc": np.nan, "logloss": np.nan, "brier": np.nan, "ece": np.nan}
    return {"n": int(len(p)), "auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0, 1])), "brier": float(brier_score_loss(y, p)), "ece": ece(p, y)}

def calibration_buckets(p, y, n_bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float); ok = np.isfinite(p) & np.isfinite(y); p, y = p[ok], y[ok]
    fp, fy = np.maximum(p, 1-p), np.where(p >= .5, y, 1-y); edges = np.linspace(.5, 1, max(n_bins // 2, 1) + 1); rows = []
    for i in range(len(edges)-1):
        m = (fp >= edges[i]) & (fp < edges[i+1]) | ((i == len(edges)-2) & (fp == edges[i+1]))
        if not m.any(): continue
        rows.append({"bucket": f"{edges[i]*100:.0f}–{edges[i+1]*100:.0f}%", "mean_predicted": round(float(fp[m].mean()), 4), "mean_actual": round(float(fy[m].mean()), 4), "count": int(m.sum()), "gap": round(float(fp[m].mean()-fy[m].mean()), 4)})
    return rows

def nb_distribution_metrics(oof, params, n_draws=2000):
    """Causal full-support CRPS; final serving params never grade old games."""
    if oof is None or not len(oof):
        return {"run_line": {"n": 0}, "totals": {"n": 0}}
    frame, bundle = dist_mod.causal_distribution_frame(oof)
    return dist_mod.distribution_score_report(frame, bundle)
