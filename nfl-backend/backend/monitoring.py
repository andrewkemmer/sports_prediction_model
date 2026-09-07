"""Production monitoring / diagnostics.

Emits the ``nfl_model_monitor_<date>.json`` record the shared frontend
monitor page renders (MLB-shaped schema): feature drift (PSI), coverage,
ensemble member diagnostics, rolling OOF Brier, and version history.

All statistics derive from OOF predictions and point-in-time features —
never from full-history target relationships used for selection.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

try:
    from backend import config
except ImportError:
    import config

logger = logging.getLogger(__name__)

PSI_WARN = 0.10
PSI_ALERT = 0.25


def _psi(current: np.ndarray, baseline: np.ndarray, n_bins: int = 10) -> float:
    """Population stability index between the current-window and baseline
    distributions of a feature (quantile-binned on the baseline)."""
    c = pd.Series(current).dropna()
    b = pd.Series(baseline).dropna()
    if len(c) < 10 or len(b) < 10:
        return np.nan
    try:
        qs = np.unique(np.quantile(b, np.linspace(0, 1, n_bins + 1)))
    except Exception:
        return np.nan
    if len(qs) < 2:
        return np.nan
    qb = np.clip(np.searchsorted(qs, b, side="right") - 1, 0, len(qs) - 2)
    qc = np.clip(np.searchsorted(qs, c, side="right") - 1, 0, len(qs) - 2)
    pb = np.bincount(qb, minlength=len(qs) - 1) / len(b)
    pc = np.bincount(qc, minlength=len(qs) - 1) / len(c)
    pb, pc = np.clip(pb, 1e-6, None), np.clip(pc, 1e-6, None)
    return float(np.sum((pc - pb) * np.log(pc / pb)))


def feature_drift(full_df: pd.DataFrame, recent_df: pd.DataFrame) -> list[dict]:
    """PSI per served feature: recent slate window vs full-history baseline."""
    rows = []
    for f in config.FEATURE_COLUMNS:
        if f not in full_df.columns:
            continue
        psi = _psi(recent_df[f].to_numpy(float), full_df[f].to_numpy(float))
        rows.append({
            "feature": f,
            "current_mean": (float(np.nanmean(recent_df[f])) if len(recent_df) else np.nan),
            "baseline_mean": float(np.nanmean(full_df[f])),
            "psi": psi,
            "psi_adjusted": psi,
        })
    return rows


def coverage(full_df: pd.DataFrame) -> list[dict]:
    rows = []
    for f in config.FEATURE_COLUMNS:
        if f not in full_df.columns:
            rows.append({"feature": f, "window": "decided pool",
                         "n_games": len(full_df), "pct_measured": 0.0,
                         "pct_nonnull": 0.0})
            continue
        v = pd.to_numeric(full_df[f], errors="coerce")
        rows.append({
            "feature": f, "window": "decided pool", "n_games": int(len(full_df)),
            "pct_measured": round(100.0 * float(v.notna().mean()), 2),
            "pct_nonnull": round(100.0 * float(v.notna().mean()), 2),
        })
    return rows


def ensemble_table(oof: pd.DataFrame, weights: dict[str, float],
                   cal_p: np.ndarray | None = None) -> list[dict]:
    """Per-member OOF diagnostics + earned adaptive weights."""
    y = oof["home_win"].to_numpy(float)
    rows = []
    n = len(oof)
    for name in config.ENSEMBLE_MEMBERS:
        col = f"p_{name}"
        if col not in oof.columns:
            continue
        from evaluation import binary_metrics  # local import avoids cycles
        m = binary_metrics(oof[col].to_numpy(float), y)
        rows.append({
            "name": name, "weight": round(float(weights.get(name, 0.0)), 4),
            "auc": m.get("auc"), "brier": m.get("brier"),
            "logloss": m.get("logloss"), "n_eval": m.get("n"),
        })
    return rows


def rolling_brier(oof: pd.DataFrame, p_col: str = "p_ensemble_calibrated",
                  window_days: int = 30) -> list[dict]:
    """Per-game rolling Brier over the OOF timeline."""
    if p_col not in oof.columns:
        return []
    df = oof.dropna(subset=[p_col]).copy()
    df["gameday"] = pd.to_datetime(df["gameday"])
    df = df.sort_values("gameday")
    df["brier"] = (df[p_col] - df["home_win"]) ** 2
    out = []
    for day, grp in df.groupby(df["gameday"].dt.date):
        out.append({"date": str(day), "brier": float(grp["brier"].mean())})
    return out


def _dump_json(path, record: dict) -> None:
    """JSON-strict dump: NaN/Inf -> None (serving parity with serving.py)."""
    def _safe(o):
        if isinstance(o, dict):
            return {k: _safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_safe(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    path.write_text(json.dumps(_safe(record), indent=1, allow_nan=False,
                               default=str))


def write_monitor_json(path, run_date: str, drift: list[dict],
                       cov: list[dict], ensemble: list[dict],
                       rb: list[dict], baseline: float,
                       config_meta: dict, fold_info: dict) -> dict:
    baseline_label = "Constant home-edge" if np.isfinite(baseline) else "n/a"
    record = {
        "last_retrained": run_date,
        "last_retrained_note": "Fresh ensemble trained this run (expanding walk-forward OOF)",
        "next_retrain": run_date,
        "next_retrain_note": "retrains every run",
        "upset_note": (f"Model upset rate over the walk-forward history — "
                       f"{fold_info.get('total_val_games', 0)} OOF games scored"),
        "feature_drift": drift,
        "features_metadata": {r["feature"]: {"definition": "see backend/manifest.py",
                                             "source": "nflverse / stadiums table"}
                              for r in cov},
        "feature_coverage": cov,
        "ensemble": ensemble,
        "rolling_brier": rb,
        "brier_baseline": baseline,
        "brier_baseline_label": baseline_label,
        "rolling_brier_meta": {"window": "per OOF game day",
                               "n_days": len(rb)},
        "version_history": [{
            "version": run_date, "date": run_date,
            "weights": {r["name"]: r["weight"] for r in ensemble},
            "auc": (ensemble[0].get("auc") if ensemble else None),
            "note": "rebuild run",
        }],
        "fold_geometry": fold_info,
        "config": config_meta,
    }
    _dump_json(path, record)
    return record
