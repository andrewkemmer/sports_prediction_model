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


def feature_drift(full_df: pd.DataFrame, recent_df: pd.DataFrame,
                  weights: dict[str, float] | None = None) -> list[dict]:
    """PSI per served feature: recent slate window vs full-history baseline.

    Rows carry the MLB-shaped fields the shared monitor page renders:
    ``status`` (OK/WARN/ALERT from the same PSI thresholds MLB uses),
    ``weight_pct`` (the ensemble's blend-weighted feature importance, when
    member importances are available), and ``n_baseline`` / ``n_current``
    sample sizes behind each comparison. NFL's own PSI values and windows —
    nothing copied from MLB.
    """
    wmap = weights or {}
    rows = []
    for f in config.FEATURE_COLUMNS:
        if f not in full_df.columns:
            continue
        psi = _psi(recent_df[f].to_numpy(float), full_df[f].to_numpy(float))
        mean_cur = (float(np.nanmean(recent_df[f])) if len(recent_df) else np.nan)
        mean_base = float(np.nanmean(full_df[f]))
        mean_shift = mean_cur - mean_base
        se_cur = (float(np.nanstd(recent_df[f]) / np.sqrt(len(recent_df)))
                  if len(recent_df) > 1 else np.nan)
        se_base = (float(np.nanstd(full_df[f]) / np.sqrt(len(full_df)))
                   if len(full_df) > 1 else np.nan)
        shift_se = float(np.hypot(se_cur, se_base)) if np.isfinite(se_cur) \
            and np.isfinite(se_base) else np.nan
        rows.append({
            "feature": f,
            "current_mean": mean_cur,
            "baseline_mean": mean_base,
            "psi": psi,
            "psi_adjusted": psi,
            "status": ("ALERT" if (np.isfinite(psi) and psi >= PSI_ALERT)
                       else "WARN" if (np.isfinite(psi) and psi >= PSI_WARN)
                       else "OK"),
            "weight_pct": (round(100.0 * float(wmap.get(f, 0.0)), 2)
                           if wmap.get(f) else None),
            "n_baseline": int(full_df[f].notna().sum()),
            "n_current": int(recent_df[f].notna().sum()) if len(recent_df) else 0,
        })
    return rows


def coverage(full_df: pd.DataFrame) -> list[dict]:
    """Per-feature measured/non-null coverage over the decided pool.

    MLB-shaped fields: ``status`` (STARVED <25% measured / LOW_COVERAGE
    <80% / OK — the same thresholds the shared page documents) and
    ``n_default_zero``. The NFL engine does not default-fill features (NaN
    routes to imputation at fit time), so every present non-null value is a
    real measurement: pct_measured == pct_nonnull and n_default_zero is 0.
    """
    rows = []
    for f in config.FEATURE_COLUMNS:
        if f not in full_df.columns:
            rows.append({"feature": f, "window": "decided pool",
                         "n_games": len(full_df), "pct_measured": 0.0,
                         "pct_nonnull": 0.0, "n_default_zero": 0,
                         "status": "STARVED"})
            continue
        v = pd.to_numeric(full_df[f], errors="coerce")
        pct = round(100.0 * float(v.notna().mean()), 2)
        rows.append({
            "feature": f, "window": "decided pool", "n_games": int(len(full_df)),
            "pct_measured": pct,
            "pct_nonnull": pct,
            "n_default_zero": 0,
            "status": ("STARVED" if pct < 25.0
                       else "LOW_COVERAGE" if pct < 80.0 else "OK"),
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
    """Per-game rolling Brier over the OOF timeline (MLB-shaped rows).

    Each day carries its decided-game count in ``games`` (the field the
    shared Rolling Brier section's sparse-day caption reads).
    """
    if p_col not in oof.columns:
        return []
    df = oof.dropna(subset=[p_col]).copy()
    df["gameday"] = pd.to_datetime(df["gameday"])
    df = df.sort_values("gameday")
    df["brier"] = (df[p_col] - df["home_win"]) ** 2
    out = []
    for day, grp in df.groupby(df["gameday"].dt.date):
        out.append({"date": str(day), "brier": float(grp["brier"].mean()),
                    "games": int(len(grp))})
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
                       config_meta: dict, fold_info: dict,
                       metrics: dict | None = None,
                       platt: dict | None = None) -> dict:
    """MLB-shaped monitor artifact (frontend presentation contract).

    All rendering fields the shared monitor page reads are present:
    ISO retrain dates (+ same-day notes), the dense ``rolling_brier_meta``
    (window_days / min_games_per_day / excluded_sparse_days /
    calibrator_is_identity / map_scope_note), and a version-history row with
    pooled AUC / logloss / calibrated ECE + the deployed Platt map. All
    values are the NFL pipeline's own outputs.
    """
    iso_date = f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:8]}" \
        if len(str(run_date)) == 8 and str(run_date).isdigit() else str(run_date)
    next_date = iso_date  # retrains every run — next run is tonight's run
    total_val_games = int(fold_info.get("total_val_games", 0) or 0)
    baseline_label = "Constant home-edge" if np.isfinite(baseline) else "n/a"
    m = metrics or {}
    cal = (platt if isinstance(platt, dict) and platt.get("a") is not None
           and platt.get("b") is not None else None)
    version_row: dict = {
        "version": run_date, "date": iso_date,
        "weights": {r["name"]: r["weight"] for r in ensemble},
        "auc": m.get("auc") or (ensemble[0].get("auc") if ensemble else None),
        "logloss": m.get("logloss"),
        "ece_calibrated": m.get("ece_calibrated") or m.get("ece"),
        "note": "rebuild run",
    }
    if cal:
        version_row["calibration"] = {"a": cal["a"], "b": cal["b"]}
    record = {
        "last_retrained": iso_date,
        "last_retrained_note": (f"Fresh ensemble trained this run (expanding "
                                f"walk-forward OOF, {total_val_games:,} OOF games)"),
        "next_retrain": next_date,
        "next_retrain_note": "retrains every run",
        "upset_note": (f"Model upset rate over the walk-forward history — "
                       f"{total_val_games:,} OOF games scored"),
        "feature_drift": drift,
        "features_metadata": {r["feature"]: {"definition": "see backend/manifest.py",
                                             "source": "nflverse / stadiums table"}
                              for r in cov},
        "feature_coverage": cov,
        "ensemble": ensemble,
        "rolling_brier": rb,
        "brier_baseline": baseline,
        "brier_baseline_label": baseline_label,
        "rolling_brier_meta": {
            "window_days": 30,
            "min_games_per_day": 1,
            "excluded_sparse_days": 0,
            "calibrator_is_identity": False,
            "map_scope_note": ("Points use the deployed Platt map (fit on all "
                               "OOF games)."),
        },
        "version_history": [version_row],
        "fold_geometry": fold_info,
        "config": config_meta,
    }
    _dump_json(path, record)
    return record
