"""NBA production monitoring and drift/coverage artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from backend import config
except ImportError:
    import config

PSI_WARN, PSI_ALERT = 0.10, 0.25


def feature_status(psi: float | None) -> str:
    if psi is None or not np.isfinite(psi):
        return "OK"
    return "ALERT" if psi >= PSI_ALERT else "WARN" if psi >= PSI_WARN else "OK"


def _psi(current, baseline, bins: int = 10) -> float:
    c, b = pd.Series(current).dropna(), pd.Series(baseline).dropna()
    if len(c) < 10 or len(b) < 10:
        return np.nan
    q = np.unique(np.quantile(b, np.linspace(0, 1, bins + 1)))
    if len(q) < 2:
        return np.nan
    bi = np.clip(np.searchsorted(q, b, side="right") - 1, 0, len(q) - 2)
    ci = np.clip(np.searchsorted(q, c, side="right") - 1, 0, len(q) - 2)
    pb = np.bincount(bi, minlength=len(q) - 1) / len(b)
    pc = np.bincount(ci, minlength=len(q) - 1) / len(c)
    pb, pc = np.clip(pb, 1e-6, None), np.clip(pc, 1e-6, None)
    return float(np.sum((pc - pb) * np.log(pc / pb)))


def coverage(df: pd.DataFrame) -> list[dict]:
    rows = []
    for feature in config.active_moneyline_feature_cols():
        values = pd.to_numeric(df[feature], errors="coerce") if feature in df else pd.Series(dtype=float)
        pct = round(100 * float(values.notna().mean()), 2) if len(values) else 0.0
        rows.append({"feature": feature, "window": "decided pool", "n_games": int(len(df)),
                     "pct_measured": pct, "pct_nonnull": pct,
                     "n_default_zero": int((values == 0).sum()) if len(values) else 0,
                     "status": "STARVED" if pct < 25 else "LOW_COVERAGE" if pct < 80 else "OK"})
    return rows


def feature_drift(full: pd.DataFrame, recent: pd.DataFrame,
                  weights: dict[str, float] | None = None) -> list[dict]:
    out = []
    for feature in config.active_moneyline_feature_cols():
        if feature not in full:
            continue
        psi = _psi(recent[feature].to_numpy(float) if feature in recent else [],
                   full[feature].to_numpy(float))
        current = pd.to_numeric(recent[feature], errors="coerce") if feature in recent else pd.Series(dtype=float)
        baseline = pd.to_numeric(full[feature], errors="coerce")
        out.append({"feature": feature,
                    "current_mean": float(current.mean()) if len(current) else np.nan,
                    "baseline_mean": float(baseline.mean()) if len(baseline) else np.nan,
                    "psi": psi, "psi_adjusted": psi,
                    "status": feature_status(psi),
                    "weight_pct": round(100 * float((weights or {}).get(feature, 0)), 2) if weights else None,
                    "n_baseline": int(baseline.notna().sum()),
                    "n_current": int(current.notna().sum())})
    return out


def write_run_engine_feature_artifacts(out_dir, date_c: str, full: pd.DataFrame,
                                      recent: pd.DataFrame, weights=None) -> tuple[str, str]:
    out_dir = Path(out_dir)
    drift_name = f"{config.RUN_ENGINE_FEATURE_DRIFT_PREFIX}{date_c}.csv"
    coverage_name = f"{config.RUN_ENGINE_FEATURE_COVERAGE_PREFIX}{date_c}.csv"
    pd.DataFrame(feature_drift(full, recent, weights)).to_csv(out_dir / drift_name, index=False)
    pd.DataFrame(coverage(full)).to_csv(out_dir / coverage_name, index=False)
    return drift_name, coverage_name


def ensemble_table(oof: pd.DataFrame, weights: dict[str, float]) -> list[dict]:
    try:
        from backend.evaluation import binary_metrics
    except ImportError:
        from evaluation import binary_metrics
    y = oof.home_win.to_numpy(float)
    rows = []
    for name in config.ENSEMBLE_MEMBERS:
        col = f"p_{name}"
        if col not in oof:
            continue
        metrics = binary_metrics(oof[col].to_numpy(float), y)
        rows.append({
            "name": name,
            "weight": round(float(weights.get(name, 0)), 4),
            "n_eval": int(metrics.get("n", 0) or 0),
            **{key: metrics.get(key) for key in ("auc", "brier", "logloss")},
        })
    return rows


def rolling_brier(oof: pd.DataFrame, p_col="p_ensemble_calibrated",
                  window_days: int = 30) -> list[dict]:
    if oof is None or p_col not in oof or not len(oof):
        return []
    frame = oof.dropna(subset=[p_col, "home_win"]).copy()
    frame["gameday"] = pd.to_datetime(frame.gameday, errors="coerce")
    frame["brier"] = (pd.to_numeric(frame[p_col], errors="coerce") - frame.home_win) ** 2
    rows = []
    for day, group in frame.dropna(subset=["gameday"]).sort_values("gameday").groupby(frame.gameday.dt.date):
        rows.append({"date": str(day), "brier": float(group.brier.mean()), "games": int(len(group))})
    return rows


def _json_safe(value):
    """Convert pandas/numpy values and non-finite floats to strict JSON."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _dump(path, record: dict) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(record), indent=1, allow_nan=False,
                               default=str))
    return record


def write_monitor_json(path, date_c: str, drift, cov, members, rolling,
                       baseline, config_meta=None, fold_info=None,
                       metrics=None, platt=None) -> dict:
    drift = list(drift or [])
    cov = list(cov or [])
    feature_alerts = [row for row in drift if row.get("status") != "OK"]
    coverage_alerts = [row for row in cov if row.get("status") != "OK"]
    # Shared frontend monitor cards use these explicit fields.
    retrain = str(date_c)
    next_retrain = (
        pd.Timestamp(retrain) + pd.Timedelta(days=config.RETRAIN_CADENCE_DAYS)
    ).strftime("%Y%m%d")
    record = {
        "created_utc": pd.Timestamp.utcnow().isoformat(), "date": date_c,
        "config": config_meta or {},
        "last_retrained": retrain,
        "last_retrained_note": "Fresh NBA model trained this run",
        "next_retrain": next_retrain,
        "next_retrain_note": (
            f"next expected run in {config.RETRAIN_CADENCE_DAYS} day(s)"
        ),
        "upset_note": "NBA upset rate is computed from settled walk-forward history.",
        "alerts": {"retrain": False, "feature_drift": feature_alerts, "coverage": coverage_alerts},
        "baseline_brier": baseline, "brier_baseline": baseline,
        "brier_baseline_label": "Constant home-edge",
        "metrics": metrics or {}, "calibration": platt or {},
        "ensemble": members or [], "rolling_brier": rolling or [],
        "feature_drift": drift, "coverage": cov,
        "feature_coverage": cov, "folds": fold_info or {},
        "version_history": [], "features_metadata": {},
    }
    return _dump(path, record)


def write_run_engine_monitor(path, date_c: str, metrics=None, markets=None,
                             calibration=None, config_meta=None,
                             slate_history=None) -> dict:
    record = {
        "created_utc": pd.Timestamp.utcnow().isoformat(), "date": date_c,
        "config": config_meta or {}, "winner_cards": metrics or {},
        "market_metrics": metrics or {}, "calibration_cards": calibration or {},
        "slate_history": slate_history or [],
        "fit": {"distribution": "negative_binomial", "mc_draws": 4000, "seed": 42,
                "spread_grid": [min(config.SPREAD_GRID), max(config.SPREAD_GRID)],
                "total_grid": [min(config.TOTAL_GRID), max(config.TOTAL_GRID)],
                "half_stops": list(config.HALF_STOP_LINES)},
    }
    return _dump(path, record)
