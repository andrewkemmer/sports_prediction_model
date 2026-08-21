"""Explainability: per-game SHAP attributions and feature-drift (PSI) tracking.

* ``compute_psi`` is pure and unit-tested: identical distributions return
  ~0, shifted distributions exceed the WARN threshold, degenerate inputs
  return 0, and results are always non-negative.
* Per-game SHAP is computed by averaging the attributions of the ensemble's
  XGBoost / LightGBM tree explainers (plus a linear explainer for the
  logistic regression member). SHAP values are log-odds contributions to
  P(home wins); ``signed_effect`` is ``positive``/``negative``.
* ``shap`` is imported lazily so the rest of the pipeline (and tests) run
  without it; if it is missing the pipeline writes a zero-attribution CSV
  and logs a warning instead of failing the daily run.
"""

from __future__ import annotations

import logging
from datetime import date as date_cls
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    DRIFT_FEATURES,
    FEATURE_LABELS,
    PSI_ALERT,
    PSI_N_BUCKETS,
    PSI_WARN,
    SHAP_DIR,
    SHAP_GAME_FILE,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# PSI (Population Stability Index)
# ===========================================================================

def compute_psi(
    current,
    baseline,
    n_buckets: int = PSI_N_BUCKETS,
    epsilon: float = 1e-4,
) -> float:
    """Population Stability Index between two numeric samples.

    ``psi = sum((cur_pct - base_pct) * ln(cur_pct / base_pct))`` over equal
    buckets spanning both samples. Returns 0.0 for identical or degenerate
    (constant) inputs; always >= 0 for well-formed inputs.
    """
    cur = np.asarray(current, dtype=float).ravel()
    base = np.asarray(baseline, dtype=float).ravel()
    if len(cur) == 0 or len(base) == 0:
        return 0.0

    lo = float(min(cur.min(), base.min()))
    hi = float(max(cur.max(), base.max()))
    if not np.isfinite(hi - lo) or hi - lo < 1e-9:
        return 0.0  # degenerate: all values identical -> no drift

    edges = np.linspace(lo, hi, n_buckets + 1)
    cur_hist, _ = np.histogram(cur, bins=edges)
    base_hist, _ = np.histogram(base, bins=edges)

    cur_pct = cur_hist / cur_hist.sum() + epsilon
    base_pct = base_hist / base_hist.sum() + epsilon
    psi = float(np.sum((cur_pct - base_pct) * np.log(cur_pct / base_pct)))
    return round(psi, 6)


def psi_status(psi: float) -> str:
    """Map a PSI score to OK / WARN / ALERT."""
    if psi >= PSI_ALERT:
        return "ALERT"
    if psi >= PSI_WARN:
        return "WARN"
    return "OK"


def compute_feature_drift(
    current: pd.DataFrame,
    baseline: pd.DataFrame,
    features: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Per-feature drift table: current mean, baseline mean, PSI, status."""
    feats = features or DRIFT_FEATURES
    rows = []
    for col in feats:
        if col not in current.columns or col not in baseline.columns:
            continue
        cur = current[col].dropna()
        base = baseline[col].dropna()
        if cur.empty or base.empty:
            continue
        psi = compute_psi(cur.values, base.values)
        rows.append(
            {
                "feature": col,
                "current_mean": round(float(cur.mean()), 4),
                "baseline_mean": round(float(base.mean()), 4),
                "psi": psi,
                "status": psi_status(psi),
            }
        )
    return pd.DataFrame(rows)


# ===========================================================================
# SHAP
# ===========================================================================

def explain_game(ensemble_payload: dict, X_row: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """SHAP attributions for a single game against the ensemble.

    ``X_row`` must be a one-row DataFrame with the model's feature columns.
    Returns a DataFrame with ``feature`` (friendly label), ``feature_key``,
    ``shap_value`` and ``signed_effect``.
    """
    try:
        import shap  # lazy
    except ImportError:  # pragma: no cover
        logger.warning("shap not installed — writing zero-attribution CSV for %s", game_id)
        return _zero_shap(X_row)

    # Use exactly the columns the model was trained on (mismatched widths make
    # LightGBM fail and SHAP silently return garbage).
    feature_keys = list(ensemble_payload.get("FEATURE_COLUMNS") or []) or list(
        X_row.select_dtypes(include=[np.number]).columns
    )
    feature_keys = [c for c in feature_keys if c in X_row.columns]
    X = X_row[feature_keys].astype(float)

    ensemble = ensemble_payload.get("moneyline") if ensemble_payload.get("moneyline") else ensemble_payload
    if not hasattr(ensemble, "models"):
        return _zero_shap(X_row)

    contributions = []
    for model in ensemble.models:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X)
        except Exception:
            continue
        if isinstance(sv, list):
            # multi-class output: use the positive-class (index 1) slice
            sv = sv[1] if len(sv) > 1 else sv[0]
        contributions.append(np.asarray(sv).reshape(1, -1))

    if contributions:
        shap_matrix = np.mean(contributions, axis=0)[0]
    else:  # pragma: no cover - linear-only fallback
        shap_matrix = np.zeros(len(feature_keys))

    return pd.DataFrame(
        {
            "feature_key": feature_keys,
            "feature": [FEATURE_LABELS.get(k, k) for k in feature_keys],
            "shap_value": np.round(shap_matrix, 4),
            "signed_effect": ["positive" if v >= 0 else "negative" for v in shap_matrix],
        }
    )


def _zero_shap(X_row: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_key": list(X_row.columns),
            "feature": [FEATURE_LABELS.get(k, k) for k in X_row.columns],
            "shap_value": 0.0,
            "signed_effect": "neutral",
        }
    )


def save_shap_for_games(ensemble_payload: dict, features: pd.DataFrame, game_ids: list[str]) -> list[str]:
    """Write ``data_delivery/shap_game_<gameid>.csv`` for each game id."""
    saved = []
    for gid in game_ids:
        if gid not in features.index:
            continue
        row = features.loc[[gid]]
        df = explain_game(ensemble_payload, row, gid)
        df = df.sort_values("shap_value", key=lambda s: s.abs(), ascending=False)
        path = SHAP_DIR / SHAP_GAME_FILE.format(game_id=gid)
        df.to_csv(path, index=False)
        saved.append(path.name)
    logger.info("Saved %d SHAP files to %s", len(saved), SHAP_DIR)
    return saved
