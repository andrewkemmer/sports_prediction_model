"""
Explainability for MLB Bet Predictor.

Provides per-game SHAP attributions (averaged across ensemble members)
and PSI (Population Stability Index) feature-drift computation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from config import (
    DATA_DELIVERY_DIR,
    DATE_FMT,
    FEATURE_DRIFT,
    PSI_ALERT_THRESHOLD,
    PSI_WARN_THRESHOLD,
    RANDOM_SEED,
    SHAP_GAME,
)
from training import FEATURE_COLS

logger = logging.getLogger(__name__)


# ── SHAP per-game attributions ──────────────────────────────────────────────

def compute_shap_per_game(
    models: dict[str, Any],
    games: pd.DataFrame,
) -> None:
    """Compute and save SHAP attributions for each game.

    Averages TreeExplainer values across XGBoost and LightGBM members
    in log-odds space. If shap is unavailable, writes zero-attribution CSVs.

    Output: data_delivery/shap_game_<game_id>.csv
    """
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import shap
        has_shap = True
    except ImportError:
        has_shap = False
        logger.warning("shap not available; writing zero-attribution CSVs")

    cols = [c for c in FEATURE_COLS if c in games.columns]
    # Preserve NaN: tree explainers handle missing values natively and a
    # zero-fill would fabricate attributions for unobserved features.
    X = games[cols].to_numpy(dtype=float)

    for idx, row in games.iterrows():
        game_id = row["game_id"]
        shap_values = {}

        if has_shap:
            # Collect SHAP from tree-based models
            tree_shaps = []
            for name, model in models.items():
                if name in ("scaler", "logistic"):
                    continue
                try:
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(X[idx:idx + 1])
                    if isinstance(sv, list):
                        sv = sv[1]  # For binary classification, class 1
                    tree_shaps.append(sv.flatten())
                except Exception as e:
                    logger.debug("SHAP failed for %s on model %s: %s", game_id, name, e)

            if tree_shaps:
                avg_shap = np.mean(tree_shaps, axis=0)
                for i, col in enumerate(cols):
                    shap_values[col] = round(float(avg_shap[i]), 6)
            else:
                for col in cols:
                    shap_values[col] = 0.0
        else:
            for col in cols:
                shap_values[col] = 0.0

        # Sort by absolute SHAP value (descending)
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)

        rows = []
        for feat, val in sorted_features:
            rows.append({
                "feature": feat,
                "shap_value": val,
                "signed_effect": "positive" if val > 0 else "negative",
            })

        df = pd.DataFrame(rows)
        out_path = DATA_DELIVERY_DIR / f"{SHAP_GAME}_{game_id}.csv"
        df.to_csv(out_path, index=False)

    logger.info("SHAP attributions written for %d games", len(games))


# ── PSI (Population Stability Index) ────────────────────────────────────────

def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index between two distributions.

    PSI = sum((current_pct - baseline_pct) * ln(current_pct / baseline_pct))

    Implementation notes (fixed after false-alert investigation):
    - Bin edges are QUANTILES of the combined sample (deduplicated), not
      equal-width slices of the range. Equal-width bins let a single outlier
      stretch the range so most edge bins end up empty on one side.
    - Empty bins are handled with add-one-half smoothing instead of an
      epsilon of 1e-10. The old epsilon made one empty bin contribute ~+2.0
      to PSI by itself, flagging stable features as ALERT.

    Returns a non-negative float. 0 means identical distributions.
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    # Remove NaN
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    combined = np.concatenate([baseline, current])
    min_val, max_val = combined.min(), combined.max()

    if min_val == max_val:
        return 0.0

    # Quantile bin edges from the combined sample; deduplicate so sparse or
    # heavily-discrete features don't produce zero-width bins.
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.unique(np.quantile(combined, quantiles))
    if len(bin_edges) < 2:
        return 0.0
    bin_edges[-1] = max_val + 1e-10  # Include right edge

    baseline_counts = np.histogram(baseline, bins=bin_edges)[0].astype(float)
    current_counts = np.histogram(current, bins=bin_edges)[0].astype(float)

    # Add-one-half smoothing keeps empty bins bounded and the term-wise
    # contribution (c - b) * ln(c / b) always >= 0.
    k = len(bin_edges) - 1
    baseline_pct = (baseline_counts + 0.5) / (baseline_counts.sum() + 0.5 * k)
    current_pct = (current_counts + 0.5) / (current_counts.sum() + 0.5 * k)

    psi = float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))

    return round(max(psi, 0.0), 6)


def psi_status(psi_value: float) -> str:
    """Map PSI value to status: OK, WARN, or ALERT."""
    if psi_value >= PSI_ALERT_THRESHOLD:
        return "ALERT"
    elif psi_value >= PSI_WARN_THRESHOLD:
        return "WARN"
    return "OK"


def compute_feature_drift(
    baseline_games: pd.DataFrame,
    current_games: pd.DataFrame,
    target_date_str: str,
) -> pd.DataFrame:
    """Compute PSI for each numeric feature and save feature_drift CSV.

    Output: data_delivery/feature_drift_YYYYMMDD.csv
    """
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)

    drift_rows = []
    for col in FEATURE_COLS:
        if col not in baseline_games.columns or col not in current_games.columns:
            continue

        baseline_vals = baseline_games[col].dropna().values
        current_vals = current_games[col].dropna().values

        if len(baseline_vals) == 0 or len(current_vals) == 0:
            continue

        psi = compute_psi(baseline_vals, current_vals)

        # Small windows make PSI statistically meaningless — report them as
        # INSUFFICIENT rather than WARN/ALERT so they never page anyone.
        if len(baseline_vals) < 100 or len(current_vals) < 30:
            status = "INSUFFICIENT"
        else:
            status = psi_status(psi)

        drift_rows.append({
            "feature": col,
            "current_mean": round(float(current_vals.mean()), 4),
            "baseline_mean": round(float(baseline_vals.mean()), 4),
            "psi": psi,
            "status": status,
            "n_baseline": int(len(baseline_vals)),
            "n_current": int(len(current_vals)),
        })

    df = pd.DataFrame(drift_rows)
    out_path = DATA_DELIVERY_DIR / f"{FEATURE_DRIFT}_{target_date_str}.csv"
    df.to_csv(out_path, index=False)

    n_warns = (df["status"] == "WARN").sum()
    n_alerts = (df["status"] == "ALERT").sum()
    logger.info(
        "Feature drift: %d features, %d warnings, %d alerts",
        len(df), n_warns, n_alerts,
    )

    return df
