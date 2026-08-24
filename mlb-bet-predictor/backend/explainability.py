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
from training import (
    FEATURE_COLS,
    TREE_CATEGORICAL_COLS,
    UNK_TEAM_ID,
    _add_team_ids,
    _categorical_matrix,
    _impute_median,
    _tree_dataframe,
)

logger = logging.getLogger(__name__)


# ── SHAP per-game attributions ──────────────────────────────────────────────

def _shap_vector(sv, n_cols: int):
    """Normalize one member's explainer output to a length-n_cols vector.

    Different explainers return different shapes for binary classification:
      - XGBoost/LightGBM: (1, n_features) or list [class0, class1]
      - sklearn RandomForest (recent shap): (1, n_features, 2)
    Averaging raw outputs of mixed shapes is what crashed the pipeline with
    'inhomogeneous shape'. Returns None when the output can't be reconciled.
    """
    if isinstance(sv, (list, tuple)):
        sv = sv[-1]  # binary classifiers: last element = class 1
    arr = np.asarray(sv, dtype=float)
    if arr.ndim == 3:            # (batch, features, classes)
        arr = arr[:, :, -1]
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    vec = arr.ravel()
    return vec if vec.size == n_cols else None

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

    # Tree members were trained on numeric features PLUS team-ID categorical
    # columns (58 + 2 = 60 wide). Feeding them a numeric-only matrix makes
    # LightGBM fatal-error with a shape mismatch and silently misaligns the
    # others. Build per-model inputs mirroring ensemble_predict's routing.
    if not {"home_team_id", "away_team_id"} <= set(games.columns):
        games = _add_team_ids(games)
    X_cat = _categorical_matrix(games)
    medians = models.get("impute_median")
    n_full = X.shape[1] + len(TREE_CATEGORICAL_COLS)

    def _model_input(name: str, i: int):
        """Row-slice input shaped exactly like that member's training matrix."""
        xn = X[i:i + 1]
        xc = X_cat[i:i + 1]
        if name == "xgboost":
            Xi = _impute_median(xn, medians)[0] if medians is not None else xn
            return _tree_dataframe(Xi, xc, cols)
        if name == "lightgbm":
            dfp = pd.DataFrame(xn, columns=cols)
            for j, c in enumerate(TREE_CATEGORICAL_COLS):
                dfp[c] = np.where(xc[:, j] < 0, UNK_TEAM_ID, xc[:, j]).astype(int)
            return dfp
        if name == "randomforest":
            rf = models.get("randomforest")
            n_feat = getattr(rf, "n_features_in_", None)
            if n_feat is not None and n_feat == len(FEATURE_COLS):
                return xn  # ablation RF trained without team IDs
            return np.hstack([xn, xc])
        return xn

    for idx, row in games.iterrows():
        game_id = row["game_id"]
        shap_values = {}
        perspective_team = home if (home := row.get("home_team")) else "HOME"

        if has_shap:
            # Collect SHAP from tree-based models
            tree_shaps = []
            for name, model in models.items():
                if name in ("scaler", "logistic", "impute_median"):
                    continue
                try:
                    Xin = _model_input(name, idx)
                    explainer = shap.TreeExplainer(model)
                    sv = _shap_vector(explainer.shap_values(Xin), n_full)
                    if sv is not None:
                        tree_shaps.append(sv)
                except Exception as e:
                    logger.debug("SHAP failed for %s on model %s: %s", game_id, name, e)

            if tree_shaps:
                avg_shap = np.mean(tree_shaps, axis=0)
                # FAVORED-team perspective: the model outputs P(home win).
                # When the AWAY team is favored (p < 0.5), negate — shap
                # values then describe the favorite's win probability, with
                # positive = pushes the favorite toward winning. Consistent
                # with the calibration page's favored-side view.
                p_home = pd.to_numeric(pd.Series([row.get("home_win_prob_model")]),
                                       errors="coerce").iloc[0]
                if pd.notna(p_home) and float(p_home) < 0.5:
                    avg_shap = -avg_shap
                    away = row.get("away_team")
                    perspective_team = away if isinstance(away, str) and away else "AWAY"
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
                "perspective_team": perspective_team,
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


def psi_noise_floor(n_baseline: int, n_current: int, n_bins: int = 10) -> float:
    """Expected PSI from sampling noise alone when both samples are drawn
    from the SAME distribution.

    For two independent samples the per-bin proportion error is O(1/sqrt(n)),
    giving E[PSI] ≈ (k−1)/2 · (1/n_base + 1/n_cur). At the drift step's
    adjacent-window sizes (~110 vs ~150 games) this is ≈0.07 — most of the
    way to the WARN threshold (0.10). Statuses must therefore be assigned on
    the NOISE-ADJUSTED PSI, or identical distributions page constantly.
    """
    if n_baseline <= 0 or n_current <= 0:
        return 0.0
    return (n_bins - 1) / 2.0 * (1.0 / n_baseline + 1.0 / n_current)


def compute_feature_drift(
    baseline_games: pd.DataFrame,
    current_games: pd.DataFrame,
    target_date_str: str,
    model_weights: dict | None = None,
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

        n_b, n_c = len(baseline_vals), len(current_vals)

        if n_b == 0 or n_c == 0:
            drift_rows.append({
                "feature": col,
                "current_mean": round(float(current_vals.mean()), 4) if n_c > 0 else 0.0,
                "baseline_mean": round(float(baseline_vals.mean()), 4) if n_b > 0 else 0.0,
                "psi": 0.0,
                "psi_adjusted": 0.0,
                "noise_floor": 0.0,
                "mean_shift": 0.0,
                "shift_se": 0.0,
                "location_shift": False,
                "status": "INSUFFICIENT",
                "weight_pct": round(float(model_weights.get(col, 0.0)), 3)
                              if model_weights else None,
                "n_baseline": int(n_b),
                "n_current": int(n_c),
            })
            continue

        psi = compute_psi(baseline_vals, current_vals)
        noise = psi_noise_floor(len(baseline_vals), len(current_vals))
        psi_adjusted = max(psi - noise, 0.0)

        # Location gate. PSI responds to ANY distributional change — including
        # pure binning wiggle on heavily-tied/quantized features (win_pct and
        # hardhit% rounded to 2–3 decimals put many teams on one value, so a
        # quantile edge landing inside a tie cluster moves whole teams between
        # bins while the distribution is unchanged). Only escalate above OK
        # when the mean ALSO moved beyond its sampling noise. Games in a
        # 7-day window share teams (~7 starts each), so the naive SE
        # understates true variance — inflate by a clustering factor of 1.5.
        n_b, n_c = len(baseline_vals), len(current_vals)
        if n_b + n_c > 2:
            pooled_sd = np.sqrt(
                ((n_b - 1) * baseline_vals.var(ddof=1)
                 + (n_c - 1) * current_vals.var(ddof=1))
                / (n_b + n_c - 2)
            )
        else:
            pooled_sd = 0.0
        mean_shift = float(current_vals.mean() - baseline_vals.mean())
        if pooled_sd > 0:
            shift_se = float(pooled_sd * np.sqrt(1.0 / n_b + 1.0 / n_c) * 1.5)
            location_shift = abs(mean_shift) > 2.0 * shift_se
        else:
            shift_se = 0.0
            location_shift = psi_adjusted > 0  # degenerate: fall back to PSI

        # Small windows make PSI statistically meaningless — report them as
        # INSUFFICIENT rather than WARN/ALERT so they never page anyone.
        if n_b < 100 or n_c < 30:
            status = "INSUFFICIENT"
        else:
            # Threshold on the NOISE-ADJUSTED PSI: raw PSI between two
            # same-distribution samples of this size already averages ~0.07,
            # so near-equal means with raw PSI 0.10–0.30 are binning wiggle,
            # not regime change. Raw PSI stays in the CSV for transparency.
            status = psi_status(psi_adjusted) if location_shift else "OK"

        drift_rows.append({
            "feature": col,
            "current_mean": round(float(current_vals.mean()), 4),
            "baseline_mean": round(float(baseline_vals.mean()), 4),
            "psi": psi,
            "psi_adjusted": round(psi_adjusted, 6),
            "noise_floor": round(noise, 6),
            "mean_shift": round(mean_shift, 6),
            "shift_se": round(shift_se, 6),
            "location_shift": bool(location_shift),
            "status": status,
            "weight_pct": round(float(model_weights.get(col, 0.0)), 3)
                          if model_weights else None,
            "n_baseline": int(n_b),
            "n_current": int(n_c),
        })

    df = pd.DataFrame(drift_rows)
    out_path = DATA_DELIVERY_DIR / f"{FEATURE_DRIFT}_{target_date_str}.csv"
    df.to_csv(out_path, index=False)

    n_warns = (df["status"] == "WARN").sum()
    n_alerts = (df["status"] == "ALERT").sum()
    logger.info(
        "Feature drift: %d features, %d warnings, %d alerts "
        "(statuses on noise-adjusted PSI; mean noise floor %.3f)",
        len(df), n_warns, n_alerts,
        float(df["noise_floor"].mean()) if "noise_floor" in df.columns else float("nan"),
    )

    return df
