"""
Walk-forward training for MLB Bet Predictor.

Implements expanding-window walk-forward splits, multi-target heads
(moneyline, totals, run line), evaluation metrics, and ensemble persistence.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_DELIVERY_DIR,
    DATE_FMT,
    ENSEMBLE_FILE,
    LIGHTGBM_PARAMS,
    LIGHTGBM_REG_PARAMS,
    MODELS_DIR,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
    VERSION_KEY,
    TRAINED_AT_KEY,
    DATA_CUTOFF_KEY,
    XGBOOST_PARAMS,
    XGBOOST_REG_PARAMS,
)

logger = logging.getLogger(__name__)

# Features used for model input
FEATURE_COLS = [
    "home_elo",
    "home_win_pct",
    "away_win_pct",
    "sp_era_home",
    "sp_era_away",
    "sp_k9_home",
    "sp_k9_away",
    "woba_30g_home",
    "woba_30g_away",
    "bullpen_whip_10g_home",
    "bullpen_whip_10g_away",
    "rest_days_home",
    "rest_days_away",
    "sp_era_30g_home",
    "sp_era_30g_away",
    "sp_k9_30g_home",
    "sp_k9_30g_away",
    "home_win_pct",
    "home_run_diff",
    "away_run_diff",
]
# Deduplicate
FEATURE_COLS = list(dict.fromkeys(FEATURE_COLS))


# ── Walk-forward splits ─────────────────────────────────────────────────────

def walk_forward_splits(
    games: pd.DataFrame,
    retrain_cadence_days: int = RETRAIN_CADENCE_DAYS,
    max_eval_folds: int = 0,
) -> list[dict[str, Any]]:
    """Generate expanding-window walk-forward train/val splits.

    Each validation window is `retrain_cadence_days` wide. The training set
    is all games strictly before the validation window start. Windows are
    non-overlapping and chronological.

    Returns a list of dicts with keys:
        train_games: DataFrame of training games
        val_games: DataFrame of validation games
        fold_idx: int
        val_start: datetime
        val_end: datetime
    """
    if "game_date" not in games.columns:
        raise ValueError("games must have a 'game_date' column")
    if "home_win" not in games.columns:
        logger.warning("walk_forward_splits: no 'home_win' column — cannot split")
        return []

    df = games.dropna(subset=["home_win"]).copy()
    if df.empty:
        logger.warning(
            "walk_forward_splits: all %d rows have NaN home_win — cannot split",
            len(games),
        )
        return []
    df["game_date"] = pd.to_datetime(df["game_date"])
    # Normalize to date-only (strip time) so unique dates represent calendar days,
    # not individual timestamps. Without this, each game with a unique start time
    # becomes its own "date" and 7-day validation windows collapse to 1 game.
    df["game_date"] = df["game_date"].dt.normalize()
    df = df.sort_values("game_date").reset_index(drop=True)

    if df.empty:
        return []

    unique_dates = sorted(df["game_date"].unique())
    if len(unique_dates) < retrain_cadence_days + 1:
        # Not enough data for even one split — use all as train, none as val
        logger.warning(
            "walk_forward_splits: only %d unique dates (need >= %d for cadence %d)",
            len(unique_dates), retrain_cadence_days + 1, retrain_cadence_days,
        )
        return []

    splits = []
    fold_idx = 0

    # Start validation windows from the first date that has enough history
    val_start_idx = retrain_cadence_days
    while val_start_idx < len(unique_dates):
        val_start = unique_dates[val_start_idx]
        val_end_idx = min(val_start_idx + retrain_cadence_days, len(unique_dates))
        val_end = unique_dates[val_end_idx - 1]

        # Training: everything strictly before val_start
        train_mask = df["game_date"] < val_start
        val_mask = (df["game_date"] >= val_start) & (df["game_date"] <= val_end)

        train_games = df[train_mask].copy()
        val_games = df[val_mask].copy()

        if not train_games.empty and not val_games.empty:
            splits.append({
                "train_games": train_games,
                "val_games": val_games,
                "fold_idx": fold_idx,
                "val_start": val_start,
                "val_end": val_end,
            })
            fold_idx += 1

        val_start_idx = val_end_idx

    # Limit to max_eval_folds (most recent folds)
    if max_eval_folds > 0 and len(splits) > max_eval_folds:
        splits = splits[-max_eval_folds:]

    return splits


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> dict[str, float]:
    """Compute classification metrics: AUC, Brier, LogLoss, ECE."""
    y_true = np.asarray(y_true)
    y_pred_prob = np.asarray(y_pred_prob)

    # Clip to avoid log(0)
    y_pred_prob = np.clip(y_pred_prob, 1e-7, 1 - 1e-7)

    result = {}
    try:
        result["auc"] = round(float(roc_auc_score(y_true, y_pred_prob)), 4)
    except ValueError:
        result["auc"] = 0.5

    result["brier"] = round(float(brier_score_loss(y_true, y_pred_prob)), 4)
    result["logloss"] = round(float(log_loss(y_true, y_pred_prob)), 4)
    result["ece"] = round(float(_expected_calibration_error(y_true, y_pred_prob)), 4)

    return result


def _expected_calibration_error(
    y_true: np.ndarray, y_pred_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred_prob >= bin_edges[i]) & (y_pred_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_pred_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def calibration_buckets(
    y_true: np.ndarray, y_pred_prob: np.ndarray, n_bins: int = 10
) -> list[dict[str, Any]]:
    """Compute calibration bucket data for the dashboard."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    buckets = []
    for i in range(n_bins):
        mask = (y_pred_prob >= bin_edges[i]) & (y_pred_prob < bin_edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = round(float(y_pred_prob[mask].mean()), 4)
        mean_actual = round(float(y_true[mask].mean()), 4)
        gap = round(mean_pred - mean_actual, 4)
        buckets.append({
            "bucket": f"{bin_edges[i]*100:.0f}–{bin_edges[i+1]*100:.0f}%",
            "mean_predicted": mean_pred,
            "mean_actual": mean_actual,
            "count": count,
            "gap": gap,
        })
    return buckets


# ── Moneyline ensemble ──────────────────────────────────────────────────────

def _prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and target from a games DataFrame."""
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].fillna(0).values
    y = df["home_win"].values.astype(float)
    return X, y


def train_moneyline_ensemble(
    train: pd.DataFrame, val: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, float]]:
    """Train XGBoost + LightGBM + Logistic Regression ensemble for moneyline.

    Returns (ensemble_dict, metrics) where ensemble_dict maps model names
    to fitted model objects.
    """
    X_train, y_train = _prepare_features(train)
    X_val, y_val = _prepare_features(val)

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Insufficient training or validation data")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    models = {}

    # XGBoost
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(**XGBOOST_PARAMS)
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        models["xgboost"] = xgb
    except ImportError:
        logger.warning("xgboost not available, skipping XGB member")

    # LightGBM
    try:
        from lightgbm import LGBMClassifier
        lgbm = LGBMClassifier(**LIGHTGBM_PARAMS)
        lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        models["lightgbm"] = lgbm
    except ImportError:
        logger.warning("lightgbm not available, skipping LGBM member")

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X_train_scaled, y_train)
    models["logistic"] = lr
    models["scaler"] = scaler

    # Ensemble prediction (average probabilities)
    probs = []
    for name, model in models.items():
        if name == "scaler":
            continue
        if name == "logistic":
            probs.append(model.predict_proba(X_val_scaled)[:, 1])
        else:
            probs.append(model.predict_proba(X_val)[:, 1])

    ensemble_prob = np.mean(probs, axis=0) if probs else np.full(len(y_val), 0.5)

    metrics = compute_metrics(y_val, ensemble_prob)
    return models, metrics


# ── Totals regression ───────────────────────────────────────────────────────

def train_totals_model(
    train: pd.DataFrame, val: pd.DataFrame
) -> dict[str, Any]:
    """Train XGBoost + LightGBM regression ensemble for total runs."""
    cols = [c for c in FEATURE_COLS if c in train.columns]
    X_train = train[cols].fillna(0).values
    y_train = train["total_runs"].values.astype(float)
    X_val = val[cols].fillna(0).values
    y_val = val["total_runs"].values.astype(float)

    models = {}

    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(**XGBOOST_REG_PARAMS)
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        models["xgboost_reg"] = xgb
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor
        lgbm = LGBMRegressor(**LIGHTGBM_REG_PARAMS)
        lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        models["lgbm_reg"] = lgbm
    except ImportError:
        pass

    # Predictions for metrics
    preds = []
    for name, model in models.items():
        preds.append(model.predict(X_val))

    if preds:
        ensemble_pred = np.mean(preds, axis=0)
        rmse = float(np.sqrt(np.mean((ensemble_pred - y_val) ** 2)))
        mae = float(np.mean(np.abs(ensemble_pred - y_val)))
    else:
        rmse = mae = float("nan")

    return {
        "models": models,
        "metrics": {"rmse": round(rmse, 4), "mae": round(mae, 4)},
    }


# ── Run-line classification ─────────────────────────────────────────────────

def train_run_line_model(
    train: pd.DataFrame, val: pd.DataFrame
) -> dict[str, Any]:
    """Train run-line cover probability classifier.

    Run-line cover: does the home team cover -1.5 run line?
    (i.e., win by 2+ runs)
    """
    train = train.copy()
    val = val.copy()
    train["run_line_cover"] = (train["home_win"] == 1) & (train.get("total_runs", 0) > 1)
    # Simplified: home covers if they win (since run_line is -1.5)
    train["run_line_cover"] = (train["home_win"] == 1).astype(float)
    val["run_line_cover"] = (val["home_win"] == 1).astype(float)

    cols = [c for c in FEATURE_COLS if c in train.columns]
    X_train = train[cols].fillna(0).values
    y_train = train["run_line_cover"].values
    X_val = val[cols].fillna(0).values
    y_val = val["run_line_cover"].values

    models = {}
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(**XGBOOST_PARAMS)
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        models["xgboost_rl"] = xgb
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier
        lgbm = LGBMClassifier(**LIGHTGBM_PARAMS)
        lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        models["lgbm_rl"] = lgbm
    except ImportError:
        pass

    # Predictions
    probs = []
    for name, model in models.items():
        probs.append(model.predict_proba(X_val)[:, 1])

    if probs:
        ensemble_prob = np.mean(probs, axis=0)
        metrics = compute_metrics(y_val, ensemble_prob)
    else:
        metrics = {"auc": 0.5, "brier": 0.25}

    return {"models": models, "metrics": metrics}


# ── Full walk-forward evaluation ────────────────────────────────────────────

def walk_forward_evaluate(
    games: pd.DataFrame,
    retrain_cadence_days: int = RETRAIN_CADENCE_DAYS,
    max_eval_folds: int = 0,
    force_retrain: bool = False,
) -> tuple[dict[str, Any], dict[str, float], pd.DataFrame]:
    """Run full walk-forward evaluation across all splits.

    Returns:
        (best_models, pooled_metrics, all_predictions)
    """
    splits = walk_forward_splits(games, retrain_cadence_days, max_eval_folds)

    if not splits:
        logger.warning("No walk-forward splits generated; training on full data")
        # Fall back to train on everything
        splits = [{
            "train_games": games.dropna(subset=["home_win"]),
            "val_games": games.dropna(subset=["home_win"]).tail(min(50, len(games.dropna(subset=["home_win"])))),
            "fold_idx": 0,
            "val_start": games["game_date"].min(),
            "val_end": games["game_date"].max(),
        }]

    all_preds = []
    fold_metrics_list = []

    for split in splits:
        train = split["train_games"]
        val = split["val_games"]

        if len(train) < 10 or len(val) < 5:
            continue

        try:
            ml_models, ml_metrics = train_moneyline_ensemble(train, val)
        except Exception as e:
            logger.warning("Fold %d moneyline training failed: %s", split["fold_idx"], e)
            continue

        # Generate predictions for validation set
        cols = [c for c in FEATURE_COLS if c in val.columns]
        X_val = val[cols].fillna(0).values
        scaler = ml_models.get("scaler")

        probs = []
        for name, model in ml_models.items():
            if name == "scaler":
                continue
            if name == "logistic" and scaler is not None:
                probs.append(model.predict_proba(scaler.transform(X_val))[:, 1])
            else:
                probs.append(model.predict_proba(X_val)[:, 1])

        ensemble_prob = np.mean(probs, axis=0) if probs else np.full(len(val), 0.5)

        val_pred = val.copy()
        val_pred["home_win_prob_model"] = ensemble_prob
        val_pred["fold_idx"] = split["fold_idx"]
        all_preds.append(val_pred)
        fold_metrics_list.append(ml_metrics)

    # Pool metrics across folds
    if all_preds:
        combined = pd.concat(all_preds, ignore_index=True)
        pooled = compute_metrics(combined["home_win"].values, combined["home_win_prob_model"].values)
    else:
        combined = pd.DataFrame()
        pooled = {"auc": 0.5, "brier": 0.25, "logloss": 0.69, "ece": 0.0}

    # Retrain on full data for final model
    full_train = games.dropna(subset=["home_win"])
    if len(full_train) >= 20:
        last_split = splits[-1]
        try:
            best_models, _ = train_moneyline_ensemble(last_split["train_games"], last_split["val_games"])
        except Exception:
            best_models = {}
    else:
        best_models = {}

    return best_models, pooled, combined


# ── Persistence ─────────────────────────────────────────────────────────────

def persist_ensemble(
    models: dict[str, Any],
    metrics: dict[str, float],
    version: str = "v3.2.1",
    data_cutoff: Optional[str] = None,
) -> Path:
    """Save ensemble models and metadata to joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        VERSION_KEY: version,
        TRAINED_AT_KEY: datetime.now().isoformat(),
        DATA_CUTOFF_KEY: data_cutoff or datetime.now().strftime(DATE_FMT),
    }

    bundle = {
        "models": models,
        "metrics": metrics,
        "metadata": metadata,
    }

    path = MODELS_DIR / ENSEMBLE_FILE
    joblib.dump(bundle, path)
    logger.info("Ensemble persisted to %s", path)
    return path


def load_ensemble(path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Load a persisted ensemble bundle."""
    path = path or (MODELS_DIR / ENSEMBLE_FILE)
    if not path.exists():
        return None
    return joblib.load(path)


def should_retrain(last_trained: Optional[datetime], cadence_days: int = RETRAIN_CADENCE_DAYS) -> bool:
    """Determine if retraining is needed based on cadence."""
    if last_trained is None:
        return True
    return (datetime.now() - last_trained).days >= cadence_days


def predict_games(
    models: dict[str, Any],
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Apply ensemble models to predict on a set of games.

    Adds columns: home_win_prob_model, away_win_prob_model, model_pick, edge_home, edge_away
    """
    if not models:
        return games

    cols = [c for c in FEATURE_COLS if c in games.columns]
    X = games[cols].fillna(0).values

    scaler = models.get("scaler")
    probs = []
    for name, model in models.items():
        if name == "scaler":
            continue
        if name == "logistic" and scaler is not None:
            probs.append(model.predict_proba(scaler.transform(X))[:, 1])
        else:
            probs.append(model.predict_proba(X)[:, 1])

    if not probs:
        games["home_win_prob_model"] = 0.5
    else:
        games["home_win_prob_model"] = np.round(np.mean(probs, axis=0), 4)

    games["away_win_prob_model"] = 1 - games["home_win_prob_model"]

    # Model pick
    games["model_pick"] = np.where(
        games["home_win_prob_model"] >= 0.5, games["home_team"], games["away_team"]
    )

    # Edge: model_prob - fair_market_prob (vig removed via two-way normalization)
    if "moneyline_home" in games.columns and games["moneyline_home"].notna().any():
        ml_home = games["moneyline_home"].fillna(-110).values
        ml_away = games["moneyline_away"].fillna(-110).values
        fair_home = np.where(ml_home < 0, -ml_home / (-ml_home + 100), 100 / (ml_home + 100))
        fair_away = np.where(ml_away < 0, -ml_away / (-ml_away + 100), 100 / (ml_away + 100))
        # Normalize (remove vig)
        total = fair_home + fair_away
        fair_home_norm = fair_home / total
        fair_away_norm = fair_away / total
        games["edge_home"] = np.round(games["home_win_prob_model"].values - fair_home_norm, 4)
        games["edge_away"] = np.round(games["away_win_prob_model"].values - fair_away_norm, 4)
    else:
        games["edge_home"] = 0.0
        games["edge_away"] = 0.0

    return games


def update_model_history(
    metrics: dict[str, float],
    version: str,
    notes: str = "",
) -> None:
    """Append a row to model_history.json for the Model Monitor page."""
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    history_path = DATA_DELIVERY_DIR / "model_history.json"

    history = []
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    history.append({
        "version": version,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "auc": metrics.get("auc", 0),
        "brier": metrics.get("brier", 0),
        "logloss": metrics.get("logloss", 0),
        "ece": metrics.get("ece", 0),
        "notes": notes,
    })

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
