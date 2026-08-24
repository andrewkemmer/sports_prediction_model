"""Central configuration for MLB Bet Predictor backend.

All paths, hyperparameters, seeds, PSI thresholds, and version metadata
keys live here. Import from this module to avoid hardcoding values.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"
MODELS_DIR = DATA_DELIVERY_DIR / "models"

# Full-history weather backfill (default ON): real StatsAPI first pitches +
# strictly-prior Open-Meteo archive observations for EVERY decided game,
# cached by game_pk so each run fetches only games missing from the cache.
# Set MLB_WEATHER_BACKFILL_ALL=0 to keep the old trailing-35-day window only.
WEATHER_BACKFILL_ALL = os.getenv("MLB_WEATHER_BACKFILL_ALL", "1").strip().lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
ELO_SEED = 1500  # Starting Elo for every team

# ---------------------------------------------------------------------------
# Elo parameters
# ---------------------------------------------------------------------------
ELO_K = 20  # Update factor
ELO_HOME_ADV = 65  # Home-field advantage in Elo points
ELO_REVERT_FACTOR = 1 / 3  # Season-to-season regression toward mean

# ---------------------------------------------------------------------------
# Feature rolling windows
# ---------------------------------------------------------------------------
WOBA_WINDOW = 30  # Games
BULLPEN_WHIP_WINDOW = 10  # Games
SP_ERA_WINDOW = 30  # Games
SP_K9_WINDOW = 30  # Games

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
RETRAIN_CADENCE_DAYS = 7
# Walk-forward validation folds below this many games are skipped entirely:
# a handful of games (postseason tails, offseason gaps) produce wild AUC/Brier
# swings (e.g. AUC 0.18 on 11 games) that pollute pooled metrics and the
# adaptive blend weights earned from them.
MIN_VAL_FOLD_GAMES = 40
DEFAULT_MAX_EVAL_FOLDS = 0  # 0 = full history
TRAIN_TEST_SPLIT_RATIO = 0.2  # Not used directly; walk-forward handles splits

# Ensemble members and their blend weights. These are FALLBACK PRIORS used
# only before the first adaptive weighting is computed (and when a member
# fails to produce OOF predictions); after every walk-forward run the blend
# switches to ADAPTIVE weights derived from pooled out-of-sample log-loss.
ENSEMBLE_WEIGHTS = {
    "xgboost": 0.25,
    "lightgbm": 0.25,
    "logistic": 0.30,
    "randomforest": 0.10,
    "mlp": 0.10,
}

# Adaptive ensemble weighting: softmax over pooled out-of-fold log-loss.
# A member beating another by Δ log-loss earns exp(Δ / TEMPERATURE) times
# its weight; FLOOR keeps every candidate alive (diversity), CAP prevents
# any single member from dominating.
ADAPTIVE_WEIGHT_TEMPERATURE = 0.03
ADAPTIVE_WEIGHT_FLOOR = 0.05
ADAPTIVE_WEIGHT_CAP = 0.45
# Blend-weight objective: "auc" (pooled OOF discrimination — pushes the
# blend toward the best-separating members) or "logloss" (calibration-
# oriented softmax, the previous behavior). Default "auc": the blend was
# measuring 0.525 while its best member scored 0.546 OOF — log-loss
# weighting let near-coin-flip members dilute the strong ones.
ADAPTIVE_WEIGHT_METRIC = "auc"
# AUC softmax uses a sharper temperature: AUC edges among members are
# ~0.005–0.045 (vs log-loss edges ~0.002–0.027), so T=0.015 separates
# signal members (edge > 0.02) from noise members (edge < 0.01) cleanly,
# landing near-coin-flip members on the 5% floor.
ADAPTIVE_WEIGHT_AUC_TEMPERATURE = 0.015
# Regularized: Optuna-tuned on 4,144-games/44-fold walk-forward
# (pooled OOF logloss 0.68107 vs 0.69115 for the old depth-5/300-r config).
# Shallow depth + high gamma + subsampling suppress variance in the
# MLB low-signal regime. The fold trainer adds early_stopping_rounds=20
# and n_estimators=2000 (generous ceiling ~50 median rounds at refit) when
# a validation window is available; fit-only refits use the params below
# directly with no early stopping. Train-median imputation is now applied
# alongside logistic/MLP (the Optuna winner consistently preferred it).
XGBOOST_PARAMS = {
    "max_depth": 2,
    "min_child_weight": 8,
    "gamma": 2.13,
    "subsample": 0.60,
    "colsample_bytree": 0.56,
    "learning_rate": 0.058,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
    "enable_categorical": True,
}
# n_estimators ceiling + early-stopping rounds for walk-forward folds.
# Separate from the constructor dict because xgboost 3.2 sklearn API
# requires eval_set when early_stopping_rounds is set, and the full-refit
# path has no validation window.
XGBOOST_FOLD_ROUNDS = 2000
XGBOOST_EARLY_STOP = 20
# Optuna-tuned on 4,159-games/44-fold walk-forward (tune_lightgbm_optuna.py,
# 50 trials, pooled OOF logloss 0.68066 vs 0.78465 for the old depth-5/300-r
# config; sealed holdout 2026-08-03→08-23 confirmed: 0.68150/AUC 0.5573 vs
# 0.71444/0.5492). Strongly regularized: tiny leaf count + high min gain +
# heavy bagging suppress variance in the MLB low-signal regime. Native NaN
# routing kept (impute_medians=False won); team-ID categoricals route via
# categorical_feature BY NAME in the fold trainer — unchanged.
LIGHTGBM_PARAMS = {
    "n_estimators": 50,
    "max_depth": 5,
    "num_leaves": 6,
    "min_child_samples": 59,
    "min_gain_to_split": 1.745,
    "bagging_fraction": 0.556,
    "bagging_freq": 1,
    "feature_fraction": 0.749,
    "learning_rate": 0.053,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}
# Totals regression variants
XGBOOST_REG_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": RANDOM_SEED,
}
LIGHTGBM_REG_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

# ---------------------------------------------------------------------------
# Coin-flip threshold
# ---------------------------------------------------------------------------
COIN_FLIP_THRESHOLD = 0.02  # |P - 0.5| < this → coin flip

# ---------------------------------------------------------------------------
# PSI thresholds
# ---------------------------------------------------------------------------
PSI_WARN_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25

# ---------------------------------------------------------------------------
# Artifact date format
# ---------------------------------------------------------------------------
DATE_FMT = "%Y%m%d"
DATE_READABLE_FMT = "%B %d, %Y"

# ---------------------------------------------------------------------------
# Version metadata keys
# ---------------------------------------------------------------------------
VERSION_KEY = "VERSION"
TRAINED_AT_KEY = "TRAINED_AT"
DATA_CUTOFF_KEY = "DATA_CUTOFF"

# ---------------------------------------------------------------------------
# Supported sports (MLB primary, others scaffolded)
# ---------------------------------------------------------------------------
SUPPORTED_SPORTS = ["MLB", "NBA", "NHL", "NFL", "CFB", "CBBM", "Tennis"]

# ---------------------------------------------------------------------------
# Tracking strings
# ---------------------------------------------------------------------------
FEATURE_DRIFT = "feature_drift"
TODAYS_GAMES = "todays_games"
POWER_RANKINGS = "power_rankings"
CALIBRATION = "calibration"
MODEL_MONITOR = "model_monitor"
SHAP_GAME = "shap_game"
MODEL_HISTORY = "model_history"
ENSEMBLE_FILE = "ensemble_latest.joblib"
