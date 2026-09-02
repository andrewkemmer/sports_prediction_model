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

# Multi-sport restructure (Phase A): repo-relative directory holding this
# sport's backend + data_delivery. Mirrored in master_pipeline.py (which
# needs the name before backend/ is importable) and frontend/sports_config.py
# (repo_subdir, the GitHub raw-URL prefix + local fallback dir). Phase C
# renames the directory to mlb-backend/ and flips all three at once; the
# path math above is already relative, so it survives the rename untouched.
SPORT_DIR_NAME = "mlb-backend"
REPO_SUBDIR = SPORT_DIR_NAME  # GitHub raw-URL path prefix for data_delivery

# Full-history weather backfill (default ON): real StatsAPI first pitches +
# strictly-prior Open-Meteo archive observations for EVERY decided game,
# cached by game_pk so each run fetches only games missing from the cache.
# Set MLB_WEATHER_BACKFILL_ALL=0 to keep the old trailing-35-day window only.
WEATHER_BACKFILL_ALL = os.getenv("MLB_WEATHER_BACKFILL_ALL", "1").strip().lower() in ("1", "true", "yes")

# Calibration map applied to the MONEYLINE blend after blending (calibration.py).
# "platt" = today's shipped 2-parameter logistic map; "identity" = publish the
# raw blend uncalibrated (calibrated == raw). Default stays "platt" until the
# blend-level gate (run_calibration_flip_test.py) passes; reversible via the
# CALIBRATION_MODE environment variable. Run-engine calibration is untouched.
CALIBRATION_MODE = os.getenv("CALIBRATION_MODE", "platt").strip().lower()

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
# Walk-forward FOLD cadence: fold boundaries every N days (weekly on the
# ~7,018-game frame -> ~74 folds). This is GEOMETRY, not a retrain schedule —
# the pipeline retrains the ensemble on EVERY run (no per-day dedup; the
# run-line/run-engine refits per run too). Do not repurpose this constant as
# a monitoring heuristic; the monitor's NEXT RETRAIN card uses
# NEXT_RUN_HEURISTIC_DAYS below.
RETRAIN_CADENCE_DAYS = 7
# Model-monitor "next expected run" heuristic for the NEXT RETRAIN card:
# retrain-every-run decision (2026-09-02) means the next expected run lands
# ~1 day after this run's emission. NOT scheduler-backed (no cron/next_run
# mechanism exists in the repo) — it drives card text only.
NEXT_RUN_HEURISTIC_DAYS = 1
# Walk-forward validation folds below this many games are skipped entirely:
# a handful of games (postseason tails, offseason gaps) produce wild AUC/Brier
# swings (e.g. AUC 0.18 on 11 games) that pollute pooled metrics and the
# adaptive blend weights earned from them.
MIN_VAL_FOLD_GAMES = 40

# Run-engine agreement filter: |moneyline_win_prob − derived_win_prob| above
# this marks a game as a CONFLICT on the dashboard and suppresses it from any
# future recommendation surface (see run_engine.agreement_stats).
AGREEMENT_FILTER_DELTA = 0.08
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
# Full-harness rounds test (run_lgb_rounds_test.py,
# lgb_rounds_22e1c5cb7895afe2, 65-col matrix incl. run_margin_diff, sealed
# 284-game holdout 2026-08-05→08-25): 15/17/20-round variants improve
# pooled OOF (logloss 0.6844→0.6837, ECE-cal 0.0085→0.0046) but degrade
# sealed-holdout ECE-cal (0.0566→0.0669 at 17r) → the member-level
# calibration gain does not survive the blend; config unchanged, 50 rounds
# stays.
#
# Fair winner re-verification (verify_lgb_winner.py, 2026-08-26): the
# Colab Optuna winner was re-measured under its OWN early-stopping
# discipline (not the tuner's forced-50 clamp). At its natural count
# (6 rounds, tail-early-stop condition) the winner scored sealed-holdout
# 0.6917 LL / 0.5134 AUC / 0.0401 ECE vs current (LIGHTGBM_PARAMS, 17r)
# 0.6823 / 0.5499 / 0.0548 → the winner still loses logloss and AUC on
# the sealed set. Verdict: DON'T ADOPT; this config stays.
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
# MLP member — small neural net with early stopping; the ensemble's
# diversity wildcard whose weight is earned (or starved) by the adaptive
# blend. Params below are the production values (hidden 32x16, alpha 0.01),
# plus the sklearn defaults the fold trainer relied on implicitly. MLP
# cannot consume NaN: train-fold-median imputation + StandardScaler, never
# a native-NaN route.
#
# Optuna study (tune_mlp_optuna.py, 2026-08-25, 75 trials, same 44
# walk-forward folds as production): best pooled OOF logloss 0.70992
# (hidden 32x16, tanh, alpha 1.1e-3, adaptive lr 2.3e-3, batch 256,
# max_iter 600, val 0.1, n_iter_no_change 10) vs current 0.79879 — the
# winner's edge came from the tiny early folds. On the SEALED 21-day
# holdout (refit on all pre-holdout games, the production condition) the
# current config WON: logloss 0.70283 vs 0.71069, AUC 0.5263 vs 0.5066.
# Verdict: NOT ADOPTED — the current config stays (honesty contract).
#
# Harness reconciliation (2026-08-25): per-fold MLP probabilities are
# bit-identical to production's walk_forward_evaluate on the same folds
# (logistic control 0.7052 matches to 4dp; AUC identical), and the pooled
# numbers agree exactly once the tuner clips at the same 1e-7 as
# compute_metrics (the 1e-6 clip had hidden 19 degenerate early-fold
# points: 0.79105 vs 0.79879). The ~0.7076 figure from the v2026.08.24
# run was a different data snapshot (47 folds), not a harness artifact.
MLP_PARAMS = {
    "hidden_layer_sizes": (32, 16),
    "alpha": 0.01,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "max_iter": 300,
    "learning_rate": "constant",
    "learning_rate_init": 0.001,
    "batch_size": "auto",
    "n_iter_no_change": 10,
    "activation": "relu",
    "random_state": RANDOM_SEED,
}

# Random Forest member — bagged trees, decorrelated from boosting errors;
# ~20% of the blend weight before adaptive re-weighting. sklearn trees
# cannot consume NaN: train-fold-median imputation (the same imputed
# matrix logistic/MLP use), plus integer team-ID categoricals when
# RF_WITH_TEAM_IDS is on (the production default).
#
# PROVENANCE (2026-08-31): tuned by backend/tune_rf_optuna.py — 75 Optuna
# trials, pooled OOF logloss objective over 71 walk-forward folds with OOF
# run_margin_diff attached (production-correct), study 'rf_moneyline'
# (sqlite:///rf_study.db).
#   MEMBER GATE (PASSED): winner vs the prior inline defaults (300 trees /
# min_samples_leaf 20): pooled OOF logloss 0.68655 vs 0.68783; sealed
# 21-day holdout (2026-08-09..08-29, n=282) logloss 0.68050 vs 0.68150,
# AUC 0.5783 vs 0.5770, ECE 0.0357 vs 0.0593 — all three improve on both
# views, so the tuner's harness gate printed ADOPT.
#   BLEND GATE (NEUTRAL, adopted per policy): run_rf_tuned_blend_ablation.py
# (production-correct, 3 sealed windows) shows the member gain does NOT move
# the blend — pooled blend ll 0.6836 vs 0.6837 / AUC 0.5657 vs 0.5656
# (slightly better), sealed deltas all within ±0.001 AUC / ±0.0006 ll
# (mixed-sign noise), strict multi-window verdict DON'T SHIP (0/3).
# Policy (user, 2026-08-31): each member as strong as possible as long as
# the blend is not measurably impacted — the tuned RF qualifies, so these
# params are ADOPTED. Trade-off: 800 trees makes the RF fold fit ~2.7x
# slower than 300.
RF_PARAMS = {
    "n_estimators": 800,
    "max_depth": 6,
    "min_samples_leaf": 17,
    "min_samples_split": 6,
    "max_features": "log2",
    "bootstrap": True,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
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
