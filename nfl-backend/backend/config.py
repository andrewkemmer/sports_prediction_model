"""Central configuration for the NFL production backend.

Single source of truth for every knob the production pipeline reads:
seeds, Elo parameters, trailing-window lookbacks, walk-forward fold
geometry, model hyperparameters, grids, and artifact paths.

Market-independence policy: no sportsbook/market data is ingested or used
anywhere in this backend. All outputs are model-derived/fair.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent          # nfl-backend/
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"
MODELS_DIR = DATA_DELIVERY_DIR / "models"

SPORT_DIR_NAME = "nfl-backend"
REPO_SUBDIR = SPORT_DIR_NAME

# ---------------------------------------------------------------------------
# Reproducibility — explicit seeds for every stochastic component
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
NUMPY_SEED = 42

# ---------------------------------------------------------------------------
# Historical eligibility policy
# ---------------------------------------------------------------------------
WARMUP_SEASONS = [2018]          # trailing priors only — never OOF-evaluated
OOF_FIRST_SEASON = 2019          # OOF population starts here
CORE_SEASONS = list(range(OOF_FIRST_SEASON, 2027))   # 2019..2026 inclusive
ALL_SEASONS = WARMUP_SEASONS + CORE_SEASONS
GAME_TYPES = {"REG"}             # regular season only; no pre/post season

# ---------------------------------------------------------------------------
# Elo (authoritative semantics — unchanged from the validated definitions)
# ---------------------------------------------------------------------------
ELO_PRIOR = 1500.0
ELO_K = 32.0
ELO_SCALE = 400.0

# ---------------------------------------------------------------------------
# Trailing windows (authoritative semantics — unchanged)
# ---------------------------------------------------------------------------
FORM_WINDOW = 4       # net pts/game window
WINPCT_WINDOW = 12    # trailing win% window
YPP_WINDOW = 5        # net yards/play window
EWM_HALFLIFE = 2      # decaying-window halflife (games)
OPP_ADJ_WINDOW = 6    # opponent-adjusted trailing-margin window (games)
PACE_WINDOW = 4       # trailing plays/min window (games)

# Venue / schedule facts
PRIME_TIME_HOUR = 17  # nflverse gametime is ET; >= this = evening kickoff

# ---------------------------------------------------------------------------
# Walk-forward fold geometry (calendar-day based, NEVER week-ID based)
# ---------------------------------------------------------------------------
RETRAIN_CADENCE_DAYS = 7   # validation-window width in calendar days

# ---------------------------------------------------------------------------
# Feature set version
# ---------------------------------------------------------------------------
FEATURE_SET_VERSION = "nfl-prod-v1"

# Authoritative starting production feature pool (spec section 9). The
# trailing ladder may compose additional candidates, but only these are
# served to the production models. is_home is a constant anchor: reported
# in the manifest, excluded from the model matrix.
FEATURE_COLUMNS = [
    "elo_diff", "win_pct_diff", "rest_days_diff", "is_dome_home",
    "ewm_net_pts_diff", "ewm_ypp_diff",
    "pace_plays_min_diff", "rest_short_diff", "div_game",
    "travel_miles_diff", "altitude_home", "prime_time",
]
ANCHOR_COLUMNS = ["is_home"]

# ---------------------------------------------------------------------------
# Model-family feature representations (spec section 14)
# ---------------------------------------------------------------------------
# Linear-family view: difference-oriented + the constant is_home anchor
# (documented representation — the home edge needs the anchor in a
# difference-only design).
LINEAR_FEATURES = FEATURE_COLUMNS + ANCHOR_COLUMNS

# Tree-family view: differences + raw home/away values where meaningful.
# Built by features.build_tree_view from the ladder's per-side columns.
TREE_FEATURES = None  # resolved at runtime by features.build_tree_view

# MLP view: same columns as linear, standard-scaled; documented explicitly.
MLP_FEATURES = LINEAR_FEATURES

# ---------------------------------------------------------------------------
# Ensemble members (moneyline) — NFL-specific hyperparameters
# ---------------------------------------------------------------------------
ENSEMBLE_MEMBERS = ["xgboost", "lightgbm", "logistic", "randomforest", "mlp"]

# Fallback prior weights; replaced by adaptive OOF-derived weights when
# available (see models.moneyline).
ENSEMBLE_WEIGHTS = {
    "xgboost": 0.25,
    "lightgbm": 0.25,
    "logistic": 0.20,
    "randomforest": 0.15,
    "mlp": 0.15,
}

# Adaptive blend: softmax over pooled OOF AUC edges (MLB-mirrored, tuned for
# the wider AUC spread an NFL season produces). FLOOR keeps members alive;
# CAP prevents domination.
ADAPTIVE_WEIGHT_METRIC = "auc"
ADAPTIVE_WEIGHT_TEMPERATURE = 0.015
ADAPTIVE_WEIGHT_FLOOR = 0.05
ADAPTIVE_WEIGHT_CAP = 0.45

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "min_child_weight": 5,
    "gamma": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.05,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
    "verbosity": 0,
}
LIGHTGBM_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "num_leaves": 12,
    "min_child_samples": 30,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}
LOGISTIC_PARAMS = {
    "C": 0.5,
    "max_iter": 2000,
    "random_state": RANDOM_SEED,
}
RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "min_samples_split": 10,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}
MLP_PARAMS = {
    "hidden_layer_sizes": (32, 16),
    "alpha": 0.01,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "max_iter": 400,
    "learning_rate_init": 0.001,
    "n_iter_no_change": 12,
    "activation": "relu",
    "random_state": RANDOM_SEED,
}

# Regression members (margin + total point regressions)
XGBOOST_REG_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "verbosity": 0,
}
LIGHTGBM_REG_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "num_leaves": 12,
    "min_child_samples": 30,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

# ---------------------------------------------------------------------------
# Margin / total distribution grids (spec sections 21/23)
# ---------------------------------------------------------------------------
SPREAD_GRID = list(range(-14, 15))          # margin thresholds L: -14..+14
TOTAL_GRID = list(range(24, 67))            # totals U: 24..66
# Half-stop lines the ±0.5 derived-ML stop prices from.
HALF_STOP_LINES = [-0.5, 0.5]
# Margin/total PMF support (integer points; discrete-normal base)
MARGIN_PMF_MAX = 30
TOTAL_PMF_MAX = 70

# Distribution model defaults
MARGIN_SIGMA = 13.5       # initial NFL margin std (points)
TOTAL_SIGMA = 10.0        # initial total std (points)
SIGMA_FLOOR_MARGIN = 9.0
SIGMA_CAP_MARGIN = 20.0
SIGMA_FLOOR_TOTAL = 7.0
SIGMA_CAP_TOTAL = 16.0
P_TIE_MAX = 0.02          # cap on the tied-margin mass (push bands)

# ---------------------------------------------------------------------------
# Artifact naming (frontend family contracts)
# ---------------------------------------------------------------------------
DATE_FMT = "%Y%m%d"

MONEYLINE_JSON = "nfl_moneyline_v1_{date}.json"
CALIBRATION_JSON = "nfl_calibration_{date}.json"
PREDICTIONS_HISTORY_CSV = "nfl_predictions_history_{date}.csv"
POWER_RANKINGS_CSV = "nfl_power_rankings_{date}.csv"
MARKETS_CSV = "nfl_run_engine_markets_{date}.csv"
MARKETS_META_JSON = "nfl_run_engine_markets_{date}.meta.json"
MARKETS_MONITOR_JSON = "nfl_run_engine_monitor_{date}.json"
QB_MATCHUP_JSON = "nfl_qb_matchup_{date}.json"
FEATURE_JSON = "nfl_feature_v1_{date}.json"
MODEL_MONITOR_JSON = "nfl_model_monitor_{date}.json"
MODEL_BUNDLE = MODELS_DIR / "nfl_ensemble_latest.joblib"
OOF_STORE_CSV = DATA_DELIVERY_DIR / "nfl_oof_store.csv"

# QB serving contract (enrichment only — never fabricated)
QB_FIELDS = [
    "qb_home_name", "qb_home_rating", "qb_home_td_per_game",
    "qb_home_cmp_pct", "qb_home_yards_per_attempt", "qb_home_ints",
    "qb_away_name", "qb_away_rating", "qb_away_td_per_game",
    "qb_away_cmp_pct", "qb_away_yards_per_attempt", "qb_away_ints",
]

# Coin-flip threshold for model_pick display
COIN_FLIP_THRESHOLD = 0.02
