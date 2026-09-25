"""Central configuration for the standalone NBA production backend.

The ensemble, seed, fold cadence, warm-up, and member hyper-parameters are
intentionally copied from MLB's production contract.  Only the feature
representation and score/line grids are NBA-native.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"
MODELS_DIR = DATA_DELIVERY_DIR / "models"
# The normalized cache is deliberately outside the repository.  It is never
# an input to git or to the frontend artifact resolver.  Respect an explicit
# NBA_CACHE_DIR first (Kaggle/CI can point it at a mounted volume); otherwise
# use the user cache rather than creating a hidden directory at the repo root.
CACHE_DIR = Path(
    os.environ.get("NBA_CACHE_DIR", "")
    or (Path.home() / ".cache" / "sports_prediction_model" / "nba")
).expanduser()

SPORT_DIR_NAME = "nba-backend"
REPO_SUBDIR = SPORT_DIR_NAME

# Canonical source pin used by the Kaggle orchestration notebook and recorded
# in every normalized-cache manifest.  Dataset version 238 was the published
# ``wyattowalsh/basketball`` version when this backend was approved.
NBA_DATASET_REF = "wyattowalsh/basketball"
NBA_DATASET_VERSION = "238"
NBA_DATASET_URL = "https://www.kaggle.com/datasets/wyattowalsh/basketball"

RANDOM_SEED = 42
NUMPY_SEED = 42

# Historical eligibility starts with the 2024-25 season.  The warehouse's
# season_year is normalized to its starting year (2024 for 2024-25).
NBA_FIRST_SEASON = 2024
OOF_FIRST_SEASON = NBA_FIRST_SEASON
CORE_SEASONS = list(range(OOF_FIRST_SEASON, 2100))
ALL_SEASONS = CORE_SEASONS
GAME_TYPES = {1, 2, 3}
GAME_TYPE_REG = 1
GAME_TYPE_POST = 2

WARMUP_DAYS = 30
MIN_TRAIN_DAYS = WARMUP_DAYS
RETRAIN_CADENCE_DAYS = 7
MIN_VAL_FOLD_GAMES = 40

# MLB/NHL rating constants retained exactly for a common Elo state contract.
ELO_PRIOR = 1500.0
ELO_K = 20.0
ELO_SCALE = 400.0
ELO_HOME_ADV = 65.0
ELO_REVERT_FACTOR = 1 / 3

FORM_WINDOW = 5
WINPCT_WINDOW = 12
DEFENSE_WINDOW = 5
EWM_HALFLIFE = 3
OPP_ADJ_WINDOW = 6
PBP_ROLL_WINDOW = 5
RFE_COMMIT_SE_MULTIPLE = 1.0
RFE_NOISE_SIGMA = 1.0
RFE_MAX_STEPS = 120

FEATURE_SET_VERSION = "nba-prod-v1.0-team-categories"

CALIBRATION_MODE = "platt"
MIN_OOF_FOR_FIT = 300

# One authoritative moneyline feature contract.  Every model, monitor,
# manifest, and run-line regressor projects this list.
MONEYLINE_FEATURE_COLS = [
    "elo_diff", "win_pct_diff", "rest_days_diff", "back_to_back_diff",
    "ewm_net_points_diff", "ewm_off_rating_diff", "ewm_def_rating_diff",
    "ewm_pace_diff", "ewm_efg_pct_diff", "ewm_turnover_margin_diff",
    "ewm_rebound_margin_diff", "ewm_ast_per_game_diff", "is_playoffs",
    "elo_home", "elo_away", "win_pct_home", "win_pct_away",
    "ewm_off_rating_home", "ewm_off_rating_away",
    "ewm_def_rating_home", "ewm_def_rating_away", "rest_days_home",
    "rest_days_away", "is_home",
]

# Stable current-team categories.  The warehouse's numeric team IDs are
# resolved to these abbreviations during ingestion; unknown values receive a
# reserved category rather than being silently mapped to a real team.
NBA_TEAM_ID: dict[str, int] = {
    "ATL": 0, "BOS": 1, "BKN": 2, "CHA": 3, "CHI": 4, "CLE": 5,
    "DAL": 6, "DEN": 7, "DET": 8, "GSW": 9, "HOU": 10, "IND": 11,
    "LAC": 12, "LAL": 13, "MEM": 14, "MIA": 15, "MIL": 16, "MIN": 17,
    "NOP": 18, "NYK": 19, "OKC": 20, "ORL": 21, "PHI": 22, "PHX": 23,
    "POR": 24, "SAC": 25, "SAS": 26, "TOR": 27, "UTA": 28, "WAS": 29,
}
TEAM_ALIASES = {
    "WSH": "WAS", "NJN": "BKN", "NJ": "BKN", "NOH": "NOP", "SEA": "OKC",
    "VAN": "MEM", "CHH": "CHA", "SAN": "SAS", "PHX": "PHX",
}
UNK_TEAM_ID = 99
TREE_CATEGORICAL_COLS = ["home_team_id", "away_team_id"]


def normalize_team_abbr(value: object) -> str:
    text = str(value or "").strip().upper()
    return TEAM_ALIASES.get(text, text)


def team_category_id(abbr: object) -> int:
    return NBA_TEAM_ID.get(normalize_team_abbr(abbr), UNK_TEAM_ID)


# RFE candidates are a declared, PIT-safe trailing pool.  They are
# record-only by default and cannot silently mutate the served contract.
TEAM_CANDIDATE_TRAILING_SPECS: dict[str, tuple[str, ...]] = {
    "points_for_pg": ("ewm", "roll"),
    "points_against_pg": ("ewm", "roll"),
    "assists_per_game": ("ewm", "roll"),
    "rebounds_per_game": ("ewm", "roll"),
    "turnovers_per_game": ("ewm", "roll"),
    "three_point_pct": ("ewm", "roll"),
    "free_throw_pct": ("ewm", "roll"),
}
CANDIDATE_FAMILIES = {"nba": {"base": TEAM_CANDIDATE_TRAILING_SPECS}}
NBA_CANDIDATE_COLS = list(dict.fromkeys(
    f"nba_{metric}_{window}_{rep}"
    for metric, windows in TEAM_CANDIDATE_TRAILING_SPECS.items()
    for window in windows for rep in ("diff", "home", "away")
))
RAW_PER_SIDE_COLS = frozenset({
    "elo_home", "elo_away", "win_pct_home", "win_pct_away",
    "ewm_off_rating_home", "ewm_off_rating_away",
    "ewm_def_rating_home", "ewm_def_rating_away", "rest_days_home",
    "rest_days_away",
} | {f"nba_{m}_{w}_{s}" for m, ws in TEAM_CANDIDATE_TRAILING_SPECS.items()
     for w in ws for s in ("home", "away")})
RFE_CANDIDATE_COLS = [c for c in NBA_CANDIDATE_COLS
                      if c not in set(MONEYLINE_FEATURE_COLS)]
KNOWN_FEATURE_COLS = list(dict.fromkeys(MONEYLINE_FEATURE_COLS + RFE_CANDIDATE_COLS))
_FEATURE_SUBSET: list[str] | None = None


def active_moneyline_feature_cols() -> list[str]:
    return list(_FEATURE_SUBSET) if _FEATURE_SUBSET is not None else list(MONEYLINE_FEATURE_COLS)


def set_feature_subset(cols: list[str]) -> None:
    global _FEATURE_SUBSET
    if not cols:
        raise ValueError("NBA feature subset cannot be empty")
    unknown = [c for c in cols if c not in KNOWN_FEATURE_COLS]
    if unknown:
        raise ValueError(f"unknown NBA feature columns: {unknown}")
    chosen = set(cols)
    _FEATURE_SUBSET = [c for c in KNOWN_FEATURE_COLS if c in chosen]


def reset_feature_subset() -> None:
    global _FEATURE_SUBSET
    _FEATURE_SUBSET = None


# Exact MLB-tuned member parameters and cold-start priors.
ENSEMBLE_MEMBERS = ["xgboost", "lightgbm", "elasticnet"]
ENSEMBLE_WEIGHTS = {"xgboost": 0.3333, "lightgbm": 0.3333, "elasticnet": 0.3334}
ADAPTIVE_WEIGHT_METRIC = "logloss"
BLEND_SPACE = "logit"
XGBOOST_PARAMS = {
    "max_depth": 3, "min_child_weight": 12, "gamma": 2.4178,
    "subsample": 0.812, "colsample_bytree": 0.6382,
    "learning_rate": 0.1097, "random_state": RANDOM_SEED,
    "eval_metric": "logloss", "enable_categorical": True,
}
XGBOOST_FOLD_ROUNDS = 2000
XGBOOST_EARLY_STOP = 20
LIGHTGBM_PARAMS = {
    "n_estimators": 50, "max_depth": 6, "num_leaves": 6,
    "min_child_samples": 70, "min_gain_to_split": 1.2224,
    "bagging_fraction": 0.4518, "bagging_freq": 1,
    "feature_fraction": 0.7632, "learning_rate": 0.0332,
    "random_state": RANDOM_SEED, "verbose": -1,
}
ELASTICNET_PARAMS = {
    "penalty": "elasticnet", "l1_ratio": 0.5, "C": 0.03,
    "solver": "saga", "max_iter": 4000, "tol": 1e-4,
    "random_state": RANDOM_SEED,
}

XGBOOST_REG_PARAMS = {
    "n_estimators": 300, "max_depth": 3, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "random_state": RANDOM_SEED,
    "verbosity": 0,
}
LIGHTGBM_REG_PARAMS = {
    "n_estimators": 200, "max_depth": 4, "num_leaves": 12,
    "min_child_samples": 30, "learning_rate": 0.05, "subsample": 0.8,
    "subsample_freq": 1, "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED, "verbose": -1,
}

# NBA line grids.  Lines are integer thresholds; half-point spread stops are
# represented by the explicit HALF_STOP_LINES namespace.  Totals use the
# dedicated p_push_total_<U> namespace to avoid any spread collision.
SPREAD_GRID = list(range(-20, 21))
TOTAL_GRID = list(range(180, 281))
HALF_STOP_LINES = [-0.5, 0.5]
RUN_ENGINE_FIXED_TOTALS = (220, 225, 230)
RUN_ENGINE_CANONICAL_SPREADS = (2, 5)
MARGIN_PMF_MAX = 60
TOTAL_PMF_MAX = 300
MARGIN_SIGMA = 12.0
TOTAL_SIGMA = 18.0

DATE_FMT = "%Y%m%d"
MONEYLINE_JSON = "nba_moneyline_v1_{date}.json"
CALIBRATION_JSON = "nba_calibration_{date}.json"
PREDICTIONS_HISTORY_CSV = "nba_predictions_history_{date}.csv"
POWER_RANKINGS_CSV = "nba_power_rankings_{date}.csv"
MARKETS_CSV = "nba_run_engine_markets_{date}.csv"
MARKETS_META_JSON = "nba_run_engine_markets_{date}.meta.json"
MARKETS_MONITOR_JSON = "nba_run_engine_monitor_{date}.json"
PLAYER_MATCHUP_JSON = "nba_player_leader_matchup_{date}.json"
FEATURE_JSON = "nba_feature_v1_{date}.json"
MODEL_MONITOR_JSON = "nba_model_monitor_{date}.json"
SHAP_GAME_PREFIX = "nba_shap_game"
MODEL_BUNDLE = MODELS_DIR / "nba_ensemble_latest.joblib"
OOF_STORE_CSV = DATA_DELIVERY_DIR / "nba_oof_store.csv"
FEATURE_SELECTION_JSON = "nba_feature_selection_{date}.json"
FEATURE_WORKBOOK_XLSX = "nba_feature_workbook_{date}.xlsx"
RUN_ENGINE_FEATURE_DRIFT_PREFIX = "nba_run_engine_feature_drift_"
RUN_ENGINE_FEATURE_COVERAGE_PREFIX = "nba_run_engine_feature_coverage_"

PLAYER_FIELDS = [
    "p_home_name", "p_home_ppg", "p_home_apg", "p_home_games",
    "p_away_name", "p_away_ppg", "p_away_apg", "p_away_games",
]
PLAYER_WINDOW_GAMES = 5
PLAYER_MIN_GAMES = 3
PLAYER_MIN_MINUTES = 15.0
COIN_FLIP_THRESHOLD = 0.02

RFE_FORCE_ENV = "NBA_RFE_FORCE"
RFE_ADDS_ENV = "NBA_RFE_ADDITION_MONEYLINE_LIST"
RFE_REMOVES_ENV = "NBA_RFE_REMOVAL_MONEYLINE_LIST"
