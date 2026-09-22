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
GAME_TYPES = {"REG", "POST"}       # regular season + postseason; preseason excluded

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
MIN_VAL_FOLD_GAMES = 15    # ordinary OOF validation minimum; final tail retained

# ---------------------------------------------------------------------------
# Feature set version
# ---------------------------------------------------------------------------
FEATURE_SET_VERSION = "nfl-prod-v1"

# ---------------------------------------------------------------------------
# Moneyline calibration (MLB structural parity; favored-team space ONLY)
# ---------------------------------------------------------------------------
# "platt" (default): the favored-space 2-parameter logistic map.
# "identity": publish the raw blend — the calibrated path returns p unchanged.
# The switch is reversible (CALIBRATION_MODE env var / set_calibration_mode)
# and default stays "platt". All fits and applications happen in FAVORED-team
# space (the side with probability > 50%), never home-team space.
CALIBRATION_MODE = "platt"

# Pooled OOF games required before trusting a fitted Platt correction. Below
# this, a 2-param fit can chase noise; identity is the safer map (MLB parity:
# calibration.MIN_OOF_FOR_FIT).
MIN_OOF_FOR_FIT = 300

# ---------------------------------------------------------------------------
# THE feature contract — ONE master list (MLB structural parity)
# ---------------------------------------------------------------------------
# The binary moneyline defines the production feature list below. Every other
# consumer PULLS it; none of them declares or synthesizes its own list:
#
#   binary moneyline members ....... active_moneyline_feature_cols()
#   run-line / totals regressors .. features.tree_view() over the same list
#                                   (distributions.ScoreRegressor._matrix)
#   RFE trials ..................... same list for removals; additions come
#                                   only from KNOWN_FEATURE_COLS (candidates)
#   monitoring / manifest / workbook  same list
#
# Member routing is a RULE over this list, never a second list:
#   tree members (xgboost, lightgbm) -> the full list
#   linear members (elasticnet, mlp) -> the list minus RAW_PER_SIDE_COLS
# so `features.linear_view` / `features.tree_view` are pure projections of
# whatever appears here. A column that is not in this list cannot reach a
# model, and a column that is in it is unique by construction — no view needs
# duplicate protection.
MONEYLINE_FEATURE_COLS = [
    # served diffs (home − away; positive = home advantage)
    "elo_diff", "win_pct_diff", "rest_days_diff", "is_dome_home",
    "ewm_net_pts_diff", "ewm_ypp_diff",
    "pace_plays_min_diff", "rest_short_diff", "div_game",
    "travel_miles_diff", "altitude_home", "prime_time",
    # raw per-side levels (the tree family's home/away representations).
    # Declared HERE, never synthesized by a view: the served list stays the
    # only place a feature can appear.
    "elo_home", "elo_away",
    "win_pct_home", "win_pct_away",
    "ewm_net_pts_home", "ewm_net_pts_away",
    "ewm_ypp_home", "ewm_ypp_away",
    "rest_days_home", "rest_days_away",
    # constant home anchor last, so the linear member's positional contract is
    # unchanged by the raw-side block above (MLB parity: training.RAW_PER_SIDE_COLS)
    "is_home",
]

# Member-family routing over the master list — a SELECTOR, not a feature list.
# Mirrors MLB's training.RAW_PER_SIDE_COLS: the linear/MLP family consumes the
# diff/anchor view, tree members consume every column.
RAW_PER_SIDE_COLS = frozenset({
    "elo_home", "elo_away",
    "win_pct_home", "win_pct_away",
    "ewm_net_pts_home", "ewm_net_pts_away",
    "ewm_ypp_home", "ewm_ypp_away",
    "rest_days_home", "rest_days_away",
})

# The full candidate list (RFE trial space), defined ONCE. Additions may only
# name these; the RFE never derives candidates from a frame. A candidate must
# be a PIT-safe, pre-game column the feature engine produces and that is NOT
# already in the universe above. Empty today: the engine computes no PIT-safe
# columns beyond the universe (the frame's other numerics are structural
# identity/outcome fields or whole-timeline display records).
RFE_CANDIDATE_COLS: list[str] = []

# Trial / validation pool: universe first (canonical), then candidates.
# set_feature_subset validates against the POOL, because an adopted RFE record
# may promote candidates into serving width. Nothing is ever removed from it.
KNOWN_FEATURE_COLS = list(dict.fromkeys(
    list(MONEYLINE_FEATURE_COLS) + list(RFE_CANDIDATE_COLS)))

# RFE governance: this remains None during ordinary production runs. An
# explicit adoption action may set it; RFE trials never mutate it.
_FEATURE_SUBSET: list[str] | None = None


def active_moneyline_feature_cols() -> list[str]:
    """Model-facing contract: adopted RFE subset, else the full universe."""
    return list(_FEATURE_SUBSET) if _FEATURE_SUBSET is not None \
        else list(MONEYLINE_FEATURE_COLS)


def set_feature_subset(cols: list[str]) -> None:
    """Apply an adopted subset, rebuilt in canonical pool order.

    Membership (not the caller's sequence) and canonical order make a subset
    unique and positionally stable by construction, so no consumer needs its
    own duplicate or ordering protection. Raises on non-pool names — a subset
    naming an unknown feature is an upstream bug, not something to intersect
    away.
    """
    global _FEATURE_SUBSET
    if len(cols) < 1:
        raise ValueError("invalid NFL feature subset: empty")
    unknown = [c for c in cols if c not in KNOWN_FEATURE_COLS]
    if unknown:
        raise ValueError(f"NFL feature subset contains non-pool columns: {unknown[:6]}")
    chosen = set(cols)
    _FEATURE_SUBSET = [c for c in KNOWN_FEATURE_COLS if c in chosen]


def reset_feature_subset() -> None:
    global _FEATURE_SUBSET
    _FEATURE_SUBSET = None

# ---------------------------------------------------------------------------
# Ensemble members (moneyline) — NFL-specific hyperparameters
# ---------------------------------------------------------------------------
ENSEMBLE_MEMBERS = ["xgboost", "lightgbm", "elasticnet"]

# Fallback prior weights; replaced by adaptive OOF-derived weights when
# available (see models.moneyline).
ENSEMBLE_WEIGHTS = {
    "xgboost": 1 / 3,
    "lightgbm": 1 / 3,
    "elasticnet": 1 / 3,
}

# Adaptive blend: softmax over pooled OOF AUC edges (MLB-mirrored, tuned for
# the wider AUC spread an NFL season produces). FLOOR keeps members alive;
# CAP prevents domination.
ADAPTIVE_WEIGHT_METRIC = "logloss"
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
ELASTICNET_PARAMS = {
    "penalty": "elasticnet",
    "l1_ratio": 0.5,
    "C": 0.03,
    "solver": "saga",
    "max_iter": 4000,
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
# Canonical lines the Run-Engine Model card scores per-line OOF metrics at
# (the pooled-diagnostics tab's fixed totals + the NFL key-number spread).
RUN_ENGINE_FIXED_TOTALS = (38, 42, 46, 50, 54)
RUN_ENGINE_CANONICAL_SPREADS = (3,)
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
SHAP_GAME_PREFIX = "nfl_shap_game"
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
