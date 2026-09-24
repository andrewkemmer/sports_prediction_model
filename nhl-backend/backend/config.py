"""Central configuration for the NHL production backend.

Structural mirror of ``nfl-backend/backend/config.py`` (which itself mirrors
MLB): single source of truth for every knob the production pipeline reads —
seeds, Elo parameters, trailing-window lookbacks, walk-forward fold geometry,
model hyperparameters, grids, and artifact paths.

The ensemble is MLB-IDENTICAL (XGBoost / LightGBM / elastic-net with MLB's
tuned member params; equal-thirds fold-0 priors with rolling SLSQP blend
re-earning in logit space). The feature contract is NHL-native (official NHL
API boxscore rollups + Elo + goalie-insensitive team stats).

Market-independence policy: no sportsbook/market data is ingested or used
anywhere in this backend. All outputs are model-derived/fair.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent          # nhl-backend/
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"
MODELS_DIR = DATA_DELIVERY_DIR / "models"

SPORT_DIR_NAME = "nhl-backend"
REPO_SUBDIR = SPORT_DIR_NAME

# ---------------------------------------------------------------------------
# Reproducibility — explicit seeds for every stochastic component
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
NUMPY_SEED = 42

# ---------------------------------------------------------------------------
# Historical eligibility policy
# ---------------------------------------------------------------------------
# Data starts season 2024 (the 2024-25 NHL season, the first with complete
# official-API boxscore coverage for this pipeline). There is NO NFL-style
# warm-up season; instead the walk-forward carries an MLB-style 30-DAY
# warm-up: OOF validation windows begin only after the first 30 days of
# core data, so every fold trains on at least ~30 days of settled history.
# All of season 2024+ is OOF-eligible.
NHL_FIRST_SEASON = 2024
WARMUP_SEASONS: list[int] = []            # no warm-up season (30-day warm-up instead)
OOF_FIRST_SEASON = NHL_FIRST_SEASON       # OOF population starts here
CORE_SEASONS = list(range(OOF_FIRST_SEASON, 2027))   # 2024..2026 inclusive
ALL_SEASONS = CORE_SEASONS
# gameType 2 = regular season, 3 = playoffs (official NHL API convention).
GAME_TYPES = {2, 3}
GAME_TYPE_REG = 2
GAME_TYPE_POST = 3

# MLB-style warm-up: the first OOF validation window starts this many
# calendar days after the first core game date (fold geometry in folds.py).
WARMUP_DAYS = 30
MIN_TRAIN_DAYS = WARMUP_DAYS

# ---------------------------------------------------------------------------
# Elo (MLB values — K=20, home advantage 65, 1/3 season-to-season revert)
# ---------------------------------------------------------------------------
ELO_PRIOR = 1500.0
ELO_K = 20.0
ELO_SCALE = 400.0
ELO_HOME_ADV = 65.0
ELO_REVERT_FACTOR = 1 / 3     # applied at each season boundary

# ---------------------------------------------------------------------------
# Trailing windows (NHL game cadence is dense; windows are game-count based
# like MLB's, with the EWM halflife and rolling windows sized for ~3 games
# per team-week)
# ---------------------------------------------------------------------------
FORM_WINDOW = 5       # net goals/game window
WINPCT_WINDOW = 12    # trailing win% window (NFL/MLB value)
GOALS_AGAINST_WINDOW = 5   # trailing goals-allowed window
EWM_HALFLIFE = 3      # decaying-window halflife (games; dense NHL cadence)
PBP_ROLL_WINDOW = 5   # trailing flat window for shot/faceoff candidate metrics
OPP_ADJ_WINDOW = 6    # opponent-adjusted trailing window (games)

# RFE commit gate (MLB parity, 2026-09-23: RFE_NOISE_SIGMA = 1.0 — the
# commit threshold is max(0.0005, 1.0 x paired per-game SE). The paired
# construction still makes junk features self-calibrate a high bar.
RFE_COMMIT_SE_MULTIPLE = 1.0
RFE_NOISE_SIGMA = 1.0
RFE_MAX_STEPS = 120

# ---------------------------------------------------------------------------
# Walk-forward fold geometry (calendar-day based, NEVER season-day based)
# ---------------------------------------------------------------------------
RETRAIN_CADENCE_DAYS = 7   # validation-window width in calendar days
MIN_VAL_FOLD_GAMES = 40    # MLB value (NFL uses 15; NHL's dense slate fills
                           # 7-day windows well past 40 games)

# ---------------------------------------------------------------------------
# Feature set version
# ---------------------------------------------------------------------------
FEATURE_SET_VERSION = "nhl-prod-v1.0-goalies-team-categories"

# ---------------------------------------------------------------------------
# Moneyline calibration (MLB structural parity; favored-team space ONLY)
# ---------------------------------------------------------------------------
CALIBRATION_MODE = "platt"
MIN_OOF_FOR_FIT = 300

# ---------------------------------------------------------------------------
# THE feature contract — ONE master list (MLB/NFL structural parity)
# ---------------------------------------------------------------------------
# The binary moneyline defines the production feature list below. Every other
# consumer PULLS it; none declares its own list. Member routing is a RULE over
# this list (linear members -> the list minus RAW_PER_SIDE_COLS; tree members
# -> the full list + the categorical team-ID pair).
MONEYLINE_FEATURE_COLS = [
    # served diffs (home − away; positive = home advantage)
    "elo_diff", "win_pct_diff", "rest_days_diff",
    "ewm_net_goals_diff", "ewm_goal_share_diff",
    "ga_per_game_diff", "shots_for_per_game_diff",
    "shots_against_per_game_diff", "pp_success_diff",
    "faceoff_win_diff", "back_to_back_diff",
    # goalie quality (season-to-date rolling SV%/GAA diffs — the NHL analog
    # of MLB's starting-pitcher ERA/K-9 pair; pre-game facts from the
    # season's rolling goalie stats, NaN when a goalie has no prior starts)
    "goalie_sv_pct_diff", "goalie_gaa_diff", "goalie_starts_diff",
    # game-level facts
    "is_playoffs",
    # raw per-side levels (tree family's home/away representations).
    "elo_home", "elo_away",
    "win_pct_home", "win_pct_away",
    "ewm_net_goals_home", "ewm_net_goals_away",
    "rest_days_home", "rest_days_away",
    "goalie_sv_pct_home", "goalie_sv_pct_away",
    "goalie_gaa_home", "goalie_gaa_away",
    # constant home anchor last, so the linear member's positional contract
    # is unchanged by the raw-side block above (MLB parity)
    "is_home",
]

# ---------------------------------------------------------------------------
# NHL team IDs for the tree-member categorical context (NFL pattern: a stable
# abbreviation -> integer ID map declared ONCE, never derived from data
# order, so bundles map teams to the same IDs across processes/retrains).
# 32 NHL franchise abbreviations (official NHL API `abbrev`), fixed
# alphabetical order. Historical abbreviations (ATL, PHX, etc.) are NOT
# special-cased: they map to UNK_TEAM_ID like any unseen label.
# ---------------------------------------------------------------------------
NHL_TEAM_ID: dict[str, int] = {
    "ANA": 0, "BOS": 1, "BUF": 2, "CAR": 3, "CBJ": 4, "CGY": 5,
    "CHI": 6, "COL": 7, "DAL": 8, "DET": 9, "EDM": 10, "FLA": 11,
    "LAK": 12, "MIN": 13, "MTL": 14, "NJD": 15, "NSH": 16, "NYI": 17,
    "NYR": 18, "OTT": 19, "PHI": 20, "PIT": 21, "SEA": 22, "SJS": 23,
    "STL": 24, "TBL": 25, "TOR": 26, "UTA": 27, "VAN": 28, "VGK": 29,
    "WPG": 30, "WSH": 31,
}
UNK_TEAM_ID = 99


def team_category_id(abbr: object) -> int:
    """Map a team abbreviation to its categorical ID (UNK_TEAM_ID fallback)."""
    if not isinstance(abbr, str):
        return UNK_TEAM_ID
    return NHL_TEAM_ID.get(abbr.strip().upper(), UNK_TEAM_ID)


# Adopted categorical set (tree members only; home first, away second).
TREE_CATEGORICAL_COLS = ["home_team_id", "away_team_id"]

# ---------------------------------------------------------------------------
# Candidate pool (RFE trial space). Declared ONCE here; served names are
# derived. Every candidate rides features._trailing_ewm / _trailing_per_team
# so the shift(1) leakage discipline is inherited, never reimplemented.
# ---------------------------------------------------------------------------
TEAM_CANDIDATE_TRAILING_SPECS: dict[str, tuple[str, ...]] = {
    "pp_goals_pg": ("ewm", "roll"),        # power-play goals per game
    "pp_attempts_pg": ("ewm", "roll"),    # power-play opportunities per game
    "pim_pg": ("roll",),                  # penalty minutes per game (discipline)
    "hits_pg": ("ewm", "roll"),           # physicality
    "blocked_shots_pg": ("ewm", "roll"),  # shot-blocking volume
    "giveaways_pg": ("ewm",),             # puck management
    "takeaways_pg": ("ewm",),             # puck management
    "goal_diff_pg": ("ewm", "roll"),      # raw scoring margin per game
    "sog_pg": ("ewm", "roll"),            # shot volume
    "faceoff_win_pct": ("ewm",),          # possession-start quality
    "pp_success_rate": ("ewm",),          # PP conversion rate
}

CANDIDATE_FAMILIES: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    "nhl": {"base": TEAM_CANDIDATE_TRAILING_SPECS},
}

# Served candidate names, derived from the specs — never hand-listed.
NHL_CANDIDATE_COLS: list[str] = list(dict.fromkeys(
    f"nhl_{metric}_{window}_{rep}"
    for spec in CANDIDATE_FAMILIES.values()
    for base_spec in spec.values()
    for metric, windows in base_spec.items()
    for window in windows
    for rep in ("diff", "home", "away")
))

# Member-family routing over the master list — a SELECTOR, not a feature list.
RAW_PER_SIDE_COLS = frozenset({
    "elo_home", "elo_away",
    "win_pct_home", "win_pct_away",
    "ewm_net_goals_home", "ewm_net_goals_away",
    "rest_days_home", "rest_days_away",
    "goalie_sv_pct_home", "goalie_sv_pct_away",
    "goalie_gaa_home", "goalie_gaa_away",
} | {f"nhl_{m}_{w}_{s}"
     for spec in CANDIDATE_FAMILIES.values()
     for base_spec in spec.values()
     for m, ws in base_spec.items() for w in ws for s in ("home", "away")})

# The full candidate list (RFE trial space), defined ONCE.
RFE_CANDIDATE_COLS: list[str] = list(dict.fromkeys(
    [c for c in NHL_CANDIDATE_COLS if c not in set(MONEYLINE_FEATURE_COLS)]))

# Trial / validation pool: universe first (canonical), then candidates.
KNOWN_FEATURE_COLS = list(dict.fromkeys(
    list(MONEYLINE_FEATURE_COLS) + list(RFE_CANDIDATE_COLS)))

# RFE governance: None during ordinary production runs.
_FEATURE_SUBSET: list[str] | None = None


def active_moneyline_feature_cols() -> list[str]:
    """Model-facing contract: adopted RFE subset, else the full universe."""
    return list(_FEATURE_SUBSET) if _FEATURE_SUBSET is not None \
        else list(MONEYLINE_FEATURE_COLS)


def set_feature_subset(cols: list[str]) -> None:
    """Apply an adopted subset, rebuilt in canonical pool order."""
    global _FEATURE_SUBSET
    if len(cols) < 1:
        raise ValueError("invalid NHL feature subset: empty")
    unknown = [c for c in cols if c not in KNOWN_FEATURE_COLS]
    if unknown:
        raise ValueError(f"NHL feature subset contains non-pool columns: {unknown[:6]}")
    chosen = set(cols)
    _FEATURE_SUBSET = [c for c in KNOWN_FEATURE_COLS if c in chosen]


def reset_feature_subset() -> None:
    global _FEATURE_SUBSET
    _FEATURE_SUBSET = None


def assert_candidate_manifest_parity() -> list[str]:
    """Declared candidates <-> manifest.CANDIDATE_MANIFEST, name-for-name."""
    try:
        from backend import manifest as _m
    except ImportError:  # running as a top-level module
        import manifest as _m
    problems: list[str] = []
    declared = list(RFE_CANDIDATE_COLS)
    documented = list(_m.CANDIDATE_MANIFEST)
    for f in declared:
        if f not in documented:
            problems.append(f"declared candidate {f!r} missing from candidate manifest")
    for f in documented:
        if f not in declared:
            problems.append(f"candidate-manifest entry {f!r} is not a declared candidate")
    return problems

# ---------------------------------------------------------------------------
# Ensemble members (moneyline) — MLB-IDENTICAL tuned params
# ---------------------------------------------------------------------------
ENSEMBLE_MEMBERS = ["xgboost", "lightgbm", "elasticnet"]

# Fallback prior weights: fold 0 blends on these (equal thirds); after every
# fold the blend weights are re-earned by moneyline.compute_adaptive_weights
# (simplex SLSQP minimizing pooled OOF log-loss in LOGIT space, no floor/cap).
ENSEMBLE_WEIGHTS = {
    "xgboost": 1 / 3,
    "lightgbm": 1 / 3,
    "elasticnet": 1 / 3,
}

# MLB production values (mlb-backend/backend/config.py, L5 re-tune 2026-09-21)
# — the NHL ensemble is deliberately the identical ensemble.
XGBOOST_PARAMS = {
    "max_depth": 3,
    "min_child_weight": 12,
    "gamma": 2.4178,
    "subsample": 0.812,
    "colsample_bytree": 0.6382,
    "learning_rate": 0.1097,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
    "enable_categorical": True,
}
XGBOOST_FOLD_ROUNDS = 2000
XGBOOST_EARLY_STOP = 20
LIGHTGBM_PARAMS = {
    "n_estimators": 50,
    "max_depth": 6,
    "num_leaves": 6,
    "min_child_samples": 70,
    "min_gain_to_split": 1.2224,
    "bagging_fraction": 0.4518,
    "bagging_freq": 1,
    "feature_fraction": 0.7632,
    "learning_rate": 0.0332,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}
ELASTICNET_PARAMS = {
    "penalty": "elasticnet",
    "l1_ratio": 0.5,
    "C": 0.03,
    "solver": "saga",
    "max_iter": 4000,
    "tol": 1e-4,
    "random_state": RANDOM_SEED,
}

# Regression members (goals + total regressions) — the run-line model's
# per-side Poisson regressors (MLB/NFL structural parity).
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
# Goal / total distribution grids (run-line model)
# ---------------------------------------------------------------------------
SPREAD_GRID = list(range(-8, 9))            # margin thresholds L: -8..+8
TOTAL_GRID = list(range(4, 13))             # totals U: 4..12
# Canonical lines the Run-Engine Model card scores per-line OOF metrics at.
RUN_ENGINE_FIXED_TOTALS = (5, 6, 7)
RUN_ENGINE_CANONICAL_SPREADS = (1, 2)
# Half-stop lines the ±0.5 derived-ML stop prices from.
HALF_STOP_LINES = [-0.5, 0.5]
# Goal/total PMF support (integer points; NB base)
MARGIN_PMF_MAX = 12
TOTAL_PMF_MAX = 18

# Distribution model defaults
MARGIN_SIGMA = 2.2          # initial NHL goal-margin sigma (goals)
TOTAL_SIGMA = 3.2           # initial total sigma (goals)
SIGMA_FLOOR_MARGIN = 1.5
SIGMA_CAP_MARGIN = 3.5
SIGMA_FLOOR_TOTAL = 2.0
SIGMA_CAP_TOTAL = 5.0
P_TIE_MAX = 0.25            # cap on the tied-margin mass (hockey ties ~5-8%)

# ---------------------------------------------------------------------------
# Artifact naming (frontend family contracts)
# ---------------------------------------------------------------------------
DATE_FMT = "%Y%m%d"

MONEYLINE_JSON = "nhl_moneyline_v1_{date}.json"
CALIBRATION_JSON = "nhl_calibration_{date}.json"
PREDICTIONS_HISTORY_CSV = "nhl_predictions_history_{date}.csv"
POWER_RANKINGS_CSV = "nhl_power_rankings_{date}.csv"
MARKETS_CSV = "nhl_run_engine_markets_{date}.csv"
MARKETS_META_JSON = "nhl_run_engine_markets_{date}.meta.json"
MARKETS_MONITOR_JSON = "nhl_run_engine_monitor_{date}.json"
GOALIE_MATCHUP_JSON = "nhl_goalie_matchup_{date}.json"
FEATURE_JSON = "nhl_feature_v1_{date}.json"
MODEL_MONITOR_JSON = "nhl_model_monitor_{date}.json"
SHAP_GAME_PREFIX = "nhl_shap_game"
MODEL_BUNDLE = MODELS_DIR / "nhl_ensemble_latest.joblib"
OOF_STORE_CSV = DATA_DELIVERY_DIR / "nhl_oof_store.csv"

# Goalie serving contract (enrichment only — never fabricated). The NHL
# analog of the NFL QB matchup: per-side starting-goalie blocks rendered as
# SV% · GAA on the game cards (the pitcher ERA · K/9 analog).
GOALIE_FIELDS = [
    "g_home_name", "g_home_sv_pct", "g_home_gaa", "g_home_starts",
    "g_away_name", "g_away_sv_pct", "g_away_gaa", "g_away_starts",
]

# Coin-flip threshold for model_pick display
COIN_FLIP_THRESHOLD = 0.02
