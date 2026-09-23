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
OPP_ADJ_WINDOW = 6    # opponent-adjusted trailing window (games; roll_opp)
OPP_ADJ_SHRINKAGE = 8.0  # games of opponent-defensive evidence before the prior is fully trusted

# RFE commit gate: a trial commits when its pooled logloss gain beats this
# multiple of the paired per-game SE (floor 0.0005). 1.0 = ~84% one-sided
# confidence — deliberately looser than the old 2.0 (97.5%); every trial
# stays visible in the trace/workbook either way.
RFE_COMMIT_SE_MULTIPLE = 1.0

# Precipitation thresholds for the committed weather table (inches).
PRECIP_FLAG_IN = 0.1
SNOW_FLAG_IN = 0.1
PACE_WINDOW = 4       # trailing plays/min window (games)
PBP_ROLL_WINDOW = 4   # trailing flat window for the pbp candidate metrics

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
FEATURE_SET_VERSION = "nfl-prod-v4-air-yards-promoted"

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
    # playing surface + observed game-day environment (static pre-game facts;
    # NaN for domes/international/missing rows — never fabricated)
    "is_turf_home", "temp_f", "wind_mph", "is_precip", "is_snow",
    # starter availability (weekly injury reports, pre-game facts; diffs of
    # Out/IR counts by position group)
    "inj_qb_out_diff", "inj_tackle_out_diff", "inj_edge_out_diff",
    "inj_starters_out_diff",
    # RFE promotion (2026-09-22 sweep, 1-SE gate): trailing passing depth,
    # the sweep's only committed addition. The diff serves every family; the
    # raw per-side levels route tree-only via RAW_PER_SIDE_COLS below.
    "pbp_air_yards_att_ewm_diff", "pbp_air_yards_att_ewm_home",
    "pbp_air_yards_att_ewm_away",
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

# ---------------------------------------------------------------------------
# pbp candidate-pool spec (2026-09-22 expansion)
# ---------------------------------------------------------------------------
# Per-game play-by-play metrics the feature engine rolls up (features.py
# _add_pbp_metrics), the trailing windows each is served at, and the served
# names derived from them — the RFE candidate list, defined ONCE here.
#
#   "ewm"  — per-team EWM (halflife EWM_HALFLIFE) of the per-game metric
#   "roll" — per-team 4-game mean (PBP_ROLL_WINDOW)
#   "roll_opp" — per-team OPP_ADJ_WINDOW-game mean of an opponent-ADJUSTED
#                series (see PBP_OPP_ADJ_TRAILING_SPECS below)
# All ride features._trailing_ewm / _trailing_per_team, so every candidate
# inherits the production shift(1) leakage discipline.
#
# Each <metric>_<window> serves THREE candidates:
#   pbp_<metric>_<window>_diff  home−away gap (every model family)
#   pbp_<metric>_<window>_home  raw home level (tree family only)
#   pbp_<metric>_<window>_away  raw away level (tree family only)
# Candidates stay OUT of serving width until an RFE adoption promotes them;
# being declared here only makes them triable (features._attach_pbp_candidate
# _features still generates the columns so trials score real signal, and a
# declared-but-ungenerated column is narrowed out of trials by _trial_space).
PBP_CANDIDATE_TRAILING_SPECS: dict[str, tuple[str, ...]] = {
    # Offensive efficiency beyond yards (High impact / Low cost in the audit)
    "epa_play": ("ewm", "roll"),
    "qb_epa_dropback": ("ewm",),
    "cpoe_play": ("ewm",),          # needs pbp cache v2 (air_yards-family pull)
    "air_yards_att": ("ewm",),      # needs pbp cache v2
    "yac_epa_att": ("ewm",),        # YAC-as-EPA per attempt (nflreadpy has no raw yac; pbp cache v3)
    # Opponent-adjusted EPA lives in PBP_OPP_ADJ_TRAILING_SPECS below (the
    # ladder pre-pass produces it from epa_play x def_epa_play).
    # Defensive efficiency: EPA allowed per play on the team's defensive
    # snaps (lower = stingier defense; the served diff is home minus away).
    "def_epa_play": ("ewm", "roll"),
    # Turnover margin as two symmetric sides (protection vs takeaway edge)
    "turnovers": ("ewm", "roll"),
    "takeaways": ("ewm", "roll"),
    # Pressure / playcalling (OL + tendency)
    "sack_rate": ("ewm",),
    "dropback_rate": ("ewm",),
    # Situational strength
    "third_down_rate": ("ewm",),
    "redzone_td_rate": ("ewm",),
    "start_field_pos": ("ewm",),
    # Discipline (flat 4-game volume window)
    "penalty_yards_pg": ("roll",),
    "penalties_pg": ("roll",),
    # Special teams
    "fg_accuracy": ("ewm", "roll"),
    # Formation/pace complements to pace_plays_min
    "shotgun_rate": ("ewm",),
    "no_huddle_rate": ("ewm",),
    "drives_pg": ("roll",),
}

# ---------------------------------------------------------------------------
# Opponent-adjustment pre-pass (features.team_stats_ladder)
# ---------------------------------------------------------------------------
# base per-game metric -> the DEFENSIVE metric it is adjusted against. The
# adjusted series  <base>_opp_adj = base + (prior league mean of the defense
# metric − the opponent's shrunk prior EWM of it)  rewards production against
# good defenses and discounts production against bad ones, using ONLY
# strictly-prior information: the opponent strength is the opponent's own
# shift(1) halflife-EWM shrunk toward the prior expanding league mean
# (OPP_ADJ_SHRINKAGE games of evidence before full trust) — the leakage
# contract is unchanged, and the adjusted series then rides the ordinary
# trailing windows in PBP_OPP_ADJ_TRAILING_SPECS.
PBP_OPP_ADJ_METRICS: dict[str, str] = {
    "epa_play": "def_epa_play",
}

# Trailing windows for the opponent-adjusted series (same causal primitives;
# the flat window is the long-standing OPP_ADJ_WINDOW knob, finally armed).
PBP_OPP_ADJ_TRAILING_SPECS: dict[str, tuple[str, ...]] = {
    "epa_play_opp_adj": ("ewm", "roll_opp"),
}

# Skill-position usage (weekly player stats): per-(game, team) shares served
# at the same causal windows. Family prefix ps_.
PS_CANDIDATE_TRAILING_SPECS: dict[str, tuple[str, ...]] = {
    # RB carries / team rushing attempts (committee vs workhorse backfield)
    "rb_load_share": ("ewm", "roll"),
    # max WR target share (a true WR1 threat vs target dispersal)
    "wr1_target_share": ("ewm", "roll"),
}

# Next-Gen Stats weekly tracking efficiency, trailed like every per-game
# metric. The family prefix makes the served names (ngs_cpoe_* etc.); week-0
# season aggregates never reach here.
NGS_CANDIDATE_TRAILING_SPECS: dict[str, tuple[str, ...]] = {
    "cpoe": ("ewm", "roll"),      # NGS completion % above expectation (QBs)
    "rush_eff": ("ewm", "roll"),  # NGS rush yards over expected / attempt
    "sep": ("ewm", "roll"),       # NGS avg separation (WR/TE receiving)
}

# family prefix -> the trailing specs behind it (served candidate names are
# DERIVED from this — never hand-listed).
CANDIDATE_FAMILIES: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    "pbp": {
        "base": PBP_CANDIDATE_TRAILING_SPECS,
        "opp_adj": PBP_OPP_ADJ_TRAILING_SPECS,
    },
    "ps": {"base": PS_CANDIDATE_TRAILING_SPECS},
    "ngs": {"base": NGS_CANDIDATE_TRAILING_SPECS},
}

# Served candidate names, derived from the specs — never hand-listed.
PBP_CANDIDATE_COLS: list[str] = list(dict.fromkeys(
    f"{family}_{metric}_{window}_{rep}"
    for family, specs in CANDIDATE_FAMILIES.items()
    for spec in specs.values()
    for metric, windows in spec.items()
    for window in windows
    for rep in ("diff", "home", "away")
))

# Member-family routing over the master list — a SELECTOR, not a feature list.
# Mirrors MLB's training.RAW_PER_SIDE_COLS: the linear/MLP family consumes the
# diff/anchor view, tree members consume every column. Includes the pbp
# candidate raw levels too: an adopted candidate promotes into serving width,
# where the same routing rule must apply (a promoted raw level must never
# reach the linear members).
RAW_PER_SIDE_COLS = frozenset({
    "elo_home", "elo_away",
    "win_pct_home", "win_pct_away",
    "ewm_net_pts_home", "ewm_net_pts_away",
    "ewm_ypp_home", "ewm_ypp_away",
    "rest_days_home", "rest_days_away",
} | {f"{family}_{m}_{w}_{s}"
     for family, specs in CANDIDATE_FAMILIES.items()
     for spec in specs.values()
     for m, ws in spec.items() for w in ws for s in ("home", "away")})

# The full candidate list (RFE trial space), defined ONCE. Additions may only
# name these; the RFE never derives candidates from a frame. A candidate must
# be a PIT-safe, pre-game column the feature engine produces and that is NOT
# already in the universe above — RFE promotions leave the trial space here
# (the structural promotion lives in MONEYLINE_FEATURE_COLS, not in an RFE
# adoption record, so the universe stays the single source of truth).
RFE_CANDIDATE_COLS: list[str] = [c for c in PBP_CANDIDATE_COLS
                                 if c not in set(MONEYLINE_FEATURE_COLS)]

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


def assert_candidate_manifest_parity() -> list[str]:
    """Declared candidates <-> manifest.CANDIDATE_MANIFEST, name-for-name.

    The pipeline calls this before the RFE phase so an undocumented candidate
    (a spec name with no manifest entry, or a stale manifest entry) fails
    loudly at run start instead of surfacing as blank workbook rows inside a
    trial. Returns the problem list (empty = parity)."""
    try:
        from backend import manifest as _m
    except ImportError:  # running as a top-level module
        import manifest as _m
    problems: list[str] = []
    declared = list(RFE_CANDIDATE_COLS)   # the effective trial space (post-promotion)
    documented = list(_m.CANDIDATE_MANIFEST)
    for f in declared:
        if f not in documented:
            problems.append(f"declared candidate {f!r} missing from candidate manifest")
    for f in documented:
        if f not in declared:
            problems.append(f"candidate-manifest entry {f!r} is not a declared candidate")
    return problems

# ---------------------------------------------------------------------------
# Ensemble members (moneyline) — NFL-specific hyperparameters
# ---------------------------------------------------------------------------
ENSEMBLE_MEMBERS = ["xgboost", "lightgbm", "elasticnet"]

# Fallback prior weights: fold 0 blends on these (1/3 each); after every
# fold the blend weights are re-earned by moneyline.compute_adaptive_weights
# (MLB structural parity: simplex SLSQP minimizing pooled OOF log-loss in
# LOGIT space, re-earned from strictly-prior evidence, no floor/cap).
ENSEMBLE_WEIGHTS = {
    "xgboost": 1 / 3,
    "lightgbm": 1 / 3,
    "elasticnet": 1 / 3,
}

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
