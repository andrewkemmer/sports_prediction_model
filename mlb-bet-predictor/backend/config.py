"""Central configuration for the MLB Bet Predictor.

Everything that tunes the pipeline lives here so Colab runs, tests, and the
Streamlit frontend all agree on one source of truth.

Conventions
-----------
* All paths are built with :mod:`pathlib.Path` relative to the repository root
  (``data_delivery`` lives at the repo root, per the spec).
* No secrets live here. GitHub credentials are read from environment variables
  at runtime (see ``github_sync.py``).
* ``SYNTHETIC_DATA`` defaults to ``True`` so the pipeline runs end-to-end in
  Colab without depending on live MLB APIs. Set it to ``False`` to use the
  real ``pybaseball`` ingestion path.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Global paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]           # repo root
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"

DATA_DIR = ROOT_DIR / "data_delivery"                    # canonical artifact sink
MODELS_DIR = DATA_DIR / "models"                         # persisted ensembles
SHAP_DIR = DATA_DIR                                     # per-game SHAP CSVs live at the
                                                         # data_delivery root (easy raw-URL fetch)
RAW_DIR = DATA_DIR / "raw"                               # cached raw API payloads (gitignored)
LOGS_DIR = DATA_DIR / "logs"                             # pipeline run logs (gitignored)

# Ensure artifact directories exist at import time.
for _d in (DATA_DIR, MODELS_DIR, SHAP_DIR, RAW_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Tracking strings (file name patterns, prefixes, metadata keys)
# ---------------------------------------------------------------------------
TODAYS_GAMES_FILE = "todays_games_{yyyymmdd}.csv"
POWER_RANKINGS_FILE = "power_rankings_{yyyymmdd}.csv"
CALIBRATION_FILE = "calibration_{yyyymmdd}.json"
MODEL_MONITOR_FILE = "model_monitor_{yyyymmdd}.json"
FEATURE_DRIFT = "feature_drift_{yyyymmdd}.csv"
SHAP_GAME_PREFIX = "shap_game_"
SHAP_GAME_FILE = "shap_game_{game_id}.csv"
ENSEMBLE_FILE = "ensemble_latest.joblib"
MODEL_MANIFEST_FILE = "model_manifest.json"

# Metadata keys persisted with every retrain (and surfaced in Model Monitor).
VERSION = "VERSION"
TRAINED_AT = "TRAINED_AT"
DATA_CUTOFF = "DATA_CUTOFF"
MODEL_TYPE = "MODEL_TYPE"
FEATURE_COLUMNS = "FEATURE_COLUMNS"
N_TRAIN_GAMES = "N_TRAIN_GAMES"
METRICS = "METRICS"
EVALUATION_NOTES = "EVALUATION_NOTES"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
N_JOBS = -1

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# Rolling windows (number of prior games) for point-in-time aggregates.
ROLL_WINDOWS = {
    "woba": 30,            # rolling wOBA
    "runs_for": 30,        # rolling runs scored
    "runs_against": 30,    # rolling runs allowed
    "bullpen_whip": 10,    # bullpen WHIP (10g)
    "sp_era": 30,          # starting pitcher ERA (30g)
    "sp_k9": 30,           # starting pitcher K/9 (30g)
    "record": None,        # career-to-date record % (expanding window)
}

# Elo
ELO_K = 20
ELO_HOME_ADVANTAGE = 25.0
ELO_MOV_MULTIPLIER = 1.0   # margin-of-victory factor (K * log(1 + margin) * mult)

# Walk-forward training
RETRAIN_CADENCE_DAYS = 7          # weekly retrain cadence
MIN_TRAIN_DAYS = 90               # warm-up history before first validation window
VALIDATION_WINDOW_DAYS = 7
MIN_TRAIN_GAMES = 200             # refuse to train with fewer historical games
N_CALIBRATION_BINS = 10           # ECE / reliability-diagram bins

# Model hyperparameters (per head)
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "eval_metric": "logloss",
    "verbosity": 0,
}
LGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "verbosity": -1,
}
LR_PARAMS = {"C": 1.0, "max_iter": 2000, "solver": "lbfgs", "random_state": SEED}

# ---------------------------------------------------------------------------
# PSI thresholds (drift detection)
# ---------------------------------------------------------------------------
PSI_WARN = 0.10     # 0.10 <= psi < 0.25  -> WARN
PSI_ALERT = 0.25    # psi >= 0.25         -> ALERT
PSI_N_BUCKETS = 10

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
# Canonical model feature columns (order matters for training / SHAP).
FEATURE_COLUMNS = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "home_rest_days",
    "away_rest_days",
    "home_woba_30g",
    "away_woba_30g",
    "home_runs_for_30g",
    "home_runs_against_30g",
    "home_bullpen_whip_10g",
    "away_bullpen_whip_10g",
    "sp_home_era",
    "sp_home_k9",
    "sp_away_era",
    "sp_away_k9",
    "home_record_pct",
    "away_record_pct",
    "home_field",
    "market_vig",
    "total_line",
    "run_line",
    "weather_wind_speed",
]

# Friendly labels used for SHAP charts and drift tables.
FEATURE_LABELS = {
    "elo_diff": "Elo diff",
    "home_elo": "Home Elo",
    "away_elo": "Away Elo",
    "home_rest_days": "Home rest days",
    "away_rest_days": "Away rest days",
    "home_woba_30g": "Home wOBA (30g)",
    "away_woba_30g": "Away wOBA (30g)",
    "home_runs_for_30g": "Home runs scored (30g)",
    "home_runs_against_30g": "Home runs allowed (30g)",
    "home_bullpen_whip_10g": "Home bullpen WHIP (10g)",
    "away_bullpen_whip_10g": "Away bullpen WHIP (10g)",
    "sp_home_era": "Home SP ERA",
    "sp_home_k9": "Home SP K/9",
    "sp_away_era": "Away SP ERA",
    "sp_away_k9": "Away SP K/9",
    "home_record_pct": "Home team record",
    "away_record_pct": "Away team record",
    "home_field": "Home field",
    "market_vig": "Market vig",
    "total_line": "Total line",
    "run_line": "Run line",
    "weather_wind_speed": "Wind speed",
}

# Features tracked for drift (subset that is cheap to compute daily).
DRIFT_FEATURES = [
    "home_team_elo",
    "away_sp_era_10g",
    "bullpen_whip_10g",
    "home_woba_30g",
    "weather_wind_speed",
]
DRIFT_FEATURE_LABELS = {
    "home_team_elo": "Home team ELO",
    "away_sp_era_10g": "Away SP ERA (10g)",
    "bullpen_whip_10g": "Bullpen WHIP (10g)",
    "home_woba_30g": "Home wOBA (30g)",
    "weather_wind_speed": "Wind speed (mph)",
}

# ---------------------------------------------------------------------------
# Targets / model heads
# ---------------------------------------------------------------------------
TARGETS = {
    "moneyline": "home_win",      # classification: P(home wins)
    "totals": "total_runs",       # regression: projected total runs
    "runline": "home_cover",      # classification: P(home covers -1.5)
}

# ---------------------------------------------------------------------------
# Market / odds helpers
# ---------------------------------------------------------------------------
JUICE_FLOOR = 0.015               # ignore juice below this (rounding noise)
EVEN_MONEY_EPS = 1e-4

# ---------------------------------------------------------------------------
# GitHub sync
# ---------------------------------------------------------------------------
# No secrets here. Resolved from the environment at runtime.
GITHUB_REPO_URL = os.environ.get("GITHUB_REPO_URL", "")          # full https/ssh URL
GITHUB_OWNER = os.environ.get("GITHUB_OWNER", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"                                # PAT env var name
REPO_LOCAL_CLONE = Path(os.environ.get("REPO_LOCAL_CLONE", "/content/mlb-bet-predictor"))

# ---------------------------------------------------------------------------
# Sports scope scaffolding (hooks for future leagues)
# ---------------------------------------------------------------------------
SUPPORTED_SPORTS = {
    "mlb": {"primary": True, "season_start_month": 3, "season_end_month": 11},
    "nba": {"primary": False, "season_start_month": 10, "season_end_month": 6},
    "nhl": {"primary": False, "season_start_month": 10, "season_end_month": 6},
    "nfl": {"primary": False, "season_start_month": 9, "season_end_month": 2},
    "cfb": {"primary": False, "season_start_month": 8, "season_end_month": 1},
    "ncaab": {"primary": False, "season_start_month": 11, "season_end_month": 4},
    "tennis": {"primary": False, "season_start_month": 1, "season_end_month": 12},
}


def date_to_yyyymmdd(d: object) -> str:
    """Return ``YYYYMMDD`` for a date, date-like string, or Timestamp."""
    import pandas as pd

    ts = pd.Timestamp(d)
    return f"{ts.year:04d}{ts.month:02d}{ts.day:02d}"


def artifact_path(template: str, d: object) -> Path:
    """Resolve a ``{yyyymmdd}``-template into a real path under data_delivery."""
    return DATA_DIR / template.format(yyyymmdd=date_to_yyyymmdd(d))
