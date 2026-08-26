"""Canonical feature metadata → features_metadata_<date>.json artifact.

Single source of truth for dashboard tooltips (Feature Drift Analysis) and any
consumer that needs to explain what a feature IS. Entries are keyed by exact
FEATURE_COLS names; generation walks FEATURE_COLS itself so a newly added
feature automatically appears (authored entry) or triggers a LOUD WARNING plus
a clearly-marked placeholder (never a silent gap).

Member routing is DERIVED from the live feature-routing config at generation
time (training._logistic_feature_cols honors LOGISTIC_USE_RAW_COLS) — never
hardcoded here. Trees + MLP consume every feature; logistic sees diffs only.

The one-line summaries are the dashboard's existing blurbs, moved here so
frontend and pipeline read ONE source.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from config import DATA_DELIVERY_DIR

logger = logging.getLogger(__name__)

TREE_MEMBERS = ["xgboost", "lightgbm", "randomforest", "mlp"]
_ALL_MEMBERS = TREE_MEMBERS + ["logistic"]

# ---------------------------------------------------------------------------
# Rich authored entries. Keys must exactly match FEATURE_COLS names; anything
# missing at generation time triggers a loud warning + placeholder.
# ---------------------------------------------------------------------------
_RICH: dict[str, dict[str, str]] = {
    # ---- baseline -------------------------------------------------------
    "is_home": {
        "summary": "Always 1 — anchors the ~53% MLB home-field win advantage",
        "definition": (
            "Constant intercept column marking the home side. Every row is 1 "
            "because the model always scores the home team's chance of winning."
        ),
        "formula": "1",
        "source": "Feature engineering: static column added to every game row",
        "window": "n/a (constant)",
        "units": "binary",
        "direction": "n/a (constant)",
    },
    # ---- core pre-game diffs --------------------------------------------
    "win_pct_diff": {
        "summary": "Home win% − away win% (smoothed to .500 early season)",
        "definition": (
            "Season-long record quality gap between the teams, shrunk toward "
            ".500 early in the season so tiny samples don't dominate."
        ),
        "formula": "smoothed_win_pct(home_wins, home_losses) − smoothed_win_pct(away_wins, away_losses)",
        "source": "DuckDB feature engineering: official results records",
        "window": "season to date",
        "units": "win% (0–1)",
        "direction": "higher = home advantage",
    },
    "elo_diff": {
        "summary": "Home Elo − away Elo (skill-gap anchor, updated each game)",
        "definition": (
            "Elo ratings updated after every completed game (K=20, home "
            "advantage 65 pts, off-season regression to the mean)."
        ),
        "formula": "home_elo − away_elo",
        "source": "DuckDB feature engineering: Elo engine over full game log",
        "window": "all prior games (decaying)",
        "units": "Elo points",
        "direction": "higher = home advantage",
    },
    "rest_days_diff": {
        "summary": "Home rest days − away rest days (schedule fatigue)",
        "definition": "Days since each team's previous game.",
        "formula": "rest_days_home − rest_days_away",
        "source": "Schedule: gap between consecutive game dates",
        "window": "per-game",
        "units": "days",
        "direction": "higher = home advantage (more rested)",
    },
    # ---- SP season + last-5 diffs ----------------------------------------
    "sp_era_diff": {
        "summary": "Home SP season-to-date ERA − away SP",
        "definition": "Starting-pitcher quality gap on season-long earned-run average.",
        "formula": "sp_era_home − sp_era_away",
        "source": "Statcast pitching aggregates (season to date)",
        "window": "season to date",
        "units": "ERA runs",
        "direction": "lower = home advantage (ERA is a cost)",
    },
    "sp_era_5g_diff": {
        "summary": "Home SP last-5-start ERA − away SP (recent form)",
        "definition": "Same ERA gap but only each pitcher's last 5 starts — captures hot/cold streaks.",
        "formula": "sp_era_5g_home − sp_era_5g_away",
        "source": "Statcast pitching aggregates (last-5-start window)",
        "window": "5g",
        "units": "ERA runs",
        "direction": "lower = home advantage",
    },
    "sp_k9_diff": {
        "summary": "Home SP season-to-date K/9 − away SP",
        "definition": "Strikeout rate gap: strikeouts per 9 innings, season to date.",
        "formula": "sp_k9_home − sp_k9_away",
        "source": "Statcast pitching aggregates (season to date)",
        "window": "season to date",
        "units": "K/9",
        "direction": "higher = home advantage",
    },
    "sp_k9_5g_diff": {
        "summary": "Home SP last-5-start K/9 − away SP (recent form)",
        "definition": "Strikeout-rate gap over each pitcher's last 5 starts.",
        "formula": "sp_k9_5g_home − sp_k9_5g_away",
        "source": "Statcast pitching aggregates (last-5-start window)",
        "window": "5g",
        "units": "K/9",
        "direction": "higher = home advantage",
    },
    # ---- SP trailing-3 stuff diffs ----------------------------------------
    "sp_fbvelo_diff": {
        "summary": "Home SP fastball velo (last 3 starts) − away SP (mph)",
        "definition": "Average four-seam/fastball velocity gap over recent starts.",
        "formula": "sp_fbvelo_3g_home − sp_fbvelo_3g_away",
        "source": "Statcast pitch-level: mean fastball speed, last-3-start window",
        "window": "3g",
        "units": "mph",
        "direction": "higher = home advantage",
    },
    "sp_fbpct_diff": {
        "summary": "Home SP fastball usage (last 3 starts) − away SP",
        "definition": "How heavily each pitcher is leaning on the fastball right now.",
        "formula": "sp_fbpct_3g_home − sp_fbpct_3g_away",
        "source": "Statcast pitch-level: fastball share of pitches, last-3-start window",
        "window": "3g",
        "units": "share (0–1)",
        "direction": "n/a (mix signal)",
    },
    "sp_whiff_diff": {
        "summary": "Home SP whiff rate (last 3 starts) − away SP",
        "definition": "Swinging-strike rate generated per swing over the last 3 starts — raw stuff indicator.",
        "formula": "sp_whiff_3g_home − sp_whiff_3g_away",
        "source": "Statcast pitch-level: whiffs / swings, last-3-start window",
        "window": "3g",
        "units": "rate (0–1)",
        "direction": "higher = home advantage",
    },
    # ---- SP xwOBA allowed -------------------------------------------------
    "sp_xwoba_diff": {
        "summary": "Home SP last-6-start xwOBA allowed − away SP",
        "definition": "Expected weighted-OBA conceded to opposing batters — contact-quality-based pitcher effectiveness.",
        "formula": "sp_xwoba_home − sp_xwoba_away",
        "source": "Statcast xwOBA on balls in play + K/BB, last-6-start window",
        "window": "6g",
        "units": "xwOBA",
        "direction": "lower = home advantage",
    },
    "sp_xwoba_vs_l_diff": {
        "summary": "Home SP xwOBA vs LHB (season to date) − away SP",
        "definition": "Platoon exposure: expected production allowed specifically to left-handed batters.",
        "formula": "sp_xwoba_vs_l_home − sp_xwoba_vs_l_away",
        "source": "Statcast xwOBA split by batter handedness (season)",
        "window": "season to date",
        "units": "xwOBA",
        "direction": "lower = home advantage",
    },
    # ---- lineup wOBA ------------------------------------------------------
    "lineup_woba_mean_diff": {
        "summary": "Home lineup avg wOBA − away lineup avg wOBA",
        "definition": "Projected nine-man lineup quality gap (each hitter's wOBA shrunk toward league mean by sample size).",
        "formula": "lineup_woba_mean_home − lineup_woba_mean_away",
        "source": "Statcast hitter aggregates, projected lineups",
        "window": "season to date (shrunk)",
        "units": "wOBA points",
        "direction": "higher = home advantage",
    },
    "lineup_woba_top3_diff": {
        "summary": "Home top-3 hitter wOBA − away top-3 hitter wOBA",
        "definition": "Star-power gap at the top of the card.",
        "formula": "lineup_woba_top3_home − lineup_woba_top3_away",
        "source": "Statcast hitter aggregates, projected lineups",
        "window": "season to date (shrunk)",
        "units": "wOBA points",
        "direction": "higher = home advantage",
    },
    "lineup_woba_std_diff": {
        "summary": "Home lineup wOBA dispersion − away lineup dispersion",
        "definition": "Depth signal: low std = deep balanced lineup; high std = stars-and-scrubs.",
        "formula": "lineup_woba_std_home − lineup_woba_std_away",
        "source": "Statcast hitter aggregates, projected lineups",
        "window": "season to date (shrunk)",
        "units": "wOBA points (std)",
        "direction": "n/a (shape signal)",
    },
    "woba_30g_diff": {
        "summary": "Home team 30-game wOBA − away team 30-game wOBA",
        "definition": "Team-level offensive form over the trailing month.",
        "formula": "woba_30g_home − woba_30g_away",
        "source": "Statcast team batting aggregates",
        "window": "30g",
        "units": "wOBA points",
        "direction": "higher = home advantage",
    },
    # ---- bullpen ----------------------------------------------------------
    "bullpen_whip_diff": {
        "summary": "Home bullpen 10-game WHIP − away bullpen (lower = better)",
        "definition": "Relief corps baserunner allowance over the last 10 games.",
        "formula": "bullpen_whip_10g_home − bullpen_whip_10g_away",
        "source": "Statcast relief-pitching aggregates",
        "window": "10g",
        "units": "WHIP",
        "direction": "lower = home advantage",
    },
    "bullpen_whip_3g_diff": {
        "summary": "Home bullpen 3-game WHIP − away bullpen (short-term form)",
        "definition": "Very recent bullpen form; noisy but catches slumps fast.",
        "formula": "bullpen_whip_3g_home − bullpen_whip_3g_away",
        "source": "Statcast relief-pitching aggregates",
        "window": "3g",
        "units": "WHIP",
        "direction": "lower = home advantage",
    },
    "bullpen_pitches_diff": {
        "summary": "Home bullpen 3-day pitch count − away (fatigue signal)",
        "definition": "Cumulative relief workload over the last three days — availability and fatigue.",
        "formula": "bullpen_pitches_3d_home − bullpen_pitches_3d_away",
        "source": "Statcast pitch counts by reliever by date",
        "window": "3d",
        "units": "pitches",
        "direction": "lower = home advantage (fresher arms)",
    },
    # ---- contact form -----------------------------------------------------
    "team_barrel_diff": {
        "summary": "Home barrel% (15g) − away barrel% (quality of contact)",
        "definition": "Barreled-ball rate gap (optimal exit velo × launch angle buckets).",
        "formula": "team_barrel_15g_home − team_barrel_15g_away",
        "source": "Statcast batted-ball data",
        "window": "15g",
        "units": "rate (0–1)",
        "direction": "higher = home advantage",
    },
    "team_hardhit_diff": {
        "summary": "Home hard-hit% (15g) − away hard-hit%",
        "definition": "Share of batted balls ≥95 mph over the last 15 games.",
        "formula": "team_hardhit_15g_home − team_hardhit_15g_away",
        "source": "Statcast batted-ball data",
        "window": "15g",
        "units": "rate (0–1)",
        "direction": "higher = home advantage",
    },
    "team_exitvelo_diff": {
        "summary": "Home avg exit velo (15g) − away avg exit velo (mph)",
        "definition": "Average exit velocity on balls in play — raw-contact strength.",
        "formula": "team_exitvelo_15g_home − team_exitvelo_15g_away",
        "source": "Statcast batted-ball data",
        "window": "15g",
        "units": "mph",
        "direction": "higher = home advantage",
    },
    # ---- matchup context ----------------------------------------------------
    "lineup_handedness_matchup_advantage": {
        "summary": "Lineup OPS vs tonight's opposing starter hand, home − away",
        "definition": "How productive each lineup is specifically against the handedness it faces tonight.",
        "formula": "ops_vs_starter_hand(home lineup) − ops_vs_starter_hand(away lineup)",
        "source": "Statcast hitter splits vs LHP/RHP + confirmed starter hand",
        "window": "season to date",
        "units": "OPS",
        "direction": "higher = home advantage",
    },
    "travel_fatigue_diff": {
        "summary": "Home timezone crossings (last 3 days) − away (schedule fatigue)",
        "definition": "Travel wear: timezone crossings in the last 72 hours per club.",
        "formula": "time_zones_crossed_last_3d_home − time_zones_crossed_last_3d_away",
        "source": "Schedule geography: venue timezone changes",
        "window": "3d",
        "units": "crossings",
        "direction": "lower = home advantage (less travel)",
    },
    "closer_availability_diff": {
        "summary": "Home closer available − away closer available (late-inning edge)",
        "definition": "Whether each club's primary closer is rested and usable tonight.",
        "formula": "closer_available_home − closer_available_away",
        "source": "Recent reliever usage (2-day rest heuristic)",
        "window": "per-game",
        "units": "binary",
        "direction": "higher = home advantage",
    },
    "dome_is_neutral": {
        "summary": "1 if home park is a fixed dome/closed roof, 0 if open-air",
        "description_gate": True,  # type: ignore[dict-item]
        "definition": (
            "Weather hallucination gate: indoor parks get neutral weather "
            "values regardless of outside conditions."
        ),
        "formula": "1 if venue in DOMED_VENUES else 0",
        "source": "Static venue table",
        "window": "n/a (venue attribute)",
        "units": "binary",
        "direction": "n/a (gate flag)",
    },
    # ---- weather interactions ---------------------------------------------
    "park_factor_slug_diff": {
        "summary": "Home park SLG factor × lineup top-3 wOBA diff (hitter-friendly parks amplify lineup edges)",
        "definition": "Interaction: does tonight's park amplify whichever lineup holds the star-power edge?",
        "formula": "(home_park_slg_factor − 1) × lineup_woba_top3_diff",
        "source": "DuckDB feature engineering: park factors × Statcast top-3 wOBA",
        "window": "season (park) × season (lineup)",
        "units": "index",
        "direction": "higher = home advantage",
    },
    "wind_advantage_flyball_factor": {
        "summary": "Wind direction multiplier × SP ERA diff (flyball risk in windy conditions)",
        "definition": (
            "Wind blowing out multiplies the cost of a flyball-prone, weaker "
            "SP gap; measured from observed first-pitch weather (dome → exact 0)."
        ),
        "formula": "wind_direction_multiplier × sp_era_diff",
        "source": "Open-Meteo archive / StatsAPI game-feed weather × stadium bearing",
        "window": "per-game (observed)",
        "units": "index",
        "direction": "higher = more flyball risk against the home SP",
    },
    "air_density_velocity_boost": {
        "summary": "Stadium air density × SP velo diff (cold/thin air affects velocity)",
        "definition": "Thin/cold air (Coors, cold nights) changes how velocity carries; interaction with velo gap.",
        "formula": "air_density_factor × sp_fbvelo_diff",
        "source": "Open-Meteo archive (temp/RH/pressure → density) × Statcast velo",
        "window": "per-game (observed)",
        "units": "index",
        "direction": "higher = home advantage (velocity edge amplified)",
    },
    # ---- engineered interactions -------------------------------------------
    "bullpen_meltdown_risk": {
        "summary": "Bullpen pitches diff × WHIP diff (overworked + low quality = meltdown)",
        "definition": "Flags games where a tired pen is also performing poorly — late-inning blowup potential.",
        "formula": "bullpen_pitches_diff × bullpen_whip_3g_diff",
        "source": "DuckDB feature engineering: workload × form interaction",
        "window": "3d × 3g",
        "units": "index",
        "direction": "higher = home-side meltdown risk (negative for home)",
    },
    "pitcher_regression_indicator": {
        "summary": "SP velo diff × ERA diff (physical drop vs surface results = regression)",
        "definition": "Detects starters whose results outrun their stuff (or vice versa) — regression candidates.",
        "formula": "sp_fbvelo_diff × sp_era_diff",
        "source": "DuckDB feature engineering: velo × results interaction",
        "window": "season × 3g",
        "units": "index",
        "direction": "n/a (regression signal)",
    },
    "lineup_depth_multiplier": {
        "summary": "Lineup mean wOBA diff × top-3 wOBA diff (star power × depth)",
        "definition": "Rewards lineups that are BOTH deep AND star-heavy; punishes one-dimensional construction.",
        "formula": "lineup_woba_mean_diff × lineup_woba_top3_diff",
        "source": "DuckDB feature engineering: depth × star-power interaction",
        "window": "season to date (shrunk)",
        "units": "index",
        "direction": "higher = home advantage",
    },
    "ace_efficiency_factor": {
        "summary": "SP K/9 diff × whiff rate diff (high strikeout volume from raw stuff)",
        "definition": "Confirms strikeout gaps are backed by genuine swing-and-miss stuff, not luck.",
        "formula": "sp_k9_diff × sp_whiff_diff",
        "source": "DuckDB feature engineering: volume × stuff interaction",
        "window": "season × 3g",
        "units": "index",
        "direction": "higher = home advantage",
    },
    # ---- raw per-side columns (trees+MLP only; logistic is diffs-only) -----
    "home_elo": {
        "summary": "Home team Elo rating (level)",
        "definition": "Absolute strength level of the home club (the diff version carries the matchup signal).",
        "formula": "home_elo",
        "source": "DuckDB feature engineering: Elo engine",
        "window": "all prior games (decaying)",
        "units": "Elo points",
        "direction": "n/a (level; diff is the matchup term)",
    },
    "away_elo": {
        "summary": "Away team Elo rating (level)",
        "definition": "Absolute strength level of the away club.",
        "formula": "away_elo",
        "source": "DuckDB feature engineering: Elo engine",
        "window": "all prior games (decaying)",
        "units": "Elo points",
        "direction": "n/a (level)",
    },
    "home_win_pct": {
        "summary": "Home team season win% (level)",
        "definition": "Raw season record quality of the home club.",
        "formula": "smoothed_win_pct(home_wins, home_losses)",
        "source": "Official results records",
        "window": "season to date",
        "units": "win% (0–1)",
        "direction": "n/a (level)",
    },
    "away_win_pct": {
        "summary": "Away team season win% (level)",
        "definition": "Raw season record quality of the away club.",
        "formula": "smoothed_win_pct(away_wins, away_losses)",
        "source": "Official results records",
        "window": "season to date",
        "units": "win% (0–1)",
        "direction": "n/a (level)",
    },
    # ---- run-engine margin (SHIPPED 2026-08-26, see margin_ablation_*.json)
    "run_margin_diff": {
        "summary": "Run engine's expected-run margin: λ_home − λ_away (Poisson per-side)",
        "definition": (
            "One column: the run engine's per-side LightGBM Poisson models' "
            "expected runs for the home club minus the away club — a MODEL "
            "OUTPUT from a different feature view (levels + environment, no "
            "diffs), so it is genuinely decorrelated from the existing "
            "matchup columns rather than a linear combo of them. Computed "
            "OUT-OF-FOLD on the moneyline's own walk-forward split: each "
            "game's margin comes from a run-engine model trained strictly "
            "before it (fold-boundary asserted), so the moneyline never "
            "trains on a margin from a model that saw that game. Slate/"
            "inference margins use a fit-only refit on all decided games at "
            "the median fold round count. The run engine itself can never "
            "consume the column: derive_run_features drops every *_diff "
            "except park_factor_slug_diff."
        ),
        "formula": "λ_home − λ_away (run engine per-side Poisson, OOF on the moneyline folds)",
        "source": "Run engine (run_engine.py) per-side LightGBM Poisson, reused READ-ONLY by build_oof_margin.py",
        "window": "per-game (model output, strictly-prior training window)",
        "units": "expected runs",
        "direction": "positive = run engine expects the home club to outscore",
    },
}

_PER_SIDE_FAMILIES = {
    "sp_era": ("Starting-pitcher earned-run average", "ERA runs", "lower = better for that side"),
    "sp_k9": ("Starting-pitcher strikeouts per 9 innings", "K/9", "higher = better"),
    "sp_xwoba": ("Expected wOBA allowed by the starter", "xwOBA", "lower = better"),
    "lineup_woba_mean": ("Projected lineup average wOBA", "wOBA points", "higher = better"),
    "lineup_woba_top3": ("Top-3 hitters' projected wOBA", "wOBA points", "higher = better"),
    "woba_30g": ("Team offensive wOBA", "wOBA points", "higher = better"),
    "bullpen_whip_10g": ("Bullpen walks+hits per inning", "WHIP", "lower = better"),
    "bullpen_whip_3g": ("Bullpen walks+hits per inning, short form", "WHIP", "lower = better"),
    "team_barrel_15g": ("Team barreled-ball rate", "rate (0–1)", "higher = better"),
    "team_exitvelo_15g": ("Team average exit velocity", "mph", "higher = better"),
}

# Momentum form-delta families (recent window − season-to-date baseline, per
# side). Tuple: (label, units, direction, window). Direction is from the
# DELTA's perspective: positive = recent better than the season baseline
# (for cost stats like ERA/WHIP a positive delta means worse).
_FORM_DELTA_FAMILIES = {
    "sp_era_delta": ("SP ERA momentum (last 5 starts − season)", "ERA runs", "negative = hot streak (recent better)", "5g − season"),
    "sp_k9_delta": ("SP K/9 momentum (last 5 starts − season)", "K/9", "positive = strikeout surge", "5g − season"),
    "sp_bb9_delta": ("SP BB/9 momentum (30g − season)", "BB/9", "negative = control improvement", "30g − season"),
    "sp_whip_delta": ("SP WHIP momentum (30g − season)", "WHIP", "negative = form improvement", "30g − season"),
    "sp_xwoba_delta": ("SP xwOBA-allowed momentum (30g − season)", "xwOBA", "negative = recent better", "30g − season"),
    "sp_fbvelo_delta": ("SP fastball velo momentum (3g − season)", "mph", "positive = velo up (stuff gains)", "3g − season"),
    "sp_fbpct_delta": ("SP fastball-usage momentum (3g − season)", "share (0–1)", "n/a (mix signal)", "3g − season"),
    "sp_whiff_delta": ("SP whiff-rate momentum (3g − season)", "rate (0–1)", "positive = swing-and-miss surge", "3g − season"),
    "woba_delta": ("Team wOBA momentum (30g − season)", "wOBA points", "positive = lineup heating up", "30g − season"),
    "team_iso_delta": ("Team ISO momentum (30g − season)", "ISO points", "positive = power surge", "30g − season"),
    "team_k_rate_delta": ("Team strikeout-rate momentum (30g − season)", "rate (0–1)", "negative = K% down (better contact)", "30g − season"),
    "team_bb_rate_delta": ("Team walk-rate momentum (30g − season)", "rate (0–1)", "positive = more patience", "30g − season"),
    "team_barrel_delta": ("Team barrel-rate momentum (15g − season)", "rate (0–1)", "positive = quality-of-contact surge", "15g − season"),
    "team_hardhit_delta": ("Team hard-hit-rate momentum (15g − season)", "rate (0–1)", "positive = contact quality up", "15g − season"),
    "team_exitvelo_delta": ("Team exit-velocity momentum (15g − season)", "mph", "positive = velo up", "15g − season"),
    "bullpen_whip_delta": ("Bullpen WHIP momentum (10g − season)", "WHIP", "negative = pen tightening up", "10g − season"),
    "bullpen_era_delta": ("Bullpen ERA momentum (10g − season)", "ERA runs", "negative = recent better", "10g − season"),
    "lineup_woba_mean_delta": ("Lineup wOBA momentum (today's lineup − season lineup)", "wOBA points", "positive = current lineup stronger than season average", "per-game lineup − season"),
    "lineup_woba_top3_delta": ("Top-3 wOBA momentum (today's top-3 − season top-3)", "wOBA points", "positive = star power up today", "per-game lineup − season"),
}


# Phase 2 lineup-delta families (actual starting-9 wOBA vs team season, per
# side). Tuple: (label, units, direction).
_LINEUP_DELTA_FAMILIES = {
    "lineup_actual_woba_delta": ("Lineup wOBA delta (actual 9 − team season)", "wOBA points", "positive = tonight's 9 better than the season-average lineup"),
    "lineup_actual_top3_delta": ("Lineup top-3 wOBA delta (actual 9 top-3 − team top-3 regulars)", "wOBA points", "positive = star power up tonight"),
    "lineup_rest_count": ("Resting regulars (team top-5 wOBA not in tonight's 9)", "count (0–5)", "higher = more stars resting"),
}


# Categorical-context columns (TREE_CATEGORICAL_COLS inputs, NOT in
# FEATURE_COLS — so they live in a dedicated payload section rather than the
# FEATURE_COLS walk, which would trip the stale-entry warning). Emitted as
# payload["categorical_context"] with the same tooltip schema; additive for
# frontend consumers that only read payload["features"].
_CATEGORICAL_CONTEXT: dict[str, dict[str, str]] = {
    "venue": {
        "summary": "Home ballpark — native categorical for LGB/XGB tree members",
        "definition": (
            "The game's venue name, encoded as a stable integer category for "
            "the LightGBM/XGBoost members (native categorical, never one-hot). "
            "Park context (dome, altitude, dimensions) is a real input the "
            "numeric level features only approximate; the literal 'Unknown' "
            "value and predict-time newcomers map to a dedicated UNK category."
        ),
        "formula": "venue name → venue_id (stable int category, UNK-safe)",
        "source": "game_level_features.csv venue column; slate rows carry the scheduled venue",
        "window": "n/a (game context)",
        "units": "categorical",
        "direction": "context only — trees learn per-park splits",
    },
    "home_starter_id": {
        "summary": "Home starting pitcher (MLB player ID) — native categorical for LGB/XGB",
        "definition": (
            "The home starter's MLB StatsAPI player ID, remapped to a compact "
            "integer category for the LightGBM/XGBoost members (native "
            "categorical, never one-hot). Pitcher identity carries skill level "
            "the numeric diffs (ERA/K9/xwOBA) only approximate, especially for "
            "elite or struggling arms. Starters never seen in training "
            "(callups, trades, spot starts) map to a dedicated UNK category."
        ),
        "formula": "home_starter_id → home_starter_cat_id (dense int category, UNK-safe)",
        "source": "Probable-pitcher data (StatsAPI), announced well before first pitch; game_level_features.csv home_starter_id",
        "window": "per game",
        "units": "categorical",
        "direction": "context only — trees learn per-pitcher splits",
    },
    "away_starter_id": {
        "summary": "Away starting pitcher (MLB player ID) — native categorical for LGB/XGB",
        "definition": (
            "The away starter's MLB StatsAPI player ID, remapped to a compact "
            "integer category for the LightGBM/XGBoost members (native "
            "categorical, never one-hot). Mirror of home_starter_id; same "
            "shared player→category map, so the same pitcher is the same "
            "category on either side. Unseen starters map to a dedicated UNK "
            "category."
        ),
        "formula": "away_starter_id → away_starter_cat_id (dense int category, UNK-safe)",
        "source": "Probable-pitcher data (StatsAPI), announced well before first pitch; game_level_features.csv away_starter_id",
        "window": "per game",
        "units": "categorical",
        "direction": "context only — trees learn per-pitcher splits",
    },
}


def _rich_entry(name: str) -> Optional[dict[str, str]]:
    """Authored entry, or a synthesized one for *_home/*_away family members."""
    if name in _RICH:
        entry = {k: v for k, v in _RICH[name].items() if k != "description_gate"}
        return entry
    for suffix in ("_home", "_away"):
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            side = "home" if suffix == "_home" else "away"
            fam = _LINEUP_DELTA_FAMILIES.get(base)
            if fam:
                label, units, direction = fam
                return {
                    "summary": f"{label} — {side} team",
                    "definition": (
                        f"Lineup-delta column: the {side} club's ACTUAL starting "
                        f"nine's season-to-date wOBA minus the team's own "
                        f"season-to-date wOBA as of game day, so resting-star "
                        f"days are visible to the model (the level columns only "
                        f"see season-average lineup quality). Point-in-time: "
                        f"batter/team wOBA through games strictly before the "
                        f"game date — no lookahead. Batters below the min-PA "
                        f"floor use the team season mean."
                    ),
                    "formula": name,
                    "source": "StatsAPI battingOrder (lineups.parquet) + Statcast pbp point-in-time wOBA",
                    "window": "per-game lineup − season to date",
                    "units": units,
                    "direction": f"{direction} ({side} side)",
                }
            fam = _FORM_DELTA_FAMILIES.get(base)
            if fam:
                label, units, direction, window = fam
                return {
                    "summary": f"{label} — {side} team",
                    "definition": (
                        f"Momentum form-delta column: the {side} club's recent "
                        f"window minus its season-to-date baseline, so the "
                        f"model sees hot streaks/slumps directly instead of "
                        f"only the levels. Continuous (no binary flags); "
                        f"trees learn their own thresholds. Computed from the "
                        f"same shifted per-game stats as the level twins."
                    ),
                    "formula": f"{base}_recent_{side} − {base}_season_{side}",
                    "source": "Statcast aggregates via DuckDB feature engineering",
                    "window": window,
                    "units": units,
                    "direction": f"{direction} ({side} side)",
                }
            fam = _PER_SIDE_FAMILIES.get(base)
            if not fam:
                return None
            label, units, direction = fam
            return {
                "summary": f"{label} — {side} team",
                "definition": (
                    f"Per-side level column: the {side} club's {label.lower()}. "
                    f"The corresponding '{base}_diff' carries the matchup signal; "
                    f"this level column lets tree models learn nonlinear context."
                ),
                "formula": name,
                "source": "Statcast aggregates via DuckDB feature engineering",
                "window": _family_window(base),
                "units": units,
                "direction": f"{direction} for the {side} side (level column)",
            }
    return None


def _family_window(base: str) -> str:
    if "15g" in base:
        return "15g"
    if "3g" in base:
        return "3g"
    if "30g" in base:
        return "30g"
    return "season to date"


def members_for_feature(name: str, logistic_cols: set[str]) -> list[str]:
    """Routing DERIVED from live config: trees + MLP see everything; logistic
    sees its configured slice (diffs-only when LOGISTIC_USE_RAW_COLS=False)."""
    members = list(TREE_MEMBERS)
    if name in logistic_cols:
        members.append("logistic")
    return members


def build_features_metadata() -> tuple[dict[str, dict], list[str]]:
    """Build the metadata dict keyed by FEATURE_COLS names.

    Returns (metadata, warnings_list). Every FEATURE_COLS entry gets a row;
    unauthored features get a clearly-marked placeholder AND a warning string
    so absence is never silent."""
    from training import FEATURE_COLS, _logistic_feature_cols

    logistic_cols = set(_logistic_feature_cols())
    meta: dict[str, dict] = {}
    warnings: list[str] = []
    for name in FEATURE_COLS:
        entry = _rich_entry(name)
        members = members_for_feature(name, logistic_cols)
        if entry is None:
            msg = (
                f"Feature metadata: no authored entry for '{name}' — shipping "
                f"a PLACEHOLDER (fill in feature_metadata._RICH); tooltip will "
                f"say 'no detailed metadata'"
            )
            logger.warning(msg)
            warnings.append(msg)
            entry = {
                "summary": name,
                "definition": "No detailed metadata authored yet.",
                "formula": "—",
                "source": "—",
                "window": "—",
                "units": "—",
                "direction": "—",
            }
        row = {"name": name, **entry, "members": members}
        row["tooltip"] = format_tooltip(row)
        meta[name] = row
    # Authored-but-no-longer-in-FEATURE_COLS entries would silently rot — warn.
    stale = sorted(set(_RICH) - set(FEATURE_COLS))
    if stale:
        msg = f"Feature metadata: entries no longer in FEATURE_COLS: {stale}"
        logger.warning(msg)
        warnings.append(msg)
    return meta, warnings


def format_tooltip(meta: dict[str, Any]) -> str:
    """Plain-text tooltip body (rendered into an HTML title attribute by the
    frontend). Pure function so tests can exercise it without Streamlit."""
    def fmt_members(members: Any) -> str:
        try:
            return ", ".join(str(m) for m in members)
        except TypeError:
            return str(members)

    return (
        f"What: {meta.get('definition', '—')}\n"
        f"Formula: {meta.get('formula', '—')}\n"
        f"Source: {meta.get('source', '—')}\n"
        f"Window: {meta.get('window', '—')} · Units: {meta.get('units', '—')}\n"
        f"Direction: {meta.get('direction', '—')}\n"
        f"Consumed by: {fmt_members(meta.get('members'))}"
    )


FALLBACK_TOOLTIP_SUFFIX = "\n(no detailed metadata)"


def generate_features_metadata(target_date_str: str,
                               out_dir: Optional[Path] = None) -> dict:
    """Generate and persist the artifact; returns the embedded-ready dict."""
    meta, warns = build_features_metadata()
    ctx: dict[str, dict] = {}
    for name, entry in _CATEGORICAL_CONTEXT.items():
        row = {"name": name, **entry, "members": ["xgboost", "lightgbm"]}
        row["tooltip"] = format_tooltip(row)
        ctx[name] = row
    payload = {
        "generated_for": target_date_str,
        "n_features": len(meta),
        "warnings": warns,
        "features": meta,
        "categorical_context": ctx,
    }
    out_path = (out_dir or DATA_DELIVERY_DIR) / f"features_metadata_{target_date_str}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(out_path)  # atomic
    logger.info("Feature metadata: %d features written -> %s", len(meta), out_path.name)
    return payload
