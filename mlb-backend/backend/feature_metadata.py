"""Canonical feature metadata → features_metadata_<date>.json artifact.

Single source of truth for dashboard tooltips (Feature Drift Analysis) and any
consumer that needs to explain what a feature IS. Entries are keyed by exact
MONEYLINE_FEATURE_COLS names; generation walks MONEYLINE_FEATURE_COLS itself so a newly added
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

TREE_MEMBERS = ["xgboost", "lightgbm"]
_ALL_MEMBERS = TREE_MEMBERS + ["elasticnet"]

# ---------------------------------------------------------------------------
# Rich authored entries. Keys must exactly match MONEYLINE_FEATURE_COLS names; anything
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
    # sp_era_diff RETIRED 2026-10-03 (pl_[pos] + removals plan — plain SP
    # ERA trio left serving; sp_era_5g_diff stays). Column still generated;
    # candidate pool carries it. Restore its authored entry if re-promoted.
    "sp_era_5g_diff": {
        "summary": "Home SP recent runs allowed per nine − away SP (shrunk recent form)",
        "definition": (
            "Recent rate uses up to five prior pitcher appearances, weighted by "
            "innings and blended toward the same pitcher's older, non-overlapping "
            "career rate with 30 pseudo innings. A strictly prior league rate is "
            "the cold-start fallback. This is runs allowed per nine, not official "
            "earned-run ERA."
        ),
        "formula": "shrunk_recent_runs_per_9_home − shrunk_recent_runs_per_9_away",
        "source": "Statcast pitcher appearance aggregates (LAG-shifted)",
        "window": "5 prior appearances + older pitcher history (30 pseudo-IP)",
        "units": "runs / 9 innings",
        "direction": "lower = home advantage",
    },
    # RETIRED 2026-09-07 (Experiment #2 E/F replacement — the 6 baseline
    # S-family features left MONEYLINE_FEATURE_COLS): sp_k9_5g_diff,
    # sp_fbpct_diff, sp_whiff_diff, sp_xwoba_diff, sp_xwoba_vs_l_diff.
    # (sp_k9_diff READMITTED 2026-09-30 — authored entry restored below —
    # see training._EXP2_REMOVALS for the revert rationale.) Their authored
    # dashboard entries were removed (the dashboard only renders
    # MONEYLINE_FEATURE_COLS members); the columns remain generated in
    # the dataset but are no longer part of the active run-engine contract.
    "sp_k9_diff": {
        "summary": "Home SP season-to-date K/9 − away SP",
        "definition": (
            "Starting-pitcher strikeout-volume gap. READMITTED 2026-09-30: "
            "raw twins sp_k9_home/away re-entered serving 2026-09-27, so the "
            "diff was restored for structural alignment."
        ),
        "formula": "sp_k9_home − sp_k9_away",
        "source": "Statcast pitching aggregates (season to date)",
        "window": "season to date",
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
        "units": "mph",        "direction": "higher = home advantage",
    },


    # ---- SP xwOBA allowed -------------------------------------------------
    # ---- position-pool xwOBA diffs (pl_[pos] + removals plan, 2026-10-03)
    # The lineup re24 family, lineup_depth_multiplier_diff, sp_era_diff and
    # park_factor_slug_diff left MONEYLINE_FEATURE_COLS for this family
    # (plan: add 24 = 8 pools × home/away/diff, replace 9, remove 7 →
    # width 109; no DH). Their authored entries were removed with them —
    # the dashboard only renders serving members — and the columns remain
    # generated (features.py) under config.RFE_CANDIDATE_COLS, so RFE may
    # re-trial them; restore their entries here if one is re-promoted (same
    # pattern as the 2026-09-07 S-family retirement below).
    "pl_c_xwoba_diff": {
        "summary": "Home lineup's catcher-pool xwOBA − away lineup's",
        "definition": (
            "Position-pool batting quality for the C seat: aggregate "
            "Statcast xwOBA of the projected lineup's players at the "
            "position, shrunk toward the position-segmented league prior."
        ),
        "formula": "pl_c_xwoba_home − pl_c_xwoba_away",
        "source": "StatsAPI roster positions × Statcast estimated_woba_using_speedangle (LAG-shifted position pools)",
        "window": "season to date (shrunk, position-segmented prior)",
        "units": "xwOBA (0–1)",
        "direction": "higher = home lineup deeper at catcher",
    },
    "pl_fb_xwoba_diff": {
        "summary": "Home lineup's first-base-pool xwOBA − away lineup's",
        "definition": "Position-pool batting quality for the 1B seat (StatsAPI 1B → fb), shrunk toward the position-segmented league prior.",
        "formula": "pl_fb_xwoba_home − pl_fb_xwoba_away",
        "source": "StatsAPI roster positions × Statcast estimated_woba_using_speedangle (LAG-shifted position pools)",
        "window": "season to date (shrunk, position-segmented prior)",
        "units": "xwOBA (0–1)",
        "direction": "higher = home lineup deeper at first base",
    },
    "pl_sb_xwoba_diff": {
        "summary": "Home lineup's second-base-pool xwOBA − away lineup's",
        "definition": "Position-pool batting quality for the 2B seat (StatsAPI 2B → sb), shrunk toward the position-segmented league prior.",
        "formula": "pl_sb_xwoba_home − pl_sb_xwoba_away",
        "source": "StatsAPI roster positions × Statcast estimated_woba_using_speedangle (LAG-shifted position pools)",
        "window": "season to date (shrunk, position-segmented prior)",
        "units": "xwOBA (0–1)",
        "direction": "higher = home lineup deeper at second base",
    },
    "pl_ss_xwoba_diff": {
        "summary": "Home lineup's shortstop-pool xwOBA − away lineup's",
        "definition": "Position-pool batting quality for the SS seat, shrunk toward the position-segmented league prior.",
        "formula": "pl_ss_xwoba_home − pl_ss_xwoba_away",
        "source": "StatsAPI roster positions × Statcast estimated_woba_using_speedangle (LAG-shifted position pools)",
        "window": "season to date (shrunk, position-segmented prior)",
        "units": "xwOBA (0–1)",
        "direction": "higher = home lineup deeper at shortstop",
    },
    "pl_tb_xwoba_diff": {
        "summary": "Home lineup's third-base-pool xwOBA − away lineup's",
        "definition": "Position-pool batting quality for the 3B seat (StatsAPI 3B → tb), shrunk toward the position-segmented league prior.",
        "formula": "pl_tb_xwoba_home − pl_tb_xwoba_away",
        "source": "StatsAPI roster positions × Statcast estimated_woba_using_speedangle (LAG-shifted position pools)",
        "window": "season to date (shrunk, position-segmented prior)",
        "units": "xwOBA (0–1)",
        "direction": "higher = home lineup deeper at third base",
    },
    "pl_rf_xwoba_diff": {
        "summary": "Home lineup's right-field-pool xwOBA − away lineup's",
        "definition": "Position-pool batting quality for the RF seat, shrunk toward the position-segmented league prior.",
        "formula": "pl_rf_xwoba_home − pl_rf_xwoba_away",
        "source": "StatsAPI roster positions × Statcast estimated_woba_using_speedangle (LAG-shifted position pools)",
        "window": "season to date (shrunk, position-segmented prior)",
        "units": "xwOBA (0–1)",
        "direction": "higher = home lineup deeper in right field",
    },
    "pl_cf_xwoba_diff": {
        "summary": "Home lineup's center-field-pool xwOBA − away lineup's",
        "definition": "Position-pool batting quality for the CF seat, shrunk toward the position-segmented league prior.",
        "formula": "pl_cf_xwoba_home − pl_cf_xwoba_away",
        "source": "StatsAPI roster positions × Statcast estimated_woba_using_speedangle (LAG-shifted position pools)",
        "window": "season to date (shrunk, position-segmented prior)",
        "units": "xwOBA (0–1)",
        "direction": "higher = home lineup deeper in center field",
    },
    "pl_lf_xwoba_diff": {
        "summary": "Home lineup's left-field-pool xwOBA − away lineup's",
        "definition": "Position-pool batting quality for the LF seat, shrunk toward the position-segmented league prior.",
        "formula": "pl_lf_xwoba_home − pl_lf_xwoba_away",
        "source": "StatsAPI roster positions × Statcast estimated_woba_using_speedangle (LAG-shifted position pools)",
        "window": "season to date (shrunk, position-segmented prior)",
        "units": "xwOBA (0–1)",
        "direction": "higher = home lineup deeper in left field",
    },

    # pl_dh_xwoba_diff is generated in the frame but NOT served (the plan
    # sheet lists 8 pools; pl_dh rides TWP→DH only at the frame level).
    "lineup_il_flag_home": {
        "summary": "Home projected nine includes an OUT/IR player (raw flag)",
        "definition": "1 when at least one of the home game-eligible top-9-by-PA candidates is on the injured list as of the game date, else 0. Sides with no eligible pool read 0. The injured player's own trailing RE24 rating is untouched — the flag marks eligibility, not quality.",
        "formula": "MAX(on_il) over the projected nine",
        "source": "MLB StatsAPI transactions via build_il_stints.py (transaction dates, PA-reconciled)",
        "window": "as of game date (point-in-time)",
        "units": "binary",
        "direction": "1 = home availability risk",
    },
    "lineup_il_flag_away": {
        "summary": "Away projected nine includes an OUT/IR player (raw flag)",
        "definition": "Away-side companion to lineup_il_flag_home: 1 when at least one of the away game-eligible top-9-by-PA candidates is on the injured list as of the game date.",
        "formula": "MAX(on_il) over the projected nine",
        "source": "MLB StatsAPI transactions via build_il_stints.py",
        "window": "as of game date (point-in-time)",
        "units": "binary",
        "direction": "1 = away availability risk",
    },
    "lineup_il_flag_diff": {
        "summary": "Home OUT/IR flag − away OUT/IR flag",
        "definition": "Availability-risk gap: positive when the HOME projected nine carries an injured player the away nine does not. Distinguishes which side is depleted — the wOBA aggregates move for many reasons, this moves only on an OUT/IR designation.",
        "formula": "lineup_il_flag_home − lineup_il_flag_away",
        "source": "MLB StatsAPI transactions via build_il_stints.py",
        "window": "as of game date (point-in-time)",
        "units": "binary (-1..1)",
        "direction": "higher = home availability risk",
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
    # 2026-10-09: summaries spell the metric out as "walks+hits per
    # inning" (the same wording the per-side level rows use) so the drift
    # table never advertises the WHIP brand name; the units field keeps
    # the technical unit.
    "bullpen_whip_10g_diff": {
        "summary": "Home bullpen 10-game walks+hits per inning − away bullpen (lower = better)",
        "definition": "Relief corps baserunner allowance over the last 10 games.",
        "formula": "bullpen_whip_10g_home − bullpen_whip_10g_away",
        "source": "Statcast relief-pitching aggregates",
        "window": "10g",
        "units": "WHIP",
        "direction": "lower = home advantage",
    },
    "bullpen_whip_3g_diff": {
        "summary": "Home bullpen 3-game walks+hits per inning − away bullpen (short-term form)",
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
        "definition": "Observed Statcast launch_speed_angle = 6 rate gap; missing classifications stay NULL, never a narrow exit-velocity/angle proxy.",
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
    "closer_available_home": {
        "summary": "Home closer rested and available tonight (raw flag)",
        "definition": (
            "1 if the home club's primary closer is rested and usable tonight, "
            "0 otherwise. The raw side-level companion to closer_availability_diff: "
            "the diff alone maps both-closers-available and both-closers-out to the "
            "same 0, so this flag restores the distinction for each club."
        ),
        "formula": "1 if home closer available else 0",
        "source": "Recent reliever usage (2-day rest heuristic)",
        "window": "per-game",
        "units": "binary",
        "direction": "1 = home bullpen at full strength",
    },
    "closer_available_away": {
        "summary": "Away closer rested and available tonight (raw flag)",
        "definition": (
            "1 if the away club's primary closer is rested and usable tonight, "
            "0 otherwise. Raw side-level companion to closer_availability_diff; "
            "see closer_available_home."
        ),
        "formula": "1 if away closer available else 0",
        "source": "Recent reliever usage (2-day rest heuristic)",
        "window": "per-game",
        "units": "binary",
        "direction": "1 = away bullpen at full strength",
    },
    "dome_is_neutral": {
        "summary": "1 if home park is a fixed dome/closed roof, 0 if open-air",
        "description_gate": True,  # type: ignore[dict-item]
        "definition": (
            "Weather hallucination gate: indoor games get neutral weather "
            "values regardless of outside conditions."
        ),
        "formula": (
            "1 if this game's roof is closed (or the park is a fixed dome), "
            "0 if open-air — resolved per game from the StatsAPI roof state "
            "for retractable parks, venue type otherwise"
        ),
        "source": "Per-game roof state (StatsAPI) over the static venue table",
        "window": "n/a (per-game venue attribute)",
        "units": "binary",
        "direction": "n/a (gate flag)",
    },
    # ---- weather interactions ---------------------------------------------
    # park_factor_slug_diff RETIRED 2026-10-03 (pl_[pos] + removals plan —
    # the 7th removal; built on lineup_re24_top3_diff, which left with the
    # re24 family).
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
    "bullpen_meltdown_risk_diff": {
        "summary": "Bullpen pitches 3d diff × WHIP 10g diff (overworked + low quality = meltdown)",
        "definition": "Flags games where a pen fatigued over the prior 3 calendar days is also performing poorly over its last 10 games — late-inning blowup potential.",
        "formula": "bullpen_pitches_diff × bullpen_whip_10g_diff",
        "source": "DuckDB feature engineering: workload × form interaction",
        "window": "3d × 10g",
        "units": "index",
        "direction": "higher = home-side meltdown risk (negative for home)",
    },
    "bullpen_meltdown_risk_home": {
        "summary": "Home bullpen meltdown risk (3-day pitch count × 10-game WHIP)",
        "definition": "Within-side fatigue × quality product for the home pen.",
        "formula": "bullpen_pitches_3d_home × bullpen_whip_10g_home",
        "source": "DuckDB feature engineering: workload × form interaction",
        "window": "3d × 10g",
        "units": "index",
        "direction": "higher = home pen more melt-prone",
    },
    "bullpen_meltdown_risk_away": {
        "summary": "Away bullpen meltdown risk (3-day pitch count × 10-game WHIP)",
        "definition": "Within-side fatigue × quality product for the away pen.",
        "formula": "bullpen_pitches_3d_away × bullpen_whip_10g_away",
        "source": "DuckDB feature engineering: workload × form interaction",
        "window": "3d × 10g",
        "units": "index",
        "direction": "higher = away pen more melt-prone (home advantage)",
    },
    "bullpen_budget_2d_home": {
        "summary": "Home bullpen pitches over the prior 2 calendar days",
        "definition": "Total relief-pitcher workload (pitches, starters excluded) entering tonight's game — the acute-fatigue budget the manager draws from.",
        "formula": "SUM(bp_outing.n_pitches) for the home pen, day ∈ (game_date − 2, game_date)",
        "source": "Statcast reliever outings (per-arm ledger, starters excluded)",
        "window": "2d",
        "units": "pitches",
        "direction": "higher = more spent pen (away advantage)",
    },
    "bullpen_budget_2d_away": {
        "summary": "Away bullpen pitches over the prior 2 calendar days",
        "definition": "Total relief-pitcher workload (pitches, starters excluded) entering tonight's game for the away pen.",
        "formula": "SUM(bp_outing.n_pitches) for the away pen, day ∈ (game_date − 2, game_date)",
        "source": "Statcast reliever outings (per-arm ledger, starters excluded)",
        "window": "2d",
        "units": "pitches",
        "direction": "higher = away pen more spent (home advantage)",
    },
    "bp_ready_share_home": {
        "summary": "Home pen readiness: pitch-weighted next-day availability of arms used in the last 2 days (0-1)",
        "definition": "Each recent outing is scored by the measured next-day availability staircase (last outing <20 pitches = 0.25, 20-34 = 0.13, 35+ = 0.007, on-ledger = 0; calibrated on 51,487 outings 2024-26), pitch-weighted, then normalized by the fresh ceiling: 1.0 = everyone threw <20 pitches, ~0.05 = everyone heavy, 0 = all unavailable.",
        "formula": "SUM(n_pitches × ready_p) / SUM(n_pitches) / 0.25 over the home pen's last-2-day outings",
        "source": "Statcast reliever outings + pitcher availability ledger",
        "window": "2d",
        "units": "normalized readiness [0, 1]",
        "direction": "higher = fresher home pen (home advantage)",
    },
    "bp_ready_share_away": {
        "summary": "Away pen readiness: pitch-weighted next-day availability of arms used in the last 2 days (0-1)",
        "definition": "Same readiness staircase for the away pen; higher = fresher away arms.",
        "formula": "SUM(n_pitches × ready_p) / SUM(n_pitches) / 0.25 over the away pen's last-2-day outings",
        "source": "Statcast reliever outings + pitcher availability ledger",
        "window": "2d",
        "units": "normalized readiness [0, 1]",
        "direction": "higher = fresher away pen (away advantage)",
    },
    "pitcher_regression_indicator_diff": {
        "summary": "SP velo diff × shrunk recent runs/9 diff (stuff vs results)",
        "definition": (
            "Detects starters whose recent runs allowed outrun their fastball "
            "velocity (or vice versa); the recent runs/9 inputs are shrunk "
            "toward older pitcher history."
        ),
        "formula": "sp_fbvelo_diff × sp_era_5g_diff",
        "source": "DuckDB feature engineering: velo × shrunk recent runs/9 interaction",
        "window": "season × 3g; recent runs/9 uses up to 5 prior appearances",
        "units": "index",
        "direction": "n/a (regression signal)",
    },
    # lineup_depth_multiplier_diff RETIRED 2026-10-03 (pl_[pos] + removals
    # plan — a lineup_re24_mean_diff × lineup_re24_top3_diff product; both
    # inputs left with the re24 family).
    "ace_efficiency_factor_diff": {
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
    # ---- Experiment #2 matchup candidates (SHIPPED 2026-09-07, C+E/D+F) ---
    "exp2_centered_k_diff": {
        "summary": "Centered strikeout matchup: (SP K/9 − league K%) × (opp K% − league K%), home − away",
        "definition": (
            "Interaction of both sides' strikeout tendency centered on the "
            "point-in-time league prior — positive when BOTH the home "
            "starter and the away offense are extreme (either direction) "
            "relative to league. Known denominator mix: SP K/9 vs opponent/"
            "league K/PA (frozen pre-test in the experiment registry)."
        ),
        "formula": "(sp_k9_home − league_k_pct)(team_k_rate_30g_home − league_k_pct) − (away mirror)",
        "source": "Experiment #2 source layer (Statcast season-to-date aggregates)",
        "window": "season to date (strictly prior games)",
        "units": "rate² product",
        "direction": "positive = home side's K matchup more extreme",
    },
    "exp2_cat_k_fastball_diff": {
        "summary": "Fastball K matchup: SP FB usage × (SP − league) × (opp − league) FB K%, home − away",
        "definition": (
            "Usage-weighted amplification of fastball strikeout extremes on "
            "both sides of the home matchup, minus the away mirror."
        ),
        "formula": "sp_usage_cat_fastball·(sp_k_pct_cat_fastball − lg)·(team_k_pct_cat_fastball − lg), home − away",
        "source": "Experiment #2 source layer (pitch-category K% aggregates)",
        "window": "season to date (strictly prior games)",
        "units": "rate³ product",
        "direction": "positive = home FB-K matchup more extreme",
    },
    "exp2_cat_k_breaking_diff": {
        "summary": "Breaking-ball K matchup: usage × (SP − lg) × (opp − lg) breaking K%, home − away",
        "definition": (
            "Usage-weighted breaking-ball strikeout matchup. Best single "
            "moneyline candidate in the frozen experiment (+0.0055 ΔAUC, "
            "CI excluding zero, survives S_FB_K removal)."
        ),
        "formula": "sp_usage_cat_breaking·(sp_k_pct_cat_breaking − lg)·(team_k_pct_cat_breaking − lg), home − away",
        "source": "Experiment #2 source layer (pitch-category K% aggregates)",
        "window": "season to date (strictly prior games)",
        "units": "rate³ product",
        "direction": "positive = home breaking-K matchup more extreme",
    },
    "exp2_cat_k_offspeed_diff": {
        "summary": "Offspeed K matchup: usage × (SP − lg) × (opp − lg) offspeed K%, home − away",
        "definition": "Usage-weighted offspeed strikeout matchup. Sparsest category (~56% coverage).",
        "formula": "sp_usage_cat_offspeed·(sp_k_pct_cat_offspeed − lg)·(team_k_pct_cat_offspeed − lg), home − away",
        "source": "Experiment #2 source layer (pitch-category K% aggregates)",
        "window": "season to date (strictly prior games)",
        "units": "rate³ product",
        "direction": "positive = home offspeed-K matchup more extreme",
    },
    "exp2_cat_xwoba_fastball_diff": {
        "summary": "Fastball xwOBA matchup: usage × (SP − lg) × (opp − lg) FB xwOBA, home − away",
        "definition": (
            "Usage-weighted fastball quality-of-contact matchup. Higher "
            "xwOBA = worse for the pitcher; the product amplifies when both "
            "sides sit off-league, same form as the K family (no sign flip)."
        ),
        "formula": "sp_usage_cat_fastball·(sp_xwoba_cat_fastball − lg)·(team_xwoba_cat_fastball − lg), home − away",
        "source": "Experiment #2 source layer (pitch-category xwOBA aggregates)",
        "window": "season to date (strictly prior games)",
        "units": "xwOBA² product",
        "direction": "positive = home FB quality matchup more extreme",
    },
    "exp2_cat_xwoba_breaking_diff": {
        "summary": "Breaking-ball xwOBA matchup: usage × (SP − lg) × (opp − lg) breaking xwOBA, home − away",
        "definition": "Usage-weighted breaking-ball quality-of-contact matchup (same amplification form).",
        "formula": "sp_usage_cat_breaking·(sp_xwoba_cat_breaking − lg)·(team_xwoba_cat_breaking − lg), home − away",
        "source": "Experiment #2 source layer (pitch-category xwOBA aggregates)",
        "window": "season to date (strictly prior games)",
        "units": "xwOBA² product",
        "direction": "positive = home breaking quality matchup more extreme",
    },
    "exp2_cat_xwoba_offspeed_diff": {
        "summary": "Offspeed xwOBA matchup: usage × (SP − lg) × (opp − lg) offspeed xwOBA, home − away",
        "definition": "Usage-weighted offspeed quality-of-contact matchup (sparsest category).",
        "formula": "sp_usage_cat_offspeed·(sp_xwoba_cat_offspeed − lg)·(team_xwoba_cat_offspeed − lg), home − away",
        "source": "Experiment #2 source layer (pitch-category xwOBA aggregates)",
        "window": "season to date (strictly prior games)",
        "units": "xwOBA² product",
        "direction": "positive = home offspeed quality matchup more extreme",
    },
    "exp2_cat_platoon_k_fastball_diff": {
        "summary": "Platoon fastball-K matchup: hand-mix-weighted (SP − lg) × (opp − lg) FB K × SP FB usage, home − away",
        "definition": (
            "Fastball strikeout matchup weighted by the opposing lineup's "
            "actual L/R share: each side's SP and offense K-vs-fastball "
            "splits are centered on the league FB-K prior per hand, "
            "platoon-weighted, multiplied, then scaled by SP fastball usage."
        ),
        "formula": "Σ_hand share_hand·(sp_fb_vs_hand − lg_hand)·Σ_hand share_hand·(opp_fb_vs_hand − lg_hand)·sp_fb_usage, home − away",
        "source": "Experiment #2 source layer (platoon fastball-K splits + opp_lefty_share)",
        "window": "season to date (strictly prior games)",
        "units": "rate³ product",
        "direction": "positive = home platoon-K matchup more extreme",
    },
}

_PER_SIDE_FAMILIES = {
    "sp_era": ("Starting-pitcher earned-run average", "ERA runs", "lower = better for that side", "season to date (prior in-season starts; LAG-shifted)"),
    "sp_k9": ("Starting-pitcher strikeouts per 9 innings", "K/9", "higher = better", "season to date (prior in-season starts; LAG-shifted)"),
    "sp_xwoba": ("Expected wOBA allowed by the starter", "xwOBA", "lower = better", "last 6 appearances (LAG-shifted; the legacy _30g name)"),
    "lineup_re24_mean": ("Projected lineup average RE24", "RE24 (runs per PA)", "higher = better"),
    "lineup_re24_top3": ("Top-3 hitters' projected RE24", "RE24 (runs per PA)", "higher = better"),
    "woba_30g": ("Team offensive wOBA", "wOBA points", "higher = better"),
    "bullpen_whip_10g": ("Bullpen walks+hits per inning", "WHIP", "lower = better", "10 team games (opportunity-shrunk, k = 20% of mean reliever-season pitches)"),
    "bullpen_whip_3g": ("Bullpen walks+hits per inning, short form", "WHIP", "lower = better", "3 team games (opportunity-shrunk)"),
    "team_barrel_15g": ("Team barreled-ball rate", "rate (0–1)", "higher = better"),
    "team_exitvelo_15g": ("Team average exit velocity", "mph", "higher = better"),
    # Position-pool xwOBA levels (pl_[pos] + removals plan, 2026-10-03):
    # aggregate Statcast xwOBA of the projected lineup's hitters at the
    # position (StatsAPI 1B→fb, 2B→sb, 3B→tb; TWP→DH), shrunk toward the
    # position-segmented league prior. Levels are tree-only; the _diff
    # siblings carry the matchup signal to the logistic slice.
    "pl_c_xwoba": ("Position-pool xwOBA — catcher seat", "xwOBA (0–1)", "higher = deeper lineup at the position", "season to date (shrunk toward position prior)"),
    "pl_fb_xwoba": ("Position-pool xwOBA — first-base seat", "xwOBA (0–1)", "higher = deeper lineup at the position", "season to date (shrunk toward position prior)"),
    "pl_sb_xwoba": ("Position-pool xwOBA — second-base seat", "xwOBA (0–1)", "higher = deeper lineup at the position", "season to date (shrunk toward position prior)"),
    "pl_ss_xwoba": ("Position-pool xwOBA — shortstop seat", "xwOBA (0–1)", "higher = deeper lineup at the position", "season to date (shrunk toward position prior)"),
    "pl_tb_xwoba": ("Position-pool xwOBA — third-base seat", "xwOBA (0–1)", "higher = deeper lineup at the position", "season to date (shrunk toward position prior)"),
    "pl_rf_xwoba": ("Position-pool xwOBA — right-field seat", "xwOBA (0–1)", "higher = deeper lineup at the position", "season to date (shrunk toward position prior)"),
    "pl_cf_xwoba": ("Position-pool xwOBA — center-field seat", "xwOBA (0–1)", "higher = deeper lineup at the position", "season to date (shrunk toward position prior)"),
    "pl_lf_xwoba": ("Position-pool xwOBA — left-field seat", "xwOBA (0–1)", "higher = deeper lineup at the position", "season to date (shrunk toward position prior)"),
}

# 2026-09-27 per-side twin families the originals above don't cover.
# Level twins: tuple (label, units, direction, window) — the twin IS its
# diff's input for that side, so home − away reproduces the diff.
_LEVEL_TWIN_FAMILIES = {
    "rest_days": ("Days of rest entering the game", "days", "more rest = fresher club", "per game (capped 1–6)"),
    "sp_era_5g": ("SP recent runs allowed per nine", "runs / 9 innings", "lower = better", "5 prior appearances; shrunk toward older pitcher history with 30 pseudo-IP"),
    "sp_fbvelo_3g": ("SP fastball velocity, last 3 starts", "mph", "higher = better", "3g"),
    "lineup_re24_std": ("Projected lineup RE24 dispersion (std dev)", "RE24 (std, runs per PA)", "n/a (order-quality spread)", "season to date (shrunk)"),
    "bullpen_pitches_3d": ("Bullpen pitches thrown, last 3 days", "pitches", "more = heavier workload", "3d"),
    "team_hardhit_15g": ("Team hard-hit rate", "rate (0–1)", "higher = better", "15g"),
    "time_zones_crossed_last_3d": ("Time zones crossed over the last 3 days", "zones", "more = travel fatigue", "last 3 days"),
}

# Interaction twins: each side's OWN product of the interaction's factors.
_INTERACTION_TWIN_FAMILIES = {
    "pitcher_regression_indicator": ("SP regression indicator (fastball velo × ERA, last 5 starts)", "index", "n/a (regression signal)"),
    "lineup_depth_multiplier": ("Lineup depth multiplier (mean RE24 × top-3 RE24)", "index", "higher = deeper, star-heavier lineup"),
    "ace_efficiency_factor": ("Ace efficiency factor (K/9 × whiff rate)", "index", "higher = strikeout volume backed by raw stuff"),
}

# Experiment #2 per-side halves (the diffs' own scratch, served).
_EXP2_FAMILIES = {
    "exp2_centered_k": ("Centered strikeout matchup (SP K/9 vs opponent K-rate around league)", "index", "positive = more extreme home K matchup"),
    "exp2_cat_k_fastball": ("Fastball category strikeout matchup", "index", "positive = more extreme home fastball-K matchup"),
    "exp2_cat_k_breaking": ("Breaking category strikeout matchup", "index", "positive = more extreme home breaking-K matchup"),
    "exp2_cat_k_offspeed": ("Offspeed category strikeout matchup", "index", "positive = more extreme home offspeed-K matchup"),
    "exp2_cat_xwoba_fastball": ("Fastball category xwOBA matchup", "index", "positive = more extreme home fastball-xwOBA matchup"),
    "exp2_cat_xwoba_breaking": ("Breaking category xwOBA matchup", "index", "positive = more extreme home breaking-xwOBA matchup"),
    "exp2_cat_xwoba_offspeed": ("Offspeed category xwOBA matchup", "index", "positive = more extreme home offspeed-xwOBA matchup"),
    "exp2_cat_platoon_k_fastball": ("Platoon fastball strikeout matchup", "index", "positive = more extreme home platoon-K matchup"),
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
    "lineup_re24_mean_delta": ("Lineup RE24 momentum (today's lineup − season lineup)", "RE24 (runs per PA)", "positive = current lineup stronger than season average", "per-game lineup − season"),
    "lineup_re24_top3_delta": ("Top-3 RE24 momentum (today's top-3 − season top-3)", "RE24 (runs per PA)", "positive = star power up today", "per-game lineup − season"),
}


# Phase 2 lineup-delta families (actual starting-9 wOBA vs team season, per
# side). Tuple: (label, units, direction).
_LINEUP_DELTA_FAMILIES = {
    "lineup_actual_woba_delta": ("Lineup wOBA delta (actual 9 − team season)", "wOBA points", "positive = tonight's 9 better than the season-average lineup"),
    "lineup_actual_top3_delta": ("Lineup top-3 wOBA delta (actual 9 top-3 − team top-3 regulars)", "wOBA points", "positive = star power up tonight"),
    "lineup_rest_count": ("Resting regulars (team top-5 wOBA not in tonight's 9)", "count (0–5)", "higher = more stars resting"),
}


# Categorical-context columns (TREE_CATEGORICAL_COLS inputs, NOT in
# MONEYLINE_FEATURE_COLS — so they live in a dedicated payload section rather than the
# MONEYLINE_FEATURE_COLS walk, which would trip the stale-entry warning). Emitted as
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
            fam = _LEVEL_TWIN_FAMILIES.get(base)
            if fam:
                label, units, direction, window = fam
                return {
                    "summary": f"{label} — {side} team",
                    "definition": (
                        f"Per-side level column: the {side} club's {label.lower()}. "
                        f"This IS the {base}_diff input for its side — the same "
                        f"strictly-prior source, so home − away reproduces the diff "
                        f"by construction. Tree-only routing; a side with no "
                        f"observation ships NULL like its diff."
                    ),
                    "formula": name,
                    "source": "Ingest-layer rolling/team state (same source as the diff)",
                    "window": window,
                    "units": units,
                    "direction": f"{direction} ({side} side)",
                }
            fam = _INTERACTION_TWIN_FAMILIES.get(base)
            if fam:
                label, units, direction = fam
                return {
                    "summary": f"{label} — {side} side",
                    "definition": (
                        f"Per-side interaction level: the {side} club's OWN product "
                        f"of the factors behind {base}_diff — the within-side form "
                        f"the cross-side gap summarizes (home − away of the pair "
                        f"equals the diff only up to cross terms). Tree-only "
                        f"routing; missing factors propagate NULL."
                    ),
                    "formula": name,
                    "source": "DuckDB feature engineering: within-side product",
                    "window": "per game",
                    "units": units,
                    "direction": f"{direction} ({side} side)",
                }
            fam = _EXP2_FAMILIES.get(base)
            if fam:
                label, units, direction = fam
                return {
                    "summary": f"{label} — {side} side",
                    "definition": (
                        f"Per-side half of the exp2 matchup composite {base}_diff: "
                        f"the same frozen per-side arithmetic the diff was computed "
                        f"from, served under a final name (home − away == diff by "
                        f"construction). Tree-only routing; a side without tracked "
                        f"pitch-category usage stays NULL."
                    ),
                    "formula": name,
                    "source": "Experiment #2 source layer (PIT-safe season-to-date aggregates)",
                    "window": "season to date (strictly prior games)",
                    "units": units,
                    "direction": f"{direction} ({side} side)",
                }
            fam = _PER_SIDE_FAMILIES.get(base)
            if not fam:
                return None
            # (label, units, direction[, window]) — window is optional;
            # families that carry one (e.g. the sp_* lookbacks, added
            # 2026-09-30) pin their artifact label explicitly, the rest
            # fall back to _family_window's name-suffix inference.
            label, units, direction = fam[0], fam[1], fam[2]
            window = fam[3] if len(fam) > 3 else _family_window(base)
            return {
                "summary": f"{label} — {side} team",
                "definition": (
                    f"Per-side level column: the {side} club's {label.lower()}. "
                    f"The corresponding '{base}_diff' carries the matchup signal; "
                    f"this level column lets tree models learn nonlinear context."
                ),
                "formula": name,
                "source": "Statcast aggregates via DuckDB feature engineering",
                "window": window,
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
    """Build the metadata dict keyed by ACTIVE SERVING names.

    Returns (metadata, warnings_list). Every active-moneyline serving col
    (adopted RFE subset, else the full universe) gets a row; unauthored
    features get a clearly-marked placeholder AND a warning string
    so absence is never silent."""
    from training import (
        KNOWN_FEATURE_COLS,
        MONEYLINE_FEATURE_COLS,
        _logistic_feature_cols,
        active_moneyline_feature_cols,
    )

    serving_cols = active_moneyline_feature_cols()
    logistic_cols = set(_logistic_feature_cols())
    meta: dict[str, dict] = {}
    warnings: list[str] = []
    for name in serving_cols:
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
    # Authored-but-not-serving entries would silently rot — warn. Scope the
    # check to entries that are ORPHANED: authored, not serving, AND not in
    # the known pool (universe + every RFE candidate). A feature RFE
    # considered and did not select is intentional, not rot; warning on it
    # is a false positive that teaches reviewers to ignore the line. The
    # 2026-09-26 run warned on lineup_il_flag_{home,away,diff} purely
    # because they are candidates rather than selected — three warnings a
    # day for a healthy state. Only a name the pool no longer knows at all
    # (renamed or dropped) still warns.
    orphaned = sorted(set(_RICH) - set(serving_cols) - set(KNOWN_FEATURE_COLS))
    if orphaned:
        msg = (
            "Feature metadata: authored entries in neither the active "
            "serving width nor the known feature pool (renamed or dropped?): "
            f"{orphaned}"
        )
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
