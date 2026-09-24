"""Authoritative NFL feature manifest — the documentation of the one list.

Every production feature is documented here: definition, source, lookback,
aggregation, point-in-time rule, missing-value policy, representation, and
version. ``config.MONEYLINE_FEATURE_COLS`` is the served contract and the
single source of truth; this manifest is its documentation (validated
name-for-name by ``validate()``), never a parallel list.

``representation`` carries the member routing that ``config.RAW_PER_SIDE_COLS``
encodes: raw home/away levels reach the tree members only, while diffs,
flags and the anchor reach every family.

Feature-set version: see config.FEATURE_SET_VERSION.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# pbp candidate-pool documentation (2026-09-22 expansion)
# ---------------------------------------------------------------------------
# Generated entries for every declared RFE candidate
# (config.PBP_CANDIDATE_COLS = pbp_<metric>_<window>_{diff,home,away}).
# Candidates are NOT in config.MONEYLINE_FEATURE_COLS until an RFE adoption
# promotes them, and validate() only checks the SERVED list name-for-name —
# so candidate entries live in a side table consumed at admission time:
#   * feature_selection._feature_context merges it into the workbook's
#     per-feature metadata, so trials and the decision workbook document
#     every candidate exactly like a served feature.
#   * config.assert_candidate_manifest_parity (called by the pipeline before
#     RFE) asserts the coverage is complete — no undocumented candidate.
CANDIDATE_MANIFEST: dict[str, dict] = {}


def _build_candidate_manifest() -> None:
    """Populate CANDIDATE_MANIFEST from the config specs — name-for-name with
    config.PBP_CANDIDATE_COLS (asserted by config.assert_candidate_manifest
    _parity)."""
    try:
        from backend import config as _c
    except ImportError:  # running as a top-level module
        import config as _c

    metric_doc = {
        "epa_play": (
            "Trailing EPA per play",
            "Per-team EWM/rolling mean of the team's prior games' mean offensive EPA per play (nflverse epa over the possession snaps)",
            "efficiency"),
        "def_epa_play": (
            "Trailing EPA allowed per play",
            "Per-team EWM/rolling mean of mean EPA allowed per play on the team's defensive snaps (nflverse epa over defteam snaps; lower = stingier defense)",
            "defense"),
        "epa_play_opp_adj": (
            "Trailing opponent-adjusted EPA per play",
            "Per-game offensive EPA per play rescaled by the PRIOR quality of the defense faced: epa_play + (prior expanding league mean of def EPA allowed − the opponent's shrunk shift(1) halflife-EWM of def EPA allowed, weight n/(n+OPP_ADJ_SHRINKAGE)); production against a good defense counts more, against a bad one less",
            "efficiency"),
        "qb_epa_dropback": (
            "Trailing QB EPA per dropback",
            "Per-team trailing mean of QB EPA on dropbacks (pass attempts + sacks); QB play quality beyond volume",
            "efficiency"),
        "cpoe_play": (
            "Trailing completion percentage over expectation",
            "Per-team trailing mean of nflverse cpoe on dropbacks — accuracy over expectation (qb play quality)",
            "efficiency"),
        "air_yards_att": (
            "Trailing air yards per attempt",
            "Per-team trailing mean of air yards over pass attempts — passing depth (aggressive downfield vs checkdown profile)",
            "passing profile"),
        "yac_epa_att": (
            "Trailing YAC-as-EPA per attempt",
            "Per-team trailing mean of nflverse yac_epa over pass attempts — separation/open-field value expressed as EPA (nflreadpy publishes no raw yac column; pbp cache v3)",
            "passing profile"),
        "turnovers": (
            "Trailing giveaways",
            "Per-team trailing mean of interceptions + lost fumbles committed on offense (protection)",
            "turnovers"),
        "takeaways": (
            "Trailing takeaways",
            "Per-team trailing mean of opponent interceptions + lost fumbles on the team's defensive snaps (takeaway edge)",
            "turnovers"),
        "sack_rate": (
            "Trailing sack rate",
            "Per-team trailing mean of sacks / dropbacks — pressure allowed (offensive line)",
            "protection"),
        "dropback_rate": (
            "Trailing dropback rate",
            "Per-team trailing mean of dropbacks / plays — passing volume tendency (playcalling)",
            "tendency"),
        "third_down_rate": (
            "Trailing third-down conversion rate",
            "Per-team trailing mean of third downs converted / third-down outcomes (situational strength)",
            "situation"),
        "redzone_td_rate": (
            "Trailing red-zone TD rate",
            "Per-team trailing share of red-zone drives (a drive entering possession inside the 20) ending in a touchdown (finishing)",
            "situation"),
        "start_field_pos": (
            "Trailing starting field position",
            "Per-team trailing mean of starting yardline_100 across drives — field-position edge (special teams / turnovers)",
            "field position"),
        "penalty_yards_pg": (
            "Trailing penalty yards per game",
            "Per-team 4-game mean of penalty yards (discipline)",
            "discipline"),
        "penalties_pg": (
            "Trailing penalty count per game",
            "Per-team 4-game mean of accepted penalties (discipline)",
            "discipline"),
        "fg_accuracy": (
            "Trailing field-goal accuracy",
            "Per-team trailing mean share of field-goal attempts made (kicker/special teams)",
            "special teams"),
        "shotgun_rate": (
            "Trailing shotgun rate",
            "Per-team trailing mean of shotgun snaps / plays (formation tendency)",
            "tendency"),
        "no_huddle_rate": (
            "Trailing no-huddle rate",
            "Per-team trailing mean of no-huddle snaps / plays (pace/formation tendency)",
            "tendency"),
        "drives_pg": (
            "Trailing drives per game",
            "Per-team 4-game mean of distinct offensive drives (possessions/pace)",
            "pace"),
        "rb_load_share": (
            "Trailing RB load share",
            "Per-team trailing mean of RB+FB carries / team rushing attempts — committee vs workhorse backfield (weekly player stats)",
            "usage"),
        "wr1_target_share": (
            "Trailing WR1 target share",
            "Per-team trailing mean of the highest-WR target share of team targets — a true WR1 threat vs target dispersal (weekly player stats)",
            "usage"),
        "cpoe": (
            "Trailing NGS completion % above expectation",
            "Per-team attempts-weighted mean of Next-Gen-Stats completion_percentage_above_expectation across the week's QBs (weekly tracking, week>0 only)",
            "tracking"),
        "rush_eff": (
            "Trailing NGS rush efficiency",
            "Per-team attempts-weighted mean of Next-Gen-Stats rush_yards_over_expected_per_attempt across the week's RB/FBs (weekly tracking)",
            "tracking"),
        "sep": (
            "Trailing NGS separation",
            "Per-team targets-weighted mean of Next-Gen-Stats avg_separation across the week's WR/TEs (weekly tracking)",
            "tracking"),
        "two_min_trail_share": (
            "Trailing two-minute trailing play share",
            "Per-team trailing share of plays with the team BEHIND in the final 2 minutes of a half (qtr 2/4, half_seconds_remaining <= 120, score_differential < 0) - hurry-up tendency",
            "situational"),
        "fourth_go_rate": (
            "Trailing fourth-down aggression",
            "Per-team trailing share of 4th downs with <= 2 yards to go that were gone-for (pass/run) - fourth-down aggressiveness",
            "situational"),
        "fourth_downs_pg": (
            "Trailing fourth-down volume",
            "Per-team trailing 4-game sum of 4th-down plays (how often a team faces - and stays in - fourth down)",
            "situational"),
        "close_run_rate": (
            "Trailing close-game run rate",
            "Per-team trailing share of run plays on snaps with |score_differential| <= 8 outside goal-to-go - run balance under scoreboard pressure",
            "situational"),
        "def_box": (
            "Trailing defenders-in-the-box",
            "Per-team trailing mean of charted defenders in the box per offensive play (run-fit commitment faced; FTN charting)",
            "platoon"),
        "off_backfield": (
            "Trailing backfield count",
            "Per-team trailing mean of charted offensive players in the backfield per play (single-back vs two-back shape; FTN charting)",
            "platoon"),
        "motion_rate": (
            "Trailing pre-snap motion rate",
            "Per-team trailing share of plays with pre-snap motion (FTN charting)",
            "platoon"),
        "play_action_rate": (
            "Trailing play-action rate",
            "Per-team trailing share of plays with play action (FTN charting)",
            "platoon"),
        "rpo_rate": (
            "Trailing RPO rate",
            "Per-team trailing share of run-pass-option plays (FTN charting)",
            "platoon"),
        "screen_rate": (
            "Trailing screen rate",
            "Per-team trailing share of screen passes (FTN charting)",
            "platoon"),
        "rb_snap_share": (
            "Trailing RB/FB snap share",
            "Per-team trailing share of offensive snaps taken by RBs+FBs (backfield commitment by participation)",
            "platoon"),
        "te_snap_share": (
            "Trailing TE snap share",
            "Per-team trailing share of offensive snaps taken by TEs (multi-TE personnel heaviness proxy)",
            "platoon"),
        "te2_snap_share": (
            "Trailing secondary-TE snap share",
            "Per-team trailing share of offensive snaps taken by TEs beyond the team's most-used TE that game (true two-TE usage)",
            "platoon"),
        "wr1_snap_share": (
            "Trailing WR1 snap share",
            "Per-team trailing share of offensive snaps taken by the team's most-used WR (WR1 workload/availability)",
            "platoon"),
        "qb_snap_share": (
            "Trailing QB snap share",
            "Per-team trailing share of offensive snaps taken by QBs (starter availability/health proxy; 1.0 = one QB all game)",
            "platoon"),
        "db_snap_share": (
            "Trailing DB snap share",
            "Per-team trailing share of defensive snaps taken by defensive backs (nickel/dime sub-package rate)",
            "platoon"),
        "dl_snap_share": (
            "Trailing DL snap share",
            "Per-team trailing 4-game share of defensive snaps taken by defensive linemen (front rotation)",
            "platoon"),
    }
    window_doc = {
        "ewm": "decaying (halflife=2 games)",
        "roll": "4 games",
        "roll_opp": "6 games (config.OPP_ADJ_WINDOW; opponent-adjusted series)",
    }
    family_source = {
        "pbp": "nflverse play-by-play (per-game rollup)",
        "ps": "nflverse weekly player stats (per-game rollup)",
        "ngs": "nflverse Next-Gen Stats weekly tracking (per-game rollup)",
        "ftn": "nflverse FTN charting (per-game rollup; published 2022+)",
        "sc": "nflverse snap counts (per-game rollup; published 2013+)",
    }
    for _family, family_specs in _c.CANDIDATE_FAMILIES.items():
        for _spec_name, spec in family_specs.items():
            for metric, windows in spec.items():
                desc, definition, _cat = metric_doc[metric]
                for window in windows:
                    base = f"{_family}_{metric}_{window}"
                    pit = (f"per-team EWM (halflife={_c.EWM_HALFLIFE}) of the per-game metric "
                           "then shift(1) — current and future games excluded"
                           if window == "ewm" else
                           f"per-team rolling({_c.PBP_ROLL_WINDOW}).mean().shift(1) — current "
                           "and future games excluded")
                    lookback = (f"decaying (halflife={_c.EWM_HALFLIFE} games)"
                                if window == "ewm" else f"{_c.PBP_ROLL_WINDOW} games")
                    if window == "roll_opp":
                        pit = (f"per-team rolling({_c.OPP_ADJ_WINDOW}).mean().shift(1) — current "
                               "and future games excluded")
                        lookback = f"{_c.OPP_ADJ_WINDOW} games"
                    mvp = ("NaN when the team has no prior games, the source data is "
                           "unavailable for a season (pre-v3 pbp cache / unpublished "
                           "season / absent source column"
                           + ("; no opponent-prior games shrinks fully to the league mean"
                              if metric.endswith("_opp_adj") else "")
                           + "); in-model handling")
                    CANDIDATE_MANIFEST[f"{base}_diff"] = {
                        "description": f"Home minus away {desc.lower()}",
                        "definition": f"{base}_home - {base}_away — {definition}",
                        "source": family_source[_family],
                        "lookback": lookback,
                        "aggregation": "per-team trailing mean of the per-game metric",
                        "point_in_time_rule": pit,
                        "missing_value_policy": mvp,
                        "representation": "difference (all model families)",
                        "model_family_availability": ["linear", "tree", "mlp"],
                        "feature_version": 2,
                        "candidate": True,
                    }
                    for side, rep_name, fams in (("home", "raw home level", ["tree"]),
                                                 ("away", "raw away level", ["tree"])):
                        CANDIDATE_MANIFEST[f"{base}_{side}"] = {
                            "description": f"{side.capitalize()} team's {desc.lower()}",
                            "definition": f"The {side} team's own {definition} — {definition}",
                            "source": family_source[_family],
                            "lookback": lookback,
                            "aggregation": "per-team trailing mean of the per-game metric",
                            "point_in_time_rule": pit,
                            "missing_value_policy": mvp,
                            "representation": f"{rep_name} (tree members)",
                            "model_family_availability": fams,
                            "feature_version": 2,
                            "candidate": True,
                        }


_build_candidate_manifest()

# ---------------------------------------------------------------------------
# Static per-side candidate facts (config.STATIC_SIDE_CANDIDATES): raw
# home/away levels of served DIFF-only pre-game facts, attached by
# features._attach_static_team_facts from the weekly injury reports and the
# venue geometry table. The served diff remains the primary signal; these
# levels give the tree members the sides separately.
_STATIC_SIDE_DOC = {
    "travel_miles": (
        "Distance to game venue",
        "Great-circle (haversine) miles from the team's home stadium to the game venue",
        "nflverse stadium geography (committed venue table)",
        "pre-game static fact",
        "NaN when either stadium is unlisted"),
    "inj_qb_out": (
        "QB Out count on the weekly report",
        "Count of QBs listed Out on the team's weekly injury report",
        "nflverse weekly injury reports",
        "report entering the game week",
        "NaN when the team-week is absent from the reports entirely"),
    "inj_tackle_out": (
        "Tackle Out count on the weekly report",
        "Count of tackles (T/OT/LT/RT) listed Out on the team's weekly injury report",
        "nflverse weekly injury reports",
        "report entering the game week",
        "NaN when the team-week is absent from the reports entirely"),
    "inj_edge_out": (
        "Edge Out count on the weekly report",
        "Count of edge defenders (EDGE/DE/OLB) listed Out on the team's weekly injury report",
        "nflverse weekly injury reports",
        "report entering the game week",
        "NaN when the team-week is absent from the reports entirely"),
    "inj_starters_out": (
        "Total Out count on the weekly report",
        "Count of ALL players listed Out on the team's weekly injury report",
        "nflverse weekly injury reports",
        "report entering the game week",
        "NaN when the team-week is absent from the reports entirely"),
}


def _build_static_side_manifest() -> None:
    """Document config.STATIC_SIDE_CANDIDATES name-for-name (5 bases x 2 sides)."""
    try:
        from backend import config as _c
    except ImportError:  # running as a top-level module
        import config as _c
    for _base in _c.STATIC_SIDE_CANDIDATES:
        desc, definition, source, pit, mvp = _STATIC_SIDE_DOC[_base]
        for side, rep_name in (("home", "raw home level"), ("away", "raw away level")):
            CANDIDATE_MANIFEST[f"{_base}_{side}"] = {
                "description": f"{side.capitalize()} team's {desc.lower()}",
                "definition": f"The {side} team's {definition}",
                "source": source,
                "lookback": "current week only",
                "aggregation": "per-side weekly fact",
                "point_in_time_rule": pit,
                "missing_value_policy": mvp + "; in-model handling",
                "representation": f"{rep_name} (tree members)",
                "model_family_availability": ["tree"],
                "feature_version": 3,
                "candidate": True,
            }


_build_static_side_manifest()

# One entry per served feature. Field order mirrors the spec (section 11).
FEATURE_MANIFEST = {
    "elo_diff": {
        "description": "Home minus away pre-game Elo rating",
        "definition": "elo_home_entering - elo_away_entering; Elo update r += K*(actual - expected), expected = 1/(1+10**((r_opp - r_self)/400)); actual = 1 win / 0 loss / 0.5 tie",
        "source": "nflverse schedules (all decided REG games, 2018 warmup onward)",
        "lookback": "full history (iterative)",
        "aggregation": "iterative state update",
        "point_in_time_rule": "rating entering kickoff; updated only AFTER a game settles",
        "missing_value_policy": "ELO_PRIOR (1500) for a team's first-ever game; never NaN",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 1,
    },
    "win_pct_diff": {
        "description": "Home minus away trailing win percentage",
        "definition": "mean(team_win) over the team's prior 12 games (ties = 0.5)",
        "source": "decided game outcomes",
        "lookback": 12,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(12).mean().shift(1) — current and future games excluded",
        "missing_value_policy": "NaN when the team has no prior games; in-model handling",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 1,
    },
    "rest_days_diff": {
        "description": "Home minus away days since each team's previous game",
        "definition": "(gameday_t - gameday_{t-1}).days per team",
        "source": "nflverse schedules",
        "lookback": 1,
        "aggregation": "date difference",
        "point_in_time_rule": "days since the team's own strictly-prior game",
        "missing_value_policy": "NaN for a team's first game of the window; in-model handling",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 1,
    },
    "is_dome_home": {
        "description": "Home venue roof is a dome (or closed retractable)",
        "definition": "1.0 if roof in {dome, closed}; 0.0 if outdoors; NaN unknown",
        "source": "nflverse schedule roof field",
        "lookback": 0,
        "aggregation": "static pre-game fact",
        "point_in_time_rule": "venue attribute known before kickoff",
        "missing_value_policy": "NaN when roof is unknown/unlisted (e.g. international sites); never fabricated",
        "representation": "game-level flag (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 1,
    },
    "ewm_net_pts_diff": {
        "description": "Home minus away exponentially-weighted net points per game",
        "definition": "ewm(halflife=2).mean() of the team's net points (for - against) over strictly-prior games",
        "source": "decided game scores",
        "lookback": "decaying (halflife=2 games)",
        "aggregation": "per-team EWM",
        "point_in_time_rule": "per-team ewm over prior games then shift(1)",
        "missing_value_policy": "NaN when the team has no prior games",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 1,
    },
    "ewm_ypp_diff": {
        "description": "Home minus away exponentially-weighted net yards per play",
        "definition": "ewm(halflife=2).mean() of the team's game-level yards_gained/n_plays over strictly-prior games",
        "source": "nflverse play-by-play",
        "lookback": "decaying (halflife=2 games)",
        "aggregation": "per-team EWM of per-game yardage efficiency",
        "point_in_time_rule": "per-team ewm over prior games then shift(1)",
        "missing_value_policy": "NaN when no prior games or PBP unavailable for a season",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 1,
    },
    "pace_plays_min_diff": {
        "description": "Home minus away trailing plays per minute",
        "definition": "trailing 4-game mean of n_plays/elapsed_min per game",
        "source": "nflverse play-by-play clock",
        "lookback": 4,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(4).mean().shift(1)",
        "missing_value_policy": "NaN when prior games lack clock data",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 1,
    },
    "rest_short_diff": {
        "description": "Home minus away short-rest flag (rest < 7 days)",
        "definition": "1.0 when rest_days < 7 else 0.0; home flag minus away flag",
        "source": "nflverse schedules",
        "lookback": 1,
        "aggregation": "thresholded date difference",
        "point_in_time_rule": "function of the team's strictly-prior game date",
        "missing_value_policy": "NaN when either team has no prior game",
        "representation": "difference flag (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 1,
    },
    "div_game": {
        "description": "Division matchup flag",
        "definition": "1.0 when home and away share a division else 0.0",
        "source": "nflverse schedule div_game field",
        "lookback": 0,
        "aggregation": "static pre-game fact",
        "point_in_time_rule": "known before kickoff",
        "missing_value_policy": "NaN when the source field is missing",
        "representation": "game-level flag (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 1,
    },
    "travel_miles_diff": {
        "description": "Home minus away distance traveled to the venue (miles)",
        "definition": "haversine(team home stadium, game stadium) home minus away; earth radius 3958.8 mi",
        "source": "committed nfl_stadiums.csv (real coordinates) + nflverse stadium names",
        "lookback": 0,
        "aggregation": "static pre-game fact",
        "point_in_time_rule": "venue geography, known before kickoff",
        "missing_value_policy": "NaN when a stadium is missing from the table; never fabricated",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 1,
    },
    "altitude_home": {
        "description": "Game-venue elevation in feet",
        "definition": "altitude_ft of the game stadium",
        "source": "committed nfl_stadiums.csv (SRTM/Open-Elevation)",
        "lookback": 0,
        "aggregation": "static pre-game fact",
        "point_in_time_rule": "venue attribute known before kickoff",
        "missing_value_policy": "NaN when the venue is missing from the table",
        "representation": "home-side value (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 1,
    },
    "prime_time": {
        "description": "Evening-kickoff flag",
        "definition": "1.0 when ET kickoff hour >= 17 else 0.0",
        "source": "nflverse schedule gametime (ET)",
        "lookback": 0,
        "aggregation": "static pre-game fact",
        "point_in_time_rule": "schedule fact known before kickoff",
        "missing_value_policy": "NaN when gametime is missing",
        "representation": "game-level flag (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 1,
    },
    "is_turf_home": {
        "description": "Home venue playing surface is artificial turf",
        "definition": "1.0 when the schedule surface normalizes to a turf family (fieldturf/matrixturf/sportturf/astroturf/a_turf); 0.0 when grass; NaN when unlisted",
        "source": "nflverse schedule surface field",
        "lookback": 0,
        "aggregation": "static pre-game fact",
        "point_in_time_rule": "venue attribute known before kickoff",
        "missing_value_policy": "NaN when surface is unlisted/blank; never fabricated",
        "representation": "game-level flag (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 4,
    },
    "temp_f": {
        "description": "Observed game-day temperature (F) at the game venue",
        "definition": "daily mean temperature on the game date from the committed observed-weather table (build_weather_table.py, Open-Meteo archive); schedule temp payload as fallback",
        "source": "committed nfl_weather.csv + nflverse schedule temp",
        "lookback": 0,
        "aggregation": "static pre-game fact",
        "point_in_time_rule": "observed game-day environment, known before kickoff",
        "missing_value_policy": "NaN for domes/international/missing rows; never fabricated",
        "representation": "game-level value (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 4,
    },
    "wind_mph": {
        "description": "Observed game-day wind speed (mph) at the game venue",
        "definition": "daily max sustained wind speed on the game date from the committed observed-weather table; schedule wind payload as fallback",
        "source": "committed nfl_weather.csv + nflverse schedule wind",
        "lookback": 0,
        "aggregation": "static pre-game fact",
        "point_in_time_rule": "observed game-day environment, known before kickoff",
        "missing_value_policy": "NaN for domes/international/missing rows; never fabricated",
        "representation": "game-level value (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 4,
    },
    "is_precip": {
        "description": "Observed precipitation at the game venue (game day)",
        "definition": "1.0 when observed daily precipitation >= PRECIP_FLAG_IN (0.1 in), 0.0 otherwise; NaN unknown",
        "source": "committed nfl_weather.csv (Open-Meteo archive precipitation)",
        "lookback": 0,
        "aggregation": "static pre-game fact",
        "point_in_time_rule": "observed game-day environment, known before kickoff",
        "missing_value_policy": "NaN when the weather table lacks the (stadium, gameday) row; never fabricated",
        "representation": "game-level flag (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 4,
    },
    "is_snow": {
        "description": "Observed snowfall at the game venue (game day)",
        "definition": "1.0 when observed daily snowfall >= SNOW_FLAG_IN (0.1 in), 0.0 otherwise; NaN unknown",
        "source": "committed nfl_weather.csv (Open-Meteo archive snowfall)",
        "lookback": 0,
        "aggregation": "static pre-game fact",
        "point_in_time_rule": "observed game-day environment, known before kickoff",
        "missing_value_policy": "NaN when the weather table lacks the (stadium, gameday) row; never fabricated",
        "representation": "game-level flag (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 4,
    },
    "inj_qb_out_diff": {
        "description": "Home minus away QBs ruled Out on the week's report",
        "definition": "count of QBs with report_status == 'Out' on the week's injury report, home minus away",
        "source": "nflverse weekly injury reports (load_injuries)",
        "lookback": 0,
        "aggregation": "pre-game fact (weekly report)",
        "point_in_time_rule": "the week's published report — known before kickoff",
        "missing_value_policy": "NaN when the season's report is unavailable; never fabricated",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 4,
    },
    "inj_tackle_out_diff": {
        "description": "Home minus away tackles ruled Out on the week's report",
        "definition": "count of T/OT/LT/RT with report_status == 'Out' on the week's injury report, home minus away",
        "source": "nflverse weekly injury reports (load_injuries)",
        "lookback": 0,
        "aggregation": "pre-game fact (weekly report)",
        "point_in_time_rule": "the week's published report — known before kickoff",
        "missing_value_policy": "NaN when the season's report is unavailable; never fabricated",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 4,
    },
    "inj_edge_out_diff": {
        "description": "Home minus away edge rushers ruled Out on the week's report",
        "definition": "count of EDGE/DE/OLB/OL with report_status == 'Out' on the week's injury report, home minus away",
        "source": "nflverse weekly injury reports (load_injuries)",
        "lookback": 0,
        "aggregation": "pre-game fact (weekly report)",
        "point_in_time_rule": "the week's published report — known before kickoff",
        "missing_value_policy": "NaN when the season's report is unavailable; never fabricated",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 4,
    },
    "inj_starters_out_diff": {
        "description": "Home minus away players ruled Out on the week's report",
        "definition": "count of ALL players with report_status == 'Out' on the week's injury report, home minus away",
        "source": "nflverse weekly injury reports (load_injuries)",
        "lookback": 0,
        "aggregation": "pre-game fact (weekly report)",
        "point_in_time_rule": "the week's published report — known before kickoff",
        "missing_value_policy": "NaN when the season's report is unavailable; never fabricated",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 4,
    },
    "is_home": {
        "description": "Constant 1.0 anchor for the home-field edge",
        "definition": "constant 1.0 on every row",
        "source": "structural constant",
        "lookback": 0,
        "aggregation": "constant",
        "point_in_time_rule": "structural — no data dependency",
        "missing_value_policy": "never missing",
        "representation": "constant anchor (part of the served contract; every family receives it)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 1,
        "role": "anchor — in config.MONEYLINE_FEATURE_COLS, so trees receive it too",
    },
    # ---- raw per-side levels (tree members only; config.RAW_PER_SIDE_COLS) ---
    # Each pair is the level half of the corresponding served difference. They
    # are declared in the served contract itself (never synthesized by a view)
    # so the tree matrix is a projection of the one list.
    "elo_home": {
        "description": "Home team's pre-game Elo rating",
        "definition": "team Elo entering kickoff (home side of elo_diff)",
        "source": "nflverse schedules (all decided REG games, 2018 warmup onward)",
        "lookback": "full history (iterative)",
        "aggregation": "iterative state update",
        "point_in_time_rule": "rating entering kickoff; updated only AFTER a game settles",
        "missing_value_policy": "ELO_PRIOR (1500) for a team's first-ever game; never NaN",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "elo_away": {
        "description": "Away team's pre-game Elo rating",
        "definition": "team Elo entering kickoff (away side of elo_diff)",
        "source": "nflverse schedules (all decided REG games, 2018 warmup onward)",
        "lookback": "full history (iterative)",
        "aggregation": "iterative state update",
        "point_in_time_rule": "rating entering kickoff; updated only AFTER a game settles",
        "missing_value_policy": "ELO_PRIOR (1500) for a team's first-ever game; never NaN",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "win_pct_home": {
        "description": "Home team's trailing win percentage",
        "definition": "mean(team_win) over the home team's prior 12 games (ties = 0.5)",
        "source": "decided game outcomes",
        "lookback": 12,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(12).mean().shift(1) — current and future games excluded",
        "missing_value_policy": "NaN when the team has no prior games; in-model handling",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "win_pct_away": {
        "description": "Away team's trailing win percentage",
        "definition": "mean(team_win) over the away team's prior 12 games (ties = 0.5)",
        "source": "decided game outcomes",
        "lookback": 12,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(12).mean().shift(1) — current and future games excluded",
        "missing_value_policy": "NaN when the team has no prior games; in-model handling",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "ewm_net_pts_home": {
        "description": "Home team's exponentially-weighted net points per game",
        "definition": "ewm(halflife=2).mean() of the team's net points (for - against) over strictly-prior games",
        "source": "decided game scores",
        "lookback": "decaying (halflife=2 games)",
        "aggregation": "per-team EWM",
        "point_in_time_rule": "per-team ewm over prior games then shift(1)",
        "missing_value_policy": "NaN when the team has no prior games",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "ewm_net_pts_away": {
        "description": "Away team's exponentially-weighted net points per game",
        "definition": "ewm(halflife=2).mean() of the team's net points (for - against) over strictly-prior games",
        "source": "decided game scores",
        "lookback": "decaying (halflife=2 games)",
        "aggregation": "per-team EWM",
        "point_in_time_rule": "per-team ewm over prior games then shift(1)",
        "missing_value_policy": "NaN when the team has no prior games",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "ewm_ypp_home": {
        "description": "Home team's exponentially-weighted net yards per play",
        "definition": "ewm(halflife=2).mean() of the team's game-level yards_gained/n_plays over strictly-prior games",
        "source": "nflverse play-by-play",
        "lookback": "decaying (halflife=2 games)",
        "aggregation": "per-team EWM of per-game yardage efficiency",
        "point_in_time_rule": "per-team ewm over prior games then shift(1)",
        "missing_value_policy": "NaN when no prior games or PBP unavailable for a season",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "ewm_ypp_away": {
        "description": "Away team's exponentially-weighted net yards per play",
        "definition": "ewm(halflife=2).mean() of the team's game-level yards_gained/n_plays over strictly-prior games",
        "source": "nflverse play-by-play",
        "lookback": "decaying (halflife=2 games)",
        "aggregation": "per-team EWM of per-game yardage efficiency",
        "point_in_time_rule": "per-team ewm over prior games then shift(1)",
        "missing_value_policy": "NaN when no prior games or PBP unavailable for a season",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "rest_days_home": {
        "description": "Home team's days since its previous game",
        "definition": "(gameday_t - gameday_{t-1}).days for the home team",
        "source": "nflverse schedules",
        "lookback": 1,
        "aggregation": "date difference",
        "point_in_time_rule": "days since the team's own strictly-prior game",
        "missing_value_policy": "NaN for a team's first game of the window; in-model handling",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "rest_days_away": {
        "description": "Away team's days since its previous game",
        "definition": "(gameday_t - gameday_{t-1}).days for the away team",
        "source": "nflverse schedules",
        "lookback": 1,
        "aggregation": "date difference",
        "point_in_time_rule": "days since the team's own strictly-prior game",
        "missing_value_policy": "NaN for a team's first game of the window; in-model handling",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
}
# ---------------------------------------------------------------------------
# RFE promotions: candidates structurally promoted into the served contract
# (config.MONEYLINE_FEATURE_COLS). Their manifest entries move OUT of the
# candidate manifest (the trial space excludes served names), so re-home the
# generated entries here with candidate=False — one documentation source,
# generated, never re-listed.
_RFE_PROMOTED = [
    "pbp_air_yards_att_ewm_diff",
    "pbp_air_yards_att_ewm_home",
    "pbp_air_yards_att_ewm_away",
]
for _name in _RFE_PROMOTED:
    _entry = CANDIDATE_MANIFEST.pop(_name, None)
    if _entry is None:
        raise RuntimeError(f"promoted feature {_name!r} has no generated manifest entry")
    _entry["candidate"] = False
    _entry["feature_version"] = 4
    FEATURE_MANIFEST[_name] = _entry


def validate() -> list[str]:
    """Manifest consistency checks; returns a list of problems (empty = OK)."""
    problems = []
    try:
        from backend import config as _c
    except ImportError:  # running as a top-level module
        import config as _c
    # The served contract is the ONE list; the manifest documents exactly it.
    served = list(_c.MONEYLINE_FEATURE_COLS)
    if len(set(served)) != len(served):
        problems.append("config.MONEYLINE_FEATURE_COLS contains duplicate names")
    for f in served:
        if f not in FEATURE_MANIFEST:
            problems.append(f"served feature {f!r} missing from manifest")
    for f in FEATURE_MANIFEST:
        if f not in served:
            problems.append(f"manifest feature {f!r} is not served")
    # Raw per-side routing columns must be TRIABLE names (the declared pool):
    # served raws live in MONEYLINE_FEATURE_COLS; candidate raws are triable
    # from RFE_CANDIDATE_COLS and — only after an adoption — served. Both
    # cases are covered by pool membership; a promoted raw's routing (linear
    # members never see it) is enforced by features.linear_view at matrix
    # time, not by this static check.
    pool = list(_c.KNOWN_FEATURE_COLS)
    for f in _c.RAW_PER_SIDE_COLS:
        if f not in pool:
            problems.append(f"raw per-side column {f!r} is not in the declared pool")
    # Candidate documentation must name the effective trial space exactly.
    # (Candidates are triable-but-unserved, so they live in CANDIDATE_MANIFEST,
    # not FEATURE_MANIFEST; structurally promoted names leave both the trial
    # space and the candidate manifest together.)
    problems.extend(config_assert_candidate_parity(_c))
    required_fields = ("definition", "source", "lookback", "aggregation",
                       "point_in_time_rule", "missing_value_policy",
                       "representation", "model_family_availability",
                       "feature_version")
    for f, entry in FEATURE_MANIFEST.items():
        for field_name in required_fields:
            if field_name not in entry:
                problems.append(f"manifest entry {f!r} missing field {field_name!r}")
    return problems


def config_assert_candidate_parity(_c) -> list[str]:
    """Declared-candidate <-> CANDIDATE_MANIFEST parity (shared helper so the
    same check runs in validate() and via config.assert_candidate_manifest
    _parity at pipeline time)."""
    problems: list[str] = []
    declared = list(getattr(_c, "RFE_CANDIDATE_COLS", []))  # effective trial space
    documented = list(CANDIDATE_MANIFEST)
    for f in declared:
        if f not in documented:
            problems.append(f"declared candidate {f!r} missing from candidate manifest")
    for f in documented:
        if f not in declared:
            problems.append(f"candidate-manifest entry {f!r} is not a declared candidate")
    return problems
