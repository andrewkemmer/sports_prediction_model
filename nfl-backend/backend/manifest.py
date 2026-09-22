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
    for f in _c.RAW_PER_SIDE_COLS:
        if f not in served:
            problems.append(f"raw per-side column {f!r} is not in the served contract")
    required_fields = ("definition", "source", "lookback", "aggregation",
                       "point_in_time_rule", "missing_value_policy",
                       "representation", "model_family_availability",
                       "feature_version")
    for f, entry in FEATURE_MANIFEST.items():
        for field_name in required_fields:
            if field_name not in entry:
                problems.append(f"manifest entry {f!r} missing field {field_name!r}")
    return problems
