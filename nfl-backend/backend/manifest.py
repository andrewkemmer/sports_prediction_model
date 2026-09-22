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
        "yac_att": (
            "Trailing yards after catch per attempt",
            "Per-team trailing mean of yards after catch over pass attempts — separation / open-field yards (receiver play)",
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
    }
    window_doc = {
        "ewm": "decaying (halflife=2 games)",
        "roll": "4 games",
        "roll_opp": "6 games (config.OPP_ADJ_WINDOW; opponent-adjusted series)",
    }
    for spec in (_c.PBP_CANDIDATE_TRAILING_SPECS, _c.PBP_OPP_ADJ_TRAILING_SPECS):
        for metric, windows in spec.items():
            desc, definition, _cat = metric_doc[metric]
            for window in windows:
                base = f"pbp_{metric}_{window}"
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
                mvp = ("NaN when the team has no prior games, PBP is unavailable for a "
                       "season, or the source column is absent (pre-v2 pbp cache / "
                       "unpublished season"
                       + ("; no opponent-prior games shrinks fully to the league mean"
                          if metric.endswith("_opp_adj") else "")
                       + "); in-model handling")
                CANDIDATE_MANIFEST[f"{base}_diff"] = {
                    "description": f"Home minus away {desc.lower()}",
                    "definition": f"pbp_{metric}_{window}_home - pbp_{metric}_{window}_away — {definition}",
                    "source": "nflverse play-by-play (per-game rollup)",
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
                        "source": "nflverse play-by-play (per-game rollup)",
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
    # Candidate documentation must name the declared candidate list exactly.
    # (Candidates are triable-but-unserved, so they live in CANDIDATE_MANIFEST,
    # not FEATURE_MANIFEST — until an adoption promotes them.)
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
    declared = list(getattr(_c, "PBP_CANDIDATE_COLS", []))
    documented = list(CANDIDATE_MANIFEST)
    for f in declared:
        if f not in documented:
            problems.append(f"declared candidate {f!r} missing from candidate manifest")
    for f in documented:
        if f not in declared:
            problems.append(f"candidate-manifest entry {f!r} is not a declared candidate")
    return problems
