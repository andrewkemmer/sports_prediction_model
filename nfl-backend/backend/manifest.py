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
# promotes them (or a feature family is explicitly promoted into the contract),
# and validate() only checks the SERVED list name-for-name —
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
            "Per-game offensive EPA per play rescaled by the PRIOR quality of the defense faced: epa_play + (prior expanding league mean of def EPA allowed − the opponent's shrunk shift(1) halflife-EWM of def EPA allowed, weight n/(n+OPP_ADJ_SHRINKAGE) with n = the opponent's full-timeline prior games — prior seasons included, never season-reset); production against a good defense counts more, against a bad one less",
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
# Static per-side facts: the unserved travel-distance candidate plus the
# production EPA lineup levels. The EPA-side descriptions are generated from
# _STATIC_SIDE_DOC_EPA and live in FEATURE_MANIFEST, never in the RFE pool.
_STATIC_SIDE_DOC = {
    "travel_miles": (
        "Distance to game venue",
        "Great-circle (haversine) miles from the team's most recent strictly-prior scheduled home stadium to the game venue",
        "nflverse schedule stadium history + committed venue coordinates",
        "each side uses its latest home stadium strictly before kickoff; no current/future team→venue association",
        "NaN when the team has no prior home venue or either stadium is unlisted"),
}


_EPA_POSITION_LABEL = {
    "epa_qb": "quarterback", "epa_wr": "wide receiver",
    "epa_te": "tight end", "epa_rb": "running back",
}
# The per-position quality family is a DIFFERENT kind of static side fact from
# travel_miles: its lookback is a rolling per-player window and its
# aggregation is a mean of shrunk rates, not a per-game venue fact. It gets
# its own metadata rather than being forced through the venue wording.
_STATIC_SIDE_DOC_EPA = {
    "epa_qb": (
        "Projected-lineup quality",
        "mean shrunk EPA per opportunity of the projected quarterbacks, over an "
        "8-game strictly-prior window, shrunk to a position-segmented league prior",
        "nflverse play-by-play EPA + qb_dropback/pass_attempt/rush_attempt "
        "opportunity flags; weekly player stats for position labels; weekly "
        "roster snapshots (nflverse weekly_rosters)",
        "rolling(8) player rating from games dated before the target date; "
        "candidate ratings come from the prior 21 calendar days to span bye weeks; "
        "an Out, "
        "IR/Injured Reserve, or Doubtful designation excludes a player from "
        "the target pool through any of three unioned channels: a strict-PIT "
        "report published before kickoff (2016-2024), the weekly report-cycle "
        "row for the player's own team-week (the only channel covering "
        "2025/2026, where the PIT feed lacks timestamps), or a weekly-roster-"
        "snapshot unavailability",
        "NaN when no projected player at this position has a prior rating; "
        "non-injury statuses (incl. Questionable), missing/unreported status, "
        "and missing PIT timestamps do not erase player history; exclusion "
        "requires an Out/IR/Doubtful designation on one of the three "
        "channels"),
    "epa_wr": (
        "Projected-lineup quality",
        "mean shrunk EPA per opportunity of the projected wide receivers",
        "nflverse play-by-play EPA + opportunity flags; weekly player stats "
        "for position labels; weekly roster snapshots (nflverse weekly_rosters)",
        "rolling(8) player rating from games dated before the target date; "
        "candidate ratings come from the prior 21 calendar days to span bye weeks; "
        "an Out, "
        "IR/Injured Reserve, or Doubtful designation excludes a player from "
        "the target pool through any of three unioned channels: a strict-PIT "
        "report published before kickoff (2016-2024), the weekly report-cycle "
        "row for the player's own team-week (the only channel covering "
        "2025/2026, where the PIT feed lacks timestamps), or a weekly-roster-"
        "snapshot unavailability",
        "NaN when no projected player at this position has a prior rating; "
        "other/missing statuses (incl. Questionable) and missing PIT timestamps "
        "do not exclude the player; exclusion requires an Out/IR/Doubtful "
        "designation on one of the three channels"),
    "epa_te": (
        "Projected-lineup quality",
        "mean shrunk EPA per opportunity of the projected tight ends",
        "nflverse play-by-play EPA + opportunity flags; weekly player stats "
        "for position labels; weekly roster snapshots (nflverse weekly_rosters)",
        "rolling(8) player rating from games dated before the target date; "
        "candidate ratings come from the prior 21 calendar days to span bye weeks; "
        "an Out, "
        "IR/Injured Reserve, or Doubtful designation excludes a player from "
        "the target pool through any of three unioned channels: a strict-PIT "
        "report published before kickoff (2016-2024), the weekly report-cycle "
        "row for the player's own team-week (the only channel covering "
        "2025/2026, where the PIT feed lacks timestamps), or a weekly-roster-"
        "snapshot unavailability",
        "NaN when no projected player at this position has a prior rating; "
        "other/missing statuses (incl. Questionable) and missing PIT timestamps "
        "do not exclude the player; exclusion requires an Out/IR/Doubtful "
        "designation on one of the three channels"),
    "epa_rb": (
        "Projected-lineup quality",
        "mean shrunk EPA per opportunity of the projected running backs",
        "nflverse play-by-play EPA + opportunity flags; weekly player stats "
        "for position labels; weekly roster snapshots (nflverse weekly_rosters)",
        "rolling(8) player rating from games dated before the target date; "
        "candidate ratings come from the prior 21 calendar days to span bye weeks; "
        "an Out, "
        "IR/Injured Reserve, or Doubtful designation excludes a player from "
        "the target pool through any of three unioned channels: a strict-PIT "
        "report published before kickoff (2016-2024), the weekly report-cycle "
        "row for the player's own team-week (the only channel covering "
        "2025/2026, where the PIT feed lacks timestamps), or a weekly-roster-"
        "snapshot unavailability",
        "NaN when no projected player at this position has a prior rating; "
        "other/missing statuses (incl. Questionable) and missing PIT timestamps "
        "do not exclude the player; exclusion requires an Out/IR/Doubtful "
        "designation on one of the three channels"),
}

_STATIC_SIDE_AGGREGATION = {
    "travel_miles": ("prior home games", "per-side point-in-time venue fact"),
}
_STATIC_SIDE_AGGREGATION_EPA = (
    "8 player-games; 21-calendar-day roster window",
    "opportunity-weighted mean by position among the team's top 11 "
    "prior-opportunity leaders after excluding players with an Out/IR/Doubtful "
    "report published strictly before target kickoff or a weekly-roster-"
    "snapshot unavailability (carried RES/SUS/PUP, same-week INA/CUT); each "
    "member's weight is his own rolling-8 opportunity total, so the blend is "
    "the projected lineup's combined shrunk EPA over combined opportunities",
)


def _build_static_side_manifest() -> None:
    """Document every config.STATIC_SIDE_CANDIDATES base and both sides."""
    try:
        from backend import config as _c
    except ImportError:  # running as a top-level module
        import config as _c
    for _base in _c.STATIC_SIDE_CANDIDATES:
        if _base in _STATIC_SIDE_DOC_EPA:
            desc, definition, source, pit, mvp = _STATIC_SIDE_DOC_EPA[_base]
            lookback, aggregation = _STATIC_SIDE_AGGREGATION_EPA
        else:
            desc, definition, source, pit, mvp = _STATIC_SIDE_DOC[_base]
            lookback, aggregation = _STATIC_SIDE_AGGREGATION[_base]
        for side, rep_name in (("home", "raw home level"), ("away", "raw away level")):
            if f"{_base}_{side}" in _c.MONEYLINE_FEATURE_COLS:
                # Structurally promoted into the served contract (the
                # travel_miles twins, 2026-09-27): documented directly in
                # FEATURE_MANIFEST below, never as a trial candidate.
                continue
            CANDIDATE_MANIFEST[f"{_base}_{side}"] = {
                "description": f"{side.capitalize()} team's {desc.lower()}",
                "definition": f"The {side} team's {definition}",
                "source": source,
                "lookback": lookback,
                "aggregation": aggregation,
                "point_in_time_rule": pit,
                "missing_value_policy": mvp + "; in-model handling",
                "representation": f"{rep_name} (tree members)",
                "model_family_availability": ["tree"],
                "feature_version": 6,
                "candidate": True,
            }


def _build_static_diff_manifest() -> None:
    """Document every config.STATIC_DIFF_CANDIDATES base.

    The diff is the home-side aggregate minus the away-side aggregate for the
    same target game; the tree-only side levels remain available for RFE.
    """
    try:
        from backend import config as _c
    except ImportError:  # running as a top-level module
        import config as _c
    for _base in getattr(_c, "STATIC_DIFF_CANDIDATES", []):
        src = _STATIC_SIDE_DOC_EPA.get(_base) or _STATIC_SIDE_DOC[_base]
        desc, definition, source, pit, mvp = src
        label = _EPA_POSITION_LABEL.get(_base, _base)
        lookback, aggregation = (
            _STATIC_SIDE_AGGREGATION_EPA if _base in _STATIC_SIDE_DOC_EPA
            else _STATIC_SIDE_AGGREGATION[_base])
        CANDIDATE_MANIFEST[f"{_base}_diff"] = {
            "description": f"Home minus away {label} quality",
            "definition": f"{_base}_home - {_base}_away",
            "source": source,
            "lookback": lookback,
            "aggregation": aggregation,
            "point_in_time_rule": pit,
            "missing_value_policy": mvp + "; in-model handling",
            "representation": "difference (all model families)",
            "model_family_availability": ["linear", "tree", "mlp"],
            "feature_version": 6,
            "candidate": True,
        }


_build_static_side_manifest()
_build_static_diff_manifest()

# One entry per served feature. Field order mirrors the spec (section 11).
FEATURE_MANIFEST = {
    "elo_diff": {
        "description": "Home minus away pre-game Elo rating",
        "definition": "elo_home_entering - elo_away_entering; Elo update r += K*(actual - expected), expected = 1/(1+10**((r_opp - r_self)/400)); actual = 1 win / 0 loss / 0.5 tie; ELO_SEASON_REVERT (1/3) toward ELO_PRIOR at each season boundary (MLB/NHL/NBA parity)",
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
        "definition": "(gameday_t - gameday_{t-1}).days within the same (team, season)",
        "source": "nflverse schedules",
        "lookback": 1,
        "aggregation": "date difference",
        "point_in_time_rule": "days since the team's own strictly-prior game in the same season",
        "missing_value_policy": "NaN for a team's first game of the season; in-model handling",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 1,
    },
    "is_dome_home": {
        "description": "Home venue roof is a dome (or closed retractable)",
        "definition": "1.0 when the schedule roof is {dome, closed}, or the schedule is silent and the committed venue is roofed (fixed or retractable); 0.0 when the schedule roof is {outdoors, open} or the committed venue has no roof at all; NaN when both are unknown",
        "source": "nflverse schedule roof field, falling back to the committed nfl_stadiums.csv roof column",
        "lookback": 0,
        "aggregation": "static pre-game fact",
        "point_in_time_rule": "venue attribute known before kickoff",
        "missing_value_policy": "NaN when the schedule roof and the committed venue classification are both unknown; a roofed venue never implies an open-air game day",
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
    "pace_plays_min_home": {
        "description": "Home team's trailing plays per minute",
        "definition": "team pace entering kickoff (home side of pace_plays_min_diff)",
        "source": "nflverse play-by-play clock",
        "lookback": 4,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(4).mean().shift(1) — current and future games excluded",
        "missing_value_policy": "NaN when prior games lack clock data (early-season openers)",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "pace_plays_min_away": {
        "description": "Away team's trailing plays per minute",
        "definition": "team pace entering kickoff (away side of pace_plays_min_diff)",
        "source": "nflverse play-by-play clock",
        "lookback": 4,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(4).mean().shift(1) — current and future games excluded",
        "missing_value_policy": "NaN when prior games lack clock data (early-season openers)",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "rest_short_diff": {
        "description": "Home minus away short-rest flag (rest < 7 days)",
        "definition": "1.0 when within-season rest_days < 7 else 0.0; home flag minus away flag; a season opener prices 0.0 on each side (no in-season predecessor is never short rest), so the diff equals 0 at opener-vs-opener",
        "source": "nflverse schedules",
        "lookback": 1,
        "aggregation": "thresholded date difference",
        "point_in_time_rule": "function of the team's strictly-prior game date in the same season",
        "missing_value_policy": "0.0 at a season opener (was NaN before the side-twin promotion; the opener flag is definitionally 0)",
        "representation": "difference flag (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 10,
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
        "definition": "haversine(prior scheduled home stadium, game stadium) home minus away; a team's first home venue is unavailable rather than inferred from a current team→stadium map",
        "source": "committed nfl_stadiums.csv (real coordinates) + nflverse schedule stadium names",
        "lookback": "prior home games",
        "aggregation": "point-in-time static venue fact",
        "point_in_time_rule": "each side uses its most recent home venue strictly before kickoff; no current/future venue association",
        "missing_value_policy": "NaN until the team has a prior home venue or the venue is unlisted; never fabricated",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 5,
    },
    "altitude_home": {
        "description": "Game-venue elevation in feet",
        "definition": "altitude_ft of the game stadium",
        "source": "committed nfl_stadiums.csv (SRTM/Open-Elevation)",
        "lookback": 0,
        "aggregation": "static pre-game fact",
        "point_in_time_rule": "venue attribute known before kickoff",
        "missing_value_policy": "NaN when the venue is missing from the committed table; every stadium the schedule references is listed",
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
        "description": "Hourly venue temperature immediately before kickoff (F)",
        "definition": "Open-Meteo hourly temperature_2m at the committed stadium coordinates, converted to Fahrenheit; latest API timestamp strictly before kickoff",
        "source": "Open-Meteo hourly archive for past games / hourly forecast for pending games (validated PIT cache)",
        "lookback": "latest hourly reading before kickoff",
        "aggregation": "single latest strictly-prior hourly value; no daily aggregation",
        "point_in_time_rule": "weather_time_utc < kickoff_utc; forecast rows additionally require fetched_at_utc < kickoff_utc; historical archive rows may be retrieved later using the same pre-kickoff observation",
        "missing_value_policy": "NaN for indoor/closed games, unknown venues, invalid kickoffs, missing rows, or unavailable values; never use a daily aggregate or raw schedule fallback",
        "representation": "game-level value (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 6,
    },
    "wind_mph": {
        "description": "Hourly venue wind speed immediately before kickoff (mph)",
        "definition": "Open-Meteo hourly wind_speed_10m at the committed stadium coordinates, converted to mph; latest API timestamp strictly before kickoff",
        "source": "Open-Meteo hourly archive for past games / hourly forecast for pending games (validated PIT cache)",
        "lookback": "latest hourly reading before kickoff",
        "aggregation": "single latest strictly-prior hourly value; no daily max/gust aggregate",
        "point_in_time_rule": "weather_time_utc < kickoff_utc; forecast rows additionally require fetched_at_utc < kickoff_utc",
        "missing_value_policy": "NaN for indoor/closed games, unknown venues, invalid kickoffs, missing rows, or unavailable values; never fabricated",
        "representation": "game-level value (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 6,
    },
    "is_precip": {
        "description": "Hourly precipitation flag immediately before kickoff",
        "definition": "1.0 when the latest strictly-prior hourly Open-Meteo precipitation is >= PRECIP_FLAG_IN (0.1 in), 0.0 when below threshold, NaN when unavailable",
        "source": "Open-Meteo hourly precipitation archive/forecast (validated PIT cache)",
        "lookback": "latest hourly reading before kickoff",
        "aggregation": "threshold of one hourly precipitation total; never a daily sum",
        "point_in_time_rule": "weather_time_utc < kickoff_utc; forecast rows additionally require fetched_at_utc < kickoff_utc",
        "missing_value_policy": "NaN for indoor/closed games and when no strictly-prior hourly precipitation value exists; never inferred from a daily archive or schedule payload",
        "representation": "game-level flag (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 6,
    },
    "is_snow": {
        "description": "Hourly snowfall flag immediately before kickoff",
        "definition": "1.0 when the latest strictly-prior hourly Open-Meteo snowfall, converted to inches from the unit the API itself reports, is >= SNOW_FLAG_IN (0.1 in), 0.0 when below threshold, NaN when unavailable",
        "source": "Open-Meteo hourly snowfall archive/forecast (validated PIT cache)",
        "lookback": "latest hourly reading before kickoff",
        "aggregation": "threshold of one hourly snowfall total; never a daily sum",
        "point_in_time_rule": "weather_time_utc < kickoff_utc; forecast rows additionally require fetched_at_utc < kickoff_utc",
        "missing_value_policy": "NaN for indoor/closed games and when no strictly-prior hourly snowfall value exists; never inferred from a daily archive or schedule payload",
        "representation": "game-level flag (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 6,
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
        "definition": "team Elo entering kickoff (home side of elo_diff); ratings carry across the offseason with a 1/3 revert toward ELO_PRIOR at each season boundary",
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
        "definition": "(gameday_t - gameday_{t-1}).days within the home team's season",
        "source": "nflverse schedules",
        "lookback": 1,
        "aggregation": "date difference",
        "point_in_time_rule": "days since the team's own strictly-prior game in the same season",
        "missing_value_policy": "NaN for a team's first game of the season; in-model handling",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "rest_days_away": {
        "description": "Away team's days since its previous game",
        "definition": "(gameday_t - gameday_{t-1}).days within the away team's season",
        "source": "nflverse schedules",
        "lookback": 1,
        "aggregation": "date difference",
        "point_in_time_rule": "days since the team's own strictly-prior game in the same season",
        "missing_value_policy": "NaN for a team's first game of the season; in-model handling",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
}

# Side-twin promotion (2026-09-27, structure parity with the NHL level
# twins): the served contract carries every diff WITH its raw home/away
# halves. rest_short_home/away are read from the same ladder flag as the
# diff (opener prices 0.0, never NaN, so the twin coherence holds), and
# travel_miles_home/away are the existing per-side outputs of the same
# strictly-prior home-venue computation the diff uses.
FEATURE_MANIFEST["rest_short_home"] = {
    "description": "Home team's short-rest flag (rest < 7 days)",
    "definition": "1.0 when the home team's within-season rest_days < 7 else 0.0; a season opener prices 0.0 (no in-season predecessor is never short rest)",
    "source": "nflverse schedules",
    "lookback": 1,
    "aggregation": "thresholded date difference",
    "point_in_time_rule": "function of the team's strictly-prior game date in the same season",
    "missing_value_policy": "0.0 at a season opener; in-model handling",
    "representation": "raw home level (tree members)",
    "model_family_availability": ["tree"],
    "feature_version": 10,
}
FEATURE_MANIFEST["rest_short_away"] = {
    "description": "Away team's short-rest flag (rest < 7 days)",
    "definition": "1.0 when the away team's within-season rest_days < 7 else 0.0; a season opener prices 0.0 (no in-season predecessor is never short rest)",
    "source": "nflverse schedules",
    "lookback": 1,
    "aggregation": "thresholded date difference",
    "point_in_time_rule": "function of the team's strictly-prior game date in the same season",
    "missing_value_policy": "0.0 at a season opener; in-model handling",
    "representation": "raw away level (tree members)",
    "model_family_availability": ["tree"],
    "feature_version": 10,
}
FEATURE_MANIFEST["travel_miles_home"] = {
    "description": "Home team's travel distance to the game venue (miles)",
    "definition": "haversine(home team's prior scheduled home stadium, game stadium); the home side of travel_miles_diff, from the same strictly-prior home-venue computation",
    "source": "committed nfl_stadiums.csv (real coordinates) + nflverse schedule stadium names",
    "lookback": "prior home games",
    "aggregation": "point-in-time static venue fact",
    "point_in_time_rule": "the home team's most recent home venue strictly before kickoff; no current/future venue association",
    "missing_value_policy": "NaN until the team has a prior home venue or the venue is unlisted; never fabricated; in-model handling",
    "representation": "raw home level (tree members)",
    "model_family_availability": ["tree"],
    "feature_version": 10,
}
FEATURE_MANIFEST["travel_miles_away"] = {
    "description": "Away team's travel distance to the game venue (miles)",
    "definition": "haversine(away team's prior scheduled home stadium, game stadium); the away side of travel_miles_diff, from the same strictly-prior home-venue computation",
    "source": "committed nfl_stadiums.csv (real coordinates) + nflverse schedule stadium names",
    "lookback": "prior home games",
    "aggregation": "point-in-time static venue fact",
    "point_in_time_rule": "the away team's most recent home venue strictly before kickoff; no current/future venue association",
    "missing_value_policy": "NaN until the team has a prior home venue or the venue is unlisted; never fabricated; in-model handling",
    "representation": "raw away level (tree members)",
    "model_family_availability": ["tree"],
    "feature_version": 10,
}


# Weekly-report injury-share family (2026-09-27 Tier B promotion): served
# entries generated from one doc table, name-for-name with
# config.INJURY_SHARE_BASES (validated by validate()). These were promoted
# straight into the contract (never RFE candidates), so they are documented
# here directly instead of being re-homed from the candidate manifest.
# 2026-09-28: the flagged set additionally covers roster-snapshot
# unavailables (same-week RES/INA/CUT/SUS/PUP and carried RES/SUS/PUP
# unless back to ACT) - min(report, roster): the set only grows.
_INJURY_SHARE_DOC = {
    "inj_ol_out": (
        "Out/IR/Doubtful offensive-line report rows",
        "Count of the team's OWN-week Out/IR/Doubtful report rows plus roster-snapshot unavailables (same-week RES/INA/CUT/SUS/PUP and carried RES/SUS/PUP) whose player's most recent prior active snap position is an offensive-line position (T/G/C)",
        "nflverse weekly injury reports + weekly roster snapshots (nflverse weekly_rosters)",
        "this week's report cycle",
        "count of Out/IR/Doubtful report rows",
        "a (season, week, team) report row is pre-kickoff information for that team-week's game by league rule (report cycle; >=99% agreement with the strict-PIT loader on 2016-2024); a row never applies to another week or team. The roster overlay adds weekly-snapshot designations — same-week RES/INA/CUT/SUS/PUP and carried RES/SUS/PUP unless back to ACT — frozen before the week's games",
        "0.0 when the report source is unavailable or the team-week has no admissible rows; measured 100% team-game coverage 2016-2025"),
    "ol_snaps_lost_share": (
        "Offensive-line snap share lost to Out/IR/Doubtful designations",
        "Sum over the week's flagged OL players (Out/IR/Doubtful report rows plus roster-snapshot unavailables) of the player's mean offensive snap share over HIS OWN last 8 active games (any team; offense_snaps > 0 and a published offense_pct)",
        "nflverse weekly injury reports + weekly roster snapshots (nflverse weekly_rosters) + snap counts keyed by the GSIS-PFR player crosswalk",
        "each flagged player's last 8 active games, strictly before the flag week",
        "sum of per-player rolling(8, min_periods=1) mean unit-snap shares",
        "share history is an as-of join strictly BEFORE the flag week (the flag week's own game is excluded, cross-team); a player with no prior active game prices 0.0, never NaN; roster-snapshot unavailables join by (season, week, team, player)",
        "0.0 when a player has no prior history or the report/snap source is unavailable; measured 100% team-game coverage 2016-2025"),
    "ol_key_out": (
        "Key offensive lineman unavailable flag",
        "1.0 when any OL player flagged this week (Out/IR/Doubtful report rows plus roster-snapshot unavailables) carries a mean offensive snap share >= 0.60 over his own last 8 active games, else 0.0",
        "nflverse weekly injury reports + weekly roster snapshots (nflverse weekly_rosters) + snap counts keyed by the GSIS-PFR player crosswalk",
        "each flagged player's last 8 active games, strictly before the flag week",
        "threshold flag over per-player rolling(8) mean unit-snap shares",
        "share history is an as-of join strictly BEFORE the flag week (the flag week's own game is excluded, cross-team); a player with no prior active game prices 0.0, never NaN; roster-snapshot unavailables join by (season, week, team, player)",
        "0.0 when no flagged player clears the 0.60 share threshold or the source is unavailable; measured 100% team-game coverage 2016-2025"),
    "inj_def_out": (
        "Out/IR/Doubtful defensive report rows",
        "Count of the team's OWN-week Out/IR/Doubtful report rows plus roster-snapshot unavailables (same-week RES/INA/CUT/SUS/PUP and carried RES/SUS/PUP) whose player's most recent prior active snap position is a defensive position (LB/CB/S/DE/DT/NT/ILB/OLB/MLB/DB/SAF/SS/FS/DL/EDGE)",
        "nflverse weekly injury reports + weekly roster snapshots (nflverse weekly_rosters)",
        "this week's report cycle",
        "count of Out/IR/Doubtful report rows",
        "a (season, week, team) report row is pre-kickoff information for that team-week's game by league rule (report cycle; >=99% agreement with the strict-PIT loader on 2016-2024); a row never applies to another week or team. The roster overlay adds weekly-snapshot designations — same-week RES/INA/CUT/SUS/PUP and carried RES/SUS/PUP unless back to ACT — frozen before the week's games",
        "0.0 when the report source is unavailable or the team-week has no admissible rows; measured 100% team-game coverage 2016-2025"),
    "def_snaps_lost_share": (
        "Defensive snap share lost to Out/IR/Doubtful designations",
        "Sum over the week's flagged defensive players (Out/IR/Doubtful report rows plus roster-snapshot unavailables) of the player's mean defensive snap share over HIS OWN last 8 active games (any team; defense_snaps > 0 and a published defense_pct)",
        "nflverse weekly injury reports + weekly roster snapshots (nflverse weekly_rosters) + snap counts keyed by the GSIS-PFR player crosswalk",
        "each flagged player's last 8 active games, strictly before the flag week",
        "sum of per-player rolling(8, min_periods=1) mean unit-snap shares",
        "share history is an as-of join strictly BEFORE the flag week (the flag week's own game is excluded, cross-team); a player with no prior active game prices 0.0, never NaN; roster-snapshot unavailables join by (season, week, team, player)",
        "0.0 when a player has no prior history or the report/snap source is unavailable; measured 100% team-game coverage 2016-2025"),
    "def_key_out": (
        "Key defensive player unavailable flag",
        "1.0 when any defensive player flagged this week (Out/IR/Doubtful report rows plus roster-snapshot unavailables) carries a mean defensive snap share >= 0.60 over his own last 8 active games, else 0.0",
        "nflverse weekly injury reports + weekly roster snapshots (nflverse weekly_rosters) + snap counts keyed by the GSIS-PFR player crosswalk",
        "each flagged player's last 8 active games, strictly before the flag week",
        "threshold flag over per-player rolling(8) mean unit-snap shares",
        "share history is an as-of join strictly BEFORE the flag week (the flag week's own game is excluded, cross-team); a player with no prior active game prices 0.0, never NaN; roster-snapshot unavailables join by (season, week, team, player)",
        "0.0 when no flagged player clears the 0.60 share threshold or the source is unavailable; measured 100% team-game coverage 2016-2025"),
}
_INJURY_REP_DOC = {
    "home": ("The home team's {desc}", "raw home level (tree members)", ["tree"]),
    "away": ("The away team's {desc}", "raw away level (tree members)", ["tree"]),
    "diff": ("Home minus away {desc}", "difference (all model families)",
             ["linear", "tree", "mlp"]),
}
for _base, _doc in _INJURY_SHARE_DOC.items():
    _desc, _definition, _source, _lookback, _aggregation, _pit, _mvp = _doc
    for _rep, (_desc_t, _repr, _fams) in _INJURY_REP_DOC.items():
        FEATURE_MANIFEST[f"{_base}_{_rep}"] = {
            "description": _desc_t.format(desc=_desc),
            "definition": _definition,
            "source": _source,
            "lookback": _lookback,
            "aggregation": _aggregation,
            "point_in_time_rule": _pit,
            "missing_value_policy": _mvp,
            "representation": _repr,
            "model_family_availability": _fams,
            "feature_version": 10,
        }


# All 12 EPA columns are production members, not candidates. Create their
# metadata from the same shared side/PIT description; the two raw levels are
# tree-only while the home-away diff is shared by every model family.
try:
    from backend import config as _epa_config
except ImportError:  # running as a top-level module
    import config as _epa_config
for _base, _label in _EPA_POSITION_LABEL.items():
    _desc, _definition, _source, _pit, _mvp = _STATIC_SIDE_DOC_EPA[_base]
    _lookback, _aggregation = _STATIC_SIDE_AGGREGATION_EPA
    for _side in ("home", "away"):
        _name = f"{_base}_{_side}"
        FEATURE_MANIFEST[_name] = {
            "description": f"{_side.capitalize()} team's {_desc.lower()}",
            "definition": f"The {_side} team's {_definition}",
            "source": _source,
            "lookback": _lookback,
            "aggregation": _aggregation,
            "point_in_time_rule": _pit,
            "missing_value_policy": _mvp + "; in-model handling",
            "representation": f"raw {_side} level (tree members)",
            "model_family_availability": ["tree"],
            "feature_version": 9,
        }
    _name = f"{_base}_diff"
    FEATURE_MANIFEST[_name] = {
        "description": f"Home minus away {_label} quality",
        "definition": f"{_base}_home - {_base}_away",
        "source": _source,
        "lookback": _lookback,
        "aggregation": _aggregation,
        "point_in_time_rule": _pit,
        "missing_value_policy": _mvp + "; in-model handling",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree", "mlp"],
        "feature_version": 9,
    }

# RFE promotions: candidates structurally promoted into the served contract
# (config.MONEYLINE_FEATURE_COLS). Their manifest entries move OUT of the
# candidate manifest (the trial space excludes served names), so re-home the
# generated entries here with candidate=False — one documentation source,
# generated, never re-listed.
_RFE_PROMOTED = [
    "pbp_air_yards_att_ewm_diff",
    "pbp_air_yards_att_ewm_home",
    "pbp_air_yards_att_ewm_away",
    # Promoted structurally over an RFE decline — see the config comment on the
    # same names for the full rationale and the measured deltas.
    "pbp_def_epa_play_ewm_diff",
    "pbp_def_epa_play_ewm_home",
    "pbp_def_epa_play_ewm_away",
]
for _name in _RFE_PROMOTED:
    _entry = CANDIDATE_MANIFEST.pop(_name, None)
    if _entry is None:
        raise RuntimeError(f"promoted feature {_name!r} has no generated manifest entry")
    _entry["candidate"] = False
    _entry["feature_version"] = 4
    FEATURE_MANIFEST[_name] = _entry


# The shared monitor page builds its hover text as
# ``<span title='{html.escape(tooltip, quote=False)}'>``. With quote=False the
# escaper leaves BOTH quote characters alone, so a single apostrophe in the
# tooltip terminates that attribute and the rest of the tooltip is parsed as
# markup. Every one of the 46 manifest definitions contains at least one
# ("a team's first-ever game"), so this is not hypothetical. The page is
# presentation-only and must stay that way, so the escaping happens HERE, at
# the one place the string is produced.
_ATTR_UNSAFE = {"'": "’"}


def _attr_safe(text: str) -> str:
    """Make a string safe inside a single-quoted HTML attribute."""
    for bad, good in _ATTR_UNSAFE.items():
        text = text.replace(bad, good)
    return text


def format_tooltip(entry: dict) -> str:
    """Plain-text tooltip body for one FEATURE_MANIFEST entry.

    MLB parity (``feature_metadata.format_tooltip``): the shared monitor page
    renders a drift/coverage row's hover text from a PRE-FORMATTED ``tooltip``
    string, because the frontend is presentation-only and must not own feature
    semantics. The manifest is already the one place those semantics live, so
    the tooltip is formatted here rather than re-derived downstream.

    Field names follow the manifest's own vocabulary (``lookback`` /
    ``aggregation`` / ``point_in_time_rule``), NOT MLB's ``window`` / ``units``
    / ``direction``: those describe MLB columns the NFL manifest has no
    equivalent of, and emitting them as "—" would be noise.
    """
    def _get(key: str) -> str:
        val = str(entry.get(key, "") or "").strip()
        return val or "—"

    return _attr_safe(
        f"What: {_get('description')}\n"
        f"Definition: {_get('definition')}\n"
        f"Source: {_get('source')}\n"
        f"Window: {_get('lookback')} · Built as: {_get('aggregation')}\n"
        f"Point-in-time rule: {_get('point_in_time_rule')}\n"
        f"Missing values: {_get('missing_value_policy')}\n"
        f"Available to: {_get('model_family_availability')}"
    )


def feature_tooltips(names: list[str] | None = None) -> dict:
    """``{feature: {tooltip, ...}}`` for the monitor artifact.

    Only features with a real manifest entry get an entry; an undocumented
    feature is simply absent, which is what the page needs to tell apart
    "no metadata" from "metadata exists".
    """
    out: dict = {}
    for name in (names if names is not None else FEATURE_MANIFEST):
        entry = FEATURE_MANIFEST.get(name)
        if not entry:
            continue
        out[name] = {"tooltip": format_tooltip(entry),
                     "description": entry.get("description"),
                     "definition": entry.get("definition"),
                     "source": entry.get("source"),
                     "lookback": entry.get("lookback")}
    return out


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
    # Triable-but-unserved names live in CANDIDATE_MANIFEST; structurally
    # promoted names live in FEATURE_MANIFEST and leave the trial space.
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
