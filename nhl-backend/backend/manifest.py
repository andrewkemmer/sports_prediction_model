"""Authoritative NHL feature manifest — the documentation of the one list.

Structural mirror of the NFL manifest: every production feature is documented
here (description, definition, source, lookback, aggregation, point-in-time
rule, missing-value policy, representation, version).
``config.MONEYLINE_FEATURE_COLS`` is the served contract and the single
source of truth; this manifest is its documentation (validated name-for-name
by ``validate()``), never a parallel list.

``representation`` carries the member routing that ``config.RAW_PER_SIDE_COLS``
encodes: raw home/away levels reach the tree members only, while diffs,
flags and the anchor reach every family.

Feature-set version: see config.FEATURE_SET_VERSION.
"""
from __future__ import annotations

# Candidate-pool documentation (mirrors the NFL side table consumed at
# admission time by the RFE workbook and config.assert_candidate_manifest
# _parity).
CANDIDATE_MANIFEST: dict[str, dict] = {}


def _build_candidate_manifest() -> None:
    """Populate CANDIDATE_MANIFEST from the config specs — name-for-name with
    config.NHL_CANDIDATE_COLS."""
    try:
        from backend import config as _c
    except ImportError:  # running as a top-level module
        import config as _c

    metric_doc = {
        "pp_goals_pg": (
            "Trailing power-play goals per game",
            "Per-team trailing mean of power-play goals scored",
            "special teams"),
        "pp_attempts_pg": (
            "Trailing power-play opportunities per game",
            "Per-team trailing mean of power-play chances drawn",
            "special teams"),
        "pim_pg": (
            "Trailing penalty minutes per game",
            "Per-team rolling mean of penalty minutes (discipline)",
            "discipline"),
        "hits_pg": (
            "Trailing hits per game",
            "Per-team trailing mean of recorded hits (physicality)",
            "physicality"),
        "blocked_shots_pg": (
            "Trailing blocked shots per game",
            "Per-team trailing mean of blocked shots (defensive commitment)",
            "defense"),
        "giveaways_pg": (
            "Trailing giveaways per game",
            "Per-team trailing mean of giveaways (puck management)",
            "puck management"),
        "takeaways_pg": (
            "Trailing takeaways per game",
            "Per-team trailing mean of takeaways (puck management)",
            "puck management"),
        "goal_diff_pg": (
            "Trailing goal differential per game",
            "Per-team trailing mean of goals for minus goals against (raw margin)",
            "scoring"),
        "sog_pg": (
            "Trailing shots on goal per game",
            "Per-team trailing mean of shots on goal (shot volume)",
            "volume"),
        "faceoff_win_pct": (
            "Trailing faceoff win rate",
            "Per-team trailing mean share of faceoffs won (possession starts)",
            "possession"),
        "pp_success_rate": (
            "Trailing power-play conversion rate",
            "Per-team trailing mean of power-play goals / opportunities (PP efficiency)",
            "special teams"),
    }
    family_source = {
        "nhl": "official NHL API boxscores (per-game rollup)",
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
                    mvp = ("NaN when the team has no prior games or the boxscore "
                           "source column is absent for a season); in-model handling")
                    CANDIDATE_MANIFEST[f"{base}_diff"] = {
                        "description": f"Home minus away {desc.lower()}",
                        "definition": f"{base}_home - {base}_away — {definition}",
                        "source": family_source[_family],
                        "lookback": lookback,
                        "aggregation": "per-team trailing mean of the per-game metric",
                        "point_in_time_rule": pit,
                        "missing_value_policy": mvp,
                        "representation": "difference (all model families)",
                        "model_family_availability": ["linear", "tree"],
                        "feature_version": 1,
                        "candidate": True,
                    }
                    for side, rep_name, fams in (("home", "raw home level", ["tree"]),
                                                 ("away", "raw away level", ["tree"])):
                        CANDIDATE_MANIFEST[f"{base}_{side}"] = {
                            "description": f"{side.capitalize()} team's {desc.lower()}",
                            "definition": f"The {side} team's own {definition}",
                            "source": family_source[_family],
                            "lookback": lookback,
                            "aggregation": "per-team trailing mean of the per-game metric",
                            "point_in_time_rule": pit,
                            "missing_value_policy": mvp,
                            "representation": f"{rep_name} (tree members)",
                            "model_family_availability": fams,
                            "feature_version": 1,
                            "candidate": True,
                        }


_build_candidate_manifest()

# One entry per served feature. Field order mirrors the NFL manifest.
FEATURE_MANIFEST = {
    "elo_diff": {
        "description": "Home minus away pre-game Elo rating",
        "definition": "elo_home_entering - elo_away_entering; Elo update "
                      "r += K*(actual - expected), expected = 1/(1+10**((r_opp + HOME_ADV - r_self)/400)); "
                      "actual = 1 win / 0 loss / 0.5 tie; 1/3 revert toward ELO_PRIOR at each season boundary",
        "source": "official NHL API scores (all decided games, 2024+)",
        "lookback": "full history (iterative)",
        "aggregation": "iterative state update",
        "point_in_time_rule": "rating entering puck drop; updated only AFTER a game settles",
        "missing_value_policy": "ELO_PRIOR (1500) for a team's first-ever game; never NaN",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree"],
        "feature_version": 1,
    },
    "win_pct_diff": {
        "description": "Home minus away trailing win percentage",
        "definition": "mean(team_win) over the team's prior 12 games (no ties in the "
                      "shootout era; OT/SO decided games award the W)",
        "source": "decided game outcomes",
        "lookback": 12,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(12).mean().shift(1) — current and future games excluded",
        "missing_value_policy": "NaN when the team has no prior games; in-model handling",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree"],
        "feature_version": 1,
    },
    "rest_days_diff": {
        "description": "Home minus away days since each team's previous game",
        "definition": "(game_date_t - game_date_{t-1}).days per team",
        "source": "official NHL API schedule",
        "lookback": 1,
        "aggregation": "date difference",
        "point_in_time_rule": "days since the team's own strictly-prior game",
        "missing_value_policy": "NaN for a team's first game of the window; in-model handling",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree"],
        "feature_version": 1,
    },
    "ewm_net_goals_diff": {
        "description": "Home minus away exponentially-weighted net goals per game",
        "definition": "ewm(halflife=3).mean() of the team's net goals (for - against) over strictly-prior games",
        "source": "decided game scores",
        "lookback": "decaying (halflife=3 games)",
        "aggregation": "per-team EWM",
        "point_in_time_rule": "per-team ewm over prior games then shift(1)",
        "missing_value_policy": "NaN when the team has no prior games",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree"],
        "feature_version": 1,
    },
    "ewm_goal_share_diff": {
        "description": "Home minus away exponentially-weighted goal share",
        "definition": "ewm(halflife=3).mean() of goals_for / (goals_for + goals_against) over "
                      "strictly-prior games — possession-neutral scoring strength in [0, 1]",
        "source": "decided game scores",
        "lookback": "decaying (halflife=3 games)",
        "aggregation": "per-team EWM of the per-game goal share",
        "point_in_time_rule": "per-team ewm over prior games then shift(1)",
        "missing_value_policy": "NaN when the team has no prior games",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree"],
        "feature_version": 1,
    },
    "ga_per_game_diff": {
        "description": "Home minus away trailing goals allowed per game",
        "definition": "rolling(5).mean() of goals against over strictly-prior games "
                      "(lower = stingier defense; the diff is home − away)",
        "source": "decided game scores",
        "lookback": 5,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(5).mean().shift(1)",
        "missing_value_policy": "NaN when the team has no prior games",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree"],
        "feature_version": 1,
    },
    "shots_for_per_game_diff": {
        "description": "Home minus away trailing shots on goal per game",
        "definition": "rolling(5).mean() of team shots on goal over strictly-prior games",
        "source": "official NHL API boxscores (team SOG)",
        "lookback": 5,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(5).mean().shift(1)",
        "missing_value_policy": "NaN when the team has no prior games with SOG recorded",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree"],
        "feature_version": 1,
    },
    "shots_against_per_game_diff": {
        "description": "Home minus away trailing shots against per game",
        "definition": "rolling(5).mean() of opponent shots on goal over strictly-prior games "
                      "(lower = better defensive structure)",
        "source": "official NHL API boxscores (team SOG)",
        "lookback": 5,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(5).mean().shift(1)",
        "missing_value_policy": "NaN when the team has no prior games with SOG recorded",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree"],
        "feature_version": 1,
    },
    "pp_success_diff": {
        "description": "Home minus away trailing power-play success rate",
        "definition": "rolling(5).mean() of power-play goals / power-play opportunities "
                      "over strictly-prior games",
        "source": "official NHL API boxscores (powerPlayGoals)",
        "lookback": 5,
        "aggregation": "trailing windowed mean of a per-game rate",
        "point_in_time_rule": "per-team rolling(5).mean().shift(1)",
        "missing_value_policy": "NaN when the team has no prior games; a game with zero PP "
                                "opportunities is excluded from that game's rate",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree"],
        "feature_version": 1,
    },
    "faceoff_win_diff": {
        "description": "Home minus away trailing faceoff win rate",
        "definition": "rolling(5).mean() of faceoffWinningPctg over strictly-prior games",
        "source": "official NHL API boxscores (faceoffWinningPctg)",
        "lookback": 5,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(5).mean().shift(1)",
        "missing_value_policy": "NaN when the team has no prior games with faceoff data",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree"],
        "feature_version": 1,
    },
    "back_to_back_diff": {
        "description": "Home minus away back-to-back flag (rest < 1 day)",
        "definition": "1.0 when the team played the previous calendar day else 0.0; "
                      "home flag minus away flag",
        "source": "official NHL API schedule",
        "lookback": 1,
        "aggregation": "thresholded date difference",
        "point_in_time_rule": "function of the team's strictly-prior game date",
        "missing_value_policy": "NaN when either team has no prior game",
        "representation": "difference flag (all model families)",
        "model_family_availability": ["linear", "tree"],
        "feature_version": 1,
    },
    "goalie_sv_pct_diff": {
        "description": "Home minus away expected-starter rolling save percentage",
        "definition": "season-to-date (rolling-30-start EWM) save percentage of each team's "
                      "expected starting goalie; home − away. The NHL analog of MLB's SP ERA diff",
        "source": "official NHL API boxscores (goalie TOI + decision + shots faced)",
        "lookback": "decaying (halflife=3 starts) over the season's starts",
        "aggregation": "per-goalie EWM of the per-start save fraction",
        "point_in_time_rule": "each goalie's strictly-prior starts only; the expected starter is "
                              "the team's most-recent-game goalie with the most season starts",
        "missing_value_policy": "NaN when the team has no goalie with a prior start this season "
                                "(e.g. opening-night unknown starter); never fabricated",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree"],
        "feature_version": 1,
    },
    "goalie_gaa_diff": {
        "description": "Home minus away expected-starter rolling goals-against average",
        "definition": "season-to-date (rolling-30-start EWM) GAA of each team's expected "
                      "starting goalie; home − away. The NHL analog of MLB's SP K/9 diff",
        "source": "official NHL API boxscores (goalie goals against + TOI)",
        "lookback": "decaying (halflife=3 starts) over the season's starts",
        "aggregation": "per-goalie EWM of the per-start goals-against rate (per 60 min)",
        "point_in_time_rule": "each goalie's strictly-prior starts only",
        "missing_value_policy": "NaN when no goalie has a prior start; never fabricated",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree"],
        "feature_version": 1,
    },
    "goalie_starts_diff": {
        "description": "Home minus away expected-starter season starts",
        "definition": "season-to-date starts of each team's expected starting goalie — a "
                      "workload/experience proxy for the SV%/GAA pair",
        "source": "official NHL API boxscores (goalie appearances)",
        "lookback": "season to date",
        "aggregation": "per-goalie start count",
        "point_in_time_rule": "strictly-prior starts only",
        "missing_value_policy": "NaN when no goalie has a prior start",
        "representation": "difference (all model families)",
        "model_family_availability": ["linear", "tree"],
        "feature_version": 1,
    },
    "is_playoffs": {
        "description": "Postseason game flag",
        "definition": "1.0 when the game's gameType is 3 (playoffs) else 0.0",
        "source": "official NHL API gameType",
        "lookback": 0,
        "aggregation": "static pre-game fact",
        "point_in_time_rule": "known before puck drop",
        "missing_value_policy": "0.0 when gameType is absent (defaults to regular season)",
        "representation": "game-level flag (all model families)",
        "model_family_availability": ["linear", "tree"],
        "feature_version": 1,
    },
    "is_home": {
        "description": "Constant home anchor",
        "definition": "1.0 for every row — anchors the league home-ice advantage level",
        "source": "frame construction",
        "lookback": 0,
        "aggregation": "constant",
        "point_in_time_rule": "static",
        "missing_value_policy": "never NaN",
        "representation": "constant anchor (all model families)",
        "model_family_availability": ["linear", "tree"],
        "feature_version": 1,
    },
    # ---- raw per-side levels (tree members only; config.RAW_PER_SIDE_COLS) ---
    # Each pair is the level half of the corresponding served difference. They
    # are declared in the served contract itself (never synthesized by a view)
    # so the tree matrix is a projection of the one list.
    "elo_home": {
        "description": "Home team's pre-game Elo rating",
        "definition": "team Elo entering puck drop (home side of elo_diff)",
        "source": "official NHL API scores (all decided games, 2024+)",
        "lookback": "full history (iterative)",
        "aggregation": "iterative state update",
        "point_in_time_rule": "rating entering puck drop; updated only AFTER a game settles",
        "missing_value_policy": "ELO_PRIOR (1500) for a team's first-ever game; never NaN",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "elo_away": {
        "description": "Away team's pre-game Elo rating",
        "definition": "team Elo entering puck drop (away side of elo_diff)",
        "source": "official NHL API scores (all decided games, 2024+)",
        "lookback": "full history (iterative)",
        "aggregation": "iterative state update",
        "point_in_time_rule": "rating entering puck drop; updated only AFTER a game settles",
        "missing_value_policy": "ELO_PRIOR (1500) for a team's first-ever game; never NaN",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "win_pct_home": {
        "description": "Home team's trailing win percentage",
        "definition": "mean(team_win) over the home team's prior 12 games (no ties in the "
                      "shootout era; OT/SO decided games award the W)",
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
        "definition": "mean(team_win) over the away team's prior 12 games (no ties in the "
                      "shootout era; OT/SO decided games award the W)",
        "source": "decided game outcomes",
        "lookback": 12,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(12).mean().shift(1) — current and future games excluded",
        "missing_value_policy": "NaN when the team has no prior games; in-model handling",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "ewm_net_goals_home": {
        "description": "Home team's exponentially-weighted net goals per game",
        "definition": "ewm(halflife=3).mean() of the team's net goals (for - against) over strictly-prior games",
        "source": "decided game scores",
        "lookback": "decaying (halflife=3 games)",
        "aggregation": "per-team EWM",
        "point_in_time_rule": "per-team ewm over prior games then shift(1)",
        "missing_value_policy": "NaN when the team has no prior games",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "ewm_net_goals_away": {
        "description": "Away team's exponentially-weighted net goals per game",
        "definition": "ewm(halflife=3).mean() of the team's net goals (for - against) over strictly-prior games",
        "source": "decided game scores",
        "lookback": "decaying (halflife=3 games)",
        "aggregation": "per-team EWM",
        "point_in_time_rule": "per-team ewm over prior games then shift(1)",
        "missing_value_policy": "NaN when the team has no prior games",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "rest_days_home": {
        "description": "Home team's days since its previous game",
        "definition": "(game_date_t - game_date_{t-1}).days for the home team",
        "source": "official NHL API schedule",
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
        "definition": "(game_date_t - game_date_{t-1}).days for the away team",
        "source": "official NHL API schedule",
        "lookback": 1,
        "aggregation": "date difference",
        "point_in_time_rule": "days since the team's own strictly-prior game",
        "missing_value_policy": "NaN for a team's first game of the window; in-model handling",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "goalie_sv_pct_home": {
        "description": "Home team's expected-starter rolling save percentage",
        "definition": "season-to-date (rolling-30-start EWM) save percentage of the home "
                      "team's expected starting goalie (home side of goalie_sv_pct_diff)",
        "source": "official NHL API boxscores (goalie TOI + decision + shots faced)",
        "lookback": "decaying (halflife=3 starts) over the season's starts",
        "aggregation": "per-goalie EWM of the per-start save fraction",
        "point_in_time_rule": "each goalie's strictly-prior starts only",
        "missing_value_policy": "NaN when the team has no goalie with a prior start this season; never fabricated",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "goalie_sv_pct_away": {
        "description": "Away team's expected-starter rolling save percentage",
        "definition": "season-to-date (rolling-30-start EWM) save percentage of the away "
                      "team's expected starting goalie (away side of goalie_sv_pct_diff)",
        "source": "official NHL API boxscores (goalie TOI + decision + shots faced)",
        "lookback": "decaying (halflife=3 starts) over the season's starts",
        "aggregation": "per-goalie EWM of the per-start save fraction",
        "point_in_time_rule": "each goalie's strictly-prior starts only",
        "missing_value_policy": "NaN when the team has no goalie with a prior start this season; never fabricated",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "goalie_gaa_home": {
        "description": "Home team's expected-starter rolling goals-against average",
        "definition": "season-to-date (rolling-30-start EWM) GAA of the home team's "
                      "expected starting goalie (home side of goalie_gaa_diff)",
        "source": "official NHL API boxscores (goalie goals against + TOI)",
        "lookback": "decaying (halflife=3 starts) over the season's starts",
        "aggregation": "per-goalie EWM of the per-start goals-against rate (per 60 min)",
        "point_in_time_rule": "each goalie's strictly-prior starts only",
        "missing_value_policy": "NaN when no goalie has a prior start; never fabricated",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "goalie_gaa_away": {
        "description": "Away team's expected-starter rolling goals-against average",
        "definition": "season-to-date (rolling-30-start EWM) GAA of the away team's "
                      "expected starting goalie (away side of goalie_gaa_diff)",
        "source": "official NHL API boxscores (goalie goals against + TOI)",
        "lookback": "decaying (halflife=3 starts) over the season's starts",
        "aggregation": "per-goalie EWM of the per-start goals-against rate (per 60 min)",
        "point_in_time_rule": "each goalie's strictly-prior starts only",
        "missing_value_policy": "NaN when no goalie has a prior start; never fabricated",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "ewm_goal_share_home": {
        "description": "Home team's exponentially-weighted goal share",
        "definition": "ewm(halflife=3).mean() of goals_for / (goals_for + goals_against) "
                      "over strictly-prior games -- home side of ewm_goal_share_diff",
        "source": "decided game scores",
        "lookback": "decaying (halflife=3 games)",
        "aggregation": "per-team EWM of the per-game goal share",
        "point_in_time_rule": "per-team ewm over prior games then shift(1)",
        "missing_value_policy": "NaN when the team has no prior games",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "ewm_goal_share_away": {
        "description": "Away team's exponentially-weighted goal share",
        "definition": "ewm(halflife=3).mean() of goals_for / (goals_for + goals_against) "
                      "over strictly-prior games -- away side of ewm_goal_share_diff",
        "source": "decided game scores",
        "lookback": "decaying (halflife=3 games)",
        "aggregation": "per-team EWM of the per-game goal share",
        "point_in_time_rule": "per-team ewm over prior games then shift(1)",
        "missing_value_policy": "NaN when the team has no prior games",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "ga_per_game_home": {
        "description": "Home team's trailing goals allowed per game",
        "definition": "rolling(5).mean() of goals against over strictly-prior games "
                      "(lower = stingier defense) -- home side of ga_per_game_diff",
        "source": "decided game scores",
        "lookback": 5,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(5).mean().shift(1)",
        "missing_value_policy": "NaN when the team has no prior games",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "ga_per_game_away": {
        "description": "Away team's trailing goals allowed per game",
        "definition": "rolling(5).mean() of goals against over strictly-prior games "
                      "(lower = stingier defense) -- away side of ga_per_game_diff",
        "source": "decided game scores",
        "lookback": 5,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(5).mean().shift(1)",
        "missing_value_policy": "NaN when the team has no prior games",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "shots_for_per_game_home": {
        "description": "Home team's trailing shots on goal per game",
        "definition": "rolling(5).mean() of team shots on goal over strictly-prior games "
                      "(home side of shots_for_per_game_diff)",
        "source": "official NHL API boxscores (team SOG)",
        "lookback": 5,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(5).mean().shift(1)",
        "missing_value_policy": "NaN when the team has no prior games with SOG recorded",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "shots_for_per_game_away": {
        "description": "Away team's trailing shots on goal per game",
        "definition": "rolling(5).mean() of team shots on goal over strictly-prior games "
                      "(away side of shots_for_per_game_diff)",
        "source": "official NHL API boxscores (team SOG)",
        "lookback": 5,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(5).mean().shift(1)",
        "missing_value_policy": "NaN when the team has no prior games with SOG recorded",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "shots_against_per_game_home": {
        "description": "Home team's trailing shots allowed per game",
        "definition": "rolling(5).mean() of opponent shots on goal over strictly-prior "
                      "games (lower = better defensive structure) -- home side of "
                      "shots_against_per_game_diff",
        "source": "official NHL API boxscores (team SOG)",
        "lookback": 5,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(5).mean().shift(1)",
        "missing_value_policy": "NaN when the team has no prior games with SOG recorded",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "shots_against_per_game_away": {
        "description": "Away team's trailing shots allowed per game",
        "definition": "rolling(5).mean() of opponent shots on goal over strictly-prior "
                      "games (lower = better defensive structure) -- away side of "
                      "shots_against_per_game_diff",
        "source": "official NHL API boxscores (team SOG)",
        "lookback": 5,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(5).mean().shift(1)",
        "missing_value_policy": "NaN when the team has no prior games with SOG recorded",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "pp_success_home": {
        "description": "Home team's trailing power-play success rate",
        "definition": "pooled power-play goals / opportunities over the trailing "
                      "window (volume-weighted) -- home side of pp_success_diff",
        "source": "official NHL API boxscores (powerPlayGoals)",
        "lookback": 5,
        "aggregation": "trailing pooled count ratio",
        "point_in_time_rule": "per-team rolling(5) sums then shift(1) -- a "
                              "zero-opportunity game contributes nothing rather "
                              "than voiding the window",
        "missing_value_policy": "NaN when the team has no power-play opportunities in the window",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "pp_success_away": {
        "description": "Away team's trailing power-play success rate",
        "definition": "pooled power-play goals / opportunities over the trailing "
                      "window (volume-weighted) -- away side of pp_success_diff",
        "source": "official NHL API boxscores (powerPlayGoals)",
        "lookback": 5,
        "aggregation": "trailing pooled count ratio",
        "point_in_time_rule": "per-team rolling(5) sums then shift(1) -- a "
                              "zero-opportunity game contributes nothing rather "
                              "than voiding the window",
        "missing_value_policy": "NaN when the team has no power-play opportunities in the window",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "faceoff_win_home": {
        "description": "Home team's trailing faceoff win rate",
        "definition": "rolling(5).mean() of faceoffWinningPctg over strictly-prior games "
                      "(home side of faceoff_win_diff)",
        "source": "official NHL API boxscores (faceoffWinningPctg)",
        "lookback": 5,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(5).mean().shift(1)",
        "missing_value_policy": "NaN when the team has no prior games with faceoff data",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "faceoff_win_away": {
        "description": "Away team's trailing faceoff win rate",
        "definition": "rolling(5).mean() of faceoffWinningPctg over strictly-prior games "
                      "(away side of faceoff_win_diff)",
        "source": "official NHL API boxscores (faceoffWinningPctg)",
        "lookback": 5,
        "aggregation": "trailing windowed mean",
        "point_in_time_rule": "per-team rolling(5).mean().shift(1)",
        "missing_value_policy": "NaN when the team has no prior games with faceoff data",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "back_to_back_home": {
        "description": "Home team's back-to-back flag (rest < 1 day)",
        "definition": "1.0 when the team played the previous calendar day else 0.0 "
                      "(home side of back_to_back_diff)",
        "source": "official NHL API schedule",
        "lookback": 1,
        "aggregation": "thresholded date difference",
        "point_in_time_rule": "function of the team's strictly-prior game date",
        "missing_value_policy": "NaN for a team's first game of the window",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "back_to_back_away": {
        "description": "Away team's back-to-back flag (rest < 1 day)",
        "definition": "1.0 when the team played the previous calendar day else 0.0 "
                      "(away side of back_to_back_diff)",
        "source": "official NHL API schedule",
        "lookback": 1,
        "aggregation": "thresholded date difference",
        "point_in_time_rule": "function of the team's strictly-prior game date",
        "missing_value_policy": "NaN for a team's first game of the window",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "goalie_starts_home": {
        "description": "Home team's expected-starter season starts",
        "definition": "season-to-date starts of the home team's expected starting "
                      "goalie (home side of goalie_starts_diff) -- a workload/"
                      "experience proxy for the SV%/GAA pair",
        "source": "official NHL API boxscores (goalie appearances)",
        "lookback": "season to date",
        "aggregation": "per-goalie start count",
        "point_in_time_rule": "strictly-prior starts only",
        "missing_value_policy": "NaN when no goalie has a prior start; never fabricated",
        "representation": "raw home level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
    "goalie_starts_away": {
        "description": "Away team's expected-starter season starts",
        "definition": "season-to-date starts of the away team's expected starting "
                      "goalie (away side of goalie_starts_diff) -- a workload/"
                      "experience proxy for the SV%/GAA pair",
        "source": "official NHL API boxscores (goalie appearances)",
        "lookback": "season to date",
        "aggregation": "per-goalie start count",
        "point_in_time_rule": "strictly-prior starts only",
        "missing_value_policy": "NaN when no goalie has a prior start; never fabricated",
        "representation": "raw away level (tree members)",
        "model_family_availability": ["tree"],
        "feature_version": 1,
    },
}

# ---------------------------------------------------------------------------
# Player pool (24 columns)
# ---------------------------------------------------------------------------
# One entry per situation x position x representation, generated from the same
# rules the feature builder uses so the two cannot drift: the manifest is
# documentation that is checked against config.MONEYLINE_FEATURE_COLS by
# manifest.validate(), and a hand-written block of 24 near-identical entries is
# exactly where documentation drifts from behaviour.
_POOL_SITUATION_NAME = {"5on5": "even-strength (EVO)", "5on4": "power-play (PPO)"}
_POOL_POSITION_NAME = {"C": "centres", "L": "left wings", "R": "right wings",
                       "D": "defencemen"}
_POOL_REPRESENTATION = {
    "away": "raw away level (tree members)",
    "home": "raw home level (tree members)",
    "diff": "difference (all model families)",
}

for _sit, _metric in (("5on5", "EVO"), ("5on4", "PPO")):
    for _pos in ("C", "L", "R", "D"):
        for _rep in ("away", "home", "diff"):
            FEATURE_MANIFEST[f"pl_{_metric.lower()}_{_pos.lower()}_{_rep}"] = {
                "description":
                    f"{_POOL_SITUATION_NAME[_sit]} {_POOL_POSITION_NAME[_pos]} "
                    f"healthy-pool mean shrunk rate, {_rep}",
                "definition":
                    f"mean shrunk_rate_per60 over the player's rating for "
                    f"{_sit}/{_pos}, aggregated over the healthy pool -- the "
                    f"most recent trailing-30-game rating per player whose "
                    f"source game date is strictly earlier than the target "
                    f"date, gated at >=900s prior ice time, with any player on "
                    f"an Out/Injured Reserve/IR/Doubtful interval captured "
                    f"before puck drop REMOVED from the pool before the mean. "
                    f"MLB's _LINEUP_AGG_ROSTER shape. No top-N cut: all 18 "
                    f"dressed skaters contribute in hockey. The rating itself "
                    f"is never modified; the flag removes a row, it does not "
                    f"discount a value. diff = home - away.",
                "source": "MoneyPuck regular-season skater game-by-game "
                          "archives (xGoals, icetime) + ESPN timestamped "
                          "injury status snapshots",
                "lookback": "trailing 30 played games per player and situation "
                            "(game grain); source games within "
                            "injury_stints.POOL_LOOKBACK_DAYS = 45 of the "
                            "target date -- a slate whose team ratings all "
                            "trail that gate (the season opener after the "
                            "offseason) serves the position prior, matching "
                            "MLB's LINEUP_POOL_LOOKBACK_DAYS = 10 geometry",
                "aggregation": "opportunity-weighted mean of surviving "
                          "player-level shrunk rates (NFL snap-share parity): "
                          "each candidate's weight is his own served prior "
                          "icetime, applied AFTER the injury exclusion so an "
                          "absent star's ice leaves the blend with him; sides "
                          "with zero served weight fall back to the plain "
                          "mean (counted in the pool audit)",
                "point_in_time_rule":
                    "a candidate rating enters a target game's pool only when "
                    "its source game date is STRICTLY earlier than the target "
                    "date (date-grain source feeds cannot prove within-day "
                    "publish ordering), so a target game's own row can never "
                    "enter its own pool; the injury interval opens only on a "
                    "captured snapshot timestamp strictly before exact puck "
                    "drop (Out / Injured Reserve / IR / Doubtful -- never a "
                    "provider filing or projected return date), and the "
                    "exclusion is evaluated at the game date, per game.",
                "missing_value_policy":
                    "position prior (measured 2023-25 default) when a side has "
                    "no pool row; never NaN, never 0 -- 0 would read as "
                    "deliberately inert",
                "representation": _POOL_REPRESENTATION[_rep],
                "model_family_availability": (
                    ["linear", "tree"] if _rep == "diff" else ["tree"]),
                "feature_version": 2,
            }


# The shared monitor page builds its hover text as
# ``<span title='{html.escape(tooltip, quote=False)}'>``. With quote=False the
# escaper leaves quote characters alone, so a single apostrophe in the
# tooltip terminates that attribute and the rest of the tooltip is parsed as
# markup. The manifest text is full of them ("the player's rating", "a team's
# first-ever game"), so this is not hypothetical. The page is
# presentation-only and must stay that way, so the escaping happens HERE, at
# the one place the string is produced (NFL manifest parity).
_ATTR_UNSAFE = {"'": "’"}


def _attr_safe(text: str) -> str:
    """Make a string safe inside a single-quoted HTML attribute."""
    for bad, good in _ATTR_UNSAFE.items():
        text = text.replace(bad, good)
    return text


def format_tooltip(entry: dict) -> str:
    """Plain-text tooltip body for one FEATURE_MANIFEST entry.

    MLB/NFL parity: the shared monitor page renders a drift/coverage row's
    hover text from a PRE-FORMATTED ``tooltip`` string, because the frontend
    is presentation-only and must not own feature semantics. The manifest is
    already the one place those semantics live, so the tooltip is formatted
    here rather than re-derived downstream.
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
    """Name-for-name parity: FEATURE_MANIFEST <-> the served contract.

    Returns the problem list (empty = parity). The pipeline's validation
    gates and the feature-name consistency test call this.
    """
    try:
        from backend import config as _c
    except ImportError:  # running as a top-level module
        import config as _c
    problems: list[str] = []
    served = list(_c.MONEYLINE_FEATURE_COLS)
    documented = list(FEATURE_MANIFEST)
    for f in served:
        if f not in documented:
            problems.append(f"served feature {f!r} missing from manifest")
    for f in documented:
        if f not in served:
            problems.append(f"manifest entry {f!r} is not a served feature")
    return problems
