"""The sources that feed the NBA pipeline, declared against the contract.

``source_contract.py`` owns the vocabulary; this module owns the translations.
Each source is a ``SourceSpec`` whose adapters map one upstream's shape onto
the contract, so a change to an upstream's column names is a change to one
dict in one function rather than an edit that reaches into the feature code.

Two sources are declared, and the second one exists to prove the abstraction
is real rather than decorative:

``stats.nba.com / LeagueGameLog``
    The backbone. One request per season and season type returns every player
    line of a season, which is why it is worth protecting: no per-game API
    reaches the same data in fewer requests, because per-game APIs are one
    request per game.

``stats.nba.com / LeagueDash*Stats``
    A per-season aggregate in one call, reached through a different endpoint
    and a different parameter set. It cannot replace the season log - it has
    no game identity, so it cannot produce a game-level frame - but it does
    mean the pipeline has a second way to ask the same league about the same
    season, and that is what a fallback route is supposed to be.

Note on parameters, because it is the reason this file exists in this shape:
``leaguedashplayerstats`` answers HTTP 500 when ``Weight`` carries a value it
does not recognise, and answers 200 with the full 69-column result set when the
parameter is omitted. An endpoint that reports a bad enum as a server error
rather than a client error is the reason the parameter list below is built
explicitly and pinned by a test, instead of being whatever a wrapper library
happened to send.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import source_contract as contract
from source_contract import ContractError, SourceSpec, make_source

SEASON_LOG_URL = "https://stats.nba.com/stats/LeagueGameLog"
PLAYER_DASH_URL = "https://stats.nba.com/stats/leaguedashplayerstats"
TEAM_DASH_URL = "https://stats.nba.com/stats/leaguedashteamstats"
BOXSCORE_URL = ("https://cdn.nba.com/static/json/liveData/boxscore/"
                "boxscore_{game_id}.json")
PLAY_BY_PLAY_URL = ("https://cdn.nba.com/static/json/liveData/playbyplay/"
                    "playbyplay_{game_id}.json")

# stats.nba.com validates its parameter set strictly and reports a mismatch as
# a 500. These are the parameters each endpoint is known to accept, expressed
# once so a test can pin them and a reader can see the contract without
# guessing. The `Weight` key is present on the dash endpoints and MUST be empty
# on the player endpoint: sending "Basic" there is a 500, not an empty result.
_STATS_QUERY: dict[str, str] = {
    "LeagueID": "00", "PerMode": "PerGame", "College": "", "Conference": "",
    "Country": "", "Division": "", "DraftPick": "", "DraftYear": "",
    "GameScope": "", "GameSegment": "", "Height": "", "LastNGames": "0",
    "Location": "", "MeasureType": "Base", "Month": "0", "OpponentTeamID": "0",
    "Outcome": "", "PORound": "0", "PaceAdjust": "N", "Period": "0",
    "PlayerExperience": "", "PlayerPosition": "", "PlusMinus": "N",
    "Rank": "N", "SeasonSegment": "", "ShotClockRange": "",
    "StarterBench": "", "TeamID": "0", "VsConference": "", "VsDivision": "",
}


def season_log_query(season: str, season_type: str) -> str:
    """The season log's query string, shared by the pull and any probe."""
    import urllib.parse
    return urllib.parse.urlencode({
        **_STATS_QUERY, "Season": season, "SeasonType": season_type,
        "DateFrom": "", "DateTo": "",
    })


def dash_query(season: str, season_type: str, *, player: bool) -> dict[str, str]:
    """The dash endpoints' parameter set.

    ``Weight`` is included for the team endpoint and left empty for the player
    endpoint, because the player endpoint 500s on a non-empty value. Keeping
    that asymmetry in one place, with a comment, is cheaper than rediscovering
    it from a 500 during a run.
    """
    return {
        **_STATS_QUERY, "Season": season, "SeasonType": season_type,
        "DateFrom": "", "DateTo": "", "Weight": "" if player else "Basic",
    }


def _stat_sum_columns(log: pd.DataFrame) -> list[str]:
    """Counting columns present in this log, for the team aggregate."""
    return [c for c in ("fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb",
                        "dreb", "reb", "ast", "tov", "stl", "blk", "pf",
                        "points") if c in log.columns]


# ---------------------------------------------------------------------------
# Adapters: upstream shape -> contract shape
# ---------------------------------------------------------------------------


def _rename_log(log: pd.DataFrame) -> pd.DataFrame:
    """``LeagueGameLog`` headers to contract names, leaving the rest alone."""
    out = log.copy()
    out = out.rename(columns={
        "GAME_ID": "game_id", "GAME_DATE": "gameday", "MATCHUP": "matchup",
        "TEAM_ABBREVIATION": "team", "TEAM_ID": "team_id",
        "PLAYER_ID": "player_id", "PLAYER_NAME": "player_name",
        "MIN": "minutes", "WL": "win", "PLUS_MINUS": "plus_minus",
        "FG_PCT": "fg_pct", "FG3_PCT": "three_point_pct",
        "FT_PCT": "free_throw_pct",
    })
    # The log reports a single `team` per player line; the contract's games
    # frame needs sides, which only the matchup string carries. A row that
    # arrives without it cannot be placed, so it is dropped here rather than
    # becoming a game with an unknown home side.
    if "game_type" not in out.columns:
        out["game_type"] = np.nan
    return out


def _team_stats_from_log(log: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one season's player lines into one row per team per game."""
    counted = _stat_sum_columns(log)
    if log.empty or games.empty or not counted:
        return pd.DataFrame()
    out = (log.groupby(["game_id", "team"], as_index=False)[counted]
           .sum(min_count=1)
           .rename(columns={"points": "points_for"}))
    context = games[["game_id", "gameday", "home_team", "away_team",
                     "home_score", "away_score", "is_home"]]
    out = out.merge(context, on="game_id", how="inner")
    out["opponent"] = np.where(out.is_home, out.away_team, out.home_team)
    out["points_against"] = np.where(out.is_home, out.away_score,
                                     out.home_score)
    if "dreb" in out and "oreb" in out and "reb" not in out:
        out["reb"] = out.oreb + out.dreb
    for made, att, name in (("fgm", "fga", "fg_pct"),
                            ("fg3m", "fg3a", "three_point_pct"),
                            ("ftm", "fta", "free_throw_pct")):
        if name not in out and {made, att}.issubset(out.columns):
            with np.errstate(divide="ignore", invalid="ignore"):
                out[name] = np.where(out[att] > 0, out[made] / out[att], np.nan)
    if "net_points" not in out:
        out["net_points"] = out.points_for - out.points_against
    if "efg_pct" not in out and {"fgm", "fg3m", "fga"}.issubset(out.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            out["efg_pct"] = np.where(
                out.fga > 0, (out.fgm + 0.5 * out.fg3m) / out.fga, np.nan)
    return out


def _dash_to_team(frame: pd.DataFrame) -> pd.DataFrame:
    """A per-season team aggregate, shaped like the contract's team facts.

    This is a season total, not a game row, so it deliberately produces no
    ``game_id``. It exists to prove the same contract can be filled from a
    different granularity, and to be refused by the game-level gates - which
    is the correct outcome, and is asserted in the tests rather than assumed.
    """
    out = frame.rename(columns={
        "TEAM_ID": "team_id", "TEAM_ABBREVIATION": "team", "TEAM_NAME": "team",
        "GP": "games", "W": "wins", "L": "losses", "W_PCT": "win_pct",
        "PTS": "points_for", "FG_PCT": "fg_pct", "FG3_PCT": "three_point_pct",
        "FT_PCT": "free_throw_pct", "OREB": "oreb", "DREB": "dreb",
        "REB": "reb", "AST": "ast", "TOV": "tov", "STL": "stl", "BLK": "blk",
        "PF": "pf", "MIN": "minutes",
    })
    return out


def _dash_to_player(frame: pd.DataFrame) -> pd.DataFrame:
    """A per-season player aggregate, shaped like the contract's player facts.

    Like the team aggregate this has no game identity, so it cannot be mistaken
    for a game-level row: the absence of ``game_id`` is what stops it, and the
    contract is what makes that absence a required-column error rather than a
    surprise three phases later.
    """
    out = frame.rename(columns={
        "PLAYER_ID": "player_id", "PLAYER_NAME": "player_name",
        "TEAM_ABBREVIATION": "team", "GP": "games", "W_PCT": "win_pct",
        "MIN": "minutes", "PTS": "points", "REB": "reb", "AST": "ast",
        "STL": "stl", "BLK": "blk", "FG_PCT": "fg_pct",
        "FG3_PCT": "three_point_pct", "FT_PCT": "free_throw_pct",
        "OREB": "oreb", "DREB": "dreb", "TOV": "tov", "PF": "pf",
        "FGM": "fgm", "FGA": "fga", "FG3M": "fg3m", "FG3A": "fg3a",
        "FTM": "ftm", "FTA": "fta", "PLUS_MINUS": "plus_minus",
    })
    return out


def _play_by_play_actions(payload: Any, game_id: str, gameday: Any) -> pd.DataFrame:
    """Flatten one game's action list using the upstream's real field names.

    The previous mapping read ``actionId``, ``sequenceNumber``, ``gameClock``,
    ``isFieldGoalAttempted``, ``isMade``, ``loc`` and ``isHundred``.  Not one of
    those names is returned by either NBA.com play-by-play surface, so seven of
    nineteen columns were null in every row and the frame looked populated by
    column count alone.  The names below are the ones the payloads actually
    carry, and the test pins that every contract column is non-null on a real
    action.
    """
    actions = ((payload or {}).get("game") or {}).get("actions") or []
    if not actions:
        return pd.DataFrame()
    fields = ("actionId", "actionNumber", "period", "clock", "teamTricode",
              "personId", "actionType", "subType", "description",
              "scoreHome", "scoreAway", "pointsTotal", "shotDistance",
              "shotResult", "isFieldGoal", "shotValue", "x", "y",
              "xLegacy", "yLegacy")
    records = [{**{k: a.get(k) for k in fields},
                "game_id": game_id, "gameday": gameday} for a in actions]
    frame = pd.DataFrame(records).rename(columns={
        "actionNumber": "action_number",
        "clock": "clock", "teamTricode": "team", "personId": "player_id",
        "actionType": "action_type", "subType": "sub_type",
        "scoreHome": "score_home", "scoreAway": "score_away",
        "pointsTotal": "points", "shotDistance": "shot_distance",
        "shotResult": "shot_result", "isFieldGoal": "is_field_goal",
    })
    # The two surfaces disagree about identity, and the cache's merge key is
    # (game_id, action_id).  cdn.nba.com omits ``actionId`` on every single
    # action, so keying on it alone gives every row in a game the same null
    # key: a 707-action game deduplicated to ONE row, silently, with no error
    # anywhere.  So the contract's identity is derived - the upstream's own
    # ``actionId`` when it has one, otherwise the per-game sequence number,
    # which is present and unique on all 707 of the CDN's rows.
    frame["action_id"] = frame.get("actionId", pd.Series(dtype=object))
    if "actionId" in frame.columns:
        frame["action_id"] = frame["actionId"].where(
            frame["actionId"].notna(), frame["action_number"])
    else:
        frame["action_id"] = frame["action_number"]
    # ``shotValue`` is a stats.nba.com field the CDN lacks, but the CDN's
    # actionType encodes the same fact in its name ("2pt", "3pt", "freethrow").
    # Deriving it here means an event-derived points feature is written once,
    # against one definition, whichever surface answered.
    if "shotValue" in frame.columns and frame["shotValue"].notna().any():
        frame["shot_value"] = frame["shotValue"]
    else:
        values = frame["action_type"].astype(str).str.lower()
        frame["shot_value"] = np.select(
            [values.eq("3pt"), values.eq("2pt"), values.eq("freethrow")],
            [3.0, 2.0, 1.0], default=np.nan)
    return frame


# ---------------------------------------------------------------------------
# Declared sources
# ---------------------------------------------------------------------------

SEASON_LOG = make_source(
    name="stats.nba.com LeagueGameLog",
    route="stats.nba.com LeagueGameLog",
    adapters={
        "player_stats": lambda f: _rename_log(f),
        "team_stats": lambda f: f,  # needs games; handled by the pull
    },
    frames=("games", "team_stats", "player_stats"),
)

PLAYER_DASH = make_source(
    name="stats.nba.com LeagueDashPlayerStats",
    route="stats.nba.com LeagueDashPlayerStats",
    adapters={"player_stats": _dash_to_player, "team_stats": _dash_to_player},
    frames=("player_stats", "team_stats"),
)

TEAM_DASH = make_source(
    name="stats.nba.com LeagueDashTeamStats",
    route="stats.nba.com LeagueDashTeamStats",
    adapters={"team_stats": _dash_to_team},
    frames=("team_stats",),
)

CDN_BOXSCORE = make_source(
    name="cdn.nba.com boxscore",
    route="cdn.nba.com boxscore",
    adapters={"games": lambda f: f, "team_stats": lambda f: f,
              "player_stats": lambda f: f},
    frames=("games", "team_stats", "player_stats"),
)

CDN_PLAY_BY_PLAY = make_source(
    name="cdn.nba.com playbyplay",
    route="cdn.nba.com playbyplay",
    adapters={"play_by_play": lambda f: f},
    frames=("play_by_play",),
)

# The order the pull tries them in.  The season log is first because it is the
# only route that returns a whole season of player lines in one request; the
# dash aggregates are second because they answer for a season without naming a
# game; the CDN is last because it is one request per game and is therefore a
# fallback, not a plan.
SEASON_LEVEL_SOURCES = (SEASON_LOG, PLAYER_DASH, TEAM_DASH)
GAME_LEVEL_SOURCES = (SEASON_LOG, CDN_BOXSCORE)
ALL_SOURCES = SEASON_LEVEL_SOURCES + GAME_LEVEL_SOURCES + (CDN_PLAY_BY_PLAY,)


def source_by_name(name: str) -> SourceSpec:
    for source in ALL_SOURCES:
        if source.name == name:
            return source
    raise ContractError(f"no source named {name!r}; known: "
                        f"{[s.name for s in ALL_SOURCES]}")


__all__ = [
    "SEASON_LOG", "PLAYER_DASH", "TEAM_DASH", "CDN_BOXSCORE",
    "CDN_PLAY_BY_PLAY", "SEASON_LEVEL_SOURCES", "GAME_LEVEL_SOURCES",
    "ALL_SOURCES", "SEASON_LOG_URL", "PLAYER_DASH_URL", "TEAM_DASH_URL",
    "BOXSCORE_URL", "PLAY_BY_PLAY_URL", "season_log_query", "dash_query",
    "source_by_name", "_play_by_play_actions", "_team_stats_from_log",
    "_rename_log",
]
