"""The NBA sources, declared against the contract, and their translations.

``source_contract.py`` owns the vocabulary; this module owns the translations.
Each source is a ``SourceSpec`` whose adapters map one upstream's shape onto
the contract, so a change to an upstream's column names is a change to one
dict in one function rather than an edit that reaches into the feature code.

The division of labour is the one MLB already runs, which is the reason it is
worth copying rather than inventing:

``ESPN scoreboard`` owns the SCHEDULE
    Which games exist, on what date, between which teams, and whether they are
    final. It is the only one of the three that can answer for a game nobody
    has played yet, which is what a pending slate *is*. ``stats.nba.com`` has
    no future games to report and ``cdn.nba.com`` neither.

``stats.nba.com LeagueGameLog`` owns the FEATURES
    One request per season and season type returns every player line of that
    season - 26,306 rows for 2024-25. No per-game API reaches the same data in
    fewer requests, because per-game APIs are one request per game.

``stats.nba.com playbyplayv3`` owns the PLAY-BY-PLAY
    One request per game, ~0.1s, and the response is ``{game: {actions: [...]}}``
    rather than the ``resultSets`` envelope every other ``/stats/`` endpoint
    uses. It carries 23 fields, ``actionId`` present on every action, and
    ``StartPeriod``/``EndPeriod`` are mandatory: omitting them is an HTTP 500,
    not an empty result.

Identity is ESPN's. ``game_id`` is the ESPN event id, and stats.nba.com's own
game id rides alongside in ``nba_game_id`` because that is the key the
play-by-play endpoint is addressed by. The two are joined on
``(gameday, home_team, away_team)`` - a team plays at most once a day, so that
triple is unique within a league, and it is the only join key both upstreams
agree on.

Note on parameters, because it is the reason the query dicts below are built
explicitly and pinned by tests: ``leaguedashplayerstats`` answers HTTP 500 when
``Weight`` carries a value it does not recognise, and answers 200 with the full
69-column result set when the parameter is omitted. An endpoint that reports a
bad enum as a server error rather than a client error is exactly why this file
exists in this shape.
"""
from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Any

import numpy as np
import pandas as pd

import config
import source_contract as contract
from source_contract import ContractError, SourceSpec, make_source

logger = logging.getLogger(__name__)

#: Eastern time. ESPN's ``dates`` parameter is a UTC calendar key, so a game is
#: grouped under the UTC date it tipped off on; an 8pm ET tipoff therefore
#: arrives under the next UTC day. Deciding the game day in UTC would move
#: every late game onto the wrong date and, worse, onto a date the season-log
#: join then fails to match.
_EASTERN = "America/New_York"

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

ESPN_SCOREBOARD_URL = ("https://site.api.espn.com/apis/site/v2/sports/"
                       "basketball/nba/scoreboard")
SEASON_LOG_URL = "https://stats.nba.com/stats/LeagueGameLog"
PLAY_BY_PLAY_URL = "https://stats.nba.com/stats/playbyplayv3"

# stats.nba.com validates its parameter set strictly and reports a mismatch as
# a 500. These are the parameters ``LeagueGameLog`` is known to accept,
# expressed once so a test can pin them and a reader can see the contract
# without guessing.
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

SEASON_TYPE_REGULAR = "Regular Season"
SEASON_TYPE_PLAYOFFS = "Playoffs"

# ESPN's ``season.type`` codes. They are 2 and 3, and the league writes them
# that way deliberately - they are not an offset of anything - so they are
# mapped explicitly instead of being offset into our own game types.
ESPN_SEASON_REGULAR = 2
ESPN_SEASON_POST = 3
ESPN_GAME_TYPE = {ESPN_SEASON_REGULAR: 1, ESPN_SEASON_POST: 2}

# A shot's ``distance`` is measured in feet from the rim. The three cut points
# are the league's own shot-chart bands, chosen here so a team can be split
# into rim / mid-range / corner without consulting a vendor's opinion.
RIM_FEET = 5.0
MID_FEET = 22.0


def season_log_query(season: str, season_type: str) -> str:
    """``LeagueGameLog``'s query string, shared by the pull and any probe."""
    return urllib.parse.urlencode({
        **_STATS_QUERY, "Season": season, "SeasonType": season_type,
        "DateFrom": "", "DateTo": "",
    })


def play_by_play_query(game_id: str) -> str:
    """``playbyplayv3``'s query string.

    ``StartPeriod``/``EndPeriod`` are not optional. ``playbyplayv3?GameID=...``
    with no period range answers HTTP 500, and a wrapper that forgets them
    therefore looks like an upstream outage rather than a missing argument.
    Period 0 to 14 is the whole game, including any overtime.
    """
    return urllib.parse.urlencode({
        "GameID": str(game_id), "StartPeriod": 0, "EndPeriod": 14,
    })


def espn_scoreboard_url(when) -> str:
    return (f"{ESPN_SCOREBOARD_URL}?dates={when:%Y%m%d}&limit=400")


# ---------------------------------------------------------------------------
# ESPN scoreboard -> the games frame
# ---------------------------------------------------------------------------


def _parse_espn_event(event: dict, game_type: int | None) -> dict | None:
    """One ESPN event into one contract games row, or ``None`` if unusable.

    The two decisions worth reading are both about not inventing a game:

    * A score is kept only when the event is ``post`` AND completed. ESPN
      publishes a ``'0'`` score for a scheduled game, so a naive read of the
      score field would turn tonight's slate into 0-0 results and train on
      them. ``status.type.completed`` is the league's own final flag.
    * The date is converted to Eastern before the day is decided, because
      ESPN's ``dates`` parameter is a UTC calendar key and a 10pm ET tipoff
      lands on the following UTC day. Filtering on the raw timestamp would
      move every late game onto the wrong day.
    """
    comp = (event.get("competitions") or [{}])[0]
    competitors = comp.get("competitors") or []
    if len(competitors) != 2:
        return None
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home or not away:
        return None
    event_id = str(event.get("id") or "").strip()
    if not event_id:
        return None

    status = (comp.get("status") or {}).get("type") or {}
    is_final = bool(status.get("completed")) or str(status.get("state")) == "post"
    if is_final:
        detail = str(status.get("detail") or "").lower()
        if any(word in detail for word in ("postpon", "cancel", "suspend",
                                           "delay", "reschedul")):
            is_final = False

    def _score(competitor: dict) -> float:
        if not is_final:
            return np.nan
        text = str(competitor.get("score") or "").strip()
        try:
            return float(text)
        except ValueError:
            return np.nan

    home_score = _score(home)
    away_score = _score(away)
    if is_final and home_score == 0.0 and away_score == 0.0:
        # ESPN files a postponed or abandoned game as a completed 0-0. No NBA
        # game has ever finished 0-0, so this is always a game nobody played.
        # Accepting it would put a scoreless tie into the training set, hand
        # both teams a zero net rating, and add a phantom settled game to every
        # trailing window - all of it invisible, because the row looks settled.
        is_final = False
        home_score = away_score = np.nan

    teams = {}
    for side, competitor in (("home", home), ("away", away)):
        team = competitor.get("team") or {}
        abbr = (team.get("abbreviation") or team.get("shortDisplayName")
                or team.get("name") or "").strip()
        teams[side] = config.normalize_team_abbr(abbr)
    if not teams["home"] or not teams["away"]:
        return None

    raw_date = event.get("date") or comp.get("date")
    if not raw_date:
        return None
    gameday = pd.to_datetime(raw_date, errors="coerce", utc=True)
    if pd.isna(gameday):
        return None

    return {
        "game_id": event_id,
        "gameday": gameday.tz_convert(_EASTERN).tz_localize(None).normalize(),
        "home_team": teams["home"],
        "away_team": teams["away"],
        "home_score": home_score,
        "away_score": away_score,
        "game_type": game_type if game_type is not None else np.nan,
        "is_final": is_final,
        "espn_name": event.get("name") or "",
    }


def _to_eastern(value):
    """Eastern time as a tz-aware ``Timestamp``; None if unparseable."""
    stamp = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(stamp) else stamp.tz_convert(_EASTERN)


def _is_franchise_game(row: dict) -> bool:
    """Whether both sides of a parsed event are one of the 30 franchises.

    The season-type check below cannot catch the all-star event, because ESPN
    files it as ``regular-season``: the league has no other code for it. What
    gives it away is the participants. A league game is played by two
    franchises, and these are not franchises - they are the all-star squads,
    which ESPN abbreviates STARS, STRIPES and WORLD. The season log agrees and
    is the proof: it returns zero player lines for every one of them, so such
    an event can never contribute a feature, a win, or a point.

    The 2026 all-star made this a correctness problem rather than a tidiness
    one. It was a four-game tournament on a single date, and two of those games
    were the same two teams, so the orientation-free date/team-pair key that
    joins the schedule to the season log - which is unique precisely because a
    team plays at most once a day - was not unique. That raised
    ``MergeError: Merge keys are not unique in left dataset`` and took the
    whole pipeline down over four games nobody can train on.
    """
    for side in ("home_team", "away_team"):
        if config.team_category_id(row.get(side)) == config.UNK_TEAM_ID:
            return False
    return True


def espn_schedule_frame(day_events: list[dict]) -> pd.DataFrame:
    """A list of raw ESPN scoreboard events into contract games rows.

    The event's own ``season.type`` decides the game type, and a date whose
    season type is neither regular season nor post-season is dropped: the
    model's windows are defined over those two.

    An event between two non-franchises is dropped as well, which is the only
    thing that catches the all-star event. Both drops happen here rather than
    in the eligibility filter so that these games never enter the frames.
    """
    rows = []
    for event in day_events or []:
        if not isinstance(event, dict):
            continue
        espn_type = (event.get("season") or {}).get("type")
        game_type = ESPN_GAME_TYPE.get(espn_type)
        if game_type is None:
            # A game whose season type is neither regular season nor
            # post-season is not one the model can score. The all-star event
            # is NOT caught here - it arrives as ``regular-season`` - which is
            # what ``_is_franchise_game`` below is for.
            continue
        row = _parse_espn_event(event, game_type)
        if row is None:
            continue
        if not _is_franchise_game(row):
            logger.warning(
                "dropping %s on %s (%s vs %s): a side is not one of the 30 "
                "franchises, and the season log has no player lines for it, "
                "so it is not a game this model can score",
                row.get("game_id"), row.get("gameday"),
                row.get("home_team"), row.get("away_team"))
            continue
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if frame.duplicated("game_id").any():
        # The same event legitimately arrives from two UTC probes when a late
        # game straddles midnight, so the second copy is a duplicate to drop,
        # never a second game.
        frame = frame.drop_duplicates("game_id", keep="first")
    return frame


# ---------------------------------------------------------------------------
# LeagueGameLog -> player lines and the team facts derived from them
# ---------------------------------------------------------------------------

_LOG_RENAME = {
    "GAME_ID": "nba_game_id", "GAME_DATE": "gameday", "MATCHUP": "matchup",
    "TEAM_ABBREVIATION": "team", "TEAM_ID": "team_id",
    "PLAYER_ID": "player_id", "PLAYER_NAME": "player_name",
    "MIN": "minutes", "WL": "win", "PLUS_MINUS": "plus_minus",
    "FG_PCT": "fg_pct", "FG3_PCT": "three_point_pct",
    "FT_PCT": "free_throw_pct",
    # The counting columns. Every name here is one the team rollup sums, and
    # one that must be renamed before it can be summed - the log says FGA and
    # the contract says fga, and a rollup that sums "FGA" would produce a
    # column the contract does not declare and the ladder never reads.
    "PTS": "points", "FGM": "fgm", "FGA": "fga", "FG3M": "fg3m",
    "FG3A": "fg3a", "FTM": "ftm", "FTA": "fta", "OREB": "oreb",
    "DREB": "dreb", "REB": "reb", "AST": "ast", "TOV": "tov",
    "STL": "stl", "BLK": "blk", "PF": "pf",
}


def rename_log(log: pd.DataFrame) -> pd.DataFrame:
    """``LeagueGameLog`` headers to contract names, leaving the rest alone."""
    out = log.rename(columns=_LOG_RENAME).copy()
    if "game_type" not in out.columns:
        out["game_type"] = np.nan
    # Both upstreams spell six of the thirty teams differently, so both are
    # folded to the model's own abbreviation here. The schedule and the season
    # log are joined on team names; if one side is left un-normalized the join
    # quietly loses every game involving a renamed team.
    if "team" in out.columns:
        out["team"] = out.team.map(config.normalize_team_abbr)
    return out


def matchup_teams(matchup: Any) -> tuple[str, str] | None:
    """The two team abbreviations in a matchup string, in written order.

    The ORDER is deliberately not interpreted. The log writes the matchup from
    each row's own team's perspective, and it is not even self-consistent: for
    most games the home team's rows use ``vs.`` and the visitors' use ``@``,
    which the connector alone would decode correctly - but 5 games in the
    2024-25 log use ``@`` on both sides, so ``WAS @ MIA`` appears on WAS's rows
    even though WAS is home. Any rule that reads order out of this string
    therefore inverts the sides of some games, silently, and a schedule with
    swapped venues still looks like a schedule.

    So the string contributes an unordered pair, and the schedule supplies the
    orientation. ``matchup_sides`` below is kept for the one caller that wants
    a best-effort reading, and is not used to key anything.
    """
    parts = str(matchup or "").replace(".", "").split()
    if len(parts) != 3 or parts[1].lower() not in {"vs", "@"}:
        return None
    return (config.normalize_team_abbr(parts[0]),
            config.normalize_team_abbr(parts[2]))


def matchup_sides(matchup: Any) -> tuple[str, str] | None:
    """Best-effort (away, home) from a matchup string.

    Correct for the 1,225 of 1,230 games the log spells consistently, and
    wrong for the handful that use ``@`` on both sides. Use this only where a
    guess is better than nothing; key joins on ``pair_key`` instead.
    """
    teams = matchup_teams(matchup)
    if teams is None:
        return None
    parts = str(matchup or "").replace(".", "").split()
    if parts[1].lower() == "@":
        return teams[0], teams[1]
    return teams[1], teams[0]


#: Columns the log carries per player line. These are summed per team per
#: game to make the box-score-equivalent team facts, so a column missing from
#: the log is a column the team frame cannot fill and the coverage report is
#: where that shows up.
STAT_SUMS: tuple[str, ...] = (
    "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb",
    "ast", "tov", "stl", "blk", "pf", "points", "minutes",
)


def team_stats_from_log(log: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one season's player lines into one row per team per game.

    ``games`` supplies sides, because the log only names the team the player
    was on: home/away lives in the matchup string, and the *result* lives only
    in the schedule. So this is the join point where stats.nba.com's features
    and ESPN's schedule meet, and it is done on ``nba_game_id`` because that
    is the only identity the log carries.
    """
    counted = [c for c in STAT_SUMS if c in log.columns]
    if log.empty or games.empty or not counted:
        return pd.DataFrame()
    out = (log.groupby(["nba_game_id", "team"], as_index=False)[counted]
           .sum(min_count=1)
           .rename(columns={"points": "points_for"}))
    context = games.dropna(subset=["nba_game_id"]) if "nba_game_id" in games else games
    if context.empty:
        return pd.DataFrame()
    # The schedule's own ``game_id`` is the contract's id and the log's is not,
    # so the context is projected onto exactly what a team row needs. Merging
    # the whole frame would bring a second ``game_id`` in behind the one the
    # log carries and the later rename would collide with it.
    context = context[["nba_game_id", "game_id", "gameday", "home_team",
                       "away_team", "home_score", "away_score"]]
    out = out.merge(context, on="nba_game_id", how="inner")
    # Which side this team was on is the join itself: the log named the team,
    # the schedule named the sides, and the two meeting is what says "home".
    out["is_home"] = out.team.astype(str) == out.home_team.astype(str)
    out["opponent"] = np.where(out.is_home, out.away_team, out.home_team)
    out["points_against"] = np.where(out.is_home, out.away_score, out.home_score)
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
    # The contract's id is ESPN's, and the context already supplied it, so the
    # log's own id is dropped rather than renamed over it.
    return out.drop(columns=["nba_game_id"], errors="ignore")


def pair_key(gameday: Any, first: Any, second: Any) -> str:
    """A join key for a game that does not depend on which side is which.

    Sorting the two abbreviations makes ``A@B`` and ``B@A`` produce the same
    key, which is what allows the two upstreams to be matched on WHO played
    rather than on WHO WAS HOME - and the schedule, which is reliable about
    that, then supplies the orientation.
    """
    day = pd.to_datetime(gameday, errors="coerce")
    stamp = "" if pd.isna(day) else f"{day:%Y-%m-%d}"
    teams = sorted([str(first or ""), str(second or "")])
    return f"{stamp}|{teams[0]}|{teams[1]}"


def games_from_log(log: pd.DataFrame) -> pd.DataFrame:
    """Which games the log says were played, keyed without an orientation.

    Returns ``nba_game_id``, ``gameday``, ``pair_key`` and the two teams in
    written order. The schedule supplies home/away; this only has to say that
    a game happened, on what date, between whom, under which NBA id.

    The two-team check is enforced rather than assumed: a game whose rows name
    more than one pair of teams is not a game this function can key, and
    guessing which pair is right would attach a box score to the wrong game.
    """
    if log.empty or not {"nba_game_id", "matchup", "gameday"}.issubset(log.columns):
        return pd.DataFrame()
    teams = log["matchup"].map(matchup_teams)
    out = log.assign(_a=teams.map(lambda t: t[0] if t else None),
                     _b=teams.map(lambda t: t[1] if t else None))
    out = out.dropna(subset=["_a", "_b"])
    out = out[out._a != out._b]
    per_game = out.groupby("nba_game_id")[["_a", "_b"]].nunique().max(axis=1)
    broken = per_game[per_game > 2]
    if len(broken):
        raise ValueError(
            f"{len(broken)} game(s) in the season log name more than two teams, "
            "so they cannot be keyed to a schedule game")
    out = (out.groupby("nba_game_id", as_index=False)
           .agg(gameday=("gameday", "first"), _a=("_a", "first"),
                _b=("_b", "first")))
    out["pair_key"] = [pair_key(g, a, b)
                       for g, a, b in zip(out.gameday, out._a, out._b)]
    out = out.rename(columns={"_a": "team_a", "_b": "team_b"})
    return out[["nba_game_id", "gameday", "team_a", "team_b", "pair_key"]]


# ---------------------------------------------------------------------------
# playbyplayv3 -> the action list, and the team rollup counted from it
# ---------------------------------------------------------------------------

#: The action types stats.nba.com actually emits, verified against live
#: payloads rather than assumed. The notable absences are assists, steals and
#: blocks: this feed has no action for them, so no feature may be built from
#: them, and an assist appears only inside a made shot's free-text
#: ``description``. Anything derived from this frame is limited to what is
#: listed here.
ACTION_TYPES = ("Made Shot", "Missed Shot", "Free Throw", "Rebound",
                "Turnover", "Foul", "Substitution", "Timeout", "period",
                "Jump Ball", "Instant Replay", "Violation")

#: Turnover sub-types that cost a live possession. The rest (traveling,
#: offensive fouls, shot-clock violations) are treated separately because
#: they do not exchange the ball the same way.
LIVE_TURNOVER_SUBTYPES = ("Bad Pass", "Lost Ball", "Backcourt Turnover",
                          "Traveling")

#: A free throw that was made. The feed does not set ``shotResult`` on a free
#: throw - it is empty on every one of them - so a make is only identifiable
#: from the text, which opens with ``MISS`` for a miss and has no such prefix
#: for a make. The trailing ``(N PTS)`` cannot be used for this: the number is
#: the points so far in the trip, not on the shot, so the second free throw of
#: a made pair reads ``(2 PTS)`` and a naive match on ``(1 PTS)`` counts three
#: made free throws in a game that had twenty-two.
FT_MISSED_PATTERN = re.compile(r"^\s*MISS\b", re.IGNORECASE)

#: The feed reports a rebound's side in its own text - ``Murray REBOUND
#: (Off:1 Def:0)`` - and its sub-type says only ``Unknown`` or ``Normal
#: Rebound``, which is no help at all. The two numbers are the REBOUNDING
#: PLAYER's running totals, not the count on this rebound and not the team's:
#: a player's third defensive board reads ``Def:3``, and a different teammate
#: starts again from their own zero. So a team total is the sum of each
#: player's maximum, never the team maximum - which for this game gives an
#: offensive-rebound count of 6 against a true 19, and no error anywhere to
#: say so. A row with no counts - ``Hawks Rebound`` - is a rebound credited to
#: the team rather than a player, which the box score's player-line sum
#: excludes.
REBOUND_COUNTS_PATTERN = re.compile(r"Off:(\d+)\s*Def:(\d+)", re.IGNORECASE)

#: An assist is named in a made shot's text - ``(Sabonis 1 AST)`` - and the
#: action's own ``personId`` is the *shooter*, so assists can be counted per
#: team but not attributed to a player id from this feed.
ASSIST_PATTERN = re.compile(r"\(\s*[A-Za-z'.\- ]+\s+\d+\s+AST\s*\)",
                            re.IGNORECASE)

#: Foul sub-types that count against a player in the box score. ``Loose Ball``
#: is in this list because it does: a real game here charged 12 team fouls
#: against Boston with 11 of them personal, shooting or offensive, and the one
#: loose-ball foul is the difference between matching the box and being short
#: by one. Technicals, double technicals and flagrants are excluded - they
#: appear in no player's PF.
PLAYER_FOUL_SUBTYPES = ("Personal", "Shooting", "Offensive",
                        "Offensive Charge", "Loose Ball")


def play_by_play_actions(payload: Any, nba_game_id: str, gameday: Any) -> pd.DataFrame:
    """Flatten one game's action list into contract rows.

    ``playbyplayv3`` does not return the ``resultSets`` envelope that every
    other ``/stats/`` endpoint returns; it returns ``{game: {actions: [...]}}``,
    the same shape cdn.nba.com serves. A client that looks for ``resultSets``
    finds none and reports zero actions for a game that returned 569 of them,
    which is the failure this function's docstring exists to prevent.

    Unlike cdn.nba.com, this endpoint *does* populate ``actionId`` on every
    action, so the cache key below is the upstream's own and needs no
    substitute.

    The row is keyed by ``nba_game_id`` - the id the request was addressed by -
    and not by the contract's ``game_id``, because at this point the schedule
    has not been consulted and the ESPN id is not yet known. The ingestion
    joins the schedule on this id and renames it, which is why the contract's
    ``game_id`` is absent here rather than guessed.
    """
    actions = ((payload or {}).get("game") or {}).get("actions") or []
    if not actions:
        return pd.DataFrame()
    fields = ("actionId", "actionNumber", "period", "clock", "teamTricode",
              "teamId", "personId", "playerName", "actionType", "subType",
              "description", "location", "shotDistance", "shotResult",
              "isFieldGoal", "shotValue", "pointsTotal", "scoreHome",
              "scoreAway", "xLegacy", "yLegacy")
    frame = pd.DataFrame([{**{k: a.get(k) for k in fields},
                           "nba_game_id": nba_game_id, "gameday": gameday}
                          for a in actions])
    return frame.rename(columns={
        "actionId": "action_id", "actionNumber": "action_number",
        "teamTricode": "team", "personId": "player_id",
        "playerName": "player_name", "actionType": "action_type",
        "subType": "sub_type", "shotDistance": "shot_distance",
        "shotResult": "shot_result", "isFieldGoal": "is_field_goal",
        "shotValue": "shot_value", "pointsTotal": "points",
        "scoreHome": "score_home", "scoreAway": "score_away",
        "xLegacy": "x", "yLegacy": "y",
    })


def _and_one_flags(acts: pd.DataFrame) -> pd.Series:
    """1.0 on each shooting foul that produced a single free throw.

    An and-one is the one foul in basketball that is followed by exactly one
    free throw, so "shooting foul, then a 1-of-1 free throw" identifies it
    without trusting the order of the shot itself. The obvious alternative -
    look back from the foul for the made basket - does not work on this feed:
    its action list is not strictly causal, and a real game here shows a
    shooting foul recorded *after* the rebound that preceded it, so a
    look-back window misses the make it is supposed to find.

    The scan runs over the game's own action order, NOT grouped by team: a
    shooting foul is charged to the team defending and the free throw to the
    team attacking, so within any one team's actions the pair is never
    adjacent and grouping by team finds no and-ones at all. The flag is put on
    the free throw, so the count lands on the team that drew the foul.
    """
    ordered = acts.sort_values(["nba_game_id", "_seq"], kind="stable")
    types = ordered["action_type"].to_numpy()
    subs = ordered["sub_type"].astype(str).to_numpy()
    flags = np.zeros(len(ordered), dtype=float)
    for i in range(len(ordered) - 1):
        if types[i] != "Foul" or str(subs[i]).strip() != "Shooting":
            continue
        if types[i + 1] == "Free Throw" and "1 of 1" in str(subs[i + 1]):
            flags[i + 1] = 1.0
    out = pd.Series(0.0, index=acts.index, dtype=float)
    out.loc[ordered.index] = flags
    return out


def unattributed_rebounds(actions: pd.DataFrame) -> int:
    """How many rebounds in these actions the feed attributes to no side.

    The feed writes a team rebound as ``Hawks Rebound`` with an empty player
    and an empty team, so there is nothing to key it on and nothing to
    aggregate it into. Reporting the count keeps the gap between the event
    stream's board totals and the box score's visible instead of letting the
    two quietly disagree.
    """
    if actions is None or actions.empty or "action_type" not in actions:
        return 0
    kind = actions["action_type"].astype(str)
    team = actions.get("team", pd.Series("", index=actions.index)).fillna("").astype(str)
    return int((kind.eq("Rebound") & team.str.len().eq(0)).sum())


def cross_check(events: pd.DataFrame, team_stats: pd.DataFrame) -> dict:
    """Compare the play-by-play rollup against the box score, column by column.

    Two independent counts of the same game that agree are evidence the pull is
    right. This is advisory, not a gate: the box score is a sum of player lines
    and the event stream is a narration, so small residual differences are
    expected. A column that disagrees on most games is a defect; one that
    disagrees by a rebound on a few is not, and refusing to run over it would
    be a worse mistake than recording it.
    """
    if events is None or team_stats is None or events.empty or team_stats.empty:
        return {}
    pairs = {"fga": "fga", "fgm": "fgm", "fg3a": "fg3a", "fg3m": "fg3m",
             "fta": "fta", "ftm": "ftm", "oreb": "oreb", "dreb": "dreb",
             "tov": "tov", "pf": "pf", "assists": "ast", "points": "points_for"}
    left = events.copy()
    left["_key"] = left.game_id.astype(str) + "|" + left.team.astype(str)
    right = team_stats.copy()
    right["_key"] = right.game_id.astype(str) + "|" + right.team.astype(str)
    # The right frame's columns are suffixed on the merge. Reading the box side
    # by its unsuffixed name would look up the EVENT column instead and compare
    # the rollup against itself - which agrees perfectly, always, and says
    # nothing at all.
    merged = left.merge(right, on="_key", how="inner", suffixes=("", "_box"))
    if merged.empty:
        return {}
    report = {}
    for event_col, box_col in pairs.items():
        box_column = f"{box_col}_box" if f"{box_col}_box" in merged else box_col
        if event_col not in merged or box_column not in merged:
            continue
        a = pd.to_numeric(merged[event_col], errors="coerce")
        b = pd.to_numeric(merged[box_column], errors="coerce")
        both = a.notna() & b.notna()
        if not both.any():
            continue
        diff = (a[both] - b[both]).abs()
        report[event_col] = {
            "compared": int(both.sum()),
            "exact": int((diff == 0).sum()),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": round(float(diff.mean()), 4),
        }
    return report


def _rebound_sides(description: pd.Series) -> tuple[np.ndarray, np.ndarray,
                                                   np.ndarray]:
    """(offensive, defensive, team-flag) read out of each rebound's text.

    The feed writes ``Murray REBOUND (Off:1 Def:0)`` for a player rebound and a
    bare ``Hawks Rebound`` for a team rebound. The two numbers are that
    player's running totals, returned unmodified; the caller takes a
    per-player maximum and then sums - see ``REBOUND_COUNTS_PATTERN``.
    """
    text = description.fillna("").astype(str)
    pairs = text.str.extract(REBOUND_COUNTS_PATTERN)
    offensive = pd.to_numeric(pairs[0], errors="coerce")
    defensive = pd.to_numeric(pairs[1], errors="coerce")
    # Whether a row had counts at all must be decided BEFORE filling: filling
    # first makes notna() true everywhere and every rebound reads as a player
    # rebound, which is how the team-rebound count silently stayed at zero.
    is_player_rebound = offensive.notna() | defensive.notna()
    team = (~is_player_rebound).astype(float)
    return (offensive.to_numpy(float), defensive.to_numpy(float),
            team.to_numpy(float))


def team_events_from_actions(actions: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Count one game's action list into one row per team per game.

    Everything here is counted, not inferred. Two derivations are worth
    naming:

    * **Possessions** use the standard ``FGA + 0.44*FTA - OREB + TOV`` estimate.
      The event stream has no possession counter, so this is an estimate - but
      it is computed on the same basis for every team and every game, which
      makes the error a level shift in pace rather than a per-game distortion.
    * **And-ones** come from the ordered action list, not from the description
      text, because the wording varies and a missed and-one is invisible.

    The rollup deliberately re-counts the box score's own columns (``fga``,
    ``fgm``, ``fta``, ``ftm``, ``oreb``, ``dreb``, ``tov``, ``pf``). Two
    independent counts of the same game that agree are evidence the pull is
    right; two that disagree mean one of them is wrong, which is the only
    cross-check available and is checked in ``ingestion``.
    """
    if actions is None or actions.empty or games is None or games.empty:
        return pd.DataFrame()
    if "nba_game_id" in games.columns:
        games = games.dropna(subset=["nba_game_id"])
    if games.empty:
        return pd.DataFrame()

    acts = actions.copy()
    acts["action_type"] = acts.get("action_type", pd.Series("", index=acts.index)).astype(str)
    acts["nba_game_id"] = acts.get("nba_game_id", pd.Series("", index=acts.index)).astype(str)
    acts["sub_type"] = acts.get("sub_type", pd.Series("", index=acts.index)).fillna("").astype(str)
    acts["team"] = acts.get("team", pd.Series("", index=acts.index)).fillna("").astype(str)
    acts["_order"] = pd.to_numeric(acts.get("action_number"), errors="coerce").fillna(0.0)
    acts = acts[acts.team.str.len() > 0].copy()
    if acts.empty:
        return pd.DataFrame()
    # ``action_number`` is not the list's own order - it runs 2..788 across 569
    # actions - so anything that depends on narrative sequence (which foul
    # precedes which free throw) has to sort on the position the feed listed
    # the action in, captured before any filtering renumbered anything.
    acts = acts.reset_index(drop=True)
    acts["_seq"] = np.arange(len(acts), dtype=float)

    kind = acts.action_type
    sub = acts.sub_type.str.lower()
    desc = acts.get("description", pd.Series("", index=acts.index)).fillna("").astype(str).str.lower()
    result = acts.get("shot_result", pd.Series("", index=acts.index)).fillna("").astype(str)
    period = pd.to_numeric(acts.get("period"), errors="coerce")
    dist = pd.to_numeric(acts.get("shot_distance"), errors="coerce")
    value = pd.to_numeric(acts.get("shot_value"), errors="coerce")

    is_shot = kind.isin(["Made Shot", "Missed Shot"])
    is_made = kind.eq("Made Shot")
    is_three = value.eq(3)
    is_ft = kind.eq("Free Throw")
    is_reb = kind.eq("Rebound")
    is_tov = kind.eq("Turnover")
    is_foul = kind.eq("Foul")
    # A free throw's make is marked only in the text, as the absence of a
    # leading "MISS". ``shotResult`` is empty on every free throw this feed
    # returns, so matching "Made" against it counts zero made free throws for
    # a whole game while looking like a well-formed column.
    is_ft_made = is_ft & ~acts.get(
        "description", pd.Series("", index=acts.index)).fillna("").astype(
            str).str.contains(FT_MISSED_PATTERN)
    # The assister is named in a made shot's text; the action's own personId
    # is the shooter, so assists are counted for the team and never attributed
    # to a player id from this feed.
    is_assist = is_made & acts.get("description", pd.Series(
        "", index=acts.index)).fillna("").astype(str).str.contains(ASSIST_PATTERN)

    oreb, dreb, team_reb = _rebound_sides(
        acts.get("description", pd.Series("", index=acts.index)))
    # These three are numpy arrays in ``acts`` row order. They must stay
    # arrays: wrapping them in ``pd.Series`` gives them a fresh RangeIndex
    # while ``acts`` carries the index it kept from filtering out teamless
    # actions, and the two then align to different rows - which lands the
    # rebound counts on the wrong teams with no error anywhere.
    reb_mask = is_reb.to_numpy()
    is_oreb = reb_mask & (oreb > 0)
    is_dreb = reb_mask & (dreb > 0)
    is_team_reb = reb_mask & (team_reb > 0)
    is_shooting_foul = is_foul & sub.str.strip().str.lower().eq("shooting")
    is_personal_foul = is_foul & sub.str.strip().str.lower().eq("personal")
    is_live_tov = is_tov & acts.sub_type.isin(LIVE_TURNOVER_SUBTYPES)
    is_player_foul = is_foul & acts.sub_type.isin(PLAYER_FOUL_SUBTYPES)

    rim = is_shot & (dist <= RIM_FEET)
    mid = is_shot & (dist > RIM_FEET) & (dist <= MID_FEET)
    deep = is_shot & (dist > MID_FEET)

    # Points on the play. ``pointsTotal`` on this endpoint is the RUNNING team
    # total, so using it here would give a team's last basket a 40-point row;
    # the point value of the play is recomputed from the shot type instead.
    play_points = is_made.astype(float) * np.where(is_three, 3.0, 2.0)
    play_points = play_points + is_ft_made.astype(float)

    acts["_and_in"] = _and_one_flags(acts)
    for period_no in (1, 2, 3, 4):
        acts[f"_q{period_no}_points"] = np.where(period == period_no,
                                                 play_points, 0.0)
    # Overtime is period 5 and up. Without this column the quarters sum to
    # less than the game total for every game that went to overtime, which is
    # roughly one in twelve and would read as a scoring bug.
    acts["_ot_points"] = np.where(period > 4, play_points, 0.0)

    # Rebound sides are per-player running counters, so a team's total is the
    # sum of each of its players' maxima. A player with no name - a rebound
    # the feed attributes to the team - collapses into one bucket, whose
    # maximum is that bucket's own running total.
    player = acts.get("player_name", pd.Series("", index=acts.index)).fillna("").astype(str)
    acts["_reb_player"] = player
    acts["_reb_off"] = np.where(reb_mask, np.nan_to_num(oreb), np.nan)
    acts["_reb_def"] = np.where(reb_mask, np.nan_to_num(dreb), np.nan)
    acts["_reb_team"] = np.where(reb_mask, team_reb, 0.0)

    counts = {
        "fga": is_shot.astype(float), "fgm": is_made.astype(float),
        "fg3a": (is_shot & is_three).astype(float),
        "fg3m": (is_made & is_three).astype(float),
        "fta": is_ft.astype(float), "ftm": is_ft_made.astype(float),
        "tov": is_tov.astype(float), "live_turnovers": is_live_tov.astype(float),
        "pf": is_player_foul.astype(float), "assists": is_assist.astype(float),
        "shooting_fouls": is_shooting_foul.astype(float),
        "personal_fouls": is_personal_foul.astype(float),
        "and_in": acts["_and_in"].astype(float),
        "rim_attempts": rim.astype(float), "mid_attempts": mid.astype(float),
        "corner_three_attempts": deep.astype(float),
        "points": play_points.astype(float),
    }
    # Each count becomes a column before the groupby, so the rollup is a plain
    # sum over named columns rather than a bespoke loop per team.
    for name, values in counts.items():
        acts[name] = np.asarray(values, dtype=float)
    events = (acts.groupby(["nba_game_id", "team"], sort=False)[list(counts)]
              .sum().reset_index())
    groups = acts.groupby(["nba_game_id", "team"], sort=False)
    events["shot_distance"] = (acts.assign(_d=dist.where(dist > 0))
                                   .groupby(["nba_game_id", "team"], sort=False)["_d"]
                                   .mean().reset_index()["_d"])
    for q in (1, 2, 3, 4):
        events[f"q{q}_points"] = groups[f"_q{q}_points"].sum().reset_index()[f"_q{q}_points"]
    events["ot_points"] = groups["_ot_points"].sum().reset_index()["_ot_points"]
    # Max per player, then sum per team. Team rebounds carry no team in the
    # feed, so they were already dropped by the team filter and are counted per
    # game by ``unattributed_rebounds`` rather than being forced onto a side.
    per_player = (acts.groupby(["nba_game_id", "team", "_reb_player"], sort=False)
                  .agg(_off=("_reb_off", "max"), _def=("_reb_def", "max"))
                  .reset_index())
    events = events.merge(
        per_player.groupby(["nba_game_id", "team"], sort=False)
        [["_off", "_def"]].sum().reset_index(),
        on=["nba_game_id", "team"], how="left")
    events = events.rename(columns={"_off": "oreb", "_def": "dreb"})
    for col in ("oreb", "dreb"):
        events[col] = events[col].fillna(0.0)
    # Possessions is the standard estimate, which needs the corrected
    # offensive-rebound count: a triangular rebound total would subtract far
    # more possessions than the game has.
    events["possessions"] = (events.fga + 0.44 * events.fta - events.oreb
                             + events.tov)

    # Attach sides from the schedule and re-key onto the contract's game id.
    # The join is on ``nba_game_id`` because the play-by-play was fetched by
    # that id; the contract's ``game_id`` is ESPN's.
    context = games[["nba_game_id", "game_id", "gameday", "home_team",
                     "away_team", "home_score", "away_score"]]
    events = events.merge(context, on="nba_game_id", how="inner",
                          suffixes=("", "_sched"))
    events["is_home"] = events.team.astype(str) == events.home_team.astype(str)
    events["opponent"] = np.where(events.is_home, events.away_team,
                                  events.home_team)
    return events.drop(columns=[c for c in events.columns
                                if c.endswith("_sched")])


# ---------------------------------------------------------------------------
# Declared sources
# ---------------------------------------------------------------------------

ESPN_SCHEDULE = make_source(
    name="espn scoreboard",
    route="espn scoreboard",
    adapters={"games": espn_schedule_frame},
    frames=("games",),
)

SEASON_LOG = make_source(
    name="stats.nba.com LeagueGameLog",
    route="stats.nba.com LeagueGameLog",
    adapters={
        "player_stats": rename_log,
        "team_stats": lambda f: f,   # needs the games context; the pull does it
    },
    frames=("games", "team_stats", "player_stats"),
)

PLAY_BY_PLAY = make_source(
    name="stats.nba.com playbyplayv3",
    route="stats.nba.com playbyplayv3",
    adapters={
        "play_by_play": lambda f: f,
        "team_events": lambda f: f,
    },
    frames=("play_by_play", "team_events"),
)

#: The pull order. The schedule comes first because everything else is keyed
#: on it: a feature row with no schedule row has no game to attach to and no
#: sides to aggregate.
ALL_SOURCES = (ESPN_SCHEDULE, SEASON_LOG, PLAY_BY_PLAY)

#: Which source answers which frame. Declared rather than inferred, so a
#: reader can see the division of labour without tracing the pull.
FRAME_SOURCE = {
    "games": ESPN_SCHEDULE,
    "team_stats": SEASON_LOG,
    "player_stats": SEASON_LOG,
    "play_by_play": PLAY_BY_PLAY,
    "team_events": PLAY_BY_PLAY,
}


def source_by_name(name: str) -> SourceSpec:
    for source in ALL_SOURCES:
        if source.name == name:
            return source
    raise ContractError(f"no source named {name!r}; known: "
                        f"{[s.name for s in ALL_SOURCES]}")


__all__ = [
    "ESPN_SCHEDULE", "SEASON_LOG", "PLAY_BY_PLAY", "ALL_SOURCES",
    "FRAME_SOURCE", "ESPN_SCOREBOARD_URL", "SEASON_LOG_URL", "PLAY_BY_PLAY_URL",
    "SEASON_TYPE_REGULAR", "SEASON_TYPE_PLAYOFFS", "ESPN_SEASON_REGULAR",
    "ESPN_SEASON_POST", "ESPN_GAME_TYPE", "STAT_SUMS", "ACTION_TYPES",
    "LIVE_TURNOVER_SUBTYPES", "PLAYER_FOUL_SUBTYPES", "RIM_FEET", "MID_FEET",
    "season_log_query", "play_by_play_query", "espn_scoreboard_url",
    "espn_schedule_frame", "rename_log", "matchup_sides", "matchup_teams",
    "pair_key", "team_stats_from_log",
    "games_from_log", "play_by_play_actions", "team_events_from_actions",
    "unattributed_rebounds", "cross_check", "source_by_name",
]
