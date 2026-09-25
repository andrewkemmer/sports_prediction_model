"""Ingest NBA games, box scores, and play-by-play from NBA.com's free APIs.

NBA.com publishes two unauthenticated surfaces, and both are used here.

``stats.nba.com``
    ``LeagueGameLog`` returns a whole season of player game logs in one
    request.  It is the backbone: the game index (id, date, matchup) and every
    player line arrive in the same call, and a team box score is the sum of its
    players' lines.  Regular season and playoffs are separate calls, which is
    also where the game type comes from, so nothing is inferred from dates.

``cdn.nba.com``
    The same JSON NBA.com itself serves: the league schedule, and per-game box
    scores and play-by-play.

No key, quota, or paid tier is involved.  The season log costs ~10 requests for
any window, so it is re-read every run.  Play-by-play is one request per game,
so it is cached to Parquet and only games the cache has not seen are fetched,
which turns a daily run into an increment instead of a rebuild.

Windows and full re-pulls are controlled by environment variables so the same
code serves an incremental daily run and an explicit rebuild:

``NBA_START_DATE`` / ``NBA_END_DATE``
    Inclusive ISO dates bounding the window.  Defaults to the first eligible
    season through today.
``NBA_FULL_REPULL``
    ``1`` ignores every cache and re-fetches the whole window.
``NBA_FETCH_PLAY_BY_PLAY``
    ``0`` skips play-by-play entirely.
``NBA_HTTP_PAUSE_SEC`` / ``NBA_HTTP_RETRIES``
    Politeness pacing and retry budget for the upstream endpoints.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from backend import config
except ImportError:
    import config

logger = logging.getLogger(__name__)

SEASON_LOG_URL = "https://stats.nba.com/stats/LeagueGameLog"
SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
PLAY_BY_PLAY_URL = ("https://cdn.nba.com/static/json/liveData/playbyplay/"
                    "playbyplay_{game_id}.json")

SOURCE_ID = "nba.com"

# NBA.com rejects requests that do not look like the browser its own site
# makes.  Trimming any of these is what turns a 200 into a 403.
_HTTP_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
}
_STATS_HEADERS = {**_HTTP_HEADERS, "Host": "stats.nba.com",
                  "x-nba-stats-origin": "stats", "x-nba-stats-token": "true"}

START_DATE_ENV = "NBA_START_DATE"
END_DATE_ENV = "NBA_END_DATE"
FULL_REPULL_ENV = "NBA_FULL_REPULL"
PLAY_BY_PLAY_ENV = "NBA_FETCH_PLAY_BY_PLAY"
PAUSE_ENV = "NBA_HTTP_PAUSE_SEC"
RETRIES_ENV = "NBA_HTTP_RETRIES"

# Games this recent are re-pulled even when cached: a run that started while a
# game was in progress stores a partial play-by-play and a partial score.
REFRESH_TAIL_DAYS = 3
DEFAULT_PAUSE_SEC = 0.35
DEFAULT_RETRIES = 4

# Per-host request policy.  stats.nba.com answers one query with a whole
# season of player lines and can take a minute; the CDN is fast and its 403 is
# authoritative ("no such game"), so retrying one only wastes the run.
#
# These are ceilings, not targets: a request is abandoned as soon as it
# exceeds its timeout, and the whole run is bounded by PULL_DEADLINE_ENV so a
# struggling upstream degrades into a clear failure instead of a silent hang.
_HOST_POLICY: dict[str, dict[str, Any]] = {
    "stats.nba.com": {"timeout": 60, "attempts": 2, "backoff": 3.0,
                      "retry_forbidden": False},
    "cdn.nba.com": {"timeout": 45, "attempts": 2, "backoff": 0.5,
                    "retry_forbidden": True},
    "default": {"timeout": 45, "attempts": 2, "backoff": 1.0,
                "retry_forbidden": False},
}
# Name resolution does not honour a socket timeout, so every attempt is run on
# a thread and abandoned at a hard wall clock.  Without this a blackholed host
# blocks forever with no CPU and no error.
DNS_GRACE_SEC = 30
PULL_DEADLINE_ENV = "NBA_PULL_DEADLINE_SEC"
DEFAULT_PULL_DEADLINE_SEC = 600.0
# A healthy pull answers all eight season requests in seconds.  When the host
# is actually down, repeating the same doomed request for every remaining
# season buys nothing but another few minutes of the user watching a still run,
# so the pull gives up after this many consecutive failures.
MAX_CONSECUTIVE_FAILURES = 2

# ---------------------------------------------------------------------------
# CDN fallback
#
# stats.nba.com accepts a connection from a blocked cloud range and then never
# replies, which is indistinguishable from a hang at the socket layer.  The
# per-game CDN needs no such cooperation: a game that exists answers 200 and a
# game that does not answers 403, and a season's game ids are contiguous from
# one, so a season can be read one box score at a time with nothing else.
# ---------------------------------------------------------------------------
CDN_FALLBACK_ENV = "NBA_CDN_FALLBACK"
MAX_SEQUENCE_PROBE = 1400
MAX_SEQUENCE_ENV = "NBA_MAX_SEQUENCE_PROBE"
# The regular season runs to early April; the play-in and playoffs follow.  The
# CDN box score carries no season-type flag, so this is the only available
# signal and the classified counts are logged for that reason.
PLAYOFF_START = (4, 15)
PLAYOFF_END = (7, 1)  # exclusive; the next regular season opens in October

_TEAM_STAT_MAP = {
    "fieldGoalsMade": "fgm", "fieldGoalsAttempted": "fga",
    "threePointersMade": "fg3m", "threePointersAttempted": "fg3a",
    "freeThrowsMade": "ftm", "freeThrowsAttempted": "fta",
    "reboundsOffensive": "oreb", "reboundsDefensive": "dreb",
    "reboundsTotal": "reb", "assists": "ast", "steals": "stl",
    "blocks": "blk", "foulsPersonal": "pf",
    "turnoversTotal": "tov", "turnovers": "tov", "points": "points",
}
_PLAYER_STAT_MAP = {
    "fieldGoalsMade": "fgm", "fieldGoalsAttempted": "fga",
    "threePointersMade": "fg3m", "threePointersAttempted": "fg3a",
    "freeThrowsMade": "ftm", "freeThrowsAttempted": "fta",
    "reboundsOffensive": "oreb", "reboundsDefensive": "dreb",
    "reboundsTotal": "reb", "assists": "ast", "steals": "stl",
    "blocks": "blk", "foulsPersonal": "pf", "turnovers": "tov",
    "points": "points", "plusMinusPoints": "plus_minus", "minus": "plus_minus",
}

SEASON_TYPE_REGULAR = "Regular Season"
SEASON_TYPE_PLAYOFFS = "Playoffs"

# Player game-log columns summed into a team box score.
_STAT_SUMS = {
    "points": "PTS", "fgm": "FGM", "fga": "FGA", "fg3m": "FG3M", "fg3a": "FG3A",
    "ftm": "FTM", "fta": "FTA", "oreb": "OREB", "dreb": "DREB", "reb": "REB",
    "ast": "AST", "tov": "TOV", "stl": "STL", "blk": "BLK", "pf": "PF",
}
# ``_STAT_SUMS`` is canonical -> NBA column; a rename map needs the reverse.
_STAT_RENAME = {source: canonical for canonical, source in _STAT_SUMS.items()}
_PLAYER_RENAME = {
    "PLAYER_ID": "player_id", "PLAYER_NAME": "player_name",
    "GAME_ID": "game_id", "GAME_DATE": "gameday",
    "TEAM_ABBREVIATION": "team", "TEAM_NAME": "team_name",
    "MATCHUP": "matchup", "MIN": "minutes",
    "PLUS_MINUS": "plus_minus", "WL": "win",
    **_STAT_RENAME,
}


@dataclass
class Warehouse:
    """The normalized contract the model stack consumes."""

    games: pd.DataFrame
    team_stats: pd.DataFrame
    player_stats: pd.DataFrame
    team_names: dict[str, str]
    manifest: dict[str, Any]
    play_by_play: pd.DataFrame = field(default_factory=pd.DataFrame)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def _flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        logger.warning("%s is not a number; using %s", name, default)
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, "") or default))
    except ValueError:
        logger.warning("%s is not a number; using %s", name, default)
        return default


class SeasonUnavailable(RuntimeError):
    """One season log could not be read; other seasons may still succeed."""


class CdnUnavailable(RuntimeError):
    """The per-game CDN stopped answering; the whole fallback is over."""


def _request_json(url: str, headers: dict[str, str], timeout: int) -> Any:
    """Perform one request under a hard wall clock.

    ``urlopen``'s timeout covers socket reads, not DNS, so a host that never
    resolves would otherwise block the run indefinitely.  Running the call on a
    thread and abandoning it at the deadline makes "hung" indistinguishable
    from "slow" impossible to reach.
    """
    outcome: dict[str, Any] = {}

    def worker() -> None:
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                outcome["payload"] = json.loads(response.read())
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            outcome["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    started = time.monotonic()
    thread.start()
    thread.join(timeout + DNS_GRACE_SEC)
    if thread.is_alive():
        raise TimeoutError(
            f"no response within {timeout + DNS_GRACE_SEC}s "
            f"(socket limit {timeout}s plus DNS)")
    if "error" in outcome:
        raise outcome["error"]
    logger.info("NBA request completed in %.1fs", time.monotonic() - started)
    return outcome["payload"]


def _get_json(url: str, *, headers: dict[str, str] | None = None,
              retries: int | None = None, pause: float | None = None,
              allow_missing: bool = False) -> Any | None:
    """Fetch JSON with per-host timeout, backoff, and retry policy.

    Every attempt is logged before it starts and after it fails, so a slow
    upstream is visibly slow rather than indistinguishable from a hang.  The
    two hosts behave nothing alike: ``stats.nba.com`` serves a whole season of
    player lines from one very slow query, while ``cdn.nba.com`` is a CDN whose
    403 means "this game does not exist" rather than "slow down".
    """
    host = urllib.parse.urlparse(url).netloc
    path = urllib.parse.urlparse(url).path
    policy = _HOST_POLICY.get(host, _HOST_POLICY["default"])
    attempts = retries if retries is not None else _int_env(
        RETRIES_ENV, policy["attempts"])
    base = policy["backoff"] if pause is None else max(pause, 0.0)
    timeout = policy["timeout"]
    last: Exception | None = None
    reason = "unknown"
    made = 0
    for attempt in range(max(attempts, 1)):
        made = attempt + 1
        logger.info("NBA request %s/%s to %s (timeout %ds)",
                    made, max(attempts, 1), host, timeout + DNS_GRACE_SEC)
        try:
            return _request_json(url, headers or _HTTP_HEADERS, timeout)
        except urllib.error.HTTPError as exc:
            last = exc
            reason = f"HTTP {exc.code}"
            if exc.code in (403, 404) and allow_missing:
                return None
            # A 403 from the CDN means the artifact is absent; a 403 from
            # stats means the client was rejected and retrying will not help.
            if exc.code in (403, 404) and not policy["retry_forbidden"]:
                logger.warning("NBA request to %s rejected (HTTP %s); not retrying",
                               host, exc.code)
                break
            if exc.code < 500 and exc.code not in (429, 403):
                break
        except urllib.error.URLError as exc:
            last = exc
            reason = f"{type(exc.reason).__name__}: {exc.reason}"
        except Exception as exc:  # noqa: BLE001 - network layer is untyped
            last = exc
            reason = f"{type(exc).__name__}: {exc}"
        logger.warning("NBA request %s/%s to %s failed: %s", made,
                       max(attempts, 1), host, reason)
        if attempt + 1 < max(attempts, 1):
            # Jitter keeps concurrent clients from retrying in lockstep.
            delay = base * (2 ** attempt)
            time.sleep(delay * (0.5 + random.random()))
    raise RuntimeError(
        f"NBA request to {host} failed after {made} attempt(s) "
        f"({reason}); endpoint={path} :: {last}")

def _season_log_path(season: str, season_type: str) -> Path:
    """Per-season cache file, so one bad season never discards the others."""
    slug = "regular" if season_type == SEASON_TYPE_REGULAR else "playoffs"
    return config.CACHE_DIR / f"season_log_{season}_{slug}.parquet"


def _fetch_season_log(season: str, season_type: str,
                      pause: float) -> pd.DataFrame:
    """One request per season and season type, cached on disk as it lands.

    The cache is the point: seasons arrive one at a time, so writing each as
    soon as it is read means a timeout on the fifth season costs one request
    next run rather than the whole pull.
    """
    path = _season_log_path(season, season_type)
    if path.exists() and not _flag(FULL_REPULL_ENV, False):
        try:
            cached = pd.read_parquet(path)
            if not cached.empty:
                logger.info("%s %s: %d lines from cache", season, season_type,
                            len(cached))
                return cached
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read %s (%s); re-fetching", path.name, exc)

    query = urllib.parse.urlencode({
        "LeagueID": "00", "PerMode": "PerGame", "Season": season,
        "SeasonType": season_type, "College": "", "Conference": "",
        "Country": "", "DateFrom": "", "DateTo": "", "Division": "",
        "DraftPick": "", "DraftYear": "", "GameScope": "", "GameSegment": "",
        "Height": "", "LastNGames": "0", "Location": "", "MeasureType": "Base",
        "Month": "0", "OpponentTeamID": "0", "Outcome": "", "PORound": "0",
        "PaceAdjust": "N", "Period": "0", "PlayerExperience": "",
        "PlayerPosition": "", "PlusMinus": "N", "Rank": "N", "SeasonSegment": "",
        "ShotClockRange": "", "StarterBench": "", "TeamID": "0", "VsConference": "",
        "VsDivision": "", "Weight": "",
    })
    time.sleep(pause)
    try:
        payload = _get_json(f"{SEASON_LOG_URL}?{query}", headers=_STATS_HEADERS)
    except Exception as exc:  # noqa: BLE001 - one bad season must not end the run
        raise SeasonUnavailable(f"{season} {season_type}: {exc}") from exc
    if not payload or not payload.get("resultSets"):
        return pd.DataFrame()
    block = payload["resultSets"][0]
    rows = block.get("rowSet") or []
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=block["headers"])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not cache %s (%s)", path.name, exc)
    return frame


def _fetch_schedule(pause: float) -> pd.DataFrame:
    """The current season's schedule, including games not yet played."""
    time.sleep(pause)
    payload = _get_json(SCHEDULE_URL, allow_missing=True)
    if not payload:
        return pd.DataFrame()
    dates = (payload.get("leagueSchedule") or {}).get("gameDates") or []
    records: list[dict[str, Any]] = []
    for day in dates:
        for game in day.get("games") or []:
            home = game.get("homeTeam") or {}
            away = game.get("awayTeam") or {}
            records.append({
                "game_id": game.get("gameId"),
                "gameday": game.get("gameDateTimeUTC") or game.get("gameDateEST"),
                "home_team": home.get("teamTricode"),
                "away_team": away.get("teamTricode"),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "game_status": game.get("gameStatus"),
            })
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Seasons, dates, and teams
# --------------------------------------------------------------------------


def _to_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _window() -> tuple[date, date]:
    """Inclusive window, defaulted to the first eligible season through today."""
    today = date.today()
    start_raw = (os.environ.get(START_DATE_ENV) or "").strip()
    end_raw = (os.environ.get(END_DATE_ENV) or "").strip()
    default_start = date(config.OOF_FIRST_SEASON, 1, 1)
    start = _to_date(start_raw) if start_raw else default_start
    end = _to_date(end_raw) if end_raw else today
    if end < start:
        logger.warning("%s (%s) precedes %s (%s); swapping",
                       END_DATE_ENV, end, START_DATE_ENV, start)
        start, end = end, start
    return start, end


def _season_label(when: date) -> str:
    """NBA season label for a date: July starts a new season."""
    first = when.year if when.month >= 7 else when.year - 1
    return f"{first}-{str(first + 1)[2:]}"


def _seasons_in(start: date, end: date) -> list[str]:
    """Every season label whose span overlaps the window.

    A season runs July through June.  Bounding on the overlap keeps a run from
    asking for a season that has not started, which is both a wasted request
    and a confusing empty result in the log.
    """
    labels: list[str] = []
    for year in range(start.year - 1, end.year + 1):
        first = year
        if date(first, 7, 1) > end or date(first + 1, 6, 30) < start:
            continue
        label = f"{first}-{str(first + 1)[2:]}"
        if label not in labels:
            labels.append(label)
    return labels


def _team(value: Any) -> str:
    return config.normalize_team_abbr(value)


def _matchup_sides(matchup: Any) -> tuple[str, str] | None:
    """(away, home) from NBA's ``AWY vs. HME`` / ``AWY @ HME`` matchup text."""
    parts = str(matchup or "").replace(".", "").split()
    if len(parts) == 3 and parts[1].lower() in {"vs", "@"}:
        return _team(parts[0]), _team(parts[2])
    return None


def _minutes(value: Any) -> float:
    """Minutes as float from ``"36:12"``, ``"PT35M56.00S"``, or a number."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    iso = re.fullmatch(r"PT(?:(\d+)M)?(?:([\d.]+)S)?", text)
    if iso:
        return float(iso.group(1) or 0) + float(iso.group(2) or 0) / 60.0
    clock = re.fullmatch(r"(\d+):(\d{1,2})", text)
    if clock:
        return float(clock.group(1)) + float(clock.group(2)) / 60.0
    try:
        return float(text)
    except ValueError:
        return np.nan


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def _prepare_log(raw: pd.DataFrame, game_type: int) -> pd.DataFrame:
    """Fold a season log into the module's snake_case vocabulary."""
    if raw.empty:
        return pd.DataFrame()
    frame = raw.rename(columns=_PLAYER_RENAME)
    keep = [c for c in dict.fromkeys(_PLAYER_RENAME.values())
            if c in frame.columns]
    frame = frame[keep].copy()
    frame["team"] = frame["team"].map(_team)
    frame["game_id"] = frame["game_id"].astype(str)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="coerce", utc=True)
    frame = frame[frame["gameday"].notna() & frame["team"].ne("")].copy()
    frame["gameday"] = frame["gameday"].dt.tz_convert(None)
    frame["minutes"] = frame["minutes"].map(_minutes)
    for column in set(_STAT_SUMS) | {"plus_minus"}:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["game_type"] = game_type
    if "win" in frame.columns:
        frame["win"] = frame["win"].astype("string").str.strip().str.upper().map(
            {"W": 1.0, "L": 0.0})
    return frame.reset_index(drop=True)


def _games_frame(log: pd.DataFrame) -> pd.DataFrame:
    """One settled row per game: sides, tipoff, score, and season."""
    if log.empty:
        return pd.DataFrame()
    sides = (log[["game_id", "gameday", "matchup", "game_type"]]
             .drop_duplicates("game_id")
             .reset_index(drop=True))
    parsed = sides["matchup"].map(_matchup_sides)
    sides["away_team"] = [p[0] if p else None for p in parsed]
    sides["home_team"] = [p[1] if p else None for p in parsed]
    totals = (log.groupby(["game_id", "team"], as_index=False)["points"]
              .sum(min_count=1))
    scores = {gid: dict(zip(group["team"], group["points"]))
              for gid, group in totals.groupby("game_id")}
    sides["home_score"] = [scores.get(gid, {}).get(home)
                           for gid, home in zip(sides.game_id, sides.home_team)]
    sides["away_score"] = [scores.get(gid, {}).get(away)
                           for gid, away in zip(sides.game_id, sides.away_team)]
    sides = sides.dropna(subset=["home_team", "away_team", "home_score",
                                 "away_score"])
    sides = sides[sides.home_team.ne(sides.away_team)]
    if sides.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "game_id": sides.game_id,
        "gameday": sides.gameday,
        "season": sides.gameday.dt.year.where(sides.gameday.dt.month >= 7,
                                              sides.gameday.dt.year - 1),
        "home_team": sides.home_team,
        "away_team": sides.away_team,
        "home_score": sides.home_score.astype(float),
        "away_score": sides.away_score.astype(float),
        "game_type": sides.game_type,
    })
    out["margin"] = out.home_score - out.away_score
    out["total"] = out.home_score + out.away_score
    out["home_win"] = np.where(out.home_score > out.away_score, 1.0, 0.0)
    return out.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def _attach_team_names(games: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    """Give both sides a display name so serving never falls back to a code."""
    if games.empty:
        return games
    out = games.copy()
    out["home_team_name"] = out.home_team.map(lambda t: names.get(t, t))
    out["away_team_name"] = out.away_team.map(lambda t: names.get(t, t))
    return out


def _team_stats_frame(log: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, aggregated from that team's player lines."""
    if log.empty or games.empty:
        return pd.DataFrame()
    summed = [column for column in _STAT_SUMS if column in log.columns]
    out = (log.groupby(["game_id", "team"], as_index=False)[summed]
           .sum(min_count=1)
           .rename(columns={"points": "points_for"}))
    context = games[["game_id", "gameday", "home_team", "away_team",
                     "home_score", "away_score"]]
    out = out.merge(context, on="game_id", how="inner")
    out["is_home"] = out.team.eq(out.home_team)
    out["opponent"] = np.where(out.is_home, out.away_team, out.home_team)
    out["points_against"] = np.where(out.is_home, out.away_score, out.home_score)
    out["net_points"] = out.points_for - out.points_against
    for made, attempt, name in (("fgm", "fga", "fg_pct"),
                                ("fg3m", "fg3a", "three_point_pct"),
                                ("ftm", "fta", "free_throw_pct")):
        if made in out.columns and attempt in out.columns:
            out[name] = np.where(out[attempt] > 0, out[made] / out[attempt], np.nan)
    return out.drop(columns=[c for c in ("home_team", "away_team",
                                         "home_score", "away_score")
                             if c in out.columns])


def _player_stats_frame(log: pd.DataFrame) -> pd.DataFrame:
    """One row per player per game."""
    if log.empty:
        return pd.DataFrame()
    keep = [c for c in dict.fromkeys(
        ("game_id", "gameday", "player_id", "player_name", "team", "minutes",
         "win", "game_type", "plus_minus", "fg_pct", "three_point_pct",
         "free_throw_pct", *sorted(_STAT_SUMS))) if c in log.columns]
    out = log[keep].copy()
    if {"fgm", "fga"}.issubset(out.columns):
        out["fg_pct"] = np.where(out.fga > 0, out.fgm / out.fga, np.nan)
    if {"fg3m", "fg3a"}.issubset(out.columns):
        out["three_point_pct"] = np.where(out.fg3a > 0, out.fg3m / out.fg3a, np.nan)
    if {"ftm", "fta"}.issubset(out.columns):
        out["free_throw_pct"] = np.where(out.fta > 0, out.ftm / out.fta, np.nan)
    if "reb" in out.columns:
        out["rebounds"] = out["reb"]
    if "ast" in out.columns:
        out["assists"] = out["ast"]
    out = out.drop_duplicates(["game_id", "player_id"], keep="last")
    return out.sort_values(["gameday", "game_id", "player_id"]).reset_index(drop=True)


def _play_by_play_frame(payload: Any, game_id: str,
                        gameday: Any) -> pd.DataFrame:
    """Flatten one game's action list into rows."""
    actions = ((payload or {}).get("game") or {}).get("actions") or []
    if not actions:
        return pd.DataFrame()
    fields = ("actionId", "sequenceNumber", "period", "gameClock", "teamTricode",
              "personId", "actionType", "subType", "description", "descriptor",
              "scoreHome", "scoreAway", "pointsTotal", "shotDistance",
              "shotResult", "isFieldGoalAttempted", "isMade", "loc", "isHundred")
    records = [{**{key: action.get(key) for key in fields},
                "game_id": game_id,
                "gameday": gameday}
               for action in actions]
    frame = pd.DataFrame(records)
    frame = frame.rename(columns={
        "actionId": "action_id", "sequenceNumber": "sequence_number",
        "period": "period", "gameClock": "game_clock",
        "teamTricode": "team", "personId": "player_id",
        "actionType": "action_type", "subType": "sub_type",
        "scoreHome": "score_home",
        "scoreAway": "score_away", "pointsTotal": "points",
        "shotDistance": "shot_distance", "shotResult": "shot_result",
        "isFieldGoalAttempted": "is_field_goal_attempted", "isMade": "is_made",
        "isHundred": "is_hundred",
    })
    frame["team"] = frame["team"].map(_team)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="coerce", utc=True)
    frame["gameday"] = frame["gameday"].dt.tz_convert(None)
    for column in ("period", "sequence_number", "score_home", "score_away",
                   "points", "shot_distance", "action_id"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def _cache_paths() -> tuple[Path, Path, Path, Path, Path]:
    base = config.CACHE_DIR
    return (base / "games.parquet", base / "team_stats.parquet",
            base / "player_stats.parquet", base / "play_by_play.parquet",
            base / "warehouse_manifest.json")


def _read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read %s (%s); rebuilding it", path.name, exc)
        return pd.DataFrame()


def _write_cache(tables: dict[str, pd.DataFrame], paths: dict[str, Path],
                 keys: dict[str, list[str]]) -> None:
    """Merge each table into its cache, newest row wins on identity."""
    for name, path in paths.items():
        frame = tables.get(name)
        if frame is None:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = pd.DataFrame() if _flag(FULL_REPULL_ENV, False) else _read_cache(path)
        if existing.empty:
            merged = frame
        else:
            identity = [c for c in keys.get(name, []) if c in frame.columns
                        and c in existing.columns]
            merged = pd.concat([frame, existing], ignore_index=True)
            if identity:
                merged = merged.drop_duplicates(subset=identity, keep="first")
        merged.to_parquet(path, index=False)


# --------------------------------------------------------------------------
# Validation and manifest
# --------------------------------------------------------------------------


def _validate_dataset(wh: Warehouse) -> None:
    """The gates that decide whether this data may train a model."""
    required = {"game_id", "gameday", "season", "home_team", "away_team"}
    missing = required - set(wh.games.columns)
    if missing:
        raise RuntimeError(f"NBA games missing required columns: {sorted(missing)}")
    if wh.games.empty:
        raise RuntimeError("NBA pull produced no settled games")
    seasons = pd.to_numeric(wh.games.season, errors="coerce")
    eligible = wh.games[seasons >= config.OOF_FIRST_SEASON]
    if eligible.empty:
        newest = (pd.to_datetime(wh.games.gameday, errors="coerce").max())
        raise RuntimeError(
            f"NBA pull has no {config.OOF_FIRST_SEASON}-or-later season "
            f"(seasons {sorted(set(seasons.dropna()))}, newest game "
            f"{None if pd.isna(newest) else newest.date()}, window "
            f"{wh.manifest.get('window')}). Check NBA_START_DATE/"
            f"NBA_END_DATE, or widen the window upstream.")
    observed = (set(eligible.home_team.astype(str))
                | set(eligible.away_team.astype(str)))
    missing_teams = sorted(set(config.NBA_TEAM_ID) - observed)
    if missing_teams:
        raise RuntimeError("NBA pull is missing current-team coverage: "
                           + ", ".join(missing_teams))
    if wh.team_stats.empty:
        raise RuntimeError("NBA pull is missing required team box scores")
    covered = set(wh.team_stats.game_id.astype(str)) & set(eligible.game_id.astype(str))
    needed = (wh.team_stats[wh.team_stats.game_id.astype(str).isin(covered)]
              .team.astype(str).map(config.normalize_team_abbr))
    missing_facts = sorted(set(config.NBA_TEAM_ID) - set(needed))
    if missing_facts:
        raise RuntimeError("NBA pull is missing team box scores for: "
                           + ", ".join(missing_facts))


def _season_game_prefix(season_start_year: int) -> str:
    """NBA game ids are ``002`` plus the two-digit season start year."""
    return f"002{season_start_year % 100:02d}"


def _gameday_from_code(game_code: Any) -> pd.Timestamp | None:
    """Tipoff date from ``gameCode`` (``YYYYMMDD/AWAYATHOME``).

    ``gameTimeUTC`` on a box score is when the record was finalized, not when
    the tip happened, so the date has to come from the code.
    """
    text = str(game_code or "")
    head = text.split("/", 1)[0]
    if len(head) == 8 and head.isdigit():
        return pd.to_datetime(head, format="%Y%m%d", errors="coerce")
    return None


def _game_type_for(gameday: pd.Timestamp) -> int:
    """Postseason only inside the playoff window, regular season outside it.

    The box score carries no season-type flag, so the date decides.  The
    window has to be bounded at both ends: an open-ended ``>= (4, 15)`` is
    true for every October game, because a plain tuple comparison puts month
    before day.
    """
    month_day = (gameday.month, gameday.day)
    post = PLAYOFF_START <= month_day < PLAYOFF_END
    return config.GAME_TYPE_POST if post else config.GAME_TYPE_REG


def _frames_from_boxscore(payload: Any) -> tuple[pd.DataFrame, pd.DataFrame,
                                                 pd.DataFrame] | None:
    """Normalize one box score into the same three frames the season log yields."""
    game = (payload or {}).get("game") or {}
    gameday = _gameday_from_code(game.get("gameCode"))
    home, away = game.get("homeTeam") or {}, game.get("awayTeam") or {}
    if gameday is None or not home.get("score") or not away.get("score"):
        return None
    game_id = str(game.get("gameId") or "")
    if not game_id:
        return None
    game_type = _game_type_for(gameday)
    season = gameday.year if gameday.month >= 7 else gameday.year - 1

    def side_stats(team: dict, is_home: bool) -> dict[str, Any]:
        raw = team.get("statistics") or {}
        city, nickname = team.get("teamCity"), team.get("teamName")
        row: dict[str, Any] = {
            "game_id": game_id, "gameday": gameday, "team": _team(team.get("teamTricode")),
            "team_name": " ".join(p for p in (city, nickname) if p) or None,
            "is_home": is_home, "opponent": _team(
                (away if is_home else home).get("teamTricode")),
            "points_for": raw.get("points"),
            "points_against": (away if is_home else home).get("statistics", {}).get("points"),
        }
        for source, canonical in _TEAM_STAT_MAP.items():
            if source in raw and (canonical not in row or pd.isna(row[canonical])):
                row[canonical] = raw[source]
        row["net_points"] = (row["points_for"] or 0) - (row["points_against"] or 0)
        for made, attempt, name in (("fgm", "fga", "fg_pct"),
                                    ("fg3m", "fg3a", "three_point_pct"),
                                    ("ftm", "fta", "free_throw_pct")):
            if row.get(attempt):
                row[name] = row.get(made, 0) / row[attempt]
        return row

    games = pd.DataFrame([{
        "game_id": game_id, "gameday": gameday, "season": float(season),
        "home_team": _team(home.get("teamTricode")),
        "away_team": _team(away.get("teamTricode")),
        "home_score": float(home.get("score") or 0),
        "away_score": float(away.get("score") or 0),
        "game_type": game_type,
        "margin": float(home.get("score") or 0) - float(away.get("score") or 0),
        "total": float(home.get("score") or 0) + float(away.get("score") or 0),
        "home_win": 1.0 if (home.get("score") or 0) > (away.get("score") or 0) else 0.0,
    }])
    team_rows = [side_stats(home, True), side_stats(away, False)]

    player_rows: list[dict[str, Any]] = []
    for team, is_home in ((home, True), (away, False)):
        team_abbr = _team(team.get("teamTricode"))
        opponent = _team((away if is_home else home).get("teamTricode"))
        for player in team.get("players") or []:
            raw = player.get("statistics") or {}
            if not player.get("played") and not raw.get("secondsPlayed"):
                continue
            row: dict[str, Any] = {
                "game_id": game_id, "gameday": gameday,
                "player_id": str(player.get("personId") or ""),
                "player_name": player.get("name"), "team": team_abbr,
                "opponent": opponent, "is_home": is_home,
                "minutes": _minutes(raw.get("minutes") or raw.get("minutesCalculated")),
                "game_type": game_type,
            }
            for source, canonical in _PLAYER_STAT_MAP.items():
                if source in raw:
                    row[canonical] = raw[source]
            for made, attempt, name in (("fgm", "fga", "fg_pct"),
                                        ("fg3m", "fg3a", "three_point_pct"),
                                        ("ftm", "fta", "free_throw_pct")):
                if row.get(attempt):
                    row[name] = row.get(made, 0) / row[attempt]
            if "reb" in row:
                row["rebounds"] = row["reb"]
            if "ast" in row:
                row["assists"] = row["ast"]
            player_rows.append(row)
    players = (pd.DataFrame(player_rows) if player_rows
               else pd.DataFrame(columns=["game_id", "player_id"]))
    return games, pd.DataFrame(team_rows), players


def _cdn_season_path(season_year: int) -> Path:
    return config.CACHE_DIR / f"cdn_season_{season_year}.parquet"


def _pull_season_from_cdn(season_year: int, pause: float) -> pd.DataFrame | None:
    """Read one season from per-game box scores and cache the result.

    A season's game ids are contiguous, so the walk stops at the first 403.
    The walk is also the data pull, so nothing is fetched twice, and a season
    is only re-walked when it has no cache, a full re-pull is requested, or it
    is still in progress.
    """
    path = _cdn_season_path(season_year)
    if path.exists() and not _flag(FULL_REPULL_ENV, False):
        try:
            cached = pd.read_parquet(path)
            newest = pd.to_datetime(cached.get("gameday"), errors="coerce").max()
            if (not cached.empty and (pd.isna(newest) or (
                    newest.date() >= date.today() - timedelta(days=REFRESH_TAIL_DAYS)))):
                logger.info("%d-%s from CDN cache (%d rows)", season_year,
                            season_year + 1, len(cached))
                return cached
            if not cached.empty:
                logger.info("%d-%s from CDN cache (%d rows)", season_year,
                            season_year + 1, len(cached))
                return cached
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read %s (%s); re-walking", path.name, exc)

    prefix = _season_game_prefix(season_year)
    logger.info("reading %d-%s from per-game box scores (no stats.nba.com)",
                season_year, season_year + 1)
    frames: list[pd.DataFrame] = []
    failures = 0
    for sequence in range(1, _int_env(MAX_SEQUENCE_ENV, MAX_SEQUENCE_PROBE) + 1):
        game_id = f"{prefix}{sequence:05d}"
        try:
            payload = _get_json(BOXSCORE_URL.format(game_id=game_id),
                                allow_missing=True, pause=pause)
        except Exception as exc:  # noqa: BLE001 - the walk decides what to do
            failures += 1
            logger.error("NBA box score %s failed: %s", game_id, exc)
            if failures >= MAX_CONSECUTIVE_FAILURES:
                raise CdnUnavailable(
                    f"cdn.nba.com stopped answering after {failures} box "
                    f"scores in {season_year}-{season_year + 1} "
                    f"({game_id}): {exc}") from exc
            continue
        failures = 0
        if payload is None:
            logger.info("%s ended at game %d", prefix, sequence - 1)
            break
        built = _frames_from_boxscore(payload)
        if built is not None:
            frames.append(pd.concat(built, ignore_index=True))
        if sequence % 100 == 0:
            logger.info("  %s: %d games read", prefix, len(frames))
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not cache %s (%s)", path.name, exc)
    return merged


def _cdn_seasons_in(start: date, end: date) -> list[int]:
    """Season start years whose schedule can overlap the window.

    A season runs from late October to the following June, so walking one
    whose games all fall outside the window costs a thousand requests and
    yields nothing.
    """
    first = start.year if start.month >= 10 else start.year - 1
    return [year for year in range(first, end.year + 1)
            if date(year, 10, 1) <= end and date(year + 1, 6, 30) >= start]


def _pull_seasons_from_cdn(start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame,
                                                           pd.DataFrame, dict[str, str]]:
    """Build the normalized frames from cached per-game box scores alone."""
    years = _cdn_seasons_in(start, end)
    pause = _float_env(PAUSE_ENV, DEFAULT_PAUSE_SEC)
    collected: list[pd.DataFrame] = []
    for year in years:
        try:
            frame = _pull_season_from_cdn(year, pause)
        except CdnUnavailable:
            # The host itself is gone: walking the remaining seasons would
            # only repeat the same timeout, so name it and stop.
            raise
        except RuntimeError as exc:
            logger.error("CDN season %d unavailable: %s", year, exc)
            continue
        if frame is not None and not frame.empty:
            collected.append(frame)
    if not collected:
        raise RuntimeError(
            f"No NBA data could be read for {start}..{end} from either "
            "stats.nba.com or cdn.nba.com. Both are unreachable from this "
            "host; the pipeline cannot run on an empty window.")
    blob = pd.concat(collected, ignore_index=True)
    games = blob[blob.get("home_score").notna()].drop_duplicates("game_id")
    games = games[(pd.to_datetime(games.gameday) >= pd.Timestamp(start))
                  & (pd.to_datetime(games.gameday) <= pd.Timestamp(end)
                     + pd.Timedelta(days=1))]
    team_stats = blob[blob.get("team").notna() & blob.get("points_for").notna()]
    team_stats = team_stats[team_stats.game_id.isin(set(games.game_id))]
    if "team_name" not in team_stats.columns:
        team_stats = team_stats.assign(team_name=None)
    names = (team_stats[["team", "team_name"]].dropna(subset=["team_name"])
             .drop_duplicates("team")
             .set_index("team").team_name.to_dict())
    player_stats = blob[blob.get("player_id").notna()]
    player_stats = player_stats[player_stats.game_id.isin(set(games.game_id))]
    regular = int((games.game_type == config.GAME_TYPE_REG).sum())
    logger.warning(
        "NBA rebuilt from %d CDN box scores: %d games (%d regular, %d "
        "classified postseason by date, as the box score carries no "
        "season-type flag), %d team rows, %d player rows",
        len(games), len(games), regular, len(games) - regular,
        len(team_stats), len(player_stats))
    return (games.reset_index(drop=True), team_stats.reset_index(drop=True),
            player_stats.reset_index(drop=True), names)


def _manifest(wh: Warehouse, start: date, end: date) -> dict[str, Any]:
    games = wh.games
    dates = pd.to_datetime(games.gameday, errors="coerce")
    seasons = pd.to_numeric(games.season, errors="coerce").dropna()
    return {
        "dataset_id": SOURCE_ID,
        "dataset_version": f"through-{end.isoformat()}",
        "source_path": SOURCE_ID,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "tables": {name: int(len(frame))
                   for name, frame in (("games", games),
                                       ("team_stats", wh.team_stats),
                                       ("player_stats", wh.player_stats),
                                       ("play_by_play", wh.play_by_play))},
        "schemas": {name: [str(c) for c in frame.columns]
                    for name, frame in (("games", games),
                                        ("team_stats", wh.team_stats),
                                        ("player_stats", wh.player_stats),
                                        ("play_by_play", wh.play_by_play))},
        "coverage": {
            "min_date": None if dates.dropna().empty else str(dates.min().date()),
            "max_date": None if dates.dropna().empty else str(dates.max().date()),
            "seasons": sorted(seasons.unique().tolist()),
            "games": int(len(games)),
            "teams": sorted(set(games.home_team) | set(games.away_team)),
        },
        "team_names": wh.team_names,
        "schema_version": "nba-normalized-v3",
    }


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------


def eligible_games(games: pd.DataFrame) -> pd.DataFrame:
    df = games.copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df[df["season"] >= config.OOF_FIRST_SEASON].copy()
    if "game_type" in df:
        df = df[pd.to_numeric(df["game_type"], errors="coerce").isin(config.GAME_TYPES)]
    return df.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def _load_manifest_cache(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}


def load_dataset(source: str | Path | None = None, use_cache: bool = True,
                 allow_download: bool = True) -> Warehouse:
    """Pull, cache, and normalize the NBA data the model stack consumes.

    ``source`` is accepted for CLI compatibility and, when it names a
    directory, is used as the cache location.  ``allow_download=False`` keeps
    the run entirely on cached data, which is what ``--skip-pull`` means now
    that the source is an API rather than an archive.
    """
    paths = {name: path for name, path in zip(
        ("games", "team_stats", "player_stats", "play_by_play"), _cache_paths()[:4])}
    manifest_path = _cache_paths()[4]
    if source not in (None, "", "none", "null"):
        candidate = Path(str(source)).expanduser()
        if candidate.is_dir():
            config.CACHE_DIR = candidate
            paths = {name: candidate / path.name
                     for name, path in paths.items()}
            manifest_path = candidate / manifest_path.name
            logger.info("using %s as the NBA cache directory", candidate)

    start, end = _window()
    if not allow_download:
        logger.info("pull disabled; serving cached NBA data only")
        games = _read_cache(paths["games"])
        if games.empty:
            raise RuntimeError(
                "NBA pull is disabled and no cached games exist; run once with "
                "pulls enabled to populate the cache.")
        team_stats = _read_cache(paths["team_stats"])
        player_stats = _read_cache(paths["player_stats"])
        team_names = dict(_load_manifest_cache(manifest_path).get("team_names") or {})
    else:
        games, team_stats, player_stats, team_names = _pull_seasons(start, end)

    play_by_play = _pull_play_by_play(games, start, end, paths["play_by_play"],
                                      enabled=allow_download and _flag(PLAY_BY_PLAY_ENV, True))

    wh = Warehouse(games, team_stats, player_stats, team_names, {}, play_by_play)
    wh.manifest = _manifest(wh, start, end)
    _validate_dataset(wh)
    _write_cache({"games": games, "team_stats": team_stats,
                  "player_stats": player_stats, "play_by_play": play_by_play},
                 paths,
                 {"games": ["game_id"], "team_stats": ["game_id", "team"],
                  "player_stats": ["game_id", "player_id"],
                  "play_by_play": ["game_id", "action_id"]})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(wh.manifest, indent=2, default=str))
    return wh


def _pull_seasons(start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame,
                                                   pd.DataFrame, dict[str, str]]:
    """Fetch every season log overlapping the window and normalize it.

    A season that cannot be read is recorded and skipped rather than ending the
    run, because the coverage gates downstream decide whether what did arrive
    is enough to train on.  Only a total failure raises here, and it names the
    endpoint so the cause is diagnosable from the log alone.
    """
    pause = _float_env(PAUSE_ENV, DEFAULT_PAUSE_SEC)
    deadline = time.monotonic() + _float_env(PULL_DEADLINE_ENV,
                                             DEFAULT_PULL_DEADLINE_SEC)
    logs: list[pd.DataFrame] = []
    unavailable: list[str] = []
    skipped: list[str] = []
    consecutive_failures = 0
    gave_up = False
    for season in _seasons_in(start, end):
        for season_type, game_type in ((SEASON_TYPE_REGULAR, config.GAME_TYPE_REG),
                                       (SEASON_TYPE_PLAYOFFS, config.GAME_TYPE_POST)):
            if gave_up:
                skipped.append(f"{season} {season_type}")
                continue
            if time.monotonic() > deadline:
                skipped.append(f"{season} {season_type}")
                logger.warning("NBA pull budget exhausted; skipping %s %s. "
                               "Raise %s to pull more of the window.",
                               season, season_type, PULL_DEADLINE_ENV)
                continue
            try:
                raw = _fetch_season_log(season, season_type, pause)
            except SeasonUnavailable as exc:
                unavailable.append(f"{season} {season_type}")
                consecutive_failures += 1
                logger.error("NBA season log unavailable, continuing without "
                             "it: %s", exc)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    gave_up = True
                    logger.error(
                        "NBA pull stopping after %d consecutive season "
                        "failures; the endpoint is not answering, so the "
                        "remaining seasons would only repeat it.",
                        consecutive_failures)
                continue
            consecutive_failures = 0
            frame = _prepare_log(raw, game_type)
            if frame.empty:
                logger.info("%s %s returned no games", season, season_type)
                continue
            logger.info("%s %s: %d player lines across %d games", season,
                        season_type, len(frame), frame.game_id.nunique())
            logs.append(frame)
    if skipped:
        logger.warning("NBA pull stopped early; %d season logs were not "
                       "attempted: %s", len(skipped), ", ".join(skipped))
    if gave_up and logs:
        logger.warning("NBA pull completed on %d season log(s) despite %d "
                       "failures; check the seasons listed above before "
                       "trusting the window.", len(logs), len(unavailable))
    if unavailable:
        logger.warning("NBA could not read %d of the requested season logs: %s. "
                       "Anything already fetched is cached, so the next run "
                       "retries only these.", len(unavailable),
                       ", ".join(unavailable))
    if not logs:
        if _flag(CDN_FALLBACK_ENV, True):
            logger.warning("No season log could be read; rebuilding the window "
                           "from per-game CDN box scores instead.")
            return _pull_seasons_from_cdn(start, end)
        raise RuntimeError(
            f"NBA returned no games for {start}..{end}; every season log "
            f"failed ({', '.join(unavailable) or 'none attempted'}). The "
            "upstream endpoint is stats.nba.com/stats/LeagueGameLog, which "
            "needs no key but can be slow or blocked by cloud hosts. Retry, or "
            "narrow NBA_START_DATE/NBA_END_DATE to a window whose seasons are "
            f"already cached, or unset {CDN_FALLBACK_ENV}=0 to use the CDN.")
    log = pd.concat(logs, ignore_index=True)
    log = log[(log.gameday >= pd.Timestamp(start))
              & (log.gameday <= pd.Timestamp(end) + pd.Timedelta(days=1))]

    games = _games_frame(log)
    if games.empty:
        raise RuntimeError(f"NBA returned no settled games for {start}..{end}")
    team_stats = _team_stats_frame(log, games)
    player_stats = _player_stats_frame(log)
    names = (log[["team", "team_name"]].dropna().drop_duplicates("team")
             .set_index("team")["team_name"].astype(str).to_dict()
             if "team_name" in log.columns else {})
    team_names = {abbr: names.get(abbr, abbr)
                  for abbr in sorted(set(games.home_team) | set(games.away_team))}
    games = _attach_team_names(games, team_names)
    logger.info("NBA pull: %d games, %d team-stat rows, %d player rows",
                len(games), len(team_stats), len(player_stats))
    return games, team_stats, player_stats, team_names


def _pull_play_by_play(games: pd.DataFrame, start: date, end: date,
                       path: Path, *, enabled: bool) -> pd.DataFrame:
    """Fetch play-by-play for every cached game, incrementally.

    Play-by-play is the only per-game request in the pipeline, so the cache is
    the difference between an increment and a rebuild.  Games the cache has not
    seen are fetched, and a short trailing window is always re-pulled because a
    run that overlapped a live game would otherwise keep that partial file.
    """
    if games.empty or not enabled:
        return _read_cache(path)
    cached = _read_cache(path)
    have = set(cached.game_id.astype(str)) if not cached.empty else set()
    wanted = (games[["game_id", "gameday"]]
              .drop_duplicates("game_id")
              .reset_index(drop=True))
    tail = pd.Timestamp(date.today() - timedelta(days=REFRESH_TAIL_DAYS))
    missing = wanted[~wanted.game_id.astype(str).isin(have)
                     | (wanted.gameday >= tail)]
    logger.info("play-by-play: %d games cached, %d to fetch", len(have),
                len(missing))
    if missing.empty:
        return cached
    pause = _float_env(PAUSE_ENV, DEFAULT_PAUSE_SEC)
    frames: list[pd.DataFrame] = []
    failures = 0
    for row in missing.itertuples(index=False):
        payload = _get_json(PLAY_BY_PLAY_URL.format(game_id=row.game_id),
                            allow_missing=True, retries=2, pause=pause)
        frame = _play_by_play_frame(payload, str(row.game_id), row.gameday)
        if frame.empty:
            failures += 1
        else:
            frames.append(frame)
        time.sleep(pause)
    if failures:
        logger.warning("play-by-play empty for %d of %d requested games",
                       failures, len(missing))
    if not frames:
        return cached
    increment = pd.concat(frames, ignore_index=True)
    if not cached.empty:
        increment = pd.concat([increment, cached], ignore_index=True)
        increment = increment.drop_duplicates(
            subset=["game_id", "action_id"], keep="first")
    logger.info("play-by-play: %d actions across %d games", len(increment),
                increment.game_id.nunique())
    return increment


def load_games(source: str | Path | None = None, use_cache: bool = True) -> pd.DataFrame:
    return load_dataset(source, use_cache).games


def load_team_stats(source: str | Path | None = None,
                    use_cache: bool = True) -> pd.DataFrame:
    return load_dataset(source, use_cache).team_stats


def load_player_stats(source: str | Path | None = None,
                      use_cache: bool = True) -> pd.DataFrame:
    return load_dataset(source, use_cache).player_stats
