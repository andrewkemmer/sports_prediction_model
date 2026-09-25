"""Ingest NBA games, box scores, and play-by-play from NBA.com's free APIs.

NBA.com publishes two unauthenticated surfaces, and both are used here.

``stats.nba.com``
    ``LeagueGameLog`` returns a whole season of player game logs in one
    request.  It is the backbone: the game index (id, date, matchup) and every
    player line arrive in the same call, and a team box score is the sum of its
    players' lines.  Regular season and playoffs are separate calls, which is
    also where the game type comes from, so nothing is inferred from dates.

``cdn.nba.com``
    The same JSON NBA.com itself serves: per-game box scores and play-by-play.
    It is the fallback when the season log cannot be read: the box score
    carries the schedule, the team lines, and the player lines, so one route
    rebuilds the whole window where the season log rebuilds it in one call.

There is no third route.  A backstop outside NBA.com cannot rescue a run
whose actual problem is that this host cannot reach NBA.com, and every route
added is a contract to keep alive.  The game index, the team lines, and the
player lines all come from the two surfaces above, so if both refuse, the
error says so and the remedy is the network, not another vendor.

No key, quota, or paid tier is involved on any route.  The season log costs ~10
requests for any window, so it is re-read every run.  Play-by-play is one
request per game, so it is cached to Parquet and only games the cache has not
seen are fetched, which turns a daily run into an increment instead of a
rebuild.

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

import http.cookiejar
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
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from backend import config, progress
except ImportError:
    import config
    import progress

logger = logging.getLogger(__name__)

SEASON_LOG_URL = "https://stats.nba.com/stats/LeagueGameLog"
BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
PLAY_BY_PLAY_URL = ("https://cdn.nba.com/static/json/liveData/playbyplay/"
                    "playbyplay_{game_id}.json")

SOURCE_ID = "nba.com"
# Which upstream actually answered. Cloud hosts refuse nba.com at the IP
# level and something else has to serve the window, so the manifest records
# the route that won rather than claiming NBA.com every time.
SOURCE_USED: dict[str, str] = {"source": "nba.com"}

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

# stats.nba.com sits behind Akamai Bot Manager, and the /stats/ tree is the one
# path on the whole site it drops unauthenticated: without a session the edge
# accepts the TLS handshake and then never answers, so every request costs a
# full timeout and the season log — a whole season of player lines in one call,
# and the backbone of the whole pipeline — is unreachable.  It is not a rate
# limit and not a header problem: the host root answers 200 in half a second
# and /stats/ times out at any rate, with any headers, from any rung.
#
# A cookie jar primed once from nba.com carries _abck and friends, and the same
# request that timed out for the whole timeout budget then answers in about
# three seconds.  So the session is primed before the first stats request and
# re-primed if a stats request ever comes back silent, since an expired _abck
# looks exactly like the blocked case.
_SESSION_URL = "https://www.nba.com/"
_SESSION_HOSTS = ("stats.nba.com",)
_session_cookies: http.cookiejar.CookieJar | None = None
_session_primed_at = 0.0
SESSION_TTL_SEC = 1800.0


def _prime_nba_session(force: bool = False) -> int:
    """Fetch nba.com once and keep the cookies the stats edge wants.

    Returns the number of cookies now available.  Failure is not an error: the
    caller still tries the request, because a host that has stopped demanding a
    session would answer perfectly well without one, and refusing to ask would
    turn a passing tolerance into an outage.
    """
    global _session_cookies, _session_primed_at
    if not force and _session_cookies is not None and (
            time.monotonic() - _session_primed_at < SESSION_TTL_SEC):
        return len(_session_cookies)
    jar: http.cookiejar.CookieJar | None = None
    try:
        jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(jar))
        request = urllib.request.Request(_SESSION_URL, headers=dict(_HTTP_HEADERS))
        with opener.open(request, timeout=20) as response:
            response.read(2048)
    except Exception as exc:  # noqa: BLE001 - a missing session is survivable
        logger.debug("could not prime an nba.com session (%s)", exc)
    if jar is not None:
        _session_cookies = jar
        _session_primed_at = time.monotonic()
        if jar:
            logger.info("primed an nba.com session: %d cookie(s) for the "
                        "stats edge", len(jar))
    return len(_session_cookies) if _session_cookies is not None else 0


def _session_header() -> dict[str, str]:
    """The ``Cookie`` line for a primed session, or an empty header set."""
    if _session_cookies is None or not len(_session_cookies):
        return {}
    request = urllib.request.Request(_SESSION_URL)
    try:
        _session_cookies.add_cookie_header(request)
    except Exception:  # noqa: BLE001
        return {}
    return {"Cookie": request.get_header("Cookie")} if request.has_header("Cookie") else {}

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

# Headers are a PER-HOST contract, and the two hosts disagree about it. The
# order below was measured, not guessed - each rung is the request that host
# actually answered with:
#
#   cdn.nba.com        serves the full browser costume, and 403s a bare
#                      User-Agent - which is why the costume stays first here.
#   stats.nba.com      needs the same costume plus its own x-nba-stats-* tokens.
#
# Each host is a LADDER rather than a single profile, because which rung answers
# differs by edge and by caller: a refusal that arrives in milliseconds means
# the host ANSWERED, and the cheapest way to find out whether it answered a
# different client is to send a different client. The rung that answers is
# remembered for the run, so a season of thousands of box scores is asked once
# in the right voice instead of re-probed on every request.
_HEADER_LADDERS: dict[str, tuple[dict[str, str], ...]] = {
    "cdn.nba.com": (_HTTP_HEADERS, {}),
    "stats.nba.com": (_STATS_HEADERS, {}),
}
# Which rung a host last answered on. Process-local on purpose: it describes
# this network path, not the host, and a fresh run re-learns it in one request.
_HEADER_RUNG: dict[str, int] = {}


def _header_ladder(host: str) -> tuple[dict[str, str], ...]:
    return _HEADER_LADDERS.get(host, (_HTTP_HEADERS,))


def _host_profile(host: str) -> tuple[int, dict[str, str]]:
    """The rung to open this host with: the one that last answered, if any."""
    ladder = _header_ladder(host)
    rung = min(_HEADER_RUNG.get(host, 0), len(ladder) - 1)
    return rung, ladder[rung]


def _next_rung(host: str, rung: int) -> int | None:
    """The next profile to try after a refusal, or None if the ladder is spent."""
    following = rung + 1
    return following if following < len(_header_ladder(host)) else None


def _reprobe_refusal(url: str, host: str) -> bool:
    """Settle, for one request, whether a burst of refusals is about our headers.

    A season of 1,723 identical 403s reads as "1,723 games do not exist", which
    is a very different conclusion from "this client is refused" - and it takes
    exactly one request to tell them apart, because a profile that works works
    for all of them.  Returns True when the next rung answered, and leaves that
    rung in place for the rest of the run.
    """
    rung, _ = _host_profile(host)
    following = _next_rung(host, rung)
    if following is None:
        return False
    profile = _header_ladder(host)[following]
    try:
        payload = _get_json(url, headers=profile, allow_missing=True)
    except Exception as exc:  # noqa: BLE001 - any failure is a "no"
        logger.warning("%s refused the re-probe too (%s); it is not our headers",
                       host, exc)
        return False
    if payload is None:
        logger.warning("%s refused the re-probe with %d header(s) too; it is not "
                       "our headers", host, len(profile))
        return False
    _HEADER_RUNG[host] = following
    logger.warning("%s answered once we changed what we sent; that profile is "
                   "used for the rest of the run", host)
    return True


# Name resolution does not honour a socket timeout, so every attempt is run on
# a thread and abandoned at a hard wall clock.  Without this a blackholed host
# blocks forever with no CPU and no error.
DNS_GRACE_SEC = 30
PULL_DEADLINE_ENV = "NBA_PULL_DEADLINE_SEC"
DEFAULT_PULL_DEADLINE_SEC = 600.0
# The CDN fallback asks about every game in the window, so it needs a far larger
# budget than the season-log pull above: a full window is a few thousand
# requests, and this is the only source of player lines on a host that blocks
# stats.nba.com.  The cap exists to stop a slow host running a session dry, not
# to interrupt a run that is making progress - the box scores already fetched
# are cached, so a run that exhausts it resumes where it left off.
CDN_WALK_BUDGET_ENV = "NBA_CDN_WALK_BUDGET_SEC"
DEFAULT_CDN_WALK_BUDGET_SEC = 5400.0
# A healthy pull answers all eight season requests in seconds.  When the host
# is actually down, repeating the same doomed request for every remaining
# season buys nothing but another few minutes of the user watching a still run,
# so the pull gives up after this many consecutive failures.  One is enough
# now that a dead season log only costs a fallback route rather than the run:
# a season log answers from one host in one request, so a single read timeout
# on a cold host is the answer, not a sample of one.
MAX_CONSECUTIVE_FAILURES = 1

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
# A game id is 00 + season type + 2-digit season start year + 5 digits, and the
# season type is the whole reason the id space is walkable:
#
#   001SSnnnnn  preseason          (not part of a decided season)
#   002SSnnnnn  regular season     nnnnn counts 1..N with no gaps
#   004SS00RCSG playoffs           R round, C series, G game in that series
#   006SSnnnnn  in-season cup final (a regular-season game with its own id)
#
# The playoff numbering is sparse — a 4-game series leaves 5-game holes and
# each round starts at series 0 — so it cannot be walked like the regular
# season and has to be enumerated.  This is what makes the whole postseason
# reachable: 2024-25 alone is 84 games across four rounds, ending with the
# finals on 2025-06-22.
REGULAR_PREFIX = "002"
PLAYOFF_PREFIX = "004"
CUP_PREFIX = "006"
PLAYOFF_ROUNDS = (0, 1, 2, 3, 4)
PLAYOFF_SERIES = tuple(range(0, 9))
PLAYOFF_GAMES = tuple(range(1, 8))
CUP_SEQUENCE_PROBE = 8
# The window is pulled in slices, as the MLB Statcast pull is, so a long range
# is bounded work, a failure costs one slice, and progress is durable per slice.
# 60 days is MLB's ``statcast_chunk_days``: a slice should be big enough that
# the pause between slices is noise, and small enough that a killed run resumes
# without re-pulling a season. The two knobs are independent, because widening
# the population slice must not coarsen the hole detector below.
CDN_CHUNK_DAYS = 60
CHUNK_DAYS_ENV = "NBA_CHUNK_DAYS"
# The gap scan stays at 30 days no matter how wide the pull slices get. A hole
# in the middle of the season is what this exists to catch, and a 60-day slice
# can hide a 10-day hole behind two months of games that did arrive. The scan is
# in memory, so its resolution costs nothing.
GAP_SCAN_DAYS = 30
# A game, its two team lines and every player line all share a game id, so the
# cache key has to name the row kind as well.
ROW_KEY = ["row_kind", "game_id", "team", "player_id"]
# Months where an NBA game is always on the calendar somewhere in the league.
# Outside these, an empty slice is the schedule, not a gap in the pull.
CORE_SEASON_MONTHS = frozenset({11, 12, 1, 2, 3, 4, 5})

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
class NBAFacts:
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


class HostBlocked(RuntimeError):
    """This run already established that a host will not serve it.

    A verdict, not a failure.  The host is not down — it answered, and the
    answer was "no" — so the distinction matters to the callers: a walk that is
    still storing games should stop asking and keep what it has, while a caller
    with nothing to lose should fall through to the next source.  Both are
    ``RuntimeError``s, so any handler that only catches the base type behaves
    exactly as it did before this existed.
    """


def _request_json(url: str, headers: dict[str, str], timeout: int) -> Any:
    """Perform one request under a hard wall clock.

    ``urlopen``'s timeout covers socket reads, not DNS, so a host that never
    resolves would otherwise block the run indefinitely.  Running the call on a
    thread and abandoning it at the deadline makes "hung" indistinguishable
    from "slow" impossible to reach.

    The stats edge is handed a primed session cookie when there is one; every
    other host is asked exactly as it was before.
    """
    outcome: dict[str, Any] = {}
    sent = dict(headers)
    if urllib.parse.urlparse(url).netloc in _SESSION_HOSTS:
        sent.update(_session_header())

    def worker() -> None:
        try:
            request = urllib.request.Request(url, headers=sent)
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
    logger.debug("NBA request completed in %.1fs", time.monotonic() - started)
    return outcome["payload"]


HOST_VERDICT_TTL_HOURS = 24.0
HOST_VERDICT_FILE = "host_verdicts.json"
# The verdict is consulted once per request, and a per-game walk makes that
# thousands of times.  Reading and parsing the file each time is wasted I/O
# for data that changes at most once a run, so it is memoized for a moment.
HOST_VERDICT_CACHE_SEC = 5.0
# How many refusals in a row make a host's verdict worth recording.  A refusal
# is cheap on its own — it arrives in milliseconds — so a run that meets one
# should not write the host off over what may be a single absent artifact.  But
# ``allow_missing`` callers keep asking, and a walk over a season's game ids
# turns that into thousands of requests that were never going to succeed: the
# 2026-09-25 Kaggle run answered 1,723 box-score refusals, fell back to
# per-game CDN reads, and then asked the same blocked host again for 3,474
# play-by-play files.  Sixteen in a row is a statement about the client, not about the
# artifact, and it is a statement the rest of the run needs to inherit.
REFUSAL_VERDICT_COUNT = 16

# Consecutive refusals per host, for this run.  Reset by any success.
_HOST_REFUSALS: dict[str, int] = {}


def _note_refusal(host: str, code: int) -> None:
    """Count a refusal.  Only counting — the caller decides what it means.

    Only 403 counts.  A 404 is the host saying an artifact is absent, which is
    the expected answer for most of the game ids a season probe asks about, so
    counting it here would condemn a perfectly healthy host.  A 403 is the host
    saying it will not serve this client at all.

    Counting rather than recording is deliberate.  A run of refusals only means
    something next to what the caller was trying to do: the box-score walk
    owes a season a re-probe before it will condemn a host, and the
    play-by-play sweep does not.  Let each escalate where it holds the evidence.
    """
    if code != 403:
        return
    _HOST_REFUSALS[host] = _HOST_REFUSALS.get(host, 0) + 1


def _refusals_conclusive(host: str) -> bool:
    """Whether a host has refused enough in a row to stop asking."""
    return _HOST_REFUSALS.get(host, 0) >= REFUSAL_VERDICT_COUNT


def _record_refusal_verdict(host: str, streak: int) -> None:
    """Turn a conclusive run of refusals into a verdict the run will honour."""
    _record_host_verdict(
        host, f"answered and refused us (HTTP 403) on {streak} "
              "consecutive requests")


def _note_success(host: str) -> None:
    """A served request clears the host's refusal streak."""
    if host in _HOST_REFUSALS:
        _HOST_REFUSALS[host] = 0


def _verdicts_path() -> Path:
    return config.CACHE_DIR / HOST_VERDICT_FILE


def _host_verdicts() -> dict[str, str]:
    """The recorded verdicts, memoized briefly.

    The returned dict is the live one: ``_record_host_verdict`` mutates it and
    writes it out, so a verdict recorded in this process is visible to the next
    caller here without touching the disk again.

    The memo is keyed on the verdicts file itself, so pointing ``CACHE_DIR``
    somewhere else reads somewhere else rather than serving another location's
    verdicts from memory.
    """
    global _VERDICT_MEMO
    path = _verdicts_path()
    stamp, where, memo = _VERDICT_MEMO
    if (memo is not None and where == path
            and time.time() - stamp < HOST_VERDICT_CACHE_SEC):
        return memo
    loaded: dict[str, str] = {}
    try:
        stored = json.loads(path.read_text())
        if isinstance(stored, dict):
            loaded = stored
    except Exception:  # noqa: BLE001 - no verdict file is the normal case
        pass
    _VERDICT_MEMO = (time.time(), path, loaded)
    return loaded


_SILENCE_MARKERS = ("timed out", "timeout", "did not resolve",
                    "name or service not known", "unreachable")

# The closing advice for a run where every route failed.  It is the last thing
# an operator reads, so it names what to try rather than what went wrong.
_NO_SOURCE_REMEDY = (
    "Nothing this pipeline sends will change any of those answers, so the run "
    "has to happen somewhere the upstreams are reachable, or against a cache "
    "that already holds the window. Check the log above for which route "
    "refused, and re-run from a host that can reach the source.")

_VERDICT_MEMO: tuple[float, Path, dict[str, str] | None] = (0.0, Path("."), None)


def _failure_verdict(exc: BaseException) -> str:
    """One phrase saying whether a host ANSWERED us or said nothing at all.

    A 403 in 33 ms and a 60-second read timeout look identical in a log line
    that just says "request failed", but they are different problems with
    different fixes: the first is about who we are, the second is about the
    network path.  The run's final error is often the only place the
    difference is visible, so every leg states which one it hit.
    """
    text = str(exc).lower()
    if any(marker in text for marker in _SILENCE_MARKERS):
        return "never answered (no response before the timeout)"
    code = re.search(r"http (?:error )?(\d{3})", text)
    if code:
        return f"answered and refused us (HTTP {code.group(1)})"
    if "refused" in text or "blocked" in text:
        return "answered and served nothing"
    return f"failed ({type(exc).__name__})"


def _record_host_verdict(host: str, reason: str) -> None:
    """Remember what a host just told us, so the rest of the run does not re-pay.

    A blackholed host costs a full timeout budget per season log, and on Kaggle
    that is minutes of the run spent re-proving a fact.  A refused host is the
    same problem wearing a faster mask: the cost per request is small, but the
    callers that walk thousands of ids — the per-game box-score route and the
    play-by-play sweep — will happily spend half an hour collecting it.  So
    both verdicts are written next to the cache and honoured for a day.

    What is NOT recorded is a lone refusal.  It arrives in milliseconds and may
    be about headers rather than the machine, so it only becomes a verdict once
    ``_note_refusal`` has seen a run of them long enough to mean the client.
    """
    verdicts = _host_verdicts()
    verdicts[host] = f"{time.time():.0f}|{reason}"[:400]
    try:
        _verdicts_path().parent.mkdir(parents=True, exist_ok=True)
        _verdicts_path().write_text(json.dumps(verdicts, indent=1))
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not record the verdict for %s (%s)", host, exc)


def _known_blocked_host(host: str) -> str | None:
    """Why a host was recorded as silent, if the verdict still counts."""
    entry = _host_verdicts().get(host)
    if not entry or "|" not in entry:
        return None
    stamp, _, reason = entry.partition("|")
    try:
        age_hours = (time.time() - float(stamp)) / 3600.0
    except ValueError:
        return None
    if age_hours > HOST_VERDICT_TTL_HOURS:
        return None
    return reason


def _get_json(url: str, *, headers: dict[str, str] | None = None,
              retries: int | None = None, pause: float | None = None,
              allow_missing: bool = False, verbose: bool = True) -> Any | None:
    """Fetch JSON with per-host timeout, backoff, and retry policy.

    Every attempt is logged before it starts and after it fails, so a slow
    upstream is visibly slow rather than indistinguishable from a hang.  The
    two hosts behave nothing alike: ``stats.nba.com`` serves a whole season of
    player lines from one very slow query, while ``cdn.nba.com`` is a CDN whose
    403 means "this game does not exist" rather than "slow down".

    A bulk walk passes ``verbose=False``: a season is a couple of thousand box
    scores, and one INFO line per request buries the summary — and the one line
    that matters when a host refuses everything, which is that no request
    completed at all.

    A refusal that arrives in milliseconds means the host ANSWERED, which is a
    different problem from a host that never replies. So a 403 or 404 advances
    one rung of the host's header ladder, outside the retry budget and without
    backoff: the block may be about what we sent rather than who we are, and a
    host that answers in 30 ms has nothing to make us wait for.
    """
    host = urllib.parse.urlparse(url).netloc
    path = urllib.parse.urlparse(url).path
    policy = _HOST_POLICY.get(host, _HOST_POLICY["default"])
    if host in _SESSION_HOSTS:
        # The session is what makes this path answer at all, so it is primed
        # before the first request rather than discovered by a timeout.
        _prime_nba_session()
    blocked = _known_blocked_host(host)
    if blocked is not None:
        # "Never answered" would be a lie here: a refused host answers, it just
        # answers "no".  Name the verdict and let it speak for itself.
        raise HostBlocked(
            f"NBA request to {host} skipped; this run's verdict against it "
            f"still stands: {blocked}")
    attempts = retries if retries is not None else _int_env(
        RETRIES_ENV, policy["attempts"])
    budget = max(attempts, 1)
    base = policy["backoff"] if pause is None else max(pause, 0.0)
    timeout = policy["timeout"]
    if headers is None:
        rung, profile = _host_profile(host)
    else:
        # A caller that names a header set knows what this request needs; it
        # does not get the ladder's opinion.
        rung, profile = 0, dict(headers)
    last: Exception | None = None
    reason = "unknown"
    made = 0
    used_budget = 0
    _session_reprimed = False
    # A header retry is not a budget retry. It gets its own request outside the
    # ``attempts`` count, so a host configured for one attempt still gets the
    # second chance when what is refused is plausibly what we sent.
    rerung = False
    while True:
        made += 1
        if rerung:
            # Consumed here. Leaving it set would make every later iteration a
            # "free" profile retry, so ``used_budget`` would stop climbing and
            # the loop would never reach its own exit. cdn.nba.com sets
            # ``retry_forbidden``, so the 403 below does not break out either,
            # and the walk spins on a host that is refusing every request.
            logger.log(logging.INFO if verbose else logging.DEBUG,
                       "NBA next-header-profile retry to %s (timeout %ds)",
                       host, timeout + DNS_GRACE_SEC)
            rerung = False
        else:
            used_budget += 1
            logger.log(logging.INFO if verbose else logging.DEBUG,
                       "NBA request %s/%s to %s (timeout %ds)",
                       used_budget, budget, host, timeout + DNS_GRACE_SEC)
        try:
            payload = _request_json(url, profile, timeout)
            _note_success(host)
            if headers is None and rung:
                # This profile answered, so the run has learned the host's
                # preferred voice; stop re-deriving it per request.
                _HEADER_RUNG[host] = rung
            return payload
        except urllib.error.HTTPError as exc:
            last = exc
            reason = f"HTTP {exc.code}"
            if exc.code in (403, 404) and allow_missing:
                _note_refusal(host, exc.code)
                return None
            if exc.code in (403, 404) and headers is None:
                following = _next_rung(host, rung)
                if following is not None:
                    # It answered, so it is refusing us rather than the artifact
                    # being absent. Try the next profile we keep for this host:
                    # a client it will serve is one request away.
                    rung, profile = following, _header_ladder(host)[following]
                    rerung = True
                    logger.warning("NBA request to %s rejected (HTTP %s) with %d "
                                   "header(s); trying the next profile for this "
                                   "host", host, exc.code, len(profile))
                    continue
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
            if "timed out" in str(exc).lower() or isinstance(exc, TimeoutError):
                # Silence, not a refusal: this is the one verdict worth
                # remembering, because a re-run would otherwise re-pay the
                # whole timeout budget to learn the same thing.
                if host in _SESSION_HOSTS:
                    # An expired _abck is indistinguishable from the blocked
                    # case from out here, so the session is re-primed and the
                    # request retried once.  Only if that is also silent is
                    # the host actually written off.
                    if not _session_reprimed:
                        _session_reprimed = True
                        _prime_nba_session(force=True)
                        continue
                _record_host_verdict(host, f"no response: {reason}")
        logger.warning("NBA request %s/%s to %s failed: %s", used_budget,
                       budget, host, reason)
        if used_budget < budget:
            # Jitter keeps concurrent clients from retrying in lockstep.
            delay = base * (2 ** (used_budget - 1))
            time.sleep(delay * (0.5 + random.random()))
            continue
        break
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


def _derive_player_shooting(out: pd.DataFrame) -> pd.DataFrame:
    """The shooting percentages and name aliases every player frame carries.

    Two routes that disagreed about how to turn made/attempted into a
    percentage would be two definitions of the same feature, and the second
    one to ship would silently retrain on numbers the first never produced.
    So the rule lives here once and each route calls it.  A zero attempt is
    missing, not zero percent: nobody who played took no shots.
    """
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
    return out


def _player_stats_frame(log: pd.DataFrame) -> pd.DataFrame:
    """One row per player per game."""
    if log.empty:
        return pd.DataFrame()
    keep = [c for c in dict.fromkeys(
        ("game_id", "gameday", "player_id", "player_name", "team", "minutes",
         "win", "game_type", "plus_minus", "fg_pct", "three_point_pct",
         "free_throw_pct", *sorted(_STAT_SUMS))) if c in log.columns]
    out = _derive_player_shooting(log[keep].copy())
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
            base / "run_manifest.json")


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


def _validate_ingested(facts: NBAFacts) -> None:
    """The gates that decide whether this data may train a model."""
    required = {"game_id", "gameday", "season", "home_team", "away_team"}
    missing = required - set(facts.games.columns)
    if missing:
        raise RuntimeError(f"NBA games missing required columns: {sorted(missing)}")
    if facts.games.empty:
        raise RuntimeError("NBA pull produced no settled games")
    seasons = pd.to_numeric(facts.games.season, errors="coerce")
    eligible = facts.games[seasons >= config.OOF_FIRST_SEASON]
    if eligible.empty:
        newest = (pd.to_datetime(facts.games.gameday, errors="coerce").max())
        raise RuntimeError(
            f"NBA pull has no {config.OOF_FIRST_SEASON}-or-later season "
            f"(seasons {sorted(set(seasons.dropna()))}, newest game "
            f"{None if pd.isna(newest) else newest.date()}, window "
            f"{facts.manifest.get('window')}). Check NBA_START_DATE/"
            f"NBA_END_DATE, or widen the window upstream.")
    observed = (set(eligible.home_team.astype(str))
                | set(eligible.away_team.astype(str)))
    missing_teams = sorted(set(config.NBA_TEAM_ID) - observed)
    if missing_teams:
        raise RuntimeError("NBA pull is missing current-team coverage: "
                           + ", ".join(missing_teams))
    if facts.team_stats.empty:
        raise RuntimeError("NBA pull is missing required team box scores")
    covered = set(facts.team_stats.game_id.astype(str)) & set(eligible.game_id.astype(str))
    needed = (facts.team_stats[facts.team_stats.game_id.astype(str).isin(covered)]
              .team.astype(str).map(config.normalize_team_abbr))
    missing_facts = sorted(set(config.NBA_TEAM_ID) - set(needed))
    if missing_facts:
        raise RuntimeError("NBA pull is missing team box scores for: "
                           + ", ".join(missing_facts))
    if facts.player_stats.empty:
        # Every player-derived feature is a default in this frame, and a
        # default looks exactly like a measurement to anything downstream: the
        # models fit, the calibration curve is smooth, the artifacts publish,
        # and the degradation is invisible in every one of them.  So this is a
        # hard stop rather than a warning, and there is no flag to wave it
        # through.  A run that cannot see its players does not get to publish a
        # model about them.
        source = facts.manifest.get("source_route", "the resolved source")
        raise RuntimeError(
            "NBA window has no player detail, so every player-derived feature "
            "would silently fall back to its default. Refusing to train and "
            f"publish. Source was {source!r}, which returned games and team "
            "box scores but no player lines. Fix the data, not this check: "
            "the player box score is fetched from the source's per-game "
            "summary endpoint, so this means that route answered without one. "
            "Check the log for refusal or emptiness warnings above.")


def _season_game_prefix(season_start_year: int) -> str:
    """NBA game ids are ``002`` plus the two-digit season start year."""
    return f"{REGULAR_PREFIX}{season_start_year % 100:02d}"


def _game_type_from_id(game_id: str) -> int | None:
    """Read the season type straight off the id, or None if it is not a game.

    The box score payload carries no season-type flag, so the id is the only
    exact signal.  Guessing from the date instead gets it wrong at both ends
    of the calendar: the regular season runs from late October into April.
    """
    kind = str(game_id)[:3]
    if kind in (REGULAR_PREFIX, CUP_PREFIX):
        return config.GAME_TYPE_REG
    if kind == PLAYOFF_PREFIX:
        return config.GAME_TYPE_POST
    return None


def _playoff_game_ids(season_start_year: int) -> list[str]:
    """Every postseason id that could exist for a season, in bracket order.

    Enumerated rather than walked: the tree is only 5 rounds x 9 series x 7
    games, and an id that does not exist is a cheap 403.
    """
    year = season_start_year % 100
    return [f"{PLAYOFF_PREFIX}{year:02d}00{round_no}{series}{game}"
            for round_no in PLAYOFF_ROUNDS
            for series in PLAYOFF_SERIES
            for game in PLAYOFF_GAMES]


def _candidate_game_ids(season_start_year: int) -> list[str]:
    """Every game id a season could use, in the order the games are played."""
    year = season_start_year % 100
    regular = [f"{REGULAR_PREFIX}{year:02d}{n:05d}"
               for n in range(1, _int_env(MAX_SEQUENCE_ENV, MAX_SEQUENCE_PROBE) + 1)]
    playoffs = _playoff_game_ids(season_start_year)
    cup = [f"{CUP_PREFIX}{year:02d}{n:05d}"
           for n in range(1, CUP_SEQUENCE_PROBE + 1)]
    return regular + playoffs + cup


def _chunk_start(gameday: pd.Timestamp, start: date, days: int) -> date:
    """The slice of the window a game belongs to, anchored at the window start."""
    offset = (gameday.date() - start).days
    return start + timedelta(days=max(0, offset // days) * days)


def _chunk_bounds(cursor: date, end: date, days: int) -> tuple[date, date]:
    return cursor, min(cursor + timedelta(days=days - 1), end)


def _slice_count(start: date, end: date, days: int) -> int:
    """How many slices a window is cut into, for a progress bar's total.

    ``_chunk_bounds`` hands out full-width slices and only the last one is
    clipped, so this is the ceiling of the window over the width.  It is
    arithmetic rather than a walk of the real loop on purpose: the count must
    not be able to disagree with the loop it is counting.
    """
    span = (end - start).days + 1
    return max(1, -(-span // max(1, days)))


def _is_core_season_chunk(cursor: date, chunk_end: date) -> bool:
    """True when a chunk sits where games are always being played.

    November through May is the only stretch an empty slice is real damage:
    October runs into the preseason and the All-Star break, June ends with
    the finals, and July through September is the offseason.  A future slice
    is empty because the games have not been played yet.
    """
    midpoint = cursor + (chunk_end - cursor) / 2
    if cursor >= date.today():
        return False
    return midpoint.month in CORE_SEASON_MONTHS


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
    game_type = _game_type_from_id(game_id)
    if game_type is None:
        return None  # preseason: not a decided game, so not training data
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


def _cdn_chunk_path(season_year: int, cursor: date) -> Path:
    return config.CACHE_DIR / f"cdn_{season_year}_{cursor.isoformat()}.parquet"


def _read_chunk(path: Path) -> pd.DataFrame:
    try:
        cached = pd.read_parquet(path)
    except Exception:  # noqa: BLE001 - an unreadable chunk is simply re-fetched
        return pd.DataFrame()
    return cached if not cached.empty else pd.DataFrame()


def _season_chunks(season_year: int) -> list[Path]:
    """The stored slices of a season, and nothing else.

    The glob is deliberately narrow: a ledger that also started with
    ``cdn_<season>_`` used to be picked up here, and a season that had stored
    no games at all then returned the ledger instead of nothing.
    """
    return sorted(path for path in config.CACHE_DIR.glob(f"cdn_{season_year}_*.parquet")
                  if not path.stem.endswith("_probed"))


def _cached_game_ids(season_year: int) -> set[str]:
    """Every game id already stored for a season, across its chunk files."""
    found: set[str] = set()
    for frame in (_read_chunk(path) for path in _season_chunks(season_year)):
        if not frame.empty and "game_id" in frame.columns:
            found |= set(frame.game_id.astype(str))
    return found


def _probed_path(season_year: int) -> Path:
    """The probe ledger lives OUTSIDE the ``cdn_<season>_`` slice namespace."""
    """Ledger of every id already asked about, hits and 403s alike.

    A 403 is a fact about the upstream as durable as a 200, so without this a
    warm run would re-ask for the ~400 bracket ids that do not exist on every
    single run.
    """
    return config.CACHE_DIR / f"probe_{season_year}.parquet"


def _probed_ids(season_year: int) -> set[str]:
    frame = _read_chunk(_probed_path(season_year))
    if frame.empty or "game_id" not in frame.columns:
        return set()
    return set(frame.game_id.astype(str))


def _save_probed_ids(season_year: int, probed: set[str]) -> None:
    try:
        _probed_path(season_year).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"game_id": sorted(probed)}).to_parquet(
            _probed_path(season_year), index=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not record the probe ledger for %d (%s)",
                       season_year, exc)


def _store_chunk(season_year: int, cursor: date,
                 frames: list[pd.DataFrame]) -> None:
    """Write one slice of the window, keeping every row kind.

    A game contributes a games row, two team rows and one row per player, all
    sharing a game id, so deduplicating on the id alone would throw away the
    team and player rows the models train on.  ``row_kind`` is the part of the
    key that tells them apart; it is dropped again when the frames are read.
    """
    if not frames:
        return
    path = _cdn_chunk_path(season_year, cursor)
    existing = _read_chunk(path)
    fresh = _tag_rows(pd.concat(frames, ignore_index=True))
    merged = (pd.concat([existing, fresh], ignore_index=True)
              if not existing.empty else fresh)
    if "row_kind" in merged.columns:
        merged = merged.drop_duplicates(ROW_KEY).sort_values(ROW_KEY)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not cache %s (%s)", path.name, exc)


def _tag_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Label each row as a game, a team line or a player line."""
    if frame.empty or "row_kind" in frame.columns:
        return frame
    kind = np.where(frame.get("home_score").notna(), "game",
                    np.where(frame.get("player_id").notna(), "player", "team"))
    out = frame.copy()
    out["row_kind"] = kind
    out["player_id"] = out.get("player_id", pd.Series(index=out.index)).astype(object)
    out["team"] = out.get("team", pd.Series(index=out.index)).astype(object)
    return out


def _untag_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["row_kind"], errors="ignore")


def _pull_season_from_cdn(season_year: int, pause: float, start: date,
                          end: date, deadline: float,
                          rerunged: bool = False) -> pd.DataFrame | None:
    """Read every game of a season from per-game box scores, chunk by chunk.

    The regular season is a contiguous run of ids; the postseason is a sparse
    bracket and is enumerated, which is the only way to reach it.  Every id is
    recorded in a probe ledger whether it answered or 403'd, so no id is ever
    asked about twice, and each game is stored in the slice of the window its
    own date falls in, so a killed run costs one slice rather than a season.

    A season where EVERY id is refused is ambiguous - 1,723 absent games and
    1,723 refusals of this client look the same from here - so that verdict is
    settled with one request in a different voice before the season is written
    off.  ``rerunged`` bounds that to a single retry.

    ``deadline`` is the window's absolute wall clock, shared by every season in
    the fallback so the cap bounds the whole walk rather than each season.  A
    walk that runs out returns what it stored, and the probe ledger means the
    next run resumes at the first id it has not seen.
    """
    days = _int_env(CHUNK_DAYS_ENV, CDN_CHUNK_DAYS)
    full = _flag(FULL_REPULL_ENV, False)
    known = set() if full else _cached_game_ids(season_year)
    probed = set() if (full or rerunged) else _probed_ids(season_year)
    logger.info("reading %d-%s from per-game box scores (%d already stored, "
                "%d ids already asked about)", season_year, season_year + 1,
                len(known), len(probed))

    frames: list[pd.DataFrame] = []
    pending: dict[date, list[pd.DataFrame]] = {}
    failures = 0
    stored = 0
    asked = 0
    refused = 0
    first_refusal: str | None = None
    candidates = _candidate_game_ids(season_year)
    for game_id in progress.wrap(candidates, len(candidates),
                                 f"{season_year} box scores", unit="game"):
        if game_id in known or game_id in probed:
            continue
        if time.monotonic() > deadline:
            logger.warning(
                "%d-%s: box-score walk exhausted its budget after %d/%d ids; "
                "the remaining %d will retry next run (raise %s).",
                season_year, season_year + 1, asked, len(candidates),
                len(candidates) - asked, CDN_WALK_BUDGET_ENV)
            break
        probed.add(game_id)
        asked += 1
        try:
            payload = _get_json(BOXSCORE_URL.format(game_id=game_id),
                                allow_missing=True, pause=pause, verbose=False)
        except HostBlocked as exc:
            # The host's verdict landed mid-walk.  That is not a failure of
            # this season: anything already stored is real, and the walk stops
            # at the bottom of the function with the chunks it wrote.  Counting
            # it as a failure here would throw away games this run did get,
            # because MAX_CONSECUTIVE_FAILURES is 1.
            logger.warning("%d-%s: stopping the box-score walk after %d ids "
                           "and %d stored: %s", season_year,
                           season_year + 1, asked, stored, exc)
            break
        except Exception as exc:  # noqa: BLE001 - the walk decides what to do
            probed.discard(game_id)
            failures += 1
            logger.error("NBA box score %s failed: %s", game_id, exc)
            if failures >= MAX_CONSECUTIVE_FAILURES:
                raise CdnUnavailable(
                    f"cdn.nba.com stopped answering after {failures} box "
                    f"scores in {season_year}-{season_year + 1} "
                    f"({game_id}): {exc}") from exc
            continue
        failures = 0
        built = _frames_from_boxscore(payload)
        if built is None:
            refused += 1
            first_refusal = first_refusal or BOXSCORE_URL.format(game_id=game_id)
            if stored == 0 and _refusals_conclusive("cdn.nba.com"):
                # A season in which nothing at all has been stored, and a host
                # that has now refused every id asked of it, has one more thing
                # to prove before it is written off: that this is about the
                # client rather than about the artifacts.  One request in
                # another voice decides that.  Asking the remaining 1,700 ids
                # first would decide nothing.
                if not rerunged and first_refusal and _reprobe_refusal(
                        first_refusal, "cdn.nba.com"):
                    return _pull_season_from_cdn(season_year, pause, start,
                                                 end, deadline, rerunged=True)
                _record_refusal_verdict("cdn.nba.com",
                                        _HOST_REFUSALS.get("cdn.nba.com", 0))
                raise CdnUnavailable(
                    f"cdn.nba.com refused all {refused} box scores asked for in "
                    f"{season_year}-{season_year + 1}, including a re-probe "
                    "with different headers: the host answers instantly and "
                    "serves nothing to this client. It is blocked from this "
                    "machine the same way stats.nba.com is, so the per-game "
                    "CDN cannot rescue the run here.")
            continue
        frame = pd.concat(built, ignore_index=True)
        gameday = pd.to_datetime(frame.gameday.iloc[0], errors="coerce")
        if pd.isna(gameday):
            logger.warning("box score %s has no usable date; not stored", game_id)
            continue
        stored += 1
        cursor = _chunk_start(gameday, start, days)
        pending.setdefault(cursor, []).append(frame)
        frames.append(frame)
        if stored % 100 == 0:
            logger.info("  %d-%s: %d games read", season_year,
                        season_year + 1, stored)
            _save_probed_ids(season_year, probed)
        # Store per slice, so a crash costs one slice rather than a season.
        for finished in sorted(pending):
            if finished < cursor:
                _store_chunk(season_year, finished, pending.pop(finished))
    for finished, ready in sorted(pending.items()):
        _store_chunk(season_year, finished, ready)
    _save_probed_ids(season_year, probed)
    if asked:
        logger.info("%d-%s: %d box scores asked about, %d stored, %d refused",
                    season_year, season_year + 1, asked, stored, refused)
    if stored == 0 and asked > 20 and refused == asked:
        # Every single id was refused. That is usually a blocked host rather
        # than a schedule, but "usually" is not good enough to throw away a
        # season on: one request in the next profile decides it, and the ids
        # come back unasked so nothing is double-counted.
        if not rerunged and first_refusal and _reprobe_refusal(first_refusal,
                                                               "cdn.nba.com"):
            return _pull_season_from_cdn(season_year, pause, start, end,
                                         deadline, rerunged=True)
        raise CdnUnavailable(
            f"cdn.nba.com refused all {asked} box scores for "
            f"{season_year}-{season_year + 1}, including a re-probe with "
            "different headers: the host answers instantly and serves nothing "
            "to this client. It is blocked from this machine the same way "
            "stats.nba.com is, so the per-game CDN cannot rescue the run here.")
    if frames:
        return pd.concat(frames, ignore_index=True)
    cached = [_read_chunk(path) for path in _season_chunks(season_year)]
    cached = [frame for frame in cached if not frame.empty]
    return (_untag_rows(pd.concat(cached, ignore_index=True)) if cached else None)


def _cdn_seasons_in(start: date, end: date) -> list[int]:
    """Season start years whose schedule can overlap the window.

    A season runs from late October to the following June, so walking one
    whose games all fall outside the window costs a thousand requests and
    yields nothing.
    """
    first = start.year if start.month >= 10 else start.year - 1
    return [year for year in range(first, end.year + 1)
            if date(year, 10, 1) <= end and date(year + 1, 6, 30) >= start]


def _report_chunk_gaps(games: pd.DataFrame, start: date, end: date) -> list[str]:
    """Name the window slices that came back with no games at all.

    A silent gap is how a run ends up training on a hole and looking fine, so
    the slices are logged with their counts exactly as the MLB Statcast pull
    reports its chunks.  Offseason and still-future slices are excluded: an
    empty July means nobody played, not that something went missing.

    The scan is deliberately at ``GAP_SCAN_DAYS`` and not at the pull's own
    slice width: the pull can be asked for 60-day slices to match the MLB pull,
    but a detector that only looks every 60 days can walk straight past a
    fortnight of missing games.
    """
    days = GAP_SCAN_DAYS
    gamedays = pd.to_datetime(games.get("gameday"), errors="coerce").dropna()
    cursor = start
    empty: list[str] = []
    with progress.track(_slice_count(start, end, days),
                        desc="NBA gap scan", unit="slice") as bar:
        while cursor <= end:
            _, chunk_end = _chunk_bounds(cursor, end, days)
            # Half-open: a game on the boundary belongs to this slice only, or
            # every slice would also count its neighbour's first day.
            count = int(((gamedays >= pd.Timestamp(cursor))
                         & (gamedays < pd.Timestamp(chunk_end)
                            + pd.Timedelta(days=1))).sum())
            core = _is_core_season_chunk(cursor, chunk_end)
            logger.info("  chunk %s -> %s: %d games%s", cursor, chunk_end, count,
                        "" if core else " (edge of the season)")
            bar.set_postfix(f"{chunk_end} {count}g")
            if not count and core:
                empty.append(f"{cursor}->{chunk_end}")
            cursor = chunk_end + timedelta(days=1)
            bar.update(1)
    return empty


def _abort_on_empty_core_chunks(empty: list[str], start: date,
                                end: date) -> None:
    """Refuse to train on a window with a hole in the middle of a season.

    An empty core-season slice means games are missing, not that the league
    was idle, and training on the remainder would quietly shrink the frame the
    models are scored on.  Past-dated only: a slice still in progress is
    legitimately empty until the games are played.
    """
    if not empty:
        return
    past = [name for name in empty
            if date.fromisoformat(name.split("->")[0]) < date.today()]
    if not past:
        return
    raise RuntimeError(
        f"NBA window {start}..{end} has {len(past)} empty core-season slice(s) "
        f"after the CDN pull: {', '.join(past)}. Games are missing from a part "
        "of the season, so the run is stopped rather than trained on a hole.")


def _pull_seasons_from_cdn(start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame,
                                                           pd.DataFrame, dict[str, str]]:
    """Build the normalized frames from cached per-game box scores alone."""
    years = _cdn_seasons_in(start, end)
    pause = _float_env(PAUSE_ENV, DEFAULT_PAUSE_SEC)
    # One budget for the whole fallback, not one per season: a window of five
    # seasons is five walks, and a cap that resets per walk is no cap at all.
    deadline = time.monotonic() + _float_env(CDN_WALK_BUDGET_ENV,
                                             DEFAULT_CDN_WALK_BUDGET_SEC)
    logger.warning("NBA pulling %s..%s via cdn.nba.com in %d-day slices "
                   "(stats.nba.com answered nothing)", start, end,
                   _int_env(CHUNK_DAYS_ENV, CDN_CHUNK_DAYS))
    collected: list[pd.DataFrame] = []
    for year in years:
        try:
            frame = _pull_season_from_cdn(year, pause, start, end, deadline)
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
    blob = _untag_rows(pd.concat(collected, ignore_index=True))
    if "home_score" not in blob.columns or not len(blob):
        raise RuntimeError(
            f"The cdn.nba.com pull for {start}..{end} returned {len(blob)} "
            f"row(s) with no game data (columns: {sorted(map(str, blob.columns))[:8]}). "
            "That means every box score request was refused or unreadable, so "
            "cdn.nba.com is blocked from this host the same way "
            "stats.nba.com is. Set NBA_CDN_FALLBACK=0 to fail fast instead.")
    games = blob[blob.home_score.notna()].drop_duplicates("game_id")
    games = games[(pd.to_datetime(games.gameday) >= pd.Timestamp(start))
                  & (pd.to_datetime(games.gameday) <= pd.Timestamp(end)
                     + pd.Timedelta(days=1))]
    _abort_on_empty_core_chunks(_report_chunk_gaps(games, start, end), start, end)
    team_stats = blob[blob["team"].notna() & blob["points_for"].notna()]
    team_stats = team_stats[team_stats.game_id.isin(set(games.game_id))]
    if "team_name" not in team_stats.columns:
        team_stats = team_stats.assign(team_name=None)
    names = (team_stats[["team", "team_name"]].dropna(subset=["team_name"])
             .drop_duplicates("team")
             .set_index("team").team_name.to_dict())
    player_stats = blob[blob["player_id"].notna()]
    player_stats = player_stats[player_stats.game_id.isin(set(games.game_id))]
    regular = int((games.game_type == config.GAME_TYPE_REG).sum())
    logger.warning(
        "NBA rebuilt from CDN box scores: %d games (%d regular, %d postseason, "
        "classified from the game id), %d team rows, %d player rows",
        len(games), regular, len(games) - regular,
        len(team_stats), len(player_stats))
    return (games.reset_index(drop=True), team_stats.reset_index(drop=True),
            player_stats.reset_index(drop=True), names)


def _manifest(facts: NBAFacts, start: date, end: date) -> dict[str, Any]:
    games = facts.games
    dates = pd.to_datetime(games.gameday, errors="coerce")
    seasons = pd.to_numeric(games.season, errors="coerce").dropna()
    return {
        "source_id": SOURCE_ID,
        "source_version": f"through-{end.isoformat()}",
        "source_route": SOURCE_USED.get("source", SOURCE_ID),
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "tables": {name: int(len(frame))
                   for name, frame in (("games", games),
                                       ("team_stats", facts.team_stats),
                                       ("player_stats", facts.player_stats),
                                       ("play_by_play", facts.play_by_play))},
        "schemas": {name: [str(c) for c in frame.columns]
                    for name, frame in (("games", games),
                                        ("team_stats", facts.team_stats),
                                        ("player_stats", facts.player_stats),
                                        ("play_by_play", facts.play_by_play))},
        "coverage": {
            "min_date": None if dates.dropna().empty else str(dates.min().date()),
            "max_date": None if dates.dropna().empty else str(dates.max().date()),
            "seasons": sorted(seasons.unique().tolist()),
            "games": int(len(games)),
            "teams": sorted(set(games.home_team) | set(games.away_team)),
        },
        "team_names": facts.team_names,
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


def load_ingested(use_cache: bool = True,
                 allow_download: bool = True) -> NBAFacts:
    """Pull, cache, and normalize the NBA data the model stack consumes.

    The pipeline reads live APIs and its own cache, and nothing else, so there
    is no source directory to point it at.  ``allow_download=False`` keeps the
    run entirely on cached data, which is what ``--skip-pull`` means.
    """
    paths = {name: path for name, path in zip(
        ("games", "team_stats", "player_stats", "play_by_play"), _cache_paths()[:4])}
    manifest_path = _cache_paths()[4]

    start, end = _window()
    if not allow_download:
        logger.info("serving NBA data from %s without a live pull",
                    config.CACHE_DIR)
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

    play_by_play = _pull_play_by_play(
        games, start, end, paths["play_by_play"],
        enabled=allow_download and _flag(PLAY_BY_PLAY_ENV, True))

    facts = NBAFacts(games, team_stats, player_stats, team_names, {}, play_by_play)
    facts.manifest = _manifest(facts, start, end)
    _validate_ingested(facts)
    _write_cache({"games": games, "team_stats": team_stats,
                  "player_stats": player_stats, "play_by_play": play_by_play},
                 paths,
                 {"games": ["game_id"], "team_stats": ["game_id", "team"],
                  "player_stats": ["game_id", "player_id"],
                  "play_by_play": ["game_id", "action_id"]})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(facts.manifest, indent=2, default=str))
    return facts


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
        if not _flag(CDN_FALLBACK_ENV, True):
            raise RuntimeError(
                f"NBA returned no games for {start}..{end}; every season log "
                f"failed ({', '.join(unavailable) or 'none attempted'}). The "
                "upstream endpoint is stats.nba.com/stats/LeagueGameLog, which "
                "needs no key but can be slow or blocked by cloud hosts. Retry, or "
                "narrow NBA_START_DATE/NBA_END_DATE to a window whose seasons are "
                f"already cached, or set {CDN_FALLBACK_ENV}=0 to use the CDN.")
        # The one remaining fallback. cdn.nba.com carries no season log, so
        # this is one request per game instead of one per season - slow, but it
        # is still NBA.com, and it is the last surface that answers at all.
        logger.warning("No season log could be read; rebuilding the window "
                       "from per-game box scores instead.")
        try:
            result = _pull_seasons_from_cdn(start, end)
        except (CdnUnavailable, RuntimeError) as exc:
            raise RuntimeError(
                f"NBA could not read {start}..{end} from any source. "
                f"stats.nba.com: every season log failed "
                f"({', '.join(unavailable) or 'none attempted'}). "
                f"cdn.nba.com box scores {_failure_verdict(exc)}."
                ". A host that never answers is a network block, not a bad "
              "request: nothing this pipeline sends will change it, so run "
              "where these hosts are reachable, or warm the cache here. "
            + _NO_SOURCE_REMEDY) from exc
        SOURCE_USED["source"] = "cdn.nba.com box scores"
        return result
    log = pd.concat(logs, ignore_index=True)
    SOURCE_USED["source"] = "stats.nba.com LeagueGameLog"
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

    It is enrichment, not a dependency: the window, the features and every
    artifact train and publish without it, and player-derived features fall
    back to their defaults.  So a host that refuses the sweep is a reason to
    stop asking, never a reason to stop the run — which is the whole difference
    between degrading and hanging.  A host already recorded as refused is not
    asked at all, and a verdict that lands mid-sweep ends it rather than
    escaping as an exception.
    """
    if games.empty or not enabled:
        return _read_cache(path)
    cached = _read_cache(path)
    have = set(cached.game_id.astype(str)) if not cached.empty else set()
    host = urllib.parse.urlparse(PLAY_BY_PLAY_URL).netloc
    blocked = _known_blocked_host(host)
    if blocked is not None:
        logger.warning(
            "play-by-play skipped: %s %s, and the verdict is still fresh; "
            "keeping the %d games already cached", host, blocked, len(have))
        return cached
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
    asked = 0
    # A bulk walk, so it says so: one INFO line per request buries the summary,
    # and the bar is the progress signal instead.  This is the same rule the
    # per-game box-score walk already follows.
    with progress.track(len(missing), desc="play-by-play", unit="game") as bar:
        for row in missing.itertuples(index=False):
            if _known_blocked_host(host) is not None:
                # Another route already established this run's verdict.
                logger.warning("play-by-play stopped after %d of %d games: %s",
                               asked, len(missing), _known_blocked_host(host))
                break
            try:
                payload = _get_json(PLAY_BY_PLAY_URL.format(game_id=row.game_id),
                                    allow_missing=True, retries=2, pause=pause,
                                    verbose=False)
            except HostBlocked as exc:
                # A recorded verdict surfaces here.  This is enrichment: the
                # run continues without it.
                logger.warning("play-by-play stopped after %d of %d games: %s",
                               asked, len(missing), exc)
                break
            asked += 1
            frame = _play_by_play_frame(payload, str(row.game_id), row.gameday)
            if frame.empty:
                failures += 1
            else:
                frames.append(frame)
            if _refusals_conclusive(host) and not frames:
                # Every game asked for so far came back empty, and the host has
                # refused every one of them.  There is no profile left to try
                # and no artifact to wait for: asking for the other 3,400
                # would only make this line true slower.  Record the verdict so
                # the rest of the run inherits the finding.
                _record_refusal_verdict(host, _HOST_REFUSALS.get(host, 0))
                logger.warning(
                    "play-by-play stopped after %d of %d games: %s has refused "
                    "every one of them. Player features fall back to their "
                    "defaults; the run continues.", asked, len(missing), host)
                break
            bar.set_postfix(f"{len(frames)} ok {failures} empty")
            bar.update(1)
            time.sleep(pause)
    if failures:
        logger.warning("play-by-play empty for %d of %d requested games",
                       failures, asked)
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


def load_games(use_cache: bool = True) -> pd.DataFrame:
    return load_ingested(use_cache).games


def load_team_stats(use_cache: bool = True) -> pd.DataFrame:
    return load_ingested(use_cache).team_stats


def load_player_stats(use_cache: bool = True) -> pd.DataFrame:
    return load_ingested(use_cache).player_stats
