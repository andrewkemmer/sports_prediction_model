"""NBA data acquisition, rewritten around the division of labour MLB already runs.

Three upstreams, each asked for the one thing it is actually good at, and no
fallback route between them. The rewrite exists because the previous version
had one vendor with two hostnames pretending to be two routes.

**ESPN scoreboard owns the schedule.**  ``_fetch_schedule`` sweeps one request
per day across the window. It is the only source here that can answer for a
game nobody has played yet, which is what a pending slate *is*; stats.nba.com
reports no future games at all. It also supplies the result, so the model never
has to decide a winner from a narration.

**stats.nba.com ``LeagueGameLog`` owns the features.**  One request per season
and season type returns every player line of that season - 26,306 rows for
2024-25. Nothing else reaches the same data in fewer requests, because a
per-game endpoint is one request per game.

**stats.nba.com ``playbyplayv3`` owns the play-by-play.**  One request per
game, about 0.1s. The response is ``{game: {actions: [...]}}``, NOT the
``resultSets`` envelope every other ``/stats/`` endpoint uses, and
``StartPeriod``/``EndPeriod`` are mandatory - omitting them is an HTTP 500.
The counted rollup is a declared contract frame (``team_events``), so
play-by-play is part of the model rather than a parquet nobody reads.

Three things this module deliberately does **not** do, each of which was a bug
or a dead end in the version it replaces:

* **No session priming as a fix.**  An earlier version treated a primed
  ``www.nba.com`` cookie as the reason a local run succeeded against
  stats.nba.com, and shipped that as a remedy for a Kaggle timeout. It was
  wrong: the season log, ``playbyplayv3`` and the CDN all answer with bare
  headers and no cookie at all, on this host, in under a quarter second. A
  cookie can therefore be sent if one is free, but nothing here depends on it,
  and a timeout is a host problem, not a handshake problem.
* **No per-game box-score walk as a fallback route.**  The CDN was a second
  hostname for the same vendor wearing a fallback's clothes, and it was the
  thing that made a blocked host look like a flaky pipeline.
* **No feature fallback.**  If stats.nba.com cannot be read, the run fails and
  says so. A second vendor's column names silently becoming the model's is a
  worse failure than a stopped run, and ``source_contract.py`` exists to make
  that seam visible if one is ever added deliberately.

The cache is per-source and per-key: a day's schedule, a season's log, a game's
play-by-play. A failed or partial sweep therefore re-fetches only what it did
not get, which is what makes a 3,461-game play-by-play sweep affordable to run
incrementally.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import config
import nba_sources as sources
import source_contract as contract

logger = logging.getLogger("nba_ingestion")

# ---------------------------------------------------------------------------
# Environment switches. Each one exists because a run on a different host
# needs a different budget, and each names what it bounds.
# ---------------------------------------------------------------------------

WINDOW_START_ENV = "NBA_START_DATE"
WINDOW_END_ENV = "NBA_END_DATE"
CACHE_DIR_ENV = "NBA_CACHE_DIR"
FULL_REPULL_ENV = "NBA_FULL_REPULL"
PBP_ENABLED_ENV = "NBA_FETCH_PLAY_BY_PLAY"
PBP_LOOKBACK_ENV = "NBA_PBP_LOOKBACK_DAYS"
PBP_BUDGET_ENV = "NBA_PBP_BUDGET_SEC"
PBP_MAX_GAMES_ENV = "NBA_PBP_MAX_GAMES"
SCHEDULE_BUDGET_ENV = "NBA_SCHEDULE_BUDGET_SEC"
REQUEST_TIMEOUT_ENV = "NBA_REQUEST_TIMEOUT_SEC"
REQUEST_ATTEMPTS_ENV = "NBA_REQUEST_ATTEMPTS"
PBP_PAUSE_SEC_ENV = "NBA_PBP_PAUSE_SEC"

#: A single request's ceiling. 90s is deliberately long: a slow answer that
#: eventually arrives is worth waiting for, while the retry policy below means a
#: timeout costs two minutes of a budget that is itself bounded.
DEFAULT_TIMEOUT_SEC = 90.0
DEFAULT_ATTEMPTS = 3
#: Seconds between play-by-play requests. 3,461 games at 0.1s each is six
#: minutes of transfer; the pause is what keeps the host from deciding an
#: unusually fast client is a scraper.
DEFAULT_PBP_PAUSE_SEC = 0.25
#: Default ceiling on the play-by-play sweep for one run. A game's event
#: features are trailing, so they only carry signal where the sweep reaches;
#: 40 games out of a 1,200-game window leaves 98% of the rows with no event
#: history at all. The sweep is therefore sized to cover a real share of the
#: window, and the cache is keyed per game so the cost is paid once rather than
#: per run.
DEFAULT_PBP_MAX_GAMES = 1500
DEFAULT_PBP_BUDGET_SEC = 5400.0
DEFAULT_SCHEDULE_BUDGET_SEC = 1800.0
#: How far back the play-by-play sweep reaches for a cold cache. A team-event
#: feature needs history behind it to have any trailing value, so the window is
#: the recent past rather than the whole window.
DEFAULT_PBP_LOOKBACK_DAYS = 240

#: The window's own default, in days back from today. The schedule sweep asks
#: ESPN one question per day, so this is what bounds a cold schedule pull.
DEFAULT_WINDOW_DAYS = 900

#: The User-Agent is a BARE ``Mozilla/5.0``, and that is load-bearing rather
#: than lazy. A full browser impersonation - the string the previous version of
#: this pipeline sent - is refused by both upstreams on this host:
#:
#: * ``stats.nba.com`` accepts the connection and then never answers. The read
#:   times out at 35s with no status line, which is what an edge that has
#:   decided to tarpit a client looks like from Python. It is also exactly the
#:   symptom reported from a remote host, which is why this is worth pinning
#:   rather than leaving to chance.
#: * ``site.api.espn.com`` answers HTTP 403 immediately.
#:
#: With the bare agent both answer in about 0.15s. So the request is not the
#: problem and neither host is blocked; the costume was. Anything that re-adds
#: a browser string here will reintroduce a failure that presents as an
#: unreachable host, so a test asserts this value.
USER_AGENT = "Mozilla/5.0"

STATS_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}

#: ESPN's scoreboard is a different vendor behind a different edge, and it
#: needs none of stats.nba.com's tokens. The one header the two share is the
#: user agent, and for the same reason.
ESPN_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

#: The route this run is recorded as having taken, for the run manifest. Kept
#: as a module constant because the manifest is read by things that do not
#: import the ingestion module's internals.
SOURCE_ID = "espn schedule + stats.nba.com LeagueGameLog + playbyplayv3"

#: The declared source for each frame. A reader can see the division of labour
#: without tracing a single call.
SOURCE_USED = {name: spec.route for name, spec in sources.FRAME_SOURCE.items()}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HostUnavailable(RuntimeError):
    """A host did not answer, and retrying did not help."""


class ScheduleUnavailable(RuntimeError):
    """ESPN's scoreboard could not be read for the window."""


class SeasonLogUnavailable(RuntimeError):
    """stats.nba.com's season log could not be read."""


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


@dataclass
class NBAFacts:
    """Everything the model stack reads, in contract frames.

    ``play_by_play`` is the raw action list and ``team_events`` its per-team
    rollup. Both are kept: the rollup is what the ladder consumes, and the raw
    actions are what makes a new event feature possible without another 3,461
    requests.
    """

    games: pd.DataFrame
    team_stats: pd.DataFrame
    player_stats: pd.DataFrame
    team_events: pd.DataFrame
    play_by_play: pd.DataFrame
    team_names: dict[str, str]
    manifest: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    try:
        value = float(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _int_env(name: str, default: int) -> int:
    try:
        value = int(float(str(os.environ.get(name, "")).strip()))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _to_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    stamp = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(stamp) else stamp.date()


def window() -> tuple[date, date]:
    """The inclusive date window this run covers.

    Defaults to the first eligible season through today, which is what the
    model's training window is defined over. The schedule sweep is one request
    per day, so the start is also the thing that bounds a cold pull.
    """
    today = date.today()
    start_raw = _to_date(os.environ.get(WINDOW_START_ENV))
    end_raw = _to_date(os.environ.get(WINDOW_END_ENV))
    start = start_raw or (today - timedelta(days=DEFAULT_WINDOW_DAYS))
    start = max(start, date(config.OOF_FIRST_SEASON, 1, 1))
    end = end_raw or today
    if end < start:
        logger.warning("%s (%s) precedes %s (%s); swapping",
                       WINDOW_END_ENV, end, WINDOW_START_ENV, start)
        start, end = end, start
    return start, end


def season_label(when: date) -> str:
    """``2024-10-20`` -> ``2024-25``; the league's season turns over in July."""
    first = when.year if when.month >= 7 else when.year - 1
    return f"{first}-{str(first + 1)[2:]}"


def season_starts(start: date, end: date) -> list[int]:
    """Every season-start year whose span overlaps the window."""
    years: list[int] = []
    for first in range(start.year - 1, end.year + 1):
        if date(first, 7, 1) > end or date(first + 1, 6, 30) < start:
            continue
        if first not in years:
            years.append(first)
    return years


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _decompress(raw: bytes, encoding: str | None) -> bytes:
    """Undo content-encoding the stdlib will not undo for us.

    urllib applies no transfer decoding, so asking for gzip in the request
    obliges us to undo it in the response. Forgetting produces a JSON parse
    error whose first bytes are the gzip magic number, which is a confusing
    way to learn that.
    """
    if (encoding or "").lower() not in {"gzip", "x-gzip"}:
        return raw
    try:
        return gzip.decompress(raw)
    except OSError:
        return raw


def http_json(url: str, headers: dict[str, str] | None = None,
              *, timeout: float | None = None,
              attempts: int | None = None) -> Any:
    """GET a JSON document, retrying only what is worth retrying.

    Retried: timeouts, connection resets, and 5xx. Not retried: 4xx. A 400 or
    a 404 means the request is wrong, and repeating it unchanged wastes a
    budget and delays the failure that would have explained it. A 500 from
    these endpoints is genuinely ambiguous - ``leaguedashplayerstats`` answers
    500 for a bad enum - so it is retried once and then reported verbatim,
    because the status text is the only clue about which of the two it was.
    """
    timeout = _float_env(REQUEST_TIMEOUT_ENV,
                         DEFAULT_TIMEOUT_SEC if timeout is None else timeout)
    attempts = _int_env(REQUEST_ATTEMPTS_ENV,
                        DEFAULT_ATTEMPTS if attempts is None else attempts)
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers=dict(headers or {}))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = _decompress(response.read(),
                                   response.headers.get("Content-Encoding"))
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            last = exc
            retryable = exc.code >= 500
            if not retryable:
                raise HostUnavailable(
                    f"{url} answered HTTP {exc.code} {exc.reason}; a request "
                    f"this pipeline sends will not change it") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt < attempts:
            backoff = min(2.0 ** attempt, 8.0)
            logger.info("  %s failed (%s); retry %d/%d in %.0fs",
                        url.split("?")[0], _short(last), attempt, attempts,
                        backoff)
            time.sleep(backoff)
    raise HostUnavailable(
        f"{url.split('?')[0]} did not answer after {attempts} attempt(s): "
        f"{_short(last)}")


def _short(exc: Exception | None) -> str:
    if exc is None:
        return "no error recorded"
    text = str(exc).strip() or type(exc).__name__
    return f"{type(exc).__name__}: {text[:160]}"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_dir() -> Any:
    override = os.environ.get(CACHE_DIR_ENV, "").strip()
    return Path(override).expanduser() if override else Path(config.CACHE_DIR)


def _read_parquet(path) -> pd.DataFrame:
    try:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read %s (%s); treating it as empty",
                       path.name, exc)
        return pd.DataFrame()


def _write_parquet(frame: pd.DataFrame, path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Normalize before writing, not only after reading: a frame written by
        # an older adapter holds different dtypes, and concatenating the two
        # produces a mixed column that ``to_parquet`` refuses with a type error
        # that names neither cause. This makes the cache self-healing.
        frame.to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not write %s (%s); it will be refetched",
                       path.name, exc)


def _merge_cache(old: pd.DataFrame, new: pd.DataFrame,
                 keys: list[str]) -> pd.DataFrame:
    """Union two frames on ``keys``, newest row wins."""
    if old is None or old.empty:
        return new.reset_index(drop=True) if new is not None else pd.DataFrame()
    if new is None or new.empty:
        return old.reset_index(drop=True)
    combined = pd.concat([old, new], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=keys, keep="last")
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. The schedule - ESPN
# ---------------------------------------------------------------------------


def _schedule_path(day: date):
    return _cache_dir() / "schedule" / f"{day:%Y%m%d}.parquet"


def _fetch_schedule(start: date, end: date) -> pd.DataFrame:
    """Every game in the window, from ESPN, one request per day.

    MLB's ``_espn_events_for_et_date`` solves the same problem for baseball by
    probing two UTC days and filtering on the converted Eastern date; that
    fix is applied inside the adapter instead, so a caller here only has to ask
    for an ET date and get that date's games. An empty day is cached as an
    empty frame rather than retried forever: most days in a window are genuinely
    game-free, and treating those as failures would report a broken host.
    """
    if _flag(FULL_REPULL_ENV):
        logger.info("%s is set; every schedule day is refetched", FULL_REPULL_ENV)
    deadline = time.time() + _float_env(SCHEDULE_BUDGET_ENV,
                                        DEFAULT_SCHEDULE_BUDGET_SEC)
    days = (end - start).days + 1
    frames: list[pd.DataFrame] = []
    fetched = cached = 0
    for offset in range(days):
        if time.time() > deadline:
            logger.warning("schedule sweep hit its budget after %d of %d days; "
                           "the rest is cached and the next run continues",
                           offset, days)
            break
        day = start + timedelta(days=offset)
        path = _schedule_path(day)
        existing = pd.DataFrame() if _flag(FULL_REPULL_ENV) else _read_parquet(path)
        if not existing.empty or path.exists():
            cached += 1
            frames.append(existing)
            continue
        try:
            payload = http_json(sources.espn_scoreboard_url(day), ESPN_HEADERS,
                                timeout=25.0, attempts=2)
        except HostUnavailable as exc:
            logger.warning("ESPN scoreboard unavailable for %s: %s", day,
                           _short(exc))
            continue
        day_frame = sources.espn_schedule_frame(payload.get("events", []))
        day_frame = day_frame.assign(_day=day)
        _write_parquet(day_frame, path)
        fetched += 1
        if not day_frame.empty:
            frames.append(day_frame)
    if not frames:
        return pd.DataFrame()
    games = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    games = games.drop_duplicates("game_id", keep="last")
    logger.info("schedule: %d games over %d days (%d fetched, %d from cache)",
                len(games), days, fetched, cached)
    return games


# ---------------------------------------------------------------------------
# 2. The features - stats.nba.com LeagueGameLog
# ---------------------------------------------------------------------------


def _season_log_path(season: str, season_type: str):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", season_type)
    return _cache_dir() / "season_logs" / f"log_{season}_{slug}.parquet"


def _fetch_season_logs(start: date, end: date) -> pd.DataFrame:
    """Every player line in the window, one request per season and season type.

    There is no fallback and no partial substitute. If a season log cannot be
    read the run reports which one failed and stops, because a feature set
    assembled from a different set of games each run is not a feature set - it
    is a series of unrelated models wearing one name.
    """
    seasons = season_starts(start, end)
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for first in seasons:
        label = f"{first}-{str(first + 1)[2:]}"
        for season_type, game_type in ((sources.SEASON_TYPE_REGULAR,
                                        config.GAME_TYPE_REG),
                                       (sources.SEASON_TYPE_PLAYOFFS,
                                        config.GAME_TYPE_POST)):
            path = _season_log_path(label, season_type)
            if _flag(FULL_REPULL_ENV):
                _write_parquet(pd.DataFrame(), path)
                try:
                    path.unlink()
                except OSError:
                    pass
            cached = _read_parquet(path)
            if not cached.empty:
                logger.info("%s %s: %d rows from cache", label, season_type,
                            len(cached))
                frames.append(cached)
                continue
            url = (f"{sources.SEASON_LOG_URL}?"
                   f"{sources.season_log_query(label, season_type)}")
            try:
                payload = http_json(url, STATS_HEADERS, timeout=90.0, attempts=3)
            except HostUnavailable as exc:
                failures.append(f"{label} {season_type}: {_short(exc)}")
                logger.error("stats.nba.com season log unavailable, %s %s: %s",
                             label, season_type, _short(exc))
                continue
            log = _result_set_to_frame(payload)
            if log.empty:
                # A season type that has not started answers with zero rows. That
                # is an answer, not a failure, and caching it stops the next run
                # asking again for a season nobody has played.
                logger.info("%s %s: no rows (the season type has not started)",
                            label, season_type)
            else:
                logger.info("%s %s: %d player rows", label, season_type, len(log))
            log = sources.rename_log(log)
            log = log.assign(game_type=game_type)
            _write_parquet(log, path)
            frames.append(log)
    if failures:
        raise SeasonLogUnavailable(
            "stats.nba.com could not return the season log for "
            f"{len(failures)} season-type(s): {'; '.join(failures)}. "
            "There is no second feature source, so a missing season log is a "
            "stopped run rather than a thinner one. Re-run where stats.nba.com "
            "is reachable, or warm the cache.")
    if not frames:
        return pd.DataFrame()
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


def _result_set_to_frame(payload: Any) -> pd.DataFrame:
    """``resultSets[0]`` into a frame, whichever header shape came back.

    stats.nba.com answers some endpoints with ``headers`` as a list of strings
    and others as a list of objects carrying ``columnNames``. Assuming either
    one produces a frame of the wrong width and a downstream error about a
    column that should exist.
    """
    result_sets = (payload or {}).get("resultSets") or []
    if not result_sets:
        return pd.DataFrame()
    first = result_sets[0]
    rows = first.get("rowSet") or []
    if not rows:
        return pd.DataFrame()
    headers = first.get("headers")
    if isinstance(headers, list) and headers and isinstance(headers[0], dict):
        columns = headers[0].get("columnNames") or []
    else:
        columns = list(headers or [])
    if not columns:
        columns = [f"col{i}" for i in range(len(rows[0]))]
    width = len(columns)
    # A short row is padded rather than dropped: dropping loses a player line
    # silently, and a padded one becomes a NaN that the coverage report sees.
    padded = [list(r) + [None] * (width - len(r)) if len(r) < width else list(r)[:width]
              for r in rows]
    return pd.DataFrame(padded, columns=columns)


# ---------------------------------------------------------------------------
# 3. Play-by-play - stats.nba.com playbyplayv3
# ---------------------------------------------------------------------------


def _pbp_path(nba_game_id: str):
    return _cache_dir() / "play_by_play" / f"pbp_{nba_game_id}.parquet"


def _fetch_play_by_play(games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The action list for the games in the window, one request per game.

    Cached per game, which is what makes this affordable: a cold run sweeps the
    most recent slice, and every later run adds the games that have since been
    played. The budget is a deadline, not a count, so a slow host costs less
    data rather than the whole run.

    Only games the season log has already identified are requested. A game the
    log has not seen has no ``nba_game_id`` because nobody has played it, and a
    scheduled game has no play-by-play to fetch.
    """
    info: dict[str, Any] = {"enabled": _flag(PBP_ENABLED_ENV, True),
                            "requested": 0, "cached": 0, "fetched": 0,
                            "failed": 0, "unattributed_rebounds": 0}
    if not info["enabled"]:
        logger.info("%s is off; play-by-play is not fetched", PBP_ENABLED_ENV)
        return pd.DataFrame(), info
    eligible = (games.loc[games.get("nba_game_id", pd.Series(dtype=str)).astype(str).str.len() > 0]
                .drop_duplicates("nba_game_id")
                .sort_values("gameday"))
    if eligible.empty:
        logger.warning("no game in the window has an NBA game id, so there is "
                       "no play-by-play to fetch")
        return pd.DataFrame(), info

    lookback = _int_env(PBP_LOOKBACK_ENV, DEFAULT_PBP_LOOKBACK_DAYS)
    cap = _int_env(PBP_MAX_GAMES_ENV, DEFAULT_PBP_MAX_GAMES)
    cutoff = pd.Timestamp(date.today() - timedelta(days=lookback))
    recent = eligible[pd.to_datetime(eligible.gameday) >= cutoff]
    if recent.empty:
        # A window that is entirely older than the lookback would otherwise
        # sweep nothing and report a clean run, which is the worst possible
        # outcome: the budget is spent and the feature is silently absent.
        # Falling back to the most recent games in the window keeps a
        # historical or backtest run populated.
        logger.info("no game in the window is within %d days of today; "
                    "sweeping the most recent %d instead", lookback, cap)
        recent = eligible
    targets = recent.tail(cap)
    logger.info("play-by-play: %d candidate game(s), sweeping %d",
                len(eligible), len(targets))

    deadline = time.time() + _float_env(PBP_BUDGET_ENV, DEFAULT_PBP_BUDGET_SEC)
    pause = _float_env(PBP_PAUSE_SEC_ENV, DEFAULT_PBP_PAUSE_SEC)
    frames: list[pd.DataFrame] = []
    for number, (_, row) in enumerate(targets.iterrows(), start=1):
        nba_id = str(row.nba_game_id)
        path = _pbp_path(nba_id)
        existing = _read_parquet(path)
        if not existing.empty:
            info["cached"] += 1
            frames.append(existing)
            continue
        if time.time() > deadline:
            logger.warning("play-by-play sweep hit its budget at %d of %d "
                           "games; the rest is cached and the next run "
                           "continues", number - 1, len(targets))
            break
        info["requested"] += 1
        url = (f"{sources.PLAY_BY_PLAY_URL}?"
               f"{sources.play_by_play_query(nba_id)}")
        try:
            payload = http_json(url, STATS_HEADERS, timeout=45.0, attempts=2)
        except HostUnavailable as exc:
            info["failed"] += 1
            # Log sparsely: a budget-limited sweep can hit thousands of these.
            if info["failed"] <= 3 or info["failed"] % 50 == 0:
                logger.warning("play-by-play unavailable for %s: %s", nba_id,
                               _short(exc))
            continue
        actions = sources.play_by_play_actions(payload, nba_id, row.gameday)
        if actions.empty:
            logger.warning("play-by-play for %s returned no actions", nba_id)
            info["failed"] += 1
            continue
        info["unattributed_rebounds"] += sources.unattributed_rebounds(actions)
        _write_parquet(actions, path)
        frames.append(actions)
        info["fetched"] += 1
        if pause:
            time.sleep(pause)
    logger.info("play-by-play: %d fetched, %d from cache, %d unavailable",
                info["fetched"], info["cached"], info["failed"])
    if not frames:
        return pd.DataFrame(), info
    return pd.concat(frames, ignore_index=True), info


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _attach_game_ids(games: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    """Join the schedule to the season log on (date, and the two teams).

    This is the seam where the two vendors meet. The key is deliberately
    ORIENTATION-FREE - a date plus the unordered pair of teams - because the
    season log's matchup text cannot be trusted for which side was home. It
    spells the matchup from each row's own team's perspective, and for 5 games
    in the 2024-25 log it uses ``@`` on both sides, so ``WAS @ MIA`` appears on
    WAS's rows even though WAS is home. Keying on the ordered pair therefore
    fails to match those games, and a rule that guesses which spelling to keep
    inverts the sides of every game it guesses about - which is a schedule that
    looks entirely normal and poisons the 65-point home advantage in Elo.

    So the log contributes only the NBA game id and the schedule contributes
    home/away, and the key is the one fact both upstreams agree on: these two
    teams played on this date. A team plays at most once a day, so that key is
    unique, which is what makes the one-to-one validation meaningful.
    """
    if log.empty or games.empty:
        return games.assign(nba_game_id="")
    log_games = sources.games_from_log(log)
    if log_games.empty:
        return games.assign(nba_game_id="")
    keys = log_games[["pair_key", "nba_game_id"]]
    if keys.pair_key.duplicated().any():
        dupes = int(keys.pair_key.duplicated().sum())
        logger.warning("the season log reports %d date/team-pair key(s) twice; "
                       "the schedule join may multiply rows", dupes)
    out = games.drop(columns=["nba_game_id"], errors="ignore").copy()
    out["pair_key"] = [sources.pair_key(g, a, b) for g, a, b in
                       zip(out.gameday, out.home_team, out.away_team)]
    merged = out.merge(keys, on="pair_key", how="left", validate="one_to_one")
    missing = int((merged.nba_game_id.fillna("") == "").sum())
    if missing:
        logger.warning("%d scheduled game(s) have no season-log match; they "
                       "have no player lines and no play-by-play", missing)
    known = set(merged.nba_game_id.dropna().astype(str)) - {""}
    orphaned = log_games[~log_games.nba_game_id.astype(str).isin(known)]
    if len(orphaned):
        # Only worth reporting when the schedule covered the log's dates. In a
        # short window the log returns whole seasons, so most of it is outside
        # the schedule on purpose and saying so would be noise.
        in_window = orphaned[
            pd.to_datetime(orphaned.gameday).dt.normalize().between(
                pd.to_datetime(out.gameday).min(),
                pd.to_datetime(out.gameday).max())]
        if len(in_window):
            logger.warning("%d game(s) in the season log fall inside the "
                           "schedule window but are absent from it; the "
                           "schedule is missing those dates", len(in_window))
    return merged.drop(columns=["pair_key"])


def _with_contract_ids(log: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Put the contract's ``game_id`` on the log rows, and drop the rest.

    The log only knows stats.nba.com's game id. Adding the schedule's id once,
    here, is what lets the player frame and the team rollup key on the same
    identity the published artifacts use.

    Rows that found no schedule game are DROPPED rather than kept with an empty
    id. A season log returns whole seasons, so a short window leaves most of
    its rows unmatched; keeping them would put thousands of feature rows with
    no game into the frames, and they would all share one empty ``game_id`` -
    which is indistinguishable, to every duplicate check, from thousands of
    genuinely duplicated players.
    """
    mapping = (games[["nba_game_id", "game_id"]]
               .dropna(subset=["nba_game_id"])
               .drop_duplicates("nba_game_id"))
    out = log.drop(columns=["game_id"], errors="ignore").merge(
        mapping, on="nba_game_id", how="left")
    out["game_id"] = out.game_id.fillna("").astype(str)
    unmatched = int((out.game_id == "").sum())
    if unmatched:
        logger.info("%d season-log row(s) belong to games outside the schedule "
                    "window and are not part of this run", unmatched)
    return out[out.game_id != ""].reset_index(drop=True)


def _team_names(games: pd.DataFrame) -> dict[str, str]:
    """Longest name seen per abbreviation, for the published artifacts."""
    names: dict[str, str] = {}
    for abbr, full in (("ATL", "Atlanta Hawks"), ("BOS", "Boston Celtics"),
                       ("BKN", "Brooklyn Nets"), ("CHA", "Charlotte Hornets"),
                       ("CHI", "Chicago Bulls"), ("CLE", "Cleveland Cavaliers"),
                       ("DAL", "Dallas Mavericks"), ("DEN", "Denver Nuggets"),
                       ("DET", "Detroit Pistons"), ("GSW", "Golden State Warriors"),
                       ("HOU", "Houston Rockets"), ("IND", "Indiana Pacers"),
                       ("LAC", "LA Clippers"), ("LAL", "Los Angeles Lakers"),
                       ("MEM", "Memphis Grizzlies"), ("MIA", "Miami Heat"),
                       ("MIL", "Milwaukee Bucks"), ("MIN", "Minnesota Timberwolves"),
                       ("NOP", "New Orleans Pelicans"), ("NYK", "New York Knicks"),
                       ("OKC", "Oklahoma City Thunder"), ("ORL", "Orlando Magic"),
                       ("PHI", "Philadelphia 76ers"), ("PHX", "Phoenix Suns"),
                       ("POR", "Portland Trail Blazers"), ("SAC", "Sacramento Kings"),
                       ("SAS", "San Antonio Spurs"), ("TOR", "Toronto Raptors"),
                       ("UTA", "Utah Jazz"), ("WAS", "Washington Wizards")):
        if abbr in set(games.get("home_team", pd.Series(dtype=str)).astype(str)) | \
                   set(games.get("away_team", pd.Series(dtype=str)).astype(str)):
            names[abbr] = full
    return names


def eligible_games(games: pd.DataFrame) -> pd.DataFrame:
    """Games the model may train or predict on.

    Both game types the contract declares are kept - regular season and
    playoffs - because ``is_playoffs`` is a model feature. Anything outside them
    is dropped, along with rows with no identity, since a feature row that
    cannot be joined to a schedule card is not a game.
    """
    if games is None or games.empty:
        return pd.DataFrame()
    out = games.copy()
    out = out[out.game_id.astype(str).str.len() > 0]
    out = out[pd.to_numeric(out.game_type, errors="coerce").isin(
        config.GAME_TYPES)]
    return out.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def trainable_games(games: pd.DataFrame) -> pd.DataFrame:
    """Settled games that have feature rows to learn from.

    A finished game is only trainable if the season log has player lines for
    it, which is exactly what an ``nba_game_id`` records. Some finished games
    have none, and they are not edge cases in a schedule:

    * a **postponed** game ESPN kept on the calendar;
    * the **NBA Cup final**, played at a neutral site and therefore absent from
      ``LeagueGameLog`` under ``Regular Season`` (the 2024 one is the
      Bucks-Thunder game on 2024-12-17, which ESPN reports as a 97-81 final);
    * the **All-Star** games, which ESPN files as ``regular-season`` and whose
      squads (KEN, CHK, CAN, SHQ) are not league teams at all.

    Every one of these looks finished and settled, so a pipeline that splits on
    the score alone turns it into a training row whose every feature is NaN -
    and NaN is a value a model will fit to rather than reject. The requirement
    is general rather than a list of known exceptions: a game nobody in the
    season log played cannot become a training label.
    """
    if games is None or games.empty:
        return pd.DataFrame()
    out = games.copy()
    settled = out[out.home_score.notna() & out.away_score.notna()]
    has_lines = settled.get("nba_game_id", pd.Series("", index=settled.index))
    keep = settled[has_lines.astype(str).str.len() > 0]
    missing = settled[has_lines.astype(str).str.len() == 0]
    if len(missing):
        logger.info(
            "%d finished game(s) have no season-log player lines and so no "
            "features; they are excluded from training. These are the "
            "postponed, neutral-site and all-star games, not a join failure: "
            "%s", len(missing),
            ", ".join(
                f"{pd.to_datetime(r.gameday):%Y-%m-%d} {r.away_team}@{r.home_team}"
                for r in missing.head(6).itertuples()))
    return keep.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def _validate(facts: NBAFacts) -> None:
    """Refuse a fact set the model cannot be trained on, naming what is wrong."""
    games = facts.games
    if games.empty:
        raise RuntimeError("NBA schedule returned no games; nothing to train on")
    settled = games[games.home_score.notna() & games.away_score.notna()]
    if len(settled) < max(10, config.MIN_VAL_FOLD_GAMES):
        raise RuntimeError(
            f"NBA window has only {len(settled)} settled game(s); the model "
            f"needs at least {max(10, config.MIN_VAL_FOLD_GAMES)} for a "
            "walk-forward fold")
    if facts.team_stats.empty:
        raise RuntimeError(
            "no team facts were built; the season log did not join to the "
            "schedule. Check that both upstreams spell the teams the same way "
            "- a join on team name drops every game involving a renamed team "
            "with no error anywhere.")
    if facts.player_stats.empty:
        raise RuntimeError("no player lines were read from the season log")
    # The identity of a row is not the same in every frame: games are one row
    # per game, team facts are one per team per game, and player lines are one
    # per player per game. Checking ``game_id`` alone would call a correct
    # two-row team frame a duplicate and stop every run.
    for name, frame, keys in (("games", games, ["game_id"]),
                              ("team_stats", facts.team_stats,
                               ["game_id", "team"]),
                              ("player_stats", facts.player_stats,
                               ["game_id", "player_id"])):
        if frame.duplicated(keys).any():
            dupes = int(frame.duplicated(keys).sum())
            raise RuntimeError(
                f"the {name} frame has {dupes} duplicated "
                f"{' + '.join(keys)} row(s); a feature row that appears "
                "twice is counted twice by every trailing window")
    if not facts.team_events.empty:
        duplicated = facts.team_events.duplicated(["game_id", "team"]).sum()
        if duplicated:
            raise RuntimeError(
                f"the play-by-play rollup has {duplicated} duplicated "
                "(game_id, team) row(s); two actions for one team-game would "
                "double its counts")


def load_ingested(use_cache: bool = True, allow_download: bool = True,
                  start: date | None = None, end: date | None = None,
                  ) -> NBAFacts:
    """Read the whole NBA fact set: schedule, features, play-by-play.

    ``allow_download=False`` builds from the cache alone, which is how a host
    that cannot reach either upstream still runs the model against a window
    that was fetched elsewhere.
    """
    window_start, window_end = (start, end) if start and end else window()
    logger.info("NBA window %s..%s (%s)", window_start, window_end,
                "download allowed" if allow_download else "cache only")

    if not allow_download:
        # A host that cannot reach either upstream can still run the model
        # against a window fetched elsewhere. This is the cache-warm path, and
        # it is the reason the cache is per-source and per-key rather than one
        # file per run.
        games = _read_schedule_only(window_start, window_end)
        log = _read_logs_only(window_start, window_end)
        pbp = _read_pbp_only()
        pbp_info: dict = {"mode": "cache only"}
    else:
        games = _fetch_schedule(window_start, window_end)
        log = _fetch_season_logs(window_start, window_end)
        if games.empty:
            raise ScheduleUnavailable(
                f"no NBA schedule for {window_start}..{window_end}. The "
                "schedule comes from ESPN's scoreboard and there is no second "
                "source, so a run that cannot read it stops here rather than "
                "training on a partial league.")
        if log.empty:
            raise SeasonLogUnavailable(
                "no season log was read, so the window has no features")
        # The NBA game ids live in the season log, so the play-by-play sweep
        # cannot start until the log has been joined to the schedule. Asking
        # for play-by-play first would request nothing at all and report a
        # clean sweep.
        games = _attach_game_ids(games, log)
        log = _with_contract_ids(log, games)
        pbp, pbp_info = _fetch_play_by_play(games)

    if games.empty:
        raise ScheduleUnavailable(
            f"no NBA schedule for {window_start}..{window_end}")
    if log.empty:
        raise SeasonLogUnavailable("no season log was read, so the window has "
                                   "no features")
    if not allow_download:
        games = _attach_game_ids(games, log)
        log = _with_contract_ids(log, games)

    normalized_games = contract.normalize(games, "games", source="espn")
    team_stats = contract.normalize(
        sources.team_stats_from_log(log, games), "team_stats",
        source="stats.nba.com")
    player_stats = contract.normalize(
        log.drop(columns=["nba_game_id"], errors="ignore"),
        "player_stats", source="stats.nba.com")

    team_events = pd.DataFrame()
    normalized_pbp = pd.DataFrame()
    pbp_report: dict = {}
    if pbp is not None and not pbp.empty:
        # The actions are keyed by the NBA game id they were fetched with; the
        # contract's id is ESPN's, so it arrives from the schedule here.
        context = games[["nba_game_id", "game_id", "gameday", "home_team",
                         "away_team", "home_score", "away_score"]].dropna(
                             subset=["nba_game_id"])
        keyed = pbp.merge(context, on="nba_game_id", how="inner",
                          suffixes=("", "_sched"))
        if not keyed.empty:
            normalized_pbp = contract.normalize(
                keyed.drop(columns=["nba_game_id"], errors="ignore"),
                "play_by_play", source="stats.nba.com")
            team_events = contract.normalize(
                sources.team_events_from_actions(
                    keyed.merge(context, on="nba_game_id", how="inner",
                                suffixes=("", "_ctx")), games),
                "team_events", source="stats.nba.com")
            pbp_report = sources.cross_check(team_events, team_stats)
    pbp_info["cross_check"] = pbp_report

    facts = NBAFacts(
        games=normalized_games, team_stats=team_stats,
        player_stats=player_stats, team_events=team_events,
        play_by_play=normalized_pbp,
        team_names=_team_names(normalized_games),
        manifest={
            "source_id": SOURCE_ID,
            "source_route": SOURCE_ID,
            "frame_sources": dict(SOURCE_USED),
            "window": [str(window_start), str(window_end)],
            "seasons": [f"{y}-{str(y + 1)[2:]}" for y in
                        season_starts(window_start, window_end)],
            "tables": {name: int(len(frame)) for name, frame in (
                    ("games", normalized_games), ("team_stats", team_stats),
                    ("player_stats", player_stats),
                    ("team_events", team_events),
                    ("play_by_play", normalized_pbp))},
            "play_by_play": pbp_report,
            "feature_set_version": config.FEATURE_SET_VERSION,
        })
    _validate(facts)
    logger.info("NBA facts: %d games, %d team rows, %d player rows, "
                "%d play-by-play actions, %d event rows",
                len(normalized_games), len(team_stats), len(player_stats),
                len(normalized_pbp), len(team_events))
    return facts


def _read_schedule_only(start: date, end: date) -> pd.DataFrame:
    frames = []
    for offset in range((end - start).days + 1):
        frame = _read_parquet(_schedule_path(start + timedelta(days=offset)))
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _read_logs_only(start: date, end: date) -> pd.DataFrame:
    frames = []
    for first in season_starts(start, end):
        label = f"{first}-{str(first + 1)[2:]}"
        for season_type in (sources.SEASON_TYPE_REGULAR,
                            sources.SEASON_TYPE_PLAYOFFS):
            frame = _read_parquet(_season_log_path(label, season_type))
            if not frame.empty:
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _read_pbp_only() -> pd.DataFrame:
    root = _cache_dir() / "play_by_play"
    if not root.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(root.glob("pbp_*.parquet")):
        frame = _read_parquet(path)
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_games(**kwargs) -> pd.DataFrame:
    return load_ingested(**kwargs).games


def load_team_stats(**kwargs) -> pd.DataFrame:
    return load_ingested(**kwargs).team_stats


def load_player_stats(**kwargs) -> pd.DataFrame:
    return load_ingested(**kwargs).player_stats


def load_play_by_play(**kwargs) -> pd.DataFrame:
    return load_ingested(**kwargs).play_by_play
