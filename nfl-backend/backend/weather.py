"""Point-in-time Open-Meteo weather for NFL games.

Production rule (mirrors the MLB implementation):

* past games use Open-Meteo's hourly archive, with the forecast endpoint's
  recent observed-past window as a lag fallback;
* current and future games use Open-Meteo's hourly forecast;
* the selected API timestamp is the latest hour **strictly before kickoff**;
* a forecast row is usable only if it was fetched strictly before kickoff;
* indoor/closed games, unknown venues, invalid kickoffs, and missing rows stay
  unavailable (NaN), never climatology, a daily aggregate, or schedule data.

The cache stores both the API source and the weather's valid timestamp, plus
the fetch time.  Production reads that cache through ``features.py``, which
revalidates every row at the feature boundary.
"""
from __future__ import annotations

import functools
import logging
import os
import random
import time
from datetime import date, datetime, timedelta, timezone

# ingestion owns StageProgress (the run-wide progress bar). Importing it here
# is acyclic: ingestion never imports weather. A local import keeps the module
# import graph unchanged for anything that imports weather first.
from ingestion import StageProgress  # noqa: E402
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:  # package-relative import when used as backend.weather
    from backend import config
except ImportError:  # running as a top-level module
    import config

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = "temperature_2m,wind_speed_10m,precipitation,snowfall"
HOURLY_KEYS = ("time", "temperature_2m", "wind_speed_10m",
               "precipitation", "snowfall")

OPEN_METEO_ARCHIVE = "open_meteo_archive"
OPEN_METEO_FORECAST_PAST = "open_meteo_forecast_past"
OPEN_METEO_FORECAST = "open_meteo_forecast"
PIT_WEATHER_SOURCES = frozenset({
    OPEN_METEO_ARCHIVE,
    OPEN_METEO_FORECAST_PAST,
    OPEN_METEO_FORECAST,
})

ROOF_OUTDOOR = "outdoor"
ROOF_RETRACTABLE = "retractable"
ROOF_DOME = "dome"
VENUE_ROOF_VALUES = frozenset({ROOF_OUTDOOR, ROOF_RETRACTABLE, ROOF_DOME})
# Schedule roof states that describe the game itself (nflverse).
SCHEDULE_OPEN_ROOFS = ("outdoors", "outdoor", "open", "retractable")
SCHEDULE_CLOSED_ROOFS = ("dome", "closed")

CACHE_VERSION = "v1"
CACHE_FILE = f"weather_pit_{CACHE_VERSION}.parquet"
CACHE_COLUMNS = [
    "game_id", "stadium", "kickoff_utc", "weather_time_utc",
    "fetched_at_utc", "source", "temp_f", "wind_mph", "precip_in",
    "snow_in", "is_precip", "is_snow",
]
REQUIRED_CACHE_COLUMNS = frozenset(CACHE_COLUMNS)

# Open-Meteo archive window: 14 days, matching MLB's _WEATHER_BATCH_DAYS
# exactly (also 15 locations per batch; the PAUSE is deliberately slower
# than MLB's 1.0s -- see _BATCH_PAUSE_SEC below).  60 is MLB's
# SCHEDULE_CHUNK_DAYS -- the StatsAPI pull, not the weather pull -- and is
# carried in ingestion.POPULATE_CHUNK_DAYS as the reporting interval. The two
# constants answer different questions and are deliberately not the same
# number: 60-day weather windows were measured and work (a decade of history
# drops from 357 to 111 request batches), but the 14-day cadence is the
# proven one and wins on request size and rate-limit headroom, so it stays.
_BATCH_DAYS = 14
_BATCH_SIZE = 15
# Sustained sweep pace.  The 2026-10-11 15:50 run measured the trip rate:
# 444 batches in 689s (~39 requests/min with the old 1.0s pause) interleaved
# 429 bursts roughly once a minute with clean stretches, even though
# Open-Meteo's documented free-tier limits (600/min, 5000/hr, 10000/day) are
# enforced PER IP -- and Kaggle's egress IP is shared with every other
# tenant hitting the same endpoint, so the effective budget under burst is
# far below the documented one.  2.0s between batches halves our contribution
# (~23 req/min; 444 batches ~= 19 min, still well under the ~30-minute storm
# cost this breaker replaced) and buys headroom on that shared IP.
_BATCH_PAUSE_SEC = 2.0
_RETRIABLE_STATUSES = frozenset({429, 502, 503, 504})
# Gateway errors (502/503/504) and network blips answer quick retries: the
# MLB-parity ladder of 2**attempt + jitter across 7 attempts stays right for
# those.  429 is NOT that kind of error -- see _get_with_retry: a 429 is a
# quota minute closing, and only the two quick retries below are spent on
# the marginal case before the cooldown/probe takes over.
_RETRY_ATTEMPTS = 7
_RETRY_BASE_SEC = 2.0
_RETRY_JITTER_SEC = 0.5
# --- Rate-limit breaker (2026-10-11 run-log reviews, first + second pass) --
# The per-request ladder above is right for an ISOLATED blip, but a closed
# quota window answers every FOLLOWING batch with 429 as well, and each batch
# re-armed the full ladder on its own: the reviewed run spent 156 retry
# requests across 25 back-to-back dead batches (05:30:35 -> 05:39:42), and
# the storms grow run over run (13 -> 72 -> 119 -> 156 retries over
# 10-07 -> 10-11).  Requests cannot land while the window is shut;
# hammering only extends the penalty, and 7 attempts x a 45s timeout per
# batch is how a rate-limited sweep becomes a wall-clock timeout risk.
# The breaker trades hammering for waiting and bounds total attempts:
#   * a batch dead to a retriable status OPENS the window: the next request
#     cools down for one quota minute (60s), doubling per consecutive dead
#     batch to a 4-minute cap, then probes ONCE -- never a ladder;
#   * any definitive (non-retriable) response CLOSES the window (streak 0);
#   * _429_RETRY_BUDGET caps rate-limited retries for the whole process;
#     past it, a rate-limited request fails after a single attempt;
#   * _RATE_LIMIT_ABORT_STREAK consecutive dead batches stop the REQUEUE
#     pass below with a loud count of what was skipped -- the same-run
#     top-up and the next run's cache-miss pass refetch exactly the games
#     that stay missing (features stay NaN, never substituted).
# Third review (same day, evening 18:10 run -- the first WITH the breaker):
# the breaker behaved as designed (9 dead batches, one cooldown each,
# streak 10, under the abort line, full 2022/2022 coverage) yet the
# shared-IP window flapped shut for ~37 minutes anyway and the escalating
# cooldowns stretched a ~26-minute sweep to 51m28s -- sleeping through
# closed windows is itself the wall-clock risk this breaker exists to
# remove.  The archive sweep therefore never sleeps on a 429 at all:
# _get_with_retry(defer_on_429=True) returns the batch immediately (at
# most the two marginal quick retries when the streak was 0) and the
# sweep DEFERS it into a requeue list; every subsequent paced batch
# doubles as the probe that notices a reopened window, so a flap costs
# ~one batch interval (~4s) per deferred batch instead of 60-240s of
# cooldown, and the sweep runs to completion at its steady pace no
# matter how long the window stays shut.  One paced re-attempt per
# deferred batch follows the sweep (_requeue_deferred_batches), and the
# same-run top-up resets the streak first so the run's final recovery
# re-probes a window that may have reopened.
_RATE_LIMIT_COOLDOWN_SEC = 60.0
_RATE_LIMIT_COOLDOWN_MAX_SEC = 240.0
_RATE_LIMIT_PROBE_ATTEMPTS = 1
# A 429 can be a MARGINAL moment (the quota nearly full, reopens in seconds)
# or a closed minute.  The 15:50 run measured both: batches that recovered
# after 2 retries waited ~3-9s in total, while quick retries fired into a
# truly closed minute re-429'd every time -- 21 logged retries across 7
# groups, one batch burning all 6 retries (~65s) and dying anyway.  Two quick retries
# cover the marginal case at MLB's ladder shape; past them the window is
# treated as shut: cool down, probe once.
_429_QUICK_RETRIES = 2
# The requeue stops after this many CONSECUTIVE dead batches.  Under defer
# mode a dead batch costs one paced request (~4s), so 12 bounds the requeue
# at roughly a minute of probing into a window that is not coming back;
# anything still deferred flows to the same-run top-up (fresh streak) and
# the next run's cache-miss pass.
_RATE_LIMIT_ABORT_STREAK = 12
_429_RETRY_BUDGET = 40
# One stalled connection is retried; a stalled ENDPOINT is not.  Three
# consecutive timeouts end the request (3 x 20s + backoff ~= 63s) instead
# of the old 7 x 45s = 5+ minutes per dead batch, cascading across a sweep.
# 20s is still 4-10x Open-Meteo's normal answer time for a ~400KB batch.
_REQUEST_TIMEOUT_SEC = 20
_TIMEOUT_RETRY_LIMIT = 3
# Process-local breaker state (the pipeline is single-threaded).
# _reset_rate_limiter() restores it for tests and fresh sweeps.
_rate_limit: dict = {"streak": 0, "retries": 0, "budget_logged": False}
_SNOWFALL_UNIT_TO_INCH = {
    "inch": 1.0,
    "in": 1.0,
    "inches": 1.0,
    "cm": 1.0 / 2.54,
    "mm": 1.0 / 25.4,
    "millimeter": 1.0 / 25.4,
}


def cache_path(path: str | Path | None = None) -> Path:
    """Return the external PIT weather-cache path (override for tests/tools)."""
    if path is not None:
        return Path(path)
    override = os.getenv("NFL_WEATHER_CACHE", "").strip()
    if override:
        return Path(override)
    return Path(config.ROOT_DIR.parent) / ".nfl_cache" / CACHE_FILE


@functools.lru_cache(maxsize=1)
def stadium_locations() -> dict[str, tuple[float, float]]:
    """Load committed venue coordinates, keyed by schedule stadium name."""
    return {name: (info["lat"], info["lon"])
            for name, info in _venue_table().items()}


@functools.lru_cache(maxsize=1)
def stadium_roofs() -> dict[str, str]:
    """Committed static roof classification per schedule stadium name.

    ``outdoor``  - no roof over the field at all (open-air bowl)
    ``retractable`` - roofed venue whose game-day state varies
    ``dome``     - fixed roof

    This is a venue-structure fact, knowable long before any kickoff, so it is
    a legitimate fallback when the schedule's per-game roof field is missing.
    It is NOT evidence of what the roof did on a given day: a ``retractable``
    venue stays unresolved for weather purposes (see :func:`_outdoor`).
    """
    return {name: info["roof"] for name, info in _venue_table().items()
            if info["roof"] in VENUE_ROOF_VALUES}


@functools.lru_cache(maxsize=1)
def _venue_table() -> dict[str, dict]:
    path = config.BACKEND_DIR / "nfl_stadiums.csv"
    if not path.exists():
        return {}
    venues = pd.read_csv(path)
    required = {"stadium", "lat", "lon"}
    if not required <= set(venues.columns):
        return {}
    out: dict[str, dict] = {}
    for row in venues.itertuples(index=False):
        lat = pd.to_numeric(getattr(row, "lat", np.nan), errors="coerce")
        lon = pd.to_numeric(getattr(row, "lon", np.nan), errors="coerce")
        stadium = str(getattr(row, "stadium", "")).strip()
        if not stadium or pd.isna(lat) or pd.isna(lon):
            continue
        out[stadium] = {
            "lat": float(lat), "lon": float(lon),
            "roof": str(getattr(row, "roof", "") or "").strip().lower(),
        }
    return out


def kickoff_utc(games: pd.DataFrame) -> pd.Series:
    """Parse exact schedule kickoff as ET, then convert to UTC; fail closed."""
    out = pd.Series(pd.NaT, index=games.index, dtype="datetime64[ns, UTC]")
    if not {"gameday", "gametime"} <= set(games.columns):
        return out
    day = pd.to_datetime(games["gameday"], errors="coerce")
    clock = games["gametime"].astype("string").str.strip()
    parsed = pd.to_datetime(
        day.dt.strftime("%Y-%m-%d") + " " + clock.fillna(""),
        errors="coerce",
    )
    try:
        return (parsed.dt.tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="NaT"
        ).dt.tz_convert("UTC"))
    except (TypeError, ValueError):
        return out


def schedule_roof(games: pd.DataFrame) -> pd.Series:
    """Normalize the schedule's per-game roof field (game-day evidence)."""
    if games is None or "roof" not in getattr(games, "columns", []):
        return pd.Series(pd.NA, index=getattr(games, "index", None),
                         dtype="string")
    return games["roof"].astype("string").str.strip().str.lower()


def venue_roof(games: pd.DataFrame) -> pd.Series:
    """Committed static roof classification for each row's stadium."""
    roofs = stadium_roofs()
    stadium = (games["stadium"].astype("string").str.strip()
               if "stadium" in getattr(games, "columns", [])
               else pd.Series(pd.NA, index=getattr(games, "index", None),
                              dtype="string"))
    return stadium.map(lambda s: roofs.get(str(s), "") if pd.notna(s) else "")


def is_roofed(games: pd.DataFrame) -> pd.Series:
    """True when the home venue is roofed (fixed or retractable).

    The schedule's own per-game value wins: an explicit ``outdoors``/``open``
    means the game was played with the roof open, even at a roofed venue. Only
    when the schedule is silent does the committed venue classification decide.
    Unknown at both levels stays unknown rather than becoming a guess.
    """
    sched = schedule_roof(games)
    venue = venue_roof(games)
    return (sched.isin(SCHEDULE_CLOSED_ROOFS)
            | (sched.isna() & venue.isin([ROOF_DOME, ROOF_RETRACTABLE]))).fillna(False)


def _outdoor(games: pd.DataFrame) -> pd.Series:
    """Whether outdoor weather provably applies to a game.

    Explicit per-game open-air states are admitted. When the schedule is silent
    the committed venue classification decides ONLY for venues that have no roof
    at all: a retractable or domed venue tells us nothing about the game-day
    state, so it keeps failing closed.
    """
    sched = schedule_roof(games)
    venue = venue_roof(games)
    return (sched.isin(SCHEDULE_OPEN_ROOFS)
            | (sched.isna() & venue.eq(ROOF_OUTDOOR))).fillna(False)


def outdoor_mask(games: pd.DataFrame) -> pd.Series:
    """Public alias of the outdoor-eligibility rule (feature boundary uses it)."""
    return _outdoor(games)


def _targets(games: pd.DataFrame) -> list[dict]:
    """Build fetch/cache targets only for outdoor games with exact kickoffs."""
    required = {"game_id", "stadium"}
    if games is None or games.empty or not required <= set(games.columns):
        return []
    kickoff = kickoff_utc(games)
    outdoor = _outdoor(games)
    locations = stadium_locations()
    rows: list[dict] = []
    for idx, game in games.iterrows():
        ko = kickoff.loc[idx]
        stadium = str(game.get("stadium", "")).strip()
        if (pd.isna(ko) or not outdoor.loc[idx] or not stadium
                or stadium not in locations):
            continue
        rows.append({
            "game_id": str(game["game_id"]),
            "stadium": stadium,
            "kickoff_utc": ko,
            "lat": locations[stadium][0],
            "lon": locations[stadium][1],
        })
    # A stable key is required for cache lookups. Duplicate game IDs are
    # ambiguous and therefore fail closed rather than choosing one schedule.
    counts = pd.Series([r["game_id"] for r in rows]).value_counts()
    duplicate_ids = set(counts[counts > 1].index)
    return [row for row in rows if row["game_id"] not in duplicate_ids]


def _reset_rate_limiter() -> None:
    """Restore breaker state (tests, or a caller starting a fresh sweep)."""
    _rate_limit.update(streak=0, retries=0, budget_logged=False)


def _rate_limit_streak() -> int:
    """Consecutive requests that died without a definitive answer."""
    return int(_rate_limit["streak"])


def _cooldown_wait(streak: int) -> float:
    """Seconds to quiet down before the next request: one quota window,
    doubling per consecutive dead batch, capped at four minutes."""
    return float(min(_RATE_LIMIT_COOLDOWN_SEC * (2 ** (streak - 1)),
                     _RATE_LIMIT_COOLDOWN_MAX_SEC))


def _rate_limited_out() -> bool:
    """True once consecutive dead batches reach the sweep-abort line."""
    return _rate_limit_streak() >= _RATE_LIMIT_ABORT_STREAK


def _get_with_retry(url: str, params: dict, attempts: int = _RETRY_ATTEMPTS,
                    timeout: int = _REQUEST_TIMEOUT_SEC, *,
                    defer_on_429: bool = False):
    """GET with bounded retries for transient Open-Meteo errors.

    Gateway errors (502/503/504) and network exceptions keep the MLB-parity
    ladder: ``2**attempt`` + jitter across ``attempts`` rounds, or the
    server's ``Retry-After`` when it sends one (whichever is longer); every
    other status returns immediately (fail fast).  A retriable answer that
    spends the ladder still counts as a dead batch for the streak below.

    429 is a quota event, not a blip (2026-10-11 second run-log review; the
    15:50 run burned 21 logged 429 retries on 1/2/4s ladder waits that can only land
    inside the same closed quota minute, and one batch burned all six
    retries (~65s) and died anyway):

      * up to ``_429_QUICK_RETRIES`` short retries ride out a MARGINAL
        window (the observed recoveries answered within ~3-9s);
      * a still-closed window OPENS the rate-limit streak and buys ONE
        cooldown (one quota minute, doubling per consecutive dead batch,
        capped at four) + one probe.  A probe that 429s is terminal for
        this call; any definitive answer resets the streak to 0.

    One call therefore costs at most ``_429_QUICK_RETRIES + 2`` requests
    and one cooldown; batches that arrive while the window is already shut
    (streak > 0) cool down at entry and probe exactly once.  The per-run
    ``_429_RETRY_BUDGET`` bounds rate-limited attempts outright.

    ``defer_on_429=True`` (the archive sweep's mode; third review,
    2026-10-11 evening run) trades the cooldown/probe stage for DEFERRAL:
    a 429 opens the streak and returns immediately so the caller can
    queue the batch for one paced re-attempt later.  No request in this
    mode ever sleeps on a 429 -- the streak-0 marginal case still gets its
    two quick retries, a streak-open batch costs exactly ONE request, and
    the next paced batch of the sweep is itself the probe that notices a
    reopened window (within one batch interval instead of after a
    60-240s cooldown).  Gateway-error and timeout ladders are identical in
    both modes.
    """
    probe_only = _rate_limit_streak() > 0
    if probe_only and not defer_on_429:
        wait = _cooldown_wait(_rate_limit_streak())
        logger.warning(
            "Open-Meteo rate limited: cooling down %.0fs before the next "
            "request (consecutive dead batches: %d)", wait,
            _rate_limit_streak())
        time.sleep(wait)
        attempts = min(attempts, _RATE_LIMIT_PROBE_ATTEMPTS)
    last_exc: Exception | None = None
    response = None
    timeouts = 0
    quick_429 = 0
    # The cooldown+probe below fires at most ONCE per call: an entry probe
    # (streak already open) or a ladder-exhausted 429 that just opened it.
    probed = probe_only
    attempt = -1
    while True:
        attempt += 1
        try:
            response = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_exc = exc
            timeouts = timeouts + 1 if isinstance(
                exc, (requests.Timeout, requests.ConnectionError)) else 0
            if timeouts >= _TIMEOUT_RETRY_LIMIT:
                # A stalled endpoint is not fixed by four more hangs: end
                # the request and let the dead-batch streak account for it.
                _rate_limit["streak"] = _rate_limit_streak() + 1
                raise
            if attempt < attempts - 1:
                time.sleep(_RETRY_BASE_SEC ** attempt + random.uniform(
                    0, _RETRY_JITTER_SEC))
                continue
            _rate_limit["streak"] = _rate_limit_streak() + 1
            raise last_exc
        if response.status_code in _RETRIABLE_STATUSES:
            if response.status_code == 429:
                budget_spent = _rate_limit["retries"] >= _429_RETRY_BUDGET
                may_quick = (not probe_only
                             and quick_429 < _429_QUICK_RETRIES
                             and not budget_spent)
                if may_quick:
                    quick_429 += 1
                    _rate_limit["retries"] += 1
                    wait = _RETRY_BASE_SEC ** (quick_429 - 1) + random.uniform(
                        0, _RETRY_JITTER_SEC)
                    try:
                        wait = max(wait, float(
                            response.headers.get("Retry-After", "")))
                    except (AttributeError, TypeError, ValueError):
                        pass
                    logger.warning(
                        "Open-Meteo %s; retrying in %.1fs (%d/%d)",
                        response.status_code, wait, quick_429,
                        _429_QUICK_RETRIES)
                    time.sleep(wait)
                    continue
                if budget_spent:
                    if not _rate_limit["budget_logged"]:
                        _rate_limit["budget_logged"] = True
                        logger.warning(
                            "Open-Meteo 429 retry budget exhausted (%d per "
                            "run) — rate-limited requests now fail after "
                            "one attempt; the same-run top-up and the next "
                            "run's cache-miss pass recover the gaps",
                            _429_RETRY_BUDGET)
                    _rate_limit["streak"] = _rate_limit_streak() + 1
                    return response
                if defer_on_429:
                    # Sweep mode: the caller DEFERS this batch (one paced
                    # re-attempt after the sweep) instead of this call
                    # waiting out a quota minute.  Nothing here sleeps.
                    _rate_limit["streak"] = _rate_limit_streak() + 1
                    return response
                if probed:
                    # This call already waited out a quota minute; a second
                    # 429 proves the window is still shut.  Terminal: open
                    # the streak and let the NEXT batch decide whether to
                    # keep probing or trip the sweep abort.
                    _rate_limit["streak"] = _rate_limit_streak() + 1
                    return response
                # Window confirmed shut: open the streak, cool down one
                # quota minute, and let the next loop round be the single
                # probe.  Retry-After stays authoritative when sent.
                probed = True
                _rate_limit["retries"] += 1
                _rate_limit["streak"] = _rate_limit_streak() + 1
                wait = _cooldown_wait(_rate_limit["streak"])
                try:
                    wait = max(wait, float(
                        response.headers.get("Retry-After", "")))
                except (AttributeError, TypeError, ValueError):
                    pass
                logger.warning(
                    "Open-Meteo 429: quota window closed — cooling down "
                    "%.0fs, then one probe (dead batches: %d)", wait,
                    _rate_limit["streak"])
                time.sleep(wait)
                continue
            if attempt < attempts - 1:
                # 502/503/504: gateway trouble -- same ladder, but still a
                # dead answer, so the terminal one counts toward the streak.
                wait = _RETRY_BASE_SEC ** attempt + random.uniform(
                    0, _RETRY_JITTER_SEC)
                logger.warning(
                    "Open-Meteo %s; retrying in %.1fs (%d/%d)",
                    response.status_code, wait, attempt + 1, attempts - 1)
                time.sleep(wait)
                continue
            _rate_limit["streak"] = _rate_limit_streak() + 1
            return response
        # 2xx (or a fail-fast 4xx): the endpoint gave a definitive answer.
        _rate_limit["streak"] = 0
        return response


def _utc_timestamp(value: object) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _split_hourly_by_utc_day(
    hourly: dict[str, list],
    source: str,
    fetched_at_utc: datetime,
    snowfall_unit: str | None = None,
) -> dict[date, dict[str, list]]:
    """Split an Open-Meteo GMT response into exact UTC-day series."""
    times = hourly.get("time", []) or []
    by_day: dict[date, dict[str, list]] = {}
    for i, raw_time in enumerate(times):
        stamp = _utc_timestamp(raw_time)
        if stamp is None:
            continue
        series = by_day.setdefault(
            stamp.date(),
            {"time": [], "_source": source,
             "_fetched_at_utc": fetched_at_utc,
             "_snowfall_unit": snowfall_unit,
             **{key: [] for key in HOURLY_KEYS if key != "time"}},
        )
        series["time"].append(stamp)
        for key in HOURLY_KEYS:
            if key == "time":
                continue
            values = hourly.get(key, []) or []
            series[key].append(values[i] if i < len(values) else None)
    return by_day


def _parse_batch_response(
    payload: object,
    locations: list[tuple[str, float, float]],
    source: str,
    fetched_at_utc: datetime,
) -> dict[tuple[str, date], dict[str, list]]:
    items = payload if isinstance(payload, list) else [payload]
    if len(items) != len(locations):
        logger.warning(
            "Open-Meteo returned %d locations for %d requested",
            len(items), len(locations),
        )
    out: dict[tuple[str, date], dict[str, list]] = {}
    for (stadium, _lat, _lon), item in zip(locations, items):
        if not isinstance(item, dict):
            continue
        hourly = item.get("hourly", {}) or {}
        # Snowfall has no dedicated unit parameter: Open-Meteo returns it in the
        # same unit as precipitation (inch when precipitation_unit=inch, cm by
        # default). Trust the response's own unit string instead of assuming.
        units = item.get("hourly_units", {}) or {}
        snowfall_unit = str(units.get("snowfall", "") or "").strip().lower() or None
        for day, series in _split_hourly_by_utc_day(
                hourly, source, fetched_at_utc, snowfall_unit).items():
            out[(stadium, day)] = series
    return out


def _requeue_deferred_batches(
    deferred: list[tuple[list[tuple[str, float, float]], date, date]],
    by_key: dict[tuple[str, date], dict[str, list]],
) -> None:
    """One paced re-attempt per sweep batch a closed window deferred.

    Each deferred batch gets exactly one more request, at the sweep's pace
    and still in defer mode, so a still-closed window re-defers it without
    sleeping and ``_RATE_LIMIT_ABORT_STREAK`` consecutive dead batches
    stop the pass outright.  Recovered series merge into ``by_key``;
    anything still deferred stays missing and the documented recovery
    paths take it from there -- the same-run top-up refetches the gap
    games with a fresh breaker, and the next run's cache-miss pass
    refetches exactly and only the games whose rows never landed.
    """
    recovered = 0
    attempted = 0
    unattempted = 0
    # 2026-10-13 run-log review: this pass used to inherit the SWEEP's
    # streak, so a sweep that ended at/over the abort line (streak 48 on
    # that run, 18 on the top-up's) tripped the check on index 0 and the
    # "one paced re-attempt" announced on the line above never happened —
    # all 48 deferred batches went unattempted.  The abort line therefore
    # bounds THIS pass on ITS OWN consecutive dead batches (12 paced probes,
    # ~a minute); the global streak keeps describing the sweep for the
    # callers that follow.
    dead_streak = 0
    for index, (batch, chunk_start, chunk_end) in enumerate(deferred):
        if dead_streak >= _RATE_LIMIT_ABORT_STREAK:
            unattempted = len(deferred) - index
            break
        if index:
            time.sleep(_BATCH_PAUSE_SEC)
        result = _fetch_batch_range(
            batch, chunk_start, chunk_end,
            source=OPEN_METEO_ARCHIVE, defer_on_429=True,
        )
        attempted += 1
        if result is None:
            dead_streak += 1
            continue
        dead_streak = 0
        by_key.update(result)
        recovered += 1
    if unattempted:
        logger.warning(
            "Open-Meteo archive requeue: stopping after %d consecutive "
            "dead batches (abort line %d) - %d deferred batch(es) not "
            "re-attempted; missing games recover via the same-run top-up "
            "and the next run's cache-miss pass",
            dead_streak, _RATE_LIMIT_ABORT_STREAK, unattempted)
    elif attempted:
        logger.info(
            "Open-Meteo archive requeue: recovered %d of %d deferred "
            "batch(es); %d stay deferred for the same-run top-up and the "
            "next run's cache-miss pass",
            recovered, attempted, attempted - recovered)


def _fetch_batch_range(
    locations: list[tuple[str, float, float]],
    start_date: date,
    end_date: date,
    *,
    source: str,
    forecast: bool = False,
    past_days: int | None = None,
    forecast_days: int | None = None,
    defer_on_429: bool = False,
) -> dict[tuple[str, date], dict[str, list]] | None:
    """One bounded multi-coordinate request; None = rate-limited, deferred.

    ``defer_on_429`` passes through to :func:`_get_with_retry`; a batch
    that comes back 429 in that mode returns ``None`` (not ``{}``) so the
    sweep can distinguish "deferred by the quota" from "dead for good" and
    requeue it.  Callers that merge with ``by_key.update`` must therefore
    only pass ``defer_on_429`` where they handle ``None``.
    """
    params = {
        "latitude": ",".join(str(info[1]) for info in locations),
        "longitude": ",".join(str(info[2]) for info in locations),
        "hourly": HOURLY_VARS,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "GMT",
    }
    if forecast:
        params.update({
            "past_days": 1 if past_days is None else past_days,
            "forecast_days": 1 if forecast_days is None else forecast_days,
        })
        url = FORECAST_URL
    else:
        params.update({
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        })
        url = ARCHIVE_URL
    try:
        response = _get_with_retry(url, params=params,
                                   timeout=_REQUEST_TIMEOUT_SEC,
                                   defer_on_429=defer_on_429)
        if response.status_code == 429:
            # Rate-limited, never dead-for-good: None in BOTH modes, so a
            # 429 joins the requeue instead of raising into the generic
            # failure handler (which returns {} and silently drops the
            # batch from the deferred list).  The sweep stays quiet — one
            # warning per dead batch would bury the log during a long
            # closure, and the sweep counts its deferrals in a single line —
            # while a non-defer caller (the recovery pass's first batch, the
            # recent/forecast phases) has already logged its cooldown and
            # gets one explicit line here.
            if not defer_on_429:
                logger.warning(
                    "Open-Meteo rate limited after cooldown+probe "
                    "(%s..%s, source=%s) - no data for this batch; missing "
                    "games stay NaN and recover via the requeue, the "
                    "same-run top-up or the next run's cache-miss pass",
                    start_date, end_date, source)
            return None
        response.raise_for_status()
        payload = response.json()
        # Record the actual response time, not the start of a potentially long
        # multi-batch build. A forecast fetched while kickoff was in progress
        # is past evidence, never a pre-kickoff forecast snapshot.
        response_fetched_at = datetime.now(timezone.utc)
        return _parse_batch_response(
            payload, locations, source, response_fetched_at)
    except Exception as exc:  # noqa: BLE001 - missing weather must remain NaN
        logger.warning(
            "Open-Meteo batch failed (%s..%s, %d locations, source=%s): %s [%s]",
            start_date, end_date, len(locations), source, exc,
            ",".join(item[0] for item in locations),
        )
        return {}


def _fetch_batched_weather(
    targets: list[dict],
    now: datetime,
    *,
    cooldown_first_batch: bool = False,
) -> dict[tuple[str, date], dict[str, list]]:
    """Fetch paced archive/recent-past/forecast hourly series for all targets.

    ``cooldown_first_batch`` is the RECOVERY pass's mode (the same-run
    top-up): its first archive batch runs the non-defer client, so a closed
    window costs one bounded cooldown + one probe before the batch defers,
    instead of the whole gap set burning into a closed window at sweep pace
    (2026-10-13 run: 18/18 top-up batches deferred in 39s and 142 games
    still shipped without weather).  The remaining batches keep defer mode —
    the sweep itself never sleeps.
    """
    by_key: dict[tuple[str, date], dict[str, list]] = {}
    if not targets:
        return by_key

    locations_map: dict[str, tuple[str, float, float]] = {}
    for row in targets:
        locations_map.setdefault(
            row["stadium"], (row["stadium"], row["lat"], row["lon"]))
    locations = list(locations_map.values())
    needed_days = {row["kickoff_utc"].date() for row in targets}
    # The prior UTC day is needed for late ET games whose kickoff is just after
    # 00:00 UTC and therefore has no earlier row on its own UTC date.
    start_date = min(needed_days) - timedelta(days=1)
    end_date = max(needed_days)
    today = now.date()

    archive_end = min(end_date, today - timedelta(days=1))
    if start_date <= archive_end:
        # Build the window list FIRST, applying the same end clamp and the
        # same "contains a needed day" filter the loop used to apply inline.
        # The fetched set is therefore identical -- this only moves the
        # decision out of the loop so the progress denominator is exact.
        # Without it this phase is the longest in the run (~7 min on a
        # decade of history, longer when Open-Meteo answers 429) and logged
        # NOTHING but retry warnings, which reads exactly like a hang.
        windows: list[tuple[date, date]] = []
        for offset in range(0, (archive_end - start_date).days + 1, _BATCH_DAYS):
            chunk_start = start_date + timedelta(days=offset)
            chunk_end = min(chunk_start + timedelta(days=_BATCH_DAYS - 1),
                            archive_end)
            if any(chunk_start <= day <= chunk_end for day in needed_days):
                windows.append((chunk_start, chunk_end))
        batches_per_window = max(1, -(-len(locations) // _BATCH_SIZE))
        total_batches = len(windows) * batches_per_window
        logger.info("PIT weather archive: %d window(s) x %d batch(es) of %d "
                    "stadiums = %d request batches",
                    len(windows), batches_per_window, _BATCH_SIZE, total_batches)
        # Reuse the pipeline's tested bar rather than a second hand-rolled
        # one, so the throttling is identical: 384 batches would otherwise
        # print 384 lines and bury the rest of the run's log.
        bar = StageProgress(total_batches, "PIT weather archive batches")
        attempted = 0
        # A 429 here DEFERS the batch instead of killing it (third review,
        # 2026-10-11 evening run): the batch is queued for one paced
        # re-attempt after the sweep, and the sweep immediately sends its
        # next request -- that request IS the quota probe, so a reopened
        # window is detected within one batch interval instead of after a
        # 60-240s cooldown.  The sweep therefore always runs to completion
        # at its steady pace however long the window stays shut; only the
        # requeue below is bounded by the dead-batch streak.
        deferred: list[tuple[list[tuple[str, float, float]], date, date]] = []
        first_batch = True
        for chunk_start, chunk_end in windows:
            for i in range(0, len(locations), _BATCH_SIZE):
                batch = locations[i:i + _BATCH_SIZE]
                result = _fetch_batch_range(
                    batch, chunk_start, chunk_end,
                    source=OPEN_METEO_ARCHIVE,
                    defer_on_429=not (cooldown_first_batch and first_batch),
                )
                first_batch = False
                if result is None:
                    deferred.append((batch, chunk_start, chunk_end))
                else:
                    by_key.update(result)
                attempted += 1
                bar.advance()
                if i + _BATCH_SIZE < len(locations):
                    time.sleep(_BATCH_PAUSE_SEC)
            if chunk_end < archive_end:
                time.sleep(_BATCH_PAUSE_SEC)
        bar.close()
        if deferred:
            logger.info(
                "Open-Meteo: %d of %d archive batches came back 429 and "
                "were deferred — one paced re-attempt follows",
                len(deferred), attempted)
            _requeue_deferred_batches(deferred, by_key)

    # ERA5/archive publication lags. The forecast endpoint's observed past
    # window exposes the same hourly records immediately, with the identical
    # strict pre-kickoff cutoff applied below.
    recent_start = max(start_date, today - timedelta(days=91))
    recent_end = min(end_date, today)
    if recent_start <= recent_end:
        recent_target_days = needed_days & {
            recent_start + timedelta(days=i)
            for i in range((recent_end - recent_start).days + 1)
        }
        missing = any(
            row["kickoff_utc"].date() in recent_target_days
            and (row["stadium"], row["kickoff_utc"].date()) not in by_key
            for row in targets
        )
        if missing:
            # ``recent_start``/``recent_end`` are datetime.date, so %d raises
            # TypeError INSIDE logging: the handler catches it, prints a
            # "--- Logging error ---" traceback, and the line is lost. %s
            # renders the dates and cannot fail.
            logger.info("PIT weather recent: fetching %s..%s observed "
                        "(forecast endpoint) for %d stadiums",
                        recent_start, recent_end, len(locations))
            recent = _fetch_batch_range(
                locations, recent_start, recent_end,
                source=OPEN_METEO_FORECAST,
                forecast=True,
                past_days=(today - recent_start).days + 1,
                forecast_days=1,
            )
            # A rate-limited/failed batch yields None (never data), so the
            # phase skips instead of crashing on .items() (2026-10-13
            # review: the 429 path used to raise into the caller here).
            for key, series in (recent or {}).items():
                by_key.setdefault(key, series)

    # Today/future forecasts are capped at Open-Meteo's supported horizon.
    # Missing rows beyond that horizon remain unavailable, never substituted.
    forecast_start = max(start_date, today)
    if forecast_start <= end_date:
        forecast_days = min((end_date - today).days + 1, 16)
        future = _fetch_batch_range(
            locations, forecast_start, end_date,
            source=OPEN_METEO_FORECAST,
            forecast=True, past_days=1, forecast_days=forecast_days,
        )
        for key, series in (future or {}).items():
            by_key.setdefault(key, series)
    return by_key


def latest_hour_index_before(
    times: list,
    target_utc: datetime | pd.Timestamp,
) -> int:
    """Index of the latest valid hourly timestamp strictly before kickoff."""
    target = _utc_timestamp(target_utc)
    if target is None:
        return -1
    best = -1
    best_stamp: pd.Timestamp | None = None
    for i, raw_time in enumerate(times):
        stamp = _utc_timestamp(raw_time)
        if stamp is None or stamp >= target:
            continue
        if best_stamp is None or stamp > best_stamp:
            best, best_stamp = i, stamp
    return best


def _float(value: object) -> float:
    if value is None:
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def select_weather_record(
    target: dict,
    series: dict[str, list] | None,
    fetched_at_utc: datetime | pd.Timestamp,
) -> dict | None:
    """Select and shape the latest hourly row strictly before one kickoff."""
    if series is None:
        return None
    idx = latest_hour_index_before(
        series.get("time", []) or [], target["kickoff_utc"])
    if idx < 0:
        return None

    values = {}
    for key in HOURLY_KEYS:
        if key == "time":
            continue
        raw = series.get(key, []) or []
        values[key] = _float(raw[idx] if idx < len(raw) else None)
    if not any(pd.notna(value) for value in values.values()):
        return None

    weather_time = _utc_timestamp((series.get("time", []) or [])[idx])
    fetched_at = _utc_timestamp(series.get("_fetched_at_utc") or fetched_at_utc)
    kickoff = _utc_timestamp(target["kickoff_utc"])
    if weather_time is None or fetched_at is None or kickoff is None:
        return None
    if weather_time >= kickoff:
        return None

    base_source = series.get("_source")
    if base_source not in {OPEN_METEO_ARCHIVE, OPEN_METEO_FORECAST}:
        return None
    if base_source == OPEN_METEO_ARCHIVE:
        source = OPEN_METEO_ARCHIVE
    elif kickoff <= fetched_at:
        # Today's earlier games are observed rows from the forecast endpoint's
        # past window, never a forecast reconstructed after kickoff.
        source = OPEN_METEO_FORECAST_PAST
    else:
        source = OPEN_METEO_FORECAST
    if source == OPEN_METEO_FORECAST and fetched_at >= kickoff:
        return None

    precip = values["precipitation"]
    snowfall = values["snowfall"]
    if pd.notna(snowfall):
        # An unrecognized unit is not guessed: the feature stays unavailable.
        factor = _SNOWFALL_UNIT_TO_INCH.get(series.get("_snowfall_unit"))
        snow_in = snowfall * factor if factor is not None else np.nan
    else:
        snow_in = np.nan
    return {
        "game_id": str(target["game_id"]),
        "stadium": str(target["stadium"]),
        "kickoff_utc": kickoff,
        "weather_time_utc": weather_time,
        "fetched_at_utc": fetched_at,
        "source": source,
        "temp_f": values["temperature_2m"],
        "wind_mph": values["wind_speed_10m"],
        "precip_in": precip,
        "snow_in": snow_in,
        "is_precip": (float(precip >= config.PRECIP_FLAG_IN)
                      if pd.notna(precip) else np.nan),
        "is_snow": (float(snow_in >= config.SNOW_FLAG_IN)
                    if pd.notna(snow_in) else np.nan),
    }


def _empty_weather() -> pd.DataFrame:
    return pd.DataFrame(columns=CACHE_COLUMNS)


def _validate_cache_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Return only unambiguous, provenance-complete, strictly-PIT cache rows."""
    if frame is None or frame.empty or not REQUIRED_CACHE_COLUMNS <= set(frame.columns):
        return _empty_weather()
    cache = frame[list(CACHE_COLUMNS)].copy()
    for col in ("kickoff_utc", "weather_time_utc", "fetched_at_utc"):
        cache[col] = pd.to_datetime(cache[col], errors="coerce", utc=True)
    cache["source"] = cache["source"].astype("string")
    valid = (
        cache["game_id"].notna() & cache["game_id"].astype(str).ne("")
        & cache["stadium"].notna() & cache["stadium"].astype(str).ne("")
        & cache["kickoff_utc"].notna() & cache["weather_time_utc"].notna()
        & cache["fetched_at_utc"].notna()
        & cache["source"].isin(PIT_WEATHER_SOURCES)
        & (cache["weather_time_utc"] < cache["kickoff_utc"])
    )
    # Forecast provenance proves what was known at prediction time. Archive /
    # recent-past rows may be retrieved later (MLB semantics), but their
    # selected hourly value still must predate kickoff.
    forecast_rows = cache["source"].eq(OPEN_METEO_FORECAST)
    observed_rows = cache["source"].isin(
        [OPEN_METEO_ARCHIVE, OPEN_METEO_FORECAST_PAST])
    valid &= (~forecast_rows) | (cache["fetched_at_utc"] < cache["kickoff_utc"])
    valid &= (~observed_rows) | (cache["fetched_at_utc"] >= cache["weather_time_utc"])
    cache = cache[valid].copy()
    if cache.empty:
        return _empty_weather()
    for col in ("temp_f", "wind_mph", "precip_in", "snow_in",
                "is_precip", "is_snow"):
        cache[col] = pd.to_numeric(cache[col], errors="coerce")
    value_cols = ["temp_f", "wind_mph", "precip_in", "snow_in"]
    cache = cache[cache[value_cols].notna().any(axis=1)].copy()
    if cache.empty:
        return _empty_weather()
    # Never trust persisted threshold flags over their raw selected-hour
    # amounts; normalize them on every cache read.
    cache["is_precip"] = np.where(
        cache["precip_in"].notna(),
        (cache["precip_in"] >= config.PRECIP_FLAG_IN).astype(float),
        np.nan,
    )
    cache["is_snow"] = np.where(
        cache["snow_in"].notna(),
        (cache["snow_in"] >= config.SNOW_FLAG_IN).astype(float),
        np.nan,
    )
    cache = cache.drop_duplicates(list(CACHE_COLUMNS), keep="last")
    # Conflicting rows for one game are ambiguous; omit that game rather than
    # selecting a convenient source/value combination.
    conflicts = set(cache.groupby("game_id", sort=False).size().loc[lambda s: s > 1].index)
    if conflicts:
        cache = cache[~cache["game_id"].isin(conflicts)]
    return cache.reset_index(drop=True)


def _load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_weather()
    try:
        return _validate_cache_frame(pd.read_parquet(path))
    except Exception as exc:  # noqa: BLE001 - unreadable cache rebuilds safely
        logger.warning("PIT weather cache unreadable (%s); rebuilding", exc)
        return _empty_weather()


def _save_cache(path: Path, cache: pd.DataFrame) -> None:
    if cache.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = _validate_cache_frame(cache)
    if cache.empty:
        return
    # Keep the .parquet suffix so pandas can infer the engine, then replace
    # atomically so an interrupted write cannot destroy the last good cache.
    temp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    cache[list(CACHE_COLUMNS)].to_parquet(temp, index=False)
    temp.replace(path)


def _records_for_targets(cache: pd.DataFrame,
                         targets: list[dict]) -> pd.DataFrame:
    if cache.empty or not targets:
        return _empty_weather()
    valid = _validate_cache_frame(cache)
    if valid.empty:
        return _empty_weather()
    by_id = valid.set_index("game_id", drop=False)
    rows = []
    for target in targets:
        game_id = target["game_id"]
        if game_id not in by_id.index:
            continue
        row = by_id.loc[game_id]
        if isinstance(row, pd.DataFrame):
            continue  # defensive: conflicts were removed above
        if (str(row["stadium"]) != target["stadium"]
                or _utc_timestamp(row["kickoff_utc"])
                != _utc_timestamp(target["kickoff_utc"])):
            continue
        rows.append(row[list(CACHE_COLUMNS)].to_dict())
    return pd.DataFrame(rows, columns=list(CACHE_COLUMNS))


def cached_weather_for_games(
    games: pd.DataFrame,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Read only validated PIT weather already present in the cache."""
    targets = _targets(games)
    return _records_for_targets(_load_cache(cache_path(path)), targets)


def fetch_games_weather(
    games: pd.DataFrame,
    *,
    refresh: bool = False,
    path: str | Path | None = None,
    now: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch/cache hourly Open-Meteo weather for all eligible NFL games.

    The cache is incremental for settled games. Missing outdoor games trigger
    network calls, and still-pending forecast games are refreshed on every run
    so serving uses a current pre-kickoff forecast. A refresh re-fetches all
    requested games but never trusts or consumes the legacy daily table.
    """
    targets = _targets(games)
    if not targets:
        return _empty_weather()
    # Always normalize: the default clock path must produce the same UTC
    # Timestamp type as an explicit ``now``, or the fetch/batch split below
    # receives a bare datetime and fails at runtime.
    fetched_at = _utc_timestamp(
        now if now is not None else datetime.now(timezone.utc))
    if fetched_at is None:
        return _empty_weather()

    target_path = cache_path(path)
    old = _load_cache(target_path)
    existing = _empty_weather() if refresh else _records_for_targets(old, targets)
    if existing.empty:
        have: set[str] = set()
    else:
        cached = existing.set_index("game_id", drop=False)
        have = set()
        for target in targets:
            game_id = target["game_id"]
            if game_id not in cached.index:
                continue
            row = cached.loc[game_id]
            pending_forecast = (
                row["source"] == OPEN_METEO_FORECAST
                and target["kickoff_utc"] > fetched_at
            )
            if not pending_forecast:
                have.add(game_id)
    missing = [target for target in targets if target["game_id"] not in have]

    fresh_rows: list[dict] = []
    # SAME-RUN TOP-UP (2026-10-09): twelve Open-Meteo 429 ladders exhausted
    # themselves mid-repull and 58 eligible games ended the run weather-less
    # under a 97.1% coverage figure that printed nothing but INFO — the
    # silent-starvation class this pipeline's monitoring exists to catch.
    # The cache-miss set is self-healing on the NEXT run, but a second pass
    # over only the gaps costs seconds once the rate limit has cleared and
    # closes the hole inside the delivery that opened it.
    still_missing = list(missing)
    for _pass in range(2):
        if not still_missing:
            break
        if _pass:
            logger.info(
                "PIT weather top-up: %d game(s) still missing after the "
                "first pass — refetching the gaps only", len(still_missing))
            # Fresh breaker budget for the run's final recovery pass: the
            # dead-batch streak describes the pass that just ended, and the
            # top-up must re-probe a window that may have reopened (the
            # 2026-10-11 18:10 run recovered all 48 of its gaps in 12s at
            # the tail of a flap).  The pass stays bounded: it covers only
            # the gap games' windows, every 429 defers instead of waiting,
            # and _RATE_LIMIT_ABORT_STREAK still caps its requeue.
            _reset_rate_limiter()
        # The cooldown belongs to the RECOVERY pass (``_pass`` > 0): the
        # main sweep keeps its never-sleep contract from the 2026-10-11
        # review, while the gap-only pass — whose failure means shipping a
        # hole — earns one bounded cooldown + probe before deferring.
        series_by_key = _fetch_batched_weather(
            still_missing, fetched_at.to_pydatetime(),
            cooldown_first_batch=bool(_pass))
        fetched_ids: set[str] = set()
        for target in still_missing:
            # A kickoff at (or just after) 00:00 UTC has no earlier hour on
            # its own UTC day, so the prior day's last row is the true latest
            # strictly-pre-kickoff reading. The kickoff day is always tried
            # first: its rows are newer than anything the prior day can offer.
            kickoff_day = target["kickoff_utc"].date()
            record = None
            for day in (kickoff_day, kickoff_day - timedelta(days=1)):
                record = select_weather_record(
                    target, series_by_key.get((target["stadium"], day)),
                    fetched_at)
                if record is not None:
                    break
            if record is not None:
                fresh_rows.append(record)
                fetched_ids.add(target["game_id"])
        still_missing = [t for t in still_missing
                         if t["game_id"] not in fetched_ids]

    # pandas 2.3 warns when concat receives an empty or all-NA placeholder.
    # The cache is already validated, so omit those frames and retain the
    # existing empty-cache fallback and replacement/deduplication semantics.
    fresh = pd.DataFrame(fresh_rows, columns=list(CACHE_COLUMNS))
    frames = [
        frame for frame in (old, fresh)
        if not frame.empty and bool(frame.notna().to_numpy().any())
    ]
    combined = (pd.concat(frames, ignore_index=True) if frames
                else _empty_weather())
    # New rows replace a stale cache row for the same game, while historical
    # games outside the current request remain available for later fold builds.
    combined = (combined.drop_duplicates("game_id", keep="last")
                if not combined.empty else _empty_weather())
    combined = _validate_cache_frame(combined)
    _save_cache(target_path, combined)
    result = _records_for_targets(combined, targets)
    coverage = len(result) / max(len(targets), 1)
    logger.info(
        "PIT weather: %d/%d eligible games available from strict pre-kickoff "
        "hourly records", len(result), len(targets),
    )
    if coverage < 0.8:
        logger.warning(
            "PIT weather coverage only %d/%d (%.0f%%) — affected weather "
            "features remain NaN; never substitute daily/schedule data",
            len(result), len(targets), 100.0 * coverage,
        )
    elif len(result) < len(targets):
        # Any hole is visible, not only a catastrophic one (2026-10-09:
        # 1964/2022 = 97.1% sat ABOVE the 80% line, so 58 429-blocked games
        # shipped with a single INFO). The note documents the self-heal:
        # these game_ids stay out of the cache, so the next run's cache-miss
        # set refetches exactly and only them.
        # 2026-10-13 run-log review: the numerator printed len(result) —
        # the AVAILABLE count — as the number that LACKS records
        # ("1881/2023 eligible games lack ... (142 remain missing)").
        # It is the gap count; the INFO line above owns the available one.
        have_ids = set(result["game_id"].astype(str)) if not result.empty \
            else set()
        gaps = [t["game_id"] for t in targets
                if str(t["game_id"]) not in have_ids]
        logger.warning(
            "PIT weather: %d/%d eligible games lack strict pre-kickoff "
            "records after the same-run top-up (e.g. %s); their weather "
            "features stay NaN and the next run's cache-miss pass refetches "
            "only these games",
            len(gaps), len(targets), ", ".join(map(str, gaps[:5])))
    return result
