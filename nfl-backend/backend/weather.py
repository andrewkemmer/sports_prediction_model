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
the fetch time, and is written every run as provenance history for the
``--skip-pull`` cache-only reader.  It never gates what is fetched: every
run is a FULL repull of every eligible game (2026-10-13 structural review,
MLB parity), paced below Open-Meteo's shared-IP quota with MLB's drain
ladder, and delivery is what THAT run observed — features.py revalidates
every row at the feature boundary.
"""
from __future__ import annotations

import copy
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

# Open-Meteo archive window: 60 days (2026-10-13 structural review — "MLB
# for structural guidance", full repull every run).  This module inherited
# MLB's 14-day cadence; what matters from MLB is the SHAPE — no cache in
# the fetch path, one paced full sweep per run, a drain ladder on 429 —
# not the window count.  At 14 days a decade of NFL game days spans 112
# windows x 4 batches = 448 requests, and the 2026-10-13 run lost 142
# games when the shared egress IP's quota closed at minute 18 of that
# sweep.  60-day windows were already MEASURED in this module's history
# (a decade of history drops from 357 to 111 request batches) and keep the
# MLB-proven 15 locations per batch, so a FULL repull of every eligible
# game every run costs ~111 requests over ~6-8 minutes instead of 448 over
# ~20: a quarter of the requests and a third of the quota exposure for the
# same observations.
_BATCH_DAYS = 60
_BATCH_SIZE = 15
# Sustained sweep pace.  The 2026-10-11 15:50 run measured the trip rate:
# 444 batches in 689s (~39 requests/min with the old 1.0s pause) interleaved
# 429 bursts roughly once a minute with clean stretches, even though
# Open-Meteo's documented free-tier limits (600/min, 5000/hr, 10000/day) are
# enforced PER IP -- and Kaggle's egress IP is shared with every other
# tenant hitting the same endpoint, so the effective budget under burst is
# far below the documented one.  2.0s between batches holds our
# contribution to ~15 req/min across the much shorter 111-batch sweep and
# buys headroom on that shared IP.
_BATCH_PAUSE_SEC = 2.0
_RETRIABLE_STATUSES = frozenset({429, 502, 503, 504})
# MLB's drain ladder — one shape for every retriable status: 2**attempt +
# jitter across 7 attempts = 1+2+4+8+16+32 = 63s of waiting, Retry-After
# authoritative when the server sends one.  63s outlasts one full
# Open-Meteo quota minute, so a rate-limited batch DRAINS instead of being
# lost (mlb weather._get_with_retry records the evidence: the 2026-09-25
# full repull lost batches to a 15s ladder; this one drains them).
_RETRY_ATTEMPTS = 7
_RETRY_BASE_SEC = 2.0
_RETRY_JITTER_SEC = 0.5
# --- Rate-limit brake (2026-10-11 and 2026-10-13 run-log reviews) ---
# History that shapes the knob below:
#   * 10-07 -> 10-11: each batch re-armed its own ladder into a shut
#     window — 156 retry requests across 25 back-to-back dead batches,
#     growing run over run (13 -> 72 -> 119 -> 156);
#   * 10-11 evening: escalating cooldown sleeps (60-240s per batch)
#     stretched a ~26-minute sweep to 51m28s while the shared-IP window
#     flapped shut for ~37 minutes — sleeping through a LONG closure is
#     wall-clock risk;
#   * 10-13 10:23: the defer/requeue machinery bounded everything, yet
#     142/2023 games still shipped weather-less after three recovery
#     passes all ran into the same shut window.
# The lesson that survives all three: drain ONE quota minute (the ladder
# above), and when the window stays shut beyond that, stop sleeping and
# fail fast at sweep pace — the paced request itself is the probe that
# notices a reopened window, and the first definitive answer re-arms
# draining.  _DRAIN_FAIL_FAST_AFTER consecutive full-ladder 429 failures
# engage that brake, which bounds a 37-minute flap at 2 x 63s of sleep
# plus fast-fail batches (minutes, not hours), instead of either 156
# wasted retries or ~30 minutes of pure cooldown sleep.
_DRAIN_FAIL_FAST_AFTER = 2
# Process-local breaker state (the pipeline is single-threaded).
# _reset_rate_limiter() restores it for tests and fresh sweeps.
_rate_limit: dict = {"drain_fails": 0, "brake": False}
# One stalled connection is retried; a stalled ENDPOINT is not.  Three
# consecutive timeouts end the request (3 x 20s + backoff ~= 63s) instead
# of the old 7 x 45s = 5+ minutes per dead batch, cascading across a sweep.
# 20s is still 4-10x Open-Meteo's normal answer time for a ~400KB batch.
_REQUEST_TIMEOUT_SEC = 20
_TIMEOUT_RETRY_LIMIT = 3
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
    _rate_limit.update(drain_fails=0, brake=False)


def _note_drain_failure() -> None:
    """Record one batch that spent the full drain ladder on a 429.

    ``_DRAIN_FAIL_FAST_AFTER`` consecutive ones engage the brake (logged
    once): from then on a still-shut window costs ONE request per batch
    and the paced sweep keeps probing instead of sleeping; any definitive
    answer re-arms draining via :func:`_note_window_open`.
    """
    _rate_limit["drain_fails"] = _rate_limit["drain_fails"] + 1
    if (_rate_limit["drain_fails"] >= _DRAIN_FAIL_FAST_AFTER
            and not _rate_limit["brake"]):
        _rate_limit["brake"] = True
        logger.warning(
            "Open-Meteo rate-limit brake: %d consecutive batches exhausted "
            "the %.0fs drain ladder — failing fast from here (one request "
            "per batch, the paced sweep keeps probing); any success "
            "re-arms draining",
            _rate_limit["drain_fails"],
            sum(_RETRY_BASE_SEC ** i for i in range(_RETRY_ATTEMPTS - 1)))


def _note_window_open() -> None:
    """A definitive answer: the quota window (re)opened — re-arm draining."""
    if _rate_limit["drain_fails"] or _rate_limit["brake"]:
        _rate_limit.update(drain_fails=0, brake=False)


def _get_with_retry(url: str, params: dict, attempts: int = _RETRY_ATTEMPTS,
                    timeout: int = _REQUEST_TIMEOUT_SEC):
    """GET with MLB's drain ladder for transient Open-Meteo errors.

    One ladder answers every retriable status (429/502/503/504) and every
    network exception: ``2**attempt`` + jitter across ``attempts`` rounds,
    or the server's ``Retry-After`` when it sends one (whichever is
    longer); every other status returns immediately (fail fast).  Six
    waits total 1+2+4+8+16+32 = 63s, which outlasts one full Open-Meteo
    quota minute — a rate-limited batch DRAINS instead of being lost
    (2026-10-13 structural review; mlb weather._get_with_retry carries the
    evidence that a shorter ladder loses batches).

    A shared-IP window can stay shut for much longer than one quota minute
    (2026-10-11: a ~37-minute flap), so ``_note_drain_failure`` engages a
    brake after ``_DRAIN_FAIL_FAST_AFTER`` consecutive batches spend the
    whole ladder on a 429: from then on this function returns a 429 after
    a SINGLE request (no sleeps) and the caller's paced sweep keeps
    probing; the first definitive answer re-arms draining via
    ``_note_window_open``.  Three consecutive hangs still end the request
    (``_TIMEOUT_RETRY_LIMIT``) — a stalled connection is not fixed by more
    waiting.
    """
    last_exc: Exception | None = None
    response = None
    timeouts = 0
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
                # the request; _fetch_batch_range reports the dead batch.
                raise
            if attempt < attempts - 1:
                time.sleep(_RETRY_BASE_SEC ** attempt + random.uniform(
                    0, _RETRY_JITTER_SEC))
                continue
            raise last_exc
        if response.status_code in _RETRIABLE_STATUSES:
            # One ladder for 429 and gateway trouble alike; a 429 while
            # the brake is engaged returns after a single request.
            braked = (response.status_code == 429
                      and _rate_limit["brake"])
            if attempt < attempts - 1 and not braked:
                wait = _RETRY_BASE_SEC ** attempt + random.uniform(
                    0, _RETRY_JITTER_SEC)
                try:
                    wait = max(wait, float(
                        response.headers.get("Retry-After", "")))
                except (AttributeError, TypeError, ValueError):
                    pass
                logger.warning(
                    "Open-Meteo %s; retrying in %.1fs (%d/%d)",
                    response.status_code, wait, attempt + 1, attempts - 1)
                time.sleep(wait)
                continue
            if response.status_code == 429:
                # Terminal for this call: the ladder ran out (or the brake
                # is engaged).  Record it so the brake can engage; the
                # batch-level line is _fetch_batch_range's to print.
                _note_drain_failure()
            return response
        # 2xx (or a fail-fast 4xx): definitive answer — the window (re)opened.
        _note_window_open()
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


def _fan_out_coord_aliases(
    result: dict[tuple[str, date], dict[str, list]] | None,
    aliases: dict[str, list[str]],
) -> dict[tuple[str, date], dict[str, list]]:
    """Replicate a site's series under every other name that site carries.

    The sweep requests ONE coordinate per physical stadium site (MLB keys
    its locations the same way — one entry per venue), so a rename pair
    ships as a single location keyed by the first name seen.  A game's
    schedule row may use either name, so each alias receives the identical
    series here, before the merge: same coordinates, same response grid,
    byte-identical content (probed against Open-Meteo 2026-10-13: a
    3-coordinate request with one duplicated coordinate returned 3 items,
    the duplicate identical to the original).  Copies are deep so no two
    cache keys ever share one mutable series object.
    """
    if not result or not aliases:
        return result or {}
    out = dict(result)
    for (stadium, day), series in result.items():
        for alias in aliases.get(stadium, ()):
            out[(alias, day)] = copy.deepcopy(series)
    return out


def _fetch_batch_range(
    locations: list[tuple[str, float, float]],
    start_date: date,
    end_date: date,
    *,
    source: str,
    forecast: bool = False,
    past_days: int | None = None,
    forecast_days: int | None = None,
) -> dict[tuple[str, date], dict[str, list]] | None:
    """One bounded multi-coordinate request; None = rate-limited, {} = failed.

    A 429 the drain ladder cannot save returns ``None`` (never ``{}``) so
    callers can tell "the quota said no" from "the request died": both
    leave the batch's days missing, and the recovery is the same — the
    same-run top-up refetches the gap games, the next run's full repull
    refetches everything.  Callers that merge with ``by_key.update`` must
    therefore handle ``None`` (``if result:``).
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
                                   timeout=_REQUEST_TIMEOUT_SEC)
        if response.status_code == 429:
            # Rate-limited, never dead-for-good: None (not {}), so callers
            # never treat a shut quota window as a parse failure.  The
            # batch-level line prints only while draining is still awake —
            # once the brake is engaged, one brake warning plus the final
            # coverage/gap lines own the story instead of one line per
            # fast-failed batch (the 2026-10-11 review: per-dead-batch
            # warnings bury a long closure).
            if not _rate_limit["brake"]:
                logger.warning(
                    "Open-Meteo 429 after the drain ladder "
                    "(%s..%s, source=%s) - no data for this batch; missing "
                    "games stay NaN and recover via the same-run top-up, "
                    "else the next run's full repull",
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
) -> dict[tuple[str, date], dict[str, list]]:
    """Fetch ALL targets' paced archive/recent-past/forecast hourly series.

    One MLB-shaped flow, no deferral modes (2026-10-13 structural review):
    archive windows (game-day filtered) at sweep pace, then the forecast
    endpoint's observed-past window for ERA5's publication lag, then the
    future forecast.  A batch the drain ladder cannot save returns ``None``
    / ``{}`` and simply leaves its days missing: the same-run top-up
    refetches exactly the gaps with a fresh breaker, and the next run's
    FULL repull refetches everything regardless — no cache stands between
    a hole and its recovery.
    """
    by_key: dict[tuple[str, date], dict[str, list]] = {}
    if not targets:
        return by_key

    locations_map: dict[str, tuple[str, float, float]] = {}
    for row in targets:
        locations_map.setdefault(
            row["stadium"], (row["stadium"], row["lat"], row["lon"]))
    # ONE request per physical site, not per stadium name (2026-10-13 log
    # review, MLB parity): renames are several names on one coordinate
    # (Heinz/Acrisure, FedEx/Commanders, ...), and today's 50 names are 37
    # sites — every extra copy spends quota on coordinates already inside
    # the same batch and inflates each sweep batch by a third.  Open-Meteo
    # answers identical coordinates with identical series (probed
    # 2026-10-13); ``_fan_out_coord_aliases`` restores each name's key
    # before the merge, so delivered rows are unchanged — only the request
    # count drops (136 -> 102 on the 2026-10-13 production span).
    by_coord: dict[tuple[float, float], list[str]] = {}
    for name, lat, lon in locations_map.values():
        by_coord.setdefault((lat, lon), []).append(name)
    aliases = {names[0]: names[1:] for names in by_coord.values()
               if len(names) > 1}
    locations = [(names[0], coord[0], coord[1])
                 for coord, names in by_coord.items()]
    needed_days = {row["kickoff_utc"].date() for row in targets}
    # The prior UTC day is needed for late ET games whose kickoff is just after
    # 00:00 UTC and therefore has no earlier row on its own UTC date.
    start_date = min(needed_days) - timedelta(days=1)
    end_date = max(needed_days)
    today = now.date()

    archive_end = min(end_date, today - timedelta(days=1))
    skipped_windows = 0
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
            else:
                # MLB reports these skips after the sweep; without the line
                # the logged window count looks arbitrary (34 of ~62 spans
                # in the 2026-10-13 run) and reads like under-fetching.
                skipped_windows += 1
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
        # A 429 the drain ladder cannot save leaves its days MISSING for
        # this pass (None), never a silent {}: the top-up refetches exactly
        # the gaps, and the next run's full repull refetches everything.
        # The sweep runs to completion at its steady pace either way — a
        # shut window costs one request per batch once the brake engages.
        for chunk_start, chunk_end in windows:
            for i in range(0, len(locations), _BATCH_SIZE):
                batch = locations[i:i + _BATCH_SIZE]
                result = _fetch_batch_range(
                    batch, chunk_start, chunk_end,
                    source=OPEN_METEO_ARCHIVE,
                )
                if result:
                    by_key.update(
                        _fan_out_coord_aliases(result, aliases))
                attempted += 1
                bar.advance()
                if i + _BATCH_SIZE < len(locations):
                    time.sleep(_BATCH_PAUSE_SEC)
            if chunk_end < archive_end:
                time.sleep(_BATCH_PAUSE_SEC)
        bar.close()

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
            for key, series in _fan_out_coord_aliases(
                    recent, aliases).items():
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
        for key, series in _fan_out_coord_aliases(
                future, aliases).items():
            by_key.setdefault(key, series)
    if skipped_windows:
        # MLB's line, structurally verbatim: the sweep's window count only
        # makes sense next to the count it did NOT send — gameless 60-day
        # windows (NFL off-seasons) are dropped before any request.
        logger.info(
            "PIT weather batches: skipped %d gameless window(s) "
            "(%s→%s span) — no Open-Meteo requests sent for them",
            skipped_windows, start_date, archive_end)
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
    """Fetch PIT hourly Open-Meteo weather for ALL eligible NFL games.

    FULL REPULL EVERY RUN (2026-10-13 structural review, MLB parity): no
    cache file gates what is fetched.  The earlier design read the cache
    first and requested only uncached games ("top-off"), which made a run's
    completeness depend on files surviving between runs — on Kaggle the
    repo is cloned fresh every run, so the top-off never fired there while
    local runs silently under-fetched.  MLB's fetch path has no cache at
    all; this one keeps the file as provenance history for the
    ``--skip-pull`` cache-only reader (:func:`cached_weather_for_games`)
    and ``build_weather_table``, but delivery is what THIS run observed: a
    failed fetch ships NaN and the next run refetches everything.
    ``refresh`` is retained for caller compatibility and is a no-op —
    every run refreshes every game.
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
    # FULL REPULL: the cache is NEVER consulted to decide WHAT gets fetched
    # — every eligible game is requested on every run.  ``old`` exists only
    # so the saved cache keeps its provenance history for the --skip-pull
    # cache-only path; nothing in the delivered result is backfilled from
    # disk.  ``refresh`` is the documented no-op above.
    old = _load_cache(target_path)

    fresh_rows: list[dict] = []
    # SAME-RUN TOP-UP (2026-10-09): twelve Open-Meteo 429 ladders exhausted
    # themselves mid-repull and 58 eligible games ended the run weather-less
    # under a 97.1% coverage figure that printed nothing but INFO — the
    # silent-starvation class this pipeline's monitoring exists to catch.
    # The next run's full repull is self-healing, but a second pass over
    # only the gaps costs seconds once the rate limit has cleared and
    # closes the hole inside the delivery that opened it.
    still_missing = list(targets)
    for _pass in range(2):
        if not still_missing:
            break
        if _pass:
            logger.info(
                "PIT weather top-up: %d game(s) still missing after the "
                "first pass — refetching the gaps only", len(still_missing))
            # Fresh breaker budget for the run's recovery pass: the brake
            # (if engaged) describes the pass that just ended, and the
            # top-up must re-probe a window that may have reopened (the
            # 2026-10-11 18:10 run recovered all 48 of its gaps in 12s at
            # the tail of a flap).  The pass stays bounded: it covers only
            # the gap games' windows, drains one quota minute per batch,
            # then fails fast under the brake.
            _reset_rate_limiter()
        series_by_key = _fetch_batched_weather(
            still_missing, fetched_at.to_pydatetime())
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
    # The cache file keeps its provenance HISTORY (old rows survive for the
    # --skip-pull cache-only reader), but delivery is this run's observations
    # ONLY: a game this run failed to fetch ships NaN even when an older row
    # exists on disk — the next run's full repull, not the file, recovers it.
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
    result = _records_for_targets(fresh, targets)
    coverage = len(result) / max(len(targets), 1)
    logger.info(
        "PIT weather: %d/%d eligible games available from strict pre-kickoff "
        "hourly records", len(result), len(targets),
    )
    have_ids = set(result["game_id"].astype(str)) if not result.empty \
        else set()
    gaps = [t["game_id"] for t in targets
            if str(t["game_id"]) not in have_ids]
    if coverage < 0.8:
        logger.warning(
            "PIT weather coverage only %d/%d (%.0f%%) — affected weather "
            "features remain NaN; never substitute daily/schedule data",
            len(result), len(targets), 100.0 * coverage,
        )
    elif gaps:
        # Any hole is visible, not only a catastrophic one (2026-10-09:
        # 1964/2022 = 97.1% sat ABOVE the 80% line, so 58 429-blocked games
        # shipped with a single INFO). The note documents the self-heal:
        # the next run repulls EVERYTHING, gaps included — no cache-miss
        # set exists (or is needed) under the full-repull contract.
        # 2026-10-13 run-log review: the numerator printed len(result) —
        # the AVAILABLE count — as the number that LACKS records
        # ("1881/2023 eligible games lack ... (142 remain missing)").
        # It is the gap count; the INFO line above owns the available one.
        logger.warning(
            "PIT weather: %d/%d eligible games lack strict pre-kickoff "
            "records after the same-run top-up (e.g. %s); their weather "
            "features stay NaN for this run and the next run's full repull "
            "refetches them",
            len(gaps), len(targets), ", ".join(map(str, gaps[:5])))
    # Season-level starvation gate (MLB parity): a healthy aggregate must
    # not dilute one starved year — MLB prints "that season's weather
    # features are starving" per calendar year at >= 40 games, and this
    # run's 71% aggregate (outdoor-only eligibility) would mask a fully
    # failed early season exactly the way MLB's could.
    if len(targets) >= 40:
        by_year: dict[int, tuple[int, int]] = {}
        for target in targets:
            year = target["kickoff_utc"].year
            tot, good = by_year.get(year, (0, 0))
            by_year[year] = (
                tot + 1,
                good + (1 if str(target["game_id"]) in have_ids else 0))
        for year, (tot, good) in sorted(by_year.items()):
            if tot and good < 0.8 * tot:
                logger.warning(
                    "PIT weather coverage %d: only %d/%d games observed "
                    "— that season's weather features are starving",
                    year, good, tot)
    return result
