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

_BATCH_DAYS = 14
_BATCH_SIZE = 15
_BATCH_PAUSE_SEC = 1.0
_RETRIABLE_STATUSES = frozenset({429, 502, 503, 504})
_RETRY_ATTEMPTS = 5
_RETRY_BASE_SEC = 2.0
_RETRY_JITTER_SEC = 0.5
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


def _get_with_retry(url: str, params: dict, attempts: int = _RETRY_ATTEMPTS,
                    timeout: int = 45):
    """GET with bounded exponential backoff for transient Open-Meteo errors."""
    last_exc: Exception | None = None
    response = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if (response.status_code in _RETRIABLE_STATUSES
                    and attempt < attempts - 1):
                wait = _RETRY_BASE_SEC ** attempt + random.uniform(
                    0, _RETRY_JITTER_SEC)
                try:
                    wait = max(wait, float(response.headers.get("Retry-After", "")))
                except (AttributeError, TypeError, ValueError):
                    pass
                logger.warning(
                    "Open-Meteo %s; retrying in %.1fs (%d/%d)",
                    response.status_code, wait, attempt + 1, attempts - 1,
                )
                time.sleep(wait)
                continue
            return response
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(_RETRY_BASE_SEC ** attempt + random.uniform(
                    0, _RETRY_JITTER_SEC))
    if last_exc is not None:
        raise last_exc
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


def _fetch_batch_range(
    locations: list[tuple[str, float, float]],
    start_date: date,
    end_date: date,
    *,
    source: str,
    forecast: bool = False,
    past_days: int | None = None,
    forecast_days: int | None = None,
) -> dict[tuple[str, date], dict[str, list]]:
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
        response = _get_with_retry(url, params=params, timeout=45)
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
    """Fetch paced archive/recent-past/forecast hourly series for all targets."""
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
        windows = range(0, (archive_end - start_date).days + 1, _BATCH_DAYS)
        for offset in windows:
            chunk_start = start_date + timedelta(days=offset)
            chunk_end = min(chunk_start + timedelta(days=_BATCH_DAYS - 1),
                            archive_end)
            if not any(chunk_start <= day <= chunk_end for day in needed_days):
                continue
            for i in range(0, len(locations), _BATCH_SIZE):
                by_key.update(_fetch_batch_range(
                    locations[i:i + _BATCH_SIZE], chunk_start, chunk_end,
                    source=OPEN_METEO_ARCHIVE,
                ))
                if i + _BATCH_SIZE < len(locations):
                    time.sleep(_BATCH_PAUSE_SEC)
            if chunk_end < archive_end:
                time.sleep(_BATCH_PAUSE_SEC)

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
            recent = _fetch_batch_range(
                locations, recent_start, recent_end,
                source=OPEN_METEO_FORECAST,
                forecast=True,
                past_days=(today - recent_start).days + 1,
                forecast_days=1,
            )
            for key, series in recent.items():
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
        for key, series in future.items():
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
    if missing:
        locations = [target for target in missing]
        series_by_key = _fetch_batched_weather(locations, fetched_at.to_pydatetime())
        for target in missing:
            # A kickoff at (or just after) 00:00 UTC has no earlier hour on its
            # own UTC day, so the prior day's last row is the true latest
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

    combined = pd.concat(
        [old, pd.DataFrame(fresh_rows, columns=list(CACHE_COLUMNS))],
        ignore_index=True,
    )
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
    return result
