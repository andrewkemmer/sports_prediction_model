"""
Point-in-time weather data for MLB games via Open-Meteo (free, no API key).

Provides:
- Hourly temperature, humidity, wind speed/direction, surface pressure
- Air density computation from weather + altitude
- Wind advantage multiplier (out=+1, in=-1, cross=0)

Point-in-time enforcement: for historical games, uses the archive API
(historical weather). For future/today's games, uses the forecast API.
All data is matched to the LOCAL game start hour (stadium timezone).

Open-Meteo terms: https://open-meteo.com/en/terms
"""
from __future__ import annotations

import logging
from datetime import datetime, date, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Stadium database ────────────────────────────────────────────────────────
# (lat, lon, altitude_m, center_field_bearing_degrees)
# bearing = compass direction from home plate to center field (0=N, 90=E).
# Dome stadiums get bearing=0 and are handled by DOME_STATUS in features.py.
STADIUMS: dict[str, dict] = {
    # AL East
    "NYY": {"lat": 40.8296, "lon": -73.9262, "alt_m": 9,   "bearing": 10,  "tz": "America/New_York"},
    "BOS": {"lat": 42.3467, "lon": -71.0972, "alt_m": 3,   "bearing": 120, "tz": "America/New_York"},
    "BAL": {"lat": 39.2838, "lon": -76.6216, "alt_m": 10,  "bearing": 345, "tz": "America/New_York"},
    "TB":  {"lat": 27.9797, "lon": -82.5054, "alt_m": 13,  "bearing": 0,   "tz": "America/New_York"},  # dome
    "TOR": {"lat": 43.6414, "lon": -79.3894, "alt_m": 176, "bearing": 0,   "tz": "America/Toronto"},   # retractable roof
    # AL Central
    "CLE": {"lat": 41.4962, "lon": -81.6852, "alt_m": 210, "bearing": 5,   "tz": "America/New_York"},
    "CWS": {"lat": 41.8299, "lon": -87.6338, "alt_m": 180, "bearing": 15,  "tz": "America/Chicago"},
    "DET": {"lat": 42.3390, "lon": -83.0485, "alt_m": 182, "bearing": 350, "tz": "America/New_York"},
    "KC":  {"lat": 39.0517, "lon": -94.4803, "alt_m": 253, "bearing": 345, "tz": "America/Chicago"},
    "MIN": {"lat": 44.9817, "lon": -93.2776, "alt_m": 264, "bearing": 340, "tz": "America/Chicago"},
    # AL West
    "HOU": {"lat": 29.7573, "lon": -95.3555, "alt_m": 15,  "bearing": 0,   "tz": "America/Chicago"},   # retractable
    "LAA": {"lat": 33.8003, "lon": -117.8827,"alt_m": 54,  "bearing": 350, "tz": "America/Los_Angeles"},
    "ATH": {"lat": 38.5803, "lon": -121.5085,"alt_m": 6,   "bearing": 350, "tz": "America/Los_Angeles"},  # Sutter Health Park (2025)
    "SEA": {"lat": 47.5914, "lon": -122.3325,"alt_m": 5,   "bearing": 0,   "tz": "America/Los_Angeles"},  # retractable
    "TEX": {"lat": 32.7473, "lon": -97.0845, "alt_m": 170, "bearing": 0,   "tz": "America/Chicago"},   # retractable
    # NL East
    "ATL": {"lat": 33.8907, "lon": -84.4677, "alt_m": 315, "bearing": 345, "tz": "America/New_York"},
    "MIA": {"lat": 25.7781, "lon": -80.2196, "alt_m": 5,   "bearing": 0,   "tz": "America/New_York"},   # retractable
    "NYM": {"lat": 40.7571, "lon": -73.8458, "alt_m": 12,  "bearing": 15,  "tz": "America/New_York"},
    "PHI": {"lat": 39.9061, "lon": -75.1665, "alt_m": 10,  "bearing": 10,  "tz": "America/New_York"},
    "WSH": {"lat": 38.8730, "lon": -77.0074, "alt_m": 15,  "bearing": 15,  "tz": "America/New_York"},
    # NL Central
    "CHC": {"lat": 41.9484, "lon": -87.6553, "alt_m": 180, "bearing": 160, "tz": "America/Chicago"},
    "MIL": {"lat": 43.0280, "lon": -87.9712, "alt_m": 195, "bearing": 0,   "tz": "America/Chicago"},   # retractable
    "PIT": {"lat": 40.4469, "lon": -80.0057, "alt_m": 230, "bearing": 10,  "tz": "America/New_York"},
    "STL": {"lat": 38.6226, "lon": -90.1928, "alt_m": 140, "bearing": 10,  "tz": "America/Chicago"},
    "CIN": {"lat": 39.0974, "lon": -84.5065, "alt_m": 225, "bearing": 15,  "tz": "America/New_York"},
    # NL West
    "ARI": {"lat": 33.4455, "lon": -112.0667,"alt_m": 331, "bearing": 0,   "tz": "America/Phoenix"},   # retractable
    "COL": {"lat": 39.7559, "lon": -104.9942,"alt_m": 1610,"bearing": 10,  "tz": "America/Denver"},
    "LAD": {"lat": 34.0739, "lon": -118.2400,"alt_m": 125, "bearing": 180, "tz": "America/Los_Angeles"},
    "SD":  {"lat": 32.7076, "lon": -117.1570,"alt_m": 5,   "bearing": 350, "tz": "America/Los_Angeles"},
    "SF":  {"lat": 37.7786, "lon": -122.3893,"alt_m": 6,   "bearing": 350, "tz": "America/Los_Angeles"},
}

# Alias map: venue name fragments → team code
_VENUE_ALIASES: dict[str, str] = {
    "yankee": "NYY", "fenway": "BOS", "camden": "BAL", "oriole": "BAL",
    "tropicana": "TB", "rogers": "TOR", "skydome": "TOR",
    "progressive": "CLE", "guardian": "CLE", "rate field": "CWS", "guaranteed": "CWS",
    "comerica": "DET", "kauffman": "KC", "target field": "MIN", "twins": "MIN",
    "minute maid": "HOU", "angel stadium": "LAA", "angel": "LAA",
    "sutter health": "ATH", "oakland": "ATH", "coliseum": "ATH",
    "t-mobile": "SEA", "safeco": "SEA", "globe life": "TEX",
    "truist": "ATL", "suntrust": "ATL", "loandepot": "MIA", "marlins park": "MIA",
    "citi field": "NYM", "citizens bank": "PHI",
    "nationals park": "WSH", "nationals": "WSH",
    "wrigley": "CHC", "american family": "MIL", "miller": "MIL",
    "pnc park": "PIT", "busch": "STL", "great american": "CIN", "gabp": "CIN",
    "chase field": "ARI", "coors field": "COL", "coors": "COL",
    "dodger": "LAD", "petco": "SD", "oracle": "SF", "at&t": "SF", "pac bell": "SF",
}

# Team code → team code (for cases where venue name IS the team name)
_TEAM_VENUE_MAP: dict[str, str] = {
    "NYY": "NYY", "BOS": "BOS", "BAL": "BAL", "TB": "TB", "TOR": "TOR",
    "CLE": "CLE", "CWS": "CWS", "DET": "DET", "KC": "KC", "MIN": "MIN",
    "HOU": "HOU", "LAA": "LAA", "ATH": "ATH", "SEA": "SEA", "TEX": "TEX",
    "ATL": "ATL", "MIA": "MIA", "NYM": "NYM", "PHI": "PHI", "WSH": "WSH",
    "CHC": "CHC", "MIL": "MIL", "PIT": "PIT", "STL": "STL", "CIN": "CIN",
    "ARI": "ARI", "COL": "COL", "LAD": "LAD", "SD": "SD", "SF": "SF",
    # Statcast aliases
    "AZ": "ARI", "OAK": "ATH",
}


def _resolve_team_code(home_team: str, venue: str = "") -> str:
    """Resolve a team code from team name or venue name."""
    code = str(home_team).upper().strip()
    if code in _TEAM_VENUE_MAP:
        return _TEAM_VENUE_MAP[code]
    v = str(venue).lower().strip()
    for frag, tc in _VENUE_ALIASES.items():
        if frag in v:
            return tc
    return code


# ── Air density ─────────────────────────────────────────────────────────────

def compute_air_density(
    temp_c: float,
    rh_pct: float,
    pressure_hpa: float,
    altitude_m: float,
) -> float:
    """Compute air density (kg/m³) from weather conditions.

    Uses the virtual-temperature correction for humidity.  When
    ``pressure_hpa`` is surface (station) pressure it is used directly;
    when it approximates sea-level pressure the barometric formula
    corrects it to station level via ``altitude_m``.
    Standard sea-level value ≈ 1.225 kg/m³.

    Point-in-time / null rule: any MISSING observation returns NaN.  A
    missing humidity is not silently assumed to be 50% — if we do not
    observe it, the feature is null.  (Pressure may be derived from
    stadium altitude, which is a known constant, so a missing pressure
    reading is a real calculation rather than a fabrication.)
    """
    if np.isnan(temp_c) or np.isnan(rh_pct):
        return np.nan
    if np.isnan(pressure_hpa) and np.isnan(altitude_m):
        return np.nan

    T = temp_c + 273.15  # Kelvin

    # If pressure is provided, use it; otherwise compute from altitude
    if not np.isnan(pressure_hpa):
        P = pressure_hpa * 100.0  # Pa
        # Open-Meteo surface_pressure is station-level.  Altitude provides
        # a sanity correction: if pressure looks like sea-level (>1000 hPa)
        # but altitude is high, apply the barometric reduction.
        if altitude_m > 200 and pressure_hpa > 950:
            # Barometric formula: P_stn ≈ P_sea * exp(-g*M*h/(R*T))
            g = 9.80665
            M = 0.0289644
            R = 8.31447
            P = P * np.exp(-g * M * altitude_m / (R * T))
    else:
        # No pressure available — compute from altitude (standard atmosphere)
        P0 = 101325.0  # Pa
        g = 9.80665
        M = 0.0289644
        R = 8.31447
        P = P0 * np.exp(-g * M * altitude_m / (R * T))

    # Saturation vapor pressure (Tetens formula)
    esat = 610.78 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    rh_frac = max(0.0, min(100.0, rh_pct)) / 100.0
    Pv = rh_frac * esat  # partial pressure of water vapor
    Pd = P - Pv  # dry air partial pressure

    Rd = 287.05  # dry air gas constant (J/(kg·K))
    Rv = 461.495  # water vapor gas constant (J/(kg·K))

    rho = Pd / (Rd * T) + Pv / (Rv * T)
    return round(rho, 5)


# ── Wind advantage multiplier ───────────────────────────────────────────────

def compute_wind_multiplier(
    wind_direction_deg: float,
    wind_speed_kmh: float,
    stadium_bearing: float,
) -> float:
    """Compute wind advantage multiplier.

    Wind out (blowing toward center field, same direction as batted balls)
    helps hitters → +1.0.  Wind in (from center field toward home plate)
    helps pitchers → -1.0.  Calm or cross winds → near 0.

    Returns a value in [-1, 1] scaled by wind speed.  NaN for any MISSING
    observation (never a fabricated 0); 0.0 is reserved for genuinely calm
    wind or a genuinely perpendicular (cross) wind.
    """
    if np.isnan(wind_direction_deg) or np.isnan(wind_speed_kmh):
        return np.nan
    if wind_speed_kmh < 3.0:  # calm (< 2 mph) — a real, valid observation
        return 0.0

    # wind_direction_deg = direction wind is COMING FROM (meteorological)
    # bearing = direction from home plate TO center field
    # If wind comes from home plate direction, it blows OUT toward center field
    diff_rad = np.radians(wind_direction_deg - stadium_bearing)
    cos_component = np.cos(diff_rad)  # +1 = blowing out, -1 = blowing in

    # Scale by wind speed (cap at 40 km/h ≈ 25 mph for saturation)
    speed_factor = min(wind_speed_kmh / 40.0, 1.0)

    return round(cos_component * speed_factor, 4)


# ── Weather fetching ────────────────────────────────────────────────────────

_HOURLY_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure"


def _nearest_hour(
    times: list[str],
    target_utc: datetime,
) -> int:
    """Find the index of the hourly timestamp closest to target_utc."""
    # Strip timezone info from both sides to avoid naive/aware mismatch
    target = target_utc.replace(tzinfo=None)
    best_i, best_dt = 0, timedelta(days=999)
    for i, t in enumerate(times):
        ts = datetime.fromisoformat(t.replace("Z", "")).replace(tzinfo=None)
        dt = abs(ts - target)
        if dt < best_dt:
            best_dt = dt
            best_i = i
    return best_i


def _hour_prior(
    times: list[str],
    target_utc: datetime,
) -> int:
    """Index of the latest hourly timestamp STRICTLY before target_utc.

    Point-in-time rule: weather used as a pre-game feature must be observed
    strictly before first pitch, so the hour in which the game starts is
    never used (its window overlaps the game).  Returns -1 when no hourly
    row is strictly prior (e.g. a past game whose archive row does not
    exist yet, or a start at/before the earliest available hour).
    """
    target = target_utc.replace(tzinfo=None)
    best_i = -1
    for i, t in enumerate(times):
        ts = datetime.fromisoformat(t.replace("Z", "")).replace(tzinfo=None)
        if ts < target:
            best_i = i
    return best_i


_HOURLY_KEYS = (
    "time", "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "wind_direction_10m", "surface_pressure",
)


def _get_with_retry(url: str, params: dict, attempts: int = 3,
                    timeout: int = 15):
    """GET with exponential backoff and server-directed retry delays."""
    import random
    import time
    last_exc = None
    for attempt in range(attempts):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code in (429, 502, 503, 504) and attempt < attempts - 1:
                wait = (2 ** attempt) + random.uniform(0, 0.5)
                try:
                    retry_after = float(resp.headers.get("Retry-After", ""))
                    wait = max(wait, retry_after)
                except (AttributeError, TypeError, ValueError):
                    pass
                logger.warning(
                    "Weather API %d for %s — retrying in %.1fs (%d/%d)",
                    resp.status_code, url.split("/")[-1], wait,
                    attempt + 1, attempts - 1,
                )
                time.sleep(wait)
                continue
            return resp
        except requests.RequestException as e:
            last_exc = e
            if attempt < attempts - 1:
                time.sleep((2 ** attempt) + random.uniform(0, 0.5))
    if last_exc is not None:
        raise last_exc
    return resp


def _fetch_hourly_series(
    lat: float,
    lon: float,
    archive_date: date,
) -> dict[str, list] | None:
    """Full-day hourly arrays for one stadium/date (one HTTP request).

    Past dates (strictly before today) use the archive API — the observed
    historical record.  Today/future use the forecast API, which is
    produced before game time and therefore strictly prior knowledge.
    Returns None on any failure or empty payload (missing observation).
    """
    today = date.today()
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": _HOURLY_VARS,
    }
    if archive_date < today:
        # Archive API (historical) — free, no key.
        params["start_date"] = archive_date.isoformat()
        params["end_date"] = archive_date.isoformat()
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        # Forecast API — free, no key (covers 7–16 days ahead)
        params["forecast_days"] = 16
        url = "https://api.open-meteo.com/v1/forecast"

    try:
        resp = _get_with_retry(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
    except Exception as e:
        logger.warning("Weather fetch failed for %.2f,%.2f: %s", lat, lon, e)
        hourly = {}

    if not hourly.get("time"):
        # Archive latency: the historical archive lags ~5 days.  Recent past
        # dates fall back to the forecast API's observed past window (up to
        # 92 days), which carries the same strictly-prior hourly record.
        if archive_date < today:
            recent = _fetch_recent_past_series(lat, lon, archive_date)
            if recent is not None:
                return recent
        return None
    return {k: hourly.get(k, []) for k in _HOURLY_KEYS}


def _fetch_recent_past_series(
    lat: float,
    lon: float,
    day: date,
) -> dict[str, list] | None:
    """Observed hourly series for a recent past date via the forecast API.

    The archive API publishes with ~5-day latency, leaving the freshest
    decided games without weather.  The forecast endpoint exposes its
    observed past window (`past_days`, up to 92) — same hourly record,
    available immediately.  Returns only rows belonging to ``day``
    (stadium-local), or None when unavailable.
    """
    past = min((date.today() - day).days + 1, 92)
    if past < 1:
        return None
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": _HOURLY_VARS,
        "past_days": past,
        "forecast_days": 1,
    }
    try:
        resp = _get_with_retry(
            "https://api.open-meteo.com/v1/forecast", params=params, timeout=15
        )
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
    except Exception as e:
        logger.warning("Recent-past weather fetch failed for %.2f,%.2f: %s", lat, lon, e)
        return None

    prefix = day.isoformat()
    times = hourly.get("time", []) or []
    keep = [i for i, t in enumerate(times) if str(t).startswith(prefix)]
    if not keep:
        return None
    out = {"time": [times[i] for i in keep]}
    for k in _HOURLY_KEYS:
        if k == "time":
            continue
        vals = hourly.get(k, []) or []
        out[k] = [vals[i] if i < len(vals) else None for i in keep]
    return out


def _pick_row(series: dict[str, list] | None, game_local_time: datetime | None) -> dict:
    """Pick the strictly-prior hourly row from a day's series (NaN when missing)."""
    if series is None or game_local_time is None:
        return {
            "temp_c": np.nan, "rh_pct": np.nan,
            "wind_speed_kmh": np.nan, "wind_direction_deg": np.nan,
            "pressure_hpa": np.nan,
        }
    idx = _hour_prior(series["time"], game_local_time)
    if idx < 0:
        return {
            "temp_c": np.nan, "rh_pct": np.nan,
            "wind_speed_kmh": np.nan, "wind_direction_deg": np.nan,
            "pressure_hpa": np.nan,
        }

    def _get(key: str) -> float:
        vals = series.get(key, [])
        return float(vals[idx]) if idx < len(vals) else np.nan

    return {
        "temp_c": _get("temperature_2m"),
        "rh_pct": _get("relative_humidity_2m"),
        "wind_speed_kmh": _get("wind_speed_10m"),
        "wind_direction_deg": _get("wind_direction_10m"),
        "pressure_hpa": _get("surface_pressure"),
    }


def _has_observation(raw: dict) -> bool:
    """Whether a selected row contains at least one observed weather value."""
    return any(pd.notna(raw.get(key)) for key in (
        "temp_c", "rh_pct", "wind_speed_kmh", "wind_direction_deg", "pressure_hpa"
    ))


def fetch_weather(
    lat: float,
    lon: float,
    game_local_time: datetime,
    archive_date: date,
) -> dict:
    """Fetch hourly weather for a single stadium at game time.

    Uses archive API for past dates, forecast API for today/future.  The
    hourly row used is the latest one STRICTLY before game time (PIT).

    Returns dict with keys: temp_c, rh_pct, wind_speed_kmh,
    wind_direction_deg, pressure_hpa (all NaN when unavailable).
    """
    return _pick_row(_fetch_hourly_series(lat, lon, archive_date), game_local_time)


def _localize(start_utc: datetime, tz_name: str, lon: float = 0.0) -> datetime:
    """Convert a UTC game start to stadium-local time (DST-correct)."""
    try:
        from zoneinfo import ZoneInfo
        start = start_utc if start_utc.tzinfo else start_utc.replace(tzinfo=timezone.utc)
        return start.astimezone(ZoneInfo(tz_name))
    except Exception:
        # Fallback: fixed offset ≈ longitude / 15 (ignores DST)
        return start_utc + timedelta(hours=round(lon / 15.0))


def _stadium_weather(
    info: dict,
    game_start_utc: datetime,
    series: dict[str, list] | None,
) -> dict:
    """Build the extended weather dict for one game from a day's series."""
    local = _localize(game_start_utc, info["tz"], info["lon"])
    raw = _pick_row(series, local)
    air_density = compute_air_density(
        raw["temp_c"], raw["rh_pct"], raw["pressure_hpa"], info["alt_m"]
    )
    wind_mult = compute_wind_multiplier(
        raw["wind_direction_deg"], raw["wind_speed_kmh"], info["bearing"]
    )
    return {
        "available": _has_observation(raw),
        "source": "open_meteo_archive" if _has_observation(raw) else "open_meteo_unavailable",
        "temp_c": raw["temp_c"],
        "rh_pct": raw["rh_pct"],
        "wind_speed_kmh": raw["wind_speed_kmh"],
        "wind_direction_deg": raw["wind_direction_deg"],
        "pressure_hpa": raw["pressure_hpa"],
        "air_density": air_density,
        "wind_multiplier": wind_mult,
        "stadium_alt_m": info["alt_m"],
        "stadium_bearing": info["bearing"],
    }


def fetch_game_weather(
    home_team: str,
    venue: str,
    game_start_utc: datetime,
    game_start_local: Optional[datetime] = None,
) -> dict:
    """Fetch weather for one game.  Returns extended dict with raw weather
    plus computed air_density and wind_multiplier.

    ``game_start_local`` is used to pick the right hourly row.  If not
    provided, it is derived from ``game_start_utc`` via the stadium's
    registered timezone (DST-correct).
    """
    team_code = _resolve_team_code(home_team, venue)
    info = STADIUMS.get(team_code)
    if info is None:
        logger.warning("No stadium data for %s (venue=%s)", home_team, venue)
        return {"available": False}

    if game_start_local is None:
        game_start_local = _localize(game_start_utc, info["tz"], info["lon"])

    game_date = game_start_local.date()

    raw = fetch_weather(info["lat"], info["lon"], game_start_local, game_date)

    air_density = compute_air_density(
        raw["temp_c"], raw["rh_pct"], raw["pressure_hpa"], info["alt_m"]
    )
    wind_mult = compute_wind_multiplier(
        raw["wind_direction_deg"], raw["wind_speed_kmh"], info["bearing"]
    )

    observed = _has_observation(raw)
    return {
        "available": observed,
        "source": "open_meteo_archive" if observed else "open_meteo_unavailable",
        "temp_c": raw["temp_c"],
        "rh_pct": raw["rh_pct"],
        "wind_speed_kmh": raw["wind_speed_kmh"],
        "wind_direction_deg": raw["wind_direction_deg"],
        "pressure_hpa": raw["pressure_hpa"],
        "air_density": air_density,
        "wind_multiplier": wind_mult,
        "stadium_alt_m": info["alt_m"],
        "stadium_bearing": info["bearing"],
    }


_WEATHER_BATCH_DAYS = 14
_WEATHER_BATCH_SIZE = 30
_WEATHER_BATCH_PAUSE_SEC = 0.5


def _utc_naive(value: datetime | pd.Timestamp | str) -> datetime | None:
    """Normalize a timestamp to a naive UTC datetime for API matching."""
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        else:
            ts = ts.tz_localize(None)
        return ts.to_pydatetime()
    except (TypeError, ValueError, OverflowError):
        return None


def _split_hourly_by_utc_day(
    hourly: dict[str, list],
    source: str,
) -> dict[date, dict[str, list]]:
    """Split a UTC hourly response into exact-date series."""
    times = hourly.get("time", []) or []
    by_day: dict[date, dict[str, list]] = {}
    for i, raw_time in enumerate(times):
        try:
            day = datetime.fromisoformat(str(raw_time).replace("Z", "")).date()
        except ValueError:
            continue
        series = by_day.setdefault(
            day,
            {"time": [], "_source": source, **{k: [] for k in _HOURLY_KEYS if k != "time"}},
        )
        series["time"].append(raw_time)
        for key in _HOURLY_KEYS:
            if key == "time":
                continue
            values = hourly.get(key, []) or []
            series[key].append(values[i] if i < len(values) else None)
    return by_day


def _parse_batch_response(
    payload: object,
    locations: list[tuple[str, dict]],
    source: str,
) -> dict[tuple[str, date], dict[str, list]]:
    """Parse Open-Meteo's object-or-list response for multiple coordinates."""
    items = payload if isinstance(payload, list) else [payload]
    if len(items) != len(locations):
        logger.warning(
            "Weather batch returned %d locations for %d requested",
            len(items), len(locations),
        )
    out: dict[tuple[str, date], dict[str, list]] = {}
    for (team_code, _), item in zip(locations, items):
        if not isinstance(item, dict):
            continue
        for day, series in _split_hourly_by_utc_day(item.get("hourly", {}) or {}, source).items():
            out[(team_code, day)] = series
    return out


def _fetch_batch_range(
    locations: list[tuple[str, dict]],
    start_date: date,
    end_date: date,
    *,
    source: str,
    forecast: bool = False,
    past_days: int | None = None,
    forecast_days: int | None = None,
) -> dict[tuple[str, date], dict[str, list]]:
    """Fetch one bounded multi-coordinate Open-Meteo request."""
    params = {
        "latitude": ",".join(str(info["lat"]) for _, info in locations),
        "longitude": ",".join(str(info["lon"]) for _, info in locations),
        "hourly": _HOURLY_VARS,
        "timezone": "GMT",
    }
    if forecast:
        params.update({
            "past_days": 1 if past_days is None else past_days,
            "forecast_days": 1 if forecast_days is None else forecast_days,
        })
        url = "https://api.open-meteo.com/v1/forecast"
    else:
        params.update({"start_date": start_date.isoformat(), "end_date": end_date.isoformat()})
        url = "https://archive-api.open-meteo.com/v1/archive"
    try:
        resp = _get_with_retry(url, params=params, timeout=45)
        resp.raise_for_status()
        return _parse_batch_response(resp.json(), locations, source)
    except Exception as exc:
        logger.warning(
            "Weather batch failed (%s → %s, %d locations, source=%s): %s",
            start_date, end_date, len(locations), source, exc,
        )
        return {}


def _fetch_batched_weather(
    locations: list[tuple[str, dict]],
    start_date: date,
    end_date: date,
    needed_days: set | None = None,
) -> dict[tuple[str, date], dict[str, list]]:
    """Fetch historical weather in paced multi-coordinate date batches.

    The previous implementation made one request per stadium-day and ran
    eight workers concurrently. This sends roughly one request per 14-day
    window for all stadiums, then fills only recent archive gaps from the
    forecast past window. Missing responses remain missing; no climatology is
    promoted to an observed record.

    ``needed_days`` (dates that actually have games) lets the loop SKIP
    windows containing no games — MLB's Nov–Feb off-season otherwise costs
    ~10 pointless 14-day batch requests per full-history run, each of which
    Open-Meteo answers with 429s under pacing. ``None`` preserves the old
    fetch-everything behavior for callers without game-date knowledge.
    """
    import time

    def _window_has_games(a: date, b: date) -> bool:
        if needed_days is None:
            return True
        return any(a <= d <= b for d in needed_days)

    by_key: dict[tuple[str, date], dict[str, list]] = {}
    if not locations or start_date > end_date:
        return by_key

    today = date.today()
    archive_end = min(end_date, today - timedelta(days=1))
    skipped_windows = 0

    # Archive requests contain only dates that are definitely historical.
    # Never send today's or a future slate to the archive endpoint.
    if start_date <= archive_end:
        for batch_start in range(0, (archive_end - start_date).days + 1, _WEATHER_BATCH_DAYS):
            chunk_start = start_date + timedelta(days=batch_start)
            chunk_end = min(chunk_start + timedelta(days=_WEATHER_BATCH_DAYS - 1), archive_end)
            if not _window_has_games(chunk_start, chunk_end):
                skipped_windows += 1
                logger.debug(
                    "Weather archive window %s → %s skipped: no scheduled games",
                    chunk_start, chunk_end)
                continue
            for i in range(0, len(locations), _WEATHER_BATCH_SIZE):
                batch = locations[i:i + _WEATHER_BATCH_SIZE]
                by_key.update(_fetch_batch_range(
                    batch, chunk_start, chunk_end, source="open_meteo_archive"
                ))
                if i + _WEATHER_BATCH_SIZE < len(locations):
                    time.sleep(_WEATHER_BATCH_PAUSE_SEC)
            if chunk_end < archive_end:
                time.sleep(_WEATHER_BATCH_PAUSE_SEC)

    # The archive has a publication delay. Recover recent missing dates from
    # the forecast endpoint's observed-past window, still using PIT cutoffs.
    recent_start = max(start_date, today - timedelta(days=91))
    recent_end = min(end_date, today)
    if recent_start <= recent_end and _window_has_games(recent_start, recent_end):
        recent_days = {
            recent_start + timedelta(days=i)
            for i in range((recent_end - recent_start).days + 1)
        }
        if any((team_code, day) not in by_key for team_code, _ in locations for day in recent_days):
            recent = _fetch_batch_range(
                locations, recent_start, recent_end,
                source="open_meteo_forecast_past", forecast=True,
                past_days=(today - recent_start).days + 1,
            )
            for key, series in recent.items():
                by_key.setdefault(key, series)

    # Future slates use the forecast endpoint directly. Open-Meteo supports
    # up to 16 forecast days; dates beyond that remain unavailable rather
    # than being filled with climatology.
    future_start = max(start_date, today)
    if future_start <= end_date:
        forecast_days = min((end_date - today).days + 1, 16)
        future = _fetch_batch_range(
            locations, today, end_date,
            source="open_meteo_forecast", forecast=True,
            past_days=1, forecast_days=forecast_days,
        )
        for key, series in future.items():
            by_key.setdefault(key, series)
    if skipped_windows:
        logger.info(
            "Weather batches: skipped %d gameless window(s) (%s→%s span) — "
            "no Open-Meteo requests sent for them",
            skipped_windows, start_date, archive_end)
    return by_key


def _stadium_weather_utc(
    info: dict,
    game_start_utc: datetime,
    series: dict[str, list],
) -> dict:
    """Build weather from a GMT/UTC series using a strict UTC cutoff."""
    raw = _pick_row(series, _utc_naive(game_start_utc))
    air_density = compute_air_density(
        raw["temp_c"], raw["rh_pct"], raw["pressure_hpa"], info["alt_m"]
    )
    wind_mult = compute_wind_multiplier(
        raw["wind_direction_deg"], raw["wind_speed_kmh"], info["bearing"]
    )
    observed = _has_observation(raw)
    return {
        "available": observed,
        "source": series.get("_source", "open_meteo_archive") if observed else "open_meteo_unavailable",
        "temp_c": raw["temp_c"],
        "rh_pct": raw["rh_pct"],
        "wind_speed_kmh": raw["wind_speed_kmh"],
        "wind_direction_deg": raw["wind_direction_deg"],
        "pressure_hpa": raw["pressure_hpa"],
        "air_density": air_density,
        "wind_multiplier": wind_mult,
        "stadium_alt_m": info["alt_m"],
        "stadium_bearing": info["bearing"],
    }


def _is_missing_key(value: object) -> bool:
    """Return whether a scalar dataframe key is missing."""
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _game_weather_key(row: pd.Series, fallback: object) -> object:
    """Choose one stable key for a weather result.

    ``game_pk`` is the authoritative identifier shared by StatsAPI, the
    results overlay, and the history cache.  ``game_id`` remains the
    fallback for schedule/slate frames that do not carry a game_pk.
    """
    game_pk = pd.to_numeric(row.get("game_pk"), errors="coerce")
    if pd.notna(game_pk):
        return int(game_pk)
    game_id = row.get("game_id")
    return fallback if _is_missing_key(game_id) else game_id


def _weather_key_candidates(value: object) -> list[object]:
    """Return equivalent scalar forms used by mixed CSV/pandas key types."""
    if _is_missing_key(value):
        return []
    candidates: list[object] = [value]
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric):
        candidates.extend([int(numeric), str(int(numeric))])
    else:
        candidates.append(str(value))
    out: list[object] = []
    for candidate in candidates:
        if candidate not in out:
            out.append(candidate)
    return out


def _lookup_weather(
    weather_data: dict,
    row: pd.Series,
    index: object,
) -> dict:
    """Look up weather by game_pk first, then game_id/index fallbacks."""
    if not isinstance(weather_data, dict):
        return {}
    values = [row.get("game_pk"), row.get("game_id"), index]
    for value in values:
        for key in _weather_key_candidates(value):
            if key in weather_data:
                return weather_data[key]
    return {}


def fetch_games_weather(
    games_df: pd.DataFrame,
) -> dict[object, dict]:
    """Fetch point-in-time weather with paced multi-coordinate requests.

    The API is queried in GMT/UTC so each game uses the latest hourly record
    strictly before its actual UTC first pitch. Failed or unavailable days
    return ``available=False`` and are never replaced with climatology.
    Results are keyed by ``game_pk`` when present, otherwise by ``game_id``
    (or the input index as a final fallback).
    """
    results: dict[object, dict] = {}
    if games_df is None or games_df.empty:
        return results

    df = games_df.copy()
    targets: list[tuple[object, str, dict, datetime]] = []
    locations: dict[str, dict] = {}
    for index, row in df.iterrows():
        result_key = _game_weather_key(row, index)
        start = _utc_naive(row.get("start_time_utc"))
        if start is None:
            results[result_key] = {"available": False, "source": "unavailable"}
            continue
        team_code = _resolve_team_code(row.get("home_team", ""), row.get("venue", ""))
        info = STADIUMS.get(team_code)
        if info is None:
            results[result_key] = {"available": False, "source": "unknown_stadium"}
            continue
        targets.append((result_key, team_code, info, start))
        locations[team_code] = info

    if not targets:
        return results

    min_day = min(start.date() for _, _, _, start in targets) - timedelta(days=1)
    max_day = max(start.date() for _, _, _, start in targets)
    series_by_key = _fetch_batched_weather(
        list(locations.items()), min_day, max_day,
        needed_days={s.date() for _, _, _, s in targets},
    )

    for result_key, team_code, info, start in targets:
        series = series_by_key.get((team_code, start.date()))
        if series is None:
            results[result_key] = {"available": False, "source": "open_meteo_unavailable"}
        else:
            results[result_key] = _stadium_weather_utc(info, start, series)

    n_ok = sum(1 for value in results.values() if value.get("available"))
    logger.info("Weather fetched: %d/%d games from batched observations", n_ok, len(results))
    # Coverage gate, per calendar year when the batch spans seasons. A
    # season-wide starvation (bad date range, endpoint change) must be loud,
    # not a healthy-looking aggregate ratio that one good year dilutes.
    if n_ok < 0.8 * len(results):
        logger.warning(
            "Weather fetched only %d/%d games (%.0f%%) — observations "
            "unavailable for many games; wind/air-density features stay NULL",
            n_ok, len(results), 100.0 * n_ok / max(len(results), 1))
    if len(results) >= 40:
        by_year: dict[int, tuple[int, int]] = {}
        for _key, _code, _info, start in targets:
            ok = results.get(_key, {}).get("available")
            tot, good = by_year.get(start.year, (0, 0))
            by_year[start.year] = (tot + 1, good + (1 if ok else 0))
        for year, (tot, good) in sorted(by_year.items()):
            if tot and good < 0.8 * tot:
                logger.warning(
                    "Weather coverage %d: only %d/%d games observed "
                    "— that season's weather features are starving", year, good, tot)
    return results


def fetch_day_weather(
    games_df: pd.DataFrame,
) -> dict[str, dict]:
    """Backward-compatible alias for :func:`fetch_games_weather`."""
    return fetch_games_weather(games_df)


# ── StatsAPI game-feed gap filler ──────────────────────────────────────────

STATSAPI_GAMEFEED_SOURCE = "statsapi_gamefeed"


def _wind_phrase_deg(text: str, bearing: float) -> float | None:
    """Estimate wind direction from the feed's compass phrase.

    StatsAPI phrases look like ``"9 mph, In from CF"`` / ``"12 mph, Out to
    LF"`` / ``"8 mph, L to RF"``. Mapped relative to the outfield bearing:
    blowing OUT → tailwind (bearing), IN → headwind (+180°), L/R → crosswind
    (±45°). Unrecognized phrases return None rather than guessing.
    """
    t = text.lower()
    if "out" in t:
        return float(bearing)
    if "in" in t:
        return float((bearing + 180) % 360)
    if " l" in t or t.startswith("l"):
        return float((bearing - 45) % 360)
    if " r" in t or t.startswith("r"):
        return float((bearing + 45) % 360)
    return None


def statsapi_weather_to_record(
    parsed: dict,
    home_team: str,
    venue: str = "",
) -> dict:
    """Convert :func:`results.fetch_statsapi_weather` output to a cache record.

    Honest-fill rules mirror the module's null contract:
      * wind_multiplier only when BOTH an mph value and a recognizable
        direction phrase exist (real observation → real calculation);
      * air_density stays NULL — gameData.weather carries no humidity and the
        density formula refuses to fabricate RH;
      * ``available`` is True only when the wind multiplier exists, since the
        wind feature is the gap being filled.
    Returns the record with ``available=False`` (and reason in ``source``)
    when nothing usable was parsed.
    """
    info = STADIUMS.get(_resolve_team_code(home_team, venue))
    temp_f = parsed.get("temp_f")
    mph = parsed.get("wind_mph")
    text = parsed.get("wind_text", "")

    temp_c = (temp_f - 32.0) * 5.0 / 9.0 if temp_f is not None else None
    kmh = mph * 1.60934 if mph is not None else None
    mult = None
    deg = None
    if info is not None and kmh is not None:
        deg = _wind_phrase_deg(text, info["bearing"])
        if deg is not None:
            mult = compute_wind_multiplier(deg, kmh, info["bearing"])

    available = mult is not None
    return {
        "available": available,
        "source": STATSAPI_GAMEFEED_SOURCE if available else "statsapi_gamefeed_unusable",
        "temp_c": temp_c,
        "rh_pct": None,
        "wind_speed_kmh": kmh,
        "wind_direction_deg": deg,
        "pressure_hpa": None,
        "air_density": None,
        "wind_multiplier": mult,
        "stadium_alt_m": info["alt_m"] if info else None,
        "stadium_bearing": info["bearing"] if info else None,
    }


# ── Climatology fallback ────────────────────────────────────────────────────

# Retained only as an explicit, opt-in diagnostic helper. It is never treated
# as an observed record by fetch_games_weather or the history cache.
_CLIMO_MONTHLY_TEMP_C = (3.0, 5.0, 9.0, 14.0, 19.0, 24.5, 27.0, 26.0, 21.5, 15.0, 8.0, 4.0)
_CLIMO_RH_PCT = 50.0


def climatology_weather(home_team: str, venue: str, month: int) -> dict:
    """Return an explicitly labeled seasonal norm for diagnostics only.

    ``available`` is deliberately false: climatology is not a point-in-time
    observation and must never enter training features or the weather cache.
    """
    team_code = _resolve_team_code(home_team, venue)
    info = STADIUMS.get(team_code)
    alt_m = float(info["alt_m"]) if info else 100.0
    month = max(1, min(12, int(month)))
    temp_c = _CLIMO_MONTHLY_TEMP_C[month - 1]
    air_density = compute_air_density(temp_c, _CLIMO_RH_PCT, np.nan, alt_m)
    return {
        "available": False,
        "source": "climatology",
        "temp_c": temp_c,
        "rh_pct": _CLIMO_RH_PCT,
        "wind_speed_kmh": 0.0,
        "wind_direction_deg": np.nan,
        "pressure_hpa": np.nan,
        "air_density": air_density if not np.isnan(air_density) else 1.225,
        "wind_multiplier": 0.0,
        "stadium_alt_m": alt_m,
    }


def apply_weather_features(
    df: pd.DataFrame,
    weather_data: dict | None,
) -> pd.DataFrame:
    """Fill wind_advantage_flyball_factor / air_density_velocity_boost on an
    existing diff-feature frame from a ``{game_id: weather}`` mapping.

    Same formulas as ``features.add_diff_features``:
      * wind_advantage_flyball_factor = wind_multiplier × sp_era_diff
      * air_density_velocity_boost   = (air_density − 1.225) × sp_fbvelo_diff
    Dome games are genuinely neutral → a valid 0.  Rows without weather stay
    NULL.  Idempotent: rows absent from ``weather_data`` are left untouched.
    """
    df = df.copy()
    if weather_data is None or df.empty:
        return df
    for col in ("wind_advantage_flyball_factor", "air_density_velocity_boost"):
        if col not in df.columns:
            df[col] = np.nan

    sea_level_rho = 1.225
    n_applied = 0
    # Series-safe: df.get returns None when the column is absent, and
    # to_numeric(None) yields a scalar nan — .iloc on it then crashes.
    # Absent inputs mean the formulas produce NULL, never a fabricated value.
    era = pd.to_numeric(df.get("sp_era_diff"), errors="coerce")
    if not isinstance(era, pd.Series):
        era = pd.Series(np.nan, index=df.index)
    velo = pd.to_numeric(df.get("sp_fbvelo_diff"), errors="coerce")
    if not isinstance(velo, pd.Series):
        velo = pd.Series(np.nan, index=df.index)
    # Preserve values already attached by another weather pass. A fetch often
    # covers only a subset (for example, cache misses), and absence from that
    # subset is not evidence that a prior valid observation should be erased.
    wind_vals = pd.to_numeric(df["wind_advantage_flyball_factor"], errors="coerce").tolist()
    air_vals = pd.to_numeric(df["air_density_velocity_boost"], errors="coerce").tolist()
    for i, (idx, row) in enumerate(df.iterrows()):
        w = _lookup_weather(weather_data, row, idx)
        dome = pd.notna(row.get("dome_is_neutral")) and float(row["dome_is_neutral"]) == 1
        if w.get("available"):
            # Real fetched observation — use the formulas as-is (a dome with
            # a fetch reports wind_multiplier≈0 and its actual indoor density).
            wm = w.get("wind_multiplier", np.nan)
            ad = w.get("air_density", np.nan)
            wv = float(wm) * era.iloc[i] if pd.notna(wm) and pd.notna(era.iloc[i]) else np.nan
            av = (float(ad) - sea_level_rho) * velo.iloc[i] if pd.notna(ad) and pd.notna(velo.iloc[i]) else np.nan
            wind_vals[i] = wv
            air_vals[i] = av
            n_applied += 1
        elif dome:
            # No weather fetched. Indoors the wind component is genuinely
            # zero — a valid 0, but only when the ERA-diff input exists (a
            # missing input keeps it NULL, never a fabricated 0). The
            # air-density boost is UNKNOWN without a fetch, so leave any
            # existing value untouched and otherwise keep it NULL.
            if pd.notna(era.iloc[i]):
                wind_vals[i] = 0.0
        # For non-dome rows with no matching observation, preserve the
        # pre-existing values and leave genuinely missing values as NULL.
    df["wind_advantage_flyball_factor"] = pd.Series(wind_vals, index=df.index, dtype="float64")
    df["air_density_velocity_boost"] = pd.Series(air_vals, index=df.index, dtype="float64")

    # Standalone ENV-LEVEL columns (run engine Phase 3.5b): the LEVEL values
    # behind the interactions, un-multiplied by any SP diff. Additive — the
    # interaction columns above are untouched.
    wm_col = pd.to_numeric(df.get("wind_advantage_flyball_factor"), errors="coerce")
    era2 = pd.to_numeric(df.get("sp_era_diff"), errors="coerce")
    park_wind = []
    air_level = []
    for i, (idx, row) in enumerate(df.iterrows()):
        w = _lookup_weather(weather_data, row, idx)
        if w.get("available"):
            wm = w.get("wind_multiplier", np.nan)
            ad = w.get("air_density", np.nan)
            park_wind.append(float(wm) if pd.notna(wm) else np.nan)
            air_level.append(float(ad) if pd.notna(ad) else np.nan)
        else:
            # No fetched observation: a dome's wind level is genuinely 0
            # (only when the interaction confirms it), anything else UNKNOWN.
            wv = wm_col.iloc[i] if isinstance(wm_col, pd.Series) else np.nan
            ev = era2.iloc[i] if isinstance(era2, pd.Series) else np.nan
            dome = pd.notna(row.get("dome_is_neutral")) and float(row["dome_is_neutral"]) == 1
            park_wind.append(float(wv) if dome and pd.notna(wv)
                             and abs(float(ev)) > 1e-12 else np.nan)
            air_level.append(np.nan)
    df["park_wind_factor"] = pd.Series(park_wind, index=df.index, dtype="float64")
    df["air_density_level"] = pd.Series(air_level, index=df.index, dtype="float64")
    logger.info("Weather features applied to %d/%d games", n_applied, len(df))
    return df
