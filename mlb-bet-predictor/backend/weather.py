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
from datetime import datetime, date, timedelta
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
        # Archive API (historical) — free, no key.  Note the archive has
        # ~5-day latency; dates inside that window return no rows, which
        # correctly yields NaN (null) features instead of stale values.
        params["start_date"] = archive_date.isoformat()
        params["end_date"] = archive_date.isoformat()
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        # Forecast API — free, no key (covers 7–16 days ahead)
        params["forecast_days"] = 16
        url = "https://api.open-meteo.com/v1/forecast"

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Weather fetch failed for %.2f,%.2f: %s", lat, lon, e)
        return None

    hourly = data.get("hourly", {})
    if not hourly.get("time"):
        return None
    return {k: hourly.get(k, []) for k in _HOURLY_KEYS}


def _pick_row(series: dict[str, list] | None, game_local_time: datetime) -> dict:
    """Pick the strictly-prior hourly row from a day's series (NaN when missing)."""
    if series is None:
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
        "available": True,
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

    return {
        "available": True,
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


def fetch_games_weather(
    games_df: pd.DataFrame,
) -> dict[str, dict]:
    """Weather for every game in a frame — one request per (stadium, day).

    Expects columns: game_id (or the frame index), home_team, venue,
    start_time_utc.  All games at the same stadium on the same local date
    share a single Open-Meteo request; each game then picks its own
    STRICTLY-PRIOR hourly row (point-in-time).

    Returns dict keyed by game_id → weather dict (available=False when the
    stadium is unknown or the start time is missing — those games yield
    null weather features, never fabricated zeros).
    """
    results: dict[str, dict] = {}
    if games_df is None or games_df.empty:
        return results

    df = games_df.copy()
    if "game_id" not in df.columns:
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "game_id"})

    # Cache one day-series per (stadium team code, local date)
    series_cache: dict[tuple[str, date], dict[str, list] | None] = {}

    def _series(team_code: str, info: dict, start_utc: datetime) -> dict[str, list] | None:
        local = _localize(start_utc, info["tz"], info["lon"])
        key = (team_code, local.date())
        if key not in series_cache:
            series_cache[key] = _fetch_hourly_series(
                info["lat"], info["lon"], local.date()
            )
        return series_cache[key]

    for gid, row in df.iterrows():
        game_id = row.get("game_id", str(gid))
        home = row.get("home_team", "")
        venue = row.get("venue", "")
        start = row.get("start_time_utc")
        if start is None or (isinstance(start, float) and pd.isna(start)):
            results[game_id] = {"available": False}
            continue
        if isinstance(start, str):
            start = pd.Timestamp(start).to_pydatetime()
        elif isinstance(start, pd.Timestamp):
            start = start.to_pydatetime()

        team_code = _resolve_team_code(home, venue)
        info = STADIUMS.get(team_code)
        if info is None:
            results[game_id] = {"available": False}
            continue

        results[game_id] = _stadium_weather(info, start, _series(team_code, info, start))

    n_ok = sum(1 for v in results.values() if v.get("available"))
    logger.info("Weather fetched: %d/%d games", n_ok, len(results))
    return results


def fetch_day_weather(
    games_df: pd.DataFrame,
) -> dict[str, dict]:
    """Backward-compatible alias for :func:`fetch_games_weather`."""
    return fetch_games_weather(games_df)
