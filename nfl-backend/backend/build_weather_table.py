"""Build the committed observed-weather table ``nfl_weather.csv``.

For every decided game in the nflverse schedules payload, this program pulls
the Open-Meteo *archive* API (free, no key) for the venue's coordinates on
the game date and writes one row per (stadium, gameday):

    stadium, gameday, temp_f, wind_mph, precip_in, snow_in, source

The table feeds ``features._attach_weather``: temp_f / wind_mph / is_precip /
is_snow served features. Only OUTDOOR games are fetched — domes and indoor
venues have no meaningful outdoor observation and their served values stay
NaN (never fabricated). The schedule's own temp/wind payload columns remain
the fallback for any row this table lacks.

Usage (from nfl-backend/backend/):

    python build_weather_table.py               # incremental: only missing rows
    python build_weather_table.py --refresh     # re-pull every decided game

A one-row-per-day courtesy delay keeps the public API happy; a full
2018-2026 build is a few thousand requests and runs in roughly 10 minutes.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
    import ingestion
else:  # pragma: no cover - package import path
    from . import config, ingestion

logger = logging.getLogger("build_weather_table")

WEATHER_FILE = config.BACKEND_DIR / "nfl_weather.csv"
SOURCE_TAG = "open-meteo-archive"

# Archive API: daily aggregates on the venue's grid cell. Columns map to the
# served features: temperature_2m_mean -> temp_f, wind_gusts_10m_max is the
# best observed proxy for game-day wind, rain+showers -> precip_in,
# snowfall (cm -> inches) -> snow_in. (The archive's daily catalog has no
# drizzle variable — drizzle exists only in the hourly API.)
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_COLS = ("temperature_2m_mean", "wind_gusts_10m_max",
              "rain_sum", "showers_sum", "snowfall_sum")
CM_PER_IN = 2.54
# ERA5 archive lags roughly a week behind today; requesting later days is a
# guaranteed 400. Clamped here so incremental runs stay clean; the most
# recent games fall back to the schedule payload until the archive catches up.
ARCHIVE_LAG_DAYS = 7


def decided_outdoor_games() -> pd.DataFrame:
    """Decided regular-season games with an outdoor venue and coordinates."""
    schedule = ingestion.load_schedule(use_cache=True)
    schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce")
    decided = schedule[schedule["home_score"].notna()
                       & schedule["away_score"].notna()].copy()
    decided["season"] = pd.to_numeric(decided["season"], errors="coerce")
    decided = decided[decided["season"].isin(config.ALL_SEASONS)]

    stadiums = pd.read_csv(config.BACKEND_DIR / "nfl_stadiums.csv")
    coords = stadiums.set_index("stadium")[["lat", "lon"]]
    decided = decided.join(coords, on="stadium")

    # Indoor venues: no outdoor observation exists. nflverse roof values:
    # outdoors / dome / closed / open / retractable. "open" and "retractable"
    # had (at least partially) open roofs — outdoor conditions applied — so
    # only sealed venues are excluded; missing roof falls through to fetch.
    roof = decided.get("roof", pd.Series(index=decided.index, dtype="object"))
    roof_norm = roof.astype("string").str.strip().str.lower().fillna("")
    decided = decided[~roof_norm.isin(["dome", "closed"])]
    # Defensive second gate: any venue the stadium table can't geolocate.
    decided = decided[decided["lat"].notna() & decided["lon"].notna()]
    return decided


def _fetch_range(lat: float, lon: float, days: list[str]) -> pd.DataFrame:
    """One archive request spanning days[0]..days[-1]; observations indexed
    by day for the requested days only."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": days[0], "end_date": days[-1],
        "daily": ",".join(DAILY_COLS),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "UTC",
    }
    for attempt in range(3):
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            daily = r.json()["daily"]
            dates = pd.to_datetime(daily["time"]).strftime("%Y-%m-%d")
            # A dry day is a real 0.0 observation, not missing: only when the
            # API returns null for ALL rain/drizzle/showers fields do we keep
            # precipitation unknown (None).
            rain = pd.DataFrame({k: daily[k] for k in
                                 ("rain_sum", "showers_sum")})
            precip = (rain.sum(axis=1).round(4)
                      .where(rain.notna().any(axis=1), other=None)
                      .tolist())
            snow = [None if v is None else round(v / CM_PER_IN, 4)
                    for v in daily["snowfall_sum"]]
            obs = pd.DataFrame({
                "temp_f": daily["temperature_2m_mean"],
                "wind_mph": daily["wind_gusts_10m_max"],
                "precip_in": precip,
                "snow_in": snow,
            }, index=dates)
            return obs.loc[obs.index.intersection(days)]
        except Exception:
            if attempt == 2:
                return pd.DataFrame()
            time.sleep(2 * (attempt + 1))
    return pd.DataFrame()


def _fetch_stadium(lat: float, lon: float, days: list[str]) -> pd.DataFrame:
    """Archive observations for a stadium's game days. The archive edge
    varies by grid cell, so a single long range can 400 wholesale; on
    failure the range is split recursively until the offending days are
    isolated (they fall back to the schedule payload instead)."""
    if not days:
        return pd.DataFrame()
    obs = _fetch_range(lat, lon, days)
    if not obs.empty or len(days) == 1:
        if obs.empty:
            logger.warning("no archive observation for (%s, %s) on %s",
                           lat, lon, days[0])
        return obs
    mid = len(days) // 2
    left = _fetch_stadium(lat, lon, days[:mid])
    right = _fetch_stadium(lat, lon, days[mid:])
    return pd.concat([left, right]) if not left.empty or not right.empty \
        else pd.DataFrame()


def build(refresh: bool = False) -> pd.DataFrame:
    games = decided_outdoor_games()
    games["gameday"] = games["gameday"].dt.date.astype(str)
    keys = games[["stadium", "gameday", "lat", "lon"]].drop_duplicates(
        ["stadium", "gameday"])
    logger.info("decided outdoor games: %d rows, %d unique (stadium, gameday)",
                len(games), len(keys))

    existing = pd.DataFrame(columns=["stadium", "gameday"])
    if WEATHER_FILE.exists() and not refresh:
        existing = pd.read_csv(WEATHER_FILE)[["stadium", "gameday"]]
    have = set(map(tuple, existing.to_numpy()))
    edge = (pd.Timestamp.utcnow().tz_localize(None).normalize()
            - pd.Timedelta(days=ARCHIVE_LAG_DAYS)).date()
    todo = [tuple(k) for k in keys[["stadium", "gameday"]].to_numpy()
            if tuple(k) not in have and pd.Timestamp(k[1]).date() <= edge]
    skipped = len(keys) - len(todo) - len(have & set(map(tuple, keys[["stadium", "gameday"]].to_numpy())))
    if skipped > 0:
        logger.info("%d recent games beyond the archive edge (%s) left to "
                    "the schedule fallback; re-run later to backfill",
                    skipped, edge)
    logger.info("rows to fetch: %d (have %d)", len(todo), len(have))

    latlon = keys.set_index(["stadium", "gameday"])[["lat", "lon"]]
    rows: list[dict] = []
    stadiums_todo = sorted({s for s, _ in todo})
    logger.info("stadiums to fetch: %d (batched date-range requests)",
                len(stadiums_todo))
    for i, stadium in enumerate(stadiums_todo, 1):
        days = sorted(d for s, d in todo if s == stadium)
        lat = float(latlon.loc[(stadium, days[0]), "lat"])
        lon = float(latlon.loc[(stadium, days[0]), "lon"])
        obs = _fetch_stadium(lat, lon, days)
        for day in days:
            if day in obs.index:
                rows.append({"stadium": stadium, "gameday": day,
                             **obs.loc[day].to_dict(), "source": SOURCE_TAG})
        logger.info("[%d/%d] %s: %d/%d days observed",
                    i, len(stadiums_todo), stadium, len(obs), len(days))
        time.sleep(1.0)  # courtesy rate limit for the free API

    fresh = pd.DataFrame(rows,
                         columns=["stadium", "gameday", "temp_f", "wind_mph",
                                  "precip_in", "snow_in", "source"])
    if WEATHER_FILE.exists() and not refresh:
        old = pd.read_csv(WEATHER_FILE)
        fresh = pd.concat([old, fresh], ignore_index=True)
    fresh = fresh.drop_duplicates(["stadium", "gameday"], keep="last")
    fresh = fresh.sort_values(["gameday", "stadium"]).reset_index(drop=True)
    fresh.to_csv(WEATHER_FILE, index=False)
    logger.info("wrote %s: %d rows", WEATHER_FILE.name, len(fresh))
    return fresh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="re-pull every decided game (default: incremental)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(message)s")
    build(refresh=args.refresh)


if __name__ == "__main__":
    main()
