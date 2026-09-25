"""Build the production NFL point-in-time Open-Meteo weather cache.

This command delegates to :mod:`weather`, the single production weather
provider. For every outdoor stadium game it stores the latest **hourly**
Open-Meteo record whose API timestamp is strictly before kickoff:

    game_id, stadium, kickoff_utc, weather_time_utc, fetched_at_utc,
    source, temp_f, wind_mph, precip_in, snow_in, is_precip, is_snow

Past games use the hourly archive (with the forecast endpoint's recent
observed-past fallback); pending games use hourly forecasts. Forecast rows
must themselves have been fetched before kickoff. Indoor/closed games and
missing hourly values remain unavailable.

The cache defaults to ``../../.nfl_cache/weather_pit_v1.parquet`` (the same
repo-root cache area used by NFL ingestion). The committed ``nfl_weather.csv``
is a legacy daily-aggregate archive/reference artifact; production never
reads it.

Usage (from ``nfl-backend/backend/``)::

    python build_weather_table.py             # incremental cache build
    python build_weather_table.py --refresh   # re-fetch requested games
    python build_weather_table.py --cache /path/to/weather.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config
    import ingestion
    import weather
else:  # pragma: no cover - package import path
    from . import config, ingestion, weather

logger = logging.getLogger("build_weather_table")


def build(refresh: bool = False,
          path: str | Path | None = None) -> pd.DataFrame:
    """Build/refresh the PIT cache for the configured NFL schedule window."""
    schedule = ingestion.eligible_games(
        ingestion.load_schedule(seasons=config.ALL_SEASONS, use_cache=True))
    target = weather.cache_path(path)
    logger.info(
        "building hourly PIT weather for %d schedule rows -> %s",
        len(schedule), target,
    )
    result = weather.fetch_games_weather(schedule, refresh=refresh, path=path)
    logger.info(
        "PIT weather cache ready: %d validated game rows (latest hour strictly "
        "before kickoff)", len(result),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true",
        help="re-fetch requested games while preserving failed-refresh cache rows",
    )
    parser.add_argument(
        "--cache", default=None,
        help="override the external PIT weather-cache path",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build(refresh=args.refresh, path=args.cache)


if __name__ == "__main__":
    main()
