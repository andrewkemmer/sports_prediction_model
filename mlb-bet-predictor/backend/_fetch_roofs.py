#!/usr/bin/env python3
"""Backfill the StatsAPI roof-state cache for retractable-roof games.

Reads game_level_features.csv, identifies retractable-home games missing
from statsapi_roof_cache.json, and fetches each game_pk's
gameData.weather.condition from the StatsAPI live feed.

Idempotent: re-runs only fetch games still missing from the cache.
Budget-capped: pauses and exits cleanly when time runs out.

Usage:
    python _fetch_roofs.py              # fetch missing, respect budget
    python _fetch_roofs.py --dry-run    # show what would be fetched
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="  %(message)s")
logger = logging.getLogger(__name__)

DATA_DELIVERY_DIR = Path(__file__).resolve().parents[1] / "data_delivery"
FEATURES_CSV = DATA_DELIVERY_DIR / "game_level_features.csv"
ROOF_CACHE = DATA_DELIVERY_DIR / "statsapi_roof_cache.json"

RETRACTABLE_TEAMS = frozenset({"ARI", "AZ", "HOU", "MIA", "MIL", "SEA", "TEX", "TOR"})

FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
PAUSE_SEC = 0.5          # ~2 req/s (matches pipeline's _LINEUP_PAUSE_SEC)
BUDGET_SEC = 128.0       # hard wall — exit cleanly when exceeded
RETRIES = 2              # per-game retry count


def _roof_from_condition(condition: str | None) -> str | None:
    """Map gameData.weather.condition to a roof state.

    Retractable-roof parks:
      "Roof Closed" / "Closed Roof" / "Indoor" -> "closed"
      Anything with real weather (Clear, Sunny, etc.) -> "open"
      Empty / missing -> None (unknown — skip, retry next run)
    """
    if condition is None:
        return None
    text = str(condition).strip().lower()
    if not text:
        return None
    if ("roof closed" in text or text == "indoor" or "closed roof" in text
            or text == "dome"):
        return "closed"
    # Any real weather description means the roof was open
    if any(w in text for w in ("clear", "sunny", "cloud", "rain", "snow",
                                "drizzle", "overcast", "fog", "wind",
                                "hot", "cold", "warm", "cool", "fair",
                                "partly", "mostly", "hazy", "mist")):
        return "open"
    # "Roof Open" explicitly
    if "roof open" in text or "open roof" in text:
        return "open"
    return None


def _load_cache() -> dict[int, str]:
    if not ROOF_CACHE.exists():
        return {}
    raw = json.loads(ROOF_CACHE.read_text())
    return {int(k): v for k, v in raw.items() if v in ("open", "closed")}


def _save_cache(cache: dict[int, str]) -> None:
    ROOF_CACHE.write_text(json.dumps({str(k): v for k, v in sorted(cache.items())},
                                      indent=2, sort_keys=True) + "\n")


def _fetch_condition(pk: int) -> str | None:
    """Fetch gameData.weather.condition for a single game_pk."""
    for attempt in range(RETRIES):
        try:
            resp = requests.get(FEED_URL.format(pk=pk), timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return (data.get("gameData") or {}).get("weather", {}).get("condition")
        except Exception:
            pass
        if attempt < RETRIES - 1:
            time.sleep(PAUSE_SEC * 3)
        time.sleep(PAUSE_SEC)
    return None


def backfill(dry_run: bool = False) -> dict:
    """Run the backfill. Returns a summary dict."""
    start = time.monotonic()

    # Load current state
    df = pd.read_csv(FEATURES_CSV)
    cache = _load_cache()

    # Find retractable-home games missing from cache
    home = df["home_team"].astype(str).str.upper().str.strip()
    retract_mask = home.isin(RETRACTABLE_TEAMS)
    retract_pks = set(int(pk) for pk in df.loc[retract_mask, "game_pk"].tolist())
    missing = sorted(retract_pks - set(cache.keys()))

    logger.info("Roof backfill: %d retractable-home games, %d cached, %d missing",
                len(retract_pks), len(retract_pks & set(cache.keys())), len(missing))

    if not missing:
        logger.info("Nothing to fetch — cache is complete for retractable games")
        return {"fetched": 0, "skipped": 0, "errors": 0, "total_missing": 0}

    if dry_run:
        # Show year breakdown
        date_map = dict(zip(df["game_pk"], df["game_date"]))
        by_year = {}
        for pk in missing:
            yr = str(date_map.get(pk, ""))[:4]
            by_year[yr] = by_year.get(yr, 0) + 1
        for yr in sorted(by_year):
            logger.info("  %s: %d missing", yr, by_year[yr])
        return {"fetched": 0, "skipped": 0, "errors": 0,
                "total_missing": len(missing), "by_year": by_year}

    fetched = 0
    errors = 0
    for i, pk in enumerate(missing):
        elapsed = time.monotonic() - start
        if elapsed >= BUDGET_SEC:
            logger.info("Budget exhausted after %d/%d fetches (%.0fs elapsed) — "
                        "remaining %d will be picked up next run",
                        fetched, len(missing), elapsed, len(missing) - i)
            break

        condition = _fetch_condition(pk)
        roof = _roof_from_condition(condition)
        if roof is not None:
            cache[pk] = roof
            fetched += 1
        else:
            errors += 1  # will retry on next run

        # Progress log every 50 games
        if (i + 1) % 50 == 0:
            logger.info("  Progress: %d/%d fetched, %d errors, %.0fs elapsed",
                        fetched, i + 1, errors, time.monotonic() - start)

    # Save updated cache
    if fetched > 0:
        _save_cache(cache)
        logger.info("Cache saved: %d total entries (%d new)", len(cache), fetched)

    # Report per-team resolution for missing that remain
    remaining = sorted(set(missing) - set(cache.keys()))
    if remaining:
        date_map = dict(zip(df["game_pk"], df["game_date"]))
        team_map = dict(zip(df["game_pk"], home))
        by_team = {}
        by_year = {}
        for pk in remaining:
            t = team_map.get(pk, "?")
            yr = str(date_map.get(pk, ""))[:4]
            by_team[t] = by_team.get(t, 0) + 1
            by_year[yr] = by_year.get(yr, 0) + 1
        logger.info("Still missing: %d games (teams: %s)", len(remaining), dict(by_team))
        for yr in sorted(by_year):
            logger.info("  %s: %d still missing", yr, by_year[yr])

    return {"fetched": fetched, "skipped": len(missing) - fetched - errors,
            "errors": errors, "total_missing": len(missing),
            "still_missing": len(remaining)}


def main():
    dry_run = "--dry-run" in sys.argv
    result = backfill(dry_run=dry_run)
    logger.info("Summary: %s", json.dumps(result))


if __name__ == "__main__":
    main()
