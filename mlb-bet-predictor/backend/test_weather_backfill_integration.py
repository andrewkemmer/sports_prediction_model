"""Integration test: retroactive weather backfill over a REAL 2026 range.

The start-time truncation fix only matters if the weather chain actually
repopulates already-processed past games. This test runs the real chain —
StatsAPI schedule (chunked) → Open-Meteo archive → _attach_weather_history
cache — over a small decided-2026 window and asserts open-air games get
REAL observations, not dome-default zeros.

Network-gated: skipped unless RUN_NETWORK_TESTS=1 so CI stays offline.
"""
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from results import fetch_game_start_times


@unittest.skipUnless(
    os.getenv("RUN_NETWORK_TESTS") == "1",
    "hits live StatsAPI + Open-Meteo; set RUN_NETWORK_TESTS=1 to run")
class TestRetroactiveWeatherBackfill(unittest.TestCase):
    JULY_4TH_WEEKEND = (date(2026, 7, 3), date(2026, 7, 5))

    def _pick_games(self, starts: dict, limit: int = 6) -> list[tuple]:
        """Deterministically pick a few games: (game_pk, game_date).

        Start times are ISO-8601 UTC strings per the fetch_game_start_times
        contract.
        """
        picked = []
        for pk in sorted(starts):
            day = pd.Timestamp(starts[pk]).date()
            picked.append((int(pk), day))
            if len(picked) >= limit:
                break
        return picked

    def test_past_2026_games_get_real_observations(self):
        start_d, end_d = self.JULY_4TH_WEEKEND
        starts = fetch_game_start_times(start_d, end_d)
        self.assertGreater(len(starts), 20,
                           "chunked schedule returned almost nothing")

        from pipeline import _attach_weather_history

        picked = self._pick_games(starts)
        mini = pd.DataFrame({
            "game_pk": [pk for pk, _ in picked],
            "game_date": pd.to_datetime([d for _, d in picked]),
            "home_team": ["NYY"] * len(picked),
            "home_win": [1.0] * len(picked),
            "dome_is_neutral": [0.0] * len(picked),
            # Unit inputs so the formulas produce numbers wherever an
            # observation exists.
            "sp_era_diff": [1.0] * len(picked),
            "sp_fbvelo_diff": [1.0] * len(picked),
        })

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        old_cache_dir = os.environ.get("MLB_CACHE_DIR")
        os.environ["MLB_CACHE_DIR"] = tmp.name
        try:
            out = _attach_weather_history(mini, end_d)
        finally:
            if old_cache_dir is None:
                os.environ.pop("MLB_CACHE_DIR", None)
            else:
                os.environ["MLB_CACHE_DIR"] = old_cache_dir

        wind = pd.to_numeric(out["wind_advantage_flyball_factor"], errors="coerce")
        air = pd.to_numeric(out["air_density_velocity_boost"], errors="coerce")
        n = len(out)
        pct_wind_measured = 100.0 * (wind.notna() & ~((wind == 0.0))).sum() / n
        pct_air_measured = 100.0 * air.notna().sum() / n

        # THE regression: past decided 2026 games must receive real archive
        # observations — not dome-default zeros, not NULL.
        self.assertGreaterEqual(pct_wind_measured, 80.0,
                                f"wind measured in only {pct_wind_measured:.0f}% of past 2026 games")
        self.assertGreaterEqual(pct_air_measured, 80.0,
                                f"air density present in only {pct_air_measured:.0f}% of past 2026 games")
        # Cache persisted for these game_pks (retroactive + durable).
        cache_path = Path(tmp.name) / "weather_history.parquet"
        self.assertTrue(cache_path.exists())
        cached = pd.read_parquet(cache_path)
        self.assertGreaterEqual(len(cached), int(0.8 * n))


if __name__ == "__main__":
    unittest.main()
