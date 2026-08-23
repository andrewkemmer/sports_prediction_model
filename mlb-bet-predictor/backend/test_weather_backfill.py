"""Regression tests for the full-history weather backfill (_attach_weather_history).

Verifies the cache-backed flow: only decided games missing from the cache
are fetched, results persist to the cache file, re-runs skip fetching, and
apply_weather_features fills the two weather-driven columns from cache.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline import _attach_weather_history, _load_weather_cache


def _games() -> pd.DataFrame:
    """3 decided games + 1 undecided (not fetched, stays null)."""
    return pd.DataFrame([
        {"game_pk": 101, "game_id": "20260801_NYY@BOS", "game_date": "2026-08-01",
         "home_team": "BOS", "venue": "Fenway Park", "home_win": 1.0,
         "dome_is_neutral": 0.0, "sp_era_diff": 0.5, "sp_fbvelo_diff": 1.0},
        {"game_pk": 102, "game_id": "20260802_CHC@MIL", "game_date": "2026-08-02",
         "home_team": "MIL", "venue": "American Family Field", "home_win": 0.0,
         "dome_is_neutral": 1.0, "sp_era_diff": -0.3, "sp_fbvelo_diff": -0.5},
        {"game_pk": 103, "game_id": "20260803_TEX@HOU", "game_date": "2026-08-03",
         "home_team": "HOU", "venue": "Minute Maid Park", "home_win": 1.0,
         "dome_is_neutral": 1.0, "sp_era_diff": np.nan, "sp_fbvelo_diff": 2.0},
        {"game_pk": 104, "game_id": "20260804_ATL@MIA", "game_date": "2026-08-04",
         "home_team": "MIA", "venue": "LoanDepot Park", "home_win": np.nan,
         "dome_is_neutral": 1.0, "sp_era_diff": 0.2, "sp_fbvelo_diff": 0.3},
    ])


_WEATHER = {
    "20260801_NYY@BOS": {"available": True, "wind_multiplier": 1.0,
                         "air_density": 1.19, "temp_c": 26.0, "rh_pct": 60.0,
                         "wind_speed_kmh": 15.0, "wind_direction_deg": 180.0,
                         "pressure_hpa": 1013.0, "stadium_alt_m": 10.0,
                         "stadium_bearing": 90},
    "20260802_CHC@MIL": {"available": True, "wind_multiplier": 0.0,
                         "air_density": 1.20, "temp_c": 25.0, "rh_pct": 50.0,
                         "wind_speed_kmh": 5.0, "wind_direction_deg": 90.0,
                         "pressure_hpa": 1015.0, "stadium_alt_m": 200.0,
                         "stadium_bearing": 45},
    "20260803_TEX@HOU": {"available": True, "wind_multiplier": 0.0,
                         "air_density": 1.21, "temp_c": 30.0, "rh_pct": 70.0,
                         "wind_speed_kmh": 8.0, "wind_direction_deg": 0.0,
                         "pressure_hpa": 1010.0, "stadium_alt_m": 100.0,
                         "stadium_bearing": 0},
}

_STARTS = {101: "2026-08-01T19:05:00", 102: "2026-08-02T18:40:00",
           103: "2026-08-03T19:10:00", 104: "2026-08-04T18:10:00"}


class TestWeatherHistoryBackfill(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_path = Path(self._tmp.name) / "weather_history.parquet"
        patcher = patch("pipeline._weather_cache_path",
                        return_value=self.cache_path)
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        patcher.start()

    def _wx(self, game_id, **kw):
        w = dict(_WEATHER[game_id])
        w.update(kw)
        return w

    @patch("weather.fetch_games_weather")
    @patch("results.fetch_game_start_times")
    def test_first_run_fetches_decided_games_and_caches(self, mock_starts, mock_wx):
        mock_starts.return_value = dict(_STARTS)
        mock_wx.return_value = {gid: self._wx(gid) for gid in _WEATHER}

        out = _attach_weather_history(_games(), date(2026, 8, 4))

        # Only the 3 decided games fetched — the undecided game (104) skipped.
        self.assertEqual(mock_wx.call_count, 1)
        fetched = mock_wx.call_args[0][0]
        self.assertEqual(sorted(fetched["game_pk"]), [101, 102, 103])

        # Cache persisted with exactly those 3.
        cache = _load_weather_cache(self.cache_path)
        self.assertEqual(sorted(cache), [101, 102, 103])

        # Weather columns filled for decided games; undecided stays null.
        self.assertTrue(np.isfinite(out["wind_advantage_flyball_factor"].iloc[0]))
        # Dome games get valid neutral 0.
        self.assertEqual(float(out["wind_advantage_flyball_factor"].iloc[1]), 0.0)
        # Dome with MISSING sp_era_diff -> NULL, never a fabricated 0.
        self.assertTrue(pd.isna(out["wind_advantage_flyball_factor"].iloc[2]))
        self.assertTrue(pd.isna(out["air_density_velocity_boost"].iloc[3]))

    @patch("weather.fetch_games_weather")
    @patch("results.fetch_game_start_times")
    def test_rerun_skips_fetch_when_cached(self, mock_starts, mock_wx):
        mock_starts.return_value = dict(_STARTS)
        mock_wx.return_value = {gid: self._wx(gid) for gid in _WEATHER}
        _attach_weather_history(_games(), date(2026, 8, 4))
        self.assertEqual(mock_wx.call_count, 1)

        # Second run: everything cached -> no fetch calls at all.
        out = _attach_weather_history(_games(), date(2026, 8, 4))
        self.assertEqual(mock_starts.call_count, 1)
        self.assertEqual(mock_wx.call_count, 1)
        self.assertTrue(np.isfinite(out["wind_advantage_flyball_factor"].iloc[0]))

    def test_cache_roundtrip_handles_nan(self):
        from pipeline import _save_weather_cache
        cache = {7: {"available": True, "temp_c": 22.0, "rh_pct": None,
                     "wind_speed_kmh": None, "wind_direction_deg": None,
                     "pressure_hpa": None, "air_density": 1.19,
                     "wind_multiplier": 1.0, "stadium_alt_m": None,
                     "stadium_bearing": None}}
        _save_weather_cache(self.cache_path, cache)
        back = _load_weather_cache(self.cache_path)
        self.assertEqual(back[7]["temp_c"], 22.0)
        self.assertIsNone(back[7]["rh_pct"])


if __name__ == "__main__":
    unittest.main()
