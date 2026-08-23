"""
Regression tests for the weather-backfill remediation.

Covers the INSUFFICIENT-sample fix: decided history must receive real
point-in-time weather (StatsAPI first pitches -> strictly-prior hourly
observation), keyed correctly onto frames without a game-id column, with
dome games getting genuine neutral zeros and unmatched games staying null.
"""
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import weather


def _games_frame():
    idx = [100, 101, 102, 103]
    return pd.DataFrame({
        "game_date": pd.Timestamp("2026-08-20"),
        "home_win": [1.0, 0.0, 1.0, 0.0],
        "home_team": ["HOU", "LAD", "TOR", "CHC"],
        "away_team": ["ATH", "SF", "NYY", "PIT"],
        "dome_is_neutral": [0.0, 0.0, 1.0, 0.0],
        "sp_era_diff": [0.5, -0.2, 1.0, np.nan],
        "sp_fbvelo_diff": [1.0, 0.5, -0.5, 2.0],
    }, index=idx)


class TestApplyWeatherFeatures(unittest.TestCase):
    def test_values_land_on_index_labels_and_domes_get_valid_zero(self):
        df = _games_frame()
        wx = {
            "100": {"available": True, "wind_multiplier": 0.4, "air_density": 1.185},
            101: {"available": True, "wind_multiplier": -0.5, "air_density": 1.225},
            # 102 = dome -> valid 0; 103 missing -> stays null
        }
        out = weather.apply_weather_features(df, wx)
        w = out["wind_advantage_flyball_factor"]
        a = out["air_density_velocity_boost"]
        self.assertAlmostEqual(w.loc[100], 0.4 * 0.5)
        self.assertAlmostEqual(w.loc[101], -0.5 * -0.2)
        self.assertEqual(w.loc[102], 0.0)                      # dome zero
        self.assertTrue(np.isnan(w.loc[103]))                  # no weather -> null
        self.assertAlmostEqual(a.loc[100], (1.185 - 1.225) * 1.0)
        self.assertEqual(a.loc[102], 0.0)
        # 103 has a velo diff input but no weather: air stays null too
        self.assertTrue(np.isnan(a.loc[103]))

    def test_empty_weather_dict_nulls_non_dome_rows_only(self):
        """No weather data -> non-dome rows stay NULL; dome rows keep their
        genuine structural zero (closed roof = no wind effect, independent
        of any API pull)."""
        df = _games_frame()
        out = weather.apply_weather_features(df, {})
        w = out["wind_advantage_flyball_factor"]
        a = out["air_density_velocity_boost"]
        self.assertTrue(w.drop(102).isna().all())
        self.assertEqual(w.loc[102], 0.0)
        self.assertTrue(a.drop(102).isna().all())
        self.assertEqual(a.loc[102], 0.0)


class TestFetchGamesWeatherKeying(unittest.TestCase):
    def test_index_keyed_frame_resolves_every_game(self):
        """A frame without game_id (like game_level_features.csv) must not
        collapse to a single weather entry."""
        df = _games_frame()
        starts_utc = datetime(2026, 8, 20, 23, 5, tzinfo=timezone.utc)
        wx_df = pd.DataFrame({
            "home_team": df["home_team"].values,
            "venue": "",
            "start_time_utc": [starts_utc] * len(df),
        }, index=df.index)
        orig = weather.fetch_games_weather(wx_df)
        keys = set(orig.keys())
        self.assertEqual(len(keys), len(df))
        self.assertTrue(all(v.get("available") for v in orig.values()))


if __name__ == "__main__":
    unittest.main()
