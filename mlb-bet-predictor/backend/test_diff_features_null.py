"""Null-not-zero + alias + weather behavior for add_diff_features.

Rules under test (per the project's PIT/null contract):
  * A feature whose raw observation is missing is NaN — never a fabricated 0.
  * 0 is reserved for real valid observations (calm wind, dome, cross wind,
    a genuinely zero diff) and real valid calculations.
  * Renamed raw columns are resolved through RAW_COLUMN_ALIASES.
  * Features 30–31 (wind_advantage_flyball_factor, air_density_velocity_boost)
    are NaN without real weather and populated when weather_data is supplied.
"""
import unittest

import numpy as np
import pandas as pd

from backend.data_ingestion import enrich_elo_and_records, load_game_features
from backend.features import add_diff_features, RAW_COLUMN_ALIASES


def _raw() -> pd.DataFrame:
    """Complete raw frame: every observation present, 2 games."""
    return pd.DataFrame([
        {
            "game_id": "g1", "home_team": "NYY", "away_team": "BOS",
            "home_win_pct": 0.60, "away_win_pct": 0.55,
            "home_elo": 1520.0, "away_elo": 1490.0,
            "rest_days_home": 1, "rest_days_away": 2,
            "sp_era_home": 3.2, "sp_era_away": 4.1,
            "sp_era_30g_home": 3.0, "sp_era_30g_away": 4.4,
            "sp_k9_home": 9.5, "sp_k9_away": 8.0,
            "sp_k9_30g_home": 10.0, "sp_k9_30g_away": 7.5,
            "sp_fbvelo_3g_home": 95.0, "sp_fbvelo_3g_away": 92.0,
            "sp_fbpct_3g_home": 0.45, "sp_fbpct_3g_away": 0.40,
            "sp_whiff_3g_home": 0.30, "sp_whiff_3g_away": 0.24,
            "sp_xwoba_home": 0.300, "sp_xwoba_away": 0.320,
            "sp_xwoba_vs_l_home": 0.290, "sp_xwoba_vs_l_away": 0.330,
            "lineup_woba_mean_home": 0.330, "lineup_woba_mean_away": 0.310,
            "lineup_woba_top3_home": 0.360, "lineup_woba_top3_away": 0.340,
            "lineup_woba_std_home": 0.020, "lineup_woba_std_away": 0.030,
            "woba_30g_home": 0.335, "woba_30g_away": 0.305,
            "bullpen_whip_10g_home": 1.15, "bullpen_whip_10g_away": 1.35,
            "bullpen_whip_3g_home": 1.10, "bullpen_whip_3g_away": 1.40,
            "bullpen_pitches_3d_home": 120.0, "bullpen_pitches_3d_away": 90.0,
            "bullpen_ip_3d_home": 8.0, "bullpen_ip_3d_away": 6.0,
            "team_barrel_15g_home": 0.09, "team_barrel_15g_away": 0.07,
            "team_hardhit_15g_home": 0.38, "team_hardhit_15g_away": 0.34,
            "team_exitvelo_15g_home": 89.0, "team_exitvelo_15g_away": 87.5,
            "lineup_ops_vs_starter_hand_home": 0.740,
            "lineup_ops_vs_starter_hand_away": 0.700,
            "time_zones_crossed_last_3d_home": 1,
            "time_zones_crossed_last_3d_away": 0,
            "closer_available_home": 1.0, "closer_available_away": 1.0,
            "lineup_ops_vs_l_home": 0.700, "lineup_ops_vs_l_away": 0.680,
            "lineup_ops_vs_r_home": 0.780, "lineup_ops_vs_r_away": 0.740,
        },
        {
            "game_id": "g2", "home_team": "LAD", "away_team": "SEA",
            "home_win_pct": 0.52, "away_win_pct": 0.62,
            "home_elo": 1470.0, "away_elo": 1540.0,
            "rest_days_home": 2, "rest_days_away": 1,
            "sp_era_home": 4.0, "sp_era_away": 2.9,
            "sp_era_30g_home": 4.2, "sp_era_30g_away": 2.7,
            "sp_k9_home": 8.2, "sp_k9_away": 10.5,
            "sp_k9_30g_home": 8.0, "sp_k9_30g_away": 11.0,
            "sp_fbvelo_3g_home": 91.0, "sp_fbvelo_3g_away": 97.0,
            "sp_fbpct_3g_home": 0.42, "sp_fbpct_3g_away": 0.50,
            "sp_whiff_3g_home": 0.22, "sp_whiff_3g_away": 0.34,
            "sp_xwoba_home": 0.325, "sp_xwoba_away": 0.290,
            "sp_xwoba_vs_l_home": 0.335, "sp_xwoba_vs_l_away": 0.280,
            "lineup_woba_mean_home": 0.305, "lineup_woba_mean_away": 0.335,
            "lineup_woba_top3_home": 0.330, "lineup_woba_top3_away": 0.365,
            "lineup_woba_std_home": 0.035, "lineup_woba_std_away": 0.018,
            "woba_30g_home": 0.300, "woba_30g_away": 0.340,
            "bullpen_whip_10g_home": 1.40, "bullpen_whip_10g_away": 1.05,
            "bullpen_whip_3g_home": 1.45, "bullpen_whip_3g_away": 1.00,
            "bullpen_pitches_3d_home": 95.0, "bullpen_pitches_3d_away": 130.0,
            "bullpen_ip_3d_home": 6.5, "bullpen_ip_3d_away": 9.0,
            "team_barrel_15g_home": 0.06, "team_barrel_15g_away": 0.10,
            "team_hardhit_15g_home": 0.33, "team_hardhit_15g_away": 0.40,
            "team_exitvelo_15g_home": 87.0, "team_exitvelo_15g_away": 90.5,
            "lineup_ops_vs_starter_hand_home": 0.690,
            "lineup_ops_vs_starter_hand_away": 0.760,
            "time_zones_crossed_last_3d_home": 0,
            "time_zones_crossed_last_3d_away": 2,
            "closer_available_home": 0.0, "closer_available_away": 1.0,
            "lineup_ops_vs_l_home": 0.670, "lineup_ops_vs_l_away": 0.710,
            "lineup_ops_vs_r_home": 0.730, "lineup_ops_vs_r_away": 0.790,
        },
    ])


class TestMissingObservationsAreNull(unittest.TestCase):
    def test_missing_raw_column_yields_nan_not_zero(self):
        raw = _raw()
        raw = raw.drop(columns=["bullpen_ip_3d_home", "bullpen_ip_3d_away"])
        out = add_diff_features(raw)
        self.assertTrue(out["bullpen_ip_diff"].isna().all())

    def test_partial_missing_observation_yields_nan(self):
        raw = _raw()
        raw.loc[0, "sp_era_away"] = np.nan
        out = add_diff_features(raw)
        # g1's sp_era_diff is missing -> NaN; g2's is computed
        self.assertTrue(pd.isna(out.loc[0, "sp_era_diff"]))
        self.assertFalse(pd.isna(out.loc[1, "sp_era_diff"]))

    def test_win_pct_missing_is_nan(self):
        out = add_diff_features(pd.DataFrame({"home_team": ["NYY"], "away_team": ["BOS"]}))
        self.assertTrue(out["win_pct_diff"].isna().all())

    def test_unknown_dome_and_park_are_nan(self):
        raw = _raw()
        raw["home_team"] = "ZZZ"  # unknown team -> unknown dome/park
        out = add_diff_features(raw)
        self.assertTrue(out["dome_is_neutral"].isna().all())
        self.assertTrue(out["park_factor_slug_diff"].isna().all())


class TestWeatherFeatures(unittest.TestCase):
    def test_no_weather_data_means_nan_not_zero(self):
        out = add_diff_features(_raw())
        self.assertTrue(out["wind_advantage_flyball_factor"].isna().all())
        self.assertTrue(out["air_density_velocity_boost"].isna().all())

    def test_weather_populates_open_air_games(self):
        weather = {
            "g1": {"available": True, "wind_multiplier": 0.5, "air_density": 1.18},
            "g2": {"available": True, "wind_multiplier": -0.4, "air_density": 1.21},
        }
        out = add_diff_features(_raw(), weather_data=weather)
        # g1: 0.5 * sp_era_diff(3.2-4.1 = -0.9) = -0.45
        self.assertAlmostEqual(float(out.loc[0, "wind_advantage_flyball_factor"]), 0.5 * -0.9, places=3)
        # g2: -0.4 * (4.0-2.9 = 1.1) = -0.44
        self.assertAlmostEqual(float(out.loc[1, "wind_advantage_flyball_factor"]), -0.4 * 1.1, places=3)
        # air boost: (rho - 1.225) * sp_fbvelo_diff
        self.assertAlmostEqual(
            float(out.loc[0, "air_density_velocity_boost"]), (1.18 - 1.225) * (95.0 - 92.0), places=3)

    def test_missing_weather_game_stays_nan(self):
        weather = {"g1": {"available": True, "wind_multiplier": 0.5, "air_density": 1.18}}
        out = add_diff_features(_raw(), weather_data=weather)
        self.assertFalse(pd.isna(out.loc[0, "wind_advantage_flyball_factor"]))
        self.assertTrue(pd.isna(out.loc[1, "wind_advantage_flyball_factor"]))

    def test_unavailable_weather_stays_nan(self):
        weather = {"g1": {"available": False}}
        out = add_diff_features(_raw(), weather_data=weather)
        self.assertTrue(out["wind_advantage_flyball_factor"].isna().all())
        self.assertTrue(out["air_density_velocity_boost"].isna().all())

    def test_weather_missing_sp_diff_is_nan(self):
        raw = _raw()
        raw.loc[0, "sp_era_home"] = np.nan
        weather = {"g1": {"available": True, "wind_multiplier": 0.5, "air_density": 1.18}}
        out = add_diff_features(raw, weather_data=weather)
        self.assertTrue(pd.isna(out.loc[0, "wind_advantage_flyball_factor"]))

    def test_dome_game_gets_valid_zero(self):
        raw = _raw()
        raw["home_team"] = "TB"  # fixed dome (DOME_STATUS=1)
        out = add_diff_features(raw)  # no weather fetched
        self.assertEqual(float(out.loc[0, "wind_advantage_flyball_factor"]), 0.0)
        self.assertEqual(float(out.loc[0, "air_density_velocity_boost"]), 0.0)


class TestPitEnrichment(unittest.TestCase):
    """Enriching DuckDB frames: Elo + records derived PIT, never fabricated."""

    def _history(self) -> pd.DataFrame:
        # 3 chronological games: BOS beats NYY, NYY loses at home to BOS, NYY wins.
        return pd.DataFrame([
            {"game_date": "2026-04-01", "home_team": "BOS", "away_team": "NYY",
             "home_win": 1.0, "home_score": 5, "away_score": 3},
            {"game_date": "2026-04-02", "home_team": "NYY", "away_team": "BOS",
             "home_win": 0.0, "home_score": 2, "away_score": 4},
            {"game_date": "2026-04-03", "home_team": "NYY", "away_team": "TB",
             "home_win": 1.0, "home_score": 6, "away_score": 1},
        ])

    def test_elo_and_records_are_point_in_time(self):
        out = enrich_elo_and_records(self._history())
        # Game 1: neither team has history -> both enter at 1500
        self.assertEqual(float(out.loc[0, "home_elo"]), 1500.0)
        self.assertEqual(float(out.loc[0, "away_elo"]), 1500.0)
        # Game 2: BOS won game 1 (rating up), NYY lost it (rating down)
        self.assertGreater(float(out.loc[1, "away_elo"]), 1500.0)   # BOS entering g2
        self.assertLess(float(out.loc[1, "home_elo"]), 1500.0)      # NYY entering g2
        # Records entering each game exclude the game itself (PIT)
        self.assertEqual(int(out.loc[0, "home_wins"]), 0)
        self.assertEqual(int(out.loc[1, "away_wins"]), 1)   # BOS won g1
        self.assertEqual(int(out.loc[1, "home_wins"]), 0)    # NYY lost g1
        self.assertEqual(int(out.loc[2, "home_wins"]), 0)    # NYY 0-2 entering g3
        self.assertEqual(int(out.loc[2, "away_wins"]), 0)    # TB hasn't played yet

    def test_present_columns_are_untouched(self):
        h = self._history()
        h["home_elo"] = [1600.0] * 3
        h["away_elo"] = [1550.0] * 3
        h["home_wins"] = h["home_losses"] = h["away_wins"] = h["away_losses"] = 0
        out = enrich_elo_and_records(h)
        self.assertEqual(out["home_elo"].tolist(), [1600.0] * 3)
        self.assertEqual(out["home_wins"].tolist(), [0, 0, 0])

    def test_rename_team_woba(self):
        h = self._history()
        h["team_woba_30g_home"] = [0.30, 0.31, 0.32]
        h["team_woba_30g_away"] = [0.29, 0.28, 0.27]
        out = enrich_elo_and_records(h, rename_team_woba=True)
        self.assertIn("woba_30g_home", out.columns)
        self.assertNotIn("team_woba_30g_home", out.columns)

    def test_missing_chronology_stays_unchanged(self):
        out = enrich_elo_and_records(pd.DataFrame({"home_team": ["NYY"], "away_team": ["BOS"]}))
        self.assertNotIn("home_elo", out.columns)
        self.assertNotIn("home_wins", out.columns)


class TestLoadGameFeaturesHealing(unittest.TestCase):
    """load_game_features rebuilds diff features the DuckDB export NaN'd."""

    def test_heals_nan_diffs_from_duckdb_schema(self):
        import os
        import tempfile
        raw = pd.DataFrame([
            {"game_pk": 1, "game_date": "2026-04-01", "home_team": "BOS", "away_team": "NYY",
             "home_win": 1.0, "home_score": 5, "away_score": 3, "total_runs": 8,
             "team_woba_30g_home": 0.30, "team_woba_30g_away": 0.29,
             "win_pct_diff": np.nan, "elo_diff": np.nan, "woba_30g_diff": np.nan},
            {"game_pk": 2, "game_date": "2026-04-02", "home_team": "NYY", "away_team": "BOS",
             "home_win": 0.0, "home_score": 2, "away_score": 4, "total_runs": 6,
             "team_woba_30g_home": 0.31, "team_woba_30g_away": 0.28,
             "win_pct_diff": np.nan, "elo_diff": np.nan, "woba_30g_diff": np.nan},
        ])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            raw.to_csv(f, index=False)
            path = f.name
        try:
            out = load_game_features(path)
        finally:
            os.unlink(path)
        self.assertTrue(out["win_pct_diff"].notna().all())
        self.assertTrue(out["elo_diff"].notna().all())
        self.assertTrue(out["woba_30g_diff"].notna().all())
        # woba diff resolves through the team_woba_30g_* -> woba_30g_* mapping
        self.assertAlmostEqual(float(out.loc[0, "woba_30g_diff"]), 0.30 - 0.29, places=6)
        # g1 both teams 0-0: smoothed win pct is exactly .500 - .500 = 0
        self.assertAlmostEqual(float(out.loc[0, "win_pct_diff"]), 0.0, places=6)


class TestColumnAliases(unittest.TestCase):
    def test_renamed_column_is_sourced(self):
        raw = _raw()
        # rename to an alias from RAW_COLUMN_ALIASES (canonical name absent)
        raw = raw.rename(columns={"sp_xwoba_vs_l_home": "sp_xwoba_l_home"})
        out = add_diff_features(raw)
        self.assertAlmostEqual(float(out.loc[0, "sp_xwoba_vs_l_diff"]), 0.290 - 0.330, places=3)

    def test_alias_with_missing_value_is_nan(self):
        raw = _raw()
        raw["sp_xwoba_l_home"] = np.nan
        raw = raw.drop(columns=["sp_xwoba_vs_l_home"])
        out = add_diff_features(raw)
        self.assertTrue(out["sp_xwoba_vs_l_diff"].isna().all())

    def test_aliases_exist_for_known_renames(self):
        self.assertIn("bullpen_ip_3d_home", RAW_COLUMN_ALIASES)
        self.assertIn("lineup_ops_vs_l_home", RAW_COLUMN_ALIASES)
        self.assertIn("time_zones_crossed_last_3d_home", RAW_COLUMN_ALIASES)


if __name__ == "__main__":
    unittest.main()
