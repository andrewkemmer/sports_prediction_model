"""
Tests for weather.py — Open-Meteo fetching, air density, wind multiplier.
"""
import unittest
from datetime import datetime, date
from unittest.mock import patch, MagicMock

import numpy as np

from weather import (
    _get_with_retry,
    STADIUMS,
    compute_air_density,
    compute_wind_multiplier,
    fetch_weather,
    fetch_game_weather,
    fetch_day_weather,
    _resolve_team_code,
    _nearest_hour,
    _parse_batch_response,
    fetch_games_weather,
)


class TestAirDensity(unittest.TestCase):
    def test_standard_sea_level(self):
        """Standard atmosphere: 15°C, 0% RH, 1013.25 hPa, sea level → ≈1.225 kg/m³."""
        rho = compute_air_density(15.0, 0.0, 1013.25, 0.0)
        self.assertAlmostEqual(rho, 1.225, delta=0.015)

    def test_standard_sea_level_humid(self):
        """15°C, 100% RH should be slightly less dense (water vapor is lighter)."""
        rho_dry = compute_air_density(15.0, 0.0, 1013.25, 0.0)
        rho_humid = compute_air_density(15.0, 100.0, 1013.25, 0.0)
        self.assertLess(rho_humid, rho_dry)

    def test_high_altitude_thinner(self):
        """Denver (1610m) with ~840 hPa station pressure should be less dense."""
        rho_sea = compute_air_density(15.0, 50.0, 1013.25, 0.0)
        # Denver station pressure ≈ 840 hPa
        rho_denv = compute_air_density(15.0, 50.0, 840.0, 1610.0)
        self.assertLess(rho_denv, rho_sea)
        self.assertAlmostEqual(rho_denv, 1.01, delta=0.03)

    def test_hot_day_thinner(self):
        """40°C at sea level should be less dense than 0°C."""
        rho_cold = compute_air_density(0.0, 50.0, 1013.25, 0.0)
        rho_hot = compute_air_density(40.0, 50.0, 1013.25, 0.0)
        self.assertLess(rho_hot, rho_cold)

    def test_nan_inputs(self):
        self.assertTrue(np.isnan(compute_air_density(np.nan, 50.0, 1013.25, 0.0)))
        # Pressure may be derived from a KNOWN altitude (a real calculation), so
        # missing pressure + known altitude is valid, not NaN.
        self.assertFalse(np.isnan(compute_air_density(15.0, 50.0, np.nan, 0.0)))
        # Missing pressure AND unknown altitude -> null
        self.assertTrue(np.isnan(compute_air_density(15.0, 50.0, np.nan, np.nan)))

    def test_missing_humidity_is_null_not_assumed(self):
        """A missing RH observation must NOT be silently assumed 50%."""
        self.assertTrue(np.isnan(compute_air_density(15.0, np.nan, 1013.25, 0.0)))

    def test_pressure_derived_from_altitude_is_valid(self):
        """Missing pressure at a KNOWN altitude is a real calculation, not a
        fabrication — altitude is a stadium constant."""
        rho = compute_air_density(15.0, 50.0, np.nan, 0.0)
        self.assertFalse(np.isnan(rho))


class TestWindMultiplier(unittest.TestCase):
    def test_blowing_out(self):
        """Wind from home plate direction toward center field = OUT (positive)."""
        mult = compute_wind_multiplier(0.0, 20.0, 0.0)
        self.assertGreater(mult, 0.0)  # positive = blowing out
        self.assertAlmostEqual(mult, 0.5, places=1)  # speed factor 20/40=0.5

    def test_blowing_out_strong(self):
        """Strong wind fully out → near +1.0."""
        mult = compute_wind_multiplier(0.0, 40.0, 0.0)
        self.assertAlmostEqual(mult, 1.0, places=1)

    def test_blowing_in(self):
        """Wind from center field toward home plate = IN (negative)."""
        mult = compute_wind_multiplier(180.0, 20.0, 0.0)
        self.assertLess(mult, 0.0)  # negative = blowing in
        self.assertAlmostEqual(mult, -0.5, places=1)

    def test_cross_wind(self):
        """Perpendicular wind ≈ 0."""
        mult = compute_wind_multiplier(90.0, 20.0, 0.0)
        self.assertAlmostEqual(mult, 0.0, delta=0.1)

    def test_calm_wind(self):
        """Wind < 3 km/h = no effect."""
        mult = compute_wind_multiplier(0.0, 2.0, 0.0)
        self.assertEqual(mult, 0.0)

    def test_speed_scaling(self):
        """Stronger wind = larger magnitude (capped at 40 km/h)."""
        weak = abs(compute_wind_multiplier(0.0, 10.0, 0.0))
        strong = abs(compute_wind_multiplier(0.0, 30.0, 0.0))
        self.assertGreater(strong, weak)

    def test_speed_cap(self):
        """Wind > 40 km/h is capped (no saturation beyond 1.0)."""
        at_cap = abs(compute_wind_multiplier(0.0, 40.0, 0.0))
        over_cap = abs(compute_wind_multiplier(0.0, 80.0, 0.0))
        self.assertAlmostEqual(at_cap, over_cap, places=2)

    def test_nan_inputs(self):
        """Missing wind observations are NULL — never a fabricated 0."""
        self.assertTrue(np.isnan(compute_wind_multiplier(np.nan, 20.0, 0.0)))
        self.assertTrue(np.isnan(compute_wind_multiplier(0.0, np.nan, 0.0)))


class TestNearestHour(unittest.TestCase):
    def test_exact_match(self):
        times = ["2026-08-22T19:00", "2026-08-22T20:00", "2026-08-22T21:00"]
        target = datetime(2026, 8, 22, 19, 0)
        self.assertEqual(_nearest_hour(times, target), 0)

    def test_between_hours(self):
        times = ["2026-08-22T19:00", "2026-08-22T20:00", "2026-08-22T21:00"]
        target = datetime(2026, 8, 22, 19, 20)
        self.assertEqual(_nearest_hour(times, target), 0)
        target2 = datetime(2026, 8, 22, 19, 40)
        self.assertEqual(_nearest_hour(times, target2), 1)


class TestResolveTeamCode(unittest.TestCase):
    def test_known_team(self):
        self.assertEqual(_resolve_team_code("NYY"), "NYY")

    def test_statcast_alias(self):
        self.assertEqual(_resolve_team_code("AZ"), "ARI")
        self.assertEqual(_resolve_team_code("OAK"), "ATH")

    def test_venue_lookup(self):
        self.assertEqual(_resolve_team_code("XXX", "Yankee Stadium"), "NYY")
        self.assertEqual(_resolve_team_code("XXX", "Fenway Park"), "BOS")
        self.assertEqual(_resolve_team_code("XXX", "Coors Field"), "COL")

    def test_unknown_falls_through(self):
        self.assertEqual(_resolve_team_code("ZZZ"), "ZZZ")


class TestFetchWeather(unittest.TestCase):
    @patch("weather.requests.get")
    def test_archive_api_called_for_past_dates(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "hourly": {
                "time": ["2025-07-15T19:00", "2025-07-15T20:00"],
                "temperature_2m": [25.0, 24.0],
                "relative_humidity_2m": [60, 65],
                "wind_speed_10m": [15.0, 12.0],
                "wind_direction_10m": [180.0, 190.0],
                "surface_pressure": [1013.0, 1012.0],
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_weather(40.83, -73.93, datetime(2025, 7, 15, 19, 30), date(2025, 7, 15))
        self.assertAlmostEqual(result["temp_c"], 25.0)
        self.assertEqual(result["wind_speed_kmh"], 15.0)
        self.assertEqual(result["pressure_hpa"], 1013.0)

        # Should use archive API
        call_url = mock_get.call_args[0][0]
        self.assertIn("archive-api", call_url)

    @patch("weather.requests.get")
    def test_strictly_prior_hour_selected(self, mock_get):
        """PIT: weather used must be observed strictly before first pitch —
        the hour in which the game starts is never used."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "hourly": {
                "time": ["2025-07-15T18:00", "2025-07-15T19:00", "2025-07-15T20:00"],
                "temperature_2m": [23.0, 25.0, 26.0],
                "relative_humidity_2m": [60, 62, 64],
                "wind_speed_10m": [14.0, 15.0, 16.0],
                "wind_direction_10m": [180.0, 185.0, 190.0],
                "surface_pressure": [1013.0, 1012.0, 1011.0],
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # Game starts 19:05 — the 19:00 hourly timestamp is strictly prior to
        # 19:05, so it is the latest usable row (mirrors the 1-second-shift
        # convention used for market lines).
        result = fetch_weather(40.83, -73.93, datetime(2025, 7, 15, 19, 5), date(2025, 7, 15))
        self.assertAlmostEqual(result["temp_c"], 25.0)

        # Game starts exactly 19:00 — 19:00 is NOT strictly prior → 18:00.
        result = fetch_weather(40.83, -73.93, datetime(2025, 7, 15, 19, 0), date(2025, 7, 15))
        self.assertAlmostEqual(result["temp_c"], 23.0)

    @patch("weather.requests.get")
    def test_forecast_api_for_future(self, mock_get):
        mock_resp = MagicMock()
        future = date(2026, 8, 25)
        mock_resp.json.return_value = {
            "hourly": {
                "time": [f"2026-08-25T{h:02d}:00" for h in range(24)],
                "temperature_2m": [20.0] * 24,
                "relative_humidity_2m": [50] * 24,
                "wind_speed_10m": [10.0] * 24,
                "wind_direction_10m": [90.0] * 24,
                "surface_pressure": [1010.0] * 24,
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_weather(40.83, -73.93, datetime(2026, 8, 25, 19, 0), future)
        self.assertEqual(result["temp_c"], 20.0)

        call_url = mock_get.call_args[0][0]
        self.assertIn("api.open-meteo.com/v1/forecast", call_url)

    @patch("weather.requests.get")
    def test_fetch_failure_returns_nan(self, mock_get):
        mock_get.side_effect = Exception("connection error")
        result = fetch_weather(40.83, -73.93, datetime(2025, 7, 15, 19, 0), date(2025, 7, 15))
        self.assertTrue(np.isnan(result["temp_c"]))
        self.assertTrue(np.isnan(result["wind_speed_kmh"]))

    @patch("weather.requests.get")
    def test_game_fetch_failure_is_unavailable(self, mock_get):
        mock_get.side_effect = Exception("rate limited")
        result = fetch_game_weather(
            "NYY", "Yankee Stadium", datetime(2025, 7, 15, 23, 0)
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["source"], "open_meteo_unavailable")


class TestFetchGameWeather(unittest.TestCase):
    @patch("weather.fetch_weather")
    def test_full_pipeline(self, mock_fetch):
        mock_fetch.return_value = {
            "temp_c": 22.0, "rh_pct": 55.0,
            "wind_speed_kmh": 18.0, "wind_direction_deg": 350.0,
            "pressure_hpa": 1012.0,
        }
        result = fetch_game_weather(
            "NYY", "Yankee Stadium",
            datetime(2025, 7, 15, 23, 0),  # 7pm ET in UTC
        )
        self.assertTrue(result["available"])
        self.assertIn("air_density", result)
        self.assertIn("wind_multiplier", result)
        self.assertFalse(np.isnan(result["air_density"]))

    def test_unknown_stadium(self):
        result = fetch_game_weather("ZZZ", "Unknown Park", datetime(2025, 7, 15, 23, 0))
        self.assertFalse(result["available"])


class TestStadiums(unittest.TestCase):
    def test_all_30_teams(self):
        """We have entries for all 30 MLB teams."""
        self.assertGreaterEqual(len(STADIUMS), 30)

    def test_dome_bearing_zero(self):
        """Dome stadiums have bearing=0 (wind handled by dome_is_neutral flag)."""
        for team in ("TB", "HOU", "SEA", "TEX", "TOR", "MIL", "MIA", "ARI"):
            self.assertEqual(STADIUMS[team]["bearing"], 0, f"{team} should have bearing=0")


class TestBatchWeather(unittest.TestCase):
    def test_parse_multi_location_response_preserves_source_and_days(self):
        locations = [("BOS", STADIUMS["BOS"]), ("NYY", STADIUMS["NYY"])]
        hourly = {
            "time": ["2025-07-15T18:00", "2025-07-15T19:00", "2025-07-16T00:00"],
            "temperature_2m": [20.0, 21.0, 22.0],
            "relative_humidity_2m": [50.0, 51.0, 52.0],
            "wind_speed_10m": [10.0, 11.0, 12.0],
            "wind_direction_10m": [90.0, 90.0, 90.0],
            "surface_pressure": [1010.0, 1010.0, 1010.0],
        }
        parsed = _parse_batch_response([{"hourly": hourly}, {"hourly": hourly}], locations, "open_meteo_archive")
        self.assertEqual(sorted(parsed), [
            ("BOS", date(2025, 7, 15)), ("BOS", date(2025, 7, 16)),
            ("NYY", date(2025, 7, 15)), ("NYY", date(2025, 7, 16)),
        ])
        self.assertEqual(parsed[("BOS", date(2025, 7, 15))]["_source"], "open_meteo_archive")
        self.assertEqual(parsed[("BOS", date(2025, 7, 15))]["temperature_2m"], [20.0, 21.0])

    @patch("weather._fetch_batch_range")
    def test_rate_limited_batch_returns_no_observations(self, mock_batch):
        mock_batch.return_value = {}
        from weather import _fetch_batched_weather
        locations = [("BOS", STADIUMS["BOS"])]
        result = _fetch_batched_weather(locations, date(2025, 7, 15), date(2025, 7, 15))
        self.assertEqual(result, {})
        self.assertEqual(mock_batch.call_count, 1)


class TestGetWithRetry(unittest.TestCase):
    """429/5xx responses must retry with backoff, not degrade to climatology."""

    def _resp(self, status):
        m = MagicMock()
        m.status_code = status
        return m

    @patch("time.sleep", lambda *_: None)
    @patch("weather.requests.get")
    def test_recovers_after_429s(self, mock_get):
        mock_get.side_effect = [self._resp(429), self._resp(429),
                                self._resp(200)]
        resp = _get_with_retry("https://x/archive", {}, attempts=3)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_get.call_count, 3)

    @patch("time.sleep", lambda *_: None)
    @patch("weather.requests.get")
    def test_gives_up_after_attempts(self, mock_get):
        mock_get.side_effect = [self._resp(429)] * 5
        resp = _get_with_retry("https://x/archive", {}, attempts=3)
        self.assertEqual(resp.status_code, 429)
        self.assertEqual(mock_get.call_count, 3)

    @patch("time.sleep", lambda *_: None)
    @patch("weather.requests.get")
    def test_non_retryable_status_returned_immediately(self, mock_get):
        mock_get.side_effect = [self._resp(400)]
        resp = _get_with_retry("https://x/archive", {}, attempts=3)
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(mock_get.call_count, 1)

    @patch("time.sleep", lambda *_: None)
    @patch("weather.requests.get")
    def test_network_exception_retried(self, mock_get):
        import requests as _rq
        mock_get.side_effect = [_rq.exceptions.ConnectionError(), self._resp(200)]
        resp = _get_with_retry("https://x/archive", {}, attempts=3)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
