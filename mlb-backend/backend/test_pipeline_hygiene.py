"""Regression tests for the v2026.08.24 pipeline-hygiene fixes.

Covers:
- Issue 2: future-dated empty Statcast chunks are EXPECTED (DEBUG, not WARNING)
- Issue 3: Open-Meteo batches skip windows containing no scheduled games
- Issue 4: StatsAPI game-feed weather parses + converts under honest-fill rules
- Issue 5: win_pct_diff warns loudly ONLY on the final diff computation;
  pre-overlay absence is a DEBUG note, and records-present frames ship no NaNs

Imports use the TOP-LEVEL module names exactly as production code does
(``from results import ...``, ``import ingestion``) so patches hit the same
module instances the code under test reads.
"""
import re
import unittest
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd

import ingestion
import results as results_mod
from features import add_diff_features
from weather import (
    STADIUMS,
    compute_wind_multiplier,
    statsapi_weather_to_record,
    _fetch_batched_weather,
)


class TestFutureChunkGuard(unittest.TestCase):
    """Empty Statcast chunks entirely at/after today are expected."""

    def test_future_chunk_is_debug_not_warning(self):
        today = date(2026, 8, 24)
        with patch.object(ingestion, "date", _FrozenDate(today)):
            with self.assertNoLogs(ingestion.logger, level="WARNING"):
                with self.assertLogs(ingestion.logger, level="DEBUG") as captured:
                    ingestion._warn_if_core_season_chunk_empty(
                        today, today + timedelta(days=5), "empty response")
        self.assertIn("future-dated", " ".join(captured.output))

    def test_past_core_season_chunk_still_warns(self):
        past = date(2026, 7, 1)
        with patch.object(ingestion, "date", _FrozenDate(date(2026, 8, 24))):
            with self.assertLogs(ingestion.logger, level="WARNING") as captured:
                ingestion._warn_if_core_season_chunk_empty(
                    past, past + timedelta(days=3), "empty response")
        self.assertIn("EMPTY", " ".join(captured.output))


class _FrozenDate:
    """Stand-in for the module-level ``date`` name pinning today()."""

    def __init__(self, today: date):
        self._today = today

    def today(self):  # noqa: N802 - matches datetime.date API
        return self._today


class TestWeatherWindowSkipping(unittest.TestCase):
    """Open-Meteo batches must skip windows containing no scheduled games."""

    def test_gameless_offseason_windows_are_skipped(self):
        requested = []

        def fake_batch(locations, chunk_start, chunk_end, source="", **kw):
            requested.append((chunk_start, chunk_end))
            return {}

        needed = {date(2026, 3, 28), date(2026, 4, 15)}
        with patch("time.sleep"), \
             patch("weather._fetch_batch_range", side_effect=fake_batch):
            out = _fetch_batched_weather(
                [("NYY", STADIUMS["NYY"])],
                date(2025, 11, 1), date(2026, 4, 30),
                needed_days=needed,
            )
        self.assertEqual(out, {})
        self.assertTrue(requested, "in-season windows must still be fetched")
        for start, end in requested:
            overlaps_needed = any(start <= d <= end for d in needed)
            self.assertTrue(
                overlaps_needed,
                f"window {start}→{end} contains no game dates but was fetched")

    def test_none_needed_days_preserves_fetch_everything(self):
        requested = []

        def fake_batch(locations, chunk_start, chunk_end, source="", **kw):
            requested.append((chunk_start, chunk_end))
            return {}

        with patch("time.sleep"), \
             patch("weather._fetch_batch_range", side_effect=fake_batch):
            _fetch_batched_weather(
                [("NYY", STADIUMS["NYY"])],
                date(2025, 12, 1), date(2025, 12, 20),
                needed_days=None,
            )
        # Old behavior: every window in range gets a request.
        spans = [(s, e) for s, e in requested]
        self.assertTrue(any(s <= date(2025, 12, 10) <= e for s, e in spans),
                        "December window must be fetched when no game knowledge")


class TestStatsapiWeatherFiller(unittest.TestCase):
    """gameData.weather parses and converts under the honest-fill rules."""

    def test_feed_parsing(self):
        class Resp:
            status_code = 200

            def json(self):
                return {"gameData": {"weather": {
                    "condition": "Clear", "temp": 70, "wind": "8 mph, In from LF"}}}

        with patch.object(results_mod.requests, "get", return_value=Resp()), \
             patch("time.sleep"):
            out = results_mod.fetch_statsapi_weather([12345])
        self.assertEqual(out[12345]["wind_mph"], 8.0)
        self.assertEqual(out[12345]["temp_f"], 70.0)

    def test_record_conversion_out_wind_positive(self):
        rec = statsapi_weather_to_record(
            {"temp_f": 72.0, "wind_mph": 9.0, "wind_text": "9 mph, Out to CF"},
            home_team="NYY", venue="Yankee Stadium")
        bearing = STADIUMS["NYY"]["bearing"]
        expected = compute_wind_multiplier(bearing, 9.0 * 1.60934, bearing)
        self.assertTrue(rec["available"])
        self.assertAlmostEqual(rec["wind_multiplier"], expected, places=6)
        self.assertGreater(rec["wind_multiplier"], 0.0)   # tailwind
        self.assertAlmostEqual(rec["temp_c"], 22.222, places=2)
        # Honest nulls: feed carries neither RH nor pressure → density NULL.
        self.assertIsNone(rec["air_density"])
        self.assertIsNone(rec["rh_pct"])

    def test_record_conversion_in_wind_negative(self):
        rec = statsapi_weather_to_record(
            {"temp_f": 70.0, "wind_mph": 8.0, "wind_text": "8 mph, In from LF"},
            home_team="NYY", venue="Yankee Stadium")
        self.assertTrue(rec["available"])
        self.assertLess(rec["wind_multiplier"], 0.0)      # headwind

    def test_unusable_observation_marks_unavailable(self):
        rec = statsapi_weather_to_record(
            {"temp_f": None, "wind_mph": None, "wind_text": "calm"},
            home_team="NYY", venue="Yankee Stadium")
        self.assertFalse(rec["available"])
        self.assertIn("unusable", rec["source"])


def _base_frame(**extra) -> pd.DataFrame:
    n = 4
    base = {
        "game_pk": list(range(1, n + 1)),
        "game_date": pd.date_range("2026-08-20", periods=n).strftime("%Y-%m-%d"),
        "home_team": ["NYY"] * n,
        "away_team": ["BOS"] * n,
        "home_elo": [1500.0] * n,
        "away_elo": [1480.0] * n,
    }
    base.update(extra)
    return pd.DataFrame(base)


class TestWinPctDiffStageAwareness(unittest.TestCase):
    """Loud warning ONLY on the final computation; pre-overlay is DEBUG."""

    RECORDS = dict(home_wins=[50] * 4, home_losses=[40] * 4,
                   away_wins=[45] * 4, away_losses=[45] * 4)

    def test_records_present_no_warning_no_nans(self):
        df = _base_frame(**self.RECORDS)
        with self.assertNoLogs(level="WARNING"):
            out = add_diff_features(df, require_records=True)
        self.assertTrue(np.isfinite(out["win_pct_diff"]).all(),
                        "win_pct_diff must never be NaN when records exist")
        # Smoothed rates: equal-ish records → small diff, not garbage.

    def test_pre_overlay_absence_is_debug_only(self):
        df = _base_frame()
        with self.assertLogs("features", level="DEBUG") as captured:
            out = add_diff_features(df, require_records=False)
        joined = " ".join(captured.output)
        self.assertNotIn(" WARNING ", joined)
        self.assertIn("not present yet", joined)
        self.assertTrue(pd.isna(out["win_pct_diff"]).all())

    def test_final_computation_missing_records_warns(self):
        df = _base_frame()
        with self.assertLogs("features", level="WARNING") as captured:
            out = add_diff_features(df, require_records=True)
        self.assertIn("FINAL computation", " ".join(captured.output))
        self.assertTrue(pd.isna(out["win_pct_diff"]).all())


if __name__ == "__main__":
    unittest.main()
