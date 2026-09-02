"""Regression tests for the MLB board's evening-game count
(_count_evening_games / _calibration_json evening_games_league).

The badge reads ``X evening games begin 7 PM ET+``. The pre-fix emitter
hardcoded the count to 0; this verifies the real count: ALL slate games
(any status) whose start is at/after 7 PM ET, computed by converting the
naive-UTC start_time_utc to America/New_York BEFORE comparing the hour —
the UTC-midnight rollover (a 9:38 PM ET game is 01:38 UTC next-day) is the
bug class the failed comparison dropped.
"""
import json
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

if "resource" not in sys.modules:  # POSIX-only module, stub on Windows
    _res = types.ModuleType("resource")
    _res.getrusage = lambda *_: types.SimpleNamespace(ru_maxrss=0)
    _res.RUSAGE_SELF = 0
    sys.modules["resource"] = _res

import pipeline  # noqa: E402

# 2026-09-01 is EDT (UTC-4): 19:00 ET == 23:00 UTC; 21:38 ET == 01:38 UTC
# the NEXT day. 2026-11-03 is EST (UTC-5): 19:30 ET == 00:30 UTC next day.
EDT = timezone(timedelta(hours=-4))
EST = timezone(timedelta(hours=-5))


def _utc_naive(dt: datetime) -> str:
    """Repo convention: start_time_utc is stored as a NAIVE UTC string."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class TestCountEveningGames(unittest.TestCase):
    def test_slate_fixture_boundary_rollover_and_statuses(self):
        """6:40 PM ET = day; 7:00 PM ET = evening (>= 19); 9:38 PM ET =
        evening despite its UTC repr being NEXT-DAY 01:38; status must not
        matter."""
        rows = [
            {"game_id": "g1", "start_time_utc": _utc_naive(
                datetime(2026, 9, 1, 18, 40, tzinfo=EDT)),
             "game_state": "post", "home_win": 0.0},
            {"game_id": "g2", "start_time_utc": _utc_naive(
                datetime(2026, 9, 1, 19, 0, tzinfo=EDT)),
             "game_state": "pre", "home_win": None},
            {"game_id": "g3", "start_time_utc": _utc_naive(
                datetime(2026, 9, 1, 21, 38, tzinfo=EDT)),
             "game_state": "in", "home_win": None},
            {"game_id": "g4", "start_time_utc": _utc_naive(
                datetime(2026, 9, 1, 16, 10, tzinfo=EDT)),
             "game_state": "post", "home_win": 1.0},
        ]
        games = pd.DataFrame(rows)
        # The rollover game's UTC repr is next-day 01:38 — a raw-UTC hour
        # comparison (hour 1) would drop it; the ET conversion keeps it.
        self.assertIn("2026-09-02", games.loc[2, "start_time_utc"])
        self.assertEqual(pipeline._count_evening_games(games), 2)

    def test_exactly_nineteen_is_evening_and_1810_is_not(self):
        games = pd.DataFrame({"start_time_utc": [
            _utc_naive(datetime(2026, 9, 1, 18, 10, tzinfo=EDT)),
            _utc_naive(datetime(2026, 9, 1, 19, 0, tzinfo=EDT)),
        ]})
        self.assertEqual(pipeline._count_evening_games(games), 1)

    def test_dst_transition_in_winter(self):
        """2026-11-03 is EST: 19:30 ET == 00:30 UTC next day — still evening."""
        games = pd.DataFrame({"start_time_utc": [
            _utc_naive(datetime(2026, 11, 3, 19, 30, tzinfo=EST)),
        ]})
        self.assertIn("2026-11-04", games.loc[0, "start_time_utc"])
        self.assertEqual(pipeline._count_evening_games(games), 1)

    def test_nan_and_missing_starts_are_skipped_not_zeroed(self):
        """A missing start drops only that game; valid rows still count.
        A fully-missing column is the only 0 case."""
        games = pd.DataFrame({"start_time_utc": [
            _utc_naive(datetime(2026, 9, 1, 20, 5, tzinfo=EDT)),
            None,
            _utc_naive(datetime(2026, 9, 1, 18, 40, tzinfo=EDT)),
        ]})
        self.assertEqual(pipeline._count_evening_games(games), 1)
        self.assertEqual(pipeline._count_evening_games(
            pd.DataFrame({"other": [1, 2]})), 0)
        self.assertEqual(pipeline._count_evening_games(None), 0)

    def test_real_slate_counts(self):
        """Hermetic regeneration of the committed slates' expected counts:
        09-01 = 9 evening of 15; 09-02 = 6 of 15 (both had all starts)."""
        for fname, expected_n, expected_evening in (
                ("todays_games_20260901.csv", 15, 9),
                ("todays_games_20260902.csv", 15, 6)):
            p = pipeline.DATA_DELIVERY_DIR / fname
            if not p.exists():
                continue
            g = pd.read_csv(p)
            self.assertEqual(len(g), expected_n)
            self.assertEqual(pipeline._count_evening_games(g), expected_evening)


class TestCalibrationJsonEveningWiring(unittest.TestCase):
    def _write(self, evening_games=None):
        with tempfile.TemporaryDirectory() as td:
            orig = pipeline.DATA_DELIVERY_DIR
            pipeline.DATA_DELIVERY_DIR = Path(td)
            try:
                path = pipeline._calibration_json(
                    {"auc": 0.5, "brier": 0.25, "logloss": 0.69, "ece": 0.0},
                    [], [], "20260901", 15, oof=None,
                    evening_games=evening_games,
                )
                return json.loads(Path(path).read_text())
            finally:
                pipeline.DATA_DELIVERY_DIR = orig

    def test_evening_count_is_emitted(self):
        data = self._write(evening_games=9)
        self.assertEqual(data["evening_games_league"], 9)

    def test_default_keeps_zero_for_direct_calls(self):
        # Backward compatibility: direct callers that don't thread the
        # count (existing tests, other tools) still get a valid artifact.
        data = self._write(evening_games=None)
        self.assertIn("evening_games_league", data)
        self.assertEqual(data["evening_games_league"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)