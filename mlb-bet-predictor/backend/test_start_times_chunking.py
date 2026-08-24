"""Regression tests for StatsAPI start-time retrieval.

The schedule endpoint SILENTLY TRUNCATES long date ranges: one request for
2025-01-01→2026-08-23 returned only 2025-02-20→2025-11-01. Every game after
the cutoff got no first-pitch timestamp, was skipped before any weather
fetch, and the wind/air-density features went null for the rest of the
season while all pipeline logs looked healthy ("2470/2477 fetched" — of
only the games that had timestamps).

Locks in:
- fetch_game_start_times chunks long ranges and merges results completely
- per-chunk truncation cannot lose games (each chunk re-anchors its window)
- a failing chunk doesn't destroy the other chunks' data
- total failure returns {} (existing caller contract)
- _attach_weather_history warns loudly when start-time coverage is poor
"""
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import requests as _requests

from results import SCHEDULE_CHUNK_DAYS, fetch_game_start_times


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _pk(day: date, i: int) -> int:
    return int(day.strftime("%Y%m%d")) * 100 + i


def _payload_for(start: date, end: date, per_day: int = 2) -> dict:
    """Honest schedule payload covering exactly [start, end]."""
    dates = []
    day = start
    while day <= end:
        games = [{"gamePk": _pk(day, i),
                  "gameDate": f"{day.isoformat()}T19:{10 + i * 5}:00Z"}
                 for i in range(per_day)]
        dates.append({"date": day.isoformat(), "games": games})
        day += timedelta(days=1)
    return {"dates": dates}


def _expected_pks(start: date, end: date, per_day: int = 2) -> set[int]:
    out = set()
    day = start
    while day <= end:
        out |= {_pk(day, i) for i in range(per_day)}
        day += timedelta(days=1)
    return out


def _fake_get_honest(requested_windows: list[tuple[str, str]]):
    """requests.get stand-in that serves exactly each requested window."""
    def fake(url, params=None, timeout=None, **kw):
        s = date.fromisoformat(params["startDate"])
        e = date.fromisoformat(params["endDate"])
        requested_windows.append((params["startDate"], params["endDate"]))
        return _Resp(_payload_for(s, e))
    return fake


def _fake_get_capped(cap_days: int):
    """Endpoint that serves AT MOST ``cap_days`` from startDate of any ask.

    This models the real observed behavior: one 20-month request returned
    only ~243 days (2025-02-20→2025-11-01), silently dropping the rest.
    Chunked requests stay far below the cap and are therefore honored.
    """
    def fake(url, params=None, timeout=None, **kw):
        s = date.fromisoformat(params["startDate"])
        e = date.fromisoformat(params["endDate"])
        return _Resp(_payload_for(s, min(e, s + timedelta(days=cap_days - 1))))
    return fake



class TestChunkedStartTimes(unittest.TestCase):

    def test_long_range_is_chunked_and_merged(self):
        # 150-day span must produce >= 3 bounded requests, all merged.
        start, end = date(2026, 4, 1), date(2026, 8, 28)
        windows: list[tuple[str, str]] = []
        with patch("results.requests.get",
                   side_effect=_fake_get_honest(windows)):
            out = fetch_game_start_times(start, end)
        self.assertGreaterEqual(len(windows), 3)
        for (_, w_end), (_next_start, _) in zip(windows, windows[1:]):
            # Chunks advance without gaps or overlaps.
            self.assertEqual(
                date.fromisoformat(w_end) + timedelta(days=1),
                date.fromisoformat(_next_start))
        self.assertEqual(set(out), _expected_pks(start, end))

    def test_truncating_endpoint_cannot_starve_late_season(self):
        """THE regression: a response-length-capped endpoint (real-world
        behavior) drops everything past the cap in ONE request; chunked
        asks never reach the cap, so all games are recovered."""
        start, end = date(2025, 11, 15), date(2026, 4, 30)
        with patch("results.requests.get", side_effect=_fake_get_capped(100)):
            out = fetch_game_start_times(start, end)
        self.assertEqual(set(out), _expected_pks(start, end))
        self.assertIn(_pk(date(2026, 4, 15), 0), out,
                      "late-season games lost despite chunking")
        # Prove the fixture bites: ONE capped request stops before April —
        # exactly how the production run lost all of 2026.
        one = _fake_get_capped(100)(None, params={
            "startDate": start.isoformat(), "endDate": end.isoformat()})
        single_pks = {g["gamePk"] for d in one.json()["dates"]
                      for g in d["games"]}
        self.assertNotIn(_pk(date(2026, 4, 15), 0), single_pks)

    def test_chunk_failure_keeps_other_chunks(self):
        start, end = date(2026, 6, 1), date(2026, 8, 29)  # 2 chunks at 60d
        calls = {"n": 0}

        def flaky(url, params=None, timeout=None, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise _requests.exceptions.ConnectionError("simulated outage")
            s = date.fromisoformat(params["startDate"])
            e = date.fromisoformat(params["endDate"])
            return _Resp(_payload_for(s, e))

        with patch("results.requests.get", side_effect=flaky):
            out = fetch_game_start_times(start, end)
        self.assertEqual(calls["n"], 2)
        self.assertIn(_pk(date(2026, 6, 15), 0), out)   # chunk 1 survived...
        self.assertNotIn(_pk(date(2026, 8, 15), 0), out)  # ...chunk 2 didn't poison it

    def test_total_failure_returns_empty_dict(self):
        with patch("results.requests.get",
                   side_effect=_requests.exceptions.ConnectionError("down")):
            out = fetch_game_start_times(date(2026, 6, 1), date(2026, 8, 29))
        self.assertEqual(out, {})

    def test_chunk_size_constant_is_under_truncation_cutoff(self):
        # Observed cutoff ≈ 243 returned days for a 20-month ask; stay far under.
        self.assertLessEqual(SCHEDULE_CHUNK_DAYS, 90)


class TestWeatherHistoryCoverageGate(unittest.TestCase):
    """_attach_weather_history must announce poor start-time coverage."""

    def test_warns_when_start_times_sparse(self):
        import pandas as pd
        import pipeline

        games = pd.DataFrame([
            {"game_pk": 101, "game_id": "g1", "game_date": "2026-08-01",
             "home_team": "BOS", "venue": "Fenway Park", "home_win": 1.0},
            {"game_pk": 102, "game_id": "g2", "game_date": "2026-08-02",
             "home_team": "MIL", "venue": "American Family Field", "home_win": 0.0},
        ])
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with \
            patch.object(pipeline, "_weather_cache_path",
                         return_value=Path(tmp.name) / "wx.parquet"), \
            patch.object(pipeline, "STATSAPI_WEATHER_FILL", False), \
            patch("results.fetch_game_start_times", return_value={102: "2026-08-02T23:10:00Z"}), \
            patch("weather.fetch_games_weather", return_value={}), \
            self.assertLogs("pipeline", level="WARNING") as logs:
            pipeline._attach_weather_history(games, date(2026, 8, 2))
        joined = "\n".join(logs.output)
        self.assertIn("matched only 1/2 decided games", joined)


if __name__ == "__main__":
    unittest.main()
