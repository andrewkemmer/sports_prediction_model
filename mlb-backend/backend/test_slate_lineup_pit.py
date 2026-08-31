"""Strict point-in-time guard for the slate lineup-delta fallback.

Regression: a slate game whose StatsAPI battingOrder is not yet posted
(projected-only morning) must emit NULL for all six lineup_actual_* columns —
NEVER a fabricated 0.0. A fake "projected lineup equals the team season mean"
zero would leak a value into the model that cannot exist at bet time,
violating the same strict point-in-time discipline the market-line as-of join
(data_ingestion._attach_market_lines rejects lines posted at/after start) and
the weather/roof missing-observation handling enforce.
"""
import unittest
from datetime import date
from unittest import mock

import pandas as pd

from backend import pipeline

LINEUP_DELTA_COLS = [
    "lineup_actual_woba_delta_home", "lineup_actual_woba_delta_away",
    "lineup_actual_top3_delta_home", "lineup_actual_top3_delta_away",
    "lineup_rest_count_home", "lineup_rest_count_away",
]


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _fake_get(url, *args, **kwargs):
    if "feed/live" in url:
        # Game exists on the schedule, but the battingOrder is NOT YET POSTED
        # (projected-only morning) → both sides empty.
        teams_bs = {
            "home": {"battingOrder": []},
            "away": {"battingOrder": []},
        }
        return _Resp({"liveData": {"boxscore": {"teams": teams_bs}}})
    # StatsAPI schedule: one NYY/BOS game at 13:05 ET.
    return _Resp({
        "dates": [{
            "games": [{
                "gamePk": 900001,
                "gameDate": "2026-08-29T17:05:00Z",
                "teams": {
                    "away": {"team": {"abbreviation": "BOS"}},
                    "home": {"team": {"abbreviation": "NYY"}},
                },
            }],
        }],
    })


class TestSlateLineupPitNull(unittest.TestCase):
    @mock.patch("requests.get", side_effect=_fake_get)
    def test_not_posted_lineup_emits_null_not_zero(self, _mock_get):
        slate = pd.DataFrame([{
            "home_team": "NYY",
            "away_team": "BOS",
            "start_time_utc": pd.Timestamp("2026-08-29 13:05:00"),
        }])
        out = pipeline._fetch_slate_lineups(slate, date(2026, 8, 29))
        for c in LINEUP_DELTA_COLS:
            self.assertIn(c, out.columns)
            self.assertTrue(out[c].isna().all(),
                            f"{c} must be NULL when lineup not posted")


if __name__ == "__main__":
    unittest.main()