"""Unit tests for the nflverse schedule loader (network-free)."""

import unittest

import pandas as pd

from nfl_nflverse_schedule import select_season_rows


def _games_df():
    return pd.DataFrame([
        {"game_id": "2026_01_NE_SEA", "season": 2026, "week": 1,
         "gameday": "2026-09-09", "home_team": "SEA", "away_team": "NE",
         "home_score": None, "away_score": None, "roof": "outdoors",
         "div_game": 0},
        {"game_id": "2026_02_KC_CIN", "season": 2026, "week": 2,
         "gameday": "2026-09-20", "home_team": "KC", "away_team": "CIN",
         "home_score": None, "away_score": None, "roof": "outdoors",
         "div_game": 0},
        {"game_id": "2025_18_BUF_NYJ", "season": 2025, "week": 18,
         "gameday": "2026-01-04", "home_team": "BUF", "away_team": "NYJ",
         "home_score": 19, "away_score": 12, "roof": "outdoors", "div_game": 1},
    ])


class TestSelectSeasonRows(unittest.TestCase):
    def test_scheduled_rows_have_nan_scores(self):
        out = select_season_rows(_games_df(), 2026)
        self.assertEqual(len(out), 2)
        self.assertTrue(pd.isna(out["home_score"]).all())
        self.assertTrue(pd.isna(out["away_score"]).all())

    def test_decided_season_scores_stay_numeric(self):
        out = select_season_rows(_games_df(), 2025)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["home_score"], 19)

    def test_empty_when_season_absent(self):
        out = select_season_rows(_games_df(), 1998)
        self.assertTrue(out.empty)


if __name__ == "__main__":
    unittest.main()