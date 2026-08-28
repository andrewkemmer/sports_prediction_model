"""
Regression tests for today's-games artifact completeness.

Two dashboard-visible bugs locked out here:
1. _today_games_csv dropped home_score/away_score, so finished games showed
   no final score on the scoreboard.
2. ESPN removes probablePitcher from the scoreboard once a game starts, so
   an evening rerun rebuilt the slate with sp_name_* = 'TBD' and erased the
   pitching matchup published that morning (blanking ERA/K9 too).
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend import pipeline


def _slate_row(gid="20260822_TOR@NYY", name="TBD"):
    row = {c: np.nan for c in (
        ["game_id", "home_team", "away_team", "sp_name_home", "sp_name_away",
         "sp_era_home", "sp_k9_home", "sp_era_away", "sp_k9_away",
         "moneyline_home", "moneyline_away", "total_line", "run_line_home", "juice"])}
    row.update({
        "game_id": gid,
        "home_team": "NYY",
        "away_team": "TOR",
        "sp_name_home": name,
        "sp_name_away": name,
    })
    return row


class TestTodayGamesCsv(unittest.TestCase):
    def setUp(self):
        pipeline.DATA_DELIVERY_DIR = Path(tempfile.mkdtemp())

    def test_final_scores_written_to_artifact(self):
        games = pd.DataFrame([{
            "game_id": "20260822_TOR@NYY", "game_date": "2026-08-22",
            "home_team": "NYY", "away_team": "TOR",
            "home_score": 3.0, "away_score": 4.0, "total_runs": 7.0,
            "home_win": 0.0,
        }])
        path = pipeline._today_games_csv(games, "20990101")
        df = pd.read_csv(path)
        for col in ("home_score", "away_score", "total_runs"):
            self.assertIn(col, df.columns)
        self.assertEqual(int(df.iloc[0]["home_score"]), 3)
        self.assertEqual(int(df.iloc[0]["away_score"]), 4)


class TestCarryForwardSlateDetails(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        pipeline.DATA_DELIVERY_DIR = self.tmp

    def test_pitcher_names_restored_from_morning_artifact(self):
        morning = pd.DataFrame([
            {**_slate_row(), "sp_name_home": "Cole, Gerrit", "sp_name_away": "Gausman, Kevin",
             "moneyline_home": -150},
        ])
        morning.to_csv(self.tmp / "todays_games_20990101.csv", index=False)

        rebuilt = pd.DataFrame([{**_slate_row(), "moneyline_home": np.nan}])
        out = pipeline._carry_forward_slate_details(rebuilt, "20990101")
        self.assertEqual(out.iloc[0]["sp_name_home"], "Cole, Gerrit")
        self.assertEqual(out.iloc[0]["sp_name_away"], "Gausman, Kevin")
        self.assertEqual(out.iloc[0]["moneyline_home"], -150)

    def test_pitcher_stats_restored_with_names(self):
        """Evening rerun must restore ERA/K9 too — names alone don't re-derive
        stats without the pbp mapping, so the morning's stat lines carry."""
        morning = pd.DataFrame([
            {**_slate_row(), "sp_name_home": "Cole, Gerrit", "sp_name_away": "Gausman, Kevin",
             "sp_era_home": 3.10, "sp_k9_home": 9.9,
             "sp_era_away": 4.05, "sp_k9_away": 8.2},
        ])
        morning.to_csv(self.tmp / "todays_games_20990101.csv", index=False)

        rebuilt = pd.DataFrame([{**_slate_row()}])  # all TBD / NaN
        out = pipeline._carry_forward_slate_details(rebuilt, "20990101")
        self.assertEqual(out.iloc[0]["sp_name_home"], "Cole, Gerrit")
        self.assertAlmostEqual(float(out.iloc[0]["sp_era_home"]), 3.10)
        self.assertAlmostEqual(float(out.iloc[0]["sp_k9_home"]), 9.9)
        self.assertAlmostEqual(float(out.iloc[0]["sp_era_away"]), 4.05)

    def test_existing_values_not_overwritten(self):
        """A fresher non-TBD value on the rebuilt slate must win."""
        morning = pd.DataFrame([
            {**_slate_row(), "sp_name_home": "Cole, Gerrit", "sp_era_home": 3.10},
        ])
        morning.to_csv(self.tmp / "todays_games_20990101.csv", index=False)
        rebuilt = pd.DataFrame([{**_slate_row(), "sp_name_home": "New Guy", "sp_era_home": 2.95}])
        out = pipeline._carry_forward_slate_details(rebuilt, "20990101")
        self.assertEqual(out.iloc[0]["sp_name_home"], "New Guy")
        self.assertAlmostEqual(float(out.iloc[0]["sp_era_home"]), 2.95)

    def test_missing_previous_artifact_is_noop(self):
        slate = pd.DataFrame([_slate_row()])
        out = pipeline._carry_forward_slate_details(slate.copy(), "20990101")
        self.assertEqual(out.iloc[0]["sp_name_home"], "TBD")


if __name__ == "__main__":
    unittest.main()
