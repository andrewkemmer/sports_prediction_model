"""
Regression tests for artifact builders that must tolerate NULL outcomes.

Ties/postponements ship home_win = NULL (by design, after the missing-data
fixes); every consumer must skip them instead of crashing on int(NaN).
"""
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from backend import pipeline


class TestPowerRankingsNullOutcome(unittest.TestCase):
    def test_undecided_games_are_skipped_not_crashed(self):
        """home_win = NULL rows must not break int() conversion or pct math."""
        pipeline.DATA_DELIVERY_DIR = Path(tempfile.mkdtemp())
        games = pd.DataFrame([
            {"game_date": "2026-08-20", "home_team": "BOS", "away_team": "NYY",
             "home_win": 1.0, "home_elo": 1520.0},
            {"game_date": "2026-08-21", "home_team": "BOS", "away_team": "SEA",
             "home_win": float("nan"), "home_elo": 1521.0},  # tied/postponed
            {"game_date": "2026-08-19", "home_team": "SEA", "away_team": "BOS",
             "home_win": 0.0, "home_elo": 1519.0},
        ])
        path = pipeline._power_rankings_csv(games, "20990101")
        df = pd.read_csv(path, index_col=0)

        bos = df[df["team"] == "BOS"].iloc[0]
        # Decided only: W at home vs NYY + W away @ SEA -> 2-0; the NaN game
        # (tied/postponed) is excluded entirely.
        self.assertEqual(int(bos["wins"]), 2)
        self.assertEqual(int(bos["losses"]), 0)
        self.assertEqual(bos["record"], "2-0")
        wins10, losses10 = bos["l10"].split("-")
        self.assertEqual(int(wins10) + int(losses10), 2)  # NaN game not counted

    def test_all_undecided_team_gets_neutral_record(self):
        """A team with zero decided games must produce a valid row."""
        pipeline.DATA_DELIVERY_DIR = Path(tempfile.mkdtemp())
        games = pd.DataFrame([
            {"game_date": "2026-08-21", "home_team": "LAD", "away_team": "SF",
             "home_win": float("nan"), "home_elo": 1505.0},
        ])
        path = pipeline._power_rankings_csv(games, "20990102")
        df = pd.read_csv(path, index_col=0)
        # Teams list comes from home_team; SF (away-only) still must not crash
        lad = df[df["team"] == "LAD"].iloc[0]
        self.assertEqual(lad["record"], "0-0")


if __name__ == "__main__":
    unittest.main()
