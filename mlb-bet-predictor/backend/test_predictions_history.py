"""
Regression tests for the per-game prediction-history artifact.

predictions_history_<date>.csv feeds the Calibration page's game-level
history table; it must contain exactly the walk-forward OOF games that
feed the reliability diagram — decided games only, with pick/winner/
correct derived consistently from home-side probabilities.
"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend import pipeline


def _oof():
    return pd.DataFrame([
        {"game_id": "20260820_A@B", "game_date": "2026-08-20", "home_team": "BOS",
         "away_team": "NYY", "home_score": 5, "away_score": 3, "home_win": 1.0,
         "home_win_prob_model": 0.62},
        {"game_id": "20260821_C@D", "game_date": "2026-08-21", "home_team": "PHI",
         "away_team": "ATL", "home_score": 2, "away_score": 7, "home_win": 0.0,
         "home_win_prob_model": 0.55},  # picked PHI, ATL won -> miss
        {"game_id": "20260822_E@F", "game_date": "2026-08-22", "home_team": "LAD",
         "away_team": "SD", "home_score": None, "away_score": None,
         "home_win": float("nan"), "home_win_prob_model": 0.70},  # undecided: drop
    ])


class TestPredictionsHistory(unittest.TestCase):
    def setUp(self):
        pipeline.DATA_DELIVERY_DIR = Path(tempfile.mkdtemp())

    def test_writes_decided_games_only_with_derived_columns(self):
        path = pipeline._predictions_history_csv(_oof(), "20990101")
        self.assertIsNotNone(path)
        df = pd.read_csv(path)
        self.assertEqual(len(df), 2)  # undecided row dropped
        self.assertEqual(list(df["game_date"]), ["2026-08-21", "2026-08-20"])  # newest first

        row = df[df["game_id"] == "20260821_C@D"].iloc[0]
        self.assertEqual(row["model_pick"], "PHI")   # prob 0.55 >= 0.5
        self.assertEqual(row["actual_winner"], "ATL")
        self.assertEqual(int(row["correct"]), 0)

        row = df[df["game_id"] == "20260820_A@B"].iloc[0]
        self.assertEqual(row["model_pick"], "BOS")
        self.assertEqual(row["actual_winner"], "BOS")
        self.assertEqual(int(row["correct"]), 1)

    def test_away_pick_probability_consistency(self):
        """A pick < 50% home prob means the model took the AWAY team."""
        oof = _oof()
        oof.loc[0, "home_win_prob_model"] = 0.40
        path = pipeline._predictions_history_csv(oof, "20990101")
        df = pd.read_csv(path)
        row = df[df["game_id"] == "20260820_A@B"].iloc[0]
        self.assertEqual(row["model_pick"], "NYY")
        self.assertEqual(row["actual_winner"], "BOS")
        self.assertEqual(int(row["correct"]), 0)

    def test_empty_or_missing_prob_column_returns_none(self):
        self.assertIsNone(pipeline._predictions_history_csv(None, "20990101"))
        self.assertIsNone(pipeline._predictions_history_csv(pd.DataFrame(), "20990101"))
        no_prob = _oof().drop(columns=["home_win_prob_model"])
        self.assertIsNone(pipeline._predictions_history_csv(no_prob, "20990101"))


if __name__ == "__main__":
    unittest.main()
