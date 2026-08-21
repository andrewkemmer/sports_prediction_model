"""Walk-forward training tests.

Prove the expanding-window procedure never leaks the future:

* every train fold contains only games strictly before the validation window;
* training data expands monotonically across folds;
* validation windows never overlap;
* removing games after a window's end does not change its train fold.
"""

import unittest

import pandas as pd

import training as tr


def make_games(n_days: int = 400, start: str = "2026-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "game_id": [f"G{i}" for i in range(n_days)],
            "start_time": dates,
            "home_team": "BOS",
            "away_team": "NYY",
            "home_runs": 1,
            "away_runs": 1,
        }
    )


class WalkForwardTests(unittest.TestCase):
    def setUp(self):
        self.games = make_games()

    def test_train_folds_are_strictly_historical(self):
        """max(train.start_time) must be strictly before min(valid.start_time)."""
        splits = list(tr.walk_forward_splits(self.games, cadence_days=7, min_train_days=90))
        self.assertGreater(len(splits), 10)
        for s in splits:
            self.assertGreater(
                s.valid["start_time"].min(), s.train["start_time"].max(),
                msg=f"fold {s.fold}: training set contains future games",
            )

    def test_no_train_row_at_or_after_valid_start(self):
        for s in tr.walk_forward_splits(self.games, cadence_days=7, min_train_days=90):
            self.assertTrue((s.train["start_time"] < s.valid_start).all())

    def test_expanding_window(self):
        """Training size must be non-decreasing across folds."""
        sizes = [len(s.train) for s in tr.walk_forward_splits(self.games, cadence_days=7, min_train_days=90)]
        for prev, cur in zip(sizes, sizes[1:]):
            self.assertGreaterEqual(cur, prev)

    def test_validation_windows_do_not_overlap(self):
        prev_end = None
        for s in tr.walk_forward_splits(self.games, cadence_days=7, min_train_days=90):
            if prev_end is not None:
                self.assertGreaterEqual(s.valid_start, prev_end)
            prev_end = s.valid_end

    def test_removing_future_games_does_not_change_train_fold(self):
        """Point-in-time for training: outcomes after valid_end are invisible."""
        splits_full = list(tr.walk_forward_splits(self.games, cadence_days=7, min_train_days=90))
        s = splits_full[3]
        truncated = self.games[self.games["start_time"] < s.valid_end]
        splits_trunc = list(
            tr.walk_forward_splits(truncated, cadence_days=7, min_train_days=90)
        )
        s2 = splits_trunc[3]
        self.assertEqual(
            set(s.train["game_id"]), set(s2.train["game_id"]),
            "train fold changed when future games were removed",
        )

    def test_insufficient_history_yields_no_splits(self):
        short = make_games(n_days=30)
        self.assertEqual(list(tr.walk_forward_splits(short, min_train_days=90)), [])

    def test_prepare_training_frame_keeps_completed_rows(self):
        feat = pd.DataFrame(
            {
                "game_id": ["a", "b"],
                "start_time": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
                "home_runs": [3.0, None],
                "away_runs": [2.0, 1.0],
                "home_win": [1.0, 0.0],
                "total_runs": [5.0, None],
                "home_cover": [1.0, 0.0],
                "elo_diff": [10.0, 5.0],
            }
        )
        for col in tr.FEATURE_COLUMNS:
            feat[col] = 1.0
        cleaned = tr.prepare_training_frame(feat)
        self.assertEqual(len(cleaned), 1)


if __name__ == "__main__":
    unittest.main()
