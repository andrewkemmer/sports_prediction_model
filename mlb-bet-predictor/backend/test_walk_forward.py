"""
Unit tests for walk-forward training.

Verifies that:
- Train folds are strictly historical (no future data)
- Validation windows are non-overlapping and chronological
- Removing future games leaves folds unchanged
- Expanding window grows over time
"""
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from backend.training import walk_forward_splits


def _make_games(n_days: int = 30, games_per_day: int = 5, seed: int = 42) -> pd.DataFrame:
    """Generate a simple game DataFrame for testing."""
    rng = np.random.RandomState(seed)
    rows = []
    start = datetime(2026, 4, 1)
    game_counter = 0
    for d in range(n_days):
        day = start + timedelta(days=d)
        for g in range(games_per_day):
            rows.append({
                "game_id": f"g{game_counter}",
                "game_date": day,
                "start_time_utc": day + timedelta(hours=19),
                "home_team": f"T{g % 5}",
                "away_team": f"T{(g + 2) % 5}",
                "home_win": float(rng.randint(0, 2)),
                "home_elo": 1500 + rng.normal(0, 50),
                "home_win_pct": 0.5 + rng.normal(0, 0.1),
                "away_win_pct": 0.5 + rng.normal(0, 0.1),
                "sp_era_home": 3.5 + rng.normal(0, 0.5),
                "sp_era_away": 3.5 + rng.normal(0, 0.5),
                "sp_k9_home": 9.0 + rng.normal(0, 1),
                "sp_k9_away": 9.0 + rng.normal(0, 1),
                "woba_30g_home": 0.320 + rng.normal(0, 0.02),
                "woba_30g_away": 0.320 + rng.normal(0, 0.02),
                "bullpen_whip_10g_home": 1.2 + rng.normal(0, 0.1),
                "bullpen_whip_10g_away": 1.2 + rng.normal(0, 0.1),
                "rest_days_home": rng.randint(0, 3),
                "rest_days_away": rng.randint(0, 3),
                "sp_era_5g_home": 3.5 + rng.normal(0, 0.5),
                "sp_era_5g_away": 3.5 + rng.normal(0, 0.5),
                "sp_k9_5g_home": 9.0 + rng.normal(0, 1),
                "sp_k9_5g_away": 9.0 + rng.normal(0, 1),
                "total_runs": rng.randint(2, 15),
                "home_run_diff": 0,
                "away_run_diff": 0,
            })
            game_counter += 1
    return pd.DataFrame(rows)


class TestWalkForwardSplits(unittest.TestCase):
    """Tests for walk-forward split generation."""

    def test_train_is_strictly_before_val(self):
        """Every training game must be before every validation game."""
        games = _make_games(n_days=60, games_per_day=3)
        splits = walk_forward_splits(games, retrain_cadence_days=7)

        self.assertGreater(len(splits), 0)
        for split in splits:
            train_max = split["train_games"]["game_date"].max()
            val_min = split["val_games"]["game_date"].min()
            self.assertLess(
                train_max, val_min,
                f"Train max {train_max} is not strictly before val min {val_min}"
            )

    def test_validation_windows_non_overlapping(self):
        """Validation windows must not overlap."""
        games = _make_games(n_days=60, games_per_day=3)
        splits = walk_forward_splits(games, retrain_cadence_days=7)

        for i in range(len(splits) - 1):
            val_end_i = splits[i]["val_end"]
            val_start_next = splits[i + 1]["val_start"]
            self.assertLessEqual(
                val_end_i, val_start_next,
                f"Fold {i} val_end {val_end_i} overlaps with fold {i+1} val_start {val_start_next}"
            )

    def test_expanding_window(self):
        """Training set should grow over successive folds."""
        games = _make_games(n_days=90, games_per_day=3)
        splits = walk_forward_splits(games, retrain_cadence_days=7)

        if len(splits) >= 2:
            train_sizes = [len(s["train_games"]) for s in splits]
            # Each training set should be >= the previous
            for i in range(1, len(train_sizes)):
                self.assertGreaterEqual(
                    train_sizes[i], train_sizes[i - 1],
                    f"Fold {i} train size {train_sizes[i]} < fold {i-1} {train_sizes[i-1]}"
                )

    def test_no_future_leakage(self):
        """Removing a future game doesn't change earlier folds."""
        games = _make_games(n_days=60, games_per_day=3)
        splits_orig = walk_forward_splits(games, retrain_cadence_days=7)

        # Add an extreme future game
        future_game = pd.DataFrame([{
            "game_id": "future_extreme",
            "game_date": datetime(2099, 1, 1),
            "start_time_utc": datetime(2099, 1, 1, 19, 0),
            "home_team": "T0",
            "away_team": "T1",
            "home_win": 1.0,
            "home_elo": 9999.0,
            "home_win_pct": 0.999,
            "away_win_pct": 0.001,
            "sp_era_home": 0.01,
            "sp_era_away": 20.0,
            "sp_k9_home": 20.0,
            "sp_k9_away": 0.1,
            "woba_30g_home": 0.500,
            "woba_30g_away": 0.100,
            "bullpen_whip_10g_home": 0.5,
            "bullpen_whip_10g_away": 3.0,
            "rest_days_home": 10,
            "rest_days_away": 0,
            "sp_era_5g_home": 0.5,
            "sp_era_5g_away": 10.0,
            "sp_k9_5g_home": 15.0,
            "sp_k9_5g_away": 3.0,
            "total_runs": 20,
            "home_run_diff": 100,
            "away_run_diff": -100,
        }])
        games_with_future = pd.concat([games, future_game], ignore_index=True)
        splits_with_future = walk_forward_splits(games_with_future, retrain_cadence_days=7)

        # The future game may extend the last validation window,
        # but all earlier folds must be identical (no future leakage).
        # Compare all folds except the last one.
        common_folds = min(len(splits_orig), len(splits_with_future)) - 1
        for i in range(common_folds):
            train_ids_orig = set(splits_orig[i]["train_games"]["game_id"])
            train_ids_future = set(splits_with_future[i]["train_games"]["game_id"])
            self.assertEqual(train_ids_orig, train_ids_future, f"Fold {i} train game_ids differ")

            val_ids_orig = set(splits_orig[i]["val_games"]["game_id"])
            val_ids_future = set(splits_with_future[i]["val_games"]["game_id"])
            self.assertEqual(val_ids_orig, val_ids_future, f"Fold {i} val game_ids differ")

            self.assertEqual(
                len(splits_orig[i]["train_games"]),
                len(splits_with_future[i]["train_games"]),
                f"Fold {i} train size differs",
            )
            self.assertEqual(
                len(splits_orig[i]["val_games"]),
                len(splits_with_future[i]["val_games"]),
                f"Fold {i} val size differs",
            )

    def test_max_eval_folds_limits_output(self):
        """max_eval_folds caps the number of returned splits."""
        games = _make_games(n_days=90, games_per_day=3)
        all_splits = walk_forward_splits(games, retrain_cadence_days=7, max_eval_folds=0)
        limited = walk_forward_splits(games, retrain_cadence_days=7, max_eval_folds=3)

        self.assertLessEqual(len(limited), 3)
        self.assertGreaterEqual(len(all_splits), len(limited))

    def test_empty_when_insufficient_data(self):
        """Very short data returns empty splits."""
        games = _make_games(n_days=3, games_per_day=2)
        splits = walk_forward_splits(games, retrain_cadence_days=7)
        self.assertEqual(len(splits), 0)

    def test_requires_game_date_column(self):
        """Missing game_date column raises ValueError."""
        df = pd.DataFrame({"home_win": [1, 0]})
        with self.assertRaises(ValueError):
            walk_forward_splits(df)


class TestPartialTailFold(unittest.TestCase):
    """A frame whose final week is shorter than the cadence must surface OOF
    predictions for the tail days (08-23-cap regression).

    The last fold's full cadence window would overrun the frame's max date,
    so it runs PARTIALLY into the tail (val_end = frame max) and is kept even
    below the min-val gate — leakage-free because train stays strictly before
    val_start.
    """

    def _tail_frame(self, n_days=38, games_per_day=8):
        """38 days = 5 full 7-day folds + a 3-day partial tail (24 games
        < MIN_VAL_FOLD_GAMES=40 but >= 5), so the tail fold would be dropped
        by the gate without the partial-tail exemption."""
        return _make_games(n_days=n_days, games_per_day=games_per_day, seed=11)

    def test_splits_flag_final_overrun_fold(self):
        """walk_forward_splits emits is_partial_tail=True only on the final
        fold, with val_end == frame max, and train strictly < val_start."""
        from backend.training import walk_forward_splits
        games = self._tail_frame()
        splits = walk_forward_splits(games, retrain_cadence_days=7)
        self.assertGreaterEqual(len(splits), 2)
        last = splits[-1]
        self.assertTrue(last["is_partial_tail"])
        self.assertEqual(last["val_end"].date(),
                         games["game_date"].max().date())
        # Leakage: the tail fold trains strictly before its val_start.
        self.assertLess(last["train_games"]["game_date"].max(),
                        last["val_games"]["game_date"].min())
        # Only the final fold is flagged partial tail.
        self.assertEqual(sum(1 for s in splits if s["is_partial_tail"]), 1)

    def test_eval_produces_oof_rows_for_tail_days(self):
        """walk_forward_evaluate keeps the final partial-tail fold below the
        min-val gate, so its tail-day games get OOF predictions (elt NEVER
        dropped by min_val_games)."""
        import numpy as np
        from backend.training import walk_forward_evaluate
        games = self._tail_frame()
        frame_max = games["game_date"].max()
        tail_min_date = games[games["game_date"] >= games["game_date"].max()
                              - np.timedelta64(2, "D")]["game_date"].min()

        _models, _pooled, combined = walk_forward_evaluate(
            games, retrain_cadence_days=7, min_train_days=0)
        self.assertFalse(combined.empty)
        combined["game_date"] = pd.to_datetime(combined["game_date"])
        # The combined OOF frame now reaches the frame's max date.
        self.assertEqual(combined["game_date"].max(), frame_max)
        # Tail-day games are present and carry a finite OOF probability.
        tail_rows = combined[combined["game_date"] >= tail_min_date]
        self.assertGreater(len(tail_rows), 0)
        self.assertTrue(np.isfinite(tail_rows["home_win_prob_model"]).all())
        # Per-fold leakage: the executed tail fold's train < its val_start.
        from backend.training import walk_forward_splits
        splits = walk_forward_splits(games, retrain_cadence_days=7)
        last = splits[-1]
        self.assertLess(last["train_games"]["game_date"].max(),
                        last["val_games"]["game_date"].min())


if __name__ == "__main__":
    unittest.main()
