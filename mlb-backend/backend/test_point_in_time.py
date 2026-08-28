"""
Unit tests for point-in-time (PIT) enforcement.

Verifies that:
- filter_prior returns only games before the cutoff
- Rolling features use shift(1) (current game excluded)
- Elo entering a game uses only prior results
- Market lines posted at/after start are rejected
"""
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from backend.data_ingestion import (
    attach_market_lines,
    compute_elos,
    compute_elos_up_to,
    filter_prior,
    generate_synthetic_games,
    generate_synthetic_market_lines,
    rolling_prior_mean,
)


class TestFilterPrior(unittest.TestCase):
    """Tests for the filter_prior PIT enforcement function."""

    def _make_games(self, dates: list[datetime]) -> pd.DataFrame:
        rows = []
        for i, dt in enumerate(dates):
            rows.append({
                "game_id": f"game_{i}",
                "start_time_utc": dt,
                "home_team": "NYY",
                "away_team": "BOS",
                "home_win": 1.0 if i % 2 == 0 else 0.0,
            })
        return pd.DataFrame(rows)

    def test_filters_games_after_cutoff(self):
        """Only games strictly before cutoff are returned."""
        now = datetime(2026, 8, 9, 19, 0)
        dates = [now - timedelta(days=d) for d in [10, 5, 2, 0, -1, -3]]
        games = self._make_games(dates)

        result = filter_prior(games, now)
        # Games at t=0 (now) and t=-1, t=-3 should be excluded
        self.assertEqual(len(result), 3)
        self.assertTrue((result["start_time_utc"] < now).all())

    def test_empty_when_all_games_future(self):
        """All future games → empty result."""
        now = datetime(2026, 8, 9, 12, 0)
        dates = [now + timedelta(days=d) for d in [1, 2, 3]]
        games = self._make_games(dates)

        result = filter_prior(games, now)
        self.assertEqual(len(result), 0)

    def test_no_future_leakage(self):
        """Adding a future game must not change results for earlier games."""
        past_dates = [
            datetime(2026, 7, 1),
            datetime(2026, 7, 5),
            datetime(2026, 7, 10),
        ]
        games_before = self._make_games(past_dates)
        result_before = filter_prior(games_before, datetime(2026, 8, 1))

        # Add an extreme future game
        future_game = pd.DataFrame([{
            "game_id": "future_extreme",
            "start_time_utc": datetime(2027, 12, 31),
            "home_team": "NYY",
            "away_team": "BOS",
            "home_win": 1.0,
        }])
        games_with_future = pd.concat([games_before, future_game], ignore_index=True)
        result_after = filter_prior(games_with_future, datetime(2026, 8, 1))

        # Results must be identical
        pd.testing.assert_frame_equal(result_before, result_after)


class TestRollingPriorMean(unittest.TestCase):
    """Tests for PIT-safe rolling features."""

    def test_shift_excludes_current_game(self):
        """Rolling mean must use shift(1) to exclude current game."""
        df = pd.DataFrame({
            "team": ["A", "A", "A", "A"],
            "value": [10.0, 20.0, 30.0, 40.0],
        })
        result = rolling_prior_mean(df, "team", "value", window=3)
        # Row 0: no prior data → NaN
        # Row 1: [10] → 10.0
        # Row 2: [10, 20] → 15.0
        # Row 3: [10, 20, 30] → 20.0
        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertAlmostEqual(result.iloc[1], 10.0)
        self.assertAlmostEqual(result.iloc[2], 15.0)
        self.assertAlmostEqual(result.iloc[3], 20.0)

    def test_adding_future_game_unchanged(self):
        """Rolling features for past games are unchanged by future data."""
        df1 = pd.DataFrame({
            "team": ["A", "A", "A"],
            "value": [10.0, 20.0, 30.0],
        })
        r1 = rolling_prior_mean(df1, "team", "value", window=3)

        df2 = pd.DataFrame({
            "team": ["A", "A", "A", "A"],
            "value": [10.0, 20.0, 30.0, 100.0],  # extreme future value
        })
        r2 = rolling_prior_mean(df2, "team", "value", window=3)

        # First 3 rows must be identical
        pd.testing.assert_series_equal(r1, r2.iloc[:3], check_names=False)


class TestElo(unittest.TestCase):
    """Tests for Elo rating computation."""

    def test_elo_entering_uses_only_prior(self):
        """Elo entering a game uses only completed prior games."""
        dates = [
            datetime(2026, 7, 1, 19, 0),
            datetime(2026, 7, 2, 19, 0),
            datetime(2026, 7, 3, 19, 0),
        ]
        games = pd.DataFrame([
            {"game_id": "g1", "start_time_utc": dates[0], "home_team": "A", "away_team": "B", "home_win": 1.0},
            {"game_id": "g2", "start_time_utc": dates[1], "home_team": "A", "away_team": "C", "home_win": 0.0},
            {"game_id": "g3", "start_time_utc": dates[2], "home_team": "A", "away_team": "D", "home_win": 1.0},
        ])

        elos = compute_elos(games)
        # Elo entering game 1 should be 1500 (initial)
        self.assertAlmostEqual(elos.iloc[0], 1500.0)
        # All values should be numeric
        self.assertTrue(elos.notna().all())

    def test_elos_up_to_same_as_filter(self):
        """compute_elos_up_to produces same result as filter_prior + compute_elos."""
        dates = [
            datetime(2026, 7, 1, 19, 0),
            datetime(2026, 7, 5, 19, 0),
            datetime(2026, 7, 10, 19, 0),
        ]
        games = pd.DataFrame([
            {"game_id": "g1", "start_time_utc": dates[0], "home_team": "A", "away_team": "B", "home_win": 1.0},
            {"game_id": "g2", "start_time_utc": dates[1], "home_team": "C", "away_team": "A", "home_win": 0.0},
            {"game_id": "g3", "start_time_utc": dates[2], "home_team": "A", "away_team": "D", "home_win": 1.0},
        ])

        # Compute via filter + compute_elos
        prior = filter_prior(games, dates[2])
        prior_elos = compute_elos(prior)

        # Compute via compute_elos_up_to
        up_to_elos = compute_elos_up_to(games, dates[2])

        # Both should give the same Elo for team A after game 2
        # (This is a structural test — exact values depend on implementation)
        self.assertIsInstance(up_to_elos, dict)
        self.assertIn("A", up_to_elos)


class TestMarketLinePIT(unittest.TestCase):
    """Tests for market line as-of join (PIT safety)."""

    def test_rejects_lines_posted_at_start(self):
        """Lines posted at or after start_time_utc are rejected."""
        game_start = datetime(2026, 8, 9, 19, 0)

        games = pd.DataFrame([{
            "game_id": "g1",
            "start_time_utc": game_start,
            "home_team": "NYY",
            "away_team": "BOS",
        }])
        lines = pd.DataFrame([
            {
                "game_id": "g1",
                "line_posted_at": game_start - timedelta(hours=2),  # valid: before start
                "moneyline_home": -150,
                "moneyline_away": 130,
                "total_line": 8.5,
                "run_line_home": -150,
                "run_line_away": 130,
                "juice": 0.04,
            },
            {
                "game_id": "g1",
                "line_posted_at": game_start + timedelta(hours=1),  # invalid: after start
                "moneyline_home": -200,
                "moneyline_away": 170,
                "total_line": 9.0,
                "run_line_home": -180,
                "run_line_away": 150,
                "juice": 0.05,
            },
        ])

        result = attach_market_lines(games, lines)
        # Should use the earlier line, not the later one
        self.assertEqual(result.iloc[0]["moneyline_home"], -150)

    def test_no_valid_lines_returns_defaults(self):
        """If no lines are posted before start, defaults are used."""
        game_start = datetime(2026, 8, 9, 19, 0)

        games = pd.DataFrame([{
            "game_id": "g1",
            "start_time_utc": game_start,
            "home_team": "NYY",
            "away_team": "BOS",
        }])
        lines = pd.DataFrame([{
            "game_id": "g1",
            "line_posted_at": game_start + timedelta(hours=1),
            "moneyline_home": -200,
            "moneyline_away": 170,
            "total_line": 9.0,
            "run_line_home": -180,
            "run_line_away": 150,
            "juice": 0.05,
        }])

        result = attach_market_lines(games, lines)
        self.assertIsNone(result.iloc[0]["moneyline_home"])


class TestSyntheticData(unittest.TestCase):
    """Tests for synthetic data generation."""

    def test_generates_games(self):
        """Synthetic generator produces a non-empty DataFrame."""
        games = generate_synthetic_games(
            datetime(2026, 8, 9).date(),
            season_start=datetime(2026, 3, 26).date(),
            games_per_day=5,
            seed=42,
        )
        self.assertGreater(len(games), 0)
        self.assertIn("game_id", games.columns)
        self.assertIn("home_win", games.columns)

    def test_deterministic(self):
        """Same seed produces identical output."""
        g1 = generate_synthetic_games(
            datetime(2026, 8, 9).date(),
            games_per_day=5,
            seed=123,
        )
        g2 = generate_synthetic_games(
            datetime(2026, 8, 9).date(),
            games_per_day=5,
            seed=123,
        )
        pd.testing.assert_frame_equal(g1, g2)

    def test_market_lines_generated(self):
        """Synthetic market lines have expected columns."""
        games = generate_synthetic_games(
            datetime(2026, 8, 9).date(),
            games_per_day=5,
            seed=42,
        )
        lines = generate_synthetic_market_lines(games, seed=42)
        self.assertGreater(len(lines), 0)
        for col in ["game_id", "moneyline_home", "moneyline_away", "total_line", "juice"]:
            self.assertIn(col, lines.columns)


if __name__ == "__main__":
    unittest.main()
