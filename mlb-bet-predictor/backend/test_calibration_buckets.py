"""
Regression tests for favored-team calibration bucketing.

calibration_buckets() takes the FAVORED team's perspective: each game
contributes ONE observation at max(p_home, p_away) ∈ [0.5, 1], labeled by
whether the favorite won. Information-equivalent to the old home-side view —
(p, y) and (1 − p, 1 − y) are exact complements — but buckets read the way a
bettor sees them ("a team at 70% wins 70% of the time").
"""
import unittest

import numpy as np

from backend.training import calibration_buckets


class TestFavoredCalibrationBuckets(unittest.TestCase):
    def test_home_favorite_win_counts_once(self):
        """Home team at 0.6 that wins: one obs in 60–70%, mean_actual 1.0."""
        out = calibration_buckets(np.array([1.0]), np.array([0.60]))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["bucket"], "60–70%")
        self.assertEqual(out[0]["count"], 1)
        self.assertEqual(out[0]["mean_predicted"], 0.6)
        self.assertEqual(out[0]["mean_actual"], 1.0)

    def test_away_favorite_is_mirrored_not_double_counted(self):
        """Home prob 0.40, away team (favorite at 0.60) wins → same as above."""
        out = calibration_buckets(np.array([0.0]), np.array([0.40]))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["bucket"], "60–70%")
        self.assertEqual(out[0]["mean_predicted"], 0.6)
        self.assertEqual(out[0]["mean_actual"], 1.0)

    def test_away_favorite_losing_shows_miss(self):
        """Home prob 0.40, HOME wins (favorite lost) → mean_actual 0.0, gap +0.6."""
        out = calibration_buckets(np.array([1.0]), np.array([0.40]))
        self.assertEqual(out[0]["mean_actual"], 0.0)
        self.assertEqual(out[0]["gap"], 0.6)

    def test_no_observation_below_fifty_percent(self):
        """Every observation sits at ≥ 0.5; lower half is never populated."""
        rng = np.random.RandomState(0)
        p = rng.uniform(0.05, 0.95, 500)
        y = rng.randint(0, 2, 500).astype(float)
        out = calibration_buckets(y, p)
        for b in out:
            lo = int(b["bucket"].split("–")[0])
            self.assertGreaterEqual(lo, 50)
            self.assertGreaterEqual(b["mean_predicted"], 0.5)
        # Every game lands exactly once
        self.assertEqual(sum(b["count"] for b in out), 500)

    def test_probability_of_exactly_one_lands_in_top_bucket(self):
        out = calibration_buckets(np.array([1.0]), np.array([1.0]))
        self.assertEqual(out[0]["bucket"], "90–100%")


if __name__ == "__main__":
    unittest.main()
