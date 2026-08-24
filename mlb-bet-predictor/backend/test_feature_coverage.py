"""Tests for the per-feature coverage artifact.

The 2026 weather truncation produced 'healthy-looking' PSI rows over data
that was mostly default zeros — nothing surfaced the absence. These tests
lock in the measured-vs-default separation that makes starvation visible.
"""
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import pandas as pd

import explainability
from explainability import compute_feature_coverage


def _frame(n: int, wind_vals, dome_vals, air_vals) -> pd.DataFrame:
    return pd.DataFrame({
        "game_date": pd.date_range("2026-08-10", periods=n),
        "home_win": [1.0] * n,
        "dome_is_neutral": dome_vals,
        "wind_advantage_flyball_factor": wind_vals,
        "air_density_velocity_boost": air_vals,
        # A couple of ordinary features so FEATURE_COLS iteration is exercised.
        "elo_diff": np.linspace(0, 1, n),
        "is_home": [1.0] * n,
    })


class TestFeatureCoverage(unittest.TestCase):
    def setUp(self):
        # Keep test artifacts out of the real data_delivery directory.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = unittest.mock.patch.object(
            explainability, "DATA_DELIVERY_DIR", Path(tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_zeros_are_separated_from_measured(self):
        # 5 real observations (nonzero), 3 exact-0.0 at domes = DEFAULT branch,
        # 2 NaN. Measured must count only the 5 real ones.
        wind = [-0.02, 0.03, -0.01, 0.05, 0.02, 0.0, 0.0, 0.0, np.nan, np.nan]
        dome = [0.0] * 5 + [1.0, 1.0, 1.0, 0.0, 0.0]
        air = [0.01, -0.02, 0.03, 0.04, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        cur = _frame(10, wind, dome, air)
        base = _frame(4, [0.01, 0.0, np.nan, -0.02], [0.0, 1.0, 1.0, 0.0],
                      [0.01, np.nan, np.nan, 0.02])

        df = compute_feature_coverage(base, cur, "20260824")

        wcur = df[(df.feature == "wind_advantage_flyball_factor") & (df.window == "current")].iloc[0]
        self.assertEqual(wcur.n_games, 10)
        self.assertEqual(wcur.n_nonnull, 8)
        self.assertEqual(wcur.n_default_zero, 3)
        self.assertEqual(wcur.n_measured, 5)
        self.assertAlmostEqual(wcur.pct_measured, 50.0)
        self.assertEqual(wcur.status, "LOW_COVERAGE")

    def test_air_density_nonnull_implies_measured(self):
        # The dome branch never writes air density; every non-null value came
        # from a fetched observation.
        cur = _frame(6, [np.nan] * 6, [1.0] * 6,
                     [0.01, np.nan, 0.02, np.nan, np.nan, np.nan])
        df = compute_feature_coverage(cur.head(2), cur, "20260824")
        acur = df[(df.feature == "air_density_velocity_boost") & (df.window == "current")].iloc[0]
        self.assertEqual(acur.n_nonnull, 2)
        self.assertEqual(acur.n_default_zero, 0)
        self.assertEqual(acur.n_measured, 2)

    def test_starved_status_when_nothing_measured(self):
        # All-dome zeros with no observations: exactly the shipped-artifact
        # state that hid the 2026 truncation.
        cur = _frame(5, [0.0] * 5, [1.0] * 5, [np.nan] * 5)
        df = compute_feature_coverage(cur.head(2), cur, "20260824")
        wcur = df[(df.feature == "wind_advantage_flyball_factor") & (df.window == "current")].iloc[0]
        self.assertEqual(wcur.n_default_zero, 5)
        self.assertEqual(wcur.n_measured, 0)
        self.assertEqual(wcur.status, "STARVED")

    def test_healthy_frame_is_ok(self):
        rng = np.random.default_rng(7)
        cur = _frame(30, rng.normal(0, 0.05, 30), [0.0] * 30, rng.normal(0, 0.02, 30))
        df = compute_feature_coverage(cur.tail(5), cur, "20260824")
        wcur = df[(df.feature == "wind_advantage_flyball_factor") & (df.window == "current")].iloc[0]
        self.assertEqual(wcur.status, "OK")


if __name__ == "__main__":
    unittest.main()
