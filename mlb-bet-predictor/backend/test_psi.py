"""PSI computation tests.

Validates that ``compute_psi`` returns expected numeric ranges:

* identical distributions -> ~0 (below the WARN threshold);
* shifted distributions -> clearly above the WARN threshold;
* degenerate (constant) inputs -> 0.0;
* outputs are always non-negative;
* status mapping OK / WARN / ALERT follows config thresholds.
"""

import unittest

import numpy as np
import pandas as pd

import explainability as ex
from config import PSI_ALERT, PSI_WARN


class PSITests(unittest.TestCase):
    def test_identical_distributions_are_near_zero(self):
        rng = np.random.default_rng(7)
        a = rng.normal(0.5, 0.1, 500)
        b = a.copy()
        self.assertLess(ex.compute_psi(a, b), 1e-6)

    def test_shifted_distribution_exceeds_warn(self):
        rng = np.random.default_rng(7)
        a = rng.normal(0.30, 0.08, 500)   # low-wOBA baseline
        b = rng.normal(0.40, 0.08, 500)   # inflated current values
        psi = ex.compute_psi(a, b)
        self.assertGreater(psi, PSI_WARN)
        # PSI values for clearly separated distributions are large (often > 1),
        # so no upper bound is asserted here.

    def test_heavy_shift_is_alert(self):
        rng = np.random.default_rng(7)
        a = rng.uniform(0.0, 1.0, 1000)
        b = rng.uniform(2.0, 3.0, 1000)
        self.assertGreaterEqual(ex.compute_psi(a, b), PSI_ALERT)

    def test_degenerate_constant_inputs_return_zero(self):
        a = np.full(100, 3.5)
        b = np.full(100, 3.5)
        self.assertEqual(ex.compute_psi(a, b), 0.0)
        # single unique value on one side only
        self.assertGreaterEqual(ex.compute_psi(a, np.full(100, 4.0)), 0.0)

    def test_psi_is_never_negative(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            a = rng.normal(0, 1, 300)
            b = rng.normal(0.3, 1.4, 300)
            self.assertGreaterEqual(ex.compute_psi(a, b), 0.0)

    def test_status_mapping(self):
        self.assertEqual(ex.psi_status(0.05), "OK")
        self.assertEqual(ex.psi_status(PSI_WARN - 1e-6), "OK")
        self.assertEqual(ex.psi_status(PSI_WARN + 1e-6), "WARN")
        self.assertEqual(ex.psi_status(PSI_ALERT - 1e-6), "WARN")
        self.assertEqual(ex.psi_status(PSI_ALERT + 1e-6), "ALERT")

    def test_feature_drift_table(self):
        rng = np.random.default_rng(11)
        current = pd.DataFrame({"home_team_elo": rng.normal(1520, 40, 300)})
        baseline = pd.DataFrame({"home_team_elo": rng.normal(1500, 40, 300)})
        drift = ex.compute_feature_drift(current, baseline, features=["home_team_elo"])
        self.assertEqual(list(drift.columns), ["feature", "current_mean", "baseline_mean", "psi", "status"])
        self.assertEqual(drift.iloc[0]["feature"], "home_team_elo")
        self.assertIn(drift.iloc[0]["status"], {"OK", "WARN", "ALERT"})

    def test_psi_detects_dramatic_shift(self):
        """Uniform(0,1) vs a point mass at 0.5 must produce large positive PSI."""
        rng = np.random.default_rng(5)
        a = rng.uniform(0.0, 1.0, 2000)
        b = np.full(2000, 0.5)
        psi = ex.compute_psi(a, b)
        self.assertGreater(psi, PSI_ALERT)
        self.assertTrue(np.isfinite(psi))


if __name__ == "__main__":
    unittest.main()
