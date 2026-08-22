"""
Unit tests for PSI (Population Stability Index) computation.

Verifies that:
- Identical distributions ≈ 0
- Shifted distributions exceed WARN threshold
- Degenerate inputs return 0
- PSI is never negative
- Status mapping works correctly
"""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from backend import explainability
from backend.explainability import compute_psi, psi_status, psi_noise_floor, compute_feature_drift
from backend.config import PSI_WARN_THRESHOLD, PSI_ALERT_THRESHOLD


class TestPSIComputation(unittest.TestCase):
    """Tests for the compute_psi function."""

    def test_identical_distributions_near_zero(self):
        """Identical distributions should produce PSI ≈ 0."""
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        self.assertAlmostEqual(psi, 0.0, places=5)

    def test_shifted_distributions_positive(self):
        """Shifted distributions should produce positive PSI."""
        rng = np.random.RandomState(42)
        baseline = rng.normal(0, 1, 1000)
        current = rng.normal(1, 1, 1000)  # shifted mean
        psi = compute_psi(baseline, current)
        self.assertGreater(psi, 0.0)

    def test_shifted_exceeds_warn(self):
        """A large shift should exceed the WARN threshold."""
        rng = np.random.RandomState(42)
        baseline = rng.normal(0, 0.5, 500)
        current = rng.normal(3, 0.5, 500)  # very large shift
        psi = compute_psi(baseline, current)
        self.assertGreater(psi, PSI_WARN_THRESHOLD)

    def test_psi_never_negative(self):
        """PSI must never be negative regardless of input."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            a = rng.normal(0, 1, 200)
            b = rng.normal(rng.uniform(-3, 3), rng.uniform(0.1, 2), 200)
            psi = compute_psi(a, b)
            self.assertGreaterEqual(psi, 0.0)

    def test_degenerate_empty_arrays(self):
        """Empty arrays should return 0."""
        psi = compute_psi(np.array([]), np.array([1, 2, 3]))
        self.assertEqual(psi, 0.0)

        psi = compute_psi(np.array([1, 2, 3]), np.array([]))
        self.assertEqual(psi, 0.0)

        psi = compute_psi(np.array([]), np.array([]))
        self.assertEqual(psi, 0.0)

    def test_degenerate_constant_values(self):
        """All-same values in both distributions should return 0."""
        psi = compute_psi(np.ones(100), np.ones(100))
        self.assertEqual(psi, 0.0)

    def test_degenerate_nan_handling(self):
        """NaN values should be ignored gracefully."""
        baseline = np.array([1, 2, np.nan, 3, 4])
        current = np.array([1, 2, 3, np.nan, 4])
        psi = compute_psi(baseline, current)
        self.assertGreaterEqual(psi, 0.0)

    def test_symmetry_approximate(self):
        """Swapping baseline and current should give similar PSI."""
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 500)
        b = rng.normal(0.5, 1, 500)
        psi_ab = compute_psi(a, b)
        psi_ba = compute_psi(b, a)
        # Not exactly symmetric, but should be in the same ballpark
        self.assertAlmostEqual(psi_ab, psi_ba, delta=0.05)


class TestPSIStatus(unittest.TestCase):
    """Tests for the psi_status mapping."""

    def test_ok_below_warn(self):
        self.assertEqual(psi_status(0.0), "OK")
        self.assertEqual(psi_status(0.05), "OK")
        self.assertEqual(psi_status(PSI_WARN_THRESHOLD - 0.001), "OK")

    def test_warn_at_threshold(self):
        self.assertEqual(psi_status(PSI_WARN_THRESHOLD), "WARN")
        self.assertEqual(psi_status(PSI_WARN_THRESHOLD + 0.02), "WARN")

    def test_alert_at_threshold(self):
        self.assertEqual(psi_status(PSI_ALERT_THRESHOLD), "ALERT")
        self.assertEqual(psi_status(PSI_ALERT_THRESHOLD + 0.1), "ALERT")

    def test_boundary_values(self):
        """Exact boundary values map correctly."""
        self.assertEqual(psi_status(0.099), "OK")
        self.assertEqual(psi_status(0.10), "WARN")
        self.assertEqual(psi_status(0.249), "WARN")
        self.assertEqual(psi_status(0.25), "ALERT")


if __name__ == "__main__":
    unittest.main()

    def test_outlier_does_not_explode_psi(self):
        """Regression: a one-sided outlier must not create empty-bin explosions.

        The old equal-width + epsilon implementation returned PSI > 1.7 for
        near-identical distributions (e.g. home_win_pct 0.5003 vs 0.4999)
        because a single outlier stretched the bin range, leaving edge bins
        empty and log(1e-10) blowing up the sum. Quantile bins plus
        add-one-half smoothing must keep this firmly in OK territory.
        """
        rng = np.random.RandomState(0)
        baseline = rng.uniform(0.45, 0.55, 500)
        current = np.append(rng.uniform(0.45, 0.55, 70), [0.95])
        psi = compute_psi(baseline, current)
        self.assertLess(psi, PSI_WARN_THRESHOLD)

    def test_true_drift_still_flagged(self):
        """A genuine distribution shift must still exceed WARN."""
        rng = np.random.RandomState(1)
        baseline = rng.normal(0, 1, 500)
        current = rng.normal(1.5, 1, 200)
        self.assertGreater(compute_psi(baseline, current), PSI_WARN_THRESHOLD)

    def test_disjoint_distributions_large_psi(self):
        """Fully disjoint distributions should produce a large (bounded) PSI."""
        rng = np.random.RandomState(2)
        baseline = rng.normal(0, 0.5, 300)
        current = rng.normal(10, 0.5, 100)
        psi = compute_psi(baseline, current)
        self.assertGreater(psi, PSI_ALERT_THRESHOLD)
        self.assertLess(psi, 100.0)  # bounded by smoothing, not infinite


class TestPSINoiseFloor(unittest.TestCase):
    """At adjacent-window sizes (~110 vs ~150 games) raw PSI between two
    SAME-distribution samples averages ~0.07 — most of the WARN band.
    Statuses must be assigned on the noise-adjusted PSI."""

    def test_noise_floor_formula(self):
        self.assertAlmostEqual(psi_noise_floor(150, 110), 4.5 * (1/150 + 1/110))
        self.assertEqual(psi_noise_floor(0, 100), 0.0)

    def _drift(self, n_base, n_cur, psi_stub_value):
        """Run compute_feature_drift with a stubbed raw PSI and inspect status."""
        base = pd.DataFrame({"woba_30g_home": np.random.RandomState(0).normal(.30, .01, n_base)})
        cur = pd.DataFrame({"woba_30g_home": np.random.RandomState(1).normal(.30, .01, n_cur)})
        tmp = Path(tempfile.mkdtemp())
        with patch.object(explainability, "DATA_DELIVERY_DIR", tmp), patch.object(
            explainability, "compute_psi", return_value=psi_stub_value,
        ):
            df = compute_feature_drift(base, cur, "20990101")
        return df.iloc[0]

    def test_small_windows_identical_means_do_not_page(self):
        """Raw PSI 0.12 on tiny windows is inside the noise floor -> OK,
        even though the same raw value pages at large samples."""
        row = self._drift(n_base=150, n_cur=110, psi_stub_value=0.12)
        self.assertEqual(row["status"], "OK")          # floor ≈ 0.071 eats it
        big = self._drift(n_base=20000, n_cur=5000, psi_stub_value=0.12)
        self.assertEqual(big["status"], "WARN")        # floor ≈ 0.002: real signal
        self.assertAlmostEqual(float(big["psi_adjusted"]), 0.12 - 4.5*(1/20000+1/5000), places=5)

    def test_large_genuine_shift_still_alerts(self):
        row = self._drift(n_base=150, n_cur=110, psi_stub_value=0.60)
        # 0.60 - 0.071 = 0.53 > ALERT threshold
        self.assertEqual(row["status"], "ALERT")
        self.assertGreater(float(row["psi_adjusted"]), PSI_ALERT_THRESHOLD)
