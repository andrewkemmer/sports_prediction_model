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

import numpy as np

from backend.explainability import compute_psi, psi_status
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
