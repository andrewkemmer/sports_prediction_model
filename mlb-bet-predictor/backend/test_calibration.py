"""Tests for post-hoc Platt calibration (backend/calibration.py)."""

from __future__ import annotations

import unittest

import numpy as np

from calibration import (
    MIN_OOF_FOR_FIT,
    apply_platt,
    fit_platt,
    is_identity,
)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class TestFitPlatt(unittest.TestCase):
    def test_insufficient_data_returns_none(self):
        rng = np.random.default_rng(0)
        p = np.clip(rng.normal(0.55, 0.1, MIN_OOF_FOR_FIT - 1), 0.01, 0.99)
        y = (rng.random(p.size) < p).astype(float)
        self.assertIsNone(fit_platt(y, p))

    def test_single_class_returns_none(self):
        rng = np.random.default_rng(1)
        p = np.full(MIN_OOF_FOR_FIT + 10, 0.7)
        y = np.ones(p.size)
        self.assertIsNone(fit_platt(y, p))

    def test_perfectly_calibrated_input_stays_near_identity(self):
        """p generated FROM its own label distribution should map ~identity."""
        rng = np.random.default_rng(2)
        n = 20000
        p = np.clip(rng.normal(0.55, 0.12, n), 0.02, 0.98)
        y = (rng.random(n) < p).astype(float)
        cal = fit_platt(y, p)
        self.assertIsNotNone(cal)
        out = apply_platt(p, cal)
        # Calibration error of the calibrated output must be tiny
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        for i in range(10):
            m = (out >= bins[i]) & (out < bins[i + 1])
            if m.sum():
                ece += m.mean() * abs(y[m].mean() - out[m].mean())
        self.assertLess(ece, 0.01)

    def test_overconfident_model_gets_corrected(self):
        """A model that says 70% but wins 60% should be pulled down."""
        rng = np.random.default_rng(3)
        n = 20000
        p = np.clip(rng.normal(0.70, 0.08, n), 0.05, 0.95)
        # Reality: true prob is 6/7 of claimed logit-odds → underconfident truth
        true_p = _sigmoid(np.log(p / (1 - p)) * 0.72)
        y = (rng.random(n) < true_p).astype(float)
        cal = fit_platt(y, p)
        self.assertIsNotNone(cal)
        out = apply_platt(p, cal)
        self.assertLess(out.mean(), p.mean())  # pulled toward reality
        # And better calibrated on a held-out-style check (same dist here)
        bins = np.linspace(0, 1, 11)
        ece_before, ece_after = 0.0, 0.0
        for i in range(10):
            mb = (p >= bins[i]) & (p < bins[i + 1])
            ma = (out >= bins[i]) & (out < bins[i + 1])
            if mb.sum():
                ece_before += mb.mean() * abs(y[mb].mean() - p[mb].mean())
            if ma.sum():
                ece_after += ma.mean() * abs(y[ma].mean() - out[ma].mean())
        self.assertLess(ece_after, ece_before)

    def test_degenerate_slope_falls_back_to_identity(self):
        """Inverted labels would produce slope < 0 → identity (rank-safe)."""
        rng = np.random.default_rng(4)
        n = 5000
        p = np.clip(rng.normal(0.5, 0.15, n), 0.02, 0.98)
        y = (rng.random(n) > p).astype(float)  # inverted relationship
        cal = fit_platt(y, p)
        if cal is not None:
            self.assertTrue(is_identity(cal))

    def test_nan_and_out_of_range_inputs_filtered(self):
        rng = np.random.default_rng(5)
        n = MIN_OOF_FOR_FIT + 100
        p = np.clip(rng.normal(0.55, 0.1, n), 0.01, 0.99)
        y = (rng.random(n) < p).astype(float)
        p[:10] = np.nan
        y[10:20] = np.nan
        p[20:25] = 3.0  # invalid probs dropped by the (0,1) filter
        cal = fit_platt(y, p)
        self.assertIsNotNone(cal)


class TestApplyPlatt(unittest.TestCase):
    def test_none_is_identity(self):
        p = np.array([0.2, 0.5, 0.8])
        np.testing.assert_allclose(apply_platt(p, None), p)

    def test_invalid_dict_is_identity(self):
        p = np.array([0.2, 0.5, 0.8])
        np.testing.assert_allclose(apply_platt(p, {"method": "platt"}), p)

    def test_known_map_values(self):
        cal = {"method": "platt", "a": 1.0, "b": -1.0, "n": 100}
        p = np.array([0.5, 0.7310585786300049])  # logit 0, 1
        out = apply_platt(p, cal)
        # sigmoid(logit(p) - 1)
        expected = _sigmoid(np.log(p / (1 - p)) - 1.0)
        np.testing.assert_allclose(out, expected, rtol=1e-9)

    def test_output_bounded_and_monotonic(self):
        cal = {"method": "platt", "a": 0.8, "b": 0.05, "n": 900}
        p = np.linspace(0.001, 0.999, 500)
        out = apply_platt(p, cal)
        self.assertTrue(((out >= 0) & (out <= 1)).all())
        self.assertTrue((np.diff(out) > 0).all())

    def test_extreme_probs_do_not_overflow(self):
        cal = {"method": "platt", "a": 2.0, "b": -30.0, "n": 900}
        out = apply_platt(np.array([0.999999]), cal)
        self.assertTrue(np.isfinite(out).all())


class TestIsIdentity(unittest.TestCase):
    def test_various(self):
        self.assertTrue(is_identity(None))
        self.assertTrue(is_identity({}))
        self.assertTrue(is_identity({"method": "other"}))
        self.assertTrue(is_identity({"method": "platt", "a": 1.0, "b": 0.0}))
        self.assertFalse(is_identity({"method": "platt", "a": 0.9, "b": 0.0}))
        self.assertFalse(is_identity({"method": "platt", "a": 1.0, "b": 0.2}))


if __name__ == "__main__":
    unittest.main()
