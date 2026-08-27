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
from pipeline import _attach_drift_run_margins


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

    def _drift(self, n_base, n_cur, psi_stub_value, cur_mean=50.0):
        """Run compute_feature_drift with a stubbed raw PSI and inspect status."""
        base = pd.DataFrame({"elo_diff": np.random.RandomState(0).normal(50, 20, n_base)})
        cur = pd.DataFrame({"elo_diff": np.random.RandomState(1).normal(cur_mean, 20, n_cur)})
        tmp = Path(tempfile.mkdtemp())
        with patch.object(explainability, "DATA_DELIVERY_DIR", tmp), patch.object(
            explainability, "compute_psi", return_value=psi_stub_value,
        ):
            df = compute_feature_drift(base, cur, "20990101")
        return df.iloc[0]

    def test_small_windows_identical_means_do_not_page(self):
        """Raw PSI 0.12 on tiny windows is inside the noise floor -> OK,
        even though the same raw value pages at large samples."""
        row = self._drift(n_base=150, n_cur=110, psi_stub_value=0.12,
                          cur_mean=50.5)
        self.assertEqual(row["status"], "OK")          # floor ≈ 0.071 eats it
        big = self._drift(n_base=20000, n_cur=5000, psi_stub_value=0.12,
                          cur_mean=51.0)
        self.assertEqual(big["status"], "WARN")        # floor ≈ 0.002 + real shift
        self.assertAlmostEqual(float(big["psi_adjusted"]), 0.12 - 4.5*(1/20000+1/5000), places=5)

    def test_high_psi_without_mean_shift_does_not_page(self):
        """Quantization wiggle: PSI is huge but the mean did not move.
        This is the tie-cluster artifact seen on hardhit%/win_pct —
        identical distributions must stay green regardless of raw PSI."""
        row = self._drift(n_base=20000, n_cur=5000, psi_stub_value=0.45,
                          cur_mean=50.0)
        self.assertEqual(row["status"], "OK")
        self.assertFalse(bool(row["location_shift"]))

    def test_large_genuine_shift_still_alerts(self):
        row = self._drift(n_base=150, n_cur=110, psi_stub_value=0.60,
                          cur_mean=62.0)
        # Mean moved 3 pooled-SDs (≫ 2·SE even with clustering factor),
        # and 0.60 - 0.071 = 0.53 > ALERT threshold
        self.assertEqual(row["status"], "ALERT")
        self.assertGreater(float(row["psi_adjusted"]), PSI_ALERT_THRESHOLD)
        self.assertTrue(bool(row["location_shift"]))


class TestRunMarginDiffDriftRow(unittest.TestCase):
    """The shipped run_margin_diff feature must appear in the drift table
    like any other numeric feature once its values are present: finite PSI,
    means from non-NaN rows only, honest coverage counts (early warm-up
    games are NaN → imputed at training; the drift distribution must
    exclude them, never impute them)."""

    @staticmethod
    def _fixture():
        rng = np.random.RandomState(7)
        n = 400
        base = pd.DataFrame({
            "elo_diff": rng.normal(0, 1, n),
            "run_margin_diff": rng.normal(0.2, 0.5, n),
        })
        cur = pd.DataFrame({
            "elo_diff": rng.normal(0, 1, n),
            "run_margin_diff": rng.normal(0.2, 0.5, n),
        })
        cur.loc[::5, "run_margin_diff"] = np.nan   # warm-up style NaN
        base.loc[::7, "run_margin_diff"] = np.nan
        return base, cur

    def _drift_row(self, base, cur):
        tmp = Path(tempfile.mkdtemp())
        with patch.object(explainability, "DATA_DELIVERY_DIR", tmp):
            df = compute_feature_drift(base, cur, "20990102")
        rows = df[df["feature"] == "run_margin_diff"]
        self.assertEqual(len(rows), 1, "run_margin_diff row must be present")
        return rows.iloc[0]

    def test_row_present_with_finite_psi(self):
        """run_margin_diff gets a REAL drift row: finite PSI (never the
        dead 0.0/INSUFFICIENT placeholder), and a same-distribution fixture
        stays OK-sized."""
        row = self._drift_row(*self._fixture())
        self.assertTrue(np.isfinite(float(row["psi"])))
        self.assertGreaterEqual(float(row["psi"]), 0.0)
        self.assertLess(float(row["psi"]), 0.15)   # identical windows ≈ 0
        self.assertNotEqual(row["status"], "INSUFFICIENT")

    def test_means_and_counts_use_non_nan_rows(self):
        """current/baseline means and n counts come from the non-NaN rows
        only — NaN is excluded from the distribution, never imputed."""
        base, cur = self._fixture()
        row = self._drift_row(base, cur)
        exp_b = float(base["run_margin_diff"].dropna().mean())
        exp_c = float(cur["run_margin_diff"].dropna().mean())
        self.assertAlmostEqual(float(row["baseline_mean"]),
                               round(exp_b, 4), places=4)
        self.assertAlmostEqual(float(row["current_mean"]),
                               round(exp_c, 4), places=4)
        self.assertEqual(int(row["n_baseline"]),
                         int(base["run_margin_diff"].notna().sum()))
        self.assertEqual(int(row["n_current"]),
                         int(cur["run_margin_diff"].notna().sum()))


class TestDriftMarginAttachWiring(unittest.TestCase):
    """The pipeline drift step must enrich the decided frame with OOF run
    margins before slicing windows (so run_margin_diff's row appears), and
    a failed derivation must never take down drift — the frame returns
    unchanged and the margin row is omitted, never fabricated."""

    def _decided(self):
        return pd.DataFrame({
            "game_pk": [1, 2, 3, 4],
            "game_date": pd.to_datetime(
                ["2026-07-01", "2026-07-08", "2026-07-15", "2026-07-22"]),
            "home_win": [1.0, 0.0, 1.0, 0.0],
            "home_score": [3, 2, 5, 1],
            "away_score": [1, 4, 2, 6],
        })

    def test_successful_attach_adds_margin_column(self):
        decided = self._decided()

        def _fake_attach(games, splits, *a, **k):
            out = games.copy()
            out["run_margin_diff"] = [0.4, -0.2, 0.1, -0.5]
            return out, splits

        with patch("pipeline._attach_oof_run_margins",
                   side_effect=_fake_attach), \
             patch("pipeline.get_last_walk_forward_splits",
                   return_value=[{"fold_idx": 0}]):
            out = _attach_drift_run_margins(decided)
        self.assertIn("run_margin_diff", out.columns)
        self.assertEqual(list(out["run_margin_diff"]), [0.4, -0.2, 0.1, -0.5])

    def test_failed_attach_returns_frame_unchanged(self):
        decided = self._decided()
        with patch("pipeline._attach_oof_run_margins",
                   side_effect=RuntimeError("derivation failed")), \
             patch("pipeline.get_last_walk_forward_splits",
                   return_value=[{"fold_idx": 0}]):
            out = _attach_drift_run_margins(decided)
        self.assertNotIn("run_margin_diff", out.columns)
        self.assertEqual(len(out), len(decided))
        self.assertTrue((out["game_pk"] == decided["game_pk"]).all())


if __name__ == "__main__":
    unittest.main()
