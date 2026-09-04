"""NFL run-engine feature drift + coverage emitters — tests for
nfl_explainability (MLB explainability mirror over the 12-pool per-side
view).

Covers: PSI computation + status ladder + noise floor; the 12-pool feature
view; adjacent-window construction (strictly-prior baseline, decided-only,
sport-adjusted anchor); drift/coverage CSV schema parity with MLB's
columns; INSUFFICIENT on small windows; coverage statuses; and byte-identical
determinism of a double emit.

Pure Python (no Streamlit, no artifact dependency) except the real-artifact
schema test which skips when the emitter CSVs aren't committed.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import nfl_explainability as X  # noqa: E402
from nfl_per_side_engine import SIDE_FEATURES  # noqa: E402

# MLB's emitter column sets (from the committed MLB artifacts) — the schema
# the frontend renderers read; NFL must match column-for-column.
MLB_DRIFT_COLS = [
    "feature", "current_mean", "baseline_mean", "psi", "psi_adjusted",
    "noise_floor", "mean_shift", "shift_se", "location_shift", "status",
    "weight_pct", "n_baseline", "n_current",
]
MLB_COVERAGE_COLS = [
    "feature", "window", "n_games", "n_nonnull", "pct_nonnull", "n_measured",
    "pct_measured", "n_default_zero", "status",
]


def _synthetic_decided(n_days: int = 400, feat: str = "elo_diff") -> pd.DataFrame:
    """Decided feature frame: daily games over ``n_days`` days with the
    12-pool columns. ``feat`` shifts by +2.0 over the final 30 days so a
    drifted feature can be detected."""
    days = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(7)
    n = len(days)
    df = pd.DataFrame({"gameday": days, "game_id": [f"g{i}" for i in range(n)]})
    for c in SIDE_FEATURES:
        base = rng.normal(0.0, 1.0, n)
        if c == feat:
            base[-30:] += 2.0
        df[c] = base
    return df


class TestPSI(unittest.TestCase):
    def test_identical_distributions_zero(self):
        a = np.random.default_rng(1).normal(0, 1, 500)
        self.assertAlmostEqual(X.compute_psi(a, a.copy()), 0.0, places=5)

    def test_shifted_distribution_positive(self):
        a = np.random.default_rng(1).normal(0, 1, 500)
        b = np.random.default_rng(1).normal(1.5, 1, 500)
        self.assertGreater(X.compute_psi(a, b), 0.0)

    def test_nan_dropped(self):
        a = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        b = np.array([1.0, np.nan, 2.0, 3.0, 5.0])
        # identical after NaN drop -> ~0
        self.assertGreaterEqual(X.compute_psi(a, b), 0.0)

    def test_constant_returns_zero(self):
        # identical constants -> 0 (both distributions degenerate the same)
        self.assertEqual(X.compute_psi(np.ones(50), np.ones(50)), 0.0)

    def test_empty_current_zero(self):
        self.assertEqual(X.compute_psi(np.ones(50), np.array([])), 0.0)


class TestStatusLadder(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(X.psi_status(0.05), "OK")
        self.assertEqual(X.psi_status(0.10), "WARN")
        self.assertEqual(X.psi_status(0.24), "WARN")
        self.assertEqual(X.psi_status(0.25), "ALERT")
        self.assertEqual(X.psi_status(0.30), "ALERT")

    def test_noise_floor_formula(self):
        # (k-1)/2 * (1/n_b + 1/n_c)  with k=10
        self.assertAlmostEqual(X.psi_noise_floor(100, 200),
                               (9 / 2.0) * (1 / 100 + 1 / 200), places=9)
        self.assertEqual(X.psi_noise_floor(0, 10), 0.0)


class TestFeatureView(unittest.TestCase):
    def test_run_engine_feature_cols_is_12_pool(self):
        self.assertEqual(X.run_engine_feature_cols(), SIDE_FEATURES)
        self.assertEqual(len(SIDE_FEATURES), 12)
        self.assertNotIn("is_home", SIDE_FEATURES)


class TestWindows(unittest.TestCase):
    def test_current_and_strictly_prior_baseline(self):
        df = _synthetic_decided(n_days=400)
        baseline, current, meta = X.build_drift_windows(df)
        self.assertIsNotNone(baseline)
        self.assertIsNotNone(current)
        # current = last 28 days of decided games, INCLUSIVE lower bound
        # (MLB's gd >= cutoff semantics: 29 calendar days at daily freq)
        self.assertEqual(meta["current_n"], 29)
        # baseline strictly prior + sized max(3*len(current), 250)
        self.assertEqual(meta["baseline_n"], 250)
        self.assertLess(pd.to_datetime(baseline["gameday"]).max(),
                        pd.to_datetime(current["gameday"]).min())
        # decided-only: the frame has no slate rows (all have gameday)
        self.assertFalse(current["gameday"].isna().any())

    def test_anchor_override(self):
        df = _synthetic_decided(n_days=100)
        anchor = pd.Timestamp("2024-03-01")
        baseline, current, meta = X.build_drift_windows(df, anchor_dt=anchor)
        self.assertIsNotNone(current)
        # cutoff = anchor - 28d = 2024-02-02; current = Feb 2..Mar 1
        # INCLUSIVE (29 days); rows AFTER the anchor are excluded (as-of)
        self.assertEqual(meta["current_n"], 29)
        self.assertEqual(meta["anchor"], "2024-03-01")
        self.assertLessEqual(pd.to_datetime(current["gameday"]).max(),
                             pd.Timestamp("2024-03-01"))

    def test_no_prior_baseline_returns_none(self):
        df = _synthetic_decided(n_days=10)
        baseline, current, meta = X.build_drift_windows(df)
        # all 10 days within 28d of the newest -> current exists but there
        # is NO strictly-prior baseline -> cannot judge drift (None/None)
        self.assertIsNone(baseline)
        self.assertIsNone(current)
        self.assertEqual(meta["current_n"], 10)
        self.assertEqual(meta["baseline_n"], 0)


class TestDriftEmitter(unittest.TestCase):
    def test_schema_parity_with_mlb(self):
        df = _synthetic_decided(n_days=400)
        baseline, current, _ = X.build_drift_windows(df)
        with tempfile.TemporaryDirectory() as td:
            out = X.compute_run_engine_feature_drift(
                baseline, current, "20260904", out_dir=Path(td))
            self.assertEqual(list(out.columns), MLB_DRIFT_COLS)
            self.assertEqual(len(out), 12)
            # weight_pct None (no run-engine blend weight artifact)
            self.assertTrue(out["weight_pct"].isna().all())

    def test_small_windows_insufficient(self):
        df = _synthetic_decided(n_days=40)
        baseline, current, _ = X.build_drift_windows(df)
        with tempfile.TemporaryDirectory() as td:
            out = X.compute_run_engine_feature_drift(
                baseline, current, "20260904", out_dir=Path(td))
            # current n < 30 -> INSUFFICIENT on every feature
            self.assertTrue((out["status"] == "INSUFFICIENT").all())

    def test_full_size_drift_detected(self):
        # 2 rows/day over 330 days: current = 58 (>= 30 so not INSUFFICIENT),
        # baseline = 250; a +3.0 shift on elo_diff in the current window must
        # produce a real PSI + a noise-adjusted ALERT status.
        days = pd.date_range("2024-01-01", periods=330, freq="D")
        rng = np.random.default_rng(7)
        rows = [{"gameday": d, "game_id": f"g{d:%Y%m%d}_{k}"}
                for d in days for k in range(2)]
        df = pd.DataFrame(rows)
        for c in SIDE_FEATURES:
            df[c] = rng.normal(0.0, 1.0, len(df))
        mask = df["gameday"] >= df["gameday"].max() - pd.Timedelta(days=28)
        df.loc[mask, "elo_diff"] += 3.0
        baseline, current, meta = X.build_drift_windows(df)
        self.assertEqual(meta["current_n"], 58)
        self.assertEqual(meta["baseline_n"], 250)
        with tempfile.TemporaryDirectory() as td:
            out = X.compute_run_engine_feature_drift(
                baseline, current, "20260904", out_dir=Path(td))
            elo = out[out["feature"] == "elo_diff"].iloc[0]
            self.assertGreater(elo["psi"], 0.0)
            self.assertEqual(elo["status"], "ALERT")

    def test_deterministic_double_emit(self):
        df = _synthetic_decided(n_days=400)
        baseline, current, _ = X.build_drift_windows(df)
        with tempfile.TemporaryDirectory() as td:
            d1 = Path(td) / "a"
            d2 = Path(td) / "b"
            X.compute_run_engine_feature_drift(
                baseline, current, "20260904", out_dir=d1)
            X.compute_run_engine_feature_drift(
                baseline, current, "20260904", out_dir=d2)
            f1 = (d1 / "run_engine_feature_drift_20260904.csv").read_bytes()
            f2 = (d2 / "run_engine_feature_drift_20260904.csv").read_bytes()
            self.assertEqual(f1, f2)


class TestCoverageEmitter(unittest.TestCase):
    def test_schema_parity_with_mlb(self):
        df = _synthetic_decided(n_days=400)
        baseline, current, _ = X.build_drift_windows(df)
        with tempfile.TemporaryDirectory() as td:
            out = X.compute_run_engine_feature_coverage(
                baseline, current, "20260904", out_dir=Path(td))
            self.assertEqual(list(out.columns), MLB_COVERAGE_COLS)
            # both windows, 12 features each
            self.assertEqual(len(out), 24)
            self.assertEqual(set(out["window"]), {"current", "baseline"})
            # n_default_zero always 0 (no default-signature feature on NFL)
            self.assertTrue((out["n_default_zero"] == 0).all())

    def test_statuses_by_measured_share(self):
        df = _synthetic_decided(n_days=400)
        df.loc[df["gameday"] >= df["gameday"].max()
               - pd.Timedelta(days=28), "altitude_home"] = np.nan
        baseline, current, _ = X.build_drift_windows(df)
        with tempfile.TemporaryDirectory() as td:
            out = X.compute_run_engine_feature_coverage(
                baseline, current, "20260904", out_dir=Path(td))
            cur = out[(out["feature"] == "altitude_home")
                      & (out["window"] == "current")].iloc[0]
            self.assertEqual(cur["status"], "STARVED")
            self.assertEqual(cur["pct_measured"], 0.0)

    def test_deterministic_double_emit(self):
        df = _synthetic_decided(n_days=400)
        baseline, current, _ = X.build_drift_windows(df)
        with tempfile.TemporaryDirectory() as td:
            d1 = Path(td) / "a"
            d2 = Path(td) / "b"
            X.compute_run_engine_feature_coverage(
                baseline, current, "20260904", out_dir=d1)
            X.compute_run_engine_feature_coverage(
                baseline, current, "20260904", out_dir=d2)
            f1 = (d1 / "run_engine_feature_coverage_20260904.csv").read_bytes()
            f2 = (d2 / "run_engine_feature_coverage_20260904.csv").read_bytes()
            self.assertEqual(f1, f2)


class TestRealArtifacts(unittest.TestCase):
    """Round-trip the committed emitter CSVs (when present)."""

    DD = BACKEND.parent / "data_delivery"

    def _latest(self, prefix: str, ext: str) -> Path | None:
        cands = sorted(self.DD.glob(f"{prefix}*{ext}"))
        return cands[-1] if cands else None

    def test_drift_csv_schema_and_features(self):
        path = self._latest("run_engine_feature_drift_", ".csv")
        if path is None:
            self.skipTest("no run-engine drift CSV committed")
        df = pd.read_csv(path)
        self.assertEqual(list(df.columns), MLB_DRIFT_COLS)
        self.assertEqual(set(df["feature"]), set(SIDE_FEATURES))

    def test_coverage_csv_schema_and_windows(self):
        path = self._latest("run_engine_feature_coverage_", ".csv")
        if path is None:
            self.skipTest("no run-engine coverage CSV committed")
        df = pd.read_csv(path)
        self.assertEqual(list(df.columns), MLB_COVERAGE_COLS)
        self.assertEqual(set(df["feature"]), set(SIDE_FEATURES))
        self.assertTrue({"current", "baseline"}.issubset(set(df["window"])))


if __name__ == "__main__":
    unittest.main()