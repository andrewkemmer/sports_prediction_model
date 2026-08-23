"""
Regression tests for favored-team calibration bucketing.

calibration_buckets() takes the FAVORED team's perspective: each game
contributes ONE observation at max(p_home, p_away) ∈ [0.5, 1], labeled by
whether the favorite won. Information-equivalent to the old home-side view —
(p, y) and (1 − p, 1 − y) are exact complements — but buckets read the way a
bettor sees them ("a team at 70% wins 70% of the time").
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training import calibration_buckets


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


class TestCalibrationArtifactOnPreGameRuns(unittest.TestCase):
    """calibration_YYYYMMDD.json must be written even when today's slate has
    no finals (pre-game-only runs).  Phase 6 prunes stale calibration files,
    so skipping the write leaves the Calibration dashboard empty."""

    def test_oof_only_artifact_written_without_day_finals(self):
        import pipeline

        oof = pd.DataFrame({
            "game_date": ["2026-08-20", "2026-08-20", "2026-08-21"],
            "home_win": [1.0, 0.0, 1.0],
            "home_win_prob_model": [0.62, 0.41, 0.58],
        })
        metrics = {"auc": 0.5, "brier": 0.25, "logloss": 0.69, "ece": 0.0}
        with tempfile.TemporaryDirectory() as td:
            orig = pipeline.DATA_DELIVERY_DIR
            pipeline.DATA_DELIVERY_DIR = Path(td)
            try:
                path = pipeline._calibration_json(
                    metrics, np.array([]), np.array([]),
                    "20260823", 15, oof=oof,
                )
                self.assertTrue(path.exists())
                data = json.loads(Path(path).read_text())
                # Buckets came from the OOF pairs, not the (empty) finals.
                self.assertTrue(data["calibration_buckets"])
                self.assertEqual(len(data["daily"]), 2)
            finally:
                pipeline.DATA_DELIVERY_DIR = orig

    def test_no_fabricated_labels_when_oof_missing(self):
        """No OOF + no finals → no buckets can be computed; artifact must not
        silently invent zeros."""
        import pipeline

        with tempfile.TemporaryDirectory() as td:
            orig = pipeline.DATA_DELIVERY_DIR
            pipeline.DATA_DELIVERY_DIR = Path(td)
            try:
                path = pipeline._calibration_json(
                    {"auc": 0.5, "brier": 0.25, "logloss": 0.69, "ece": 0.0},
                    np.array([]), np.array([]), "20260823", 15, oof=None,
                )
                data = json.loads(Path(path).read_text()) if path.exists() else None
            finally:
                pipeline.DATA_DELIVERY_DIR = orig
        if data is not None:
            self.assertEqual(data["calibration_buckets"], [])


class TestModelHistoryDedup(unittest.TestCase):
    """update_model_history must keep one row per day — reruns replace, not
    append — so the Version History table can't fill with duplicate rows."""

    def _run(self, metrics, tmp):
        import training
        orig = training.DATA_DELIVERY_DIR
        training.DATA_DELIVERY_DIR = Path(tmp)
        try:
            training.update_model_history(metrics, "vTest", notes="n1")
            training.update_model_history({"auc": 0.6, "brier": 0.2,
                                           "logloss": 0.68, "ece": 0.05},
                                          "vTest", notes="n2")
            return json.loads((Path(tmp) / "model_history.json").read_text())
        finally:
            training.DATA_DELIVERY_DIR = orig

    def test_same_day_rerun_replaces_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            hist = self._run({"auc": 0.5, "brier": 0.25, "logloss": 0.69,
                              "ece": 0.0}, tmp)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["notes"], "n2")
        self.assertAlmostEqual(hist[0]["auc"], 0.6)

    def test_other_days_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            import training
            orig = training.DATA_DELIVERY_DIR
            training.DATA_DELIVERY_DIR = Path(tmp)
            try:
                (Path(tmp) / "model_history.json").write_text(json.dumps([
                    {"version": "vOld", "date": "2026-08-20", "auc": 0.51,
                     "brier": 0.25, "logloss": 0.69, "ece": 0.01, "notes": ""},
                ]))
                training.update_model_history({"auc": 0.55, "brier": 0.24,
                                               "logloss": 0.685, "ece": 0.03},
                                              "vNew")
                hist = json.loads(
                    (Path(tmp) / "model_history.json").read_text())
            finally:
                training.DATA_DELIVERY_DIR = orig
        self.assertEqual(len(hist), 2)
        self.assertEqual([r["date"] for r in hist],
                         sorted(r["date"] for r in hist))
