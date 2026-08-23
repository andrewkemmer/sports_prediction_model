"""
Regression tests for per-feature model weights in the drift report.

feature_importance_weights() answers "what fraction of the final blended
ensemble rides on this feature?" — member importances are normalized
internally, then averaged with each member's configured ENSEMBLE_WEIGHTS
share. The result must sum to exactly 100% across features and ride
through compute_feature_drift as the weight_pct column.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend import explainability, pipeline, training


class _FakeTree:
    def __init__(self, imp):
        self.feature_importances_ = np.asarray(imp, dtype=float)


class _FakeLogistic:
    def __init__(self, coef):
        self.coef_ = np.asarray([coef], dtype=float)


class TestFeatureImportanceWeights(unittest.TestCase):
    def setUp(self):
        # Isolate from adaptive weights left by other tests
        training.set_adaptive_weights(None)

    def test_blend_weighted_and_sums_to_100(self):
        n = len(training.FEATURE_COLS)
        imp = np.zeros(n); imp[0] = 1.0            # xgb rides feature 0 only
        coef = np.zeros(n); coef[1] = 2.0          # logistic rides feature 1 only
        models = {"xgboost": _FakeTree(imp), "logistic": _FakeLogistic(coef)}

        w = training.feature_importance_weights(models)
        self.assertIsNotNone(w)
        self.assertAlmostEqual(sum(w.values()), 100.0, places=6)
        # Member shares: xgb .25 / logistic .30 of the .55 configured total
        self.assertAlmostEqual(w[training.FEATURE_COLS[0]], 100 * (0.25 / 0.55), places=2)
        self.assertAlmostEqual(w[training.FEATURE_COLS[1]], 100 * (0.30 / 0.55), places=2)

    def test_no_usable_member_returns_none(self):
        self.assertIsNone(training.feature_importance_weights({}))
        self.assertIsNone(training.feature_importance_weights({"xgboost": object()}))

    def test_weight_pct_flows_through_drift_csv(self):
        n = len(training.FEATURE_COLS)
        base = pd.DataFrame({"elo_diff": np.random.RandomState(0).normal(50, 20, 300)})
        cur = pd.DataFrame({"elo_diff": np.random.RandomState(1).normal(50, 20, 100)})
        weights = {"elo_diff": 12.5}
        tmp = Path(tempfile.mkdtemp())
        with self._patch_dir(tmp):
            df = explainability.compute_feature_drift(
                base, cur, "20990101", model_weights=weights)
        self.assertIn("weight_pct", df.columns)
        row = df[df["feature"] == "elo_diff"].iloc[0]
        self.assertEqual(float(row["weight_pct"]), 12.5)
        # Features absent from the weights dict get explicit 0.0, not NaN
        other = df[df["feature"] != "elo_diff"]["weight_pct"].fillna(0.0)
        self.assertTrue((other == 0.0).all())

    def _patch_dir(self, tmp):
        import unittest.mock as mock
        return mock.patch.object(explainability, "DATA_DELIVERY_DIR", tmp)


if __name__ == "__main__":
    unittest.main()
