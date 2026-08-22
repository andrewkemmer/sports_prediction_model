"""
Regression tests for SHAP robustness in the expanded ensemble.

The last run crashed at Step 6: RandomForest's TreeExplainer returns
(1, n_features, 2) while XGBoost/LightGBM return (1, n_features) —
averaging raw outputs of mixed shapes raised 'inhomogeneous shape'.
_shap_vector reconciles every known shape; compute_shap_per_game now
also reports attributions from the FAVORED team's perspective.
"""
import unittest

import numpy as np

from backend.explainability import _shap_vector


class TestShapVector(unittest.TestCase):
    N = 6  # feature count

    def test_2d_single_row(self):
        sv = np.arange(self.N, dtype=float).reshape(1, self.N)
        out = _shap_vector(sv, self.N)
        self.assertEqual(out.shape, (self.N,))
        np.testing.assert_array_equal(out, sv[0])

    def test_list_of_class_arrays_takes_class1(self):
        c0 = np.zeros((1, self.N))
        c1 = np.ones((1, self.N))
        out = _shap_vector([c0, c1], self.N)
        np.testing.assert_array_equal(out, np.ones(self.N))

    def test_randomforest_three_dim_takes_last_class(self):
        # (batch=1, features=N, classes=2): class-1 slice is the last axis
        raw = np.stack([np.zeros((1, self.N)), np.full((1, self.N), 2.0)], axis=-1)
        self.assertEqual(raw.shape, (1, self.N, 2))
        out = _shap_vector(raw, self.N)
        np.testing.assert_array_equal(out, np.full(self.N, 2.0))

    def test_wrong_size_returns_none_not_crash(self):
        self.assertIsNone(_shap_vector(np.zeros((1, self.N + 3)), self.N))
        self.assertIsNone(_shap_vector(np.zeros((2, 2)), self.N))

    def test_mixed_member_shapes_can_now_average(self):
        """The exact crash scenario: xgb (1,N) + rf (1,N,2) average cleanly."""
        xgb = np.arange(self.N, dtype=float).reshape(1, self.N)
        rf = np.stack([np.zeros((1, self.N)), np.ones((1, self.N)) * 2], axis=-1)
        vecs = [_shap_vector(xgb, self.N), _shap_vector(rf, self.N)]
        avg = np.mean(vecs, axis=0)
        self.assertEqual(avg.shape, (self.N,))
        np.testing.assert_allclose(avg[:3], [1.0, 1.5, 2.0])


if __name__ == "__main__":
    unittest.main()
