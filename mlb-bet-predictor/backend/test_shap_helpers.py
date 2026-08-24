"""
Regression tests for SHAP robustness in the expanded ensemble.

The last run crashed at Step 6: RandomForest's TreeExplainer returns
(1, n_features, 2) while XGBoost/LightGBM return (1, n_features) —
averaging raw outputs of mixed shapes raised 'inhomogeneous shape'.
_shap_vector reconciles every known shape; compute_shap_per_game now
also reports attributions from the FAVORED team's perspective.
"""
import math
import unittest

import numpy as np
import pandas as pd

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


class TestNativeXgbContribs(unittest.TestCase):
    """Primary XGBoost attribution path must be the booster's OWN TreeSHAP.

    Production evidence: shap's Python-side XGBoost parser walked categorical
    splits as numeric code thresholds while native predict uses category
    set-membership — under one Colab-resolved xgboost/shap pair that diverged
    to a 2.63e-03 additivity violation (LightGBM: 1.78e-15). pred_contribs is
    computed by xgboost itself, so Σφ + bias == margin exactly regardless of
    library pairing.
    """

    def setUp(self):
        try:
            import xgboost as xgb_lib  # noqa: F401
            import shap  # noqa: F401
        except ImportError:
            self.skipTest("xgboost/shap not installed")
        from backend.explainability import (
            _add_team_ids,
            _ensure_shap_xgb_compat,
            FEATURE_COLS,
            TREE_CATEGORICAL_COLS,
            _tree_dataframe,
        )
        _ensure_shap_xgb_compat()
        self.FEATURE_COLS = FEATURE_COLS
        self.TREE_CATEGORICAL_COLS = TREE_CATEGORICAL_COLS
        self._tree_dataframe = _tree_dataframe
        self._add_team_ids = _add_team_ids

    def _tiny_model(self):
        import xgboost as xgb_lib
        rng = np.random.default_rng(3)
        n = 300
        num = rng.normal(size=(n, 4))
        team = rng.integers(0, 6, size=n)
        y = ((num[:, 0] > 0).astype(int) ^ (team % 2))
        cols = [f"f{i}" for i in range(4)]
        df_num = pd.DataFrame(num, columns=cols)
        ids = pd.DataFrame({"home_team_id": team, "away_team_id": (team + 1) % 6})
        frame = self._tree_dataframe(df_num, ids.to_numpy(), list(cols))
        m = xgb_lib.XGBClassifier(
            n_estimators=40, max_depth=3, enable_categorical=True,
            random_state=0,
        )
        m.fit(frame, y)
        return m, frame

    def test_native_contribs_exact_additivity_and_width(self):
        import xgboost as xgb_lib
        from backend.explainability import _native_xgb_contribs

        model, frame = self._tiny_model()
        row = frame.iloc[[3]]
        out = _native_xgb_contribs(model, row)
        self.assertIsNotNone(out)
        sv, base = out
        n_feat = frame.shape[1]
        self.assertEqual(sv.size, n_feat)

        p = float(model.predict_proba(row)[0, 1])
        target = math.log(p) - math.log(1 - p)
        self.assertLess(abs(float(sv.sum()) + base - target), 1e-5,
                        "native TreeSHAP must reconstruct the margin exactly")

    def test_native_failure_falls_back_loudly(self):
        """When the booster path is unavailable the caller must see it."""
        from backend.explainability import _native_xgb_contribs

        class _Broken:
            def get_booster(self):
                raise RuntimeError("no booster")

        with self.assertLogs("backend.explainability", level="WARNING"):
            out = _native_xgb_contribs(_Broken(), pd.DataFrame({"a": [1]}))
        self.assertIsNone(out)


class TestXgbBaseScoreShim(unittest.TestCase):
    """End-to-end guard for the shap/xgboost base_score incompatibility.

    xgboost>=2 serializes learner_model_param.base_score as a bracketed
    string inside its UBJSON dump; shap's loader crashes on it and every
    XGBoost attribution silently vanished. _ensure_shap_xgb_compat wraps
    the decoder; these tests prove the fix on a REAL booster so a future
    upgrade that breaks the pairing fails here instead of in production.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import xgboost  # noqa: F401
            import shap  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("xgboost/shap not installed")
        rng = np.random.RandomState(0)
        cls.X = rng.normal(size=(120, 4))
        y = (cls.X[:, 0] + cls.X[:, 1] > 0).astype(int)
        cls.model = xgboost.XGBClassifier(
            n_estimators=8, max_depth=2, eval_metric="logloss")
        cls.model.fit(cls.X, y)

    def test_shim_is_idempotent_and_marker_set(self):
        from backend.explainability import _ensure_shap_xgb_compat
        import shap.explainers._tree as st
        _ensure_shap_xgb_compat()
        _ensure_shap_xgb_compat()  # second call must be a no-op, not double-wrap
        self.assertTrue(getattr(st, "_mlb_base_score_shim", False))
        # The wrapper must wrap the ORIGINAL decoder exactly once.
        self.assertEqual(st.decode_ubjson_buffer.__name__, "_decode_fixed")

    def test_real_booster_produces_nonempty_attributions(self):
        import shap
        from backend.explainability import _ensure_shap_xgb_compat
        _ensure_shap_xgb_compat()
        ex = shap.TreeExplainer(self.model)  # crashed pre-shim on xgboost>=2
        sv = np.asarray(ex.shap_values(self.X[:5]), dtype=float)
        self.assertEqual(sv.shape[-1], 4)
        self.assertTrue(np.isfinite(sv).all())
        self.assertGreater(float(np.abs(sv).sum()), 0.0,
                           "attributions all zero — shim not effective")

    def test_additivity_reconstructs_model_logodds(self):
        """Σφ + base must equal the booster's own log-odds (end-to-end proof)."""
        import math
        import shap
        from backend.explainability import _ensure_shap_xgb_compat
        _ensure_shap_xgb_compat()
        ex = shap.TreeExplainer(self.model)
        row = self.X[:1]
        vec = _shap_vector(ex.shap_values(row), 4)
        self.assertIsNotNone(vec)
        ev = float(np.ravel(ex.expected_value)[-1])
        p = float(self.model.predict_proba(row)[0, 1])
        p = min(max(p, 1e-12), 1 - 1e-12)
        target = math.log(p) - math.log(1 - p)
        self.assertAlmostEqual(float(vec.sum()) + ev, target, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
