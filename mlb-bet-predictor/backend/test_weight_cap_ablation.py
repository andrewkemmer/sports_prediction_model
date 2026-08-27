"""Unit tests for the adaptive-weight CAP ablation machinery.

Locks in the three measurement legs the harness answers:
(1) raw_softmax_weights — the UNCONSTRAINED softmax preference (pre
    floor/cap projection), including that it reproduces
    training.compute_adaptive_weights exactly when projected at the
    production cap;
(2) project_weights — the floor/cap projection itself: the 45% cap
    binds on a dominant member (and would NOT bind at cap=1.0), the
    floor keeps every member alive, and every projection sums to 1.0;
(3) prequential_weight_path — per-fold weights earned only from folds
    <= k, with any_binding/binding clip amounts computed against the
    production cap.

All synthetic — no training, no data files.
"""
import unittest

import numpy as np

from backend import run_weight_cap_ablation as rwca
from backend import training


def _probs(rng: np.random.RandomState, y: np.ndarray, sep: float) -> np.ndarray:
    """Synthetic member probs via a logistic process; sep controls AUC.

    z = sep * (2y - 1) + N(0, 1): larger sep -> higher AUC, strictly
    monotone in expectation (sep 3.5 ~ AUC 0.99, sep 1.0 ~ AUC 0.76),
    so a ladder of seps gives a strict AUC ordering across members.
    """
    z = sep * (2.0 * y - 1.0) + rng.normal(0.0, 1.0, len(y))
    return 1.0 / (1.0 + np.exp(-z))


class TestRawSoftmaxWeights(unittest.TestCase):
    def test_best_member_gets_highest_weight_and_sums_to_one(self):
        # compute_metrics rounds AUC to 4dp, so the ladder must stay in the
        # non-saturated range (seps that all round to AUC 1.0 tie):
        # sep 2.2/1.8/1.4/1.0/0.6 -> AUC 0.9994/0.9948/0.9762/0.9205/0.7938.
        rng = np.random.RandomState(1)
        n = 1000
        y = (rng.rand(n) > 0.5).astype(float)
        members = {
            "xgboost": _probs(rng, y, 2.2),
            "lightgbm": _probs(rng, y, 1.8),
            "logistic": _probs(rng, y, 1.4),
            "randomforest": _probs(rng, y, 1.0),
            "mlp": _probs(rng, y, 0.6),
        }
        w = rwca.raw_softmax_weights(members, y)
        self.assertEqual(set(w), set(members))
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
        # Strict preference ordering by AUC at the production temperature.
        self.assertGreater(w["xgboost"], w["lightgbm"])
        self.assertGreater(w["lightgbm"], w["logistic"])
        self.assertGreater(w["logistic"], w["randomforest"])
        self.assertGreater(w["randomforest"], w["mlp"])

    def test_dominant_member_raw_preference_exceeds_cap(self):
        # A member clearly better than the rest has an unconstrained
        # softmax weight far above 0.45 (the question the ablation asks:
        # is xgboost trying to reach 48% or 85%?).
        rng = np.random.RandomState(2)
        n = 400
        y = (rng.rand(n) > 0.5).astype(float)
        members = {
            "xgboost": _probs(rng, y, 4.0),
            "lightgbm": _probs(rng, y, 1.0),
            "logistic": _probs(rng, y, 1.0),
            "randomforest": _probs(rng, y, 1.0),
            "mlp": _probs(rng, y, 1.0),
        }
        w = rwca.raw_softmax_weights(members, y)
        self.assertGreater(w["xgboost"], 0.45)

    def test_missing_and_empty_members_skipped(self):
        rng = np.random.RandomState(3)
        n = 200
        y = (rng.rand(n) > 0.5).astype(float)
        members = {
            "xgboost": _probs(rng, y, 3.0),
            "lightgbm": [],                      # empty -> skipped
            "logistic": _probs(rng, y, 3.0),
            "randomforest": np.asarray(_probs(rng, y, 3.0)),  # ndarray ok
        }
        w = rwca.raw_softmax_weights(members, y)
        self.assertNotIn("lightgbm", w)
        self.assertEqual(set(w), {"xgboost", "logistic", "randomforest"})
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)

    def test_reproduces_production_projection(self):
        # The replicated machinery must equal the shipped
        # compute_adaptive_weights once the production cap is applied —
        # otherwise the harness measures a different blend than production.
        rng = np.random.RandomState(4)
        n = 500
        y = (rng.rand(n) > 0.5).astype(float)
        members = {
            "xgboost": _probs(rng, y, 3.5),
            "lightgbm": _probs(rng, y, 3.2),
            "logistic": _probs(rng, y, 2.4),
            "randomforest": _probs(rng, y, 2.0),
            "mlp": _probs(rng, y, 1.6),
        }
        raw = rwca.raw_softmax_weights(members, y)
        projected = rwca.project_weights(raw, rwca.CAP_PRODUCTION)
        prod = training.compute_adaptive_weights(
            {k: v.tolist() for k, v in members.items()}, y)
        self.assertEqual(projected, prod)


class TestProjectWeights(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(5)
        self.n = 400
        self.y = (self.rng.rand(self.n) > 0.5).astype(float)
        self.members = {
            "xgboost": _probs(self.rng, self.y, 4.0),
            "lightgbm": _probs(self.rng, self.y, 1.0),
            "logistic": _probs(self.rng, self.y, 1.0),
            "randomforest": _probs(self.rng, self.y, 1.0),
            "mlp": _probs(self.rng, self.y, 1.0),
        }
        self.raw = rwca.raw_softmax_weights(self.members, self.y)

    def test_cap_binds_at_production_cap(self):
        capped = rwca.project_weights(self.raw, rwca.CAP_PRODUCTION)
        eff_cap = max(rwca.CAP_PRODUCTION, 1.02 / len(self.raw))
        self.assertLessEqual(max(capped.values()), eff_cap + 1e-4)
        self.assertAlmostEqual(sum(capped.values()), 1.0, places=4)

    def test_uncapped_allows_above_cap(self):
        uncapped = rwca.project_weights(self.raw, rwca.CAP_REMOVED)
        self.assertGreater(max(uncapped.values()), rwca.CAP_PRODUCTION)
        self.assertAlmostEqual(sum(uncapped.values()), 1.0, places=4)

    def test_floor_keeps_every_member_alive(self):
        for cap in (rwca.CAP_PRODUCTION, rwca.CAP_REMOVED):
            w = rwca.project_weights(self.raw, cap)
            for v in w.values():
                self.assertGreaterEqual(v, training.ADAPTIVE_WEIGHT_FLOOR - 1e-4)

    def test_near_equal_members_do_not_bind(self):
        # With five members within a hair of each other, raw weights sit
        # near 0.2 — the 0.45 cap is slack and the projection is identity.
        y = (self.rng.rand(300) > 0.5).astype(float)
        members = {f"m{i}": _probs(self.rng, y, 2.0) for i in range(5)}
        raw = rwca.raw_softmax_weights(members, y)
        self.assertLess(max(raw.values()), rwca.CAP_PRODUCTION)
        projected = rwca.project_weights(raw, rwca.CAP_PRODUCTION)
        for name in members:
            self.assertAlmostEqual(projected[name], raw[name], places=3)


class TestPrequentialWeightPath(unittest.TestCase):
    def _fold_recs(self, n_folds: int = 5, games_per_fold: int = 60):
        rng = np.random.RandomState(6)
        names = ["xgboost", "lightgbm", "logistic", "randomforest", "mlp"]
        recs = []
        for k in range(n_folds):
            y = (rng.rand(games_per_fold) > 0.5).astype(float)
            members = {n: _probs(rng, y, 4.0 if n == "xgboost" else 1.0)
                       for n in names}
            recs.append({
                "fold_idx": k,
                "y": y,
                "members": {n: np.asarray(p, dtype=float)
                            for n, p in members.items()},
            })
        return recs, names

    def test_weights_earned_prequentially(self):
        recs, names = self._fold_recs()
        path = rwca.prequential_weight_path(recs, names)
        self.assertEqual(len(path), len(recs))
        # n_games grows monotonically — fold k sees only folds <= k.
        ns = [r["n_games"] for r in path]
        self.assertEqual(ns, sorted(ns))
        self.assertEqual(ns[0], 60)
        self.assertEqual(ns[-1], 60 * len(recs))

    def test_cap_binding_detected_on_dominant_member(self):
        recs, names = self._fold_recs(n_folds=6, games_per_fold=80)
        path = rwca.prequential_weight_path(recs, names)
        binding_folds = [r for r in path if r["any_binding"]]
        self.assertTrue(binding_folds, "dominant member should bind the cap")
        # The binding amount is uncapped - capped for the bound member, and
        # the bound member is xgboost in every binding fold.
        for r in binding_folds:
            self.assertIn("xgboost", r["binding"])
            self.assertGreater(r["binding"]["xgboost"], 0.0)
        # Max clip across the path is positive and recorded.
        max_clip = max((max(r["binding"].values()) for r in binding_folds),
                       default=0.0)
        self.assertGreater(max_clip, 0.0)

    def test_weights_sum_to_one_at_every_fold(self):
        recs, names = self._fold_recs(n_folds=4)
        path = rwca.prequential_weight_path(recs, names)
        for r in path:
            self.assertAlmostEqual(sum(r["capped"].values()), 1.0, places=3)
            self.assertAlmostEqual(sum(r["uncapped"].values()), 1.0, places=3)
            self.assertAlmostEqual(sum(r["raw"].values()), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
