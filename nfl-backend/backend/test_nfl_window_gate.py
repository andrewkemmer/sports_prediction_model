"""Tests for the NFL window gate (run_nfl_window_gate.py) — W2014/W2016
through the within-run incumbent gate.

Pins, per the task spec (fully-within-run revision 2026-09-02):
  - the shared ECE_TOL constant (harness == production gate, one constant);
  - the incumbent WITHIN-RUN ISOLATION property (corrupting sealed outcomes
    leaves earlier fits byte-identical — predictions are a pure function of
    the features, never the target);
  - the within-run incumbent baseline is ALWAYS present on BOTH views (no
    advisory verdict mode) and is byte-identical to the candidate's own
    arms for a production-window candidate (RANDOM_SEED determinism);
  - candidates differ ONLY in the window: same val/sealed geometry, same
    12-pool market-free features, same sealed-2025 holdout;
  - the 12-pool market-free invariant (no market/line/implied feature
    anywhere near the model input);
  - the persisted bundle is a DIAGNOSTIC cross-check only: an unusable or
    missing bundle degrades the cross-check row, never the verdict.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import nfl_moneyline as M
import run_nfl_window_gate as g


def _synth_gate_feats(seasons=None, n_games_per_week: int = 8) -> pd.DataFrame:
    """Synthetic decided frame with all 12 deployed features + target,
    spanning the given seasons (weekly cadence) — the window-ablation test's
    pattern. Train-side missingness only (2025 fully observed, like real
    sealed rows); home_win is always non-null so the hold-out stays clean."""
    from test_nfl_moneyline import _synth_fold_frame
    seasons = seasons or list(range(2016, 2026))
    feats = _synth_fold_frame(seasons=seasons,
                              n_games_per_week=n_games_per_week)
    rng = np.random.default_rng(7)
    extras = {
        "win_pct_diff": 0.5, "ewm_net_pts_diff": 0.0,
        "ewm_ypp_diff": 0.0, "pace_plays_min_diff": 0.0,
        "rest_short_diff": 0.0, "div_game": 0.0,
        "travel_miles_diff": 0.0, "altitude_home": 0.0, "prime_time": 0.0,
    }
    for c, val in extras.items():
        feats[c] = rng.normal(size=len(feats)) + val
        pre25 = feats.index[feats["season"] < 2025]
        feats.loc[pre25[::7], c] = np.nan
    return feats


def _train_mini_bundle(feats: pd.DataFrame) -> dict:
    """A REAL trained 5-member bundle on a half split of the synthetic frame
    (the TestIncumbentGate pattern) — features = the 12-pool, no Platt map
    (the harness's platt-predict path already covers the None branch)."""
    half = len(feats) // 2
    models, _ = M.train_ensemble(feats.iloc[:half], feats.iloc[half:],
                                 features=g.DEPLOYED_12)
    return {"models": models,
            "adaptive_weights": {"xgboost": 0.5, "logistic": 0.5},
            "platt": None, "features": list(g.DEPLOYED_12)}


class TestGateConstants(unittest.TestCase):
    def test_tolerances_shared_with_production_gate(self):
        """All three tolerances are the production gate's constants — the
        harness imports them from nfl_moneyline (single source of truth, no
        duplicated values), never hardcoded."""
        self.assertIs(g.ECE_TOL, M.ECE_TOL)
        self.assertIs(g.TOL_LL, M.TOL_LL)
        self.assertIs(g.TOL_AUC, M.TOL_AUC)
        self.assertEqual(g.ECE_TOL, 0.01)
        self.assertGreater(g.TOL_LL, 0.0)
        self.assertGreater(g.TOL_AUC, 0.0)

    def test_candidate_boundaries_and_geometry_constants(self):
        self.assertEqual(g.BOUNDARIES, {"W2016": 2016, "W2014": 2014})
        self.assertEqual(g.SEALED_SEASON, 2025)
        self.assertEqual(g.TRAIN_END, 2024)
        self.assertEqual(g.VAL_SEASONS, [2021, 2022, 2023, 2024])

    def test_served_12_pool_market_free(self):
        self.assertEqual(len(g.DEPLOYED_12), 12)
        self.assertEqual(g.DEPLOYED_12, [
            "elo_diff", "win_pct_diff", "rest_days_diff", "is_dome_home",
            "ewm_net_pts_diff", "ewm_ypp_diff",
            "pace_plays_min_diff", "rest_short_diff", "div_game",
            "travel_miles_diff", "altitude_home", "prime_time",
        ])
        for c in g.DEPLOYED_12:
            self.assertNotIn("market", c.lower())
            self.assertNotIn("line", c.lower())
            self.assertNotIn("implied", c.lower())

    def test_arm_features_filters_market_columns(self):
        feats = pd.DataFrame([{**{c: 0.0 for c in g.DEPLOYED_12},
                               "market_home_implied": 0.62,
                               "spread_line": 3.0, "home_win": 1}])
        self.assertEqual(g.arm_features(feats), g.DEPLOYED_12)


class TestIncumbentIsolation(unittest.TestCase):
    """The within-run isolation property: the incumbent's predictions are a
    pure function of (features, bundle). Corrupting the sealed OUTCOMES must
    leave its predictions byte-identical — only its scored metrics change."""

    @classmethod
    def setUpClass(cls):
        cls.feats = _synth_gate_feats()          # 2016-2025, 8 g/week
        cls.bundle = _train_mini_bundle(cls.feats)

    def test_incumbent_predictions_are_target_independent(self):
        from nfl_moneyline import _valid_rows
        sld = self.feats[self.feats["season"] == 2025]
        sld = sld[_valid_rows(sld, g.DEPLOYED_12)].copy()
        p_clean = g.incumbent_predict(sld, self.bundle, g.DEPLOYED_12)

        # The blend's own predict-time float noise (dominated by
        # randomforest): the same frame predicted twice differs at ~1e-16.
        # The assertion is: corrupting the OUTCOMES changes predictions no
        # more than re-predicting the identical frame does.
        p_repeat = g.incumbent_predict(sld, self.bundle, g.DEPLOYED_12)
        noise = float(np.abs(p_clean - p_repeat).max())

        corrupted = sld.copy()
        corrupted["home_win"] = 1 - corrupted["home_win"]   # flip EVERY outcome
        p_corrupt = g.incumbent_predict(corrupted, self.bundle, g.DEPLOYED_12)

        self.assertEqual(len(p_clean), len(p_corrupt))
        diff = float(np.abs(p_clean - p_corrupt).max())
        # corruption must sit INSIDE the noise floor — never exceed it by
        # more than a hair of accumulated rounding
        self.assertLessEqual(diff, max(noise * 2.0, 1e-12),
                             "incumbent predictions must not depend on the "
                             f"target (diff {diff:e} vs noise floor {noise:e})")
        self.assertTrue(np.all(np.isfinite(p_clean)))

    def test_bundle_mismatch_only_disables_diagnostic(self):
        """A bundle whose stored features can't align only degrades the
        DIAGNOSTIC cross-check row — the within-run incumbent baseline is
        unaffected and the verdict never goes advisory (no advisory mode)."""
        bad = dict(self.bundle)
        bad["features"] = list(g.DEPLOYED_12) + ["market_home_implied"]
        feats = self.feats.copy()
        res = g.run_walk_forward_gate(feats, g.DEPLOYED_12,
                                      list(range(2016, 2025)),
                                      incumbent_bundle=bad)
        v = res["verdict"]
        self.assertEqual(v["ece_mode"], "within-run incumbent (both views)")
        self.assertIn("ll_ok_pooled", v)
        self.assertIn("ece_ok_sealed", v)
        # arms always present — the within-run baseline, not the bundle
        self.assertIn("incumbent", res["sealed_2025"])
        self.assertIn("incumbent", res["pooled_preq_2021_2024"])
        self.assertIsNone(res["bundle_crosscheck"])

    def test_injected_bundle_yields_diagnostic_crosscheck(self):
        feats = self.feats.copy()
        res = g.run_walk_forward_gate(feats, g.DEPLOYED_12,
                                      list(range(2016, 2025)),
                                      incumbent_bundle=self.bundle)
        self.assertIn("incumbent", res["pooled_preq_2021_2024"])
        self.assertIn("incumbent", res["sealed_2025"])
        for key in ("logloss", "auc", "ece"):
            self.assertIn(key, res["sealed_2025"]["incumbent"])
            self.assertIn(key, res["pooled_preq_2021_2024"]["incumbent"])
        self.assertEqual(res["verdict"]["ece_mode"],
                         "within-run incumbent (both views)")
        # the injected bundle surfaces as the diagnostic cross-check row
        bc = res["bundle_crosscheck"]
        self.assertIsNotNone(bc)
        for key in ("logloss", "auc", "ece"):
            self.assertIn(key, bc["sealed"])
            self.assertIn(key, bc["drift_vs_within_run"])
        self.assertIn("diagnostic", bc["note"])
        # inherited from the PRODUCTION adopt_decision: the six tolerance
        # conditions, nothing else gates
        for k in ("adopt", "ll_ok_pooled", "auc_ok_pooled", "ece_ok_pooled",
                  "ll_ok_sealed", "auc_ok_sealed", "ece_ok_sealed", "tol"):
            self.assertIn(k, res["verdict"], k)
        self.assertEqual(res["verdict"]["tol"],
                         {"ll": M.TOL_LL, "auc": M.TOL_AUC, "ece": M.ECE_TOL})


class TestWithinRunBaseline(unittest.TestCase):
    """The baseline ALWAYS exists on BOTH views — no bundle needed, no
    advisory verdict mode; for a production-window candidate it is the
    candidate's own arms by RANDOM_SEED determinism."""

    def test_incumbent_arms_present_without_bundle(self):
        feats = _synth_gate_feats()
        res = g.run_walk_forward_gate(feats, g.DEPLOYED_12,
                                      list(range(2016, 2025)),
                                      incumbent_bundle=None,
                                      load_default_bundle=False)
        self.assertIn("incumbent", res["pooled_preq_2021_2024"])
        self.assertIn("incumbent", res["sealed_2025"])
        self.assertIsNone(res["bundle_crosscheck"])
        self.assertEqual(res["verdict"]["ece_mode"],
                         "within-run incumbent (both views)")
        for k in ("ll_ok_pooled", "auc_ok_pooled", "ece_ok_pooled",
                  "ll_ok_sealed", "auc_ok_sealed", "ece_ok_sealed", "tol"):
            self.assertIn(k, res["verdict"], k)

    def test_same_config_retrain_is_deterministic(self):
        """The byte-identity basis for the within-run pooled incumbent: two
        same-config, same-seed trainings on the SAME fold rows produce
        predictions identical to float noise (~1e-16) — so re-fitting the
        production-config arm in the candidate's own fold loop IS the
        candidate's fold model, and the pooled legs are a self-identity
        noise floor for production-window candidates."""
        feats = _synth_gate_feats()
        half = len(feats) // 2
        tr, va = feats.iloc[:half], feats.iloc[half:]
        m1, _ = M.train_ensemble(tr, va, features=g.DEPLOYED_12)
        m2, _ = M.train_ensemble(tr, va, features=g.DEPLOYED_12)
        _, mem1, w1 = M.ensemble_predict(m1, va, features=g.DEPLOYED_12)
        _, mem2, _ = M.ensemble_predict(m2, va, features=g.DEPLOYED_12)
        w = M._member_weights(list(mem1))

        def _blend(members):
            out = np.zeros(len(va))
            for name, p in members.items():
                out += w[name] * np.asarray(p, dtype=float)
            return out

        self.assertTrue(np.allclose(_blend(mem1), _blend(mem2), atol=1e-12),
                        "same-config same-seed retrain must be byte-identical")
        self.assertEqual(set(mem1), set(mem2))


class TestCandidatesDifferOnlyInWindow(unittest.TestCase):
    """W2016 vs W2014 on the same synthetic pull: the ONLY difference must be
    the training window — identical val/sealed geometry and identical
    model features (the 12-pool), sealed 2025 holdout constant."""

    def test_windows_geometry_and_features(self):
        feats = _synth_gate_feats(seasons=list(range(2013, 2026)),
                                  n_games_per_week=4)
        # The within-run baseline is bundle-independent — load_default_bundle
        # =False keeps the test hermetic (this checkout carries the seeded
        # incumbent bundle, which the harness would otherwise auto-load into
        # the diagnostic cross-check).
        res16 = g.run_walk_forward_gate(feats, g.DEPLOYED_12,
                                        list(range(2016, 2025)),
                                        load_default_bundle=False)
        res14 = g.run_walk_forward_gate(feats, g.DEPLOYED_12,
                                        list(range(2014, 2025)),
                                        load_default_bundle=False)

        g16, g14 = res16["fold_geometry"], res14["fold_geometry"]
        self.assertEqual(g16["train_seasons"], list(range(2016, 2025)))
        self.assertEqual(g14["train_seasons"], list(range(2014, 2025)))
        # val/sealed geometry IDENTICAL across windows
        self.assertEqual(g16["val_seasons"], g14["val_seasons"])
        self.assertEqual(g16["sealed_season"], g14["sealed_season"])
        self.assertEqual(g16["sealed_season"], 2025)
        self.assertEqual(g16["pooled_oof_games"], g14["pooled_oof_games"])
        self.assertGreater(g16["pooled_oof_games"], 0)
        self.assertEqual(g16["sealed_games"], g14["sealed_games"])
        # the 12-pool, market-free, identical for both candidates
        self.assertEqual(res16["_deployed"]["features"], g.DEPLOYED_12)
        self.assertEqual(res14["_deployed"]["features"], g.DEPLOYED_12)
        self.assertEqual(res16["_deployed"]["features"],
                         res14["_deployed"]["features"])
        for c in res16["_deployed"]["features"]:
            self.assertNotIn("market", c.lower())
        # fully within-run verdict without any bundle — no advisory mode
        self.assertEqual(res16["verdict"]["ece_mode"],
                         "within-run incumbent (both views)")
        self.assertIsNone(res16["bundle_crosscheck"])
        self.assertIsNone(res14["bundle_crosscheck"])


if __name__ == "__main__":
    unittest.main()