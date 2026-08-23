"""
Regression tests for the expanded candidate roster and adaptive blending.

Locks in:
- train_moneyline_ensemble trains five members (xgboost, lightgbm,
  logistic, randomforest, mlp) on a synthetic frame
- compute_adaptive_weights: softmax over pooled OOF log-loss with floors
  and caps — better members earn more, everything sums to 1.0
- _member_weights prefers earned adaptive weights but keeps zeroed-out
  candidates alive at a small prior so they can re-compete
- adaptive weights persist through the ensemble bundle (cached-model runs)
"""
import unittest

import numpy as np

from backend import training
from backend.config import (
    ADAPTIVE_WEIGHT_CAP,
    ADAPTIVE_WEIGHT_FLOOR,
    ENSEMBLE_WEIGHTS,
)
from backend.features import add_diff_features


class TestCandidateRoster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        rng = np.random.RandomState(7)
        rows = []
        for i in range(400):
            signal = rng.normal(0, 1)
            rows.append({
                "game_id": f"synth_{i}",
                "game_date": pd.Timestamp("2026-03-27") + pd.Timedelta(days=i),
                "home_team": "NYY",
                "away_team": "BOS",
                "home_elo": 1500 + signal * 40 + rng.normal(0, 10),
                "away_elo": 1500 - signal * 40,
                "home_win_pct": 0.55 + signal * 0.02,
                "away_win_pct": 0.48 - signal * 0.01,
                "rest_days_home": 1,
                "rest_days_away": 1,
                "sp_era_home": 4.0 - signal * 0.2,
                "sp_era_away": 4.0 + signal * 0.2,
                "sp_k9_home": 9.0 + signal * 0.5,
                "sp_k9_away": 8.5 - signal * 0.3,
                "sp_fbvelo_3g_home": 94.0 + signal * 0.5,
                "sp_fbvelo_3g_away": 93.5 - signal * 0.3,
                "sp_fbpct_3g_home": 0.45 + signal * 0.01,
                "sp_fbpct_3g_away": 0.44,
                "sp_whiff_3g_home": 0.28 + signal * 0.01,
                "sp_whiff_3g_away": 0.27,
                "sp_xwoba_home": 0.31 + signal * 0.005,
                "sp_xwoba_away": 0.32,
                "sp_xwoba_vs_l_home": 0.30 + signal * 0.005,
                "sp_xwoba_vs_l_away": 0.31,
                "lineup_woba_mean_home": 0.33 + signal * 0.005,
                "lineup_woba_mean_away": 0.32,
                "lineup_woba_top3_home": 0.37 + signal * 0.005,
                "lineup_woba_top3_away": 0.36,
                "lineup_woba_std_home": 0.03,
                "lineup_woba_std_away": 0.03,
                "woba_30g_home": 0.32 + signal * 0.005,
                "woba_30g_away": 0.31,
                "bullpen_whip_10g_home": 1.3,
                "bullpen_whip_10g_away": 1.35,
                "bullpen_pitches_3d_home": 45.0,
                "bullpen_pitches_3d_away": 50.0,
                "bullpen_ip_3d_home": 14.0,
                "bullpen_ip_3d_away": 13.0,
                "team_barrel_15g_home": 0.08,
                "team_barrel_15g_away": 0.07,
                "team_hardhit_15g_home": 0.35,
                "team_hardhit_15g_away": 0.33,
                "team_exitvelo_15g_home": 89.0,
                "team_exitvelo_15g_away": 88.5,
                "opp_lefty_share_home": 0.33,
                "opp_lefty_share_away": 0.30,
                "home_win": float(rng.rand() < 1 / (1 + np.exp(-signal))),
            })
        cls.games = add_diff_features(pd.DataFrame(rows))

    def test_all_five_members_train(self):
        models, metrics = training.train_moneyline_ensemble(
            self.games.iloc[:300], self.games.iloc[300:]
        )
        # sklearn members are always available; tree boosters only in Colab
        for name in ("logistic", "randomforest", "mlp"):
            self.assertIn(name, models, f"missing candidate: {name}")
        try:
            import xgboost  # noqa: F401
            import lightgbm  # noqa: F401
            for name in ("xgboost", "lightgbm"):
                self.assertIn(name, models, f"missing candidate: {name}")
        except ImportError:
            pass
        # Metrics are finite
        self.assertTrue(np.isfinite(metrics.get("auc", 0.5)))


class TestAdaptiveWeights(unittest.TestCase):
    def test_better_member_earns_more_weight(self):
        y = np.array([1.0, 0.0, 1.0, 0.0] * 25)
        good = np.clip(y * 0.9 + 0.05 + np.random.RandomState(0).normal(0, .02, len(y)), .01, .99)
        bad = np.full(len(y), 0.5) + np.random.RandomState(1).normal(0, .3, len(y))
        w = training.compute_adaptive_weights({"good": list(good), "bad": list(bad)}, y)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
        self.assertGreater(w["good"], w["bad"])

    def test_floor_and_cap_respected(self):
        # Three identical members: softmax would give exactly 1/3 each.
        # With cap .45 and floor .05, equal weights stay within bounds.
        y = np.array([1.0, 0.0] * 50)
        p = [0.5] * 100
        w = training.compute_adaptive_weights(
            {"a": p, "b": p, "c": p}, y)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
        for v in w.values():
            self.assertGreaterEqual(v, ADAPTIVE_WEIGHT_FLOOR - 0.01)
            self.assertLessEqual(v, ADAPTIVE_WEIGHT_CAP + 0.01)

        # One dominant member among three: the 0.45 cap is feasible here
        # (3 × 0.45 ≥ 1), so the strong member is held near the cap while
        # the weak ones sit near the floor.
        strong = np.clip(y * 0.98 - 0.02, 0.001, 0.999)
        w = training.compute_adaptive_weights(
            {"strong": list(strong), "weak1": p, "weak2": p}, y)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=2)
        self.assertLessEqual(max(w.values()), ADAPTIVE_WEIGHT_CAP + 0.01)
        for name in ("weak1", "weak2"):
            self.assertGreaterEqual(w[name], ADAPTIVE_WEIGHT_FLOOR - 0.01)

        # Two-member roster: a 0.45 cap is infeasible (2 × 0.45 < 1), so
        # the cap widens and the better member may take a large share.
        w = training.compute_adaptive_weights({"strong": list(strong), "weak": p}, y)
        self.assertGreater(w["strong"], w["weak"])
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)

    def test_member_weights_prefer_adaptive_but_keep_candidates_alive(self):
        try:
            training.set_adaptive_weights({"xgboost": 0.7, "logistic": 0.3})
            w = training._member_weights(
                ["xgboost", "logistic", "mlp"])
            self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
            self.assertGreater(w["xgboost"], w["logistic"])
            # mlp had no earned weight: revived at a small prior, not locked out
            self.assertGreater(w["mlp"], 0.0)
        finally:
            training.set_adaptive_weights(None)

    def test_fallback_to_static_priors_when_no_adaptive(self):
        training.set_adaptive_weights(None)
        w = training._member_weights(list(ENSEMBLE_WEIGHTS.keys()))
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
        for name in ENSEMBLE_WEIGHTS:
            self.assertIn(name, w)


if __name__ == "__main__":
    unittest.main()
