"""
Unit tests for ensemble weighting / imputation / candidate reporting.

Verifies that:
- Member weights renormalize to exactly 1.0 over trained members
- Untrained candidates report 0% weight (roster always complete)
- Median imputation fits on train and applies consistently at predict time
- walk_forward_evaluate publishes a roster via last_ensemble_info()
"""
import unittest

import numpy as np
import pandas as pd

from backend.training import (
    _impute_median,
    _member_weights,
    last_ensemble_info,
    set_adaptive_weights,
    walk_forward_evaluate,
)


class TestMemberWeights(unittest.TestCase):
    def test_all_members_sums_to_one(self):
        w = _member_weights(["xgboost", "lightgbm", "logistic"])
        self.assertAlmostEqual(sum(w.values()), 1.0, places=9)

    def test_subset_renormalizes(self):
        """If trees fail to train, logistic takes their weight."""
        w = _member_weights(["logistic"])
        self.assertAlmostEqual(w["logistic"], 1.0, places=9)

        w = _member_weights(["xgboost", "logistic"])
        self.assertAlmostEqual(sum(w.values()), 1.0, places=9)
        self.assertGreater(w["logistic"], w["xgboost"])  # config favors logistic

    def test_unknown_member_gets_zero_before_renormalization(self):
        w = _member_weights(["mystery_model"])
        self.assertAlmostEqual(w["mystery_model"], 1.0, places=9)  # equal-split fallback


class TestImputeMedian(unittest.TestCase):
    def test_fit_and_apply(self):
        X = np.array([[1.0, np.nan], [3.0, 5.0], [np.nan, 7.0]])
        Xi, med = _impute_median(X)
        self.assertTrue(not np.isnan(Xi).any())
        self.assertEqual(med[0], 2.0)  # median of (1,3)
        # Applying stored medians reproduces the same fill
        Xj, _ = _impute_median(np.array([[10.0, np.nan]]), med)
        self.assertEqual(Xj[0, 1], med[1])

    def test_all_nan_column_falls_back_to_zero(self):
        X = np.array([[np.nan], [np.nan]])
        Xi, med = _impute_median(X)
        self.assertEqual(med[0], 0.0)
        self.assertTrue(not np.isnan(Xi).any())


class TestRosterReporting(unittest.TestCase):
    def test_walk_forward_publishes_complete_roster(self):
        """Every configured candidate appears; weights sum to exactly 1.0."""
        from backend.features import add_diff_features
        rng = np.random.RandomState(0)
        n = 60
        dates = pd.date_range("2026-06-01", periods=n, freq="D")
        raw = pd.DataFrame({
            "game_id": [f"g{i}" for i in range(n)],
            "game_date": dates,
            "home_team": ["NYY"] * n,
            "away_team": ["BOS"] * n,
            "home_win": rng.rand(n).round(),
            "home_elo": rng.normal(1500, 20, n),
            "away_elo": rng.normal(1500, 20, n),
            "home_win_pct": rng.uniform(0.4, 0.6, n),
            "away_win_pct": rng.uniform(0.4, 0.6, n),
            "rest_days_home": rng.randint(0, 4, n).astype(float),
            "rest_days_away": rng.randint(0, 4, n).astype(float),
            "sp_era_home": rng.uniform(3, 5, n),
            "sp_era_away": rng.uniform(3, 5, n),
            "sp_k9_home": rng.uniform(7, 11, n),
            "sp_k9_away": rng.uniform(7, 11, n),
            "sp_fbvelo_3g_home": rng.uniform(92, 97, n),
            "sp_fbvelo_3g_away": rng.uniform(92, 97, n),
            "sp_fbpct_3g_home": rng.uniform(0.4, 0.5, n),
            "sp_fbpct_3g_away": rng.uniform(0.4, 0.5, n),
            "sp_whiff_3g_home": rng.uniform(0.22, 0.32, n),
            "sp_whiff_3g_away": rng.uniform(0.22, 0.32, n),
            "sp_xwoba_home": rng.uniform(0.28, 0.34, n),
            "sp_xwoba_away": rng.uniform(0.28, 0.34, n),
            "sp_xwoba_vs_l_home": rng.uniform(0.28, 0.34, n),
            "sp_xwoba_vs_l_away": rng.uniform(0.28, 0.34, n),
            "lineup_woba_mean_home": rng.uniform(0.30, 0.36, n),
            "lineup_woba_mean_away": rng.uniform(0.30, 0.36, n),
            "lineup_woba_top3_home": rng.uniform(0.34, 0.40, n),
            "lineup_woba_top3_away": rng.uniform(0.34, 0.40, n),
            "lineup_woba_std_home": rng.uniform(0.02, 0.04, n),
            "lineup_woba_std_away": rng.uniform(0.02, 0.04, n),
            "woba_30g_home": rng.uniform(0.30, 0.34, n),
            "woba_30g_away": rng.uniform(0.30, 0.34, n),
            "bullpen_whip_10g_home": rng.uniform(1.1, 1.5, n),
            "bullpen_whip_10g_away": rng.uniform(1.1, 1.5, n),
            "bullpen_pitches_3d_home": rng.uniform(30, 60, n),
            "bullpen_pitches_3d_away": rng.uniform(30, 60, n),
            "bullpen_ip_3d_home": rng.uniform(10, 16, n),
            "bullpen_ip_3d_away": rng.uniform(10, 16, n),
            "team_barrel_15g_home": rng.uniform(0.06, 0.10, n),
            "team_barrel_15g_away": rng.uniform(0.06, 0.10, n),
            "team_hardhit_15g_home": rng.uniform(0.30, 0.40, n),
            "team_hardhit_15g_away": rng.uniform(0.30, 0.40, n),
            "team_exitvelo_15g_home": rng.uniform(87, 91, n),
            "team_exitvelo_15g_away": rng.uniform(87, 91, n),
            "opp_lefty_share_home": rng.uniform(0.25, 0.40, n),
            "opp_lefty_share_away": rng.uniform(0.25, 0.40, n),
        })
        df = add_diff_features(raw)
        try:
            walk_forward_evaluate(df, retrain_cadence_days=30, min_train_days=0)
        except Exception:
            self.fail("walk_forward_evaluate raised unexpectedly")
        finally:
            # Don't leak adaptive weights into other tests
            set_adaptive_weights(None)
        info = last_ensemble_info()
        names = {e["name"] for e in info}
        self.assertTrue({"xgboost", "lightgbm", "logistic"} <= names,
                        f"roster missing candidates: {names}")
        total = sum(e["weight"] for e in info)
        self.assertAlmostEqual(total, 1.0, places=9)
        zero_w = [e for e in info if e["weight"] == 0.0]
        for e in zero_w:
            self.assertIsNone(e["auc"])  # untrained candidates have no metrics


if __name__ == "__main__":
    unittest.main()
