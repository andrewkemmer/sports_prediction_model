"""
Unit tests for ensemble weighting / imputation / candidate reporting.

Verifies that:
- Member weights renormalize to exactly 1.0 over trained members
- Untrained candidates report 0% weight (roster always complete)
- Median imputation fits on train and applies consistently at predict time
- walk_forward_evaluate publishes a roster via last_ensemble_info()
"""
import importlib
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backend import calibration as backend_calibration
from backend import training as backend_training
from backend.calibration import apply_platt, is_identity

# training.py imports the TOP-LEVEL ``calibration`` module (backend dir on
# sys.path), which is a distinct module object from ``backend.calibration``.
# Guards read constants from their own module, so both copies must be patched.
calibration_toplevel = importlib.import_module("calibration")
from backend.training import (
    _impute_median,
    _member_weights,
    compute_metrics,
    get_last_calibrator,
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


class TestFeatureMatrixWidthInvariant(unittest.TestCase):
    """_feature_matrix must ALWAYS return len(FEATURE_COLS) columns.

    The old filtered-subset behavior silently produced narrower matrices when
    the source frame lacked feature columns (synthetic slates did exactly
    that), which is how SHAP attributions went quietly empty. Absence must
    be loud; width must be invariant.
    """

    def test_missing_columns_become_nan_not_dropped(self):
        from backend.training import FEATURE_COLS, _feature_matrix
        df = pd.DataFrame({FEATURE_COLS[0]: [1.0, 2.0]})
        with self.assertLogs("backend.training", level="WARNING"):
            X = _feature_matrix(df)
        self.assertEqual(X.shape, (2, len(FEATURE_COLS)))
        # Present column lands in canonical position 0.
        self.assertEqual(X[0, 0], 1.0)
        # Every absent column is all-NaN — never dropped, never zero-filled.
        self.assertTrue(np.isnan(X[:, 1:]).all())

    def test_full_frame_is_canonical_order(self):
        from backend.training import FEATURE_COLS, _feature_matrix
        shuffled = {c: [float(i)] for i, c in enumerate(reversed(FEATURE_COLS))}
        X = _feature_matrix(pd.DataFrame(shuffled))
        self.assertEqual(X.shape, (1, len(FEATURE_COLS)))
        # Column named FEATURE_COLS[-1] carries value 0 → sits at last position.
        self.assertEqual(X[0, -1], 0.0)
        self.assertEqual(X[0, 0], float(len(FEATURE_COLS) - 1))


class TestCalibratedMetricsArePrequential(unittest.TestCase):
    """Reported *_calibrated metrics must score the PREQUENTIAL column only.

    Regression: the old code composed final_calibrator on top of the already
    per-fold-corrected values — G(platt_k(raw_k)) — a double correction no
    production path applies, flattering ECE/logloss/brier (verified live:
    0.2473 reported vs 0.2715 honest on a synthetic run).
    """

    def test_reported_calibrated_equals_prequential_column(self):
        # Overconfident member: claims 0.34–0.93, actual win rates stay ≤60% —
        # so the fitted Platt map shrinks (a>1 after logit rescale) and is
        # decisively non-identity.
        probs_a = np.array([0.92, 0.88, 0.84, 0.80, 0.76, 0.62, 0.58, 0.54, 0.44, 0.34])
        y_a = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=float)
        probs_b = np.array([0.93, 0.87, 0.81, 0.77, 0.71, 0.66, 0.61, 0.57, 0.51,
                            0.47, 0.42, 0.38, 0.63, 0.72, 0.49])
        y_b = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0], dtype=float)

        hw = np.array([float(i % 2) for i in range(40)])
        hw[10:20] = y_a   # fold A validation labels
        hw[25:40] = y_b   # fold B validation labels
        games = pd.DataFrame({
            "game_date": pd.date_range("2026-06-01", periods=40, freq="D"),
            "home_win": hw,
        })
        fold_a = {
            "train_games": games.iloc[:10].copy(),
            "val_games": games.iloc[10:20].copy(),
            "fold_idx": 0,
            "val_start": games.iloc[10]["game_date"],
            "val_end": games.iloc[19]["game_date"],
        }
        fold_b = {
            "train_games": games.iloc[:25].copy(),
            "val_games": games.iloc[25:40].copy(),
            "fold_idx": 1,
            "val_start": games.iloc[25]["game_date"],
            "val_end": games.iloc[39]["game_date"],
        }

        def fake_predict(models, val):
            p = probs_a if len(val) == 10 else probs_b
            return p, {"logistic": p}, {"logistic": 1.0}

        try:
            with patch("backend.training.walk_forward_splits",
                       return_value=[fold_a, fold_b]), \
                 patch("backend.training.train_moneyline_ensemble",
                       return_value=({"logistic": object()}, {})), \
                 patch("backend.training.ensemble_predict",
                       side_effect=fake_predict), \
                 patch.object(backend_training, "MIN_OOF_FOR_FIT", 5), \
                 patch.object(backend_calibration, "MIN_OOF_FOR_FIT", 5), \
                 patch.object(calibration_toplevel, "MIN_OOF_FOR_FIT", 5):
                _, pooled, combined = walk_forward_evaluate(games, min_val_games=0)
            final_map = get_last_calibrator()
            self.assertFalse(is_identity(final_map))

            y = combined["home_win"].to_numpy(dtype=float)
            preq = combined["home_win_prob_model_calibrated"].to_numpy(dtype=float)
            honest = compute_metrics(y, preq)["brier"]
            double = compute_metrics(y, apply_platt(preq, final_map))["brier"]

            # THE contract: reported value == honest prequential estimate
            # (compute_metrics rounds, so compare at its precision).
            self.assertAlmostEqual(
                pooled["brier_calibrated"], honest, places=3,
                msg="reported calibrated metrics must equal the prequential column")
            # And they must NOT be the double-applied composition.
            self.assertGreater(abs(double - pooled["brier_calibrated"]), 1e-3,
                               "final map must not be composed onto prequential values")
        finally:
            set_adaptive_weights(None)
            from backend.training import set_calibration
            set_calibration(None)


class TestFinalModelTraining(unittest.TestCase):
    @staticmethod
    def _games(n=20):
        return pd.DataFrame({
            "game_date": pd.date_range("2026-06-01", periods=n, freq="D"),
            "home_win": [float(i % 2) for i in range(n)],
        })

    def test_final_refit_uses_all_decided_games_without_validation(self):
        games = self._games()
        fold = {
            "train_games": games.iloc[:10].copy(),
            "val_games": games.iloc[10:15].copy(),
            "fold_idx": 0,
            "val_start": games.iloc[10]["game_date"],
            "val_end": games.iloc[14]["game_date"],
        }
        calls = []

        def fake_train(train, val=None):
            calls.append((len(train), None if val is None else len(val)))
            return {"logistic": object()}, {}

        def fake_predict(models, val):
            p = np.full(len(val), 0.5)
            return p, {"logistic": p}, {"logistic": 1.0}

        try:
            with patch("backend.training.walk_forward_splits", return_value=[fold]), \
                 patch("backend.training.train_moneyline_ensemble", side_effect=fake_train), \
                 patch("backend.training.ensemble_predict", side_effect=fake_predict):
                walk_forward_evaluate(games, min_val_games=0)
        finally:
            set_adaptive_weights(None)

        self.assertEqual(calls, [(10, 5), (20, None)])

    def test_oof_run_clears_adaptive_weights_from_previous_run(self):
        games = self._games()
        fold = {
            "train_games": games.iloc[:10].copy(),
            "val_games": games.iloc[10:15].copy(),
            "fold_idx": 0,
            "val_start": games.iloc[10]["game_date"],
            "val_end": games.iloc[14]["game_date"],
        }
        observed_weights = []

        def fake_train(train, val=None):
            return {"xgboost": object(), "logistic": object()}, {}

        def fake_predict(models, val):
            weights = _member_weights(["xgboost", "logistic"])
            observed_weights.append(weights)
            p = np.full(len(val), 0.5)
            return p, {"xgboost": p, "logistic": p}, weights

        set_adaptive_weights({"xgboost": 0.95, "logistic": 0.05})
        try:
            with patch("backend.training.walk_forward_splits", return_value=[fold]), \
                 patch("backend.training.train_moneyline_ensemble", side_effect=fake_train), \
                 patch("backend.training.ensemble_predict", side_effect=fake_predict):
                walk_forward_evaluate(games, min_val_games=0)
        finally:
            set_adaptive_weights(None)

        self.assertEqual(len(observed_weights), 1)
        self.assertAlmostEqual(observed_weights[0]["xgboost"], 0.25 / 0.55)
        self.assertAlmostEqual(observed_weights[0]["logistic"], 0.30 / 0.55)


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
        })
        df = add_diff_features(raw)
        try:
            walk_forward_evaluate(
                df, retrain_cadence_days=30, min_train_days=0, min_val_games=0
            )
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
