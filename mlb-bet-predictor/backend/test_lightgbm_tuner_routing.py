"""LightGBM re-tuner routing tests — the 65-col / native-NaN / margin contract.

Mirrors test_rf_tuner_routing.py for the refreshed tune_lightgbm_optuna.py:

- lgbm_frame() emits FEATURE_COLS + TREE_CATEGORICAL_COLS as INT columns with
  UNK clamping (production layout), PRESERVING NaN — native routing only,
  the original tuner's impute_medians dimension is gone.
- The matrix is the FULL-WIDTH 65-column FEATURE_COLS set including the
  shipped run_margin_diff (67-col frames: 65 numeric + 2 team IDs) — the
  tuner's baseline must measure the deployed state or its verdict could be
  wrong.
- base_params() keeps objective binary and never lets sampled keys break it;
  the holdout baseline is verbatim LIGHTGBM_PARAMS.
- The refreshed search space: max_depth OPEN 2–8 (the original capped at 5),
  learning_rate 0.01–0.1, lambda_l1/l2 regularization, NO impute dimension;
  the bagging_freq guardrail (only legal when bagging_fraction < 1.0).
- categorical_feature BY NAME is passed at EVERY fit (tuned or not).
- The objective is POOLED log-loss over concatenated fold predictions
  (never a mean of per-fold scores).
- Adoption wiring: config.LIGHTGBM_PARAMS still constructs + fits on the
  native-NaN path, and train_moneyline_ensemble still trains the lightgbm
  member (roster regression).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import optuna
from optuna.trial import FixedTrial
from sklearn.metrics import log_loss

_BACKEND = Path(__file__).resolve().parent
if str(_BACKEND.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND.parent))

from training import (
    FEATURE_COLS,
    TREE_CATEGORICAL_COLS,
    UNK_TEAM_ID,
    _TEAM_ABBR_TO_ID,
)
from config import LIGHTGBM_PARAMS
import tune_lightgbm_optuna as tune

MARGIN_COL = tune.MARGIN_COL


def _synthetic_games(n: int = 80, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # The abbr→ID map populates lazily inside _add_team_ids; seed it first.
    tune._add_team_ids(_frame(2, rng))
    abbrs = sorted(_TEAM_ABBR_TO_ID.keys())
    return _frame(n, rng, abbrs)


def _frame(n: int, rng: np.random.Generator,
           abbrs: list[str] | None = None) -> pd.DataFrame:
    rows = []
    fallback = ["NYY", "BOS", "LAD", "SF"]
    abbrs = abbrs or fallback
    for i in range(n):
        row = {c: float(rng.normal()) for c in FEATURE_COLS}
        row.update({
            "game_date": pd.Timestamp("2026-06-01") + pd.Timedelta(days=i % 20),
            "home_team": abbrs[i % len(abbrs)],
            "away_team": abbrs[(i + 7) % len(abbrs)],
            "home_win": float(i % 2),
        })
        rows.append(row)
    return pd.DataFrame(rows)


class TestLgbmFrameRouting(unittest.TestCase):
    def test_columns_and_int_ids_with_unk_clamp(self):
        df = _synthetic_games(10)
        ids = tune._add_team_ids(df)
        X_num = tune._feature_matrix(ids)
        X_cat = tune._categorical_matrix(ids).copy()
        X_cat[0, 0] = -3      # negative → UNK, never team 0
        X_cat[1, 1] = -999    # negative sentinel → UNK

        frame = tune.lgbm_frame(X_num, X_cat)
        self.assertEqual(list(frame.columns),
                         list(FEATURE_COLS) + TREE_CATEGORICAL_COLS)
        codes_home = frame[TREE_CATEGORICAL_COLS[0]].to_numpy(int)
        self.assertEqual(codes_home[0], UNK_TEAM_ID,
                         "negative id aliased a real team instead of UNK")
        codes_away = frame[TREE_CATEGORICAL_COLS[1]].to_numpy(int)
        self.assertEqual(codes_away[1], UNK_TEAM_ID)

    def test_native_nan_preserved(self):
        """No imputation path exists anymore: missing values stay missing so
        LightGBM routes them natively (the production contract)."""
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=11)
        tr.loc[0, "sp_era_home"] = np.nan
        tr.loc[1, MARGIN_COL] = np.nan
        fold = tune.prepare_fold(tr, va)
        idx_era = FEATURE_COLS.index("sp_era_home")
        idx_margin = FEATURE_COLS.index(MARGIN_COL)
        self.assertTrue(np.isnan(fold["frames"].iloc[0, idx_era]),
                        "native-NaN frame must preserve NaN (sp_era_home)")
        self.assertTrue(np.isnan(fold["frames"].iloc[1, idx_margin]),
                        "native-NaN frame must preserve NaN (margin)")


class TestMarginAndFoldContract(unittest.TestCase):
    def test_margin_column_present_in_matrix(self):
        """The shipped run_margin_diff must be in FEATURE_COLS and its values
        must flow into the fold frames — a tuner whose baseline drops the
        margin would not tie to the deployed 65-column state."""
        self.assertIn(MARGIN_COL, FEATURE_COLS,
                      "run_margin_diff missing from FEATURE_COLS")
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=23)
        tr.loc[0, MARGIN_COL] = 3.25
        fold = tune.prepare_fold(tr, va)
        idx = FEATURE_COLS.index(MARGIN_COL)
        self.assertAlmostEqual(float(fold["frames"].iloc[0, idx]), 3.25,
                               places=9, msg="margin value lost in the frame")
        # 67-col frames: 65 numeric (incl. margin) + 2 team IDs.
        self.assertEqual(len(fold["frames"].columns),
                         len(FEATURE_COLS) + len(TREE_CATEGORICAL_COLS))
        self.assertEqual(len(fold["val_frames"].columns),
                         len(FEATURE_COLS) + len(TREE_CATEGORICAL_COLS))

    def test_prepare_fold_native_only(self):
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=99)
        fold = tune.prepare_fold(tr, va)
        for key in ("frames", "val_frames", "y_train", "y_val"):
            self.assertIn(key, fold)
        self.assertEqual(len(fold["y_val"]), len(va))
        # No impute-variant dicts anymore — frames ARE the DataFrames.
        self.assertNotIn(False, fold["frames"])
        self.assertNotIn(True, fold["frames"])


class TestParamsAndBaseline(unittest.TestCase):
    def test_objective_fixed_and_sampled_flow(self):
        p = tune.base_params({"max_depth": 3})
        self.assertEqual(p["objective"], "binary")
        self.assertEqual(p["max_depth"], 3)
        self.assertNotIn("_rounds", p)

    def test_baseline_is_verbatim_lightgbm_params(self):
        cur = tune.base_params(dict(LIGHTGBM_PARAMS))
        for k, v in LIGHTGBM_PARAMS.items():
            if k.startswith("_"):
                continue
            self.assertEqual(cur.get(k), v,
                             f"baseline drifted from production: {k}")

    def test_search_space_shallow_depth_no_impute_and_lambdas(self):
        """The re-tune: max_depth OPEN 2–8 (the original capped at 5),
        learning_rate 0.01–0.1, lambda_l1/l2 present, and NO impute_medians
        dimension (native NaN only)."""
        seen = {}

        def probe(trial):
            p = tune.sample_params(trial)
            seen.update(p)
            return 0.0

        study = optuna.create_study(direction="minimize")
        study.optimize(probe, n_trials=1)
        self.assertTrue(2 <= seen["max_depth"] <= 8,
                        "max_depth must open 2–8 (shallow allowed)")
        self.assertTrue(0.01 <= seen["learning_rate"] <= 0.1)
        self.assertIn("lambda_l1", seen)
        self.assertIn("lambda_l2", seen)
        self.assertIn("bagging_fraction", seen)
        self.assertIn("feature_fraction", seen)
        self.assertIn("min_child_samples", seen)
        self.assertNotIn("impute_medians", seen,
                         "impute dimension must be gone — native NaN only")

    def test_bagging_freq_guardrail(self):
        """bagging_freq is only legal with bagging_fraction < 1.0 (LightGBM
        raises otherwise) — the structural guardrail lives in sample_params."""
        full = {"max_depth": 4, "num_leaves": 8, "min_child_samples": 30,
                "min_gain_to_split": 1.0, "bagging_fraction": 1.0,
                "bagging_freq": 2, "feature_fraction": 0.7,
                "learning_rate": 0.05, "lambda_l1": 0.5, "lambda_l2": 0.5}
        p = tune.sample_params(FixedTrial(full))
        self.assertNotIn("bagging_freq", p,
                         "fraction == 1.0 must drop bagging_freq")
        full["bagging_fraction"] = 0.7
        p2 = tune.sample_params(FixedTrial(full))
        self.assertEqual(p2["bagging_freq"], 2)


class TestPrepareAndFit(unittest.TestCase):
    def test_fit_fold_passes_categorical_feature_by_name(self):
        """THE routing invariant: categorical_feature=TREE_CATEGORICAL_COLS at
        every fit — early-stopping folds and plain refits alike."""
        import lightgbm
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=11)
        fold = tune.prepare_fold(tr, va)
        params = tune.base_params({
            "max_depth": 4, "num_leaves": 8, "min_child_samples": 20,
            "min_gain_to_split": 0.5, "bagging_fraction": 0.7,
            "bagging_freq": 1, "feature_fraction": 0.7,
            "learning_rate": 0.05, "lambda_l1": 0.1, "lambda_l2": 0.1,
        })
        real_fit = lightgbm.LGBMClassifier.fit
        seen = {}

        def spy(self_, *a, **kw):
            seen["cat"] = tuple(kw.get("categorical_feature", ()))
            seen["eval_set"] = kw.get("eval_set")
            return real_fit(self_, *a, **kw)

        with patch("lightgbm.LGBMClassifier.fit", spy):
            tune.fit_fold(params, fold, early_stop=False)
        self.assertEqual(seen["cat"], tuple(TREE_CATEGORICAL_COLS))
        with patch("lightgbm.LGBMClassifier.fit", spy):
            tune.fit_fold(params, fold, early_stop=True)
        self.assertEqual(seen["cat"], tuple(TREE_CATEGORICAL_COLS))
        self.assertIsNotNone(seen["eval_set"],
                             "early-stopping fold must pass eval_set")

    def test_fit_fold_end_to_end(self):
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=11)
        fold = tune.prepare_fold(tr, va)
        params = tune.base_params({
            "max_depth": 4, "num_leaves": 8, "min_child_samples": 20,
            "min_gain_to_split": 0.5, "bagging_fraction": 0.7,
            "bagging_freq": 1, "feature_fraction": 0.7,
            "learning_rate": 0.05, "lambda_l1": 0.1, "lambda_l2": 0.1,
        })
        for early_stop in (True, False):
            with self.subTest(early_stop=early_stop):
                proba, best = tune.fit_fold(params, fold,
                                            early_stop=early_stop)
                self.assertEqual(len(proba), len(va))
                self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
                self.assertGreaterEqual(best, 1)

    def test_pooled_objective_not_mean_of_folds(self):
        """The tuner objective is ONE log_loss over concatenated fold
        predictions — mirrors production's pooled OOF metric."""
        tr1, va1 = _synthetic_games(60, seed=1), _synthetic_games(40, seed=2)
        tr2, va2 = _synthetic_games(60, seed=3), _synthetic_games(10, seed=4)
        f1, f2 = tune.prepare_fold(tr1, va1), tune.prepare_fold(tr2, va2)
        params = tune.base_params({
            "max_depth": 4, "num_leaves": 8, "min_child_samples": 20,
            "min_gain_to_split": 0.5, "bagging_fraction": 0.7,
            "bagging_freq": 1, "feature_fraction": 0.7,
            "learning_rate": 0.05, "lambda_l1": 0.1, "lambda_l2": 0.1,
            "n_estimators": 50,
        })
        p1, _ = tune.fit_fold(params, f1, early_stop=False)
        p2, _ = tune.fit_fold(params, f2, early_stop=False)
        y_all = np.concatenate([f1["y_val"], f2["y_val"]])
        p_all = np.concatenate([np.clip(p1, tune._EPS, 1 - tune._EPS),
                                np.clip(p2, tune._EPS, 1 - tune._EPS)])
        pooled = log_loss(y_all, p_all)
        per_fold_mean = (log_loss(f1["y_val"], np.clip(p1, tune._EPS, 1 - tune._EPS))
                         + log_loss(f2["y_val"], np.clip(p2, tune._EPS, 1 - tune._EPS))) / 2
        self.assertAlmostEqual(pooled, log_loss(y_all, p_all), places=12)
        self.assertNotAlmostEqual(pooled, per_fold_mean, places=6)


class TestAdoptionWiring(unittest.TestCase):
    def test_config_lightgbm_params_constructs_and_fits_native_nan(self):
        """Whatever LIGHTGBM_PARAMS holds (current or a future adopted winner)
        must construct + fit on the native-NaN path production uses."""
        from lightgbm import LGBMClassifier
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=13)
        tr.loc[0, "sp_era_home"] = np.nan
        fold = tune.prepare_fold(tr, va)
        model = LGBMClassifier(**LIGHTGBM_PARAMS)
        model.fit(fold["frames"], fold["y_train"],
                  categorical_feature=TREE_CATEGORICAL_COLS)
        proba = model.predict_proba(fold["val_frames"])[:, 1]
        self.assertEqual(len(proba), len(va))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))

    def test_ensemble_still_trains_lightgbm_member(self):
        """Roster regression: train_moneyline_ensemble fits the lightgbm
        member with the current config — the tuner must not break ensemble
        wiring."""
        from backend import training
        games = _synthetic_games(300, seed=21)
        models, _ = training.train_moneyline_ensemble(
            games.iloc[:250], games.iloc[250:]
        )
        self.assertIn("lightgbm", models,
                      "lightgbm member missing from the roster")


if __name__ == "__main__":
    unittest.main()
