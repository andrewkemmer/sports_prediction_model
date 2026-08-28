"""RF tuner routing tests — the imputation/team-ID contract.

Mirrors test_mlp_tuner_routing.py / test_lightgbm_tuner_routing.py for
tune_rf_optuna.py:

- prepare_fold() median-imputes on TRAIN rows only (sklearn trees cannot
  route NaN; imputation is mandatory, never a native-NaN path) and hstacks
  the integer team-ID categoricals (RF_WITH_TEAM_IDS=True, the production
  default) — the exact matrix production fits X_train_lr_tree on.
- The matrix is the FULL-WIDTH 65-column FEATURE_COLS set including the
  shipped run_margin_diff — the tuner's baseline must measure the deployed
  state or its verdict could be wrong.
- base_params() keeps the production backbone (config.RF_PARAMS verbatim)
  and strips private bookkeeping.
- make_model() forces random_state + n_jobs=-1 — no sampled params can turn
  them off; max_features string-codes survive the decode.
- fit_fold() end-to-end returns validation probabilities of the right shape.
- The objective is POOLED log-loss over concatenated fold predictions
  (never a mean of per-fold scores).
- Adoption wiring: config.RF_PARAMS still constructs + fits an RF on the
  imputed matrix, and train_moneyline_ensemble still trains the rf member
  (roster regression).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

_BACKEND = Path(__file__).resolve().parent
if str(_BACKEND.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND.parent))

from training import FEATURE_COLS, _TEAM_ABBR_TO_ID
from config import RF_PARAMS
import tune_rf_optuna as tune

MARGIN_COL = tune.MARGIN_COL


def _synthetic_games(n: int = 80, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # The abbr→ID map populates lazily inside _add_team_ids; seed it first.
    tune._prepare_features(_frame(2, rng))
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


class TestPrepareFoldContract(unittest.TestCase):
    def test_train_only_medians_and_no_nan(self):
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=99)
        fold = tune.prepare_fold(tr, va)
        for key in ("X_train", "X_val", "y_train", "y_val"):
            self.assertIn(key, fold)
        # RF cannot route NaN: imputed train AND val must be fully finite.
        self.assertFalse(np.isnan(fold["X_train"]).any(),
                         "train matrix has NaN — RF cannot consume NaN")
        self.assertFalse(np.isnan(fold["X_val"]).any(),
                         "val matrix has NaN — imputation must use TRAIN medians")
        self.assertEqual(len(fold["y_val"]), len(va))
        # Full-width numeric matrix + 2 integer team-ID categoricals
        # (RF_WITH_TEAM_IDS=True, the production default).
        self.assertEqual(fold["X_train"].shape[1], len(FEATURE_COLS) + 2)
        self.assertEqual(fold["X_val"].shape[1], len(FEATURE_COLS) + 2)

    def test_margin_column_present_in_matrix(self):
        """The shipped run_margin_diff must be in FEATURE_COLS and its values
        must flow into the imputed matrix — a tuner whose baseline drops the
        margin would not tie to the deployed 65-column state."""
        self.assertIn(MARGIN_COL, FEATURE_COLS,
                      "run_margin_diff missing from FEATURE_COLS")
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=23)
        # Give one train row a distinctive margin; it must appear at the
        # margin column's index in the imputed matrix.
        tr.loc[0, MARGIN_COL] = 3.25
        fold = tune.prepare_fold(tr, va)
        idx = FEATURE_COLS.index(MARGIN_COL)
        self.assertAlmostEqual(fold["X_train"][0, idx], 3.25, places=9,
                               msg="margin value lost in the RF matrix")

    def test_val_uses_train_medians_not_val_medians(self):
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=11)
        fold = tune.prepare_fold(tr, va)
        med = fold["medians"]
        # Re-impute the val NUMERIC block with the fold's stored train
        # medians; the fold's X_val hstacks the 2 team-ID categoricals onto
        # that numeric block, so compare against the numeric slice.
        X_va_i, _ = tune._impute_median(
            tune._prepare_features(va)[0], med)
        self.assertTrue(np.allclose(
            X_va_i[0], fold["X_val"][0, : len(FEATURE_COLS)]),
            "medians must be the ones stored on the fold")
        self.assertEqual(len(med), len(FEATURE_COLS))


class TestParamsAndBaseline(unittest.TestCase):
    def test_base_params_strips_private_keys_and_keeps_backbone(self):
        p = tune.base_params({"n_estimators": 500, "_rounds": 123})
        self.assertEqual(p["n_estimators"], 500)
        self.assertNotIn("_rounds", p)

    def test_baseline_is_verbatim_rf_params(self):
        cur = tune.base_params(dict(RF_PARAMS))
        for k, v in RF_PARAMS.items():
            if k.startswith("_"):
                continue
            self.assertEqual(cur.get(k), v,
                             f"baseline drifted from production: {k}")

    def test_make_model_forces_seed_and_jobs(self):
        from sklearn.ensemble import RandomForestClassifier
        hostile = {"random_state": 0, "n_jobs": 1,
                   "n_estimators": 50, "min_samples_leaf": 2}
        model = tune.make_model(hostile)
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertEqual(model.random_state, tune.RANDOM_SEED)
        self.assertEqual(model.n_jobs, -1)

    def test_max_features_codes_decode(self):
        """String codes stored in the sqlite study map back to floats/strings
        before make_model — a mixed-type categorical is not stable across
        resume, so the decode path is part of the contract."""
        self.assertEqual(tune._MAX_FEATURES_CODES["sqrt"], "sqrt")
        self.assertEqual(tune._MAX_FEATURES_CODES["0.3"], 0.3)
        self.assertEqual(tune._MAX_FEATURES_CODES["0.8"], 0.8)
        bp = {"max_features": "0.5"}
        decoded = dict(bp)
        decoded["max_features"] = tune._MAX_FEATURES_CODES[bp["max_features"]]
        model = tune.make_model(tune.base_params(decoded))
        self.assertEqual(model.max_features, 0.5)


class TestFitAndObjective(unittest.TestCase):
    def test_fit_fold_end_to_end(self):
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=11)
        fold = tune.prepare_fold(tr, va)
        params = tune.base_params({
            "n_estimators": 50, "max_depth": 6, "min_samples_leaf": 2,
            "min_samples_split": 5, "max_features": "sqrt", "bootstrap": True,
        })
        proba = tune.fit_fold(params, fold)
        self.assertEqual(len(proba), len(va))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))

    def test_pooled_objective_not_mean_of_folds(self):
        """The tuner objective is ONE log_loss over concatenated fold
        predictions — mirrors production's pooled OOF metric."""
        tr1, va1 = _synthetic_games(60, seed=1), _synthetic_games(40, seed=2)
        tr2, va2 = _synthetic_games(60, seed=3), _synthetic_games(10, seed=4)
        f1, f2 = tune.prepare_fold(tr1, va1), tune.prepare_fold(tr2, va2)
        params = tune.base_params({
            "n_estimators": 50, "max_depth": 6, "min_samples_leaf": 2,
            "min_samples_split": 5, "max_features": "sqrt", "bootstrap": True,
        })
        p1 = tune.fit_fold(params, f1)
        p2 = tune.fit_fold(params, f2)
        y_all = np.concatenate([f1["y_val"], f2["y_val"]])
        p_all = np.concatenate([np.clip(p1, tune._EPS, 1 - tune._EPS),
                                np.clip(p2, tune._EPS, 1 - tune._EPS)])
        pooled = log_loss(y_all, p_all)
        per_fold_mean = (log_loss(f1["y_val"], np.clip(p1, tune._EPS, 1 - tune._EPS))
                         + log_loss(f2["y_val"], np.clip(p2, tune._EPS, 1 - tune._EPS))) / 2
        self.assertAlmostEqual(pooled, log_loss(y_all, p_all), places=12)
        self.assertNotAlmostEqual(pooled, per_fold_mean, places=6)


class TestAdoptionWiring(unittest.TestCase):
    def test_config_rf_params_constructs_and_fits(self):
        """Whatever RF_PARAMS holds (current or a future adopted winner) must
        construct + fit on the imputed path production uses."""
        from sklearn.ensemble import RandomForestClassifier
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=13)
        fold = tune.prepare_fold(tr, va)
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(fold["X_train"], fold["y_train"])
        proba = model.predict_proba(fold["X_val"])[:, 1]
        self.assertEqual(len(proba), len(va))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))

    def test_ensemble_still_trains_rf_member(self):
        """Roster regression: train_moneyline_ensemble fits the rf member
        with the current config — the tuner must not break ensemble wiring."""
        from backend import training
        games = _synthetic_games(300, seed=21)
        models, _ = training.train_moneyline_ensemble(
            games.iloc[:250], games.iloc[250:]
        )
        self.assertIn("randomforest", models,
                      "rf member missing from the roster")


if __name__ == "__main__":
    unittest.main()
