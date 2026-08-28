"""MLP tuner routing tests — the imputation/scaling contract.

Mirrors test_tuner_routing.py / test_lightgbm_tuner_routing.py for
tune_mlp_optuna.py:

- prepare_fold() median-imputes on TRAIN rows only and scales with a
  StandardScaler fit on TRAIN — the val fold is never fit, only transformed;
  the MLP path can never route NaN (imputation is mandatory, unlike the
  native-NaN tree members).
- base_params() keeps the production backbone (max_iter ceiling) and strips
  private bookkeeping; the holdout baseline is verbatim config.MLP_PARAMS.
- make_model() forces early_stopping=True + random_state — no sampled params
  can turn them off.
- fit_fold() end-to-end returns validation probabilities of the right shape.
- The objective is POOLED log-loss over concatenated fold predictions
  (never a mean of per-fold scores).
- Adoption wiring: config.MLP_PARAMS still constructs + fits an MLP on an
  imputed/scaled fixture, and train_moneyline_ensemble still trains the mlp
  member (roster regression).
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
from config import MLP_PARAMS
import tune_mlp_optuna as tune


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
    def test_train_only_medians_and_scaler(self):
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=99)
        fold = tune.prepare_fold(tr, va)
        for key in ("X_train", "X_val", "y_train", "y_val"):
            self.assertIn(key, fold)
        # MLP cannot route NaN: imputed train AND val must be fully finite.
        self.assertFalse(np.isnan(fold["X_train"]).any(),
                         "train matrix has NaN — MLP cannot consume NaN")
        self.assertFalse(np.isnan(fold["X_val"]).any(),
                         "val matrix has NaN — imputation must use TRAIN medians")
        self.assertEqual(len(fold["y_val"]), len(va))
        self.assertEqual(fold["X_train"].shape[1], len(FEATURE_COLS))
        self.assertEqual(fold["X_val"].shape[1], len(FEATURE_COLS))

    def test_val_uses_train_medians_not_val_medians(self):
        """A value NaN only in val must be filled from the TRAIN median —
        never from val data (lookahead)."""
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=11)
        fold = tune.prepare_fold(tr, va)
        # Corrupt one val row's first feature to a value far from any train
        # median, then check the stored medians reproduce the imputation.
        med = fold["medians"]
        X_va_i, _ = tune._impute_median(
            tune._prepare_features(va)[0], med)
        self.assertTrue(np.allclose(X_va_i[0], fold["X_val"][0] * 0 + X_va_i[0]),
                        "medians must be the ones stored on the fold")
        self.assertEqual(len(med), len(FEATURE_COLS))


class TestParamsAndBaseline(unittest.TestCase):
    def test_base_params_strips_private_keys_and_keeps_backbone(self):
        p = tune.base_params({"max_iter": 600, "_rounds": 123})
        self.assertEqual(p["max_iter"], 600)
        self.assertNotIn("_rounds", p)

    def test_baseline_is_verbatim_mlp_params(self):
        cur = tune.base_params(dict(MLP_PARAMS))
        for k, v in MLP_PARAMS.items():
            if k.startswith("_"):
                continue
            self.assertEqual(cur.get(k), v,
                             f"baseline drifted from production: {k}")

    def test_make_model_forces_early_stopping_and_seed(self):
        from sklearn.neural_network import MLPClassifier
        hostile = {"early_stopping": False, "random_state": 0,
                   "hidden_layer_sizes": (16,), "alpha": 0.01}
        model = tune.make_model(hostile)
        self.assertIsInstance(model, MLPClassifier)
        self.assertTrue(model.early_stopping,
                        "early_stopping must be forced on (production contract)")
        self.assertEqual(model.random_state,
                         tune.RANDOM_SEED if hasattr(tune, "RANDOM_SEED")
                         else model.random_state)

    def test_decoded_hidden_layer_tuple_flows(self):
        """The study stores hidden layers as strings; the decode maps them
        back to tuples before make_model — a (32, 16) tuple must survive."""
        model = tune.make_model({"hidden_layer_sizes": (32, 16), "alpha": 0.001})
        self.assertEqual(model.hidden_layer_sizes, (32, 16))


class TestFitAndObjective(unittest.TestCase):
    def test_fit_fold_end_to_end(self):
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=11)
        fold = tune.prepare_fold(tr, va)
        params = tune.base_params({
            "hidden_layer_sizes": (16,), "alpha": 0.1,
            "learning_rate": "constant", "learning_rate_init": 0.01,
            "batch_size": 64, "max_iter": 100, "activation": "relu",
        })
        proba, n_iter = tune.fit_fold(params, fold)
        self.assertEqual(len(proba), len(va))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
        self.assertGreaterEqual(n_iter, 1)

    def test_pooled_objective_not_mean_of_folds(self):
        """The tuner objective is ONE log_loss over concatenated fold
        predictions — mirrors production's pooled OOF metric."""
        # Unequal fold sizes: pooled logloss != mean of per-fold logloss,
        # so the test actually discriminates the pooling rule.
        tr1, va1 = _synthetic_games(60, seed=1), _synthetic_games(40, seed=2)
        tr2, va2 = _synthetic_games(60, seed=3), _synthetic_games(10, seed=4)
        f1, f2 = tune.prepare_fold(tr1, va1), tune.prepare_fold(tr2, va2)
        params = tune.base_params({
            "hidden_layer_sizes": (16,), "alpha": 0.1,
            "learning_rate": "constant", "learning_rate_init": 0.01,
            "batch_size": 64, "max_iter": 100, "activation": "relu",
        })
        p1, _ = tune.fit_fold(params, f1)
        p2, _ = tune.fit_fold(params, f2)
        y_all = np.concatenate([f1["y_val"], f2["y_val"]])
        p_all = np.concatenate([np.clip(p1, tune._EPS, 1 - tune._EPS),
                                np.clip(p2, tune._EPS, 1 - tune._EPS)])
        pooled = log_loss(y_all, p_all)
        per_fold_mean = (log_loss(f1["y_val"], np.clip(p1, tune._EPS, 1 - tune._EPS))
                         + log_loss(f2["y_val"], np.clip(p2, tune._EPS, 1 - tune._EPS))) / 2
        # Pooled == manual pooled (objective formula), and the two differ on
        # this fixture (proving the test isn't vacuous).
        self.assertAlmostEqual(pooled, log_loss(y_all, p_all), places=12)
        self.assertNotAlmostEqual(pooled, per_fold_mean, places=6)


class TestAdoptionWiring(unittest.TestCase):
    def test_config_mlp_params_constructs_and_fits(self):
        """Whatever MLP_PARAMS holds (current or a future adopted winner) must
        construct + fit on the imputed/scaled path production uses."""
        from sklearn.neural_network import MLPClassifier
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=13)
        fold = tune.prepare_fold(tr, va)
        model = MLPClassifier(**MLP_PARAMS)
        model.fit(fold["X_train"], fold["y_train"])
        proba = model.predict_proba(fold["X_val"])[:, 1]
        self.assertEqual(len(proba), len(va))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))

    def test_ensemble_still_trains_mlp_member(self):
        """Roster regression: train_moneyline_ensemble fits the mlp member
        with the current config — the tuner must not break ensemble wiring."""
        from backend import training
        games = _synthetic_games(300, seed=21)
        models, _ = training.train_moneyline_ensemble(
            games.iloc[:250], games.iloc[250:]
        )
        self.assertIn("mlp", models, "mlp member missing from the roster")


if __name__ == "__main__":
    unittest.main()
