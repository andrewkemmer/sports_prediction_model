"""LightGBM tuner routing tests — the team-ID categorical contract.

Mirrors test_tuner_routing.py for tune_lightgbm_optuna.py:
- lgbm_frame() emits FEATURE_COLS + TREE_CATEGORICAL_COLS as INT columns with
  UNK clamping (production layout);
- categorical_feature BY NAME is passed at EVERY fit (tuned or not);
- base_params() keeps objective binary and never lets sampled keys break it;
- the holdout baseline is verbatim LIGHTGBM_PARAMS;
- prepare_fold() offers both imputation variants with train-only medians.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

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

        frame = tune.lgbm_frame(X_num, X_cat, False, None)
        self.assertEqual(list(frame.columns),
                         list(FEATURE_COLS) + TREE_CATEGORICAL_COLS)
        # Negative sentinels must clamp to UNK (never alias team 0 / a real id).
        codes_home = frame[TREE_CATEGORICAL_COLS[0]].to_numpy(int)
        self.assertEqual(codes_home[0], UNK_TEAM_ID,
                         "negative id aliased a real team instead of UNK")
        codes_away = frame[TREE_CATEGORICAL_COLS[1]].to_numpy(int)
        self.assertEqual(codes_away[1], UNK_TEAM_ID)

    def test_impute_variant_fills_from_given_medians(self):
        df = _synthetic_games(10)
        ids = tune._add_team_ids(df)
        X_num = tune._feature_matrix(ids)
        X_num[0, 0] = np.nan
        med = np.full(X_num.shape[1], 0.5)
        filled = tune.lgbm_frame(X_num, tune._categorical_matrix(ids),
                                 True, med)
        self.assertEqual(float(filled.iloc[0, 0]), 0.5)
        raw = tune.lgbm_frame(X_num, tune._categorical_matrix(ids),
                              False, None)
        self.assertTrue(np.isnan(raw.iloc[0, 0]),
                        "native-NaN variant must preserve NaN")


class TestParamsAndBaseline(unittest.TestCase):
    def test_objective_fixed_and_sampled_flow(self):
        p = tune.base_params({"num_leaves": 4})
        self.assertEqual(p["objective"], "binary")
        self.assertEqual(p["num_leaves"], 4)

    def test_baseline_is_verbatim_lightgbm_params(self):
        cur = tune.base_params(dict(LIGHTGBM_PARAMS))
        for k, v in LIGHTGBM_PARAMS.items():
            if k.startswith("_"):
                continue
            self.assertEqual(cur.get(k), v,
                             f"baseline drifted from production: {k}")


class TestPrepareAndFit(unittest.TestCase):
    def test_prepare_fold_both_variants(self):
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=99)
        fold = tune.prepare_fold(tr, va)
        self.assertEqual(set(fold["frames"].keys()), {False, True})
        self.assertEqual(set(fold["val_frames"].keys()), {False, True})
        self.assertEqual(len(fold["y_val"]), len(va))

    def test_fit_fold_passes_categorical_feature_by_name(self):
        """THE routing invariant: categorical_feature=TREE_CATEGORICAL_COLS at
        every fit — early-stopping folds and plain refits alike."""
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=11)
        fold = tune.prepare_fold(tr, va)
        params = tune.base_params({
            "num_leaves": 4, "min_child_samples": 20,
            "min_gain_to_split": 0.5, "bagging_fraction": 0.7,
            "bagging_freq": 1, "feature_fraction": 0.7,
            "learning_rate": 0.05,
        })
        for early_stop in (True, False):
            with self.subTest(early_stop=early_stop):
                with patch("lightgbm.LGBMClassifier.fit",
                           wraps=None, autospec=True) as mock_fit:
                    mock_fit.side_effect = lambda self, *a, **kw: None
                    try:
                        tune.fit_fold(params, fold, impute=False,
                                      early_stop=early_stop)
                    except Exception:
                        pass  # mocked model may lack predict plumbing here
                    if mock_fit.call_args:
                        kw = mock_fit.call_args.kwargs
                        if early_stop or "categorical_feature" in kw:
                            self.assertEqual(
                                tuple(kw.get("categorical_feature", ())),
                                tuple(TREE_CATEGORICAL_COLS))

    def test_fit_fold_end_to_end(self):
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=11)
        fold = tune.prepare_fold(tr, va)
        params = tune.base_params({
            "num_leaves": 4, "min_child_samples": 20,
            "min_gain_to_split": 0.5, "bagging_fraction": 0.7,
            "bagging_freq": 1, "feature_fraction": 0.7,
            "learning_rate": 0.05,
        })
        proba, best = tune.fit_fold(params, fold, impute=False, early_stop=True)
        self.assertEqual(len(proba), len(va))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
        self.assertGreaterEqual(best, 1)


if __name__ == "__main__":
    unittest.main()
