"""Tuner routing tests — the team-ID categorical contract.

The Optuna tuner must select hyperparameters for THE MODEL WE DEPLOY. These
tests lock the details most likely to silently break:

- tree_frame() emits FEATURE_COLS + TREE_CATEGORICAL_COLS with pd.Categorical
  ID columns whose explicit category set contains every known team + UNK;
- unknown / negative IDs clamp to UNK_TEAM_ID (never alias a real team);
- base_params() always carries enable_categorical=True (not overridable);
- prepare_fold() median-imputes numerics on TRAIN rows only and produces
  ready-to-fit frames;
- fit_fold() end-to-end trains the production-style model and returns
  validation probabilities of the right shape.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

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
    _cat_known_ids,
    _cat_unk_for,
)
from config import XGBOOST_PARAMS
import tune_xgboost_optuna as tune


def _synthetic_games(n: int = 80, seed: int = 7) -> pd.DataFrame:
    """Decided games frame carrying every FEATURE_COL (zeros + noise) so
    _feature_matrix never has to warn about absent columns."""
    rng = np.random.default_rng(seed)
    # The abbr→ID map populates lazily inside _add_team_ids; seed it first.
    tune._add_team_ids(_synthetic_frame(2, rng))
    abbrs = sorted(_TEAM_ABBR_TO_ID.keys())
    return _synthetic_frame(n, rng, abbrs)


def _synthetic_frame(n: int, rng: np.random.Generator,
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


class TestTreeFrameRouting(unittest.TestCase):
    def test_columns_are_production_layout(self):
        df = _synthetic_games(10)
        ids = tune._add_team_ids(df)
        X_num = tune._feature_matrix(ids)
        X_cat = tune._categorical_matrix(ids)
        frame = tune.tree_frame(X_num, X_cat)

        self.assertEqual(list(frame.columns),
                         list(FEATURE_COLS) + TREE_CATEGORICAL_COLS)
        for col in TREE_CATEGORICAL_COLS:
            self.assertIsInstance(frame[col].dtype, pd.CategoricalDtype,
                                  f"{col} must be pd.Categorical, not int")
        # Per-column explicit category set: every ID seen in THAT space +
        # its own reserved UNK slot (team/venue/starter vocabularies never
        # leak into each other).
        for col in TREE_CATEGORICAL_COLS:
            cats = set(frame[col].cat.categories)
            self.assertTrue(
                set(_cat_known_ids(col)).issubset(cats),
                f"{col} category set missing its own known/UNK ids")
            self.assertIn(_cat_unk_for(col), cats,
                          f"{col} missing its reserved UNK slot")

    def test_unknown_and_negative_ids_clamp_to_unk(self):
        df = _synthetic_games(6)
        ids = tune._add_team_ids(df)
        X_num = tune._feature_matrix(ids)
        X_cat = tune._categorical_matrix(ids).copy()
        X_cat[0, 0] = -5          # negative → must clamp to UNK, not 0
        X_cat[1, 1] = 999         # out-of-range → same
        frame = tune.tree_frame(X_num, X_cat)
        self.assertEqual(int(frame[TREE_CATEGORICAL_COLS[0]].cat.categories[0]),
                         min(sorted(_TEAM_ABBR_TO_ID.values())[0], UNK_TEAM_ID))
        codes_home = frame[TREE_CATEGORICAL_COLS[0]].cat.codes.to_numpy()
        cats = list(frame[TREE_CATEGORICAL_COLS[0]].cat.categories)
        self.assertEqual(cats[codes_home[0]], UNK_TEAM_ID,
                         "negative id aliased a real team instead of UNK")
        codes_away = frame[TREE_CATEGORICAL_COLS[1]].cat.codes.to_numpy()
        cats_away = list(frame[TREE_CATEGORICAL_COLS[1]].cat.categories)
        self.assertEqual(cats_away[codes_away[1]], UNK_TEAM_ID)

    def test_venue_and_starter_columns_unk_safe_when_sources_missing(self):
        """Synthetic frames have no venue/starter source columns — the
        categorical columns must still exist, all mapping to their own UNK
        slots (never the team vocabulary, never a crash)."""
        df = _synthetic_games(8)
        ids = tune._add_team_ids(df)
        for col in ("venue_id", "home_starter_cat_id", "away_starter_cat_id"):
            self.assertIn(col, ids.columns)
            self.assertEqual(int(ids[col].iloc[0]), _cat_unk_for(col))


class TestBaseParams(unittest.TestCase):
    def test_enable_categorical_is_always_on(self):
        p = tune.base_params(None)
        self.assertTrue(p["enable_categorical"])
        # Even a hostile "sampled" override cannot turn it off.
        p2 = tune.base_params({"enable_categorical": False})
        self.assertTrue(p2["enable_categorical"])

    def test_sampled_params_flow_through(self):
        p = tune.base_params({"max_depth": 3, "_rounds": 123})
        self.assertEqual(p["max_depth"], 3)
        self.assertNotIn("_rounds", p)  # private bookkeeping stripped

    def test_current_config_reference_is_verbatim_xgboost_params(self):
        """The holdout baseline must be production's actual dict — no stale
        hardcodes (the old tuner fabricated depth5/mcw1/gamma0)."""
        cur = tune.base_params(dict(XGBOOST_PARAMS))
        for k, v in XGBOOST_PARAMS.items():
            if k.startswith("_"):
                continue
            self.assertEqual(cur.get(k), v,
                             f"baseline drifted from production: {k}")


class TestPrepareAndFit(unittest.TestCase):
    def test_prepare_fold_imputes_train_medians_only(self):
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=99)
        fold = tune.prepare_fold(tr, va)
        for key in ("train_frame", "val_frame", "y_train", "y_val"):
            self.assertIn(key, fold)
        # Numeric block imputed where train medians exist (all-NaN columns
        # legitimately keep NaN — native routing).
        num_block = fold["train_frame"][list(FEATURE_COLS)].to_numpy(float)
        med = tune.train_medians(tune._feature_matrix(tune._add_team_ids(tr)))
        finite_cols = np.isfinite(med)
        self.assertFalse(
            np.isnan(num_block[:, finite_cols]).any(),
            "numeric columns with valid train medians must be imputed")
        self.assertEqual(len(fold["y_val"]), len(va))

    def test_fit_fold_end_to_end(self):
        tr, va = _synthetic_games(60), _synthetic_games(20, seed=11)
        fold = tune.prepare_fold(tr, va)
        params = tune.base_params({
            "max_depth": 2, "min_child_weight": 5, "gamma": 1.0,
            "subsample": 0.8, "colsample_bytree": 0.8, "eta": 0.05,
        })
        proba, best = tune.fit_fold(params, fold, early_stop=True)
        self.assertEqual(len(proba), len(va))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
        self.assertGreaterEqual(best, 1)


if __name__ == "__main__":
    unittest.main()
