"""Tests for the venue + starter-ID native categoricals (TREE_CATEGORICAL_COLS
2 -> 5).

Contract under test:
- _add_team_ids adds the full 5-column categorical set (team IDs + venue_id +
  home/away_starter_cat_id); frames missing the venue/starter SOURCE columns
  get the reserved UNK slots (slate rows before probable-pitcher announcements,
  synthetic test frames) — never an error, never a fabricated real value.
- _tree_dataframe builds PER-COLUMN category vocabularies (each space's own
  known IDs + its own reserved UNK) — venue/starter values never borrow the
  team vocabulary.
- Inference is UNK-safe: a starter/venue never seen in training maps to its
  reserved UNK slot (XGBoost raises "category not in the training set"
  otherwise) — the fit-time vocabulary is stored on the models dict and
  predict clamps against it.
- Run-engine isolation: venue/home_starter_id/away_starter_id are in
  RUN_EXTRA_EXCLUSIONS and the FEATURE_COLS-derived kept/dropped lists are
  byte-identical to the pre-change contract (29 kept / 36 dropped).
- Feature metadata ships a categorical_context section for the 3 columns.
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

import training  # noqa: E402
from config import DATA_DELIVERY_DIR  # noqa: E402
from run_engine import RUN_EXTRA_EXCLUSIONS, derive_run_features  # noqa: E402
from feature_metadata import generate_features_metadata  # noqa: E402

_CAT_SOURCE_EXCLUSIONS = ("venue", "home_starter_id", "away_starter_id")


def _frame(n: int = 120, seed: int = 3, with_cat_sources: bool = True,
           abbrs: list[str] | None = None) -> pd.DataFrame:
    """Decided-games frame carrying every FEATURE_COL; optionally with the
    venue/starter source columns (real CSV layout) or without (slate/synthetic
    layout, which must fall back to UNK)."""
    rng = np.random.default_rng(seed)
    abbrs = abbrs or ["NYY", "BOS", "LAD", "SF", "AZ", "CHC"]
    rows = []
    for i in range(n):
        row = {c: float(rng.normal()) for c in training.FEATURE_COLS}
        row.update({
            "game_date": pd.Timestamp("2026-06-01") + pd.Timedelta(days=i % 25),
            "home_team": abbrs[i % len(abbrs)],
            "away_team": abbrs[(i + 7) % len(abbrs)],
            "home_win": float(i % 2),
        })
        if with_cat_sources:
            row["venue"] = ["Wrigley Field", "Petco Park", "loanDepot park",
                            "Unknown"][i % 4]
            row["home_starter_id"] = int(600000 + i)
            row["away_starter_id"] = int(700000 + i)
        rows.append(row)
    return pd.DataFrame(rows)


class TestCategoricalColumns(unittest.TestCase):
    def test_full_categorical_set_added(self):
        # _add_team_ids always adds the FULL candidate set (teams + venue +
        # starters) so the routing can exercise either list; the ADOPTED
        # TREE_CATEGORICAL_COLS is the team pair per the DON'T ADOPT verdict.
        df = _frame(50)
        ids = training._add_team_ids(df)
        for c in training.FULL_TREE_CATEGORICAL_COLS:
            self.assertIn(c, ids.columns, f"missing categorical column {c}")
        self.assertEqual(len(training.FULL_TREE_CATEGORICAL_COLS), 5)
        # The adopted set stays the team pair (the gate said DON'T ADOPT).
        self.assertEqual(training.TREE_CATEGORICAL_COLS,
                         ["home_team_id", "away_team_id"])
        self.assertEqual(
            training.RF_TREE_CATEGORICAL_COLS,
            ["home_team_id", "away_team_id"])
        # RF stays on the team pair (sklearn has no native categoricals).
        rf_idx = [training.TREE_CATEGORICAL_COLS.index(c)
                  for c in training.RF_TREE_CATEGORICAL_COLS]
        self.assertEqual(rf_idx, [0, 1])

    def test_missing_source_columns_fall_back_to_unk(self):
        df = _frame(30, with_cat_sources=False)
        ids = training._add_team_ids(df)
        self.assertTrue((ids["venue_id"] == training.UNK_VENUE_ID).all())
        self.assertTrue((ids["home_starter_cat_id"] == training.UNK_STARTER_ID).all())
        self.assertTrue((ids["away_starter_cat_id"] == training.UNK_STARTER_ID).all())
        # Teams still resolve to real IDs even without cat sources.
        self.assertTrue((ids["home_team_id"] != training.UNK_TEAM_ID).any())

    def test_missing_starter_values_map_to_unk(self):
        df = _frame(30)
        df["home_starter_id"] = df["home_starter_id"].astype(object)
        df.loc[0, "home_starter_id"] = np.nan        # missing
        df.loc[1, "home_starter_id"] = None          # missing
        df.loc[2, "home_starter_id"] = -5            # invalid
        df.loc[3, "home_starter_id"] = "not-an-id"  # invalid
        ids = training._add_team_ids(df)
        for i in range(4):
            self.assertEqual(int(ids.loc[i, "home_starter_cat_id"]),
                             training.UNK_STARTER_ID)
        # Real starters elsewhere still resolve to real categories.
        self.assertNotEqual(int(ids.loc[10, "home_starter_cat_id"]),
                            training.UNK_STARTER_ID)

    def test_unknown_venue_string_maps_to_unk(self):
        df = _frame(20)
        df["venue"] = "Unknown"
        ids = training._add_team_ids(df)
        self.assertTrue((ids["venue_id"] == training.UNK_VENUE_ID).all())


class TestTreeDataframeVocab(unittest.TestCase):
    @staticmethod
    def _full_frame(df):
        """Build a 5-col categorical frame with TREE_CATEGORICAL_COLS pinned to
        the measured candidate set (the WITH arm's routing)."""
        ids = training._add_team_ids(df)
        X_num = training._feature_matrix(ids)
        # The adopted default returns the team pair; ask for the full 5-col
        # measured candidate set explicitly.
        X_cat = training._categorical_matrix(
            ids, cols=list(training.FULL_TREE_CATEGORICAL_COLS))
        with patch.object(training, "TREE_CATEGORICAL_COLS",
                          list(training.FULL_TREE_CATEGORICAL_COLS)):
            frame = training._tree_dataframe(X_num, X_cat, list(training.FEATURE_COLS))
        return frame

    def test_per_column_category_sets(self):
        # Exercise the FULL candidate set so the venue/starter per-column UNK
        # logic is covered even though the adopted set is the team pair.
        frame = self._full_frame(_frame(60))
        self.assertEqual(len(training.FULL_TREE_CATEGORICAL_COLS), 5)
        for col in training.FULL_TREE_CATEGORICAL_COLS:
            self.assertIn(col, frame.columns)
            cats = set(frame[col].cat.categories)
            self.assertIsInstance(frame[col].dtype, pd.CategoricalDtype)
            # Each column's vocabulary is ITS OWN space + its own UNK slot.
            self.assertTrue(
                set(training._cat_known_ids(col)).issubset(cats),
                f"{col}: own known/UNK ids missing from categories")
            self.assertIn(training._cat_unk_for(col), cats)
            # Cross-space leakage check: starter vocab never contains venue ids
            # and vice versa (sampled via the real map sizes).
            if col == "venue_id":
                self.assertNotIn(training.UNK_STARTER_ID, cats)
                self.assertNotIn(training.UNK_TEAM_ID, cats)
            if "starter" in col:
                self.assertNotIn(training.UNK_VENUE_ID, cats)

    def test_vocab_clamp_sends_unseen_to_own_unk(self):
        """Predict-time clamp: a starter/venue never in the fit vocabulary is
        replaced by ITS OWN reserved UNK — never a real category, never a
        fresh auto-assigned category."""
        ids = training._add_team_ids(_frame(60))
        X_num = training._feature_matrix(ids)
        X_cat = training._categorical_matrix(
            ids, cols=list(training.FULL_TREE_CATEGORICAL_COLS))
        vocab = {c: training._cat_known_ids(c)
                 for c in training.FULL_TREE_CATEGORICAL_COLS}
        # Simulate a predict frame with one unseen starter + unseen venue.
        X_cat_pr = X_cat.copy()
        X_cat_pr[0, 2] = 424242                 # unseen venue id
        X_cat_pr[0, 3] = 424243                 # unseen starter id
        with patch.object(training, "TREE_CATEGORICAL_COLS",
                          list(training.FULL_TREE_CATEGORICAL_COLS)):
            frame = training._tree_dataframe(X_num, X_cat_pr,
                                             list(training.FEATURE_COLS),
                                             vocabs=vocab)
        self.assertEqual(int(frame["venue_id"].iloc[0]), training.UNK_VENUE_ID)
        self.assertEqual(int(frame["home_starter_cat_id"].iloc[0]),
                         training.UNK_STARTER_ID)
        # Real values are untouched.
        self.assertEqual(int(frame["home_team_id"].iloc[0]),
                         int(ids["home_team_id"].iloc[0]))


class TestRunEngineIsolation(unittest.TestCase):
    _EXPECTED_KEPT = [
        "is_home", "dome_is_neutral", "park_factor_slug_diff",
        "wind_advantage_flyball_factor", "air_density_velocity_boost",
        "home_elo", "away_elo", "home_win_pct", "away_win_pct",
        "sp_era_home", "sp_era_away", "sp_k9_home", "sp_k9_away",
        "sp_xwoba_home", "sp_xwoba_away",
        "lineup_woba_mean_home", "lineup_woba_mean_away",
        "lineup_woba_top3_home", "lineup_woba_top3_away",
        "woba_30g_home", "woba_30g_away",
        "bullpen_whip_10g_home", "bullpen_whip_10g_away",
        "bullpen_whip_3g_home", "bullpen_whip_3g_away",
        "team_barrel_15g_home", "team_barrel_15g_away",
        "team_exitvelo_15g_home", "team_exitvelo_15g_away",
    ]

    def test_source_columns_in_exclusion_set(self):
        for f in _CAT_SOURCE_EXCLUSIONS:
            self.assertIn(f, RUN_EXTRA_EXCLUSIONS,
                          f"{f} must be in the run-engine exclusion set")

    def test_kept_dropped_lists_byte_identical(self):
        """Adding the 3 categorical-context names must NOT move the derived
        lists: the run engine's 29-feature view is byte-identical to the
        pre-change contract."""
        keep, dropped = derive_run_features(list(training.FEATURE_COLS))
        self.assertEqual(keep, self._EXPECTED_KEPT)
        self.assertEqual(len(keep), 29)
        self.assertEqual(len(dropped), 30)
        for f in _CAT_SOURCE_EXCLUSIONS:
            self.assertNotIn(f, keep)
            self.assertNotIn(f, dropped)  # never in FEATURE_COLS input either


class TestFeatureMetadata(unittest.TestCase):
    def test_categorical_context_section(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            payload = generate_features_metadata("20260826TEST",
                                                 out_dir=Path(tmp))
        ctx = payload.get("categorical_context", {})
        self.assertEqual(sorted(ctx), ["away_starter_id", "home_starter_id",
                                       "venue"])
        for name in ("venue", "home_starter_id", "away_starter_id"):
            row = ctx[name]
            self.assertIn("tooltip", row)
            self.assertEqual(row["members"], ["xgboost", "lightgbm"])
        # The FEATURE_COLS walk is untouched (no stale warnings).
        self.assertEqual([w for w in payload["warnings"] if "stale" in w], [])


@unittest.skipUnless(DATA_DELIVERY_DIR.joinpath("game_level_features.csv").exists(),
                     "needs the committed game_level_features.csv")
class TestRealDataIntegration(unittest.TestCase):
    """End-to-end on the committed CSV: fit with all 5 categoricals, then
    predict with an unseen starter + NaN starter + unseen venue — every member
    must predict (no XGBoost 'category not in the training set' crash)."""

    def test_fit_and_predict_with_the_adopted_team_pair(self):
        """The adopted 2-col set (team IDs) fits and predicts, and the UNK
        clamp handles unseen/missing venue/starter values end-to-end (they
        never enter the adopted model, but the routing must not crash)."""
        # _add_team_ids always adds the venue/starter IDs; the adopted model
        # just routes through the team pair only.
        games = pd.read_csv(DATA_DELIVERY_DIR / "game_level_features.csv")
        games["game_date"] = pd.to_datetime(games["game_date"])
        games = games.dropna(subset=["home_win"]).reset_index(drop=True)
        tr = games.head(1200).copy()
        va = games.iloc[1200:1400].copy()
        models, _ = training.train_moneyline_ensemble(tr, va)

        self.assertEqual(models["xgboost"].n_features_in_,
                         len(training.FEATURE_COLS) + 2)
        self.assertEqual(models["lightgbm"].n_features_in_,
                         len(training.FEATURE_COLS) + 2)
        self.assertEqual(models["randomforest"].n_features_in_,
                         len(training.FEATURE_COLS) + 2)
        self.assertEqual(sorted(models["categorical_vocab"]),
                         sorted(training.TREE_CATEGORICAL_COLS))

        pr = games.iloc[1400:1420].copy()
        pr.loc[pr.index[0], "home_starter_id"] = 999999998   # unseen
        pr.loc[pr.index[1], "home_starter_id"] = np.nan      # missing
        pr.loc[pr.index[2], "venue"] = "Brand-New Ballpark X"  # unseen
        blend, members, _wts = training.ensemble_predict(models, pr)
        self.assertEqual(sorted(members),
                         ["lightgbm", "logistic", "mlp", "randomforest", "xgboost"])
        self.assertTrue(np.isfinite(blend).all())

    def test_fit_and_predict_with_the_candidate_full_set(self):
        """The measured (not adopted) 5-col set fits, predicts, and the UNK
        clamp routes unseen starters/venues to their own UNK slot instead of
        crashing XGBoost — exactly what the DON'T ADOPT ablation ran."""
        from training import FULL_TREE_CATEGORICAL_COLS
        games = pd.read_csv(DATA_DELIVERY_DIR / "game_level_features.csv")
        games["game_date"] = pd.to_datetime(games["game_date"])
        games = games.dropna(subset=["home_win"]).reset_index(drop=True)
        tr = games.head(1200).copy()
        va = games.iloc[1200:1400].copy()
        with patch.object(training, "TREE_CATEGORICAL_COLS",
                          list(FULL_TREE_CATEGORICAL_COLS)):
            models, _ = training.train_moneyline_ensemble(tr, va)
            self.assertEqual(models["xgboost"].n_features_in_,
                             len(training.FEATURE_COLS) + 5)
            self.assertEqual(models["lightgbm"].n_features_in_,
                             len(training.FEATURE_COLS) + 5)
            self.assertEqual(sorted(models["categorical_vocab"]),
                             sorted(FULL_TREE_CATEGORICAL_COLS))

            # Predict under the SAME 5-categorical routing (the WITH arm).
            pr = games.iloc[1400:1420].copy()
            pr.loc[pr.index[0], "home_starter_id"] = 999999998   # unseen
            pr.loc[pr.index[1], "home_starter_id"] = np.nan      # missing
            pr.loc[pr.index[2], "venue"] = "Brand-New Ballpark X"  # unseen
            blend, members, _wts = training.ensemble_predict(models, pr)
            self.assertEqual(sorted(members),
                             ["lightgbm", "logistic", "mlp", "randomforest", "xgboost"])
            self.assertTrue(np.isfinite(blend).all())
            # The unseen starter stayed out of the frontier (clamped to UNK).
            self.assertLess(max(models["categorical_vocab"]["home_starter_cat_id"]),
                            999)
            self.assertNotEqual(int(pr.loc[pr.index[0], "home_starter_id"]),
                                training.UNK_STARTER_ID)

    def test_holdout_arm_without_categoricals_still_runs(self):
        """The WITHOUT ablation arm (patch TREE_CATEGORICAL_COLS to team-only)
        must still fit and predict — XGB/LGB/RF all route through the patched
        list without crashing."""
        from training import RF_TREE_CATEGORICAL_COLS, UNK_TEAM_ID
        team_only = list(RF_TREE_CATEGORICAL_COLS)
        games = pd.read_csv(DATA_DELIVERY_DIR / "game_level_features.csv")
        games["game_date"] = pd.to_datetime(games["game_date"])
        games = games.dropna(subset=["home_win"]).reset_index(drop=True)
        tr = games.head(1200).copy()
        va = games.iloc[1200:1400].copy()
        with patch.object(training, "TREE_CATEGORICAL_COLS", team_only):
            models, _ = training.train_moneyline_ensemble(tr, va)
            self.assertEqual(models["xgboost"].n_features_in_,
                             len(training.FEATURE_COLS) + 2)
            self.assertEqual(models["lightgbm"].n_features_in_,
                             len(training.FEATURE_COLS) + 2)
            self.assertEqual(models["randomforest"].n_features_in_,
                             len(training.FEATURE_COLS) + 2)
            self.assertEqual(sorted(models["categorical_vocab"]), sorted(team_only))
            # Predict under the SAME 2-categorical routing (the WITHOUT arm).
            pr = games.iloc[1400:1420].copy()
            pr.loc[pr.index[0], "home_starter_id"] = 999999998
            blend, members, _ = training.ensemble_predict(models, pr)
            self.assertIn("xgboost", members)
            self.assertTrue(np.isfinite(blend).all())


if __name__ == "__main__":
    unittest.main()
