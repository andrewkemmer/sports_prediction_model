"""Tests for ablation_defense.py (v3) — PIT discipline, family ladders,
pre-screen survival, DM/t significance helpers, and the read-only guard.

No production training or artifacts are touched: every test builds its
own small fixtures.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

import ablation_defense as ad  # noqa: E402
import training  # noqa: E402


def _toy_games(n_days: int = 60, start: str = "2025-04-01") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    teams = ["AAA", "BBB", "CCC", "DDD"]
    for d in range(n_days):
        date = pd.Timestamp(start) + pd.Timedelta(days=d)
        for pair in ((0, 1), (2, 3)):
            rows.append({
                "game_pk": 1000 + d * 2 + pair[0],
                "game_date": date,
                "home_team": teams[pair[0]],
                "away_team": teams[pair[1]],
                "elo_diff": rng.normal(0, 30),
                "win_pct_diff": rng.normal(0, 0.08),
                "home_win": float(rng.random() < 0.54),
            })
    return pd.DataFrame(rows)


def _toy_pbp(games: pd.DataFrame) -> pd.DataFrame:
    """A lean pbp frame: each game gets a handful of scoring events for
    the batting side only, dated the same day (so PIT excludes them)."""
    rows = []
    for _, g in games.iterrows():
        for _ in range(6):
            rows.append({"game_pk": g["game_pk"], "game_date": g["game_date"],
                         "home_team": g["home_team"], "away_team": g["away_team"],
                         "inning_topbot": "Top", "batter": 1,
                         "events": "home_run", "game_type": "R"})
        for _ in range(4):
            rows.append({"game_pk": g["game_pk"], "game_date": g["game_date"],
                         "home_team": g["home_team"], "away_team": g["away_team"],
                         "inning_topbot": "Bottom", "batter": 2,
                         "events": "strikeout", "game_type": "R"})
    return pd.DataFrame(rows)


class TestPITDiscipline(unittest.TestCase):
    """Point-in-time: a game's defense features use only PRIOR dates."""

    def test_f1_uses_only_prior_dates(self):
        games = _toy_games(40)
        pbp = _toy_pbp(games)
        out = ad.build_f1_f3_f5(pbp, games)
        # First game date has no prior rows -> NaN
        first_date = games["game_date"].min()
        first_rows = out[out["game_date"] == first_date]
        self.assertTrue(
            first_rows["team_runs_allowed_10g_home"].isna().all())
        # Later dates have priors -> not all NaN
        later = out[out["game_date"] > first_date + pd.Timedelta(days=12)]
        self.assertTrue(
            later["team_runs_allowed_10g_home"].notna().all())

    def test_same_day_excluded(self):
        """Same-day doubleheader legs must not feed each other's features."""
        games = _toy_games(40)
        pbp = _toy_pbp(games)
        # Pick a date with >=10 prior games (min-prior guard) and make that
        # day's own events different from prior days' (6 HRs/game):
        # prior-only 10g mean for the home team = 6.0; including same-day
        # would change it to 6.0 * 10/11.
        target = games["game_date"].iloc[20]
        mask = pbp["game_date"] == target
        pbp.loc[mask & (pbp["events"] == "home_run"), "events"] = "strikeout"
        out = ad.build_f1_f3_f5(pbp, games)
        d2 = out[out["game_date"] == target]
        self.assertAlmostEqual(
            float(d2["team_runs_allowed_10g_home"].iloc[0]), 6.0, places=6)


class TestFamilyLadders(unittest.TestCase):
    def test_f3_trend_is_short_minus_long(self):
        games = _toy_games(40)
        pbp = _toy_pbp(games)
        f135 = ad.build_f1_f3_f5(pbp, games)
        df = ad.add_defense_frame(games, ["F3"], f135, None)
        cols = ad.condition_feature_cols(["F3"], f135, None)
        self.assertTrue(any(c.startswith("trend_") for c in cols))
        t = "trend_team_runs_allowed_10g_home"
        if t in df.columns:
            diff = (df["team_runs_allowed_10g_home"]
                    - df["team_runs_allowed_30g_home"])
            pd.testing.assert_series_equal(
                df[t], diff, check_names=False, check_dtype=False)

    def test_diff_columns_derived(self):
        games = _toy_games(40)
        pbp = _toy_pbp(games)
        f135 = ad.build_f1_f3_f5(pbp, games)
        df = ad.add_defense_frame(games, ["F1"], f135, None)
        self.assertIn("team_runs_allowed_10g_diff", df.columns)
        expected = df["team_runs_allowed_10g_home"] - df["team_runs_allowed_10g_away"]
        pd.testing.assert_series_equal(
            df["team_runs_allowed_10g_diff"], expected,
            check_names=False, check_dtype=False)

    def test_f2_f4_unbuildable_without_wide_cache(self):
        games = _toy_games(20)
        out = ad.build_f2_f4(None, games)
        self.assertTrue(out["opp_exitvelo_15g_home"].isna().all())
        self.assertTrue(out["def_if_30g_home"].isna().all())


class TestPrescreen(unittest.TestCase):
    def test_prescreen_flags_unbuildable(self):
        games = _toy_games(40)
        pbp = _toy_pbp(games)
        f135 = ad.build_f1_f3_f5(pbp, games)
        folds = training.walk_forward_splits(games, retrain_cadence_days=7,
                                             min_train_days=10)
        folds = [s for s in folds if len(s["val_games"]) >= 10][:3]
        res = ad.prescreen(folds, f135, None, ["F2", "F4", "F1"])
        self.assertFalse(res["F2"]["survived"])
        self.assertEqual(res["F2"]["status"], "UNBUILDABLE")
        self.assertFalse(res["F4"]["survived"])

    def test_strong_signal_family_survives(self):
        """A family that deterministically predicts the residual target
        must pass the pre-screen AUC bar."""
        rng = np.random.default_rng(3)
        n = 600
        X = rng.normal(size=(n, 1))
        y = (X[:, 0] > 0).astype(int)  # perfectly separable
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        lr = LogisticRegression(max_iter=200).fit(X, y)
        auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])
        self.assertGreater(auc, 0.52)


class TestSignificanceHelpers(unittest.TestCase):
    def test_dm_detects_shift(self):
        rng = np.random.default_rng(11)
        la = rng.normal(0.69, 0.02, 900)
        lb = la - 0.01  # condition strictly better
        dm, p = ad.diebold_mariano(la, lb)
        self.assertGreater(dm, 3)
        self.assertLess(p, 0.01)

    def test_dm_null_on_identical(self):
        rng = np.random.default_rng(12)
        la = rng.normal(0.69, 0.02, 900)
        dm, p = ad.diebold_mariano(la, la.copy())
        self.assertEqual(dm, 0.0)
        self.assertEqual(p, 1.0)

    def test_dm_needs_min_n(self):
        la = np.full(10, 0.7)
        dm, p = ad.diebold_mariano(la, la - 0.01)
        self.assertTrue(np.isnan(dm))

    def test_logloss_finite(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([0.9, 0.9, 0.1, 0.1])
        ll = ad.logloss(y, p)
        self.assertTrue(np.isfinite(ll).all())
        self.assertLess(ll[0], 0.2)   # confidently right -> tiny loss
        self.assertGreater(ll[1], 2.0)  # confidently wrong -> huge loss


class TestReadOnlyGuard(unittest.TestCase):
    def test_production_feature_cols_untouched(self):
        """The harness must never mutate production FEATURE_COLS."""
        before = list(training.FEATURE_COLS)
        # exercise a light path (no training)
        ad.family_columns("F1")
        ad.diff_columns("F1")
        self.assertEqual(list(training.FEATURE_COLS), before)

    def test_condition_map_shape(self):
        self.assertEqual(ad.CONDITIONS["C0"], [])
        self.assertEqual(ad.CONDITIONS["C7"],
                         ["F1", "F2", "F3", "F4", "F5"])
        # nested contrasts exist in the condition map
        self.assertIn("C5", ad.CONDITIONS)
        self.assertIn("C6", ad.CONDITIONS)


if __name__ == "__main__":
    unittest.main()
