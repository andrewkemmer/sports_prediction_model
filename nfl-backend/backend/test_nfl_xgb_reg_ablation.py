"""Regularized-XGB ablation tests — pure, no-network.

Covers ``run_nfl_xgb_reg_ablation`` (the regularized-xgb OPPADJ harness):
- REGULARIZED_XGB_PARAMS is the exact before/after vs the production xgb_kw
  (max_depth 2->3, min_child_weight 8->10, colsample_bytree 0.6->0.5; the
  other keys unchanged) and the patched params reach the trained model.
- Both arms use IDENTICAL xgb params: train_ensemble_reg applies the same
  REGULARIZED_XGB_PARAMS regardless of the column set (asserted on trained
  models for both the C0 and C1 column sets).
- Arms: C0 = deployed 12 (no OPPADJ), C1 = 12 + the 3 OPPADJ features.
- Logistic mask = C0 pool in both arms (zero OPPADJ); trees/mlp see the arm
  columns.
- OPPADJ coverage >= 95% on a frame with enough history.
- PIT: OPPADJ values on each row use strictly-prior games only (a changed
  future game's outcome leaves every earlier row untouched).
- Invariant: no OPPADJ feature is in FEATURE_COLUMNS (composed-but-
  unregistered until an ablation verdict says otherwise).

Run: python -m unittest test_nfl_xgb_reg_ablation -v   (no network needed)
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import nfl_features as nf
from nfl_tier4 import TIER4_OPPADJ_FEATURES, compose_tier4_features
from run_feature_winpct_ablation import DEPLOYED_12
from run_nfl_xgb_reg_ablation import (REGULARIZED_XGB_PARAMS, _member_plan,
                                      build_arms, train_ensemble_reg)

# The production xgb_kw (nfl_moneyline.train_ensemble, inline) — pinned here
# so the before/after is asserted against the real production values.
PROD_XGB_KW = dict(objective="binary:logistic", max_depth=2,
                   min_child_weight=8, gamma=1.0, subsample=0.6,
                   colsample_bytree=0.6, learning_rate=0.06, n_estimators=600,
                   random_state=42, enable_categorical=True,
                   eval_metric="logloss")


def _pbp_row(**kw) -> dict:
    base = dict(game_id="g1", posteam="A", defteam="B", qb_epa=0.0, epa=0.0,
                yards_gained=0, drive=1, wp=0.6, qtr=2, touchdown=0,
                field_goal_result=np.nan, passer_id=None)
    base.update(kw)
    return base


def _synth_schedule(rows: list[dict]) -> pd.DataFrame:
    default = dict(season=2021, week=1, gameday="2021-09-12",
                   home_score=None, away_score=None)
    out = pd.DataFrame([{**default, **r} for r in rows])
    for i in range(len(out)):
        if "gameday" not in rows[i]:
            out.loc[out.index[i], "gameday"] = (
                pd.Timestamp("2021-09-12") + pd.Timedelta(days=7 * i))
    return out


def _small_pbp(games: list[tuple]) -> pd.DataFrame:
    """Three plays per (game_id, team) with the Tier-4 source fields."""
    rows = []
    for gid, home, away in games:
        for team, opp in ((home, away), (away, home)):
            for k in range(3):
                rows.append(_pbp_row(
                    game_id=gid, posteam=team, defteam=opp, qb_epa=0.3 + k * 0.1,
                    epa=0.4 + k * 0.1, yards_gained=6 + k * 4, drive=k + 1,
                    wp=0.5 + k * 0.1, qtr=2, touchdown=0,
                    field_goal_result=np.nan, passer_id="qA"))
    return pd.DataFrame(rows)


def _three_game_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sched = _synth_schedule([
        dict(game_id="g1", home_team="X", away_team="Y",
             home_score=24, away_score=17),
        dict(game_id="g2", home_team="Y", away_team="X",
             home_score=10, away_score=20),
        dict(game_id="g3", home_team="X", away_team="Y",
             home_score=27, away_score=20),
    ])
    decided = sched.copy()
    pbp = _small_pbp([("g1", "X", "Y"), ("g2", "Y", "X"), ("g3", "X", "Y")])
    feats = nf.build_features(decided, sched, pbp)
    return compose_tier4_features(feats, sched, pbp), sched, pbp


class TestRegularizedParams(unittest.TestCase):
    def test_exact_before_after(self):
        # the three changed keys match the run prescription
        self.assertEqual(REGULARIZED_XGB_PARAMS["max_depth"], 3)
        self.assertEqual(REGULARIZED_XGB_PARAMS["min_child_weight"], 10)
        self.assertEqual(REGULARIZED_XGB_PARAMS["colsample_bytree"], 0.5)
        # the unchanged keys match production exactly
        for k in ("gamma", "subsample", "learning_rate", "n_estimators",
                  "objective", "enable_categorical", "eval_metric",
                  "random_state"):
            self.assertEqual(REGULARIZED_XGB_PARAMS[k], PROD_XGB_KW[k], k)

    def test_differs_from_production(self):
        self.assertNotEqual(REGULARIZED_XGB_PARAMS["max_depth"],
                            PROD_XGB_KW["max_depth"])
        self.assertNotEqual(REGULARIZED_XGB_PARAMS["min_child_weight"],
                            PROD_XGB_KW["min_child_weight"])
        self.assertNotEqual(REGULARIZED_XGB_PARAMS["colsample_bytree"],
                            PROD_XGB_KW["colsample_bytree"])


class TestRegParamsReachModel(unittest.TestCase):
    def _feats(self) -> pd.DataFrame:
        return _three_game_frame()[0]

    def _train_on(self, cols: list[str]):
        feats = self._feats()
        feats = feats[feats["home_score"].notna()]
        feats["home_win"] = (feats["home_score"] > feats["away_score"]).astype(int)
        # train_ensemble_reg mirrors production: needs >= a few rows; the
        # 3-game synthetic frame is enough for the xgb member to fit.
        models, _ = train_ensemble_reg(feats, None, features=cols)
        self.assertIn("xgboost", models)
        return models["xgboost"]

    def test_same_params_both_arms(self):
        arms = build_arms(self._feats())
        xgb_c0 = self._train_on(arms["C0"])
        xgb_c1 = self._train_on(arms["C1"])
        for k in ("max_depth", "min_child_weight", "colsample_bytree",
                  "gamma", "subsample", "learning_rate", "n_estimators"):
            self.assertEqual(xgb_c0.get_params()[k], xgb_c1.get_params()[k], k)
        # and they are the REGULARIZED values, not production
        self.assertEqual(xgb_c0.get_params()["max_depth"], 3)
        self.assertEqual(xgb_c0.get_params()["min_child_weight"], 10)
        self.assertEqual(xgb_c0.get_params()["colsample_bytree"], 0.5)


class TestArmsAndMasks(unittest.TestCase):
    def _feats(self) -> pd.DataFrame:
        return _three_game_frame()[0]

    def test_arm_composition(self):
        arms = build_arms(self._feats())
        self.assertEqual(arms["C0"], DEPLOYED_12)
        self.assertEqual(len(arms["C0"]), 12)
        self.assertEqual(len(arms["C1"]), 15)
        self.assertEqual(set(arms["C1"]) - set(arms["C0"]),
                         set(TIER4_OPPADJ_FEATURES))

    def test_logistic_mask_zero_oppadj_trees_include_them(self):
        feats = self._feats()
        tree_cols, logi_cols = _member_plan(feats)
        self.assertEqual(logi_cols, build_arms(feats)["C0"])
        self.assertEqual(set(logi_cols) & set(TIER4_OPPADJ_FEATURES), set())
        self.assertEqual(set(tree_cols), set(build_arms(feats)["C1"]))
        self.assertTrue(set(TIER4_OPPADJ_FEATURES).issubset(tree_cols))

    def test_masks_survive_missing_oppadj(self):
        # a frame without OPPADJ columns still builds a sane mask (logi == C0)
        feats, sched, _pbp = _three_game_frame()
        feats = feats.drop(columns=TIER4_OPPADJ_FEATURES)
        tree_cols, logi_cols = _member_plan(feats)
        self.assertEqual(logi_cols, DEPLOYED_12)
        self.assertEqual(tree_cols, DEPLOYED_12)


class TestCoverage(unittest.TestCase):
    def test_oppadj_coverage_floor(self):
        # 60 decided games alternating X/Y so every game from g3 on has
        # strictly-prior games for BOTH the team and its opponent ->
        # OPPADJ coverage >= 95% on all 3 candidates.
        rows, games = [], []
        for i in range(60):
            gid = f"g{i}"
            home, away = ("X", "Y") if i % 2 == 0 else ("Y", "X")
            rows.append(dict(game_id=gid, home_team=home, away_team=away,
                             home_score=20 + i, away_score=14 + i))
            games.append((gid, home, away))
        sched = _synth_schedule(rows)
        pbp = _small_pbp(games)
        feats = nf.build_features(sched.copy(), sched, pbp)
        out = compose_tier4_features(feats, sched, pbp)
        for c in TIER4_OPPADJ_FEATURES:
            cov = 100 * float(out[c].notna().mean())
            self.assertGreaterEqual(cov, 95.0, f"{c} coverage {cov:.1f}%")

    def test_pit_strictly_prior(self):
        feats_a, sched_a, pbp = _three_game_frame()
        # same season/week/gameday, but g3's outcome differs -> only rows
        # strictly before g3 may keep their values; g1/g2 must be IDENTICAL.
        sched_b = _synth_schedule([
            dict(game_id="g1", home_team="X", away_team="Y",
                 home_score=24, away_score=17),
            dict(game_id="g2", home_team="Y", away_team="X",
                 home_score=10, away_score=20),
            dict(game_id="g3", home_team="X", away_team="Y",
                 home_score=3, away_score=31),          # flipped blowout
        ])
        feats_b = nf.build_features(sched_b.copy(), sched_b, pbp)
        out_b = compose_tier4_features(feats_b, sched_b, pbp)
        for c in TIER4_OPPADJ_FEATURES:
            a = feats_a.set_index("game_id")[c]
            b = out_b.set_index("game_id")[c]
            for gid in ("g1", "g2"):
                self.assertTrue(
                    (pd.isna(a[gid]) and pd.isna(b[gid]))
                    or (a[gid] == b[gid]),
                    f"{c}@{gid} changed when only g3's outcome changed: "
                    f"{a[gid]} vs {b[gid]}")


class TestUnregisteredInvariant(unittest.TestCase):
    def test_oppadj_not_in_served_pool(self):
        for c in TIER4_OPPADJ_FEATURES:
            self.assertNotIn(c, nf.FEATURE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
