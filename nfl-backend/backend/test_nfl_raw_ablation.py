"""Raw-columns ablation tests — pure, no-network.

Covers ``nfl_raw_columns`` (the composed-but-unregistered per-side columns)
and ``run_nfl_raw_ablation`` (the harness arms + per-member masks):
- Raw columns are emitted for EXACTLY the performance set (elo, win_pct,
  rest_days, ewm_net_pts, ewm_ypp, pace, rest_short per side) — no venue,
  travel, or schedule raws.
- PIT: raw values on each row are functions of strictly-prior games only
  (a changed FUTURE game's score leaves every earlier row's raw values
  untouched).
- Coverage: every raw column ≥ 95% on a frame with enough history.
- Member masks: the logistic column set contains ZERO raws; the tree/mlp set
  includes them; arms compose as C0 = deployed 12, RAW_ADDED = 26.
- Invariant: no raw column is in FEATURE_COLUMNS (composed-but-unregistered).

Run: python -m unittest test_nfl_raw_ablation -v   (no network needed)
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import nfl_features as nf
from nfl_raw_columns import (RAW_PER_SIDE_COLS, compose_raw_columns,
                             raw_coverage)
from run_feature_winpct_ablation import DEPLOYED_12
from run_nfl_raw_ablation import _member_plan, build_arms


def _pbp_row(**kw) -> dict:
    base = dict(game_id="g1", posteam="A", yards_gained=6, epa=0.3, qb_epa=0.2,
                game_seconds_remaining=3300)
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
    """Two plays per (game_id, team) with a shrinking clock (pace > 0)."""
    rows = []
    for gid, home, away in games:
        for team, opp in ((home, away), (away, home)):
            for k in range(2):
                rows.append(_pbp_row(
                    game_id=gid, posteam=team, yards_gained=6 + k * 4,
                    epa=0.3 + k * 0.1, qb_epa=0.2 + k * 0.1,
                    game_seconds_remaining=3300 - k * 300))
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
    return feats, sched, pbp


class TestRawColumns(unittest.TestCase):
    def test_raw_set_exact(self):
        feats, sched, pbp = _three_game_frame()
        out = compose_raw_columns(feats, sched, pbp)
        for c in RAW_PER_SIDE_COLS:
            self.assertIn(c, out.columns)
        # no venue / travel / schedule raws
        for c in ("is_dome_home_home", "is_dome_home_away", "altitude_home_home",
                  "prime_time_home", "div_game_home", "div_game_away",
                  "travel_miles_home", "travel_miles_away"):
            self.assertNotIn(c, out.columns)
        # diffs untouched
        for c in DEPLOYED_12:
            self.assertIn(c, out.columns)

    def test_pit_strictly_prior(self):
        feats_a, sched_a, pbp = _three_game_frame()
        out_a = compose_raw_columns(feats_a, sched_a, pbp)
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
        out_b = compose_raw_columns(feats_b, sched_b, pbp)
        for c in RAW_PER_SIDE_COLS:
            a = out_a.set_index("game_id")[c]
            b = out_b.set_index("game_id")[c]
            for gid in ("g1", "g2"):
                self.assertTrue(
                    (pd.isna(a[gid]) and pd.isna(b[gid]))
                    or (a[gid] == b[gid]),
                    f"{c}@{gid} changed when only g3's outcome changed: "
                    f"{a[gid]} vs {b[gid]}")

    def test_missing_pbp_degrades_to_nan(self):
        feats, sched, _pbp = _three_game_frame()
        out = compose_raw_columns(feats, sched, None)
        for c in ("ewm_ypp_home", "ewm_ypp_away",
                  "pace_plays_min_home", "pace_plays_min_away"):
            self.assertTrue(out[c].isna().all(), c)
        # schedule-derived raws still populate after the first game
        g3 = out[out["game_id"] == "g3"].iloc[0]
        for c in ("elo_home", "elo_away", "win_pct_home", "rest_days_home",
                  "ewm_net_pts_home", "ewm_net_pts_away"):
            self.assertTrue(pd.notna(g3[c]), c)

    def test_coverage_floor(self):
        # 25 decided games between X and Y (alternating home/away) so every
        # raw has 24 prior appearances -> >= 95% coverage on all 14 raws.
        rows, games = [], []
        for i in range(25):
            gid = f"g{i}"
            home, away = ("X", "Y") if i % 2 == 0 else ("Y", "X")
            rows.append(dict(game_id=gid, home_team=home, away_team=away,
                             home_score=20 + i, away_score=14 + i))
            games.append((gid, home, away))
        sched = _synth_schedule(rows)
        pbp = _small_pbp(games)
        feats = nf.build_features(sched.copy(), sched, pbp)
        out = compose_raw_columns(feats, sched, pbp)
        cov = raw_coverage(out)
        self.assertEqual(set(cov), set(RAW_PER_SIDE_COLS))
        for c, pct in cov.items():
            self.assertGreaterEqual(pct, 95.0, f"{c} coverage {pct:.1f}%")

    def test_not_in_feature_columns(self):
        for c in RAW_PER_SIDE_COLS:
            self.assertNotIn(c, nf.FEATURE_COLUMNS)


class TestMemberMasks(unittest.TestCase):
    def _feats(self) -> pd.DataFrame:
        feats, sched, pbp = _three_game_frame()
        return compose_raw_columns(feats, sched, pbp)

    def test_arm_composition(self):
        arms = build_arms(self._feats())
        self.assertEqual(arms["C0"], DEPLOYED_12)
        self.assertEqual(len(arms["C0"]), 12)
        self.assertEqual(len(arms["RAW_ADDED"]), 26)
        self.assertEqual(set(arms["RAW_ADDED"]) - set(arms["C0"]),
                         set(RAW_PER_SIDE_COLS))

    def test_logistic_mask_zero_raws_trees_include_them(self):
        feats = self._feats()
        tree_cols, logi_cols = _member_plan(feats)
        self.assertEqual(logi_cols, build_arms(feats)["C0"])
        self.assertEqual(set(logi_cols) & set(RAW_PER_SIDE_COLS), set())
        self.assertEqual(set(tree_cols), set(build_arms(feats)["RAW_ADDED"]))
        self.assertTrue(set(RAW_PER_SIDE_COLS).issubset(tree_cols))

    def test_masks_survive_missing_raw_columns(self):
        # a frame without any raws still builds a sane mask (logi == C0)
        feats, sched, _pbp = _three_game_frame()
        tree_cols, logi_cols = _member_plan(feats)
        self.assertEqual(logi_cols, DEPLOYED_12)
        self.assertEqual(tree_cols, DEPLOYED_12)


if __name__ == "__main__":
    unittest.main()
