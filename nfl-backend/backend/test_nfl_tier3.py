"""Tier-3 (market de-vig / officials / roster) candidate + harness tests.

Covers the five v5 candidates composed by ``nfl_features``:
market_home_implied (no-vig closing-moneyline de-vig), ref_pen_tend and
ref_pace (strictly-prior head-referee crew tendencies), and roster_age_diff /
roster_exp_diff (pre-season team means from the committed snapshot CSV), plus
the run_tier3_ablation.py arm composition.

Run: python -m unittest test_nfl_tier3 -v   (no network needed)
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from nfl_features import (_american_implied, _compose_market_candidates,
                          _compose_officials_candidates,
                          _compose_roster_candidates, build_features,
                          build_slate_features)
from run_tier1_ablation import WITHOUT_FEATURES
from run_tier2_ablation import VENUE_3_FEATURES
from run_tier3_ablation import build_arms

EWM_ALPHA = 1 - 0.5 ** (1 / 2)          # halflife=2, adjust=False


def _synth(rows: list[dict]) -> pd.DataFrame:
    default = dict(game_id=None, season=2020, week=1, gameday=None,
                   home_team=None, away_team=None, home_score=None,
                   away_score=None, roof="outdoors")
    return pd.DataFrame([{**default, **r} for r in rows])


class TestMarket(unittest.TestCase):
    def test_american_implied_arithmetic(self):
        s = _american_implied(pd.Series([-180.0, 150.0, -110.0, 100.0]))
        self.assertAlmostEqual(s.iloc[0], 180 / 280, places=6)
        self.assertAlmostEqual(s.iloc[1], 100 / 250, places=6)
        self.assertAlmostEqual(s.iloc[2], 110 / 210, places=6)
        self.assertAlmostEqual(s.iloc[3], 100 / 200, places=6)

    def test_no_vig_normalization(self):
        df = pd.DataFrame({"game_id": ["g1", "g2", "g3"],
                           "home_moneyline": [-180.0, -110.0, 250.0],
                           "away_moneyline": [150.0, -110.0, -400.0]})
        out = _compose_market_candidates(df.copy(), None)
        ph, pa = 180 / 280, 100 / 250
        self.assertAlmostEqual(out["market_home_implied"].iloc[0],
                               ph / (ph + pa), places=6)
        # Vig-free pair -> exactly the fair-odds implied prob (0.5).
        self.assertAlmostEqual(out["market_home_implied"].iloc[1], 0.5, places=9)

    def test_both_odds_positive_symmetric(self):
        df = pd.DataFrame({"game_id": ["g1", "g2"],
                           "home_moneyline": [150.0, 300.0],
                           "away_moneyline": [150.0, 120.0]})
        out = _compose_market_candidates(df.copy(), None)
        self.assertAlmostEqual(out["market_home_implied"].iloc[0], 0.5, places=9)
        ph, pa = 100 / 400, 100 / 220
        self.assertAlmostEqual(out["market_home_implied"].iloc[1],
                               ph / (ph + pa), places=6)

    def test_missing_side_is_nan(self):
        df = pd.DataFrame({"game_id": ["g1"], "home_moneyline": [-180.0],
                           "away_moneyline": [np.nan]})
        out = _compose_market_candidates(df.copy(), None)
        self.assertTrue(pd.isna(out["market_home_implied"].iloc[0]))

    def test_merge_from_schedule(self):
        sched = pd.DataFrame({"game_id": ["g1"], "home_moneyline": [-180.0],
                              "away_moneyline": [150.0]})
        out = _compose_market_candidates(
            pd.DataFrame({"game_id": ["g1"]}), sched)
        self.assertFalse(pd.isna(out["market_home_implied"].iloc[0]))


class TestOfficials(unittest.TestCase):
    """ref_pen_tend: strictly-prior (team, crew) EWM of penalty yards called
    against each team, home − away; unknown referee / first encounter -> NaN.
    ref_pace: crew trailing plays/game, with the same slate fallback."""

    def _frame(self):
        s = _synth([
            dict(game_id="g1", gameday="2020-09-13", home_team="X",
                 away_team="Y", home_score=1, away_score=0, referee="R"),
            dict(game_id="g2", gameday="2020-09-20", home_team="X",
                 away_team="Y", home_score=1, away_score=0, referee="R"),
            dict(game_id="g3", gameday="2020-09-27", home_team="X",
                 away_team="Y", home_score=1, away_score=0, referee="R"),
            dict(game_id="g4", gameday="2020-10-04", home_team="X",
                 away_team="Y", home_score=1, away_score=0, referee=None),
        ])
        agg = pd.DataFrame({
            "game_id": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "team": ["X", "Y", "X", "Y", "X", "Y"],
            "n_plays": [100, 100, 110, 110, 120, 120],
            "penalty_yds": [0.0, 0.0, 40.0, 0.0, 40.0, 0.0]})
        return s, agg

    def test_first_encounter_and_unknown_ref_are_nan(self):
        s, agg = self._frame()
        out = _compose_officials_candidates(s.copy(), s, agg)
        self.assertTrue(pd.isna(out["ref_pen_tend"].iloc[0]))   # no prior
        self.assertTrue(pd.isna(out["ref_pen_tend"].iloc[3]))   # no referee
        self.assertTrue(pd.isna(out["ref_pace"].iloc[3]))

    def test_strictly_prior_ewm(self):
        s, agg = self._frame()
        out = _compose_officials_candidates(s.copy(), s, agg)
        # g3 home X prior = EWM([g1:0, g2:40]) = alpha*40; away Y prior = 0.
        self.assertAlmostEqual(out["ref_pen_tend"].iloc[1], 0.0, places=6)
        expect = EWM_ALPHA * 40.0
        self.assertAlmostEqual(out["ref_pen_tend"].iloc[2], expect, places=4)
        # own game excluded: g3's own 40-yd flag not in its own value
        self.assertLess(out["ref_pen_tend"].iloc[2], 40.0)

    def test_slate_fallback_uses_crew_history(self):
        s, agg = self._frame()
        s = pd.concat([s, _synth([dict(game_id="g5", gameday="2020-10-11",
                                       home_team="X", away_team="Y",
                                       home_score=None, away_score=None,
                                       referee="R")])], ignore_index=True)
        out = _compose_officials_candidates(s.copy(), s, agg)
        # undecided row (not in team_agg) -> latest prior (X, R) EWM and
        # latest prior crew pace.
        self.assertAlmostEqual(out["ref_pen_tend"].iloc[4], EWM_ALPHA * 40.0,
                               places=4)
        pace_expect = EWM_ALPHA * 220 + (1 - EWM_ALPHA) * 200
        self.assertAlmostEqual(out["ref_pace"].iloc[4], pace_expect, places=4)

    def test_no_team_agg_degrades_to_nan(self):
        s, _ = self._frame()
        out = _compose_officials_candidates(s.copy(), s, None)
        for col in ("ref_pen_tend", "ref_pace"):
            self.assertTrue(out[col].isna().all())


class TestRoster(unittest.TestCase):
    def test_2026_values_match_committed_csv(self):
        tab = pd.read_csv("nfl_roster_age_exp.csv")
        df = pd.DataFrame({"game_id": ["x1"], "season": [2026],
                           "home_team": ["ARI"], "away_team": ["ATL"]})
        out = _compose_roster_candidates(df)
        ari = tab[(tab.team == "ARI") & (tab.season == 2026)].iloc[0]
        atl = tab[(tab.team == "ATL") & (tab.season == 2026)].iloc[0]
        self.assertAlmostEqual(out["roster_age_diff"].iloc[0],
                               float(ari.mean_age - atl.mean_age), places=4)
        self.assertAlmostEqual(out["roster_exp_diff"].iloc[0],
                               float(ari.mean_exp - atl.mean_exp), places=4)

    def test_missing_team_season_falls_back(self):
        df = pd.DataFrame({"game_id": ["x2"], "season": [2019],
                           "home_team": ["KC"], "away_team": ["BUF"]})
        out = _compose_roster_candidates(df)
        # 2019 has only ~20 teams in the release; KC/BUF may or may not be
        # present — either way the fallback must not fabricate NaN from
        # absent pairs with data available in another season.
        self.assertFalse(np.isnan(out["roster_age_diff"].iloc[0]))
        self.assertFalse(np.isnan(out["roster_exp_diff"].iloc[0]))

    def test_unknown_team_is_nan(self):
        df = pd.DataFrame({"game_id": ["x3"], "season": [2026],
                           "home_team": ["ZZZ"], "away_team": ["ARI"]})
        out = _compose_roster_candidates(df)
        self.assertTrue(np.isnan(out["roster_age_diff"].iloc[0]))
        self.assertTrue(np.isnan(out["roster_exp_diff"].iloc[0]))


class TestEndToEnd(unittest.TestCase):
    def test_build_features_composes_all_candidates(self):
        decided = _synth([
            dict(game_id="2019_01_KC_JAX", gameday="2019-09-08",
                 home_team="JAX", away_team="KC", home_score=26, away_score=40),
        ])
        schedule = pd.DataFrame({
            "game_id": ["2019_01_KC_JAX"],
            "season": [2019], "week": [1], "gameday": ["2019-09-08"],
            "home_team": ["JAX"], "away_team": ["KC"],
            "home_score": [26], "away_score": [40], "roof": ["outdoors"],
            "home_moneyline": [-120.0], "away_moneyline": [105.0],
            "referee": ["Bob Jones"]})
        pbp = pd.DataFrame({
            "game_id": ["2019_01_KC_JAX", "2019_01_KC_JAX"],
            "posteam": ["KC", "JAX"], "defteam": ["JAX", "KC"],
            "yards_gained": [10, 6], "epa": [0.2, -0.1],
            "penalty": [1, 0], "penalty_yards": [10, 0],
            "penalty_team": ["JAX", "KC"]})
        feats = build_features(decided, schedule, pbp)
        for col in ("market_home_implied", "ref_pen_tend", "ref_pace",
                    "roster_age_diff", "roster_exp_diff"):
            self.assertIn(col, feats.columns)
        self.assertFalse(pd.isna(feats["market_home_implied"].iloc[0]))
        # JAX/KC both have no prior (team, crew) history -> honest NaN.
        self.assertTrue(pd.isna(feats["ref_pen_tend"].iloc[0]))
        self.assertFalse(pd.isna(feats["roster_exp_diff"].iloc[0]))

    def test_build_slate_features_composes_candidates(self):
        schedule = pd.concat([
            _synth([dict(game_id="2024_01_KC_JAX", gameday="2024-09-08",
                         home_team="JAX", away_team="KC",
                         home_score=26, away_score=40)]),
            _synth([dict(game_id="2026_01_KC_JAX", gameday="2026-09-06",
                         season=2026, home_team="JAX", away_team="KC",
                         home_score=None, away_score=None)]),
        ], ignore_index=True)
        schedule["home_moneyline"] = [-120.0, -185.0]
        schedule["away_moneyline"] = [105.0, 160.0]
        schedule["referee"] = ["Bob Jones", None]
        slate = build_slate_features(
            schedule, None, _synth([]), slate_season=2026)
        self.assertEqual(len(slate), 1)
        row = slate.iloc[0]
        for col in ("market_home_implied", "ref_pen_tend", "ref_pace",
                    "roster_age_diff", "roster_exp_diff"):
            self.assertIn(col, slate.columns)
        # 2026 roster snapshot present -> roster facts fill; no referee yet
        # -> OFF facts are NaN; moneyline known -> market fills.
        self.assertFalse(pd.isna(row["roster_exp_diff"]))
        self.assertTrue(pd.isna(row["ref_pen_tend"]))
        ph, pa = 185 / 285, 100 / 260
        self.assertAlmostEqual(row["market_home_implied"], ph / (ph + pa),
                               places=6)


class TestHarnessArms(unittest.TestCase):
    def _feats(self):
        cols = (WITHOUT_FEATURES + VENUE_3_FEATURES
                + ["market_home_implied", "ref_pen_tend", "ref_pace",
                   "roster_age_diff", "roster_exp_diff", "home_win"])
        return pd.DataFrame({c: [1.0] * 3 for c in cols})

    def test_arms_compose_as_13_14_15_15_18(self):
        feats = self._feats()
        arms = build_arms(feats)
        self.assertEqual(len(arms["WITHOUT"]), 13)
        self.assertEqual(len(arms["MARK"]), 14)
        self.assertEqual(len(arms["OFF"]), 15)
        self.assertEqual(len(arms["ROSTER"]), 15)
        self.assertEqual(len(arms["ALL"]), 18)
        self.assertEqual(set(arms["MARK"]) - set(arms["WITHOUT"]),
                         {"market_home_implied"})
        self.assertEqual(set(arms["ROSTER"]) - set(arms["WITHOUT"]),
                         {"roster_age_diff", "roster_exp_diff"})

    def test_arms_keep_only_present_columns(self):
        feats = self._feats().drop(columns=["ref_pen_tend", "ref_pace"])
        arms = build_arms(feats)
        self.assertEqual(len(arms["OFF"]), 13)     # both OFF cols absent


if __name__ == "__main__":
    unittest.main()