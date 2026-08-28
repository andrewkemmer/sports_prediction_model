"""NFL feature engineering v1 (``nfl_features.py``) — pure, no-network tests.

Pins the admission gate's invariants so feature ADMISSION never leaks the
future or admits a redundant candidate silently:
- ELO update rule + prior, and strict no-future use (a later game can never
  change an earlier game's entering rating / trailing stats).
- Trailing windows use ONLY strictly-prior games (per-team shift), and the
  code-level strict-monotonicity assertion actually fires on bad input.
- ``univariate_auc`` correctness (perfect separation = 1, inverse = 0, random ~0.5).
- coverage floor drops <95%-coverage features.
- ``build_features`` composes the dome flag, ELO/form/rest/ypp diffs end-to-end.
"""
import unittest
from datetime import date

import numpy as np
import pandas as pd

from nfl_features import (
    COVERAGE_FLOOR, FEATURE_COLUMNS, build_features, compute_elo,
    team_events, team_stats_ladder, univariate_auc,
)


def synth_games(rows: list[dict]) -> pd.DataFrame:
    """Decided-shaped frame with the columns build_features/team_events need."""
    default = {
        "game_id": None, "season": 2019, "week": 1, "game_type": "REG",
        "gameday": None, "home_team": None, "away_team": None,
        "home_score": None, "away_score": None, "roof": "outdoors",
    }
    games = []
    for r in rows:
        d = dict(default)
        d.update(r)
        games.append(d)
    return pd.DataFrame(games)


class TestElo(unittest.TestCase):
    def test_update_rule_and_prior(self):
        """Two teams, one game, A wins: exp=0.5 for both (equal ratings) ->
        A 1516, B 1484. Entering ratings equal the prior (1500)."""
        g = synth_games([
            dict(game_id="G1", gameday="2019-09-08", home_team="A", away_team="B",
                 home_score=24, away_score=10),
        ])
        ev = compute_elo(team_events(g))
        a = ev[ev["team"] == "A"].iloc[0]
        b = ev[ev["team"] == "B"].iloc[0]
        self.assertEqual(a["elo_entering"], 1500.0)
        self.assertEqual(b["elo_entering"], 1500.0)
        # recompute post-game ratings (not stored) by replaying the update
        self.assertAlmostEqual(1500 + 32 * (a["team_win"] - 0.5), 1516.0, places=3)
        self.assertAlmostEqual(1500 + 32 * (b["team_win"] - 0.5), 1484.0, places=3)

    def test_entering_rating_uses_only_prior_games(self):
        """Ratings entering game k equal the ratings AFTER game k-1 for that
        team, so no future game can influence an earlier game."""
        g = synth_games([
            dict(game_id="G1", gameday="2019-09-08", home_team="A", away_team="B",
                 home_score=24, away_score=10),
            dict(game_id="G2", gameday="2019-09-15", home_team="B", away_team="A",
                 home_score=7, away_score=21),   # A wins again
            dict(game_id="G3", gameday="2019-09-22", home_team="A", away_team="C",
                 home_score=3, away_score=30),   # A gets blown out (future for G2)
        ])
        ev = compute_elo(team_events(g))
        # G2's entering ratings must reflect ONLY G1 (G2 is the 2nd A/B game).
        g2 = ev[ev["game_id"] == "G2"]
        self.assertEqual(g2[g2["team"] == "A"]["elo_entering"].iloc[0], 1516.0)
        self.assertEqual(g2[g2["team"] == "B"]["elo_entering"].iloc[0], 1484.0)


class TestTrailingLeakage(unittest.TestCase):
    def test_trailing_uses_only_strictly_prior_games(self):
        g = synth_games([
            dict(game_id="G1", gameday="2019-09-01", home_team="A", away_team="B",
                 home_score=21, away_score=10),   # A += 11
            dict(game_id="G2", gameday="2019-09-08", home_team="A", away_team="C",
                 home_score=14, away_score=20),   # A += -6
            dict(game_id="G3", gameday="2019-09-15", home_team="A", away_team="D",
                 home_score=28, away_score=24),   # A += 4
        ])
        ladder = team_stats_ladder(team_events(g)).set_index(["game_id", "team"])
        # G3's trailing form (window 4) = mean over G1,G2 only = (11 + -6)/2 = 2.5
        self.assertAlmostEqual(ladder.loc[("G3", "A"), "form_pts"], 2.5, places=6)
        # G2's trailing form = just G1 = 11.0 (not affected by the later G3)
        self.assertAlmostEqual(ladder.loc[("G2", "A"), "form_pts"], 11.0, places=6)
        self.assertTrue(pd.isna(ladder.loc[("G1", "A"), "form_pts"]))

    def test_appending_future_game_does_not_change_earlier_stats(self):
        g1 = synth_games([
            dict(game_id="G1", gameday="2019-09-01", home_team="A", away_team="B",
                 home_score=21, away_score=10),
            dict(game_id="G2", gameday="2019-09-08", home_team="A", away_team="C",
                 home_score=14, away_score=20),
        ])
        base = team_stats_ladder(team_events(g1)).set_index(["game_id", "team"])
        with_future = synth_games([
            dict(game_id="G1", gameday="2019-09-01", home_team="A", away_team="B",
                 home_score=21, away_score=10),
            dict(game_id="G2", gameday="2019-09-08", home_team="A", away_team="C",
                 home_score=14, away_score=20),
            dict(game_id="G3", gameday="2019-12-01", home_team="A", away_team="D",
                 home_score=70, away_score=0),   # extreme future blowout
        ])
        alt = team_stats_ladder(team_events(with_future)).set_index(["game_id", "team"])
        for stat in ("form_pts", "win_pct", "rest_days", "ypp"):
            for gid in ("G1", "G2"):
                v_base = base.loc[(gid, "A"), stat]
                v_alt = alt.loc[(gid, "A"), stat]
                if pd.isna(v_base) and pd.isna(v_alt):
                    continue
                self.assertEqual(v_base, v_alt,
                                 f"{stat} of {gid} changed by a future game")

    def test_rest_days_diff(self):
        g = synth_games([
            dict(game_id="G1", gameday="2019-09-01", home_team="A", away_team="B",
                 home_score=21, away_score=10),
            dict(game_id="G2", gameday="2019-09-08", home_team="A", away_team="B",
                 home_score=7, away_score=14),
        ])
        ladder = team_stats_ladder(team_events(g)).set_index(["game_id", "team"])
        self.assertEqual(ladder.loc[("G2", "A"), "rest_days"], 7)
        self.assertEqual(ladder.loc[("G2", "B"), "rest_days"], 7)
        self.assertTrue(pd.isna(ladder.loc[("G1", "A"), "rest_days"]))

    def test_strict_monotonicity_assertion_fires(self):
        """Two games with the SAME gameday for one team must raise — proves the
        leakage assertion is live, not decorative."""
        g = synth_games([
            dict(game_id="G1", gameday="2019-09-08", home_team="A", away_team="B",
                 home_score=21, away_score=10),
            dict(game_id="G2", gameday="2019-09-08", home_team="A", away_team="C",
                 home_score=7, away_score=3),   # A plays twice the same day
        ])
        with self.assertRaises(AssertionError):
            team_stats_ladder(team_events(g))


class TestAuc(unittest.TestCase):
    def test_perfect_separation(self):
        y = np.array([1, 1, 1, 0, 0, 0])
        x = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
        self.assertAlmostEqual(univariate_auc(y, x), 1.0, places=6)

    def test_inverse_separation(self):
        y = np.array([1, 1, 1, 0, 0, 0])
        x = -np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
        self.assertAlmostEqual(univariate_auc(y, x), 0.0, places=6)

    def test_random_and_nan_handling(self):
        rng = np.random.default_rng(7)
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        x = rng.normal(size=8)
        self.assertAlmostEqual(univariate_auc(y, x), 0.5, delta=0.2)
        self.assertTrue(np.isnan(univariate_auc(np.array([1, 1]), np.array([2.0, np.nan]))))


class TestGate(unittest.TestCase):
    def test_coverage_floor_drops_bad_feature(self):
        """A candidate below the coverage floor is dropped from the v1 set with
        a recorded reason; a fully-covered informative feature is not."""
        frame = synth_games([dict(game_id=f"G{i}", gameday=f"2019-09-{1+2*i:02d}",
                                  home_team="H", away_team="A",
                                  home_score=24, away_score=10) for i in range(20)])
        frame["season"] = 2019
        for f in FEATURE_COLUMNS:
            frame[f] = 0.0                            # fully covered, but constant
        frame["elo_diff"] = np.linspace(-2, 2, len(frame))
        frame["spread_line"] = 0.0
        frame["total_line"] = 0.0
        frame["result"] = 14.0
        # make every trailing feature 50%-covered -> should all be dropped
        for f in ("form_diff_pts", "win_pct_diff", "rest_days_diff",
                  "ypp_diff", "is_dome_home"):
            frame[f] = frame[f].mask(dummy := np.array([i % 2 == 0 for i in range(len(frame))]))

        from nfl_features import run_feature_gate
        res = run_feature_gate(frame)
        for f in ("form_diff_pts", "win_pct_diff", "rest_days_diff",
                  "ypp_diff", "is_dome_home"):
            self.assertIn(f, res["dropped"], f"{f} should be dropped for coverage")
            self.assertIn("coverage", res["reasons"][f])
        self.assertIn("elo_diff", res["v1_features"])


class TestBuildFeatures(unittest.TestCase):
    def test_compose_end_to_end(self):
        decided = synth_games([
            dict(game_id="2019_01_KC_JAX", gameday="2019-09-08", home_team="JAX",
                 away_team="KC", home_score=26, away_score=40, roof="outdoors"),
            dict(game_id="2019_01_GB_CHI", gameday="2019-09-05", home_team="CHI",
                 away_team="GB", home_score=3, away_score=10, roof="closed"),
        ])
        decided = decided.drop(columns=["roof"])
        schedule = pd.DataFrame({
            "game_id": ["2019_01_KC_JAX", "2019_01_GB_CHI"],
            "season": [2019, 2019], "week": [1, 1], "gameday": ["2019-09-08", "2019-09-05"],
            "home_team": ["JAX", "CHI"], "away_team": ["KC", "GB"],
            "home_score": [26, 3], "away_score": [40, 10], "roof": ["outdoors", "closed"],
        })
        pbp = pd.DataFrame({
            "game_id": ["2019_01_KC_JAX", "2019_01_KC_JAX", "2019_01_KC_JAX"],
            "posteam": ["KC", "KC", "JAX"], "yards_gained": [10, 6, -2],
        })
        feats = build_features(decided, schedule, pbp)
        self.assertEqual(list(feats["is_dome_home"]), [0.0, 1.0])
        jax = feats[feats["game_id"] == "2019_01_KC_JAX"].iloc[0]
        chi = feats[feats["game_id"] == "2019_01_GB_CHI"].iloc[0]
        self.assertEqual(jax["is_home"], 1.0)         # anchor
        # both are each team's only game -> trailing priors are NaN -> diff NaN
        self.assertTrue(pd.isna(jax["form_diff_pts"]))
        # ELO entering = the 1500 prior for this tiny 1-game-per-team timeline
        self.assertAlmostEqual(jax["elo_diff"], 0.0, places=6)

        for col in ("elo_diff", "form_diff_pts", "win_pct_diff",
                    "rest_days_diff", "ypp_diff", "is_dome_home", "is_home"):
            self.assertIn(col, feats.columns)


if __name__ == "__main__":
    unittest.main()