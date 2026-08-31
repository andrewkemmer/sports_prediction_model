"""Focused unit tests for the opponent-adjusted trailing ladders.

Covers the two seams that must hold for the gated measurement to be
leak-safe and honest:

  1. trailing_team_ladders / trailing_pitcher_ladders use ONLY rows with
     game_date STRICTLY before the current row's date — same-day games
     (doubleheader legs) never contribute, and no future outcome can leak.
  2. The raw-strength / opponent-adjusted math matches a naive, independent
     O(n^2) reference implementation row-by-row, and the min-games gate
     yields NaN (never imputed) below the required prior-history.

Pure function tests (no ensemble training, no component).
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from run_opponent_adjusted_ablation import (
    add_opponent_adjusted_features,
    trailing_pitcher_ladders,
    trailing_team_ladders,
    OPP_ADJ_COLS,
)
from run_opponent_adjusted_ablation_trees import (
    TREE_MEMBERS,
    filter_tree_members,
)
import training


def _side_frame(games: pd.DataFrame) -> pd.DataFrame:
    """Build the (gidx, date, team, opp, margin, starter, opp_runs) side frame
    exactly as the ablation's add_opponent_adjusted_features does. Starter
    columns default to NaN when absent (team-ladder tests don't need them)."""
    n = len(games)
    dates = pd.to_datetime(games["game_date"]).values
    hst = (games["home_starter_id"].values if "home_starter_id" in games.columns
           else np.full(n, np.nan))
    ast_ = (games["away_starter_id"].values if "away_starter_id" in games.columns
            else np.full(n, np.nan))
    home = pd.DataFrame({
        "gidx": np.arange(n), "date": dates,
        "team": games["home_team"].values, "opp": games["away_team"].values,
        "margin": (games["home_score"] - games["away_score"]).values.astype(float),
        "starter": hst,
        "opp_runs": games["away_score"].values.astype(float),
    })
    away = pd.DataFrame({
        "gidx": np.arange(n), "date": dates,
        "team": games["away_team"].values, "opp": games["home_team"].values,
        "margin": (games["away_score"] - games["home_score"]).values.astype(float),
        "starter": ast_,
        "opp_runs": games["home_score"].values.astype(float),
    })
    return pd.concat([home, away], ignore_index=True)


def _naive_team_ladders(side: pd.DataFrame, window: int, min_games: int):
    """Independent O(n^2) reference: strictly-prior rows only, recursive
    opponent raw strength at each prior row's own date."""
    rows = side.sort_values(["date", "gidx"]).to_dict("records")
    raw: dict = {}
    adj: dict = {}
    for r in rows:
        t, d, gi = r["team"], r["date"], r["gidx"]
        prior = [h for h in rows
                 if h["team"] == t and h["date"] < d]
        win = prior[-window:]
        if len(win) >= min_games:
            raw[(t, gi)] = float(np.mean([h["margin"] for h in win]))
        else:
            raw[(t, gi)] = np.nan
        if len(win) >= min_games:
            opp_vals = np.asarray(
                [raw.get((h["opp"], h["gidx"]), np.nan) for h in win],
                dtype=float)
            if np.isfinite(opp_vals).sum() >= min_games:
                adj[(t, gi)] = float(np.mean([h["margin"] for h in win])
                                     - np.nanmean(opp_vals))
            else:
                adj[(t, gi)] = np.nan
        else:
            adj[(t, gi)] = np.nan
    return raw, adj


class TestTeamLadderPointInTime(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        teams = ["A", "B", "C", "D", "E"]
        n = 40
        home = [teams[i % 5] for i in range(n)]
        away = [teams[(i + 2) % 5] for i in range(n)]
        dates = pd.date_range("2025-03-20", periods=n, freq="1D")
        hs = rng.integers(0, 9, size=n)
        as_ = rng.integers(0, 9, size=n)
        self.games = pd.DataFrame({
            "game_date": dates, "home_team": home, "away_team": away,
            "home_score": hs, "away_score": as_,
        })
        self.side = _side_frame(self.games)

    def test_matches_naive_reference(self):
        """raw and adj maps equal an independent O(n^2) strictly-prior
        implementation for every (team, gidx)."""
        for window, min_games in [(10, 5), (5, 2), (3, 1)]:
            raw, adj = trailing_team_ladders(self.side, window, min_games)
            nraw, nadj = _naive_team_ladders(self.side, window, min_games)
            for key in nraw:
                a, b = raw[key], nraw[key]
                if np.isnan(a) and np.isnan(b):
                    continue
                self.assertAlmostEqual(a, b, places=9,
                                       msg=f"raw mismatch {key} w={window} m={min_games}")
            for key in nadj:
                a, b = adj[key], nadj[key]
                if np.isnan(a) and np.isnan(b):
                    continue
                self.assertAlmostEqual(a, b, places=9,
                                       msg=f"adj mismatch {key} w={window} m={min_games}")

    def test_min_games_gate_returns_nan_not_imputed(self):
        """Below the min-games gate the ladder is NaN — never 0/median."""
        raw, adj = trailing_team_ladders(self.side, window=10, min_games=8)
        # Early rows of every team have < 8 prior games -> NaN.
        for key, val in list(raw.items())[:30]:
            self.assertTrue(np.isnan(val), f"raw {key} should be NaN")
        for key, val in list(adj.items())[:30]:
            self.assertTrue(np.isnan(val), f"adj {key} should be NaN")

    def test_same_day_rows_never_contribute(self):
        """A same-date blowout must not change a later same-date game's
        ladder (doubleheader legs excluded — nothing after first pitch can
        leak into that day's earlier legs)."""
        base = pd.DataFrame({
            "game_date": pd.to_datetime(["2025-04-01", "2025-04-02"]),
            "home_team": ["A", "B"], "away_team": ["B", "A"],
            "home_score": [1, 10], "away_score": [0, 0],
        })
        with_leg = pd.concat([base, pd.DataFrame({
            "game_date": pd.to_datetime(["2025-04-02"]),
            "home_team": ["A"], "away_team": ["B"],
            "home_score": [9], "away_score": [0],
        })], ignore_index=True)
        without_leg = base
        raw_no, _ = trailing_team_ladders(_side_frame(without_leg),
                                          window=5, min_games=1)
        raw_yes, _ = trailing_team_ladders(_side_frame(with_leg),
                                           window=5, min_games=1)
        # The 04-02 game's A ladder must be identical with or without the
        # same-date 9-0 leg (only the 04-01 +1 margin is prior).
        # (gidx of A's 04-02 home side differs between frames; compare by
        # re-keying on date+team.)
        def by_date_team(rawmap, side):
            out = {}
            for r in side.to_dict("records"):
                if r["team"] == "A":
                    out[(pd.Timestamp(r["date"]), "A")] = rawmap[(r["team"], r["gidx"])]
            return out
        no_map = by_date_team(raw_no, _side_frame(without_leg))
        yes_map = by_date_team(raw_yes, _side_frame(with_leg))
        self.assertAlmostEqual(float(no_map[pd.Timestamp("2025-04-02"), "A"]),
                               float(yes_map[pd.Timestamp("2025-04-02"), "A"]),
                               places=9,
                               msg="same-date leg must not affect the ladder")
        # The ladder uses ONLY the strictly-prior 04-01 margin (+1) — the
        # same-date 9-0 leg must be absent from the 04-02 value.
        self.assertEqual(float(yes_map[pd.Timestamp("2025-04-02"), "A"]), 1.0)
        # A has no prior game on 04-01 -> NaN there (never a fabricated 0).
        self.assertTrue(np.isnan(yes_map[pd.Timestamp("2025-04-01"), "A"]))


class TestPitcherLadder(unittest.TestCase):
    @staticmethod
    def _fixture():
        """Five games: B (and A) have prior history before S1's starts, so the
        opponent raw-strength lookup is non-NaN at S1's prior starts.

        B margins: 03-30 +2 (away), 03-31 +4 (home), 04-01 +8 (away vs A),
        04-02 +1 (home vs A). A margins: 04-01 -8, 04-02 -1, 04-03 +2.
        S1 starts: 04-01 (home vs B, allowed 8), 04-02 (away at B, allowed 1),
        04-03 (home vs C, allowed 2).
        """
        return pd.DataFrame({
            "game_date": pd.to_datetime([
                "2025-03-30", "2025-03-31", "2025-04-01",
                "2025-04-02", "2025-04-03"]),
            "home_team": ["C", "B", "A", "B", "A"],
            "away_team": ["B", "D", "B", "A", "C"],
            "home_score": [2, 5, 0, 1, 4],
            "away_score": [4, 1, 8, 9, 2],
            "home_starter_id": ["S9", "S9", "S1", "S9", "S1"],
            "away_starter_id": ["S9", "S9", "S9", "S1", "S9"],
        })

    def test_nan_below_min_starts(self):
        """A starter with fewer than min prior starts gets NaN (not imputed)."""
        side = _side_frame(self._fixture())
        raw_map, _ = trailing_team_ladders(side, window=5, min_games=2)
        sp_adj = trailing_pitcher_ladders(side, raw_map, window=5, min_games=3)
        # S1 has < 3 prior starts everywhere in this frame -> all NaN.
        self.assertTrue(np.isnan(sp_adj[("S1", 2)]))   # 04-01 first start
        self.assertTrue(np.isnan(sp_adj[("S1", 3)]))   # 04-02 (1 prior)
        self.assertTrue(np.isnan(sp_adj[("S1", 4)]))   # 04-03 (2 prior)

    def test_adjustment_subtracts_opponent_strength(self):
        """At min 1, S1's 04-02 value = runs allowed in his ONE prior start
        (8, allowed vs B on 04-01) minus B's trailing raw strength at that
        date (B was +3 entering 04-01) -> 5.0. The raw proxy alone was 8.0,
        so the adjustment demonstrably removes opponent quality."""
        side = _side_frame(self._fixture())
        raw_map, _ = trailing_team_ladders(side, window=5, min_games=2)
        sp_adj = trailing_pitcher_ladders(side, raw_map, window=5, min_games=1)
        self.assertAlmostEqual(float(sp_adj[("S1", 3)]), 5.0, places=6)
        # The raw (unadjusted) runs-allowed proxy at the prior start was 8.0.
        self.assertNotAlmostEqual(float(sp_adj[("S1", 3)]), 8.0, places=6)


class TestFeatureAttachment(unittest.TestCase):
    def test_opp_adj_columns_present_with_coverage(self):
        """add_opponent_adjusted_features attaches all six columns and, on a
        mid-season frame, most rows carry real (non-NaN) values."""
        rng = np.random.default_rng(11)
        teams = ["A", "B", "C", "D", "E"]
        n = 120
        games = pd.DataFrame({
            "game_date": pd.date_range("2025-03-20", periods=n, freq="1D"),
            "home_team": [teams[i % 5] for i in range(n)],
            "away_team": [teams[(i + 1) % 5] for i in range(n)],
            "home_score": rng.integers(0, 9, size=n),
            "away_score": rng.integers(0, 9, size=n),
            "home_starter_id": [f"S{i % 12}" for i in range(n)],
            "away_starter_id": [f"S{(i + 5) % 12}" for i in range(n)],
            "home_win": rng.integers(0, 2, size=n),
        })
        out = add_opponent_adjusted_features(games)
        self.assertEqual([c for c in OPP_ADJ_COLS if c in out.columns],
                         OPP_ADJ_COLS)
        cov = {c: float(out[c].notna().mean()) for c in OPP_ADJ_COLS}
        # Synthetic tight-rotation frame: most rows carry real ladder values
        # (never imputed), and the pitcher ladder covers the bulk but not all.
        self.assertGreater(cov["team_talent_adj_diff"], 0.7)
        self.assertGreater(cov["sp_adj_diff"], 0.4)
        self.assertLess(cov["sp_adj_diff"], 1.0)
        # Every NaN is a genuine insufficient-history row, not a 0-fill: with
        # TEAM_MIN=5, each team's first 4 games are NaN in raw but the ladder
        # values must not be constant across all rows.
        self.assertGreater(float(out["team_talent_adj_diff"].std()), 0.0)


class TestTreesOnlyMemberFilter(unittest.TestCase):
    """The trees-only variant must restrict BOTH arms to the three tree
    members (xgb/lgbm/rf), keeping the helper keys ensemble_predict needs and
    dropping logistic + MLP, and the blend weights must renormalize over the
    surviving members so the WITH-vs-WITHOUT comparison is apples-to-apples."""

    def test_filter_drops_logistic_and_mlp_keeps_trees_and_helpers(self):
        models = {
            "xgboost": object(), "lightgbm": object(),
            "randomforest": object(), "logistic": object(), "mlp": object(),
            "scaler": object(), "impute_median": object(),
            "categorical_vocab": object(),
        }
        out = filter_tree_members(models)
        for m in TREE_MEMBERS:
            self.assertIn(m, out, f"tree member {m} must survive the filter")
        self.assertNotIn("logistic", out)
        self.assertNotIn("mlp", out)
        # Helpers ensemble_predict reads are preserved.
        self.assertIn("scaler", out)
        self.assertIn("impute_median", out)
        self.assertIn("categorical_vocab", out)
        self.assertEqual(set(out.keys()), set(TREE_MEMBERS) | {
            "scaler", "impute_median", "categorical_vocab"})

    def test_blend_weights_renormalize_over_trees_only(self):
        """With adaptive weights cleared, the blend uses static priors
        renormalized over the surviving tree members (0.25/0.25/0.10 aligned
        to sum to exactly 1.0) — identical for both arms."""
        training._LAST_ADAPTIVE_WEIGHTS.clear()
        import config
        priors = {n: config.ENSEMBLE_WEIGHTS[n] for n in TREE_MEMBERS}
        total = sum(priors.values())
        w = training._member_weights(list(TREE_MEMBERS))
        self.assertAlmostEqual(sum(w.values()), 1.0, places=9)
        for n in TREE_MEMBERS:
            self.assertGreater(w[n], 0.0)
            self.assertAlmostEqual(w[n], priors[n] / total, places=9)


if __name__ == "__main__":
    unittest.main()
