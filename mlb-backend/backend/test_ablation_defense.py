"""Unit tests for the defensive-ablation seams: pbp defensive aggregates and
the leak-safe trailing ladders (point-in-time discipline) in
ablation_defense.py.

Mirrors the repo's convention (test_opponent_adjusted_ladders.py): pure
helpers are tested in isolation; nothing here imports the full ablation main.
"""
import unittest

import numpy as np
import pandas as pd

from ablation_defense import (
    pbp_defensive_aggregates,
    trailing_team_metric,
    trailing_starter_metric,
    add_defensive_features,
    add_starter_defensive_features,
    dm_pvalue,
    RAW_WINDOW,
    RAW_MIN,
    TREND_FAST,
    TREND_SLOW,
    RAW_COLS,
    TREND_COLS,
)


def _side_frame(rows):
    """rows: (team, date, value) tuples -> the pure ladder input frame."""
    return pd.DataFrame({
        "gidx": np.arange(len(rows)),
        "date": pd.to_datetime([r[1] for r in rows]),
        "team": [r[0] for r in rows],
        "value": [float(r[2]) for r in rows],
    })


class TestPbpDefensiveAggregates(unittest.TestCase):
    def test_side_mapping_and_math(self):
        # Game 1: away bats "Top" innings, home bats "Bot" innings.
        pbp = pd.DataFrame({
            "game_pk": [1, 1, 1, 1, 1, 1, 1, 1],
            "game_type": ["R"] * 8,
            "inning_topbot": ["Top", "Top", "Top", "Top",
                              "Bot", "Bot", "Bot", "Bot"],
            "events": ["single", "field_out", "strikeout", "walk",
                       "double", "field_out", "field_error", "home_run"],
        })
        agg = pbp_defensive_aggregates(pbp)
        self.assertEqual(len(agg), 1)
        row = agg.iloc[0]
        # Away batters (Top): BIP = single + field_out = 2, outs = 1
        # -> HOME defense efficiency 0.5.
        self.assertAlmostEqual(row["home_def_eff"], 0.5)
        # Home batters (Bot): BIP = double + field_out + field_error = 3,
        # outs = 1 -> AWAY defense efficiency 1/3.
        self.assertAlmostEqual(row["away_def_eff"], 1 / 3)
        # Errors: field_error happened while home batted -> AWAY defense err.
        self.assertEqual(row["home_err"], 0)
        self.assertEqual(row["away_err"], 1)
        self.assertEqual(row["home_dp"], 0)
        self.assertEqual(row["away_dp"], 0)

    def test_double_play_events_count_for_defense(self):
        pbp = pd.DataFrame({
            "game_pk": [1, 1, 1, 1, 1],
            "game_type": ["R"] * 5,
            "inning_topbot": ["Top"] * 5,
            "events": ["grounded_into_double_play", "double_play",
                       "strikeout_double_play", "field_out", "single"],
        })
        agg = pbp_defensive_aggregates(pbp)
        row = agg.iloc[0]
        # All DP events while away batted -> HOME defense turned 3 DPs.
        self.assertEqual(row["home_dp"], 3)
        # BIP excludes the strikeout_double_play (a K, no batted ball).
        # BIP = gidp + dp + field_out + single = 4; outs on BIP = 3.
        self.assertAlmostEqual(row["home_def_eff"], 3 / 4)

    def test_non_regular_season_ignored_and_zero_bip_nan(self):
        pbp = pd.DataFrame({
            "game_pk": [1, 1, 2],
            "game_type": ["S", "R", "R"],
            "inning_topbot": ["Top", "Top", "Top"],
            "events": ["single", "walk", "walk"],
        })
        agg = pbp_defensive_aggregates(pbp)
        g1 = agg[agg["game_pk"] == 1].iloc[0]
        g2 = agg[agg["game_pk"] == 2].iloc[0]
        # The spring-training ("S") row is dropped, leaving game 1 with only
        # a walk -> zero balls in play -> def_eff NaN (never 0.0 or 1.0).
        self.assertTrue(np.isnan(g1["home_def_eff"]))
        self.assertEqual(g1["home_err"], 0)
        # Game 2 also has zero balls in play -> def_eff NaN.
        self.assertTrue(np.isnan(g2["home_def_eff"]))


class TestTrailingTeamMetric(unittest.TestCase):
    def test_strictly_prior_only(self):
        # Same team, three games; the middle game shares a date with another
        # game of the SAME team (doubleheader legs).
        side = _side_frame([
            ("BOS", "2026-04-01", 0.70),
            ("BOS", "2026-04-02", 0.60),
            ("BOS", "2026-04-02", 0.90),   # same-day leg — must NOT count
            ("BOS", "2026-04-03", 0.80),
        ])
        ladder = trailing_team_metric(side, RAW_WINDOW, 1)
        # 04-03 sees all three prior games (both 04-02 legs are strictly
        # before 04-03, so both are published by then): mean(0.70,0.60,0.90).
        self.assertAlmostEqual(ladder[("BOS", 3)], 0.7333333)
        # 04-02's second leg sees only 04-01 (its own same-day legs excluded
        # by the strict `date <` rule — they finish after its first pitch).
        self.assertAlmostEqual(ladder[("BOS", 2)], 0.70)

    def test_min_games_gate_returns_nan(self):
        side = _side_frame([
            ("BOS", "2026-04-01", 0.70),
            ("BOS", "2026-04-02", 0.60),
        ])
        ladder = trailing_team_metric(side, RAW_WINDOW, RAW_MIN)
        # Fewer than RAW_MIN prior games -> NaN, never imputed.
        for i in range(2):
            self.assertTrue(np.isnan(ladder[("BOS", i)]))

    def test_per_team_isolation(self):
        side = _side_frame([
            ("BOS", "2026-04-01", 0.70),
            ("NYY", "2026-04-01", 0.50),
            ("BOS", "2026-04-02", 0.80),
        ])
        ladder = trailing_team_metric(side, RAW_WINDOW, 1)
        self.assertAlmostEqual(ladder[("BOS", 2)], 0.70)  # only BOS prior
        # NYY's FIRST game has no prior NYY games -> NaN (ladder uses only
        # strictly-earlier rows; a game never sees itself).
        self.assertTrue(np.isnan(ladder[("NYY", 1)]))

    def test_window_truncation(self):
        vals = [0.50 + 0.01 * i for i in range(12)]
        side = _side_frame([("BOS", f"2026-04-{i+1:02d}", v)
                            for i, v in enumerate(vals)])
        ladder = trailing_team_metric(side, 5, 5)
        # Last game: window = last 5 of the 11 prior = mean(vals[6:11]).
        self.assertAlmostEqual(ladder[("BOS", 11)], np.mean(vals[6:11]))


class TestAddDefensiveFeatures(unittest.TestCase):
    def _games_and_pbp(self):
        # Three games; pbp exists for all (per_game covers game_pks 1..3).
        games = pd.DataFrame({
            "game_pk": [1, 2, 3],
            "game_date": ["2026-04-01", "2026-04-02", "2026-04-03"],
            "home_team": ["BOS", "BOS", "BOS"],
            "away_team": ["NYY", "NYY", "NYY"],
            "home_win": [1.0, 0.0, 1.0],
        })
        # home defense efficiency 0.70/0.60/0.80; away defense 0.50/0.55/0.45.
        per_game = pd.DataFrame({
            "game_pk": [1, 2, 3],
            "home_def_eff": [0.70, 0.60, 0.80],
            "away_def_eff": [0.50, 0.55, 0.45],
            "home_err": [0, 1, 0],
            "away_err": [1, 0, 1],
            "home_dp": [1, 0, 2],
            "away_dp": [0, 1, 0],
        })
        return games, per_game

    def test_ladder_is_point_in_time_and_sided(self):
        games, per_game = self._games_and_pbp()
        df = add_defensive_features(games, per_game,
                                    raw_window=RAW_WINDOW, raw_min=1,
                                    trend_fast=(15, 1), trend_slow=(60, 1))
        # Game 3's home ladder uses only games 1-2: mean(0.70, 0.60).
        self.assertAlmostEqual(df.loc[2, "home_defeff_30"], 0.65)
        # Game 2's home ladder uses only game 1.
        self.assertAlmostEqual(df.loc[1, "home_defeff_30"], 0.70)
        # Away side likewise.
        self.assertAlmostEqual(df.loc[2, "away_defeff_30"], 0.525)
        # Game 1 has no prior history -> NaN (min-gate, never imputed).
        self.assertTrue(np.isnan(df.loc[0, "home_defeff_30"]))
        # Diff = home - away.
        self.assertAlmostEqual(df.loc[2, "defeff_30_diff"], 0.65 - 0.525)
        # Errors and DPs ladder too.
        self.assertAlmostEqual(df.loc[2, "home_err_30"], 0.5)
        self.assertAlmostEqual(df.loc[2, "away_dp_30"], 0.5)

    def test_trend_is_fast_minus_slow(self):
        games, per_game = self._games_and_pbp()
        # Default min gates (RAW_MIN=10, TREND_SLOW=30) are unmet with 3 games
        # -> the trend columns are honestly NaN at the production defaults.
        df = add_defensive_features(games, per_game)
        for c in TREND_COLS:
            self.assertTrue(np.isnan(df.loc[2, c]))
        # With min-gates of 1, the trend reduces to fast(15g) - slow(60g)
        # means; with only 2 prior games both legs are the same mean -> 0.
        df2 = add_defensive_features(games, per_game,
                                     raw_window=RAW_WINDOW, raw_min=1,
                                     trend_fast=(15, 1), trend_slow=(60, 1))
        self.assertAlmostEqual(df2.loc[2, "home_defeff_tr"], 0.0)
        # Directly exercise the fast/slow seam on a case where they differ.
        side = _side_frame([
            ("BOS", "2026-04-01", 0.50), ("BOS", "2026-04-02", 0.60),
            ("BOS", "2026-04-03", 0.70), ("BOS", "2026-04-04", 0.80),
            ("BOS", "2026-04-05", 0.90),
        ])
        fast = trailing_team_metric(side, 2, 2)
        slow = trailing_team_metric(side, 4, 4)
        # Last game: fast = mean(0.80, 0.90) = 0.85; slow = mean(0.60..0.90)
        # = 0.75 -> trend +0.10 (recent defense hotter than season form).
        self.assertAlmostEqual(fast[("BOS", 4)] - slow[("BOS", 4)], 0.10)


class TestStarterLadder(unittest.TestCase):
    def test_only_own_prior_starts_point_in_time(self):
        # Same starter S on a team; a second starter P interspersed must not
        # contaminate S's behind-starter mean.
        side = pd.DataFrame({
            "gidx": np.arange(6),
            "date": pd.to_datetime([
                "2026-04-01", "2026-04-03", "2026-04-05",
                "2026-04-07", "2026-04-09", "2026-04-11"]),
            "starter": ["S", "P", "S", "S", "P", "S"],
            "value": [0.70, 0.50, 0.60, 0.80, 0.40, 0.90],
        })
        ladder = trailing_starter_metric(side, 100, 1)
        # S's ladder at gidx5 (04-11) uses only S's prior starts: 0.70, 0.60,
        # 0.80 -> mean 0.7 (04-03 and 04-09 were P's starts, excluded).
        self.assertAlmostEqual(ladder[("S", 5)], 0.7)
        # S at gidx3 uses 0.70, 0.60 -> 0.65.
        self.assertAlmostEqual(ladder[("S", 3)], 0.65)
        # P at gidx4 uses only P's gidx1 -> 0.5.
        self.assertAlmostEqual(ladder[("P", 4)], 0.5)

    def test_starter_min_starts_gate_nan(self):
        side = pd.DataFrame({
            "gidx": [0, 1, 2],
            "date": pd.to_datetime(["2026-04-01", "2026-04-03", "2026-04-05"]),
            "starter": ["S", "S", "S"],
            "value": [0.70, 0.60, 0.80],
        })
        ladder = trailing_starter_metric(side, 100, 5)
        # Fewer than SP_MIN prior starts -> NaN, never imputed.
        for i in range(3):
            self.assertTrue(np.isnan(ladder[("S", i)]))

    def test_same_day_legs_excluded(self):
        side = pd.DataFrame({
            "gidx": [0, 1, 2],
            "date": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-02"]),
            "starter": ["S", "S", "S"],
            "value": [0.70, 0.60, 0.90],
        })
        ladder = trailing_starter_metric(side, 100, 1)
        # 04-02's legs both see only 04-01 (04-02's other leg is SAME DATE and
        # excluded by the strict `date <` rule).
        self.assertAlmostEqual(ladder[("S", 2)], 0.70)
        self.assertAlmostEqual(ladder[("S", 1)], 0.70)
        # 04-01 is the starter's first observed start -> no prior -> NaN.
        self.assertTrue(np.isnan(ladder[("S", 0)]))

    def test_add_starter_features_sided_and_columns(self):
        games = pd.DataFrame({
            "game_pk": [1, 2, 3],
            "game_date": ["2026-04-01", "2026-04-03", "2026-04-05"],
            "home_team": ["BOS", "BOS", "BOS"],
            "away_team": ["NYY", "NYY", "NYY"],
            "home_starter_id": ["S1", "S2", "S1"],
            "away_starter_id": ["P1", "P1", "P1"],
            "home_win": [1.0, 0.0, 1.0],
        })
        per_game = pd.DataFrame({
            "game_pk": [1, 2, 3],
            "home_def_eff": [0.70, 0.60, 0.80],
            "away_def_eff": [0.50, 0.55, 0.45],
            "home_err": [0, 1, 0], "away_err": [1, 0, 1],
            "home_dp": [1, 0, 2], "away_dp": [0, 1, 0],
        })
        df = add_starter_defensive_features(games, per_game, window=100,
                                            min_starts=1)
        # Game 1's home side is S1's FIRST start -> no prior S1 starts -> NaN.
        self.assertTrue(np.isnan(df.loc[0, "home_defeff_sp"]))
        # Game 3's home side is S1's 2nd start -> mean(S1's prior home def_eff
        # at game 1) = 0.70 (game 2 was started by S2, excluded).
        self.assertAlmostEqual(df.loc[2, "home_defeff_sp"], 0.70)
        # Away side P1 has all three prior starts: game 3 = mean(0.50, 0.55).
        self.assertAlmostEqual(df.loc[2, "away_defeff_sp"], 0.525)
        self.assertIn("defeff_sp_diff", df.columns)
        self.assertAlmostEqual(df.loc[2, "away_err_sp"], 0.5)


class TestDmPvalue(unittest.TestCase):
    def test_identical_series_p_one(self):
        d = np.zeros(200)
        self.assertEqual(dm_pvalue(d), 1.0)

    def test_consistent_gain_is_significant(self):
        rng = np.random.default_rng(7)
        d = 0.02 + rng.normal(0, 0.01, 400)  # baseline clearly better
        self.assertLess(dm_pvalue(d), 0.05)

    def test_short_series_nan(self):
        self.assertTrue(np.isnan(dm_pvalue(np.zeros(10))))


if __name__ == "__main__":
    unittest.main()
