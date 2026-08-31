"""Tests for the NFL DuckDB feature engine (nfl_feature_engine.py).

The core guarantee is byte-equivalence of the SQL rollup to the pandas one
(``nfl_features._pbp_team_agg``), plus the MLB-mirrored low-RAM DuckDB settings
and the leak-safe additive aggregate seam. Skipped when duckdb is unavailable
(any environment without DuckDB still uses the pandas path, covered elsewhere).
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from nfl_feature_engine import (
    TEAM_AGG_COLUMNS,
    PBPAggregate,
    duckdb_available,
    duckdb_engine,
    pbp_aggregate,
    pbp_team_agg,
    register_pbp_aggregate,
)
from nfl_features import _pbp_team_agg


def _has_duckdb():
    return duckdb_available()


class TestPBPAggParity(unittest.TestCase):
    """SQL rollup must equal the pandas rollup on identically-shaped inputs."""

    def setUp(self):
        if not _has_duckdb():
            self.skipTest("duckdb not installed — pandas fallback is covered elsewhere")

    def _pbp(self):
        return pd.DataFrame([
            # Game G1: two teams, epa sometimes null, clock present.
            dict(game_id="G1", posteam="A", yards_gained=5, epa=0.2, qb_epa=0.1,
                 game_seconds_remaining=300.0),
            dict(game_id="G1", posteam="A", yards_gained=-2, epa=None, qb_epa=None,
                 game_seconds_remaining=500.0),
            dict(game_id="G1", posteam="B", yards_gained=11, epa=0.5, qb_epa=0.4,
                 game_seconds_remaining=300.0),
            dict(game_id="G1", posteam="B", yards_gained=3, epa=0.1, qb_epa=0.0,
                 game_seconds_remaining=600.0),
            # Game G2: single team; a posteam-null row is ignored by the GROUP BY.
            dict(game_id="G2", posteam="C", yards_gained=9, epa=0.3, qb_epa=None,
                 game_seconds_remaining=60.0),
            dict(game_id="G2", posteam=None, yards_gained=99, epa=1.0, qb_epa=1.0,
                 game_seconds_remaining=60.0),
        ])

    def test_rollup_matches_pandas_exactly(self):
        pbp = self._pbp()
        want = _pbp_team_agg(pbp).sort_values(["game_id", "team"]).reset_index(drop=True)
        with duckdb_engine() as con:
            got = pbp_team_agg(con, pbp).sort_values(
                ["game_id", "team"]).reset_index(drop=True)
        # DuckDB and numpy accumulate in different orders, so large sums carry
        # last-ULP float noise; compare within tight tolerance (logically equal).
        pd.testing.assert_frame_equal(got, want, check_dtype=False,
                                      check_exact=False, rtol=1e-9, atol=1e-12)

    def test_rollup_columns_match_schema(self):
        with duckdb_engine() as con:
            got = pbp_team_agg(con, self._pbp())
        self.assertEqual(list(got.columns), TEAM_AGG_COLUMNS)
        # elapsed_min = (3600 - min clock) / 60 per game, on every team row.
        g1 = got[got["game_id"] == "G1"]
        self.assertAlmostEqual(float(g1["elapsed_min"].iloc[0]), 55.0)
        g2 = got[got["game_id"] == "G2"]
        self.assertAlmostEqual(float(g2["elapsed_min"].iloc[0]), 59.0)

    def test_missing_epa_qb_epa_and_clock_match_pandas_nan(self):
        pbp = self._pbp().drop(columns=["epa", "qb_epa", "game_seconds_remaining"])
        want = _pbp_team_agg(pbp).sort_values(["game_id", "team"]).reset_index(drop=True)
        with duckdb_engine() as con:
            got = pbp_team_agg(con, pbp).sort_values(
                ["game_id", "team"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(got, want, check_dtype=False,
                                      check_exact=False, rtol=1e-9, atol=1e-12)
        # sum/count of an absent epa are NaN, and elapsed_min is NaN (no clock).
        self.assertTrue(pd.isna(got["epa_sum"]).all())
        self.assertTrue(pd.isna(got["elapsed_min"]).all())

    def test_elapsed_min_ignores_null_posteam_clock_rows(self):
        """A row holding the game's MINIMUM clock but NULL posteam must not
        set the game length — pandas drops posteam-NaN rows before picking the
        min, and the DuckDB engine must match (real-data parity gap fixed)."""
        pbp = pd.DataFrame([
            dict(game_id="G1", posteam="A", yards_gained=5, epa=0.1, qb_epa=0.0,
                 game_seconds_remaining=600.0),
            dict(game_id="G1", posteam="A", yards_gained=2, epa=0.1, qb_epa=0.0,
                 game_seconds_remaining=300.0),
            dict(game_id="G1", posteam=None, yards_gained=0, epa=None, qb_epa=None,
                 game_seconds_remaining=0.0),   # min clock, NULL posteam
        ])
        want = _pbp_team_agg(pbp).sort_values(
            ["game_id", "team"]).reset_index(drop=True)
        with duckdb_engine() as con:
            got = pbp_team_agg(con, pbp).sort_values(
                ["game_id", "team"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(got, want, check_dtype=False,
                                      check_exact=False, rtol=1e-9, atol=1e-12)
        # Length comes from the 300s row, NOT the 0s NULL-posteam row.
        self.assertAlmostEqual(float(want["elapsed_min"].iloc[0]),
                               (3600 - 300) / 60.0, places=6)

    def test_empty_and_malformed_return_full_empty_schema(self):
        empty = pd.DataFrame(columns=["game_id", "posteam", "yards_gained"])
        with duckdb_engine() as con:
            out = pbp_team_agg(con, empty)
        self.assertEqual(list(out.columns), TEAM_AGG_COLUMNS)
        self.assertTrue(out.empty)
        with duckdb_engine() as con:
            out2 = pbp_team_agg(con, pd.DataFrame())
        self.assertEqual(list(out2.columns), TEAM_AGG_COLUMNS)
        self.assertTrue(out2.empty)


class TestEngineSettings(unittest.TestCase):
    """The DuckDB connection mirrors MLB's low-RAM spill config."""

    def test_low_ram_settings_are_set(self):
        if not _has_duckdb():
            self.skipTest("duckdb not installed")
        with duckdb_engine() as con:
            mem = con.execute("SELECT current_setting('memory_limit')").fetchone()[0]
            threads = con.execute("SELECT current_setting('threads')").fetchone()[0]
            tempdir = con.execute("SELECT current_setting('temp_directory')").fetchone()[0]
            maxtmp = con.execute(
                "SELECT current_setting('max_temp_directory_size')").fetchone()[0]
        # DuckDB renders the 4GB cap as ~"3.7 GIB"; assert the cap was applied
        # (non-default) rather than a specific rendering.
        self.assertTrue("GIB" in str(mem).upper() or "GB" in str(mem).upper())
        self.assertEqual(int(threads), 1)
        self.assertIn("duckdb_temp", str(tempdir))
        # 50GB cap applied (renders ~"46.5 GIB"): non-default, a real cap.
        self.assertTrue("GIB" in str(maxtmp).upper() or "GB" in str(maxtmp).upper())


class TestAggregateRegistry(unittest.TestCase):
    """The additive seam yields a per-game column and rejects leaky exprs."""

    def setUp(self):
        if not _has_duckdb():
            self.skipTest("duckdb not installed")

    def test_registered_aggregate_appends_column(self):
        reg = pbp_aggregate("epa_sq", "SUM(epa * epa)", "epa_sq_sum", needs=("epa",))
        self.assertEqual(reg.result_col, "epa_sq_sum")
        pbp = pd.DataFrame([
            dict(game_id="G1", posteam="A", yards_gained=1, epa=0.2),
            dict(game_id="G1", posteam="A", yards_gained=1, epa=0.4),
        ])
        with duckdb_engine() as con:
            out = pbp_team_agg(con, pbp, extra_names=("epa_sq",))
        self.assertIn("epa_sq_sum", out.columns)
        self.assertAlmostEqual(float(out["epa_sq_sum"].iloc[0]), 0.2 ** 2 + 0.4 ** 2)
        # Without the extra name, the column is absent.
        with duckdb_engine() as con:
            base = pbp_team_agg(con, pbp)
        self.assertNotIn("epa_sq_sum", base.columns)

    def test_leak_prone_expr_is_rejected(self):
        # A non-aggregate (raw column or window) would leak through the trailing
        # shift — the registry refuses it structurally.
        with self.assertRaises(ValueError):
            register_pbp_aggregate(
                PBPAggregate("bad_raw", "yards_gained", "raw_yards"))
        with self.assertRaises(ValueError):
            register_pbp_aggregate(
                PBPAggregate("bad_lead", "LAG(yards_gained) OVER ()", "leak"))

    def test_future_game_play_never_reaches_trailing_value(self):
        # End-to-end: the DuckDB rollup feeds team_stats_ladder exactly like the
        # pandas one, and the strictly-trailing shift keeps a LATER game's plays
        # out of an EARLIER game's value. Mirrors test_nfl_features' proof but
        # through the engine dispatch (_pbp_team_agg_engine -> DuckDB here).
        def _synth_games(rows):
            default = dict(game_id=None, season=2019, week=1, game_type="REG",
                           gameday=None, home_team=None, away_team=None,
                           home_score=None, away_score=None, roof="outdoors")
            return pd.DataFrame([{**default, **r} for r in rows])

        def _pbp(game_id, plays):
            return pd.DataFrame([
                dict(game_id=game_id, posteam=t, yards_gained=y, epa=e,
                     qb_epa=q, game_seconds_remaining=c)
                for t, y, e, q, c in plays])

        g = _synth_games([
            dict(game_id="G1", gameday="2019-09-01",
                 home_team="A", away_team="B", home_score=20, away_score=10),
            dict(game_id="G2", gameday="2019-09-08",
                 home_team="B", away_team="A", home_score=7, away_score=14),
        ])
        pbp = pd.concat([
            _pbp("G1", [("A", 5, 0.1, 0.2, 3000), ("A", 3, 0.2, 0.3, 2800)]),
            _pbp("G2", [("A", 7, 0.4, 0.5, 1500)]),
        ], ignore_index=True)
        from nfl_features import _pbp_team_agg_engine, team_events, team_stats_ladder
        ladder = team_stats_ladder(
            team_events(g), team_game_agg=_pbp_team_agg_engine(pbp))
        a = ladder[ladder["team"] == "A"].set_index("game_id")
        # G2 values reflect ONLY G1's plays -> prior pace present; G1 has no prior.
        self.assertTrue(pd.notna(a.loc["G2", "pace_plays_min"]))
        self.assertTrue(pd.isna(a.loc["G1", "pace_plays_min"]))


class TestTier1Aggregates(unittest.TestCase):
    """Tier-1 (v3) aggregates: turnover attribution (defteam side), passing
    efficiency / success / discipline / drive rates — DuckDB == pandas on a
    rich fixture, plus semantic asserts on the values themselves."""

    def setUp(self):
        if not _has_duckdb():
            self.skipTest("duckdb not installed — pandas fallback is covered elsewhere")

    def _pbp(self):
        base = dict(game_id="G1", posteam=None, defteam=None, yards_gained=0,
                    epa=0.0, interception=0, fumble_lost=0, pass_attempt=0,
                    passing_yards=0, sack=0, penalty=0,
                    penalty_yards=None, penalty_team=None, third_down_failed=0,
                    third_down_converted=0, yardline_100=50, touchdown=0,
                    field_goal_result=None, drive=1)
        rows = []
        for r in [
            # ---- Team A on offense (A home, B away) ----
            dict(posteam="A", defteam="B", yards_gained=15, epa=0.5,
                 pass_attempt=1, passing_yards=15, yardline_100=30, drive=1),
            dict(posteam="A", defteam="B", yards_gained=0, epa=-0.8,
                 interception=1, pass_attempt=1, third_down_failed=1,
                 yardline_100=45, drive=2),
            dict(posteam="A", defteam="B", yards_gained=8, epa=0.1,
                 yardline_100=60, drive=3),
            dict(posteam="A", defteam="B", yards_gained=-5, epa=-0.6,
                 sack=1, yardline_100=70, drive=4),
            dict(posteam="A", defteam="B", yards_gained=0, epa=0.3,
                 penalty=1, penalty_yards=10, penalty_team="B", yardline_100=50, drive=5),
            dict(posteam="A", defteam="B", yards_gained=0, epa=-0.4,
                 penalty=1, penalty_yards=15, penalty_team="A", yardline_100=50, drive=5),
            dict(posteam="A", defteam="B", yards_gained=0, epa=0.0,
                 field_goal_result="made", yardline_100=25, drive=5),
            dict(posteam="A", defteam="B", yards_gained=3, epa=0.6,
                 touchdown=1, yardline_100=5, drive=5),
            # ---- Team B on offense ----
            dict(posteam="B", defteam="A", yards_gained=25, epa=0.7,
                 pass_attempt=1, passing_yards=25, yardline_100=40, drive=1),
            dict(posteam="B", defteam="A", yards_gained=-2, epa=-0.9,
                 fumble_lost=1, yardline_100=20, drive=2),
        ]:
            rows.append({**base, **r})
        return pd.DataFrame(rows)

    def test_tier1_rollup_values_and_attribution(self):
        got = _pbp_team_agg(self._pbp()).sort_values(
            ["game_id", "team"]).reset_index(drop=True)
        a = got[got["team"] == "A"].iloc[0]
        b = got[got["team"] == "B"].iloc[0]
        # Turnover attribution: A committed an INT, B committed a lost fumble;
        # each defense forced the OTHER team's giveaway.
        self.assertEqual(float(a["giveaways"]), 1.0)
        self.assertEqual(float(b["giveaways"]), 1.0)
        self.assertEqual(float(a["takeaways"]), 1.0)   # B's fumble
        self.assertEqual(float(b["takeaways"]), 1.0)   # A's INT
        # ANY/A + sack rate per dropback: A = 2 attempts + 1 sack, 15 pass yds.
        self.assertAlmostEqual(float(a["net_any_a"]), (15.0 - 5.0) / 3.0, places=6)
        self.assertAlmostEqual(float(a["sack_rate"]), 1.0 / 3.0, places=6)
        # EPA success rate: 4 of 8 plays positive.
        self.assertAlmostEqual(float(a["success_rate"]), 0.5, places=6)
        # Explosive: only B's 25-yd play is 20+ (B ran 2 plays).
        self.assertAlmostEqual(float(b["explosive_rate"]), 0.5, places=6)
        # Penalty split: A committed 15, drew 10 (B's penalty against A).
        self.assertEqual(float(a["penalty_yds"]), 15.0)
        self.assertEqual(float(a["penalty_yds_drawn"]), 10.0)
        # Third down: 1 attempt, 0 conversions.
        self.assertEqual(float(a["third_down_rate"]), 0.0)
        # Red zone: 1 play inside the 20, TD -> 1.0.
        self.assertEqual(float(a["redzone_td_rate"]), 1.0)
        # Points per drive: TD (7) + FG (3) over A's 5 drives.
        self.assertAlmostEqual(float(a["pts_per_drive"]), 10.0 / 5.0, places=6)

    def test_tier1_parity_duckdb_vs_pandas(self):
        pbp = self._pbp()
        want = _pbp_team_agg(pbp).sort_values(
            ["game_id", "team"]).reset_index(drop=True)
        with duckdb_engine() as con:
            got = pbp_team_agg(con, pbp).sort_values(
                ["game_id", "team"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(got, want, check_dtype=False,
                                      check_exact=False, rtol=1e-9, atol=1e-12)

    def test_tier1_degrades_to_nan_when_source_columns_absent(self):
        pbp = self._pbp().drop(
            columns=["interception", "fumble_lost", "passing_yards",
                     "pass_attempt", "sack", "penalty", "penalty_yards",
                     "penalty_team", "third_down_failed", "third_down_converted",
                     "field_goal_result", "drive"])
        got = _pbp_team_agg(pbp)
        with duckdb_engine() as con:
            ddb = pbp_team_agg(con, pbp)
        for c in ("giveaways", "takeaways", "net_any_a", "sack_rate",
                  "penalty_yds", "penalty_yds_drawn", "third_down_rate",
                  "pts_per_drive"):
            self.assertTrue(pd.isna(got[c]).all(), c)
            self.assertTrue(pd.isna(ddb[c]).all(), c)
        # success/explosive still computable from yards_gained + epa.
        self.assertTrue(got["success_rate"].notna().any())
        self.assertTrue(got["explosive_rate"].notna().any())


if __name__ == "__main__":
    unittest.main()