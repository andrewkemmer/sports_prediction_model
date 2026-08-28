"""Regression tests for the pitcher ERA/K9 window semantics in features.py.

Runs the ACTUAL SQL from features.py (extracted at import time) against a
DuckDB instance with synthetic pitcher_game_stats rows, verifying:

  * sp_era / sp_k9 are true season-to-date (per-season cumulative through
    prior in-season starts only; NULL for a season's opening start, so the
    prior October never leaks in).
  * sp_era_5g / sp_k9_5g roll over the LAST 5 PRIOR STARTS ACROSS SEASONS
    (no season-start gap) and are NULL only for a career debut.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import duckdb
import pandas as pd

_FEATURES_SRC = Path(__file__).with_name("features.py").read_text(encoding="utf-8")


def _extract_sql(table: str) -> str:
    """Pull the exact CREATE TABLE <table> block out of features.py."""
    m = re.search(
        rf"CREATE TABLE {re.escape(table)} AS.*?\"\"\"",
        _FEATURES_SRC, re.DOTALL,
    )
    assert m, f"CREATE TABLE {table} not found in features.py"
    return m.group(0).rstrip().rstrip('"').strip()


def _fresh_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE pitcher_game_stats AS
        SELECT * FROM (VALUES
            -- Vet: three 2025 starts, six 2026 starts
            (DATE '2025-09-01', 9001, 101, 5.0, 2, 6, 1, 0, 4, 1, 0, 0.30, 0.10, 0.40),
            (DATE '2025-09-10', 9002, 101, 6.0, 3, 8, 2, 0, 5, 1, 0, 0.28, 0.08, 0.38),
            (DATE '2025-10-01', 9003, 101, 5.0, 1, 7, 1, 0, 3, 0, 0, 0.26, 0.06, 0.35),
            (DATE '2026-03-28', 1001, 101, 6.0, 2, 5, 1, 0, 4, 1, 0, 0.31, 0.09, 0.42),
            (DATE '2026-04-05', 1002, 101, 7.0, 4, 9, 2, 0, 6, 1, 0, 0.29, 0.07, 0.39),
            (DATE '2026-04-12', 1003, 101, 5.0, 0, 10, 1, 0, 2, 0, 0, 0.24, 0.04, 0.30),
            (DATE '2026-04-19', 1004, 101, 6.0, 3, 6, 1, 0, 5, 1, 0, 0.27, 0.08, 0.36),
            (DATE '2026-04-26', 1005, 101, 6.0, 2, 7, 2, 0, 4, 0, 0, 0.25, 0.05, 0.33),
            (DATE '2026-05-03', 1006, 101, 5.0, 5, 4, 1, 0, 6, 2, 0, 0.33, 0.11, 0.44),
            -- Rookie: career debut in 2026
            (DATE '2026-04-01', 1101, 202, 6.0, 3, 5, 1, 0, 4, 1, 0, 0.32, 0.10, 0.41),
            (DATE '2026-04-08', 1102, 202, 5.0, 2, 6, 1, 0, 3, 0, 0, 0.30, 0.07, 0.37)
        ) AS t(game_date, game_pk, pitcher, ip, runs, ks, bbs, hbps,
                hits_allowed, hrs_allowed, n_batters_faced, xwoba,
                barrel_rate, hard_contact_rate)
    """)
    return con


class TestPitcherWindows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        con = _fresh_con()
        # Build the full chain with features.py's own SQL: cross-season LAGs
        # (pitcher_shifted, pitcher_rolling) + season windows + 5g window.
        for tbl in ("pitcher_shifted", "pitcher_rolling",
                    "pitcher_shifted_season", "pitcher_season_rolling",
                    "pitcher_5g_rolling", "pitcher_season_features"):
            con.execute(_extract_sql(tbl))
        cls.df = con.execute(
            "SELECT * FROM pitcher_season_features ORDER BY pitcher, game_date"
        ).fetchdf()
        con.close()

    def _row(self, pitcher: int, game_pk: int):
        r = self.df[(self.df["pitcher"] == pitcher)
                    & (self.df["game_pk"] == game_pk)]
        self.assertEqual(len(r), 1, f"row pitcher={pitcher} game_pk={game_pk}")
        return r.iloc[0]

    def test_season_to_date_starts_null_at_opening(self):
        # Vet's first 2026 start: no prior 2026 starts -> season-to-date NULL
        # (no October 2025 leakage into the new season's cumulative).
        r = self._row(101, 1001)
        self.assertTrue(pd.isna(r["sp_era"]))
        self.assertTrue(pd.isna(r["sp_k9"]))
        # Rookie's career debut: also NULL.
        r = self._row(202, 1101)
        self.assertTrue(pd.isna(r["sp_era"]))

    def test_season_to_date_accumulates_within_season(self):
        # Vet's 2nd 2026 start: only his opening-day 6 IP / 2 R / 5 K count.
        r = self._row(101, 1002)
        self.assertAlmostEqual(r["sp_era"], 2 / 6 * 9.0, places=4)
        self.assertAlmostEqual(r["sp_k9"], 5 / 6 * 9.0, places=4)
        # 3rd 2026 start: 2 starts (6 IP/2 R + 7 IP/4 R; 5+9 K).
        r = self._row(101, 1003)
        self.assertAlmostEqual(r["sp_era"], 6 / 13 * 9.0, places=4)
        self.assertAlmostEqual(r["sp_k9"], 14 / 13 * 9.0, places=4)

    def test_5g_rolls_across_seasons_no_gap(self):
        # Vet's first 2026 start has exactly 3 prior starts (all 2025) — the
        # 5g ERA MUST use them (no season-start gap, no null guard).
        r = self._row(101, 1001)
        self.assertAlmostEqual(r["sp_era_5g"], 6 / 16 * 9.0, places=4)  # 2+3+1 R / 5+6+5 IP
        self.assertAlmostEqual(r["sp_k9_5g"], 21 / 16 * 9.0, places=4)  # 6+8+7 K
        # 2nd 2026 start: last 4 prior starts (3x 2025 + opening day).
        r = self._row(101, 1002)
        self.assertAlmostEqual(r["sp_era_5g"], 8 / 22 * 9.0, places=4)
        # 3rd 2026 start: 5-row frame = starts 1-5 (09-01 .. 04-05).
        r = self._row(101, 1003)
        self.assertAlmostEqual(r["sp_era_5g"], 12 / 29 * 9.0, places=4)

    def test_5g_null_only_for_career_debut(self):
        r = self._row(202, 1101)
        self.assertTrue(pd.isna(r["sp_era_5g"]))
        r = self._row(202, 1102)
        self.assertAlmostEqual(r["sp_era_5g"], 3 / 6 * 9.0, places=4)


if __name__ == "__main__":
    unittest.main()
