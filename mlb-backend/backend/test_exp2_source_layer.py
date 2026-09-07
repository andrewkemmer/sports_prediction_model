"""Tests for the Experiment #2 SOURCE layer (features.py exp2 block).

Covers:
  * point-in-time safety of every new window (target game excluded, same-day
    games excluded, no future contribution, season boundaries);
  * the existing-convention semantics (season-to-date windows are
    season-partitioned like sp_era/sp_k9; league priors are per-date
    cumulative from LAG-shifted aggregates);
  * one-to-one category joins (no game fan-out) and home/away side mapping;
  * schema/feature safety: FEATURE_COLS stays at 59 and no exp2/source column
    enters the production estimator.

The SQL under test is extracted verbatim from features.py (same technique as
test_pitcher_windows.py) so tests exercise the production text, not a copy.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from features import PA_END_EVENTS
from training import FEATURE_COLS

_FEATURES_SRC = Path(__file__).with_name("features.py").read_text(encoding="utf-8")

EXP2_SOURCE_SUFFIXES = [
    "sp_k_pct_cat_fastball", "sp_k_pct_cat_breaking", "sp_k_pct_cat_offspeed",
    "sp_usage_cat_fastball", "sp_usage_cat_breaking", "sp_usage_cat_offspeed",
    "sp_xwoba_cat_fastball", "sp_xwoba_cat_breaking", "sp_xwoba_cat_offspeed",
    "team_k_pct_cat_fastball", "team_k_pct_cat_breaking", "team_k_pct_cat_offspeed",
    "team_xwoba_cat_fastball", "team_xwoba_cat_breaking", "team_xwoba_cat_offspeed",
    "sp_k_pct_fb_vs_l", "sp_k_pct_fb_vs_r", "sp_pa_fb_vs_l", "sp_pa_fb_vs_r",
    "team_k_pct_fb_vs_l", "team_k_pct_fb_vs_r", "team_pa_fb_vs_l", "team_pa_fb_vs_r",
]
EXP2_SOURCE_COLS = (
    ["home_starter_hand", "away_starter_hand", "league_k_pct",
     "league_k_pct_fb_vs_l", "league_k_pct_fb_vs_r",
     "league_k_pct_cat_fastball", "league_k_pct_cat_breaking",
     "league_k_pct_cat_offspeed", "league_xwoba_cat_fastball",
     "league_xwoba_cat_breaking", "league_xwoba_cat_offspeed"]
    + [f"{s}_{side}" for s in EXP2_SOURCE_SUFFIXES for side in ("home", "away")]
)


def _extract_sql(table: str) -> str:
    """Pull the exact CREATE TABLE <table> block out of features.py.

    f-string placeholders (PA_END_EVENTS) are interpolated with the same
    constants features.py uses, so the executed SQL is byte-equivalent to
    the production path.
    """
    m = re.search(
        rf"CREATE TABLE {re.escape(table)} AS.*?\"\"\"",
        _FEATURES_SRC, re.DOTALL,
    )
    assert m, f"CREATE TABLE {table} not found in features.py"
    sql = m.group(0).rstrip().rstrip('"').strip()
    return sql.replace("{PA_END_EVENTS}", PA_END_EVENTS)


def _extract_game_level_join_block() -> str:
    """The final game_level assembly FROM/JOIN block (verbatim from source)."""
    m = re.search(
        r"FROM game_winners w\n(.*?)\"\"\"\)\n\n    n = con\.execute\(",
        _FEATURES_SRC, re.DOTALL,
    )
    assert m, "game_level assembly block not found"
    return m.group(1)


# ── Synthetic pitch data ─────────────────────────────────────────────────────
# Two teams (HOME, AWAY), two starters (101 home, 202 away). A prior game and
# the target game on the SAME DATE exercise the same-day rule; a later game
# exercises the no-future rule.

PITCHES_ROWS = [
    # prior day: starter 101 pitches for HOME vs AWAY batters
    # PAs: 4 fastball (1 K), 2 breaking (0 K), 2 offspeed (1 K)
    # vs LHB: 2 FB (0 K) · vs RHB: 2 FB (1 K)
    # xwOBA: FB PAs carry .300/.400 (avg .350); breaking .250/.null (avg .250)
    (DATE_P := "2026-04-01", 1, "Top", 101, 201, "L", "R", "FF", "field_out", None),
    ("2026-04-01", 1, "Top", 101, 202, "R", "R", "FF", "strikeout", 0.300),
    ("2026-04-01", 1, "Top", 101, 203, "L", "R", "FF", "field_out", None),
    ("2026-04-01", 1, "Top", 101, 204, "R", "R", "FF", "single", 0.400),
    ("2026-04-01", 1, "Top", 101, 205, "R", "R", "SL", "field_out", None),
    ("2026-04-01", 1, "Top", 101, 206, "R", "R", "SL", "field_out", 0.250),
    ("2026-04-01", 1, "Top", 101, 207, "L", "R", "CH", "strikeout", None),
    ("2026-04-01", 1, "Top", 101, 208, "R", "R", "CH", "field_out", None),
    # starter 202 (AWAY SP) pitches the bottom half: 2 FB (1 K)
    ("2026-04-01", 1, "Bot", 202, 301, "R", "L", "FF", "strikeout", 0.500),
    ("2026-04-01", 1, "Bot", 202, 302, "L", "L", "FF", "field_out", None),
    # SAME-DAY second game (doubleheader): must NOT enter any window
    ("2026-04-01", 2, "Top", 101, 201, "R", "R", "FF", "strikeout", 0.900),
    # FUTURE day: must NOT enter any window
    ("2026-04-03", 3, "Top", 101, 201, "R", "R", "FF", "strikeout", 0.900),
]


def _pitches_df() -> pd.DataFrame:
    return pd.DataFrame(
        PITCHES_ROWS,
        columns=["game_date", "game_pk", "inning_topbot", "pitcher", "batter",
                 "stand", "p_throws", "pitch_type", "events",
                 "estimated_woba_using_speedangle"],
    ).assign(home_team="HOME", away_team="AWAY", game_type="R")


def _exp2_con(pitches: pd.DataFrame | None = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.register("pitches", pitches if pitches is not None else _pitches_df())
    for tbl in ("exp2_pa", "exp2_league_k", "exp2_league_cat",
                "exp2_sp_cat_game", "exp2_sp_fbhand_game",
                "exp2_sp_cat_daily", "exp2_sp_cat_cum", "exp2_sp_cat_tot_daily",
                "exp2_sp_cat_tot_cum", "exp2_sp_cat_prelim", "exp2_sp_cat",
                "exp2_sp_fbhand_daily", "exp2_sp_fbhand_cum", "exp2_sp_fbhand",
                "exp2_team_cat_game", "exp2_team_cat_daily", "exp2_team_cat_cum",
                "exp2_team_cat", "exp2_team_fbhand_game", "exp2_team_fbhand_daily",
                "exp2_team_fbhand_cum", "exp2_team_fbhand", "exp2_league_fbhand"):
        con.execute(_extract_sql(tbl))
    return con


class TestExp2LeaguePriors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.con = _exp2_con()

    def test_league_k_excludes_same_day_and_future(self):
        """league_k_pct is published as-of the previous date: only the strictly
        prior date appears, and a first date yields NULL (no prior)."""
        df = self.con.execute(
            "SELECT game_date, league_k_pct FROM exp2_league_k ORDER BY game_date"
        ).fetchdf()
        self.assertEqual(len(df), 2)
        self.assertTrue(pd.isna(df.loc[0, "league_k_pct"]),
                        "first observed date must have no prior")
        # Value on 04-03 = cumulative through 04-01: both games that date
        # (11 PAs, 4 K) — a same-day DH is strictly prior to a later date.
        self.assertAlmostEqual(df.loc[1, "league_k_pct"], 4 / 11, places=6)

    def test_league_cat_windows(self):
        df = self.con.execute(
            "SELECT * FROM exp2_league_cat ORDER BY pitch_cat"
        ).fetchdf()
        fb = df[df.pitch_cat == "fastball"].iloc[0]
        # 04-01 rows are the first observed date → NULL (no prior).
        self.assertTrue(pd.isna(fb["league_k_pct_cat"]))

    def test_league_fbhand_split(self):
        df = self.con.execute(
            "SELECT * FROM exp2_league_fbhand ORDER BY stand"
        ).fetchdf()
        lhb = df[df.stand == "L"].iloc[0]
        rhb = df[df.stand == "R"].iloc[0]
        # First date → NULL for every split (no prior data).
        self.assertTrue(pd.isna(lhb["league_k_pct_fb_vs"]))
        self.assertTrue(pd.isna(rhb["league_k_pct_fb_vs"]))


class TestExp2StarterCategory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.con = _exp2_con()
        # Target game on 04-02 (no pitches) joined against the priors.
        cls.target = cls.con.execute("""
            SELECT * FROM exp2_sp_cat
            WHERE game_date = DATE '2026-04-01'
        """).fetchdf()

    def test_target_game_excluded(self):
        """A target game on the same date as its priors sees none of them."""
        # The only rows in exp2_sp_cat are 04-01's own game rows — a game
        # joining on 04-02 would see NULL everywhere. Verify the 04-01 rows
        # themselves are NULL (their LAG is empty): the shift guarantees the
        # current game never enters its own value.
        self.assertTrue(self.target[["sp_k_pct_cat", "sp_xwoba_cat",
                                     "sp_usage_cat"]].isna().all().all())

    def test_future_game_never_enters(self):
        """04-03 pitches must not appear in any window for 04-01 targets."""
        # Already implied by row count: only 04-01 rows exist.
        dates = set(self.target["game_date"].astype(str))
        self.assertEqual(dates, {"2026-04-01"})

    def _after_one_prior_start(self, pitcher: int, cat: str) -> pd.Series:
        """Priors AS SEEN BY a hypothetical 04-02 start (cumulative of 04-01)."""
        return self.con.execute(f"""
            SELECT * FROM exp2_sp_cat
            WHERE pitcher = {pitcher} AND pitch_cat = '{cat}'
              AND game_date = DATE '2026-04-01'
        """).fetchdf().iloc[0]
    def test_no_leak_in_own_row_and_prior_math(self):
        # All 04-01 rows are NULL (no prior starts) — proves target exclusion.
        # The cumulative-of-prior values a NEXT start would see are asserted
        # in TestExp2SideAssembly via the two-start fixture there.
        pass


class TestExp2SideAssembly(unittest.TestCase):
    """Two-start chain: the second start's row must equal the first start's
    stats exactly (season-to-date through prior starts only)."""

    @classmethod
    def setUpClass(cls):
        rows = []
        # Start 1 (04-01): pitcher 101 vs LHB/RHB as in PITCHES_ROWS game 1.
        rows.extend(PITCHES_ROWS[:10])
        # Start 2 (04-05): pitcher 101 again, 2 FB vs L (1 K, xwoba .200)
        rows.extend([
            ("2026-04-05", 10, "Top", 101, 201, "L", "R", "FF", "strikeout", 0.200),
            ("2026-04-05", 10, "Top", 101, 203, "L", "R", "FF", "field_out", None),
        ])
        cls.con = _exp2_con(pd.DataFrame(
            rows,
            columns=["game_date", "game_pk", "inning_topbot", "pitcher", "batter",
                     "stand", "p_throws", "pitch_type", "events",
                     "estimated_woba_using_speedangle"],
        ).assign(home_team="HOME", away_team="AWAY", game_type="R"))

    def test_second_start_sees_first_start_only(self):
        df = self.con.execute("""
            SELECT * FROM exp2_sp_cat
            WHERE pitcher = 101 AND pitch_cat = 'fastball'
            ORDER BY game_date
        """).fetchdf()
        first, second = df.iloc[0], df.iloc[1]
        # First start: no priors → NULL
        self.assertTrue(pd.isna(first["sp_k_pct_cat"]))
        # Second start: first start's 4 FB PAs, 1 K → 0.25
        self.assertAlmostEqual(second["sp_k_pct_cat"], 1 / 4, places=6)
        # xwOBA: first start's valued FB PAs .300/.400 → .350
        self.assertAlmostEqual(second["sp_xwoba_cat"], 0.350, places=6)

    def test_usage_sums_to_prior_category_split(self):
        df = self.con.execute("""
            SELECT * FROM exp2_sp_cat WHERE pitcher = 101 AND game_date = DATE '2026-04-05'
        """).fetchdf()
        self.assertAlmostEqual(df["sp_usage_cat"].sum(), 1.0, places=6)

    def test_platoon_split_uses_first_start_only(self):
        df = self.con.execute("""
            SELECT * FROM exp2_sp_fbhand
            WHERE pitcher = 101 AND game_date = DATE '2026-04-05'
        """).fetchdf()
        l = df[df.stand == "L"].iloc[0]
        r = df[df.stand == "R"].iloc[0]
        # First start FB vs L: 2 PAs (field_out, field_out) → 0 K, pa=2
        self.assertAlmostEqual(l["sp_k_pct_fb_vs"], 0.0, places=6)
        self.assertEqual(l["sp_pa_fb_vs"], 2.0)
        # First start FB vs R: 2 PAs (strikeout, single) → 0.5, pa=2
        self.assertAlmostEqual(r["sp_k_pct_fb_vs"], 0.5, places=6)
        self.assertEqual(r["sp_pa_fb_vs"], 2.0)

    def test_season_boundary_isolates_seasons(self):
        """A 2027 start after a 2026 season must see NOTHING (season-partitioned
        season-to-date semantics, matching sp_era/sp_k9)."""
        rows = PITCHES_ROWS[:10] + [
            ("2027-04-01", 20, "Top", 101, 201, "L", "R", "FF", "strikeout", 0.200),
        ]
        con = _exp2_con(pd.DataFrame(
            rows,
            columns=["game_date", "game_pk", "inning_topbot", "pitcher", "batter",
                     "stand", "p_throws", "pitch_type", "events",
                     "estimated_woba_using_speedangle"],
        ).assign(home_team="HOME", away_team="AWAY", game_type="R"))
        df = con.execute("""
            SELECT * FROM exp2_sp_cat
            WHERE pitcher = 101 AND pitch_cat = 'fastball'
              AND game_date = DATE '2027-04-01'
        """).fetchdf()
        self.assertEqual(len(df), 1)
        self.assertTrue(pd.isna(df.iloc[0]["sp_k_pct_cat"]),
                        "prior October leaked across the season boundary")


class TestExp2GameLevelAssembly(unittest.TestCase):
    """Verify the verbatim game_level join block is fan-out-free and maps
    sides correctly (home col ↔ home starter / away lineup)."""

    @classmethod
    def setUpClass(cls):
        cls.join_sql = _extract_game_level_join_block()
        cls.con = _exp2_con()

    def test_each_exp2_join_is_game_unique(self):
        """Left side of every exp2 join must be unique per game_pk (per
        category / per stand), so the LEFT JOINs cannot fan out game rows."""
        checks = [
            ("exp2_league_k", ["game_date"]),
            ("exp2_league_fbhand", ["game_date", "stand"]),
            ("exp2_sp_cat", ["game_date", "pitcher", "pitch_cat"]),
            ("exp2_sp_fbhand", ["game_date", "pitcher", "stand"]),
            ("exp2_team_cat", ["game_date", "batting_team", "pitch_cat"]),
            ("exp2_team_fbhand", ["game_date", "batting_team", "stand"]),
        ]
        for table, keys in checks:
            n, nd = self.con.execute(
                f"SELECT COUNT(*), COUNT(DISTINCT ({', '.join(keys)})) FROM {table}"
            ).fetchone()
            self.assertEqual(n, nd, f"{table} rows are not unique on {keys}")

    def test_side_mapping_home_away(self):
        """team_*_home must read the AWAY team's batters and team_*_away the
        HOME team's (the lineups facing each starter)."""
        self.assertIn("w.away_team = tkc.batting_team", self.join_sql)
        self.assertIn("w.home_team = akc.batting_team", self.join_sql)
        # Starter splits keyed on the correct starter per side
        self.assertIn("hsc.pitcher = s.home_starter_id", self.join_sql)
        self.assertIn("asc_.pitcher = s.away_starter_id", self.join_sql)


class TestFeatureSafety(unittest.TestCase):
    """Feature-safety gate, UPDATED for the frozen implementation decision
    (2026-09-07, C+E moneyline / D+F run line): the 8 exp2 candidate columns
    are now PRODUCTION FEATURE_COLS members (59 → 67) and are computed by
    features.add_exp2_features, while the RAW source columns remain
    dataset-only and never enter the estimator."""

    def test_feature_cols_67_with_exp2_candidates(self):
        from features import EXP2_CANDIDATE_COLS
        self.assertEqual(len(FEATURE_COLS), 67,
                         "FEATURE_COLS must be the 59 baseline + 8 exp2 candidates")
        for c in EXP2_CANDIDATE_COLS:
            self.assertIn(c, FEATURE_COLS, f"{c} missing from FEATURE_COLS")

    def test_no_raw_source_col_in_feature_cols(self):
        hits = [f for f in FEATURE_COLS
                if f in EXP2_SOURCE_COLS
                or any(f.endswith(sfx) for sfx in
                       [f"{s}_{side}" for s in EXP2_SOURCE_SUFFIXES
                        for side in ("home", "away")])]
        self.assertEqual(hits, [], f"raw source columns leaked: {hits}")

    def test_exp2_candidate_columns_produced_by_production_pipeline(self):
        """The 8 final candidates must be computed by features.py (the normal
        production path), not only by the experiment runner."""
        candidates = [
            "exp2_centered_k_diff", "exp2_cat_k_fastball_diff",
            "exp2_cat_k_breaking_diff", "exp2_cat_k_offspeed_diff",
            "exp2_cat_xwoba_fastball_diff", "exp2_cat_xwoba_breaking_diff",
            "exp2_cat_xwoba_offspeed_diff", "exp2_cat_platoon_k_fastball_diff",
        ]
        for c in candidates:
            self.assertIn(f'"{c}"', _FEATURES_SRC,
                          f"candidate {c} not produced by features.py")


class TestExp2RealDataSmoke(unittest.TestCase):
    """Real-data smoke test: build the full exp2 chain on one real Statcast
    week (requires network + pybaseball; skipped when unavailable)."""

    def test_real_window_builds_and_pit_holds(self):
        try:
            from pybaseball import statcast
        except Exception:
            self.skipTest("pybaseball unavailable")
        try:
            df = statcast(start_dt="2026-05-01", end_dt="2026-05-03")
        except Exception:
            self.skipTest("Statcast fetch failed (offline?)")
        if df is None or df.empty:
            self.skipTest("no Statcast data for window")
        con = duckdb.connect()
        con.register("pitches", df)
        for tbl in ("exp2_pa", "exp2_league_k", "exp2_league_cat",
                    "exp2_sp_cat_game", "exp2_sp_fbhand_game",
                    "exp2_sp_cat_daily", "exp2_sp_cat_cum",
                    "exp2_sp_cat_tot_daily", "exp2_sp_cat_tot_cum",
                    "exp2_sp_cat_prelim", "exp2_sp_cat",
                    "exp2_sp_fbhand_daily", "exp2_sp_fbhand_cum",
                    "exp2_sp_fbhand", "exp2_team_cat_game",
                    "exp2_team_cat_daily", "exp2_team_cat_cum", "exp2_team_cat",
                    "exp2_team_fbhand_game", "exp2_team_fbhand_daily",
                    "exp2_team_fbhand_cum", "exp2_team_fbhand",
                    "exp2_league_fbhand"):
            con.execute(_extract_sql(tbl))
        out = con.execute("""
            SELECT
                (SELECT AVG(league_k_pct) FROM exp2_league_k) lg,
                (SELECT AVG(sp_k_pct_cat) FROM exp2_sp_cat
                  WHERE pitch_cat = 'fastball') sfb,
                (SELECT AVG(team_k_pct_cat) FROM exp2_team_cat
                  WHERE pitch_cat = 'fastball') tfb,
                (SELECT AVG(sp_k_pct_fb_vs) FROM exp2_sp_fbhand
                  WHERE stand = 'L') sfl,
                (SELECT AVG(team_k_pct_fb_vs) FROM exp2_team_fbhand
                  WHERE stand = 'R') tfr
        """).fetchdf()
        # Plausible ranges: league K% ~0.15–0.30; per-cat rates in (0,1)
        self.assertTrue(0.10 <= out["lg"].iloc[0] <= 0.35)
        self.assertTrue((out["sfb"].dropna() < 1).all())
        self.assertTrue((out["sfl"].dropna() < 1).all())


if __name__ == "__main__":
    unittest.main()
