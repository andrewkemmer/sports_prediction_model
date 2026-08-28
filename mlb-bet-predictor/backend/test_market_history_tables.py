"""Tests for the Totals & Run Lines prediction-history tables.

Pure-function fixtures over frontend/market_diagnostics.py — the same
module the six diagnostics charts use, so the history tables agree with
them exactly:
  * totals: rounded-line pick (over/under by the p >= 0.5 convention),
    whole-number-line column mapping, push exclusion from the win rate,
    date filtering.
  * run lines: pick side from p_home_cover_1_5, 1-run home win → away
    +1.5 winner, no pushes (half-run lines).
  * win-rate math + header counts on fixtures; real-artifact smoke.
Render smoke (source inspection, no Streamlit): both tables present in
frontend/markets.py, one shared date-picker pair, loud honest empty
states — nothing fabricated on missing artifacts.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# frontend/ moved to the repository root (multi-sport restructure, Phase B)
FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
if str(FRONTEND) not in sys.path:
    sys.path.insert(0, str(FRONTEND))

import market_diagnostics as diag  # noqa: E402


def _grid_row(exp_h, exp_a, hs, as_, total, p_over_map=None, pk=0,
              date="2025-05-01", p_cover=None):
    """One artifact-shaped row: flat 0.5 grid overridden per line map."""
    row = {"game_pk": pk, "kind": "oof", "game_date": date,
           "home_expected_runs": exp_h, "away_expected_runs": exp_a,
           "home_score": hs, "away_score": as_, "total_runs": total}
    for g in diag.TOTAL_GRID:
        row[f"p_over_{str(g).replace('.', '_')}"] = 0.5
    for line, p in (p_over_map or {}).items():
        row[f"p_over_{str(line).replace('.', '_')}"] = p
    if p_cover is not None:
        row[diag.RUN_COVER_COL] = p_cover
    return row


class TestTotalsHistory(unittest.TestCase):
    def test_rounded_line_pick_over_and_under(self):
        # 4.7 + 4.6 = 9.3 → line 9.5. p_over 0.62 → Over pick, prob 0.62.
        over = pd.DataFrame([_grid_row(4.7, 4.6, 5, 5, 10,
                                       {9.5: 0.62})])
        f = diag.totals_history_frame(over)
        self.assertEqual(f.iloc[0]["line"], 9.5)
        self.assertEqual(f.iloc[0]["pick"], "Over")
        self.assertAlmostEqual(f.iloc[0]["pick_prob"], 0.62)
        self.assertEqual(f.iloc[0]["winner"], "Over")   # 10 > 9.5
        self.assertEqual(f.iloc[0]["correct"], 1.0)
        # Same line, p_over 0.40 → Under pick (prob 0.60), total 8 < 9.5.
        under = pd.DataFrame([_grid_row(4.7, 4.6, 4, 4, 8,
                                        {9.5: 0.40})])
        g = diag.totals_history_frame(under)
        self.assertEqual(g.iloc[0]["pick"], "Under")
        self.assertAlmostEqual(g.iloc[0]["pick_prob"], 0.60)
        self.assertEqual(g.iloc[0]["winner"], "Under")
        self.assertEqual(g.iloc[0]["correct"], 1.0)

    def test_whole_number_line_maps_to_own_column(self):
        # 4.5 + 4.5 = 9.0 → whole-number line 9.0 → p_over_9_0 column.
        row = _grid_row(4.5, 4.5, 6, 4, 10, {9.0: 0.58})
        f = diag.totals_history_frame(pd.DataFrame([row]))
        self.assertEqual(f.iloc[0]["line"], 9.0)
        self.assertEqual(f.iloc[0]["pick"], "Over")
        self.assertAlmostEqual(f.iloc[0]["pick_prob"], 0.58)
        # Same geometry but p_over_9_0 < 0.5 → Under, even though the
        # bracketing half-line columns are flat 0.5.
        row2 = _grid_row(4.5, 4.5, 6, 4, 10, {9.0: 0.42})
        g = diag.totals_history_frame(pd.DataFrame([row2]))
        self.assertEqual(g.iloc[0]["pick"], "Under")
        self.assertAlmostEqual(g.iloc[0]["pick_prob"], 0.58)

    def test_push_excluded_from_win_rate(self):
        rows = [
            _grid_row(4.5, 4.5, 5, 4, 9, {9.0: 0.52}, pk=0),   # PUSH
            _grid_row(4.5, 4.5, 6, 4, 10, {9.0: 0.52}, pk=1),  # Over hit
            _grid_row(4.5, 4.5, 4, 4, 8, {9.0: 0.52}, pk=2),   # Over miss
        ]
        f = diag.totals_history_frame(pd.DataFrame(rows))
        self.assertEqual(len(f), 3)
        self.assertEqual(f.iloc[0]["winner"], "Push")
        self.assertTrue(pd.isna(f.iloc[0]["correct"]))
        stats = diag.history_win_rate(f)
        self.assertEqual(stats["n_games"], 2)        # push dropped
        self.assertEqual(stats["n_pushes"], 1)
        self.assertAlmostEqual(stats["win_rate"], 0.5)  # 1 hit / 2

    def test_half_line_never_pushes(self):
        # Line 9.5 with total 9: 9 != 9.5 → not a push; over missed.
        f = diag.totals_history_frame(
            pd.DataFrame([_grid_row(4.7, 4.8, 4, 5, 9, {9.5: 0.62})]))
        self.assertEqual(f.iloc[0]["winner"], "Under")
        self.assertEqual(f.iloc[0]["correct"], 0.0)   # Over pick missed
        stats = diag.history_win_rate(f)
        self.assertEqual(stats["n_pushes"], 0)

    def test_date_filtering(self):
        rows = [
            _grid_row(4.7, 4.8, 5, 4, 10, {9.5: 0.62}, pk=0,
                      date="2025-04-01"),
            _grid_row(4.7, 4.8, 5, 4, 10, {9.5: 0.62}, pk=1,
                      date="2025-06-15"),
            _grid_row(4.7, 4.8, 5, 4, 10, {9.5: 0.62}, pk=2,
                      date="2025-09-30"),
        ]
        f = diag.totals_history_frame(pd.DataFrame(rows))
        import datetime
        sub = diag.filter_history_frame(f, datetime.date(2025, 5, 1),
                                        datetime.date(2025, 7, 1))
        self.assertEqual(len(sub), 1)
        self.assertEqual(sub.iloc[0]["game_pk"], 1)
        # Inclusive edges: start == earliest row date keeps it.
        sub2 = diag.filter_history_frame(f, datetime.date(2025, 4, 1),
                                         datetime.date(2025, 4, 1))
        self.assertEqual(len(sub2), 1)
        self.assertEqual(sub2.iloc[0]["game_pk"], 0)
        # Empty window → empty frame, no crash.
        sub3 = diag.filter_history_frame(f, datetime.date(2026, 1, 1),
                                         datetime.date(2026, 2, 1))
        self.assertEqual(len(sub3), 0)

    def test_missing_grid_column_dropped_not_fabricated(self):
        df = pd.DataFrame([_grid_row(4.7, 4.8, 5, 4, 10, {9.5: 0.62})])
        df = df.drop(columns=["p_over_9_5"])
        f = diag.totals_history_frame(df)
        self.assertEqual(len(f), 0)     # cannot price → dropped
        stats = diag.history_win_rate(f)
        self.assertEqual(stats["n_games"], 0)
        self.assertIsNone(stats["win_rate"])

    def test_empty_inputs_warn_not_crash(self):
        f = diag.totals_history_frame(pd.DataFrame())
        self.assertEqual(len(f), 0)
        f = diag.totals_history_frame(
            pd.DataFrame([_grid_row(4.5, 4.5, 5, 4, 9)]
                         ).drop(columns=["home_expected_runs"]))
        self.assertEqual(len(f), 0)


class TestRunlineHistory(unittest.TestCase):
    def test_pick_side_and_prob(self):
        home_fav = pd.DataFrame([_grid_row(4.5, 4.5, 5, 2, 7,
                                           p_cover=0.58)])
        f = diag.runline_history_frame(home_fav)
        self.assertEqual(f.iloc[0]["pick"], "home")
        self.assertAlmostEqual(f.iloc[0]["pick_prob"], 0.58)
        away_fav = pd.DataFrame([_grid_row(4.5, 4.5, 2, 5, 7,
                                           p_cover=0.42)])
        g = diag.runline_history_frame(away_fav)
        self.assertEqual(g.iloc[0]["pick"], "away")
        self.assertAlmostEqual(g.iloc[0]["pick_prob"], 0.58)

    def test_one_run_home_win_is_away_cover(self):
        # 5-4 home win (margin 1) → home −1.5 LOSES, away +1.5 wins.
        rows = [
            _grid_row(4.5, 4.5, 5, 4, 9, p_cover=0.60, pk=0),   # home pick
            _grid_row(4.5, 4.5, 5, 4, 9, p_cover=0.40, pk=1),   # away pick
        ]
        f = diag.runline_history_frame(pd.DataFrame(rows))
        self.assertEqual(f.iloc[0]["winner"], "away")
        self.assertEqual(f.iloc[0]["correct"], 0.0)   # home −1.5 lost
        self.assertEqual(f.iloc[1]["winner"], "away")
        self.assertEqual(f.iloc[1]["correct"], 1.0)   # away +1.5 won

    def test_two_run_home_win_covers(self):
        f = diag.runline_history_frame(
            pd.DataFrame([_grid_row(4.5, 4.5, 6, 4, 10, p_cover=0.60)]))
        self.assertEqual(f.iloc[0]["winner"], "home")
        self.assertEqual(f.iloc[0]["correct"], 1.0)

    def test_no_pushes_ever(self):
        rows = [_grid_row(4.5, 4.5, 5, 4, 9, p_cover=0.5, pk=i)
                for i in range(4)]
        f = diag.runline_history_frame(pd.DataFrame(rows))
        self.assertFalse((f["winner"] == "Push").any())
        stats = diag.history_win_rate(f)
        self.assertEqual(stats["n_pushes"], 0)
        self.assertEqual(stats["n_games"], len(f))   # every row counts

    def test_missing_cover_column_dropped(self):
        df = pd.DataFrame([_grid_row(4.5, 4.5, 5, 4, 9)])
        f = diag.runline_history_frame(df)   # no p_cover col
        self.assertEqual(len(f), 0)

    def test_empty_input(self):
        self.assertEqual(len(diag.runline_history_frame(pd.DataFrame())), 0)


class TestWinRateAndHeaders(unittest.TestCase):
    def test_header_counts_match_filtered_rows(self):
        rows = [
            _grid_row(4.5, 4.5, 5, 4, 9, {9.0: 0.52}, pk=0,   # push
                      date="2025-05-01"),
            _grid_row(4.5, 4.5, 6, 4, 10, {9.0: 0.52}, pk=1,  # over hit
                      date="2025-06-01"),
            _grid_row(4.5, 4.5, 4, 4, 8, {9.0: 0.52}, pk=2,   # over miss
                      date="2025-07-01"),
            _grid_row(4.5, 4.5, 5, 5, 11, {9.0: 0.52}, pk=3,  # over hit
                      date="2025-08-01"),
        ]
        f = diag.totals_history_frame(pd.DataFrame(rows))
        import datetime
        # June 1 – Aug 1 inclusive → pks 1, 2, 3 (all three priced rows).
        sub = diag.filter_history_frame(f, datetime.date(2025, 6, 1),
                                        datetime.date(2025, 8, 1))
        stats = diag.history_win_rate(sub)
        self.assertEqual(stats["n_games"], 3)
        # win_rate is rounded to 6 decimals by the helper.
        self.assertAlmostEqual(stats["win_rate"], 2 / 3, places=6)
        # Header counts match the FILTERED rows, not the full frame.
        self.assertEqual(stats["n_pushes"], 0)            # push (pk 0) out of range
        # Full frame: push excluded → 3 priced, 2 hits.
        all_stats = diag.history_win_rate(f)
        self.assertEqual(all_stats["n_games"], 3)
        self.assertEqual(all_stats["n_pushes"], 1)
        self.assertAlmostEqual(all_stats["win_rate"], 2 / 3, places=6)

    def test_win_rate_none_when_no_priced_rows(self):
        f = diag.totals_history_frame(pd.DataFrame())
        self.assertIsNone(diag.history_win_rate(f)["win_rate"])


class TestRealArtifactHistorySmoke(unittest.TestCase):
    """Read-only smoke over the shipped OOF artifact — the tables' pick /
    result logic must run end-to-end and produce sane pooled numbers."""

    @classmethod
    def setUpClass(cls):
        dd = Path(__file__).resolve().parents[1] / "data_delivery"
        m_path = dd / "run_engine_markets_20260825.csv"
        if not m_path.exists():
            raise unittest.SkipTest("local run-engine artifact absent")
        markets = pd.read_csv(m_path)
        cls.decided = diag.decided_rows(markets)

    def test_totals_history_on_real_artifact(self):
        tot = diag.totals_history_frame(self.decided)
        self.assertGreater(len(tot), 4000)
        stats = diag.history_win_rate(tot)
        self.assertGreater(stats["n_pushes"], 0)          # whole lines push
        self.assertLess(stats["n_pushes"] / len(tot), 0.10)
        self.assertEqual(stats["n_games"] + stats["n_pushes"], len(tot))
        self.assertTrue(0.50 < stats["win_rate"] < 0.60)  # ~54.5% shipped
        # Whole-number lines present, half-lines never push.
        self.assertTrue((tot["line"] % 1 == 0).any())
        self.assertFalse(((tot["winner"] == "Push")
                          & (tot["line"] % 1 != 0)).any())

    def test_runline_history_on_real_artifact(self):
        rl = diag.runline_history_frame(self.decided)
        self.assertGreater(len(rl), 4000)
        stats = diag.history_win_rate(rl)
        self.assertEqual(stats["n_pushes"], 0)            # half-lines never push
        self.assertEqual(stats["n_games"], len(rl))
        self.assertTrue(0.55 < stats["win_rate"] < 0.70)  # ~64% shipped


class TestMatchupResolution(unittest.TestCase):
    """Team-name join for the history tables — the markets artifact's
    game_pk column is OBJECT dtype (numeric StatsAPI game_pk mixed with
    ESPN game_id slate rows in one column), so the lookup must normalize
    the key: int/float/'float-string' game_pk resolves via the int map,
    ESPN game_id via the string map (145d841 slate-key discipline), and
    the "—" fallback remains ONLY for genuinely unresolvable rows."""

    def _glf(self):
        return pd.DataFrame({
            "game_pk": [778485, 823422, None, 999999],
            "game_id": ["20260825_HOU@CLE", "20260825_STL@PHI",
                        "20260826_TB@DET", None],
            "home_team": ["CLE", "PHI", "DET", None],
            "away_team": ["HOU", "STL", "TB", None],
        })

    def test_int_game_pk_resolves(self):
        tm = diag.build_team_map(self._glf())
        self.assertEqual(diag.resolve_matchup_teams(tm, 778485),
                         ("HOU", "CLE"))
        # numpy integer keys (pandas object/Int64 cells) also resolve.
        self.assertEqual(diag.resolve_matchup_teams(tm, np.int64(823422)),
                         ("STL", "PHI"))

    def test_float_and_float_string_keys_resolve(self):
        # The markets artifact parses the mixed column as object: numeric
        # cells arrive as float (778485.0) or, in some pipelines, the
        # string '778485.0' — both must hit the int map.
        tm = diag.build_team_map(self._glf())
        self.assertEqual(diag.resolve_matchup_teams(tm, 778485.0),
                         ("HOU", "CLE"))
        self.assertEqual(diag.resolve_matchup_teams(tm, np.float64(823422.0)),
                         ("STL", "PHI"))
        self.assertEqual(diag.resolve_matchup_teams(tm, "823422.0"),
                         ("STL", "PHI"))

    def test_game_id_fallback_resolves(self):
        # ESPN game_id key (the 145d841 slate-key convention) resolves
        # through the string map when it exists in the features frame.
        tm = diag.build_team_map(self._glf())
        self.assertEqual(diag.resolve_matchup_teams(tm, "20260825_HOU@CLE"),
                         ("HOU", "CLE"))

    def test_unresolvable_keys_stay_dash(self):
        tm = diag.build_team_map(self._glf())
        self.assertEqual(diag.resolve_matchup_teams(tm, 123456), ("—", "—"))
        self.assertEqual(diag.resolve_matchup_teams(tm, 778485.5), ("—", "—"))
        self.assertEqual(diag.resolve_matchup_teams(tm, "nonsense_id"),
                         ("—", "—"))
        self.assertEqual(diag.resolve_matchup_teams(tm, None), ("—", "—"))
        self.assertEqual(diag.resolve_matchup_teams(tm, np.nan), ("—", "—"))

    def test_build_team_map_dual_keyed_and_skips_garbage(self):
        tm = diag.build_team_map(self._glf())
        # Same game reachable by BOTH its game_pk and its game_id.
        self.assertEqual(tm[778485], ("HOU", "CLE"))
        self.assertEqual(tm["20260825_HOU@CLE"], ("HOU", "CLE"))
        self.assertEqual(tm[823422], ("STL", "PHI"))
        # game_id-only row (NaN game_pk) still joinable by id; missing
        # team names / NaN game_id rows are skipped, not fabricated.
        self.assertEqual(tm["20260826_TB@DET"], ("TB", "DET"))
        self.assertNotIn(999999, tm)
        self.assertNotIn("20260825_STL@PHI_x", tm)

    def test_empty_frame_returns_empty_map(self):
        self.assertEqual(diag.build_team_map(pd.DataFrame()), {})
        self.assertEqual(diag.build_team_map(
            pd.DataFrame({"game_pk": [1]})), {})   # no team cols


class TestRealArtifactMatchups(unittest.TestCase):
    """Real-artifact smoke: every decided history row resolves to real
    team names (no "— @ —"), using the shipped markets + features CSVs."""

    @classmethod
    def setUpClass(cls):
        dd = Path(__file__).resolve().parents[1] / "data_delivery"
        m_path = dd / "run_engine_markets_20260826.csv"
        g_path = dd / "game_level_features.csv"
        if not m_path.exists() or not g_path.exists():
            raise unittest.SkipTest("local run-engine artifacts absent")
        cls.tm = diag.build_team_map(
            pd.read_csv(g_path, usecols=lambda c: c in (
                "game_pk", "game_id", "home_team", "away_team")))
        cls.decided = diag.decided_rows(pd.read_csv(m_path))

    def test_every_decided_row_resolves(self):
        self.assertGreater(len(self.decided), 4000)
        bad = [k for k in self.decided["game_pk"]
               if diag.resolve_matchup_teams(self.tm, k) == ("—", "—")]
        self.assertEqual(bad, [],
                         f"{len(bad)} decided rows with no team names")

    def test_matchup_format_is_away_at_home(self):
        away, home = diag.resolve_matchup_teams(
            self.tm, self.decided["game_pk"].iloc[0])
        self.assertNotIn("—", (away, home))
        self.assertEqual(away, str(away).strip())
        self.assertNotEqual(away, home)


class TestRenderSmokeSourceInspection(unittest.TestCase):
    """markets.py must render both tables with one shared date-picker pair
    and loud honest empty states — verified by source inspection (no
    Streamlit import)."""

    @classmethod
    def setUpClass(cls):
        with open(FRONTEND / "markets.py") as f:
            cls.src = f.read()

    def test_both_tables_present(self):
        self.assertIn("Game Totals — Prediction History", self.src)
        self.assertIn("Run Lines — Prediction History", self.src)
        self.assertIn("Prediction History — Totals & Run Lines", self.src)

    def test_shared_date_picker_pair(self):
        # Exactly one pair of pickers controlling both tables.
        self.assertEqual(self.src.count('"History start date"'), 1)
        self.assertEqual(self.src.count('"History end date"'), 1)
        self.assertIn("fc1, fc2, _ = st.columns([1, 1, 2])", self.src)

    def test_pure_logic_lives_in_diagnostics_module(self):
        # The pick/result logic is import-safe (no Streamlit) in
        # market_diagnostics.py, not inline in the page.
        with open(FRONTEND / "market_diagnostics.py") as f:
            dsrc = f.read()
        for fn in ("def totals_history_frame(", "def runline_history_frame(",
                   "def filter_history_frame(", "def history_win_rate("):
            self.assertIn(fn, dsrc)
        # markets.py only RENDERS what the pure functions produce.
        self.assertIn("diag.totals_history_frame(decided)", self.src)
        self.assertIn("diag.runline_history_frame(decided)", self.src)

    def test_team_lookup_uses_normalized_key_discipline(self):
        # The render layer resolves matchups through the pure
        # resolve_matchup_teams helper (dual-keyed map) — it must not
        # fall back to a raw per-column .get() that misses on the
        # object-dtype game_pk.
        with open(FRONTEND / "market_diagnostics.py") as f:
            dsrc = f.read()
        for fn in ("def build_team_map(", "def resolve_matchup_teams("):
            self.assertIn(fn, dsrc)
        self.assertIn("diag.resolve_matchup_teams(teams, r[\"game_pk\"])",
                      self.src)
        self.assertIn("diag.build_team_map(gl)", self.src)
        self.assertNotIn("teams[\"away_team\"].get(r[\"game_pk\"]",
                        self.src)
        self.assertNotIn(".set_index(\"game_pk\")", self.src)

    def test_no_fabrication_on_missing_artifacts(self):
        # Empty decided games → honest empty state, never invented rows
        # (the second branch's string is split across source literals, so
        # count the "is fabricated" fragments instead).
        self.assertIn("Nothing is fabricated in the meantime", self.src)
        self.assertGreaterEqual(self.src.count("is fabricated"), 2)
        self.assertIn("diag.decided_rows(markets)", self.src)


if __name__ == "__main__":
    unittest.main()
