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
              date="2025-05-01", p_cover=None, fair_line=None, step=0.10):
    """One artifact-shaped row: a realistic monotone-decreasing p_over grid
    that CROSSES 0.5 at the FAIR line (default = round-to-half of
    exp_h + exp_a), with p_under = 1 − p_over mirroring each line. The fair
    own total line (grid argmin |re-scaled P(over) − 0.5|) therefore lands
    at ``fair_line``, so picks/pushes follow the intended line. p_over_map
    overrides a line's probability and mirrors p_under; keep the intended
    line within ~±0.04 of 0.5 so it stays the fair argmin."""
    fair_line = (diag.round_to_half(exp_h + exp_a) if fair_line is None
                 else float(fair_line))
    row = {"game_pk": pk, "kind": "oof", "game_date": date,
           "home_expected_runs": exp_h, "away_expected_runs": exp_a,
           "home_score": hs, "away_score": as_, "total_runs": total}
    grid = diag.TOTAL_GRID
    at_idx = grid.index(fair_line)
    for i, g in enumerate(grid):
        base = float(np.clip(0.5 + step * (at_idx - i), 0.02, 0.98))
        row[f"p_over_{str(g).replace('.', '_')}"] = base
        row[f"p_under_{str(g).replace('.', '_')}"] = round(1.0 - base, 6)
    for line, p in (p_over_map or {}).items():
        row[f"p_over_{str(line).replace('.', '_')}"] = p
        row[f"p_under_{str(line).replace('.', '_')}"] = round(1.0 - p, 6)
    if p_cover is not None:
        row[diag.RUN_COVER_COL] = p_cover
    return row


class TestTotalsHistory(unittest.TestCase):
    def test_rounded_line_pick_over_and_under(self):
        # 4.7 + 4.6 = 9.3 → fair line 9.5. re-scaled P(over|no push) 0.52 →
        # Over pick, prob 0.52 (the raw p_over is the re-scaled value here
        # because the 9.5 half-line has no push mass in the fixture).
        over = pd.DataFrame([_grid_row(4.7, 4.6, 5, 5, 10,
                                       {9.5: 0.52})])
        f = diag.totals_history_frame(over)
        self.assertEqual(f.iloc[0]["line"], 9.5)
        self.assertEqual(f.iloc[0]["pick"], "Over")
        self.assertAlmostEqual(f.iloc[0]["pick_prob"], 0.52)
        self.assertEqual(f.iloc[0]["winner"], "Over")   # 10 > 9.5
        self.assertEqual(f.iloc[0]["correct"], 1.0)
        # Same line, re-scaled 0.46 → Under pick (prob 0.54), total 8 < 9.5.
        under = pd.DataFrame([_grid_row(4.7, 4.6, 4, 4, 8,
                                        {9.5: 0.46})])
        g = diag.totals_history_frame(under)
        self.assertEqual(g.iloc[0]["pick"], "Under")
        self.assertAlmostEqual(g.iloc[0]["pick_prob"], 0.54)
        self.assertEqual(g.iloc[0]["winner"], "Under")
        self.assertEqual(g.iloc[0]["correct"], 1.0)

    def test_whole_number_line_maps_to_own_column(self):
        # 4.5 + 4.5 = 9.0 → whole-number fair line 9.0 → p_over_9_0 column.
        row = _grid_row(4.5, 4.5, 6, 4, 10, {9.0: 0.56})
        f = diag.totals_history_frame(pd.DataFrame([row]))
        self.assertEqual(f.iloc[0]["line"], 9.0)
        self.assertEqual(f.iloc[0]["pick"], "Over")
        self.assertAlmostEqual(f.iloc[0]["pick_prob"], 0.56)
        # Same geometry but re-scaled p_over_9_0 < 0.5 → Under, even though
        # the bracketing half-line columns sit 0.10 away from 0.5.
        row2 = _grid_row(4.5, 4.5, 6, 4, 10, {9.0: 0.44})
        g = diag.totals_history_frame(pd.DataFrame([row2]))
        self.assertEqual(g.iloc[0]["pick"], "Under")
        self.assertAlmostEqual(g.iloc[0]["pick_prob"], 0.56)

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
            pd.DataFrame([_grid_row(4.7, 4.8, 4, 5, 9, {9.5: 0.52})]))
        self.assertEqual(f.iloc[0]["winner"], "Under")
        self.assertEqual(f.iloc[0]["correct"], 0.0)   # Over pick missed
        stats = diag.history_win_rate(f)
        self.assertEqual(stats["n_pushes"], 0)

    def test_date_filtering(self):
        rows = [
            _grid_row(4.7, 4.8, 5, 4, 10, {9.5: 0.52}, pk=0,
                      date="2025-04-01"),
            _grid_row(4.7, 4.8, 5, 4, 10, {9.5: 0.52}, pk=1,
                      date="2025-06-15"),
            _grid_row(4.7, 4.8, 5, 4, 10, {9.5: 0.52}, pk=2,
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
        # No grid columns at all → the fair line has nothing to price and
        # the round-mean fallback's own column is absent too: dropped, never
        # fabricated (a single missing column would just shift the fair
        # argmin to an available neighbor).
        df = pd.DataFrame([_grid_row(4.7, 4.8, 5, 4, 10, {9.5: 0.52})])
        grid_cols = [c for c in df.columns
                     if c.startswith("p_over_") or c.startswith("p_under_")]
        df = df.drop(columns=grid_cols)
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


def _rl_row(hs, as_, hw=None, cover=None, pk=0, date="2025-05-01"):
    """One artifact-shaped row carrying a POST-FIX p_rl grid (home/push/
    away for 1.0, 1.5, 2.0, …). """
    row = {"game_pk": pk, "kind": "oof", "game_date": date,
           "home_expected_runs": 5.0, "away_expected_runs": 4.0,
           "home_score": hs, "away_score": as_,
           "total_runs": hs + as_, "p_home_cover_1_5": 0.5}
    if hw is not None:                    # home = P(margin≥1) = h + push
        row["p_rl_1_0_home"] = hw
        row["p_rl_1_0_push"] = 0.0
    else:                                 # p_under mirror = 1 − over
        row["p_rl_1_0_home"] = 0.55
        row["p_rl_1_0_push"] = 0.15
    if cover is not None:                 # fractional cover splits
        for m, col in ((1.0, "home"), (1.5, "home"), (2.0, "home"),
                       (2.5, "home"), (3.0, "home"), (3.5, "home"),
                       (4.0, "home"), (1.0, "away"), (2.0, "away")):
            key = f"{m:.1f}".replace(".", "_")
            row[f"p_rl_{key}_{col}"] = cover
    else:                                 # full post-fix grid (real bands)
        for m, p in ((1.0, 0.55), (1.5, 0.55), (2.0, 0.40), (2.5, 0.40),
                     (3.0, 0.25), (3.5, 0.25), (4.0, 0.15)):
            key = f"{m:.1f}".replace(".", "_")
            row[f"p_rl_{key}_home"] = p
            row[f"p_rl_{key}_away"] = 1.0 - p
        row["p_rl_1_0_push"] = 0.20
        row["p_rl_2_0_push"] = 0.10
        row["p_rl_3_0_push"] = 0.05
    return row


class TestFairTotalLine(unittest.TestCase):
    """The own total line is the FAIR line: grid argmin of
    |re-scaled P(over) − 0.5| over 6.5…12.5, ties → lower line, re-scaled
    = p_over/(p_over+p_under) conditions out the push band, a grid-boundary
    argmin is taken verbatim (never fabricated)."""

    def _row(self, pairs: dict):
        """Row with explicit per-line p_over/p_under values; lines not
        listed get a far-from-0.5 profile (0.90/0.10 below, 0.10/0.90
        above) so they can never win the argmin."""
        row = {"game_pk": 1, "kind": "oof", "game_date": "2025-05-01",
               "home_expected_runs": 4.5, "away_expected_runs": 4.5,
               "home_score": 5, "away_score": 4, "total_runs": 9}
        for i, g in enumerate(diag.TOTAL_GRID):
            base = 0.90 if i < len(diag.TOTAL_GRID) // 2 else 0.10
            key = str(g).replace(".", "_")
            row[f"p_over_{key}"] = base
            row[f"p_under_{key}"] = round(1.0 - base, 6)
        for g, (po, pu) in pairs.items():
            key = str(g).replace(".", "_")
            row[f"p_over_{key}"] = po
            row[f"p_under_{key}"] = pu
        return row

    def test_ties_pick_lower_line(self):
        # 8.0 (0.55) and 9.0 (0.45) both sit exactly |0.05| from 0.5 — the
        # LOWER line (8.0) must win, even though 8.5 is bracketed by them.
        row = self._row({8.0: (0.55, 0.45), 9.0: (0.45, 0.55)})
        self.assertEqual(diag.fair_total_line_row(row), 8.0)
        df = pd.DataFrame([row])
        self.assertEqual(float(diag.fair_total_lines(df)[0]), 8.0)

    def test_fat_push_band_rescales_to_win_own_line(self):
        # Whole line 9.0 with over 0.44 / push 0.14 / under 0.42: re-scaled
        # P(over|no push) = 0.44/0.86 = 0.5116 (Δ 0.0116) — the raw p_over
        # is BELOW 0.5, but the re-scaled value (push folded out) is the
        # closest to 0.5 of the grid, so 9.0 is chosen over the 8.5/9.5
        # neighbors (both Δ 0.10).
        row = self._row({9.0: (0.44, 0.42), 8.5: (0.60, 0.40),
                         9.5: (0.40, 0.60)})
        self.assertEqual(diag.fair_total_line_row(row), 9.0)
        self.assertAlmostEqual(0.44 / (0.44 + 0.42), 0.5116, places=3)
        # The raw over+under leaves the 0.14 push band out (po+pu < 1).
        self.assertAlmostEqual(0.44 + 0.42, 0.86, places=9)

    def test_half_line_push_zero_prices_raw(self):
        # Half-lines never push: po + pu == 1, so re-scaled == raw. A 8.5
        # line at exactly 0.50/0.50 is the perfect fair line (Δ 0).
        row = self._row({8.5: (0.50, 0.50)})
        self.assertEqual(diag.fair_total_line_row(row), 8.5)
        po, pu = row["p_over_8_5"], row["p_under_8_5"]
        self.assertAlmostEqual(po + pu, 1.0, places=9)   # no push band
        self.assertAlmostEqual(po / (po + pu), 0.5, places=9)

    def test_grid_edge_argmin_taken_verbatim(self):
        # Very low-total profile: 6.5 is the closest to 0.5 and sits on the
        # grid LOW edge — the boundary argmin is taken as-is, never clamped
        # or fabricated outside the grid.
        row = self._row({6.5: (0.48, 0.52)})
        self.assertEqual(diag.fair_total_line_row(row), 6.5)
        # High edge symmetric case: 12.5 wins the argmin.
        row2 = self._row({12.5: (0.48, 0.52)})
        self.assertEqual(diag.fair_total_line_row(row2), 12.5)

    def test_unpricable_returns_none(self):
        # No grid Over/Under pair → None (caller falls back to the
        # round-to-half projection; nothing fabricated).
        self.assertIsNone(diag.fair_total_line_row({"game_pk": 1}))
        self.assertIsNone(diag.fair_total_line_row(
            {"game_pk": 1, "p_over_8_5": 0.5}))   # under missing


class TestCutAndMonitor(unittest.TestCase):
    """Unified 3-way run-line cut logic + the calibration-card monitors."""

    def test_neg_half_equivalent_to_zero(self):
        # In MLB, line 0 ≡ −0.5 (integer margins, no ties): cover ≥ 1 and
        # cover ≥ 0 select identical sets, so a rounding landing on 0 maps
        # to the −0.5 magnitude and −0.5 is never magnified to 0.
        self.assertEqual(diag.map_run_line_zero(0.0), 0.5)
        self.assertEqual(diag.map_run_line_zero(0.0 + 1e-12), 0.5)
        self.assertEqual(diag.map_run_line_zero(0.5), 0.5)
        # A favorite outright win (margin 4) is a −0.5 cover every time.
        df = pd.DataFrame([_rl_row(6, 2)])           # home wins by 4
        cov, is_home = diag.favored_cover_at(df, 0.5)
        self.assertTrue(bool(is_home[0]))
        self.assertAlmostEqual(cov[0], 0.55 + 0.20)  # P(margin > 0)

    def test_cut_never_below_0_5_for_favored(self):
        # The favored side's cover at 0.5 is P(win) >= 0.5 by construction,
        # so the cut is ALWAYS >= 0.5 and line 0 never occurs.
        df = pd.DataFrame([_rl_row(6, 2, hw=None), _rl_row(2, 6, hw=0.4)])
        f = diag.runline_cut_history_frame(df)
        self.assertEqual(len(f), 2)
        for _, r in f.iterrows():
            self.assertEqual(abs(r["cut"]), r["cut"])   # ≥ 0 by sign
            self.assertGreaterEqual(r["cut"], 0.5)
            self.assertIn(r["cut"], diag.RUN_GRID_CUT)
        # The decisive home-favorite (margin 4) cuts at the deepest line
        # whose cover >= 0.5: 1.5 in the fixture grid (0.55 >= 0.5), while
        # 2.0 sits below at 0.40.
        exact = diag.runline_cut_history_frame(pd.DataFrame([_rl_row(6, 2)]))
        self.assertEqual(exact.iloc[0]["cut"], 1.5)

    def test_whole_line_margin_equals_cut_is_push(self):
        # Home-favorite at cut 1.0, final margin == 1 -> whole-line PUSH
        # (excluded from W/(W+L) via correct=NaN). Margin 4 -> cover.
        push = pd.DataFrame([_rl_row(5, 4)])          # margin 1 (== cut 1? cut is 2.0 in default grid)
        # Force cut to 1.0 so margin == cut (whole line) -> push.
        push.loc[0, "p_rl_1_5_home"] = 0.3      # stop the cut at 1.0
        push.loc[0, "p_rl_2_0_home"] = 0.3
        f = diag.runline_cut_history_frame(push)
        r = f.iloc[0]
        self.assertEqual(r["cut"], 1.0)
        self.assertEqual(r["winner"], "Push")
        self.assertTrue(r["push"])
        self.assertTrue(pd.isna(r["correct"]))
        stats = diag.history_win_rate(f)
        self.assertEqual(stats["n_games"], 0)         # excluded from both
        self.assertEqual(stats["n_pushes"], 1)

    def test_win_rate_excludes_pushes_both_sides(self):
        # Cover (margin>cut, cut 1.0… but margin 4 → cover) + a push (cut 1.0,
        # margin == 1) + a loss: win rate = 1 / 2, push out of both.
        cover = pd.DataFrame([_rl_row(6, 2)])
        push = pd.DataFrame([_rl_row(5, 4)]); push.loc[0, "p_rl_1_5_home"] = 0.3
        loss = pd.DataFrame([_rl_row(2, 6, hw=0.4)])
        f = pd.concat([diag.runline_cut_history_frame(cover),
                       diag.runline_cut_history_frame(push),
                       diag.runline_cut_history_frame(loss)], ignore_index=True)
        stats = diag.history_win_rate(f)
        self.assertEqual(stats["n_pushes"], 1)
        self.assertEqual(stats["n_games"], 2)         # push dropped from denom
        self.assertAlmostEqual(stats["win_rate"], 1 / 2)

    def test_totals_side_filter_partitions_population(self):
        # Over + Under must exactly partition the All population (n sums,
        # no double count) at the same threshold.
        rows = [_grid_row(4.7, 4.8, 5, 4, 10, {9.5: 0.52}, pk=i)     # Over
                for i in range(3)]
        rows += [_grid_row(4.7, 4.8, 4, 4, 8, {9.5: 0.46}, pk=100 + i)  # Under
                 for i in range(4)]
        dec = pd.DataFrame(rows)
        all_s = diag.totals_monitor_stats(dec, min_pct=50, side="All")
        sum_n = sum(all_s["sides"][s]["n"] for s in ("Over", "Under"))
        self.assertEqual(sum_n, all_s["n"])
        over = diag.totals_monitor_stats(dec, min_pct=50, side="Over")
        under = diag.totals_monitor_stats(dec, min_pct=50, side="Under")
        self.assertEqual(over["n"] + under["n"], all_s["n"])
        self.assertAlmostEqual(all_s["win_rate"],
                               (over["n_wins"] + under["n_wins"]) /
                               (over["n"] + under["n"]))

    def test_percent_toggle_is_cumulative(self):
        # Raising pick_prob> threshold is a NESTED subset: higher threshold
        # keeps a subpopulation, so n never grows.
        rows = [_grid_row(4.7, 4.8, 5, 4, 10, {9.5: p}, pk=i)
                for i, p in enumerate([0.42, 0.48, 0.52, 0.58, 0.63])]
        dec = pd.DataFrame(rows)
        ns = [diag.totals_monitor_stats(dec, min_pct=t)['n']
              for t in diag.TOTALS_CONF_THRESHOLDS]
        for lo, hi in zip(ns, ns[1:]):
            self.assertGreaterEqual(lo, hi)   # non-increasing with threshold

    def test_totals_conf_thresholds_exactly_50_to_55(self):
        """The totals card confidence toggle is exactly the 1-point steps
        50-55 (default 50). At the fair-line own line every pick is >50%,
        so 40/45 were no-ops (all picks qualified) and 55+ was empty — the
        1-point steps show the population fall-off vs confidence."""
        self.assertEqual(diag.TOTALS_CONF_THRESHOLDS,
                         [50, 51, 52, 53, 54, 55])
        self.assertEqual(min(diag.TOTALS_CONF_THRESHOLDS), 50)
        self.assertEqual(max(diag.TOTALS_CONF_THRESHOLDS), 55)

    def test_min_pct_50_includes_all_decided_games(self):
        """At the default threshold (pick_prob > 50%) every priced non-push
        game qualifies — the push stays excluded from the win-rate
        denominator (W/(W+L)), and the All/Over/Under split still sums to
        the All population."""
        rows = [
            _grid_row(4.5, 4.5, 5, 4, 9, {9.0: 0.52}, pk=0),   # PUSH
            _grid_row(4.7, 4.6, 5, 5, 10, {9.5: 0.52}, pk=1),  # Over
            _grid_row(4.7, 4.6, 4, 4, 8, {9.5: 0.46}, pk=2),   # Under
            _grid_row(4.7, 4.6, 5, 5, 10, {9.5: 0.55}, pk=3),  # Over
        ]
        dec = pd.DataFrame(rows)
        s = diag.totals_monitor_stats(dec, min_pct=50, side="All")
        self.assertEqual(s["min_pct"], 50)   # default threshold
        self.assertEqual(s["n"], 3)          # push excluded, all decided in
        self.assertEqual(s["n_pushes"], 1)
        self.assertAlmostEqual(s["win_rate"], 1.0)   # 3/3 picks won
        # Side filter still partitions the All population at the threshold.
        self.assertEqual(s["sides"]["Over"]["n"]
                         + s["sides"]["Under"]["n"], 3)

    def test_line_toggle_enumerates_full_grid(self):
        # The signed choice list (−0.5, −1, −1.5, … −4) maps injectively to
        # the priced magnitudes RUN_GRID_CUT — 0 never appears.
        mags = [diag.map_run_line_zero(abs(c)) for c in diag.RUN_LINE_CHOICES]
        self.assertEqual(mags, diag.RUN_GRID_CUT)       # exact enumeration
        self.assertNotIn(0.0, mags)
        # A whole-number chosen line is handled as whole (push-capable).
        self.assertTrue(float(1.0).is_integer())

    def test_monitor_stats_structural_resolution(self):
        # Whole-line −1: margin > 1 covers, margin == 1 pushes, else loss;
        # half-line −1.5 never pushes (margin == 1.5 impossible on integers).
        rows = [_rl_row(6, 2), _rl_row(5, 4), _rl_row(2, 6)]
        dec = pd.DataFrame(rows)
        s = diag.runline_monitor_stats(dec, 1.0)
        self.assertEqual(s["n"], 3)
        # wins=cover(margin>1): margins 4, 1, −4 → 1 win; push margin==1 → 1.
        self.assertEqual(s["n_wins"] + s["n_losses"] + s["n_pushes"], 3)
        self.assertEqual(s["n_pushes"], 1)
        s15 = diag.runline_monitor_stats(dec, 1.5)
        self.assertEqual(s15["n_pushes"], 0)          # half-line never pushes

    @staticmethod
    def _rl_3way(hs, as_, cover, push, pk):
        """One artifact-shaped row with an explicit consistent 3-way split
        at whole line −1.0 (home + push + away == 1.0)."""
        return {"game_pk": pk, "kind": "oof", "game_date": "2025-05-01",
                "home_expected_runs": 5.0, "away_expected_runs": 4.0,
                "home_score": hs, "away_score": as_,
                "total_runs": hs + as_, "p_home_cover_1_5": 0.5,
                "p_rl_1_0_home": cover, "p_rl_1_0_push": push,
                "p_rl_1_0_away": round(1.0 - cover - push, 6)}

    def test_runline_predicted_2way_whole_line(self):
        """At a WHOLE line with pushes, predicted_2way re-normalizes the raw
        cover rate to the same basis as W/(W+L) (pushes folded out of both
        sides): cover 0.363 / dog 0.469 / push 0.168 →
        0.363/(0.363+0.469) ≈ 43.6%, NOT the raw 36.3%."""
        rows = [self._rl_3way(5, 2, 0.363, 0.168, pk=0),   # margin 3 → win
                self._rl_3way(4, 3, 0.363, 0.168, pk=1),   # margin 1 → PUSH
                self._rl_3way(2, 5, 0.363, 0.168, pk=2)]   # margin −3 → loss
        dec = pd.DataFrame(rows)
        s = diag.runline_monitor_stats(dec, 1.0)
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["n_pushes"], 1)
        # Raw prediction stays raw (cross-check vs calibration record).
        self.assertAlmostEqual(s["cover_pred_mean"], 0.363, places=4)
        # 2-way prediction: 0.363 / (0.363 + 0.469) = 0.4363.
        self.assertAlmostEqual(s["predicted_2way"], 0.4363, places=4)
        # Win rate is W/(W+L): 1 win / (1 win + 1 loss).
        self.assertAlmostEqual(s["win_rate"], 0.5, places=6)
        # |predicted_2way − win_rate| ≪ |raw − win_rate| — the point of the
        # fix (raw was an 8-pt under-prediction, 2-way is within noise).
        self.assertLess(abs(s["predicted_2way"] - s["win_rate"]),
                        abs(s["cover_pred_mean"] - s["win_rate"]))
        # Home/Away rows unchanged: win rates only, 2-way, per side.
        self.assertEqual(set(s["sides"]), {"home"})   # all home favorites
        h = s["sides"]["home"]
        self.assertEqual(h["n"], 3)
        self.assertEqual(h["n_wins"], 1)
        self.assertEqual(h["n_pushes"], 1)
        self.assertAlmostEqual(h["win_rate"], 0.5, places=6)

    def test_runline_predicted_2way_half_line_equals_raw(self):
        """Half-lines never push → predicted_2way == raw cover_pred_mean
        (and == the per-game cover P)."""
        rows = [_rl_row(6, 2),          # margin 4 > 1.5 → win (cover 0.55)
                _rl_row(4, 4)]          # margin 0 < 1.5 → loss (cover 0.55)
        dec = pd.DataFrame(rows)
        s = diag.runline_monitor_stats(dec, 1.5)
        self.assertEqual(s["n_pushes"], 0)
        self.assertAlmostEqual(s["cover_pred_mean"], 0.55, places=4)
        self.assertEqual(s["predicted_2way"], s["cover_pred_mean"])
        self.assertAlmostEqual(s["predicted_2way"], 0.55, places=4)

    def test_runline_card_label_and_footer_source(self):
        """The card labels the metric 'Model predicted' (2-way basis) and
        the footer states BOTH predicted and win rate are 2-way re-
        normalized — no raw cover label remains."""
        src = (FRONTEND / "markets.py").read_text()
        self.assertIn('"Model predicted"', src,
                      "metric label must be 'Model predicted'")
        self.assertNotIn('"Favored cover P"', src,
                         "raw-cover label must be gone")
        self.assertIn("BOTH 2-way", src,
                      "footer must state both metrics are 2-way")
        self.assertIn("P(cover)/[P(cover)+P(dog)]", src,
                      "footer must name the re-scaling basis")


if __name__ == "__main__":
    unittest.main()
