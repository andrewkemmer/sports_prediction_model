"""NFL Totals & Run Lines dashboard tests (market-free) — the mirror of the
MLB frontend test conventions (test_frontend_markets / test_frontend_todays
_games) over the committed NFL slate-serve artifact:

  1. STRUCTURAL PARITY with the MLB markets page — the NFL page renders the
     SAME sections in the SAME order with the SAME tab labels, section
     headers, history-table schema, empty-state wording and honesty-note
     convention (source-derived from frontend/markets.py); no per-game
     "Scheduled Games" hero (MLB renders no per-game content on this page).
  2. Loader date resolution — utils resolves the newest
     nfl_run_engine_markets_*.csv / nfl_run_engine_monitor_*.json locally.
  3. Schema round-trip vs the mapping table — every fair-line / grid /
     derived column family the helpers read exists on the real artifact,
     and the spread/totals 3-way identities hold on every row.
  4. Rendering — the run-engine strip prices FAIR lines only (default +
     price-at-line), shows the integer push note, and the ±0.5 stop renders
     the per-side RAW cover WITH the grey-italic (ML X%) derived pair
     (they diverge by the tie rate on NFL — never conflated).
  5. Market-free invariants — no offered/book line, no shrink column, no
     market-derived edge is rendered by any NFL strip or page.
  6. Empty-state honesty — no artifact / no decided slate rows render
     MLB-shaped honest notices (research-pinned baseline with provenance or
     nothing); never fabricated slate calibration.
  7. Sport dispatch — markets.py delegates the NFL branch to the NFL page
     and stops before the MLB content; the MLB branch never touches it.

All pure Python (no Streamlit page context); artifact-dependent tests skip
gracefully when the committed artifacts are absent (e.g. the MLB-only
runner). Nothing is written to the repo.
"""
from __future__ import annotations

import json
import re
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
if str(FRONTEND) not in sys.path:
    sys.path.insert(0, str(FRONTEND))

import nfl_slate_view as sv  # noqa: E402

NFL_DD = ROOT / "nfl-backend" / "data_delivery"

MLB_MARKETS_SRC = (FRONTEND / "markets.py").read_text(encoding="utf-8")
NFL_MARKETS_SRC = (FRONTEND / "nfl_markets_page.py").read_text(encoding="utf-8")


def _norm(src: str) -> str:
    return re.sub(r"\s+", "", src)


def _artifact(glob_pattern: str) -> Path:
    hits = sorted(NFL_DD.glob(glob_pattern))
    if not hits:
        raise unittest.SkipTest(
            f"no NFL artifact {glob_pattern!r} on disk — nothing to verify")
    return hits[-1]


def _real_frame() -> pd.DataFrame:
    return pd.read_csv(_artifact("nfl_run_engine_markets_*.csv"))


def _synthetic_row(*, fair_spread=5, fair_total=45, mu_h=24.9, mu_a=20.4,
                   p_home_win_derived=0.635, p_push_0=0.00275,
                   home_cover_0=0.6335) -> dict:
    """Row-shaped dict exercising the pure price/render helpers."""
    row = {
        "fair_spread": fair_spread, "fair_total": fair_total,
        "mu_h": mu_h, "mu_a": mu_a, "mu_margin": mu_h - mu_a,
        "mu_total": mu_h + mu_a,
        "p_home_win_derived": p_home_win_derived,
        "p_away_win_derived": 1.0 - p_home_win_derived,
        "p_home_cover_0": home_cover_0, "p_push_0": p_push_0,
        "p_over_45": 0.50, "p_under_45": 0.47, "p_push_45": 0.03,
        "p_over_47": 0.38, "p_under_47": 0.58, "p_push_47": 0.04,
    }
    for L in sv.SPREAD_GRID:
        if L == 0:
            continue
        tag = f"m{-L}" if L < 0 else str(L)
        row[f"p_home_cover_{tag}"] = 0.5 - L * 0.01
        row[f"p_push_{tag}"] = 0.02
    return row


class TestMlbStructuralParity(unittest.TestCase):
    """The NFL page is a STRUCTURAL 1:1 mirror of the MLB markets page:
    same sections, same order, same tab labels, same history-table schema,
    same empty-state wording and honesty convention. Source-derived from
    frontend/markets.py (ground truth), never guessed."""

    def test_tab_labels_identical_and_ordered(self):
        import nfl_markets_page as page
        self.assertEqual(page.DIAG_TABS,
                         ["Distribution", "Relativized", "Pooled lines",
                          "Game Total Lines", "Run Lines"])
        mlb_norm = _norm(MLB_MARKETS_SRC)
        self.assertIn(
            'st.tabs(["Distribution","Relativized","Pooledlines",'
            '"GameTotalLines","RunLines",])', mlb_norm,
            "MLB's diagnostics tabs must be the ground-truth set")
        nfl_norm = _norm(NFL_MARKETS_SRC)
        self.assertIn('"Distribution","Relativized","Pooledlines",'
                      '"GameTotalLines","RunLines"', nfl_norm,
                      "NFL page must use the identical tab-label set/order")

    def test_section_order_identical(self):
        headers = ("### Diagnostics",
                   "### Prediction History — Totals & Run Lines",
                   "### Run-Line & Totals Monitor")
        for src, name in ((MLB_MARKETS_SRC, "MLB"), (NFL_MARKETS_SRC, "NFL")):
            idx = [src.index(h) for h in headers]
            self.assertEqual(idx, sorted(idx),
                             f"{name} section order must be Diagnostics → "
                             "Prediction History → Monitor")

    def test_title_wording_identical(self):
        self.assertIn("Today's Totals &amp; Run Lines", MLB_MARKETS_SRC)
        self.assertIn("Today's Totals &amp; Run Lines", NFL_MARKETS_SRC)

    def test_history_table_schema_identical(self):
        """The fb-table header schema is byte-identical (whitespace-
        normalized) between the two pages."""
        import nfl_markets_page as page
        mlb_run = ("<th>DATE</th><th>MATCHUP</th><th>SCORE (A–H)</th>"
                   "<th>LINE</th><th>MODEL PICK</th><th>WINNER</th>"
                   "<th>RESULT</th>")
        self.assertIn(_norm(mlb_run), _norm(MLB_MARKETS_SRC))
        self.assertEqual(_norm("<th>" + page.HISTORY_HEADERS + "</th>"),
                         _norm(mlb_run))

    def test_empty_state_wording_matches_mlb(self):
        for phrase in ("diagnostics need outcomes",
                       "Nothing is fabricated in the meantime."):
            self.assertIn(phrase, MLB_MARKETS_SRC)
            self.assertIn(phrase, NFL_MARKETS_SRC)

    def test_honesty_note_convention(self):
        for phrase in ("**Honesty note:**", "no styling hides it"):
            self.assertIn(phrase, MLB_MARKETS_SRC)
            self.assertIn(phrase, NFL_MARKETS_SRC)

    def test_no_per_game_hero(self):
        """MLB renders NO per-game content on the markets page (its
        docstring: the per-game slate board is UN-RENDERED), so the NFL
        mirror must not carry the week-selector / 'Scheduled Games'
        drill-down either."""
        self.assertNotIn("Scheduled Games", NFL_MARKETS_SRC)
        self.assertNotIn("nfl_markets_week", NFL_MARKETS_SRC)

    def test_market_free_page_source(self):
        """The page never reads the offered/shrink column families (the
        artifact keeps them; the model product does not) and never renders
        offered-line values. ('edge' wording appears only in the policy
        notes; the rendered-strip tests assert no market tokens in HTML.)"""
        for token in ("p_cover_offered", "p_push_offered", "p_over_offered",
                      "shrink_applied", "spread_line", "total_line"):
            self.assertNotIn(token, NFL_MARKETS_SRC,
                             f"page source must not read market token "
                             f"{token!r}")


class TestLoaderResolution(unittest.TestCase):
    """utils resolves the newest NFL run-engine family dates + artifacts."""

    def _empty_cfg(self):
        return {"owner": "", "repo": "", "branch": "main"}

    def test_family_dates_resolve_newest_locally(self):
        import utils
        with mock.patch.object(utils, "get_source_config",
                               return_value=self._empty_cfg()):
            dates = utils._nfl_run_engine_family_dates("nfl", "markets_csv")
        self.assertTrue(dates, "no nfl_run_engine_markets dates resolved")
        for d in dates:
            self.assertEqual(len(d), 8)
            self.assertTrue(d.isdigit())
        newest = _artifact("nfl_run_engine_markets_*.csv").name
        newest = newest[len("nfl_run_engine_markets_"):-len(".csv")]
        self.assertEqual(dates[0], newest, "newest-first resolution broken")

    def test_monitor_family_dates(self):
        import utils
        with mock.patch.object(utils, "get_source_config",
                               return_value=self._empty_cfg()):
            dates = utils._nfl_run_engine_family_dates(
                "nfl", "markets_monitor_json")
        self.assertTrue(dates)
        newest = _artifact("nfl_run_engine_monitor_*.json").name
        newest = newest[len("nfl_run_engine_monitor_"):-len(".json")]
        self.assertEqual(dates[0], newest)

    def test_load_returns_newest_markets(self):
        import utils
        with mock.patch.object(utils, "get_source_config",
                               return_value=self._empty_cfg()):
            df, date = utils.load_nfl_run_engine_markets("nfl")
        self.assertIsNotNone(date)
        self.assertGreater(len(df), 0)
        newest = _artifact("nfl_run_engine_markets_*.csv").name
        newest = newest[len("nfl_run_engine_markets_"):-len(".csv")]
        self.assertEqual(date, newest)
        self.assertTrue(sv.has_fair_columns(df))

    def test_load_monitor_returns_dict(self):
        import utils
        with mock.patch.object(utils, "get_source_config",
                               return_value=self._empty_cfg()):
            mon = utils.load_nfl_run_engine_monitor("nfl")
        self.assertIsInstance(mon, dict)
        # First-run monitor contract: research-pinned baseline + accumulating
        # slate-history section (empty by design on the first run).
        self.assertIn("oof_baseline_research_pinned", mon)
        self.assertIn("slate_history", mon)


class TestSchemaRoundTrip(unittest.TestCase):
    """The helpers' column contract round-trips against the real artifact
    and the mapping-table conventions (grid L = P(margin > L), push bands,
    derived pair complements)."""

    @classmethod
    def setUpClass(cls):
        cls.df = _real_frame()

    def test_fair_and_derived_families_present(self):
        df = self.df
        self.assertTrue(sv.has_fair_columns(df))
        for c in ("fair_spread", "fair_total", "mu_h", "mu_a",
                  "p_home_win_derived", "p_away_win_derived"):
            self.assertIn(c, df.columns, f"missing fair column {c}")

    def test_grid_families_match_engine_ranges(self):
        spreads, totals = sv.grid_rows(self.df)
        self.assertEqual(spreads, list(range(-14, 15)))
        self.assertEqual(totals, list(range(24, 67)))

    def test_derived_pair_is_complement_on_every_row(self):
        d = self.df[["p_home_win_derived", "p_away_win_derived"]]
        self.assertTrue(np.allclose(d["p_home_win_derived"]
                                    + d["p_away_win_derived"], 1.0, atol=1e-6))

    def test_spread_3way_identity_every_row(self):
        """Home cover + push + away cover == 1 over the integer margins:
        grid columns are the margin PMF split, never free numbers."""
        df = self.df
        for _, r in df.iterrows():
            for L in sv.SPREAD_GRID:
                ph, pp, pa = sv.price_spread(r, L)
                self.assertIsNotNone(ph)
                self.assertAlmostEqual(ph + pp + pa, 1.0, places=5,
                                       msg=f"row {r['game_id']} L={L}")

    def test_totals_3way_identity_every_row(self):
        df = self.df
        for _, r in df.iterrows():
            for U in sv.TOTAL_GRID:
                po, pu, pp = sv.price_total(r, U)
                self.assertIsNotNone(po)
                self.assertAlmostEqual(po + pu + pp, 1.0, places=5,
                                       msg=f"row {r['game_id']} U={U}")


class TestRenderMarketFree(unittest.TestCase):
    """Rendered strips carry ONLY model fair values + probabilities."""

    BANNED = ("offered", "shrink", "spread_line", "total_line",
              "has_offer", "edge")

    @classmethod
    def setUpClass(cls):
        cls.df = _real_frame()

    def test_no_market_tokens_in_any_strip(self):
        for _, r in self.df.iterrows():
            html = sv.runengine_html(r, str(r["home_team"]),
                                     str(r["away_team"]))
            low = html.lower()
            for tok in self.BANNED:
                self.assertNotIn(tok, low, f"market token {tok!r} rendered")

    def test_default_strip_uses_fair_lines(self):
        r = self.df.iloc[0]
        home, away = str(r["home_team"]), str(r["away_team"])
        fair_total = int(round(float(r["fair_total"])))
        fair_spread = int(round(float(r["fair_spread"])))
        html = sv.runengine_html(r, home, away)
        self.assertIn(f"O/U {fair_total}:", html)
        # Home-anchored pair at the fair line: home at -fair_spread (a home
        # 5-pt favorite renders "HOME −5"), away at the mirror. The grid
        # threshold sign is never inverted in the display.
        if fair_spread > 0:
            self.assertIn(f"{home} −{fair_spread}", html)
            self.assertIn(f"{away} +{fair_spread}", html)
        elif fair_spread < 0:
            self.assertIn(f"{home} +{abs(fair_spread)}", html)
            self.assertIn(f"{away} −{abs(fair_spread)}", html)
        # Derived ML pair present (P(H>A)/(1-P_tie)).
        self.assertIn("ML:", html)

    def test_price_at_selected_lines(self):
        r = self.df.iloc[0]
        home, away = str(r["home_team"]), str(r["away_team"])
        fair_total = int(round(float(r["fair_total"])))
        other = fair_total + 2
        if other not in sv.TOTAL_GRID:
            other = fair_total - 2
        html = sv.runengine_html(r, home, away, total_line=other)
        self.assertIn(f"O/U {other}:", html)
        # A chosen home spread renders the mirror pair (home +3 → away −3).
        spread_html = sv.runline_html(r, home, away, home_spread=3)
        self.assertIn(f"{home} +3", spread_html)
        self.assertIn(f"{away} −3", spread_html)

    def test_integer_push_notes_only_where_push_exists(self):
        seen_push = False
        for _, r in self.df.iterrows():
            home, away = str(r["home_team"]), str(r["away_team"])
            html = sv.runengine_html(r, home, away)
            fair_total = int(round(float(r["fair_total"])))
            _, _, pp = sv.price_total(r, fair_total)
            ou = html[html.index("O/U"):html.index("</span>",
                                                   html.index("O/U"))]
            if (pp or 0) > 0.005:
                self.assertIn("push", ou)
                seen_push = True
        self.assertTrue(seen_push, "no integer-total push note on the board")
        # A total with no push band shows NO totals push note, while the
        # run-line span keeps its own (2% push) — the note is per-line.
        row = _synthetic_row()
        row["p_push_45"] = 0.0
        plain = sv.runengine_html(row, "SEA", "NE", total_line=45)
        self.assertIn('O/U 45: Over 50% / Under 47%</span>', plain)
        self.assertNotIn("(3% push)", plain)
        self.assertIn("(2% push)", plain)

    def test_off_grid_lines_render_n_a_not_fabrication(self):
        r = self.df.iloc[0]
        self.assertEqual(sv.price_total(r, 100), (None, None, None))
        self.assertEqual(sv.price_spread(r, 30), (None, None, None))


class TestHalfStopRawVsDerived(unittest.TestCase):
    """The ±0.5 stop shows per-side RAW cover AND the (ML X%) derived pair;
    they diverge by the tie rate on NFL (raw -0.5 excludes ties, raw +0.5
    includes them) and are never collapsed into one number."""

    def test_home_favorite_pair(self):
        row = _synthetic_row(p_home_win_derived=0.635, home_cover_0=0.6335,
                             p_push_0=0.00275)
        html = sv.runline_html(row, "SEA", "NE", half_stop=True)
        # Favorite-anchored pair: home favorite on -0.5, away dog +0.5.
        self.assertIn("SEA −0.5 63%", html)
        self.assertIn("NE +0.5 37%", html)
        # Per-side grey-italic (ML X%) derived parentheticals present.
        self.assertIn("(ML 64%)", html)
        self.assertIn("(ML 36%)", html)
        fav_raw, dog_raw, fav_ml, dog_ml, fav_home = sv.half_stop_pair(row)
        self.assertTrue(fav_home)
        # Raw -0.5 excludes ties -> fav raw < fav ML; raw +0.5 includes
        # ties -> dog raw > dog ML.
        self.assertLess(fav_raw, fav_ml)
        self.assertGreater(dog_raw, dog_ml)
        # The tie mass sits inside the +0.5 leg: fav_raw + dog_raw == 1.
        self.assertAlmostEqual(fav_raw + dog_raw, 1.0, places=9)

    def test_away_favorite_pair(self):
        row = _synthetic_row(p_home_win_derived=0.47, home_cover_0=0.4685,
                             p_push_0=0.00275)
        html = sv.runline_html(row, "CAR", "CHI", half_stop=True)
        self.assertIn("CHI −0.5", html)      # away favorite on -0.5
        self.assertIn("CAR +0.5", html)
        fav_raw, dog_raw, fav_ml, dog_ml, fav_home = sv.half_stop_pair(row)
        self.assertFalse(fav_home)
        self.assertLess(fav_raw, fav_ml)
        self.assertGreater(dog_raw, dog_ml)
        self.assertAlmostEqual(fav_raw + dog_raw, 1.0, places=9)

    def test_divergence_holds_on_every_real_row_with_tie_mass(self):
        df = _real_frame()
        checked = 0
        for _, r in df.iterrows():
            p0 = sv._f(r, "p_push_0")
            if (p0 or 0) <= 0:
                continue
            fav_raw, dog_raw, fav_ml, dog_ml, fav_home = sv.half_stop_pair(r)
            self.assertIsNotNone(fav_raw)
            self.assertLess(fav_raw, fav_ml)
            self.assertGreater(dog_raw, dog_ml)
            self.assertAlmostEqual(fav_raw + dog_raw, 1.0, places=6)
            checked += 1
            if checked >= 25:
                break
        self.assertGreater(checked, 0,
                           "no real row carries a tie band — unexpected")

    def test_integer_lines_keep_push_note_and_no_ml_notes(self):
        row = _synthetic_row()
        html = sv.runline_html(row, "SEA", "NE", home_spread=-5)
        self.assertIn("(2% push)", html)     # p_push_5 = 0.02
        self.assertNotIn("(ML", html)        # no derived notes at integers


class TestEmptyStatesHonesty(unittest.TestCase):
    """Streamlit-stubbed page run: no artifact / no decided slate rows
    renders MLB-shaped honest notices (research-pinned baseline with
    provenance or nothing), never fabricated slate calibration."""

    def _run_page(self, slate, monitor):
        """Run the page under a fake streamlit + fake utils (the page's own
        st.* calls are recorded; real streamlit code never executes, so the
        MLB stub-state leak — real-streamlit lazy submodule imports seeing
        the MagicMock 'streamlit' — cannot happen)."""
        with mock.patch.dict(sys.modules):
            sys.modules.pop("nfl_markets_page", None)
            st = mock.MagicMock()
            st.tabs.side_effect = lambda labels: [mock.MagicMock()
                                                  for _ in labels]
            st.columns.side_effect = lambda n: [mock.MagicMock()
                                                for _ in range(n)]
            utils_fake = types.ModuleType("utils")
            utils_fake.inject_css = lambda *a, **k: None
            utils_fake.format_date_long = lambda d=None, **k: (str(d) if d
                                                               else "—")
            utils_fake.load_nfl_run_engine_markets = \
                lambda sport=None: (slate, None)
            utils_fake.load_nfl_run_engine_monitor = \
                lambda sport=None: monitor
            sys.modules["utils"] = utils_fake
            sys.modules["streamlit"] = st
            import nfl_markets_page as page
            page.run()
        return st

    def _text(self, st, *attrs):
        out = []
        for a in attrs:
            for call in getattr(st, a).call_args_list:
                args = call[0]
                if args and isinstance(args[0], str):
                    out.append(args[0])
        return "\n".join(out)

    def test_no_artifact_shows_honest_states(self):
        st = self._run_page(pd.DataFrame(), None)
        text = self._text(st, "warning", "info", "markdown")
        self.assertIn("No run-engine markets artifact for", text)
        self.assertIn("No decided OOF rows in nfl_run_engine_markets", text)
        self.assertIn("No run-engine monitor artifact for", text)
        # No fabricated calibration: with an empty baseline the tabs render
        # the honest info, never a metric grid with zeroed numbers.
        self.assertNotIn("Covers ECE", text)

    def test_artifact_without_decided_rows_shows_honest_states(self):
        df = _real_frame()
        has_scores = {"home_score", "away_score"}.issubset(df.columns)
        if has_scores and df[["home_score", "away_score"]].notna().any().any():
            self.skipTest("artifact already carries decided rows")
        mon = json.loads(_artifact("nfl_run_engine_monitor_*.json").read_text())
        st = self._run_page(df, mon)
        text = self._text(st, "warning", "info", "markdown", "caption",
                          "metric")
        # MLB-shaped decided-empty warnings + research-pinned stand-in.
        self.assertIn("No decided OOF rows in nfl_run_engine_markets", text)
        self.assertIn("No decided OOF rows in the run-engine markets "
                      "artifact", text)
        # Baseline metrics + provenance render (never fabricated slate
        # calibration).
        self.assertIn("Covers ECE (pooled OOF)", text)
        self.assertIn("research-pinned", text.lower())
        self.assertIn("Provenance:", text)
        self.assertIn("No slate history yet (first build starts empty).",
                      text)
        baseline = mon.get("oof_baseline_research_pinned") or {}
        self.assertTrue(baseline, "monitor baseline missing")

    def test_monitor_section_empty_history_state(self):
        mon = json.loads(_artifact("nfl_run_engine_monitor_*.json").read_text())
        self.assertEqual(mon.get("slate_history"), [],
                         "first-run monitor must ship an EMPTY accumulating "
                         "history — nothing fabricated")
        bl = mon.get("oof_baseline_research_pinned") or {}
        dm = bl.get("derived_ml") or {}
        # Research-pinned figures documented in the spec.
        self.assertAlmostEqual(bl["covers_ece_pooled"], 0.078, places=3)
        self.assertAlmostEqual(dm["logloss"], 0.6365, places=4)
        self.assertAlmostEqual(dm["auc"], 0.695, places=3)
        self.assertAlmostEqual(dm["ece"], 0.0435, places=4)


class TestSportDispatch(unittest.TestCase):
    """markets.py dispatches NFL to the NFL page and stops before the MLB
    content; the MLB branch never touches the NFL page module."""

    DISPATCH_MARKER = 'if utils.get_sport() == "nfl":'

    def test_dispatch_precedes_mlb_content(self):
        src = MLB_MARKETS_SRC
        self.assertIn(self.DISPATCH_MARKER, src)
        self.assertIn("nfl_markets_page.run()", src)
        self.assertIn("st.stop()", src)
        # The dispatch sits ABOVE the first MLB content statement.
        self.assertLess(src.index(self.DISPATCH_MARKER),
                        src.index("utils.inject_css()"))

    def _exec_dispatch_head(self, sport: str):
        """Exec only the module statements up to (excluding) the MLB content
        (the utils.inject_css() line), with utils.get_sport fixed to sport."""
        src = MLB_MARKETS_SRC
        head = src[:src.index("utils.inject_css()")]
        utils_fake = types.ModuleType("utils")
        utils_fake.get_sport = lambda: sport
        calls = {"run": 0, "stop": 0}
        page_fake = types.ModuleType("nfl_markets_page")
        page_fake.run = lambda: calls.__setitem__("run", calls["run"] + 1)
        st_fake = types.ModuleType("streamlit")
        st_fake.stop = lambda: calls.__setitem__("stop", calls["stop"] + 1)
        with mock.patch.dict(sys.modules, {
                "utils": utils_fake,
                "streamlit": st_fake,
                "nfl_markets_page": page_fake}):
            mod = types.ModuleType("markets")
            mod.__file__ = str(FRONTEND / "markets.py")
            sys.modules["markets"] = mod
            try:
                exec(compile(head, str(FRONTEND / "markets.py"), "exec"),
                     mod.__dict__)
            finally:
                sys.modules.pop("markets", None)
        return calls

    def test_nfl_sport_delegates_and_stops(self):
        calls = self._exec_dispatch_head("nfl")
        self.assertEqual(calls["run"], 1, "NFL must delegate to the page")
        self.assertEqual(calls["stop"], 1, "NFL must stop before MLB content")

    def test_mlb_sport_never_touches_nfl_page(self):
        calls = self._exec_dispatch_head("mlb")
        self.assertEqual(calls["run"], 0, "MLB must not delegate to NFL page")
        self.assertEqual(calls["stop"], 0, "MLB path must not stop early")

    def test_nfl_markets_page_import_is_side_effect_free(self):
        """Importing the page module must not render (render happens only
        inside run()), so the markets dispatch import is safe under tests."""
        with mock.patch.dict(sys.modules):
            sys.modules.pop("nfl_markets_page", None)
            sys.modules["streamlit"] = mock.MagicMock()
            import nfl_markets_page
            st = sys.modules["streamlit"]
            self.assertEqual(st.markdown.call_count, 0)
            self.assertEqual(st.selectbox.call_count, 0)


if __name__ == "__main__":
    unittest.main()