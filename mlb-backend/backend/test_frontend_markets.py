"""Task A tests: Markets URL fix — verify no double data_delivery/ path.

All tests are pure-Python (no Streamlit import). They verify:
  1. _raw_url produces a single data_delivery/ path.
  2. _load_markets source code does not prepend data_delivery/.
  3. Date resolution from available_dates is valid.
  4. Mocked fetch: 200 → DataFrame, 404 → None.
"""
import io
import inspect
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

# backend/ is a package under mlb-backend/ — put its parent on sys.path so
# the `backend.*` imports below resolve regardless of the runner's cwd
# (same idiom as test_ablation_defense.py; without it this module only
# imports when an earlier test module happens to have added the root).
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from backend.explainability import run_engine_feature_cols
from backend.run_engine import RUN_RESTORED_DIFF_FEATURES

# frontend/ moved to the repository root (multi-sport restructure, Phase B)
_frontend = Path(__file__).resolve().parents[2] / "frontend"
if str(_frontend) not in sys.path:
    sys.path.insert(0, str(_frontend))


# ---------------------------------------------------------------------------
# Direct URL-construction tests (no Streamlit needed)
# ---------------------------------------------------------------------------

# Replicate the _raw_url logic inline for pure testing
REPO_SUBDIR = "mlb-backend"


def _raw_url_test(relpath, owner, repo, branch):
    if owner and repo:
        return (f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}"
                f"/{REPO_SUBDIR}/data_delivery/{relpath}")
    return f"<local:{relpath}>"



def _latest_artifact(directory, pattern):
    """Find the most recent artifact matching pattern in directory.
    Returns Path or raises unittest.SkipTest if none found."""
    import unittest
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise unittest.SkipTest(f"No {pattern} artifacts found in {directory}")
    return matches[0]

class TestMarketsUrlConstruction(TestCase):
    def test_url_single_data_delivery(self):
        url = _raw_url_test("run_engine_markets_20260824.csv",
                            "owner", "repo", "main")
        self.assertIn("/data_delivery/run_engine_markets_20260824.csv", url)
        self.assertNotIn("data_delivery/data_delivery", url)
        self.assertTrue(url.startswith("https://raw.githubusercontent.com/"))

    def test_url_without_owner_shows_local(self):
        url = _raw_url_test("foo.csv", "", "", "main")
        self.assertIn("local:", url)

    def test_markets_relpath_is_bare_filename(self):
        """Source-code inspection: _load_markets must use a bare filename."""
        with open(_frontend / "markets.py") as f:
            src = f.read()
        self.assertIn('"run_engine_markets_{ds}.csv"', src,
                      "_load_markets must use bare filename")
        self.assertNotIn('"data_delivery/run_engine_markets',
                         src,
                         "BUG: markets.py prepends data_delivery/")


# ---------------------------------------------------------------------------
# Fetch smoke tests (no Streamlit)
# ---------------------------------------------------------------------------

class TestMarketsFetchMocked(TestCase):
    """Verify the URL that _load_markets would construct passes a sanity check."""

    def test_correct_repo_file_exists_on_github(self):
        """The LATEST committed run_engine_markets artifact must be accessible
        from the repo — this is the actual file the frontend fetches (the
        markets page always loads the most recent run, so the pin must
        follow the newest local artifact, not a hardcoded date that Phase 6
        retention eventually prunes; the old 20260824 pin rotted exactly
        that way). Network-gated: skipped offline so the suite stays green
        without connectivity; the published-artifact assertion remains
        meaningful whenever the network is up."""
        import requests
        import unittest
        dd = Path(__file__).resolve().parents[1] / "data_delivery"
        # Pin the CANONICAL artifact — never the *_rl bridge copy (a local
        # calibration/persist bridge artifact that is not the file the
        # frontend loads). The canonical file is what the markets page and
        # the run-engine card actually fetch.
        latest = _latest_artifact(dd, "run_engine_markets_*.csv")
        for cand in sorted(dd.glob("run_engine_markets_*.csv"),
                           reverse=True):
            if "_rl." not in cand.name:
                latest = cand
                break
        if latest is None:
            self.skipTest(
                "no local run_engine_markets_*.csv to pin against "
                "(data_delivery empty) — nothing to verify is published")
        fname = latest.name
        url = _raw_url_test(fname,
                            "andrewkemmer", "sports_prediction_model", "main")
        try:
            resp = requests.get(url, timeout=15)
        except requests.RequestException as exc:
            self.skipTest(f"network unavailable ({exc.__class__.__name__}) — "
                          f"cannot verify {fname} is published")
        self.assertTrue(resp.ok, f"Expected HTTP 200 from {url}, got {resp.status_code}")
        # Must parse as CSV
        df = pd.read_csv(io.BytesIO(resp.content))
        self.assertGreater(len(df), 0, "Markets artifact is empty")
        self.assertIn("game_pk", df.columns, "Missing game_pk column")


# ---------------------------------------------------------------------------
# Date resolution
# ---------------------------------------------------------------------------

class TestDateResolution(TestCase):
    """Local available_dates returns valid YYYYMMDD."""

    def test_local_globs_valid_dates(self):
        dd = Path(__file__).resolve().parents[1] / "data_delivery"
        dates = set()
        for p in dd.glob("todays_games_*.csv"):
            d = p.name[len("todays_games_"):-len(".csv")]
            if len(d) == 8 and d.isdigit():
                dates.add(d)
        self.assertGreater(len(dates), 0, "No todays_games files found")
        for d in dates:
            self.assertEqual(len(d), 8, f"Not YYYYMMDD: {d}")
            self.assertTrue(d.isdigit(), f"Not digits: {d}")


# ---------------------------------------------------------------------------
# Distributional-fit panel (pure extraction — no Streamlit)
# ---------------------------------------------------------------------------


class TestFitPanel(TestCase):
    """Fit-panel reader keys reconciled with pipeline._run_engine_fit_block's
    writer schema (alpha_home/alpha_away curves, dispersion_chi2_per_df,
    fit_tables, variance_check, mc_meta). The historical crash: the MC line
    formatted the '--' string default with thousands separators
    (``f"{:,.}"`` -> ValueError).
    """

    @classmethod
    def setUpClass(cls):
        import market_diagnostics as diag
        cls.diag = diag
        # backend/ and data_delivery/ stay siblings under mlb-backend/
        cls.root = Path(__file__).resolve().parents[1]

    @staticmethod
    def _fit(date_str=None):
        root = Path(__file__).resolve().parents[1] / "data_delivery"
        if date_str:
            p = root / f"run_engine_monitor_{date_str}.json"
        else:
            p = _latest_artifact(root, "run_engine_monitor_*.json")
        with open(p) as f:
            import json as _json
            return _json.load(f)["fit"]

    def test_empty_fit_dict_never_crashes(self):
        """Regression: a fit dict missing EVERY key renders all-default rows."""
        for fit in ({}, None):
            rows = self.diag.fit_panel_rows(fit)
            self.assertIsNone(rows["alpha_home"])
            self.assertIsNone(rows["alpha_away"])
            self.assertIsNone(rows["chi2_home"])
            self.assertIsNone(rows["chi2_away"])
            self.assertEqual(rows["variance_home"], (None, None))
            self.assertEqual(rows["variance_away"], (None, None))
            self.assertEqual(rows["tails"], {})
            self.assertIsNone(rows["mc_caption"])
            self.assertIsNone(rows["alpha_home_form"])

    def test_mc_caption_formats_only_numeric(self):
        """The historical crash: '--' string default hit f"{:,.}"."""
        self.assertEqual(self.diag.mc_caption({"n_draws": 10000}),
                         "Monte Carlo: 10,000 draws")
        self.assertEqual(self.diag.mc_caption({"n_draws": 10000,
                                               "mc_se_totals_max": 0.005}),
                         "Monte Carlo: 10,000 draws · totals MC se max 0.0050")
        # legacy key spelling + absent key: never raise, never format a str
        self.assertIsNone(self.diag.mc_caption({}))
        self.assertIsNone(self.diag.mc_caption({"n_samples": "--"}))
        self.assertIn("5,000", self.diag.mc_caption({"n_samples": 5000}))

    def test_real_artifact_values(self):
        """Acceptance: the current monitor JSON's fit block renders REAL
        numbers (alpha per side, chi2/df, variance, MC meta, tails)."""
        fit = self._fit()
        rows = self.diag.fit_panel_rows(fit)
        # chi2/df straight from dispersion_chi2_per_df
        self.assertAlmostEqual(rows["chi2_home"], 2.142, places=3)
        self.assertAlmostEqual(rows["chi2_away"], 2.366, places=3)
        # alpha = count-weighted bin mean (home ~0.258, away ~0.316 on the
        # 09-03 monitor artifact, post-P1 projection adoption 8cb4efc)
        self.assertAlmostEqual(rows["alpha_home"], 0.258, places=3)
        self.assertAlmostEqual(rows["alpha_away"], 0.316, places=3)
        self.assertEqual(rows["alpha_home_form"], "piecewise")
        self.assertEqual(rows["alpha_away_form"], "piecewise")
        # variance check: implied vs observed per side
        vh = rows["variance_home"]
        self.assertAlmostEqual(vh[0], 9.442, places=3)
        self.assertAlmostEqual(vh[1], 9.624, places=3)
        va = rows["variance_away"]
        self.assertAlmostEqual(va[0], 10.457, places=3)
        self.assertAlmostEqual(va[1], 10.682, places=3)
        # MC metadata: numeric n_draws with separators
        self.assertIn("10,000 draws", rows["mc_caption"])
        # tails: real k labels (unicode ≥/≤) with observed_p/modeled_p keys
        home_tail = rows["tails"]["Home"]
        self.assertIn("k≤1", home_tail)
        self.assertIn("k≥10", home_tail)
        self.assertIn("obs=0.165", home_tail)  # home ≤1 observed (3dp)
        self.assertEqual(rows.get("fitted_on"), "pre-holdout OOF only")

    def test_v1_curve_shape_supported(self):
        """v1 curves (parametric form/a/c + selection bins) also yield α."""
        fit = self._fit("20260826")
        rows = self.diag.fit_panel_rows(fit)
        self.assertIsNotNone(rows["alpha_home"])
        self.assertIsNotNone(rows["alpha_away"])
        self.assertIsNotNone(rows["alpha_home_form"])
        self.assertIsNotNone(rows["chi2_home"])
        # fixture tail rows render too
        self.assertIn("Home", rows["tails"])

    def test_lambda_edge_from_real_fit(self):
        """λ edge (home−away modeled run differential) from the fit-curve bin
        means (the pre-holdout fit scope; ~flat on the 6,953-frame artifact)."""
        fit = self._fit()
        edge = self.diag.lambda_edge(fit)
        self.assertIsNotNone(edge)
        self.assertAlmostEqual(edge, 0.0127, places=4)  # fit-curve bin means
        # (pin-synced to the 09-03 monitor artifact: the P1 projection
        # adoption 8cb4efc changed the lambda basis -> refit edge moved)
        self.assertIsNone(self.diag.lambda_edge({}))
        self.assertIsNone(self.diag.lambda_edge(None))


class TestRunEngineModelMonitorRender(TestCase):
    """The run-line monitor's new sections render the REAL artifacts without
    crashing (model card + drift + coverage), with streamlit stubbed."""

    @classmethod
    def setUpClass(cls):
        import sys as _sys
        import unittest.mock as _mock
        cls._backup = _sys.modules.get("streamlit")
        _sys.modules["streamlit"] = _mock.MagicMock()
        # If real streamlit was imported earlier in the suite (any frontend
        # test that ran before this one), `utils` is already cached with its
        # real `st` binding and markets.py's MODULE-LEVEL utils.inject_css()
        # executes against real streamlit outside a runtime (config/logger
        # crash).  Drop the streamlit-bound frontend modules so the stub is
        # in place BEFORE markets.py runs — same canonical end-state the
        # suite documents (utils.st left on the stub).
        for _mod in ("utils", "markets"):
            _sys.modules.pop(_mod, None)
        cls.markets = __import__("markets")
        # backend/ and data_delivery/ stay siblings under mlb-backend/
        cls.root = Path(__file__).resolve().parents[1]

    @classmethod
    def tearDownClass(cls):
        import sys as _sys
        if cls._backup is not None:
            _sys.modules["streamlit"] = cls._backup
        else:
            _sys.modules.pop("streamlit", None)
        # cls.markets keeps its stub-bound utils reference (its module-level
        # render code cannot run outside a runtime), but LATER suite modules
        # (e.g. test_frontend_nfl_slate loaders) must import utils under the
        # REAL streamlit — re-import it now so the stub copy cached by
        # setUpClass does not poison the rest of the process.
        _sys.modules.pop("utils", None)
        import utils as _real_utils  # noqa: F401  (re-import under real st)

    def test_model_card_renders_real_artifact(self):
        import json
        mon = json.loads((_latest_artifact(self.root / "data_delivery", "run_engine_monitor_*.json")).read_text())
        self.markets._render_run_engine_model_card(mon)  # no crash

    def test_run_engine_drift_caption_has_no_feature_counts(self):
        """The run-engine drift caption/body must not hardcode a feature
        count (it went stale at 29/36 after the 53-feature restore) — title
        is the bare 'Run-Engine Feature Drift (PSI)' and the body keeps the
        useful legend without any keep/drop numbers."""
        src = Path(__file__).resolve().parents[2] / "frontend" / "markets.py"
        text = src.read_text(encoding="utf-8")
        self.assertIn("### Run-Engine Feature Drift (PSI)", text)
        self.assertNotIn("its own 29 features", text)
        self.assertNotIn("OWN 29", text)
        self.assertNotIn("the 36 dropped", text)
        self.assertIn("INSUFFICIENT = window too small to judge drift", text)

    def test_run_engine_drift_model_weight_column_renders(self):
        """MODEL WEIGHT column (layout parity with the Model Monitor): header
        sits between PSI and STATUS, cells formatted by utils.feature_weight_pct
        (e.g. '2.47%'), and a run-engine feature with no weight renders '—'."""
        d = pd.DataFrame.from_records([
            {"feature": "elo_diff", "current_mean": 12.0, "baseline_mean": 10.0,
             "psi": 0.233, "status": "OK"},
            {"feature": "unweighted_feat", "current_mean": 1.0,
             "baseline_mean": 1.0, "psi": 0.011, "status": "OK"},
        ])
        self.markets.st.reset_mock()
        self.markets._render_run_engine_drift(
            d, weights={"elo_diff": 2.473, "other": 5.0})
        html = "".join(c[0][0] for c in self.markets.st.markdown.call_args_list)
        self.assertIn("<th>MODEL WEIGHT</th>", html)
        self.assertIn("2.47%", html)          # formatted elo_diff weight
        self.assertIn("—", html)              # missing weight → '—' fallback
        self.assertIn("<th>STATUS</th>", html)
        self.assertIn("INSUFFICIENT = window too small to judge drift", html)

    def test_run_engine_drift_model_weight_absent_omits_column(self):
        """No weight source available → the MODEL WEIGHT column (header + cells)
        is omitted entirely; the table still renders (graceful, never crashes)."""
        d = pd.DataFrame.from_records([
            {"feature": "elo_diff", "current_mean": 1.0, "baseline_mean": 1.0,
             "psi": 0.2, "status": "OK"},
        ])
        self.markets.st.reset_mock()
        self.markets._render_run_engine_drift(d, weights={})
        html = "".join(c[0][0] for c in self.markets.st.markdown.call_args_list)
        # The column HEADER/data cells are omitted (the footnote may still name
        # the column); the table still renders with FEATURE..STATUS intact.
        self.assertNotIn("<th>MODEL WEIGHT</th>", html)
        self.assertIn("<th>STATUS</th>", html)

    def test_drift_and_coverage_render_real_artifacts(self):
        d = pd.read_csv(_latest_artifact(self.root / "data_delivery",
                                          "run_engine_feature_drift_*.csv"))
        # 55 = 53 derive_run_features kept + 2 P1 projection level inputs
        # (sp_proj_era_home, sp_proj_era_away) added by the monitoring-gap fix.
        # The committed artifact is the pre-fix 53-row version until the next
        # pipeline run, so this pins a subset relationship (the count pins are
        # in the post-fix re-emit verification).
        self.assertGreaterEqual(len(d), 53)
        self.markets._render_run_engine_drift(d)
        c = pd.read_csv(_latest_artifact(self.root / "data_delivery",
                                          "run_engine_feature_coverage_*.csv"))
        self.assertGreaterEqual(len(c), 106)
        self.markets._render_run_engine_coverage(c)

    def test_empty_states_never_crash(self):
        self.markets._render_run_engine_drift(None)
        self.markets._render_run_engine_coverage(pd.DataFrame())
        self.markets._render_run_engine_model_card(
            {"fit": {}, "phase1": {}, "market_metrics": {}})

    def test_drift_monitor_covers_all_active_features(self):
        """DURABLE PIN (post-P1-adoption + monitoring-gap fix):
        the run-engine drift/coverage monitor must enumerate EVERY model-input
        column — the 53 derive_run_features kept columns (24 restored diffs +
        29 kept) PLUS the 2 P1 projection level inputs (sp_proj_era_home,
        sp_proj_era_away) that build_side_frame appends at runtime. This pin
        guarantees that a future adopted input cannot silently vanish from
        monitoring (the gap that existed before the fix is documented in
        mlb_run_engine_proj_drift_monitoring_*.json). The committed CSV still
        reflects the last pre-fix run (53) until the next pipeline run."""
        cols = run_engine_feature_cols()
        self.assertEqual(len(cols), 55)
        # All 53 derive_run_features kept columns are covered.
        from run_engine import derive_run_features
        from training import FEATURE_COLS
        feats, _ = derive_run_features(list(FEATURE_COLS))
        self.assertTrue(set(feats) <= set(cols),
                        "every derive_run_features kept col must be in the drift enumeration")
        # The 2 P1 projection level inputs are covered.
        self.assertIn("sp_proj_era_home", cols)
        self.assertIn("sp_proj_era_away", cols)
        # All 24 restored diffs are covered (still true after the fix).
        self.assertTrue(set(RUN_RESTORED_DIFF_FEATURES) <= set(cols))
        # The excluded margin/composite features are NOT monitored.
        self.assertNotIn("run_margin_diff", cols)
        for f in ("lineup_handedness_matchup_advantage", "bullpen_meltdown_risk",
                  "pitcher_regression_indicator", "lineup_depth_multiplier",
                  "ace_efficiency_factor"):
            self.assertNotIn(f, cols)

    def test_winner_cards_render_real_auc_values(self):
        """The winner-card renderer runs on the REAL monitor JSON and every
        card carries a pooled + holdout AUC (the value, not '—')."""
        import json
        import streamlit as _st
        mon = json.loads((_latest_artifact(
            self.root / "data_delivery", "run_engine_monitor_*.json"))
            .read_text())
        wc = mon.get("winner_cards") or {}
        # the stubbed st.columns(n) must unpack into n column objects
        _st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
        try:
            self.markets._render_winner_cards(wc)  # no crash
        finally:
            _st.columns.side_effect = None
        for name in ("over_under", "run_line", "derived_ml"):
            c = wc.get(name) or {}
            self.assertIsNotNone(c.get("auc"),
                                 f"{name} card must carry pooled auc")
            self.assertTrue(np.isfinite(c["auc"]), f"{name} auc finite")
            self.assertGreater(c["auc"], 0.5)
            self.assertLess(c["auc"], 1.0)
            self.assertIsNotNone((c.get("holdout") or {}).get("auc"),
                                 f"{name} holdout must carry auc")

    def test_winner_card_footer_and_auc_source_updated(self):
        """The derived-ML footer dropped the stale 'underweights the home
        edge' claim (calibrated post-fix) and the card renders holdout AUC."""
        src = (Path(__file__).resolve().parents[2]
               / "frontend" / "markets.py").read_text(encoding="utf-8")
        self.assertIn("Holdout AUC", src,
                      "renderer must show the holdout AUC metric")
        self.assertIn("calibrated post-fix", src,
                      "footer must reflect the calibrated derived ML")
        self.assertNotIn("underweights the home edge", src,
                         "stale underweighting claim must be gone")

    def test_diagnostics_tab_uses_game_total_lines_not_own_rounded_line(self):
        """The totals diagnostics tab is now 'Game Total Lines': an 'All'
        option (each game at its own fair line, 1-pt buckets) plus fixed
        lines, one shared code path. The old 'Totals picks' tab is deleted;
        per-game own-ROUNDED pricing is gone entirely."""
        src = (Path(__file__).resolve().parents[2]
               / "frontend" / "markets.py").read_text(encoding="utf-8")
        # The tab labels live in the shared market_diagnostics.DIAG_TABS
        # constant (both sports' pages render it), so the renamed label is
        # pinned in that module's source, not the page's inline list.
        diag_src = (Path(__file__).resolve().parents[2]
                    / "frontend" / "market_diagnostics.py").read_text(
                        encoding="utf-8")
        self.assertIn('"Game Total Lines"', diag_src,
                      "tab renamed to Game Total Lines (shared DIAG_TABS)")
        self.assertIn('["All"] + [str(l) for l in diag.TOTAL_GRID]', src,
                      "selector must offer All first, then the grid")
        self.assertIn("diag.game_total_calibration", src,
                      "tab must pool via the shared game-total helper")
        self.assertIn("diag.chart_game_total_curve", src,
                      "tab must render the bars + curves + diagonal view "
                      "via the restored bespoke GTL builder")
        self.assertIn("Calibration Curve — Over", src,
                      "dynamic title: 'Calibration Curve — Over <line>'")
        self.assertNotIn('built["scatter"]', src,
                         "the bottom observed-vs-predicted scatter must be gone "
                         "from the render path")
        self.assertNotIn("Observed vs predicted per bucket", src,
                         "the deleted scatter's heading/renderer must be gone "
                         "from the tab")
        self.assertIn("diag_game_total_line", src,
                      "line selector key present")
        # The GTL chart is the restored bespoke builder (chart_game_total_curve,
        # full 0-1 domain, no spec width) and renders through the SAME
        # utils.show_chart call as Distribution -- width owned by the render.
        self.assertIn('utils.show_chart(built["chart"])', src,
                      "GTL chart must render through the SAME call as Distribution")
        self.assertIn(
            "built = diag.chart_game_total_curve(",
            src, "GTL chart built via the shared bespoke builder")
        self.assertIn(
            "glc, _gl_title, curve_bins=glc.get(\"curve_bins\")",
            src, "GTL builder call feeds the 1-pt curve frame")
        # The GTL tab now mirrors the Run Lines layout exactly: same kwargs
        # (1-pt curve frame, 1% ticks, no Win rate series, table labels).
        self.assertIn("curve_bins=glc.get(\"curve_bins\")", src,
                      "GTL must feed the 1-pt curve frame to the builder")
        self.assertIn("x_tick_values=diag.X_1PCT_TICKS", src,
                      "GTL must pass the explicit 1% x-axis tick values")
        self.assertIn("show_win_rate=False", src,
                      "GTL must drop the Win rate series")
        self.assertIn('x_label="Mean Predicted"', src,
                      "GTL x-axis renamed to 'Mean Predicted'")
        self.assertIn('series_label="Mean Actual"', src,
                      "GTL series renamed to 'Mean Actual'")
        self.assertIn('obs_label="Mean Actual"', src,
                      "GTL y-axis renamed to 'Mean Actual'")
        self.assertNotIn("diag.rounded_total_pairs", src,
                         "per-game own-line pricing must be gone from the tab")
        self.assertNotIn("Per-game rounded total", src,
                         "old own-line chart title must be gone")
        self.assertNotIn('"Totals picks"', src,
                         "the Totals picks tab is deleted (absorbed by All)")
        self.assertNotIn("totals_pick_table", src,
                         "the deleted tab's renderer path must be gone")

    def test_pooled_lines_caption_explains_the_chart(self):
        """The Pooled lines footer must explain in plain terms how the chart
        works: every game priced at the four lines, what X/Y mean, the
        diagonal = perfect calibration, and the non-independence honesty
        note (27,248 pairs = 4 per game, effective sample ~6,812)."""
        src = (Path(__file__).resolve().parents[2]
               / "frontend" / "markets.py").read_text(encoding="utf-8")
        # Sub-heading unchanged
        self.assertIn("Games pooled across 7.5 / 8.5 / 9.5 / 10.5", src)
        # Chart path unchanged (same four lines, same helpers)
        self.assertIn("fixed_line_pairs(decided, (7.5, 8.5, 9.5, 10.5))", src)
        self.assertIn("chart_calibration", src)
        # New caption: the four lines, the example spread, X/Y meaning,
        # the diagonal, and the non-independence honesty note
        self.assertIn("How to read this", src)
        self.assertIn("0.62 / 0.48 / 0.32 / 0.19", src)
        self.assertIn("predicted P(over)", src)
        self.assertIn("The dashed diagonal is perfect calibration", src)
        self.assertIn("NOT independent", src)
        self.assertIn("game appears 4×", src)
        self.assertIn("effective sample is the ~6,812 games", src)
        # The old cryptic footer is gone
        self.assertNotIn("near-line degeneracy", src)


class TestRelativizedDeepOverCallout(TestCase):
    """The Relativized tab's deep-over callout was refreshed from the stale
    2026-08-24 measurement (prediction 0.66 vs actual 0.60, n≈4,156) to the
    current 2026-08-30 recheck (gap gone at −2.0; 2-way now under-prices the
    over). Source guard: the stale numbers must not return, and the page must
    cite the recheck record."""

    @classmethod
    def setUpClass(cls):
        cls.src = (_frontend / "markets.py").read_text(encoding="utf-8")

    def test_stale_20260824_numbers_gone(self):
        for stale in ("0.66 vs actual ≈ 0.60", "≈ 0.06 shortfall, n ≈ 4,156",
                      "prediction ≈ 0.66", "n ≈ 4,156", "+0.058 vs +0.054"):
            self.assertNotIn(stale, self.src,
                             f"stale deep-over number must be gone: {stale!r}")
        self.assertNotIn("weather-independent — the gap is identical",
                         self.src, "stale weather claim must be gone")

    def test_fresh_recheck_text_present(self):
        self.assertIn("deep_over_recheck_20260830.json", self.src,
                      "callout must cite the fresh recheck record")
        self.assertIn("0.606 vs actual 0.607", self.src,
                      "callout must carry the −2.0 pred/actual")
        self.assertIn("0.636 vs 0.647", self.src,
                      "callout must carry the 2-way under-price at −2.0")
        self.assertIn("EV-haircut on deep-over lines would give away",
                      self.src, "callout must warn against the harmful haircut")


# ---------------------------------------------------------------------------
# Date-pinning regression: Totals & Run Lines ignores selected_date
# ---------------------------------------------------------------------------

class TestMarketsAlwaysLatestArtifact(TestCase):
    """markets.py must always load the LATEST artifact (dates[0]),
    ignoring the date picked on Today's Games — same pattern as
    model_calibration.py.  A past-date selection must NEVER change
    which run_engine_markets_* file is loaded."""

    def test_source_always_uses_dates_0(self):
        """markets.py must resolve date_str from dates[0], not from
        st.session_state["selected_date"]."""
        src = (_frontend / "markets.py").read_text(encoding="utf-8")
        # The assignment must reference dates[0], not selected_date
        self.assertIn("dates[0]", src,
                      "markets.py must always use dates[0] (latest run)")
        self.assertNotIn('selected_date', src,
                         "markets.py must not read selected_date")

    def test_load_markets_contract_is_date_agnostic(self):
        """_load_markets resolves the CSV filename from the date string
        it receives.  The fix in markets.py ensures it always receives
        dates[0] (the latest run), so even if a user selects a past date
        on Today's Games, the function never sees it.

        This test verifies: (a) the function signature accepts date_str,
        (b) when the current artifact exists it parses cleanly with kind
        column (Phase 3+).
        """
        import io as _io
        root = Path(__file__).resolve().parents[2]
        dd = root / "mlb-backend" / "data_delivery"
        latest = _latest_artifact(dd, "run_engine_markets_*.csv")
        csv_bytes = latest.read_bytes()
        df = pd.read_csv(_io.BytesIO(csv_bytes))
        self.assertGreater(len(df), 0)
        self.assertIn("kind", df.columns,
                      "Current artifact must have the kind column (Phase 3+)")
        # Verify the _load_markets function accepts date_str as its arg
        # by reading the source directly (no Streamlit import needed)
        src = (_frontend / "markets.py").read_text(encoding="utf-8")
        self.assertIn("def _load_markets(ds):", src,
                      "_load_markets must accept a date string parameter")


class TestRunlinePicksChart(TestCase):
    """Verify the Run-line picks chart title, pick-set composition, and
    x-axis label match the code-verified behavior.

    Code verification (market_diagnostics.runline_pick_table):
    (a) Mixed rule: home -1.5 cover if P(home cover) >= 0.5, else
        away +1.5 — every game picks one side, so bucket counts sum
        to the decided-game total.
    (b) X-axis: max(P(home -1.5 cover), 1 - P(home -1.5 cover)) =
        picked-side probability.
    (c) Accuracy: match(pick, outcome) where home_covers = margin >= 2.
    (d) n_games == len(decided_rows).
    """

    def test_run_lines_tab_is_calibration_view(self):
        """The 'Run-line picks' tab is renamed 'Spread Lines' (2026-09-05;
        shared market_diagnostics.DIAG_TABS constant) and replaced by the
        favorite-side calibration view (run_line_calibration + the same
        chart_game_total_curve builder + utils.show_chart as Distribution)."""
        src = (_frontend / "markets.py").read_text(encoding="utf-8")
        self.assertIn("diag.DIAG_TABS", src,
                      "tabs must render the shared DIAG_TABS constant")
        diag_src = (_frontend / "market_diagnostics.py").read_text(
            encoding="utf-8")
        self.assertIn('"Spread Lines"', diag_src,
                      "the shared constant must carry the renamed tab")
        # The constant VALUE is what must drop the old label — the module
        # source legitimately references 'Run Lines' in docstrings/comments
        # (e.g. the ±0.5 identity note), so pin the constant, not the file.
        import market_diagnostics as _diag_mod
        self.assertNotIn("Run Lines", _diag_mod.DIAG_TABS,
                         "the old tab label must be gone from the shared "
                         "constant")
        self.assertIn("diag.run_line_calibration(decided, _rl_line)", src,
                      "tab must build the run-line calibration table")
        self.assertIn("key=\"diag_run_line\"", src,
                      "run-line selector key must be diag_run_line")
        self.assertIn("Calibration Curve — Favorite", src,
                      "dynamic title must say 'Calibration Curve — Favorite {L}'")
        # Same render path as Distribution / Game Total Lines.
        self.assertIn("utils.show_chart(built[\"chart\"])", src)
        # Run Lines: 1-pt curve frame + explicit 1% x-axis tick marks on
        # every selection; bars + table stay 5-pt. GTL passes neither.
        self.assertIn("curve_bins=rlc.get(\"curve_bins\")", src,
                      "Run Lines must feed the 1-pt curve frame to the builder")
        self.assertIn("x_tick_values=diag.X_1PCT_TICKS", src,
                      "Run Lines must pass the explicit 1% x-axis tick values")
        self.assertIn("diag.X_1PCT_TICKS", src,
                      "the 1% tick constant must be referenced by the tab")
        # Run Lines hides the Win rate series and renames the labels to the
        # table's column names; the GTL call passes none of these.
        self.assertIn("show_win_rate=False", src,
                      "Run Lines must drop the Win rate series")
        self.assertIn('x_label="Mean Predicted"', src,
                      "x-axis renamed to 'Mean Predicted'")
        self.assertIn('series_label="Mean Actual"', src,
                      "series renamed to 'Mean Actual'")
        self.assertIn('obs_label="Mean Actual"', src,
                      "y-axis renamed to 'Mean Actual'")

    def test_runline_pick_table_pick_rule_mentions_both_sides(self):
        """runline_pick_table pick_rule must mention both home -1.5
        AND away +1.5 (the mixed rule)."""
        from market_diagnostics import runline_pick_table, decided_rows
        root = Path(__file__).resolve().parents[2]
        dd = root / "mlb-backend" / "data_delivery"
        markets = _load_markets(dd)
        if markets is None:
            self.skipTest("No markets artifact available")
        decided = decided_rows(markets)
        if decided.empty:
            self.skipTest("No decided rows in markets artifact")
        result = runline_pick_table(decided)
        rule = result.get("pick_rule", "")
        self.assertIn("home", rule.lower(),
                      "Pick rule must mention home side")
        self.assertIn("away", rule.lower(),
                      "Pick rule must mention away side")
        self.assertIn("-1.5", rule, "Pick rule must mention -1.5")
        self.assertIn("+1.5", rule, "Pick rule must mention +1.5")

    def test_bucket_counts_sum_to_decided_total(self):
        """Every decided game must appear in exactly one bucket;
        sum of bucket counts == n_games."""
        from market_diagnostics import runline_pick_table, decided_rows
        root = Path(__file__).resolve().parents[2]
        dd = root / "mlb-backend" / "data_delivery"
        markets = _load_markets(dd)
        if markets is None:
            self.skipTest("No markets artifact available")
        decided = decided_rows(markets)
        if decided.empty:
            self.skipTest("No decided rows in markets artifact")
        result = runline_pick_table(decided)
        self.assertFalse(result.get("warning"),
                         f"runline_pick_table warned: {result.get('warning')}")
        buckets = result.get("buckets", [])
        self.assertGreater(len(buckets), 0, "No buckets produced")
        total_in_buckets = sum(b["count"] for b in buckets)
        self.assertEqual(total_in_buckets, result["n_games"],
                         "Bucket counts must sum to n_games")

    def test_run_lines_caption_describes_calibration(self):
        """The renamed 'Spread Lines' tab's caption describes the favorite-
        side 2-way cover calibration (P(cover) band on the x-axis, favorite
        cover rate on the y-axis, the 'V' pick convention)."""
        src = (_frontend / "markets.py").read_text(encoding="utf-8")
        self.assertIn("predicted P(cover) band", src,
                      "caption must describe the x-axis as a P(cover) band")
        self.assertIn("the favorite side covered, on the 2-way no-push basis",
                      src, "caption must describe the observed series")
        self.assertIn("pick the favorite to cover if P(cover) > 50%", src,
                      "caption must describe the win-rate 'V' pick rule")


def _latest_artifact(dd: Path, pattern: str) -> Path:
    """Return the newest file matching pattern in dd."""
    candidates = sorted(dd.glob(pattern))
    if not candidates:
        return None
    return candidates[-1]


def _load_markets(dd: Path):
    """Load the latest markets artifact."""
    import io as _io
    latest = _latest_artifact(dd, "run_engine_markets_*.csv")
    if latest is None:
        return None
    try:
        return pd.read_csv(_io.BytesIO(latest.read_bytes()))
    except Exception:
        return None


class TestMarketsFamilyAwareDateResolution(TestCase):
    """markets.py must resolve its date from ITS OWN run-engine artifact
    families (run_engine_markets_* / run_engine_monitor_*), not from
    available_dates()'s todays_games/calibration/history union — that
    union never enumerates the run-engine families, so a date shipped by
    another family but absent from the run engine blanked the page (the
    drift case). Pins the family-aware pick; the page falls back to the
    shared union only when no run-engine date resolves at all."""

    @staticmethod
    def _page_dates(root):
        """run_engine_page_dates() with the repo root pointed at a temp
        fixture dir, the remote source disabled (local-only) and the sport
        explicit, so no streamlit session state is touched."""
        import utils as u
        with patch.object(u, "REPO_ROOT", root):
            return u.run_engine_page_dates(owner="", repo="",
                                           branch="main", sport="mlb")

    @staticmethod
    def _write(dd, *names):
        dd.mkdir(parents=True, exist_ok=True)
        for n in names:
            (dd / n).write_bytes(b"placeholder")

    def test_drift_case_markets_stop_before_todays_games(self):
        """todays_games_20260904 (and calibration/history for 09-04)
        exist but the run-engine families stop at 09-03 → the page must
        pick 09-03, never blank on the drifted date."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dd = root / "mlb-backend" / "data_delivery"
            self._write(dd, "todays_games_20260904.csv",
                        "calibration_20260904.json",
                        "predictions_history_20260904.csv",
                        "run_engine_markets_20260903.csv",
                        "run_engine_monitor_20260903.json")
            dates = self._page_dates(root)
        self.assertIn("20260903", dates)
        self.assertNotIn("20260904", dates,
                         "run-engine families stop at 09-03 — the drifted "
                         "09-04 (todays_games-only) must NOT be picked")
        self.assertEqual(dates[0], "20260903")

    def test_aligned_families_pick_newest_run_engine_date(self):
        """When every family ships the same day, the newest run-engine
        date wins (the aligned steady-state case)."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dd = root / "mlb-backend" / "data_delivery"
            self._write(dd, "todays_games_20260904.csv",
                        "run_engine_markets_20260903.csv",
                        "run_engine_markets_20260904.csv",
                        "run_engine_monitor_20260904.json")
            dates = self._page_dates(root)
        self.assertEqual(dates[0], "20260904")

    def test_monitor_ships_ahead_of_markets_surfaces_in_union(self):
        """The pick is the union of the page's OWN families: a monitor
        artifact dated after the last markets CSV still surfaces (the
        markets loader then honestly warns for the CSV it cannot find)."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dd = root / "mlb-backend" / "data_delivery"
            self._write(dd, "run_engine_markets_20260903.csv",
                        "run_engine_monitor_20260903.json",
                        "run_engine_monitor_20260904.json")
            dates = self._page_dates(root)
        self.assertEqual(dates[0], "20260904")
        self.assertIn("20260903", dates)

    def test_empty_local_returns_empty_and_page_falls_back(self):
        """No run-engine artifact anywhere (offline / empty local) → the
        family resolver returns [] and markets.py falls back to the shared
        available_dates() union so the documented warning path fires
        instead of a crash."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "mlb-backend" / "data_delivery").mkdir(
                parents=True, exist_ok=True)
            dates = self._page_dates(root)
        self.assertEqual(dates, [])
        src = (_frontend / "markets.py").read_text(encoding="utf-8")
        self.assertIn("utils.run_engine_page_dates(", src,
                      "page must resolve from its own families first")
        self.assertIn("if not dates:", src,
                      "empty family union must fall back, not crash")
        self.assertIn("utils.available_dates(", src,
                      "fallback must use the shared date union")
        self.assertIn("dates[0]", src,
                      "latest-run pin (TestMarketsAlwaysLatestArtifact) "
                      "must keep passing")
        self.assertNotIn("selected_date", src,
                         "page must never read the Today's Games date")

    def test_calibration_walkback_unchanged(self):
        """Calibration keeps its own date resolution + walk-back; the
        family-aware hardening is scoped to the markets page."""
        cal_src = (_frontend / "model_calibration.py").read_text(
            encoding="utf-8")
        self.assertIn("dates[0]", cal_src,
                      "Calibration still defaults to the newest available")
        self.assertNotIn("run_engine_page_dates", cal_src,
                         "Calibration must not adopt the markets resolver")
        utils_src = (_frontend / "utils.py").read_text(encoding="utf-8")
        self.assertIn("def _pick_artifact_date", utils_src,
                      "Calibration's walk-back helper must stay")
        self.assertIn("def load_calibration", utils_src)


if __name__ == "__main__":
    import unittest
    unittest.main()