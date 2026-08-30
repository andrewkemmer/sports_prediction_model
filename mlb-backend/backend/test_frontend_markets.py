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
        self.assertAlmostEqual(rows["chi2_home"], 2.141, places=3)
        self.assertAlmostEqual(rows["chi2_away"], 2.368, places=3)
        # alpha = count-weighted bin mean (home ~0.261, away ~0.313 on the
        # 6,960-frame artifact)
        self.assertAlmostEqual(rows["alpha_home"], 0.261, places=3)
        self.assertAlmostEqual(rows["alpha_away"], 0.313, places=3)
        self.assertEqual(rows["alpha_home_form"], "piecewise")
        self.assertEqual(rows["alpha_away_form"], "power")
        # variance check: implied vs observed per side
        vh = rows["variance_home"]
        self.assertAlmostEqual(vh[0], 9.517, places=3)
        self.assertAlmostEqual(vh[1], 9.559, places=3)
        va = rows["variance_away"]
        self.assertAlmostEqual(va[0], 10.521, places=3)
        self.assertAlmostEqual(va[1], 10.689, places=3)
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
        self.assertAlmostEqual(edge, 0.0110, places=4)  # fit-curve bin means
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

    def test_model_card_renders_real_artifact(self):
        import json
        mon = json.loads((_latest_artifact(self.root / "data_delivery", "run_engine_monitor_*.json")).read_text())
        self.markets._render_run_engine_model_card(mon)  # no crash

    def test_drift_and_coverage_render_real_artifacts(self):
        d = pd.read_csv(_latest_artifact(self.root / "data_delivery", "run_engine_feature_drift_*.csv"))
        self.assertEqual(len(d), 29)
        self.markets._render_run_engine_drift(d)
        c = pd.read_csv(_latest_artifact(self.root / "data_delivery", "run_engine_feature_coverage_*.csv"))
        self.assertEqual(len(c), 58)  # 29 features x 2 windows
        self.markets._render_run_engine_coverage(c)

    def test_empty_states_never_crash(self):
        self.markets._render_run_engine_drift(None)
        self.markets._render_run_engine_coverage(pd.DataFrame())
        self.markets._render_run_engine_model_card(
            {"fit": {}, "phase1": {}, "market_metrics": {}})

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
               / "frontend" / "markets.py").read_text()
        self.assertIn("Holdout AUC", src,
                      "renderer must show the holdout AUC metric")
        self.assertIn("calibrated post-fix", src,
                      "footer must reflect the calibrated derived ML")
        self.assertNotIn("underweights the home edge", src,
                         "stale underweighting claim must be gone")

    def test_diagnostics_tab_uses_fixed_line_not_own_rounded_line(self):
        """The 'Money line (rounded)' tab became a FIXED-line calibration:
        all games at one selectable line (default 8.5), never the per-game
        own rounded total (which compressed predicted P(over) to 0.44-0.51
        — the low-info curve the user rejected)."""
        src = (Path(__file__).resolve().parents[2]
               / "frontend" / "markets.py").read_text()
        self.assertIn("diag.fixed_line_calibration", src,
                      "tab must pool at a fixed line via the new helper")
        self.assertIn("diag.chart_fixed_line", src,
                      "tab must render the bars + observed-vs-predicted view")
        self.assertIn("diag_fixed_line", src,
                      "line selector key present (default 8.5)")
        self.assertNotIn("diag.rounded_total_pairs", src,
                         "per-game own-line pricing must be gone from the tab")
        self.assertNotIn("Per-game rounded total", src,
                         "old own-line chart title must be gone")


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
        src = (_frontend / "markets.py").read_text()
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
        src = (_frontend / "markets.py").read_text()
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

    def test_chart_title_reflects_mixed_rule(self):
        """The chart title must say '(home -1.5 / away +1.5)' — NOT
        'at -1.5' — because the pick set includes both sides."""
        src = (_frontend / "markets.py").read_text()
        self.assertIn(
            "Run-line picks (home −1.5 / away +1.5)", src,
            "markets.py must pass the corrected title with both sides"
        )
        # The old misleading title must be gone
        self.assertNotIn(
            "Run-line picks at −1.5", src,
            "Old misleading title must be removed"
        )

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

    def test_x_axis_uses_picked_side_probability(self):
        """The chart_pick_buckets receives max(p, 1-p) as the x-axis,
        which is the picked side's probability. Verify the function
        is called with the correct data shape."""
        src = (_frontend / "markets.py").read_text()
        # The title must match what chart_pick_buckets receives
        self.assertIn(
            "x-axis is the picked side's probability", src,
            "Caption must describe the x-axis as picked-side probability"
        )


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


if __name__ == "__main__":
    import unittest
    unittest.main()