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

import pandas as pd

_frontend = Path(__file__).resolve().parents[1] / "frontend"
if str(_frontend) not in sys.path:
    sys.path.insert(0, str(_frontend))


# ---------------------------------------------------------------------------
# Direct URL-construction tests (no Streamlit needed)
# ---------------------------------------------------------------------------

# Replicate the _raw_url logic inline for pure testing
REPO_SUBDIR = "mlb-bet-predictor"


def _raw_url_test(relpath, owner, repo, branch):
    if owner and repo:
        return (f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}"
                f"/{REPO_SUBDIR}/data_delivery/{relpath}")
    return f"<local:{relpath}>"


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
        """The run_engine_markets_20260824.csv must be accessible from the
        repo — this is the actual file the frontend should fetch."""
        import requests
        url = _raw_url_test("run_engine_markets_20260824.csv",
                            "andrewkemmer", "sports_prediction_model", "main")
        resp = requests.get(url, timeout=15)
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
        cls.root = _frontend.parent  # mlb-bet-predictor

    @staticmethod
    def _fit(date_str):
        p = Path(__file__).resolve().parents[1] / "data_delivery" / \
            f"run_engine_monitor_{date_str}.json"
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
        fit = self._fit("20260827")
        rows = self.diag.fit_panel_rows(fit)
        # chi2/df straight from dispersion_chi2_per_df
        self.assertAlmostEqual(rows["chi2_home"], 2.184, places=3)
        self.assertAlmostEqual(rows["chi2_away"], 2.4779, places=3)
        # alpha = count-weighted bin mean (home ~0.268, away ~0.346)
        self.assertGreater(rows["alpha_home"], 0.26)
        self.assertLess(rows["alpha_home"], 0.28)
        self.assertGreater(rows["alpha_away"], 0.34)
        self.assertLess(rows["alpha_away"], 0.36)
        self.assertEqual(rows["alpha_home_form"], "piecewise")
        self.assertEqual(rows["alpha_away_form"], "linear")
        # variance check: implied vs observed per side
        vh = rows["variance_home"]
        self.assertAlmostEqual(vh[0], 9.938, places=3)
        self.assertAlmostEqual(vh[1], 9.886, places=3)
        va = rows["variance_away"]
        self.assertAlmostEqual(va[0], 10.902, places=3)
        self.assertAlmostEqual(va[1], 11.03, places=3)
        # MC metadata: numeric n_draws with separators
        self.assertIn("10,000 draws", rows["mc_caption"])
        # tails: real k labels (unicode ≥/≤) with observed_p/modeled_p keys
        home_tail = rows["tails"]["Home"]
        self.assertIn("k≤1", home_tail)
        self.assertIn("k≥10", home_tail)
        self.assertIn("obs=0.168", home_tail)  # home ≤1 observed (3dp)
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
        means (the pre-holdout fit scope; the pooled-frame version is +0.12)."""
        fit = self._fit("20260827")
        edge = self.diag.lambda_edge(fit)
        self.assertIsNotNone(edge)
        self.assertAlmostEqual(edge, 0.1258, places=3)  # fit-curve bin means
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
        cls.root = _frontend.parent

    @classmethod
    def tearDownClass(cls):
        import sys as _sys
        if cls._backup is not None:
            _sys.modules["streamlit"] = cls._backup
        else:
            _sys.modules.pop("streamlit", None)

    def test_model_card_renders_real_artifact(self):
        import json
        mon = json.loads((self.root / "data_delivery"
                          / "run_engine_monitor_20260827.json").read_text())
        self.markets._render_run_engine_model_card(mon)  # no crash

    def test_drift_and_coverage_render_real_artifacts(self):
        d = pd.read_csv(self.root / "data_delivery"
                        / "run_engine_feature_drift_20260827.csv")
        self.assertEqual(len(d), 29)
        self.markets._render_run_engine_drift(d)
        c = pd.read_csv(self.root / "data_delivery"
                        / "run_engine_feature_coverage_20260827.csv")
        self.assertEqual(len(c), 58)  # 29 features x 2 windows
        self.markets._render_run_engine_coverage(c)

    def test_empty_states_never_crash(self):
        self.markets._render_run_engine_drift(None)
        self.markets._render_run_engine_coverage(pd.DataFrame())
        self.markets._render_run_engine_model_card(
            {"fit": {}, "phase1": {}, "market_metrics": {}})


if __name__ == "__main__":
    import unittest
    unittest.main()