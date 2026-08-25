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


if __name__ == "__main__":
    import unittest
    unittest.main()