"""Tests for the frontend simplification + Today's Games card enrichment.

Pure-Python (no Streamlit import), following test_frontend_markets.py
conventions:

  1. run_engine_card_bits — fixture slate rows with a full grid produce the
     expected projected runs, p_over_8_5 / p_under_8_5 (sum ≈ 1), and run
     line −1.5/+1.5 probabilities (complement); rows missing grid columns
     degrade to has_grid=False ('n/a' render path) without crashing; no
     slate row → None.
  2. Real-artifact join: todays_games game_id == slate game_pk on the local
     run_engine_markets_<date>.csv.
  3. Page source-inspection smoke: markets.py renders exactly the six
     diagnostics tabs and none of the removed panels; todays_games.py carries
     the run-engine strip; utils.py exposes the shared loader.

Read-only over artifacts — nothing fabricated, no model/metric changes.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
if str(FRONTEND) not in sys.path:
    sys.path.insert(0, str(FRONTEND))

import market_diagnostics as diag  # noqa: E402


def make_slate_row(game_id="20260824_BOS@MIA"):
    return {
        "game_pk": game_id,
        "kind": "slate",
        "home_expected_runs": 4.3911,
        "away_expected_runs": 4.8798,
        "p_over_8_5": 0.5155,
        "p_under_8_5": 0.4845,
        "p_home_cover_1_5": 0.3253,
    }


class TestRunEngineCardBits(unittest.TestCase):
    def test_full_grid_fixture(self):
        row = make_slate_row()
        bits = diag.run_engine_card_bits("20260824_BOS@MIA", {row["game_pk"]: row})
        self.assertIsNotNone(bits)
        self.assertTrue(bits["has_grid"])
        self.assertEqual(bits["proj_home"], 4.3911)
        self.assertEqual(bits["proj_away"], 4.8798)
        # p_over + p_under must sum to 1 (the under side is the exact mirror)
        self.assertAlmostEqual(bits["p_over"] + bits["p_under"], 1.0, places=9)
        self.assertEqual(bits["p_over"], 0.5155)
        self.assertEqual(bits["p_under"], 0.4845)
        # Run line: away +1.5 is the exact complement of home −1.5
        self.assertEqual(bits["p_home_cover"], 0.3253)
        self.assertAlmostEqual(bits["p_away_cover"], 1.0 - 0.3253, places=9)
        self.assertAlmostEqual(bits["p_home_cover"] + bits["p_away_cover"],
                               1.0, places=9)

    def test_missing_grid_columns_quiet_na(self):
        row = {"game_pk": "X", "home_expected_runs": 4.4}
        bits = diag.run_engine_card_bits("X", {"X": row})
        self.assertIsNotNone(bits)              # row exists → strip renders
        self.assertFalse(bits["has_grid"])      # …but as a quiet 'n/a'
        self.assertIsNone(bits["p_over"])
        self.assertIsNone(bits["p_under"])
        self.assertIsNone(bits["p_home_cover"])
        self.assertIsNone(bits["p_away_cover"])

    def test_nan_values_not_fabricated(self):
        row = {"game_pk": "Y", "home_expected_runs": 4.4,
               "away_expected_runs": 4.5, "p_over_8_5": float("nan"),
               "p_under_8_5": 0.5, "p_home_cover_1_5": 0.3}
        bits = diag.run_engine_card_bits("Y", {"Y": row})
        self.assertIsNotNone(bits)
        self.assertFalse(bits["has_grid"])
        self.assertIsNone(bits["p_over"])

    def test_no_slate_row_returns_none(self):
        self.assertIsNone(diag.run_engine_card_bits("20260824_NOPE",
                                                    {"Z": make_slate_row()}))
        self.assertIsNone(diag.run_engine_card_bits("20260824_BOS@MIA", {}))
        self.assertIsNone(diag.run_engine_card_bits("20260824_BOS@MIA", None))

    def test_accepts_pandas_series_row(self):
        row = make_slate_row()
        series = pd.Series(row)
        bits = diag.run_engine_card_bits("20260824_BOS@MIA",
                                         {row["game_pk"]: series})
        self.assertIsNotNone(bits)
        self.assertTrue(bits["has_grid"])
        self.assertEqual(bits["proj_home"], 4.3911)
        self.assertAlmostEqual(bits["p_over"] + bits["p_under"], 1.0, places=9)

    def test_non_numeric_values_do_not_crash(self):
        row = {"game_pk": "Q", "home_expected_runs": "4.6",
               "away_expected_runs": None, "p_over_8_5": "0.5",
               "p_under_8_5": 0.5, "p_home_cover_1_5": "boom"}
        bits = diag.run_engine_card_bits("Q", {"Q": row})
        self.assertIsNotNone(bits)
        self.assertEqual(bits["proj_home"], 4.6)   # numeric string coerced
        self.assertIsNone(bits["proj_away"])
        self.assertIsNone(bits["p_home_cover"])
        self.assertFalse(bits["has_grid"])


class TestRealArtifactJoin(unittest.TestCase):
    """Local read-only proof: every todays_games game_id matches a slate
    game_pk on the shipped run_engine_markets artifact."""

    def test_all_todays_games_join_to_slate(self):
        dd = Path(__file__).resolve().parents[1] / "data_delivery"
        m_path = dd / "run_engine_markets_20260824.csv"
        t_path = dd / "todays_games_20260824.csv"
        if not m_path.exists() or not t_path.exists():
            self.skipTest("local run-engine artifacts absent in this workspace")
        markets = pd.read_csv(m_path)
        todays = pd.read_csv(t_path)
        self.assertIn("kind", markets.columns)
        sl = markets[markets["kind"] == "slate"]
        slate_map = {str(pk): rec
                     for pk, rec in zip(sl["game_pk"], sl.to_dict("records"))}
        joined = 0
        for gid in todays["game_id"].astype(str):
            bits = diag.run_engine_card_bits(gid, slate_map)
            self.assertIsNotNone(bits, f"no slate row for {gid}")
            self.assertTrue(bits["has_grid"], f"grid missing for {gid}")
            self.assertAlmostEqual(bits["p_over"] + bits["p_under"], 1.0,
                                   places=6, msg=f"O/U not complementary: {gid}")
            self.assertAlmostEqual(bits["p_home_cover"] + bits["p_away_cover"],
                                   1.0, places=6, msg=f"RL not complementary: {gid}")
            joined += 1
        self.assertEqual(joined, len(todays),
                         "every Today's Games card must join to the slate")


class TestMarketsPageStripped(unittest.TestCase):
    """Render/smoke via source inspection: the six charts render, the removed
    panels don't."""

    def setUp(self):
        self.src = (FRONTEND / "markets.py").read_text()

    def test_six_diagnostics_tabs_present(self):
        for label in ['"Distribution"', '"Relativized"', '"Pooled lines"',
                      '"Money line 8.5"', '"Overs picks"', '"Run-line picks"']:
            self.assertIn(label, self.src, f"missing tab {label}")
        self.assertIn("import market_diagnostics as diag", self.src)
        self.assertIn("No decided OOF rows", self.src)

    def test_removed_panels_absent(self):
        for marker in [
            "st.slider(", "st.select_slider(",
            "Market calibration & holdout gate",
            "Rolling totals Brier",
            "Blowout-tail fit check",
            "CONFLICT", "suppressed_game_pks",
            "P(OVER ", "P(COVER ", "DERIVED ML",
            "Holdout", "agreement_delta",
        ]:
            self.assertNotIn(marker, self.src,
                             f"removed panel marker still rendered: {marker}")

    def test_artifact_warnings_kept(self):
        self.assertIn("No run-engine markets artifact", self.src)
        self.assertIn("_load_markets", self.src)


class TestTodaysGamesCardEnrichment(unittest.TestCase):
    def setUp(self):
        self.src = (FRONTEND / "todays_games.py").read_text()

    def test_run_engine_strip_present(self):
        self.assertIn("RUN ENGINE", self.src)
        self.assertIn("run_engine_card_bits", self.src)
        self.assertIn("load_run_engine_markets", self.src)
        self.assertIn("n/a", self.src)
        self.assertIn("complement", self.src)
        # Card still renders moneyline + pitchers (untouched)
        self.assertIn("fb-odds", self.src)
        self.assertIn("fb-pitchers", self.src)

    def test_utils_exposes_shared_loader(self):
        utils_src = (FRONTEND / "utils.py").read_text()
        self.assertIn("def load_run_engine_markets", utils_src)
        # CSS for the strip exists
        self.assertIn(".fb-runengine", utils_src)


if __name__ == "__main__":
    unittest.main()
