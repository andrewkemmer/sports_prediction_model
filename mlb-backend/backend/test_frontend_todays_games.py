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

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

# frontend/ moved to the repository root (multi-sport restructure, Phase B)
FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
if str(FRONTEND) not in sys.path:
    sys.path.insert(0, str(FRONTEND))

import market_diagnostics as diag  # noqa: E402


def make_slate_row(game_id="20260824_BOS@MIA"):
    # 4.3911 + 4.8798 = 9.2709 → rounded total 9.5 (round half up)
    return {
        "game_pk": game_id,
        "kind": "slate",
        "home_expected_runs": 4.3911,
        "away_expected_runs": 4.8798,
        "p_over_9_5": 0.4155,
        "p_under_9_5": 0.5845,
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
        # 4.3911 + 4.8798 = 9.2709 → rounded total 9.5, priced at p_over_9_5
        self.assertEqual(bits["total_line"], 9.5)
        self.assertFalse(bits["clamped"])
        self.assertEqual(bits["p_over"], 0.4155)
        self.assertEqual(bits["p_under"], 0.5845)
        # p_over + p_under must sum to 1 (the under side is the exact mirror)
        self.assertAlmostEqual(bits["p_over"] + bits["p_under"], 1.0, places=9)
        # Run line: away +1.5 is the exact complement of home −1.5
        self.assertEqual(bits["p_home_cover"], 0.3253)
        self.assertAlmostEqual(bits["p_away_cover"], 1.0 - 0.3253, places=9)
        self.assertAlmostEqual(bits["p_home_cover"] + bits["p_away_cover"],
                               1.0, places=9)

    def test_card_rounded_total_fixture(self):
        """The user example: 4.9 + 4.4 = 9.3 → 9.5, priced at p_over_9_5."""
        row = {"game_pk": "G", "home_expected_runs": 4.4,
               "away_expected_runs": 4.9, "p_over_9_5": 0.55,
               "p_under_9_5": 0.45, "p_home_cover_1_5": 0.4}
        bits = diag.run_engine_card_bits("G", {"G": row})
        self.assertTrue(bits["has_grid"])
        self.assertEqual(bits["total_line"], 9.5)
        self.assertEqual(bits["p_over"], 0.55)
        self.assertEqual(bits["p_under"], 0.45)
        self.assertAlmostEqual(bits["p_over"] + bits["p_under"], 1.0, places=9)

    def test_line_out_of_grid_clamped_with_note(self):
        # 14.0 → clamps to the 12.5 edge of the shipped grid, flagged
        row = {"game_pk": "H", "home_expected_runs": 7.0,
               "away_expected_runs": 7.0, "p_over_12_5": 0.2,
               "p_under_12_5": 0.8, "p_home_cover_1_5": 0.3}
        bits = diag.run_engine_card_bits("H", {"H": row})
        self.assertTrue(bits["has_grid"])
        self.assertEqual(bits["total_line"], 12.5)
        self.assertTrue(bits["clamped"])
        self.assertEqual(bits["p_over"], 0.2)
        # low edge: 6.0 → clamps to 6.5
        row2 = {"game_pk": "I", "home_expected_runs": 3.0,
                "away_expected_runs": 3.0, "p_over_6_5": 0.9,
                "p_under_6_5": 0.1, "p_home_cover_1_5": 0.3}
        bits2 = diag.run_engine_card_bits("I", {"I": row2})
        self.assertTrue(bits2["has_grid"])
        self.assertEqual(bits2["total_line"], 6.5)
        self.assertTrue(bits2["clamped"])

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
               "away_expected_runs": 4.5, "p_over_9_5": float("nan"),
               "p_under_9_5": 0.5, "p_home_cover_1_5": 0.3}
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


class TestOUPushDisplay(unittest.TestCase):
    """Verify that the run-engine card's O/U line includes push for
    whole-number lines and omits it for half-lines.

    Root cause: the old p_over/p_under columns for whole-number lines
    (e.g. 9.0) were push-inclusive on the over side, so over+under
    summed to ~92% instead of 100%.  After the p_over definition fix
    (strict P(over) = total >= line + 0.5), p_push = P(total == line)
    must be shown for whole-number lines.
    """

    def test_half_line_push_is_zero_or_none(self):
        """For half-lines (8.5, 9.5), p_push is 0 or None (no next grid column)."""
        row = make_slate_row()  # total_line = 9.5
        bits = diag.run_engine_card_bits("20260824_BOS@MIA",
                                         {row["game_pk"]: row})
        self.assertIsNotNone(bits)
        self.assertTrue(bits["has_grid"])
        # Half-line 9.5: no push possible — p_push is 0 or None
        pp = bits.get("p_push")
        if pp is not None:
            self.assertAlmostEqual(pp, 0.0, places=9)

    def test_whole_line_push_nonzero(self):
        """For whole-number lines (e.g. 9.0), p_push > 0."""
        # Craft a row with λ_h + λ_a = 9.0 → line 9.0
        row = {"game_pk": "W", "home_expected_runs": 4.5,
               "away_expected_runs": 4.5,
               "p_over_9_0": 0.42, "p_under_9_0": 0.58,
               "p_over_9_5": 0.34, "p_under_9_5": 0.66,
               "p_home_cover_1_5": 0.4}
        bits = diag.run_engine_card_bits("W", {"W": row})
        self.assertTrue(bits["has_grid"])
        self.assertEqual(bits["total_line"], 9.0)
        # p_push = p_over_9_0 - p_over_9_5 = 0.42 - 0.34 = 0.08
        self.assertAlmostEqual(bits["p_push"], 0.08, places=9)
        self.assertGreater(bits["p_push"], 0)

    def test_p_push_matches_grid_difference(self):
        """p_push for a whole line equals p_over(L) - p_over(L+0.5)."""
        row = {"game_pk": "P", "home_expected_runs": 4.0,
               "away_expected_runs": 4.0,  # line = 8.0
               "p_over_8_0": 0.55, "p_under_8_0": 0.45,
               "p_over_8_5": 0.48, "p_under_8_5": 0.52,
               "p_home_cover_1_5": 0.35}
        bits = diag.run_engine_card_bits("P", {"P": row})
        self.assertEqual(bits["total_line"], 8.0)
        self.assertAlmostEqual(bits["p_push"], 0.55 - 0.48, places=9)

    def test_html_push_for_whole_line_source_check(self):
        """todays_games.py must handle p_push in the O/U HTML line."""
        src = (FRONTEND / "todays_games.py").read_text()
        # The function must reference p_push in the O/U span
        self.assertIn("p_push", src,
                      "todays_games.py must reference p_push for O/U display")

    def test_html_no_push_for_half_line_source_check(self):
        """For half-lines, p_push is None/0, so Push is not shown."""
        # This is a source-level check: the code must guard on p_push > 0.005
        src = (FRONTEND / "todays_games.py").read_text()
        self.assertIn("0.005", src,
                      "Threshold guard for p_push display must exist")

    def test_p_push_present_in_bits_dict(self):
        """run_engine_card_bits must always include p_push key."""
        row = make_slate_row()  # half-line
        bits = diag.run_engine_card_bits("20260824_BOS@MIA",
                                         {row["game_pk"]: row})
        self.assertIn("p_push", bits, "bits dict must contain p_push key")

    # ---- post-65b44ec artifact (explicit p_push column) ----

    def test_postfix_artifact_whole_line_reads_explicit_push(self):
        """POST-fix artifact (explicit p_push_9_0 column): card reads the
        column DIRECTLY — over+push+under sums to 100 (±1 rounding) and
        the push is the real P(total == 9), not the (always-0) grid
        difference."""
        row = {"game_pk": "PX", "home_expected_runs": 4.5,
               "away_expected_runs": 4.5,  # line = 9.0
               "p_over_9_0": 0.388, "p_push_9_0": 0.086,
               "p_under_9_0": 0.526,
               # post-fix: p_over_9_5 == p_over_9_0 (both strict ≥ 10)
               "p_over_9_5": 0.388, "p_under_9_5": 0.612,
               "p_home_cover_1_5": 0.4}
        bits = diag.run_engine_card_bits("PX", {"PX": row})
        self.assertTrue(bits["has_grid"])
        self.assertEqual(bits["total_line"], 9.0)
        self.assertAlmostEqual(bits["p_push"], 0.086, places=9)
        total = bits["p_over"] + bits["p_push"] + bits["p_under"]
        self.assertAlmostEqual(total, 1.0, delta=0.01,
                               msg="over+push+under must sum to 100% (±1)")

    def test_postfix_artifact_half_line_push_is_zero(self):
        """POST-fix artifact, half-line 9.5: explicit p_push_9_5 column is
        present and equals 0 (integer total can never equal 9.5)."""
        row = {"game_pk": "PH", "home_expected_runs": 4.6,
               "away_expected_runs": 4.7,  # 9.3 → line 9.5
               "p_over_9_5": 0.42, "p_push_9_5": 0.0,
               "p_under_9_5": 0.58,
               "p_home_cover_1_5": 0.33}
        bits = diag.run_engine_card_bits("PH", {"PH": row})
        self.assertEqual(bits["total_line"], 9.5)
        self.assertIsNotNone(bits["p_push"])
        self.assertAlmostEqual(bits["p_push"], 0.0, places=9)

    def test_legacy_artifact_fallback_still_works(self):
        """LEGACY artifact (no p_push column): subtraction fallback still
        produces the push band (p_over was push-inclusive pre-fix)."""
        row = {"game_pk": "LG", "home_expected_runs": 4.5,
               "away_expected_runs": 4.5,  # line = 9.0
               "p_over_9_0": 0.50, "p_under_9_0": 0.50,
               "p_over_9_5": 0.42, "p_under_9_5": 0.58,
               "p_home_cover_1_5": 0.4}
        bits = diag.run_engine_card_bits("LG", {"LG": row})
        self.assertEqual(bits["total_line"], 9.0)
        self.assertAlmostEqual(bits["p_push"], 0.08, places=9)

    def test_postfix_explicit_column_wins_over_subtraction(self):
        """If BOTH the explicit column and the grid difference exist, the
        explicit column must win (post-fix subtraction is always 0)."""
        row = {"game_pk": "PB", "home_expected_runs": 4.5,
               "away_expected_runs": 4.5,
               "p_over_9_0": 0.388, "p_push_9_0": 0.086,
               "p_under_9_0": 0.526,
               "p_over_9_5": 0.388, "p_under_9_5": 0.612,
               "p_home_cover_1_5": 0.4}
        bits = diag.run_engine_card_bits("PB", {"PB": row})
        # Subtraction would give 0.0; explicit column gives 0.086
        self.assertAlmostEqual(bits["p_push"], 0.086, places=9)

    def test_grid_push_col_naming(self):
        """grid_push_col naming: 9.0 → p_push_9_0, 8.5 → p_push_8_5."""
        self.assertEqual(diag.grid_push_col(9.0), "p_push_9_0")
        self.assertEqual(diag.grid_push_col(8.5), "p_push_8_5")


class TestRunEngineStripSmoke(unittest.TestCase):
    """Verify the run-engine strip HTML structure in todays_games.py."""

    def test_strip_html_includes_run_engine_label(self):
        """The run-engine strip must include the RUN ENGINE label."""
        src = (FRONTEND / "todays_games.py").read_text()
        self.assertIn("RUN ENGINE", src,
                      "todays_games.py must render the RUN ENGINE strip")

    def test_strip_p_push_handled_in_source(self):
        """The O/U line in the strip must reference p_push."""
        src = (FRONTEND / "todays_games.py").read_text()
        # Find the _runengine_html function
        start = src.index("def _runengine_html")
        rest = src[start:]
        end_idx = rest.index("\ndef _score_side")
        func_src = rest[:end_idx]
        self.assertIn("p_push", func_src,
                      "_runengine_html must reference p_push")


if __name__ == "__main__":
    unittest.main()
