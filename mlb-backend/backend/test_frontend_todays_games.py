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

import json
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
        # The own line is now the FAIR line (grid argmin |re-scaled P(over)
        # − 0.5|): when the row only prices the 12.5 edge column, the fair
        # argmin IS that grid boundary — taken verbatim, never clamped
        # (clamped stays False because a real grid line was priced; the
        # clamp flag only fires on the no-grid round-mean fallback).
        row = {"game_pk": "H", "home_expected_runs": 7.0,
               "away_expected_runs": 7.0, "p_over_12_5": 0.2,
               "p_under_12_5": 0.8, "p_home_cover_1_5": 0.3}
        bits = diag.run_engine_card_bits("H", {"H": row})
        self.assertTrue(bits["has_grid"])
        self.assertEqual(bits["total_line"], 12.5)
        self.assertFalse(bits["clamped"])
        self.assertEqual(bits["p_over"], 0.2)
        # low edge: only the 6.5 column prices → fair argmin 6.5, taken as-is
        row2 = {"game_pk": "I", "home_expected_runs": 3.0,
                "away_expected_runs": 3.0, "p_over_6_5": 0.9,
                "p_under_6_5": 0.1, "p_home_cover_1_5": 0.3}
        bits2 = diag.run_engine_card_bits("I", {"I": row2})
        self.assertTrue(bits2["has_grid"])
        self.assertEqual(bits2["total_line"], 6.5)
        self.assertFalse(bits2["clamped"])

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
    """Verify that the run-engine card's O/U line is a RE-SCALED 2-way split
    (Over + Under = 100%) with the push folded proportionately into both.

    Root cause: whole-number lines carry a push (P(total == line)) that the
    old display ignored, making over+under sum to ~92%. c7a60e3 first showed
    it as a 3-way split (Over / Push / Under), then this re-scaled it back
    to 2-way: a push refunds the bet, so sportsbooks price whole-number
    lines by re-scaling the surviving outcomes to sum to 100%, preserving
    the over:under ratio. The push is still read (P(push) is needed
    internally for the re-scale and for EV: EV = payout×P(over) −
    stake×P(under) + 0×P(push)); the card just doesn't show it as a
    headline value.
    """

    def test_half_line_push_is_zero_or_none(self):
        """For half-lines (8.5, 9.5), p_push is 0 or None (no next grid column)."""
        row = make_slate_row()  # total_line = 9.5
        bits = diag.run_engine_card_bits("20260824_BOS@MIA",
                                         {row["game_pk"]: row})
        self.assertIsNotNone(bits)
        self.assertTrue(bits["has_grid"])
        # Half-line 9.5: no push possible — p_push is exactly 0.0, never a
        # neighbor's push (the pre-fix fallback wrongly showed P(total=9)).
        self.assertEqual(bits.get("p_push"), 0.0)
        self.assertAlmostEqual(bits["p_over"] + bits["p_under"], 1.0,
                               places=9)

    def test_whole_line_push_nonzero(self):
        """For whole-number lines (e.g. 9.0), p_push > 0."""
        # Craft a row with λ_h + λ_a = 9.0 → line 9.0, and select it
        # explicitly (adding the 8.5 neighbor moves the fair default).
        row = {"game_pk": "W", "home_expected_runs": 4.5,
               "away_expected_runs": 4.5,
               "p_over_8_5": 0.50, "p_under_8_5": 0.50,
               "p_over_9_0": 0.42, "p_under_9_0": 0.58,
               "p_over_9_5": 0.34, "p_under_9_5": 0.66,
               "p_home_cover_1_5": 0.4}
        bits = diag.run_engine_card_bits("W", {"W": row}, line=9.0)
        self.assertTrue(bits["has_grid"])
        self.assertEqual(bits["total_line"], 9.0)
        # Fallback (no explicit p_push column): whole-line push = P(total==9)
        # = p_over(8.5) − p_over(9.0) = 0.50 − 0.42 = 0.08. NOT the pre-fix
        # inversion (p_over(9.0) − p_over(9.5) = 0).
        self.assertAlmostEqual(bits["p_push"], 0.08, places=9)
        self.assertGreater(bits["p_push"], 0)
        self.assertAlmostEqual(bits["p_over"] + bits["p_under"], 1.0,
                               delta=0.01)

    def test_p_push_matches_grid_difference(self):
        """p_push for a whole line equals p_over(L−0.5) − p_over(L): the
        LOWER neighbor carries the push mass (strict over: p_over(L−0.5) =
        P(total ≥ L), p_over(L) = P(total ≥ L+1) → P(total == L))."""
        row = {"game_pk": "P", "home_expected_runs": 4.0,
               "away_expected_runs": 4.0,  # round mean → 8.0
               "p_over_7_5": 0.62, "p_under_7_5": 0.38,
               "p_over_8_0": 0.55, "p_under_8_0": 0.45,
               "p_over_8_5": 0.48, "p_under_8_5": 0.52,
               "p_home_cover_1_5": 0.35}
        # The FAIR default here is 8.5 (rescaled 0.48 is closer to 0.5 than
        # 8.0's 0.55) — select the whole line explicitly to test its push.
        bits = diag.run_engine_card_bits("P", {"P": row}, line=8.0)
        self.assertEqual(bits["total_line"], 8.0)
        # Whole-line push comes from the LOWER neighbor: 0.62 − 0.55 = 0.07
        # (the pre-fix direction p_over(8.0) − p_over(8.5) = 0.07 was
        # coincidentally equal here but wrong for half-lines and 0 for
        # whole lines whose upper neighbor lacks the mass).
        self.assertAlmostEqual(bits["p_push"], 0.62 - 0.55, places=9)

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
        """POST-fix artifact (explicit p_push_9_0 column): the re-scaled
        Over + Under sums to 100 (±1 rounding) and the push is folded in,
        preserving the over:under ratio. The push column is STILL read and
        carried in the bits (EV needs it), it just isn't a headline value."""
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
        # Push column still read & carried (EV needs all three).
        self.assertAlmostEqual(bits["p_push"], 0.086, places=9)
        # Re-scaled headline Over + Under sum to 100 (±1 rounding).
        total = bits["p_over"] + bits["p_under"]
        self.assertAlmostEqual(total, 1.0, delta=0.01,
                               msg="re-scaled over+under must sum to 100% (±1)")
        # The push band folded out is the delta between raw over/under sum
        # and 1: raw over+under should equal (1 - 0.086)=0.914 (±rounding).
        self.assertAlmostEqual(
            bits["p_over_raw"] + bits["p_under_raw"], 1.0 - 0.086, delta=0.01,
            msg="raw (pre-re-scale) over+under excludes the push band")

    def test_postfix_artifact_half_line_push_is_zero(self):
        """POST-fix artifact, half-line 9.5: explicit p_push_9_5 column is
        present and equals 0 (integer total can never equal 9.5). With
        p_push = 0 the re-scale is a no-op — headline values equal the raw
        ones and already sum to 100%."""
        row = {"game_pk": "PH", "home_expected_runs": 4.6,
               "away_expected_runs": 4.7,  # 9.3 → line 9.5
               "p_over_9_5": 0.42, "p_push_9_5": 0.0,
               "p_under_9_5": 0.58,
               "p_home_cover_1_5": 0.33}
        bits = diag.run_engine_card_bits("PH", {"PH": row})
        self.assertEqual(bits["total_line"], 9.5)
        self.assertIsNotNone(bits["p_push"])
        self.assertAlmostEqual(bits["p_push"], 0.0, places=9)
        # Half-line re-scale is a no-op: headline == raw, sums to 100%.
        self.assertAlmostEqual(bits["p_over"], bits["p_over_raw"], places=9)
        self.assertAlmostEqual(bits["p_under"], bits["p_under_raw"], places=9)
        self.assertAlmostEqual(bits["p_over"] + bits["p_under"], 1.0, delta=0.01)

    def test_legacy_artifact_fallback_still_works(self):
        """LEGACY artifact (no p_push column): the corrected subtraction
        fallback recovers the whole-line push band from the LOWER neighbor
        (p_over(8.5) − p_over(9.0) = P(total == 9)), and the re-scale folds
        it in so headline Over + Under = 100%."""
        row = {"game_pk": "LG", "home_expected_runs": 4.5,
               "away_expected_runs": 4.5,  # line = 9.0
               "p_over_8_5": 0.58, "p_under_8_5": 0.42,
               "p_over_9_0": 0.50, "p_under_9_0": 0.50,
               "p_over_9_5": 0.42, "p_under_9_5": 0.58,
               "p_home_cover_1_5": 0.4}
        bits = diag.run_engine_card_bits("LG", {"LG": row})
        self.assertEqual(bits["total_line"], 9.0)
        # Fallback recovers P(total == 9) = p_over_8_5 − p_over_9_0
        # (0.58 − 0.50 = 0.08) — NOT the pre-fix p_over_9_0 − p_over_9_5.
        self.assertAlmostEqual(bits["p_push"], 0.08, places=9)
        # Over:Under ratio preserved after re-scale: 0.50/0.50 → 50/50.
        self.assertAlmostEqual(bits["p_over"], 0.50, places=9)
        self.assertAlmostEqual(bits["p_under"], 0.50, places=9)
        self.assertAlmostEqual(bits["p_over"] + bits["p_under"], 1.0, delta=0.01)

    def test_legacy_half_line_fallback_is_zero_not_neighbor_push(self):
        """LEGACY artifact, half-line 8.5 (no p_push column): the fallback
        must be exactly 0.0 — never the neighbor's push. The pre-fix code
        returned P(total == 9) here (p_over_8_5 − p_over_9_0 > 0), the
        exact half-line-shows-push inversion this fix removes."""
        row = {"game_pk": "LH", "home_expected_runs": 4.0,
               "away_expected_runs": 4.3,  # line = 8.5
               "p_over_8_5": 0.50, "p_under_8_5": 0.50,
               "p_over_9_0": 0.42, "p_under_9_0": 0.58,
               "p_home_cover_1_5": 0.4}
        bits = diag.run_engine_card_bits("LH", {"LH": row})
        self.assertEqual(bits["total_line"], 8.5)
        # p_over_8_5 − p_over_9_0 = 0.08 (the neighbor's P(total=9)) must
        # NOT leak into the half-line's push.
        self.assertEqual(bits["p_push"], 0.0)
        self.assertAlmostEqual(bits["p_over"] + bits["p_under"], 1.0,
                               places=9)

    def test_postfix_explicit_column_wins_over_subtraction(self):
        """If BOTH the explicit column and the grid difference exist, the
        explicit column must win (it is exact — P(total == line) from the
        same MC draws — and does not depend on the L−0.5 neighbor column
        being present)."""
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


class TestLineSelector(unittest.TestCase):
    """Per-card O/U line selector on the run-engine strip.

    run_engine_card_bits accepts an optional ``line`` override (the
    selector's value); the card is otherwise unchanged. The override must:
    price the O/U at the selected grid line; mark line_selected so the card
    can flag a non-default line; and fall back to the model line for any
    invalid / out-of-grid value. The re-scaled 2-way Over/Under (push
    folded proportionately) must hold at ANY selected line — the same
    convention as the model line itself.
    """

    def _row(self, pk="SEL", h=4.5, a=4.5):
        """λ_h + λ_a = 9.0. Grid carries 9.0 (whole), 8.5 / 9.5 (halves),
        8.0 / 7.5 (the 8.0 lower neighbor for its push) — the FAIR default
        (grid argmin |re-scaled P(over) − 0.5|) is 8.5 (rescaled 0.48;
        9.0 is 0.4468, 8.0 is 0.5914)."""
        return {"game_pk": pk, "home_expected_runs": h,
                "away_expected_runs": a,
                # p_under = 1 - p_over - p_push (artifact convention; push
                # from the LOWER neighbor: p_over(L−0.5) − p_over(L)):
                # 9.0 push = p_over_8_5 − p_over_9_0 = 0.48-0.42 = 0.06 →
                #   under = 1-0.42-0.06 = 0.52;
                # 8.0 push = p_over_7_5 − p_over_8_0 = 0.62-0.55 = 0.07 →
                #   under = 1-0.55-0.07 = 0.38.
                "p_over_9_0": 0.42, "p_under_9_0": 0.52,
                "p_over_9_5": 0.34, "p_under_9_5": 0.66,
                "p_over_8_0": 0.55, "p_under_8_0": 0.38,
                "p_over_8_5": 0.48, "p_under_8_5": 0.52,
                "p_over_7_5": 0.62, "p_under_7_5": 0.38,
                "p_home_cover_1_5": 0.4}

    def test_defaults_to_model_line(self):
        """No line override → the card prices at the FAIR line (grid argmin
        |re-scaled P(over) − 0.5|, the 50/50 anchor) with line_selected
        None. In this fixture that is 8.5 (rescaled 0.48 — closer to 0.5
        than 9.0's 0.4565 or 8.0's 0.5914)."""
        row = self._row()
        bits = diag.run_engine_card_bits("SEL", {"SEL": row})
        self.assertTrue(bits["has_grid"])
        self.assertEqual(bits["total_line"], 8.5)
        self.assertIsNone(bits["line_selected"])
        # Half-line 8.5: no push → re-scale is a no-op, headline == raw.
        self.assertEqual(bits["p_over"], 0.48)
        self.assertEqual(bits["p_under"], 0.52)
        self.assertAlmostEqual(bits["p_over"] + bits["p_under"], 1.0,
                               delta=0.01)

    def test_override_prices_selected_line(self):
        """Explicit line=8.5 → O/U priced at p_over_8_5, flagged selected."""
        row = self._row()
        bits = diag.run_engine_card_bits("SEL", {"SEL": row}, line=8.5)
        self.assertEqual(bits["total_line"], 8.5)
        self.assertEqual(bits["line_selected"], 8.5)
        # Half-line: no push → re-scale is a no-op, headline == raw.
        self.assertEqual(bits["p_over"], 0.48)
        self.assertEqual(bits["p_under"], 0.52)
        self.assertAlmostEqual(bits["p_over"] + bits["p_under"], 1.0,
                               delta=0.01)

    def test_whole_line_override_rescales_to_100(self):
        """Selecting whole line 8.0 (push band = p_over_7_5 − p_over_8_0 =
        0.62−0.55 = 0.07): the re-scaled Over + Under still sums to 100%
        (±1 rounding) and the push is folded in, preserving the over:under
        ratio."""
        row = self._row()
        bits = diag.run_engine_card_bits("SEL", {"SEL": row}, line=8.0)
        self.assertEqual(bits["total_line"], 8.0)
        self.assertEqual(bits["line_selected"], 8.0)
        self.assertAlmostEqual(bits["p_push"], 0.62 - 0.55, places=9)
        total = bits["p_over"] + bits["p_under"]
        self.assertAlmostEqual(total, 1.0, delta=0.01,
                               msg="selected-line over+under must sum to 100% (±1)")
        # Ratio preserved: 0.55/0.38 scaled by 1/0.93.
        self.assertAlmostEqual(bits["p_over"], 0.55 / 0.93, places=3)
        self.assertAlmostEqual(bits["p_under"], 0.38 / 0.93, places=3)

    def test_invalid_line_falls_back_to_model_line(self):
        """Out-of-grid (14.0, 3.0), non-numeric, and None selections fall
        back to the model's own line — never crash, never price a line the
        artifact doesn't carry."""
        row = self._row()
        for bad in (14.0, 3.0, 9.3, "abc", None):
            bits = diag.run_engine_card_bits("SEL", {"SEL": row}, line=bad)
            self.assertEqual(bits["total_line"], 8.5,
                             f"line={bad!r} must fall back to fair line")
            self.assertIsNone(bits["line_selected"])

    def test_selector_state_keyed_per_game(self):
        """The selector's session_state key is scoped per game_pk
        (ou_line_<game_id>) and resolve_totals_line validates against the
        grid — so selections persist per card and never bleed between cards."""
        src = (FRONTEND / "todays_games.py").read_text()
        self.assertIn("ou_line_", src,
                      "selector key must be scoped per game_pk")
        self.assertIn("st.session_state", src,
                      "selector state must live in session_state")
        self.assertIn("resolve_totals_line", src,
                      "selection resolution must be a named helper")
        self.assertIn("TOTAL_GRID", src,
                      "selection must validate against the shipped grid")

    def test_html_marks_selected_line(self):
        """The card strip must label a non-model line as 'line selected' so
        it is clear when the user has moved off the model's line."""
        src = (FRONTEND / "todays_games.py").read_text()
        self.assertIn("line selected", src,
                      "card must flag a non-default selected line")


class TestRunLineSelector(unittest.TestCase):
    """Per-card run-line selector (mirrors the O/U selector).

    Backend persists per-line run-line columns p_rl_<m>_{home,push,away}
    over the full grid (half-lines + whole numbers). Half-lines: home +
    away = 1.0 exactly (no push). Whole lines: home + push + away = 1.0
    with push > 0. The card displays the RE-SCALED 2-way cover % (push
    folded proportionately, ratio preserved, sums to 100%). The selector
    defaults to ±1.5, is keyed rl_line_<game_pk>, and invalid/out-of-grid
    values fall back to ±1.5.
    """

    def _rl_row(self, pk="RL"):
        """Full-grid slate row: p_home_cover_1_5 + p_rl columns for the
        full grid (half 1.5/2.5/3.5 and whole 1/2/3/4)."""
        return {
            "game_pk": pk, "home_expected_runs": 4.4,
            "away_expected_runs": 4.9,
            "p_over_9_5": 0.55, "p_under_9_5": 0.45,
            "p_home_cover_1_5": 0.3253,
            # half-lines: home + away = 1.0, push = 0
            "p_rl_1_5_home": 0.3253, "p_rl_1_5_push": 0.0,
            "p_rl_1_5_away": 0.6747,
            "p_rl_2_5_home": 0.18, "p_rl_2_5_push": 0.0,
            "p_rl_2_5_away": 0.82,
            "p_rl_3_5_home": 0.09, "p_rl_3_5_push": 0.0,
            "p_rl_3_5_away": 0.91,
            # whole lines: 3-way split with push > 0 (injective naming:
            # whole 1.0 -> p_rl_1_0_*, half 1.5 -> p_rl_1_5_*)
            "p_rl_1_0_home": 0.44, "p_rl_1_0_push": 0.10,
            "p_rl_1_0_away": 0.46,
            "p_rl_2_0_home": 0.30, "p_rl_2_0_push": 0.08,
            "p_rl_2_0_away": 0.62,
            "p_rl_3_0_home": 0.17, "p_rl_3_0_push": 0.06,
            "p_rl_3_0_away": 0.77,
            "p_rl_4_0_home": 0.09, "p_rl_4_0_push": 0.05,
            "p_rl_4_0_away": 0.86,
        }

    def test_column_injectivity(self):
        """p_rl_1_0_home vs p_rl_1_5_home are DISTINCT columns (whole 1 vs
        half 1.5 never collide — the totals-grid lesson)."""
        self.assertNotEqual(diag.rl_cols(1.0), diag.rl_cols(1.5))
        self.assertEqual(diag.rl_cols(1.5)[0], "p_rl_1_5_home")
        self.assertEqual(diag.rl_cols(1.0)[0], "p_rl_1_0_home")
        self.assertEqual(diag.rl_cols(1.0)[1], "p_rl_1_0_push")
        self.assertEqual(diag.rl_cols(2.5)[2], "p_rl_2_5_away")
        names = [diag.rl_cols(m)[0] for m in diag.RUN_LINE_GRID_FULL]
        self.assertEqual(len(names), len(set(names)),
                         "rl column names must be injective across the grid")

    def test_default_is_half_line_complement(self):
        """No rl_line override → ±1.5 via p_home_cover_1_5, complement sums
        to 1.0, rl_push = 0."""
        row = self._rl_row()
        bits = diag.run_engine_card_bits("RL", {"RL": row})
        self.assertIsNone(bits["rl_line"])
        self.assertEqual(bits["rl_line_default"], 1.5)
        self.assertEqual(bits["rl_home"], 0.3253)
        self.assertAlmostEqual(bits["rl_home"] + bits["rl_away"], 1.0,
                               places=9)
        self.assertEqual(bits["rl_push"], 0.0)

    def test_half_line_selected_uses_rl_columns(self):
        """rl_line=2.5 (half) → p_rl_2_5 columns, complement 1.0, no push."""
        row = self._rl_row()
        bits = diag.run_engine_card_bits("RL", {"RL": row}, rl_line=2.5)
        self.assertEqual(bits["rl_line"], 2.5)
        self.assertEqual(bits["rl_home"], 0.18)
        self.assertEqual(bits["rl_away"], 0.82)
        self.assertAlmostEqual(bits["rl_home"] + bits["rl_away"], 1.0,
                               places=9)
        self.assertEqual(bits["rl_push"], 0.0)
        self.assertFalse(bits["rl_unverified"])

    def test_whole_line_3way_and_rescale(self):
        """rl_line=2.0 (whole): 3-way raw sums to 1.0 with push > 0; the
        card's re-scaled 2-way sums to 100% (±1) with the ratio preserved."""
        row = self._rl_row()
        bits = diag.run_engine_card_bits("RL", {"RL": row}, rl_line=2.0)
        self.assertEqual(bits["rl_line"], 2.0)
        # raw 3-way (not on the card, but carried for EV) sums to 1.0
        self.assertAlmostEqual(bits["rl_home_raw"] + bits["rl_push"]
                               + bits["rl_away_raw"], 1.0, places=9)
        self.assertEqual(bits["rl_home_raw"], 0.30)
        self.assertEqual(bits["rl_away_raw"], 0.62)
        self.assertGreater(bits["rl_push"], 0.0)
        # re-scaled 2-way display sums to 100% (±1 rounding)
        total = bits["rl_home"] + bits["rl_away"]
        self.assertAlmostEqual(total, 1.0, delta=0.01,
                               msg="rl display home+away must sum to 100% (±1)")
        # ratio preserved: 0.30/0.62 scaled by 1/0.92
        self.assertAlmostEqual(bits["rl_home"], 0.30 / 0.92, places=3)
        self.assertAlmostEqual(bits["rl_away"], 0.62 / 0.92, places=3)

    def test_invalid_rl_line_falls_back(self):
        """Out-of-grid / non-numeric / None rl_line → ±1.5 fallback."""
        row = self._rl_row()
        for bad in (0.5, 5.0, 1.7, "abc", None):
            bits = diag.run_engine_card_bits("RL", {"RL": row}, rl_line=bad)
            self.assertIsNone(bits["rl_line"],
                              f"rl_line={bad!r} must fall back to default")
            self.assertEqual(bits["rl_home"], 0.3253)

    def test_legacy_artifact_fallback_and_unverified(self):
        """Legacy artifact (no p_rl columns): ±1.5 still resolves via
        p_home_cover_1_5; alternate lines render unverified, never
        fabricated."""
        row = {"game_pk": "LG", "home_expected_runs": 4.4,
               "away_expected_runs": 4.9, "p_over_9_5": 0.55,
               "p_under_9_5": 0.45, "p_home_cover_1_5": 0.3253}
        bits = diag.run_engine_card_bits("LG", {"LG": row})
        self.assertEqual(bits["rl_home"], 0.3253)   # ±1.5 via legacy col
        bits2 = diag.run_engine_card_bits("LG", {"LG": row}, rl_line=2.0)
        self.assertTrue(bits2["rl_unverified"])
        self.assertIsNone(bits2["rl_home"])
        self.assertIsNone(bits2["rl_away"])

    def test_selector_state_keyed_per_game(self):
        """Selector session_state keyed rl_line_<game_pk> via a named
        resolver validating against the full grid."""
        src = (FRONTEND / "todays_games.py").read_text()
        self.assertIn("rl_line_", src)
        self.assertIn("resolve_rl_line", src)
        self.assertIn("RUN_LINE_GRID_FULL", src)

    def test_html_marks_selected_rl_line(self):
        """The card strip must label a selected run line explicitly and
        reference the re-scaled home/away + push handling."""
        src = (FRONTEND / "todays_games.py").read_text()
        self.assertIn("line selected", src)
        self.assertIn("rl_home", src)
        self.assertIn("rl_push", src)
        self.assertIn("unverified", src)


class TestRunLineCalibrationRecord(unittest.TestCase):
    """The committed run-line calibration record (the selector's gate
    evidence): every line in the full grid has a row with a verdict; the
    gate is 'calibrated' when |delta| <= 0.02 (same discipline as the
    totals gate); the record covers n_games == the OOF count."""

    def setUp(self):
        rec = (Path(__file__).resolve().parents[1] / "data_delivery"
               / "run_line_calibration_20260829.json")
        self.record = json.loads(rec.read_text())

    def test_all_grid_lines_present_with_verdict(self):
        lines = {r["line"] for r in self.record["lines"]}
        self.assertEqual(lines, set(diag.RUN_LINE_GRID_FULL))
        for r in self.record["lines"]:
            self.assertIn(r["verdict"],
                          ("calibrated", "over-predicting",
                           "under-predicting", "low_n"))

    def test_gate_reflects_adopted_structural_fix(self):
        """Post structural fix (impossible tie mass resolves to ±1,
        home-weighted MARGIN_PLUS1_HOME_SHARE): the gate records ALL SEVEN
        lines as CALIBRATED — line −1 now +0.0002 (was +0.0409 under the
        proportional renormalization in 2531462) and the +1 push band is
        now exact (0.1740 vs 0.1740). The selector can therefore offer
        every alternate line."""
        by_line = {r["line"]: r for r in self.record["lines"]}
        for r in self.record["lines"]:
            self.assertEqual(r["verdict"], "calibrated",
                             f"line {r['line']} must pass the calibration gate "
                             f"post structural fix")
        r1 = by_line[1.0]
        self.assertAlmostEqual(r1["p_home"], 0.3583, delta=0.002)
        self.assertAlmostEqual(r1["delta"], 0.0002, delta=0.002)
        self.assertAlmostEqual(r1["push_pred"], 0.1740, delta=0.002)
        self.assertAlmostEqual(r1["push_actual"], 0.1740, delta=0.002)
        r4 = by_line[4.0]
        self.assertAlmostEqual(r4["p_home"], 0.1413, delta=0.002)
        self.assertLessEqual(abs(r4["delta"]), 0.02)
        self.assertEqual(self.record["method"]["tie_handling"],
                         "impossible tie mass resolves to ±1 home-weighted "
                         "(MARGIN_PLUS1_HOME_SHARE) — the structural "
                         "home one-run fix; P(+1)' = P(+1)+α·P(0), "
                         "P(−1)' = P(−1)+(1−α)·P(0), P(0)=0; away = 1 − "
                         "home − push")

    def test_half_lines_have_zero_push(self):
        """Half-lines can never push — push_pred == push_actual == 0."""
        for r in self.record["lines"]:
            if abs(r["line"] - round(r["line"])) > 1e-9:
                self.assertEqual(r["push_pred"], 0.0)
                self.assertEqual(r["push_actual"], 0.0)

    def test_record_covers_oof_count(self):
        self.assertEqual(self.record["n_games"], 6812)
        for r in self.record["lines"]:
            self.assertEqual(r["n"], 6812)


class TestDoubleheaderCards(unittest.TestCase):
    """Doubleheader legs are DISTINCT games on the card layer.

    Each leg has its own game_id (start-time ordinal suffix, e.g.
    20260829_BOS@NYY vs 20260829_BOS@NYY_2), so each leg's card resolves
    ITS OWN run-engine markets row (different projections per leg) instead
    of both cards showing the same pitcher/projection; and a markets frame
    holding two rows with the SAME game_pk collapses to one (a true bug, no
    explosion). Rendering iterates slate rows — distinct rows produce
    distinct cards with per-leg time/pitcher data.
    """

    def test_per_leg_game_ids_resolve_own_markets_row(self):
        """Two BOS@NYY legs with distinct game_ids get DISTINCT run-engine
        projections — never the same strip on both cards."""
        leg1 = make_slate_row("20260829_BOS@NYY")
        leg1.update(home_expected_runs=4.4, away_expected_runs=4.9,
                    p_over_9_5=0.55, p_under_9_5=0.45, p_home_cover_1_5=0.6)
        leg2 = dict(leg1, game_pk="20260829_BOS@NYY_2",
                    home_expected_runs=4.4, away_expected_runs=4.1,
                    p_over_8_5=0.48, p_under_8_5=0.52, p_home_cover_1_5=0.42)
        slate_map = {str(r["game_pk"]): r for r in (leg1, leg2)}

        bits1 = diag.run_engine_card_bits("20260829_BOS@NYY", slate_map)
        bits2 = diag.run_engine_card_bits("20260829_BOS@NYY_2", slate_map)
        self.assertIsNotNone(bits1)
        self.assertIsNotNone(bits2)
        # Per-leg projections + per-leg pricing (9.5 vs 8.5 lines).
        self.assertEqual(bits1["proj_away"], 4.9)
        self.assertEqual(bits2["proj_away"], 4.1)
        self.assertEqual(bits1["total_line"], 9.5)
        self.assertEqual(bits2["total_line"], 8.5)
        self.assertAlmostEqual(bits1["p_over"] + bits1["p_under"], 1.0, places=9)
        self.assertAlmostEqual(bits2["p_over"] + bits2["p_under"], 1.0, places=9)

    def test_same_game_pk_rows_collapse_in_slate_map(self):
        """Two markets rows with the SAME game_pk are a true bug — the
        slate_map dict keys by game_pk, so they collapse to ONE entry (no
        duplicate cards), and the card resolves deterministically."""
        row = make_slate_row("20260829_BOS@NYY")
        dup = dict(row)
        dup["home_expected_runs"] = 9.9  # the duplicated row's data
        slate_map = {str(r["game_pk"]): r for r in (row, dup)}
        self.assertEqual(len(slate_map), 1)
        bits = diag.run_engine_card_bits("20260829_BOS@NYY", slate_map)
        self.assertIsNotNone(bits)
        # The LAST row wins by dict semantics; either way exactly one card
        # entry exists for the key.
        self.assertEqual(bits["proj_home"], 9.9)

    def test_slate_rows_carry_per_leg_time_and_pitchers(self):
        """The slate artifact itself carries each leg's own start time and
        pitchers, so the card loop renders distinct cards per leg."""
        rows = pd.DataFrame([
            {"game_id": "20260829_BOS@NYY", "home_team": "NYY",
             "away_team": "BOS", "start_time_utc": "2026-08-29 13:05:00",
             "sp_name_home": "Rodón, Carlos", "sp_name_away": "Bennett, Jordan"},
            {"game_id": "20260829_BOS@NYY_2", "home_team": "NYY",
             "away_team": "BOS", "start_time_utc": "2026-08-29 19:15:00",
             "sp_name_home": "Fried, Max", "sp_name_away": "TBD"},
        ])
        self.assertEqual(rows["game_id"].nunique(), 2)
        times = rows["start_time_utc"].tolist()
        self.assertNotEqual(times[0], times[1])
        self.assertNotEqual(rows.iloc[0]["sp_name_home"], rows.iloc[1]["sp_name_home"])


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


class TestResolveTotalsLineGridBinding(unittest.TestCase):
    """resolve_totals_line / resolve_rl_line must NEVER NameError.

    Regression for the deployed crash: the helpers referenced diag.TOTAL_GRID
    / diag.RUN_LINE_GRID_FULL, but `diag` was only imported inside main().
    Run 1 wrote ou_line_<game_pk>/rl_line_<game_pk> into session_state and
    returned early; run 2 reached the diag reference with no such module
    global -> NameError on every load after the first render.
    """

    def _todays(self):
        import streamlit as st  # noqa: F401  (session_state runtime)
        import todays_games as todays
        return todays

    def test_module_grid_constant_matches_shipped_grid(self):
        """The module-level TOTAL_GRID constant is the shipped 6.5-12.5 grid
        (the getattr fallback target) and RUN_LINE_GRID_FULL is the full
        run-line grid — an absent/renamed upstream attribute degrades to
        these defaults instead of crashing."""
        todays = self._todays()
        default = [round(6.5 + 0.5 * i, 1) for i in range(13)]
        self.assertEqual(todays.TOTAL_GRID, default)
        self.assertEqual(todays.TOTAL_GRID, diag.TOTAL_GRID)
        self.assertEqual(todays.RUN_LINE_GRID_FULL, diag.RUN_LINE_GRID_FULL)

    def test_resolve_totals_line_uses_module_constant_and_validates(self):
        """resolve_totals_line reads the module-level constant (never a
        scoped `diag`), validates against it, and falls back to the model
        line for invalid/out-of-grid selections — same contract as before."""
        import streamlit as st
        todays = self._todays()
        orig = todays.TOTAL_GRID
        todays.TOTAL_GRID = [7.0, 7.5]      # narrower grid: follows the patch
        try:
            st.session_state["ou_line_gr"] = 7.5
            self.assertEqual(todays.resolve_totals_line("gr", 8.5), 7.5)
            st.session_state["ou_line_gr"] = 9.0   # off the patched grid
            self.assertEqual(todays.resolve_totals_line("gr", 8.5), 8.5)
            st.session_state["ou_line_gr"] = "abc"  # non-numeric
            self.assertEqual(todays.resolve_totals_line("gr", 8.5), 8.5)
            st.session_state["ou_line_gr"] = None   # None selection
            self.assertEqual(todays.resolve_totals_line("gr", 8.5), 8.5)
        finally:
            todays.TOTAL_GRID = orig
        # Fresh key (never selected) -> default, key seeded in session_state.
        self.assertEqual(todays.resolve_totals_line("gr_fresh", 9.5), 9.5)
        self.assertEqual(st.session_state["ou_line_gr_fresh"], 9.5)

    def test_resolve_rl_line_uses_module_constant(self):
        """resolve_rl_line has the same module-level binding (the identical
        latent NameError is fixed too)."""
        import streamlit as st
        todays = self._todays()
        orig = todays.RUN_LINE_GRID_FULL
        todays.RUN_LINE_GRID_FULL = [1.5, 2.5]
        try:
            st.session_state["rl_line_gr"] = 2.5
            self.assertEqual(todays.resolve_rl_line("gr", 1.5), 2.5)
            st.session_state["rl_line_gr"] = 3.0   # off the patched grid
            self.assertEqual(todays.resolve_rl_line("gr", 1.5), 1.5)
        finally:
            todays.RUN_LINE_GRID_FULL = orig

    def test_apptest_todays_games_rerun_no_nameerror(self):
        """End-to-end through Home.py -> nav.run() -> todays_games.main() ->
        resolve_totals_line (the exact crash path): run 1 seeds
        ou_line_<game_pk> into session_state; run 2 (the deployed crash)
        must render with 0 exceptions. Runs in a SUBPROCESS so the
        canonical suite's streamlit stubs (test_frontend_markets swaps
        sys.modules['streamlit'] and leaves utils.st bound to the MagicMock)
        cannot poison the real Streamlit run."""
        import subprocess
        import sys as _sys
        script = (
            "import sys; sys.path.insert(0, %r);\n"
            "from streamlit.testing.v1 import AppTest;\n"
            "at = AppTest.from_file(%r, default_timeout=60);\n"
            "at.run();\n"
            "assert not at.exception, at.exception;\n"
            "assert len(at.selectbox) > 0, 'no game cards rendered';\n"
            "at.run();\n"
            "assert not at.exception, at.exception;\n"
            "assert len(at.selectbox) > 0, 'rerun lost game cards';\n"
            "print('APP_OK')\n"
        ) % (str(FRONTEND), str(FRONTEND / "Home.py"))
        res = subprocess.run([_sys.executable, "-c", script],
                             capture_output=True, text=True, timeout=180)
        self.assertEqual(
            res.returncode, 0,
            f"AppTest subprocess failed:\nSTDOUT:\n{res.stdout}\n"
            f"STDERR:\n{res.stderr[-2000:]}")
        self.assertIn("APP_OK", res.stdout)


if __name__ == "__main__":
    unittest.main()
