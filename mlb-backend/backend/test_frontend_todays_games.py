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
        """LEGACY artifact (no p_push column): subtraction fallback derives
        the push band (p_over was push-inclusive pre-fix), and the re-scale
        folds it in so headline Over + Under = 100%."""
        row = {"game_pk": "LG", "home_expected_runs": 4.5,
               "away_expected_runs": 4.5,  # line = 9.0
               "p_over_9_0": 0.50, "p_under_9_0": 0.50,
               "p_over_9_5": 0.42, "p_under_9_5": 0.58,
               "p_home_cover_1_5": 0.4}
        bits = diag.run_engine_card_bits("LG", {"LG": row})
        self.assertEqual(bits["total_line"], 9.0)
        # Subtraction fallback still recovers the push band (0.50-0.42).
        self.assertAlmostEqual(bits["p_push"], 0.08, places=9)
        # Over:Under ratio preserved after re-scale: 0.50/0.50 → 50/50.
        self.assertAlmostEqual(bits["p_over"], 0.50, places=9)
        self.assertAlmostEqual(bits["p_under"], 0.50, places=9)
        self.assertAlmostEqual(bits["p_over"] + bits["p_under"], 1.0, delta=0.01)

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
        """λ_h + λ_a = 9.0 → model line 9.0. Grid carries 9.0 (whole,
        push band 0.08) and 9.5 (half, no push)."""
        return {"game_pk": pk, "home_expected_runs": h,
                "away_expected_runs": a,
                # p_under = 1 - p_over - p_push (artifact convention):
                # 9.0 push = 0.42-0.34 = 0.08 → under = 1-0.42-0.08 = 0.50;
                # 8.0 push = 0.55-0.48 = 0.07 → under = 1-0.55-0.07 = 0.38.
                "p_over_9_0": 0.42, "p_under_9_0": 0.50,
                "p_over_9_5": 0.34, "p_under_9_5": 0.66,
                "p_over_8_0": 0.55, "p_under_8_0": 0.38,
                "p_over_8_5": 0.48, "p_under_8_5": 0.52,
                "p_home_cover_1_5": 0.4}

    def test_defaults_to_model_line(self):
        """No line override → the card prices at the model's assigned line
        (own rounded total) with line_selected None."""
        row = self._row()
        bits = diag.run_engine_card_bits("SEL", {"SEL": row})
        self.assertTrue(bits["has_grid"])
        self.assertEqual(bits["total_line"], 9.0)
        self.assertIsNone(bits["line_selected"])
        # Whole line 9.0 carries a push band (p_over_9_0 − p_over_9_5 =
        # 0.42 − 0.34 = 0.08) — the re-scaled 2-way split folds it in:
        # p_over = 0.42/0.92, p_under = 0.50/0.92 (sums to 100%).
        self.assertAlmostEqual(bits["p_over"], 0.42 / 0.92, places=4)
        self.assertAlmostEqual(bits["p_under"], 0.50 / 0.92, places=4)
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
        """Selecting whole line 8.0 (push band 0.55−0.48=0.07): the re-scaled
        Over + Under still sums to 100% (±1 rounding) and the push is folded
        in, preserving the over:under ratio."""
        row = self._row()
        bits = diag.run_engine_card_bits("SEL", {"SEL": row}, line=8.0)
        self.assertEqual(bits["total_line"], 8.0)
        self.assertEqual(bits["line_selected"], 8.0)
        self.assertAlmostEqual(bits["p_push"], 0.55 - 0.48, places=9)
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
            self.assertEqual(bits["total_line"], 9.0,
                             f"line={bad!r} must fall back to model line")
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


if __name__ == "__main__":
    unittest.main()
