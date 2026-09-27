"""Regression tests for the NBA totals & point-spread page.

These lock the MLB-parity contract and the push-resolution / pricing
conventions that the page depends on. The conventions matter because the
NBA spread grid is integer (-20..20) plus ±0.5, so every whole-number line
can push where MLB's -1.5 run line never can — a convention that reads
plausibly wrong and produces plausible-looking numbers.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FRONTEND = Path(__file__).resolve().parent
sys.path.insert(0, str(FRONTEND))

import nba_market_diagnostics as nd  # noqa: E402
import nba_markets_page as page  # noqa: E402


def _decided(rows) -> pd.DataFrame:
    """Build a minimal decided frame with a real priced grid."""
    out = []
    for i, (home, away, hs, a_s, fair_t, fair_s, p_over, p_home) in enumerate(
            rows):
        r = {
            "game_id": f"00{i:08d}", "gameday": "2026-01-15",
            "kind": "oof", "decided": True,
            "home_team": home, "away_team": away,
            "home_score": hs, "away_score": a_s,
            "total": hs + a_s, "margin": hs - a_s,
            "fair_total": float(fair_t), "fair_spread": float(fair_s),
            "p_over_fair": 0.5,
            "p_home_win_derived": 0.55,
        }
        tl = int(round(fair_t))
        r[f"p_over_{tl}"] = p_over
        r[f"p_push_total_{tl}"] = 0.0
        sl = nd._spread_label(int(round(fair_s)))
        r[f"p_home_cover_{sl}"] = p_home
        r[f"p_push_{sl}"] = 0.0
        out.append(r)
    return pd.DataFrame(out)


SIMPLE = _decided([
    # total 230 vs fair 225 -> Over wins; p_over 0.60 says Over
    ("BOS", "NYK", 120, 110, 225, -2.0, 0.60, 0.60),
    # total 210 vs fair 225 -> Under wins; p_over 0.40 says Under
    ("DEN", "LAL", 105, 105, 225, 2.0, 0.40, 0.40),
    # total 225 EXACTLY on the line -> totals PUSH
    ("MIA", "CHI", 112, 113, 225, 1.0, 0.55, 0.55),
    # margin 2 vs fair spread 2 -> spread PUSH (margin -1 on the row above
    # does not reach its own line of 1)
    ("CHI", "PHX", 115, 113, 225, 2.0, 0.55, 0.55),
])


# --- The parity contract ---------------------------------------------------

def test_page_has_every_mlb_section_tab_and_expander():
    from compare_totals_dashboards import REQUIRED_EXPANDERS, REQUIRED_TABS, \
        REQUIRED_SECTIONS, render, check
    failures = check("nba", render("nba"))
    assert failures == [], failures
    for section in REQUIRED_SECTIONS:
        assert section  # the contract is what matters; keep the import live
    assert len(REQUIRED_TABS) == 5
    assert len(REQUIRED_EXPANDERS) == 3


def test_both_sports_satisfy_the_same_contract():
    from compare_totals_dashboards import check, render
    assert check("mlb", render("mlb")) == []
    assert check("nba", render("nba")) == []


# --- Push resolution ------------------------------------------------------

def test_total_equal_to_a_whole_line_is_a_push():
    frame = nd.totals_history_frame(SIMPLE)
    push = frame[frame.winner == "Push"]
    assert len(push) == 1
    assert pd.isna(push.iloc[0]["correct"])


def test_pushes_are_excluded_from_both_sides_of_the_win_rate():
    frame = nd.totals_history_frame(SIMPLE)
    stats = nd.history_win_rate(frame)
    assert len(frame) == 4
    assert stats["n_games"] == 3          # 4 rows, 1 push
    assert stats["n_pushes"] == 1
    # All three decided games were correct picks (Over 230 / Under 210 /
    # Over 228 against a 225 line).
    assert stats["win_rate"] == 1.0


def test_an_integer_spread_line_pushes_where_mlbs_half_line_cannot():
    frame = nd.runline_history_frame(SIMPLE)
    pushed = frame[frame.winner == "Push"]
    assert len(pushed) == 1
    # fair_spread 2.0 and margin 2.0 -> exactly on the line, so a push.
    assert pushed.iloc[0]["line"] == 2.0
    stats = nd.history_win_rate(frame)
    assert stats["n_games"] == 3 and stats["n_pushes"] == 1
    # Two correct of three decided: BOS covers -2, LAL covers +2, and CHI
    # misses its own -1 line because the model leaned home.
    assert stats["win_rate"] == round(2 / 3, 6)


def test_no_half_line_outside_the_priced_grid_is_ever_requested():
    """The artifact has integers plus ±0.5 only — a -1.5 column does not exist."""
    for mag in nd.SPREAD_LINE_CHOICES:
        grid = -0.5 if abs(mag) < 1.0 else -int(round(mag))
        label = nd._spread_label(int(grid))
        assert label in {"0", "0_5", "m0_5"} or label.lstrip("m").isdigit()
    assert 1.5 not in nd.SPREAD_GRID_CUT
    assert 0.5 in nd.SPREAD_GRID_CUT


def test_result_cell_renders_a_dash_for_a_push():
    assert page._result_cell(np.nan) == "<td>—</td>"
    assert page._result_cell(True).count("✓") == 1
    assert page._result_cell(False).count("✗") == 1


# --- Card / history cross-checks -----------------------------------------

def test_spread_card_agrees_with_its_own_history_frame():
    frame = nd.runline_history_frame(SIMPLE)
    stats = nd.history_win_rate(frame)
    card = nd.winner_cards(SIMPLE)["run_line"]
    assert card["n"] == stats["n_games"]
    assert card["win_rate"] == stats["win_rate"]


def test_totals_card_agrees_with_its_own_history_frame():
    frame = nd.totals_history_frame(SIMPLE)
    stats = nd.history_win_rate(frame)
    card = nd.totals_monitor_stats(SIMPLE)
    assert card["n"] == stats["n_games"]
    assert card["n_pushes"] == stats["n_pushes"]
    assert card["win_rate"] == stats["win_rate"]


def test_card_win_rate_is_conditional_accuracy_not_a_joint_rate():
    """A joint (p>0.5 AND correct) rate understates by every lean that
    pointed below 50% and was right — the under-dogs' correct calls."""
    y = np.array([1.0, 0.0, 0.0])
    p = np.array([0.9, 0.4, 0.1])
    # Every lean is right here: Over 0.9 hits, Under 0.4 misses (y=0 means
    # the event did NOT happen, so the Under call is correct), Under 0.1 too.
    metrics = nd.binary_card_metrics(y, p)
    assert metrics["win_rate"] == 1.0
    # The joint rate would read 1/3 — the bug this asserts against.
    joint = round(float(((p > 0.5) & (y == 1.0)).mean()), 6)
    assert joint == round(1 / 3, 6)
    assert metrics["win_rate"] != joint


def test_rows_at_exactly_a_half_probability_make_no_pick():
    y = np.array([1.0, 0.0, 1.0])
    p = np.array([0.9, 0.1, 0.5])
    # The p=0.5 row expresses no lean, so it leaves the denominator.
    assert nd.binary_card_metrics(y, p)["win_rate"] == 1.0


# --- Pricing source -------------------------------------------------------

def test_the_grid_is_priced_in_preference_to_the_degenerate_column():
    frame = SIMPLE.copy()
    frame["p_over_fair"] = 0.5          # what the artifact actually ships
    series = nd._over_fair_series(frame)
    assert set(np.round(series.dropna(), 3)) == {0.6, 0.4, 0.55}
    assert not (series == 0.5).any()


def test_side_filter_partitions_the_population():
    view = nd.totals_history_frame(SIMPLE)
    over = nd.filter_history_by_side(view, "Over")
    under = nd.filter_history_by_side(view, "Under")
    assert len(over) + len(under) == len(view)
    assert set(over["pick"]) == {"Over"}
    assert len(nd.filter_history_by_side(view, "All")) == len(view)


def test_pooled_win_rate_lies_between_its_two_sides():
    card = nd.totals_monitor_stats(SIMPLE)
    sides = [card["sides"][k]["win_rate"] for k in card["sides"]]
    assert min(sides) <= card["win_rate"] <= max(sides)


def test_confidence_threshold_is_a_nested_subset():
    low = nd.totals_monitor_stats(SIMPLE, min_pct=50)
    high = nd.totals_monitor_stats(SIMPLE, min_pct=54)
    assert high["n"] <= low["n"]


# --- Calibration ----------------------------------------------------------

def test_absent_calibration_map_yields_no_calibrated_ece():
    """An unavailable calibrated number must be None, not a copy of raw."""
    cards = nd.winner_cards(SIMPLE, None)
    for card in cards.values():
        assert card["ece_calibrated"] is None
        assert card["ece_raw"] is not None


def test_a_shipped_calibration_map_populates_ece_calibrated():
    monitor = {"calibration_cards": {
        "totals": {"225": {"over": {"a": 1.0, "b": 0.0}}}}}
    cards = nd.winner_cards(SIMPLE, monitor)
    assert cards["over_under"]["ece_calibrated"] is not None


def test_missing_calibration_line_is_not_invented():
    assert nd.calibration_coefficients(
        {"calibration_cards": {"totals": {"225": {"over": {"a": 1.0, "b": 0.0}}}}},
        "totals", 999) is None
    assert nd.calibration_coefficients(None, "totals", 225) is None


# --- Empty / degraded states ---------------------------------------------

def test_no_decided_rows_gives_three_honest_empty_cards():
    cards = nd.winner_cards(pd.DataFrame())
    assert set(cards) == {k for k, _, _ in nd.WINNER_CARDS}
    for card in cards.values():
        assert card["n"] == 0
        assert card["win_rate"] is None
        assert card["brier"] is None


def test_history_frames_are_empty_not_absent_without_rows():
    for frame in (nd.totals_history_frame(pd.DataFrame()),
                  nd.runline_history_frame(pd.DataFrame())):
        assert frame.empty
        assert list(frame.columns) == list(nd._HISTORY_COLUMNS)


def test_drift_and_coverage_report_missing_artifacts_instead_of_blank_tables():
    from streamlit.testing.v1 import AppTest
    script = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import nba_market_diagnostics as nd\n"
        "nd.render_run_engine_drift(None)\n"
        "nd.render_run_engine_coverage(None)\n" % str(FRONTEND)
    )
    app = AppTest.from_string(script)
    app.run()
    assert not app.exception
    text = " ".join(str(i.value).lower() for i in app.info)
    assert "no run-engine drift" in text
    assert "no run-engine coverage" in text
