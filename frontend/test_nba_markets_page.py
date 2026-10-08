"""Regression tests for the NBA Totals & Run Lines page (post-redo).

These lock the structural parity with MLB's markets page and the pricing
semantics that only hold because of two artifact facts:

  1. The spread columns are THRESHOLD form — ``p_home_cover_{L}`` =
     P(margin > L) — so a displayed home spread of -5 is threshold +5 and
     the two sides of a LINE cell must ALWAYS carry opposite signs.
  2. Fair lines are computed here by grid argmin (the artifact's own 50/50
     point); the stored ``fair_total`` / ``fair_spread`` columns are never
     trusted, because ``fair_spread`` measurably disagrees with the grid.
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
    """Decided rows priced on the grid at (fair_total, home_threshold).

    ``rows`` items: (home, away, hs, a_s, total_line, home_thr, p_over,
    p_home_cover). Ships the row's own fair-line columns PLUS a ±6 window
    of totals-grid columns with a 0.06/pt slope through 0.5 at the fair
    line, so shifted-line pricing (relativized pairs, fixed lines) has
    real brackets instead of NaN-dropping every row.
    """
    out = []
    for i, (home, away, hs, a_s, tl, thr, p_over, p_home) in enumerate(rows):
        r = {
            "game_id": f"00{i:08d}", "gameday": "2026-01-15",
            "kind": "oof", "decided": True,
            "home_team": home, "away_team": away,
            "home_score": hs, "away_score": a_s,
            "total": hs + a_s, "margin": hs - a_s,
            "p_home_win_derived": 0.55,
            # expected totals consistent with the fair line (mu_h + mu_a = tl)
            "mu_h": tl / 2.0 + 1.5, "mu_a": tl / 2.0 - 1.5,
            # stored fair columns are deliberately set to values that would
            # change the answer if they were trusted
            "fair_total": 999.0, "fair_spread": 999.0,
        }
        r[f"p_over_{tl}"] = p_over
        r[f"p_under_{tl}"] = round(1.0 - p_over - 0.02, 4)
        tlab = nd._label(float(thr))
        r[f"p_home_cover_{tlab}"] = p_home
        r[f"p_away_cover_{tlab}"] = round(1.0 - p_home - 0.03, 4)
        r[f"p_push_{tlab}"] = 0.03
        # totals-grid window AROUND (never AT) the fair line: steep enough
        # (0.25/pt, clamped) that the argmin 50/50 point stays the authored
        # fair line, whose two-way deviation from 0.5 is at most 0.112 here
        for k in range(int(tl) - 6, int(tl) + 6):
            if k == tl:
                continue
            r[f"p_over_{k}"] = round(min(max(0.5 - 0.25 * (k - tl), 0.02),
                                         0.98), 4)
            r[f"p_under_{k}"] = round(1.0 - r[f"p_over_{k}"], 4)
        out.append(r)
    return pd.DataFrame(out)


# four games: two clean picks, one totals push (total == line), one
# spread push (margin == threshold)
SIMPLE = _decided([
    ("BOS", "NYK", 120, 110, 225, 2.0, 0.60, 0.60),   # Over wins; home covers T=2
    ("DEN", "LAL", 105, 105, 225, -3.0, 0.40, 0.30),  # Under wins; away covers
    ("MIA", "CHI", 112, 113, 225, 1.0, 0.55, 0.55),   # total 225 == line -> PUSH
    ("PHX", "UTA", 115, 113, 230, 2.0, 0.52, 0.51),   # margin 2 == T=2 -> spread PUSH
])


# --- Fair lines are computed, never trusted --------------------------------

def test_stored_fair_columns_are_ignored():
    """fair_total/fair_spread are set to 999 in the fixture; the frames must
    price at the grid lines instead."""
    frame = nd.totals_history_frame(SIMPLE)
    assert set(frame["line"]) == {225.0, 230.0}
    rl = nd.runline_history_frame(SIMPLE)
    # home line = -T; the fixture thresholds are 2/-3/1/2
    assert set(rl["line"]) == {-2.0, 3.0, -1.0}


def test_fair_total_argmin_picks_the_50_50_point_ties_lower():
    """A synthetic crossing: p_over dips through 0.5 between two lines.
    Every row prices ALL candidates (the argmin runs over the whole grid)."""
    rows = []
    for tl in (220, 221, 222):
        r = {"home_team": "A", "away_team": "B", "home_score": 111,
             "away_score": 111, "total": 222, "margin": 0,
             "p_home_win_derived": 0.5, "fair_total": 999.0,
             "fair_spread": 999.0}
        for cand in (220, 221, 222):
            r[f"p_over_{cand}"] = {220: 0.45, 221: 0.50, 222: 0.55}[cand]
            r[f"p_under_{cand}"] = round(1.0 - r[f"p_over_{cand}"], 6)
        rows.append(r)
    frame = pd.DataFrame(rows)
    lines = nd.fair_total_lines(frame)
    # line 221 has |p - 0.5| = 0 -> the argmin for every row; row0's own
    # priced line (220) is never preferred over a better grid point.
    assert np.allclose(lines, [221.0, 221.0, 221.0])


def test_displayed_spread_sides_always_carry_opposite_signs():
    rl = nd.runline_history_frame(SIMPLE)
    for _, r in rl.iterrows():
        assert float(r["line"]) == -float(r["_away_line"]), \
            f"{r['home']} {r['line']} / {r['away']} {r['_away_line']}"


def test_pick_line_matches_the_pick_side_of_the_line_cell():
    rl = nd.runline_history_frame(SIMPLE)
    for _, r in rl.iterrows():
        want = float(r["line"]) if r["pick"] == r["home"] \
            else float(r["_away_line"])
        assert float(r["_pick_line"]) == want


# --- Push resolution (the 3-way convention) --------------------------------

def test_total_equal_to_the_fair_line_is_a_push():
    frame = nd.totals_history_frame(SIMPLE)
    push = frame[frame.winner == "Push"]
    assert len(push) == 1
    assert pd.isna(push.iloc[0]["correct"])


def test_margin_equal_to_the_threshold_is_a_spread_push():
    rl = nd.runline_history_frame(SIMPLE)
    push = rl[rl.winner == "Push"]
    assert len(push) == 1          # PHX margin 2 vs T=2
    stats = nd.history_win_rate(rl)
    assert stats["n_games"] == 3 and stats["n_pushes"] == 1


def test_pushes_are_excluded_from_both_sides_of_the_win_rate():
    stats = nd.history_win_rate(nd.totals_history_frame(SIMPLE))
    assert stats["n_games"] == 3 and stats["n_pushes"] == 1
    # BOS Over hit (230 > 225), DEN Under hit (210 < 225), MIA pushed out;
    # PHX is a totals LOSS (228 < 230), so 2 of 3 decided.
    assert stats["win_rate"] == round(2 / 3, 6)


def test_result_cell_renders_a_dash_for_a_push():
    assert page._result_cell(np.nan) == "<td>—</td>"
    assert page._result_cell(True).count("✓") == 1
    assert page._result_cell(False).count("✗") == 1


# --- Threshold semantics ---------------------------------------------------

def test_threshold_columns_price_threshold_probabilities():
    """P(home cover) is P(margin > T): the fixture's PHX row has T=2 and
    margin=2, so it is a push, NOT a cover."""
    rl = nd.runline_history_frame(SIMPLE)
    phx = rl[rl.home == "PHX"].iloc[0]
    assert phx["winner"] == "Push"


def test_minus_half_never_pushes():
    """margin == -0.5 is impossible with integer NBA margins."""
    rows = []
    for i, (hs, a_s) in enumerate([(110, 105), (105, 110)]):
        r = {"home_team": "A", "away_team": "B", "home_score": hs,
             "away_score": a_s, "total": hs + a_s, "margin": hs - a_s,
             "p_home_win_derived": 0.9 if hs > a_s else 0.1,
             "fair_total": 999.0, "fair_spread": 999.0}
        r["p_home_cover_m0_5"] = 0.75 if hs > a_s else 0.2
        r["p_away_cover_m0_5"] = 1.0 - r["p_home_cover_m0_5"]
        rows.append(r)
    frame = pd.DataFrame(rows)
    card = nd.runline_monitor_stats(frame, 0.5)
    assert card["n"] == 2
    assert card["n_pushes"] == 0


# --- Bucket calibration + charts (MLB's contracts) -------------------------

def test_bucket_calibration_keeps_empty_bins_and_uses_the_v_win_rate():
    pred = np.array([0.9, 0.2, 0.55])
    event = np.array([1.0, 0.0, 0.0])
    bins, pp, po, pwr, pece, pbrier, pauc = nd._bucket_calibration(
        pred, event, [0.0, 50.0, 100.0], ["0-50", "50-100"])
    # 0-50 bin holds the 0.2 row: under-pick CORRECT (y=0) -> win 1.0
    assert bins[0]["count"] == 1 and bins[0]["win_rate"] == 1.0
    # 50-100 bin holds 0.9 (over, y=1 -> correct) and 0.55 (over, y=0 ->
    # wrong) -> 1/2
    assert bins[1]["count"] == 2 and bins[1]["win_rate"] == 0.5
    # pooled auc runs over ALL pairs (two classes present)
    assert pauc is not None


def test_low_n_bins_are_flagged_and_suppressed_from_curve_points():
    pred = np.array([0.9] * 5 + [0.4] * 40)
    event = np.array([1.0] * 5 + [0.0] * 40)
    bins, *_ = nd._bucket_calibration(pred, event, [0, 50, 100],
                                      ["0-50", "50-100"])
    five = [b for b in bins if b["count"] == 5][0]
    forty = [b for b in bins if b["count"] == 40][0]
    assert five["low_n"] is True and forty["low_n"] is False
    table = {"bins": bins, "pooled_pred": 0.5, "pooled_observed": 0.1}
    stack = nd._gtl_line_points(table)          # default: low-n dropped
    assert len(stack) == 2                      # only the 40-row, 2 series
    stack_all = nd._gtl_line_points(table, curve_bins=bins)
    assert len(stack_all) == 4                  # curve frame: all plotted


def test_gtl_table_frame_carries_the_pooled_total_row():
    table = {"bins": [{"bin": "40-41", "count": 10, "mean_pred": 0.5,
                       "observed": 0.5, "win_rate": 0.5, "auc": None,
                       "ece": 0.0, "brier": 0.25, "share_pct": 100.0}],
             "pooled_pred": 0.5, "pooled_observed": 0.5,
             "pooled_auc": 0.5, "pooled_ece": 0.0, "pooled_brier": 0.25}
    frame = nd._gtl_table_frame(table)
    assert list(frame["bucket"]) == ["40-41", "Total"]
    assert frame.iloc[-1]["% of Total"] == 100.0
    assert list(frame.columns)[:2] == ["bucket", "count"]


def test_chart_game_total_curve_returns_chart_and_table():
    table = {"bins": [{"bin": "50-51", "bin_center": 0.505, "count": 40,
                       "mean_pred": 0.505, "observed": 0.5, "win_rate": 0.5,
                       "auc": None, "ece": 0.005, "brier": 0.25,
                       "low_n": False, "share_pct": 100.0}],
             "pooled_pred": 0.505, "pooled_observed": 0.5}
    built = nd.chart_game_total_curve(table, "t")
    assert built["table"].iloc[-1]["bucket"] == "Total"
    assert built["chart"] is not None


# --- Distribution + pairs --------------------------------------------------

def test_distribution_modeled_series_is_the_artifact_pmf_mean():
    frame = SIMPLE.copy()
    for k in range(180, 281):
        frame[f"p_push_total_{k}"] = 0.005
    frame[f"p_push_total_225"] = 0.5
    dist = nd.total_distribution(frame)
    assert dist["warning"] is None
    modeled = np.array(dist["modeled"])
    assert abs(modeled[TOTAL := 225 - 180] - 0.5) < 1e-9
    assert dist["callouts"]["P(total<=190)"]["modeled"] > 0


def test_relativized_pairs_drop_pushes():
    frame = SIMPLE.head(1).copy()          # BOS: total 230, fair line 225
    stripped = frame[[c for c in frame.columns if c not in ("mu_h", "mu_a")]]
    assert len(nd.relativized_pairs(stripped)) == 0
    pairs = nd.relativized_pairs(frame, offsets=[5.0])   # line 230 -> push
    assert len(pairs) == 0
    pairs = nd.relativized_pairs(frame, offsets=[4.0])   # line 229 -> over
    assert len(pairs) == 1 and pairs.iloc[0]["y"] == 1.0


def test_pooled_lines_are_four_fixed_lines():
    fpairs = nd.fixed_line_pairs(SIMPLE, (220, 225, 230, 240))
    # each priced game contributes a pair at each line where the columns
    # exist; pushes dropped (MIA 225 at line 225)
    assert set(fpairs["line"].unique()) <= {220, 225, 230, 240}


# --- History / card cross-checks -------------------------------------------

def test_spread_card_agrees_with_its_own_history_frame():
    rl = nd.runline_history_frame(SIMPLE)
    stats = nd.history_win_rate(rl)
    card = nd.winner_cards(SIMPLE)["run_line"]
    assert card["n"] == stats["n_games"]
    assert card["win_rate"] == stats["win_rate"]


def test_totals_card_agrees_with_its_own_history_frame():
    frame = nd.totals_history_frame(SIMPLE)
    stats = nd.history_win_rate(frame)
    card = nd.totals_monitor_stats(SIMPLE)
    assert card["n"] == stats["n_games"]
    assert card["n_pushes"] == stats["n_pushes"]


def test_card_win_rate_is_conditional_accuracy_not_a_joint_rate():
    y = np.array([1.0, 0.0, 0.0])
    p = np.array([0.9, 0.4, 0.1])
    metrics = nd.binary_card_metrics(y, p)
    assert metrics["win_rate"] == 1.0        # every lean was right
    joint = round(float(((p > 0.5) & (y == 1.0)).mean()), 6)
    assert joint == round(1 / 3, 6)
    assert metrics["win_rate"] != joint


def test_rows_at_exactly_a_half_probability_make_no_pick():
    y = np.array([1.0, 0.0, 1.0])
    p = np.array([0.9, 0.1, 0.5])
    assert nd.binary_card_metrics(y, p)["win_rate"] == 1.0


def test_confidence_threshold_is_a_nested_subset():
    low = nd.totals_monitor_stats(SIMPLE, min_pct=50)
    high = nd.totals_monitor_stats(SIMPLE, min_pct=54)
    assert high["n"] <= low["n"]


def test_pooled_win_rate_lies_between_its_two_sides():
    card = nd.totals_monitor_stats(SIMPLE)
    sides = [card["sides"][k]["win_rate"] for k in card["sides"]]
    assert min(sides) <= card["win_rate"] <= max(sides)


def test_favorite_side_follows_the_moneyline():
    """home_fav -> cover read at threshold -m; away fav -> +m."""
    rows = []
    for i, (hs, a_s, ml) in enumerate([(120, 100, 0.9), (100, 120, 0.1)]):
        r = {"home_team": "A", "away_team": "B", "home_score": hs,
             "away_score": a_s, "total": hs + a_s, "margin": hs - a_s,
             "p_home_win_derived": ml, "fair_total": 999.0,
             "fair_spread": 999.0}
        r["p_home_cover_m5"] = 0.8 if hs > a_s else 0.2
        r["p_away_cover_m5"] = 1.0 - r["p_home_cover_m5"]
        rows.append(r)
    frame = pd.DataFrame(rows)
    card = nd.runline_monitor_stats(frame, 5.0)
    assert card["n"] == 2
    assert card["n_pushes"] == 0
    assert card["win_rate"] == 1.0           # both favorites covered


# --- Calibration map -------------------------------------------------------

def test_absent_calibration_map_yields_no_calibrated_ece():
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
        assert card["n"] == 0 and card["win_rate"] is None


def test_history_frames_are_empty_not_absent_without_rows():
    for frame in (nd.totals_history_frame(pd.DataFrame()),
                  nd.runline_history_frame(pd.DataFrame())):
        assert frame.empty
        assert list(frame.columns)[:6] == list(nd._HISTORY_COLUMNS)[:6]


# --- The structural parity contract ---------------------------------------

def test_page_has_every_mlb_section_tab_and_expander():
    from compare_totals_dashboards import (REQUIRED_EXPANDERS,
                                           REQUIRED_SECTIONS, REQUIRED_TABS,
                                           check, render)
    failures = check("nba", render("nba"))
    assert failures == [], failures
    assert len(REQUIRED_TABS) == 5 and len(REQUIRED_EXPANDERS) == 3
    for section in REQUIRED_SECTIONS:
        assert section


def test_both_sports_satisfy_the_same_contract():
    from compare_totals_dashboards import check, render
    assert check("mlb", render("mlb")) == []
    assert check("nba", render("nba")) == []


def test_nba_has_mlbs_interactive_line_selectboxes():
    """The redo's whole point: tabs 4-5 render MLB's Line selectboxes."""
    from compare_totals_dashboards import render
    lines = render("nba")
    labels = [l.split(" ::", 1)[1] for l in lines if l.startswith("selectbox")]
    assert any("Line (All = own fair line)" in l for l in labels)
    assert any("Line (All = own fair run line)" in l for l in labels)


def test_drift_and_coverage_report_missing_artifacts():
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


# --- Run-engine drift/coverage parity with MLB (2026-10-08) ----------------
#
# The two panels must mirror MLB's markets page column for column (PSI ADJ.,
# SHIFT SE, MODEL WEIGHT), the weights must come from the DISTRIBUTION model
# shipped in the drift CSV — never the binary moneyline monitor — and the
# coverage table must show every feature-window pair (the old 12-row
# healthy-tail cap is gone, as on MLB).


def _render_markdown(script_body: str) -> str:
    from streamlit.testing.v1 import AppTest
    script = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import pandas as pd\n"
        "import nba_market_diagnostics as nd\n" % str(FRONTEND)
    ) + script_body
    app = AppTest.from_string(script)
    app.run()
    assert not app.exception, [str(e.value) for e in app.exception]
    return "\n".join(str(m.value) for m in app.markdown)


_DRIFT_ROWS = (
    "[{'feature': 'elo_diff', 'current_mean': 1.0, 'baseline_mean': 0.9,"
    " 'psi': 0.123, 'psi_adjusted': 0.023, 'shift_se': 0.01,"
    " 'status': 'OK', 'weight_pct': 2.27882,"
    " 'n_baseline': 220, 'n_current': 31},"
    " {'feature': 'pace_diff', 'current_mean': 2.0, 'baseline_mean': 2.1,"
    " 'psi': 0.05, 'psi_adjusted': 0.0, 'shift_se': 0.02,"
    " 'status': 'WARN', 'weight_pct': 0.74,"
    " 'n_baseline': 220, 'n_current': 31},"
    " {'feature': 'rest_days_diff', 'current_mean': 0.5,"
    " 'baseline_mean': 0.4, 'psi': float('nan'),"
    " 'psi_adjusted': float('nan'), 'shift_se': 0.0,"
    " 'status': 'INSUFFICIENT', 'weight_pct': float('nan'),"
    " 'n_baseline': 220, 'n_current': 31}]"
)


def test_drift_table_carries_mlbs_decision_and_weight_columns():
    html = _render_markdown(f"nd.render_run_engine_drift(pd.DataFrame({_DRIFT_ROWS}))\n")
    # the exact header of MLB's drift table
    for col in ("<th>PSI ADJ.</th>", "<th>SHIFT SE</th>",
                "<th>MODEL WEIGHT</th>"):
        assert col in html, f"missing {col}"
    # weights are the artifact's own values, formatted by the shared helper
    assert "2.28%" in html and "0.74%" in html
    # a NaN weight renders as an em-dash, never 'nan%'
    assert "nan%" not in html
    # the caption describes the distribution model and the location gate
    assert "location gate" in html
    assert "summing to 100%" in html
    assert "distribution" in html and "moneyline monitor ships" not in html


def test_drift_table_omits_the_weight_column_when_the_artifact_has_none():
    html = _render_markdown(
        "rows = [dict(r, weight_pct=float('nan')) for r in "
        f"{_DRIFT_ROWS}]\n"
        "nd.render_run_engine_drift(pd.DataFrame(rows))\n")
    assert "<th>MODEL WEIGHT</th>" not in html
    assert "<th>PSI ADJ.</th>" in html  # decision columns survive


def test_drift_weight_seam_overrides_the_csv_and_never_fabricates():
    records = [{"feature": "a", "weight_pct": 2.5},
               {"feature": "b", "weight_pct": float("nan")},
               {"feature": "c"}]
    # row's own weight when no seam; seam wins when given; absent -> None
    assert nd._run_engine_weight_pcts(records, None) == [2.5, None, None]
    assert nd._run_engine_weight_pcts(records, {"a": 9.0, "c": 1.5}) == \
        [9.0, None, 1.5]


def test_coverage_renders_every_feature_window_pair_without_truncation():
    rows = [{"feature": f"f{i:02d}", "window": "current", "n_games": 30,
             "pct_measured": 100.0, "pct_nonnull": 100.0,
             "n_default_zero": 0, "status": "OK"} for i in range(40)]
    html = _render_markdown(
        "nd.render_run_engine_coverage(pd.DataFrame(%r))\n" % rows)
    # all 40 healthy rows visible — the old cap hid all but 12
    assert html.count("fb-status-pill") == 40
    assert "every" in html and "stays visible" in html
    assert "hidden" not in html


def test_coverage_ships_mlbs_header_and_worst_first_caption():
    rows = [{"feature": "z_low", "window": "current", "n_games": 30,
             "pct_measured": 10.0, "pct_nonnull": 10.0,
             "n_default_zero": 0, "status": "STARVED"},
            {"feature": "a_ok", "window": "baseline", "n_games": 30,
             "pct_measured": 100.0, "pct_nonnull": 100.0,
             "n_default_zero": 2, "status": "OK"}]
    html = _render_markdown(
        "nd.render_run_engine_coverage(pd.DataFrame(%r))\n" % rows)
    for col in ("<th>FEATURE</th>", "<th>WINDOW</th>", "<th>GAMES</th>",
                "<th>% MEASURED</th>", "<th>% NON-NULL</th>",
                "<th>STATUS</th>"):
        assert col in html
    # worst-first: the starved row precedes the healthy one
    assert html.index("z_low") < html.index("a_ok")
    assert "worst-first" in html
    assert "default-zero" in html
