"""Regression tests for the NBA Totals & Run Lines page (post-redo).

These lock the structural parity with MLB's markets page and the pricing
semantics that only hold because of two artifact facts:

  1. The spread columns are THRESHOLD form — ``p_home_cover_{L}`` =
     P(margin > L) — so a displayed home spread of -5 is threshold +5 and
     the two sides of a LINE cell must ALWAYS carry opposite signs.
  2. Fair lines are computed here by grid argmin (the artifact's own 50/50
     point); the stored ``fair_total`` / ``fair_spread`` columns are never
     trusted, because ``fair_spread`` measurably disagrees with the grid.
  3. The totals argmin runs over the artifact's 0.5 ladder
     (``nd.TOTAL_HALF_GRID``), mirroring MLB's half-step ``TOTAL_GRID``:
     the history quotes book-postable lines (225.5, not just 225) and the
     pick is the FAVORED side's, so ``pick_prob`` is never below 50%.
  4. The run-line argmin rides the SAME 0.5 ladder
     (``nd.SPREAD_HALF_GRID``, MLB's ``RUN_GRID_CUT`` shape): half spread
     thresholds (±4.5) are first-class, pushes still only happen at whole
     thresholds, and a threshold-0 argmin can no longer render a fake
     ``−0`` line.
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


# --- The 0.5 ladder + a pick probability that is never below 50% -----------


def _ladder_frame() -> pd.DataFrame:
    """Two decided rows priced across the artifact's 0.5 totals ladder.

    Row 0's closest 50/50 rung is a HALF line (225.5, no push mass).
    Row 1's is an INTEGER line (226) where re-scaling the push band out
    lifts the 2-way to just under 0.5 — the case where the old builder
    printed a sub-50% Under pick (0.4949) instead of the Under side's own
    50.51%. Unlisted ladder rungs ship no column, exactly like an artifact
    that predates the half-step namespace.
    """
    specs = [
        {224: (0.60, 0.38), 224.5: (0.60, 0.40),
         225: (0.52, 0.46), 225.5: (0.52, 0.48),
         226: (0.44, 0.54), 226.5: (0.44, 0.56)},
        {224: (0.62, 0.36), 224.5: (0.62, 0.38),
         225: (0.55, 0.43), 225.5: (0.55, 0.45),
         226: (0.485, 0.495), 226.5: (0.42, 0.58),
         227: (0.40, 0.58), 227.5: (0.40, 0.60)},
    ]
    rows = []
    for i, spec in enumerate(specs):
        r = {"game_id": f"90000000{i}", "gameday": "2026-02-10",
             "kind": "oof", "decided": True,
             "home_team": "BOS", "away_team": "NYK",
             "home_score": 118, "away_score": 112,
             "total": 230, "margin": 6,
             "p_home_win_derived": 0.5,
             "fair_total": 999.0, "fair_spread": 999.0}
        for line, (po, pu) in spec.items():
            lab = nd._label(float(line))
            r[f"p_over_{lab}"] = po
            r[f"p_under_{lab}"] = pu
            r[f"p_push_total_{lab}"] = round(1.0 - po - pu, 6)
        rows.append(r)
    return pd.DataFrame(rows)


def test_fair_line_sits_on_the_half_step_ladder():
    """The history must be able to quote X.5 totals, not only integers —
    MLB's grid is a 0.5 ladder and so is the NBA artifact's."""
    assert 225.5 in nd.TOTAL_HALF_GRID and 225 in nd.TOTAL_HALF_GRID
    assert nd.TOTAL_HALF_GRID == sorted(nd.TOTAL_HALF_GRID)
    frame = nd.totals_history_frame(_ladder_frame())
    assert set(frame["line"]) == {225.5, 226.0}


def test_pick_is_the_ladder_line_closest_to_fifty():
    """Objective: the quoted over/under is the rung whose favored-side p is
    > 50% and closest to 50% — never a rung with a better 50/50 point."""
    decided = _ladder_frame()
    frame = nd.totals_history_frame(decided)
    assert len(frame) == len(decided)
    for j, row in enumerate(frame.iloc):
        best = np.inf
        for rung in nd.TOTAL_HALF_GRID:
            a, b = nd.grid_over_under_cols(rung)
            if a not in decided.columns or b not in decided.columns:
                continue
            x, y = decided[a].iloc[j], decided[b].iloc[j]
            if pd.isna(x) or pd.isna(y) or (float(x) + float(y)) <= 0:
                continue
            best = min(best, abs(float(x) / (float(x) + float(y)) - 0.5))
        # favored-side p is exactly 0.5 + the smallest 50/50 gap on the
        # ladder — the objective: closest to 50%, never below it.
        assert row["pick_prob"] >= 0.5
        assert abs((float(row["pick_prob"]) - 0.5) - best) < 1e-9


def test_pick_probability_is_never_below_fifty():
    """An Under pick must report P(under), not the raw P(over) < 0.5."""
    frame = nd.totals_history_frame(_ladder_frame())
    assert (frame["pick_prob"] >= 0.5).all()
    under = frame[frame["pick"] == "Under"].iloc[0]
    # rung 226 prices 2-way 0.494898 -> Under 50.51%, not "Under (49%)"
    assert float(under["line"]) == 226.0
    assert abs(float(under["pick_prob"]) - 0.505102) < 1e-6
    over = frame[frame["pick"] == "Over"].iloc[0]
    assert float(over["line"]) == 225.5
    assert abs(float(over["pick_prob"]) - 0.52) < 1e-9


def test_history_table_renders_half_step_lines_and_fifty_plus_odds():
    """The rendered cells, not just the frame: O/U 225.5 and a pick cell
    whose percentage reads >= 50% at one decimal (a whole percent would
    round every NBA pick to exactly 50%)."""
    import datetime as dt

    from streamlit.testing.v1 import AppTest
    frame = _ladder_frame()
    # repr() of a NaN column would inject a bare `nan` name into the
    # generated script — hand it None instead.
    payload = {col: [None if isinstance(v, float) and pd.isna(v) else v
                     for v in values]
               for col, values in frame.to_dict("list").items()}
    script = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import datetime as dt\n"
        "import pandas as pd\n"
        "import nba_markets_page as page\n"
        "frame = pd.DataFrame(%r)\n"
        "page._render_totals_history(frame, None, dt.date(2026, 2, 1), "
        "dt.date(2026, 2, 28))\n" % (str(FRONTEND), payload)
    )
    app = AppTest.from_string(script)
    app.run()
    assert not app.exception, [str(e.value) for e in app.exception]
    html = "\n".join(str(m.value) for m in app.markdown)
    assert "O/U 225.5" in html and "O/U 226.0" in html
    assert "Over 225.5 (52.0%)" in html
    assert "Under 226.0 (50.5%)" in html
    assert "(49" not in html, "a pick must never print a sub-50% probability"


# --- The run line rides the same 0.5 ladder -------------------------------


def _spread_ladder_frame() -> pd.DataFrame:
    """Two decided rows priced across half AND integer spread thresholds.

    Row 0's closest rung is a HALF threshold (4.5, no push mass): the old
    integer-only search would have quoted ±5 with a 2-way 0.533. Row 1's is
    an INTEGER threshold (2.0) whose 2-way is just above 0.5 after the push
    band folds out, and whose margin equals it — the one case that still
    pushes. Unlisted rungs ship no column, like an artifact without the
    half-step spread namespace.
    """
    specs = [
        {4.0: (0.56, 0.38), 4.5: (0.52, 0.48),
         5.0: (0.48, 0.42), 5.5: (0.44, 0.56)},
        {1.5: (0.51, 0.485), 2.0: (0.497, 0.487), 2.5: (0.49, 0.51)},
    ]
    rows = []
    for i, (spec, (hs, a_s), home, away) in enumerate(
            zip(specs, [(118, 111), (105, 103)], ["BOS", "MIA"],
                ["NYK", "CHI"])):
        r = {"game_id": f"80000000{i}", "gameday": "2026-02-10",
             "kind": "oof", "decided": True,
             "home_team": home, "away_team": away,
             "home_score": hs, "away_score": a_s,
             "total": hs + a_s, "margin": hs - a_s,
             "p_home_win_derived": 0.55,
             "fair_total": 999.0, "fair_spread": 999.0}
        for thr, (ph, pa) in spec.items():
            lab = nd._label(float(thr))
            r[f"p_home_cover_{lab}"] = ph
            r[f"p_away_cover_{lab}"] = pa
            r[f"p_push_{lab}"] = round(1.0 - ph - pa, 6)
        rows.append(r)
    return pd.DataFrame(rows)


def test_run_line_argmin_quotes_half_step_thresholds():
    """The run-line ladder is the mixed 0.5 shape: ±4.5 is quotable (the
    integer-only search priced it at ±5) and the threshold-0 argmin can no
    longer render a fake ``−0`` line."""
    assert 4.5 in nd.SPREAD_HALF_GRID and 2.0 in nd.SPREAD_HALF_GRID
    assert -20.5 in nd.SPREAD_HALF_GRID and 20.5 in nd.SPREAD_HALF_GRID
    assert nd.SPREAD_HALF_GRID == sorted(nd.SPREAD_HALF_GRID)
    rl = nd.runline_history_frame(_spread_ladder_frame())
    by_home = {r["home"]: r for _, r in rl.iterrows()}
    assert float(by_home["BOS"]["line"]) == -4.5
    assert float(by_home["BOS"]["_away_line"]) == 4.5
    assert float(by_home["MIA"]["line"]) == -2.0


def test_run_line_pick_is_the_ladder_line_closest_to_fifty():
    """Objective, re-scanned independently over the whole ladder: the
    quoted threshold is the rung whose favored-side cover is > 50% and
    closest to 50% — never a rung with a better 50/50 point."""
    decided = _spread_ladder_frame()
    rl = nd.runline_history_frame(decided)
    assert len(rl) == len(decided)
    for j, row in enumerate(rl.iloc):
        best = np.inf
        for rung in nd.SPREAD_HALF_GRID:
            hcol = f"p_home_cover_{nd._label(rung)}"
            acol = f"p_away_cover_{nd._label(rung)}"
            if hcol not in decided.columns or acol not in decided.columns:
                continue
            h, a = decided[hcol].iloc[j], decided[acol].iloc[j]
            if pd.isna(h) or pd.isna(a) or (float(h) + float(a)) <= 0:
                continue
            best = min(best, abs(float(h) / (float(h) + float(a)) - 0.5))
        assert row["pick_prob"] >= 0.5
        assert abs((float(row["pick_prob"]) - 0.5) - best) < 1e-9


def test_run_line_pushes_only_at_whole_thresholds():
    """Margin 2 vs the integer threshold 2 pushes; margin 7 vs the half
    threshold 4.5 never can."""
    rl = nd.runline_history_frame(_spread_ladder_frame())
    pushed = rl[rl.winner == "Push"]
    assert len(pushed) == 1
    assert float(pushed.iloc[0]["line"]) == -2.0
    assert pd.isna(pushed.iloc[0]["correct"])
    assert nd.history_win_rate(rl)["n_pushes"] == 1


def test_run_line_history_table_renders_half_step_lines():
    """The rendered cells, not just the frame: ±4.5 in the line cell and a
    one-decimal percentage ≥ 50% in the pick cell."""
    import datetime as dt

    from streamlit.testing.v1 import AppTest
    frame = _spread_ladder_frame()
    payload = {col: [None if isinstance(v, float) and pd.isna(v) else v
                     for v in values]
               for col, values in frame.to_dict("list").items()}
    script = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import datetime as dt\n"
        "import pandas as pd\n"
        "import nba_markets_page as page\n"
        "frame = pd.DataFrame(%r)\n"
        "page._render_runline_history(frame, None, dt.date(2026, 2, 1), "
        "dt.date(2026, 2, 28))\n" % (str(FRONTEND), payload)
    )
    app = AppTest.from_string(script)
    app.run()
    assert not app.exception, [str(e.value) for e in app.exception]
    html = "\n".join(str(m.value) for m in app.markdown)
    assert "RL BOS -4.5 / NYK +4.5" in html
    assert "BOS -4.5 (52.0%)" in html
    assert "MIA -2 (50.5%)" in html        # integer rungs keep the +g shape
    assert "-0" not in html and "(49" not in html


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

def test_distribution_models_the_shipped_pmf_without_smoothing():
    """A spike in shipped mass must remain visible, never plotted away."""
    frame = SIMPLE.copy()
    grid = pd.DataFrame({f"p_push_total_{k}": np.full(len(frame), .005)
                         for k in range(180, 281)}, index=frame.index)
    grid["p_push_total_225"] = .5
    frame = pd.concat([frame, grid], axis=1)
    dist = nd.total_distribution(frame)
    assert dist["warning"] is None
    np.testing.assert_array_equal(dist["modeled"], dist["modeled_grid"])
    assert dist["modeled"][225 - 180] == .5
    assert "no refit or smoothing" in dist["callouts"]["note"]
    without_means = nd.total_distribution(frame.drop(columns=["mu_h", "mu_a"]))
    np.testing.assert_array_equal(dist["modeled"], without_means["modeled"])


# ---------------------------------------------------------------------------
# NBA like-with-like spread tails (2026-10-16): the ±10 modeled callouts
# are the exact shipped cover+push composites over FULL support — P(≥10) =
# mean(p_home_cover_10 + p_push_10), P(≤−10) = 1 − mean(p_home_cover_m10)
# — against the full-tail observed rates (blowouts beyond ±30 count on
# both sides). Absent columns render None. The per-bin line reads traded
# pushes plus the push-only extension to ±30; anything beyond stays a gap.
# ---------------------------------------------------------------------------

def test_nba_spread_tail_callouts_use_cover_push_composites():
    margins = [-35, -10, 0, 10, 40]
    n = len(margins)
    frame = pd.DataFrame({
        "margin": margins,
        **{f"p_push_{nd._label(float(k))}": [0.02] * n
           for k in range(-30, 31)},
        "p_home_cover_10": [0.10] * n,
        "p_push_10": [0.02] * n,
        "p_home_cover_m10": [0.80] * n,
    })
    dist = nd.spread_distribution(frame)
    assert dist["warning"] is None
    c = dist["callouts"]
    # Observed full tails count the blowouts beyond ±30.
    assert c["P(margin<=-10)"]["observed"] == pytest.approx(0.4)
    assert c["P(margin>=10)"]["observed"] == pytest.approx(0.4)
    # Modeled tails are the composites — not a slice of the per-bin line.
    assert c["P(margin>=10)"]["modeled"] == pytest.approx(0.12)
    assert c["P(margin<=-10)"]["modeled"] == pytest.approx(0.20)
    # The per-bin line still reads the margin push family.
    assert dist["modeled"][dist["ks"].index(10)] == pytest.approx(0.02)
    # Absent cover/push columns degrade to None, never a crash.
    bare = pd.DataFrame({
        "margin": margins,
        **{f"p_push_{nd._label(float(k))}": [0.02] * n
           for k in range(-30, 31)},
    })
    bare_c = nd.spread_distribution(bare)["callouts"]
    assert bare_c["P(margin<=-10)"]["modeled"] is None
    assert bare_c["P(margin>=10)"]["modeled"] is None
    assert bare_c["P(margin<=-10)"]["observed"] == pytest.approx(0.4)


def test_nba_spread_reads_the_push_extension_and_gaps_beyond():
    """Current artifacts fill the modeled line to ±30; pre-extension
    artifacts keep their gaps there — while the tail composites hold in
    both vintages (never a slice of the per-bin line)."""
    n = 3
    full = pd.DataFrame({
        "margin": [0, 25, -25],
        **{f"p_push_{nd._label(float(k))}": [0.02] * n
           for k in range(-30, 31)},
        "p_home_cover_10": [0.10] * n,
        "p_push_10": [0.02] * n,
        "p_home_cover_m10": [0.80] * n,
    })
    dist = nd.spread_distribution(full)
    assert dist["warning"] is None
    assert dist["modeled"][dist["ks"].index(25)] == pytest.approx(0.02)
    assert dist["modeled"][dist["ks"].index(-25)] == pytest.approx(0.02)
    legacy = pd.DataFrame({
        "margin": [0, 25, -25],
        **{f"p_push_{nd._label(float(k))}": [0.02] * n
           for k in range(-20, 21)},
        "p_home_cover_10": [0.10] * n,
        "p_push_10": [0.02] * n,
        "p_home_cover_m10": [0.80] * n,
    })
    old = nd.spread_distribution(legacy)
    assert old["modeled"][old["ks"].index(10)] == pytest.approx(0.02)
    assert old["modeled"][old["ks"].index(25)] != \
        old["modeled"][old["ks"].index(25)]  # NaN: legacy gap kept
    assert old["callouts"]["P(margin>=10)"]["modeled"] == pytest.approx(0.12)
    assert old["callouts"]["P(margin<=-10)"]["modeled"] == pytest.approx(0.20)


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

def test_spread_card_is_the_home_lean_view_of_its_history_frame():
    """2026-10-12 HOME-side reframe: the run-line card grades the HOME
    cover at each game's own fair threshold, counting only home-lean
    non-push rows (P(home cover) > 50%) — the pick-side history-frame
    win rate no longer equals the card, so pin the home-lean identity
    directly on the SIMPLE fixture (G1 home pick won, G3 home pick lost,
    G2 away lean excluded, G4 spread push excluded)."""
    rl = nd.runline_history_frame(SIMPLE)
    picked_home = (rl["pick"] == rl["home"]).to_numpy()
    non_push = rl["correct"].notna().to_numpy()
    lean = rl[non_push & picked_home]
    card = nd.winner_cards(SIMPLE)["run_line"]
    assert len(lean) > 0
    assert card["n"] == len(lean)
    home_won = (lean["winner"] == lean["home"]).to_numpy(dtype=float)
    assert card["win_rate"] == round(float(home_won.mean()), 6)
    assert card["win_rate"] == card["actual_win_rate"]
    assert card["predicted_mean"] is not None and card["predicted_mean"] > 0.5


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


def test_home_side_card_metrics_grades_only_home_lean_rows():
    """2026-10-12 HOME-side reframe: no-lean rows (p <= 50%) make no pick
    and are dropped from ``n`` itself, and Actual/Predicted stay on the
    SAME home-lean population."""
    y = np.array([1.0, 0.0, 1.0, 1.0])
    p = np.array([0.60, 0.40, 0.50, 0.90])
    m = nd.home_side_card_metrics(y, p)
    assert m["n"] == 2                      # 0.60 and 0.90 only
    assert m["win_rate"] == 1.0
    assert m["actual_win_rate"] == m["win_rate"]
    assert m["predicted_mean"] == round((0.60 + 0.90) / 2, 6)


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


def test_challenger_cards_score_issued_probabilities_without_recalibration():
    monitor = {"fit": {"family": "two_sided_asymmetric_gaussian_split_skew"}}
    cards = nd.winner_cards(SIMPLE, monitor)
    unlabelled = nd.winner_cards(SIMPLE)
    for key, card in cards.items():
        assert card["ece_calibrated"] == card["ece_raw"]
        assert card["ece_calibrated"] is not None
        for metric in ("brier", "logloss", "auc", "predicted_mean", "win_rate"):
            assert card[metric] == unlabelled[key][metric]


def test_challenger_calibration_note_is_family_aware():
    monitor = {"fit": {"family": "two_sided_asymmetric_gaussian_split_skew"}}
    note = page._cal_note(monitor)
    assert "whole-CDF-calibrated" in note and "Monte Carlo" in note
    assert "Platt" not in note and "RAW" not in note
    assert "predates" in page._cal_note(None)


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


def test_nba_distribution_chart_shades_noise_band():
    """NBA's distribution chart mirrors MLB's: a 95% binomial band behind
    bars + modeled line, with NaN-gap bins excluded from the band."""
    import json
    import math

    dist = {"ks": [220, 221, 222], "observed": [0.3, 0.4, 0.3],
            "modeled": [0.28, 0.35, float("nan")],
            "callouts": {}, "n_games": 400, "warning": None}
    band = nd.distribution_noise_band(dist)
    assert list(band["k"]) == [220, 221]
    se = math.sqrt(0.28 * 0.72 / 400)
    assert band["lo"].iloc[0] == pytest.approx(0.28 - 1.96 * se)
    assert band["hi"].iloc[0] == pytest.approx(0.28 + 1.96 * se)
    spec = json.loads(nd.chart_distribution(dist).to_json())
    assert [layer["mark"]["type"] for layer in spec["layer"]] == ["area", "bar", "line"]


# ---------------------------------------------------------------------------
# Home / Away distribution toggles (2026-10-16): the per-side points law.
#
# The Distribution view toggles Total / Spread / Home / Away. The Home and
# Away views plot the observed per-side finals PLUS the model's OWN
# per-side Gaussian as the modeled line: the artifact ships no per-team
# score PMF column, but the bivariate scoring core the backend prices from
# has a plain Gaussian marginal per side, so the line is DERIVED from
# shipped parameters only (side mean split from the calibrated
# distribution_mean_total / distribution_mean_margin, variance = the
# monitor's fit.variance factor × that mean). Without a usable fit block it
# falls back to the all-gap line — never borrowed from another push
# namespace, never refitted on outcomes.
# ---------------------------------------------------------------------------

# The monitor's shipped fit.variance block, verbatim in shape.
FIT_VARIANCE = {"home_variance": 1.25, "away_variance": 1.35,
                "correlation": 0.25, "total_skew": 0.024,
                "margin_skew": -0.012}


def _calibrated(frame: pd.DataFrame) -> pd.DataFrame:
    """SIMPLE plus the calibrated mean columns the artifact actually ships.

    distribution_mean_total / distribution_mean_margin are the post-
    calibration MC means, so the per-side mean split out of them
    (home = (total + margin) / 2) IS the calibrated model's side mean.
    """
    out = frame.copy()
    out["distribution_mean_total"] = out["total"].astype(float)
    out["distribution_mean_margin"] = out["margin"].astype(float)
    return out


CALIBRATED = _calibrated(SIMPLE)
# home = (230+10)/2, (210+0)/2, (225-1)/2, (228+2)/2 -> 120, 105, 112, 115
# away = (230-10)/2, (210-0)/2, (225+1)/2, (228-2)/2 -> 110, 105, 113, 113
CAL_HOME_MEANS = [120.0, 105.0, 112.0, 115.0]
CAL_AWAY_MEANS = [110.0, 105.0, 113.0, 113.0]


def test_team_score_distribution_grid_and_observed_stay_exactly_as_before():
    dist = nd.team_score_distribution(SIMPLE, "home")
    assert dist["warning"] is None
    # Home points settle 105..120 — the span starts at the lowest points
    # total that settled (NBA sides never approach 0: no 0..104 padding).
    assert dist["ks"][0] == 105 and dist["ks"][-1] == 120
    assert dist["observed"][120 - 105] == pytest.approx(0.25)
    c = dist["callouts"]
    assert c["mean"]["observed"] == pytest.approx(113.0)  # (120+105+112+115)/4
    assert c["P(home == 0)"]["observed"] == pytest.approx(0.0)
    # margins are +10, 0, -1, +2 -> home wins 2 of 4 (the tie is no win)
    assert c["P(home wins)"]["observed"] == pytest.approx(0.5)
    # The Away toggle flips the side: away points settle 105..113.
    away = nd.team_score_distribution(SIMPLE, "away")
    assert away["ks"][0] == 105 and away["ks"][-1] == 113
    assert away["callouts"]["P(away wins)"]["observed"] == pytest.approx(0.25)
    # A missing score column warns loudly, never fabricates.
    bare = pd.DataFrame({"home_score": [110, 100]})
    assert "away_score" in nd.team_score_distribution(bare, "home")["warning"]
    with pytest.raises(ValueError):
        nd.team_score_distribution(SIMPLE, "margin")


def test_team_score_distribution_keeps_the_all_gap_fallback():
    """No usable fit.variance block -> the modeled line stays the all-gap
    line it always was, with the disclosure note and no band."""
    for variance in (None, {}, {"correlation": 0.25},          # missing block
                     {"home_variance": 0.0},                    # unusable
                     {"home_variance": float("nan")}):
        dist = nd.team_score_distribution(CALIBRATED, "home",
                                          variance=variance)
        assert all(v != v for v in dist["modeled"]), variance
        c = dist["callouts"]
        assert c["mean"]["modeled"] is None
        assert c["P(home == 0)"]["modeled"] is None
        assert c["P(home wins)"]["modeled"] is None
        assert c["off_grid"]["modeled"] is None
        assert "never derived or refitted" in c["note"]
        assert nd.distribution_noise_band(dist).empty


def test_team_score_distribution_draws_the_models_own_side_gaussian():
    """With the monitor's shipped fit block the orange line is the model's
    own per-side Gaussian: side mean = the calibrated split, variance =
    the shipped factor × that mean, lattice mass NOT renormalized."""
    dist = nd.team_score_distribution(CALIBRATED, "home",
                                      variance=FIT_VARIANCE)
    assert dist["warning"] is None
    line = dist["modeled"]
    assert len(line) == len(dist["ks"])
    assert all(np.isfinite(line)) and all(v >= 0. for v in line)
    c = dist["callouts"]
    # side mean = the shipped calibrated split (total + margin) / 2
    assert c["mean"]["modeled"] == pytest.approx(np.mean(CAL_HOME_MEANS), abs=1e-9)
    # P(side wins) stays the EXACT shipped margin authority, not a
    # Gaussian tail re-derived here.
    assert c["P(home wins)"]["modeled"] == pytest.approx(0.55)
    # A points-scale side never puts mass below 0, and the plotted grid
    # window is disclosed rather than silently summed over.
    assert c["off_grid"]["modeled"] == pytest.approx(
        1.0 - sum(line), abs=1e-3)
    assert "never refitted" in c["note"]
    # the line peaks at the bin nearest the modeled mean, not at an edge
    assert dist["ks"][int(np.argmax(line))] == pytest.approx(
        round(c["mean"]["modeled"]), abs=1)


def test_side_line_means_sum_back_to_the_calibrated_total_mean():
    """home + away modeled means == the artifact's own calibrated total
    mean — the two side views are the same model as the Total view, split.
    This is why the lattice mass is not renormalized at 0."""
    home = nd.team_score_distribution(CALIBRATED, "home",
                                      variance=FIT_VARIANCE)
    away = nd.team_score_distribution(CALIBRATED, "away",
                                      variance=FIT_VARIANCE)
    assert (home["callouts"]["mean"]["modeled"]
            + away["callouts"]["mean"]["modeled"]) == pytest.approx(
                float(CALIBRATED["distribution_mean_total"].mean()), abs=2e-3)
    assert away["callouts"]["mean"]["modeled"] == pytest.approx(
        np.mean(CAL_AWAY_MEANS), abs=1e-9)
    # away picks up the away_variance factor, not the home one
    assert away["callouts"]["mean"]["modeled"] != home["callouts"]["mean"]["modeled"]


def test_side_line_is_derived_never_fitted_on_outcomes():
    """Changing the finals must NOT move the modeled line: it comes from
    shipped parameters, so it cannot be a refit on outcomes."""
    base = nd.team_score_distribution(CALIBRATED, "home",
                                      variance=FIT_VARIANCE)
    flipped = CALIBRATED.copy()
    flipped["home_score"] = [40, 40, 40, 40]      # absurd blowout finals
    flipped["away_score"] = [40, 40, 40, 40]
    moved = nd.team_score_distribution(flipped, "home", variance=FIT_VARIANCE)
    assert (moved["callouts"]["mean"]["modeled"]
            == base["callouts"]["mean"]["modeled"])
    shared = dict(zip(base["ks"], base["modeled"]))
    for k, p in zip(moved["ks"], moved["modeled"]):
        if k in shared:
            assert p == pytest.approx(shared[k], abs=1e-12), k
    # The shipped variance factor IS the width knob: a wider fit spreads
    # the same mass without moving the mean.
    wide = nd.team_score_distribution(CALIBRATED, "home",
                                      variance={**FIT_VARIANCE,
                                                "home_variance": 5.0})
    assert (wide["callouts"]["mean"]["modeled"]
            == pytest.approx(base["callouts"]["mean"]["modeled"]))
    assert max(wide["modeled"]) < max(base["modeled"])


def test_team_score_distribution_renders_through_the_chart_grammar():
    """The gap-only line layers as bars + line with no band; a real line
    adds the 95% band, so the Home/Away chart renders exactly like the
    Total/Spread charts in both states."""
    import json

    gap = nd.team_score_distribution(SIMPLE, "away")
    spec = json.loads(nd.chart_distribution(
        gap, x_title="Away team points scored",
        y_title="P(away points = k)").to_json())
    assert [layer["mark"]["type"] for layer in spec["layer"]] == ["bar", "line"]
    assert nd.distribution_noise_band(gap).empty   # no modeled line, no band

    lined = nd.team_score_distribution(CALIBRATED, "away",
                                       variance=FIT_VARIANCE)
    spec = json.loads(nd.chart_distribution(
        lined, x_title="Away team points scored",
        y_title="P(away points = k)").to_json())
    assert [layer["mark"]["type"] for layer in spec["layer"]] == \
        ["area", "bar", "line"]
    band = nd.distribution_noise_band(lined)
    assert len(band) == len(lined["ks"])
    assert ((band["hi"] > band["lo"]).all())
