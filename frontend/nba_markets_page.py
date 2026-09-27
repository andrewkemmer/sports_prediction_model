"""NBA Totals & Run Lines — structural mirror of the MLB markets page.

Section order, headings, diagnostics tabs (with the interactive Line
selectboxes on tabs 4-5), the layered calibration charts, the two fb-box
history tables with date pickers and a side filter, and the Monitor section
all follow ``markets.py`` (MLB is the reference implementation). Only the
sport-specific wording and the priced grids differ. The render comparison in
``compare_totals_dashboards.py`` renders both pages through AppTest and holds
the NBA page to the parity contract.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import nba_market_diagnostics as nd
import utils

DIAG_TABS = nd.DIAG_TABS
HISTORY_HEADERS = ("DATE</th><th>MATCHUP</th><th>SCORE (A–H)</th><th>LINE</th>"
                   "</th><th>MODEL PICK</th><th>WINNER</th><th>RESULT</th>")
FALLBACK_DATE = "20260924"
PRIMARY, RED = utils.PRIMARY, utils.RED


def _fmt(value, digits: int = 3) -> str:
    value = nd._num(value)
    return "—" if value is None else f"{value:.{digits}f}"


def _fmt_pct(value, digits: int = 1) -> str:
    value = nd._num(value)
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def _date_str(raw) -> str:
    d = pd.to_datetime(raw, errors="coerce")
    return d.strftime("%b %d, %Y") if pd.notna(d) else "—"


def _score_cell(row) -> str:
    h, a = row.get("home_score"), row.get("away_score")
    if pd.notna(h) and pd.notna(a):
        return f"{int(a)}–{int(h)}"
    return "—"


def _result_cell(correct) -> str:
    """✓/✗ RESULT cell, '—' for pushes (neither wins nor loses)."""
    if correct is None or (isinstance(correct, float) and pd.isna(correct)):
        return "<td>—</td>"
    if bool(correct):
        return f"<td style='color:{PRIMARY};font-weight:700;'>✓</td>"
    return f"<td style='color:{RED};font-weight:700;'>✗</td>"


def _history_box(rows_html: list[str]) -> None:
    """The MLB fb-box scroll container (byte-identical markup)."""
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <div style="max-height:480px;overflow-y:auto;">
            <table class="fb-table">
              <thead><tr><th>{HISTORY_HEADERS}</th></tr></thead>
              <tbody>{''.join(rows_html)}</tbody>
            </table>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _decided_rows(frame):
    return nd.decided_rows(frame)


# --- Diagnostics tabs (the same five as MLB, same renderers) ---------------


def _render_distribution_tab(decided):
    """Tab 1 — observed bars vs modeled mean marginal, with tail callouts."""
    dist = nd.total_distribution(decided)
    if dist["warning"]:
        st.warning(dist["warning"])
        return
    utils.show_chart(nd.chart_distribution(dist))
    c = dist["callouts"]
    st.caption(
        f"Observed bars vs the model's own mean P(total=k) grid pmf over "
        f"{dist['n_games']:,} decided games · P(total≤190): observed "
        f"{c['P(total<=190)']['observed']:.3f} / modeled "
        f"{c['P(total<=190)']['modeled']:.3f} · P(total≥270): observed "
        f"{c['P(total>=270)']['observed']:.3f} / modeled "
        f"{c['P(total>=270)']['modeled']:.3f}. The modeled series is read "
        "from the artifact (the NBA monitor ships no per-game dispersion "
        "parameters) — this is the totals law."
    )


def _render_relativized_tab(decided):
    """Tab 2 — each game priced at its own expected total ± offset."""
    pairs = nd.relativized_pairs(decided)
    curve = nd.calibration_curve(pairs)
    if curve["warning"]:
        st.warning(curve["warning"])
        return
    utils.show_chart(nd.chart_calibration(
        curve, "Relativized offsets −10 … +10"))
    xs = [b["mean_pred"] for b in curve["bins"]]
    st.caption(
        f"Each game priced at ITS own expected-total ± offset "
        f"({', '.join(f'{o:+g}' for o in nd.OFFSET_EDGES)}), snapped to the "
        f"integer totals grid. {curve['n_pairs']:,} pairs, {len(xs)} valid "
        f"bins (≥30 each, {curve['n_dropped_bins']} dropped) · predicted "
        f"range {min(xs):.2f}–{max(xs):.2f} — the spread is the point; the "
        "dashed diagonal is perfect calibration. Pushes (total exactly on "
        "the shifted line) are excluded — the integer grid can push, which "
        "MLB's half-step grid never does."
    )
    st.warning(
        "**Known limitation — grid resolution:** offsets land on the "
        "integer totals grid, so a ±0.5-point offset is invisible to the "
        "pmf; the innermost non-degenerate offsets are ±1. This is coarser "
        "than MLB's half-step grid and is a property of the artifact, not "
        "the renderer."
    )


def _render_pooled_tab(decided):
    """Tab 3 — all games pooled across four fixed lines."""
    fpairs = nd.fixed_line_pairs(decided, nd.POOLED_LINES)
    fcurve = nd.calibration_curve(fpairs)
    if fcurve["warning"]:
        st.warning(fcurve["warning"])
        return
    utils.show_chart(nd.chart_calibration(
        fcurve, "Games pooled across " +
        " / ".join(f"{l:g}" for l in nd.POOLED_LINES)))
    st.caption(
        f"**How to read this:** every game is priced at FOUR lines "
        f"({', '.join(str(l) for l in nd.POOLED_LINES)}) so the predictions "
        "spread across the probability range — a single game contributes 4 "
        "pairs. X = the model's predicted P(over); Y = how often the over "
        "actually hit among the pairs in that bin. The dashed diagonal is "
        "perfect calibration — points on it mean the model's probabilities "
        "are honest at every confidence level, not just near 50%. "
        f"({fcurve['n_pairs']:,} pairs, but NOT independent: each game "
        f"appears 4×, so the effective sample is the {fcurve['n_pairs'] // 4:,} "
        "games themselves.) Pushes on whole-number lines are excluded."
    )


def _pooled_stat(value, fmt: str) -> str:
    """Format a pooled stat, or 'n/a' when the subset is degenerate.

    A fixed-line subset whose predictions are all one class has no rank
    AUC (and, with no pairs, no ECE/Brier) — that is an honest property of
    the data, not a page error, so the caption says so rather than
    crashing.
    """
    if value is None:
        return "n/a"
    return format(value, fmt)


def _render_game_total_tab(decided):
    """Tab 4 — own fair line ('All') or one fixed line, MLB's layout."""
    _gl_sel = st.selectbox(
        "Line (All = own fair line)",
        ["All"] + [str(l) for l in nd.TOTAL_GRID],
        index=1 + nd.TOTAL_GRID.index(225), key="nba_diag_game_total_line")
    _gl_line = None if _gl_sel == "All" else float(_gl_sel)
    glc = nd.game_total_calibration(decided, _gl_line)
    if glc["warning"]:
        st.warning(glc["warning"])
        return
    if _gl_line is None:
        _gl_title = "Calibration Curve — Over (All = own fair line)"
    else:
        _gl_title = f"Calibration Curve — Over {_gl_line:g}"
    built = nd.chart_game_total_curve(
        glc, _gl_title, curve_bins=glc.get("curve_bins"),
        x_tick_values=nd.X_1PCT_TICKS, show_win_rate=False,
        x_label="Mean Predicted", series_label="Mean Actual")
    utils.show_chart(built["chart"])
    st.table(built["table"])
    priced_txt = ("decided games priced at their own fair lines"
                  if _gl_line is None else
                  f"decided games priced at line {_gl_line:g}")
    st.caption(
        f"{glc['n_games']:,} {priced_txt} · bar heights = games priced in "
        f"that predicted-P(over) band (LEFT 'Games' axis) · observed curve "
        f"(RIGHT '%' axis) = how often those games went over, on the 2-way "
        f"no-push basis · {glc['n_pushes']:,} pushes excluded "
        f"({glc['push_rate']:.1%}, whole lines only, neither wins nor "
        f"losses) · % of Total = count_bin / count_total × 100 · pooled "
        f"predicted {_pooled_stat(glc['pooled_pred'], '.2f')} vs pooled observed "
        f"{_pooled_stat(glc['pooled_observed'], '.2f')} · pooled win rate "
        f"{_pooled_stat(glc['pooled_winrate'], '.1%')} · pooled ECE {_pooled_stat(glc['pooled_ece'], '.3f')} "
        f"· pooled Brier {_pooled_stat(glc['pooled_brier'], '.3f')} · pooled AUC "
        f"{_pooled_stat(glc['pooled_auc'], '.3f')} (rank AUC over ALL decided no-push games "
        "at this line). Fair lines are the grid argmin of |2-way P(over) − "
        "0.5| — the artifact's own 50/50 point — ties picking the lower "
        "line; the stored fair_total column is not used because it does not "
        "agree with the artifact's own grid (measured). "
        + ("ALL: each game priced at ITS OWN fair line — predicted hugs 50% "
           "by construction, so the band sits in the 40–60 buckets."
           if _gl_line is None else
           f"FIXED LINE {_gl_line:g}: all games at one line — the predicted "
           "spread IS the calibration surface; 5-pt bins line the 0–1 axis.")
        + " The dashed diagonal is perfect calibration. Gray bars mark "
        "buckets with n < 30 (low sample — not reliable calibration "
        "evidence). The last table row is the pooled Total (share 100%, the "
        "amber diamond on the chart)."
    )


def _render_spread_tab(decided):
    """Tab 5 — favorite-side spread calibration, MLB's Run Lines layout."""
    _rl_sel = st.selectbox(
        "Line (All = own fair run line)",
        ["All"] + [str(l) for l in nd.SPREAD_LINE_CHOICES],
        index=1 + nd.SPREAD_LINE_CHOICES.index(-2.0), key="nba_diag_spread")
    _rl_line = None if _rl_sel == "All" else float(_rl_sel)
    rlc = nd.run_line_calibration(decided, _rl_line)
    if rlc["warning"]:
        st.warning(rlc["warning"])
        return
    if _rl_line is None:
        _rl_title = "Calibration Curve — Favorite (All = own fair run line)"
    else:
        _rl_title = f"Calibration Curve — Favorite {_rl_line:.1f}"
    built = nd.chart_game_total_curve(
        rlc, _rl_title, curve_bins=rlc.get("curve_bins"),
        x_tick_values=nd.X_1PCT_TICKS, show_win_rate=False,
        x_label="Mean Predicted", series_label="Mean Actual")
    utils.show_chart(built["chart"])
    st.table(built["table"])
    priced_txt = ("decided games priced at their own fair run lines"
                  if _rl_line is None else
                  f"decided games priced at run line {_rl_line:.1f}")
    st.caption(
        f"{rlc['n_games']:,} {priced_txt} · bar heights = games priced in "
        f"that predicted P(cover) band (LEFT 'Games' axis) · observed curve "
        f"(RIGHT '%' axis) = how often the favorite side covered, on the "
        f"2-way no-push basis · {rlc['n_pushes']:,} pushes excluded "
        f"({rlc['push_rate']:.1%}, whole lines only — the favorite winning "
        f"by exactly the line is neither win nor loss) · % of Total = "
        f"count_bin / count_total × 100 · pooled predicted "
        f"{_pooled_stat(rlc['pooled_pred'], '.2f')} vs pooled observed "
        f"{_pooled_stat(rlc['pooled_observed'], '.2f')} · pooled win rate "
        f"{_pooled_stat(rlc['pooled_winrate'], '.1%')} · pooled ECE {_pooled_stat(rlc['pooled_ece'], '.3f')} "
        f"· pooled Brier {_pooled_stat(rlc['pooled_brier'], '.3f')} · pooled AUC "
        f"{_pooled_stat(rlc['pooled_auc'], '.3f')} (rank AUC over ALL decided no-push games "
        "at this line). Fair lines are the grid argmin of |2-way home "
        "cover − 0.5| over integer thresholds; the favorite's cover "
        "probability is read from the threshold column on its own side "
        "(home favorite → P(margin > −m), away favorite → P(margin < m)), "
        "push folded out of both sides. "
        + ("ALL: each game priced at ITS OWN fair line — predicted hugs 50% "
           "by construction."
           if _rl_line is None else
           f"FIXED LINE {_rl_line:.1f}: all games at one line — the "
           "predicted spread IS the calibration surface; 5-pt bins.")
        + " The dashed diagonal is perfect calibration. The NBA spread grid "
        "is integer plus the ±0.5 half stop, so a whole-number line pushes "
        "where MLB's −1.5 never can. Gray bars mark buckets with n < 30. "
        "The last table row is the pooled Total (share 100%, the amber "
        "diamond on the chart)."
    )


# --- Prediction history (MLB's two fb-box tables) --------------------------


def _cal_note(monitor: dict | None) -> str:
    cards = (monitor or {}).get("calibration_cards") or {}
    if cards:
        return (" · probabilities are post-calibration σ(a·logit(p)+b)"
                " (prequential Platt, per line)")
    return (" · probabilities are RAW (no calibration map shipped in "
            "the run-engine artifact)")


def _render_totals_history(decided, monitor, start_d, end_d) -> None:
    """Game-totals history: DATE | MATCHUP | SCORE (A–H) | LINE (O/U X) |
    MODEL PICK (Over/Under X (p%)) | WINNER | RESULT — MLB's anatomy."""
    side = st.radio("Side filter", ["All", "Over", "Under"], index=0,
                    horizontal=True, key="nba_totals_history_side")
    view = nd.filter_history_by_side(
        nd.filter_history_frame(nd.totals_history_frame(decided), start_d, end_d),
        side)
    view = view.sort_values("game_date", ascending=False)
    st.markdown("#### Game Totals — Prediction History")
    if not len(view):
        st.info("No games in the selected date range / side.")
        return
    stats = nd.history_win_rate(view)
    rate = stats["win_rate"]
    rate_txt = f"{rate * 100:.1f}% picks correct" if rate is not None \
        else "no priced games"
    push_txt = (
        f" · {stats['n_pushes']:,} push(es) excluded — total == whole-"
        "number line, neither wins nor loses") if stats["n_pushes"] else ""
    side_txt = " · all sides" if side == "All" else f" · {side} picks only"
    st.caption(
        f"{stats['n_games']:,} games{side_txt} · {rate_txt} · most recent "
        f"first — scroll for older results{push_txt}{_cal_note(monitor)}"
    )
    rows = []
    for _, r in view.iterrows():
        rows.append(
            f"<tr><td>{_date_str(r['game_date'])}</td>"
            f"<td>{r['away']} @ {r['home']}</td>"
            f"<td>{_score_cell(r)}</td>"
            f"<td>O/U {r['line']:.0f}</td>"
            f"<td>{r['pick']} {r['line']:.0f} ({r['pick_prob']:.0%})</td>"
            f"<td>{r['winner']}</td>{_result_cell(r['correct'])}</tr>"
        )
    _history_box(rows)


def _render_runline_history(decided, monitor, start_d, end_d) -> None:
    """Run-line history: LINE (RL home −L / away +L) | MODEL PICK (side ±L
    (p%)) | WINNER | RESULT — MLB's anatomy, NBA's integer-push rules."""
    view = nd.filter_history_frame(
        nd.runline_history_frame(decided), start_d, end_d)
    view = view.sort_values("game_date", ascending=False)
    st.markdown("#### Run Lines — Prediction History")
    if not len(view):
        st.info("No games in the selected date range.")
        return
    stats = nd.history_win_rate(view)
    rate = stats["win_rate"]
    rate_txt = f"{rate * 100:.1f}% picks correct" if rate is not None \
        else "no priced games"
    push_txt = (
        f" · {stats['n_pushes']:,} push(es) excluded — margin == the "
        "whole-number line, neither wins nor loses") if stats["n_pushes"] else ""
    st.caption(
        f"{stats['n_games']:,} games · {rate_txt} · most recent first — "
        f"scroll for older results{push_txt} · each game priced at its OWN "
        f"fair line (grid argmin), probability 2-way re-normalized"
        f"{_cal_note(monitor)}"
    )
    rows = []
    for _, r in view.iterrows():
        # home_line = -T, away_line = +T — ALWAYS opposite signs, whatever
        # the pick. The first attempt rendered the home line on both sides
        # when the pick was home; the harness screenshot caught it.
        line_txt = (f"RL {r['home']} {float(r['line']):+.0f} / "
                    f"{r['away']} {float(r['_away_line']):+.0f}")
        pick_txt = f"{r['pick']} {float(r['_pick_line']):+.0f} ({r['pick_prob']:.0%})"
        rows.append(
            f"<tr><td>{_date_str(r['game_date'])}</td>"
            f"<td>{r['away']} @ {r['home']}</td>"
            f"<td>{_score_cell(r)}</td>"
            f"<td>{line_txt}</td>"
            f"<td>{pick_txt}</td>"
            f"<td>{r['winner']}</td>{_result_cell(r['correct'])}</tr>"
        )
    _history_box(rows)


# --- Monitor section (MLB's cards, fit panel, rolling history) -------------


def _ec_indicator(raw, cal) -> str:
    """Color-coded ECE badge: green < 0.01, yellow < 0.02, red >= 0.02."""
    val = cal if cal is not None else raw
    if val is None:
        return "⚪"
    if val < 0.01:
        return "🟢"
    if val < 0.02:
        return "🟡"
    return "🔴"


def _render_winner_cards(cards: dict) -> None:
    """The three binary winner cards — MLB's exact 7-metric layout."""
    items = [(k, label, rule) for k, label, rule in nd.WINNER_CARDS if k in cards]
    if not items:
        st.info("No winner-card data in this monitor artifact.")
        return
    cols = st.columns(len(items))
    for i, (key, label, rule) in enumerate(items):
        card = cards[key]
        badge = _ec_indicator(card.get("ece_raw"), card.get("ece_calibrated"))
        with cols[i]:
            st.markdown(f"**{label}** {badge}")
            st.caption(rule)
            awr, pm = card.get("actual_win_rate"), card.get("predicted_mean")
            compact = (f"Actual {_fmt_pct(awr)} · Predicted {_fmt_pct(pm)}"
                       if awr is not None and pm is not None
                       else "Actual — · Predicted —")
            st.markdown(f"**{compact}**")
            c1, c2 = st.columns(2)
            c1.metric("Win rate", _fmt_pct(card.get("win_rate")))
            c2.metric("AUC", _fmt(card.get("auc"), 4))
            c1, c2 = st.columns(2)
            c1.metric("Pooled ECE-cal", _fmt(card.get("ece_calibrated")))
            c2.metric("Pooled ECE-raw", _fmt(card.get("ece_raw")))
            c1, c2 = st.columns(2)
            c1.metric("Pooled Brier", _fmt(card.get("brier")))
            c2.metric("Pooled Logloss", _fmt(card.get("logloss"), 4))
            st.caption(f"n = {card.get('n', '--'):,} OOF games · Win rate is "
                       "W/(W+L): rows with no side lean (P = 50%) and "
                       "whole-line pushes are excluded from both sides.")


def _render_totals_calibration_card(decided) -> None:
    """Interactive totals card — MLB's 4-metric layout + side split."""
    st.markdown("**Totals — Over/Under calibration card**")
    c1, c2 = st.columns([1, 1])
    thresh = c1.selectbox("Confidence (pick_prob > …)",
                          nd.TOTALS_CONF_THRESHOLDS, index=0,
                          key="nba_totals_card_conf")
    side = c2.selectbox("Side", ["All", "Over", "Under"], index=0,
                        key="nba_totals_card_side")
    s = nd.totals_monitor_stats(decided, min_pct=thresh, side=side)
    if not s["n"]:
        st.caption("No priced games at this confidence / side.")
        return
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Games", f"{s['n']:,}")
    m2.metric("Win rate (W/(W+L))", _fmt_pct(s["win_rate"]))
    m3.metric("Wins / Losses", f"{s['n_wins']:,} / {s['n_losses']:,}")
    m4.metric("Pushes excluded", f"{s['n_pushes']:,}")
    if s["sides"]:
        rows = [{"Side": name, "n": r["n"],
                 "Win rate": _fmt_pct(r["win_rate"]), "Pushes": r["n_pushes"]}
                for name, r in s["sides"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)
    st.caption(
        f"Thresh: pick_prob > {thresh}% (cumulative — nested subsets). "
        "Win rate is 2-way re-normalized: whole-number-line pushes are "
        "neither wins nor losses (excluded from both). This side split is "
        "the Scoring-Mean diagnostic — compare each side vs its own mean "
        "predicted P."
    )


def _render_runline_calibration_card(decided) -> None:
    """Interactive favorite-cover card — MLB's 5-metric layout + sides."""
    st.markdown("**Run Line — favorite cover calibration card**")
    line = st.selectbox("Line (favorite −L)", nd.SPREAD_LINE_CHOICES,
                        index=2, key="nba_runline_card_line")
    mag = abs(float(line))
    s = nd.runline_monitor_stats(decided, mag)
    if not s["n"]:
        st.caption("No priced games at this line.")
        return
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Games", f"{s['n']:,}")
    m2.metric("Model predicted", _fmt_pct(s["predicted_2way"]))
    m3.metric("Win rate (W/(W+L))", _fmt_pct(s["win_rate"]))
    m4.metric("Wins / Losses", f"{s['n_wins']:,} / {s['n_losses']:,}")
    m5.metric("Pushes", f"{s['n_pushes']:,}")
    if s["sides"]:
        rows = [{"Favorite": name, "n": r["n"],
                 "Win rate": _fmt_pct(r["win_rate"]), "Pushes": r["n_pushes"]}
                for name, r in s["sides"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)
    st.caption(
        f"Line {float(line):+.1f} on the home side. Favorite = MONEYLINE "
        "favorite (P(win) > 50%); its cover probability is read from the "
        "threshold column on its own side, push folded out of both. Model "
        "predicted and Win rate are BOTH 2-way re-normalized, so a "
        "whole-line card is never read as a large under-prediction. "
        "−0.5 ≡ outright win (integer margins, no NBA ties)."
    )


def _render_fit_panel(fit: dict) -> None:
    """Distributional fit diagnostics over the NBA monitor's fit block."""
    st.markdown("#### Distributional Fit (NB Monte Carlo)")
    if not fit:
        st.info("No fit diagnostics in the monitor artifact.")
        return
    draws = nd._num(fit.get("mc_draws"))
    seed = nd._num(fit.get("seed"))
    spread = fit.get("spread_grid") or []
    total = fit.get("total_grid") or []
    half = fit.get("half_stops") or []
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("distribution", str(fit.get("distribution") or "—"))
    c2.metric("MC draws", f"{int(draws):,}" if draws else "—")
    c3.metric("seed", str(int(seed)) if seed is not None else "—")
    c4.metric("half stops", ", ".join(f"{v:+g}" for v in half) if half else "—")
    st.caption(
        f"Priced grids — spread {spread[0]:g}…{spread[-1]:g} "
        f"(integer steps + the ±0.5 half stop) · totals "
        f"{total[0]:g}…{total[-1]:g}. The spread grid has no half steps "
        "beyond ±0.5, so every whole-number line is push-capable — which is "
        "why the history tables resolve 3-way."
    )


def _render_model_card(monitor: dict) -> None:
    """Run-engine model card — MLB's per-market-line fb-box table."""
    fit = (monitor or {}).get("fit") or {}
    st.markdown("### Run-Engine Model (NB Monte Carlo sampler)")
    draws = nd._num(fit.get("mc_draws"))
    seed = nd._num(fit.get("seed"))
    draws_txt = f"{int(draws):,}" if draws is not None else "—"
    seed_txt = str(int(seed)) if seed is not None else "—"
    st.caption(
        f"Sampler — {fit.get('distribution') or '—'} · "
        f"MC draws {draws_txt} · seed {seed_txt} · two Poisson score "
        "regressors feed a seeded negative-binomial paired score sampler. "
        "Totals and point spreads are model probabilities only; no "
        "sportsbook price appears anywhere on this page."
    )
    mm = (monitor or {}).get("market_metrics") or {}
    rows = []
    for key, value in mm.items():
        if not isinstance(value, dict):
            continue
        n = value.get("n")
        rows.append(
            f"<tr>"
            f"<td style='color:#E2E8F0;font-weight:700;'>{key}</td>"
            f"<td>{_fmt(value.get('ece'))}</td>"
            f"<td>{_fmt(value.get('ece_calibrated'))}</td>"
            f"<td>{_fmt(value.get('brier'), 4)}</td>"
            f"<td>{_fmt(value.get('logloss'), 4)}</td>"
            f"<td style='color:#E2E8F0;'>{int(n):,}</td>"
            f"</tr>")
    if not rows:
        st.info("Per-line engine OOF metrics appear after a pipeline run "
                "that emits market_metrics.")
        return
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <table class="fb-table">
            <thead><tr><th>MARKET LINE</th><th>ECE (RAW)</th><th>ECE (CAL)</th>
            <th>BRIER</th><th>LOG LOSS</th><th>N (OOF)</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        <div style="color:#64748B;font-size:0.78rem;margin-top:6px;">
          Run engine is a single NB(λ, α(λ)) paired-score Monte Carlo
          sampler — no member blend; rows are per market line. ECE-CAL is
          prequentially calibrated when the run ships a calibration map;
          otherwise the cell shows the em-dash rather than a copy of the raw
          figure. N = pooled OOF games scored for the line.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_rolling_history(rolling: dict) -> None:
    """Per-card rolling history as MLB's compact table."""
    rows = nd.rolling_history_rows(rolling)
    if not rows:
        st.info("No rolling history yet (first build starts empty).")
        return
    st.markdown("#### Rolling ECE-Calibrated History (per winner card)")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def run() -> None:
    utils.inject_css()
    markets, date_str = utils.load_nba_run_engine_markets("nba")
    date_str = date_str or FALLBACK_DATE
    monitor = utils.load_nba_run_engine_monitor("nba") or {}

    st.markdown(
        "<div style='font-size:1.7rem;font-weight:800;color:#E2E8F0;'>"
        "Today's Totals &amp; Run Lines</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='color:#94A3B8;margin:2px 0 14px;'>"
        "Run-engine model diagnostics from the per-side era model + pinned "
        "joint (NB) over the line grid — per-game projections and full "
        "probabilities live in nba_run_engine_markets_*.csv.</div>",
        unsafe_allow_html=True,
    )

    if markets is None or not len(markets):
        st.warning(
            f"No run-engine markets artifact for {date_str}. Attempted "
            f"file: `nba-backend/data_delivery/"
            f"nba_run_engine_markets_{date_str}.csv`. This page always "
            "loads the latest available artifact. The panel fills after the "
            "next pipeline run. Nothing is fabricated."
        )
        return
    decided = _decided_rows(markets)

    # ------------------------------------------------------------------
    # Diagnostics — the SAME five tabs as MLB
    # ------------------------------------------------------------------
    st.markdown("### Diagnostics")
    st.caption(f"OOF artifact: {date_str} · {len(decided):,} decided games · "
               "historical performance uses OOF predictions only")
    if decided.empty:
        st.warning(
            "No decided OOF rows in nba_run_engine_markets for this date — "
            "diagnostics need outcomes; nothing is fabricated.")
    else:
        tabs = st.tabs(DIAG_TABS)
        with tabs[0]:
            _render_distribution_tab(decided)
        with tabs[1]:
            _render_relativized_tab(decided)
        with tabs[2]:
            _render_pooled_tab(decided)
        with tabs[3]:
            _render_game_total_tab(decided)
        with tabs[4]:
            _render_spread_tab(decided)

    # ------------------------------------------------------------------
    # Prediction history — the two MLB fb-box tables
    # ------------------------------------------------------------------
    st.markdown("### Prediction History — Totals & Run Lines")
    if decided.empty:
        st.info("No decided OOF rows in the run-engine markets artifact — "
                "the totals/run-line history fills after a run that ships "
                "decided games. Nothing is fabricated in the meantime.")
    else:
        dts = pd.to_datetime(decided.get("gameday"), errors="coerce").dropna()
        if len(dts):
            lo, hi = dts.min().date(), dts.max().date()
            fc1, fc2, _ = st.columns([1, 1, 2])
            start_d = fc1.date_input("History start date", value=lo,
                                     min_value=lo, max_value=hi)
            end_d = fc2.date_input("History end date", value=hi,
                                   min_value=lo, max_value=hi)
            if start_d > end_d:
                start_d, end_d = end_d, start_d
            _render_totals_history(decided, monitor, start_d, end_d)
            _render_runline_history(decided, monitor, start_d, end_d)
        else:
            st.info("Decided rows carry no parsable gameday — the history "
                    "tables need dates and are skipped. Nothing is fabricated.")

    # ------------------------------------------------------------------
    # Monitor — winner cards, calibration cards, fit, drift, model card
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Run-Line & Totals Monitor")
    st.caption(
        "Run-engine winner cards (over/under, run line, derived ML) + "
        "distributional fit + rolling history from "
        "nba_run_engine_monitor_*.json. **Honesty note:** the run-engine "
        "ECE-calibrated figure is typically weaker than a raw pooled ECE on "
        "this model — that is the actual calibration quality, and no "
        "styling hides it."
    )
    if not monitor:
        st.warning(
            f"No run-engine monitor artifact for {date_str}. Attempted "
            f"file: `nba-backend/data_delivery/nba_run_engine_monitor_"
            f"{date_str}.json`. The monitor fills after the next pipeline "
            "run."
        )
        return

    _render_winner_cards(nd.winner_cards(decided, monitor))

    if not decided.empty and {"home_score", "away_score"}.issubset(decided.columns):
        st.markdown("---")
        with st.expander("Calibration Cards — Totals & Run Line",
                         expanded=True):
            _render_totals_calibration_card(decided)
            st.markdown("<div style='height:12px'></div>",
                        unsafe_allow_html=True)
            _render_runline_calibration_card(decided)

    with st.expander("Distributional Fit Diagnostics", expanded=False):
        _render_fit_panel(monitor.get("fit") or {})

    nd.render_run_engine_drift(
        nd.load_run_engine_csv(date_str, "nba_run_engine_feature_drift"))
    nd.render_run_engine_coverage(
        nd.load_run_engine_csv(date_str, "nba_run_engine_feature_coverage"))
    _render_model_card(monitor)

    with st.expander("Rolling History (last 10 points per card)",
                     expanded=False):
        _render_rolling_history(monitor.get("rolling") or {})


if __name__ == "__main__":
    run()
