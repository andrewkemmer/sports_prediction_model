"""NBA Totals & Point Spread — structural mirror of the MLB markets page.

Section order, headings, widget kinds and card layout all follow
``markets.py`` (MLB is the reference implementation); only the sport-specific
wording and the pricing lines differ. The local render comparison lives in
``compare_totals_dashboards.py``, which renders both pages through Streamlit's
AppTest and diffs the resulting element trees.
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
    """MLB's ``_fmt`` — a number to ``digits``, or the em-dash placeholder."""
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


# --- Diagnostics tabs (the same five as MLB) ------------------------------


def _render_distribution_tab(decided):
    result = nd.total_distribution(decided)
    if result.get("warning"):
        st.warning(result["warning"])
        return
    chart = nd.chart_distribution(result)
    if chart is not None:
        utils.show_chart(chart)
    st.caption(f"Observed NBA totals across {result['n_games']:,} decided OOF games.")


def _render_relativized_tab(decided):
    result = nd.calibration_curve(nd.relativized_pairs(decided), "NBA relativized totals")
    if result.get("warning"):
        st.warning(result["warning"])
        return
    chart = nd.chart_calibration(result, "NBA relativized totals")
    if chart is not None:
        utils.show_chart(chart)
    st.caption(f"{result['n_pairs']:,} prequential line observations; solid curve is model calibration.")


def _render_pooled_tab(decided):
    result = nd.calibration_curve(nd.fixed_line_pairs(decided, (220, 225, 230)), "Pooled NBA totals")
    if result.get("warning"):
        st.warning(result["warning"])
        return
    chart = nd.chart_calibration(result, "Pooled NBA totals")
    if chart is not None:
        utils.show_chart(chart)
    st.caption("The same games are evaluated at 220, 225, and 230; these are not independent observations.")


def _render_total_tab(decided):
    result = nd.game_total_calibration(decided)
    if result.get("warning"):
        st.info(result["warning"])
    else:
        chart = nd.chart_calibration(result, "Game total calibration")
        if chart is not None:
            utils.show_chart(chart)
        st.caption(f"Fair-total calibration · {result['n_pairs']:,} priced games.")


def _render_spread_tab(decided):
    result = nd.run_line_calibration(decided, 2.0)
    if result.get("warning"):
        st.info(result["warning"])
    else:
        chart = nd.chart_calibration(result, "NBA point-spread calibration")
        if chart is not None:
            utils.show_chart(chart)
        st.caption("Point-spread lines are model thresholds; no sportsbook edge is shown.")


# --- Prediction history (MLB's two fb-box tables) -------------------------


def _cal_note(monitor: dict | None) -> str:
    """MLB's _market_prob_note: is the shipped probability map calibrated?"""
    cards = (monitor or {}).get("calibration_cards") or {}
    if cards:
        return (" · probabilities are post-calibration σ(a·logit(p)+b)"
                " (prequential Platt, per line)")
    return (" · probabilities are RAW (no calibration map shipped in "
            "the run-engine artifact)")


def _render_totals_history(decided, monitor, start_d, end_d) -> None:
    """Game-totals prediction history: DATE | MATCHUP | SCORE (A–H) |
    LINE (O/U X) | MODEL PICK (Over/Under X (p%)) | WINNER | RESULT.

    Mirrors MLB's ``_render_totals_history``: side radio, header markdown,
    caption (n · side · win rate · pushes · cal note), fb-box table, most
    recent first. Pushes (total == whole-number line) render '—' and are
    excluded from the W/(W+L) rate, matching the diagnostics convention.
    The All/Over/Under filter is required, not cosmetic: Over
    under-predicting and Under over-predicting otherwise net to zero pooled.
    """
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
        f"first — scroll for older results{push_txt}"
        f"{_cal_note(monitor)}"
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
    """Point-spread prediction history at each game's OWN fair line.

    Same seven columns and caption shape as the totals table, with 3-way
    resolution: the pick covers when margin > line, a margin exactly equal to
    the line is a PUSH (renders '—', excluded from the win rate), anything
    less is a loss. The displayed probability is already 2-way normalized
    (push mass folded out of both sides), so the LINE and MODEL PICK columns
    share one basis.
    """
    view = nd.filter_history_frame(
        nd.runline_history_frame(decided), start_d, end_d)
    view = view.sort_values("game_date", ascending=False)
    st.markdown("#### Point Spread — Prediction History")
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
        f"fair line, probability 2-way re-normalized{_cal_note(monitor)}"
    )
    rows = []
    for _, r in view.iterrows():
        sign = "+" if r["line"] >= 0 else "−"
        rows.append(
            f"<tr><td>{_date_str(r['game_date'])}</td>"
            f"<td>{r['away']} @ {r['home']}</td>"
            f"<td>{_score_cell(r)}</td>"
            f"<td>RL {r['home']} {sign}{abs(r['line']):.0f} / "
            f"{r['away']} {sign}{abs(r['line']):.0f}</td>"
            f"<td>{r['pick']} ({r['pick_prob']:.0%})</td>"
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
    """The three binary winner cards (pick framing) in a row — MLB's layout."""
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
                       "W/(W+L): rows with no side lean (P = 50%) and whole-"
                       "line pushes are excluded from both sides.")


def _render_totals_calibration_card(decided) -> None:
    """Interactive Totals calibration card — MLB's card, NBA's lines.

    Percent-confidence toggle (cumulative) + All/Over/Under side filter, win
    rate 2-way re-normalized W/(W+L) with whole-line pushes folded out of
    both numerator and denominator. The side split is the Scoring-Mean
    diagnostic: compare each side against its own mean predicted P.
    """
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
    m2.metric("Model predicted", _fmt_pct(s["predicted_2way"]))
    m3.metric("Win rate (W/(W+L))", _fmt_pct(s["win_rate"]))
    m4.metric("Wins / Losses", f"{s['n_wins']:,} / {s['n_losses']:,}")
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
    """Interactive point-spread calibration card at a chosen grid line.

    The line options are restricted to the grid the run engine actually
    prices: integer magnitudes plus the ±0.5 half stop. A whole-number line
    CAN push here (NBA margins are integers and there is no -1.5 column), so
    this card uses MLB's 3-way resolution rather than its never-push path.
    """
    st.markdown("**Point Spread — favorite cover calibration card**")
    line = st.selectbox("Line (favorite −L)", nd.SPREAD_LINE_CHOICES,
                        index=2, key="nba_runline_card_line")
    mag = abs(float(line))
    if mag not in nd.SPREAD_GRID_CUT:
        st.caption("Line outside the priced grid.")
        return
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
        f"Line {float(line):+.1f} on the home side. Model predicted and Win "
        "rate are BOTH 2-way re-normalized (a whole-line push, favored "
        "margin == L, is folded out of both sides — predicted = "
        "P(cover)/[P(cover)+P(dog)] on the same basis as W/(W+L)), so a "
        "whole-line card is never read as a large under-prediction. "
        "−0.5 ≡ outright win (integer margins, no NBA ties)."
    )


def _render_fit_panel(fit: dict) -> None:
    """Distributional fit diagnostics over the NBA monitor's fit block.

    The NBA monitor records the sampler's shape (distribution, draw count,
    seed, and the priced grids) rather than MLB's per-side alpha/chi2
    curves, so this reads the keys that exist and shows '—' for the ones
    that do not. Nothing is inferred to fill a gap.
    """
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
        "why both history tables resolve 3-way."
    )


def _render_model_card(monitor: dict) -> None:
    """Run-engine model card — MLB's per-market-line fb-box table.

    The NBA engine is a single NB(λ, α(λ)) paired-score sampler with no
    member blend, exactly as on the MLB page, so there is one row per priced
    line and no ensemble column. Columns are read defensively because the
    NBA monitor uses short metric names (``brier``/``logloss``/``ece``) where
    MLB uses ``engine_*``-prefixed ones.
    """
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
          otherwise the run reports raw-based values and the ECE (CAL) cell
          shows the em-dash rather than a copy of the raw figure. N = pooled
          OOF games scored for the line.
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
        "Today's Totals &amp; Point Spread</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='color:#94A3B8;margin:2px 0 14px;'>"
        "NBA model diagnostics from Poisson score regressors and a seeded "
        "negative-binomial score distribution — fair lines and probabilities "
        "only.</div>",
        unsafe_allow_html=True,
    )

    if markets is None or markets.empty:
        st.warning(
            f"No usable NBA run-engine markets artifact found for {date_str}. "
            f"Attempted file: `nba-backend/data_delivery/"
            f"nba_run_engine_markets_{date_str}.csv`. The panel fills after a "
            "valid NBA artifact is reachable. Nothing is fabricated."
        )
        return
    decided = _decided_rows(markets)

    st.markdown("### Diagnostics")
    st.caption(f"OOF artifact: {date_str} · {len(decided):,} decided games · "
               "historical performance uses OOF predictions only")
    if decided.empty:
        st.warning("No decided OOF rows in the NBA markets artifact — "
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
            _render_total_tab(decided)
        with tabs[4]:
            _render_spread_tab(decided)

    # ------------------------------------------------------------------
    # Prediction history — the two MLB fb-box tables
    # ------------------------------------------------------------------
    st.markdown("### Prediction History — Totals & Point Spread")
    if decided.empty:
        st.info("No decided OOF rows in the run-engine markets artifact — "
                "the totals/spread history fills after a run that ships "
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
        "Run-engine winner cards (over/under, point spread, derived ML) + "
        "distributional fit + rolling history from "
        "nba_run_engine_monitor_*.json. **Honesty note:** the run-engine "
        "ECE-calibrated figure is typically weaker than a raw pooled ECE on "
        "this model — that is the actual calibration quality, and no styling "
        "hides it."
    )
    if not monitor:
        st.warning(
            f"No run-engine monitor artifact for {date_str}. Attempted file: "
            f"`nba-backend/data_delivery/nba_run_engine_monitor_"
            f"{date_str}.json`. The monitor fills after the next pipeline run."
        )
        return

    _render_winner_cards(nd.winner_cards(decided, monitor))

    if not decided.empty and {"home_score", "away_score"}.issubset(decided.columns):
        st.markdown("---")
        with st.expander("Calibration Cards — Totals & Point Spread",
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
