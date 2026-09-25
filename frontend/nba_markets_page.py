"""NBA Totals & Point Spread — structural mirror of the shared markets page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import nba_market_diagnostics as nd
import utils

DIAG_TABS = nd.DIAG_TABS
HISTORY_HEADERS = ("DATE</th><th>MATCHUP</th><th>SCORE (A–H)</th><th>LINE</th>"
                   "</th><th>MODEL PICK</th><th>WINNER</th><th>RESULT")
FALLBACK_DATE = "20260924"


def _fmt(value, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _decided_rows(frame):
    return nd.decided_rows(frame)


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


def _history_view(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    if "gameday" in frame:
        frame = frame.copy()
        frame["_date"] = pd.to_datetime(frame.gameday, errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        frame["_date"] = frame.get("game_date", "")
    return frame


def _render_history(frame: pd.DataFrame) -> None:
    st.markdown("### Prediction History — Totals & Point Spread")
    view = _history_view(frame)
    if view.empty:
        st.info("No decided NBA run-line history is available yet.")
        return
    cols = st.columns(2)
    with cols[0]:
        start = st.text_input("From", value="", key="nba_market_history_start")
    with cols[1]:
        end = st.text_input("To", value="", key="nba_market_history_end")
    if start:
        view = view[view._date >= start]
    if end:
        view = view[view._date <= end]
    st.dataframe(view.head(200), use_container_width=True)


def _render_winner_cards(cards: dict) -> None:
    st.markdown("#### Winner Cards")
    if not cards:
        st.info("Winner-card metrics appear after a decided OOF run.")
        return
    cols = st.columns(3)
    for col, key in zip(cols, ("total", "home", "away")):
        card = cards.get(key, {}) or {}
        col.metric(key.title(), f"{card.get('brier', float('nan')):.3f}" if card.get("brier") is not None else "—",
                   f"n={card.get('n', 0):,}")


def _render_fit_panel(decided) -> None:
    with st.expander("Distributional Fit Diagnostics", expanded=False):
        if decided is None or decided.empty:
            st.info("No decided distribution rows are available for fit checks.")
        else:
            total = pd.to_numeric(decided.total, errors="coerce")
            margin = pd.to_numeric(decided.margin, errors="coerce")
            st.caption(f"Negative-binomial paired score sampler · {len(decided):,} decided games · seed 42")
            st.write({"mean_total": float(total.mean()), "std_total": float(total.std()),
                      "mean_margin": float(margin.mean()), "std_margin": float(margin.std())})


def _render_model_card(monitor: dict) -> None:
    st.markdown("### Run-Engine Model (Poisson + NB Monte Carlo)")
    st.caption("Two Poisson score regressors feed a seeded negative-binomial paired score sampler; totals and point spreads are model probabilities only.")
    metrics = (monitor or {}).get("market_metrics") or {}
    if metrics:
        st.dataframe(pd.DataFrame([{"line": key, **value} for key, value in metrics.items()
                                  if isinstance(value, dict)]), use_container_width=True)
    else:
        st.info("Per-line engine metrics appear after a pipeline run emits market_metrics.")


def _render_rolling_history(rolling: dict) -> None:
    with st.expander("Rolling History (last 10 points per card)", expanded=False):
        if not rolling:
            st.info("No rolling history yet (first build starts empty).")
        else:
            rows = [{"game_id": key, "points": len(value)} for key, value in rolling.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


def run() -> None:
    utils.inject_css()
    markets, date_str = utils.load_nba_run_engine_markets("nba")
    date_str = date_str or FALLBACK_DATE
    st.markdown("<div style='font-size:1.7rem;font-weight:800;color:#E2E8F0;'>Today's Totals &amp; Point Spread</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#94A3B8;margin:2px 0 14px;'>NBA model diagnostics from Poisson score regressors and a seeded negative-binomial score distribution — fair lines and probabilities only.</div>", unsafe_allow_html=True)
    if markets is None or markets.empty:
        st.warning(f"No usable NBA run-engine markets artifact found for {date_str}. The panel fills after a valid NBA artifact is reachable.")
        return
    decided = _decided_rows(markets)
    st.caption(f"OOF artifact: {date_str} · {len(decided):,} decided games · historical performance uses OOF predictions only")
    st.markdown("### Diagnostics")
    if decided.empty:
        st.warning("No decided OOF rows in the NBA markets artifact — diagnostics need outcomes; nothing is fabricated.")
    else:
        tabs = st.tabs(DIAG_TABS)
        with tabs[0]: _render_distribution_tab(decided)
        with tabs[1]: _render_relativized_tab(decided)
        with tabs[2]: _render_pooled_tab(decided)
        with tabs[3]: _render_total_tab(decided)
        with tabs[4]: _render_spread_tab(decided)
    st.divider()
    _render_history(markets)
    st.divider()
    st.markdown("### Run-Line & Totals Monitor")
    st.caption("Monitor metrics are computed from settled OOF rows; no current-slate or sportsbook outcome is substituted.")
    monitor = utils.load_nba_run_engine_monitor("nba") or {}
    _render_winner_cards(nd.winner_cards(decided))
    with st.expander("Calibration Cards", expanded=False):
        st.caption("Totals: 220 / 225 / 230 · Point spread: 2 / 5")
        st.dataframe(pd.DataFrame(nd.totals_monitor_stats(decided).get("rows", [])), use_container_width=True)
        st.dataframe(pd.DataFrame(nd.runline_monitor_stats(decided).get("rows", [])), use_container_width=True)
    _render_fit_panel(decided)
    nd.render_run_engine_drift(nd.load_run_engine_csv(date_str, "nba_run_engine_feature_drift"))
    nd.render_run_engine_coverage(nd.load_run_engine_csv(date_str, "nba_run_engine_feature_coverage"))
    _render_model_card(monitor)
    _render_rolling_history(nd.fold_slate_history(utils.load_nba_run_engine_monitor_series("nba")))


if __name__ == "__main__":
    run()
