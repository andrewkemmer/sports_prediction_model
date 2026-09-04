"""NFL Today's Totals & Run Lines (run-engine slate-serve, MARKET-FREE).

Mirrors the MLB ``markets.py`` page STRUCTURE over the NFL artifacts
(``nfl_run_engine_markets_YYYYMMDD.csv`` + ``nfl_run_engine_monitor_*.json``,
emitted by the NFL slate-serve runner), with the NFL conventions documented
in the slate record's mapping table:

  * The slate carries only ``kind == 'slate'`` rows (scheduled games priced
    ahead of play). No OOF/decided rows exist until a run ships decided
    slate games — Diagnostics and Prediction History therefore show the
    research-pinned OOF baseline (from the monitor JSON) and honest empty
    states respectively; nothing is fabricated.
  * Fair lines are MODEL medians of the margin/total PMFs (``fair_spread`` /
    ``fair_total``); offered lines and shrink columns are NEVER rendered.
  * The NFL margin PMF is integer-support, so whole-number totals and
    spreads carry a real push band (the slate prices integer lines only —
    ``p_home_cover_<L>/p_push_<L>`` over −14..+14 and
    ``p_over_<U>/p_under_<U>/p_push_<U>`` over 24..66).
  * The derived-ML pair (``p_home_win_derived``/``p_away_win_derived``) is
    P(H>A)/(1−P(tie)) from the calibrated 76×76 joint — the NFL-specific
    raw-vs-derived pair a ±0.5 stop needs. Raw half-point covers are the
    ``p_*_cover_{minus,plus}_half`` columns.

MARKET-FREE POLICY (the product decision): this page renders model fair
lines and model probabilities ONLY. Offered/book lines, shrink columns,
market comparison and market-derived "edge" displays never render — the
dashboard is a model product, not a market tool.

The page is registered once under the ``markets`` url_path (Home.py); the
sport dispatcher in ``markets.py`` imports this module and calls ``run()``
when the active sport is NFL. Import is side-effect-free (render happens
only inside ``run``), so tests can import the module without a Streamlit
page context.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import nfl_slate_view as sv
import utils


def run() -> None:
    """Render the NFL Totals & Run Lines page (market-free)."""
    utils.inject_css()

    # Always the most recent run (like the MLB markets page / Calibration):
    # never key to the date picked on Today's Games.
    slate, date = utils.load_nfl_run_engine_markets("nfl")
    monitor = utils.load_nfl_run_engine_monitor("nfl")

    st.markdown(
        "<div style='font-size:1.7rem;font-weight:800;color:#E2E8F0;'>"
        "🏈 NFL — Totals &amp; Run Lines</div>",
        unsafe_allow_html=True,
    )
    _d = utils.format_date_long(date) if date else "—"
    st.markdown(
        f"<div style='color:#94A3B8;margin:2px 0 14px;'>Model-fair "
        f"run-engine lines from the 76×76 joint score model · slate "
        f"artifact {date or 'n/a'} ({_d}) · market-free: model fair lines "
        f"and probabilities only — offered lines, shrink and 'edge' are "
        f"never shown.</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Per-game projections (fair lines + model probabilities only)
    # ------------------------------------------------------------------
    st.markdown("### Scheduled Games — Model Fair Lines")
    if slate is None or not len(slate):
        st.warning(
            "No NFL run-engine markets artifact is available yet. The "
            "panel fills after a slate-serve run ships "
            "nfl_run_engine_markets_*.csv."
        )
    elif not sv.has_fair_columns(slate):
        st.warning("Slate artifact predates the fair-line grid — waiting "
                   "for the next slate-serve run.")
    else:
        weeks = sorted(slate["week"].dropna().unique().tolist())
        week = st.selectbox("Week", weeks, index=0,
                            format_func=lambda w: f"Week {int(w)}",
                            key="nfl_markets_week")
        view = slate[slate["week"] == week].copy()
        view = view.sort_values("gameday").reset_index(drop=True)

        totals = [int(t) for t in sv.grid_rows(slate)[1]]
        spreads = [int(s) for s in sv.grid_rows(slate)[0]]

        st.caption(
            f"{len(view)} scheduled games · fair spread/total are the "
            "model's median margin/total; P(cover)/P(over) at the FAIR line "
            "(whole-number lines carry a real push band). Expand a game to "
            "price any line on the precomputed grid."
        )
        for _, r in view.iterrows():
            home = str(r.get("home_team", "") or "")
            away = str(r.get("away_team", "") or "")
            gd = str(r.get("gameday", "") or "")
            label = f"{away} @ {home} · {gd}"
            with st.expander(label, expanded=False):
                st.markdown(sv.runengine_html(r, home, away),
                            unsafe_allow_html=True)
                fair_total = int(round(float(r["fair_total"])))
                fair_spread = int(round(float(r["fair_spread"])))
                t_opts = sorted(set(totals) | {fair_total})
                s_opts = sorted(set(spreads) | {fair_spread})
                c1, c2 = st.columns(2)
                t_line = c1.selectbox(
                    "Total", t_opts, index=t_opts.index(fair_total),
                    format_func=lambda t: f"{t}  (fair)"
                    if t == fair_total else str(t),
                    key=f"nfl_tot_{r['game_id']}")
                s_line = c2.selectbox(
                    "Spread (home side)", s_opts,
                    index=s_opts.index(fair_spread),
                    format_func=lambda s: f"{_home_spread(s):+d}  (fair)"
                    if s == fair_spread else f"{_home_spread(s):+d}",
                    key=f"nfl_sp_{r['game_id']}")
                _price_row(r, home, away, t_line, s_line)

    # ------------------------------------------------------------------
    # Diagnostics — per-line-type calibration. Decided slate rows do not
    # exist until a run ships decided games; until then the research-pinned
    # OOF baseline (monitor JSON) is shown with provenance — never
    # fabricated slate calibration.
    # ------------------------------------------------------------------
    st.markdown("### Diagnostics")
    baseline = (monitor or {}).get("oof_baseline_research_pinned") or {}
    if baseline:
        st.caption(
            "No decided slate rows yet — diagnostics will compute real "
            "calibration once runs ship decided games. The numbers below "
            "are the research-pinned pooled-OOF baseline from the "
            "committed era / market records (provenance in the Run-Line & "
            "Totals Monitor below)."
        )
        dm = baseline.get("derived_ml") or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("Covers ECE (pooled OOF)",
                  f"{baseline.get('covers_ece_pooled', 0):.3f}")
        c2.metric("Seam totals ECE (pooled OOF)",
                  f"{baseline.get('totals_ece_pooled_own', 0):.3f}")
        c3.metric("Derived-ML logloss", f"{dm.get('logloss', 0):.4f}")
        c1.metric("Derived-ML AUC", f"{dm.get('auc', 0):.3f}")
        c2.metric("Derived-ML ECE", f"{dm.get('ece', 0):.3f}")
    else:
        st.info("No diagnostics baseline in the monitor artifact — a "
                "slate-serve run will publish nfl_run_engine_monitor_*.json "
                "with the research-pinned OOF figures.")

    # ------------------------------------------------------------------
    # Prediction History — fills only after runs ship decided rows.
    # ------------------------------------------------------------------
    st.markdown("### Prediction History — Totals & Run Lines")
    _has_scores = bool(slate is not None and len(slate)
                       and {"home_score", "away_score"}.issubset(
                           slate.columns)
                       and slate[["home_score", "away_score"]]
                       .notna().any().any())
    if slate is None or not len(slate):
        st.info("No slate artifact yet — history appears with the first "
                "slate-serve run.")
    elif not _has_scores:
        st.info(
            "No decided slate rows in the NFL run-engine markets artifact — "
            "the totals/run-line history fills after a run that ships "
            "decided games. Nothing is fabricated in the meantime."
        )
    else:
        decided = slate[slate[["home_score", "away_score"]]
                        .notna().all(axis=1)]
        st.caption(f"{len(decided)} decided games — history table "
                   "populates from the decided slate rows.")

    # ------------------------------------------------------------------
    # Run-Line & Totals Monitor — research-pinned OOF baseline +
    # accumulating slate history (empty by design on the first run).
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Run-Line & Totals Monitor")
    st.caption(
        "Research-pinned pooled-OOF baseline from the committed era / "
        "market records + the accumulating slate history. **Honesty "
        "note:** run-engine calibration is weaker than the moneyline "
        "monitor's — this is the actual calibration quality; no styling "
        "hides it."
    )
    if monitor:
        hist = monitor.get("slate_history") or []
        st.caption(f"Slate history: {len(hist)} recorded run(s) so far.")
        if hist:
            st.dataframe(pd.DataFrame(hist), use_container_width=True,
                         hide_index=True)
        else:
            st.info("No slate history yet — this is the first slate-serve "
                    "run. The accumulating section fills as runs ship "
                    "decided games.")
        prov = baseline.get("provenance") or []
        if prov:
            st.caption("Provenance: " + " · ".join(str(p) for p in prov))
    else:
        st.warning("No NFL run-engine monitor artifact — the monitor fills "
                   "after a slate-serve run ships "
                   "nfl_run_engine_monitor_*.json.")


def _home_spread(s_threshold: int) -> int:
    """The HOME side's quoted spread for a grid margin threshold s: home is
    at -s (home -5 at threshold +5; home +3 at threshold -3)."""
    return -s_threshold


def _price_row(r, home: str, away: str, t_line: int, s_line: int) -> None:
    """Compact model-probability row at caller-chosen grid lines."""
    po, pu, pp = sv.price_total(r, t_line)
    # s_line is the margin threshold: home -s_line / away +s_line, shared
    # push at margin == s_line (both sides push on the same margin).
    ph, psp, pa = sv.price_spread(r, s_line)
    row = {
        "": "Model probability",
        f"Over {t_line}": sv._pct(po, 1) if po is not None else "—",
        f"Under {t_line}": sv._pct(pu, 1) if pu is not None else "—",
        f"Push {t_line}": sv._pct(pp, 1) if pp is not None else "—",
        f"Home {_home_spread(s_line):+d}": sv._pct(ph, 1)
        if ph is not None else "—",
        f"Away {-_home_spread(s_line):+d}": sv._pct(pa, 1)
        if pa is not None else "—",
        f"Push {s_line:+d}": sv._pct(psp, 1) if psp is not None else "—",
    }
    st.dataframe(pd.DataFrame([row]), hide_index=True,
                 use_container_width=True)


if __name__ == "__main__":
    run()
