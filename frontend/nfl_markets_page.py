"""NFL Today's Totals & Run Lines — STRUCTURAL 1:1 MIRROR of the MLB markets
page (``frontend/markets.py``), with NFL artifacts substituted.

ANATOMY CHECKLIST (the MLB page's render order — this page mirrors it
exactly; see the parity tests, which pin the same order/labels/wording):

  1. Title "Today's Totals & Run Lines" + subtitle caption.
  2. Markets CSV load (always the most recent run) + loud missing/stale
     warnings (``nfl_run_engine_markets_*`` substituted for MLB's).
  3. "### Diagnostics" — decided rows: empty → the MLB wording warning (no
     fabricated calibration); then the SAME five tabs as MLB
     (Distribution / Relativized / Pooled lines / Game Total Lines /
     Run Lines). Until runs ship decided games the tabs carry the
     research-pinned pooled-OOF baseline from the monitor JSON, labeled
     with provenance — NEVER fabricated slate calibration. When decided
     rows exist, the same tab structure renders real calibration.
  4. "### Prediction History — Totals & Run Lines" — empty → the MLB
     wording info; decided rows → the same date-range + side-filter
     widgets and the same fb-table schema (DATE | MATCHUP | SCORE (A–H) |
     LINE | MODEL PICK | WINNER | RESULT) over the NFL columns.
  5. "---" + "### Run-Line & Totals Monitor" + the honesty caption →
     research-pinned OOF baseline (the NFL monitor's winner-card
     equivalent) + the accumulating slate-history section (first-run
     empty state, MLB rolling-history convention).
  6. NO per-game projections hero on this page — MLB does not render
     per-game content on the markets page (its docstring: the per-game
     slate board is UN-RENDERED); per-game pricing lives in the artifact
     and on the Today's Games cards. The week-selector / expandable-row
     layout previously shipped here is REMOVED (layout parity).

MARKET-FREE BY POLICY (unchanged): this page renders model fair lines and
model probabilities ONLY. Offered/book lines, shrink columns and
market-derived "edge" never render — the dashboard is a model product, not
a market tool. The artifact keeps those columns (test-pinned); the page
never reads them.

LEGITIMATE NFL WORDING DELTAS (documented, MLB wording is the default
elsewhere):
  * Push vs tie: NFL integer grids carry a real push band (P(margin == L),
    P(total == U)) and whole-number lines push — MLB half-line grids never
    push. The history caption notes pushes excluded, same convention as
    MLB's whole-line notes.
  * The ±0.5 stop's raw-vs-derived pair: raw −0.5 excludes ties, raw +0.5
    includes them, and the derived ML pair normalizes ties out — rendered
    on the Today's Games cards, not on this page.
  * Subtitle: the NFL run engine is the per-side era model + pinned 76×76
    joint (DN) — not MLB's NB(λ, α(λ)) sampler; the sentence shape is
    identical, the model description is NFL-accurate.
  * Monitor sub-sections MLB renders from pipeline artifacts the NFL
    pipeline does not emit (fit panel, drift/coverage CSVs, per-line model
    card) are absent until the NFL slate-serve runner ships them; the
    research-pinned baseline + slate history are the honest first-run
    content. No empty fake sections.

Import is side-effect-free (render happens only inside ``run()``), so
tests can import the module without a Streamlit page context.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import nfl_slate_view as sv
import utils

# MLB's exact tab labels + section headers (parity pins — see the tests).
DIAG_TABS = ["Distribution", "Relativized", "Pooled lines",
             "Game Total Lines", "Run Lines"]

# The most recent run, like the MLB page (never the date picked on Today's
# Games). Fallback constant when no artifact resolves yet.
FALLBACK_DATE = "20260903"

# The history table schema (byte-identical to the MLB fb-table header).
HISTORY_HEADERS = ("DATE</th><th>MATCHUP</th><th>SCORE (A–H)</th><th>LINE"
                   "</th><th>MODEL PICK</th><th>WINNER</th><th>RESULT")


def _decided_rows(df: pd.DataFrame | None) -> pd.DataFrame:
    """Rows with decided outcomes (home/away scores) — the slate artifact's
    analog of MLB's ``decided_rows``. Empty when the artifact carries only
    scheduled slate rows (the current first-run state)."""
    if df is None or not len(df):
        return pd.DataFrame()
    if not {"home_score", "away_score"}.issubset(df.columns):
        return pd.DataFrame()
    d = df[df[["home_score", "away_score"]].notna().all(axis=1)]
    return d.reset_index(drop=True)


def _baseline(monitor: dict | None) -> dict:
    return ((monitor or {}).get("oof_baseline_research_pinned") or {})


def _provenance(monitor: dict | None) -> list:
    return _baseline(monitor).get("provenance") or []


def _render_baseline_tab(baseline: dict, tab_index: int, prov: list) -> None:
    """Research-pinned OOF baseline figures per diagnostics tab (the honest
    stand-in until runs ship decided games — nothing fabricated)."""
    if not baseline:
        st.info("No research-pinned OOF baseline in the monitor artifact — "
                "a slate-serve run will publish "
                "nfl_run_engine_monitor_*.json with the figures.")
        return
    dm = baseline.get("derived_ml") or {}
    if tab_index == 0:          # Distribution (totals distribution)
        st.metric("Totals ECE (pooled OOF, own line)",
                  f"{baseline.get('totals_ece_pooled_own', 0):.4f}")
        st.caption(
            "Research-pinned pooled-OOF totals-distribution baseline from "
            "the committed era / market records (seam totals ECE 0.087). "
            "Real per-game calibration replaces this once a run ships "
            "decided games.")
    elif tab_index == 1:        # Relativized (covers, relativized)
        st.metric("Covers ECE (pooled OOF)",
                  f"{baseline.get('covers_ece_pooled', 0):.4f}")
        st.caption(
            "Research-pinned pooled-OOF covers baseline from the committed "
            "adoption records (covers ECE 0.078 → 0.0537 spread-shrunk). "
            "No fabricated slate calibration.")
    elif tab_index == 2:        # Pooled lines (derived-ML discrimination)
        c1, c2, c3 = st.columns(3)
        c1.metric("Derived-ML logloss", f"{dm.get('logloss', 0):.4f}")
        c2.metric("Derived-ML AUC", f"{dm.get('auc', 0):.3f}")
        c3.metric("Derived-ML ECE", f"{dm.get('ece', 0):.4f}")
        st.caption(
            "Research-pinned pooled-OOF derived-ML baseline "
            "P(H>A)/(1−P(tie)) of the calibrated 76×76 joint. The pooled "
            "curve appears here once decided games ship.")
    elif tab_index == 3:        # Game Total Lines (own-line totals)
        st.metric("Totals ECE (pooled OOF, own line)",
                  f"{baseline.get('totals_ece_pooled_own', 0):.4f}")
        st.caption(
            "Research-pinned totals baseline at each game's own fair line. "
            "NFL totals are integer-support — whole-number lines carry a "
            "real push band (P(total == U)), excluded 2-way like MLB's "
            "whole-line notes.")
    else:                       # Run Lines (favorite cover)
        st.metric("Covers ECE (pooled OOF)",
                  f"{baseline.get('covers_ece_pooled', 0):.4f}")
        st.caption(
            "Research-pinned run-line covers baseline (spread shrink "
            "d_spread 0.3075, sealed ECE 0.1145 → 0.0714). NFL run lines "
            "are integer margins with a shared push band at margin == L; "
            "the ±0.5 stop's raw-vs-derived pair (ties excluded/included) "
            "is the NFL-specific convention.")
    if prov:
        st.caption("Provenance: " + " · ".join(str(p) for p in prov))
    st.caption("All figures research-pinned pooled-OOF — nothing fabricated "
               "until decided slate rows ship.")


def _render_history_table(df: pd.DataFrame, line_kind: str, side_filter,
                          start_d, end_d, cal_note: str) -> None:
    """The MLB-schema history table (DATE | MATCHUP | SCORE (A–H) | LINE |
    MODEL PICK | WINNER | RESULT) over NFL decided rows. LINE and the pick
    come from the model FAIR lines only (never the offered columns); whole-
    number-line pushes resolve 3-way and are excluded from the result ✓/✗
    (the NFL integer convention, same as MLB's whole-line notes).
    ``line_kind`` is 'totals' (O/U at the fair total) or 'runline' (the
    run-line pair at the fair home spread)."""
    view = df.copy()
    view["_d"] = pd.to_datetime(
        view.get("game_date", view.get("gameday")), errors="coerce")
    view = view[(view["_d"].notna()) & (view["_d"].dt.date >= start_d)
                & (view["_d"].dt.date <= end_d)]
    view = view.sort_values("_d", ascending=False)
    if not len(view):
        st.info("No games in the selected date range / side.")
        return
    rows = []
    n_pushes = 0
    for _, r in view.iterrows():
        away = str(r.get("away_team", "") or "")
        home = str(r.get("home_team", "") or "")
        h_s, a_s = r.get("home_score"), r.get("away_score")
        score = ("—" if (pd.isna(h_s) or pd.isna(a_s))
                 else f"{int(a_s)}–{int(h_s)}")
        winner = "—"
        if pd.notna(h_s) and pd.notna(a_s):
            winner = home if h_s > a_s else (away if a_s > h_s else "Tie")
        if line_kind == "totals":
            U = int(round(sv._f(r, "fair_total") or 0))
            po = sv.price_total(r, U)[0]
            over = po is not None and po > 0.5
            line_txt, pick_txt = f"O/U {U}", "—"
            correct = None
            if po is not None:
                pick_txt = f"{'Over' if over else 'Under'} {U} ({po:.0%})"
            if pd.notna(h_s) and pd.notna(a_s) and not pd.isna(po):
                tot = int(h_s) + int(a_s)
                if tot == U:
                    n_pushes += 1
                else:
                    correct = over == (tot > U)
        else:
            L = int(round(sv._f(r, "fair_spread") or 0))
            ph = sv.price_spread(r, L)[0]
            fav_home = ph is not None and ph > 0.5
            line_txt = (f"RL {home} {-L:+d} / {away} {L:+d}"
                        if L >= 0 else f"RL {home} {L:+d} / {away} {-L:+d}")
            pick_txt, correct = "—", None
            if ph is not None:
                pick_txt = (f"{home if fav_home else away} "
                            f"{-L:+d} ({ph:.0%})")
            if pd.notna(h_s) and pd.notna(a_s) and not pd.isna(ph):
                margin = int(h_s) - int(a_s)
                if margin == L:
                    n_pushes += 1
                else:
                    correct = (margin > L) == fav_home
        if correct is None:
            res = "<td>—</td>"
        elif correct:
            res = "<td style='color:#10B981;font-weight:700;'>✓</td>"
        else:
            res = "<td style='color:#EF4444;font-weight:700;'>✗</td>"
        rows.append(
            f"<tr><td>{r['_d'].strftime('%b %d, %Y')}</td>"
            f"<td>{away} @ {home}</td>"
            f"<td>{score}</td>"
            f"<td>{line_txt}</td>"
            f"<td>{pick_txt}</td>"
            f"<td>{winner}</td>{res}</tr>")
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <div style="max-height:480px;overflow-y:auto;">
            <table class="fb-table">
              <thead><tr><th>{HISTORY_HEADERS}</th></tr></thead>
              <tbody>{''.join(rows)}</tbody>
            </table>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    push_txt = (f" · {n_pushes:,} push(es) excluded — whole-number "
                "line, neither wins nor loses" if n_pushes else "")
    st.caption(f"{len(view):,} games · most recent first — scroll for older "
               f"results{push_txt}{cal_note}")


def run() -> None:
    """Render the NFL Totals & Run Lines page — the MLB markets-page mirror
    (sections in the same order: title, load, Diagnostics, Prediction
    History, Monitor). Market-free: model fair lines and probabilities only."""
    utils.inject_css()

    # Always the most recent run (like the MLB markets page / Calibration):
    # never key to the date picked on Today's Games.
    slate, date = utils.load_nfl_run_engine_markets("nfl")
    monitor = utils.load_nfl_run_engine_monitor("nfl")
    date_str = date if date else FALLBACK_DATE

    st.markdown(
        "<div style='font-size:1.7rem;font-weight:800;color:#E2E8F0;'>"
        "Today's Totals &amp; Run Lines</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='color:#94A3B8;margin:2px 0 14px;'>"
        "Run-engine model diagnostics from the per-side era model + pinned "
        "76×76 joint (DN) over the line grid — per-game projections and "
        "full probabilities live in nfl_run_engine_markets_*.csv.</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Markets load + loud missing/stale warnings (MLB-shaped)
    # ------------------------------------------------------------------
    if slate is None or not len(slate):
        st.warning(
            f"No run-engine markets artifact for {date_str}. "
            f"Attempted file: `nfl-backend/data_delivery/"
            f"nfl_run_engine_markets_{date_str}.csv`. "
            "This page always loads the latest available artifact. "
            "The panel fills after the next pipeline run ships "
            "nfl_run_engine_markets_*.csv."
        )
        slate = pd.DataFrame()
    elif not sv.has_fair_columns(slate):
        st.warning(
            "Markets artifact predates the fair-line grid / slate rows. "
            "Waiting for the next pipeline run."
        )

    decided = _decided_rows(slate)

    # ------------------------------------------------------------------
    # Diagnostics — the SAME five tabs as MLB, research-pinned baseline
    # as the honest stand-in until runs ship decided games.
    # ------------------------------------------------------------------
    st.markdown("### Diagnostics")
    if decided.empty:
        st.warning(
            "No decided OOF rows in nfl_run_engine_markets for this date — "
            "diagnostics need outcomes. They appear after a run that ships "
            "decided games; nothing is fabricated in the meantime."
        )
    else:
        st.caption(
            f"{len(decided):,} decided slate rows — real per-line-type "
            "calibration renders below."
        )
    baseline = _baseline(monitor)
    prov = _provenance(monitor)
    _tabs = st.tabs(DIAG_TABS)
    for i, tab in enumerate(_tabs):
        with tab:
            _render_baseline_tab(baseline, i, prov)

    # ------------------------------------------------------------------
    # Prediction History — the MLB structure (date range + side filter +
    # the exact fb-table schema) over NFL decided rows.
    # ------------------------------------------------------------------
    st.markdown("### Prediction History — Totals & Run Lines")
    if decided.empty:
        st.info(
            "No decided OOF rows in the run-engine markets artifact — the "
            "totals/run-line history fills after a run that ships decided "
            "games. Nothing is fabricated in the meantime."
        )
    else:
        lo, hi = None, None
        dts = pd.to_datetime(decided.get("game_date") if
                             "game_date" in decided.columns
                             else decided.get("gameday"), errors="coerce")
        dts = dts.dropna()
        if len(dts):
            lo, hi = dts.min().date(), dts.max().date()
        fc1, fc2, _ = st.columns([1, 1, 2])
        start_d = fc1.date_input("History start date", value=lo,
                                 min_value=lo, max_value=hi) if lo else None
        end_d = fc2.date_input("History end date", value=hi,
                               min_value=lo, max_value=hi) if hi else None
        if start_d and end_d and start_d > end_d:
            start_d, end_d = end_d, start_d
        _cal_note = (" · probabilities are RAW (no calibration map shipped "
                     "in the run-engine artifact)")
        st.markdown("#### Game Totals — Prediction History")
        side = st.radio("Side filter", ["All", "Over", "Under"],
                        index=0, horizontal=True, key="nfl_totals_history_side")
        _render_history_table(decided, "totals",
                              side if side != "All" else None,
                              start_d, end_d, _cal_note)
        st.markdown("#### Run Lines — Prediction History")
        _render_history_table(decided, "runline", None,
                              start_d, end_d, _cal_note)

    # ------------------------------------------------------------------
    # Run-Line & Totals Monitor — research-pinned OOF baseline (the NFL
    # monitor's winner-card equivalent) + accumulating slate history.
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Run-Line & Totals Monitor")
    st.caption(
        "Run-engine winner cards (over/under, run line, derived ML) + "
        "distributional fit + rolling history from "
        "nfl_run_engine_monitor_*.json. **Honesty note:** run-engine "
        "calibration is weaker than the moneyline monitor's — this is the "
        "actual calibration quality; no styling hides it."
    )
    if monitor is None:
        st.warning(
            f"No run-engine monitor artifact for {date_str}. "
            f"Attempted file: `nfl-backend/data_delivery/"
            f"nfl_run_engine_monitor_{date_str}.json`. "
            "The monitor fills after the next pipeline run ships "
            "nfl_run_engine_monitor_*.json."
        )
    else:
        # Research-pinned OOF baseline — the winner-card equivalent (the
        # first-run monitor ships no per-card holdout data, only the pinned
        # pooled-OOF figures with provenance).
        if baseline:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Covers ECE (pooled OOF)",
                      f"{baseline.get('covers_ece_pooled', 0):.3f}")
            c2.metric("Totals ECE (own line, pooled OOF)",
                      f"{baseline.get('totals_ece_pooled_own', 0):.3f}")
            dm = baseline.get("derived_ml") or {}
            c3.metric("Derived-ML logloss", f"{dm.get('logloss', 0):.4f}")
            c4.metric("Derived-ML AUC", f"{dm.get('auc', 0):.3f}")
            c5.metric("Derived-ML ECE", f"{dm.get('ece', 0):.3f}")
            st.caption(
                "**Research-pinned pooled-OOF baseline** — from the "
                "committed era / market / adoption records, not a slate run. "
                "The accumulating slate history below replaces it as runs "
                "ship decided games."
            )
            if prov:
                st.caption("Provenance: " + " · ".join(str(p) for p in prov))
        else:
            st.info("No research-pinned OOF baseline in the monitor artifact "
                    "— a slate-serve run will publish "
                    "nfl_run_engine_monitor_*.json with the figures.")

        # Slate history (the MLB rolling-history convention: first run is
        # empty by design — nothing fabricated).
        hist = monitor.get("slate_history") or []
        with st.expander("Slate History (accumulating runs)", expanded=False):
            if not hist:
                st.info("No slate history yet (first build starts empty).")
            else:
                st.dataframe(pd.DataFrame(hist), use_container_width=True,
                             hide_index=True)


if __name__ == "__main__":
    run()