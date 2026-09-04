"""NFL Today's Totals & Run Lines — STRUCTURAL 1:1 MIRROR of the MLB markets
page (``frontend/markets.py``), with NFL artifacts substituted.

ANATOMY CHECKLIST (MLB render order — this page mirrors it exactly; the
parity tests pin the same order/labels/wording/table schemas):

  1. Title "Today's Totals & Run Lines" + subtitle caption (MLB's sentence
     shape, NFL-accurate model description).
  2. Markets CSV load (always the most recent run) + loud missing/stale
     warnings (``nfl_run_engine_markets_*`` substituted for MLB's).
  3. "### Diagnostics" — decided rows: empty → MLB's OWN no-data code path
     (a single warning, NO tabs — never a custom panel); decided rows →
     the SAME five tabs as MLB (Distribution / Relativized / Pooled lines /
     Game Total Lines / Run Lines) fed by REAL per-game calibration from
     the decided OOF store (``nfl_market_diagnostics`` mirrors MLB's
     ``market_diagnostics`` API on the integer-support NFL grids). No
     research-pinned stand-in tabs — that bespoke panel is REMOVED.
  4. "### Prediction History — Totals & Run Lines" — empty → MLB's wording
     info; decided rows → the same date-range + side-filter widgets and
     the same fb-table schema (DATE | MATCHUP | SCORE (A–H) | LINE |
     MODEL PICK | WINNER | RESULT) over the NFL columns.
  5. "---" + "### Run-Line & Totals Monitor" + the honesty caption →
     winner cards (computed from the decided OOF store, the NFL monitor's
     card-equivalent data), the Calibration Cards expander (Totals +
     Run Line, MLB widget set), the fit/drift/coverage/model-card sections
     (MLB's own empty-state wording where the NFL artifact does not yet
     carry those pipeline sections), and the rolling-history expander
     (first-run empty state, MLB wording).
  6. NO per-game projections hero on this page — MLB does not render
     per-game content on the markets page; per-game pricing lives in the
     artifact and on the Today's Games cards.

MARKET-FREE BY POLICY (unchanged): this page renders model fair lines and
model probabilities ONLY. Offered/book lines, shrink columns and
market-derived "edge" never render — the dashboard is a model product, not
a market tool. The artifact keeps those columns (test-pinned); the page
never reads them.

LEGITIMATE NFL WORDING DELTAS (documented; MLB wording is the default
elsewhere):
  * Push vs tie: NFL integer grids carry a REAL push band (P(margin == L),
    P(total == U)) and whole-number lines push — excluded 2-way exactly
    like MLB's own whole-line notes.
  * The ±0.5 stop's raw-vs-derived pair (raw −0.5 excludes ties, raw +0.5
    includes them, derived ML normalizes them out) lives on the Today's
    Games cards, not on this page.
  * Subtitle: the NFL run engine is the per-side era model + pinned 76×76
    joint (DN) — not MLB's NB(λ, α(λ)) sampler; the sentence shape is
    identical.
  * Monitor: MLB's fit panel / drift / coverage / model card render from
    pipeline sections the NFL pipeline does not yet emit — the NFL page
    mirrors MLB's own "no data in this monitor artifact" wording for those
    sections (nothing fabricated), while the winner cards + calibration
    cards + rolling history render from the NFL decided store.

Import is side-effect-free (render happens only inside ``run()``), so
tests can import the module without a Streamlit page context.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import nfl_market_diagnostics as nd
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

# The three winner cards (MLB _WINNER_CARDS structure, NFL rules).
_WINNER_CARDS = (
    ("over_under", "Over/Under",
     "Pick Over if P(over the game's own fair total) > 50%, else Under"),
    ("run_line", "Run Line (favorite)",
     "Pick the favorite side to cover at its own fair run line"),
    ("derived_ml", "Derived ML (run-line model moneyline)",
     "Pick the side with P > 50% — home if P(home win) > 50%, else away "
     "(P(H>A)/(1−P(tie)) of the calibrated 76×76 joint)"),
)


def _fmt(v, nd_=3, pct=False) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "--"
    return f"{v * 100:.0f}%" if pct else f"{v:.{nd_}f}"


def _ec_indicator(raw, cal) -> str:
    """MLB's exact ECE indicator: green < 0.01, yellow < 0.02, red
    >= 0.02 (identical thresholds + emoji — no styling hides weak
    calibration)."""
    val = cal if cal is not None else raw
    if val is None:
        return "⚪"
    if val < 0.01:
        return "🟢"
    if val < 0.02:
        return "🟡"
    return "🔴"


def _decided_rows(df: pd.DataFrame | None) -> pd.DataFrame:
    """The decided OOF rows (kind == 'oof') with outcomes — the artifact's
    analog of MLB's decided_rows."""
    if df is None or not len(df):
        return pd.DataFrame()
    return nd.decided_rows(df)


# ---------------------------------------------------------------------------
# Diagnostics tabs (real data)
# ---------------------------------------------------------------------------
def _render_distribution_tab(decided: pd.DataFrame) -> None:
    dist = nd.total_distribution(decided)
    if dist["warning"]:
        st.warning(dist["warning"])
        return
    utils.show_chart(nd.chart_distribution(dist))
    c = dist["callouts"]
    st.caption(
        f"Observed bars vs modeled mean per-game total PMF (average of the "
        f"calibrated 76×76 joints' total marginals) over {dist['n_games']:,} "
        f"decided games · P(total≤35): observed {c['P(total<=35)']['observed']:.3f} "
        f"/ modeled {c['P(total<=35)']['modeled']:.3f} · P(total≥60): "
        f"observed {c['P(total>=60)']['observed']:.3f} / modeled "
        f"{c['P(total>=60)']['modeled']:.3f}. Integer-support means the "
        "whole-number mass IS the push band."
    )


def _render_relativized_tab(decided: pd.DataFrame) -> None:
    pairs = nd.relativized_pairs(decided)
    curve = nd.calibration_curve(pairs)
    if curve["warning"]:
        st.warning(curve["warning"])
        return
    utils.show_chart(nd.chart_calibration(
        curve, "Relativized offsets −3 … +3"))
    xs = [b["mean_pred"] for b in curve["bins"]]
    st.caption(
        f"Each game priced at ITS OWN fair total ± offset "
        f"({', '.join(f'{o:+d}' for o in nd.OFFSET_EDGES)}); lines priced by "
        f"the integer grid. {curve['n_pairs']:,} pairs, {len(xs)} valid bins "
        f"(≥30 each, {curve['n_dropped_bins']} dropped) · predicted range "
        f"{min(xs):.2f}–{max(xs):.2f} — the spread is the point; the dashed "
        "diagonal is perfect calibration."
    )


def _render_pooled_tab(decided: pd.DataFrame) -> None:
    fpairs = nd.fixed_line_pairs(decided, nd.FIXED_TOTAL_LINES)
    fcurve = nd.calibration_curve(fpairs)
    if fcurve["warning"]:
        st.warning(fcurve["warning"])
        return
    utils.show_chart(nd.chart_calibration(
        fcurve, "Games pooled across 38 / 42 / 46 / 50 / 54"))
    st.caption(
        "**How to read this:** every game is priced at FIVE integer lines "
        "(38 / 42 / 46 / 50 / 54) so the predictions spread across the full "
        "probability range — a single game contributes 5 pairs. X = the "
        "model's predicted P(over); Y = how often the over actually hit "
        "among the pairs in that bin. The dashed diagonal is perfect "
        f"calibration. ({fcurve['n_pairs']:,} pairs, but NOT independent: "
        "each game appears 5×, so the shape is trustworthy while the "
        "effective sample is the ~1,376 decided games themselves.)"
    )


def _render_gtl_tab(decided: pd.DataFrame) -> None:
    _gl_sel = st.selectbox(
        "Line (All = own fair line)",
        ["All"] + [str(l) for l in nd.TOTAL_GRID],
        index=1 + nd.TOTAL_GRID.index(nd.DEFAULT_TOTAL),
        key="nfl_diag_gtl_line")
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
        x_label="Mean Predicted", series_label="Mean Actual",
        obs_label="Mean Actual")
    utils.show_chart(built["chart"])
    st.table(built["table"])
    priced_txt = ("decided games priced at their own fair totals"
                  if _gl_line is None else
                  f"decided games priced at line {_gl_line:g}")
    st.caption(
        f"{glc['n_games']:,} {priced_txt} · bar heights = games priced in "
        f"that predicted-P(over) band (LEFT 'Games' axis) · observed and "
        f"win-rate curves (RIGHT '%' axis) = how often those games hit, on "
        f"the 2-way no-push basis · {glc['n_pushes']:,} pushes excluded "
        f"({glc['push_rate']:.1%}, whole lines only — total == line, neither "
        f"wins nor losses) · % of Total = count_bin / count_total × 100 · "
        f"pooled predicted {glc['pooled_pred']:.2f} vs pooled observed "
        f"{glc['pooled_observed']:.2f} · pooled win rate "
        f"{glc['pooled_winrate']:.1%} · pooled ECE {glc['pooled_ece']:.3f} · "
        f"pooled Brier {glc['pooled_brier']:.3f} · pooled AUC "
        f"{glc['pooled_auc']:.3f} (roc_auc over ALL decided no-push games "
        "at this line). Per-bin AUC is degenerate (~0.5) by construction — "
        "predictions are rank-compressed inside a narrow band, so read it "
        "as a within-bin consistency check, not discrimination power; bins "
        "with n < 30 or a single outcome class show blank. "
        + ("ALL: each game priced at ITS OWN fair total (the integer median "
           "of the total PMF) — predicted = the re-scaled 2-way P(over) the "
           "Today's Games card shows at its default line, so the band hugs "
           "50% by construction; observed = over rate."
           if _gl_line is None else
           f"FIXED LINE {_gl_line:g}: all games at one line — the predicted "
           "spread IS the calibration surface; observed = over frequency; "
           "5-pt bins line the 0–1 axis.")
        + " The dashed diagonal is perfect calibration: points on it mean "
        "the model's probabilities are honest at every level, not just near "
        "50%. The win-rate line is the picked-side W/(W+L) (over if P(over) "
        "> 50%; below 50% the under pick flips it — a 'V' around 50%). The "
        "last table row is the pooled Total (share 100%, the amber diamond "
        "on the chart). Gray bars + dropped curve points mark buckets with "
        "n < 30 (low sample — not reliable calibration evidence)."
    )


def _render_rl_tab(decided: pd.DataFrame) -> None:
    _rl_sel = st.selectbox(
        "Line (favorite −L)",
        ["All"] + [str(l) for l in nd.RUN_LINE_MAGS],
        index=1 + nd.RUN_LINE_MAGS.index(nd.DEFAULT_RUN_MAG),
        key="nfl_diag_run_line")
    _rl_line = None if _rl_sel == "All" else float(_rl_sel)
    rlc = nd.run_line_calibration(decided, _rl_line)
    if rlc["warning"]:
        st.warning(rlc["warning"])
        return
    if _rl_line is None:
        _rl_title = "Calibration Curve — Favorite (All = own fair run line)"
    else:
        _rl_title = f"Calibration Curve — Favorite −{_rl_line:g}"
    built = nd.chart_game_total_curve(
        rlc, _rl_title, curve_bins=rlc.get("curve_bins"),
        x_tick_values=nd.X_1PCT_TICKS, show_win_rate=False,
        x_label="Mean Predicted", series_label="Mean Actual",
        obs_label="Mean Actual")
    utils.show_chart(built["chart"])
    st.table(built["table"])
    priced_txt = ("decided games priced at their own fair run lines"
                  if _rl_line is None else
                  f"decided games priced at run line −{_rl_line:g}")
    st.caption(
        f"{rlc['n_games']:,} {priced_txt} · bar heights = games priced in "
        f"that predicted P(cover) band (LEFT 'Games' axis) · observed and "
        f"win-rate curves (RIGHT '%' axis) = how often the favorite side "
        f"covered, on the 2-way no-push basis · {rlc['n_pushes']:,} pushes "
        f"excluded ({rlc['push_rate']:.1%}, whole lines only — the favorite "
        "loses by exactly the line, neither wins nor losses) · % of Total = "
        f"count_bin / count_total × 100 · pooled predicted "
        f"{rlc['pooled_pred']:.2f} vs pooled observed "
        f"{rlc['pooled_observed']:.2f} · pooled win rate "
        f"{rlc['pooled_winrate']:.1%} · pooled ECE {rlc['pooled_ece']:.3f} · "
        f"pooled Brier {rlc['pooled_brier']:.3f} · pooled AUC "
        f"{rlc['pooled_auc']:.3f} (roc_auc over ALL decided no-push games "
        "at this line). Per-bin AUC is degenerate (~0.5) by construction — "
        "predictions are rank-compressed inside a narrow band, so read it "
        "as a within-bin consistency check, not discrimination power; bins "
        "with n < 30 or a single outcome class show blank. "
        + ("ALL: each game priced at ITS OWN fair run line (the integer "
           "median of the margin PMF, favorite-anchored) — predicted = the "
           "2-way P(cover) the favorite side is quoted at, so the band hugs "
           "50% by construction; observed = favorite-cover rate."
           if _rl_line is None else
           f"FIXED LINE −{_rl_line:g}: all games at one line — the predicted "
           "spread IS the calibration surface; observed = favorite-cover "
           "frequency; 5-pt bins line the 0–1 axis.")
        + " The dashed diagonal is perfect calibration: points on it mean "
        "the model's probabilities are honest at every level, not just near "
        "50%. The win-rate line is the picked-side W/(W+L) (pick the "
        "favorite to cover if P(cover) > 50%; below 50% the dog pick flips "
        "it — a 'V' around 50%). The last table row is the pooled Total "
        "(share 100%, the amber diamond on the chart). Gray bars + dropped "
        "curve points mark buckets with n < 30 (low sample — not reliable "
        "calibration evidence)."
    )


# ---------------------------------------------------------------------------
# Prediction History (MLB-schema tables over NFL decided rows)
# ---------------------------------------------------------------------------
def _render_history_table(df: pd.DataFrame, line_kind: str, start_d, end_d,
                          cal_note: str) -> None:
    """The MLB-schema history table (DATE | MATCHUP | SCORE (A–H) | LINE |
    MODEL PICK | WINNER | RESULT) over NFL decided rows. LINE and the pick
    come from the model FAIR lines only (never the offered columns); whole-
    number-line pushes resolve 3-way and are excluded from the result ✓/✗
    (the NFL integer convention)."""
    view = df.copy()
    view["_d"] = pd.to_datetime(view.get("gameday"), errors="coerce")
    view = view[(view["_d"].notna()) & (view["_d"].dt.date >= start_d)
                & (view["_d"].dt.date <= end_d)]
    view = view.sort_values("_d", ascending=False)
    if not len(view):
        st.info("No games in the selected date range.")
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


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------
def _render_winner_cards(cards: dict) -> None:
    items = [(k, label, rule) for k, label, rule in _WINNER_CARDS
             if k in cards]
    if not items:
        st.info("No winner-card data in this monitor artifact.")
        return
    cols = st.columns(len(items))
    for i, (key, label, rule) in enumerate(items):
        card = cards[key]
        h = card.get("holdout") or {}
        badge = _ec_indicator(card.get("ece_raw"),
                              card.get("ece_calibrated"))
        with cols[i]:
            st.markdown(f"**{label}** {badge}")
            st.caption(rule)
            awr = card.get("actual_win_rate")
            pm = card.get("predicted_mean")
            if awr is not None and pm is not None:
                compact = (f"Actual {awr * 100:.1f}% · "
                           f"Predicted {pm * 100:.1f}%")
            else:
                compact = "Actual -- · Predicted --"
            st.markdown(f"**{compact}**")
            c1, c2 = st.columns(2)
            c1.metric("Win rate", _fmt(card.get("win_rate"), pct=True))
            c2.metric("AUC", _fmt(card.get("auc"), 4))
            c1, c2 = st.columns(2)
            c1.metric("Pooled ECE-cal", _fmt(card.get("ece_calibrated")))
            c2.metric("Holdout ECE-cal",
                      _fmt(h.get("ece_calibrated")) if h else "--")
            c1, c2 = st.columns(2)
            c1.metric("Pooled Brier", _fmt(card.get("brier")))
            c2.metric("Pooled Logloss", _fmt(card.get("logloss"), 4))
            c1, c2 = st.columns(2)
            c1.metric("Holdout AUC", _fmt(h.get("auc"), 4) if h else "--")
            n_pooled = card.get("n")
            n_txt = (f"{n_pooled:,}" if isinstance(n_pooled, int)
                     else str(n_pooled or "--"))
            h_n = (h or {}).get("n")
            h_txt = (f"{h_n:,}" if isinstance(h_n, int) else str(h_n or 0))
            st.caption(f"n = {n_txt} pooled"
                       + (f" / {h_txt} holdout" if h else ""))


def _render_totals_calibration_card(decided: pd.DataFrame) -> None:
    st.markdown("**Totals — Over/Under calibration card**")
    c1, c2 = st.columns([1, 1])
    thresh = c1.selectbox("Confidence (pick_prob > …)",
                          [0, 50, 55, 60, 65, 70], index=0,
                          key="nfl_totals_card_conf")
    side = c2.selectbox("Side", ["All", "Over", "Under"], index=0,
                        key="nfl_totals_card_side")
    s = nd.totals_monitor_stats(decided, min_pct=float(thresh), side=side)
    if not s["n"]:
        st.caption("No priced games at this confidence / side.")
        return
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Games", f"{s['n']:,}")
    wr = s["win_rate"]
    m2.metric("Win rate (W/(W+L))", f"{wr * 100:.1f}%" if wr else "—")
    m3.metric("Wins / Losses", f"{s['n_wins']:,} / {s['n_losses']:,}")
    m4.metric("Pushes excluded", f"{s['n_pushes']:,}")
    if s["sides"]:
        rows = []
        for name, r in s["sides"].items():
            rw = r["win_rate"]
            rows.append({
                "Side": name,
                "n": r["n"],
                "Win rate": f"{rw * 100:.1f}%" if rw else "—",
                "Pushes": r["n_pushes"],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)
    st.caption(
        f"Thresh: pick_prob > {thresh}% (cumulative — nested subsets). "
        "Win rate is 2-way re-normalized: whole-number-line pushes are "
        "neither wins nor losses (excluded from both). This side split is "
        "the Scoring-Mean diagnostic — compare each side vs its own mean "
        "predicted P."
    )


def _render_runline_calibration_card(decided: pd.DataFrame) -> None:
    st.markdown("**Run Line — favorite cover calibration card**")
    line = st.selectbox("Line (favorite −L)",
                        [str(m) for m in nd.RUN_LINE_MAGS],
                        index=nd.RUN_LINE_MAGS.index(nd.DEFAULT_RUN_MAG),
                        key="nfl_runline_card_line")
    mag = float(line)
    s = nd.runline_monitor_stats(decided, mag)
    if not s["n"]:
        st.caption("No priced games at this line.")
        return
    wr = s["win_rate"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Games", f"{s['n']:,}")
    m2.metric("Model predicted", "--")   # 2-way pred needs the card stat
    m3.metric("Win rate (W/(W+L))", f"{wr * 100:.1f}%" if wr else "—")
    m4.metric("Wins / Losses", f"{s['n_wins']:,} / {s['n_losses']:,}")
    m5.metric("Pushes", f"{s['n_pushes']:,}")
    if s["sides"]:
        rows = []
        for name, r in s["sides"].items():
            rw = r["win_rate"]
            rows.append({"Favorite": "Home" if name == "home" else "Away",
                         "n": r["n"],
                         "Win rate": f"{rw * 100:.1f}%" if rw else "—",
                         "Pushes": r["n_pushes"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)
    st.caption(
        f"Favored side = derived-ML favorite (P(home win) > 50%). At deeper "
        "lines its cover P can fall below 50% — that is a calibration "
        "finding, not a pick rule. Win rate is 2-way re-normalized "
        "(whole-line pushes, favored margin == ±m, folded out of both "
        "sides)."
    )


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
    # Diagnostics — the SAME five tabs as MLB, fed by REAL data when
    # decided rows exist; MLB's own no-data path when they don't.
    # ------------------------------------------------------------------
    st.markdown("### Diagnostics")
    if decided.empty:
        st.warning(
            "No decided OOF rows in nfl_run_engine_markets for this date — "
            "diagnostics need outcomes. They appear after a run that ships "
            "decided games; nothing is fabricated in the meantime."
        )
    else:
        _tabs = st.tabs(DIAG_TABS)
        with _tabs[0]:
            _render_distribution_tab(decided)
        with _tabs[1]:
            _render_relativized_tab(decided)
        with _tabs[2]:
            _render_pooled_tab(decided)
        with _tabs[3]:
            _render_gtl_tab(decided)
        with _tabs[4]:
            _render_rl_tab(decided)

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
        dts = pd.to_datetime(decided.get("gameday"), errors="coerce").dropna()
        lo, hi = dts.min().date(), dts.max().date()
        fc1, fc2, _ = st.columns([1, 1, 2])
        start_d = fc1.date_input("History start date", value=lo,
                                 min_value=lo, max_value=hi)
        end_d = fc2.date_input("History end date", value=hi,
                               min_value=lo, max_value=hi)
        if start_d > end_d:
            start_d, end_d = end_d, start_d
        _cal_note = (" · probabilities are RAW (no calibration map shipped "
                     "in the run-engine artifact)")
        st.markdown("#### Game Totals — Prediction History")
        st.radio("Side filter", ["All", "Over", "Under"], index=0,
                 horizontal=True, key="nfl_totals_history_side")
        _render_history_table(decided, "totals", start_d, end_d, _cal_note)
        st.markdown("#### Run Lines — Prediction History")
        _render_history_table(decided, "runline", start_d, end_d, _cal_note)

    # ------------------------------------------------------------------
    # Run-Line & Totals Monitor — winner cards + calibration cards +
    # fit/drift/coverage/model-card empty states + rolling history.
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
        persisted = monitor.get("markets_persisted", True)
        if not persisted:
            st.error(
                f"**Markets CSV was NOT persisted for {date_str}**: "
                f"{monitor.get('markets_persist_error') or 'unknown error'}. "
                "The monitor data below is from the in-memory summary; "
                "downstream consumers should not trust the markets artifact."
            )
        # Winner cards — computed from the decided OOF store (the NFL
        # monitor's card-equivalent data; nothing fabricated).
        cards = nd.winner_cards(decided) if not decided.empty else {}
        if cards:
            _render_winner_cards(cards)
        else:
            st.info("No winner-card data in this monitor artifact.")

        # Interactive calibration cards (unified push resolution) — computed
        # from the decided OOF rows of the markets artifact.
        if not decided.empty:
            st.markdown("---")
            with st.expander("Calibration Cards — Totals & Run Line",
                             expanded=True):
                _render_totals_calibration_card(decided)
                st.markdown("<div style='height:12px'></div>",
                            unsafe_allow_html=True)
                _render_runline_calibration_card(decided)

        # Fit panel — the NFL pipeline does not yet emit this section;
        # MLB's own empty-state wording (nothing fabricated).
        with st.expander("Distributional Fit Diagnostics", expanded=False):
            st.info("No fit diagnostics in the monitor artifact.")

        # Drift / coverage / model card — absent until the NFL pipeline
        # emits those sections; MLB's own empty-state wording.
        st.markdown("### Run-Engine Feature Drift (PSI)")
        st.info("No run-engine drift data for this date — the NFL "
                "slate-serve runner does not yet emit run_engine_feature_"
                "drift files.")
        st.markdown("### Run-Engine Feature Coverage (non-null / measured)")
        st.info("No run-engine coverage data for this date — the NFL "
                "slate-serve runner does not yet emit run_engine_feature_"
                "coverage files.")
        st.markdown("### Run-Engine Model (per-side era + 76×76 joint)")
        st.caption("Per-side era model (E2, ewm_2w, median rounds 20/23) + "
                   "pinned DN joint (σ 9.663/9.0789, ρ 0.0076, tie 0.275%) "
                   "— the run-engine OOF metrics live on the Diagnostics "
                   "tabs and the winner cards above.")
        st.info("Per-line engine OOF metrics appear in the Diagnostics tabs "
                "from the decided OOF store.")

        # Rolling history (the MLB convention: first run is empty by design
        # — nothing fabricated).
        hist = monitor.get("slate_history") or []
        with st.expander("Rolling History (last 10 points per card)",
                         expanded=False):
            if not hist:
                st.info("No rolling history yet (first build starts empty).")
            else:
                st.dataframe(pd.DataFrame(hist), use_container_width=True,
                             hide_index=True)


if __name__ == "__main__":
    run()