"""Page 5 — Today's Totals & Run Lines (run engine, Phase 3).

Per slate game: projected total and per-team expected runs, a LINE SELECTOR
for the total (6.5–12.5) showing p_over/p_under at the chosen line, a run-line
selector (−0.5…−3.5) with the favored side's cover probability, and the
derived ML win% cross-referenced against the moneyline ensemble — including a
CONFLICT badge when the two disagree beyond AGREEMENT_FILTER_DELTA.

The toggle reads the precomputed per-line probability GRID straight from
run_engine_markets_<date>.csv — zero frontend math, works offline. Market
calibration at the reference lines, the rolling totals-Brier timeline, the
alpha(lambda) fit-check tails, and the sealed-holdout gate verdict render
from model_monitor_<date>.json. Missing/stale artifacts produce loud warnings
and graceful empty states — never fake data.
"""

from __future__ import annotations

import io

import altair as alt
import pandas as pd
import streamlit as st

import utils

utils.inject_css()

dates = utils.available_dates(**utils.get_source_config())
date_str = st.session_state.get("selected_date", dates[0] if dates else "20260809")
mon = utils.load_model_monitor(date_str)

st.markdown(
    "<div style='font-size:1.7rem;font-weight:800;color:#E2E8F0;'>"
    "Today's Totals &amp; Run Lines</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='color:#94A3B8;margin:2px 0 14px;'>"
    "Run-engine market probabilities from NB(λ, α(λ)) Monte Carlo — pick any "
    "line; the grid is priced by the model, not the market.</div>",
    unsafe_allow_html=True,
)


def _load_markets(ds):
    """Fetch the run-engine markets CSV.  relpath is a bare filename —
    _fetch_bytes prepends the repo subdir + data_delivery/ internally."""
    import logging
    _log = logging.getLogger("markets")
    fname = f"run_engine_markets_{ds}.csv"
    cfg = utils.get_source_config()
    try:
        raw, src = utils._fetch_bytes(fname, **cfg)
    except Exception as exc:
        # Build the attempted URL for actionable diagnostics
        url = utils._raw_url(fname, **cfg)
        _log.error("Markets fetch exception for %s (%s): %s", fname, url, exc)
        st.warning(f"Fetch error for run_engine_markets_{ds}.csv – see log.")
        return None
    if raw is None:
        url = utils._raw_url(fname, **cfg)
        _log.warning("Markets artifact not found: %s (URL: %s, source: %s)",
                     fname, url, src)
        return None
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        _log.error("Markets CSV parse failed for %s: %s", fname, exc)
        return None


markets = _load_markets(date_str)
re_block = (mon or {}).get("run_engine") or {}
if markets is None or not len(markets):
    url = utils._raw_url(f"run_engine_markets_{date_str}.csv",
                         **utils.get_source_config())
    st.warning(
        f"No run-engine markets artifact for {date_str}. "
        f"Attempted URL: `{url}`. "
        "The panel fills after the next pipeline run ships "
        "run_engine_markets_*.csv."
    )
    markets = pd.DataFrame()
elif "kind" not in markets.columns:
    st.warning(
        "Markets artifact predates Phase 3 (no line grid / slate rows). "
        "Waiting for the next pipeline run."
    )

# ---------------------------------------------------------------------------
# Line selectors — read precomputed grid columns, zero math
# ---------------------------------------------------------------------------
TOTAL_LINES = re_block.get("line_grid", {}).get("totals") or [
    round(6.5 + 0.5 * i, 1) for i in range(13)
]
RUN_LINES = re_block.get("line_grid", {}).get("run_lines") or [-0.5, -1.5, -2.5, -3.5]


def _line_key(prefix, line):
    return f"{prefix}_{str(line).replace('.', '_')}"


sel_total = st.slider(
    "Total line (O/U)",
    min_value=float(min(TOTAL_LINES)),
    max_value=float(max(TOTAL_LINES)),
    step=0.5,
    value=8.5 if 8.5 in TOTAL_LINES else float(TOTAL_LINES[len(TOTAL_LINES) // 2]),
    format="%.1f",
)
sel_run = st.select_slider("Run line (home favorite)", options=RUN_LINES,
                           value=-1.5)

over_col = _line_key("p_over", sel_total)
under_col = _line_key("p_under", sel_total)
cover_col = _line_key("p_home_cover", abs(sel_run))

missing_cols = [c for c in (over_col, under_col, cover_col)
                if c not in markets.columns]
if missing_cols and len(markets):
    st.warning(f"Artifact lacks columns {missing_cols} — cannot price that line.")

# ---------------------------------------------------------------------------
# Slate board
# ---------------------------------------------------------------------------
st.markdown("### Slate")
slate = (markets[markets["kind"] == "slate"].copy()
         if "kind" in markets.columns else pd.DataFrame())
if slate.empty:
    st.info(
        "No slate rows yet — today's games are priced during the daily "
        "pipeline run once the schedule and point-in-time features exist."
    )
else:
    delta = float(re_block.get("agreement_delta", 0.08))
    rows, conflicted = [], 0
    for _, g in slate.iterrows():
        p_over = float(g[over_col])
        p_under = float(g[under_col])
        p_cover = float(g[cover_col])
        p_win_h = float(g["p_home_win_derived"])
        ml_p = g.get("ml_win_prob")
        conflict = bool(g.get("agreement_conflict")) and pd.notna(ml_p)
        conflicted += int(conflict)
        home_t = g.get("home_team", "Home")
        away_t = g.get("away_team", "Away")
        fav_team, fav_p = ((home_t, p_cover) if p_cover >= 1 - p_cover
                           else (away_t, 1 - p_cover))
        rows.append({
            "GAME": f"{away_t} @ {home_t}",
            "PROJ TOTAL": round(float(g["home_expected_runs"])
                                + float(g["away_expected_runs"]), 2),
            "L HOME": round(float(g["home_expected_runs"]), 2),
            "L AWAY": round(float(g["away_expected_runs"]), 2),
            f"P(OVER {sel_total})": round(p_over, 3),
            f"P(UNDER {sel_total})": round(p_under, 3),
            "FAVORITE": f"{fav_team} {sel_run:+.1f}".replace("+-", "-"),
            f"P(COVER {sel_run})": round(fav_p, 3),
            "DERIVED ML": f"{100 * p_win_h:.0f}% home",
            "ML ENS": (f"{100 * float(ml_p):.0f}% home"
                       if pd.notna(ml_p) else "-"),
            "CONFLICT": "! CONFLICT" if conflict else "",
        })
    st.table(pd.DataFrame(rows).reset_index(drop=True))
    n_draws = re_block.get("mc_meta", {}).get("n_draws", "-")
    seed = re_block.get("mc_meta", {}).get("seed", "-")
    st.caption(
        f"{len(slate)} slate games · the toggle reads the precomputed grid "
        f"(N={n_draws:,} draws, seed={seed}) · derived ML vs the moneyline "
        f"ensemble; CONFLICT marks |Δp| > ±{delta:.2f}."
    )
    if conflicted:
        st.warning(
            f"{conflicted} game(s) flagged moneyline-vs-run-engine conflicts "
            "— suppressed from any recommendation surface."
        )
    # Suppression hook: downstream recommendation consumers must filter on
    # this session list. No betting engine exists; this is the contract.
    suppressed = (slate[slate["agreement_conflict"] == True]["game_pk"]
                  .astype(int).tolist()
                  if "agreement_conflict" in slate.columns else [])
    st.session_state["suppressed_game_pks"] = suppressed

# ---------------------------------------------------------------------------
# Market calibration + holdout gate
# ---------------------------------------------------------------------------
st.markdown("### Market calibration & holdout gate")
metrics = re_block.get("market_metrics") or {}
ref_rows = []
for name, m in sorted(metrics.items()):
    if not isinstance(m, dict):
        continue
    h = m.get("holdout") or {}
    ref_rows.append({
        "MARKET": name,
        "LOGLOSS": m.get("engine_logloss"),
        "BRIER": m.get("engine_brier"),
        "ECE CAL": m.get("engine_ece_calibrated"),
        "BASE LL": m.get("baseline_logloss"),
        "HOLDOUT LL": h.get("engine_logloss"),
        "HOLDOUT BASE LL": h.get("baseline_logloss"),
        "BEATS BASE?": (
            "YES" if h.get("beats_baseline_logloss") is True
            else ("NO" if h.get("beats_baseline_logloss") is False else "-")
        ),
    })
if ref_rows:
    st.table(pd.DataFrame(ref_rows).reset_index(drop=True))
    gate = re_block.get("holdout_gate") or {}
    all_beat = gate.get("totals_beat_baselines_holdout")
    verdict = ("YES — totals beat their baselines out-of-sample" if all_beat
               else ("NO — totals failed to beat baselines out-of-sample"
                     if all_beat is False else "pending"))
    st.caption(
        f"Sealed-holdout gate: last {gate.get('n_holdout', '-')} games "
        f"(since {gate.get('cutoff', '-')}) were never used for alpha "
        f"fitting — {verdict}. Reference lines 7.5/8.5/9.5 and -1.5/-2.5."
    )
else:
    st.info("Market metrics appear after the next pipeline run embeds them.")

# Rolling totals Brier — mirrors the moneyline rolling Brier panel.
rtb_block = re_block.get("rolling_totals_brier") or {}
rtb = rtb_block.get("series") or []
if rtb:
    bdf = pd.DataFrame(rtb)
    bdf["date"] = pd.to_datetime(bdf["date"])
    chart = alt.Chart(bdf).mark_line(color="#F59E0B", strokeWidth=2.2).encode(
        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %d")),
        y=alt.Y("brier:Q", title="Totals Brier (trailing 30d)",
                scale=alt.Scale(zero=False)),
    ).properties(height=260)
    utils.show_chart(chart)
    hb = rtb_block.get("history_mean_brier")
    if isinstance(hb, (int, float)):
        st.caption(f"Trailing-30-day Brier of P(over 8.5) on decided OOF "
                   f"games · history mean {hb:.4f}.")
else:
    st.info("Rolling totals Brier appears once decided OOF markets ship.")

# Alpha(lambda) fit-check tails vs the Phase-2 single-alpha baseline.
fc3 = re_block.get("fit_check_alpha_lambda") or {}
fc2 = re_block.get("fit_check_single_alpha") or {}
if fc3 and fc2:
    tail_rows = []
    for side in ("home", "away"):
        single = {r["k"]: r for r in fc2.get(side, []) if isinstance(r["k"], str)}
        for r in fc3.get(side, []):
            k = r.get("k")
            if isinstance(k, str) and k.startswith("≥"):
                tail_rows.append({
                    "SIDE": side,
                    "TAIL": k,
                    "SINGLE A": single.get(k, {}).get("modeled_p"),
                    "A(LAMBDA)": r.get("modeled_p"),
                    "OBSERVED": r.get("observed_p"),
                })
    if tail_rows:
        st.markdown("#### Blowout-tail fit check (single alpha → alpha(lambda))")
        st.table(pd.DataFrame(tail_rows).reset_index(drop=True))
        var_bits = []
        for s in ("home", "away"):
            v = (re_block.get("variance_check") or {}).get(s)
            if v:
                var_bits.append(
                    f"{s}: implied {v['implied_var']} vs observed "
                    f"{v['observed_var']}")
        st.caption(
            "Modeled share of games with >=10/>=11/>=12 runs per team. The "
            "Phase-2 single global alpha underestimated these tails; "
            "alpha(lambda) should track OBSERVED. Variance check — "
            + "; ".join(var_bits) + "."
        )

# ---------------------------------------------------------------------------
# Diagnostics — six charts, computed by frontend/market_diagnostics.py
# (pure functions over the artifact; the render layer only draws).
# ---------------------------------------------------------------------------
import market_diagnostics as diag  # noqa: E402  (page module, imported late)

decided = diag.decided_rows(markets)
st.markdown("### Diagnostics")
if decided.empty:
    st.warning(
        "No decided OOF rows in run_engine_markets for this date — "
        "diagnostics need outcomes. They appear after a run that ships "
        "decided games; nothing is fabricated in the meantime."
    )
else:
    _tabs = st.tabs([
        "Distribution", "Relativized", "Pooled lines",
        "Money line 8.5", "Overs picks", "Run-line picks",
    ])

    with _tabs[0]:   # 1 — totals distribution fit-check
        dist = diag.total_distribution(decided)
        if dist["warning"]:
            st.warning(dist["warning"])
        else:
            utils.show_chart(diag.chart_distribution(dist))
            c = dist["callouts"]
            st.caption(
                f"Observed bars vs modeled mean NB(λ, α(λ)) convolution over "
                f"{dist['n_games']:,} decided games · P(total≤1): observed "
                f"{c['P(total<=1)']['observed']:.3f} / modeled "
                f"{c['P(total<=1)']['modeled']:.3f} · P(total≥10): observed "
                f"{c['P(total>=10)']['observed']:.3f} / modeled "
                f"{c['P(total>=10)']['modeled']:.3f}. Per-TEAM tail checks are "
                "the fit-check table above — this is the totals law."
            )

    with _tabs[1]:   # 2 — relativized-offset calibration (primary curve)
        pairs = diag.relativized_pairs(decided)
        curve = diag.calibration_curve(pairs)
        if curve["warning"]:
            st.warning(curve["warning"])
        else:
            utils.show_chart(diag.chart_calibration(
                curve, "Relativized offsets −2.0 … +2.0"))
            xs = [b["mean_pred"] for b in curve["bins"]]
            st.caption(
                f"Each game priced at ITS own expected-total ± offset "
                f"({', '.join(f'{o:+.1f}' for o in diag.OFFSET_EDGES)}); lines "
                f"priced by monotone logit interpolation of the grid (documented "
                f"in market_diagnostics). {curve['n_pairs']:,} pairs, "
                f"{len(xs)} valid bins (≥30 each, {curve['n_dropped_bins']} "
                f"dropped) · predicted range {min(xs):.2f}–{max(xs):.2f} — the "
                "spread is the point; the dashed diagonal is perfect calibration."
            )
            st.warning(
                "**Known limitation — deep-over region:** At lines well below "
                "expected total (offset ≤ −2.0), p_over is high but the model "
                "is overconfident. Measured gap: calibrated prediction ≈ 0.66 "
                "vs actual ≈ 0.58 (≈ 0.08 shortfall, n ≈ 2,997). This is "
                "weather-independent — it reflects a tail-pricing limitation "
                "inherent to the mean-model's tight λ clustering interacting "
                "with extreme offsets. Deep-over lines carry lower confidence."
            )

    with _tabs[2]:   # 3 — pooled fixed lines
        fpairs = diag.fixed_line_pairs(decided, (7.5, 8.5, 9.5, 10.5))
        fcurve = diag.calibration_curve(fpairs)
        if fcurve["warning"]:
            st.warning(fcurve["warning"])
        else:
            utils.show_chart(diag.chart_calibration(
                fcurve, "Games pooled across 7.5 / 8.5 / 9.5 / 10.5"))
            st.caption(
                f"{fcurve['n_pairs']:,} (game × line) pairs; every game counts "
                "once per line — near-line degeneracy is diluted by the off-"
                "line games pooled in."
            )

    with _tabs[3]:   # 4 — fixed money line only, zoomed
        m85 = diag.fixed_line_pairs(decided, (8.5,))
        m85c = diag.calibration_curve(m85)
        if m85c["warning"]:
            st.warning(m85c["warning"])
        else:
            utils.show_chart(diag.chart_calibration(
                m85c, "Line 8.5 only",
                x_domain=[0.40, 0.60]))
            st.caption(
                "The x-axis is zoomed to 0.40–0.60 and the blob is NARROW BY "
                "CONSTRUCTION: at a fixed line most games project near the "
                "league mean, so predicted probabilities cluster tightly. "
                "This is what a bettor actually sees at the money line — the "
                "wide spread lives in the Relativized tab."
            )
            st.warning(
                "**Known limitation — deep-over side:** When the 8.5 line sits "
                "well above a game's expected total (p_over high, deep over "
                "region), the model overstates the probability. At offset −2.0, "
                "calibrated pred ≈ 0.66 vs actual ≈ 0.58 (gap ≈ 0.08, "
                "weather-independent). Treat high-confidence over picks at "
                "this line with caution."
            )

    with _tabs[4]:   # 5 — overs pick accuracy buckets
        opicks = diag.overs_pick_table(decided, line=8.5)
        if opicks["warning"] or not opicks["buckets"]:
            st.warning(opicks.get("warning")
                       or "No overs picks could be formed.")
        else:
            built = diag.chart_pick_buckets(opicks, "Overs picks @ 8.5")
            utils.show_chart(built["chart"])
            st.table(built["table"])
            st.caption(
                f"Pick rule: {opicks['pick_rule']} · {opicks['n_games']:,} "
                "decided games. Hit rate is NOT calibration — it is binary "
                "pick accuracy per favored-side confidence bucket."
            )

    with _tabs[5]:   # 6 — run-line pick accuracy buckets
        rpicks = diag.runline_pick_table(decided)
        if rpicks["warning"] or not rpicks["buckets"]:
            st.warning(rpicks.get("warning")
                       or "No run-line picks could be formed.")
        else:
            built = diag.chart_pick_buckets(rpicks, "Run-line picks at −1.5")
            utils.show_chart(built["chart"])
            st.table(built["table"])
            st.caption(
                f"Pick rule: {rpicks['pick_rule']} · {rpicks['n_games']:,} "
                "decided games · hit rate is NOT calibration."
            )
