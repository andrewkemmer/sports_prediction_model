"""Page 5 — Today's Totals & Run Lines (run engine, Phase 3).

Temporary UI simplification: this page renders ONLY the six diagnostics
charts (Distribution, Relativized, Pooled lines, Money line (rounded),
Totals picks, Run-line picks) from
frontend/market_diagnostics.py — pure functions
over run_engine_markets_<date>.csv; the render layer only draws what they
produce. The per-game slate board, total/run-line selectors, calibration
table, rolling totals-Brier, tail fit-check, holdout verdict, and the
derived-ML cross-reference are UN-RENDERED for now — nothing is deleted from
the backend or the artifact, and the charts' empty states and loud warnings
still fire exactly as before.

Missing/stale artifacts produce loud warnings and graceful empty states —
never fake data.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

import utils

utils.inject_css()

dates = utils.available_dates(**utils.get_source_config())
date_str = st.session_state.get("selected_date", dates[0] if dates else "20260809")

st.markdown(
    "<div style='font-size:1.7rem;font-weight:800;color:#E2E8F0;'>"
    "Today's Totals &amp; Run Lines</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='color:#94A3B8;margin:2px 0 14px;'>"
    "Run-engine model diagnostics from NB(λ, α(λ)) Monte Carlo over the "
    "line grid — per-game projections and full probabilities live in "
    "run_engine_markets_*.csv.</div>",
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
        "Money line (rounded)", "Totals picks", "Run-line picks",
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
                "is overconfident. Measured gap (3.5c full-coverage re-check, "
                "2026-08-24 OOF): prediction ≈ 0.66 vs actual ≈ 0.60 "
                "(≈ 0.06 shortfall, n ≈ 4,156). This is "
                "weather-independent — the gap is identical WITHOUT the "
                "env-level features (+0.058 vs +0.054), reflecting a "
                "tail-pricing limitation inherent to the mean-model's tight "
                "λ clustering interacting with extreme offsets. Deep-over "
                "lines carry lower confidence."
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

    with _tabs[3]:   # 4 — per-game rounded-total money line, pooled
        mrt = diag.rounded_total_pairs(decided)
        mrtc = diag.calibration_curve(mrt)
        if mrtc["warning"]:
            st.warning(mrtc["warning"])
        else:
            utils.show_chart(diag.chart_calibration(
                mrtc, "Per-game rounded total"))
            xs = [b["mean_pred"] for b in mrtc["bins"]]
            _ps = diag.push_stats(decided)
            st.caption(
                f"Each game priced at ITS OWN rounded total — nearest 0.5 of "
                f"λ_home + λ_away (round half up; lines outside the shipped "
                f"6.5–12.5 grid clamp to the edge) — then pooled. "
                f"{_ps['n_games']:,} games · {_ps['n_pushes']:,} pushes "
                f"excluded ({_ps['push_rate']:.1%}) · predicted range "
                f"{min(xs):.2f}–{max(xs):.2f}. Pushes are UNDER-favored "
                "games landing exactly on the line (rounded line at/above "
                "the expected total → under favored) — excluded from the "
                "curve because they are neither wins nor losses. Every game "
                "sits at its own line, so probabilities concentrate near "
                "the money line by construction — the wide calibration "
                "spread lives in the Relativized tab."
            )

    with _tabs[4]:   # 5 — totals picks at each game's rounded line
        tpicks = diag.totals_pick_table(decided)
        if tpicks["warning"] or not tpicks["buckets"]:
            st.warning(tpicks.get("warning")
                       or "No totals picks could be formed.")
        else:
            built = diag.chart_pick_buckets(
                tpicks, "Totals picks (per-game rounded line)",
                total_line=True, acc_y_max=75.0)
            utils.show_chart(built["chart"])
            st.table(built["table"])
            st.caption(
                f"Pick rule: {tpicks['pick_rule']} · {tpicks['n_games']:,} "
                f"decided games · {tpicks['n_pushes']:,} pushes excluded "
                f"({tpicks['push_rate']:.1%}) · pooled win rate: "
                f"{tpicks['win_rate']:.1%}. Pushes are UNDER-favored games "
                "landing exactly on the line (rounded line at/above the "
                "expected total → under favored) and were "
                "previously scored as wins — excluding them LOWERS the honest "
                "win rate vs the inflated one (2026-08-24 artifact: 56.1% → 54.1%, "
                "≈2,420 wins/4,314 → ≈2,200 wins/4,066). Every game is priced "
                "at its own rounded total, so high-confidence buckets are "
                "small. Hit rate is NOT calibration — it is binary pick "
                "accuracy per favored-side confidence bucket."
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


