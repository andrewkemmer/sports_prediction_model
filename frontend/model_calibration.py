"""Page 3 — Model Calibration.

Header, summary (today's record + upsets), KPI cards (AUC-ROC, Brier,
Log-Loss, Cal. Error), a merged confidence-vs-accuracy + calibration curve
(count bars + actual rate vs the stored OOF calibration series vs the
perfect-calibration diagonal), and the reliability table with color-coded GAP values.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

# On the NFL page, the summary card is fed the LIFETIME OOF/sealed pool
# (1,000+ decided games), so the upset list can number in the hundreds. Show
# only the most-surprising few and collapse the rest into a ``+N more`` tail;
# the MLB page (a handful of upsets among one day's games) is untouched.
NFL_UPSET_CAP = 10

import inspect
import numpy as np
import pandas as pd
import streamlit as st

import moneyline_calibration as mlc
import utils

utils.inject_css()

dates = utils.available_dates(**utils.get_source_config())
# Always show the most recent run (like Power Rankings / Model Monitor):
# ignore the date picked on Today's Games so the tab never drills into a
# past day's small per-day slice.
date_str = dates[0] if dates else "20260809"
if "use_daily" in inspect.signature(utils.load_calibration).parameters:
    cal = utils.load_calibration(date_str, use_daily=False)
else:
    # Deployed utils.py may predate the use_daily param (stale snapshot):
    # fall back to the plain call — date pinning alone still yields the
    # latest pooled view for current artifacts.
    cal = utils.load_calibration(date_str)
if not cal:
    st.warning(f"No calibration artifacts found for {date_str} or any recent date.")
    st.stop()

artifact_date = cal.get("_artifact_date", date_str)
# Population behind the pooled KPI cards: the walk-forward GRADING rows
# (n_eval — regular/non-provisional OOF games). MLB's writer used to carry
# the day's slate size in n_games while NFL/NHL/NBA carry the grading pool
# there, so the MLB dashboard read "n = 4 games" beside AUC graded on
# 6,879 pooled OOF games. Resolve the graded count first so pooled metrics
# are never labeled with the slate size (2026-10-07 pooled-run parity).
n_games = cal.get("n_eval") or cal.get("n_games", 0)
kpis = cal.get("kpis", {})
curve = cal.get("calibration_curve", [])
record = cal.get("today_record", {})
upsets = cal.get("upsets", [])


def _trained_label(raw: str) -> str:
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        et = ts.astimezone(ZoneInfo("America/New_York"))
        return f"{et.strftime('%B')} {et.day}, {et.year} {et.strftime('%H:%M')} ET"
    except (ValueError, TypeError):
        return raw or "—"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("<div style='font-size:1.7rem;font-weight:800;color:#E2E8F0;'>Model Calibration Dashboard</div>",
            unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="display:inline-flex;align-items:center;gap:6px;margin:6px 0 2px;color:#94A3B8;
                border:1px solid #1E293B;border-radius:999px;padding:3px 12px;font-size:0.85rem;">
      As of {utils.format_date_long(artifact_date)} · n = {n_games:,} games · Trained {_trained_label(cal.get('trained_at', ''))}
    </div>
    {f'<div style="color:#64748B;font-size:0.82rem;margin-top:2px;">ℹ No artifact for {utils.format_date_long(date_str)} — showing latest snapshot ({utils.format_date_long(artifact_date)})</div>' if artifact_date != date_str else ''}
    <div style="color:#94A3B8;font-size:0.9rem;margin-top:4px;">
      Assessing prediction reliability and accuracy across probability buckets
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Summary card
# ---------------------------------------------------------------------------
wins, losses = record.get("wins", 0), record.get("losses", 0)
completed = record.get("completed", wins + losses)
acc = (wins / completed * 100) if completed else 0.0
if utils.get_sport() in {"nfl", "nba"} and len(upsets) > NFL_UPSET_CAP:
    # Biggest upsets first = the winner with the LOWEST model probability.
    top = sorted(upsets, key=lambda u: float(u.get("prob", 1.0) or 1.0))[:NFL_UPSET_CAP]
    upset_text = " · ".join(f"{u['team']} {u['prob']:.0%} upset" for u in top)
    upset_text += f" · +{len(upsets) - NFL_UPSET_CAP} more upsets"
else:
    upset_text = (" · ".join(f"{u['team']} {u['prob']:.0%} upset" for u in upsets)
                  or "No upsets today")
st.markdown(
    f"""
    <div class="fb-box" style="margin:14px 0;padding:14px 18px;">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;color:#E2E8F0;">
        <span style="font-weight:700;">Today's Record:</span>
        <span style="background:rgba(16,185,129,.18);color:#34D399;border-radius:999px;padding:2px 12px;font-weight:800;">✓ {wins}-{losses}</span>
        <span style="color:#94A3B8;font-size:0.9rem;">{completed} completed games · {wins} correct picks ({acc:.1f}%) · {len(upsets)} upsets</span>
      </div>
      <div style="margin-top:8px;">
        <span style="background:rgba(245,158,11,.18);color:#FBBF24;border-radius:999px;padding:2px 12px;font-size:0.82rem;font-weight:700;">⚡ {upset_text}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
# Calibrator provenance, resolved ONCE above the cards. With the
# prequential gate deployed (``calibrator_gated_out`` / calibration.method
# == "identity") the artifact's raw and calibrated twins are the SAME
# probabilities — no map was applied — so the cards' "after calibration"
# wording would advertise a correction that never ran, and the identical
# values would look like a rendering bug (2026-10-08 dashboard review:
# "0.5719 → 0.5719 after calibration" with no explanation anywhere). The
# gate banner below carries the full story; the cards just stop claiming a
# transformation.
cal_sec = cal.get("calibration") or {}
_gated = bool((cal.get("metrics") or {}).get("calibrator_gated_out")) \
    or cal_sec.get("method") == "identity"
_cal_tail = (" · calibrator off (identity)" if _gated
             else " · after calibration")

kpi_specs = [
    ("AUC-ROC", kpis.get("auc_roc", "—"), utils.BLUE, "Discrimination"),
    ("BRIER SCORE", _brier_disp := (
        f"{kpis['brier_score']} → {kpis['brier_calibrated']}"
        if kpis.get("brier_calibrated") is not None else kpis.get("brier_score", "—")
     ), utils.PRIMARY, "Lower is better" + (_cal_tail if kpis.get("brier_calibrated") is not None else "")),
    ("LOG-LOSS", _ll_disp := (
        f"{kpis['log_loss']} → {kpis['log_loss_calibrated']}"
        if kpis.get("log_loss_calibrated") is not None else kpis.get("log_loss", "—")
     ), "#FBBF24", "Penalizes confidence" + (_cal_tail if kpis.get("log_loss_calibrated") is not None else "")),
    ("CAL. ERROR", _ece_disp := (
        f"{kpis['cal_error']} → {kpis['cal_error_calibrated']}"
        if kpis.get("cal_error_calibrated") is not None else kpis.get("cal_error", "—")
     ), "#F472B6", ("ECE raw → calibrated" + _cal_tail) if kpis.get("cal_error_calibrated") is not None else "ECE metric"),
]
kcols = st.columns(4)
for col, (label, value, color, cap) in zip(kcols, kpi_specs):
    with col:
        st.markdown(
            f'<div class="fb-kpi"><div class="label">{label}</div>'
            f'<div class="value" style="color:{color};">{value}</div>'
            f'<div class="cap">{cap}</div></div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Post-hoc recalibration banner (raw vs calibrated)
# ---------------------------------------------------------------------------
if cal_sec.get("method") in ("platt", "favored_platt_floor"):
    _mr = cal_sec.get("metrics_raw") or {}
    _mc = cal_sec.get("metrics_calibrated") or {}
    _params = cal_sec.get("params") or {}
    _ece_raw = _mr.get("ece")
    _ece_cal = _mc.get("ece")
    if _ece_raw is not None and _ece_cal is not None:
        _delta = (_ece_raw - _ece_cal) * 100
        _arrow = "🟢" if _delta >= 0 else "🔴"
        st.markdown(
            f"""
            <div class="fb-box" style="margin:12px 0;padding:12px 18px;">
              <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;color:#E2E8F0;">
                <span style="font-weight:700;">Post-Hoc Recalibration:</span>
                <span style="background:rgba(59,130,246,.18);color:#60A5FA;border-radius:999px;padding:2px 12px;font-size:0.82rem;font-weight:700;">
                  Platt scaling · a={_params.get('a', '—')}, b={_params.get('b', '—')}
                </span>
                <span style="color:#94A3B8;font-size:0.9rem;">
                  fitted on {int(_params.get('n', 0) or 0):,} out-of-sample games
                </span>
                <span style="background:rgba(16,185,129,.15);color:#34D399;border-radius:999px;padding:2px 12px;font-size:0.82rem;font-weight:700;">
                  {_arrow} ECE {_ece_raw:.4f} → {_ece_cal:.4f} ({_delta:+.2f} pts)
                </span>
              </div>
              <div style="color:#64748B;font-size:0.8rem;margin-top:6px;">
                Published probabilities are corrected after blending: p<sub>cal</sub> = σ(a·logit(p) + b).
                Fitted only on out-of-fold predictions — each evaluation fold is scored by a map trained strictly on prior folds.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
elif _gated:
    # Gated-out calibrator (identity) — the deployed model applies NO
    # probability map, which is why the KPI twins read identically. Shown
    # for every identity run (incl. legacy artifacts carrying only
    # ``calibration.method == "identity"``); the Platt banner above owns the
    # fitted-map case, so the two can never both render (2026-10-08
    # dashboard review: identical raw/calibrated values with no explanation).
    st.markdown(
        """
        <div class="fb-box" style="margin:12px 0;padding:12px 18px;">
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;color:#E2E8F0;">
            <span style="font-weight:700;">Calibration Gate:</span>
            <span style="background:rgba(245,158,11,.18);color:#FBBF24;border-radius:999px;padding:2px 12px;font-size:0.82rem;font-weight:700;">
              Gated out · identity
            </span>
            <span style="color:#94A3B8;font-size:0.9rem;">
              The prequential gate found no gain, so this run deploys no recalibration map.
            </span>
          </div>
          <div style="color:#64748B;font-size:0.8rem;margin-top:6px;">
            Probabilities ship RAW — no map is applied, so the raw → calibrated KPI
            pairs are the same values by design, not a rendering bug (Model Monitor
            labels this “no calibration map deployed”). The gate re-fits a pooled
            Platt map on strictly prior folds every run and deploys it only when it
            beats the raw probabilities.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------
st.markdown("### Calibration Curve — Favored Team")

# Per-1%-probability calibration, built from the game-level prediction
# history: each OOF prediction is taken from the FAVORED team's side
# (probability >= 50%), binned to the nearest 1%; each 1% slice yields one
# calibration point (win_rate) AND one count bar (n) from the same frame.
hist_curve = utils.load_prediction_history(artifact_date)
pts = mlc.favored_calibration_pts(hist_curve)

# Bucketed curve from the artifact (also feeds the reliability table below).
curve_df = pd.DataFrame(curve) if curve else pd.DataFrame()

if pts.empty:
    # Fallback: 10-point bucket curve from the calibration artifact
    if curve_df.empty:
        st.info("No calibration curve data available.")
    else:
        pts = curve_df.rename(columns={
            "mean_predicted": "prob", "mean_actual": "win_rate", "count": "n",
        })[["prob", "win_rate", "n"]]

# Green reference curve: the DEPLOYED pooled Platt map applied to the same
# raw favored-probability bins as the blue curve. MLB's three-probability
# contract marks the per-fold prequential column "honest for scoring/metrics;
# NEVER display" — beside a raw curve it draws a reference the deployed model
# never produces (the 2026-09-28 NFL dashboard: green claimed 57-59% where
# the model shipped 53-56% and actual was ~50-52%). The deployed map is read
# from the artifact's calibration.params (a, b) — the exact mapping serve
# time applies; the frontend never refits it. Built AFTER the pts fallback
# so the bucket-curve path also gets the reference layer.
_platt = (cal.get("calibration") or {}).get("params") or {}
pts_cal = mlc.deployed_calibration_pts(pts, _platt)
if pts_cal.empty and not _platt:
    # Legacy fallback for artifacts predating the published params (or an
    # identity calibration with none): group the STORED per-game calibrated
    # column instead. With no deployed map to draw, the artifact's own
    # prequential convention is the closest honest reference.
    pts_cal = mlc.favored_oof_calibration_pts(hist_curve)

if not pts.empty:
    # Merged confidence-vs-accuracy + calibration curve: count bars (LEFT
    # 'Games' axis) + blue actual-rate curve / green Platt map (RIGHT '%'
    # axis, independent) + gray dashed perfect-calibration diagonal. Bars and
    # the blue curve come from the SAME filled 1% bins, so bar height = games
    # in that confidence bucket and the curve = their accuracy — one chart, no
    # information lost from the former standalone 'Prediction Confidence &
    # Accuracy' section.
    built = mlc.chart_favored_calibration(pts, pts_cal)
    # The chart's own population: sum(n) over the binned favored-side points
    # is every decided game in the plotted history (all OOF rows, incl.
    # postseason/provisional) — the bars/curve cover exactly these games, so
    # the caption labels them with their own count, never the KPI cards'
    # grading pool or the day's slate size.
    chart_n = int(pts["n"].sum()) if "n" in pts.columns else n_games
    legend_extra = ""
    if not pts_cal.empty:
        legend_extra = (" · Green dashed: the deployed pooled Platt map "
                        f"(a={_platt.get('a', '—')}, b={_platt.get('b', '—')}) "
                        "applied to the same raw bins — exactly what serving applies; "
                        "vertical gap at each bin = the correction the map makes to the raw model")
    utils.show_chart(built["chart"])
    st.caption(
        f"Model (n={chart_n:,}) · Count bars (left 'Games' axis): games per "
        f"1% predicted-probability bin — the bars are the confidence-vs-"
        f"accuracy view, bar height = how many games the model priced in that "
        f"confidence band and the blue curve = how often those games won · "
        f"Blue: actual win rate at each raw probability · "
        f"Green: the deployed pooled Platt map at each raw probability · "
        f"Perfect Calibration (dashed diagonal)"
        f"{legend_extra} · each game counted once from the favored side; "
        "blue curve binned to the nearest 1% — hover for games per point"
    )

# ---------------------------------------------------------------------------
# Reliability table
# ---------------------------------------------------------------------------
st.markdown("### Reliability Diagram — Binned Data")
# Prequential calibrated buckets (each point corrected by a map fitted on
# strictly PRIOR folds) shown alongside the raw view, so overconfidence can
# be judged at BOTH stages of the deployed chain.
_cal_buckets = {
    b.get("bucket"): b
    for b in ((cal.get("calibration") or {}).get("calibration_buckets_calibrated") or [])
}
if curve_df.empty:
    st.info("No reliability data available.")
else:
    rows = []
    for _, r in curve_df.iterrows():
        gap = r["gap"]
        gap_color = utils.PRIMARY if gap > 0 else utils.RED
        gap_txt = f"{gap:+.3f}"
        _cb = _cal_buckets.get(r["bucket"])
        cal_cell = (
            f"<td style='color:#34D399;'>{_cb['mean_predicted']:.3f}</td>"
            if _cb else "<td style='color:#475569;'>—</td>"
        )
        rows.append(
            f"<tr><td>{r['bucket']}</td><td>{r['mean_predicted']:.3f}</td>"
            f"{cal_cell}"
            f"<td>{r['mean_actual']:.3f}</td><td>{int(r['count'])}</td>"
            f"<td style='color:{gap_color};font-weight:700;'>{gap_txt}</td></tr>"
        )
    # TOTAL row: overall win rate across ALL predictions, count-weighted
    n_tot = int(curve_df["count"].sum())
    if n_tot > 0:
        mp_tot = float((curve_df["mean_predicted"] * curve_df["count"]).sum() / n_tot)
        ma_tot = float((curve_df["mean_actual"] * curve_df["count"]).sum() / n_tot)
        # TOTAL calibrated win probability: count-weighted mean over the
        # prequential calibrated buckets — each game sits in the bucket its
        # CALIBRATED probability falls in (re-binned vs the raw view, so
        # per-bucket counts differ while the grand total stays n_tot).
        # Shown only when the calibrated partition covers every displayed
        # bucket AND counts the same games; a partial partition keeps the
        # honest em-dash instead of masquerading as a total (2026-10-05).
        _cal_cov = [_cal_buckets.get(b) for b in curve_df["bucket"]]
        mp_cal_tot = None
        if all(_cal_cov):
            _n_cal = sum(int(b["count"]) for b in _cal_cov)
            if _n_cal == n_tot:
                mp_cal_tot = (
                    sum(float(b["mean_predicted"]) * int(b["count"])
                        for b in _cal_cov) / _n_cal)
        cal_tot_cell = (
            f"<td style='color:#34D399;'>{mp_cal_tot:.3f}</td>"
            if mp_cal_tot is not None
            else "<td style='color:#64748B;'>—</td>"
        )
        gap_tot = mp_tot - ma_tot
        tot_color = utils.PRIMARY if gap_tot > 0 else utils.RED
        rows.append(
            f"<tr style='border-top:2px solid #334155;font-weight:700;'><td>TOTAL</td>"
            f"<td>{mp_tot:.3f}</td>{cal_tot_cell}"
            f"<td>{ma_tot:.3f}</td><td>{n_tot}</td>"
            f"<td style='color:{tot_color};font-weight:700;'>{gap_tot:+.3f}</td></tr>"
        )
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <table class="fb-table">
            <thead><tr><th>BUCKET</th><th>MEAN PREDICTED (RAW)</th><th>CALIBRATED</th>
            <th>MEAN ACTUAL</th><th>COUNT</th><th>GAP (RAW)</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        <div style="color:#64748B;font-size:0.78rem;margin-top:6px;">
          Favored-team view: every game counted once at its pick probability (≥ 50%). GAP = mean predicted − mean actual. Green: overconfident (positive). Red: underconfident (negative).
          CALIBRATED = prequential Platt-corrected prediction per bucket — each game corrected by a map fitted only on prior games, the same convention as deployment. TOTAL CALIBRATED = count-weighted mean across those calibrated buckets (each game counted once in the bucket its calibrated probability falls in).
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Game-level history: every walk-forward prediction vs its actual result
# ---------------------------------------------------------------------------
st.markdown("### Prediction History — Every Game")
hist = utils.load_prediction_history(date_str)
if hist is None or hist.empty or "home_win_prob_model" not in hist.columns:
    st.info("No per-game prediction history available yet (generated on the next pipeline run).")
else:
    h = hist.copy()
    h["_date"] = pd.to_datetime(h["game_date"], errors="coerce")
    lo, hi = h["_date"].min().date(), h["_date"].max().date()

    fc1, fc2, _ = st.columns([1, 1, 2])
    start_d = fc1.date_input("Start date", value=lo, min_value=lo, max_value=hi)
    end_d = fc2.date_input("End date", value=hi, min_value=lo, max_value=hi)
    if start_d > end_d:
        start_d, end_d = end_d, start_d

    in_range = h[(h["_date"].dt.date >= start_d) & (h["_date"].dt.date <= end_d)]
    view = in_range.sort_values("_date", ascending=False)
    n_rng = len(view)
    if n_rng == 0:
        st.info("No games in the selected date range.")
    else:
        acc_rng = float(pd.to_numeric(view["correct"], errors="coerce").mean() * 100)
        # The history writer persists the per-fold deployed probability. Use
        # it directly: applying the final pooled map here would make an OOF
        # row disagree with the probability that was actually evaluated.
        _p_raw = pd.to_numeric(view["home_win_prob_model"], errors="coerce")
        if "home_win_prob_model_calibrated" in view.columns:
            _p_disp = pd.to_numeric(
                view["home_win_prob_model_calibrated"], errors="coerce").fillna(_p_raw)
            _cal_note = " · probabilities are stored prequential deployed outputs"
        else:
            _p_disp = _p_raw
            _cal_note = ""
        st.caption(
            f"{n_rng:,} games · {acc_rng:.1f}% picks correct · most recent first — "
            "scroll for older results" + _cal_note
        )
        rows = []
        for _, r in view.iterrows():
            ok = pd.to_numeric(pd.Series([r.get("correct")]), errors="coerce").iloc[0]
            if pd.isna(ok):
                res = "<td>—</td>"
            elif bool(ok):
                res = f"<td style='color:{utils.PRIMARY};font-weight:700;'>✓</td>"
            else:
                res = f"<td style='color:{utils.RED};font-weight:700;'>✗</td>"
            prob = _p_disp.loc[r.name] if r.name in _p_disp.index else r.get("home_win_prob_model")
            pick_prob = prob if str(r.get("model_pick")) == str(r.get("home_team")) else 1 - prob
            score = "—"
            hs, asc = r.get("home_score"), r.get("away_score")
            if pd.notna(hs) and pd.notna(asc):
                score = f"{int(asc)}–{int(hs)}"
            rows.append(
                f"<tr><td>{r['_date'].strftime('%b %d, %Y')}</td>"
                f"<td>{r.get('away_team','')} @ {r.get('home_team','')}</td>"
                f"<td>{score}</td>"
                f"<td>{r.get('model_pick','')} ({pick_prob:.0%})</td>"
                f"<td>{r.get('actual_winner','')}</td>{res}</tr>"
            )
        st.markdown(
            f"""
            <div class="fb-box" style="padding:6px 8px;">
              <div style="max-height:480px;overflow-y:auto;">
                <table class="fb-table">
                  <thead><tr><th>DATE</th><th>MATCHUP</th><th>SCORE (A–H)</th>
                  <th>MODEL PICK</th><th>WINNER</th><th>RESULT</th></tr></thead>
                  <tbody>{''.join(rows)}</tbody>
                </table>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
