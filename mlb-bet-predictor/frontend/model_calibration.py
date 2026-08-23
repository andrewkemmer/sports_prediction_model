"""Page 3 — Model Calibration.

Header, summary (today's record + upsets), KPI cards (AUC-ROC, Brier,
Log-Loss, Cal. Error), calibration curve vs a perfect-calibration diagonal,
the Prediction Confidence & Accuracy combo chart, and the reliability table
with color-coded GAP values.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import utils

utils.inject_css()

dates = utils.available_dates(**utils.get_source_config())
date_str = st.session_state.get("selected_date", dates[0] if dates else "20260809")
cal = utils.load_calibration(date_str)
if not cal:
    st.warning(f"No calibration artifacts found for {date_str} or any recent date.")
    st.stop()

artifact_date = cal.get("_artifact_date", date_str)
n_games = cal.get("n_games", 0)
kpis = cal.get("kpis", {})
curve = cal.get("calibration_curve", [])
confidence = cal.get("confidence", [])
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
upset_text = " · ".join(f"{u['team']} {u['prob']:.0%} upset" for u in upsets) or "No upsets today"
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
kpi_specs = [
    ("AUC-ROC", kpis.get("auc_roc", "—"), utils.BLUE, "Discrimination"),
    ("BRIER SCORE", _brier_disp := (
        f"{kpis['brier_score']} → {kpis['brier_calibrated']}"
        if kpis.get("brier_calibrated") is not None else kpis.get("brier_score", "—")
     ), utils.PRIMARY, "Lower is better" + (" · after calibration" if kpis.get("brier_calibrated") is not None else "")),
    ("LOG-LOSS", _ll_disp := (
        f"{kpis['log_loss']} → {kpis['log_loss_calibrated']}"
        if kpis.get("log_loss_calibrated") is not None else kpis.get("log_loss", "—")
     ), "#FBBF24", "Penalizes confidence" + (" · after calibration" if kpis.get("log_loss_calibrated") is not None else "")),
    ("CAL. ERROR", _ece_disp := (
        f"{kpis['cal_error']} → {kpis['cal_error_calibrated']}"
        if kpis.get("cal_error_calibrated") is not None else kpis.get("cal_error", "—")
     ), "#F472B6", "ECE raw → calibrated" if kpis.get("cal_error_calibrated") is not None else "ECE metric"),
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
cal_sec = cal.get("calibration") or {}
if cal_sec.get("method") == "platt":
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

# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------
st.markdown("### Calibration Curve — Favored Team")

# Per-1%-probability calibration, built from the game-level prediction
# history: each OOF prediction is taken from the FAVORED team's side
# (probability >= 50%), rounded to the nearest 1%, and the actual win rate
# of that 1% slice is plotted at that point.
hist_curve = utils.load_prediction_history(date_str)
pts = pd.DataFrame()
if (hist_curve is not None and not hist_curve.empty
        and {"home_win_prob_model", "correct"} <= set(hist_curve.columns)):
    _p = pd.to_numeric(hist_curve["home_win_prob_model"], errors="coerce")
    _w = pd.to_numeric(hist_curve["correct"], errors="coerce")
    _ok = _p.notna() & _w.notna()
    _p, _w = _p[_ok], _w[_ok]
    _fav = np.maximum(_p, 1.0 - _p)          # favored-side probability
    _fav = (_fav * 100).round() / 100        # nearest 1%
    pts = (
        pd.DataFrame({"prob": _fav, "won": _w})
        .groupby("prob")
        .agg(win_rate=("won", "mean"), n=("won", "size"))
        .reset_index()
    )

# Calibrated twin of the raw curve (same construction, corrected probs).
pts_cal = pd.DataFrame()
if (hist_curve is not None and not hist_curve.empty
        and {"home_win_prob_model_calibrated", "correct"} <= set(hist_curve.columns)):
    _pc = pd.to_numeric(hist_curve["home_win_prob_model_calibrated"], errors="coerce")
    _wc = pd.to_numeric(hist_curve["correct"], errors="coerce")
    _okc = _pc.notna() & _wc.notna()
    _pc, _wc = _pc[_okc], _wc[_okc]
    _favc = np.maximum(_pc, 1.0 - _pc)
    _favc = (_favc * 100).round() / 100
    pts_cal = (
        pd.DataFrame({"prob": _favc, "won": _wc})
        .groupby("prob")
        .agg(win_rate=("won", "mean"), n=("won", "size"))
        .reset_index()
    )

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

if not pts.empty:
    model_pts = alt.Chart(pts).mark_line(
        point=alt.OverlayMarkDef(filled=True, size=55), color=utils.BLUE, strokeWidth=2.5,
    ).encode(
        x=alt.X("prob:Q", title="Predicted win probability", scale=alt.Scale(domain=[0.45, 1.0])),
        y=alt.Y("win_rate:Q", title="Actual win rate", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("prob:Q", title="Predicted", format=".0%"),
                 alt.Tooltip("win_rate:Q", title="Actual win rate", format=".1%"),
                 alt.Tooltip("n:Q", title="Games")],
    )
    diag_df = pd.DataFrame({"prob": [0.45, 1.0], "win_rate": [0.45, 1.0]})
    diag = alt.Chart(diag_df).mark_line(color="#64748B", strokeDash=[5, 5], strokeWidth=1.5).encode(
        x=alt.X("prob:Q", scale=alt.Scale(domain=[0.45, 1.0])),
        y=alt.Y("win_rate:Q", scale=alt.Scale(domain=[0, 1])),
    )
    # Shared scales + explicit container width: without these, layered
    # charts fall back to natural size and collapse to a narrow strip.
    layers = [diag, model_pts]
    legend_extra = ""
    if not pts_cal.empty:
        cal_line = alt.Chart(pts_cal).mark_line(
            point=alt.OverlayMarkDef(filled=True, size=45), color="#34D399",
            strokeDash=[6, 4], strokeWidth=2,
        ).encode(
            x=alt.X("prob:Q"),
            y=alt.Y("win_rate:Q"),
            tooltip=[alt.Tooltip("prob:Q", title="Calibrated", format=".0%"),
                     alt.Tooltip("win_rate:Q", title="Actual win rate", format=".1%"),
                     alt.Tooltip("n:Q", title="Games")],
        )
        layers.append(cal_line)
        legend_extra = " · Green dashed: after recalibration"
    layer = alt.layer(*layers).resolve_scale(x="shared", y="shared").properties(
        width="container", height=340,
    )
    utils.show_chart(layer)
    st.caption(
        f"Model (n={n_games:,}) · Perfect Calibration (dashed diagonal)"
        f"{legend_extra} · each game counted once from the favored side; "
        "probabilities rounded to the nearest 1% — hover for games per point"
    )

# ---------------------------------------------------------------------------
# Confidence & accuracy combo chart
# ---------------------------------------------------------------------------
st.markdown("### Prediction Confidence & Accuracy")
conf_df = pd.DataFrame(confidence) if confidence else pd.DataFrame()
if conf_df.empty:
    st.info("No confidence distribution data available.")
else:
    bars = alt.Chart(conf_df).mark_bar(color=utils.BLUE, opacity=0.85).encode(
        x=alt.X("bucket:N", title="Confidence bucket", sort=None),
        y=alt.Y("count:Q", title="Game Count", axis=alt.Axis(grid=True)),
    )
    line = alt.Chart(conf_df).mark_line(point=alt.OverlayMarkDef(filled=True, size=50), color=utils.PRIMARY, strokeWidth=2.5).encode(
        x=alt.X("bucket:N", sort=None),
        y=alt.Y("accuracy_pct:Q", title="Actual Accuracy %", axis=alt.Axis(orient="right", grid=False),
                scale=alt.Scale(domain=[0, 100])),
    )
    combo = alt.layer(bars, line).resolve_scale(y="independent").properties(width="container", height=330)
    utils.show_chart(combo)
    st.caption("Blue bars: game count per bucket (left axis) · green line: actual accuracy % (right axis)")

# ---------------------------------------------------------------------------
# Reliability table
# ---------------------------------------------------------------------------
st.markdown("### Reliability Diagram — Binned Data")
if curve_df.empty:
    st.info("No reliability data available.")
else:
    rows = []
    for _, r in curve_df.iterrows():
        gap = r["gap"]
        gap_color = utils.PRIMARY if gap > 0 else utils.RED
        gap_txt = f"{gap:+.3f}"
        rows.append(
            f"<tr><td>{r['bucket']}</td><td>{r['mean_predicted']:.3f}</td>"
            f"<td>{r['mean_actual']:.3f}</td><td>{int(r['count'])}</td>"
            f"<td style='color:{gap_color};font-weight:700;'>{gap_txt}</td></tr>"
        )
    # TOTAL row: overall win rate across ALL predictions, count-weighted
    n_tot = int(curve_df["count"].sum())
    if n_tot > 0:
        mp_tot = float((curve_df["mean_predicted"] * curve_df["count"]).sum() / n_tot)
        ma_tot = float((curve_df["mean_actual"] * curve_df["count"]).sum() / n_tot)
        gap_tot = mp_tot - ma_tot
        tot_color = utils.PRIMARY if gap_tot > 0 else utils.RED
        rows.append(
            f"<tr style='border-top:2px solid #334155;font-weight:700;'><td>TOTAL</td>"
            f"<td>{mp_tot:.3f}</td><td>{ma_tot:.3f}</td><td>{n_tot}</td>"
            f"<td style='color:{tot_color};font-weight:700;'>{gap_tot:+.3f}</td></tr>"
        )
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <table class="fb-table">
            <thead><tr><th>BUCKET</th><th>MEAN PREDICTED</th><th>MEAN ACTUAL</th><th>COUNT</th><th>GAP</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        <div style="color:#64748B;font-size:0.78rem;margin-top:6px;">
          Favored-team view: every game counted once at its pick probability (≥ 50%). GAP = mean predicted − mean actual. Green: overconfident (positive). Red: underconfident (negative).
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
        st.caption(
            f"{n_rng:,} games · {acc_rng:.1f}% picks correct · most recent first — "
            "scroll for older results"
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
            prob = r.get("home_win_prob_model")
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
