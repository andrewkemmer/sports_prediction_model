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
import pandas as pd
import streamlit as st

import utils

utils.inject_css()

dates = utils.available_dates(**utils.get_source_config())
date_str = st.session_state.get("selected_date", dates[0] if dates else "20260809")
cal = utils.load_calibration(date_str)
if not cal:
    st.warning(f"No calibration artifacts found for {date_str}.")
    st.stop()

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
      As of {utils.format_date_long(date_str)} · n = {n_games:,} games · Trained {_trained_label(cal.get('trained_at', ''))}
    </div>
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
    ("BRIER SCORE", kpis.get("brier_score", "—"), utils.PRIMARY, "Lower is better"),
    ("LOG-LOSS", kpis.get("log_loss", "—"), "#FBBF24", "Penalizes confidence"),
    ("CAL. ERROR", kpis.get("cal_error", "—"), "#F472B6", "ECE metric"),
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
# Calibration curve
# ---------------------------------------------------------------------------
st.markdown("### Calibration Curve — Favored Team")
curve_df = pd.DataFrame(curve) if curve else pd.DataFrame()
if curve_df.empty:
    st.info("No calibration curve data available.")
else:
    model_pts = alt.Chart(curve_df).mark_line(point=alt.OverlayMarkDef(filled=True, size=55), color=utils.BLUE, strokeWidth=2.5).encode(
        x=alt.X("mean_predicted:Q", title="Mean Predicted Probability (favorite)", scale=alt.Scale(domain=[0.45, 1.0])),
        y=alt.Y("mean_actual:Q", title="Mean Actual Win Rate", scale=alt.Scale(domain=[0, 1])),
    )
    diag_df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    diag = alt.Chart(diag_df).mark_line(color="#64748B", strokeDash=[5, 5], strokeWidth=1.5).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0.45, 1.0])),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1])),
    )
    layer = alt.layer(diag, model_pts).properties(height=340)
    utils.show_chart(layer)
    st.caption(f"Model (n={n_games:,}) · Perfect Calibration (dashed diagonal) · each game counted once from the favored side (probability ≥ 50%)")

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
    combo = alt.layer(bars, line).resolve_scale(y="independent").properties(height=330)
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
