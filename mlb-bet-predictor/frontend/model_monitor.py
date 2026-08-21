"""Page 4 — Model & Data Drift Monitor.

Health cards (last retrain / next retrain / drift alerts), the upset
monitoring callout, the feature-drift (PSI) matrix with status pills, the
rolling 30-day Brier timeline vs a fixed baseline, and the model version
history table.
"""

from __future__ import annotations

from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

import utils

utils.inject_css()

dates = utils.available_dates(**utils.get_source_config())
date_str = st.session_state.get("selected_date", dates[0] if dates else "20260809")
mon = utils.load_model_monitor(date_str)
if not mon:
    st.warning(f"No model monitor artifacts found for {date_str}.")
    st.stop()


def _fmt_date(raw: str) -> str:
    try:
        d = date.fromisoformat(str(raw)[:10])
        return f"{d.strftime('%b')} {d.day}, {d.year}"
    except (ValueError, TypeError):
        return str(raw) or "—"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("<div style='font-size:1.7rem;font-weight:800;color:#E2E8F0;'>Model & Data Drift Monitor</div>",
            unsafe_allow_html=True)
st.markdown("<div style='color:#94A3B8;margin:2px 0 14px;'>Tracking model health, feature drift, and performance over time</div>",
            unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Top alert boxes
# ---------------------------------------------------------------------------
last_retrained = mon.get("last_retrained", "")
next_retrain = mon.get("next_retrain", "")
drift_alerts = mon.get("drift_alerts", [])

try:
    days_since = (utils.parse_date(date_str) - date.fromisoformat(last_retrained[:10])).days
except (ValueError, TypeError):
    days_since = 0
try:
    days_until = (date.fromisoformat(next_retrain[:10]) - utils.parse_date(date_str)).days
except (ValueError, TypeError):
    days_until = 0

next_note = mon.get("next_retrain_note", "")
if "tonight" not in next_note and days_until <= 1:
    next_note = f"{next_note} — tonight" if next_note else "tonight"
last_note = mon.get("last_retrained_note", "Model healthy")
if "ago" not in last_note:
    last_note = f"{last_note} — {days_since} days ago"

n_warn = sum(1 for a in drift_alerts if a.get("status") == "WARN")
n_alert = sum(1 for a in drift_alerts if a.get("status") == "ALERT")
drift_value = "No drift" if not drift_alerts else (
    f"{n_warn + n_alert} Alert" if n_alert else f"{n_warn} Warning"
)
drift_sub = "—" if not drift_alerts else (
    f"{drift_alerts[0].get('feature', '')} — elevated PSI"
)

boxes = [
    ("LAST RETRAIN", _fmt_date(last_retrained), last_note, utils.PRIMARY),
    ("NEXT RETRAIN", _fmt_date(next_retrain), next_note, utils.BLUE),
    ("DRIFT ALERTS", drift_value, drift_sub, utils.AMBER),
]
bcols = st.columns(3)
for col, (label, value, sub, dot) in zip(bcols, boxes):
    with col:
        st.markdown(
            f"""
            <div class="fb-box" style="height:100%;">
              <div style="color:#94A3B8;font-size:0.72rem;font-weight:700;letter-spacing:1px;">
                <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{dot};margin-right:6px;"></span>{label}
              </div>
              <div style="font-size:1.35rem;font-weight:800;color:#E2E8F0;margin:4px 0 2px;">{value}</div>
              <div style="color:#94A3B8;font-size:0.82rem;">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Upset monitoring note
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="border:1px solid rgba(245,158,11,.55);background:rgba(245,158,11,.06);border-radius:12px;
                padding:12px 16px;margin:14px 0;">
      <div style="color:#FBBF24;font-weight:800;">Upset Monitoring Note — {utils.format_date_short(date_str)}</div>
      <div style="color:#E2E8F0;font-size:0.92rem;margin-top:4px;">{mon.get('upset_note', 'No note available.')}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Feature drift matrix
# ---------------------------------------------------------------------------
st.markdown("### Feature Drift Analysis (PSI Scores)")
drift = mon.get("feature_drift", [])
if drift:
    rows = []
    for r in drift:
        psi = r.get("psi", 0.0)
        psi_color = utils.AMBER if r.get("status") == "WARN" else (utils.RED if r.get("status") == "ALERT" else utils.TEXT)
        pill_cls = {"OK": "ok", "WARN": "warn", "ALERT": "alert"}.get(r.get("status", "OK"), "ok")
        label = {
            "home_team_elo": "Home team ELO",
            "away_sp_era_10g": "Away SP ERA (10g)",
            "bullpen_whip_10g": "Bullpen WHIP (10g)",
            "home_woba_30g": "Home wOBA (30g)",
            "weather_wind_speed": "Wind speed (mph)",
        }.get(r.get("feature", ""), r.get("feature", ""))
        rows.append(
            f"<tr><td style='color:#E2E8F0;'>{label}</td>"
            f"<td>{r.get('current_mean', '—')}</td>"
            f"<td>{r.get('baseline_mean', '—')}</td>"
            f"<td style='color:{psi_color};font-weight:700;'>{psi:.3f}</td>"
            f"<td><span class='fb-status-pill {pill_cls}'>{r.get('status', 'OK')}</span></td></tr>"
        )
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <table class="fb-table">
            <thead><tr><th>FEATURE</th><th>CURRENT MEAN</th><th>BASELINE MEAN</th><th>PSI</th><th>STATUS</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("No drift data available.")

# ---------------------------------------------------------------------------
# Rolling Brier score timeline
# ---------------------------------------------------------------------------
st.markdown("### Rolling Brier Score (Last 30 Days)")
brier = mon.get("rolling_brier", [])
baseline = mon.get("brier_baseline", 0.23)
baseline_version = mon.get("brier_baseline_version", "v3.2.0")
if brier:
    bdf = pd.DataFrame(brier)
    bdf["date"] = pd.to_datetime(bdf["date"])
    line = alt.Chart(bdf).mark_line(point=alt.OverlayMarkDef(filled=True, size=40), color="#F97316", strokeWidth=2.2).encode(
        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %d")),
        y=alt.Y("brier:Q", title="Brier Score", scale=alt.Scale(zero=False)),
    )
    base_df = pd.DataFrame({"x": [bdf["date"].min(), bdf["date"].max()], "y": [baseline, baseline]})
    base_line = alt.Chart(base_df).mark_line(color="#64748B", strokeDash=[5, 5], strokeWidth=1.5).encode(
        x="x:T", y=alt.Y("y:Q", scale=alt.Scale(zero=False)),
    )
    utils.show_chart(alt.layer(base_line, line).properties(height=300))
    st.caption(f"Orange: Brier Score · dashed: Baseline ({baseline_version} = {baseline})")
else:
    st.info("No rolling Brier data available.")

# ---------------------------------------------------------------------------
# Model version history
# ---------------------------------------------------------------------------
st.markdown("### Model Version History")
history = mon.get("version_history", [])
if history:
    vdf = pd.DataFrame(history)[["version", "date", "auc", "brier", "notes"]].rename(
        columns={"version": "VERSION", "date": "DATE", "auc": "ACC", "brier": "BRIER", "notes": "NOTES"}
    )
    vdf["DATE"] = vdf["DATE"].map(lambda d: _fmt_date(str(d)))
    st.table(vdf.reset_index(drop=True))
else:
    st.info("No version history available.")
