"""Page 2 — Power Rankings.

Full-width table of the top 15 teams sorted by model (Elo) strength, with a
thin team-color accent line left of each team name cell.

Columns (exact order): RANK | TEAM | ELO | W-L | PCT | RUN DIFF | L10 | HOME% | AWAY%
"""

from __future__ import annotations

import streamlit as st

import utils

utils.inject_css()

dates = utils.available_dates(**utils.get_source_config())
date_str = st.session_state.get("selected_date", dates[0] if dates else "20260809")

rankings = utils.load_power_rankings(date_str)
if rankings.empty:
    st.warning(f"No power rankings found for {date_str}.")
    st.stop()

top15 = rankings.head(15).reset_index(drop=True)

st.markdown(f"<div style='font-size:1.7rem;font-weight:800;color:#E2E8F0;'>Power Rankings</div>",
            unsafe_allow_html=True)
st.markdown(
    f"<div style='color:#94A3B8;margin:2px 0 14px;'>"
    f"Current Elo-based power rankings · As of {utils.format_date_long(date_str)} · Top 15 teams</div>",
    unsafe_allow_html=True,
)

rows = []
for _, r in top15.iterrows():
    color = r.get("color", "#64748B")
    w, l = int(r["w"]), int(r["l"])
    elo_color = "#38BDF8" if r["elo"] >= 1520 else ("#FBBF24" if r["elo"] >= 1500 else "#94A3B8")
    rd = int(r["run_diff"])
    rd_color = utils.PRIMARY if rd > 0 else (utils.RED if rd < 0 else "#94A3B8")
    rd_str = f"{rd:+d}" if rd != 0 else "0"
    team_cell = (
        f'<span class="fb-accent-cell">'
        f'<span class="fb-accent-line" style="background:{color};"></span>'
        f'<span style="font-weight:800;color:#E2E8F0;">{r["team"]}</span>'
        f'<span style="color:#94A3B8;font-size:0.85rem;">{r.get("team_name", "")}</span>'
        f'</span>'
    )
    rows.append(
        f"<tr>"
        f"<td style='color:#94A3B8;font-weight:700;'>{int(r['rank'])}</td>"
        f"<td>{team_cell}</td>"
        f"<td style='color:{elo_color};font-weight:700;'>{r['elo']:,.1f}</td>"
        f"<td>{w}-{l}</td>"
        f"<td style='color:#94A3B8;'>{utils.record_pct(w, l)}</td>"
        f"<td style='color:{rd_color};font-weight:700;'>{rd_str}</td>"
        f"<td>{r.get('l10', '—')}</td>"
        f"<td>{utils.pct(r.get('home_pct', 0), 1)}</td>"
        f"<td>{utils.pct(r.get('away_pct', 0), 1)}</td>"
        f"</tr>"
    )

st.markdown(
    f"""
    <div class="fb-box" style="padding:6px 8px;">
      <table class="fb-table">
        <thead>
          <tr><th>RANK</th><th>TEAM</th><th>ELO</th><th>W-L</th><th>PCT</th>
              <th>RUN DIFF</th><th>L10</th><th>HOME%</th><th>AWAY%</th></tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Elo is computed strictly from completed games prior to each date "
           "('As of' reflects games through the previous day).")
