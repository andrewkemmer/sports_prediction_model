"""Shared frontend helpers.

This module is the single place the four pages talk to the artifact sink
(``data_delivery``). It reads artifacts from **raw.githubusercontent.com**
URLs when a repo is configured, and transparently falls back to the sample
artifacts shipped in the repo's local ``data_delivery/`` folder so the app
always renders, even offline or before the first Colab push.

Only pandas / requests / altair / streamlit are used — no sklearn, xgboost,
lightgbm, or shap (heavy ML stays in the backend).
"""

from __future__ import annotations

import io
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_DATA_DIR = ROOT_DIR / "data_delivery"
# Artifacts live under mlb-bet-predictor/data_delivery/ inside the GitHub repo
REPO_SUBDIR = "mlb-bet-predictor"

PRIMARY = "#10B981"      # emerald accent
BLUE = "#3B82F6"
RED = "#EF4444"
AMBER = "#F59E0B"
SLATE = "#94A3B8"
TEXT = "#E2E8F0"
CARD_BG = "#131B2E"
BORDER = "#1E293B"
PAGE_BG = "#0A1128"


# ---------------------------------------------------------------------------
# Repo / source configuration (session state, set by Home.py)
# ---------------------------------------------------------------------------

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def get_source_config() -> dict:
    """Return the current GitHub source config from session state or env."""
    return {
        "owner": st.session_state.get("gh_owner", "") or _env("GITHUB_OWNER"),
        "repo": st.session_state.get("gh_repo", "") or _env("GITHUB_REPO"),
        "branch": st.session_state.get("gh_branch", "") or _env("GITHUB_BRANCH") or "main",
    }


def source_label() -> str:
    cfg = get_source_config()
    if cfg["owner"] and cfg["repo"]:
        return f"GitHub raw · {cfg['owner']}/{cfg['repo']}@{cfg['branch']}"
    return "Local sample data (no GitHub repo configured)"


# ---------------------------------------------------------------------------
# Artifact loading (GitHub raw -> local sample fallback)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_bytes(relpath: str, owner: str, repo: str, branch: str):
    """Fetch one artifact. Returns (bytes | None, source)."""
    if owner and repo:
        url = (f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}"
               f"/{REPO_SUBDIR}/data_delivery/{relpath}")
        try:
            resp = requests.get(url, timeout=15)
            if resp.ok:
                return resp.content, "github"
        except requests.RequestException:
            pass
    local = LOCAL_DATA_DIR / relpath
    if local.exists():
        return local.read_bytes(), "local"
    return None, "missing"


@st.cache_data(ttl=300, show_spinner=False)
def available_dates(owner: str, repo: str, branch: str) -> list[str]:
    """All artifact dates (YYYYMMDD), newest first, from GitHub API or local dir."""
    dates: set[str] = set()
    if owner and repo:
        try:
            api = (f"https://api.github.com/repos/{owner}/{repo}/contents"
                   f"/{REPO_SUBDIR}/data_delivery")
            resp = requests.get(api, timeout=15)
            if resp.ok:
                for item in resp.json():
                    name = item.get("name", "")
                    if name.startswith("todays_games_") and name.endswith(".csv"):
                        dates.add(name[len("todays_games_"):-len(".csv")])
        except requests.RequestException:
            pass
    for p in LOCAL_DATA_DIR.glob("todays_games_*.csv"):
        dates.add(p.name[len("todays_games_"):-len(".csv")])
    return sorted(dates, reverse=True)


def _pick_date(date_str: str) -> str:
    dates = available_dates(**get_source_config())
    if date_str and date_str in dates:
        return date_str
    if dates:
        return dates[0]
    return "20260809"  # bundled sample


def load_todays_games(date_str: str) -> pd.DataFrame:
    cfg = get_source_config()
    data, src = _fetch_bytes(f"todays_games_{_pick_date(date_str)}.csv", **cfg)
    st.session_state["data_source"] = src
    if data is None:
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(data))
    for col in ["home_team_name", "away_team_name", "model_pick", "final_inning", "venue"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


def load_power_rankings(date_str: str) -> pd.DataFrame:
    cfg = get_source_config()
    data, src = _fetch_bytes(f"power_rankings_{_pick_date(date_str)}.csv", **cfg)
    st.session_state["data_source"] = src
    if data is None:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(data))


def load_calibration(date_str: str) -> dict:
    cfg = get_source_config()
    data, src = _fetch_bytes(f"calibration_{_pick_date(date_str)}.json", **cfg)
    st.session_state["data_source"] = src
    if data is None:
        return {}
    return json.loads(data)


def load_model_monitor(date_str: str) -> dict:
    cfg = get_source_config()
    data, src = _fetch_bytes(f"model_monitor_{_pick_date(date_str)}.json", **cfg)
    st.session_state["data_source"] = src
    if data is None:
        return {}
    return json.loads(data)


def load_shap(game_id: str, date_str: str) -> pd.DataFrame:
    cfg = get_source_config()
    data, _ = _fetch_bytes(f"shap_game_{game_id}.csv", **cfg)
    if data is None:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(data))


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def parse_date(date_str: str) -> date:
    return date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))


def format_date_long(date_str: str) -> str:
    d = parse_date(date_str)
    return f"{d.strftime('%A, %B')} {d.day}, {d.year}"


def format_date_short(date_str: str) -> str:
    d = parse_date(date_str)
    return f"{d.strftime('%b')} {d.day}, {d.year}"


def start_time_et(iso_utc: str) -> str:
    """Convert a UTC ISO timestamp to an '7:05 PM ET' label."""
    try:
        ts = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        et = ts.astimezone(ZoneInfo("America/New_York"))
        hour = et.hour % 12 or 12
        return f"{hour}:{et.strftime('%M')} {et.strftime('%p')} ET"
    except (ValueError, TypeError):
        return ""


def arrow_nav(dates: list[str]) -> None:
    """Render ◀ date ▶ navigation over the available artifact dates."""
    dates = sorted(dates)
    if not dates:
        st.session_state["selected_date"] = "20260809"
        return
    current = st.session_state.get("selected_date", dates[-1])
    if current not in dates:
        current = dates[-1]
    idx = dates.index(current)
    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        if st.button("◀", key="prev_day", help="Previous day", use_container_width=True):
            st.session_state["selected_date"] = dates[max(0, idx - 1)]
            st.rerun()
    with c2:
        st.markdown(
            f"<div style='text-align:center;font-size:1.05rem;font-weight:600;color:{TEXT};"
            f"padding:0.45rem 0;border:1px solid {BORDER};border-radius:10px;background:{CARD_BG};'>"
            f"{format_date_long(current)}</div>",
            unsafe_allow_html=True,
        )
    with c3:
        if st.button("▶", key="next_day", help="Next day", use_container_width=True):
            st.session_state["selected_date"] = dates[min(len(dates) - 1, idx + 1)]
            st.rerun()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def pct(x, digits: int = 0) -> str:
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def american(odds) -> str:
    try:
        v = int(odds)
        return f"{v:+d}"
    except (TypeError, ValueError):
        return "—"


def edge_str(edge) -> str:
    try:
        return f"{float(edge):+.2f}"
    except (TypeError, ValueError):
        return "+0.00"


def edge_color(edge) -> str:
    try:
        v = float(edge)
    except (TypeError, ValueError):
        return SLATE
    if v > 0.001:
        return PRIMARY
    if v < -0.001:
        return RED
    return BLUE


def record_pct(w: int, l: int) -> str:
    total = w + l
    return f"{w / total:.3f}" if total else ".000"


# ---------------------------------------------------------------------------
# Altair dark theme
# ---------------------------------------------------------------------------

def _fb_theme():
    return {
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent", "continuousWidth": 400, "continuousHeight": 260},
            "axis": {
                "labelColor": SLATE, "titleColor": TEXT, "gridColor": BORDER,
                "domainColor": "#334155", "tickColor": "#334155", "labelFontSize": 11,
            },
            "legend": {"labelColor": SLATE, "titleColor": TEXT, "labelFontSize": 11},
            "title": {"color": TEXT, "fontSize": 15, "anchor": "start"},
            "range": {"category": [BLUE, PRIMARY, AMBER, "#A78BFA", RED]},
        }
    }


alt.themes.register("fb_dark", _fb_theme)
alt.themes.enable("fb_dark")


def show_chart(chart: alt.Chart) -> None:
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:  # older Streamlit
        st.altair_chart(chart, use_container_width=True)


def shap_chart(shap_df: pd.DataFrame, max_features: int = 12) -> alt.Chart:
    """Horizontal SHAP bar chart: green positive bars, red negative bars,
    signed value labels, sorted by |value| descending."""
    d = shap_df.copy()
    if d.empty:
        return None
    d = d[d["shap_value"].notna()].copy()
    d["abs"] = d["shap_value"].abs()
    d = d.sort_values("abs", ascending=False).head(max_features).reset_index(drop=True)
    d["label"] = d["shap_value"].map(lambda v: f"{v:+.2f}")

    # Explicit shared x-domain avoids Vega-Lite "Infinite extent" warnings
    # from empty per-layer scale extents in layered charts.
    pad = max(0.005, float(d["shap_value"].abs().max()) * 0.08)
    domain = [float(d["shap_value"].min()) - pad, float(d["shap_value"].max()) + pad]
    x_enc = alt.X("shap_value:Q", title="SHAP value (P home win)", scale=alt.Scale(domain=domain))

    bars = alt.Chart(d).mark_bar(cornerRadiusEnd=3).encode(
        x=x_enc,
        y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=170)),
        color=alt.condition("datum.shap_value >= 0", alt.value(PRIMARY), alt.value(RED)),
    )
    pos_data, neg_data = d[d["shap_value"] >= 0], d[d["shap_value"] < 0]
    layers = [bars]
    if not pos_data.empty:
        layers.append(
            alt.Chart(pos_data)
            .mark_text(dx=5, align="left", color="#34D399", fontSize=11)
            .encode(x=x_enc, y=alt.Y("feature:N", sort=None), text="label:N")
        )
    if not neg_data.empty:
        layers.append(
            alt.Chart(neg_data)
            .mark_text(dx=-5, align="right", color="#F87171", fontSize=11)
            .encode(x=x_enc, y=alt.Y("feature:N", sort=None), text="label:N")
        )
    return alt.layer(*layers).properties(height=max(120, 26 * len(d)))


# ---------------------------------------------------------------------------
# Shared page chrome (CSS)
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {PAGE_BG}; }}
        .block-container {{ padding-top: 1.4rem; }}
        .fb-card {{
            background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 14px;
            padding: 14px 16px 12px; margin-bottom: 12px;
        }}
        .fb-top {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }}
        .fb-top .spacer {{ flex: 1; }}
        .fb-tag {{ color: {SLATE}; font-size: 0.78rem; }}
        .fb-pill {{ border-radius: 999px; padding: 2px 9px; font-size: 0.72rem; font-weight: 700; white-space: nowrap; }}
        .fb-pill.correct {{ background: rgba(16,185,129,.15); color: #34D399; }}
        .fb-pill.miss {{ background: rgba(239,68,68,.15); color: #F87171; }}
        .fb-pill.final {{ background: rgba(59,130,246,.15); color: #93C5FD; }}
        .fb-pill.upset {{ background: rgba(245,158,11,.18); color: #FBBF24; }}
        .fb-pill.coinflip {{ background: rgba(250,204,21,.18); color: #FDE047; }}
        .fb-pill.live {{ background: rgba(239,68,68,.18); color: #F87171; }}
        .fb-pill.pick {{ background: rgba(59,130,246,.9); color: #fff; }}
        .fb-score {{ display: flex; align-items: center; justify-content: center; gap: 26px; margin: 6px 0 4px; }}
        .fb-score .side {{ text-align: center; }}
        .fb-score .num {{ font-size: 2.1rem; font-weight: 800; line-height: 1; }}
        .fb-score .abbr {{ color: {SLATE}; font-size: 0.8rem; letter-spacing: 1px; }}
        .fb-score .mid {{ color: {SLATE}; font-size: 0.95rem; font-weight: 600; }}
        .fb-team {{ display: flex; align-items: center; gap: 8px; margin: 7px 0 3px; }}
        .fb-accent {{ width: 4px; border-radius: 3px; align-self: stretch; min-height: 26px; }}
        .fb-team .name {{ font-weight: 700; color: {TEXT}; }}
        .fb-team .sub {{ color: {SLATE}; font-size: 0.8rem; }}
        .fb-team .pct {{ margin-left: auto; font-weight: 800; font-size: 1.05rem; }}
        .fb-bar {{ height: 7px; border-radius: 5px; background: #0F172A; overflow: hidden; display: flex; margin: 3px 0 2px; }}
        .fb-bar .fill {{ height: 100%; }}
        .fb-pregame {{ text-align: center; color: #64748B; font-size: 0.78rem; margin: 6px 0 8px; }}
        .fb-pitchers {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 4px 0; }}
        .fb-pitcher {{ background: #0F172A; border: 1px solid {BORDER}; border-radius: 9px; padding: 7px 9px; }}
        .fb-pitcher .pname {{ font-weight: 700; color: {TEXT}; font-size: 0.86rem; }}
        .fb-pitcher .pstats {{ color: {SLATE}; font-size: 0.76rem; margin-top: 2px; }}
        .fb-venue {{ color: {SLATE}; font-size: 0.8rem; margin: 6px 0 2px; }}
        .fb-odds {{ display: flex; justify-content: space-between; align-items: center; margin: 4px 0 8px; color: {TEXT}; font-size: 0.85rem; }}
        .fb-odds .edge {{ font-weight: 700; }}
        .fb-banner {{ border-radius: 10px; padding: 8px 12px; text-align: center; font-size: 0.85rem; font-weight: 700; }}
        .fb-banner.green {{ background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.45); color: #34D399; }}
        .fb-banner.red {{ background: rgba(239,68,68,.10); border: 1px solid rgba(239,68,68,.45); color: #F87171; }}
        .fb-banner.amber {{ background: rgba(250,204,21,.10); border: 1px solid rgba(250,204,21,.5); color: #FDE047; }}
        .fb-banner.blue {{ background: rgba(59,130,246,.10); border: 1px solid rgba(59,130,246,.45); color: #93C5FD; }}
        .fb-kpi {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px; text-align: center; }}
        .fb-kpi .label {{ color: {SLATE}; font-size: 0.72rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }}
        .fb-kpi .value {{ font-size: 1.9rem; font-weight: 800; margin: 4px 0 2px; }}
        .fb-kpi .cap {{ color: {SLATE}; font-size: 0.72rem; }}
        .fb-box {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 12px; padding: 12px 16px; }}
        .fb-status-pill {{ border-radius: 999px; padding: 1px 10px; font-size: 0.72rem; font-weight: 700; }}
        .fb-status-pill.ok {{ background: rgba(16,185,129,.15); color: #34D399; }}
        .fb-status-pill.warn {{ background: rgba(245,158,11,.15); color: #FBBF24; }}
        .fb-status-pill.alert {{ background: rgba(239,68,68,.15); color: #F87171; }}
        table.fb-table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
        table.fb-table th {{ text-align: left; color: {SLATE}; font-size: 0.72rem; letter-spacing: 1px; text-transform: uppercase; padding: 8px 10px; border-bottom: 1px solid {BORDER}; }}
        table.fb-table td {{ padding: 8px 10px; border-bottom: 1px solid rgba(30,41,59,.6); color: {TEXT}; }}
        table.fb-table tr:nth-child(even) td {{ background: rgba(30,41,59,.25); }}
        .fb-accent-cell {{ display: inline-flex; align-items: center; gap: 8px; }}
        .fb-accent-line {{ width: 4px; border-radius: 3px; height: 20px; display: inline-block; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_source_note() -> None:
    src = st.session_state.get("data_source", "")
    if src == "github":
        note, icon = "Streaming from GitHub raw URLs", "🌐"
    elif src == "local":
        note, icon = "Showing bundled sample data (offline fallback)", "📦"
    elif src == "missing" and not list(LOCAL_DATA_DIR.glob("todays_games_*.csv")):
        note, icon = "No artifacts found", "⚠️"
    else:
        note, icon = "Showing bundled sample data (offline fallback)", "📦"
    st.caption(f"{icon} {note}")
