"""MLB Predictions — Streamlit entry point.

Defines the five-page navigation used across the app (the single source of
truth for the sidebar page order):

    Today's Games · Power Rankings · Calibration · Model Monitor ·
    Totals & Run Lines

Artifacts are fetched from raw.githubusercontent URLs (configurable via the
GITHUB_OWNER / GITHUB_REPO / GITHUB_BRANCH env vars) with a local fallback
to the real committed artifacts in ``data_delivery/`` when GitHub is
unavailable.

Multi-sport restructure (Phase B): the sidebar carries a sport selector
above the dashboard nav — a single entry today (MLB, the default) rendered
through ``sports_config.SPORTS`` so switching sports later needs no layout
change. The per-sport nav list is built from the same registry; the literal
``pages`` list below remains the sidebar-order contract.

Run from the repository root::

    streamlit run frontend/Home.py
"""

import streamlit as st

import sports_config
import utils

st.set_page_config(
    page_title="MLB Predictions",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

utils.inject_css()

# ---------------------------------------------------------------------------
# Sidebar: branding + sport toggle + GitHub source configuration (shared)
# ---------------------------------------------------------------------------
with st.sidebar:
    # Single branding block — rendered ABOVE the dashboard list by the
    # sidebar reorder CSS in utils.inject_css.
    utils.render_brand_header()
    # Sport selector (multi-sport restructure, Phase B): one entry today
    # (MLB), rendered as a toggle above the dashboard nav so switching
    # sports later needs no layout change. The per-sport nav list below is
    # built from sports_config.SPORTS.
    st.segmented_control(
        "Sport",
        options=list(sports_config.SPORTS.keys()),
        format_func=lambda s: f"{sports_config.SPORTS[s]['emoji']} {sports_config.SPORTS[s]['label']}",
        default=sports_config.DEFAULT_SPORT,
        key="sport",
    )
    # The artifact fetch honors GITHUB_OWNER / GITHUB_REPO / GITHUB_BRANCH
    # env vars with a local fallback to the committed data_delivery
    # artifacts. A one-line note renders only when that fallback is active
    # (see render_source_note).
    utils.render_source_note()
    # Divider keeps the menu sitting cleanly under the logo/subtitle now
    # that the backend-source caption is gone.
    st.divider()

# ---------------------------------------------------------------------------
# Navigation (single source of truth for the sidebar page order)
# ---------------------------------------------------------------------------
pages = [
    st.Page("todays_games.py", title="Today's Games", icon="📅", url_path="todays-games", default=True),
    st.Page("power_rankings.py", title="Power Rankings", icon="🏆", url_path="power-rankings"),
    st.Page("model_calibration.py", title="Calibration", icon="📊", url_path="calibration"),
    st.Page("model_monitor.py", title="Model Monitor", icon="🛰️", url_path="model-monitor"),
    st.Page("markets.py", title="Totals & Run Lines", icon="🎯", url_path="markets"),
]
# Per-sport nav list — the literal `pages` list above is the sidebar-order
# contract; the active sport's page set (sports_config) selects which
# dashboards render (the run-engine pages are MLB-only; the generic pages
# render any sport that publishes the shared artifact contract).
_sport = st.session_state.get("sport", sports_config.DEFAULT_SPORT)
_active_pages = [p for p in pages if p.url_path in sports_config.SPORTS[_sport]["pages"]]
nav = st.navigation(_active_pages, position="sidebar")
nav.run()
