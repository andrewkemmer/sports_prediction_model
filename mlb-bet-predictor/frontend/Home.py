"""MLB Predictions — Streamlit entry point.

Defines the five-page navigation used across the app (the single source of
truth for the sidebar page order):

    Today's Games · Power Rankings · Calibration · Model Monitor ·
    Totals & Run Lines

Artifacts are fetched from raw.githubusercontent URLs (configurable via the
GITHUB_OWNER / GITHUB_REPO / GITHUB_BRANCH env vars) with a local fallback
to the bundled sample data shipped in ``data_delivery/``.

Run from the repository root::

    streamlit run frontend/Home.py
"""

import streamlit as st

import utils

st.set_page_config(
    page_title="MLB Predictions",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

utils.inject_css()

# ---------------------------------------------------------------------------
# Sidebar: branding + GitHub source configuration (shared across pages)
# ---------------------------------------------------------------------------
with st.sidebar:
    # Single branding block — rendered ABOVE the dashboard list by the
    # sidebar reorder CSS in utils.inject_css (a sport toggle can be added
    # to this header later without touching the page list below).
    utils.render_brand_header()
    # Data-source caption block removed (display-only): the artifact fetch
    # with local fallback is unchanged and still honors GITHUB_OWNER /
    # GITHUB_REPO / GITHUB_BRANCH env vars. A one-line note renders only
    # when the app fell back to the bundled samples (see render_source_note).
    utils.render_source_note()
    st.divider()
    st.caption("Backend: Colab pipeline → data_delivery → GitHub. See README.md.")

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
nav = st.navigation(pages, position="sidebar")
nav.run()
