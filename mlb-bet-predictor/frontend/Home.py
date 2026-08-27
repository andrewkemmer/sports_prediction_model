"""MLB Predictions — Streamlit entry point.

Defines the four-page navigation used across the app:

    Today's Games · Power Rankings · Calibration · Model Monitor

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
    st.markdown(
        "<div style='font-size:1.25rem;font-weight:800;color:#E2E8F0;'>⚾ MLB Predictions</div>"
        "<div style='color:#64748B;font-size:0.8rem;margin-bottom:10px;'>MLB betting model dashboard</div>",
        unsafe_allow_html=True,
    )
    # Data-source caption block removed (display-only): the artifact fetch
    # with local fallback is unchanged and still honors GITHUB_OWNER /
    # GITHUB_REPO / GITHUB_BRANCH env vars. A one-line note renders only
    # when the app fell back to the bundled samples (see render_source_note).
    utils.render_source_note()
    st.divider()
    st.caption("Backend: Colab pipeline → data_delivery → GitHub. See README.md.")

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
pages = [
    st.Page("todays_games.py", title="Today's Games", icon="📅", url_path="todays-games", default=True),
    st.Page("markets.py", title="Totals & Run Lines", icon="🎯", url_path="markets"),
    st.Page("power_rankings.py", title="Power Rankings", icon="🏆", url_path="power-rankings"),
    st.Page("model_calibration.py", title="Calibration", icon="📊", url_path="calibration"),
    st.Page("model_monitor.py", title="Model Monitor", icon="🛰️", url_path="model-monitor"),
]
nav = st.navigation(pages, position="sidebar")
nav.run()
