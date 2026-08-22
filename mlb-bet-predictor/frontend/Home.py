"""MLB Predictions — Streamlit entry point.

Defines the four-page navigation used across the app:

    Today's Games · Power Rankings · Calibration · Model Monitor

The sidebar lets you point the app at your GitHub repo (raw.githubusercontent
URLs are used for every artifact). Leave it blank to render the bundled
sample data shipped in ``data_delivery/``.

Run from the repository root::

    streamlit run frontend/Home.py
"""

import os

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
    st.markdown("#### Data source")
    owner_default = st.session_state.get("gh_owner", "") or os.environ.get("GITHUB_OWNER", "")
    repo_default = st.session_state.get("gh_repo", "") or os.environ.get("GITHUB_REPO", "")
    branch_default = st.session_state.get("gh_branch", "") or os.environ.get("GITHUB_BRANCH", "") or "main"
    st.text_input("GitHub owner", value=owner_default, key="gh_owner",
                  help="e.g. your-github-username")
    st.text_input("GitHub repo", value=repo_default, key="gh_repo",
                  help="e.g. mlb-bet-predictor")
    st.text_input("Branch", value=branch_default, key="gh_branch")
    st.caption(
        "Artifacts are fetched from "
        "`raw.githubusercontent.com/<owner>/<repo>/<branch>/mlb-bet-predictor/data_delivery/…` "
        "with a local fallback to the bundled samples."
    )
    utils.render_source_note()
    st.divider()
    st.caption("Backend: Colab pipeline → data_delivery → GitHub. See README.md.")

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
pages = [
    st.Page("todays_games.py", title="Today's Games", icon="📅", url_path="todays-games", default=True),
    st.Page("power_rankings.py", title="Power Rankings", icon="🏆", url_path="power-rankings"),
    st.Page("model_calibration.py", title="Calibration", icon="📊", url_path="calibration"),
    st.Page("model_monitor.py", title="Model Monitor", icon="🛰️", url_path="model-monitor"),
]
nav = st.navigation(pages, position="sidebar")
nav.run()
