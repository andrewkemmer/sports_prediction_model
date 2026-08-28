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
# contract; sports_config.active_page_url_paths resolves the active sport's
# ordered dashboard set (Today's Games is always present; markets is
# MLB-only). Home.py pairs each ``st.Page`` with its DECLARED url_path by
# zip (never reading ``p.url_path`` off the objects — Streamlit only attaches
# that inside st.navigation and reading it pre-attach raises AttributeError
# or returns "", the deployed line-81 crash that hid Today's Games).
_sport = str(st.session_state.get("sport", sports_config.DEFAULT_SPORT)).strip().lower()
_sport_config = sports_config.resolve_sport(_sport)
if _sport not in sports_config.SPORTS:
    # The toggle can never KeyError on an unknown sport — fall back to MLB
    # with a visible warning.
    st.warning(f"Unknown sport '{_sport}' — showing {_sport_config['label']}.")
_active_url_paths = sports_config.active_page_url_paths(_sport)
_active_pages = [
    page for page, url_path in zip(pages, sports_config.ALL_PAGE_URL_PATHS)
    if url_path in _active_url_paths
]
# Never render an empty nav: if the zip above dropped every page, fall back
# to the full set so the app never shows a blank sidebar (and Today's Games
# is always present).
if not _active_pages:
    _active_pages = list(pages)
nav = st.navigation(_active_pages, position="sidebar")
nav.run()
