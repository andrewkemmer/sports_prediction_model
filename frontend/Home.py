"""MLB Predictions — Streamlit entry point.

Defines the five-page navigation used across the app (the single source of
truth for the sidebar page order):

    Today's Games · Power Rankings · Calibration · Model Monitor ·
    Totals & Run Lines

Artifacts are fetched from raw.githubusercontent URLs (configurable via the
GITHUB_OWNER / GITHUB_REPO / GITHUB_BRANCH env vars) with a local fallback
to the real committed artifacts in ``data_delivery/`` when GitHub is
unavailable.

Multi-sport restructure (Phase B): the sidebar carries a registry-driven
sport picker (``st.pills`` over ``sports_config.SPORTS``) above the
dashboard nav, and the brand header title/subtitle also come from the
registry — adding a sport (MLB, NFL, NBA, NHL, ...) needs zero UI-code
changes. ``st.session_state["sport"]`` is the single source of truth
(utils.get_sport() reads it); the picker widget mirrors it via its own
key and writes changes back. The per-sport nav list is built from the
same registry; the literal ``pages`` list below remains the
sidebar-order contract.

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

# Set the sport default BEFORE the toggle/widget renders. A rerun that
# navigates (e.g. clicking the brand above the dashboard list) can reach
# this point with session_state["sport"] unset/None before the sport picker
# materializes — setdefault guarantees it defaults to the first registry
# key (MLB) silently instead of surfacing an "Unknown sport" warning on
# every such click. Stale/unknown values are normalized to a valid key
# here so the widget always renders against a real registry entry.
st.session_state.setdefault("sport", sports_config.DEFAULT_SPORT)
_sport_key = sports_config.normalize_sport_key(st.session_state["sport"])
if _sport_key != st.session_state["sport"]:
    st.session_state["sport"] = _sport_key

# ---------------------------------------------------------------------------
# Sidebar: branding + sport toggle + GitHub source configuration (shared)
# ---------------------------------------------------------------------------
with st.sidebar:
    # Single branding block — rendered ABOVE the dashboard list by the
    # sidebar reorder CSS in utils.inject_css.
    utils.render_brand_header()
    # Sport selector — a registry-driven pills toggle (Streamlit >= 1.40;
    # installed 1.62.0). The WIDGET is a mirror with its own key
    # ("sport_picker"); st.session_state["sport"] stays the single source of
    # truth that every loader/page reads via utils.get_sport(). A picked
    # value that differs is written back to session_state and triggers a
    # rerun so the whole sidebar/header/nav render for the new sport.
    _picked = st.pills(
        "Sport",
        options=list(sports_config.SPORTS.keys()),
        format_func=lambda s: f"{sports_config.SPORTS[s]['emoji']} {sports_config.SPORTS[s]['label']}",
        selection_mode="single",
        default=st.session_state["sport"],
        key="sport_picker",
        label_visibility="collapsed",
    )
    if _picked is not None and _picked != st.session_state["sport"]:
        st.session_state["sport"] = _picked
        st.rerun()
    # Defensive empty-state note (only when the sport ships no artifacts at
    # all); the former fetch-failure warning is gone. In its place a muted,
    # sport-aware 'Last updated' line resolves the ACTIVE sport toggle's
    # committed artifact set.
    utils.render_source_note()
    # 'Last updated' resolves the ACTIVE sport's committed artifact set
    # (MLB stocks CSV cards/calibration; NFL stocks JSON moneyline records).
    utils.render_last_updated(utils.get_sport())
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
_sport = utils.get_sport()
_sport_config = sports_config.resolve_sport(_sport)
if sports_config.is_unknown_sport(_sport):
    # Only a genuinely unknown NON-EMPTY sport warns. None/""/"none" (the
    # missing-default state when a navigation rerun resets the toggle before
    # it renders) fall back to MLB silently — no noise.
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
