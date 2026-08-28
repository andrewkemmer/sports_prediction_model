"""Sport registry for the shared Streamlit frontend (multi-sport restructure).

Single source of truth for:

* which sports exist and their labels/emojis,
* the repo-relative directory each sport publishes artifacts from — this is
  both the GitHub raw-URL prefix AND the local committed-artifact fallback
  dir, so flipping it (Phase C: ``mlb-backend`` -> ``mlb-backend``)
  moves the fetcher and the local sink atomically,
* which dashboards each sport supports (the run-engine pages are MLB-only;
  the four generic dashboards render any sport that publishes the shared
  contract).

The sport toggle in ``Home.py`` writes the selected sport to
``st.session_state["sport"]``; ``utils.py`` resolves artifact paths through
this registry. Keeping this module free of streamlit/pandas imports lets the
backend tests import it without a Streamlit runtime.

Future sports (nfl/nba/nhl) add one registry entry here plus their
``<sport>-backend/data_delivery/`` contract — no frontend layout changes.
"""

SPORTS = {
    "mlb": {
        "label": "MLB",
        "emoji": "⚾",
        # Repo-relative directory holding this sport's backend + data_delivery.
        # Phase C renames the directory to mlb-backend/ — flip ONLY this value.
        "repo_subdir": "mlb-backend",
        # MLB publishes run-engine artifacts (Totals & Run Lines, umpire/roof/
        # weather features); other sports render the generic dashboards only.
        "has_run_engine": True,
        # Sidebar page order (url_paths, matching Home.py's literal pages list).
        "pages": [
            "todays-games",
            "power-rankings",
            "calibration",
            "model-monitor",
            "markets",
        ],
    },
}

DEFAULT_SPORT = "mlb"

# Every dashboard the app can render, in the fixed sidebar-registration order
# that MUST mirror Home.py's literal ``pages = [st.Page(...)]`` list (the
# sidebar-order contract). url_path is the string Home.py passes to
# st.Page(url_path=...); it is NOT read back off the Page objects because
# Streamlit only attaches it inside st.navigation.
ALL_PAGE_URL_PATHS = [
    "todays-games",
    "power-rankings",
    "calibration",
    "model-monitor",
    "markets",
]


def resolve_sport(sport_key: str) -> dict:
    """Resolve a segmented-control value to a sport config, never raising.

    - Coerces to lowercase + strips whitespace (the toggle may return a
      display label rather than the config key on some Streamlit versions).
    - Falls back to DEFAULT_SPORT when the key is unknown or missing, so an
      unknown sport degrades to MLB instead of KeyErroring.

    Returns the sport config dict (always a valid SPORTS entry).
    """
    key = str(sport_key or "").strip().lower()
    if key not in SPORTS:
        key = DEFAULT_SPORT
    return SPORTS[key]


def active_page_url_paths(sport_key: str) -> list[str]:
    """The ordered subset of ALL_PAGE_URL_PATHS the given sport renders.

    Today's Games (``todays-games``) is a shared-contract page and is ALWAYS
    present for every sport; the run-engine pages (markets) are MLB-only.
    The ordering follows ALL_PAGE_URL_PATHS, which mirrors Home.py's literal
    ``pages`` list (the sidebar-order contract).

    A sport whose page set is empty/list is exhausted degrades to the full
    set so the sidebar never renders blank.
    """
    cfg = resolve_sport(sport_key)
    allowed = cfg.get("pages", []) or []
    active = [p for p in ALL_PAGE_URL_PATHS if p in allowed]
    if not active:
        return list(ALL_PAGE_URL_PATHS)
    return active
