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
