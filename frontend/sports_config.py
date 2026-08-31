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

from pathlib import Path


SPORTS = {
    "mlb": {
        "label": "MLB",
        "emoji": "⚾",
        # Sidebar brand header (title/subtitle) — rendered by
        # utils.render_brand_header from this registry so a new sport needs
        # zero UI-code changes.
        "title": "MLB Predictions",
        "subtitle": "MLB betting model dashboard",
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
        # Artifact resolver (step 2): sport → the dated artifact families under
        # ``<repo_subdir>/data_delivery`` the loader/adapter layer dispatches
        # on. Patterns are globs relative to that dir.
        "artifacts": {
            "todays_games_csv": "todays_games_*.csv",
            "markets_csv": "run_engine_markets_*.csv",
            "calibration_json": "calibration_*.json",
        },
    },
    "nfl": {
        "label": "NFL",
        "emoji": "🏈",
        "title": "NFL Predictions",
        "subtitle": "NFL betting model dashboard",
        "repo_subdir": "nfl-backend",
        "has_run_engine": False,
        # NFL ships the generic shared-contract dashboards only for now; the
        # run-engine Totals & Run Lines page (markets) is MLB-only. Today's
        # Games keeps a moneyline-first board; Calibration / Model Monitor /
        # Power Rankings render the shared contract or a step-3 notice.
        "pages": [
            "todays-games",
            "power-rankings",
            "calibration",
            "model-monitor",
        ],
        "artifacts": {
            "moneyline_json": "nfl_moneyline_v1_*.json",
            "feature_json": "nfl_feature_v1_*.json",
            "calibration_json": "nfl_calibration_*.json",
            "predictions_history_csv": "nfl_predictions_history_*.csv",
            "power_rankings_csv": "nfl_power_rankings_*.csv",
        },
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
    - Falls back to DEFAULT_SPORT when the key is unknown OR missing (None,
      "", whitespace), so any degraded state shows MLB instead of KeyErroring.

    Returns the sport config dict (always a valid SPORTS entry).
    """
    key = normalize_sport_key(sport_key)
    return SPORTS[key]


def normalize_sport_key(sport_key: str) -> str:
    """Normalize a segmented-control value to a valid SPORTS key.

    Lowercases + strips whitespace; unknown or empty values fall back to
    DEFAULT_SPORT. ``None``/""/whitespace (the missing-default state, e.g.
    an app rerun that resets the toggle) resolve to the default silently;
    a genuinely unknown non-empty string (e.g. "nfl") also resolves to the
    default so the sport is always valid.
    """
    key = str(sport_key or "").strip().lower()
    if key not in SPORTS:
        key = DEFAULT_SPORT
    return key


def is_unknown_sport(sport_key: str) -> bool:
    """True only for a genuinely unknown NON-EMPTY sport value.

    ``None``, empty string, and whitespace are the *missing-default* state
    (e.g. clicking the brand reruns the app before the toggle materializes)
    — not a real unknown sport — so they return False. Only a non-empty
    string that matches no SPORTS key triggers True (caller warns then).
    """
    key = str(sport_key or "").strip().lower()
    # None / "" / all-whitespace / the string 'none' are the missing-default
    # state — never warn (they silently fall back to MLB).
    if not key or key == "none":
        return False
    return key not in SPORTS


def artifact_patterns(sport_key: str) -> dict:
    """The sport → artifact-family map for the loader/adapter resolver.

    Returns ``config.artifacts`` (name → glob pattern under the sport's
    ``data_delivery`` dir). Falls back to {} for a degenerate/unregistered
    sport so callers can safely iterate. MLB carries the card/calibration
    CSV·JSON families; NFL carries its JSON-only v1 artifacts.
    """
    return dict(resolve_sport(sport_key).get("artifacts", {}) or {})


def data_delivery_dir(sport_key: str) -> Path:
    """Local ``data_delivery`` directory for the sport, resolved via the
    repo-relative ``repo_subdir`` registry entry."""
    return Path(__file__).resolve().parents[1] / resolve_sport(sport_key)["repo_subdir"] / "data_delivery"


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
