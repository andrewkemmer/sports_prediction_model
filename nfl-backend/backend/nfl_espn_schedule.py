"""ESPN 2026 NFL schedule loader for the moneyline slate.

nflreadpy caps its season feed at the latest nflverse season it knows (2025),
so there is no 2026 schedule available through nflreadpy yet. To let the
pipeline populate predictions for the whole 2026 season *before* week 1, this
module pulls the scheduled (unplayed) games straight from ESPN's public
scoreboard API — the same source the MLB backend uses for its upcoming slate.

Fetch strategy
--------------
One request per REGULAR-SEASON WEEK (``season=2026&seasontype=2&week=N``,
N = 1..18) instead of per calendar date. A per-date sweep is ~153 requests and
gets throttled/403-blocked from Kaggle; per-week is ~18 requests and returns a
whole week's games in each response (plus the ``week`` directly, so no date-to-
week estimation). The week-1..18 regular season covers every game a "today's
games" board can show across the year.

The returned frame mirrors the nflreadpy ``load_schedules`` column names so it
can be concatenated onto the decided schedule and consumed unchanged by
``build_slate_features``: each row carries ``game_id``, ``season``, ``week``,
``gameday`` (venue-local ET date), ``gametime``, ``stadium``, ``home_team`` /
``away_team`` (nflverse abbreviations via the ESPN alias map), ``home_score`` /
``away_score`` (NaN — these are scheduled rows), plus ``roof``, ``temp``,
``wind`` and ``div_game`` synthesized from per-team facts (roof/temp/wind for a
pre-game row are best-effort; ``div_game`` is exact from membership).

Design
------
- ``parse_event`` / ``season_weeks`` / ``team_alias`` are pure and unit testable
  without network.
- ``load_espn_schedule_rows`` does the network pull: ~18 weekly requests, fetched
  with a light thread pool, each with per-request UA + retry. Rows with a state
  other than ``pre`` (already started/post) are dropped so only scheduled games
  are added.
"""

from __future__ import annotations

import concurrent.futures
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
_UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "sports_prediction_model nfl-slate")}

# Number of NFL regular-season weeks (seasontype=2) ESPN serves.
_REGULAR_WEEKS = range(1, 19)  # 1..18

# --- ESPN abbreviation -> nflverse abbreviation (the internal key) -----------
# nflreadpy game_ids / home_team / away_team use the nflverse abbreviations;
# ESPN uses the same codes with a handful of differences (LA/LAR, JAC/JAX, WSH/WAS).
ESPN_ABBREV_TO_KEY = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF", "CAR": "CAR",
    "CHI": "CHI", "CIN": "CIN", "CLE": "CLE", "DAL": "DAL", "DEN": "DEN",
    "DET": "DET", "GB": "GB", "HOU": "HOU", "IND": "IND", "JAC": "JAX",
    "JAX": "JAX", "KC": "KC", "LAC": "LAC", "LA": "LAR", "LAR": "LAR",
    "LV": "LV", "MIA": "MIA", "MIN": "MIN", "NE": "NE", "NO": "NO",
    "NYG": "NYG", "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT", "SEA": "SEA",
    "SF": "SF", "TB": "TB", "TEN": "TEN", "WAS": "WAS", "WSH": "WAS",
}

# division membership by nflverse abbreviation -> (conference, division)
_TEAM_DIVISION = {
    "BUF": ("AFC", "East"), "MIA": ("AFC", "East"), "NE": ("AFC", "East"),
    "NYJ": ("AFC", "East"),
    "BAL": ("AFC", "North"), "CIN": ("AFC", "North"), "CLE": ("AFC", "North"),
    "PIT": ("AFC", "North"),
    "HOU": ("AFC", "South"), "IND": ("AFC", "South"), "JAX": ("AFC", "South"),
    "TEN": ("AFC", "South"),
    "DEN": ("AFC", "West"), "KC": ("AFC", "West"), "LV": ("AFC", "West"),
    "LAC": ("AFC", "West"),
    "DAL": ("NFC", "East"), "NYG": ("NFC", "East"), "PHI": ("NFC", "East"),
    "WAS": ("NFC", "East"),
    "CHI": ("NFC", "North"), "DET": ("NFC", "North"), "GB": ("NFC", "North"),
    "MIN": ("NFC", "North"),
    "ATL": ("NFC", "South"), "CAR": ("NFC", "South"), "NO": ("NFC", "South"),
    "TB": ("NFC", "South"),
    "ARI": ("NFC", "West"), "LAR": ("NFC", "West"), "SEA": ("NFC", "West"),
    "SF": ("NFC", "West"),
}

# teams whose home venue is (normally) indoor/closed — best-effort roof for
# pre-game slate rows; all others treated as outdoors.
_DOME_TEAMS = {"ARI", "ATL", "DAL", "DET", "HOU", "IND", "LAC", "LAR", "MIN", "NO"}

_ET = ZoneInfo("America/New_York")


def team_alias(abbr: str) -> str | None:
    """Map an ESPN abbreviation to the nflverse key (None if unknown)."""
    return ESPN_ABBREV_TO_KEY.get(str(abbr).strip().upper())


def _get_event_date_et(iso_utc: str) -> date:
    """ESPN event.date is UTC; the game's DATE is the ET date (matches nflreadpy
    gameday semantics, keeps late-night games on the US day they're played)."""
    dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
    return dt.astimezone(_ET).date()


def season_weeks(season: int) -> list[int]:
    """Regular-season weeks (1..18) for the target season."""
    return list(_REGULAR_WEEKS)


def parse_event(event: dict, season: int, week: int) -> dict | None:
    """Convert one ESPN scoreboard event into an nflreadpy-shaped schedule row,
    or None when it isn't a scheduled (``pre``) game or a team looks unknown."""
    competition = (event.get("competitions") or [{}])[0]
    state = (competition.get("status") or {}).get("type", {}).get("state")
    if state != "pre":
        return None  # started/post games are never part of the pre-game slate
    comps = competition.get("competitors") or []
    home = away = None
    for c in comps:
        team = c.get("team") or {}
        key = team_alias((team.get("abbreviation") or ""))
        if key is None:
            continue
        if c.get("homeAway") == "home":
            home = key
        else:
            away = key
    if home is None or away is None:
        return None
    if home not in _TEAM_DIVISION or away not in _TEAM_DIVISION:
        return None

    eid = str(event.get("id") or "")
    if not eid:
        return None
    iso = event.get("date") or ""
    try:
        gameday = _get_event_date_et(iso)
    except Exception:  # noqa: BLE001 — a malformed date never crashes the slate
        gameday = None
    venue = (competition.get("venue") or {})
    stadium = venue.get("fullName") or "TBD"

    return {
        "game_id": f"{season}_espn_{eid}",
        "season": int(season),
        "week": int(week),
        "gameday": gameday,
        "gametime": iso,
        "stadium": stadium,
        "home_team": home,
        "away_team": away,
        "home_score": None,
        "away_score": None,
        "roof": "dome" if home in _DOME_TEAMS else "outdoors",
        "temp": None,
        "wind": None,
        "div_game": int(_TEAM_DIVISION[home] == _TEAM_DIVISION[away]),
    }


def _fetch_week(season: int, week: int, timeout: int = 25,
                attempts: int = 3) -> list[dict]:
    url = f"{SCOREBOARD_URL}?season={season}&seasontype=2&week={week}"
    for attempt in range(1, attempts + 1):
        try:
            import requests
            resp = requests.get(url, headers=_UA, timeout=timeout)
            if resp.status_code == 200:
                return resp.json().get("events") or []
            if resp.status_code in (403, 429):
                time.sleep(1.0 * attempt)  # polite backoff on throttling
                continue
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.5 * attempt)
    return []


def load_espn_schedule_rows(season: int, max_workers: int = 4) -> pd.DataFrame:
    """Fetch every scheduled game in ``season`` from ESPN (one request per
    regular-season week) and return an nflreadpy-shaped pandas frame (empty if
    the feed has none)."""
    weeks = season_weeks(season)
    if not weeks:
        return pd.DataFrame()

    rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_week, season, w): w for w in weeks}
        for fut in concurrent.futures.as_completed(futures):
            week = futures[fut]
            for ev in fut.result():
                parsed = parse_event(ev, season, week)
                if parsed is not None:
                    rows.append(parsed)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = (df.drop_duplicates(subset="game_id", keep="first")
            .sort_values("gameday", kind="mergesort").reset_index(drop=True))
    return df