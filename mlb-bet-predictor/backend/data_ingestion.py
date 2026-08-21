"""Data ingestion and strict point-in-time feature construction.

This module is the heart of the point-in-time guarantee:

* Every feature for a game is computed **only from data whose timestamp is
  strictly prior to that game's scheduled start time**.
* Rolling aggregates use ``shift(1)`` so the current game is never included.
* Market lines are attached via an as-of join that only accepts lines whose
  posted timestamp is before the game start.

Two data sources are supported:

* ``synthetic=True`` (default) — a seeded, deterministic generator that
  produces a realistic MLB game log. This is what makes the pipeline run
  end-to-end in Colab with zero network dependencies.
* ``synthetic=False`` — the real path via ``pybaseball`` (schedule, Statcast
  game metadata, season-to-date pitcher/batter aggregates). Requires network
  access and ``pip install pybaseball``.

The normalized per-game table both paths produce is called *game events*
(one row per game). Feature construction consumes only that table, so the
real and synthetic paths share identical, tested logic.
"""

from __future__ import annotations

import logging
import math
from datetime import date as date_cls
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from config import (
    ELO_HOME_ADVANTAGE,
    ELO_K,
    ELO_MOV_MULTIPLIER,
    FEATURE_COLUMNS,
    RAW_DIR,
    ROLL_WINDOWS,
    SEED,
)

logger = logging.getLogger(__name__)

# ===========================================================================
# Static league data (teams, venues, colors, pitcher pools)
# ===========================================================================

MLB_TEAMS = [
    {"abbrev": "WSH", "name": "Washington", "venue": "Nationals Park", "color": "#C8102E"},
    {"abbrev": "CIN", "name": "Cincinnati", "venue": "Great American Ball Park", "color": "#C6011F"},
    {"abbrev": "BOS", "name": "Boston", "venue": "Fenway Park", "color": "#BD3039"},
    {"abbrev": "ATH", "name": "Athletics", "venue": "Sutter Health Park", "color": "#003831"},
    {"abbrev": "PIT", "name": "Pittsburgh", "venue": "PNC Park", "color": "#FDB827"},
    {"abbrev": "NYM", "name": "New York", "venue": "Citi Field", "color": "#FF5910"},
    {"abbrev": "PHI", "name": "Philadelphia", "venue": "Citizens Bank Park", "color": "#E81828"},
    {"abbrev": "TOR", "name": "Toronto", "venue": "Rogers Centre", "color": "#134A8E"},
    {"abbrev": "NYY", "name": "New York", "venue": "Yankee Stadium", "color": "#0C2340"},
    {"abbrev": "ATL", "name": "Atlanta", "venue": "Truist Park", "color": "#CE1141"},
    {"abbrev": "TB", "name": "Tampa Bay", "venue": "Tropicana Field", "color": "#092C5C"},
    {"abbrev": "CHC", "name": "Chicago", "venue": "Wrigley Field", "color": "#0E3386"},
    {"abbrev": "CWS", "name": "Chicago", "venue": "Rate Field", "color": "#27251F"},
    {"abbrev": "HOU", "name": "Houston", "venue": "Daikin Park", "color": "#EB6E1F"},
    {"abbrev": "MIA", "name": "Miami", "venue": "loanDepot park", "color": "#00A3E0"},
    {"abbrev": "LAD", "name": "Los Angeles", "venue": "Dodger Stadium", "color": "#005A9C"},
    {"abbrev": "SF", "name": "San Francisco", "venue": "Oracle Park", "color": "#FD5A1E"},
    {"abbrev": "TEX", "name": "Texas", "venue": "Globe Life Field", "color": "#003278"},
    {"abbrev": "STL", "name": "St. Louis", "venue": "Busch Stadium", "color": "#C41E3A"},
    {"abbrev": "SD", "name": "San Diego", "venue": "Petco Park", "color": "#2F241D"},
    {"abbrev": "LAA", "name": "Los Angeles", "venue": "Angel Stadium", "color": "#BA0021"},
    {"abbrev": "SEA", "name": "Seattle", "venue": "T-Mobile Park", "color": "#0C2C56"},
    {"abbrev": "ARI", "name": "Arizona", "venue": "Chase Field", "color": "#A71930"},
    {"abbrev": "COL", "name": "Colorado", "venue": "Coors Field", "color": "#333366"},
    {"abbrev": "KC", "name": "Kansas City", "venue": "Kauffman Stadium", "color": "#004687"},
    {"abbrev": "MIN", "name": "Minnesota", "venue": "Target Field", "color": "#002B5C"},
    {"abbrev": "DET", "name": "Detroit", "venue": "Comerica Park", "color": "#0C2340"},
    {"abbrev": "CLE", "name": "Cleveland", "venue": "Progressive Field", "color": "#E31937"},
    {"abbrev": "MIL", "name": "Milwaukee", "venue": "American Family Field", "color": "#12284B"},
    {"abbrev": "BAL", "name": "Baltimore", "venue": "Oriole Park", "color": "#DF4601"},
]

TEAM_BY_ABBREV = {t["abbrev"]: t for t in MLB_TEAMS}

# (name, era, k9) baselines — a curated pool; rotations are drawn deterministically.
PITCHER_POOL = [
    ("Brad Lord", 3.45, 8.2), ("Brady Singer", 4.12, 7.8), ("Erik Miller", 3.88, 9.1),
    ("J.T. Ginn", 4.65, 7.4), ("Jared Jones", 3.72, 9.5), ("Sean Manaea", 4.08, 8.1),
    ("Jesus Luzardo", 3.24, 9.8), ("Shane Bieber", 3.91, 8.6), ("Cam Schlittler", 4.02, 8.0),
    ("Grant Holmes", 3.85, 8.4), ("Bobby Miller", 3.55, 9.6), ("Logan Webb", 3.28, 8.2),
    ("Hunter Brown", 3.41, 9.9), ("Jacob deGrom", 3.12, 11.2), ("Shota Imanaga", 3.18, 9.3),
    ("Sonny Gray", 3.44, 9.0), ("Tarik Skubal", 2.96, 10.4), ("Seth Lugo", 3.21, 8.0),
    ("Zack Wheeler", 2.87, 10.1), ("Chris Sale", 2.98, 11.0), ("Corbin Burnes", 3.01, 9.6),
    ("Dylan Cease", 3.36, 11.4), ("Framber Valdez", 3.24, 9.1), ("Garrett Crochet", 3.11, 10.8),
    ("Michael King", 3.42, 9.4), ("Yu Darvish", 3.58, 9.2), ("Joe Musgrove", 3.63, 8.9),
    ("Blake Snell", 3.39, 11.1), ("Logan Gilbert", 3.33, 9.7), ("George Kirby", 3.29, 8.8),
    ("Luis Castillo", 3.47, 9.5), ("Pablo Lopez", 3.52, 9.6), ("Bailey Ober", 3.61, 9.0),
    ("Tanner Bibee", 3.43, 9.2), ("Shane McClanahan", 3.26, 10.2), ("Zach Eflin", 3.68, 7.9),
    ("Taj Bradley", 3.87, 10.0), ("Spencer Schwellenbach", 3.55, 8.7), ("Max Fried", 3.19, 9.0),
    ("Shohei Ohtani", 3.05, 10.9), ("Yoshinobu Yamamoto", 3.22, 9.8), ("Tyler Glasnow", 3.08, 11.6),
    ("Gavin Stone", 3.78, 8.3), ("Mitch Keller", 3.92, 8.6), ("Paul Skenes", 2.88, 11.9),
    ("David Peterson", 4.11, 8.4), ("Kodai Senga", 3.35, 10.5), ("Cristopher Sanchez", 3.49, 8.5),
    ("Aaron Nola", 3.30, 9.1), ("Kevin Gausman", 3.71, 9.3), ("Jose Berrios", 3.84, 8.8),
    ("Chris Bassitt", 3.76, 8.5), ("Gerrit Cole", 3.15, 10.6), ("Carlos Rodon", 3.87, 10.1),
    ("Max Scherzer", 3.53, 10.2), ("Luis Gil", 3.44, 9.9), ("Justin Steele", 3.27, 8.9),
    ("Jameson Taillon", 3.95, 7.8), ("Kyle Hendricks", 4.21, 7.2), ("Erick Fedde", 3.99, 8.0),
    ("Jonathan Cannon", 4.33, 7.6), ("Drew Thorpe", 4.08, 8.7), ("Davis Martin", 4.17, 8.3),
    ("Ronel Blanco", 3.72, 9.4), ("Spencer Arrighetti", 4.05, 10.3), ("Hunter Gaddis", 3.69, 8.2),
    ("Eury Perez", 3.40, 10.7), ("Ryan Weathers", 4.12, 8.5), ("Max Meyer", 4.24, 8.9),
    ("Shane Baz", 3.66, 9.4), ("Ryan Pepiot", 3.83, 9.8), ("Drew Rasmussen", 3.58, 9.0),
    ("Jackson Jobe", 3.74, 9.6), ("Reese Olson", 3.90, 8.4), ("Casey Mize", 4.06, 8.1),
    ("Bailey Falter", 4.29, 6.9), ("Johan Oviedo", 4.14, 8.8), ("Colin Rea", 4.19, 7.7),
    ("DL Hall", 4.02, 9.5), ("Freddy Peralta", 3.37, 10.0), ("Tobias Myers", 3.93, 8.6),
    ("Brandon Pfaadt", 3.89, 9.3), ("Zac Gallen", 3.13, 9.2), ("Merrill Kelly", 3.45, 8.6),
    ("Kyle Gibson", 4.23, 7.5), ("Steven Matz", 4.16, 8.0), ("Andre Pallante", 3.98, 7.9),
    ("Erick Miller", 4.10, 8.4), ("Kyle Harrison", 4.04, 9.6), ("Hayden Birdsong", 4.31, 10.1),
    ("Keaton Winn", 4.27, 8.2), ("Tyler Anderson", 4.38, 7.4), ("Jack Kochanowicz", 4.44, 6.8),
    ("Reid Detmers", 4.15, 9.7), ("Jose Soriano", 3.85, 9.1), ("Grayson Rodriguez", 3.46, 9.8),
    ("Kyle Bradish", 3.63, 9.0), ("Dean Kremer", 4.18, 7.9), ("Cade Povich", 4.26, 8.6),
    ("Bryce Miller", 3.51, 9.5), ("Luis Gil Jr", 3.97, 9.3), ("Emerson Hancock", 4.22, 7.8),
    ("Ty Blach", 4.49, 5.9), ("Austin Gomber", 4.37, 7.6), ("Ryan Feltner", 4.51, 8.1),
    ("Kris Bubic", 3.96, 8.9), ("Cole Ragans", 3.32, 10.3), ("Michael Wacha", 3.73, 7.7),
    ("Simeon Woods Richardson", 4.07, 8.5), ("Chris Paddack", 4.20, 8.3), ("Joe Ryan", 3.60, 9.2),
    ("Triston McKenzie", 4.30, 9.0), ("Gavin Williams", 3.94, 9.4), ("Ben Lively", 4.40, 7.2),
    ("Aaron Civale", 4.13, 8.4), ("Zach Eflin II", 3.75, 7.8), ("Osvaldo Bido", 4.21, 8.9),
    ("JP Sears", 4.25, 7.3), ("Mitch Spence", 4.09, 8.1), ("Luis Medina", 4.35, 8.8),
]


def _build_rotations() -> dict:
    """Deterministically assign a 5-man rotation from the pool to every team."""
    n_teams = len(MLB_TEAMS)
    rotations = {}
    for i, team in enumerate(MLB_TEAMS):
        start = (i * 5) % (len(PITCHER_POOL) - 5)
        rotations[team["abbrev"]] = PITCHER_POOL[start : start + 5]
    return rotations


ROTATIONS = _build_rotations()


def team_lookup(abbrev: str) -> dict:
    return TEAM_BY_ABBREV.get(abbrev, {"abbrev": abbrev, "name": abbrev, "venue": "TBD", "color": "#64748B"})


# ===========================================================================
# Point-in-time primitives (pure, unit-testable)
# ===========================================================================

def filter_prior(games: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Return only games whose scheduled start is *strictly before* ``as_of``.

    This is the single enforcement point for point-in-time logic: callers
    must never pass a game at or after ``as_of`` into feature computation.
    """
    ts = pd.to_datetime(games["start_time"], utc=True)
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    else:
        as_of_ts = as_of_ts.tz_convert("UTC")
    return games[ts < as_of_ts].copy()


def rolling_prior_mean(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """Rolling mean of the *previous* ``window`` rows (current row excluded).

    ``shift(1)`` guarantees the value for row ``i`` never sees row ``i``
    itself, which is what makes rolling features point-in-time safe.
    """
    return series.shift(1).rolling(window, min_periods=min_periods).mean()


def expanding_prior_mean(series: pd.Series) -> pd.Series:
    """Cumulative mean of all rows before the current one."""
    return series.shift(1).expanding(min_periods=1).mean()


def compute_elo(
    games: pd.DataFrame,
    k: float = ELO_K,
    home_advantage: float = ELO_HOME_ADVANTAGE,
    mov_multiplier: float = ELO_MOV_MULTIPLIER,
) -> dict:
    """Return ``{(team, game_id): elo_entering_game}``.

    Elo is updated strictly from completed games processed in chronological
    order, so the value returned for a game is always a *pre-game* rating.
    """
    df = games[games["home_runs"].notna() & games["away_runs"].notna()].copy()
    df = df.sort_values("start_time").reset_index(drop=True)

    ratings: dict[str, float] = {}
    elo_before: dict[tuple, float] = {}
    for _, g in df.iterrows():
        home, away = g["home_team"], g["away_team"]
        rh = ratings.get(home, 1500.0)
        ra = ratings.get(away, 1500.0)
        elo_before[(home, g["game_id"])] = rh
        elo_before[(away, g["game_id"])] = ra

        expected_home = 1.0 / (1.0 + 10 ** ((ra - rh - home_advantage) / 400.0))
        hr, ar = float(g["home_runs"]), float(g["away_runs"])
        actual_home = 1.0 if hr > ar else (0.5 if hr == ar else 0.0)
        margin = abs(hr - ar)
        mov = math.log1p(margin) * mov_multiplier
        delta = k * mov * (actual_home - expected_home)
        ratings[home] = rh + delta
        ratings[away] = ra - delta
    return elo_before


def attach_market_lines(game_events: pd.DataFrame, market_lines: pd.DataFrame) -> pd.DataFrame:
    """Attach the most recent market line posted strictly before each game.

    ``market_lines`` must contain ``timestamp`` (posting time, tz-aware UTC)
    plus ``ml_home``, ``ml_away``, ``total_line``, ``run_line`` columns. Any
    line posted at or after the game start is ignored — enforced twice
    (as-of join + explicit filter) for safety.
    """
    games = game_events.sort_values("start_time").reset_index(drop=True)
    lines = market_lines.sort_values("timestamp").reset_index(drop=True)
    # Shift the search key 1s back so a line posted at the exact start time
    # is treated as *not* strictly prior (merge_asof matches <=, we need <).
    games = games.assign(_search=games["start_time"] - pd.Timedelta(seconds=1))
    merged = pd.merge_asof(
        games,
        lines,
        left_on="_search",
        right_on="timestamp",
        direction="backward",
        suffixes=("", "_line"),
    )
    merged = merged.drop(columns=["_search"])
    # Defensive: drop any line whose timestamp is not strictly before start.
    merged = merged[
        (merged["timestamp"].isna()) | (merged["timestamp"] < merged["start_time"])
    ]
    # The attached market columns win over any columns the game already carried.
    line_cols = ["ml_home", "ml_away", "total_line", "run_line"]
    merged = merged.drop(columns=line_cols, errors="ignore")
    merged = merged.rename(columns={f"{c}_line": c for c in line_cols})
    return merged.drop(columns=["timestamp"], errors="ignore")


# ===========================================================================
# Market / odds helpers
# ===========================================================================

def american_to_implied(odds: float) -> float:
    """American odds -> raw implied probability (includes vig)."""
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def remove_vig(p_home: float, p_away: float) -> tuple[float, float]:
    """Normalize two raw implied probabilities to a fair (vig-free) pair."""
    total = p_home + p_away
    if total <= 0:
        return 0.5, 0.5
    return p_home / total, p_away / total


def implied_to_american(p: float) -> int:
    """Fair probability -> American odds (for display)."""
    p = min(max(p, 0.001), 0.999)
    if p >= 0.5:
        return int(round(-100.0 * p / (1.0 - p)))
    return int(round(100.0 * (1.0 - p) / p))


# ===========================================================================
# Game-event construction (synthetic + real)
# ===========================================================================

def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "game_id", "start_time", "home_team", "away_team",
            "home_runs", "away_runs", "home_woba", "away_woba",
            "home_bullpen_whip", "away_bullpen_whip",
            "sp_home", "sp_away", "sp_home_era", "sp_home_k9", "sp_away_era", "sp_away_k9",
            "ml_home", "ml_away", "total_line", "run_line", "venue",
            "day_game", "status", "weather_wind_speed", "final_inning",
        ]
    )


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def generate_synthetic_events(
    start_date: date_cls,
    end_date: date_cls,
    seed: Optional[int] = None,
    target_date: Optional[date_cls] = None,
) -> pd.DataFrame:
    """Generate a deterministic, realistic MLB game log between two dates.

    Latent team strengths drive results; per-game stats (runs, wOBA, bullpen
    WHIP, SP ERA/K9) are sampled around team/pitcher baselines *before* each
    game, so the downstream feature builder sees genuine pre-game values.
    """
    rng = np.random.default_rng(seed if seed is not None else SEED)

    strengths = {t["abbrev"]: rng.normal(0.0, 1.2) for t in MLB_TEAMS}
    offense = {t["abbrev"]: float(rng.uniform(3.7, 5.3)) for t in MLB_TEAMS}
    defense = {t["abbrev"]: float(rng.uniform(3.7, 5.3)) for t in MLB_TEAMS}
    rotation_idx = {t["abbrev"]: int(rng.integers(0, 5)) for t in MLB_TEAMS}

    rows = []
    cursor = start_date
    while cursor <= end_date:
        day_rng = np.random.default_rng(100000 + cursor.toordinal())
        abbrevs = list(TEAM_BY_ABBREV.keys())
        day_rng.shuffle(abbrevs)

        # ~half the league plays each day; leftovers rest.
        n_games = min(len(abbrevs) // 2, 15)
        is_target = target_date is not None and cursor == target_date

        for i in range(n_games):
            home, away = abbrevs[2 * i], abbrevs[2 * i + 1]

            # Start time: 5 day games at 13:05 ET, the rest at 19:05 ET.
            hour_et = 13 if i < 5 else 19
            # EDT is UTC-4, so 13:05 ET == 17:05 UTC and 19:05 ET == 23:05 UTC.
            start = pd.Timestamp(cursor, tz="UTC") + pd.Timedelta(
                hours=hour_et + 4, minutes=5
            )

            # Results from latent strengths (used by elo downstream).
            p_home_raw = _sigmoid((strengths[home] - strengths[away]) * 0.5 + 0.35)
            home_win = bool(day_rng.random() < p_home_raw)
            lh = max(0.0, offense[home] - defense[away] * 0.55 + 0.15 + day_rng.normal(0, 0.9))
            la = max(0.0, offense[away] - defense[home] * 0.55 + day_rng.normal(0, 0.9))
            hr, ar = int(day_rng.poisson(lh)), int(day_rng.poisson(la))
            if home_win and hr <= ar:
                hr = ar + int(day_rng.integers(1, 4))
            elif not home_win and ar <= hr:
                ar = hr + int(day_rng.integers(1, 4))

            h_sp = ROTATIONS[home][rotation_idx[home] % 5]
            a_sp = ROTATIONS[away][rotation_idx[away] % 5]
            rotation_idx[home] = (rotation_idx[home] + 1) % 5
            rotation_idx[away] = (rotation_idx[away] + 1) % 5

            woba_h = float(np.clip(day_rng.normal(0.305 + strengths[home] * 0.012, 0.045), 0.22, 0.42))
            woba_a = float(np.clip(day_rng.normal(0.305 + strengths[away] * 0.012, 0.045), 0.22, 0.42))
            whip_h = float(np.clip(day_rng.normal(1.28, 0.16), 0.9, 1.9))
            whip_a = float(np.clip(day_rng.normal(1.28, 0.16), 0.9, 1.9))

            p_home_mkt = float(np.clip(p_home_raw + day_rng.normal(0, 0.04), 0.08, 0.92))
            juice = float(day_rng.uniform(0.03, 0.05))
            p_away_mkt = (1.0 - p_home_mkt) * (1.0 + juice)
            p_home_mkt = p_home_mkt * (1.0 + juice)
            ml_home = implied_to_american(p_home_mkt)
            ml_away = implied_to_american(p_away_mkt)
            total_line = float(round((lh + la + day_rng.normal(0, 0.6)) * 2) / 2)
            run_line = -1.5

            status, final_inning = "Final", "F"
            if is_target:
                # Fabricate a "today" state: 7 finals + 1 live game.
                status = "Final" if i < 7 else "Live"
                final_inning = "F" if status == "Final" else f"L{int(day_rng.integers(4, 7))}"

            rows.append(
                {
                    "game_id": f"{cursor:%Y%m%d}_{away}_{home}",
                    "start_time": start,
                    "home_team": home, "away_team": away,
                    "home_runs": hr, "away_runs": ar,
                    "home_woba": woba_h, "away_woba": woba_a,
                    "home_bullpen_whip": whip_h, "away_bullpen_whip": whip_a,
                    "sp_home": h_sp[0], "sp_away": a_sp[0],
                    "sp_home_era": h_sp[1], "sp_home_k9": h_sp[2],
                    "sp_away_era": a_sp[1], "sp_away_k9": a_sp[2],
                    "ml_home": ml_home, "ml_away": ml_away,
                    "total_line": total_line, "run_line": run_line,
                    "venue": team_lookup(home)["venue"],
                    "day_game": hour_et == 13,
                    "status": status,
                    "weather_wind_speed": float(np.clip(day_rng.normal(7.0, 3.0), 0, 25)),
                    "final_inning": final_inning,
                }
            )
        cursor += pd.Timedelta(days=1)

    df = pd.DataFrame(rows)
    return df.sort_values("start_time").reset_index(drop=True)


def load_real_game_events(start_date: date_cls, end_date: date_cls) -> pd.DataFrame:
    """Real ingestion path via pybaseball (schedule + Statcast + season-to-date stats).

    * Game metadata (home/away, venue, starting pitchers) is derived from
      Statcast plate appearances for the requested window.
    * Final scores come from ``schedule_and_record`` for each team.
    * Pitcher/batter season-to-date aggregates are queried *as of the day
      before each game date*, so SP ERA/K/9 and team wOBA never leak
      same-day information.

    Requires ``pip install pybaseball`` and network access. On any failure a
    descriptive error is raised; callers should fall back to synthetic mode.
    """
    try:
        from pybaseball import schedule_and_record, statcast  # type: ignore
    except ImportError as exc:  # pragma: no cover - Colab dependency check
        raise RuntimeError(
            "pybaseball is required for real ingestion. Install it with "
            "`pip install pybaseball` or use synthetic mode (synthetic=True)."
        ) from exc

    cache_dir = RAW_DIR / "pybaseball"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"statcast_{start_date:%Y%m%d}_{end_date:%Y%m%d}.parquet"

    if cache_path.exists():
        pa = pd.read_parquet(cache_path)
    else:  # pragma: no cover - requires network
        pa = statcast(start_date.isoformat(), end_date.isoformat())
        pa.to_parquet(cache_path)

    # ---- game metadata: home/away + starting pitcher per game_pk ----
    meta = pa.sort_values(["game_pk", "at_bat_number"])
    meta = meta[["game_pk", "game_date", "home_team", "away_team", "venue_name", "inning", "inning_topbot", "player_name"]]
    first_at_bat = meta.groupby(["game_pk", "inning_topbot"], as_index=False).first()
    sp_map = {}
    for _, r in first_at_bat.iterrows():
        side = "home" if r["inning_topbot"] == "Bottom" else "away"
        sp_map[(r["game_pk"], side)] = r["player_name"]

    meta = meta.groupby("game_pk", as_index=False).first()
    meta["home_sp"] = meta["game_pk"].map(lambda g: sp_map.get((g, "home"), ""))
    meta["away_sp"] = meta["game_pk"].map(lambda g: sp_map.get((g, "away"), ""))

    # ---- final scores from per-team schedules ----
    scores = []
    for team in MLB_TEAMS:  # pragma: no cover - requires network
        sched = schedule_and_record(end_date.year, team["abbrev"])
        sched = sched[sched["Date"].between(start_date.isoformat(), end_date.isoformat())]
        if sched.empty:
            continue
        scores.append(
            pd.DataFrame(
                {
                    "game_date": pd.to_datetime(sched["Date"]),
                    "team": team["abbrev"],
                    "r": pd.to_numeric(sched["R"], errors="coerce"),
                    "ra": pd.to_numeric(sched["RA"], errors="coerce"),
                }
            )
        )
    scores_df = pd.concat(scores, ignore_index=True) if scores else _empty_events()
    scores_df = scores_df.merge(meta[["game_pk", "game_date", "home_team", "away_team"]], on="game_date", how="left")
    scores_df = scores_df.dropna(subset=["home_team"])

    out = _empty_events()
    if scores_df.empty:  # pragma: no cover
        return out

    for _, g in scores_df.iterrows():
        if g["team"] == g["home_team"]:
            home_runs, away_runs = g["r"], g["ra"]
        else:
            away_runs, home_runs = g["r"], g["ra"]
        h_team, a_team = g["home_team"], g["away_team"]
        start = pd.Timestamp(g["game_date"], tz="UTC") + pd.Timedelta(hours=23, minutes=5)
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    [{
                        "game_id": f"{pd.Timestamp(g['game_date']):%Y%m%d}_{a_team}_{h_team}",
                        "start_time": start, "home_team": h_team, "away_team": a_team,
                        "home_runs": home_runs, "away_runs": away_runs,
                        "sp_home": g["home_sp"], "sp_away": g["away_sp"],
                        "venue": team_lookup(h_team)["venue"],
                        "status": "Final", "final_inning": "F",
                    }]
                ),
            ],
            ignore_index=True,
        )
    logger.warning(
        "Real ingestion is a scaffold: SP season-to-date stats and team wOBA "
        "aggregation should be joined here from pitching_stats_range/"
        "batting_stats_range as-of (game_date - 1) before production use."
    )
    return out


def load_game_events(
    start_date: date_cls,
    end_date: date_cls,
    synthetic: bool = True,
    seed: Optional[int] = None,
    target_date: Optional[date_cls] = None,
) -> pd.DataFrame:
    """Load normalized per-game events for a date range."""
    if synthetic:
        return generate_synthetic_events(start_date, end_date, seed=seed, target_date=target_date)
    return load_real_game_events(start_date, end_date)


# ===========================================================================
# Feature construction (strictly point-in-time)
# ===========================================================================

def _team_game_log(game_events: pd.DataFrame) -> pd.DataFrame:
    """Long-form per-(team, game) log used for all rolling aggregates."""
    ev = game_events.copy()

    home = pd.DataFrame(
        {
            "game_id": ev["game_id"], "start_time": ev["start_time"],
            "team": ev["home_team"], "opponent": ev["away_team"],
            "runs_for": ev["home_runs"], "runs_against": ev["away_runs"],
            "woba": ev["home_woba"], "bullpen_whip": ev["home_bullpen_whip"],
            "win": (ev["home_runs"] > ev["away_runs"]).astype(float),
            "is_home": 1.0,
        }
    )
    away = pd.DataFrame(
        {
            "game_id": ev["game_id"], "start_time": ev["start_time"],
            "team": ev["away_team"], "opponent": ev["home_team"],
            "runs_for": ev["away_runs"], "runs_against": ev["home_runs"],
            "woba": ev["away_woba"], "bullpen_whip": ev["away_bullpen_whip"],
            "win": (ev["away_runs"] > ev["home_runs"]).astype(float),
            "is_home": 0.0,
        }
    )
    log = pd.concat([home, away], ignore_index=True)
    log = log.sort_values(["team", "start_time"]).reset_index(drop=True)
    return log


def build_point_in_time_features(
    game_events: pd.DataFrame,
    min_team_games: int = 5,
    include_unready: bool = False,
) -> pd.DataFrame:
    """Compute the full feature table, one row per game, PIT-safe.

    Every rolling column uses only games scheduled strictly before the row's
    own game. Teams with fewer than ``min_team_games`` of history are dropped
    unless ``include_unready=True`` (used for tests/exploration).
    """
    ev = game_events.copy()
    if ev.empty:
        return _empty_events()
    ev["start_time"] = pd.to_datetime(ev["start_time"], utc=True)

    log = _team_game_log(ev)
    g = log.groupby("team", group_keys=False)

    log["runs_for_30g"] = g["runs_for"].transform(lambda s: rolling_prior_mean(s, ROLL_WINDOWS["runs_for"]))
    log["runs_against_30g"] = g["runs_against"].transform(lambda s: rolling_prior_mean(s, ROLL_WINDOWS["runs_against"]))
    log["woba_30g"] = g["woba"].transform(lambda s: rolling_prior_mean(s, ROLL_WINDOWS["woba"]))
    log["bullpen_whip_10g"] = g["bullpen_whip"].transform(lambda s: rolling_prior_mean(s, ROLL_WINDOWS["bullpen_whip"]))
    log["record_pct"] = g["win"].transform(lambda s: expanding_prior_mean(s))
    log["rest_days"] = g["start_time"].diff().dt.days.fillna(1.0).clip(lower=0.0)
    # Rolling SP stats use the team's prior games' SP ERA as a form proxy.
    sp_era = _team_sp_log(ev)
    log = log.merge(
        sp_era[["game_id", "team", "sp_era_prior"]],
        on=["game_id", "team"], how="left",
    )
    log["sp_era_10g"] = (
        log.groupby("team", group_keys=False)["sp_era_prior"]
        .transform(lambda s: rolling_prior_mean(s, ROLL_WINDOWS["sp_era"]))
    )

    elo_before = compute_elo(ev)
    log["elo"] = log.apply(lambda r: elo_before.get((r["team"], r["game_id"]), 1500.0), axis=1)

    # Pivot team-level aggregates back to one row per game.
    feat = ev.copy()
    for suffix, side in (("home", "home"), ("away", "away")):
        sub = log[log["is_home"] == (1.0 if side == "home" else 0.0)].set_index("game_id")
        for col, out_col in [
            ("runs_for_30g", f"{suffix}_runs_for_30g"),
            ("runs_against_30g", f"{suffix}_runs_against_30g"),
            ("woba_30g", f"{suffix}_woba_30g"),
            ("bullpen_whip_10g", f"{suffix}_bullpen_whip_10g"),
            ("record_pct", f"{suffix}_record_pct"),
            ("rest_days", f"{suffix}_rest_days"),
            ("elo", f"{suffix}_elo"),
        ]:
            feat[out_col] = feat["game_id"].map(sub[col])

    feat["elo_diff"] = feat["home_elo"] - feat["away_elo"]
    feat["home_field"] = 1.0

    p_home = feat["ml_home"].map(american_to_implied)
    p_away = feat["ml_away"].map(american_to_implied)
    feat["market_vig"] = (p_home + p_away) - 1.0

    # Drift aliases used by explainability.
    feat["home_team_elo"] = feat["home_elo"]
    feat["away_sp_era_10g"] = (
        log[log["is_home"] == 0.0].set_index("game_id")["sp_era_10g"].reindex(feat["game_id"]).values
    )
    feat["bullpen_whip_10g"] = (feat["home_bullpen_whip_10g"] + feat["away_bullpen_whip_10g"]) / 2.0

    feat["home_win"] = (feat["home_runs"] > feat["away_runs"]).astype(float)
    feat["total_runs"] = feat["home_runs"] + feat["away_runs"]
    feat["home_cover"] = ((feat["home_runs"] - feat["away_runs"]) > 1.5).astype(float)
    feat["home_margin"] = feat["home_runs"] - feat["away_runs"]

    if not include_unready:
        # Drop games for teams without enough history (avoids noisy rows).
        for side in ("home", "away"):
            ready = log.groupby("team")["game_id"].count() >= min_team_games
            ready_teams = set(ready[ready].index)
            feat = feat[feat[f"{side}_team"].isin(ready_teams)]
        feat = feat.reset_index(drop=True)

    return feat


def _team_sp_log(game_events: pd.DataFrame) -> pd.DataFrame:
    """Per (game, team) starting-pitcher season stats, from the event frame."""
    ev = game_events.copy()
    home_sp = ev[["game_id", "home_team", "sp_home_era"]].rename(
        columns={"home_team": "team", "sp_home_era": "sp_era_prior"}
    )
    away_sp = ev[["game_id", "away_team", "sp_away_era"]].rename(
        columns={"away_team": "team", "sp_away_era": "sp_era_prior"}
    )
    return pd.concat([home_sp, away_sp], ignore_index=True)


# ===========================================================================
# Daily artifact builders
# ===========================================================================

def build_todays_games(
    game_events: pd.DataFrame,
    features: pd.DataFrame,
    home_win_probs: pd.Series,
    target_date: date_cls,
    coin_flip_threshold: float = 0.52,
    upset_threshold: float = 0.40,
) -> pd.DataFrame:
    """Assemble the ``todays_games_YYYYMMDD.csv`` table.

    ``home_win_probs`` is the model's predicted P(home wins) per game,
    aligned to ``features``. Records and edges are computed point-in-time
    from the features table.
    """
    ev = game_events.copy()
    ev["start_time"] = pd.to_datetime(ev["start_time"], utc=True)
    f = features.set_index("game_id") if "game_id" in features.columns else features

    rows = []
    for _, g in ev[ev["start_time"].dt.date == target_date].iterrows():
        gid = g["game_id"]
        if gid not in f.index:
            continue
        fr = f.loc[gid]
        p_home = float(home_win_probs.get(gid, 0.5))
        p_away = 1.0 - p_home

        ml_h, ml_a = float(g["ml_home"]), float(g["ml_away"])
        raw_h, raw_a = american_to_implied(ml_h), american_to_implied(ml_a)
        fair_h, _ = remove_vig(raw_h, raw_a)
        juice = raw_h + raw_a - 1.0

        winner = g["home_team"] if g["home_runs"] > g["away_runs"] else (
            g["away_team"] if g["away_runs"] > g["home_runs"] else ""
        )
        pick = g["home_team"] if p_home >= coin_flip_threshold else (
            g["away_team"] if p_away >= coin_flip_threshold else ""
        )
        is_coin_flip = pick == ""
        model_correct = bool(winner and (winner == pick or is_coin_flip))

        rows.append(
            {
                "game_id": gid,
                "game_date": f"{target_date:%Y-%m-%d}",
                "start_time_utc": g["start_time"].isoformat(),
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "home_team_name": team_lookup(g["home_team"])["name"],
                "away_team_name": team_lookup(g["away_team"])["name"],
                "home_record": _record_string(fr.get("home_record_pct"), g["home_team"], ev, g["start_time"]),
                "away_record": _record_string(fr.get("away_record_pct"), g["away_team"], ev, g["start_time"]),
                "home_win_prob_model": round(p_home, 4),
                "away_win_prob_model": round(p_away, 4),
                "moneyline_home": int(ml_h),
                "moneyline_away": int(ml_a),
                "total_line": float(g["total_line"]),
                "run_line": float(g["run_line"]),
                "juice": round(juice, 4),
                "edge_home": round(p_home - fair_h, 4),
                "edge_away": round(p_away - (1.0 - fair_h), 4),
                "starting_pitcher_home": g["sp_home"],
                "starting_pitcher_away": g["sp_away"],
                "sp_home_era": round(float(g["sp_home_era"]), 2),
                "sp_home_k9": round(float(g["sp_home_k9"]), 1),
                "sp_away_era": round(float(g["sp_away_era"]), 2),
                "sp_away_k9": round(float(g["sp_away_k9"]), 1),
                "shap_pointer": f"shap_game_{gid}.csv",
                "game_status": g.get("status", "Scheduled"),
                "home_score": _safe_int(g.get("home_runs")),
                "away_score": _safe_int(g.get("away_runs")),
                "final_inning": g.get("final_inning", ""),
                "venue": g.get("venue", ""),
                "day_game": bool(g.get("day_game", False)),
                "evening_game": not bool(g.get("day_game", False)),
                "model_pick": pick,
                "model_correct": model_correct,
                "is_upset": bool(winner and not is_coin_flip and min(p_home, p_away) < upset_threshold and winner != pick),
                "is_coin_flip": is_coin_flip,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("start_time_utc").reset_index(drop=True)
    return df


def _safe_int(v) -> Optional[int]:
    try:
        f = float(v)
        return None if pd.isna(f) else int(f)
    except (TypeError, ValueError):
        return None


def _record_string(pct, team, ev, start_time) -> str:
    """Win-loss record entering this game (point-in-time, from prior games)."""
    prior = filter_prior(ev, start_time)
    log = _team_game_log(prior)
    team_log = log[log["team"] == team]
    wins = int(team_log["win"].sum())
    played = int(len(team_log))
    return f"{wins}-{played - wins}"


def build_power_rankings(game_events: pd.DataFrame, as_of: date_cls) -> pd.DataFrame:
    """Elo-based power rankings as of a date (one row per team)."""
    ev = game_events.copy()
    ev["start_time"] = pd.to_datetime(ev["start_time"], utc=True)
    prior = filter_prior(ev, pd.Timestamp(as_of, tz="UTC") + pd.Timedelta(days=1))
    log = _team_game_log(prior)
    elo_before = compute_elo(prior)

    out = []
    for team in MLB_TEAMS:
        tl = log[log["team"] == team["abbrev"]].sort_values("start_time")
        if tl.empty:
            continue
        last = tl.iloc[-1]
        elo = elo_before.get((team["abbrev"], last["game_id"]), 1500.0)
        recent = tl.tail(10)
        out.append(
            {
                "rank": 0,  # assigned below
                "team": team["abbrev"],
                "team_name": team["name"],
                "elo": round(float(elo), 1),
                "w": int(tl["win"].sum()),
                "l": int(len(tl) - tl["win"].sum()),
                "pct": round(float(tl["win"].mean()), 3),
                "run_diff": int(tl["runs_for"].sum() - tl["runs_against"].sum()),
                "l10": f"{int(recent['win'].sum())}-{int(len(recent) - recent['win'].sum())}",
                "home_pct": round(float(tl[tl["is_home"] == 1.0]["win"].mean()), 3) if (tl["is_home"] == 1.0).any() else 0.0,
                "away_pct": round(float(tl[tl["is_home"] == 0.0]["win"].mean()), 3) if (tl["is_home"] == 0.0).any() else 0.0,
                "color": team["color"],
            }
        )
    df = pd.DataFrame(out).sort_values("elo", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ===========================================================================
# Convenience
# ===========================================================================

def history_window(target_date: date_cls, seasons_back: int = 2) -> tuple[date_cls, date_cls]:
    """Return (start, end) covering ~2 full seasons before target_date."""
    start = date_cls(target_date.year - seasons_back, 3, 1)
    return start, target_date
