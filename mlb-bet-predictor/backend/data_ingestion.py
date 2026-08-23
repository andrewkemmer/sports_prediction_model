"""
Data ingestion for MLB Bet Predictor.

Provides:
- Synthetic deterministic game logs (zero network, for demo/CI).
- Real pybaseball ingestion scaffold (--real flag).
- Strict point-in-time (PIT) feature construction.
- Elo rating computation.
- Synthetic market-line generation with realistic vig.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    BULLPEN_WHIP_WINDOW,
    ELO_HOME_ADV,
    ELO_K,
    ELO_REVERT_FACTOR,
    RANDOM_SEED,
    SP_ERA_WINDOW,
    SP_K9_WINDOW,
    WOBA_WINDOW,
)

logger = logging.getLogger(__name__)

# ── MLB team metadata ────────────────────────────────────────────────────────
MLB_TEAMS = {
    "NYY": "New York Yankees",
    "BOS": "Boston Red Sox",
    "TB": "Tampa Bay Rays",
    "TOR": "Toronto Blue Jays",
    "BAL": "Baltimore Orioles",
    "CLE": "Cleveland Guardians",
    "DET": "Detroit Tigers",
    "MIN": "Minnesota Twins",
    "CWS": "Chicago White Sox",
    "KC": "Kansas City Royals",
    "HOU": "Houston Astros",
    "SEA": "Seattle Mariners",
    "TEX": "Texas Rangers",
    "LAA": "Los Angeles Angels",
    "OAK": "Oakland Athletics",
    "ATL": "Atlanta Braves",
    "PHI": "Philadelphia Phillies",
    "NYM": "New York Mets",
    "MIA": "Miami Marlins",
    "WSH": "Washington Nationals",
    "MIL": "Milwaukee Brewers",
    "CHC": "Chicago Cubs",
    "STL": "St. Louis Cardinals",
    "PIT": "Pittsburgh Pirates",
    "CIN": "Cincinnati Reds",
    "LAD": "Los Angeles Dodgers",
    "SD": "San Diego Padres",
    "SF": "San Francisco Giants",
    "ARI": "Arizona Diamondbacks",
    "COL": "Colorado Rockies",
}

STADIUMS = {
    "NYY": "Yankee Stadium",
    "BOS": "Fenway Park",
    "TB": "Tropicana Field",
    "TOR": "Rogers Centre",
    "BAL": "Oriole Park at Camden Yards",
    "CLE": "Progressive Field",
    "DET": "Comerica Park",
    "MIN": "Target Field",
    "CWS": "Guaranteed Rate Field",
    "KC": "Kauffman Stadium",
    "HOU": "Minute Maid Park",
    "SEA": "T-Mobile Park",
    "TEX": "Globe Life Field",
    "LAA": "Angel Stadium",
    "OAK": "Sutter Health Park",
    "ATL": "Truist Park",
    "PHI": "Citizens Bank Park",
    "NYM": "Citi Field",
    "MIA": "loanDepot park",
    "WSH": "Nationals Park",
    "MIL": "American Family Field",
    "CHC": "Wrigley Field",
    "STL": "Busch Stadium",
    "PIT": "PNC Park",
    "CIN": "Great American Ball Park",
    "LAD": "Dodger Stadium",
    "SD": "Petco Park",
    "SF": "Oracle Park",
    "ARI": "Chase Field",
    "COL": "Coors Field",
}

# Seed pitchers for synthetic mode (name, ERA, K9)
_SYNTH_PITCHERS = [
    ("G. Cole", 2.85, 11.2),
    ("C. Sale", 3.10, 10.5),
    ("Z. Wheeler", 2.95, 10.8),
    ("S. Strider", 2.70, 12.1),
    ("J. Verlander", 3.20, 9.8),
    ("C. Burnes", 2.60, 10.9),
    ("F. Valdez", 3.30, 9.2),
    ("Y. Darvish", 3.45, 9.5),
    ("P. Lopez", 3.50, 8.9),
    ("K. Bubic", 4.10, 7.8),
    ("M. Gore", 3.80, 8.5),
    ("B. Webb", 3.25, 8.2),
    ("D. Cease", 3.40, 10.1),
    ("L. Castillo", 3.55, 9.0),
    ("C. Bassitt", 3.65, 8.4),
]


# ── PIT enforcement ──────────────────────────────────────────────────────────

def filter_prior(games: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    """Return only games scheduled strictly before `as_of`.

    This is the single enforcement point for point-in-time integrity.
    Adding a future game must never change the output for earlier games.
    """
    if not np.issubdtype(games["start_time_utc"].dtype, np.datetime64):
        games = games.copy()
        games["start_time_utc"] = pd.to_datetime(games["start_time_utc"])
    return games[games["start_time_utc"] < as_of].copy()


# ── Rolling helpers (PIT-safe) ───────────────────────────────────────────────

def rolling_prior_mean(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    """Rolling mean of `value_col` within each group, shifted by 1.

    shift(1) ensures the current game is excluded — only prior games count.
    """
    return (
        df.groupby(group_col)[value_col]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
    )


# ── Season boundaries ──────────────────────────────────────────────────────

def _season_break(prev_date, cur_date) -> bool:
    """True when two chronologically adjacent games cross an offseason.

    MLB seasons never span a calendar-year change with games, so any change
    in game_date year between consecutive games is an offseason boundary
    (e.g. 2025-10-24 → 2026-03-17).
    """
    if prev_date is None or (isinstance(prev_date, float) and pd.isna(prev_date)) or pd.isna(prev_date):
        return False
    return pd.Timestamp(cur_date).year != pd.Timestamp(prev_date).year


def _revert_elo(elo: float) -> float:
    """Regress one Elo rating toward the 1500 mean at a season boundary."""
    return elo + ELO_REVERT_FACTOR * (1500.0 - elo)


def _row_year(row) -> int | None:
    """Calendar year of a game row, from game_date or start_time_utc."""
    for col in ("game_date", "start_time_utc"):
        v = row.get(col)
        if v is not None and not pd.isna(v):
            return pd.Timestamp(v).year
    return None


# ── Elo ──────────────────────────────────────────────────────────────────────

def compute_elos(games: pd.DataFrame) -> pd.Series:
    """Compute Elo ratings entering each game (PIT-safe).

    Returns a Series aligned with the input DataFrame index containing the
    home team's Elo *entering* that game.
    """
    elos: dict[str, float] = {}
    home_elo_entry = pd.Series(np.nan, index=games.index, dtype=float)
    prev_year = None

    for idx, row in games.iterrows():
        home, away = row["home_team"], row["away_team"]
        # Offseason crossed: regress every team toward the mean before the new season
        cur_year = _row_year(row)
        if prev_year is not None and cur_year is not None and cur_year != prev_year:
            elos = {t: _revert_elo(e) for t, e in elos.items()}
        prev_year = cur_year or prev_year
        h_elo = elos.get(home, 1500.0)
        a_elo = elos.get(away, 1500.0)
        home_elo_entry.at[idx] = h_elo

        # Determine outcome (1 = home win, 0 = home loss)
        if pd.notna(row.get("home_win")):
            actual = float(row["home_win"])
        else:
            # No result yet (future game) — skip update
            continue

        # Expected score
        exp_home = 1.0 / (1.0 + 10 ** ((a_elo - h_elo - ELO_HOME_ADV) / 400))
        # Update
        elos[home] = h_elo + ELO_K * (actual - exp_home)
        elos[away] = a_elo + ELO_K * ((1 - actual) - (1 - exp_home))

    return home_elo_entry


def compute_elos_up_to(games: pd.DataFrame, as_of: datetime) -> dict[str, float]:
    """Return Elo ratings for all teams as of `as_of` (PIT-safe)."""
    prior = filter_prior(games, as_of)
    elos: dict[str, float] = {}
    prev_year = None
    for _, row in prior.iterrows():
        home, away = row["home_team"], row["away_team"]
        cur_year = _row_year(row)
        if prev_year is not None and cur_year is not None and cur_year != prev_year:
            elos = {t: _revert_elo(e) for t, e in elos.items()}
        prev_year = cur_year or prev_year
        h_elo = elos.get(home, 1500.0)
        a_elo = elos.get(away, 1500.0)
        if pd.isna(row.get("home_win")):
            continue
        actual = float(row["home_win"])
        exp_home = 1.0 / (1.0 + 10 ** ((a_elo - h_elo - ELO_HOME_ADV) / 400))
        elos[home] = h_elo + ELO_K * (actual - exp_home)
        elos[away] = a_elo + ELO_K * ((1 - actual) - (1 - exp_home))
    return elos


# ── Synthetic game generation ────────────────────────────────────────────────

def generate_synthetic_games(
    target_date: date,
    season_start: date = date(2026, 3, 26),
    games_per_day: int = 15,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a deterministic synthetic MLB season up to and including target_date.

    Returns a DataFrame with columns:
        game_id, game_date, start_time_utc, home_team, away_team,
        home_wins, away_wins, home_losses, away_losses,
        home_win (float 0/1 or NaN), total_runs,
        sp_name_home, sp_era_home, sp_k9_home,
        sp_name_away, sp_era_away, sp_k9_away,
        venue, rest_days_home, rest_days_away,
        woba_30g_home, woba_30g_away,
        bullpen_whip_10g_home, bullpen_whip_10g_away,
        sp_era_30g_home, sp_era_30g_away,
        sp_k9_30g_home, sp_k9_30g_away,
    """
    rng = np.random.RandomState(seed)
    teams = list(MLB_TEAMS.keys())
    n_teams = len(teams)

    # Generate schedule: each team plays ~162 games over ~180 days
    rows = []
    game_counter = 0
    day = season_start
    team_pitcher_idx = {t: 0 for t in teams}
    team_wins = {t: 0 for t in teams}
    team_losses = {t: 0 for t in teams}
    team_rest = {t: 0 for t in teams}  # days since last game
    team_game_dates: dict[str, list[date]] = {t: [] for t in teams}
    team_games_played: dict[str, list[dict]] = {t: [] for t in teams}  # for rolling stats

    while day <= target_date:
        # Pick matchups: pair teams that haven't played each other too much today
        available = [t for t in teams if team_rest[t] >= 1 and len(team_game_dates[t]) < 162]
        rng.shuffle(available)
        n_games_today = min(games_per_day, len(available) // 2)

        for g in range(n_games_today):
            home = available[2 * g]
            away = away_team = available[2 * g + 1]

            sp_idx_home = team_pitcher_idx[home] % len(_SYNTH_PITCHERS)
            sp_idx_away = team_pitcher_idx[away] % len(_SYNTH_PITCHERS)
            sp_home = _SYNTH_PITCHERS[sp_idx_home]
            sp_away = _SYNTH_PITCHERS[sp_idx_away]
            team_pitcher_idx[home] += 1
            team_pitcher_idx[away] += 1

            # Determine winner with slight home advantage
            home_strength = rng.random() + 0.04  # home advantage
            home_win = int(home_strength > 0.5)
            total_runs = int(rng.poisson(8.5) + rng.normal(0, 1.5))
            total_runs = max(1, min(total_runs, 22))

            team_wins[home] += home_win
            team_losses[home] += 1 - home_win
            team_wins[away] += 1 - home_win
            team_losses[away] += home_win

            # Rest days
            rest_h = team_rest[home]
            rest_a = team_rest[away]

            # Compute rolling stats from prior games (PIT-safe)
            prior_h = team_games_played[home]
            prior_a = team_games_played[away]

            def _rolling_mean(data: list, key: str, window: int) -> float:
                vals = [d[key] for d in prior_h if key in d][-window:]
                return float(np.mean(vals)) if vals else 0.0

            woba_h = _rolling_mean(prior_h, "woba", WOBA_WINDOW)
            woba_a = _rolling_mean(prior_a, "woba", WOBA_WINDOW)
            bp_whip_h = _rolling_mean(prior_h, "bullpen_whip", BULLPEN_WHIP_WINDOW)
            bp_whip_a = _rolling_mean(prior_a, "bullpen_whip", BULLPEN_WHIP_WINDOW)
            sp_era_h = _rolling_mean(prior_h, "sp_era", SP_ERA_WINDOW)
            sp_era_a = _rolling_mean(prior_a, "sp_era", SP_ERA_WINDOW)
            sp_k9_h = _rolling_mean(prior_h, "sp_k9", SP_K9_WINDOW)
            sp_k9_a = _rolling_mean(prior_a, "sp_k9", SP_K9_WINDOW)

            hour = rng.choice([13, 13, 13, 19, 19, 19, 19, 19, 20, 21])
            start = datetime.combine(day, datetime.min.time().replace(hour=hour))

            game_id = f"{day.strftime('%Y%m%d')}_{away}@{home}"

            rows.append({
                "game_id": game_id,
                "game_date": day,
                "start_time_utc": start,
                "home_team": home,
                "away_team": away,
                "home_wins": team_wins[home] - home_win,
                "away_wins": team_wins[away] - (1 - home_win),
                "home_losses": team_losses[home] - (1 - home_win),
                "away_losses": team_losses[away] - home_win,
                "home_win": float(home_win),
                "total_runs": total_runs,
                "sp_name_home": sp_home[0],
                "sp_era_home": round(sp_home[1] + rng.normal(0, 0.3), 2),
                "sp_k9_home": round(sp_home[2] + rng.normal(0, 0.5), 1),
                "sp_name_away": sp_away[0],
                "sp_era_away": round(sp_away[1] + rng.normal(0, 0.3), 2),
                "sp_k9_away": round(sp_away[2] + rng.normal(0, 0.5), 1),
                "venue": STADIUMS.get(home, "Unknown"),
                "rest_days_home": rest_h,
                "rest_days_away": rest_a,
                "woba_30g_home": round(woba_h, 3),
                "woba_30g_away": round(woba_a, 3),
                "bullpen_whip_10g_home": round(bp_whip_h, 3),
                "bullpen_whip_10g_away": round(bp_whip_a, 3),
                "sp_era_30g_home": round(sp_era_h, 2),
                "sp_era_30g_away": round(sp_era_a, 2),
                "sp_k9_30g_home": round(sp_k9_h, 1),
                "sp_k9_30g_away": round(sp_k9_a, 1),
            })

            # Record for future rolling stats
            team_games_played[home].append({
                "woba": 0.250 + rng.normal(0, 0.030),
                "bullpen_whip": 1.20 + rng.normal(0, 0.15),
                "sp_era": sp_home[1],
                "sp_k9": sp_home[2],
                "runs_scored": total_runs // 2 + home_win,
                "runs_allowed": total_runs - (total_runs // 2 + home_win),
            })
            team_games_played[away].append({
                "woba": 0.250 + rng.normal(0, 0.030),
                "bullpen_whip": 1.20 + rng.normal(0, 0.15),
                "sp_era": sp_away[1],
                "sp_k9": sp_away[2],
                "runs_scored": total_runs - (total_runs // 2 + home_win),
                "runs_allowed": total_runs // 2 + home_win,
            })

            team_rest[home] = 0
            team_rest[away] = 0
            game_counter += 1

        # Advance rest days for teams that didn't play
        for t in teams:
            if t not in [available[2 * g] if 2 * g < len(available) else None
                         for g in range(n_games_today)]:
                team_rest[t] += 1

        day += timedelta(days=1)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Compute Elo ratings
    df["home_elo"] = compute_elos(df)
    # Fill initial Elo for games with no prior history
    df["home_elo"] = df["home_elo"].fillna(1500.0)

    # Win records
    df["home_record"] = df.apply(
        lambda r: f"{int(r['home_wins'])}-{int(r['home_losses'])}", axis=1
    )
    df["away_record"] = df.apply(
        lambda r: f"{int(r['away_wins'])}-{int(r['away_losses'])}", axis=1
    )
    df["home_win_pct"] = df.apply(
        lambda r: round(r["home_wins"] / max(r["home_wins"] + r["home_losses"], 1), 3),
        axis=1,
    )
    df["away_win_pct"] = df.apply(
        lambda r: round(r["away_wins"] / max(r["away_wins"] + r["away_losses"], 1), 3),
        axis=1,
    )
    # Run differential (approximate from rolling data)
    df["home_run_diff"] = 0  # placeholder; real data would compute this
    df["away_run_diff"] = 0

    return df


def generate_synthetic_market_lines(games: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate realistic synthetic market lines (moneyline, total, run line).

    Lines are attached with a timestamp 2-12 hours before first pitch.
    """
    rng = np.random.RandomState(seed + 1)
    lines = []

    for _, row in games.iterrows():
        if pd.isna(row.get("home_win")):
            continue  # skip future games without results

        # Implied fair probability from "true" win rate + noise
        fair_home = 0.50 + 0.04 + rng.normal(0, 0.08)  # home adv + noise
        fair_home = np.clip(fair_home, 0.20, 0.80)

        # Moneyline with vig
        vig = rng.uniform(0.02, 0.06)
        ml_home = _prob_to_american(fair_home + vig / 2)
        ml_away = _prob_to_american(1 - fair_home + vig / 2)

        # Total line
        total_line = round(rng.normal(8.5, 1.0), 1)
        total_line = max(5.5, min(total_line, 14.5))

        # Run line (-1.5 / +1.5)
        run_line_home = round(rng.uniform(-180, -110))
        run_line_away = round(rng.uniform(-110, 150))

        # Juice (vig)
        juice = round(rng.uniform(0.02, 0.06), 4)

        # Timestamp: line posted 2-12 hours before start
        start = row["start_time_utc"]
        if isinstance(start, pd.Timestamp):
            start = start.to_pydatetime()
        hours_before = rng.uniform(2, 12)
        line_posted = start - timedelta(hours=hours_before)

        lines.append({
            "game_id": row["game_id"],
            "line_posted_at": line_posted,
            "moneyline_home": ml_home,
            "moneyline_away": ml_away,
            "total_line": total_line,
            "run_line_home": run_line_home,
            "run_line_away": run_line_away,
            "juice": juice,
        })

    return pd.DataFrame(lines)


def _prob_to_american(prob: float) -> int:
    """Convert a probability to American odds."""
    if prob >= 0.5:
        return int(round(-100 * prob / (1 - prob)))
    else:
        return int(round(100 * (1 - prob) / prob))


def attach_market_lines(games: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """As-of join: attach lines posted strictly before each game's start.

    Lines posted at or after start_time_utc are rejected (PIT safety).
    """
    if lines.empty:
        return games

    games = games.copy()
    lines = lines.copy()

    if not pd.api.types.is_datetime64_any_dtype(lines["line_posted_at"]):
        lines["line_posted_at"] = pd.to_datetime(lines["line_posted_at"])
    if not pd.api.types.is_datetime64_any_dtype(games["start_time_utc"]):
        games["start_time_utc"] = pd.to_datetime(games["start_time_utc"])
    # Normalize both to tz-naive to avoid comparison errors
    if hasattr(lines["line_posted_at"].dt, "tz") and lines["line_posted_at"].dt.tz is not None:
        lines["line_posted_at"] = lines["line_posted_at"].dt.tz_convert(None)
    if hasattr(games["start_time_utc"].dt, "tz") and games["start_time_utc"].dt.tz is not None:
        games["start_time_utc"] = games["start_time_utc"].dt.tz_convert(None)

    # For each game, pick the latest line posted strictly before start_time_utc
    merged = []
    for _, game in games.iterrows():
        game_lines = lines[
            (lines["game_id"] == game["game_id"])
            & (lines["line_posted_at"] < game["start_time_utc"])
        ]
        if game_lines.empty:
            # No valid line — fill with defaults
            merged.append({**game.to_dict(), "moneyline_home": None, "moneyline_away": None,
                           "total_line": None, "run_line_home": None, "run_line_away": None,
                           "juice": 0.04})
        else:
            latest = game_lines.sort_values("line_posted_at").iloc[-1]
            merged.append({**game.to_dict(), **latest.drop(["game_id", "line_posted_at"]).to_dict()})

    return pd.DataFrame(merged)


# ── Real data (ESPN API) ───────────────────────────────────────────────────

# ESPN abbreviation → our internal team key
ESPN_ABBREV_TO_KEY = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CWS", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KC": "KC",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SD": "SD", "SF": "SF",
    "SEA": "SEA", "STL": "STL", "TB": "TB", "TEX": "TEX",
    "TOR": "TOR", "WSH": "WSH",
}


def _espn_date_key(d: date) -> str:
    return d.strftime("%Y%m%d")


def _fetch_espn_scoreboard(target_date: date) -> list[dict]:
    """Fetch all MLB games for a given date from ESPN's public API."""
    import requests

    url = (
        "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
        f"?dates={_espn_date_key(target_date)}"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("events", [])


def _parse_espn_event(event: dict) -> dict | None:
    """Parse a single ESPN event into our internal game dict."""
    competition = event.get("competitions", [{}])[0]
    competitors = competition.get("competitors", [])
    if len(competitors) != 2:
        return None

    home_comp = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away_comp = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home_comp or not away_comp:
        return None

    home_abbr = home_comp.get("team", {}).get("abbreviation", "")
    away_abbr = away_comp.get("team", {}).get("abbreviation", "")
    home_key = ESPN_ABBREV_TO_KEY.get(home_abbr, home_abbr)
    away_key = ESPN_ABBREV_TO_KEY.get(away_abbr, away_abbr)

    # Parse date
    date_str = event.get("date", "")
    try:
        game_dt = pd.Timestamp(date_str)
    except Exception:
        return None
    game_date = game_dt.date()

    # Game state from ESPN — only "post" means the game is truly final.
    game_state = (competition.get("status", {}).get("type", {}).get("state") or "")
    game_status_detail = (competition.get("status", {}).get("type", {}).get("detail") or "")

    # Scores (may be None if game hasn't started/finished)
    home_score_str = home_comp.get("score", "")
    away_score_str = away_comp.get("score", "")
    home_score = int(home_score_str) if home_score_str.isdigit() else None
    away_score = int(away_score_str) if away_score_str.isdigit() else None

    # Only set home_win when the game is FINAL — live scores must never
    # become training labels or appear as "Final" on the dashboard.
    home_win = None
    if game_state == "post" and home_score is not None and away_score is not None:
        home_win = float(home_score > away_score)

    # Starting pitchers from ESPN projectedStats or probablePitcher
    sp_home = "TBD"
    sp_away = "TBD"
    for comp in competitors:
        pitcher = comp.get("probablePitcher", {})
        if pitcher and pitcher.get("fullName"):
            if comp.get("homeAway") == "home":
                sp_home = pitcher["fullName"]
            else:
                sp_away = pitcher["fullName"]

    venue = competition.get("venue", {}).get("fullName", "Unknown")
    game_id = f"{game_date.strftime('%Y%m%d')}_{away_key}@{home_key}"

    return {
        "game_id": game_id,
        "game_date": game_date,
        "start_time_utc": game_dt,
        "home_team": home_key,
        "away_team": away_key,
        "game_state": game_state,
        "game_status_detail": game_status_detail,
        "home_win": home_win,
        # The actual runs — without these the dashboard shows winners with
        # no final score (home_win alone was carried, scores were dropped).
        "home_score": home_score,
        "away_score": away_score,
        "total_runs": (home_score + away_score) if home_score is not None and away_score is not None else None,
        "sp_name_home": sp_home,
        "sp_name_away": sp_away,
        "sp_era_home": None,
        "sp_k9_home": None,
        "sp_era_away": None,
        "sp_k9_away": None,
        "venue": venue,
        "rest_days_home": None,
        "rest_days_away": None,
        "woba_30g_home": None,
        "woba_30g_away": None,
        "bullpen_whip_10g_home": None,
        "bullpen_whip_10g_away": None,
        "sp_era_30g_home": None,
        "sp_era_30g_away": None,
        "sp_k9_30g_home": None,
        "sp_k9_30g_away": None,
    }


def load_real_game_events(target_date: date, season: int | None = None) -> pd.DataFrame:
    """Load real MLB game events for a single date via ESPN API.

    Fetches all games for `target_date` in one HTTP call. If no games are
    found (off-season), walks backwards up to 7 days to find the most recent
    game day.
    """
    logger.info("Loading real MLB schedule for %s via ESPN API...", target_date)

    # Try the target date, then walk backwards up to 7 days to find games
    for offset in range(8):
        try_date = date.fromordinal(target_date.toordinal() - offset)
        events = _fetch_espn_scoreboard(try_date)
        games = []
        for ev in events:
            parsed = _parse_espn_event(ev)
            if parsed:
                games.append(parsed)
        if games:
            if offset > 0:
                logger.info("No games on %s, using %s instead (%d games)",
                            target_date, try_date, len(games))
            break

    if not games:
        logger.warning("No MLB games found in the last 7 days around %s", target_date)
        return pd.DataFrame()

    df = pd.DataFrame(games)
    for _tc in ("home_team", "away_team"):
        if _tc in df.columns:
            df[_tc] = df[_tc].map(normalize_team)
    df["home_wins"] = 0
    df["away_wins"] = 0
    df["home_losses"] = 0
    df["away_losses"] = 0
    df["home_elo"] = 1500.0
    df["home_record"] = "0-0"
    df["away_record"] = "0-0"
    df["home_win_pct"] = 0.5
    df["away_win_pct"] = 0.5
    df["home_run_diff"] = 0
    df["away_run_diff"] = 0
    logger.info("Loaded %d real games from ESPN for %s", len(df), try_date)
    return df


def load_game_events(target_date: date, real: bool = False, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Unified entry point: load synthetic or real game events."""
    if real:
        return load_real_game_events(target_date)
    return generate_synthetic_games(target_date, seed=seed)


# Canonical MLBAM team codes as returned by Statcast (verified against live
# pulls): AZ (not ARI), ATH (not OAK), CWS. Schedule feeds wobble between
# spellings across seasons/vendors — every team code entering the pipeline is
# canonicalized here so joins never split a franchise into two identities.
TEAM_ALIASES = {
    "CHW": "CWS", "CHA": "CWS",
    "OAK": "ATH",
    "ARI": "AZ",
}


def normalize_team(code) -> str:
    """Map a vendor-specific team abbreviation to its canonical Statcast code."""
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return code
    return TEAM_ALIASES.get(str(code).strip().upper(), str(code).strip())


def load_game_features(path: str | Path) -> pd.DataFrame:
    """Load game-level features from DuckDB output and map columns for training.

    The features.py module produces a game_level_features.csv with columns like:
        game_pk, game_date, home_team, away_team, home_win, total_runs,
        sp_era_home, team_woba_30g_home, bullpen_whip_10g_home, etc.

    This function maps those to the column names expected by training.py:
        game_id, home_elo, home_win_pct, away_win_pct,
        woba_30g_home, sp_era_30g_home, sp_k9_30g_home, etc.

    It also computes ELO, win percentage, and run differential from the data.
    """
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Features file not found: {p}")

    if p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_parquet(p)

    # Canonicalize team codes BEFORE anything derives ids/records from them
    for _tc in ("home_team", "away_team"):
        if _tc in df.columns:
            df[_tc] = df[_tc].map(normalize_team)

    logger.info("Loaded %d games from %s (columns: %s)", len(df), p.name, list(df.columns))

    # Ensure game_date is datetime
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    # Add game_id if missing
    if "game_id" not in df.columns:
        df["game_id"] = df.apply(
            lambda r: f"{pd.Timestamp(r['game_date']).strftime('%Y%m%d')}_{r.get('away_team','?')}@{r['home_team']}",
            axis=1,
        )

    # Add start_time_utc if missing (use game_date at 19:00 UTC)
    if "start_time_utc" not in df.columns:
        df["start_time_utc"] = df["game_date"].apply(
            lambda d: datetime.combine(d.date(), datetime.min.time().replace(hour=19))
            if pd.notna(d) else None
        )

    # Compute ELO from game results (PIT-safe: chronological)
    df["home_elo"] = compute_elos(df)
    df["home_elo"] = df["home_elo"].fillna(1500.0)

    # Compute win percentages and records from cumulative results
    team_records: dict[str, dict[str, int]] = {}
    prev_year = None
    home_wins_list = []
    home_losses_list = []
    away_wins_list = []
    away_losses_list = []
    home_win_pcts = []
    away_win_pcts = []
    home_run_diffs = []
    away_run_diffs = []

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        # Offseason crossed: cumulative W/L and run diff reset each season
        cur_year = _row_year(row)
        if prev_year is not None and cur_year is not None and cur_year != prev_year:
            team_records = {}
        prev_year = cur_year or prev_year
        if home not in team_records:
            team_records[home] = {"w": 0, "l": 0, "rs": 0, "ra": 0}
        if away not in team_records:
            team_records[away] = {"w": 0, "l": 0, "rs": 0, "ra": 0}

        hw = team_records[home]
        aw = team_records[away]

        # Before this game
        home_wins_list.append(hw["w"])
        home_losses_list.append(hw["l"])
        away_wins_list.append(aw["w"])
        away_losses_list.append(aw["l"])
        total_h = hw["w"] + hw["l"]
        total_a = aw["w"] + aw["l"]
        # Season opener (no games yet): rate is undefined → NULL, never 0
        home_win_pcts.append(round(hw["w"] / total_h, 3) if total_h else np.nan)
        away_win_pcts.append(round(aw["w"] / total_a, 3) if total_a else np.nan)
        home_run_diffs.append(hw["rs"] - hw["ra"])
        away_run_diffs.append(aw["rs"] - aw["ra"])

        # Update after this game
        if pd.notna(row.get("home_win")):
            home_won = int(row["home_win"])
            home_runs = int(row.get("home_score", row.get("total_runs", 8) // 2))
            away_runs = int(row.get("away_score", row.get("total_runs", 8) - home_runs))

            hw["w"] += home_won
            hw["l"] += 1 - home_won
            aw["w"] += 1 - home_won
            aw["l"] += home_won
            hw["rs"] += home_runs
            hw["ra"] += away_runs
            aw["rs"] += away_runs
            aw["ra"] += home_runs

    df["home_wins"] = home_wins_list
    df["home_losses"] = home_losses_list
    df["away_wins"] = away_wins_list
    df["away_losses"] = away_losses_list
    df["home_win_pct"] = home_win_pcts
    df["away_win_pct"] = away_win_pcts
    df["home_run_diff"] = home_run_diffs
    df["away_run_diff"] = away_run_diffs

    df["home_record"] = df.apply(
        lambda r: f"{int(r['home_wins'])}-{int(r['home_losses'])}", axis=1
    )
    df["away_record"] = df.apply(
        lambda r: f"{int(r['away_wins'])}-{int(r['away_losses'])}", axis=1
    )

    # Map column names to match FEATURE_COLS in training.py
    col_map = {
        "team_woba_30g_home": "woba_30g_home",
        "team_woba_30g_away": "woba_30g_away",
        "sp_era_home": "sp_era_30g_home",
        "sp_era_away": "sp_era_30g_away",
        "sp_k9_home": "sp_k9_30g_home",
        "sp_k9_away": "sp_k9_30g_away",
    }
    # Add mapped columns (keep originals too for SHAP labels)
    for src, dst in col_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    logger.info("Feature mapping complete: %d games, columns: %s", len(df), list(df.columns))
    return df


# ── Upcoming-slate construction (pre-game predictions for TODAY) ───────────

def _norm_player_name(name) -> str:
    """Normalize a player name for matching across vendors.

    Statcast stores 'Wheeler, Zack'; ESPN ships 'Zack Wheeler'. Both
    normalize to 'zack wheeler' so probable pitchers can be joined to
    their rolling stat lines without a paid ID crosswalk.
    """
    s = str(name or "").strip().lower()
    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"
    return " ".join(s.split())


def load_espn_schedule(target_date: date) -> pd.DataFrame:
    """Games scheduled for EXACTLY target_date via the ESPN API.

    Unlike load_real_game_events() there is no 7-day walk-back: an empty
    off-day returns an empty frame instead of yesterday's slate, which is
    exactly what pre-game prediction needs.
    """
    rows = []
    try:
        events = _fetch_espn_scoreboard(target_date)
    except Exception as e:
        logger.warning("ESPN schedule fetch failed for %s: %s", target_date, e)
        return pd.DataFrame()
    for ev in events:
        parsed = _parse_espn_event(ev)
        if parsed:
            rows.append(parsed)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for _tc in ("home_team", "away_team"):
        df[_tc] = df[_tc].map(normalize_team)
    return df


def _final_team_records(hist: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Per-team W/L/RS/RA AFTER the last decided game (chronological walk).

    Mirrors the update rules used by load_game_features so carried records
    match what the model saw entering historical games (including the
    per-season reset — only the current season's record carries forward).
    """
    rec: dict[str, dict[str, int]] = {}
    prev_year = None
    for _, row in hist.sort_values("game_date").iterrows():
        hw = row.get("home_win")
        if pd.isna(hw):
            continue  # undecided: no record/run impact
        h, a = row["home_team"], row["away_team"]
        # Offseason crossed: start fresh for the new season
        cur_year = _row_year(row)
        if prev_year is not None and cur_year is not None and cur_year != prev_year:
            rec = {}
        prev_year = cur_year or prev_year
        rh = rec.setdefault(h, {"w": 0, "l": 0, "rs": 0, "ra": 0})
        ra = rec.setdefault(a, {"w": 0, "l": 0, "rs": 0, "ra": 0})
        hs = row.get("home_score")
        asc = row.get("away_score")
        hs_i = int(hs) if pd.notna(hs) else 0
        asc_i = int(asc) if pd.notna(asc) else 0
        won = int(hw)
        rh["w"] += won; rh["l"] += 1 - won; rh["rs"] += hs_i; rh["ra"] += asc_i
        ra["w"] += 1 - won; ra["l"] += won; ra["rs"] += asc_i; ra["ra"] += hs_i
    return rec


def _latest_side_state(hist: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, float]]:
    """team → {feature_base: most recent non-null value across BOTH sides}.

    A team's home-start and road-start rows feed different suffixed columns
    (woba_30g_home / woba_30g_away) but describe the same rolling stat, so
    the carry-forward resolves per feature base in chronological order — a
    team's latest road value must override its older home value and vice
    versa. Returns keys WITHOUT the side suffix; callers re-suffix by slot.
    """
    state: dict[str, dict[str, float]] = {}
    ordered = hist.sort_values("game_date")
    for col in cols:
        if col not in hist.columns:
            continue
        if col.endswith("_home"):
            base, side = col[: -len("_home")], "home"
        elif col.endswith("_away"):
            base, side = col[: -len("_away")], "away"
        else:
            continue
        sub = ordered.dropna(subset=[col])
        if sub.empty:
            continue
        last_per_team = sub.groupby(f"{side}_team", sort=False).tail(1)
        for _, r in last_per_team.iterrows():
            state.setdefault(r[f"{side}_team"], {})[base] = float(r[col])
    return state


def _own_lefty_share(hist: pd.DataFrame) -> dict[str, float]:
    """team → its OWN lineup lefty share (trailing 30g).

    opp_lefty_share_home actually holds the AWAY lineup's share and vice
    versa (each measures the lineup the opposing starter faces), so the two
    stored sides must be mirrored back onto the batting team before they can
    be re-paired with a new matchup.
    """
    pieces = []
    if "opp_lefty_share_away" in hist.columns:
        pieces.append(hist.dropna(subset=["opp_lefty_share_away"])
                      .groupby("home_team")["opp_lefty_share_away"].last())
    if "opp_lefty_share_home" in hist.columns:
        pieces.append(hist.dropna(subset=["opp_lefty_share_home"])
                      .groupby("away_team")["opp_lefty_share_home"].last())
    if not pieces:
        return {}
    combined = pd.concat(pieces)
    return {t: float(v) for t, v in combined.groupby(level=0).last().items()}


def _latest_pitcher_state(hist: pd.DataFrame) -> dict[Any, dict[str, float]]:
    """pitcher_id → {sp_* feature base: latest non-null value across starts}.

    A starter's road starts land in *_away columns and home starts in *_home
    columns, but the stat describes the pitcher — resolve per feature base in
    date order and re-suffix by the slot he occupies in the new game.
    """
    state: dict[Any, dict[str, float]] = {}
    ordered = hist.sort_values("game_date")
    for side in ("home", "away"):
        id_col = f"{side}_starter_id"
        if id_col not in hist.columns:
            continue
        sp_cols = [c for c in hist.columns
                   if c.startswith("sp_") and c.endswith(f"_{side}")]
        if not sp_cols:
            continue
        sub = ordered[ordered[id_col].notna()]
        for col in sp_cols:
            base = col[: -len(f"_{side}")]
            s = sub.dropna(subset=[col])
            if s.empty:
                continue
            last_per_pid = s.groupby(id_col, sort=False).tail(1)
            for _, r in last_per_pid.iterrows():
                state.setdefault(r[id_col], {})[base] = float(r[col])
    return state


def build_upcoming_slate(
    history_df: pd.DataFrame,
    target_date: date,
    pbp_df: Optional[pd.DataFrame] = None,
    schedule_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build pre-game feature rows for target_date's scheduled games.

    Statcast-derived history only contains games that have been played, so
    on a normal morning run there are zero rows for today. This carries each
    team's and each probable starter's most recent point-in-time feature
    state forward onto today's real schedule — strictly prior data only,
    same semantics the training frame used entering every historical game.

    Args:
        history_df:   Feature-enriched game history (from load_game_features).
        target_date:  Slate date.
        pbp_df:       Optional pitch-level frame; player_name + pitcher IDs
                      are used to map ESPN probable-pitcher NAMES to their
                      rolling stat lines. Without it, pitcher features ship NaN.
        schedule_df:  Pre-fetched schedule (tests). Default fetches ESPN.

    Returns:
        DataFrame with one row per scheduled game. Undecided games carry
        home_win/home_score/away_score = NULL; games ESPN already scores
        (earlier finals) keep their results. Every FEATURE_COLS column exists
        so predict_games sees the exact trained feature layout.
    """
    from training import FEATURE_COLS  # lazy: avoids import cycles at module load

    sched = schedule_df if schedule_df is not None else load_espn_schedule(target_date)
    if sched.empty:
        return pd.DataFrame()

    hist = history_df.copy()
    hist["game_date"] = pd.to_datetime(hist["game_date"])
    if "start_time_utc" not in hist.columns:
        hist["start_time_utc"] = pd.to_datetime(hist["game_date"])

    # Point-in-time states as of the start of target_date
    as_of = datetime.combine(target_date, datetime.min.time())
    elos = compute_elos_up_to(hist, as_of=as_of)
    records = _final_team_records(hist)

    # Carry forward the raw input columns that add_diff_features() needs
    # to compute the model's diff FEATURE_COLS.  (FEATURE_COLS itself now
    # uses diff names, so we derive the raw list from add_diff_features'
    # expected inputs.)
    _RAW_CARRY = [
        "woba_30g_home", "woba_30g_away",
        "rest_days_home", "rest_days_away",
        "team_barrel_15g_home", "team_barrel_15g_away",
        "team_hardhit_15g_home", "team_hardhit_15g_away",
        "team_exitvelo_15g_home", "team_exitvelo_15g_away",
        "lineup_woba_mean_home", "lineup_woba_mean_away",
        "lineup_woba_top3_home", "lineup_woba_top3_away",
        "lineup_woba_std_home", "lineup_woba_std_away",
        "bullpen_whip_10g_home", "bullpen_whip_10g_away",
        "bullpen_pitches_3d_home", "bullpen_pitches_3d_away",
        "bullpen_ip_3d_home", "bullpen_ip_3d_away",
    ]
    carry_cols = [c for c in _RAW_CARRY if c in hist.columns]
    team_state = _latest_side_state(hist, carry_cols)
    own_lefty = _own_lefty_share(hist)
    pitcher_state = _latest_pitcher_state(hist)

    # ESPN name → Statcast pitcher id (latest mapping wins)
    name_to_id: dict[str, Any] = {}
    if pbp_df is not None and {"player_name", "pitcher"}.issubset(pbp_df.columns):
        m = pbp_df[["player_name", "pitcher"]].dropna()
        m = m.assign(_norm=m["player_name"].map(_norm_player_name))
        m = m[m["_norm"] != ""].drop_duplicates(subset=["_norm"], keep="last")
        name_to_id = dict(zip(m["_norm"], m["pitcher"]))

    last_played_h = hist.groupby("home_team")["game_date"].max()
    last_played_a = hist.groupby("away_team")["game_date"].max()
    last_played = pd.concat([last_played_h, last_played_a]).groupby(level=0).max()

    def _rest_days(team: str):
        lp = last_played.get(team)
        if pd.isna(lp):
            return np.nan
        return max((pd.Timestamp(target_date) - lp).days, 1)

    rows = []
    for _, s in sched.iterrows():
        home, away = s["home_team"], s["away_team"]
        row = {c: np.nan for c in FEATURE_COLS}
        # Initialize raw input columns that add_diff_features() needs.
        # FEATURE_COLS now has diff names only, but the raw home/away
        # columns must exist for add_diff_features() to compute them.
        _RAW_INPUTS = [
            "home_elo", "away_elo", "home_win_pct", "away_win_pct",
            "rest_days_home", "rest_days_away",
            "sp_era_home", "sp_era_away",
            "sp_k9_home", "sp_k9_away",
            "sp_fbvelo_3g_home", "sp_fbvelo_3g_away",
            "sp_fbpct_3g_home", "sp_fbpct_3g_away",
            "sp_whiff_3g_home", "sp_whiff_3g_away",
            "sp_xwoba_home", "sp_xwoba_away",
            "sp_xwoba_vs_l_home", "sp_xwoba_vs_l_away",
            "sp_era_30g_home", "sp_era_30g_away",
            "sp_k9_30g_home", "sp_k9_30g_away",
            "woba_30g_home", "woba_30g_away",
            "lineup_woba_mean_home", "lineup_woba_mean_away",
            "lineup_woba_top3_home", "lineup_woba_top3_away",
            "lineup_woba_std_home", "lineup_woba_std_away",
            "bullpen_whip_10g_home", "bullpen_whip_10g_away",
            "bullpen_pitches_3d_home", "bullpen_pitches_3d_away",
            "bullpen_ip_3d_home", "bullpen_ip_3d_away",
            "team_barrel_15g_home", "team_barrel_15g_away",
            "team_hardhit_15g_home", "team_hardhit_15g_away",
            "team_exitvelo_15g_home", "team_exitvelo_15g_away",
            "opp_lefty_share_home", "opp_lefty_share_away",
        ]
        for _c in _RAW_INPUTS:
            row.setdefault(_c, np.nan)
        row.update({
            "game_id": s.get("game_id") or (
                f"{pd.Timestamp(target_date).strftime('%Y%m%d')}_{away}@{home}"),
            "game_date": pd.Timestamp(target_date),
            "start_time_utc": s.get("start_time_utc"),
            "home_team": home,
            "away_team": away,
            "venue": s.get("venue", "Unknown"),
            "total_runs": np.nan,
            # Results: keep them when ESPN already has a final, else NULL
            "home_win": s.get("home_win"),
            "home_score": s.get("home_score"),
            "away_score": s.get("away_score"),
            "sp_name_home": s.get("sp_name_home", "TBD"),
            "sp_name_away": s.get("sp_name_away", "TBD"),
        })

        # Team-level state (records, elo, run diff, rolling offense/bullpen)
        for team, side in ((home, "home"), (away, "away")):
            r = records.get(team, {"w": 0, "l": 0, "rs": 0, "ra": 0})
            total = r["w"] + r["l"]
            row[f"{side}_wins"] = r["w"]
            row[f"{side}_losses"] = r["l"]
            row[f"{side}_record"] = f"{r['w']}-{r['l']}"
            row[f"{side}_win_pct"] = round(r["w"] / max(total, 1), 3)
            row[f"{side}_run_diff"] = r["rs"] - r["ra"]
            row[f"rest_days_{side}"] = _rest_days(team)
            # Re-suffix the side-agnostic latest values onto this game's slot
            for base, val in team_state.get(team, {}).items():
                row[f"{base}_{side}"] = val
            # Mirror: each lineup's share pairs with the OPPOSING starter
            share = own_lefty.get(team)
            if share is not None:
                row[f"opp_lefty_share_{'away' if side == 'home' else 'home'}"] = share
        row["home_elo"] = elos.get(home, 1500.0)
        row["away_elo"] = elos.get(away, 1500.0)

        # Pitcher-level state via name → id → latest stat line, re-suffixed
        # to the slot he occupies tonight (his last start may have been on
        # the other side of a box score).
        for side in ("home", "away"):
            pid = name_to_id.get(_norm_player_name(row[f"sp_name_{side}"]))
            if pid is None:
                continue
            for base, val in pitcher_state.get(pid, {}).items():
                row[f"{base}_{side}"] = val

        rows.append(row)

    slate = pd.DataFrame(rows)
    slate = slate.sort_values("start_time_utc").reset_index(drop=True)
    logger.info("Upcoming slate built: %d games for %s", len(slate), target_date)
    return slate
