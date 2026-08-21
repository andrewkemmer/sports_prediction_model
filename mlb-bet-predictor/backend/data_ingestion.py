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


# ── Elo ──────────────────────────────────────────────────────────────────────

def compute_elos(games: pd.DataFrame) -> pd.Series:
    """Compute Elo ratings entering each game (PIT-safe).

    Returns a Series aligned with the input DataFrame index containing the
    home team's Elo *entering* that game.
    """
    elos: dict[str, float] = {}
    home_elo_entry = pd.Series(np.nan, index=games.index, dtype=float)

    for idx, row in games.iterrows():
        home, away = row["home_team"], row["away_team"]
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
    for _, row in prior.iterrows():
        home, away = row["home_team"], row["away_team"]
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

    if not np.issubdtype(lines["line_posted_at"].dtype, np.datetime64):
        lines["line_posted_at"] = pd.to_datetime(lines["line_posted_at"])
    if not np.issubdtype(games["start_time_utc"].dtype, np.datetime64):
        games["start_time_utc"] = pd.to_datetime(games["start_time_utc"])

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


# ── Real data (pybaseball scaffold) ─────────────────────────────────────────

def load_real_game_events(target_date: date, season: int = 2026) -> pd.DataFrame:
    """Load real MLB game events via pybaseball.

    Iterates over all MLB teams to fetch each team's schedule, deduplicates
    games (keeping only home games to avoid double-counting), and normalizes
    to our internal schema.

    Note: SP ERA/K9 and team wOBA should be joined from
    pitching_stats_range / batting_stats_range as-of game_date - 1
    for full PIT compliance (TODO).
    """
    logger.info("Loading real MLB schedule for season %d...", season)

    try:
        from pybaseball import schedule_and_record
    except ImportError:
        raise ImportError(
            "pybaseball is required for real data. "
            "Install with: pip install pybaseball"
        )

    # Try the requested season, then fall back to recent seasons if unavailable
    working_season = None
    all_games = []
    for attempt_season in [season, season - 1, season - 2]:
        if attempt_season < 2015:
            break
        logger.info("Trying season %d...", attempt_season)
        all_games = []
        for team in MLB_TEAMS:
            try:
                team_schedule = schedule_and_record(attempt_season, team)
                if team_schedule is not None and not team_schedule.empty:
                    if "Home" in team_schedule.columns:
                        home_games = team_schedule[team_schedule["Home"] == team].copy()
                    else:
                        home_games = team_schedule.copy()
                    all_games.append(home_games)
            except Exception as e:
                logger.debug("Failed to fetch schedule for %s/%d: %s", team, attempt_season, e)
                continue
        if len(all_games) >= 10:  # Got a decent chunk of teams
            working_season = attempt_season
            break
        logger.warning("Season %d returned only %d teams, trying next year...", attempt_season, len(all_games))

    if not all_games:
        logger.warning("No schedule data returned for seasons %d-%d", season - 2, season)
        return pd.DataFrame()

    if working_season and working_season != season:
        logger.info("Using season %d data (requested %d not yet available)", working_season, season)

    schedule = pd.concat(all_games, ignore_index=True)

    # Filter to games up to target_date
    schedule["Date"] = pd.to_datetime(schedule["Date"])
    schedule = schedule[schedule["Date"].dt.date <= target_date].copy()

    # Deduplicate by date + home + away
    schedule = schedule.drop_duplicates(subset=["Date", "Home", "Away"]).copy()

    # Basic normalization to our schema
    rows = []
    for _, game in schedule.iterrows():
        home = game.get("Home", "")
        away = game.get("Away", "")
        game_date = game["Date"].date()
        home_score = game.get("R", game.get("Home Starter Runs", None))
        away_score = game.get("RA", game.get("Away Starter Runs", None))
        home_win = None
        if pd.notna(home_score) and pd.notna(away_score):
            home_win = float(int(home_score) > int(away_score))

        game_id = f"{game_date.strftime('%Y%m%d')}_{away}@{home}"

        rows.append({
            "game_id": game_id,
            "game_date": game_date,
            "start_time_utc": game["Date"],
            "home_team": home,
            "away_team": away,
            "home_win": home_win,
            "total_runs": (int(home_score) + int(away_score)) if pd.notna(home_score) and pd.notna(away_score) else None,
            "sp_name_home": game.get("Home Starter", "TBD"),
            "sp_name_away": game.get("Away Starter", "TBD"),
            "sp_era_home": None,
            "sp_k9_home": None,
            "sp_era_away": None,
            "sp_k9_away": None,
            "venue": game.get("Stadium", game.get("field", "Unknown")),
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
        })

    df = pd.DataFrame(rows)
    if not df.empty:
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
        logger.info("Loaded %d real games from pybaseball", len(df))

    return df


def load_game_events(target_date: date, real: bool = False, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Unified entry point: load synthetic or real game events."""
    if real:
        return load_real_game_events(target_date)
    return generate_synthetic_games(target_date, seed=seed)
