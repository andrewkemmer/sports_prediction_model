"""PIT player-card enrichment for the NBA board.

The card leader is the highest prior rolling scorer over the team's five most
recent completed games before the target.  A player must appear at least three
times and average at least 15 minutes in that window; PPG/APG are reported on
the same appearance-qualified window.  Missing qualification is represented by
NaN/None, never by a fabricated player.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from backend import config
except ImportError:
    import config


def _completed_team_games(games: pd.DataFrame | None, team: str,
                          target_date) -> list[str]:
    if games is None or not len(games):
        return []
    frame = games.copy()
    if not {"gameday", "game_id", "home_score", "away_score"}.issubset(frame.columns):
        return []
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="coerce")
    target = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(target):
        return []
    team_key = config.normalize_team_abbr(team)
    home = frame.get("home_team", pd.Series("", index=frame.index)).map(config.normalize_team_abbr)
    away = frame.get("away_team", pd.Series("", index=frame.index)).map(config.normalize_team_abbr)
    mask = ((home == team_key) | (away == team_key)) & frame.gameday.lt(target)
    mask &= frame.home_score.notna() & frame.away_score.notna()
    selected = frame.loc[mask, ["gameday", "game_id"]].copy()
    selected["game_id"] = selected.game_id.astype(str)
    return list(dict.fromkeys(selected.sort_values(
        ["gameday", "game_id"], ascending=[False, False]).game_id.tolist()))


def _player_for_team(stats: pd.DataFrame | None, team: str, target_date,
                     games: pd.DataFrame | None = None) -> dict:
    if stats is None or not len(stats):
        return {}
    required = {"team", "player_id", "player_name", "game_id",
                "points", "assists", "minutes"}
    if not required.issubset(stats.columns):
        return {}
    game_ids = _completed_team_games(games, team, target_date)
    frame = stats.copy()
    if not game_ids:
        # A supplied schedule with no completed prior game is an explicit
        # fail-closed state.  Never fall back to arbitrary stat rows: doing so
        # could make an unqualified card look like a five-game leader.
        if games is not None:
            return {}
        # Direct-stat callers may omit a schedule; use only prior completed
        # rows, but retain the same five-game window semantics.
        if "gameday" not in frame or "game_id" not in frame:
            return {}
        frame["gameday"] = pd.to_datetime(frame["gameday"], errors="coerce")
        target = pd.to_datetime(target_date, errors="coerce")
        team_key = config.normalize_team_abbr(team)
        mask = (frame.team.astype(str).map(config.normalize_team_abbr).eq(team_key)
                & frame.gameday.lt(target))
        game_ids = list(dict.fromkeys(
            frame.loc[mask].sort_values("gameday", ascending=False)
            .game_id.astype(str)))[:config.PLAYER_WINDOW_GAMES]
    else:
        frame = frame[frame.game_id.astype(str).isin(set(game_ids))].copy()
    if frame.empty:
        return {}
    frame["team_key"] = frame.team.astype(str).map(config.normalize_team_abbr)
    frame = frame[frame.team_key.eq(config.normalize_team_abbr(team))]
    frame = frame[frame.game_id.astype(str).isin(set(game_ids[:config.PLAYER_WINDOW_GAMES]))]
    if frame.empty:
        return {}
    frame["points"] = pd.to_numeric(frame.get("points"), errors="coerce")
    frame["assists"] = pd.to_numeric(frame.get("assists"), errors="coerce")
    frame["minutes"] = pd.to_numeric(frame.get("minutes"), errors="coerce")
    candidates = []
    for pid, group in frame.groupby("player_id", dropna=False):
        names = group.player_name.dropna().astype(str)
        name = names.iloc[0] if len(names) else str(pid)
        appearances = int(group.game_id.astype(str).nunique())
        if appearances < config.PLAYER_MIN_GAMES:
            continue
        minutes = group.minutes.dropna()
        # The qualification rule is an average over the same appearance-
        # qualified window.  Missing minutes make that average unknowable, so
        # they fail closed rather than being silently dropped from the mean.
        if len(minutes) < appearances or float(minutes.mean()) < config.PLAYER_MIN_MINUTES:
            continue
        points = group.points.dropna()
        if len(points) < appearances:
            continue
        # Stable tie break: highest PPG, then player ID.  This keeps cards
        # deterministic when two players share a rounded average.
        candidates.append((float(points.mean()),
                           float(group.assists.mean()) if group.assists.notna().any() else np.nan,
                           str(name or pid), appearances, str(pid)))
    if not candidates:
        return {}
    ppg, apg, name, games, _pid = max(candidates, key=lambda x: (x[0], x[4]))
    return {"name": name, "ppg": ppg, "apg": apg, "games": games}


def build_player_leader(slate: pd.DataFrame, player_stats: pd.DataFrame | None,
                        games: pd.DataFrame | None = None) -> pd.DataFrame:
    out = slate.copy().reset_index(drop=True)
    for col in config.PLAYER_FIELDS:
        out[col] = None if col.endswith("_name") else np.nan
    for i, row in out.iterrows():
        target = row.get("gameday", row.get("game_date"))
        for side, team_col in (("home", "home_team"), ("away", "away_team")):
            record = _player_for_team(player_stats, row.get(team_col, ""), target, games)
            out.loc[i, f"p_{side}_name"] = record.get("name") or None
            out.loc[i, f"p_{side}_ppg"] = record.get("ppg", np.nan)
            out.loc[i, f"p_{side}_apg"] = record.get("apg", np.nan)
            out.loc[i, f"p_{side}_games"] = record.get("games", np.nan)
    return out
