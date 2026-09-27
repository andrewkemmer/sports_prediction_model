"""Point-in-time NBA feature engine.

Every aggregate is built from rows strictly before its target game.  The
module accepts settled game/team/player-normalized frames and never consults a
future result while constructing a prior feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from backend import config
    from backend import nba_sources as sources
except ImportError:
    import config
    import nba_sources as sources

REQUIRED_GAME_COLS = [
    "game_id", "season", "gameday", "home_team", "away_team",
    "home_score", "away_score",
]

#: The columns the play-by-play rollup contributes to the ladder.
#:
#: This is an ALLOW-LIST, and that is the point. The rollup also re-counts the
#: box score's own columns - fga, fgm, fta, ftm, oreb, dreb, tov, pf - and those
#: counts are the cross-check, not the model's input: where they disagree the
#: box score wins, because it is a sum over player lines while the event stream
#: is a narration, and the residual differences are real (offensive rebounds
#: agree on 88 of 90 team-games, personal fouls on 60). Merging them anyway
#: would leave a noisier duplicate of every fact in the frame, one suffix away
#: from being read by a future change that does not know which one is which.
#:
#: So the rollup contributes only what an event stream knows and a box score
#: does not: where shots came from, which turnovers were live ball, how many
#: possessions ended in a foul with the shot made, and the quarter split.
#:
#: The rate denominators are on the list too, and they are the one deliberate
#: exception: a run with no team facts still needs ``fga`` to divide by. They
#: are merged with the event frame keeping the left (box-score) value on a
#: collision, so they are a fallback and not an override.
EVENT_ONLY_COLUMNS = (
    "rim_attempts", "mid_attempts", "corner_three_attempts",
    "live_turnovers", "shooting_fouls", "personal_fouls", "and_in",
    "shot_distance", "possessions",
    "q1_points", "q2_points", "q3_points", "q4_points", "ot_points",
)

#: Rate denominators, merged as a fallback when the box score has none.
EVENT_DENOMINATOR_COLUMNS = tuple(sorted(set(config.EVENT_RATE_DENOMINATORS.values())))

#: Numerator for each event rate. The denominators live in
#: ``config.EVENT_RATE_DENOMINATORS``; naming the numerators here keeps the two
#: halves of each ratio in the same place instead of spread across a config
#: module and the feature code that reads it.
_EVENT_RATE_NUMERATORS = {
    "three_rate": "fg3a",
    "rim_rate": "rim_attempts",
    "live_tov_rate": "live_turnovers",
    "and_in_rate": "and_in",
}


def team_events(games: pd.DataFrame) -> pd.DataFrame:
    """Explode game rows into one chronological row per team."""
    missing = [c for c in REQUIRED_GAME_COLS if c not in games.columns]
    if missing:
        raise ValueError(f"team_events missing columns: {missing}")
    dates = pd.to_datetime(games["gameday"], errors="coerce")
    common = {
        "game_id": games["game_id"].map(str),
        "season": games["season"],
        "gameday": dates,
        "game_type": games.get("game_type", pd.Series(config.GAME_TYPE_REG, index=games.index)),
    }
    home = pd.DataFrame({
        **common, "team": games["home_team"].astype(str),
        "opponent": games["away_team"].astype(str), "is_home": True,
        "for": pd.to_numeric(games["home_score"], errors="coerce"),
        "against": pd.to_numeric(games["away_score"], errors="coerce"),
    })
    away = pd.DataFrame({
        **common, "team": games["away_team"].astype(str),
        "opponent": games["home_team"].astype(str), "is_home": False,
        "for": pd.to_numeric(games["away_score"], errors="coerce"),
        "against": pd.to_numeric(games["home_score"], errors="coerce"),
    })
    ev = pd.concat([home, away], ignore_index=True)
    ev["net_from_team"] = ev["for"] - ev["against"]
    ev["team_win"] = np.select(
        [ev["for"] > ev["against"], ev["for"] < ev["against"]],
        [1.0, 0.0], default=np.nan,
    )
    return ev.sort_values(["gameday", "game_id", "team", "is_home"],
                          ascending=[True, True, True, False]).reset_index(drop=True)


def _elo_apply(events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Apply Elo in game order and attach each team's entering rating.

    A game updates ratings only when both scores are known.  Rows from a
    scheduled slate never update the state, so later scheduled cards cannot
    inherit a result from another pending game.
    """
    ev = events.sort_values(["gameday", "game_id", "is_home"],
                            ascending=[True, True, False]).reset_index(drop=True)
    ratings: dict[str, float] = {}
    entering: dict[tuple[str, str], float] = {}
    last_season: float | None = None
    for gid, grouped in ev.groupby("game_id", sort=False):
        rows = list(grouped.itertuples(index=False))
        if len(rows) < 2:
            continue
        home = next((r for r in rows if bool(getattr(r, "is_home", False))), rows[0])
        away = next((r for r in rows if not bool(getattr(r, "is_home", False))), rows[1])
        try:
            season = float(getattr(home, "season"))
        except (TypeError, ValueError):
            season = None
        if last_season is not None and season is not None and season != last_season:
            ratings = {t: r + config.ELO_REVERT_FACTOR * (config.ELO_PRIOR - r)
                       for t, r in ratings.items()}
        if season is not None:
            last_season = season
        ra = ratings.get(home.team, config.ELO_PRIOR)
        rb = ratings.get(away.team, config.ELO_PRIOR)
        exp_home = 1.0 / (1.0 + 10.0 ** ((rb + config.ELO_HOME_ADV - ra) / config.ELO_SCALE))
        exp_away = 1.0 - exp_home
        entering[(str(gid), str(home.team))] = ra
        entering[(str(gid), str(away.team))] = rb
        home_win, away_win = getattr(home, "team_win", np.nan), getattr(away, "team_win", np.nan)
        if pd.notna(home_win) and pd.notna(away_win):
            ratings[home.team] = ra + config.ELO_K * (float(home_win) - exp_home)
            ratings[away.team] = rb + config.ELO_K * (float(away_win) - exp_away)
    ev["elo_entering"] = [entering.get((str(g), str(t)), config.ELO_PRIOR)
                          for g, t in zip(ev.game_id, ev.team)]
    return ev, ratings


def compute_elo(events: pd.DataFrame) -> pd.DataFrame:
    return _elo_apply(events)[0]


def _trailing(srt: pd.DataFrame, col: str, window: int) -> np.ndarray:
    if col not in srt:
        return np.full(len(srt), np.nan)
    roll = srt.groupby("team", sort=False)[col].rolling(
        window, min_periods=1).mean()
    return (roll.groupby(level=0).shift(1).reset_index(level=0, drop=True)
            .to_numpy(dtype=float))


def _ewm(srt: pd.DataFrame, col: str,
         halflife: float = config.EWM_HALFLIFE) -> np.ndarray:
    if col not in srt:
        return np.full(len(srt), np.nan)
    roll = srt.groupby("team", sort=False)[col].ewm(
        halflife=halflife, min_periods=1, adjust=True).mean()
    return (roll.groupby(level=0).shift(1).reset_index(level=0, drop=True)
            .to_numpy(dtype=float))


def _attach_stats(ev: pd.DataFrame, team_stats: pd.DataFrame | None) -> pd.DataFrame:
    out = ev.copy()
    if team_stats is not None and len(team_stats):
        # Only merge fact columns; dimensions such as gameday/home_team are
        # already represented by the exploded event frame.  A team frame can
        # carry both a traditional and an advanced fact family, so make
        # the key unique before joining rather than allowing a many-to-many
        # merge to duplicate a game's team event.
        stats = team_stats.copy()
        if "game_id" not in stats.columns or "team" not in stats.columns:
            return out
        stat_cols = [
            c for c in stats.columns
            if c not in {
                "game_id", "team", "gameday", "home_team", "away_team",
                "is_home", "opponent", "home_score", "away_score",
            }
        ]
        if stat_cols:
            stats = stats[["game_id", "team", *stat_cols]].copy()
            stats["game_id"] = stats["game_id"].astype(str)
            stats["team"] = stats["team"].astype(str)
            # Prefer the most complete duplicate fact row deterministically.
            quality = stats[stat_cols].notna().sum(axis=1)
            stats = (stats.assign(_quality=quality)
                     .sort_values(["game_id", "team", "_quality"], ascending=[True, True, False])
                     .drop_duplicates(["game_id", "team"], keep="first")
                     .drop(columns="_quality"))
            out = out.merge(
                stats,
                on=["game_id", "team"],
                how="left",
                validate="many_to_one",
            )
    defaults = {
        "points_for": out["for"], "points_against": out["against"],
        "net_points": out["for"] - out["against"],
        "off_rating": 100.0 + (out["for"] - out["against"]),
        "def_rating": 100.0 - (out["for"] - out["against"]),
        "pace": (out["for"] + out["against"]).clip(lower=1),
        "efg_pct": 0.5, "fg_pct": 0.45, "three_point_pct": 0.35,
        "free_throw_pct": 0.78, "ast": 0.0, "tov": 0.0, "reb": 0.0,
        "oreb": 0.0, "dreb": 0.0, "stl": 0.0, "blk": 0.0,
    }
    for col, default in defaults.items():
        if col not in out:
            out[col] = default
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].fillna(pd.Series(default, index=out.index))
    for col, default in (("turnover_margin", 0.0), ("rebound_margin", 0.0)):
        if col not in out:
            out[col] = default
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)
    # Normalize candidate metric aliases to one canonical per-team field.
    aliases = {
        "points_for_pg": "points_for", "points_against_pg": "points_against",
        "assists_per_game": "ast", "rebounds_per_game": "reb",
        "turnovers_per_game": "tov",
    }
    for canonical, source in aliases.items():
        source_values = out[source] if source in out else pd.Series(np.nan, index=out.index)
        out[canonical] = pd.to_numeric(source_values, errors="coerce")
    out["ast_per_game"] = out["ast"]
    return out


def _attach_events(ev: pd.DataFrame,
                   event_stats: pd.DataFrame | None) -> pd.DataFrame:
    """Merge the play-by-play rollup onto the event frame and derive its rates.

    The rollup is optional. A window with no play-by-play still builds every
    other feature; the event columns are simply absent and the ladder fills them
    with neutral defaults rather than failing. That is deliberate - the event
    features are then NaN-free but constant, and a constant column carries no
    signal instead of a wrong one.

    The rates are computed here, on the per-game values, BEFORE any window is
    applied, so a rolling mean of a rate is a mean of rates rather than a ratio
    of means. Those are different numbers and only the first is a rate.
    """
    out = ev.copy()
    raw_columns = list(config.EVENT_TRAILING_SPECS) + list(
        config.EVENT_RATE_DENOMINATORS.values())
    if event_stats is not None and len(event_stats):
        if {"game_id", "team"}.issubset(event_stats.columns):
            extra = [c for c in dict.fromkeys(
                EVENT_ONLY_COLUMNS + EVENT_DENOMINATOR_COLUMNS)
                if c in event_stats.columns]
            if extra:
                stats = event_stats[["game_id", "team", *extra]].copy()
                stats["game_id"] = stats.game_id.astype(str)
                stats["team"] = stats.team.astype(str)
                # The rollup can carry both a traditional and an event family
                # for the same fact; keep the most complete row per team-game
                # so the join stays one-to-one.
                quality = stats[extra].notna().sum(axis=1)
                stats = (stats.assign(_q=quality)
                         .sort_values(["game_id", "team", "_q"],
                                      ascending=[True, True, False])
                         .drop_duplicates(["game_id", "team"], keep="first")
                         .drop(columns="_q"))
                out = out.merge(stats, on=["game_id", "team"], how="left",
                                validate="many_to_one",
                                suffixes=("", "_event"))
    # Neutral defaults. A rate and a mean are both undefined with no
    # denominator and no observations, so they are NaN; a count is zero when
    # the rollup said nothing happened. The distinction matters: defaulting a
    # mean to 0 would make a team with no play-by-play look like the league's
    # best close-range team. ``fga`` and ``possessions`` are rate denominators:
    # the box score normally supplies fga, but a run with no team facts must
    # leave it undefined rather than zero, or every rate divides by zero and
    # reads as "no rim pressure" instead of "unknown".
    neutral = {"fga": np.nan, "possessions": np.nan, "shot_distance": np.nan}
    for column in raw_columns:
        if column not in out:
            out[column] = neutral.get(column, 0.0)
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column, default in neutral.items():
        if column not in out:
            out[column] = default
    for rate, denominator in config.EVENT_RATE_DENOMINATORS.items():
        numerator = _EVENT_RATE_NUMERATORS[rate]
        num = (pd.to_numeric(out[numerator], errors="coerce")
               if numerator in out else pd.Series(np.nan, index=out.index))
        den = (pd.to_numeric(out[denominator], errors="coerce")
               if denominator in out else pd.Series(np.nan, index=out.index))
        with np.errstate(divide="ignore", invalid="ignore"):
            out[rate] = np.where(den > 0, num / den, np.nan)
    return out


def team_stats_ladder(events: pd.DataFrame,
                      team_stats: pd.DataFrame | None = None,
                      event_stats: pd.DataFrame | None = None
                      ) -> pd.DataFrame:
    """Build per-team trailing state, sorted chronologically."""
    ev = _attach_stats(events, team_stats)
    ev = _attach_events(ev, event_stats)
    srt = ev.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    # Carry each team's event profile forward across games that have no
    # play-by-play. The sweep is incremental - a few hundred games per run - so
    # for most of a run's history the rollup is simply absent, and an EWM over
    # a column with interior NaNs propagates them: every value after the first
    # gap is NaN and every event feature is empty. Filling forward uses only
    # what the team had already shown, so it stays point-in-time safe, and a
    # team whose profile is genuinely unknown stays NaN rather than becoming a
    # league average.
    event_columns = [c for c in srt.columns
                     if c in config.EVENT_TRAILING_SPECS
                     or c in config.EVENT_RATE_DENOMINATORS
                     or c in set(config.EVENT_RATE_DENOMINATORS.values())
                     or c in {"rim_attempts", "mid_attempts",
                              "corner_three_attempts", "live_turnovers",
                              "and_in", "shooting_fouls", "avg_shot_distance",
                              "possessions", "q4_points", "ot_points"}]
    if event_columns:
        srt[event_columns] = (srt.groupby("team", sort=False)[event_columns]
                              .ffill())
    srt["elo_entering"] = pd.to_numeric(srt["elo_entering"], errors="coerce")
    srt["win_pct"] = _trailing(srt, "team_win", config.WINPCT_WINDOW)
    prior_date = srt.groupby("team", sort=False)["gameday"].shift()
    srt["rest_days"] = (srt["gameday"] - prior_date).dt.total_seconds() / 86400.0
    srt["back_to_back"] = np.where(srt.rest_days.notna(),
                                    (srt.rest_days <= 1).astype(float), np.nan)
    srt["ewm_net_points"] = _ewm(srt, "net_points")
    srt["ewm_off_rating"] = _ewm(srt, "off_rating")
    srt["ewm_def_rating"] = _ewm(srt, "def_rating")
    srt["ewm_pace"] = _ewm(srt, "pace")
    srt["ewm_efg_pct"] = _ewm(srt, "efg_pct")
    srt["ewm_turnover_margin"] = _ewm(srt, "turnover_margin")
    srt["ewm_rebound_margin"] = _ewm(srt, "rebound_margin")
    srt["ewm_ast_per_game"] = _ewm(srt, "ast_per_game")
    for metric, windows in config.TEAM_CANDIDATE_TRAILING_SPECS.items():
        if metric not in srt:
            srt[metric] = np.nan
        for window in windows:
            col = f"{metric}_{window}"
            srt[col] = (_ewm(srt, metric) if window == "ewm"
                        else _trailing(srt, metric, config.PBP_ROLL_WINDOW))
    # The play-by-play contribution, read over the same EWM machinery the
    # box-score features use. ``shot_distance`` arrives as the rollup's
    # per-game mean rather than as a sum, which is why it is excluded from the
    # sum-aggregated list above and read here.
    for metric, window in config.EVENT_TRAILING_SPECS.items():
        if metric not in srt:
            srt[metric] = np.nan
        srt[f"{metric}_{window}"] = (_ewm(srt, metric)
                                     if window == "ewm"
                                     else _trailing(srt, metric,
                                                    config.PBP_ROLL_WINDOW))
    return srt


def _side(ladder: pd.DataFrame, ids: pd.Index, col: str, home: bool) -> np.ndarray:
    if col not in ladder:
        return np.full(len(ids), np.nan)
    mask = ladder.is_home if home else ~ladder.is_home
    side = ladder.loc[mask, ["game_id", col]].drop_duplicates("game_id").set_index("game_id")[col]
    return side.reindex(ids).to_numpy(dtype=float)


def _diff(ladder: pd.DataFrame, ids: pd.Index, col: str) -> np.ndarray:
    return _side(ladder, ids, col, True) - _side(ladder, ids, col, False)


def _prior_records(events: pd.DataFrame) -> dict[tuple[str, str], str]:
    counts: dict[str, list[int]] = {}
    result: dict[tuple[str, str], str] = {}
    if events is None or events.empty or not {"gameday", "game_id", "team"}.issubset(events.columns):
        return result
    ev = events.sort_values(["gameday", "game_id"])
    for gid, grouped in ev.groupby("game_id", sort=False):
        rows = list(grouped.itertuples(index=False))
        for row in rows:
            result[(str(gid), str(row.team))] = (
                f"{counts.get(row.team, [0, 0])[0]}-{counts.get(row.team, [0, 0])[1]}")
        for row in rows:
            if pd.notna(getattr(row, "team_win", np.nan)):
                rec = counts.setdefault(row.team, [0, 0])
                if float(row.team_win) >= 0.5:
                    rec[0] += 1
                else:
                    rec[1] += 1
    return result


def _attach_contract(df: pd.DataFrame, ladder: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    ids = out.game_id.astype(str)
    out["elo_diff"] = _diff(ladder, ids, "elo_entering")
    out["win_pct_diff"] = _diff(ladder, ids, "win_pct")
    out["rest_days_diff"] = _diff(ladder, ids, "rest_days")
    out["back_to_back_diff"] = _diff(ladder, ids, "back_to_back")
    for base in ("ewm_net_points", "ewm_off_rating", "ewm_def_rating",
                 "ewm_pace", "ewm_efg_pct", "ewm_turnover_margin",
                 "ewm_rebound_margin", "ewm_ast_per_game"):
        out[f"{base}_diff"] = _diff(ladder, ids, base)
    for col, side in (("elo_entering", "elo"), ("win_pct", "win_pct"),
                      ("ewm_off_rating", "ewm_off_rating"),
                      ("ewm_def_rating", "ewm_def_rating"), ("rest_days", "rest_days")):
        out[f"{side}_home"] = _side(ladder, ids, col, True)
        out[f"{side}_away"] = _side(ladder, ids, col, False)
    out["is_home"] = 1.0
    raw_type = out.get("game_type", pd.Series(config.GAME_TYPE_REG, index=out.index))
    type_num = pd.to_numeric(raw_type, errors="coerce")
    type_text = raw_type.astype("string").str.lower()
    out["is_playoffs"] = ((type_num == config.GAME_TYPE_POST) |
                          type_text.str.contains("play|post", regex=True, na=False)).astype(float)
    for metric, windows in config.TEAM_CANDIDATE_TRAILING_SPECS.items():
        for window in windows:
            col = f"{metric}_{window}"
            h, a = _side(ladder, ids, col, True), _side(ladder, ids, col, False)
            out[f"nba_{col}_diff"] = h - a
            out[f"nba_{col}_home"] = h
            out[f"nba_{col}_away"] = a
    # The play-by-play contribution: every event feature is a home-minus-away
    # difference of the same trailing statistic, which is the form the rest of
    # the contract uses and the form a level feature must not take. The window
    # is part of how the ladder reads the metric, not part of the feature's
    # name, so the contract column is ``event_<metric>_diff`` and the config
    # declaration and this line cannot drift apart.
    for metric, window in config.EVENT_TRAILING_SPECS.items():
        out[f"event_{metric}_diff"] = _diff(ladder, ids, f"{metric}_{window}")
    for col in config.MONEYLINE_FEATURE_COLS:
        if col not in out:
            out[col] = np.nan
    return out


def _records_for(out: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if out is None or out.empty or events is None or events.empty:
        if out is not None and not out.empty:
            out["home_record"] = ""
            out["away_record"] = ""
        return out
    records = _prior_records(events)
    out["home_record"] = [records.get((str(g), str(h)), "")
                          for g, h in zip(out.game_id, out.home_team)]
    out["away_record"] = [records.get((str(g), str(a)), "")
                          for g, a in zip(out.game_id, out.away_team)]
    return out


def build_game_features(games: pd.DataFrame,
                        team_stats: pd.DataFrame | None = None,
                        event_stats: pd.DataFrame | None = None
                        ) -> pd.DataFrame:
    ev = compute_elo(team_events(games))
    ladder = team_stats_ladder(ev, team_stats, event_stats)
    out = _attach_contract(games.copy(), ladder)
    out = _records_for(out, ev)
    out["home_score"] = pd.to_numeric(out.get("home_score"), errors="coerce")
    out["away_score"] = pd.to_numeric(out.get("away_score"), errors="coerce")
    out["margin"] = out.home_score - out.away_score
    out["total"] = out.home_score + out.away_score
    out["home_win"] = np.where(out.margin > 0, 1.0,
                               np.where(out.margin < 0, 0.0, np.nan))
    return out


def build_slate_features(schedule: pd.DataFrame,
                         team_stats: pd.DataFrame | None = None,
                         event_stats: pd.DataFrame | None = None
                         ) -> pd.DataFrame:
    sched = schedule.copy()
    for col in ("home_score", "away_score"):
        if col not in sched:
            sched[col] = np.nan
        sched[col] = pd.to_numeric(sched[col], errors="coerce")
    decided = sched[sched.home_score.notna() & sched.away_score.notna()].copy()
    pending = sched[sched.home_score.isna() | sched.away_score.isna()].copy()
    # A postponed game is neither decided nor upcoming, and MLB's decided-frame
    # rule says so in one line: "postponements and pregame rows are excluded."
    # Left in ``pending`` it is indistinguishable from a real upcoming game, so
    # a slate fills with games nobody is going to play and the board stops
    # showing tonight. When ESPN reschedules it the event keeps its id and
    # moves to the new date, so it re-enters here on the day it is played.
    if "game_status_detail" in pending.columns and len(pending):
        detail = pending.game_status_detail.astype(str).str.lower()
        postponed = detail.map(sources.is_postponed_detail)
        if postponed.any():
            sched = sched[~(sched.index.isin(pending.index[postponed]))].copy()
            pending = pending[~postponed].copy()
    if pending.empty:
        return pd.DataFrame()
    decided_events = team_events(decided) if len(decided) else pd.DataFrame()
    if len(decided_events):
        _, ratings = _elo_apply(decided_events)
    else:
        ratings = {}
    pending_events = team_events(pending)
    pending_events["elo_entering"] = pending_events.team.map(
        lambda t: ratings.get(t, config.ELO_PRIOR))
    combined = pd.concat([decided_events, pending_events], ignore_index=True)
    ladder = team_stats_ladder(combined, team_stats, event_stats)
    out = _attach_contract(pending, ladder)
    out = _records_for(out, decided_events)
    out["home_score"] = np.nan
    out["away_score"] = np.nan
    out["margin"] = np.nan
    out["total"] = np.nan
    out["home_win"] = np.nan
    return out


def linear_feature_columns(active: list[str] | None = None) -> list[str]:
    cols = config.active_moneyline_feature_cols() if active is None else list(active)
    return [c for c in cols if c not in config.RAW_PER_SIDE_COLS]


def linear_view(df: pd.DataFrame) -> pd.DataFrame:
    cols = linear_feature_columns()
    return df.reindex(columns=cols).astype(float)


def team_category_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    home = df.get("home_team", pd.Series("", index=df.index))
    away = df.get("away_team", pd.Series("", index=df.index))
    out["home_team_id"] = home.map(config.team_category_id).astype("int64")
    out["away_team_id"] = away.map(config.team_category_id).astype("int64")
    return out


def tree_numeric_columns() -> list[str]:
    return list(config.active_moneyline_feature_cols())


def tree_view(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in config.active_moneyline_feature_cols()]
    out = df.reindex(columns=cols).astype(float)
    cats = team_category_ids(df)
    categories = list(range(config.UNK_TEAM_ID + 1))
    for col in config.TREE_CATEGORICAL_COLS:
        # Fixed categories make serving independent of the slate's team set.
        out[col] = pd.Categorical(cats[col], categories=categories)
    return out


def feature_coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in config.active_moneyline_feature_cols():
        values = pd.to_numeric(df[feature], errors="coerce") if feature in df else pd.Series(dtype=float)
        rows.append({
            "feature": feature, "n_games": int(len(df)),
            "coverage_pct": round(100 * float(values.notna().mean()), 2) if len(values) else 0.0,
            "mean": float(values.mean()) if len(values) and values.notna().any() else np.nan,
            "std": float(values.std()) if len(values) and values.notna().sum() > 1 else np.nan,
        })
    return pd.DataFrame(rows)
