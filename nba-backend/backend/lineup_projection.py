"""Projected-lineup aggregates over the shrunk player TS ratings.

The NBA analogue of MLB's ``_LINEUP_AGG_ROSTER``, and it exists for the reason
MLB's exists. MLB's comment on the widen step is the whole argument:

    "Without this the IL filter cannot bind AT ALL: an injured player is absent
    from the participant pool by construction, so subtracting him is a no-op."

That failure is easy to reproduce by accident. A pool drawn only from players
who appeared recently cannot subtract an injured player, because an injured
player is precisely the one who did not appear. So the pool is WIDENED to every
team member with a rating row inside a short lookback, and only THEN is the
injury filter applied. The order is the feature.

Three further properties are inherited deliberately:

* **LOOKUP-BACK IS MOST-RECENT-AT-OR-BEFORE.** A candidate's rating is the
  latest row on or before the game, never a row after it. MLB additionally
  LAG-shifts so a player's own game supplies his PRIOR rating; here the rating
  rows are already strictly prior, so the date comparison does that work.
* **REMOVAL, NOT ZERO-WEIGHTING.** An injured player leaves the pool so the
  best healthy replacement inherits the slot. Multiplying his rating by zero
  instead would leave him in the pool dragging the mean toward zero, which is
  the opposite of the intent and produces a quietly wrong number.
* **NO PADDING.** A short-handed team is averaged over the players it actually
  has. Padding to eight would fabricate full strength from a depleted roster.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from backend import config
except ImportError:
    import config

AGG_COLUMNS = ["gameday", "team", "pool_size", "healthy_size",
               "lineup_out_count", "lineup_healthy_frac",
               "lineup_ts_concentration",
               "lineup_ts_mean", "lineup_ts_top3", "lineup_ts_std",
               "lineup_ts_rest_count"] + [
    f"pl_ts_{p.lower()}" for p in config.PLAYER_TS_POSITIONS]

#: The seven model-facing features, as home-minus-away diffs, which is the
#: convention ``features._attach_contract`` already uses for every other
#: paired feature. Each one earns its place against a specific failure:
#:
#: 1-4 are MLB's ``lineup_woba_*`` mirror. 5-7 exist because 1-4 cannot see
#: WHY two projected lineups differ - a team missing a star and a team missing
#: a bench player can carry an identical ``lineup_ts_mean`` and they are not
#: the same bet. Availability asymmetry (5), proportional depletion (6) and
#: star-dependency (7) are the three ways that shows up.
DIFF_FEATURES = [
    "lineup_ts_mean_diff",
    "lineup_ts_top3_diff",
    "lineup_ts_std_diff",
    "lineup_ts_rest_count_diff",
    "lineup_out_count_diff",
    "lineup_healthy_frac_diff",
    "lineup_ts_concentration_diff",
]

#: The nine position-segmented features: each position's projected-lineup
#: shooting on BOTH sides plus the difference.
#:
#: The sides are kept rather than differenced away because the two halves of a
#: position split carry different information. A diff says the home lineup is
#: better at shooting; the sides say WHO is better and by how much, and a
#: model that can see "away is thin at centre" does not have to infer it from a
#: near-zero difference between two bad numbers. This is the same reason
#: ``lineup_out_count`` is kept alongside ``lineup_healthy_frac``.
#:
#: Only G/F/C exist because that is all the free source publishes; see
#: ``nba_injury_report`` / ``nba_sources.position_query``. A five-way guard and
#: wing split is what a lineup model would really want and is not obtainable.
POSITION_TS_SOURCES = [f"pl_ts_{p.lower()}" for p in config.PLAYER_TS_POSITIONS]
#: Published order is per-POSITION grouped - ``c_away, c_home, c_diff``, then
#: ``f_*``, then ``g_*`` - C first, matching the contract order in
#: ``config.PLAYER_TS_POSITION_FEATURE_COLS``. ``PLAYER_TS_POSITIONS`` itself
#: is ordered G/F/C for the prior tables; the PUBLISHED column order follows
#: the contract, not the config tuple's internal order. The sides are kept
#: rather than differenced away because the two halves of a position split
#: carry different information: a diff says the home lineup is better at
#: shooting; the sides say WHO is better and by how much.
_POSITION_PUBLISHED_ORDER = ("c", "f", "g")
POSITION_TS_FEATURES = [
    f"pl_ts_{position}_{side}"
    for position in _POSITION_PUBLISHED_ORDER
    for side in ("away", "home", "diff")]


# ---------------------------------------------------------------------------
# The point-in-time designation filter.
#
# The contract, in one place so production and the A/B harness cannot drift:
#
# * A player designated Out, Doubtful or Recovery in the last report published
#   STRICTLY BEFORE his game's tipoff is REMOVED from that game's pool (the
#   replacement inherits the slot - never zero-weighted in place, which would
#   drag the mean toward zero). "Injured reserve" is inside Out: the league
#   files IR players as Out, see nba_injury_report.DESIGNATION_STATE.
# * A designation applies ONLY to the game it was filed for. Every other game
#   the player appears in keeps his rating, so a player who sits one night out
#   still carries his full rolling lagged rating into the next game he plays -
#   the rating is a property of the games he PLAYED, the removal is a property
#   of the single game he was filed out of.
# * The removal is strictly subtractive and keyed at (date, team, PLAYER).
#   A team-level key would bench whole rosters; a date-level key would bench a
#   player from games he was cleared for.
# ---------------------------------------------------------------------------

#: Designations that remove a player from the upcoming game's pool. Mirrors
#: ``nba_injury_report.WILL_NOT_PLAY`` plus Doubtful, which the measured
#: play-rate table (0 for 5) also proves to be an absence. IR is not listed
#: separately: the league never files it, an IR player IS filed as Out.
REMOVING_DESIGNATIONS = ("out", "doubtful", "recovery")


def designation_out_keys(designations) -> set:
    """``(Timestamp, team, player)`` keys whose owners are out of THAT game.

    ``designations`` is the PIT designation frame (one row per player per
    game per filing) as produced by ``backfill_injury_designations`` or
    ``nba_injury_report.pre_game_designations``. Only the removing
    designations produce keys, and only for games they actually cover.
    """
    if designations is None or not len(designations):
        return set()
    needed = {"gameday", "team", "player", "status"}
    if not needed.issubset(designations.columns):
        return set()
    keys = set()
    for gameday, team, player, status in zip(
            pd.to_datetime(designations.gameday, errors="coerce"),
            designations.team, designations.player, designations.status):
        if pd.isna(gameday):
            continue
        if str(status).strip().lower() in REMOVING_DESIGNATIONS:
            keys.add((gameday, str(team).strip(), str(player).strip()))
    return keys


def load_designations(root) -> "pd.DataFrame | None":
    """Every backfilled PIT designation shard under ``root``, as one frame.

    The backfill writes one shard per window it is run over, so the directory
    holds a UNION of windows, not a sequence of replacements. Reading one file
    - and the production path read ``sorted(...)[-1]``, the lexically LAST -
    makes the archive's coverage a function of filename sort order. That is
    exactly the bug this fixes: with shards for 2025-10-21..2026-04-12 (13,168
    records) and 2026-01-08..2026-01-12 (400), the run used the 400-row shard,
    so the injury removal bound on five days of a six-month training window and
    looked perfectly healthy on the other ~180. ``20260108_20260112`` sorts
    after ``20251021_20260412`` and ``2025...`` sorts after ``2026...`` at the
    second character, so neither "first" nor "last" is ever the right answer.

    A designation is a (gameday, team, player) removal and
    :func:`designation_out_keys` reads them into a set, so a record present in
    two shards is idempotent: the union is taken and exact duplicates dropped.
    Overlapping windows therefore cost nothing and cannot double-remove.

    Returns ``None`` when there is no shard at all, which callers must report
    rather than treat as "nobody was injured" - an unfiltred pool and a league
    with no injuries are different facts.
    """
    if root is None:
        return None
    root = Path(root).expanduser()
    # The consolidated archive first, when this root holds one: the pipeline
    # publishes the shard union as ``nba_designations.parquet`` (the same way
    # MLB ships ``il_stints.parquet``), and it is a superset of what the
    # shards in this directory hold by construction. Preferring it keeps one
    # read where several would do and keeps the published copy authoritative.
    consolidated = root / "nba_designations.parquet"
    if consolidated.exists():
        try:
            frame = pd.read_parquet(consolidated)
            if len(frame):
                logger.info("PIT designations: %d record(s) from the "
                            "published archive", len(frame))
                return frame
        except Exception as exc:  # noqa: BLE001
            logger.warning("published designation archive unreadable (%s); "
                           "falling back to shards", exc)
    shards = sorted(root.glob("nba_designations_*.parquet"))
    if not shards:
        return None
    frames = []
    for path in shards:
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("designation shard %s unreadable (%s); its window "
                           "is not applied", path.name, exc)
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return None
    designations = pd.concat(frames, ignore_index=True).drop_duplicates()
    logger.info("PIT designations: %d record(s) from %d shard(s) [%s]",
                len(designations), len(frames),
                ", ".join(p.name for p in shards))
    return designations


def apply_pit_designations(ratings: pd.DataFrame, designations,
                           name_by_id: dict | None = None,
                           ) -> pd.DataFrame:
    """Remove rated players from the single game they were filed out of.

    The rating frame gains an ``is_available`` column: False exactly on rows
    matching a (date, team, player) removing designation, True everywhere
    else - including every other game the same player appears in, which is
    the "rolling lagged rating on prior games he played" requirement. The
    rating VALUES are never touched: removal happens at pool construction in
    :func:`projected_lineup`, so a replacement inherits the slot.

    ``name_by_id`` maps rating-frame ``player_id`` -> the "First Last" form
    the report uses. A player whose name cannot be resolved is never removed,
    which is the honest direction: an unmatched designation suppresses
    nothing rather than suppressing the wrong player.

    A share guard is built in. The earlier team-level-key bug removed 39,285
    rows (whole rosters) while producing plausible aggregates, so anything
    beyond a tenth of the frame means the key is too coarse and the build
    fails loudly instead of publishing a quietly disabled feature.
    """
    out = ratings.copy() if ratings is not None else pd.DataFrame()
    out["is_available"] = True
    if not out.empty and "gameday" in out.columns:
        # Normalize once, here: the keys are Timestamps, so a frame whose
        # gameday is still strings would match nothing and the filter would
        # silently not bind - the failure mode this module exists to prevent.
        out["gameday"] = pd.to_datetime(out["gameday"], errors="coerce")
    keys = designation_out_keys(designations)
    if not keys or out.empty:
        return out
    if "report_name" not in out.columns:
        if not name_by_id:
            return out
        out["report_name"] = [(name_by_id or {}).get(str(pid), "")
                              for pid in out["player_id"]]
    matched = {(g, str(t).strip(), str(n).strip())
               for g, t, n in keys}
    keep = []
    for gameday, team, name in zip(out["gameday"], out["team"],
                                   out["report_name"]):
        keep.append((pd.Timestamp(gameday), str(team).strip(),
                     str(name).strip()) not in matched)
    out["is_available"] = keep
    # The share guard only means something on a production-sized frame: one
    # removal in a six-row fixture is a third of it, and that is the test
    # working, not the key being too coarse. The earlier team-level-key bug
    # showed up at ~30% of a 100k-row frame, so the guard watches frames where
    # a share that size could only come from a coarse key.
    if len(out) >= 500:
        removed = int(len(out) - sum(keep))
        share = removed / len(out)
        if share > 0.10:
            raise RuntimeError(
                f"designation filter removed {share:.1%} of rating rows; a "
                "player-level key should remove far fewer. A team-level key "
                "would disable whole rosters.")
    return out


def _normalize(ratings: pd.DataFrame) -> pd.DataFrame:
    out = ratings.copy()
    out["player_id"] = out.player_id.astype(str)
    out["team"] = out.team.astype(str)
    out["gameday"] = pd.to_datetime(out.gameday, errors="coerce")
    for column in ("ts_shrunk", "prior_plays"):
        out[column] = pd.to_numeric(out.get(column), errors="coerce")
    return out


def projected_lineup(ratings: pd.DataFrame | None,
                     games: pd.DataFrame | None = None,
                     stints=None,
                     lookback_days: int = config.PLAYER_TS_POOL_LOOKBACK_DAYS,
                     min_plays: float = config.PLAYER_TS_MIN_PLAYS,
                     top_k: int = config.PLAYER_TS_TOP_K,
                     rest_plays: float = config.PLAYER_TS_REST_PLAYS,
                     top5_k: int = config.PLAYER_TS_TOP5_K,
                     ) -> pd.DataFrame:
    """One row per (game, team) describing the projected healthy lineup.

    Returns a well-formed EMPTY frame - never None - when there is nothing to
    project, so a caller can distinguish "no lineup" from "the pull failed"
    without a special case.
    """
    if ratings is None or not len(ratings):
        return pd.DataFrame({c: pd.Series(dtype="float64")
                             for c in AGG_COLUMNS})
    work = _normalize(ratings)
    work = work[work.team.ne("") & work.gameday.notna()]
    if work.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64")
                             for c in AGG_COLUMNS})
    if "is_available" not in work.columns:
        work["is_available"] = True

    # The games to project. A game is identified by its date and its two teams
    # rather than by an id, so this works off the rating frame alone when no
    # schedule is supplied - the ratings already know who played whom.
    if games is not None and len(games) and {"gameday"} <= set(games.columns):
        schedule = games.copy()
        schedule["gameday"] = pd.to_datetime(schedule.gameday, errors="coerce")
        targets = schedule[schedule.gameday.notna()]
    else:
        targets = pd.DataFrame({"gameday": sorted(work.gameday.unique())})

    rows = []
    # Pre-group by team ONCE. Without this, every (date, team) pair rescans
    # the whole rating frame, which is quadratic: the full decided frame is
    # ~250k rating rows over ~5,000 team-games, and the naive version spends
    # over a billion comparisons doing it. Grouping makes each team's slice the
    # only thing a projection has to look at, and changes nothing about the
    # answer - the filter inside _project_team is the same filter, over the
    # same rows, just no longer re-finding them each time.
    by_team = {str(team): slice_ for team, slice_
               in work.groupby("team", sort=False)}
    for gameday, day_games in targets.groupby("gameday", sort=False):
        teams = []
        for column in ("home_team", "away_team"):
            if column in day_games.columns:
                teams.extend(day_games[column].dropna().astype(str).tolist())
        if not teams:
            teams = sorted(work.loc[work.gameday == gameday, "team"].unique())
        for team in dict.fromkeys(teams):
            team_rows = by_team.get(str(team))
            if team_rows is None or team_rows.empty:
                continue
            row = _project_team(team_rows, str(team), gameday, stints,
                                lookback_days, min_plays, top_k, rest_plays,
                                top5_k)
            if row is not None:
                rows.append(row)
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="float64")
                             for c in AGG_COLUMNS})
    out = pd.DataFrame(rows)
    for column in AGG_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    return out[AGG_COLUMNS].sort_values(["gameday", "team"]).reset_index(drop=True)


def _project_team(work: pd.DataFrame, team: str, gameday, stints,
                  lookback_days: int, min_plays: float, top_k: int,
                  rest_plays: float, top5_k: int):
    """Widen, filter, rank - in that order, for one team and one game."""
    window_start = pd.Timestamp(gameday) - pd.Timedelta(days=lookback_days)
    # STEP 1 - WIDEN. Every team member with a rating row in the window,
    # whether or not they appeared recently. This is the step that gives the
    # injury filter something to remove.
    pool = work[(work.team == team)
                & (work.gameday <= pd.Timestamp(gameday))
                & (work.gameday >= window_start)
                & work.ts_shrunk.notna()]
    if pool.empty:
        return None
    # One row per player: the most recent rating at or before the game. A
    # player's own current game can supply his prior rating but never a rating
    # that includes the game being projected.
    latest = (pool.sort_values(["player_id", "gameday"])
              .groupby("player_id", as_index=False).tail(1))
    pool_size = len(latest)

    # STEP 2 - ELIGIBILITY. The min-plays floor is about POOL MEMBERSHIP, not
    # about the rating: shrinkage already handles a thin rating, but a player
    # with three career games is not a candidate for tonight's lineup however
    # well his rate is estimated.
    latest = latest[latest.prior_plays >= min_plays]

    # STEP 3 - FILTER. Injury removes the player from the pool entirely so a
    # replacement inherits the slot. Availability is re-evaluated AS OF the
    # game, strictly point-in-time, from the stint table when one is supplied.
    #
    # The two sources are ANDed, never replaced. The carried flag is what the
    # roster says right now; the stint table is the interval history. Letting
    # the table OVERWRITE the flag would resurrect a player the roster marks
    # unavailable but who has no interval yet - which is exactly the state a
    # fresh absence is in before the next snapshot closes it. ANDing is
    # strictly subtractive, which is the direction MLB's reconciliation relies
    # on: a source can only ever remove a player, never add one back.
    if stints is not None and len(stints):
        from injury_stints import out_at
        available = [
            bool(carried) and not out_at(stints, player, gameday)
            for carried, player in zip(latest.is_available, latest.player_id)
        ]
        latest = latest.assign(is_available=available)
    healthy = latest[latest.is_available.astype(bool)]
    healthy_size = len(healthy)
    if healthy.empty:
        return {"gameday": pd.Timestamp(gameday), "team": team,
                "pool_size": pool_size, "healthy_size": 0,
                "lineup_ts_mean": np.nan, "lineup_ts_top3": np.nan,
                "lineup_ts_std": np.nan, "lineup_ts_rest_count": np.nan}

    # STEP 4 - RANK BY PARTICIPATION, NOT BY RATING. MLB orders the pool by
    # trailing PA and averages the rating over the top nine. Ranking by the
    # rating instead would project the eight best-rated regulars and quietly
    # redefine the question from "who plays" to "who scores", which is a
    # different feature wearing this one's name.
    healthy = healthy.sort_values(["prior_plays", "gameday"], ascending=False)
    projected = healthy.head(top_k)
    ratings = projected.ts_shrunk.to_numpy(dtype=float)

    # The rest count: top-5 regulars by participation who are NOT in the
    # projected lineup. Below rest_plays a player is not a rotation regular,
    # so his absence is not news.
    regulars = healthy[healthy.prior_plays >= rest_plays].head(top5_k)
    rest_count = int(len(regulars) - len(
        regulars[regulars.player_id.isin(set(projected.player_id))]))

    mean = float(np.mean(ratings))
    top3 = float(np.mean(ratings[:3]))
    # Position-segmented shooting over the SAME projected lineup, not a second
    # projection. Segmented afterwards rather than projecting one lineup per
    # position keeps a single definition of "who plays tonight"; three separate
    # projections would each pick their own top eight and quietly disagree
    # about the team.
    segmented: dict = {}
    if "position" in projected.columns:
        for position, group in projected.groupby("position", sort=False):
            if position in config.PLAYER_TS_POSITIONS and len(group):
                segmented[f"pl_ts_{str(position).lower()}"] = float(
                    np.mean(group.ts_shrunk.to_numpy(dtype=float)))
    return {
        "gameday": pd.Timestamp(gameday),
        "team": team,
        "pool_size": pool_size,
        "healthy_size": healthy_size,
        # How many rotation players are missing, and the same as a proportion.
        # Both are kept: the COUNT says a two-man loss is a blow, the FRACTION
        # says it is a tenth of the roster rather than a third, and the two
        # move the line for very different reasons.
        "lineup_out_count": int(pool_size - healthy_size),
        "lineup_healthy_frac": (healthy_size / pool_size) if pool_size
        else np.nan,
        # Star dependency: the top three's share of the pool's average. A high
        # value means the projection rests on a few players, so losing one of
        # them costs more than the mean alone suggests.
        "lineup_ts_concentration": (top3 / mean) if mean else np.nan,
        "lineup_ts_mean": mean,
        "lineup_ts_top3": top3,
        "lineup_ts_std": float(np.std(ratings, ddof=0)) if len(ratings) > 1
        else np.nan,
        "lineup_ts_rest_count": rest_count,
        **segmented,
    }


def attach_to_slate(slate: pd.DataFrame, aggregates: pd.DataFrame) -> pd.DataFrame:
    """Attach the seven projected-lineup diff features to a slate.

    Each aggregate is resolved PER SIDE and then differenced, matching
    ``features._diff`` and every other paired feature in the contract. Doing
    the sides first and differencing second is what keeps the arithmetic
    consistent with the rest of the model: a home-minus-away difference of two
    independently-joined columns is the same number, but computing it in one
    place means there is exactly one definition of "home advantage in projected
    shooting" in the codebase.

    The join is on (date, team) because that is the grain the aggregates are
    built at, and because it degrades honestly: a slate row whose team has no
    aggregate gets NaN, which the existing imputation path handles, rather than
    a value borrowed from another team's game.

    Columns are ALWAYS created, populated or not. A feature that appears only
    on days it could be computed is a feature whose train-serve behaviour
    depends on the calendar, which is precisely the skew that retired MLB's
    six ``lineup_actual_*`` columns.
    """
    out = slate.copy() if slate is not None else pd.DataFrame()
    for column in DIFF_FEATURES:
        out[column] = np.nan
    if out.empty or aggregates is None or not len(aggregates):
        return out
    if "gameday" not in out.columns:
        return out

    sources = {
        "lineup_ts_mean_diff": "lineup_ts_mean",
        "lineup_ts_top3_diff": "lineup_ts_top3",
        "lineup_ts_std_diff": "lineup_ts_std",
        "lineup_ts_rest_count_diff": "lineup_ts_rest_count",
        "lineup_out_count_diff": "lineup_out_count",
        "lineup_healthy_frac_diff": "lineup_healthy_frac",
        "lineup_ts_concentration_diff": "lineup_ts_concentration",
    }
    out = _attach_sides(out, aggregates, list(sources.values()))
    for feature, column in sources.items():
        home = f"_home_{column}"
        away = f"_away_{column}"
        if home in out.columns and away in out.columns:
            out[feature] = out[home] - out[away]
    drop = [c for c in out.columns if c.startswith("_home_") or c.startswith("_away_")]
    return out.drop(columns=drop)


def attach_position_ts(slate: pd.DataFrame,
                       aggregates: pd.DataFrame) -> pd.DataFrame:
    """Attach the nine position-segmented shooting features, sides retained.

    Same join and same (date, team) grain as :func:`attach_to_slate`, and the
    same "always create the column" rule - a feature that exists only on days
    it could be computed is a feature whose train-serve behaviour depends on
    the calendar.

    Unlike :func:`attach_to_slate` this KEEPS the per-side columns instead of
    dropping them as scaffolding, because the sides are the point: the nine
    published features are ``{away, home, diff}`` per position, not three diffs.
    """
    out = slate.copy() if slate is not None else pd.DataFrame()
    for column in POSITION_TS_FEATURES:
        out[column] = np.nan
    if out.empty or aggregates is None or not len(aggregates):
        return out
    out = _attach_sides(out, aggregates, POSITION_TS_SOURCES)
    for column in POSITION_TS_SOURCES:
        away = f"_away_{column}"
        home = f"_home_{column}"
        if away in out.columns and home in out.columns:
            out[f"{column}_away"] = out[away]
            out[f"{column}_home"] = out[home]
            out[f"{column}_diff"] = out[home] - out[away]
    drop = [c for c in out.columns if c.startswith("_home_") or c.startswith("_away_")]
    return out.drop(columns=drop)


def _attach_sides(out: pd.DataFrame, aggregates: pd.DataFrame,
                  columns: Sequence[str]) -> pd.DataFrame:
    """Join per-team aggregates onto a slate for both sides, as ``_side_col``.

    Kept separate from the differencing so both feature families resolve a
    team's aggregate in exactly one place. A second copy of this join would be
    a second definition of "this team's number for this game", and the two
    would drift.
    """
    out = out.copy()
    out["gameday"] = pd.to_datetime(out["gameday"], errors="coerce")
    aggs = aggregates.copy()
    aggs["gameday"] = pd.to_datetime(aggs["gameday"], errors="coerce")
    lookup = aggs.set_index(["gameday", "team"])
    for side, team_column in (("home", "home_team"), ("away", "away_team")):
        if team_column not in out.columns:
            continue
        for column in columns:
            if column not in aggs.columns:
                continue
            name = f"_{side}_{column}"
            values = []
            for gameday, team in zip(out["gameday"], out[team_column]):
                try:
                    values.append(lookup.loc[(gameday, str(team)), column])
                except KeyError:
                    values.append(np.nan)
            out[name] = pd.Series(values, index=out.index, dtype="float64")
    return out


def feature_coverage(slate: pd.DataFrame) -> pd.DataFrame:
    """How many slate rows actually carry each feature, and how varied it is.

    The check that would have caught MLB's retired ``lineup_actual_*`` columns:
    a feature populated in the decided frame but always NULL at bet time trains
    dead and looks perfectly healthy. Constant columns are the other failure -
    they consume a slot in a 32-column contract and teach the model nothing.
    """
    rows = []
    for feature in DIFF_FEATURES:
        if slate is None or feature not in slate.columns:
            rows.append({"feature": feature, "rows": 0, "populated": 0,
                         "coverage": 0.0, "distinct_values": 0, "constant": True})
            continue
        values = pd.to_numeric(slate[feature], errors="coerce")
        populated = int(values.notna().sum())
        total = int(len(values))
        nunique = int(values.dropna().nunique())
        rows.append({
            "feature": feature, "rows": total, "populated": populated,
            "coverage": (populated / total) if total else 0.0,
            "distinct_values": nunique, "constant": nunique <= 1,
        })
    return pd.DataFrame(rows)
