"""Player-level True Shooting ratings, shrunk to a position-segmented prior.

The NBA analogue of MLB's shrunk wOBA and NHL's shrunk player ratings. The rate
itself is arithmetic and exact::

    TS  = points / (2 * (FGA + 0.44 * FTA))

What makes it a RATING rather than a rate is the shrinkage. A raw season TS is
noise-dominated for a player who has taken 30 shots, so each player's
accumulated scoring plays are pulled toward a league prior before use::

    TS_shrunk = (points_prior + TS_league[pos] * k[pos]) / (plays_prior + k[pos])

Two properties of this construction are load-bearing and are asserted by tests
rather than left to convention.

**The prior is POSITION-SEGMENTED.** A centre and a guard do not share an
opportunity, so one league mean would over-rate every big - the same reason
NHL segments by position. ``k`` is itself per position: 20% of the mean
PLAYER-SEASON of scoring plays at that position, carrying MLB's fixed 120-PA
prior convention across by its fraction (120 is 20% of a 600-PA season, so the
fraction is the portable part and the season length is sport-specific).

**The prior is strictly point-in-time.** Both the player's own numerator and
denominator, and the league mean, are summed over rows STRICTLY BEFORE the
target date, per season. A rating that included the target game would be a
leak, and because the target game's contribution is small relative to a season
the leak is invisible in the output - it looks like a slightly better number,
not like a bug.

The averaging unit for ``k`` is a PLAYER-SEASON, not a player, and that is the
correction NHL documents at length: dividing by seasons first weights a 3-game
cameo exactly like an 82-game regular and collapses the reference season, which
would make ``k`` several times too small and under-shrink every rating.

Availability is applied as a separate multiplier on the rating's weight, never
folded into the rate. An injured player still has a true shooting percentage;
what changes is how much the lineup projection should lean on it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    from backend import config
except ImportError:
    import config

#: Columns this module reads out of the player-level season log. The contract
#: spells assists ``ast`` and this does not need assists at all, so the set is
#: the shooting triple plus the keys.
_REQUIRED_COLS = ("player_id", "points", "fga", "fta")


def _player_id_str(values) -> pd.Series:
    """Player ids as clean strings, without a float tail.

    The season log's ``player_id`` arrives as a float column (it shares a dtype
    with rows whose id was never filled), so a bare ``astype(str)`` produces
    ``"1628983.0"``. The position table is keyed on the integer id the feed
    publishes, so those two never join - and the failure is silent, because a
    frame that simply failed to match still produces ratings, just with no
    position and therefore no prior cell. Integral floats are written without
    the tail; anything genuinely non-integral is left alone rather than
    mangled, since no such id exists in this feed.
    """
    series = pd.Series(values)
    out = series.astype(str).str.strip()
    numeric = pd.to_numeric(series, errors="coerce")
    integral = numeric.notna() & (numeric % 1 == 0)
    if integral.any():
        out.loc[integral] = numeric[integral].astype("int64").astype(str)
    return out


def scoring_plays(fga, fta) -> pd.Series:
    """Scoring plays = ``FGA + 0.44 * FTA`` for one player-game.

    The 0.44 is the standard NBA free-throw weight and lives in config so the
    number is changed in exactly one place.
    """
    return (pd.to_numeric(fga, errors="coerce").fillna(0.0)
            + config.PLAYER_TS_FTA_WEIGHT
            * pd.to_numeric(fta, errors="coerce").fillna(0.0))


def true_shooting(points, fga, fta) -> pd.Series:
    """Raw TS for one player-game. NaN where there were no scoring plays.

    A game with zero attempts is genuinely undefined rather than zero, so it is
    NaN and is excluded from both the numerator and the denominator sums. Coding
    it 0.0 would drag every rate the player is part of toward zero.
    """
    plays = scoring_plays(fga, fta)
    pts = pd.to_numeric(points, errors="coerce")
    return pts / (2.0 * plays).where(plays > 0)


def prepare_player_games(stats: pd.DataFrame | None,
                         positions: pd.DataFrame | None = None,
                         ) -> pd.DataFrame:
    """One row per player-game with plays, points, TS and position attached.

    Returns an EMPTY frame with the full column set when the input cannot
    support a rating, rather than None. A caller that gets None has to guess
    whether the pull failed or the league played no games; an empty frame with
    the right columns is a state it can count and report.
    """
    columns = ["player_id", "gameday", "season", "team", "points", "fga",
               "fta", "plays", "ts"]
    if positions is not None and len(positions):
        columns = columns + ["position"]
    empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})
    if stats is None or not len(stats):
        return empty
    if not set(_REQUIRED_COLS).issubset(stats.columns):
        return empty

    frame = stats.copy()
    frame["player_id"] = _player_id_str(frame.player_id).values
    for col in ("points", "fga", "fta"):
        frame[col] = pd.to_numeric(frame.get(col), errors="coerce")
    frame["plays"] = scoring_plays(frame.fga, frame.fta)
    frame["ts"] = true_shooting(frame.points, frame.fga, frame.fta)

    # The season partition is what keeps the prior honest across a season
    # boundary. Without it a rating dated 2024-10-22 would be shrunk toward a
    # league mean that includes games from the previous season - and, worse,
    # toward a mean the player himself contributed to in a different role.
    if "gameday" not in frame.columns:
        return empty
    frame["gameday"] = pd.to_datetime(frame.gameday, errors="coerce")
    frame = frame[frame.gameday.notna()].copy()
    if frame.empty:
        return empty
    # The team is carried because the rating is per-PLAYER but the projected
    # lineup is per-TEAM, and the join between them is this column. A rating
    # frame with no team cannot be pooled at all, and the failure is an empty
    # aggregate rather than an error.
    if "team" not in frame.columns:
        frame["team"] = ""
    else:
        frame["team"] = frame.team.astype(str)
    if "season" not in frame.columns or frame.season.isna().all():
        # An NBA season SPANS two calendar years (2023-24 runs 2023-10-24 to
        # 2024-04-14), so the calendar year of a game is not its season and
        # deriving one here would split every season in half at New Year. The
        # league's own ``SEASON_ID`` is the opaque internal code "22023", so
        # neither of the two columns the log actually carries is usable as-is.
        # ``ingestion.season_label`` is the repo's canonical mapping (the season
        # turns over in July) and is imported lazily because ingestion pulls
        # this module's inputs and a module-level import would be circular.
        from ingestion import season_label
        frame["season"] = [season_label(d.date()) if hasattr(d, "date")
                           else season_label(d) for d in frame.gameday]
    # One row per player-game. The season log can repeat a player-game across
    # overlapping fetch windows, and a duplicate would double both sides of the
    # rate, which looks like a much better performance than it was.
    dedupe = ["player_id", "gameday"] + (["game_id"] if "game_id" in frame.columns
                                         else [])
    frame = frame.drop_duplicates(subset=dedupe, keep="first")
    frame = frame.reset_index(drop=True)

    # The merge above is season-keyed, so this should be a no-op. It is here
    # because the failure it guards against is silent and the invariant is
    # cheap: one row per player per game is what every downstream sum assumes,
    # and a violation inflates the prior, the gates and the league mean at once.
    if positions is not None and len(positions):
        before = len(frame)
        frame = frame.drop_duplicates(subset=["player_id", "gameday"],
                                      keep="first").reset_index(drop=True)
        if before != len(frame):
            raise ValueError(
                f"prepare_player_games produced {before - len(frame)} duplicate "
                "player-game row(s) after the position join; the rating's "
                "prior, its shrinkage weight and the pool gates all assume one "
                "row per player per game")

    if positions is not None and len(positions):
        pos = positions.copy()
        pos["player_id"] = _player_id_str(pos.player_id).values
        # Join on the SEASON as well as the player. Joining on player_id alone
        # matched one game against that player's row in EVERY season file, so a
        # player present in three seasons got three identical rows per game:
        # 73,020 of 79,358 player-games (92%) were duplicated, ``prior_plays``
        # and ``prior_games`` were inflated up to 3x, and the min-plays and
        # rest-plays gates were defeated for most of the league. The rating's
        # RATE is a ratio and so mostly cancelled, which is exactly why this
        # produced plausible numbers instead of an error - and why the
        # shrinkage weight and the pool gates were quietly wrong.
        if "season" in pos.columns and "season" in frame.columns:
            pos = pos.drop_duplicates(subset=["player_id", "season"],
                                      keep="first")
            frame = frame.merge(pos[["player_id", "season", "position"]],
                                on=["player_id", "season"], how="left",
                                suffixes=("", "_resolved"))
        else:
            pos = pos.drop_duplicates(subset=["player_id"], keep="first")
            frame = frame.merge(pos[["player_id", "position"]], on="player_id",
                                how="left", suffixes=("", "_resolved"))
    elif "position" not in frame.columns:
        # Absent position means every player falls out of every prior cell, so
        # the frame is explicitly marked rather than silently missing the
        # column. An EXISTING position column is left alone: overwriting it
        # with NaN would turn a correctly-labelled frame into one that rates
        # nobody, and the only symptom would be an empty result.
        frame["position"] = np.nan
    return frame


def season_plays_table(games: pd.DataFrame,
                       shrink_fraction: float = config.PLAYER_TS_SHRINK_FRACTION,
                       participation_floor_plays: float = 0.0,
                       ) -> dict:
    """``k[position] = shrink_fraction * mean plays per PLAYER-SEASON``.

    The averaging unit is a player-season, not a player: a player appearing in
    two seasons contributes two player-seasons of evidence, each carrying that
    season's own total. Averaging per player instead (total plays / seasons,
    then mean across players) weights a 3-game cameo like a full season and
    collapses the reference season to roughly 40% of its true length, making
    ``k`` several times too small and under-shrinking every rating.

    ``participation_floor_plays`` optionally drops player-seasons below a
    threshold, moving the reference from "a season of any participation" toward
    "a full-time season". It defaults to 0.0 so the table stays comparable with
    the measured reference figures.

    Any cell with no evidence keeps ``config.PLAYER_TS_FALLBACK_K_PLAYS``, so a
    partially populated frame yields a COMPLETE prior table instead of one that
    raises at lookup time.
    """
    table = {pos: float(config.PLAYER_TS_FALLBACK_K_PLAYS)
             for pos in config.PLAYER_TS_POSITIONS}
    if games is None or not len(games):
        return table
    if not {"player_id", "season", "plays", "position"}.issubset(games.columns):
        return table

    work = games[["player_id", "season", "plays", "position"]].copy()
    work = work[work.position.isin(table)]
    if work.empty:
        return table
    per_season = (work.groupby(["player_id", "season", "position"],
                               as_index=False)
                  .agg(plays=("plays", "sum")))
    if participation_floor_plays > 0:
        per_season = per_season[
            per_season.plays >= participation_floor_plays]
    if per_season.empty:
        return table
    means = per_season.groupby("position").plays.mean()
    for position, mean_plays in means.items():
        if mean_plays and math.isfinite(float(mean_plays)):
            table[position] = float(shrink_fraction) * float(mean_plays)
    return table


def _season_of(when) -> str:
    """The season label a timestamp belongs to, via the repo's own mapping."""
    if when is None or (isinstance(when, float) and math.isnan(when)):
        return ""
    stamp = pd.Timestamp(when)
    if pd.isna(stamp):
        return ""
    try:
        from ingestion import season_label
    except ImportError:  # pragma: no cover - ingestion is always importable
        first = stamp.year if stamp.month >= 7 else stamp.year - 1
        return f"{first}-{str(first + 1)[2:]}"
    return season_label(stamp.date())


def _season_evidence_index(games: pd.DataFrame) -> dict:
    """``season -> sorted gamedays``, so "has this season any row yet?" is a
    ``searchsorted`` rather than a scan of the whole frame.

    Built once per call to ``build_player_ts`` because the answer is asked once
    per target date per position, and the whole-frame build asks it thousands of
    times. A filter per question turns a 40-second build into minutes.
    """
    index: dict = {}
    if games is None or not len(games) or "season" not in games.columns:
        return index
    for season, part in games.groupby("season", sort=False):
        labels = part.season.dropna()
        if not len(labels):
            continue
        days = pd.to_datetime(part.gameday, errors="coerce").dropna()
        if not len(days):
            continue
        index[str(labels.iloc[0])] = np.sort(days.to_numpy())
    return index


def _evidence_season(games: pd.DataFrame, target,
                     index: dict | None = None) -> str:
    """The season a target's rating is allowed to draw evidence from.

    A target rates against its OWN season, and only its own season: that
    partition is what stops a rating dated 2024-11-02 from being shrunk toward
    a league mean the player contributed to in a different role, and
    ``test_a_rating_does_not_carry_the_previous_season`` holds that line.

    But the own season is not always the season that HAS the evidence. On the
    first day of a season the target's season has no rows before it, and the
    strict partition then returns nothing at all - not a thin rating, no
    rating. That is the worst possible time for it: the games with no result,
    which are the only ones this file ever rates for, ARE the first games of a
    season, so the partition blanked the ratings artifact and the slate's
    lineup features on exactly the days they are needed and carried them
    silently for the other 200.

    So the own-season rule stays primary and applies the moment the season has
    ANY row at or before the target; only when it has none does the rating fall
    back to the most recent season that does. Every row in that fallback is
    still strictly before the target, so the point-in-time floor is untouched:
    this widens WHICH SEASON the evidence comes from, never WHEN it may come
    from. A player who never appears in the last completed season has no row
    either way, which is the honest answer rather than a borrowed one.
    """
    own = _season_of(target)
    if index is None:
        index = _season_evidence_index(games)
    days = index.get(own)
    if days is not None and len(days) and bool((days <= target).any()):
        return own
    best_label, best_day = None, None
    for season, season_days in index.items():
        prior = season_days[season_days <= target]
        if not len(prior):
            continue
        latest = prior.max()
        if best_day is None or latest > best_day:
            best_label, best_day = season, latest
    return best_label if best_label is not None else own


def _prior_for(games: pd.DataFrame, target, season: str | None = None
               ) -> pd.DataFrame:
    """Per-player totals over rows STRICTLY BEFORE ``target``.

    The prior is a plain filtered sum rather than a running total read at the
    last row. The two are not the same thing, and the difference bites on a
    player's SECOND game: a prefix-sum-minus-self column excludes each row's
    own line, so the last row before the target reports everything before
    ITSELF - which omits exactly the game the target is being rated for. Read
    that way a player is rated as though they had not yet played the game
    immediately preceding the one being projected, and a player's first game
    reports a prior of zero no matter how much they scored in it.

    Summing the rows that precede the target is the definition of the prior, so
    that is what happens here. The target row itself is excluded by the strict
    ``<``, which is the entire leakage guarantee.
    """
    if "season" in games.columns:
        if season is None:
            season = _evidence_season(games, target)
        if season:
            games = games[games.season == season]
    earlier = games[games.gameday < target]
    if earlier.empty:
        return pd.DataFrame(columns=["player_id", "position", "season",
                                     "prior_points", "prior_plays",
                                     "prior_games"])
    keys = ["player_id", "position"] + (["season"] if "season" in earlier.columns
                                       else [])
    # ``size`` not ``sum``: prior_games counts the games that contributed
    # evidence, and a sum of a count column would be its square.
    return (earlier.groupby(keys, as_index=False)
            .agg(prior_points=("points", "sum"),
                 prior_plays=("plays", "sum"),
                 prior_games=("points", "size")))


def league_prior_table(games: pd.DataFrame, target_dates=None) -> pd.DataFrame:
    """Position-segmented league TS through each target date, PIT-safe.

    One row per (target date, position). The numerator and denominator are
    accumulated from player-games strictly before the date, which is what makes
    the prior usable at bet time rather than only retrospectively.

    Position cells are independent: each is its own sum, so a date where only
    guards have played still yields a guard row. The alternative - a single
    league-wide sum sliced by position - is the same arithmetic, but doing it
    per cell keeps a missing cell missing instead of silently borrowing another
    cell's evidence.
    """
    columns = ["target_date", "position", "lg_points", "lg_plays", "lg_ts"]
    if games is None or not len(games):
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})
    if "position" not in games.columns:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})

    work = games[games.position.isin(config.PLAYER_TS_POSITIONS)].copy()
    if work.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})
    work = work.sort_values(["position", "season", "gameday"])

    if target_dates is None:
        dates = pd.Series(sorted(work.gameday.dropna().unique()))
    else:
        dates = pd.Series(pd.to_datetime(pd.Series(target_dates),
                                         errors="coerce").dropna().unique())
    dates = dates.sort_values().reset_index(drop=True)
    if dates.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})

    rows = []
    season_index = _season_evidence_index(work)
    for position, group in work.groupby("position", sort=False):
        # The league mean is also season-partitioned, for the same reason the
        # player's own prior is: a rating dated into the current season is
        # shrunk toward the league as it played THAT season. A target is only
        # ever matched against the part for ITS OWN season - summing every
        # part whose dates precede the target would quietly reintroduce the
        # carry-over this partition exists to prevent.
        if "season" in group.columns:
            group = group.assign(
                _season=[_season_of(d) for d in group.gameday])
            parts = list(group.groupby("_season", sort=False))
        else:
            parts = [(None, group)]
        for season, part in parts:
            daily_points = part.groupby("gameday").points.sum().sort_index()
            daily_plays = part.groupby("gameday").plays.sum().sort_index()
            cum_points = daily_points.cumsum()
            cum_plays = daily_plays.cumsum()
            for target in dates:
                if season is not None and _evidence_season(
                        work, target, season_index) != season:
                    continue
                earlier = cum_points.index < target
                rows.append((target, position,
                             float(cum_points[earlier].sum())
                             if earlier.any() else 0.0,
                             float(cum_plays[earlier].sum())
                             if earlier.any() else 0.0))
    out = pd.DataFrame(rows, columns=["target_date", "position",
                                      "lg_points", "lg_plays"])
    out["lg_ts"] = out.lg_points / (2.0 * out.lg_plays).where(out.lg_plays > 0)
    return out


def shrunk_ts(prior_points, prior_plays, league_ts, k) -> pd.Series:
    """The shrinkage itself, in TS units.

    TS carries a factor of TWO in its denominator - one scoring play is worth
    two points - so the prior has to be carried in the same units or the answer
    comes out at roughly double the truth::

        TS = (prior_points + 2 * lg_ts * k) / (2 * (prior_plays + k))

    The prior contributes ``2 * lg_ts * k`` points because ``k`` is expressed
    in SCORING PLAYS and the observed points are compared against a denominator
    of ``2 * plays``. Dropping either factor of two does not raise an error, it
    returns a plausible number near 0.93 where the truth is near 0.56 - which
    is why this is one named function instead of inlined arithmetic at the
    call site.

    ``k`` is per-position, so ``league_ts`` and ``k`` are both series here.
    """
    pts = pd.to_numeric(pd.Series(prior_points), errors="coerce")
    plays = pd.to_numeric(pd.Series(prior_plays), errors="coerce")
    lg = pd.to_numeric(pd.Series(league_ts), errors="coerce")
    weight = pd.to_numeric(pd.Series(k), errors="coerce")
    return (pts + 2.0 * lg * weight) / (2.0 * (plays + weight))


def build_player_ts(games: pd.DataFrame,
                    target_dates=None,
                    shrink_fraction: float = config.PLAYER_TS_SHRINK_FRACTION,
                    availability: pd.DataFrame | None = None,
                    ) -> pd.DataFrame:
    """Player-level TS ratings as of each target date.

    Returns one row per (target date, player) with the raw prior, the league
    prior it was shrunk toward, the strength used, and the resulting rating.

    A player with no prior evidence at all is still emitted, with a NaN rating
    and a zero prior, rather than dropped. The distinction matters for the
    caller: "this player has no rating" and "this player was never in the data"
    are different facts, and a projection that cannot distinguish them will
    happily project a player who does not exist.
    """
    # ``availability`` and ``availability_multiplier`` were removed rather than
    # left behind. Once the design moved to a stint table and a removal-based
    # pool filter, they emitted a constant "healthy" and 1.0 for every row -
    # a column that looks populated and carries no information, and a
    # multiplier that invites a future caller to reintroduce exactly the
    # zero-weighting the pool deliberately avoids.
    columns = ["target_date", "player_id", "position", "team", "prior_points",
               "prior_plays", "prior_games", "lg_ts", "k_plays", "ts_raw",
               "ts_shrunk", "days_since_appearance"]
    empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})
    if games is None or not len(games):
        return empty
    if "position" not in games.columns or not games.position.notna().any():
        return empty

    work = games[games.position.isin(config.PLAYER_TS_POSITIONS)].copy()
    if work.empty:
        return empty

    k_table = season_plays_table(games, shrink_fraction=shrink_fraction)

    if target_dates is None:
        dates = pd.Series(sorted(work.gameday.dropna().unique()))
    else:
        dates = pd.Series(pd.to_datetime(pd.Series(target_dates),
                                         errors="coerce").dropna().unique())
    dates = dates.sort_values().reset_index(drop=True)
    if dates.empty:
        return empty

    league = league_prior_table(games, dates)
    season_index = _season_evidence_index(work)
    rows = []
    for target in dates:
        # The target rates against its own season, and only its own season, so
        # a player who appears on both sides of a boundary is rated against the
        # season they are actually in rather than a blend of two. The one
        # exception is a target whose own season has no evidence before it -
        # the first day of a season - which falls back to the last completed
        # one, still strictly before the target. See _evidence_season.
        season = _evidence_season(work, target, season_index)
        known = work[work.gameday <= target]
        if "season" in known.columns and season:
            known = known[known.season == season]
        if known.empty:
            continue
        snapshot = _prior_for(work, target, season=season)
        # A player who has already appeared on or before the target date is
        # rated even with no prior evidence, so the row exists with a zero
        # prior. The distinction matters to a caller: "this player has no
        # rating yet" and "this player was never in the data" are different
        # facts, and a projection that cannot tell them apart will happily
        # project a player who does not exist.
        roster = known[["player_id", "position"]].drop_duplicates(
            subset=["player_id", "position"])
        if "team" in known.columns:
            # The team as of the most recent row the player appears in, which
            # is the club he would be projected for. A mid-season trade
            # therefore moves him, because the later row wins.
            latest_team = (known.sort_values("gameday")
                           .groupby("player_id").tail(1)[["player_id", "team"]])
            roster = roster.merge(latest_team, on="player_id", how="left")
        if snapshot.empty:
            snapshot = roster.copy()
            for col in ("prior_points", "prior_plays", "prior_games"):
                snapshot[col] = 0.0
        else:
            keys = ["player_id", "position"]
            snapshot = roster.merge(snapshot, on=keys, how="left")
            for col in ("prior_points", "prior_plays", "prior_games"):
                snapshot[col] = snapshot[col].fillna(0.0)
        lg = league[league.target_date == target]
        snapshot = snapshot.merge(
            lg[["position", "lg_ts"]], on="position", how="left")
        snapshot["k_plays"] = snapshot.position.map(k_table).astype(float)
        # The unshrunk rate, for auditing. Recomputed from the snapshot's own
        # prior rather than carried down from the per-row frame, because the
        # per-row value belongs to a single game, not to the accumulated prior.
        snapshot["ts_raw"] = snapshot.prior_points / (
            2.0 * snapshot.prior_plays).where(snapshot.prior_plays > 0)
        # A player at a position with no league evidence falls back to the
        # position's OWN reference rather than to nothing: the prior strength
        # is defined for every position, so a NaN league mean leaves the
        # rating undefined instead of collapsing it to the player's raw rate.
        snapshot["ts_shrunk"] = shrunk_ts(
            snapshot.prior_points, snapshot.prior_plays,
            snapshot.lg_ts, snapshot.k_plays)
        snapshot["target_date"] = target
        # Recency of the player's actual evidence, carried for the pool's
        # availability gate: days from the player's LAST appearance strictly
        # at or before the target, within the rated season. The gap is only
        # defined when the evidence season IS the target's own season; a
        # fallback-evidence target (the first days of a season, rated from
        # the last completed one) carries NaN — that is the season-start
        # carryover case, governed by the min-plays floor, and NOT an
        # infinitely stale row. (Computing the gap across the fallback made
        # every carryover member look ~150 days stale on the 2026-27 slate
        # and the gate emptied every projected lineup; the frame's OOF side
        # was unaffected and mid-season games never hit this path.)
        if season and season == _season_of(target):
            known_early = known[known.gameday <= target]
            if len(known_early):
                last_seen = (known_early.groupby("player_id").gameday.max()
                             .rename("_last_seen"))
                snapshot = snapshot.merge(last_seen, on="player_id", how="left")
                snapshot["days_since_appearance"] = (
                    (pd.Timestamp(target)
                     - snapshot["_last_seen"]).dt.days)
                snapshot = snapshot.drop(columns=["_last_seen"])
            else:
                snapshot["days_since_appearance"] = np.nan
        else:
            snapshot["days_since_appearance"] = np.nan
        rows.append(snapshot)
    if not rows:
        return empty
    out = pd.concat(rows, ignore_index=True)
    out["player_id"] = _player_id_str(out.player_id).values

    if availability is not None and len(availability):
        # Carried through for callers that want the raw roster label, but it
        # is NOT what the pool filter reads - that goes through
        # injury_stints.out_at, which is point-in-time.
        avail = availability.copy()
        avail["player_id"] = _player_id_str(avail.player_id).values
        keep = [c for c in ("player_id", "status") if c in avail.columns]
        if keep:
            out = out.merge(avail[keep].drop_duplicates(subset=["player_id"]),
                            on="player_id", how="left")
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[columns].sort_values(
        ["target_date", "player_id"]).reset_index(drop=True)
