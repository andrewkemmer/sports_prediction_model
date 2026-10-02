"""Player-level Estimated Plus-Minus ratings, shrunk to a position prior.

The NBA analogue of NFL's EPA-per-target and MLB's shrunk wOBA: a
player's impact expressed per 100 possessions he participated in.
The rate itself is arithmetic and exact::

    EPM  = 100 * plus_minus / participated_possessions

where a player's participated possessions are his share of his
team's game possessions - ``minutes / 48`` of them, the standard
pro-rata estimate - and a team's game possessions are Dean
Oliver's ``FGA + 0.44 * FTA + TOV`` summed over its box scores.

What makes it a RATING rather than a rate is the shrinkage. A raw
season EPM is noise-dominated for a player with 200 possessions of
evidence, so each player's accumulated plus-minus is pulled toward
a position-segmented league prior before use::

    EPM_shrunk = (100 * pm_prior + EPM_league[pos] * k[pos])
                 / (poss_prior + k[pos])

Two properties of this construction are load-bearing and are asserted
by tests rather than left to convention.

**The prior is POSITION-SEGMENTED.** A centre and a guard do not
share an opportunity, so one league mean would over-rate every big -
the same reason NHL segments by position. ``k`` is itself per
position: 20% of the mean PLAYER-SEASON of participated possessions
at that position, carrying MLB's fixed 120-PA prior convention
across by its fraction (120 is 20% of a 600-PA season, so the
fraction is the portable part and the season length is sport-specific).

**The prior is strictly point-in-time.** Both the player's own
numerator and denominator, and the league mean, are summed over rows
STRICTLY BEFORE the target date, per season. A rating that included
the target game would be a leak, and because the target game's
contribution is small relative to a season the leak is invisible in
the output - it looks like a slightly better number, not like a bug.

The averaging unit for ``k`` is a PLAYER-SEASON, not a player, and
that is the correction NHL documents at length: dividing by seasons
first weights a 3-game cameo exactly like an 82-game regular and
collapses the reference season, which would make ``k`` several times
too small and under-shrink every rating.

A player who did not play has no plus-minus to rate. The feed can
carry a stale nonzero PM on a DNP row (120 such rows exist in the
2023-24..2025-26 logs), so participation is gated on MINUTES, not
on the PM column: a zero-minute row contributes nothing to either
side of the rate. Coding it as an observation would drag every
rating the player is part of toward a number he never earned on the
court.

Availability is applied as a separate removal at pool construction,
never folded into the rate. An injured player still has an EPM;
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

#: Columns this module reads out of the player-level season log. The
#: rating needs the box-score triple that estimates possessions
#: (attempts, free throws, turnovers), the participation measure
#: (minutes) and the impact measure (plus-minus), plus the keys.
_REQUIRED_COLS = ("player_id", "plus_minus", "fga", "fta", "tov",
                  "minutes")


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


def team_game_possessions(frame: pd.DataFrame) -> pd.Series:
    """Possessions per (game, team): ``FGA + 0.44*FTA + TOV``.

    Dean Oliver's possession estimate, summed over the team's box
    scores in one game. A team plays at most one game a day, so
    ``(gameday, team)`` keys a game exactly - and it is always
    present, where ``game_id`` is not (unit fixtures, hand-built
    frames). Computed on a DEDUPED frame: a box score counted twice
    would double the game's possessions and halve every player's
    participation in it.
    """
    fga = pd.to_numeric(frame.get("fga"), errors="coerce").fillna(0.0)
    fta = pd.to_numeric(frame.get("fta"), errors="coerce").fillna(0.0)
    tov = pd.to_numeric(frame.get("tov"), errors="coerce").fillna(0.0)
    poss = fga + config.PLAYER_EPM_FTA_WEIGHT * fta + tov
    keys = ["gameday", "team"]
    work = frame[keys].copy()
    work["_poss"] = poss
    totals = work.groupby(keys, dropna=False).agg(
        _game_poss=("_poss", "sum"))
    merged = frame[keys].merge(totals, on=keys, how="left")
    return merged["_game_poss"].to_numpy()


def participated_possessions(game_possessions, minutes) -> pd.Series:
    """Possessions a player participated in, for one player-game.

    ``game_possessions * minutes / 48``: the pro-rata share of his
    team's possessions that ran while he was on the floor, the same
    estimate every per-100-possessions rate in basketball uses. A
    player who did not play participated in nothing, so the share is
    exactly zero when ``minutes`` is missing or zero - not NaN,
    because a DNP row is a defined observation of zero opportunity,
    and the rate it feeds must be undefined (see
    :func:`estimated_plus_minus`), not missing.
    """
    # ``game_possessions`` arrives as a numpy array (aligned to the
    # frame's rows) from :func:`team_game_possessions`, so it is
    # wrapped before the numeric coercion - an ndarray has no
    # ``fillna``.
    poss = pd.to_numeric(
        pd.Series(game_possessions), errors="coerce").fillna(0.0)
    mins = pd.to_numeric(pd.Series(minutes), errors="coerce").fillna(0.0)
    return (poss * mins / 48.0).where(mins > 0, 0.0)


def estimated_plus_minus(plus_minus, plays) -> pd.Series:
    """Raw EPM for one player-game: ``100 * PM / participated``.

    NaN where there were no participated possessions. A game with
    zero participation is genuinely undefined rather than zero, so it
    is NaN and is excluded from both the numerator and the denominator
    sums. Coding it 0.0 would drag every rating the player is part of
    toward zero.
    """
    pm = pd.to_numeric(plus_minus, errors="coerce")
    plays = pd.to_numeric(plays, errors="coerce")
    return (100.0 * pm / plays).where(plays > 0)


def prepare_player_games(stats: pd.DataFrame | None,
                         positions: pd.DataFrame | None = None,
                         ) -> pd.DataFrame:
    """One row per player-game with possessions, PM, EPM and position.

    Returns an EMPTY frame with the full column set when the input cannot
    support a rating, rather than None. A caller that gets None has to guess
    whether the pull failed or the league played no games; an empty frame with
    the right columns is a state it can count and report.
    """
    columns = ["player_id", "gameday", "season", "team", "plus_minus",
               "minutes", "fga", "fta", "tov", "plays", "epm"]
    if positions is not None and len(positions):
        columns = columns + ["position"]
    empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})
    if stats is None or not len(stats):
        return empty
    if not set(_REQUIRED_COLS).issubset(stats.columns):
        return empty

    frame = stats.copy()
    frame["player_id"] = _player_id_str(frame.player_id).values
    for col in ("plus_minus", "fga", "fta", "tov", "minutes"):
        frame[col] = pd.to_numeric(frame.get(col), errors="coerce")

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

    # Possessions, participation and the rate itself are computed LAST, on the
    # deduped frame: the game's possession total is a sum over the team's box
    # scores, so it is only correct once each box score is counted once.
    frame["plays"] = participated_possessions(
        team_game_possessions(frame), frame.minutes)
    # A player who did not participate has no plus-minus to rate. The feed can
    # carry a stale nonzero PM on a DNP row (120 such rows in the 2023-24
    # through 2025-26 logs, against 305 zero-minute rows), and those points
    # were never earned on the court - leaving them in the numerator would
    # rate a player for impact he did not have while paying him for the
    # possessions in the denominator. Zeroed, the row contributes nothing to
    # either side of the rate, exactly like a zero-attempt game did under the
    # old scoring-play denominator.
    frame["plus_minus"] = frame.plus_minus.where(frame.plays > 0, 0.0)
    frame["epm"] = estimated_plus_minus(frame.plus_minus, frame.plays)
    return frame


def season_plays_table(games: pd.DataFrame,
                       shrink_fraction: float = config.PLAYER_EPM_SHRINK_FRACTION,
                       participation_floor_plays: float = 0.0,
                       through_season: str | None = None,
                       ) -> dict:
    """``k[position] = shrink_fraction * mean possessions per PLAYER-SEASON``.

    The averaging unit is a player-season, not a player: a player appearing in
    two seasons contributes two player-seasons of evidence, each carrying that
    season's own total. Averaging per player instead (total possessions /
    seasons, then mean across players) weights a 3-game cameo like a full
    season and collapses the reference season to roughly 40% of its true
    length, making ``k`` several times too small and under-shrinking every
    rating.

    ``participation_floor_plays`` optionally drops player-seasons below a
    threshold, moving the reference from "a season of any participation" toward
    "a full-time season". It defaults to 0.0 so the table stays comparable with
    the measured reference figures.

    ``through_season`` applies the NHL discipline (``season_ice_time_table``'s
    ``as_of``): only player-seasons from seasons STRICTLY BEFORE it enter the
    mean, so a season cannot tune its own shrinkage strength. Without it the
    whole-frame mean also absorbs the season in progress - and an in-progress
    season holds PARTIAL player-seasons, which drags the mean down as the
    season accumulates and silently weakens every rating's prior weight game by
    game. ``None`` keeps the whole-frame mean (the pre-2026-10-01 behavior).

    Any cell with no evidence keeps ``config.PLAYER_EPM_FALLBACK_K_PLAYS``, so a
    partially populated frame yields a COMPLETE prior table instead of one that
    raises at lookup time.
    """
    table = {pos: float(config.PLAYER_EPM_FALLBACK_K_PLAYS)
             for pos in config.PLAYER_EPM_POSITIONS}
    if games is None or not len(games):
        return table
    if not {"player_id", "season", "plays", "position"}.issubset(games.columns):
        return table

    work = games[["player_id", "season", "plays", "position"]].copy()
    work = work[work.position.isin(table)]
    if through_season is not None:
        cutoff = _season_key(through_season)
        work = work[work["season"].map(_season_key) < cutoff]
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


def _season_key(label) -> tuple:
    """Sort key for season labels: the calendar year the season opens.

    Labels are ``"2024-25"``-shaped, so the leading year orders them. A
    label that does not parse still sorts - deterministically, after every
    real season - rather than raising mid-build.
    """
    text = str(label)
    try:
        return (0, int(text[:4]))
    except ValueError:
        return (1, text)


def _k_table_for_season(k_by_season: dict, full_frame_table: dict,
                        season: str) -> dict:
    """The shrinkage-strength table for a target in ``season``.

    Exact season first; else the most recent table built from an earlier
    season (a target in a season the frame has no rows for - a preseason
    date, say - still rates on a completed-season scale); else the
    whole-frame table, the only scale the frame knows.
    """
    if season in k_by_season:
        return k_by_season[season]
    key = _season_key(season)
    earlier = [label for label in k_by_season
               if _season_key(label) < key]
    if earlier:
        return k_by_season[max(earlier, key=_season_key)]
    return full_frame_table


def _season_evidence_index(games: pd.DataFrame) -> dict:
    """``season -> sorted gamedays``, so "has this season any row yet?" is a
    ``searchsorted`` rather than a scan of the whole frame.

    Built once per call to ``build_player_epm`` because the answer is asked once
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
                                     "prior_pm", "prior_plays",
                                     "prior_games"])
    keys = ["player_id", "position"] + (["season"] if "season" in earlier.columns
                                       else [])
    # ``size`` not ``sum``: prior_games counts the games that contributed
    # evidence, and a sum of a count column would be its square.
    return (earlier.groupby(keys, as_index=False)
            .agg(prior_pm=("plus_minus", "sum"),
                 prior_plays=("plays", "sum"),
                 prior_games=("plus_minus", "size")))


def league_prior_table(games: pd.DataFrame, target_dates=None) -> pd.DataFrame:
    """Position-segmented league EPM through each target date, PIT-safe.

    One row per (target date, position). The numerator (plus-minus) and the
    denominator (participated possessions) are accumulated from player-games
    strictly before the date, which is what makes the prior usable at bet time
    rather than only retrospectively. The league cell's rate is carried in
    per-100-possession units, ``100 * lg_pm / lg_plays`` - the same scale the
    player rate lives on, so the shrinkage mixes the two without a unit
    conversion.

    Position cells are independent: each is its own sum, so a date where only
    guards have played still yields a guard row. The alternative - a single
    league-wide sum sliced by position - is the same arithmetic, but doing it
    per cell keeps a missing cell missing instead of silently borrowing another
    cell's evidence.

    **Season boundary (2026-09-29 audit).** The cell follows the same evidence
    season the player priors do. When that season has NO plays strictly before
    the target - the season's first decided game(s) - the cell borrows the most
    recent prior season's plays through the target and ``lg_season_source``
    names the borrowed season. Without it, opening night has a zero player
    prior AND an empty league cell, so the shrink target is NaN and a whole
    slate's epm_shrunk collapses - all-the-way shrinkage with no prior to shrink
    TO. The borrow is strictly point-in-time (prior-season rows are, by
    construction, before the target) and it is the same bridge the player-side
    fallback already makes for pre-season targets, extended to the league side
    so game 1 behaves like game 2: the player lands on the league mean, just
    from last season's cell. A target with no evidence in ANY season keeps the
    empty cell (lg_epm NaN) - nothing is invented.
    """
    columns = ["target_date", "position", "lg_pm", "lg_plays", "lg_epm",
               "lg_season_source"]
    if games is None or not len(games):
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})
    if "position" not in games.columns:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})

    work = games[games.position.isin(config.PLAYER_EPM_POSITIONS)].copy()
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
    # Per (position, season) cumulative possession tables, kept so the
    # season-boundary borrow below can read a prior season's cell.
    cum_by_season: dict = {}
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
            daily_pm = part.groupby("gameday").plus_minus.sum().sort_index()
            daily_plays = part.groupby("gameday").plays.sum().sort_index()
            cum_pm = daily_pm.cumsum()
            cum_plays = daily_plays.cumsum()
            if season is not None:
                cum_by_season.setdefault(position, {})[season] = (
                    cum_pm, cum_plays)
            for target in dates:
                if season is not None and _evidence_season(
                        work, target, season_index) != season:
                    continue
                earlier = cum_pm.index < target
                rows.append((target, position,
                             float(cum_pm[earlier].sum())
                             if earlier.any() else 0.0,
                             float(cum_plays[earlier].sum())
                             if earlier.any() else 0.0,
                             season if (season is not None and earlier.any())
                             else ""))
    out = pd.DataFrame(rows, columns=["target_date", "position",
                                      "lg_pm", "lg_plays",
                                      "lg_season_source"])
    # Season-boundary borrow. A cell with no in-season evidence strictly
    # before the target - the season's first decided game(s) - shrinks toward
    # the most recent season that HAS such evidence, so a zero-prior player on
    # opening night lands EXACTLY on the prior season's position league mean:
    # all-the-way shrinkage, the same behavior game 2 already gets. Strictly
    # point-in-time (prior-season rows are, by construction, before the
    # target); a target with no evidence in ANY season keeps the empty cell.
    for idx in out.index[out.lg_plays <= 0]:
        season_cells = cum_by_season.get(out.at[idx, "position"], {})
        target = out.at[idx, "target_date"]
        best_season, best_day = None, None
        for season, (_cp, _cpl) in season_cells.items():
            prior_days = _cpl.index[_cpl.index < target]
            if not len(prior_days):
                continue
            latest = prior_days.max()
            if best_day is None or latest > best_day:
                best_season, best_day = season, latest
        if best_season is None:
            continue
        cp, cpl = season_cells[best_season]
        earlier = cp.index < target
        out.at[idx, "lg_pm"] = float(cp[earlier].sum())
        out.at[idx, "lg_plays"] = float(cpl[earlier].sum())
        out.at[idx, "lg_season_source"] = best_season
    out["lg_epm"] = (100.0 * out.lg_pm
                     / out.lg_plays.where(out.lg_plays > 0))
    return out


def shrunk_epm(prior_pm, prior_plays, league_epm, k) -> pd.Series:
    """The shrinkage itself, in EPM units.

    EPM's numerator is ``100 * plus_minus`` and its denominator is
    participated possessions, so the prior is carried in those units and the
    algebra is the generic one - prior plus k pseudo-observations of the league
    mean::

        EPM = (100 * prior_pm + league_epm * k) / (prior_plays + k)

    A zero-prior player lands exactly on ``league_epm``: with no evidence the
    rating IS the prior. Dropping the ``100`` does not raise an error, it
    returns a plausible number near 0.01 where the truth is near 1.4 - which
    is why this is one named function instead of inlined arithmetic at the
    call site.

    ``k`` is per-position, so ``league_epm`` and ``k`` are both series here.
    """
    pm = pd.to_numeric(pd.Series(prior_pm), errors="coerce")
    plays = pd.to_numeric(pd.Series(prior_plays), errors="coerce")
    lg = pd.to_numeric(pd.Series(league_epm), errors="coerce")
    weight = pd.to_numeric(pd.Series(k), errors="coerce")
    return (100.0 * pm + lg * weight) / (plays + weight)


def build_player_epm(games: pd.DataFrame,
                     target_dates=None,
                     shrink_fraction: float = config.PLAYER_EPM_SHRINK_FRACTION,
                     availability: pd.DataFrame | None = None,
                     ) -> pd.DataFrame:
    """Player-level EPM ratings as of each target date.

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
    columns = ["target_date", "player_id", "position", "team", "prior_pm",
               "prior_plays", "prior_games", "lg_epm", "k_plays", "epm_raw",
               "epm_shrunk", "days_since_appearance"]
    empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})
    if games is None or not len(games):
        return empty
    if "position" not in games.columns or not games.position.notna().any():
        return empty

    work = games[games.position.isin(config.PLAYER_EPM_POSITIONS)].copy()
    if work.empty:
        return empty

    # Shrinkage strength is per TARGET SEASON, derived from completed prior
    # seasons only (NHL's season_ice_time_table discipline): a season cannot
    # tune its own k, and an in-progress season - whose player-seasons are
    # partial - would otherwise drag the mean down as the season accumulates
    # and silently weaken every rating's prior weight game by game. The
    # whole-frame table stays as the fallback for a target whose season
    # predates every season the frame knows, where no prior-season scale
    # exists at all. A frame of completed seasons only (the opening-slate
    # build) is numerically identical to the old single global table.
    k_table = season_plays_table(games, shrink_fraction=shrink_fraction)
    k_by_season: dict = {}
    if "season" in games.columns:
        frame_seasons = sorted(
            {s for s in games["season"].dropna().unique()},
            key=_season_key)
        for index, season in enumerate(frame_seasons):
            if index == 0:
                # The frame's earliest season has no completed prior
                # season to measure against, so it keeps the
                # whole-frame mean - the pre-2026-10-01 behavior -
                # rather than a constant that ignores the frame's own
                # season scale.
                k_by_season[season] = k_table
            else:
                k_by_season[season] = season_plays_table(
                    games, shrink_fraction=shrink_fraction,
                    through_season=season)

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
            for col in ("prior_pm", "prior_plays", "prior_games"):
                snapshot[col] = 0.0
        else:
            keys = ["player_id", "position"]
            snapshot = roster.merge(snapshot, on=keys, how="left")
            for col in ("prior_pm", "prior_plays", "prior_games"):
                snapshot[col] = snapshot[col].fillna(0.0)
        lg = league[league.target_date == target]
        snapshot = snapshot.merge(
            lg[["position", "lg_epm"]], on="position", how="left")
        k_for_target = _k_table_for_season(
            k_by_season, k_table, _season_of(target))
        snapshot["k_plays"] = snapshot.position.map(
            k_for_target).astype(float)
        # The unshrunk rate, for auditing. Recomputed from the snapshot's own
        # prior rather than carried down from the per-row frame, because the
        # per-row value belongs to a single game, not to the accumulated prior.
        snapshot["epm_raw"] = (100.0 * snapshot.prior_pm
                               / snapshot.prior_plays).where(
                                   snapshot.prior_plays > 0)
        # A player at a position with no league evidence falls back to the
        # position's OWN reference rather than to nothing: the prior strength
        # is defined for every position, so a NaN league mean leaves the
        # rating undefined instead of collapsing it to the player's raw rate.
        snapshot["epm_shrunk"] = shrunk_epm(
            snapshot.prior_pm, snapshot.prior_plays,
            snapshot.lg_epm, snapshot.k_plays)
        snapshot["target_date"] = target
        # Recency of the player's actual evidence, carried for the pool's
        # availability gate: days from the player's LAST appearance strictly
        # at or before the target, within the rated season. The gap is only
        # defined when the evidence season IS the target's own season; a
        # fallback-evidence target (the first days of a season, rated from
        # the prior one) carries NaN — that is the season-start carryover case,
        # governed by the min-plays floor, and NOT an infinitely stale row.
        # (Computing the gap across the fallback made every carryover member
        # look ~150 days stale on the 2026-27 slate and the gate emptied
        # every projected lineup; the frame's OOF side was unaffected and
        # mid-season games never hit this path.)
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
