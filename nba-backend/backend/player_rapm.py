"""Player-level Regularized Adjusted Plus-Minus (RAPM), per game.

The NBA analogue of NFL's EPA-per-target and MLB's shrunk wOBA: a
player's impact expressed as an adjusted plus-minus PER GAME. The
rating is not arithmetic but REGRESSED::

    y_g = sum over players on the floor of share_i * beta_i + noise

where each team-game is one row, ``y`` is that team's point margin
(``net_points`` from the team table, box-sum fallback), and
``share = MIN / 48`` is the player's participation in the game's
minutes. Solving the ridge ``(X'X + lambda I) beta = X'y`` over
all players simultaneously is what makes the rating ADJUSTED: every
teammate's contribution is claimed by that teammate's own column,
so a player's beta is conditional on game-level teammate/opponent minutes.
This is a GAME-LEVEL ridge proxy, not possession/stint RAPM: it cannot
identify who shared the floor or separate offensive and defensive impact.
Each team's MIN/48 shares sum to about FIVE (more in overtime), not one.
Beta is a full-48-minute coefficient in the game-margin equation, not a
player's observed points per game or conventional points per 100 possessions.

The evidence window is the target's own season - with the most
recent season that has a game before the target as the fallback
while the own season has none yet (opening night rates against
last season's fit instead of nothing) - and every game in it is
STRICTLY BEFORE the target date - point-in-time, like the EPM it
replaces.

The ridge is one regularizer; a second, smaller layer makes the
rating a SEGMENTED one. A player whose window evidence is thin
(a cameo, a trade-deadline acquisition) gets his beta pulled
toward a position-segmented league prior - the mean beta of his
position's members in the same fit::

    RAPM_shrunk = (eff_games * beta + RAPM_league[pos] * k[pos])
                  / (eff_games + k[pos])

where ``eff_games = sum of share**2`` is the player's
participation-weighted game count in the window and ``k`` is a
fraction of the mean player-season eff-games at that position.
The ridge already pulls every player toward the league mean; this
layer is what keeps a thin player's rating POSITION-honest, and
it is what a rostered player with no window evidence falls back
to (his rating IS the prior).

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
never folded into the rate. An injured player still has a RAPM;
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
_REQUIRED_COLS = ("player_id", "minutes")


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


def participation_share(minutes) -> pd.Series:
    """A player's share of his team's game minutes: ``MIN / 48``.

    The design-matrix entry of the RAPM fit: a starter who plays 40
    of 48 minutes carries a 0.833 in his row, a DNP exactly 0.0.
    A regulation team's shares sum to about five, one per on-court slot.
    This is a per-48-minute exposure, not a fraction of total team minutes;
    overtime can increase the team sum. It is not possession-normalized.
    """
    mins = pd.to_numeric(pd.Series(minutes), errors="coerce").fillna(0.0)
    # Full-48-minute exposure is linear even in overtime. Clipping 50+
    # minutes to one undercounts the exposure in X and in eff-games.
    return (mins / 48.0).where(np.isfinite(mins) & (mins > 0), 0.0)


def team_game_entries(games: pd.DataFrame,
                      team_stats: pd.DataFrame | None = None,
                      ) -> pd.DataFrame:
    """One row per player-game: signed share, game margin, game key.

    The fit this feeds is ONE row per game: ``y = sum_i x_i * beta_i``
    with ``x = +share`` on the home side, ``x = -share`` on the away
    side, and ``y`` the home margin. Both sides' players sit in the
    same equation - that is what makes a rating ADJUSTED, the
    opponent's columns priced alongside yours. Two rows each carrying
    their OWN margin would instead fit each side alone against the full
    margin, which is the unadjusted fit and it rates a role player on a
    good schedule like a star.

    Consequently both rows of a game carry the SAME ``y`` and the side
    travels separately as ``sign``. Nothing about the solved betas
    depends on WHICH side is called home: flipping one game's sign and
    margin flips both, leaving ``outer(x, x)`` and ``x * y`` unchanged
    - only the intercept's reading moves, and it absorbs that.    ``team_stats`` supplies sides and margins from the official scores
    (``net_points``, ``is_home``). Without it the box score is the
    fallback: the two teams of a game - paired on ``game_id`` (or the
    ``nba_game_id``/``GAME_ID`` spelling some feeds carry), or on the
    gameday when the frame carries no game id at all, which is only
    sound when the day has exactly ONE game - and the lexicographically
    first team stands in as "home".
    """
    keep = ["player_id", "gameday", "season", "team", "minutes"]
    work = games[keep].copy()
    # Normalize the date the same way the team table is normalized below:
    # a merge between a string gameday and a datetime one raises a dtype
    # error rather than silently matching nothing.
    work["gameday"] = pd.to_datetime(work["gameday"], errors="coerce")
    key_col = next((c for c in ("game_id", "nba_game_id", "GAME_ID")
                    if c in games.columns), None)
    if key_col is not None:
        work["game_key"] = games[key_col].astype(str)
    else:
        work["game_key"] = games.gameday.astype(str)
    work["share"] = participation_share(work.minutes)
    work = work[work.share > 0]
    if work.empty:
        return work.assign(points=np.nan, y=np.nan, sign=np.nan)
    has_points = "points" in games.columns
    if has_points:
        work["points"] = pd.to_numeric(
            games.points, errors="coerce").fillna(0.0)
    work["y"] = np.nan
    work["sign"] = np.nan
    if team_stats is not None and len(team_stats):
        ts = team_stats.copy()
        ts["gameday"] = pd.to_datetime(ts.get("gameday"),
                                        errors="coerce")
        ts["team"] = ts.get("team").astype(str)
        needed = {"gameday", "team", "net_points", "is_home"}
        if needed <= set(ts.columns):
            ts = ts[list(needed)].dropna(subset=["gameday", "team"])
            merged = work.merge(ts, on=["gameday", "team"], how="left")
            net = pd.to_numeric(merged.net_points, errors="coerce")
            home = (pd.to_numeric(merged.is_home, errors="coerce")
                    .fillna(0).astype(bool))
            merged["sign"] = np.where(home, 1.0, -1.0)
            merged["y"] = np.where(home, net, -net)
            work = work.drop(columns=["y", "sign"], errors="ignore").merge(
                merged[["player_id", "gameday", "y", "sign"]],
                on=["player_id", "gameday"], how="left")
    if work["y"].isna().any() and has_points:
        # Box-score fallback: pair the two teams of each game, take the
        # lexicographically first as the nominal home side. y = its
        # margin. A game whose opponent never appears in the log gets
        # no y and is skipped by the fit rather than half-priced.
        tp = work.groupby(["game_key", "team"]).points.sum()
        side: dict = {}
        margin: dict = {}
        for key in tp.index.levels[0]:
            teams = sorted(t for kk, t in tp.index if kk == key)
            if len(teams) != 2:
                continue
            side[(key, teams[0])] = 1.0
            side[(key, teams[1])] = -1.0
            margin[key] = float(tp.loc[(key, teams[0])]
                                - tp.loc[(key, teams[1])])
        # Reorient only unsupported games. A missing official row in ONE
        # game must not relabel every other game's genuine home-court side.
        unsupported = work.loc[work.y.isna() | work.sign.isna(),
                               "game_key"].unique()
        missing_game = work.game_key.isin(unsupported)
        fallback = work.loc[missing_game]
        work.loc[missing_game, "sign"] = [
            side.get((k, t), np.nan)
            for k, t in zip(fallback.game_key, fallback.team)]
        work.loc[missing_game, "y"] = [margin.get(k, np.nan)
                                      for k in fallback.game_key]
    return work[["player_id", "gameday", "season", "team", "game_key",
                 "minutes", "share", "sign", "y"]]


def prepare_player_games(stats: pd.DataFrame | None,
                         positions: pd.DataFrame | None = None,
                         ) -> pd.DataFrame:
    """One row per player-game with share, points and position.

    Returns an EMPTY frame with the full column set when the input cannot
    support a rating, rather than None. A caller that gets None has to guess
    whether the pull failed or the league played no games; an empty frame with
    the right columns is a state it can count and report.
    """
    columns = ["player_id", "gameday", "season", "team", "points",
               "minutes", "share"]
    if positions is not None and len(positions):
        columns = columns + ["position"]
        if "positions" in positions.columns:
            # The feed's full listing rides along beside the collapsed cell:
            # the rating is still computed against one cell (the prior's
            # denominator), but the team segments need "F|C" to know a roster
            # whose centers are all listed forward-centre HAS a center.
            columns = columns + ["positions"]
    empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})
    if stats is None or not len(stats):
        return empty
    if not set(_REQUIRED_COLS).issubset(stats.columns):
        return empty

    frame = stats.copy()
    frame["player_id"] = _player_id_str(frame.player_id).values
    for col in ("points", "minutes"):
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
    # 73,020 of 79,358 player-games (92%) were duplicated, ``prior_eff``
    # and ``prior_games`` were inflated up to 3x, and the min-eff and
    # rest gates were defeated for most of the league. The rating's
        # RATE is a ratio and so mostly cancelled, which is exactly why this
        # produced plausible numbers instead of an error - and why the
        # shrinkage weight and the pool gates were quietly wrong.
        if "season" in pos.columns and "season" in frame.columns:
            pos = pos.drop_duplicates(subset=["player_id", "season"],
                                      keep="first")
            join = ["player_id", "season", "position"]
            if "positions" in pos.columns:
                join.append("positions")
            frame = frame.merge(pos[join],
                                on=["player_id", "season"], how="left",
                                suffixes=("", "_resolved"))
        else:
            pos = pos.drop_duplicates(subset=["player_id"], keep="first")
            join = ["player_id", "position"]
            if "positions" in pos.columns:
                join.append("positions")
            frame = frame.merge(pos[join], on="player_id",
                                how="left", suffixes=("", "_resolved"))
    elif "position" not in frame.columns:
        # Absent position means every player falls out of every prior cell, so
        # the frame is explicitly marked rather than silently missing the
        # column. An EXISTING position column is left alone: overwriting it
        # with NaN would turn a correctly-labelled frame into one that rates
        # nobody, and the only symptom would be an empty result.
        frame["position"] = np.nan

    # The participation share is computed LAST, on the deduped
    # frame: it is the design-matrix entry of the RAPM fit, and a
    # box score counted twice would double the player's share in
    # that game. A DNP row carries a zero share and simply does
    # not enter the fit (team_game_entries drops it).
    frame["share"] = participation_share(frame.minutes)
    return frame


def season_eff_table(games: pd.DataFrame,
                     shrink_fraction: float = config.PLAYER_RAPM_SHRINK_FRACTION,
                     participation_floor_eff: float = 0.0,
                     through_season: str | None = None,
                     ) -> dict:
    """``k[position] = shrink_fraction * mean EFF-GAMES per PLAYER-SEASON``.

    The shrinkage strength is measured in the units the second layer
    weighs in: ``eff_games = sum of share**2``, the same participation-
    squared quantity that makes a player's weight in the fit. A
    player-season here means one player in one season, his window
    eff-games summed - a 3-game cameo contributes one tiny player-
    season, a full season one large one, and the mean over player-
    seasons (not over players) is what keeps the cameo from dragging
    the reference down the way per-player averaging would. On the
    cached logs the mean sits near 13 eff-games, so the 20% fraction
    yields ``k`` around 2.5-2.8 per position.

    ``participation_floor_eff`` optionally drops player-seasons below a
    threshold, moving the reference from "a season of any participation"
    toward "a full-time season".

    ``through_season`` applies the NHL discipline (``season_ice_time_table``'s
    ``as_of``): only player-seasons from seasons STRICTLY BEFORE it enter the
    mean, so a season cannot tune its own shrinkage strength. Without it the
    whole-frame mean also absorbs the season in progress - and an in-progress
    season holds PARTIAL player-seasons, which drags the mean down as the
    season accumulates and silently weakens every rating's prior weight game by
    game.

    Any cell with no evidence keeps ``config.PLAYER_RAPM_FALLBACK_K_EFF``, so a
    partially populated frame yields a COMPLETE shrinkage table instead of one
    that raises at lookup time.
    """
    table = {pos: float(config.PLAYER_RAPM_FALLBACK_K_EFF)
             for pos in config.PLAYER_EPM_POSITIONS}
    if games is None or not len(games):
        return table
    if not {"player_id", "season", "share", "position"}.issubset(games.columns):
        return table

    work = games[["player_id", "season", "share", "position"]].copy()
    work = work[work.position.isin(table)]
    if through_season is not None:
        cutoff = _season_key(through_season)
        work = work[work["season"].map(_season_key) < cutoff]
    if work.empty:
        return table
    work["eff"] = pd.to_numeric(work.share, errors="coerce").fillna(0.0) ** 2
    per_season = (work.groupby(["player_id", "season", "position"],
                               as_index=False)
                  .agg(eff=("eff", "sum")))
    if participation_floor_eff > 0:
        per_season = per_season[
            per_season.eff >= participation_floor_eff]
    if per_season.empty:
        return table
    means = per_season.groupby("position").eff.mean()
    for position, mean_eff in means.items():
        if mean_eff and math.isfinite(float(mean_eff)):
            table[position] = float(shrink_fraction) * float(mean_eff)
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
    date, say - still rates on a completed-season scale);    else the
    fixed fallback table; no future season may calibrate an early target.
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

    Built once per call to ``build_player_rapm`` because the answer is asked once
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
                     index: dict | None = None,
                     strict: bool = False) -> str:
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

    ``strict=True`` qualifies only seasons with a row STRICTLY BEFORE the
    target. Production uses this for BOTH roster membership and the fit:
    today's participation is a post-game fact, not a pregame roster source.
    The non-strict option remains available for diagnostics only.
    """
    own = _season_of(target)
    if index is None:
        index = _season_evidence_index(games)
    before = (lambda days: days < target) if strict else (
        lambda days: days <= target)
    days = index.get(own)
    if days is not None and len(days) and bool(before(days).any()):
        return own
    best_label, best_day = None, None
    for season, season_days in index.items():
        # Index the eligibility mask back onto the DATES. ``before`` returns
        # a boolean array, and ``.max()`` of a boolean array is just "was any
        # row eligible" (np.True_): the first season carrying any pre-target
        # row became unbeatable and the fallback selected the OLDEST season
        # in the frame. Two-season fixtures never caught it - first eligible
        # and most recent eligible coincide there - but every preseason
        # target walks into it: measured on the 2026-10-04 delivery, the
        # player board's priors matched 2023-24 minutes for 550/550 players
        # while the slate was rating 2026-27 games, two seasons stale.
        eligible = season_days[before(season_days)]
        if not len(eligible):
            continue
        latest = eligible.max()
        if best_day is None or latest > best_day:
            best_label, best_day = season, latest
    return best_label if best_label is not None else own


def _prior_for(games: pd.DataFrame, target, season: str | None = None
               ) -> pd.DataFrame:
    """Per-player participation totals over rows STRICTLY BEFORE ``target``.

    The prior is a plain filtered sum rather than a running total read at the
    last row. The two are not the same thing, and the difference bites on a
    player's SECOND game: a prefix-sum-minus-self column excludes each row's
    own line, so the last row before the target reports everything before
    ITSELF - which omits exactly the game the target is being rated for. Read
    that way a player is rated as though they had not yet played the game
    immediately preceding the one being projected, and a player's first game
    reports a prior of zero no matter how much they played in it.

    Summing the rows that precede the target is the definition of the prior, so
    that is what happens here. The target row itself is excluded by the strict
    ``<``, which is the entire leakage guarantee.

    Three totals come out, in the units the RAPM layers actually weigh:
    ``prior_eff`` (sum of share**2, the shrinkage weight), ``prior_minutes``
    (the team blend's numerator - minutes per game is the agreed aggregation
    weight), and ``prior_games`` (games PLAYED, the denominator). A DNP row
    carries zero share, so it is filtered before the count: it is a game the
    player did not participate in, and counting it would dilute the
    minutes-per-game blend with games he never took the floor in.
    """
    if "season" in games.columns:
        if season is None:
            season = _evidence_season(games, target)
        if season:
            games = games[games.season == season]
    earlier = games[games.gameday < target]
    if not earlier.empty:
        played = pd.to_numeric(earlier.share, errors="coerce").fillna(0.0) > 0
        earlier = earlier[played]
    if earlier.empty:
        return pd.DataFrame(columns=["player_id", "position", "season",
                                     "prior_eff", "prior_minutes",
                                     "prior_games"])
    keys = ["player_id", "position"] + (["season"] if "season" in earlier.columns
                                       else [])
    eff = pd.to_numeric(earlier.share, errors="coerce").fillna(0.0) ** 2
    minutes = pd.to_numeric(earlier.minutes, errors="coerce").fillna(0.0)
    work = earlier.assign(_eff=eff, _min=minutes)
    # ``size`` not ``sum``: prior_games counts the games that contributed
    # evidence, and a sum of a count column would be its square.
    return (work.groupby(keys, as_index=False)
            .agg(prior_eff=("_eff", "sum"),
                 prior_minutes=("_min", "sum"),
                 prior_games=("_eff", "size")))


class _SeasonDesign:
    """The running normal equations of the RAPM fit for ONE evidence season.

    The fit is ONE row per game: ``x`` is each participant's signed share
    (``+share`` home, ``-share`` away), ``y`` the home margin, plus one
    unregularized intercept column that absorbs home court. ``advance``
    folds in every game STRICTLY BEFORE a target date and ``solve`` returns
    the ridge betas, so a build over hundreds of target dates pays for each
    game's outer product ONCE instead of once per date - the difference
    between one pass over the season and a pass per target.

    Targets must arrive in ascending order; that is what lets the day
    pointer move forward only. The season partition around this class is
    what keeps a November rating from being solved against April's rows:
    each evidence season gets its own design.

    Both sides of every game sit in the same equation, which is what makes
    the rating ADJUSTED - the opponent's columns are priced alongside
    yours. A game that arrives with only one side present is skipped: two
    rows each carrying their OWN margin against the full margin is the
    unadjusted fit, and it is what rated Huerter +10 and LeBron -2 while
    no star reached the top of the list (the design this replaced).
    """

    def __init__(self, entries: pd.DataFrame,
                 lam: float = config.PLAYER_RAPM_LAMBDA):
        self.lam = float(lam)
        self.players: list = []
        self.pidx: dict = {}
        self.n = 0
        self.A = np.zeros((1, 1))
        self.b = np.zeros(1)
        self.eff = np.zeros(0)
        self.games = 0
        self._days: list = []
        self._ptr = 0
        self._cache = None
        self.blocks: dict = {}
        work = entries.dropna(subset=["y", "sign"]).copy()
        if work.empty:
            return
        work["player_id"] = _player_id_str(work.player_id).values
        self.players = list(pd.unique(work.player_id))
        self.pidx = {p: i for i, p in enumerate(self.players)}
        self.n = len(self.players)
        self.A = np.zeros((self.n + 1, self.n + 1))
        self.b = np.zeros(self.n + 1)
        self.eff = np.zeros(self.n)
        work["_col"] = work.player_id.map(self.pidx).astype(int)
        blocks: dict = {}
        for day, chunk in work.groupby("gameday", sort=True):
            day_blocks = []
            for _, game in chunk.groupby("game_key", sort=False):
                signs = game.sign.unique()
                if (len(game) < 2 or len(signs) != 2
                        or not np.isin([1.0, -1.0], signs).all()):
                    continue
                cols = game._col.to_numpy(dtype=int)
                x = (game.share.to_numpy(dtype=float)
                     * game.sign.to_numpy(dtype=float))
                day_blocks.append((cols, x, float(game.y.iloc[0])))
            if day_blocks:
                blocks[pd.Timestamp(day)] = day_blocks
        self.blocks = blocks
        self._days = sorted(blocks)

    def advance(self, target=None) -> None:
        """Fold in every game STRICTLY BEFORE ``target`` (all games if None).

        The strict ``<`` is the whole leakage guarantee: a game played ON
        the target day cannot have been solved for yet at that day's own
        rating.
        """
        if not self._days:
            return
        if target is None:
            target = self._days[-1] + pd.Timedelta(days=1)
        else:
            target = pd.Timestamp(target)
        while self._ptr < len(self._days) and self._days[self._ptr] < target:
            for cols, x, y in self.blocks[self._days[self._ptr]]:
                self.A[np.ix_(cols, cols)] += np.outer(x, x)
                self.A[cols, self.n] += x
                self.A[self.n, cols] += x
                self.A[self.n, self.n] += 1.0
                self.b[cols] += x * y
                self.b[self.n] += y
                self.eff[cols] += x * x
                self.games += 1
            self._ptr += 1

    def solve(self):
        """``(betas keyed by player_id, home-court intercept)``.

        Returns None before the first game is folded in: there is nothing
        to solve, and a fabricated zero would read as "neutral impact"
        rather than "no evidence".

        The intercept column is NEVER ridged - lambda sits on players only
        - or home court would be priced as if it were a player and the
        measured ~+1.7 point edge would be pulled toward zero.
        """
        if self.games == 0 or self.n == 0:
            return None
        if self._cache is not None and self._cache[0] == self.games:
            return self._cache[1], self._cache[2]
        M = self.A.copy()
        diag = np.arange(self.n)
        M[diag, diag] += self.lam
        try:
            # Cholesky instead of the general solver: a season's player
            # count solves in ~4ms versus ~430ms, which matters when a
            # whole-frame build asks for a solve per target date.
            chol = np.linalg.cholesky(M)
            full = np.linalg.solve(chol.T, np.linalg.solve(chol, self.b))
        except np.linalg.LinAlgError:  # singular by pathology, not by design
            full = np.linalg.lstsq(M, self.b, rcond=None)[0]
        raw = full[:self.n]
        seen = self.eff > 0
        # Center on the players this fit EVIDENCED. The ridge's min-norm
        # already sits near zero; centering makes the raw league mean
        # exactly zero without letting players who have not appeared yet
        # (zero columns, beta = 0 by construction) drag the reference
        # toward itself. Each row's signed shares sum to ~0 (home share -
        # away share ~ 5 - 5), so the shift moves every beta together and
        # leaves the intercept's reading of the home edge alone.
        center = raw[seen].mean() if seen.any() else raw.mean()
        beta = pd.Series(raw - center,
                         index=pd.Index(self.players, dtype="object"),
                         dtype="float64")
        self._cache = (self.games, beta, float(full[self.n]))
        return beta, float(full[self.n])


def fit_rapm(entries: pd.DataFrame, target=None,
             lam: float = config.PLAYER_RAPM_LAMBDA):
    """Solve the signed single-row design over ``entries`` in one shot.

    The production build solves incrementally through ``_SeasonDesign``;
    this is the same arithmetic for one target, for tests and audits that
    want a fit without a rating frame. ``target`` applies the same
    point-in-time cut the build uses (strictly before); ``None`` fits
    every row. Returns ``(betas, home intercept)`` or ``None`` when the
    input holds no complete game.
    """
    design = _SeasonDesign(entries, lam=lam)
    design.advance(target)
    return design.solve()


def shrunk_rapm(prior_eff, beta, league_rapm, k) -> pd.Series:
    """The shrinkage itself, in RAPM per-game units.

        RAPM_shrunk = (eff * beta + league[pos] * k) / (eff + k)

    Both sides are already in points per game - the beta the fit solved
    and the position mean of those same betas - so the algebra mixes them
    with no unit conversion, which is exactly why the EPM version's
    ``100 *`` numerator cannot be carried over: dropping it there did not
    raise, it returned a plausible 0.01 where the truth sat near 1.4.

    A player with no fitted beta (no evidence in the window) shrinks ALL
    the way to the position prior: ``eff`` counts only rows whose beta was
    observed, so his rating IS the prior rather than a number merely
    pulled toward it. A NaN league cell (a position with no evidence at
    all) keeps the rating NaN instead of manufacturing one.
    """
    eff = pd.to_numeric(pd.Series(prior_eff), errors="coerce")
    raw = pd.to_numeric(pd.Series(beta), errors="coerce")
    lg = pd.to_numeric(pd.Series(league_rapm), errors="coerce")
    weight = pd.to_numeric(pd.Series(k), errors="coerce")
    eff = eff.where(raw.notna(), 0.0)
    raw = raw.fillna(0.0)
    return (eff * raw + lg * weight) / (eff + weight)


def _roster_as_of(history, target):
    """Latest complete observation before target-day midnight Eastern.

    This is deliberately earlier than tipoff: current callers rate dates,
    not instants. Seven-day expiry prevents indefinite transaction carries.
    A partial archive or a conflicting player/team observation is refused.
    """
    if history is None or not len(history):
        return None
    required = {"player_id", "team", "position", "positions", "observed_at", "season", "source"}
    if not required <= set(history.columns):
        raise ValueError("roster history is missing required evidence columns")
    cutoff = pd.Timestamp(target).normalize().tz_localize("America/New_York").tz_convert("UTC")
    stamps = pd.to_datetime(history.observed_at, errors="coerce", utc=True, format="mixed")
    eligible = history[(stamps < cutoff)
                       & (stamps >= cutoff - pd.Timedelta(days=config.PLAYER_RAPM_ROSTER_MAX_AGE_DAYS))
                       & history.season.eq(_season_of(target))
                       & history.source.eq("nba_playerindex")].copy()
    if eligible.empty:
        return None
    eligible["_observed"] = stamps.loc[eligible.index]
    chosen = eligible[eligible._observed == eligible._observed.max()].copy()
    chosen["player_id"] = _player_id_str(chosen.player_id).values
    chosen["team"] = chosen.team.map(config.normalize_team_abbr)
    chosen = chosen.drop_duplicates(subset=["player_id", "team", "position", "positions"])
    counts = chosen.groupby("team").size()
    if (chosen.player_id.duplicated().any()
            or set(counts.index) != set(config.NBA_TEAM_ID) or counts.min() < 10
            or not chosen.position.isin((*config.PLAYER_EPM_POSITIONS, "")).all()
            or not chosen.positions.map(lambda s: isinstance(s, str) and
                (not s or set(s.split("|")) <= set(config.PLAYER_EPM_POSITIONS))).all()):
        raise ValueError("roster history contains an incomplete/ambiguous observation")
    return chosen


def _apply_roster_observation(snapshot, observed, known, target, lg, k):
    """Replace membership only; preserve fit evidence and appearance recency.

    Complete observations establish exits as well as arrivals. A subsequent
    strictly-prior appearance wins over the older roster, even for a player
    absent from it. A debutant retains zero exposure/NaN raw impact and cannot
    satisfy the existing evidence floor for a projected rotation slot.
    """
    members = observed[["player_id", "team", "position", "positions"]].copy()
    # Membership can be known while position is unpublished. Retain a
    # strictly-prior label if one exists; never guess a newcomer's segment.
    previous = snapshot.set_index("player_id")
    for col in ("position", "positions"):
        missing = members[col].eq("")
        if col in previous:
            members.loc[missing, col] = members.loc[missing, "player_id"].map(previous[col])
    stamp = observed._observed.iloc[0]
    members["roster_source"] = "nba_playerindex"
    members["roster_observed_at"] = stamp.isoformat()
    if len(known):
        latest = known.sort_values("gameday").groupby("player_id").tail(1).copy()
        # A box line is admitted only on the following calendar day, just as
        # in the fit. This makes a later appearance supersede a prior pull.
        knowledge = (latest.gameday + pd.Timedelta(days=1)).dt.tz_localize(
            "America/New_York").dt.tz_convert("UTC")
        latest = latest[knowledge > stamp]
        if len(latest):
            members = members[~members.player_id.isin(latest.player_id)]
            if "positions" not in latest:
                latest["positions"] = latest.position
            latest["roster_source"] = "prior_appearance"
            latest["roster_observed_at"] = ""
            members = pd.concat([members, latest[members.columns]], ignore_index=True)
    evidence_cols = [c for c in snapshot if c not in
                     ("team", "position", "positions", "roster_source", "roster_observed_at")]
    out = members.merge(snapshot[evidence_cols], on="player_id", how="left", validate="one_to_one")
    for col in ("prior_eff", "prior_minutes", "prior_games"):
        out[col] = out[col].fillna(0.0)
    out["prior_minutes_per_game"] = out.prior_minutes / out.prior_games.where(out.prior_games > 0)
    out["lg_rapm"] = out.position.map(lg)
    out["k_eff"] = out.position.map(k).astype(float)
    out["rapm_shrunk"] = shrunk_rapm(out.prior_eff, out.rapm_raw, out.lg_rapm, out.k_eff)
    out["target_date"] = target
    return out


def build_player_rapm(games: pd.DataFrame,
                      target_dates=None,
                      shrink_fraction: float = config.PLAYER_RAPM_SHRINK_FRACTION,
                      team_stats: pd.DataFrame | None = None,
                      availability: pd.DataFrame | None = None,
                      roster_history: pd.DataFrame | None = None,
                      ) -> pd.DataFrame:
    """Player-level RAPM ratings as of each target date.

    Returns one row per (target date, player): his participation prior,
    the position prior the fit's own betas define for that date, and the
    raw and shrunk ratings in points per game.

    Membership uses strictly-prior positive-minute appearances too: target-
    day box rows cannot establish a debut, trade, or renewed recency. A prior
    roster member without a fitted beta keeps NaN raw impact and may shrink
    to the position prior; an unknown player is not fabricated.

    ``roster_history`` optionally replaces membership with complete official
    observations fetched strictly before target-day midnight Eastern, no
    older than seven days and in the target season. It cannot backdate a
    trade, refresh appearance recency or contribute exposure to the fit.

    ``team_stats`` supplies official sides and margins to the design
    (``net_points``/``is_home``); without it ``team_game_entries`` falls
    back to pairing each game's two teams out of the box score.

    ``availability`` and ``availability_multiplier`` were removed rather
    than carried over: once the design moved to a stint table and a
    removal-based pool filter, they emitted a constant "healthy" and 1.0
    for every row - columns that look populated and carry nothing, and a
    multiplier that invites a future caller to reintroduce exactly the
    zero-weighting the pool deliberately avoids.
    """
    columns = ["target_date", "player_id", "position", "positions", "team",
               "prior_eff", "prior_minutes", "prior_minutes_per_game",
               "prior_games", "lg_rapm", "k_eff",
               "rapm_raw", "rapm_shrunk", "days_since_appearance",
               "roster_source", "roster_observed_at"]
    empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})
    if games is None or not len(games):
        return empty
    if "position" not in games.columns or not games.position.notna().any():
        return empty

    work = games[games.position.isin(config.PLAYER_EPM_POSITIONS)].copy()
    if work.empty:
        return empty

    if target_dates is None:
        dates = pd.Series(sorted(work.gameday.dropna().unique()))
    else:
        dates = pd.Series(pd.to_datetime(pd.Series(target_dates),
                                         errors="coerce").dropna().unique())
    dates = dates.sort_values().reset_index(drop=True)
    if dates.empty:
        return empty

    # The design is built from the WHOLE frame, not the position-filtered
    # one: a positionless player still occupies minutes of some game, and
    # dropping his column would leave his share to be claimed by his
    # teammates' betas - the mis-attribution the regression exists to
    # remove. He simply gets no OUTPUT row, because the prior he would
    # shrink toward is segmented by a position he does not have.
    entries = team_game_entries(games, team_stats=team_stats)
    fit_index = _season_evidence_index(entries)
    roster_index = _season_evidence_index(work)

    # Shrinkage strength is per TARGET SEASON, derived from completed prior
    # seasons only (NHL's season_ice_time_table discipline): a season cannot
    # tune its own k, and an in-progress season - whose player-seasons are
    # partial - would otherwise drag the mean down as the season accumulates
    # and silently weaken every rating's prior weight game by game.
    # A target with no completed prior season uses the fixed fallback.
    # The old whole-frame fallback saw future games/seasons and changed
    # earliest-season ratings when the same history was rebuilt later.
    k_table = {pos: float(config.PLAYER_RAPM_FALLBACK_K_EFF)
               for pos in config.PLAYER_EPM_POSITIONS}
    k_by_season: dict = {}
    if "season" in games.columns:
        frame_seasons = sorted(
            {s for s in games["season"].dropna().unique()},
            key=_season_key)
        # Include target seasons even when no player row has appeared in
        # them yet (opening slates). Their k uses ALL completed priors,
        # rather than reusing the previous target season's older table.
        target_seasons = {_season_of(d) for d in dates}
        for season in set(frame_seasons) | target_seasons:
            k_by_season[season] = season_eff_table(
                games, shrink_fraction=shrink_fraction, through_season=season)

    designs: dict = {}
    rows = []
    for target in dates:
        # Box-score membership, team changes and recency are post-game
        # facts too. Use the SAME strict cutoff as the solve: otherwise a
        # traded player moves before his first known appearance and an
        # actual target-day appearance resurrects a stale player in backtests
        # while the pending slate cannot see it. A debut is unknown until
        # its first PRIOR appearance unless a dated roster observation exists.
        roster_season = _evidence_season(work, target, roster_index,
                                         strict=True)
        fit_season = _evidence_season(entries, target, fit_index,
                                      strict=True)
        known = work[(work.gameday < target)
                     & (pd.to_numeric(work.share, errors="coerce") > 0)]
        if "season" in known.columns and roster_season:
            known = known[known.season == roster_season]
        if known.empty:
            continue
        design = designs.get(fit_season)
        if design is None:
            chunk = entries
            if "season" in entries.columns and fit_season:
                chunk = entries[entries.season == fit_season]
            design = _SeasonDesign(chunk)
            designs[fit_season] = design
        design.advance(target)
        solved = design.solve()
        beta = solved[0] if solved is not None else None

        # Only players with an actual PRIOR appearance may enter the roster.
        # A target-day box line, including a DNP, cannot establish pregame
        # membership, a trade or freshness.
        snapshot = _prior_for(work, target, season=fit_season)
        roster_cols = ["player_id", "position"]
        if "positions" in known.columns:
            # The full listing rides along for the team segments; it is a
            # player-level attribute like the collapsed cell, so it comes
            # from the roster side of every merge and never becomes a key.
            roster_cols.append("positions")
        roster = known[roster_cols].drop_duplicates(
            subset=["player_id", "position"])
        if "team" in known.columns:
            # The team as of the most recent row the player appears in,
            # which is the club he would be projected for. A mid-season
            # trade therefore moves him, because the later row wins.
            latest_team = (known.sort_values("gameday")
                           .groupby("player_id").tail(1)[["player_id", "team"]])
            roster = roster.merge(latest_team, on="player_id", how="left")
        if snapshot.empty:
            snapshot = roster.copy()
            for col in ("prior_eff", "prior_minutes", "prior_games"):
                snapshot[col] = 0.0
        else:
            keys = ["player_id", "position"]
            snapshot = roster.merge(snapshot, on=keys, how="left")
            for col in ("prior_eff", "prior_minutes", "prior_games"):
                snapshot[col] = snapshot[col].fillna(0.0)
        snapshot["prior_minutes_per_game"] = (
            snapshot.prior_minutes
            / snapshot.prior_games.where(snapshot.prior_games > 0))
        # The unshrunk rating is the solved beta for that player, NaN when
        # he has no window evidence: a zero column solves to beta 0, and
        # publishing that as "neutral impact" would be a fabrication - it
        # is "not in the fit".
        if beta is None:
            snapshot["rapm_raw"] = np.nan
        else:
            snapshot["rapm_raw"] = snapshot.player_id.map(beta)
        snapshot["rapm_raw"] = snapshot.rapm_raw.where(
            snapshot.prior_eff > 0)
        # The position prior is the mean raw beta among the SAME fit's
        # evidenced members at that position: shrink target and thing
        # shrunk come out of one solve, so they can never disagree about
        # scale or about which date they belong to. A position with no
        # evidenced member stays NaN, and the rating with it - nothing is
        # invented for a cell with no data.
        evidenced = snapshot[snapshot.rapm_raw.notna()]
        lg = evidenced.groupby("position").rapm_raw.mean()
        snapshot["lg_rapm"] = snapshot.position.map(lg)
        k_for_target = _k_table_for_season(
            k_by_season, k_table, _season_of(target))
        snapshot["k_eff"] = snapshot.position.map(k_for_target).astype(float)
        snapshot["rapm_shrunk"] = shrunk_rapm(
            snapshot.prior_eff, snapshot.rapm_raw,
            snapshot.lg_rapm, snapshot.k_eff)
        snapshot["target_date"] = target
        # Recency of the player's actual evidence, carried for the pool's
        # availability gate: days from the player's LAST strictly-prior
        # positive-minute appearance within the rated season. The gap is only
        # defined when the roster season IS the target's own season; a
        # carryover target (rated across a season boundary) carries NaN -
        # that is the season-start carryover case, governed by the min-eff
        # floor, and NOT an infinitely stale row.
        if roster_season == _season_of(target):
            known_early = known[known.gameday < target]
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
        snapshot["roster_source"] = "prior_appearance"
        snapshot["roster_observed_at"] = ""
        observation = _roster_as_of(roster_history, target)
        if observation is not None:
            snapshot = _apply_roster_observation(
                snapshot, observation, known, target, lg, k_for_target)
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
