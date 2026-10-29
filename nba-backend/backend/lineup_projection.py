"""Projected-lineup aggregates over the shrunk player RAPM ratings.

The NBA analogue of MLB's ``_LINEUP_AGG_ROSTER``, and it exists for the reason
MLB's exists. MLB's comment on the widen step is the whole argument:

    "Without this the IL filter cannot bind AT ALL: an injured player is absent
    from the participant pool by construction, so subtracting him is a no-op."

That failure is easy to reproduce by accident. A pool drawn only from players
who appeared recently cannot subtract an injured player, because an injured
player is precisely the one who did not appear. So the pool is WIDENED to every
team member with a rating row inside a short lookback, and only THEN is the
injury filter applied. The order is the feature.

Four further properties are inherited deliberately:

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
* **POSITION SEGMENTS READ THE ELIGIBLE ROSTER.** The top-k cut answers
  "who plays tonight" and governs the lineup aggregates only. A position
  segment asks "how does this team rate at this position", and the cut is
  arbitrary there: a team whose centers rank below the eight by participation
  is centerless in the projected lineup but not on its roster, so a segment cut
  from the lineup starves pl_dpm_c coverage for no evidentiary reason. Each
  segment therefore blends every eligible member of the position, priced by the
  same MINUTES-PER-GAME weights - the population MLB's lineup_agg adopted when it
  dropped its nine-cut, because opportunity weighting is what makes an
  arbitrary cut redundant.
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
try:
    from backend import nba_injury_report as injury_report_mod
except ImportError:
    import nba_injury_report as injury_report_mod

AGG_COLUMNS = ["gameday", "team", "pool_size", "healthy_size",
               "lineup_out_count", "lineup_healthy_frac",
               "lineup_rapm_concentration",
               "lineup_rapm_mean", "lineup_rapm_top3", "lineup_rapm_std",
               "lineup_rapm_rest_count",
               # PIT star availability (see _project_team): the top-three
               # prior-MPG reference set is fixed BEFORE the availability
               # filter, so an out star stays in the denominator.
               "star_out_impact", "star_out_weight_share"] + [
    f"pl_dpm_{p.lower()}" for p in config.PLAYER_EPM_POSITIONS] + [
    f"pl_srapm_{stem}_{p.lower()}"
    for stem in ("net", "off", "def")
    for p in config.PLAYER_EPM_POSITIONS] + [
    # The per-position ELIGIBILITY MASK (2026-10-08 coverage audit): 1 when
    # the eligible pool listed at least one member at the position, 0 when
    # the gates (prior-minutes floor, 30-day recency, availability) or the
    # feed's position listing left the segment empty. The value column stays
    # NaN on a refusal - that is correct and stays - but without the flag the
    # monitor cannot tell "no eligible centre for this club" (source-supported
    # missingness) from a build hole, and the audit measured both paging as
    # one flat low-coverage alarm: 43.84% at day 10 of the season and
    # team-dependent holes like WAS 68.7% post-day-30.
    f"pl_dpm_{p.lower()}_eligible" for p in config.PLAYER_EPM_POSITIONS]

#: The seven model-facing features, as home-minus-away diffs, which is the
#: convention ``features._attach_contract`` already uses for every other
#: paired feature. Each one earns its place against a specific failure:
#:
#: 1-4 are MLB's ``lineup_woba_*`` mirror. 5-7 exist because 1-4 cannot see
#: WHY two projected lineups differ - a team missing a star and a team missing
#: a bench player can carry an identical ``lineup_rapm_mean`` and they are not
#: the same bet. Availability asymmetry (5), proportional depletion (6) and
#: star-dependency (7) are the three ways that shows up.
DIFF_FEATURES = [
    "lineup_rapm_mean_diff",
    "lineup_rapm_top3_diff",
    "lineup_rapm_std_diff",
    "lineup_rapm_rest_count_diff",
    "lineup_out_count_diff",
    "lineup_healthy_frac_diff",
    "lineup_rapm_concentration_diff",
]

#: The two star-out aggregates, published as away/home/diff. The sources are
#: the per-team columns ``_project_team`` emits; the feature names are the
#: contract candidates in ``config.STAR_OUT_FEATURE_COLS``.
STAR_OUT_SOURCES = ["star_out_impact", "star_out_weight_share"]
STAR_OUT_FEATURES = list(config.STAR_OUT_FEATURE_COLS)

#: The nine available-lineup impact features: top-three, mean and weighted
#: dispersion of the projected healthy lineup, away/home/diff. The diff already
#: exists in ``DIFF_FEATURES`` for the legacy path; this family publishes the
#: sides too so the model can see which lineup is better, not only the gap.
LINEUP_RAPM_IMPACT_SOURCES = ["lineup_rapm_top3", "lineup_rapm_mean",
                              "lineup_rapm_std"]
LINEUP_RAPM_IMPACT_FEATURES = list(config.LINEUP_RAPM_IMPACT_FEATURE_COLS)

#: The nine position-segmented features: each position's eligible-roster
#: impact, opportunity-weighted by playing time, on BOTH sides plus the
#: difference. The segment blends the team's whole eligible roster at the
#: position (shrinkage already prices a thin member), not just the
#: projected eight - the cut that starved pl_dpm_c coverage is a lineup
#: concept, not a position concept.
#:
#: The sides are kept rather than differenced away because the two halves of a
#: position split carry different information. A diff says the home lineup is
#: better on impact; the sides say WHO is better and by how much, and a
#: model that can see "away is thin at centre" does not have to infer it from a
#: near-zero difference between two bad numbers. This is the same reason
#: ``lineup_out_count`` is kept alongside ``lineup_healthy_frac``.
#:
#: Only G/F/C exist because that is all the free source publishes; see
#: ``nba_injury_report`` / ``nba_sources.position_query``. A five-way guard and
#: wing split is what a lineup model would really want and is not obtainable.
POSITION_RAPM_SOURCES = [f"pl_dpm_{p.lower()}" for p in config.PLAYER_EPM_POSITIONS]
#: Published order is per-POSITION grouped - ``c_away, c_home, c_diff``, then
#: ``f_*``, then ``g_*`` - C first, matching the contract order in
#: ``config.PLAYER_RAPM_POSITION_FEATURE_COLS``. ``PLAYER_EPM_POSITIONS`` itself
#: is ordered G/F/C for the prior tables; the PUBLISHED column order follows
#: the contract, not the config tuple's internal order. The sides are kept
#: rather than differenced away because the two halves of a position split
#: carry different information: a diff says the home lineup is better at
#: impact; the sides say WHO is better and by how much.
_POSITION_PUBLISHED_ORDER = ("c", "f", "g")
POSITION_RAPM_FEATURES = [
    f"pl_dpm_{position}_{side}"
    for position in _POSITION_PUBLISHED_ORDER
    for side in ("away", "home", "diff")]

#: Stint-RAPM sources per team-game: net/offense/defense segment blends.
#: Published as net sides + diff (nine, mirroring the game-level family)
#: plus offense/defense diffs only (six): the off/def sides double the
#: count for little extra signal, and the diff is the model-facing shape.
STINT_RAPM_SOURCES = [
    f"pl_srapm_{stem}_{p.lower()}"
    for stem in ("net", "off", "def")
    for p in config.PLAYER_EPM_POSITIONS]
STINT_RAPM_FEATURES = (
    [f"pl_srapm_net_{position}_{side}"
     for position in _POSITION_PUBLISHED_ORDER
     for side in ("away", "home", "diff")]
    + [f"pl_srapm_{stem}_{position}_diff"
       for stem in ("off", "def")
       for position in _POSITION_PUBLISHED_ORDER])



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
                return injury_report_mod.normalize_designation_teams(frame)
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
    return injury_report_mod.normalize_designation_teams(designations)


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


def _position_segments(frame: pd.DataFrame):
    """Yield ``(position, members)`` for every position the FEED lists.

    The rating's ``position`` is one collapsed cell - the league prior's
    denominator, exactly one per player (``nba_sources.assign_positions``) -
    but a team segment answers a different question: what is this club's RAPM
    at centre? Collapsing every forward-centre into F answers that with NaN
    for whole rosters: ORL's centers are both listed F-C, and 140 of the 626
    team-games missing ``pl_dpm_c`` in the OOF frame are that labeling
    artifact rather than an injury. ``positions`` is the feed's full listing,
    so one player may price two segments - which is NOT the prior's
    double-count: each segment is its own weighted mean and nothing is summed
    ACROSS segments.

    A frame without the listing (a v1 fallback cache, a test fixture) yields
    exactly the previous behaviour: one group per collapsed cell.
    """
    labels = frame["positions"] if "positions" in frame.columns else frame["position"]
    if "positions" in frame.columns:
        # Rows from a single-label (v1) cache keep their collapsed cell.
        labels = labels.where(labels.notna(), frame["position"])
    exploded = frame.assign(
        position=labels.astype("object").str.split("|", regex=False)
    ).explode("position")
    # Equal-weighting (2026-10-26 ablation arm A-vs-B). A dual-listed player
    # keeps his MEMBERSHIP in every segment the listing names - that is what
    # holds C-segment coverage at 96% - but his opportunity weight is divided
    # across them, so one 29.7 mpg forward-centre adds 14.85 to F and 14.85 to
    # C instead of 29.7 to both. The segments' weights then sum to the roster
    # rather than 118% of it, and each segment's mean moves away from the
    # shared player toward its single-listed members.
    #
    # ``spans`` counts the segments he actually appears in, not the letters in
    # his listing, so a listed position the club has no group for cannot
    # silently discard a share of his weight.
    if (getattr(config, "PLAYER_POSITION_WEIGHT_SPLIT", False)
            and "_w" in exploded.columns
            and "player_id" in exploded.columns):
        spans = exploded.groupby("player_id")["_w"].transform("size")
        exploded["_w"] = (pd.to_numeric(exploded["_w"], errors="coerce")
                          .fillna(0.0) / spans.clip(lower=1).astype(float))
    yield from exploded.groupby("position", sort=False)


def _normalize(ratings: pd.DataFrame) -> pd.DataFrame:
    out = ratings.copy()
    out["player_id"] = out.player_id.astype(str)
    out["team"] = out.team.astype(str)
    out["gameday"] = pd.to_datetime(out.gameday, errors="coerce")
    for column in ("rapm_shrunk", "prior_minutes", "prior_minutes_per_game",
                   "off_raw", "def_raw"):
        out[column] = pd.to_numeric(out.get(column), errors="coerce")
    return out


#: Season-aware membership ramp (2026-10-21 opening-week remediation). The
#: prior-minutes floor measures evidence that cannot exist yet in the first
#: games of a season (priors are season-partitioned and strictly prior), so
#: a team with fewer than this many decided games ramps the floor linearly
#: instead of refusing the whole roster. Full strength by game 5; the
#: rating layer still shrinks thin evidence toward its priors, so an opener
#: with no evidence at all still refuses at the blend, honestly.
#:
#: 2026-10-26: the ramp is a MEMBERSHIP RULE, not a rescue. It used to fire
#: only when the full floor admitted NOBODY, which made it all-or-nothing:
#: on 2026-10-25 five SAC players cleared 50 minutes after two games (73,
#: 73, 72, 55, 54), so nothing fired and the remaining twelve stayed out of
#: the pool - healthy_size 5 of 17, and pl_dpm_c equal to one player's
#: rating in every digit. A floor the calendar has not yet permitted is now
#: scaled for EVERY candidate inside the ramp window, and a second arm
#: admits players on their cross-season trailing minutes (a veteran's last
#: thirty participated games), so week one prices a rotation instead of a
#: fragment of one. Outside the window the full floor still governs.
_FLOOR_RAMP_TEAM_GAMES = 5

#: Trailing window (in participated games) for the cross-season membership
#: basis. Thirty games is roughly a quarter-season of a regular's workload:
#: long enough that a rotation player clears any reasonable floor on it,
#: short enough that a changed role is not buried under ancient minutes.
_TRAILING_MEMBERSHIP_GAMES = 30


def trailing_membership_minutes(player_games: pd.DataFrame | None, target,
                                n_games: int = _TRAILING_MEMBERSHIP_GAMES
                                ) -> pd.Series:
    """Per-player minutes over the last ``n_games`` PARTICIPATED games
    strictly before ``target``, across season boundaries.

    The mirror of the rating prior without its season partition: the same
    strictly-before-target PIT floor (nothing at or after the target counts),
    the same DNP exclusion (a zero-share row is a game never played, not a
    scoreless one), but no season filter - a veteran's 2025-26 rotation is
    evidence he is a rotation candidate tonight even though the rating must
    not be fit on it. Never raises: bad input yields an empty table, and an
    absent player maps to zero minutes (refused) rather than erroring.
    """
    empty = pd.Series(dtype="float64", name="trailing_minutes")
    try:
        if player_games is None or not len(player_games):
            return empty
        need = {"player_id", "gameday", "minutes"}
        if not need.issubset(player_games.columns):
            return empty
        frame = player_games.copy()
        frame["gameday"] = pd.to_datetime(frame.gameday, errors="coerce")
        frame = frame[frame.gameday.notna()
                      & (frame.gameday < pd.Timestamp(target))]
        if "share" in frame.columns:
            played = pd.to_numeric(frame.share, errors="coerce").fillna(0.0) > 0
            frame = frame[played]
        if frame.empty:
            return empty
        frame["minutes"] = pd.to_numeric(frame.minutes,
                                           errors="coerce").fillna(0.0)
        key = frame.player_id.astype(str)
        tail = frame.sort_values("gameday").groupby(key, sort=False).tail(n_games)
        return (tail.groupby(tail.player_id.astype(str))["minutes"].sum()
                .rename("trailing_minutes").astype(float))
    except Exception:  # noqa: BLE001 - membership evidence must never fail a run
        return empty


def projected_lineup(ratings: pd.DataFrame | None,
                     games: pd.DataFrame | None = None,
                     stints=None,
                     trailing_minutes: pd.Series | dict | None = None,
                     lookback_days: int = config.PLAYER_EPM_POOL_LOOKBACK_DAYS,
                     min_minutes: float = config.PLAYER_RAPM_MIN_MINUTES,
                     top_k: int = config.PLAYER_EPM_TOP_K,
                     rest_minutes: float = config.PLAYER_RAPM_REST_MINUTES,
                     top5_k: int = config.PLAYER_EPM_TOP5_K,
                     reweight: str = "legacy",
                     diagnostics: dict | None = None,
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
    # Decided games per team in the trailing window, for the season-aware
    # membership ramp. The window (not the season label) is the clock: a
    # mid-season team always has a full window behind it, so the full floor
    # applies, while an opening-week team has one or none. A schedule-only
    # frame (no scores yet) contributes no history, so the ramp arms exactly
    # when the calendar says it should; without a schedule there is no
    # count and the full floor always applies.
    window_days = config.PLAYER_EPM_RECENCY_DAYS
    played_days: dict = {}
    if games is not None and len(games) and {"gameday"} <= set(games.columns) \
            and {"home_team", "away_team"} <= set(games.columns):
        sched = games.copy()
        sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")
        decided = sched[sched.gameday.notna()]
        if {"home_score", "away_score"} <= set(decided.columns):
            decided = decided[decided.home_score.notna()
                              & decided.away_score.notna()]
        sides = pd.concat([
            decided[["gameday", "home_team"]].rename(columns={"home_team": "team"}),
            decided[["gameday", "away_team"]].rename(columns={"away_team": "team"}),
        ], ignore_index=True)
        for side_team, grp in sides.groupby("team", sort=False):
            played_days[str(side_team)] = np.sort(
                grp.gameday.dropna().to_numpy(dtype="datetime64[ns]"))
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
            days = played_days.get(str(team))
            if days is None:
                team_games_played = None
            else:
                moment = np.datetime64(pd.Timestamp(gameday))
                floor = np.datetime64(
                    pd.Timestamp(gameday) - pd.Timedelta(days=window_days))
                team_games_played = int(((days < moment)
                                         & (days >= floor)).sum())
            row = _project_team(team_rows, str(team), gameday, stints,
                                lookback_days, min_minutes, top_k,
                                rest_minutes, top5_k,
                                team_games_played=team_games_played,
                                trailing_minutes=trailing_minutes,
                                reweight=reweight,
                                diagnostics=diagnostics)
            if row is not None:
                rows.append(row)
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="float64")
                             for c in AGG_COLUMNS})
    out = pd.DataFrame(rows)
    for column in AGG_COLUMNS:
        if column not in out.columns:
            # An ABSENT eligibility flag is a refusal (0), not a hole: the
            # rows that never reached the segment loop - the pool-empty and
            # healthy-empty paths (the season-opener recency gate empties
            # every roster) - are exactly "no eligible member on this
            # team-game". An absent VALUE stays NaN, as it always has.
            out[column] = 0.0 if column.endswith("_eligible") else np.nan
    return out[AGG_COLUMNS].sort_values(["gameday", "team"]).reset_index(drop=True)


#: Cap (minutes per game) on a survivor's FINAL weight under the capped
#: waterfall. The league's minutes leaders sit around 37-38 MPG; 40+ is
#: overtime or emergency territory, so 40 says "nobody is projected for more
#: than the league's heaviest realistic workload" and the excess spills to
#: the next man - which is what rotations actually do when they shorten.
#: The cap governs the REDISTRIBUTION, not the measurement: a trailing
#: average already above 40 is left alone.
NEXT_MAN_UP_MINUTES_CAP = 40.0


def _position_label_sets(frame: pd.DataFrame):
    """Per-row position sets from the feed listing ``_position_segments`` reads."""
    if "positions" in frame.columns:
        labels = frame["positions"].where(
            frame["positions"].notna(), frame.get("position"))
    else:
        labels = frame.get("position")
    if labels is None:
        return None
    # Dated rookie roster rows can have no position from either source yet.
    # Unknown is an empty set, not an invented role or an iterable NaN.
    return (labels.fillna("").astype(str).str.split("|", regex=False)
            .map(lambda parts: {str(p).strip() for p in parts
                                 if str(p).strip()}))


def _next_man_up_absorption(healthy: pd.DataFrame,
                            latest: pd.DataFrame,
                            cap=None) -> tuple[pd.Series, dict]:
    """Donate each removed member's weight to same-position survivor(s).

    The challenger to the legacy survivor-mean: a removed (Out/Doubtful/
    Recovery) member's opportunity weight does not vanish - it is absorbed in
    full by the healthy same-position member with the most trailing minutes
    (ties broken by player_id, so the choice is deterministic); with a cap
    each survivor is filled only up to ``cap`` final weight and the remainder
    waterfalls to the next same-position survivor in minutes order. A uniform
    proportional rescale of the survivors would be a no-op here: every blend
    divides by its weight sum, so only a NON-uniform transfer moves the
    means. Position matching reads the same feed listing ``_position_segments``
    does (multi-label ``positions`` with the collapsed ``position`` fallback),
    so a donor and his absorber always agree with the segments about who
    plays where. A donor with no same-position survivor (or no finite weight)
    vanishes exactly as in legacy; the counts say how often each happened.
    """
    absorbed = pd.Series(0.0, index=healthy.index)
    stats = {"donors": 0, "absorbed_weight": 0.0, "vanished_weight": 0.0,
             "vanished_no_survivor": 0, "spills": 0}
    if healthy.empty:
        return absorbed, stats
    healthy_sets = _position_label_sets(healthy)
    if healthy_sets is None:
        return absorbed, stats
    removed = latest[~latest.is_available.astype(bool)]
    if removed.empty:
        return absorbed, stats
    removed_sets = _position_label_sets(removed)
    order = (healthy.assign(_mpg=pd.to_numeric(healthy["prior_minutes"],
                                                errors="coerce"))
             .sort_values(["_mpg", "player_id"], ascending=[False, True]))
    running = pd.to_numeric(healthy["_w"], errors="coerce").fillna(0.0)
    # Scalar fast path (2026-10-26 run-log review): the per-donor
    # ``pd.to_numeric(pd.Series([scalar]))`` round-trip rebuilt a Series,
    # coerced it and indexed back out for every donor (~12k a full repull),
    # and the inner ``order.iterrows()`` rebuilt a Series per candidate per
    # donor. One vectorised coerce per column plus index iteration reads
    # the identical scalars: coerce maps unparseable/None to NaN exactly
    # as the Series round-trip did, and ``np.isfinite`` sees the same
    # operands, so the skip/absorb verdicts are unchanged.
    if "rapm_shrunk" in removed.columns:
        _rating_by_idx = pd.to_numeric(removed["rapm_shrunk"],
                                       errors="coerce")
    else:
        _rating_by_idx = pd.Series(np.nan, index=removed.index)
    if "prior_minutes_per_game" in removed.columns:
        _weight_by_idx = pd.to_numeric(removed["prior_minutes_per_game"],
                                       errors="coerce")
    else:
        _weight_by_idx = pd.Series(np.nan, index=removed.index)
    # One (name, position-set) pair per candidate, in absorption order:
    # the same rows ``order.iterrows()`` visited, without a Series per row.
    _cands = [(name, healthy_sets.loc[name]) for name in order.index]
    # Positional reads (not label lookups): identical per-row scalars even
    # if an index label ever repeated, exactly as the old iterrows walk.
    _rating_vals = _rating_by_idx.to_numpy()
    _weight_vals = _weight_by_idx.to_numpy()
    for _pos, idx in enumerate(removed.index):
        rating = _rating_vals[_pos]
        weight = _weight_vals[_pos]
        try:
            rating = float(rating)
            weight = float(weight)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(rating) or not np.isfinite(weight) or weight <= 0:
            continue
        parts = (removed_sets.loc[idx]
                 if removed_sets is not None and idx in removed_sets.index
                 else set())
        donor_set = set(parts)
        if not donor_set:
            stats["vanished_weight"] += float(weight)
            stats["vanished_no_survivor"] += 1
            continue
        stats["donors"] += 1
        remaining = float(weight)
        for _name, _cand_set in _cands:
            if not donor_set & _cand_set:
                continue
            if cap is not None:
                room = float(cap) - float(running.loc[_name])
                if room <= 0:
                    continue
                take = min(remaining, room)
            else:
                take = remaining
            running.loc[_name] += take
            absorbed.loc[_name] += take
            stats["absorbed_weight"] += take
            remaining -= take
            if take < float(weight):
                stats["spills"] += 1
            if remaining <= 0:
                break
        if remaining > 0:
            stats["vanished_weight"] += remaining
            if remaining == float(weight):
                stats["vanished_no_survivor"] += 1
    return absorbed, stats


def _star_out_aggregates(latest: pd.DataFrame) -> tuple[float, float]:
    """Unavailable talent and unavailable role share among the top three.

    The reference set is built from ``latest`` - the ELIGIBLE pool BEFORE
    availability is applied - so an out star remains in the denominator and
    the next healthy man does NOT replace him. That distinction is the whole
    feature: a top-three built from the healthy pool would read near zero
    exactly when a star is missing.

    Membership is the rating pool's, not the 18-man roster's: the same
    prior-minutes floor and recency gate that decide who can be priced decide
    who can be a "star" here. ``prior_minutes_per_game`` is the weight, in
    keeping with every other lineup aggregate. When the top three carry no
    measurable per-game weight (the opening-week case), equal weights are the
    honest fallback, mirroring ``_blend``. A rating-less member contributes no
    rating but still contributes role weight, so ``star_out_weight_share``
    survives even when ``star_out_impact`` is undefined.

    Returns ``(impact, share)``; both are NaN only when there is no eligible
    player at all.
    """
    if latest is None or not len(latest):
        return np.nan, np.nan
    ref = latest.copy()
    weights = pd.to_numeric(ref.get("prior_minutes_per_game"),
                            errors="coerce")
    if weights is None:
        weights = pd.Series(0.0, index=ref.index)
    weights = weights.fillna(0.0).clip(lower=0.0).astype(float)
    minutes = pd.to_numeric(ref.get("prior_minutes"), errors="coerce")
    if minutes is None:
        minutes = pd.Series(0.0, index=ref.index)
    minutes = minutes.fillna(0.0).astype(float)
    ratings = pd.to_numeric(ref.get("rapm_shrunk"), errors="coerce")
    if ratings is None:
        ratings = pd.Series(np.nan, index=ref.index)
    if "player_id" in ref.columns:
        player_id = ref["player_id"].astype(str)
    else:
        player_id = pd.Series("", index=ref.index)
    order = ref.assign(_w=weights, _minutes=minutes,
                       _rating=ratings, _pid=player_id)
    top = order.sort_values(["_w", "_minutes", "_rating", "_pid"],
                            ascending=[False, False, False, True]).head(3)
    if top.empty:
        return np.nan, np.nan
    w = top["_w"].to_numpy(dtype=float)
    total = float(w.sum())
    if total <= 0:
        w = np.ones(len(top), dtype=float)
        total = float(len(top))
    available = (top["is_available"].astype(bool).to_numpy()
                 if "is_available" in top.columns
                 else np.ones(len(top), dtype=bool))
    out = ~available
    share = float((w * out).sum() / total)
    r = top["_rating"].to_numpy(dtype=float)
    finite = np.isfinite(r)
    impact = (float((w[finite] * r[finite] * out[finite]).sum() / total)
              if finite.any() else np.nan)
    return impact, share


def _project_team(work: pd.DataFrame, team: str, gameday, stints,
                  lookback_days: int, min_minutes: float, top_k: int,
                  rest_minutes: float, top5_k: int,
                  team_games_played: int | None = None,
                  trailing_minutes: pd.Series | dict | None = None,
                  reweight: str = "legacy",
                  diagnostics: dict | None = None):
    """Widen, filter, rank - in that order, for one team and one game."""
    window_start = pd.Timestamp(gameday) - pd.Timedelta(days=lookback_days)
    # STEP 1 - WIDEN. Every team member with a rating row in the window,
    # whether or not they appeared recently. This is the step that gives the
    # injury filter something to remove.
    if "roster_source" in work.columns:
        # Rating history is not membership history: yesterday's old-team row
        # must not resurrect a player removed/moved by today's observation.
        work = work[work.gameday == pd.Timestamp(gameday)]
    pool = work[(work.team == team)
                & (work.gameday <= pd.Timestamp(gameday))
                & (work.gameday >= window_start)
                & work.rapm_shrunk.notna()]
    if pool.empty:
        # POOL refusal, emitted as a row: no team-member rating lands in the
        # lookback — the season opener, where every prior target date sits
        # outside the 10-day window (the run-engine's own gap note calls
        # early-season rows "structural, not a build failure"). Emitting the
        # row with the eligibility mask at 0 is what lets monitoring tell
        # this apart from a team the ratings frame never contained at all:
        # that case never reaches here (no row, mask unknown, alarm kept).
        return {"gameday": pd.Timestamp(gameday), "team": team,
                "pool_size": 0, "healthy_size": 0,
                "lineup_rapm_mean": np.nan, "lineup_rapm_top3": np.nan,
                "lineup_rapm_std": np.nan, "lineup_rapm_rest_count": np.nan,
                **{f"pl_dpm_{p.lower()}_eligible": 0.0
                   for p in config.PLAYER_EPM_POSITIONS}}
    # One row per player: the most recent rating at or before the game. A
    # player's own current game can supply his prior rating but never a rating
    # that includes the game being projected.
    tail = (pool.sort_values(["player_id", "gameday"])
            .groupby("player_id", as_index=False).tail(1))
    pool_size = len(tail)
    # Opening-week signal, shared by the membership fallback and the weight
    # fallback below: fewer ramp games than the floor's full strength means
    # the calendar, not the data, is why evidence is thin. Unknown calendar
    # (no schedule) keeps the conservative full-floor, no-rescue behavior.
    early_season = (team_games_played is not None
                    and 0 <= team_games_played < _FLOOR_RAMP_TEAM_GAMES)

    # STEP 2 - ELIGIBILITY. The min-minutes floor is about POOL MEMBERSHIP,
    # not about the rating: shrinkage already handles a thin rating, but a
    # player with three career games is not a candidate for tonight's lineup
    # however well his rate is estimated. The recency gate is the second
    # membership clause: a rating row dated after a player stopped appearing
    # is a phantom (injury never filed, quiet shutdown, roster cut), and no
    # designation will ever remove him - the row only LOOKS fresh because it
    # is re-emitted per target date. The gap is the player's actual last
    # strictly-prior appearance within the rated season;
    # NaN means the season has not started for him yet, which is the
    # season-start carryover the min-minutes floor already governs, not
    # staleness.
    def _membership(frame: pd.DataFrame, floor: float,
                    basis: np.ndarray | None = None) -> pd.DataFrame:
        minutes = (frame.prior_minutes.to_numpy(dtype=float) if basis is None
                   else np.asarray(basis, dtype=float))
        gated = frame[minutes >= floor]
        if "days_since_appearance" in gated.columns:
            recency = gated["days_since_appearance"]
            gated = gated[recency.isna()
                          | (recency <= config.PLAYER_EPM_RECENCY_DAYS)]
        return gated

    def _trailing_basis(frame: pd.DataFrame):
        """Per-player cross-season minutes, or None when no table was sent."""
        if trailing_minutes is None:
            return None
        try:
            lookup = (trailing_minutes.to_dict()
                      if hasattr(trailing_minutes, "to_dict")
                      else dict(trailing_minutes))
        except Exception:  # noqa: BLE001 - corrupt lookup refuses nothing extra
            lookup = {}
        return (frame.player_id.astype(str).map(lookup)
                .fillna(0.0).astype(float).to_numpy())

    if early_season:
        # The ramped floor applies to EVERY candidate, and the cross-season
        # arm is ORed in rather than tried only after a total refusal. A
        # veteran's 2025-26 rotation proves he is a rotation candidate even
        # when this season has given him nothing yet (2026-10-21: 423/519
        # pool members cleared 50 on their last 30); a rookie with no
        # trailing history maps to zero and has to clear the ramped floor on
        # his own minutes, which is the honest answer for him. The recency
        # gate inside ``_membership`` still applies to both arms.
        ramped = min_minutes * team_games_played / _FLOOR_RAMP_TEAM_GAMES
        own = _membership(tail, ramped)
        basis = _trailing_basis(tail)
        if basis is None:
            latest = own
        else:
            cross = _membership(tail, min_minutes, basis=basis)
            latest = pd.concat([own, cross], ignore_index=True).drop_duplicates(
                subset=["player_id"], keep="first")
    else:
        latest = _membership(tail, min_minutes)

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

    # PIT STAR AVAILABILITY. Read the reference set BEFORE the availability
    # filter is allowed to choose the top three: an out star stays in the
    # denominator, and the fourth man does not become a "star" by promotion.
    #
    # It is also read BEFORE the healthy-empty refusal below, because that is
    # the case the feature exists for. ``star_out_*`` describes the FILINGS
    # (who is designated out), not the ratings, so a pool that collapses to
    # nobody healthy is the MOST informative reading of the six columns and
    # returning NaN there inverts the feature: the quieter the designation
    # news, the more the model was told. The refusal's own columns
    # (``lineup_rapm_*``, the nine) stay NaN - those really are unpriceable.
    #
    # A pool refusal (no rated member in the lookback) keeps NaN for both,
    # honestly: with no reference set there is no top three to report on, and
    # that row is the season opener's structural hole, not a filing.
    star_out_impact, star_out_weight_share = _star_out_aggregates(latest)

    healthy = latest[latest.is_available.astype(bool)]
    healthy_size = len(healthy)
    if healthy.empty:
        # Every candidate the gates admitted is unavailable (or the gates
        # emptied the pool - the season-opener recency gate does it to every
        # roster): refusal at every position, stated EXPLICITLY so a mixed
        # frame still carries the 0. The column-level default below only
        # fires when no row at all carries the flag, and a real run always
        # mixes refused team-games with priced ones (the 2026-10-09 real-
        # frame smoke caught exactly that: 56 healthy-empty rows riding NaN
        # flags beside priced rows).
        return {"gameday": pd.Timestamp(gameday), "team": team,
                "pool_size": pool_size, "healthy_size": 0,
                "lineup_rapm_mean": np.nan, "lineup_rapm_top3": np.nan,
                "lineup_rapm_std": np.nan, "lineup_rapm_rest_count": np.nan,
                "star_out_impact": star_out_impact,
                "star_out_weight_share": star_out_weight_share,
                **{f"pl_dpm_{p.lower()}_eligible": 0.0
                   for p in config.PLAYER_EPM_POSITIONS}}

    # STEP 4 - RANK BY PARTICIPATION, NOT BY RATING. MLB orders the pool by
    # trailing PA and averages the rating over the top nine. Ranking by the
    # rating instead would project the eight best-rated regulars and quietly
    # redefine the question from "who plays" to "who produces", which is a
    # different feature wearing this one's name.
    healthy = healthy.sort_values(["prior_minutes", "gameday"], ascending=False)
    # MINUTES-PER-GAME weighting (the agreed team-level aggregation): a
    # member's weight is his OWN minutes per game over the window, so each
    # aggregate is the lineup's RAPM averaged PER GAME over the minutes
    # distribution it would actually field - "what does this roster rate per
    # game when its real rotation prices it". Cumulative minutes would fold
    # durability into the blend (an ironman outweighing an equal per-game
    # player who missed time) and the result would no longer be a per-game
    # average, which is the unit the ratings themselves live in. The
    # min-minutes floor is what stops a two-game call-up from pricing like a
    # regular. Weights are the same strictly-prior per-player totals the pool
    # ranking already reads; nothing new is fetched and nothing leaks. A NaN
    # rating contributes no weight and no rating (removal, not zeroing); a
    # family with no finite-weight member keeps NaN, except in the
    # opening-week ramp where the whole pool is unweighted by calendar, not
    # by data (see below). The weights are assigned
    # on the whole ELIGIBLE roster before the top-k slice so the lineup
    # aggregates and the position segments price members identically.
    healthy = healthy.copy()
    healthy["_w"] = healthy["prior_minutes_per_game"].where(
        healthy.rapm_shrunk.notna(), other=0.0).astype(float)
    if early_season and not float(healthy["_w"].sum()):
        # No measurable per-game opportunity anywhere in the pool: every
        # per-game weight is NaN (no one has played) or zero. Average the
        # finite ratings equally rather than refusing - the values are
        # already shrunk toward their priors, so this prices the priors,
        # honestly, instead of failing the slate gate for the calendar.
        # Only fires where the old code returned NaN, and only before the
        # ramp fills; a mid-season weight hole keeps its NaN alarm.
        healthy["_w"] = healthy.rapm_shrunk.notna().astype(float)
    if reweight in ("next_man_up", "waterfall"):
        # The challenger: each removed member's weight is absorbed by
        # same-position survivor(s) - one absorber uncapped, or a capped
        # waterfall spilling down the minutes order (see
        # _next_man_up_absorption for why a proportional rescale would be a
        # no-op). Unknown modes fall through to legacy weights rather than
        # failing the run: an experimental knob must never break the
        # shipping path.
        cap = NEXT_MAN_UP_MINUTES_CAP if reweight == "waterfall" else None
        absorbed, stats = _next_man_up_absorption(healthy, latest, cap=cap)
        healthy["_w"] = healthy["_w"] + absorbed
        if diagnostics is not None:
            for key, value in stats.items():
                diagnostics[key] = diagnostics.get(key, 0) + value
    projected = healthy.head(top_k)
    w_sum = float(projected["_w"].sum())

    def _blend(frame: pd.DataFrame, col: str = "rapm_shrunk") -> float:
        """Minutes-per-game-weighted mean of ``col`` over ``frame``."""
        w = frame["_w"]
        den = float(w.sum())
        if not den or col not in frame.columns:
            return np.nan
        return float((w * frame[col].fillna(0.0)).sum() / den)

    ratings = projected.rapm_shrunk.to_numpy(dtype=float)

    # The rest count: top-5 regulars by participation who are NOT in the
    # projected lineup. Below rest_minutes a player is not a rotation
    # regular, so his absence is not news.
    regulars = healthy[healthy.prior_minutes >= rest_minutes].head(top5_k)
    rest_count = int(len(regulars) - len(
        regulars[regulars.player_id.isin(set(projected.player_id))]))

    mean = _blend(projected)
    top3 = _blend(projected.head(3))
    # Weighted dispersion around the weighted mean: the same population form
    # (ddof=0) the plain std used, with each member's deviation priced by his
    # share of the lineup's opportunities.
    std = np.nan
    finite = projected[projected.rapm_shrunk.notna()]
    if len(finite) >= 2 and w_sum:
        dev = finite.rapm_shrunk.to_numpy(dtype=float) - mean
        std = float(np.sqrt((finite._w.to_numpy(dtype=float)
                             * dev ** 2).sum() / w_sum))
    # Position-segmented impact over the team's ELIGIBLE ROSTER at
    # each position, not over the projected eight. The lineup aggregates
    # describe who plays tonight, so they are cut at top_k; a position
    # segment describes the team's RAPM at that position, and the cut
    # is arbitrary there - a team whose centers all rank below the eight
    # by participation is centerless in the projected lineup but NOT
    # centerless on its roster, so its pl_dpm_c stayed NaN for no reason
    # other than the cut. That is the pl_dpm_c coverage hole the monitor
    # flags as STARVED: 3 of 6 team-games in the 2026-10-01 artifact,
    # each with a 16-17 man healthy roster. So the segment blends every
    # eligible member of the position, priced by the same opportunity
    # weights - the population MLB's lineup_agg adopted when it dropped
    # its nine-cut, because playing-time weighting is what makes an
    # arbitrary cut redundant. One projection still defines the pool, so
    # the segments cannot disagree with the lineup about who is eligible,
    # who is available, or how much evidence each member has; they simply
    # read all of it. The membership itself is the FEED's listing, not the
    # collapsed cell: the prior may put a forward-centre in exactly one
    # denominator, but a segment asks which players THIS club plays at
    # centre, and collapsing every F-C to F left whole rosters (ORL:
    # Wendell Carter Jr., Goga Bitadze) with no C segment at all - most
    # of the 62% pl_dpm_c coverage the monitor flags.
    segmented: dict = {}
    eligible: dict = {}
    if "position" in healthy.columns:
        for position, group in _position_segments(healthy):
            if position in config.PLAYER_EPM_POSITIONS and len(group):
                # Eligibility is MEMBERSHIP, not blend success: a segment
                # whose members all carry unweightable ratings is eligible
                # (1) with a refused value, so that null keeps its alarm.
                # Only an empty segment is a refusal (0).
                eligible[f"pl_dpm_{str(position).lower()}_eligible"] = 1.0
                value = _blend(group)
                if value == value:
                    segmented[f"pl_dpm_{str(position).lower()}"] = value
                # Stint O/D parallels, priced on the same weights over the
                # same members. Present only when the ratings frame carries
                # the stint fit (off_raw/def_raw); a game-level frame simply
                # emits none of these and the AGG_COLUMNS fill covers them
                # with NaN. Net reuses rapm_shrunk, which the stint frame
                # sets to the stint net outright.
                if {"off_raw", "def_raw"}.issubset(healthy.columns):
                    pos = str(position).lower()
                    for stem, col in (("net", "rapm_shrunk"),
                                      ("off", "off_raw"),
                                      ("def", "def_raw")):
                        svalue = _blend(group, col)
                        if svalue == svalue:
                            segmented[f"pl_srapm_{stem}_{pos}"] = svalue
        for position in config.PLAYER_EPM_POSITIONS:
            eligible.setdefault(f"pl_dpm_{position.lower()}_eligible", 0.0)
    else:
        # No position listing at all (a v1 frame is handled inside
        # _position_segments; this is a frame with no position column): the
        # refusal cannot be attributed, so the mask stays UNKNOWN (NaN) -
        # never "refused", which would silence a real hole, and never a
        # fabricated eligibility claim.
        eligible = {f"pl_dpm_{p.lower()}_eligible": np.nan
                    for p in config.PLAYER_EPM_POSITIONS}
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
        # Star dependency: the top three's opportunity-weighted share of the
        # lineup's weighted mean. A high value means the projection rests on a
        # few players, so losing one of them costs more than the mean alone
        # suggests.
        "lineup_rapm_concentration": (top3 / mean) if mean else np.nan,
        "lineup_rapm_mean": mean,
        "lineup_rapm_top3": top3,
        "lineup_rapm_std": std if std == std else np.nan,
        "lineup_rapm_rest_count": rest_count,
        "star_out_impact": star_out_impact,
        "star_out_weight_share": star_out_weight_share,
        **segmented,
        **eligible,
    }


def attach_to_slate(slate: pd.DataFrame, aggregates: pd.DataFrame) -> pd.DataFrame:
    """Attach the seven projected-lineup diff features to a slate.

    Each aggregate is resolved PER SIDE and then differenced, matching
    ``features._diff`` and every other paired feature in the contract. Doing
    the sides first and differencing second is what keeps the arithmetic
    consistent with the rest of the model: a home-minus-away difference of two
    independently-joined columns is the same number, but computing it in one
    place means there is exactly one definition of "home advantage in projected
    impact" in the codebase.

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
        "lineup_rapm_mean_diff": "lineup_rapm_mean",
        "lineup_rapm_top3_diff": "lineup_rapm_top3",
        "lineup_rapm_std_diff": "lineup_rapm_std",
        "lineup_rapm_rest_count_diff": "lineup_rapm_rest_count",
        "lineup_out_count_diff": "lineup_out_count",
        "lineup_healthy_frac_diff": "lineup_healthy_frac",
        "lineup_rapm_concentration_diff": "lineup_rapm_concentration",
    }
    out = _attach_sides(out, aggregates, list(sources.values()))
    for feature, column in sources.items():
        home = f"_home_{column}"
        away = f"_away_{column}"
        if home in out.columns and away in out.columns:
            out[feature] = out[home] - out[away]
    drop = [c for c in out.columns if c.startswith("_home_") or c.startswith("_away_")]
    return out.drop(columns=drop)


def attach_star_out_features(slate: pd.DataFrame,
                             aggregates: pd.DataFrame) -> pd.DataFrame:
    """Attach the six PIT star-availability features, sides retained.

    ``star_out_impact_{away,home,diff}`` and
    ``star_out_weight_share_{away,home,diff}`` are ALWAYS created, populated
    or not, matching the always-create rule every other feature family here
    follows. The sides carry different information: the diff says which team
    is more depleted, the sides say whether either team lost top-end talent at
    all. The diff is computed here, not by a caller, so there is one definition
    of home-minus-away for this family.
    """
    out = slate.copy() if slate is not None else pd.DataFrame()
    for column in STAR_OUT_FEATURES:
        out[column] = np.nan
    if out.empty or aggregates is None or not len(aggregates):
        return out
    if "gameday" not in out.columns:
        return out
    out = _attach_sides(out, aggregates, STAR_OUT_SOURCES)
    for source in STAR_OUT_SOURCES:
        away = f"_away_{source}"
        home = f"_home_{source}"
        if away not in out.columns or home not in out.columns:
            continue
        out[f"{source}_away"] = out[away]
        out[f"{source}_home"] = out[home]
        out[f"{source}_diff"] = out[home] - out[away]
    drop = [c for c in out.columns
            if c.startswith("_home_") or c.startswith("_away_")]
    return out.drop(columns=drop)


def project_lineup_features(ratings: pd.DataFrame,
                            games: pd.DataFrame,
                            designations=None,
                            name_by_id: dict | None = None,
                            reweight: str = "waterfall",
                            diagnostics: dict | None = None,
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Designations -> projection -> the nine + six: THE production path.

    One function, so the training build, the slate build and every A/B harness
    resolve a team's lineup number in exactly one place. They did not, and the
    harness drifted twice: it ran ``reweight="legacy"`` while the pipeline ran
    ``"waterfall"``, so a "replicated" incumbent arm was a different feature
    family, and it attached the two families itself with a different sequence.
    A harness that re-implements the thing it measures is measuring itself.

    ``reweight`` defaults to the pipeline's mode deliberately: the default is
    the shipping behaviour, and an experimental override has to say so.

    Returns ``(frame, aggregates)`` - the input frame with the fifteen
    columns attached, and the per-(game, team) projection that produced them.
    Nothing is dropped and no row is filtered: the caller keeps its own
    game split, because slicing here would hide which rows failed the join.
    """
    rated = ratings.copy() if ratings is not None else pd.DataFrame()
    if "target_date" in rated.columns:
        rated = rated.rename(columns={"target_date": "gameday"})
    if "gameday" in rated.columns:
        rated["gameday"] = pd.to_datetime(rated.gameday, errors="coerce")
    if "is_available" not in rated.columns:
        rated["is_available"] = True
    rated = apply_pit_designations(rated, designations, name_by_id)
    aggregates = projected_lineup(rated, games=games, reweight=reweight,
                                  diagnostics=diagnostics)
    frame = attach_position_rapm(games, aggregates)
    return attach_star_out_features(frame, aggregates), aggregates


def attach_lineup_rapm_impact(slate: pd.DataFrame,
                              aggregates: pd.DataFrame) -> pd.DataFrame:
    """Attach the nine available-lineup impact features, sides retained.

    Same join and always-create rule as :func:`attach_position_rapm`. These
    read the projected HEALTHY lineup, so a star's absence promotes the next
    healthy man into the top three. That is deliberate: the family answers
    "how good is the lineup that will play", while ``attach_star_out_features``
    answers "who is missing". The ablation separates the two arms for exactly
    that reason.
    """
    out = slate.copy() if slate is not None else pd.DataFrame()
    for column in LINEUP_RAPM_IMPACT_FEATURES:
        out[column] = np.nan
    if out.empty or aggregates is None or not len(aggregates):
        return out
    if "gameday" not in out.columns:
        return out
    out = _attach_sides(out, aggregates, LINEUP_RAPM_IMPACT_SOURCES)
    for source in LINEUP_RAPM_IMPACT_SOURCES:
        away = f"_away_{source}"
        home = f"_home_{source}"
        if away not in out.columns or home not in out.columns:
            continue
        out[f"{source}_away"] = out[away]
        out[f"{source}_home"] = out[home]
        out[f"{source}_diff"] = out[home] - out[away]
    drop = [c for c in out.columns
            if c.startswith("_home_") or c.startswith("_away_")]
    return out.drop(columns=drop)


def attach_position_rapm(slate: pd.DataFrame,
                        aggregates: pd.DataFrame) -> pd.DataFrame:
    """Attach the nine position-segmented impact features, sides retained.

    Same join and same (date, team) grain as :func:`attach_to_slate`, and the
    same "always create the column" rule - a feature that exists only on days
    it could be computed is a feature whose train-serve behaviour depends on
    the calendar.

    Unlike :func:`attach_to_slate` this KEEPS the per-side columns instead of
    dropping them as scaffolding, because the sides are the point: the nine
    published features are ``{away, home, diff}`` per position, not three diffs.

    The ``_eligible_*`` provenance columns are the audit's mask (same
    underscore convention as ``_measured_*``): one flag per published
    feature saying whether a null on this row is a builder REFUSAL (0,
    source-supported) or not (1 / unknown). The diff's flag is the minimum
    of the sides - a refused side makes the difference NaN by construction,
    so one refusal explains the diff; two eligible sides with a NaN diff
    stay unexplained. A team-game the projection never produced a row for
    joins to NaN: unknown cause is never "refused".
    """
    out = slate.copy() if slate is not None else pd.DataFrame()
    for column in POSITION_RAPM_FEATURES:
        out[column] = np.nan
    if out.empty or aggregates is None or not len(aggregates):
        return out
    out = _attach_sides(out, aggregates, POSITION_RAPM_SOURCES)
    out = _attach_sides(out, aggregates,
                        [f"{column}_eligible" for column in POSITION_RAPM_SOURCES])
    for column in POSITION_RAPM_SOURCES:
        away = f"_away_{column}"
        home = f"_home_{column}"
        if away in out.columns and home in out.columns:
            out[f"{column}_away"] = out[away]
            out[f"{column}_home"] = out[home]
            out[f"{column}_diff"] = out[home] - out[away]
        away_flag = f"_away_{column}_eligible"
        home_flag = f"_home_{column}_eligible"
        if away_flag in out.columns and home_flag in out.columns:
            away_elig = pd.to_numeric(out[away_flag], errors="coerce")
            home_elig = pd.to_numeric(out[home_flag], errors="coerce")
            out[f"_eligible_{column}_away"] = away_elig
            out[f"_eligible_{column}_home"] = home_elig
            out[f"_eligible_{column}_diff"] = np.minimum(away_elig, home_elig)
    drop = [c for c in out.columns if c.startswith("_home_") or c.startswith("_away_")]
    return out.drop(columns=drop)


def attach_stint_rapm(slate: pd.DataFrame,
                      aggregates: pd.DataFrame) -> pd.DataFrame:
    """Attach the fifteen stint-RAPM features, same grain as the nine.

    Same join, same (date, team) grain, same always-create-the-column rule
    as :func:`attach_position_rapm`. Net keeps sides plus diff (nine);
    offense and defense publish diffs only (six). Eligibility reuses the
    position membership flags: the stint segments read the same members
    with the same weights, so a separate mask would always agree.
    """
    out = slate.copy() if slate is not None else pd.DataFrame()
    for column in STINT_RAPM_FEATURES:
        out[column] = np.nan
    if out.empty or aggregates is None or not len(aggregates):
        return out
    out = _attach_sides(out, aggregates, STINT_RAPM_SOURCES)
    for stem in ("net", "off", "def"):
        for position in _POSITION_PUBLISHED_ORDER:
            src_col = f"pl_srapm_{stem}_{position}"
            away = f"_away_{src_col}"
            home = f"_home_{src_col}"
            if away not in out.columns or home not in out.columns:
                continue
            if stem == "net":
                out[f"{src_col}_away"] = out[away]
                out[f"{src_col}_home"] = out[home]
                out[f"{src_col}_diff"] = out[home] - out[away]
            else:
                out[f"{src_col}_diff"] = out[home] - out[away]
    drop = [c for c in out.columns if c.startswith("_home_")
            or c.startswith("_away_")]
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
