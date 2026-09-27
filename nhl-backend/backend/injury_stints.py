"""Binary injury-status intervals for NHL player-pool features.

The only production injury evidence is a timestamped status-report snapshot.
Exactly Out, Injured Reserve, IR, and Doubtful remove a player; every other
status is available under the approved feature policy. Boxscore appearances and
nonappearances are not injury designations and are never used to construct the
production flag.

Intervals are replayed at the report snapshot's capture timestamp, not the
provider's potentially backdated report date or projected return date. A
successful complete snapshot closes a prior Out/IR/Doubtful interval when the
player is no longer reported Out/IR/Doubtful (including when absent from the
report). At pool construction, a status snapshot must be known strictly before
puck drop. The
interval exclusion is binary and happens after player rating/shrinkage, before
the team/position/situation mean, matching the MLB lineup-wOBA structure.

``appearances_to_stints`` remains only as a legacy availability diagnostic; it
measures roster nonparticipation, which can also mean healthy scratch, roster
change, or other causes. Production injury loading does not call it.
"""
from __future__ import annotations

import math
import re
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config (mirrors mlb-backend/backend/features.py names where they transfer)
# ---------------------------------------------------------------------------
STINT_START = "stint_start"
STINT_END = "stint_end"
OUT_PLAYER = "player_id"
OUT_STATUS = "availability_status"

#: Exact report designations that set the binary injury flag. ``IR`` is the
#: conventional abbreviation for ``Injured Reserve``. Day-To-Day and
#: Suspension remain available; no appearance or missed-game inference is allowed.
IL_STATUSES: frozenset[str] = frozenset(
    {"out", "injured reserve", "ir", "doubtful"})

#: How far the stint table's window may trail the frame being scored before the
#: staleness tripwire fires. Same value as MLB's IL_STINT_MAX_LAG_DAYS: an NHL
#: offseason legitimately leaves a multi-month gap, so the bar is not a day.
STINT_MAX_LAG_DAYS = 45

#: Maximum source-rating age for a candidate in the target-game pool.
#: Keeps stale player/team associations (trades, retirements, injuries) out
#: while covering ordinary rest gaps. MLB-faithful: MLB's own gate is
#: LINEUP_POOL_LOOKBACK_DAYS = 10, and its documented season-boundary answer
#: is the same shape -- a slate whose team ratings all trail the gate serves
#: the position prior for that slate (the NHL offseason is the one such
#: slate per year), then real ratings resume once the season's first games
#: decide. Measured in the 2026-09-29 run: 43,616 decided-pool rows, one
#: 5-game slate on priors.
POOL_LOOKBACK_DAYS = 45

#: Retained for legacy season-grain diagnostics only. Production player-game
#: serving is strictly by source date < target game date.
POOL_SERVE_OFFSET_YEARS = 0

#: Minimum prior situation ice for an eligible player-game candidate. Applied
#: only after the rolling rate is computed; it never modifies historical rows.
MIN_PRIOR_ICE_SECONDS = 900.0

#: How many dressed skaters a (game, team) side must report before its
#: appearance record is allowed to say anybody was ABSENT.
#:
#: 10 is not a hockey number -- a decided game dresses 18 a side (measured:
#: min 17, median 18, max 20 over 2,892 OOF games) -- it is a truncation
#: guard. A side whose boxscore came back partial must not be read as "these
#: 3 skaters played and the other 20 were all injured", because that reading
#: is indistinguishable from a real mass absence and takes the whole team out
#: of the pool on one bad fetch. Below the floor the side is simply not
#: evaluated, which is the honest outcome for missing evidence.
MIN_DRESSED_PER_SIDE = 10

SITUATION_EVO = "5on5"
SITUATION_PPO = "5on4"


# ---------------------------------------------------------------------------
# Stint intervals
# ---------------------------------------------------------------------------
def _is_il_status(status: object) -> bool:
    if status is None or (isinstance(status, float) and math.isnan(status)):
        return False
    text = str(status).strip().lower()
    if not text or text in {"nan", "none", "healthy", "active"}:
        return False
    return text in IL_STATUSES


def _utc_naive(value) -> pd.Timestamp:
    """Parse a timestamp as UTC and drop the timezone for safe comparisons."""
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).tz_localize(None)


def build_stint_intervals(
    reports: pd.DataFrame,
    *,
    player_col: str = OUT_PLAYER,
    status_col: str = "status",
    date_col: str = "report_date",
    return_col: str = "return_date",
    snapshot_col: str = "snapshot_at",
    marker_col: str = "snapshot_marker",
) -> tuple[pd.DataFrame, dict]:
    """Build binary injury intervals from observed status reports.

    Only exact Out, Injured Reserve, IR, and Doubtful labels set the flag.
    Every other reported status sets the player available. Projected
    ``return_date`` is deliberately ignored: the flag clears only when a later
    status snapshot says the player is no longer excluded.

    Production snapshots carry ``snapshot_at`` and a marker row for an empty
    but successful report. Each snapshot is the complete current injury list:
    Out/IR/Doubtful players are unavailable; any other status or absence from that complete
    list is available. Intervals begin/end at capture time, never at a
    retrospective report date, so a report first captured after puck drop
    cannot affect that game. The report-date-only path is retained for pure
    interval tests/legacy callers, but production loading rejects it.
    """
    audit = {
        "report_rows": 0, "players": 0, "stints": 0, "open_stints": 0,
        "closed_on_return_date": 0, "closed_on_clean_report": 0,
        "closed_on_snapshot": 0, "non_il_reports": 0,
        "snapshot_count": 0, "snapshot_based": False,
    }
    empty = pd.DataFrame(columns=[OUT_PLAYER, STINT_START, STINT_END])
    if reports is None or len(reports) == 0:
        return empty, audit

    # Snapshot replay is the production path. It models the report as known at
    # the time it was captured; a provider's backdated report_date is not proof
    # that our predictor had seen the status before puck drop.
    if snapshot_col in reports.columns:
        snapshots = reports.copy()
        snapshots["_snapshot_at"] = snapshots[snapshot_col].map(_utc_naive)
        snapshots = snapshots.dropna(subset=["_snapshot_at"])
        if len(snapshots):
            audit["snapshot_based"] = True
            marker = (snapshots[marker_col].fillna(False).astype(bool)
                      if marker_col in snapshots.columns
                      else pd.Series(False, index=snapshots.index))
            actual = snapshots.loc[~marker].copy()
            if player_col in actual.columns:
                actual = actual.dropna(subset=[player_col])
                actual[OUT_PLAYER] = actual[player_col].astype(str)
            else:
                actual = actual.iloc[:0].copy()
            audit["report_rows"] = int(len(actual))
            audit["players"] = int(actual[OUT_PLAYER].nunique()) if len(actual) else 0
            times = sorted(pd.Timestamp(t) for t in snapshots["_snapshot_at"].unique())
            audit["snapshot_count"] = len(times)
            rows: list[dict] = []
            active: set[str] = set()
            open_start: dict[str, pd.Timestamp] = {}

            for captured_at in times:
                at_snapshot = actual[actual["_snapshot_at"] == captured_at].copy()
                if len(at_snapshot):
                    if status_col not in at_snapshot.columns:
                        out_now: set[str] = set()
                    else:
                        # If a provider emits more than one row per player in
                        # a snapshot, its latest dated record is authoritative.
                        if date_col in at_snapshot.columns:
                            at_snapshot["_report_at"] = at_snapshot[date_col].map(_utc_naive)
                            at_snapshot = at_snapshot.sort_values(
                                [OUT_PLAYER, "_report_at"], kind="mergesort")
                        latest = at_snapshot.drop_duplicates(OUT_PLAYER, keep="last")
                        out_now = set(latest.loc[
                            latest[status_col].map(_is_il_status), OUT_PLAYER].astype(str))
                        audit["non_il_reports"] += int(
                            (~latest[status_col].map(_is_il_status)).sum())
                else:
                    # A marker with no player rows records a successful,
                    # complete empty snapshot: all previously flagged players
                    # are now absent from the report and therefore available.
                    out_now = set()

                for player in active - out_now:
                    rows.append({OUT_PLAYER: player, STINT_START: open_start[player],
                                 STINT_END: captured_at})
                    audit["closed_on_snapshot"] += 1
                    open_start.pop(player, None)
                for player in out_now - active:
                    open_start[player] = captured_at
                active = out_now

            for player in active:
                rows.append({OUT_PLAYER: player, STINT_START: open_start[player],
                             STINT_END: pd.NaT})
                audit["open_stints"] += 1
            out = pd.DataFrame(rows, columns=[OUT_PLAYER, STINT_START, STINT_END])
            if len(out):
                out = out.sort_values([OUT_PLAYER, STINT_START], kind="mergesort") \
                         .reset_index(drop=True)
            audit["stints"] = int(len(out))
            out.attrs["snapshot_based"] = True
            out.attrs["snapshot_times"] = times
            return out, audit

    # Legacy report-date path for deterministic unit tests and callers that
    # provide a trusted event stream. It never uses projected return dates.
    for c in (player_col, status_col, date_col):
        if c not in reports.columns:
            return empty, audit
    r = reports[[player_col, status_col, date_col]].copy()
    r[date_col] = (pd.to_datetime(r[date_col], errors="coerce", utc=True)
                     .dt.tz_localize(None).dt.normalize())
    r = r.dropna(subset=[date_col, player_col])
    r[OUT_PLAYER] = r[player_col].astype(str)
    r = r.sort_values([OUT_PLAYER, date_col], kind="mergesort")
    audit["report_rows"] = int(len(r))
    rows = []
    for player, grp in r.groupby(OUT_PLAYER, sort=False):
        audit["players"] += 1
        open_start: Optional[pd.Timestamp] = None
        for _, row in grp.iterrows():
            date = row[date_col]
            is_il = _is_il_status(row[status_col])
            if not is_il:
                audit["non_il_reports"] += 1
            if open_start is None:
                if is_il:
                    open_start = date
            elif not is_il:
                rows.append({OUT_PLAYER: player, STINT_START: open_start,
                             STINT_END: date})
                audit["closed_on_clean_report"] += 1
                open_start = None
        if open_start is not None:
            rows.append({OUT_PLAYER: player, STINT_START: open_start,
                         STINT_END: pd.NaT})
            audit["open_stints"] += 1

    out = pd.DataFrame(rows, columns=[OUT_PLAYER, STINT_START, STINT_END])
    if len(out):
        out = out.sort_values([OUT_PLAYER, STINT_START], kind="mergesort") \
                 .reset_index(drop=True)
    audit["stints"] = int(len(out))
    return out, audit


def reconcile_against_appearances(
    stints: pd.DataFrame,
    appearances: pd.DataFrame,
    *,
    player_col: str = OUT_PLAYER,
    date_col: str = "game_date",
) -> tuple[pd.DataFrame, dict]:
    """Legacy availability diagnostic; never use this to label injuries.

    An appearance can show participation, but nonparticipation cannot establish
    injury. This helper is retained for diagnostics only and is not called by
    the production injury loader or player-pool builder.
    """
    audit = {"appearances": 0, "stints_in": 0, "stints_shortened": 0,
             "player_games_removed": 0}
    if stints is None or len(stints) == 0:
        return (stints.copy() if stints is not None else
                pd.DataFrame(columns=[OUT_PLAYER, STINT_START, STINT_END])), audit
    audit["stints_in"] = int(len(stints))
    if appearances is None or len(appearances) == 0:
        return stints, audit
    for c in (player_col, date_col):
        if c not in appearances.columns:
            return stints, audit
    app = appearances[[player_col, date_col]].copy()
    app[OUT_PLAYER] = app[player_col].astype(str)
    app["_d"] = pd.to_datetime(app[date_col], errors="coerce")
    app = app.dropna(subset=["_d"])
    audit["appearances"] = int(len(app))
    by_player = {k: np.sort(v.to_numpy()) for k, v in app.groupby(OUT_PLAYER)["_d"]}

    rows = []
    for _, s in stints.iterrows():
        start = pd.to_datetime(s[STINT_START], errors="coerce")
        end = pd.to_datetime(s[STINT_END], errors="coerce") if pd.notna(
            s.get(STINT_END)) else pd.NaT
        dates = by_player.get(s[OUT_PLAYER])
        new_end = end
        if dates is not None and len(dates) and pd.notna(start):
            # First appearance STRICTLY after the placement. A same-date
            # appearance is the announcement, not a return (MLB's rule).
            i = int(np.searchsorted(dates, np.datetime64(start), side="right"))
            if i < len(dates):
                first_back = pd.Timestamp(dates[i])
                if pd.isna(new_end) or first_back < new_end:
                    removed = 1 if pd.isna(new_end) else max(
                        0, int((new_end - first_back).days))
                    audit["player_games_removed"] += removed
                    new_end = first_back
                    audit["stints_shortened"] += 1
        rows.append({OUT_PLAYER: s[OUT_PLAYER], STINT_START: start, STINT_END: new_end})
    out = pd.DataFrame(rows, columns=[OUT_PLAYER, STINT_START, STINT_END])
    return out, audit


# ---------------------------------------------------------------------------
# The exclusion test
# ---------------------------------------------------------------------------
def is_unavailable(
    stints: pd.DataFrame,
    player_id: str,
    game_date,
    *,
    strict_start: bool = False,
    snapshot_times: Optional[Sequence] = None,
    max_snapshot_lag_days: int = STINT_MAX_LAG_DAYS,
) -> bool:
    """Does an Out/IR interval cover the game decision timestamp?

    Production snapshot intervals require ``stint_start < puck_drop``: a
    report captured exactly at puck drop is not proven pregame. For snapshot
    data, both status edges must be strictly before puck drop:
    an Out snapshot at puck drop cannot newly exclude the player, while a clear
    snapshot at puck drop cannot erase an earlier known Out state. Legacy
    date-interval callers keep the usual inclusive-start/exclusive-end rule.
    """
    if stints is None or len(stints) == 0:
        return False
    d = _utc_naive(game_date)
    if pd.isna(d):
        return False
    if strict_start:
        if snapshot_times is None:
            snapshot_times = stints.attrs.get("snapshot_times", ())
        prior_snapshots = [
            stamp for stamp in (_utc_naive(value) for value in snapshot_times)
            if pd.notna(stamp) and stamp < d]
        if not prior_snapshots:
            return False
        latest_snapshot = max(prior_snapshots)
        if (d - latest_snapshot).total_seconds() > max_snapshot_lag_days * 86400:
            return False
    mine = stints[stints[OUT_PLAYER] == str(player_id)]
    for _, s in mine.iterrows():
        start = _utc_naive(s[STINT_START])
        if pd.isna(start) or (start >= d if strict_start else start > d):
            continue
        end = _utc_naive(s[STINT_END])
        # In snapshot mode the clean state represented by the end is only
        # usable when captured strictly before puck drop. At equality the
        # previous Out state remains the latest known pregame state.
        if pd.isna(end) or (end >= d if strict_start else end > d):
            return True
    return False


def build_unavailable_mask(
    ratings: pd.DataFrame,
    stints: pd.DataFrame,
) -> pd.Series:
    """Boolean per RATING ROW: is this player on the IL as of the row's date?

    Evaluated per row rather than per player because the answer is a function of
    the game date — a player unavailable in November is available by April.
    """
    if stints is None or len(stints) == 0:
        return pd.Series(False, index=ratings.index)
    by_player = {p: g for p, g in stints.groupby(OUT_PLAYER)}
    out = []
    for pid, date in zip(ratings[OUT_PLAYER].astype(str), ratings["game_date"]):
        g = by_player.get(pid)
        out.append(False if g is None else is_unavailable(g, pid, date))
    return pd.Series(out, index=ratings.index)


# ---------------------------------------------------------------------------
# The MLB-shaped pool -> team aggregate
# ---------------------------------------------------------------------------
def _expand_to_games(
    agg: pd.DataFrame,
    games: pd.DataFrame,
    game_lookback_days: Optional[int],
    *,
    strict_source_date: bool = False,
) -> pd.DataFrame:
    """Select each target team's latest prior candidate rating efficiently.

    The production path uses an as-of join per player/team/situation/position,
    rather than materialising the much larger team x every historical player-
    game Cartesian product. Calendar dates are conservative: equal source and
    target dates never match because the source feed has no verified publish
    timestamp. Legacy season-grain inputs retain the old serving fallback.
    """
    game_cols = [c for c in ("game_date", "team", "season", "game_id",
                              "start_time_utc") if c in games.columns]
    g = games.loc[:, game_cols].copy()
    g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce").dt.normalize()
    g = g.dropna(subset=["game_date"]).drop_duplicates()
    if g.empty:
        return agg.assign(game_date=pd.Series(dtype="datetime64[ns]")).iloc[:0]

    p = agg.copy()
    p["_rdate"] = pd.to_datetime(p["_cand"], errors="coerce").dt.normalize()
    p = p.dropna(subset=["_rdate"])

    if strict_source_date:
        by = ["team", "situation", "position", OUT_PLAYER]
        p["_source_date"] = pd.to_datetime(
            p.get("_source_date", p["_rdate"]), errors="coerce").dt.normalize()
        p = p.dropna(subset=["_source_date"])
        if p.empty:
            return g.iloc[:0].copy()

        # Cross only target sides with distinct candidate skaters, then use an
        # as-of lookup for the latest source game. This is orders of magnitude
        # smaller than merging every historical game row to every target game.
        roster = p[by].drop_duplicates()
        targets = g.merge(roster, on="team", how="inner", sort=False)
        if targets.empty:
            return targets.assign(
                _source_date=pd.Series(dtype="datetime64[ns]"),
                rate=pd.Series(dtype=float),
                n_players=pd.Series(dtype=float),
                evidence=pd.Series(dtype=float),
                latest=pd.Series(dtype="datetime64[ns]"),
            )
        targets["_target_order"] = np.arange(len(targets), dtype=np.int64)
        source_cols = by + ["_source_date", "rate", "n_players", "evidence", "latest"]
        source = p[source_cols].drop_duplicates(
            by + ["_source_date"], keep="last")
        targets = targets.sort_values(["game_date"] + by, kind="mergesort")
        source = source.sort_values(["_source_date"] + by, kind="mergesort")
        tolerance = (None if game_lookback_days is None else
                     pd.Timedelta(days=max(0, int(game_lookback_days))))
        selected = pd.merge_asof(
            targets, source, left_on="game_date", right_on="_source_date",
            by=by, direction="backward", tolerance=tolerance,
            allow_exact_matches=False)
        selected = selected.dropna(subset=["_source_date"])
        return selected.sort_values("_target_order", kind="mergesort") \
            .drop(columns=["_target_order"]).reset_index(drop=True)

    if "season" in g.columns:
        g["_served"] = pd.to_numeric(g["season"], errors="coerce")
        g = g.dropna(subset=["_served"])
        g["_served"] = g["_served"].astype(int)
    else:
        g["_served"] = g["game_date"].dt.year
    p["_served"] = p["_rdate"].dt.year + POOL_SERVE_OFFSET_YEARS

    g = g.rename(columns={"team": "_team"})
    j = p.merge(g, on="_served", how="inner")
    j = j[j["_team"].astype(str) == j["team"].astype(str)]
    gap = (j["game_date"] - j["_rdate"]).dt.days
    j = j[gap >= 0]
    if game_lookback_days is not None:
        j = j[gap <= int(game_lookback_days)]
    return j.drop(columns=["_served", "_team"])


def team_game_rates(
    ratings: pd.DataFrame,
    *,
    stints: Optional[pd.DataFrame] = None,
    lookback_days: Optional[int] = POOL_LOOKBACK_DAYS,
    min_prior_ice_seconds: float = MIN_PRIOR_ICE_SECONDS,
    ice_col: str = "prior_ice_seconds",
    rate_col: str = "shrunk_rate_per60",
    require_pool: bool = True,
    games: Optional[pd.DataFrame] = None,
    game_lookback_days: Optional[int] = None,
) -> tuple[pd.DataFrame, dict]:
    """Per (game_date, team, situation, position) healthy-pool mean rate.

    This mirrors MLB's ``_LINEUP_AGG_ROSTER`` sequence:

      1. POOL — for each target game, take each player's most recent rolling
         rating whose source game date is strictly earlier and within
         ``lookback_days``. The target date comparison is strict because
         provider rows have date grain, not verified publish timestamps.
      2. MINIMUM EVIDENCE — a player enters the pool only with a non-null rate
         and at least ``min_prior_ice_seconds`` of evidence in that situation. This
         is where the shrinkage's own sample size is enforced at the consumer,
         not just at the estimate.
      3. INJURY EXCLUSION — drop any player with an open Out/IR/Doubtful
         interval captured strictly before exact puck drop. A BINARY removal:
         the player leaves the pool, and the
         rate he carries is never altered. This is why ``games`` exists. MLB's
         pool is materialised on ``SELECT DISTINCT game_date, game_pk, team``
         and the predicate tests ``p.game_date``, so the exclusion is
         evaluated once per game. Evaluating it at the rating's own date
         instead -- the pre-revision behaviour here -- makes the filter unable
         to bind at all when ratings are season grain: every rating for a
         season carries the same date, so a player hurt in November and back
         in December is either in the whole season's pool or out of all of it.
      4. AGGREGATE — the mean rate of the surviving pool.

    MLB then ranks by trailing PA and takes the top 9. Hockey has no such cut:
    every skater who dresses contributes, and there is no ninth-inning
    substitute to model, so the roster is averaged whole. Taking a top-N here
    would import a baseball rule that has no hockey meaning.

    With ``require_pool=False`` or no stints, step 3 is skipped and the result
    is the documented participant-pool fallback — the NHL equivalent of MLB
    falling back to ``_LINEUP_AGG_PARTICIPANTS`` when the IL table is absent.

    ``games`` is MLB's target-game grid: a frame with ``game_date`` and
    ``team``. Production always passes the grid so target-game injuries and
    source-date eligibility are evaluated per game, never against rating dates.
    The production source-rating age is bounded by ``POOL_LOOKBACK_DAYS``;
    ``game_lookback_days`` may override that limit for a caller.
    """
    audit = {
        "pool_rows_in": 0, "pool_rows": 0, "dropped_no_rate": 0,
        "dropped_below_min_ice": 0, "dropped_stale": 0,
        "dropped_unavailable": 0, "teams": 0, "games": 0,
        "il_filter_active": bool(stints is not None and len(stints)),
    }
    if ratings is None or len(ratings) == 0:
        return pd.DataFrame(), audit

    df = ratings.copy()
    df["_d"] = pd.to_datetime(df["game_date"], errors="coerce")
    df[OUT_PLAYER] = df[OUT_PLAYER].astype(str)
    audit["pool_rows_in"] = int(len(df))

    for c in (rate_col, ice_col):
        if c not in df.columns:
            df[c] = np.nan
    df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")
    df[ice_col] = pd.to_numeric(df[ice_col], errors="coerce")

    # Step 2 first: a row with no rate or too little evidence is not a candidate.
    no_rate = df[rate_col].isna()
    audit["dropped_no_rate"] = int(no_rate.sum())
    df = df[~no_rate]
    if require_pool:
        thin = ~(df[ice_col] >= float(min_prior_ice_seconds))
        audit["dropped_below_min_ice"] = int(thin.sum())
        df = df[~thin]
    if len(df) == 0:
        return pd.DataFrame(), audit

    # Step 1: recency. Production rows carry their actual source game date;
    # legacy season-grain rows still use their ``game_date`` as candidate time.
    has_pool_date = "pool_date" in df.columns
    if has_pool_date:
        cand = pd.to_datetime(df["pool_date"], errors="coerce").dt.normalize()
    else:
        cand = df["_d"].dt.normalize()
    df = df.assign(_cand=cand).dropna(subset=["_cand"])

    keys = ["_cand", "team", "situation", "position", OUT_PLAYER]
    agg = (df.groupby(keys, dropna=False)
             .agg(rate=(rate_col, "mean"),
                  n_players=(OUT_PLAYER, "nunique"),
                  evidence=(ice_col, "max"),
                  latest=("_cand", "max"))
             .reset_index())
    if has_pool_date:
        # Carry actual source date separately because ``_cand`` may be a pool
        # timestamp in legacy compatibility rows.
        source_dates = (df.assign(_source_date=df["_d"].dt.normalize())
                          .groupby(keys, dropna=False)["_source_date"].max()
                          .reset_index())
        agg = agg.merge(source_dates, on=keys, how="left", validate="one_to_one")

    # Per-game grid (MLB's `pool` CTE). One rating per PLAYER PER GAME, most
    # recent first-wins -- the dedup key must keep the player in it, and the
    # game once the grid is in play: dropping either collapses a whole team's
    # pool to a single player.
    if games is not None:
        # ``lookback_days`` is the bounded source-rating age for production.
        # ``game_lookback_days`` remains a compatibility override.
        age_limit = (game_lookback_days if game_lookback_days is not None
                     else (lookback_days if has_pool_date else None))
        agg = _expand_to_games(agg, games, age_limit,
                               strict_source_date=has_pool_date)
        player_key = ["game_date", "team"]
        if "game_id" in agg.columns:
            player_key.append("game_id")
        player_key.extend(["situation", "position", OUT_PLAYER])
    else:
        # No grid: a diagnostic-only flat aggregation at the candidate date.
        agg = agg.assign(game_date=agg["_cand"])
        player_key = ["team", "situation", "position", OUT_PLAYER]
    agg = (agg.sort_values(player_key + ["latest"], kind="mergesort")
           .drop_duplicates(player_key, keep="last")
           .reset_index(drop=True))

    audit["pool_rows"] = int(len(agg))

    # Step 3: the binary injury exclusion, evaluated at the GAME date (MLB) or,
    # with no grid, at the rating's own date (a pre-season-dated fallback).
    dropped_by_side = None
    if require_pool and stints is not None and len(stints):
        strict_start = bool(stints.attrs.get("snapshot_based", False))
        if strict_start:
            # A capture timestamp is meaningful only relative to exact puck
            # drop. Missing/invalid start times do not prove that the report
            # was known before the decision, so such rows stay unfiltered.
            at = ([ _utc_naive(start) for start in agg["start_time_utc"]]
                  if "start_time_utc" in agg.columns
                  else [pd.NaT] * len(agg))
            audit["missing_decision_time"] = int(
                sum(pd.isna(value) for value in at))
        elif "start_time_utc" in agg.columns:
            at = [start if pd.notna(_utc_naive(start)) else game_date
                  for start, game_date in zip(agg["start_time_utc"],
                                              agg["game_date"])]
        else:
            at = agg["game_date"] if "game_date" in agg.columns else agg["_cand"]
        snapshot_times = stints.attrs.get("snapshot_times", ())
        mask = pd.Series(
            [is_unavailable(stints, p, d, strict_start=strict_start,
                            snapshot_times=snapshot_times)
             for p, d in zip(agg[OUT_PLAYER], at)],
            index=agg.index)
        audit["dropped_unavailable"] = int(mask.sum())
        # Counted PER SIDE, because "how much of this team's pool survived the
        # injury filter" is a per-game question and a frame total answers a
        # different one: a single number for the whole run is constant on every
        # row, which is a receipt that looks like a feature and is not one.
        if "game_date" in agg.columns and audit["dropped_unavailable"]:
            side_key = ["game_date", "team"]
            if "game_id" in agg.columns:
                side_key.insert(0, "game_id")
            dropped_by_side = (agg[mask].groupby(side_key, dropna=False)
                               .size().rename("n_unavailable").reset_index())
        agg = agg[~mask]

    if len(agg) == 0:
        return pd.DataFrame(), audit

    group_keys = ["game_date", "team"]
    if "game_id" in agg.columns:
        group_keys.append("game_id")
    group_keys.extend(["situation", "position"])
    out = (agg.groupby(group_keys, dropna=False)
             .agg(rate=("rate", "mean"),
                  n_players=("n_players", "sum"),
                  evidence=("evidence", "sum"))
             .reset_index())
    if dropped_by_side is not None:
        side_key = ["game_date", "team"]
        if "game_id" in out.columns and "game_id" in dropped_by_side.columns:
            side_key.insert(0, "game_id")
        out = out.merge(dropped_by_side, on=side_key, how="left")
        out["n_unavailable"] = out["n_unavailable"].fillna(0).astype(int)
        out["n_candidates"] = out["n_players"] + out["n_unavailable"]
        audit["sides_touched"] = int((out["n_unavailable"] > 0).sum())
    else:
        out["n_unavailable"] = 0
        out["n_candidates"] = out["n_players"]
        audit["sides_touched"] = 0
    audit["teams"] = int(out["team"].nunique())
    audit["games"] = int(out["game_id"].nunique()
                         if "game_id" in out.columns
                         else out["game_date"].nunique())
    return out, audit


# ---------------------------------------------------------------------------
# Tripwires
# ---------------------------------------------------------------------------
def staleness_report(
    stints: pd.DataFrame,
    decided_max_date,
    window_end,
    *,
    max_lag_days: int = STINT_MAX_LAG_DAYS,
) -> dict:
    """How far the stint table trails the frame being scored.

    An open stint carries no end, so ``max(stint_start)`` says nothing about
    freshness — the same reasoning as MLB, where the builder's own meta sidecar
    records the window it actually fetched through. Without a sidecar this
    returns ``{}``, because a missing sidecar is not a second failure mode worth
    inventing. Timestamps are normalized to UTC before comparing.
    """
    if window_end is None or decided_max_date is None:
        return {}
    decided = _utc_naive(decided_max_date)
    fetched = _utc_naive(window_end)
    if pd.isna(decided) or pd.isna(fetched):
        return {}
    lag = int((decided - fetched).total_seconds() // 86400)
    return {
        "window_end": str(fetched.date()),
        "decided_max_date": str(decided.date()),
        "lag_days": lag,
        "stale": bool(lag > max_lag_days),
        "n_stints": 0 if stints is None else int(len(stints)),
    }


def bind_check(ratings: pd.DataFrame, stints: pd.DataFrame) -> dict:
    """MLB's id-space overlap guard, as a returned verdict rather than a log.

    MLB logs an ERROR when the IL table shares no ids with the ratings table,
    because the filter then cannot bind and every "healthy" lineup is really the
    unfiltered pool — indistinguishable from correct output. The caller is
    expected to treat ``ok=False`` as a build failure, not a warning.
    """
    n_stints = 0 if stints is None else int(len(stints))
    if n_stints == 0:
        return {"ok": True, "stints": 0, "matched": 0, "reason": "no stints"}
    if ratings is None or len(ratings) == 0 or OUT_PLAYER not in ratings.columns:
        return {"ok": False, "stints": n_stints, "matched": 0,
                "reason": "ratings frame has no player id"}
    ids = set(ratings[OUT_PLAYER].astype(str))
    matched = len({str(p) for p in stints[OUT_PLAYER]} & ids)
    return {
        "ok": bool(matched > 0), "stints": n_stints, "matched": matched,
        "match_rate": round(matched / max(1, len(set(stints[OUT_PLAYER]))), 4),
        "reason": "bound" if matched else
                  "the injury table shares NO player ids with the ratings: the "
                  "filter cannot bind and every pool is silently unfiltered",
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Legacy appearance-based availability diagnostics
# ---------------------------------------------------------------------------
def appearances_to_stints(
    appearances: pd.DataFrame,
    ratings: pd.DataFrame,
    games: pd.DataFrame,
    *,
    game_date_col: str = "game_date",
    team_col: str = "team",
    player_col: str = OUT_PLAYER,
    min_dressed_per_side: int = MIN_DRESSED_PER_SIDE,
) -> tuple[pd.DataFrame, dict]:
    """Legacy diagnostic: derive roster-nonparticipation intervals.

    This measures whether a rated skater dressed in a prior game. It does not
    distinguish injury from a healthy scratch, suspension, roster move, or
    missing boxscore, so it is not an injury classifier and must not feed the
    production injury flag. The production loader uses timestamped ESPN status
    snapshots and this helper remains only for isolated availability analysis.

    The legacy diagnostic shifts observed absences to the next game (never the
    game whose boxscore reveals them), and treats a missing or truncated side
    as unknown rather than as league-wide nonparticipation.
    """
    audit = {"stints": 0, "open_stints": 0, "players": 0, "absences": 0,
             "runs": 0, "sides": 0, "sides_observed": 0, "sides_unobserved": 0,
             "sides_with_absence": 0, "players_removed_total": 0,
             "mean_removed_per_game": 0.0}
    empty = pd.DataFrame(columns=[OUT_PLAYER, STINT_START, STINT_END])
    if appearances is None or ratings is None or games is None:
        return empty, audit
    if len(games) == 0 or len(ratings) == 0 or len(appearances) == 0:
        return empty, audit
    missing_cols = [c for c in (game_date_col, team_col, player_col)
                    if c not in appearances.columns]
    if missing_cols:
        # A frame that does not say who dressed says nothing about who did
        # not. Guessing at a column name here would turn a schema drift into
        # a silent league-wide absence.
        audit["reason"] = f"appearance frame missing {missing_cols}"
        return empty, audit

    # Who COULD have played each side: the same (season, team, player) pool the
    # exclusion is applied to, so a player with no rating row is not "absent",
    # he simply is not in the pool to remove.
    r = ratings.copy()
    r["_d"] = pd.to_datetime(r["game_date"], errors="coerce").dt.normalize()
    r["served"] = r["_d"].dt.year + POOL_SERVE_OFFSET_YEARS
    r[player_col] = r[player_col].astype(str)
    r[team_col] = r[team_col].astype(str)
    roster_by_side: dict[tuple, set] = {}
    for s, t, p in zip(r["served"], r[team_col], r[player_col]):
        roster_by_side.setdefault((int(s), t), set()).add(p)
    audit["players"] = len({p for v in roster_by_side.values() for p in v})

    # Each team's own game list, in order. The index into THIS list is what
    # "the next game" means -- a calendar day is not, because a team plays on
    # roughly every other day and D[i+1] is 2-3 days later.
    g = games[[c for c in (game_date_col, team_col, "season")
               if c in games.columns]].copy()
    g[game_date_col] = pd.to_datetime(g[game_date_col],
                                      errors="coerce").dt.normalize()
    g[team_col] = g[team_col].astype(str)
    if "season" in g.columns:
        g["served"] = (pd.to_numeric(g["season"], errors="coerce")
                       .fillna(0).astype(int))
    else:
        g["served"] = g[game_date_col].dt.year + POOL_SERVE_OFFSET_YEARS
    g = g.dropna(subset=[game_date_col]).drop_duplicates(
        [game_date_col, team_col]).sort_values([team_col, game_date_col],
                                               kind="mergesort")
    per_team: dict[str, list] = {}
    for t, grp in g.groupby(team_col, sort=False):
        per_team[t] = [(d, int(s)) for d, s in zip(grp[game_date_col], grp["served"])]

    # Who dressed, per (date, team) side. Kept as per-side sets rather than one
    # flat (date, team, player) tuple set because the truncation floor below is
    # a per-side test and a flat set cannot answer it.
    a = appearances.copy()
    a[game_date_col] = pd.to_datetime(a[game_date_col],
                                      errors="coerce").dt.normalize()
    a = a.dropna(subset=[game_date_col])
    a[team_col] = a[team_col].astype(str)
    a[player_col] = a[player_col].astype(str)
    dressed_by_side: dict[tuple, set] = {}
    for d, t, p in zip(a[game_date_col], a[team_col], a[player_col]):
        dressed_by_side.setdefault((d, t), set()).add(p)

    audit["sides"] = sum(len(v) for v in per_team.values())
    rows: list[dict] = []
    removed_by_day: dict = {}

    for team, glist in per_team.items():
        # The timeline is the team's EVALUABLE sides: a side with no pool for
        # its served season carries no roster to compare against, and leaving
        # it in the list would make "the previous game" a different game than
        # the index suggests.
        dates, pools = [], []
        for d, served in glist:
            elig = roster_by_side.get((served, team))
            if elig:
                dates.append(d)
                pools.append(elig)
        n = len(dates)
        if n == 0:
            continue

        observed, dressed = [], []
        for d in dates:
            here = dressed_by_side.get((d, team), set())
            ok = len(here) >= int(min_dressed_per_side)
            observed.append(ok)
            dressed.append(here if ok else set())
        audit["sides_observed"] += sum(observed)
        audit["sides_unobserved"] += n - sum(observed)

        roster = set().union(*pools)
        side_misses = [0] * n
        for pid in sorted(roster):
            in_pool = [pid in p for p in pools]
            # A MISS is a positive, dated observation. An unobserved side is
            # not a miss -- it is silence, and silence must not read as a
            # confirmed absence.
            missed = [in_pool[m] and observed[m] and pid not in dressed[m]
                      for m in range(n)]
            # The STATE at game m, read from game m-1 and strictly earlier.
            # m == 0 has no prior game, so it is available by default: the
            # first game of a window cannot know anything about the roster.
            unavail = [in_pool[m] and m > 0 and missed[m - 1] for m in range(n)]

            for m in range(n):
                if missed[m]:
                    side_misses[m] += 1
                    audit["absences"] += 1

            i = 0
            while i < n:
                if not unavail[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < n and unavail[j + 1]:
                    j += 1
                audit["runs"] += 1
                start = dates[i]
                end = dates[j + 1] if j + 1 < n else None
                rows.append({OUT_PLAYER: pid, STINT_START: start,
                             STINT_END: end})
                audit["stints"] += 1
                audit["open_stints"] += int(end is None)
                span = j - i + 1
                audit["players_removed_total"] += span
                for m in range(i, j + 1):
                    removed_by_day[dates[m]] = removed_by_day.get(dates[m], 0) + 1
                i = j + 1

        audit["sides_with_absence"] += sum(1 for c in side_misses if c)

    out = pd.DataFrame(rows, columns=[OUT_PLAYER, STINT_START, STINT_END])
    audit["mean_removed_per_game"] = round(
        audit["players_removed_total"] / max(1, len(removed_by_day)), 3)
    return out, audit


# Player identity: ESPN athlete id -> MoneyPuck rating id
# ---------------------------------------------------------------------------
#: Generational suffixes dropped before matching. "Carey Price Jr" and
#: "Carey Price" are the same player; keeping the suffix splits them.
_NAME_SUFFIXES = ("jr", "sr", "ii", "iii", "iv", "v")


def normalise_player_name(name: object) -> str:
    """Fold a display name to a comparison key.

    MoneyPuck and ESPN both write "A.J. Greer" and "Adam Edstrom" (no
    diacritic), but the two feeds disagree on accents, apostrophes, hyphens
    and generational suffixes often enough that exact string equality loses
    about a quarter of the league. Folding is deliberately aggressive on
    FORM and never on IDENTITY: it removes marks, punctuation and case, and
    nothing else. It does not shorten to initials, because "Alex Lee" and
    "Anthony Lee" would then collide and a collision here removes the WRONG
    player from the pool — strictly worse than removing nobody.
    """
    if name is None:
        return ""
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("'", "").replace("`", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    parts = [p for p in s.split() if p]
    while parts and parts[-1] in _NAME_SUFFIXES:
        parts.pop()
    return " ".join(parts)


def map_reports_to_rating_ids(
    reports: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    name_col: str = "player_name",
) -> tuple[pd.DataFrame, dict]:
    """Rewrite the reports' player ids into the RATING id space.

    The injury feed and the rating feed are different id spaces — ESPN athlete
    ids and MoneyPuck player ids share nothing — so without this the exclusion
    predicate compares two disjoint vocabularies and returns False for every
    player on every date. That is the silent no-op ``bind_check`` exists to
    catch, and the only fix is a mapping.

    The mapping is by NORMALISED NAME, and it is deliberately conservative in
    the one direction that matters:

    * a name matching exactly one rating id is accepted;
    * a name matching SEVERAL rating ids is REFUSED, because picking one would
      remove an arbitrary player from the pool, and a wrong exclusion is worse
      than no exclusion;
    * a name matching none is left carrying its ESPN id, which will simply
      never bind — visible in the audit rather than guessed at.

    Team is NOT used as a tie-breaker here even though both feeds carry it:
    an injured player's team may legitimately have changed, and using it would
    make a mid-season move silently unmatch a player who is otherwise a
    certain match.
    """
    audit = {"report_rows": 0, "matched": 0, "ambiguous": 0, "unmatched": 0,
             "match_rate": 0.0, "ratings_names": 0}
    if reports is None or len(reports) == 0:
        return reports, audit
    if name_col not in reports.columns or ratings is None or len(ratings) == 0:
        audit["report_rows"] = int(len(reports))
        audit["unmatched"] = audit["report_rows"]
        return reports, audit
    if OUT_PLAYER not in ratings.columns:
        return reports, audit

    rname = "player_name" if "player_name" in ratings.columns else name_col
    key_to_ids: dict[str, set[str]] = {}
    for nm, pid in zip(ratings[rname], ratings[OUT_PLAYER]):
        k = normalise_player_name(nm)
        if k:
            key_to_ids.setdefault(k, set()).add(str(pid))
    audit["ratings_names"] = len(key_to_ids)

    out = reports.copy()
    out["_espn_player_id"] = out[OUT_PLAYER].astype(str)
    out["_match_key"] = [normalise_player_name(n) for n in out[name_col]]
    resolved, ambig, unres = [], 0, 0
    for k in out["_match_key"]:
        ids = key_to_ids.get(k, set())
        if len(ids) == 1:
            resolved.append(next(iter(ids)))
        elif len(ids) > 1:
            resolved.append(None)
            ambig += 1
        else:
            resolved.append(None)
            unres += 1
    out[OUT_PLAYER] = [r if r is not None else e
                       for r, e in zip(resolved, out["_espn_player_id"])]
    out = out.drop(columns=["_match_key"])

    audit["report_rows"] = int(len(out))
    audit["matched"] = int(len(out) - ambig - unres)
    audit["ambiguous"] = ambig
    audit["unmatched"] = unres
    audit["match_rate"] = round(
        audit["matched"] / max(1, len(out)), 4)
    return out, audit


class PitViolation(RuntimeError):
    """The injury filter cannot be trusted point-in-time. Fail the build.

    Named and distinct from ``AssertionError`` so a caller can catch a data
    integrity failure without swallowing a genuine logic bug, and so the
    production pipeline can map it onto its own build-failure path.
    """


#: The independently checked conditions under which the report-snapshot
#: exclusion is safe to use.
PIT_CONDITIONS = ("bind", "fresh", "snapshot_based")


def assert_pit(
    ratings: pd.DataFrame,
    stints: pd.DataFrame,
    *,
    decided_max_date=None,
    window_end=None,
    snapshot_based: bool = False,
    max_lag_days: int = STINT_MAX_LAG_DAYS,
) -> dict:
    """Fail CLOSED unless the injury filter is point-in-time sound.

    ``bind_check`` is necessary but NOT sufficient. Binding only proves the
    filter can remove somebody; it does not prove the report state was captured
    before puck drop. Three conditions are checked here:

    ``bind``
        The stint table shares ids with the ratings. Otherwise the filter is
        inert and every "healthy" pool is the unfiltered pool.
    ``fresh``
        The latest captured snapshot trails the latest game date by no more
        than ``max_lag_days``. ``window_end`` is the actual capture time, never
        a report or return date.
    ``snapshot_based``
        Intervals came from timestamped full-report snapshots, never a
        backdated report date, projected return date, or appearance inference.
        The pool predicate separately requires the interval start to be
        strictly before the supplied puck-drop timestamp.

    Returns the verdicts on success. Raises ``PitViolation`` otherwise, naming
    the first failed condition and what would fix it.
    """
    verdicts: dict[str, dict] = {}

    verdicts["bind"] = bind_check(ratings, stints)
    if not verdicts["bind"]["ok"]:
        raise PitViolation(
            "IL pit gate [bind]: " + str(verdicts["bind"]["reason"]) +
            ". Build a name-to-id bridge from the injury feed onto the rating "
            "ids; until then the exclusion must be treated as absent, not as a "
            "filter that happened to remove nobody.")

    verdicts["fresh"] = staleness_report(
        stints, decided_max_date, window_end, max_lag_days=max_lag_days)
    if not verdicts["fresh"]:
        raise PitViolation(
            "IL pit gate [fresh]: no archive window is known, so freshness "
            "cannot be established. Pass window_end as the FETCH date from a "
            "meta sidecar — never max(stint_end), which is a projected return "
            "and reports a dead archive as fresh.")
    if verdicts["fresh"]["stale"]:
        raise PitViolation(
            f"IL pit gate [fresh]: the archive was fetched only through "
            f"{verdicts['fresh']['window_end']} but the frame runs to "
            f"{verdicts['fresh']['decided_max_date']} "
            f"({verdicts['fresh']['lag_days']} days > {max_lag_days}). Stints "
            f"placed in the gap are invisible, so every 'healthy' pool is "
            f"quietly stale.")
    # Uniform verdict shape: every condition reports ``ok``, including the two
    # that are attestations rather than measurements. A caller that filters
    # ``v[c]["ok"]`` must not have to know which is which.
    verdicts["fresh"]["ok"] = True

    raw_snapshot_times = stints.attrs.get("snapshot_times", ())
    parsed_snapshot_times = sorted({
        _utc_naive(value) for value in raw_snapshot_times
        if pd.notna(_utc_naive(value))})
    has_snapshot_history = bool(
        snapshot_based and (parsed_snapshot_times
                           or stints.attrs.get("snapshot_history", False)))
    verdicts["snapshot_based"] = {
        "ok": has_snapshot_history,
        "snapshot_count": len(parsed_snapshot_times),
        "first_snapshot": (str(parsed_snapshot_times[0])
                           if parsed_snapshot_times else None),
        "last_snapshot": (str(parsed_snapshot_times[-1])
                          if parsed_snapshot_times else None),
    }
    if not has_snapshot_history:
        raise PitViolation(
            "IL pit gate [snapshot_based]: captured full-report snapshot "
            "timestamps are missing. Backdated report dates, projected return "
            "dates, and postgame appearances cannot establish what was known "
            "before puck drop.")

    return verdicts
