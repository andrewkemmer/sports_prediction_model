"""NFL data ingestion — nflverse pull with local parquet caching.

Pulls the schedules and play-by-play the pipeline needs through
``nflreadpy`` and caches each season's frame beside the repo (never inside
the git tree). Deterministic column narrowing;regular-season and postseason filtering happens here so every downstream module sees the eligible population only.

All frames carry the nflverse column names; the feature engine is the only
place that interprets them.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

try:  # package-relative import when used as backend.ingestion
    from backend import config  # type: ignore
except ImportError:  # running as a top-level module
    import config

logger = logging.getLogger(__name__)

# Cache directory: OUTSIDE the git tree (repo root's parent) so caches never
# pollute the working tree; overridable for tests.
CACHE_DIR = Path(config.ROOT_DIR.parent) / ".nfl_cache"

# MLB parity: MLB walks its StatsAPI schedule in SCHEDULE_CHUNK_DAYS = 60
# windows (mlb-backend/backend/results.py) and its Statcast pull in the same
# 60-day granularity (ingestion._chunked_statcast), so a decade-long window
# stays rate-limit friendly and a partially completed run is legible from the
# log alone. NFL's nflverse loaders are per-SEASON, so this constant is the
# same 60-day reporting granularity applied to the ingestion window rather
# than a different fetch plan — see chunk_date_range.
POPULATE_CHUNK_DAYS = 60


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def clear_cache() -> None:
    """Remove all cached nflverse parquet artifacts."""
    if CACHE_DIR.exists():
        for path in CACHE_DIR.glob("*.parquet"):
            path.unlink(missing_ok=True)


def _polars_to_pandas(frame):
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    return frame


def eligible_games(schedule: pd.DataFrame) -> pd.DataFrame:
    """Filter a schedules frame to the eligible population: settled
    regular-season or postseason games within the configured season window (2018+)."""
    df = schedule.copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df[df["season"].isin(config.ALL_SEASONS)]
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(config.GAME_TYPES)]
    for c in ("home_score", "away_score"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def load_schedule(seasons: list[int] | None = None,
                  use_cache: bool = True) -> pd.DataFrame:
    """ nflverse schedules for the given seasons (default: config window)."""
    seasons = seasons or config.ALL_SEASONS
    tag = "_".join(str(s) for s in sorted(seasons))
    path = _cache_path(f"schedules_{SCHEDULE_CACHE_VERSION}_{tag}.parquet")
    if use_cache and path.exists():
        return pd.read_parquet(path)
    from nflreadpy import load_schedules
    df = _polars_to_pandas(load_schedules(seasons))
    keep = [c for c in (
        "game_id", "season", "week", "game_type", "gameday", "gametime",
        "home_team", "away_team", "home_score", "away_score", "roof",
        "div_game", "stadium", "surface", "location", "referee",
        # Venue/schedule metadata only. Raw observed weather is intentionally
        # excluded: it lacks a PIT publication timestamp. Production weather
        # is fetched separately from hourly Open-Meteo with full provenance.
        "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name",
    ) if c in df.columns]
    df = df[keep]
    df.to_parquet(path, index=False)
    return df


# "v3" removes observed temp/wind from the production schedule cache. The
# PIT Open-Meteo provider is separate and does not depend on these columns.
SCHEDULE_CACHE_VERSION = "v3"


PBP_NEEDS = [
    "game_id", "posteam", "defteam", "yards_gained", "epa", "qb_epa",
    "game_seconds_remaining", "interception", "fumble_lost",
    "passing_yards", "pass_attempt", "sack", "penalty", "penalty_yards",
    "penalty_team", "third_down_converted", "third_down_failed",
    "yardline_100", "touchdown", "field_goal_result", "drive",
    # 2026-09-22 candidate-pool expansion (pbp feature family): passing depth /
    # accuracy-over-expectation, formation tendency, drive ordering, and the
    # TD-attribution column the red-zone rollup guards with. 2026-09-23:
    # yac_epa replaces the never-available raw "yac" (nflreadpy does not
    # publish it) — separation is served as EPA per attempt instead.
    "air_yards", "yac_epa", "cpoe", "shotgun", "no_huddle", "play_id", "td_team",
    # 2026-09-23 platoon/situational expansion: quarter, down/distance and
    # scoring context for the two-minute / fourth-down / close-game rollups
    # (features._add_pbp_metrics). personnel_o/d are NOT published by the
    # nflreadpy pbp release — charting structure rides the ftn family instead.
    "qtr", "down", "ydstogo", "goal_to_go", "score_differential",
    "half_seconds_remaining", "play_type",
]

# Cache schema version for the pbp parquets. Bump whenever PBP_NEEDS widens:
# the per-season caches store the NARROWED frame, so a previously cached
# season would otherwise keep serving the old column set (features built from
# the missing columns would degrade to all-NaN and read like evidence).
# "v1" = the original 21-column set; "v2" adds the candidate-pool columns;
# "v3" swaps the never-available raw "yac" for "yac_epa" (the YAC-as-EPA
# decomposition nflreadpy actually publishes); "v4" adds the situational
# columns (qtr/down/ydstogo/goal_to_go/score_differential/half_seconds_
# remaining) behind the platoon rollups.
PBP_CACHE_VERSION = "v4"


def load_pbp(seasons: list[int] | None = None,
             use_cache: bool = True, progress=None) -> pd.DataFrame | None:
    """ nflverse play-by-play narrowed to the columns the rollup needs.

    Per-season parquet caches (a season without published PBP — e.g. the
    in-progress season — is warned and skipped, never fatal; downstream
    features degrade to NaN per the documented missing-value policy).
    Returns None only when NO season could be loaded."""
    seasons = seasons or config.ALL_SEASONS
    frames: list[pd.DataFrame] = []
    missing: list[int] = []
    for season in seasons:
        try:
            path = _cache_path(f"pbp_{PBP_CACHE_VERSION}_{season}.parquet")
            if use_cache and path.exists():
                try:
                    frames.append(pd.read_parquet(path))
                    continue
                except Exception as exc:  # corrupt cache → re-pull
                    logger.warning("pbp cache %s unreadable (%s)", path.name, exc)
            missing.append(season)
        finally:
            # In a finally so a season that raises still advances the bar:
            # "unreadable cache" and "network error" are exactly the stalls a
            # progress bar exists to surface, so they must not freeze it.
            if progress is not None:
                progress()
    if missing:
        try:
            from nflreadpy import load_pbp
        except Exception as exc:  # noqa: BLE001
            logger.error("nflreadpy unavailable: %s", exc)
            return None
        for season in missing:
            logger.info("loading pbp season %s", season)
            try:
                df = _polars_to_pandas(load_pbp(season))
            except Exception as exc:  # noqa: BLE001
                logger.warning("pbp unavailable for %s: %s", season, exc)
                continue
            keep = [c for c in PBP_NEEDS if c in df.columns]
            df = df[keep]
            df.to_parquet(_cache_path(f"pbp_{PBP_CACHE_VERSION}_{season}.parquet"),
                          index=False)
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Player stats / Next-Gen Stats — the skill inputs
# (each narrowed at load, cached per season, degrading to NaN downstream)
# ---------------------------------------------------------------------------
# NOTE: the injury-report loader (INJ_* / load_injuries) was removed together
# with the four inj_*_out_diff features. The source stopped publishing
# per-report timestamps after 2024, so those flags could not be populated for
# every game without either leaking a week-level status that postdates kickoff
# or accepting a permanently NaN slate. Availability is now expressed by the
# expected-participation feature family, derived from strictly-prior games.

PS_NEEDS = [
    "game_id", "team", "position", "carries", "rushing_yards", "targets",
    "receptions", "receiving_yards", "receiving_tds", "season_type",
]

# Weekly per-player tracking efficiency (week-0 rows are SEASON aggregates —
# they mix future games into a week-1 value, so they are dropped at load).
NGS_GROUPS = ("passing", "rushing", "receiving")
NGS_NEEDS = {
    "passing": ["season", "week", "team_abbr", "player_position", "attempts",
                "completion_percentage_above_expectation"],
    "rushing": ["season", "week", "team_abbr", "player_position", "rush_attempts",
                "rush_yards_over_expected_per_att"],
    "receiving": ["season", "week", "team_abbr", "player_position", "targets",
                  "avg_separation"],
}


def load_player_stats(seasons: list[int] | None = None,
                      use_cache: bool = True, progress=None) -> pd.DataFrame | None:
    """nflverse weekly player stats narrowed to the usage rollup needs.

    Per-season parquet caches (PS cache v1); a failed season is warned and
    skipped, never fatal. Returns None only when NO season could be loaded."""
    seasons = seasons or config.ALL_SEASONS
    frames: list[pd.DataFrame] = []
    for season in seasons:
        try:
            path = _cache_path(f"ps_v1_{season}.parquet")
            if use_cache and path.exists():
                try:
                    frames.append(pd.read_parquet(path))
                    continue
                except Exception as exc:  # corrupt cache → re-pull
                    logger.warning("player-stats cache %s unreadable (%s)", path.name, exc)
        finally:
            if progress is not None:
                progress()
        try:
            from nflreadpy import load_player_stats
            logger.info("loading player stats season %s", season)
            df = _polars_to_pandas(load_player_stats(season))
        except Exception as exc:  # noqa: BLE001
            logger.warning("player stats unavailable for %s: %s", season, exc)
            continue
        keep = [c for c in PS_NEEDS if c in df.columns]
        df = df[keep]
        df.to_parquet(path, index=False)
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_nextgen(seasons: list[int] | None = None,
                 use_cache: bool = True, progress=None) -> pd.DataFrame | None:
    """nflverse Next-Gen Stats weekly tracking efficiency, all three groups.

    Week-0 rows (season aggregates that would leak future performance into
    early-game features) are dropped at load. Per-(season, group) caches
    (NGS cache v2 = rush_attempts weight fix); a failed season/group is
    warned and skipped. Returns None only when NOTHING could be loaded."""
    seasons = seasons or config.ALL_SEASONS
    frames: list[pd.DataFrame] = []
    for season in seasons:
        for group in NGS_GROUPS:
            try:
                path = _cache_path(f"ngs_v2_{season}_{group}.parquet")
                if use_cache and path.exists():
                    try:
                        frames.append(pd.read_parquet(path))
                        continue
                    except Exception as exc:
                        logger.warning("ngs cache %s unreadable (%s)", path.name, exc)
            finally:
                # One unit per (season, group): NGS iterates three groups, so
                # the bar denominator must count all three or it stops short.
                if progress is not None:
                    progress()
            try:
                from nflreadpy import load_nextgen_stats
                logger.info("loading ngs %s season %s", group, season)
                df = _polars_to_pandas(load_nextgen_stats(season, group))
            except Exception as exc:  # noqa: BLE001
                logger.warning("ngs %s unavailable for %s: %s", group, season, exc)
                continue
            need = NGS_NEEDS[group]
            keep = [c for c in need if c in df.columns]
            df = df[keep]
            if "week" in df.columns:
                df = df[pd.to_numeric(df["week"], errors="coerce").fillna(0) > 0]
            df.to_parquet(path, index=False)
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

# Snap counts: per (game, team, player, position) offense/defense snap
# totals + shares. Available 2013+ (full coverage of the configured window;
# a season without the endpoint is warned and skipped). Cache SNAPS cache v1.
SNAPS_NEEDS = ["game_id", "season", "week", "team", "opponent", "position",
               "offense_snaps", "offense_pct", "defense_snaps", "defense_pct"]


def load_snap_counts(seasons: list[int] | None = None,
                     use_cache: bool = True, progress=None) -> pd.DataFrame | None:
    """nflverse snap counts narrowed to the participation rollup needs.

    Per-season parquet caches (SNAPS cache v1); a failed season is warned
    and skipped, never fatal. Returns None only when NO season loaded."""
    seasons = seasons or config.ALL_SEASONS
    frames: list[pd.DataFrame] = []
    for season in seasons:
        try:
            path = _cache_path(f"snaps_v1_{season}.parquet")
            if use_cache and path.exists():
                try:
                    frames.append(pd.read_parquet(path))
                    continue
                except Exception as exc:  # corrupt cache → re-pull
                    logger.warning("snap-counts cache %s unreadable (%s)", path.name, exc)
        finally:
            if progress is not None:
                progress()
        try:
            from nflreadpy import load_snap_counts
            logger.info("loading snap counts season %s", season)
            df = _polars_to_pandas(load_snap_counts(season))
        except Exception as exc:  # noqa: BLE001
            logger.warning("snap counts unavailable for %s: %s", season, exc)
            continue
        keep = [c for c in SNAPS_NEEDS if c in df.columns]
        df = df[keep]
        df.to_parquet(path, index=False)
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# FTN charting: per-play offensive structure/tendency flags (box count,
# backfield, motion, play action, RPO, screen). Published 2022+; earlier
# seasons are simply unavailable (the ftn candidate family degrades to NaN
# there — the documented policy). Cache FTN cache v1.
FTN_NEEDS = ["nflverse_game_id", "nflverse_play_id", "season", "week",
             "n_defense_box", "n_offense_backfield", "is_motion",
             "is_play_action", "is_rpo", "is_screen_pass"]


def load_ftn_charting(seasons: list[int] | None = None,
                      use_cache: bool = True, progress=None) -> pd.DataFrame | None:
    """nflverse FTN charting narrowed to the platoon rollup needs.

    Per-season parquet caches (FTN cache v1); seasons outside the 2022+
    publication window (and any failed season) are warned and skipped,
    never fatal. Returns None only when NO season loaded."""
    seasons = seasons or config.ALL_SEASONS
    frames: list[pd.DataFrame] = []
    for season in seasons:
        try:
            path = _cache_path(f"ftn_v1_{season}.parquet")
            if use_cache and path.exists():
                try:
                    frames.append(pd.read_parquet(path))
                    continue
                except Exception as exc:  # corrupt cache → re-pull
                    logger.warning("ftn cache %s unreadable (%s)", path.name, exc)
        finally:
            if progress is not None:
                progress()
        try:
            from nflreadpy import load_ftn_charting
            logger.info("loading ftn charting season %s", season)
            df = _polars_to_pandas(load_ftn_charting(season))
        except Exception as exc:  # noqa: BLE001
            logger.warning("ftn charting unavailable for %s: %s", season, exc)
            continue
        keep = [c for c in FTN_NEEDS if c in df.columns]
        df = df[keep]
        df.to_parquet(path, index=False)
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_team_names() -> dict[str, str]:
    """team abbr -> full team name (frontend games[] display fields)."""
    path = _cache_path("teams.parquet")
    if path.exists():
        df = pd.read_parquet(path)
    else:
        from nflreadpy import load_teams
        df = _polars_to_pandas(load_teams())
        df.to_parquet(path, index=False)
    if "team_abbr" in df.columns and "team_name" in df.columns:
        return dict(zip(df["team_abbr"], df["team_name"]))
    return {}


# ---------------------------------------------------------------------------
# Run visibility: 60-day chunk plan + a progress bar.
#
# BOTH are display only. Nothing here reads or writes the frames being
# reported on, and no loader's call signature changes, so a run with the
# reporting removed produces byte-identical artifacts.
# ---------------------------------------------------------------------------
def chunk_date_range(start, end, chunk_days: int = POPULATE_CHUNK_DAYS):
    """Yield inclusive ``(chunk_start, chunk_end)`` windows of ``chunk_days``.

    Mirrors MLB's chunk walk: each window is ``chunk_days`` long except the
    last, which is truncated at ``end`` (the same truncation MLB does with
    ``min(cursor + timedelta(days=chunk_days - 1), end)``). Yields nothing
    when the window is empty or reversed, so a caller never has to special
    case a run whose date window collapsed.

    The 60-day plan is REPORTING granularity. The nflverse loaders fetch per
    season and that is deliberately unchanged: chunking a per-season pull
    would change which rows arrive, which is a functional change this
    reporting layer must not make.
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if pd.isna(start) or pd.isna(end) or end < start:
        return
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + pd.Timedelta(days=1)


class StageProgress:
    """A logger-rendered progress bar with no third-party dependency.

    Deliberately not tqdm: NFL's runtime dependency set is what the Kaggle
    bootstrap installs, and adding a package to observe a run is a change to
    the production environment, which this work is not allowed to make. MLB
    installs tqdm but never imports it, so there is no bar to be structurally
    identical to — this renders through the same ``logger`` the rest of the
    run uses, so it appears in the same captured log a reader already reads.

    Display only. The bar holds counters; it never touches the data.
    """

    WIDTH = 24
    # A 105-fold stage must not write 101 log lines. Emit at most every 5%
    # (~21 lines) and on every step of a short stage, so a 5-source bar still
    # moves once per source.
    LONG_STAGE = 25
    PCT_STEP = 5

    def __init__(self, total: int, label: str, width: int = WIDTH):
        self.total = max(0, int(total))
        self.label = label
        self.width = max(1, int(width))
        # A zero-length stage has no progress to show; stay silent rather than
        # printing a bar that jumps 0 -> 100% the instant it is created.
        self.enabled = self.total > 0
        self.n = 0
        self._step = 1 if self.total <= self.LONG_STAGE else self.PCT_STEP
        self._last_pct = -1

    def advance(self, n: int = 1) -> "StageProgress":
        """Move the bar forward. Callable, so it can be a bare callback."""
        self.n += n
        self.render()
        return self

    def render(self, force: bool = False) -> None:
        if not self.enabled:
            return
        frac = min(1.0, self.n / self.total)
        pct = int(frac * 100)
        # 100% always reports, so a completed stage is never left looking
        # unfinished; otherwise only redraw once the bar has moved a step.
        if not force and pct < 100 and pct - self._last_pct < self._step:
            return
        self._last_pct = pct
        filled = int(round(frac * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        logger.info("  [%s] %3d%%  %d/%d  %s", bar, pct,
                    min(self.n, self.total), self.total, self.label)

    def close(self) -> None:
        """Finish the bar, and say so loudly if the stage stopped short."""
        if not self.enabled:
            return
        if self.n < self.total:
            # A short stage is usually an exception escaping a loader, and a
            # full-width bar here would read as success.
            logger.warning("  [%s] %3d%%  stopped short: %d/%d  %s",
                           "#" * self.width,
                           int(100 * self.n / self.total), self.n, self.total,
                           self.label)
        elif self._last_pct < 100:
            # advance() already draws 100% on the final step; re-drawing here
            # would print the same line twice.
            self.render(force=True)


def population_unit_counts(core_seasons: list[int]) -> dict[str, int]:
    """Fetch units Phase 2 will advance, per source, for a progress denominator.

    Lives beside the loader loops it describes so the two cannot drift: each
    loader calls its ``progress`` hook exactly once per unit, and NGS calls it
    once per (season, group). Getting this wrong does not corrupt data -- the
    bar just stops short of 100% and warns, which is the safe direction to
    fail, but the count should still be right.
    """
    extended = len(core_seasons) + 1          # +1 warmup season
    return {
        "pbp": len(core_seasons),
        "player_stats": extended,
        "nextgen": len(NGS_GROUPS) * extended,
        "snap_counts": extended,
        "ftn_charting": len(core_seasons),
    }
