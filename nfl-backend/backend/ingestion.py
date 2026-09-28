"""NFL data ingestion — nflverse pull with local parquet caching.

Pulls the schedules and play-by-play the pipeline needs through
``nflreadpy`` and caches each season's frame beside the repo (never inside
the git tree). Deterministic column narrowing;regular-season and postseason filtering happens here so every downstream module sees the eligible population only.

All frames carry the nflverse column names; the feature engine is the only
place that interprets them.
"""
from __future__ import annotations

import logging
import os
import re
import time
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
    # 2026-09-26 per-position player-quality family (features.epa_opportunity_
    # table): the three role-attribution ids and the three opportunity flags.
    # The flags are the denominator — they handle what play_type cannot, a
    # scramble is a qb_dropback on a play_type of "run" — and they exist in
    # every season 2016-2025, so the family is full history. pass_attempt is
    # already in PBP_NEEDS above.
    "passer_player_id", "receiver_player_id", "rusher_player_id",
    "qb_dropback", "rush_attempt",
]

# Cache schema version for the pbp parquets. Bump whenever PBP_NEEDS widens:
# the per-season caches store the NARROWED frame, so a previously cached
# season would otherwise keep serving the old column set (features built from
# the missing columns would degrade to all-NaN and read like evidence).
# "v1" = the original 21-column set; "v2" adds the candidate-pool columns;
# "v3" swaps the never-available raw "yac" for "yac_epa" (the YAC-as-EPA
# decomposition nflreadpy actually publishes); "v4" adds the situational
# columns (qtr/down/ydstogo/goal_to_go/score_differential/half_seconds_
# remaining) behind the platoon rollups; "v5" adds player-role IDs and
# opportunity flags for per-position projected-lineup EPA features.
PBP_CACHE_VERSION = "v5"


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
# The former inj_*_out_diff feature family was removed because the source
# stopped publishing per-report timestamps after 2024. The PIT injury loader
# below is separate: it is used only for the projected-player EPA candidates,
# with strict publication-time filtering and the explicitly requested status
# classifier; seasons without timestamps produce no eligible status rows.
#
# load_injuries_pit is the PIT-only injury source for the EPA lineup features.
# Only an Out, IR (including the spelled-out Injured Reserve), or Doubtful
# designation excludes a player from THAT target game's projected lineup.
# Other statuses and the absence of an admissible report do not erase the
# player's lagged EPA or remove him from the candidate pool.
#
# nflverse carries per-report date_modified through 2024, but not in the
# 2025/2026 pulls currently available. Missing timestamps fail closed: no
# injury exclusion is inferred without a strictly pre-kickoff publication.
# v2 added team/game_type keys so a week's report can only apply to that
# team's target game; v3 canonicalizes the source's game_type/season_type
# spelling before caching so source-schema variants remain joinable.
INJ_PIT_NEEDS = ("gsis_id", "season", "game_type", "team", "week",
                 "report_status", "date_modified")
INJ_PIT_CACHE_VERSION = "v3"
INJ_PIT_INJURED_TOKENS = frozenset({"out", "ir", "doubtful"})


def injury_availability_weight(status: object) -> float:
    """1=not designated injured, 0=Out/IR/Doubtful.

    Match the named designation tokens only (plus the equivalent phrase
    ``Injured Reserve``); statuses such as ``Injury`` or ``Reserve`` by
    themselves are not an exclusion. The weight is lineup membership only,
    never a multiplier on the player's historical EPA.
    """
    if status is None or pd.isna(status):
        return 1.0
    tokens = re.findall(r"[a-z]+", str(status).strip().lower())
    injured_reserve = any(tokens[i:i + 2] == ["injured", "reserve"]
                          for i in range(len(tokens) - 1))
    return 0.0 if (set(tokens) & INJ_PIT_INJURED_TOKENS) or injured_reserve else 1.0


def _strict_utc_timestamp(value):
    """Return UTC only when a report timestamp carries an explicit timezone.

    Naive timestamps have no trustworthy timezone provenance and cannot be
    compared strictly with the scheduled kickoff; reject rather than guessing.
    """
    if value is None or pd.isna(value):
        return pd.NaT
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return pd.NaT
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        return pd.NaT
    return stamp.tz_convert("UTC")


def _normalize_id(value) -> str | None:
    """Canonical gsis player id, or None for the several null spellings."""
    if value is None:
        return None
    try:
        if value is not None and value != value:  # NaN
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "null"}:
        return None
    return text


def load_injuries_pit(games: pd.DataFrame,
                      seasons: list[int] | None = None,
                      use_cache: bool = True, refresh_upcoming: bool = True,
                      progress=None) -> pd.DataFrame:
    """Latest per-player injury availability published strictly before kickoff.

    ``gametime`` is ET and is converted to UTC. For each game/player, the
    latest report row with ``date_modified < kickoff`` wins. Equal-to-kickoff
    and later reports are excluded. Rows lacking their own publication
    timestamp are never assigned a guessed time.

    Any season with a future scheduled kickoff in ``games`` is refreshed on
    every run because injury reports change during the week; historical
    seasons may use their stable per-season cache. Reports are joined to the
    exact target team and game type, then the latest matching publication
    strictly before kickoff determines only lineup membership—not the
    player's historical EPA rate.
    """
    cols = ["game_id", "team", "player_id", "status",
            "availability_weight", "published"]
    seasons = seasons or config.ALL_SEASONS
    need_games = {"game_id", "gameday", "gametime", "season", "week",
                  "game_type", "home_team", "away_team"}
    if games is None or not need_games.issubset(set(games.columns)):
        return pd.DataFrame(columns=cols)

    g = games[["game_id", "gameday", "gametime", "season", "week",
               "game_type", "home_team", "away_team"]].copy()
    g["game_id"] = g["game_id"].astype(str)
    g["season"] = pd.to_numeric(g["season"], errors="coerce")
    g["week"] = pd.to_numeric(g["week"], errors="coerce")
    for c in ("game_type", "home_team", "away_team"):
        g[c] = g[c].astype("string").str.strip().str.upper()
    g = g.drop_duplicates("game_id")
    for c in ("game_type", "home_team", "away_team"):
        g = g[g[c].notna() & g[c].ne("")]
    gd = pd.to_datetime(g["gameday"], errors="coerce")
    gt = g["gametime"].astype("string").str.strip()
    local = pd.to_datetime(
        gd.dt.strftime("%Y-%m-%d") + " " + gt.fillna(""), errors="coerce")
    try:
        g["kickoff"] = local.dt.tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert(
                "UTC")
    except (TypeError, ValueError):
        # Bad or missing local kickoff is not replaced by a guessed midnight.
        g["kickoff"] = pd.Series(pd.NaT, index=g.index, dtype="datetime64[ns, UTC]")
    g = g.dropna(subset=["kickoff", "season", "week"])
    if g.empty:
        return pd.DataFrame(columns=cols)

    now = pd.Timestamp.now(tz="UTC")
    refresh_seasons = (set(g.loc[g["kickoff"] > now, "season"].astype(int))
                       if refresh_upcoming else set())
    frames = []
    for s in seasons:
        path = _cache_path(f"inj_pit_{INJ_PIT_CACHE_VERSION}_{s}.parquet")
        cached = None
        try:
            if path.exists():
                try:
                    cached = pd.read_parquet(path)
                except Exception as exc:  # corrupt cache -> re-pull
                    logger.warning("injury PIT cache %s unreadable (%s)",
                                   path.name, exc)
            if use_cache and cached is not None and s not in refresh_seasons:
                frames.append(cached)
                continue
            try:
                from nflreadpy import load_injuries
                df = _polars_to_pandas(load_injuries(s))
            except Exception as exc:  # noqa: BLE001
                logger.warning("injury PIT source unavailable for %s: %s", s, exc)
                # A stale current-week cache can miss a new pre-kickoff update.
                # Never reuse it for a season with a future target game.
                if cached is not None and s not in refresh_seasons:
                    logger.warning("using historical cached PIT report rows for %s", s)
                    frames.append(cached)
                continue
            df = df.loc[:, ~df.columns.duplicated()].copy()
            # nflreadr has published both game_type and season_type spellings;
            # canonicalize before narrowing so the matchup join stays exact.
            if "game_type" not in df.columns and "season_type" in df.columns:
                df["game_type"] = df["season_type"]
            # Keep a stable cache schema even when the source omits a field.
            for c in INJ_PIT_NEEDS:
                if c not in df.columns:
                    df[c] = pd.NA
            df = df[list(INJ_PIT_NEEDS)]
            df.to_parquet(path, index=False)
            frames.append(df)
        finally:
            if progress is not None:
                progress()
    if not frames:
        return pd.DataFrame(columns=cols)

    # A season whose source omits a field contributes an all-NA COLUMN, and
    # pandas now warns that concat will stop ignoring all-NA entries when
    # inferring result dtypes. Filtering empty frames is not enough -- the
    # deprecation is about columns, and 2025/2026 injury sources carry no
    # ``date_modified`` at all, so those frames are all-NA in that column.
    # Drop each frame's all-NA columns before concatenating, exactly as the
    # notice prescribes, then restore the narrow schema below so a field that
    # is all-NA in EVERY season still exists for the fail-closed checks.
    _nonempty = [f for f in frames if not f.empty]
    if _nonempty:
        inj = pd.concat([f.dropna(axis=1, how="all") for f in _nonempty],
                        ignore_index=True)
        for _c in INJ_PIT_NEEDS:
            if _c not in inj.columns:
                inj[_c] = pd.NA
        inj = inj[list(INJ_PIT_NEEDS)]
    else:
        inj = pd.DataFrame(columns=list(INJ_PIT_NEEDS))
    # Also tolerate an older v2 cache written by a source exposing only
    # season_type (the v3 cache path ensures normal pulls are rebuilt).
    if "game_type" not in inj.columns and "season_type" in inj.columns:
        inj["game_type"] = inj["season_type"]
    if not {"gsis_id", "date_modified", "team", "game_type"} <= set(inj.columns):
        return pd.DataFrame(columns=cols)
    inj["player_id"] = inj["gsis_id"].map(_normalize_id)
    inj["team"] = inj["team"].astype("string").str.strip().str.upper()
    inj["game_type"] = inj["game_type"].astype("string").str.strip().str.upper()
    inj = inj.dropna(subset=["player_id", "team", "game_type"])
    inj = inj[inj["team"].ne("") & inj["game_type"].ne("")]
    # FAIL CLOSED: no explicitly zoned publication timestamp, no designation.
    inj["published"] = pd.to_datetime(
        inj["date_modified"].map(_strict_utc_timestamp), errors="coerce", utc=True)
    inj = inj[inj["published"].notna()]
    if inj.empty:
        return pd.DataFrame(columns=cols)
    inj["status"] = (inj["report_status"].astype("string").str.strip()
                     .str.lower().fillna(""))
    inj["availability_weight"] = inj["status"].map(
        injury_availability_weight).fillna(1.0).astype(float)
    inj["season"] = pd.to_numeric(inj["season"], errors="coerce")
    inj["week"] = pd.to_numeric(inj["week"], errors="coerce")
    inj = inj.dropna(subset=["season", "week"])

    # Expand each scheduled game into its two participating teams before
    # joining the week-level feed. Joining only on (season, week) would leak
    # another team's designation onto this matchup's player pool.
    team_games = pd.concat([
        g[["game_id", "season", "week", "game_type", "kickoff", "home_team"]]
        .rename(columns={"home_team": "team"}),
        g[["game_id", "season", "week", "game_type", "kickoff", "away_team"]]
        .rename(columns={"away_team": "team"}),
    ], ignore_index=True).drop_duplicates(["game_id", "team"])
    pairs = team_games.merge(
        inj[["player_id", "team", "season", "game_type", "week", "published",
             "status", "availability_weight"]],
        on=["team", "season", "game_type", "week"], how="inner")
    pairs = pairs[pairs["published"] < pairs["kickoff"]]
    if pairs.empty:
        return pd.DataFrame(columns=cols)
    # Latest admissible publication wins. If conflicting rows tie exactly,
    # an explicit Out/IR/Doubtful (weight 0) wins that tie.
    out = (pairs.sort_values(["game_id", "team", "player_id", "published",
                              "availability_weight"],
                             ascending=[True, True, True, True, False],
                             kind="mergesort")
           .drop_duplicates(["game_id", "team", "player_id"], keep="last")
           [["game_id", "team", "player_id", "status", "availability_weight",
             "published"]]
           .reset_index(drop=True))
    out["game_id"] = out["game_id"].astype(str)
    return out[cols]


PS_NEEDS = [
    "game_id", "team", "position", "carries", "rushing_yards", "targets",
    "receptions", "receiving_yards", "receiving_tds", "season_type",
    # player_id joins the weekly row to the per-player EPA/opportunity table
    # built from pbp.  Position labels ride along with it, which is why the
    # per-position quality family does not need a separate roster source.
    "player_id",
]

PS_CACHE_VERSION = "v2"  # v2 includes player_id so EPA histories join by player

# Weekly per-player tracking efficiency (week-0 rows are SEASON aggregates —
# they mix future games into a week-1 value, so they are dropped at load).
NGS_GROUPS = ("passing", "rushing", "receiving")
# Published windows (nflreadpy rejects out-of-window seasons outright —
# "Season must be between ..." — so requesting them is a guaranteed skip,
# not a degradation to warn about at runtime). Declared once here so the
# loaders, the progress-bar denominator, and the tests share one truth.
NGS_FIRST_SEASON = 2016
FTN_FIRST_SEASON = 2022
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

    Per-season parquet caches (PS cache v2 includes player_id); a failed
    season is warned and skipped, never fatal. Returns None only when NO
    season could be loaded."""
    seasons = seasons or config.ALL_SEASONS
    frames: list[pd.DataFrame] = []
    for season in seasons:
        try:
            path = _cache_path(f"ps_{PS_CACHE_VERSION}_{season}.parquet")
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


def ngs_seasons(seasons: list[int]) -> list[int]:
    """Seasons the NGS source can actually serve (published window)."""
    return [s for s in seasons if s >= NGS_FIRST_SEASON]


def ftn_charting_seasons(seasons: list[int]) -> list[int]:
    """Seasons the FTN charting source can actually serve (2022+)."""
    return [s for s in seasons if s >= FTN_FIRST_SEASON]


def load_nextgen(seasons: list[int] | None = None,
                 use_cache: bool = True, progress=None) -> pd.DataFrame | None:
    """nflverse Next-Gen Stats weekly tracking efficiency, all three groups.

    Week-0 rows (season aggregates that would leak future performance into
    early-game features) are dropped at load. Per-(season, group) caches
    (NGS cache v2 = rush_attempts weight fix); a failed season/group is
    warned and skipped. Seasons below the source's published window are
    pre-filtered OUT (one INFO line, not a per-season WARNING — the skip
    is the documented policy, not a surprise). Returns None only when
    NOTHING could be loaded."""
    seasons = seasons or config.ALL_SEASONS
    _skipped = [s for s in seasons if s < NGS_FIRST_SEASON]
    if _skipped:
        logger.info("ngs published from %d — skipping %s (documented window)",
                    NGS_FIRST_SEASON, _skipped)
    seasons = ngs_seasons(seasons)
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
# a season without the endpoint is warned and skipped). Cache v2 adds the
# per-row ``pfr_player_id``: the weekly injury-share family prices a report
# row with the player's own snap history, and the report feed keys players by
# gsis_id while the snap feed carries pfr_player_id — so the id must survive
# the cache narrow. (v1 stored no id column and its frames cannot feed the
# family; the bump forces the one-time re-pull.)
SNAPS_CACHE_VERSION = "v2"
SNAPS_NEEDS = ["game_id", "season", "week", "team", "opponent", "position",
               "pfr_player_id",
               "offense_snaps", "offense_pct", "defense_snaps", "defense_pct"]

# The injury-share family prices a flag with the player's last-8 snap history
# across BOTH sides of the ball, so the pull must reach well before the 2016
# feature window: a week-1 2016 report is priced from 2013-2015 snaps.
# A constant (not a derived offset) so the pull, the feature builder's
# history expectation and the progress-bar denominator cannot drift apart.
SNAPS_HISTORY_FIRST_SEASON = 2013


def snap_count_seasons(seasons: list[int]) -> list[int]:
    """Season window for the snap-count pull.

    The pipeline window extended back to SNAPS_HISTORY_FIRST_SEASON (and at
    least one warmup season before the window, like every trailing source)
    so the earliest report week still has multi-season prior snap history."""
    first = min(SNAPS_HISTORY_FIRST_SEASON, min(seasons) - 1)
    return list(range(first, max(seasons) + 1))


def load_snap_counts(seasons: list[int] | None = None,
                     use_cache: bool = True, progress=None) -> pd.DataFrame | None:
    """nflverse snap counts narrowed to the participation rollup needs.

    Per-season parquet caches (SNAPS cache v2, includes pfr_player_id); a
    failed season is warned and skipped, never fatal. Returns None only when
    NO season loaded."""
    seasons = seasons or config.ALL_SEASONS
    frames: list[pd.DataFrame] = []
    for season in seasons:
        try:
            path = _cache_path(f"snaps_{SNAPS_CACHE_VERSION}_{season}.parquet")
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
    _skipped = [s for s in seasons if s < FTN_FIRST_SEASON]
    if _skipped:
        logger.info("ftn charting published from %d — skipping %s "
                    "(documented window)", FTN_FIRST_SEASON, _skipped)
    seasons = ftn_charting_seasons(seasons)
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


# GSIS <-> PFR player-id crosswalk (nflreadpy.load_players, one league-wide
# request): the weekly injury report keys players by gsis_id while snap
# counts carry pfr_player_id. Cached once (v1). A report row whose player is
# absent from the crosswalk cannot be priced and is dropped by the family
# builder — measured against 2016-2025 reports, >= 99.87% of Out/Doubtful
# OL/DEF report rows resolve in every season.
PLAYERS_CACHE_VERSION = "v1"


def load_player_id_crosswalk(use_cache: bool = True) -> pd.DataFrame:
    """GSIS id -> PFR id for every published player (one league-wide pull)."""
    path = _cache_path(f"players_crosswalk_{PLAYERS_CACHE_VERSION}.parquet")
    if use_cache and path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # corrupt cache -> re-pull
            logger.warning("player crosswalk cache unreadable (%s): %s",
                           path.name, exc)
    try:
        from nflreadpy import load_players
    except Exception as exc:  # noqa: BLE001
        logger.error("nflreadpy unavailable: %s", exc)
        return pd.DataFrame(columns=["gsis_id", "pfr_id"])
    df = _polars_to_pandas(load_players())
    keep = [c for c in ("gsis_id", "pfr_id") if c in df.columns]
    if len(keep) < 2:
        logger.warning("player crosswalk source missing id columns; empty")
        return pd.DataFrame(columns=["gsis_id", "pfr_id"])
    df = df[keep].dropna(how="any").drop_duplicates("gsis_id", keep="first")
    df.to_parquet(path, index=False)
    return df


# Weekly injury reports (nflreadpy.load_injuries), REPORT-CYCLE semantics.
# The strict-PIT loader above (load_injuries_pit) fails closed on a missing
# date_modified and feeds the projected-lineup EPA family; nflverse stopped
# publishing per-report timestamps after 2024, so it yields nothing for
# 2025/2026. THIS loader needs no timestamp at all: a (season, week, team)
# report row is, by league rule, published during that week's report cycle
# and is therefore pre-kickoff information for that team-week's game. Each
# row is joined to its OWN team-week game only — never a date guess — which
# is exactly the join the strict-PIT loader performs when timestamps exist
# (gate: report-cycle vs strict-PIT agrees on >= 99% of 2016-2024 rows).
INJ_WEEKLY_NEEDS = ("gsis_id", "season", "game_type", "team", "week",
                    "report_status")
INJ_WEEKLY_CACHE_VERSION = "v1"


def load_injuries_weekly(seasons: list[int] | None = None,
                         use_cache: bool = True, progress=None) -> pd.DataFrame:
    """Raw weekly report rows for the injury-share family.

    Narrow (gsis_id, season, game_type, team, week, report_status) frame,
    per-season parquet caches (INJ_WEEKLY cache v1). No timestamp gate: the
    report-cycle rule substitutes for the missing publication timestamps of
    the 2025/2026 sources. Status interpretation (which spellings exclude a
    player) lives in features.injury_share_table, so this stays a faithful
    cache of the source rows."""
    seasons = seasons or config.ALL_SEASONS
    frames: list[pd.DataFrame] = []
    for season in seasons:
        try:
            path = _cache_path(
                f"inj_weekly_{INJ_WEEKLY_CACHE_VERSION}_{season}.parquet")
            if use_cache and path.exists():
                try:
                    frames.append(pd.read_parquet(path))
                    continue
                except Exception as exc:  # corrupt cache -> re-pull
                    logger.warning("weekly injury cache %s unreadable (%s)",
                                   path.name, exc)
        finally:
            if progress is not None:
                progress()
        try:
            from nflreadpy import load_injuries
            logger.info("loading weekly injury report season %s", season)
            df = _polars_to_pandas(load_injuries(season))
        except Exception as exc:  # noqa: BLE001
            logger.warning("weekly injury report unavailable for %s: %s",
                           season, exc)
            continue
        df = df.loc[:, ~df.columns.duplicated()].copy()
        # Canonicalize the season_type spelling before narrowing (same
        # source-schema tolerance as load_injuries_pit).
        if "game_type" not in df.columns and "season_type" in df.columns:
            df["game_type"] = df["season_type"]
        for c in INJ_WEEKLY_NEEDS:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[list(INJ_WEEKLY_NEEDS)]
        df.to_parquet(path, index=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(INJ_WEEKLY_NEEDS))
    out = pd.concat(frames, ignore_index=True)
    for c in INJ_WEEKLY_NEEDS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[list(INJ_WEEKLY_NEEDS)]


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


# ---------------------------------------------------------------------------
# Progress bar — MLB statcast's idiom, verbatim.
# ---------------------------------------------------------------------------
# MLB's statcast pull draws its bar with ``tqdm``:
# ``pybaseball.statcast`` wraps its per-day sub-requests in
# ``tqdm(total=len(date_range))`` and constructs it with STOCK DEFAULTS. That
# is the whole reference. The captured Kaggle log shows exactly what those
# defaults produce:
#
#   0%|          | 0/36 [00:00<?, ?it/s]
#   100%|########| 36/36 [00:40<00:00, 1.14s/it]
#        -> 136267 pitches
#
# Two properties of that output are load-bearing and are the reason this class
# exists in this shape.
#
# 1. STOCK DEFAULTS. No ``bar_format``, no ``dynamic_ncols``, ``leave=True``,
#    ``disable`` left at its own default. A hand-rolled format is the one thing
#    that would make this bar look nothing like the one it imitates. The
#    previous hand-rolled bar did exactly that: ``[####----------------]  50%
#    42/93  nflverse population`` shares no glyph, no punctuation, no timing
#    and no rate with MLB's, so an operator reading both logs had to learn two
#    vocabularies for the same fact.
#
# 2. IT DRAWS ON A CAPTURED STREAM. ``tqdm`` CAN suppress itself on a
#    non-terminal -- ``std.py`` says ``if disable is None and not
#    file.isatty(): disable = True`` -- but the parameter's default is
#    ``False``, not ``None``, so that branch is unreachable unless a caller
#    opts in. A stock bar therefore writes into a pipe, a file and a captured
#    notebook cell exactly as it writes to a terminal. MLB's bars survive into
#    its Kaggle log for that reason and no other. Any tty gate added here would
#    be inventing a stricter rule than the one MLB runs under, and would be the
#    single reason these bars vanished where MLB's do not.
#
# ``position=0`` is deliberate too. Left alone tqdm stacks a new bar ABOVE one
# already open and rewinds the cursor a line per redraw; on a terminal that is
# invisible, in a capture it is an ``ESC[A`` and a blank line after every
# refresh. NFL's bars never overlap (the population bar closes before the OOF
# bars open), so pinning position states that instead of leaving it to a
# collision that does not happen.
#
# ``tqdm`` stays an OPTIONAL import (NBA parity). When it is absent the bar
# degrades to the heartbeat counter below, which keeps the run's output a log
# line rather than silence, and which is the only thing drawn when
# ``NFL_PROGRESS=0``.
#
# Display only: the bar holds a counter, never touches the data, and cannot
# change which requests a loader makes.
def _hms(seconds: float) -> str:
    """Seconds as tqdm renders a duration: ``00:00``, ``01:14``, ``1:02:03``.

    Same units and same shape as the ``[00:40<00:00, 1.14s/it]`` MLB's
    statcast bar prints, so the no-tqdm fallback is not a different dialect
    of the same information. Over an hour gains the hour field, as tqdm does.
    """
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class StageProgress:
    """MLB-statcast-style progress bar: tqdm when importable, else a counter.

    Same surface as before (``advance`` / ``close`` / ``enabled``) so the
    call sites in this module, ``weather.py`` and ``master_pipeline`` are
    unchanged, but the rendering is now MLB's rather than bespoke.
    """

    #: Bar geometry, kept only for the no-tqdm fallback. tqdm's own width
    #: applies whenever the library is importable.
    WIDTH = 24
    # A 105-fold stage must not write 105 log lines in the fallback path.
    LONG_STAGE = 25
    PCT_STEP = 5
    #: Longest silence between fallback progress lines. Long enough to stay
    #: legible in a captured log, short enough that a stalled run is obviously
    #: stalled rather than merely quiet.
    HEARTBEAT_SEC = 30.0

    ENV = "NFL_PROGRESS"
    OFF_WORDS = {"0", "false", "no", "off"}

    @classmethod
    def enabled_by_env(cls) -> bool:
        """Default ON; an explicit ``0``/``off``/``no``/``false`` always wins.

        Same rule as the NBA backend's ``NBA_PROGRESS``: the bar is the
        feature, and silence should have to be asked for.
        """
        raw = os.environ.get(cls.ENV)
        if raw is None or not raw.strip():
            return True
        return raw.strip().lower() not in cls.OFF_WORDS

    def __init__(self, total: int, label: str, width: int = WIDTH,
                 show_rate: bool = True):
        self.total = max(0, int(total))
        self.label = label
        self.width = max(1, int(width))
        self.show_rate = show_rate
        # A zero-length stage has no progress to show; stay silent rather than
        # printing a bar that jumps 0 -> 100% the instant it is created.
        self.enabled = self.total > 0
        self.n = 0
        self._step = 1 if self.total <= self.LONG_STAGE else self.PCT_STEP
        self._last_pct = -1
        self._inner = None
        self._started = time.monotonic()
        self._last_beat = self._started
        # A total of 0 disables tqdm too: a bar with no denominator renders
        # nothing useful and would print a bare 0it.
        if self.enabled and self.enabled_by_env():
            self._inner = self._make_bar()

    def _make_bar(self):
        """A stock tqdm bar, or ``None`` when the library is unavailable.

        Optional import, imported lazily so a missing dependency is a ``None``
        return and never an ImportError at pipeline start.
        """
        try:
            from tqdm import tqdm  # type: ignore[import-not-found]
        except Exception:  # noqa: BLE001 - absence is a supported state
            return None
        kwargs: dict = {"position": 0}
        if not self.show_rate:
            # Phases and pulls are wildly uneven in cost, so a rate and an ETA
            # computed off one or two samples read as a measurement and are
            # nothing of the kind. Default tqdm format minus the rate.
            kwargs["bar_format"] = "{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}"
        return tqdm(total=self.total, desc=self.label, leave=True, **kwargs)

    def advance(self, n: int = 1) -> "StageProgress":
        """Move the bar forward. Callable, so it can be a bare callback."""
        self.n += n
        if not self.enabled:
            return self
        if self._inner is not None:
            self._inner.update(n)
            return self
        now = time.monotonic()
        if now - self._last_beat >= self.HEARTBEAT_SEC or self.n >= self.total:
            self._last_beat = now
            logger.info("%s", self._fallback_line(now))
        return self

    def _fallback_line(self, now: float | None = None) -> str:
        """One line in MLB's shape, for when tqdm is not importable.

        Deliberately formatted like the line it stands in for --
        ``  50.0%|#####     | 42/93 [00:12<00:12, 3.5 batch/s]`` -- so a run
        without tqdm still reads as the same vocabulary as a run with it.
        """
        now = time.monotonic() if now is None else now
        elapsed = max(0.0, now - self._started)
        frac = min(1.0, self.n / self.total)
        # tqdm reports a rate as elapsed/unit, and treats a rate too fast to
        # measure as "?it/s". Dividing count by elapsed instead reported
        # "1343285.02 unit/s" for a loop that finished in under a millisecond,
        # which is a number no operator can read as a rate at all.
        rate = 0.0
        if elapsed > 0 and self.n > 0:
            per_unit = elapsed / self.n
            rate = 1.0 / per_unit if per_unit > 0 else 0.0
        filled = int(round(frac * self.width))
        bar = "#" * filled + " " * (self.width - filled)
        head = f"  {100.0 * frac:5.1f}%|{bar}| {self.n}/{self.total}"
        if self.show_rate:
            # Anything faster than a millisecond per unit is a counter, not
            # work, and reads as "?" exactly as tqdm renders it.
            rate_s = (f"{elapsed / self.n:.2f}s/unit" if rate >= 1000.0
                      else (f"{rate:.2f} unit/s" if rate > 0 else "? unit/s"))
            if self.n < self.total and rate > 0:
                head += (f" [{_hms(elapsed)}<"
                         f"{_hms((self.total - self.n) / rate)}, {rate_s}]")
            else:
                head += f" [{_hms(elapsed)}<00:00, {rate_s}]"
        return f"{head}: {self.label}"

    def close(self) -> None:
        """Finish the bar, and say so loudly if the stage stopped short.

        The short-stage warning is the one behaviour of the old bar worth
        keeping verbatim: a stage that ended early usually means an exception
        escaped a loader, and a completed-looking bar would read as success.
        """
        if not self.enabled:
            return
        if self._inner is not None:
            self._inner.close()
        if self.n < self.total:
            logger.warning(
                "  stopped short: %d/%d  %s", self.n, self.total, self.label)
        elif self._inner is None and self._last_pct < 100:
            logger.info("%s", self._fallback_line())


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
        # NGS publishes 2016+ and FTN 2022+: the requested window is
        # pre-filtered by the same helpers the loaders use, so the bar never
        # counts units that can only fail (the old count included the
        # doomed 2015 NGS pulls and six 2016-2021 FTN units). The pipeline
        # pulls NGS with one warmup season in front — mirror that here.
        "nextgen": len(NGS_GROUPS) * len(
            ngs_seasons(([core_seasons[0] - 1] + core_seasons)
                        if core_seasons else [])),
        # The snap pull reaches back to SNAPS_HISTORY_FIRST_SEASON so the
        # injury-share family can price the earliest report weeks.
        "snap_counts": len(snap_count_seasons(core_seasons)),
        "ftn_charting": len(ftn_charting_seasons(core_seasons)),
        "injuries": len(core_seasons),
        "injuries_weekly": len(core_seasons),
    }
