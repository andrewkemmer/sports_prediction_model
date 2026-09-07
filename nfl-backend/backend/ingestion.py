"""NFL data ingestion — nflverse pull with local parquet caching.

Pulls the schedules and play-by-play the pipeline needs through
``nflreadpy`` and caches each season's frame beside the repo (never inside
the git tree). Deterministic column narrowing; regular-season filtering
happens here so every downstream module sees the eligible population only.

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
    regular-season games within the configured season window (2018+)."""
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
    path = _cache_path(f"schedules_{tag}.parquet")
    if use_cache and path.exists():
        return pd.read_parquet(path)
    from nflreadpy import load_schedules
    df = _polars_to_pandas(load_schedules(seasons))
    keep = [c for c in (
        "game_id", "season", "week", "game_type", "gameday", "gametime",
        "home_team", "away_team", "home_score", "away_score", "roof",
        "div_game", "stadium", "surface", "location", "referee",
        "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name",
    ) if c in df.columns]
    df = df[keep]
    df.to_parquet(path, index=False)
    return df


PBP_NEEDS = [
    "game_id", "posteam", "defteam", "yards_gained", "epa", "qb_epa",
    "game_seconds_remaining", "interception", "fumble_lost",
    "passing_yards", "pass_attempt", "sack", "penalty", "penalty_yards",
    "penalty_team", "third_down_converted", "third_down_failed",
    "yardline_100", "touchdown", "field_goal_result", "drive",
]


def load_pbp(seasons: list[int] | None = None,
             use_cache: bool = True) -> pd.DataFrame | None:
    """ nflverse play-by-play narrowed to the columns the rollup needs.

    Per-season parquet caches (a season without published PBP — e.g. the
    in-progress season — is warned and skipped, never fatal; downstream
    features degrade to NaN per the documented missing-value policy).
    Returns None only when NO season could be loaded."""
    seasons = seasons or config.ALL_SEASONS
    frames: list[pd.DataFrame] = []
    missing: list[int] = []
    for season in seasons:
        path = _cache_path(f"pbp_{season}.parquet")
        if use_cache and path.exists():
            try:
                frames.append(pd.read_parquet(path))
                continue
            except Exception as exc:  # corrupt cache → re-pull
                logger.warning("pbp cache %s unreadable (%s)", path.name, exc)
        missing.append(season)
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
            df.to_parquet(_cache_path(f"pbp_{season}.parquet"), index=False)
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
