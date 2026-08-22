"""
Statcast data ingestion.

Pulls raw pitch-by-pitch data via pybaseball, normalizes columns for schema
drift, downcasts for memory, and saves to Parquet on disk.

No feature engineering — this module's only job is to produce a clean
``pitches.parquet`` file.

Usage:
    from ingestion import pull_statcast
    pull_statcast("2026-08-01", "2026-08-20", out_path="pitches.parquet")
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Statcast column schema ──────────────────────────────────────────────────

STATCAST_COLS = [
    "game_date", "game_pk", "game_type", "home_team", "away_team",
    "inning", "inning_topbot", "outs_when_up", "balls", "strikes",
    "on_1b", "on_2b", "on_3b",
    "at_bat_number", "pitch_number", "pitcher", "batter",
    "p_throws", "stand",
    "pitch_type", "release_speed", "release_pos_x", "release_pos_z",
    "player_name",
    "description", "events",
    "spin_rate", "spin_axis",
    "release_spin_rate", "release_extension",
    "plate_x", "plate_z",
    "zone", "pfx_x", "pfx_z",
    "hit_distance_sc", "launch_speed", "launch_angle",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
    "woba_value", "babip_value", "iso_value",
    "barrel", "hard_contact",
    "home_score", "away_score",
    "delta_home_win_exp", "delta_run_exp",
]

COLUMN_ALIASES = {
    "barrel_pct": "barrel", "is_barrel": "barrel",
    "hardhit": "hard_contact", "hard_hit": "hard_contact",
    "exit_velocity": "launch_speed", "exit_velo": "launch_speed",
    "la": "launch_angle",
    "xwoba": "estimated_woba_using_speedangle",
    "xwOBA": "estimated_woba_using_speedangle",
    "xba": "estimated_ba_using_speedangle",
    "pitcher_name": "player_name",
    "event": "events",
}

UNUSED_COLS = [
    "fielder_2", "fielder_3", "fielder_4", "fielder_5",
    "fielder_6", "fielder_7", "fielder_8", "fielder_9",
    "if_fielding_alignment", "of_fielding_alignment",
    "post_home_score", "post_away_score",
    "event", "type", "launch_speed_angle",
]


# ── Public API ──────────────────────────────────────────────────────────────

def _chunked_statcast(
    start: date,
    end: date,
    chunk_days: int,
    pause_sec: float,
) -> list[pd.DataFrame]:
    """Pull Statcast in rate-limit-friendly chunks. Returns non-empty chunks.

    Verified empirically against the live Savant CSV endpoint: gt/lt bounds
    are INCLUSIVE (statcast(D, D) returns day D). Empty results for recent
    dates mean Savant hasn't posted them yet — posting lags game completion
    by hours to a day — NOT a query-semantics problem.
    """
    from pybaseball import statcast

    chunks: list[pd.DataFrame] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        logger.info("  Chunk: %s → %s", cursor, chunk_end)
        try:
            df = statcast(str(cursor), str(chunk_end))
            if df is not None and not df.empty:
                chunks.append(df)
                logger.info("    → %d pitches", len(df))
        except Exception as e:
            logger.warning("    → Chunk failed: %s", e)
        cursor = chunk_end + timedelta(days=1)
        if cursor <= end:
            time.sleep(pause_sec)
    return chunks


# Re-pull this many trailing days on every resume so games captured mid-game
# (partial Statcast posts) get their real finals instead of frozen scores.
REFRESH_TAIL_DAYS = 3


def _cache_bounds(path: Path) -> tuple[date | None, date | None]:
    """(earliest, latest) game_date in a cached parquet; (None, None) if
    unreadable. Both ends are needed: the max detects stale tails (the usual
    daily case), the min detects a request to extend history earlier."""
    try:
        gd = pd.read_parquet(path, columns=["game_date"])["game_date"]
        lo = pd.Timestamp(gd.min()).date() if pd.notna(gd.min()) else None
        hi = pd.Timestamp(gd.max()).date() if pd.notna(gd.max()) else None
        return lo, hi
    except Exception as e:
        logger.warning("Could not read cache date bounds from %s: %s", path, e)
        return None, None


def _merge_and_save(
    out: Path,
    inc: pd.DataFrame,
    existing_first: bool,
) -> Path:
    """Merge an increment with the existing cache, dedupe on pitch identity,
    and overwrite the file. ``existing_first`` puts cache rows before the
    increment so re-delivered rows resolve to the NEWER copy (keep=last)."""
    existing = pd.read_parquet(out)
    n_existing = len(existing)
    parts = ([existing, inc] if existing_first else [inc, existing])
    merged = pd.concat(parts, ignore_index=True)
    del existing, inc
    n_before_dedupe = len(merged)
    merged = merged.drop_duplicates(
        subset=[c for c in ("game_pk", "at_bat_number", "pitch_number")
                if c in merged.columns],
        keep="last",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    logger.info(
        "Cache updated: %d pitches now (%d new vs old cache, %d overlap rows dropped) → %s",
        len(merged), max(len(merged) - n_existing, 0),
        n_before_dedupe - len(merged), out,
    )
    del merged
    return out


def pull_statcast(
    start_date: str | date,
    end_date: str | date,
    out_path: str | Path = "pitches.parquet",
    chunk_days: int = 7,
    pause_sec: float = 2.0,
    resume: bool = True,
) -> Path:
    """Pull Statcast data and save to Parquet.

    Args:
        start_date:  Inclusive start (YYYY-MM-DD or date).
        end_date:    Inclusive end.
        out_path:    Where to write the Parquet file.
        chunk_days:  Days per API chunk (Statcast rate-limits large queries).
        pause_sec:   Seconds to pause between chunks.
        resume:      If True and a cached file exists, reuse it — but ONLY
                     when its coverage covers the full request. A stale tail
                     gets a forward top-up (cached_max+1 → end); a requested
                     start EARLIER than the cache's first day back-fills the
                     head. Set resume=False (or MLB_FULL_REPULL=1 in
                     master_pipeline) for a clean full historical re-pull.

    Returns:
        Path to the written Parquet file.

    Raises:
        ValueError: If no data is returned by pybaseball.
    """
    out = Path(out_path)
    start = _to_date(start_date)
    end = _to_date(end_date)

    if resume and out.exists():
        cached_lo, cached_hi = _cache_bounds(out)
        if cached_hi is None:
            logger.warning("Existing cache unreadable — re-pulling full range")
        else:
            # NOTE: there is deliberately NO "cache fully covers the range —
            # skip" shortcut here. A prior run can cache PARTIAL data for the
            # most recent days (games pulled mid-progress), making the bounds
            # look complete while finals are frozen. The tail refresh below
            # re-pulls the trailing window on every resume so those games get
            # their real finals; the merge dedupes to the newest copy.

            # Backward gap first (history extension): start .. cached_min-1
            if cached_lo is not None and start < cached_lo:
                back_end = cached_lo - timedelta(days=1)
                logger.info(
                    "Request starts before cache (%s < %s) — back-filling %s → %s",
                    start, cached_lo, start, back_end,
                )
                chunks_back = _chunked_statcast(start, back_end, chunk_days, pause_sec)
                if chunks_back:
                    inc = _normalize_columns(pd.concat(chunks_back, ignore_index=True))
                    del chunks_back
                    inc = _downcast(inc)
                    for col in UNUSED_COLS:
                        if col in inc.columns:
                            inc.drop(columns=[col], inplace=True)
                    out = _merge_and_save(out, inc, existing_first=False)
                else:
                    logger.warning("No Statcast data found for %s → %s", start, back_end)
                # Recompute bounds after prepending
                cached_lo, cached_hi = _cache_bounds(out)

            # Forward top-up + TAIL REFRESH. Two failure modes this fixes:
            # 1. cached_hi < end: plain forward gap (new days to append).
            # 2. cached_hi >= end: cache LOOKS current but a prior run may
            #    have stored PARTIAL data for recent days (games pulled mid-
            #    progress keep frozen scores forever unless revisited).
            # Either way we re-pull the trailing REFRESH_TAIL_DAYS window;
            # merge dedupe keeps the NEWEST copy of each pitch, so completed
            # games overwrite their partial selves.
            if cached_hi is not None:
                refresh_start = end - timedelta(days=REFRESH_TAIL_DAYS - 1)
                inc_start = min(cached_hi + timedelta(days=1), refresh_start)
                inc_start = max(inc_start, start)
                logger.info(
                    "Cache covers through %s — pulling %s → %s "
                    "(includes %d-day tail refresh to fix in-progress games)",
                    cached_hi, inc_start, end, REFRESH_TAIL_DAYS,
                )
                inc_chunks = _chunked_statcast(inc_start, end, chunk_days, pause_sec)
                if not inc_chunks:
                    if cached_hi < end:
                        logger.warning(
                            "No Statcast data posted for %s → %s yet — keeping cache "
                            "(Savant lags game completion by hours to a day)",
                            inc_start, end,
                        )
                    else:
                        logger.error(
                            "Tail refresh returned ZERO pitches — likely Savant rate "
                            "limiting. STALE CACHE KEPT: recent finals may be frozen or "
                            "missing. Wait a few minutes and re-run; if it persists, "
                            "set MLB_FULL_REPULL=1 for one run."
                        )
                    return out
                inc = _normalize_columns(pd.concat(inc_chunks, ignore_index=True))
                del inc_chunks
                inc = _downcast(inc)
                for col in UNUSED_COLS:
                    if col in inc.columns:
                        inc.drop(columns=[col], inplace=True)
                out = _merge_and_save(out, inc, existing_first=True)
                new_lo, new_hi = _cache_bounds(out)
                logger.info("Cache now covers %s → %s", new_lo, new_hi)
                return out

            return out

    logger.info("Pulling Statcast: %s → %s (chunk_days=%d)", start, end, chunk_days)

    chunks = _chunked_statcast(start, end, chunk_days, pause_sec)

    if not chunks:
        raise ValueError(f"No Statcast data for {start} to {end}")

    raw = pd.concat(chunks, ignore_index=True)
    del chunks
    # Trimmed chunk ranges should not overlap, but boundary rows from vendor
    # quirks are cheap to guard against.
    before = len(raw)
    raw = raw.drop_duplicates(
        subset=[c for c in ("game_pk", "at_bat_number", "pitch_number")
                if c in raw.columns],
        keep="first",
    )
    if len(raw) < before:
        logger.warning("Dropped %d duplicate pitches at chunk boundaries", before - len(raw))
    logger.info("Total raw pitches: %d", len(raw))
    raw = _normalize_columns(raw)
    raw = _downcast(raw)

    for col in UNUSED_COLS:
        if col in raw.columns:
            raw.drop(columns=[col], inplace=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(out, index=False)
    logger.info("Saved %d pitches → %s (%.1f MB)", len(raw), out, out.stat().st_size / 1e6)

    del raw
    return out


# ── Helpers ─────────────────────────────────────────────────────────────────

def _to_date(d: str | date) -> date:
    if isinstance(d, date):
        return d
    return date.fromisoformat(d)


# Statcast game_type codes to KEEP. 'S' = Spring Training and 'E' =
# Exhibition are excluded: Savant posts pitch data for them, so a pull
# spanning Feb–March silently ingests hundreds of scrimmages whose stats
# are not representative of regular-season play.
KEEP_GAME_TYPES = {"R", "F", "D", "L", "W"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply aliases and ensure all expected columns exist (missing → NaN).

    Also drops non-regular-season games (spring training, exhibitions);
    postseason codes (F/D/L/W) are kept.
    """
    n_before = len(df)
    if "game_type" in df.columns:
        df = df[df["game_type"].astype(str).str.strip().isin(KEEP_GAME_TYPES)]
        dropped = n_before - len(df)
        if dropped:
            logger.info("Dropped %d non-regular-season pitches (game_type S/E)", dropped)
    else:
        logger.warning("game_type column missing — cannot filter spring training games")
    rename_map = {}
    for col in df.columns:
        canonical = COLUMN_ALIASES.get(col)
        if canonical and canonical != col:
            rename_map[col] = canonical
    if rename_map:
        logger.info("Renamed aliased columns: %s", rename_map)
        df = df.rename(columns=rename_map)

    for col in STATCAST_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numerics and convert low-cardinality strings to category."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        # Cannot downcast int64 to int16/int32 if ANY value is NaN
        # (pandas raises ValueError: cannot convert NA to integer)
        if df[col].isna().any():
            continue
        col_max = df[col].max()
        if col_max < 32767 and df[col].min() >= -32768:
            df[col] = df[col].astype("int16")
        else:
            df[col] = df[col].astype("int32")

    _cat_cols = [
        "game_type", "home_team", "away_team", "inning_topbot",
        "pitch_type", "description", "events", "p_throws", "stand",
    ]
    for col in _cat_cols:
        if col in df.columns and df[col].dtype == "object" and df[col].nunique() < 100:
            df[col] = df[col].astype("category")

    return df
