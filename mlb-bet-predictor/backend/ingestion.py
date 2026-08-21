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
        resume:      If True and out_path exists, skip the pull.

    Returns:
        Path to the written Parquet file.

    Raises:
        ValueError: If no data is returned by pybaseball.
    """
    out = Path(out_path)

    if resume and out.exists():
        logger.info("Resuming from existing file: %s", out)
        return out

    from pybaseball import statcast

    start = _to_date(start_date)
    end = _to_date(end_date)

    logger.info("Pulling Statcast: %s → %s (chunk_days=%d)", start, end, chunk_days)

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

    if not chunks:
        raise ValueError(f"No Statcast data for {start} to {end}")

    raw = pd.concat(chunks, ignore_index=True)
    del chunks
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


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply aliases and ensure all expected columns exist (missing → NaN)."""
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
        if df[col].max() < 32767 and df[col].min() >= -32768:
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
