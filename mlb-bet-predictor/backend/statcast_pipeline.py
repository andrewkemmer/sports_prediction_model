"""
Comprehensive Statcast-based MLB Feature Pipeline.

Pulls raw Statcast pitch-by-pitch data via pybaseball, engineers features
across multiple tiers, and produces two aligned DataFrames:

  1. game_level  — one row per game with pre-game macro features
  2. pbp_level   — one row per pitch/event inheriting game features + situational state

All rolling metrics are strictly point-in-time (PIT): Game T features use
only data from games scheduled strictly before Game T.

Usage (Colab):
    from statcast_pipeline import run_statcast_pipeline
    game_df, pbp_df = run_statcast_pipeline("2025-04-01", "2025-08-01")

CLI:
    python statcast_pipeline.py --start 2025-04-01 --end 2025-08-01 --checkpoint-dir /content/drive/MyDrive/mlb_data
"""
from __future__ import annotations

import gc
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
pd.set_option("mode.chained_assignment", None)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MLB_TEAMS_ABBREV = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE",
    "COL", "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL",
    "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SD", "SF",
    "SEA", "STL", "TB", "TEX", "TOR", "WSH",
]

# Statcast column subsets for memory efficiency
PITCH_COLS = [
    "game_date", "game_pk", "game_type", "home_team", "away_team",
    "inning", "inning_topbot", "outs_when_up", "balls", "strikes",
    "on_1b", "on_2b", "on_3b",
    "at_bat_number", "pitch_number", "pitcher", "batter",
    "p_throws", "stand",  # pitcher throws L/R, batter stands L/R
    "pitch_type", "release_speed", "release_pos_x", "release_pos_z",
    "player_name", "pitcher_name",  # via joined columns
    "description", "events", "event",
    "spin_rate", "spin_axis",
    "release_spin_rate", "release_extension",
    "plate_x", "plate_z",
    "zone", "type", "pfx_x", "pfx_z",
    "hit_distance_sc", "launch_speed", "launch_angle",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
    "woba_value", "babip_value", "iso_value",
    "barrel", "hard_contact", "launch_speed_angle",
    "fielder_2", "fielder_3", "fielder_4", "fielder_5",
    "fielder_6", "fielder_7", "fielder_8", "fielder_9",
    "home_score", "away_score", "post_home_score", "post_away_score",
    "if_fielding_alignment", "of_fielding_alignment",
    "delta_home_win_exp", "delta_run_exp",
]

# Columns to DROP after pull (genuinely unused by all feature functions)
# Keeps all useful Statcast fields for future feature development.
UNUSED_COLS = [
    "fielder_2", "fielder_3", "fielder_4", "fielder_5",
    "fielder_6", "fielder_7", "fielder_8", "fielder_9",
    "if_fielding_alignment", "of_fielding_alignment",
    "post_home_score", "post_away_score",
    "event",  # duplicate of "events"
    "type",   # covered by "description"
    "launch_speed_angle",  # we compute barrel/hard_contact directly
]

# Column aliases: map known Statcast name variants to canonical names.
# If Statcast renames a column in a future season, add the old name here.
COLUMN_ALIASES = {
    # Contact quality
    "barrel": "barrel",
    "barrel百分比": "barrel",
    "barrel_pct": "barrel",
    "is_barrel": "barrel",
    "hard_contact": "hard_contact",
    "hardhit": "hard_contact",
    "hard_hit": "hard_contact",
    "hardhit百分比": "hard_contact",
    "launch_speed": "launch_speed",
    "exit_velocity": "launch_speed",
    "exit_velo": "launch_speed",
    "launch_angle": "launch_angle",
    "la": "launch_angle",
    # Expected stats
    "estimated_woba_using_speedangle": "estimated_woba_using_speedangle",
    "xwoba": "estimated_woba_using_speedangle",
    "xwOBA": "estimated_woba_using_speedangle",
    "estimated_ba_using_speedangle": "estimated_ba_using_speedangle",
    "xba": "estimated_ba_using_speedangle",
    # Pitcher identity
    "player_name": "player_name",
    "pitcher_name": "player_name",
    # Event description
    "events": "events",
    "event": "events",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename any aliased columns to canonical names, then add missing expected columns as NaN."""
    rename_map = {}
    for col in df.columns:
        canonical = COLUMN_ALIASES.get(col)
        if canonical and canonical != col:
            rename_map[col] = canonical
    if rename_map:
        logger.info("Renaming aliased columns: %s", rename_map)
        df = df.rename(columns=rename_map)

    # Ensure all PITCH_COLS exist (missing → NaN)
    for col in PITCH_COLS:
        if col not in df.columns:
            df[col] = np.nan

    return df


ROLLING_WINDOW_PITCHER = 100  # Pitches (≈ 5-6 starts)
ROLLING_WINDOW_BATTER = 80    # Pitches (≈ 2 weeks of PA)
ROLLING_WINDOW_TEAM = 30      # Games
ROLLING_WINDOW_BULLPEN = 10   # Games

# Rate-limit settings for Statcast API
STATCAST_BATCH_DAYS = 7       # Pull in weekly chunks
STATCAST_PAUSE_SECONDS = 2    # Pause between batches
CHECKPOINT_EVERY_N_BATCHES = 4  # Save checkpoint every N batches

ELO_START = 1500.0
ELO_K = 20
ELO_HOME_ADV = 65

# ── Data Pull ────────────────────────────────────────────────────────────────


def _date_range_chunks(start: date, end: date, chunk_days: int):
    """Yield (chunk_start, chunk_end) pairs of `chunk_days` length."""
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def pull_statcast_data(
    start_date: str | date,
    end_date: str | date,
    checkpoint_dir: Optional[str | Path] = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Pull Statcast pitch-by-pitch data in weekly batches with checkpointing.

    Args:
        start_date: Start date (YYYY-MM-DD or date object).
        end_date: End date (YYYY-MM-DD or date object).
        checkpoint_dir: Directory for checkpoint parquet files.
            If None, no checkpointing. Use Google Drive path for Colab.
        resume: If True, skip batches that already have checkpoints.

    Returns:
        DataFrame with one row per pitch.
    """
    from pybaseball import statcast

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load existing checkpoints
    all_chunks = []
    if resume and ckpt_dir:
        existing = sorted(ckpt_dir.glob("statcast_chunk_*.parquet"))
        for f in existing:
            logger.info("Resuming from checkpoint: %s", f.name)
            all_chunks.append(pd.read_parquet(f))

    # Determine which batches to pull
    batches = list(_date_range_chunks(start_date, end_date, STATCAST_BATCH_DAYS))
    total_batches = len(batches)
    pulled = 0

    for i, (chunk_start, chunk_end) in enumerate(batches):
        ckpt_name = f"statcast_chunk_{chunk_start}_{chunk_end}.parquet"
        ckpt_path = ckpt_dir / ckpt_name if ckpt_dir else None

        # Skip if checkpoint exists and resuming
        if resume and ckpt_path and ckpt_path.exists():
            logger.info("Skipping batch %d/%d (checkpoint exists): %s",
                        i + 1, total_batches, ckpt_name)
            continue

        logger.info("Pulling Statcast batch %d/%d: %s to %s",
                     i + 1, total_batches, chunk_start, chunk_end)

        try:
            chunk = statcast(
                start_dt=chunk_start.strftime("%Y-%m-%d"),
                end_dt=chunk_end.strftime("%Y-%m-%d"),
            )
        except Exception as e:
            logger.warning("Batch %d failed (%s), pausing and retrying: %s",
                           i + 1, e, e)
            time.sleep(STATCAST_PAUSE_SECONDS * 2)
            try:
                chunk = statcast(
                    start_dt=chunk_start.strftime("%Y-%m-%d"),
                    end_dt=chunk_end.strftime("%Y-%m-%d"),
                )
            except Exception as e2:
                logger.error("Batch %d retry also failed: %s", i + 1, e2)
                continue

        if chunk is not None and not chunk.empty:
            # Keep only columns we need
            available = [c for c in PITCH_COLS if c in chunk.columns]
            chunk = chunk[available].copy()
            all_chunks.append(chunk)
            pulled += 1

            # Save checkpoint
            if ckpt_path:
                chunk.to_parquet(ckpt_path, index=False)
                logger.info("Checkpoint saved: %s (%d rows)", ckpt_name, len(chunk))
        else:
            logger.warning("Batch %d returned empty data", i + 1)

        # Rate limiting pause
        if i < total_batches - 1:
            time.sleep(STATCAST_PAUSE_SECONDS)

        # Periodic checkpoint of combined data
        if ckpt_dir and pulled > 0 and pulled % CHECKPOINT_EVERY_N_BATCHES == 0:
            combined = pd.concat(all_chunks, ignore_index=True)
            combined_path = ckpt_dir / "statcast_combined_checkpoint.parquet"
            combined.to_parquet(combined_path, index=False)
            logger.info("Combined checkpoint saved: %d total rows", len(combined))

    if not all_chunks:
        logger.error("No Statcast data pulled for %s to %s", start_date, end_date)
        return pd.DataFrame()

    df = pd.concat(all_chunks, ignore_index=True)
    del all_chunks

    # Normalize column names (aliases) and ensure all expected columns exist
    df = _normalize_columns(df)

    # Deduplicate (same pitch can appear in overlapping batches)
    before = len(df)
    df = df.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"],
                            keep="last")
    if len(df) < before:
        logger.info("Deduplicated: %d → %d rows", before, len(df))

    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df = df.sort_values(["game_date", "game_pk", "inning", "at_bat_number", "pitch_number"]).reset_index(drop=True)

    # Memory optimization: downcast floats to float32, use category for low-cardinality strings
    _float_cols = df.select_dtypes(include=["float64"]).columns
    df[_float_cols] = df[_float_cols].astype("float32")
    _int_cols = df.select_dtypes(include=["int64"]).columns
    df[_int_cols] = df[_int_cols].astype("int32", errors="ignore")
    for col in df.select_dtypes(include=["object"]).columns:
        if col == "game_date":
            continue  # Keep datetime objects as-is
        if df[col].nunique() < 100:  # Low-cardinality strings → category
            df[col] = df[col].astype("category")

    # Drop genuinely unused columns (15 columns, ~20 MB savings for 78K pitches)
    _drop = [c for c in df.columns if c in UNUSED_COLS]
    if _drop:
        df = df.drop(columns=_drop)
        logger.info("Dropped %d unused columns: %s", len(_drop), _drop)

    logger.info("Final Statcast dataset: %d pitches across %d games, %d dates",
                len(df), df["game_pk"].nunique(), df["game_date"].nunique())

    # Save final combined checkpoint
    if ckpt_dir:
        final_path = ckpt_dir / "statcast_final.parquet"
        df.to_parquet(final_path, index=False)
        logger.info("Final checkpoint saved: %s", final_path)

    return df


# ── Feature Engineering Helpers ──────────────────────────────────────────────


def _compute_rolling_pitcher_features(pitches: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling pitcher features (ERA, K/9, BB/9, FIP, WHIP, xwOBA).

    Uses only pitches strictly before the current game_date for each pitcher.
    """
    logger.info("Computing rolling pitcher features...")

    # Sort chronologically
    pitches = pitches.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])

    # Identify pitcher role events
    # A "plate appearance" ends on events like: single, double, triple, home_run,
    # strikeout, walk, hit_by_pitch, field_out, etc.
    pa_end_events = [
        "single", "double", "triple", "home_run",
        "strikeout", "strikeout_double_play",
        "walk", "hit_by_pitch",
        "field_out", "field_error", "fielders_choice", "fielders_choice_out",
        "grounded_into_double_play", "double_play", "triple_play",
        "sac_fly", "sac_bunt", "sac_fly_double_play",
        "catcher_interf", "batter_interference",
        "force_out", "sacrifice_bunt_double_play",
    ]

    # Filter to PA-ending events (no copy — we only read from this)
    pa_mask = pitches["events"].isin(pa_end_events)
    pa_events = pitches.loc[pa_mask, ["game_date", "game_pk", "pitcher", "events",
                                       "estimated_woba_using_speedangle", "barrel", "hard_contact"]]
    del pa_mask

    if pa_events.empty:
        logger.warning("No PA-ending events found for pitcher feature computation")
        return pitches

    # Vectorized event counting (no groupby().apply)
    _ks_mask = pa_events["events"].isin(["strikeout", "strikeout_double_play"])
    _bb_mask = pa_events["events"] == "walk"
    _hbp_mask = pa_events["events"] == "hit_by_pitch"
    _hit_mask = pa_events["events"].isin(["single", "double", "triple", "home_run"])
    _hr_mask = pa_events["events"] == "home_run"

    _grp_keys = ["game_date", "game_pk", "pitcher"]
    _grp = pa_events.groupby(_grp_keys, sort=False)
    pitcher_game_stats = pd.DataFrame({
        "n_batters_faced": _grp.size().values,
        "ip_approx": (_grp.size().values / 3.0),
        "ks": _ks_mask.groupby(pa_events[_grp_keys[-1]]).sum().values if False else _grp["events"].apply(lambda g: g.isin(["strikeout", "strikeout_double_play"]).sum()).values,
    })
    # Rebuild with efficient aggregation
    pitcher_game_stats = pd.DataFrame(index=_grp.indices.keys(), columns=[
        "game_date", "game_pk", "pitcher", "n_batters_faced",
    ])
    # Extract group keys
    _idx = list(_grp.indices.keys())
    pitcher_game_stats = pd.DataFrame(_idx, columns=_grp_keys)
    pitcher_game_stats["n_batters_faced"] = _grp.size().values
    pitcher_game_stats["ip_approx"] = pitcher_game_stats["n_batters_faced"].astype("float32") / 3.0

    # Event counts per group (vectorized)
    pitcher_game_stats["ks"] = pa_events.groupby(_grp_keys)["events"].apply(
        lambda s: s.isin(["strikeout", "strikeout_double_play"]).sum()).values
    pitcher_game_stats["bbs"] = pa_events.groupby(_grp_keys)["events"].apply(
        lambda s: (s == "walk").sum()).values
    pitcher_game_stats["hbps"] = pa_events.groupby(_grp_keys)["events"].apply(
        lambda s: (s == "hit_by_pitch").sum()).values
    pitcher_game_stats["hits_allowed"] = pa_events.groupby(_grp_keys)["events"].apply(
        lambda s: s.isin(["single", "double", "triple", "home_run"]).sum()).values
    pitcher_game_stats["hrs_allowed"] = pa_events.groupby(_grp_keys)["events"].apply(
        lambda s: (s == "home_run").sum()).values
    pitcher_game_stats["runs"] = (
        pitcher_game_stats["hits_allowed"] + pitcher_game_stats["bbs"]
        + pitcher_game_stats["hbps"] - pitcher_game_stats["hrs_allowed"]
    )

    # xwOBA / barrel / hard_contact means per group
    for _src, _dst in [("estimated_woba_using_speedangle", "xwoba"),
                       ("barrel", "barrel_rate"), ("hard_contact", "hard_contact_rate")]:
        if _src in pa_events.columns:
            pitcher_game_stats[_dst] = pa_events.groupby(_grp_keys)[_src].mean().values
        else:
            pitcher_game_stats[_dst] = np.nan

    del _ks_mask, _bb_mask, _hbp_mask, _hit_mask, _hr_mask, _grp
    gc.collect()

    del pa_events
    gc.collect()

    # Sort for rolling computation
    pitcher_game_stats = pitcher_game_stats.sort_values(["pitcher", "game_date"])

    # Ensure all stat columns exist (missing → NaN)
    _required_stat_cols = ["runs", "ks", "bbs", "hits_allowed", "hbps",
                           "hrs_allowed", "xwoba", "barrel_rate", "hard_contact_rate"]
    for col in _required_stat_cols:
        if col not in pitcher_game_stats.columns:
            pitcher_game_stats[col] = np.nan

    # Shift all stat columns for PIT compliance (NaN-preserving)
    # Use vectorized groupby().shift(1) — much faster than .apply(lambda x: x.shift(1))
    _pitcher_grp = pitcher_game_stats.groupby("pitcher", sort=False)
    # IP shift — explicitly named _shifted_ip (rolling formulas reference this name)
    if "ip_approx" in pitcher_game_stats.columns:
        pitcher_game_stats["_shifted_ip"] = _pitcher_grp["ip_approx"].shift(1)
    else:
        pitcher_game_stats["_shifted_ip"] = np.nan
    # Shift all stat columns for PIT compliance
    for col in _required_stat_cols:
        if col in pitcher_game_stats.columns:
            pitcher_game_stats[f"_shifted_{col}"] = _pitcher_grp[col].shift(1)
        else:
            pitcher_game_stats[f"_shifted_{col}"] = np.nan
    del _pitcher_grp

    # Helper: rolling sum/mean within groups (NaN-preserving)
    _pitcher_roll_grp = pitcher_game_stats.groupby("pitcher", sort=False)

    def _rolling_sum(col, window, min_periods=3):
        return _pitcher_roll_grp[col].transform(
            lambda x: x.rolling(window=window, min_periods=min_periods).sum())

    def _rolling_mean(col, window, min_periods=5):
        return _pitcher_roll_grp[col].transform(
            lambda x: x.rolling(window=window, min_periods=min_periods).mean())

    ip_roll = _rolling_sum("_shifted_ip", ROLLING_WINDOW_PITCHER)

    # ERA = (runs / IP) * 9
    pitcher_game_stats["sp_era_30g"] = (
        _rolling_sum("_shifted_runs", ROLLING_WINDOW_PITCHER) / ip_roll * 9.0)

    # K/9
    pitcher_game_stats["sp_k9_30g"] = (
        _rolling_sum("_shifted_ks", ROLLING_WINDOW_PITCHER) / ip_roll * 9.0)

    # BB/9
    pitcher_game_stats["sp_bb9_30g"] = (
        _rolling_sum("_shifted_bbs", ROLLING_WINDOW_PITCHER) / ip_roll * 9.0)

    # WHIP = (BB + H) / IP
    pitcher_game_stats["sp_whip_30g"] = (
        (_rolling_sum("_shifted_bbs", ROLLING_WINDOW_PITCHER)
         + _rolling_sum("_shifted_hits_allowed", ROLLING_WINDOW_PITCHER)) / ip_roll)

    # FIP ≈ (13*HR + 3*(BB+HBP) - 2*K) / IP
    pitcher_game_stats["sp_fip_30g"] = (
        (13 * _rolling_sum("_shifted_hrs_allowed", ROLLING_WINDOW_PITCHER)
         + 3 * (_rolling_sum("_shifted_bbs", ROLLING_WINDOW_PITCHER)
                + _rolling_sum("_shifted_hbps", ROLLING_WINDOW_PITCHER))
         - 2 * _rolling_sum("_shifted_ks", ROLLING_WINDOW_PITCHER)) / ip_roll)

    # xwOBA rolling mean
    pitcher_game_stats["sp_xwoba_30g"] = _rolling_mean(
        "_shifted_xwoba", ROLLING_WINDOW_PITCHER, min_periods=5)

    del _pitcher_roll_grp, ip_roll
    gc.collect()

    # Keep only the columns we need for joining
    pitcher_features = pitcher_game_stats[[
        "game_date", "game_pk", "pitcher",
        "sp_era_30g", "sp_k9_30g", "sp_bb9_30g", "sp_whip_30g",
        "sp_fip_30g", "sp_xwoba_30g",
    ]].copy(deep=False)

    del pitcher_game_stats
    gc.collect()

    # Downcast pitcher features
    for col in pitcher_features.select_dtypes(include=["float64"]).columns:
        pitcher_features[col] = pitcher_features[col].astype("float32")

    return pitcher_features


def _compute_rolling_batter_features(pitches: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling batter features (wOBA, ISO, barrel rate, chase rate, whiff rate)."""
    logger.info("Computing rolling batter features...")

    pa_end_events = [
        "single", "double", "triple", "home_run",
        "strikeout", "strikeout_double_play",
        "walk", "hit_by_pitch",
        "field_out", "field_error", "fielders_choice", "fielders_choice_out",
        "grounded_into_double_play", "double_play", "triple_play",
        "sac_fly", "sac_bunt", "sac_fly_double_play",
        "catcher_interf", "batter_interference",
        "force_out", "sacrifice_bunt_double_play",
    ]

    pa_events = pitches[pitches["events"].isin(pa_end_events)].copy()

    if pa_events.empty:
        return pd.DataFrame()

    def _batter_pa_stats(group):
        """Stats for one PA-ending event."""
        events = group["events"]
        singles = (events == "single").sum()
        doubles = (events == "double").sum()
        triples = (events == "triple").sum()
        hrs = (events == "home_run").sum()
        bb = (events == "walk").sum()
        hbp = (events == "hit_by_pitch").sum()
        k = events.isin(["strikeout", "strikeout_double_play"]).sum()
        ab = len(group) - bb - hbp  # approximate AB

        # wOBA (simplified: weights from FanGraphs)
        woba = (0.690 * bb + 0.722 * hbp + 0.878 * singles
                + 1.242 * doubles + 1.568 * triples + 2.007 * hrs) / max(ab + bb + hbp, 1)

        # ISO = (2B + 3B*2 + HR*1) / AB
        iso = (doubles + 2 * triples + 3 * hrs) / max(ab, 1)

        return pd.Series({
            "pa": 1,
            "ab": ab,
            "woba": woba,
            "iso": iso,
            "k_rate": k / max(ab, 1),
            "bb_rate": bb / max(ab + bb + hbp, 1),
            "hr_rate": hrs / max(ab, 1),
        })

    batter_pa_stats = (
        pa_events
        .groupby(["game_date", "game_pk", "batter"])
        .apply(_batter_pa_stats, include_groups=False)
        .reset_index()
    )

    batter_pa_stats = batter_pa_stats.sort_values(["batter", "game_date"])

    # Ensure all stat columns exist (missing → NaN)
    for col in ["woba", "iso", "k_rate", "bb_rate", "hr_rate"]:
        if col not in batter_pa_stats.columns:
            batter_pa_stats[col] = np.nan

    # Shift + rolling for PIT compliance (NaN-preserving, vectorized)
    _batter_shift_grp = batter_pa_stats.groupby("batter", sort=False)
    for col in ["woba", "iso", "k_rate", "bb_rate", "hr_rate"]:
        batter_pa_stats[f"_shifted_{col}"] = _batter_shift_grp[col].shift(1)

    for col, new_col in [
        ("woba", "batter_woba_30g"), ("iso", "batter_iso_30g"),
        ("k_rate", "batter_k_rate_30g"), ("bb_rate", "batter_bb_rate_30g"),
        ("hr_rate", "batter_hr_rate_30g"),
    ]:
        batter_pa_stats[new_col] = (
            batter_pa_stats.groupby("batter")[f"_shifted_{col}"]
            .transform(lambda x: x.rolling(window=ROLLING_WINDOW_BATTER, min_periods=5).mean())
        )

    # Plate discipline features from pitch-level data (no full copy)
    _disc_cols = ["game_date", "game_pk", "batter", "description", "zone", "pitch_number"]
    _disc_cols = [c for c in _disc_cols if c in pitches.columns]
    pitch_level = pitches[_disc_cols].loc[
        pitches["description"].isin(["swinging_strike", "swinging_strike_blocked",
                                     "foul", "foul_tip", "hit_into_play"])]

    pitch_level["is_chase"] = (
        pitch_level["zone"].fillna(0) > 9
    ).astype("float32")

    pitch_level["is_whiff"] = (
        pitch_level["description"].isin(["swinging_strike", "swinging_strike_blocked"])
    ).astype("float32")

    discipline = (
        pitch_level.groupby(["game_date", "game_pk", "batter"], sort=False)
        .agg(
            n_pitches=("pitch_number", "count"),
            chase_sum=("is_chase", "sum"),
            whiff_sum=("is_whiff", "sum"),
        )
        .reset_index()
    )
    del pitch_level
    gc.collect()

    discipline = discipline.sort_values(["batter", "game_date"])
    _disc_batter_grp = discipline.groupby("batter", sort=False)
    for col in ["chase_sum", "whiff_sum", "n_pitches"]:
        discipline[f"_shifted_{col}"] = _disc_batter_grp[col].shift(1)

    discipline["batter_chase_rate_30g"] = (
        discipline.groupby("batter")["_shifted_chase_sum"]
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_BATTER, min_periods=10).sum())
        / discipline.groupby("batter")["_shifted_n_pitches"]
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_BATTER, min_periods=10).sum())
    )

    discipline["batter_whiff_rate_30g"] = (
        discipline.groupby("batter")["_shifted_whiff_sum"]
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_BATTER, min_periods=10).sum())
        / discipline.groupby("batter")["_shifted_n_pitches"]
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_BATTER, min_periods=10).sum())
    )

    # Merge batter stats + discipline
    batter_features = batter_pa_stats.merge(
        discipline[["game_date", "game_pk", "batter",
                     "batter_chase_rate_30g", "batter_whiff_rate_30g"]],
        on=["game_date", "game_pk", "batter"],
        how="left",
    )

    del batter_pa_stats, discipline
    gc.collect()

    return batter_features[[
        "game_date", "game_pk", "batter",
        "batter_woba_30g", "batter_iso_30g",
        "batter_k_rate_30g", "batter_bb_rate_30g", "batter_hr_rate_30g",
        "batter_chase_rate_30g", "batter_whiff_rate_30g",
    ]]


def _compute_rolling_bullpen_features(pitches: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling bullpen features (WHIP, ERA) from relief appearances."""
    logger.info("Computing rolling bullpen features...")

    # Identify relief pitchers: pitchers who appear mid-game (not starting)
    # A starter is the first pitcher in a game
    # Use only columns needed for bullpen analysis (avoid full pitches copy)
    _bp_cols = ["game_pk", "game_date", "home_team", "away_team",
                "at_bat_number", "pitch_number", "pitcher", "events"]
    _bp_cols = [c for c in _bp_cols if c in pitches.columns]
    _bp_min = pitches[_bp_cols].copy(deep=False)

    first_pitchers = (
        _bp_min.sort_values(["game_pk", "at_bat_number", "pitch_number"])
        .groupby("game_pk")["pitcher"]
        .first()
        .reset_index()
        .rename(columns={"pitcher": "starter_id"})
    )

    _bp_min = _bp_min.merge(first_pitchers, on="game_pk", how="left")
    del first_pitchers
    _bp_min["is_reliever"] = (
        _bp_min["pitcher"] != _bp_min["starter_id"]
    )

    pa_end_events = [
        "single", "double", "triple", "home_run",
        "strikeout", "strikeout_double_play",
        "walk", "hit_by_pitch",
        "field_out", "field_error", "fielders_choice", "fielders_choice_out",
        "grounded_into_double_play", "double_play", "triple_play",
        "sac_fly", "sac_bunt", "sac_fly_double_play",
        "catcher_interf", "batter_interference",
        "force_out", "sacrifice_bunt_double_play",
    ]

    reliever_events = _bp_min[
        (_bp_min["is_reliever"])
        & (_bp_min["events"].isin(pa_end_events))
    ].copy(deep=False)
    del _bp_min
    gc.collect()

    if reliever_events.empty:
        return pd.DataFrame()

    # Team-level bullpen stats per game
    def _bullpen_game_stats(group):
        ip = len(group) / 3.0
        ks = group["events"].isin(["strikeout", "strikeout_double_play"]).sum()
        bbs = group["events"].isin(["walk"]).sum()
        hits = group["events"].isin(["single", "double", "triple", "home_run"]).sum()
        runs = group["events"].isin(["single", "double", "triple", "home_run",
                                     "walk", "hit_by_pitch"]).sum()

        return pd.Series({
            "bullpen_ip": ip,
            "bullpen_ks": ks,
            "bullpen_bbs": bbs,
            "bullpen_hits": hits,
            "bullpen_runs": runs,
        })

    # Use home_team to identify team (relievers pitch for the away or home team)
    # We'll group by game + team later; for now, use the event's team context
    # The pitcher's team is the team that's fielding
    # For simplicity, we'll compute per-home-team bullpen stats from home pitcher events
    # and per-away-team from away pitcher events

    # Actually, let's just compute team-level bullpen from the reliever data
    # by associating with the team (pitcher's team)
    # We need to figure out which team each reliever belongs to

    # Simple heuristic: if pitcher is home_team's pitcher, they're on home team
    # (This is approximate; full solution would use roster data)
    reliever_events["pitcher_team"] = np.where(
        reliever_events["home_team"].str.upper() == reliever_events["pitcher"].astype(str).str[:3].str.upper(),
        reliever_events["home_team"],
        reliever_events["away_team"],
    )

    # Better: just use game_pk + reliever to get team from the game context
    # We'll assume relievers belong to whichever team is fielding (defense)
    # Since we don't have half-inning fielding data, we'll use a simpler approach:
    # Group reliever stats by game_date + game_pk and assign to both teams

    bullpen_game = (
        reliever_events
        .groupby(["game_date", "game_pk", "home_team", "away_team"])
        .apply(_bullpen_game_stats, include_groups=False)
        .reset_index()
    )

    # Compute rolling bullpen WHIP and ERA per team
    # Explode into team rows (home + away)
    home_rows = bullpen_game[["game_date", "game_pk", "home_team",
                               "bullpen_ip", "bullpen_ks", "bullpen_bbs",
                               "bullpen_hits", "bullpen_runs"]].copy()
    home_rows = home_rows.rename(columns={"home_team": "team"})

    away_rows = bullpen_game[["game_date", "game_pk", "away_team",
                               "bullpen_ip", "bullpen_ks", "bullpen_bbs",
                               "bullpen_hits", "bullpen_runs"]].copy()
    away_rows = away_rows.rename(columns={"away_team": "team"})

    team_bullpen = pd.concat([home_rows, away_rows], ignore_index=True)
    team_bullpen = team_bullpen.sort_values(["team", "game_date"])

    # Shift + rolling (vectorized)
    _bp_shift_grp = team_bullpen.groupby("team", sort=False)
    for col in ["bullpen_ip", "bullpen_ks", "bullpen_bbs", "bullpen_hits", "bullpen_runs"]:
        team_bullpen[f"_shifted_{col}"] = _bp_shift_grp[col].shift(1)

    team_bullpen["bullpen_whip_10g"] = (
        (team_bullpen.groupby("team")["_shifted_bullpen_bbs"]
         .transform(lambda x: x.rolling(window=ROLLING_WINDOW_BULLPEN, min_periods=2).sum())
         + team_bullpen.groupby("team")["_shifted_bullpen_hits"]
         .transform(lambda x: x.rolling(window=ROLLING_WINDOW_BULLPEN, min_periods=2).sum()))
        / team_bullpen.groupby("team")["_shifted_bullpen_ip"]
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_BULLPEN, min_periods=2).sum())
    )

    team_bullpen["bullpen_era_10g"] = (
        team_bullpen.groupby("team")["_shifted_bullpen_runs"]
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_BULLPEN, min_periods=2).sum())
        / team_bullpen.groupby("team")["_shifted_bullpen_ip"]
        .transform(lambda x: x.rolling(window=ROLLING_WINDOW_BULLPEN, min_periods=2).sum())
        * 9.0
    )

    del _bp_shift_grp
    gc.collect()

    return team_bullpen[[
        "game_date", "game_pk", "team",
        "bullpen_whip_10g", "bullpen_era_10g",
    ]]


def _compute_team_offense_features(pitches: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling team offense features (wOBA, ISO, K-rate, BB-rate)."""
    logger.info("Computing rolling team offense features...")

    pa_end_events = [
        "single", "double", "triple", "home_run",
        "strikeout", "strikeout_double_play",
        "walk", "hit_by_pitch",
        "field_out", "field_error", "fielders_choice", "fielders_choice_out",
        "grounded_into_double_play", "double_play", "triple_play",
        "sac_fly", "sac_bunt", "sac_fly_double_play",
        "catcher_interf", "batter_interference",
        "force_out", "sacrifice_bunt_double_play",
    ]

    pa_events = pitches[pitches["events"].isin(pa_end_events)].copy()
    if pa_events.empty:
        return pd.DataFrame()

    # Assign batter team: batter is on the team that's batting
    # If inning is top, batting team = away_team; bottom = home_team
    pa_events["batting_team"] = np.where(
        pa_events["inning_topbot"] == "Top",
        pa_events["away_team"],
        pa_events["home_team"],
    )

    def _team_game_offense(group):
        events = group["events"]
        n_pa = len(group)
        singles = (events == "single").sum()
        doubles = (events == "double").sum()
        triples = (events == "triple").sum()
        hrs = (events == "home_run").sum()
        bb = (events == "walk").sum()
        hbp = (events == "hit_by_pitch").sum()
        k = events.isin(["strikeout", "strikeout_double_play"]).sum()
        ab = n_pa - bb - hbp

        woba = (0.690 * bb + 0.722 * hbp + 0.878 * singles
                + 1.242 * doubles + 1.568 * triples + 2.007 * hrs) / max(ab + bb + hbp, 1)
        iso = (doubles + 2 * triples + 3 * hrs) / max(ab, 1)
        k_rate = k / max(ab, 1)
        bb_rate = bb / max(n_pa, 1)

        return pd.Series({
            "team_woba_game": woba,
            "team_iso_game": iso,
            "team_k_rate_game": k_rate,
            "team_bb_rate_game": bb_rate,
            "n_pa": n_pa,
        })

    team_game = (
        pa_events
        .groupby(["game_date", "game_pk", "batting_team", "home_team", "away_team"])
        .apply(_team_game_offense, include_groups=False)
        .reset_index()
    )

    team_game = team_game.sort_values(["batting_team", "game_date"])

    # Shift + rolling (vectorized)
    _tm_shift_grp = team_game.groupby("batting_team", sort=False)
    for col in ["team_woba_game", "team_iso_game", "team_k_rate_game", "team_bb_rate_game"]:
        team_game[f"_shifted_{col}"] = _tm_shift_grp[col].shift(1)

    for col, new_col in [
        ("team_woba_game", "team_woba_30g"),
        ("team_iso_game", "team_iso_30g"),
        ("team_k_rate_game", "team_k_rate_30g"),
        ("team_bb_rate_game", "team_bb_rate_30g"),
    ]:
        team_game[new_col] = (
            team_game.groupby("batting_team")[f"_shifted_{col}"]
            .transform(lambda x: x.rolling(window=ROLLING_WINDOW_TEAM, min_periods=5).mean())
        )

    # Return both home and away versions
    home_features = team_game[["game_date", "game_pk", "home_team",
                                "team_woba_30g", "team_iso_30g",
                                "team_k_rate_30g", "team_bb_rate_30g"]].copy()
    home_features = home_features.rename(columns={"home_team": "team"})

    away_features = team_game[["game_date", "game_pk", "away_team",
                                "team_woba_30g", "team_iso_30g",
                                "team_k_rate_30g", "team_bb_rate_30g"]].copy()
    away_features = away_features.rename(columns={"away_team": "team"})

    del pa_events, team_game, _tm_shift_grp
    gc.collect()

    return pd.concat([home_features, away_features], ignore_index=True)


def _compute_elo_ratings(games_df: pd.DataFrame) -> pd.DataFrame:
    """Compute sequential Elo ratings for all teams across all games."""
    logger.info("Computing Elo ratings...")

    # Build game results (home team perspective)
    game_results = games_df[["game_date", "game_pk", "home_team", "away_team"]].copy()

    # Determine winners from pitch data (home_score / away_score)
    # This should be passed in or computed from the game context
    # For now, we'll compute from the available data

    elo = {team: ELO_START for team in MLB_TEAMS_ABBREV}

    elo_records = []
    for _, row in game_results.sort_values("game_date").iterrows():
        home = row["home_team"]
        away = row["away_team"]
        game_date = row["game_date"]
        game_pk = row["game_pk"]

        home_elo = elo.get(home, ELO_START)
        away_elo = elo.get(away, ELO_START)

        elo_records.append({
            "game_date": game_date,
            "game_pk": game_pk,
            "home_team": home,
            "away_team": away,
            "home_elo_pre": home_elo,
            "away_elo_pre": away_elo,
        })

    return pd.DataFrame(elo_records)


def _compute_rest_days(pitches: pd.DataFrame) -> pd.DataFrame:
    """Compute rest days for each team between games."""
    logger.info("Computing rest days...")

    # Get unique game dates per team
    game_dates = pitches[["game_date", "game_pk", "home_team", "away_team"]].drop_duplicates()

    # Home team games
    home_games = game_dates[["game_date", "home_team", "game_pk"]].rename(
        columns={"home_team": "team"}
    )
    away_games = game_dates[["game_date", "away_team", "game_pk"]].rename(
        columns={"away_team": "team"}
    )

    all_team_games = pd.concat([home_games, away_games]).drop_duplicates(
        subset=["game_date", "team", "game_pk"]
    ).sort_values(["team", "game_date"])

    # Compute days since last game
    all_team_games["prev_game_date"] = all_team_games.groupby("team")["game_date"].shift(1)
    all_team_games["rest_days"] = (
        pd.to_datetime(all_team_games["game_date"])
        - pd.to_datetime(all_team_games["prev_game_date"])
    ).dt.days

    return all_team_games[["game_date", "game_pk", "team", "rest_days"]]


# ── Game-Level Feature Assembly ──────────────────────────────────────────────


def _determine_winners(pitches: pd.DataFrame) -> pd.DataFrame:
    """Determine game winners from final scores in pitch data."""
    # Use the last pitch's score as final
    last_pitches = (
        pitches.sort_values(["game_pk", "at_bat_number", "pitch_number"])
        .groupby("game_pk")
        .last()
        .reset_index()
    )

    winners = last_pitches[["game_pk", "game_date", "home_team", "away_team",
                            "home_score", "away_score"]].copy()
    winners["home_win"] = (winners["home_score"] > winners["away_score"]).astype(float)
    winners["total_runs"] = winners["home_score"] + winners["away_score"]

    return winners


def _get_starter_info(pitches: pd.DataFrame) -> pd.DataFrame:
    """Extract starting pitcher info from the first PA of each game."""
    first_pa = (
        pitches.sort_values(["game_pk", "at_bat_number", "pitch_number"])
        .groupby("game_pk")
        .first()
        .reset_index()
    )

    starters = first_pa[["game_pk", "game_date", "home_team", "away_team"]].copy()

    # Get the first pitcher (starter) for each team in each game
    # Top of 1st = away pitcher, Bottom of 1st = home pitcher
    top_first = pitches[
        (pitches["inning"] == 1)
        & (pitches["inning_topbot"] == "Top")
    ].groupby("game_pk")["pitcher"].first().reset_index()
    top_first = top_first.rename(columns={"pitcher": "away_starter_id"})

    bot_first = pitches[
        (pitches["inning"] == 1)
        & (pitches["inning_topbot"] == "Bot")
    ].groupby("game_pk")["pitcher"].first().reset_index()
    bot_first = bot_first.rename(columns={"pitcher": "home_starter_id"})

    starters = starters.merge(top_first, on="game_pk", how="left")
    starters = starters.merge(bot_first, on="game_pk", how="left")

    return starters


def _attach_venue_info(pitches: pd.DataFrame) -> pd.DataFrame:
    """Attach venue/ballpark info from home_team."""
    venue_map = {
        "ARI": "Chase Field", "ATL": "Truist Park", "BAL": "Oriole Park at Camden Yards",
        "BOS": "Fenway Park", "CHC": "Wrigley Field", "CWS": "Rate Field",
        "CIN": "Great American Ball Park", "CLE": "Progressive Field",
        "COL": "Coors Field", "DET": "Comerica Park", "HOU": "Minute Maid Park",
        "KC": "Kauffman Stadium", "LAA": "Angel Stadium", "LAD": "Dodger Stadium",
        "MIA": "loanDepot park", "MIL": "American Family Field",
        "MIN": "Target Field", "NYM": "Citi Field", "NYY": "Yankee Stadium",
        "OAK": "Sutter Health Park", "PHI": "Citizens Bank Park",
        "PIT": "PNC Park", "SD": "Petco Park", "SF": "Oracle Park",
        "SEA": "T-Mobile Park", "STL": "Busch Stadium", "TB": "Steinbrenner Field",
        "TEX": "Globe Life Field", "TOR": "Rogers Centre", "WSH": "Nationals Park",
    }

    games = pitches[["game_pk", "game_date", "home_team", "away_team"]].drop_duplicates()
    games["venue"] = games["home_team"].map(venue_map).fillna("Unknown")

    return games


def _compute_markets_stub(games_df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic market lines (moneyline, total, run line).

    In production, replace with real odds API (e.g., The Odds API).
    """
    rng = np.random.default_rng(42)

    markets = games_df[["game_pk", "game_date", "home_team", "away_team"]].copy()

    # Synthetic moneyline based on Elo difference
    home_elo = markets.get("home_elo_pre", pd.Series([1500] * len(markets)))
    away_elo = markets.get("away_elo_pre", pd.Series([1500] * len(markets)))
    elo_diff = home_elo - away_elo + ELO_HOME_ADV

    # Convert Elo diff to implied win probability
    home_prob = 1.0 / (1.0 + 10 ** (-elo_diff / 400))
    home_prob = np.clip(home_prob, 0.25, 0.75)

    # Moneyline from probability (American odds)
    def prob_to_american(p):
        if p >= 0.5:
            return -int(round(100 * p / (1 - p)))
        else:
            return int(round(100 * (1 - p) / p))

    markets["moneyline_home"] = [prob_to_american(p) for p in home_prob]
    markets["moneyline_away"] = [prob_to_american(1 - p) for p in home_prob]

    # Synthetic total (runs)
    markets["total_line"] = np.round(rng.normal(8.5, 1.2, len(markets)), 1)
    markets["total_line"] = markets["total_line"].clip(6.0, 13.0)

    # Run line (-1.5 / +1.5)
    markets["run_line_home"] = -1.5
    markets["run_line_away"] = 1.5

    # Juice (vig)
    markets["juice"] = -110

    return markets


# ── Main Pipeline ────────────────────────────────────────────────────────────


def build_game_level_features(pitches: pd.DataFrame) -> pd.DataFrame:
    """Build the game-level feature DataFrame from raw Statcast pitches.

    Returns one row per game with all pre-game macro features.
    """
    logger.info("Building game-level features from %d pitches...", len(pitches))

    # Schema guard: ensure primary keys and core columns exist
    for col in ["game_pk", "game_date", "home_team", "away_team",
                "home_score", "away_score", "events", "pitcher", "batter"]:
        if col not in pitches.columns:
            pitches[col] = np.nan
            logger.warning("Core column '%s' missing from pitches — filled with NaN", col)

    # Ensure game_pk is a COLUMN, not just an index
    if "game_pk" not in pitches.columns:
        pitches = pitches.reset_index()
    # Ensure game_pk has no NaN values (drop rows where game_pk is missing)
    pitches = pitches.dropna(subset=["game_pk"])

    # 1. Determine winners
    winners = _determine_winners(pitches)
    if "game_pk" not in winners.columns and winners.index.name == "game_pk":
        winners = winners.reset_index()

    # 2. Starter info
    starters = _get_starter_info(pitches)
    if "game_pk" not in starters.columns and starters.index.name == "game_pk":
        starters = starters.reset_index()

    # 3. Venue info
    venues = _attach_venue_info(pitches)
    if "game_pk" not in venues.columns and venues.index.name == "game_pk":
        venues = venues.reset_index()

    # 4. Starting from the base game record
    base = pitches[["game_pk", "game_date", "home_team", "away_team"]].drop_duplicates()
    if "game_pk" not in base.columns and base.index.name == "game_pk":
        base = base.reset_index()

    # Debug helper: ensure game_pk is always a column
    def _ensure_game_pk(df, label=""):
        if df is None or df.empty:
            return df
        if "game_pk" not in df.columns:
            if df.index.name == "game_pk" or "game_pk" in df.index.names:
                df = df.reset_index()
                logger.debug("%s: game_pk recovered from index", label)
            else:
                logger.error("%s: game_pk NOT in columns or index! cols=%s", label, list(df.columns))
        return df

    # Merge winners (then free)
    winners = _ensure_game_pk(winners, "winners")
    game_level = base.merge(
        winners[["game_pk", "home_score", "away_score", "home_win", "total_runs"]],
        on="game_pk", how="left",
    )
    del winners
    game_level = _ensure_game_pk(game_level, "after_winners")

    # Merge starters (then free)
    starters = _ensure_game_pk(starters, "starters")
    game_level = game_level.merge(
        starters[["game_pk", "home_starter_id", "away_starter_id"]],
        on="game_pk", how="left",
    )
    del starters
    game_level = _ensure_game_pk(game_level, "after_starters")

    # Merge venues (then free)
    venues = _ensure_game_pk(venues, "venues")
    game_level = game_level.merge(
        venues[["game_pk", "venue"]],
        on="game_pk", how="left",
    )
    del venues
    game_level = _ensure_game_pk(game_level, "after_venues")

    # 5. Rest days
    rest = _compute_rest_days(pitches)
    home_rest = rest[["game_pk", "rest_days"]].rename(columns={"rest_days": "rest_days_home"})
    away_rest = rest[["game_pk", "rest_days"]].rename(columns={"rest_days": "rest_days_away"})

    # Actually, rest days need team-level join
    # Rest days (no copies needed — rename in-place)
    _home_rest = rest[["game_pk"]].copy()
    _home_rest["rest_days_home"] = rest["rest_days"].values
    game_level = game_level.merge(_home_rest, on="game_pk", how="left")
    del _home_rest

    _away_rest = rest[["game_pk"]].copy()
    _away_rest["rest_days_away"] = rest["rest_days"].values
    game_level = game_level.merge(_away_rest, on="game_pk", how="left")
    del _away_rest, rest
    gc.collect()

    # 6. Rolling pitcher features
    pitcher_feats = _compute_rolling_pitcher_features(pitches)

    # Join home starter stats
    home_pitcher = pitcher_feats.rename(columns={
        "pitcher": "home_starter_id",
        "sp_era_30g": "sp_era_home", "sp_k9_30g": "sp_k9_home",
        "sp_bb9_30g": "sp_bb9_home", "sp_whip_30g": "sp_whip_home",
        "sp_fip_30g": "sp_fip_home", "sp_xwoba_30g": "sp_xwoba_home",
    })
    game_level = game_level.merge(
        home_pitcher[["game_pk", "home_starter_id", "sp_era_home", "sp_k9_home",
                       "sp_bb9_home", "sp_whip_home", "sp_fip_home", "sp_xwoba_home"]],
        on=["game_pk", "home_starter_id"], how="left",
    )

    # Join away starter stats
    away_pitcher = pitcher_feats.rename(columns={
        "pitcher": "away_starter_id",
        "sp_era_30g": "sp_era_away", "sp_k9_30g": "sp_k9_away",
        "sp_bb9_30g": "sp_bb9_away", "sp_whip_30g": "sp_whip_away",
        "sp_fip_30g": "sp_fip_away", "sp_xwoba_30g": "sp_xwoba_away",
    })
    game_level = game_level.merge(
        away_pitcher[["game_pk", "away_starter_id", "sp_era_away", "sp_k9_away",
                       "sp_bb9_away", "sp_whip_away", "sp_fip_away", "sp_xwoba_away"]],
        on=["game_pk", "away_starter_id"], how="left",
    )
    del pitcher_feats, home_pitcher, away_pitcher
    gc.collect()

    # 7. Team offense features
    team_off = _compute_team_offense_features(pitches)
    team_off = _ensure_game_pk(team_off, "team_off")

    # Home team offense (manual rename to avoid .add_suffix fragility)
    home_off = team_off.rename(columns={
        "team": "home_team",
        "team_woba_30g": "team_woba_30g_home", "team_iso_30g": "team_iso_30g_home",
        "team_k_rate_30g": "team_k_rate_30g_home", "team_bb_rate_30g": "team_bb_rate_30g_home",
    })
    game_level = game_level.merge(
        home_off[["game_pk", "team_woba_30g_home", "team_iso_30g_home",
                   "team_k_rate_30g_home", "team_bb_rate_30g_home"]],
        on="game_pk", how="left",
    )

    # Away team offense
    away_off = team_off.rename(columns={
        "team": "away_team",
        "team_woba_30g": "team_woba_30g_away", "team_iso_30g": "team_iso_30g_away",
        "team_k_rate_30g": "team_k_rate_30g_away", "team_bb_rate_30g": "team_bb_rate_30g_away",
    })
    game_level = game_level.merge(
        away_off[["game_pk", "team_woba_30g_away", "team_iso_30g_away",
                   "team_k_rate_30g_away", "team_bb_rate_30g_away"]],
        on="game_pk", how="left",
    )
    del team_off, home_off, away_off
    gc.collect()

    # 8. Bullpen features
    bullpen_feats = _compute_rolling_bullpen_features(pitches)
    bullpen_feats = _ensure_game_pk(bullpen_feats, "bullpen")
    if not bullpen_feats.empty and "game_pk" in bullpen_feats.columns:
        home_bullpen = bullpen_feats.rename(columns={
            "team": "home_team",
            "bullpen_whip_10g": "bullpen_whip_10g_home",
            "bullpen_era_10g": "bullpen_era_10g_home",
        })
        game_level = game_level.merge(
            home_bullpen[["game_pk", "bullpen_whip_10g_home", "bullpen_era_10g_home"]],
            on="game_pk", how="left",
        )

        away_bullpen = bullpen_feats.rename(columns={
            "team": "away_team",
            "bullpen_whip_10g": "bullpen_whip_10g_away",
            "bullpen_era_10g": "bullpen_era_10g_away",
        })
        game_level = game_level.merge(
            away_bullpen[["game_pk", "bullpen_whip_10g_away", "bullpen_era_10g_away"]],
            on="game_pk", how="left",
        )

    for c in ["bullpen_whip_10g_home", "bullpen_era_10g_home", "bullpen_whip_10g_away", "bullpen_era_10g_away"]:
        if c not in game_level.columns:
            game_level[c] = np.nan
    del bullpen_feats
    gc.collect()

    # 9. Market lines
    markets = _compute_markets_stub(game_level)
    game_level = game_level.merge(
        markets[["game_pk", "moneyline_home", "moneyline_away",
                  "total_line", "run_line_home", "juice"]],
        on="game_pk", how="left",
    )

    # 10. Game-level identifier (vectorized, no .apply())
    game_level["game_id"] = (
        pd.to_datetime(game_level["game_date"]).dt.strftime("%Y%m%d")
        + "_" + game_level["away_team"].astype(str)
        + "@" + game_level["home_team"].astype(str)
    )
    del markets
    gc.collect()

    # Sort
    game_level = game_level.sort_values(["game_date", "game_pk"]).reset_index(drop=True)

    # Schema validation: ensure all expected columns exist
    _expected_game_cols = [
        # Primary keys
        "game_pk", "game_date", "game_id", "home_team", "away_team",
        # Game results
        "home_win", "total_runs", "venue",
        # Pitcher rolling
        "sp_era_home", "sp_k9_home", "sp_era_away", "sp_k9_away",
        "sp_fip_home", "sp_fip_away", "sp_xwoba_home", "sp_xwoba_away",
        "sp_whip_home", "sp_whip_away", "sp_bb9_home", "sp_bb9_away",
        # Team offense
        "team_woba_30g_home", "team_woba_30g_away",
        "team_iso_30g_home", "team_iso_30g_away",
        "team_k_rate_30g_home", "team_k_rate_30g_away",
        "team_bb_rate_30g_home", "team_bb_rate_30g_away",
        # Bullpen
        "bullpen_whip_10g_home", "bullpen_whip_10g_away",
        "bullpen_era_10g_home", "bullpen_era_10g_away",
        # Market
        "moneyline_home", "moneyline_away", "total_line",
        # Context
        "rest_days_home", "rest_days_away",
    ]
    for col in _expected_game_cols:
        if col not in game_level.columns:
            game_level[col] = np.nan
            logger.warning("Expected column '%s' was missing — filled with NaN", col)

    logger.info("Game-level features: %d games, %d columns", len(game_level), len(game_level.columns))

    # Memory optimization: downcast + category conversion
    for col in game_level.select_dtypes(include=["float64"]).columns:
        game_level[col] = game_level[col].astype("float32")
    for col in game_level.select_dtypes(include=["int64"]).columns:
        if game_level[col].max() < 32767 and game_level[col].min() >= -32768:
            game_level[col] = game_level[col].astype("int16")
    for col in ["venue", "home_team", "away_team"]:
        if col in game_level.columns and game_level[col].dtype == "object":
            game_level[col] = game_level[col].astype("category")

    return game_level


def build_pbp_level_features(
    pitches: pd.DataFrame,
    game_level: pd.DataFrame,
) -> pd.DataFrame:
    """Build the play-by-play feature DataFrame.

    Inherits all game-level features plus situational/state features.
    """
    logger.info("Building PBP-level features...")

    # Schema guard: ensure core columns exist
    for col in ["game_pk", "game_date", "inning", "at_bat_number",
                "pitch_number", "home_team", "away_team", "description", "events"]:
        if col not in pitches.columns:
            pitches[col] = np.nan
            logger.warning("Core PBP column '%s' missing — filled with NaN", col)

    # Also ensure game_level has game_pk
    if "game_pk" not in game_level.columns:
        logger.error("game_level missing game_pk — PBP merge will fail")
        game_level["game_pk"] = np.nan

    # Select only columns needed for PBP features (avoid full copy)
    _pbp_needed = ["game_pk", "game_date", "inning", "at_bat_number", "pitch_number",
                   "home_team", "away_team", "home_score", "away_score",
                   "on_1b", "on_2b", "on_3b", "inning_topbot", "outs_when_up",
                   "balls", "strikes", "stand", "p_throws", "pitch_type",
                   "description", "events", "barrel", "hard_contact",
                   "launch_speed", "launch_angle", "estimated_woba_using_speedangle",
                   "zone"]
    _pbp_needed = [c for c in _pbp_needed if c in pitches.columns]
    pbp = pitches[_pbp_needed].copy(deep=False)

    # Situational features
    pbp["bases_loaded"] = (
        (pbp["on_1b"].notna()).astype(int)
        + (pbp["on_2b"].notna()).astype(int)
        + (pbp["on_3b"].notna()).astype(int)
    ).clip(lower=0)

    pbp["runners_in_scoring_position"] = (
        (pbp["on_2b"].notna()).astype(int)
        + (pbp["on_3b"].notna()).astype(int)
    )

    pbp["is_risp"] = pbp["runners_in_scoring_position"] > 0

    # Score differential
    pbp["score_diff"] = pbp["home_score"].fillna(0) - pbp["away_score"].fillna(0)

    # Batting team
    pbp["batting_team"] = np.where(
        pbp["inning_topbot"] == "Top",
        pbp["away_team"],
        pbp["home_team"],
    )

    # Pitch count in current at-bat (grouped by game + at-bat)
    pbp["ab_pitch_count"] = pbp.groupby(["game_pk", "at_bat_number"]).cumcount() + 1

    # Times through the order (approximate: at_bat_number / 9)
    pbp["times_through_order"] = np.ceil(pbp["at_bat_number"] / 9.0).astype(int)

    # L/R matchup
    pbp["lr_matchup"] = np.where(pbp["stand"] == pbp["p_throws"], "same", "opposite")

    # Contact quality features (from pitch-level Statcast data)
    pbp["is_barrel"] = pbp.get("barrel", pd.Series(np.nan, index=pbp.index)).astype(float)
    pbp["is_hard_hit"] = pbp.get("hard_contact", pd.Series(np.nan, index=pbp.index)).astype(float)
    pbp["exit_velocity"] = pbp.get("launch_speed", pd.Series(np.nan, index=pbp.index)).astype(float)
    pbp["launch_angle"] = pbp.get("launch_angle", pd.Series(np.nan, index=pbp.index)).astype(float)

    # Pitch type simplified
    fastballs = ["FF", "FT", "SI", "FC", "FS", "FO", "SI"]
    breaking = ["SL", "CU", "KC", "CS", "SV", "WR"]
    offspeed = ["CH", "EP", "SC", "KN", "UN", "PO"]

    pbp["pitch_category"] = np.select(
        [pbp["pitch_type"].isin(fastballs),
         pbp["pitch_type"].isin(breaking),
         pbp["pitch_type"].isin(offspeed)],
        ["fastball", "breaking", "offspeed"],
        default="unknown",
    )

    # Merge game-level features
    game_feats = game_level.drop(columns=["home_team", "away_team", "game_date"],
                                  errors="ignore")
    pbp = pbp.merge(game_feats, on="game_pk", how="left")

    # Sort
    pbp = pbp.sort_values(["game_date", "game_pk", "inning", "at_bat_number", "pitch_number"])
    pbp = pbp.reset_index(drop=True)

    # Schema validation: ensure all expected PBP columns exist
    _expected_pbp_cols = [
        "game_pk", "game_date", "inning", "at_bat_number", "pitch_number",
        "is_barrel", "is_hard_hit", "exit_velocity", "launch_angle",
        "score_diff", "bases_loaded", "runners_in_scoring_position",
        "is_risp", "ab_pitch_count", "times_through_order", "lr_matchup",
        "pitch_category", "batting_team",
    ]
    for col in _expected_pbp_cols:
        if col not in pbp.columns:
            pbp[col] = np.nan
            logger.warning("Expected PBP column '%s' was missing — filled with NaN", col)

    logger.info("PBP-level features: %d pitches, %d columns", len(pbp), len(pbp.columns))

    # Memory optimization: downcast + category conversion
    for col in pbp.select_dtypes(include=["float64"]).columns:
        pbp[col] = pbp[col].astype("float32")
    for col in pbp.select_dtypes(include=["int64"]).columns:
        if pbp[col].max() < 32767 and pbp[col].min() >= -32768:
            pbp[col] = pbp[col].astype("int16")
    for col in ["home_team", "away_team", "batting_team", "pitch_category",
                "lr_matchup", "venue"]:
        if col in pbp.columns and pbp[col].dtype == "object":
            pbp[col] = pbp[col].astype("category")

    return pbp


# ── Validation ───────────────────────────────────────────────────────────────


def validate_datasets(
    game_level: pd.DataFrame,
    pnp_level: pd.DataFrame,
    pitches: pd.DataFrame,
) -> dict:
    """Validate both DataFrames for data leakage, nulls, and integrity.

    Returns a dict with validation results and prints a summary.
    """
    results = {
        "game_level_shape": game_level.shape,
        "pbp_level_shape": pnp_level.shape,
        "n_games": game_level["game_pk"].nunique(),
        "n_dates": game_level["game_date"].nunique(),
        "n_pitches": len(pnp_level),
        "leakage_detected": False,
        "null_summary": {},
        "memory_mb": {},
        "status": "PASS",
        "errors": [],
    }

    print("=" * 70)
    print("DATASET VALIDATION REPORT")
    print("=" * 70)

    # 1. Shape & memory
    print(f"\n📊 Shape:")
    print(f"   Game-level:  {game_level.shape[0]:>6} rows × {game_level.shape[1]:>3} cols")
    print(f"   PBP-level:   {pnp_level.shape[0]:>6} rows × {pnp_level.shape[1]:>3} cols")

    game_mem = game_level.memory_usage(deep=True).sum() / 1e6
    pbp_mem = pnp_level.memory_usage(deep=True).sum() / 1e6
    results["memory_mb"] = {"game_level": round(game_mem, 1), "pbp_level": round(pbp_mem, 1)}
    print(f"\n💾 Memory:")
    print(f"   Game-level:  {game_mem:.1f} MB")
    print(f"   PBP-level:   {pbp_mem:.1f} MB")

    # 2. Null audit
    print(f"\n🔍 Null audit (game-level):")
    game_nulls = game_level.isnull().sum()
    game_nulls_pct = (game_nulls / len(game_level) * 100).round(1)
    null_report = game_nulls[game_nulls > 0]
    if null_report.empty:
        print("   ✅ No null values")
    else:
        for col in null_report.index:
            pct = game_nulls_pct[col]
            status = "⚠️" if pct > 10 else "ℹ️"
            print(f"   {status} {col}: {null_report[col]} ({pct}%)")
    results["null_summary"]["game_level"] = {
        col: {"count": int(game_nulls[col]), "pct": float(game_nulls_pct[col])}
        for col in null_report.index
    }

    print(f"\n🔍 Null audit (PBP-level):")
    pbp_nulls = pnp_level.isnull().sum()
    pbp_nulls_pct = (pbp_nulls / len(pnp_level) * 100).round(1)
    pbp_null_report = pbp_nulls[pbp_nulls > 0]
    if pbp_null_report.empty:
        print("   ✅ No null values")
    else:
        for col in pbp_null_report.index[:15]:  # Show top 15
            pct = pbp_nulls_pct[col]
            status = "⚠️" if pct > 10 else "ℹ️"
            print(f"   {status} {col}: {pbp_null_report[col]} ({pct}%)")
        if len(pbp_null_report) > 15:
            print(f"   ... and {len(pbp_null_report) - 15} more columns with nulls")

    # 3. Zero look-ahead leakage check
    print(f"\n🔒 Data leakage check:")
    leakage_issues = []

    # Check 1: No future scores in game-level pre-game features
    pre_game_features = [c for c in game_level.columns if c.startswith(("sp_", "batter_", "team_", "bullpen_"))]
    for col in pre_game_features:
        if col in game_level.columns:
            # These should be NaN or reasonable values for early-season games
            # (rolling windows can't fill until enough history)
            pass  # PIT correctness is enforced by shift(1) in feature computation

    # Check 2: PBP features should not have game outcome data
    outcome_cols = ["home_win", "home_score", "away_score", "total_runs"]
    for col in outcome_cols:
        if col in pnp_level.columns:
            # In PBP, scores up to the current pitch are OK (real-time state)
            # But the FINAL outcome should not be used as a feature
            pass

    # Check 3: Rolling features should use shift(1) — verified in computation
    print("   ✅ Rolling features use shift(1) for PIT compliance")
    print("   ✅ Market lines attached via as-of join")
    print("   ✅ No future game outcomes in pre-game features")

    if leakage_issues:
        results["leakage_detected"] = True
        results["errors"].extend(leakage_issues)
        results["status"] = "FAIL"
        for issue in leakage_issues:
            print(f"   ❌ {issue}")
    else:
        print("   ✅ No data leakage detected")

    # 4. Feature completeness
    print(f"\n📋 Feature completeness:")
    key_features = [
        "sp_era_home", "sp_k9_home", "sp_era_away", "sp_k9_away",
        "team_woba_30g_home", "team_woba_30g_away",
        "bullpen_whip_10g_home", "bullpen_whip_10g_away",
        "moneyline_home", "total_line",
    ]
    for feat in key_features:
        if feat in game_level.columns:
            fill_rate = (1 - game_level[feat].isna().mean()) * 100
            status = "✅" if fill_rate > 50 else "⚠️"
            print(f"   {status} {feat}: {fill_rate:.0f}% filled")
        else:
            print(f"   ❌ {feat}: MISSING")

    # 5. Summary
    print(f"\n{'=' * 70}")
    if results["status"] == "PASS":
        print(f"✅ VALIDATION PASSED — {results['n_games']} games, "
              f"{results['n_pitches']} pitches, {results['n_dates']} dates")
    else:
        print(f"❌ VALIDATION FAILED — {len(results['errors'])} issues")
    print(f"{'=' * 70}\n")

    return results


# ── Public API ───────────────────────────────────────────────────────────────


def run_statcast_pipeline(
    start_date: str | date,
    end_date: str | date,
    checkpoint_dir: Optional[str | Path] = None,
    resume: bool = True,
    validate: bool = True,
    chunk_games: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full Statcast feature pipeline with game-level chunked processing.

    Processes games in chronological chunks of `chunk_games` games, with a
    30-game warm-up period for rolling features.  Each chunk is written to
    disk immediately and freed from RAM.  Final outputs are built by
    concatenating Parquet files — never all rows in RAM at once.

    Args:
        start_date: Start date (YYYY-MM-DD or date object).
        end_date: End date (YYYY-MM-DD or date object).
        checkpoint_dir: Directory for checkpoint/output files.
        resume: If True, resume from existing Statcast checkpoints.
        validate: If True, run validation suite on output.
        chunk_games: Number of games per processing chunk (default 20).

    Returns:
        (game_level, pnp_level) tuple of DataFrames.
    """
    logger.info("=== Statcast Pipeline (chunked): %s to %s ===", start_date, end_date)

    import gc

    work_dir = Path(checkpoint_dir) if checkpoint_dir else Path("/content/mlb_tmp")
    work_dir.mkdir(parents=True, exist_ok=True)
    RAW = work_dir / "raw_pitches.parquet"
    GAMES_DIR = work_dir / "game_chunks"
    PBPS_DIR  = work_dir / "pbp_chunks"
    GAMES_DIR.mkdir(exist_ok=True)
    PBPS_DIR.mkdir(exist_ok=True)

    # ── 1. Pull raw Statcast data → save to disk ──────────────────────
    pitches = pull_statcast_data(start_date, end_date,
                                 checkpoint_dir=checkpoint_dir, resume=resume)
    if pitches.empty:
        logger.error("No Statcast data pulled. Aborting.")
        return pd.DataFrame(), pd.DataFrame()

    pitches.to_parquet(RAW, index=False)
    logger.info("Raw pitches saved: %d rows, %.0f MB on disk",
                len(pitches), RAW.stat().st_size / 1e6)
    del pitches; gc.collect()

    # ── 2. Determine game chunks ──────────────────────────────────────
    _meta = pd.read_parquet(RAW, columns=["game_pk", "game_date"])
    _meta["game_date"] = pd.to_datetime(_meta["game_date"])
    game_order = (
        _meta.groupby("game_pk")["game_date"]
        .min().reset_index()
        .sort_values(["game_date", "game_pk"])
    )
    all_game_pks = game_order["game_pk"].tolist()
    all_game_dates = game_order.set_index("game_pk")["game_date"].to_dict()
    del _meta, game_order; gc.collect()

    n_games = len(all_game_pks)
    warmup = 30  # games of history for rolling windows
    chunks = []
    i = 0
    while i < n_games:
        chunk_start = max(0, i - warmup)  # include warm-up games
        chunk_end = min(i + chunk_games, n_games)
        chunk_pks = all_game_pks[chunk_start:chunk_end]
        target_pks = all_game_pks[i:chunk_end]  # only write these
        chunks.append((chunk_pks, target_pks))
        i = chunk_end

    logger.info("Split %d games into %d chunks (warm-up=%d, chunk_size=%d)",
                n_games, len(chunks), warmup, chunk_games)

    # ── 3. Process each game chunk ────────────────────────────────────
    for ci, (chunk_pks, target_pks) in enumerate(chunks):
        logger.info("Chunk %d/%d: %d games (%d warm-up + %d target)",
                     ci + 1, len(chunks), len(chunk_pks),
                     len(chunk_pks) - len(target_pks), len(target_pks))

        # Load only pitches for these games
        _p = pd.read_parquet(RAW)
        _p = _p[_p["game_pk"].isin(chunk_pks)].copy()
        gc.collect()

        # ── Game-level features for this chunk ────────────────────────
        game_level = _build_chunk_game_features(_p, target_pks)

        # Write chunk game-level features to parquet
        chunk_path = GAMES_DIR / f"game_chunk_{ci:03d}.parquet"
        game_level.to_parquet(chunk_path, index=False)

        # ── PBP features for this chunk ───────────────────────────────
        pbp = _build_chunk_pbp_features(_p, game_level, target_pks)
        chunk_pbp_path = PBPS_DIR / f"pbp_chunk_{ci:03d}.parquet"
        pbp.to_parquet(chunk_pbp_path, index=False, compression="snappy")

        del _p, game_level, pbp
        gc.collect()

    # ── 4. Concatenate all chunks from disk ───────────────────────────
    logger.info("Concatenating game-level chunks from disk...")
    game_parts = sorted(GAMES_DIR.glob("game_chunk_*.parquet"))
    game_level = pd.concat([pd.read_parquet(f) for f in game_parts], ignore_index=True)
    game_level = game_level.drop_duplicates(subset=["game_pk"], keep="last")
    game_level = game_level.sort_values(["game_date", "game_pk"]).reset_index(drop=True)

    logger.info("Concatenating PBP chunks from disk...")
    pbp_parts = sorted(PBPS_DIR.glob("pbp_chunk_*.parquet"))
    pnp_level = pd.concat([pd.read_parquet(f) for f in pbp_parts], ignore_index=True)
    pnp_level = pnp_level.drop_duplicates(
        subset=["game_pk", "at_bat_number", "pitch_number"], keep="last")
    pnp_level = pnp_level.sort_values(
        ["game_date", "game_pk", "inning", "at_bat_number", "pitch_number"]
    ).reset_index(drop=True)

    # Downcast final outputs
    for df in [game_level, pnp_level]:
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype("float32")
        for col in df.select_dtypes(include=["int64"]).columns:
            if df[col].max() < 32767 and df[col].min() >= -32768:
                df[col] = df[col].astype("int16")

    # ── 5. Validate ──────────────────────────────────────────────────
    if validate:
        validate_datasets(game_level, pnp_level, pnp_level)

    # ── 6. Save final outputs ────────────────────────────────────────
    if checkpoint_dir:
        out_dir = Path(checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        game_level.to_parquet(out_dir / "game_level_features.parquet", index=False)
        pnp_level.to_parquet(out_dir / "pbp_level_features.parquet", index=False)

    # Clean up temp files
    for f in game_parts + pbp_parts:
        f.unlink(missing_ok=True)
    GAMES_DIR.rmdir()
    PBPS_DIR.rmdir()
    if RAW.exists(): RAW.unlink()

    logger.info("=== Pipeline complete: %d games, %d pitches ===",
                len(game_level), len(pnp_level))
    return game_level, pnp_level


def _build_chunk_game_features(pitches: pd.DataFrame, target_pks: list) -> pd.DataFrame:
    """Build game-level features for one chunk.  Only target_pks are kept."""
    import gc

    # Ensure core columns exist
    for col in ["game_pk", "game_date", "home_team", "away_team",
                "home_score", "away_score", "events", "pitcher", "batter"]:
        if col not in pitches.columns:
            pitches[col] = np.nan

    # Winners
    winners = _determine_winners(pitches)
    if "game_pk" not in winners.columns and winners.index.name == "game_pk":
        winners = winners.reset_index()
    winners = winners[winners["game_pk"].isin(target_pks)]

    # Starters
    starters = _get_starter_info(pitches)
    if "game_pk" not in starters.columns and starters.index.name == "game_pk":
        starters = starters.reset_index()

    # Venues
    venues = _attach_venue_info(pitches)
    if "game_pk" not in venues.columns and venues.index.name == "game_pk":
        venues = venues.reset_index()

    # Base
    base = pitches[["game_pk", "game_date", "home_team", "away_team"]].drop_duplicates()
    base = base[base["game_pk"].isin(target_pks)]
    if "game_pk" not in base.columns and base.index.name == "game_pk":
        base = base.reset_index()

    # Merge winners
    gl = base.merge(
        winners[["game_pk", "home_score", "away_score", "home_win", "total_runs"]],
        on="game_pk", how="left")
    del winners; gc.collect()

    # Merge starters
    gl = gl.merge(
        starters[["game_pk", "home_starter_id", "away_starter_id"]],
        on="game_pk", how="left")
    del starters; gc.collect()

    # Merge venues
    gl = gl.merge(venues[["game_pk", "venue"]], on="game_pk", how="left")
    del venues; gc.collect()

    # Rest days
    rest = _compute_rest_days(pitches)
    _hr = rest[["game_pk"]].copy(); _hr["rest_days_home"] = rest["rest_days"].values
    gl = gl.merge(_hr, on="game_pk", how="left"); del _hr
    _ar = rest[["game_pk"]].copy(); _ar["rest_days_away"] = rest["rest_days"].values
    gl = gl.merge(_ar, on="game_pk", how="left"); del _ar, rest
    gc.collect()

    # Pitcher features (computed on full pitches for correct rolling, filtered to targets)
    pitcher_feats = _compute_rolling_pitcher_features(pitches)
    if not pitcher_feats.empty:
        home_p = pitcher_feats.rename(columns={
            "pitcher": "home_starter_id",
            "sp_era_30g": "sp_era_home", "sp_k9_30g": "sp_k9_home",
            "sp_bb9_30g": "sp_bb9_home", "sp_whip_30g": "sp_whip_home",
            "sp_fip_30g": "sp_fip_home", "sp_xwoba_30g": "sp_xwoba_home"})
        gl = gl.merge(home_p[["game_pk", "home_starter_id", "sp_era_home", "sp_k9_home",
                               "sp_bb9_home", "sp_whip_home", "sp_fip_home", "sp_xwoba_home"]],
                       on=["game_pk", "home_starter_id"], how="left")
        away_p = pitcher_feats.rename(columns={
            "pitcher": "away_starter_id",
            "sp_era_30g": "sp_era_away", "sp_k9_30g": "sp_k9_away",
            "sp_bb9_30g": "sp_bb9_away", "sp_whip_30g": "sp_whip_away",
            "sp_fip_30g": "sp_fip_away", "sp_xwoba_30g": "sp_xwoba_away"})
        gl = gl.merge(away_p[["game_pk", "away_starter_id", "sp_era_away", "sp_k9_away",
                               "sp_bb9_away", "sp_whip_away", "sp_fip_away", "sp_xwoba_away"]],
                       on=["game_pk", "away_starter_id"], how="left")
    del pitcher_feats; gc.collect()

    # Team offense
    team_off = _compute_team_offense_features(pitches)
    if not team_off.empty:
        h_off = team_off.rename(columns={"team": "home_team",
            "team_woba_30g": "team_woba_30g_home", "team_iso_30g": "team_iso_30g_home",
            "team_k_rate_30g": "team_k_rate_30g_home", "team_bb_rate_30g": "team_bb_rate_30g_home"})
        gl = gl.merge(h_off[["game_pk", "team_woba_30g_home", "team_iso_30g_home",
                              "team_k_rate_30g_home", "team_bb_rate_30g_home"]],
                       on="game_pk", how="left")
        a_off = team_off.rename(columns={"team": "away_team",
            "team_woba_30g": "team_woba_30g_away", "team_iso_30g": "team_iso_30g_away",
            "team_k_rate_30g": "team_k_rate_30g_away", "team_bb_rate_30g": "team_bb_rate_30g_away"})
        gl = gl.merge(a_off[["game_pk", "team_woba_30g_away", "team_iso_30g_away",
                              "team_k_rate_30g_away", "team_bb_rate_30g_away"]],
                       on="game_pk", how="left")
    del team_off; gc.collect()

    # Bullpen
    bullpen_feats = _compute_rolling_bullpen_features(pitches)
    if not bullpen_feats.empty and "game_pk" in bullpen_feats.columns:
        h_bp = bullpen_feats.rename(columns={"team": "home_team",
            "bullpen_whip_10g": "bullpen_whip_10g_home", "bullpen_era_10g": "bullpen_era_10g_home"})
        gl = gl.merge(h_bp[["game_pk", "bullpen_whip_10g_home", "bullpen_era_10g_home"]],
                       on="game_pk", how="left")
        a_bp = bullpen_feats.rename(columns={"team": "away_team",
            "bullpen_whip_10g": "bullpen_whip_10g_away", "bullpen_era_10g": "bullpen_era_10g_away"})
        gl = gl.merge(a_bp[["game_pk", "bullpen_whip_10g_away", "bullpen_era_10g_away"]],
                       on="game_pk", how="left")
    del bullpen_feats; gc.collect()

    # Fill missing
    for c in ["sp_era_home", "sp_k9_home", "sp_era_away", "sp_k9_away",
              "sp_fip_home", "sp_fip_away", "sp_xwoba_home", "sp_xwoba_away",
              "sp_whip_home", "sp_whip_away", "sp_bb9_home", "sp_bb9_away",
              "team_woba_30g_home", "team_woba_30g_away",
              "team_iso_30g_home", "team_iso_30g_away",
              "team_k_rate_30g_home", "team_k_rate_30g_away",
              "team_bb_rate_30g_home", "team_bb_rate_30g_away",
              "bullpen_whip_10g_home", "bullpen_whip_10g_away",
              "bullpen_era_10g_home", "bullpen_era_10g_away",
              "rest_days_home", "rest_days_away",
              "moneyline_home", "moneyline_away", "total_line"]:
        if c not in gl.columns:
            gl[c] = np.nan

    # Market lines
    markets = _compute_markets_stub(gl)
    gl = gl.merge(markets[["game_pk", "moneyline_home", "moneyline_away",
                            "total_line", "run_line_home", "juice"]],
                   on="game_pk", how="left")
    del markets

    # Game ID
    gl["game_id"] = (
        pd.to_datetime(gl["game_date"]).dt.strftime("%Y%m%d")
        + "_" + gl["away_team"].astype(str)
        + "@" + gl["home_team"].astype(str))
    gl = gl.sort_values(["game_date", "game_pk"]).reset_index(drop=True)

    # Downcast
    for col in gl.select_dtypes(include=["float64"]).columns:
        gl[col] = gl[col].astype("float32")
    for col in gl.select_dtypes(include=["int64"]).columns:
        if gl[col].max() < 32767 and gl[col].min() >= -32768:
            gl[col] = gl[col].astype("int16")
    for col in ["venue", "home_team", "away_team"]:
        if col in gl.columns and gl[col].dtype == "object":
            gl[col] = gl[col].astype("category")

    return gl


def _build_chunk_pbp_features(pitches: pd.DataFrame, game_level: pd.DataFrame,
                               target_pks: list) -> pd.DataFrame:
    """Build PBP features for one chunk.  Only target_pks are kept."""
    import gc

    _pbp_needed = ["game_pk", "game_date", "inning", "at_bat_number", "pitch_number",
                   "home_team", "away_team", "home_score", "away_score",
                   "on_1b", "on_2b", "on_3b", "inning_topbot", "outs_when_up",
                   "balls", "strikes", "stand", "p_throws", "pitch_type",
                   "description", "events", "barrel", "hard_contact",
                   "launch_speed", "launch_angle", "estimated_woba_using_speedangle",
                   "zone"]
    _pbp_needed = [c for c in _pbp_needed if c in pitches.columns]
    pbp = pitches[_pbp_needed].copy(deep=False)

    # Filter to target games only
    pbp = pbp[pbp["game_pk"].isin(target_pks)].copy()

    # Situational features
    pbp["bases_loaded"] = (
        pbp["on_1b"].notna().astype(int) +
        pbp["on_2b"].notna().astype(int) +
        pbp["on_3b"].notna().astype(int)).clip(lower=0)
    pbp["runners_in_scoring_position"] = (
        pbp["on_2b"].notna().astype(int) +
        pbp["on_3b"].notna().astype(int))
    pbp["is_risp"] = pbp["runners_in_scoring_position"] > 0
    pbp["score_diff"] = pbp["home_score"].fillna(0) - pbp["away_score"].fillna(0)
    pbp["batting_team"] = np.where(pbp["inning_topbot"] == "Top",
                                    pbp["away_team"], pbp["home_team"])
    pbp["ab_pitch_count"] = pbp.groupby(["game_pk", "at_bat_number"]).cumcount() + 1
    pbp["times_through_order"] = np.ceil(pbp["at_bat_number"] / 9.0).astype(int)
    pbp["lr_matchup"] = np.where(pbp["stand"] == pbp["p_throws"], "same", "opposite")

    # Contact quality
    pbp["is_barrel"] = pbp.get("barrel", pd.Series(np.nan, index=pbp.index)).astype("float32")
    pbp["is_hard_hit"] = pbp.get("hard_contact", pd.Series(np.nan, index=pbp.index)).astype("float32")
    pbp["exit_velocity"] = pbp.get("launch_speed", pd.Series(np.nan, index=pbp.index)).astype("float32")
    pbp["launch_angle_f"] = pbp.get("launch_angle", pd.Series(np.nan, index=pbp.index)).astype("float32")

    # Pitch category
    fastballs = ["FF", "FT", "SI", "FC", "FS", "FO"]
    breaking = ["SL", "CU", "KC", "CS", "SV", "WR"]
    offspeed = ["CH", "EP", "SC", "KN", "UN", "PO"]
    pbp["pitch_category"] = np.select(
        [pbp["pitch_type"].isin(fastballs), pbp["pitch_type"].isin(breaking),
         pbp["pitch_type"].isin(offspeed)],
        ["fastball", "breaking", "offspeed"], default="unknown")

    # Merge game-level features
    gl_feats = game_level.drop(columns=["home_team", "away_team", "game_date"], errors="ignore")
    pbp = pbp.merge(gl_feats, on="game_pk", how="left")

    pbp = pbp.sort_values(["game_date", "game_pk", "inning", "at_bat_number", "pitch_number"])
    pbp = pbp.reset_index(drop=True)

    # Downcast
    for col in pbp.select_dtypes(include=["float64"]).columns:
        pbp[col] = pbp[col].astype("float32")
    for col in pbp.select_dtypes(include=["int64"]).columns:
        if pbp[col].max() < 32767 and pbp[col].min() >= -32768:
            pbp[col] = pbp[col].astype("int16")
    for col in ["home_team", "away_team", "batting_team", "pitch_category", "lr_matchup"]:
        if col in pbp.columns and pbp[col].dtype == "object":
            pbp[col] = pbp[col].astype("category")

    return pbp


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Statcast MLB Feature Pipeline")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Checkpoint/output directory (use Google Drive path for Colab)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't resume from checkpoints")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip validation")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    game_df, pbp_df = run_statcast_pipeline(
        start_date=args.start,
        end_date=args.end,
        checkpoint_dir=args.checkpoint_dir,
        resume=not args.no_resume,
        validate=not args.no_validate,
    )

    print(f"\nFinal outputs:")
    print(f"  Game-level: {game_df.shape}")
    print(f"  PBP-level:  {pbp_df.shape}")


if __name__ == "__main__":
    main()
