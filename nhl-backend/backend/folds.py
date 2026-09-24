"""Walk-forward fold generation — expanding, calendar-day based.

Structural mirror of the NFL/MLB folds.py with the NHL's MLB-style 30-DAY
warm-up (the user's approved choice over an NFL-style warm-up season):

  - validation windows are chronological, non-overlapping, 7 calendar days
    wide, keyed on game DATES (never NHL season-day IDs)
  - training = all eligible games STRICTLY BEFORE the validation window
  - training expands over time; the final partial window is retained
  - all data is season 2024+ (OOF_FIRST_SEASON); there is no warm-up season.
    The 30-day warm-up instead shifts the FIRST validation window to start
    at least WARMUP_DAYS after the first core game date, so every fold's
    training set spans at least ~30 days of settled history
  - ordinary validation windows with fewer than MIN_VAL_FOLD_GAMES games are
    skipped; the final partial tail is retained so newest games remain visible

A fold is a (fold_id, val_start, val_end, train_idx, val_idx) tuple over the
row order of the caller's frame; callers must pass chronologically sorted
frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

try:
    from backend import config
except ImportError:
    import config


@dataclass
class Fold:
    fold_id: int
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    train_idx: pd.Index
    val_idx: pd.Index


def make_folds(df: pd.DataFrame,
               date_col: str = "gameday",
               cadence_days: int | None = None) -> list[Fold]:
    """Build expanding walk-forward folds over ``df``.

    Validation windows cover ONLY core-season games (OOF_FIRST_SEASON and
    later; NHL data starts there by construction). The first window starts at
    the first core date + config.WARMUP_DAYS (the MLB-style warm-up).
    Training is every eligible row STRICTLY BEFORE ``val_start``.
    """
    cadence = cadence_days or config.RETRAIN_CADENCE_DAYS
    min_val_games = getattr(config, "MIN_VAL_FOLD_GAMES", 40)
    warmup_days = int(getattr(config, "WARMUP_DAYS", 0) or 0)
    if date_col not in df.columns:
        raise KeyError(f"make_folds: missing date column {date_col!r}")
    dates = pd.to_datetime(df[date_col], errors="coerce")
    seasons = pd.to_numeric(df["season"], errors="coerce")

    core_mask = seasons >= config.OOF_FIRST_SEASON
    core_dates = dates[core_mask].dropna()
    if core_dates.empty:
        return []

    # 30-day warm-up: the first validation window cannot start before the
    # first core date + warmup_days (MLB parity — the walk-forward never
    # scores a fold trained on fewer than ~30 days of history).
    d_min = (core_dates.min().normalize()
             + pd.Timedelta(days=warmup_days))
    d_max = core_dates.max().normalize()

    folds: list[Fold] = []
    fold_id = 0
    win_start = d_min
    while win_start <= d_max:
        win_end = win_start + pd.Timedelta(days=cadence - 1)   # inclusive
        val_mask = (core_mask & (dates >= win_start)
                    & (dates <= win_end + pd.Timedelta(hours=23, minutes=59, seconds=59)))
        val_idx = df.index[val_mask]
        is_final_window = win_end >= d_max
        if len(val_idx) and (len(val_idx) >= min_val_games or is_final_window):
            train_mask = dates < win_start
            train_idx = df.index[train_mask]
            folds.append(Fold(fold_id=fold_id, val_start=win_start,
                              val_end=win_end, train_idx=train_idx,
                              val_idx=val_idx))
            fold_id += 1
        win_start = win_end + pd.Timedelta(days=1)
    return folds


def fold_summary(folds: list[Fold]) -> dict:
    """Diagnostics for the final validation record."""
    if not folds:
        return {"n_folds": 0}
    return {
        "n_folds": len(folds),
        "first_val_start": str(folds[0].val_start.date()),
        "last_val_end": str(folds[-1].val_end.date()),
        "min_train": int(min(len(f.train_idx) for f in folds)),
        "max_train": int(max(len(f.train_idx) for f in folds)),
        "total_val_games": int(sum(len(f.val_idx) for f in folds)),
    }


def fold_table(df: pd.DataFrame, folds: list[Fold],
               date_col: str = "gameday") -> pd.DataFrame:
    """Per-fold reporting table: fold_id, train_end_date, validation window,
    n_train, n_validation. ``df`` must be the SAME frame (same row order)
    the folds were generated over."""
    rows = []
    dates = pd.to_datetime(df[date_col], errors="coerce")
    for f in folds:
        rows.append({
            "fold_id": f.fold_id,
            "train_end_date": (dates.loc[f.train_idx].max().date()
                               if len(f.train_idx) else pd.NaT),
            "validation_start": f.val_start.date(),
            "validation_end": f.val_end.date(),
            "n_train": int(len(f.train_idx)),
            "n_validation": int(len(f.val_idx)),
        })
    return pd.DataFrame(rows)
