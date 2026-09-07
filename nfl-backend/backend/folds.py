"""Walk-forward fold generation — expanding, calendar-day based.

Geometry (spec section 6):
  - validation windows are chronological, non-overlapping, 7 calendar days
    wide, keyed on game DATES (never NFL week IDs)
  - training = all eligible games STRICTLY BEFORE the validation window
  - training expands over time; the final partial window is retained
  - 2018 is warmup only: it appears in training sets (strictly-prior state)
    but never in a validation window
  - no arbitrary minimum validation-game-count filter — every legitimate
    window with eligible games is a fold

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

    ``df`` may contain warmup rows (2018) and core rows (2019+). Validation
    windows cover ONLY core-season games (OOF_FIRST_SEASON and later);
    warmup rows join training sets when they fall strictly before a window.
    Training is every eligible row STRICTLY BEFORE ``val_start``.
    """
    cadence = cadence_days or config.RETRAIN_CADENCE_DAYS
    if date_col not in df.columns:
        raise KeyError(f"make_folds: missing date column {date_col!r}")
    dates = pd.to_datetime(df[date_col], errors="coerce")
    seasons = pd.to_numeric(df["season"], errors="coerce")

    core_mask = seasons >= config.OOF_FIRST_SEASON
    core_dates = dates[core_mask].dropna()
    if core_dates.empty:
        return []

    d_min = core_dates.min().normalize()
    d_max = core_dates.max().normalize()

    folds: list[Fold] = []
    fold_id = 0
    win_start = d_min
    while win_start <= d_max:
        win_end = win_start + pd.Timedelta(days=cadence - 1)   # inclusive
        val_mask = (core_mask & (dates >= win_start)
                    & (dates <= win_end + pd.Timedelta(hours=23, minutes=59, seconds=59)))
        val_idx = df.index[val_mask]
        if len(val_idx):
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
