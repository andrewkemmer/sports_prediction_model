"""Walk-forward fold generation — expanding, calendar-day based.

Geometry (spec section 6):
  - validation windows are chronological, non-overlapping, 7 calendar days
    wide, keyed on game DATES (never NFL week IDs)
  - training = all eligible games STRICTLY BEFORE the validation window
  - training expands over time; the final partial window is retained
  - 2018 is warmup only: it appears in training sets (strictly-prior state)
    but never in a validation window
  - ordinary validation windows with fewer than MIN_VAL_FOLD_GAMES games are
    skipped; the final partial tail is retained so newest games remain visible

A fold is a (fold_id, val_start, val_end, train_idx, val_idx) tuple over the
row order of the caller's frame; callers must pass frames already in
``canonical_sort`` order (see below) so the returned labels are positional and
remain valid for every downstream consumer.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

try:
    from backend import config
except ImportError:
    import config

# Every frame that feeds fold generation MUST be put in this order first.
# Why a helper and not a bare sort_values(date_col): make_folds returns
# df.index[mask] (labels), and the OOF consumers index those labels
# positionally after their own reset_index(drop=True). A single-column
# sort_values uses an UNSTABLE quicksort, so two such sorts over the same
# data disagree on the order of same-date games -- 2206 of 2671 rows land in a
# different position. make_folds then hands out labels computed under one tie
# order and the consumer applies them under another, so the learner sees the
# right games in a different order (a reproducibility break, since the
# gradient-boosting members are row-order sensitive under a fixed seed).
# [date_col, "game_id"] is a TOTAL order (game_id is unique), and mergesort
# makes it stable, so every caller agrees exactly.
CANONICAL_TIEBREAK = "game_id"


def canonical_sort(df: pd.DataFrame, date_col: str = "gameday") -> pd.DataFrame:
    """The ONE row order fold indices are valid for.

    Stable, total ordering by (date_col, game_id) with a fresh RangeIndex, so
    fold labels are positions in a frame every consumer reproduces byte for
    byte regardless of the order the frame arrived in.
    """
    keys = [date_col, CANONICAL_TIEBREAK] if CANONICAL_TIEBREAK in df.columns \
        else [date_col]
    return df.sort_values(keys, kind="mergesort").reset_index(drop=True)


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
    min_val_games = getattr(config, "MIN_VAL_FOLD_GAMES", 15)
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
