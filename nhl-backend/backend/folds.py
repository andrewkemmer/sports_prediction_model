"""Walk-forward fold generation — expanding, observed-date based.

Structural mirror of MLB's walk-forward fold generator:

  - validation windows are chronological, non-overlapping, and seven
    OBSERVED GAME DATES wide (schedule gaps do not silently create a
    different window cadence)
  - training = all eligible games STRICTLY BEFORE the validation window
  - training expands over time; the final partial window is retained and
    explicitly marked
  - all validation rows are restricted to OOF_FIRST_SEASON and later
  - the first validation index is max(cadence, WARMUP_DAYS), matching MLB's
    min_train_days warm-up contract

A fold is a (fold_id, val_start, val_end, train_idx, val_idx) tuple over the
row order of the caller's frame; callers must pass chronologically sorted
frames.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

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
    is_partial_tail: bool = False


def make_folds(df: pd.DataFrame,
               date_col: str = "gameday",
               cadence_days: int | None = None,
               max_eval_folds: int = 0) -> list[Fold]:
    """Build MLB-shaped expanding walk-forward folds over ``df``.

    Validation windows cover only core-season games (OOF_FIRST_SEASON and
    later). Windows are defined by unique observed dates, not by arithmetic
    calendar offsets, so a schedule gap cannot change the fold cadence. The
    first validation index is ``max(cadence, WARMUP_DAYS)``. Training is every
    row strictly before that window's first observed validation date.
    """
    cadence = cadence_days or config.RETRAIN_CADENCE_DAYS
    min_val_games = getattr(config, "MIN_VAL_FOLD_GAMES", 40)
    warmup_days = int(getattr(config, "WARMUP_DAYS", 0) or 0)
    if date_col not in df.columns:
        raise KeyError(f"make_folds: missing date column {date_col!r}")

    dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    seasons = pd.to_numeric(df["season"], errors="coerce")
    core_mask = seasons >= config.OOF_FIRST_SEASON
    core_dates = dates[core_mask].dropna().drop_duplicates().sort_values()
    unique_dates = list(core_dates)
    if len(unique_dates) < cadence + 1:
        return []

    # MLB's exact warm-up/index rule: skip at least one cadence and the
    # configured minimum history index, then consume seven observed dates.
    val_start_idx = max(cadence, warmup_days)
    if val_start_idx >= len(unique_dates):
        return []

    candidates: list[Fold] = []
    fold_id = 0
    while val_start_idx < len(unique_dates):
        val_end_idx = min(val_start_idx + cadence, len(unique_dates))
        val_start = pd.Timestamp(unique_dates[val_start_idx])
        val_end = pd.Timestamp(unique_dates[val_end_idx - 1])
        is_partial_tail = val_end_idx < val_start_idx + cadence
        train_mask = dates < val_start
        val_mask = core_mask & (dates >= val_start) & (dates <= val_end)
        train_idx = df.index[train_mask]
        val_idx = df.index[val_mask]
        if len(train_idx) and len(val_idx):
            candidates.append(Fold(
                fold_id=fold_id, val_start=val_start, val_end=val_end,
                train_idx=train_idx, val_idx=val_idx,
                is_partial_tail=is_partial_tail,
            ))
            fold_id += 1
        val_start_idx = val_end_idx

    # Keep ordinary folds only when they meet the minimum validation count;
    # retain the one final partial tail exactly as MLB does.
    folds = [f for f in candidates
             if len(f.val_idx) >= min_val_games or f.is_partial_tail]
    if max_eval_folds > 0 and len(folds) > max_eval_folds:
        folds = folds[-max_eval_folds:]
    # Renumber contiguously AFTER filtering. The min-validation filter drops
    # candidate windows, which used to leave gaps in ``fold_id`` (e.g. 46
    # folds numbered up to 57). Consumers treat ``fold_id`` as an ordinal —
    # progress logging, and the strictly-prior mask
    # ``fold_ids < fold.fold_id`` in the prequential calibrator — so a gap
    # silently misreports the denominator and keeps a dead id in the prior
    # window. Renumbering keeps the prior set exactly the preceding folds.
    folds = [replace(f, fold_id=i) for i, f in enumerate(folds)]
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
            "is_partial_tail": bool(f.is_partial_tail),
            "n_train": int(len(f.train_idx)),
            "n_validation": int(len(f.val_idx)),
        })
    return pd.DataFrame(rows)
