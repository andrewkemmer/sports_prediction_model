"""Strict expanding, observed-date walk-forward folds for the NBA backend.

The production contract mirrors MLB/NHL: validation windows contain seven
observed game dates, training contains every row strictly before the window,
and the first validation window starts after the configured warm-up index.
Observed-date geometry is deliberate: a schedule gap must not silently turn a
seven-day retrain window into a different window or leak a future row.

A fold is a (fold_id, val_start, val_end, train_idx, val_idx) tuple over the
row order of the caller's frame; callers must pass frames already in
``canonical_sort`` order (see below) so the returned labels are positional and
remain valid for every downstream consumer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

try:
    from backend import config
except ImportError:  # pragma: no cover - direct script/import fallback
    import config

# Every frame that feeds fold generation MUST be put in this order first.
# Why a helper and not a bare sort_values(date_col): make_folds returns
# df.index[mask] (labels), and the OOF consumers index those labels
# positionally after their own reset_index(drop=True). A single-column
# sort_values uses an UNSTABLE quicksort, so two such sorts over the same data
# disagree on the order of same-date games -- an NBA season moves 875 of 1023
# rows that way. make_folds then hands out labels computed under one tie order
# and the consumer applies them under another, so a validation window is
# scored against the WRONG games (16 of 16 folds in the reproduction), not
# merely a differently ordered right set. [date_col, "game_id"] is a TOTAL
# order (game_id is unique), and mergesort is stable, so every consumer
# reproduces the same positions byte for byte.
CANONICAL_TIEBREAK = "game_id"


def canonical_sort(df: pd.DataFrame, date_col: str = "gameday") -> pd.DataFrame:
    """The ONE row order fold indices are valid for.

    Stable, total ordering by (date_col, game_id) with a fresh RangeIndex.
    """
    keys = [date_col, CANONICAL_TIEBREAK] if CANONICAL_TIEBREAK in df.columns \
        else [date_col]
    return df.sort_values(keys, kind="mergesort").reset_index(drop=True)


@dataclass
class Fold:
    """One causal expanding train/validation split."""

    fold_id: int
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    train_idx: pd.Index
    val_idx: pd.Index
    is_partial_tail: bool = False


def make_folds(
    df: pd.DataFrame,
    date_col: str = "gameday",
    cadence_days: int | None = None,
    min_val_games: int | None = None,
    max_eval_folds: int = 0,
) -> list[Fold]:
    """Build non-overlapping expanding folds over observed game dates.

    ``WARMUP_DAYS`` is an observed-date index, matching the authoritative
    MLB/NHL production implementation.  The first validation index is
    ``max(cadence_days, WARMUP_DAYS)``.  A normal window is retained only
    when it contains ``MIN_VAL_FOLD_GAMES`` rows; the final partial window
    is retained even when it is below that gate so the newest decided games
    are not silently discarded.

    The caller's row indices are preserved in ``train_idx``/``val_idx``.
    Dates are normalized here so timestamps cannot create spurious observed
    dates.

    ``train_idx``/``val_idx`` are POSITIONS in the canonical row order, so the
    frame is put in that order here rather than trusting the caller to have
    done it. Every consumer indexes these labels after its own
    ``reset_index(drop=True)``, so a frame that arrived out of order would
    otherwise hand each fold a set of positions that select other games
    entirely — the right dates scored against the wrong teams. Doing it inside
    the fold builder makes the agreement structural instead of a convention a
    later caller can silently break.
    """
    if date_col not in df.columns:
        raise KeyError(f"make_folds: missing date column {date_col!r}")

    cadence = int(cadence_days or config.RETRAIN_CADENCE_DAYS)
    if cadence <= 0:
        raise ValueError("cadence_days must be positive")
    minimum = int(config.MIN_VAL_FOLD_GAMES if min_val_games is None else min_val_games)
    if minimum < 0:
        raise ValueError("min_val_games cannot be negative")

    df = canonical_sort(df, date_col)
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    if "season" in df.columns:
        seasons = pd.to_numeric(df["season"], errors="coerce")
        core = seasons.ge(config.OOF_FIRST_SEASON) & dates.notna()
    else:
        # Small unit-test fixtures may omit season; dates are then the only
        # eligibility signal available.
        core = dates.notna()

    unique_dates = list(dates[core].dropna().drop_duplicates().sort_values())
    if not unique_dates:
        return []

    val_start_idx = max(cadence, int(config.WARMUP_DAYS))
    if val_start_idx >= len(unique_dates):
        return []

    candidates: list[Fold] = []
    fold_id = 0
    while val_start_idx < len(unique_dates):
        val_end_idx = min(val_start_idx + cadence, len(unique_dates))
        val_start = pd.Timestamp(unique_dates[val_start_idx])
        val_end = pd.Timestamp(unique_dates[val_end_idx - 1])
        partial = val_end_idx < val_start_idx + cadence

        # Training is restricted to the declared eligible core as well as
        # strictly prior dates.  Otherwise a caller that supplies historical
        # rows alongside 2024+ data would silently train on pre-contract
        # seasons even though validation is core-only.
        train_mask = core & dates.lt(val_start)
        val_mask = core & dates.ge(val_start) & dates.le(val_end)
        train_idx = df.index[train_mask]
        val_idx = df.index[val_mask]
        if len(train_idx) and len(val_idx):
            candidates.append(
                Fold(
                    fold_id=fold_id,
                    val_start=val_start,
                    val_end=val_end,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    is_partial_tail=partial,
                )
            )
            fold_id += 1
        val_start_idx = val_end_idx

    folds = [
        fold for fold in candidates
        if len(fold.val_idx) >= minimum or fold.is_partial_tail
    ]
    if max_eval_folds and max_eval_folds > 0 and len(folds) > max_eval_folds:
        folds = folds[-max_eval_folds:]
    return folds


def fold_summary(folds: list[Fold]) -> dict:
    """Return compact, JSON-friendly fold diagnostics."""
    if not folds:
        return {"n_folds": 0, "partial_tail_folds": 0}
    return {
        "n_folds": len(folds),
        "partial_tail_folds": int(sum(fold.is_partial_tail for fold in folds)),
        "first_val_start": str(folds[0].val_start.date()),
        "last_val_end": str(folds[-1].val_end.date()),
        "min_train": int(min(len(fold.train_idx) for fold in folds)),
        "max_train": int(max(len(fold.train_idx) for fold in folds)),
        "total_val_games": int(sum(len(fold.val_idx) for fold in folds)),
    }


def fold_table(
    df: pd.DataFrame,
    folds: list[Fold],
    date_col: str = "gameday",
) -> pd.DataFrame:
    """Return one reporting row per fold without changing its geometry."""
    dates = pd.to_datetime(df[date_col], errors="coerce")
    rows = []
    for fold in folds:
        train_end = dates.loc[fold.train_idx].max() if len(fold.train_idx) else pd.NaT
        rows.append(
            {
                "fold_id": fold.fold_id,
                "train_end_date": train_end.date() if pd.notna(train_end) else pd.NaT,
                "validation_start": fold.val_start.date(),
                "validation_end": fold.val_end.date(),
                "is_partial_tail": bool(fold.is_partial_tail),
                "n_train": int(len(fold.train_idx)),
                "n_validation": int(len(fold.val_idx)),
            }
        )
    return pd.DataFrame(rows)
