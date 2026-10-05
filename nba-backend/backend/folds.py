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

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from backend import config
except ImportError:  # pragma: no cover - direct script/import fallback
    import config

#: How many provisional windows the thin-population warning names before
#: summarising the rest as "+N more" (NFL/NHL parity, 2026-10-05 run-log
#: review): the unbounded full join measured 923 characters at production
#: length (16 of 56 windows) — the same one-line dump class the NFL and
#: NHL reviews flagged, and NBA was the last backend still emitting it.
_THIN_WINDOW_SAMPLE = 5

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
    """One causal expanding train/validation split.

    ``season_type`` classifies the VALIDATION population by row-level
    ``game_type`` (``regular`` / ``postseason`` / ``mixed``); it is
    informational metadata, never a reason to drop a window.

    ``provisional`` marks a window under ``MIN_VAL_FOLD_GAMES``. Provisional
    folds are still FIT and still scored -- their rows land in
    ``*_predictions_history`` -- but they never grade the blend weights, the
    Platt map, or the headline pooled metrics. This is the 2026-10-03 fix for
    the gate's tail contradiction: the gate existed to keep thin POSTSEASON
    folds out of pooled metrics, yet ``is_partial_tail`` re-admitted exactly
    one such fold (n=3 in the NBA, n=5 in MLB), so a 3-game Finals window
    graded the model while a 34-game playoff window did not.
    """

    fold_id: int
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    train_idx: pd.Index
    val_idx: pd.Index
    is_partial_tail: bool = False
    season_type: str = "regular"
    provisional: bool = False

    @property
    def grades_pooled(self) -> bool:
        """May this fold's rows contribute to weights/calibration/headline?"""
        return not self.provisional


def postseason_flag(values: pd.Series) -> pd.Series:
    """Row-level playoff flag for a ``game_type``-like column.

    Byte-for-byte the expression ``features.build_*`` uses to build the
    ``is_playoffs`` FEATURE, and deliberately so: the fold gate decides which
    rows may grade the model, the model consumes ``is_playoffs`` as an input,
    and those two must never disagree about what "playoff" means. A test
    pins the equivalence rather than relying on the two copies staying in
    step by hand.

    Accepts numeric codes (NBA ``GAME_TYPE_POST``) or textual spellings
    (``playoff``/``postseason``); unknown/missing values read as regular.
    """
    num = pd.to_numeric(values, errors="coerce")
    text = values.astype("string").str.lower()
    return ((num.eq(config.GAME_TYPE_POST)
             | text.str.contains("play|post", regex=True, na=False))
            .fillna(False).astype(bool))


def make_folds(
    df: pd.DataFrame,
    date_col: str = "gameday",
    cadence_days: int | None = None,
    min_val_games: int | None = None,
    max_eval_folds: int = 0,
    diagnostics: dict | None = None,
) -> list[Fold]:
    """Build non-overlapping expanding folds over observed game dates.

    ``WARMUP_DAYS`` is an observed-date index, matching the authoritative
    MLB/NHL production implementation.  The first validation index is
    ``max(cadence_days, WARMUP_DAYS)``.

    EVERY observed window with at least one eligible validation row becomes a
    fold.  ``MIN_VAL_FOLD_GAMES`` no longer discards windows: a window under
    the gate is marked ``provisional`` and is fit + scored, but it never
    contributes to pooled metrics, blend weights or calibration.  Before
    2026-10-03 the gate dropped such windows outright, which on this calendar
    removed the entire 2025 and 2026 postseason (~195 games) plus two thin
    regular-season weeks -- silently, with no log line.

    ``diagnostics`` (optional out-param) receives ``_geometry(folds, minimum)``
    so callers can publish dropped/provisional coverage in their run summary.

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

    # Row-level season type drives Fold.season_type. Prefer the raw
    # ``game_type`` (authoritative, and present on unit-test fixtures that
    # never built features); fall back to the derived ``is_playoffs`` feature
    # so a frame without game_type still classifies. Absent both, every
    # window classifies as regular, which keeps such fixtures byte for byte
    # identical to the pre-2026-10-03 behaviour.
    if "game_type" in df.columns:
        post = postseason_flag(df["game_type"])
    elif "is_playoffs" in df.columns:
        post = pd.to_numeric(df["is_playoffs"], errors="coerce") \
            .fillna(0).gt(0.5)
    else:
        post = None

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
            n_val = len(val_idx)
            if post is None:
                season_type = "regular"
            else:
                n_post = int(post.loc[val_idx].sum())
                season_type = ("regular" if n_post == 0 else
                               "postseason" if n_post == n_val else
                               "mixed")
            candidates.append(
                Fold(
                    fold_id=fold_id,
                    val_start=val_start,
                    val_end=val_end,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    is_partial_tail=partial,
                    season_type=season_type,
                    provisional=n_val < minimum,
                )
            )
            fold_id += 1
        val_start_idx = val_end_idx

    # RETAIN, DON'T DROP (2026-10-03). The previous contract discarded every
    # non-tail window under MIN_VAL_FOLD_GAMES. On the NBA calendar that
    # removed 13 of 15 rejected windows -- ~195 playoff games across the 2025
    # and 2026 postseasons -- plus two ordinary regular-season weeks that
    # missed the 40-game gate by 5 and 3 games. The rejection was never
    # random: a 7-day slice is thin BECAUSE the postseason plays 2-8 games a
    # night, so the filter selected on exactly the regime under study. NHL
    # reached the same conclusion on 2026-09-30 (folds.py history there:
    # "playoff springs were thinned out of OOF across the whole history").
    # The gate now decides only whether a window GRADES the model, not whether
    # its games are scored at all.
    folds = candidates
    provisional = [f for f in folds if f.provisional]
    if provisional:
        # Bounded (2026-10-05 run-log review): the delivered log joined
        # every provisional window into ONE 923-char line (16 windows). The
        # COUNTS carry the verdict (the fold summary right below repeats
        # them as provisional_windows/provisional_games); a sample of the
        # first few windows shows the shape WITH its season-type label, and
        # the rest collapses to "+N more" instead of extending the line
        # forever. This is also the ONLY thin-window emission — NBA never
        # carried master_pipeline's second sample, so one warning here is
        # the whole story.
        sample = ", ".join(
            f"[{f.val_start.date()}..{f.val_end.date()} "
            f"n={len(f.val_idx)} {f.season_type}]"
            for f in provisional[:_THIN_WINDOW_SAMPLE])
        rest = len(provisional) - _THIN_WINDOW_SAMPLE
        if rest > 0:
            sample += f", +{rest} more"
        logger.warning(
            "OOF validation population is thin on %d of %d window(s) "
            "(< MIN_VAL_FOLD_GAMES=%d games): %s - retained and scored, "
            "but PROVISIONAL: excluded from pooled metrics, blend weights "
            "and calibration (playoff weeks and season ramps are the usual "
            "cause)",
            len(provisional), len(folds), minimum, sample,
        )
    dropped = [f for f in folds if f.season_type != "regular"]
    if dropped:
        logger.info(
            "OOF season-type split: %d window(s) carry postseason games "
            "(%d rows); their playoff rows are reported as oof_postseason "
            "and never grade the blend",
            len(dropped),
            sum(int(post.loc[f.val_idx].sum()) if post is not None else 0
                for f in dropped),
        )
    if diagnostics is not None:
        diagnostics.update(_geometry(folds, minimum))
    if max_eval_folds and max_eval_folds > 0 and len(folds) > max_eval_folds:
        folds = folds[-max_eval_folds:]
    return folds


def _geometry(folds: list[Fold], minimum: int) -> dict:
    """JSON-friendly coverage accounting shared by every sport's summary.

    ``dropped_windows``/``dropped_games`` are always 0 since the 2026-10-03
    retain-don't-drop change; they are reported so a future regression back to
    silent dropping shows up as a non-zero rather than as a coverage number
    that quietly shrinks.
    """
    prov = [f for f in folds if f.provisional]
    post = [f for f in folds if f.season_type != "regular"]
    return {
        "n_candidates": len(folds),
        "dropped_windows": 0,
        "dropped_games": 0,
        "provisional_windows": len(prov),
        "provisional_games": int(sum(len(f.val_idx) for f in prov)),
        "postseason_windows": len(post),
        "min_val_fold_games": int(minimum),
    }


def fold_summary(folds: list[Fold],
                 minimum: int | None = None) -> dict:
    """Return compact, JSON-friendly fold diagnostics.

    ``minimum`` (defaults to ``MIN_VAL_FOLD_GAMES``) is threaded through so
    the geometry block can report provisional/dropped coverage alongside the
    fold counts -- the omission that let 15 rejected windows go unreported.
    """
    if not folds:
        return {"n_folds": 0, "partial_tail_folds": 0}
    gate = int(config.MIN_VAL_FOLD_GAMES if minimum is None else minimum)
    out = {
        "n_folds": len(folds),
        "partial_tail_folds": int(sum(fold.is_partial_tail for fold in folds)),
        "first_val_start": str(folds[0].val_start.date()),
        "last_val_end": str(folds[-1].val_end.date()),
        "min_train": int(min(len(fold.train_idx) for fold in folds)),
        "max_train": int(max(len(fold.train_idx) for fold in folds)),
        "total_val_games": int(sum(len(fold.val_idx) for fold in folds)),
    }
    out.update(_geometry(folds, gate))
    return out


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
                "season_type": fold.season_type,
                "provisional": bool(fold.provisional),
                "grades_pooled": bool(fold.grades_pooled),
                "n_train": int(len(fold.train_idx)),
                "n_validation": int(len(fold.val_idx)),
            }
        )
    return pd.DataFrame(rows)
