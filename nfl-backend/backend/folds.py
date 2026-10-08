"""Walk-forward fold generation — expanding, calendar-day based.

Geometry (spec section 6):
  - validation windows are chronological, non-overlapping, 7 calendar days
    wide, keyed on game DATES (never NFL week IDs)
  - training = all eligible games STRICTLY BEFORE the validation window
  - training expands over time; the final partial window is retained
  - 2018 is warmup only: it appears in training sets (strictly-prior state)
    but never in a validation window
  - every non-empty window is a fold; ordinary windows with fewer than
    MIN_VAL_FOLD_GAMES games are retained but marked ``provisional``, so they
    are fit and scored without ever grading pooled metrics, blend weights or
    calibration. The 2026-10-03 change: the old skip removed ~758 of 2,431
    core games from validation -- every playoff game in ten seasons (the OOF
    had no February date at all) plus hundreds of regular-season games whose
    7-calendar-day block straddled a week and fell under the gate.

A fold is a (fold_id, val_start, val_end, train_idx, val_idx) tuple over the
row order of the caller's frame; callers must pass frames already in
``canonical_sort`` order (see below) so the returned labels are positional and
remain valid for every downstream consumer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

try:
    from backend import config
except ImportError:
    import config

logger = logging.getLogger(__name__)

# How many provisional windows the thin-population WARNING names inline.
# The 2026-10-05 run listed all 57 in ONE 2,516-char line; counts plus a
# short sample carry the same verdict (run-log review, 2026-10-05).
_THIN_WINDOW_SAMPLE = 5

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
    season_type: str = "regular"     # regular | postseason | mixed
    provisional: bool = False        # under MIN_VAL_FOLD_GAMES: scored,
    #                                  never grades weights/calibration

    @property
    def grades_pooled(self) -> bool:
        """May this fold's rows contribute to weights/calibration/headline?"""
        return not self.provisional


# The postseason spellings postseason_flag accepts: the nflverse ROUND
# codes the schedule actually emits (verified against
# nflreadpy.load_schedules 2016-2026: value counts REG/WC/DIV/CON/SB only
# -- never "POST") plus the legacy "POST" spelling fixtures still carry.
POSTSEASON_GAME_TYPES = frozenset({"POST", "WC", "DIV", "CON", "SB"})


def postseason_flag(values: pd.Series) -> pd.Series:
    """Row-level playoff flag for an NFL ``game_type`` column.

    The nflverse schedule spells postseason by ROUND -- ``WC`` / ``DIV`` /
    ``CON`` / ``SB`` -- and never emits ``POST``, so the old "post"
    substring match read every real playoff row as regular (the 2026-10-08
    fix: the round codes were added to ``config.GAME_TYPES`` and here).
    Regular (``REG``), preseason (``PRE``), unknown and missing all read as
    regular. Kept as one shared helper so the gate and any future
    ``is_playoffs`` feature cannot drift apart.
    """
    return (values.astype("string").str.strip().str.upper()
            .isin(POSTSEASON_GAME_TYPES))


def make_folds(df: pd.DataFrame,
               date_col: str = "gameday",
               cadence_days: int | None = None,
               min_val_games: int | None = None,
               diagnostics: dict | None = None) -> list[Fold]:
    """Build expanding walk-forward folds over ``df``.

    ``df`` may contain warmup rows (2018) and core rows (2019+). Validation
    windows cover ONLY core-season games (OOF_FIRST_SEASON and later);
    warmup rows join training sets when they fall strictly before a window.
    Training is every eligible row STRICTLY BEFORE ``val_start``.

    Every NON-EMPTY window becomes a fold. ``MIN_VAL_FOLD_GAMES`` marks a thin
    one ``provisional`` instead of discarding it: the games are fit, scored
    and shipped, but they never grade pooled metrics, blend weights or
    calibration. Before 2026-10-03 the skip removed ~758 of 2,431 core games
    from validation -- hundreds of regular-season games whose 7-calendar-day
    block straddled a week -- while postseason games never reached this
    function at all: ingest's ``GAME_TYPES`` matched only the spelling
    "POST", which the nflverse schedule never emits (it spells the rounds
    WC/DIV/CON/SB), so every in-window playoff game was dropped before
    folding and no February date ever appeared in the OOF. Both defects
    were fixed 2026-10-08.
    """
    cadence = cadence_days or config.RETRAIN_CADENCE_DAYS
    if min_val_games is None:
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

    # Row-level season type drives Fold.season_type. Absent the column (small
    # unit-test fixtures) every window classifies as regular.
    post = (postseason_flag(df["game_type"])
            if "game_type" in df.columns else None)

    folds: list[Fold] = []
    fold_id = 0
    win_start = d_min
    while win_start <= d_max:
        win_end = win_start + pd.Timedelta(days=cadence - 1)   # inclusive
        val_mask = (core_mask & (dates >= win_start)
                    & (dates <= win_end + pd.Timedelta(hours=23, minutes=59, seconds=59)))
        val_idx = df.index[val_mask]
        # Two guards, neither of them the value gate:
        #   * an empty validation window is not a fold (a schedule gap can
        #     never manufacture a validation set out of nothing);
        #   * an empty TRAINING window is not a fold either. The old
        #     MIN_VAL_FOLD_GAMES skip used to hide this -- it discarded the
        #     first window of any frame with no pre-core warmup, and keeping
        #     it would score rows no member can predict for (lightgbm: "input
        #     data must be 2 dimensional and non empty", elasticnet: "0
        #     samples"). Production never trips it: the 2016 warmup season
        #     precedes the first core window, so fold 0 ships n_train=256.
        if len(val_idx) and len(df.index[dates < win_start]):
            n_val = len(val_idx)
            if post is None:
                season_type = "regular"
            else:
                n_post = int(post.loc[val_idx].sum())
                season_type = ("regular" if n_post == 0 else
                               "postseason" if n_post == n_val else
                               "mixed")
            train_mask = dates < win_start
            train_idx = df.index[train_mask]
            folds.append(Fold(fold_id=fold_id, val_start=win_start,
                              val_end=win_end, train_idx=train_idx,
                              val_idx=val_idx,
                              season_type=season_type,
                              provisional=n_val < min_val_games))
            fold_id += 1
        win_start = win_end + pd.Timedelta(days=1)

    provisional = [f for f in folds if f.provisional]
    if provisional:
        # The window list is bounded: at production length the full join
        # was a 2,516-char single line (57 windows, 2026-10-05 run) — the
        # same one-line dump class the run-log review flagged. The COUNTS
        # carry the verdict (the fold summary right below repeats them as
        # provisional_windows/provisional_games); a sample of the first
        # few windows shows the shape, and the per-window detail stays in
        # nfl_fold_table.csv.
        sample = ", ".join(
            f"[{f.val_start.date()}..{f.val_end.date()} "
            f"n={len(f.val_idx)} {f.season_type}]"
            for f in provisional[:_THIN_WINDOW_SAMPLE])
        rest = len(provisional) - _THIN_WINDOW_SAMPLE
        if rest > 0:
            sample += f", +{rest} more (per-window detail: nfl_fold_table.csv)"
        logger.warning(
            "OOF validation population is thin on %d of %d window(s) "
            "(< MIN_VAL_FOLD_GAMES=%d games): %s - retained and scored, "
            "but PROVISIONAL: excluded from pooled metrics, blend weights "
            "and calibration (playoff weeks, bye weeks and week-straddling "
            "blocks are the usual cause)",
            len(provisional), len(folds), min_val_games, sample,
        )
    non_regular = [f for f in folds if f.season_type != "regular"]
    if non_regular and post is not None:
        logger.info(
            "OOF season-type split: %d window(s) carry postseason games "
            "(%d rows); their playoff rows are reported as oof_postseason "
            "and never grade the blend",
            len(non_regular),
            sum(int(post.loc[f.val_idx].sum()) for f in non_regular),
        )
    if diagnostics is not None:
        diagnostics.update(_geometry(folds, min_val_games))
    return folds


def _geometry(folds: list[Fold], minimum: int) -> dict:
    """JSON-friendly coverage accounting shared with MLB/NBA/NHL.

    ``dropped_windows``/``dropped_games`` are always 0 since 2026-10-03;
    they are reported so a regression back to silent dropping surfaces as a
    non-zero instead of as a coverage number that quietly shrinks. Before
    that date these two were never reported at all, which is how ~758 of
    2,431 core games (31%) went unscored without a single log line.
    """
    prov = [f for f in folds if f.provisional]
    return {
        "n_candidates": len(folds),
        "dropped_windows": 0,
        "dropped_games": 0,
        "provisional_windows": len(prov),
        "provisional_games": int(sum(len(f.val_idx) for f in prov)),
        "postseason_windows": int(
            sum(1 for f in folds if f.season_type != "regular")),
        "min_val_fold_games": int(minimum),
    }


def fold_summary(folds: list[Fold],
                 minimum: int | None = None) -> dict:
    """Diagnostics for the final validation record."""
    if not folds:
        return {"n_folds": 0}
    gate = int(minimum if minimum is not None
               else getattr(config, "MIN_VAL_FOLD_GAMES", 15))
    out = {
        "n_folds": len(folds),
        "first_val_start": str(folds[0].val_start.date()),
        "last_val_end": str(folds[-1].val_end.date()),
        "min_train": int(min(len(f.train_idx) for f in folds)),
        "max_train": int(max(len(f.train_idx) for f in folds)),
        "total_val_games": int(sum(len(f.val_idx) for f in folds)),
    }
    out.update(_geometry(folds, gate))
    return out


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
            "validation_start": f.val_start.date(),                "validation_end": f.val_end.date(),
                "season_type": f.season_type,
                "provisional": bool(f.provisional),
                "grades_pooled": bool(f.grades_pooled),
                "n_train": int(len(f.train_idx)),
                "n_validation": int(len(f.val_idx)),
        })
    return pd.DataFrame(rows)
