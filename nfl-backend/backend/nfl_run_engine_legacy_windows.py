"""Legacy methodology pin for the run-engine modules (d1 alignment).

This module is the run-engine-side home for the PRE-CUTOVER moneyline
methodology ONLY. nfl_moneyline no longer exports TRAIN_SEASONS /
VAL_SEASONS / SEALED_SEASON (the production full-history refit uses the
wide-pool constants only), and the legacy 88 calendar-week fold generator
(generate_weekly_folds) was removed from the serving module with the
wide-pool adoption.

Nothing in this module may be re-pointed at the wide-pool constants
(CORE_SEASONS / WARMUP_SEASONS / generate_week_id_folds) — doing so would
silently adopt the new methodology in harnesses that are supposed to stay
on the old geometry this cycle.

Pin comment: legacy methodology, preserved for d1 alignment; values must
match pre-cutover nfl_moneyline.

Pre-cutover values (pinned; must match the commit whose record this file
belongs to):
    TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
    VAL_SEASONS    = [2021, 2022, 2023, 2024]
    SEALED_SEASON  = 2025
"""
from __future__ import annotations

import pandas as pd

# --- Legacy gate tokens (localized for d1 alignment) ---------------------
# Exact pre-cutover nfl_moneyline values. Run-engine modules that still
# operate under the old methodology import these HERE instead of from
# nfl_moneyline. Do NOT point any consumer at the wide-pool constants.
TRAIN_SEASONS: list[int] = list(range(2019, 2025))   # pre-sealed training: 2019..2024
VAL_SEASONS: list[int] = [2021, 2022, 2023, 2024]    # prequential validation (2-season warm-up)
SEALED_SEASON: int = 2025                            # legacy sealed-2025 hold-out


def _week_start(dates: pd.Series) -> pd.Series:
    """Monday of each date's calendar week (NFL weeks are Mon-Sun)."""
    d = pd.to_datetime(dates)
    return d - pd.to_timedelta(d.dt.weekday, unit="D")


def generate_weekly_folds(preq: pd.DataFrame,
                          val_seasons: list[int] | None = None) -> list[dict]:
    """Prequential weekly folds over ``preq`` (2019-2024 decided games).

    Each fold validates all games in one calendar week (Mon-Sun) of a
    validation season; its train set is EVERY game with gameday strictly
    before that week (so a fold can never see its own or any future week).

    LEAKAGE ASSERTION: for every fold, max(train.gameday) < min(val.gameday).

    LEGACY METHODOLOGY, preserved for d1 alignment — byte-identical to the
    pre-cutover nfl_moneyline generator. The wide-pool serving module uses
    generate_week_id_folds instead; do NOT re-point this at it.
    """
    val_seasons = val_seasons or VAL_SEASONS
    g = preq.copy()
    g["gameday"] = pd.to_datetime(g["gameday"], errors="coerce")
    g = g.sort_values("gameday").reset_index(drop=True)
    g["week_start"] = _week_start(g["gameday"])
    g["val_season"] = g["season"].isin(val_seasons)
    folds = []
    for mon, idx in g[g["val_season"]].groupby("week_start")["week_start"].groups.items():
        val = g.loc[idx]
        train = g[g["gameday"] < mon]
        if len(val) == 0 or len(train) == 0:
            continue
        tr_max = train["gameday"].max()
        va_min = val["gameday"].min()
        if not (tr_max < mon <= va_min):
            raise AssertionError(
                f"fold week {mon}: train max {tr_max} not strictly before "
                f"val min {va_min} -> future-week leak")
        folds.append({"week_start": mon, "train": train.copy(), "val": val.copy()})
    folds.sort(key=lambda f: f["week_start"])
    return folds