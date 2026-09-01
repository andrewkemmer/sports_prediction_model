"""NFL game-frame ingestion — the NFL analogue of mlb-backend's canonical
game-level frame loader (``frames.py``/``data_delivery/game_level_features.csv``).

Data source
-----------
``nflreadpy`` v0.1+ (the official Python port of the nflverse project; the
current standard — ``nfl_data_py`` was archived in late 2025 and is NOT used).
``load_schedules`` + ``load_pbp`` for a configurable season range. The nflverse
schedule feed already carries final scores per game, so the game-level
aggregation is one clean transform; play-by-play is joined only for per-game
play counts (``n_plays``) and game_id cross-validation.

MARKET-INDEPENDENCE POLICY: market/odds columns from the nflverse feed
(``spread_line``, ``total_line``, ``home_moneyline``, ``away_moneyline``) are
DROPPED AT LOAD — they never enter the decided-game frame (nor the slate
frame), are never model features, never gate benchmarks, and never reach the
board. The pipeline is end-to-end market-independent by policy.

Schema (one row per DECIDED game)
---------------------------------
``game_id``     nflverse game id (str, e.g. ``2019_01_KC_JAX``)
``season``      calendar season year (int)
``week``        week number; postseason weeks are 18+ (int)
``game_type``   REG | WC | DIV | CON | SB
``gameday``     game date (YYYY-MM-DD, local to the venue)
``away_team``   away team abbreviation (str)
``home_team``   home team abbreviation (str)
``away_score``  final away points (int, non-null = decided)
``home_score``  final home points (int, non-null = decided)
``result``      home margin = home_score - away_score (float; positive = home win)
``total``       combined final points (float)
``n_plays``     play count from play-by-play for the game (int; NaN if pbp missing)

Note: the frame carries NO market/odds columns (no spread_line / total_line /
moneylines) — market-independence policy (see module docstring).

Decided-frame rules (encoded ONCE here, mirroring mlb-backend ``frames.py``):
1. **Post-game only**: ``away_score``, ``home_score`` AND ``result`` all
   non-null. Pre-game/undecided rows never enter the decided frame.
2. **Deterministic dedup**: one row per ``game_id`` — the LATEST ``gameday``
   wins (stable mergesort, ``keep="last"``, so identical (game_id, gameday)
   ties resolve by input order).
3. **Stable chronological order**: mergesort by ``gameday`` only — within-day
   input order is preserved (same discipline as mlb-backend's canonical frame,
   where row order is part of the contract).

Artifacts
---------
``data_delivery/nfl_game_level_features.csv``            canonical frame (overwritten)
``data_delivery/nfl_game_level_features_YYYYMMDD.csv``   dated snapshot of the same frame

Usage
-----
    python3 nfl_game_frame.py                       # default 2019-2025
    python3 nfl_game_frame.py --seasons 2019 2020 2021 2022 2023 2024 2025
    from nfl_game_frame import pull_and_build
    summary = pull_and_build([2019, 2020])          # returns counts + sha256
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent          # nfl-backend/
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"

# Season range. 2019-2024 was validated by the ingestion spike; 2025 is fully
# decided (it is 2026) and becomes the sealed holdout for the moneyline model
# (feature v1 admission: game_frame + nfl_features). 2026 and later are ignored.
DEFAULT_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

# The exact game-level contract columns, in order (spike schema).
GAME_LEVEL_COLUMNS = [
    "game_id", "season", "week", "game_type", "gameday",
    "away_team", "home_team", "away_score", "home_score",
    "result", "total", "n_plays",
]

DATE_FMT = "%Y%m%d"


# ---------------------------------------------------------------------------
# Pure aggregation (no network) — the testable core
# ---------------------------------------------------------------------------
def aggregate_game_frame(schedule: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per game_id in the schedule, joined with per-game play counts.

    Raw rows are preserved here (including undecided games); the decided
    filter/dedup/sort lives in :func:`canonical_decided_frame` so consumers
    cannot diverge on the decided definition.
    """
    sched = schedule.copy()
    sched["game_id"] = sched["game_id"].astype(str)
    sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")

    plays = pbp.copy()
    plays["game_id"] = plays["game_id"].astype(str)
    play_counts = (
        plays.groupby("game_id", as_index=False)["play_id"].count().rename(
            columns={"play_id": "n_plays"})
    )

    game = sched.merge(play_counts, on="game_id", how="left")
    cols = [c for c in GAME_LEVEL_COLUMNS if c in game.columns]
    return game[cols].reset_index(drop=True)


def canonical_decided_frame(game: pd.DataFrame) -> pd.DataFrame:
    """The ONE canonical decided frame (mirrors mlb-backend frames.py policy).

    Encodes the decided rules once: post-game rows only (all three of
    away_score/home_score/result non-null), deterministic dedup by game_id
    (latest gameday wins), stable chronological order (mergesort by gameday —
    within-day input order preserved). A duplicate game_id can never silently
    split across seasons/folds.
    """
    out = game.copy()
    if out.empty:
        return out

    decided = out[out[["away_score", "home_score", "result"]].notna().all(axis=1)].copy()

    if "game_id" not in decided.columns:
        raise ValueError("canonical_decided_frame: frame must carry 'game_id'")

    # Deterministic dedup: latest gameday wins per game_id (stable mergesort,
    # so identical (game_id, gameday) pairs resolve by input order).
    if decided["game_id"].duplicated().any():
        decided = (
            decided.assign(_order=pd.to_datetime(decided["gameday"], errors="coerce"))
            .sort_values("_order", kind="mergesort")
            .drop_duplicates(subset="game_id", keep="last")
            .drop(columns="_order")
        )
    # Stable chronological order without permuting within-day rows.
    decided = decided.sort_values("gameday", kind="mergesort").reset_index(drop=True)
    return decided


# ---------------------------------------------------------------------------
# Network pull + artifact write
# ---------------------------------------------------------------------------
def _load_schedule(seasons: list[int]) -> pd.DataFrame:
    """load_schedules via nflreadpy (returns a polars frame -> pandas)."""
    import nflreadpy
    return nflreadpy.load_schedules(seasons).to_pandas()


def _load_pbp(seasons: list[int]) -> pd.DataFrame:
    """load_pbp via nflreadpy; keep only the columns the aggregation needs.

    The full play-by-play is ~300k rows x ~370 columns — converting the whole
    frame to pandas blows past memory. Selecting game_id + play_id in polars
    first keeps the pandas side tiny (the spike used the same trick, writing
    parquet directly from the polars frame).
    """
    import nflreadpy
    pbp = nflreadpy.load_pbp(seasons)
    return pbp.select(["game_id", "play_id"]).to_pandas()


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pull_and_build(seasons: list[int] | None = None,
                   out_dir: Path | None = None,
                   write_artifacts: bool = True) -> dict:
    """Pull schedule + pbp for the season range, build the decided game frame,
    write ``nfl_game_level_features.csv`` (+ dated snapshot), and return a
    validation summary dict (counts, sha256). Market/odds columns never enter
    the frame (dropped at load — market-independence policy).

    Validation mirrors the spike's go/no-go checks: per-season decided counts,
    missing-score rows, duplicate game_ids, home win rate, and combined
    points per game.
    """
    seasons = seasons or DEFAULT_SEASONS
    out_dir = Path(out_dir) if out_dir is not None else DATA_DELIVERY_DIR

    logger.info("Loading nflreadpy schedule for %s", seasons)
    schedule = _load_schedule(seasons)
    logger.info("Loading nflreadpy play-by-play for %s", seasons)
    pbp = _load_pbp(seasons)
    logger.info("schedule rows: %d | pbp rows: %d",
                len(schedule), len(pbp))

    game = aggregate_game_frame(schedule, pbp)
    decided = canonical_decided_frame(game)

    # ---- validation (the spike's go/no-go criteria, re-checked each build) ----
    n_game = len(decided)
    missing_score = int(decided[["away_score", "home_score"]].isna().any(axis=1).sum())
    raw_dup_ids = int(game["game_id"].duplicated().sum())
    dup_ids = int(decided["game_id"].duplicated().sum())
    home_win = int((decided["home_score"] > decided["away_score"]).sum())
    home_win_rate = home_win / n_game if n_game else 0.0
    combined_ppg = (decided["home_score"].sum() + decided["away_score"].sum()) / n_game \
        if n_game else 0.0

    per_season = (
        decided.assign(_missing_score=decided[["away_score", "home_score"]].isna().any(axis=1))
        .groupby("season", as_index=False)
        .agg(games=("game_id", "count"),
             missing_scores=("_missing_score", "sum"),
             unique_game_ids=("game_id", "nunique"))
    )

    summary = {
        "seasons": seasons,
        "games": n_game,
        "missing_scores": missing_score,
        "duplicate_game_ids_raw_schedule": raw_dup_ids,
        "duplicate_game_ids_decided": dup_ids,
        "home_win_rate": round(home_win_rate, 4),
        "combined_ppg": round(combined_ppg, 1),
        "per_season": per_season.to_dict("records"),
        "sha256": None,
        "artifacts": [],
    }

    if write_artifacts:
        out_dir.mkdir(parents=True, exist_ok=True)
        canonical = out_dir / "nfl_game_level_features.csv"
        dated = out_dir / f"nfl_game_level_features_{datetime.now().strftime(DATE_FMT)}.csv"
        decided.to_csv(canonical, index=False)
        decided.to_csv(dated, index=False)
        summary["artifacts"] = [str(canonical), str(dated)]
        summary["sha256"] = _sha256_of(canonical)

    # ---- console report (same shape as the spike) ----
    print(f"\n=== NFL decided games ({seasons[0]}-{seasons[-1]}) ===")
    print(per_season.to_string(index=False))
    print(f"missing-score rows: {missing_score}")
    print(f"duplicate game_ids (raw schedule): {raw_dup_ids} | (decided): {dup_ids}")
    print(f"home win rate: {home_win_rate:.3f} ({home_win}/{n_game})")
    print(f"combined points per game: {combined_ppg:.1f}")
    print(f"sha256: {summary['sha256']}")
    print(f"artifacts: {', '.join(summary['artifacts'])}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build the NFL game-level frame from nflreadpy "
                    "(schedule + pbp) and write data_delivery artifacts.")
    ap.add_argument("--seasons", nargs="+", type=int, default=DEFAULT_SEASONS,
                    help="Season years to pull (default: 2019-2025)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    pull_and_build(args.seasons)
    return 0


if __name__ == "__main__":
    sys.exit(main())
