"""Backfill the game-day-submission availability archive over a date range.

    python backfill_injury_designations.py 2024-01-01 2026-06-13

METHODOLOGY — one resolver for history AND the prediction slate
--------------------------------------------------------------
``nba_injury_report.game_day_designations`` reads each game at its league
game-day submission: the first published filing at/after the submission
window close (11 a.m.–1 p.m. home-local; 8–10 a.m. for tips at or before
5 p.m. local), strictly before tip-off.  Nothing published after that
moment — the late-scratch band — reaches ``status``; those filings are
preserved only in ``status_tipoff`` so the post-submission movement stays
measurable instead of shaping the feature.  The prediction slate resolves
its pending games with the SAME function (see master_pipeline), so the two
information sets are identical by construction.

Box scores are never read.  Who actually played never enters this table:
participation is the rating's input (strictly prior), never the
availability signal's.

OUTPUTS (resumable)
-------------------
  <cache>/designations_v2/nba_designations_{start}_{end}.parquet
      one shard per chunk, written as each chunk completes.  Re-running
      skips completed chunks, so an interrupted multi-hour backfill
      continues instead of restarting.
  <cache>/nba_designations.parquet
      the consolidated archive — lineup_projection.load_designations
      reads this FIRST, ahead of the legacy tipoff-era shards.
  <delivery>/nba_designations.parquet
      the same file where the pipeline ships it, the way MLB ships
      il_stints.parquet, so a Kaggle clone starts with the archive.

Usage:
    python backfill_injury_designations.py 2024-01-01 2026-06-13
    python backfill_injury_designations.py 2024-01-01 2024-02-01   # chunked
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nba_injury_report as ir  # noqa: E402
from config import DATA_DELIVERY_DIR  # noqa: E402

CACHE = Path(os.path.expanduser(
    "~/.cache/sports_prediction_model/nba/injury_reports"))
SHARD_DIR = CACHE.parent / "designations_v2"
ARCHIVE = CACHE.parent / "nba_designations.parquet"

#: The statuses that remove a player from a pool, mirrored from
#: lineup_projection.REMOVING_DESIGNATIONS so the late-scratch report below
#: measures exactly what a feature build would have done differently.
REMOVING = {"out", "doubtful", "recovery"}

CHUNK_DAYS = 14

#: Backoff schedule for an expected game day that resolved empty. A true
#: off-day has no filings and no expectation, so it never retries; a
#: scheduled game day that comes back with nothing is almost always the
#: host rate-limiting (403 -> "no such report"), which is how 15 months of
#: the 2024-25 season vanished silently on the first full run.
RETRY_BACKOFF_SEC = (8, 16, 24)


def load_schedule(start: date, end: date):
    """``(expected_game_days, schedule_by_day)`` from LOCAL sources.

    The ingested schedule carries matchups AND ``start_time_utc`` - so the
    resolver can be handed the day's games outright and never depend on
    which single filing happens to list them (the seed dependency that
    dropped games whose clubs had not filed by the seed hour).  The
    season logs widen EXPECTATION to the pre-schedule window (regular
    season before the ingested range); those days fall back to the
    filing-based slate, which is already collected.
    """
    days: set = set()
    for path in (CACHE.parent / "season_logs").glob("log_*.parquet"):
        try:
            frame = pd.read_parquet(path, columns=["gameday"])
        except Exception:  # noqa: BLE001 - a bad log is not a dead calendar
            continue
        days.update(pd.to_datetime(frame.gameday, errors="coerce")
                    .dropna().dt.date.tolist())
    schedule: dict = {}
    try:
        import ingestion
        facts = ingestion.load_ingested(allow_download=False)
        g = facts.games[["gameday", "away_team", "home_team",
                         "start_time_utc"]].copy()
        g["gameday"] = pd.to_datetime(g.gameday, errors="coerce")
        g = g.dropna(subset=["gameday", "away_team", "home_team",
                             "start_time_utc"])
        days.update(g.gameday.dt.date.tolist())
        for row in g.itertuples(index=False):
            try:
                tip = (pd.Timestamp(row.start_time_utc, tz="UTC")
                       .tz_convert("America/New_York")
                       .tz_localize(None).to_pydatetime())
            except Exception:  # noqa: BLE001 - no time, filing path for the day
                continue
            schedule.setdefault(row.gameday.date(), []).append(
                (f"{row.away_team}@{row.home_team}", tip))
    except Exception as exc:  # noqa: BLE001 - degrade to filing discovery
        print(f"  ! schedule unavailable ({exc}); retries limited to "
              f"season-log days and slate discovery falls back to filings")
    expected = {d for d in days if start <= d <= end}
    return expected, {d: v for d, v in schedule.items() if start <= d <= end}


def chunks(start: date, end: date, chunk_days: int = CHUNK_DAYS):
    """Disjoint [start, end] windows — the resume key for the shard files."""
    day = start
    while day <= end:
        stop = min(day + timedelta(days=chunk_days - 1), end)
        yield day, stop
        day = stop + timedelta(days=1)


def shard_path(shard_dir: Path, start: date, stop: date) -> Path:
    return shard_dir / f"nba_designations_{start:%Y%m%d}_{stop:%Y%m%d}.parquet"


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object")
                         for c in ir.DESIGNATION_COLUMNS})


def process_chunk(start: date, stop: date, chunk_days: int = CHUNK_DAYS,
                  expected: set | None = None, schedule: dict | None = None
                  ) -> Path:
    """Fetch + resolve every day in one window, then write its shard."""
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    out = shard_path(SHARD_DIR, start, stop)
    if out.exists():
        return out
    expected = expected or set()
    schedule = schedule or {}
    rows: list[pd.DataFrame] = []
    day = start
    while day <= stop:
        began = time.time()
        if day not in expected:
            print(f"  {day}  skipped (not on the schedule)", flush=True)
            day += timedelta(days=1)
            continue
        frame = _empty()
        attempts = len(RETRY_BACKOFF_SEC) if day in expected else 1
        for attempt in range(1, attempts + 1):
            try:
                frame = ir.game_day_designations(day, CACHE,
                                                 games=schedule.get(day))
            except Exception as exc:  # noqa: BLE001 - one bad day must not kill the run
                print(f"  {day}  ERROR {exc}", flush=True)
                frame = _empty()
            if len(frame) or attempt == attempts:
                break
            wait = RETRY_BACKOFF_SEC[attempt - 1]
            print(f"  {day}  empty on attempt {attempt}/{attempts} "
                  f"(403s so far: {ir.http_403_count}); retrying in {wait}s",
                  flush=True)
            time.sleep(wait)
        games = int(frame.matchup.nunique()) if len(frame) else 0
        print(f"  {day}  games={games:2d} rows={len(frame):3d} "
              f"({time.time() - began:4.1f}s)", flush=True)
        if len(frame):
            rows.append(frame)
        day += timedelta(days=1)
    frame = pd.concat(rows, ignore_index=True) if rows else _empty()
    frame.to_parquet(out, index=False)
    return out


def consolidate(delivery: Path | None = DATA_DELIVERY_DIR) -> pd.DataFrame:
    """Union every shard into the one archive both readers prefer."""
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(SHARD_DIR.glob("nba_designations_*.parquet"))
    frames = [pd.read_parquet(p) for p in shards]
    if not frames:
        return pd.DataFrame(columns=ir.DESIGNATION_COLUMNS)
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    out = out.sort_values(["gameday", "matchup", "player_report",
                           "team"]).reset_index(drop=True)
    out.to_parquet(ARCHIVE, index=False)
    if delivery is not None:
        try:
            Path(delivery).mkdir(parents=True, exist_ok=True)
            out.to_parquet(Path(delivery) / "nba_designations.parquet",
                           index=False)
        except OSError as exc:
            print(f"  ! delivery copy not written ({exc})")
    print(f"consolidated {len(out)} records from {len(shards)} shard(s) "
          f"-> {ARCHIVE}")
    return out


def report(frame: pd.DataFrame) -> None:
    """Coverage + the number that justifies the cutoff: late scratches."""
    if not len(frame):
        print("archive is EMPTY")
        return
    gamedays = pd.to_datetime(frame.gameday)
    games = frame[["gameday", "matchup"]].drop_duplicates()
    print(f"\nspan:            {gamedays.min().date()} .. {gamedays.max().date()}")
    print(f"game-days:       {frame.gameday.nunique()}")
    print(f"games covered:   {len(games)}")
    print(f"records:         {len(frame)}")
    print(f"provenance:      {frame.provenance.value_counts().to_dict()}")
    print(f"status mix:      {frame.status.value_counts().to_dict()}")

    # What the pre-tipoff state would have added on top of the submission —
    # the late scratches this methodology deliberately does NOT gate on.
    sub = frame[frame.status.astype(str).str.lower().isin(REMOVING)]
    tip = frame[frame.status_tipoff.astype(str).str.lower().isin(REMOVING)]
    sub_keys = set(zip(sub.gameday, sub.matchup, sub.player_report))
    tip_keys = set(zip(tip.gameday, tip.matchup, tip.player_report))
    late = tip_keys - sub_keys
    lifted = sub_keys - tip_keys
    late_games = {k[:2] for k in late}
    print(f"\nout at submission:   {len(sub_keys)} player-games")
    print(f"out at tip-off:      {len(tip_keys)} player-games")
    print(f"late scratches (out ONLY at tip-off): {len(late)} across "
          f"{len(late_games)} games")
    print(f"cleared after submission (out only AT submission): {len(lifted)}")
    total = len(games)
    if total:
        print(f"games with >=1 late scratch: {len(late_games)}/{total} "
              f"({100 * len(late_games) / total:.1f}%) — this is the "
              f"information the slate would not have had, excluded by design")


def main(start: date, end: date, chunk_days: int = CHUNK_DAYS) -> None:
    # A parser-less backfill fetches every PDF and records NOTHING: the
    # day comes back empty, the shard still lands, and the consolidated
    # archive shrinks without anyone deciding it should.
    ir.require_pdf_parser()
    print(f"backfilling game-day submissions {start} .. {end} "
          f"({chunk_days}-day chunks)")
    expected, schedule = load_schedule(start, end)
    print(f"expected game days in range: {len(expected)} "
          f"({len(schedule)} with schedule times)")
    pending = [(a, b) for a, b in chunks(start, end, chunk_days)
               if not shard_path(SHARD_DIR, a, b).exists()]
    print(f"chunks: {len(list(chunks(start, end, chunk_days)))} total, "
          f"{len(pending)} pending")
    for a, b in pending:
        print(f"chunk {a} .. {b}")
        process_chunk(a, b, chunk_days, expected=expected, schedule=schedule)
    frame = consolidate()
    report(frame)
    print(f"\nhttp 403s this run: {ir.http_403_count} "
          f"(high counts mean the host rate-limited and coverage must "
          f"be re-checked)")


if __name__ == "__main__":
    a = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2024, 1, 1)
    b = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2026, 6, 13)
    main(a, b)
