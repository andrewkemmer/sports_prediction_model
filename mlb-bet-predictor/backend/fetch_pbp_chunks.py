"""Fetch Statcast pbp in ~14-day windows (pybaseball, the pipeline's source).

Saves lean per-window parquet files to data_delivery/pbp_chunks/ — only the
columns the batter-wOBA builder needs (game_pk, game_date, teams,
inning_topbot, batter, events). Resumable: windows already on disk are
skipped. Chunking keeps pybaseball's peak RSS (~500–800MB for 20 days)
bounded, and lets the work fit the sandbox command timeout.

WINDOWS covers the CSV's decided-game range 2025-03-18 → 2026-08-24.
Usage:
    python fetch_pbp_chunks.py               # all windows
    python fetch_pbp_chunks.py --limit 4     # chunk across invocations
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

from config import DATA_DELIVERY_DIR  # noqa: E402

CHUNK_DAYS = 14
KEEP_COLS = ["game_pk", "game_date", "home_team", "away_team",
             "inning_topbot", "batter", "events", "game_type"]

WINDOWS: list[tuple[date, date]] = []
cur = date(2025, 3, 18)
end = date(2026, 8, 24)
while cur <= end:
    stop = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
    WINDOWS.append((cur, stop))
    cur = stop + timedelta(days=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=99)
    args = ap.parse_args()

    out_dir = DATA_DELIVERY_DIR / "pbp_chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    done = {p.name for p in out_dir.glob("pbp_*.parquet")}
    todo = [(s, e) for s, e in WINDOWS
            if f"pbp_{s.isoformat()}_{e.isoformat()}.parquet" not in done]
    print(f"windows: {len(WINDOWS)} total, {len(todo)} pending")
    t0 = time.time()
    for s, e in todo[: args.limit]:
        from pybaseball import statcast
        df = statcast(s.isoformat(), e.isoformat())
        out = out_dir / f"pbp_{s.isoformat()}_{e.isoformat()}.parquet"
        if df is None or df.empty:
            # Write an empty file so the window is marked done (off-season
            # windows must not be re-fetched every invocation).
            pd.DataFrame(columns=KEEP_COLS).to_parquet(out, index=False)
            print(f"  {s}..{e}: EMPTY (marked done)", flush=True)
            continue
        lean = df[[c for c in KEEP_COLS if c in df.columns]].copy()
        if "game_date" in lean.columns:
            lean["game_date"] = pd.to_datetime(lean["game_date"])
        lean.to_parquet(out, index=False)
        del df, lean
        gc.collect()
        print(f"  {s}..{e}: {len(pd.read_parquet(out))} rows "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)
    print(f"done this invocation; total elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
