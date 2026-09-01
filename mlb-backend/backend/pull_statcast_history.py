"""pull_statcast_history.py — incremental full-history Savant puller.

Stands in for ingestion.pull_statcast when pybaseball is unavailable
(this box): chunked, resume-aware, dedup-on-pitch-identity semantics,
talking to the Savant CSV endpoint directly with retries/backoff.

Savant CSVs are capped at 25,000 rows per query — a 7-day chunk in a busy
week silently truncates. This puller therefore uses SMALL chunks (default
3 days) and any response at exactly the cap is split in half and re-fetched
until under the cap, so games are never silently dropped.

Usage
-----
    python pull_statcast_history.py --start 2023-09-01 --end 2026-08-30 \
        --out C:/tmp/pitches_full/pitches.parquet --chunk-days 3

State: the out parquet doubles as the cache. On resume the puller reads its
date bounds, SKIPS fully-covered chunks, and ALWAYS re-pulls the trailing
window so partially-crawled finals get refreshed (dedup keeps the newest
pitch copy). Writes a .meta.json sidecar with bounds + chunk list.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = (
    "https://baseballsavant.mlb.com/statcast_search/csv?all=true"
    "&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones="
    "&hfGT=R%7CPO%7CS%7C&hfC=&hfSit="
    "&player_type=pitcher&hfOuts=&opponent_concept=&hfTeam=&home_away="
    "&hfRO=&hfFlag=&hfPull=&hfInfield=&hfInn="
    "&min_pitches=0&min_results=0&group_by=name"
    "&sort_col=pitches&player_event_sort=h_launch_speed&sort_order=desc"
    "&min_pas=0&type=details"
    # NOTE: NO hfSea filter — a season filter silently returns EMPTY for
    # chunks of other seasons (verified live). Date bounds scope the query.
)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
DEDUP = ["game_pk", "at_bat_number", "pitch_number"]
ROW_CAP = 25_000


def fetch_range(start: date, end: date, depth: int = 0) -> pd.DataFrame | None:
    """Fetch [start, end]; if the response sits at the 25k cap, split the
    range in half and refetch so no game is truncated by the export cap."""
    lo, hi = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    url = f"{BASE}&game_date_gt={lo}&game_date_lt={hi}"
    for attempt in range(4):
        try:
            t0 = time.time()
            r = requests.get(url, timeout=180, headers=HEADERS)
            if r.status_code != 200:
                time.sleep(6 * (attempt + 1))
                continue
            txt = r.content.decode("utf-8-sig", errors="replace")
            first = txt.splitlines()[0:1]
            if not first or "game_pk" not in (first[0] or ""):
                time.sleep(6 * (attempt + 1))
                continue
            df = pd.read_csv(io.StringIO(txt))
            if df is None or df.empty:
                time.sleep(6 * (attempt + 1))
                continue
            if len(df) >= ROW_CAP and start != end:
                mid = start + (end - start) // 2
                print(f"    {lo}..{hi} hit the {ROW_CAP}-row cap — splitting "
                      f"({time.time()-t0:.0f}s)", flush=True)
                a = fetch_range(start, mid, depth + 1)
                b = fetch_range(mid + timedelta(days=1), end, depth + 1)
                parts = [x for x in (a, b) if x is not None and len(x)]
                if not parts:
                    return None
                if len(parts) == 1:
                    return parts[0]
                return pd.concat(parts, ignore_index=True)
            return df
        except Exception:
            time.sleep(6 * (attempt + 1))
    return None


def cache_bounds(path: Path) -> tuple[date | None, date | None]:
    try:
        d = pd.read_parquet(path, columns=["game_date"])
        d["game_date"] = pd.to_datetime(d["game_date"])
        return d["game_date"].min().date(), d["game_date"].max().date()
    except Exception:
        return None, None


def chunk_dir(out: Path) -> Path:
    return out.parent / f"{out.stem}_chunks"


def save_chunk(cdir: Path, start: date, end: date, df: pd.DataFrame) -> int:
    """Write one chunk parquet (cheap append-only). Returns cache row count
    from the sidecar meta (approximate; exact dedup happens at final merge)."""
    cdir.mkdir(parents=True, exist_ok=True)
    f = cdir / f"{start.isoformat()}_{end.isoformat()}.parquet"
    df.to_parquet(f, index=False)
    return int(df.groupby("game_pk", dropna=False).ngroups)


def final_merge(out: Path) -> int:
    cdir = chunk_dir(out)
    files = sorted(cdir.glob("*.parquet")) if cdir.exists() else []
    if not files:
        return 0
    frames = [pd.read_parquet(f) for f in files]
    out_ = pd.concat(frames, ignore_index=True)
    d0 = len(out_)
    out_ = out_.drop_duplicates(subset=DEDUP, keep="last")
    out.parent.mkdir(parents=True, exist_ok=True)
    out_.to_parquet(out, index=False)
    return len(out_)


def pull(start: date, end: date, chunk_days: int, pause: float,
         out: Path) -> None:
    cdir = chunk_dir(out)
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        f = cdir / f"{cursor.isoformat()}_{chunk_end.isoformat()}.parquet"
        if f.exists() and (date.today() - chunk_end).days > 3:
            print(f"[pull] {cursor} → {chunk_end}: cached, skip", flush=True)
            cursor = chunk_end + timedelta(days=1)
            continue
        df = fetch_range(cursor, chunk_end)
        if df is not None and len(df):
            save_chunk(cdir, cursor, chunk_end, df)
            print(f"[pull] {cursor} → {chunk_end}: {len(df):,} pitches",
                  flush=True)
        else:
            print(f"[pull] {cursor} → {chunk_end}: EMPTY", flush=True)
        time.sleep(pause)
        cursor = chunk_end + timedelta(days=1)


def _split_range(start: date, end: date, workers: int) -> list[tuple[date, date]]:
    """Divide [start, end] into `workers` contiguous sub-ranges."""
    days = (end - start).days + 1
    size = max(1, (days + workers - 1) // workers)
    ranges = []
    cur = start
    while cur <= end:
        r_end = min(cur + timedelta(days=size - 1), end)
        ranges.append((cur, r_end))
        cur = r_end + timedelta(days=1)
    return ranges


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-days", type=int, default=7)
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes (each owns a sub-range)")
    args = ap.parse_args()

    out = Path(args.out)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    # The per-chunk parquet files in <out>_chunks/ ARE the cache: pull()
    # skips finished chunks (older than 3 days), so a restart resumes where
    # it left off without any merged-file state. Workers partition the range
    # and each pulls its own sub-range concurrently.
    print(f"[pull] pulling {start} → {end} (chunk {args.chunk_days}d, "
          f"workers {args.workers})", flush=True)
    if args.workers > 1:
        from multiprocessing import Process

        procs = []
        for r_lo, r_hi in _split_range(start, end, args.workers):
            p = Process(target=pull, args=(r_lo, r_hi, args.chunk_days,
                                           args.pause, out), daemon=True)
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        pull(start, end, args.chunk_days, args.pause, out)
    n = final_merge(out)
    lo2, hi2 = cache_bounds(out)
    meta = {"start": str(start), "end": str(end), "cache_lo": str(lo2),
            "cache_hi": str(hi2), "rows_after_dedup": n}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[pull] DONE — cache covers {lo2} → {hi2} ({n:,} rows)", flush=True)


if __name__ == "__main__":
    main()