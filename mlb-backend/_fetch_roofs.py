"""Fetch StatsAPI roof conditions; top up data_delivery/statsapi_roof_cache.json.

The cache maps game_pk -> "open" | "closed" | null for RETRACTABLE-ROOF home
games only (fixed domes never need it). It feeds features.refine_dome_game_level,
which resolves dome_is_neutral_game per game and falls back LOUDLY to the venue
flag when a retractable game's state is still unknown.

Rate safety / idempotency:
  * one StatsAPI call per game, pause_sec=0.35 between calls, batches of 40,
    hard ~128 s budget per invocation;
  * games already carrying "open"/"closed" in the cache are NEVER re-fetched —
    each run only appends NEW states, so repeated runs converge without
    duplicates (the merge is dict-update keyed by int(game_pk));
  * null entries ARE retried (a retractable park reporting real outdoor
    conditions without a roof statement classifies as "open").

HOW TO RUN A TOP-UP (terminal-written files are reverted by workspace sync,
so persist via the write_file tool):
  1. python3 mlb-backend/_fetch_roofs.py \
         mlb-backend/data_delivery/statsapi_roof_cache.json > /tmp/roofs.json
     (argv optional; omitted -> starts from the empty cache)
  2. Copy /tmp/roofs.json's "merged" object into
     mlb-backend/data_delivery/statsapi_roof_cache.json with write_file.
  3. Repeat until the printed "remaining" reaches 0 (~950 unknowns need
     several runs at ~350 fetches/run).

Output: ONE compact JSON line
  {"merged": {game_pk: "open"|"closed"|null, ...}, "fetched_this_run": N,
   "remaining": M, "open": n_open, "closed": n_closed, "unknown": n_unknown}
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
from features import RETRACTABLE_ROOF_TEAMS, load_roof_cache, roof_state_from_condition  # noqa: E402
from results import fetch_statsapi_weather  # noqa: E402

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data_delivery" / "game_level_features.csv"
FETCH_BUDGET_SEC = 128
BATCH_SIZE = 40


def retractable_game_pks(csv_path: Path = CSV) -> list[int]:
    """All game_pks hosted by a retractable-roof team, in file order."""
    df = pd.read_csv(csv_path, usecols=["game_pk", "home_team"])
    ht = df["home_team"].astype(str).str.upper().str.strip()
    return [int(pk) for pk in
            df.loc[ht.isin(RETRACTABLE_ROOF_TEAMS), "game_pk"].astype(int)]


def merge_fetch_results(cache: dict, results: dict) -> tuple[dict, int]:
    """Merge one batch of fetch_statsapi_weather results into the cache.

    Pure (no I/O): returns (updated_cache, n_resolved_this_batch). Dict-update
    keyed by int game_pk cannot produce duplicate keys; re-running with the
    same results is a no-op.
    """
    updated = dict(cache)
    resolved = 0
    for raw_pk, wx in results.items():
        pk = int(raw_pk)
        if not wx:
            updated[pk] = None
            continue
        state = roof_state_from_condition(wx.get("condition"))
        if state is None and (wx.get("temp_f") is not None
                              or wx.get("wind_mph") is not None):
            state = "open"   # real outdoor observation, no roof statement
        updated[pk] = state
        resolved += 1
    return updated, resolved


def main(argv: list[str]) -> str:
    cache = {}
    if len(argv) > 1 and argv[1] not in ("-", ""):
        cache = load_roof_cache(Path(argv[1]))  # int keys, deduped, loud on dups

    pks = retractable_game_pks()
    todo = [pk for pk in pks if not cache.get(pk)]

    t0 = time.time()
    fetched_total = 0
    i = 0
    while todo and time.time() - t0 < FETCH_BUDGET_SEC:
        batch = todo[i:i + BATCH_SIZE]
        res = fetch_statsapi_weather(batch, pause_sec=0.35)
        cache, resolved = merge_fetch_results(cache, res)
        fetched_total += len(batch)
        i += len(batch)

    states = list(cache.values())
    return json.dumps({
        "merged": cache,
        "fetched_this_run": fetched_total,
        "remaining": max(len(todo) - fetched_total, 0),
        "open": states.count("open"),
        "closed": states.count("closed"),
        "unknown": sum(1 for s in states if s is None),
    }, separators=(",", ":"))


if __name__ == "__main__":
    print(main(sys.argv))
