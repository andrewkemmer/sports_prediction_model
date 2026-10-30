"""Full-history referee crew pull (F5): all cached games, resumable.

4 workers, ~0.5s gate per worker, 3 attempts with exponential backoff.
Checkpoint parquet every 100 games; skips games already pulled.
Output: nba-backend/run_diagnostics/ref_crews_full.parquet
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from spike_ref_crews import get_summary, officials_of

CACHE = os.path.expanduser('~/.cache/sports_prediction_model/nba')
OUT = 'nba-backend/run_diagnostics/ref_crews_full.parquet'
CKPT_EVERY = 25
WORKERS = 1
GATE_SECONDS = 0.4


def get_summary_v3(stats_game_id, timeout=12):
    import json as _json
    import urllib.parse as _up
    import urllib.request as _ur
    from spike_ref_crews import STATS_HEADERS as _H
    url = ('https://stats.nba.com/stats/boxscoresummaryv3?' +
           _up.urlencode({'GameID': stats_game_id}))
    req = _ur.Request(url, headers=_H)
    with _ur.urlopen(req, timeout=timeout) as resp:
        payload = _json.loads(resp.read().decode('utf-8'))
    return [{'FIRST_NAME': o.get('firstName'), 'LAST_NAME': o.get('familyName')}
            for o in payload.get('boxScoreSummary', {}).get('officials', []) or []]


def fetch(gid):
    time.sleep(GATE_SECONDS + (hash(gid) % 10) / 10.0)  # desync workers
    last = None
    playoff = gid.startswith('004')
    # Regular season: v3 is fast and complete (v2's Officials table is
    # empty for 2025). Playoffs: v2 (v3 carries no playoff crews).
    for attempt in range(2):
        try:
            if playoff:
                off = officials_of(get_summary(gid, timeout=30))
                if not off:
                    off = get_summary_v3(gid)
            else:
                off = get_summary_v3(gid)
            refs = ["%s %s" % (o.get('FIRST_NAME'), o.get('LAST_NAME')) for o in off]
            return gid, refs, None
        except Exception as exc:  # noqa: BLE001 - tallied, retried
            last = "%s: %s" % (type(exc).__name__, str(exc)[:100])
            time.sleep(0.6 * (2 ** attempt))
        time.sleep(GATE_SECONDS)
    return gid, [], last


def main():
    games = pd.read_parquet(os.path.join(CACHE, 'games.parquet'))
    gids = games['game_id'].astype(str).tolist()
    done = {}
    if os.path.exists(OUT):
        prev = pd.read_parquet(OUT)
        done = dict(zip(prev['game_id'].astype(str),
                        prev[['ref1', 'ref2', 'ref3']].values.tolist()))
        print("resuming: %d/%d already pulled" % (len(done), len(gids)), flush=True)
    todo = [g for g in gids if g not in done]
    # Playoffs (004) fail slow; pull every regular-season game first so a
    # stall never blocks the bulk. Playoffs retry at the end, unthrottled.
    todo = sorted(todo, key=lambda g: (g.startswith('004'), g))
    print("pulling %d games with %d workers" % (len(todo), WORKERS), flush=True)
    misses = []
    t0 = time.time()
    finished = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch, g): g for g in todo}
        for fut in as_completed(futs):
            gid, refs, err = fut.result()
            if refs:
                while len(refs) < 3:
                    refs.append(None)
                done[gid] = refs[:3]
            else:
                misses.append((gid, err))
            finished += 1
            if finished % CKPT_EVERY == 0 or finished == len(todo):
                rows = [{"game_id": k, "ref1": v[0], "ref2": v[1], "ref3": v[2]}
                        for k, v in done.items()]
                pd.DataFrame(rows).to_parquet(OUT, index=False)
                el = time.time() - t0
                print("...%d/%d pulled, %d misses (%.0fs, %.1f/s)"
                      % (len(done), len(gids), len(misses), el,
                         finished / max(el, 1)), flush=True)
    print("DONE: %d/%d pulled, %d misses" % (len(done), len(gids), len(misses)), flush=True)
    for gid, err in misses[:20]:
        print("  miss %s: %s" % (gid, err), flush=True)


if __name__ == "__main__":
    import sys
    _log = open('nba-backend/run_diagnostics/pull_ref_crews_full.log', 'a', buffering=1)
    sys.stdout = _log
    sys.stderr = _log
    try:
        main()
    finally:
        _log.close()
