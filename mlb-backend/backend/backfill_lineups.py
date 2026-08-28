"""Lineups backfill — battingOrder for every decided game 2025–2026.

Emits data_delivery/lineups.parquet: game_pk, game_date, home_team,
away_team, home_order (9 MLB IDs), away_order, complete_home, complete_away,
state. Incremental + resumable (already-fetched pks are skipped), paced at
~2.4 fetches/sec (pause 0.15s, one retry) — under the roof fetcher's proven
~2.85/s. Reuses the Phase 1 parser (tested in test_phase1_lineup_probe.py).

Usage:
    python backfill_lineups.py --limit 400     # chunk across invocations
    python backfill_lineups.py --limit 400     # resumes
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path

import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

from config import DATA_DELIVERY_DIR  # noqa: E402
from phase1_lineup_coverage import fetch_feed, parse_batting_orders  # noqa: E402

PAUSE_SEC = 0.15


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    csv_path = args.csv or (DATA_DELIVERY_DIR / "game_level_features.csv")
    games = pd.read_csv(csv_path, usecols=["game_pk", "game_date",
                                           "home_team", "away_team"])
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["game_pk"])
    games["game_pk"] = games["game_pk"].astype(int)
    # Decided games only: home_win present.
    full = pd.read_csv(csv_path, usecols=["game_pk", "home_win"])
    decided_pks = set(full.dropna(subset=["home_win"])["game_pk"].astype(int))
    games = games[games["game_pk"].isin(decided_pks)].reset_index(drop=True)
    print(f"decided games to backfill: {len(games)}")

    out_path = DATA_DELIVERY_DIR / "lineups.parquet"
    done: set[int] = set()
    if out_path.exists():
        done = set(pd.read_parquet(out_path)["game_pk"].astype(int))
    todo = games[~games["game_pk"].isin(done)]
    print(f"pending: {len(todo)} (done {len(done)})")

    rows = []
    for r in todo.head(args.limit).itertuples():
        pk = int(r.game_pk)
        feed, err = fetch_feed(pk, pause_sec=PAUSE_SEC)
        if feed is None:
            rows.append({"game_pk": pk, "game_date": r.game_date,
                         "home_team": r.home_team, "away_team": r.away_team,
                         "home_order": None, "away_order": None,
                         "complete_home": False, "complete_away": False,
                         "state": f"fetch_error: {err}"})
            continue
        gd = feed.get("gameData") or {}
        state = (gd.get("status") or {}).get("abstractGameState") or "unknown"
        p = parse_batting_orders(feed)
        rows.append({
            "game_pk": pk,
            "game_date": r.game_date,
            "home_team": (gd.get("teams") or {}).get("home", {}).get("abbreviation")
                         or r.home_team,
            "away_team": (gd.get("teams") or {}).get("away", {}).get("abbreviation")
                         or r.away_team,
            "home_order": p["home"] if len(p["home"]) == 9 else None,
            "away_order": p["away"] if len(p["away"]) == 9 else None,
            "complete_home": len(p["home"]) == 9,
            "complete_away": len(p["away"]) == 9,
            "state": state,
        })
        if len(rows) % 50 == 0:
            print(f"  ...{len(rows)} this chunk", flush=True)

    df_new = pd.DataFrame(rows)
    if not df_new.empty:
        prev = pd.read_parquet(out_path) if out_path.exists() else None
        df_all = pd.concat([prev, df_new], ignore_index=True) if prev is not None \
            else df_new
        df_all = df_all.drop_duplicates(subset=["game_pk"], keep="last")
        df_all.to_parquet(out_path, index=False)
    print(f"lineups.parquet now: {len(pd.read_parquet(out_path))} games")


if __name__ == "__main__":
    main()
