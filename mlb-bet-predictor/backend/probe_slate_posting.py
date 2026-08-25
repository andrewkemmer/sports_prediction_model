"""Slate posting-curve probe — when do StatsAPI battingOrders actually post?

Each invocation snapshots ONE slate at the CURRENT time and appends it to
data_delivery/slate_posting_curve.json. Because a slate's first pitches are
staggered (~22:40Z–01:45Z), a single snapshot yields games at several
different offsets; re-running later (interleaved with other work) pushes the
same games to lower offsets, tracing the real posting curve on our endpoint.

Per game: offset hours before first pitch (T−off), home/away battingOrder
lengths, team abbreviations. "Both" = 9+9, "home-only" = 9+0, "neither" = 0+0.

Usage:
    python probe_slate_posting.py --date 2026-08-25
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import requests

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

from config import DATA_DELIVERY_DIR  # noqa: E402

PAUSE_SEC = 0.15


def snapshot(target_date: str) -> dict:
    now = datetime.now(timezone.utc)
    sched = requests.get("https://statsapi.mlb.com/api/v1/schedule",
                         params={"sportId": 1, "date": target_date},
                         timeout=15).json()
    games = []
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            pk = g["gamePk"]
            start = datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00"))
            off_h = round((start - now).total_seconds() / 3600, 2)
            feed = requests.get(
                f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live",
                timeout=15).json()
            gd = feed.get("gameData") or {}
            box = feed.get("liveData", {}).get("boxscore") or {}
            bo_h = box.get("teams", {}).get("home", {}).get("battingOrder") or []
            bo_a = box.get("teams", {}).get("away", {}).get("battingOrder") or []
            games.append({
                "game_pk": pk,
                "offset_h": off_h,
                "home_bo": len(bo_h),
                "away_bo": len(bo_a),
                "home": (gd.get("teams") or {}).get("home", {}).get("abbreviation"),
                "away": (gd.get("teams") or {}).get("away", {}).get("abbreviation"),
                "start_utc": g["gameDate"],
            })
            _time.sleep(PAUSE_SEC)
    return {"ts_utc": now.isoformat(timespec="minutes"),
            "n_games": len(games), "games": games}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", type=str, default="2026-08-25")
    args = ap.parse_args()
    out_path = DATA_DELIVERY_DIR / "slate_posting_curve.json"
    data = json.loads(out_path.read_text()) if out_path.exists() else \
        {"schema": "slate-posting-curve/v1", "snapshots": []}
    snap = snapshot(args.date)
    data["snapshots"].append(snap)
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    both = sum(1 for g in snap["games"] if g["home_bo"] == 9 and g["away_bo"] == 9)
    home = sum(1 for g in snap["games"] if g["home_bo"] == 9 and g["away_bo"] == 0)
    none = sum(1 for g in snap["games"] if g["home_bo"] == 0 and g["away_bo"] == 0)
    print(f"snapshot {snap['ts_utc']}: {snap['n_games']} games — "
          f"both={both} home_only={home} neither={none}")
    print(f"offsets: " + ", ".join(f"{g['offset_h']}h:{g['home']}:{g['home_bo']}/{g['away_bo']}"
                                   for g in sorted(snap['games'], key=lambda x: x['offset_h'])))


if __name__ == "__main__":
    main()
