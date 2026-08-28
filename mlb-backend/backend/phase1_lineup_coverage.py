"""Phase 1 — proof of coverage for the lineup-delta feature (READ-ONLY).

Do NOT build the feature. This probe answers three questions:

(a) HISTORICAL: does the StatsAPI live feed's
    ``liveData.boxscore.teams.{home,away}.battingOrder`` carry the ACTUAL
    starting 9 for decided games, and how often / what failure modes?
(b) SLATE: what does the ESPN scoreboard path (the slate builder's game_id
    source) expose pre-game for today's games — is there any lineup at all,
    and is it a "probable" vs "actual" distinction?
(c) CROSS-CHECK: do battingOrder player IDs (MLB IDs) resolve to batter IDs
    in the Statcast pbp (pybaseball path — the pipeline's own pbp source)?
    No pbp artifact exists in this workspace, so this is measured on a small
    Statcast subset for sample games.

Pacing mirrors fetch_statsapi_weather: pause_sec=0.4 between live-feed
requests, one retry with pause*2 backoff. Progress saves incrementally to
data_delivery/lineup_probe_<sha>.json and resumes (pass --limit to chunk the
sample across invocations — the sandbox command timeout is ~180s).

Usage:
    python phase1_lineup_coverage.py --sample-size 200 --limit 70
    python phase1_lineup_coverage.py --limit 70      # resumes
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time as _time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

from config import DATA_DELIVERY_DIR, RANDOM_SEED  # noqa: E402

STATSAPI_GAME_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
    "?dates={ymd}"
)
PAUSE_SEC = 0.4
SLATE_DATE = date(2026, 8, 25)  # latest slate: day after the CSV's last game


def head_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=_BACKEND_DIR.parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


# ── Pure parser (testable, no network) ───────────────────────────────────────

def parse_batting_orders(feed: dict) -> dict:
    """Extract per-side batting orders from a StatsAPI live-feed payload.

    Returns {'home': list[int], 'away': list[int], 'home_players': dict,
    'away_players': dict} where each order is the raw battingOrder list
    (empty list when the key is missing/absent — NEVER fabricated) and the
    players dicts map MLB ID -> display name from the same feed's boxscore
    (for ID-resolution checks, without extra requests).
    """
    box = ((feed or {}).get("liveData") or {}).get("boxscore") or {}
    teams = box.get("teams") or {}
    out: dict = {"home": [], "away": [], "home_players": {}, "away_players": {}}
    for side in ("home", "away"):
        t = teams.get(side) or {}
        bo = t.get("battingOrder")
        if isinstance(bo, list):
            ids = [int(x) if isinstance(x, (int, str)) and str(x).isdigit()
                   else None for x in bo]
            out[side] = [i for i in ids if i is not None]
        players = {}
        for key, p in (t.get("players") or {}).items():
            person = (p or {}).get("person") or {}
            pid = person.get("id")
            if pid is None:
                # Fallback: parse the dict key — real feeds key players by
                # "ID649966"; some shapes use "649966:P" or plain "649966".
                raw = key.split(":")[0]
                if raw.startswith("ID"):
                    raw = raw[2:]
                if raw.isdigit():
                    pid = int(raw)
            if pid is not None:
                try:
                    players[int(pid)] = person.get("fullName")
                except (TypeError, ValueError):
                    continue
        out[f"{side}_players"] = players
    return out


def classify_order(order: list[int]) -> tuple[str, int]:
    """(label, n). Labels: complete_9 / short_n / empty_array / over_9."""
    n = len(order)
    if n == 9:
        return "complete_9", n
    if n > 9:
        return "over_9", n
    if n == 0:
        return "empty_array", 0
    return "short_n", n


def classify_game(feed: dict, game_pk: int) -> dict:
    """Full per-game classification (failure modes named, never papered over)."""
    gd = (feed or {}).get("gameData") or {}
    state = (gd.get("status") or {}).get("abstractGameState") or "unknown"
    teams = (gd.get("teams") or {})
    parsed = parse_batting_orders(feed)
    row = {
        "game_pk": int(game_pk),
        "state": state,
        "home_team": (teams.get("home") or {}).get("abbreviation"),
        "away_team": (teams.get("away") or {}).get("abbreviation"),
    }
    if "boxscore" not in ((feed or {}).get("liveData") or {}):
        row["failure"] = "null_boxscore"
        row["home"] = ("missing", 0)
        row["away"] = ("missing", 0)
        return row
    for side in ("home", "away"):
        label, n = classify_order(parsed[side])
        row[side] = (label, n)
    # ID resolution: do battingOrder IDs exist in the same feed's players dict?
    res = {"home": 0, "away": 0}
    for side in ("home", "away"):
        ids = parsed[side]
        if ids:
            res[side] = sum(1 for i in ids if i in parsed[f"{side}_players"])
    row["order_ids_resolved"] = res
    return row


# ── Sampling ────────────────────────────────────────────────────────────────

def sample_decided_games(games: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    rng = np.random.RandomState(RANDOM_SEED)
    out = []
    for year in sorted(games["game_date"].dt.year.unique()):
        sub = games[games["game_date"].dt.year == year]
        k = max(1, round(n * len(sub) / len(games)))
        out.append(sub.sample(k, random_state=rng))
    s = pd.concat(out).drop_duplicates(subset=["game_pk"])
    if len(s) > n:
        s = s.sample(n, random_state=rng)
    return s.reset_index(drop=True)


# ── Paced live-feed fetch (mirrors fetch_statsapi_weather pacing) ───────────

def fetch_feed(game_pk: int, pause_sec: float = PAUSE_SEC) -> tuple[dict | None, str]:
    import requests
    last_err = ""
    for attempt in (1, 2):
        try:
            resp = requests.get(
                STATSAPI_GAME_FEED_URL.format(pk=game_pk), timeout=15)
            if resp.status_code == 200:
                return resp.json(), ""
            last_err = f"http_{resp.status_code}"
        except requests.RequestException as exc:
            last_err = f"request_error: {exc}"
        if attempt == 1:
            _time.sleep(pause_sec * 2)
    return None, last_err


# ── ESPN slate probe ────────────────────────────────────────────────────────

def espn_slate_probe(target: date) -> dict:
    """What does the ESPN scoreboard event expose for today's games?"""
    import requests
    url = ESPN_SCOREBOARD_URL.format(ymd=target.strftime("%Y%m%d"))
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        events = resp.json().get("events", [])
    except Exception as exc:
        return {"error": str(exc), "n_games": 0}
    games = []
    lineup_fields_seen: set[str] = set()
    for ev in events:
        comp = ev.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {})
        row = {
            "game_id": ev.get("id"),
            "state": status.get("state"),
            "detail": status.get("detail"),
            "start_utc": ev.get("date"),
            "probable_pitchers": sum(
                1 for c in comp.get("competitors", [])
                if c.get("probablePitcher")),
        }
        # What lineup-ish fields exist in the event payload at all?
        for c in comp.get("competitors", []):
            t = c.get("team", {})
            row[f"roster_{c.get('homeAway')}"] = bool(c.get("roster"))
            row[f"lineup_{c.get('homeAway')}"] = bool(c.get("lineup"))
            lineup_fields_seen.update(
                k for k in c.keys() if "line" in k.lower() or "roster" in k.lower())
        row["event_top_keys"] = sorted(ev.keys())
        games.append(row)
    return {"n_games": len(games), "lineup_fields_seen": sorted(lineup_fields_seen),
            "games": games}


# ── pbp batter-ID cross-check (small Statcast subset) ───────────────────────

def pbp_id_match(feed_pks: list[int], pause_sec: float = 0.6) -> dict:
    """Match battingOrder IDs vs Statcast pbp batter IDs for the sample games.

    Statcast is fetched by DATE (pybaseball's only granularity), so the
    sample's games are grouped by date; every game on those dates comes
    along for the ride (extra coverage is a bonus, not noise).
    """
    from pybaseball import statcast
    games_csv = DATA_DELIVERY_DIR / "game_level_features.csv"
    g = pd.read_csv(games_csv, usecols=["game_pk", "game_date"])
    g["game_date"] = pd.to_datetime(g["game_date"])
    sub = g[g["game_pk"].isin(feed_pks)]
    dates = sorted(sub["game_date"].dt.date.unique())
    per_game = []
    total_matched = total_ids = 0
    for d in dates:
        day = statcast(d.isoformat(), d.isoformat())
        _time.sleep(pause_sec)
        if day is None or day.empty:
            per_game.append({"date": d.isoformat(), "pbp_rows": 0})
            continue
        day = day[day["game_pk"].isin(feed_pks)]
        for pk in sorted(day["game_pk"].unique()):
            day_ids = set(day.loc[day["game_pk"] == pk, "batter"].dropna().astype(int))
            # Fresh process: the feeds were fetched in earlier invocations, so
            # re-fetch each one (paced) for the ID cross-check.
            feed, err = fetch_feed(pk)
            if feed is None:
                per_game.append({"game_pk": int(pk), "date": d.isoformat(),
                                 "fetch_error": err})
                continue
            bo = parse_batting_orders(feed)
            order_ids = set(bo["home"]) | set(bo["away"])
            matched = len(order_ids & day_ids) if order_ids else None
            if order_ids:
                total_matched += matched
                total_ids += len(order_ids)
            per_game.append({"game_pk": int(pk), "date": d.isoformat(),
                             "order_ids": len(order_ids),
                             "pbp_batter_ids": len(day_ids), "matched": matched})
    return {"games": per_game,
            "overall_match_rate": (round(total_matched / total_ids, 4)
                                   if total_ids else None),
            "n_order_ids": total_ids, "n_matched": total_matched}


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-size", type=int, default=200)
    ap.add_argument("--limit", type=int, default=70,
                    help="max games to fetch this invocation (resume-safe)")
    ap.add_argument("--slate-date", type=str, default=SLATE_DATE.isoformat())
    ap.add_argument("--pbp-check", action="store_true",
                    help="run the small Statcast pbp batter-ID cross-check")
    args = ap.parse_args()

    sha = head_sha()
    out_path = DATA_DELIVERY_DIR / f"lineup_probe_{sha[:12]}.json"
    results: dict = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())

    if "games" not in results:
        games_csv = DATA_DELIVERY_DIR / "game_level_features.csv"
        g = pd.read_csv(games_csv, usecols=["game_pk", "game_date",
                                            "home_team", "away_team"])
        g["game_date"] = pd.to_datetime(g["game_date"])
        decided = g.dropna(subset=["game_pk"]).copy()
        decided = decided[decided["game_pk"].notna()]
        sample = sample_decided_games(decided, args.sample_size)
        print(f"sample: {len(sample)} decided games "
              f"({sample['game_date'].dt.year.value_counts().to_dict()})")
        results.update({
            "schema": "phase1-lineup-coverage/v1",
            "commit_sha": sha,
            "sample_size": args.sample_size,
            "sample": [{"game_pk": int(r.game_pk),
                        "game_date": str(r.game_date.date()),
                        "home": r.home_team, "away": r.away_team}
                       for r in sample.itertuples()],
            "games": {},
        })
        out_path.write_text(json.dumps(results, indent=2) + "\n")

    todo = [s for s in results["sample"]
            if str(s["game_pk"]) not in results["games"]]
    print(f"pending: {len(todo)} of {len(results['sample'])} "
          f"(fetched {len(results['games'])})")
    for s in todo[: args.limit]:
        pk = int(s["game_pk"])
        feed, err = fetch_feed(pk)
        if feed is None:
            results["games"][str(pk)] = {"game_pk": pk, "failure": err}
        else:
            row = classify_game(feed, pk)
            if err:
                row["warn"] = err
            results["games"][str(pk)] = row
        out_path.write_text(json.dumps(results, indent=2) + "\n")

    done = [v for v in results["games"].values() if v.get("state")]
    if done:
        labels = [v.get("home", ("missing", 0))[0] for v in done]
        print(f"fetched {len(done)}: "
              f"complete_9={labels.count('complete_9')} "
              f"empty={labels.count('empty_array')} "
              f"short={labels.count('short_n')} "
              f"over9={labels.count('over_9')} "
              f"missing={sum(1 for v in done if v.get('home', ('', 0))[0] == 'missing')}")

    if "slate" not in results:
        results["slate"] = espn_slate_probe(
            date.fromisoformat(args.slate_date))
        out_path.write_text(json.dumps(results, indent=2) + "\n")
        s = results["slate"]
        print(f"slate {args.slate_date}: {s.get('n_games')} games, "
              f"lineup fields in payload: {s.get('lineup_fields_seen')}")

    if args.pbp_check and "pbp_match" not in results:
        feed_pks = [int(v.get("game_pk", k)) for k, v in results["games"].items()
                    if v.get("state") == "Final"]
        results["pbp_match"] = pbp_id_match(feed_pks[:6])
        out_path.write_text(json.dumps(results, indent=2) + "\n")
        pm = results["pbp_match"]
        print(f"pbp match: {pm.get('n_matched')}/{pm.get('n_order_ids')} "
              f"({pm.get('overall_match_rate')})")

    # pacing estimate for the full backfill
    n_total = 4451
    per_fetch = PAUSE_SEC + 0.35  # pause + typical request latency
    results["backfill_estimate_sec"] = round(n_total * per_fetch)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"full backfill ({n_total} feeds): ~{n_total * per_fetch / 60:.0f} min "
          f"at {PAUSE_SEC}s pacing (~{round(n_total * per_fetch / 128)} "
          f"batches of 365 at the roof-fetch budget)")
    print(f"probe written: {out_path}")


if __name__ == "__main__":
    main()
