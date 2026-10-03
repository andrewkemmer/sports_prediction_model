"""Lineup backfill — battingOrder for every decided game 2024–2026.

Emits data_delivery/lineups.parquet: game_pk, game_date, home_team,
away_team, home_order (9 MLB IDs), away_order, complete_home, complete_away,
state. Incremental + resumable (already-COMPLETE pks are skipped; incomplete
/ fetch_error rows are retried every run), paced at ~2.4 fetches/sec
(pause 0.15s, one retry) — under the roof fetcher's proven ~2.85/s. Reuses
the Phase 1 parser (unit-tested historically; see git history for the
retired test suite).

TWO entry points (both wired into master_pipeline Phase 1.6):
    main()                     decided-game gaps (from the previous run's
                               game_level_features.csv) — post-game tier-1
                               membership for every finished game
    fetch_scheduled_lineups()  PRE-GAME capture: today's scheduled game_pks
                               via the StatsAPI schedule, posted nine only —
                               tonight's tier 1 while the feed still has it

lineups.parquet is membership tier 1 (tonight's nine): features.py resolves
lineup_effective from it, so a gap here degrades membership to the
projected nine (tier 2) — honest, but strictly worse information.

Usage:
    python backfill_lineups.py --limit 400     # chunk across invocations
    python backfill_lineups.py --limit 400     # resumes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

from config import DATA_DELIVERY_DIR  # noqa: E402
from phase1_lineup_coverage import fetch_feed, parse_batting_orders  # noqa: E402

PAUSE_SEC = 0.15
STATSAPI_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
LINEUPS_COLUMNS = ["game_pk", "game_date", "home_team", "away_team",
                   "home_order", "away_order", "complete_home",
                   "complete_away", "state"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Column order + dtypes for a frame about to be written to the cache."""
    for c in LINEUPS_COLUMNS:
        if c not in df.columns:
            df[c] = None
    out = df[LINEUPS_COLUMNS].copy()
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce")
    out = out.dropna(subset=["game_pk"])
    out["game_pk"] = out["game_pk"].astype(int)
    out["game_date"] = pd.to_datetime(out["game_date"])
    out["complete_home"] = out["complete_home"].fillna(False).astype(bool)
    out["complete_away"] = out["complete_away"].fillna(False).astype(bool)
    return out


def upsert_lineups(df_new: pd.DataFrame, out_path: Path | None = None) -> int:
    """Merge rows into lineups.parquet by game_pk (latest write wins).

    One protection: an INCOMPLETE row never downgrades an existing COMPLETE
    one — a pre-game partial post (away side not yet out) can't erase a
    nine we already know. Returns the number of rows written.
    """
    df_new = _normalize(df_new)
    if df_new.empty:
        return 0
    path = out_path or (DATA_DELIVERY_DIR / "lineups.parquet")
    prev = pd.read_parquet(path) if path.exists() else None
    if prev is not None and len(prev):
        prev = _normalize(prev)
        known = set(prev.loc[prev["complete_home"] & prev["complete_away"],
                             "game_pk"])
        df_new = df_new[~df_new["game_pk"].isin(known)
                        | (df_new["complete_home"] & df_new["complete_away"])]
        if df_new.empty:
            return 0
        df_all = pd.concat([prev, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all = df_all.drop_duplicates(subset=["game_pk"], keep="last")
    df_all.to_parquet(path, index=False)
    return len(df_new)


def complete_game_pks(path: Path | None = None) -> set[int]:
    """Game pks whose nine is KNOWN — the resume set for main().

    Incomplete / fetch_error rows intentionally stay OUT so every run
    retries them (a transient feed failure must not freeze a game at
    projected-only membership forever).
    """
    p = path or (DATA_DELIVERY_DIR / "lineups.parquet")
    if not p.exists():
        return set()
    try:
        df = pd.read_parquet(p)
    except Exception:  # unreadable cache: re-fetch everything
        return set()
    if not len(df):
        return set()
    df = _normalize(df)
    done = df.loc[df["complete_home"] & df["complete_away"], "game_pk"]
    return set(done.astype(int))


def order_row(game_pk: int, game_date, home_team: str, away_team: str,
              feed: dict | None, state: str | None = None) -> dict:
    """One lineups.parquet row from a StatsAPI live feed (pure — testable).

    complete_* is TRUE only on an exact 9-man order per side, mirroring the
    decided backfill; anything shorter is an honest incomplete row that the
    next run retries.
    """
    p = {"home": [], "away": []}
    if feed is not None:
        try:
            p = parse_batting_orders(feed)
        except Exception:
            p = {"home": [], "away": []}
    if state is None and feed is not None:
        gd = feed.get("gameData") or {}
        state = (gd.get("status") or {}).get("abstractGameState") or "unknown"
    home_order = list(p.get("home") or [])
    away_order = list(p.get("away") or [])
    ch, ca = len(home_order) == 9, len(away_order) == 9
    return {
        "game_pk": int(game_pk),
        "game_date": pd.Timestamp(game_date),
        "home_team": home_team,
        "away_team": away_team,
        # a side ships its order ONLY on an exact 9 — a short/over-9 list
        # is withheld (the shipped convention), never padded or truncated
        "home_order": home_order if ch else None,
        "away_order": away_order if ca else None,
        "complete_home": ch,
        "complete_away": ca,
        "state": state or "unknown",
    }


def fetch_scheduled_lineups(target_date, out_path: Path | None = None,
                            pause_sec: float = PAUSE_SEC) -> int:
    """PRE-GAME capture: posted nines for ``target_date``'s scheduled games.

    Resolves the day's game_pks from the StatsAPI schedule (the same
    endpoint results.py trusts), fetches each live feed paced like the
    decided backfill, and upserts every row — complete nines upgrade the
    cache to tier 1 immediately; unposted sides stay incomplete and are
    completed post-game by main(). Returns rows written. Never raises:
    a schedule/feed outage degrades membership to the projected nine.
    """
    import requests
    try:
        resp = requests.get(
            STATSAPI_SCHEDULE_URL,
            params={"sportId": 1, "startDate": pd.Timestamp(target_date).date().isoformat(),
                    "endDate": pd.Timestamp(target_date).date().isoformat()},
            timeout=20)
        resp.raise_for_status()
        games = (resp.json().get("dates") or [{}])[0].get("games") or []
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  lineup pre-game capture: schedule fetch failed ({e}); "
              f"membership degrades to projection")
        return 0
    rows = []
    for g in games:
        try:
            pk = int(g["gamePk"])
        except Exception:
            continue
        t = (g.get("teams") or {}).get("away") or {}
        h = (g.get("teams") or {}).get("home") or {}
        away = (t.get("team") or {}).get("abbreviation") or ""
        home = (h.get("team") or {}).get("abbreviation") or ""
        # fetch_feed paces itself (pause_sec between requests, one retry)
        feed, _err = fetch_feed(pk, pause_sec=pause_sec)
        rows.append(order_row(pk, target_date, home, away, feed))
    if not rows:
        return 0
    try:
        return upsert_lineups(pd.DataFrame(rows), out_path=out_path)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  lineup pre-game capture: persist failed ({e}); "
              f"membership degrades to projection")
        return 0


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
    # COMPLETE-only resume set: incomplete/fetch_error rows retry every run.
    done = complete_game_pks(out_path)
    todo = games[~games["game_pk"].isin(done)]
    print(f"pending: {len(todo)} (complete {len(done)})")

    rows = []
    for r in todo.head(args.limit).itertuples():
        pk = int(r.game_pk)
        feed, err = fetch_feed(pk, pause_sec=PAUSE_SEC)
        if feed is None:
            rows.append(order_row(pk, r.game_date, r.home_team, r.away_team,
                                  None, state=f"fetch_error: {err}"))
            continue
        rows.append(order_row(pk, r.game_date, r.home_team, r.away_team, feed))
        if len(rows) % 50 == 0:
            print(f"  ...{len(rows)} this chunk", flush=True)

    df_new = pd.DataFrame(rows)
    if not df_new.empty:
        upsert_lineups(df_new, out_path=out_path)
    print(f"lineups.parquet now: {len(pd.read_parquet(out_path))} games")


if __name__ == "__main__":
    main()
