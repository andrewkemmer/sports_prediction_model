"""Incremental Stats Insider moneyline-price refresh — Option A.

Phase 2 calls :func:`refresh` so the production run maintains its own price
history; no separate scraper job exists.

The CSV in ``data_delivery`` is the BASE STATE, never a cache. A Kaggle run
clones the repo and has no per-match JSON cache, so every refresh merges onto
the committed file:

  * enumerate the most recent ``ROUNDS`` betting rounds of the current season
    (ids only — cheap, a few dozen requests),
  * fetch a price for any id **missing** from the base, whatever its age, so
    historical gaps heal themselves,
  * fetch a price for any id whose game falls inside the ``DAYS`` window, so
    the rows still in flight move toward the closing line,
  * merge keep-newest on ``fetched_utc`` and write atomically.

Closing-line policy (validated 2026-11-16): Stats Insider keeps serving the
LAST PRE-GAME quote after puck drop — ``LastUpdated`` lands at or just after
the start and the numbers do not move in-play. A differing quote is therefore
always a *later* pre-game line, never an in-play one, so it must replace the
earlier one for every id, past or present.

Nothing here raises into the pipeline: the caller wraps it, and a run with a
one-run-stale history is correct-but-late rather than broken.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = "https://levy-edge.statsinsider.com.au"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "nhl-backend" / "data_delivery" / "nhl_odds_history.csv"
CURSOR = ROOT / "nhl-backend" / "data_delivery" / "nhl_odds_rounds.json"
# Outside the repo: data_delivery is swept by the pipeline's own
# ``git add -A``, so a lock file living there would ride into an artifacts
# commit. The lock only guards concurrent writers on THIS machine.
LOCK = Path(tempfile.gettempdir()) / "nhl_odds_history.lock"

ROUNDS = int(os.environ.get("NHL_ODDS_ROUNDS", "30"))
DAYS = float(os.environ.get("NHL_ODDS_DAYS", "3"))
WORKERS = int(os.environ.get("NHL_ODDS_WORKERS", "8"))
BUDGET = float(os.environ.get("NHL_ODDS_BUDGET", "180"))
EMPTY_LIMIT = 30          # consecutive empty rounds that mean "end of season"
MAX_FETCH = 400           # hard cap on price requests in one refresh
TIMEOUT = 20.0

_PRICE_COLS = ["match_id", "season", "round", "date", "home", "away",
               "home_score", "away_score", "fetched_utc",
               "tab_homeodds", "tab_awayodds", "tab_homeline",
               "novig_homeodds", "novig_awayodds", "novig_homeline",
               "tab_last_updated", "novig_last_updated",
               "home_odds_agg", "away_odds_agg", "home_line_agg",
               "total_line_agg"]


def _now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _get(url: str, tries: int = 2, timeout: float = TIMEOUT):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0",
                              "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            if attempt == tries - 1:
                return None
            time.sleep(0.5 + attempt)
    return None


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def matches_for(season: int, rnd: int) -> list:
    d = _get(f"{BASE}/round/matches?Sport=NHL&Round={rnd}&Season={season}"
             "&strip=true&best_bets=true&bookmakers=novig,tab")
    if not d:
        return []
    return d if isinstance(d, list) else (d.get("matches") or d.get("data") or [])


def _game(m, season: int, rnd: int) -> dict | None:
    md = m.get("MatchData") or m
    mid = md.get("SIMatchID") or m.get("SIMatchID")

    def team(key):
        t = md.get(key)
        if isinstance(t, dict):
            return t.get("Abv") or t.get("DisplayName")
        return t

    if not mid:
        return None
    return {"match_id": mid, "season": season, "round": rnd,
            "date": md.get("Date"), "home": team("HomeTeam"),
            "away": team("AwayTeam"), "home_score": md.get("HomeScore"),
            "away_score": md.get("AwayScore")}


def enumerate_recent(season: int, cursor: int, rounds: int,
                     end=None) -> tuple[list, int]:
    """Games inside the last ``rounds`` days, plus everything still scheduled.

    The lookback trails NOW, not the furthest round on the schedule. Stats
    Insider publishes the season's fixtures up front, so the cursor sits
    weeks ahead (NHL 2026-11-17: round 94 = Dec 31). The old window started
    at ``cursor - rounds + 1`` and therefore began in December: no November
    round was enumerated at all, so the last 30 days were neither healed for
    missing matches nor re-priced before their games. That run made 3 price
    requests, added 3 December matches, moved 0, and left every game dated
    that night on its 11:09Z morning capture (2026-11-18).

    Forward from the cursor covers everything still scheduled; backward from
    the cursor covers the lookback, stopping once a round's games all fall
    outside ``rounds`` days. Only games inside that horizon are returned.

    ``end`` closes the far side at the pipeline's own last day
    (``NBA_END_DATE`` / ``NHL_END_DATE``). A game past that date is never
    priced by the model, so enumerating it only spends a price request: on
    2026-11-18 the NHL walk reached Dec 31 and fetched prices for fixtures
    the run would not touch for six weeks.
    """
    now = _now()
    horizon = now - pd.Timedelta(days=rounds)
    end_day = None
    if end is not None:
        try:
            _e = pd.Timestamp(end)
        except (TypeError, ValueError, OverflowError):
            _e = pd.NaT
        if not pd.isna(_e):
            end_day = _e.normalize()
    games: list[dict] = []
    seen: set[str] = set()
    last_with_games = 0

    def start_of(game: dict):
        try:
            ts = pd.Timestamp(game.get("date"))
        except (TypeError, ValueError, OverflowError):
            return None
        if ts is pd.NaT or pd.isna(ts):
            return None
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    def day_of(game: dict):
        ts = start_of(game)
        if ts is None:
            return None
        return ts.tz_convert("America/New_York").normalize().tz_localize(None)

    def collect(rnd: int) -> list[dict]:
        nonlocal last_with_games
        got: list[dict] = []
        for m in matches_for(season, rnd) or []:
            g = _game(m, season, rnd)
            if not g:
                continue
            mid = str(g["match_id"])
            if mid in seen:
                continue
            seen.add(mid)
            games.append(g)
            got.append(g)
        if got:
            last_with_games = max(last_with_games, rnd)
        return got

    rnd, empty = max(1, int(cursor)), 0
    while rnd < 700 and empty < EMPTY_LIMIT:
        got = collect(rnd)
        empty = 0 if got else empty + 1
        if got and end_day is not None:
            days = [d for d in (day_of(g) for g in got) if d is not None]
            if days and min(days) > end_day:
                break
        rnd += 1

    rnd, empty = max(1, int(cursor)) - 1, 0
    while rnd >= 1 and empty < EMPTY_LIMIT:
        got = collect(rnd)
        empty = 0 if got else empty + 1
        starts = [s for s in (start_of(g) for g in got) if s is not None]
        if starts and max(starts) < horizon:
            break
        rnd -= 1

    kept = [g for g in games
            if start_of(g) is not None and start_of(g) >= horizon
            and (end_day is None or (day_of(g) is not None
                                     and day_of(g) <= end_day))]
    return kept, last_with_games


def _price_record(g: dict) -> dict | None:
    """Current quote for one match, in the history's column schema."""
    mid = g["match_id"]
    p = _get(f"{BASE}/pre/{mid}?Sport=NHL&best_bets=true&bookmakers=novig,tab")
    if not p:
        return None
    abi = p.get("aggregatedBettingInfo") or {}
    bd = p.get("BettingData") or {}
    rec = dict(g)
    rec["fetched_utc"] = _now().isoformat()
    rec.update({"home_odds_agg": _num(abi.get("HomeOdds")),
                "away_odds_agg": _num(abi.get("AwayOdds")),
                "home_line_agg": _num(abi.get("HomeLine")),
                "total_line_agg": _num(abi.get("TotalLine"))})
    for book in ("novig", "tab"):
        b = bd.get(book) if isinstance(bd, dict) else None
        if isinstance(b, str):
            try:
                b = json.loads(b.replace("'", '"'))
            except Exception:  # noqa: BLE001
                b = None
        if not isinstance(b, dict):
            continue
        rec[f"{book}_last_updated"] = b.get("LastUpdated")
        rec[f"{book}_homeodds"] = _num(b.get("HomeOdds"))
        rec[f"{book}_awayodds"] = _num(b.get("AwayOdds"))
        rec[f"{book}_homeline"] = _num(b.get("HomeLine"))
    return rec


def plan_fetch(games: list[dict], base: pd.DataFrame, now: pd.Timestamp,
               days: float, cap: int = MAX_FETCH) -> list[dict]:
    """Which enumerated games need a request.

    Missing ids always qualify (they heal old gaps), and so does a row the
    module never captured - no ``fetched_utc`` means no proof the stored quote
    is a closing line (2026-11-18: 173 of the NHL lookback's 207 rows were
    unstamped legacy rows and could never heal). Ids already captured qualify
    inside the freshness window, where the quote can still move toward the
    close; everything else is already at its closing line.
    """
    known = set(base["match_id"].astype(str)) if len(base) and "match_id" in base else set()
    if len(base) and "match_id" in base:
        stamps = (pd.to_datetime(base["fetched_utc"], errors="coerce", utc=True)
                  if "fetched_utc" in base.columns
                  else pd.Series([pd.NaT] * len(base), index=base.index))
        stamped = set(base.loc[stamps.notna(), "match_id"].astype(str))
    else:
        stamped = set()
    cutoff = now - pd.Timedelta(days=days)
    out: list[dict] = []
    seen: set[str] = set()
    for g in games:
        mid = str(g["match_id"])
        if mid in seen:
            continue
        seen.add(mid)
        try:
            start = pd.Timestamp(g.get("date"))
            if start.tzinfo is None:
                start = start.tz_localize("UTC")
        except Exception:  # noqa: BLE001
            start = None
        if (mid not in known or mid not in stamped
                or (start is not None and start >= cutoff)):
            out.append(g)
    return out[:cap]


def merge_keep_newest(base: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Overlay ``new`` on ``base``, never letting an older capture win.

    Two writers (or a replayed run from a stale clone) cannot regress a row:
    the incoming record is used only when it carries a ``fetched_utc`` at
    least as recent as the stored one.
    """
    if not len(new):
        return base, 0, 0
    new = new.copy()
    new["match_id"] = new["match_id"].astype(str)
    if not len(base) or "match_id" not in base:
        return new.reset_index(drop=True), len(new), 0
    base = base.copy()
    base["match_id"] = base["match_id"].astype(str)
    prior_ts = {}
    if "fetched_utc" in base:
        prior_ts = dict(zip(base["match_id"],
                            pd.to_datetime(base["fetched_utc"], utc=True,
                                           errors="coerce")))
    incoming_ts = pd.to_datetime(new["fetched_utc"], utc=True, errors="coerce")
    keep = []
    for i, mid in enumerate(new["match_id"]):
        prior = prior_ts.get(mid)
        cur = incoming_ts.iloc[i]
        keep.append(not (prior is not None and pd.notna(prior)
                         and pd.notna(cur) and cur < prior))
    new_kept = new[keep]
    added = int((~new_kept["match_id"].isin(base["match_id"])).sum())
    updated = int(len(new_kept) - added)
    merged = pd.concat([base[~base["match_id"].isin(new_kept["match_id"])],
                        new_kept], ignore_index=True)
    if "date" in merged:
        merged = merged.sort_values("date", kind="stable")
    return merged.reset_index(drop=True), added, updated


def _locked():
    """Best-effort exclusive lock (POSIX). No-op where unsupported."""
    try:
        import fcntl
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        fh = open(LOCK, "w")
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fh
    except Exception:  # noqa: BLE001
        return None


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _read_cursor() -> tuple[int, int]:
    """(season, last round) from the committed cursor, or the calendar year."""
    try:
        d = json.loads(CURSOR.read_text())
        return int(d.get("season") or 0), int(d.get("round") or 0)
    except Exception:  # noqa: BLE001
        return 0, 0


def _save_cursor(season: int, rnd: int) -> None:
    try:
        CURSOR.parent.mkdir(parents=True, exist_ok=True)
        CURSOR.write_text(json.dumps({"season": season, "round": rnd,
                                      "updated_utc": _now().isoformat()},
                                     indent=1))
    except Exception:  # noqa: BLE001
        pass


def refresh(rounds: int = ROUNDS, days: float = DAYS,
            budget: float = BUDGET, season: int | None = None,
            end=None) -> dict:
    """Refresh the price history in place. Returns a run summary."""
    t0 = time.monotonic()
    out = {"ok": False, "rows": 0, "added": 0, "updated": 0, "fetched": 0,
           "failed": 0, "skipped": "", "elapsed": 0.0}
    fh = _locked()
    if fh is None and LOCK.exists():
        out["skipped"] = "another refresh holds the lock"
        return out
    try:
        base = (pd.read_csv(OUT) if OUT.exists() else pd.DataFrame())
        if not len(base):
            out["skipped"] = f"no base history at {OUT}"
            return out
        cur_season, cur_round = _read_cursor()
        # The cursor can point at a season the backfill finished on. Follow the
        # calendar instead: SI numbers a season by the year it starts, so a
        # stale cursor must not freeze the daily refresh on last season.
        season = int(season or 0)
        if not season:
            season = cur_season if cur_season >= _now().year else _now().year
        if season != cur_season:
            cur_round = 0
        games, last_round = enumerate_recent(season, cur_round, rounds,
                                             end=end)
        if not games:
            out["skipped"] = f"no rounds enumerated for {season}"
            return out
        targets = plan_fetch(games, base, _now(), days)
        out["fetched"] = len(targets)
        recs: list[dict] = []
        deadline = t0 + budget
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for rec in ex.map(_price_record, targets):
                if time.monotonic() > deadline:
                    out["skipped"] = "budget exhausted mid-fetch"
                    break
                if rec is None:
                    out["failed"] += 1
                    continue
                recs.append(rec)
        new = pd.DataFrame(recs)
        if len(new):
            merged, added, updated = merge_keep_newest(base, new)
            _write_atomic(merged, OUT)
            out.update(added=added, updated=updated, rows=len(merged))
        else:
            out["rows"] = len(base)
        _save_cursor(season, last_round)
        out["ok"] = True
        return out
    finally:
        out["elapsed"] = round(time.monotonic() - t0, 1)
        if fh is not None:
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass


def full_sweep(seasons=(2023, 2024, 2025, 2026)) -> dict:
    """Backfill: enumerate every round of every season, fetch what is missing.

    Manual only (``python -m odds_history --full``): ~20 minutes, and it is
    the only path that can find an id absent from the file for a whole season.
    """
    t0 = time.monotonic()
    base = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    known = set(base["match_id"].astype(str)) if len(base) else set()
    games: list[dict] = []
    for season in seasons:
        empty = 0
        for rnd in range(1, 700):
            ms = matches_for(season, rnd)
            if not ms:
                empty += 1
                if empty >= EMPTY_LIMIT:
                    break
                continue
            empty = 0
            for m in ms:
                g = _game(m, season, rnd)
                if g and str(g["match_id"]) not in known:
                    games.append(g)
    recs = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for rec in ex.map(_price_record, games):
            if rec:
                recs.append(rec)
    new = pd.DataFrame(recs)
    merged, added, updated = merge_keep_newest(base, new)
    if len(new):
        _write_atomic(merged, OUT)
    return {"rows": len(merged), "added": added, "updated": updated,
            "elapsed": round(time.monotonic() - t0, 1)}


if __name__ == "__main__":
    import sys
    if "--full" in sys.argv:
        print(full_sweep())
    else:
        print(refresh())
