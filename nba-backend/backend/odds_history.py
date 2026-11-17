"""Incremental Stats Insider moneyline-price refresh (NBA twin of the NHL module).

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

Book policy (2026-11-17): TAB is the OFFSET series and is de-vigged
proportionally at use. Novig is stored raw and is the DISPLAY price the slate
publishes, matching the NHL structure. FanDuel is deliberately absent — it is
out of scope for this adoption.

Closing-line policy (NHL precedent, re-validated for NBA 2026-11-17): Stats
Insider keeps serving the LAST PRE-GAME quote after tip-off — ``LastUpdated``
lands at or just after the start and the numbers do not move in-play. A
differing quote is therefore always a *later* pre-game line, never an in-play
one, so it must replace the earlier one for every id, past or present.

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
OUT = ROOT / "nba-backend" / "data_delivery" / "nba_odds_history.csv"
CURSOR = ROOT / "nba-backend" / "data_delivery" / "nba_odds_rounds.json"
# Outside the repo: data_delivery is swept by the pipeline's own
# ``git add -A``, so a lock file living there would ride into an artifacts
# commit. The lock only guards concurrent writers on THIS machine.
LOCK = Path(tempfile.gettempdir()) / "nba_odds_history.lock"

ROUNDS = int(os.environ.get("NBA_ODDS_ROUNDS", "30"))
DAYS = float(os.environ.get("NBA_ODDS_DAYS", "3"))
WORKERS = int(os.environ.get("NBA_ODDS_WORKERS", "8"))
BUDGET = float(os.environ.get("NBA_ODDS_BUDGET", "180"))
EMPTY_LIMIT = 30          # consecutive empty rounds that mean "end of season"
MAX_FETCH = 400           # hard cap on price requests in one refresh
TIMEOUT = 20.0
SPORT = "NBA"

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
    d = _get(f"{BASE}/round/matches?Sport={SPORT}&Round={rnd}&Season={season}"
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


def enumerate_recent(season: int, cursor: int, rounds: int) -> tuple[list, int]:
    """Games from the most recent ``rounds`` rounds, plus the last round seen.

    The walk starts inside the previous window and steps forward until the
    season runs dry, so a run that resumes after a long gap still finds the
    current rounds (only the tail past the last game is empty).
    """
    start = max(1, int(cursor) - rounds + 1)
    games: list[dict] = []
    last_with_games = 0
    empty = 0
    rnd = start
    while rnd < 700 and empty < EMPTY_LIMIT:
        ms = matches_for(season, rnd)
        if ms:
            empty = 0
            last_with_games = rnd
            for m in ms:
                g = _game(m, season, rnd)
                if g:
                    games.append(g)
        else:
            empty += 1
        rnd += 1
    keep_from = max(1, last_with_games - rounds + 1)
    games = [g for g in games if g["round"] >= keep_from]
    return games, last_with_games


def _price_record(g: dict) -> dict | None:
    """Current quote for one match, in the history's column schema."""
    mid = g["match_id"]
    p = _get(f"{BASE}/pre/{mid}?Sport={SPORT}&best_bets=true&bookmakers=novig,tab")
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

    Missing ids always qualify (they heal old gaps); ids already in the base
    qualify only inside the freshness window, where the quote can still move
    toward the close. Everything else is already at its closing line.
    """
    known = set(base["match_id"].astype(str)) if len(base) and "match_id" in base else set()
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
        if mid not in known or (start is not None and start >= cutoff):
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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _fetch_records(games: list[dict], budget: float, t0: float) -> tuple[list[dict], int, str]:
    recs: list[dict] = []
    failed = 0
    skipped = ""
    deadline = t0 + budget
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for rec in ex.map(_price_record, games):
            if time.monotonic() > deadline:
                skipped = "budget exhausted mid-fetch"
                break
            if rec is None:
                failed += 1
                continue
            recs.append(rec)
    return recs, failed, skipped


def refresh(rounds: int = ROUNDS, days: float = DAYS,
            budget: float = BUDGET, season: int | None = None) -> dict:
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
        games, last_round = enumerate_recent(season, cur_round, rounds)
        if not games:
            out["skipped"] = f"no rounds enumerated for {season}"
            return out
        targets = plan_fetch(games, base, _now(), days)
        out["fetched"] = len(targets)
        recs, failed, skipped = _fetch_records(targets, budget, t0)
        out["failed"] = failed
        out["skipped"] = skipped
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


def full_sweep(seasons=(2022, 2023, 2024, 2025, 2026),
               budget: float = 240.0, flush_every: int = 20) -> dict:
    """Backfill: enumerate every round of every season, fetch what is missing.

    Manual only (``python -m odds_history --full``), and RESUMABLE: prices are
    flushed to the CSV every ``flush_every`` rounds and the cursor records the
    last round completed per season, so a timeout loses at most one flush
    window rather than the whole sweep.
    """
    t0 = time.monotonic()
    fh = _locked()
    base = (pd.read_csv(OUT) if OUT.exists() else pd.DataFrame())
    known = set(base["match_id"].astype(str)) if len(base) else set()
    try:
        sweep = json.loads(CURSOR.read_text()).get("sweep", {})
    except Exception:  # noqa: BLE001
        sweep = {}
    totals = {"rows": len(base), "added": 0, "updated": 0, "fetched": 0,
              "rounds_scanned": 0, "elapsed": 0.0}
    try:
        for season in seasons:
            start_round = int(sweep.get(str(season), 0)) + 1
            empty = 0
            pending: list[dict] = []
            rnd = start_round
            while rnd < 700 and empty < EMPTY_LIMIT:
                if time.monotonic() - t0 > budget:
                    totals["skipped"] = f"budget exhausted in season {season} round {rnd}"
                    break
                ms = matches_for(season, rnd)
                totals["rounds_scanned"] += 1
                if not ms:
                    empty += 1
                else:
                    empty = 0
                    for m in ms:
                        g = _game(m, season, rnd)
                        if g and str(g["match_id"]) not in known:
                            pending.append(g)
                rnd += 1
                if pending and (len(pending) >= flush_every or rnd % flush_every == 0):
                    if pending:
                        recs, _, _ = _fetch_records(pending, budget, t0)
                        totals["fetched"] += len(pending)
                        new = pd.DataFrame(recs)
                        if len(new):
                            base, added, updated = merge_keep_newest(base, new)
                            known |= set(new["match_id"].astype(str))
                            totals["added"] += added
                            totals["updated"] += updated
                            _write_atomic(base, OUT)
                        pending = []
                    sweep[str(season)] = rnd
                    CURSOR.parent.mkdir(parents=True, exist_ok=True)
                    CURSOR.write_text(json.dumps(
                        {"season": season, "round": rnd, "sweep": sweep,
                         "updated_utc": _now().isoformat()}, indent=1))
            if pending:
                recs, _, _ = _fetch_records(pending, budget, t0)
                totals["fetched"] += len(pending)
                new = pd.DataFrame(recs)
                if len(new):
                    base, added, updated = merge_keep_newest(base, new)
                    known |= set(new["match_id"].astype(str))
                    totals["added"] += added
                    totals["updated"] += updated
                    _write_atomic(base, OUT)
                sweep[str(season)] = rnd
            if time.monotonic() - t0 > budget:
                break
        totals["rows"] = len(base)
        return totals
    finally:
        totals["elapsed"] = round(time.monotonic() - t0, 1)
        if fh is not None:
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass


def attach_prices(game_df: pd.DataFrame, path: str | None = None,
                  date_col: str = "gameday") -> tuple[pd.DataFrame, float]:
    """Attach the proportional TAB fair price and its logit to a game frame.

    Returns ``(frame, coverage)``. ``market_p_home`` is the proportional
    de-vig of the TAB decimals - the same estimator and the same overround
    gates as ``market_rundown.de_vig`` - and ``market_logit`` is its logit,
    which is the offset column. Unpriced games stay NaN: the offset arm must
    never see a filled-in 50%.
    """
    import numpy as np
    try:
        from backend import market_rundown as _mr
    except ImportError:  # pragma: no cover - script fallback
        import market_rundown as _mr
    d = pd.read_csv(path or OUT)
    h = pd.to_numeric(d.get("tab_homeodds"), errors="coerce")
    a = pd.to_numeric(d.get("tab_awayodds"), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        ih, ia = 1.0 / h, 1.0 / a
        ov = ih + ia
    ok = (np.isfinite(ih) & np.isfinite(ia) & (h > 1.0) & (a > 1.0)
          & (ov >= _mr.MIN_OVERROUND) & (ov <= _mr.MAX_OVERROUND))
    d = d[ok].copy()
    d["market_p_home"] = (1.0 / pd.to_numeric(d["tab_homeodds"], errors="coerce")
                          ) / (1.0 / pd.to_numeric(d["tab_homeodds"], errors="coerce")
                               + 1.0 / pd.to_numeric(d["tab_awayodds"], errors="coerce"))
    d["market_logit"] = np.log(d["market_p_home"] / (1.0 - d["market_p_home"]))
    d["_day"] = (pd.to_datetime(d["date"], utc=True, errors="coerce")
                 .dt.tz_convert("America/New_York").dt.tz_localize(None)
                 .dt.normalize())
    d = d.dropna(subset=["_day"]).drop_duplicates(["match_id"], keep="last")
    out = game_df.copy()
    out["_day"] = pd.to_datetime(out[date_col]).dt.normalize()
    right = d[["_day", "home", "away", "market_p_home", "market_logit",
               "tab_homeodds", "tab_awayodds", "novig_homeodds",
               "novig_awayodds"]].rename(columns={"home": "home_team",
                                                  "away": "away_team"})
    out = out.merge(right, on=["_day", "home_team", "away_team"], how="left")
    out = out.drop(columns=["_day"])
    cov = float(out["market_p_home"].notna().mean()) if len(out) else 0.0
    return out, cov


if __name__ == "__main__":
    import sys
    if "--full" in sys.argv:
        print(full_sweep())
    elif "--summary" in sys.argv:
        d = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
        print(f"rows {len(d)}")
        if len(d):
            print(d[["season", "match_id"]].groupby("season").size().to_string())
    else:
        print(refresh())
