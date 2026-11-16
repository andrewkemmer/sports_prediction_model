"""NBA market odds pull from Stats Insider (adapted from the user's scrape_nba.py).

Covers training seasons 2023-2026 (SI season = year the season starts).
Round scan 1..250 per season (rounds are sub-weekly; 2023 needed >59).

Output (nba-backend/run_diagnostics/):
  si_nba_{season}.csv          one row per SIMatchID, raw TAB decimal odds
  si_nba_{book}_{season}.csv   same, for a non-TAB bookmaker
  si_nba_all_seasons.csv       combined (TAB)
  si_nba_{book}_all_seasons.csv combined (non-TAB)

The bookmaker is a request parameter, not a hardcoded choice: ``--book
tab`` is the historical default (and keeps the original filenames so the
O-PTS edge slice still reads what it always read), ``--book fanduel`` pulls
the same payload's FanDuel H2H, spread and total. Only the names the
upstream actually publishes come back, so a wrong book is an empty
``BettingData`` rather than a silent substitution.

Odds are ONE quote per game (decimal): the feed's LAST PRE-GAME line, kept
after tip-off and never moved in-play, so it is the close. Two stamps prove
it and are stored per row:

  * ``book_last_updated`` -- ``BettingData.tab.LastUpdated``, when the book
    last moved the quote. Across 5,502 collected prices this sits a median
    4 minutes after the scheduled tip (p10 -9.5, p90 -0.9; 90% at or past
    the start), i.e. the line is frozen at kickoff.
  * ``fetched_utc`` -- when this scraper captured the row.

``--refresh`` re-fetches every id and merges keep-newest on those stamps, so
a re-capture walks each game toward its close and a failed fetch keeps the
row already on disk. ``--resume`` skips ids already present. De-vig + join
to the OOF frame happens in market_rundown.compare (or a follow-up step);
preseason/non-real games are filtered at join time against the graded game
list, not here.

Usage:
  python scrape_statsinsider_nba.py                       # all four seasons, TAB
  python scrape_statsinsider_nba.py --seasons 2025 2026
  python scrape_statsinsider_nba.py --book fanduel
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}
SCHEDULE_URL = (
    "https://levy-edge.statsinsider.com.au/round/matches"
    "?Sport=NBA&Round={round}&Season={season}&strip=true&best_bets=true"
    "&bookmakers={book}"
)
MATCH_URL = ("https://levy-edge.statsinsider.com.au/pre/{match_id}"
             "?best_bets=true&bookmakers={book}")
DIAG = "nba-backend/run_diagnostics"
MAX_ROUND = 250


def get_round_match_ids(season: int, round_number: int, book: str = "tab"):
    try:
        r = requests.get(
            SCHEDULE_URL.format(round=round_number, season=season, book=book),
            headers=HEADERS, timeout=20)
    except Exception:
        return []
    if r.status_code != 200:
        return []
    try:
        data = r.json()
    except Exception:
        return []
    if not isinstance(data, list) or not data:
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        sim_id = item.get("MatchData", {}).get("SIMatchID")
        if sim_id:
            out.append(sim_id)
    return out


def fetch_match(match_id: str, book: str = "tab", attempts: int = 3):
    """One SI match payload, retried.

    The host throttles a sustained sweep: at 20 workers a run starts timing
    out on connect (not on read) after ~1,200 games, and a single miss is a
    permanently unpriced game because the join is one snapshot per id. Three
    attempts with a widening pause converts that tail of transient refusals
    into data; a request that is actually wrong still fails all three.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            r = requests.get(MATCH_URL.format(match_id=match_id, book=book),
                             headers=HEADERS, timeout=25)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:  # noqa: BLE001 - any transport failure
            last = e
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    else:
        print(f"Error fetching {match_id}: {last}", flush=True)
        return None
    try:
        md = data["MatchData"]
        bd = (data.get("BettingData") or {}).get(book) or {}
        pre = data.get("PreData", {}) or {}
        agg = data.get("aggregatedBettingInfo", {}) or {}
        live = data.get("LiveData", {}) or {}

        home_obj = md.get("HomeTeam", {}) or {}
        away_obj = md.get("AwayTeam", {}) or {}
        home_name = home_obj.get("DisplayName")
        away_name = away_obj.get("DisplayName")
        if home_name is None or away_name is None:
            return None  # not a real game (same rule as the user's script)

        # The book's OWN stamp for this quote, and when we captured it. NHL
        # parity: Stats Insider keeps serving the LAST PRE-GAME quote after
        # tip-off, so the newest stamp is the closing line and a later capture
        # always supersedes an earlier one (2026-11-16).
        return {
            "sim_match_id": md.get("SIMatchID"),
            "book_last_updated": bd.get("LastUpdated"),
            "fetched_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "season": md.get("Season"),
            "round": md.get("RoundNumber"),
            # UTCDate is when the line was PUBLISHED, not when the game
            # started: joining on it needs a date tolerance and a score
            # tiebreak. ``Date`` is the tip-off instant, so a join on it is
            # exact (2026-11-16).
            "date_utc": md.get("UTCDate"),
            "tipoff_utc": md.get("Date"),
            "tipoff_epoch": md.get("DateStamp"),
            "venue": md.get("Venue"),
            "home_team": home_name,
            "home_abv": home_obj.get("Abv"),
            "away_team": away_name,
            "away_abv": away_obj.get("Abv"),
            # Decimal odds snapshot (effectively the stored/close line).
            # ``aggregatedBettingInfo`` mirrors the requested book into the
            # same fields, so a non-TAB pull is labelled here rather than
            # read as TAB by a later stage.
            "book": book,
            "home_odds_dec": agg.get("HomeOdds"),
            "away_odds_dec": agg.get("AwayOdds"),
            "home_line": agg.get("HomeLine"),
            "home_line_odds": agg.get("HomeLineOdds"),
            "away_line_odds": agg.get("AwayLineOdds"),
            "total_line": agg.get("TotalLine"),
            "over_odds": agg.get("OverOdds"),
            "under_odds": agg.get("UnderOdds"),
            "h2h_overround": bd.get("H2HOverround"),
            "pred_home_score": pre.get("PredHomeScore"),
            "pred_away_score": pre.get("PredAwayScore"),
            "pred_margin": pre.get("PredMargin"),
            "pythag_home": pre.get("PythagHome"),
            "pythag_away": pre.get("PythagAway"),
            "final_home_score": live.get("homeScore"),
            "final_away_score": live.get("awayScore"),
            "game_status": live.get("status"),
        }
    except Exception as e:
        print(f"Error parsing {match_id}: {e}", flush=True)
        return None


def scrape_season(season: int, book: str = "tab",
                  resume: bool = False, workers: int = 20,
                  recent_days: float | None = None) -> pd.DataFrame:
    print(f"--- season {season}: scanning rounds 1..{MAX_ROUND} ({book})"
          + (" [resume]" if resume else "") + " ---", flush=True)
    # One pass, kept in round order: the round list IS the id list, so asking
    # every round twice (validity probe, then collection) doubled the sweep
    # for no information.
    rounds = list(range(1, MAX_ROUND + 1))
    with ThreadPoolExecutor(max_workers=10) as ex:
        per_round = list(ex.map(lambda r: get_round_match_ids(season, r, book),
                                rounds))
    valid = [r for r, ids in zip(rounds, per_round) if ids]
    print(f"season {season}: {len(valid)} valid rounds", flush=True)
    mids = [m for ids in per_round for m in ids]
    mids = list(dict.fromkeys(mids))
    # The prior file is the BASE STATE of the merge, so it is read on every
    # path. ``resume`` only decides which ids get fetched: with it off a
    # re-capture updates rows in place, and a failed fetch keeps the row it
    # already had instead of deleting it (2026-11-16: --refresh dropped 8
    # games this way).
    prior = pd.DataFrame()
    path = f"{DIAG}/{_stem(book, season)}.csv"
    if os.path.exists(path):
        prior = pd.read_csv(path)
    if resume and len(prior):
        done = set(prior["sim_match_id"].astype(str))
        before = len(mids)
        mids = [m for m in mids if str(m) not in done]
        print(f"season {season}: resuming, {before - len(mids)} already "
              f"on disk, {len(mids)} to fetch", flush=True)
    if recent_days is not None:
        # NHL ``plan_fetch`` parity: ids still in flight get re-fetched so the
        # row walks to its closing line; everything already settled keeps the
        # quote it has. A game with no stored tip-off is fetched, because an
        # unknown age must never be assumed settled.
        tips = (pd.to_datetime(prior.set_index("sim_match_id")["tipoff_utc"],
                               errors="coerce", utc=True)
                if len(prior) and "tipoff_utc" in prior.columns
                else pd.Series(dtype="datetime64[ns, UTC]"))
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=recent_days)
        before = len(mids)
        mids = [m for m in mids
                if pd.isna(tips.get(str(m), pd.NaT))
                or tips.get(str(m)) >= cutoff]
        print(f"season {season}: recent window {recent_days}d, "
              f"{len(mids)}/{before} to re-fetch", flush=True)
    print(f"season {season}: fetching {len(mids)} matches", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_match, m, book): m for m in mids}
        done = 0
        for f in as_completed(futs):
            done += 1
            res = f.result()
            if res:
                rows.append(res)
            if done % 200 == 0:
                print(f"season {season}: {done}/{len(mids)} fetched", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(DIAG, exist_ok=True)
    if len(prior):
        df = _merge_keep_newest(prior, df)
    df.to_csv(f"{DIAG}/{_stem(book, season)}.csv", index=False)
    print(f"season {season}: saved {len(df)} rows", flush=True)
    return df


def _merge_keep_newest(prior: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Overlay ``new`` on ``prior``, one row per match, newest quote wins.

    "Newest" is the BOOK's stamp first (``book_last_updated``) and our capture
    time second, so a replayed run from a stale clone cannot regress a row that
    already holds the closing line. Mirrors
    ``nhl-backend/backend/odds_history.merge_keep_newest``.
    """
    if not len(prior):
        return new.reset_index(drop=True)
    key = "sim_match_id"
    frames = []
    for frame in (prior, new):
        f = frame.copy()
        for col in ("book_last_updated", "fetched_utc"):
            if col not in f.columns:
                f[col] = pd.NaT
        f["_lu"] = pd.to_datetime(f["book_last_updated"], errors="coerce",
                                  utc=True)
        f["_fu"] = pd.to_datetime(f["fetched_utc"], errors="coerce", utc=True)
        frames.append(f)
    p, n = frames
    n = (n.sort_values(["_lu", "_fu"]).drop_duplicates(key, keep="last"))
    p = (p.sort_values(["_lu", "_fu"]).drop_duplicates(key, keep="last"))
    merged = pd.concat([p[~p[key].isin(n[key])], n], ignore_index=True)
    return merged.drop(columns=["_lu", "_fu"])


def _stem(book: str, season: int | None = None) -> str:
    """TAB keeps the historical filenames; every other book is named."""
    base = "si_nba" if book == "tab" else f"si_nba_{book}"
    return base if season is None else f"{base}_{season}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int, default=[2023, 2024, 2025, 2026])
    ap.add_argument("--book", default="tab",
                    help="Stats Insider bookmaker key (tab, fanduel, ...)")
    ap.add_argument("--resume", action="store_true",
                    help="keep rows already on disk and fetch only missing ids")
    ap.add_argument("--recent", type=float, default=None, metavar="DAYS",
                    help="re-fetch only ids whose tip-off is within DAYS "
                         "(plus any id with no stored tip), so rows still "
                         "in flight move toward the close (NHL plan_fetch "
                         "parity)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch every id and keep the NEWEST quote per "
                         "match, so rows move toward the closing line "
                         "(NHL parity)")
    ap.add_argument("--workers", type=int, default=20,
                    help="match fetches in flight (lower it when the host "
                         "starts refusing connections)")
    a = ap.parse_args(argv)
    frames = []
    for s in a.seasons:
        df = scrape_season(s, a.book, a.resume and not a.refresh, a.workers,
                           recent_days=None if a.refresh else a.recent)
        if len(df):
            frames.append(df)
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        out = f"{DIAG}/{_stem(a.book)}_all_seasons.csv"
        all_df.to_csv(out, index=False)
        print(f"combined: {len(all_df)} rows -> {out}", flush=True)
        print(all_df[["sim_match_id", "date_utc", "away_abv", "home_abv",
                       "home_odds_dec", "away_odds_dec",
                       "final_home_score", "final_away_score"]].head().to_string())


if __name__ == "__main__":
    sys.exit(main())
