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

Odds are ONE snapshot per game (decimal). There is no open/close split in
this payload -- treat as the stored (effectively close) line. De-vig + join
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

        return {
            "sim_match_id": md.get("SIMatchID"),
            "season": md.get("Season"),
            "round": md.get("RoundNumber"),
            "date_utc": md.get("UTCDate"),
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
                  resume: bool = False, workers: int = 20) -> pd.DataFrame:
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
    prior = pd.DataFrame()
    if resume:
        path = f"{DIAG}/{_stem(book, season)}.csv"
        if os.path.exists(path):
            prior = pd.read_csv(path)
            done = set(prior["sim_match_id"].astype(str))
            before = len(mids)
            mids = [m for m in mids if str(m) not in done]
            print(f"season {season}: resuming, {before - len(mids)} already "
                  f"on disk, {len(mids)} to fetch", flush=True)
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
        df = pd.concat([prior, df], ignore_index=True)
        df = df.drop_duplicates("sim_match_id", keep="last")
    df.to_csv(f"{DIAG}/{_stem(book, season)}.csv", index=False)
    print(f"season {season}: saved {len(df)} rows", flush=True)
    return df


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
    ap.add_argument("--workers", type=int, default=20,
                    help="match fetches in flight (lower it when the host "
                         "starts refusing connections)")
    a = ap.parse_args(argv)
    frames = []
    for s in a.seasons:
        df = scrape_season(s, a.book, a.resume, a.workers)
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
