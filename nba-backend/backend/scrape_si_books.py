"""Fetch per-bookmaker H2H odds (pointsbet, bet365) for known SI matches.

Reads sim_match_id list from si_nba_all_seasons.csv -- no round scan.
Output: nba-backend/run_diagnostics/si_nba_books.csv
  (sim_match_id, book, home_odds_dec, away_odds_dec)

Usage: python scrape_si_books.py [--books pointsbet bet365]
TAB odds already live in si_nba_all_seasons.csv (home_odds_dec/away_odds_dec).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}
DIAG = "nba-backend/run_diagnostics"


def fetch_odds(match_id: str, book: str):
    try:
        r = requests.get(
            "https://levy-edge.statsinsider.com.au/pre/"
            f"{match_id}?best_bets=true&bookmakers={book}",
            headers=HEADERS, timeout=25)
        r.raise_for_status()
        d = r.json()["BettingData"][book]
        return {"sim_match_id": match_id, "book": book,
                "home_odds_dec": d.get("HomeOdds"),
                "away_odds_dec": d.get("AwayOdds")}
    except Exception as e:
        print(f"Error {match_id} {book}: {e}", flush=True)
        return {"sim_match_id": match_id, "book": book,
                "home_odds_dec": None, "away_odds_dec": None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="*", default=["pointsbet", "bet365"])
    a = ap.parse_args(argv)
    import pandas as pd
    ids = pd.read_csv(f"{DIAG}/si_nba_all_seasons.csv")["sim_match_id"].unique().tolist()
    print(f"{len(ids)} matches x {a.books}", flush=True)
    jobs = [(m, b) for m in ids for b in a.books]
    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=20) as ex:
        futs = {ex.submit(fetch_odds, m, b): (m, b) for m, b in jobs}
        for f in as_completed(futs):
            done += 1
            rows.append(f.result())
            if done % 500 == 0:
                print(f"{done}/{len(jobs)} fetched", flush=True)
    out = f"{DIAG}/si_nba_books.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
