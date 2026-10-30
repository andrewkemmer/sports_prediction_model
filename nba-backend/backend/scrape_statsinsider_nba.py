"""NBA market odds pull from Stats Insider (adapted from the user's scrape_nba.py).

Covers training seasons 2023-2026 (SI season = year the season starts).
Round scan 1..250 per season (rounds are sub-weekly; 2023 needed >59).

Output (nba-backend/run_diagnostics/):
  si_nba_{season}.csv          one row per SIMatchID, raw TAB decimal odds
  si_nba_all_seasons.csv       combined

Odds are ONE snapshot per game (TAB bookmaker, decimal). There is no
open/close split in this payload -- treat as the stored (effectively close)
line. De-vig + join to the OOF frame happens in market_rundown.compare
(or a follow-up step); preseason/non-real games are filtered at join time
against the graded game list, not here.

Usage:
  python scrape_statsinsider_nba.py            # all four seasons
  python scrape_statsinsider_nba.py --seasons 2025 2026
"""
from __future__ import annotations

import argparse
import os
import sys

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
    "?Sport=NBA&Round={round}&Season={season}&strip=true&best_bets=true&bookmakers=tab"
)
MATCH_URL = "https://levy-edge.statsinsider.com.au/pre/{match_id}?best_bets=true&bookmakers=tab"
DIAG = "nba-backend/run_diagnostics"
MAX_ROUND = 250


def get_round_match_ids(season: int, round_number: int):
    try:
        r = requests.get(SCHEDULE_URL.format(round=round_number, season=season),
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


def fetch_match(match_id: str):
    try:
        r = requests.get(MATCH_URL.format(match_id=match_id), headers=HEADERS, timeout=25)
        r.raise_for_status()
        data = r.json()
        md = data["MatchData"]
        bd = data["BettingData"]["tab"]
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
            # TAB decimal odds snapshot (effectively the stored/close line)
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
        print(f"Error fetching {match_id}: {e}", flush=True)
        return None


def scrape_season(season: int) -> pd.DataFrame:
    print(f"--- season {season}: scanning rounds 1..{MAX_ROUND} ---", flush=True)
    valid = []
    for r in range(1, MAX_ROUND + 1):
        ids = get_round_match_ids(season, r)
        if ids:
            valid.append(r)
    print(f"season {season}: {len(valid)} valid rounds", flush=True)
    mids = []
    for r in valid:
        mids.extend(get_round_match_ids(season, r))
    mids = list(dict.fromkeys(mids))
    print(f"season {season}: fetching {len(mids)} matches", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        futs = {ex.submit(fetch_match, m): m for m in mids}
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
    df.to_csv(f"{DIAG}/si_nba_{season}.csv", index=False)
    print(f"season {season}: saved {len(df)} rows", flush=True)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int, default=[2023, 2024, 2025, 2026])
    a = ap.parse_args(argv)
    frames = []
    for s in a.seasons:
        df = scrape_season(s)
        if len(df):
            frames.append(df)
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        all_df.to_csv(f"{DIAG}/si_nba_all_seasons.csv", index=False)
        print(f"combined: {len(all_df)} rows -> {DIAG}/si_nba_all_seasons.csv", flush=True)
        print(all_df[["sim_match_id", "date_utc", "away_abv", "home_abv",
                       "home_odds_dec", "away_odds_dec",
                       "final_home_score", "final_away_score"]].head().to_string())


if __name__ == "__main__":
    sys.exit(main())
