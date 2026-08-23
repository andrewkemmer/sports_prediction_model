"""Official MLB results via the MLB StatsAPI schedule endpoint.

Scores/winners derived from the last cached Statcast pitch are fragile:
partial pitch data freezes wrong finals forever, and mid-game snapshots
become bogus training labels. The StatsAPI schedule endpoint provides
authoritative scores plus an explicit game state per game_pk, so it is
used to OVERRIDE every derived label.

All network calls are fail-safe: on any error the caller keeps whatever
it had before (current behavior) instead of failing the pipeline.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

STATSAPI_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
RESULTS_TAIL_REFRESH_DAYS = 3


def fetch_mlb_results(start_date: date, end_date: date,
                      timeout: int = 20) -> pd.DataFrame:
    """Fetch official results for every game between the given dates.

    Returns a frame with columns:
        game_pk     int64   — matches Statcast game_pk
        game_date   str      — YYYY-MM-DD (ET date of the game)
        home_score  float    — official final runs (NaN until final)
        away_score  float
        home_win    float    — 1.0/0.0 once final, else NaN
        is_final    bool     — abstractGameState == 'Final'
    Empty frame (with the same columns) on any network/parse failure.
    """
    cols = ["game_pk", "game_date", "home_score", "away_score",
            "home_win", "is_final"]
    try:
        resp = requests.get(
            STATSAPI_SCHEDULE_URL,
            params={
                "sportId": 1,
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("StatsAPI results unavailable (%s) — keeping "
                       "pitch-derived scores", exc)
        return pd.DataFrame(columns=cols)

    rows = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            state = (g.get("status", {}).get("abstractGameState") or "")
            is_final = state == "Final"
            home = g.get("teams", {}).get("home", {})
            away = g.get("teams", {}).get("away", {})
            hs = home.get("score")
            as_ = away.get("score")
            if is_final and isinstance(hs, int) and isinstance(as_, int):
                home_score, away_score = float(hs), float(as_)
                home_win = float(home_score > away_score) \
                    if home_score != away_score else None
            else:
                home_score = away_score = home_win = None
            rows.append({
                "game_pk": g.get("gamePk"),
                "game_date": day.get("date"),
                "home_score": home_score,
                "away_score": away_score,
                "home_win": home_win,
                "is_final": is_final,
            })
    return pd.DataFrame(rows, columns=cols)


def apply_official_results(games: pd.DataFrame,
                           results: pd.DataFrame) -> pd.DataFrame:
    """Override pitch-derived scores/labels with official ones.

    - Final games: home_score/away_score/home_win/total_runs replaced with
      the official values (fixes frozen mid-game finals retroactively).
    - Non-final games: home_win set to NaN so a live score snapshot can
      never enter training as an outcome. Scores are kept for display but
      flagged through ``is_official_final = False`` downstream consumers
      may check.
    """
    df = games.copy()
    if results.empty or "game_pk" not in df.columns:
        return df

    res = results.dropna(subset=["game_pk"]).copy()
    res["game_pk"] = res["game_pk"].astype("int64")
    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")

    lookup = res.set_index("game_pk")
    pk = df["game_pk"].map(lambda x: x if pd.notna(x) else None)

    n_fixed = 0
    for idx, key in zip(df.index, pk):
        if key is None or key not in lookup.index:
            continue
        r = lookup.loc[key]
        if bool(r["is_final"]):
            df.at[idx, "home_score"] = r["home_score"]
            df.at[idx, "away_score"] = r["away_score"]
            if pd.notna(r["home_win"]):
                df.at[idx, "home_win"] = r["home_win"]
            df.at[idx, "total_runs"] = (
                (r["home_score"] or 0) + (r["away_score"] or 0))
            n_fixed += 1
        else:
            # Live / postponed / in-progress: never a training outcome.
            df.at[idx, "home_win"] = None
    if n_fixed:
        logger.info("Official results applied: %d finals verified/corrected",
                    n_fixed)
    return df


def merge_result_cache(cached: pd.DataFrame | None,
                       fresh: pd.DataFrame) -> pd.DataFrame:
    """Merge cached results with a fresh pull, newest/final copy wins."""
    cols = ["game_pk", "game_date", "home_score", "away_score",
            "home_win", "is_final"]
    if fresh.empty:
        return cached if cached is not None else pd.DataFrame(columns=cols)
    if cached is None or cached.empty:
        return fresh
    both = pd.concat([cached, fresh], ignore_index=True)
    both["is_final"] = both["is_final"].astype(bool)
    # Sort so final rows sort last within each game_pk, then take the last.
    both = both.sort_values(["game_pk", "is_final"])
    return both.groupby("game_pk", as_index=False).last()


def refresh_range(cache_start: date, cache_end: date,
                  requested_end: date) -> tuple[date, date]:
    """Dates to re-fetch: everything requested, widening back a tail window
    over recent days so late-finishing games get their true finals."""
    tail_start = max(cache_start, requested_end -
                     timedelta(days=RESULTS_TAIL_REFRESH_DAYS - 1))
    return tail_start, max(requested_end, cache_end)
