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
# The schedule endpoint truncates responses for long ranges (observed:
# a 20-month request returned only ~8.5 months). Stay well under the
# cutoff — same convention as the Statcast chunked pulls.
SCHEDULE_CHUNK_DAYS = 60
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

    # The schedule endpoint SILENTLY TRUNCATES long date ranges: a full
    # season-pair query returned ~3,000 games vs ~5,900 fetched per year —
    # 2,800 finals were simply absent, so the overlay 'succeeded' while
    # half the history stayed uncorrected. Fetch in ≤1-year chunks.
    chunks: list[tuple[date, date]] = []
    cur = start_date
    while cur <= end_date:
        chunk_end = min(date(cur.year, 12, 31), end_date)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)

    rows = []
    for c_start, c_end in chunks:
        try:
            resp = requests.get(
                STATSAPI_SCHEDULE_URL,
                params={
                    "sportId": 1,
                    "startDate": c_start.isoformat(),
                    "endDate": c_end.isoformat(),
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("StatsAPI results unavailable (%s–%s): %s — "
                           "keeping pitch-derived scores for this chunk",
                           c_start, c_end, exc)
            continue

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
    return _dedupe_prefer_scored(pd.DataFrame(rows, columns=cols))


_TEAM_ALIASES = {"CHW": "CWS", "OAK": "ATH", "ARI": "AZ"}


def _dedupe_prefer_scored(df: pd.DataFrame) -> pd.DataFrame:
    """One row per game_pk, preferring the listing WITH scores.

    Suspended/resumed games appear under multiple dates; the original
    date's listing can be Final-with-no-score while the completion date
    carries the real result. Keeping 'last' blindly could retain the
    scoreless row and blind the overlay to the game entirely.
    """
    if df.empty:
        return df
    df = df.copy()
    df["_scored"] = df["home_win"].notna() & df["home_score"].notna()
    df = (df.sort_values(["game_pk", "_scored"])
            .drop_duplicates(subset=["game_pk"], keep="last")
            .drop(columns=["_scored"]))
    return df.reset_index(drop=True)


def _canon_team(code) -> str:
    try:
        return _TEAM_ALIASES.get(str(code).strip().upper(), str(code).strip())
    except Exception:
        return str(code)


def apply_official_results(games: pd.DataFrame,
                           results: pd.DataFrame) -> pd.DataFrame:
    """Override pitch-derived scores/labels with official ones.

    - Final games: home_score/away_score/home_win/total_runs replaced with
      the official values (fixes frozen mid-game finals retroactively).
    - Non-final games: home_win set to NaN so a live/preview score snapshot
      can NEVER enter training or artifacts as an outcome. Scores are kept
      for display only.

    Rows are matched by StatsAPI ``game_pk`` when present; rows without a
    game_pk (e.g. the ESPN-built upcoming slate) fall back to a match on
    (game_date, home_team, away_team) with canonical team codes.
    """
    df = games.copy()
    if results.empty or results.columns.intersection(
            ["home_score", "home_win", "is_final"]).empty:
        return df

    res = results.dropna(subset=["is_final"]).copy()
    if res.empty:
        return df

    # Match key 1: StatsAPI game_pk (history frames from Statcast).
    # The schedule endpoint can list the same game under multiple dates
    # (postponements/resumptions), producing duplicate game_pk rows that
    # crash set_index().to_dict('index') — keep one row per game.
    pk_lookup = {}
    if "game_pk" in df.columns:
        pk_res = _dedupe_prefer_scored(res.dropna(subset=["game_pk"]).copy())
        pk_res["game_pk"] = pk_res["game_pk"].astype("int64")
        pk_lookup = pk_res.set_index("game_pk").to_dict("index")
        df["_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")

    # Match key 2: (game_date, canonical home/away) for frames without pk.
    date_team_lookup = {}
    has_dt_cols = all(c in df.columns for c in ("game_date", "home_team", "away_team"))
    if has_dt_cols:
        for _, r in res.iterrows():
            d = pd.to_datetime(r.get("game_date"), errors="coerce")
            if pd.isna(d):
                continue
            key = (str(d.date()), _canon_team(r.get("home_team")),
                   _canon_team(r.get("away_team")))
            date_team_lookup[key] = r

    def _apply(row_idx, r) -> None:
        if bool(r["is_final"]):
            hs, as_ = r.get("home_score"), r.get("away_score")
            if pd.notna(hs) and pd.notna(as_):
                df.at[row_idx, "home_score"] = hs
                df.at[row_idx, "away_score"] = as_
            if pd.notna(r.get("home_win")):
                df.at[row_idx, "home_win"] = r["home_win"]
            df.at[row_idx, "total_runs"] = (
                (pd.to_numeric(r.get("home_score"), errors="coerce") or 0)
                + (pd.to_numeric(r.get("away_score"), errors="coerce") or 0))
            if "game_state" in df.columns:
                df.at[row_idx, "game_state"] = "post"
        else:
            # Live / postponed / in-progress / preview: never an outcome.
            # game_state is left untouched (ESPN already supplies pre/in).
            df.at[row_idx, "home_win"] = None

    n_fixed = 0
    for idx, row in df.iterrows():
        r = None
        if "_pk" in df.columns and pd.notna(row.get("_pk")):
            pk = int(row["_pk"])
            if pk in pk_lookup:
                r = pk_lookup[pk]
        if r is None and has_dt_cols:
            d = pd.to_datetime(row.get("game_date"), errors="coerce")
            if pd.notna(d):
                key = (str(d.date()), _canon_team(row.get("home_team")),
                       _canon_team(row.get("away_team")))
                r = date_team_lookup.get(key)
        if r is not None:
            was = df.at[idx, "home_win"] if "home_win" in df.columns else None
            _apply(idx, r)
            if not (pd.isna(was) and pd.isna(df.at[idx, "home_win"])):
                n_fixed += 1
    if "_pk" in df.columns:
        df = df.drop(columns=["_pk"])
    if n_fixed:
        logger.info("Official results applied: %d games verified/corrected",
                    n_fixed)
    return df


def fetch_game_start_times(start_date: date, end_date: date,
                           timeout: int = 20) -> dict[int, str]:
    """Authoritative first-pitch UTC timestamps from the StatsAPI schedule.

    Maps StatsAPI ``gamePk`` → ISO-8601 UTC datetime string.  Used by the
    weather backfill: Statcast-derived history carries only fabricated
    19:00-UTC placeholders, and weather must be sampled strictly before the
    REAL first pitch to stay point-in-time honest.  Empty dict on failure.

    The schedule endpoint SILENTLY TRUNCATES long date ranges (a single
    2025-01-01→2026-08-23 request returns only 2025-02-20→2025-11-01),
    which starved every post-truncation game of a start time and left the
    weather features null for an entire season while every log line looked
    healthy. Query in bounded chunks and merge so coverage is complete.
    """
    out: dict[int, str] = {}
    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=SCHEDULE_CHUNK_DAYS - 1),
                        end_date)
        try:
            resp = requests.get(
                STATSAPI_SCHEDULE_URL,
                params={
                    "sportId": 1,
                    "startDate": chunk_start.isoformat(),
                    "endDate": chunk_end.isoformat(),
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("StatsAPI start times unavailable for %s→%s (%s)",
                           chunk_start, chunk_end, exc)
            chunk_start = chunk_end + timedelta(days=1)
            continue
        n_before = len(out)
        for day in data.get("dates", []):
            for g in day.get("games", []):
                pk = g.get("gamePk")
                dt = g.get("gameDate")
                if pk and dt:
                    out[int(pk)] = dt
        logger.info("StatsAPI schedule %s→%s: %d games (%d new)",
                    chunk_start, chunk_end,
                    sum(len(d.get("games", [])) for d in data.get("dates", [])),
                    len(out) - n_before)
        chunk_start = chunk_end + timedelta(days=1)
    return out


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


# ── Official park weather (StatsAPI game feed) ─────────────────────────────

STATSAPI_GAME_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"


def fetch_statsapi_weather(
    game_pks,
    timeout: int = 15,
    pause_sec: float = 0.4,
) -> dict[int, dict]:
    """Official park-reported weather from each game's live-feed endpoint.

    ``gameData.weather`` carries the stadium's OWN observation:
    ``{"condition": str, "temp": F, "wind": "9 mph, In from CF"}``. This is
    the gap-filler for games the Open-Meteo archive could not observe — it
    needs no coordinates and no first-pitch time, only the game_pk.

    Paced one request per ``pause_sec`` with a single retry per game.
    Returns ``{pk: {"temp_f", "wind_mph", "wind_text"}}``; failed or
    weather-less games are simply absent from the result (loudly counted).
    """
    import re
    import time as _time

    out: dict[int, dict] = {}
    pks = [int(pk) for pk in game_pks]
    failures = 0
    for i, pk in enumerate(pks):
        payload = None
        for attempt in (1, 2):
            try:
                resp = requests.get(
                    STATSAPI_GAME_FEED_URL.format(pk=pk), timeout=timeout)
                if resp.status_code == 200:
                    payload = resp.json()
                    break
                failures += 0  # non-200 retried once below
            except requests.RequestException:
                pass
            if attempt == 1:
                _time.sleep(pause_sec * 2)
        wx = ((payload or {}).get("gameData") or {}).get("weather") or {}
        wind_text = str(wx.get("wind") or "")
        mph_m = re.search(r"(\d+(?:\.\d+)?)\s*mph", wind_text, re.IGNORECASE)
        temp_f = wx.get("temp")
        if temp_f is None and mph_m is None:
            failures += 1
            continue
        try:
            temp_f = float(temp_f) if temp_f is not None else None
        except (TypeError, ValueError):
            temp_f = None
        out[pk] = {
            "temp_f": temp_f,
            "wind_mph": float(mph_m.group(1)) if mph_m else None,
            "wind_text": wind_text,
            # Roof state rides in the SAME condition field parks use:
            # "Roof Closed" / "Roof Open" / "Indoor" at retractable venues.
            "condition": (str(wx.get("condition") or "") or None),
        }
        if i + 1 < len(pks):
            _time.sleep(pause_sec)
    if pks:
        logger.info(
            "StatsAPI game-feed weather: %d/%d games reported conditions "
            "(%d unusable)", len(out), len(pks), failures)
    return out
