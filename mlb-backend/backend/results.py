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
from zoneinfo import ZoneInfo

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
        away_score  float        home_win    float   — 1.0/0.0 once final, else NaN
        is_final    bool    — abstractGameState == 'Final'
        home_team   str     — canonical home abbreviation (StatsAPI
        away_team   str     — hydrated), so the slate's (game_date,
                             home, away) fallback match key can exist
    Empty frame (with the same columns) on any network/parse failure.
    """
    cols = ["game_pk", "game_date", "home_score", "away_score",
            "home_win", "is_final", "home_team", "away_team"]

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
                    # Without this the team object carries NO abbreviation
                    # field, so the slate's (date+teams) fallback match key
                    # could never be built (2026-10-03 defect).
                    "hydrate": "team(abbreviation)",
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
                h_abbr = (home.get("team") or {}).get("abbreviation")
                a_abbr = (away.get("team") or {}).get("abbreviation")
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
                    "home_team": _canon_team(h_abbr) if h_abbr else None,
                    "away_team": _canon_team(a_abbr) if a_abbr else None,
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
            # Matchup/date is NOT unique on doubleheaders. Without a real
            # game_pk, leave ambiguous legs untouched rather than copying
            # game two's final onto game one.
            if key in date_team_lookup:
                date_team_lookup[key] = None
            else:
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
                pd.to_numeric(hs, errors="coerce")
                + pd.to_numeric(as_, errors="coerce"))
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
        has_real_pk = "_pk" in df.columns and pd.notna(row.get("_pk"))
        if r is None and has_dt_cols and not has_real_pk:
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
                # First-pitch provenance: a suspended-and-resumed game's
                # ``gameDate`` is the RESUME slot (days to months after the
                # first pitch — 2026-10-07 deep dive found 8 such rows
                # sealed as "observed" first pitches, mis-keying the market
                # as-of merge, weather sampling and PIT ordering). The
                # schedule carries ``resumedFrom`` = the game's actual
                # first pitch exactly when it was suspended after starting;
                # postponed games (never started) legitimately carry the
                # makeup datetime in gameDate and have no resumedFrom.
                dt = g.get("resumedFrom") or g.get("gameDate")
                if pk and dt:
                    out[int(pk)] = dt
        logger.info("StatsAPI schedule %s→%s: %d games (%d new)",
                    chunk_start, chunk_end,
                    sum(len(d.get("games", [])) for d in data.get("dates", [])),
                    len(out) - n_before)
        chunk_start = chunk_end + timedelta(days=1)
    return out


def refresh_start_times(games: pd.DataFrame) -> pd.DataFrame:
    """Persist authoritative first-pitch UTC for rows carrying placeholders.

    Statcast-derived history is born with load_game_features' fabricated
    19:00-UTC fallback and ``start_time_observed=False``; slate rows born
    from the ESPN schedule carry real times but no flag. This refresh fills
    ``start_time_utc`` from the StatsAPI schedule for every row WITHOUT an
    observed flag and marks only matched rows observed — never overwrites
    an already-observed timestamp, never marks an unmatched row observed.
    The ONE exception (2026-10-07 deep dive): a suspended-and-resumed
    game's sealed "observed" value can be the schedule's RESUME slot rather
    than its first pitch (8 of 7,397 history rows carried slots days to
    months after their game). A resume slot is always strictly AFTER the
    game's official date, so observed timestamps later than their own
    ``game_date`` are reconciled against the authoritative schedule value —
    a legitimate late/midnight start maps back to the same instant and
    changes nothing, a sealed resume slot is restored to the game's real
    first pitch. (Same-day resumes are undetectable by date and keep their
    slot — a documented limitation.) Idempotent: once every row is observed
    and none is later than its game date the schedule is not queried.
    Failure is best-effort: the frame returns unchanged with a warning so
    fabricated placeholders (which the weather paths already know how to
    work around) never kill a run.
    """
    if games is None or games.empty or "game_pk" not in games.columns \
            or "game_date" not in games.columns:
        return games
    pk = pd.to_numeric(games["game_pk"], errors="coerce")
    if "start_time_observed" in games.columns:
        observed = games["start_time_observed"].fillna(False).astype(bool)
    else:
        observed = pd.Series(False, index=games.index)
    need = pk.notna() & ~observed
    # Self-healing provenance repair candidates: an observed timestamp
    # strictly later than its game's own date (ET) cannot be that game's
    # first pitch — it is a suspended game's RESUME slot. Earlier or
    # same-day values are left alone (a resume slot never precedes its
    # game).
    repair = pd.Series(False, index=games.index)
    if "start_time_utc" in games.columns:
        _st = pd.to_datetime(games["start_time_utc"], utc=True, errors="coerce")
        _et = _st.dt.tz_convert("America/New_York")
        _gd = pd.to_datetime(games["game_date"], errors="coerce")
        _later = (_et.dt.tz_localize(None).dt.normalize()
                  > _gd.dt.normalize())
        repair = pk.notna() & observed & _st.notna() & _later.fillna(False)
    if not need.any() and not repair.any():
        return games
    dates = pd.to_datetime(
        games.loc[need | repair, "game_date"], errors="coerce").dropna()
    if dates.empty:
        return games
    try:
        times = fetch_game_start_times(dates.min().date(), dates.max().date())
    except Exception as exc:  # network/auth failures must not kill the run
        logger.warning("Start-time refresh unavailable (%s); fabricated "
                       "placeholders retained", exc)
        return games
    if not times:
        return games
    mapped = pk.map(lambda k: times.get(int(k)) if pd.notna(k) else None)
    parsed = pd.to_datetime(mapped, utc=True, errors="coerce")
    fill = need & parsed.notna()
    # Repair rows adopt the authoritative first pitch ONLY when it differs:
    # a legitimate midnight-start rain delay maps to its own instant and is
    # left untouched, while a sealed resume slot is restored to the game's
    # real first pitch.
    current = (pd.to_datetime(games["start_time_utc"], utc=True,
                              errors="coerce")
               if "start_time_utc" in games.columns
               else pd.Series(pd.NaT, index=games.index))
    fix = repair & parsed.notna() & (parsed != current)
    if not fill.any() and not fix.any():
        logger.warning("Start-time refresh matched 0/%d unobserved games", int(need.sum()))
        return games
    games = games.copy()
    if "start_time_utc" not in games.columns:
        games["start_time_utc"] = pd.NaT
    elif not pd.api.types.is_datetime64_any_dtype(games["start_time_utc"]):
        games["start_time_utc"] = pd.to_datetime(games["start_time_utc"],
                                                  utc=True, errors="coerce")
    games.loc[fill | fix, "start_time_utc"] = parsed[fill | fix]
    if "start_time_observed" not in games.columns:
        games["start_time_observed"] = False
    games.loc[fill, "start_time_observed"] = True
    logger.info("Start-time refresh: %d/%d games now carry observed first "
                "pitches (StatsAPI schedule); %d suspended-game resume "
                "slot(s) restored to their real first pitch",
                int(fill.sum()), len(games), int(fix.sum()))
    return games


def merge_result_cache(cached: pd.DataFrame | None,
                       fresh: pd.DataFrame) -> pd.DataFrame:
    """Merge cached results with a fresh pull, newest/final copy wins."""
    cols = ["game_pk", "game_date", "home_score", "away_score",
            "home_win", "is_final", "home_team", "away_team"]
    if fresh.empty:
        return cached if cached is not None else pd.DataFrame(columns=cols)
    if cached is None or cached.empty:
        return fresh
    both = pd.concat([cached, fresh], ignore_index=True)
    # NaN is_final must sort/compare as NOT final: astype(bool) alone maps
    # NaN -> True, which would let an unverifiable row win the merge and
    # carry stale scores forward as a "final".
    both["is_final"] = both["is_final"].fillna(False).astype(bool)
    # Sort so final rows sort last within each game_pk, then take the last.
    # Stable: within equal finality the concat order (cached, then fresh)
    # decides, so the NEWER copy wins deterministically.
    both = both.sort_values(["game_pk", "is_final"], kind="stable")
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


def _nearest_slate_pk(legs, start_et) -> Optional[int]:
    """Nearest StatsAPI game_pk for a slate row by first-pitch time.

    ``legs`` is [(first_pitch_et, game_pk), ...] for ONE matchup — a
    doubleheader holds two entries (two games, usually two starters). The
    nearest leg within a 4h window IS the row's own game; single-game
    matchups match trivially. Returns None when the matchup is unknown or
    nothing is close (the row falls back to projected lineups). Pure.
    """
    if not legs:
        return None
    target = pd.Timestamp(start_et)
    if pd.isna(target):
        return None
    if target.tzinfo is not None:
        target = target.tz_convert("America/New_York").tz_localize(None)
    best, best_d = None, None
    for t, pk in legs:
        if pd.isna(t):
            continue
        t = t.tz_localize(None) if t.tzinfo is not None else t
        d = abs((t - target).total_seconds())
        if best_d is None or d < best_d:
            best, best_d = pk, d
    if best is not None and best_d is not None and best_d <= 4 * 3600:
        return best
    return None


def _attach_slate_lineup_keys(slate: pd.DataFrame,
                              lineup_rows: pd.DataFrame) -> pd.DataFrame:
    """Carry resolved StatsAPI game identities onto the ESPN slate.

    Posted lineup enrichment is keyed by ``game_pk``. ESPN rows generally
    carry only ``game_id``; keeping this conversion explicit prevents a
    successful StatsAPI lineup fetch from degenerating into all-NaN lineup
    features during the subsequent PIT wOBA join.
    """
    if "game_pk" not in lineup_rows.columns or len(lineup_rows) != len(slate):
        raise ValueError("slate lineup identity rows must align one-to-one")
    out = slate.copy()
    out["game_pk"] = pd.to_numeric(
        lineup_rows["game_pk"], errors="coerce").astype("Int64")
    return out


def _fetch_slate_lineups(slate: pd.DataFrame, target_date: date) -> pd.DataFrame:
    """Attach the 6 lineup-delta columns to today's slate from posted lineups.

    Resolution: StatsAPI schedule for target_date maps (home, away) → game_pk
    (the slate carries no StatsAPI game_pk -- ESPN's game_id only), then the
    live feed per game, paced like the roof fetcher (~2.2 req/s, one retry).
    Games with a complete 9+9 battingOrder get REAL lineup-delta features
    (same point-in-time math as training: batter/team sd-wOBA through games
    strictly before today -- no lookahead).

    Projected fallback for games not yet posted (per the 2026-08-25 posting-
    curve probe, away sides generally post ~2-3h before first pitch; a morning
    slate is mostly projected): ALL SIX columns emit NULL. A lineup not yet
    posted before first pitch is genuinely UNKNOWN at bet time — it must never
    be fabricated as 0 (a fake "projected lineup equals season mean" leaks a
    value the model can't actually have). NULLs route through the existing NaN
    imputation path, so the tree learns "not yet posted" as its own missing
    state — the same strict point-in-time discipline the market-line as-of
    join (data_ingestion._attach_market_lines rejects lines posted at/after
    start) and weather/roof NULLs use. The actual-vs-projected split is logged
    loudly so a projected-only morning is visible.
    """
    if slate is None or slate.empty:
        return slate
    slate = slate.reset_index(drop=True)  # posted-mask aligns by position below
    import requests
    import time as _time
    from results import STATSAPI_SCHEDULE_URL  # same endpoint the weather backfill uses
    _FEED = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"

    # 1) game_pk resolution from the StatsAPI schedule (one day, no chunking).
    # A matchup can hold MULTIPLE games (doubleheader legs with their own
    # game_pk), so collect EVERY leg with its first-pitch time and match per
    # slate row by start time — a one-entry-per-matchup map would feed both
    # legs the first game's lineup.
    pk_by_teams: dict[tuple[str, str], list[tuple[pd.Timestamp, int]]] = {}
    try:
        resp = requests.get(STATSAPI_SCHEDULE_URL,
                            params={"sportId": 1,
                                    "startDate": target_date.isoformat(),
                                    "endDate": target_date.isoformat()},
                            timeout=20)
        resp.raise_for_status()
        for g in (resp.json().get("dates") or [{}])[0].get("games") or []:
            t = (g.get("teams") or {}).get("away") or {}
            h = (g.get("teams") or {}).get("home") or {}
            away = (t.get("team") or {}).get("abbreviation")
            home = (h.get("team") or {}).get("abbreviation")
            if home and away:
                try:
                    gd = pd.Timestamp(g.get("gameDate"))
                    if gd.tzinfo is None:
                        gd = gd.tz_localize("UTC")
                    gd = gd.tz_convert("America/New_York").tz_localize(None)
                except Exception:
                    gd = pd.NaT
                pk_by_teams.setdefault((home, away), []).append((gd, int(g["gamePk"])))
    except Exception as e:
        logger.warning("_fetch_slate_lineups: schedule resolution failed (%s); slate stays projected", e)
        pk_by_teams = {}

    # 2) per-game feed fetch (paced, one retry, cached per run)
    def _feed(pk: int) -> dict | None:
        for attempt in (0, 1):
            try:
                r = requests.get(_FEED.format(pk=pk), timeout=15)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
            if attempt == 0:
                _time.sleep(_LINEUP_PAUSE_SEC * 3)
            _time.sleep(_LINEUP_PAUSE_SEC)
        return None

    def _orders(feed: dict | None) -> tuple[list[int], list[int]]:
        out = []
        if feed:
            bs = ((feed.get("liveData") or {}).get("boxscore") or {})
            teams_bs = bs.get("teams") or {}
            for side in ("home", "away"):
                try:
                    order = [p["person"]["id"]
                             for p in (teams_bs[side].get("battingOrder") or [])]
                except Exception:
                    order = []
                out.append(order)
        return (out[0] if out else [], out[1] if len(out) > 1 else [])

    rows = []
    for _, r in slate.iterrows():
        teams_key = (r.get("home_team"), r.get("away_team"))
        pk = _nearest_slate_pk(pk_by_teams.get(teams_key), r.get("start_time_utc"))
        if pk is None:
            rows.append({"game_pk": pd.NA, "home_order": None, "away_order": None})
            continue
        feed = _feed(pk)
        ho, ao = _orders(feed)
        rows.append({"game_pk": int(pk), "home_order": ho or None,
                     "away_order": ao or None})
    lu = pd.DataFrame(rows)
    # game_pk must be TYPED before any join: rows mix resolved ints with
    # pd.NA (games whose StatsAPI identity has not resolved yet), so the
    # bare list-of-dicts frame lands as object dtype. pandas refuses
    # object-vs-Int64 key merges outright — on 2026-09-22 that ValueError
    # killed the whole slate build and shipped zero dated artifacts for the
    # day (the dashboard fell back to the previous day's files).
    if "game_pk" in lu.columns:
        lu["game_pk"] = pd.to_numeric(lu["game_pk"], errors="coerce").astype("Int64")

    # StatsAPI is the authoritative identity for posted lineups. The ESPN
    # slate normally has only game_id, but add_lineup_delta_features joins
    # both the lineup override and the PIT wOBA caches by game_pk. Carry the
    # resolved key onto the slate before enrichment; otherwise every actual
    # lineup silently misses the join and all six shipped features remain NaN.
    slate = _attach_slate_lineup_keys(slate, lu)

    # 3) real features where both sides posted; projected fallback otherwise
    slate = add_lineup_delta_features(slate, lineups_override=lu)
    from features import LINEUP_DELTA_COLS, LINEUP_TOP5_K
    posted = lu["home_order"].notna() & lu["away_order"].notna()
    n_actual = int(posted.sum())
    for idx, r in slate.iterrows():
        if not posted.iloc[idx]:
            # STRICT POINT-IN-TIME: a lineup not posted before first pitch is
            # UNKNOWN at bet time. Emit NULL for ALL SIX columns — never a
            # fabricated 0 ("projected lineup equals season mean" would inject
            # a value that cannot exist at bet time). NULLs route through the
            # existing imputation path, matching the market-line as-of join
            # and weather/roof missing-observation semantics.
            for c in LINEUP_DELTA_COLS:
                slate.at[idx, c] = pd.NA
    logger.info(
        "slate lineups: %d/%d ACTUAL (both sides posted), %d/%d projected "
        "(not yet posted → all 6 lineup-delta cols NULL, PIT-safe)",
        n_actual, len(slate), len(slate) - n_actual, len(slate))
    return slate


def _count_evening_games(games: Optional[pd.DataFrame]) -> int:
    """Count slate games beginning at/after 7 PM ET (ALL statuses).

    ``start_time_utc`` is stored as a NAIVE UTC datetime. Treat it as UTC
    and convert to America/New_York (zoneinfo, DST-aware) BEFORE comparing
    the hour — never compare the raw UTC hour: a 9:38 PM ET game is 01:38
    UTC the NEXT day, so a UTC-hour comparison drops exactly the west-coast
    night games this badge is meant to count (the UTC-midnight rollover).
    Rows with a missing/NaN start are skipped, never crashing the count.
    """
    if games is None or "start_time_utc" not in games.columns:
        return 0
    starts = pd.to_datetime(games["start_time_utc"], errors="coerce", utc=True)
    valid = starts.notna()
    if not valid.any():
        return 0
    et = starts[valid].dt.tz_convert(ZoneInfo("America/New_York"))
    return int((et.dt.hour >= 19).sum())
