"""Maintained home-plate umpire data access for the pipeline (2026-08-27).

Purpose
-------
Preserve the umpire data-access capability scoped on 2026-08-27: per-game
home-plate umpire IDs are a 2-request MLB StatsAPI sweep (one request per
season via ``schedule?hydrate=officials``) with 100% coverage of the
4,481-game frame. This module is the maintained, incremental component:
the cumulative map is refreshed run-over-run (new seasons appended,
existing rows never re-fetched) and survives Phase 6 cleanup under the
protected name ``umpire_map.csv``.

PINNED VERDICT 2026-08-27 (scoping: data_delivery/umpire_scoping_20260827.json)
-------------------------------------------------------------------------
DON'T ADOPT the runs-based umpire tendency as a model feature. First-cut
regressions (total runs / P(over 8.5) on the umpire's within-season
trailing runs/game, controlling for the run engine's lambda sum):
2025 beta = -0.172 (t = -1.58, R^2 gain +0.0016, AUC gain -0.0000);
2026 beta = +0.090 (t = +0.79, R^2 gain +0.0006, AUC gain +0.0024). The
2025-vs-2026 "compression ratio" (-0.52) is not statistically meaningful
(both legs sub-significant; the sign is unstable across window variants —
an expanding-window build gives negative betas in BOTH seasons). The
umpire-identity ceiling on runs variance is only ~3-5%. NO material
signal in either era -> the 45-fold ablation gate was never warranted.

This module is the DATA-ACCESS component ONLY. Do NOT wire any umpire
column into the run engine's 29-feature regression or the moneyline
ensemble — the verdict above stands. The deferred called-strike-rate
variant (park-independent, on non-borderline pitches) can later be built
from ``umpire_stats.csv``'s called-pitch placeholder columns once
zone/description pitch data is pulled (the savant ``umpire`` field is
null in the installed pybaseball; StatsAPI schedule is the umpire source).

Schema (data_delivery/umpire_map.csv, maintained in place):
    game_pk, game_date, home_team, away_team, hp_umpire_id,
    hp_umpire_name, season
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config import DATA_DELIVERY_DIR

logger = logging.getLogger(__name__)

STATSAPI_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
HYDRATE = "officials"
# SportId=1 is MLB. The schedule endpoint can silently truncate very long
# ranges (see results.py), so we fetch ONE calendar year per request — the
# task's "one request per season" — which is verified complete live.
SPORT_ID = 1
SEASON_START_MONTH_DAY = (2, 1)   # before spring training (all game types kept)
SEASON_END_MONTH_DAY = (12, 31)

MAP_COLS = ["game_pk", "game_date", "home_team", "away_team",
            "hp_umpire_id", "hp_umpire_name", "season"]
MAP_FILENAME = "umpire_map.csv"
STATS_FILENAME = "umpire_stats.csv"
MIN_FIRST_SEASON = 2025          # the model frame starts here

# ── Retry discipline (same as weather.py) ───────────────────────────────────
_RETRIABLE_STATUSES = frozenset({429, 502, 503, 504})
_BACKOFF_BASE_SEC = 2.0
_MAX_JITTER_SEC = 0.5
_DEFAULT_ATTEMPTS = 5


def _get_with_retry(url: str, params: dict, attempts: int = _DEFAULT_ATTEMPTS,
                    timeout: int = 20) -> requests.Response:
    """GET with exponential backoff and server-directed retry delays.

    Retries only on transient statuses (429/502/503/504); each retry waits
    ``2**attempt`` + jitter, or the API's Retry-After when present. Network
    exceptions are retried with the same backoff and re-raised if every
    attempt fails. Non-retriable statuses return immediately (fail fast).
    """
    last_exc = None
    resp = None
    for attempt in range(attempts):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code in _RETRIABLE_STATUSES and attempt < attempts - 1:
                wait = (_BACKOFF_BASE_SEC ** attempt) + random.uniform(
                    0, _MAX_JITTER_SEC)
                try:
                    retry_after = float(resp.headers.get("Retry-After", ""))
                    wait = max(wait, retry_after)
                except (AttributeError, TypeError, ValueError):
                    pass
                logger.warning(
                    "StatsAPI %d for %s — retrying in %.1fs (%d/%d)",
                    resp.status_code, url.split("/")[-1], wait,
                    attempt + 1, attempts - 1,
                )
                time.sleep(wait)
                continue
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep((_BACKOFF_BASE_SEC ** attempt) + random.uniform(
                    0, _MAX_JITTER_SEC))
    if last_exc is not None:
        raise last_exc
    assert resp is not None
    return resp


# ── Parsing (pure; unit-tested) ─────────────────────────────────────────────

def parse_schedule_officials(payload: dict) -> list[dict]:
    """Extract one row per game (regular + postseason) from a schedule payload.

    The home-plate umpire is the official whose ``officialType`` is
    ``"Home Plate"`` (person id + fullName). Resilient: a game with
    missing/empty officials, or a crew without a Home Plate entry, yields
    a row with null umpire fields — never a crash, never a dropped game.
    """
    rows: list[dict] = []
    dates = payload.get("dates") or []
    for day in dates:
        for game in day.get("games") or []:
            # Keep ALL game types: the model frame includes the 2025
            # postseason (W/D/L/F games, all with assigned crews), so an
            # R-only filter would drop those games from the map.
            try:
                game_pk = int(game["gamePk"])
            except (KeyError, TypeError, ValueError):
                continue
            teams = game.get("teams") or {}
            home = (teams.get("home") or {}).get("team") or {}
            away = (teams.get("away") or {}).get("team") or {}
            # Live schedule payloads carry NO team ``abbreviation`` field
            # (verified 2026-08-27: ``teams.*.team`` = {id, name, link}), so
            # fall back to the full name — the columns must never be blank.
            home_team = home.get("abbreviation") or home.get("name")
            away_team = away.get("abbreviation") or away.get("name")
            ump_id = None
            ump_name = None
            for off in game.get("officials") or []:
                if "Home Plate" in (off.get("officialType") or ""):
                    person = off.get("official") or {}
                    ump_id = person.get("id")
                    ump_name = person.get("fullName")
                    break
            rows.append({
                "game_pk": game_pk,
                "game_date": str(game.get("officialDate") or "")[:10],
                "home_team": home_team,
                "away_team": away_team,
                "hp_umpire_id": ump_id,
                "hp_umpire_name": ump_name,
                "season": int(str(game.get("season") or 0) or 0) or None,
            })
    return rows


def _season_bounds(season: int) -> tuple[date, date]:
    return (date(season, *SEASON_START_MONTH_DAY),
            date(season, *SEASON_END_MONTH_DAY))


def fetch_season_umpires(season: int, timeout: int = 60) -> pd.DataFrame:
    """Fetch every regular-season game's HP umpire for one season.

    One request (the schedule endpoint returns a full calendar year —
    verified live: 2025 -> 2,844 games, 2026 -> 2,973, no truncation).
    Returns the MAP_COLS frame; empty frame with the same schema on any
    fetch/parse failure (fail-safe, like results.py).
    """
    start, end = _season_bounds(season)
    try:
        resp = _get_with_retry(
            STATSAPI_SCHEDULE_URL,
            params={"sportId": SPORT_ID, "startDate": start.isoformat(),
                    "endDate": end.isoformat(), "hydrate": HYDRATE},
            timeout=timeout,
        )
        if not resp.ok:
            logger.warning("StatsAPI schedule %d failed: %s",
                           season, resp.status_code)
            return pd.DataFrame(columns=MAP_COLS)
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 — fail-safe like results.py
        logger.warning("StatsAPI schedule %d fetch failed: %s", season, exc)
        return pd.DataFrame(columns=MAP_COLS)

    rows = parse_schedule_officials(payload)
    if rows:
        # sanity: a full MLB regular season is ~2,300+ games; a far-smaller
        # payload suggests the endpoint truncated the range.
        if len(rows) < 1500:
            logger.warning("StatsAPI schedule %d returned only %d games — "
                           "possible truncation", season, len(rows))
        for r in rows:
            r["season"] = season
    return pd.DataFrame(rows, columns=MAP_COLS)


# ── Maintained cumulative map ───────────────────────────────────────────────

def map_path(base: Optional[Path] = None) -> Path:
    return (base or DATA_DELIVERY_DIR) / MAP_FILENAME


def load_umpire_map(base: Optional[Path] = None) -> pd.DataFrame:
    p = map_path(base)
    if not p.exists():
        return pd.DataFrame(columns=MAP_COLS)
    try:
        df = pd.read_csv(p, dtype={"game_pk": "int64"})
        for c in MAP_COLS:
            if c not in df.columns:
                df[c] = None
        return df[MAP_COLS]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s (starting empty): %s", p, exc)
        return pd.DataFrame(columns=MAP_COLS)


def seasons_present(df: pd.DataFrame) -> set[int]:
    s = pd.to_numeric(df.get("season"), errors="coerce").dropna()
    return {int(x) for x in s.unique()}


def maintain_umpire_map(target_date: date, base: Optional[Path] = None,
                       required_games: Optional[pd.DataFrame] = None) -> dict:
    """Incrementally refresh the cumulative umpire map.

    Two refresh passes, both append-only and fail-safe:

    1. Season cache: seasons already present in the map are NEVER re-fetched;
       only missing seasons (MIN_FIRST_SEASON .. target year) are pulled and
       appended by game_pk. A failed fetch logs a warning and keeps whatever
       the map already had — never crashes, never wipes.

    2. Frame gap-fill (``required_games``, the decided game frame with
       game_pk + game_date): games decided AFTER a season was first fetched
       (or crews assigned after the initial pull) would otherwise never
       appear, so the map would go stale run-over-run. When the frame has
       game_pks missing from the map, re-fetch ONLY the affected seasons —
       at most one request per season with a gap, zero requests in steady
       state (no gap -> no fetch). Existing rows are never overwritten;
       a null-umpire row whose crew was assigned later is patched from the
       fresh payload.

    Returns a stats dict (total_rows, n_umpires, seasons_fetched,
    rows_added, gap_filled_seasons, errors).
    """
    df = load_umpire_map(base)
    present = seasons_present(df)
    want = [s for s in range(MIN_FIRST_SEASON, target_date.year + 1)
            if s not in present]

    fetched: list[str] = []
    added = 0
    errors: list[str] = []
    for season in want:
        try:
            season_df = fetch_season_umpires(season)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Umpire season %d fetch raised (map unchanged): %s",
                           season, exc)
            errors.append(str(season))
            continue
        if season_df.empty:
            # Empty = transient fetch issue; leave the season unpresent so
            # the next run retries it (a real season always returns rows).
            logger.warning("Umpire season %d returned no rows", season)
            errors.append(str(season))
            continue
        before = set(df["game_pk"])
        fresh = season_df[~season_df["game_pk"].isin(before)]
        if df.empty:
            df = fresh
        else:
            df = pd.concat([df, fresh], ignore_index=True)
        added += len(fresh)
        fetched.append(str(season))
        logger.info("Umpire season %d: %d new games (total %d)",
                    season, len(fresh), len(df))

    gap_filled: list[str] = []
    if required_games is not None and not df.empty:
        req = required_games.copy()
        req["game_pk"] = pd.to_numeric(req["game_pk"], errors="coerce")
        req["season"] = pd.to_numeric(
            pd.to_datetime(req.get("game_date"), errors="coerce").dt.year,
            errors="coerce",
        )
        req = req.dropna(subset=["game_pk", "season"])
        present_pks = set(df["game_pk"].astype(int))
        gap = req[~req["game_pk"].astype(int).isin(present_pks)]
        if not gap.empty:
            need_seasons = sorted({int(s) for s in gap["season"].unique()})
            for season in need_seasons:
                try:
                    season_df = fetch_season_umpires(season)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Umpire gap-fill %d failed: %s", season, exc)
                    errors.append(f"gap-{season}")
                    continue
                if season_df.empty:
                    errors.append(f"gap-{season}")
                    continue
                season_df = season_df.drop_duplicates("game_pk", keep="first")
                patch = season_df.set_index("game_pk", drop=False)
                # Patch null-umpire rows (crew assigned after the initial
                # pull) from the fresh payload — never touches non-null rows.
                nulls = df["hp_umpire_id"].isna() & \
                    df["game_pk"].isin(patch.index)
                for idx in df.index[nulls]:
                    row = patch.loc[df.at[idx, "game_pk"]]
                    if pd.notna(row["hp_umpire_id"]):
                        for c in MAP_COLS:
                            df.at[idx, c] = row[c]
                fresh = season_df[~season_df["game_pk"].isin(present_pks)]
                if not fresh.empty:
                    df = pd.concat([df, fresh], ignore_index=True)
                    added += len(fresh)
                gap_filled.append(str(season))
                logger.info(
                    "Umpire gap-fill %d: appended %d, patched %d (map %d rows)",
                    season, len(fresh), int(nulls.sum()), len(df),
                )

    if (fetched or gap_filled) and not df.empty:
        df = df.sort_values(["season", "game_pk"]).drop_duplicates(
            "game_pk", keep="first").reset_index(drop=True)
        out = map_path(base)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

    n_ump = (pd.to_numeric(df.get("hp_umpire_id"), errors="coerce")
             .dropna().nunique())
    return {
        "total_rows": int(len(df)),
        "n_umpires": int(n_ump),
        "seasons_fetched": fetched,
        "rows_added": int(added),
        "gap_filled_seasons": gap_filled,
        "errors": errors,
    }


# ── Per-umpire aggregates (DIAGNOSTIC — never a model feature) ──────────────

_STATS_COLS = ["hp_umpire_id", "hp_umpire_name", "season", "games_worked",
               "mean_runs_game", "trailing_runs_game",
               "called_pitch_n", "called_strike_rate"]


def build_umpire_stats(map_df: pd.DataFrame, games: Optional[pd.DataFrame] = None,
                       base: Optional[Path] = None) -> pd.DataFrame:
    """Lightweight per-umpire diagnostics table (maintained in place).

    Per (umpire, season): games worked, mean runs/game, trailing runs/game
    over the umpire's last 25 worked games, and called-pitch placeholders
    (null until zone/description pitch data is wired). Diagnostic only —
    the runs-proxy verdict (module docstring) forbids wiring these into the
    run engine or moneyline. Games without a resolved umpire or without
    runs are excluded from the run aggregates.
    """
    m = map_df.copy()
    m["hp_umpire_id"] = pd.to_numeric(m.get("hp_umpire_id"), errors="coerce")
    m = m.dropna(subset=["hp_umpire_id"])
    if games is not None and "total_runs" in games.columns:
        g = games[["game_pk", "total_runs"]].copy()
        g["game_pk"] = pd.to_numeric(g["game_pk"], errors="coerce")
        m = m.merge(g, on="game_pk", how="left")
    m["runs"] = pd.to_numeric(m.get("total_runs"), errors="coerce")

    rows: list[dict] = []
    for (ump, season), g in m.groupby(["hp_umpire_id", "season"], sort=False):
        g = g.sort_values("game_date").reset_index(drop=True)
        has_runs = g["runs"].dropna()
        mean_runs = float(has_runs.mean()) if len(has_runs) else None
        trail = has_runs.tail(25)
        trail_runs = float(trail.mean()) if len(trail) else None
        name = g["hp_umpire_name"].dropna().iloc[0] \
            if g["hp_umpire_name"].notna().any() else None
        rows.append({
            "hp_umpire_id": int(ump),
            "hp_umpire_name": name,
            "season": int(season),
            "games_worked": int(len(g)),
            "mean_runs_game": mean_runs,
            "trailing_runs_game": trail_runs,
            # called-pitch placeholders: null until zone/description data
            # is pulled for the stable-call variant (see module docstring)
            "called_pitch_n": None,
            "called_strike_rate": None,
        })
    out = pd.DataFrame(rows, columns=_STATS_COLS).sort_values(
        ["season", "hp_umpire_id"]).reset_index(drop=True)
    p = (base or DATA_DELIVERY_DIR) / STATS_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return out


def write_scoping_record(record: dict, base: Optional[Path] = None) -> Path:
    """Write the 2026-08-27 scoping record (verdict pinned for review)."""
    out = (base or DATA_DELIVERY_DIR) / "umpire_scoping_20260827.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return out
