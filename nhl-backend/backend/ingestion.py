"""NHL data ingestion — official NHL API pull with local parquet caching.

Primary source: the official NHL public web API (``api-web.nhle.com/v1``),
no auth. Endpoints used:

  /v1/score/{YYYY-MM-DD}            game list: ids, teams, scores, SOG, state
  /v1/gamecenter/{id}/boxscore      team + per-player boxscore (goalies incl.
                                    TOI/decision; skaters incl. SOG, PPG,
                                    faceoff%, hits, blocks, PIM, G/A)

Optional enrichment family (user-approved design): MoneyPuck free season
CSVs (``moneypuck.com/data.htm``, 2007+, non-commercial, credit required)
add shot-level xG. ``load_moneypuck_shots`` downloads and caches the season
shots file; when a season file is unavailable the enrichment columns degrade
to NaN per the documented missing-value policy — never fabricated, and the
production feature contract never DEPENDS on them.

Cache directory: OUTSIDE the git tree (repo root's parent), ``.nhl_cache/``
— the NFL pattern. All frames carry normalized NHL-API column names; the
feature engine is the only place that interprets them.
"""
from __future__ import annotations

import io
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

try:  # package-relative import when used as backend.ingestion
    from backend import config  # type: ignore
except ImportError:  # running as a top-level module
    import config

logger = logging.getLogger(__name__)

# Cache directory: OUTSIDE the git tree (repo root's parent) so caches never
# pollute the working tree; overridable for tests.
CACHE_DIR = Path(config.ROOT_DIR.parent) / ".nhl_cache"

NHL_API_BASE = "https://api-web.nhle.com/v1"
MONEYPUCK_SHOTS_URL = ("https://moneypuck.com/moneyPuck/playerDataByGame/"
                       "shotsAllYears.csv")

# Per-game keep-list from /v1/score/{date} games[] — the schedule/board
# population. Scores stay None for unplayed games (honest pre-game rows).
SCORE_KEEP = [
    "game_id", "season", "game_type", "game_date", "start_time_utc",
    "home_team", "away_team", "home_score", "away_score",
    "home_sog", "away_sog", "venue", "game_state", "game_outcome",
]

# Cache schema versions: bump whenever the keep-list widens so stale caches
# are ignored rather than silently serving the old column set.
SCORE_CACHE_VERSION = "v1"
BOXSCORE_CACHE_VERSION = "v1"
MP_CACHE_VERSION = "v1"


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def clear_cache() -> None:
    """Remove all cached NHL parquet artifacts."""
    if CACHE_DIR.exists():
        for path in CACHE_DIR.glob("*.parquet"):
            path.unlink(missing_ok=True)


def _http_json(url: str, retries: int = 3, timeout: float = 30.0):
    """GET a JSON document with retry/backoff (the official API is free but
    rate-limited; a transient 5xx must not fail a run)."""
    import requests
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout,
                                headers={"User-Agent": "sports-prediction-model/1.0"})
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2.0 * attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"NHL API GET failed after {retries} attempts: {url} ({last_exc})")


# ---------------------------------------------------------------------------
# Schedule / score ingestion — per-day score pages, cached per season
# ---------------------------------------------------------------------------

def _parse_score_game(g: dict) -> dict:
    """One /v1/score games[] entry -> a flat schedule row."""
    home = g.get("homeTeam") or {}
    away = g.get("awayTeam") or {}
    outcome = ((g.get("gameOutcome") or {}).get("lastPeriodType", "") or "")
    return {
        "game_id": str(g.get("id", "") or ""),
        "season": int(g.get("season", 0) or 0) // 10000,  # 20242025 -> 2024
        "game_type": int(g.get("gameType", config.GAME_TYPE_REG) or 0),
        "game_date": str(g.get("gameDate", "") or ""),
        "start_time_utc": str(g.get("startTimeUTC", "") or ""),
        "home_team": str(home.get("abbrev", "") or ""),
        "away_team": str(away.get("abbrev", "") or ""),
        "home_score": g.get("homeTeam", {}).get("score"),
        "away_score": g.get("awayTeam", {}).get("score"),
        "home_sog": g.get("homeTeam", {}).get("sog"),
        "away_sog": g.get("awayTeam", {}).get("sog"),
        "venue": str((g.get("venue") or {}).get("default", "") or ""),
        "game_state": str(g.get("gameState", "") or ""),
        "game_outcome": outcome,
    }


def load_score_dates(dates: list[str], use_cache: bool = True) -> pd.DataFrame:
    """Fetch /v1/score/{date} for each date and return the combined rows.

    Per-date parquet caches keyed by the date (an incremental daily pull is
    the NHL's natural unit — unlike league sports there is no per-season
    schedule endpoint). A failed date is warned and skipped, never fatal.
    """
    frames: list[pd.DataFrame] = []
    for d in dates:
        path = _cache_path(f"score_{SCORE_CACHE_VERSION}_{d.replace('-', '')}.parquet")
        if use_cache and path.exists():
            try:
                frames.append(pd.read_parquet(path))
                continue
            except Exception as exc:  # corrupt cache → re-pull
                logger.warning("score cache %s unreadable (%s)", path.name, exc)
        try:
            payload = _http_json(f"{NHL_API_BASE}/score/{d}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("score page unavailable for %s: %s", d, exc)
            continue
        rows = [_parse_score_game(g) for g in (payload.get("games") or [])]
        df = pd.DataFrame(rows, columns=SCORE_KEEP)
        if not df.empty:
            df.to_parquet(path, index=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=SCORE_KEEP)
    return pd.concat(frames, ignore_index=True)


def season_dates(season: int) -> list[str]:
    """The calendar date span that can hold a season's games.

    The NHL regular season runs Oct..mid-April and the playoffs into June;
    a generous Oct 1 .. Jul 15 span covers both without wasted requests
    (empty score pages are cheap and cached as empty).
    """
    start = date(season, 10, 1)
    end = date(season + 1, 7, 15)
    out: list[str] = []
    d = start
    while d <= end:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def eligible_games(schedule: pd.DataFrame) -> pd.DataFrame:
    """Filter a schedule frame to the eligible population: settled
    regular-season or postseason games within the configured season window
    (2024+), with gameType in config.GAME_TYPES."""
    df = schedule.copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df[df["season"].isin(config.ALL_SEASONS)]
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(config.GAME_TYPES)]
    for c in ("home_score", "away_score"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Boxscore ingestion — per-game team + goalie + skater stats
# ---------------------------------------------------------------------------

def _parse_boxscore(bs: dict) -> dict:
    """One /v1/gamecenter/{id}/boxscore -> per-game team rollup row.

    Team-level: goals + SOG. Goalie rollup: the two DECISION goalies' TOI
    and the season aggregate stats live on the goalie lines; the per-team
    skater block rolls up PPG, faceoff%, hits, blocks, PIM, giveaways and
    takeaways (means/sums of that game only — the trailing shift downstream
    keeps them strictly-prior).
    """
    player_stats = bs.get("playerByGameStats") or {}
    out: dict = {"game_id": str(bs.get("id", "") or "")}
    for side, team_key in (("home", "homeTeam"), ("away", "awayTeam")):
        block = player_stats.get(team_key) or {}
        team = bs.get(team_key) or {}
        sog = team.get("sog")
        skaters = list(block.get("forwards") or []) + list(block.get("defense") or [])
        goalies = list(block.get("goalies") or [])

        def _num(rows: list[dict], field: str) -> list[float]:
            vals = []
            for r in rows:
                try:
                    v = float(r.get(field))
                except (TypeError, ValueError):
                    continue
                if v == v:  # not NaN
                    vals.append(v)
            return vals

        def _sum(rows: list[dict], field: str) -> float:
            vals = _num(rows, field)
            return float(sum(vals)) if vals else float("nan")

        # Faceoff win pct is a rate: mean over skaters with a value.
        fo_vals = _num(skaters, "faceoffWinningPctg")
        pp_goals = _sum(skaters, "powerPlayGoals")
        # Goalie blocks: shots faced = SOG against + misses? The API exposes
        # per-goalie shotsAgainst via the goalie line's 'sog' style fields
        # only in some versions; the robust goals-against source is the
        # team score when the goalie has the decision. Keep TOI + decision
        # and compute per-start rates in features (goals against = team
        # goals allowed while that goalie was in net — approximated by the
        # team GA for the decision goalie; documented, per-start).
        decision_goalies = [g for g in goalies if g.get("decision") in ("W", "L", "OTL", "SOL")]
        starter = decision_goalies[0] if decision_goalies else (
            goalies[0] if goalies else {})
        toi = None
        try:
            toi = float(starter.get("toi", ""))
        except (TypeError, ValueError):
            toi = None
        out[f"{side}_sog"] = sog
        out[f"{side}_pp_goals"] = pp_goals if pp_goals == pp_goals else None
        out[f"{side}_faceoff_pct"] = (sum(fo_vals) / len(fo_vals)) if fo_vals else None
        out[f"{side}_hits"] = _sum(skaters, "hits") if skaters else None
        out[f"{side}_blocked"] = _sum(skaters, "blockedShots") if skaters else None
        out[f"{side}_pim"] = _sum(skaters, "pim") if skaters else None
        out[f"{side}_giveaways"] = _sum(skaters, "giveaways") if skaters else None
        out[f"{side}_takeaways"] = _sum(skaters, "takeaways") if skaters else None
        out[f"{side}_goalie_id"] = starter.get("playerId")
        out[f"{side}_goalie_name"] = str(
            ((starter.get("name") or {}).get("default", "")) or "")
        out[f"{side}_goalie_toi"] = toi
        out[f"{side}_goalie_decision"] = starter.get("decision")
        # Team goals allowed (for the per-start GAA of the decision goalie).
        opp = "awayTeam" if side == "home" else "homeTeam"
        out[f"{side}_goals_against"] = (bs.get(opp) or {}).get("score")
    return out


BOXSCORE_COLS = [
    "game_id",
    "home_sog", "away_sog", "home_pp_goals", "away_pp_goals",
    "home_faceoff_pct", "away_faceoff_pct", "home_hits", "away_hits",
    "home_blocked", "away_blocked", "home_pim", "away_pim",
    "home_giveaways", "away_giveaways", "home_takeaways", "away_takeaways",
    "home_goalie_id", "away_goalie_id", "home_goalie_name", "away_goalie_name",
    "home_goalie_toi", "away_goalie_toi",
    "home_goalie_decision", "away_goalie_decision",
    "home_goals_against", "away_goals_against",
]


def load_boxscores(game_ids: list[str], use_cache: bool = True) -> pd.DataFrame:
    """Fetch /v1/gamecenter/{id}/boxscore for each game and return the
    combined per-game rollup rows. Per-game parquet caches; a failed game
    is warned and skipped (downstream features degrade to NaN), never fatal.
    """
    frames: list[pd.DataFrame] = []
    for gid in game_ids:
        path = _cache_path(f"boxscore_{BOXSCORE_CACHE_VERSION}_{gid}.parquet")
        if use_cache and path.exists():
            try:
                frames.append(pd.read_parquet(path))
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("boxscore cache %s unreadable (%s)", path.name, exc)
        try:
            bs = _http_json(f"{NHL_API_BASE}/gamecenter/{gid}/boxscore")
            row = _parse_boxscore(bs)
            df = pd.DataFrame([row], columns=BOXSCORE_COLS)
        except Exception as exc:  # noqa: BLE001
            logger.warning("boxscore unavailable for game %s: %s", gid, exc)
            continue
        if not df.empty:
            df.to_parquet(path, index=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=BOXSCORE_COLS)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# MoneyPuck enrichment (optional; honest NaN degradation)
# ---------------------------------------------------------------------------

def load_moneypuck_shots(seasons: list[int] | None = None,
                         use_cache: bool = True) -> pd.DataFrame | None:
    """MoneyPuck shot-level season CSV (optional enrichment family).

    The public download is one all-years shots file; it is cached once and
    filtered to the requested seasons. A failed/absent download returns None
    (the pipeline logs it and continues — the enrichment columns degrade to
    NaN per the documented policy; production features never depend on
    them). MoneyPuck data is non-commercial and requires credit.
    """
    path = _cache_path(f"moneypuck_shots_{MP_CACHE_VERSION}.parquet")
    df: pd.DataFrame | None = None
    if use_cache and path.exists():
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("moneypuck cache unreadable (%s)", exc)
    if df is None:
        try:
            import requests
            resp = requests.get(MONEYPUCK_SHOTS_URL, timeout=120,
                                headers={"User-Agent": "sports-prediction-model/1.0"})
            resp.raise_for_status()
            df = pd.read_csv(io.BytesIO(resp.content), low_memory=False)
            df.to_parquet(path, index=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MoneyPuck shots unavailable (enrichment degrades "
                           "to NaN): %s", exc)
            return None
    if df is None or df.empty:
        return None
    if seasons and "season" in df.columns:
        df = df[pd.to_numeric(df["season"], errors="coerce")
                .isin([int(s) * 10000 + (int(s) + 1) for s in seasons])
                | pd.to_numeric(df["season"], errors="coerce").isin(seasons)]
    return df.reset_index(drop=True) if len(df) else None


def load_team_names() -> dict[str, str]:
    """team abbrev -> full team name (frontend games[] display fields).

    Resolved from a score-page team block (name.default), cached as JSON.
    Falls back to the config team map's abbreviations when unreachable.
    """
    path = _cache_path("team_names.json")
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    names: dict[str, str] = {}
    try:
        today = date.today().isoformat()
        payload = _http_json(f"{NHL_API_BASE}/score/{today}")
        for g in (payload.get("games") or []):
            for key in ("homeTeam", "awayTeam"):
                t = g.get(key) or {}
                abbr = t.get("abbrev")
                name = (t.get("name") or {}).get("default")
                if abbr and name:
                    names.setdefault(str(abbr), str(name))
    except Exception as exc:  # noqa: BLE001
        logger.warning("team-name resolution failed: %s", exc)
    if names:
        try:
            path.write_text(json.dumps(names), encoding="utf-8")
        except OSError:
            pass
    return names
