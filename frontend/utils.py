"""Shared frontend helpers.

This module is the single place the four pages talk to the artifact sink
(``data_delivery``). It reads artifacts from **raw.githubusercontent.com**
URLs when a repo is configured, and transparently falls back to the real
committed ``data_delivery`` artifacts in the local repo (not bundled
samples) so the app always renders, even offline or before the first
Colab push.

Only pandas / requests / altair / streamlit are used — no sklearn, xgboost,
lightgbm, or shap (heavy ML stays in the backend).
"""

from __future__ import annotations

import io
import json
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import requests
import streamlit as st

from sports_config import (
    DEFAULT_SPORT,
    SPORTS,
    artifact_patterns,
    normalize_sport_key,
    resolve_sport,
)

# frontend/ lives at the repository root (multi-sport restructure, Phase B),
# so parents[1] of this module IS the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
# Per-sport artifact sink: {repo_root}/{sport.repo_subdir}/data_delivery.
# repo_subdir is BOTH the GitHub raw-URL prefix and the local fallback dir,
# so flipping it (Phase C: mlb-backend -> mlb-backend) moves the
# fetcher and the committed-artifact fallback atomically.
_REPO_SUBDIR = SPORTS[DEFAULT_SPORT]["repo_subdir"]
LOCAL_DATA_DIR = REPO_ROOT / _REPO_SUBDIR / "data_delivery"
# Artifacts live under {sport}/data_delivery/ inside the GitHub repo
REPO_SUBDIR = _REPO_SUBDIR

PRIMARY = "#10B981"      # emerald accent
BLUE = "#3B82F6"
RED = "#EF4444"
AMBER = "#F59E0B"
SLATE = "#94A3B8"
TEXT = "#E2E8F0"
CARD_BG = "#131B2E"
BORDER = "#1E293B"
PAGE_BG = "#0A1128"


# ---------------------------------------------------------------------------
# Repo / source configuration (session state, set by Home.py)
# ---------------------------------------------------------------------------

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def get_source_config() -> dict:
    """Return the current GitHub source config from session state or env."""
    return {
        "owner": st.session_state.get("gh_owner", "") or _env("GITHUB_OWNER"),
        "repo": st.session_state.get("gh_repo", "") or _env("GITHUB_REPO"),
        "branch": st.session_state.get("gh_branch", "") or _env("GITHUB_BRANCH") or "main",
    }


def source_label() -> str:
    cfg = get_source_config()
    if cfg["owner"] and cfg["repo"]:
        return f"GitHub raw · {cfg['owner']}/{cfg['repo']}@{cfg['branch']}"
    return "Local committed artifacts (no GitHub repo configured)"


# ---------------------------------------------------------------------------
# Sport state (single source of truth)
# ---------------------------------------------------------------------------

def get_sport() -> str:
    """The active sport key (normalized, always a valid SPORTS entry).

    Single source of truth for the app: reads ``st.session_state["sport"]``
    (set by the sidebar sport selector, persisted across reruns) and
    normalizes through the registry so every loader/page resolves the same
    value — never a hardcoded sport downstream. Missing/None/unknown falls
    back to DEFAULT_SPORT (MLB) so a nav rerun never KeyErrors.
    """
    return normalize_sport_key(st.session_state.get("sport", DEFAULT_SPORT))


def sport_config() -> dict:
    """The resolving config dict for the active sport (registry entry)."""
    return resolve_sport(get_sport())


# ---------------------------------------------------------------------------
# Team names + display-column normalization
# ---------------------------------------------------------------------------

MLB_TEAM_NAMES = {
    "NYY": "New York Yankees", "BOS": "Boston Red Sox", "TB": "Tampa Bay Rays",
    "TOR": "Toronto Blue Jays", "BAL": "Baltimore Orioles", "CLE": "Cleveland Guardians",
    "DET": "Detroit Tigers", "MIN": "Minnesota Twins", "CWS": "Chicago White Sox",
    "KC": "Kansas City Royals", "HOU": "Houston Astros", "SEA": "Seattle Mariners",
    "TEX": "Texas Rangers", "LAA": "Los Angeles Angels", "ATH": "Athletics",
    "OAK": "Oakland Athletics", "ATL": "Atlanta Braves", "PHI": "Philadelphia Phillies",
    "NYM": "New York Mets", "MIA": "Miami Marlins", "WSH": "Washington Nationals",
    "MIL": "Milwaukee Brewers", "CHC": "Chicago Cubs", "STL": "St. Louis Cardinals",
    "PIT": "Pittsburgh Pirates", "CIN": "Cincinnati Reds", "LAD": "Los Angeles Dodgers",
    "SD": "San Diego Padres", "SF": "San Francisco Giants", "AZ": "Arizona Diamondbacks",
    "ARI": "Arizona Diamondbacks", "COL": "Colorado Rockies",
}

COIN_FLIP_THRESHOLD = 0.02   # |p − 0.5| below this → coin flip
UPSET_PROB_THRESHOLD = 0.35  # winner's model prob below this → upset


def _is_evening_start(iso) -> bool:
    """True if the game starts at 7 PM ET or later."""
    try:
        ts = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=ZoneInfo("UTC"))
        et = ts.astimezone(ZoneInfo("America/New_York"))
        return et.hour >= 19
    except (ValueError, TypeError):
        return False


def normalize_games(df: pd.DataFrame) -> pd.DataFrame:
    """Derive every display column the dashboard cards expect.

    The training pipeline emits a compact per-game CSV (probabilities,
    market lines, model pick, outcome). This fills the presentation-layer
    gaps so pages never KeyError on missing columns:

      game_status     Final when an outcome exists, otherwise Live
      evening_game    starts 7 PM ET or later
      day_game        inverse of evening_game
      *_team_name     full names from abbreviations
      model_correct   pick matched the actual result (final games)
      is_upset        winner's model probability ≤ 35%
      is_coin_flip    |p − 0.5| < 2 pts or no pick
      home/away_score merged from game_level_features.csv when absent
      final_inning    empty (inning detail not in pipeline artifacts yet)
    """
    df = df.copy()

    # Reconcile stale boards against authoritative finals FIRST, so the
    # derived win/status/grading columns below are computed from real results
    # (todays_games_<date>.csv is a point-in-time snapshot -- a day's later
    # games can still say 'pre'/'in' at 0-0 even after they are final).
    df = _reconcile_board_finals(df)

    # Outcome flag: 1.0 home won, 0.0 away won, NaN no result yet
    win = pd.to_numeric(df.get("home_win"), errors="coerce")

    if "game_status" not in df.columns:
        # Prefer the authoritative ESPN game_state when present ("post" =
        # Final, "in" = Live, "pre" = Scheduled). When absent (e.g.
        # archive views rebuilt from prediction history), fall back to
        # the legacy heuristic: home_win notnull → Final, else Live.
        gs = df.get("game_state")
        if gs is not None and gs.notna().any():
            _MAP = {"post": "Final", "in": "Live", "pre": "Scheduled"}
            status = gs.map(_MAP).fillna("Live")
        else:
            status = win.notna().map({True: "Final", False: "Live"})
            if "start_time_utc" in df.columns:
                starts = pd.to_datetime(df["start_time_utc"], errors="coerce", utc=True)
                status = status.mask(
                    starts > pd.Timestamp.now(tz="UTC"), "Scheduled"
                )
        df["game_status"] = status

    if "evening_game" not in df.columns and "start_time_utc" in df.columns:
        df["evening_game"] = df["start_time_utc"].map(_is_evening_start)
    if "day_game" not in df.columns and "evening_game" in df.columns:
        df["day_game"] = ~df["evening_game"].astype(bool)

    for abbr_col, name_col in (("home_team", "home_team_name"),
                               ("away_team", "away_team_name")):
        if name_col not in df.columns and abbr_col in df.columns:
            df[name_col] = df[abbr_col].map(MLB_TEAM_NAMES).fillna("")

    if "final_inning" not in df.columns:
        df["final_inning"] = ""

    # Model pick + correctness
    if "model_pick" not in df.columns:
        ph = pd.to_numeric(df.get("home_win_prob_model"), errors="coerce")
        df["model_pick"] = ""
        df.loc[ph >= 0.5, "model_pick"] = df.loc[ph >= 0.5, "home_team"]
        df.loc[ph < 0.5, "model_pick"] = df.loc[ph < 0.5, "away_team"]
    pick = df["model_pick"].fillna("").astype(str)

    actual = pd.Series(pd.NA, index=df.index, dtype="object")
    finished = win.notna()
    actual = actual.mask(finished & (win == 1), df["home_team"])
    actual = actual.mask(finished & (win == 0), df["away_team"])
    if "model_correct" not in df.columns:
        df["model_correct"] = finished & (pick == actual)

    probs = pd.to_numeric(df.get("home_win_prob_model"), errors="coerce")
    winner_prob = probs.where(actual == df["home_team"], 1 - probs)
    if "is_upset" not in df.columns:
        df["is_upset"] = finished & (winner_prob <= UPSET_PROB_THRESHOLD)
    if "is_coin_flip" not in df.columns:
        df["is_coin_flip"] = (pick == "") | (
            (probs - 0.5).abs() < COIN_FLIP_THRESHOLD
        )

    # Fill missing final scores from game_level_features.csv (keyed by game_id)
    if "game_id" in df.columns:
        scores = _load_scores()
        if not scores.empty:
            df = df.merge(scores, on="game_id", how="left", suffixes=("", "_feat"))
            for col in ("home_score", "away_score"):
                if f"{col}_feat" in df.columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
                            pd.to_numeric(df[f"{col}_feat"], errors="coerce"))
                    else:
                        df[col] = df[f"{col}_feat"]
                    df.drop(columns=[f"{col}_feat"], inplace=True)
        # Defensive net: never let a duplicated game_id reach the card loop,
        # where the same keyed selectbox would be emitted twice.
        if "game_id" in df.columns:
            df = df.drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def _load_scores() -> pd.DataFrame:
    """Per-game final scores from game_level_features.csv (optional merge).

    Returns a two-column frame keyed by game_id ('YYYYMMDD_AWAY@HOME');
    empty frame when the artifact is unavailable.
    """
    cfg = get_source_config()
    data, src = _fetch_bytes("game_level_features.csv", **cfg)
    if data is None:
        return pd.DataFrame(columns=["game_id", "home_score", "away_score"])
    try:
        gl = pd.read_csv(io.BytesIO(data),
                         usecols=lambda c: c in ("game_pk", "game_date",
                                                 "home_team", "away_team",
                                                 "home_score", "away_score"))
        d = pd.to_datetime(gl["game_date"], errors="coerce").dt.strftime("%Y%m%d")
        gl["game_id"] = d + "_" + gl["away_team"].astype(str) + "@" + gl["home_team"].astype(str)
        # game_level_features can carry duplicate (game, date) rows; a one-per-
        # game scores frame is required so the leaky left-merge in normalize_games
        # never EXPLODES a todays_games row into duplicates (which crashed the
        # per-card keyed selectboxes). Keep the last occurrence.
        gl = gl.drop_duplicates(subset=["game_id"], keep="last")
        return gl[["game_id", "home_score", "away_score"]]
    except Exception:
        return pd.DataFrame(columns=["game_id", "home_score", "away_score"])


# ---------------------------------------------------------------------------
# Final-result reconciliation (stale-board fix)
# ---------------------------------------------------------------------------

# Team-abbreviation aliases that differ between artifacts (StatsAPI vs ESPN,
# historic renames). Normalized so a board row and a game_level_features row
# for the same game always join.
_TEAM_ALIAS = {
    "CWS": "CWS", "CHW": "CWS",
    "ARI": "ARI", "AZ": "ARI",
    "OAK": "OAK", "ATH": "OAK",
    "TB": "TB", "TBA": "TB",
}


def _normalize_team(code) -> str:
    """Canonical team code used to match the same game across artifacts (they
    spell teams differently: CHW vs CWS, AZ vs ARI, ATH vs OAK). Unknown codes
    pass through uppercased/stripped (best-effort), never None."""
    if code is None:
        return ""
    c = str(code).strip().upper()
    return _TEAM_ALIAS.get(c, c)


def _game_date_compact(d) -> str:
    """YYYY-MM-DD (or any leading-date form) -> YYYYMMDD, else ''."""
    try:
        s = str(d).strip()
        if len(s) >= 10 and s[4:5] == "-":
            return s[:10].replace("-", "")
    except Exception:
        pass
    return ""


@st.cache_data(ttl=3600, show_spinner=False)
def _load_final_results() -> pd.DataFrame:
    """Authoritative final scores/results from game_level_features.csv (the
    decided frame). Carries the per-leg game_pk so doubleheader identity
    survives; empty frame when the artifact is unavailable."""
    cfg = get_source_config()
    data, _src = _fetch_bytes("game_level_features.csv", **cfg)
    cols = ["game_pk", "game_date", "home_team", "away_team",
            "home_score", "away_score", "home_win"]
    if data is None:
        return pd.DataFrame(columns=cols)
    try:
        return pd.read_csv(io.BytesIO(data), usecols=lambda c: c in cols)
    except Exception:
        return pd.DataFrame(columns=cols)


def _leg_eq(a, b) -> bool:
    try:
        return bool(pd.notna(a) and pd.notna(b) and float(a) == float(b))
    except (TypeError, ValueError):
        return False


def _authority_by_key(gl: pd.DataFrame) -> dict:
    """Authority lookup: (YYYYMMDD, norm_home, norm_away) -> ordered list of
    leg row dicts (1 or 2 for a doubleheader), ordered by game_pk."""
    keyed: dict = {}
    for _, r in gl.iterrows():
        key = (_game_date_compact(r.get("game_date")),
               _normalize_team(r.get("home_team")),
               _normalize_team(r.get("away_team")))
        if not key[0]:
            continue
        keyed.setdefault(key, []).append({
            "game_pk": r.get("game_pk"),
            "home_score": r.get("home_score"),
            "away_score": r.get("away_score"),
            "home_win": r.get("home_win"),
        })
    for legs in keyed.values():
        def _pk(leg):
            try:
                return int(leg["game_pk"])
            except (TypeError, ValueError):
                return 0
        legs.sort(key=_pk)
    return keyed


def _board_rows_by_key(df: pd.DataFrame) -> dict:
    """Board row indices grouped by (date, home, away), in board order."""
    by_key: dict = {}
    for idx, row in df.iterrows():
        key = (_game_date_compact(row.get("game_date")),
               _normalize_team(row.get("home_team")),
               _normalize_team(row.get("away_team")))
        if not key[0]:
            continue
        by_key.setdefault(key, []).append(idx)
    return by_key


def _apply_leg(frame: pd.DataFrame, idx, leg) -> None:
    """Write one authoritative leg's result onto a board row and mark it Final."""
    hs = pd.to_numeric(leg["home_score"], errors="coerce")
    as_ = pd.to_numeric(leg["away_score"], errors="coerce")
    frame.at[idx, "home_score"] = hs
    frame.at[idx, "away_score"] = as_
    if pd.notna(hs) and pd.notna(as_) and "total_runs" in frame.columns:
        frame.at[idx, "total_runs"] = float(hs) + float(as_)
    frame.at[idx, "home_win"] = leg["home_win"]
    if "game_state" in frame.columns:
        frame.at[idx, "game_state"] = "post"
    if "game_status" in frame.columns:
        frame.at[idx, "game_status"] = "Final"


def _reconcile_board_finals(df: pd.DataFrame,
                            finals: pd.DataFrame = None) -> pd.DataFrame:
    """Overlay authoritative final results onto a possibly-stale board frame.

    todays_games_<date>.csv is a point-in-time snapshot: a day's later games
    can still say 'pre'/'in' at 0-0 even after they are final in
    game_level_features.csv. For every board game with an authoritative result
    we set home_score/away_score/home_win/total_runs and flip game_state to
    'post' (so normalize_games grades it Final). A genuinely still-scheduled
    future game has NO result in the decided frame and is left untouched --
    never fabricated.

    Doubleheaders carry one row per leg (distinct game_pk). If the board
    already holds a real captured final for one leg that uniquely equals one
    authority leg, that leg is anchored and the sibling gets the other;
    otherwise legs are matched deterministically by game_pk order (best-effort,
    documented) so both legs always surface a score.
    """
    if df is None or df.empty or "home_team" not in df.columns:
        return df
    if finals is None:
        finals = _load_final_results()
    if finals is None or finals.empty:
        return df

    out = df.copy()
    authority = _authority_by_key(finals)
    for key, board_rows in _board_rows_by_key(out).items():
        legs = authority.get(key)
        if not legs:
            continue  # no authoritative result in the decided frame -> keep pre/live
        if len(board_rows) == 1:
            _apply_leg(out, board_rows[0], legs[0])
            continue
        # Doubleheader: match both board rows to the two authority legs.
        assigned = {}
        # 1) Anchor rows that already captured a real final to the exact leg.
        for bi in board_rows:
            hs = pd.to_numeric(out.at[bi, "home_score"]
                               if "home_score" in out.columns else None,
                               errors="coerce")
            as_ = pd.to_numeric(out.at[bi, "away_score"]
                               if "away_score" in out.columns else None,
                               errors="coerce")
            if pd.notna(hs) and pd.notna(as_) and not (hs == 0 and as_ == 0):
                for li, leg in enumerate(legs):
                    if li in assigned.values():
                        continue
                    if _leg_eq(leg["home_score"], hs) and _leg_eq(leg["away_score"], as_):
                        assigned[bi] = li
                        break
        # 2) Assign any remaining rows to any remaining legs by order.
        for bi in board_rows:
            if bi in assigned:
                continue
            for li in range(len(legs)):
                if li in assigned.values():
                    continue
                assigned[bi] = li
                break
        for bi, li in assigned.items():
            _apply_leg(out, bi, legs[li])
    return out


# ---------------------------------------------------------------------------
# Artifact loading (GitHub raw -> local committed-artifact fallback)
# ---------------------------------------------------------------------------

def _data_dir(sport=None) -> Path:
    """The active sport's ``data_delivery`` dir (resolved via the registry),
    so loaders read the RIGHT sport's artifacts regardless of sport."""
    s = normalize_sport_key(sport if sport is not None else get_sport())
    return REPO_ROOT / resolve_sport(s)["repo_subdir"] / "data_delivery"


def _raw_url(relpath: str, owner: str, repo: str, branch: str, sport=None) -> str:
    """Return the raw.githubusercontent.com URL that _fetch_bytes would try,
    using the active sport's repo subdir."""
    s = normalize_sport_key(sport if sport is not None else get_sport())
    subdir = resolve_sport(s)["repo_subdir"]
    if owner and repo:
        return (f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}"
                f"/{subdir}/data_delivery/{relpath}")
    return f"<local:{_data_dir(s) / relpath}>"


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_bytes_cached(relpath: str, sport: str, owner: str, repo: str,
                        branch: str):
    """Fetch one artifact for a SPECIFIED sport (sport is in the cache key)."""
    s = normalize_sport_key(sport)
    subdir = resolve_sport(s)["repo_subdir"]
    dd = REPO_ROOT / subdir / "data_delivery"
    if owner and repo:
        url = (f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}"
               f"/{subdir}/data_delivery/{relpath}")
        try:
            resp = requests.get(url, timeout=15)
            if resp.ok:
                return resp.content, "github"
        except requests.RequestException:
            pass
    local = dd / relpath
    if local.exists():
        return local.read_bytes(), "local"
    return None, "missing"


def _fetch_bytes(relpath: str, owner: str, repo: str, branch: str,
                 sport=None):
    """Fetch one artifact for the active sport (or ``sport`` when given).
    Returns (bytes | None, source); the sport is resolved here and forwarded
    to the cached impl so the cache key is sport-specific."""
    s = normalize_sport_key(sport if sport is not None else get_sport())
    return _fetch_bytes_cached(relpath, s, owner, repo, branch)


@st.cache_data(ttl=300, show_spinner=False)
def available_dates(owner: str, repo: str, branch: str,
                    sport=None) -> list[str]:
    """All artifact dates (YYYYMMDD), newest first, for the ACTIVE sport.
    Resolves the sport here and delegates to the sport-keyed cached impl so
    MLB and NFL date sets never collide in the cache."""
    s = normalize_sport_key(sport if sport is not None else get_sport())
    return _available_dates_cached(s, owner, repo, branch)


@st.cache_data(ttl=300, show_spinner=False)
def _available_dates_cached(sport: str, owner: str, repo: str,
                            branch: str) -> list[str]:
    """Per-sport artifact dates (cache key includes ``sport``):
    MLB enumerates todays_games_*.csv + merges calibration ``daily`` and
    prediction-history game dates; NFL enumerates the dated moneyline /
    calibration / prediction-history families."""
    s = normalize_sport_key(sport)
    subdir = resolve_sport(s)["repo_subdir"]
    dd_dir = REPO_ROOT / subdir / "data_delivery"
    dates: set[str] = set()

    if s == "mlb":
        if owner and repo:
            try:
                api = (f"https://api.github.com/repos/{owner}/{repo}/contents"
                       f"/{subdir}/data_delivery")
                resp = requests.get(api, timeout=15)
                if resp.ok:
                    for item in resp.json():
                        name = item.get("name", "")
                        if name.startswith("todays_games_") and name.endswith(".csv"):
                            dates.add(name[len("todays_games_"):-len(".csv")])
            except requests.RequestException:
                pass
        for p in dd_dir.glob("todays_games_*.csv"):
            dates.add(p.name[len("todays_games_"):-len(".csv")])

        # Merge walk-forward calibration days so history stays selectable even
        # when only one todays_games snapshot exists in the repo.
        if dates:
            try:
                cal_bytes, _ = _fetch_bytes(
                    f"calibration_{max(dates)}.json",
                    owner=owner, repo=repo, branch=branch, sport="mlb")
                if cal_bytes:
                    for entry in json.loads(cal_bytes).get("daily", []):
                        d = str(entry.get("date", ""))
                        if len(d) == 8 and d.isdigit():
                            dates.add(d)
            except (ValueError, TypeError, KeyError):
                pass
        # Merge per-game prediction-history dates so ANY past date that has
        # predictions is navigable on Today's Games.
        if dates:
            try:
                hist_bytes, _ = _fetch_bytes(
                    f"predictions_history_{max(dates)}.csv",
                    owner=owner, repo=repo, branch=branch, sport="mlb")
                if hist_bytes:
                    hist = pd.read_csv(io.BytesIO(hist_bytes), usecols=["game_date"])
                    for d in hist["game_date"].dropna().astype(str):
                        d = d.replace("-", "")
                        if len(d) == 8 and d.isdigit():
                            dates.add(d)
            except (ValueError, TypeError, KeyError, pd.errors.EmptyDataError):
                pass
        return sorted(dates, reverse=True)

    # NFL: the named dated families the calibration/history/monitor/markets
    # pages use (the run-engine slate-serve families included so the NFL
    # Totals & Run Lines page resolves its latest artifact date).
    prefixes = [("nfl_moneyline_v1_", ".json"), ("nfl_calibration_", ".json"),
                ("nfl_predictions_history_", ".csv"), ("nfl_model_monitor_", ".json"),
                ("nfl_power_rankings_", ".csv"),
                ("nfl_run_engine_markets_", ".csv"),
                ("nfl_run_engine_monitor_", ".json")]
    if owner and repo:
        try:
            api = (f"https://api.github.com/repos/{owner}/{repo}/contents"
                   f"/{subdir}/data_delivery")
            resp = requests.get(api, timeout=15)
            if resp.ok:
                for item in resp.json():
                    name = item.get("name", "")
                    for pfx, ext in prefixes:
                        if name.startswith(pfx) and name.endswith(ext):
                            core = name[len(pfx):-len(ext)]
                            if len(core) == 8 and core.isdigit():
                                dates.add(core)
                            break
        except requests.RequestException:
            pass
    for pfx, ext in prefixes:
        for p in dd_dir.glob(f"{pfx}*{ext}"):
            dates.update(_stamp_suffixes(p))
    return sorted(dates, reverse=True)


def _pick_date(date_str: str) -> str:
    dates = available_dates(**get_source_config())
    if date_str and date_str in dates:
        return date_str
    if dates:
        return dates[0]
    return "20260809"  # last committed artifact date


def _pick_artifact_date(date_str: str, prefix: str) -> str:
    """Find the best date for an artifact file ``{prefix}_{date}.ext``.

    Tries the requested date first; if no artifact exists for it, falls
    back to the newest available date — so tabs that track the latest
    snapshot (Calibration, Model Monitor, SHAP) never show blank just
    because the user navigated to a past date.
    """
    cfg = get_source_config()
    dates = available_dates(**cfg)
    latest = dates[0] if dates else None
    # Try the requested date — check .json first (calibration/monitor), then .csv (SHAP)
    if date_str:
        for ext in (".json", ".csv"):
            if _fetch_bytes(f"{prefix}_{date_str}{ext}", **cfg)[0] is not None:
                return date_str
    # Fall back to latest available date
    if latest and latest != date_str:
        return latest
    return date_str or "20260809"


# ==========================================================================
# Artifact resolver (sport → data_delivery dir + artifact family)
# ==========================================================================

def _latest_artifact_path(sport_dir: Path, pattern: str) -> Optional[Path]:
    """Newest file matching ``pattern`` under ``sport_dir``.

    Prefers the newest embedded YYYYMMDD suffix (the pipeline's dating
    convention); falls back to newest by mtime. None when no file matches.
    """
    cands = sorted(sport_dir.glob(pattern))
    if not cands:
        return None
    dated = [p for p in cands if _stamp_suffixes(p)]
    if dated:
        return max(dated, key=lambda p: max(_stamp_suffixes(p)))
    return max(cands, key=lambda p: p.stat().st_mtime)


def resolve_sport_artifact(sport: str | None, family: str) -> Optional[Path]:
    """Latest committed artifact path for a sport/artifact-family, or None.

    Resolves the sport to its ``data_delivery`` dir + the family's glob
    pattern from the registry and returns the newest dated file. Seasons the
    loader/adapter layer so every page reads through the same resolver — no
    hardcoded paths or per-sport branching in the pages.
    """
    s = normalize_sport_key(sport if sport is not None else get_sport())
    pat = artifact_patterns(s).get(family)
    if not pat:
        return None
    sport_dir = REPO_ROOT / resolve_sport(s)["repo_subdir"] / "data_delivery"
    if not sport_dir.is_dir():
        return None
    return _latest_artifact_path(sport_dir, pat)


def latest_artifact_date(sport: str | None, family: str) -> Optional[str]:
    """Newest YYYYMMDD date (as a string) available for a family, or None."""
    path = resolve_sport_artifact(sport, family)
    if path is None:
        return None
    stamps = _stamp_suffixes(path)
    return max(stamps) if stamps else None


# ==========================================================================
# Per-sport VALID game dates (the boards a user can land on)
# ==========================================================================

@st.cache_data(ttl=300, show_spinner=False)
def _contents_todays_dates(owner: str, repo: str, branch: str) -> tuple[str, ...]:
    """todays_games_*.csv dates listed by the GitHub contents API (cached),
    or empty when the API is unreachable — callers also merge the local dir."""
    dates: set[str] = set()
    if owner and repo:
        try:
            api = (f"https://api.github.com/repos/{owner}/{repo}/contents"
                   f"/{REPO_SUBDIR}/data_delivery")
            resp = requests.get(api, timeout=15)
            if resp.ok:
                for item in resp.json():
                    name = item.get("name", "")
                    if name.startswith("todays_games_") and name.endswith(".csv"):
                        d = name[len("todays_games_"):-len(".csv")]
                        if len(d) == 8 and d.isdigit():
                            dates.add(d)
        except requests.RequestException:
            pass
    return tuple(sorted(dates))


@st.cache_data(ttl=300, show_spinner=False)
def _mlb_history_dates(owner: str, repo: str, branch: str,
                       max_snapshot: str | None) -> tuple[str, ...]:
    """Walk-forward dates a Today's Games board can be REBUILT for (cached):
    calibration ``daily`` entries + per-game prediction-history ``game_date``
    values, keyed off the newest todays_games snapshot. Missing/unparseable
    artifacts → (). This restores season-long date navigation that the
    todays_games-artifact-only set dropped."""
    if not max_snapshot:
        return ()
    dates: set[str] = set()
    try:
        cal, _ = _fetch_bytes(f"calibration_{max_snapshot}.json",
                              owner=owner, repo=repo, branch=branch)
        if cal:
            for entry in json.loads(cal).get("daily", []):
                d = str(entry.get("date", ""))
                if len(d) == 8 and d.isdigit():
                    dates.add(d)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        pass
    try:
        hist, _ = _fetch_bytes(f"predictions_history_{max_snapshot}.csv",
                               owner=owner, repo=repo, branch=branch)
        if hist:
            h = pd.read_csv(io.BytesIO(hist), usecols=["game_date"])
            for d in h["game_date"].dropna().astype(str):
                d = d.replace("-", "")
                if len(d) == 8 and d.isdigit():
                    dates.add(d)
    except (ValueError, TypeError, KeyError, pd.errors.EmptyDataError):
        pass
    return tuple(sorted(dates))


def _distinct_game_dates(frame: pd.DataFrame) -> list[str]:
    """Distinct ``game_date`` values in a game frame, as YYYYMMDD, newest first.

    NFL valid dates are the DISTINCT per-game ``game_date`` values in the
    moneyline ``games[]`` array (NOT the artifact filename date). Empty frame
    / no game_date column / aggregate-only record → [].
    """
    if frame is None or frame.empty or "game_date" not in frame.columns:
        return []
    out: set[str] = set()
    for v in frame["game_date"].dropna().astype(str):
        v = v.strip()
        if len(v) == 10 and v[4:5] == "-" and v.replace("-", "").isdigit():
            out.add(v.replace("-", ""))
        elif len(v) == 8 and v.isdigit():
            out.add(v)
    return sorted(out, reverse=True)


def _valid_dates_impl(sport_key: str, contents_dates, local_dir,
                      nfl_frame: pd.DataFrame, history_dates=()) -> list[str]:
    """Pure per-sport valid-date derivation (testable without Streamlit/net).

    MLB: a date is valid when a ``todays_games_<YYYYMMDD>.csv`` board exists
    (contents listing + local dir) OR the walk-forward history can rebuild it
    from calibration ``daily`` entries / prediction-history game dates
    (``history_dates``). NFL: distinct ``game_date`` from the moneyline
    ``games[]`` frame. Missing/empty artifacts → [] (graceful)."""
    s = normalize_sport_key(sport_key)
    if s == "nfl":
        return _distinct_game_dates(nfl_frame)
    dates = set(contents_dates or ())
    dates.update(history_dates or ())
    for p in Path(local_dir).glob("todays_games_*.csv"):
        d = p.name[len("todays_games_"):-len(".csv")]
        if len(d) == 8 and d.isdigit():
            dates.add(d)
    return sorted(dates, reverse=True)


@st.cache_data(ttl=120, show_spinner=False)
def valid_dates(sport_key: str | None = None) -> tuple[str, ...]:
    """Valid game dates (YYYYMMDD, newest first) for a sport — the boards a
    user can actually land on, cached so per-rerun enumeration is cheap.

    MLB: ``todays_games_YYYYMMDD.csv`` artifacts (contents API + local).
    NFL: distinct per-game ``game_date`` from the resolved moneyline
    ``games[]`` array (empty when the shipped record is aggregate-only)."""
    s = normalize_sport_key(sport_key if sport_key is not None else get_sport())
    cfg = get_source_config()
    contents = _contents_todays_dates(**cfg) if s == "mlb" else ()
    nfl_frame = load_nfl_moneyline("nfl") if s == "nfl" else pd.DataFrame()
    snap = set(contents)
    for p in LOCAL_DATA_DIR.glob("todays_games_*.csv"):
        d = p.name[len("todays_games_"):-len(".csv")]
        if len(d) == 8 and d.isdigit():
            snap.add(d)
    max_snap = max(snap) if snap else None
    history = (_mlb_history_dates(cfg["owner"], cfg["repo"], cfg["branch"],
                                  max_snap) if s == "mlb" else ())
    return tuple(_valid_dates_impl(s, contents, LOCAL_DATA_DIR, nfl_frame, history))


def nearest_valid_date(valid: list[str] | tuple[str, ...],
                       target: str | None = None) -> Optional[str]:
    """The valid date closest to ``target`` (default: today in America/New_York).

    Ties break to the earlier date. Returns None when ``valid`` is empty or
    has no parseable dates — callers fall back to a placeholder / empty state."""
    vs = sorted({d for d in valid if len(str(d)) == 8 and str(d).isdigit()})
    if not vs:
        return None
    if target is None:
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo as _ZI
        target = _dt.now(_ZI("America/New_York")).strftime("%Y%m%d")
    base = datetime.strptime(str(target)[:8], "%Y%m%d")
    best, best_gap = None, None
    for d in vs:
        dt = datetime.strptime(d, "%Y%m%d")
        gap = abs((dt - base).days)
        if best_gap is None or gap < best_gap or (gap == best_gap and d < best):
            best, best_gap = d, gap
    return best


# ==========================================================================
# NFL adapter — nfl_moneyline_v1_*.json → the shared card DataFrame contract
# ==========================================================================

# The moneyline/card DataFrame contract the UI already expects (the column
# set ``todays_games_*.csv`` + ``normalize_games`` produce for a card). NFL
# adapters emit this exact schema; MLB-only fields (pitchers, run engine,
# venue, records) are left null/absent rather than fabricated.
NFL_CARD_COLUMNS = [
    "game_id", "home_team", "away_team", "home_win_prob_model",
    "away_win_prob_model", "home_record", "away_record", "edge_home",
    "edge_away", "start_time_utc", "venue", "model_pick", "home_score",
    "away_score", "game_status", "game_date", "home_team_name",
    "away_team_name",
]


def _nl(row: dict) -> Optional[float]:
    """Safely coerce a numeric field, tolerating ''/None/strings."""
    if row in (None, ""):
        return None
    try:
        v = float(row)
    except (TypeError, ValueError):
        return None
    return v if v == v else None  # NaN → None


def nfl_moneyline_to_frame(data) -> pd.DataFrame:
    """Adapt an ``nfl_moneyline_v1_*.json`` record into the shared card frame.

    The canonical per-game list is read from ``games`` (or ``predictions``);
    each entry maps onto the MLB moneyline/card contract. Where MLB fields
    don't exist for NFL (pitchers, run-engine projection, venue, records) the
    columns are present but null — never fabricated. When the record carries
    no per-game list (the current v1 is an aggregate/calibration record), the
    frame is returned empty WITH the full column schema so downstream card
    code never KeyErrors. Explicit schema is pinned by a fixture test.
    """
    if not isinstance(data, dict):
        return pd.DataFrame(columns=NFL_CARD_COLUMNS)
    rows = data.get("games") or data.get("predictions") or []
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        home = str(r.get("home_team", "") or "").strip()
        away = str(r.get("away_team", "") or "").strip()
        ph = _nl(r.get("home_win_prob", r.get("home_win_prob_model",
                                                r.get("p_home"))))
        if ph is None and _nl(r.get("away_win_prob")) is not None:
            ph = 1.0 - _nl(r.get("away_win_prob"))
        if ph is not None:
            ph = min(1.0, max(0.0, ph))
        pa = None if ph is None else 1.0 - ph
        game_id = r.get("game_id") or r.get("game_pk") or ""
        game_date = r.get("game_date") or r.get("gameday") or ""
        if not game_id and (home or away):
            game_id = f"{str(game_date or '').replace('-', '')}_{away}@{home}"

        hs, as_ = _nl(r.get("home_score")), _nl(r.get("away_score"))
        status = r.get("game_status") or r.get("game_state")
        if not status:
            status = "Final" if (hs is not None and as_ is not None) else "Scheduled"
        status = {"post": "Final", "in": "Live", "pre": "Scheduled"}.get(
            str(status).lower(), str(status).title())

        pick = r.get("model_pick") or (home if (ph is not None and ph >= 0.5) else away)
        out.append({
            "game_id": game_id,
            "home_team": home,
            "away_team": away,
            "home_win_prob_model": ph,
            "away_win_prob_model": pa,
            "home_record": r.get("home_record"),
            "away_record": r.get("away_record"),
            "edge_home": _nl(r.get("edge_home")),
            "edge_away": _nl(r.get("edge_away")),
            "start_time_utc": r.get("start_time_utc")
            or (f"{game_date}T00:00:00Z" if game_date else ""),
            "venue": r.get("venue") or r.get("stadium") or r.get("roof") or "",
            "model_pick": pick or "",
            "home_score": hs,
            "away_score": as_,
            "game_status": status,
            "game_date": game_date,
            "home_team_name": r.get("home_team_name") or "",
            "away_team_name": r.get("away_team_name") or "",
        })
    return pd.DataFrame(out, columns=NFL_CARD_COLUMNS)


def load_nfl_moneyline(sport: str | None = "nfl") -> pd.DataFrame:
    """Load the latest NFL moneyline v1 artifact through the adapter.

    Resolves the newest ``nfl_moneyline_v1_*.json`` in the NFL data_delivery
    dir, reads it, and adapts to the shared card frame. Missing/invalid →
    empty frame with the full card schema (never fabricated)."""
    path = resolve_sport_artifact(sport or "nfl", "moneyline_json")
    if path is None:
        return pd.DataFrame(columns=NFL_CARD_COLUMNS)
    try:
        data = json.loads(path.read_text())
    except Exception:
        return pd.DataFrame(columns=NFL_CARD_COLUMNS)
    return nfl_moneyline_to_frame(data)


def load_nfl_moneyline_record(sport: str | None = "nfl") -> dict:
    """Load the newest NFL moneyline v1 JSON as its RAW dict record.

    The Calibration page reads the aggregate sections (pooled/sealed metrics,
    verdict, slate, members, adaptive weights) directly off this dict; the
    card adapter (``load_nfl_moneyline``) stays the games[] → DataFrame path.
    Missing / invalid / non-dict → {} (never raises).
    """
    path = resolve_sport_artifact(sport or "nfl", "moneyline_json")
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_nfl_prediction_history(sport: str | None = "nfl") -> pd.DataFrame:
    """Per-game (predicted, outcome) frame for the NFL Calibration page.

    Mirrors ``load_prediction_history``'s column contract so the per-1%
    favored-team curve (``mlc.favored_calibration_pts``) and the history
    table can be reused verbatim. Derives one row per moneyline record game
    that carries a DETERMINABLE outcome (both scores present or an explicit
    ``home_win``); only those decide per-game calibration points — scheduled
    'pre' games are skipped. The shipped v1 record is a 2026 scheduled slate
    with no outcomes, so this returns an empty frame WITH the full schema;
    the page then gates the curve / reliability / history sections behind a
    presence check (honest info lines when absent)."""
    cols = ["game_date", "home_team", "away_team", "home_win_prob_model",
            "away_win_prob_model", "model_pick", "home_score", "away_score",
            "actual_winner", "correct"]
    rec = load_nfl_moneyline_record(sport)
    if not rec:
        return pd.DataFrame(columns=cols)
    rows = (rec.get("games") or rec.get("predictions")
            or rec.get("oof_games") or [])
    out = []
    for g in rows:
        if not isinstance(g, dict):
            continue
        home = str(g.get("home_team", "") or "").strip()
        away = str(g.get("away_team", "") or "").strip()
        hs = _nl(g.get("home_score"))
        as_ = _nl(g.get("away_score"))
        hw = g.get("home_win")
        if hw is None:
            if hs is None or as_ is None:
                continue  # scheduled/pre — no outcome yet
            hw = 1.0 if hs > as_ else 0.0
        else:
            try:
                hw = 1.0 if float(hw) == 1.0 else 0.0
            except (TypeError, ValueError):
                continue
        ph = _nl(g.get("home_win_prob"))
        if ph is None:
            ph = _nl(g.get("home_win_prob_model"))
        pa = None if ph is None else 1.0 - ph
        pick = str(g.get("model_pick") or "").strip()
        if not pick and ph is not None:
            pick = home if ph >= 0.5 else away
        winner = home if hw == 1.0 else away
        out.append({
            "game_date": g.get("game_date", ""),
            "home_team": home,
            "away_team": away,
            "home_win_prob_model": ph,
            "away_win_prob_model": pa,
            "model_pick": pick,
            "home_score": hs,
            "away_score": as_,
            "actual_winner": winner,
            "correct": bool(pick and pick == winner),
        })
    return pd.DataFrame(out, columns=cols)


def load_todays_games(date_str: str, sport: str | None = None) -> pd.DataFrame:
    """Game board for a date, dispatched on the active sport (default).

    MLB (default): byte-identical to today — fetches ``todays_games_<date>.csv``
    through ``normalize_games``. NFL: adapts the latest moneyline v1 JSON
    through the resolver/adapter (no MLB CSV exists). Any sport other than
    MLB/NFL degrades to the MLB path (safe fallback)."""
    s = normalize_sport_key(sport if sport is not None else get_sport())
    if s == "nfl":
        return load_nfl_moneyline("nfl")
    cfg = get_source_config()
    data, src = _fetch_bytes(f"todays_games_{_pick_date(date_str)}.csv", **cfg)
    st.session_state["data_source"] = src
    if data is None:
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(data))
    for col in ["home_team_name", "away_team_name", "model_pick", "final_inning", "venue"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return normalize_games(df)


def load_run_engine_markets(date_str: str,
                             sport: str | None = None) -> pd.DataFrame:
    """Run-engine markets artifact for a date; empty frame when missing.

    MLB-only (run-engine); any other sport (NFL) returns an empty frame —
    the run-engine slate is a MLB artifact family. Mirrors markets._load_markets
    (bare filename — _fetch_bytes prepends the repo subdir + data_delivery/).
    Today's Games joins slate rows by game_id == game_pk to enrich the cards;
    missing/stale artifacts degrade to an empty frame with a loud log line,
    never fabricated data.
    """
    if normalize_sport_key(sport if sport is not None else get_sport()) != "mlb":
        return pd.DataFrame()
    import logging
    _log = logging.getLogger("utils.run_engine")
    fname = f"run_engine_markets_{date_str}.csv"
    cfg = get_source_config()
    try:
        raw, src = _fetch_bytes(fname, **cfg)
    except Exception as exc:
        url = _raw_url(fname, **cfg)
        _log.error("Run-engine markets fetch exception for %s (%s): %s",
                   fname, url, exc)
        return pd.DataFrame()
    if raw is None:
        url = _raw_url(fname, **cfg)
        _log.warning("Run-engine markets artifact not found: %s (URL: %s, "
                     "source: %s)", fname, url, src)
        return pd.DataFrame()
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        _log.error("Run-engine markets CSV parse failed for %s: %s", fname, exc)
        return pd.DataFrame()


def _family_dated_dates(sport: str | None,
                        prefixes: list[tuple[str, str]],
                        source_cfg: dict | None = None) -> list[str]:
    """Union of YYYYMMDD dates (newest first) across dated artifact
    families for a sport.

    ``prefixes`` is a list of (filename_prefix, extension) pairs (e.g.
    [("run_engine_markets_", ".csv")]). Enumerates the GitHub
    data_delivery contents API (when a repo is configured) plus the local
    committed dir — the same fetch semantics as ``available_dates()``,
    scoped to exactly the families the caller reads. A page whose
    artifacts ship on their own cadence therefore never inherits a date
    from an unrelated family (``available_dates()`` builds its union from
    todays_games_* / calibration_* / predictions_history_* and never
    enumerates the run-engine families). Returns [] when nothing resolves
    (no owner/repo AND an empty or missing local dir)."""
    s = normalize_sport_key(sport if sport is not None else get_sport())
    subdir = resolve_sport(s)["repo_subdir"]
    cfg = source_cfg if source_cfg is not None else get_source_config()
    dates: set[str] = set()
    # isinstance guard: under streamlit-stubbed test imports the source
    # config carries MagicMock values (truthy but not strings) — never let
    # those trigger a real contents-API request.
    if (isinstance(cfg.get("owner"), str) and isinstance(cfg.get("repo"), str)
            and cfg["owner"] and cfg["repo"]):
        try:
            api = (f"https://api.github.com/repos/{cfg['owner']}/{cfg['repo']}"
                   f"/contents/{subdir}/data_delivery")
            resp = requests.get(api, timeout=15)
            if resp.ok:
                for item in resp.json():
                    name = str(item.get("name", ""))
                    for pfx, ext in prefixes:
                        if name.startswith(pfx) and name.endswith(ext):
                            core = name[len(pfx):-len(ext)]
                            if len(core) == 8 and core.isdigit():
                                dates.add(core)
                            break
        except requests.RequestException:
            pass
    dd = REPO_ROOT / subdir / "data_delivery"
    if dd.is_dir():
        for pfx, ext in prefixes:
            for p in dd.glob(f"{pfx}*{ext}"):
                dates |= _stamp_suffixes(p)
    return sorted(dates, reverse=True)


def _nfl_run_engine_family_dates(sport: str | None, family: str) -> list[str]:
    """Available YYYYMMDD dates (newest first) for an NFL run-engine family.

    Enumerates the GitHub data_delivery contents API (when a repo is
    configured) plus the local dir, exactly like the MLB per-sport loaders
    — the family name/pattern maps to ``markets_csv`` /
    ``markets_monitor_json`` in the registry."""
    s = normalize_sport_key(sport if sport is not None else get_sport())
    pat = artifact_patterns(s).get(family)
    prefix, ext = None, None
    if family == "markets_csv":
        prefix, ext = "nfl_run_engine_markets_", ".csv"
    elif family == "markets_monitor_json":
        prefix, ext = "nfl_run_engine_monitor_", ".json"
    if not pat or not prefix:
        return []
    return _family_dated_dates(s, [(prefix, ext)])


def run_engine_page_dates(owner: str = "", repo: str = "",
                          branch: str = "main",
                          sport: str | None = None) -> list[str]:
    """Newest-first YYYYMMDD dates for the run-engine Totals & Run Lines
    page: the union of the page's OWN artifact families
    (``run_engine_markets_*.csv`` + ``run_engine_monitor_*.json`` on MLB;
    the ``nfl_run_engine_*`` twins on NFL).

    ``available_dates()`` builds its union from todays_games_*,
    calibration_* and predictions_history_* and NEVER enumerates the
    run-engine families — when one of those ships a date whose run-engine
    files are absent or late, its dates[0] points at a day the page cannot
    render. Resolving from this page's own families makes the pick
    drift-proof. Empty when nothing resolves (offline / empty local) —
    callers fall back to ``available_dates()`` so the documented warning
    path still fires instead of a crash.

    Not st.cache_data-wrapped (unlike ``available_dates``): the only
    caller is the markets page's module-level pick, and the isinstance
    guard in ``_family_dated_dates`` keeps it inert under streamlit-
    stubbed test imports."""
    s = normalize_sport_key(sport if sport is not None else get_sport())
    if s == "nfl":
        prefixes = [("nfl_run_engine_markets_", ".csv"),
                    ("nfl_run_engine_monitor_", ".json")]
    else:
        prefixes = [("run_engine_markets_", ".csv"),
                    ("run_engine_monitor_", ".json")]
    return _family_dated_dates(
        s, prefixes, {"owner": owner, "repo": repo, "branch": branch})


def load_nfl_run_engine_markets(sport: str | None = "nfl") -> tuple[pd.DataFrame, str | None]:
    """Newest NFL run-engine slate-serve markets artifact + its YYYYMMDD date.

    Walks the family dates (newest first) and fetches the CSV through the
    shared GitHub raw / local fallback (bare filename — the repo subdir +
    data_delivery/ prefix is applied internally, so the same source config
    serves MLB and NFL artifacts). Returns (empty frame, None) when no
    artifact is reachable — never fabricated. The MLB-only
    ``load_run_engine_markets`` stays untouched (the run-engine slate is a
    per-sport family: MLB ``run_engine_markets_*``, NFL
    ``nfl_run_engine_markets_*``)."""
    s = normalize_sport_key(sport if sport is not None else get_sport())
    cfg = get_source_config()
    for d in _nfl_run_engine_family_dates(s, "markets_csv"):
        raw, _src = _fetch_bytes(f"nfl_run_engine_markets_{d}.csv",
                                 **cfg, sport=s)
        if raw is None:
            continue
        try:
            return pd.read_csv(io.BytesIO(raw)), d
        except Exception:
            continue
    return pd.DataFrame(), None


def load_nfl_run_engine_monitor(sport: str | None = "nfl") -> dict | None:
    """Newest NFL run-engine monitor JSON (``nfl_run_engine_monitor_*.json``).

    Walks the family dates (newest first) and parses the JSON through the
    shared fetch fallback. Returns None when missing/unparseable — the page
    then shows the honest artifact-missing state (never fabricated
    calibration)."""
    s = normalize_sport_key(sport if sport is not None else get_sport())
    cfg = get_source_config()
    for d in _nfl_run_engine_family_dates(s, "markets_monitor_json"):
        raw, _src = _fetch_bytes(f"nfl_run_engine_monitor_{d}.json",
                                 **cfg, sport=s)
        if raw is None:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        return data if isinstance(data, dict) else None
    return None


def load_prediction_history(date_str: str,
                            sport: str | None = None) -> pd.DataFrame:
    """Per-game walk-forward predictions + results (Calibration page table).

    Sport-dispatched: MLB reads ``predictions_history_*.csv``; NFL reads
    ``nfl_predictions_history_*.csv``. Both carry the same per-game contract
    (home_win_prob_model / correct / actual_winner / game_status ...)."""
    s = normalize_sport_key(sport if sport is not None else get_sport())
    prefix = "predictions_history" if s == "mlb" else "nfl_predictions_history"
    cfg = get_source_config()
    data, src = _fetch_bytes(f"{prefix}_{_pick_date(date_str)}.csv",
                             **cfg, sport=s)
    st.session_state["data_source"] = src
    if data is None:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(data))


@st.cache_data(ttl=300, show_spinner=False)
def _history_for_date(date_str: str, owner: str, repo: str, branch: str) -> bytes | None:
    """Latest predictions_history CSV bytes containing rows for ``date_str``.

    Walk-forward history covers EVERY past game, so it can reconstruct a
    simplified board for dates whose full todays_games snapshot is gone.
    """
    dates = sorted(available_dates(owner=owner, repo=repo, branch=branch))
    # Newest artifacts first; fall back to progressively older histories so a
    # pruned repo still serves deep-past dates.
    for cand in reversed(dates):
        data, _ = _fetch_bytes(f"predictions_history_{cand}.csv",
                               owner=owner, repo=repo, branch=branch)
        if data is None:
            continue
        try:
            hist = pd.read_csv(io.BytesIO(data), usecols=["game_date"])
        except (ValueError, pd.errors.EmptyDataError):
            continue
        gd = hist["game_date"].dropna().astype(str).str.replace("-", "")
        if date_str in set(gd):
            return data
    return None


def load_history_games(date_str: str) -> pd.DataFrame:
    """Rebuild a simplified game board for a past date from prediction history."""
    cfg = get_source_config()
    data = _history_for_date(date_str, **cfg)
    if data is None:
        return pd.DataFrame()
    hist = pd.read_csv(io.BytesIO(data))
    gd = hist["game_date"].dropna().astype(str).str.replace("-", "")
    day = hist[gd == date_str].copy()
    if day.empty:
        return pd.DataFrame()

    # Reshape to what normalize_games + the card builder expect.
    ph = pd.to_numeric(day["home_win_prob_model"], errors="coerce")
    day["away_win_prob_model"] = (1.0 - ph).clip(0, 1)
    day["model_correct"] = (
        day["correct"].astype(str).str.lower().isin(("true", "1", "1.0", "yes"))
        if "correct" in day.columns
        else False
    )
    df = normalize_games(day)
    st.session_state["data_source"] = "history"
    return df


def load_power_rankings(date_str: str,
                         sport: str | None = None) -> pd.DataFrame:
    s = normalize_sport_key(sport if sport is not None else get_sport())
    cfg = get_source_config()
    # Sport-specific prefix: MLB ``power_rankings_*.csv``, NFL
    # ``nfl_power_rankings_*.csv`` (same shared page, same 1-based rank shape).
    prefix = "power_rankings" if s == "mlb" else "nfl_power_rankings"
    picked = _pick_artifact_date(date_str, prefix)
    data, src = _fetch_bytes(f"{prefix}_{picked}.csv", **cfg, sport=s)
    st.session_state["data_source"] = src
    st.session_state["power_rankings_date"] = picked
    if data is None:
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(data))
    # Backend writes wins/losses; page reads w/l — accept either spelling.
    if "w" not in df.columns and "wins" in df.columns:
        df["w"] = df["wins"]
    if "l" not in df.columns and "losses" in df.columns:
        df["l"] = df["losses"]
    return df


def load_rl_calibration() -> dict:
    """Latest committed run-line calibration record
    (run_line_calibration_*.json) — the gate evidence behind the per-card
    run-line selector. {} when missing (frontend then renders lines as
    'unverified')."""
    cfg = get_source_config()
    dates = available_dates(**cfg)
    if not dates:
        return {}
    date_str = dates[0]
    data, _src = _fetch_bytes(f"run_line_calibration_{date_str}.json", **cfg)
    if data is None:
        return {}
    try:
        return json.loads(data)
    except Exception:
        return {}


def load_calibration(date_str: str, use_daily: bool = True,
                      sport: str | None = None) -> dict:
    """The latest Calibration record for the active sport.

    Sport-dispatched: MLB ``calibration_*.json``; NFL ``nfl_calibration_*.json``
    (identical shape, so the shared page renders both with one code path)."""
    s = normalize_sport_key(sport if sport is not None else get_sport())
    prefix = "calibration" if s == "mlb" else "nfl_calibration"
    cfg = get_source_config()
    picked = _pick_artifact_date(date_str, prefix)
    data, src = _fetch_bytes(f"{prefix}_{picked}.json", **cfg, sport=s)
    st.session_state["data_source"] = src
    if data is None:
        return {}
    return _normalize_calibration(json.loads(data), picked,
                                  use_daily=use_daily, sport=s)


def _correct_bool(v) -> bool:
    """Coerce a history ``correct`` cell (True/False/1/0 strings) to bool."""
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def _record_from_history(hist: pd.DataFrame) -> dict | None:
    """Today's-record from a decided per-game prediction history: every row
    is a real predicted-vs-actual game that counts toward the summary card.
    None when there is no usable history (page then keeps its defaults)."""
    if hist is None or hist.empty or "correct" not in hist.columns:
        return None
    ok = hist["correct"].map(_correct_bool)
    completed = int(len(ok))
    wins = int(ok.sum())
    return {"wins": wins, "losses": completed - wins, "completed": completed}


def _upsets_from_history(hist: pd.DataFrame) -> list[dict]:
    """Upsets from a decided history: a game where the winner's model
    probability <= UPSET_PROB_THRESHOLD (the same rule the board path uses)."""
    if hist is None or hist.empty:
        return []
    ph = pd.to_numeric(hist.get("home_win_prob_model"), errors="coerce")
    winner, home = hist.get("actual_winner"), hist.get("home_team")
    if ph is None or winner is None or home is None:
        return []
    is_home_w = (winner.astype(str) == home.astype(str))
    prob = ph.where(is_home_w, 1.0 - ph)
    up = hist[prob.notna() & (prob <= UPSET_PROB_THRESHOLD)]
    return [{"team": str(r["actual_winner"]), "prob": float(prob.loc[i])}
            for i, r in up.iterrows()]


def _normalize_calibration(cal: dict, date_str: str, use_daily: bool = True,
                           sport: str | None = None) -> dict:
    """Map the pipeline's calibration JSON onto the Calibration page's schema.

    Pipeline emits:  metrics{auc,brier,logloss,ece}, calibration_buckets[],
                     daily[{date,n_games,wins,losses,metrics{},buckets[]}].
    Page reads:      kpis{auc_roc,brier_score,log_loss,cal_error},
                     calibration_curve[], confidence[{bucket,count,accuracy_pct}],
                     today_record{wins,losses,completed}, upsets[{team,prob}].

    With ``use_daily=True``, a per-day walk-forward entry matching
    ``date_str`` replaces the pooled view (strict point-in-time view for
    that day, used by Today's Games). With ``use_daily=False`` the pooled
    latest snapshot is always shown (Calibration tab behavior).
    """
    cal["_artifact_date"] = date_str
    # Prefer the per-day walk-forward entry when one matches the selected
    # date: it is the strict point-in-time predicted-vs-actual view for
    # that day (fold trained only on prior games).
    day = next((d for d in cal.get("daily", [])
                if str(d.get("date")) == str(date_str)), None)
    if day and use_daily:
        cal["metrics"] = day.get("metrics", {})
        if day.get("buckets"):
            cal["calibration_buckets"] = day["buckets"]
            cal.pop("calibration_curve", None)
        cal["today_record"] = {
            "wins": int(day.get("wins", 0) or 0),
            "losses": int(day.get("losses", 0) or 0),
            "completed": int(day.get("n_games", 0) or 0),
        }

    m = cal.get("metrics", {})
    cal.setdefault("kpis", {
        "auc_roc": m.get("auc"),
        "brier_score": m.get("brier"),
        "log_loss": m.get("logloss"),
        "cal_error": m.get("ece"),
        # Per-day post-hoc twins (present when the daily row carries the
        # prequential calibrated metrics).
        "cal_error_calibrated": m.get("ece_calibrated"),
        "log_loss_calibrated": m.get("logloss_calibrated"),
        "brier_calibrated": m.get("brier_calibrated"),
    })

    curve = cal.get("calibration_curve") or cal.get("calibration_buckets") or []
    cal["calibration_curve"] = curve

    if not cal.get("confidence"):
        cal["confidence"] = [
            {"bucket": b.get("bucket"),
             "count": b.get("count", 0),
             "accuracy_pct": round(float(b.get("mean_actual") or 0) * 100, 1)}
            for b in curve
        ]

    # Today's Record + upsets. MLB derives them from the day's board CSV
    # (real outcomes). NFL's 2026 season is scheduled ahead (no decided board),
    # so the summary card instead derives a real predicted-vs-actual record
    # from the DECIDED OOF/sealed prediction history — never empty.
    if normalize_sport_key(sport if sport is not None else get_sport()) == "nfl":
        hist = load_prediction_history(date_str, sport="nfl")
        rec = _record_from_history(hist)
        if rec:
            cal["today_record"] = rec
        ups = _upsets_from_history(hist)
        if ups:
            cal["upsets"] = ups
        return cal

    try:
        games = load_todays_games(date_str)
    except Exception:
        games = pd.DataFrame()
    if not games.empty:
        win = pd.to_numeric(games.get("home_win"), errors="coerce")
        finished = games[win.notna()]
        completed = len(finished)
        wins = (int(finished["model_correct"].astype(bool).sum())
                if "model_correct" in finished else 0)
        cal.setdefault("today_record",
                       {"wins": wins, "losses": completed - wins,
                        "completed": completed})
        if not cal.get("upsets") and "is_upset" in games.columns:
            probs = pd.to_numeric(games.get("home_win_prob_model"), errors="coerce")
            ups = []
            for idx, row in games[games["is_upset"].fillna(False)].iterrows():
                home_won = win.loc[idx] == 1
                team = row.get("home_team") if home_won else row.get("away_team")
                p = probs.loc[idx] if home_won else 1 - probs.loc[idx]
                ups.append({"team": team,
                            "prob": float(p) if pd.notna(p) else 0.0})
            cal["upsets"] = ups
    return cal


def load_model_monitor(date_str: str,
                        sport: str | None = None) -> dict:
    """Sport-dispatched model-monitor record. MLB ``model_monitor_*.json``;
    NFL ``nfl_model_monitor_*.json`` (the same MLB-shaped schema emitted by
    nfl_monitor.build_model_monitor), so the shared Model Monitor page runs
    both sports unchanged."""
    s = normalize_sport_key(sport if sport is not None else get_sport())
    prefix = "model_monitor" if s == "mlb" else "nfl_model_monitor"
    cfg = get_source_config()
    picked = _pick_artifact_date(date_str, prefix)
    data, src = _fetch_bytes(f"{prefix}_{picked}.json", **cfg, sport=s)
    st.session_state["data_source"] = src
    if data is None:
        return {}
    return json.loads(data)


def load_shap(game_id: str, date_str: str) -> pd.DataFrame:
    cfg = get_source_config()
    # Try the requested date first; fall back to latest available snapshot.
    # SHAP files use a game_id that embeds the date (shap_game_20260821_STL@PHI.csv),
    # so we can't do a simple prefix lookup — try requested date then latest.
    data, _ = _fetch_bytes(f"shap_game_{game_id}.csv", **cfg)
    if data is not None:
        return pd.read_csv(io.BytesIO(data))
    # Extract date from game_id (first 8 digits) and try latest
    dates = available_dates(**cfg)
    if dates and dates[0] != date_str:
        new_gid = game_id.replace(date_str, dates[0]) if date_str in game_id else game_id
        data, _ = _fetch_bytes(f"shap_game_{new_gid}.csv", **cfg)
        if data is not None:
            return pd.read_csv(io.BytesIO(data))
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def parse_date(date_str: str) -> date:
    return date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))


def format_date_long(date_str: str) -> str:
    d = parse_date(date_str)
    return f"{d.strftime('%A, %B')} {d.day}, {d.year}"


def format_date_short(date_str: str) -> str:
    d = parse_date(date_str)
    return f"{d.strftime('%b')} {d.day}, {d.year}"


def start_time_et(iso_utc: str) -> str:
    """Convert a UTC ISO timestamp to an '7:05 PM ET' label."""
    try:
        ts = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        et = ts.astimezone(ZoneInfo("America/New_York"))
        hour = et.hour % 12 or 12
        return f"{hour}:{et.strftime('%M')} {et.strftime('%p')} ET"
    except (ValueError, TypeError):
        return ""


def arrow_nav(dates: list[str]) -> None:
    """Render ◀ date ▶ navigation over the available artifact dates."""
    dates = sorted(dates)
    if not dates:
        st.session_state["selected_date"] = "20260809"
        return
    current = st.session_state.get("selected_date", dates[-1])
    if current not in dates:
        current = dates[-1]
    idx = dates.index(current)
    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        if st.button("◀", key="prev_day", help="Previous day", use_container_width=True):
            st.session_state["selected_date"] = dates[max(0, idx - 1)]
            st.rerun()
    with c2:
        st.markdown(
            f"<div style='text-align:center;font-size:1.05rem;font-weight:600;color:{TEXT};"
            f"padding:0.45rem 0;border:1px solid {BORDER};border-radius:10px;background:{CARD_BG};'>"
            f"{format_date_long(current)}</div>",
            unsafe_allow_html=True,
        )
    with c3:
        if st.button("▶", key="next_day", help="Next day", use_container_width=True):
            st.session_state["selected_date"] = dates[min(len(dates) - 1, idx + 1)]
            st.rerun()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def pct(x, digits: int = 0) -> str:
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def american(odds) -> str:
    try:
        v = int(odds)
        return f"{v:+d}"
    except (TypeError, ValueError):
        return "—"


def edge_str(edge) -> str:
    try:
        return f"{float(edge):+.2f}"
    except (TypeError, ValueError):
        return "+0.00"


def edge_color(edge) -> str:
    try:
        v = float(edge)
    except (TypeError, ValueError):
        return SLATE
    if v > 0.001:
        return PRIMARY
    if v < -0.001:
        return RED
    return BLUE


def record_pct(w: int, l: int) -> str:
    total = w + l
    return f"{w / total:.3f}" if total else ".000"


# ---------------------------------------------------------------------------
# Altair dark theme
# ---------------------------------------------------------------------------

def _fb_theme():
    return {
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent", "continuousWidth": 400, "continuousHeight": 260},
            "axis": {
                "labelColor": SLATE, "titleColor": TEXT, "gridColor": BORDER,
                "domainColor": "#334155", "tickColor": "#334155", "labelFontSize": 11,
            },
            "legend": {"labelColor": SLATE, "titleColor": TEXT, "labelFontSize": 11},
            "title": {"color": TEXT, "fontSize": 15, "anchor": "start"},
            "range": {"category": [BLUE, PRIMARY, AMBER, "#A78BFA", RED]},
        }
    }


alt.themes.register("fb_dark", _fb_theme)
alt.themes.enable("fb_dark")


def show_chart(chart: alt.Chart) -> None:
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:  # older Streamlit
        st.altair_chart(chart, use_container_width=True)


def shap_chart(shap_df: pd.DataFrame, max_features: int = 12) -> alt.Chart:
    """Horizontal SHAP bar chart: green positive bars, red negative bars,
    signed value labels, sorted by |value| descending."""
    d = shap_df.copy()
    if d.empty:
        return None
    d = d[d["shap_value"].notna()].copy()
    d["abs"] = d["shap_value"].abs()
    d = d.sort_values("abs", ascending=False).head(max_features).reset_index(drop=True)
    d["label"] = d["shap_value"].map(lambda v: f"{v:+.2f}")

    # Explicit shared x-domain avoids Vega-Lite "Infinite extent" warnings
    # from empty per-layer scale extents in layered charts.
    pad = max(0.005, float(d["shap_value"].abs().max()) * 0.08)
    domain = [float(d["shap_value"].min()) - pad, float(d["shap_value"].max()) + pad]
    x_enc = alt.X("shap_value:Q", title="SHAP value (favored team's win probability)", scale=alt.Scale(domain=domain))

    bars = alt.Chart(d).mark_bar(cornerRadiusEnd=3).encode(
        x=x_enc,
        y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=170)),
        color=alt.condition("datum.shap_value >= 0", alt.value(PRIMARY), alt.value(RED)),
    )
    pos_data, neg_data = d[d["shap_value"] >= 0], d[d["shap_value"] < 0]
    layers = [bars]
    if not pos_data.empty:
        layers.append(
            alt.Chart(pos_data)
            .mark_text(dx=5, align="left", color="#34D399", fontSize=11)
            .encode(x=x_enc, y=alt.Y("feature:N", sort=None), text="label:N")
        )
    if not neg_data.empty:
        layers.append(
            alt.Chart(neg_data)
            .mark_text(dx=-5, align="right", color="#F87171", fontSize=11)
            .encode(x=x_enc, y=alt.Y("feature:N", sort=None), text="label:N")
        )
    return alt.layer(*layers).properties(height=max(120, 26 * len(d)))


# ---------------------------------------------------------------------------
# Shared page chrome (CSS)
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {PAGE_BG}; }}
        .block-container {{ padding-top: 1.4rem; }}
        .fb-card {{
            background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 14px;
            padding: 14px 16px 12px; margin-bottom: 12px;
        }}
        .fb-top {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }}
        .fb-top .spacer {{ flex: 1; }}
        .fb-tag {{ color: {SLATE}; font-size: 0.78rem; }}
        .fb-pill {{ border-radius: 999px; padding: 2px 9px; font-size: 0.72rem; font-weight: 700; white-space: nowrap; }}
        .fb-pill.correct {{ background: rgba(16,185,129,.15); color: #34D399; }}
        .fb-pill.miss {{ background: rgba(239,68,68,.15); color: #F87171; }}
        .fb-pill.final {{ background: rgba(59,130,246,.15); color: #93C5FD; }}
        .fb-pill.upset {{ background: rgba(245,158,11,.18); color: #FBBF24; }}
        .fb-pill.coinflip {{ background: rgba(250,204,21,.18); color: #FDE047; }}
        .fb-pill.live {{ background: rgba(239,68,68,.18); color: #F87171; }}
        .fb-pill.pick {{ background: rgba(59,130,246,.9); color: #fff; }}
        .fb-score {{ display: flex; align-items: center; justify-content: center; gap: 26px; margin: 6px 0 4px; }}
        .fb-score .side {{ text-align: center; }}
        .fb-score .num-wrap {{ display: flex; align-items: center; justify-content: center; }}
        .fb-score .win-bar {{ width: 4px; border-radius: 3px; height: 26px; margin-right: 7px; }}
        .fb-score .num {{ font-size: 2.1rem; font-weight: 800; line-height: 1; color: {TEXT}; }}
        .fb-score .abbr {{ color: {SLATE}; font-size: 0.8rem; letter-spacing: 1px; }}
        .fb-score .mid {{ color: {SLATE}; font-size: 0.95rem; font-weight: 600; }}
        .fb-team {{ display: flex; align-items: center; gap: 8px; margin: 7px 0 3px; }}
        .fb-accent {{ width: 4px; border-radius: 3px; align-self: stretch; min-height: 26px; }}
        .fb-team .name {{ font-weight: 700; color: {TEXT}; }}
        .fb-team .sub {{ color: {SLATE}; font-size: 0.8rem; }}
        .fb-team .pct {{ margin-left: auto; font-weight: 800; font-size: 1.05rem; }}
        .fb-bar {{ height: 7px; border-radius: 5px; background: #0F172A; overflow: hidden; display: flex; margin: 3px 0 2px; }}
        .fb-bar .fill {{ height: 100%; }}
        .fb-pregame {{ text-align: center; color: #64748B; font-size: 0.78rem; margin: 6px 0 8px; }}
        .fb-pitchers {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 4px 0; }}
        .fb-pitcher {{ background: #0F172A; border: 1px solid {BORDER}; border-radius: 9px; padding: 7px 9px; }}
        .fb-pitcher .pname {{ font-weight: 700; color: {TEXT}; font-size: 0.86rem; }}
        .fb-pitcher .pstats {{ color: {SLATE}; font-size: 0.76rem; margin-top: 2px; }}
        .fb-venue {{ color: {SLATE}; font-size: 0.8rem; margin: 6px 0 2px; }}
        .fb-runengine {{ display: flex; flex-wrap: wrap; gap: 4px 14px; align-items: center; background: #0F172A; border: 1px solid {BORDER}; border-radius: 9px; padding: 7px 10px; margin: 2px 0 8px; color: {TEXT}; font-size: 0.8rem; }}
        .fb-runengine .re-label {{ color: {AMBER}; font-size: 0.68rem; font-weight: 800; letter-spacing: 1px; }}
        .fb-runengine .re-na {{ color: {SLATE}; font-style: italic; }}
        .fb-banner {{ border-radius: 10px; padding: 8px 12px; text-align: center; font-size: 0.85rem; font-weight: 700; }}
        .fb-banner.green {{ background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.45); color: #34D399; }}
        .fb-banner.red {{ background: rgba(239,68,68,.10); border: 1px solid rgba(239,68,68,.45); color: #F87171; }}
        .fb-banner.amber {{ background: rgba(250,204,21,.10); border: 1px solid rgba(250,204,21,.5); color: #FDE047; }}
        .fb-banner.blue {{ background: rgba(59,130,246,.10); border: 1px solid rgba(59,130,246,.45); color: #93C5FD; }}
        .fb-kpi {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px; text-align: center; }}
        .fb-kpi .label {{ color: {SLATE}; font-size: 0.72rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }}
        .fb-kpi .value {{ font-size: 1.9rem; font-weight: 800; margin: 4px 0 2px; }}
        .fb-kpi .cap {{ color: {SLATE}; font-size: 0.72rem; }}
        .fb-box {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 12px; padding: 12px 16px; }}
        .fb-status-pill {{ border-radius: 999px; padding: 1px 10px; font-size: 0.72rem; font-weight: 700; }}
        .fb-status-pill.ok {{ background: rgba(16,185,129,.15); color: #34D399; }}
        .fb-status-pill.warn {{ background: rgba(245,158,11,.15); color: #FBBF24; }}
        .fb-status-pill.alert {{ background: rgba(239,68,68,.15); color: #F87171; }}
        table.fb-table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
        table.fb-table th {{ text-align: left; color: {SLATE}; font-size: 0.72rem; letter-spacing: 1px; text-transform: uppercase; padding: 8px 10px; border-bottom: 1px solid {BORDER}; }}
        table.fb-table td {{ padding: 8px 10px; border-bottom: 1px solid rgba(30,41,59,.6); color: {TEXT}; }}
        table.fb-table tr:nth-child(even) td {{ background: rgba(30,41,59,.25); }}
        .fb-accent-cell {{ display: inline-flex; align-items: center; gap: 8px; }}
        .fb-accent-line {{ width: 4px; border-radius: 3px; height: 20px; display: inline-block; }}
        /* Branding above the dashboard list: Streamlit renders the sidebar
           page list (stSidebarNav) ABOVE user sidebar content by default.
           Flip the flex order so the brand header (stSidebarUserContent's
           first block) sits above every dashboard choice on load. */
        [data-testid="stSidebarContent"] {{ display: flex; flex-direction: column; }}
        [data-testid="stSidebarHeader"] {{ order: 1; flex-shrink: 0; }}
        [data-testid="stSidebarUserContent"] {{ order: 2; flex-shrink: 0; }}
        [data-testid="stSidebarNav"] {{ order: 3; flex-shrink: 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header() -> None:
    """The app's single branding/logo block, rendered at the top of the
    sidebar ABOVE the dashboard page list.

    Streamlit stacks the sidebar page list (``stSidebarNav``) above user
    sidebar content by default, so the sidebar reorder CSS in ``inject_css``
    flips the flex order: header -> brand block -> page list. Title/subtitle
    come from the ACTIVE sport's registry entry (``sport_config()``), so a
    new sport needs zero UI-code changes here. MLB renders byte-identically
    to the pre-registry header ("⚾ MLB Predictions" / "MLB betting model
    dashboard").
    """
    cfg = sport_config()
    st.markdown(
        f"<div style='font-size:1.25rem;font-weight:800;color:#E2E8F0;'>"
        f"{cfg['emoji']} {cfg['title']}</div>"
        f"<div style='color:#64748B;font-size:0.8rem;margin-bottom:10px;'>"
        f"{cfg['subtitle']}</div>",
        unsafe_allow_html=True,
    )


def render_source_note() -> None:
    """Empty-state note shown ONLY when the selected sport ships no committed
    artifacts at all (defensive; the pipeline always commits a snapshot). The
    former fetch-failure warning caption is gone — the sidebar instead shows a
    sport-aware "Last updated" line via ``render_last_updated``.
    """
    src = st.session_state.get("data_source", "")
    if src == "missing" and not list(LOCAL_DATA_DIR.glob("todays_games_*.csv")):
        st.caption("⚠️ No artifacts found")


def _full_run_timestamp(d: dict) -> Optional[datetime]:
    """Parse a full-run timestamp off a sport JSON record (e.g.
    ``margin_reliability_*.json`` carries ``generated``), or None."""
    for key in ("generated", "created_utc", "timestamp", "run_time",
                "created_at"):
        raw = d.get(key)
        if not raw:
            continue
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            continue
    return None


def _stamp_suffixes(path: Path) -> set[str]:
    """Any trailing YYYYMMDD suffix(es) on a dated artifact filename."""
    m = re.search(r"_(\d{8})$", path.stem)
    return {m.group(1)} if m else set()


def _max_stamp(paths) -> Optional[str]:
    """The newest YYYYMMDD suffix across the given artifact paths, or None."""
    stamps: set[str] = set()
    for p in paths:
        stamps |= _stamp_suffixes(p)
    return max(stamps) if stamps else None


def _is_snapshot_record(name: str) -> bool:
    """True when ``name`` is a PRIMARY data-snapshot JSON record.

    Only a sport's primary snapshot families carry the data refresh — the
    pipeline output the sidebar reports (MLB: ``calibration_``,
    ``model_monitor_``, ``run_engine_monitor_``, ``run_engine_markets_``
    meta; NFL: ``nfl_moneyline_v1_``, ``nfl_feature_v1_``). Stand-alone
    read-only diagnostics — ``margin_reliability_*``, ``deep_over_recheck_*``,
    ``*_cull_diagnostic_*``, ``rolling_brier_*``, ``features_metadata_*`` —
    are EXCLUDED: their ``generated`` field records when the analysis was
    authored, NOT when the pipeline refreshed predictions, so trusting it
    makes the sidebar 'Last updated' drift from the actual data snapshot
    (this is the 2026-08-30 stale-time root cause).
    """
    return name.startswith((
        "calibration_", "model_monitor_", "run_engine_monitor_",
        "run_engine_markets_", "nfl_moneyline_v1_", "nfl_feature_v1_",
    ))


def _is_snapshot_artifact(name: str) -> bool:
    """True when ``name`` is one of a sport's PRIMARY data-snapshot artifacts.

    ``_is_snapshot_record`` covers the JSON record families (their
    ``created_utc``/``generated`` set the sidebar time); this adds the dated
    ``todays_games_*`` / ``predictions_history_*`` CSV snapshot artifacts so
    their file mtime can stand in for freshness too. Stand-alone diagnostics
    are excluded (their mtime is likewise never the data refresh)."""
    return _is_snapshot_record(name) or name.startswith((
        "todays_games_", "predictions_history_",
    ))


def _mtime_utc(p: Path) -> Optional[datetime]:
    """A Path's file mtime as a timezone-aware UTC datetime, or None."""
    try:
        return datetime.fromtimestamp(p.stat().st_mtime,
                                      tz=ZoneInfo("UTC"))
    except (OSError, ValueError):
        return None


def _snapshot_dated_paths(sport_dir: Path) -> list[Path]:
    """Dated PRIMARY snapshot artifact files (csv+json) under ``sport_dir``."""
    return [p for p in list(sport_dir.glob("*.csv"))
            + list(sport_dir.glob("*.json"))
            if _stamp_suffixes(p) and _is_snapshot_artifact(p.name)]


def _last_refresh_for_dir(sport_dir: Path) -> Optional[datetime]:
    """Newest refresh datetime for one sport's committed ``data_delivery`` dir.

    Unified single-source logic shared by every sport so the sidebar renders
    the SAME full Eastern timestamp format everywhere (NFL kept as-is):
    (1) newest full-run timestamp on any PRIMARY snapshot JSON record
    (``created_utc``/``generated`` — e.g. NFL ``nfl_moneyline_v1_*``;
    diagnostics are excluded by ``_is_snapshot_record``), else
    (2) the newest PRIMARY snapshot artifact's FILE MTIME (MLB's CSV
    artifacts carry no ``created_utc``, so mtime is its freshness proxy),
    else (3) fall back to the strict date-only YYYYMMDD suffix. Returns None
    when the dir carries no dated snapshot artifacts.
    """
    best_full: Optional[datetime] = None
    for p in sport_dir.glob("*_*.json"):
        if not _is_snapshot_record(p.name):
            continue
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        ts = _full_run_timestamp(obj) if isinstance(obj, dict) else None
        if ts and (best_full is None or ts > best_full):
            best_full = ts
    if best_full is not None:
        return best_full

    # No persisted run time (MLB artifacts are date-stamped CSVs): use the
    # newest snapshot artifact's file mtime so MLB shows a full Eastern
    # timestamp, exactly like NFL's created_utc path.
    mts = [_mtime_utc(p) for p in _snapshot_dated_paths(sport_dir)]
    mts = [t for t in mts if t is not None]
    if mts:
        return max(mts)

    market = _max_stamp(sport_dir.glob("run_engine_markets_*.csv"))
    if market:
        return datetime.strptime(market, "%Y%m%d")
    for pat in ("todays_games_*.csv", "predictions_history_*.csv",
                "calibration_*.json", "model_monitor_*.json"):
        s = _max_stamp(sport_dir.glob(pat))
        if s:
            return datetime.strptime(s, "%Y%m%d")
    stamps: set[str] = set()
    for p in list(sport_dir.glob("*.csv")) + list(sport_dir.glob("*.json")):
        stamps |= _stamp_suffixes(p)
    if stamps:
        return datetime.strptime(max(stamps), "%Y%m%d")
    return None


# Eastern US timezone for the sidebar 'Last updated' line. Resolved via
# zoneinfo (never a hardcoded offset) so DST is automatic: EDT (UTC-4) in
# daylight time, EST (UTC-5) in standard time.
_EASTERN_ZONE = ZoneInfo("America/New_York")


def _is_date_only(dt: datetime) -> bool:
    """True when a refresh value is a date-only artifact (no time component).

    The ``run_engine_markets_*_YYYYMMDD`` suffix path (and similar dated
    filenames) carries only a date — parsed as a naive midnight. A naive
    datetime at 00:00:00 is the date-only marker; the formatter then keeps
    the date-only display and does NOT fabricate a time."""
    return (dt.tzinfo is None and dt.hour == 0 and dt.minute == 0
            and dt.second == 0 and dt.microsecond == 0)


def _to_eastern(dt: datetime) -> datetime:
    """Convert a refresh datetime to Eastern US time, DST-aware via zoneinfo.

    Naive timestamps are assumed to be UTC first (the Kaggle/Colab pipeline
    writes naive UTC ``generated``/``created_utc`` values), then converted.
    ``%Z`` on the result yields EDT or EST depending on the calendar date."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(_EASTERN_ZONE)


def _format_refresh(dt: Optional[datetime]) -> str:
    """Display string for a refresh time.

    Full timestamps render in Eastern US time with seconds + tz abbreviation,
    e.g. 'Aug 30, 2026, 10:05:00 PM EDT'. Date-only artifacts (no time
    component) keep the plain date 'Aug 30, 2026' — a time is never
    fabricated. Missing → 'Last updated: —'."""
    if dt is None:
        return "Last updated: —"
    if _is_date_only(dt):
        return "Last updated: " + dt.strftime("%b %d, %Y")
    et = _to_eastern(dt)
    hour = et.hour % 12 or 12
    ampm = "AM" if et.hour < 12 else "PM"
    return (f"Last updated: {et:%b} {et.day}, {et.year}, "
            f"{hour}:{et:%M}:{et:%S} {ampm} {et:%Z}")


def last_refresh_time(sport_key: str | None = None) -> str:
    """Sport-aware 'Last updated' string for the sidebar.

    Resolves the selected sport to its committed ``data_delivery`` snapshot
    and reports the newest refresh time (see ``_last_refresh_for_dir``).
    Defaults to the active sport (``get_sport``). Reads only committed
    artifacts — the same snapshot the sidebar serves regardless of
    GitHub-vs-local fetch. Missing/empty → 'Last updated: —'.
    """
    s = normalize_sport_key(sport_key if sport_key is not None else get_sport())
    cfg = resolve_sport(s)
    sport_dir = REPO_ROOT / cfg["repo_subdir"] / "data_delivery"
    if not sport_dir.is_dir():
        return _format_refresh(None)
    return _format_refresh(_last_refresh_for_dir(sport_dir))


def render_last_updated(sport_key: str | None = None) -> None:
    """Small muted 'Last updated' caption in the sidebar, wired to the active
    sport toggle so it resolves that sport's artifact set."""
    st.caption(last_refresh_time(sport_key))


# ---------------------------------------------------------------------------
# Feature descriptions (Model Monitor PSI table)
# ---------------------------------------------------------------------------

FEATURE_DESCRIPTIONS = {
    # 1. Baseline
    "is_home": "Always 1 — anchors the ~53% MLB home-field win advantage",
    # 2–4. Core pre-game diffs (home − away; positive = home advantage)
    "win_pct_diff": "Home win% − away win% (smoothed to .500 early season)",
    "elo_diff": "Home Elo − away Elo (skill-gap anchor, updated each game)",
    "rest_days_diff": "Home rest days − away rest days (schedule fatigue)",
    # 5–8. SP season-to-date + last-5-start diffs
    "sp_era_diff": "Home SP season-to-date ERA − away SP",
    "sp_era_5g_diff": "Home SP last-5-start ERA − away SP (recent form)",
    "sp_k9_diff": "Home SP season-to-date K/9 − away SP",
    "sp_k9_5g_diff": "Home SP last-5-start K/9 − away SP (recent form)",
    # 7–9. SP trailing-3-game stuff diffs
    "sp_fbvelo_diff": "Home SP fastball velo (last 3 starts) − away SP (mph)",
    "sp_fbpct_diff": "Home SP fastball usage (last 3 starts) − away SP",
    "sp_whiff_diff": "Home SP whiff rate (last 3 starts) − away SP",
    # 10–11. SP xwOBA diffs (contact quality allowed)
    "sp_xwoba_diff": "Home SP last-6-start xwOBA allowed − away SP",
    "sp_xwoba_vs_l_diff": "Home SP xwOBA vs LHB (season to date) − away SP",
    # 12–14. Lineup wOBA diffs (projected top-9, shrunk toward league mean)
    "lineup_woba_mean_diff": "Home lineup avg wOBA − away lineup avg wOBA",
    "lineup_woba_top3_diff": "Home top-3 hitter wOBA − away top-3 hitter wOBA",
    "lineup_woba_std_diff": "Home lineup wOBA dispersion − away lineup dispersion",
    # 15. Team rolling wOBA diff
    "woba_30g_diff": "Home team 30-game wOBA − away team 30-game wOBA",
    # 16–19. Bullpen diffs (workload + quality)
    "bullpen_whip_diff": "Home bullpen 10-game WHIP − away bullpen (lower = better)",
    "bullpen_whip_3g_diff": "Home bullpen 3-game WHIP − away bullpen (short-term form)",
    "bullpen_pitches_diff": "Home bullpen 3-day pitch count − away (fatigue signal)",
    "bullpen_ip_diff": "Home bullpen 3-day IP − away bullpen IP",
    # 20–22. Team contact form diffs (trailing 15g, balls in play only)
    "team_barrel_diff": "Home barrel% (15g) − away barrel% (quality of contact)",
    "team_hardhit_diff": "Home hard-hit% (15g) − away hard-hit%",
    "team_exitvelo_diff": "Home avg exit velo (15g) − away avg exit velo (mph)",
    # 23. Lineup handedness matchup advantage
    "lineup_handedness_matchup_advantage": "Lineup OPS vs tonight's opposing starter hand, home − away",
    # 24. Travel fatigue & closer availability
    "travel_fatigue_diff": "Home timezone crossings (last 3 days) − away (schedule fatigue)",
    "closer_availability_diff": "Home closer available − away closer available (late-inning edge)",
    # 25. Dome neutral flag (prevents weather hallucination indoors)
    "dome_is_neutral": "1 if home park is a fixed dome/closed roof, 0 if open-air",
    # 26–28. Context interaction features
    "dome_is_neutral": "1 if home park is a fixed dome/closed roof, 0 if open-air",
    # 25–27. Context interaction features
    "park_factor_slug_diff": "Home park SLG factor × lineup top-3 wOBA diff (hitter-friendly parks amplify lineup edges)",
    "wind_advantage_flyball_factor": "Wind direction multiplier × SP ERA diff (flyball risk in windy conditions)",
    "air_density_velocity_boost": "Stadium air density × SP velo diff (cold/thin air affects velocity)",
    # 29–32. Derived interaction features
    "bullpen_meltdown_risk": "Bullpen pitches diff × WHIP diff (overworked + low quality = meltdown)",
    "pitcher_regression_indicator": "SP velo diff × ERA diff (physical drop vs surface results = regression)",
    "lineup_depth_multiplier": "Lineup mean wOBA diff × top-3 wOBA diff (star power × depth)",
    "ace_efficiency_factor": "SP K/9 diff × whiff rate diff (high strikeout volume from raw stuff)",
}


NFL_FEATURE_DESCRIPTIONS = {
    # The served market-free 12-pool (+ the is_home report anchor) — plain-
    # language twins of nfl_features.CANONICAL_SOURCE, kept on the frontend
    # so the shared Feature Drift table renders descriptions for both sports.
    "is_home": "Constant 1 — anchors the home-field edge",
    "elo_diff": "Home Elo − away Elo (point-in-time rating gap, updated each game)",
    "win_pct_diff": "Home trailing win% (last 12 games) − away",
    "rest_days_diff": "Home days since prior game − away (schedule fatigue)",
    "is_dome_home": "1 if the home venue has a roof, 0 if open-air",
    "ewm_net_pts_diff": "Home−away net points/game, exponentially weighted (recent form)",
    "ewm_ypp_diff": "Home−away yards/play, exponentially weighted (recent form)",
    "pace_plays_min_diff": "Home−away plays per minute (pace of play)",
    "rest_short_diff": "Short-rest edge: home on <7 days rest − away flag",
    "div_game": "1 for division matchups (familiarity/rivalry factor)",
    "travel_miles_diff": "Home−away stadium distance in miles (travel fatigue)",
    "altitude_home": "Home venue elevation in meters (thin-air effects)",
    "prime_time": "1 for evening kickoffs (ET hour ≥ 17)",
}


def describe_feature(name: str, sport: str = "mlb") -> str:
    """Human description for a feature column like 'sp_era_5g_diff'.

    Sport-dispatched: MLB and NFL share this page but describe their own
    served pools (MLB's dict is keyed on MLB feature names; NFL's on the
    12-feature market-free pool). Exact match wins first; legacy per-side
    names (('sp_era_5g_home')) strip the _home/_away slot suffix and note
    which side of the matchup the value describes. 'is_home' is a baseline
    feature whose name genuinely ends in '_home' — the exact match must win
    so it is not mangled into a name-repeating label. Default sport is MLB,
    so existing callers (and MLB-only pages) are byte-unchanged.
    """
    s = str(name or "").strip()
    table = NFL_FEATURE_DESCRIPTIONS if sport == "nfl" else FEATURE_DESCRIPTIONS
    base = table.get(s)
    if base is not None:
        return base
    for suf in ("_home", "_away"):
        if s.endswith(suf):
            base = table.get(s[: -len(suf)])
            if base is not None:
                return f"{base} — home team" if suf == "_home" else f"{base} — away team"
            break
    for k, v in table.items():
        if s.startswith(k):
            return v
    return name


def feature_weight_pct(row: dict) -> str:
    """Formatted blend weight for a drift row ('—' when unavailable)."""
    v = row.get("weight_pct")
    try:
        return f"{float(v):.2f}%"
    except (TypeError, ValueError):
        return "—"
