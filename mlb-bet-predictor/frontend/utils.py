"""Shared frontend helpers.

This module is the single place the four pages talk to the artifact sink
(``data_delivery``). It reads artifacts from **raw.githubusercontent.com**
URLs when a repo is configured, and transparently falls back to the sample
artifacts shipped in the repo's local ``data_delivery/`` folder so the app
always renders, even offline or before the first Colab push.

Only pandas / requests / altair / streamlit are used — no sklearn, xgboost,
lightgbm, or shap (heavy ML stays in the backend).
"""

from __future__ import annotations

import io
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_DATA_DIR = ROOT_DIR / "data_delivery"
# Artifacts live under mlb-bet-predictor/data_delivery/ inside the GitHub repo
REPO_SUBDIR = "mlb-bet-predictor"

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
    return "Local sample data (no GitHub repo configured)"


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
        return gl[["game_id", "home_score", "away_score"]]
    except Exception:
        return pd.DataFrame(columns=["game_id", "home_score", "away_score"])


# ---------------------------------------------------------------------------
# Artifact loading (GitHub raw -> local sample fallback)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_bytes(relpath: str, owner: str, repo: str, branch: str):
    """Fetch one artifact. Returns (bytes | None, source)."""
    if owner and repo:
        url = (f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}"
               f"/{REPO_SUBDIR}/data_delivery/{relpath}")
        try:
            resp = requests.get(url, timeout=15)
            if resp.ok:
                return resp.content, "github"
        except requests.RequestException:
            pass
    local = LOCAL_DATA_DIR / relpath
    if local.exists():
        return local.read_bytes(), "local"
    return None, "missing"


@st.cache_data(ttl=300, show_spinner=False)
def available_dates(owner: str, repo: str, branch: str) -> list[str]:
    """All artifact dates (YYYYMMDD), newest first, from GitHub API or local dir."""
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
                        dates.add(name[len("todays_games_"):-len(".csv")])
        except requests.RequestException:
            pass
    for p in LOCAL_DATA_DIR.glob("todays_games_*.csv"):
        dates.add(p.name[len("todays_games_"):-len(".csv")])

    # Merge walk-forward calibration days so history stays selectable even
    # when only one todays_games snapshot exists in the repo.
    if dates:
        try:
            cal_bytes, _ = _fetch_bytes(
                f"calibration_{max(dates)}.json",
                owner=owner, repo=repo, branch=branch,
            )
            if cal_bytes:
                for entry in json.loads(cal_bytes).get("daily", []):
                    d = str(entry.get("date", ""))
                    if len(d) == 8 and d.isdigit():
                        dates.add(d)
        except (ValueError, TypeError, KeyError):
            pass

    # Merge per-game prediction-history dates so ANY past date that has
    # predictions is navigable on Today's Games — even when its full card
    # artifact was never pushed or has been pruned.
    if dates:
        try:
            hist_bytes, _ = _fetch_bytes(
                f"predictions_history_{max(dates)}.csv",
                owner=owner, repo=repo, branch=branch,
            )
            if hist_bytes:
                hist = pd.read_csv(io.BytesIO(hist_bytes), usecols=["game_date"])
                for d in hist["game_date"].dropna().astype(str):
                    d = d.replace("-", "")
                    if len(d) == 8 and d.isdigit():
                        dates.add(d)
        except (ValueError, TypeError, KeyError, pd.errors.EmptyDataError):
            pass

    return sorted(dates, reverse=True)


def _pick_date(date_str: str) -> str:
    dates = available_dates(**get_source_config())
    if date_str and date_str in dates:
        return date_str
    if dates:
        return dates[0]
    return "20260809"  # bundled sample


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


def load_todays_games(date_str: str) -> pd.DataFrame:
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


def load_prediction_history(date_str: str) -> pd.DataFrame:
    """Per-game walk-forward predictions + results (Calibration page table)."""
    cfg = get_source_config()
    data, src = _fetch_bytes(f"predictions_history_{_pick_date(date_str)}.csv", **cfg)
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


def load_power_rankings(date_str: str) -> pd.DataFrame:
    cfg = get_source_config()
    picked = _pick_artifact_date(date_str, "power_rankings")
    data, src = _fetch_bytes(f"power_rankings_{picked}.csv", **cfg)
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


def load_calibration(date_str: str) -> dict:
    cfg = get_source_config()
    picked = _pick_artifact_date(date_str, "calibration")
    data, src = _fetch_bytes(f"calibration_{picked}.json", **cfg)
    st.session_state["data_source"] = src
    if data is None:
        return {}
    return _normalize_calibration(json.loads(data), picked)


def _normalize_calibration(cal: dict, date_str: str) -> dict:
    """Map the pipeline's calibration JSON onto the Calibration page's schema.

    Pipeline emits:  metrics{auc,brier,logloss,ece}, calibration_buckets[],
                     daily[{date,n_games,wins,losses,metrics{},buckets[]}].
    Page reads:      kpis{auc_roc,brier_score,log_loss,cal_error},
                     calibration_curve[], confidence[{bucket,count,accuracy_pct}],
                     today_record{wins,losses,completed}, upsets[{team,prob}].
    """
    cal["_artifact_date"] = date_str
    # Prefer the per-day walk-forward entry when one matches the selected
    # date: it is the strict point-in-time predicted-vs-actual view for
    # that day (fold trained only on prior games).
    day = next((d for d in cal.get("daily", [])
                if str(d.get("date")) == str(date_str)), None)
    if day:
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

    # Today's Record + upsets derive from the games CSV (has real outcomes).
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


def load_model_monitor(date_str: str) -> dict:
    cfg = get_source_config()
    picked = _pick_artifact_date(date_str, "model_monitor")
    data, src = _fetch_bytes(f"model_monitor_{picked}.json", **cfg)
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
        .fb-odds {{ display: flex; justify-content: space-between; align-items: center; margin: 4px 0 8px; color: {TEXT}; font-size: 0.85rem; }}
        .fb-odds .edge {{ font-weight: 700; }}
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_source_note() -> None:
    src = st.session_state.get("data_source", "")
    if src == "github":
        note, icon = "Streaming from GitHub raw URLs", "🌐"
    elif src == "local":
        note, icon = "Showing bundled sample data (offline fallback)", "📦"
    elif src == "missing" and not list(LOCAL_DATA_DIR.glob("todays_games_*.csv")):
        note, icon = "No artifacts found", "⚠️"
    else:
        note, icon = "Showing bundled sample data (offline fallback)", "📦"
    st.caption(f"{icon} {note}")


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
    # 5–8. SP career + trailing-30g diffs
    "sp_era_diff": "Home SP career ERA − away SP career ERA",
    "sp_era_30g_diff": "Home SP 30-start ERA − away SP (recent form)",
    "sp_k9_diff": "Home SP career K/9 − away SP career K/9",
    "sp_k9_30g_diff": "Home SP 30-start K/9 − away SP (recent form)",
    # 7–9. SP trailing-3-game stuff diffs
    "sp_fbvelo_diff": "Home SP fastball velo (last 3 starts) − away SP (mph)",
    "sp_fbpct_diff": "Home SP fastball usage (last 3 starts) − away SP",
    "sp_whiff_diff": "Home SP whiff rate (last 3 starts) − away SP",
    # 10–11. SP xwOBA diffs (contact quality allowed)
    "sp_xwoba_diff": "Home SP season xwOBA allowed − away SP",
    "sp_xwoba_vs_l_diff": "Home SP xwOBA vs LHB − away SP xwOBA vs LHB",
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


def describe_feature(name: str) -> str:
    """Human description for a feature column like 'sp_era_30g_diff'.

    Exact diff-name matches win first (the current model layout). For legacy
    per-side names ('sp_era_30g_home'), strip the _home/_away slot suffix and
    note which side of the matchup the value describes. 'is_home' is a
    baseline feature whose name genuinely ends in '_home' — the exact match
    must win so it is not mangled into a name-repeating label.
    """
    s = str(name or "").strip()
    base = FEATURE_DESCRIPTIONS.get(s)
    if base is not None:
        return base
    for suf in ("_home", "_away"):
        if s.endswith(suf):
            base = FEATURE_DESCRIPTIONS.get(s[: -len(suf)])
            if base is not None:
                return f"{base} — home team" if suf == "_home" else f"{base} — away team"
            break
    for k, v in FEATURE_DESCRIPTIONS.items():
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
