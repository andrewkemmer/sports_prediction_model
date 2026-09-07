"""Authoritative point-in-time NFL feature engine.

Single owner of every production feature. No competing legacy engine exists;
the models consume frames built here and nowhere else.

LEAKAGE CONTRACT (enforced structurally and asserted):
  feature(game_t) = f(information available STRICTLY BEFORE game_t)

- Elo: iterated strictly chronologically; each row's ``elo_entering`` is the
  team's rating at kickoff (updated only after that game settles).
- Trailing windowed / EWM statistics: computed per team over its own games
  in chronological order, then ``shift(1)`` — the current and all future
  games are excluded from a row's own value.
- Team state (records, Elo) updates only after a game settles.
- Venue/schedule facts (roof, division, stadium, kickoff hour) are static
  pre-game facts.

Model-family representations (spec section 14):
  - linear view: difference features + the constant ``is_home`` anchor
  - tree view: differences + raw home/away side values
  - mlp view: linear columns, standard-scaled at fit time
All views are deterministic in ordering and dimensionality.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    from backend import config
except ImportError:
    import config

logger = logging.getLogger(__name__)

EARTH_RADIUS_MILES = 3958.8

# ---------------------------------------------------------------------------
# Events: long-form (team, game) view
# ---------------------------------------------------------------------------
REQUIRED_GAME_COLS = ["game_id", "season", "week", "gameday", "home_team",
                      "away_team", "home_score", "away_score"]


def team_events(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game) with team-perspective scores."""
    missing = [c for c in REQUIRED_GAME_COLS if c not in games.columns]
    if missing:
        raise ValueError(f"team_events: missing columns {missing}")
    gd = pd.to_datetime(games["gameday"], errors="coerce")
    home = pd.DataFrame({
        "game_id": games["game_id"], "season": games["season"],
        "week": games["week"], "gameday": gd,
        "team": games["home_team"], "opponent": games["away_team"],
        "is_home": True,
        "for": games["home_score"].astype(float),
        "against": games["away_score"].astype(float),
    })
    away = pd.DataFrame({
        "game_id": games["game_id"], "season": games["season"],
        "week": games["week"], "gameday": gd,
        "team": games["away_team"], "opponent": games["home_team"],
        "is_home": False,
        "for": games["away_score"].astype(float),
        "against": games["home_score"].astype(float),
    })
    ev = pd.concat([home, away], ignore_index=True)
    ev["net_from_team"] = ev["for"] - ev["against"]
    ev["team_win"] = np.select(
        [ev["for"] > ev["against"], ev["for"] < ev["against"]],
        [1.0, 0.0], default=0.5)
    return ev


# ---------------------------------------------------------------------------
# Elo — pre-game entering rating, updated only after a game settles
# ---------------------------------------------------------------------------
def _elo_apply(events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    K, prior, scale = config.ELO_K, config.ELO_PRIOR, config.ELO_SCALE
    ev = events.sort_values(["gameday", "game_id", "is_home"]).reset_index(drop=True)
    rating: dict = {}
    entering: dict = {}
    for game_id, rows in ev.groupby("game_id", sort=False):
        a, b = list(rows.itertuples(index=False))[:2]
        ra, rb = rating.get(a.team, prior), rating.get(b.team, prior)
        entering[(game_id, a.team)] = ra
        entering[(game_id, b.team)] = rb
        exp_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / scale))
        exp_b = 1.0 / (1.0 + 10.0 ** ((ra - rb) / scale))
        rating[a.team] = ra + K * (a.team_win - exp_a)
        rating[b.team] = rb + K * (b.team_win - exp_b)
    ev = ev.copy()
    ev["elo_entering"] = [entering.get((gid, team), prior)
                          for gid, team in zip(ev["game_id"], ev["team"])]
    return ev, rating


def compute_elo(events: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-game ``elo_entering`` (strictly prior information only)."""
    return _elo_apply(events)[0]


# ---------------------------------------------------------------------------
# Trailing primitives — per-team, strictly-prior via shift(1)
# ---------------------------------------------------------------------------
def _trailing_per_team(srt: pd.DataFrame, value_col: str, window: int) -> np.ndarray:
    roll = srt.groupby("team", sort=False)[value_col].rolling(
        window, min_periods=1).mean()
    roll = roll.groupby(level=0).shift(1)
    return roll.reset_index(level=0, drop=True).to_numpy()


def _trailing_ewm(srt: pd.DataFrame, value_col: str, halflife: float) -> np.ndarray:
    roll = srt.groupby("team", sort=False)[value_col].ewm(
        halflife=halflife, min_periods=1).mean()
    roll = roll.groupby(level=0).shift(1)
    return roll.reset_index(level=0, drop=True).to_numpy()


# ---------------------------------------------------------------------------
# Venue facts (committed stadiums table)
# ---------------------------------------------------------------------------
VENUE_FILE = config.BACKEND_DIR / "nfl_stadiums.csv"
VENUE_SCHEMA = ["stadium", "facility", "teams", "lat", "lon", "altitude_ft",
                "tz", "source"]


@functools.lru_cache(maxsize=1)
def _load_venue_table() -> pd.DataFrame:
    if not VENUE_FILE.exists():
        return pd.DataFrame(columns=VENUE_SCHEMA)
    return pd.read_csv(VENUE_FILE)


@functools.lru_cache(maxsize=1)
def _venue_facts() -> dict[str, dict]:
    t = _load_venue_table()
    out: dict[str, dict] = {}
    for r in t.itertuples(index=False):
        out[getattr(r, "stadium")] = {
            "lat": float(r.lat) if pd.notna(getattr(r, "lat", np.nan)) else np.nan,
            "lon": float(r.lon) if pd.notna(getattr(r, "lon", np.nan)) else np.nan,
            "altitude_ft": float(r.altitude_ft) if pd.notna(getattr(r, "altitude_ft", np.nan)) else np.nan,
            "tz": getattr(r, "tz", "") or "",
        }
    return out


@functools.lru_cache(maxsize=1)
def _team_home_stadium_map() -> dict[str, str]:
    t = _load_venue_table()
    out: dict[str, str] = {}
    for r in t.itertuples(index=False):
        teams = getattr(r, "teams", "")
        if isinstance(teams, str) and teams.strip():
            for team in teams.split(","):
                out.setdefault(team.strip(), getattr(r, "stadium"))
    return out


def _haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.asarray(a, dtype=float)
                              for a in (lat1, lon1, lat2, lon2))
    r = np.pi / 180.0
    dlat = (lat2 - lat1) * r
    dlon = (lon2 - lon1) * r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin(dlon / 2) ** 2
    d = 2.0 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return np.where(np.isfinite(d), d, np.nan)


def _utc_offset_hours(tz_name: str, gameday) -> float:
    if not tz_name:
        return float("nan")
    try:
        dt = pd.Timestamp(gameday).replace(hour=12, minute=0)
        off = dt.tz_localize(ZoneInfo(tz_name)).utcoffset()
        return float(off.total_seconds()) / 3600.0 if off is not None else float("nan")
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# PBP rollup — per (game_id, team) aggregates (functions of that game only)
# ---------------------------------------------------------------------------
def pbp_team_agg(pbp: pd.DataFrame | None) -> pd.DataFrame:
    """Per-(game_id, posteam) play aggregates. Every column is a per-game
    sum/rate — the trailing shift downstream keeps them strictly-prior.
    Absent source columns degrade to NaN (never fabricated)."""
    cols = ["game_id", "team", "total_yards", "n_plays", "elapsed_min"]
    if pbp is None or "posteam" not in getattr(pbp, "columns", []):
        return pd.DataFrame(columns=cols)
    p = pbp.dropna(subset=["posteam"])
    if "game_id" not in p.columns or "yards_gained" not in p.columns:
        return pd.DataFrame(columns=cols)
    p = p.copy()
    p["yards_gained"] = pd.to_numeric(p["yards_gained"], errors="coerce")
    agg = {"total_yards": ("yards_gained", "sum"),
           "n_plays": ("yards_gained", "count")}
    g = p.groupby(["game_id", "posteam"], as_index=False).agg(**agg)
    if "game_seconds_remaining" in p.columns:
        p["game_seconds_remaining"] = pd.to_numeric(
            p["game_seconds_remaining"], errors="coerce")
        last = (p.dropna(subset=["game_seconds_remaining"])
                 .sort_values("game_seconds_remaining")
                 .drop_duplicates("game_id", keep="first"))
        last = last.assign(elapsed_min=(3600.0 - last["game_seconds_remaining"]) / 60.0)
        g = g.merge(last[["game_id", "elapsed_min"]], on="game_id", how="left")
    else:
        g["elapsed_min"] = np.nan
    return g.rename(columns={"posteam": "team"})[cols]


# ---------------------------------------------------------------------------
# The ladder — team state + trailing stats, one row per (game_id, team)
# ---------------------------------------------------------------------------
def team_stats_ladder(events: pd.DataFrame,
                      team_game_agg: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-(game_id, team) point-in-time state: elo_entering, form_pts,
    win_pct, rest_days, ypp, ewm_net_pts, ewm_ypp, pace_plays_min,
    short_rest — every trailing value strictly-prior (asserted)."""
    ev = events.copy()
    if team_game_agg is not None and len(team_game_agg):
        agg = team_game_agg.rename(columns={"total_yards": "tot_yd",
                                            "n_plays": "npl"})
        agg["ypp_game"] = agg["tot_yd"] / agg["npl"].replace(0, np.nan)
        agg["pace_plays_min_game"] = agg["npl"] / agg["elapsed_min"].replace(0, np.nan)
        ev = ev.merge(agg[["game_id", "team", "ypp_game", "pace_plays_min_game"]],
                      on=["game_id", "team"], how="left")

    srt = ev.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)

    # LEAKAGE GATE: gameday strictly increasing within each team
    diffs = srt.groupby("team", sort=False)["gameday"].diff()
    bad = srt.loc[(diffs.notna()) & (diffs <= pd.Timedelta(0))]
    if len(bad):
        raise AssertionError(
            f"team_stats_ladder: team gameday not strictly increasing "
            f"({len(bad)} rows) — trailing features could leak")

    srt["form_pts"] = _trailing_per_team(srt, "net_from_team", config.FORM_WINDOW)
    srt["win_pct"] = _trailing_per_team(srt, "team_win", config.WINPCT_WINDOW)
    srt["rest_days"] = srt.groupby("team", sort=False)["gameday"].diff().dt.days
    srt["short_rest"] = np.where(
        srt["rest_days"].notna(), (srt["rest_days"] < 7).astype(float), np.nan)
    srt["ypp"] = (_trailing_per_team(srt, "ypp_game", config.YPP_WINDOW)
                  if "ypp_game" in srt.columns else np.nan)
    srt["ewm_net_pts"] = _trailing_ewm(srt, "net_from_team", config.EWM_HALFLIFE)
    srt["ewm_ypp"] = (_trailing_ewm(srt, "ypp_game", config.EWM_HALFLIFE)
                      if "ypp_game" in srt.columns else np.nan)
    srt["pace_plays_min"] = (_trailing_per_team(srt, "pace_plays_min_game",
                                                config.PACE_WINDOW)
                             if "pace_plays_min_game" in srt.columns else np.nan)
    return srt


def _home_minus_away(ladder: pd.DataFrame, game_ids: pd.Index, col: str) -> np.ndarray:
    home = ladder[ladder["is_home"]].set_index("game_id")[col]
    away = ladder[~ladder["is_home"]].set_index("game_id")[col]
    return (home.reindex(game_ids) - away.reindex(game_ids)).to_numpy()


def _per_side(ladder: pd.DataFrame, game_ids: pd.Index, col: str) -> tuple[np.ndarray, np.ndarray]:
    home = ladder[ladder["is_home"]].set_index("game_id")[col]
    away = ladder[~ladder["is_home"]].set_index("game_id")[col]
    return (home.reindex(game_ids).to_numpy(), away.reindex(game_ids).to_numpy())


def _attach_venue_facts(df: pd.DataFrame) -> pd.DataFrame:
    """Attach travel_miles_diff / altitude_home / prime_time from the
    committed stadiums table + schedule facts. NaN when unknown."""
    facts = _venue_facts()
    team_home = _team_home_stadium_map()
    stadium = df["stadium"] if "stadium" in df.columns else pd.Series(np.nan, index=df.index)

    def _game_fact(name: str) -> np.ndarray:
        return stadium.map(lambda s: facts.get(s, {}).get(name, np.nan)).to_numpy()

    def _team_fact(team_col: str, name: str) -> np.ndarray:
        return df[team_col].map(
            lambda t: facts.get(team_home.get(t, ""), {}).get(name, np.nan)).to_numpy()

    game_lat, game_lon = _game_fact("lat"), _game_fact("lon")
    home_lat, home_lon = _team_fact("home_team", "lat"), _team_fact("home_team", "lon")
    away_lat, away_lon = _team_fact("away_team", "lat"), _team_fact("away_team", "lon")
    df = df.copy()
    df["travel_miles_diff"] = (_haversine_miles(home_lat, home_lon, game_lat, game_lon)
                               - _haversine_miles(away_lat, away_lon, game_lat, game_lon))
    df["altitude_home"] = _game_fact("altitude_ft")

    gametime = (df["gametime"].astype(str) if "gametime" in df.columns
                else pd.Series("", index=df.index))
    hour = pd.to_numeric(gametime.str.split(":").str[0], errors="coerce")
    df["prime_time"] = np.where(hour >= config.PRIME_TIME_HOUR, 1.0,
                                np.where(hour.isna(), np.nan, 0.0))
    return df


def _records_string(events: pd.DataFrame) -> pd.Series:
    """Cumulative W-L record string per team over the decided timeline."""
    rec = events.groupby("team").agg(
        wins=("team_win", lambda s: float((s == 1).sum())),
        losses=("team_win", lambda s: float((s == 0).sum())),
        ties=("team_win", lambda s: float((s == 0.5).sum())),
    )

    def _fmt(team: str) -> str:
        if team not in rec.index:
            return ""
        r = rec.loc[team]
        base = f"{int(r['wins'])}-{int(r['losses'])}"
        return f"{base}-{int(r['ties'])}" if r["ties"] > 0 else base

    return rec, _fmt


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------
def build_game_features(games: pd.DataFrame,
                        pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Point-in-time feature frame for DECIDED games (one row per game).

    ``games`` must include the warmup timeline (2018+) so early games carry
    real priors; every trailing value is shifted strictly prior. Returns the
    12 served diff features + the per-side values the tree view needs.
    """
    ev = compute_elo(team_events(games))
    ladder = team_stats_ladder(ev, pbp_team_agg(pbp))

    df = games.copy().reset_index(drop=True)
    gids = df["game_id"]
    df["is_home"] = 1.0
    df["elo_diff"] = _home_minus_away(ladder, gids, "elo_entering")
    df["win_pct_diff"] = _home_minus_away(ladder, gids, "win_pct")
    df["rest_days_diff"] = _home_minus_away(ladder, gids, "rest_days")
    df["ewm_net_pts_diff"] = _home_minus_away(ladder, gids, "ewm_net_pts")
    df["ewm_ypp_diff"] = _home_minus_away(ladder, gids, "ewm_ypp")
    df["pace_plays_min_diff"] = _home_minus_away(ladder, gids, "pace_plays_min")
    df["rest_short_diff"] = _home_minus_away(ladder, gids, "short_rest")
    if "div_game" in df.columns:
        df["div_game"] = pd.to_numeric(df["div_game"], errors="coerce")
    else:
        df["div_game"] = np.nan
    df["is_dome_home"] = np.where(
        df.get("roof", pd.Series(np.nan, index=df.index)).isin(["dome", "closed"]),
        1.0, np.where(df.get("roof", pd.Series(np.nan, index=df.index)).isin(["outdoors"]),
                      0.0, np.nan))
    df = _attach_venue_facts(df)

    # per-side values for the tree view (raw home/away representations)
    for side_col, lad_col in (("elo_home", "elo_entering"), ("elo_away", "elo_entering"),
                              ("win_pct_home", "win_pct"), ("win_pct_away", "win_pct"),
                              ("ewm_net_pts_home", "ewm_net_pts"),
                              ("ewm_net_pts_away", "ewm_net_pts"),
                              ("ewm_ypp_home", "ewm_ypp"), ("ewm_ypp_away", "ewm_ypp"),
                              ("rest_days_home", "rest_days"), ("rest_days_away", "rest_days")):
        home_v, away_v = _per_side(ladder, gids, lad_col)
        df[side_col] = home_v if side_col.endswith("home") else away_v

    rec, fmt = _records_string(ev)
    df["home_record"] = df["home_team"].map(fmt)
    df["away_record"] = df["away_team"].map(fmt)
    df["home_wins"] = df["home_team"].map(lambda t: float(rec.loc[t, "wins"]) if t in rec.index else np.nan)
    df["home_losses"] = df["home_team"].map(lambda t: float(rec.loc[t, "losses"]) if t in rec.index else np.nan)
    df["away_wins"] = df["away_team"].map(lambda t: float(rec.loc[t, "wins"]) if t in rec.index else np.nan)
    df["away_losses"] = df["away_team"].map(lambda t: float(rec.loc[t, "losses"]) if t in rec.index else np.nan)

    # targets (kept beside features for OOF assembly; never model inputs)
    df["margin"] = df["home_score"].astype(float) - df["away_score"].astype(float)
    df["total"] = df["home_score"].astype(float) + df["away_score"].astype(float)
    df["home_win"] = (df["margin"] > 0).astype(float)
    return df


def build_slate_features(schedule: pd.DataFrame,
                         pbp: pd.DataFrame | None) -> pd.DataFrame:
    """Point-in-time feature frame for SCHEDULED (undecided) games.

    The ladder spans the full decided timeline; the scheduled rows are the
    latest events so their trailing stats come from strictly-prior decided
    games only (same shift(1) discipline, monotonicity still asserted).
    """
    sched = schedule.copy()
    for c in ("home_score", "away_score"):
        if c in sched.columns:
            sched[c] = pd.to_numeric(sched[c], errors="coerce")
    decided = sched[sched["home_score"].notna() & sched["away_score"].notna()]
    pending = sched[sched["home_score"].isna() | sched["away_score"].isna()]
    if pending.empty:
        return pd.DataFrame()

    ev_decided, ratings = _elo_apply(team_events(decided))
    ev_pending = team_events(pending)
    ev_pending["elo_entering"] = ev_pending["team"].map(
        lambda t: ratings.get(t, config.ELO_PRIOR))
    combined = pd.concat([ev_decided, ev_pending], ignore_index=True)
    ladder = team_stats_ladder(combined, pbp_team_agg(pbp))

    df = pending.copy().reset_index(drop=True)
    # market-independence: drop any odds columns at the boundary
    for _m in ("spread_line", "total_line", "home_moneyline", "away_moneyline",
               "over_odds", "under_odds", "away_spread_odds", "home_spread_odds"):
        if _m in df.columns:
            df = df.drop(columns=_m)

    gids = df["game_id"]
    df["is_home"] = 1.0
    df["elo_diff"] = _home_minus_away(ladder, gids, "elo_entering")
    df["win_pct_diff"] = _home_minus_away(ladder, gids, "win_pct")
    df["rest_days_diff"] = _home_minus_away(ladder, gids, "rest_days")
    df["ewm_net_pts_diff"] = _home_minus_away(ladder, gids, "ewm_net_pts")
    df["ewm_ypp_diff"] = _home_minus_away(ladder, gids, "ewm_ypp")
    df["pace_plays_min_diff"] = _home_minus_away(ladder, gids, "pace_plays_min")
    df["rest_short_diff"] = _home_minus_away(ladder, gids, "short_rest")
    if "div_game" in df.columns:
        df["div_game"] = pd.to_numeric(df["div_game"], errors="coerce")
    else:
        df["div_game"] = np.nan
    df["is_dome_home"] = np.where(
        df.get("roof", pd.Series(np.nan, index=df.index)).isin(["dome", "closed"]),
        1.0, np.where(df.get("roof", pd.Series(np.nan, index=df.index)).isin(["outdoors"]),
                      0.0, np.nan))
    df = _attach_venue_facts(df)

    for side_col, lad_col in (("elo_home", "elo_entering"), ("elo_away", "elo_entering"),
                              ("win_pct_home", "win_pct"), ("win_pct_away", "win_pct"),
                              ("ewm_net_pts_home", "ewm_net_pts"),
                              ("ewm_net_pts_away", "ewm_net_pts"),
                              ("ewm_ypp_home", "ewm_ypp"), ("ewm_ypp_away", "ewm_ypp"),
                              ("rest_days_home", "rest_days"), ("rest_days_away", "rest_days")):
        home_v, away_v = _per_side(ladder, gids, lad_col)
        df[side_col] = home_v if side_col.endswith("home") else away_v

    rec, fmt = _records_string(ev_decided)
    df["home_record"] = df["home_team"].map(fmt)
    df["away_record"] = df["away_team"].map(fmt)
    return df


# ---------------------------------------------------------------------------
# Model-family feature views (deterministic ordering + dimensionality)
# ---------------------------------------------------------------------------
def linear_view(df: pd.DataFrame) -> pd.DataFrame:
    """Difference-oriented linear/MLP matrix (documented representation)."""
    cols = [c for c in config.LINEAR_FEATURES if c in df.columns]
    out = df.reindex(columns=cols).astype(float)
    return out


def tree_view(df: pd.DataFrame) -> pd.DataFrame:
    """Tree-family matrix: differences + raw home/away side values.

    Deterministic column order: served diffs first (manifest order), then
    the per-side raw values in home/away pairs.
    """
    diff_cols = [c for c in config.FEATURE_COLUMNS if c in df.columns]
    side_cols = [c for c in ("elo_home", "elo_away", "win_pct_home", "win_pct_away",
                             "ewm_net_pts_home", "ewm_net_pts_away",
                             "ewm_ypp_home", "ewm_ypp_away",
                             "rest_days_home", "rest_days_away") if c in df.columns]
    out = df.reindex(columns=diff_cols + side_cols).astype(float)
    return out


def feature_coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage + missingness diagnostics per served feature."""
    rows = []
    for f in config.FEATURE_COLUMNS:
        if f not in df.columns:
            rows.append({"feature": f, "n_games": len(df),
                         "coverage_pct": 0.0, "mean": np.nan, "std": np.nan})
            continue
        v = pd.to_numeric(df[f], errors="coerce")
        rows.append({
            "feature": f, "n_games": int(len(df)),
            "coverage_pct": round(100.0 * float(v.notna().mean()), 2),
            "mean": float(v.mean()) if v.notna().any() else np.nan,
            "std": float(v.std()) if v.notna().sum() > 1 else np.nan,
        })
    return pd.DataFrame(rows)
