"""NFL feature engineering — leakage-safe raw candidates + admission gate.

Builds on the committed game-level frame produced by ``nfl_game_frame.py``
(``nfl-backend/data_delivery/nfl_game_level_features.csv``). This module is
the feature-ADMISSION stage only: it proposes raw candidates, audits coverage
and (point-in-time) leakage, and gates them into an admitted set. It does NOT
train any model — the walk-forward ensemble lives in ``nfl_moneyline.py``.

v1 base (admitted 2026-08-28): elo_diff, form_diff_pts, rest_days_diff,
ypp_diff, is_dome_home (+ is_home anchor). v2 candidates (this file):
trailing per-team strength aggregates with SMALL DECAYING WINDOWS
(exponentially-weighted net-points margin, EPA/play, yards/play, scoring
output), opponent-adjusted variants (trailing margin minus the trailing form
of the opponents faced), pace (plays/min), rest-days edge (short-rest flag),
QB/offense-quality edge (decaying QB EPA/play), weather beyond the dome flag
(game temp / wind at the home venue), and the division-game flag. All are
leak-safe by construction and pass the SAME admission gate (coverage floor,
redundancy pruning, near-random-AUC pruning) with 2025 kept sealed.

Leakage discipline (MLB retrospective lessons — "raw-not-clever, pre-game
coverage rule, gated entry, no model-output-as-input"):
- Same rule, extended to v2: every trailing feature (windowed OR decaying-
  ewm) is a function ONLY of that team's games with ``gameday`` STRICTLY
  BEFORE the target game — the ewm/opponent-adjusted columns use the same
  per-team shift(1) as the v1 windowed columns, asserted by the same
  strict-monotonicity check in :func:`team_stats_ladder`.
- Every trailing feature is a function ONLY of that team's games with
  ``gameday`` STRICTLY BEFORE the target game. Enforced by chronological sort
  + per-team window shift + an explicit per-team strict-monotonicity assertion
  in :func:`team_stats_ladder`.
- No feature uses market lines, model probabilities, or later results. No
  hand-multiplied "risk" interactions, no injury reports (not reliably final
  12h pre-kickoff), no weather.

  v5 (Tier-3): the market de-vig, referee-crew, and roster age/exp
  candidates ARE composed by build_features/build_slate_features and,
  apart from market_home_implied — which was admitted by the 2026-09-01
  MARK verdict and then DELIBERATELY REVERSED BY POLICY so the model stays
  an independent fundamentals predictor (the market is compared, not
  consumed) — stay OUT of FEATURE_COLUMNS unless the sealed-2025 ablation
  admits them (Tier-1/Tier-2 rule).

12h-pre-kickoff availability assumption (stated): a feature counts as
"available 12h pre-kickoff" iff it is non-null and depends only on completed
prior games or a static venue/prior fact. Nothing here depends on live
intraday state, so availability == non-null coverage for every candidate.

Sources
-------
- Game-level frame: committed ``nfl_game_frame.py`` output (2019-2024 decided).
- Schedule (for ``roof`` -> ``is_dome_home``): nflreadpy ``load_schedules``.
- Play-by-play (for net yards/play): nflreadpy ``load_pbp``.
- ``WARMUP_SEASONS`` (2018) is pulled for the SAME sources purely so the first
  2019 games have clean trailing priors; the decided frame is 2019-2025.
  Season 2025 is the moneyline model's SEALED HOLD-OUT: it is scored for
  coverage here, but the admission gate's AUC stays on seasons < 2025 so the
  hold-out rows are never used for fitting or feature admission.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import functools
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nfl_feature_engine import TEAM_AGG_COLUMNS, TIER1_NEEDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent          # nfl-backend/
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"
DECIDED_FRAME = DATA_DELIVERY_DIR / "nfl_game_level_features.csv"

WARMUP_SEASONS = [2018]                          # trailing priors only
CORE_SEASONS = list(range(2019, 2026))           # 2019..2025 scored/reported
DEFAULT_SEASONS = WARMUP_SEASONS + CORE_SEASONS
# The SEALED HOLD-OUT for the moneyline model: 2025 is decided but is never
# used for feature admission/fitting in this module's gate (AUC stays on
# seasons < HOLD_SEASON so the hold-out row remains clean for nfl_moneyline).
HOLD_SEASON = 2025

# ELO (prior + update rule, fully specified / reproducible)
ELO_PRIOR = 1500.0
ELO_K = 32.0                                     # standard logistic gain
ELO_SCALE = 400.0

# trailing windows
FORM_WINDOW = 4       # net pts/game window
WINPCT_WINDOW = 12    # trailing win% window
YPP_WINDOW = 5        # net yards/play window
# v2 windows: small decaying (ewm) windows for strength aggregates
EWM_HALFLIFE = 2      # decaying-window halflife (games) — ewm net pts/EPA/scoring/ypp
OPP_ADJ_WINDOW = 6    # opponent-adjusted trailing-margin window (games)
PACE_WINDOW = 4       # trailing plays/min window (games)

# Tier-1 (v3) per-game PBP aggregates the ladder turns into strictly-prior
# decaying-window team features (ewm halflife=2, same as the v2 strength set).
TIER1_AGG_COLUMNS = [
    "giveaways", "takeaways", "net_any_a", "sack_rate", "success_rate",
    "explosive_rate", "penalty_yds", "penalty_yds_drawn",
    "third_down_rate", "redzone_td_rate", "pts_per_drive",
]

# admission gate
# GATE_AUTO_PRUNE — user policy (2026-09-01): the gate NEVER automatically
# removes features from the served pool. False (default) = REPORT-ONLY:
# coverage / redundancy / univariate-AUC are computed and recorded, and any
# registered feature below the coverage floor or above the correlation bar
# triggers a LOUD warning (informational, never blocking). True re-enables
# the LEGACY pruning (coverage floor, then redundant-pair / near-random-AUC
# pruning) — a deliberate opt-in, never the default.
GATE_AUTO_PRUNE = False
COVERAGE_FLOOR = 0.95
# The redundancy bar is the reporting bar the user specified ("|r| > 0.8"):
# a feature ~83% correlated with a slightly-stronger one is redundant enough to
# prune (measured elo_diff ~ win_pct_diff r = 0.826 -> keep elo, drop win_pct).
CORR_REDUNDANCY = 0.80
DISC_BAND = 0.05                                  # |auc - 0.5| below = ~random

DATE_FMT = "%Y%m%d"
RECORD_TEMPLATE = f"nfl_feature_v1_{{date}}.json"

# deterministic keep-order for redundant pairs (lower = kept first)
FEATURE_PRIORITY = {
    "elo_diff": 0, "ypp_diff": 1, "form_diff_pts": 2, "win_pct_diff": 3,
    "rest_days_diff": 4, "is_dome_home": 5,
    "ewm_net_pts_diff": 6, "ewm_epa_play_diff": 7,
    "ewm_scoring_diff": 8, "ewm_ypp_diff": 9, "opp_adj_net_pts_diff": 10,
    "pace_plays_min_diff": 11, "rest_short_diff": 12, "temp_f": 13,
    "wind_mph": 14, "div_game": 15,
    "travel_miles_diff": 16, "altitude_home": 17, "prime_time": 18,
}

# The SERVED pool, in gating order: what survives is EXACTLY this list
# (minus the ``is_home`` anchor at predict time). The gate (run_feature_gate)
# no longer prunes by default (GATE_AUTO_PRUNE=False), so this list IS what
# ``nfl_moneyline`` consumes.
FEATURE_COLUMNS = [
    # ---- v1 base (admitted 2026-08-28) --------------------------------
    "elo_diff", "win_pct_diff", "rest_days_diff", "is_dome_home",
    # ---- v2 decaying-window strength aggregates (admitted 2026-08-28;
    # ewm_qb_epa_play_diff removed 2026-09-01 by the corr-pair twin verdict
    # — see the composed-but-unregistered block below) --------------------
    "ewm_net_pts_diff", "ewm_ypp_diff",
    # ---- v2 schedule facts (admitted 2026-08-28) ----------------------
    "pace_plays_min_diff", "rest_short_diff", "div_game",
    # ---- v4 Tier-2 venue slices, admitted 2026-09-01 (b2205f2) --------
    # VENUE_3 won the sealed-2025 holdout (logloss 0.6382 vs 0.6401, AUC
    # 0.6933 vs 0.6913); the full 6-feature VENUE block DON'T ADOPT.
    "travel_miles_diff", "altitude_home", "prime_time",
    # ---- constant anchor (reported, never a model column) --------------
    "is_home",
]

# ---------------------------------------------------------------------------
# Composed-but-unregistered candidates — DELIBERATELY not in the served
# pool. TRUE HISTORY (no ablation invented for any of them):
#   temp_f / wind_mph          — 0.0% coverage in every nflreadpy pull (the
#       schedule's temp/wind columns are empty in this feed) — unservable.
#   form_diff_pts              — redundant twin of ewm_net_pts_diff
#       (|r| 0.94, similar discrimination); kept composed so the Tier-1/2/3
#       WITHOUT baselines and reuse stay intact.
#   ypp_diff                   — redundant twin of ewm_ypp_diff (|r| 0.94).
#   ewm_epa_play_diff          — redundant twin of ewm_qb_epa_play_diff
#       (|r| 0.99).
#   ewm_qb_epa_play_diff       — trailing QB EPA/play, ewm halflife=2.
#       ADMITTED with the v1/v2 base (2026-08-28), then REMOVED BY VERDICT:
#       the twin-removal ablation (run_feature_corr_ablation.py, record
#       nfl_feature_corr_ablation_e4aee120a4b8.json, commit cd3c26b) found
#       WITHOUT_QBEPA beats the 13-pool on SEALED 2025 logloss (−0.0124) AND
#       AUC (+0.0129) with ECE-cal improving 0.0937 → 0.0656 (under 0.08),
#       pooled OOF corroborating (−0.0116). Note: the market revert
#       (2f79669) changed which twin is droppable — with the market out,
#       yards-per-play (ewm_ypp_diff) carries the retained signal, so
#       QB-EPA/play does not. ewm_ypp_diff STAYS (the WITHOUT_YPP arm lost
#       sealed AUC −0.0019 — it is the keeper). Composition stays so the
#       harness and re-runs keep working.
#   ewm_scoring_diff           — redundant twin of ewm_epa_play_diff
#       (|r| 0.85).
#   opp_adj_net_pts_diff       — redundant twin of form_diff_pts (|r| 0.91).
#   market_home_implied        — no-vig closing-moneyline home win prob.
#       ADMITTED by the Tier-3 MARK verdict (76002fb: sealed 0.6339/0.7121/
#       ECE 0.0759 vs WITHOUT 0.6507/0.6817/0.0937; all five members improve
#       sealed both axes), then DELIBERATELY REVERSED BY POLICY — the model
#       is to remain an independent fundamentals predictor: the market line
#       is COMPARED (the moneyline gate's market_line reference arm), never
#       CONSUMED as a model input. Composition stays so the reference arm
#       and run_tier3_ablation.py keep working.
# None of these was ever removed by a SEALED-ABLATION verdict (market_home_
# implied was admitted by one and then reversed by policy; qb_epa was
#   removed by the corr-pair twin verdict above): the rest were pruned at
#   admission time by the LEGACY coverage/redundancy gate (now the opt-in
#   GATE_AUTO_PRUNE=True path), and they keep appearing in the ablation
#   WITHOUT baselines because build_features/build_slate_features still
#   compose them. Unregistering them here is the deliberate policy decision
#   that the served pool is exactly the 12 features above and that the gate
#   no longer removes anything from it automatically. (The v3 Tier-1
#   candidates, the v4 venue remainder — timezone_diff/turf_home/
#   neutral_site — and the v5
#   officials/roster families are also composed-but-unregistered, each with
#   a real ablation verdict recorded in its comment block.)
# ---------------------------------------------------------------------------

CANONICAL_SOURCE = {
    "elo_diff": "ELO prior 1500, K=32, strictly-prior games",
    "form_diff_pts": "trailing net pts/game (last 4)",
    "win_pct_diff": "trailing win% (last 12)",
    "rest_days_diff": "days since each team's prior game",
    "ypp_diff": "trailing net yards/play (last 5, from pbp)",
    "is_dome_home": "home venue roof (nflverse schedule field)",
    "ewm_net_pts_diff": "trailing net pts/game, ewm halflife=2 (decaying)",
    "ewm_epa_play_diff": "trailing EPA/play, ewm halflife=2 (from pbp epa)",
    "ewm_scoring_diff": "trailing points-for/game, ewm halflife=2 (scoring output)",
    "ewm_ypp_diff": "trailing yards/play, ewm halflife=2 (from pbp)",
    "opp_adj_net_pts_diff": "trailing net pts minus avg trailing form of opponents faced (last 6)",
    "pace_plays_min_diff": "trailing plays per minute (last 4, from pbp clock)",
    "rest_short_diff": "short-rest flag (home rest < 7 days) − away flag",
    "temp_f": "home-venue game temperature F (nflverse schedule field)",
    "wind_mph": "home-venue game wind mph (nflverse schedule field)",
    "div_game": "division game flag (nflverse schedule field)",
    "travel_miles_diff": "home−away stadium distance, haversine (nfl_stadiums.csv)",
    "altitude_home": "home venue elevation, meters SRTM (nfl_stadiums.csv)",
    "prime_time": "evening-kickoff flag, ET hour >= 17 (nflverse gametime)",
    "is_home": "constant anchor for the home edge",
}

# ---------------------------------------------------------------------------
# Tier-2 (v4) venue / travel / schedule candidates — STATIC facts only.
#
# Ablation verdict (run_tier2_ablation.py, frame 5aa6121b2849, 2026-09-01):
# VENUE_3 (travel_miles_diff, altitude_home, prime_time) ADOPTED into the
# deployed pool (sealed logloss 0.6382 vs 0.6401, AUC 0.6933 vs 0.6913,
# ECE-cal 0.0571 vs 0.0776); the full 6-feature VENUE block DON'T ADOPT
# (sealed logloss 0.6438, AUC tie). travel_miles_diff/altitude_home/
# prime_time are registered below; timezone_diff/turf_home/neutral_site
# remain composed but unregistered (Tier-1 pattern).
#
# Source data:
#   - ``nfl_stadiums.csv`` (committed): real coordinates (Wikipedia geodata,
#     Nominatim fallback for coordinate-less articles), SRTM elevation via the
#     Open-Elevation API, and the IANA timezone of the venue's metro. Keyed on
#     the exact nflverse games.csv ``stadium`` strings (verified 2026-08-31:
#     45 distinct names across 2018-2025; nflreadpy 0.1.5 ships NO load_stadiums
#     and nflverse publishes no stadiums asset, so the table is curated once,
#     committed, and keyed on the real schedule column).
#   - per-game ``surface`` / ``gametime`` / ``location`` fields from the same
#     nflverse schedule rows the existing roof/temp/wind merge uses.
# Unknown stadiums / missing source fields resolve to NaN — never fabricated.
# ---------------------------------------------------------------------------
VENUE_FILE = BACKEND_DIR / "nfl_stadiums.csv"

# Tier-2 candidates (order mirrors the run_tier2_ablation.py arm lists)
VENUE_FEATURES = [
    "travel_miles_diff", "timezone_diff", "altitude_home",
    "turf_home", "prime_time", "neutral_site",
]

# "evening kickoff" threshold: nflverse gametime is ET; hours >= this are
# national-window evening games (SNF/MNF/TNF/international night games).
PRIME_TIME_HOUR = 17

# Real nflverse ``surface`` values observed 2018-2025: grass / grass(space) /
# '' / fieldturf / a_turf / sportturf / matrixturf / astroturf. Anything named
# grass -> 0, anything synthetic (turf/astro) -> 1, empty/unknown -> NaN.
_TURF_MARKERS = ("turf", "astro")
_GRASS_MARKERS = ("grass",)

EARTH_RADIUS_MILES = 3958.8


# ---------------------------------------------------------------------------
# Core primitives (pure; testable without network)
# ---------------------------------------------------------------------------
def _haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in miles (earth radius 3958.8 mi). Vectorized;
    any NaN coordinate -> NaN distance (an unknown venue never fabricates)."""
    lat1, lon1, lat2, lon2 = (np.asarray(a, dtype=float)
                              for a in (lat1, lon1, lat2, lon2))
    r = np.pi / 180.0
    dlat = (lat2 - lat1) * r
    dlon = (lon2 - lon1) * r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin(dlon / 2.0) ** 2
    d = 2.0 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return np.where(np.isfinite(d), d, np.nan)


@functools.lru_cache(maxsize=1)
def _load_venue_table() -> pd.DataFrame:
    """The committed, curated stadium table (real coords / elevation / tz),
    keyed on the exact nflverse games.csv ``stadium`` strings. Read-only;
    missing file -> empty frame (candidates degrade to NaN, never crash)."""
    if not VENUE_FILE.exists():
        return pd.DataFrame(columns=["stadium", "facility", "teams",
                                     "lat", "lon", "altitude_ft", "tz", "source"])
    return pd.read_csv(VENUE_FILE)


def _venue_facts() -> dict[str, dict]:
    """stadium name -> {lat, lon, altitude_ft, tz} for every real name."""
    t = _load_venue_table()
    out = {}
    for r in t.itertuples(index=False):
        lat = getattr(r, "lat", None)
        out[getattr(r, "stadium")] = {
            "lat": lat if pd.notna(lat) else np.nan,
            "lon": getattr(r, "lon", np.nan) if pd.notna(getattr(r, "lon", np.nan)) else np.nan,
            "altitude_ft": getattr(r, "altitude_ft", np.nan)
            if pd.notna(getattr(r, "altitude_ft", np.nan)) else np.nan,
            "tz": getattr(r, "tz", "") if pd.notna(getattr(r, "tz", "")) else "",
        }
    return out


@functools.lru_cache(maxsize=1)
def _team_home_stadium_map() -> dict[str, str]:
    """team abbr -> canonical home stadium name, from the real schedule modal
    (see the ``teams`` column of nfl_stadiums.csv). Relocation-era games (e.g.
    LAC 2018-19 at StubHub/Coliseum, LV 2018-19 at Oakland) use the canonical
    venue as a documented approximation."""
    t = _load_venue_table()
    out = {}
    for r in t.itertuples(index=False):
        teams = getattr(r, "teams", "")
        if isinstance(teams, str) and teams.strip():
            for team in teams.split(","):
                out.setdefault(team.strip(), getattr(r, "stadium"))
    return out


def _utc_offset_hours(tz_name: str, gameday) -> float:
    """The venue timezone's UTC offset in hours on the game's local date
    (DST-aware). Unknown tz / bad date -> NaN."""
    if not tz_name:
        return float("nan")
    try:
        gd = pd.Timestamp(gameday)
        dt = gd.replace(hour=12, minute=0)      # midday local avoids DST edges
        off = dt.tz_localize(ZoneInfo(tz_name)).utcoffset()
        return float(off.total_seconds()) / 3600.0 if off is not None else float("nan")
    except Exception:
        return float("nan")


def _compose_venue_candidates(df: pd.DataFrame,
                             schedule: pd.DataFrame | None) -> pd.DataFrame:
    """Attach the six Tier-2 venue/travel candidates to a game frame.

    Static pre-game facts only (leak-safe by construction): per-game venue
    (stadium/surface/gametime/location) merged from the nflverse schedule the
    same way roof/temp/wind/div_game are merged above; travel/timezone/altitude
    resolved through ``nfl_stadiums.csv`` keyed on the real stadium name.
    Missing source fields or unknown stadiums -> NaN, never fabricated.
    """
    sched = schedule if schedule is not None else pd.DataFrame()
    for col in ("surface", "gametime", "stadium", "location"):
        if col not in df.columns and col in sched.columns:
            sub = sched[["game_id", col]].drop_duplicates("game_id")
            df = df.merge(sub, on="game_id", how="left")
        if col not in df.columns:
            df[col] = np.nan

    facts = _venue_facts()
    team_home = _team_home_stadium_map()

    def _game_fact(name: str) -> np.ndarray:
        return df["stadium"].map(lambda s: facts.get(s, {}).get(name, np.nan)).to_numpy()

    def _team_fact(team_col: str, name: str) -> np.ndarray:
        return df[team_col].map(lambda t: facts.get(team_home.get(t, ""), {}).get(name, np.nan)).to_numpy()

    game_lat, game_lon = _game_fact("lat"), _game_fact("lon")
    home_lat = _team_fact("home_team", "lat")
    home_lon = _team_fact("home_team", "lon")
    away_lat = _team_fact("away_team", "lat")
    away_lon = _team_fact("away_team", "lon")

    gd = pd.to_datetime(df["gameday"], errors="coerce")
    game_tz = df["stadium"].map(lambda s: facts.get(s, {}).get("tz", "")).to_numpy()
    home_tz = df["home_team"].map(lambda t: facts.get(team_home.get(t, ""), {}).get("tz", "")).to_numpy()
    away_tz = df["away_team"].map(lambda t: facts.get(team_home.get(t, ""), {}).get("tz", "")).to_numpy()

    crossed = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        day = gd.iloc[i]
        crossed[i] = (abs(_utc_offset_hours(game_tz[i], day) - _utc_offset_hours(home_tz[i], day))
                      if game_tz[i] and home_tz[i] else np.nan)
    crossed_away = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        day = gd.iloc[i]
        crossed_away[i] = (abs(_utc_offset_hours(game_tz[i], day) - _utc_offset_hours(away_tz[i], day))
                           if game_tz[i] and away_tz[i] else np.nan)

    df["travel_miles_diff"] = (
        _haversine_miles(home_lat, home_lon, game_lat, game_lon)
        - _haversine_miles(away_lat, away_lon, game_lat, game_lon))
    df["timezone_diff"] = crossed - crossed_away
    df["altitude_home"] = _game_fact("altitude_ft")

    def _as_str(col: pd.Series) -> pd.Series:
        # explicit NaN -> "" coercion (the .str accessor rejects astype(str)
        # on None-carrying columns under pandas 3.x)
        return pd.Series(["" if pd.isna(v) else str(v) for v in col],
                         index=col.index)

    surf = _as_str(df["surface"]).str.strip().str.lower()
    grass = surf.str.contains("|".join(_GRASS_MARKERS), regex=True)
    turf = surf.str.contains("|".join(_TURF_MARKERS), regex=True)
    df["turf_home"] = pd.to_numeric(
        np.where(grass, 0.0, np.where(turf, 1.0, np.nan)), errors="coerce")

    hour = _as_str(df["gametime"]).str.split(":").str[0]
    hour_num = pd.to_numeric(hour, errors="coerce")
    df["prime_time"] = np.where(hour_num >= PRIME_TIME_HOUR, 1.0,
                                 np.where(hour_num.isna(), np.nan, 0.0))

    loc = _as_str(df["location"]).str.strip()
    df["neutral_site"] = np.where(loc == "Neutral", 1.0,
                                   np.where(loc == "Home", 0.0, np.nan))
    return df


# ---------------------------------------------------------------------------
# Tier-3 (v5) candidates — market de-vig / officials / roster age+experience.
#
# Composed by build_features/build_slate_features but deliberately NOT
# registered in FEATURE_COLUMNS / CANONICAL_SOURCE / FEATURE_PRIORITY: the
# deployed pool changes only when a sealed-2025 ablation admits a feature
# (Tier-1/Tier-2 rule), so run_tier3_ablation.py is the only admission path.
#
#   market_home_implied — no-vig home win prob from the closing home/away
#       moneyline (100% coverage on decided seasons 2018-2025, same schedule
#       frame). Calendar-gated on the slate: lines post ~2 weeks out, so
#       far-future scheduled rows are NaN (default-filled) — the board
#       sharpens near-term and degrades gracefully far-term.
#   ref_pen_tend        — EWM (halflife=2) of penalty yards called AGAINST a
#       team in games worked by the assigned head referee crew, strictly
#       prior for that (team, crew) pair; home − away. Unknown referee or no
#       prior meeting -> NaN (never fabricated).
#   ref_pace            — crew game-pace proxy: EWM of total plays/game in
#       games the assigned crew worked, strictly prior league-wide.
#   roster_age_diff / roster_exp_diff — team-level mean age / years-exp from
#       the committed nfl_roster_age_exp.csv snapshot (verified live
#       2026-09-01: load_rosters schema, weekly snapshots, and a 2026
#       pre-season week-1 REG snapshot for all 32 teams). Pre-season-known
#       for every game of the season, so horizon-safe on the slate. (team,
#       season) pairs absent from the table (2018/2019 partial releases)
#       fall back to that team's nearest available season (documented).
# ---------------------------------------------------------------------------
TIER3_MARK_FEATURES = ["market_home_implied"]
TIER3_OFF_FEATURES = ["ref_pen_tend", "ref_pace"]
TIER3_ROSTER_FEATURES = ["roster_age_diff", "roster_exp_diff"]

# Ablation verdict (run_tier3_ablation.py, frame e4aee120a4b8, 2026-09-01):
#   MARK   ADOPT       (sealed 0.6339/0.7121/ECE 0.0759 vs WITHOUT
#                       0.6507/0.6817/0.0937; pooled 0.6026 corroborates; all
#                       five members improve sealed both axes) — admitted
#                       into FEATURE_COLUMNS (76002fb), then DELIBERATELY
#                       REVERSED BY POLICY 2026-09-01: the model must stay
#                       market-independent; market_home_implied is composed
#                       only, consumed by the gate's market_line reference
#                       arm (external benchmark), never as a model input.
#   OFF    DON'T ADOPT (sealed 0.6578/0.6815 misses on both axes) AND
#                       ref_pen_tend decided coverage is 84.6% — below the
#                       95% floor (team x crew meetings are ~1/yr), so it
#                       would fail admission regardless. Composed only.
#   ROSTER DON'T ADOPT (sealed 0.6523/0.6763 misses on both axes).
#   ALL    DON'T ADOPT (sealed 0.6590/0.7046 wins neither axis; ECE 0.1134
#                       degraded; mlp pooled collapses to near-random) —
#                       the small-slice lesson again.
# Note: this run's WITHOUT baseline (0.6507/0.6817) equals the Tier-2 local
# pull's numbers (frames e4aee120a4b8 / 49d58bfac1fb), a different nflreadpy
# cache than the 0.6401/0.6913 of the 5aa6121b2849 record — cross-pull
# absolute values drift; the verdict compares arms within the same pull.

ROSTER_FILE = BACKEND_DIR / "nfl_roster_age_exp.csv"


@functools.lru_cache(maxsize=1)
def _roster_table() -> tuple[dict, dict]:
    """{(team, season): (mean_age, mean_exp)} + per-team sorted seasons.
    Reads the committed snapshot CSV (produced by _curate_roster_age_exp.py)."""
    tab = pd.read_csv(ROSTER_FILE).dropna(subset=["mean_age", "mean_exp"])
    facts: dict = {}
    by_team: dict[str, list[int]] = {}
    for _, r in tab.iterrows():
        t, s = str(r["team"]), int(r["season"])
        facts[(t, s)] = (float(r["mean_age"]), float(r["mean_exp"]))
        by_team.setdefault(t, []).append(s)
    for t in by_team:
        by_team[t].sort()
    return facts, by_team


def _roster_fact(facts: dict, by_team: dict, team: str, season: int,
                 what: str) -> float:
    """(team, season) mean, falling back to the team's nearest available
    season (prefer the closest prior season; else the earliest). Unknown
    team -> NaN."""
    val = facts.get((team, season))
    if val is None:
        seasons = by_team.get(team)
        if not seasons:
            return float("nan")
        prior = [s for s in seasons if s <= season]
        pick = prior[-1] if prior else seasons[0]
        val = facts[(team, pick)]
    return val[0] if what == "age" else val[1]


def _american_implied(odds: pd.Series) -> pd.Series:
    """American odds -> implied win probability (0-1); NaN -> NaN."""
    odds = pd.to_numeric(odds, errors="coerce").astype(float)
    pos = odds > 0
    prob = np.where(pos, 100.0 / (100.0 + odds), -odds / (100.0 - odds))
    return pd.Series(np.where(odds.isna(), np.nan, prob),
                     index=odds.index, dtype=float)


def _compose_market_candidates(df: pd.DataFrame,
                               schedule: pd.DataFrame | None) -> pd.DataFrame:
    """market_home_implied — no-vig home win prob from the closing moneylines.
    Missing side or degenerate total -> NaN (never fabricated)."""
    for col in ("home_moneyline", "away_moneyline"):
        if col not in df.columns and schedule is not None and col in schedule.columns:
            sub = schedule[["game_id", col]].drop_duplicates("game_id")
            df = df.merge(sub, on="game_id", how="left")
        if col not in df.columns:
            df[col] = np.nan
    ph = _american_implied(df["home_moneyline"])
    pa = _american_implied(df["away_moneyline"])
    total = ph + pa
    df["market_home_implied"] = (ph / total).where(total > 0)
    return df


def _compose_officials_candidates(df: pd.DataFrame,
                                  schedule: pd.DataFrame | None,
                                  team_agg: pd.DataFrame | None) -> pd.DataFrame:
    """ref_pen_tend / ref_pace — strictly-prior crew-conditioned facts.

    ref_pen_tend: for each team, the EWM (halflife=2) of penalty yards called
    against that team in games the assigned head referee crew worked, using
    only those (team, crew) games STRICTLY prior (per-group shift). Unknown
    referee, no prior (team, crew) meeting, or either side missing -> NaN.
    For scheduled slate rows with a known crew, the value is the crew's most
    recent strictly-prior (team, crew) EWM.

    ref_pace: crew game-pace proxy — EWM of total plays/game over games the
    crew worked, strictly prior league-wide (a level feature, not a diff).
    """
    if "referee" not in df.columns:
        if schedule is not None and "referee" in schedule.columns:
            sub = schedule[["game_id", "referee"]].drop_duplicates("game_id")
            df = df.merge(sub, on="game_id", how="left")
        else:
            df["referee"] = np.nan
    df["ref_pen_tend"] = np.nan
    df["ref_pace"] = np.nan
    if team_agg is None or "n_plays" not in team_agg.columns:
        return df

    if schedule is not None and {"game_id", "referee", "gameday"}.issubset(
            schedule.columns):
        ref_join = schedule[["game_id", "referee", "gameday"]].drop_duplicates("game_id")
    else:
        ref_join = df[["game_id", "referee", "gameday"]].drop_duplicates("game_id")
    ref_join = ref_join.dropna(subset=["referee"])
    if ref_join.empty:
        return df

    agg = team_agg[["game_id", "team", "n_plays"]].copy()
    agg["pen_against"] = (pd.to_numeric(team_agg["penalty_yds"], errors="coerce")
                           if "penalty_yds" in team_agg.columns else np.nan)

    long = agg.merge(ref_join, on="game_id", how="left").dropna(subset=["referee"])
    if long.empty:
        return df
    long["gameday"] = pd.to_datetime(long["gameday"], errors="coerce")
    long = long.sort_values(["team", "referee", "gameday", "game_id"])
    prior = long.groupby(["team", "referee"], sort=False)["pen_against"].transform(
        lambda s: s.ewm(halflife=EWM_HALFLIFE, adjust=False).mean().shift(1))
    long["_prior"] = prior
    lut = long.set_index(["game_id", "team"])["_prior"]
    by_pair = (long.sort_values("gameday")
               .drop_duplicates(["team", "referee"], keep="last")
               .set_index(["team", "referee"])["_prior"])

    home = lut.reindex(pd.MultiIndex.from_arrays([df["game_id"], df["home_team"]]))
    home_pair = by_pair.reindex(
        pd.MultiIndex.from_arrays([df["home_team"], df["referee"]]))
    away = lut.reindex(pd.MultiIndex.from_arrays([df["game_id"], df["away_team"]]))
    away_pair = by_pair.reindex(
        pd.MultiIndex.from_arrays([df["away_team"], df["referee"]]))
    # by_game is the strictly-prior (team, crew) EWM for DECIDED games; the
    # (team, crew) by_pair fallback applies only to undecided slate rows (a
    # decided game with no (team, crew) history must stay NaN — a first
    # encounter is exactly where we have no information).
    decided = df["game_id"].isin(set(long["game_id"])).to_numpy()
    home_v = np.where(home.notna().to_numpy(), home.to_numpy(),
                      np.where(decided, np.nan, home_pair.to_numpy()))
    away_v = np.where(away.notna().to_numpy(), away.to_numpy(),
                      np.where(decided, np.nan, away_pair.to_numpy()))
    df["ref_pen_tend"] = home_v - away_v

    plays = long.groupby("game_id")["n_plays"].sum().rename("plays").reset_index()
    pace = plays.merge(ref_join[["game_id", "referee", "gameday"]], on="game_id",
                       how="left").dropna(subset=["referee"])
    pace["gameday"] = pd.to_datetime(pace["gameday"], errors="coerce")
    pace = pace.sort_values(["referee", "gameday", "game_id"])
    pv = pace.groupby("referee", sort=False)["plays"].transform(
        lambda s: s.ewm(halflife=EWM_HALFLIFE, adjust=False).mean().shift(1))
    pace["_v"] = pv
    pace_lut = pace.drop_duplicates("game_id").set_index("game_id")["_v"]
    by_ref = (pace.sort_values("gameday")
              .drop_duplicates("referee", keep="last")
              .set_index("referee")["_v"])
    pace_v = df["game_id"].map(pace_lut)
    df["ref_pace"] = np.where(
        pace_v.notna().to_numpy(), pace_v.to_numpy(),
        df["referee"].map(by_ref).to_numpy())
    return df


def _compose_roster_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """roster_age_diff / roster_exp_diff — pre-season-known team means.
    A team absent from the snapshot CSV degrades to NaN; an absent (team,
    season) pair falls back to the team's nearest available season."""
    facts, by_team = _roster_table()
    age_home = np.array([_roster_fact(facts, by_team, str(t), int(s), "age")
                         for t, s in zip(df["home_team"], df["season"])],
                        dtype=float)
    age_away = np.array([_roster_fact(facts, by_team, str(t), int(s), "age")
                         for t, s in zip(df["away_team"], df["season"])],
                        dtype=float)
    exp_home = np.array([_roster_fact(facts, by_team, str(t), int(s), "exp")
                         for t, s in zip(df["home_team"], df["season"])],
                        dtype=float)
    exp_away = np.array([_roster_fact(facts, by_team, str(t), int(s), "exp")
                         for t, s in zip(df["away_team"], df["season"])],
                        dtype=float)
    df["roster_age_diff"] = age_home - age_away
    df["roster_exp_diff"] = exp_home - exp_away
    return df


def team_events(game: pd.DataFrame) -> pd.DataFrame:
    """Long-form one-row-per-(team,game) view used by all trailing features.

    Adds, from the team's perspective: ``team``, ``opponent``, ``is_home``,
    ``net_from_team`` (score diff, + for that team), ``team_win`` (1/0/0.5 tie).
    """
    required = ["game_id", "season", "week", "gameday", "home_team",
                "away_team", "home_score", "away_score"]
    missing = [c for c in required if c not in game.columns]
    if missing:
        raise ValueError(f"team_events: missing columns {missing}")

    gd = pd.to_datetime(game["gameday"], errors="coerce")
    home = pd.DataFrame({
        "game_id": game["game_id"], "season": game["season"], "week": game["week"],
        "gameday": gd, "team": game["home_team"], "opponent": game["away_team"],
        "is_home": True,
        "for": game["home_score"].astype(float), "against": game["away_score"].astype(float),
    })
    away = pd.DataFrame({
        "game_id": game["game_id"], "season": game["season"], "week": game["week"],
        "gameday": gd, "team": game["away_team"], "opponent": game["home_team"],
        "is_home": False,
        "for": game["away_score"].astype(float), "against": game["home_score"].astype(float),
    })
    ev = pd.concat([home, away], ignore_index=True)
    ev["net_from_team"] = ev["for"] - ev["against"]
    ev["team_win"] = np.select(
        [ev["for"] > ev["against"], ev["for"] < ev["against"]],
        [1.0, 0.0], default=0.5)
    return ev


def _elo_apply(events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Iterate ELO over events chronologically; returns (events+elo, ratings).

    ``ratings`` is the final per-team rating dict after the last game — used
    by the slate builder to give scheduled (pre-game) rows their entering
    rating. The per-game ``elo_entering`` values come from ONLY strictly
    prior games (see compute_elo)."""
    K, prior, scale = ELO_K, ELO_PRIOR, ELO_SCALE
    ev = events.sort_values(["gameday", "game_id", "is_home"]).reset_index(drop=True)
    rating: dict = {}
    entering: dict = {}
    for game_id, rows in ev.groupby("game_id", sort=False):
        rows = list(rows.itertuples(index=False))
        a, b = rows[0], rows[1] if len(rows) == 2 else rows[0]
        ra, rb = rating.get(a.team, prior), rating.get(b.team, prior)
        entering[(game_id, a.team)] = ra
        entering[(game_id, b.team)] = rb
        exp_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / scale))
        exp_b = 1.0 / (1.0 + 10.0 ** ((ra - rb) / scale))
        rating[a.team] = ra + K * (a.team_win - exp_a)
        rating[b.team] = rb + K * (b.team_win - exp_b)
    ev = ev.copy()
    ev["elo_entering"] = ev.apply(
        lambda r: entering.get((r["game_id"], r["team"]), prior), axis=1)
    return ev, rating


def compute_elo(events: pd.DataFrame) -> pd.DataFrame:
    """Attach ``elo_entering`` to each (team,event) row: the team's rating at
    kickoff, from ONLY games strictly before this game's gameday.

    Update rule: expected = 1/(1+10**((r_opp - r_self)/400));  r += K*(actual-exp).
    actual = win(1)/loss(0)/tie(0.5). Prior = 1500. Iterated strictly
    chronologically, so a future game can never feed an earlier rating.
    """
    return _elo_apply(events)[0]


def _trailing_per_team(srt: pd.DataFrame, value_col: str, window: int) -> np.ndarray:
    """Per-team windowed mean of ``value_col`` over STRICTLY-PRIOR games.

    ``srt`` must be sorted by (team, gameday, game_id). Rolling mean per team,
    then a per-team shift(1) drops the current row, so each value is the mean
    over that team's prior ``window`` games only. Returned in ``srt`` row order.
    """
    roll = srt.groupby("team", sort=False)[value_col].rolling(
        window, min_periods=1).mean()
    roll = roll.groupby(level=0).shift(1)
    return roll.reset_index(level=0, drop=True).to_numpy()


def _trailing_ewm(srt: pd.DataFrame, value_col: str, halflife: float) -> np.ndarray:
    """Per-team exponentially-weighted mean over STRICTLY-PRIOR games.

    Same shift(1) discipline as ``_trailing_per_team`` — the ewm is computed
    per team over its own games, then shifted so the current row only sees
    strictly-prior games. Small halflife = recent form dominates (decaying
    window, per the v2 candidate spec)."""
    roll = srt.groupby("team", sort=False)[value_col].ewm(
        halflife=halflife, min_periods=1).mean()
    roll = roll.groupby(level=0).shift(1)
    return roll.reset_index(level=0, drop=True).to_numpy()


def team_stats_ladder(events: pd.DataFrame,
                      team_game_agg: pd.DataFrame | None = None) -> pd.DataFrame:
    """For every (game_id, team): elo_entering, form_pts (prior net pts/gm),
    win_pct (prior), rest_days (days since the team's previous game), ypp
    (prior net yards/play), plus the v2 trailing columns: ewm_net_pts,
    ewm_epa, ewm_qb_epa, ewm_scoring, ewm_ypp (decaying windows),
    pace_plays_min, short_rest, and opp_adj_form (opponent-adjusted margin).

    ``team_game_agg``: per (game_id, team) play aggregates from
    ``_pbp_team_agg`` (total_yards, n_plays, epa_sum/epa_n, qb_epa_sum/
    qb_epa_n, elapsed_min). Backward-compatible with the old ``ypp_game``
    shape (game_id, team, total_yards, n_plays).

    LEAKAGE GATE: after sorting by (team, gameday, game_id), gameday must be
    strictly increasing within each team. Combined with the per-team shift
    (windowed AND ewm), no future game can touch any row's trailing
    statistics. This is asserted.
    """
    ev = events.copy()
    if team_game_agg is not None:
        agg = team_game_agg.rename(columns={"total_yards": "tot_yd",
                                            "n_plays": "npl"})
        agg["ypp_game"] = agg["tot_yd"] / agg["npl"].replace(0, np.nan)
        agg["epa_play"] = agg["epa_sum"] / agg["epa_n"].replace(0, np.nan)
        agg["qb_epa_play"] = agg["qb_epa_sum"] / agg["qb_epa_n"].replace(0, np.nan)
        agg["pace_plays_min_game"] = agg["npl"] / agg["elapsed_min"].replace(0, np.nan)
        drop = [c for c in ("tot_yd", "npl", "epa_sum", "epa_n",
                            "qb_epa_sum", "qb_epa_n") if c in agg.columns]
        ev = ev.merge(agg.drop(columns=drop), on=["game_id", "team"], how="left")

    srt = ev.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)

    diffs = srt.groupby("team", sort=False)["gameday"].diff()
    bad = srt.loc[(diffs.notna()) & (diffs <= pd.Timedelta(0))]
    if len(bad):
        raise AssertionError(
            f"team_stats_ladder: team gameday not strictly increasing -> trailing "
            f"features could reference non-prior games ({len(bad)} rows)")

    srt["form_pts"] = _trailing_per_team(srt, "net_from_team", FORM_WINDOW)
    srt["win_pct"] = _trailing_per_team(srt, "team_win", WINPCT_WINDOW)
    srt["rest_days"] = srt.groupby("team", sort=False)["gameday"].diff().dt.days
    srt["short_rest"] = np.where(
        srt["rest_days"].notna(), (srt["rest_days"] < 7).astype(float), np.nan)
    if "ypp_game" in srt.columns:
        srt["ypp"] = _trailing_per_team(srt, "ypp_game", YPP_WINDOW)
    else:
        srt["ypp"] = np.nan

    # ---- v2 decaying-window aggregates (strictly-prior, per team) --------
    srt["ewm_net_pts"] = _trailing_ewm(srt, "net_from_team", EWM_HALFLIFE)
    srt["ewm_scoring"] = _trailing_ewm(srt, "for", EWM_HALFLIFE)
    if "epa_play" in srt.columns:
        srt["ewm_epa"] = _trailing_ewm(srt, "epa_play", EWM_HALFLIFE)
        srt["ewm_qb_epa"] = _trailing_ewm(srt, "qb_epa_play", EWM_HALFLIFE)
    else:
        srt["ewm_epa"] = np.nan
        srt["ewm_qb_epa"] = np.nan
    if "ypp_game" in srt.columns:
        srt["ewm_ypp"] = _trailing_ewm(srt, "ypp_game", EWM_HALFLIFE)
    else:
        srt["ewm_ypp"] = np.nan
    if "pace_plays_min_game" in srt.columns:
        srt["pace_plays_min"] = _trailing_per_team(srt, "pace_plays_min_game", PACE_WINDOW)
    else:
        srt["pace_plays_min"] = np.nan

    # ---- Tier-1 (v3) decaying-window aggregates (strictly-prior, per team) --
    for c in TIER1_AGG_COLUMNS:
        if c in srt.columns:
            srt[f"ewm_{c}"] = _trailing_ewm(srt, c, EWM_HALFLIFE)
        else:
            srt[f"ewm_{c}"] = np.nan
    # net per-team values: takeaways − giveaways, committed − drawn penalties
    srt["ewm_net_turnovers"] = srt["ewm_takeaways"] - srt["ewm_giveaways"]
    srt["ewm_net_penalty"] = srt["ewm_penalty_yds"] - srt["ewm_penalty_yds_drawn"]

    # ---- opponent-adjusted trailing margin -------------------------------
    # opp_form = the opponent's OWN trailing form entering the same game
    # (strictly-prior for the opponent too). The trailing mean over THIS
    # team's prior games of opp_form is the schedule strength faced; the
    # team's margin minus that is the opponent-adjusted variant.
    opp = srt[["game_id", "team", "form_pts"]].rename(
        columns={"team": "opponent", "form_pts": "opp_form"})
    srt = srt.merge(opp, on=["game_id", "opponent"], how="left")
    srt["opp_adj_form"] = srt["form_pts"] - _trailing_per_team(
        srt, "opp_form", OPP_ADJ_WINDOW)
    return srt


# ---------------------------------------------------------------------------
# Feature composition
# ---------------------------------------------------------------------------
def _home_minus_away(ladder: pd.DataFrame, game_ids: pd.Index,
                     col: str) -> np.ndarray:
    home = ladder[ladder["is_home"]].set_index("game_id")[col]
    away = ladder[~ladder["is_home"]].set_index("game_id")[col]
    return (home.reindex(game_ids) - away.reindex(game_ids)).to_numpy()


def _pbp_team_agg(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, team) play-level aggregates for the v2/v3 candidates.

    Base columns (total_yards, n_plays, epa/qb_epa sums+counts, elapsed_min)
    plus the Tier-1 columns: giveaways (interceptions + lost fumbles committed
    on offense), takeaways (same, FORCED — grouped by the defending team),
    net yards per dropback (ANY/A), sack rate, EPA success rate, explosive
    (20+ yd) play rate, penalty yards committed / drawn, third-down conversion
    rate, red-zone TD rate, and points per drive. Every column is a function
    of that game's plays only (sums/rates) — the trailing shift in
    ``team_stats_ladder`` keeps them strictly-prior. Absent source columns
    degrade to NaN (never fabricated), matching the DuckDB engine byte-for-byte
    (the parity test pins this).
    """
    cols = list(TEAM_AGG_COLUMNS)
    if pbp is None or "posteam" not in pbp.columns:
        return pd.DataFrame(columns=cols)
    p = pbp.copy()
    p = p.dropna(subset=["posteam"])
    if "game_id" not in p.columns or "yards_gained" not in p.columns:
        return pd.DataFrame(columns=cols)
    agg = {"total_yards": ("yards_gained", "sum"),
           "n_plays": ("yards_gained", "count")}
    if "epa" in p.columns:
        agg["epa_sum"] = ("epa", "sum")
        agg["epa_n"] = ("epa", "count")
    if "qb_epa" in p.columns:
        agg["qb_epa_sum"] = ("qb_epa", "sum")
        agg["qb_epa_n"] = ("qb_epa", "count")
    g = p.groupby(["game_id", "posteam"], as_index=False).agg(**agg)
    for c in ("epa_sum", "epa_n", "qb_epa_sum", "qb_epa_n"):
        if c not in g.columns:
            g[c] = np.nan
    if "game_seconds_remaining" in p.columns:
        last = (p.dropna(subset=["game_seconds_remaining"])
                 .sort_values("game_seconds_remaining")
                 .drop_duplicates("game_id", keep="first"))
        last["elapsed_min"] = (3600.0 - last["game_seconds_remaining"]) / 60.0
        g = g.merge(last[["game_id", "elapsed_min"]], on="game_id", how="left")
    else:
        g["elapsed_min"] = np.nan
    g = g.rename(columns={"posteam": "team"})

    has = set(pbp.columns)

    def ok(*names: str) -> bool:
        return all(n in has for n in names)

    def _merge(sub: pd.DataFrame) -> None:
        nonlocal g
        g = g.merge(sub, on=["game_id", "team"], how="outer")

    # giveaways — turnovers the offense committed (INT + lost fumbles).
    if ok("interception", "fumble_lost"):
        p2 = p.copy()
        p2["_giveaways"] = p2["interception"].fillna(0.0) + p2["fumble_lost"].fillna(0.0)
        sub = p2.groupby(["game_id", "posteam"], as_index=False).agg(
            giveaways=("_giveaways", "sum")).rename(columns={"posteam": "team"})
        _merge(sub)
    else:
        g["giveaways"] = np.nan
    # takeaways — giveaways the defense forced (opponent's INT + lost fumbles).
    if ok("interception", "fumble_lost") and ok("defteam"):
        d = p.dropna(subset=["defteam"]).copy()
        if not d.empty:
            d["_takeaways"] = d["interception"].fillna(0.0) + d["fumble_lost"].fillna(0.0)
            sub = d.groupby(["game_id", "defteam"], as_index=False).agg(
                takeaways=("_takeaways", "sum")).rename(columns={"defteam": "team"})
            _merge(sub)
        else:
            g["takeaways"] = np.nan
    else:
        g["takeaways"] = np.nan
    # net ANY/A + sack rate (per dropback). nflverse pbp has no sack-yards
    # column; on sack plays ``yards_gained`` is negative (or 0), so sack yards
    # lost = -yards_gained (NaN rows contribute 0, matching the SQL SUM skip).
    if ok("passing_yards", "pass_attempt", "sack"):
        p2 = p.copy()
        p2["_any_num"] = (p2["passing_yards"].fillna(0.0)
                           + np.where(p2["sack"] == 1,
                                      p2["yards_gained"].fillna(0.0), 0.0))
        p2["_any_den"] = p2["pass_attempt"].fillna(0.0) + p2["sack"].fillna(0.0)
        p2["_sack"] = p2["sack"].fillna(0.0)
        sub = p2.groupby(["game_id", "posteam"], as_index=False).agg(
            _any_num=("_any_num", "sum"), _any_den=("_any_den", "sum"),
            _sack=("_sack", "sum")).rename(columns={"posteam": "team"})
        sub["net_any_a"] = np.where(sub["_any_den"] > 0,
                                     sub["_any_num"] / sub["_any_den"], np.nan)
        sub["sack_rate"] = np.where(sub["_any_den"] > 0,
                                     sub["_sack"] / sub["_any_den"], np.nan)
        _merge(sub.drop(columns=["_any_num", "_any_den", "_sack"]))
    else:
        g["net_any_a"] = np.nan
        g["sack_rate"] = np.nan
    # EPA success rate — share of plays with positive EPA.
    if ok("epa"):
        p2 = p.copy()
        p2["_pos"] = (p2["epa"] > 0).astype(float)
        sub = p2.groupby(["game_id", "posteam"], as_index=False).agg(
            _pos=("_pos", "sum"), _n=("epa", "count")).rename(
            columns={"posteam": "team"})
        sub["success_rate"] = np.where(sub["_n"] > 0, sub["_pos"] / sub["_n"], np.nan)
        _merge(sub.drop(columns=["_pos", "_n"]))
    else:
        g["success_rate"] = np.nan
    # Explosive (20+ yd) play rate.
    p2 = p.copy()
    p2["_expl"] = (p2["yards_gained"] >= 20).astype(float)
    sub = p2.groupby(["game_id", "posteam"], as_index=False).agg(
        _expl=("_expl", "sum"), _n=("yards_gained", "count")).rename(
        columns={"posteam": "team"})
    sub["explosive_rate"] = np.where(sub["_n"] > 0,
                                      sub["_expl"] / sub["_n"], np.nan)
    _merge(sub.drop(columns=["_expl", "_n"]))
    # Penalty yards committed (enforced yards only).
    if ok("penalty", "penalty_yards", "penalty_team"):
        p2 = p.copy()
        p2["_comm"] = np.where(
            (p2["penalty"] == 1) & (p2["penalty_team"] == p2["posteam"]),
            p2["penalty_yards"], 0.0)
        sub = p2.groupby(["game_id", "posteam"], as_index=False).agg(
            penalty_yds=("_comm", "sum")).rename(columns={"posteam": "team"})
        _merge(sub)
    else:
        g["penalty_yds"] = np.nan
    # Penalty yards drawn (opponent's enforced yards against this team).
    if ok("penalty", "penalty_yards", "penalty_team", "defteam"):
        p2 = p.copy()
        p2["_drawn"] = np.where(
            (p2["penalty"] == 1) & (p2["penalty_team"] == p2["defteam"]),
            p2["penalty_yards"], 0.0)
        sub = p2.groupby(["game_id", "posteam"], as_index=False).agg(
            penalty_yds_drawn=("_drawn", "sum")).rename(columns={"posteam": "team"})
        _merge(sub)
    else:
        g["penalty_yds_drawn"] = np.nan
    # Third-down conversion rate. nflverse pbp has no third_down_att column:
    # attempts = converted + failed (verified: no row carries both flags).
    if ok("third_down_converted", "third_down_failed"):
        p2 = p.copy()
        p2["_c"] = (p2["third_down_converted"] == 1).astype(float)
        p2["_f"] = (p2["third_down_failed"] == 1).astype(float)
        sub = p2.groupby(["game_id", "posteam"], as_index=False).agg(
            _c=("_c", "sum"), _f=("_f", "sum")).rename(
            columns={"posteam": "team"})
        sub["third_down_rate"] = np.where((sub["_c"] + sub["_f"]) > 0,
                                           sub["_c"] / (sub["_c"] + sub["_f"]), np.nan)
        _merge(sub.drop(columns=["_c", "_f"]))
    else:
        g["third_down_rate"] = np.nan
    # Red-zone TD rate — TD on plays inside the opponent's 20.
    if ok("touchdown", "yardline_100"):
        p2 = p.copy()
        p2["_rz"] = (p2["yardline_100"] <= 20).astype(float)
        p2["_rz_td"] = ((p2["touchdown"] == 1) & (p2["yardline_100"] <= 20)).astype(float)
        sub = p2.groupby(["game_id", "posteam"], as_index=False).agg(
            _rz_td=("_rz_td", "sum"), _rz=("_rz", "sum")).rename(
            columns={"posteam": "team"})
        sub["redzone_td_rate"] = np.where(sub["_rz"] > 0,
                                           sub["_rz_td"] / sub["_rz"], np.nan)
        _merge(sub.drop(columns=["_rz_td", "_rz"]))
    else:
        g["redzone_td_rate"] = np.nan
    # Points per drive (7/TD + 3/FG over distinct drives).
    if ok("touchdown", "field_goal_result", "drive"):
        p2 = p.copy()
        p2["_pts"] = (p2["touchdown"] == 1).astype(float) * 7.0 \
            + (p2["field_goal_result"] == "made").astype(float) * 3.0
        sub = p2.groupby(["game_id", "posteam"], as_index=False).agg(
            _pts=("_pts", "sum"), _drives=("drive", "nunique")).rename(
            columns={"posteam": "team"})
        sub["pts_per_drive"] = np.where(sub["_drives"] > 0,
                                         sub["_pts"] / sub["_drives"], np.nan)
        _merge(sub.drop(columns=["_pts", "_drives"]))
    else:
        g["pts_per_drive"] = np.nan
    for c in cols:
        if c not in g.columns:
            g[c] = np.nan
    return g[cols]


def _decided_rows(sched: pd.DataFrame) -> pd.DataFrame:
    """Rows of a schedule frame with both scores decided (numeric)."""
    s = sched.copy()
    for c in ("home_score", "away_score"):
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    return s[pd.to_numeric(s["home_score"], errors="coerce").notna() &
             pd.to_numeric(s["away_score"], errors="coerce").notna()].copy()


def _pbp_team_agg_engine(pbp: pd.DataFrame | None,
                         extra_names: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Play-by-play rollup preferring the DuckDB engine (MLB-mirrored spill
    config) with a pandas fallback.

    ``nfl_feature_engine`` owns the PBP aggregation in DuckDB the same way MLB
    keeps its big Statcast table in DuckDB (memory_limit + disk spill), while
    ``_pbp_team_agg`` (pandas) stays the source of truth / fallback. Output is
    identical either way (answer-key test in test_nfl_feature_engine.py).
    """
    if pbp is None or "posteam" not in getattr(pbp, "columns", []):
        return _pbp_team_agg(pbp)
    try:
        from nfl_feature_engine import duckdb_available, duckdb_engine, pbp_team_agg
    except Exception:
        return _pbp_team_agg(pbp)
    if not duckdb_available():
        return _pbp_team_agg(pbp)
    try:
        with duckdb_engine() as con:
            return pbp_team_agg(con, pbp, extra_names=tuple(extra_names or ()))
    except Exception as exc:  # noqa: BLE001 — fall back rather than fail the run
        logger.warning("DuckDB pbp rollup failed (%s); using pandas", exc)
        return _pbp_team_agg(pbp)


def build_features(decided: pd.DataFrame,
                   schedule: pd.DataFrame | None = None,
                   pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compose candidate feature columns on a decided game frame.

    The trailing/ELO ladder is computed over ALL decided games present in the
    *schedule* (warmup + core, e.g. 2018-2024) so the earliest scored games get
    real priors; each ladder value is attached to its ``game_id`` row, shifted
    so it uses only strictly-prior games. ``decided`` (2019-2024) is the frame
    scored/reported.
    """
    # --- full decided timeline across warmup+core seasons, from the schedule ---
    sched = schedule.copy() if schedule is not None else decided.copy()
    full = _decided_rows(sched)

    events = compute_elo(team_events(full))
    team_agg = None
    if pbp is not None and {"yards_gained", "posteam"}.issubset(pbp.columns):
        team_agg = _pbp_team_agg_engine(pbp)
    ladder = team_stats_ladder(events, team_agg)

    df = decided.copy()
    # --- per-game schedule facts (roof -> dome, temp, wind, division) ---
    for col, out in (("roof", "roof"), ("temp", "temp_f"),
                     ("wind", "wind_mph"), ("div_game", "div_game")):
        # Never merge a column the frame already carries (pandas would
        # suffix it into roof_x/roof_y and drop the plain name).
        if col not in df.columns and schedule is not None and col in schedule.columns:
            sub = schedule[["game_id", col]].drop_duplicates("game_id")
            df = df.merge(sub, on="game_id", how="left")
        if out not in df.columns:
            df[out] = np.nan
    df["is_dome_home"] = np.where(
        df["roof"].isin(["dome", "closed"]), 1.0,
        np.where(df["roof"].isin(["outdoors"]), 0.0, np.nan))

    gids = df["game_id"]
    df["is_home"] = 1.0                                   # anchor for the home edge
    df["elo_diff"] = _home_minus_away(ladder, gids, "elo_entering")
    df["form_diff_pts"] = _home_minus_away(ladder, gids, "form_pts")
    df["win_pct_diff"] = _home_minus_away(ladder, gids, "win_pct")
    df["rest_days_diff"] = _home_minus_away(ladder, gids, "rest_days")
    df["ypp_diff"] = _home_minus_away(ladder, gids, "ypp")
    # --- v2 diff candidates (decaying windows / opponent-adj / pace / rest) ---
    df["ewm_net_pts_diff"] = _home_minus_away(ladder, gids, "ewm_net_pts")
    df["ewm_epa_play_diff"] = _home_minus_away(ladder, gids, "ewm_epa")
    df["ewm_qb_epa_play_diff"] = _home_minus_away(ladder, gids, "ewm_qb_epa")
    df["ewm_scoring_diff"] = _home_minus_away(ladder, gids, "ewm_scoring")
    df["ewm_ypp_diff"] = _home_minus_away(ladder, gids, "ewm_ypp")
    df["opp_adj_net_pts_diff"] = _home_minus_away(ladder, gids, "opp_adj_form")
    df["pace_plays_min_diff"] = _home_minus_away(ladder, gids, "pace_plays_min")
    df["rest_short_diff"] = _home_minus_away(ladder, gids, "short_rest")
    # --- v3 (Tier-1) diff candidates: turnovers / efficiency / discipline ---
    df["turnover_diff"] = _home_minus_away(ladder, gids, "ewm_net_turnovers")
    df["any_a_diff"] = _home_minus_away(ladder, gids, "ewm_net_any_a")
    df["sack_rate_diff"] = _home_minus_away(ladder, gids, "ewm_sack_rate")
    df["success_rate_diff"] = _home_minus_away(ladder, gids, "ewm_success_rate")
    df["explosive_rate_diff"] = _home_minus_away(ladder, gids, "ewm_explosive_rate")
    df["penalty_diff"] = _home_minus_away(ladder, gids, "ewm_net_penalty")
    df["third_down_rate_diff"] = _home_minus_away(ladder, gids, "ewm_third_down_rate")
    df["redzone_td_rate_diff"] = _home_minus_away(ladder, gids, "ewm_redzone_td_rate")
    df["pts_per_drive_diff"] = _home_minus_away(ladder, gids, "ewm_pts_per_drive")
    # --- v4 (Tier-2) static venue/travel/schedule candidates ----------------
    df = _compose_venue_candidates(df, schedule)
    # --- v5 (Tier-3) candidates: market de-vig / officials / roster ---------
    df = _compose_market_candidates(df, schedule)
    df = _compose_officials_candidates(df, schedule, team_agg)
    df = _compose_roster_candidates(df)
    return df


def build_slate_features(schedule: pd.DataFrame,
                         pbp: pd.DataFrame | None,
                         decided: pd.DataFrame,
                         slate_season: int) -> pd.DataFrame:
    """Leak-safe features + games[] fields for SCHEDULED (undecided) games.

    The trailing/ELO ladder spans the full decided timeline (warmup + core,
    2018-2025), then the scheduled rows of ``slate_season`` are appended so
    their per-team trailing stats are computed from strictly-prior DECIDED
    games only (same shift(1) discipline as ``build_features``; the
    monotonicity assertion in ``team_stats_ladder`` still holds because the
    scheduled rows are the latest). Each scheduled row's ``elo_entering`` is
    the team's rating after the last decided game.

    Returns a frame with one row per scheduled game carrying every
    FEATURE_COLUMNS candidate plus the games[]-shaped fields: season, week,
    gameday, gametime, stadium, spread_line, total_line, home_record,
    away_record.
    """
    full = _decided_rows(schedule)
    ev, ratings = _elo_apply(team_events(full))
    team_agg = None
    if pbp is not None and {"yards_gained", "posteam"}.issubset(pbp.columns):
        team_agg = _pbp_team_agg_engine(pbp)

    sched = schedule.copy()
    for c in ("home_score", "away_score"):
        if c in sched.columns:
            sched[c] = pd.to_numeric(sched[c], errors="coerce")
    sched_rows = sched[(sched.get("season") == slate_season) &
                       (sched["home_score"].isna() | sched["away_score"].isna())].copy()
    if sched_rows.empty:
        return pd.DataFrame()

    ev_sched = team_events(sched_rows)
    ev_sched["elo_entering"] = ev_sched["team"].map(
        lambda t: ratings.get(t, ELO_PRIOR))
    combined = pd.concat([ev, ev_sched], ignore_index=True)
    ladder = team_stats_ladder(combined, team_agg)

    df = sched_rows.copy()
    for col, out in (("roof", "roof"), ("temp", "temp_f"),
                     ("wind", "wind_mph"), ("div_game", "div_game")):
        if col not in df.columns and col in sched.columns:
            sub = sched[["game_id", col]].drop_duplicates("game_id")
            df = df.merge(sub, on="game_id", how="left")
        if out not in df.columns:
            df[out] = np.nan
    df["is_dome_home"] = np.where(
        df["roof"].isin(["dome", "closed"]), 1.0,
        np.where(df["roof"].isin(["outdoors"]), 0.0, np.nan))

    gids = df["game_id"]
    df["is_home"] = 1.0
    df["elo_diff"] = _home_minus_away(ladder, gids, "elo_entering")
    df["form_diff_pts"] = _home_minus_away(ladder, gids, "form_pts")
    df["win_pct_diff"] = _home_minus_away(ladder, gids, "win_pct")
    df["rest_days_diff"] = _home_minus_away(ladder, gids, "rest_days")
    df["ypp_diff"] = _home_minus_away(ladder, gids, "ypp")
    df["ewm_net_pts_diff"] = _home_minus_away(ladder, gids, "ewm_net_pts")
    df["ewm_epa_play_diff"] = _home_minus_away(ladder, gids, "ewm_epa")
    df["ewm_qb_epa_play_diff"] = _home_minus_away(ladder, gids, "ewm_qb_epa")
    df["ewm_scoring_diff"] = _home_minus_away(ladder, gids, "ewm_scoring")
    df["ewm_ypp_diff"] = _home_minus_away(ladder, gids, "ewm_ypp")
    df["opp_adj_net_pts_diff"] = _home_minus_away(ladder, gids, "opp_adj_form")
    df["pace_plays_min_diff"] = _home_minus_away(ladder, gids, "pace_plays_min")
    df["rest_short_diff"] = _home_minus_away(ladder, gids, "short_rest")
    # --- v3 (Tier-1) diff candidates (same strictly-prior ladder) ----------
    df["turnover_diff"] = _home_minus_away(ladder, gids, "ewm_net_turnovers")
    df["any_a_diff"] = _home_minus_away(ladder, gids, "ewm_net_any_a")
    df["sack_rate_diff"] = _home_minus_away(ladder, gids, "ewm_sack_rate")
    df["success_rate_diff"] = _home_minus_away(ladder, gids, "ewm_success_rate")
    df["explosive_rate_diff"] = _home_minus_away(ladder, gids, "ewm_explosive_rate")
    df["penalty_diff"] = _home_minus_away(ladder, gids, "ewm_net_penalty")
    df["third_down_rate_diff"] = _home_minus_away(ladder, gids, "ewm_third_down_rate")
    df["redzone_td_rate_diff"] = _home_minus_away(ladder, gids, "ewm_redzone_td_rate")
    df["pts_per_drive_diff"] = _home_minus_away(ladder, gids, "ewm_pts_per_drive")
    # --- v4 (Tier-2) static venue/travel/schedule candidates ----------------
    df = _compose_venue_candidates(df, sched)
    # --- v5 (Tier-3) candidates: market de-vig / officials / roster ---------
    df = _compose_market_candidates(df, sched)
    df = _compose_officials_candidates(df, sched, team_agg)
    df = _compose_roster_candidates(df)

    # --- cumulative records entering the slate (from the decided timeline) ---
    rec = ev.groupby("team").agg(
        wins=("team_win", lambda s: float((s == 1).sum())),
        losses=("team_win", lambda s: float((s == 0).sum())),
        ties=("team_win", lambda s: float((s == 0.5).sum())),
    )

    def _record(team: str) -> str:
        if team not in rec.index:
            return ""
        r = rec.loc[team]
        base = f"{int(r['wins'])}-{int(r['losses'])}"
        return f"{base}-{int(r['ties'])}" if r["ties"] > 0 else base

    df["home_record"] = df["home_team"].map(_record)
    df["away_record"] = df["away_team"].map(_record)
    return df


# ---------------------------------------------------------------------------
# Audit + admission gate
# ---------------------------------------------------------------------------
def audit_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """% non-null per candidate + 12h-availability (== coverage, per the stated
    assumption that every candidate is a function of prior games or a static
    venue/prior fact)."""
    rows = {}
    for f in FEATURE_COLUMNS:
        cov = float(df[f].notna().mean())
        rows[f] = {
            "coverage_pct": round(100 * cov, 2),
            "available_12h_pct": round(100 * cov, 2),
            "source": CANONICAL_SOURCE[f],
        }
    return pd.DataFrame(rows).T


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (shared with scipy.stats.rankdata 'average')."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty(n, dtype=np.intp)
    inv[sorter] = np.arange(n)
    xs = x[sorter]
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks[inv]


def univariate_auc(home_win: np.ndarray, feature: np.ndarray) -> float:
    """P(feature of a win > feature of a loss), mean tie=0.5. AUC>0.5 ->
    higher feature -> home win. NaN if either class absent."""
    y = np.asarray(home_win)
    x = np.asarray(feature, dtype=float)
    mask = ~np.isnan(x) & np.isfinite(x)
    y, x = y[mask], x[mask]
    pos = x[y == 1]
    neg = x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = _rankdata(np.concatenate([pos, neg]))
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) \
        / (len(pos) * len(neg))


def audit_correlation(df: pd.DataFrame) -> pd.DataFrame:
    num = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return num.corr()


def _strong_pairs(corr: pd.DataFrame) -> list[dict]:
    seen = set()
    out = []
    for f in corr.columns:
        for g in corr.columns:
            if f >= g or (f, g) in seen or (g, f) in seen:
                continue
            v = corr[f][g]
            if pd.notna(v) and abs(v) > 0.8:
                seen.add((f, g))
                out.append({"feat_a": f, "feat_b": g, "corr": round(float(v), 4)})
    return out


def run_feature_gate(df: pd.DataFrame,
                     auto_prune: bool = GATE_AUTO_PRUNE) -> dict:
    """Audit + admission report over FEATURE_COLUMNS (the served pool).

    DEFAULT POLICY (auto_prune=False — GATE_AUTO_PRUNE): the gate NEVER
    removes features from the served pool. Coverage / redundancy /
    near-random-AUC are computed and RECORDED for monitoring, and any
    registered feature below the coverage floor or in a strong-correlation
    pair (|r| > {CORR_REDUNDANCY}) logs a LOUD warning — informational,
    never blocking. The returned pool is exactly FEATURE_COLUMNS minus the
    ``is_home`` anchor, unchanged.

    auto_prune=True re-enables the LEGACY pruning (coverage floor, then
    redundant-pair / near-random-AUC pruning) — a deliberate opt-in, never
    the default.
    """
    covered = [f for f in FEATURE_COLUMNS
               if float(df[f].notna().mean()) >= COVERAGE_FLOOR]
    below = [f for f in FEATURE_COLUMNS if f not in covered]

    pre = df[df["season"] < HOLD_SEASON]        # 2025 sealed holdout stays out
    y = (pre["home_score"] > pre["away_score"]).astype(int).to_numpy()
    auc = {}
    for f in FEATURE_COLUMNS:
        x = pre[f].to_numpy()
        if pre[f].nunique(dropna=True) <= 1:    # constant -> no discriminative info
            auc[f] = float("nan")
        else:
            auc[f] = univariate_auc(y, x)
    corr = audit_correlation(df)
    strong = _strong_pairs(corr)

    if not auto_prune:
        for f in below:
            logger.warning("GATE [no-prune]: %s coverage %.1f%% is below the "
                           "%.0f%% floor (report-only — the served pool is "
                           "unchanged)", f, 100 * float(df[f].notna().mean()),
                           COVERAGE_FLOOR * 100)
        for p in strong:
            logger.warning("GATE [no-prune]: |r| %.2f between %s and %s "
                           "exceeds %.2f (report-only — the served pool is "
                           "unchanged)", p["corr"], p["feat_a"],
                           p["feat_b"], CORR_REDUNDANCY)
        v1 = [f for f in FEATURE_COLUMNS if f != "is_home"]
        return {
            "covered_features": covered,
            "below_coverage_floor": below,
            "v1_features": v1,
            "kept_home_anchor": "is_home" in FEATURE_COLUMNS,
            "auto_prune": False,
            "audit_coverage": audit_coverage(df).to_dict(orient="index"),
            "univariate_auc": {k: (None if pd.isna(v) else round(float(v), 4))
                               for k, v in auc.items()},
            "correlation_pairs_over_0_8": strong,
            "reasons": {},
            "dropped": [],
        }

    # ---- LEGACY pruning (opt-in via GATE_AUTO_PRUNE=True) -------------
    v1 = list(FEATURE_COLUMNS)
    reasons = {}
    # R0 coverage floor
    for f in FEATURE_COLUMNS:
        if f not in covered:
            if f in v1:
                v1.remove(f)
            reasons[f] = (f"coverage {float(df[f].notna().mean()):.1%} below "
                          f"{COVERAGE_FLOOR:.0%} floor")
    # R1 near-random AND redundant with a stronger feature -> prune
    for f in list(v1):
        if f == "is_home" or f not in auc or pd.isna(auc[f]):
            continue
        if abs(auc[f] - 0.5) < DISC_BAND:
            for g in v1:
                if g == f or g == "is_home" or g not in auc or pd.isna(auc[g]):
                    continue
                if abs(corr[f][g]) > CORR_REDUNDANCY and \
                        abs(auc[g] - 0.5) > abs(auc[f] - 0.5) + 1e-9:
                    if f in v1:
                        v1.remove(f)
                        reasons[f] = (f"auc {auc[f]:.3f} ~ random and |r| "
                                      f"{abs(corr[f][g]):.2f} with {g} "
                                      f"(stronger discriminator)")
                    break
    # R2 redundant pair (|r| large, similar discrimination) -> keep one
    kept = set()
    for f in list(v1):
        if f in kept:
            continue
        for g in list(v1):
            if g == f or g in kept or f not in auc or g not in auc or \
                    pd.isna(auc[f]) or pd.isna(auc[g]):
                continue
            if abs(corr[f][g]) > CORR_REDUNDANCY and \
                    abs(abs(auc[f] - 0.5) - abs(auc[g] - 0.5)) < DISC_BAND:
                disc_f, disc_g = abs(auc[f] - 0.5), abs(auc[g] - 0.5)
                strong, weak = f, g
                if disc_g > disc_f:
                    strong, weak = g, f
                elif disc_f == disc_g and \
                        FEATURE_PRIORITY.get(weak, 99) < FEATURE_PRIORITY.get(strong, 99):
                    strong, weak = weak, strong
                if weak in v1:
                    v1.remove(weak)
                    reasons[weak] = (f"redundant with {strong} (|r| "
                                     f"{abs(corr[f][g]):.2f}, similar discrimination); "
                                     f"keep one")
                kept.add(strong)
                kept.add(weak)

    is_home_in = "is_home" in v1          # constant anchor stays in v1 set
    return {
        "covered_features": covered,
        "below_coverage_floor": below,
        "v1_features": v1,
        "kept_home_anchor": is_home_in,
        "auto_prune": True,
        "audit_coverage": audit_coverage(df).to_dict(orient="index"),
        "univariate_auc": {k: (None if pd.isna(v) else round(float(v), 4))
                           for k, v in auc.items()},
        "correlation_pairs_over_0_8": strong,
        "reasons": reasons,
        "dropped": [f for f in FEATURE_COLUMNS if f not in v1],
    }


# ---------------------------------------------------------------------------
# Loaders + orchestration
# ---------------------------------------------------------------------------
def _load_raw(seasons: list[int]):
    import nflreadpy
    sched = nflreadpy.load_schedules(seasons).to_pandas()
    pbp = nflreadpy.load_pbp(seasons)
    # v2 candidates need EPA / QB-EPA / the game clock (pace); select only the
    # needed columns in polars before converting (pbp is ~370 columns wide).
    keep = [c for c in (("game_id", "posteam", "yards_gained", "epa",
                         "qb_epa", "game_seconds_remaining") + TIER1_NEEDS)
            if c in pbp.columns]
    pbp = pbp.select(keep).to_pandas()
    return sched, pbp


def pull_and_build(out_dir: Path | None = None,
                   write_record: bool = True,
                   seasons: list[int] | None = None) -> dict:
    """Build + gate feature candidates over an optional season window.

    ``seasons`` limits both the decided frame and the schedule+pbp pull to the
    given seasons (e.g. ``[2021, 2022, 2023]``). None (default) keeps the full
    warmup+core range, so behavior is unchanged for normal runs.
    """
    seasons = seasons or DEFAULT_SEASONS
    out_dir = Path(out_dir) if out_dir is not None else DATA_DELIVERY_DIR
    if not DECIDED_FRAME.exists():
        raise FileNotFoundError(
            f"{DECIDED_FRAME} absent — run `python3 nfl_game_frame.py` first")
    decided = pd.read_csv(DECIDED_FRAME)
    if "season" in decided.columns:
        decided = decided[decided["season"].isin(seasons)]

    logger.info("Loading nflreadpy schedule+pbp (window): %s", seasons)
    schedule, pbp = _load_raw(seasons)
    feats = build_features(decided, schedule, pbp)

    result = run_feature_gate(feats)

    if write_record:
        record = {
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "version": "v2",
            "build": {
                "core_seasons": CORE_SEASONS,
                "warmup_seasons": WARMUP_SEASONS,
                "decided_games": int(len(feats)),
                "decided_frame": str(DECIDED_FRAME),
                "elo_prior": ELO_PRIOR, "elo_k": ELO_K,
                "windows": {"form": FORM_WINDOW, "win_pct": WINPCT_WINDOW,
                            "ypp": YPP_WINDOW, "ewm_halflife": EWM_HALFLIFE,
                            "opp_adj": OPP_ADJ_WINDOW, "pace": PACE_WINDOW},
                "candidate_set": ("v1 base + v2: decaying-window strength "
                                  "aggregates (net pts / EPA / yards / scoring), "
                                  "opponent-adjusted margin, pace, short-rest "
                                  "edge, QB EPA, weather (temp/wind), division"),
                "leakage_rule": ("every trailing feature (windowed OR decaying-ewm) "
                                 "uses only games with gameday strictly before the "
                                 "target (asserted in code: team_stats_ladder strict "
                                 "monotonicity)."),
                "holdout": "2025 is not in the decided frame; all AUC computed on "
                           "seasons < 2025.",
            },
            "feature_admission": result,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        rec_path = out_dir / RECORD_TEMPLATE.format(date=datetime.now().strftime(DATE_FMT))
        with open(rec_path, "w") as fh:
            json.dump(record, fh, indent=2)
        result["record"] = str(rec_path)

    _print_report(feats, result)
    return result


def _print_report(feats: pd.DataFrame, result: dict) -> None:
    print("\n=== NFL feature admission (v1 base + v2 candidates, no model) ===")
    print(f"decided games scored: {len(feats)}")
    cov = pd.DataFrame(result["audit_coverage"]).T
    print("\ncoverage / 12h availability:")
    print(cov[["coverage_pct", "available_12h_pct", "source"]].to_string())
    print("\nstrong correlation pairs (|r| > 0.8):")
    for p in result.get("correlation_pairs_over_0_8", []):
        print(f"  {p['feat_a']} ~ {p['feat_b']}: r={p['corr']}")
    print("\nunivariate AUC (seasons < 2025):")
    for f, v in result["univariate_auc"].items():
        print(f"  {f:18s} {v if v is not None else 'n/a'}")
    print("\nadmitted features:", result["v1_features"])
    print("dropped:", result.get("dropped"))
    for f, r in result.get("reasons", {}).items():
        print(f"  drop {f}: {r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        description="Build + gate NFL feature candidates v1 (no model).")
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON record")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    pull_and_build(write_record=not args.no_record)
    return 0


if __name__ == "__main__":
    sys.exit(main())