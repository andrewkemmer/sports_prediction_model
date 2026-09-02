"""NFL feature engineering — leakage-safe raw candidates + served-pool manifest.

TIER-4 NOTE: v6 game-script / opponent-adjusted / drive-level / QB-conditional
PBP candidates are composed by this module + ``nfl_tier4.py`` and stay OUT of
FEATURE_COLUMNS until the Tier-4 sealed ablation admits them.

Builds on the committed game-level frame produced by ``nfl_game_frame.py``
(``nfl-backend/data_delivery/nfl_game_level_features.csv``). This module
builds the served features and writes the STATIC served-pool manifest
(``nfl_feature_v1_<date>.json``) — the Phase-2 feature-ADMISSION gate
(coverage / available_12h / univariate-AUC / corr-pair |r|>0.8 / auto-prune)
was RETIRED 2026-09-02 by the NFL↔MLB parity pass. It does NOT train any
model — the walk-forward ensemble lives in ``nfl_moneyline.py``.

v1 base (admitted 2026-08-28): elo_diff, form_diff_pts, rest_days_diff,
ypp_diff, is_dome_home (+ is_home anchor). v2 candidates (this file):
trailing per-team strength aggregates with SMALL DECAYING WINDOWS
(exponentially-weighted net-points margin, EPA/play, yards/play, scoring
output), opponent-adjusted variants (trailing margin minus the trailing form
of the opponents faced), pace (plays/min), rest-days edge (short-rest flag),
QB/offense-quality edge (decaying QB EPA/play), weather beyond the dome flag
(game temp / wind at the home venue), and the division-game flag. All are
leak-safe by construction; the former admission gate (coverage floor /
redundancy pruning / near-random-AUC pruning) was RETIRED 2026-09-02 and the
served pool is FEATURE_COLUMNS by declaration (2025 stays the sealed
hold-out, untouched).

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
- No feature uses model probabilities or later results. No hand-multiplied
  "risk" interactions, no injury reports (not reliably final 12h
  pre-kickoff), no weather.
- MARKET-INDEPENDENCE POLICY (2026-09-01): no market/odds data anywhere in
  the NFL pipeline — not in the source frame, not as a feature, not as a
  gate arm, not on the board. The former market de-vig candidate
  (market_home_implied) and its helpers were DELETED, not merely
  unregistered. The v5 (Tier-3) referee-crew and roster age/exp candidates
  ARE composed by build_features/build_slate_features but stay OUT of
  FEATURE_COLUMNS unless a sealed-2025 ablation admits them (the
  Tier-1/Tier-2 rule).

Missingness policy (parity with MLB's lineup-actuals approach, 2026-09-02):
features are admitted on merit by sealed ablation and missingness is handled
IN-MODEL at serve time — no feature is ever excluded for sparse coverage, and
no data-dependent admission audit runs in production anymore.

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
from nfl_tier4 import (TIER4_PBP_NEEDS, qb_map_from_schedule, tier4_team_agg)

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
# used for feature admission/fitting (the retired gate's AUC window stayed on
# seasons < HOLD_SEASON so the hold-out row remained clean for nfl_moneyline;
# the hold-out itself is unchanged by the 2026-09-02 gate retirement).
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

# ---------------------------------------------------------------------------
# RETIRED admission gate (2026-09-02, NFL↔MLB parity pass)
# The Phase-2 feature-admission gate — coverage floor / available_12h /
# univariate-AUC / corr-pair |r|>0.8 / auto-prune — no longer runs anywhere
# in production. The served pool is FEATURE_COLUMNS BY DECLARATION (features
# are admitted on merit by sealed ablation only; missingness is handled
# IN-MODEL at serve time). ``GATE_AUTO_PRUNE`` / ``COVERAGE_FLOOR`` /
# ``CORR_REDUNDANCY`` / ``DISC_BAND`` / ``FEATURE_PRIORITY`` and the gate
# functions below are KEPT ONLY for ablation-harness back-compat
# (``run_tier1_ablation`` / ``run_nfl_unified_confirm_ablation`` import
# ``run_feature_gate``); the pipeline emits ``served_pool_manifest()``.
# ---------------------------------------------------------------------------
GATE_AUTO_PRUNE = False
COVERAGE_FLOOR = 0.95
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

# The SERVED pool, in manifest order: EXACTLY this list (minus the ``is_home``
# anchor at predict time) is what ``nfl_moneyline`` consumes. The admission
# gate was RETIRED 2026-09-02 — this list is the static manifest, not a gate
# output, and it is identical on every fold/pull.
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
#   market_home_implied        — NO LONGER EXISTS (DELETED 2026-09-01 under
#       the market-independence policy: zero market data anywhere in the
#       NFL pipeline — frame, features, gate arms, board). The de-vig
#       candidate admitted by the Tier-3 MARK verdict (76002fb) and then
#       policy-reverted was removed from the codebase entirely, along with
#       its _american_implied / _compose_market_candidates helpers.
# None of these was ever removed by a SEALED-ABLATION verdict (qb_epa was
#   removed by the corr-pair twin verdict above): the rest were pruned at
#   admission time by the LEGACY coverage/redundancy gate (retired 2026-09-02
#   with the parity pass; the function is kept for ablation-harness
#   back-compat), and they keep appearing in the ablation
#   WITHOUT baselines because build_features/build_slate_features still
#   compose them. Unregistering them here is the deliberate policy decision
#   that the served pool is exactly the 12 features above and that nothing
#   is ever removed from it automatically. (The v3 Tier-1
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
# Tier-3 (v5) candidates — officials / roster age+experience. (The former
# market de-vig family was DELETED under the market-independence policy —
# no market data anywhere in the NFL pipeline, so no ablation arm either.)
#
# Composed by build_features/build_slate_features but deliberately NOT
# registered in FEATURE_COLUMNS / CANONICAL_SOURCE / FEATURE_PRIORITY: the
# deployed pool changes only when a sealed-2025 ablation admits a feature
# (Tier-1/Tier-2 rule), so run_tier3_ablation.py is the only admission path.
#
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
TIER3_OFF_FEATURES = ["ref_pen_tend", "ref_pace"]
TIER3_ROSTER_FEATURES = ["roster_age_diff", "roster_exp_diff"]

# Ablation verdict (run_tier3_ablation.py, frame e4aee120a4b8, 2026-09-01):
#   MARK   ADOPT       (sealed 0.6339/0.7121/ECE 0.0759 vs WITHOUT
#                       0.6507/0.6817/0.0937; pooled 0.6026 corroborates; all
#                       five members improve sealed both axes) — admitted
#                       into FEATURE_COLUMNS (76002fb), then DELIBERATELY
#                       REVERSED BY POLICY and DELETED 2026-09-01: the
#                       market-independence policy forbids market data
#                       anywhere in the pipeline, so the de-vig candidate
#                       and helpers were removed from the codebase, not
#                       just the pool.
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


# ---------------------------------------------------------------------------
# Tier-5 (v7, PLAYER-LEVEL) expected-QB-starter identity candidates -
# composed but UNREGISTERED (absent from FEATURE_COLUMNS; admission only by
# a sealed-2025 ablation verdict - the Tier-1 rule). First player-level
# expansion (2026-09-02).
#
# WHY THIS FAMILY: the served 12-pool trails TEAM-level strength (ewm_ypp /
# ewm_net_pts / pace ...), which is stale in QB-change games (~10-15% of
# games) - those windows describe the OTHER quarterback who started the
# team's prior games. The team-level QB-EPA composite was already rejected
# (ewm_qb_epa_play_diff |r| 0.8055 with ewm_ypp_diff - corr-pair twin
# verdict, DON'T ADOPT, cd3c26b). This family tests starter-IDENTITY
# conditioning instead - a different hypothesis (per-game player-state
# priors, not another trailing team aggregate).
#
# EXPECTED STARTER = the team's published QB1 from the nflverse depth chart
# as of BEFORE the game's kickoff (never the pbp/schedule ACTUAL starter of
# the target game - that is post-game truth and would leak the target):
#   - seasons with weekly charts (nflverse parsed files, 2001-2024): the QB
#     row with ``depth_team == 1`` for the game's (season, week). The chart
#     is published mid-week pre-game, so it is pre-kickoff by construction.
#   - seasons without weekly charts (nflverse publishes only dated rolling
#     snapshots from 2025 on): the QB row with ``pos_abb == QB`` and
#     ``pos_rank == 1`` from the LATEST snapshot whose ``dt`` (UTC) is
#     STRICTLY before the game's kickoff (UTC) - an exact pre-cutoff state.
# The ACTUAL starter (schedule home_qb_id/away_qb_id) enters ONLY as
# strictly-prior facts: the team's prior-game starter, and the trailing
# starts / EPA counts used by the skill fallback. Same per-team
# chronological discipline as the 12-pool ladder; nothing below ever reads
# the target game's own actual starter to build its value.
#
# qb1_skill_diff        - expected starter's QB EPA/play with a fallback
#       structure (deliberately NOT separate season/last-5 columns: new
#       starters have neither a full season nor 5 starts): current-season
#       mean over the starter's >= QB1_MIN_STARTS_CURRENT_SEASON starts
#       (strictly prior) -> the starter's prior-season (S-1) mean over his
#       starts (any team) -> QB1_REPLACEMENT_EPA_PLAY (league-average
#       prior, ~0 by construction of nflverse qb_epa). Home - away.
# qb1_continuity_diff   - expected starter's prior starts with this team
#       (strictly prior, capped at QB1_CONTINUITY_CAP) - system-familiarity
#       prior, defined even with zero current-season EPA. Home - away.
# qb1_change_diff       - flag: expected starter != team's prior-game
#       actual starter (strictly prior), in {0, 1} per side; home - away
#       in {-1, 0, 1}. THE conditioning variable for the Tier-5 decision
#       surface (the pooled/sealed marginal averages are dilution-heavy).
# qb1_primary_out_diff  - flag: team's primary QB (prior-season starts
#       leader) is NOT the expected starter this week, per side; home - away
#       in {-1, 0, 1} - the directional "lost star" signal.
#
# Coverage is verified in the ablation record (target: >= 95% in-frame per
# column, as with every candidate family); corr vs the 12-pool is recorded
# as DIAGNOSTICS only - the corr-pair admission gate is retired (2026-09-02)
# and must not be reintroduced as a filter.
# ---------------------------------------------------------------------------
TIER5_QB_FEATURES = [
    "qb1_skill_diff",
    "qb1_continuity_diff",
    "qb1_change_diff",
    "qb1_primary_out_diff",
]

# Skill fallback chain thresholds / constants.
QB1_MIN_STARTS_CURRENT_SEASON = 4    # >= this many prior starts -> season EPA
QB1_REPLACEMENT_EPA_PLAY = 0.0        # league-average prior (qb_epa ~ centered)
QB1_CONTINUITY_CAP = 16               # a full season of starts saturates familiarity

# nflverse depth-chart keying (see the resolver helpers below): weekly parsed
# charts mark the starter with ``depth_team == 1``; rolling snapshots mark it
# with ``pos_abb == QB`` and ``pos_rank == 1``.
_DC_DEPTH1 = 1
_DC_QB_POS = "QB"

# Excluded nflverse ``game_type`` values when building the decided long frame
# (preseason / bye-week pseudo-games must never act as "prior games" for a
# team's starter sequence).
_TIER5_EXCLUDED_GAME_TYPES = ("PRE", "SBBYE")


def _parse_kickoff_et(row) -> pd.Timestamp | None:
    """Kickoff as a naive-UTC Timestamp from gameday + gametime.

    nflverse ``gametime`` is the scheduled ET kickoff; recent pulls are
    24-hour ("13:00"), older feeds "1:00PM" - both are handled. A missing /
    unparseable gametime falls back to noon ET on gameday (still pre-kickoff
    for snapshot selection; never post-game).
    """
    gd = row.get("gameday")
    if gd is None or pd.isna(gd):
        return None
    try:
        day = pd.Timestamp(gd).normalize()
    except Exception:
        return None
    gt = row.get("gametime")
    hour, minute = 12, 0
    if gt is not None and pd.notna(gt):
        txt = str(gt).strip()
        try:
            t = txt.upper()
            if "PM" in t or "AM" in t:
                hhmm = t.replace("PM", "").replace("AM", "").strip()
                hh, mm = (int(x) for x in hhmm.split(":"))
                if "PM" in t and hh != 12:
                    hh += 12
                if "AM" in t and hh == 12:
                    hh = 0
            else:
                hh, mm = (int(x) for x in txt.split(":"))
            hour, minute = hh, mm
        except Exception:
            hour, minute = 12, 0
    try:
        return (day.replace(hour=hour, minute=minute)
                .tz_localize("America/New_York").tz_convert("UTC")
                .tz_localize(None))
    except Exception:
        return day.replace(hour=12).tz_localize("America/New_York")


def _dc_qb1_weekly(weekly: pd.DataFrame | None) -> dict:
    """(season, week, team) -> gsis_id of the depth-chart QB1.

    Weekly parsed charts (2001-2024): QB1 = the QB row with
    ``depth_team == 1`` for the (season, week, team). Duplicate QB1 cells
    (a chart quirk) resolve deterministically to the row that sorts first by
    gsis_id - never fabricated.
    """
    out: dict = {}
    if weekly is None:
        return out
    team_col = ("team" if "team" in weekly.columns else
                ("club_code" if "club_code" in weekly.columns else None))
    if team_col is None or not {"season", "week", "gsis_id", "position"
                                }.issubset(weekly.columns):
        return out
    w = weekly.copy()
    w = w[w["position"].astype(str).str.upper() == _DC_QB_POS]
    if "depth_team" not in w.columns:
        return out
    w = w[pd.to_numeric(w["depth_team"], errors="coerce") == _DC_DEPTH1]
    w = w.dropna(subset=["season", "week", team_col, "gsis_id"])
    w["_team"] = w[team_col].astype(str).str.strip().str.upper()
    w = w.sort_values("gsis_id").drop_duplicates(
        ["season", "week", "_team"], keep="first")
    for r in w[["season", "week", "_team", "gsis_id"]].itertuples(
            index=False, name=None):
        out[(int(r[0]), float(r[1]), r[2])] = r[3]
    return out


def _dc_qb1_snapshots(snapshots: pd.DataFrame | None) -> dict:
    """team -> sorted [(dt_utc_naive, gsis_id)] QB1 snapshots (dt ascending).

    Rolling snapshots (2025+): QB1 = the row with ``pos_abb == QB`` and
    ``pos_rank == 1``; the per-(team, dt) cell is unique in the real data
    (verified: 0 multi-QB1 cells across the 2025-2026 feed); a duplicate
    cell resolves to the first row by gsis_id defensively.
    """
    out: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    if snapshots is None or not {"team", "dt", "gsis_id"}.issubset(
            snapshots.columns):
        return out
    s = snapshots.copy()
    if "pos_abb" in s.columns:
        s = s[s["pos_abb"].astype(str).str.upper() == _DC_QB_POS]
    if "pos_rank" in s.columns:
        s = s[pd.to_numeric(s["pos_rank"], errors="coerce") == 1]
    s = s.dropna(subset=["dt", "team", "gsis_id"])
    s["_dt"] = pd.to_datetime(s["dt"], errors="coerce", utc=True).dt.tz_localize(None)
    s = s.dropna(subset=["_dt"])
    s["team"] = s["team"].astype(str).str.strip().str.upper()
    s = s.sort_values(["team", "_dt", "gsis_id"]).drop_duplicates(
        ["team", "_dt"], keep="first")
    for team, sub in s.groupby("team"):
        out[team] = sorted(
            [(ts, gid) for ts, gid in zip(sub["_dt"], sub["gsis_id"])],
            key=lambda x: x[0])
    return out


def _expected_starter_for(team: str, season: int, week, kickoff_utc,
                          weekly_map: dict, snap_map: dict) -> str | None:
    """The team's published QB1 before this game (gsis_id), else None.

    Weekly charts first (by season/week when the chart exists for the team),
    then the rolling-snapshot state strictly before kickoff."""
    team = str(team).strip().upper()
    if weekly_map:
        gid = weekly_map.get((int(season), float(week), team))
        if gid is not None:
            return gid
    if snap_map and kickoff_utc is not None and team in snap_map:
        prior = [gid for ts, gid in snap_map[team] if ts < kickoff_utc]
        if prior:
            return prior[-1]
    return None


def compose_tier5_qb_features(df: pd.DataFrame,
                              schedule: pd.DataFrame | None = None,
                              pbp: pd.DataFrame | None = None,
                              depth_weekly: pd.DataFrame | None = None,
                              depth_snapshots: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the 4 Tier-5 QB-starter identity candidates to a built frame.

    ``df`` is ``build_features`` output (game_id, season, week, gameday,
    teams, scores). ``schedule`` carries the decided rows + the schedule
    ``home_qb_id``/``away_qb_id`` (the RECORDED actual starter, used ONLY
    as strictly-prior facts); ``pbp`` supplies per-pass ``qb_epa`` for the
    skill fallback; ``depth_weekly`` / ``depth_snapshots`` are the two
    nflverse depth-chart shapes (see above - the compose seam stays pure and
    network-free; the harness loads them).

    Every candidate value is a per-(game, team) state computed by one
    chronological pass over the decided timeline (warmup 2018 included, so
    the first 2019 games have real priors): reads strictly-prior counters
    FIRST, then updates them with the current row - future rows can never
    touch an earlier value. Candidates are home - away via the same
    ``_home_minus_away`` seam as the 12-pool. Rows whose expected starter
    cannot be resolved (no weekly chart AND no snapshot before kickoff) get
    NaN on all four candidates - never fabricated.
    """
    for c in TIER5_QB_FEATURES:
        df[c] = np.nan
    need = {"game_id", "season", "week", "gameday", "home_team", "away_team",
            "home_score", "away_score"}
    if schedule is None or not need.issubset(df.columns) or "game_id" not in \
            schedule.columns:
        return df
    sched = schedule.copy()
    for c in ("home_score", "away_score"):
        if c in sched.columns:
            sched[c] = pd.to_numeric(sched[c], errors="coerce")
    sched = sched[sched["home_score"].notna() & sched["away_score"].notna()]
    if "game_type" in sched.columns:
        sched = sched[~sched["game_type"].astype(str).isin(
            list(_TIER5_EXCLUDED_GAME_TYPES))]
    if "gameday" not in sched.columns or "home_qb_id" not in sched.columns:
        return df
    sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")

    # ---- long per-(game, team) decided frame with the ACTUAL starter ----
    parts = []
    for team_col, opp_col, qb_col, is_home in (
            ("home_team", "away_team", "home_qb_id", True),
            ("away_team", "home_team", "away_qb_id", False)):
        if qb_col not in sched.columns:
            continue
        sub = pd.DataFrame({
            "game_id": sched["game_id"], "season": sched["season"],
            "week": sched["week"], "gameday": sched["gameday"],
            "team": sched[team_col], "opponent": sched[opp_col],
            "is_home": is_home,
            "actual": sched[qb_col].astype(str).str.strip(),
            "kickoff_utc": sched.apply(_parse_kickoff_et, axis=1),
        })
        parts.append(sub)
    if not parts:
        return df
    long = pd.concat(parts, ignore_index=True)
    long = long[long["actual"].notna() & (long["actual"] != "")]
    long = long.sort_values(["team", "gameday", "game_id", "is_home"],
                            kind="mergesort").reset_index(drop=True)
    long["act_id"] = long["actual"]
    long["gid"] = long["game_id"]

    # ---- per-(game, passer) QB EPA/play from pbp (for the skill axis) ----
    game_epa: dict[tuple, float] = {}
    if pbp is not None and {"game_id", "passer_id", "qb_epa"}.issubset(
            pbp.columns):
        p = pbp.dropna(subset=["passer_id", "qb_epa"])
        if not p.empty:
            g = p.groupby(["game_id", "passer_id"])["qb_epa"].agg(
                ["sum", "count"]).reset_index()
            for r in g.itertuples(index=False):
                if r.count > 0:
                    game_epa[(r.game_id, r.passer_id)] = float(r.sum / r.count)
    long["epa_v"] = long.apply(
        lambda r: game_epa.get((r["gid"], r["act_id"]), np.nan), axis=1)

    # ---- expected-starter maps (published pre-game, both chart shapes) ----
    weekly_map = _dc_qb1_weekly(depth_weekly)
    snap_map = _dc_qb1_snapshots(depth_snapshots)
    long["exp_id"] = long.apply(
        lambda r: _expected_starter_for(r["team"], int(r["season"]),
                                        r["week"], r["kickoff_utc"],
                                        weekly_map, snap_map), axis=1)

    # ---- prior-season facts (whole past season -> strictly prior by
    # construction; needed for the skill fallback and primary-out) ----
    qb_season_epa: dict[tuple[str, int], tuple[float, int]] = {}
    epa_rows = long.dropna(subset=["epa_v"])
    if not epa_rows.empty:
        for (qb, season), sub in epa_rows.groupby(["act_id", "season"]):
            qb_season_epa[(str(qb), int(season))] = (float(sub["epa_v"].mean()),
                                                     int(len(sub)))
    primaries: dict[tuple[str, int], str] = {}
    st = long.groupby(["team", "season", "act_id"]).size().reset_index(
        name="n")
    last = long.sort_values("gameday").drop_duplicates(
        ["team", "season", "act_id"], keep="last")
    st = st.merge(last[["team", "season", "act_id", "gameday"]],
                  on=["team", "season", "act_id"])
    for (team, season), sub in st.groupby(["team", "season"]):
        best = sub.sort_values(["n", "gameday"], ascending=[False, False])
        primaries[(str(team).strip().upper(), int(season))] = best.iloc[0]["act_id"]

    # ---- one chronological pass: read prior state, then update -------
    target_ids = set(df["game_id"])
    prev_actual: dict[str, str | None] = {}
    total_starts: dict[str, dict[str, int]] = {}
    season_starts: dict[tuple[str, int], dict[str, int]] = {}
    season_epa: dict[tuple[str, int], dict[str, tuple[float, int]]] = {}
    skill: dict[tuple, float] = {}
    continuity: dict[tuple, float] = {}
    change: dict[tuple, float] = {}
    primary_out: dict[tuple, float] = {}

    for r in long.itertuples(index=False):
        team = str(r.team).strip().upper()
        season = int(r.season)
        key = (r.game_id, team)
        expected = r.exp_id
        actual = r.act_id
        is_target = r.game_id in target_ids

        if is_target and expected is not None:
            # continuity: prior starts of the EXPECTED starter with this team
            cont = total_starts.get(team, {}).get(expected, 0)
            continuity[key] = float(min(cont, QB1_CONTINUITY_CAP))
            # skill fallback chain
            sdict = season_starts.get((team, season), {})
            n_cur = sdict.get(expected, 0)
            if n_cur >= QB1_MIN_STARTS_CURRENT_SEASON:
                s = season_epa.get((team, season), {}).get(expected)
                skill[key] = float(s[0] / s[1]) if s and s[1] else np.nan
            else:
                prior = qb_season_epa.get((expected, season - 1))
                skill[key] = (float(prior[0]) if prior is not None
                              else QB1_REPLACEMENT_EPA_PLAY)
            # change vs the team's prior-game ACTUAL starter
            prev = prev_actual.get(team)
            change[key] = (0.0 if prev == expected
                           else (1.0 if prev is not None else np.nan))
            # primary out (prior-season starts leader != expected)
            prim = primaries.get((team, season - 1))
            primary_out[key] = (0.0 if prim == expected
                                else (1.0 if prim is not None else np.nan))

        # ---- update strictly-prior state with this row ----------------
        prev_actual[team] = actual
        tdict = total_starts.setdefault(team, {})
        tdict[actual] = tdict.get(actual, 0) + 1
        sdict = season_starts.setdefault((team, season), {})
        sdict[actual] = sdict.get(actual, 0) + 1
        epa_v = r.epa_v
        if pd.notna(epa_v):
            edict = season_epa.setdefault((team, season), {})
            s, n = edict.get(actual, (0.0, 0))
            edict[actual] = (s + float(epa_v), n + 1)

    # ---- home - away over the target games -----------------------------
    ladder = long[long["game_id"].isin(target_ids)].copy()
    gids = df["game_id"]
    for feat, store in (("qb1_skill_diff", skill),
                        ("qb1_continuity_diff", continuity),
                        ("qb1_change_diff", change),
                        ("qb1_primary_out_diff", primary_out)):
        ladder["t5v"] = ladder.apply(
            lambda r: store.get((r["game_id"], str(r["team"]).strip().upper()),
                                np.nan), axis=1)
        df[feat] = _home_minus_away(ladder, gids, "t5v")
        ladder = ladder.drop(columns=["t5v"])
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

    # ---- Tier-4 (v6) per-game rates + ewm (conditional; absent -> NaN) ----
    # GS (non-garbage-time) rates — net points from non-garbage scoring (own
    # minus allowed), yards/play and QB-EPA/play over non-garbage plays.
    if "n_plays_gs" in srt.columns and "total_yards_gs" in srt.columns:
        srt["ypp_gs_game"] = (srt["total_yards_gs"]
                               / srt["n_plays_gs"].replace(0, np.nan))
        srt["ewm_ypp_gs"] = _trailing_ewm(srt, "ypp_gs_game", EWM_HALFLIFE)
    else:
        srt["ewm_ypp_gs"] = np.nan
    if "qb_epa_n_gs" in srt.columns and "qb_epa_sum_gs" in srt.columns:
        srt["qb_epa_gs"] = (srt["qb_epa_sum_gs"]
                             / srt["qb_epa_n_gs"].replace(0, np.nan))
        srt["ewm_qb_epa_gs"] = _trailing_ewm(srt, "qb_epa_gs", EWM_HALFLIFE)
    else:
        srt["ewm_qb_epa_gs"] = np.nan
    if "pts_scored_gs" in srt.columns and "pts_allowed_gs" in srt.columns:
        srt["net_pts_gs_game"] = srt["pts_scored_gs"] - srt["pts_allowed_gs"]
        srt["ewm_net_pts_gs"] = _trailing_ewm(srt, "net_pts_gs_game", EWM_HALFLIFE)
    else:
        srt["ewm_net_pts_gs"] = np.nan
    # drive-level rates (value / distinct drives)
    if "yds_per_drive" in srt.columns:
        srt["ewm_yds_per_drive"] = _trailing_ewm(srt, "yds_per_drive", EWM_HALFLIFE)
    else:
        srt["ewm_yds_per_drive"] = np.nan
    if "epa_per_drive" in srt.columns:
        srt["ewm_epa_per_drive"] = _trailing_ewm(srt, "epa_per_drive", EWM_HALFLIFE)
    else:
        srt["ewm_epa_per_drive"] = np.nan
    if "qb_epa_per_drive" in srt.columns:
        srt["ewm_qb_epa_per_drive"] = _trailing_ewm(srt, "qb_epa_per_drive", EWM_HALFLIFE)
    else:
        srt["ewm_qb_epa_per_drive"] = np.nan
    # QB-conditional: the ANNOUNCED/RECORDED starter's plays only (schedule
    # qb_id). A pending game's OWN starter is unknown (nflverse posts no
    # expected starter pre-kickoff), but the trailing shift needs only PAST
    # games' recorded starters — so the slate rows are populated honestly
    # (never faked with an invented starter).
    if "qb_epa_n_start" in srt.columns and "qb_epa_sum_start" in srt.columns:
        srt["qb_epa_start"] = (srt["qb_epa_sum_start"]
                                / srt["qb_epa_n_start"].replace(0, np.nan))
        srt["ewm_qb_epa_starter"] = _trailing_ewm(srt, "qb_epa_start", EWM_HALFLIFE)
    else:
        srt["ewm_qb_epa_starter"] = np.nan

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

    # ---- Tier-4 opponent-adjusted axes (same schedule-strength pattern) ---
    # trailing value minus the trailing mean of the OPPONENT's same-axis value
    # entering each prior game.
    for base, out in (("ewm_qb_epa", "ewm_qb_epa_oppadj"),
                      ("ewm_net_pts", "ewm_net_pts_oppadj"),
                      ("ewm_ypp", "ewm_ypp_oppadj")):
        if base not in srt.columns:
            srt[out] = np.nan
            continue
        opp_col = f"opp_{base}"
        opp_v = srt[["game_id", "team", base]].rename(
            columns={"team": "opponent", base: opp_col})
        srt = srt.merge(opp_v, on=["game_id", "opponent"], how="left")
        srt[out] = srt[base] - _trailing_per_team(srt, opp_col, OPP_ADJ_WINDOW)
        srt = srt.drop(columns=[opp_col])
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
    # --- v5 (Tier-3) candidates: officials / roster (the market de-vig family was deleted — market-independence policy) ---------
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
    gameday, gametime, stadium, home_record, away_record. Market/odds
    columns (spread_line, total_line, moneylines) are dropped — the slate
    frame is market-free by policy.
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
    # market-independence policy: the slate frame carries ZERO market/odds
    # columns (spread_line, total_line, moneylines never leave ingestion).
    for _mcol in ("spread_line", "total_line", "home_moneyline",
                  "away_moneyline"):
        if _mcol in df.columns:
            df = df.drop(columns=_mcol)
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
    # --- v5 (Tier-3) candidates: officials / roster (the market de-vig family was deleted — market-independence policy) ---------
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
# RETIRED admission gate (2026-09-02, NFL↔MLB parity pass) — KEPT ONLY for
# ablation-harness back-compat (run_tier1_ablation / run_nfl_unified_confirm
# _ablation import run_feature_gate). Production (pull_and_build) emits the
# static served_pool_manifest() below; NOTHING in this section runs in the
# pipeline anymore.
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

    RETIRED FROM PRODUCTION 2026-09-02 (NFL↔MLB parity pass) — kept only for
    ablation-harness back-compat (run_tier1_ablation / run_nfl_unified_confirm
    _ablation). The pipeline emits served_pool_manifest() instead; nothing
    served is gated by this function anymore.

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
def served_pool_manifest() -> dict:
    """STATIC served-pool manifest (the retired admission gate's replacement).

    The Phase-2 admission gate (coverage / available_12h / univariate-AUC /
    corr-pair |r|>0.8 / auto-prune) was RETIRED 2026-09-02 by the NFL↔MLB
    parity pass: the served pool is FEATURE_COLUMNS BY DECLARATION and
    missingness is handled in-model at serve time (MLB lineup-actuals
    approach). This manifest is data-independent — every fold/pull serves the
    exact same 12-pool plus the ``is_home`` anchor.

    Keeps the legacy ``feature_admission.v1_features`` shape so
    ``nfl_moneyline.admitted_model_features`` (and every other consumer of
    ``nfl_feature_v1_*.json``) resolves the same list as before.
    """
    v1 = [f for f in FEATURE_COLUMNS if f != "is_home"]
    return {
        "v1_features": v1,
        "kept_home_anchor": "is_home" in FEATURE_COLUMNS,
        "served_pool_size": len(v1),
        "policy": ("STATIC MANIFEST — feature-admission gate retired 2026-09-02; "
                   "served pool = FEATURE_COLUMNS by declaration; "
                   "missingness handled in-model at serve time"),
        "features": [
            {"name": f, "served": f != "is_home",
             "source": CANONICAL_SOURCE.get(f)}
            for f in FEATURE_COLUMNS
        ],
        "auto_prune": None,
        "dropped": [],
        "reasons": {},
    }


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
    """Build features + write the STATIC served-pool manifest.

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

    # The Phase-2 feature-admission gate no longer runs in the pipeline
    # (retired 2026-09-02, NFL↔MLB parity pass): the served pool is a STATIC
    # manifest from FEATURE_COLUMNS — no data-dependent audit (coverage /
    # available_12h / univariate-AUC / corr-pair / auto-prune). The gate
    # function stays importable for the ablation harnesses only.
    result = served_pool_manifest()

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
                "holdout": "2025 is the moneyline model's sealed hold-out: it stays "
                           "OUT of the decided frame (unchanged by the retired "
                           "admission gate).",
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
    print("\n=== NFL served-pool manifest (static; admission gate retired 2026-09-02) ===")
    print(f"decided games scored: {len(feats)}")
    print("policy:", result.get("policy"))
    print("admitted features (v1_features):", result["v1_features"])
    print("kept home anchor:", result.get("kept_home_anchor"))
    print("features:")
    for e in result.get("features", []):
        src = e.get("source") or ""
        print(f"  {e['name']:24s} served={e['served']}  {src}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        description="Build NFL features + write the served-pool manifest "
                    "(admission gate retired; no model).")
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON record")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    pull_and_build(write_record=not args.no_record)
    return 0


if __name__ == "__main__":
    sys.exit(main())