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
- Weather: only provenance-complete Open-Meteo hourly rows whose selected
  timestamp is strictly before kickoff; forecast rows must also have been
  fetched before kickoff. Daily aggregates and raw schedule weather are barred.

Model-family representations (spec section 14):
  - linear view: difference features + the constant ``is_home`` anchor
  - tree view: differences + raw home/away side values
  - mlp view: linear columns, standard-scaled at fit time
All views are deterministic in ordering and dimensionality.
"""
from __future__ import annotations

import functools
import logging
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    from backend import config
except ImportError:
    import config

try:
    from backend import weather as weather_provider
except ImportError:
    import weather as weather_provider

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
    # Schedule gametime is ET. Keep the exact kickoff ordering when present;
    # date-only synthetic inputs fall back to UTC midnight for stable ordering.
    kickoff = (_kickoff_utc(games)
               if {"gameday", "gametime"} <= set(games.columns)
               else pd.Series(pd.NaT, index=games.index,
                              dtype="datetime64[ns, UTC]"))
    fallback_kickoff = pd.to_datetime(gd, errors="coerce", utc=True)
    event_time = kickoff.where(kickoff.notna(), fallback_kickoff)
    home = pd.DataFrame({
        "game_id": games["game_id"], "season": games["season"],
        "week": games["week"], "gameday": gd, "kickoff_utc": event_time,
        "team": games["home_team"], "opponent": games["away_team"],
        "is_home": True,
        "for": games["home_score"].astype(float),
        "against": games["away_score"].astype(float),
    })
    away = pd.DataFrame({
        "game_id": games["game_id"], "season": games["season"],
        "week": games["week"], "gameday": gd,
        "team": games["away_team"], "opponent": games["home_team"],
        "is_home": False, "kickoff_utc": event_time,
        "for": games["away_score"].astype(float),
        "against": games["home_score"].astype(float),
    })
    ev = pd.concat([home, away], ignore_index=True)
    ev["net_from_team"] = ev["for"] - ev["against"]
    ev["team_win"] = np.select(
        [ev["for"] > ev["against"], ev["for"] < ev["against"],
         ev["for"] == ev["against"]],
        [1.0, 0.0, 0.5], default=np.nan)
    return ev


# ---------------------------------------------------------------------------
# Elo — pre-game entering rating, updated only after a game settles
# ---------------------------------------------------------------------------
def _elo_apply(events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    K, prior, scale = config.ELO_K, config.ELO_PRIOR, config.ELO_SCALE
    time_col = "kickoff_utc" if "kickoff_utc" in events.columns else "gameday"
    ev = events.sort_values([time_col, "game_id", "is_home"]).reset_index(drop=True)
    rating: dict = {}
    entering: dict = {}
    for game_id, rows in ev.groupby("game_id", sort=False):
        a, b = list(rows.itertuples(index=False))[:2]
        ra, rb = rating.get(a.team, prior), rating.get(b.team, prior)
        entering[(game_id, a.team)] = ra
        entering[(game_id, b.team)] = rb
        # A scheduled/unsettled row has no result. Preserve its entering
        # rating, but never update team state with a fabricated tie/loss.
        if pd.isna(a.team_win) or pd.isna(b.team_win):
            continue
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


# Trailing-window spec for the pbp candidate-pool metrics: per-game metric ->
# the ladder suffixes it is served at. DECLARED ONCE in config
# (PBP_CANDIDATE_TRAILING_SPECS — the same declaration that names the RFE
# candidates) and pulled here: "ewm" = decaying halflife-2 mean
# (config.EWM_HALFLIFE, the contract's existing recency primitive); "roll" =
# 4-game mean (config.PBP_ROLL_WINDOW). Every derivation rides the shared
# _trailing_ewm / _trailing_per_team primitives, so the shift(1) leakage
# discipline is inherited, not reimplemented.
PBP_TRAILING_SPECS: dict[str, tuple[str, ...]] = {
    **config.PBP_CANDIDATE_TRAILING_SPECS,
    **config.PBP_OPP_ADJ_TRAILING_SPECS,
    **config.PS_CANDIDATE_TRAILING_SPECS,
    **config.NGS_CANDIDATE_TRAILING_SPECS,
    **config.FTN_CANDIDATE_TRAILING_SPECS,
    **config.SC_CANDIDATE_TRAILING_SPECS,
}

# metric -> the family prefix its served candidates carry (pbp_ / ps_ / ngs_)
_FAMILY_OF: dict[str, str] = {
    metric: family
    for family, specs in config.CANDIDATE_FAMILIES.items()
    for spec in specs.values()
    for metric in spec
}


# ---------------------------------------------------------------------------
# Venue facts (committed stadiums table)
# ---------------------------------------------------------------------------
VENUE_FILE = config.BACKEND_DIR / "nfl_stadiums.csv"
VENUE_SCHEMA = ["stadium", "facility", "teams", "lat", "lon", "altitude_ft",
                "roof", "tz", "source"]


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
            "roof": str(getattr(r, "roof", "") or "").strip().lower(),
            "tz": getattr(r, "tz", "") or "",
        }
    return out


def _is_dome_home(df: pd.DataFrame) -> np.ndarray:
    """1.0 when the home venue is roofed, 0.0 when open-air, NaN when unknown.

    The schedule's per-game ``roof`` is the primary evidence (``dome``/
    ``closed`` = roofed, ``outdoors``/``open`` = played with the roof open).
    Only when the schedule is silent does the committed venue classification
    decide, and a venue with no roof at all is the only case that can settle
    weather-adjacent questions: a retractable or fixed roof tells us nothing
    about the game-day state, so it stays unknown there.
    """
    sched = weather_provider.schedule_roof(df)
    venue = weather_provider.venue_roof(df)
    roofed = weather_provider.is_roofed(df)
    open_air = (sched.isin(["outdoors", "outdoor", "open"])
                | (sched.isna() & venue.eq(weather_provider.ROOF_OUTDOOR))
                ).fillna(False)
    return np.where(roofed, 1.0, np.where(open_air, 0.0, np.nan))


def _prior_home_stadiums(games: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Map each game/team to its most recent STRICTLY-prior home stadium.

    The venue timeline is schedule data, never the current team→stadium
    association in the venue table. A team's first home game in the supplied
    timeline has no prior venue and therefore receives NaN travel distance.
    Exact kickoff ordering is used when available so a same-time row can never
    become a "prior" venue.
    """
    required = {"game_id", "gameday", "gametime", "home_team", "away_team"}
    if not required <= set(games.columns):
        return {}
    rows = games[list(required)].copy()
    rows["_kickoff"] = _kickoff_utc(rows)
    rows = rows.dropna(subset=["_kickoff", "home_team", "away_team"])
    rows = rows.sort_values(["_kickoff", "game_id"])
    if "stadium" not in games.columns:
        return {}
    home_rows = games[["game_id", "gameday", "gametime", "home_team", "stadium"]].copy()
    home_rows["_kickoff"] = _kickoff_utc(home_rows)
    home_rows = home_rows.dropna(subset=["_kickoff", "home_team", "stadium"])
    home_rows = home_rows.sort_values(["home_team", "_kickoff", "game_id"])
    histories = {
        str(team): group[["_kickoff", "stadium"]]
        for team, group in home_rows.groupby("home_team", sort=False)
    }
    out: dict[tuple[str, str], str] = {}
    for _, game in rows.iterrows():
        for team_col in ("home_team", "away_team"):
            team = str(game[team_col])
            history = histories.get(team)
            if history is None:
                continue
            prior = history[history["_kickoff"] < game["_kickoff"]]
            if prior.empty:
                continue
            stadium = prior.iloc[-1]["stadium"]
            if isinstance(stadium, str) and stadium.strip():
                out[(str(game["game_id"]), team)] = stadium
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
# PBP per-game metric spec — the candidate-pool expansion (2026-09-22)
# ---------------------------------------------------------------------------
# One row per (game_id, posteam): every metric is a per-game sum/rate of THAT
# game only; the trailing shift downstream keeps them strictly-prior.
#
#   epa_play          mean EPA per play (offensive efficiency beyond yards)
#   qb_epa_dropback   mean QB EPA per dropback (pass_attempt + sack)
#   def_epa_play      mean EPA allowed per play on the team's defensive snaps
#                     (lower = stingier defense; the raw input to the
#                     epa_play opponent-adjustment below)
#   cpoe_play         mean completion-probability-over-expectation (dropbacks)
#   air_yards_att     mean air yards per pass attempt (passing depth)
#   yac_epa_att       mean YAC-as-EPA per pass attempt (separation value;
#                     nflreadpy publishes no raw yac column)
#   turnovers         interceptions + fumbles lost (giveaways)
#   takeaways         opponent turnovers on the team's defensive snaps
#   sack_rate         sacks / dropbacks (protection)
#   dropback_rate     dropbacks / plays (playcalling tendency)
#   third_down_rate   conversions / third-down outcomes (situational strength)
#   penalty_yards_pg  penalty yards (raw per-game volume, discipline)
#   penalties_pg      penalty count (raw per-game volume, discipline)
#   redzone_td_rate   TD drives / red-zone drives (finishing)
#   start_field_pos   mean starting yardline_100 on drives (field-position edge)
#   fg_accuracy       made FGs / attempted FGs (special teams)
#   shotgun_rate      shotgun snaps / plays (formation tendency)
#   no_huddle_rate    no-huddle snaps / plays (pace/formation tendency)
#   drives_pg         distinct drive count (possessions/pace)
#
# Passing-depth metrics need the widened PBP_NEEDS (pbp cache v3); on older
# caches / unavailable seasons the source column is absent and the metric
# degrades to NaN per the documented missing-value policy.


def _add_pbp_metrics(p: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Attach the per-game metric columns to the (game_id, posteam) rollup.

    ``p`` is the play frame (posteam non-null, ``game_id`` + ``yards_gained``
    present); ``g`` is its grouped total_yards/n_plays rollup. Every metric
    guards its own source columns — an absent source yields an all-NaN
    column, never a fabricated value.
    """

    def _num(name: str) -> pd.Series | None:
        if name not in p.columns:
            return None
        return pd.to_numeric(p[name], errors="coerce")

    def _group_mean(name: str, out: str) -> None:
        v = _num(name)
        if v is None:
            g[out] = np.nan
        else:
            g[out] = (v.groupby([p["game_id"], p["posteam"]])
                      .mean().reindex(
                          pd.MultiIndex.from_frame(g[["game_id", "posteam"]]))
                      .to_numpy())

    def _group_sum(name: str, out: str) -> None:
        v = _num(name)
        if v is None:
            g[out] = np.nan
        else:
            g[out] = (v.fillna(0.0).groupby([p["game_id"], p["posteam"]])
                      .sum().reindex(
                          pd.MultiIndex.from_frame(g[["game_id", "posteam"]]))
                      .to_numpy())

    _group_mean("epa", "epa_play")

    # Defensive efficiency: mean EPA ALLOWED per play from the DEFENSIVE snaps
    # (defteam perspective, same play rows). Lower = stingier defense; this is
    # the opponent-strength series the epa_play adjustment is denominated in.
    if "defteam" in p.columns and "epa" in p.columns:
        d_eff = p.dropna(subset=["defteam"])
        d_epa = pd.to_numeric(d_eff["epa"], errors="coerce")
        idx0 = pd.MultiIndex.from_frame(g[["game_id", "posteam"]])
        g["def_epa_play"] = (d_epa.groupby([d_eff["game_id"], d_eff["defteam"]])
                             .mean().reindex(idx0).to_numpy())
    else:
        g["def_epa_play"] = np.nan

    # QB EPA / cpoe / air yards / YAC live on the dropback population.
    pa = _num("pass_attempt")
    sack = _num("sack")
    if pa is not None and sack is not None:
        dropback = (pa.fillna(0.0) + sack.fillna(0.0))
        dropback = dropback.where(dropback > 0)
    else:
        dropback = None
    for src, out in (("qb_epa", "qb_epa_dropback"), ("cpoe", "cpoe_play")):
        v = _num(src)
        if dropback is None or v is None:
            g[out] = np.nan
        else:
            masked = v.where(dropback.notna())
            g[out] = (masked.groupby([p["game_id"], p["posteam"]])
                      .mean().reindex(
                          pd.MultiIndex.from_frame(g[["game_id", "posteam"]]))
                      .to_numpy())
    for src, out in (("air_yards", "air_yards_att"), ("yac_epa", "yac_epa_att")):
        v = _num(src)
        if pa is None or v is None:
            g[out] = np.nan
        else:
            masked = v.where(pa.fillna(0.0) > 0)
            g[out] = (masked.groupby([p["game_id"], p["posteam"]])
                      .mean().reindex(
                          pd.MultiIndex.from_frame(g[["game_id", "posteam"]]))
                      .to_numpy())

    # Turnovers: own giveaways from the possession frame; takeaways from the
    # DEFENSIVE snaps (same play rows, defteam perspective).
    for src, out in (("interception", "_int"), ("fumble_lost", "_fumble")):
        _group_sum(src, out)
    own_tos = (g["_int"].fillna(0.0) + g["_fumble"].fillna(0.0)
               if "_int" in g.columns else np.nan)
    g["turnovers"] = own_tos
    d = p if "defteam" in p.columns else None
    if d is not None:
        d = d.dropna(subset=["defteam"])
        d_int = (pd.to_numeric(d["interception"], errors="coerce").fillna(0.0)
                 if "interception" in d.columns else None)
        d_fum = (pd.to_numeric(d["fumble_lost"], errors="coerce").fillna(0.0)
                 if "fumble_lost" in d.columns else None)
        if d_int is not None and d_fum is not None:
            idx = pd.MultiIndex.from_frame(g[["game_id", "posteam"]])
            opp = d_int.add(d_fum, fill_value=0.0).groupby(
                [d["game_id"], d["defteam"]]).sum().reindex(idx)
            g["takeaways"] = opp.to_numpy()
        else:
            g["takeaways"] = np.nan
    else:
        g["takeaways"] = np.nan
    g.drop(columns=["_int", "_fumble"], errors="ignore", inplace=True)

    # Rates: numerator/denominator play populations per side.
    plays = p["yards_gained"].notna().astype(float)  # the n_plays population

    def _rate(num: pd.Series | None, den: pd.Series, out: str) -> None:
        if num is None:
            g[out] = np.nan
            return
        idx = pd.MultiIndex.from_frame(g[["game_id", "posteam"]])
        n = num.fillna(0.0).groupby([p["game_id"], p["posteam"]]).sum().reindex(idx)
        dd = den.fillna(0.0).groupby([p["game_id"], p["posteam"]]).sum().reindex(idx)
        g[out] = (n / dd.replace(0, np.nan)).to_numpy()

    _rate(sack, dropback, "sack_rate")
    _rate(dropback, plays, "dropback_rate")
    td_c = _num("third_down_converted")
    td_f = _num("third_down_failed")
    if td_c is not None and td_f is not None:
        _rate(td_c, td_c + td_f.fillna(0.0), "third_down_rate")
    else:
        g["third_down_rate"] = np.nan

    _group_sum("penalty_yards", "penalty_yards_pg")
    pen = _num("penalty")
    _rate(pen, plays, "penalties_pg")

    # Red zone: a drive enters when a possession snap starts inside the 20.
    # td_team guards the touchdown count (post-1999 attribution column); when
    # absent the TD numerator degrades to NaN and the rate stays NaN.
    yl = _num("yardline_100")
    dr = p["drive"] if "drive" in p.columns else None
    if yl is not None and dr is not None:
        rz_mask = (yl < 20) & yl.notna() & dr.notna()
        rz_drives = (p.loc[rz_mask].groupby(["game_id", "posteam"])["drive"]
                     .nunique())
        td_team = p["td_team"] if "td_team" in p.columns else None
        if td_team is not None:
            td_drives = (p.loc[p["td_team"].notna()]
                         .groupby(["game_id", "posteam"])["drive"].nunique())
        else:
            td_drives = None
        idx = pd.MultiIndex.from_frame(g[["game_id", "posteam"]])
        g["redzone_td_rate"] = (td_drives.reindex(idx)
                                / rz_drives.reindex(idx).replace(0, np.nan)
                                ).to_numpy() if td_drives is not None else np.nan
        # Starting field position: mean yardline_100 on possession snaps.
        _group_mean("yardline_100", "start_field_pos")
        # Drives per game: distinct drive ids on the possession snaps.
        g["drives_pg"] = (p.loc[dr.notna()].groupby(["game_id", "posteam"])["drive"]
                          .nunique().reindex(idx).to_numpy())
    else:
        g["redzone_td_rate"] = np.nan
        g["start_field_pos"] = np.nan
        g["drives_pg"] = np.nan

    # Field-goal accuracy from the result strings (made / attempted).
    if "field_goal_result" in p.columns:
        fgm = (p["field_goal_result"].eq("made")
               .groupby([p["game_id"], p["posteam"]]).sum())
        fga = (p["field_goal_result"].notna()
               .astype(float).groupby([p["game_id"], p["posteam"]]).sum())
        idx = pd.MultiIndex.from_frame(g[["game_id", "posteam"]])
        g["fg_accuracy"] = (fgm.reindex(idx)
                            / fga.reindex(idx).replace(0, np.nan)).to_numpy()
    else:
        g["fg_accuracy"] = np.nan

    # Formation/pace tendencies.
    shotgun = _num("shotgun")
    _rate(shotgun, plays, "shotgun_rate")
    nh = _num("no_huddle")
    _rate(nh, plays, "no_huddle_rate")

    # ------------------------------------------------------------------
    # Situational / platoon rollups (2026-09-23): two-minute tendency,
    # fourth-down aggression/volume, close-game run balance. Shares are
    # sums over a play mask (mean of a 0/1 mask); rows missing context
    # stay NaN and are skipped by the aggregation — never fabricated to 0.
    # ------------------------------------------------------------------
    qtr = _num("qtr")
    half_sec = _num("half_seconds_remaining")
    score_diff = _num("score_differential")
    down = _num("down")
    ydstogo = _num("ydstogo")
    g2g = _num("goal_to_go")
    pt = (p["play_type"].astype(str).str.lower()
          if "play_type" in p.columns else None)
    idx = pd.MultiIndex.from_frame(g[["game_id", "posteam"]])

    def _share(mask: pd.Series, out: str) -> None:
        """Group-mean of a 0/1 series with NaN context rows skipped."""
        v = mask.where(plays.notna())
        g[out] = (v.groupby([p["game_id"], p["posteam"]]).mean()
                  .reindex(idx).to_numpy())

    def _count(mask: pd.Series, out: str) -> None:
        """Group-sum of a 0/1 series (0 where context valid but absent)."""
        v = mask.where(plays.notna()).fillna(0.0)
        g[out] = (v.groupby([p["game_id"], p["posteam"]]).sum()
                  .reindex(idx).to_numpy())

    if qtr is not None and half_sec is not None and score_diff is not None:
        ctx_ok = qtr.notna() & half_sec.notna() & score_diff.notna()
        _share(((qtr.isin([2, 4])) & (half_sec <= 120) & (score_diff < 0))
               .where(ctx_ok), "two_min_trail_share")
    else:
        g["two_min_trail_share"] = np.nan

    if down is not None and ydstogo is not None and pt is not None:
        is_fourth = down.eq(4)
        short_fourth = is_fourth & ydstogo.le(2)
        went_for = (short_fourth & pt.isin(["pass", "run"])
                    ).where(down.notna() & ydstogo.notna())
        _share(went_for, "fourth_go_rate")
        _count(is_fourth, "fourth_downs_pg")
    else:
        g["fourth_go_rate"] = np.nan
        g["fourth_downs_pg"] = np.nan

    if score_diff is not None and g2g is not None and pt is not None:
        close = (score_diff.abs().le(8) & ~g2g.eq(1).fillna(False)
                 ).where(score_diff.notna() & g2g.notna())
        _share(close & pt.eq("run"), "close_run_rate")
    else:
        g["close_run_rate"] = np.nan
    return g


# ---------------------------------------------------------------------------
# PBP rollup — per (game_id, team) aggregates (functions of that game only)
# ---------------------------------------------------------------------------
PBP_AGG_COLS = ["game_id", "team", "total_yards", "n_plays", "elapsed_min",
                "epa_play", "qb_epa_dropback", "def_epa_play", "cpoe_play", "air_yards_att",
                "yac_epa_att", "turnovers", "takeaways", "sack_rate",
                "dropback_rate", "third_down_rate", "penalty_yards_pg",
                "penalties_pg", "redzone_td_rate", "start_field_pos",
                "fg_accuracy", "shotgun_rate", "no_huddle_rate", "drives_pg",
                "two_min_trail_share", "fourth_go_rate", "fourth_downs_pg",
                "close_run_rate"]


def pbp_team_agg(pbp: pd.DataFrame | None) -> pd.DataFrame:
    """Per-(game_id, posteam) play aggregates. Every column is a per-game
    sum/rate — the trailing shift downstream keeps them strictly-prior.
    Absent source columns degrade to NaN (never fabricated)."""
    cols = list(PBP_AGG_COLS)
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
    g = _add_pbp_metrics(p, g)
    return g.rename(columns={"posteam": "team"})[cols]


# ---------------------------------------------------------------------------
# FTN charting rollup — per (game_id, team) structure/tendency aggregates
# ---------------------------------------------------------------------------
FTN_AGG_COLS = ["game_id", "team", "def_box", "off_backfield", "motion_rate",
                "play_action_rate", "rpo_rate", "screen_rate"]


def ftn_team_agg(ftn: pd.DataFrame | None,
                 game_teams: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-(game_id, team) charting aggregates (functions of that game only;
    the trailing shift downstream keeps them strictly-prior).

    The charting payload is game-level (no posteam), so ``game_teams`` — the
    pbp rollup's (game_id, team) pairs — resolves each game's two offenses;
    every (game, team) row carries the game's charted means. Absent source
    or mapping degrades to an empty frame (metrics stay NaN downstream)."""
    cols = list(FTN_AGG_COLS)
    if (ftn is None or ftn.empty or game_teams is None or game_teams.empty
            or "nflverse_game_id" not in ftn.columns):
        return pd.DataFrame(columns=cols)
    p = ftn.rename(columns={"nflverse_game_id": "game_id"}).copy()
    if "game_id" not in p.columns:
        return pd.DataFrame(columns=cols)
    for src in ("n_defense_box", "n_offense_backfield", "is_motion",
                "is_play_action", "is_rpo", "is_screen_pass"):
        p[src] = (pd.to_numeric(p[src], errors="coerce")
                  if src in p.columns else np.nan)
    per_game = (p.dropna(subset=["game_id"])
                 .groupby("game_id", as_index=False)
                 .agg(def_box=("n_defense_box", "mean"),
                      off_backfield=("n_offense_backfield", "mean"),
                      motion_rate=("is_motion", "mean"),
                      play_action_rate=("is_play_action", "mean"),
                      rpo_rate=("is_rpo", "mean"),
                      screen_rate=("is_screen_pass", "mean")))
    out = per_game.merge(
        game_teams[["game_id", "team"]].drop_duplicates(),
        on="game_id", how="inner")
    return out[cols]


# ---------------------------------------------------------------------------
# Snap-count rollup — per (game_id, team) participation shares by position
# ---------------------------------------------------------------------------
SC_AGG_COLS = ["game_id", "team", "rb_snap_share", "te_snap_share",
               "te2_snap_share", "wr1_snap_share", "qb_snap_share",
               "db_snap_share", "dl_snap_share"]

_RB_POS = {"RB", "FB"}
_TE_POS = {"TE"}
_WR_POS = {"WR"}
_QB_POS = {"QB"}
_DB_POS = {"CB", "DB", "S", "FS", "SS", "NB"}
_DL_POS = {"DE", "DT", "NT", "DL", "EDGE"}


def snap_counts_team_agg(snaps: pd.DataFrame | None) -> pd.DataFrame:
    """Per-(game_id, team) snap-share aggregates (functions of that game
    only; the trailing shift downstream keeps them strictly-prior).

    Offense: RB/FB, TE, TE-beyond-the-most-used, most-used WR and QB snap
    shares of team offensive snaps. Defense: DB (sub-package rate) and DL
    snap shares of team defensive snaps. Absent source degrades to an empty
    frame; a group with zero snaps at a side yields NaN shares (never 0)."""
    cols = list(SC_AGG_COLS)
    if (snaps is None or snaps.empty
            or not {"game_id", "team", "position", "offense_snaps",
                    "defense_snaps"} <= set(snaps.columns)):
        return pd.DataFrame(columns=cols)
    p = snaps.copy()
    for c in ("offense_snaps", "defense_snaps"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["_pos"] = p["position"].astype(str).str.upper().str.strip()

    off = p[p["offense_snaps"].gt(0).fillna(False)]
    defn = p[p["defense_snaps"].gt(0).fillna(False)]

    def _side_shares(df: pd.DataFrame, snap_col: str,
                     shares: dict[str, set[str] | str]) -> pd.DataFrame:
        """shares: name -> position set, or 'te2' / 'wr1' special forms."""
        g = df.groupby(["game_id", "team"])[snap_col]
        tot = g.sum()
        parts = {}
        for name, positions in shares.items():
            if isinstance(positions, set):
                sub = df[df["_pos"].isin(positions)]
                s = sub.groupby(["game_id", "team"])[snap_col].sum()
                parts[name] = s.reindex(tot.index) / tot.where(tot.gt(0))
            elif positions == "te2":
                te = df[df["_pos"].isin({"TE"})]
                s = te.groupby(["game_id", "team"])[snap_col].sum().reindex(tot.index).fillna(0.0)
                m = te.groupby(["game_id", "team"])[snap_col].max().reindex(tot.index).fillna(0.0)
                parts[name] = (s - m) / tot.where(tot.gt(0))
            elif positions == "wr1":
                wr = df[df["_pos"].isin({"WR"})]
                m = wr.groupby(["game_id", "team"])[snap_col].max().reindex(tot.index)
                parts[name] = m / tot.where(tot.gt(0))
        frame = pd.concat(list(parts.values()), axis=1)
        frame.columns = list(parts.keys())
        return frame.reset_index()

    out = pd.DataFrame(columns=["game_id", "team"])
    if len(off):
        off_shares = _side_shares(
            off, "offense_snaps",
            {"rb_snap_share": _RB_POS, "te_snap_share": _TE_POS,
             "te2_snap_share": "te2", "wr1_snap_share": "wr1",
             "qb_snap_share": _QB_POS})
        out = off_shares if out.empty else out.merge(off_shares, on=["game_id", "team"], how="outer")
    if len(defn):
        def_shares = _side_shares(
            defn, "defense_snaps",
            {"db_snap_share": _DB_POS, "dl_snap_share": _DL_POS})
        out = def_shares if out.empty else out.merge(def_shares, on=["game_id", "team"], how="outer")
    if out.empty:
        return pd.DataFrame(columns=cols)
    return out[cols]


# ---------------------------------------------------------------------------
# The ladder — team state + trailing stats, one row per (game_id, team)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Opponent adjustment — production rescaled by the defense actually faced
# ---------------------------------------------------------------------------
# For each (base, defense) pair in config.PBP_OPP_ADJ_METRICS the per-game
# series becomes
#
#   <base>_opp_adj = base + (shrunk opponent prior def_−def_epa_play)
#
# where the opponent prior is the opponent's shift(1) halflife-EWM of the
# defensive metric shrunk toward the prior expanding league mean — producing
# against a good (negative-EPA-allowed) defense RAISES the adjusted value,
# producing against a bad one LOWERS it. Every input is strictly prior: the
# strength series is the ordinary shift(1) primitive, and the league mean is
# the expanding mean of the same shifted series (so game t's strength uses
# only games strictly before t). The adjusted series is then consumed by the
# normal PBP_TRAILING_SPECS machinery, so its candidates inherit the same
# shift(1) discipline a second time (shift-of-shift, still causal).

def _add_opp_adj_metrics(srt: pd.DataFrame) -> None:
    """Attach the per-game opponent-adjusted columns to the sorted ladder IN
    PLACE. Degenerates to all-NaN when either source metric is absent.

    Row (team T, opponent B, game g):
      adj = base_T(g) + (league_mean(g) − strength_B(g))
    where strength_B(g) is B's defensive quality entering game g — the
    shift(1) halflife-EWM of B's defensive metric on B's OWN timeline,
    shrunk toward the expanding prior league mean with weight
    n_B / (n_B + OPP_ADJ_SHRINKAGE), n_B = B's games strictly before g.
    league_mean(g) is the expanding mean over gameday of per-team shift(1)
    defensive values, so every input is strictly prior and game g's own
    performances never enter either side's adjustment."""
    for base, defense in config.PBP_OPP_ADJ_METRICS.items():
        out = f"{base}_opp_adj"
        if base not in srt.columns or "opponent" not in srt.columns:
            srt[out] = np.nan
            continue
        if defense not in srt.columns:
            srt[out] = np.nan
            continue
        d = pd.to_numeric(srt[defense], errors="coerce")
        if not d.notna().any():
            srt[out] = np.nan
            continue

        # Per-team strictly-prior defensive value (the shift(1) primitive).
        d_prior = d.groupby(srt["team"], sort=False).shift(1)

        # Expanding prior league mean keyed by gameday: at date G it averages
        # every team's PRIOR-game defensive value known by G.
        by_day = d_prior.groupby(srt["gameday"], sort=True).mean()
        league_mean_by_day = by_day.sort_index().expanding(min_periods=1).mean()
        league_mean = srt["gameday"].map(league_mean_by_day).to_numpy(dtype=float)

        # Opponent's prior quality entering THIS game: its own shift(1) EWM
        # value on its row for the same game (exact keying, no timeline
        # mixing) + its prior-games count for the shrinkage weight.
        prior = _trailing_ewm(srt, defense, config.EWM_HALFLIFE)
        row_key = pd.MultiIndex.from_frame(srt[["game_id", "team"]])
        strength_by_row = pd.Series(prior, index=row_key)
        n_prior_by_row = pd.Series(
            srt.groupby("team", sort=False).cumcount().to_numpy(dtype=float),
            index=row_key)
        opp_key = pd.MultiIndex.from_frame(srt[["game_id", "opponent"]])
        opp_prior = strength_by_row.reindex(opp_key).to_numpy(dtype=float)
        opp_n = n_prior_by_row.reindex(opp_key).to_numpy(dtype=float)

        shrink = float(config.OPP_ADJ_SHRINKAGE)
        w = opp_n / (opp_n + shrink)
        # An opponent with no prior games (or no defensive data yet) shrinks
        # fully to the league mean; NaN n (own games missing) degrades to NaN.
        opp_prior = pd.Series(opp_prior, index=srt.index).fillna(
            pd.Series(league_mean, index=srt.index))
        strength = w * opp_prior + (1.0 - w) * league_mean
        srt[out] = (pd.to_numeric(srt[base], errors="coerce")
                    + (league_mean - strength))


# ---------------------------------------------------------------------------
def team_stats_ladder(events: pd.DataFrame,
                      team_game_agg: pd.DataFrame | None = None,
                      extra_per_game: dict[str, pd.DataFrame | None] | None = None,
                      ) -> pd.DataFrame:
    """Per-(game_id, team) point-in-time state: elo_entering, form_pts,
    win_pct, rest_days, ypp, ewm_net_pts, ewm_ypp, pace_plays_min,
    short_rest + the candidate-pool trailing metrics (pbp incl. the
    opponent-adjusted EPA series, ps usage shares, ngs tracking efficiency)
    — every trailing value strictly-prior (asserted).

    ``extra_per_game`` maps a rollup name to its per-game frame; frames are
    merged on their natural keys ((game_id, team) or (season, week, team))
    and their metric columns ride the SAME trailing machinery."""
    ev = events.copy()
    if team_game_agg is not None and len(team_game_agg):
        agg = team_game_agg.rename(columns={"total_yards": "tot_yd",
                                            "n_plays": "npl"})
        agg["ypp_game"] = agg["tot_yd"] / agg["npl"].replace(0, np.nan)
        agg["pace_plays_min_game"] = agg["npl"] / agg["elapsed_min"].replace(0, np.nan)
        metric_cols = [c for c in PBP_AGG_COLS[3:]
                       if c in agg.columns and c != "elapsed_min"]
        ev = ev.merge(agg[["game_id", "team", "ypp_game", "pace_plays_min_game"]
                          + metric_cols],
                      on=["game_id", "team"], how="left")
    else:
        metric_cols = []
    # Skill / tracking per-game rollups (ps, ngs): natural-key merges; their
    # metric columns join the trailing loop below like any pbp metric.
    for frame in (extra_per_game or {}).values():
        if frame is None or not len(frame):
            continue
        if {"game_id", "team"} <= set(frame.columns):
            keys = ["game_id", "team"]
        elif {"season", "week", "team"} <= set(frame.columns):
            keys = ["season", "week", "team"]
        else:
            continue
        fr = frame.drop(columns=[c for c in frame.columns
                                 if c not in keys and c in ev.columns])
        ev = ev.merge(fr, on=keys, how="left")

    time_col = "kickoff_utc" if "kickoff_utc" in ev.columns else "gameday"
    srt = ev.sort_values(["team", time_col, "gameday", "game_id"]).reset_index(drop=True)

    # LEAKAGE GATE: kickoff time strictly increasing within each team. The
    # date-only fallback still rejects multiple same-day rows rather than
    # silently choosing a future observation as a prior.
    diffs = srt.groupby("team", sort=False)[time_col].diff()
    bad = srt.loc[(diffs.notna()) & (diffs <= pd.Timedelta(0))]
    if len(bad):
        raise AssertionError(
            f"team_stats_ladder: team kickoff order not strictly increasing "
            f"({len(bad)} rows) — trailing features could leak")

    srt["form_pts"] = _trailing_per_team(srt, "net_from_team", config.FORM_WINDOW)
    srt["win_pct"] = _trailing_per_team(srt, "team_win", config.WINPCT_WINDOW)
    # Rest is a within-season property.  A season opener has no in-season
    # predecessor; differencing the whole team history would turn the offseason
    # into a fake 150-260 day "rest" interval.
    rest_groups = ["team", "season"] if "season" in srt.columns else ["team"]
    srt["rest_days"] = (
        srt.groupby(rest_groups, sort=False, dropna=False)["gameday"]
           .diff().dt.days
    )
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

    # Opponent-adjusted per-game series (epa_play vs prior opponent-defense
    # quality), then the pbp candidate-pool trailing metrics (2026-09-22
    # expansion). Every game metric rides the SAME causal primitives as the
    # served features:
    #   ewm      — per-team EWM (halflife = EWM_HALFLIFE), shift(1)
    #   roll     — per-team rolling mean (window = PBP_ROLL_WINDOW), shift(1)
    #   roll_opp — per-team rolling mean over OPP_ADJ_WINDOW games, shift(1)
    # A metric lists the window(s) it is served at; absent source columns are
    # all-NaN per-game and degrade to all-NaN trailing (never fabricated).
    _add_opp_adj_metrics(srt)
    for metric, windows in PBP_TRAILING_SPECS.items():
        if metric not in srt.columns:
            for w in windows:
                srt[f"{metric}_{w}"] = np.nan
            continue
        for w in windows:
            if w == "ewm":
                srt[f"{metric}_{w}"] = _trailing_ewm(srt, metric,
                                                     config.EWM_HALFLIFE)
            elif w == "roll_opp":
                srt[f"{metric}_{w}"] = _trailing_per_team(
                    srt, metric, config.OPP_ADJ_WINDOW)
            else:
                srt[f"{metric}_{w}"] = _trailing_per_team(
                    srt, metric, config.PBP_ROLL_WINDOW)
    return srt


def _home_minus_away(ladder: pd.DataFrame, game_ids: pd.Index, col: str) -> np.ndarray:
    home = ladder[ladder["is_home"]].set_index("game_id")[col]
    away = ladder[~ladder["is_home"]].set_index("game_id")[col]
    return (home.reindex(game_ids) - away.reindex(game_ids)).to_numpy()


def _per_side(ladder: pd.DataFrame, game_ids: pd.Index, col: str) -> tuple[np.ndarray, np.ndarray]:
    home = ladder[ladder["is_home"]].set_index("game_id")[col]
    away = ladder[~ladder["is_home"]].set_index("game_id")[col]
    return (home.reindex(game_ids).to_numpy(), away.reindex(game_ids).to_numpy())


# ---------------------------------------------------------------------------
# Skill rollups — player stats and Next-Gen Stats
# ---------------------------------------------------------------------------
PS_AGG_COLS = ["game_id", "team", "rb_load_share", "wr1_target_share"]


def player_stats_team_agg(ps: pd.DataFrame | None) -> pd.DataFrame:
    """Per-(game_id, team) skill-position usage shares from weekly player
    stats. Every column is a per-game ratio of THAT game only; absent source
    columns degrade to NaN (never fabricated)."""
    cols = list(PS_AGG_COLS)
    if ps is None or "game_id" not in getattr(ps, "columns", []):
        return pd.DataFrame(columns=cols)
    p = ps.copy()
    need = ["game_id", "team", "position"]
    if any(c not in p.columns for c in need):
        return pd.DataFrame(columns=cols)

    def _num(name: str) -> pd.Series | None:
        if name not in p.columns:
            return None
        return pd.to_numeric(p[name], errors="coerce")

    carries = _num("carries")
    targets = _num("targets")
    pos = p["position"].astype("string")
    rb_carries = (carries.where(pos.isin(["RB", "FB"]))
                  if carries is not None else None)
    rush_att = carries
    wr_targets = (targets.where(pos.eq("WR")) if targets is not None else None)

    g = pd.MultiIndex.from_frame(p[["game_id", "team"]].drop_duplicates())
    out = pd.DataFrame(index=g).reset_index().rename_axis(None)
    idx = pd.MultiIndex.from_frame(out[["game_id", "team"]])

    def _ratio(num: pd.Series | None, den: pd.Series | None, name: str) -> None:
        if num is None or den is None:
            out[name] = np.nan
            return
        n = num.fillna(0.0).groupby([p["game_id"], p["team"]]).sum().reindex(idx)
        d = den.fillna(0.0).groupby([p["game_id"], p["team"]]).sum().reindex(idx)
        out[name] = (n / d.replace(0, np.nan)).to_numpy()

    _ratio(rb_carries, rush_att, "rb_load_share")
    if wr_targets is not None and targets is not None:
        wr1 = wr_targets.groupby([p["game_id"], p["team"]]).max().reindex(idx)
        tt = targets.fillna(0.0).groupby([p["game_id"], p["team"]]).sum().reindex(idx)
        out["wr1_target_share"] = (wr1 / tt.replace(0, np.nan)).to_numpy()
    else:
        out["wr1_target_share"] = np.nan
    return out[cols]


NGS_AGG_COLS = ["season", "week", "team", "cpoe", "rush_eff", "sep"]


def ngs_team_agg(ngs: pd.DataFrame | None) -> pd.DataFrame:
    """Per-(season, week, team) Next-Gen-Stats efficiency aggregates.

    Weekly NGS rows describe the COMPLETED game — joined to the week's games
    they become per-game metrics and the ordinary shift(1) trailing makes
    them strictly prior. Volume-weighted means (attempts / attempts /
    targets); week-0 aggregates never reach here (dropped at load)."""
    cols = list(NGS_AGG_COLS)
    if ngs is None or "season" not in getattr(ngs, "columns", []):
        return pd.DataFrame(columns=cols)
    p = ngs.copy()
    need = ["season", "week", "team_abbr"]
    if any(c not in p.columns for c in need):
        return pd.DataFrame(columns=cols)
    p = p.rename(columns={"team_abbr": "team"})
    p["week"] = pd.to_numeric(p["week"], errors="coerce")
    p = p[p["week"].notna() & (p["week"] > 0)]
    if p.empty:
        return pd.DataFrame(columns=cols)

    def _num(name: str) -> pd.Series | None:
        if name not in p.columns:
            return None
        return pd.to_numeric(p[name], errors="coerce")

    pos = p["player_position"].astype("string") if "player_position" in p.columns else None
    att = _num("attempts")
    rush_att = _num("rush_attempts")
    tgt = _num("targets")
    cpoe_v = _num("completion_percentage_above_expectation")
    rye_v = _num("rush_yards_over_expected_per_att")
    sep_v = _num("avg_separation")

    keys = ["season", "week", "team"]
    g = pd.MultiIndex.from_frame(p[keys].drop_duplicates())
    out = pd.DataFrame(index=g).reset_index().rename_axis(None)
    idx = pd.MultiIndex.from_frame(out[keys])

    def _wmean(value: pd.Series | None, weight: pd.Series | None,
               name: str, pos_mask: pd.Series | None = None) -> None:
        if value is None or pos_mask is None:
            out[name] = np.nan
            return
        v = value.where(pos_mask)
        if weight is not None:
            w = weight.where(pos_mask).fillna(0.0)
            num = (v * w).groupby([p["season"], p["week"], p["team"]]).sum().reindex(idx)
            den = w.groupby([p["season"], p["week"], p["team"]]).sum().reindex(idx)
            out[name] = (num / den.replace(0, np.nan)).to_numpy()
        else:
            out[name] = (v.groupby([p["season"], p["week"], p["team"]])
                         .mean().reindex(idx).to_numpy())

    qb_mask = pos.eq("QB") if pos is not None else None
    rb_mask = pos.isin(["RB", "FB"]) if pos is not None else None
    wr_mask = pos.isin(["WR", "TE"]) if pos is not None else None
    _wmean(cpoe_v, att, "cpoe", qb_mask)
    _wmean(rye_v, rush_att, "rush_eff", rb_mask)
    _wmean(sep_v, tgt, "sep", wr_mask)
    return out[cols]


def _kickoff_utc(games: pd.DataFrame) -> pd.Series:
    """Parse schedule kickoff timestamps; schedule ``gametime`` is ET."""
    out = pd.Series(pd.NaT, index=games.index, dtype="datetime64[ns, UTC]")
    if not {"gameday", "gametime"} <= set(games.columns):
        return out
    day = pd.to_datetime(games["gameday"], errors="coerce")
    clock = games["gametime"].astype("string").str.strip()
    parsed = pd.to_datetime(
        day.dt.strftime("%Y-%m-%d") + " " + clock.fillna(""),
        errors="coerce",
    )
    try:
        out = parsed.dt.tz_localize("America/New_York", ambiguous="NaT",
                                     nonexistent="NaT").dt.tz_convert("UTC")
    except (TypeError, ValueError):
        # A malformed/missing schedule time must not become a permissive
        # timestamp; leave the value unavailable instead.
        return pd.Series(pd.NaT, index=games.index, dtype="datetime64[ns, UTC]")
    return out


def _attach_static_team_facts(df: pd.DataFrame,
                              venue_timeline: pd.DataFrame | None = None,
                              weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach PIT-gated weather and structural venue facts."""
    df = df.copy()

    # Venue geometry (travel distance, altitude) and prime-time flag. Travel
    # uses each team's most recent PRIOR home venue, not a current team→venue
    # map: the latter would encode relocations/stadium names that were not
    # knowable at historical kickoff.
    facts = _venue_facts()
    stadium = df["stadium"] if "stadium" in df.columns else pd.Series(np.nan, index=df.index)
    prior_home = _prior_home_stadiums(venue_timeline if venue_timeline is not None else df)

    def _game_fact(name: str) -> np.ndarray:
        return stadium.map(lambda s: facts.get(s, {}).get(name, np.nan)).to_numpy()

    game_lat, game_lon = _game_fact("lat"), _game_fact("lon")
    home_lat = np.array([
        facts.get(prior_home.get((str(gid), str(team)), ""), {}).get("lat", np.nan)
        for gid, team in zip(df["game_id"], df["home_team"])
    ], dtype=float)
    home_lon = np.array([
        facts.get(prior_home.get((str(gid), str(team)), ""), {}).get("lon", np.nan)
        for gid, team in zip(df["game_id"], df["home_team"])
    ], dtype=float)
    away_lat = np.array([
        facts.get(prior_home.get((str(gid), str(team)), ""), {}).get("lat", np.nan)
        for gid, team in zip(df["game_id"], df["away_team"])
    ], dtype=float)
    away_lon = np.array([
        facts.get(prior_home.get((str(gid), str(team)), ""), {}).get("lon", np.nan)
        for gid, team in zip(df["game_id"], df["away_team"])
    ], dtype=float)
    df["travel_miles_diff"] = (_haversine_miles(home_lat, home_lon, game_lat, game_lon)
                               - _haversine_miles(away_lat, away_lon, game_lat, game_lon))
    # Per-side candidate levels (config.STATIC_SIDE_CANDIDATES): each team's
    # own PIT distance to the game venue.
    df["travel_miles_home"] = _haversine_miles(home_lat, home_lon, game_lat, game_lon)
    df["travel_miles_away"] = _haversine_miles(away_lat, away_lon, game_lat, game_lon)
    df["altitude_home"] = _game_fact("altitude_ft")

    gametime = (df["gametime"].astype(str) if "gametime" in df.columns
                else pd.Series("", index=df.index))
    hour = pd.to_numeric(gametime.str.split(":").str[0], errors="coerce")
    df["prime_time"] = np.where(hour >= config.PRIME_TIME_HOUR, 1.0,
                                np.where(hour.isna(), np.nan, 0.0))

    # Playing surface of the HOME venue: normalize the schedule's messy
    # surface strings to a turf flag. NaN when unlisted — never fabricated.
    surface = (df["surface"].astype(str).str.strip().str.lower()
               if "surface" in df.columns
               else pd.Series("", index=df.index))
    turf_surfaces = {"fieldturf", "matrixturf", "sportturf", "astroturf", "a_turf"}
    df["is_turf_home"] = np.where(surface.isin(turf_surfaces), 1.0,
                                  np.where(surface.eq("grass"), 0.0, np.nan))

    return _attach_weather(df, weather)


def _attach_weather(df: pd.DataFrame,
                    weather: pd.DataFrame | None) -> pd.DataFrame:
    """Attach only provenance-proven hourly weather strictly before kickoff.

    The accepted provider contract is intentionally narrow: an allowed
    Open-Meteo source, an exact schedule kickoff match, a selected weather
    timestamp strictly before that kickoff, and — for forecast rows — a fetch
    timestamp strictly before kickoff. Legacy daily tables, raw schedule
    temperature/wind, missing rows, and indoor/closed games remain NaN.
    """
    out = df.copy()
    for col in ("temp_f", "wind_mph", "is_precip", "is_snow"):
        out[col] = np.nan
    for col in ("pit_weather_source", "pit_weather_time_utc",
                "pit_weather_fetched_at_utc", "pit_weather_kickoff_utc"):
        out[col] = (pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
                    if col.endswith("_utc") else "")

    required = {
        "game_id", "stadium", "kickoff_utc", "weather_time_utc",
        "fetched_at_utc", "source", "temp_f", "wind_mph", "precip_in",
        "snow_in",
    }
    if (weather is None or weather.empty or out.empty
            or not required <= set(weather.columns)):
        return out

    records = weather[list(required)].copy()
    for col in ("kickoff_utc", "weather_time_utc", "fetched_at_utc"):
        records[col] = pd.to_datetime(records[col], errors="coerce", utc=True)
    records["source"] = records["source"].astype("string")
    records = records[
        records["source"].isin(weather_provider.PIT_WEATHER_SOURCES)
        & records["kickoff_utc"].notna()
        & records["weather_time_utc"].notna()
        & records["fetched_at_utc"].notna()
        & (records["weather_time_utc"] < records["kickoff_utc"])
    ].copy()
    observed = records["source"].isin([
        weather_provider.OPEN_METEO_ARCHIVE,
        weather_provider.OPEN_METEO_FORECAST_PAST,
    ])
    records = records[
        (~observed) | (records["fetched_at_utc"] >= records["weather_time_utc"])
    ].copy()
    if records.empty:
        return out

    # One unambiguous record per game. Conflicting duplicate IDs fail closed.
    records = records.drop_duplicates(list(required), keep="last")
    duplicate_ids = set(
        records.groupby("game_id", sort=False).size().loc[lambda s: s > 1].index
    )
    records = records[~records["game_id"].isin(duplicate_ids)]
    by_id = records.set_index("game_id", drop=False)
    game_kickoffs = _kickoff_utc(out)
    outdoor = weather_provider.outdoor_mask(out)

    def _numeric(value: object) -> float:
        number = pd.to_numeric(value, errors="coerce")
        return float(number) if pd.notna(number) else np.nan

    for idx, game in out.iterrows():
        game_id = str(game.get("game_id", ""))
        game_stadium = game.get("stadium")
        if (not game_id or game_id not in by_id.index
                or pd.isna(game_stadium) or not str(game_stadium).strip()
                or not bool(outdoor.loc[idx])):
            continue
        row = by_id.loc[game_id]
        if isinstance(row, pd.DataFrame):
            continue
        game_ko = game_kickoffs.loc[idx]
        row_ko = row["kickoff_utc"]
        weather_time = row["weather_time_utc"]
        fetched_at = row["fetched_at_utc"]
        if (pd.isna(game_ko) or pd.isna(row_ko) or row_ko != game_ko
                or pd.isna(weather_time) or weather_time >= game_ko
                or str(row["stadium"]) != str(game_stadium)):
            continue
        if (row["source"] == weather_provider.OPEN_METEO_FORECAST
                and (pd.isna(fetched_at) or fetched_at >= game_ko)):
            continue

        temp = _numeric(row["temp_f"])
        wind = _numeric(row["wind_mph"])
        precip = _numeric(row["precip_in"])
        snow = _numeric(row["snow_in"])
        out.at[idx, "temp_f"] = temp
        out.at[idx, "wind_mph"] = wind
        out.at[idx, "is_precip"] = (
            float(precip >= config.PRECIP_FLAG_IN) if pd.notna(precip) else np.nan
        )
        out.at[idx, "is_snow"] = (
            float(snow >= config.SNOW_FLAG_IN) if pd.notna(snow) else np.nan
        )
        out.at[idx, "pit_weather_source"] = str(row["source"])
        out.at[idx, "pit_weather_time_utc"] = weather_time
        out.at[idx, "pit_weather_fetched_at_utc"] = fetched_at
        out.at[idx, "pit_weather_kickoff_utc"] = row_ko
    return out


# ---------------------------------------------------------------------------
# pbp candidate-pool serving (shared by the decided and slate builders)
# ---------------------------------------------------------------------------
def _attach_pbp_candidate_features(df: pd.DataFrame,
                                   ladder: pd.DataFrame,
                                   gids: pd.Index) -> pd.DataFrame:
    """Serve every trailing candidate family onto a game frame IN PLACE.

    Every PBP_TRAILING_SPECS metric (pbp, the opponent-adjusted series, ps
    usage, ngs tracking) appears exactly once per representation — the
    home−away diff and the raw per-side levels — under its family prefix
    (config.CANDIDATE_FAMILIES; config.PBP_CANDIDATE_COLS is derived from the
    same specs, so the served names and the declared RFE candidates stay
    name-for-name identical). Columns the ladder could not derive (absent
    source data / old caches) are served as all-NaN, matching the documented
    missing-value policy.

    Returns the frame with all candidate columns joined in ONE concat
    (pandas fragmentation otherwise degrades every later operation on this
    frame — even ``df[list] = frame`` inserts column-by-column internally);
    per metric the ladder is joined once for the per-side values, and the
    diff is the home−away difference of those sides (identical to a
    separate join).
    """
    if not PBP_TRAILING_SPECS:
        return df
    sides: dict[str, np.ndarray] = {}
    for metric, windows in PBP_TRAILING_SPECS.items():
        family = _FAMILY_OF.get(metric, "pbp")
        for w in windows:
            col = f"{metric}_{w}"
            home_v, away_v = _per_side(ladder, gids, col)
            sides[f"{family}_{col}_diff"] = home_v - away_v
            sides[f"{family}_{col}_home"] = home_v
            sides[f"{family}_{col}_away"] = away_v
    return pd.concat([df, pd.DataFrame(sides, index=df.index)], axis=1)


def _record_frame(events: pd.DataFrame) -> pd.DataFrame:
    """Return each event team's record entering that event's kickoff."""
    cols = ["game_id", "team", "prior_wins", "prior_losses", "prior_ties", "record"]
    if events.empty:
        return pd.DataFrame(columns=cols)
    time_col = "kickoff_utc" if "kickoff_utc" in events.columns else "gameday"
    srt = events.sort_values(["team", time_col, "gameday", "game_id"]).copy()
    for name, value in (("wins", 1.0), ("losses", 0.0), ("ties", 0.5)):
        flag = (srt["team_win"] == value).astype(float)
        cumulative = flag.groupby(srt["team"], sort=False).cumsum()
        srt[f"prior_{name}"] = cumulative - flag
    srt["record"] = [
        (f"{int(w)}-{int(l)}" if not t else f"{int(w)}-{int(l)}-{int(t)}")
        for w, l, t in zip(srt["prior_wins"], srt["prior_losses"],
                          srt["prior_ties"])
    ]
    return srt[cols].reset_index(drop=True)


def _attach_record_fields(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach PIT (entering) records; never use a season-final cumulative row."""
    out = df.copy()
    records = _record_frame(events)
    if records.empty:
        for side in ("home", "away"):
            out[f"{side}_record"] = ""
            out[f"{side}_wins"] = np.nan
            out[f"{side}_losses"] = np.nan
        return out
    records["game_id"] = records["game_id"].astype(str)
    records["team"] = records["team"].astype(str)
    key = records.set_index(["game_id", "team"])
    for side in ("home", "away"):
        team_col = f"{side}_team"
        idx = pd.MultiIndex.from_arrays([
            out["game_id"].astype(str), out[team_col].astype(str)
        ])
        rows = key.reindex(idx)
        out[f"{side}_record"] = rows["record"].fillna("").to_numpy()
        out[f"{side}_wins"] = rows["prior_wins"].to_numpy(dtype=float)
        out[f"{side}_losses"] = rows["prior_losses"].to_numpy(dtype=float)
    return out


# ---------------------------------------------------------------------------
# Player quality — per-position shrunk EPA
#
# MLB's shrunk_woba + lineup_agg structural analogue (MLB features.py):
# retain lagged player ratings and remove players explicitly designated
# Out/IR/Doubtful before selecting the projected offensive pool. EPA and
# opportunity history is constructed independently of current-game status.
# The per-position shrunk ratings are aggregated to team features, with raw
# home/away inputs routed tree-only and home-away diffs shared by all members.
# k is 20% of the position's pooled prior median rolling-8 opportunities.

# ---------------------------------------------------------------------------
EPA_QB_POSITIONS = ("QB", "WR", "TE", "RB")
EPA_QUALITY_WINDOW = 8
EPA_QUALITY_K_FRAC = 0.20
# NFL bye weeks can leave 14–17 days between a team's games; keep the same
# widened-candidate-roster pattern as MLB while allowing a three-week roster
# freshness window so a bye does not erase every projected-lineup candidate.
EPA_QUALITY_POOL_DAYS = 21
EPA_QUALITY_TOP_N = 11


# Fullbacks are grouped with running backs, consistent with the existing
# player usage and snap-count aggregates.
def _normalize_epa_lineup_positions(positions: pd.Series) -> pd.Series:
    return positions.astype("string").str.upper().str.strip().replace({"FB": "RB"})


# nflverse opportunity flags, summed per player-game.  These replace a
# play_type derivation because they handle the cases play_type gets wrong:
# a scramble is a qb_dropback on a play_type of "run", and pass_attempt is 0
# on a play a defensive penalty nullified.  Measured league-wide 2025, a
# player-game with EPA but all three flags at zero does not exist (0 of
# 5,598), so the denominator has no structural zero. The flag names live in
# EPA_ROLE_COLS, which is the only thing that reads them.
EPA_ROLE_COLS = (("passer_player_id", "qb_dropback"),
                 ("receiver_player_id", "pass_attempt"),
                 ("rusher_player_id", "rush_attempt"))

EPA_QUALITY_METRICS = tuple(f"epa_{p.lower()}" for p in EPA_QB_POSITIONS)
EPA_QUALITY_AGG_COLS = ["game_id", "team", "position", "epa_q"]


def epa_opportunity_table(pbp: pd.DataFrame | None) -> pd.DataFrame:
    """Per-(game_id, team, player) EPA and role-specific opportunity count.

    Each player is matched to the opportunity flag for his play role (passer
    -> qb_dropback, receiver -> pass_attempt, rusher -> rush_attempt). If one
    player is credited under multiple roles on a single play, the play is
    counted once, with the maximum applicable role flag, for both EPA and
    opportunity totals.
    """
    cols = ["game_id", "team", "player_id", "epa", "opp"]
    if pbp is None or "epa" not in getattr(pbp, "columns", []):
        return pd.DataFrame(columns=cols)
    need = ["game_id", "posteam", "epa", "play_id"] + \
        [c for pair in EPA_ROLE_COLS for c in pair]
    if any(c not in pbp.columns for c in need):
        return pd.DataFrame(columns=cols)
    p = pbp.copy()
    p["epa"] = pd.to_numeric(p["epa"], errors="coerce")
    p = p[p["epa"].notna() & p["posteam"].notna()]

    long: list[pd.DataFrame] = []
    for role_col, flag in EPA_ROLE_COLS:
        s = p[["game_id", "play_id", "posteam", "epa", role_col, flag]].rename(
            columns={role_col: "player_id", flag: "opp"})
        s["player_id"] = s["player_id"].astype("string")
        s = s[s["player_id"].notna()
              & ~s["player_id"].isin(["", "nan", "None"])]
        long.append(s)
    if not long:
        return pd.DataFrame(columns=cols)
    L = pd.concat(long, ignore_index=True)
    L["opp"] = pd.to_numeric(L["opp"], errors="coerce").fillna(0.0)
    # A player's opportunity is the flag for the role that identified him
    # (dropback for passer, pass attempt for receiver, rush attempt for rusher).
    # Deduplicate same-player/same-play dual-role rows, counting EPA once and
    # one play opportunity rather than adding unrelated PBP flags together.
    per_play = (L.groupby(["game_id", "posteam", "player_id", "play_id"],
                          as_index=False)
                .agg(epa=("epa", "first"), opp=("opp", "max")))
    out = (per_play.groupby(["game_id", "posteam", "player_id"], as_index=False)
           .agg(epa=("epa", "sum"), opp=("opp", "sum"))
           .rename(columns={"posteam": "team", "epa": "epa"}))
    return out[cols]


def epa_quality_ratings(obs: pd.DataFrame) -> pd.DataFrame:
    """Post-game rolling player totals used to rate the next game.

    ``_num`` / ``_den`` are inclusive through this player's most recent
    observed game. The team aggregator admits rows only from an earlier
    calendar date than the target, so neither the target result nor another
    game still in progress that date can enter the rating.
    """
    empty_cols = list(obs.columns) + ["_num", "_den", "_vol"]
    if obs.empty:
        return obs.assign(_num=np.nan, _den=np.nan, _vol=np.nan)
    d = obs.copy()
    required = {"game_id", "team", "player_id", "position", "gameday",
                "kickoff_utc", "epa", "opp"}
    if not required.issubset(d.columns):
        return pd.DataFrame(columns=empty_cols)
    d["game_id"] = d["game_id"].astype(str)
    d["player_id"] = d["player_id"].astype("string").str.strip()
    d["gameday"] = pd.to_datetime(d["gameday"], errors="coerce").dt.normalize()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], errors="coerce", utc=True)
    d["position"] = _normalize_epa_lineup_positions(d["position"])
    d["epa"] = pd.to_numeric(d["epa"], errors="coerce")
    d["opp"] = pd.to_numeric(d["opp"], errors="coerce")
    d = d.dropna(subset=["player_id", "gameday", "kickoff_utc", "position",
                         "epa", "opp"])
    d = d[d["position"].isin(EPA_QB_POSITIONS)]
    d = d[~d["player_id"].isin(["", "nan", "none", "<na>", "null"])]
    if d.empty:
        return d.assign(_num=np.nan, _den=np.nan, _vol=np.nan)
    d = d.sort_values(["player_id", "gameday", "kickoff_utc", "game_id"])
    # Exactly one observation per player-game, in exact chronological order.
    d = d.drop_duplicates(["game_id", "player_id"], keep="last")
    g = d.groupby("player_id", sort=False)
    # The current row represents the completed historical game; target
    # aggregation chooses only rows with gameday < target gameday.
    d["_num"] = g["epa"].transform(
        lambda s: s.rolling(EPA_QUALITY_WINDOW, min_periods=1).sum())
    d["_den"] = g["opp"].transform(
        lambda s: s.rolling(EPA_QUALITY_WINDOW, min_periods=1).sum())
    d["_vol"] = d["_den"]
    return d


def _position_priors_asof(history: pd.DataFrame,
                          target_days: pd.Series) -> pd.DataFrame:
    """Position mu and k known before each target calendar date.

    ``mu`` is a pooled EPA/opportunity ratio over all earlier dates.
    ``k`` is 20% of the median rolling-8 opportunity total across all prior
    player-game rows. Same-day rows are excluded, avoiding ordering leaks for
    games that are still in progress when a same-date target kicks off.
    """
    targets = pd.Series(pd.to_datetime(target_days, errors="coerce")).dropna()
    targets = targets.dt.normalize().drop_duplicates().sort_values()
    if targets.empty:
        return pd.DataFrame(columns=["target_day", "position", "mu", "k"])
    target_values = targets.to_numpy(dtype="datetime64[ns]")
    rows: list[dict] = []
    for pos in EPA_QB_POSITIONS:
        p = (history.loc[history["position"].eq(pos),
                         ["game_id", "gameday", "epa", "opp", "_den"]]
             .dropna(subset=["gameday", "epa", "opp", "_den"])
             .sort_values(["gameday", "game_id"]))
        if p.empty:
            rows.extend({"target_day": day, "position": pos,
                         "mu": np.nan, "k": np.nan} for day in targets)
            continue

        daily = (p.groupby("gameday", sort=True)
                 .agg(day_epa=("epa", "sum"), day_opp=("opp", "sum")))
        daily["prior_epa"] = daily["day_epa"].cumsum()
        daily["prior_opp"] = daily["day_opp"].cumsum()
        dates = daily.index.to_numpy(dtype="datetime64[ns]")
        idx = np.searchsorted(dates, target_values, side="left") - 1
        mu = np.full(len(target_values), np.nan, dtype=float)
        valid = idx >= 0
        if valid.any():
            num = daily["prior_epa"].to_numpy(dtype=float)[idx[valid]]
            den = daily["prior_opp"].to_numpy(dtype=float)[idx[valid]]
            mu[valid] = np.divide(num, den, out=np.full(len(num), np.nan),
                                  where=den > 0)

        # Expanding medians across historical player-games are computed in
        # chronological order. Keep the last value of each date (so all
        # player-games on that date are included together), then as-of join
        # strictly before the target date.
        p["expanding_den_median"] = p["_den"].expanding(min_periods=1).median()
        med_by_day = p.groupby("gameday", sort=True)["expanding_den_median"].last()
        med_dates = med_by_day.index.to_numpy(dtype="datetime64[ns]")
        med_idx = np.searchsorted(med_dates, target_values, side="left") - 1
        k = np.full(len(target_values), np.nan, dtype=float)
        med_valid = med_idx >= 0
        if med_valid.any():
            k[med_valid] = (EPA_QUALITY_K_FRAC
                            * med_by_day.to_numpy(dtype=float)[med_idx[med_valid]])
        rows.extend({"target_day": pd.Timestamp(day), "position": pos,
                     "mu": float(m), "k": float(kv)}
                    for day, m, kv in zip(targets, mu, k))
    return pd.DataFrame(rows, columns=["target_day", "position", "mu", "k"])


def _strict_pit_timestamp(value):
    """Normalize explicitly zoned PIT timestamps to UTC; reject naive values."""
    if value is None or pd.isna(value):
        return pd.NaT
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return pd.NaT
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        return pd.NaT
    return stamp.tz_convert("UTC")


def epa_quality_team_agg(history: pd.DataFrame, games: pd.DataFrame,
                         injuries: pd.DataFrame | None = None
                         ) -> pd.DataFrame:
    """Per-(game, team, position) mean of PIT-shrunk EPA player ratings.

    Candidate pool: each team's latest player EPA rating from a game date
    strictly before the target date and within the 21-calendar-day window,
    covering ordinary bye-week gaps.
    Only a strictly pre-kickoff Out/IR/Doubtful designation excludes the
    player; other statuses and no admissible report leave him eligible. The
    rolling player rating is computed before current-game membership filtering,
    so a current injury never erases prior-game EPA. Eligible player ratings
    are averaged unweighted by position (no absent-player padding). This
    mirrors MLB's lagged player ratings -> candidate roster -> IL membership
    filter -> team aggregate structure, with NFL positional outputs.
    """
    cols = list(EPA_QUALITY_AGG_COLS)
    if history.empty or games is None or games.empty:
        return pd.DataFrame(columns=cols)
    needed_history = {"game_id", "team", "player_id", "position", "gameday",
                      "kickoff_utc", "epa", "opp", "_num", "_den"}
    needed_games = {"game_id", "gameday", "gametime", "home_team", "away_team"}
    if (not needed_history.issubset(history.columns)
            or not needed_games.issubset(games.columns)):
        return pd.DataFrame(columns=cols)

    d = history.copy()
    d["game_id"] = d["game_id"].astype(str)
    d["player_id"] = d["player_id"].astype("string").str.strip()
    d["team"] = d["team"].astype("string").str.strip().str.upper()
    d["gameday"] = pd.to_datetime(d["gameday"], errors="coerce").dt.normalize()
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], errors="coerce", utc=True)
    d["position"] = _normalize_epa_lineup_positions(d["position"])
    d = d.dropna(subset=["gameday", "kickoff_utc", "player_id", "position",
                         "_num", "_den"])
    if d.empty:
        return pd.DataFrame(columns=cols)

    tgt = games[["game_id", "gameday", "gametime", "home_team", "away_team"]].copy()
    tgt["game_id"] = tgt["game_id"].astype(str)
    tgt["home_team"] = tgt["home_team"].astype("string").str.strip().str.upper()
    tgt["away_team"] = tgt["away_team"].astype("string").str.strip().str.upper()
    tgt["gameday"] = pd.to_datetime(tgt["gameday"], errors="coerce").dt.normalize()
    tgt["kickoff_utc"] = _kickoff_utc(tgt)
    tgt = tgt.dropna(subset=["gameday", "kickoff_utc"])
    if tgt.empty:
        return pd.DataFrame(columns=cols)
    tgt = pd.concat([
        tgt[["game_id", "gameday", "kickoff_utc", "home_team"]]
        .rename(columns={"home_team": "team"}),
        tgt[["game_id", "gameday", "kickoff_utc", "away_team"]]
        .rename(columns={"away_team": "team"}),
    ], ignore_index=True).drop_duplicates(["game_id", "team"])
    tgt = tgt.rename(columns={"gameday": "target_day"})

    pool = (d[["game_id", "team", "player_id", "position", "gameday",
               "_num", "_den"]]
            .rename(columns={"game_id": "history_game_id",
                             "gameday": "rating_day"}))
    j = tgt.merge(pool, on="team", how="inner")
    # Only prior calendar dates qualify. PBP/player stats do not supply an
    # authoritative end-of-game publication time, so a same-day prior kickoff
    # is not proof that the completed EPA rating was known before target kickoff.
    j = j[(j["rating_day"] < j["target_day"])
          & (j["rating_day"] >= j["target_day"]
             - pd.Timedelta(days=EPA_QUALITY_POOL_DAYS))]
    j = (j.sort_values(["game_id", "team", "player_id", "rating_day",
                        "history_game_id"])
         .drop_duplicates(["game_id", "team", "player_id"], keep="last"))

    if injuries is not None and not injuries.empty:
        pit_cols = {"game_id", "team", "player_id", "availability_weight",
                    "published"}
        if not pit_cols.issubset(injuries.columns):
            raise ValueError("injuries must come from load_injuries_pit and carry "
                             "game_id, team, player_id, availability_weight, published")
        pit = injuries[["game_id", "team", "player_id", "availability_weight",
                        "published"]].copy()
        pit["game_id"] = pit["game_id"].astype(str)
        pit["team"] = pit["team"].astype("string").str.strip().str.upper()
        pit["player_id"] = pit["player_id"].astype("string").str.strip()
        pit = pit.dropna(subset=["team", "player_id"])
        pit = pit[pit["team"].ne("") & pit["player_id"].ne("")]
        pit["availability_weight"] = pd.to_numeric(
            pit["availability_weight"], errors="coerce")
        pit["published"] = pd.to_datetime(
            pit["published"].map(_strict_pit_timestamp), errors="coerce", utc=True)
        target_kickoff = tgt[["game_id", "kickoff_utc"]].drop_duplicates("game_id")
        pit = pit.merge(target_kickoff, on="game_id", how="inner")
        pit = pit[pit["published"] < pit["kickoff_utc"]]
        # For conflicting rows at an identical latest timestamp, an explicit
        # injury designation (weight 0) wins; latest admissible PIT report wins
        # otherwise. The team key prevents another matchup's report bleeding
        # onto this player's target-team lineup.
        pit = (pit.sort_values(["game_id", "team", "player_id", "published",
                                "availability_weight"],
                               ascending=[True, True, True, True, False],
                               kind="mergesort")
               .drop_duplicates(["game_id", "team", "player_id"], keep="last")
               [["game_id", "team", "player_id", "availability_weight"]])
        j = j.merge(pit, on=["game_id", "team", "player_id"], how="left")
        # Missing reports/timestamps and every status other than the explicit
        # Out/IR/Doubtful classifier value leave the player in the candidate
        # pool. The historical rolling EPA/_den was computed before this join.
        j["availability_weight"] = pd.to_numeric(
            j["availability_weight"], errors="coerce")
        j = j[~j["availability_weight"].eq(0.0)]

    if j.empty:
        return pd.DataFrame(columns=cols)
    # Mirror MLB's candidate-roster -> injury filter -> lineup-rank structure:
    # after exclusions, select the team's top 11 eligible offensive players by
    # prior-window opportunities, then aggregate their ratings by position.
    # Player ID breaks workload ties deterministically.
    j = (j.sort_values(["game_id", "team", "_den", "player_id"],
                       ascending=[True, True, False, True],
                       kind="mergesort")
         .assign(rn=lambda x: x.groupby(["game_id", "team"]).cumcount() + 1))
    top = j[j["rn"] <= EPA_QUALITY_TOP_N].copy()
    if top.empty:
        return pd.DataFrame(columns=cols)

    priors = _position_priors_asof(d, tgt["target_day"])
    top = top.merge(priors, on=["target_day", "position"], how="left")
    denom = top["_den"] + top["k"]
    ok = (top["mu"].notna() & top["k"].notna() & denom.gt(0))
    top["epa_q"] = np.where(
        ok, (top["_num"] + top["mu"] * top["k"]) / denom, np.nan)
    out = (top.groupby(["game_id", "team", "position"], as_index=False)["epa_q"]
           .mean())
    return out[cols]


def _epa_quality_agg(games: pd.DataFrame, pbp: pd.DataFrame | None,
                     ps: pd.DataFrame | None,
                     injuries: pd.DataFrame | None) -> pd.DataFrame:
    """Build PIT per-position player quality aggregates, degrading to NaN
    when play-by-play, player IDs/positions, or exact schedule kickoff is
    unavailable.
    """
    empty = pd.DataFrame(columns=EPA_QUALITY_AGG_COLS)
    obs = epa_opportunity_table(pbp)
    if (obs.empty or ps is None
            or not {"game_id", "team", "player_id", "position"} <= set(ps.columns)
            or not {"game_id", "gameday", "gametime"} <= set(games.columns)):
        return empty

    days = games[["game_id", "gameday", "gametime"]].copy()
    days["game_id"] = days["game_id"].astype(str)
    days["gameday"] = pd.to_datetime(days["gameday"], errors="coerce").dt.normalize()
    days["kickoff_utc"] = _kickoff_utc(days)
    obs["game_id"] = obs["game_id"].astype(str)
    obs["player_id"] = obs["player_id"].astype("string").str.strip()
    obs["team"] = obs["team"].astype("string").str.strip()
    obs = obs.merge(days, on="game_id", how="inner", validate="many_to_one")
    pos = ps[["game_id", "team", "player_id", "position"]].copy()
    pos["game_id"] = pos["game_id"].astype(str)
    pos["team"] = pos["team"].astype("string").str.strip()
    pos["player_id"] = pos["player_id"].astype("string").str.strip()
    pos["position"] = _normalize_epa_lineup_positions(pos["position"])
    pos = pos.dropna(subset=["player_id", "position"])
    pos = pos[~pos["player_id"].isin(["", "nan", "none", "<na>", "null"])]
    pos = pos.drop_duplicates(["game_id", "team", "player_id"], keep="last")
    obs = obs.merge(pos, on=["game_id", "team", "player_id"],
                    how="inner", validate="one_to_one")
    obs = obs.dropna(subset=["gameday", "kickoff_utc"])
    history = epa_quality_ratings(obs)
    return epa_quality_team_agg(history, games, injuries)


def _attach_epa_quality_features(df: pd.DataFrame, agg: pd.DataFrame,
                                 games: pd.DataFrame) -> pd.DataFrame:
    """Serve the per-position quality columns in ONE concat (home/away/diff).

    The team lookup joins against the SCHEDULE, not df: df is the served game
    frame and is not contracted to carry home_team/away_team, so reading them
    from there would make this family depend on an incidental column.
    """
    n = len(df)
    sides: dict[str, np.ndarray] = {}
    sched_ok = (games is not None
                and {"game_id", "home_team", "away_team"}.issubset(
                    set(games.columns)))
    if agg is None or agg.empty or not sched_ok:
        for metric in EPA_QUALITY_METRICS:
            for rep in ("home", "away", "diff"):
                sides[f"{metric}_{rep}"] = np.full(n, np.nan)
        return pd.concat([df, pd.DataFrame(sides, index=df.index)], axis=1)

    gids = df["game_id"].astype(str)
    sched = (games[["game_id", "home_team", "away_team"]].copy()
             .assign(game_id=lambda x: x["game_id"].astype(str))
             .drop_duplicates("game_id").set_index("game_id"))
    for metric, pos in zip(EPA_QUALITY_METRICS, EPA_QB_POSITIONS):
        sub = agg.loc[agg["position"] == pos, ["game_id", "team", "epa_q"]]
        keyed = (sub.assign(game_id=sub["game_id"].astype(str))
                 .set_index(["game_id", "team"])["epa_q"])
        for side, team_col in (("home", "home_team"), ("away", "away_team")):
            idx = pd.MultiIndex.from_arrays(
                [gids.to_numpy(), sched["home_team"].reindex(gids).to_numpy()
                 if side == "home"
                 else sched["away_team"].reindex(gids).to_numpy()])
            sides[f"{metric}_{side}"] = pd.to_numeric(
                keyed.reindex(idx), errors="coerce").to_numpy(dtype=float)
    for metric in EPA_QUALITY_METRICS:
        sides[f"{metric}_diff"] = sides[f"{metric}_home"] - sides[f"{metric}_away"]
    return pd.concat([df, pd.DataFrame(sides, index=df.index)], axis=1)


# ---------------------------------------------------------------------------
# Weekly-report injury-share family (2026-09-27 Tier B promotion)
# ---------------------------------------------------------------------------
# The served contract's injury availability signal: for each unit (ol, def)
# it reads THIS week's own team report and prices every Out / Injured
# Reserve / Doubtful row with the player's own recent unit-snap share, all
# strictly before the target week. Three definitions:
#
#   inj_<unit>_out_<rep>      count of Out/IR/Doubtful report rows
#   <unit>_snaps_lost_share   SUM of the flagged players' mean unit-snap
#                             share over HIS OWN last 8 active games
#                             (cross-team, as-of strictly before the flag
#                             week; no prior history -> 0.0, never NaN)
#   <unit>_key_out_<rep>      1 if any flagged player's mean share >= 0.60
#
# PIT rule (REPORT CYCLE): a (season, week, team) report row is pre-kickoff
# information for that team-week's game by league rule. The strict-PIT
# loader fails closed on the missing date_modified of the 2025/2026 sources,
# which would zero the family exactly where a slate needs it; the report-
# cycle join replicates the strict-PIT loader's own (team, season, week)
# join and agrees with it on >= 99% of 2016-2024 rows (pinned in
# test_production). No date guessing: a row applies only to its own
# (team, season, week) game.
INJURY_SHARE_UNIT_POSITIONS = {
    "ol": ("T", "G", "C"),
    "def": ("LB", "CB", "S", "DE", "DT", "NT", "ILB", "OLB", "MLB",
            "DB", "SAF", "SS", "FS", "DL", "EDGE"),
}
INJURY_SHARE_SNAP_PCT = {"ol": "offense_pct", "def": "defense_pct"}
INJURY_SHARE_HISTORY_WINDOW = 8   # player's last 8 ACTIVE games, cross-team
INJURY_SHARE_KEY_THRESHOLD = 0.60
# Match the named designation tokens only (plus Injured Reserve), the same
# classifier as ingestion.injury_availability_weight.
INJURY_SHARE_OUT_TOKENS = frozenset({"out", "ir", "doubtful"})

INJURY_SHARE_COLS = [
    f"{name}_{rep}"
    for base in INJURY_SHARE_UNIT_POSITIONS
    for name in (f"inj_{base}_out", f"{base}_snaps_lost_share",
                 f"{base}_key_out")
    for rep in ("home", "away", "diff")
]


def _injured_report_status(status: object) -> bool:
    """True for the Out / IR (incl. Injured Reserve) / Doubtful spellings."""
    if status is None or (isinstance(status, float) and status != status):
        return False
    tokens = re.findall(r"[a-z]+", str(status).strip().lower())
    injured_reserve = any(tokens[i:i + 2] == ["injured", "reserve"]
                          for i in range(len(tokens) - 1))
    return bool(set(tokens) & INJURY_SHARE_OUT_TOKENS) or injured_reserve


def _normalize_player_id(value) -> str | None:
    """Canonical player id, or None for the several null spellings."""
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except TypeError:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "null"}:
        return None
    return text


def injury_share_table(snaps: pd.DataFrame | None,
                       weekly_injuries: pd.DataFrame | None,
                       crosswalk: pd.DataFrame | None) -> pd.DataFrame:
    """Per (team, game) injury-share aggregates, or an empty frame.

    Inputs are the raw snap-count cache rows (needs pfr_player_id), the raw
    weekly report rows and the GSIS->PFR crosswalk. Degrades to empty when
    any input is missing, so the served columns fall back to the
    no-source default (0.0) in _attach_injury_share_features.
    """
    empty = pd.DataFrame(
        columns=["season", "week", "team"]
        + [f"inj_{u}_out" for u in INJURY_SHARE_UNIT_POSITIONS]
        + [f"{u}_snaps_lost_share" for u in INJURY_SHARE_UNIT_POSITIONS]
        + [f"{u}_key_out" for u in INJURY_SHARE_UNIT_POSITIONS])
    if (snaps is None or snaps.empty
            or weekly_injuries is None or weekly_injuries.empty
            or crosswalk is None or crosswalk.empty
            or not {"gsis_id", "pfr_id"} <= set(crosswalk.columns)):
        return empty

    s = snaps.copy()
    s["season"] = pd.to_numeric(s["season"], errors="coerce")
    s["week"] = pd.to_numeric(s["week"], errors="coerce")
    s["team"] = s["team"].astype("string").str.strip().str.upper()
    s["position"] = s["position"].astype("string").str.strip().str.upper()
    s = s.dropna(subset=["pfr_player_id", "season", "week"])
    s = s[~s["pfr_player_id"].isin(["", "nan", "none", "<na>", "null"])]
    # merge_asof requires EXACT key dtypes on both sides; parquet-backed
    # frames arrive with pandas StringDtype while dict-built frames carry
    # object, so normalize to plain object before any asof join.
    s["pfr_player_id"] = s["pfr_player_id"].astype(object)

    inj = weekly_injuries.copy()
    inj["season"] = pd.to_numeric(inj["season"], errors="coerce")
    inj["week"] = pd.to_numeric(inj["week"], errors="coerce")
    inj["team"] = inj["team"].astype("string").str.strip().str.upper()
    inj["player_id"] = inj["gsis_id"].map(_normalize_player_id)
    inj = inj.dropna(subset=["player_id", "team", "season", "week"])
    inj = inj[inj["report_status"].map(_injured_report_status)]
    if inj.empty:
        return empty
    xw = crosswalk[["gsis_id", "pfr_id"]].dropna(how="any").drop_duplicates(
        "gsis_id", keep="first")
    inj["pfr_id"] = inj["player_id"].map(
        xw.set_index("gsis_id")["pfr_id"].to_dict())
    inj = inj.dropna(subset=["pfr_id"])
    if inj.empty:
        return empty
    inj["pfr_id"] = inj["pfr_id"].astype(object)  # asof key dtype parity
    # Report-cycle ordinal (season*30 + week; nflverse weeks never reach 30):
    # one sortable key for the asof join below. Both sides are cast int64 so
    # merge_asof sees one dtype.
    inj = inj[inj["team"].ne("")]
    inj["ord"] = (inj["season"] * 30 + inj["week"]).astype("int64")
    s["ord"] = (s["season"] * 30 + s["week"]).astype("int64")
    inj = inj.sort_values("ord", kind="mergesort")

    # Unit attribution: a report row belongs to the unit of the player's
    # most recent PRIOR active snap (the snap feed is the only position
    # source; the report feed carries none). A player with no prior active
    # snap has no attributable unit and prices 0.0 in both.
    _active = s[pd.to_numeric(s["offense_snaps"], errors="coerce").fillna(0).gt(0)
                | pd.to_numeric(s["defense_snaps"], errors="coerce").fillna(0).gt(0)]
    _pos_asof = pd.merge_asof(
        inj,
        _active[["pfr_player_id", "ord", "position"]]
        .rename(columns={"pfr_player_id": "pfr_id"})
        .sort_values("ord", kind="mergesort"),
        on="ord", by="pfr_id", direction="backward",
        allow_exact_matches=False)
    inj["unit_position"] = pd.Series(
        _pos_asof["position"].to_numpy(), index=inj.index)

    agg_parts = []
    for unit, positions in INJURY_SHARE_UNIT_POSITIONS.items():
        pct_col = INJURY_SHARE_SNAP_PCT[unit]
        snaps_col = "offense_snaps" if unit == "ol" else "defense_snaps"
        hist = s[s["position"].isin(positions)].copy()
        hist[pct_col] = pd.to_numeric(hist[pct_col], errors="coerce")
        hist[snaps_col] = pd.to_numeric(hist[snaps_col], errors="coerce")
        # ACTIVE unit snaps only: share history is priced from games the
        # player actually took the field at this unit (probe-pinned rule 2;
        # the team-window alternative measures ~0% key_out — structurally
        # blind — which is why the window is cross-team).
        hist = hist[hist[snaps_col].gt(0) & hist[pct_col].notna()]
        hist = hist[["pfr_player_id", "ord", pct_col]].sort_values(
            ["pfr_player_id", "ord"], kind="mergesort")
        if hist.empty:
            hist = hist.assign(share=pd.Series(dtype=float))
        else:
            # Mean share over HIS OWN last 8 ACTIVE games, CROSS-TEAM (no
            # team key in the grouping), min_periods=1: the first active
            # game is already a usable prior, and no prior history at all
            # prices 0.0 below — never NaN (probe-pinned rule 1; measured
            # 100% team-game coverage this way).
            hist["share"] = (hist.groupby("pfr_player_id", sort=False)
                             [pct_col].transform(
                                 lambda x: x.rolling(
                                     INJURY_SHARE_HISTORY_WINDOW,
                                     min_periods=1).mean()))
        # ASOF STRICTLY BEFORE the flag week, cross-team: the flag week's
        # own game is excluded (it has not happened when the report cycle
        # runs). A player with no prior active game gets share 0.0.
        unit_inj = inj[inj["unit_position"].isin(positions)]
        priced = pd.merge_asof(
            unit_inj,
            hist.rename(columns={"pfr_player_id": "pfr_id"})
            .sort_values("ord", kind="mergesort"),
            on="ord", by="pfr_id", direction="backward",
            allow_exact_matches=False)
        priced["share"] = pd.to_numeric(priced["share"],
                                        errors="coerce").fillna(0.0)
        grp = (priced.groupby(["season", "week", "team"], as_index=False)
               .agg(out_cnt=("share", "size"),
                    snaps_lost=("share", "sum"),
                    max_share=("share", "max")))
        grp[f"inj_{unit}_out"] = grp["out_cnt"].astype(float)
        grp[f"{unit}_snaps_lost_share"] = grp["snaps_lost"].astype(float)
        grp[f"{unit}_key_out"] = grp["max_share"].ge(
            INJURY_SHARE_KEY_THRESHOLD).astype(float)
        agg_parts.append(grp[["season", "week", "team", f"inj_{unit}_out",
                              f"{unit}_snaps_lost_share",
                              f"{unit}_key_out"]])

    res = agg_parts[0]
    for extra in agg_parts[1:]:
        res = res.merge(extra, on=["season", "week", "team"], how="outer")
    for u in INJURY_SHARE_UNIT_POSITIONS:
        for c in (f"inj_{u}_out", f"{u}_snaps_lost_share", f"{u}_key_out"):
            res[c] = pd.to_numeric(res[c], errors="coerce").fillna(0.0)
    return res[["season", "week", "team"]
               + [f"inj_{u}_out" for u in INJURY_SHARE_UNIT_POSITIONS]
               + [f"{u}_snaps_lost_share" for u in INJURY_SHARE_UNIT_POSITIONS]
               + [f"{u}_key_out" for u in INJURY_SHARE_UNIT_POSITIONS]]


def _attach_injury_share_features(df: pd.DataFrame,
                                  table: pd.DataFrame | None,
                                  games: pd.DataFrame) -> pd.DataFrame:
    """Serve the injury-share family in ONE concat (home/away/diff).

    The table is keyed (season, week, team) — the report-cycle key — and is
    joined to each side of the SCHEDULE's team-games, so the current week's
    slate rows get THIS week's report values even though snap counts only
    cover settled games. Rows with no table entry (no report row, or a
    missing source) fill the documented default 0.0 — measured 100%
    team-game coverage on 2016-2025 — and the diff is home minus away.
    """
    n = len(df)
    sides: dict[str, np.ndarray] = {}
    sched_ok = (games is not None
                and {"game_id", "home_team", "away_team"}.issubset(
                    set(games.columns)))
    if not sched_ok or table is None or table.empty:
        for col in INJURY_SHARE_COLS:
            sides[col] = np.zeros(n, dtype=float)
        return pd.concat([df, pd.DataFrame(sides, index=df.index)], axis=1)

    gids = df["game_id"].astype(str)
    g = (games[["game_id", "season", "week", "home_team", "away_team"]]
         .copy())
    g["game_id"] = g["game_id"].astype(str)
    g["season"] = pd.to_numeric(g["season"], errors="coerce")
    g["week"] = pd.to_numeric(g["week"], errors="coerce")
    for c in ("home_team", "away_team"):
        g[c] = g[c].astype("string").str.strip().str.upper()
    g = g.dropna(subset=["season", "week"])
    keyed = table.copy()
    for c in ("season", "week"):
        keyed[c] = pd.to_numeric(keyed[c], errors="coerce")
    keyed["team"] = keyed["team"].astype("string").str.strip().str.upper()
    keyed = keyed.dropna(subset=["season", "week"])
    for col in INJURY_SHARE_COLS:
        base = col.rsplit("_", 1)[0]
        if base not in keyed.columns:
            keyed[base] = 0.0
    for col in INJURY_SHARE_COLS:
        base = col.rsplit("_", 1)[0]
        for side, team_col in (("home", "home_team"), ("away", "away_team")):
            side_team = (g[["game_id", "season", "week", team_col]]
                         .rename(columns={team_col: "team"})
                         .drop_duplicates("game_id"))
            joined = side_team.merge(
                keyed, on=["season", "week", "team"], how="left")
            m = (joined.set_index("game_id")[base].reindex(gids))
            sides[f"{base}_{side}"] = pd.to_numeric(
                m, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        sides[col] = sides[f"{base}_home"] - sides[f"{base}_away"]
    return pd.concat([df, pd.DataFrame(sides, index=df.index)], axis=1)


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------
def build_game_features(games: pd.DataFrame,
                        pbp: pd.DataFrame | None = None,
                        ps: pd.DataFrame | None = None,
                        ngs: pd.DataFrame | None = None,
                        snaps: pd.DataFrame | None = None,
                        ftn: pd.DataFrame | None = None,
                        weather: pd.DataFrame | None = None,
                        injuries: pd.DataFrame | None = None,
                        weekly_injuries: pd.DataFrame | None = None,
                        crosswalk: pd.DataFrame | None = None) -> pd.DataFrame:
    """Point-in-time feature frame for DECIDED games (one row per game).

    ``games`` must include the warmup timeline (2018+) so early games carry
    real priors; every trailing value is shifted strictly prior. ``ps``/
    ``ngs`` is an optional skill source; ``weather`` is a
    provenance-complete hourly Open-Meteo PIT table.    ``injuries`` is the
    strictly-PIT designation table from ingestion.load_injuries_pit; only a
    PIT Out/IR/Doubtful report removes a player's target-game lineup membership.
    Historical player-game EPA is calculated independently and is retained.

    Absent or inadmissible sources degrade their features to NaN per the
    missing-value policy. Returns the served diff features + the per-side
    values the tree view needs.
    """
    ev = compute_elo(team_events(games))
    agg = pbp_team_agg(pbp)
    ladder = team_stats_ladder(
        ev, agg,
        extra_per_game={"ps": player_stats_team_agg(ps),
                        "ngs": ngs_team_agg(ngs),
                        "ftn": ftn_team_agg(ftn, agg[["game_id", "team"]]),
                        "sc": snap_counts_team_agg(snaps)})

    # Strip unproven schedule/legacy weather payload. Values are attached only
    # from the separately validated hourly Open-Meteo PIT provider below.
    untrusted_weather = {"temp", "wind", "temp_f", "wind_mph",
                         "is_precip", "is_snow"}
    df = games.drop(columns=[c for c in untrusted_weather
                             if c in games.columns]).copy().reset_index(drop=True)
    gids = df["game_id"]
    df["is_home"] = 1.0
    df["elo_diff"] = _home_minus_away(ladder, gids, "elo_entering")
    df["win_pct_diff"] = _home_minus_away(ladder, gids, "win_pct")
    df["rest_days_diff"] = _home_minus_away(ladder, gids, "rest_days")
    df["ewm_net_pts_diff"] = _home_minus_away(ladder, gids, "ewm_net_pts")
    df["ewm_ypp_diff"] = _home_minus_away(ladder, gids, "ewm_ypp")
    df["pace_plays_min_diff"] = _home_minus_away(ladder, gids, "pace_plays_min")
    df["rest_short_diff"] = _home_minus_away(ladder, gids, "short_rest")
    df = _attach_pbp_candidate_features(df, ladder, gids)
    if "div_game" in df.columns:
        df["div_game"] = pd.to_numeric(df["div_game"], errors="coerce")
    else:
        df["div_game"] = np.nan
    df["is_dome_home"] = _is_dome_home(df)
    df = _attach_static_team_facts(
        df, venue_timeline=games, weather=weather)

    # per-side values for the tree view (raw home/away representations)
    for side_col, lad_col in (("elo_home", "elo_entering"), ("elo_away", "elo_entering"),
                              ("win_pct_home", "win_pct"), ("win_pct_away", "win_pct"),
                              ("ewm_net_pts_home", "ewm_net_pts"),
                              ("ewm_net_pts_away", "ewm_net_pts"),
                              ("ewm_ypp_home", "ewm_ypp"), ("ewm_ypp_away", "ewm_ypp"),
                              ("rest_days_home", "rest_days"), ("rest_days_away", "rest_days"),
                              ("pace_plays_min_home", "pace_plays_min"),
                              ("pace_plays_min_away", "pace_plays_min")):
        home_v, away_v = _per_side(ladder, gids, lad_col)
        df[side_col] = home_v if side_col.endswith("home") else away_v

    df = _attach_record_fields(df, ev)
    df = _attach_epa_quality_features(
        df, _epa_quality_agg(games, pbp, ps, injuries), games)
    df = _attach_injury_share_features(
        df, injury_share_table(snaps, weekly_injuries, crosswalk), games)

    # targets (kept beside features for OOF assembly; never model inputs)
    df["margin"] = df["home_score"].astype(float) - df["away_score"].astype(float)
    df["total"] = df["home_score"].astype(float) + df["away_score"].astype(float)
    df["home_win"] = (df["margin"] > 0).astype(float)
    return df


def build_slate_features(schedule: pd.DataFrame,
                         pbp: pd.DataFrame | None,
                         ps: pd.DataFrame | None = None,
                         ngs: pd.DataFrame | None = None,
                         snaps: pd.DataFrame | None = None,
                         ftn: pd.DataFrame | None = None,
                         weather: pd.DataFrame | None = None,
                         injuries: pd.DataFrame | None = None,
                         weekly_injuries: pd.DataFrame | None = None,
                         crosswalk: pd.DataFrame | None = None) -> pd.DataFrame:
    """Point-in-time feature frame for SCHEDULED (undecided) games.

    The ladder spans the full schedule timeline. A pending row's trailing
    values come only from its own strictly-prior rows; settled rows after it
    cannot leak backward (same shift(1) discipline, monotonicity asserted).
    Player EPA candidates use only prior calendar dates (conservative for
    overlapping kickoffs), and injury status is accepted only from the
    strict-PIT injury loader. ``weather`` follows the same boundary as decided
    games: forecasts and provenance must both be pre-kickoff.
    """
    sched = schedule.copy()
    for c in ("home_score", "away_score"):
        if c in sched.columns:
            sched[c] = pd.to_numeric(sched[c], errors="coerce")
    pending = sched[sched["home_score"].isna() | sched["away_score"].isna()]
    if pending.empty:
        return pd.DataFrame()

    # Build one chronological event ladder from the entire schedule. Elo skips
    # unsettled rows, and every trailing series shift(1)s, so a target pending
    # game cannot inherit a later decided outcome merely because that outcome
    # was present in the same schedule object.
    combined = compute_elo(team_events(sched))
    agg = pbp_team_agg(pbp)
    ladder = team_stats_ladder(
        combined, agg,
        extra_per_game={"ps": player_stats_team_agg(ps),
                        "ngs": ngs_team_agg(ngs),
                        "ftn": ftn_team_agg(ftn, agg[["game_id", "team"]]),
                        "sc": snap_counts_team_agg(snaps)})

    untrusted_weather = {"temp", "wind", "temp_f", "wind_mph",
                         "is_precip", "is_snow"}
    df = pending.drop(columns=[c for c in untrusted_weather
                               if c in pending.columns]).copy().reset_index(drop=True)
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
    df = _attach_pbp_candidate_features(df, ladder, gids)
    if "div_game" in df.columns:
        df["div_game"] = pd.to_numeric(df["div_game"], errors="coerce")
    else:
        df["div_game"] = np.nan
    df["is_dome_home"] = _is_dome_home(df)
    df = _attach_static_team_facts(
        df, venue_timeline=sched, weather=weather)

    for side_col, lad_col in (("elo_home", "elo_entering"), ("elo_away", "elo_entering"),
                              ("win_pct_home", "win_pct"), ("win_pct_away", "win_pct"),
                              ("ewm_net_pts_home", "ewm_net_pts"),
                              ("ewm_net_pts_away", "ewm_net_pts"),
                              ("ewm_ypp_home", "ewm_ypp"), ("ewm_ypp_away", "ewm_ypp"),
                              ("rest_days_home", "rest_days"), ("rest_days_away", "rest_days"),
                              ("pace_plays_min_home", "pace_plays_min"),
                              ("pace_plays_min_away", "pace_plays_min")):
        home_v, away_v = _per_side(ladder, gids, lad_col)
        df[side_col] = home_v if side_col.endswith("home") else away_v

    df = _attach_record_fields(df, combined)
    df = _attach_epa_quality_features(
        df, _epa_quality_agg(sched, pbp, ps, injuries), sched)
    df = _attach_injury_share_features(
        df, injury_share_table(snaps, weekly_injuries, crosswalk), sched)
    return df


# ---------------------------------------------------------------------------
# Model-family feature views (deterministic ordering + dimensionality)
# ---------------------------------------------------------------------------
def linear_feature_columns(active: list[str] | None = None) -> list[str]:
    """Diff/anchor projection of the contract (linear + MLP member view).

    Routing RULE, not a second list: the linear family never sees the raw
    per-side levels (config.RAW_PER_SIDE_COLS) — those exist for the tree
    members, which consume the whole contract.
    """
    cols = config.active_moneyline_feature_cols() if active is None else list(active)
    return [c for c in cols if c not in config.RAW_PER_SIDE_COLS]


def linear_view(df: pd.DataFrame) -> pd.DataFrame:
    """Linear/MLP matrix: pure projection of the active contract (diff view)."""
    cols = [c for c in linear_feature_columns() if c in df.columns]
    out = df.reindex(columns=cols).astype(float)
    return out


def team_category_ids(df: pd.DataFrame) -> pd.DataFrame:
    """The config.TREE_CATEGORICAL_COLS pair from home/away abbreviations.

    Stable integer IDs (config.NFL_TEAM_ID) with the reserved UNK_TEAM_ID
    slot for historical/abandoned/international/missing labels — never a
    silent alias of a real team. Tree-family context ONLY: never in
    MONEYLINE_FEATURE_COLS, never seen by the linear/MLP family.
    """
    out = pd.DataFrame(index=df.index)
    if "home_team" in df.columns:
        out["home_team_id"] = df["home_team"].map(config.team_category_id).astype("int64")
    else:
        out["home_team_id"] = config.UNK_TEAM_ID
    if "away_team" in df.columns:
        out["away_team_id"] = df["away_team"].map(config.team_category_id).astype("int64")
    else:
        out["away_team_id"] = config.UNK_TEAM_ID
    return out


def tree_numeric_columns() -> list[str]:
    """The numeric part of the tree view (the served contract projection).

    The run-line/totals regressors slice to this: LightGBM Poisson regressors
    are plain numeric models, so their numeric matrix must stay purely
    numeric (MLB parity — ScoreRegressor-equivalents there consume the
    numeric matrix, not the categorical context).
    """
    return [c for c in config.active_moneyline_feature_cols()]


def tree_view(df: pd.DataFrame) -> pd.DataFrame:
    """Tree-family matrix: the served contract + the team-ID categorical pair.

    The numeric block comes from config.MONEYLINE_FEATURE_COLS (the one
    master list — no column is synthesized here). The categorical pair is
    APPENDED after it for the tree members only, exactly like MLB's adopted
    TREE_CATEGORICAL_COLS routing: the linear/MLP family (linear_view) never
    receives these, and no view emits a duplicate. Diffs, raw per-side
    levels and the anchor all come from the master list, so the matrix is
    unique and drift-free by construction.
    """
    cols = [c for c in config.active_moneyline_feature_cols() if c in df.columns]
    out = df.reindex(columns=cols).astype(float)
    cats = team_category_ids(df)
    for c in config.TREE_CATEGORICAL_COLS:
        if c not in out.columns:
            out[c] = cats[c]
    return out


def feature_coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage + missingness diagnostics per served feature."""
    rows = []
    for f in config.active_moneyline_feature_cols():
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
