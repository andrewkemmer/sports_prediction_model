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


def pbp_ladder_columns() -> list[str]:
    """Every ladder column the pbp candidate family produces
    (``<metric>_<window>`` per the config specs), in declaration order."""
    return [f"{metric}_{w}" for metric, windows in PBP_TRAILING_SPECS.items()
            for w in windows]


# ---------------------------------------------------------------------------
# Starter-QB player-linked series (2026-09-24, MLB starting-pitcher parity)
# ---------------------------------------------------------------------------
# The announced starting QB (schedule home/away_qb_id — published BEFORE the
# game, so the join is point-in-time-safe) keys each game to a PLAYER. The
# trailing series is that player's OWN weekly player-stats line
# (passing_epa / passing_cpoe), trailed on the PLAYER's start timeline with
# the same shift(1) primitives the team machinery uses — NOT the per-team
# ladder loop, whose rows are team-games. A backup start therefore carries
# the backup's own level and an unknown/missing starter degrades to NaN;
# the team-mean pbp_qb_epa_dropback_* candidates remain in the pool as the
# always-populated complement.

def starter_qb_series(ps: pd.DataFrame | None,
                      games: pd.DataFrame) -> pd.DataFrame:
    """Per-(game_id, team) announced-starter trailing metrics.

    Rows: every (game_id, home/away team) pair whose announced QB id exists.
    Columns: qbs_epa_starter_ewm, qbs_epa_starter_roll, qbs_cpoe_starter_ewm
    — the player's strictly-prior trailing values at his OWN announcement.
    Determinism: identical (player, gameday) inputs produce identical
    series; no randomness anywhere.
    """
    empty = pd.DataFrame(columns=["game_id", "team", "qbs_epa_starter_ewm",
                                  "qbs_epa_starter_roll",
                                  "qbs_cpoe_starter_ewm"])
    if ps is None or not len(ps):
        return empty
    need = {"player_id", "passing_epa", "passing_cpoe", "week", "season",
            "game_id", "team"}
    if not need <= set(ps.columns):
        return empty
    if not {"home_qb_id", "away_qb_id"} <= set(games.columns):
        return empty

    gd = pd.to_datetime(games["gameday"], errors="coerce")

    def _qid(series: pd.Series) -> pd.Series:
        """Normalize GSIS-style QB ids ('00-0033873') to clean strings."""
        return (series.astype("string").str.strip()
                .replace({"": pd.NA, "None": pd.NA, "nan": pd.NA, "NA": pd.NA}))

    rows = pd.DataFrame({
        "game_id": pd.concat([games["game_id"], games["game_id"]],
                             ignore_index=True),
        "gameday": pd.concat([gd, gd], ignore_index=True),
        "team": pd.concat([games["home_team"], games["away_team"]],
                          ignore_index=True).to_numpy(),
        "qb_id": pd.concat([_qid(games["home_qb_id"]),
                            _qid(games["away_qb_id"])],
                           ignore_index=True).to_numpy(),
    }).dropna(subset=["qb_id", "gameday"])

    # Per-player per-game efficiency lines from the weekly stats payload.
    # The narrowed ps cache never carries gameday — the date comes from the
    # decided game map (game_id -> gameday), then each player's start
    # timeline is his own rows sorted by that date.
    p = ps.copy()
    p["player_id"] = p["player_id"].astype("string").str.strip()
    p = p.drop(columns=[c for c in ("gameday",) if c in p.columns]).merge(
        pd.DataFrame({"game_id": games["game_id"], "gameday": gd}),
        on="game_id", how="left")
    p["epa"] = pd.to_numeric(p["passing_epa"], errors="coerce")
    p["cpoe"] = pd.to_numeric(p["passing_cpoe"], errors="coerce")
    p = p[p["player_id"].notna()]

    # One row per player-start: sum is the safe per-game aggregator (a
    # player's weekly row is his single game; splits would double-count).
    pg = (p.dropna(subset=["gameday"])
           .groupby(["player_id", "game_id", "gameday"], as_index=False)
           .agg(epa=("epa", "sum"), cpoe=("cpoe", "mean"),
                team=("team", "first")))

    out_parts = []
    for (qb, team), grp in rows.groupby(["qb_id", "team"], sort=False):
        starts = (pg[(pg["player_id"] == qb) & (pg["team"] == team)]
                  .sort_values("gameday"))
        if not len(starts):
            continue
        # NO shift(1) here: the merge_asof below (backward,
        # allow_exact_matches=False) already guarantees the joined value is
        # the latest start STRICTLY BEFORE the announced game — the player's
        # most recent completed performance is included, and the announced
        # game's own result never is. (A shift here would double-protect and
        # drop that most-recent start — the off-by-one the 2026-09-24 PIT
        # spot-check caught.)
        idx = pd.Index(starts["gameday"])
        ewm = starts["epa"].ewm(halflife=config.EWM_HALFLIFE, min_periods=1).mean()
        roll = starts["epa"].rolling(config.PBP_ROLL_WINDOW, min_periods=1).mean()
        cp = starts["cpoe"].ewm(halflife=config.EWM_HALFLIFE, min_periods=1).mean()
        ser = pd.DataFrame({"qbs_epa_starter_ewm": ewm.to_numpy(),
                            "qbs_epa_starter_roll": roll.to_numpy(),
                            "qbs_cpoe_starter_ewm": cp.to_numpy()}, index=idx)
        # Join THIS player's series onto his announced games at the game date
        # (merge_asof backward = the latest start STRICTLY BEFORE gameday).
        mine = rows[(rows["qb_id"] == qb) & (rows["team"] == team)].copy()
        mine = mine.sort_values("gameday")
        joined = pd.merge_asof(mine, ser.reset_index().rename(
            columns={"index": "gameday"}),
            on="gameday", direction="backward",
            allow_exact_matches=False)
        out_parts.append(joined)
    if not out_parts:
        return empty
    out = pd.concat(out_parts, ignore_index=True)
    return out[["game_id", "team", "qbs_epa_starter_ewm",
                "qbs_epa_starter_roll", "qbs_cpoe_starter_ewm"]]


def _attach_starter_qb_features(df: pd.DataFrame,
                                qbs: pd.DataFrame | None) -> pd.DataFrame:
    """Serve the qbs_ candidate family onto the game frame (diff + sides).

    One concat; games without an announced/known starter keep NaN (the
    documented degradation — never the team mean masquerading as a player).
    """
    cols = [f"qbs_{m}_{w}_{r}"
            for m, ws in config.QBS_CANDIDATE_TRAILING_SPECS.items()
            for w in ws for r in ("diff", "home", "away")]
    if qbs is None or not len(qbs):
        return pd.concat([df, pd.DataFrame(
            {c: np.nan for c in cols}, index=df.index)], axis=1)
    idx = pd.MultiIndex.from_frame(qbs[["game_id", "team"]])
    lut = {c: qbs.set_index(["game_id", "team"])[c] for c in
           ("qbs_epa_starter_ewm", "qbs_epa_starter_roll",
            "qbs_cpoe_starter_ewm")}
    home_idx = pd.MultiIndex.from_arrays([df["game_id"], df["home_team"]])
    away_idx = pd.MultiIndex.from_arrays([df["game_id"], df["away_team"]])
    sides = {}
    for col, series in lut.items():
        h = series.reindex(home_idx).to_numpy(dtype=float)
        a = series.reindex(away_idx).to_numpy(dtype=float)
        sides[f"{col}_diff"] = h - a
        sides[f"{col}_home"] = h
        sides[f"{col}_away"] = a
    frame = pd.DataFrame(sides, index=df.index)
    return pd.concat([df, frame.reindex(columns=cols)], axis=1)


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
# Skill & availability rollups — player stats, Next-Gen Stats, injuries
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


# Weekly injury reports → per-(season, week, team) availability counts. These
# are PRE-GAME facts (the report publishes before kickoff), so they attach as
# static per-game values — no trailing windows, no candidate phase. The
# payload's availability field is report_status; "Out" = ruled out for the
# game (the endpoint's rows carry no separate IR status string).
_INJ_GROUPS = {
    "qb": ("QB",),
    "tackle": ("T", "OT", "LT", "RT"),
    "edge": ("EDGE", "DE", "OLB", "OL"),
}


def injuries_week_facts(inj: pd.DataFrame | None) -> pd.DataFrame:
    """(season, week, team) -> Out counts by position group (report_status).

    Columns: inj_qb_out, inj_tackle_out, inj_edge_out, inj_starters_out
    (total Out across the report). Empty frame when data is absent."""
    cols = ["season", "week", "team", "inj_qb_out", "inj_tackle_out",
            "inj_edge_out", "inj_starters_out"]
    if inj is None or inj.empty:
        return pd.DataFrame(columns=cols)
    p = inj.copy()
    need = ["season", "week", "team", "position", "report_status"]
    if any(c not in p.columns for c in need):
        return pd.DataFrame(columns=cols)
    # Zero-fill over the report's own (season, week, team) universe: a
    # team-week with NO Out entries is a real zero, not missing — only
    # team-weeks absent from the reports entirely stay NaN.
    p = p[p["report_status"].astype(str).str.strip().str.lower().eq("out")]
    universe = (inj[["season", "week", "team"]]
                .drop_duplicates().reset_index(drop=True))
    if p.empty:
        rows = universe.copy()
        for c in cols[3:]:
            rows[c] = 0.0
        return rows[cols]
    pos = p["position"].astype(str).str.upper()
    p = p.assign(_pos=pos)
    rows = (p.groupby(["season", "week", "team"])
             .apply(lambda d: pd.Series({
                 "inj_qb_out": float(d["_pos"].isin(_INJ_GROUPS["qb"]).sum()),
                 "inj_tackle_out": float(d["_pos"].isin(_INJ_GROUPS["tackle"]).sum()),
                 "inj_edge_out": float(d["_pos"].isin(_INJ_GROUPS["edge"]).sum()),
                 "inj_starters_out": float(len(d)),
             }), include_groups=False)
             .reset_index())
    rows = universe.merge(rows, on=["season", "week", "team"], how="left")
    rows[cols[3:]] = rows[cols[3:]].fillna(0.0)
    for c in cols[3:]:
        rows[c] = pd.to_numeric(rows[c], errors="coerce")
    return rows[cols]


def _attach_static_team_facts(df: pd.DataFrame,
                              inj_facts: pd.DataFrame | None) -> pd.DataFrame:
    """Attach per-side injury availability diffs (home − away) from the
    weekly reports, plus the game-level environment facts (venue/surface/
    observed weather). All-NaN when the sources are absent."""
    df = df.copy()
    names = ("inj_qb_out", "inj_tackle_out", "inj_edge_out", "inj_starters_out")
    if (inj_facts is None or inj_facts.empty or "season" not in df.columns
            or "week" not in df.columns):
        for n in names:
            df[f"{n}_diff"] = np.nan
    else:
        f = inj_facts.copy()
        f["season"] = pd.to_numeric(f["season"], errors="coerce")
        f["week"] = pd.to_numeric(f["week"], errors="coerce")
        key = f.set_index(["season", "week", "team"])
        ssn = pd.to_numeric(df["season"], errors="coerce")
        wk = pd.to_numeric(df["week"], errors="coerce")

        def _side(team_col: str, name: str) -> np.ndarray:
            return key.reindex(pd.MultiIndex.from_arrays(
                [ssn, wk, df[team_col].astype(str)]))[name].to_numpy(dtype=float)

        for n in names:
            home_v = _side("home_team", n)
            away_v = _side("away_team", n)
            df[f"{n}_diff"] = home_v - away_v
            # Per-side candidate levels (config.STATIC_SIDE_CANDIDATES):
            # the tree family's raw representations of the served diffs.
            df[f"{n}_home"] = home_v
            df[f"{n}_away"] = away_v

    # Venue geometry (travel distance, altitude) and prime-time flag.
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
    df["travel_miles_diff"] = (_haversine_miles(home_lat, home_lon, game_lat, game_lon)
                               - _haversine_miles(away_lat, away_lon, game_lat, game_lon))
    # Per-side candidate levels (config.STATIC_SIDE_CANDIDATES): each team's
    # own distance to the game venue.
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

    # Observed game-day weather (committed table + schedule fallback).
    df = _attach_weather(df)
    return df


@functools.lru_cache(maxsize=1)
def _load_weather_table() -> pd.DataFrame:
    """The committed observed-weather table (build_weather_table.py output)."""
    f = config.BACKEND_DIR / "nfl_weather.csv"
    if not f.exists():
        return pd.DataFrame(columns=["stadium", "gameday", "temp_f",
                                     "wind_mph", "precip_in", "snow_in"])
    w = pd.read_csv(f)
    w["gameday"] = pd.to_datetime(w["gameday"], errors="coerce").dt.date
    return w.drop_duplicates(["stadium", "gameday"], keep="last")


def _attach_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Attach temp_f / wind_mph / is_precip / is_snow keyed by (stadium,
    gameday) from the committed weather table; the schedule's own temp/wind
    payload columns are the fallback where the table lacks a row. NaN =
    unknown (domes, international sites, missing rows) — never fabricated."""
    df = df.copy()
    n = len(df)
    wx = _load_weather_table()
    if wx.empty or not {"stadium", "gameday"} <= set(df.columns):
        t = np.full(n, np.nan); w = t.copy(); pr = t.copy(); sn = t.copy()
    else:
        gd = pd.to_datetime(df["gameday"], errors="coerce").dt.date
        rows = wx.set_index(["stadium", "gameday"]).reindex(
            pd.MultiIndex.from_arrays([df["stadium"].astype(str), gd]))
        t = rows["temp_f"].to_numpy(dtype=float)
        w = rows["wind_mph"].to_numpy(dtype=float)
        pr = rows["precip_in"].to_numpy(dtype=float)
        sn = rows["snow_in"].to_numpy(dtype=float)
    sched_t = (pd.to_numeric(df["temp"], errors="coerce").to_numpy(dtype=float)
               if "temp" in df.columns else np.full(n, np.nan))
    sched_w = (pd.to_numeric(df["wind"], errors="coerce").to_numpy(dtype=float)
               if "wind" in df.columns else np.full(n, np.nan))

    def _coalesce(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.where(pd.notna(a), a, b)

    df["temp_f"] = _coalesce(t, sched_t)
    df["wind_mph"] = _coalesce(w, sched_w)
    df["is_precip"] = np.where(pd.notna(pr),
                               (pr >= config.PRECIP_FLAG_IN).astype(float), np.nan)
    df["is_snow"] = np.where(pd.notna(sn),
                             (sn >= config.SNOW_FLAG_IN).astype(float), np.nan)
    return df


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
                        pbp: pd.DataFrame | None = None,
                        ps: pd.DataFrame | None = None,
                        ngs: pd.DataFrame | None = None,
                        inj: pd.DataFrame | None = None,
                        snaps: pd.DataFrame | None = None,
                        ftn: pd.DataFrame | None = None,
                        qbs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Point-in-time feature frame for DECIDED games (one row per game).

    ``games`` must include the warmup timeline (2018+) so early games carry
    real priors; every trailing value is shifted strictly prior. ``ps``/
    ``ngs``/``inj``/``qbs`` are the optional skill/availability sources
    (weekly player stats, Next-Gen Stats, injury reports, starter-QB series
    from the announced schedule QB ids); absent sources degrade their
    features to NaN per the missing-value policy. Returns the served diff
    features + the per-side values the tree view needs.
    """
    ev = compute_elo(team_events(games))
    agg = pbp_team_agg(pbp)
    ladder = team_stats_ladder(
        ev, agg,
        extra_per_game={"ps": player_stats_team_agg(ps),
                        "ngs": ngs_team_agg(ngs),
                        "ftn": ftn_team_agg(ftn, agg[["game_id", "team"]]),
                        "sc": snap_counts_team_agg(snaps)})
    inj_facts = injuries_week_facts(inj)

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
    df = _attach_pbp_candidate_features(df, ladder, gids)
    df = _attach_starter_qb_features(df, qbs)
    if "div_game" in df.columns:
        df["div_game"] = pd.to_numeric(df["div_game"], errors="coerce")
    else:
        df["div_game"] = np.nan
    df["is_dome_home"] = np.where(
        df.get("roof", pd.Series(np.nan, index=df.index)).isin(["dome", "closed"]),
        1.0, np.where(df.get("roof", pd.Series(np.nan, index=df.index)).isin(["outdoors"]),
                      0.0, np.nan))
    df = _attach_static_team_facts(df, inj_facts)

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
                         pbp: pd.DataFrame | None,
                         ps: pd.DataFrame | None = None,
                         ngs: pd.DataFrame | None = None,
                         inj: pd.DataFrame | None = None,
                         snaps: pd.DataFrame | None = None,
                         ftn: pd.DataFrame | None = None,
                         qbs: pd.DataFrame | None = None) -> pd.DataFrame:
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
    agg = pbp_team_agg(pbp)
    ladder = team_stats_ladder(
        combined, agg,
        extra_per_game={"ps": player_stats_team_agg(ps),
                        "ngs": ngs_team_agg(ngs),
                        "ftn": ftn_team_agg(ftn, agg[["game_id", "team"]]),
                        "sc": snap_counts_team_agg(snaps)})
    inj_facts = injuries_week_facts(inj)

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
    df = _attach_pbp_candidate_features(df, ladder, gids)
    df = _attach_starter_qb_features(df, qbs)
    if "div_game" in df.columns:
        df["div_game"] = pd.to_numeric(df["div_game"], errors="coerce")
    else:
        df["div_game"] = np.nan
    df["is_dome_home"] = np.where(
        df.get("roof", pd.Series(np.nan, index=df.index)).isin(["dome", "closed"]),
        1.0, np.where(df.get("roof", pd.Series(np.nan, index=df.index)).isin(["outdoors"]),
                      0.0, np.nan))
    df = _attach_static_team_facts(df, inj_facts)

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
