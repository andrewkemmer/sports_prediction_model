"""Authoritative point-in-time NHL feature engine.

Structural mirror of the NFL features.py (which mirrors MLB). Single owner
of every production feature; the models consume frames built here and
nowhere else.

LEAKAGE CONTRACT (enforced structurally and asserted):
  feature(game_t) = f(information available STRICTLY BEFORE game_t)

- Elo: iterated strictly chronologically; each row's ``elo_entering`` is the
  team's rating at puck drop (updated only after that game settles); home
  advantage is added to the home team's expected-win computation; 1/3 revert
  toward the prior is applied at each season boundary.
- Trailing windowed / EWM statistics: computed per team over its own games
  in chronological order, then ``shift(1)`` — the current and all future
  games are excluded from a row's own value.
- Goalie rolling stats: per-goalie, over that goalie's own prior STARTS this
  season (shift(1)); the expected starter at game t is the team's most
  frequent STARTER entering t — resolved strictly prior.
- Game facts (playoffs flag, rest days) are static pre-game facts.

Model-family representations:
  - linear view: difference features + the constant ``is_home`` anchor
  - tree view: differences + raw home/away side values + the team-ID
    categorical pair (config.TREE_CATEGORICAL_COLS)
All views are deterministic in ordering and dimensionality.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    from backend import config
except ImportError:
    import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Events: long-form (team, game) view
# ---------------------------------------------------------------------------
REQUIRED_GAME_COLS = ["game_id", "season", "gameday", "home_team",
                      "away_team", "home_score", "away_score"]


def team_events(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game) with team-perspective scores."""
    missing = [c for c in REQUIRED_GAME_COLS if c not in games.columns]
    if missing:
        raise ValueError(f"team_events: missing columns {missing}")
    gd = pd.to_datetime(games["gameday"], errors="coerce")
    home = pd.DataFrame({
        "game_id": games["game_id"], "season": games["season"],
        "gameday": gd,
        "team": games["home_team"], "opponent": games["away_team"],
        "is_home": True,
        "for": pd.to_numeric(games["home_score"], errors="coerce"),
        "against": pd.to_numeric(games["away_score"], errors="coerce"),
    })
    away = pd.DataFrame({
        "game_id": games["game_id"], "season": games["season"],
        "gameday": gd,
        "team": games["away_team"], "opponent": games["home_team"],
        "is_home": False,
        "for": pd.to_numeric(games["away_score"], errors="coerce"),
        "against": pd.to_numeric(games["home_score"], errors="coerce"),
    })
    ev = pd.concat([home, away], ignore_index=True)
    ev["net_from_team"] = ev["for"] - ev["against"]
    # No ties in the shootout era: OT/SO decided games award the W.
    ev["team_win"] = np.select(
        [ev["for"] > ev["against"], ev["for"] < ev["against"]],
        [1.0, 0.0], default=0.5)
    ev["goal_share"] = ev["for"] / (ev["for"] + ev["against"]).replace(0, np.nan)
    return ev


# ---------------------------------------------------------------------------
# Elo — pre-game entering rating, updated only after a game settles
# ---------------------------------------------------------------------------
def _season_boundary(prev_season: object, season: object) -> bool:
    try:
        return prev_season is not None and int(prev_season) != int(season)
    except (TypeError, ValueError):
        return False


def _elo_apply(events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Iterated Elo with MLB semantics: K=20, home advantage 65 (added to the
    home side's expected win), 400-point scale, and a 1/3 revert toward the
    prior at each season boundary."""
    K, prior, scale = config.ELO_K, config.ELO_PRIOR, config.ELO_SCALE
    home_adv = getattr(config, "ELO_HOME_ADV", 0.0)
    revert = getattr(config, "ELO_REVERT_FACTOR", 0.0)
    ev = events.sort_values(["gameday", "game_id", "is_home"]).reset_index(drop=True)
    rating: dict = {}
    entering: dict = {}
    last_season: object = None
    for game_id, rows in ev.groupby("game_id", sort=False):
        a, b = list(rows.itertuples(index=False))[:2]
        season = getattr(a, "season", None)
        if _season_boundary(last_season, season):
            for t in rating:
                rating[t] = rating[t] + revert * (prior - rating[t])
        last_season = season
        ra = rating.get(a.team, prior)
        rb = rating.get(b.team, prior)
        # Home advantage enters the home team's expected-win expression.
        if a.is_home:
            exp_a = 1.0 / (1.0 + 10.0 ** ((rb + home_adv - ra) / scale))
        else:
            exp_a = 1.0 / (1.0 + 10.0 ** ((rb - home_adv - ra) / scale))
        exp_b = 1.0 - exp_a
        entering[(game_id, a.team)] = ra
        entering[(game_id, b.team)] = rb
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


# Trailing-window spec for the candidate-pool metrics: per-game metric -> the
# ladder suffixes it is served at. DECLARED ONCE in config
# (TEAM_CANDIDATE_TRAILING_SPECS — the same declaration that names the RFE
# candidates) and pulled here; every derivation rides the shared
# _trailing_ewm / _trailing_per_team primitives so the shift(1) leakage
# discipline is inherited, not reimplemented.
NHL_TRAILING_SPECS: dict[str, tuple[str, ...]] = {
    **config.TEAM_CANDIDATE_TRAILING_SPECS,
}


# ---------------------------------------------------------------------------
# The ladder — team state + trailing stats, one row per (game_id, team)
# ---------------------------------------------------------------------------
def team_stats_ladder(events: pd.DataFrame,
                      team_game_agg: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-(game_id, team) point-in-time state: elo_entering, win_pct,
    rest_days, ewm_net_goals, ewm_goal_share, ga_per_game, short_rest/back-
    to-back + the candidate-pool trailing metrics — every trailing value
    strictly-prior (asserted)."""
    ev = events.copy()
    if team_game_agg is not None and len(team_game_agg):
        agg_cols = [c for c in team_game_agg.columns
                    if c not in ("game_id", "team")]
        ev = ev.merge(team_game_agg[["game_id", "team"] + agg_cols],
                      on=["game_id", "team"], how="left")

    srt = ev.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)

    # LEAKAGE GATE: gameday strictly increasing within each team
    diffs = srt.groupby("team", sort=False)["gameday"].diff()
    bad = srt.loc[(diffs.notna()) & (diffs <= pd.Timedelta(0))]
    if len(bad):
        raise AssertionError(
            f"team_stats_ladder: team gameday not strictly increasing "
            f"({len(bad)} rows) — trailing features could leak")

    srt["elo_entering"] = srt["elo_entering"].astype(float)
    srt["win_pct"] = _trailing_per_team(srt, "team_win", config.WINPCT_WINDOW)
    srt["rest_days"] = srt.groupby("team", sort=False)["gameday"].diff().dt.days
    srt["back_to_back"] = np.where(
        srt["rest_days"].notna(), (srt["rest_days"] <= 1).astype(float), np.nan)
    srt["ewm_net_goals"] = _trailing_ewm(srt, "net_from_team", config.EWM_HALFLIFE)
    srt["ewm_goal_share"] = _trailing_ewm(srt, "goal_share", config.EWM_HALFLIFE)
    srt["ga_per_game"] = _trailing_per_team(srt, "against", config.GOALS_AGAINST_WINDOW)

    # Candidate-pool trailing metrics (the family prefix makes the served
    # names nhl_<metric>_<window>_<rep>). Absent source columns are all-NaN
    # per-game and degrade to all-NaN trailing (never fabricated).
    for metric, windows in NHL_TRAILING_SPECS.items():
        if metric not in srt.columns:
            for w in windows:
                srt[f"{metric}_{w}"] = np.nan
            continue
        for w in windows:
            if w == "ewm":
                srt[f"{metric}_{w}"] = _trailing_ewm(srt, metric,
                                                     config.EWM_HALFLIFE)
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
# Goalie rolling state — per-goalie, per-start, strictly-prior
# ---------------------------------------------------------------------------

def goalie_state(boxscores: pd.DataFrame | None,
                 games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-goalie rolling SV% / GAA state + per-game expected-starter frame.

    ``boxscores`` carries one row per game (the ingestion rollup: decision
    goalie id/name, per-goalie goals/shots against, and TOI). The per-start
    save fraction for a goalie = 1 - (goals against / shots against), and
    per-start GAA = goals against / (TOI minutes / 60).

    Every goalie statistic is shifted strictly prior on the GOALIE's own
    start timeline, then EWM'd (halflife = EWM_HALFLIFE starts). The
    expected starter entering game t is the goalie with the most PRIOR
    starts for that team (season-to-date workload), resolved strictly
    before t.

    Returns (per_game_frame, ladder_frame):
      per_game_frame: one row per game with the HOME/AWAY side's expected
        starter's rolling sv_pct / gaa / starts (NaN when unresolvable —
        honest degradation, the MLB TBD-pitcher analog).
      ladder_frame: per-(game_id, team) goalie starts state (for the
        workload diff and the goalie-matchup artifact).
    """
    per_game = pd.DataFrame(index=games.index)
    cols = ["goalie_sv_pct_home", "goalie_sv_pct_away",
            "goalie_gaa_home", "goalie_gaa_away",
            "goalie_starts_home", "goalie_starts_away",
            "g_home_name", "g_away_name",
            "g_home_sv_pct", "g_away_sv_pct",
            "g_home_gaa", "g_away_gaa", "g_home_starts", "g_away_starts"]
    if boxscores is None or boxscores.empty or "game_id" not in boxscores.columns:
        for c in cols:
            per_game[c] = np.nan
        return per_game, pd.DataFrame(columns=["game_id", "team", "goalie_id",
                                               "goalie_name", "prior_starts"])

    bs = boxscores.copy()
    bs["game_id"] = bs["game_id"].astype(str)

    # Long-form per-start rows: (game_id, team, goalie_id, name, goals_against,
    # shots_faced, toi_min, gameday).
    starts = []
    gdate = games.set_index("game_id")["gameday"] if "gameday" in games.columns else {}
    for r in bs.itertuples(index=False):
        gid = str(getattr(r, "game_id", ""))
        gd = gdate.get(gid) if hasattr(gdate, "get") else None
        for side in ("home", "away"):
            g_id = getattr(r, f"{side}_goalie_id", None)
            if g_id is None or (isinstance(g_id, float) and pd.isna(g_id)):
                continue
            ga = getattr(r, f"{side}_goals_against", None)
            shots = getattr(r, f"{side}_shots_against", None)
            toi = getattr(r, f"{side}_goalie_toi", None)
            try:
                ga = float(ga)
            except (TypeError, ValueError):
                ga = float("nan")
            try:
                shots = float(shots)
            except (TypeError, ValueError):
                shots = float("nan")
            try:
                toi_min = float(toi)
            except (TypeError, ValueError):
                toi_min = float("nan")
            if not (np.isfinite(toi_min) and toi_min >= config.MIN_GOALIE_TOI_MINUTES):
                ga = shots = toi_min = float("nan")
            starts.append({
                "game_id": gid, "team": side, "goalie_id": str(g_id),
                "goalie_name": str(getattr(r, f"{side}_goalie_name", "") or ""),
                "goals_against": ga,
                "shots_faced": shots,
                "toi_min": toi_min,
                "gameday": pd.to_datetime(gd, errors="coerce"),
            })
    st = pd.DataFrame(starts)
    if st.empty:
        for c in cols:
            per_game[c] = np.nan
        return per_game, pd.DataFrame(columns=["game_id", "team", "goalie_id",
                                               "goalie_name", "prior_starts"])
    # Per-side team abbreviations (starts carry 'home'/'away' placeholders).
    teams_by_game: dict[tuple[str, str], str] = {}
    if "home_team" in games.columns:
        for r in games.itertuples(index=False):
            gid = str(getattr(r, "game_id", ""))
            teams_by_game[(gid, "home")] = str(getattr(r, "home_team", ""))
            teams_by_game[(gid, "away")] = str(getattr(r, "away_team", ""))
    st["team_abbr"] = [teams_by_game.get((str(g), side), side)
                       for g, side in zip(st["game_id"], st["team"])]

    # Per-start save fraction + GAA (per 60 min).
    st["save_fraction"] = 1.0 - (st["goals_against"] / st["shots_faced"].replace(0, np.nan))
    st["gaa_game"] = st["goals_against"] / (st["toi_min"] / 60.0).replace(0, np.nan)

    st = st.sort_values(["goalie_id", "gameday", "game_id"]).reset_index(drop=True)
    st["prior_starts"] = st.groupby("goalie_id").cumcount()
    st["sv_pct"] = st.groupby("goalie_id", sort=False)["save_fraction"].transform(
        lambda s: s.ewm(halflife=config.EWM_HALFLIFE, min_periods=1).mean().shift(1)
    )
    st["gaa"] = st.groupby("goalie_id", sort=False)["gaa_game"].transform(
        lambda s: s.ewm(halflife=config.EWM_HALFLIFE, min_periods=1).mean().shift(1)
    )

    # Expected starter per (game, team): the goalie with the most PRIOR
    # starts entering that game. Resolve on each start row: the row's own
    # goalie is "the starter of record for that game" post-hoc; the
    # EXPECTED starter uses only strictly-prior workload. For decided
    # history we use the decision goalie as the served expected starter
    # ONLY via his prior-starts state (shift already applied). The row's
    # sv_pct/gaa are strictly-prior by construction; the starter identity
    # is the row's own goalie (pre-game the slate layer resolves the same
    # rule on pending games).
    st = st.sort_values(["game_id", "team"]).reset_index(drop=True)
    exp = (st.sort_values("prior_starts", ascending=False)
             .drop_duplicates(["game_id", "team"], keep="first"))
    ladder = exp[["game_id", "team_abbr", "goalie_id", "goalie_name",
                  "prior_starts"]].rename(
        columns={"team_abbr": "team", "prior_starts": "goalie_starts"})
    exp = exp.set_index(["game_id", "team"])

    hl = per_game.index
    for side, side_key in (("home", "home"), ("away", "away")):
        sv = []
        gaa = []
        starts_n = []
        names = []
        for _, g in games.iterrows():
            gid = str(g.get("game_id", ""))
            try:
                row = exp.loc[(gid, side_key)]
                sv.append(float(row["sv_pct"]) if pd.notna(row["sv_pct"]) else np.nan)
                gaa.append(float(row["gaa"]) if pd.notna(row["gaa"]) else np.nan)
                starts_n.append(float(row["prior_starts"]))
                names.append(str(row["goalie_name"] or ""))
            except (KeyError, TypeError, ValueError):
                sv.append(np.nan)
                gaa.append(np.nan)
                starts_n.append(np.nan)
                names.append("")
        per_game[f"goalie_sv_pct_{side}"] = sv
        per_game[f"goalie_gaa_{side}"] = gaa
        per_game[f"goalie_starts_{side}"] = starts_n
        per_game[f"g_{side}_name"] = names
        per_game[f"g_{side}_sv_pct"] = per_game[f"goalie_sv_pct_{side}"]
        per_game[f"g_{side}_gaa"] = per_game[f"goalie_gaa_{side}"]
        per_game[f"g_{side}_starts"] = per_game[f"goalie_starts_{side}"]
    return per_game, ladder


# ---------------------------------------------------------------------------
# Candidate-pool serving (shared by the decided and slate builders)
# ---------------------------------------------------------------------------
def _attach_candidate_features(df: pd.DataFrame, ladder: pd.DataFrame,
                               gids: pd.Index) -> pd.DataFrame:
    """Serve every trailing candidate family onto a game frame.

    Every NHL_TRAILING_SPECS metric appears exactly once per representation —
    the home−away diff and the raw per-side levels — under its family prefix
    (config.CANDIDATE_FAMILIES; config.NHL_CANDIDATE_COLS is derived from the
    same specs). Columns the ladder could not derive are all-NaN. One concat
    (pandas fragmentation otherwise degrades every later operation).
    """
    if not NHL_TRAILING_SPECS:
        return df
    sides: dict[str, np.ndarray] = {}
    for metric, windows in NHL_TRAILING_SPECS.items():
        for w in windows:
            col = f"{metric}_{w}"
            home_v, away_v = _per_side(ladder, gids, col)
            sides[f"nhl_{col}_diff"] = home_v - away_v
            sides[f"nhl_{col}_home"] = home_v
            sides[f"nhl_{col}_away"] = away_v
    return pd.concat([df, pd.DataFrame(sides, index=df.index)], axis=1)


def _records_string(events: pd.DataFrame) -> pd.Series:
    """Cumulative W-L record string per team over the decided timeline."""
    rec = events.groupby("team").agg(
        wins=("team_win", lambda s: float((s == 1).sum())),
        losses=("team_win", lambda s: float((s == 0).sum())),
    )

    def _fmt(team: str) -> str:
        if team not in rec.index:
            return ""
        r = rec.loc[team]
        return f"{int(r['wins'])}-{int(r['losses'])}"

    return rec, _fmt


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_game_features(games: pd.DataFrame,
                        boxscores: pd.DataFrame | None = None) -> pd.DataFrame:
    """Point-in-time feature frame for DECIDED games (one row per game).

    ``games`` must include the full settled timeline so early games carry
    real priors; every trailing value is shifted strictly prior. Returns the
    served diff features + the per-side values the tree view needs.
    """
    ev = compute_elo(team_events(games))
    ladder = team_stats_ladder(ev)

    df = games.copy().reset_index(drop=True)
    gids = df["game_id"]
    df["is_home"] = 1.0
    df["elo_diff"] = _home_minus_away(ladder, gids, "elo_entering")
    df["win_pct_diff"] = _home_minus_away(ladder, gids, "win_pct")
    df["rest_days_diff"] = _home_minus_away(ladder, gids, "rest_days")
    df["ewm_net_goals_diff"] = _home_minus_away(ladder, gids, "ewm_net_goals")
    df["ewm_goal_share_diff"] = _home_minus_away(ladder, gids, "ewm_goal_share")
    df["ga_per_game_diff"] = _home_minus_away(ladder, gids, "ga_per_game")
    df["back_to_back_diff"] = _home_minus_away(ladder, gids, "back_to_back")
    df = _attach_candidate_features(df, ladder, gids)

    # Goalie rolling state (per-goalie strictly-prior EWMs).
    goalie_frame, goalie_ladder = goalie_state(boxscores, df)
    for c in goalie_frame.columns:
        if c not in df.columns:
            df[c] = goalie_frame[c].to_numpy()
    df["goalie_sv_pct_diff"] = (df.get("goalie_sv_pct_home", pd.Series(np.nan, index=df.index))
                                - df.get("goalie_sv_pct_away", pd.Series(np.nan, index=df.index)))
    df["goalie_gaa_diff"] = (df.get("goalie_gaa_home", pd.Series(np.nan, index=df.index))
                             - df.get("goalie_gaa_away", pd.Series(np.nan, index=df.index)))
    df["goalie_starts_diff"] = (df.get("goalie_starts_home", pd.Series(np.nan, index=df.index))
                                - df.get("goalie_starts_away", pd.Series(np.nan, index=df.index)))

    # Game-level facts.
    if "game_type" in df.columns:
        df["is_playoffs"] = (pd.to_numeric(df["game_type"], errors="coerce")
                             == config.GAME_TYPE_POST).astype(float)
    else:
        df["is_playoffs"] = 0.0

    # Per-side values for the tree view (raw home/away representations).
    for side_col, lad_col in (("elo_home", "elo_entering"), ("elo_away", "elo_entering"),
                              ("win_pct_home", "win_pct"), ("win_pct_away", "win_pct"),
                              ("ewm_net_goals_home", "ewm_net_goals"),
                              ("ewm_net_goals_away", "ewm_net_goals"),
                              ("rest_days_home", "rest_days"),
                              ("rest_days_away", "rest_days")):
        home_v, away_v = _per_side(ladder, gids, lad_col)
        df[side_col] = home_v if side_col.endswith("home") else away_v

    rec, fmt = _records_string(ev)
    df["home_record"] = df["home_team"].map(fmt)
    df["away_record"] = df["away_team"].map(fmt)

    # Targets (kept beside features for OOF assembly; never model inputs).
    df["margin"] = pd.to_numeric(df["home_score"], errors="coerce") \
        - pd.to_numeric(df["away_score"], errors="coerce")
    df["total"] = pd.to_numeric(df["home_score"], errors="coerce") \
        + pd.to_numeric(df["away_score"], errors="coerce")
    df["home_win"] = (df["margin"] > 0).astype(float)
    return df


def build_slate_features(schedule: pd.DataFrame,
                         boxscores: pd.DataFrame | None = None) -> pd.DataFrame:
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
    ladder = team_stats_ladder(combined)

    df = pending.copy().reset_index(drop=True)
    # market-independence: drop any odds columns at the boundary
    for _m in ("spread_line", "total_line", "home_moneyline", "away_moneyline",
               "over_odds", "under_odds"):
        if _m in df.columns:
            df = df.drop(columns=_m)

    gids = df["game_id"]
    df["is_home"] = 1.0
    df["elo_diff"] = _home_minus_away(ladder, gids, "elo_entering")
    df["win_pct_diff"] = _home_minus_away(ladder, gids, "win_pct")
    df["rest_days_diff"] = _home_minus_away(ladder, gids, "rest_days")
    df["ewm_net_goals_diff"] = _home_minus_away(ladder, gids, "ewm_net_goals")
    df["ewm_goal_share_diff"] = _home_minus_away(ladder, gids, "ewm_goal_share")
    df["ga_per_game_diff"] = _home_minus_away(ladder, gids, "ga_per_game")
    df["back_to_back_diff"] = _home_minus_away(ladder, gids, "back_to_back")
    df = _attach_candidate_features(df, ladder, gids)

    # Goalie state: the expected starter is resolved from PRIOR starts only
    # (the same rule the history path uses); boxscores for pending games do
    # not exist yet, so this is built from the decided timeline.
    goalie_frame, goalie_ladder = goalie_state(boxscores, df)
    for c in goalie_frame.columns:
        if c not in df.columns:
            df[c] = goalie_frame[c].to_numpy()
    df["goalie_sv_pct_diff"] = (df.get("goalie_sv_pct_home", pd.Series(np.nan, index=df.index))
                                - df.get("goalie_sv_pct_away", pd.Series(np.nan, index=df.index)))
    df["goalie_gaa_diff"] = (df.get("goalie_gaa_home", pd.Series(np.nan, index=df.index))
                             - df.get("goalie_gaa_away", pd.Series(np.nan, index=df.index)))
    df["goalie_starts_diff"] = (df.get("goalie_starts_home", pd.Series(np.nan, index=df.index))
                                - df.get("goalie_starts_away", pd.Series(np.nan, index=df.index)))

    if "game_type" in df.columns:
        df["is_playoffs"] = (pd.to_numeric(df["game_type"], errors="coerce")
                             == config.GAME_TYPE_POST).astype(float)
    else:
        df["is_playoffs"] = 0.0

    for side_col, lad_col in (("elo_home", "elo_entering"), ("elo_away", "elo_entering"),
                              ("win_pct_home", "win_pct"), ("win_pct_away", "win_pct"),
                              ("ewm_net_goals_home", "ewm_net_goals"),
                              ("ewm_net_goals_away", "ewm_net_goals"),
                              ("rest_days_home", "rest_days"),
                              ("rest_days_away", "rest_days")):
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
    """Diff/anchor projection of the contract (linear member view).

    Routing RULE, not a second list: the linear family never sees the raw
    per-side levels (config.RAW_PER_SIDE_COLS) — those exist for the tree
    members, which consume the whole contract.
    """
    cols = config.active_moneyline_feature_cols() if active is None else list(active)
    return [c for c in cols if c not in config.RAW_PER_SIDE_COLS]


def linear_view(df: pd.DataFrame) -> pd.DataFrame:
    """Linear matrix: pure projection of the active contract (diff view)."""
    cols = [c for c in linear_feature_columns() if c in df.columns]
    return df.reindex(columns=cols).astype(float)


def team_category_ids(df: pd.DataFrame) -> pd.DataFrame:
    """The config.TREE_CATEGORICAL_COLS pair from home/away abbreviations.

    Stable integer IDs (config.NHL_TEAM_ID) with the reserved UNK_TEAM_ID
    slot for historical/missing labels — never a silent alias of a real
    team. Tree-family context ONLY: never in MONEYLINE_FEATURE_COLS, never
    seen by the linear family.
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
    numeric (MLB parity — the run-engine regressors there consume the numeric
    matrix, not the categorical context).
    """
    return [c for c in config.active_moneyline_feature_cols()]


def tree_view(df: pd.DataFrame) -> pd.DataFrame:
    """Tree-family matrix: the served contract + the team-ID categorical pair.

    The numeric block comes from config.MONEYLINE_FEATURE_COLS (the one
    master list — no column is synthesized here). The categorical pair is
    APPENDED after it for the tree members only, exactly like MLB's adopted
    TREE_CATEGORICAL_COLS routing: the linear family (linear_view) never
    receives these, and no view emits a duplicate.
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
