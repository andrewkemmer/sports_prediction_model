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


def _trailing_pooled_ratio(srt: pd.DataFrame, num_col: str, den_col: str,
                           window: int) -> np.ndarray:
    """Strictly-prior ``sum(num) / sum(den)`` over each team's last ``window``.

    Pooling the COUNTS rather than averaging per-game rates is both the
    statistically correct aggregate (volume-weighted) and the one that
    survives a zero-denominator game: averaging per-game rates drops any game
    where the team had no opportunity, so a team whose recent games happened
    to contain one such game got an empty window and a null feature. Here that
    game contributes 0 to both sums and leaves the window intact.

    A window whose total opportunities are still zero stays NaN — there is
    genuinely no conversion rate to report, and that is honest.
    """
    def _rolled(col: str) -> pd.Series:
        return (srt.groupby("team", sort=False)[col]
                .rolling(window, min_periods=1).sum()
                .groupby(level=0).shift(1)
                .reset_index(level=0, drop=True))
    num = pd.to_numeric(_rolled(num_col), errors="coerce")
    den = pd.to_numeric(_rolled(den_col), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den.replace(0, np.nan)
    return out.to_numpy(float)


# Per-game metrics whose trailing value is a ratio and must therefore be
# POOLED over the window instead of averaged across games. The per-game
# ``pp_success_rate`` is undefined whenever a team had no power play, which
# made the rolling mean of it a hole rather than a rate.
NHL_POOLED_RATIO_SPECS: dict[str, tuple[str, str]] = {
    "pp_success_rate": ("pp_goals_pg", "pp_attempts_pg"),
}


# Trailing-window spec for the candidate-pool metrics: per-game metric -> the
# ladder suffixes it is served at. DECLARED ONCE in config
# (TEAM_CANDIDATE_TRAILING_SPECS — the same declaration that names the RFE
# candidates) and pulled here; every derivation rides the shared
# _trailing_ewm / _trailing_per_team primitives so the shift(1) leakage
# discipline is inherited, not reimplemented.
NHL_TRAILING_SPECS: dict[str, tuple[str, ...]] = {
    **config.TEAM_CANDIDATE_TRAILING_SPECS,
}

# Served base features (config.MONEYLINE_FEATURE_COLS) that are a trailing
# flat window over a boxscore per-team metric. Maps the contract name to the
# ladder column it reads; both ride _trailing_per_team with shift(1).
NHL_SERVED_ROLL_SPECS: dict[str, str] = {
    "shots_for_per_game_diff": "sog_pg",
    "shots_against_per_game_diff": "shots_against_pg",
    "pp_success_diff": "pp_success_rate",
    "faceoff_win_diff": "faceoff_win_pct",
}
NHL_SERVED_ROLL_METRICS: tuple[str, ...] = tuple(
    dict.fromkeys(NHL_SERVED_ROLL_SPECS.values()))


# Boxscore rollup column -> the ladder metric it feeds. The candidate pool
# (TEAM_CANDIDATE_TRAILING_SPECS) names per-game metrics like ``sog_pg``;
# ingestion emits ONE wide row per game with ``home_``/``away_`` prefixed
# columns. This table is the ONLY place that unwraps the prefixes, so the
# ladder keeps consuming plain per-team metric names. ``opp_sog`` resolves to
# the OPPONENT's SOG, which is the team's true shots-against total (a single
# goalie line misses relief appearances).
BOXSCORE_TEAM_METRICS: dict[str, str] = {
    "sog": "sog_pg",
    "opp_sog": "shots_against_pg",
    "pp_goals": "pp_goals_pg",
    "pp_opportunities": "pp_attempts_pg",
    "faceoff_pct": "faceoff_win_pct",
    "pim": "pim_pg",
    "hits": "hits_pg",
    "blocked": "blocked_shots_pg",
    "giveaways": "giveaways_pg",
    "takeaways": "takeaways_pg",
}


def team_game_rollup(games: pd.DataFrame,
                     boxscores: pd.DataFrame | None) -> pd.DataFrame:
    """The per-(game_id, team) boxscore aggregate the ladder merges in.

    ``boxscores`` is one WIDE row per game (``home_sog``/``away_sog``/...);
    this unpivots it to one row per (game_id, team) under the plain metric
    names the ladder's trailing specs declare. Goals for/against come from
    the settled ``games`` frame (already loaded), not the boxscore, so the
    rollup carries only boxscore-sourced facts plus the opponent's SOG
    (that IS ``shots against`` — the team's own goalie line records shots
    faced by ONE goalie, which is not the team's total when a relief
    appearance happens).

    Returns an empty frame (not None) when there is no boxscore coverage, so
    every metric degrades to all-NaN rather than raising.
    """
    if boxscores is None or len(boxscores) == 0 or "game_id" not in boxscores.columns:
        return pd.DataFrame(columns=["game_id", "team"])
    bs = boxscores.copy()
    bs["game_id"] = bs["game_id"].astype(str)
    # Last write wins; the loader concatenates one row per game already.
    bs = bs.drop_duplicates(subset=["game_id"], keep="last")

    frames: list[pd.DataFrame] = []
    for side in ("home", "away"):
        cols: dict[str, pd.Series] = {"game_id": bs["game_id"]}
        for src, metric in BOXSCORE_TEAM_METRICS.items():
            if src == "opp_sog":
                other = "away" if side == "home" else "home"
                col = f"{other}_sog"
            else:
                col = f"{side}_{src}"
            if col in bs.columns:
                cols[metric] = pd.to_numeric(bs[col], errors="coerce")
            else:
                cols[metric] = np.nan
        long = pd.DataFrame(cols)
        long["team"] = games.set_index(games["game_id"].astype(str)) \
            .reindex(long["game_id"])[f"{side}_team"].to_numpy() \
            if f"{side}_team" in games.columns else None
        frames.append(long)
    roll = pd.concat(frames, ignore_index=True)
    roll = roll.dropna(subset=["team"])
    # Per-game power-play rate: goals / opportunities. A game with zero PP
    # opportunities has NO conversion rate (you cannot convert zero chances),
    # so it stays undefined rather than being scored as a 0% conversion. The
    # SERVED trailing feature does not average these per-game rates; it pools
    # the counts (NHL_POOLED_RATIO_SPECS), so a zero-opportunity game costs
    # nothing instead of voiding the whole window.
    if "pp_attempts_pg" in roll.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = roll["pp_goals_pg"] / roll["pp_attempts_pg"].replace(0, np.nan)
        roll["pp_success_rate"] = rate.where(np.isfinite(rate))
    return roll.reset_index(drop=True)


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
    # ``goal_diff_pg`` is the one candidate whose per-game value is already on
    # the events frame under a different name; alias it here so the declared
    # spec finds a source (it was otherwise a permanently all-NaN candidate
    # silently occupying a slot in the RFE trial space).
    if "goal_diff_pg" not in srt.columns and "net_from_team" in srt.columns:
        srt["goal_diff_pg"] = pd.to_numeric(srt["net_from_team"], errors="coerce")
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

    # The four boxscore-derived diffs named in the MONEYLINE contract are
    # served as a trailing flat window (manifest: rolling(5) of the per-game
    # value, strictly prior). They ride the SAME primitives, so the shift(1)
    # discipline is inherited rather than reimplemented. Declared separately
    # from NHL_TRAILING_SPECS so the RFE trial space is not widened.
    for metric in NHL_SERVED_ROLL_METRICS:
        col = f"{metric}_roll"
        num, den = NHL_POOLED_RATIO_SPECS.get(metric, (None, None))
        if num is not None and num in srt.columns and den in srt.columns:
            srt[col] = _trailing_pooled_ratio(srt, num, den, config.PBP_ROLL_WINDOW)
            continue
        if col in srt.columns:
            continue  # already derived by the candidate loop above
        srt[col] = _trailing_per_team(srt, metric, config.PBP_ROLL_WINDOW) \
            if metric in srt.columns else np.nan
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
                 games: pd.DataFrame,
                 team_source: pd.DataFrame | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-goalie rolling SV% / GAA state + per-game expected-starter frame.

    ``boxscores`` carries one row per game (the ingestion rollup: decision
    goalie id/name, per-goalie goals/shots against, and TOI). The per-start
    save fraction for a goalie = 1 - (goals against / shots against), and
    per-start GAA = goals against / (TOI minutes / 60).

    Every goalie statistic is shifted strictly prior on the GOALIE's own
    start timeline, then EWM'd (halflife = EWM_HALFLIFE starts). The
    EXPECTED STARTER entering game t is the goalie with the most STARTS for
    that team strictly before t (season-to-date workload) — resolved purely
    from prior starts, never from game t's own boxscore, so the same rule
    serves decided history and a scheduled slate alike.

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
    # The gameday lookup MUST span every boxscored game, not just the rows
    # being emitted. ``build_slate_features`` emits only pending games, so a
    # map built from ``games`` alone would date no historical start at all —
    # every start would be NaT, the strictly-prior scan would find nothing,
    # and the whole goalie family would silently serve null. ``team_source``
    # is the full schedule there, so union it over ``games``.
    gdate: dict[str, object] = {}
    for _src in (team_source, games):
        if _src is not None and "gameday" in getattr(_src, "columns", ()):
            for _g, _d in zip(_src["game_id"].astype(str), _src["gameday"]):
                gdate.setdefault(_g, _d)
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
    # ``team_source`` must cover EVERY game in ``boxscores``, not just the rows
    # being emitted: the slate builder emits only pending games but is fed the
    # whole decided boxscore set, so resolving the map from the emitted frame
    # alone would leave every historical start keyed 'home'/'away' and the
    # expected-starter lookup would find nothing.
    _tsrc = team_source if team_source is not None else games
    teams_by_game: dict[tuple[str, str], str] = {}
    if "home_team" in _tsrc.columns:
        for r in _tsrc.itertuples(index=False):
            gid = str(getattr(r, "game_id", ""))
            teams_by_game[(gid, "home")] = str(getattr(r, "home_team", ""))
            teams_by_game[(gid, "away")] = str(getattr(r, "away_team", ""))
    st["team_abbr"] = [teams_by_game.get((str(g), side), side)
                       for g, side in zip(st["game_id"], st["team"])]
    st.loc[st["team_abbr"].isin(["home", "away"]), "team_abbr"] = np.nan

    # Per-start save fraction + GAA (per 60 min).
    st["save_fraction"] = 1.0 - (st["goals_against"] / st["shots_faced"].replace(0, np.nan))
    st["gaa_game"] = st["goals_against"] / (st["toi_min"] / 60.0).replace(0, np.nan)
    # Only a genuine start (TOI >= MIN_GOALIE_TOI_MINUTES) counts toward
    # workload; a short relief appearance is neither a start nor evidence of
    # starter status.
    st["is_start"] = (pd.to_numeric(st["toi_min"], errors="coerce")
                      >= config.MIN_GOALIE_TOI_MINUTES)

    st = st.sort_values(["goalie_id", "gameday", "game_id"]).reset_index(drop=True)
    st["prior_starts"] = st.groupby("goalie_id").cumcount()
    st["sv_pct"] = st.groupby("goalie_id", sort=False)["save_fraction"].transform(
        lambda s: s.ewm(halflife=config.EWM_HALFLIFE, min_periods=1).mean().shift(1)
    )
    st["gaa"] = st.groupby("goalie_id", sort=False)["gaa_game"].transform(
        lambda s: s.ewm(halflife=config.EWM_HALFLIFE, min_periods=1).mean().shift(1)
    )
    # Quality ENTERING THE NEXT game: the same EWM over this goalie's starts
    # THROUGH this one. Used to report the expected starter of a LATER game so
    # his most recent start stays inside the window (the ``shift(1)`` columns
    # above would drop it).
    st["sv_pct_incl"] = st.groupby("goalie_id", sort=False)["save_fraction"].transform(
        lambda s: s.ewm(halflife=config.EWM_HALFLIFE, min_periods=1).mean())
    st["gaa_incl"] = st.groupby("goalie_id", sort=False)["gaa_game"].transform(
        lambda s: s.ewm(halflife=config.EWM_HALFLIFE, min_periods=1).mean())

    # ---------------------------------------------------------------------
    # EXPECTED STARTER, resolved from STRICTLY-PRIOR workload.
    #
    # The expected starter entering game t is the goalie with the most starts
    # for that team BEFORE t — nothing else. This is deliberately NOT the
    # decision goalie recorded in game t's own boxscore: that identity is only
    # knowable after the game, so selecting on it leaked current-game
    # information into a pre-game feature, AND it could not resolve at all for
    # a scheduled game (no boxscore exists yet), which left every goalie
    # feature 100% null on the slate the model actually predicts.
    #
    # Only genuine starts count (TOI >= MIN_GOALIE_TOI_MINUTES): a short
    # relief appearance is not a start and must not win the workload vote.
    # Ties break on the most recent start, then goalie id, so the choice is
    # deterministic.
    # ---------------------------------------------------------------------
    _starts = st[st["is_start"]].copy()
    g_dates: dict[str, np.ndarray] = {}
    g_sv: dict[str, np.ndarray] = {}
    g_gaa: dict[str, np.ndarray] = {}
    g_name: dict[str, str] = {}
    for gid_, grp in _starts.groupby("goalie_id", sort=False):
        grp = grp.sort_values(["gameday", "game_id"])
        g_dates[gid_] = grp["gameday"].to_numpy(dtype="datetime64[ns]")
        g_sv[gid_] = pd.to_numeric(grp["sv_pct_incl"], errors="coerce").to_numpy(float)
        g_gaa[gid_] = pd.to_numeric(grp["gaa_incl"], errors="coerce").to_numpy(float)
        nm = [str(x or "") for x in grp["goalie_name"]]
        g_name[gid_] = nm[-1] if nm else ""
    # team -> (gameday, goalie_id) sorted, for the candidate scan
    team_goals: dict[str, list[tuple[np.datetime64, str]]] = {}
    for r in _starts.sort_values(["gameday", "game_id"]).itertuples(index=False):
        team_goals.setdefault(str(r.team_abbr), []).append(
            (pd.Timestamp(r.gameday).to_datetime64(), str(r.goalie_id)))

    def _expected(team: str, when) -> tuple[float, float, float, str]:
        """(sv_pct, gaa, prior_starts, name) of the expected starter entering
        ``when`` for ``team`` — from strictly-earlier starts only."""
        entries = team_goals.get(str(team) or "")
        if not entries or when is None or pd.isna(when):
            return np.nan, np.nan, np.nan, ""
        cut = pd.Timestamp(when).to_datetime64()
        best = None
        for day, gid_ in entries:
            if day >= cut:            # strictly prior only
                continue
            dates = g_dates.get(gid_)
            if dates is None:
                continue
            n_before = int(np.searchsorted(dates, cut, side="left"))
            if n_before <= 0:
                continue
            key = (n_before, day, gid_)
            if best is None or key > best[0]:
                best = (key, gid_, n_before)
        if best is None:
            return np.nan, np.nan, np.nan, ""
        _, gid_, n_before = best
        return (float(g_sv[gid_][n_before - 1]), float(g_gaa[gid_][n_before - 1]),
                float(n_before), g_name.get(gid_, ""))

    ladder_rows = []
    for side in ("home", "away"):
        sv, gaa, starts_n, names = [], [], [], []
        for r in games.itertuples(index=False):
            gid = str(getattr(r, "game_id", ""))
            team = str(getattr(r, f"{side}_team", "") or "")
            when = pd.to_datetime(getattr(r, "gameday", None), errors="coerce")
            s_, g_, n_, nm_ = _expected(team, when)
            sv.append(s_); gaa.append(g_); starts_n.append(n_); names.append(nm_)
            if pd.notna(n_):
                ladder_rows.append({"game_id": gid, "team": team,
                                    "goalie_starts": n_})
        per_game[f"goalie_sv_pct_{side}"] = sv
        per_game[f"goalie_gaa_{side}"] = gaa
        per_game[f"goalie_starts_{side}"] = starts_n
        per_game[f"g_{side}_name"] = names
        per_game[f"g_{side}_sv_pct"] = per_game[f"goalie_sv_pct_{side}"]
        per_game[f"g_{side}_gaa"] = per_game[f"goalie_gaa_{side}"]
        per_game[f"g_{side}_starts"] = per_game[f"goalie_starts_{side}"]
    ladder = pd.DataFrame(ladder_rows, columns=["game_id", "team", "goalie_starts"])
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
    ladder = team_stats_ladder(ev, team_game_rollup(games, boxscores))

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
    for served, metric in NHL_SERVED_ROLL_SPECS.items():
        df[served] = _home_minus_away(ladder, gids, f"{metric}_roll")
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
    ladder = team_stats_ladder(combined, team_game_rollup(sched, boxscores))

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
    for served, metric in NHL_SERVED_ROLL_SPECS.items():
        df[served] = _home_minus_away(ladder, gids, f"{metric}_roll")
    df = _attach_candidate_features(df, ladder, gids)

    # Goalie state: the expected starter is resolved from PRIOR starts only
    # (the same rule the history path uses); boxscores for pending games do
    # not exist yet, so this is built from the decided timeline. ``sched`` is
    # the team-map source because it covers every boxscored game, not just the
    # pending rows being emitted.
    goalie_frame, goalie_ladder = goalie_state(boxscores, df, sched)
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
