"""Fitted two-sided NHL team rating: goals for and against per 60.

The previous arm C was arithmetic on columns the moneyline model already
served: a weighted mean of the eight ``pl_evo_*``/``pl_ppo_*`` cells plus a
goalie term.  Gradient-boosted trees can form any monotone combination of
columns they already have, so that arm could not add information — measured,
it moved the pooled logloss by 0.00009.

This module supplies the information that was missing: an explicit
opponent-adjusted team effect.  The unit is a side row (two per game, one per
team scoring) and the target is that team's goals per 60 minutes of team ice
time — a regulation game is 60 minutes, so this is goals per game with the
unit named honestly, and a ``toi`` column is honoured when a frame carries
one.  A team-offence one-hot and a team-defence one-hot sit on the same row,
so each coefficient is estimated net of the opponents actually faced.

The design is MLB's ``team_run_ratings`` translated to hockey, with the target
changed from runs per game to goals per 60.  Three nuisance blocks sit beside
the team one-hots:

``components``
    The club's own offence form — the NHL counterparts of MLB's wOBA / ISO /
    K / BB / barrel / hard-hit / lineup counters — plus the healthy-pool XPM
    mean (``pl_xpm_{side}``), the analogue of MLB's ``off_pool_xwoba``.
    Skater and team-offence columns are therefore *not* excluded: MLB's fit
    reads its served offence columns too, and the earlier exclusion was the
    one design difference left between the two models.

``opp_goalie``
    The goaltender the scoring team actually faces (the *opposing* side's
    ``goalie_sv_pct`` / ``goalie_gaa`` / ``goalie_starts``).  This is the
    structural twin of MLB's opposing starter/bullpen block: it stops the
    team-defence coefficient from absorbing whichever goalie happened to be in
    net, so ``def_rating`` measures the skaters and system in front of him.

``prior``
    Shrunk prior-season goals for and against per club (20-game prior toward
    that season's league mean), carried for both the rated team and its
    opponent.  This is the April cold-start bridge: a 240-day half-life alone
    reaches back only part of an offseason.

The ``own_goalie`` knob is the mirror of MLB's ``own_starter``: off by
default, so the shipped ``def_rating`` is total goal prevention including the
club's own goaltending.  ``likelihood`` mirrors MLB's family switch
(``gaussian`` shipped, ``poisson`` available), and ``MIN_TRAIN_GAMES`` matches
MLB's 200.

``blocks=()`` reproduces the original thin design (team dummies plus rest,
back-to-back and playoff context), which is what the first NHL A/B measured.

Exports ``off_rating_{home,away,diff}``, ``def_rating_{home,away,diff}``,
``net_rating_diff`` and the game-level ``net_rating_sum`` in goals per game
(identical to goals per 60 at regulation ice time), signed higher-is-better,
with ``net_rating_diff == off_rating_diff + def_rating_diff`` and
``net_rating_sum == net_home + net_away``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

#: Ridge penalty on standardised numerics / raw 0-1 dummies.
ALPHA = 50.0
#: Refit cadence for the as-of panel, in distinct game dates.
REFIT_DAYS = 7
#: Recency half-life for the training weights.  240 days shipped with the
#: 2026-11-24 G3S240 adoption (120 days was the trial default).
HALF_LIFE_DAYS = 240.0
#: A fit needs this many decided games before its ratings are published.
#: MLB's ``team_run_ratings`` uses the same 200.
MIN_TRAIN_GAMES = 200
#: Own-offence components (base name; the side suffix is added per row).
#: MLB's counterparts are ``woba_30g``/``team_iso_30g``/``team_k_rate_30g``/
#: ``team_bb_rate_30g``/``team_barrel_15g``/``team_hardhit_15g``/
#: ``lineup_re24_mean``/``lineup_ops_vs_starter_hand``.
OFF_COMPONENTS = (
    "ewm_net_goals",
    "ewm_goal_share",
    "shots_for_per_game",
    "pp_success",
    "nhl_pp_goals_pg_ewm",
    "faceoff_win",
    "nhl_giveaways_pg_ewm",
    "nhl_takeaways_pg_ewm",
)
#: Opposing-goalie block (base name; the *opposing* side suffix is added per
#: row).  The twin of MLB's opposing starter/bullpen block.
OPP_GOALIE_COMPONENTS = (
    "goalie_sv_pct",
    "goalie_gaa",
    "goalie_starts",
)
#: The club's OWN goaltender.  Not in the design by default, exactly like
#: MLB's ``own_starter``: the shipped ``def_rating`` is total prevention, and
#: turning this on makes it prevention net of the goalie on the ice.
OWN_GOALIE_COMPONENTS = (
    "goalie_sv_pct",
    "goalie_gaa",
    "goalie_starts",
)
#: Chance-quality block: the club's own 5v5 xG generation and the opponent's
#: 5v5 xG prevention, both trailing 15-game means (``xg_for_15g_{side}`` and
#: ``xg_against_15g_{side}`` in the frame).  MLB's counterpart is the contact
#: quality block -- wOBA / ISO / barrel / hard-hit -- which says how the runs
#: were earned rather than only how many.  Available as a block but NOT in the
#: default design until it has been measured against the goal-only version.
CHANCE_COMPONENTS = ("xg_for_15g", "xg_against_15g")
#: Which nuisance blocks sit beside the team one-hots.  The default is all
#: three, i.e. the configuration that matches MLB's ``team_run_ratings``;
#: ``blocks=()`` is the thin design measured in the first NHL A/B.
DESIGN_BLOCKS = ("components", "opp_goalie", "prior")
#: Residual family.  ``poisson`` is the correctly-specified one for goal
#: counts; ``gaussian`` is what the A/B harness measures.
LIKELIHOOD = "gaussian"
#: Dependent variable.  ``goals`` is the shipped target: the raw count, which
#: is MLB's exact form (runs scored in the game as played) and the quantity the
#: moneyline ultimately prices.  ``goals_per_60`` is the same number at
#: regulation length and is the form that survives a real TOI column; the two
#: were measured identical (panel max |diff| 0.0, OOF max |diff| 8e-17), so the
#: raw count is the default for literal parity rather than for any numeric gain.
TARGET = "goals"
TARGETS = ("goals_per_60", "goals")
#: Whether the own-goalie block is part of the default design (it is not).
OWN_GOALIE = False
#: Prior-season shrinkage, in games of that season.
PRIOR_SHRINK_GAMES = 20.0
#: A season starts in July; earlier months keep the previous season label.
SEASON_START_MONTH = 7
#: Regulation ice time.  Overridden by a served TOI column when one exists.
REGULATION_MINUTES = 60.0
TOI_COLS = ("toi_minutes", "time_on_ice", "ice_time")
#: Position pools for the fallback pool mean when the rolled-up XPM column is
#: absent.  Production serves ``pl_xpm_{side}``; research frames carry cells.
POOLS = ("c", "l", "r", "d")

GOAL_RATING_COLS = [
    "off_rating_diff", "off_rating_home", "off_rating_away",
    "def_rating_diff", "def_rating_home", "def_rating_away",
    "net_rating_diff", "net_rating_sum",
]
#: Side levels: routed tree-only by the caller, never into the linear member.
SIDE_GOAL_RATING_COLS = [
    "off_rating_home", "off_rating_away",
    "def_rating_home", "def_rating_away",
]

CONTEXT = ("rest_days", "back_to_back")
#: Columns served once per GAME rather than once per side.
GAME_LEVEL = ("is_playoffs",)


def _context(games: pd.DataFrame, name: str, side: str) -> pd.Series:
    """Read a context column, honouring the per-side vs per-game split."""
    column = name if name in GAME_LEVEL else f"{name}_{side}"
    return _num(games, column)


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    """Numeric column or an all-NaN series, never a KeyError."""
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _resolve_gameday(games: pd.DataFrame) -> pd.Series:
    """The frame's date column as datetime.

    Research frames name it ``gameday``; the production delivery frame names
    it ``game_date``.  Accept both, exactly like MLB's runner.
    """
    column = "gameday" if "gameday" in games.columns else "game_date"
    if column not in games.columns:
        raise ValueError("team_goal_ratings: frame has no gameday/game_date")
    return pd.to_datetime(games[column], errors="coerce")


def _score_column(games: pd.DataFrame, side: str) -> pd.Series:
    for name in (f"{side}_score", f"{side}_goals", f"{side}_pts"):
        if name in games.columns:
            return pd.to_numeric(games[name], errors="coerce")
    raise ValueError(f"team_goal_ratings: no score column for {side}")


def _team_minutes(games: pd.DataFrame, side: str) -> pd.Series:
    """Team ice time in minutes, defaulting to a regulation 60."""
    for name in TOI_COLS:
        column = f"{name}_{side}"
        if column not in games.columns:
            continue
        values = pd.to_numeric(games[column], errors="coerce")
        values = values.where((values >= 30.0) & (values <= 90.0))
        if values.notna().sum() >= max(1, int(0.25 * len(values))):
            # Fill with regulation, never with the column median: a median over
            # the whole frame would carry future games into a past row's
            # target denominator, which is a (latent) look-ahead leak.
            return values.fillna(REGULATION_MINUTES)
    return pd.Series(REGULATION_MINUTES, index=games.index, dtype=float)


def _pool_xpm(games: pd.DataFrame, side: str) -> pd.Series:
    """The healthy-pool XPM mean: the analogue of MLB's ``off_pool_xwoba``.

    Production serves ``pl_xpm_{side}`` (the adopted position-free rollup);
    research frames may instead carry the four C/L/R/D cells, which are
    averaged unweighted here with the same ``skipna=False`` rule the rollup
    uses.
    """
    rolled = _num(games, f"pl_xpm_{side}")
    if rolled.notna().any():
        return rolled
    cells = [f"pl_xpm_{p}_{side}" for p in POOLS]
    if all(c in games.columns for c in cells):
        return pd.concat([pd.to_numeric(games[c], errors="coerce")
                          for c in cells], axis=1).mean(axis=1, skipna=False)
    return rolled


def _resolve_season(games: pd.DataFrame, gameday: pd.Series) -> pd.Series:
    """Season label, taken from the frame or derived from the date.

    The reviewed frame carries an explicit October-to-June season label; the
    fallback derives the same label with a July cutoff, so a January game
    inherits the season that started the previous autumn.
    """
    season = _num(games, "season")
    derived = (gameday.dt.year
               - (gameday.dt.month < SEASON_START_MONTH).astype(int)).astype(float)
    if season.isna().all():
        return derived
    return season.fillna(derived)


def _prior_goals(games: pd.DataFrame) -> pd.DataFrame:
    """Shrunk prior-season goals for/against per (season, team).

    ``own_p_gf``/``own_p_ga`` for season S come from S-1 and are shrunk toward
    that season's league mean with a 20-game prior, so a club with a short
    history is pulled to average rather than trusted.  Only completed earlier
    seasons enter, so no row can see its own season, let alone its own game.
    """
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        parts.append(pd.DataFrame({
            "season": games["season"].to_numpy(),
            "team": games[f"{side}_team"].to_numpy(),
            "gf": _score_column(games, side).to_numpy(float),
            "ga": _score_column(games, other).to_numpy(float),
        }))
    long = pd.concat(parts, ignore_index=True).dropna(subset=["gf", "ga"])
    if long.empty:
        return pd.DataFrame(columns=["season", "team", "own_p_gf", "own_p_ga"])
    team = (long.groupby(["season", "team"], as_index=False)
            .agg(gf=("gf", "mean"), ga=("ga", "mean"), n=("gf", "size")))
    league = long.groupby("season", as_index=False).agg(
        gf_lg=("gf", "mean"), ga_lg=("ga", "mean"))
    team = team.merge(league, on="season", how="left")
    k = PRIOR_SHRINK_GAMES
    team["own_p_gf"] = ((team["n"] * team["gf"] + k * team["gf_lg"])
                        / (team["n"] + k))
    team["own_p_ga"] = ((team["n"] * team["ga"] + k * team["ga_lg"])
                        / (team["n"] + k))
    team["season"] = team["season"] + 1.0
    return team[["season", "team", "own_p_gf", "own_p_ga"]]


def build_side_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game: one per team scoring."""
    g = games.reset_index(drop=True)
    g["gameday"] = _resolve_gameday(g)
    g["season"] = _resolve_season(g, g["gameday"])
    prior = _prior_goals(g)
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        goals = _score_column(g, side)
        minutes = _team_minutes(g, side)
        part = pd.DataFrame({
            "pos": g.index.to_numpy(),
            "gameday": g["gameday"].to_numpy(),
            "season": g["season"].to_numpy(),
            "team": g[f"{side}_team"].to_numpy(),
            "opp": g[f"{other}_team"].to_numpy(),
            "is_home": float(side == "home"),
            "minutes": minutes.to_numpy(float),
            "goals": goals.to_numpy(float),
            # Goals per 60 minutes of team ice time.
            "rating": 60.0 * goals.to_numpy(float) / minutes.to_numpy(float),
        })
        for name in OFF_COMPONENTS:
            part[f"off_{name}"] = _num(g, f"{name}_{side}").to_numpy()
        part["off_pool_xpm"] = _pool_xpm(g, side).to_numpy()
        for name in OPP_GOALIE_COMPONENTS:
            part[f"opp_{name}"] = _num(g, f"{name}_{other}").to_numpy()
        # Chance quality: own xG for, opponent's xG against (i.e. the chance
        # prevention the side actually faces).
        part["off_xg_for_15g"] = _num(g, f"xg_for_15g_{side}").to_numpy()
        part["opp_xg_against_15g"] = _num(
            g, f"xg_against_15g_{other}").to_numpy()
        for name in OWN_GOALIE_COMPONENTS:
            part[f"own_{name}"] = _num(g, f"{name}_{side}").to_numpy()
        for name in CONTEXT + GAME_LEVEL:
            part[name] = _context(g, name, side).to_numpy()
        if not prior.empty:
            part = part.merge(prior, on=["season", "team"], how="left")
            mirror = prior.rename(columns={
                "team": "opp", "own_p_gf": "opp_p_gf",
                "own_p_ga": "opp_p_ga"})
            part = part.merge(mirror, on=["season", "opp"], how="left")
        else:
            for column in ("own_p_gf", "own_p_ga", "opp_p_gf", "opp_p_ga"):
                part[column] = np.nan
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _numeric_columns(rows: pd.DataFrame, blocks=DESIGN_BLOCKS,
                     own_goalie: bool = OWN_GOALIE) -> list[str]:
    """The nuisance design, in a fixed order.  Team dummies are added later."""
    blocks = tuple(blocks)
    cols: list[str] = []
    if "components" in blocks:
        cols.extend(f"off_{name}" for name in OFF_COMPONENTS)
        cols.append("off_pool_xpm")
    if "opp_goalie" in blocks:
        cols.extend(f"opp_{name}" for name in OPP_GOALIE_COMPONENTS)
    if "chance" in blocks:
        # Own chance generation and the OPPONENT's chance prevention, the
        # structural twin of MLB's own contact block and opposing-pitcher
        # block: the side row already carries both sides' columns.
        cols.append("off_xg_for_15g")
        cols.append("opp_xg_against_15g")
    if "prior" in blocks:
        cols.extend(["own_p_gf", "own_p_ga", "opp_p_gf", "opp_p_ga"])
    # Context is schedule here rather than MLB's park/weather, and is always
    # on: it is part of every arm the A/B has ever measured.
    cols.extend(list(CONTEXT) + list(GAME_LEVEL))
    if own_goalie:
        cols.extend(f"own_{name}" for name in OWN_GOALIE_COMPONENTS)
    cols.append("is_home")
    return [c for c in cols if c in rows.columns]


def _design(rows: pd.DataFrame, teams: list[str],
            numeric: list[str]) -> pd.DataFrame:
    dummies = pd.DataFrame(index=rows.index)
    for team in teams:
        dummies[f"toff_{team}"] = (rows["team"] == team).astype(float)
    for team in teams:
        dummies[f"tdef_{team}"] = (rows["opp"] == team).astype(float)
    return pd.concat([dummies, rows[numeric]], axis=1)


def _fit_once(train: pd.DataFrame, teams: list[str], anchor: pd.Timestamp,
              alpha: float, half_life_days: float,
              blocks=DESIGN_BLOCKS, likelihood: str = LIKELIHOOD,
              own_goalie: bool = OWN_GOALIE,
              target: str = TARGET) -> dict[str, float]:
    """One weighted fit.  Refuses to look at or past ``anchor``."""
    if target not in TARGETS:
        raise ValueError(f"team_goal_ratings: unknown target {target!r}")
    column = "rating" if target == "goals_per_60" else "goals"
    if (train["gameday"] >= anchor).any():
        raise ValueError("team_goal_ratings: training row on or after the anchor")
    numeric = _numeric_columns(train, blocks, own_goalie)
    train = train.dropna(subset=[column])
    train = train[np.isfinite(train[column].to_numpy(float))]
    if train.empty:
        raise ValueError("team_goal_ratings: no decided rows to fit")
    design = _design(train, teams, numeric)
    med = design[numeric].median()
    design[numeric] = design[numeric].fillna(med).fillna(0.0)
    mu = design[numeric].mean()
    sd = design[numeric].std().replace(0.0, 1.0).fillna(1.0)
    design[numeric] = (design[numeric] - mu) / sd
    age = (anchor - train["gameday"]).dt.days.to_numpy(float)
    weight = np.power(0.5, age / float(half_life_days))
    y = train[column].to_numpy(float)
    X = design.to_numpy(float)
    if likelihood == "poisson":
        # A Poisson family belongs on goal COUNTS with a log(minutes/60)
        # offset.  Every frame the repo builds is regulation ice time, where
        # the offset is identically zero, so the rate target is the count
        # target.  A frame that really carries TOI is refused rather than
        # silently fitted without its offset.
        if "minutes" in train.columns and not np.allclose(
                train["minutes"].to_numpy(float), REGULATION_MINUTES):
            raise ValueError(
                "team_goal_ratings: poisson needs the log(minutes/60) offset, "
                "which this fit cannot express")
        # PoissonRegressor minimises (1/2n) * deviance + alpha * ||w||^2 while
        # Ridge minimises ||y - Xw||^2 + alpha * ||w||^2, so the same alpha is
        # 2n times stronger here; rescale so a sweep point means the same
        # thing in both families.
        from sklearn.linear_model import PoissonRegressor
        model = PoissonRegressor(alpha=float(alpha) / (2.0 * len(y)),
                                 fit_intercept=True, max_iter=400)
        model.fit(X, y, sample_weight=weight)
    elif likelihood == "gaussian":
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X, y, sample_weight=weight)
    else:
        raise ValueError(f"team_goal_ratings: unknown likelihood {likelihood!r}")
    return dict(zip(design.columns, model.coef_))


def build_causal_panel(games: pd.DataFrame, alpha: float = ALPHA,
                       refit_days: int = REFIT_DAYS,
                       half_life_days: float = HALF_LIFE_DAYS,
                       min_train_games: int = MIN_TRAIN_GAMES,
                       blocks=DESIGN_BLOCKS, likelihood: str = LIKELIHOOD,
                       own_goalie: bool = OWN_GOALIE,
                       target: str = TARGET) -> pd.DataFrame:
    """As-of rating panel aligned to ``games``' positional order."""
    g = games.reset_index(drop=True)
    gameday = _resolve_gameday(g)
    if gameday.isna().all():
        raise ValueError("team_goal_ratings: frame has no gameday")
    rows = build_side_rows(g)
    teams = sorted(set(rows["team"].dropna()) | set(rows["opp"].dropna()))
    dates = np.array(sorted(pd.Series(gameday).dropna().unique()))
    anchors = dates[::max(1, int(refit_days))]
    panel = pd.DataFrame(np.nan, index=g.index, columns=GOAL_RATING_COLS,
                         dtype=float)
    fits = 0
    for i, anchor in enumerate(anchors):
        anchor = pd.Timestamp(anchor)
        stop = pd.Timestamp(anchors[i + 1]) if i + 1 < len(anchors) else None
        in_window = gameday >= anchor
        if stop is not None:
            in_window &= gameday < stop
        if not in_window.any():
            continue
        train = rows[rows["gameday"] < anchor]
        if train["pos"].nunique() < min_train_games:
            continue
        coef = _fit_once(train, teams, anchor, alpha, half_life_days,
                         blocks=blocks, likelihood=likelihood,
                         own_goalie=own_goalie, target=target)
        off = {t: coef.get(f"toff_{t}", 0.0) for t in teams}
        dfn = {t: -coef.get(f"tdef_{t}", 0.0) for t in teams}
        net = {t: off[t] + dfn[t] for t in teams}
        home = g.loc[in_window, "home_team"]
        away = g.loc[in_window, "away_team"]
        block = panel.loc[in_window]
        block["off_rating_home"] = home.map(off)
        block["off_rating_away"] = away.map(off)
        block["def_rating_home"] = home.map(dfn)
        block["def_rating_away"] = away.map(dfn)
        block["off_rating_diff"] = block["off_rating_home"] - block["off_rating_away"]
        block["def_rating_diff"] = block["def_rating_home"] - block["def_rating_away"]
        # Symmetric match strength: the same value on both sides of the game.
        block["net_rating_sum"] = (block["off_rating_home"]
                                   + block["def_rating_home"]
                                   + block["off_rating_away"]
                                   + block["def_rating_away"])
        block["net_rating_diff"] = home.map(net) - away.map(net)
        panel.loc[in_window] = block
        fits += 1
    panel.attrs["fits"] = fits
    panel.attrs["teams"] = len(teams)
    return panel


def attach_goal_ratings(games: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Attach the eight goal-rating columns to a game frame."""
    panel = build_causal_panel(games, **kwargs)
    out = games.copy()
    for column in GOAL_RATING_COLS:
        out[column] = panel[column].to_numpy()
    coverage = {c: float(out[c].notna().mean()) for c in GOAL_RATING_COLS}
    print(f"NHL goal ratings: {panel.attrs.get('fits')} causal fits, "
          f"{panel.attrs.get('teams')} teams, coverage "
          + ", ".join(f"{c} {v:.1%}" for c, v in coverage.items()),
          flush=True)
    if coverage["net_rating_diff"] <= 0.0:
        raise ValueError("team_goal_ratings: the rating panel is empty")
    return out
