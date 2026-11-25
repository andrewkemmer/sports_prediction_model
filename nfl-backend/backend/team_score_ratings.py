"""Two-sided NFL team scoring model: fitted offence/defence ratings.

This is the NFL translation of MLB's ``team_run_ratings`` and NHL's
``team_goal_ratings``.  The unit is a side row (two per game, one per team
scoring); the target is that team's points in the game.  A team-offence
one-hot and an opponent-defence one-hot sit on the same row, so each
coefficient is estimated net of the opponents actually faced.  A weekly
causal ridge refit produces the as-of panel; every fit sees only rows
strictly before its anchor.  The shipped nuisance block is the trailing
own-offence/opponent-defence component panel beside the shrunk prior-season
scoring bridge (adopted 2026-11-25); schedule context stays an opt-in trial
block.

Exports seven columns, signed so higher is always better:

``off_rating_{home,away,diff}``
    Points per game above league average added by the team's offence.
``def_rating_{home,away,diff}``
    Points per game above league average *prevented* by the team's defence;
    the sign is flipped from the raw opponent-defence coefficient.
``net_rating_diff``
    ``net_home - net_away``, the matchup strength index.  ``net`` is a
    variance-reduced blend of three causal channels: the scoring net
    ``off + def``, a fitted win channel and the pre-kickoff Elo prior; see
    `Result channel` below.  The blend adds a direction the off/def
    decomposition does not contain, so the net is deliberately *not* the sum
    of the two served components.

Causality
---------
``build_causal_panel`` refits on a ``REFIT_DAYS`` date grid.  For each anchor
it trains on rows strictly older than the anchor (``_fit_once`` asserts this)
and scores games in ``[anchor, next anchor)`` with those coefficients.  A
rating attached to a game therefore never saw that game or anything later.
History enters through an exponential recency weight with a
``HALF_LIFE_DAYS`` half-life, plus shrunk prior-season scoring for the cold
start.  Team labels are canonicalised through ``config.canonical_team``
before the fit, so a franchise keeps one identity across the OAK->LV and
SD->LAC relocation seams.

Result channel
--------------
The scoring net is a two-parameter-per-team estimate (offence and defence,
summed).  Two further causal channels estimate the same team net:

``win``
    A per-team win-strength ridge on one row per decided game (``+1`` home,
    ``-1`` away), recency-weighted.  Wins come from the same settled scores,
    so this is not a new data source.
``elo``
    The incumbent pre-kickoff Elo rating, used as a *prior mean* on the net
    and never served: the raw ``elo_*`` columns stay retired from the
    feature contract, and only the blended rating is published.

The net is built in two layers.  First the shipped z-score blend of the
scoring net and the win ridge (75/25, on the scoring net's points centre and
spread); its difference from the scoring net is split equally across ``off``
and ``def``, so the two components are exactly the previously adopted ones
and ``off + def`` is that two-channel net.  Then the Elo prior is mixed in at ``ELO_BLEND`` per game (and is used alone
during the warmup, before a club has enough settled history for a fitted
net):
``net = (1 - 0.45) * (off + def) + 0.45 * ELO_POINTS_PER_RATING * elo_diff``,
using the point-in-time ``elo_home``/``elo_away`` entering that game.  This
layer moves the net only, which is why the net carries a strength direction
the off/def decomposition cannot.  Measured on the 2016-2025 walk-forward, the
Elo prior moved C's pooled AUC 0.69943 -> 0.70230 and logloss
0.62866 -> 0.62724 (incumbent 0.70228 / 0.62683), with the 40-60 band at
0.55633 / 0.68880 versus the incumbent's 0.55027 / 0.69027.

Design blocks
-------------
``components``
    The club's own trailing offence (yards per play, pace, projected-lineup
    EPA and yardage) and the opponent's trailing defensive EPA.
``context``
    Rest, home, division, roof/surface and the pre-kickoff weather state.
``prior``
    Shrunk prior-season points for/against for both the rated team and its
    opponent.  This is the September cold-start bridge that a pure recency
    half-life cannot provide.

Missing component columns degrade to the training median, never to a
fabricated value.  A fit needs ``MIN_TRAIN_GAMES`` decided games before its
ratings are published; the production feature-coverage contract is the gate
that turns a wholly-empty panel into a failed rebuild rather than a silent
all-NaN serve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config import canonical_team

#: Ridge penalty on standardised numerics / raw 0-1 dummies.
ALPHA = 50.0
#: Refit cadence.  NFL games are weekly, so every distinct game date is one
#: refit; ``REFIT_DAYS`` is in distinct dates, mirroring MLB/NHL's knob.
REFIT_DAYS = 1
#: Recency half-life for the training weights (the NHL G3S240 default).
HALF_LIFE_DAYS = 240.0
#: A fit needs this many decided games before its ratings are published.
MIN_TRAIN_GAMES = 200
#: Shrink a team's prior-season points toward the league mean with this many
#: games of evidence.  Twenty is MLB/NHL parity; the NFL season is shorter,
#: but the prior is a cold-start bridge, not a headline estimate.
PRIOR_SHRINK_GAMES = 20.0
#: Result channel: causal win-strength ridge on one row per decided game.
#: A shorter half-life than the scoring channel reacts faster to wins, and
#: the 180-day grid point was stable against 120/240/365 in the screen.
WIN_HALF_LIFE_DAYS = 180.0
#: Ridge penalty for the 0/1 win target (raw dummies).
WIN_ALPHA = 10.0
#: Weight on the win channel in the net blend.  The screen tested 0.25/0.50/
#: 0.75; 0.25 was best and is the conservative tilt.
WIN_BLEND = 0.25
#: Weight on the pre-kickoff Elo prior in the net blend.  Elo keeps the
#: margin information that the scoring split and the binary win channel both
#: discard.  The 2016-2025 walk-forward screened 0.25 and 0.45; 0.45 gave the
#: better pooled AUC (0.70267 versus 0.70215, incumbent 0.70228) with pooled
#: logloss flat (+0.0002) and the 40-60 band gain intact.  The remaining
#: weight is split between the scoring net and the win channel in their
#: existing 75/25 ratio.
ELO_BLEND = 0.45
#: Points per Elo rating point, i.e. the scale of the prior.  One team-level
#: Elo standard deviation is ~84 points and one scoring-net standard
#: deviation is ~0.83 points per game, so 0.0098 puts the prior on the same
#: points scale as the fitted net.  A fixed conversion, like ``WIN_BLEND``,
#: keeps the prior's weight stable across folds; a per-anchor re-fit of the
#: slope added noise and lost 0.0008 logloss on the graded population.
ELO_POINTS_PER_RATING = 0.0098
#: Points scored is the target; the unit is points per game because one side
#: row is one game.
TARGET = "points"

#: Own-offence components (base name; the side suffix is added per row).
OFF_COMPONENTS = (
    "ewm_ypp",
    "pace_plays_min",
    "pl_epa_qb",
    "pl_epa_wr",
    "pl_epa_te",
    "pl_epa_rb",
    "pl_exp_pass_yds",
    "pl_exp_rush_yds",
)
#: Opposing-defence components (base name; the *opposing* side suffix is
#: added per row).  ``pbp_def_epa_play_ewm`` is the served defensive EPA
#: series; lower is better, so a strong opponent defence lowers the target.
OPP_COMPONENTS = ("pbp_def_epa_play_ewm",)
#: Per-side context.
SIDE_CONTEXT = ("rest_days", "rest_short")
#: Game-level context (declared once per game, not per side).
GAME_CONTEXT = (
    "div_game",
    "is_dome_home",
    "is_turf_home",
    "temp_f",
    "wind_mph",
    "is_precip",
    "is_snow",
)
CONTEXT = SIDE_CONTEXT + GAME_CONTEXT
#: Which nuisance blocks sit beside the team one-hots.  Adoption 2026-11-25
#: adds the trailing component panel to the prior-season bridge.  An earlier
#: screen of the bare scoring net preferred the lean bridge (univariate AUC
#: 0.6905 vs 0.6776), but on the shipped two-channel net with the Elo prior
#: the panel wins: same 2016-2025 walk-forward, folds and seeds, 0.702999
#: pooled AUC / 0.627083 logloss (prior-only 0.702301 / 0.627241), 0.560449 /
#: 0.688260 at 40-60 (0.556333 / 0.688798) and 0.556241 / 0.694747 at 50-60
#: (0.551363 / 0.696076).  Schedule context stays an opt-in trial block.
DESIGN_BLOCKS = ("components", "prior")
#: Residual family.  Gaussian is what MLB/NHL ship; Poisson is available for
#: a measured count-model trial.
LIKELIHOOD = "gaussian"

SCORE_RATING_COLS = [
    "off_rating_diff", "off_rating_home", "off_rating_away",
    "def_rating_diff", "def_rating_home", "def_rating_away",
    "net_rating_diff",
]
#: Side levels: routed tree-only by the caller, never into the linear member.
SIDE_SCORE_RATING_COLS = [
    "off_rating_home", "off_rating_away",
    "def_rating_home", "def_rating_away",
]


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    """Numeric column or an all-NaN series, never a KeyError."""
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _resolve_gameday(games: pd.DataFrame) -> pd.Series:
    """The frame's date column as datetime.

    Research frames name it ``gameday``; the production delivery frame names
    it ``game_date``.  Accept both, exactly like MLB's and NHL's runners.
    """
    column = "gameday" if "gameday" in games.columns else "game_date"
    if column not in games.columns:
        raise ValueError("team_score_ratings: frame has no gameday/game_date")
    return pd.to_datetime(games[column], errors="coerce")


def _score_column(games: pd.DataFrame, side: str) -> pd.Series:
    for name in (f"{side}_score", f"{side}_pts", f"{side}_points"):
        if name in games.columns:
            return pd.to_numeric(games[name], errors="coerce")
    raise ValueError(f"team_score_ratings: no score column for {side}")


def _resolve_season(games: pd.DataFrame, gameday: pd.Series) -> pd.Series:
    """Season label, taken from the frame or derived from the date.

    The NFL season starts in September and a January/February playoff game
    belongs to the previous autumn's season.  Production frames carry an
    explicit ``season``; the fallback reproduces it for a delivery frame that
    does not.
    """
    season = _num(games, "season")
    derived = (gameday.dt.year
               - (gameday.dt.month <= 2).astype(int)).astype(float)
    if season.isna().all():
        return derived
    return season.fillna(derived)


def _context(games: pd.DataFrame, name: str, side: str) -> pd.Series:
    """Read a context column, honouring the per-side vs per-game split."""
    column = name if name in GAME_CONTEXT else f"{name}_{side}"
    return _num(games, column)


def _prior_points(games: pd.DataFrame) -> pd.DataFrame:
    """Shrunk prior-season points for/against per (season, team).

    ``own_p_pf``/``own_p_pa`` for season S come from S-1 and are shrunk toward
    that season's league mean with ``PRIOR_SHRINK_GAMES`` of evidence, so a
    club with a short history is pulled to average rather than trusted.  Only
    completed earlier seasons enter, so no row can see its own season.
    """
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        parts.append(pd.DataFrame({
            "season": games["season"].to_numpy(),
            "team": games[f"{side}_team"].map(canonical_team).to_numpy(),
            "pf": _score_column(games, side).to_numpy(float),
            "pa": _score_column(games, other).to_numpy(float),
        }))
    long = pd.concat(parts, ignore_index=True).dropna(subset=["pf", "pa"])
    if long.empty:
        return pd.DataFrame(columns=["season", "team", "own_p_pf",
                                     "own_p_pa"])
    team = (long.groupby(["season", "team"], as_index=False)
            .agg(pf=("pf", "mean"), pa=("pa", "mean"),
                 n=("pf", "size")))
    league = long.groupby("season", as_index=False).agg(
        pf_lg=("pf", "mean"), pa_lg=("pa", "mean"))
    team = team.merge(league, on="season", how="left")
    k = PRIOR_SHRINK_GAMES
    team["own_p_pf"] = ((team["n"] * team["pf"] + k * team["pf_lg"])
                        / (team["n"] + k))
    team["own_p_pa"] = ((team["n"] * team["pa"] + k * team["pa_lg"])
                        / (team["n"] + k))
    team["season"] = team["season"] + 1.0
    return team[["season", "team", "own_p_pf", "own_p_pa"]]


def build_side_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game: one per team scoring, with own/opponent features."""
    g = games.reset_index(drop=True)
    g["gameday"] = _resolve_gameday(g)
    g["season"] = _resolve_season(g, g["gameday"])
    for side in ("home", "away"):
        g[f"{side}_team"] = g[f"{side}_team"].map(canonical_team)
    prior = _prior_points(g)
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        part = pd.DataFrame({
            "pos": g.index.to_numpy(),
            "gameday": g["gameday"].to_numpy(),
            "season": g["season"].to_numpy(),
            "team": g[f"{side}_team"].to_numpy(),
            "opp": g[f"{other}_team"].to_numpy(),
            "is_home": float(side == "home"),
            "points": _score_column(g, side).to_numpy(float),
        })
        for name in OFF_COMPONENTS:
            part[f"off_{name}"] = _num(g, f"{name}_{side}").to_numpy()
        for name in OPP_COMPONENTS:
            part[f"opp_{name}"] = _num(g, f"{name}_{other}").to_numpy()
        for name in CONTEXT:
            part[name] = _context(g, name, side).to_numpy()
        if not prior.empty:
            part = part.merge(prior, on=["season", "team"], how="left")
            mirror = prior.rename(columns={
                "team": "opp",
                "own_p_pf": "opp_p_pf",
                "own_p_pa": "opp_p_pa",
            })
            part = part.merge(mirror, on=["season", "opp"], how="left")
        else:
            for column in ("own_p_pf", "own_p_pa", "opp_p_pf",
                           "opp_p_pa"):
                part[column] = np.nan
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _numeric_columns(rows: pd.DataFrame, blocks=DESIGN_BLOCKS) -> list[str]:
    """The nuisance design, in a fixed order.  Team dummies are added later."""
    blocks = tuple(blocks)
    cols: list[str] = []
    if "components" in blocks:
        cols.extend(f"off_{name}" for name in OFF_COMPONENTS)
        cols.extend(f"opp_{name}" for name in OPP_COMPONENTS)
    if "context" in blocks:
        cols.extend(CONTEXT)
    if "prior" in blocks:
        cols.extend(["own_p_pf", "own_p_pa", "opp_p_pf", "opp_p_pa"])
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
              target: str = TARGET) -> dict[str, float]:
    """One weighted fit.  Refuses to look at or past ``anchor``."""
    if target != TARGET:
        raise ValueError(f"team_score_ratings: unknown target {target!r}")
    if (train["gameday"] >= anchor).any():
        raise ValueError(
            "team_score_ratings: training row on or after the anchor")
    numeric = _numeric_columns(train, blocks)
    train = train.dropna(subset=[target])
    train = train[np.isfinite(train[target].to_numpy(float))]
    if train.empty:
        raise ValueError("team_score_ratings: no decided rows to fit")
    design = _design(train, teams, numeric)
    med = design[numeric].median()
    design[numeric] = design[numeric].fillna(med).fillna(0.0)
    mu = design[numeric].mean()
    sd = design[numeric].std().replace(0.0, 1.0).fillna(1.0)
    design[numeric] = (design[numeric] - mu) / sd
    age = (anchor - train["gameday"]).dt.days.to_numpy(float)
    weight = np.power(0.5, age / float(half_life_days))
    y = train[target].to_numpy(float)
    X = design.to_numpy(float)
    if likelihood == "poisson":
        # PoissonRegressor minimises (1/2n) * deviance + alpha * ||w||^2 while
        # Ridge minimises ||y - Xw||^2 + alpha * ||w||^2; rescale alpha so a
        # sweep point means the same thing in both families.
        from sklearn.linear_model import PoissonRegressor
        model = PoissonRegressor(alpha=float(alpha) / (2.0 * len(y)),
                                 fit_intercept=True, max_iter=400)
        model.fit(X, y, sample_weight=weight)
    elif likelihood == "gaussian":
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X, y, sample_weight=weight)
    else:
        raise ValueError(
            f"team_score_ratings: unknown likelihood {likelihood!r}")
    return dict(zip(design.columns, model.coef_))


def _win_fit(games: pd.DataFrame, teams: list[str], anchor: pd.Timestamp,
             alpha: float = WIN_ALPHA,
             half_life_days: float = WIN_HALF_LIFE_DAYS,
             min_train_games: int = MIN_TRAIN_GAMES) -> dict[str, float] | None:
    """One causal win-strength ridge.  Returns team coefficients, or None.

    The row is a decided game, ``+1`` for the home team and ``-1`` for the
    away team; the intercept is home advantage.  Training rows are strictly
    before ``anchor`` (the caller has already enforced that for the scoring
    fit, and the same table is used here).
    """
    train = games[games["gameday"] < anchor].dropna(subset=["home_win"])
    if len(train) < min_train_games:
        return None
    y = train["home_win"].to_numpy(float)
    if len(np.unique(y)) < 2:
        return None
    index = {team: i for i, team in enumerate(teams)}
    home = train["home_team"].map(index)
    away = train["away_team"].map(index)
    if home.isna().any() or away.isna().any():
        return None
    X = np.zeros((len(train), len(teams)), dtype=float)
    rows = np.arange(len(train))
    X[rows, home.to_numpy(int)] = 1.0
    X[rows, away.to_numpy(int)] = -1.0
    age = (anchor - train["gameday"]).dt.days.to_numpy(float)
    weight = np.power(0.5, age / float(half_life_days))
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y, sample_weight=weight)
    return dict(zip(teams, model.coef_))


def _blend_net(net_values: np.ndarray,
               win_values: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """The two-channel team net plus the component adjustment that defines it.

    The shipped z-score blend of the scoring net and the win ridge, on the
    scoring net's points centre/spread.  Its difference from the scoring net
    is returned as ``adjustment`` so the caller can split it equally across
    ``off_rating`` and ``def_rating``; that keeps the served components
    exactly as previously adopted and ``off + def`` equal to this net.

    The Elo prior is applied per game afterwards (``build_causal_panel``),
    because the frame's ``elo_home``/``elo_away`` are the point-in-time
    ratings entering that game.  It moves the net only, which is the extra
    strength direction the off/def decomposition cannot carry.
    """
    net_values = np.asarray(net_values, float)
    center = float(np.mean(net_values))
    sd_net = float(np.std(net_values, ddof=1))
    if not sd_net > 0.0:
        return net_values.copy(), np.zeros_like(net_values)
    blended = center + sd_net * ((1.0 - WIN_BLEND)
                                 * ((net_values - center) / sd_net))
    if win_values is not None:
        win = np.asarray(win_values, float)
        sd_win = float(np.std(win, ddof=1))
        if sd_win > 0.0:
            blended = blended + sd_net * WIN_BLEND * (
                (win - float(np.mean(win))) / sd_win)
    return blended, blended - net_values


def build_causal_panel(games: pd.DataFrame, alpha: float = ALPHA,
                       refit_days: int = REFIT_DAYS,
                       half_life_days: float = HALF_LIFE_DAYS,
                       min_train_games: int = MIN_TRAIN_GAMES,
                       blocks=DESIGN_BLOCKS, likelihood: str = LIKELIHOOD,
                       target: str = TARGET) -> pd.DataFrame:
    """As-of rating panel aligned to ``games``' positional order."""
    g = games.reset_index(drop=True)
    if not len(g):
        return pd.DataFrame(columns=SCORE_RATING_COLS)
    gameday = _resolve_gameday(g)
    if gameday.isna().all():
        raise ValueError("team_score_ratings: frame has no gameday")
    if not {"home_team", "away_team"} <= set(g.columns):
        raise ValueError("team_score_ratings: frame has no team columns")
    rows = build_side_rows(g)
    elo_home = _num(g, "elo_home")
    elo_away = _num(g, "elo_away")
    has_elo = bool(elo_home.notna().any() and elo_away.notna().any())
    teams = sorted(set(rows["team"].dropna()) | set(rows["opp"].dropna()))
    home_score = _score_column(g, "home")
    away_score = _score_column(g, "away")
    decided = home_score.notna() & away_score.notna()
    game_rows = pd.DataFrame({
        "gameday": gameday.to_numpy(),
        "home_team": g["home_team"].map(canonical_team).to_numpy(),
        "away_team": g["away_team"].map(canonical_team).to_numpy(),
        "home_win": np.select(
            [home_score > away_score, home_score < away_score],
            [1.0, 0.0], default=0.5),
    })
    game_rows = game_rows[decided.to_numpy()].reset_index(drop=True)
    dates = np.array(sorted(pd.Series(gameday).dropna().unique()))
    anchors = dates[::max(1, int(refit_days))]
    panel = pd.DataFrame(np.nan, index=g.index, columns=SCORE_RATING_COLS,
                         dtype=float)
    team_index = {t: j for j, t in enumerate(teams)}
    fits = 0
    blended_fits = 0
    elo_fits = 0
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
            # No fitted net yet: publish ELO_BLEND * prior, the same blend
            # with a zero fitted net, rather than a NaN the model would have
            # to impute.
            if has_elo and ELO_BLEND > 0.0:
                prior = (elo_home.loc[in_window].to_numpy(float)
                         - elo_away.loc[in_window].to_numpy(float))
                warmup = np.where(np.isfinite(prior),
                                  ELO_BLEND * ELO_POINTS_PER_RATING * prior,
                                  np.nan)
                if np.isfinite(warmup).any():
                    block = panel.loc[in_window]
                    block["net_rating_diff"] = warmup
                    panel.loc[in_window] = block
                    elo_fits += 1
            continue
        coef = _fit_once(train, teams, anchor, alpha, half_life_days,
                         blocks=blocks, likelihood=likelihood, target=target)
        off = {t: coef.get(f"toff_{t}", 0.0) for t in teams}
        dfn = {t: -coef.get(f"tdef_{t}", 0.0) for t in teams}
        # The components stay pure scoring estimates; the net is the blended
        # strength index and is not their sum (the Elo prior adds a direction
        # the decomposition does not carry).
        net_values = np.array([off[t] + dfn[t] for t in teams], dtype=float)
        win_coef = _win_fit(game_rows, teams, anchor)
        win_values = (np.array([win_coef[t] for t in teams], dtype=float)
                      if win_coef is not None else None)
        if win_coef is not None:
            blended_fits += 1
        net_team, adjustment = _blend_net(net_values, win_values)
        for j, team in enumerate(teams):
            off[team] += 0.5 * float(adjustment[j])
            dfn[team] += 0.5 * float(adjustment[j])
        net = {t: float(net_team[j]) for j, t in enumerate(teams)}
        home = g.loc[in_window, "home_team"].map(canonical_team)
        away = g.loc[in_window, "away_team"].map(canonical_team)
        block = panel.loc[in_window]
        block["off_rating_home"] = home.map(off)
        block["off_rating_away"] = away.map(off)
        block["def_rating_home"] = home.map(dfn)
        block["def_rating_away"] = away.map(dfn)
        block["off_rating_diff"] = (block["off_rating_home"]
                                    - block["off_rating_away"])
        block["def_rating_diff"] = (block["def_rating_home"]
                                    - block["def_rating_away"])
        net_diff = (home.map(net) - away.map(net)).to_numpy(float)
        if has_elo and ELO_BLEND > 0.0:
            prior = (elo_home.loc[in_window].to_numpy(float)
                     - elo_away.loc[in_window].to_numpy(float))
            prior_pts = ELO_POINTS_PER_RATING * prior
            known = np.isfinite(net_diff)
            # A club without enough settled history has no fitted net yet.
            # The Elo prior is exactly the warmup value, so publish it alone
            # rather than a NaN that the model would have to impute.
            tilted = np.where(known,
                              (1.0 - ELO_BLEND) * net_diff
                              + ELO_BLEND * prior_pts,
                              prior_pts)
            net_diff = np.where(np.isfinite(prior), tilted, net_diff)
            elo_fits += int(np.isfinite(prior).sum() > 0)
        block["net_rating_diff"] = net_diff
        panel.loc[in_window] = block
        fits += 1
    panel.attrs["fits"] = fits
    panel.attrs["blended_fits"] = blended_fits
    panel.attrs["elo_fits"] = elo_fits
    panel.attrs["teams"] = len(teams)
    return panel


def attach_score_ratings(games: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Attach the seven scoring-rating columns to a game frame.

    A frame with fewer than ``MIN_TRAIN_GAMES`` decided games leaves the
    columns all-NaN rather than raising: the production feature-coverage
    contract is the authoritative fail-closed gate, and research/test frames
    with a handful of games must still be buildable.
    """
    panel = build_causal_panel(games, **kwargs)
    out = games.copy()
    for column in SCORE_RATING_COLS:
        if len(panel):
            out[column] = panel[column].to_numpy()
        else:
            out[column] = np.nan
    coverage = {c: float(out[c].notna().mean()) for c in SCORE_RATING_COLS}
    print(f"NFL score ratings: {panel.attrs.get('fits', 0)} causal fits "
          f"({panel.attrs.get('blended_fits', 0)} with the win channel, "
          f"{panel.attrs.get('elo_fits', 0)} with the Elo prior), "
          f"{panel.attrs.get('teams', 0)} teams, coverage "
          + ", ".join(f"{c} {v:.1%}" for c, v in coverage.items()),
          flush=True)
    return out
