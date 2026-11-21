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

Deliberately excluded from the design: every served skater and goalie column.
Those are exactly the columns arm B already sees; including them would make
the fitted effect a residual of a proxy the model can already read.  The team
dummies therefore carry the whole team — skaters, system and goaltending —
which is why ``def_rating`` here is a team defence number and not a goalie
number.

Exports ``off_rating_{home,away,diff}``, ``def_rating_{home,away,diff}`` and
``net_rating_diff`` in goals per 60, signed higher-is-better, with
``net_rating_diff == off_rating_diff + def_rating_diff``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

#: Ridge penalty on standardised numerics / raw 0-1 dummies.
ALPHA = 50.0
#: Refit cadence for the as-of panel, in distinct game dates.
REFIT_DAYS = 7
#: Recency half-life for the training weights.
HALF_LIFE_DAYS = 120.0
#: A fit needs this many decided games before its ratings are published.
MIN_TRAIN_GAMES = 150
#: Regulation ice time.  Overridden by a served TOI column when one exists.
REGULATION_MINUTES = 60.0
TOI_COLS = ("toi_minutes", "time_on_ice", "ice_time")

GOAL_RATING_COLS = [
    "off_rating_diff", "off_rating_home", "off_rating_away",
    "def_rating_diff", "def_rating_home", "def_rating_away",
    "net_rating_diff",
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
            return values.fillna(float(values.median()))
    return pd.Series(REGULATION_MINUTES, index=games.index, dtype=float)


def build_side_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game: one per team scoring."""
    g = games.reset_index(drop=True)
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        goals = _score_column(g, side)
        minutes = _team_minutes(g, side)
        part = pd.DataFrame({
            "pos": g.index.to_numpy(),
            "gameday": pd.to_datetime(g["gameday"], errors="coerce").to_numpy(),
            "team": g[f"{side}_team"].to_numpy(),
            "opp": g[f"{other}_team"].to_numpy(),
            "is_home": float(side == "home"),
            # Goals per 60 minutes of team ice time.
            "rating": 60.0 * goals.to_numpy(float) / minutes.to_numpy(float),
        })
        for name in CONTEXT + GAME_LEVEL:
            part[name] = _context(g, name, side).to_numpy()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _numeric_columns(rows: pd.DataFrame) -> list[str]:
    cols = ["is_home"] + list(CONTEXT) + list(GAME_LEVEL)
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
              alpha: float, half_life_days: float) -> dict[str, float]:
    """One weighted fit.  Refuses to look at or past ``anchor``."""
    if (train["gameday"] >= anchor).any():
        raise ValueError("team_goal_ratings: training row on or after the anchor")
    numeric = _numeric_columns(train)
    train = train.dropna(subset=["rating"])
    train = train[np.isfinite(train["rating"].to_numpy(float))]
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
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(design.to_numpy(float), train["rating"].to_numpy(float),
              sample_weight=weight)
    return dict(zip(design.columns, model.coef_))


def build_causal_panel(games: pd.DataFrame, alpha: float = ALPHA,
                       refit_days: int = REFIT_DAYS,
                       half_life_days: float = HALF_LIFE_DAYS,
                       min_train_games: int = MIN_TRAIN_GAMES) -> pd.DataFrame:
    """As-of rating panel aligned to ``games``' positional order."""
    g = games.reset_index(drop=True)
    gameday = pd.to_datetime(g["gameday"], errors="coerce")
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
        coef = _fit_once(train, teams, anchor, alpha, half_life_days)
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
        block["net_rating_diff"] = home.map(net) - away.map(net)
        panel.loc[in_window] = block
        fits += 1
    panel.attrs["fits"] = fits
    panel.attrs["teams"] = len(teams)
    return panel


def attach_goal_ratings(games: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Attach the seven goal-rating columns to a game frame."""
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
