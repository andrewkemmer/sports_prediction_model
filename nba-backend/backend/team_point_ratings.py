"""Two-sided NBA team rating: offence and defence in points per 100.

The contract already carries a strength block — ``pl_o_dpm_*``/``pl_d_dpm_*``
(pace-scaled DARKO team blends), ``pl_o_pts_*`` (expected points) and
``pl_net_o_*`` (home offence against away defence, pre-computed).  Those are a
*roster sum*: they never look at who the team has played, so two teams with
the same player ratings get the same number regardless of schedule, and the
offence/defence split is a sum of player splits rather than a team effect.

This module fits the substitute the same way the MLB run model does.  The unit
is a side row — two per game, one per team scoring — and the target is that
team's points per 100 possessions.  A team-offence one-hot and a team-defence
one-hot sit on the *same row*, so each coefficient is estimated net of the
opponents it actually faced.

Exports
-------
``off_rating_{home,away,diff}``, ``def_rating_{home,away,diff}`` and
``net_rating_diff``, in points per 100 possessions, signed higher-is-better.
``net_rating_diff == off_rating_diff + def_rating_diff`` and is the matchup
number: home offence against away defence minus the reverse.

Availability is deliberately *not* in the rating.  ``star_out_*`` stays in
both arms so the comparison isolates strength; the team one-hot then measures
the roster as actually fielded, which is the availability-weighted quantity.
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

#: Possession denominators, in preference order.  Both are trailing
#: pre-game estimates, so the target stays causal.
POSSESSION_COLS = ("event_possessions", "ewm_pace")
#: League possessions per game, used only when neither column resolves.
FALLBACK_POSSESSIONS = 100.0

POINT_RATING_COLS = [
    "off_rating_diff", "off_rating_home", "off_rating_away",
    "def_rating_diff", "def_rating_home", "def_rating_away",
    "net_rating_diff",
]
#: Side levels: routed tree-only by the caller, never into the linear member.
SIDE_POINT_RATING_COLS = [
    "off_rating_home", "off_rating_away",
    "def_rating_home", "def_rating_away",
]

CONTEXT = ("rest_days", "back_to_back", "is_playoffs")
AVAILABILITY = ("star_out_impact", "star_out_weight_share")


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    """Numeric column or an all-NaN series, never a KeyError."""
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _score_column(games: pd.DataFrame, side: str) -> pd.Series:
    for name in (f"{side}_score", f"{side}_pts", f"{side}_points"):
        if name in games.columns:
            return pd.to_numeric(games[name], errors="coerce")
    raise ValueError(f"team_point_ratings: no score column for {side}")


def _possessions(games: pd.DataFrame, side: str) -> pd.Series:
    """The side's trailing possessions per game, clipped to a plausible band."""
    for name in POSSESSION_COLS:
        column = f"{name}_{side}"
        if column not in games.columns:
            continue
        values = pd.to_numeric(games[column], errors="coerce")
        values = values.where((values >= 80.0) & (values <= 130.0))
        # A column is usable if it resolves for at least a quarter of the
        # frame (and never zero rows, which is what an all-clipped column
        # looks like).
        need = max(1, int(0.25 * len(values)))
        if values.notna().sum() >= need:
            fill = float(values.median())
            return values.fillna(fill)
    return pd.Series(FALLBACK_POSSESSIONS, index=games.index, dtype=float)


def build_side_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game: one per team scoring."""
    g = games.reset_index(drop=True)
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        points = _score_column(g, side)
        possessions = _possessions(g, side)
        part = pd.DataFrame({
            "pos": g.index.to_numpy(),
            "gameday": pd.to_datetime(g["gameday"], errors="coerce").to_numpy(),
            "team": g[f"{side}_team"].to_numpy(),
            "opp": g[f"{other}_team"].to_numpy(),
            "is_home": float(side == "home"),
            # Points per 100 possessions, the unit the market prices.
            "rating": 100.0 * points.to_numpy(float) / possessions.to_numpy(float),
        })
        for name in CONTEXT:
            part[name] = _num(g, f"{name}_{side}").to_numpy()
        for name in AVAILABILITY:
            part[name] = _num(g, f"{name}_{side}").to_numpy()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _numeric_columns(rows: pd.DataFrame) -> list[str]:
    cols = ["is_home"] + list(CONTEXT) + list(AVAILABILITY)
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
        raise ValueError("team_point_ratings: training row on or after the anchor")
    numeric = _numeric_columns(train)
    train = train.dropna(subset=["rating"])
    train = train[np.isfinite(train["rating"].to_numpy(float))]
    if train.empty:
        raise ValueError("team_point_ratings: no decided rows to fit")
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
        raise ValueError("team_point_ratings: frame has no gameday")
    rows = build_side_rows(g)
    teams = sorted(set(rows["team"].dropna()) | set(rows["opp"].dropna()))
    dates = np.array(sorted(pd.Series(gameday).dropna().unique()))
    anchors = dates[::max(1, int(refit_days))]
    panel = pd.DataFrame(np.nan, index=g.index, columns=POINT_RATING_COLS,
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


def attach_point_ratings(games: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Attach the seven point-rating columns to a game frame."""
    panel = build_causal_panel(games, **kwargs)
    out = games.copy()
    for column in POINT_RATING_COLS:
        out[column] = panel[column].to_numpy()
    coverage = {c: float(out[c].notna().mean()) for c in POINT_RATING_COLS}
    print(f"NBA point ratings: {panel.attrs.get('fits')} causal fits, "
          f"{panel.attrs.get('teams')} teams, coverage "
          + ", ".join(f"{c} {v:.1%}" for c, v in coverage.items()),
          flush=True)
    if coverage["net_rating_diff"] <= 0.0:
        raise ValueError("team_point_ratings: the rating panel is empty")
    return out
