"""Two-sided MLB run model: the served team offence/defence ratings.

ADOPTED 2026-11-21 (MLB scenario C): these seven columns are the moneyline
contract's strength block.  They replaced ``elo_diff``/``win_pct_diff`` and
the ``home_elo``/``away_elo``/``home_win_pct``/``away_win_pct`` levels, none
of which is opponent-adjusted.  The A/B/C trial (kernel
mlb-moneyline-a-b-c-run-model-v3) measured the swap at 0.68309 -> 0.68156
pooled logloss, and in the 45-55% band C beat the incumbent by -0.00157
(98%) with band AUC 0.53531 vs 0.51961.  The daily pipeline attaches the
panel in master_pipeline before the decided snapshot (training + the
re-exported CSV) and again on the history+slate concat, so tonight's board is
priced from the same causal fit as history.

This module fits the substitute.  The unit of observation is a *side row*: two
rows per game, one for each team batting.  The target is runs scored.  The
regressors are a team-offence one-hot and a team-defence one-hot placed on the
same row, so each team's coefficient is estimated net of the opponents it
actually faced, plus a component block (team contact form, the opposing
starter, the opposing bullpen) and context (park, dome, wind, air density,
home).  Sixty plus nuisance parameters and a closed-form ridge: the fit is
milliseconds, the plumbing is the work.

Exports
-------
Seven columns, signed so that **higher is always better**:

``off_rating_home/away/diff``
    Runs per game above league average that the team's offence adds.
``def_rating_home/away/diff``
    Runs per game above league average that the team's pitching/defence
    *prevents*, so the sign is flipped from the raw ``tdef_*`` coefficient.
``net_rating_diff``
    ``(off + def)_home - (off + def)_away``.  This is the matchup number:
    it equals ``(off_H + def_A) - (off_A + def_H)`` by construction, which is
    exactly home offence against away pitching minus the reverse.  Note the
    identity ``net_rating_diff == off_rating_diff + def_rating_diff``; the
    column is kept because tree members consume one summary instead of two.

Causality
---------
``build_causal_panel`` refits on a ``REFIT_DAYS`` grid and, for each anchor,
uses only rows strictly older than the anchor (``_fit_once`` asserts this).
Games newer than the anchor are scored with the anchor's coefficients, so a
rating attached to a game never saw that game or anything after it.  History
enters through an exponential recency weight with a ``HALF_LIFE_DAYS``
half-life, which is what lets April rely on last season and September rely on
current form without a separate cold-start branch.
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
MIN_TRAIN_GAMES = 200

POOLS = ("c", "fb", "sb", "ss", "tb", "rf", "cf", "lf", "dh")

#: Own-offence components (base name; the side suffix is added per row).
OFF_COMPONENTS = (
    "woba_30g",
    "team_iso_30g",
    "team_k_rate_30g",
    "team_bb_rate_30g",
    "team_barrel_15g",
    "team_hardhit_15g",
    "lineup_re24_mean",
    "lineup_ops_vs_starter_hand",
)
#: Opposing-pitcher components (base name; the *opposing* side suffix).
OPP_COMPONENTS = (
    "sp_xfip_5g",
    "sp_k9_5g",
    "sp_xwoba",
    "sp_fbvelo_3g",
    "sp_whiff_3g",
    "bullpen_kbb_10g",
    "bullpen_xfip_10g",
)
CONTEXT = (
    "park_factor_slug",
    "dome_is_neutral",
    "wind_advantage_flyball_factor",
    "air_density_velocity_boost",
)
#: The team's OWN starting pitcher.  Not in the design by default, so the
#: shipped ``def_rating`` is total run prevention (rotation included).  Turned
#: on, the coefficient becomes defence net of the starter on the mound, which
#: is the version that composes with the model's own starter columns instead
#: of double-counting them.
OWN_STARTER_COMPONENTS = (
    "sp_xfip_5g",
    "sp_k9_5g",
    "sp_xwoba",
    "sp_fbvelo_3g",
    "sp_whiff_3g",
)
#: Which nuisance blocks sit beside the team one-hots.  The default is every
#: block, i.e. the configuration that has been through the A/B/C harness.
DESIGN_BLOCKS = ("components", "context", "prior")
#: Residual family.  ``poisson`` is the correctly-specified one for run
#: counts; ``gaussian`` is what shipped.
LIKELIHOOD = "gaussian"
#: Whether the own-starter block is part of the default design (it is not).
OWN_STARTER = False

RUN_RATING_COLS = [
    "off_rating_diff", "off_rating_home", "off_rating_away",
    "def_rating_diff", "def_rating_home", "def_rating_away",
    "net_rating_diff",
]
#: Side levels: routed tree-only by the caller, never into the linear member.
SIDE_RATING_COLS = [
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

    The committed delivery frame and the production game frame name it
    ``game_date``; the A/B/C harness renames it to ``gameday``.  Accept both
    so the same fit serves research and production without a copy step.
    """
    column = "gameday" if "gameday" in games.columns else "game_date"
    if column not in games.columns:
        raise ValueError("team_run_ratings: frame has no gameday/game_date")
    return pd.to_datetime(games[column], errors="coerce")


def _resolve_season(games: pd.DataFrame, gameday: pd.Series) -> pd.Series:
    """Season label, derived from the date when absent or NULL.

    The run-model prior uses season S-1 for season S, so every row needs a
    season even when the source frame does not carry one (the production
    delivery CSV does not).  A slate row appended to history inherits its
    date's year and never enters the fit (its scores are NULL).
    """
    season = _num(games, "season")
    if season.isna().all():
        season = gameday.dt.year.astype(float)
    else:
        season = season.fillna(gameday.dt.year.astype(float))
    return season


def _prior_runs(games: pd.DataFrame) -> pd.DataFrame:
    """Shrunk prior-season runs scored/allowed per (season, team).

    ``p_rs``/``p_ra`` for season S come from S-1 and are shrunk toward that
    season's league mean with a 20-game prior, so a team with a short history
    is pulled to average rather than trusted.  Seasons absent from the frame
    (2024's 2023) stay NaN and are imputed by the fit.
    """
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        parts.append(pd.DataFrame({
            "season": games["season"].to_numpy(),
            "team": games[f"{side}_team"].to_numpy(),
            "rs": pd.to_numeric(games[f"{side}_score"], errors="coerce"),
            "ra": pd.to_numeric(games[f"{other}_score"], errors="coerce"),
        }))
    long = pd.concat(parts, ignore_index=True).dropna(subset=["rs", "ra"])
    if long.empty:
        return pd.DataFrame(columns=["season", "team", "own_p_rs", "own_p_ra"])
    team = (long.groupby(["season", "team"], as_index=False)
            .agg(rs=("rs", "mean"), ra=("ra", "mean"), n=("rs", "size")))
    league = long.groupby("season", as_index=False).agg(
        rs_lg=("rs", "mean"), ra_lg=("ra", "mean"))
    team = team.merge(league, on="season", how="left")
    k = 20.0
    team["own_p_rs"] = (team["n"] * team["rs"] + k * team["rs_lg"]) / (team["n"] + k)
    team["own_p_ra"] = (team["n"] * team["ra"] + k * team["ra_lg"]) / (team["n"] + k)
    team["season"] = team["season"] + 1
    return team[["season", "team", "own_p_rs", "own_p_ra"]]


def build_side_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game: one per team batting, with own/opponent features."""
    g = games.reset_index(drop=True)
    gameday = _resolve_gameday(g)
    g["gameday"] = gameday
    g["season"] = _resolve_season(g, gameday)
    prior = _prior_runs(g)
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        part = pd.DataFrame({
            "pos": g.index.to_numpy(),
            "gameday": gameday.to_numpy(),
            "season": g["season"].to_numpy(),
            "team": g[f"{side}_team"].to_numpy(),
            "opp": g[f"{other}_team"].to_numpy(),
            "runs": pd.to_numeric(g[f"{side}_score"], errors="coerce").to_numpy(),
            "is_home": float(side == "home"),
        })
        for name in OFF_COMPONENTS:
            part[f"off_{name}"] = _num(g, f"{name}_{side}").to_numpy()
        part["off_pool_xwoba"] = pd.concat(
            [_num(g, f"pl_{p}_xwoba_{side}") for p in POOLS], axis=1
        ).mean(axis=1, skipna=True).to_numpy()
        for name in OPP_COMPONENTS:
            part[f"opp_{name}"] = _num(g, f"{name}_{other}").to_numpy()
        for name in OWN_STARTER_COMPONENTS:
            # The base names already carry the ``sp_`` prefix.
            part[f"own_{name}"] = _num(g, f"{name}_{side}").to_numpy()
        for name in CONTEXT:
            part[name] = _num(g, name).to_numpy()
        if not prior.empty:
            part = part.merge(
                prior.rename(columns={"team": "team"}),
                on=["season", "team"], how="left")
            mirror = prior.rename(columns={
                "team": "opp", "own_p_rs": "opp_p_rs", "own_p_ra": "opp_p_ra"})
            part = part.merge(mirror, on=["season", "opp"], how="left")
        else:
            part["own_p_rs"] = part["own_p_ra"] = np.nan
            part["opp_p_rs"] = part["opp_p_ra"] = np.nan
        parts.append(part)
    rows = pd.concat(parts, ignore_index=True)
    rows["own_off"] = rows["off_pool_xwoba"]
    return rows


def _numeric_columns(rows: pd.DataFrame, blocks=DESIGN_BLOCKS,
                     own_starter: bool = OWN_STARTER) -> list[str]:
    blocks = tuple(blocks)
    cols: list[str] = []
    if "components" in blocks:
        for name in OFF_COMPONENTS:
            cols.append(f"off_{name}")
        cols.append("off_pool_xwoba")
        for name in OPP_COMPONENTS:
            cols.append(f"opp_{name}")
    if "context" in blocks:
        cols.extend(CONTEXT)
    if "prior" in blocks:
        cols.extend(["own_p_rs", "own_p_ra", "opp_p_rs", "opp_p_ra"])
    if own_starter:
        cols.extend([f"own_{n}" for n in OWN_STARTER_COMPONENTS])
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
              own_starter: bool = False) -> dict[str, float]:
    """One weighted fit.  Refuses to look at or past ``anchor``."""
    if (train["gameday"] >= anchor).any():
        raise ValueError("team_run_ratings: training row on or after the anchor")
    numeric = _numeric_columns(train, blocks, own_starter)
    train = train.dropna(subset=["runs"])
    if train.empty:
        raise ValueError("team_run_ratings: no decided rows to fit")
    design = _design(train, teams, numeric)
    med = design[numeric].median()
    design[numeric] = design[numeric].fillna(med).fillna(0.0)
    mu = design[numeric].mean()
    sd = design[numeric].std().replace(0.0, 1.0).fillna(1.0)
    design[numeric] = (design[numeric] - mu) / sd
    age = (anchor - train["gameday"]).dt.days.to_numpy(float)
    weight = np.power(0.5, age / float(half_life_days))
    y = train["runs"].to_numpy(float)
    X = design.to_numpy(float)
    if likelihood == "poisson":
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
        raise ValueError(f"team_run_ratings: unknown likelihood {likelihood!r}")
    return dict(zip(design.columns, model.coef_))


def build_causal_panel(games: pd.DataFrame, alpha: float = ALPHA,
                       refit_days: int = REFIT_DAYS,
                       half_life_days: float = HALF_LIFE_DAYS,
                       min_train_games: int = MIN_TRAIN_GAMES,
                       blocks=DESIGN_BLOCKS, likelihood: str = LIKELIHOOD,
                       own_starter: bool = False) -> pd.DataFrame:
    """As-of rating panel aligned to ``games``' positional order."""
    g = games.reset_index(drop=True)
    gameday = _resolve_gameday(g)
    if gameday.isna().all():
        raise ValueError("team_run_ratings: frame has no gameday")
    rows = build_side_rows(g)
    teams = sorted(set(rows["team"].dropna()) | set(rows["opp"].dropna()))
    dates = np.array(sorted(pd.Series(gameday).dropna().unique()))
    anchors = dates[::max(1, int(refit_days))]
    panel = pd.DataFrame(np.nan, index=g.index, columns=RUN_RATING_COLS,
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
                         own_starter=own_starter)
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


def attach_run_ratings(games: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Attach the seven run-model columns to a game frame."""
    panel = build_causal_panel(games, **kwargs)
    out = games.copy()
    for column in RUN_RATING_COLS:
        out[column] = panel[column].to_numpy()
    coverage = {c: float(out[c].notna().mean()) for c in RUN_RATING_COLS}
    print(f"run ratings: {panel.attrs.get('fits')} causal fits, "
          f"{panel.attrs.get('teams')} teams, coverage "
          + ", ".join(f"{c} {v:.1%}" for c, v in coverage.items()),
          flush=True)
    if coverage["net_rating_diff"] <= 0.0:
        raise ValueError("team_run_ratings: the rating panel is empty")
    return out
