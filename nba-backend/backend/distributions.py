"""NBA totals and point-spread/run-line distribution engine.

The engine fits two score regressors, estimates negative-binomial dispersion
from leakage-free OOF residuals, and prices a complete NBA line grid from
paired score draws.  It never reads a sportsbook or silently changes source.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    from backend import config
    from backend import features as feat_mod
    from backend import folds as folds_mod
except ImportError:
    import config
    import features as feat_mod
    import folds as folds_mod

logger = logging.getLogger(__name__)
MC_DRAWS = 4000
MC_SEED = 42
ALPHA_FLOOR = 1e-8
ALPHA_CAP = 2.0


def _grid_key(prefix: str, value: float | int) -> str:
    text = str(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    return f"{prefix}_{text.replace('-', 'm').replace('.', '_')}"


def _make_reg():
    from lightgbm import LGBMRegressor
    params = dict(config.LIGHTGBM_REG_PARAMS)
    params["objective"] = "poisson"
    return LGBMRegressor(**params)


class ScoreRegressor:
    """Two score regressors sharing the active moneyline feature view."""

    def __init__(self) -> None:
        self.home_model: Any | None = None
        self.away_model: Any | None = None
        self.feature_columns: list[str] = []
        self.fallback: tuple[Any, Any] | None = None

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        X = feat_mod.tree_view(df)
        if not self.feature_columns:
            self.feature_columns = list(X.columns)
        return X.reindex(columns=self.feature_columns)

    @staticmethod
    def _target(df: pd.DataFrame, col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce")

    def fit(self, df: pd.DataFrame) -> "ScoreRegressor":
        X = self._matrix(df)
        h = self._target(df, "home_score")
        a = self._target(df, "away_score")
        valid = h.notna() & a.notna()
        if valid.sum() < 2:
            raise ValueError("score regressor requires at least two settled games")
        X, h, a = X.loc[valid], h.loc[valid], a.loc[valid]
        try:
            self.home_model = _make_reg()
            self.away_model = _make_reg()
            self.home_model.fit(X, h.clip(lower=0))
            self.away_model.fit(X, a.clip(lower=0))
            self.fallback = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("LightGBM score regressor unavailable, using ridge: %s", exc)
            from sklearn.linear_model import Ridge
            numeric = X.copy()
            for col in config.TREE_CATEGORICAL_COLS:
                if col in numeric:
                    numeric[col] = numeric[col].astype(float)
            self.home_model = self.away_model = None
            self.fallback = (Ridge(alpha=1.0), Ridge(alpha=1.0))
            self.fallback[0].fit(numeric, h)
            self.fallback[1].fit(numeric, a)
            self._matrix = lambda frame, _numeric=numeric.columns: self._matrix_numeric(frame, _numeric)  # type: ignore[method-assign]
        return self

    def _matrix_numeric(self, df: pd.DataFrame, columns) -> pd.DataFrame:
        X = feat_mod.tree_view(df).reindex(columns=list(columns))
        for col in config.TREE_CATEGORICAL_COLS:
            if col in X:
                X[col] = X[col].astype(float)
        return X

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.fallback is not None:
            X = self._matrix(df).astype(float)
            return (np.clip(self.fallback[0].predict(X), 1e-6, None),
                    np.clip(self.fallback[1].predict(X), 1e-6, None))
        if self.home_model is None or self.away_model is None:
            raise RuntimeError("score regressor is not fitted")
        X = self._matrix(df)
        return (np.clip(self.home_model.predict(X), 1e-6, None),
                np.clip(self.away_model.predict(X), 1e-6, None))


def estimate_alpha(y, mu) -> float:
    y, mu = np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
    ok = np.isfinite(y) & np.isfinite(mu) & (mu > 0)
    if ok.sum() < 2:
        return 0.0
    excess = np.sum((y[ok] - mu[ok]) ** 2 - y[ok])
    denom = np.sum(mu[ok] ** 2)
    return float(np.clip(max(excess / max(denom, 1e-12), 0.0), 0.0, ALPHA_CAP))


def calibrate_dispersion(oof: pd.DataFrame | None) -> dict[str, Any]:
    if oof is None or not len(oof):
        return {"alpha_home": 0.0, "alpha_away": 0.0,
                "distribution": "negative_binomial", "mc_draws": MC_DRAWS,
                "poisson_limit": True}
    ah = estimate_alpha(oof.home_score, oof.mu_h)
    aa = estimate_alpha(oof.away_score, oof.mu_a)
    return {"alpha_home": ah, "alpha_away": aa,
            "distribution": "negative_binomial", "mc_draws": MC_DRAWS,
            "poisson_limit": bool(max(ah, aa) <= ALPHA_FLOOR)}


def _nb_draws(mu: np.ndarray, alpha: float, rng: np.random.Generator,
              n_draws: int) -> np.ndarray:
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-6)
    if alpha <= ALPHA_FLOOR:
        return rng.poisson(mu[:, None], size=(len(mu), n_draws))
    size = np.full(len(mu), 1.0 / max(alpha, ALPHA_FLOOR))
    prob = size / (size + mu)
    return rng.negative_binomial(size[:, None], prob[:, None],
                                 size=(len(mu), n_draws))


def simulate_distributions(mu_h, mu_a, alpha_home=0.0, alpha_away=0.0,
                           n_draws=MC_DRAWS, seed=MC_SEED) -> pd.DataFrame:
    """Price all configured lines from paired NBA score draws."""
    mu_h = np.asarray(mu_h, dtype=float)
    mu_a = np.asarray(mu_a, dtype=float)
    if len(mu_h) != len(mu_a):
        raise ValueError("home/away expected-score arrays must have equal length")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    # Keep memory bounded for a full slate while preserving deterministic order.
    chunk = max(1, min(len(mu_h), 2_000_000 // max(int(n_draws), 1)))
    for start in range(0, len(mu_h), chunk):
        end = min(start + chunk, len(mu_h))
        hdraw = _nb_draws(mu_h[start:end], alpha_home, rng, int(n_draws))
        adraw = _nb_draws(mu_a[start:end], alpha_away, rng, int(n_draws))
        for i in range(end - start):
            h, a = hdraw[i], adraw[i]
            total, margin = h + a, h - a
            row: dict[str, Any] = {
                "mu_h": float(mu_h[start + i]), "mu_a": float(mu_a[start + i]),
                "mu_margin": float(mu_h[start + i] - mu_a[start + i]),
                "mu_total": float(mu_h[start + i] + mu_a[start + i]),
                "p_tie": float(np.mean(margin == 0)),
            }
            # Keep the three-way distribution coherent.  The old conditional
            # normalization removed tie mass from home/away and then emitted
            # that mass separately, so the three probabilities summed > 1.
            row["p_home_win_derived"] = float(np.mean(margin > 0))
            row["p_away_win_derived"] = float(np.mean(margin < 0))
            for line in config.SPREAD_GRID:
                key = _grid_key("p_home_cover", line)
                row[key] = float(np.mean(margin > line))
                row[_grid_key("p_push", line)] = float(np.mean(margin == line))
                row[_grid_key("p_away_cover", line)] = float(np.mean(margin < line))
            for line in config.HALF_STOP_LINES:
                # Half stops have no push mass.
                row[_grid_key("p_home_cover", line)] = float(np.mean(margin > line))
                row[_grid_key("p_away_cover", line)] = float(np.mean(margin < line))
                row[_grid_key("p_push", line)] = 0.0
            for line in config.TOTAL_GRID:
                row[_grid_key("p_over", line)] = float(np.mean(total > line))
                row[_grid_key("p_push_total", line)] = float(np.mean(total == line))
                row[_grid_key("p_under", line)] = float(np.mean(total < line))
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["fair_spread"] = [_fair(row, "p_home_cover", config.SPREAD_GRID)
                              for _, row in out.iterrows()]
        out["fair_total"] = [_fair(row, "p_over", config.TOTAL_GRID)
                             for _, row in out.iterrows()]
        out["p_cover_fair"] = [float(row[_grid_key("p_home_cover", row.fair_spread)])
                               for _, row in out.iterrows()]
        out["p_over_fair"] = [float(row[_grid_key("p_over", row.fair_total)])
                              for _, row in out.iterrows()]
    return out


def _fair(row: dict[str, Any], prefix: str, lines: list[int]) -> float:
    values = np.asarray([float(row.get(_grid_key(prefix, line), np.nan))
                         for line in lines], dtype=float)
    if not np.isfinite(values).any():
        return float(lines[0])
    return float(lines[int(np.nanargmin(np.abs(values - 0.5)))])


def apply_distribution(df: pd.DataFrame, params: dict | None = None, **kwargs) -> pd.DataFrame:
    params = params or {}
    dist = simulate_distributions(
        df.mu_h.to_numpy(float), df.mu_a.to_numpy(float),
        float(params.get("alpha_home", 0)), float(params.get("alpha_away", 0)),
        n_draws=int(params.get("mc_draws", MC_DRAWS)), seed=int(params.get("seed", MC_SEED)))
    base = df.drop(columns=[c for c in dist.columns if c in df.columns], errors="ignore")
    return pd.concat([base.reset_index(drop=True), dist.reset_index(drop=True)], axis=1)


def walk_forward_oof(game_df: pd.DataFrame, date_col: str = "gameday",
                     fold_list=None) -> dict:
    df = folds_mod.canonical_sort(game_df, date_col)
    folds = fold_list if fold_list is not None else folds_mod.make_folds(df, date_col)
    parts: list[pd.DataFrame] = []
    rows: list[dict] = []
    for fold in folds:
        train, val = df.loc[fold.train_idx], df.loc[fold.val_idx]
        try:
            mu_h, mu_a = ScoreRegressor().fit(train).predict(val)
        except Exception as exc:  # noqa: BLE001
            logger.warning("distribution fold %s failed: %s", fold.fold_id, exc)
            mu_h = np.full(len(val), np.nan)
            mu_a = np.full(len(val), np.nan)
        parts.append(pd.DataFrame({
            "game_id": val.game_id.to_numpy(),
            "gameday": pd.to_datetime(val[date_col]).to_numpy(),
            "season": val.season.to_numpy() if "season" in val else np.nan,
            "fold_id": fold.fold_id, "mu_h": mu_h, "mu_a": mu_a,
            "home_score": val.home_score.to_numpy(float),
            "away_score": val.away_score.to_numpy(float),
        }))
        rows.append({"fold_id": fold.fold_id, "val_start": str(fold.val_start.date()),
                     "val_end": str(fold.val_end.date()), "n_train": len(train),
                     "n_val": len(val)})
    oof = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(oof):
        oof["margin"] = oof.home_score - oof.away_score
        oof["total"] = oof.home_score + oof.away_score
    return {"oof": oof, "fold_table": pd.DataFrame(rows)}


def fit_final(game_df: pd.DataFrame) -> ScoreRegressor:
    return ScoreRegressor().fit(game_df)


def _fit_platt(p, y):
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) < 30 or len(np.unique(y)) < 2:
        return None
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z.reshape(-1, 1), y.astype(int))
    a, b = float(model.coef_[0, 0]), float(model.intercept_[0])
    return {"a": a, "b": b} if np.isfinite(a) and np.isfinite(b) else None


def _sigmoid(z):
    """Numerically stable logistic transform for calibration logits."""
    values = np.asarray(z, dtype=float)
    out = np.empty_like(values, dtype=float)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_z = np.exp(values[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def _apply_platt(p, cal):
    values = np.asarray(p, dtype=float)
    if not cal:
        return values
    clipped = np.clip(values, 1e-7, 1 - 1e-7)
    z = np.log(clipped / (1 - clipped))
    return np.clip(_sigmoid(cal["a"] * z + cal["b"]), 1e-7, 1 - 1e-7)


def _apply_derived_calibration(p_home, p_tie, cal):
    """Calibrate home/conditional win probability without losing tie mass."""
    p_home = np.asarray(p_home, dtype=float)
    tie = np.clip(np.asarray(p_tie, dtype=float), 0.0, 1.0)
    if not cal:
        return p_home, tie
    non_tie = np.maximum(1.0 - tie, 1e-12)
    conditional = np.clip(p_home / non_tie, 1e-7, 1 - 1e-7)
    favorite_home = conditional >= 0.5
    favored = np.where(favorite_home, conditional, 1.0 - conditional)
    calibrated_favorite = np.maximum(0.5, _apply_platt(favored, cal))
    calibrated_home = np.where(
        favorite_home,
        tie + non_tie * calibrated_favorite,
        non_tie * (1.0 - calibrated_favorite),
    )
    calibrated_home = np.clip(calibrated_home, 0.0, 1.0 - tie)
    return calibrated_home, tie


def _fit_derived_calibration(p_home, p_tie, margin):
    """Fit a favored-space map on non-ties and return the three-way map."""
    p_home = np.asarray(p_home, dtype=float)
    tie = np.asarray(p_tie, dtype=float)
    margin = np.asarray(margin, dtype=float)
    non_tie_mass = np.maximum(1.0 - tie, 1e-12)
    conditional = np.clip(p_home / non_tie_mass, 1e-7, 1 - 1e-7)
    favorite_home = conditional >= 0.5
    y_favorite = np.where(favorite_home, margin > 0, margin < 0).astype(float)
    # A tie is a real third outcome, not a negative example for either team.
    # Exclude the OBSERVED zero margin as well as rows whose model tie mass
    # is degenerate; fitting a binary favored-space map on ``margin == 0``
    # would silently turn every actual tie into an away-team loss.
    valid = (np.isfinite(p_home) & np.isfinite(tie) & np.isfinite(margin)
             & (margin != 0) & (tie < 1.0 - 1e-12))
    y_favorite = np.where(valid, y_favorite, np.nan)
    p_favorite = np.where(favorite_home, conditional, 1.0 - conditional)
    return _fit_platt(p_favorite, y_favorite)


def _prequential(p, y, folds):
    """Fit each fold's map on prior folds only, returning full OOF values."""
    values = np.asarray(p, dtype=float)
    out = values.copy()
    prior_p: list[np.ndarray] = []
    prior_y: list[np.ndarray] = []
    for fold in pd.unique(folds):
        mask = np.asarray(folds) == fold
        cal = _fit_platt(np.concatenate(prior_p) if prior_p else [],
                         np.concatenate(prior_y) if prior_y else [])
        if cal:
            out[mask] = _apply_platt(values[mask], cal)
        prior_p.append(values[mask])
        prior_y.append(np.asarray(y)[mask])
    return out, _fit_platt(values, y)


def calibrate_market_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply causal line-specific calibration to a priced OOF market frame."""
    out = df.copy()
    bundle: dict[str, Any] = {"method": "prequential_platt", "scope": "line_specific",
                              "totals": {}, "run_lines": {}}
    if not len(out) or "fold_id" not in out:
        return out, bundle
    folds = out.fold_id.to_numpy()
    margin = out.get("margin", pd.Series(np.nan, index=out.index)).to_numpy(float)
    total = out.get("total", pd.Series(np.nan, index=out.index)).to_numpy(float)
    for line in config.TOTAL_GRID:
        key, push_key, under_key = (_grid_key("p_over", line),
                                    _grid_key("p_push_total", line),
                                    _grid_key("p_under", line))
        if key not in out:
            continue
        over = out[key].to_numpy(float)
        co, mo = _prequential(over, total > line, folds)
        if line == int(line):
            push = out[push_key].to_numpy(float)
            cp, mp = _prequential(push, total == line, folds)
            under = out[under_key].to_numpy(float)
            cu, mu = _prequential(under, total < line, folds)
            values = np.maximum(np.column_stack([co, cp, cu]), 1e-9)
            values /= values.sum(axis=1, keepdims=True)
            out[key], out[push_key], out[under_key] = values.T
            bundle["totals"][str(line)] = {"over": mo, "push": mp, "under": mu}
        else:
            out[key] = co
            bundle["totals"][str(line)] = {"over": mo, "push": None, "under": None}
    for line in [*config.SPREAD_GRID, *config.HALF_STOP_LINES]:
        key, push_key, away_key = (_grid_key("p_home_cover", line),
                                   _grid_key("p_push", line),
                                   _grid_key("p_away_cover", line))
        if key not in out:
            continue
        home = out[key].to_numpy(float)
        ch, mh = _prequential(home, margin > line, folds)
        if float(line).is_integer():
            push = out[push_key].to_numpy(float)
            cp, mp = _prequential(push, margin == line, folds)
            away = out[away_key].to_numpy(float) if away_key in out else 1 - home - push
            ca, ma = _prequential(away, margin < line, folds)
            values = np.maximum(np.column_stack([ch, cp, ca]), 1e-9)
            values /= values.sum(axis=1, keepdims=True)
            out[key], out[push_key], out[away_key] = values.T
            bundle["run_lines"][str(line)] = {"home": mh, "push": mp, "away": ma}
        else:
            # Half stops have no push mass, but both sides are still calibrated
            # independently so the two-way probabilities remain coherent.
            away = out[away_key].to_numpy(float) if away_key in out else 1 - home
            ca, ma = _prequential(away, margin < line, folds)
            values = np.maximum(np.column_stack([ch, ca]), 1e-9)
            values /= values.sum(axis=1, keepdims=True)
            out[key], out[away_key] = values.T
            out[push_key] = 0.0
            bundle["run_lines"][str(line)] = {"home": mh, "push": None, "away": ma}
    if "p_home_win_derived" in out:
        p = out.p_home_win_derived.to_numpy(float)
        tie = out.get("p_tie", pd.Series(np.zeros(len(out)), index=out.index)).to_numpy(float)
        cal = _fit_derived_calibration(p, tie, margin)
        if cal:
            p, tie = _apply_derived_calibration(p, tie, cal)
            out["p_home_win_derived"] = p
            out["p_tie"] = tie
            bundle["derived_moneyline"] = cal
        out["p_away_win_derived"] = np.maximum(0.0, 1.0 - tie - p)
        bundle["derived_moneyline_tie_semantics"] = "three_way_unconditional"
    return out, bundle


def apply_market_calibration(df: pd.DataFrame, bundle: dict | None) -> pd.DataFrame:
    """Apply a fitted calibration bundle to a newly priced slate frame.

    OOF rows already carry prequential values from ``calibrate_market_frame``
    and are returned unchanged.  Slate rows receive the final maps learned
    from all OOF folds, with three-way probabilities renormalized.
    """
    out = df.copy()
    if not bundle or "method" not in bundle or not len(out):
        return out
    for line, maps in (bundle.get("totals") or {}).items():
        try:
            number = int(line)
        except (TypeError, ValueError):
            continue
        over_key, push_key, under_key = (_grid_key("p_over", number),
                                          _grid_key("p_push_total", number),
                                          _grid_key("p_under", number))
        if over_key not in out:
            continue
        over = _apply_platt(out[over_key].to_numpy(float), maps.get("over"))
        if number == int(number) and push_key in out:
            push = _apply_platt(out[push_key].to_numpy(float), maps.get("push"))
            under = _apply_platt(out[under_key].to_numpy(float), maps.get("under"))
            values = np.maximum(np.column_stack([over, push, under]), 1e-9)
            values /= values.sum(axis=1, keepdims=True)
            out[over_key], out[push_key], out[under_key] = values.T
        else:
            out[over_key] = over
    for line, maps in (bundle.get("run_lines") or {}).items():
        try:
            number = float(line)
        except (TypeError, ValueError):
            continue
        label = int(number) if number.is_integer() else number
        home_key, push_key, away_key = (_grid_key("p_home_cover", label),
                                        _grid_key("p_push", label),
                                        _grid_key("p_away_cover", label))
        if home_key not in out:
            continue
        home = _apply_platt(out[home_key].to_numpy(float), maps.get("home"))
        away = (_apply_platt(out[away_key].to_numpy(float), maps.get("away"))
                if away_key in out else 1.0 - home)
        if number.is_integer() and push_key in out:
            push = _apply_platt(out[push_key].to_numpy(float), maps.get("push"))
            values = np.maximum(np.column_stack([home, push, away]), 1e-9)
            values /= values.sum(axis=1, keepdims=True)
            out[home_key], out[push_key], out[away_key] = values.T
        else:
            values = np.maximum(np.column_stack([home, away]), 1e-9)
            values /= values.sum(axis=1, keepdims=True)
            out[home_key], out[away_key] = values.T
            out[push_key] = 0.0
    cal = bundle.get("derived_moneyline")
    if cal and "p_home_win_derived" in out:
        p = out.p_home_win_derived.to_numpy(float)
        tie = out.get("p_tie", pd.Series(np.zeros(len(out)), index=out.index)).to_numpy(float)
        p, tie = _apply_derived_calibration(p, tie, cal)
        out["p_home_win_derived"] = p
        out["p_tie"] = tie
        out["p_away_win_derived"] = np.maximum(0.0, 1.0 - tie - p)
    return out
