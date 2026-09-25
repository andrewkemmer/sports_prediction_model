"""NHL run-line and totals distribution engine.

Structural mirror of the NFL distributions.py (MLB lineage):

* one LightGBM Poisson regressor for home goals;
* one LightGBM Poisson regressor for away goals;
* the active binary-moneyline feature contract is the sole source feature list;
* genuine NaNs remain intact for native LightGBM missing-value routing;
* NHL-specific negative-binomial dispersion is estimated from walk-forward OOF;
* one Monte Carlo score-pair sample supplies every total/margin probability.

The binary moneyline model is intentionally not imported or modified here.
Grids are NHL-sized: spread -8..+8 (goals), totals 4..12, sigma ≈ 2.2.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    from backend import config
    from backend import folds as folds_mod
    from backend import features as feat_mod
except ImportError:
    import config
    import folds as folds_mod
    import features as feat_mod

logger = logging.getLogger(__name__)

MC_DRAWS = 10_000
MC_SEED = 42
ALPHA_FLOOR = 1e-8
ALPHA_CAP = 2.0

# Retained for backwards-compatible diagnostics/tests; production uses NB MC.
MARGIN_SUPPORT = np.arange(-config.MARGIN_PMF_MAX, config.MARGIN_PMF_MAX + 1)
TOTAL_SUPPORT = np.arange(0, config.TOTAL_PMF_MAX + 1)


def discrete_normal_pmf(mu: float, sigma: float, support: np.ndarray) -> np.ndarray:
    """Compatibility helper for old diagnostics; not the production sampler."""
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
        return np.full(len(support), np.nan)
    z = (support - mu) / sigma
    p = np.exp(-0.5 * z * z)
    return p / p.sum()


def _pmf_median(pmf: np.ndarray, support: np.ndarray) -> float:
    """Compatibility median helper for legacy diagnostics."""
    if not np.isfinite(pmf).all():
        return np.nan
    return float(support[min(int(np.searchsorted(np.cumsum(pmf), 0.5)),
                           len(support) - 1)])


def margin_cdf_above(pmf: np.ndarray, support: np.ndarray, line: float) -> float:
    threshold = int(np.floor(line)) + 1
    return float(pmf[support >= threshold].sum()) if np.isfinite(pmf).all() else np.nan


def margin_pmf_at(pmf: np.ndarray, support: np.ndarray, line: float) -> float:
    if line != int(line):
        return 0.0
    return float(pmf[support == int(line)].sum()) if np.isfinite(pmf).all() else np.nan


def total_probabilities(pmf: np.ndarray, support: np.ndarray, line: float) -> tuple[float, float, float]:
    if not np.isfinite(pmf).all():
        return np.nan, np.nan, np.nan
    push = float(pmf[support == int(line)].sum()) if line == int(line) else 0.0
    over = float(pmf[support > line].sum())
    under = float(pmf[support < line].sum())
    return over, push, under


def _make_reg():
    from lightgbm import LGBMRegressor
    params = dict(config.LIGHTGBM_REG_PARAMS)
    params["objective"] = "poisson"
    return LGBMRegressor(**params)


class ScoreRegressor:
    """Two same-contract LightGBM Poisson regressors, one per score side."""

    def __init__(self) -> None:
        self.home_model = _make_reg()
        self.away_model = _make_reg()
        self.feature_columns: list[str] = []

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        # The binary moneyline tree view WITH the categorical team-ID context
        # (config.TREE_CATEGORICAL_COLS): the same structural treatment the
        # binary tree members get (MLB parity — the run-engine regressors
        # there also see the team-ID pair). This module never owns a run-line
        # feature list, so the run line PULLS the moneyline contract by
        # construction. Do not fill NaN: LightGBM handles missing natively.
        X = feat_mod.tree_view(df)
        if not self.feature_columns:
            self.feature_columns = list(X.columns)
        return X.reindex(columns=self.feature_columns)

    def fit(self, df: pd.DataFrame) -> "ScoreRegressor":
        X = self._matrix(df)
        self.home_model.fit(X, pd.to_numeric(df["home_score"], errors="coerce"))
        self.away_model.fit(X, pd.to_numeric(df["away_score"], errors="coerce"))
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = self._matrix(df)
        return (
            np.clip(self.home_model.predict(X), 1e-6, None),
            np.clip(self.away_model.predict(X), 1e-6, None),
        )


def estimate_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    """Estimate NB alpha from OOF residual dispersion.

    For NB variance ``mu + alpha*mu²``, the method-of-moments estimate is the
    non-negative ratio of excess squared residuals to squared means. A value
    near zero is the Poisson limit.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    ok = np.isfinite(y) & np.isfinite(mu) & (mu > 0)
    if ok.sum() < 2:
        return 0.0
    excess = np.sum((y[ok] - mu[ok]) ** 2 - y[ok])
    denom = np.sum(mu[ok] ** 2)
    return float(np.clip(max(excess / max(denom, 1e-12), 0.0), 0.0, ALPHA_CAP))


def calibrate_dispersion(oof: pd.DataFrame) -> dict[str, float]:
    """Fit NHL-specific NB dispersion from leakage-free OOF score predictions."""
    ah = estimate_alpha(oof["home_score"], oof["mu_h"])
    aa = estimate_alpha(oof["away_score"], oof["mu_a"])
    return {"alpha_home": ah, "alpha_away": aa,
            "distribution": "negative_binomial",
            "poisson_limit": bool(max(ah, aa) <= ALPHA_FLOOR),
            "mc_draws": MC_DRAWS}


def _nb_draws(mu: np.ndarray, alpha: float, rng: np.random.Generator,
              n_draws: int) -> np.ndarray:
    """Draw NB(mu, alpha); alpha≈0 uses Poisson exactly."""
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-6)
    if alpha <= ALPHA_FLOOR:
        return rng.poisson(mu[:, None], size=(len(mu), n_draws)).astype(np.int16)
    size = np.full(len(mu), 1.0 / max(alpha, ALPHA_FLOOR))
    prob = size / (size + mu)
    return rng.negative_binomial(size[:, None], prob[:, None],
                                 size=(len(mu), n_draws)).astype(np.int16)


def _grid_key(prefix: str, value: float | int) -> str:
    s = str(float(value))
    if s.endswith(".0"):
        s = s[:-2]
    return f"{prefix}_{s.replace('-', 'm').replace('.', '_')}"


def simulate_distributions(mu_h: np.ndarray, mu_a: np.ndarray,
                           alpha_home: float, alpha_away: float,
                           n_draws: int = MC_DRAWS,
                           seed: int = MC_SEED) -> pd.DataFrame:
    """Monte Carlo all NHL grid probabilities from paired score draws."""
    rng = np.random.default_rng(seed)
    mu_h = np.asarray(mu_h, dtype=float)
    mu_a = np.asarray(mu_a, dtype=float)
    # Keep memory bounded while retaining deterministic per-row output.
    rows: list[dict[str, Any]] = []
    chunk = max(1, min(len(mu_h), 2_000_000 // max(n_draws, 1)))
    for start in range(0, len(mu_h), chunk):
        end = min(start + chunk, len(mu_h))
        h = _nb_draws(mu_h[start:end], alpha_home, rng, n_draws)
        a = _nb_draws(mu_a[start:end], alpha_away, rng, n_draws)
        total = h + a
        margin = h - a
        for i in range(end - start):
            t = total[i]
            m = margin[i]
            row: dict[str, Any] = {
                "mu_h": float(mu_h[start + i]),
                "mu_a": float(mu_a[start + i]),
                "mu_margin": float(mu_h[start + i] - mu_a[start + i]),
                "mu_total": float(mu_h[start + i] + mu_a[start + i]),
                "p_home_win_derived": float((m > 0).mean()),
                "p_away_win_derived": float((m < 0).mean()),
                "p_tie": float((m == 0).mean()),
            }
            # Integer spread grid and half-stop lines.
            for line in config.SPREAD_GRID:
                row[_grid_key("p_home_cover", line)] = float((m > line).mean())
                row[_grid_key("p_push", line)] = float((m == line).mean())
            for line in config.HALF_STOP_LINES:
                row[_grid_key("p_home_cover", line)] = float((m > line).mean())
            # Integer totals grid. The same draws provide over/push/under.
            # Totals pushes live in their OWN namespace (p_push_total_{U}):
            # the NHL spread grid (-8..+8) overlaps the totals grid (4..12),
            # so a shared p_push_{N} column would collide with the margin
            # push — MLB/NFL never hit this because their grid ranges are
            # disjoint. The artifact writer back-compat maps any legacy
            # p_push_{U} column.
            for line in config.TOTAL_GRID:
                row[_grid_key("p_over", line)] = float((t > line).mean())
                row[_grid_key("p_push_total", line)] = float((t == line).mean())
                row[_grid_key("p_under", line)] = float((t < line).mean())
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["fair_spread"] = out.apply(
            lambda r: _fair_from_grid(r, "p_home_cover", config.SPREAD_GRID), axis=1)
        out["fair_total"] = out.apply(
            lambda r: _fair_from_grid(r, "p_over", config.TOTAL_GRID), axis=1)
        out["p_cover_fair"] = [
            float(r[_grid_key("p_home_cover", r["fair_spread"])])
            for _, r in out.iterrows()
        ]
        out["p_over_fair"] = [
            float(r[_grid_key("p_over", r["fair_total"])])
            for _, r in out.iterrows()
        ]
    return out


def _fair_from_grid(row: pd.Series, prefix: str, lines: list) -> float:
    vals = np.array([float(row[_grid_key(prefix, x)]) for x in lines])
    # The fair line is the closest grid threshold to 50%, preserving the
    # existing artifact contract while the probabilities come from MC.
    return float(lines[int(np.argmin(np.abs(vals - 0.5)))])


def game_distribution(mu_h: float, mu_a: float,
                      sigma_margin: float | None = None,
                      sigma_total: float | None = None,
                      *, alpha_home: float = 0.0,
                      alpha_away: float = 0.0,
                      n_draws: int = MC_DRAWS,
                      seed: int = MC_SEED) -> dict:
    """Single-game NB/MC output; sigma args remain accepted for compatibility."""
    row = simulate_distributions(np.array([mu_h]), np.array([mu_a]),
                                  alpha_home, alpha_away, n_draws, seed).iloc[0].to_dict()
    # Preserve the historical in-memory negative labels used by direct unit
    # tests; serving.py converts them to the MLB-style mN artifact labels.
    for line in config.SPREAD_GRID:
        if line < 0:
            row[f"p_home_cover_{line}"] = row[_grid_key("p_home_cover", line)]
            row[f"p_push_{line}"] = row[_grid_key("p_push", line)]
    return row


def apply_distribution(df: pd.DataFrame, params: dict | None = None,
                       sigma_total: float | None = None) -> pd.DataFrame:
    """Expand mu predictions into the complete NB/MC market grid."""
    if not isinstance(params, dict):
        params = {}
    ah = float(params.get("alpha_home", 0.0))
    aa = float(params.get("alpha_away", 0.0))
    dist = simulate_distributions(df["mu_h"].to_numpy(float),
                                  df["mu_a"].to_numpy(float), ah, aa)
    base = df.drop(columns=[c for c in dist.columns if c in df.columns], errors="ignore")
    return pd.concat([base.reset_index(drop=True), dist.reset_index(drop=True)], axis=1)


def walk_forward_oof(game_df: pd.DataFrame, date_col: str = "gameday",
                     fold_list: list | None = None) -> dict:
    """Fit two LightGBM Poisson models on shared walk-forward folds."""
    df = folds_mod.canonical_sort(game_df, date_col)
    fold_list = fold_list if fold_list is not None else folds_mod.make_folds(df, date_col=date_col)
    parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    for fold in fold_list:
        train, val = df.loc[fold.train_idx], df.loc[fold.val_idx]
        try:
            reg = ScoreRegressor().fit(train)
            mu_h, mu_a = reg.predict(val)
        except Exception as exc:  # noqa: BLE001
            logger.warning("dist OOF fold %s failed: %s", fold.fold_id, exc)
            mu_h = np.full(len(val), np.nan)
            mu_a = np.full(len(val), np.nan)
        part = pd.DataFrame({
            "game_id": val["game_id"].to_numpy(),
            "gameday": pd.to_datetime(val[date_col]).to_numpy(),
            "season": val["season"].to_numpy(),
            "fold_id": fold.fold_id,
            "mu_h": mu_h, "mu_a": mu_a,
            "home_score": val["home_score"].astype(float).to_numpy(),
            "away_score": val["away_score"].astype(float).to_numpy(),
        })
        part["margin"] = part["home_score"] - part["away_score"]
        part["total"] = part["home_score"] + part["away_score"]
        parts.append(part)
        fold_rows.append({"fold_id": fold.fold_id,
                          "val_start": str(fold.val_start.date()),
                          "val_end": str(fold.val_end.date()),
                          "n_train": int(len(train)), "n_val": int(len(val))})
    oof = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(oof):
        oof["resid_margin"] = oof["margin"] - (oof["mu_h"] - oof["mu_a"])
        oof["resid_total"] = oof["total"] - (oof["mu_h"] + oof["mu_a"])
    return {"oof": oof, "fold_table": pd.DataFrame(fold_rows)}


def fit_final(game_df: pd.DataFrame) -> ScoreRegressor:
    return ScoreRegressor().fit(game_df)


def _fit_platt(p: np.ndarray, y: np.ndarray) -> dict | None:
    """Local Platt fit for distributional lines; avoids a moneyline import cycle."""
    from sklearn.linear_model import LogisticRegression
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, int)
    if len(p) < 30 or len(np.unique(y)) < 2:
        return None
    z = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(z, y)
    return {"a": round(float(model.coef_[0, 0]), 6),
            "b": round(float(model.intercept_[0]), 6)}


def _apply_platt(p: np.ndarray, cal: dict | None) -> np.ndarray:
    if not cal:
        return np.asarray(p, float)
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    z = np.log(p / (1.0 - p))
    return np.clip(1.0 / (1.0 + np.exp(-(cal["a"] * z + cal["b"]))),
                   1e-7, 1 - 1e-7)


def _prequential_line(raw: np.ndarray, y: np.ndarray,
                      folds: np.ndarray) -> tuple[np.ndarray, dict | None]:
    """Apply a prior-fold-only Platt map and return the all-OOF final map."""
    raw, y, folds = np.asarray(raw, float), np.asarray(y, int), np.asarray(folds)
    out = raw.copy()
    history_p: list[np.ndarray] = []
    history_y: list[np.ndarray] = []
    for fold in np.unique(folds):
        m = folds == fold
        cal = _fit_platt(np.concatenate(history_p) if history_p else [],
                         np.concatenate(history_y) if history_y else [])
        if cal:
            out[m] = _apply_platt(raw[m], cal)
        history_p.append(raw[m])
        history_y.append(y[m])
    return out, _fit_platt(raw, y)


def calibrate_market_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Calibrate every published NHL total/run-line grid independently.

    Whole-number lines calibrate over/push/under separately and normalize the
    three outputs. Half-stop run lines calibrate the favored-side binary event.
    The returned map is reusable for the current slate.
    """
    out = df.copy()
    bundle: dict[str, Any] = {"method": "prequential_platt",
                              "scope": "line_specific", "totals": {},
                              "run_lines": {}, "derived_moneyline": None}
    if not len(out) or "fold_id" not in out:
        return out, bundle
    folds = out["fold_id"].to_numpy()
    total = out["total"].to_numpy(float)
    margin = out["margin"].to_numpy(float)
    for line in config.TOTAL_GRID:
        key = _grid_key("p_over", line)
        if key not in out:
            continue
        over = out[key].to_numpy(float)
        push = out[_grid_key("p_push_total", line)].to_numpy(float)
        under = out[_grid_key("p_under", line)].to_numpy(float)
        co, mo = _prequential_line(over, (total > line).astype(int), folds)
        if line == int(line):
            cp, mp = _prequential_line(push, (total == line).astype(int), folds)
            cu, mu = _prequential_line(under, (total < line).astype(int), folds)
            vals = np.maximum(np.column_stack([co, cp, cu]), 1e-9)
            vals /= vals.sum(axis=1, keepdims=True)
            out[key], out[_grid_key("p_push_total", line)], out[_grid_key("p_under", line)] = vals.T
            bundle["totals"][str(line)] = {"over": mo, "push": mp, "under": mu}
        else:
            out[key] = co
            bundle["totals"][str(line)] = {"over": mo, "push": None, "under": None}
    for line in config.SPREAD_GRID:
        key = _grid_key("p_home_cover", line)
        if key not in out:
            continue
        home = out[key].to_numpy(float)
        push = out[_grid_key("p_push", line)].to_numpy(float)
        away = np.maximum(1.0 - home - push, 1e-9)
        ch, mh = _prequential_line(home, (margin > line).astype(int), folds)
        if line == int(line):
            cp, mp = _prequential_line(push, (margin == line).astype(int), folds)
            ca, ma = _prequential_line(away, (margin < line).astype(int), folds)
            vals = np.maximum(np.column_stack([ch, cp, ca]), 1e-9)
            vals /= vals.sum(axis=1, keepdims=True)
            out[key], out[_grid_key("p_push", line)] = vals[:, 0], vals[:, 1]
            bundle["run_lines"][str(line)] = {"home": mh, "push": mp, "away": ma}
        else:
            out[key] = ch
            bundle["run_lines"][str(line)] = {"home": mh, "push": None, "away": None}
    # Derived model moneyline uses the same favored-team calibration contract.
    if "p_home_win_derived" in out:
        p = out["p_home_win_derived"].to_numpy(float)
        fav_home = p >= 0.5
        pf = np.where(fav_home, p, 1.0 - p)
        yf = np.where(fav_home, margin > 0, margin < 0).astype(int)
        cal = _fit_platt(pf, yf)
        pc = np.maximum(0.5, _apply_platt(pf, cal)) if cal else pf
        out["p_home_win_derived"] = np.where(fav_home, pc, 1.0 - pc)
        out["p_away_win_derived"] = 1.0 - out["p_home_win_derived"] - out.get("p_tie", 0.0)
        bundle["derived_moneyline"] = cal
    # Recompute fair-line aliases from calibrated grid columns.
    if "fair_spread" in out:
        out["p_cover_fair"] = [float(r[_grid_key("p_home_cover", r["fair_spread"])])
                               for _, r in out.iterrows()]
    if "fair_total" in out:
        out["p_over_fair"] = [float(r[_grid_key("p_over", r["fair_total"])])
                              for _, r in out.iterrows()]
    if _grid_key("p_push", 0) in out:
        out["p_tie"] = out[_grid_key("p_push", 0)]
    return out, bundle


def apply_market_calibration(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Apply final line calibrators to a slate frame (no outcomes required)."""
    out = df.copy()
    for line in config.TOTAL_GRID:
        key = _grid_key("p_over", line)
        if key not in out:
            continue
        rec = bundle.get("totals", {}).get(str(line), {})
        co = _apply_platt(out[key].to_numpy(float), rec.get("over"))
        if line == int(line):
            cp = _apply_platt(out[_grid_key("p_push_total", line)].to_numpy(float), rec.get("push"))
            cu = _apply_platt(out[_grid_key("p_under", line)].to_numpy(float), rec.get("under"))
            vals = np.maximum(np.column_stack([co, cp, cu]), 1e-9)
            vals /= vals.sum(axis=1, keepdims=True)
            out[key], out[_grid_key("p_push_total", line)], out[_grid_key("p_under", line)] = vals.T
        else:
            out[key] = co
    for line in config.SPREAD_GRID:
        key = _grid_key("p_home_cover", line)
        if key not in out:
            continue
        rec = bundle.get("run_lines", {}).get(str(line), {})
        raw_home = out[key].to_numpy(float)
        raw_push = out[_grid_key("p_push", line)].to_numpy(float)
        ch = _apply_platt(raw_home, rec.get("home"))
        if line == int(line):
            cp = _apply_platt(raw_push, rec.get("push"))
            ca = _apply_platt(1.0 - raw_home - raw_push, rec.get("away"))
            vals = np.maximum(np.column_stack([ch, cp, ca]), 1e-9)
            vals /= vals.sum(axis=1, keepdims=True)
            out[key], out[_grid_key("p_push", line)] = vals[:, 0], vals[:, 1]
        else:
            out[key] = ch
    if _grid_key("p_push", 0) in out:
        out["p_tie"] = out[_grid_key("p_push", 0)]
    if "fair_spread" in out:
        out["p_cover_fair"] = [float(r[_grid_key("p_home_cover", r["fair_spread"])])
                               for _, r in out.iterrows()]
    if "fair_total" in out:
        out["p_over_fair"] = [float(r[_grid_key("p_over", r["fair_total"])])
                              for _, r in out.iterrows()]
    return out


def calibrate_sigma(resid_margin: np.ndarray, resid_total: np.ndarray) -> dict:
    """Compatibility shim; callers should use calibrate_dispersion."""
    return {"sigma_margin": float(np.nanstd(resid_margin)),
            "sigma_total": float(np.nanstd(resid_total)),
            "alpha_home": 0.0, "alpha_away": 0.0,
            "distribution": "negative_binomial", "mc_draws": MC_DRAWS}
