"""NFL run-line and totals distribution engine.

The production structure mirrors MLB:

* one LightGBM Poisson regressor for home score;
* one LightGBM Poisson regressor for away score;
* the active binary-moneyline feature contract is the sole source feature list;
* genuine NaNs remain intact for native LightGBM missing-value routing;
* NFL-specific negative-binomial dispersion is estimated from walk-forward OOF;
* one Monte Carlo score-pair sample supplies every total/margin probability.

The binary moneyline model is intentionally not imported or modified here.
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


def _make_reg():
    from lightgbm import LGBMRegressor
    params = dict(config.LIGHTGBM_REG_PARAMS)
    params["objective"] = "poisson"
    return LGBMRegressor(**params)


def _assert_run_line_contract(columns: list[str]) -> None:
    """The run line must consume EXACTLY the binary moneyline contract.

    MLB parity (run_engine.build_side_frame, strict parity mode): the run-line
    regressors receive the full active moneyline feature list, in order, with
    the team-ID pair appended and nothing else. This module owns no run-line
    list of its own, so the guarantee is checked rather than assumed — a
    divergence here means the run line silently priced a different feature set
    than the moneyline it is supposed to extend.
    """
    expected = list(config.active_moneyline_feature_cols())
    if list(columns[:len(expected)]) != expected:
        missing = [c for c in expected if c not in columns]
        extra = [c for c in columns[:len(expected)] if c not in expected]
        raise RuntimeError(
            "run-line feature set diverges from the binary moneyline contract "
            f"(expected {len(expected)} columns; missing={missing[:4]} "
            f"unexpected={extra[:4]}) — the run line and the moneyline must "
            "share one feature set")
    tail = list(columns[len(expected):])
    if tail != list(config.TREE_CATEGORICAL_COLS):
        raise RuntimeError(
            "run-line feature set must append exactly the team-ID pair "
            f"{list(config.TREE_CATEGORICAL_COLS)} after the moneyline "
            f"contract; found {tail}")


class ScoreRegressor:
    """Two same-contract LightGBM Poisson regressors, one per score side."""

    def __init__(self) -> None:
        self.home_model = _make_reg()
        self.away_model = _make_reg()
        self.feature_columns: list[str] = []

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        # The binary moneyline tree view WITH the categorical team-ID context
        # (config.TREE_CATEGORICAL_COLS): the same structural treatment the
        # binary tree members get (MLB parity — the run engine appends the same
        # RUN_TREE_CATEGORICAL_COLS pair after the active moneyline list). This
        # module never owns a run-line feature list, so the run line PULLS the
        # moneyline contract by construction; the pair rides features.tree_view
        # after the served columns. Do not fill NaN: LightGBM handles missing
        # values natively, like MLB's run engine.
        X = feat_mod.tree_view(df)
        if not self.feature_columns:
            self.feature_columns = list(X.columns)
            _assert_run_line_contract(self.feature_columns)
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
    """Fit NFL-specific NB dispersion from leakage-free OOF score predictions."""
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
    """Monte Carlo all NFL grid probabilities from paired score draws."""
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
            for line in config.TOTAL_GRID:
                row[_grid_key("p_over", line)] = float((t > line).mean())
                row[_grid_key("p_push", line)] = float((t == line).mean())
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
    # existing NFL artifact contract while the probabilities come from MC.
    return float(lines[int(np.argmin(np.abs(vals - 0.5)))])


def game_distribution(mu_h: float, mu_a: float,
                      *, alpha_home: float = 0.0,
                      alpha_away: float = 0.0,
                      n_draws: int = MC_DRAWS,
                      seed: int = MC_SEED) -> dict:
    """Single-game NB/MC output.

    The legacy ``sigma_margin``/``sigma_total`` keyword arguments are GONE,
    not ignored: this engine is the NB Monte Carlo (the shipped model), a
    sigma is a different (Gaussian) parameterization, and accepting the name
    while silently discarding its value is exactly how a caller comes to
    believe a variance was applied when none was. No production or test caller
    passed one (verified 2026-09-27); a caller that genuinely wants a
    different dispersion passes ``alpha_home``/``alpha_away``."""
    row = simulate_distributions(np.array([mu_h]), np.array([mu_a]),
                                  alpha_home, alpha_away, n_draws, seed).iloc[0].to_dict()
    # Preserve the historical in-memory negative labels used by direct unit
    # tests; serving.py converts them to the MLB-style mN artifact labels.
    for line in config.SPREAD_GRID:
        if line < 0:
            row[f"p_home_cover_{line}"] = row[_grid_key("p_home_cover", line)]
            row[f"p_push_{line}"] = row[_grid_key("p_push", line)]
    return row


def apply_distribution(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Expand mu predictions into the complete NB/MC market grid."""
    # Numeric positional arguments are retained for legacy unit callers; the
    # production pipeline passes the NB parameter dictionary.
    if not isinstance(params, dict):
        params = {}
    ah = float(params.get("alpha_home", 0.0))
    aa = float(params.get("alpha_away", 0.0))
    return simulate_distributions_into(df, ah, aa)


def simulate_distributions_into(df: pd.DataFrame, alpha_home: float,
                                alpha_away: float) -> pd.DataFrame:
    """Attach the MC grid columns to ``df`` (dropping any colliding ones)."""
    dist = simulate_distributions(df["mu_h"].to_numpy(float),
                                  df["mu_a"].to_numpy(float), alpha_home, alpha_away)
    base = df.drop(columns=[c for c in dist.columns if c in df.columns], errors="ignore")
    return pd.concat([base.reset_index(drop=True), dist.reset_index(drop=True)], axis=1)


def walk_forward_oof(game_df: pd.DataFrame, date_col: str = "gameday",
                     fold_list: list | None = None, progress=None) -> dict:
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
        # Optional display-only progress hook; default None skips the call,
        # so an unhooked run is byte-identical to before.
        if progress is not None:
            progress()
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
    # Numerically stable logistic.  The naive 1/(1+exp(-x)) form overflows exp
    # for x < -709, which a confident slate probability reaches routinely, and
    # the RuntimeWarning it prints is a symptom of an avoidable loss of
    # precision, not a number that is merely large.  Branching on the sign of
    # x keeps the exponent non-positive on both sides, so exp never overflows
    # and the result is identical to the naive form in the safe range.
    x = cal["a"] * z + cal["b"]
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return np.clip(out, 1e-7, 1 - 1e-7)


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
    """Calibrate every published NFL total/run-line grid independently.

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
        push = out[_grid_key("p_push", line)].to_numpy(float)
        under = out[_grid_key("p_under", line)].to_numpy(float)
        co, mo = _prequential_line(over, (total > line).astype(int), folds)
        if line == int(line):
            cp, mp = _prequential_line(push, (total == line).astype(int), folds)
            cu, mu = _prequential_line(under, (total < line).astype(int), folds)
            vals = np.maximum(np.column_stack([co, cp, cu]), 1e-9)
            vals /= vals.sum(axis=1, keepdims=True)
            out[key], out[_grid_key("p_push", line)], out[_grid_key("p_under", line)] = vals.T
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
        ch, mh = _prequential_line(home, (margin > line).astype(int), folds)
        if line == int(line):
            # The away leg is an EVENT with its own history (margin < L), not
            # the residual of two home-anchored maps. Deriving it as
            # 1 - cal(home) - cal(push) embedded the home leg's Platt slope in
            # the away price, and a 1627-row sample of the shipped artifact
            # showed the residual sitting up to 0.036 from the away leg the
            # away map itself produces -- a systematic away-side bias, largest
            # exactly at the deep lines the card quotes. Calibrate the third
            # leg from its own outcome, exactly like MLB's run-engine away
            # -favorite block (run_engine.py: p_rl_away_fav_grid), then
            # normalize the three.
            cp, mp = _prequential_line(push, (margin == line).astype(int), folds)
            away = np.maximum(1.0 - home - push, 1e-9)
            ca, ma = _prequential_line(away, (margin < line).astype(int), folds)
            vals = np.maximum(np.column_stack([ch, cp, ca]), 1e-9)
            vals /= vals.sum(axis=1, keepdims=True)
            out[key], out[_grid_key("p_push", line)] = vals[:, 0], vals[:, 1]
            out[_grid_key("p_away_cover", line)] = vals[:, 2]
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
        # The tie leg must be the CALIBRATED push-at-0 (the spread loop above
        # already recalibrated it), not the raw MC tie this block used to
        # read two lines later get overwritten anyway. On the shipped
        # 2026-09-27 OOF store the raw/calibrated gap made the published
        # p_away_win_derived incoherent with the published p_tie on up to
        # 0.036 per row, and on AWAY-favored rows the outright leg lost the
        # tie mass outright (away = pc - tie instead of pc).
        tie_col = out.get(_grid_key("p_push", 0))
        if tie_col is not None:
            tie = tie_col.to_numpy(float)
        else:
            _t = out.get("p_tie")
            tie = (_t.to_numpy(float) if _t is not None
                   else np.zeros(len(out)))
        # Favorite side carries the favored map; the dog outright is the
        # coherent residual 1 - favorite - tie (never pc - tie, which double-
        # subtracted the tie on away-favored rows).
        dog = np.maximum(1.0 - pc - tie, 1e-9)
        out["p_home_win_derived"] = np.where(fav_home, pc, dog)
        out["p_away_win_derived"] = np.where(fav_home, dog, pc)
        # Keep the published tie coherent with the pair just normalized: the
        # raw MC tie this column still carried (apply_distribution wrote it)
        # disagrees with the calibrated push-at-0 the residual above consumed,
        # and the pre-fix artifact shipped rows summing to 0.96-1.03.
        out["p_tie"] = tie
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
            cp = _apply_platt(out[_grid_key("p_push", line)].to_numpy(float), rec.get("push"))
            cu = _apply_platt(out[_grid_key("p_under", line)].to_numpy(float), rec.get("under"))
            vals = np.maximum(np.column_stack([co, cp, cu]), 1e-9)
            vals /= vals.sum(axis=1, keepdims=True)
            out[key], out[_grid_key("p_push", line)], out[_grid_key("p_under", line)] = vals.T
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
            # The away leg applies its OWN map to its OWN raw event (built
            # from the raw home + raw push legs, pre-calibration -- mixing a
            # calibrated leg into the raw residual would double-count the
            # home map). Falls back to the 1-home-push residual only when the
            # bundle predates the away map, and mirrors calibrate_market_frame
            # so the artifact stays the single source of the three-way split.
            raw_away = np.maximum(1.0 - raw_home - raw_push, 1e-9)
            if rec.get("away") is not None:
                ca = _apply_platt(raw_away, rec.get("away"))
            else:
                ca = np.maximum(1.0 - ch - cp, 1e-9)
            vals = np.maximum(np.column_stack([ch, cp, ca]), 1e-9)
            vals /= vals.sum(axis=1, keepdims=True)
            out[key], out[_grid_key("p_push", line)] = vals[:, 0], vals[:, 1]
            # Always materialize the away leg (slate rows ship it too, not
            # just OOF rows): a reader deriving it as 1 - home - push gets
            # the same number either way now, and the artifact is honest
            # about the away side being its own calibrated event.
            out[_grid_key("p_away_cover", line)] = vals[:, 2]
        else:
            out[key] = ch
    if _grid_key("p_push", 0) in out:
        out["p_tie"] = out[_grid_key("p_push", 0)]
    # Derived-ML pair: the slate frame carries the RAW pair from
    # apply_distribution, while the OOF frame was normalized through the
    # favored map in calibrate_market_frame -- the same raw/calibrated
    # incoherence that made the shipped 2026-09-27 OOF rows sum to 0.96-1.03.
    # Apply the stored favored map here so both paths publish from ONE
    # contract, then keep the dog as the residual of the SAME tie the pair
    # was normalized against.
    if "p_home_win_derived" in out and bundle.get("derived_moneyline") is not None:
        p = out["p_home_win_derived"].to_numpy(float)
        fav_home = p >= 0.5
        pf = np.where(fav_home, p, 1.0 - p)
        pc = np.maximum(0.5, _apply_platt(pf, bundle["derived_moneyline"]))
        _t = out.get("p_tie")
        tie = (_t.to_numpy(float) if _t is not None else np.zeros(len(out)))
        dog = np.maximum(1.0 - pc - tie, 1e-9)
        out["p_home_win_derived"] = np.where(fav_home, pc, dog)
        out["p_away_win_derived"] = np.where(fav_home, dog, pc)
    if "fair_spread" in out:
        out["p_cover_fair"] = [float(r[_grid_key("p_home_cover", r["fair_spread"])])
                               for _, r in out.iterrows()]
    if "fair_total" in out:
        out["p_over_fair"] = [float(r[_grid_key("p_over", r["fair_total"])])
                              for _, r in out.iterrows()]
    return out
