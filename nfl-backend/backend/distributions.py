"""NFL PIT distribution engine: score features plus CRPS CDF calibration.

Production uses the existing market-free moneyline tree feature contract.
Score means are cross-fitted on strictly prior dates. Total and margin CDF
location/width maps optimize discrete CRPS on prior OOF forecasts, and are
admitted using prior-origin paired CRPS evidence. A single monotone CDF
supplies every line, push, half-point fair line and derived moneyline.
NBA uses high-count discretized normal marginals; NFL/NHL use exact count
convolution with sport-fitted residual dispersion. Final NBA/NHL ties are
resolved before scoring; NFL retains tie mass. No sportsbook is an input.

Legacy Poisson/NB simulation, alpha-curve and undated Platt helpers remain
for compatibility diagnostics only: they do NOT price dated production.
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
    """Outcome-free analytic CDF; final CRPS maps are applied by calibration.

    Legacy alpha/MC params describe diagnostics only, not these prices.
    """
    return _distribution_price(df, {})


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
        train_dates = pd.to_datetime(train[date_col], errors="raise").dt.normalize()
        val_dates = pd.to_datetime(val[date_col], errors="raise").dt.normalize()
        if train_dates.max() >= val_dates.min():
            raise ValueError("distribution training must precede validation dates")
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


def calibrate_market_frame(df):
    """Dated production frames use causal whole-CDF CRPS calibration.

    Undated legacy diagnostic fixtures are not claimed as PIT evaluation.
    """
    if "gameday" in df and len(df):
        priced, bundle = causal_distribution_frame(df)
        metrics = distribution_score_report(priced, bundle)
        bundle["oof_metrics"] = metrics
        priced.attrs["distribution_metrics"] = metrics
        return priced, bundle
    return _legacy_calibrate_market_frame(df)


def _legacy_calibrate_market_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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
    # Calibrated grid columns are BATCHED here and materialized in ONE wide
    # pd.concat after the loops. Assigning ~70 grid columns one-by-one into
    # the frame fragmented it so badly pandas printed PerformanceWarning
    # ("DataFrame is highly fragmented ... consider pd.concat(axis=1)") per
    # slate row on every run. Reads stay inside the loops (each line reads
    # only its own RAW columns), so batching cannot change a value.
    grid_updates: dict[str, np.ndarray] = {}
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
            grid_updates[key] = vals[:, 0]
            grid_updates[_grid_key("p_push", line)] = vals[:, 1]
            grid_updates[_grid_key("p_under", line)] = vals[:, 2]
            bundle["totals"][str(line)] = {"over": mo, "push": mp, "under": mu}
        else:
            grid_updates[key] = co
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
            grid_updates[key] = vals[:, 0]
            grid_updates[_grid_key("p_push", line)] = vals[:, 1]
            grid_updates[_grid_key("p_away_cover", line)] = vals[:, 2]
            bundle["run_lines"][str(line)] = {"home": mh, "push": mp, "away": ma}
        else:
            grid_updates[key] = ch
            bundle["run_lines"][str(line)] = {"home": mh, "push": None, "away": None}
    if grid_updates:
        out = pd.concat(
            [out.drop(columns=[c for c in grid_updates if c in out.columns],
                      errors="ignore"),
             pd.DataFrame(grid_updates, index=out.index)], axis=1)
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
    """Apply final line calibrators to a slate frame (no outcomes required).

    The calibrated grid columns are BATCHED here and materialized in ONE wide
    pd.concat after the loops, mirroring calibrate_market_frame: per-column
    assignment into a slate frame fragmented it enough for pandas to print a
    PerformanceWarning per row of the serving slate.
    """
    if bundle and bundle.get("method") == "causal_lattice_crps":
        out = _distribution_price(df, bundle.get("maps", {}))
        out["distribution_fit_n"] = bundle.get("fit_n", 0)
        origins = bundle.get("origins") or []
        out["distribution_fit_end"] = origins[-1]["end"] if origins else None
        return out

    out = df.copy()
    grid_updates: dict[str, np.ndarray] = {}
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
            grid_updates[key] = vals[:, 0]
            grid_updates[_grid_key("p_push", line)] = vals[:, 1]
            grid_updates[_grid_key("p_under", line)] = vals[:, 2]
        else:
            grid_updates[key] = co
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
            grid_updates[key] = vals[:, 0]
            grid_updates[_grid_key("p_push", line)] = vals[:, 1]
            # Always materialize the away leg (slate rows ship it too, not
            # just OOF rows): a reader deriving it as 1 - home - push gets
            # the same number either way now, and the artifact is honest
            # about the away side being its own calibrated event.
            grid_updates[_grid_key("p_away_cover", line)] = vals[:, 2]
        else:
            grid_updates[key] = ch
    if grid_updates:
        out = pd.concat(
            [out.drop(columns=[c for c in grid_updates if c in out.columns],
                      errors="ignore"),
             pd.DataFrame(grid_updates, index=out.index)], axis=1)
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


# ---------------------------------------------------------------------------
# Causal whole-distribution layer. Mirrored across sports deliberately: these
# backends run independently in Kaggle; no new shared runtime is required.
# ---------------------------------------------------------------------------
DISTRIBUTION_SPORT = "nfl"
CRPS_MIN_HISTORY = 200


def _distribution_blocks(frame):
    """Date-ordered, indivisible forecast origins; never trust fold label order."""
    date_col = "game_date" if DISTRIBUTION_SPORT == "mlb" else "gameday"
    fold_col = "fold_idx" if DISTRIBUTION_SPORT == "mlb" else "fold_id"
    dates = pd.to_datetime(frame[date_col], errors="coerce", utc=True).dt.normalize()
    if dates.isna().any():
        raise ValueError("distribution evaluation requires finite forecast dates")
    labels = frame[fold_col] if fold_col in frame else dates
    if labels.isna().any():
        raise ValueError("distribution evaluation requires finite origin labels")
    blocks = []
    for label in pd.unique(labels):
        mask = (labels == label).to_numpy()
        blocks.append((dates[mask].min(), dates[mask].max(), mask))
    blocks.sort(key=lambda b: b[0])
    for previous, current in zip(blocks, blocks[1:]):
        if previous[1] >= current[0]:
            raise ValueError("distribution forecast blocks overlap in calendar time")
    return blocks


def _distribution_inputs(frame):
    hc, ac = (("home_expected_runs", "away_expected_runs")
              if DISTRIBUTION_SPORT == "mlb" else ("mu_h", "mu_a"))
    h, a = frame[hc].to_numpy(float), frame[ac].to_numpy(float)
    if not (np.isfinite(h).all() and np.isfinite(a).all()):
        raise ValueError("non-finite score predictions cannot be priced")
    if np.any(h <= 0.) or np.any(a <= 0.):
        raise ValueError("expected scores must be positive")
    return h, a


def _distribution_support(kind):
    # Fixed, outcome-independent integration ranges. Saturation is checked,
    # not hidden by clipping observations at the integration boundary.
    high = {"nba": 600, "nfl": 250, "nhl": 100, "mlb": 160}[DISTRIBUTION_SPORT]
    return np.arange(0, high + 1) if kind == "total" else np.arange(-high, high + 1)


def _distribution_design(h, a):
    reference = {"nba": 220., "nfl": 44., "nhl": 6., "mlb": 9.}[DISTRIBUTION_SPORT]
    return np.column_stack([np.ones(len(h)),
                            np.clip((h + a) / reference - 1., -1., 2.),
                            np.clip(np.abs(h - a) / reference, 0., 1.)])


def _distribution_raw_cdf(h, a, kind, alphas=None):
    """Outcome-free sport baseline; discrete supports and exact CDFs, no MC."""
    from scipy.special import ndtr
    from scipy.stats import poisson, nbinom
    from scipy.signal import fftconvolve
    support = _distribution_support(kind)
    if DISTRIBUTION_SPORT == "nba":
        location = h + a if kind == "total" else h - a
        # A baseline, not fitted dispersion. Conditional width is subsequently
        # learned by CRPS on strictly earlier OOF predictions.
        scale = np.sqrt(np.maximum(h + a, 1.))
        cdf = ndtr((support[None, :] + .5 - location[:, None]) / scale[:, None])
        if kind == "total":
            lower = ndtr((-.5 - location) / scale)
            cdf = (cdf - lower[:, None]) / (1. - lower[:, None])
    else:
        scores = np.arange(0, support[-1] + 1)
        def side_pmf(mu, alpha):
            mu = np.maximum(mu, 1e-6)[:, None]
            if alpha <= 1e-6:
                return poisson.pmf(scores[None, :], mu)
            return nbinom.pmf(scores[None, :], 1. / alpha, 1. / (1. + alpha * mu))
        ah, aa = alphas or (0., 0.)
        hp, ap = side_pmf(h, ah), side_pmf(a, aa)
        if np.any(hp.sum(axis=1) < 1 - 1e-9) or np.any(ap.sum(axis=1) < 1 - 1e-9):
            raise ValueError("count distribution exceeds its integration support")
        if kind == "total":
            pmf = fftconvolve(hp, ap, axes=1)[:, :len(support)]
        else:
            pmf = fftconvolve(hp, ap[:, ::-1], axes=1)
        cdf = np.cumsum(np.maximum(pmf, 0.), axis=1)
    if np.any(cdf[:, -1] < 1 - 1e-8):
        raise ValueError("distribution tail exceeds its integration support")
    return support, np.clip(cdf, 0., 1.)


def _distribution_transform(cdf, design, coefficients):
    """One increasing map per game shared by ALL thresholds (valid CDF)."""
    from scipy.special import expit, logit
    coefficients = np.asarray(coefficients, float)
    shift = design @ coefficients[:3]
    slope = np.exp(design @ coefficients[3:])
    out = expit(logit(np.clip(cdf, 1e-12, 1 - 1e-12)) * slope[:, None]
                + shift[:, None])
    out[cdf <= 0.] = 0.
    out[cdf >= 1.] = 1.
    return out


def _distribution_no_tie(cdf, support):
    """Final-score NBA/NHL/MLB have no draws; resolve 0 into +/-1 equally.

    NFL retains its genuine tie outcome. This settlement adjustment is part
    of the distribution BEFORE fitting/scoring, never a separate ML map.
    """
    if DISTRIBUTION_SPORT == "nfl" or support[0] >= 0:
        return cdf
    out = cdf.copy()
    j = int(np.searchsorted(support, 0))
    mass = np.maximum(out[:, j] - out[:, j - 1], 0.)
    out[:, j - 1] += .5 * mass
    out[:, j] -= .5 * mass
    return out


def lattice_crps(cdf, support, outcome):
    """Exact unit-lattice CRPS, including left/right boundary tail error."""
    y = np.asarray(outcome, float)
    support = np.asarray(support, float)
    cdf = np.asarray(cdf, float)
    if cdf.shape != (len(y), len(support)) or not np.isfinite(y).all():
        raise ValueError("CRPS requires aligned finite outcomes and CDFs")
    if not np.isfinite(cdf).all() or np.any(np.diff(cdf, axis=1) < -1e-9):
        raise ValueError("CRPS requires a finite nondecreasing CDF")
    if np.any(cdf < 0.) or np.any(cdf > 1.):
        raise ValueError("CDF probabilities must be in [0, 1]")
    return (np.square(cdf - (y[:, None] <= support[None, :])).sum(axis=1)
            + np.maximum(support[0] - y, 0.)
            + np.maximum(y - support[-1] - 1., 0.))


def _fit_distribution_map(h, a, outcome, kind, alphas=None):
    """CRPS-optimal conditional location/scale recalibration on past OOF.

    Bias and log-slope depend on expected total and absolute expected margin:
    heteroscedasticity is learned independently for total and spread. A small
    fixed ridge penalty limits extrapolation; no data-driven tuning on tests.
    """
    from scipy.optimize import minimize
    from scipy.special import logit
    y = np.asarray(outcome, float)
    valid = np.isfinite(y) & np.isfinite(h) & np.isfinite(a)
    h, a, y = h[valid], a[valid], y[valid]
    if len(y) < CRPS_MIN_HISTORY:
        return [0.] * 6
    support, raw = _distribution_raw_cdf(h, a, kind, alphas)
    design = _distribution_design(h, a)
    logits = logit(np.clip(raw, 1e-12, 1 - 1e-12))
    target = (y[:, None] <= support[None, :]).astype(float)
    def loss_gradient(beta):
        from scipy.special import expit
        shift, slope = design @ beta[:3], np.exp(design @ beta[3:])
        cdf = expit(logits * slope[:, None] + shift[:, None])
        cdf[raw <= 0.] = 0.; cdf[raw >= 1.] = 1.
        derivative = cdf * (1 - cdf)
        if kind == "margin" and DISTRIBUTION_SPORT != "nfl":
            j = int(np.searchsorted(support, 0))
            # Both CDF thresholds share the average after tie resolution.
            mid = .5 * (cdf[:, j - 1] + cdf[:, j])
            cdf[:, j - 1] = mid; cdf[:, j] = mid
        error = cdf - target
        loss = np.square(error).sum(axis=1).mean()
        if kind == "margin" and DISTRIBUTION_SPORT != "nfl":
            mid_error = .5 * (error[:, j - 1] + error[:, j])
            error[:, j - 1] = mid_error; error[:, j] = mid_error
        g = 2. * error * derivative / len(y)
        gradient = np.r_[design.T @ g.sum(axis=1),
                         design.T @ (g * logits * slope[:, None]).sum(axis=1)]
        penalty = .01 * np.dot(beta, beta)
        return loss + penalty, gradient + .02 * beta
    result = minimize(loss_gradient, np.zeros(6), jac=True, method="L-BFGS-B",
                      bounds=[(-2., 2.)] * 3 + [(-1.5, 1.5)] * 3,
                      options={"maxiter": 100, "ftol": 1e-8})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"CRPS calibration failed: {result.message}")
    return result.x.tolist()


def _distribution_cdfs(frame, maps):
    h, a = _distribution_inputs(frame)
    design = _distribution_design(h, a)
    out = {}
    for kind in ("total", "margin"):
        support, raw = _distribution_raw_cdf(h, a, kind, maps.get("alphas"))
        cdf = _distribution_transform(raw, design, maps.get(kind, [0.] * 6))
        if kind == "margin":
            cdf = _distribution_no_tie(cdf, support)
        out[kind] = (support, cdf)
    return out


def _distribution_price(frame, maps):
    """All prices, pushes, fair HALF lines, and ML from the SAME two CDFs."""
    cdfs = _distribution_cdfs(frame, maps)
    h, a = _distribution_inputs(frame)
    updates = {"mu_margin": h - a, "mu_total": h + a}
    for kind, (support, cdf) in cdfs.items():
        mass = np.diff(np.column_stack([np.zeros(len(frame)), cdf]), axis=1)
        mean = mass @ support
        updates[f"distribution_mean_{kind}"] = mean
        updates[f"distribution_sd_{kind}"] = np.sqrt(np.maximum(mass @ (support ** 2) - mean ** 2, 0.))
        def at(threshold):
            index = int(np.floor(threshold) - support[0])
            if index < 0: return np.zeros(len(frame))
            if index >= len(support): return np.ones(len(frame))
            return cdf[:, index]
        if DISTRIBUTION_SPORT == "mlb":
            lines = (TOTAL_LINE_GRID if kind == "total" else
                     sorted(set([*RUN_LINE_GRID, *RUN_LINE_GRID_FULL,
                                 *[-x for x in RUN_LINE_GRID_FULL], 0.])))
        else:
            original = config.TOTAL_GRID if kind == "total" else config.SPREAD_GRID
            lines = sorted(set([*original, *[float(x) + .5 for x in original],
                                *([] if kind == "total" else config.HALF_STOP_LINES)]))
        half = support[:-1] + .5
        j = np.argmin(np.abs(1. - cdf[:, :-1] - .5), axis=1)
        lines = sorted(set([*lines, *half[j].tolist()]))
        for line in lines:
            upper, lower = at(line), at(np.ceil(line) - 1.)
            over, push, under = 1. - upper, np.maximum(upper - lower, 0.), lower
            if DISTRIBUTION_SPORT == "mlb":
                if kind == "total":
                    suffix = str(float(line)).replace('.', '_')
                    updates[f"p_over_{suffix}"] = over
                    updates[f"p_push_{suffix}"] = push
                    updates[f"p_under_{suffix}"] = under
                elif line >= .5:
                    updates[rl_col(line, "home")] = over
                    updates[rl_col(line, "push")] = push
                    updates[rl_col(line, "away")] = under
                    updates[rl_col(line, "away_favorite")] = at(-line - (1. if float(line).is_integer() else 0.))
                    updates[rl_col(line, "away_push")] = at(-line) - at(np.ceil(-line) - 1.)
                    updates[rl_col(line, "home_dog")] = 1. - at(-line)
                    updates[f"p_home_cover_{str(float(line)).replace('.', '_')}"] = over
            else:
                prefix = "p_over" if kind == "total" else "p_home_cover"
                pp = "p_push_total" if kind == "total" else "p_push"
                updates[_grid_key(prefix, line)] = over
                updates[_grid_key(pp, line)] = push
                updates[_grid_key("p_under" if kind == "total" else "p_away_cover", line)] = under
                if kind == "total" and DISTRIBUTION_SPORT == "nfl":
                    updates[_grid_key("p_push", line)] = push
        if kind == "margin":
            updates["p_home_win_derived"] = 1. - at(0.)
            updates["p_tie"] = at(0.) - at(-1.)
            updates["p_away_win_derived"] = at(-1.)
            updates["derived_ml"] = updates["p_home_win_derived"]
        # Nearest half-line to the actual CDF median, not the old integer
        # grid boundary. This remains defined even outside displayed grids.
        half = support[:-1] + .5
        j = np.argmin(np.abs(1. - cdf[:, :-1] - .5), axis=1)
        updates["fair_total" if kind == "total" else "fair_spread"] = half[j]
        updates["p_over_fair" if kind == "total" else "p_cover_fair"] = 1. - cdf[np.arange(len(frame)), j]
    out = pd.concat([frame.drop(columns=list(updates), errors="ignore"),
                     pd.DataFrame(updates, index=frame.index)], axis=1)
    out.attrs["distribution_method"] = "causal_lattice_crps"
    out.attrs["distribution_objective"] = "discrete_crps"
    if "home_score" in out and "away_score" in out:
        margin = out.home_score - out.away_score
        total = out.home_score + out.away_score
        for col, values in {"y_cover_fair": margin > out.fair_spread,
                            "y_push_spread_fair": margin == out.fair_spread,
                            "y_over_fair": total > out.fair_total,
                            "y_under_fair": total < out.fair_total,
                            "y_push_fair": total == out.fair_total}.items():
            out[col] = np.where(margin.notna() & total.notna(), values.astype(float), np.nan)
    return out


def _distribution_fit_maps(h, a, outcomes):
    alphas = None
    if DISTRIBUTION_SPORT in ("nfl", "mlb", "nhl"):
        hs = .5 * (outcomes["total"] + outcomes["margin"])
        aws = .5 * (outcomes["total"] - outcomes["margin"])
        def residual_alpha(y, mu):
            if len(y) < CRPS_MIN_HISTORY: return 0.
            return float(np.clip(np.sum((y - mu) ** 2 - mu) /
                                 max(np.sum(mu ** 2), 1e-12), 0., .5))
        alphas = [residual_alpha(hs, h), residual_alpha(aws, a)]
    maps = {kind: _fit_distribution_map(h, a, y, kind, alphas)
            for kind, y in outcomes.items()}
    if alphas is not None:
        maps["alphas"] = alphas
    return maps


def causal_distribution_frame(frame):
    """Fit distribution mechanics at each origin on PRIOR OOF outcomes only.

    Final serving maps may use all settled past OOF; NEVER apply those maps
    back to OOF rows. An undated frame is refused rather than claimed PIT.
    """
    if frame is None or not len(frame):
        return frame, {"method": "causal_lattice_crps", "maps": {}}
    h, a = _distribution_inputs(frame)
    outcomes = {"total": frame.home_score.to_numpy(float) + frame.away_score.to_numpy(float),
                "margin": frame.home_score.to_numpy(float) - frame.away_score.to_numpy(float)}
    if not all(np.isfinite(y).all() for y in outcomes.values()):
        raise ValueError("distribution evaluation requires settled finite outcomes")
    if not frame.index.is_unique:
        raise ValueError("distribution evaluation requires unique row indices")
    settled = frame[["home_score", "away_score"]].to_numpy(float)
    if np.any(settled < 0.) or np.any(settled != np.floor(settled)):
        raise ValueError("distribution outcomes must be nonnegative integer final scores")
    blocks = _distribution_blocks(frame)
    history = np.zeros(len(frame), dtype=bool)
    parts, origins = [], []
    evidence = {"total": [], "margin": []}
    for start, end, mask in blocks:
        candidate = _distribution_fit_maps(h[history], a[history],
                                           {k: y[history] for k, y in outcomes.items()})
        maps = dict(candidate)
        admitted = {}
        for kind in ("total", "margin"):
            differences = np.concatenate(evidence[kind]) if evidence[kind] else np.array([])
            # Paired losses are from previously issued, honest candidate and
            # identity CDFs. No same-origin fit performance selects the map.
            admitted[kind] = bool(len(differences) >= CRPS_MIN_HISTORY
                                  and differences.mean() < 0.)
            if not admitted[kind]: maps[kind] = [0.] * 6
        current = frame.iloc[np.flatnonzero(mask)]
        part = _distribution_price(current, maps)
        baseline = dict(candidate, total=[0.] * 6, margin=[0.] * 6)
        cc, bc = _distribution_cdfs(current, candidate), _distribution_cdfs(current, baseline)
        for kind in ("total", "margin"):
            support, cdf = cc[kind]
            difference = (lattice_crps(cdf, support, outcomes[kind][mask])
                          - lattice_crps(bc[kind][1], support, outcomes[kind][mask]))
            evidence[kind].append(difference)
        part["distribution_fit_n"] = int(history.sum())
        part["distribution_fit_end"] = str(origins[-1]["end"]) if origins else None
        parts.append(part)
        origins.append({"start": str(start.date()), "end": str(end.date()),
                        "n_prior": int(history.sum()), "n_scored": int(mask.sum()),
                        "maps": maps, "candidate_maps": candidate,
                        "calibration_admitted": admitted})
        history |= mask
    maps = _distribution_fit_maps(h, a, outcomes)
    for kind in ("total", "margin"):
        difference = np.concatenate(evidence[kind])
        if len(difference) < CRPS_MIN_HISTORY or difference.mean() >= 0.:
            maps[kind] = [0.] * 6
    bundle = {"method": "causal_lattice_crps", "scope": "whole_cdf",
              "sport": DISTRIBUTION_SPORT, "objective": "discrete_crps",
              "maps": maps, "origins": origins, "fit_n": len(frame),
              "serving_fit_scope": "all_settled_prior_oof_only",
              "admission": "negative_mean_paired_crps_on_prior_issued_origins"}
    return pd.concat(parts).reindex(frame.index), bundle


def distribution_score_report(frame, bundle):
    """Rebuild each origin's SAME CDF for full-support CRPS and binary scores."""
    from sklearn.metrics import roc_auc_score
    values = {"total": [], "margin": []}
    row_losses = {"total": np.empty(len(frame)), "margin": np.empty(len(frame))}
    for (_, _, mask), origin in zip(_distribution_blocks(frame), bundle["origins"]):
        part = frame.iloc[np.flatnonzero(mask)]
        cdfs = _distribution_cdfs(part, origin["maps"])
        for kind, (support, cdf) in cdfs.items():
            y = (part.home_score + part.away_score if kind == "total"
                 else part.home_score - part.away_score).to_numpy(float)
            loss = lattice_crps(cdf, support, y)
            values[kind].append(loss)
            row_losses[kind][mask] = loss
    result = {"method": "causal_lattice_crps", "roi_verified": False}
    for kind, key, line_key, name in (("total", "p_over_fair", "fair_total", "totals"),
                                      ("margin", "p_cover_fair", "fair_spread", "run_line")):
        p = frame[key].to_numpy(float)
        y = (frame.home_score + frame.away_score if kind == "total"
             else frame.home_score - frame.away_score).to_numpy(float)
        event = y > frame[line_key].to_numpy(float)
        clipped = np.clip(p, 1e-7, 1 - 1e-7)
        ll = float(-np.mean(event * np.log(clipped) + (~event) * np.log1p(-clipped)))
        bins = np.minimum((p * 10).astype(int), 9)
        reliability = []
        error = 0.
        for b in np.unique(bins):
            m = bins == b
            gap = float(p[m].mean() - event[m].mean())
            error += float(m.mean()) * abs(gap)
            reliability.append({"n": int(m.sum()), "predicted": float(p[m].mean()),
                                "actual": float(event[m].mean()), "gap": gap})
        result[name] = {"n": len(p), "n_pushes": 0,
                        "crps": float(np.concatenate(values[kind]).mean()),
                        "auc": float(roc_auc_score(event, p)) if len(np.unique(event)) == 2 else None,
                        "logloss": ll, "logscore": ll,
                        "brier": float(np.mean((p - event) ** 2)),
                        "predicted_mean": float(p.mean()), "actual_win_rate": float(event.mean()),
                        "ece_cover": error, "reliability": reliability,
                        "calibration_in_the_large": float(p.mean() - event.mean()),
                        "threshold_policy": "closest_half_point_to_model_50_percent"}
    probability = frame.p_home_win_derived.to_numpy(float)
    tie = frame.p_tie.to_numpy(float)
    margin = (frame.home_score - frame.away_score).to_numpy(float)
    eligible = margin != 0
    conditional = np.clip(probability[eligible] / np.maximum(1. - tie[eligible], 1e-12), 1e-7, 1 - 1e-7)
    target = margin[eligible] > 0
    result["derived_moneyline"] = {
        "n": int(eligible.sum()), "n_ties": int((~eligible).sum()),
        "auc": float(roc_auc_score(target, conditional)) if len(np.unique(target)) == 2 else None,
        "logloss": float(-np.mean(target * np.log(conditional) + (~target) * np.log1p(-conditional))),
        "brier": float(np.mean((conditional - target) ** 2)),
        "settlement": "conditional_non_tie" if DISTRIBUTION_SPORT == "nfl" else "final_score_no_tie"}
    result["fixed_thresholds"] = {}
    for kind, prefix, lines in (
            ("total", "p_over", {"nba": [210., 220., 230., 240.],
                                  "nfl": [38., 44., 50.], "nhl": [5., 6., 7.],
                                  "mlb": [7.5, 8.5, 9.5]}[DISTRIBUTION_SPORT]),
            ("margin", "p_home_cover", [-1.5, .5, 1.5] if DISTRIBUTION_SPORT in ("mlb", "nhl")
                                       else [-5.5, .5, 5.5])):
        outcome = (frame.home_score + frame.away_score if kind == "total"
                   else frame.home_score - frame.away_score).to_numpy(float)
        for line in lines:
            key = (f"{prefix}_{str(float(line)).replace('.', '_')}" if DISTRIBUTION_SPORT == "mlb"
                   else _grid_key(prefix, line))
            if key not in frame: continue
            p = np.clip(frame[key].to_numpy(float), 1e-7, 1 - 1e-7)
            event = outcome > line
            bins = np.minimum((p * 10).astype(int), 9)
            reliability = []
            for b in np.unique(bins):
                m = bins == b
                reliability.append({"n": int(m.sum()), "predicted": float(p[m].mean()),
                                    "actual": float(event[m].mean())})
            result["fixed_thresholds"][f"{kind}_{line}"] = {
                "n": len(p), "auc": float(roc_auc_score(event, p)) if len(np.unique(event)) == 2 else None,
                "logloss": float(-np.mean(event * np.log(p) + (~event) * np.log1p(-p))),
                "brier": float(np.mean((p - event) ** 2)), "reliability": reliability,
                "event": "strict_exceedance_including_push_as_nonwin"}
    result["conditional_crps"] = {}
    # All strata remain visible including small postseason groups; no exclusion
    # based on how a regime happens to perform.
    if "season" in frame:
        for season in pd.unique(frame.season.dropna()):
            mask = frame.season.eq(season).to_numpy()
            result["conditional_crps"][f"season_{season}"] = {
                "n": int(mask.sum()), **{k: float(v[mask].mean()) for k, v in row_losses.items()}}
    for column in ("is_playoffs", "provisional"):
        if column in frame:
            for flag in (False, True):
                mask = frame[column].fillna(False).astype(bool).eq(flag).to_numpy()
                if mask.any():
                    result["conditional_crps"][f"{column}_{flag}"] = {
                        "n": int(mask.sum()), **{k: float(v[mask].mean()) for k, v in row_losses.items()}}
    return result


def distribution_probability(frame, bundle, kind, line):
    """Exact win/push/loss at ANY integer/half line from serving CRPS CDF.

    line is a margin threshold, not a signed sportsbook home handicap:
    home handicap -3.5 maps to margin threshold +3.5.
    """
    if kind not in ("total", "margin") or not np.isfinite(line):
        raise ValueError("expected total/margin and finite line")
    support, cdf = _distribution_cdfs(frame, bundle.get("maps", {}))[kind]
    def at(value):
        index = int(np.floor(value) - support[0])
        if index < 0: return np.zeros(len(frame))
        if index >= len(support): return np.ones(len(frame))
        return cdf[:, index]
    upper, lower = at(line), at(np.ceil(line) - 1.)
    return {"win": 1. - upper, "push": np.maximum(upper - lower, 0.), "loss": lower}
