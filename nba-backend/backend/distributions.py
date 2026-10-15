"""NBA split-skew Gaussian variance engine with causal CRPS calibration.

Market-free, point-in-time score means feed home/away residual variances
and correlation. Totals and margins have independently learned asymmetric
left/right Gaussian widths. One monotone calibrated CDF per market is
sampled by reproducible Monte Carlo; its empirical mass supplies all lines,
pushes, fair half-lines and derived moneyline. Forecast origins only learn
from earlier outcomes, with the recent evaluation tail sealed. Serving can
refit on all settled past forecasts. No sportsbook is a model input.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    from backend import config
    from backend import folds as folds_mod
    from backend import moneyline as ml_mod
except ImportError:
    import config
    import folds as folds_mod
    import moneyline as ml_mod

logger = logging.getLogger(__name__)

# The moneyline member whose matrix the run-line regressors consume. The
# run-line models are LightGBM Poisson — tree family — so the tree member's
# view is the correct one (mirrors MLB's RUN_TREE_CATEGORICAL_COLS routing:
# the categorical team-ID pair rides with the tree view, never the linear one).
MONEYLINE_TREE_MEMBER = "lightgbm"

# Sealed-holdout convention (MLB derive_markets_v3 HOLDOUT_DAYS parity,
# NHL 62d00fd port): the LAST ``HOLDOUT_DAYS`` of OOF games are SEALED — the
# CRPS admission evidence never sees them, so the tail scores the market
# engine honestly instead of validating itself. Nothing is fitted on the
# sealed rows; the gate exists to keep the evaluation population honest.
HOLDOUT_DAYS = 21
try:
    from backend import gaussian_family as family
except ImportError:
    import gaussian_family as family

MC_DRAWS = family.MC_DRAWS
MC_SEED = family.MC_SEED
MC_SE_TARGET = family.MC_SE_TARGET


def _grid_key(prefix: str, value: float | int) -> str:
    text = str(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    return f"{prefix}_{text.replace('-', 'm').replace('.', '_')}"


class GaussianCRPSObjective:
    """Analytic proper-score location objective at a train-only scale.

    Initial score is supplied explicitly to LightGBM and restored at predict.
    Scale adaptation then uses past OOF via the lattice-CRPS CDF layer.
    """

    def __init__(self, scale):
        self.scale = max(float(scale), 1.)

    def __call__(self, y, prediction):
        from scipy.special import ndtr
        z = (np.asarray(prediction) - y) / self.scale
        gradient = 2. * ndtr(z) - 1.
        hessian = 2. * np.exp(-.5 * z ** 2) / (np.sqrt(2 * np.pi) * self.scale)
        return gradient, np.maximum(hessian, 1e-5)


def _make_reg(scale=12., crps=False):
    from lightgbm import LGBMRegressor
    params = dict(config.LIGHTGBM_REG_PARAMS)
    params["objective"] = GaussianCRPSObjective(scale) if crps else "poisson"
    return LGBMRegressor(**params)


def gaussian_crps(y, location, scale):
    """Analytic continuous Gaussian CRPS, useful for objective verification."""
    from scipy.special import ndtr
    scale = np.asarray(scale, float)
    if np.any(scale <= 0):
        raise ValueError("CRPS scale must be positive")
    z = (np.asarray(y, float) - np.asarray(location, float)) / scale
    return scale * (z * (2 * ndtr(z) - 1.)
                    + 2 * np.exp(-.5 * z ** 2) / np.sqrt(2 * np.pi)
                    - 1. / np.sqrt(np.pi))


class ScoreRegressor:
    """PIT score-mean models; direct CRPS location trees are an opt-in challenger."""

    def __init__(self, crps_location=False) -> None:
        # Direct CRPS location trees are a research challenger, NOT the
        # production default: the full 2026-10-11 walk underperformed.
        self.crps_location = bool(crps_location)
        self.home_model: Any | None = None
        self.away_model: Any | None = None
        self.feature_columns: list[str] = []
        self.fallback: tuple[Any, Any] | None = None

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        # Resolved through the binary moneyline's OWN member_matrix helper —
        # the NBA equivalent of MLB's
        # ``from training import active_moneyline_feature_cols``. That helper
        # is the single point where the production moneyline decides its
        # feature contract (adopted RFE subset or the full universe, plus the
        # tree family's categorical team-ID pair), so calling it here makes
        # "the run line uses exactly the moneyline production feature set" a
        # structural property instead of a convention two paths happen to
        # agree on. Do not fill NaN: LightGBM handles missing natively.
        X = ml_mod.member_matrix(MONEYLINE_TREE_MEMBER, df)
        if not self.feature_columns:
            self.feature_columns = list(X.columns)
        return X.reindex(columns=self.feature_columns)

    @staticmethod
    def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce")

    def fit(self, df: pd.DataFrame) -> "ScoreRegressor":
        X = self._matrix(df)
        h = self._numeric(df, "home_score")
        a = self._numeric(df, "away_score")
        valid = h.notna() & a.notna()
        if valid.sum() < 2:
            raise ValueError("score regressor requires at least two settled games")
        X, h, a = X.loc[valid], h.loc[valid], a.loc[valid]
        # No silent ridge fallback: a failed fit must be visible upstream.
        categorical = [c for c in config.TREE_CATEGORICAL_COLS if c in X]
        if self.crps_location:
            margin, total = h - a, h + a
            self.margin_center = float(margin.mean())
            self.total_center = float(total.mean())
            self.home_model = _make_reg(max(float(margin.std(ddof=0)), 1.), crps=True)
            self.away_model = _make_reg(max(float(total.std(ddof=0)), 1.), crps=True)
            self.home_model.fit(X, margin, init_score=np.full(len(X), self.margin_center),
                                categorical_feature=categorical)
            self.away_model.fit(X, total, init_score=np.full(len(X), self.total_center),
                                categorical_feature=categorical)
        else:
            self.home_model, self.away_model = _make_reg(), _make_reg()
            self.home_model.fit(X, h, categorical_feature=categorical)
            self.away_model.fit(X, a, categorical_feature=categorical)
        self.fallback = None
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.fallback is not None:
            X = self._matrix(df).astype(float)
            return (np.clip(self.fallback[0].predict(X), 1e-6, None),
                    np.clip(self.fallback[1].predict(X), 1e-6, None))
        if self.home_model is None or self.away_model is None:
            raise RuntimeError("score regressor is not fitted")
        X = self._matrix(df)
        if self.crps_location:
            margin = self.margin_center + self.home_model.predict(X)
            total = self.total_center + self.away_model.predict(X)
            return (np.clip((total + margin) / 2., 1e-6, None),
                    np.clip((total - margin) / 2., 1e-6, None))
        return (np.clip(self.home_model.predict(X), 1e-6, None),
                np.clip(self.away_model.predict(X), 1e-6, None))


def feat_mod_tree(df: pd.DataFrame) -> pd.DataFrame:
    """The moneyline tree view, without the member wrapper (fallback path)."""
    return ml_mod.member_matrix(MONEYLINE_TREE_MEMBER, df)


def run_engine_feature_cols(reg: "ScoreRegressor | None" = None) -> list[str]:
    """Numeric feature view of the DISTRIBUTION model (Totals & Run Lines).

    The monitoring view is resolved from the FITTED regressor's own
    ``feature_columns`` — literally the matrix ``ScoreRegressor`` consumed —
    falling back to the ``tree_view`` contract it is built from when no
    model is fitted yet. The team-ID categoricals are excluded: PSI and
    coverage are numeric-only views (MLB's run-engine view excludes them
    too — they carry no weight row of their own). This is what the run-
    engine drift/coverage artifacts report so the dashboard measures the
    distribution model instead of the binary moneyline contract.
    """
    cols = list(getattr(reg, "feature_columns", None) or [])
    if not cols:
        cols = list(config.active_moneyline_feature_cols())
    return [c for c in cols if c not in config.TREE_CATEGORICAL_COLS]


def distribution_feature_weights(
        reg: "ScoreRegressor | None") -> dict[str, float] | None:
    """Per-feature MODEL WEIGHT (%) of the distribution model.

    Pooled feature importances of the two per-side Poisson LightGBM fits
    (home/away), normalized to sum to 100 — the NBA analog of MLB's run
    line weights (pooled split-gain across its two side Poisson models).
    The team-ID categoricals are excluded from BOTH the map and the
    denominator (they carry no drift/coverage row, and the panel's caption
    promises the reported weights sum to 100 across the view it shows).
    None when the regressor is unfitted or on the ridge fallback (no
    LightGBM importances there): the drift column then reports no weight
    rather than borrowing the binary moneyline blend's numbers.
    """
    home = getattr(reg, "home_model", None)
    away = getattr(reg, "away_model", None)
    if home is None or away is None:
        return None
    try:
        h = [float(v) for v in home.feature_importances_]
        a = [float(v) for v in away.feature_importances_]
    except Exception:  # noqa: BLE001 - any non-importance model degrades honestly
        return None
    names = list(getattr(reg, "feature_columns", None) or [])
    if not names or len(names) != len(h) or len(names) != len(a):
        return None
    keep = [(n, hv, av) for n, hv, av in zip(names, h, a)
            if n not in config.TREE_CATEGORICAL_COLS]
    total = sum(hv + av for _, hv, av in keep)
    if not keep or total <= 0:
        return None
    return {name: round(100.0 * (hv + av) / total, 6)
            for name, hv, av in keep}



def pearson_poisson_adequacy(y: np.ndarray, mu: np.ndarray) -> float:
    """Pearson chi-square / df for the score means. ≈1 → Poisson variance
    would describe the residuals; >1 means the per-game variance exceeds
    the mean, which is what the two-sided CDF family's inflation factor
    ``c`` absorbs. Kept as a variance diagnostic on μ, not a price input."""
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(mu, float), 1e-9, None)
    ok = np.isfinite(y) & np.isfinite(mu)
    y, mu = y[ok], mu[ok]
    if len(y) < 2:
        return float("nan")
    return float(((y - mu) ** 2 / mu).sum() / len(y))


def _poisson_deviance_mean(y: np.ndarray, mu: np.ndarray) -> float:
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(mu, float), 1e-9, None)
    ok = np.isfinite(y) & np.isfinite(mu)
    y, mu = y[ok], mu[ok]
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return float(2.0 * np.mean(term - (y - mu)))


def run_line_fit_check(oof: pd.DataFrame) -> dict[str, Any]:
    """Pooled fit diagnostics for the run line (MLB run_oof metrics shape):
    the Poisson-adequacy probe per side plus deviance/RMSE of the μ
    predictions against the constant league-mean baseline — the model must
    beat the baseline it replaces, per fold population and pooled."""
    out: dict[str, Any] = {}
    for side, y_col, mu_col in (("home", "home_score", "mu_h"),
                                ("away", "away_score", "mu_a")):
        y = oof[y_col].to_numpy(float)
        mu = oof[mu_col].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(mu)
        y, mu = y[ok], mu[ok]
        base = float(y.mean()) if len(y) else float("nan")
        out[side] = {
            "pearson": round(pearson_poisson_adequacy(y, mu), 4),
            "deviance_model": round(_poisson_deviance_mean(y, mu), 5),
            "deviance_baseline": round(_poisson_deviance_mean(y, np.full(len(y), base)), 5),
            "rmse_model": round(float(np.sqrt(np.mean((y - mu) ** 2))), 4),
            "rmse_baseline": round(float(np.sqrt(np.mean((y - base) ** 2))), 4),
            "n": int(len(y)),
        }
    return out


def _fair(row: dict[str, Any], prefix: str, lines: list[int]) -> float:
    values = np.asarray([float(row.get(_grid_key(prefix, line), np.nan))
                         for line in lines], dtype=float)
    if not np.isfinite(values).any():
        return float(lines[0])
    return float(lines[int(np.nanargmin(np.abs(values - 0.5)))])


def apply_distribution(df: pd.DataFrame, bundle: dict | None = None,
                       meta_out: dict | None = None, n_draws=MC_DRAWS,
                       seed=MC_SEED) -> pd.DataFrame:
    """Sample the challenger CDF without consulting any outcomes."""
    return _distribution_price(df, (bundle or {}).get("maps", {}),
                               n_draws=n_draws, seed=seed, meta_out=meta_out)


def walk_forward_oof(game_df: pd.DataFrame, date_col: str = "gameday",
                     fold_list=None) -> dict:
    df = folds_mod.canonical_sort(game_df, date_col)
    folds = fold_list if fold_list is not None else folds_mod.make_folds(df, date_col)
    parts: list[pd.DataFrame] = []
    rows: list[dict] = []
    for fold in folds:
        train, val = df.loc[fold.train_idx], df.loc[fold.val_idx]
        train_dates = pd.to_datetime(train[date_col], errors="raise").dt.normalize()
        val_dates = pd.to_datetime(val[date_col], errors="raise").dt.normalize()
        if train_dates.max() >= val_dates.min():
            raise ValueError("distribution training must precede validation dates")
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


def calibrate_market_frame(df, n_draws=MC_DRAWS, seed=MC_SEED):
    """Dated OOF only: fit split-skew calibration on strictly prior origins."""
    priced, bundle = causal_distribution_frame(df, n_draws=n_draws, seed=seed)
    if priced is not None and len(priced):
        metrics = distribution_score_report(priced, bundle)
        bundle["oof_metrics"] = metrics
        priced.attrs["distribution_metrics"] = metrics
    return priced, bundle


def apply_market_calibration(df, bundle):
    if not bundle or bundle.get("method") != family.METHOD:
        raise ValueError("serving requires a fitted split-skew challenger bundle")
    mc = bundle.get("monte_carlo", {})
    out = _distribution_price(df, bundle.get("maps", {}),
                              n_draws=mc.get("requested_draws", MC_DRAWS),
                              seed=mc.get("seed", MC_SEED))
    out["distribution_fit_n"] = bundle.get("fit_n", 0)
    out["distribution_fit_end"] = bundle.get("fit_end")
    return out


DISTRIBUTION_SPORT = "nba"
CRPS_MIN_HISTORY = family.MIN_HISTORY


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
    return family.support_for(kind, DISTRIBUTION_SPORT)


def _distribution_design(h, a):
    return family.design(h, a, DISTRIBUTION_SPORT)


def _distribution_raw_cdf(h, a, kind, parameters=None):
    return family.raw_cdf(h, a, kind, DISTRIBUTION_SPORT, parameters)


def _distribution_transform(cdf, design, coefficients):
    return family.transform(cdf, design, coefficients)


def _distribution_no_tie(cdf, support, home_share=None):
    # Structural no-tie rule of the margin lattice. The NHL supplies a
    # causal OT/SO home-win share (maps["lattice"]["margin"]
    # ["ot_home_share"], estimated from prior OOF one-goal finals) so the
    # P(M=0) mass is redistributed to +/-1 by OT win probabilities from OOF;
    # None keeps the exact equal split every pre-lattice bundle used.
    return family.no_tie(cdf, support, DISTRIBUTION_SPORT, home_share)


def lattice_crps(cdf, support, outcome):
    return family.crps(cdf, support, outcome)


def _fit_distribution_map(h, a, outcome, kind, parameters=None):
    return family.fit_map(h, a, outcome, kind, DISTRIBUTION_SPORT,
                          parameters or family.defaults(DISTRIBUTION_SPORT))


def _distribution_cdfs(frame, maps, sampled=False, n_draws=MC_DRAWS,
                       seed=MC_SEED, meta_out=None):
    h, a = _distribution_inputs(frame)
    design = _distribution_design(h, a)
    identity_col = "game_pk" if DISTRIBUTION_SPORT == "mlb" else "game_id"
    if sampled and (identity_col not in frame or frame[identity_col].isna().any()):
        raise ValueError("Monte Carlo pricing requires finite game identities")
    out = {}
    # Integer-lattice corrections ride the maps: totals and margin are fit
    # INDEPENDENTLY (see gaussian_family.LATTICE_CONFIG) and applied AFTER
    # the monotone calibration map and the structural rules, BEFORE Monte
    # Carlo sampling. Stored bundles that predate lattice fitting carry no
    # "lattice" key and price through the exact pre-lattice path (NHL totals
    # keep their structural OT/SO parity from the variance parameters).
    lattice = maps.get("lattice") or {}
    for kind in ("total", "margin"):
        target = "totals" if kind == "total" else "margin"
        support, raw = _distribution_raw_cdf(h, a, kind, maps.get("variance"))
        cdf = _distribution_transform(raw, design, maps.get(kind, [0.] * 9))
        spec = dict(lattice.get(target) or {})
        if kind == "total" and DISTRIBUTION_SPORT == "nhl" and "alpha" not in spec:
            # NHL odd/even parity rides the VARIANCE parameters (structural,
            # fitted from strictly-prior finals), never the CRPS admission
            # gate for the calibration maps: alpha <= 0 (stored bundles
            # predating the key) is a byte-identical no-op.
            variance = maps.get("variance") or family.defaults(DISTRIBUTION_SPORT)
            spec["alpha"] = float(variance.get("total_parity_alpha", 0.))
        if kind == "margin":
            cdf = _distribution_no_tie(cdf, support, spec.get("ot_home_share"))
        cdf = family.lattice_correct(cdf, support, spec,
                                     DISTRIBUTION_SPORT, target)
        if sampled:
            cdf, meta = family.sample_cdf(cdf, support, frame[identity_col].astype(str),
                                          kind, DISTRIBUTION_SPORT, n_draws, seed)
            if meta_out is not None:
                meta_out.update(meta)
        out[kind] = (support, cdf)
    return out


def _distribution_price(frame, maps, n_draws=MC_DRAWS, seed=MC_SEED, meta_out=None):
    """All prices and settlement legs from the same sampled challenger CDFs."""
    mc_meta = {}
    cdfs = _distribution_cdfs(frame, maps, sampled=True, n_draws=n_draws,
                               seed=seed, meta_out=mc_meta)
    if meta_out is not None:
        meta_out.update(mc_meta)
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
    out.attrs["distribution_method"] = family.METHOD
    out.attrs["mc_meta"] = mc_meta
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


def _fit_lattice_maps(outcomes, parameters, evidence):
    """Independent per-target lattice fits — totals AND margin.

    Structural terms ride the VARIANCE parameters (NHL totals OT/SO parity
    alpha; the no-tie +/-1 split weight — NHL OT/SO home-win share, MLB
    walk-off home-win share — estimated causally from prior OOF one-goal
    own BASE-model evidence — pmfs of strictly-prior OOF rows, each priced
    with its own origin's maps — per target, behind the held-out
    significance gate. Every value is plain JSON so stored bundles reprice
    byte-identically.
    """
    totals: dict = {}
    margin: dict = {}
    if DISTRIBUTION_SPORT == "nhl":
        alpha = float(parameters.get("total_parity_alpha", 0.))
        if alpha > 0.:
            totals["alpha"] = alpha
    if DISTRIBUTION_SPORT in ("nhl", "mlb"):
        # The no-tie +/-1 split weight (NHL OT/SO, MLB walk-off extras) is
        # admitted only when the margin evidence's held-out tail confirms
        # the shift (see family.fit_ot_home_share); cold starts and
        # unconfirmed shifts keep the exact equal split.
        margin_ev = (evidence or {}).get("margin") or {}
        share = family.fit_ot_home_share(margin_ev.get("outcomes"),
                                         pmfs=margin_ev.get("pmfs"),
                                         support=margin_ev.get("support"))
        if share is not None:
            margin["ot_home_share"] = share
    for target, kind in (("totals", "total"), ("margin", "margin")):
        ev = (evidence or {}).get(kind) or None
        if ev is None:
            continue
        (totals if target == "totals" else margin).update(
            family.fit_lattice_correction(ev.get("pmfs"), ev.get("outcomes"),
                                          ev.get("support"), DISTRIBUTION_SPORT,
                                          target))
    return {"totals": totals, "margin": margin}


def _distribution_fit_maps(h, a, outcomes, lattice_evidence=None):
    parameters = family.fit_variance(h, a, outcomes, DISTRIBUTION_SPORT)
    return {"variance": parameters,
            **{kind: _fit_distribution_map(h, a, y, kind, parameters)
               for kind, y in outcomes.items()},
            "lattice": _fit_lattice_maps(outcomes, parameters, lattice_evidence)}


def causal_distribution_frame(frame, n_draws=MC_DRAWS, seed=MC_SEED):
    """Fit distribution mechanics at each origin on PRIOR OOF outcomes only.

    Final serving maps may use all settled past OOF; NEVER apply those maps
    back to OOF rows. An undated frame is refused rather than claimed PIT.
    """
    if frame is None or not len(frame):
        return frame, {"method": family.METHOD, "maps": {}}
    original_index = frame.index
    date_key = "game_date" if DISTRIBUTION_SPORT == "mlb" else "gameday"
    identity_key = "game_pk" if DISTRIBUTION_SPORT == "mlb" else "game_id"
    frame = frame.sort_values([date_key, identity_key], kind="stable")
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
    date_col = "game_date" if DISTRIBUTION_SPORT == "mlb" else "gameday"
    dates = pd.to_datetime(frame[date_col], utc=True).dt.normalize()
    cutoff = dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    pre_holdout = (dates < cutoff).to_numpy()
    history = np.zeros(len(frame), dtype=bool)
    parts, origins = [], []
    evidence = {"total": [], "margin": []}
    # BASE-model pmfs (post-calibration + structural rules, PRE-lattice) of
    # already-priced prior blocks, per target: the causal evidence the next
    # origin's lattice residual is fit from. Never the row being priced.
    lattice_prior = {"total": [], "margin": []}

    def _lattice_evidence(include_sealed=False):
        # The sealed-holdout tail never trains the maps, so it never trains
        # the lattice fits at in-window origins either; the FINAL serving
        # fit (include_sealed=True) learns from all now-settled rows — the
        # same scope the serving variance/calibration maps use.
        def _select(kind):
            blocks = []
            for pmf, y, sup, pre in lattice_prior[kind]:
                keep = pre if not include_sealed else np.ones(len(pre), dtype=bool)
                if keep.any():
                    blocks.append((pmf[keep], y[keep], sup))
            return family.lattice_evidence(blocks)
        return {kind: _select(kind) for kind in ("total", "margin")}

    for start, end, mask in blocks:
        candidate = _distribution_fit_maps(h[history], a[history],
                                           {k: y[history] for k, y in outcomes.items()},
                                           _lattice_evidence())
        maps = dict(candidate)
        admitted = {}
        for kind in ("total", "margin"):
            differences = np.concatenate(evidence[kind]) if evidence[kind] else np.array([])
            # Paired losses are from previously issued, honest candidate and
            # identity CDFs. No same-origin fit performance selects the map.
            admitted[kind] = bool(len(differences) >= CRPS_MIN_HISTORY
                                  and differences.mean() < 0.)
            if not admitted[kind]: maps[kind] = [0.] * len(maps[kind])
        current = frame.iloc[np.flatnonzero(mask)]
        part = _distribution_price(current, maps, n_draws=n_draws, seed=seed)
        baseline = dict(candidate,
                        total=[0.] * len(candidate["total"]),
                        margin=[0.] * len(candidate["margin"]))
        # The identity baseline prices the BASE family — structural rules
        # from the variance parameters, no lattice delta — so the admission
        # gate asks whether the whole candidate (calibration maps + lattice
        # residual) beats it on prior issued origins.
        baseline.pop("lattice", None)
        cc, bc = _distribution_cdfs(current, candidate), _distribution_cdfs(current, baseline)
        for kind in ("total", "margin"):
            support, cdf = cc[kind]
            difference = (lattice_crps(cdf, support, outcomes[kind][mask])
                          - lattice_crps(bc[kind][1], support, outcomes[kind][mask]))
            evidence[kind].append(difference[pre_holdout[mask]])
        # Accumulate this block's BASE-model pmfs for the NEXT origin's
        # lattice fit (the candidate minus its fitted delta).
        base_maps = dict(candidate)
        base_maps.pop("lattice", None)
        sb = _distribution_cdfs(current, base_maps)
        for kind in ("total", "margin"):
            support_b, cdf_b = sb[kind]
            lattice_prior[kind].append(
                (np.diff(np.column_stack([np.zeros(len(current)), cdf_b]), axis=1),
                 outcomes[kind][mask].copy(), support_b,
                 pre_holdout[mask].copy()))
        part["distribution_fit_n"] = int(history.sum())
        part["distribution_fit_end"] = str(origins[-1]["end"]) if origins else None
        parts.append(part)
        origins.append({"start": str(start.date()), "end": str(end.date()),
                        "n_prior": int(history.sum()), "n_scored": int(mask.sum()),
                        "maps": maps, "candidate_maps": candidate,
                        "calibration_admitted": admitted})
        history |= mask & pre_holdout
    # Freeze evaluation learning through the sealed tail; serving can learn
    # from all now-settled past games, but cannot rewrite issued OOF prices.
    maps = _distribution_fit_maps(h, a, outcomes, _lattice_evidence(True))
    for kind in ("total", "margin"):
        difference = np.concatenate(evidence[kind])
        if len(difference) < CRPS_MIN_HISTORY or difference.mean() >= 0.:
            maps[kind] = [0.] * len(maps[kind])
    bundle = {"method": family.METHOD, "family": family.FAMILY, "scope": "whole_cdf",
              "sport": DISTRIBUTION_SPORT, "objective": "discrete_crps",
              "maps": maps, "origins": origins, "fit_n": len(frame),
              "lattice_summary": {target: family.lattice_summary(
                  (maps.get("lattice") or {}).get(target) or {})
                  for target in family.LATTICE_TARGETS},
              "fit_end": str(dates.max().date()),
              "monte_carlo": parts[-1].attrs["mc_meta"],
              "holdout": {"cutoff": str(cutoff.date()), "n_pre": int(pre_holdout.sum()),
                          "n_holdout": int((~pre_holdout).sum()),
                          "holdout_days": HOLDOUT_DAYS,
                          "fitted_on": "strictly prior pre-holdout origins"},
              "serving_fit_scope": "all_settled_prior_oof_only",
              "admission": "negative_mean_paired_crps_on_prior_issued_origins"}
    priced = pd.concat(parts).reindex(original_index)
    priced.attrs["mc_meta"] = bundle["monte_carlo"]
    return priced, bundle


def distribution_score_report(frame, bundle):
    """Rebuild each origin's SAME CDF for full-support CRPS and binary scores."""
    from sklearn.metrics import roc_auc_score
    values = {"total": [], "margin": []}
    row_losses = {"total": np.empty(len(frame)), "margin": np.empty(len(frame))}
    for (_, _, mask), origin in zip(_distribution_blocks(frame), bundle["origins"]):
        part = frame.iloc[np.flatnonzero(mask)]
        mc = bundle.get("monte_carlo", {})
        cdfs = _distribution_cdfs(part, origin["maps"], sampled=True,
                                  n_draws=mc.get("requested_draws", MC_DRAWS),
                                  seed=mc.get("seed", MC_SEED))
        for kind, (support, cdf) in cdfs.items():
            y = (part.home_score + part.away_score if kind == "total"
                 else part.home_score - part.away_score).to_numpy(float)
            loss = lattice_crps(cdf, support, y)
            values[kind].append(loss)
            row_losses[kind][mask] = loss
    result = {"method": family.METHOD, "roi_verified": False}
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
    mc = bundle.get("monte_carlo", {})
    support, cdf = _distribution_cdfs(frame, bundle.get("maps", {}), sampled=True,
                                     n_draws=mc.get("requested_draws", MC_DRAWS),
                                     seed=mc.get("seed", MC_SEED))[kind]
    def at(value):
        index = int(np.floor(value) - support[0])
        if index < 0: return np.zeros(len(frame))
        if index >= len(support): return np.ones(len(frame))
        return cdf[:, index]
    upper, lower = at(line), at(np.ceil(line) - 1.)
    return {"win": 1. - upper, "push": np.maximum(upper - lower, 0.), "loss": lower}
