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


# ---------------------------------------------------------------------------
# Negative-binomial challenger (2026-10-17): NB2 with log link vs the
# Poisson incumbent. Poisson forces variance == mean; NB adds one
# train-fold-estimated dispersion r per side (variance = mu + mu^2/r,
# r -> inf recovers Poisson). Same tree matrix, same LightGBM hyper-params,
# same folds — the ONLY difference is the likelihood. Trial-only until a
# walk-forward verdict clears the 2.5 gate.
#
# LightGBM contract (verified 2026-10-17): the callable receives RAW
# margins with init_score INCLUDED during training, but predict() returns
# raw WITHOUT the init — so the objective uses yp as-is while predict()
# adds the stored log-mean back (the GaussianCRPSObjective precedent).
# ---------------------------------------------------------------------------
NB_DISPERSION_MIN = 1.0
NB_DISPERSION_MAX = 1e4
NB_RAW_CLIP = (-5.0, 12.0)


def estimate_nb_dispersion(y: np.ndarray) -> float:
    """Method-of-moments NB dispersion r from train labels (per side)."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y) & (y >= 0)]
    if len(y) < 2:
        return NB_DISPERSION_MAX
    m, v = float(y.mean()), float(y.var(ddof=1))
    if not np.isfinite(m) or m <= 0 or not np.isfinite(v) or v <= m:
        return NB_DISPERSION_MAX  # Poisson limit: no over-dispersion
    return float(np.clip(m * m / (v - m),
                         NB_DISPERSION_MIN, NB_DISPERSION_MAX))


class NegativeBinomialObjective:
    """NB2 NLL grad/hess w.r.t. the raw log-mean score."""

    def __init__(self, dispersion: float) -> None:
        self.r = float(np.clip(dispersion, NB_DISPERSION_MIN,
                               NB_DISPERSION_MAX))

    def __call__(self, y_true, y_pred):
        y = np.asarray(y_true, dtype=float)
        mu = np.exp(np.clip(np.asarray(y_pred, dtype=float), *NB_RAW_CLIP))
        r = self.r
        grad = -y + (y + r) * mu / (mu + r)
        hess = (y + r) * mu * r / (mu + r) ** 2
        return grad, np.maximum(hess, 1e-5)


def _make_reg_nb(dispersion: float):
    from lightgbm import LGBMRegressor
    params = dict(config.LIGHTGBM_REG_PARAMS)
    params["objective"] = NegativeBinomialObjective(dispersion)
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

    def __init__(self, crps_location=False, nb=False) -> None:
        # Direct CRPS location trees are a research challenger, NOT the
        # production default: the full 2026-10-11 walk underperformed.
        self.crps_location = bool(crps_location)
        # Negative-binomial challenger (trial-only): NB2 log-link LightGBM
        # per side with train-fold dispersion; mutually exclusive with CRPS.
        self.nb = bool(nb)
        if self.crps_location and self.nb:
            raise ValueError("ScoreRegressor takes one challenger at a time")
        self.home_model: Any | None = None
        self.away_model: Any | None = None
        self.feature_columns: list[str] = []
        self.fallback: tuple[Any, Any] | None = None
        self.nb_log_mean: tuple[float, float] | None = None
        self.nb_dispersion: tuple[float, float] | None = None

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
        elif self.nb:
            rs, lms, models = [], [], []
            for y in (h.to_numpy(dtype=float), a.to_numpy(dtype=float)):
                r = estimate_nb_dispersion(y)
                lm = float(np.log(max(np.mean(y), 1e-6)))
                model = _make_reg_nb(r)
                model.fit(X, y, init_score=np.full(len(X), lm),
                          categorical_feature=categorical)
                rs.append(r)
                lms.append(lm)
                models.append(model)
            self.home_model, self.away_model = models
            self.nb_dispersion = (rs[0], rs[1])
            self.nb_log_mean = (lms[0], lms[1])
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
        if self.nb:
            assert self.nb_log_mean is not None
            lo, hi = NB_RAW_CLIP
            mh = np.exp(np.clip(
                np.asarray(self.home_model.predict(X), dtype=float)
                + self.nb_log_mean[0], lo, hi))
            ma = np.exp(np.clip(
                np.asarray(self.away_model.predict(X), dtype=float)
                + self.nb_log_mean[1], lo, hi))
            return (np.clip(mh, 1e-6, None), np.clip(ma, 1e-6, None))
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
        cols = _ensemble_union_columns(reg)
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
    if isinstance(getattr(reg, "members", None), dict) and reg.members:
        # Promoted 2026-10-17: the distribution model is the score ensemble.
        # Blend-weighted pooled importances across members (the moneyline
        # convention): each member's normalized vector (split-gain for the
        # trees, |coef| on the standardized linear view for the elastic net)
        # pooled over sides, scaled by the regressor's own earned weights.
        return _ensemble_feature_weights(reg)
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



def _ensemble_union_columns(reg) -> list[str]:
    """Union of fitted ensemble member input columns, member order."""
    seen: list[str] = []
    for name in config.SCORE_ENSEMBLE_MEMBERS:
        for col in ((getattr(reg, "columns", None) or {}).get(name) or []):
            if col not in seen:
                seen.append(col)
    return seen


def _member_importance_vector(model, columns: list) -> dict | None:
    """One member's normalized non-negative importance over its columns."""
    try:
        if hasattr(model, "feature_importances_"):
            vals = [float(v) for v in model.feature_importances_]
        elif hasattr(model, "coef_"):
            vals = [float(abs(v)) for v in
                    np.ravel(np.asarray(model.coef_, dtype=float))]
        else:
            return None
    except Exception:  # noqa: BLE001 - unfitted/foreign models skip honestly
        return None
    if len(vals) != len(columns) or sum(vals) <= 0:
        return None
    total = sum(vals)
    return {c: v / total for c, v in zip(columns, vals)}


def _ensemble_feature_weights(reg) -> dict[str, float] | None:
    """Blend-weighted pooled importances of the fitted score ensemble."""
    weights = dict(getattr(reg, "weights", None) or {})
    acc: dict[str, float] = {}
    for name in config.SCORE_ENSEMBLE_MEMBERS:
        w = weights.get(name, 0.0)
        try:
            w = float(w)
        except (TypeError, ValueError):
            continue
        if w <= 0:
            continue
        cols = list(((getattr(reg, "columns", None) or {}).get(name) or []))
        for side in ScoreEnsembleRegressor.SIDES:
            try:
                model, _ = reg.members[side][name]
            except (KeyError, TypeError):
                continue
            vec = _member_importance_vector(model, cols)
            if not vec:
                continue
            for col, val in vec.items():
                if col in config.TREE_CATEGORICAL_COLS:
                    continue
                acc[col] = acc.get(col, 0.0) + w * val
    total = sum(acc.values())
    if not acc or total <= 0:
        return None
    return {col: round(100.0 * val / total, 6)
            for col, val in acc.items()}


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


def walk_forward_oof_nb(game_df: pd.DataFrame, date_col: str = "gameday",
                        fold_list=None) -> dict:
    """NB-challenger twin of walk_forward_oof: identical folds and frame.

    Trial-only (2026-10-17): NB2 log-link LightGBM per side with
    train-fold dispersion, vs the Poisson incumbent. Fold rows carry the
    fitted per-side dispersions for diagnosis.
    """
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
            reg = ScoreRegressor(nb=True).fit(train)
            mu_h, mu_a = reg.predict(val)
            rh, ra = reg.nb_dispersion or (float("nan"), float("nan"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("NB distribution fold %s failed: %s", fold.fold_id, exc)
            mu_h = np.full(len(val), np.nan)
            mu_a = np.full(len(val), np.nan)
            rh, ra = float("nan"), float("nan")
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
                     "n_val": len(val), "r_home": rh, "r_away": ra})
    oof = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(oof):
        oof["margin"] = oof.home_score - oof.away_score
        oof["total"] = oof.home_score + oof.away_score
    return {"oof": oof, "fold_table": pd.DataFrame(rows)}


# ---------------------------------------------------------------------------
# Score ensemble (elasticnet + xgboost + lightgbm per side), structured on
# the binary moneyline (moneyline.py): the same member matrices and
# train-fold-only preprocessing (ml_mod.member_matrix / member_fit_input /
# TrainFoldPreprocessor), the same causal expanding OOF with prior-fold
# weights — but the blending objective is pooled Poisson deviance (the
# count-data logloss) in log space instead of logloss in logit space. One
# weight vector serves both sides (home+away pooled as independent Poisson
# observations). Strictly point-in-time: fold k is blended with weights
# earned on folds strictly before k, and no member fit ever sees val rows.
# ---------------------------------------------------------------------------


def _elasticnet_full_view(df: pd.DataFrame) -> pd.DataFrame:
    """Linear view plus raw per-side columns (arm C trial only).

    The moneyline linear view deliberately drops raw sides; this challenger
    view adds back every raw per-side column present in the frame so the
    elastic net can learn side-strength logic, not only gaps. Team-ID
    categoricals are excluded (arbitrary IDs as linear ordinals would be a
    bug). Standardization/NaN handling ride the existing train-fold
    preprocessor, which is column-generic. Default-off; see
    config.SCORE_EN_ELASTICNET_FULL_VIEW.
    """
    base = ml_mod.member_matrix("elasticnet", df)
    extra = [c for c in config.RAW_PER_SIDE_COLS
             if c in df.columns and c not in base.columns
             and c not in config.TREE_CATEGORICAL_COLS]
    if not extra:
        return base
    return pd.concat([base, df[extra]], axis=1)


def _make_score_member(name: str):
    """Construct one score-regression member (twin of moneyline._make_member)."""
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**config.SCORE_XGBOOST_PARAMS)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        params = dict(config.LIGHTGBM_REG_PARAMS)
        params["objective"] = "poisson"
        return LGBMRegressor(**params)
    if name == "lightgbm_nb":
        # NB-inside-ensemble challenger leg (trial-only, 2026-10-17): same
        # tree matrix and hyper-params as the Poisson leg; the NB objective
        # with its train-fold dispersion is set per fold at fit time (there
        # is no fold-independent default), so construction ships the
        # Poisson placeholder and fit() overrides it via set_params.
        from lightgbm import LGBMRegressor
        params = dict(config.LIGHTGBM_REG_PARAMS)
        params["objective"] = "poisson"
        return LGBMRegressor(**params)
    if name == "elasticnet":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(**config.SCORE_ELASTICNET_PARAMS)
    raise KeyError(f"unknown score ensemble member {name!r}")


def poisson_deviance(y, mu) -> float:
    """Mean Poisson deviance (2 * mean(y log(y/mu) - (y - mu)))."""
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, None)
    ok = np.isfinite(y) & np.isfinite(mu) & (y >= 0)
    if not ok.any():
        return float("nan")
    y, mu = y[ok], mu[ok]
    term = np.empty_like(y)
    pos = y > 0
    term[pos] = y[pos] * np.log(y[pos] / mu[pos]) - (y[pos] - mu[pos])
    term[~pos] = mu[~pos]  # limit of y*log(y/mu) at y=0 is 0
    return float(2.0 * term.mean())


def _log_blend_mus(P: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Geometric blend with per-row NaN-member skipping.

    Log-space twin of moneyline._logit_blend_matrix: positivity-preserving
    for score means, renormalized over the finite members of each row.
    """
    P = np.asarray(P, dtype=float)
    W = np.asarray(weights, dtype=float)
    if P.ndim != 2 or P.shape[1] != len(W):
        return np.full(len(P), np.nan)
    mask = np.isfinite(P) & (P > 0)
    active = (W > 0)[None, :] & mask
    logs = np.log(np.where(mask, P, 1.0))
    weighted = np.where(active, W[None, :], 0.0)
    denom = weighted.sum(axis=1)
    z = np.divide((logs * weighted).sum(axis=1), denom,
                  out=np.full(len(P), np.nan), where=denom > 0)
    return np.where(denom > 0, np.exp(z), np.nan)


def compute_score_weights(
    oof_members: dict[str, list[float]],
    y_oof,
) -> dict[str, float]:
    """Earn a deterministic simplex blend by pooled Poisson deviance.

    Regression twin of moneyline.compute_adaptive_weights: the objective is
    evaluated in log space with SLSQP over pooled home+away OOF pairs. A
    member missing on a row is skipped there; the remaining finite members
    are renormalized for that row.
    """
    y = np.asarray(y_oof, dtype=float)
    if y.size == 0:
        return {}
    labeled = np.isfinite(y) & (y >= 0)
    if labeled.sum() < 2:
        return {}
    arrays: dict[str, np.ndarray] = {}
    for name, values in oof_members.items():
        if values is None:
            continue
        p = np.asarray(values, dtype=float)
        if p.ndim != 1 or len(p) != len(y):
            continue
        arrays[name] = np.where(np.isfinite(p) & (p > 0), p, np.nan)
    names = sorted(name for name, p in arrays.items()
                   if np.isfinite(p[labeled]).sum() >= 2)
    if not names:
        return {}
    yy = y[labeled]
    P = np.column_stack([arrays[name][labeled] for name in names])
    finite_any = np.isfinite(P).any(axis=1)
    if finite_any.sum() < 2:
        return {}
    P = P[finite_any]
    yy = yy[finite_any]
    Z = np.log(np.clip(P, 1e-6, None))

    def blend(w: np.ndarray) -> np.ndarray:
        active = np.isfinite(P) & (w[None, :] > 0)
        weights = np.where(active, w[None, :], 0.0)
        denom = weights.sum(axis=1)
        z = np.divide((np.where(active, Z, 0.0) * weights).sum(axis=1),
                      denom, out=np.full(len(P), np.nan), where=denom > 0)
        return np.exp(z)

    def objective(w: np.ndarray) -> float:
        return poisson_deviance(yy, blend(w))

    w0 = np.full(len(names), 1.0 / len(names))
    if len(names) == 1:
        return {names[0]: 1.0}
    solo = {name: poisson_deviance(yy, P[:, i])
            for i, name in enumerate(names)}
    best = min(solo, key=solo.get)

    from scipy.optimize import minimize
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(names),
        constraints=({"type": "eq",
                      "fun": lambda w: float(w.sum() - 1.0)},),
        options={"maxiter": 300, "ftol": 1e-9},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return {name: float(name == best) for name in names}
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    weights = weights / weights.sum() if weights.sum() else w0
    if solo[best] < objective(weights) - 1e-12:
        return {name: float(name == best) for name in names}
    rounded = {name: round(float(value), 4)
               for name, value in zip(names, weights)}
    drift = round(1.0 - sum(rounded.values()), 4)
    if drift:
        top = max(names, key=rounded.get)
        rounded[top] = round(rounded[top] + drift, 4)
    return rounded


def _check_score_xgb_categories(model, X: pd.DataFrame) -> None:
    """XGBoost team columns must stay categorical (twin of the moneyline check)."""
    booster = model.get_booster()
    types = dict(zip(booster.feature_names, booster.feature_types))
    if any(types.get(c) != "c" for c in config.TREE_CATEGORICAL_COLS if c in X):
        raise RuntimeError("XGBoost team columns must remain categorical")


class ScoreEnsembleRegressor:
    """PIT per-side score-mean ensemble; blend weights set from prior OOF."""

    SIDES = ("home_score", "away_score")

    def __init__(self) -> None:
        self.members: dict[str, dict[str, tuple[Any, Any]]] = {}
        self.columns: dict[str, list[str]] = {}
        self.weights: dict[str, float] = dict(config.SCORE_ENSEMBLE_WEIGHTS)
        # NB-leg side-car: (log-mean init, dispersion) per side. Kept out
        # of the (model, pre) member tuples so fitted bundles keep their
        # shape and serving predict_member needs no signature change.
        self.nb_leg_state: dict[str, tuple[float, float]] = {}

    def set_weights(self, weights: dict[str, float]) -> "ScoreEnsembleRegressor":
        self.weights = dict(weights)
        return self

    def _matrix(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        # Same matrices the binary moneyline members consume (structural
        # property, not convention): tree views keep the categorical team-ID
        # pair, the elastic net takes the linear view — unless the full-view
        # challenger flag is set (arm C trial only): then the elastic net
        # additionally sees the standardized raw per-side columns (team-ID
        # categoricals stay excluded). Default-off is byte-identical arm B.
        if name == "elasticnet" and config.SCORE_EN_ELASTICNET_FULL_VIEW:
            X = _elasticnet_full_view(df)
        elif name == "lightgbm_nb":
            # Same tree view the Poisson leg consumes — the ONLY difference
            # between the legs is the likelihood, never the inputs.
            X = ml_mod.member_matrix("lightgbm", df)
        else:
            X = ml_mod.member_matrix(name, df)
        if name not in self.columns:
            self.columns[name] = list(X.columns)
        return X.reindex(columns=self.columns[name])

    def fit(self, df: pd.DataFrame) -> "ScoreEnsembleRegressor":
        df = folds_mod.canonical_sort(df)
        for side in self.SIDES:
            y = pd.to_numeric(df[side], errors="coerce")
            valid = y.notna() & (y >= 0)
            if valid.sum() < 2:
                raise ValueError("score ensemble requires settled scores")
            frame = df.loc[valid]
            y = y.loc[valid].to_numpy(dtype=float)
            fitted: dict[str, tuple[Any, Any]] = {}
            for name in config.SCORE_ENSEMBLE_MEMBERS:
                X = self._matrix(name, frame)
                pre = (ml_mod.TrainFoldPreprocessor().fit(X)
                       if name in ml_mod.LINEAR_MEMBERS else None)
                model = _make_score_member(name)
                X_fit = ml_mod.member_fit_input(name, X, pre)
                fit_kwargs: dict[str, Any] = {}
                if name in ("lightgbm", "lightgbm_nb"):
                    categorical = [c for c in config.TREE_CATEGORICAL_COLS
                                   if c in getattr(X_fit, "columns", [])]
                    if categorical:
                        fit_kwargs["categorical_feature"] = categorical
                if name == "lightgbm_nb":
                    r = estimate_nb_dispersion(y)
                    lm = float(np.log(max(float(np.mean(y)), 1e-6)))
                    model.set_params(
                        objective=NegativeBinomialObjective(r))
                    fit_kwargs["init_score"] = np.full(len(X_fit), lm)
                    self.nb_leg_state[side] = (lm, r)
                model.fit(X_fit, y, **fit_kwargs)
                if name == "xgboost":
                    _check_score_xgb_categories(model, X)
                fitted[name] = (model, pre)
            self.members[side] = fitted
        return self

    def predict_member(self, df: pd.DataFrame, name: str,
                       side: str) -> np.ndarray:
        model, pre = self.members[side][name]
        X = ml_mod.member_fit_input(name, self._matrix(name, df), pre)
        if name == "lightgbm_nb":
            # Raw margins exclude the init (LightGBM custom-objective
            # contract): add the stored train-fold log-mean back, then exp.
            lm, _ = self.nb_leg_state[side]
            lo, hi = NB_RAW_CLIP
            mu = np.exp(np.clip(
                np.asarray(model.predict(X), dtype=float) + lm, lo, hi))
            return np.clip(mu, 1e-6, None)
        return np.clip(np.asarray(model.predict(X), dtype=float), 1e-6, None)

    def predict(self, df: pd.DataFrame,
                weights: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        if not self.members:
            raise RuntimeError("score ensemble is not fitted")
        weights = self.weights if weights is None else weights
        out = []
        for side in self.SIDES:
            cols = []
            for name in config.SCORE_ENSEMBLE_MEMBERS:
                try:
                    cols.append(self.predict_member(df, name, side))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("score ensemble member %s/%s failed: %s",
                                   name, side, exc)
                    cols.append(np.full(len(df), np.nan))
            P = np.column_stack(cols)
            w = np.asarray([weights.get(name, 0.0)
                            for name in config.SCORE_ENSEMBLE_MEMBERS],
                           dtype=float)
            out.append(_log_blend_mus(P, w))
        return out[0], out[1]


def walk_forward_oof_ensemble(game_df: pd.DataFrame, date_col: str = "gameday",
                              fold_list=None, progress_every: int = 25) -> dict:
    """Causal expanding score-ensemble OOF with prior-fold deviance weights.

    Twin of walk_forward_oof above and moneyline.walk_forward_oof: fold k is
    blended with weights earned on folds strictly before k (pooled prior
    home+away OOF vs outcomes, Poisson deviance, plain simplex). Only
    regular-season rows from non-provisional folds grade the blender — the
    moneyline grading policy, so postseason/provisional rows are scored but
    never evidence for the model.
    """
    df = folds_mod.canonical_sort(game_df, date_col)
    folds = fold_list if fold_list is not None else folds_mod.make_folds(df, date_col)
    parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    prior_weights = dict(config.SCORE_ENSEMBLE_WEIGHTS)
    pooled: dict[str, list[float]] = {name: []
                                      for name in config.SCORE_ENSEMBLE_MEMBERS}
    pooled_y: list[float] = []

    for fold in folds:
        train, val = df.loc[fold.train_idx], df.loc[fold.val_idx]
        train_dates = pd.to_datetime(train[date_col], errors="raise").dt.normalize()
        val_dates = pd.to_datetime(val[date_col], errors="raise").dt.normalize()
        if train_dates.max() >= val_dates.min():
            raise ValueError("distribution training must precede validation dates")
        fold_weights = dict(prior_weights)
        if (pd.to_numeric(train["home_score"], errors="coerce").notna()
                .sum() < 2):
            logger.warning("skipping score-ensemble fold %s: insufficient labels",
                           fold.fold_id)
            continue
        try:
            ens = ScoreEnsembleRegressor().fit(train)
        except Exception as exc:  # noqa: BLE001
            logger.warning("score-ensemble fold %s failed: %s", fold.fold_id, exc)
            continue

        member_mu: dict[str, dict[str, np.ndarray]] = {}
        for name in config.SCORE_ENSEMBLE_MEMBERS:
            member_mu[name] = {}
            for side in ScoreEnsembleRegressor.SIDES:
                try:
                    member_mu[name][side] = ens.predict_member(val, name, side)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("score-ensemble fold %s member %s/%s failed: %s",
                                   fold.fold_id, name, side, exc)
                    member_mu[name][side] = np.full(len(val), np.nan)
        w = np.asarray([prior_weights.get(name, 0.0)
                        for name in config.SCORE_ENSEMBLE_MEMBERS], dtype=float)
        mu_h = _log_blend_mus(
            np.column_stack([member_mu[name]["home_score"]
                             for name in config.SCORE_ENSEMBLE_MEMBERS]), w)
        mu_a = _log_blend_mus(
            np.column_stack([member_mu[name]["away_score"]
                             for name in config.SCORE_ENSEMBLE_MEMBERS]), w)

        y_h = pd.to_numeric(val["home_score"], errors="coerce")
        y_a = pd.to_numeric(val["away_score"], errors="coerce")
        row = pd.DataFrame({
            "game_id": val.game_id.to_numpy(),
            "gameday": pd.to_datetime(val[date_col]).to_numpy(),
            "season": val.season.to_numpy() if "season" in val else np.nan,
            "fold_id": fold.fold_id, "mu_h": mu_h, "mu_a": mu_a,
            "home_score": y_h.to_numpy(float),
            "away_score": y_a.to_numpy(float),
        })
        for name in config.SCORE_ENSEMBLE_MEMBERS:
            row[f"mu_h_{name}"] = member_mu[name]["home_score"]
            row[f"mu_a_{name}"] = member_mu[name]["away_score"]
            row[f"weight_{name}"] = fold_weights.get(name, 0.0)

        is_post = (folds_mod.postseason_flag(val["game_type"]).to_numpy()
                   if "game_type" in val
                   else (pd.to_numeric(val["is_playoffs"], errors="coerce")
                         .fillna(0).gt(0.5).to_numpy()
                         if "is_playoffs" in val
                         else np.zeros(len(val), dtype=bool)))
        fold_provisional = bool(getattr(fold, "provisional", False))
        grades = np.zeros(len(val), dtype=bool) if fold_provisional else ~is_post
        row["grades_blend"] = grades

        # Pooled prior-fold evidence only after this fold is fixed.
        for name in config.SCORE_ENSEMBLE_MEMBERS:
            pooled[name].extend(row[f"mu_h_{name}"].to_numpy(float)[grades].tolist())
            pooled[name].extend(row[f"mu_a_{name}"].to_numpy(float)[grades].tolist())
        pooled_y.extend(row.home_score.to_numpy(float)[grades].tolist())
        pooled_y.extend(row.away_score.to_numpy(float)[grades].tolist())
        earned = compute_score_weights(pooled, np.asarray(pooled_y, dtype=float))
        if earned:
            prior_weights = earned

        fold_rows.append({"fold_id": fold.fold_id,
                         "val_start": str(fold.val_start.date()),
                         "val_end": str(fold.val_end.date()),
                         "n_train": int(len(train)), "n_val": int(len(val)),
                         "n_grading": int(grades.sum()),
                         "weights": fold_weights})
        parts.append(row)
        if progress_every and (fold.fold_id + 1) % progress_every == 0:
            logger.info("score-ensemble OOF fold %d/%d", fold.fold_id + 1, len(folds))

    oof = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(oof):
        oof["margin"] = oof.home_score - oof.away_score
        oof["total"] = oof.home_score + oof.away_score
    return {"oof": oof, "member_weights": prior_weights,
            "fold_table": pd.DataFrame(fold_rows)}


def fit_final_ensemble(game_df: pd.DataFrame) -> dict[str, dict[str, tuple[Any, Any]]]:
    """Fit all score members on every settled game for production serving.

    Twin of moneyline.fit_final_models: blend weights are NOT fitted here —
    the caller serves the walk's member_weights (OOF evidence), never an
    in-sample optimum.
    """
    frame = folds_mod.canonical_sort(game_df)
    ens = ScoreEnsembleRegressor().fit(frame)
    return dict(ens.members)


def predict_ensemble_slate(models: dict, slate_df: pd.DataFrame,
                           weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Blend a pending slate's per-side means with the deployed OOF weights."""
    mus = []
    for side in ScoreEnsembleRegressor.SIDES:
        cols = []
        for name in config.SCORE_ENSEMBLE_MEMBERS:
            entry = (models.get(side) or {}).get(name)
            if not entry:
                continue
            model, pre = entry
            try:
                X = ml_mod.member_fit_input(
                    name, ml_mod.member_matrix(name, slate_df), pre)
                cols.append((name, np.clip(np.asarray(model.predict(X),
                                                     dtype=float), 1e-6, None)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("serving score member %s/%s failed: %s",
                               name, side, exc)
        if not cols:
            mus.append(np.full(len(slate_df), np.nan))
            continue
        P = np.column_stack([c for _, c in cols])
        w = np.asarray([weights.get(name, 0.0) for name, _ in cols],
                       dtype=float)
        mus.append(_log_blend_mus(P, w))
    return mus[0], mus[1]


# ---------------------------------------------------------------------------
# Level-2 mixture (Option B): each member's mus price their OWN CDF through
# an origin's fitted maps (variance + calibration + lattice, all fitted on
# strictly-prior evidence exactly as production fits them), and the served
# predictive CDF is the per-row mixture of the member CDFs with weights
# earned on prior-fold OOF CRPS. A mixture of CDFs IS a CDF (monotone,
# [0, 1]), so Monte Carlo sampling and every downstream leg apply
# UNCHANGED — no replumbing below this layer. Member supports can differ
# (the lattice expands with each member's means), so mixture aligns member
# PMFs onto their union integer lattice with zero-padding, the same rule
# family.lattice_evidence uses. Strictly point-in-time: fold k mixes with
# weights earned on folds strictly before k.
# ---------------------------------------------------------------------------


def _pmf_from_cdf(cdf: np.ndarray) -> np.ndarray:
    cdf = np.asarray(cdf, dtype=float)
    return np.diff(np.column_stack([np.zeros(len(cdf)), cdf]), axis=1)


def _align_pmfs_to_union(pmfs: list[np.ndarray],
                         supports: list[np.ndarray]):
    """Zero-pad member PMFs onto their union integer lattice."""
    lo = min(int(s[0]) for s in supports)
    hi = max(int(s[-1]) for s in supports)
    union = np.arange(lo, hi + 1)
    out = []
    for pmf, sup in zip(pmfs, supports):
        wide = np.zeros((len(pmf), len(union)))
        j0 = int(sup[0]) - lo
        wide[:, j0:j0 + len(sup)] = pmf
        out.append(wide)
    return out, union


def mixture_cdf(member_cdfs: dict[str, tuple[np.ndarray, np.ndarray]],
                weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Per-row CDF mixture over available members, renormalized per row.

    member_cdfs maps name -> (support, cdf) with matching row counts; rows
    that are NaN for a member are skipped for that member. Returns
    (union_support, cdf) with NaN rows where no member prices. The output
    is a valid CDF wherever any member prices (convex combination of
    nondecreasing [0, 1] functions).
    """
    names = [n for n in member_cdfs if (weights.get(n, 0.0) or 0.0) > 0]
    if not names:
        raise ValueError("mixture needs at least one positively weighted member")
    n_rows = len(member_cdfs[names[0]][1])
    if any(len(member_cdfs[n][1]) != n_rows for n in names):
        raise ValueError("mixture members must share row order and count")
    pmfs, supports, avail = [], [], []
    for name in names:
        sup, cdf = member_cdfs[name]
        ok = np.isfinite(cdf).all(axis=1)
        avail.append(ok)
        pmfs.append(_pmf_from_cdf(np.where(ok[:, None], cdf, 0.0)))
        supports.append(np.asarray(sup))
    aligned, union = _align_pmfs_to_union(pmfs, supports)
    w = np.asarray([float(weights[n]) for n in names], dtype=float)
    pmf_mix = np.zeros((n_rows, len(union)))
    mass = np.zeros(n_rows)
    for wi, pmf, ok in zip(w, aligned, avail):
        pmf_mix[ok] += wi * pmf[ok]
        mass[ok] += wi
    none = mass <= 0
    pmf_mix[~none] /= mass[~none, None]
    cdf_mix = np.cumsum(pmf_mix, axis=1)
    cdf_mix = np.clip(cdf_mix, 0.0, 1.0)
    cdf_mix[none] = np.nan
    return union, cdf_mix


def compute_mixture_weights(blocks) -> dict[str, float]:
    """Earn a deterministic simplex blend by pooled CRPS.

    Distribution twin of compute_score_weights: ``blocks`` are
    (member_cdfs, y) pairs with member_cdfs mapping name to
    (support, cdf) sharing that block's row order, supports possibly
    differing block to block. Only rows where y is finite and EVERY
    candidate member prices are comparable evidence (complete-case per
    block); a member missing wholesale (< 2 finite pooled rows) cannot
    earn weight and is dropped from candidacy first. Objective is pooled
    mean CRPS of the mixture, SLSQP simplex, one-hot fallback to the
    solo-best member. (No y >= 0 filter: margin outcomes go negative.)
    """
    per_member: dict[str, list] = {}
    for member_cdfs, y in blocks:
        y = np.asarray(y, dtype=float)
        for name, (support, cdf) in member_cdfs.items():
            cdf = np.asarray(cdf, dtype=float)
            if len(cdf) != len(y):
                raise ValueError("mixture evidence members must share row order")
            ok = np.isfinite(cdf).all(axis=1) & np.isfinite(y)
            per_member.setdefault(name, []).append(
                (cdf[ok], np.asarray(support), y[ok]))
    names = sorted(name for name, parts in per_member.items()
                   if sum(len(p[0]) for p in parts) >= 2)
    if not names:
        return {}
    # Complete-case rows per block: every candidate prices, so the pooled
    # mixture, every solo score, and the one-hot guard share one row set.
    pooled: dict[str, list] = {name: [] for name in names}
    for member_cdfs, y in blocks:
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(y)
        for name in names:
            _, cdf = member_cdfs.get(name, (None, None))
            if cdf is None:
                ok = np.zeros(len(y), dtype=bool)
                break
            cdf = np.asarray(cdf, dtype=float)
            if len(cdf) != len(y):
                raise ValueError("mixture evidence members must share row order")
            ok = ok & np.isfinite(cdf).all(axis=1)
        if ok.sum() < 1:
            continue
        for name in names:
            support, cdf = member_cdfs[name]
            pooled[name].append((np.asarray(cdf, dtype=float)[ok],
                                 np.asarray(support), y[ok]))
    total = sum(len(p[0]) for p in pooled[names[0]])
    if total < 2:
        return {}
    union = np.arange(
        min(int(p[1][0]) for parts in pooled.values() for p in parts),
        max(int(p[1][-1]) for parts in pooled.values() for p in parts) + 1)
    stacked, yy = [], []
    for name in names:
        parts = pooled[name]
        wide = np.zeros((sum(len(p[0]) for p in parts), len(union)))
        ys = []
        r = 0
        for cdf, support, y in parts:
            pmf = _pmf_from_cdf(cdf)
            j0 = int(support[0]) - int(union[0])
            wide[r:r + len(pmf), j0:j0 + len(support)] = pmf
            ys.append(y)
            r += len(pmf)
        stacked.append(np.cumsum(wide, axis=1))
        yy.append(np.concatenate(ys))
    P = np.stack(stacked)  # (member, row, bin)
    yy = yy[0]

    def objective(w: np.ndarray) -> float:
        # Renormalize inside the objective: SLSQP's finite-difference
        # probes leave the simplex (negative / summing past 1), which can
        # push mixture CDF values outside [0, 1] and trip the CRPS range
        # validation. Pricing renormalizes per row the same way, so the
        # optimum is unaffected — only the probes become valid CDFs.
        w = np.clip(np.asarray(w, dtype=float), 0.0, None)
        total_w = w.sum()
        if total_w <= 0:
            return float(np.mean(family.crps(P.mean(axis=0), union, yy)))
        mix = np.tensordot(w / total_w, P, axes=([0], [0]))
        # Same 1-ulp clip the pricing path (mixture_cdf) applies: the
        # convex combination of saturating CDF tails can round to
        # 1 + 1ulp, which trips the CRPS range validation.
        mix = np.clip(mix, 0.0, 1.0)
        return float(np.mean(family.crps(mix, union, yy)))

    w0 = np.full(len(names), 1.0 / len(names))
    if len(names) == 1:
        return {names[0]: 1.0}
    solo = {}
    for i, name in enumerate(names):
        try:
            solo[name] = float(np.mean(family.crps(P[i], union, yy)))
        except ValueError:
            solo[name] = float("inf")
    best = min(solo, key=solo.get)

    from scipy.optimize import minimize
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(names),
        constraints=({"type": "eq",
                      "fun": lambda w: float(w.sum() - 1.0)},),
        options={"maxiter": 300, "ftol": 1e-9},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return {name: float(name == best) for name in names}
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    weights = weights / weights.sum() if weights.sum() else w0
    if solo[best] < objective(weights) - 1e-12:
        return {name: float(name == best) for name in names}
    rounded = {name: round(float(value), 4)
               for name, value in zip(names, weights)}
    drift = round(1.0 - sum(rounded.values()), 4)
    if drift:
        top = max(names, key=rounded.get)
        rounded[top] = round(rounded[top] + drift, 4)
    return rounded


def price_member_cdfs(frame: pd.DataFrame, mus: dict[str, tuple[np.ndarray, np.ndarray]],
                      maps: dict) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Price each member's mus through fitted origin maps, per kind.

    mus maps member name -> (mu_h, mu_a). Rows where a member's mus are
    non-finite/non-positive are skipped for that member (NaN CDF there).
    Returns {kind: {name: (support, cdf)}} with full row counts.
    """
    out: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for name, (mu_h, mu_a) in mus.items():
        mu_h = np.asarray(mu_h, dtype=float)
        mu_a = np.asarray(mu_a, dtype=float)
        ok = (np.isfinite(mu_h) & np.isfinite(mu_a)
              & (mu_h > 0) & (mu_a > 0))
        if not ok.any():
            continue
        part = frame.iloc[np.flatnonzero(ok)].copy()
        part["mu_h"] = mu_h[ok]
        part["mu_a"] = mu_a[ok]
        cdfs = _distribution_cdfs(part, maps, sampled=False)
        for kind, (support, cdf) in cdfs.items():
            full = np.full((len(frame), len(support)), np.nan)
            full[np.flatnonzero(ok)] = cdf
            out.setdefault(kind, {})[name] = (np.asarray(support), full)
    return out


def walk_forward_mixture(oof: pd.DataFrame, bundle: dict,
                         kinds: tuple[str, ...] = ("total", "margin"),
                         progress_every: int = 25) -> dict:
    """Causal per-origin CDF mixture over ensemble member mus.

    Twin of distribution_score_report's loop: for each origin in the causal
    bundle, price every member's OOF mus through that origin's fitted maps,
    mix the member CDFs with weights earned on strictly-prior OOF CRPS
    (same pooled simplex discipline as the level-1 blender, per kind), and
    score the mixture against the blended-single-mus CDF. Only
    grading rows (oof["grades_blend"] when present, else all finite rows)
    are weight evidence — never the row being scored.
    """
    members = list(config.SCORE_ENSEMBLE_MEMBERS)
    have = [m for m in members
            if f"mu_h_{m}" in oof.columns and f"mu_a_{m}" in oof.columns]
    if not have:
        raise ValueError("mixture walk needs per-member OOF mus columns")
    if "mu_h" not in oof.columns or "mu_a" not in oof.columns:
        raise ValueError("mixture walk needs blended-single OOF mus columns")
    frame = oof.reset_index(drop=True)
    grades_all = (frame["grades_blend"].to_numpy(bool)
                  if "grades_blend" in frame.columns
                  else np.ones(len(frame), dtype=bool))
    pooled: dict[str, list] = {k: [] for k in kinds}
    pooled_n = {k: 0 for k in kinds}
    prior_weights = {k: {m: 1.0 / len(have) for m in have} for k in kinds}
    parts: list[pd.DataFrame] = []
    weight_rows: list[dict] = []
    origins = bundle.get("origins") or []
    blocks = _distribution_blocks(frame)
    if len(blocks) != len(origins):
        raise ValueError("mixture walk needs one bundle origin per OOF block")
    for bi, ((_, _, mask), origin) in enumerate(zip(blocks, origins)):
        idx = np.flatnonzero(mask)
        part = frame.iloc[idx]
        maps = origin["maps"]
        mus = {m: (part[f"mu_h_{m}"].to_numpy(float),
                   part[f"mu_a_{m}"].to_numpy(float)) for m in have}
        # Blended-single baseline through the same maps.
        single = {}
        ok_single = (np.isfinite(part["mu_h"].to_numpy(float))
                       & np.isfinite(part["mu_a"].to_numpy(float)))
        part_y_total = (part.home_score + part.away_score).to_numpy(float) \
            if "home_score" in part.columns else np.full(len(part), np.nan)
        part_y_margin = (part.home_score - part.away_score).to_numpy(float) \
            if "home_score" in part.columns else np.full(len(part), np.nan)
        if ok_single.any():
            single = _distribution_cdfs(part.iloc[np.flatnonzero(ok_single)],
                                        maps, sampled=False)
        priced = price_member_cdfs(part.reset_index(drop=True), mus, maps)
        row = pd.DataFrame({"game_id": part.game_id.to_numpy(),
                            "fold_id": (part.fold_id.to_numpy()
                                         if "fold_id" in part.columns else bi)})
        mix_weights = {k: dict(prior_weights[k]) for k in kinds}
        for kind in kinds:
            y = part_y_total if kind == "total" else part_y_margin
            row[f"crps_single_{kind}"] = np.nan
            row[f"crps_mix_{kind}"] = np.nan
            for name in have:
                row[f"weight_mix_{kind}_{name}"] = mix_weights[kind].get(name, 0.0)
            if kind in single:
                sup_s, cdf_s = single[kind]
                sub = np.flatnonzero(ok_single)
                keep = np.isfinite(y[sub])
                if keep.any():
                    full_s = np.full(len(part), np.nan)
                    full_s[sub[keep]] = lattice_crps(
                        cdf_s[keep], sup_s, y[sub[keep]])
                    row[f"crps_single_{kind}"] = full_s
            if kind not in priced:
                continue
            try:
                union, cdf_mix = mixture_cdf(priced[kind], prior_weights[kind])
            except ValueError:
                continue
            finite = np.isfinite(cdf_mix).all(axis=1) & np.isfinite(y)
            if finite.any():
                full_m = np.full(len(part), np.nan)
                full_m[np.flatnonzero(finite)] = lattice_crps(
                    cdf_mix[finite], union, y[finite])
                row[f"crps_mix_{kind}"] = full_m
            # Pool this block's grading rows AFTER scoring it (row-aligned
            # across members; complete-case comparability is enforced at
            # earn time, never by dropping rows here).
            g = grades_all[idx] & np.isfinite(y)
            if g.any() and kind in priced:
                keep = np.flatnonzero(g)
                pooled[kind].append(
                    ({name: (sup, cdf[keep])
                      for name, (sup, cdf) in priced[kind].items()
                      if name in have}, y[g]))
            pooled_n[kind] += int(g.sum())
        # Earn next block's weights from strictly-prior pooled evidence.
        for kind in kinds:
            if pooled_n[kind] < 2:
                continue
            earned = compute_mixture_weights(pooled[kind])
            if earned:
                prior_weights[kind] = earned
        weight_rows.append({"block": bi,
                           "used": {k: dict(mix_weights[k]) for k in kinds},
                           "next": {k: dict(prior_weights[k])
                                    for k in kinds}})
        parts.append(row)
        if progress_every and (bi + 1) % progress_every == 0:
            logger.info("mixture OOF block %d/%d", bi + 1, len(blocks))
    mix_oof = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    summary = {}
    for kind in kinds:
        s = mix_oof[f"crps_single_{kind}"].to_numpy(float)
        m = mix_oof[f"crps_mix_{kind}"].to_numpy(float)
        ok = np.isfinite(s) & np.isfinite(m)
        summary[kind] = {
            "n": int(ok.sum()),
            "crps_single": round(float(s[ok].mean()), 6) if ok.any() else None,
            "crps_mix": round(float(m[ok].mean()), 6) if ok.any() else None,
            "paired_gain": (round(float((s[ok] - m[ok]).mean()), 6)
                             if ok.any() else None),
            "final_weights": dict(prior_weights[kind]),
        }
    return {"oof": mix_oof, "weights": prior_weights,
            "weight_history": weight_rows, "summary": summary}


def predict_mixture_slate(models: dict, slate_df: pd.DataFrame,
                          serving_maps: dict,
                          weights_l2: dict[str, dict[str, float]],
                          kinds: tuple[str, ...] = ("total", "margin")
                          ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Blend a pending slate's per-member CDFs with deployed OOF weights.

    Each member's own mus price their own CDF through the serving maps;
    weights_l2 (per kind, from the mixture walk) mixes the CDFs. Returns
    {kind: (support, cdf)} for the caller to sample exactly like
    _distribution_cdfs output.
    """
    mus = {}
    for side in ScoreEnsembleRegressor.SIDES:
        for name in config.SCORE_ENSEMBLE_MEMBERS:
            entry = (models.get(side) or {}).get(name)
            if not entry:
                continue
            model, pre = entry
            try:
                X = ml_mod.member_fit_input(
                    name, ml_mod.member_matrix(name, slate_df), pre)
                mus.setdefault(name, {})[side] = np.clip(
                    np.asarray(model.predict(X), dtype=float), 1e-6, None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("serving mixture member %s/%s failed: %s",
                               name, side, exc)
    per_member = {}
    for name, sides in mus.items():
        if set(sides) != {"home_score", "away_score"}:
            continue
        per_member[name] = (sides["home_score"], sides["away_score"])
    priced = price_member_cdfs(slate_df.reset_index(drop=True),
                               per_member, serving_maps)
    out = {}
    for kind in kinds:
        if kind not in priced:
            continue
        try:
            out[kind] = mixture_cdf(priced[kind], weights_l2.get(kind, {}))
        except ValueError:
            continue
    return out


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


def _distribution_no_tie(cdf, support, home_share=None, tie_rate=None):
    # Structural no-tie rule of the margin lattice. The NHL supplies a
    # causal OT/SO home-win share (maps["lattice"]["margin"]
    # ["ot_home_share"], estimated from prior OOF one-goal finals) so the
    # P(M=0) mass is redistributed to +/-1 by OT win probabilities from OOF;
    # None keeps the exact equal split every pre-lattice bundle used. The
    # NFL instead supplies a fitted overtime tie_rate (preregistered
    # ("nfl", "margin", "tie_mass")); None keeps the exact pre-lattice
    # valid tie mass. Other sports ignore tie_rate (passthrough).
    return family.no_tie(cdf, support, DISTRIBUTION_SPORT, home_share,
                         tie_rate=tie_rate)


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
            cdf = _distribution_no_tie(cdf, support, spec.get("ot_home_share"),
                                         tie_rate=spec.get("tie_rate"))
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
        if kind == "margin" and DISTRIBUTION_SPORT == "nba":
            # Diagnostic margin-push extension (2026-10-16, config.
            # MARGIN_PUSH_EXTENSION): push-only mass for +-21..30 from the
            # SAME sampled CDF (never refitted) so the Distribution tab's
            # modeled line is gapless to +-30. Push-only, like the totals
            # full-support PMF: no cover/away legs outside the market grid.
            # Additive setdefault: a future grid change that prices these
            # lines owns them instead.
            for k in config.MARGIN_PUSH_EXTENSION:
                upper, lower = at(float(k)), at(np.ceil(float(k)) - 1.)
                updates.setdefault(_grid_key("p_push", float(k)),
                                   np.maximum(upper - lower, 0.))
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


# ---------------------------------------------------------------------------
# NBA spread key-number mass transfer (2026-10-16, PRE-REGISTERED)
# ---------------------------------------------------------------------------
# WHY (measured on the shipped 20261016 artifact from its own delivered
# columns — run_diagnostics/nba_keynumber_align_probe.py, untracked): the
# spread lattice's teeth sit at the registered key numbers. The model sits
# ABOVE the actuals at M = -1, +1, -6, +14, +15, +18 (pooled residuals
# -0.0027/-0.0042/-0.0025/-0.0029/-0.0034/-0.0036) and BELOW them at
# M = -10, -9, -2, +6, +7, +11, +13, +16 (+0.0031/+0.0032/+0.0029/+0.0035/
# +0.0051/+0.0042/+0.0037/+0.0033) — every one of the fourteen bins holding
# its sign on the time-ordered held-out tail as well (the -2 tooth is the
# only one that flips, at n=720 noise). Each tooth is smaller than its own
# 1-sigma binomial floor alone (~0.0037 at n_train ~1680), which is why the
# generic localized residual prices them shrink-limited: it can only REMOVE
# mass with a mass-proportional sink smeared over every bin, so it cannot
# make the one trade that fixes two teeth at once — the measured NHL empty-net
# and MLB one-run failure mode.
#
# MECHANISM — registers and direction declared BEFORE fitting (this module IS
# the disclosure record; the shared gaussian_family.py copies stay
# byte-identical and its four-backend test keeps them that way — the MLB
# one-run rebalance precedent). Mass moves from the six registered DONORS
# into the eight registered RECIPIENTS, and every quantity is DYNAMIC:
#
#   demand(recipient) = SE-shrunk train residual at the recipient bin under
#       the declared demand relief KEYNUMBER_DEMAND_RELIEF on the
#       mass-proportional prior scale (the NFL/NHL spike-relief precedent,
#       shipped at the value the ladder measured — see the probe log),
#   capacity(donor)   = clip(-train residual, 0, model mass) — the donor's
#       OWN ACTUALS-SUPPORTED excess (the NHL empty-net supply rule): a
#       donor the actuals show SHORT funds exact 0, and mass is never taken
#       from a bin the evidence does not over-predict. Supply is deliberately
#       UN-shrunk: the trade moves existing mass, so shrinking both sides
#       double-shrinks it — measured as the binding defect of the first pass
#       (supply cut to 0.0020 against 0.0130 of demand, a near no-op that
#       moved no pooled residual by more than 0.0002); with the raw supply
#       rule the amount rises to meet the declared demand and the held-out z
#       rises with it (ladder rows for both definitions are in
#       run_diagnostics/nba_keynumber_ladder.log),
#   amount            = SUM of the declared recipient demands (the
#       recipients' claim; one scalar, not fourteen),
#   gains             = an EQUAL unit at each ACTIVE recipient (the
#       registered template — proportions are rule-derived and carry NO
#       outcome data: re-estimating fourteen proportions from train
#       residuals that are individually below their noise floor cost a
#       measured 0.9 held-out z on the same evidence
#       (run_diagnostics/nba_keynumber_shape_probe.py: z 3.023 -> 3.995
#       at relief 3, same amount)),
#   draws             = each donor funds pro rata over its capacity,
#       actuals-CAPPED per donor (never beyond its own excess),
#   sink              = the un-funded remainder falls to the family's
#       mass-proportional sink over bins that are neither registered nor
#       the no-tie forbidden bin — the MLB one-run rebalance funding rule.
#
# So the vector stays zero-sum by construction, the sink can only take from
# bins OUTSIDE the registered set, and every registered bin carries its
# residual / demand / capacity / gain / drawn entry in the report. GATES: the declared DIRECTION is enforced by the sizing itself on
# TRAIN evidence (a recipient not under-predicted demands 0, a donor not
# over-predicted funds 0), then a ONE-SIDED paired held-out test at the
# UNCHANGED GENERIC bar (family.LATTICE_GATE_Z — no structural-bar claim is
# made or needed: these teeth are empirical spread-key-number structure, not
# a rule-derived parameter), and the per-bin collateral guard (no bin's
# |residual| may worsen past its own noise floor). A refused transfer
# returns no delta and prices byte-identically to before.
#
# Registered bins (declared 2026-10-16 from the measurement above):
#   donors      = over-fit spikes the actuals do not support
#   recipients  = under-fit spikes the actuals demand
KEYNUMBER_TRANSFER_DONORS = (-1, 1, -6, 14, 15, 18)
KEYNUMBER_TRANSFER_RECIPIENTS = (-10, -9, -2, 6, 7, 11, 13, 16)
# Demand relief on the recipient's prior scale (tau = LATTICE_PRIOR_FRAC *
# relief * local model mass): the unboosted mass-proportional prior keeps a
# rule-sized tooth at a fraction of its own residual — the "spikes underfit
# the residual" complaint of the NFL (-3/+3, ~0.36) and NHL (|M|=3, ~1/3)
# ladders. Shipped value is chosen by the captured-evidence ladder in
# run_diagnostics/nba_keynumber_ladder.* (untracked), disclosed in the
# report; a deeper demand still loses to its donor capacity, the collateral
# guard or the held-out gate, so relief can never force an admission.
KEYNUMBER_DEMAND_RELIEF = 3.


def fit_keynumber_transfer(pmfs, outcomes, support,
                           demand_relief=None,
                           bar=family.LATTICE_GATE_Z,
                           min_history=family.MIN_HISTORY):
    """Preregistered NBA spread key-number donor -> recipient mass transfer.

    See the module header above for the pre-registration. Sizing is fully
    dynamic (demand from the recipients' own shrunk under-fit residuals under
    the declared relief, supply from the donors' own actuals-capped excess,
    amount = min of the two totals, pro-rata on each side); nothing here is
    a constant amplitude.

    Returns ``(delta | None, report)``: the zero-sum, capped, rounded pure
    transfer vector, or None when the evidence, the direction, the collateral
    guard or the held-out gate refuses — plus the per-bin disclosure report.
    """
    if demand_relief is None:
        # Resolved at CALL time so the declared constant is a single knob
        # (the causal-walk ladder reads it at runtime; tests can see it).
        demand_relief = KEYNUMBER_DEMAND_RELIEF
    report: dict = {"estimator": "keynumber_spread_transfer",
                    "sport": DISTRIBUTION_SPORT,
                    "mechanism": "keynumber_donor_recipient",
                    "bar": float(bar),
                    "demand_relief": float(demand_relief),
                    "donors": list(KEYNUMBER_TRANSFER_DONORS),
                    "recipients": list(KEYNUMBER_TRANSFER_RECIPIENTS),
                    "bins": []}
    support = np.asarray(support)
    pmfs = np.empty((0, len(support))) if pmfs is None else np.asarray(pmfs, float)
    outcomes = np.empty(0) if outcomes is None else np.asarray(outcomes, float)
    if pmfs.ndim != 2 or pmfs.shape[1] != len(support) or len(pmfs) != len(outcomes):
        report["reason"] = "misaligned_evidence"
        return None, report
    inside = ((outcomes >= support[0]) & (outcomes <= support[-1])
              & np.isfinite(outcomes))
    n_test = int(family.LATTICE_TEST_FRACTION * len(outcomes))
    if (inside.sum() < int(min_history) or n_test < family.LATTICE_MIN_TEST
            or len(outcomes) - n_test < int(min_history)):
        report["reason"] = "insufficient_held_out_evidence"
        return None, report
    train = np.flatnonzero(inside[:len(outcomes) - n_test])
    # Absolute tail positions (the fit_ot_home_share discipline): the gate
    # scores the time-ordered held-out tail itself, never rows the sizing
    # was computed from.
    hold = np.arange(len(outcomes) - n_test, len(outcomes))
    hold = hold[inside[hold]]
    if len(train) < int(min_history) or len(hold) < family.LATTICE_MIN_TEST:
        report["reason"] = "insufficient_held_out_evidence"
        return None, report
    n_train = max(int(len(train)), 1)
    pbar = np.clip(pmfs[train].mean(axis=0), 0., 1.)
    obs = np.bincount(np.searchsorted(support, outcomes[train]),
                      minlength=len(support)).astype(float) / n_train
    resid = obs - pbar
    se = np.sqrt(pbar * (1. - pbar) / n_train)
    index = {int(v): i for i, v in enumerate(support)}
    # Sizing, per registered bin, on TRAIN evidence only. A registered bin
    # outside this origin's support simply contributes 0 (per-bin skip, the
    # NHL per-pair rule) — it is never re-guessed from a neighbouring bin.
    demands: dict = {}
    supplies: dict = {}
    for v in KEYNUMBER_TRANSFER_RECIPIENTS:
        if int(v) not in index:
            continue
        i = index[int(v)]
        demand = float(family.lattice_local_deltas(
            float(demand_relief) * pbar[i:i + 1], resid[i:i + 1],
            se[i:i + 1])[0])
        demand = max(demand, 0.)          # an over-predicted recipient demands 0
        demands[int(v)] = demand
        report["bins"].append({"bin": int(v), "role": "recipient",
                               "residual": round(float(resid[i]), 6),
                               "demand": round(demand, 6), "gain": 0.0})
    for v in KEYNUMBER_TRANSFER_DONORS:
        if int(v) not in index:
            continue
        i = index[int(v)]
        capacity = float(np.clip(-resid[i], 0., pbar[i]))
        supplies[int(v)] = capacity       # an under-predicted donor funds 0
        report["bins"].append({"bin": int(v), "role": "donor",
                               "residual": round(float(resid[i]), 6),
                               "capacity": round(capacity, 6), "drawn": 0.0})
    demand_total = float(sum(demands.values()))
    capacity_total = float(sum(supplies.values()))
    report["demand_total"] = round(demand_total, 6)
    report["capacity_total"] = round(capacity_total, 6)
    if demand_total <= 0.:
        report["reason"] = "no_recipient_underfit"
        return None, report
    if capacity_total <= 0.:
        report["reason"] = "no_donor_excess"
        return None, report
    active = {v: d for v, d in demands.items() if d > 0.}
    amount = float(demand_total)          # the declared claim — 1 fitted scalar
    report["amount"] = round(amount, 6)
    # Registered template: an equal unit at each ACTIVE recipient. The
    # proportions are rule-derived (declared in the header BEFORE fitting)
    # and hold no outcome data, so the per-origin gate does not pay for
    # re-estimating fourteen noisy proportions before it can score anything.
    report["template"] = "equal_per_active_recipient"
    delta = np.zeros(len(support))
    rows = {row["bin"]: row for row in report["bins"]}
    unit = amount / len(active)
    for v in active:
        delta[index[v]] += unit
        rows[v]["gain"] = round(float(unit), 6)
    # Funding: donors pro rata over capacity, actuals-CAPPED per donor; the
    # whole donor side scales by min(1, amount / capacity_total).
    scale = min(1., amount / capacity_total) if capacity_total > 0. else 0.
    drawn_total = 0.
    for v, capacity in supplies.items():
        draw = capacity * scale
        delta[index[v]] -= draw
        drawn_total += draw
        rows[v]["drawn"] = round(float(draw), 6)
    report["funded_by_donors"] = round(float(drawn_total), 6)
    # Un-funded remainder falls to the MLP mass-proportional sink over bins
    # that are neither registered nor forbidden (the MLB funding rule).
    registered = {index[int(v)] for v in (*demands, *supplies) if int(v) in index}
    banned = {index[int(v)] for v in
              (family.lattice_config(DISTRIBUTION_SPORT, "margin")
               or {}).get("forbidden_bins", ()) if int(v) in index}
    sink = np.array([i for i in range(len(support))
                     if i not in registered and i not in banned])
    sink_mass = float(np.clip(pbar[sink], 0., None).sum()) if len(sink) else 0.
    shortfall = float(amount - drawn_total)
    if shortfall > 0. and sink_mass > 0.:
        delta[sink] -= shortfall * np.clip(pbar[sink], 0., None) / sink_mass
    report["funded_by_sink"] = round(max(shortfall, 0.), 6)
    report["zero_sum_free"] = round(float(delta.sum()), 9)
    # Per-bin collateral guard: no bin's |residual| may worsen by more than
    # its own 1-sigma binomial noise floor (pooled over all inside rows).
    n_all = max(int(inside.sum()), 1)
    pooled = np.clip(pmfs[inside].mean(axis=0), 0., 1.)
    pooled_obs = np.bincount(np.searchsorted(support, outcomes[inside]),
                             minlength=len(support)).astype(float) / n_all
    pooled_resid = pooled_obs - pooled
    se_pool = np.sqrt(pooled * (1. - pooled) / n_all)
    excess = ((np.abs(pooled_resid - delta) - np.abs(pooled_resid)) - se_pool)
    report["collateral_max_excess"] = round(float(excess.max()), 6)
    if not (excess <= 0.).all():
        report["reason"] = "collateral_violation"
        return None, report
    # One-sided paired held-out gate: the SAME squared-probability loss the
    # generic fit uses, at the UNCHANGED generic bar.
    indices_hold = np.searchsorted(support, outcomes[hold])
    d = (-2. * delta[indices_hold] + 2. * (pmfs[hold] @ delta)
         + float(delta @ delta))
    ses = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
    z = float(-d.mean() / ses) if ses > 0. else 0.
    report["z"] = round(z, 3)
    report["n_test"] = int(len(hold))
    if not (d.mean() < 0. and z >= bar):
        report["reason"] = "held_out_not_significant"
        return None, report
    delta = family._zero_sum_mass_proportional(delta, pbar)
    # Forbidden bins (margin 0, the no-tie structural rule) are zeroed here
    # as well as in lattice_correct — the transfer never registers them, and
    # the mass-proportional zero-sum must not graduate into one.
    for v in (family.lattice_config(DISTRIBUTION_SPORT, "margin")
              or {}).get("forbidden_bins", ()):
        if int(v) in index:
            delta[index[int(v)]] = 0.
    delta = np.clip(delta, -family.LATTICE_DELTA_CAP, family.LATTICE_DELTA_CAP)
    delta[np.abs(delta) < family.LATTICE_DELTA_EPS] = 0.
    delta = np.round(delta, family.LATTICE_ROUND)
    report["significant"] = True
    report["delta_l1"] = round(float(np.abs(delta).sum()), 6)
    report["delta_max"] = round(float(np.abs(delta).max()), 6)
    return [float(v) for v in delta], report


def _fit_lattice_maps(outcomes, parameters, evidence):
    """Independent per-target lattice fits — totals AND margin.

    Structural terms ride the VARIANCE parameters (NHL totals OT/SO parity
    alpha; the no-tie +/-1 split weight — NHL OT/SO home-win share, MLB
    walk-off home-win share — estimated causally from prior OOF one-goal
    own BASE-model evidence — pmfs of strictly-prior OOF rows, each priced
    with its own origin's maps — per target, behind the held-out
    significance gate. Every value is plain JSON so stored bundles reprice
    byte-identically.

    The preregistered NBA spread key-number transfer is fitted FIRST on the
    margin evidence (each side sized dynamically from actuals, gated once at
    the unchanged generic bar); the generic localized residual then runs on
    transfer-adjusted pmfs so it prices the leftover teeth instead of
    re-fighting them, and both components compose into one stored delta
    (the NHL transfer-then-generic order). Cold starts and refused
    transfers keep the exact shipped behavior.
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
    # Preregistered key-number transfer (module header above): fitted from
    # the margin evidence FIRST, so the generic residual never has to
    # rediscover (and currently cannot make) the donor -> recipient trade.
    # The disclosure always rides the margin spec; an unconfirmed transfer
    # ships no delta and prices byte-identically.
    fit_evidence = dict(evidence or {})
    transfer_delta = None
    margin_ev = (evidence or {}).get("margin") or {}
    if margin_ev.get("pmfs") is not None:
        transfer_delta, transfer_report = fit_keynumber_transfer(
            margin_ev.get("pmfs"), margin_ev.get("outcomes"),
            margin_ev.get("support"))
        # Disclosure always rides the margin spec (the MLB one_run_rebalance
        # record): admitted, refused or direction-failed — the verdict and
        # the per-bin sizing are visible either way.
        margin["keynumber_report"] = transfer_report
        if transfer_delta is not None:
            adjusted = np.clip(np.asarray(margin_ev["pmfs"], float)
                               + np.asarray(transfer_delta, float)[None, :],
                               0., None)
            row_total = adjusted.sum(axis=1, keepdims=True)
            collapsed = (~np.isfinite(row_total[:, 0])
                         | (row_total[:, 0] <= 0.))
            adjusted[collapsed] = np.asarray(margin_ev["pmfs"], float)[collapsed]
            row_total = adjusted.sum(axis=1, keepdims=True)
            fit_evidence["margin"] = dict(margin_ev,
                                          pmfs=adjusted / row_total)
    for target, kind in (("totals", "total"), ("margin", "margin")):
        ev = fit_evidence.get(kind) or None
        if ev is None:
            continue
        (totals if target == "totals" else margin).update(
            family.fit_lattice_correction(ev.get("pmfs"), ev.get("outcomes"),
                                          ev.get("support"), DISTRIBUTION_SPORT,
                                          target))
    if transfer_delta is not None:
        # Compose structural transfer + generic residual into the single
        # stored delta: family.compose_lattice_deltas' own contract (sum,
        # mass-proportional zero-sum, clamp, eps-zero, round). Both
        # components are far inside the LATTICE_DELTA_CAP fix bound, so the
        # uniform clamp is not binding here and no per-bin cap is needed.
        parts = [np.asarray(transfer_delta, float)]
        if "delta" in margin:
            parts.append(np.asarray(margin["delta"], float))
        weight = np.clip(np.asarray(margin_ev["pmfs"], float).mean(axis=0),
                         0., None)
        margin["delta"] = family.compose_lattice_deltas(parts, weight)
        margin["significant"] = True
        margin["delta_source"] = ("generic+keynumber_transfer"
                                  if len(parts) > 1 else "keynumber_transfer")
        fitted = np.asarray(margin_ev.get("support"))
        margin["support"] = [int(fitted[0]), int(fitted[-1])]
    return {"totals": totals, "margin": margin}


def _distribution_fit_maps(h, a, outcomes, lattice_evidence=None):
    parameters = family.fit_variance(h, a, outcomes, DISTRIBUTION_SPORT)
    if DISTRIBUTION_SPORT == "nba" and bool(getattr(config, "DISTRIBUTION_TOTAL_TILT_PACE", False)):
        # Adopted 2026-10-18 (V3a verdict: top-quintile over-230 gap
        # +0.0438 -> +0.0081, CRPS neutral). Totals pace-tilt location
        # correction rides the variance params; raw_cdf shifts the TOTAL
        # location only, so calibration refits against the tilted base.
        parameters.update(family.fit_tilt(h, a, outcomes, DISTRIBUTION_SPORT))
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
    if DISTRIBUTION_SPORT == "nba":
        # V3a tilt watch (adopted 2026-10-18): slope-sign guard on the
        # serving fit plus the top-quintile over-230 gap panel, so the bias
        # the tilt was adopted to fix stays watched every run. Absent tilt
        # block (flag off) still reports the gap; other sports skip this.
        tilt, gap = {}, {}
        serving_var = (bundle.get("maps") or {}).get("variance") or {}
        block = serving_var.get("total_tilt") or {}
        if block:
            slopes = []
            for origin in bundle.get("origins") or []:
                ob = ((origin.get("maps") or {}).get("variance") or {}).get("total_tilt") or {}
                if ob.get("slope") is not None:
                    slopes.append(float(ob["slope"]))
            tilt = {"slope": round(float(block["slope"]), 3),
                    "intercept": round(float(block["intercept"]), 3),
                    "sign_ok": bool(float(block["slope"]) > 0),
                    "origin_slopes": [round(s, 3) for s in slopes],
                    "origin_sign_flips": int(sum(1 for s in slopes if s <= 0))}
            if not tilt["sign_ok"]:
                logger.warning("totals tilt slope flipped non-positive (%.3f); "
                               "the pace-tilt correction is suspect", block["slope"])
        if "mu_h" in frame and "mu_a" in frame and "p_over_230" in frame:
            exp = frame.mu_h.to_numpy(float) + frame.mu_a.to_numpy(float)
            ok = np.isfinite(exp)
            if int(ok.sum()) >= 100:
                q80 = float(np.quantile(exp[ok], 0.8))
                hot = ok & (exp >= q80)
                if int(hot.sum()) >= 20:
                    p = np.clip(frame.p_over_230.to_numpy(float)[hot], 1e-7, 1 - 1e-7)
                    y = ((frame.home_score + frame.away_score).to_numpy(float)[hot] > 230.).astype(float)
                    gap = {"n": int(hot.sum()), "q80": round(q80, 2),
                           "predicted": round(float(p.mean()), 4),
                           "actual": round(float(y.mean()), 4),
                           "gap": round(float(y.mean() - p.mean()), 4)}
        result["tilt_watch"] = {"tilt": tilt, "top_quintile_over_230": gap}
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
