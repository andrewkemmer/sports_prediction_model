"""NHL split-skew Gaussian variance, causal CRPS calibration and Monte Carlo."""
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

try:
    from backend import gaussian_family as family
except ImportError:
    import gaussian_family as family
logger = logging.getLogger(__name__)
MONEYLINE_TREE_MEMBER = "lightgbm"
HOLDOUT_DAYS = 21
MC_DRAWS = family.MC_DRAWS
MC_SEED = family.MC_SEED
MC_SE_TARGET = family.MC_SE_TARGET

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
# Negative-binomial challenger (2026-10-17, NBA template): NB2 with log
# link vs the Poisson incumbent. Same tree matrix, same LightGBM
# hyper-params, same folds — the ONLY difference is the likelihood.
# Dispersion r is train-fold-estimated per side (variance = mu + mu^2/r).
# Trial-only until a walk-forward verdict clears the 2.5 gate.
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
        return NB_DISPERSION_MAX
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


# ---------------------------------------------------------------------------
# Conway-Maxwell-Poisson challenger (trial-only): CMP with mean-parameterized
# log link vs the Poisson incumbent. Same tree matrix, same LightGBM
# hyper-params, same folds — the ONLY difference is the likelihood.
# Motivation (2026-10-17): NB refused everywhere with dispersions pegged at
# caps — the signature of a shape parameter with nothing to grip in the
# overdispersion direction. CMP's nu goes BOTH ways (nu = 1 is Poisson,
# nu > 1 underdispersed, nu < 1 overdispersed), so it can express what NB
# structurally cannot. nu is train-fold moment-matched per side; the trees
# fit the log-mean with exact CMP moments (truncated-sum E[Y], Var[Y] —
# NHL means ~3 keep the support tiny, so exact beats approximate).
# Trial-only until a walk-forward verdict clears the 2.5 gate.
# ---------------------------------------------------------------------------
CMP_NU_MIN = 0.25
CMP_NU_MAX = 4.0
CMP_JMAX = 200
CMP_TOL = 1e-12


def _cmp_log_terms(lam: np.ndarray, nu: float, jmax: int = CMP_JMAX
                   ) -> np.ndarray:
    """log pmf-terms j*log(lam) - nu*lgamma(j+1), rows obs x cols j."""
    from scipy.special import gammaln
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-300)
    js = np.arange(jmax)
    return (js[None, :] * np.log(lam)[:, None]
            - nu * gammaln(js + 1)[None, :])


def cmp_moments(lam: np.ndarray, nu: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact CMP E[Y], Var[Y] by truncated normalized sum (log-space)."""
    lt = _cmp_log_terms(lam, nu)
    lt = lt - lt.max(axis=1, keepdims=True)
    w = np.exp(lt)
    w = w / w.sum(axis=1, keepdims=True)
    js = np.arange(lt.shape[1])
    mean = w @ js
    var = w @ (js ** 2) - mean ** 2
    return mean, np.maximum(var, 1e-12)


def estimate_cmp_nu(y: np.ndarray) -> float:
    """Moment-match CMP nu from train labels (Var ~= mu/nu), per side."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y) & (y >= 0)]
    if len(y) < 2:
        return 1.0
    m, v = float(y.mean()), float(y.var(ddof=1))
    if not np.isfinite(m) or m <= 0 or not np.isfinite(v) or v <= 0:
        return 1.0
    return float(np.clip(m / v, CMP_NU_MIN, CMP_NU_MAX))


def _cmp_lam_from_mu(mu: np.ndarray, nu: float) -> np.ndarray:
    """Huang (2017) mean-parametrization inverse: lam = (mu + c)^nu."""
    c = (nu - 1.0) / (2.0 * nu)
    return np.power(np.maximum(np.asarray(mu, dtype=float) + c, 1e-12), nu)


class ComPoissonObjective:
    """CMP NLL grad/hess w.r.t. the raw log-mean score (mean-parameterized).

    y_pred carries the full log-mean (init_score included, same contract
    as the NB leg): mu = exp(clip(y_pred)), lam solved from mu, E[Y] and
    Var[Y] exact by truncated sum. At nu = 1 the grad is exactly the
    Poisson grad -(y - mu) — pinned by test, not by trust.
    """

    def __init__(self, nu: float) -> None:
        nu = float(nu)
        if not np.isfinite(nu) or nu <= 0:
            raise ValueError("CMP nu must be positive")
        self.nu = float(np.clip(nu, CMP_NU_MIN, CMP_NU_MAX))

    def __call__(self, y_true, y_pred):
        y = np.asarray(y_true, dtype=float)
        mu = np.exp(np.clip(np.asarray(y_pred, dtype=float), *NB_RAW_CLIP))
        nu = self.nu
        lam = _cmp_lam_from_mu(mu, nu)
        exp_y, var_y = cmp_moments(lam, nu)
        k = nu * mu / np.maximum(mu + (nu - 1.0) / (2.0 * nu), 1e-12)
        grad = -(y - exp_y) * k
        hess = var_y * k ** 2
        return grad, np.maximum(hess, 1e-5)


def _make_reg_cmp(nu: float):
    from lightgbm import LGBMRegressor
    params = dict(config.LIGHTGBM_REG_PARAMS)
    params["objective"] = ComPoissonObjective(nu)
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
    """Two same-contract LightGBM Poisson regressors, one per score side.

    ``crps_location`` is the opt-in objective challenger NBA carries (direct
    CRPS location trees; NOT the production default -- the full 2026-10-11
    NBA walk underperformed). Default stays Poisson.
    """

    def __init__(self, declare_categoricals: bool = True,
                 crps_location: bool = False, nb: bool = False,
                 cmp: bool = False) -> None:
        self.crps_location = bool(crps_location)
        self.nb = bool(nb)
        self.cmp = bool(cmp)
        if sum((self.crps_location, self.nb, self.cmp)) > 1:
            raise ValueError("ScoreRegressor takes one challenger at a time")
        self.home_model = _make_reg(crps=self.crps_location)
        self.away_model = _make_reg(crps=self.crps_location)
        # Categorical routing MUST match the OOF walk exactly (train/serve
        # skew): the walk passes the team-ID pair via categorical_feature by
        # name, so the production refit declares the same columns by default.
        self._declare_categoricals = bool(declare_categoricals)
        self.feature_columns: list[str] = []
        self.margin_center = 0.
        self.total_center = 0.
        self.nb_log_mean: tuple[float, float] | None = None
        self.nb_dispersion: tuple[float, float] | None = None
        self.cmp_log_mean: tuple[float, float] | None = None
        self.cmp_nu: tuple[float, float] | None = None

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        # Resolved through the binary moneyline's OWN member_matrix helper —
        # the NHL equivalent of MLB's
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

    def _fit_kwargs(self, X: pd.DataFrame) -> dict:
        return ({"categorical_feature": list(config.TREE_CATEGORICAL_COLS)}
                if self._declare_categoricals
                and config.TREE_CATEGORICAL_COLS
                and all(c in X.columns for c in config.TREE_CATEGORICAL_COLS)
                else {})

    def fit(self, df: pd.DataFrame) -> "ScoreRegressor":
        X = self._matrix(df)
        h = pd.to_numeric(df["home_score"], errors="coerce")
        a = pd.to_numeric(df["away_score"], errors="coerce")
        valid = h.notna() & a.notna()
        if valid.sum() < 2:
            raise ValueError("score regressor requires at least two settled games")
        X, h, a = X.loc[valid], h.loc[valid], a.loc[valid]
        fit_kwargs = self._fit_kwargs(X)
        if self.crps_location:
            margin, total = h - a, h + a
            self.margin_center = float(margin.mean())
            self.total_center = float(total.mean())
            self.home_model = _make_reg(max(float(margin.std(ddof=0)), 1.),
                                        crps=True)
            self.away_model = _make_reg(max(float(total.std(ddof=0)), 1.),
                                        crps=True)
            self.home_model.fit(X, margin,
                                init_score=np.full(len(X), self.margin_center),
                                **fit_kwargs)
            self.away_model.fit(X, total,
                                init_score=np.full(len(X), self.total_center),
                                **fit_kwargs)
        elif self.nb:
            rs, lms = [], []
            for side, y in (("home", h.to_numpy(dtype=float)),
                            ("away", a.to_numpy(dtype=float))):
                r = estimate_nb_dispersion(y)
                lm = float(np.log(max(np.mean(y), 1e-6)))
                model = _make_reg_nb(r)
                model.fit(X, y, init_score=np.full(len(X), lm),
                          **fit_kwargs)
                if side == "home":
                    self.home_model = model
                else:
                    self.away_model = model
                rs.append(r)
                lms.append(lm)
            self.nb_dispersion = (rs[0], rs[1])
            self.nb_log_mean = (lms[0], lms[1])
        elif self.cmp:
            nus, lms = [], []
            for side, y in (("home", h.to_numpy(dtype=float)),
                            ("away", a.to_numpy(dtype=float))):
                nu = estimate_cmp_nu(y)
                lm = float(np.log(max(np.mean(y), 1e-6)))
                model = _make_reg_cmp(nu)
                model.fit(X, y, init_score=np.full(len(X), lm),
                          **fit_kwargs)
                if side == "home":
                    self.home_model = model
                else:
                    self.away_model = model
                nus.append(nu)
                lms.append(lm)
            self.cmp_nu = (nus[0], nus[1])
            self.cmp_log_mean = (lms[0], lms[1])
        else:
            self.home_model.fit(X, h, **fit_kwargs)
            self.away_model.fit(X, a, **fit_kwargs)
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
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
        if self.cmp:
            # Mean-parameterized by construction: served mus are the fitted
            # log-means, exactly like the NB leg (the dispersion only
            # steered training gradients). CMP-CDF pricing is step 2, gated
            # on this training-gate passing first.
            assert self.cmp_log_mean is not None
            lo, hi = NB_RAW_CLIP
            mh = np.exp(np.clip(
                np.asarray(self.home_model.predict(X), dtype=float)
                + self.cmp_log_mean[0], lo, hi))
            ma = np.exp(np.clip(
                np.asarray(self.away_model.predict(X), dtype=float)
                + self.cmp_log_mean[1], lo, hi))
            return (np.clip(mh, 1e-6, None), np.clip(ma, 1e-6, None))
        return (
            np.clip(self.home_model.predict(X), 1e-6, None),
            np.clip(self.away_model.predict(X), 1e-6, None),
        )


def pearson_poisson_adequacy(y: np.ndarray, mu: np.ndarray) -> float:
    """Pearson chi-square / df. ≈1 → Poisson variance is adequate; >1 means
    over-dispersion the Gaussian variance layer learns (MLB run_engine fit probe)."""
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


def feat_mod_tree(df: pd.DataFrame) -> pd.DataFrame:
    """The moneyline tree view, without the member wrapper (fallback path)."""
    return ml_mod.member_matrix(MONEYLINE_TREE_MEMBER, df)


def run_engine_feature_cols(reg: "ScoreRegressor | None" = None) -> list[str]:
    """Numeric feature view of the DISTRIBUTION model (Totals & Run Lines).

    The monitoring view is resolved from the FITTED regressor's own
    ``feature_columns`` -- literally the matrix ``ScoreRegressor`` consumed --
    falling back to the member-matrix contract it is built from when no model
    is fitted yet. The team-ID categoricals are excluded: PSI and coverage are
    numeric-only views (MLB's run-engine view excludes them too -- they carry
    no weight row of their own). This is what the run-engine drift/coverage
    artifacts report so the dashboard measures the distribution model instead
    of the binary moneyline contract.
    """
    cols = list(getattr(reg, "feature_columns", None) or [])
    if not cols:
        cols = list(config.active_moneyline_feature_cols())
    return [c for c in cols if c not in config.TREE_CATEGORICAL_COLS]


def distribution_feature_weights(
        reg: "ScoreRegressor | None") -> dict[str, float] | None:
    """Per-feature MODEL WEIGHT (%) of the distribution model.

    Pooled feature importances of the two per-side Poisson LightGBM fits
    (home/away), normalized to sum to 100. The team-ID categoricals are
    excluded from BOTH the map and the denominator (they carry no drift/
    coverage row, and the panel's caption promises the reported weights sum
    to 100 across the view it shows). None when the regressor is unfitted or
    carries no LightGBM importances: the drift column then reports no weight
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


def _grid_key(prefix: str, value: float | int) -> str:
    s = str(float(value))
    if s.endswith(".0"):
        s = s[:-2]
    return f"{prefix}_{s.replace('-', 'm').replace('.', '_')}"


def feature_contract(df: pd.DataFrame | None = None) -> dict[str, Any]:
    """MLB-shaped record of the feature contract the run line actually consumes.

    Mirrors ``run_engine.run_oof``'s ``summary["feature_contract"]`` so the two
    engines publish the same audit trail. When ``df`` is supplied the contract
    is resolved from the real moneyline matrix for that frame and the record
    reports the FITTED width; the ``active_moneyline_feature_cols`` count is
    reported alongside it so a frame that silently dropped a contract column is
    visible instead of silent.
    """
    active = list(config.active_moneyline_feature_cols())
    contract: dict[str, Any] = {
        "mode": "strict_active_moneyline",
        "n_features": len(active),
        "feature_cols": active,
        "tree_categorical_cols": list(config.TREE_CATEGORICAL_COLS),
        "resolved_via": (f"moneyline.member_matrix({MONEYLINE_TREE_MEMBER!r})"),
    }
    if df is not None:
        fitted = list(ml_mod.member_matrix(MONEYLINE_TREE_MEMBER, df).columns)
        contract["n_features_fitted"] = len(fitted)
        contract["fitted_cols"] = fitted
        if len(fitted) < len(active) + len(config.TREE_CATEGORICAL_COLS):
            missing = [c for c in active if c not in fitted]
            logger.warning(
                "run-line feature contract: %d active moneyline feature(s) "
                "absent from this frame and therefore NOT fitted: %s",
                len(missing), missing)
    return contract


def walk_forward_oof(game_df: pd.DataFrame, date_col: str = "gameday",
                     progress_every: int = 25,
                     fold_list: list | None = None) -> dict:
    """Fit two LightGBM Poisson models on shared walk-forward folds.

    ``fold_list`` is the SAME expanding walk-forward fold list the binary
    moneyline walks (master_pipeline builds it once and hands the identical
    object to both), so the run line's OOF periods are the moneyline's OOF
    periods by construction — the MLB structural requirement that expected-
    scoring folds never drift from the moneyline fold geometry.
    """
    df = folds_mod.canonical_sort(game_df, date_col)
    fold_list = fold_list if fold_list is not None else folds_mod.make_folds(df, date_col=date_col)
    parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    n_folds = len(fold_list)
    # This walk-forward logged ONLY on failure, so a healthy 13s phase and a
    # crash were indistinguishable in the run log. Same checkpoint cadence as
    # moneyline.walk_forward_oof (see folds.progress_checkpoints).
    announce = set(folds_mod.progress_checkpoints(n_folds, progress_every))
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
        if (fold.fold_id + 1) in announce:
            logger.info("dist OOF fold %d/%d", fold.fold_id + 1, n_folds)
    oof = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    logger.info("dist OOF complete: %d fold(s), %d scored row(s)",
                n_folds, len(oof))
    if len(oof):
        oof["resid_margin"] = oof["margin"] - (oof["mu_h"] - oof["mu_a"])
        oof["resid_total"] = oof["total"] - (oof["mu_h"] + oof["mu_a"])
    return {"oof": oof, "fold_table": pd.DataFrame(fold_rows),
            "feature_contract": feature_contract(df),
            "n_folds": n_folds}


def walk_forward_oof_nb(game_df: pd.DataFrame, date_col: str = "gameday",
                        progress_every: int = 25,
                        fold_list: list | None = None) -> dict:
    """NB-challenger twin of walk_forward_oof (trial-only, 2026-10-17)."""
    df = folds_mod.canonical_sort(game_df, date_col)
    fold_list = fold_list if fold_list is not None else folds_mod.make_folds(df, date_col=date_col)
    parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    n_folds = len(fold_list)
    announce = set(folds_mod.progress_checkpoints(n_folds, progress_every))
    for fold in fold_list:
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
            logger.warning("NB dist OOF fold %s failed: %s", fold.fold_id, exc)
            mu_h = np.full(len(val), np.nan)
            mu_a = np.full(len(val), np.nan)
            rh, ra = float("nan"), float("nan")
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
                          "n_train": int(len(train)), "n_val": int(len(val)),
                          "r_home": rh, "r_away": ra})
        if (fold.fold_id + 1) in announce:
            logger.info("NB dist OOF fold %d/%d", fold.fold_id + 1, n_folds)
    oof = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(oof):
        oof["resid_margin"] = oof["margin"] - (oof["mu_h"] - oof["mu_a"])
        oof["resid_total"] = oof["total"] - (oof["mu_h"] + oof["mu_a"])
    return {"oof": oof, "fold_table": pd.DataFrame(fold_rows),
            "n_folds": n_folds}


def walk_forward_oof_cmp(game_df: pd.DataFrame, date_col: str = "gameday",
                         progress_every: int = 25,
                         fold_list: list | None = None) -> dict:
    """CMP-challenger twin of walk_forward_oof (trial-only)."""
    df = folds_mod.canonical_sort(game_df, date_col)
    fold_list = fold_list if fold_list is not None else folds_mod.make_folds(df, date_col=date_col)
    parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    n_folds = len(fold_list)
    announce = set(folds_mod.progress_checkpoints(n_folds, progress_every))
    for fold in fold_list:
        train, val = df.loc[fold.train_idx], df.loc[fold.val_idx]
        train_dates = pd.to_datetime(train[date_col], errors="raise").dt.normalize()
        val_dates = pd.to_datetime(val[date_col], errors="raise").dt.normalize()
        if train_dates.max() >= val_dates.min():
            raise ValueError("distribution training must precede validation dates")
        try:
            reg = ScoreRegressor(cmp=True).fit(train)
            mu_h, mu_a = reg.predict(val)
            nh, na = reg.cmp_nu or (float("nan"), float("nan"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("CMP dist OOF fold %s failed: %s", fold.fold_id, exc)
            mu_h = np.full(len(val), np.nan)
            mu_a = np.full(len(val), np.nan)
            nh, na = float("nan"), float("nan")
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
                          "n_train": int(len(train)), "n_val": int(len(val)),
                          "nu_home": nh, "nu_away": na})
        if (fold.fold_id + 1) in announce:
            logger.info("CMP dist OOF fold %d/%d", fold.fold_id + 1, n_folds)
    oof = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(oof):
        oof["resid_margin"] = oof["margin"] - (oof["mu_h"] - oof["mu_a"])
        oof["resid_total"] = oof["total"] - (oof["mu_h"] + oof["mu_a"])
    return {"oof": oof, "fold_table": pd.DataFrame(fold_rows),
            "n_folds": n_folds}


def fit_final(game_df: pd.DataFrame) -> ScoreRegressor:
    return ScoreRegressor().fit(game_df)


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


def _elasticnet_full_view(df: pd.DataFrame) -> pd.DataFrame:
    """Linear view plus raw per-side columns (arm C trial only).

    The moneyline linear view deliberately drops raw sides; this challenger
    view adds back every raw per-side column present in the frame so the
    elastic net can learn side-strength logic, not only gaps. Team-ID
    categoricals are excluded (arbitrary IDs as linear ordinals would be a
    bug). Standardization/NaN handling ride the existing train-fold
    preprocessor, which is column-generic. Default-off; see
    config.SCORE_EN_ELASTICNET_FULL_VIEW (NBA precedent, 2026-10-17).
    """
    base = ml_mod.member_matrix("elasticnet", df)
    extra = [c for c in config.RAW_PER_SIDE_COLS
             if c in df.columns and c not in base.columns
             and c not in config.TREE_CATEGORICAL_COLS]
    if not extra:
        return base
    return pd.concat([base, df[extra]], axis=1)


class ScoreEnsembleRegressor:
    """PIT per-side score-mean ensemble; blend weights set from prior OOF."""

    SIDES = ("home_score", "away_score")

    def __init__(self) -> None:
        self.members: dict[str, dict[str, tuple[Any, Any]]] = {}
        self.columns: dict[str, list[str]] = {}
        self.weights: dict[str, float] = dict(config.SCORE_ENSEMBLE_WEIGHTS)

    def set_weights(self, weights: dict[str, float]) -> "ScoreEnsembleRegressor":
        self.weights = dict(weights)
        return self

    def _matrix(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        # Same matrices the binary moneyline members consume (structural
        # property, not convention): tree views keep the categorical team-ID
        # pair, the elastic net takes the linear view.
        if name == "elasticnet" and config.SCORE_EN_ELASTICNET_FULL_VIEW:
            X = _elasticnet_full_view(df)
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
                if name == "lightgbm":
                    categorical = [c for c in config.TREE_CATEGORICAL_COLS
                                   if c in getattr(X_fit, "columns", [])]
                    if categorical:
                        fit_kwargs["categorical_feature"] = categorical
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



def apply_distribution(df: pd.DataFrame, bundle: dict | None = None,
                       meta_out: dict | None = None, n_draws=MC_DRAWS,
                       seed=MC_SEED) -> pd.DataFrame:
    """Sample the challenger CDF without consulting any outcomes."""
    return _distribution_price(df, (bundle or {}).get("maps", {}),
                               n_draws=n_draws, seed=seed, meta_out=meta_out)


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


DISTRIBUTION_SPORT = "nhl"
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
        if kind == "total" and DISTRIBUTION_SPORT != "mlb":
            # Full-support totals PMF (2026-10-14): the diagnostics
            # distribution chart draws P(total = k) only where the artifact
            # ships the totals-push family, and the market grid alone left the
            # sub-grid band with NO modeled line (NHL totals <= 3 are 13.3% of
            # finals; the 2026-10-14 gap the dashboard review flagged). Ship
            # the integer PMF mass over the priced support wherever the model
            # places ANY mass — the exact same sampled CDF the grid columns
            # come from, never a refit — so the shipped distribution spans the
            # settlement range. Grid lines keep their priced values; the
            # margin/spread push namespace is untouched.
            for idx in np.flatnonzero(mass.any(axis=0)):
                updates.setdefault(_grid_key("p_push_total", int(support[idx])),
                                   mass[:, idx])
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


# NHL empty-net |M|=2 -> |M|=3 structural transfer (declared 2026-10-16
# BEFORE fitting; preregistered as family.LATTICE_PREREGISTERED
# ("nhl", "margin", "empty_net_2_to_3")). Trailing teams pull the goalie,
# so 2-goal leads finish as 3-goal finals more often than the smooth
# family allows. The shipped BASE margin residual (2026-10-14/15 walk,
# n=2707 — run_diagnostics/nhl_lattice.json, nhl_struct_check.log) puts
# the worst UNDER-predictions at +3/+0.0363 and -3/+0.0295 while +2/-2
# OVER-predict by -0.0244/-0.0097 — the mass the actuals want at |M|=3
# is sitting at |M|=2. The generic localized residual could not trade it
# (pooled MAE worsened 0.000793 -> 0.000871, held-out gate z -0.619,
# 0/60 margin origins admitted): its mass-proportional zero-sum removal
# is smeared across every bin, so the collateral guard vetoes the one
# trade that fixes two bins at once. This mechanism prices that trade as
# a STRUCTURAL transfer: mass is taken ONLY from the same-sign |M|=2
# donor, and every quantity is sized dynamically from prior-OOF actuals
# on both ends (see fit_empty_net_transfer) — nothing here is a hard-coded
# amplitude. The transfer estimator lives in this sport-specific module,
# NOT in the shared gaussian_family.py, whose four backend copies must
# stay byte-identical (test_shared_gaussian_family_copies_are_byte_identical).
EMPTY_NET_TRANSFER_PAIRS = ((2, 3), (-2, -3))   # (donor, recipient) margins
# Structural demand relief (the NFL LATTICE_SPIKE_RELIEF_BOOST precedent,
# same value and same measured failure): the mass-proportional prior
# (tau = PRIOR_FRAC * local model mass) OVER-SHRINKS a rule-derived spike
# — on synthetic evidence shaped like the shipped residual the unboosted
# prior kept the transfer demand at ~1/3 of the +/-3 residual it must
# absorb, the exact "spikes underfit the residual" complaint (NFL measured
# the same ~0.36 shrink on their +/-3 bins). The recipient's prior scale
# is therefore multiplied by this factor BEFORE the same per-bin
# noise-floor shrinkage; gates, the donor supply limit and the generic
# path are untouched — relief changes only the proposed demand, and an
# over-deep demand still loses to its supply cap or fails its own gate.
EMPTY_NET_DEMAND_RELIEF = 3.
# Magnitude caps REMOVED (2026-10-18, owner decision): the admitted
# transfer ships at its full fitted size. The transfer itself never
# exceeds the donor's observed excess, and every leg still clears its
# own one-sided held-out gate.


def fit_empty_net_transfer(pmfs, outcomes, support,
                           pairs=EMPTY_NET_TRANSFER_PAIRS,
                           bar=family.LATTICE_STRUCTURAL_Z,
                           min_history=family.MIN_HISTORY):
    """Preregistered empty-net |M|=2 -> |M|=3 mass transfer (NHL margin).

    For each (donor, recipient) pair the transfer t >= 0 moves model mass
    from the donor bin to the recipient bin (+t at recipient, -t at
    donor) — a pure, exactly zero-sum shift: mass moves, none is made.
    Every quantity is DYNAMIC, sized from prior-OOF actuals on both ends:

      demand = the SE-shrunk train residual at the recipient bin
        (family.lattice_local_deltas under the STRUCTURAL prior scale —
        relief-weighted model mass, see EMPTY_NET_DEMAND_RELIEF) — the
        under-fit depth against the bin's own binomial noise floor, never
        a constant;
      supply = clip(-train residual at the donor, 0, donor model mass) —
        the donor's OBSERVED excess. Actuals showing no excess at the
        donor (residual >= 0) supply exact zero, so mass is never taken
        from a bin the actuals do not over-predict;
      t = min(demand, supply) — no magnitude cap.

    Direction checks (recipient under-predicted, donor over-predicted)
    run on TRAIN evidence only and are enforced per pair. Admission is
    then ONE one-sided held-out gate — z >= bar (the pre-registered
    structural bar; direction: the COMPOSED two-pair shift improves the
    squared-probability loss) — scored on the time-ordered held-out tail
    of the same evidence, on the exact vector that would price if
    admitted: the preregistration declares ONE mechanism
    (empty_net_2_to_3) whose two pairs are its symmetric legs, and the
    MLB one-run rebalance gates its single multi-donor vector the same
    way. Measured on the 20261016 delivered walk
    (run_diagnostics/nhl17_transfer_gate_policy_probe.py): per-pair
    gating at this same bar admitted 19/61 origins for a pooled
    rows*transfer visibility of 0.0073; the joint gate admits 30/61 for
    0.0132 — two half-evidence tests each paid the full bar, so the
    pooled chart could not see the correction. The per-pair z stays in
    the report as disclosure; a refused joint gate refuses the whole
    mechanism — no partial transfer ships.

    Returns (delta | None, report): the composed transfer delta
    (zero-sum, rounded — the generic post pipeline) or None when the
    joint gate refuses, plus the per-pair
    verdict table (demand/supply/transfer/z per pair) and the joint
    report["gate"] block for disclosure.
    """
    report: dict = {"estimator": "empty_net_2_to_3_transfer",
                     "sport": DISTRIBUTION_SPORT,
                     "mechanism": "empty_net_2_to_3",
                     "bar": float(bar), "pairs": []}
    support = np.asarray(support)
    pmfs = np.empty((0, len(support))) if pmfs is None else np.asarray(pmfs, float)
    outcomes = np.empty(0) if outcomes is None else np.asarray(outcomes, float)
    if pmfs.ndim != 2 or pmfs.shape[1] != len(support) or len(pmfs) != len(outcomes):
        raise ValueError("empty-net transfer evidence requires aligned pmfs and outcomes")
    inside = ((outcomes >= support[0]) & (outcomes <= support[-1])
              & np.isfinite(outcomes))
    if inside.sum() < int(min_history):
        report["reason"] = "insufficient_history"
        return None, report
    n_test = int(family.LATTICE_TEST_FRACTION * len(outcomes))
    if n_test < family.LATTICE_MIN_TEST or len(outcomes) - n_test < int(min_history):
        report["reason"] = "insufficient_held_out_evidence"
        return None, report
    train = np.flatnonzero(inside[:len(outcomes) - n_test])
    # Absolute tail positions (the fit_ot_home_share discipline): the gate
    # must score the time-ordered held-out tail itself, never rows the
    # demand was fitted on.
    hold = np.arange(len(outcomes) - n_test, len(outcomes))
    hold = hold[inside[hold]]
    if len(train) < int(min_history) or len(hold) < family.LATTICE_MIN_TEST:
        report["reason"] = "insufficient_held_out_evidence"
        return None, report
    n_train = max(int(len(train)), 1)
    pbar = np.clip(pmfs[train].mean(axis=0), 0., 1.)
    se = np.sqrt(pbar * (1. - pbar) / n_train)
    obs = np.bincount(np.searchsorted(support, outcomes[train]),
                      minlength=len(support)).astype(float)
    obs /= max(int(len(train)), 1)
    resid = obs - pmfs[train].mean(axis=0)
    indices_hold = np.searchsorted(support, outcomes[hold])
    sized = []        # direction-passing pairs: (i3, i2, transfer, row)
    for donor, recipient in pairs:
        i2 = int(np.searchsorted(support, donor))
        i3 = int(np.searchsorted(support, recipient))
        row: dict = {"donor": int(donor), "recipient": int(recipient)}
        if not (0 <= i2 < len(support) and 0 <= i3 < len(support)
                and int(support[i2]) == donor and int(support[i3]) == recipient):
            row.update(admitted=False, reason="support_lacks_pair")
            report["pairs"].append(row)
            continue
        # Demand = SE-shrunk recipient residual under the STRUCTURAL prior
        # scale (relief-weighted mass; lattice_local_deltas uses the weight
        # only as the prior scale, so the boosted call stays one shared
        # shrinkage code path).
        demand = float(family.lattice_local_deltas(
            EMPTY_NET_DEMAND_RELIEF * pbar[i3:i3 + 1],
            resid[i3:i3 + 1], se[i3:i3 + 1])[0])
        supply = float(np.clip(-resid[i2], 0., pbar[i2]))
        row["demand"] = round(demand, 6)
        row["supply"] = round(supply, 6)
        if not demand > 0.:
            row.update(admitted=False, reason="recipient_not_underpredicted")
            report["pairs"].append(row)
            continue
        if supply <= 0.:
            row.update(admitted=False, reason="no_donor_excess")
            report["pairs"].append(row)
            continue
        transfer = float(min(demand, supply))
        row["transfer"] = round(transfer, 6)
        trial = np.zeros(len(support))
        trial[i3], trial[i2] = transfer, -transfer
        d = (-2. * trial[indices_hold] + 2. * (pmfs[hold] @ trial)
             + float(trial @ trial))
        tse = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
        z = float(-d.mean() / tse) if tse > 0. else 0.
        # Single-pair diagnostic z; admission is the joint gate below.
        row["z"] = round(z, 3)
        sized.append((i3, i2, transfer, row))
        report["pairs"].append(row)
    report["n_test"] = int(len(hold))
    if not sized:
        report["reason"] = "no_transfer_admitted"
        return None, report
    # ONE held-out gate for the composed mechanism delta — the MLB
    # rebalance precedent (fit_one_run_spread_rebalance gates its single
    # multi-donor vector once) applied to the exact vector that would
    # price. Bar, one-sided direction and time-ordered tail unchanged.
    joint = np.zeros(len(support))
    for i3, i2, transfer, _row in sized:
        joint[i3] += transfer
        joint[i2] -= transfer
    d = (-2. * joint[indices_hold] + 2. * (pmfs[hold] @ joint)
         + float(joint @ joint))
    tse = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
    z = float(-d.mean() / tse) if tse > 0 else 0.
    admitted = d.mean() < 0. and z >= bar
    report["gate"] = {"scope": "joint_pairs_composed",
                      "z": round(z, 3),
                      "n_test": int(len(hold)),
                      "bar": float(bar),
                      "bar_applied": ("generic" if z >= family.LATTICE_GATE_Z
                                      else "structural"),
                      "direction": "one_sided_improvement",
                      "mean_negative": bool(d.mean() < 0.),
                      "admitted": bool(admitted)}
    if not admitted:
        for _i3, _i2, _t, row in sized:
            row["admitted"] = False
            row["reason"] = "joint_gate_refused"
        report["reason"] = "no_transfer_admitted"
        return None, report
    for _i3, _i2, _t, row in sized:
        row["admitted"] = True
    delta = np.zeros(len(support))
    for i3, i2, transfer, _row in sized:
        delta[i3] += transfer
        delta[i2] -= transfer
    delta = family._zero_sum_mass_proportional(delta, pbar)
    delta[np.abs(delta) < family.LATTICE_DELTA_EPS] = 0.
    delta = np.round(delta, family.LATTICE_ROUND)
    report["significant"] = True
    report["delta_l1"] = round(float(np.abs(delta).sum()), 6)
    return [float(v) for v in delta], report


def _fit_lattice_maps(outcomes, parameters, evidence):
    """Independent per-target lattice fits — totals AND margin.

    Structural terms ride the VARIANCE parameters (NHL totals OT/SO parity
    alpha; the no-tie +/-1 split weight — NHL OT/SO home-win share, MLB
    walk-off home-win share — estimated causally from prior OOF one-goal
    finals). The generic sawtooth residual is fit from the walk's
    own BASE-model evidence — pmfs of strictly-prior OOF rows, each priced
    with its own origin's maps — per target, behind the held-out
    significance gate. The preregistered empty-net |M|=2 -> |M|=3 transfer
    is fitted FIRST on the margin evidence (each pair sized dynamically
    from actuals behind one joint held-out gate on the composed two-pair
    delta at the structural bar); the
    generic residual then runs on transfer-adjusted pmfs and both
    components price as one composed delta with no magnitude clamp.
    Every value is plain JSON so stored
    bundles reprice byte-identically.
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
    # Preregistered empty-net transfer (family.LATTICE_PREREGISTERED
    # nhl/margin/empty_net_2_to_3): the |M|=2 -> |M|=3 trade is fitted from
    # the margin evidence FIRST — dynamically sized from actuals on both
    # ends and gated as one composed vector one-sided at the structural
    # bar — so the
    # generic localized residual never has to rediscover (and currently
    # cannot make) the trade its mass-proportional removal forbids. The
    # generic fit then runs on transfer-adjusted pmfs and is admitted only
    # on its own generic gate; the pair verdicts ride
    # margin["empty_net_report"]. The +/-1 split and no-tie rule are
    # untouched, and an unconfirmed transfer is a byte-identical no-op.
    fit_evidence = dict(evidence or {})
    transfer_delta = None
    margin_ev = (evidence or {}).get("margin") or {}
    if margin_ev.get("pmfs") is not None:
        transfer_delta, transfer_report = fit_empty_net_transfer(
            margin_ev.get("pmfs"), margin_ev.get("outcomes"),
            margin_ev.get("support"))
        if transfer_delta is not None:
            margin["empty_net_report"] = transfer_report
            adjusted = np.clip(np.asarray(margin_ev["pmfs"], float)
                               + np.asarray(transfer_delta, float)[None, :], 0., None)
            row_total = adjusted.sum(axis=1, keepdims=True)
            collapsed = ~np.isfinite(row_total[:, 0]) | (row_total[:, 0] <= 0.)
            adjusted[collapsed] = np.asarray(margin_ev["pmfs"], float)[collapsed]
            row_total = adjusted.sum(axis=1, keepdims=True)
            fit_evidence["margin"] = dict(margin_ev,
                                          pmfs=adjusted / row_total)
        elif transfer_report.get("pairs"):
            margin["empty_net_report"] = transfer_report
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
        # stored delta (same post pipeline each fit applies alone;
        # family.compose_lattice_deltas' own contract — sum,
        # mass-proportional zero-sum, eps-zero, round — done here with no
        # magnitude clamp (caps removed 2026-10-18). Generic-only metrics
        # stay as fitted; delta_l1/delta_max describe the COMPOSED vector
        # that prices.
        parts = [np.asarray(transfer_delta, float)]
        if "delta" in margin:
            parts.append(np.asarray(margin["delta"], float))
        weight = np.clip(np.asarray(margin_ev["pmfs"], float).mean(axis=0),
                         0., None)
        composed = family._zero_sum_mass_proportional(np.sum(parts, axis=0),
                                                       weight)
        _msup = np.asarray(margin_ev.get("support"))
        composed[np.abs(composed) < family.LATTICE_DELTA_EPS] = 0.
        margin["delta"] = [float(v)
                           for v in np.round(composed, family.LATTICE_ROUND)]
        margin["delta_l1"] = round(float(np.abs(margin["delta"]).sum()), 6)
        margin["delta_max"] = round(float(np.abs(margin["delta"]).max()), 6)
        margin["significant"] = True
        # Fitted-lattice bounds for the composed delta (same union lattice
        # both components were fitted on) — see family._realign_lattice_delta.
        margin["support"] = [int(_msup[0]), int(_msup[-1])]
    return {"totals": totals, "margin": margin}


def _distribution_fit_maps(h, a, outcomes, lattice_evidence=None):
    parameters = family.fit_variance(h, a, outcomes, DISTRIBUTION_SPORT)
    if DISTRIBUTION_SPORT in ("nba", "nfl") and bool(getattr(config, "DISTRIBUTION_TOTAL_TILT_PACE", False)):
        # Adopted 2026-10-18 (V3a verdict: top-quintile over-230 gap
        # +0.0438 -> +0.0081, CRPS neutral). Totals pace-tilt location
        # correction rides the variance params; raw_cdf shifts the TOTAL
        # location only, so calibration refits against the tilted base.
        parameters.update(family.fit_tilt(h, a, outcomes, DISTRIBUTION_SPORT))
    if DISTRIBUTION_SPORT == "nhl" and bool(getattr(config, "DISTRIBUTION_TOTAL_SHRINK_NHL", False)):
        # 2026-10-19: totals slope-shrinkage (measured pooled-OOF slope of
        # actual totals on mu_total ~0.14 at corr ~0.02 — the spread carries
        # no signal and the fair-line card ran 56.2% predicted vs 50.8%
        # actual). Rides the variance params like the NBA/NFL tilt;
        # raw_cdf shrinks the TOTAL location only, so margin / run-line /
        # derived-ML price byte-identical and calibration refits against
        # the shrunk base.
        parameters.update(family.fit_total_shrinkage(h, a, outcomes, DISTRIBUTION_SPORT))
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
    if DISTRIBUTION_SPORT in ("nba", "nfl"):
        # V3a tilt watch (NBA adopted 2026-10-18, NFL 2026-10-18): slope-sign
        # guard on the serving fit plus the top-quintile over-line gap panel
        # (over-230 NBA, over-47 NFL), so the bias the tilt was adopted to
        # fix stays watched every run. Absent tilt block (flag off) still
        # reports the gap; other sports skip this.
        _hot_line = 230. if DISTRIBUTION_SPORT == "nba" else 47.
        _hot_key = f"p_over_{int(_hot_line)}"
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
        if "mu_h" in frame and "mu_a" in frame and _hot_key in frame:
            exp = frame.mu_h.to_numpy(float) + frame.mu_a.to_numpy(float)
            ok = np.isfinite(exp)
            if int(ok.sum()) >= 100:
                q80 = float(np.quantile(exp[ok], 0.8))
                hot = ok & (exp >= q80)
                if int(hot.sum()) >= 20:
                    p = np.clip(frame[_hot_key].to_numpy(float)[hot], 1e-7, 1 - 1e-7)
                    y = ((frame.home_score + frame.away_score).to_numpy(float)[hot] > _hot_line).astype(float)
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
