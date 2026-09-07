"""Production NFL run-line / margin and totals distributional model.

Architecture (spec sections 21-23): a JOINT score distribution.

  mu_h, mu_a  <- gradient-boosted regressions on point-in-time features
  margin      ~ Skellam-like discrete normal on mu_margin = mu_h - mu_a,
                 sigma_margin estimated from OOF residuals
  total       ~ discrete normal on mu_total = mu_h + mu_a, sigma_total
                 estimated from OOF residuals

Everything derives from ONE (mu_h, mu_a) pair per game, so the margin and
total distributions are mathematically coherent: moneyline-derived,
fair spread, fair total, every cover/over/push probability, and the mu
quartet all come from the same fitted score distribution.

Calibration preserves distributional coherence: sigma (and a documented
mean-bias scalar) is calibrated ONCE on pooled OOF residuals — never per
line — so home_cover + push + away_cover = 1 and over + push + under = 1
hold exactly by PMF construction, and adjacent lines stay consistent.
"""
from __future__ import annotations

import logging

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


# ---------------------------------------------------------------------------
# Discrete normal PMF helpers (integer support, NFL point mechanics)
# ---------------------------------------------------------------------------
def discrete_normal_pmf(mu: float, sigma: float, support: np.ndarray) -> np.ndarray:
    """P(X = k) for integer support k, X ~ Normal(mu, sigma), normalized."""
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
        return np.full(len(support), np.nan)
    z = (support - mu) / sigma
    pdf = np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))
    pmf = pdf / pdf.sum()
    return pmf


def margin_cdf_above(pmf: np.ndarray, support: np.ndarray, L: float) -> float:
    """P(margin > L) for a real-valued threshold L over integer support.

    A margin m covers L iff m > L, i.e. m >= floor(L) + 1 for non-integer L,
    and m >= L + 1 for integer L. (NFL margins are integers; the contract
    P(m > L) is honored exactly.)"""
    if L == int(L):
        thresh = int(L) + 1
    else:
        thresh = int(np.floor(L)) + 1
    idx = support >= thresh
    return float(pmf[idx].sum()) if np.isfinite(pmf).all() else np.nan


def margin_pmf_at(pmf: np.ndarray, support: np.ndarray, L: float) -> float:
    """P(margin == L) — zero for non-integer L, PMF mass at int(L) otherwise."""
    if L != int(L):
        return 0.0
    hit = support == int(L)
    return float(pmf[hit].sum()) if np.isfinite(pmf).all() else np.nan


def total_probabilities(pmf: np.ndarray, support: np.ndarray, U: float) -> tuple[float, float, float]:
    """(P(total > U), P(total = U), P(total < U)) — sums to 1 exactly."""
    if not np.isfinite(pmf).all():
        return (np.nan, np.nan, np.nan)
    p_eq = float(pmf[support == int(U)].sum()) if U == int(U) else 0.0
    p_gt = float(pmf[support > U].sum())
    p_lt = float(pmf[support < U].sum())
    return (p_gt, p_eq, p_lt)


# ---------------------------------------------------------------------------
# Regression members (per-side score regressions on point-in-time features)
# ---------------------------------------------------------------------------
def _make_reg(name: str):
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**config.XGBOOST_REG_PARAMS)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**config.LIGHTGBM_REG_PARAMS)
    raise KeyError(f"unknown regression member {name!r}")


class ScoreRegressor:
    """mu_h / mu_a from a boosted-tree regression pair (tree view)."""

    def __init__(self) -> None:
        self.home_model = _make_reg("xgboost")
        self.away_model = _make_reg("lightgbm")

    def fit(self, df: pd.DataFrame) -> "ScoreRegressor":
        X = feat_mod.tree_view(df).to_numpy(dtype=np.float64)
        # NaN-safe: trees route NaN natively; guard all-NaN columns by
        # filling with the training median (fit-time only).
        X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
        self.home_model.fit(X, df["home_score"].astype(float).to_numpy())
        self.away_model.fit(X, df["away_score"].astype(float).to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = feat_mod.tree_view(df).to_numpy(dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
        return self.home_model.predict(X), self.away_model.predict(X)


# ---------------------------------------------------------------------------
# sigma estimation / calibration — pooled OOF residuals, one scalar pair
# ---------------------------------------------------------------------------
def calibrate_sigma(resid_margin: np.ndarray, resid_total: np.ndarray) -> dict:
    """Robust sigma estimates from pooled OOF residuals (1.4826 * MAD =
    sigma of a normal). One scalar pair keeps the PMF coherent."""
    rm = resid_margin[np.isfinite(resid_margin)]
    rt = resid_total[np.isfinite(resid_total)]
    s_m = float(1.4826 * np.median(np.abs(rm - np.median(rm)))) if len(rm) else config.MARGIN_SIGMA
    s_t = float(1.4826 * np.median(np.abs(rt - np.median(rt)))) if len(rt) else config.TOTAL_SIGMA
    s_m = float(np.clip(s_m, config.SIGMA_FLOOR_MARGIN, config.SIGMA_CAP_MARGIN))
    s_t = float(np.clip(s_t, config.SIGMA_FLOOR_TOTAL, config.SIGMA_CAP_TOTAL))
    return {"sigma_margin": s_m, "sigma_total": s_t}


# ---------------------------------------------------------------------------
# Distribution engine — everything derives from (mu_h, mu_a, sigma_m, sigma_t)
# ---------------------------------------------------------------------------
MARGIN_SUPPORT = np.arange(-config.MARGIN_PMF_MAX, config.MARGIN_PMF_MAX + 1)
TOTAL_SUPPORT = np.arange(0, config.TOTAL_PMF_MAX + 1)


def game_distribution(mu_h: float, mu_a: float,
                      sigma_margin: float, sigma_total: float) -> dict:
    """All distributional outputs for one game from the joint score model."""
    mu_margin = mu_h - mu_a
    mu_total = mu_h + mu_a
    pmf_m = discrete_normal_pmf(mu_margin, sigma_margin, MARGIN_SUPPORT)
    pmf_t = discrete_normal_pmf(mu_total, sigma_total, TOTAL_SUPPORT)

    # derived moneyline: P(home > away) = P(margin > 0) = P(margin >= 1)
    p_home_win = margin_cdf_above(pmf_m, MARGIN_SUPPORT, 0.0)
    p_tie = margin_pmf_at(pmf_m, MARGIN_SUPPORT, 0.0)
    p_away_win = 1.0 - p_home_win - p_tie

    out = {
        "mu_h": float(mu_h), "mu_a": float(mu_a),
        "mu_margin": float(mu_margin), "mu_total": float(mu_total),
        "p_home_win_derived": p_home_win,
        "p_away_win_derived": p_away_win,
        "p_tie": p_tie,
    }
    # fair spread / fair total: integer medians of the PMFs
    out["fair_spread"] = _pmf_median(pmf_m, MARGIN_SUPPORT)
    out["fair_total"] = _pmf_median(pmf_t, TOTAL_SUPPORT)
    # spread grid: p_home_cover_L / p_push_L for L in -14..+14
    for L in config.SPREAD_GRID:
        out[f"p_home_cover_{L}"] = margin_cdf_above(pmf_m, MARGIN_SUPPORT, float(L))
        out[f"p_push_{L}"] = margin_pmf_at(pmf_m, MARGIN_SUPPORT, float(L))
    # half-stop lines (±0.5): no push band (margins are integers)
    for L in config.HALF_STOP_LINES:
        out[f"p_home_cover_{str(L).replace('.', '_').replace('-', 'm')}"] = \
            margin_cdf_above(pmf_m, MARGIN_SUPPORT, L)
    # totals grid: over/push/under for U in 24..66
    for U in config.TOTAL_GRID:
        p_over, p_push, p_under = total_probabilities(pmf_t, TOTAL_SUPPORT, float(U))
        out[f"p_over_{U}"] = p_over
        out[f"p_push_{U}"] = p_push
        out[f"p_under_{U}"] = p_under
    return out


def _pmf_median(pmf: np.ndarray, support: np.ndarray) -> float:
    """Smallest integer k with cumulative PMF >= 0.5 (the fair line)."""
    if not np.isfinite(pmf).all():
        return np.nan
    c = np.cumsum(pmf)
    idx = int(np.searchsorted(c, 0.5))
    return float(support[min(idx, len(support) - 1)])


def apply_distribution(df: pd.DataFrame, sigma_margin: float,
                       sigma_total: float) -> pd.DataFrame:
    """Expand a frame with mu_h/mu_a into the full distributional columns.

    Distribution columns already present on the input are replaced (never
    duplicated) — the engine is the single authority for these values."""
    rows = [
        game_distribution(r.mu_h, r.mu_a, sigma_margin, sigma_total)
        for r in df.itertuples(index=False)
    ]
    dist = pd.DataFrame(rows, index=df.index)
    base = df.drop(columns=[c for c in dist.columns if c in df.columns])
    return pd.concat([base.reset_index(drop=True), dist.reset_index(drop=True)],
                     axis=1)


# ---------------------------------------------------------------------------
# Walk-forward OOF for the distribution model
# ---------------------------------------------------------------------------
def walk_forward_oof(game_df: pd.DataFrame,
                     date_col: str = "gameday") -> dict:
    """Expanding walk-forward OOF: per fold, fit mu regressions on strictly
    prior training games, predict validation, collect residuals for the
    pooled sigma calibration. Returns {oof, fold_table}."""
    df = game_df.sort_values(date_col).reset_index(drop=True)
    fold_list = folds_mod.make_folds(df, date_col=date_col)
    parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    for fold in fold_list:
        train = df.loc[fold.train_idx]
        val = df.loc[fold.val_idx]
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
        part["resid_margin"] = part["margin"] - (part["mu_h"] - part["mu_a"])
        part["resid_total"] = part["total"] - (part["mu_h"] + part["mu_a"])
        parts.append(part)
        fold_rows.append({
            "fold_id": fold.fold_id,
            "val_start": str(fold.val_start.date()),
            "val_end": str(fold.val_end.date()),
            "n_train": int(len(train)), "n_val": int(len(val)),
        })
    oof = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return {"oof": oof, "fold_table": pd.DataFrame(fold_rows)}


def fit_final(game_df: pd.DataFrame) -> ScoreRegressor:
    """Final full-history refit of the mu regressions."""
    return ScoreRegressor().fit(game_df)
