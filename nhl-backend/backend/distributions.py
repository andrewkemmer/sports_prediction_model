"""NHL run-line and totals distribution engine.

Structural mirror of the NFL distributions.py (MLB lineage):

* one LightGBM Poisson regressor for home goals;
* one LightGBM Poisson regressor for away goals;
* the active binary-moneyline feature contract is the sole source feature list;
* genuine NaNs remain intact for native LightGBM missing-value routing;
* negative-binomial dispersion is estimated from walk-forward OOF with MLB's
  pooled method-of-moments estimator;
* one Monte Carlo score-pair sample supplies every total/margin probability.

FEATURE PARITY (MLB parity, structural not incidental). MLB's run engine does
not own a run-line feature list: ``build_side_frame(..., strict_feature_parity
=True)`` imports ``training.active_moneyline_feature_cols`` and hands the SAME
resolved list to both side regressors. This module does the same thing through
the NHL equivalent: the run-line matrix is built by the binary moneyline's own
``moneyline.member_matrix`` helper, so the run line PULLS the production moneyline
contract by construction rather than through a parallel path that merely agrees
today. Adopted RFE additions/removals therefore reach the run line
automatically, and there is no second list to keep in sync. The binary moneyline
is only ever READ here — never modified, and never re-fit.

Grids are NHL-sized: spread -8..+8 (goals), totals 4..12.
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

MC_DRAWS = 10_000
MC_SEED = 42
# MLB parity (run_engine.py): a conditional SE-guard on the OOF market
# derivation — if the worst totals-line MC standard error exceeds the target
# at the default draw count, re-simulate the whole derivation at the tail
# resolution. A TRANSPARENCY constant pair: the guard only ever fires when
# 10k draws are demonstrably too noisy for the tail lines.
MC_DRAWS_TAIL = 50_000
MC_SE_TARGET = 5e-3
# Sealed-holdout convention (MLB derive_markets_v3 HOLDOUT_DAYS parity): the
# LAST ``HOLDOUT_DAYS`` of OOF games are SEALED — no α(λ) fitting, no curve
# binning, and no final market-line calibrator may see them. The tail scores
# the market engine honestly instead of validating itself. The gate covers
# the α(λ) curve AND the fitting scalars: alpha_home/alpha_away are the
# method-of-moments estimates over the PRE-HOLDOUT rows (calibrate_dispersion
# passes the gated mask to estimate_alpha), and the same sig dict both ships
# for serving and derives the OOF market rows — one honest, pre-holdout-gated
# dispersion on both sides of that boundary.
HOLDOUT_DAYS = 21
# Alpha-machinery parity (run_engine.py 1e-6): the floor under every alpha
# below which NB degenerates to Poisson. NHL previously ran 1e-8.
ALPHA_FLOOR = 1e-6
# α(λ) curve machinery constants (MLB run_engine.py 784-786).
ALPHA_N_BINS = 7
ALPHA_MIN_BIN = 250
ALPHA_CAP = 2.0          # sane max — beyond this variance is degenerate

# Retained for backwards-compatible diagnostics/tests; production uses NB MC.
MARGIN_SUPPORT = np.arange(-config.MARGIN_PMF_MAX, config.MARGIN_PMF_MAX + 1)
TOTAL_SUPPORT = np.arange(0, config.TOTAL_PMF_MAX + 1)


def _pmf_median(pmf: np.ndarray, support: np.ndarray) -> float:
    """Compatibility median helper for legacy diagnostics."""
    if not np.isfinite(pmf).all():
        return np.nan
    return float(support[min(int(np.searchsorted(np.cumsum(pmf), 0.5)),
                           len(support) - 1)])


def _make_reg():
    from lightgbm import LGBMRegressor
    params = dict(config.LIGHTGBM_REG_PARAMS)
    params["objective"] = "poisson"
    return LGBMRegressor(**params)


class ScoreRegressor:
    """Two same-contract LightGBM Poisson regressors, one per score side."""

    def __init__(self, declare_categoricals: bool = True) -> None:
        self.home_model = _make_reg()
        self.away_model = _make_reg()
        # Categorical routing MUST match the OOF walk exactly (train/serve
        # skew): the walk passes the team-ID pair via categorical_feature by
        # name, so the production refit declares the same columns by default.
        self._declare_categoricals = bool(declare_categoricals)
        self.feature_columns: list[str] = []

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

    def fit(self, df: pd.DataFrame) -> "ScoreRegressor":
        X = self._matrix(df)
        fit_kwargs = ({"categorical_feature": list(config.TREE_CATEGORICAL_COLS)}
                      if self._declare_categoricals
                      and config.TREE_CATEGORICAL_COLS
                      and all(c in X.columns for c in config.TREE_CATEGORICAL_COLS)
                      else {})
        self.home_model.fit(X, pd.to_numeric(df["home_score"], errors="coerce"),
                            **fit_kwargs)
        self.away_model.fit(X, pd.to_numeric(df["away_score"], errors="coerce"),
                            **fit_kwargs)
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = self._matrix(df)
        return (
            np.clip(self.home_model.predict(X), 1e-6, None),
            np.clip(self.away_model.predict(X), 1e-6, None),
        )


# ---------------------------------------------------------------------------
# α(λ) curve machinery — byte-shape port of MLB run_engine.py's dispersion
# layer (alpha_bins / _fit_curve_* / alpha_of / eval_alpha_fit). MLB models
# dispersion as a λ-DEPENDENT curve selected out-of-bag among piecewise /
# linear / power forms; the NHL previously shipped a single pooled scalar α
# per side. On hockey's current data both collapse to the Poisson limit
# (alpha ≈ 0 everywhere), so the curve is flat — but if over-dispersion ever
# appears it will be modeled as a function of λ, never clipped away.
# ---------------------------------------------------------------------------
def alpha_bins(y: np.ndarray, lam: np.ndarray,
               n_bins: int = ALPHA_N_BINS,
               min_count: int = ALPHA_MIN_BIN) -> list[dict]:
    """Binned method-of-moments points: quantile bins on λ, underfilled bins
    merged into their nearest neighbor until every bin holds ≥ min_count
    games. Per bin: α = max(0, (Var(y) − mean(λ)) / mean(λ)²)."""
    y = np.asarray(y, float)
    lam = np.asarray(lam, float)
    edges = np.unique(np.quantile(lam, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([lam.min() - 1e-9, lam.max() + 1e-9])
    idx = np.clip(np.digitize(lam, edges[1:-1], right=False), 0, len(edges) - 2)
    groups = [np.where(idx == b)[0] for b in range(len(edges) - 1)]
    # Merge any bin below min_count into its smaller neighbor (loop: merges
    # can cascade). Edge bins merge with their single ADJACENT neighbor:
    # the last bin merges left and the first bin merges right. The original
    # loop crashed on the last bin (j = i + 1 one past the end) and wrapped
    # the first bin's merge to the LAST group (j = -1), a non-adjacent
    # merge that mislabeled a right-edge bin as the first bin's neighbor.
    # Latent since the MLB port: full-OOF pools always cleared min_count,
    # so neither edge fired until the sealed-holdout gate shrank the fit
    # pool and made small-sample paths reachable.
    while True:
        sizes = [len(g) for g in groups]
        if len(groups) <= 1 or min(sizes) >= min_count:
            break
        i = int(np.argmin(sizes))
        if i == 0:
            j = 1
        elif i == len(groups) - 1:
            j = i - 1
        else:
            j = i - 1 if sizes[i - 1] <= sizes[i + 1] else i + 1
        lo, hi = min(i, j), max(i, j)
        groups[lo] = np.concatenate([groups[lo], groups[hi]])
        del groups[hi]
    bins = []
    for g in groups:
        if not len(g):
            continue
        mu, var = float(lam[g].mean()), float(y[g].var(ddof=0))
        bins.append({
            "count": int(len(g)),
            "mean_lam": round(mu, 4),
            "alpha": round(max((var - mu) / (mu ** 2), 0.0), 4),
        })
    return sorted(bins, key=lambda b: b["mean_lam"])


def _bin_direction(lams: list[float], alphas: list[float]) -> int:
    """+1 when dispersion rises with λ, −1 when it falls. Data decides."""
    if len(lams) < 2 or np.std(alphas) == 0 or np.std(lams) == 0:
        return +1
    corr = np.corrcoef(lams, alphas)[0, 1]
    return -1 if corr < 0 else +1


def _fit_curve_piecewise(bins: list[dict]) -> dict:
    """Weighted isotonic fit through the bin points: monotone in the
    data-chosen direction, count-weighted, clipped to [0, ALPHA_CAP]."""
    from sklearn.isotonic import IsotonicRegression

    xs = np.array([b["mean_lam"] for b in bins])
    ys = np.array([max(b["alpha"], 0.0) for b in bins])
    w = np.array([b["count"] for b in bins], dtype=float)
    d = _bin_direction(xs.tolist(), ys.tolist())
    iso = IsotonicRegression(increasing=bool(d > 0), out_of_bounds="clip")
    iso.fit(xs, ys, sample_weight=w)
    grid = np.linspace(float(xs.min()), float(xs.max()), 40)
    vals = np.clip(iso.predict(grid), 0.0, ALPHA_CAP)
    return {"form": "piecewise", "lam": [round(float(v), 5) for v in grid],
            "alpha": [round(float(v), 5) for v in vals],
            "direction": "rising" if d > 0 else "falling"}


def _fit_curve_linear(bins: list[dict]) -> dict:
    xs = np.array([b["mean_lam"] for b in bins])
    ys = np.array([max(b["alpha"], 0.0) for b in bins])
    if len(xs) < 2:   # degenerate: single bin → constant level, no polyfit
        return {"form": "linear", "a": float(ys.mean()), "b": 0.0}
    b_, a_ = np.polyfit(xs, ys, 1)
    return {"form": "linear", "a": float(a_), "b": float(b_)}


def _fit_curve_power(bins: list[dict]) -> dict:
    pos = [b for b in bins if b["alpha"] > 0]
    if len(pos) < 2:
        return _fit_curve_linear(bins)
    xs = np.log(np.array([b["mean_lam"] for b in pos]))
    ys = np.log(np.array([b["alpha"] for b in pos]))
    if len(xs) < 2:
        return _fit_curve_linear(bins)
    c, log_a = np.polyfit(xs, ys, 1)
    return {"form": "power", "a": float(np.exp(log_a)), "c": float(c)}


def alpha_of(lam: np.ndarray, curve: dict) -> np.ndarray:
    """Evaluate the fitted α(λ) — always in [0, ALPHA_CAP]."""
    lam = np.asarray(lam, float)
    form = curve["form"]
    if form == "piecewise":
        out = np.interp(lam, curve["lam"], curve["alpha"])
    elif form == "linear":
        out = curve["a"] + curve["b"] * lam
    else:  # power
        out = curve["a"] * np.power(np.maximum(lam, 1e-9), curve["c"])
    return np.clip(out, 0.0, ALPHA_CAP)


def nb_pmf_matrix(ks: np.ndarray, mu_col: np.ndarray,
                  alpha_col: np.ndarray) -> np.ndarray:
    """Vectorized NB pmf: (n_games, len(ks)). ks ints ≥0; columns (n,1)."""
    from scipy.special import gammaln
    ks = np.asarray(ks, dtype=float)[None, :]
    n_size = 1.0 / np.maximum(alpha_col, ALPHA_FLOOR)
    p = n_size / (n_size + mu_col)
    logpmf = (gammaln(ks + n_size) - gammaln(n_size) - gammaln(ks + 1.0)
              + n_size * np.log(p) + ks * np.log1p(-p))
    return np.exp(logpmf)


def eval_alpha_fit(y: np.ndarray, lam: np.ndarray, alpha: np.ndarray,
                   tail_k: int = 10, kmax: int = 80) -> dict:
    """Validation metrics for an α vector on held-out games: absolute gap in
    P(X≥tail_k) and mean NB log-likelihood (higher is better)."""
    y = np.asarray(y, float)
    mu_col = np.maximum(np.asarray(lam, float), 1e-6)[:, None]
    a_col = np.maximum(np.asarray(alpha, float), ALPHA_FLOOR)[:, None]
    M = nb_pmf_matrix(np.arange(kmax + 1), mu_col, a_col)
    modeled_tail = float(M[:, tail_k:].sum(axis=1).mean())
    obs_tail = float((y >= tail_k).mean())
    loglik = float(np.log(np.maximum(
        M[np.arange(len(y)), np.clip(y.astype(int), 0, kmax)], 1e-12)).mean())
    return {"tail_gap": round(abs(modeled_tail - obs_tail), 5),
            "modeled_tail": round(modeled_tail, 5),
            "observed_tail": round(obs_tail, 5),
            "loglik": round(loglik, 5)}


def select_alpha_curve(y: np.ndarray, lam: np.ndarray,
                       seed: int = MC_SEED) -> tuple[dict, dict]:
    """Out-of-bag selection among piecewise/linear/power forms.

    Two-fold cross-fit (fit half A → score half B, swap); primary metric =
    |P(X≥tail_k) modeled − observed| on the held-out half,    tie-break = mean NB log-likelihood. The chosen form is then REFIT on all rows passed here
    by the caller. PRE-HOLDOUT DISCIPLINE: when the caller holds a dated
    OOF frame it must pass only the pre-holdout rows (production does —
    calibrate_dispersion's sealed-holdout gate); undated test fixtures fit
    everything they are given. Returns (curve, diagnostics)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    halves = [perm[:len(perm) // 2], perm[len(perm) // 2:]]
    fitters = {"piecewise": _fit_curve_piecewise,
               "linear": _fit_curve_linear,
               "power": _fit_curve_power}
    diag: dict[str, dict] = {}
    for name, fit_fn in fitters.items():
        scores = []
        for fit_idx, eval_idx in ((halves[0], halves[1]),
                                  (halves[1], halves[0])):
            curve = fit_fn(alpha_bins(y[fit_idx], lam[fit_idx]))
            ev = eval_alpha_fit(y[eval_idx], lam[eval_idx],
                                alpha_of(lam[eval_idx], curve))
            scores.append(ev)
        diag[name] = {
            "tail_gap_avg": round(
                (scores[0]["tail_gap"] + scores[1]["tail_gap"]) / 2, 5),
            "loglik_avg": round(
                (scores[0]["loglik"] + scores[1]["loglik"]) / 2, 5),
        }
    best = min(fitters,
               key=lambda n: (diag[n]["tail_gap_avg"], -diag[n]["loglik_avg"]))
    bins = alpha_bins(y, lam)
    if len(bins) < 2:
        # Everything merged into one bin (small samples): every parametric
        # form degenerates to a constant level — ship piecewise directly.
        curve = _fit_curve_piecewise(bins)
    else:
        curve = fitters[best](bins)
    if curve["form"] == "piecewise":
        curve["cross_fit_diagnostics"] = diag
        curve["selection_metric"] = diag[best]["tail_gap_avg"] \
            if best in diag else None
    return curve, {"selected": curve["form"], "candidates": diag,
                   "bins": bins}


def _alpha_vector_for_side(y: np.ndarray, mu: np.ndarray,
                           curve: dict) -> np.ndarray:
    """Per-game α column for one side under a fitted curve. The curve is
    whatever the caller fitted; the pre-holdout row selection is the
    caller's discipline (production fits on pre-holdout rows only via
    calibrate_dispersion's sealed-holdout gate). Falls back to the pooled
    scalar estimate when the frame cannot support binning — identical
    numbers to the old path."""
    lam = np.asarray(mu, float)
    if len(lam) < 2 * ALPHA_MIN_BIN:
        return np.full(len(lam), float(estimate_alpha(y, mu)))
    return alpha_of(lam, curve)


def estimate_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    """Estimate NB alpha from OOF residual dispersion — MLB's estimator.

    For NB variance ``mu + alpha*mu²``, the method-of-moments estimate is
    ``alpha = max((var_obs - lam_bar) / lam_bar², 0)``: the excess of observed
    variance over the Poisson expectation, divided by the squared mean
    intensity. A value near zero is the Poisson limit.

    This is byte-for-byte the same estimator as MLB's ``run_engine.fit_alpha``
    (mlb-backend/backend/run_engine.py:786) — pooled and UNWEIGHTED, rounded to
    4 dp, and deliberately UNCAPPED. The prior NHL form was a different,
    mu²-weighted moment ratio with a 2.0 saturation cap: a genuinely
    over-dispersed fit could be silently clipped, and the weighting made the
    estimate disagree with MLB whenever lambda was heterogeneous (mildly so on
    the full walk-forward OOF, where both forms sit at the Poisson limit). Both
    are gone. The only addition over MLB is the degenerate-input guard (MLB
    never needs it because its lambda is clipped at 1e-6 before it arrives
    here).
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    ok = np.isfinite(y) & np.isfinite(mu) & (mu > 0)
    if ok.sum() < 2:
        return 0.0
    lam = mu[ok]
    lam_bar = float(lam.mean())
    if lam_bar <= 0.0:
        return 0.0
    var_obs = float(y[ok].var(ddof=0))
    return round(max((var_obs - lam_bar) / (lam_bar ** 2), 0.0), 4)


def calibrate_dispersion(oof: pd.DataFrame) -> dict[str, Any]:
    """Fit the run line's NB dispersion from leakage-free OOF score predictions.

    MLB-shaped (run_engine.derive_markets_v3's alpha layer): a per-side
    α(λ) curve is selected out-of-bag among piecewise/linear/power forms.
    SEALED-HOLDOUT GATE (MLB v3 parity): when the frame carries ``gameday``
    the curve and the fitting scalars see only the PRE-HOLDOUT rows (dates
    strictly before ``max − HOLDOUT_DAYS``); the sealed tail is never fit
    on. A frame without ``gameday`` (unit-test fixtures, callers that never
    held a timeline) is ungated and fits the full frame, exactly as before.
    The shipped ``alpha_home``/``alpha_away`` scalars remain the
    method-of-moments estimates — the numbers the draw path actually
    consumes — while the fitted curves ride alongside in ``alpha_*_curve``
    as the diagnostic layer, with the per-row max under each curve reported
    as ``alpha_*_max``. ``poisson_limit`` keeps its SCALAR semantics — "the
    NB term is inactive in scoring" — because scoring draws from the
    scalars; bin-level MoM noise makes a per-row verdict hypersensitive (a
    pure-Poisson sample reads α≈0.005 per bin), so the curve's measured
    verdict lives in ``run_line_fit_check``'s Pearson probe instead. The
    gate scope is recorded in ``holdout`` (cutoff, n_pre, n_holdout,
    fitted_on); ``fitted_on`` mirrors MLB's ``pre-holdout OOF only`` tag,
    and an undersized pre-holdout pool degrades honestly to the full-frame
    fit (tagged ``full OOF (pre-holdout pool too small)``).
    """
    y_h = oof["home_score"].to_numpy(float)
    mu_h = oof["mu_h"].to_numpy(float)
    y_a = oof["away_score"].to_numpy(float)
    mu_a = oof["mu_a"].to_numpy(float)
    # Sealed-holdout mask: only rows with a real timeline can be gated. A
    # frame without dates (test fixtures, ad-hoc callers) is ungated and
    # fits everything — identical behavior to the pre-gate path.
    if "gameday" in oof.columns and len(oof):
        dates = pd.to_datetime(oof["gameday"], errors="coerce")
        if dates.notna().any():
            cutoff = dates.max().normalize() - pd.Timedelta(days=HOLDOUT_DAYS)
            gate = (dates < cutoff).to_numpy()
        else:
            cutoff, gate = None, np.ones(len(oof), dtype=bool)
    else:
        cutoff, gate = None, np.ones(len(oof), dtype=bool)

    def _side_mask(y: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, bool]:
        """Fit pool for one side: the gated pre-holdout rows when they can
        support a fit (>=2 valid rows), else every valid row. A timeline
        entirely inside the holdout window (early-season small samples)
        must degrade to the full-frame fit, never crash on an empty pool
        and never fabricate a fit from nothing."""
        valid = np.isfinite(y) & np.isfinite(mu) & (mu > 0)
        pre = valid & gate
        if int(pre.sum()) >= 2:
            return pre, True
        return valid, False

    ok_h, pre_h = _side_mask(y_h, mu_h)
    ok_a, pre_a = _side_mask(y_a, mu_a)
    curve_h, diag_h = select_alpha_curve(y_h[ok_h], mu_h[ok_h])
    curve_a, diag_a = select_alpha_curve(y_a[ok_a], mu_a[ok_a])
    alpha_home_vec = _alpha_vector_for_side(y_h[ok_h], mu_h[ok_h], curve_h)
    alpha_away_vec = _alpha_vector_for_side(y_a[ok_a], mu_a[ok_a], curve_a)
    ah = estimate_alpha(y_h[ok_h], mu_h[ok_h])
    aa = estimate_alpha(y_a[ok_a], mu_a[ok_a])
    return {
        "alpha_home": ah, "alpha_away": aa,
        "alpha_home_curve": curve_h, "alpha_away_curve": curve_a,
        "alpha_selection": {"home": diag_h, "away": diag_a},
        "alpha_home_max": round(float(alpha_home_vec.max()), 4),
        "alpha_away_max": round(float(alpha_away_vec.max()), 4),
        "distribution": "negative_binomial",
        "poisson_limit": bool(ah <= ALPHA_FLOOR and aa <= ALPHA_FLOOR),
        "mc_draws": MC_DRAWS,
        "mc_draws_tail": MC_DRAWS_TAIL,
        "mc_se_target": MC_SE_TARGET,
        "holdout": {
            "cutoff": (str(pd.Timestamp(cutoff).date())
                       if cutoff is not None else None),
            "n_pre": int(gate.sum()),
            "n_holdout": int(len(gate) - gate.sum()),
            "fitted_on": ("pre-holdout OOF only"
                          if cutoff is not None and pre_h and pre_a
                          else ("full OOF (pre-holdout pool too small)"
                                if cutoff is not None
                                else "full OOF (no gameday on frame)")),
        },
    }


def pearson_poisson_adequacy(y: np.ndarray, mu: np.ndarray) -> float:
    """Pearson chi-square / df. ≈1 → Poisson variance is adequate; >1 means
    over-dispersion the NB term should absorb (MLB run_engine fit probe)."""
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


def _mc_se_totals_max(grid: pd.DataFrame, n_draws: int) -> float:
    """Worst MC standard error over the totals grid (MLB's mc_se_totals_max):
    se = sqrt(p(1-p)/n) per row per line; the max is the guard's trigger."""
    if not len(grid) or n_draws <= 0:
        return 0.0
    se = 0.0
    for line in config.TOTAL_GRID:
        col = _grid_key("p_over", line)
        if col not in grid.columns:
            continue
        p = pd.to_numeric(grid[col], errors="coerce").to_numpy(float)
        p = p[np.isfinite(p)]
        if len(p):
            se = max(se, float(np.sqrt((p * (1 - p)) / n_draws).max()))
    return se


def simulate_distributions(mu_h: np.ndarray, mu_a: np.ndarray,
                           alpha_home: float, alpha_away: float,
                           n_draws: int = MC_DRAWS,
                           seed: int = MC_SEED,
                           meta_out: dict | None = None) -> pd.DataFrame:
    """Monte Carlo all NHL grid probabilities from paired score draws.

    MLB SE-guard parity (run_engine.derive_markets_v3): when the worst
    totals-line standard error exceeds :data:`MC_SE_TARGET` at this draw
    count, the whole derivation is re-simulated at :data:`MC_DRAWS_TAIL` and
    the bump is recorded. Pass ``meta_out`` to receive the ``mc_meta`` block
    (n_draws, requested_draws, mc_se_totals_max, reason) — the same
    transparency MLB writes into its markets summary.
    """
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
    se = _mc_se_totals_max(out, n_draws)
    meta = {"n_draws": int(n_draws), "requested_draws": int(n_draws),
            "mc_se_totals_max": round(se, 6), "reason": "default"}
    if se > MC_SE_TARGET and n_draws < MC_DRAWS_TAIL:
        # Bump once, whole derivation (MLB's derive_markets_v3 discipline).
        bumped = simulate_distributions(mu_h, mu_a, alpha_home, alpha_away,
                                        n_draws=MC_DRAWS_TAIL, seed=seed)
        se = _mc_se_totals_max(bumped, MC_DRAWS_TAIL)
        meta = {"n_draws": int(MC_DRAWS_TAIL),
                "requested_draws": int(n_draws),
                "mc_se_totals_max": round(se, 6),
                "reason": (f"SE {se:.4f} > {MC_SE_TARGET} at N={n_draws} "
                           "— bumped")}
        out = bumped
    if meta_out is not None:
        meta_out.update(meta)
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
                       sigma_total: float | None = None,
                       meta_out: dict | None = None) -> pd.DataFrame:
    """Expand mu predictions into the complete NB/MC market grid.

    ``meta_out`` receives the MC ``mc_meta`` block (see
    :func:`simulate_distributions`) when the caller wants the derivation's
    resolution recorded.
    """
    if not isinstance(params, dict):
        params = {}
    ah = float(params.get("alpha_home", 0.0))
    aa = float(params.get("alpha_away", 0.0))
    dist = simulate_distributions(df["mu_h"].to_numpy(float),
                                  df["mu_a"].to_numpy(float), ah, aa,
                                  meta_out=meta_out)
    base = df.drop(columns=[c for c in dist.columns if c in df.columns], errors="ignore")
    return pd.concat([base.reset_index(drop=True), dist.reset_index(drop=True)], axis=1)


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
    # Sigmoid saturates by |x|=35 (values ~1e-16), far outside the clip bounds
    # below, so clamping the linear term is output-identical while keeping
    # np.exp inside float range (a=1.27 slate logits reached ~-1e3 in the
    # 2026-09-29 run and overflowed the exp before the clip could save it).
    lin = np.clip(cal["a"] * z + cal["b"], -35.0, 35.0)
    return np.clip(1.0 / (1.0 + np.exp(-lin)),
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
    # Derived model moneyline uses the same favored-team calibration contract
    # as every published grid line: the IN-FRAME column is calibrated by the
    # PREQUENTIAL per-fold map (fold k sees only folds < k), so the sealed
    # tail and the derived_ml winner card score probabilities their own
    # calibrator never saw. (2026-09-30 leakage audit: this was the one
    # pooled in-place calibrator — a map fitted on ALL OOF rows, sealed tail
    # included, was stamped onto the frame and the monitor's sealed-window
    # derived_moneyline metrics then validated on rows that map had seen.
    # The pooled all-OOF map below is the serving-layer record, exactly like
    # the pooled moneyline Platt; apply_market_calibration deliberately does
    # not apply it to the slate, whose derived ML stays the raw MC value.)
    if "p_home_win_derived" in out:
        p = out["p_home_win_derived"].to_numpy(float)
        fav_home = p >= 0.5
        pf = np.where(fav_home, p, 1.0 - p)
        yf = np.where(fav_home, margin > 0, margin < 0).astype(int)
        pc_seq, cal = _prequential_line(pf, yf, folds)
        pc = np.maximum(0.5, pc_seq)
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


