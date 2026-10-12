"""NHL PIT distribution engine: score features plus CRPS CDF calibration.

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
    # Input order is chronology. Prefix-only rolling selection; no reverse
    # train-on-future half. Kept for legacy NB diagnostics, not CRPS pricing.
    y, lam = np.asarray(y, float), np.asarray(lam, float)
    if len(y) < 4:
        bins = alpha_bins(y, lam) if len(y) else [{"count": 1, "mean_lam": 1., "alpha": 0.}]
        return _fit_curve_piecewise(bins), {"selected": "piecewise", "candidates": {}, "bins": bins}
    cuts = sorted(set([max(2, len(y) // 2), max(3, 3 * len(y) // 4), len(y)]))
    splits = [(np.arange(cuts[i]), np.arange(cuts[i], cuts[i + 1]))
              for i in range(len(cuts) - 1)]
    fitters = {"piecewise": _fit_curve_piecewise,
               "linear": _fit_curve_linear,
               "power": _fit_curve_power}
    diag: dict[str, dict] = {}
    for name, fit_fn in fitters.items():
        scores = []
        for fit_idx, eval_idx in splits:
            curve = fit_fn(alpha_bins(y[fit_idx], lam[fit_idx]))
            ev = eval_alpha_fit(y[eval_idx], lam[eval_idx],
                                alpha_of(lam[eval_idx], curve))
            scores.append(ev)
        diag[name] = {
            "tail_gap_avg": round(
                np.mean([s["tail_gap"] for s in scores]), 5),
            "loglik_avg": round(
                np.mean([s["loglik"] for s in scores]), 5),
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
    """Outcome-free analytic CDF; final CRPS maps are applied by calibration.

    Legacy alpha/MC params describe diagnostics only, not these prices.
    """
    if meta_out is not None:
        meta_out.update({"n_draws": 0, "requested_draws": 0,
                         "mc_se_totals_max": 0., "reason": "analytic_cdf"})
    return _distribution_price(df, {})


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
    if bundle and bundle.get("method") == "causal_lattice_crps":
        out = _distribution_price(df, bundle.get("maps", {}))
        out["distribution_fit_n"] = bundle.get("fit_n", 0)
        origins = bundle.get("origins") or []
        out["distribution_fit_end"] = origins[-1]["end"] if origins else None
        return out

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




# ---------------------------------------------------------------------------
# Causal whole-distribution layer. Mirrored across sports deliberately: these
# backends run independently in Kaggle; no new shared runtime is required.
# ---------------------------------------------------------------------------
DISTRIBUTION_SPORT = "nhl"
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
