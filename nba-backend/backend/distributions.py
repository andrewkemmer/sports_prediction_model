"""NBA PIT distribution engine: score features plus CRPS CDF calibration.

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

# MLB parity (run_engine.py): 10k draws at the default resolution, with a
# conditional SE-guard that re-simulates the whole derivation at the tail
# resolution when the worst totals-line standard error exceeds the target.
# The NBA totals grid is the family's widest (101 integer lines), so tail
# lines are exactly where MC noise shows up first.
MC_DRAWS = 10_000
MC_SEED = 42
MC_DRAWS_TAIL = 50_000
MC_SE_TARGET = 5e-3
# Sealed-holdout convention (MLB derive_markets_v3 HOLDOUT_DAYS parity,
# NHL 62d00fd port): the LAST ``HOLDOUT_DAYS`` of OOF games are SEALED — no
# α(λ) fitting, no curve binning, and the fitting scalars never see them.
# The tail scores the market engine honestly instead of validating itself.
# The shipped serving dispersion (alpha_home/alpha_away scalars in the
# bundle) rides the same gate here; the gate disciplines what the run
# engine's evaluation numbers are allowed to have fit on.
HOLDOUT_DAYS = 21
# Alpha-machinery parity (run_engine.py 1e-6): the floor under every alpha
# below which NB degenerates to Poisson. The NBA previously ran 1e-8.
ALPHA_FLOOR = 1e-6
# α(λ) curve machinery constants (MLB run_engine.py 784-786).
ALPHA_N_BINS = 7
ALPHA_MIN_BIN = 250
ALPHA_CAP = 2.0          # sane max — beyond this variance is degenerate


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


# ---------------------------------------------------------------------------
# α(λ) curve machinery — byte-shape port of MLB run_engine.py's dispersion
# layer (alpha_bins / _fit_curve_* / alpha_of / eval_alpha_fit). MLB models
# dispersion as a λ-DEPENDENT curve selected out-of-bag among piecewise /
# linear / power forms; the NBA previously shipped a single pooled scalar α
# per side. On basketball's current data the pooled estimate is small, so the
# curve may be flat — but if over-dispersion ever appears it will be modeled
# as a function of λ, never clipped away.
# ---------------------------------------------------------------------------
def alpha_bins(y: np.ndarray, lam: np.ndarray,
               n_bins: int = ALPHA_N_BINS,
               min_count: int = ALPHA_MIN_BIN) -> list[dict]:
    """Binned method-of-moments points: quantile bins on λ, underfilled bins
    merged into their nearest neighbor until every bin holds ≥ min_count
    games. Per bin: α = max(0, (Var(y) − mean(λ)) / mean(λ)²). The caller
    (``calibrate_dispersion``) owns the sealed-holdout gate and passes only
    the rows this fit is allowed to see; edge bins merge with their single
    adjacent neighbor exactly as the family standard fixes it (first bin
    right, last bin left), so a shrunken pre-holdout pool cannot trip the
    latent wraparound the NHL port already repaired."""
    y = np.asarray(y, float)
    lam = np.asarray(lam, float)
    edges = np.unique(np.quantile(lam, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([lam.min() - 1e-9, lam.max() + 1e-9])
    idx = np.clip(np.digitize(lam, edges[1:-1], right=False), 0, len(edges) - 2)
    groups = [np.where(idx == b)[0] for b in range(len(edges) - 1)]
    # Merge any bin below min_count into its smaller neighbor (loop: merges
    # can cascade). The neighbor rule must respect the array's edges — the
    # first group has only a right neighbor, the last group only a left one
    # (the byte-shape port indexed past the end when the LAST bin was the
    # underfilled one, and read sizes[-1] as a wraparound when the first was).
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
    |P(X≥tail_k) modeled − observed| on the held-out half, tie-break = mean
    NB log-likelihood. The chosen form is then REFIT on all rows passed here
    by the caller. PRE-HOLDOUT DISCIPLINE: when the caller holds a dated
    OOF frame it must pass only the pre-holdout rows (production does —
    ``calibrate_dispersion``'s sealed-holdout gate); undated test fixtures
    fit everything they are given. Returns (curve, diagnostics)."""
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
    ``calibrate_dispersion``'s sealed-holdout gate). Falls back to the pooled
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
    — pooled and UNWEIGHTED, rounded to 4 dp, and deliberately UNCAPPED. The
    prior NBA form was a different, mu²-weighted moment ratio with a 2.0
    saturation cap: a genuinely over-dispersed fit could be silently clipped,
    and the weighting made the estimate disagree with MLB whenever lambda was
    heterogeneous. Both are gone. The only addition over MLB is the
    degenerate-input guard (MLB never needs it because its lambda is clipped
    at 1e-6 before it arrives here).
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


def calibrate_dispersion(oof: pd.DataFrame | None) -> dict[str, Any]:
    """Fit the run line's NB dispersion from leakage-free OOF score predictions.

    An absent or empty OOF frame (a contained training failure upstream)
    degrades to the Poisson draw path exactly as before — a missing fit is
    never an exception on the artifact path.

    MLB-shaped (run_engine.derive_markets_v3's alpha layer): a per-side
    α(λ) curve is selected out-of-bag among piecewise/linear/power forms.
    SEALED-HOLDOUT GATE (NHL 62d00fd port of MLB v3 parity): when the frame
    carries ``gameday``, the curve and the fitting scalars see only the
    PRE-HOLDOUT rows (dates strictly before ``max − HOLDOUT_DAYS``); the
    sealed tail is never fit on. A frame without ``gameday`` (unit-test
    fixtures, callers that never held a timeline) is ungated and fits the
    full frame, exactly as before. The shipped ``alpha_home``/``alpha_away``
    scalars remain the pooled method-of-moments estimates — the numbers the
    draw path actually consumes — while the fitted curves ride alongside in
    ``alpha_*_curve`` as the diagnostic layer, with the per-row max under
    each curve reported as ``alpha_*_max``. ``poisson_limit`` keeps its
    SCALAR semantics — "the NB term is inactive in scoring" — because
    scoring draws from the scalars; bin-level MoM noise makes a per-row
    verdict hypersensitive (a pure-Poisson sample reads α≈0.005 per bin),
    so the curve's measured verdict lives in ``run_line_fit_check``'s
    Pearson probe instead. The gate scope is recorded in ``holdout``
    (cutoff, n_pre, n_holdout, fitted_on); an undersized pre-holdout pool
    degrades honestly to the full-frame fit (tagged
    ``full OOF (pre-holdout pool too small)``).
    """
    if oof is None or not len(oof) or "mu_h" not in oof:
        return {"alpha_home": 0.0, "alpha_away": 0.0,
                "distribution": "negative_binomial", "mc_draws": MC_DRAWS,
                "mc_draws_tail": MC_DRAWS_TAIL, "mc_se_target": MC_SE_TARGET,
                "poisson_limit": True}
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
                           alpha_home: float = 0.0, alpha_away: float = 0.0,
                           n_draws: int = MC_DRAWS, seed: int = MC_SEED,
                           meta_out: dict | None = None) -> pd.DataFrame:
    """Price all configured lines from paired NBA score draws.

    MLB SE-guard parity (run_engine.derive_markets_v3): when the worst
    totals-line standard error exceeds :data:`MC_SE_TARGET` at this draw
    count, the whole derivation is re-simulated at :data:`MC_DRAWS_TAIL` and
    the bump is recorded. Pass ``meta_out`` to receive the ``mc_meta`` block
    (n_draws, requested_draws, mc_se_totals_max, reason) — the same
    transparency MLB writes into its markets summary."""
    rng = np.random.default_rng(seed)
    mu_h = np.asarray(mu_h, dtype=float)
    mu_a = np.asarray(mu_a, dtype=float)
    if len(mu_h) != len(mu_a):
        raise ValueError("home/away expected-score arrays must have equal length")
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
    se = _mc_se_totals_max(out, int(n_draws))
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


def _fair(row: dict[str, Any], prefix: str, lines: list[int]) -> float:
    values = np.asarray([float(row.get(_grid_key(prefix, line), np.nan))
                         for line in lines], dtype=float)
    if not np.isfinite(values).any():
        return float(lines[0])
    return float(lines[int(np.nanargmin(np.abs(values - 0.5)))])


def apply_distribution(df: pd.DataFrame, params: dict | None = None,
                       meta_out: dict | None = None, **kwargs) -> pd.DataFrame:
    """Outcome-free analytic CDF; final CRPS maps are applied by calibration.

    Legacy alpha/MC params describe diagnostics only, not these prices.
    """
    if meta_out is not None:
        meta_out.update({"n_draws": 0, "requested_draws": 0,
                         "mc_se_totals_max": 0., "reason": "analytic_cdf"})
    return _distribution_price(df, {})


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
    values = np.asarray(z, dtype=float)
    out = np.empty_like(values, dtype=float)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_z = np.exp(values[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def _apply_platt(p, cal):
    p = np.asarray(p, dtype=float)
    if not cal:
        return np.clip(p, 0.0, 1.0)
    z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    return _sigmoid(float(cal.get("a", 1)) * z + float(cal.get("b", 0)))


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
    calibrated_home = np.minimum(np.maximum(calibrated_home, 0.0), 1.0 - tie)
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


def _legacy_calibrate_market_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply causal line-specific calibration to a priced OOF market frame.

    Third-leg audit (NHL commit 8d6f9b0 parity, verified 2026-09-27): every
    leg of every three-way market is calibrated from its OWN outcome — over
    vs ``total > line``, push vs ``total == line``, under vs ``total < line``
    (and symmetrically across the margin legs) — never from another leg's
    map. The renormalization afterwards is the only cross-leg interaction.
    """
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
    """Apply the OOF-fitted calibration bundle to a slate frame."""
    if bundle and bundle.get("method") == "causal_lattice_crps":
        out = _distribution_price(df, bundle.get("maps", {}))
        out["distribution_fit_n"] = bundle.get("fit_n", 0)
        origins = bundle.get("origins") or []
        out["distribution_fit_end"] = origins[-1]["end"] if origins else None
        return out

    out = df.copy()
    if not bundle or not len(out):
        return out
    for line, maps in bundle.get("totals", {}).items():
        line_val = int(float(line))
        key, push_key, under_key = (_grid_key("p_over", line_val),
                                    _grid_key("p_push_total", line_val),
                                    _grid_key("p_under", line_val))
        if key in out:
            out[key] = np.clip(_apply_platt(out[key].to_numpy(float),
                                            maps.get("over")), 0.0, 1.0)
            if maps.get("push") and push_key in out:
                out[push_key] = np.clip(_apply_platt(out[push_key].to_numpy(float),
                                                     maps["push"]), 0.0, 1.0)
            if maps.get("under") and under_key in out:
                out[under_key] = np.clip(_apply_platt(out[under_key].to_numpy(float),
                                                      maps["under"]), 0.0, 1.0)
    for line, maps in bundle.get("run_lines", {}).items():
        line_val = float(line)
        key, push_key, away_key = (_grid_key("p_home_cover", line_val),
                                   _grid_key("p_push", line_val),
                                   _grid_key("p_away_cover", line_val))
        if key in out:
            out[key] = np.clip(_apply_platt(out[key].to_numpy(float),
                                            maps.get("home")), 0.0, 1.0)
            if maps.get("push") and push_key in out:
                out[push_key] = np.clip(_apply_platt(out[push_key].to_numpy(float),
                                                     maps["push"]), 0.0, 1.0)
            if maps.get("away") and away_key in out:
                out[away_key] = np.clip(_apply_platt(out[away_key].to_numpy(float),
                                                     maps["away"]), 0.0, 1.0)
    if bundle.get("derived_moneyline") and "p_home_win_derived" in out:
        p = out.p_home_win_derived.to_numpy(float)
        tie = out.get("p_tie", pd.Series(np.zeros(len(out)), index=out.index)).to_numpy(float)
        p, tie = _apply_derived_calibration(p, tie, bundle["derived_moneyline"])
        out["p_home_win_derived"] = p
        out["p_tie"] = tie
        out["p_away_win_derived"] = np.maximum(0.0, 1.0 - tie - p)
    return out


# ---------------------------------------------------------------------------
# Causal whole-distribution layer. Mirrored across sports deliberately: these
# backends run independently in Kaggle; no new shared runtime is required.
# ---------------------------------------------------------------------------
DISTRIBUTION_SPORT = "nba"
CRPS_MIN_HISTORY = 200
# NBA two-sided dispersion family (Step 1; judged in scratch A/B
# scratch/nba_side_dispersion_ab.py before enabling): fit per-side variance
# inflation and residual side correlation into the RAW family instead of one
# shared Poisson-sum scale, so total and margin carry different variances.
# Enabled 2026-10-12 after the pre-registered scratch A/B won: lattice CRPS
# down for both kinds, total var-ratio by mu-decile 1.12 -> 1.04 (margin mean
# 0.95 -> 1.01), far-tail gaps halved, every margin fixed line improved,
# derived-ML logloss 0.6083 -> 0.6075; card metrics stayed pinned. Set False
# to revert to legacy pricing; maps bundles without the NBA alphas key also
# price legacy regardless of this flag.
NBA_SIDE_DISPERSION = True


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
        # Two-sided Gaussian family (2026-10-12): per-side variances
        # c_side * mu_side plus residual side correlation rho, so total and
        # margin carry DIFFERENT variances (s2h + s2a +/- 2*rho*sqrt(s2h*s2a))
        # instead of one shared Poisson-sum scale. Measured on 2024-26 OOF:
        # c ~ 1.3 per side, rho ~ 0.25 -> total variance x1.25, margin x0.75
        # versus the legacy shared scale (side residuals share pace shocks).
        # alphas = [c_h, c_a, rho], moment-fitted on PRIOR folds only (see
        # _distribution_fit_maps); absent (legacy bundles) falls back to the
        # original shared scale -- exactly this family at c=1, rho=0.
        if alphas is not None and len(alphas) >= 3:
            c_h, c_a, rho = (float(alphas[0]), float(alphas[1]),
                             float(alphas[2]))
            s2h = np.maximum(c_h * h, 1e-9)
            s2a = np.maximum(c_a * a, 1e-9)
            cross = 2. * rho * np.sqrt(s2h * s2a)
            location = h + a if kind == "total" else h - a
            variance = s2h + s2a + (cross if kind == "total" else -cross)
            scale = np.sqrt(np.maximum(variance, 1e-9))
        else:
            location = h + a if kind == "total" else h - a
            # Legacy baseline: one Poisson-sum scale shared by total and
            # margin. Conditional width is subsequently learned by CRPS on
            # strictly earlier OOF predictions.
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
    elif DISTRIBUTION_SPORT == "nba" and NBA_SIDE_DISPERSION:
        # Two-sided family dispersion (Step 1): moment estimates on PRIOR
        # folds only -- per-side variance inflation c = sum(e^2)/sum(mu) and
        # residual side correlation rho = sum(eh*ea)/sqrt(sum eh^2 * sum ea^2).
        # The causal lattice-CRPS warp keeps learning level-dependent width
        # on top; the admission gate is unchanged and still gates the warp.
        hs = .5 * (outcomes["total"] + outcomes["margin"])
        aws = .5 * (outcomes["total"] - outcomes["margin"])
        if len(hs) >= CRPS_MIN_HISTORY:
            eh, ea = hs - h, aws - a
            c_h = float(np.clip(np.sum(eh ** 2) / max(np.sum(h), 1e-12),
                                .2, 5.))
            c_a = float(np.clip(np.sum(ea ** 2) / max(np.sum(a), 1e-12),
                                .2, 5.))
            denom = float(np.sqrt(np.sum(eh ** 2) * np.sum(ea ** 2)))
            rho = (float(np.clip(np.sum(eh * ea) / denom, -.5, .95))
                   if denom > 1e-12 else 0.)
            alphas = [c_h, c_a, rho]
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
