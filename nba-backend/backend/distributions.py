"""NBA totals and point-spread/run-line distribution engine.

Structural mirror of the NFL/NHL distributions.py (MLB lineage):

* two LightGBM Poisson regressors (expected points, one per side) sharing the
  active binary-moneyline feature contract;
* genuine NaNs remain intact for native LightGBM missing-value routing;
* negative-binomial dispersion is estimated from leakage-free walk-forward
  OOF with MLB's pooled method-of-moments estimator, with the MLB α(λ) curve
  machinery riding alongside as the diagnostic layer;
* one Monte Carlo score-pair sample supplies every total/margin probability,
  under a conditional SE-guard that records (and repairs) its own resolution;
* every run logs the MLB-shaped fit-diagnostics pack (Pearson Poisson
  adequacy, pooled deviance/RMSE against the constant league-mean baseline)
  and records it in the markets meta.

FEATURE PARITY (MLB parity, structural not incidental). MLB's run engine does
not own a run-line feature list: the side models receive the SAME resolved
list the production moneyline serves. This module does that through the NBA
equivalent: the run-line matrix is built by the binary moneyline's own
``moneyline.member_matrix`` helper, so the run line PULLS the production
moneyline contract by construction rather than through a parallel path that
merely agrees today. Adopted RFE additions/removals therefore reach the run
line automatically, and there is no second list to keep in sync. The binary
moneyline is only ever READ here — never modified, and never re-fit.

It never reads a sportsbook or silently changes source.
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


def _make_reg():
    from lightgbm import LGBMRegressor
    params = dict(config.LIGHTGBM_REG_PARAMS)
    params["objective"] = "poisson"
    return LGBMRegressor(**params)


class ScoreRegressor:
    """Two same-contract LightGBM Poisson regressors, one per score side."""

    def __init__(self) -> None:
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
        try:
            self.home_model = _make_reg()
            self.away_model = _make_reg()
            # Declare the team-ID categoricals by name in the refit, exactly
            # as the OOF walk's matrix marks them (family standard after the
            # NHL train/serve skew guard): pandas Categorical columns are
            # auto-detected by LightGBM, but the explicit declaration makes
            # the refit's routing identical to the walk's by contract rather
            # than by pandas courtesy.
            categorical = [c for c in config.TREE_CATEGORICAL_COLS
                           if c in getattr(X, "columns", [])]
            try:
                if categorical:
                    self.home_model.fit(X, h.clip(lower=0),
                                        categorical_feature=categorical)
                    self.away_model.fit(X, a.clip(lower=0),
                                        categorical_feature=categorical)
                else:
                    self.home_model.fit(X, h.clip(lower=0))
                    self.away_model.fit(X, a.clip(lower=0))
            except TypeError:
                # Older LightGBM wrappers without the keyword keep the
                # pandas auto-detection routing.
                self.home_model.fit(X, h.clip(lower=0))
                self.away_model.fit(X, a.clip(lower=0))
            self.fallback = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("LightGBM score regressor unavailable, using ridge: %s", exc)
            from sklearn.linear_model import Ridge
            numeric = X.copy()
            for col in config.TREE_CATEGORICAL_COLS:
                if col in numeric:
                    numeric[col] = numeric[col].astype(float)
            self.home_model = self.away_model = None
            self.fallback = (Ridge(alpha=1.0), Ridge(alpha=1.0))
            self.fallback[0].fit(numeric, h)
            self.fallback[1].fit(numeric, a)
            self._matrix = lambda frame, _numeric=numeric.columns: self._matrix_numeric(frame, _numeric)  # type: ignore[method-assign]
        return self

    def _matrix_numeric(self, df: pd.DataFrame, columns) -> pd.DataFrame:
        X = feat_mod_tree(df).reindex(columns=list(columns))
        for col in config.TREE_CATEGORICAL_COLS:
            if col in X:
                X[col] = X[col].astype(float)
        return X

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.fallback is not None:
            X = self._matrix(df).astype(float)
            return (np.clip(self.fallback[0].predict(X), 1e-6, None),
                    np.clip(self.fallback[1].predict(X), 1e-6, None))
        if self.home_model is None or self.away_model is None:
            raise RuntimeError("score regressor is not fitted")
        X = self._matrix(df)
        return (np.clip(self.home_model.predict(X), 1e-6, None),
                np.clip(self.away_model.predict(X), 1e-6, None))


def feat_mod_tree(df: pd.DataFrame) -> pd.DataFrame:
    """The moneyline tree view, without the member wrapper (fallback path)."""
    return ml_mod.member_matrix(MONEYLINE_TREE_MEMBER, df)


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
    params = params or {}
    dist = simulate_distributions(
        df.mu_h.to_numpy(float), df.mu_a.to_numpy(float),
        float(params.get("alpha_home", 0)), float(params.get("alpha_away", 0)),
        n_draws=int(params.get("mc_draws", MC_DRAWS)),
        seed=int(params.get("seed", MC_SEED)), meta_out=meta_out)
    base = df.drop(columns=[c for c in dist.columns if c in df.columns], errors="ignore")
    return pd.concat([base.reset_index(drop=True), dist.reset_index(drop=True)], axis=1)


def walk_forward_oof(game_df: pd.DataFrame, date_col: str = "gameday",
                     fold_list=None) -> dict:
    df = folds_mod.canonical_sort(game_df, date_col)
    folds = fold_list if fold_list is not None else folds_mod.make_folds(df, date_col)
    parts: list[pd.DataFrame] = []
    rows: list[dict] = []
    for fold in folds:
        train, val = df.loc[fold.train_idx], df.loc[fold.val_idx]
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
    calibrated_home = np.clip(calibrated_home, 0.0, 1.0 - tie)
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


def calibrate_market_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
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
