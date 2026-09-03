"""NFL per-side joint layer — step 1: correlated integer-score joint PMF.

Turns the two per-side mean regressors (step 1, commit 688d417) into a
coherent joint distribution over NFL final integer scores — the NFL analog
of MLB's NB(lambda, alpha(lambda)) layer, adapted for near-normal integer
scores, positive cross-side residual covariance, and the ~0.5%
regular-season final-tie rate (pushes at books). ONE joint PMF prices
spread, totals, derived moneyline, and tie mass.

Step-1 scope (record-only — no wiring):
- Marginal family (DN vs NB) chosen EMPIRICALLY on pooled-OOF integer
  log-likelihood, never hardcoded. NB does not auto-win because MLB uses it
  — the NFL mean/variance regime differs. Within ~10 LL units default to DN
  for grid simplicity. Report both.
- Sigma curve: per-side power law sigma(mu) = sigma0 * mu^q fit on
  log|resid| vs log mu_hat (the MLB alpha(lambda) analog), compared against
  constant sigma (per-side residual RMSE) on OOF marginal integer LL.
- Cross-side dependence: ONE global Pearson rho on pooled-OOF standardized
  pairs z = resid / sigma(mu), CI via Fisher's z; marginal PMFs coupled with
  a Gaussian copula (bivariate-normal CDF at half-integer boundaries,
  inclusion-exclusion rectangle on the integer grid 0..75). Pooled rho is
  applied to sealed 2025 — NEVER refit on sealed.
- Tie diagonal: a rounded-latent joint on FINAL scores overstates tie mass
  (~2-3%) vs the true ~0.5% because tied-regulation games resolve in OT.
  Calibrate the diagonal mass to the pooled empirical final-tie rate
  (constant base; covariate model is data-limited at ~5-10 positives — not
  attempted) while preserving row/column marginals via iterative
  proportional fitting (IPF) with the diagonal sum fixed, shifting excess
  mass to near-diagonal cells. Report raw and calibrated.
- Derived probabilities per game from the CALIBRATED joint: margin PMF,
  total PMF, P(home covers -L) = P(margin > L), P(total over U),
  tie mass, derived ML = P(H>A)/(1 - P_tie).

Scope pin (verbatim):
  "Step-1 delivers the correlated per-side joint PMF plus validated derived
   probabilities. Market pricing/calibration paths, artifact emitters, and
   wire-in are later phases. Marginal family and sigma chosen empirically
   on pooled OOF; rho is a global scalar. Tie handling uses an IPF-calibrated
   final-tie diagonal because tied-regulation games resolve in OT;
   regulation-score/OT modeling is a deferred improvement."

All operations are deterministic (no RNG): identical inputs → byte-identical
outputs (the G5 determinism pin).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)

# ── Grid ─────────────────────────────────────────────────────────────────────
# Integer score support 0..75 (upper tail absorbed into cell 75; NFL scores
# 2019-2025 never exceed it — checked against the decided frame).
GRID_MAX = 75
GRID = np.arange(GRID_MAX + 1, dtype=float)

# Log-LL floor: scores are on the grid, so every cell has positive mass by
# construction; clip only against catastrophic float underflow.
LL_FLOOR = 1e-300

# Family selection: within this many pooled-OOF integer-LL units, default to
# DN for grid simplicity (NB does not auto-win on the NFL regime).
DN_TIEBREAK_LL = 10.0

# Sigma clamp (power-law extrapolation guard at extreme mu).
SIGMA_MIN = 1.0
SIGMA_MAX = 15.0

# IPF convergence.
IPF_MAX_ITER = 3000
IPF_TOL = 1e-11
# Floor for IPF: the p_tie diagonal refix can crush already-tiny tail-row
# mass (scores near 0/75) to exact 0 → row-normalize divides by 0 → NaN.
# 1e-300 is far below any real mass and keeps every division safe.
IPF_FLOOR = 1e-300

# Copula quantile clamp (norm.ppf(0) / norm.ppf(1) boundary handling).
_Q_CLIP = 37.0

# Artifact pin: the step-1 residual artifact contains exactly these rows.
ARTIFACT_N_EXPECTED = 1091
ARTIFACT_COLUMNS = ["game_id", "fold_idx", "pred_home", "pred_away",
                    "resid_home", "resid_away", "best_iter_home",
                    "best_iter_away"]


# ── Marginal families (per side) ────────────────────────────────────────────

def dn_pmf(mu: float, sigma: float) -> np.ndarray:
    """Discretized normal PMF on GRID via half-integer CDF boundaries.

    P(X=k) = Phi((k+0.5-mu)/sigma) - Phi((k-0.5-mu)/sigma), with cell 0
    absorbing everything below 0.5 and cell 75 absorbing the upper tail.
    """
    mu = float(mu)
    sigma = max(float(sigma), 1e-9)
    hi = stats.norm.cdf((GRID + 0.5 - mu) / sigma)
    lo = stats.norm.cdf((GRID - 0.5 - mu) / sigma)
    lo[0] = 0.0
    hi[-1] = 1.0
    pmf = np.clip(hi - lo, 0.0, None)
    pmf /= pmf.sum()
    return pmf


def _nb_r(mu: float, sigma: float) -> float:
    """NB dispersion r from mean/variance: Var = mu + mu^2/r → r = mu^2/(Var-mu).

    When sigma^2 <= mu the NB is degenerate (Poisson limit): return a large r
    (numerically Poisson). NFL regime (mu ~ 15-30, sigma ~ 4-9) is well
    inside the valid region.
    """
    var = max(float(sigma) ** 2, float(mu) + 1e-9)
    if var <= float(mu):
        return 1e6
    return float(mu) ** 2 / (var - float(mu))


def nb_pmf(mu: float, sigma: float) -> np.ndarray:
    """Negative-binomial PMF on GRID with mean mu, dispersion linked to sigma."""
    mu = max(float(mu), 1e-9)
    r = _nb_r(mu, sigma)
    pmf = stats.nbinom.pmf(GRID, r, r / (r + mu))
    pmf = np.clip(pmf, 0.0, None)
    pmf[-1] += 1.0 - pmf.sum()          # upper-tail absorption onto 75
    pmf = np.clip(pmf, 0.0, None)
    pmf /= pmf.sum()
    return pmf


MARGINAL_PMF = {"dn": dn_pmf, "nb": nb_pmf}
FAMILIES = tuple(sorted(MARGINAL_PMF))


def marginal_breakpoints(mu: float, sigma: float, family: str) -> np.ndarray:
    """CDF at half-integer boundaries: b[0]=0, b[i]=F(i-0.5), b[76]=1."""
    if family == "dn":
        mu = float(mu)
        sigma = max(float(sigma), 1e-9)
        b = stats.norm.cdf((np.arange(GRID_MAX + 1) - 0.5 - mu) / sigma)
    else:  # nb
        mu = max(float(mu), 1e-9)
        r = _nb_r(mu, sigma)
        b = stats.nbinom.cdf(np.arange(GRID_MAX + 1) - 0.5, r, r / (r + mu))
    b = np.clip(b, 0.0, 1.0)
    b = np.concatenate([[0.0], b, [1.0]])
    # enforce monotone (clip can create plateaus; diff() below stays >= 0)
    b = np.maximum.accumulate(b)
    return b


def marginal_pmf(mu: float, sigma: float, family: str) -> np.ndarray:
    pmf = np.diff(marginal_breakpoints(mu, sigma, family))
    pmf = np.clip(pmf, 0.0, None)
    pmf /= pmf.sum()
    return pmf


def integer_ll(pmf: np.ndarray, actual: float) -> float:
    """Log-likelihood of one actual integer score under a grid PMF."""
    k = int(min(max(float(actual), 0.0), GRID_MAX))
    return float(np.log(max(pmf[k], LL_FLOOR)))


# ── Sigma curves (heteroskedastic; MLB alpha(lambda) analog) ────────────────

def fit_sigma_powerlaw(mu_hat: np.ndarray, resid: np.ndarray
                       ) -> dict[str, float] | None:
    """OLS of log|resid| on log mu_hat → sigma(mu) = sigma0 * mu^q.

    Returns {"spec": "power", "sigma0", "q", "n_fit"} or None when the fit is
    degenerate (no valid points / non-finite params).
    """
    mu = np.asarray(mu_hat, dtype=float)
    r = np.asarray(resid, dtype=float)
    ok = (mu > 0.0) & (np.abs(r) > 0.0) & np.isfinite(mu) & np.isfinite(r)
    x = np.log(mu[ok])
    y = np.log(np.abs(r[ok]))
    if len(x) < 20:
        return None
    q, log_s0 = np.polyfit(x, y, 1)
    s0 = float(np.exp(log_s0))
    if not (np.isfinite(s0) and np.isfinite(q)):
        return None
    return {"spec": "power", "sigma0": round(s0, 4), "q": round(float(q), 4),
            "n_fit": int(len(x))}


def constant_sigma(resid: np.ndarray) -> dict[str, float]:
    r = np.asarray(resid, dtype=float)
    return {"spec": "const", "sigma0": round(float(np.sqrt(np.mean(r ** 2))), 4),
            "q": 0.0, "n_fit": int(len(r))}


def sigma_callable(spec: dict[str, float] | None) -> Any:
    """Callable sigma(mu) from a spec dict (power law or constant)."""
    if not spec or spec.get("spec") == "const":
        s0 = float((spec or {}).get("sigma0", 1.0))
        return lambda mu: float(np.clip(s0, SIGMA_MIN, SIGMA_MAX))
    s0 = float(spec.get("sigma0", 1.0))
    q = float(spec.get("q", 0.0))
    return lambda mu: float(np.clip(
        s0 * np.power(np.maximum(float(mu), 1e-6), q),
        SIGMA_MIN, SIGMA_MAX))


def _side_ll(family: str, sig_spec: dict[str, float] | None,
             mu_hat: np.ndarray, actual: np.ndarray) -> float:
    sig_fn = sigma_callable(sig_spec)
    pmfs = [marginal_pmf(m, sig_fn(m), family) for m in mu_hat]
    return float(sum(integer_ll(p, a) for p, a in zip(pmfs, actual)))


# ── Cross-side dependence ────────────────────────────────────────────────────

def estimate_rho(z_h: np.ndarray, z_a: np.ndarray) -> dict[str, float]:
    """Global Pearson rho on standardized pairs + Fisher's-z CI."""
    zh = np.asarray(z_h, dtype=float)
    za = np.asarray(z_a, dtype=float)
    ok = np.isfinite(zh) & np.isfinite(za)
    zh, za = zh[ok], za[ok]
    n = len(zh)
    if n < 10:
        raise ValueError(f"estimate_rho: need >=10 finite pairs, got {n}")
    r = float(np.corrcoef(zh, za)[0, 1])
    z = float(np.arctanh(np.clip(r, -0.999999, 0.999999)))
    se = 1.0 / np.sqrt(n - 3)
    lo = float(np.tanh(z - 1.96 * se))
    hi = float(np.tanh(z + 1.96 * se))
    return {"rho": round(r, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "n": int(n)}


# ── Gaussian-copula joint PMF on the integer grid ───────────────────────────

def joint_pmf_copula(mu_h: float, mu_a: float, params: dict[str, Any]
                     ) -> np.ndarray:
    """76x76 joint PMF via a Gaussian copula on the per-side marginal PMFs.

    J(i,j) = C(bH[i+1], bA[j+1]) - C(bH[i], bA[j+1]) - C(bH[i+1], bA[j])
             + C(bH[i], bA[j])
    with C(u,v) = Phi_rho(Phi^-1(u), Phi^-1(v)) and half-integer CDF
    breakpoints b (inclusion-exclusion rectangle). Row/col sums equal the
    marginal PMFs exactly (up to float) by construction.
    """
    rho = float(params["rho"])
    family = str(params["family"])
    b_h = marginal_breakpoints(mu_h, sigma_callable(params["sigma_h"])(mu_h),
                               family)
    b_a = marginal_breakpoints(mu_a, sigma_callable(params["sigma_a"])(mu_a),
                               family)
    qh = np.clip(stats.norm.ppf(np.clip(b_h, 1e-12, 1 - 1e-12)), -_Q_CLIP, _Q_CLIP)
    qa = np.clip(stats.norm.ppf(np.clip(b_a, 1e-12, 1 - 1e-12)), -_Q_CLIP, _Q_CLIP)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    mv = multivariate_normal(mean=[0.0, 0.0], cov=cov)
    pts = np.column_stack([np.repeat(qh, len(qa)), np.tile(qa, len(qh))])
    C = np.asarray(mv.cdf(pts)).reshape(len(qh), len(qa))
    J = C[1:, 1:] - C[:-1, 1:] - C[1:, :-1] + C[:-1, :-1]
    J = np.clip(J, 0.0, None)
    J = J / J.sum()
    return J


# ── IPF tie calibration ──────────────────────────────────────────────────────

def calibrate_tie_diagonal(J: np.ndarray, marg_h: np.ndarray,
                           marg_a: np.ndarray, p_tie: float) -> np.ndarray:
    """Calibrate the joint's diagonal mass to ``p_tie`` via IPF.

    Scales the diagonal to sum to ``p_tie`` (relative proportions preserved —
    excess mass flows to near-diagonal cells via the row/col normalization),
    then alternates row-normalize → column-normalize → diagonal-refix until
    row/col sums match the marginals to IPF_TOL. Returns the calibrated
    table (or raises RuntimeError on non-convergence).
    """
    p_tie = float(p_tie)
    J = np.array(J, dtype=float, copy=True)
    J = np.maximum(J, IPF_FLOOR)
    d = np.diag(J).copy()
    s = float(d.sum())
    if s > 0:
        J[np.diag_indices_from(J)] = d * (p_tie / s)
    J = np.maximum(J, IPF_FLOOR)
    for _ in range(IPF_MAX_ITER):
        # rows
        row_s = J.sum(axis=1)
        if row_s.min() <= 0:
            break
        J *= (marg_h / row_s)[:, None]
        # columns
        col_s = J.sum(axis=0)
        if col_s.min() <= 0:
            break
        J *= (marg_a / col_s)[None, :]
        # diagonal refix
        d = np.diag(J).copy()
        ds = float(d.sum())
        if ds > 0:
            J[np.diag_indices_from(J)] = d * (p_tie / ds)
        J = np.maximum(J, IPF_FLOOR)
        err = max(float(np.max(np.abs(J.sum(axis=1) - marg_h))),
                  float(np.max(np.abs(J.sum(axis=0) - marg_a))))
        if err < IPF_TOL:
            return J
    raise RuntimeError(
        f"IPF tie calibration did not converge to {IPF_TOL} "
        f"(p_tie={p_tie:.4f})")


# ── Derived probabilities from a calibrated joint ───────────────────────────

def margin_pmf_from_joint(J: np.ndarray) -> np.ndarray:
    """Margin PMF P_m(m) = sum_i J(i, i-m) via diagonal (trace) sums.

    P_m(m) = trace(J, offset=-m): each margin m corresponds to the cells
    (i, i-m), which form one diagonal of J.
    """
    J = np.asarray(J, dtype=float)
    n = J.shape[0]
    pmf = np.zeros(2 * n - 1)
    for m in range(-(n - 1), n):
        pmf[m + (n - 1)] = np.trace(J, offset=-m)
    return pmf


def total_pmf_from_joint(J: np.ndarray) -> np.ndarray:
    """Total PMF P_t(s) = sum_i J(i, s-i) via anti-diagonal (trace) sums.

    A = J[:, ::-1] mirrors the anti-diagonals onto normal diagonals:
    sum_i J[i, s-i] = trace(A, offset=n-1-s).
    """
    J = np.asarray(J, dtype=float)
    n = J.shape[0]
    A = J[:, ::-1]
    pmf = np.zeros(2 * n - 1)
    for s in range(2 * n - 1):
        pmf[s] = np.trace(A, offset=(n - 1) - s)
    return pmf


def derived_from_joint(J: np.ndarray, L: float | None = None,
                       U: float | None = None) -> dict[str, float]:
    """Derived markets from one calibrated joint PMF (see module docstring)."""
    J = np.asarray(J, dtype=float)
    n = J.shape[0]
    margin_pmf = margin_pmf_from_joint(J)
    total_pmf = total_pmf_from_joint(J)
    p_home = float(np.tril(J, -1).sum())
    p_away = float(np.triu(J, 1).sum())
    p_tie = float(np.trace(J))
    out: dict[str, float] = {
        "p_home_win": round(p_home, 8),
        "p_away_win": round(p_away, 8),
        "p_tie": round(p_tie, 8),
        "derived_ml": round(p_home / (1.0 - p_tie), 8) if p_tie < 1 else 0.5,
    }
    if L is not None:
        out[f"p_cover_{L}"] = round(cover_prob(margin_pmf, L), 8)
    if U is not None:
        out[f"p_over_{U}"] = round(over_prob(total_pmf, U), 8)
    return out


def cover_prob(margin_pmf: np.ndarray, L: float) -> float:
    """P(margin > L) from a margin PMF (half-lines land naturally)."""
    n = (len(margin_pmf) + 1) // 2
    k = int(np.floor(L))
    if k >= n - 1:
        return 0.0
    return float(margin_pmf[n - 1 + k + 1:].sum())


def over_prob(total_pmf: np.ndarray, U: float) -> float:
    """P(total > U) from a total PMF."""
    k = int(np.floor(U))
    if k >= len(total_pmf) - 1:
        return 0.0
    return float(total_pmf[k + 1:].sum())


# ── Discrete CRPS (two independent formulations) ────────────────────────────

def crps_discrete(pmf: np.ndarray, actual: float) -> float:
    """Integer-support CRPS via the CDF-sum identity:
    CRPS(F, x) = sum_k (F(k) - 1{x <= k})^2 over the grid
    (derived from the integral form on half-open unit intervals;
    the degenerate point-mass at m gives exactly |m - x|).
    """
    cdf = np.cumsum(np.asarray(pmf, dtype=float))
    k = int(min(max(float(actual), 0.0), GRID_MAX))
    ind = np.zeros(len(cdf), dtype=float)
    ind[k:] = 1.0                # 1{x <= k} == 1 for k >= x
    return float(np.sum((cdf - ind) ** 2))


def crps_discrete_pairs(pmf: np.ndarray, actual: float) -> float:
    """Independent formulation: CRPS = E|X - x| - 0.5 * E|X - X'| (X, X' iid
    from the forecast PMF), computed exactly by double summation on the grid.
    """
    p = np.asarray(pmf, dtype=float)
    x = float(actual)
    e_abs = float(np.sum(p * np.abs(GRID - x)))
    grid = GRID
    diff = np.abs(grid[:, None] - grid[None, :])
    e_pair = float(np.sum(p[:, None] * p[None, :] * diff))
    return e_abs - 0.5 * e_pair


# ── Param fitting (POOLED OOF ONLY — the sealed leak guard) ─────────────────

def fit_joint_params(pooled: pd.DataFrame) -> dict[str, Any]:
    """Fit the joint-layer params on the POOLED-OOF residual artifact.

    STRUCTURALLY sealed-safe: this function takes only a residual table
    (no y, no game identities beyond the table) and hardcodes the
    ``fit_on`` marker to ``pooled_oof``. ``build_joint_pmfs`` refuses any
    params without that marker (the sealed leak guard). It is impossible to
    refit on sealed rows through this API.
    """
    required = ["pred_home", "pred_away", "resid_home", "resid_away",
                "home_score", "away_score"]
    missing = [c for c in required if c not in pooled.columns]
    if missing:
        raise ValueError(f"fit_joint_params: missing columns {missing}")
    mu_h = pooled["pred_home"].to_numpy(float)
    mu_a = pooled["pred_away"].to_numpy(float)
    r_h = pooled["resid_home"].to_numpy(float)
    r_a = pooled["resid_away"].to_numpy(float)
    y_h = pooled["home_score"].to_numpy(float)
    y_a = pooled["away_score"].to_numpy(float)

    # ---- per-side sigma candidates ----
    sig_h = {"power": fit_sigma_powerlaw(mu_h, r_h),
             "const": constant_sigma(r_h)}
    sig_a = {"power": fit_sigma_powerlaw(mu_a, r_a),
             "const": constant_sigma(r_a)}

    # ---- family x sigma LL table (pooled-OOF marginal integer LL) ----
    table: dict[str, dict[str, float]] = {}
    for fam in FAMILIES:
        for spec_name, spec in (("power", sig_h["power"]), ("const", sig_h["const"])):
            ll_h = _side_ll(fam, spec, mu_h, y_h)
            ll_a = _side_ll(fam, sig_a[spec_name], mu_a, y_a)
            table[f"{fam}_{spec_name}"] = {
                "family": fam, "sigma_spec": spec_name,
                "ll_home": round(ll_h, 3), "ll_away": round(ll_a, 3),
                "ll_total": round(ll_h + ll_a, 3),
            }
    best_key = max(table, key=lambda k: table[k]["ll_total"])
    best_fam = table[best_key]["family"]
    best_sig = table[best_key]["sigma_spec"]

    # DN tiebreak: within DN_TIEBREAK_LL units of the winner, default to DN
    # for grid simplicity (the flagged judgment call — data decides, but NB
    # does not auto-win).
    dn_best = max((k for k in table if table[k]["family"] == "dn"),
                  key=lambda k: table[k]["ll_total"], default=None)
    if dn_best is not None and best_fam != "dn":
        gap = table[best_key]["ll_total"] - table[dn_best]["ll_total"]
        if gap < DN_TIEBREAK_LL:
            best_fam = "dn"
            best_sig = table[dn_best]["sigma_spec"]

    # Per-side sigma winner under the chosen family (by per-side LL) — the
    # sigma seam is designed to accept covariates later (weather sigma-arm).
    ll_h_power = _side_ll(best_fam, sig_h["power"], mu_h, y_h)
    ll_h_const = _side_ll(best_fam, sig_h["const"], mu_h, y_h)
    ll_a_power = _side_ll(best_fam, sig_a["power"], mu_a, y_a)
    ll_a_const = _side_ll(best_fam, sig_a["const"], mu_a, y_a)
    sig_h_spec = sig_h["power"] if ll_h_power >= ll_h_const else sig_h["const"]
    sig_a_spec = sig_a["power"] if ll_a_power >= ll_a_const else sig_a["const"]

    # ---- rho on standardized pairs ----
    sig_h_fn = sigma_callable(sig_h_spec)
    sig_a_fn = sigma_callable(sig_a_spec)
    z_h = r_h / np.array([sig_h_fn(m) for m in mu_h])
    z_a = r_a / np.array([sig_a_fn(m) for m in mu_a])
    rho_info = estimate_rho(z_h, z_a)

    params: dict[str, Any] = {
        "family": best_fam,
        "sigma_h": sig_h_spec,
        "sigma_a": sig_a_spec,
        "rho": rho_info["rho"],
        "rho_ci": {"low": rho_info["ci_low"], "high": rho_info["ci_high"]},
        "rho_n": rho_info["n"],
        "ll_table": table,
        "fit_on": "pooled_oof",
        "grid_max": GRID_MAX,
    }
    return params


def build_joint_pmfs(pred_df: pd.DataFrame, params: dict[str, Any],
                     p_tie: float, require_pooled_fit: bool = True
                     ) -> tuple[np.ndarray, dict[str, Any]]:
    """Build calibrated per-game joint PMFs for a prediction frame.

    Args:
        pred_df: per-game rows with pred_home/pred_away (and home_score/
                 away_score actuals when present — used for CRPS/LL).
        params: joint params from ``fit_joint_params`` (pooled-OOF only).
        p_tie: constant final-tie base rate for the IPF diagonal.
        require_pooled_fit: if True (default), params must carry
            fit_on == "pooled_oof" (the sealed leak guard).

    Returns:
        (pmfs [n, 76, 76] calibrated joints, summary dict with per-game
        derived probabilities and diagnostics).
    """
    if require_pooled_fit and params.get("fit_on") != "pooled_oof":
        raise ValueError(
            "build_joint_pmfs: params not fitted on pooled OOF "
            f"(fit_on={params.get('fit_on')!r}) — sealed refit is forbidden")
    family = str(params["family"])
    sig_h_fn = sigma_callable(params["sigma_h"])
    sig_a_fn = sigma_callable(params["sigma_a"])

    pmfs: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    crps_h: list[float] = []
    crps_a: list[float] = []
    ll_h: list[float] = []
    ll_a: list[float] = []
    d_raw: list[float] = []
    marg_err: list[float] = []

    for _, row in pred_df.iterrows():
        mu_h = float(row["pred_home"])
        mu_a = float(row["pred_away"])
        J = joint_pmf_copula(mu_h, mu_a, params)
        marg_h = marginal_pmf(mu_h, sig_h_fn(mu_h), family)
        marg_a = marginal_pmf(mu_a, sig_a_fn(mu_a), family)
        raw_tie = float(np.trace(J))
        J_cal = calibrate_tie_diagonal(J, marg_h, marg_a, p_tie)
        pmfs.append(J_cal)
        d_raw.append(raw_tie)
        marg_err.append(max(float(np.max(np.abs(J_cal.sum(axis=1) - marg_h))),
                            float(np.max(np.abs(J_cal.sum(axis=0) - marg_a)))))
        derived = derived_from_joint(J_cal)
        rows.append({"game_id": row["game_id"], **derived})
        if "home_score" in pred_df.columns and pd.notna(row.get("home_score")):
            crps_h.append(crps_discrete(marg_h, float(row["home_score"])))
            crps_a.append(crps_discrete(marg_a, float(row["away_score"])))
            ll_h.append(integer_ll(marg_h, float(row["home_score"])))
            ll_a.append(integer_ll(marg_a, float(row["away_score"])))

    summary = {
        "n": int(len(pmfs)),
        "d_raw_mean": round(float(np.mean(d_raw)), 5) if d_raw else None,
        "p_tie_target": round(float(p_tie), 5),
        "d_calibrated_mean": round(float(np.mean(
            [np.trace(p) for p in pmfs])), 8) if pmfs else None,
        "max_marginal_err_post_ipf": round(float(max(marg_err)), 12)
        if marg_err else None,
        "crps_home": round(float(np.mean(crps_h)), 4) if crps_h else None,
        "crps_away": round(float(np.mean(crps_a)), 4) if crps_a else None,
        "ll_home": round(float(np.mean(ll_h)), 4) if ll_h else None,
        "ll_away": round(float(np.mean(ll_a)), 4) if ll_a else None,
    }
    return np.stack(pmfs), {"derived": pd.DataFrame(rows), "summary": summary}


# ── Artifact loading (loud-failure guard) ───────────────────────────────────

def load_residual_artifact(path: str | Path, expected_n: int = ARTIFACT_N_EXPECTED
                           ) -> pd.DataFrame:
    """Load the step-1 residual artifact; FAILS LOUDLY on any deviation.

    Missing columns / empty frame / duplicate game_ids / wrong row count all
    raise RuntimeError — the artifact is the raw material for the joint
    layer, so a silently-bad input must never pass as success.
    """
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"residual artifact missing: {path}")
    df = pd.read_csv(path)
    missing = [c for c in ARTIFACT_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"residual artifact refused: missing columns {missing}")
    if len(df) == 0:
        raise RuntimeError("residual artifact refused: empty frame")
    if df["game_id"].duplicated().any():
        raise RuntimeError(
            f"residual artifact refused: {int(df['game_id'].duplicated().sum())} "
            "duplicate game_ids")
    if len(df) != expected_n:
        raise RuntimeError(
            f"residual artifact refused: expected {expected_n} rows, got {len(df)} "
            f"— this runner is pinned to the step-1 artifact")
    return df