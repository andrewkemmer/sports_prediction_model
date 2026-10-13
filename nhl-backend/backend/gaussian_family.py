"""Standalone split-skew Gaussian mechanics, mirrored in each Kaggle backend.

The two-piece normal has a continuous density at its mode, distinct left
and right scales, and mean mode + sqrt(2/pi)*(right-left). We center that
mean on the score forecast before truncating totals at the zero lattice.
Sport parameters are fixed cold-start priors; fitted parameters use prior
OOF outcomes only. Calibration is a single increasing map for all lines.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.special import expit, logit, ndtr
from scipy.optimize import minimize

FAMILY = "two_sided_asymmetric_gaussian_split_skew"
METHOD = "causal_split_skew_crps_mc"
MC_DRAWS = 50_000
MC_SEED = 42
MC_SE_TARGET = .005
MIN_HISTORY = 200
# Reference total, initial side residual variance/mean, maximum support.
SPORT_PRIORS = {"nba": (220., 1.3, 600), "nfl": (44., 5., 250),
                "nhl": (6., 1., 100), "mlb": (9., 2., 160)}


def defaults(sport):
    _, variance, _ = SPORT_PRIORS[sport]
    return {"home_variance": variance, "away_variance": variance,
            "correlation": 0., "total_skew": 0., "margin_skew": 0.}


def design(h, a, sport):
    reference = SPORT_PRIORS[sport][0]
    return np.column_stack([np.ones(len(h)),
                            np.clip((h + a) / reference - 1., -1., 2.),
                            np.clip(np.abs(h - a) / reference, 0., 1.)])


def support_for(kind, sport):
    if kind not in ("total", "margin"):
        raise ValueError("distribution kind must be total or margin")
    high = SPORT_PRIORS[sport][2]
    return np.arange(0, high + 1) if kind == "total" else np.arange(-high, high + 1)


def fit_variance(h, a, outcomes, sport):
    parameters = defaults(sport)
    if len(h) < MIN_HISTORY:
        return parameters
    home = .5 * (outcomes["total"] + outcomes["margin"])
    away = .5 * (outcomes["total"] - outcomes["margin"])
    eh, ea = home - h, away - a
    # Center covariance: common pace is not confused with side bias.
    ch, ca = eh - eh.mean(), ea - ea.mean()
    vh = np.clip(np.sum(ch ** 2) / np.sum(h), .05, 30.)
    va = np.clip(np.sum(ca ** 2) / np.sum(a), .05, 30.)
    denom = np.sqrt(np.sum(ch ** 2) * np.sum(ca ** 2))
    rho = np.clip(np.sum(ch * ca) / denom, -.9, .9) if denom > 1e-12 else 0.
    parameters.update(home_variance=float(vh), away_variance=float(va),
                      correlation=float(rho))
    for kind, residual in (("total", eh + ea), ("margin", eh - ea)):
        residual = residual - residual.mean()
        left, right = residual[residual < 0], residual[residual >= 0]
        if len(left) >= 20 and len(right) >= 20:
            # Half log scale ratio: left=scale*exp(-skew), right=scale*exp(skew).
            l = np.sqrt(np.mean(left ** 2))
            r = np.sqrt(np.mean(right ** 2))
            parameters[f"{kind}_skew"] = float(np.clip(.5 * np.log(max(r, 1e-9) / max(l, 1e-9)), -.7, .7))
    return parameters


def split_cdf(x, mean, sd, skew):
    """Mean-centered two-piece Gaussian CDF (skew=0 is Gaussian)."""
    left, right = sd * np.exp(-skew), sd * np.exp(skew)
    # Normalize the asymmetric widths to preserve the requested variance.
    variance = left * right + (1. - 2. / np.pi) * (right - left) ** 2
    correction = sd / np.sqrt(variance)
    left, right = left * correction, right * correction
    mode = mean - np.sqrt(2. / np.pi) * (right - left)
    zleft, zright = (x - mode) / left, (x - mode) / right
    return np.where(x <= mode, 2. * left / (left + right) * ndtr(zleft),
                    1. - 2. * right / (left + right) * ndtr(-zright))


def raw_cdf(h, a, kind, sport, parameters=None):
    p = defaults(sport) if parameters is None else parameters
    vh, va = float(p["home_variance"]) * h, float(p["away_variance"]) * a
    rho = float(p["correlation"])
    if np.any(vh <= 0) or np.any(va <= 0) or not -.99 < rho < .99:
        raise ValueError("invalid Gaussian variance/correlation parameters")
    cross = 2. * rho * np.sqrt(vh * va)
    mean = h + a if kind == "total" else h - a
    sd = np.sqrt(np.maximum(vh + va + (cross if kind == "total" else -cross), 1e-9))
    skew = float(p[f"{kind}_skew"])
    # Expand from predictions/parameters only, never from scored outcomes.
    # This preserves refusal checks while accommodating high-variance refits.
    high = max(SPORT_PRIORS[sport][2], int(np.ceil(np.max(np.abs(mean) + 12. * sd * np.exp(abs(skew))))) if len(mean) else 0)
    if high > 10000:
        raise ValueError("Gaussian integration support exceeds resource limit")
    support = np.arange(0, high + 1) if kind == "total" else np.arange(-high, high + 1)
    cdf = split_cdf(support[None, :] + .5, mean[:, None], sd[:, None], skew)
    if kind == "total":
        lower = split_cdf(-.5, mean, sd, skew)
        if np.any(1. - lower < 1e-12):
            raise ValueError("Gaussian total has no nonnegative score mass")
        cdf = (cdf - lower[:, None]) / (1. - lower[:, None])
    else:
        lower = split_cdf(support[0] - .5, mean, sd, skew)
        if np.any(lower > 1e-8):
            raise ValueError("Gaussian lower tail exceeds integration support")
    if np.any(cdf[:, -1] < 1. - 1e-8):
        raise ValueError("Gaussian upper tail exceeds integration support")
    return support, np.clip(cdf, 0., 1.)


def transform(cdf, x, beta):
    beta = np.asarray(beta, float)
    if beta.shape != (9,):
        raise ValueError("split-skew calibration requires nine coefficients")
    z = logit(np.clip(cdf, 1e-12, 1 - 1e-12))
    # Separate positive left/right logit slopes are continuous and increasing
    # at zero. Their ratio adds conditional skew without crossing lines.
    slope = np.exp(np.where(z < 0, (x @ beta[3:6])[:, None], (x @ beta[6:9])[:, None]))
    out = expit(z * slope + (x @ beta[:3])[:, None])
    out[cdf <= 0.] = 0.
    out[cdf >= 1.] = 1.
    return out


def no_tie(cdf, support, sport):
    if sport == "nfl" or support[0] >= 0:
        return cdf
    out = cdf.copy()
    j = int(np.searchsorted(support, 0))
    mid = .5 * (out[:, j - 1] + out[:, j])
    out[:, j - 1] = mid
    out[:, j] = mid
    return out


def crps(cdf, support, y):
    y, support, cdf = np.asarray(y, float), np.asarray(support, float), np.asarray(cdf, float)
    if cdf.shape != (len(y), len(support)) or not np.isfinite(y).all():
        raise ValueError("CRPS requires aligned finite outcomes and CDFs")
    if not np.isfinite(cdf).all() or np.any(np.diff(cdf, axis=1) < -1e-9):
        raise ValueError("CRPS requires a finite nondecreasing CDF")
    if np.any(cdf < 0.) or np.any(cdf > 1.):
        raise ValueError("CDF probabilities must be in [0, 1]")
    return (np.square(cdf - (y[:, None] <= support[None, :])).sum(axis=1)
            + np.maximum(support[0] - y, 0.) + np.maximum(y - support[-1] - 1., 0.))


def fit_map(h, a, y, kind, sport, parameters):
    if len(y) < MIN_HISTORY:
        return [0.] * 9
    support, raw = raw_cdf(h, a, kind, sport, parameters)
    x = design(h, a, sport)
    logits = logit(np.clip(raw, 1e-12, 1 - 1e-12))
    target = np.asarray(y)[:, None] <= support[None, :]
    def objective(beta):
        left = logits < 0
        slope = np.exp(np.where(left, (x @ beta[3:6])[:, None], (x @ beta[6:9])[:, None]))
        cdf = expit(logits * slope + (x @ beta[:3])[:, None])
        cdf[raw <= 0.] = 0.; cdf[raw >= 1.] = 1.
        derivative = cdf * (1. - cdf)
        adjusted = no_tie(cdf, support, sport) if kind == "margin" else cdf
        error = adjusted - target
        loss = np.square(error).sum(axis=1).mean()
        if kind == "margin" and sport != "nfl":
            j = int(np.searchsorted(support, 0))
            error[:, j - 1] = error[:, j] = .5 * (error[:, j - 1] + error[:, j])
        g = 2. * error * derivative / len(y)
        width = g * logits * slope
        gradient = np.r_[x.T @ g.sum(axis=1),
                         x.T @ np.where(left, width, 0.).sum(axis=1),
                         x.T @ np.where(left, 0., width).sum(axis=1)]
        return loss + .01 * np.dot(beta, beta), gradient + .02 * beta
    result = minimize(objective, np.zeros(9), jac=True, method="L-BFGS-B",
                      bounds=[(-2., 2.)] * 3 + [(-1.5, 1.5)] * 6,
                      options={"maxiter": 200, "ftol": 1e-8})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"split-skew CRPS calibration failed: {result.message}")
    return result.x.tolist()


def sample_cdf(cdf, support, identities, kind, sport, n_draws=MC_DRAWS, seed=MC_SEED):
    """IID inverse-CDF Monte Carlo with a stable per-game stream.

    Process one game at a time: O(draws + support) memory, not games*draws.
    Identity-keyed streams preserve forecasts under shuffling and batching.
    Histogramming enforces mass conservation and cross-line monotonicity.
    """
    if not isinstance(n_draws, (int, np.integer)) or n_draws < 100:
        raise ValueError("Monte Carlo requires an integer draw count >= 100")
    used = max(int(n_draws), int(np.ceil(.25 / MC_SE_TARGET ** 2)))
    empirical = np.empty_like(cdf)
    for i, identity in enumerate(identities):
        digest = hashlib.blake2b(f"{sport}|{kind}|{seed}|{identity}".encode(), digest_size=16).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        draws = np.searchsorted(cdf[i], rng.random(used), side="right")
        if np.any(draws >= len(support)):
            raise ValueError("Monte Carlo draw exceeds finite support")
        empirical[i] = np.cumsum(np.bincount(draws, minlength=len(support))) / used
    return empirical, {"method": "inverse_calibrated_cdf_monte_carlo", "family": FAMILY,
                       "n_draws": used, "requested_draws": int(n_draws), "seed": int(seed),
                       "mc_se_totals_max": float(.5 / np.sqrt(used)),
                       "reason": "precision_guard" if used != n_draws else "requested",
                       "memory_policy": "one_game_at_a_time"}
