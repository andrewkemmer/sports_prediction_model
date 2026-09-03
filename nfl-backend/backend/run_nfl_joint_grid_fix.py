"""NFL joint-engine grid-index fix — before/after mechanical delta.

Correctness fix (NOT record-only): ``nfl_joint_engine.marginal_breakpoints``
now places P(score k) at index k, matching ``dn_pmf`` and the documented
convention. Pre-fix the breakpoints sat at F(arange(76) - 0.5) plus
endpoints (cell k held the mass of score k-1): argmax of
``marginal_pmf(25, 9, "dn")`` was 26 (now 25), joints were 77x77 off the
0..75 grid (now 76x76), and every derived total carried a systematic +2
shift (now ~0 — the residual is the DN 0-floor clamp bias).

This runner prints the before/after mechanical delta on the canonical
frame's era-centered pooled-OOF artifact (n=1,091, frame sha
3e8c8a510f04):
  - argmax index (26 -> 25)
  - joint shape (77 -> 76)
  - derived-total mean shift (+2.05 -> +0.07)
  - per-side LL/CRPS movement (corrected marginals read P(actual) at the
    right cell)
The "before" column uses a VERBATIM copy of the pre-fix implementation
(commit 3480b05) embedded below — nothing is refit, nothing re-run.

Per spec, the full joint/seam chain is NOT re-run here (that is the
re-baseline delta's job, next commit). Relative verdicts in the prior
records (joint/era/sigma) stand: every comparison was measured with the
same bug in both arms; only absolute PMF-derived quality numbers
(seam totals ECE, the 0.83-vs-0.55 top bin, derived-ML G4 figures) get
honest post-fix values in the re-baseline.

Usage: cd nfl-backend && python3 backend/run_nfl_joint_grid_fix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nfl_joint_engine as je  # noqa: E402

GRID_MAX = 75
POOLED_ART = "/tmp/nfl_era_e2_pooled.csv"  # era-centered pooled OOF, canonical frame


# ── Verbatim pre-fix implementation (commit 3480b05) for the before column ──

def _old_marginal_breakpoints(mu: float, sigma: float, family: str
                              ) -> np.ndarray:
    """PRE-FIX copy: breakpoints at F(arange(76) - 0.5) + endpoints -> cell k
    held the mass of score k-1 (P(score k) lived at index k+1)."""
    if family == "dn":
        mu = float(mu)
        sigma = max(float(sigma), 1e-9)
        b = stats.norm.cdf((np.arange(GRID_MAX + 1) - 0.5 - mu) / sigma)
    else:  # nb
        mu = max(float(mu), 1e-9)
        r = je._nb_r(mu, sigma)
        b = stats.nbinom.cdf(np.arange(GRID_MAX + 1) - 0.5, r, r / (r + mu))
    b = np.clip(b, 0.0, 1.0)
    b = np.concatenate([[0.0], b, [1.0]])
    b = np.maximum.accumulate(b)
    return b


def _old_marginal_pmf(mu: float, sigma: float, family: str) -> np.ndarray:
    pmf = np.diff(_old_marginal_breakpoints(mu, sigma, family))
    pmf = np.clip(pmf, 0.0, None)
    pmf /= pmf.sum()
    return pmf


def _old_joint(mu_h: float, mu_a: float, params: dict) -> np.ndarray:
    """PRE-FIX joint PMF (old breakpoints through the same copula rectangle)."""
    rho = float(params["rho"])
    family = str(params["family"])
    b_h = _old_marginal_breakpoints(
        mu_h, je.sigma_callable(params["sigma_h"])(mu_h), family)
    b_a = _old_marginal_breakpoints(
        mu_a, je.sigma_callable(params["sigma_a"])(mu_a), family)
    qh = np.clip(stats.norm.ppf(np.clip(b_h, 1e-12, 1 - 1e-12)), -37.0, 37.0)
    qa = np.clip(stats.norm.ppf(np.clip(b_a, 1e-12, 1 - 1e-12)), -37.0, 37.0)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    mv = stats.multivariate_normal(mean=[0.0, 0.0], cov=cov)
    pts = np.column_stack([np.repeat(qh, len(qa)), np.tile(qa, len(qh))])
    C = np.asarray(mv.cdf(pts)).reshape(len(qh), len(qa))
    J = C[1:, 1:] - C[:-1, 1:] - C[1:, :-1] + C[:-1, :-1]
    J = np.clip(J, 0.0, None)
    return J / J.sum()


def _total_mean(J: np.ndarray) -> float:
    s = np.arange(J.shape[0] + J.shape[1] - 1, dtype=float)
    return float((s * je.total_pmf_from_joint(J)).sum())


def main() -> int:
    art = pd.read_csv(POOLED_ART)
    params = je.fit_joint_params(art)  # dn/const sigma 9.663/9.0789, rho 0.0076
    muh = art["pred_home"].to_numpy(float)
    mua = art["pred_away"].to_numpy(float)
    yh = art["home_score"].to_numpy(float)
    ya = art["away_score"].to_numpy(float)
    sig_h = np.array([je.sigma_callable(params["sigma_h"])(m) for m in muh])
    sig_a = np.array([je.sigma_callable(params["sigma_a"])(m) for m in mua])

    # 1) argmax + marginal equality on a score-25 DN
    p_old = _old_marginal_pmf(25.0, 9.0, "dn")
    p_new = je.marginal_pmf(25.0, 9.0, "dn")
    am_old, am_new = int(np.argmax(p_old)), int(np.argmax(p_new))
    max_diff = float(np.max(np.abs(p_old[1:] - p_new)))  # old cell k+1 = score k

    # 2) joint shape + total mean on the pooled mean-pair
    pair = (float(np.mean(muh)), float(np.mean(mua)))
    J_old = _old_joint(*pair, params)
    J_new = je.joint_pmf_copula(*pair, params)
    tot_old, tot_new = _total_mean(J_old), _total_mean(J_new)

    # 3) per-side LL/CRPS, before vs after, pooled OOF
    def ll_crps(pmf_fn):
        ll_h = np.mean([je.integer_ll(pmf_fn(m, s, "dn"), y)
                        for m, s, y in zip(muh, sig_h, yh)])
        ll_a = np.mean([je.integer_ll(pmf_fn(m, s, "dn"), y)
                        for m, s, y in zip(mua, sig_a, ya)])
        cr_h = np.mean([je.crps_discrete(pmf_fn(m, s, "dn"), y)
                        for m, s, y in zip(muh, sig_h, yh)])
        cr_a = np.mean([je.crps_discrete(pmf_fn(m, s, "dn"), y)
                        for m, s, y in zip(mua, sig_a, ya)])
        return ll_h, ll_a, cr_h, cr_a

    before = ll_crps(_old_marginal_pmf)
    after = ll_crps(je.marginal_pmf)

    print("=" * 78)
    print("NFL joint-engine grid-index fix — before/after mechanical delta")
    print("frame sha 3e8c8a510f04 (era-centered pooled OOF, n=%d)" % len(art))
    print("params (fit_joint_params, unchanged): family=%s sigma_h=%s "
          "sigma_a=%s rho=%s" % (params["family"],
                                 params["sigma_h"]["sigma0"],
                                 params["sigma_a"]["sigma0"], params["rho"]))
    print("=" * 78)
    print(f"{'quantity':<42}{'before (bug)':>16}{'after (fix)':>16}")
    print("-" * 78)
    print(f"{'argmax marginal_pmf(25, 9, dn)':<42}{am_old:>16}{am_new:>16}")
    print(f"{'joint shape':<42}{str(J_old.shape):>16}{str(J_new.shape):>16}")
    print(f"{'derived-total mean (muH+muA=%.2f)' % sum(pair):<42}"
          f"{tot_old:>16.4f}{tot_new:>16.4f}")
    print(f"{'total-mean shift vs muH+muA':<42}{tot_old - sum(pair):>16.4f}"
          f"{tot_new - sum(pair):>16.4f}")
    print(f"{'per-side LL home':<42}{before[0]:>16.4f}{after[0]:>16.4f}")
    print(f"{'per-side LL away':<42}{before[1]:>16.4f}{after[1]:>16.4f}")
    print(f"{'per-side CRPS home':<42}{before[2]:>16.4f}{after[2]:>16.4f}")
    print(f"{'per-side CRPS away':<42}{before[3]:>16.4f}{after[3]:>16.4f}")
    print("-" * 78)
    print("read: the +2 totals shift is gone; the residual +%.3f is the DN "
          % (tot_new - sum(pair)))
    print("0-floor clamp bias (E[clamp(round(N(mu,s)),0,75)] > mu), not the")
    print("index bug. Margins/tie/ML (index differences) are shift-invariant")
    print("and unchanged. max|old[k+1]-new[k]| on the mu=25 marginal: %.3e"
          % max_diff)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())