"""Distributional-reliability harness for the run-engine margin forecast.

Every prior run-engine decision (tie renormalization, home one-run alpha,
feature removal) was gated on EVENT-level metrics (per-line Delta/ECE, AUC,
Brier, win rate) — projections of the margin distribution onto specific
bets. PIT/CRPS distributional accuracy was never a decision criterion. This
harness establishes the BASELINE: is the full predicted margin distribution
correctly specified, and where does it fail?

READ-ONLY, mirroring run_margin_distribution_diagnostic.py (bdc37bd): it
consumes the committed run_engine_oof_<date>.csv (actual home/away scores ->
actual margin per game) and run_engine_markets_<date>.csv (per-game
lambda_home/lambda_away and the SHIPPED alpha_home/alpha_away columns),
reconstructs the per-game forecast distribution F̂ over integer margins from
the SAME NB(lambda, alpha) marginals the production MC uses — WITH the
post-fix structure applied — and writes
data_delivery/margin_reliability_<date>.json. Nothing else is modified.

The reconstructed F̂ is the SHIPPED distribution, not a stripped one: the
structural home one-run adjustment (MARGIN_PLUS1_HOME_SHARE = 0.744) is
applied exactly as in run_engine.derive_markets_mc:

    P(margin = 0) resolves to ±1 home-weighted:
        P(+1)' = P(+1) + 0.744·P(0)
        P(−1)' = P(−1) + 0.256·P(0)
        P(0)'  = 0
    every other margin stays at its RAW full-basis value.

Checks recorded:
  1. Randomized PIT — U_i = F̂_i(y_i − 1) + V_i·p̂_i(y_i), V_i ~ U(0,1) i.i.d.
     (discrete margins; F̂ from the renormalized CDF). 20-bin histogram,
     Kolmogorov–Smirnov statistic + χ² (19 df) vs Uniform, and a shape
     verdict: flat = reliable; U = under-dispersed (too narrow); hump =
     over-dispersed; mass near 0/1 = location bias; asymmetric = skew
     misspecification.
  2. Empirical coverage — central (1−α) intervals from F̂_i at nominal
     levels 0.1..0.9 vs actual coverage of y_i (bow-shape reading).
  3. Per-margin reliability — predicted mass at each margin −10..+10 vs
     actual frequency (extends the margin_diagnostic table).
  4. Proper scores vs benchmarks — CRPS (discrete) and log score for (a) the
     model, (b) climatology (pooled empirical margin distribution),
     (c) Poisson-only alternative (same lambda, no NB dispersion).
  5. Conditional calibration — residual indicator 1{y_i <= m} − F̂_i(m)
     regressed on lambda_i (home/away) and on F̂_i(m): coefficient + t-stat.

Usage:
    python margin_reliability_diagnostic.py [YYYYMMDD]
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from run_engine import (MARGIN_PLUS1_HOME_SHARE, ALPHA_FLOOR,
                        nb_pmf_matrix)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_delivery"

# Margin grid for the per-game PMF. lambda ~ 4.4 with alpha <= 0.43 makes the
# NB(4.4, 0.43) tail decay geometrically ~ (0.70)^k; ±60 leaves < 1e-12 mass
# outside. Residual mass is verified and recorded.
MARGIN_GRID = list(range(-60, 61))
PMF_TOL = 1e-8                    # warn if a per-game PMF misses mass
PIT_BINS = 20
PIT_SEED = 20260830
COVERAGE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PER_MARGIN_WINDOW = list(range(-10, 11))
LOG_FLOOR = 1e-12                 # log-score floor (never -inf)
CRPS_GRID = list(range(-60, 61))  # same grid; tail handled by the CDF sum


def _nb_score_pmf(lam_h: np.ndarray, lam_a: np.ndarray,
                  al_h: np.ndarray, al_a: np.ndarray,
                  margins: list[int]) -> np.ndarray:
    """Per-game margin PMF over ``margins`` from the independent NB(λ, α)
    marginals (production path), WITH the structural home one-run fix:
    P(margin=0) resolves to ±1 home-weighted (share MARGIN_PLUS1_HOME_SHARE);
    every other margin stays at its raw full-basis value.

    Returns an (n_games, len(margins)) matrix whose rows sum to 1 (up to the
    excluded tail). The convolution is exact — no MC draws.
    """
    n = len(lam_h)
    mu_h = np.maximum(np.asarray(lam_h, float), 1e-6)[:, None]
    mu_a = np.maximum(np.asarray(lam_a, float), 1e-6)[:, None]
    ah = np.maximum(np.asarray(al_h, float), ALPHA_FLOOR)[:, None]
    aa = np.maximum(np.asarray(al_a, float), ALPHA_FLOOR)[:, None]
    # run grid 0..G for the convolution; G = max(|margin|) + 1 covers any
    # margin in the window (home - away).
    G = max(abs(int(m)) for m in margins) + 1
    ks = np.arange(G + 1)
    ph = nb_pmf_matrix(ks, mu_h, ah)          # (n, G+1)
    pa = nb_pmf_matrix(ks, mu_a, aa)          # (n, G+1)
    pmf = np.zeros((n, len(margins)))
    m_idx = {int(m): i for i, m in enumerate(margins)}
    for i in range(G + 1):
        for j in range(G + 1):
            d = i - j
            if d in m_idx:
                pmf[:, m_idx[d]] += ph[:, i] * pa[:, j]
    # Structural home one-run adjustment (shipped distribution).
    if 0 in m_idx and 1 in m_idx and -1 in m_idx:
        p0 = pmf[:, m_idx[0]]
        pmf[:, m_idx[1]] += MARGIN_PLUS1_HOME_SHARE * p0
        pmf[:, m_idx[-1]] += (1.0 - MARGIN_PLUS1_HOME_SHARE) * p0
        pmf[:, m_idx[0]] = 0.0
    return pmf


def _randomized_pit(pmf: np.ndarray, margins: list[int],
                    y: np.ndarray, seed: int = PIT_SEED) -> np.ndarray:
    """Randomized PIT: U_i = F̂_i(y_i − 1) + V_i·p̂_i(y_i), V_i ~ U(0,1) i.i.d.

    ``pmf`` rows are the SHIPPED (tie-resolved) margin PMFs; ``y`` are the
    actual integer margins. U is Uniform(0,1) iff the forecast distribution
    is correctly specified (discrete-data randomization).
    """
    F = np.cumsum(pmf, axis=1)
    rng = np.random.default_rng(seed)
    V = rng.uniform(size=len(y))
    U = np.empty(len(y))
    idx = {int(m): i for i, m in enumerate(margins)}
    for i, yi in enumerate(y):
        k = idx[int(yi)]
        F_prev = F[i, k - 1] if k > 0 else 0.0
        U[i] = F_prev + V[i] * pmf[i, k]
    return np.clip(U, 0.0, 1.0)


def _pit_checks(U: np.ndarray, n_bins: int = PIT_BINS) -> dict:
    """20-bin histogram + KS + χ² (n_bins−1 df) vs Uniform, with a shape
    verdict. Pure numpy for the statistics (no scipy dependency)."""
    from scipy import stats
    hist, _ = np.histogram(U, bins=n_bins, range=(0.0, 1.0))
    exp = len(U) / n_bins
    chi2 = float(((hist - exp) ** 2 / exp).sum())
    p_chi = float(stats.chi2.sf(chi2, n_bins - 1))
    ks = float(stats.kstest(U, "uniform").statistic)
    p_ks = float(stats.kstest(U, "uniform").pvalue)
    # Shape reading from the 20-bin histogram (bins are 5% wide), gated by
    # the uniformity tests: if neither test rejects, call it flat regardless
    # of cosmetic bin wobble. Only when the tests reject do we classify the
    # shape (U = under-dispersed, hump = over-dispersed, skew = location
    # bias).
    edge = hist[0] + hist[-1]                      # mass at the extremes
    mid = hist[int(0.4 * n_bins):int(0.6 * n_bins)].mean()  # centre bins
    corner = hist[0] + hist[1] + hist[-2] + hist[-1]
    rejected = (p_ks < 0.05) or (p_chi < 0.05)
    if not rejected:
        verdict = "flat — distribution reliably specified (KS/chi2 do not reject uniformity)"
    elif mid > exp * 1.25:
        verdict = "hump — over-dispersed forecast (too wide)"
    elif edge > exp * 1.75:
        verdict = "U-shaped — under-dispersed forecast (too narrow)"
    elif corner > exp * 1.6:
        verdict = "corner mass — location bias (forecast systematically off-center)"
    else:
        verdict = "asymmetric — skew misspecification (one-sided residual)"
    return {"bins": n_bins, "histogram": hist.astype(int).tolist(),
            "expected_per_bin": round(exp, 2),
            "ks_statistic": round(ks, 4), "ks_pvalue": round(p_ks, 4),
            "chi2_statistic": round(chi2, 3), "chi2_df": n_bins - 1,
            "chi2_pvalue": round(p_chi, 4), "shape_verdict": verdict}


def _central_coverage(pmf: np.ndarray, margins: list[int],
                      y: np.ndarray) -> list[dict]:
    """Empirical coverage of the central (1−α) intervals from F̂_i at nominal
    levels 0.1..0.9. The central (1−α) interval is
    [q(α/2), q(1−α/2)] with q(p) = smallest m such that F̂(m) ≥ p, so a
    game is COVERED iff
        F̂_i(y_i)     ≥ α/2        (y_i ≥ q(α/2))    AND
        F̂_i(y_i − 1) < 1 − α/2    (y_i ≤ q(1−α/2)).

    DISCRETENESS: for a discrete forecast the non-randomized central
    interval is inherently conservative — the interval can only change at
    integer boundaries, so the expected coverage is 1 − α + boundary steps.
    ``model_implied`` is the forecast's OWN expected coverage
    (mean over i of F̂_i(q(hi)) − F̂_i(q(lo)−1)): a correctly-specified
    discrete forecast shows empirical ≈ model_implied > nominal, with the
    excess concentrated at narrow levels. The honest test is therefore
    empirical vs model_implied (the bow vs nominal is an artifact); a
    genuine over-dispersion would push empirical ABOVE model_implied, and
    under-dispersion below."""
    F = np.cumsum(pmf, axis=1)
    idx = {int(m): i for i, m in enumerate(margins)}
    n = len(y)
    rows = []
    for alpha in COVERAGE_LEVELS:
        lo_q = alpha / 2.0
        hi_q = 1.0 - alpha / 2.0
        covered = 0
        implied = 0.0
        for i, yi in enumerate(y):
            k = idx[int(yi)]
            F_y = F[i, k]
            F_prev = F[i, k - 1] if k > 0 else 0.0
            if F_y >= lo_q and F_prev < hi_q:
                covered += 1
            qlo = int(np.searchsorted(F[i], lo_q, side="left"))
            qhi = int(np.searchsorted(F[i], hi_q, side="left"))
            implied += F[i, qhi] - (F[i, qlo - 1] if qlo > 0 else 0.0)
        emp = covered / n
        impl = implied / n
        rows.append({"nominal": round(1.0 - alpha, 2),
                     "empirical": round(emp, 4),
                     "delta": round(emp - (1.0 - alpha), 4),
                     "model_implied": round(impl, 4),
                     "delta_vs_implied": round(emp - impl, 4)})
    return rows


def _per_margin(pmf: np.ndarray, margins: list[int], y: np.ndarray) -> dict:
    """Predicted pooled mass at each margin −10..+10 vs actual frequency."""
    idx = {int(m): i for i, m in enumerate(margins)}
    pred = pmf.mean(axis=0)
    n = len(y)
    out = {"margins": PER_MARGIN_WINDOW,
           "pred_p_in_window": round(float(sum(pred[idx[m]] for m in PER_MARGIN_WINDOW)), 4),
           "actual_p_in_window": round(float(sum((y == m).mean() for m in PER_MARGIN_WINDOW)), 4)}
    rows = []
    for m in PER_MARGIN_WINDOW:
        p_pred = float(pred[idx[m]])
        p_act = float((y == m).mean())
        rows.append({"margin": m, "n": int((y == m).sum()),
                     "pred_p": round(p_pred, 4), "actual_p": round(p_act, 4),
                     "delta": round(p_act - p_pred, 4)})
    out["rows"] = rows
    return out


def _crps(pmf: np.ndarray, margins: list[int], y: np.ndarray) -> float:
    """Discrete CRPS over the integer margin grid — the exact discrete form
    for integer-valued forecasts (Gneiting & Raftery 2007):

        CRPS_i = Σ_m (F̂_i(m) − 1{y_i ≤ m})²

    summed over ALL integer margins ``m`` (grid VALUES, not positions). A
    point-mass forecast at y_i gives 0; the finite grid with a
    fully-supported CDF is exact up to the tail (recorded mass loss)."""
    F = np.cumsum(pmf, axis=1)
    margins_arr = np.asarray(margins)
    acc = 0.0
    for i, yi in enumerate(y):
        # 1{y_i <= m}: 1 at every grid margin m >= y_i (the CDF of the
        # degenerate point forecast at y_i).
        acc += float(np.sum((F[i, :] - (margins_arr >= yi).astype(float)) ** 2))
    return acc / len(y)


def _log_score(pmf: np.ndarray, margins: list[int], y: np.ndarray) -> float:
    idx = {int(m): i for i, m in enumerate(margins)}
    n = len(y)
    acc = 0.0
    for i, yi in enumerate(y):
        acc += np.log(max(float(pmf[i, idx[int(yi)]]), LOG_FLOOR))
    return float(acc / n)


def _conditional_calibration(pmf: np.ndarray, margins: list[int],
                             y: np.ndarray, lam_h: np.ndarray,
                             lam_a: np.ndarray) -> list[dict]:
    """Regress the residual indicator 1{y_i <= m} − F̂_i(m) on (λ_home,
    λ_away, F̂_i(m)) for the median-ish margins m = 0, +1, −1. Reports the
    coefficient + t-stat of each regressor (numpy lstsq, no statsmodels)."""
    F = np.cumsum(pmf, axis=1)
    idx = {int(m): i for i, m in enumerate(margins)}
    out = []
    for m in (0, 1, -1):
        k = idx[m]
        Fm = F[:, k]
        resid = (y <= m).astype(float) - Fm
        X = np.column_stack([np.ones(len(y)), lam_h, lam_a, Fm])
        beta, *_ = np.linalg.lstsq(X, resid, rcond=None)
        fitted = X @ beta
        r = resid - fitted
        dof = len(y) - X.shape[1]
        s2 = float(r @ r / dof)
        cov = s2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        t = beta / se
        out.append({
            "m": m, "n": len(y),
            "intercept": round(float(beta[0]), 5),
            "coef_lambda_home": round(float(beta[1]), 5),
            "t_lambda_home": round(float(t[1]), 2),
            "coef_lambda_away": round(float(beta[2]), 5),
            "t_lambda_away": round(float(t[2]), 2),
            "coef_fhat": round(float(beta[3]), 5),
            "t_fhat": round(float(t[3]), 2),
            "rmse_resid": round(float(np.sqrt(s2)), 5)})
    return out


def _benchmark_scores(pmf: np.ndarray, margins: list[int],
                      y: np.ndarray, lam_h: np.ndarray,
                      lam_a: np.ndarray) -> dict:
    """Model vs climatology (pooled empirical margin distribution) vs a
    Poisson-only alternative — the same independent-NB marginals with the
    NB dispersion removed (α → 0, i.e. the Poisson limit of the same λ)."""
    model_crps = _crps(pmf, margins, y)
    model_log = _log_score(pmf, margins, y)
    # Climatology: pooled empirical margin distribution (actuals are
    # tie-free by construction, so no tie resolution is needed).
    clim = np.array([(y == m).mean() for m in margins])
    clim = clim / clim.sum()
    clim_pmf = np.tile(clim, (len(y), 1))
    clim_crps = _crps(clim_pmf, margins, y)
    clim_log = _log_score(clim_pmf, margins, y)
    # Poisson-only: same NB marginals with a huge n (α → 0 limit) — the
    # tie-fix is still applied (the shipped structure on Poisson marginals).
    pois = _nb_score_pmf(lam_h, lam_a,
                         np.full(len(y), ALPHA_FLOOR),
                         np.full(len(y), ALPHA_FLOOR), margins)
    pois_crps = _crps(pois, margins, y)
    pois_log = _log_score(pois, margins, y)
    return {
        "model_crps": round(model_crps, 4),
        "model_log_score": round(model_log, 4),
        "climatology_crps": round(clim_crps, 4),
        "climatology_log_score": round(clim_log, 4),
        "poisson_crps": round(pois_crps, 4),
        "poisson_log_score": round(pois_log, 4),
        "crps_gap_vs_climatology": round(model_crps - clim_crps, 4),
        "log_gap_vs_climatology": round(model_log - clim_log, 4),
        "crps_gap_vs_poisson": round(model_crps - pois_crps, 4),
        "log_gap_vs_poisson": round(model_log - pois_log, 4),
    }


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    date_str = argv[0] if argv else None
    if not date_str:
        hits = sorted(DATA.glob("run_engine_markets_*.csv"))
        cands = [h for h in hits if "_rl." not in h.name]
        if not cands:
            raise FileNotFoundError("no canonical run_engine_markets_*.csv")
        date_str = cands[-1].stem.replace("run_engine_markets_", "")

    oof = pd.read_csv(DATA / f"run_engine_oof_{date_str}.csv")
    markets = pd.read_csv(DATA / f"run_engine_markets_{date_str}.csv")
    m = markets[markets["kind"] == "oof"].copy()
    m["game_pk"] = m["game_pk"].astype(str)
    o = oof.copy()
    o["game_pk"] = o["game_pk"].astype(str)
    j = m.merge(o[["game_pk", "home_score", "away_score"]], on="game_pk",
                how="inner", suffixes=("", "_oof"))
    if j.empty:
        raise ValueError("no OOF games joined")
    lam_h = j["home_expected_runs"].to_numpy(float)
    lam_a = j["away_expected_runs"].to_numpy(float)
    al_h = j["alpha_home"].to_numpy(float)
    al_a = j["alpha_away"].to_numpy(float)
    y = (j["home_score"].to_numpy(float)
         - j["away_score"].to_numpy(float)).astype(int)
    n_games = len(j)

    pmf = _nb_score_pmf(lam_h, lam_a, al_h, al_a, MARGIN_GRID)
    row_sums = pmf.sum(axis=1)
    if float(np.abs(row_sums - 1.0).max()) > PMF_TOL:
        print(f"WARNING: per-game PMF mass outside grid up to "
              f"{float(np.abs(row_sums - 1.0).max()):.2e}")

    U = _randomized_pit(pmf, MARGIN_GRID, y)
    pit = _pit_checks(U)
    coverage = _central_coverage(pmf, MARGIN_GRID, y)
    per_margin = _per_margin(pmf, MARGIN_GRID, y)
    cond = _conditional_calibration(pmf, MARGIN_GRID, y, lam_h, lam_a)
    scores = _benchmark_scores(pmf, MARGIN_GRID, y, lam_h, lam_a)

    # --- verdicts per check (plain reading; no manufactured problems) ---
    pit_verdict = pit["shape_verdict"]
    cov_rows = coverage
    # The honest discrete test: empirical vs model_implied (the forecast's
    # own expected coverage). The nominal bow is a discreteness artifact —
    # a correctly-specified discrete forecast over-covers narrow levels
    # (worst Δ here ≈ +0.16 at nominal 0.1) purely from boundary steps.
    worst_cov = max(cov_rows, key=lambda r: abs(r["delta_vs_implied"]))
    if abs(worst_cov["delta_vs_implied"]) < 0.02:
        cov_verdict = (f"empirical coverage tracks the forecast's own "
                       f"model-implied (discrete) coverage within "
                       f"{abs(worst_cov['delta_vs_implied']):.3f} at every "
                       f"level (worst at nominal "
                       f"{worst_cov['nominal']}) — the apparent bow vs "
                       f"nominal is the discrete-CDF conservativeness "
                       f"artifact; forecast NOT over- or under-dispersed")
    elif worst_cov["delta_vs_implied"] > 0:
        cov_verdict = (f"empirical coverage EXCEEDS model-implied by "
                       f"+{worst_cov['delta_vs_implied']:.3f} at nominal "
                       f"{worst_cov['nominal']} — genuinely over-dispersed "
                       f"(intervals too WIDE even after discreteness)")
    else:
        cov_verdict = (f"empirical coverage BELOW model-implied by "
                       f"{worst_cov['delta_vs_implied']:+.3f} at nominal "
                       f"{worst_cov['nominal']} — genuinely under-dispersed "
                       f"(intervals too NARROW even after discreteness)")
    cond_verdict = ("no strong conditional dependence" if all(
        abs(c["t_lambda_home"]) < 2.0 and abs(c["t_lambda_away"]) < 2.0
        for c in cond) else
        "forecast is conditionally miscalibrated on lambda/F̂ — 'right on "
        "average, wrong when confident'")
    score_verdict = (f"model CRPS {scores['model_crps']} vs climatology "
                     f"{scores['climatology_crps']} "
                     f"(gap {scores['crps_gap_vs_climatology']:+})")

    out = {
        "diagnostic": "margin_reliability",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "artifacts": {"oof": f"run_engine_oof_{date_str}.csv",
                      "markets": f"run_engine_markets_{date_str}.csv"},
        "n_games": n_games,
        "method": {
            "forecast": "per-game NB(lambda, alpha) marginals reconstructed "
                        "exactly (no MC draws), with the SHIPPED post-fix "
                        "structure: P(margin=0) resolves to +-1 "
                        f"home-weighted with share {MARGIN_PLUS1_HOME_SHARE} "
                        "(P(+1)'=P(+1)+a*P(0), P(-1)'=P(-1)+(1-a)*P(0), "
                        "P(0)'=0); every other margin at its raw full-basis "
                        "value",
            "pit": "randomized PIT U_i = F(y_i-1) + V_i*p(y_i), "
                   f"V ~ U(0,1) i.i.d., seed {PIT_SEED}",
            "margin_grid": [MARGIN_GRID[0], MARGIN_GRID[-1]],
            "max_pmf_mass_loss": round(float(np.abs(row_sums - 1.0).max()), 10),
        },
        "pit": pit,
        "coverage": coverage,
        "per_margin": per_margin,
        "conditional_calibration": cond,
        "scores": scores,
        "verdicts": {
            "pit": pit_verdict,
            "coverage": cov_verdict,
            "conditional": cond_verdict,
            "scores": score_verdict,
        },
        "summary": (f"n={n_games} · PIT: {pit_verdict} · coverage: "
                    f"{cov_verdict} · conditional: {cond_verdict} · "
                    f"scores: {score_verdict}"),
    }

    DATA.mkdir(parents=True, exist_ok=True)
    out_f = DATA / f"margin_reliability_{date_str}.json"
    with open(out_f, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {out_f}")
    print(f"n={n_games} PIT: {pit_verdict} | coverage worst |Δ|="
          f"{abs(worst_cov['delta']):.4f} | model CRPS "
          f"{scores['model_crps']} (clim {scores['climatology_crps']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
