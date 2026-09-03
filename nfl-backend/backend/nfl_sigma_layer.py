"""NFL sigma/dispersion layer (record-only).

Target: totals tail overconfidence and any genuine conditional dispersion,
measured on the ENGINE'S INTERNAL calibration — NOT the external-market
top-bin (that is a model-vs-market disagreement artifact for the future
market layer).

Scope note (verbatim): "the totals top-bin (pred 0.83 vs actual 0.55)
implies z +0.95 vs +0.13, i.e. sigma_true would need to be ~7.5x
sigma_model to close by dispersion — that defect is NOT in scope here;
this layer only fixes internal PIT miscalibration if Step 0 shows one."

DESIGN RULE: NO pooled-OOF static overlay of any kind — every parameter
that touches a scored row is fitted over the folds and transferred to
sealed by median-of-fold, mirroring MLB's median-fold-rounds refit
convention. A pooled-OOF static sigma/gamma applied to a scored row raises
AssertionError (the mean-bias pooled-map failure mode is forbidden).

This module is pure machinery (Step-0 diagnostic helpers, discrete PIT,
uniformity tests, the sigma-inflation sweep, per-game joint rebuilds with
per-game sigma injection, the fold-disciplined Arm-U walk). It NEVER
touches nfl_joint_engine.py / nfl_per_side_engine.py / nfl_era_features.py
— all engine calls are imports of the committed public entrypoints
(joint_pmf_copula / marginal_pmf / calibrate_tie_diagonal /
derived_from_joint / marginal_breakpoints), and the mu walk mirrors the
era module's centered-target walk (the era module's own code path, which
already mirrors oof_per_side). Orchestration, Step-0 stop rules, gates and
the record live in run_nfl_sigma.py.

PIT conventions (deterministic, no RNG):
  side PIT  u = F(y-0.5) + 0.5*P(X=y)   (mid-point PIT on the integer grid)
  total PIT u_T = CDF_T(T-1) + 0.5*P_T(T) from the total marginal PMF
             (convolution of the per-side marginals; after IPF, the joint's
             own total marginal is used — see build_joints_per_game_sigma).
  Uniform calibration: KS + 10-bin chi-square + mean/sd; ECE = mean
  |bin share - 0.1| over fixed deciles.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nfl_joint_engine import (  # noqa: E402 — committed engine entrypoints
    LL_FLOOR, calibrate_tie_diagonal, derived_from_joint,
    joint_pmf_copula, marginal_breakpoints, marginal_pmf,
)
from nfl_per_side_engine import LGB_PARAMS, _fit_side  # noqa: E402

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Gamma inflation grid for the 0d sweep / Arm-U inner-CV selection.
GAMMA_GRID = np.arange(1.0, 2.00001, 0.1)   # 1.0 .. 2.0 step 0.1
DEFAULT_GAMMA = 1.0

# PIT uniformity tables.
PIT_N_BINS = 10
# ECE here = mean |decile share - 0.1| (uniformity, not binary reliability).
UNIFORM_ECE_CLEAN = 0.03    # primary pass threshold (total PIT)
UNIFORM_CHI2_P = 0.05       # alt pass: chi2 p >= 0.05 in the failing stratum

# Internal totals grid for the totals-ECE check (no market information).
INTERNAL_TOTAL_THRESHOLDS = [42.5, 47.5, 52.5]

# Arm-U inner chronological split (PIT-optimal gamma per fold, per side).
INNER_FRAC = 0.8
INNER_MIN_VAL = 25          # Tv smaller than this → gamma falls back to 1.0

# Arm-C sigma clip (anti centering-cost guard): [CLIP_LO, CLIP_HI] x median
# per-fold sigma_const per side (spec).
CLIP_LO, CLIP_HI = 0.5, 2.0

SIDES = ("home", "away")
SCORE_COL = {"home": "home_score", "away": "away_score"}
PRED_COL = {"home": "pred_home", "away": "pred_away"}
RESID_COL = {"home": "resid_home", "away": "resid_away"}
# resid = actual − pred (artifact convention; negative ⇒ prediction HIGH).
SIGMA_CONST_COL = {"home": "sigma_const_home", "away": "sigma_const_away"}
GAMMA_COL = {"home": "gamma_home", "away": "gamma_away"}
SIGMA_COL = {"home": "sigma_home", "away": "sigma_away"}


# ── Discrete PIT + uniformity tables ─────────────────────────────────────────
#
# ENGINE GRID-CONVENTION NOTE (flagged finding, recorded in run_nfl_sigma):
# nfl_joint_engine.marginal_breakpoints places its breakpoints at
# (arange(76) − 0.5) then prepends 0 / appends 1, which makes its cell k
# hold the mass of SCORE k−1 (argmax of marginal_pmf(25, 9, "dn") sits at
# index 26). Its own dn_pmf and docstrings use the textbook convention
# (cell k ↔ score k). The engine's actual-score lookups (integer_ll,
# crps_discrete) index at int(actual), so they read the mass of score
# y−1 for observed y: per-side LL/CRPS are slightly degraded and derived
# totals carry a systematic +2 shift (margins/tie/ML are shift-invariant
# because they use index DIFFERENCES). The PIT gates below therefore use
# the DOCUMENTED convention (cell k ↔ score k) — the fair test of the
# model's mu/sigma; engine-convention PIT is reported for the record.


def dn_mass_correct(mu: float, sigma: float, k: int) -> float:
    """Textbook DN cell mass P(round(N(mu, sigma)) = k):
    Phi((k+0.5-mu)/s) - Phi((k-0.5-mu)/s), k=0 clamps below -0.5,
    k=75 absorbs the upper tail. Mirrors the engine's dn_pmf.
    """
    if k <= 0:
        return float(stats.norm.cdf((0.5 - mu) / max(sigma, 1e-9)))
    if k >= 75:
        return float(1.0 - stats.norm.cdf((74.5 - mu) / max(sigma, 1e-9)))
    return float(stats.norm.cdf((k + 0.5 - mu) / max(sigma, 1e-9))
                 - stats.norm.cdf((k - 0.5 - mu) / max(sigma, 1e-9)))


def dn_marginal_correct_vec(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """(n, 76) documented-convention DN mass matrix, vectorized over games
    (the PIT hot path — scalar scipy cdf loops dominate Step 0 otherwise)."""
    mu = np.asarray(mu, dtype=float)[:, None]
    sigma = np.maximum(np.asarray(sigma, dtype=float)[:, None], 1e-9)
    k = np.arange(76)[None, :]
    hi = stats.norm.cdf((k + 0.5 - mu) / sigma)
    lo = stats.norm.cdf((k - 0.5 - mu) / sigma)
    lo[:, 0] = 0.0
    hi[:, -1] = 1.0
    pmf = np.clip(hi - lo, 0.0, None)
    pmf /= pmf.sum(axis=1, keepdims=True)
    return pmf


def side_pit(mu: np.ndarray, sigma: np.ndarray, actual: np.ndarray,
             family: str = "dn") -> np.ndarray:
    """Mid-point PIT u = F(y−1) + 0.5·P(X=y) under the DOCUMENTED DN
    convention (cell k ↔ score k). Uniform iff mu/sigma are calibrated.
    Vectorized (family "dn" — the joint engine's chosen family)."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    y = np.clip(np.asarray(actual, dtype=float), 0.0, 75.0).astype(int)
    pmf = dn_marginal_correct_vec(mu, sigma)
    cum = np.cumsum(pmf, axis=1)
    n = len(mu)
    f_prev = np.where(y > 0, cum[np.arange(n), y - 1], 0.0)
    p_y = pmf[np.arange(n), y]
    return f_prev + 0.5 * p_y


def side_pit_engine_convention(mu: np.ndarray, sigma: np.ndarray,
                               actual: np.ndarray) -> np.ndarray:
    """PIT as the ENGINE's shifted cells consume actuals (mid-point of the
    cell-y mass, i.e. the mass of score y−1). Reported only, to quantify
    the grid-convention artifact (expected mean ≈ 0.5 − 0.4/σ when the
    model is actually calibrated). Vectorized.
    """
    mu = np.asarray(mu, dtype=float)[:, None]
    sigma = np.maximum(np.asarray(sigma, dtype=float)[:, None], 1e-9)
    y = np.clip(np.asarray(actual, dtype=float), 0.0, 75.0).astype(int)
    k = np.arange(76)[None, :]
    b_raw = stats.norm.cdf((k - 0.5 - mu) / sigma)
    b = np.concatenate([np.zeros((len(mu), 1)), b_raw,
                        np.ones((len(mu), 1))], axis=1)  # (n, 78)
    n = len(mu)
    return b[np.arange(n), y] + 0.5 * (b[np.arange(n), y + 1]
                                       - b[np.arange(n), y])


def total_pit(mu_h: np.ndarray, sig_h: np.ndarray, mu_a: np.ndarray,
              sig_a: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Mid-point PIT of the actual total under the independent convolution
    of the two DOCUMENTED-convention DN marginals (rho ≈ 0 ⇒ copula ≈
    independence). Fair test of the joint's total level/dispersion.
    Vectorized: one (n, 76) mass build per side, then per-game convolutions.
    """
    mu_h = np.asarray(mu_h, dtype=float)
    mu_a = np.asarray(mu_a, dtype=float)
    sig_h = np.asarray(sig_h, dtype=float)
    sig_a = np.asarray(sig_a, dtype=float)
    t = np.clip(np.asarray(total, dtype=float), 0.0, 150.0).astype(int)
    ph = dn_marginal_correct_vec(mu_h, sig_h)
    pa = dn_marginal_correct_vec(mu_a, sig_a)
    out = np.empty(len(mu_h), dtype=float)
    for i in range(len(mu_h)):
        pt = np.convolve(ph[i], pa[i])
        cum = np.cumsum(pt)
        k = t[i]
        out[i] = cum[k] - 0.5 * pt[k]
    return out


def uniformity_table(u: np.ndarray, n_bins: int = PIT_N_BINS,
                     label: str = "") -> dict[str, Any]:
    """KS + chi-square + moment table for a PIT vector (uniform iff valid)."""
    u = np.asarray(u, dtype=float)
    ok = np.isfinite(u)
    u = u[ok]
    n = int(len(u))
    if n < 20:
        return {"label": label, "n": n, "ks_stat": None, "ks_p": None,
                "chi2_stat": None, "chi2_p": None, "mean": None, "sd": None,
                "ece": None, "note": "n < 20 — table not scored"}
    ks = stats.kstest(u, "uniform")
    counts, _ = np.histogram(u, bins=n_bins, range=(0.0, 1.0))
    chi2 = stats.chisquare(counts)
    ece = float(np.mean(np.abs(counts / n - 1.0 / n_bins)))
    return {
        "label": label, "n": n,
        "ks_stat": round(float(ks.statistic), 4),
        "ks_p": round(float(ks.pvalue), 4),
        "chi2_stat": round(float(chi2.statistic), 3),
        "chi2_p": round(float(chi2.pvalue), 4),
        "mean": round(float(np.mean(u)), 4),
        "sd": round(float(np.std(u)), 4),
        "ece": round(ece, 4),
        "is_uniform": bool(ks.pvalue > 0.05 and chi2.pvalue > 0.05),
    }


# ── Step-0 dispersion machinery ──────────────────────────────────────────────

def ols_f_test(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Overall F-test (H0: all slopes 0) for an n×(k) design, intercept col
    already included as the first column. Manual OLS — no statsmodels."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    n, k = X.shape
    if n < k + 5 or np.linalg.matrix_rank(X) < k:
        return {"n": int(n), "k": int(k), "r2": None, "f": None, "p": None,
                "note": "degenerate design"}
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    df1, df2 = k - 1, n - k
    f = (r2 / df1) / ((1.0 - r2) / df2) if df2 > 0 else np.nan
    p = float(1.0 - stats.f.cdf(f, df1, df2)) if np.isfinite(f) else np.nan
    return {"n": int(n), "k": int(k), "r2": round(float(r2), 4),
            "f": round(float(f), 3) if np.isfinite(f) else None,
            "p": round(p, 4) if np.isfinite(p) else None}


def _design_with_controls(df: pd.DataFrame, features: list[str],
                          mu_hat: np.ndarray) -> np.ndarray:
    X = df[features].to_numpy(float)
    X = np.column_stack([np.ones(len(df)), X, mu_hat, mu_hat ** 2])
    X = np.nan_to_num(X, nan=0.0)
    return X


def dispersion_screen(df: pd.DataFrame, features: list[str], side: str
                      ) -> dict[str, Any]:
    """0c per-side: |resid| and resid² on (12-pool + mu_hat + mu_hat²),
    F-test each; Levene across mu_hat terciles."""
    r = df[RESID_COL[side]].to_numpy(float)
    mu = df[PRED_COL[side]].to_numpy(float)
    X = _design_with_controls(df, features, mu)
    out: dict[str, Any] = {"side": side, "n": int(len(df)), "k": int(X.shape[1]),
                           "abs_resid": ols_f_test(X, np.abs(r)),
                           "resid_sq": ols_f_test(X, r ** 2)}
    # Levene on |resid| across mu_hat terciles (equal-count qcuts).
    ok = np.isfinite(mu) & np.isfinite(r)
    m, rr = mu[ok], np.abs(r[ok])
    if len(m) >= 60:
        qs = np.quantile(m, [1 / 3, 2 / 3])
        groups = [rr[m <= qs[0]], rr[(m > qs[0]) & (m <= qs[1])],
                  rr[m > qs[1]]]
        lev = stats.levene(*groups)
        out["levene_mu_terciles"] = {
            "stat": round(float(lev.statistic), 3),
            "p": round(float(lev.pvalue), 4)}
    else:
        out["levene_mu_terciles"] = {"stat": None, "p": None}
    return out


def split_half_screen(df: pd.DataFrame, candidates: list[str], side: str,
                      n_halves: int = 8, min_r: float = 0.05
                      ) -> dict[str, Any]:
    """0c feature screen: a candidate qualifies iff its per-half corr with
    |resid| has one stable sign in >= 6 of 8 chronological halves with
    |r| >= 0.05 in those halves."""
    r = np.abs(df[RESID_COL[side]].to_numpy(float))
    n = len(df)
    out: dict[str, Any] = {"side": side, "rule": (
        f"sign stable in >= {max(1, n_halves - 2)} of {n_halves} "
        f"chronological split-halves with |r| >= {min_r}")}
    order = np.argsort(pd.to_datetime(df["gameday"]).to_numpy(), kind="stable")
    edges = np.linspace(0, n, n_halves + 1).astype(int)
    per_cand: dict[str, Any] = {}
    for cand in candidates:
        if cand not in df.columns:
            continue
        v = df[cand].to_numpy(float)
        half_corrs = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            idx = order[lo:hi]
            vv, rr = v[idx], r[idx]
            ok = np.isfinite(vv) & np.isfinite(rr)
            if ok.sum() < 10:
                half_corrs.append(None)
                continue
            c = float(np.corrcoef(vv[ok], rr[ok])[0, 1])
            half_corrs.append(c if np.isfinite(c) else None)
        finite = [c for c in half_corrs if c is not None]
        if not finite:
            per_cand[cand] = {"qualifies": False, "note": "no finite halves",
                              "half_corrs": half_corrs}
            continue
        overall = float(np.corrcoef(
            v[np.isfinite(v) & np.isfinite(r)],
            r[np.isfinite(v) & np.isfinite(r)])[0, 1]) if (
                np.isfinite(v) & np.isfinite(r)).sum() > 10 else 0.0
        sign = 1.0 if np.mean(finite) >= 0 else -1.0
        strong = [c for c in finite if sign * c >= min_r]
        qualifies = bool(len(strong) >= max(1, n_halves - 2)
                         and abs(overall) >= min_r)
        per_cand[cand] = {
            "overall_r": round(overall, 4),
            "sign": "pos" if sign > 0 else "neg",
            "n_same_sign_strong": int(len(strong)),
            "qualifies": qualifies,
            "half_corrs": [round(c, 4) if c is not None else None
                           for c in half_corrs]}
    out["candidates"] = per_cand
    out["qualified"] = [c for c, d in per_cand.items() if d.get("qualifies")]
    return out


# ── 0d sigma-inflation sweep ─────────────────────────────────────────────────

def gamma_sweep_total_pit_ece(mu_h: np.ndarray, sig_h0: float,
                              mu_a: np.ndarray, sig_a0: float,
                              total: np.ndarray,
                              gammas: np.ndarray = GAMMA_GRID
                              ) -> list[dict[str, Any]]:
    """Total-PIT ECE vs per-side gamma inflation (sigma × gamma both sides).

    Diagnostic only — a pooled gamma is NEVER applied to scored rows (the
    fold-disciplined Arm-U per-fold gamma replaces this pooled curve).
    """
    rows = []
    for g in gammas:
        u = total_pit(mu_h, np.full(len(mu_h), float(sig_h0 * g)),
                      mu_a, np.full(len(mu_a), float(sig_a0 * g)), total)
        t = uniformity_table(u)
        rows.append({"gamma": round(float(g), 2), "total_pit_ece": t["ece"],
                     "pit_mean": t["mean"], "pit_sd": t["sd"],
                     "chi2_p": t["chi2_p"]})
    return rows


def clean_gamma_minimum(sweep: list[dict[str, Any]],
                        min_ece_drop: float = 0.005) -> dict[str, Any]:
    """Clean-minimum rule: the argmin must beat gamma=1.0 by >= min_ece_drop
    (else flat ⇒ uniform-scale story falsified). Tie → smallest gamma."""
    if not sweep:
        return {"clean": False, "reason": "empty sweep"}
    eces = np.array([r["total_pit_ece"] for r in sweep])
    g0 = eces[0]
    best_i = int(np.argmin(eces))
    best = sweep[best_i]["gamma"]
    drop = float(g0 - eces[best_i])
    clean = bool(drop >= min_ece_drop and best > 1.0)
    return {"clean": clean, "argmin_gamma": float(best),
            "ece_at_1_0": round(float(g0), 4),
            "ece_at_argmin": round(float(eces[best_i]), 4),
            "drop_vs_gamma1": round(drop, 4),
            "reason": ("clean interior/greater-than-1 minimum with "
                       f">= {min_ece_drop} ECE drop" if clean else
                       f"flat or < {min_ece_drop} drop at gamma > 1 — "
                       "uniform-scale story not supported")}


# ── Arm U: fold-disciplined sigma_const + per-fold PIT-optimal gamma ─────────

def rmse(resid: np.ndarray) -> float:
    r = np.asarray(resid, dtype=float)
    return float(np.sqrt(np.mean(r ** 2)))


def clip_sigma_to_anchor(pred_sigma: float | np.ndarray,
                         median_sigma_const: float) -> np.ndarray | float:
    """Arm-C anti centering-cost guard: predicted sigma clipped to
    [CLIP_LO, CLIP_HI] x median per-fold sigma_const (spec). A pure helper
    — Arm C only runs if a future Step-0 screen justifies it.
    """
    lo = float(median_sigma_const) * CLIP_LO
    hi = float(median_sigma_const) * CLIP_HI
    return np.clip(pred_sigma, lo, hi)


def median_of_fold_transfer(fold_stats: list[dict[str, float]], key: str
                            ) -> float:
    """Median over folds of a per-fold train-only statistic (spec: median
    of the fold distribution transfers to sealed; product of the two
    medians — median(gamma) x median(sigma_const) per side)."""
    vals = [float(f[key]) for f in fold_stats if np.isfinite(f.get(key))]
    if not vals:
        raise ValueError(f"median_of_fold_transfer: no finite {key} values")
    return float(np.median(vals))


def _inner_gamma(df_tr: pd.DataFrame, features: list[str], side: str,
                 sigma_const: float, family: str = "lgb"
                 ) -> tuple[float, dict[str, Any]]:
    """PIT-optimal gamma for one fold on its TRAIN split via a
    chronological inner split (first INNER_FRAC rows train, the rest val).
    Returns (gamma, info). gamma falls back to 1.0 when the inner val is
    too small or the mu fit fails."""
    if len(df_tr) < 60 or sigma_const <= 0:
        return DEFAULT_GAMMA, {"fallback": "train < 60 or sigma_const<=0"}
    from nfl_era_features import CENTER_AWAY, CENTER_HOME  # noqa: PLC0415
    cc = CENTER_HOME if side == "home" else CENTER_AWAY
    sc = SCORE_COL[side]
    cc_arr = df_tr[cc].to_numpy(float)
    d = df_tr.sort_values("gameday").reset_index(drop=True)
    cut = int(np.ceil(INNER_FRAC * len(d)))
    ti, tv = d.iloc[:cut], d.iloc[cut:]
    cols = features + [sc, cc]
    ti = ti[cols].dropna()
    tv = tv[cols].dropna()
    if len(ti) < 30 or len(tv) < INNER_MIN_VAL:
        return DEFAULT_GAMMA, {"fallback": f"inner split too small "
                                           f"(ti={len(ti)}, tv={len(tv)})"}
    X_ti = ti[features].to_numpy(float)
    X_tv = tv[features].to_numpy(float)
    y_ti = ti[sc].to_numpy(float) - ti[cc].to_numpy(float)
    y_tv = tv[sc].to_numpy(float) - tv[cc].to_numpy(float)
    try:
        _m, pred_c, _b = _fit_side(family, X_ti, y_ti, X_tv, y_tv)
    except Exception:  # noqa: BLE001
        return DEFAULT_GAMMA, {"fallback": "inner mu fit failed"}
    mu_tv = pred_c + tv[cc].to_numpy(float)
    best_g, best_ece = DEFAULT_GAMMA, None
    for g in GAMMA_GRID:
        u = side_pit(mu_tv, np.full(len(mu_tv), sigma_const * float(g)),
                     tv[sc].to_numpy(float))
        e = uniformity_table(u)["ece"]
        if best_ece is None or e < best_ece:
            best_ece, best_g = e, float(g)
    return best_g, {"n_inner_train": int(len(ti)), "n_inner_val": int(len(tv)),
                    "gamma_ece": round(float(best_ece), 4)}


def arm_u_walk(folds: list[dict], features: list[str],
               frame: pd.DataFrame, family: str = "lgb"
               ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Era-centered mu walk + fold-local sigma machinery (Arm U).

    Per fold, per side (center columns must be attached to ``frame`` and
    the folds built from it — mirror of oof_centered_per_side):
      - mu model on (target − center) at the fold's own best iter;
        val preds + resid exactly as the era walk (same folds, features,
        seed ⇒ byte-identical mu table to C0).
      - sigma_const = RMSE of the fold-TRAIN residuals at best iter
        (in-sample; gamma compensates scale via the inner CV).
      - gamma PIT-optimal on a chronological inner split of the fold's
        train (never touches val / sealed rows).
    Returns (out, info): out carries one row per covered val game with
    [game_id, fold_idx, pred/resid/best_iter per side, sigma_const_side,
    gamma_side, sigma_side = gamma × sigma_const]. info carries the fold
    distributions (min/median/max per side) + rounds + n.
    """
    from nfl_era_features import (  # noqa: PLC0415
        CENTER_AWAY, CENTER_HOME, CENTER_COLS)
    missing_cols = [c for c in CENTER_COLS if c not in frame.columns]
    if missing_cols:
        raise ValueError(f"arm_u_walk: frame missing center columns "
                         f"{missing_cols}")

    parts: list[pd.DataFrame] = []
    fold_stats: list[dict[str, float]] = []
    best_h: list[int] = []
    best_a: list[int] = []
    n_skipped = 0

    for i, f in enumerate(folds):
        tr = f["train"].copy()
        va = f["val"].copy()
        tr_max = pd.to_datetime(tr["gameday"]).max()
        va_min = pd.to_datetime(va["gameday"]).min()
        if not (tr_max < va_min):
            raise AssertionError(
                f"fold {i}: train max {tr_max} not strictly before "
                f"val min {va_min} → leakage-safe split violated")

        id_cols = [c for c in ("game_id", "gameday") if c in va.columns]
        cols = features + [SCORE_COL["home"], SCORE_COL["away"]] + CENTER_COLS
        tr_valid = tr[cols].dropna()
        va_valid = va[id_cols + cols].dropna()
        if len(tr_valid) < 30 or len(va_valid) < 5:
            logger.warning("sigma layer: fold %d too small (tr=%d, va=%d), skipping",
                           i, len(tr_valid), len(va_valid))
            n_skipped += 1
            continue

        X_tr = tr_valid[features].to_numpy(float)
        X_va = va_valid[features].to_numpy(float)
        rec: dict[str, Any] = {"game_id": va_valid["game_id"].values,
                               "fold_idx": i}
        fs: dict[str, float] = {"fold_idx": i}
        for side in SIDES:
            sc = SCORE_COL[side]
            cc = (CENTER_HOME if side == "home" else CENTER_AWAY)
            c_tr = tr_valid[cc].to_numpy(float)
            c_va = va_valid[cc].to_numpy(float)
            y_tr_c = tr_valid[sc].to_numpy(float) - c_tr
            y_va_c = va_valid[sc].to_numpy(float) - c_va
            _m, pred_c, best = _fit_side(family, X_tr, y_tr_c, X_va, y_va_c)
            pred = np.round(pred_c + c_va, 4)
            rec[PRED_COL[side]] = pred
            rec[RESID_COL[side]] = np.round(
                va_valid[sc].to_numpy(float) - pred, 4)
            rec[f"best_iter_{side}"] = best
            (best_h if side == "home" else best_a).append(best)

            # Fold-TRAIN residual RMSE at the fold's best iter (in-sample;
            # early-stopped models predict at their best iteration).
            pred_tr_c = np.asarray(_m.predict(X_tr), dtype=float)
            resid_tr = (tr_valid[sc].to_numpy(float)
                        - (pred_tr_c + c_tr))
            s_const = rmse(resid_tr)
            rec[SIGMA_CONST_COL[side]] = round(s_const, 4)
            gamma, ginfo = _inner_gamma(tr_valid, features, side, s_const,
                                        family)
            rec[GAMMA_COL[side]] = gamma
            rec[SIGMA_COL[side]] = round(gamma * s_const, 4)
            fs[SIGMA_CONST_COL[side]] = s_const
            fs[GAMMA_COL[side]] = gamma
            fs[f"n_train_{side}"] = float(len(tr_valid))
        parts.append(pd.DataFrame(rec))
        fold_stats.append(fs)

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["game_id", "fold_idx"] + [PRED_COL[s] for s in SIDES]
                + [RESID_COL[s] for s in SIDES]
                + [f"best_iter_{s}" for s in SIDES]
                + [SIGMA_CONST_COL[s] for s in SIDES]
                + [GAMMA_COL[s] for s in SIDES]
                + [SIGMA_COL[s] for s in SIDES])
    rounds = {"home": int(np.median(best_h)) if best_h else 2000,
              "away": int(np.median(best_a)) if best_a else 2000}
    dist: dict[str, Any] = {"n_folds": len(fold_stats),
                            "n_skipped": n_skipped}
    for side in SIDES:
        for key in (SIGMA_CONST_COL[side], GAMMA_COL[side]):
            vals = [f[key] for f in fold_stats if f.get(key) is not None]
            dist[f"{key}_fold_dist"] = {
                "min": round(float(np.min(vals)), 4) if vals else None,
                "median": round(float(np.median(vals)), 4) if vals else None,
                "max": round(float(np.max(vals)), 4) if vals else None,
                "n": len(vals)}
    info = {"rounds": rounds, "n_covered": int(len(out)), "fold_dist": dist}
    return out, info


# ── Per-game sigma joint rebuild (engine public functions only) ──────────────

def build_joints_per_game_sigma(rows: pd.DataFrame, rho: float,
                                p_tie: float, family: str = "dn",
                                sigma_h_col: str = SIGMA_COL["home"],
                                sigma_a_col: str = SIGMA_COL["away"],
                                allow_constant: bool = False
                                ) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-game joint PMFs with PER-GAME scalar sigma injection.

    Calls the committed engine's public entrypoints in a per-game loop with
    per-game const-sigma params — build_joint_pmfs's sigma seam is the
    params dict, so per-game override happens here with zero engine edits.
    Requires sigma_h_col/sigma_a_col on ``rows``. By default any constant
    sigma across rows raises AssertionError via the fold-discipline guard
    (see assert_fold_local_sigmas) — Arm U/C scored rows must carry
    per-fold sigmas. ``allow_constant=True`` is reserved for the C0-era
    reproduction, which deliberately re-prices with the ENGINE's own
    pooled-const params (fit_on=pooled_oof convention of the joint
    engine); that path is documented as a reproduction, not a new pooled
    overlay.

    Returns (calibrated pmfs, {derived df with game-level metrics, summary
    incl. per-game total PIT ECE and joint integer LL at (y_h, y_a)}).
    """
    need = ["game_id", "pred_home", "pred_away", sigma_h_col, sigma_a_col]
    missing = [c for c in need if c not in rows.columns]
    if missing:
        raise ValueError(f"build_joints_per_game_sigma: missing {missing}")
    if not allow_constant:
        assert_fold_local_sigmas(rows, sigma_h_col, sigma_a_col)

    pmfs: list[np.ndarray] = []
    out_rows: list[dict[str, Any]] = []
    pit_rows: list[float] = []
    ll_rows: list[float] = []
    ll_rows_corr: list[float] = []
    marg_errs: list[float] = []
    mu_h_all: list[float] = []
    mu_a_all: list[float] = []
    s_h_all: list[float] = []
    s_a_all: list[float] = []
    tot_all: list[float] = []

    for _, row in rows.iterrows():
        mu_h = float(row["pred_home"])
        mu_a = float(row["pred_away"])
        s_h = float(row[sigma_h_col])
        s_a = float(row[sigma_a_col])
        g_params = {"rho": rho, "family": family,
                    "sigma_h": {"spec": "const", "sigma0": s_h},
                    "sigma_a": {"spec": "const", "sigma0": s_a}}
        J = joint_pmf_copula(mu_h, mu_a, g_params)
        marg_h = marginal_pmf(mu_h, s_h, family)
        marg_a = marginal_pmf(mu_a, s_a, family)
        J_cal = calibrate_tie_diagonal(J, marg_h, marg_a, float(p_tie))
        pmfs.append(J_cal)
        marg_errs.append(max(float(np.max(np.abs(J_cal.sum(axis=1) - marg_h))),
                             float(np.max(np.abs(J_cal.sum(axis=0) - marg_a)))))
        d = derived_from_joint(J_cal)
        d["game_id"] = row["game_id"]
        d["sigma_home"] = round(s_h, 4)
        d["sigma_away"] = round(s_a, 4)
        out_rows.append(d)
        mu_h_all.append(mu_h)
        mu_a_all.append(mu_a)
        s_h_all.append(s_h)
        s_a_all.append(s_a)
        if "home_score" in rows.columns and pd.notna(row.get("home_score")):
            yh = int(min(max(float(row["home_score"]), 0.0), 75.0))
            ya = int(min(max(float(row["away_score"]), 0.0), 75.0))
            # Joint LL in the ENGINE's index convention (cell y = mass the
            # engine reads for observed y) and the DOCUMENTED convention
            # (cell y+1 = true mass of score y) — both reported; gates use
            # the documented one (fair internal test).
            ll_rows.append(float(np.log(max(J_cal[yh, ya], LL_FLOOR))))
            ll_rows_corr.append(float(np.log(max(
                J_cal[min(yh + 1, 76), min(ya + 1, 76)], LL_FLOOR))))
            tot_all.append(float(yh + ya))

    pmf_arr = np.stack(pmfs) if pmfs else np.empty((0, 76, 76))
    if tot_all:
        pit_rows = list(total_pit(
            np.asarray(mu_h_all), np.asarray(s_h_all),
            np.asarray(mu_a_all), np.asarray(s_a_all),
            np.asarray(tot_all)))
    derived = pd.DataFrame(out_rows)
    # Engine-schema parity: game_id first (build_joint_pmfs emits it first;
    # the parity sample compares CSV bytes).
    if len(out_rows):
        first = [c for c in out_rows[0] if c != "game_id"]
        derived = derived[["game_id"] + first]
    hist = (np.histogram(pit_rows, bins=10, range=(0, 1))[0]
            if pit_rows else None)
    summary = {
        "n": int(len(pmfs)),
        "max_marginal_err_post_ipf": round(float(max(marg_errs)), 12)
        if marg_errs else None,
        "total_pit": ({"mean": round(float(np.mean(pit_rows)), 4),
                       "ece": round(float(np.mean(np.abs(
                           hist / len(pit_rows) - 0.1))), 4),
                       "chi2_p": round(float(stats.chisquare(
                           hist).pvalue), 4),
                       "ks_p": round(float(stats.kstest(
                           pit_rows, "uniform").pvalue), 4)}
                      if pit_rows else None),
        "joint_ll_mean": round(float(np.mean(ll_rows)), 4) if ll_rows else None,
        "joint_ll_mean_corrected": round(float(np.mean(ll_rows_corr)), 4)
        if ll_rows_corr else None,
        "pit_convention_note": (
            "total PIT + corrected LL use the DOCUMENTED DN convention "
            "(cell k = score k); joint_ll_mean is the engine-index "
            "convention the joint actually prices — see the engine "
            "grid-convention finding in the record"),
    }
    return pmf_arr, {"derived": derived, "summary": summary}


def assert_fold_local_sigmas(df: pd.DataFrame, sig_h_col: str,
                             sig_a_col: str) -> None:
    """FOLD-DISCIPLINE GUARD: every scored row's sigma must come from a
    per-fold value — a single pooled-OOF static sigma/gamma applied across
    folds raises AssertionError (the mean-bias pooled-map failure mode).
    Fold-locality is detected by the sigma column varying across rows
    (constant sigma over a multi-fold frame is a pooled static by
    construction). Rows without a fold identity (sealed transfer) are
    allowed ONLY via the explicit median-of-fold product, which this guard
    cannot distinguish from a pooled static — the runner is responsible for
    carrying the transfer through median_of_fold_transfer and recording it.
    """
    if len(df) < 2:
        return
    n_h = float(df[sig_h_col].nunique())
    n_a = float(df[sig_a_col].nunique())
    if n_h <= 1.0 or n_a <= 1.0:
        raise AssertionError(
            "assert_fold_local_sigmas: sigma column(s) constant across "
            f"{len(df)} rows ({sig_h_col} nunique={n_h:.0f}, "
            f"{sig_a_col} nunique={n_a:.0f}) — a pooled-OOF static sigma "
            "must never be applied to scored rows")


def totals_ece_internal(pmfs: np.ndarray, derived: pd.DataFrame,
                        thresholds: list[float] | None = None
                        ) -> dict[str, Any]:
    """Totals calibration on the INTERNAL CDF grid (no market info):
    for each internal threshold U, mean P(total > U) vs the actual
    exceedance rate over scored rows (requires actual totals on derived)."""
    from nfl_joint_engine import (  # noqa: PLC0415
        over_prob, total_pmf_from_joint)
    thresholds = thresholds or INTERNAL_TOTAL_THRESHOLDS
    if "total" not in derived.columns:
        if "home_score" in derived.columns and "away_score" in derived.columns:
            derived = derived.copy()
            derived["total"] = (derived["home_score"].to_numpy(float)
                                + derived["away_score"].to_numpy(float))
    rows = []
    totals = derived["total"].to_numpy(float)
    for U in thresholds:
        y_over = (totals > U).astype(float)
        p_over = np.array([over_prob(total_pmf_from_joint(J), float(U))
                           for J in pmfs])
        ok = np.isfinite(p_over) & np.isfinite(y_over)
        rows.append({"threshold": float(U),
                     "n": int(ok.sum()),
                     "pred_over_rate": round(float(p_over[ok].mean()), 4)
                     if ok.any() else None,
                     "actual_over_rate": round(float(y_over[ok].mean()), 4)
                     if ok.any() else None,
                     "gap": round(float(np.mean(p_over[ok] - y_over[ok])), 4)
                     if ok.any() else None})
    ece = float(np.mean([abs(r["gap"]) for r in rows
                         if r["gap"] is not None]))
    return {"thresholds": rows, "ece": round(ece, 4)}
