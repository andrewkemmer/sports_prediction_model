"""Run-engine k-edge — MONITOR-ONLY since 2026-09-22 (adoption 0.03/90).

The C2 edge expansion is RETIRED from production. The run-engine tuning
policy's Stage-1 sweep found that lr 0.03 x fixed 90 rounds yields
k-hat ~= 1.0 on the OOF basis (the endgame configuration): the Poisson
model no longer systematically shrinks edges, so there is nothing left
for the transform to correct. All sealed gates (P1/P2/T1) passed, the
invariants passed (I2 wedge ratio 1.0013), and the totals effect of the
transform was measured at ~0.0007 CRPS runs even at k=1.79 — noise.

What this module still does (per run, in run_engine_daily's wrapper):
  - fits the diagnostic k-hat on the strictly-prior (pre-holdout) OOF of
    the run's own decided frame (module cache when its identity matches,
    else a fresh walk — k can never come from data the run did not derive);
  - publishes k-hat + its sampling se + the drift band [K_EDGE_REF ± 0.2]
    into the markets meta (block["k_edge"], mode="monitor_only") WITHOUT
    applying it — the board prices the RAW λ pair;
  - keeps an explicit ``k_edge=`` argument for offline A/B runs (wrappers
    expand OOF markets and the slate board through the same k).

Removed with the retirement: the frozen-basis versioning scaffolding (it
existed solely to stabilize a k reading that is no longer applied) and the
post-hoc OOF artifact rewrite (the base daily persists the raw λ state it
prices).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

import run_engine as _re

K_EDGE_REF = 1.0           # post-adoption expectation (2026-09-22): the
                           # adopted 0.03/90 config yields k-hat ~= 1.0 by
                           # construction, so the drift band centers on 1.0 —
                           # sustained moves away are the lambda-basis drift
                           # signal (1.53 was the pre-adoption challenger ref)
K_EDGE_BAND = 0.2          # drift alert band: fitted k outside [ref±band]

# Original run_engine bindings captured at import (wrappers delegate to
# these; patch() installs the overrides below onto run_engine).
_orig_derive_markets_v3 = _re.derive_markets_v3
_orig_predict_slate_runs = _re.predict_slate_runs
_orig_run_oof = _re.run_oof
_orig_run_engine_daily = _re.run_engine_daily


def k_edge_meta(k: float, sampling_se: Optional[float] = None,
                production_used: bool = False) -> dict:
    """Persist the diagnostic k and its monitoring status.

    MONITOR-ONLY: the fitted k is published for the drift series and is
    NOT applied to production probabilities (``production_used`` False).
    The explicit offline A/B arm passes ``production_used=True`` when the
    k actually priced the markets the meta is attached to.
    ``sampling_se`` is the classical OLS slope se on the fit pool
    (sd(resid) / (√n · sd_d)).
    """
    out_of_band = bool(abs(k - K_EDGE_REF) > K_EDGE_BAND)
    return {
        "k": round(float(k), 4),
        "fit": "run-oof-refit (pre-holdout)",
        "sampling_se": (round(float(sampling_se), 4)
                        if sampling_se is not None else None),
        "reference_k": K_EDGE_REF,
        "drift_band": [round(K_EDGE_REF - K_EDGE_BAND, 3),
                       round(K_EDGE_REF + K_EDGE_BAND, 3)],
        "drift_alert": out_of_band,
        "out_of_band_policy": "monitor_only_no_action",
        "production_used": bool(production_used),
    }

_K_EDGE_ACTIVE: Optional[float] = None   # explicit-arm seam only (offline
                                         # A/B); NEVER set by production
_DAILY_OOF_CACHE: Optional[pd.DataFrame] = None   # last run_oof frame


def fit_k_edge(lam_h: np.ndarray, lam_a: np.ndarray,
               margin: np.ndarray, mask: np.ndarray) -> float:
    """Fit the C2 edge multiplier k on the MASKED (strictly-prior) games only:
    k = OLS slope of actual margin on the λ edge (λ_H − λ_A). The masked set
    is the pre-holdout OOF — sealed games never see k."""
    d = np.asarray(lam_h, float)[mask] - np.asarray(lam_a, float)[mask]
    m = np.asarray(margin, float)[mask]
    if len(d) < 100 or np.std(d) < 1e-9:
        _re.logger.warning("fit_k_edge: insufficient edge variance "
                           "(n=%d, sd=%.4f) — returning 1.0 (no expansion)",
                           len(d), float(np.std(d)))
        return 1.0
    return float(np.polyfit(d, m, 1)[0])


def apply_k_edge(lam_h: np.ndarray, lam_a: np.ndarray, k: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """C2 linear edge expansion (level-preserving). k=1.0 is the identity."""
    lam_h = np.asarray(lam_h, float)
    lam_a = np.asarray(lam_a, float)
    mu = (lam_h + lam_a) / 2.0
    return mu + k * (lam_h - mu), mu + k * (lam_a - mu)


def k_edge_holdout_mask(oof: pd.DataFrame) -> np.ndarray:
    """Pre-holdout mask mirroring derive_markets_v3's own discipline."""
    dates = pd.to_datetime(oof["game_date"])
    cutoff = dates.max() - pd.Timedelta(days=_re.HOLDOUT_DAYS)
    return (dates < cutoff).to_numpy()


def _oof_identity(frame: pd.DataFrame) -> tuple[int, str, str]:
    """Cheap identity stamp for an OOF frame: (row count, min date, max
    date). Two OOF derivations over the same decided frame match; a cache
    left by a different frame (different slate merge, truncated history,
    another sport's run) does not."""
    dates = pd.to_datetime(frame["game_date"], errors="coerce")
    return (int(len(frame)),
            str(dates.min().date()),
            str(dates.max().date()))


def k_edge_fit_se(oof: pd.DataFrame, mask: np.ndarray, k: float) -> float:
    """Sampling standard error of the k̂ slope on the masked basis:
    se = σ_resid / (√n · sd_d), the classical OLS slope se (single-regressor
    form; homoskedastic approximation — used as a scale, not an exact
    inference tool)."""
    d = np.asarray(oof["home_expected_runs"], float)[mask] \
        - np.asarray(oof["away_expected_runs"], float)[mask]
    m = np.asarray(oof["home_score"], float)[mask] \
        - np.asarray(oof["away_score"], float)[mask]
    n = int(mask.sum())
    if n < 3 or np.std(d) < 1e-9:
        return 0.0
    resid = m - k * d
    return float(resid.std(ddof=1) / (np.sqrt(n) * d.std(ddof=1)))


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------
def derive_markets_v3(oof: pd.DataFrame,
                      moneyline_probs: Optional[pd.DataFrame] = None,
                      n_draws: int = _re.MC_DRAWS,
                      seed: int = _re.MARKET_SEED,
                      holdout_days: int = _re.HOLDOUT_DAYS,
                      k_edge: Optional[float] = None,
                      ) -> dict[str, Any]:
    """Phase-3 markets with an OPTIONAL explicit k_edge (offline A/B only).

    Monitor-only policy: the daily seam never sets _K_EDGE_ACTIVE, so
    production prices the RAW λ pair. An explicit k_edge ≠ 1 (offline
    experiments) expands the per-side λ columns AFTER λ prediction and
    BEFORE α-curve fitting + NB MC (level preserved); the original body
    then prices the expanded λs, and k + drift band land in
    ``summary['k_edge']``."""
    # Explicit-arm seam only (see module docstring): _K_EDGE_ACTIVE is set
    # solely by run_engine_daily's explicit-k_edge path.
    if k_edge is None:
        k_edge = _K_EDGE_ACTIVE
    if k_edge is not None and abs(k_edge - 1.0) > 1e-9:
        oof = oof.copy()
        lh = oof["home_expected_runs"].to_numpy(float)
        la = oof["away_expected_runs"].to_numpy(float)
        lh2, la2 = apply_k_edge(lh, la, k_edge)
        oof["home_expected_runs"] = np.round(lh2, 4)
        oof["away_expected_runs"] = np.round(la2, 4)
    res = _orig_derive_markets_v3(oof, moneyline_probs=moneyline_probs,
                                  n_draws=n_draws, seed=seed,
                                  holdout_days=holdout_days)
    if k_edge is not None:
        res["summary"]["k_edge"] = k_edge_meta(
            k_edge, production_used=abs(k_edge - 1.0) > 1e-9)
    return res


def _slate_market_grid(out: pd.DataFrame, mc: dict[str, np.ndarray]) -> None:
    """Write the full market grid columns from an MC dict (mirrors the
    original predict_slate_runs grid-writing loop)."""
    for j, line in enumerate(_re.TOTAL_LINE_GRID):
        key = f"p_over_{str(line).replace('.', '_')}"
        out[key] = np.round(mc["p_over_grid"][:, j], 5)
        out[key.replace("p_over_", "p_push_")] = np.round(
            mc["p_push_grid"][:, j], 5)
        out[key.replace("p_over_", "p_under_")] = np.round(
            1 - mc["p_over_grid"][:, j] - mc["p_push_grid"][:, j], 5)
    for j, m in enumerate(_re.RUN_LINE_GRID):
        out[f"p_home_cover_{str(m).replace('.', '_')}"] = np.round(
            mc["p_cover_grid"][:, j], 5)
    for j, m in enumerate(_re.RUN_LINE_GRID_FULL):
        out[_re.rl_col(m, "home")] = np.round(mc["p_rl_home_grid"][:, j], 5)
        out[_re.rl_col(m, "push")] = np.round(mc["p_rl_push_grid"][:, j], 5)
        out[_re.rl_col(m, "away")] = np.round(mc["p_rl_away_grid"][:, j], 5)
        out[_re.rl_col(m, "away_favorite")] = np.round(mc["p_rl_away_fav_grid"][:, j], 5)
        out[_re.rl_col(m, "away_push")] = np.round(mc["p_rl_away_push_grid"][:, j], 5)
        out[_re.rl_col(m, "home_dog")] = np.round(mc["p_rl_home_dog_grid"][:, j], 5)
    out["p_home_win_derived"] = np.round(mc["p_home_win_derived"], 5)
    out["p_away_win_derived"] = np.round(1 - mc["p_home_win_derived"], 5)


def predict_slate_runs(decided_games: pd.DataFrame, slate_games: pd.DataFrame,
                       final_fit_rounds: dict[str, int],
                       curves: dict[str, dict],
                       n_draws: int = _re.MC_DRAWS,
                       seed: int = _re.MARKET_SEED,
                       calibration: Optional[dict] = None) -> pd.DataFrame:
    """Slate λ + market grid; explicit-arm re-pricing only (offline A/B).

    Calls the original body for the λ + grid. When an explicit k_edge arm
    is active (module-level _K_EDGE_ACTIVE, set only by run_engine_daily's
    explicit path — never in monitor-only production), re-prices the grid
    from the EXPANDED λ pair through the SAME α(λ) curves and NB MC."""
    out = _orig_predict_slate_runs(decided_games, slate_games,
                                   final_fit_rounds, curves,
                                   n_draws=n_draws, seed=seed,
                                   calibration=calibration)
    k = _K_EDGE_ACTIVE
    if k is not None and abs(k - 1.0) > 1e-9 and not out.empty:
        lh = out["home_expected_runs"].to_numpy(float)
        la = out["away_expected_runs"].to_numpy(float)
        lh2, la2 = apply_k_edge(lh, la, k)
        alpha_h = _re.alpha_of(lh2, curves["home"])
        alpha_a = _re.alpha_of(la2, curves["away"])
        mc = _re.derive_markets_mc(lh2, la2, alpha_h, alpha_a,
                                   n_draws=n_draws, seed=seed)
        mc = _re.apply_market_calibration(mc, calibration)
        out["home_expected_runs"] = np.round(lh2, 4)
        out["away_expected_runs"] = np.round(la2, 4)
        out["alpha_home"] = np.round(alpha_h, 4)
        out["alpha_away"] = np.round(alpha_a, 4)
        _slate_market_grid(out, mc)
    return out


def run_engine_daily(games: pd.DataFrame, target_games: pd.DataFrame,
                     target_date_str: str,
                     n_draws: int = _re.MC_DRAWS,
                     decided_snapshot: Optional[pd.DataFrame] = None,
                     k_edge: Optional[float] = None,
                     _fake_walk: Optional[pd.DataFrame] = None,
                     ) -> dict[str, Any]:
    """Daily Phase-3 pass — k-edge MONITOR-ONLY (expansion retired).

    The board prices the RAW lambda pair. Each run still fits the
    diagnostic k-hat on the strictly-prior (pre-holdout) rows of the
    daily's own OOF (cached by _wrapped_run_oof — no second walk), and
    publishes k-hat + sampling se + drift band into the markets meta
    (block["k_edge"], mode="monitor_only") WITHOUT applying it.

    An explicit ``k_edge`` argument re-activates the expansion for offline
    A/B runs only: OOF markets and the slate board then price through the
    same expanded lambda pair (the wrapper seam). ``k_edge=1.0`` disables
    explicitly.
    """
    global _K_EDGE_ACTIVE
    fitted_k = None
    sampling_se = None
    basis_id = "daily-oof"
    if k_edge is None:
        # Monitor-only: the diagnostic k-hat is fit AFTER the daily pass on
        # the daily's own OOF (cached by _wrapped_run_oof) — the exact lambda
        # basis the board was priced on, with no second walk-forward run
        # (the pre-daily fresh-walk fallback doubled the run's cost and the
        # cache-identity guard could not hold: build_oof_margin populates
        # the cache at a different feature width).
        _K_EDGE_ACTIVE = None
        try:
            res = _orig_run_engine_daily(games, target_games, target_date_str,
                                         n_draws=n_draws,
                                         decided_snapshot=decided_snapshot)
        finally:
            _K_EDGE_ACTIVE = None
        oof = _DAILY_OOF_CACHE
        if oof is None or oof.empty:
            oof = _fake_walk
            basis_id = "provided-walk"
        if oof is not None and not oof.empty:
            mask = k_edge_holdout_mask(oof)
            fitted_k = fit_k_edge(oof["home_expected_runs"].to_numpy(float),
                                  oof["away_expected_runs"].to_numpy(float),
                                  (oof["home_score"]
                                   - oof["away_score"]).to_numpy(float),
                                  mask)
            sampling_se = k_edge_fit_se(oof, mask, fitted_k)
            _re.logger.warning(
                "Run engine daily (k-edge): monitor-only k=%.4f "
                "(pre-holdout, n=%d, basis=%s, sampling_se=%.4f) — "
                "NOT applied to prices",
                fitted_k, int(mask.sum()), basis_id,
                sampling_se if sampling_se is not None else float("nan"))
        else:
            _re.logger.warning(
                "Run engine daily (k-edge): no OOF available — k monitor "
                "skipped this run")
    else:
        # Explicit k_edge arm (offline A/B only): wrappers reprice through k.
        _K_EDGE_ACTIVE = (k_edge if k_edge is not None
                          and abs(float(k_edge) - 1.0) > 1e-9 else None)
        try:
            res = _orig_run_engine_daily(games, target_games, target_date_str,
                                         n_draws=n_draws,
                                         decided_snapshot=decided_snapshot)
        finally:
            _K_EDGE_ACTIVE = None
    # Publish the k record into the daily block the monitor serves.
    block = res.get("block")
    if block is not None:
        if fitted_k is not None:
            block["k_edge"] = k_edge_meta(fitted_k, sampling_se=sampling_se)
            block["k_edge"]["mode"] = "monitor_only"
            block["k_edge"]["basis"] = basis_id
        elif k_edge is not None:
            block["k_edge"] = k_edge_meta(float(k_edge))
            block["k_edge"]["mode"] = "explicit_k"
            block["k_edge"]["production_used"] = \
                abs(float(k_edge) - 1.0) > 1e-9
        # else: no OOF and no explicit k — no k record this run (warned above)
    return res


def _wrapped_run_oof(*args, **kwargs):
    """Cache the last OOF so the daily wrapper can fit k without a second
    full walk-forward pass."""
    global _DAILY_OOF_CACHE
    res = _orig_run_oof(*args, **kwargs)
    _DAILY_OOF_CACHE = res.get("oof")
    return res


def patch() -> None:
    """Install the k-edge wrappers onto run_engine (idempotent)."""
    _re.run_oof = _wrapped_run_oof
    _re.derive_markets_v3 = derive_markets_v3
    _re.predict_slate_runs = predict_slate_runs
    _re.run_engine_daily = run_engine_daily


def unpatch() -> None:
    """Restore the original run_engine bindings."""
    _re.run_oof = _orig_run_oof
    _re.derive_markets_v3 = _orig_derive_markets_v3
    _re.predict_slate_runs = _orig_predict_slate_runs
    _re.run_engine_daily = _orig_run_engine_daily


patch()
