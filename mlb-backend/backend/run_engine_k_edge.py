"""C2 edge expansion — the k machinery for the run engine (challenger 4feff51).

The λ-edge probe showed the run engine retains only ~60% of true margin
spread (actual_margin ≈ 0.014 + 1.66·λ_edge). Phase A of the challenger
ablation picked the C2 linear edge expansion:

    λ'_H = μ + k(λ_H − μ),  λ'_A = μ + k(λ_A − μ),  μ = (λ_H + λ_A)/2

The LEVEL (λ_H + λ_A) is preserved exactly; the EDGE is scaled by k.
k = 1.0 is the identity (current engine).

Stability analysis (2a/2b/2c of the rollout), verified on the production
OOF (run_engine_oof_20260901.csv, 6,829 games 2024-04..2026-08):
  - per-window k (monthly folds) is NOISY: min −0.28, max 2.96, mean
    1.53, sd 0.71 — small windows cannot anchor k.
  - season-sliced k is STABLE and holds per season (production α curves,
    full-season eval): fit-through-2024 → k=1.21 (2025 margin CRPS
    −0.003), fit-through-2025 → k=1.42 (2026 −0.004); totals delta
    ≈ 0.0000 (level preserved).
  - sensitivity: k ∈ {1.3, 1.5, 1.5306, 1.7} is flat (INSENSITIVE):
    sealed margin CRPS 2.3807–2.3870 vs C0 2.3948 (α fit on each arm's
    λ, the production path) — every in-band k beats C0 by 0.008–0.014,
    spread ≤ 0.006. Design: per-run refit on that run's OOF + drift band
    (fitted k ± 0.2 vs reference 1.53) as the alert signal.

This module monkey-patches run_engine at import (call patch()):
  - derive_markets_v3(oof, ..., k_edge=None): expands the OOF λ pair
    BEFORE α-curve fitting + NB MC when k_edge is given; logs k + drift
    band into summary['k_edge'].
  - predict_slate_runs(...): re-prices the slate grid from the EXPANDED
    λ pair through the same curves when the daily seam is active.
  - run_engine_daily(...): the k-edge aware daily engine — fits k on
    this run's OOF pre-holdout games (per-run refit policy), applies it
    to the OOF markets and the slate board, and ALWAYS logs the fitted k
    + drift band into the markets meta. k_edge=1.0 disables explicitly.

Gate discipline: k is fit on the PRE-HOLDOUT OOF only; sealed games
never see it. No isotonic-on-ML recalibration is added here.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

import run_engine as _re

K_EDGE_REF = 1.53          # challenger C2 k on the 2021-2025 pre-sealed OOF
K_EDGE_BAND = 0.2          # drift alert band: fitted k outside [ref±band]

_K_EDGE_ACTIVE: Optional[float] = None   # daily seam; consumed by the slate
                                          # path so OOF + slate price the same
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


def k_edge_meta(k: float) -> dict:
    return {
        "k": round(float(k), 4),
        "fit": "run-oof-refit (pre-holdout)",
        "reference_k": K_EDGE_REF,
        "drift_band": [round(K_EDGE_REF - K_EDGE_BAND, 3),
                       round(K_EDGE_REF + K_EDGE_BAND, 3)],
        "drift_alert": bool(abs(k - K_EDGE_REF) > K_EDGE_BAND),
    }


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------
_orig_derive_markets_v3 = _re.derive_markets_v3
_orig_predict_slate_runs = _re.predict_slate_runs
_orig_run_oof = _re.run_oof
_orig_run_engine_daily = _re.run_engine_daily


def _wrapped_run_oof(*args, **kwargs):
    """Cache the last OOF so the daily wrapper can fit k without a second
    full walk-forward pass."""
    global _DAILY_OOF_CACHE
    res = _orig_run_oof(*args, **kwargs)
    _DAILY_OOF_CACHE = res.get("oof")
    return res


def derive_markets_v3(oof: pd.DataFrame,
                      moneyline_probs: Optional[pd.DataFrame] = None,
                      n_draws: int = _re.MC_DRAWS,
                      seed: int = _re.MARKET_SEED,
                      holdout_days: int = _re.HOLDOUT_DAYS,
                      k_edge: Optional[float] = None,
                      ) -> dict[str, Any]:
    """Phase-3 markets with the optional C2 edge expansion (k_edge).

    When k_edge is not None, the per-side λ columns are expanded AFTER λ
    prediction and BEFORE α-curve fitting + NB MC (level preserved); the
    original body then prices the expanded λs. The k + drift band land in
    ``summary['k_edge']`` so the markets meta ALWAYS records it."""
    # Daily seam: when run_engine_daily is active (_K_EDGE_ACTIVE set), the
    # OOF markets MUST expand with the same k as the slate so both sides of
    # the board price identically. An explicit k_edge argument wins.
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
        res["summary"]["k_edge"] = k_edge_meta(k_edge)
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
    out["p_home_win_derived"] = np.round(mc["p_home_win_derived"], 5)
    out["p_away_win_derived"] = np.round(1 - mc["p_home_win_derived"], 5)


def predict_slate_runs(decided_games: pd.DataFrame, slate_games: pd.DataFrame,
                       final_fit_rounds: dict[str, int],
                       curves: dict[str, dict],
                       n_draws: int = _re.MC_DRAWS,
                       seed: int = _re.MARKET_SEED) -> pd.DataFrame:
    """Slate λ + market grid through the SAME C2 expansion as the OOF side.

    Calls the original body for the λ + grid, then — when the daily seam is
    active (module-level _K_EDGE_ACTIVE) — re-prices the grid from the
    EXPANDED λ pair through the SAME α(λ) curves and NB MC."""
    out = _orig_predict_slate_runs(decided_games, slate_games,
                                   final_fit_rounds, curves,
                                   n_draws=n_draws, seed=seed)
    k = _K_EDGE_ACTIVE
    if k is not None and abs(k - 1.0) > 1e-9 and not out.empty:
        lh = out["home_expected_runs"].to_numpy(float)
        la = out["away_expected_runs"].to_numpy(float)
        lh2, la2 = apply_k_edge(lh, la, k)
        alpha_h = _re.alpha_of(lh2, curves["home"])
        alpha_a = _re.alpha_of(la2, curves["away"])
        mc = _re.derive_markets_mc(lh2, la2, alpha_h, alpha_a,
                                   n_draws=n_draws, seed=seed)
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
                     ) -> dict[str, Any]:
    """Daily Phase-3 pass with the C2 edge expansion (k_edge).

    Same contract as the original (monitor block + artifact paths). When
    k_edge is None, k is REFIT on this run's OOF pre-holdout games (per-run
    refit policy) and applied to the OOF markets (k_edge into
    derive_markets_v3) and the slate board (seam around predict_slate_runs).
    The fitted k + drift band ALWAYS land in the markets meta
    (summary['k_edge']). k_edge=1.0 disables the expansion."""
    global _K_EDGE_ACTIVE
    if k_edge is None:
        oof = _DAILY_OOF_CACHE
        if oof is None or oof.empty:
            decided = (decided_snapshot.copy() if decided_snapshot is not None
                       else _re.get_decided_frame(games))
            # P1 projection input (adoption 7e4c529): enrich the fallback
            # decided the same way the original daily does so k is fit on
            # the SAME lambda basis the markets are priced on (the daily's
            # own OOF is P1 after its internal attach).
            decided, _, _ = _re.attach_projection_levels(decided)
            oof = _orig_run_oof(decided, decided_snapshot=decided)["oof"]
        mask = k_edge_holdout_mask(oof)
        k_edge = fit_k_edge(oof["home_expected_runs"].to_numpy(float),
                            oof["away_expected_runs"].to_numpy(float),
                            (oof["home_score"] - oof["away_score"]).to_numpy(
                                float),
                            mask)
        _re.logger.warning("Run engine daily (k-edge): fitted k=%.4f "
                           "(pre-holdout OOF, n=%d)", k_edge, int(mask.sum()))
    _K_EDGE_ACTIVE = k_edge
    try:
        res = _orig_run_engine_daily(games, target_games, target_date_str,
                                     n_draws=n_draws,
                                     decided_snapshot=decided_snapshot)
    finally:
        _K_EDGE_ACTIVE = None
    # Log k into the markets meta regardless (the original persisted the
    # markets + meta inside; re-derive the summary block is NOT needed — the
    # monitor block carries market_metrics already; inject k for the record).
    block = res.get("block")
    if block is not None:
        block["k_edge"] = k_edge_meta(k_edge)
        block["k_edge"]["k_fitted_run"] = round(float(k_edge), 4)
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
