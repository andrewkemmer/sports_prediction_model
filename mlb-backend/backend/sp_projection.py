"""Candidate SP projection-composite producer (record-only; NOT served).

The xFIP/SIERA-family projection composite measured on the run engine in the
SP-sensitivity arm-test (record data_delivery/mlb_sp_projection_arm_<sha>.json,
commit b7eed32). This module reproduces that composite VERBATIM so the same
candidate feature can be measured on the binary moneyline without re-deriving
a different metric.

Composite definition (locked to b7eed32):
  * Components (per side, lower-is-better / higher-is-better pitching):
        lower:  sp_fip, sp_xwoba, sp_whip, sp_bb9
        higher: sp_k9_5g, sp_whiff_3g, sp_fbvelo_3g
  * z = (x - mean_pre) / sd_pre per component, stats fit on the PRE-HOLDOUT
    rows only (strictly prior to the sealed window).
  * comp = (-sum(z_lower) + sum(z_higher)) / n_components, requiring >= 3
    non-null components (else NaN).
  * ERA-equivalent scale: OLS sp_era_side ~ comp_side fit on pre-holdout rows
    with |sp_era| <= 15 (junk ERA excluded); sp_proj_era_side = comp / |slope|.
    +1 unit ~= 1 ERA point of quality; HIGHER = better pitching.

PIT discipline: z-statistics and the OLS scale slope are fit exclusively on
rows where `pre_mask` is True (the pre-holdout pool). Trailing-window
components (sp_fip etc.) are themselves strictly-prior features already in the
frame. Nothing here touches served FEATURE_COLS / training / run engine.

Usage:
    from sp_projection import PROJ_LO_BETTER, PROJ_HI_BETTER, attach_projection_cols
    df, meta = attach_projection_cols(frame, pre_mask)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Projection composite components (locked verbatim to the b7eed32 arm-test).
PROJ_LO_BETTER = ["sp_fip", "sp_xwoba", "sp_whip", "sp_bb9"]
PROJ_HI_BETTER = ["sp_k9_5g", "sp_whiff_3g", "sp_fbvelo_3g"]
MIN_PROJ_COMPONENTS = 3
SP_JUNK_ERA = 15.0


def _ols_slope(y: np.ndarray, x: np.ndarray) -> float:
    """Simple OLS slope of y ~ x (intercept fitted)."""
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])


def _side_components(side: str) -> tuple[list[str], list[str]]:
    """(lo_better, hi_better) per-side component column names."""
    return ([f"{c}_{side}" for c in PROJ_LO_BETTER],
            [f"{c}_{side}" for c in PROJ_HI_BETTER])


def projection_components_present(games: pd.DataFrame,
                                  side: str | None = None) -> bool:
    """True when every component the producer needs (per side) exists in the
    frame. Guards the production attach seam so a frame that never carried
    the Statcast-derived trailing components (synthetic fixtures, a fresh
    cold-start frame) degrades to the legacy view instead of raising."""
    sides = ("home", "away") if side is None else (side,)
    for s in sides:
        lo, hi = _side_components(s)
        need = lo + hi + [f"sp_era_{s}"]
        if any(c not in games.columns for c in need):
            return False
    return True


def _composite(frame: pd.DataFrame, lo: list[str], hi: list[str],
               mu: dict, sd: dict) -> pd.Series:
    """Composite = mean over the z-scored components (higher = better),
    requiring >= MIN_PROJ_COMPONENTS non-null. mu/sd keyed by component
    column. Arithmetic is the verbatim b7eed32 definition."""
    z = pd.DataFrame(index=frame.index)
    for c in lo + hi:
        z[c] = (frame[c] - mu[c]) / sd[c]
    n_comp = z[lo + hi].notna().sum(axis=1)
    comp = (-z[lo].sum(axis=1, min_count=1)
            + z[hi].sum(axis=1, min_count=1))
    comp = comp.where(n_comp >= MIN_PROJ_COMPONENTS)
    comp = comp / n_comp.where(n_comp >= MIN_PROJ_COMPONENTS)
    return comp


def fit_projection_stats(games: pd.DataFrame,
                         pre_mask: np.ndarray) -> dict:
    """Fit the per-side projection stats on the PRE-HOLDOUT rows only.

    Returns a stats dict consumable by apply_projection_stats:
        {side: {"mu": {col: float}, "sd": {col: float},
                "slope": float, "n_cal": int}}
    slope = OLS sp_era ~ composite over pre rows with |sp_era| <= 15
    (junk ERA excluded); 0.0 when the calibration set is empty.
    """
    pre = np.asarray(pre_mask, dtype=bool)
    if len(pre) != len(games):
        raise ValueError(
            f"pre_mask length {len(pre)} != frame rows {len(games)}")
    stats: dict = {}
    for side in ("home", "away"):
        lo, hi = _side_components(side)
        era = f"sp_era_{side}"
        missing = [c for c in lo + hi + [era] if c not in games.columns]
        if missing:
            raise ValueError(f"missing component columns: {missing}")
        mu = {c: float(games.loc[pre, c].mean()) for c in lo + hi}
        sd = {c: float(games.loc[pre, c].std()) for c in lo + hi}
        comp_pre = _composite(games.loc[pre], lo, hi, mu, sd)
        era_pre = games.loc[pre, era]
        cal = comp_pre[era_pre.notna().to_numpy()
                       & (era_pre.abs() <= SP_JUNK_ERA).to_numpy()]
        cal = cal[cal.notna()]
        slope = _ols_slope(era_pre.loc[cal.index].to_numpy(),
                           cal.to_numpy()) if len(cal) else 0.0
        stats[side] = {"mu": mu, "sd": sd, "slope": float(slope),
                       "n_cal": int(len(cal))}
    return stats


def apply_projection_stats(games: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Add sp_proj_era_{home,away} to a copy of ``games`` using ALREADY-FIT
    stats (fit_projection_stats) — the cross-frame seam (a slate row set
    transformed with the decided-frame fit, never refit on itself)."""
    df = games.copy()
    for side in ("home", "away"):
        st = stats[side]
        lo, hi = _side_components(side)
        comp = _composite(df, lo, hi, st["mu"], st["sd"])
        slope = float(st["slope"])
        df[f"sp_proj_era_{side}"] = comp / abs(slope) if slope else comp
    return df


def attach_projection_cols(
    games: pd.DataFrame, pre_mask: np.ndarray
) -> tuple[pd.DataFrame, dict]:
    """Add sp_proj_era_{home,away} to a copy of the frame.

    pre_mask: boolean array over ``games`` marking the PRE-HOLDOUT pool
    (rows strictly prior to the sealed window). z-stats + OLS scale are fit
    on those rows only; the transform then applies to every row.

    Returns (frame_with_cols, meta) where meta holds the per-side OLS slope
    and coverage on the pre pool and the complement (sealed) pool.
    """
    if len(pre_mask) != len(games):
        raise ValueError(
            f"pre_mask length {len(pre_mask)} != frame rows {len(games)}")
    pre = np.asarray(pre_mask, dtype=bool)
    stats = fit_projection_stats(games, pre_mask)
    df = apply_projection_stats(games, stats)
    meta: dict = {}
    for side in ("home", "away"):
        s = stats[side]
        col = f"sp_proj_era_{side}"
        meta[side] = {
            "era_on_proj_slope": round(s["slope"], 4),
            "coverage_pre": round(float(df.loc[pre, col].notna().mean()), 4),
            "coverage_sealed": round(
                float(df.loc[~pre, col].notna().mean()), 4),
            "n_cal_rows": int(s["n_cal"]),
        }
    return df, meta
