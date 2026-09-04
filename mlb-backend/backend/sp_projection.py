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
    df = games.copy()
    if len(pre_mask) != len(df):
        raise ValueError(
            f"pre_mask length {len(pre_mask)} != frame rows {len(df)}")
    pre = np.asarray(pre_mask, dtype=bool)
    meta: dict = {}
    for side in ("home", "away"):
        lo = [f"{c}_{side}" for c in PROJ_LO_BETTER]
        hi = [f"{c}_{side}" for c in PROJ_HI_BETTER]
        missing = [c for c in lo + hi + [f"sp_era_{side}"]
                   if c not in df.columns]
        if missing:
            raise ValueError(f"missing component columns: {missing}")
        z = pd.DataFrame(index=df.index)
        for c in lo + hi:
            mu = float(df.loc[pre, c].mean())
            sd = float(df.loc[pre, c].std())
            z[c] = (df[c] - mu) / sd
        n_comp = z[lo + hi].notna().sum(axis=1)
        comp = (-z[lo].sum(axis=1, min_count=1)
                + z[hi].sum(axis=1, min_count=1))
        comp = comp.where(n_comp >= MIN_PROJ_COMPONENTS)
        comp = comp / n_comp.where(n_comp >= MIN_PROJ_COMPONENTS)
        df[f"sp_proj_{side}"] = comp
        # ERA-equivalent scale: OLS sp_era ~ comp on pre-holdout, junk out.
        era = f"sp_era_{side}"
        cal = df.loc[pre & comp.notna() & df[era].notna()
                     & (df[era].abs() <= SP_JUNK_ERA)]
        slope = _ols_slope(cal[era].to_numpy(), comp[cal.index].to_numpy()) \
            if len(cal) else 0.0
        df[f"sp_proj_era_{side}"] = comp / abs(slope) if slope else comp
        meta[side] = {
            "era_on_proj_slope": round(slope, 4),
            "coverage_pre": round(float(comp[pre].notna().mean()), 4),
            "coverage_sealed": round(float(comp[~pre].notna().mean()), 4),
            "n_cal_rows": int(len(cal)),
        }
    return df, meta
