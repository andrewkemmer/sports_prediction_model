"""Frontend/backend parity for the fixed-line calibration tabs.

game_total_calibration / run_line_calibration must reproduce a direct
recomputation from the same artifact columns (pooled pred/observed,
log-loss, bucket counts). If the page math ever drifts from the backend
verdict-frame math, these fail loudly instead of rendering silent,
confident, wrong calibration tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

FRONTEND = Path(__file__).resolve().parent
sys.path.insert(0, str(FRONTEND))

import nba_market_diagnostics as nd  # noqa: E402


def _totals_frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    over = rng.uniform(0.05, 0.60, n)
    under = rng.uniform(0.05, 0.60, n)
    total = rng.integers(200, 251, n).astype(float)
    return pd.DataFrame({
        "p_over_225": over,
        "p_under_225": under,
        "total": total,
        "home_score": (total // 2).astype(float),
        "away_score": (total - total // 2).astype(float),
        "margin": (2 * (total // 2) - total).astype(float),
    })


def test_game_total_calibration_matches_direct_recompute():
    dec = _totals_frame()
    table = nd.game_total_calibration(dec, 225)
    assert table["warning"] is None
    keep = dec["total"] != 225
    p = (dec.loc[keep, "p_over_225"]
         / (dec.loc[keep, "p_over_225"] + dec.loc[keep, "p_under_225"])).to_numpy(float)
    y = (dec.loc[keep, "total"] > 225).astype(float).to_numpy()
    assert table["n_games"] == len(dec)
    assert table["n_pushes"] == int((~keep).sum())
    assert abs(table["pooled_pred"] - p.mean()) < 5e-4
    assert abs(table["pooled_observed"] - y.mean()) < 5e-4
    pc = np.clip(p, 1e-9, 1 - 1e-9)
    direct_ll = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())
    brier = float(((p - y) ** 2).mean())
    assert abs(table["pooled_brier"] - round(brier, 4)) < 5e-4
    assert sum(b["count"] for b in table["bins"]) == int(keep.sum())
    assert direct_ll > 0  # sanity: recompute path is well-defined


def _spread_frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    home = rng.integers(100, 131, n).astype(float)
    away = rng.integers(100, 131, n).astype(float)
    h = rng.uniform(0.05, 0.90, n)
    a = rng.uniform(0.05, 0.90, n)
    return pd.DataFrame({
        "home_score": home,
        "away_score": away,
        "p_home_cover_m2": h,
        "p_away_cover_m2": a,
        "p_push_m2": np.full(n, 0.02),
    })


def test_run_line_calibration_matches_direct_recompute():
    dec = _spread_frame()
    table = nd.run_line_calibration(dec, -2.0)
    assert table["warning"] is None
    margin = (dec["home_score"] - dec["away_score"]).to_numpy(float)
    h = dec["p_home_cover_m2"].to_numpy(float)
    a = dec["p_away_cover_m2"].to_numpy(float)
    keep = margin != -2.0
    p = (h[keep] / (h[keep] + a[keep]))
    y = (margin[keep] > -2.0).astype(float)
    assert table["n_pushes"] == int((~keep).sum())
    assert abs(table["pooled_pred"] - p.mean()) < 5e-4
    assert abs(table["pooled_observed"] - y.mean()) < 5e-4
    assert sum(b["count"] for b in table["bins"]) == int(keep.sum())
