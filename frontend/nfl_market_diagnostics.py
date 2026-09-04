"""NFL run-engine market diagnostics — the MLB ``market_diagnostics`` mirror
over the NFL decided OOF store (``kind == "oof"`` rows of
``nfl_run_engine_markets_YYYYMMDD.csv``).

The NFL artifact carries per game: the mu pair, fair spread/total (integer
medians of the margin/total PMFs), FULL grids (``p_home_cover_<±L>`` /
``p_push_<±L>`` over −14…+14 PLUS the extended favorite-magnitude grid
0.5…24.0 in 0.5 steps — 48 magnitudes, both signs; ``p_over_<U>`` /
``p_under_<U>`` / ``p_push_<U>`` over 24…66), the derived-ML pair, the raw
±0.5 pair, shrink columns (flagged), and ACTUALS (home/away score, total,
margin) + honest outcomes (``y_over_fair`` / ``y_cover_fair`` /
``y_home_win`` …).

This module mirrors MLB's diagnostics API — same dict schemas, same table
conventions, same captions — on the NFL grids. The chart builders
(``chart_distribution`` / ``chart_calibration`` / ``chart_game_total_curve``)
and the generic ``calibration_curve`` / ``_bucket_calibration`` helpers are
REUSED from ``market_diagnostics`` (pure, schema-compatible), so the rendered
charts and tables are structurally identical to MLB's.

NFL-specific semantics (documented deltas, not approximations):
  * Integer-support lines: whole-number totals/spreads carry a REAL push
    band (P(total == U), P(margin == L)) — pushes are excluded from the
    2-way calibration population and reported as n_pushes/push_rate (the
    same convention MLB uses for its own whole-number lines).
  * The favorite side of a run line resolves from the derived-ML pair
    (P(H>A)/(1−P(tie)) ≥ 0.5 ⇒ home favorite); cover = P(margin > m) when
    the favorite is home, P(margin < −m) when away, at magnitude m. The
    −0.5 stop prices from the P(margin > 0.5) = win-probability identity
    (ties are dead mass excluded from both sides), so its 2-way curve IS
    the derived-ML curve.
  * Fair lines come from the artifact's ``fair_total`` / ``fair_spread``
    columns (model medians — never the offered lines).
"""

from __future__ import annotations

import io
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

import utils
from market_diagnostics import (DIAG_TABS, LOW_N,  # noqa: F401
                                OWN_LINE_EDGES, OWN_LINE_LABELS,
                                X_1PCT_TICKS, _auc, _bucket_calibration,
                                calibration_curve, chart_calibration,
                                chart_distribution, chart_game_total_curve)

# Integer-support grids (mirror of the slate engine's TOTAL_INT_LINES /
# SPREAD_INT_LINES + the extended SPREAD_COVER_MAGS — the artifact's own
# column families). SPREAD_GRID stays the integer grid (the artifact's
# p_home_cover_<±L>/p_push_<±L> integer columns); the Spread Lines tab
# selector prices ANY magnitude in SPREAD_LINE_CHOICES below.
TOTAL_GRID = list(range(24, 67))          # 24 … 66 (points totals)
SPREAD_GRID = list(range(-14, 15))        # −14 … +14 (margin thresholds)
# Favorite-side line choices for the Spread Lines tab + the run-line
# calibration card: favorite-anchored NEGATIVE values, 0.5 steps, -0.5 …
# -24.0 (48 lines — the NFL equivalent of MLB's -0.5 … -4.0 at the
# NFL's ~3x larger margin σ; see the diagnostics-v2 record for the
# cover-rate table). The -0.5 stop prices from the P(margin > 0.5) =
# win-probability identity (its 2-way curve IS the derived-ML curve).
SPREAD_LINE_CHOICES = [round(-0.5 - 0.5 * i, 1) for i in range(48)]
DEFAULT_RUN_MAG = -3.0   # default selector line (favorite −3)
DEFAULT_TOTAL = 46
# Deep-line honesty threshold: beyond −20 the sealed evidence is n < ~20
# (see the diagnostics-v2 record) — the page adds an honesty caption.
DEEP_LINE_CAPTION_MAG = 20.0

# Relativized offsets (integers — NFL totals are integer-support).
OFFSET_EDGES = [-3, -2, -1, 0, 1, 2, 3]

# Pooled fixed lines for the "Pooled lines" tab (spanning the range).
FIXED_TOTAL_LINES = (38, 42, 46, 50, 54)

OWN_LINE_EDGES_ = OWN_LINE_EDGES      # 40–41 … 60+ (percent, shared with MLB)
OWN_LINE_LABELS_ = OWN_LINE_LABELS


def _col(base: str, x: float) -> str:
    """Artifact grid column tag — MLB-style (mirror of the slate engine's
    ``_fname``): '-' -> 'm', '.' -> '_', and an integral line drops its
    trailing '.0' (integer grids). ``p_home_cover_m3`` = P(margin > -3);
    ``p_home_cover_0_5`` = P(margin > 0.5); ``p_home_cover_3`` =
    P(margin > 3)."""
    s = str(float(x))
    if s.endswith(".0"):
        s = s[:-2]
    return f"{base}_{s.replace('-', 'm').replace('.', '_')}"


def decided_rows(markets: Optional[pd.DataFrame]) -> pd.DataFrame:
    """The decided OOF rows (kind == 'oof') with outcomes. Empty frame when
    the artifact carries only slate rows — never fabricated."""
    if markets is None or not len(markets):
        return pd.DataFrame()
    df = markets[markets.get("kind") == "oof"].copy()
    if {"home_score", "away_score"}.issubset(df.columns):
        df = df[df[["home_score", "away_score"]].notna().all(axis=1)]
    return df.reset_index(drop=True)


def _grid_cols(decided: pd.DataFrame, base: str,
               grid: list[int]) -> list[str]:
    return [_col(base, g) for g in grid]


def _2way(po: float, pu: float) -> Optional[float]:
    """Re-scaled 2-way P(over) = p_over / (p_over + p_under) — the card's
    no-push probability. None when the denom is non-positive (never
    fabricated)."""
    denom = po + pu
    if not np.isfinite(po) or not np.isfinite(pu) or denom <= 0:
        return None
    return float(po / denom)


# ---------------------------------------------------------------------------
# Chart 1 — totals distribution fit-check
# ---------------------------------------------------------------------------
def total_distribution(decided: pd.DataFrame, kmax: int = 66) -> dict[str, Any]:
    """Observed P(total=k) vs modeled mean per-game total PMF.

    Modeled = average over games of the total PMF — read directly from the
    artifact's per-game ``p_push_<U>`` columns (P(total == U)); observed =
    the actual totals histogram. Mirrors MLB's total_distribution (observed
    vs modeled bars/line).
    """
    empty = {"ks": list(range(kmax + 1)), "observed": [], "modeled": [],
             "callouts": {}, "n_games": 0,
             "warning": "No decided games with outcomes in the artifact."}
    if not len(decided) or "total" not in decided.columns:
        return empty
    total = decided["total"].to_numpy(float)
    ks = np.arange(0, kmax + 1)
    observed = np.array([(total == k).mean() for k in ks])
    modeled = np.zeros(len(ks))
    for U in TOTAL_GRID:
        col = _col("p_push", U)
        if col not in decided.columns:
            continue
        if U <= kmax:
            modeled[U] = float(decided[col].mean())
    # normalized (grid mass may not capture < 24 / > 66 exactly; keep both
    # series on the same support and renormalize the modeled mass)
    msum = modeled.sum()
    if msum > 0:
        modeled = modeled / msum
    return {
        "ks": ks.tolist(),
        "observed": [round(float(v), 5) for v in observed],
        "modeled": [round(float(v), 5) for v in modeled],
        "callouts": {
            "P(total<=35)": {"observed": round(float(observed[:36].sum()), 4),
                             "modeled": round(float(modeled[:36].sum()), 4)},
            "P(total>=60)": {"observed": round(float(observed[60:].sum()), 4),
                             "modeled": round(float(modeled[60:].sum()), 4)},
            "note": ("Per-game total PMFs from the calibrated 76×76 joint; "
                     "integer-support means the whole-number mass IS the "
                     "push band."),
        },
        "n_games": int(len(decided)),
        "warning": None,
    }


# ---------------------------------------------------------------------------
# Charts 2–4 — calibration curves
# ---------------------------------------------------------------------------
def _grid_p_over(decided: pd.DataFrame, line: float) -> np.ndarray:
    """Grid p_over at an integer total line (clamped to the grid)."""
    U = int(np.clip(round(line), TOTAL_GRID[0], TOTAL_GRID[-1]))
    col = _col("p_over", U)
    if col not in decided.columns:
        return np.full(len(decided), np.nan)
    return decided[col].to_numpy(float)


def relativized_pairs(decided: pd.DataFrame,
                      offsets: Optional[list[int]] = None) -> pd.DataFrame:
    """(p_over, did_go_over) pairs at line = fair_total + offset.

    Each offset re-prices every game at ITS OWN shifted fair total — the
    pooled probability axis spans the full range. NFL integer lines: pushes
    (total == line) are excluded from the pairs (2-way no-push basis, the
    same convention as the GTL/RL tabs).
    """
    offsets = OFFSET_EDGES if offsets is None else offsets
    if not len(decided) or "total" not in decided.columns:
        return pd.DataFrame(columns=["p", "y", "offset"])
    if "fair_total" not in decided.columns:
        return pd.DataFrame(columns=["p", "y", "offset"])
    fair = decided["fair_total"].to_numpy(float)
    total = decided["total"].to_numpy(float)
    frames = []
    for off in offsets:
        lines = fair + off
        p = _grid_p_over(decided, lines[0])  # placeholder, replaced below
        p = np.array([_grid_p_over(decided, float(l))[i]
                      if np.isfinite(l) else np.nan
                      for i, l in enumerate(lines)])
        y = np.zeros(len(decided))
        push = np.zeros(len(decided), bool)
        for i, l in enumerate(lines):
            if not np.isfinite(l):
                continue
            if total[i] == l:
                push[i] = True
            else:
                y[i] = float(total[i] > l)
        ok = np.isfinite(p) & ~push
        if ok.any():
            frames.append(pd.DataFrame({"p": p[ok], "y": y[ok],
                                        "offset": float(off)}))
    return pd.concat(frames, ignore_index=True) if frames \
        else pd.DataFrame(columns=["p", "y", "offset"])


def fixed_line_pairs(decided: pd.DataFrame,
                     lines: tuple[int, ...] = FIXED_TOTAL_LINES,
                     ) -> pd.DataFrame:
    """(p_over, outcome) pairs at fixed published integer totals, one row
    per (game, line). Pushes excluded (2-way no-push basis)."""
    if not len(decided) or "total" not in decided.columns:
        return pd.DataFrame(columns=["p", "y", "line"])
    total = decided["total"].to_numpy(float)
    frames = []
    for line in lines:
        col = _col("p_over", line)
        if col not in decided.columns:
            continue
        p = decided[col].to_numpy(float)
        push = (total == line)
        ok = np.isfinite(p) & ~push
        if ok.any():
            frames.append(pd.DataFrame({"p": p[ok],
                                        "y": (total[ok] > line).astype(float),
                                        "line": float(line)}))
    return pd.concat(frames, ignore_index=True) if frames \
        else pd.DataFrame(columns=["p", "y", "line"])


# ---------------------------------------------------------------------------
# Game Total Lines + Run Lines tabs (MLB's game_total_calibration /
# run_line_calibration mirrors)
# ---------------------------------------------------------------------------
def _total_line_calibration(decided: pd.DataFrame, line: Optional[float],
                            n_bins: int = 20) -> dict[str, Any]:
    empty = {"line": line, "bins": [], "curve_bins": [],
             "n_games": 0, "n_pushes": 0,
             "push_rate": 0.0, "pooled_pred": None, "pooled_observed": None,
             "pooled_winrate": None, "pooled_ece": None, "pooled_brier": None,
             "pooled_auc": None,
             "warning": "No decided games available for this view."}
    if not len(decided) or "total" not in decided.columns:
        return empty
    total = decided["total"].to_numpy(float)
    n_all = len(decided)
    pred = np.full(n_all, np.nan)
    event = np.zeros(n_all)
    push = np.zeros(n_all, bool)
    priced = np.zeros(n_all, bool)
    if line is None:
        if "fair_total" not in decided.columns:
            empty["warning"] = "Missing fair_total column."
            return empty
        lines = decided["fair_total"].to_numpy(float)
        for i in range(n_all):
            l = lines[i]
            if not np.isfinite(l):
                continue
            po_col = _col("p_over", int(l))
            pu_col = _col("p_under", int(l))
            if po_col not in decided.columns or pu_col not in decided.columns:
                continue
            rso = _2way(float(decided[po_col].iloc[i]),
                        float(decided[pu_col].iloc[i]))
            if rso is None:
                continue
            pred[i] = rso
            priced[i] = True
            if total[i] == l:
                push[i] = True
                continue
            event[i] = float(total[i] > l)
        edges, labels = OWN_LINE_EDGES_, OWN_LINE_LABELS_
    else:
        po_col = _col("p_over", int(line))
        pu_col = _col("p_under", int(line))
        if po_col not in decided.columns or pu_col not in decided.columns:
            empty["warning"] = (f"Grid columns for line {line:g} missing — "
                                "cannot price at this line.")
            return empty
        po = decided[po_col].to_numpy(float)
        pu = decided[pu_col].to_numpy(float)
        for i in range(n_all):
            rso = _2way(float(po[i]), float(pu[i]))
            if rso is None:
                continue
            pred[i] = rso
            priced[i] = True
            if total[i] == line:
                push[i] = True
                continue
            event[i] = float(total[i] > line)
        edges = [round(5.0 * b, 2) for b in range(n_bins + 1)]
        labels = [f"{int(edges[b])}-{int(edges[b + 1])}"
                  for b in range(n_bins)]
    ok = priced & ~push
    n = int(priced.sum())
    n_pushes = int(push.sum())
    if not ok.any():
        empty.update({"n_games": n, "n_pushes": n_pushes,
                      "push_rate": (round(n_pushes / n, 4) if n else 0.0),
                      "warning": "No non-push games priceable in this view."})
        return empty
    (bins, pooled_pred, pooled_obs, pooled_winrate, pooled_ece,
     pooled_brier, pooled_auc) = _bucket_calibration(pred[ok], event[ok],
                                                     edges, labels)
    curve_edges = [float(b) for b in range(101)]
    curve_labels = [f"{b}-{b + 1}" for b in range(100)]
    (curve_bins, _, _, _, _, _, _) = _bucket_calibration(
        pred[ok], event[ok], curve_edges, curve_labels)
    curve_bins = [b for b in curve_bins if b["count"] > 0]
    return {"line": line, "bins": bins, "curve_bins": curve_bins,
            "n_games": n, "n_pushes": n_pushes,
            "push_rate": round(n_pushes / n, 4) if n else 0.0,
            "pooled_pred": pooled_pred, "pooled_observed": pooled_obs,
            "pooled_winrate": pooled_winrate, "pooled_ece": pooled_ece,
            "pooled_brier": pooled_brier, "pooled_auc": pooled_auc,
            "warning": None}


def game_total_calibration(decided: pd.DataFrame,
                           line: Optional[float] = None,
                           n_bins: int = 20) -> dict[str, Any]:
    """Calibration for the 'Game Total Lines' diagnostics tab.

    line=None ('All') → every game priced at ITS OWN fair total (the
    artifact's integer fair_total): predicted = re-scaled 2-way P(over) at
    the own line; observed = over rate, pushes excluded. line given → all
    games at that ONE fixed integer total. Mirrors MLB's
    game_total_calibration with the NFL integer-support push band.
    """
    return _total_line_calibration(decided, line, n_bins)


def _favorite_cover(decided: pd.DataFrame, mag: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(cover_prob, dog_prob, push_prob, is_home_fav) at favorite magnitude
    mag — the favorite's 2-way P(cover) denominator = cover + dog (pushes
    folded out). Favorite = derived-ML side (≥ 0.5 home, else away).

    mag may be a half-integer (0.5 … 23.5): the -0.5 stop prices from the
    P(margin > 0.5) = win-probability identity — cover = P(favored wins
    outright), dog = P(other side wins outright), and the TIE (margin == 0)
    is dead mass excluded from both, so the 2-way predicted
    cover/(cover+dog) = P(H>A)/(P(H>A)+P(A>H)) = the derived ML by
    construction. Integer magnitudes keep the push-band semantics
    (push = P(margin == ±m))."""
    hw = decided["derived_ml"].to_numpy(float)
    is_home = np.isfinite(hw) & (hw >= 0.5)
    m = float(mag)
    cov = np.full(len(decided), np.nan)
    dog = np.full(len(decided), np.nan)
    psh = np.zeros(len(decided))
    whole = float(m).is_integer()
    for i in range(len(decided)):
        if not np.isfinite(hw[i]):
            continue
        if m == 0.5:
            # -0.5 stop: ties (margin == 0) are dead mass — exclude from
            # both sides so the 2-way equals the derived ML.
            if is_home[i]:
                hc = _col("p_home_cover", 0.5)
                if hc not in decided.columns:
                    continue
                cov[i] = float(decided[hc].iloc[i])   # P(margin > 0.5)
                ma = _col("p_home_cover", -0.5)
                if ma not in decided.columns:
                    continue
                dog[i] = 1.0 - float(decided[ma].iloc[i])  # P(margin < -0.5)
            else:
                ma = _col("p_home_cover", -0.5)
                if ma not in decided.columns:
                    continue
                cov[i] = 1.0 - float(decided[ma].iloc[i])  # P(margin < -0.5)
                hc = _col("p_home_cover", 0.5)
                if hc not in decided.columns:
                    continue
                dog[i] = float(decided[hc].iloc[i])       # P(margin > 0.5)
            continue
        if is_home[i]:
            hc = _col("p_home_cover", m)
            hp = _col("p_push", m) if whole else None
            if hc not in decided.columns or (whole and hp not in decided.columns):
                continue
            cov[i] = float(decided[hc].iloc[i])
            push_v = float(decided[hp].iloc[i]) if whole else 0.0
            psh[i] = push_v
            dog[i] = 1.0 - cov[i] - push_v
        else:
            hc = _col("p_home_cover", -m)
            hp = _col("p_push", -m) if whole else None
            if hc not in decided.columns or (whole and hp not in decided.columns):
                continue
            ph = float(decided[hc].iloc[i])
            push_v = float(decided[hp].iloc[i]) if whole else 0.0
            psh[i] = push_v
            cov[i] = 1.0 - ph - push_v     # away covers: margin < −m
            dog[i] = ph
    return cov, dog, psh, is_home


def run_line_calibration(decided: pd.DataFrame,
                         line: Optional[float] = None,
                         n_bins: int = 20) -> dict[str, Any]:
    """Calibration for the 'Run Lines' diagnostics tab — mirrors MLB's
    run_line_calibration for the FAVORITE side.

    line=None ('All') → every game priced at ITS OWN fair run line (the
    artifact's fair_spread, favorite-anchored); line given → all games at
    that ONE favorite magnitude m. 2-way P(cover) = cover/(cover + dog);
    pushes (margin == ±m) excluded + reported; win_rate = the moneyline-card
    'V' convention (pick the favorite to cover if P(cover) > 0.5 else the
    dog)."""
    empty = {"line": line, "bins": [], "curve_bins": [],
             "n_games": 0, "n_pushes": 0,
             "push_rate": 0.0, "pooled_pred": None, "pooled_observed": None,
             "pooled_winrate": None, "pooled_ece": None, "pooled_brier": None,
             "pooled_auc": None,
             "warning": "No decided games available for this view."}
    if not len(decided) or "margin" not in decided.columns:
        empty["warning"] = "Missing margin column (need decided outcomes)."
        return empty
    if "derived_ml" not in decided.columns:
        empty["warning"] = "Missing derived_ml column — cannot resolve the favorite."
        return empty
    margin = decided["margin"].to_numpy(float)
    n_all = len(decided)
    pred = np.full(n_all, np.nan)
    event = np.zeros(n_all)
    push = np.zeros(n_all, bool)
    priced = np.zeros(n_all, bool)
    if line is None:
        if "fair_spread" not in decided.columns:
            empty["warning"] = "Missing fair_spread column."
            return empty
        fair = decided["fair_spread"].to_numpy(float)
        for i in range(n_all):
            fs = fair[i]
            if not np.isfinite(fs):
                continue
            mag = abs(fs)
            if mag < 0.5:
                mag = 0.5
            cov, dog, psh, is_home = _favorite_cover(
                decided.iloc[[i]].reset_index(drop=True), float(mag))
            c, d, p = float(cov[0]), float(dog[0]), float(psh[0])
            denom = c + d
            if denom <= 0:
                continue
            pred[i] = c / denom
            priced[i] = True
            # Outcome-based push (the margin LANDED on the favorite's line):
            # the 2-way probability already folds the push mass out.
            if (margin[i] == mag and is_home[0]) or \
                    (margin[i] == -mag and not is_home[0]):
                push[i] = True
                continue
            event[i] = float((margin[i] > mag) if is_home[0]
                             else (margin[i] < -mag))
        edges, labels = OWN_LINE_EDGES_, OWN_LINE_LABELS_
    else:
        mag = abs(float(line))
        cov, dog, psh, is_home = _favorite_cover(decided, mag)
        denom = cov + dog
        valid = (np.isfinite(cov) & np.isfinite(dog) & (denom > 0))
        pred[valid] = cov[valid] / denom[valid]
        priced = valid
        # Outcome-based push: the margin LANDED exactly on the favorite's
        # line (±m) — the 2-way probability already folds the push mass out.
        push = np.where(is_home, margin == mag, margin == -mag) & valid
        ev = np.where(is_home, margin > mag, margin < -mag)
        event = (ev & valid & ~push).astype(float)
        edges = [round(5.0 * b, 2) for b in range(n_bins + 1)]
        labels = [f"{int(edges[b])}-{int(edges[b + 1])}"
                  for b in range(n_bins)]
    ok = priced & ~push
    n = int(priced.sum())
    n_pushes = int(push.sum())
    if not ok.any():
        empty.update({"n_games": n, "n_pushes": n_pushes,
                      "push_rate": (round(n_pushes / n, 4) if n else 0.0),
                      "warning": "No non-push games priceable in this view."})
        return empty
    (bins, pooled_pred, pooled_obs, pooled_winrate, pooled_ece,
     pooled_brier, pooled_auc) = _bucket_calibration(pred[ok], event[ok],
                                                     edges, labels)
    curve_edges = [float(b) for b in range(101)]
    curve_labels = [f"{b}-{b + 1}" for b in range(100)]
    (curve_bins, _, _, _, _, _, _) = _bucket_calibration(
        pred[ok], event[ok], curve_edges, curve_labels)
    curve_bins = [b for b in curve_bins if b["count"] > 0]
    return {"line": line, "bins": bins, "curve_bins": curve_bins,
            "n_games": n, "n_pushes": n_pushes,
            "push_rate": round(n_pushes / n, 4) if n else 0.0,
            "pooled_pred": pooled_pred, "pooled_observed": pooled_obs,
            "pooled_winrate": pooled_winrate, "pooled_ece": pooled_ece,
            "pooled_brier": pooled_brier, "pooled_auc": pooled_auc,
            "warning": None}


# ---------------------------------------------------------------------------
# Monitor calibration cards (MLB totals_monitor_stats / runline_monitor_stats
# mirrors over the NFL decided store)
# ---------------------------------------------------------------------------
def _fnum(r: Any, key: str) -> float:
    """Safe float getter — NaN when missing/None (never crashes on
    synthetic or partial frames)."""
    v = r.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _reliability_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Simple decile reliability ECE over finite (y, p) pairs."""
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 20:
        return float("nan")
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0 + 1e-12
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    errs = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        errs.append(abs(float(p[m].mean()) - float(y[m].mean())))
    return float(np.mean(errs)) if errs else float("nan")


def _logloss(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], np.clip(p[ok], 1e-6, 1 - 1e-6)
    if len(y) < 2:
        return None
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _card_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    """Compact metric set for the winner cards (MLB schema: actual_win_rate
    == win_rate == empirical pick win rate, push-excluded; predicted_mean =
    the pooled favored-probability mean shown beside it as one compact
    stat line)."""
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], np.clip(p[ok], 1e-6, 1 - 1e-6)
    out: dict[str, Any] = {"n": int(len(y)), "actual_win_rate": None,
                           "win_rate": None, "predicted_mean": None,
                           "auc": None, "ece_calibrated": None,
                           "brier": None, "logloss": None}
    if len(y) == 0:
        return out
    rate = round(float(y.mean()), 4)
    out["actual_win_rate"] = rate
    out["win_rate"] = rate
    out["predicted_mean"] = round(float(p.mean()), 4)
    out["brier"] = round(float(((p - y) ** 2).mean()), 4)
    out["ece_calibrated"] = round(_reliability_ece(y, p), 4)
    out["logloss"] = (round(_logloss(y, p), 4)
                       if _logloss(y, p) is not None else None)
    if len(y) >= 2 and y.min() != y.max():
        out["auc"] = round(float(_auc(y, p)), 4)
    return out


def winner_cards(decided: pd.DataFrame) -> dict[str, Any]:
    """The three binary winner cards (MLB _WINNER_CARDS schema) computed
    from the decided OOF store: pooled = all rows, holdout = sealed 2025
    (frame_view == 'sealed'). Each card: win_rate / auc / ece_calibrated /
    brier / logloss / n + holdout — pushes excluded from the 2-way pick
    basis (whole-number lines, neither wins nor losses)."""
    def _split(view: pd.DataFrame) -> dict[str, Any]:
        return _card_metrics(view["y"].to_numpy(float),
                             view["p"].to_numpy(float))

    def _pairs(kind: str, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in df.iterrows():
            if kind == "over_under":
                # Raw 3-way legs at the fair line: p_over_fair is P(total
                # > U) and the artifact carries NO p_under_fair — the
                # under leg is 1 - over - push (push read from the integer
                # grid's p_push_<U> column). 2-way no-push re-scale.
                po = _fnum(r, "p_over_fair")
                U = _fnum(r, "fair_total")
                tot = _fnum(r, "total")
                if not np.isfinite(po) or not np.isfinite(U):
                    continue
                u_int = int(np.clip(round(U), TOTAL_GRID[0], TOTAL_GRID[-1]))
                pp_col = _col("p_push", u_int)
                pp = (_fnum(r, pp_col) if pp_col in r else float("nan"))
                pu = 1.0 - po - (pp if np.isfinite(pp) else 0.0)
                if po + pu <= 0:
                    continue
                p2 = po / (po + pu)
                if tot == U:          # whole-line push
                    continue
                rows.append({"p": p2, "y": float(tot > U)})
            elif kind == "run_line":
                fs = _fnum(r, "fair_spread")
                margin = _fnum(r, "margin")
                hw = _fnum(r, "derived_ml")
                if not (np.isfinite(fs) and np.isfinite(margin)
                        and np.isfinite(hw)):
                    continue
                home_fav = hw >= 0.5
                m = int(max(abs(fs), 0.5))
                if home_fav:
                    cov = _fnum(r, f"p_home_cover_{m}")
                    push_p = _fnum(r, f"p_push_{m}")
                else:
                    ph = _fnum(r, f"p_home_cover_{-m}")
                    push_p = _fnum(r, f"p_push_{-m}")
                    cov = 1.0 - ph - push_p
                dog = 1.0 - cov - push_p
                if not (cov + dog) > 0:
                    continue
                p2 = cov / (cov + dog)
                if (margin == m and home_fav) or (margin == -m
                                                  and not home_fav):
                    continue
                covered = (margin > m) if home_fav else (margin < -m)
                rows.append({"p": p2, "y": float(covered)})
            else:  # derived_ml — PICK-SIDE framing (every metric on the
                # picked side, never home-side unconditionally, MLB
                # convention): home picked when P(H>A)/(1-P(tie)) >= 0.5,
                # away otherwise.
                ml = _fnum(r, "derived_ml")
                margin = _fnum(r, "margin")
                if not np.isfinite(ml) or not np.isfinite(margin):
                    continue
                if ml >= 0.5:
                    rows.append({"p": ml, "y": float(margin > 0)})
                else:
                    rows.append({"p": 1.0 - ml, "y": float(margin < 0)})
        return pd.DataFrame(rows)

    def _build(kind: str, df: pd.DataFrame) -> dict[str, Any]:
        view = _pairs(kind, df)
        if not len(view):
            return {}
        card = _split(view)
        hold = {}
        if "frame_view" in df.columns and (df["frame_view"] == "sealed").any():
            hv = _pairs(kind, df.loc[df["frame_view"] == "sealed"])
            if len(hv):
                hold = _split(hv)
        card["holdout"] = hold
        return card

    if not len(decided):
        return {}
    return {"over_under": _build("over_under", decided),
            "run_line": _build("run_line", decided),
            "derived_ml": _build("derived_ml", decided)}


def totals_monitor_stats(decided: pd.DataFrame, min_pct: float = 0.0,
                         side: str = "All") -> dict[str, Any]:
    """Interactive Totals card stats (MLB mirror): pick Over if the 2-way
    P(over) at the game's own fair total > 50% else Under; ``min_pct``
    keeps only games with pick prob above the confidence threshold
    (cumulative); win rate = W/(W+L) with whole-number pushes excluded.
    Returns the side-split table rows for the dataframe."""
    out = {"n": 0, "n_wins": 0, "n_losses": 0, "n_pushes": 0,
           "win_rate": None, "sides": {}}
    if not len(decided) or "fair_total" not in decided.columns \
            or "total" not in decided.columns:
        return out
    rows = []
    for _, r in decided.iterrows():
        # Raw 3-way legs at the fair line (see winner_cards — the artifact
        # has p_over_fair only; under = 1 - over - push from the grid).
        po = _fnum(r, "p_over_fair")
        U = _fnum(r, "fair_total")
        tot = _fnum(r, "total")
        if not np.isfinite(po) or not np.isfinite(U):
            continue
        u_int = int(np.clip(round(U), TOTAL_GRID[0], TOTAL_GRID[-1]))
        pp_col = _col("p_push", u_int)
        pp = (_fnum(r, pp_col) if pp_col in r else float("nan"))
        pu = 1.0 - po - (pp if np.isfinite(pp) else 0.0)
        if po + pu <= 0:
            continue
        p2 = po / (po + pu)
        if p2 * 100.0 < min_pct:
            continue
        rows.append({"p": p2, "tot": tot, "U": U})
    if not rows:
        return out
    n_wins = n_losses = n_pushes = 0
    sides = {"Over": {"n": 0, "n_wins": 0, "n_losses": 0, "n_pushes": 0},
             "Under": {"n": 0, "n_wins": 0, "n_losses": 0, "n_pushes": 0}}
    for rr in rows:
        pick_over = rr["p"] > 0.5
        side_name = "Over" if pick_over else "Under"
        if side != "All" and side != side_name:
            continue
        if rr["tot"] == rr["U"]:
            n_pushes += 1
            sides[side_name]["n_pushes"] += 1
            continue
        won = pick_over == (rr["tot"] > rr["U"])
        n_wins += won
        n_losses += (not won)
        sides[side_name]["n_wins"] += won
        sides[side_name]["n_losses"] += (not won)
    n = n_wins + n_losses
    out.update({"n": n, "n_wins": int(n_wins), "n_losses": int(n_losses),
                "n_pushes": int(n_pushes),
                "win_rate": (round(n_wins / n, 4) if n else None)})
    for name, s in sides.items():
        sw = s["n_wins"]
        sl = s["n_losses"]
        s["n"] = sw + sl
        s["win_rate"] = (round(sw / (sw + sl), 4) if (sw + sl) else None)
    out["sides"] = sides
    return out


# ---------------------------------------------------------------------------
# Run-engine feature drift + coverage (MLB markets.py mirror) — the Monitor's
# drift/coverage sections over the emitter CSVs
# (run_engine_feature_drift_YYYYMMDD.csv / run_engine_feature_coverage_
# YYYYMMDD.csv, written by nfl_explainability). Identical structure/wording to
# MLB's _render_run_engine_drift / _render_run_engine_coverage; the NFL run
# engine has no per-model blend-weight artifact, so the MODEL WEIGHT column is
# omitted (MLB's own 'has_weights' gate renders the same table without it).
# ---------------------------------------------------------------------------

def load_run_engine_csv(ds: str, prefix: str) -> pd.DataFrame | None:
    """Fetch run_engine_feature_{drift,coverage}_YYYYMMDD.csv for a date
    (the run engine's own drift/coverage artifacts over its 12-pool
    feature view, emitted by the daily run). None when absent/unreadable.
    Mirror of markets._load_run_engine_csv, with the NFL sport pinned
    explicitly — this page ALWAYS loads NFL artifacts regardless of the
    app's active-sport default (which outside the app context is MLB)."""
    fname = f"{prefix}_{ds}.csv"
    cfg = utils.get_source_config()
    try:
        raw, _src = utils._fetch_bytes(fname, sport="nfl", **cfg)
    except Exception:
        raw = None
    if raw is None:
        return None
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception:
        return None


def render_run_engine_drift(drift: pd.DataFrame | None) -> None:
    """Run-engine feature drift — same PSI table as the MLB monitor over
    the NFL run engine's own 12-pool features, WITH the MODEL WEIGHT
    column (layout parity with the Model Monitor's Feature Drift Analysis
    table). MODEL WEIGHT = per-feature blend-weighted importance from the
    shared moneyline feature-drift analysis (``nfl_model_monitor_*.json``
    -> ``feature_drift`` -> ``weight_pct``), which the daily emitter now
    writes into the drift CSV's ``weight_pct`` column (MLB renders the
    same join at render time; the NFL emitter resolves it at emission —
    same source, never hardcoded). A feature with no weight renders '—';
    the column is omitted entirely when no weight data is available
    (parity with the monitor's ``has_weights`` gate). When the CSV is
    absent the MLB empty-state wording renders (nothing fabricated)."""
    st.markdown("### Run-Engine Feature Drift (PSI)")
    if drift is None or drift.empty:
        st.info("No run-engine drift data for this date "
                "(run_engine_feature_drift_*.csv appears after a pipeline "
                "run).")
        return
    records = drift.to_dict("records")
    # MODEL WEIGHT per row — every cell is formatted by the SAME helper the
    # Model Monitor uses (utils.feature_weight_pct), so the column is
    # byte-identical to MLB's. Source: the CSV's own weight_pct column,
    # populated by the emitter from the moneyline monitor. CSV empties parse
    # as NaN (NOT None) — treat any non-finite value as absent so a
    # pre-weight artifact renders without the column (MLB's has_weights
    # gate), never as "nan%".
    def _weight(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if np.isfinite(f) else None

    weight_pcts = [_weight(r.get("weight_pct")) for r in records]
    has_weights = any(w is not None for w in weight_pcts)
    weight_header = "<th>MODEL WEIGHT</th>" if has_weights else ""
    rows = []
    for r, w in zip(records, weight_pcts):
        psi = r.get("psi", 0.0)
        status = r.get("status", "OK")
        psi_color = utils.AMBER if status == "WARN" else (
            utils.RED if status == "ALERT" else utils.TEXT)
        pill_cls = {"OK": "ok", "WARN": "warn", "ALERT": "alert",
                    "INSUFFICIENT": "ok"}.get(status, "ok")
        n_base, n_cur = r.get("n_baseline"), r.get("n_current")
        samples = (f" ({n_base}/{n_cur})"
                   if n_base is not None and n_cur is not None else "")
        label = utils.describe_feature(r.get("feature", ""), sport="nfl") \
            or r.get("feature", "")
        weight_cell = (f"<td>{utils.feature_weight_pct({'weight_pct': w})}</td>"
                       if has_weights else "")
        rows.append(
            f"<tr>"
            f"<td style='color:#E2E8F0;'>{r.get('feature','')}"
            f"<div style='color:#94A3B8;font-size:0.72rem;font-weight:400;"
            f"margin-top:1px;'>{label}</div></td>"
            f"<td>{r.get('current_mean', '—')}</td>"
            f"<td>{r.get('baseline_mean', '—')}</td>"
            f"<td style='color:{psi_color};font-weight:700;'>{psi:.3f}</td>"
            f"{weight_cell}"
            f"<td><span class='fb-status-pill {pill_cls}'>{status}</span>"
            f"<span style='color:#64748B;font-size:0.72rem;margin-left:5px;'>"
            f"{samples}</span></td></tr>")
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <table class="fb-table">
            <thead><tr><th>FEATURE</th><th>CURRENT MEAN</th><th>BASELINE MEAN</th>
            <th>PSI</th>{weight_header}<th>STATUS</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        <div style="color:#64748B;font-size:0.78rem;margin-top:6px;">
          Same windows as the moneyline drift; statuses on noise-adjusted PSI.
          INSUFFICIENT = window too small to judge drift.
          MODEL WEIGHT = blend-weighted feature importance from the shared
          feature-drift analysis (run engine has no per-model weight; '—' = no
          weight for this feature).
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_run_engine_coverage(cov: pd.DataFrame | None) -> None:
    """Run-engine feature coverage — the same measured/non-null table as
    the MLB monitor over the NFL run engine's 12-pool features, with MLB's
    EXACT caption strings, column headers (FEATURE | WINDOW | GAMES |
    % MEASURED | % NON-NULL | STATUS), sub-annotations (default-zero count
    under % NON-NULL) and status pills. When the CSV is absent the MLB
    empty-state wording renders (nothing fabricated)."""
    st.markdown("### Run-Engine Feature Coverage (non-null / measured)")
    if cov is None or cov.empty:
        st.info("No run-engine coverage data for this date "
                "(run_engine_feature_coverage_*.csv appears after a "
                "pipeline run).")
        return
    cov_sorted = sorted(
        cov.to_dict("records"),
        key=lambda r: (r.get("pct_measured", 0.0), r.get("feature", "")),
    )
    n_starved = sum(1 for r in cov_sorted if r.get("status") == "STARVED")
    n_low = sum(1 for r in cov_sorted if r.get("status") == "LOW_COVERAGE")
    sub = (
        f"<span style='color:{utils.RED};font-weight:700;'>{n_starved} "
        f"starved</span> · <span style='color:{utils.AMBER};font-weight:700;'>"
        f"{n_low} low</span>"
        if (n_starved or n_low) else
        "<span style='color:#4ADE80;font-weight:700;'>all windows healthy</span>")
    st.markdown(
        f"<div style='color:#94A3B8;font-size:0.8rem;margin:-6px 0 10px;'>"
        f"Share of games in each drift window with a real observation per "
        f"feature — {sub}</div>",
        unsafe_allow_html=True)
    show_starved_only = n_starved + n_low > 0
    rows = []
    shown = 0
    for r in cov_sorted:
        status = r.get("status", "OK")
        if show_starved_only and status == "OK" and shown >= 12:
            continue
        pct_m = float(r.get("pct_measured", 0.0))
        pct_n = float(r.get("pct_nonnull", 0.0))
        n_def = int(r.get("n_default_zero", 0) or 0)
        color = utils.RED if status == "STARVED" else (
            utils.AMBER if status == "LOW_COVERAGE" else utils.TEXT)
        pill_cls = {"OK": "ok", "LOW_COVERAGE": "warn",
                    "STARVED": "alert"}.get(status, "ok")
        default_cell = (
            f"<div style='color:#94A3B8;font-size:0.72rem;font-weight:400;"
            f"margin-top:1px;'>{n_def} default-zero</div>" if n_def else "")
        rows.append(
            f"<tr>"
            f"<td style='color:#E2E8F0;'>{r.get('feature','')}</td>"
            f"<td>{r.get('window','')}</td>"
            f"<td>{r.get('n_games','—')}</td>"
            f"<td style='color:{color};font-weight:700;'>{pct_m:.0f}%</td>"
            f"<td>{pct_n:.0f}%{default_cell}</td>"
            f"<td><span class='fb-status-pill {pill_cls}'>{status}</span></td>"
            f"</tr>")
        shown += 1
    n_hidden = len(cov_sorted) - shown
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <table class="fb-table">
            <thead><tr><th>FEATURE</th><th>WINDOW</th><th>GAMES</th>
            <th>% MEASURED</th><th>% NON-NULL</th><th>STATUS</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        <div style="color:#64748B;font-size:0.78rem;margin-top:6px;">
          % MEASURED = real observations only (default-filled values excluded);
          % NON-NULL includes them. STARVED &lt;25% measured, LOW_COVERAGE &lt;80%.
          {f"{n_hidden} healthy feature-window pairs hidden." if n_hidden > 0 else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def fold_slate_history(monitors: list[dict]) -> dict[str, list[dict]]:
    """Fold the dated monitors' accumulating slate-history into per-card
    series (the MLB ``rolling`` dict shape: ``{card_key: [points]}``).

    Each dated ``nfl_run_engine_monitor_<date>.json`` ships its own
    ``slate_history`` list (accumulating across runs — each entry is a
    per-card summary point carrying ``card`` (one of the winner-card keys:
    ``over_under`` / ``run_line`` / ``derived_ml``) plus the metric fields
    ``date``, ``ece_calibrated``, ``brier``, ``logloss``,
    ``predicted_mean``, ``n`` — the MLB rolling-point shape). The fold
    concatenates every dated file's entries newest-first and groups by
    ``card``; entries without a ``card`` key are ignored (never fabricated,
    never guessed). Empty input -> empty dict -> the page renders MLB's
    first-build empty-state wording."""
    out: dict[str, list[dict]] = {}
    for mon in monitors or []:
        for entry in mon.get("slate_history") or []:
            card = entry.get("card") if isinstance(entry, dict) else None
            if not card:
                continue
            out.setdefault(str(card), []).append(entry)
    return out


def runline_monitor_stats(decided: pd.DataFrame,
                          mag: float) -> dict[str, Any]:
    """Interactive Run Line card stats (MLB mirror): at favorite magnitude
    ``mag`` pick the DERIVED-ML favorite; win_rate = how often that side
    covers at the line, 2-way no-push. ``mag`` may be a half-integer: the
    -0.5 stop uses the win-probability identity (cover = P(favored wins
    outright), dog = P(other side wins outright), ties are dead mass — the
    2-way equals the derived ML); integer magnitudes keep the push-band
    semantics (margin == ±m pushes excluded)."""
    out = {"n": 0, "n_wins": 0, "n_losses": 0, "n_pushes": 0,
           "win_rate": None, "sides": {}}
    if not len(decided) or "margin" not in decided.columns \
            or "derived_ml" not in decided.columns:
        return out
    m = float(mag)
    whole = float(m).is_integer()
    sides = {"home": {"n": 0, "n_wins": 0, "n_losses": 0, "n_pushes": 0},
             "away": {"n": 0, "n_wins": 0, "n_losses": 0, "n_pushes": 0}}
    n_wins = n_losses = n_pushes = 0
    for _, r in decided.iterrows():
        hw = _fnum(r, "derived_ml")
        margin = _fnum(r, "margin")
        if not np.isfinite(hw) or not np.isfinite(margin):
            continue
        home_fav = hw >= 0.5
        side_name = "home" if home_fav else "away"
        if m == 0.5:
            hc = _col("p_home_cover", 0.5)
            ma = _col("p_home_cover", -0.5)
            if hc not in decided.columns or ma not in decided.columns:
                continue
            cov_h = _fnum(r, hc)                 # P(margin > 0.5)
            dog_h = 1.0 - _fnum(r, ma)           # P(margin < -0.5)
            cov = cov_h if home_fav else dog_h
            dog = dog_h if home_fav else cov_h
            push_p = 0.0
        else:
            hc = _col("p_home_cover", m)
            ma = _col("p_home_cover", -m)
            if hc not in decided.columns or ma not in decided.columns:
                continue
            if home_fav:
                cov = _fnum(r, hc)
                push_p = _fnum(r, _col("p_push", m)) if whole else 0.0
                dog = 1.0 - cov - push_p
            else:
                ph = _fnum(r, ma)
                push_p = _fnum(r, _col("p_push", -m)) if whole else 0.0
                cov = 1.0 - ph - push_p
                dog = ph
        if not (cov + dog) > 0:
            continue
        p2 = cov / (cov + dog)
        if (margin == m and home_fav) or (margin == -m and not home_fav):
            n_pushes += 1
            sides[side_name]["n_pushes"] += 1
            continue
        covered = (margin > m) if home_fav else (margin < -m)
        won = (p2 > 0.5) == covered
        n_wins += won
        n_losses += (not won)
        sides[side_name]["n_wins"] += won
        sides[side_name]["n_losses"] += (not won)
    n = n_wins + n_losses
    out.update({"n": n, "n_wins": int(n_wins), "n_losses": int(n_losses),
                "n_pushes": int(n_pushes),
                "win_rate": (round(n_wins / n, 4) if n else None)})
    for name, s in sides.items():
        sw = s["n_wins"]
        sl = s["n_losses"]
        s["n"] = sw + sl
        s["win_rate"] = (round(sw / (sw + sl), 4) if (sw + sl) else None)
    out["sides"] = sides
    return out