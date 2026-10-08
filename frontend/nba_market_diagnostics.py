"""NBA totals/point-spread diagnostics over run-engine artifacts.

Structural mirror of MLB's ``market_diagnostics.py``: the same function
contracts (``total_distribution`` / ``relativized_pairs`` /
``fixed_line_pairs`` / ``calibration_curve`` / ``game_total_calibration`` /
``run_line_calibration`` / ``chart_game_total_curve``), the same bucket
calibration (win-rate 'V', guarded AUC, LOW_N suppression, share_pct,
pooled aggregates) and the same chart grammar (count bars + observed curve
+ dashed diagonal + amber pooled marker in one layered chart).

NBA-specific facts, each read off the artifact rather than assumed:

  1. The spread columns are THRESHOLD form: ``p_home_cover_{L}`` =
     P(margin > L) with ``p_push_{L}`` = P(margin == L) and
     ``p_away_cover_{L}`` = P(margin < L), summing to 1 at every integer L
     in -20..20 plus the +/-0.5 half stops. A displayed home spread of -5
     is threshold T=+5 (home covers iff margin > 5), so the displayed home
     line is MINUS its threshold. The stored ``fair_spread`` column does
     not agree with the artifact's own 50/50 grid point (measured on the
     20260926 artifact), so fair lines are computed here by grid argmin —
     the same rule as MLB's ``fair_total_lines`` / ``fair_run_lines`` —
     and the stored column is never trusted.
  2. The totals grid ships the model's own pmf directly: ``p_push_total_k``
     = P(total == k), summing to ~0.9965 per game. The distribution tab's
     modeled series is the per-game mean of that pmf — the same observed-
     vs-modeled overlay MLB computes analytically from its NB parameters,
     here read instead of recomputed.
  3. ``calibration_cards`` ships a per-line prequential Platt (a, b) map,
     so ECE-calibrated figures are computable read-only; where no map
     exists for a line the calibrated value is None, never a copy of raw.
"""
from __future__ import annotations

import io
import math
from typing import Any, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import utils

# --- Grids and shared constants (MLB's shapes, NBA's ranges) ---------------
TOTAL_GRID = list(range(180, 281))                      # integer totals grid
SPREAD_MAGS = [0.5] + [float(v) for v in range(1, 21)]  # favorite magnitudes
SPREAD_THRESHOLDS = [-int(m) if False else m for m in range(-20, 21)]  # noqa
DIAG_TABS = ["Distribution", "Relativized", "Pooled lines",
             "Game Total Lines", "Spread Lines"]
# MLB prices four pooled lines spread across the range; NBA's mean total is
# ~226, so the four lines sit 10 points either side of it.
POOLED_LINES = (210, 220, 230, 240)
# Relativized offsets: MLB spans roughly +/-45% of a ~4.5-run mean with nine
# 0.5 steps; the NBA mean is ~226 points, so the same proportional span is
# about +/-10 points. Integer steps because the NBA totals grid is integer.
OFFSET_EDGES = [-10.0, -8.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0,
                4.0, 6.0, 8.0, 10.0]
# Spread Lines tab choices (favorite magnitudes the grid serves).
SPREAD_LINE_CHOICES = [-0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -8.0, -10.0]

X_1PCT_TICKS = [round(v, 2) for v in np.arange(0.0, 1.001, 0.01)]
LOW_N = 30
OWN_LINE_EDGES = list(range(40, 61)) + [101]
OWN_LINE_LABELS = [f"{lo}-{lo + 1}" for lo in range(40, 60)] + ["60+"]

# Monitor-card contracts (MLB's).
TOTALS_CONF_THRESHOLDS = [50, 51, 52, 53, 54, 55]
WINNER_CARDS = (
    ("over_under", "Over/Under",
     "Pick Over if P(over the game's line) > 50%, else Under"),
    ("run_line", "Run Line (favorite cover)",
     "Pick the favorite to cover its own fair line if P(cover) > 50%, "
     "else the dog"),
    ("derived_ml", "Derived ML (run-line model moneyline)",
     "Pick the side with P > 50% — home if P(home win) > 50%, else away "
     "(derived from the paired score distribution; distinct from the "
     "moneyline ensemble)"),
)
_HISTORY_COLUMNS = ["game_date", "away", "home", "home_score", "away_score",
                    "line", "pick", "pick_prob", "winner", "correct"]


def _label(value: float) -> str:
    """Grid column suffix: -2 -> 'm2', 0.5 -> '0_5', 225 -> '225'."""
    text = str(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace("-", "m").replace(".", "_")


def _num(value) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def decided_rows(markets: pd.DataFrame | None) -> pd.DataFrame:
    if markets is None or not len(markets):
        return pd.DataFrame()
    frame = markets.copy()
    if "kind" in frame:
        kind_mask = frame.kind.eq("oof")
        if "decided" in frame:
            kind_mask = kind_mask | frame.decided.eq(True)
        frame = frame[kind_mask].copy()
    elif "decided" in frame:
        frame = frame[frame.decided.eq(True)].copy()
    required = [c for c in ("home_score", "away_score", "total", "margin") if c in frame]
    if len(required) < 2:
        return pd.DataFrame()
    if "total" not in frame:
        frame["total"] = pd.to_numeric(frame.home_score, errors="coerce") + pd.to_numeric(frame.away_score, errors="coerce")
    if "margin" not in frame:
        frame["margin"] = pd.to_numeric(frame.home_score, errors="coerce") - pd.to_numeric(frame.away_score, errors="coerce")
    return frame[frame.total.notna() & frame.margin.notna()].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rank-based AUC + bucket calibration — MLB's ``_bucket_calibration`` clone
# ---------------------------------------------------------------------------


def _auc(y_true, y_scores) -> Optional[float]:
    """Rank-based AUC (Mann-Whitney U) — identical to sklearn's default."""
    y_true = np.asarray(y_true, dtype=bool)
    y_scores = np.asarray(y_scores, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(y_scores).rank(method="average").to_numpy()
    return (ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _guarded_auc(pred: np.ndarray, event: np.ndarray) -> Optional[float]:
    if len(pred) < 2:
        return None
    return _auc(event, pred)


def _bucket_calibration(pred: np.ndarray, event: np.ndarray,
                        edges: list[float],
                        labels: list[str]) -> tuple[list[dict], float, float,
                                                    float, float, float,
                                                    Optional[float]]:
    """MLB's bucket calibration, verbatim semantics.

    Empty bins are kept with count 0 / None stats — never dropped.
    ``win_rate`` = W/(W+L) for the pick rule 'over/favorite if P > 50% else
    the other side' (a 'V' around 50%). ``ece`` = |mean_pred - observed|
    per bin. ``auc`` is None for low-n (n < LOW_N) or single-class bins.
    Pooled ece is count-weighted over populated bins; pooled auc runs over
    ALL pairs at the line.
    """
    pred_frac = np.clip(np.asarray(pred, float), 0.0, 1.0)
    ev = np.asarray(event, float)
    win = np.where(pred_frac > 0.5, ev, 1.0 - ev)
    pct = pred_frac * 100.0
    n = len(pct)
    bins = []
    for b, lab in enumerate(labels):
        lo, hi = edges[b], edges[b + 1]
        m = (pct >= lo) & (pct < hi) if b < len(labels) - 1 else (pct >= lo)
        cnt = int(m.sum())
        mean_pred = float(pred_frac[m].mean()) if cnt else None
        observed = float(ev[m].mean()) if cnt else None
        bin_auc = _guarded_auc(pred_frac[m], ev[m]) if cnt >= LOW_N else None
        bins.append({
            "bin": lab,
            "bin_center": round(float((lo + hi) / 200.0), 3),
            "count": cnt,
            "mean_pred": (round(mean_pred, 4) if cnt else None),
            "observed": (round(observed, 4) if cnt else None),
            "win_rate": (round(float(win[m].mean()), 4) if cnt else None),
            "auc": (round(bin_auc, 4) if bin_auc is not None else None),
            "ece": (round(abs(mean_pred - observed), 4)
                    if (cnt and mean_pred is not None
                        and observed is not None) else None),
            "brier": (round(float(((pred_frac[m] - ev[m]) ** 2).mean()), 4)
                      if cnt else None),
            "low_n": (cnt < LOW_N and cnt > 0),
            "share_pct": (round(cnt / n * 100.0, 2) if n else None),
        })
    tot = n if n else 0
    pooled_ece = sum((b["count"] / tot * b["ece"])
                     for b in bins if b["count"] and b["ece"] is not None)
    pooled_auc = _guarded_auc(pred_frac, ev)
    return (bins, round(float(pred_frac.mean()), 4),
            round(float(ev.mean()), 4),
            round(float(win.mean()), 4),
            round(pooled_ece, 4),
            round(float(((pred_frac - ev) ** 2).mean()), 4),
            (round(pooled_auc, 4) if pooled_auc is not None else None))


def _gtl_line_points(table: dict, curve_bins: Optional[list] = None) -> pd.DataFrame:
    """Line points for the layered chart — one row per (bin, series)."""
    rows = []
    for b in curve_bins if curve_bins is not None else (table.get("bins") or []):
        if b.get("observed") is None or (curve_bins is None and b.get("low_n")):
            continue
        rows.append({"bin_center": b.get("bin_center"), "series": "Win rate",
                     "pct": round(b["win_rate"] * 100.0, 4),
                     "count": b["count"]})
        rows.append({"bin_center": b.get("bin_center"), "series": "Observed",
                     "pct": round(b["observed"] * 100.0, 4),
                     "count": b["count"]})
    return pd.DataFrame(rows)


def _gtl_table_frame(table: dict) -> pd.DataFrame:
    """Display columns for the calibration table, in MLB's order, with the
    pooled Total row last (share 100%)."""
    rows = []
    for b in table.get("bins") or []:
        rows.append({
            "bucket": b.get("bin"),
            "count": b.get("count"),
            "Mean Predicted (Raw)": b.get("mean_pred"),
            "Mean Actual": b.get("observed"),
            "auc": b.get("auc"),
            "ece": b.get("ece"),
            "brier": b.get("brier"),
            "% of Total": b.get("share_pct"),
        })
    rows.append({
        "bucket": "Total",
        "count": int(sum((b.get("count") or 0)
                         for b in table.get("bins") or [])),
        "Mean Predicted (Raw)": table.get("pooled_pred"),
        "Mean Actual": table.get("pooled_observed"),
        "auc": table.get("pooled_auc"),
        "ece": table.get("pooled_ece"),
        "brier": table.get("pooled_brier"),
        "% of Total": 100.0,
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Grid pricing
# ---------------------------------------------------------------------------


def grid_over_under_cols(line: float) -> tuple[str, str]:
    key = _label(line)
    return f"p_over_{key}", f"p_under_{key}"


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    out = np.log(p / (1 - p))
    return out if out.ndim else float(out)


def _sigmoid(x):
    out = 1.0 / (1.0 + np.exp(-np.asarray(x, float)))
    return out if np.ndim(out) else float(out)


def over_prob_at_lines(df: pd.DataFrame, lines: np.ndarray) -> np.ndarray:
    """2-way P(over) at arbitrary lines via logit-linear interpolation of
    the integer grid; clamped outside [180, 280].

    Rows whose bracket columns are missing price as NaN and drop out of
    any caller's ok-mask — matching ``game_total_calibration``'s per-row
    skip — rather than discarding the whole frame.
    """
    lines = np.asarray(lines, float)
    grid = np.asarray(TOTAL_GRID, float)
    if len(lines) != len(df):
        raise ValueError("lines must be one per row (per-row pricing)")
    lo_idx = np.clip(np.floor(lines - grid[0]).astype(int), 0, len(grid) - 2)
    frac = np.clip(lines - grid[lo_idx], 0.0, 1.0)
    out = np.full(len(df), np.nan)
    if not len(df):
        return out
    lo_col = [f"p_over_{int(g)}" for g in grid[lo_idx]]
    hi_col = [f"p_over_{int(g)}" for g in
              grid[np.minimum(lo_idx + 1, len(grid) - 1)]]
    have = np.array([c in df.columns for c in lo_col]) & \
        np.array([c in df.columns for c in hi_col])
    if not have.any():
        return out
    idx = np.where(have)[0]
    mat = df[sorted(set(lo_col + hi_col))].apply(
        pd.to_numeric, errors="coerce").to_numpy(float)
    col_ix = {c: i for i, c in enumerate(sorted(set(lo_col + hi_col)))}
    lo_for_idx = [lo_col[i] for i in idx]
    hi_for_idx = [hi_col[i] for i in idx]
    p_lo = mat[idx, [col_ix[c] for c in lo_for_idx]]
    p_hi = mat[idx, [col_ix[c] for c in hi_for_idx]]
    with np.errstate(invalid="ignore"):
        out[idx] = np.clip(
            _sigmoid((1 - frac[idx]) * _logit(p_lo)
                     + frac[idx] * _logit(p_hi)), 0.0, 1.0)
    return out


def _two_way_home_cover(decided: pd.DataFrame, threshold: float):
    """(2-way P(home covers margin > T), ok mask) at integer threshold T,
    push folded out of both sides. Missing columns price as not-ok."""
    col = f"p_home_cover_{_label(threshold)}"
    away_col = f"p_away_cover_{_label(threshold)}"
    if col not in decided.columns or away_col not in decided.columns:
        return np.full(len(decided), np.nan), np.zeros(len(decided), bool)
    h = decided[col].to_numpy(float)
    a = decided[away_col].to_numpy(float)
    denom = h + a
    ok = np.isfinite(h) & np.isfinite(a) & (denom > 0)
    out = np.full(len(decided), np.nan)
    out[ok] = h[ok] / denom[ok]
    return out, ok


def fair_total_lines(decided: pd.DataFrame) -> np.ndarray:
    """Per-game FAIR total line — grid argmin of |2-way P(over) - 0.5| over
    TOTAL_GRID, ties picking the LOWER line (strict `<` keeps the first
    ascending match). NaN where a game cannot be priced — never fabricated."""
    n = len(decided)
    best = np.full(n, np.nan)
    best_delta = np.full(n, np.inf)
    for line in TOTAL_GRID:
        over_col, under_col = grid_over_under_cols(line)
        if over_col not in decided.columns or under_col not in decided.columns:
            continue
        po = decided[over_col].to_numpy(float)
        pu = decided[under_col].to_numpy(float)
        denom = po + pu
        valid = (np.isfinite(po) & np.isfinite(pu) & np.isfinite(denom)
                 & (denom > 0))
        delta = np.full(n, np.inf)
        delta[valid] = np.abs(po[valid] / denom[valid] - 0.5)
        take = valid & (delta < best_delta - 1e-12)
        best_delta[take] = delta[take]
        best[take] = line
    return best


def _home_favorite(decided: pd.DataFrame) -> np.ndarray:
    """Moneyline favorite: True = home, from p_home_win_derived."""
    p = decided.get("p_home_win_derived")
    if p is None:
        return np.zeros(len(decided), bool)
    p = pd.to_numeric(p, errors="coerce").to_numpy(float)
    return np.where(np.isfinite(p), p >= 0.5, True)


def fair_spread_thresholds(decided: pd.DataFrame) -> np.ndarray:
    """Per-game FAIR spread THRESHOLD T (home covers iff margin > T) — the
    grid argmin of |2-way P(home cover) - 0.5| over integer thresholds
    -20..20, ties lower. The DISPLAYED home line is -T. NaN when unpricable."""
    n = len(decided)
    best = np.full(n, np.nan)
    best_delta = np.full(n, np.inf)
    for t in range(-20, 21):
        p2, ok = _two_way_home_cover(decided, float(t))
        delta = np.full(n, np.inf)
        delta[ok] = np.abs(p2[ok] - 0.5)
        take = ok & (delta < best_delta - 1e-12)
        best_delta[take] = delta[take]
        best[take] = float(t)
    return best


# ---------------------------------------------------------------------------
# Diagnostics 1 — totals distribution (observed vs modeled)
# ---------------------------------------------------------------------------


def total_distribution(decided: pd.DataFrame) -> dict[str, Any]:
    """Observed P(total=k) vs the model's own mean marginal.

    The modeled series is the per-game mean of ``p_push_total_k`` — the
    artifact ships the model's pmf directly, so the overlay is read rather
    than recomputed from dispersion parameters the NBA monitor does not
    carry. Callouts sit 10 points from each grid edge (MLB's P(<=1)/P(>=10)
    tails are ~10% of its span from each edge; 10 of 100 here).
    """
    empty = {"ks": TOTAL_GRID, "observed": [], "modeled": [],
             "callouts": {}, "n_games": 0,
             "warning": "No decided games with outcomes in the artifact."}
    if not len(decided) or "total" not in decided.columns:
        return empty
    total = decided["total"].to_numpy(float)
    ks = np.asarray(TOTAL_GRID)
    observed = np.array([(total == k).mean() for k in ks], dtype=float)
    pmf_cols = [c for c in (f"p_push_total_{k}" for k in TOTAL_GRID)
                if c in decided.columns]
    if pmf_cols:
        # Coerce, don't cast: a malformed artifact (string-valued grid
        # columns) prices as NaN and drops out of the mean rather than
        # crashing the tab.
        modeled = (decided[pmf_cols]
                   .apply(pd.to_numeric, errors="coerce")
                   .to_numpy(dtype=float)
                   .mean(axis=0))
    else:
        modeled = np.zeros(len(ks))
    obs_lo = float(observed[:11].sum())
    obs_hi = float(observed[-10:].sum())
    mod_lo = float(modeled[:11].sum())
    mod_hi = float(modeled[-10:].sum())
    return {
        "ks": ks.tolist(),
        "observed": [round(float(v), 5) for v in observed],
        "modeled": [round(float(v), 5) for v in modeled],
        "callouts": {
            "P(total<=190)": {"observed": round(obs_lo, 4),
                              "modeled": round(mod_lo, 4)},
            "P(total>=270)": {"observed": round(obs_hi, 4),
                              "modeled": round(mod_hi, 4)},
            "note": ("The modeled series is the artifact's own mean "
                     "P(total=k) grid pmf — read, not recomputed, because "
                     "the NBA monitor ships no per-game dispersion "
                     "parameters. This chart is the TOTALS law."),
        },
        "n_games": int(len(decided)),
        "warning": None,
    }


# ---------------------------------------------------------------------------
# Diagnostics 2-3 — relativized + pooled calibration pairs
# ---------------------------------------------------------------------------


def relativized_pairs(decided: pd.DataFrame,
                      offsets: Optional[list[float]] = None) -> pd.DataFrame:
    """(p_over, did_go_over) pairs at line = expected total + offset.

    Each offset re-prices every game at ITS OWN shifted line, snapped to
    the integer grid. Pairs whose total lands exactly on the line are
    dropped (the integer grid can push; a push is neither over nor under).
    """
    offsets = OFFSET_EDGES if offsets is None else offsets
    if not len(decided) or "total" not in decided.columns \
            or {"mu_h", "mu_a"}.difference(decided.columns):
        # The relativized view needs the per-game expected totals to shift
        # from; a frame without them prices nothing rather than crashing.
        return pd.DataFrame(columns=["p", "y", "offset"])
    mu_h = pd.to_numeric(decided["mu_h"], errors="coerce").to_numpy(float)
    mu_a = pd.to_numeric(decided["mu_a"], errors="coerce").to_numpy(float)
    exp_total = mu_h + mu_a
    total = decided["total"].to_numpy(float)
    frames = []
    for off in offsets:
        lines = np.round(exp_total + off)
        with np.errstate(invalid="ignore"):
            p = over_prob_at_lines(decided, lines)
        ok = np.isfinite(lines) & np.isfinite(p)
        y = (total >= lines + 1.0).astype(float)     # strict over, no push
        push = (total == lines)
        keep = ok & ~push
        frames.append(pd.DataFrame({"p": p[keep], "y": y[keep],
                                    "offset": off}))
    return pd.concat(frames, ignore_index=True)


def fixed_line_pairs(decided: pd.DataFrame,
                     lines=(210, 220, 230, 240)) -> pd.DataFrame:
    """(p_over, did_go_over) pairs at the four pooled fixed lines."""
    if not len(decided) or "total" not in decided.columns:
        return pd.DataFrame(columns=["p", "y", "line"])
    total = decided["total"].to_numpy(float)
    frames = []
    for line in lines:
        over_col, under_col = grid_over_under_cols(line)
        if over_col not in decided.columns or under_col not in decided.columns:
            continue
        po = decided[over_col].to_numpy(float)
        pu = decided[under_col].to_numpy(float)
        denom = po + pu
        valid = np.isfinite(po) & np.isfinite(pu) & (denom > 0)
        push = (total == line) & valid
        y = (total >= line + 1.0).astype(float)
        keep = valid & ~push
        frames.append(pd.DataFrame({"p": (po[keep] / denom[keep]),
                                    "y": y[keep], "line": line}))
    if not frames:
        return pd.DataFrame(columns=["p", "y", "line"])
    return pd.concat(frames, ignore_index=True)


def calibration_curve(pairs: pd.DataFrame, n_bins: int = 20,
                      min_count: int = 30) -> dict[str, Any]:
    """Equal-width reliability bins, dropping bins under ``min_count``."""
    empty = {"bins": [], "n_pairs": 0, "n_dropped_bins": 0,
             "warning": "No (prediction, outcome) pairs to calibrate."}
    if pairs is None or not len(pairs):
        return empty
    p = np.clip(pairs["p"].to_numpy(float), 0.0, 1.0)
    y = pairs["y"].to_numpy(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    bins, dropped = [], 0
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n < min_count:
            dropped += 1
            continue
        bins.append({
            "bin_center": round(float((edges[b] + edges[b + 1]) / 2), 3),
            "mean_pred": round(float(p[m].mean()), 4),
            "mean_actual": round(float(y[m].mean()), 4),
            "count": n,
        })
    warning = None
    if len(bins) < 2:
        warning = "Calibration curve under-specified — fewer than 2 valid bins."
    return {"bins": bins, "n_pairs": int(len(pairs)),
            "n_dropped_bins": dropped, "warning": warning}


# ---------------------------------------------------------------------------
# Diagnostics 4 — Game Total Lines (own fair line / fixed line)
# ---------------------------------------------------------------------------


def game_total_calibration(decided: pd.DataFrame,
                           line: Optional[float] = None,
                           n_bins: int = 20) -> dict[str, Any]:
    """Calibration table for the 'Game Total Lines' tab.

    line=None ('All') -> every game priced at ITS OWN FAIR line (grid argmin
    of |2-way P(over) - 0.5|, ties lower): predicted = 2-way rescaled
    P(over) at that line, bucketed at 1 pt (40-41 … 60+) because own-line
    P(over) hugs 50% by construction. line given -> ALL games at that ONE
    fixed line, 5-pt bins over [0, 1]. Pushes (total == whole-number line)
    are excluded from both sides and reported as n_pushes / push_rate.
    """
    empty = {"line": line, "bins": [], "curve_bins": [],
             "n_games": 0, "n_pushes": 0, "push_rate": 0.0,
             "pooled_pred": None, "pooled_observed": None,
             "pooled_winrate": None, "pooled_ece": None,
             "pooled_brier": None, "pooled_auc": None,
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
        lines = fair_total_lines(decided)
        for i in range(n_all):
            ln = lines[i]
            if np.isnan(ln):
                continue
            over_col, under_col = grid_over_under_cols(ln)
            if (over_col not in decided.columns
                    or under_col not in decided.columns):
                continue
            v = decided[over_col].iloc[i]
            u = decided[under_col].iloc[i]
            if pd.isna(v) or pd.isna(u):
                continue
            denom = float(v) + float(u)
            if denom <= 0:
                continue
            pred[i] = float(v) / denom
            priced[i] = True
            if total[i] == ln:
                push[i] = True
                continue
            event[i] = float(total[i] >= ln + 1.0)
        edges, labels = OWN_LINE_EDGES, OWN_LINE_LABELS
    else:
        over_col, under_col = grid_over_under_cols(line)
        if (over_col not in decided.columns
                or under_col not in decided.columns):
            empty["warning"] = (f"Grid columns for line {line} missing — "
                                "cannot price at this line.")
            return empty
        po = decided[over_col].to_numpy(float)
        pu = decided[under_col].to_numpy(float)
        denom = po + pu
        valid = np.isfinite(po) & np.isfinite(pu) & (denom > 0)
        pred[valid] = po[valid] / denom[valid]
        priced = valid
        push = (total == line) & valid
        event = ((total >= line + 1.0) & valid & ~push).astype(float)
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
# Diagnostics 5 — Spread Lines (favorite side, mirrors run-line tab)
# ---------------------------------------------------------------------------


def run_line_calibration(decided: pd.DataFrame,
                         line: Optional[float] = None,
                         n_bins: int = 20) -> dict[str, Any]:
    """Calibration table for the 'Spread Lines' tab — the FAVORITE side.

    favorite = moneyline favorite (p_home_win_derived >= 0.5; toss-up home).
    At magnitude m the favorite's cover prob comes from the threshold
    column T = -m when home is favored (cover iff margin > -m) and T = +m
    when away is favored (away covers iff margin < m); the shared push band
    is folded out of the 2-way. predicted = 2-way P(cover); observed =
    favorite-cover rate on the no-push basis; pushes reported as
    n_pushes / push_rate (-0.5 never pushes — NBA margins are integers, so
    margin == 0.5 is impossible). win_rate = the 'V' convention.

    line=None ('All') -> every game at ITS OWN fair threshold (grid argmin,
    ties lower), bucketed at 1 pt (40-41 … 60+). line given -> ALL games at
    that one magnitude, 5-pt bins over [0, 1].
    """
    empty = {"line": line, "bins": [], "curve_bins": [],
             "n_games": 0, "n_pushes": 0, "push_rate": 0.0,
             "pooled_pred": None, "pooled_observed": None,
             "pooled_winrate": None, "pooled_ece": None,
             "pooled_brier": None, "pooled_auc": None,
             "warning": "No decided games available for this view."}
    if not len(decided) or {"home_score", "away_score"}.difference(
            decided.columns):
        empty["warning"] = "Missing score columns (need home_score/away_score)."
        return empty
    margin = (decided["home_score"].to_numpy(float)
              - decided["away_score"].to_numpy(float))
    home_fav = _home_favorite(decided)

    n_all = len(decided)
    pred = np.full(n_all, np.nan)
    event = np.zeros(n_all)
    push = np.zeros(n_all, bool)
    priced = np.zeros(n_all, bool)

    if line is None:
        thresholds = fair_spread_thresholds(decided)
        for i in range(n_all):
            t = thresholds[i]
            if np.isnan(t):
                continue
            hcol = (f"p_home_cover_{_label(t)}")
            acol = (f"p_away_cover_{_label(t)}")
            if hcol not in decided.columns or acol not in decided.columns:
                continue
            h = decided[hcol].iloc[i]
            a = decided[acol].iloc[i]
            if pd.isna(h) or pd.isna(a):
                continue
            denom = float(h) + float(a)
            if denom <= 0:
                continue
            pred[i] = float(h) / denom
            priced[i] = True
            if margin[i] == t:
                push[i] = True
                continue
            event[i] = float(margin[i] > t)
        edges, labels = OWN_LINE_EDGES, OWN_LINE_LABELS
    else:
        mag = abs(float(line))
        hcol = f"p_home_cover_{_label(-mag)}"
        acol = f"p_away_cover_{_label(-mag)}"
        pcol = f"p_push_{_label(-mag)}"
        if hcol not in decided.columns or acol not in decided.columns:
            empty["warning"] = (f"Spread grid columns for line {line} "
                                "missing — cannot price at this line.")
            return empty
        h = decided[hcol].to_numpy(float)
        a = decided[acol].to_numpy(float)
        denom = h + a
        valid = (np.isfinite(h) & np.isfinite(a) & np.isfinite(margin)
                 & (denom > 0))
        pred[valid] = h[valid] / denom[valid]
        priced = valid
        push = (margin == -mag) & valid
        event = ((margin > -mag) & valid & ~push).astype(float)
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
# Charts — the MLB chart grammar
# ---------------------------------------------------------------------------


def chart_distribution(dist: dict) -> alt.Chart:
    """Observed bars vs modeled line over the totals grid."""
    df = pd.DataFrame({
        "k": dist["ks"] * 2,
        "series": ["observed"] * len(dist["ks"]) + ["modeled"] * len(dist["ks"]),
        "p": dist["observed"] + dist["modeled"],
    })
    bars = alt.Chart(df[df.series == "observed"]).mark_bar(
        color="#3B82F6", opacity=0.65).encode(
        x=alt.X("k:Q", title="Total points (home + away)"),
        y=alt.Y("p:Q", title="P(total = k)", axis=alt.Axis(format="%")),
    )
    line = alt.Chart(df[df.series == "modeled"]).mark_line(
        color="#F59E0B", strokeWidth=2.5, point=True).encode(
        x="k:Q", y="p:Q")
    return (bars + line).properties(height=300)


def chart_calibration(curve: dict, title: str,
                      x_domain: Optional[list[float]] = None) -> alt.Chart:
    """Reliability points + dashed perfect-calibration diagonal."""
    cdf = pd.DataFrame(curve["bins"])
    if cdf.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_point().encode(
            x="x:Q", y="y:Q")
    pts = alt.Chart(cdf).mark_circle(size=70, color="#22D3EE").encode(
        x=alt.X("mean_pred:Q", title="Mean predicted P(over)",
                scale=(alt.Scale(domain=x_domain, zero=False)
                       if x_domain else alt.Scale(zero=False))),
        y=alt.Y("mean_actual:Q", title="Observed over frequency",
                scale=alt.Scale(zero=False)),
        tooltip=["bin_center", "mean_pred", "mean_actual", "count"],
    )
    lo = (x_domain or [float(cdf["mean_pred"].min()) - 0.02,
                       float(cdf["mean_pred"].max()) + 0.02])
    diag_df = pd.DataFrame({"x": lo, "y": lo})
    diag = alt.Chart(diag_df).mark_line(
        color="#64748B", strokeDash=[6, 4]).encode(x="x:Q", y="y:Q")
    return (diag + pts).properties(height=300, title=title)


def chart_game_total_curve(table: dict, title: str,
                           obs_label: str = "Observed % (2-way, no push)",
                           curve_bins: Optional[list] = None,
                           x_tick_values: Optional[list] = None,
                           show_win_rate: bool = True,
                           x_label: str = "Predicted P(over)",
                           series_label: str = "Observed") -> dict:
    """MLB's layered calibration chart — count bars (left 'Games' axis) +
    observed curve (+ optional win-rate 'V') on the right '%' axis + gray
    low-n bars + dashed diagonal + amber pooled diamond, and the pooled
    table. Same layers, colors, tooltips and resolve_scale as MLB's."""
    tdf = pd.DataFrame(table["bins"])
    if tdf.empty:
        return {"chart": alt.Chart(pd.DataFrame()).mark_bar(),
                "table": _gtl_table_frame(table)}
    chart_df = pd.DataFrame(curve_bins if curve_bins else table["bins"]).copy()
    chart_df["low_n"] = chart_df["low_n"].astype(bool)
    x_dom = alt.Scale(domain=[0.0, 1.0], nice=False)
    y_pct_dom = alt.Scale(domain=[0.0, 100.0])

    bar_tip = [
        alt.Tooltip("bin_center:Q", title=x_label, format=".3f"),
        alt.Tooltip("count:Q", title="Games"),
        alt.Tooltip("mean_pred:Q", title="Mean predicted", format=".3f"),
        alt.Tooltip("observed:Q", title="Observed", format=".3f"),
        alt.Tooltip("win_rate:Q",
                    title=("Mean Actual" if not show_win_rate else "Win rate"),
                    format=".3f"),
    ]
    x_axis = (alt.Axis(values=x_tick_values, format=".2f", labelOverlap=True)
              if x_tick_values else None)
    bars = alt.Chart(chart_df).mark_bar(color="#3B82F6").encode(
        x=alt.X("bin_center:Q", title=x_label, scale=x_dom, axis=x_axis),
        y=alt.Y("count:Q", axis=alt.Axis(title="Games", grid=True)),
        tooltip=bar_tip)
    bar_layer = bars
    low_df = chart_df[chart_df["low_n"]]
    if not low_df.empty:
        low_bars = alt.Chart(low_df).mark_bar(
            color="#94A3B8", opacity=0.45).encode(
            x=alt.X("bin_center:Q", scale=x_dom),
            y=alt.Y("count:Q", axis=alt.Axis(title=None)),
            tooltip=bar_tip)
        bar_layer = bars + low_bars

    stack = _gtl_line_points(table, curve_bins=curve_bins)
    if not show_win_rate and not stack.empty:
        stack = stack[stack["series"] == "Observed"].copy()
        stack["series"] = series_label
    if stack.empty:
        line_chart = alt.Chart(pd.DataFrame()).mark_line()
    else:
        line_chart = alt.Chart(stack).mark_line(
            strokeWidth=2.5, point=alt.OverlayMarkDef(size=60)).encode(
            x=alt.X("bin_center:Q", scale=x_dom),
            y=alt.Y("pct:Q", axis=alt.Axis(title=obs_label, orient="right",
                                           grid=False),
                    scale=y_pct_dom),
            color=alt.Color("series:N",
                            scale=(alt.Scale(domain=[series_label],
                                             range=["#22C55E"])
                                   if not show_win_rate else
                                   alt.Scale(domain=["Observed", "Win rate"],
                                             range=["#22C55E", "#8B5CF6"])),
                            title=("Series" if show_win_rate else series_label)),
            tooltip=[
                alt.Tooltip("bin_center:Q", title=x_label, format=".3f"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("pct:Q", title=obs_label, format=".1f"),
                alt.Tooltip("count:Q", title="Games")])

    diag_df = pd.DataFrame({"bin_center": [0.0, 1.0], "pct": [0.0, 100.0]})
    diag = alt.Chart(diag_df).mark_line(
        color="#64748B", strokeDash=[5, 5], strokeWidth=1.5).encode(
        x=alt.X("bin_center:Q", scale=x_dom),
        y=alt.Y("pct:Q", axis=None, scale=y_pct_dom))

    layers = [bar_layer, line_chart, diag]
    pooled_pred = table.get("pooled_pred")
    pooled_obs = table.get("pooled_observed")
    if pooled_pred is not None and pooled_obs is not None:
        pool_df = pd.DataFrame({"bin_center": [pooled_pred],
                                "pct": [round(pooled_obs * 100.0, 4)]})
        pooled_marker = alt.Chart(pool_df).mark_point(
            shape="diamond", size=150, color="#F59E0B", filled=True).encode(
            x=alt.X("bin_center:Q", scale=x_dom),
            y=alt.Y("pct:Q", axis=None, scale=y_pct_dom),
            tooltip=[alt.Tooltip("bin_center:Q", title="Pooled predicted",
                                 format=".3f"),
                     alt.Tooltip("pct:Q", title="Pooled observed %",
                                 format=".1f")])
        layers.append(pooled_marker)

    chart = alt.layer(*layers).resolve_scale(
        x="shared", y="independent").properties(height=300, title=title)
    return {"chart": chart, "table": _gtl_table_frame(table)}


# ---------------------------------------------------------------------------
# Prediction-history frames (LINE | MODEL PICK | WINNER | RESULT)
# ---------------------------------------------------------------------------


def _row_date(row) -> Any:
    for key in ("gameday", "game_date"):
        if key in row:
            return row.get(key)
    return None


def totals_history_frame(decided: pd.DataFrame) -> pd.DataFrame:
    """Game-totals history priced at each game's OWN fair line (grid argmin).

    LINE is the fair line; the pick is Over/Under on the 2-way rescaled
    P(over) > 50%; a total exactly on the line is a PUSH (correct = NaN,
    excluded from the win rate).
    """
    cols = _HISTORY_COLUMNS
    if decided is None or not len(decided):
        return pd.DataFrame(columns=cols)
    lines = fair_total_lines(decided)
    rows: list[dict] = []
    for i, (_, r) in enumerate(decided.iterrows()):
        line = lines[i]
        total = _num(r.get("total"))
        if np.isnan(line) or total is None:
            continue
        over_col, under_col = grid_over_under_cols(line)
        v, u = _num(r.get(over_col)), _num(r.get(under_col))
        if v is None or u is None or (v + u) <= 0:
            continue
        p = v / (v + u)
        if _num(r.get("home_score")) is None or _num(r.get("away_score")) is None:
            continue
        over = p > 0.5
        pick = "Over" if over else "Under"
        if total == line:
            winner, correct = "Push", np.nan
        else:
            winner = pick if ((total > line) == over) else \
                ("Under" if over else "Over")
            correct = bool(winner == pick)
        rows.append({"game_date": _row_date(r),
                     "away": str(r.get("away_team", "—") or "—"),
                     "home": str(r.get("home_team", "—") or "—"),
                     "home_score": _num(r.get("home_score")),
                     "away_score": _num(r.get("away_score")),
                     "line": float(line), "pick": pick, "pick_prob": float(p),
                     "winner": winner, "correct": correct})
    return pd.DataFrame(rows, columns=cols)


def runline_history_frame(decided: pd.DataFrame) -> pd.DataFrame:
    """Run-line history at each game's OWN fair threshold (grid argmin).

    The displayed home line is MINUS its cover threshold: home covers iff
    margin > T, so home -5 means T = +5. Whole-number lines can push; the
    displayed pick probability is 2-way re-normalized (push folded out of
    both sides), and the PICK cell carries the pick's own spread number.
    """
    cols = _HISTORY_COLUMNS
    if decided is None or not len(decided):
        return pd.DataFrame(columns=cols)
    thresholds = fair_spread_thresholds(decided)
    rows: list[dict] = []
    for i, (_, r) in enumerate(decided.iterrows()):
        t = thresholds[i]
        margin = _num(r.get("margin"))
        if np.isnan(t) or margin is None:
            continue
        hcol, acol = f"p_home_cover_{_label(t)}", f"p_away_cover_{_label(t)}"
        h, a = _num(r.get(hcol)), _num(r.get(acol))
        if h is None or a is None or (h + a) <= 0:
            continue
        if _num(r.get("home_score")) is None or _num(r.get("away_score")) is None:
            continue
        p_home_2 = h / (h + a)
        home_line = -t                      # displayed home spread
        away_line = t
        away = str(r.get("away_team", "—") or "—")
        home = str(r.get("home_team", "—") or "—")
        pick = home if p_home_2 > 0.5 else away
        pick_line = home_line if pick == home else away_line
        if margin == t:
            winner, correct = "Push", np.nan
        else:
            home_covers = margin > t
            winner = home if home_covers else away
            correct = bool((p_home_2 > 0.5) == home_covers)
        rows.append({"game_date": _row_date(r), "away": away, "home": home,
                     "home_score": _num(r.get("home_score")),
                     "away_score": _num(r.get("away_score")),
                     "line": float(home_line), "pick": pick,
                     "pick_prob": float(p_home_2 if pick == home
                                        else 1.0 - p_home_2),
                     "winner": winner, "correct": correct,
                     "_away_line": float(away_line),
                     "_pick_line": float(pick_line)})
    return pd.DataFrame(rows, columns=cols + ["_away_line", "_pick_line"])


def filter_history_frame(frame: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    if not len(frame) or "game_date" not in frame.columns:
        return frame.copy()
    dts = pd.to_datetime(frame["game_date"], errors="coerce").dt.date
    keep = dts.notna() & (dts >= start_date) & (dts <= end_date)
    return frame[keep].reset_index(drop=True)


def filter_history_by_side(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    s = (str(side) or "All").strip()
    if frame is None:
        return pd.DataFrame()
    if not len(frame) or s == "All" or "pick" not in frame.columns:
        return frame.copy()
    return frame[frame["pick"].astype(str) == s].reset_index(drop=True)


def history_win_rate(frame: pd.DataFrame) -> dict:
    """n_games (non-push denominator), pooled win rate, push count."""
    if frame is None or not len(frame) or "correct" not in frame.columns:
        return {"n_games": 0, "win_rate": None, "n_pushes": 0}
    ok = frame["correct"].notna()
    n = int(ok.sum())
    wins = float(frame.loc[ok, "correct"].astype(bool).sum()) if n else 0.0
    n_pushes = int((frame["winner"] == "Push").sum()) \
        if "winner" in frame.columns else 0
    return {"n_games": n, "win_rate": (round(wins / n, 6) if n else None),
            "n_pushes": n_pushes}


# ---------------------------------------------------------------------------
# Monitor cards
# ---------------------------------------------------------------------------


def totals_monitor_stats(decided: pd.DataFrame, min_pct: int = 50,
                         side: str = "All") -> dict:
    """Pooled W/(W+L) calibration for the totals card at each game's own
    fair line — the scoring-mean diagnostic with the side filter."""
    view = filter_history_by_side(totals_history_frame(decided), side)
    if not len(view):
        return _empty_card()
    if min_pct and min_pct > 50:
        favoured = np.maximum(view["pick_prob"], 1.0 - view["pick_prob"])
        view = view[favoured * 100.0 > float(min_pct)]
    if not len(view):
        return _empty_card()
    out = _roll_up(view, "Over")
    if side == "All":
        out["sides"] = {s: _side_block(view[view["pick"] == s])
                        for s in ("Over", "Under")}
    return out


def runline_monitor_stats(decided: pd.DataFrame, magnitude: float) -> dict:
    """Favorite-cover card at a fixed magnitude, 2-way re-normalized."""
    mag = abs(float(magnitude))
    if decided is None or not len(decided):
        return _empty_card()
    margin = decided["margin"].to_numpy(float)
    home_fav = _home_favorite(decided)
    hcol = f"p_home_cover_{_label(-mag)}"
    acol = f"p_away_cover_{_label(-mag)}"
    if hcol not in decided.columns or acol not in decided.columns:
        return _empty_card()
    h = decided[hcol].to_numpy(float)
    a = decided[acol].to_numpy(float)
    denom = h + a
    valid = (np.isfinite(h) & np.isfinite(a) & np.isfinite(margin)
             & (denom > 0))
    # Favorite-side cover probability: home fav -> p_home_cover_{-m};
    # away fav -> p_away_cover_{+m}... which is 1 - home - push at the SAME
    # threshold, i.e. the away column of the home-favorite threshold.
    cover = np.where(home_fav, h, a)
    p_cover_2 = np.where(denom > 0, cover / np.where(denom > 0, denom, 1.0),
                         np.nan)
    rows: list[dict] = []
    for i in range(len(decided)):
        if not valid[i]:
            continue
        fav_is_home = bool(home_fav[i])
        thr = -mag if fav_is_home else mag
        covers = (margin[i] > thr) if fav_is_home else (margin[i] < thr)
        pushed = margin[i] == thr
        if pushed:
            correct, winner = np.nan, "Push"
        else:
            correct = bool((p_cover_2[i] > 0.5) == covers)
            winner = "Home" if covers else "Away"
        pick = "Home" if (fav_is_home == (p_cover_2[i] > 0.5)) else \
            ("Away" if fav_is_home else "Home")
        # The favorite IS the pick when its 2-way cover P > 50%; the card's
        # pick rule is favorite-cover, so the pick is the favorite side.
        pick = "Home" if fav_is_home else "Away"
        pick_prob = float(p_cover_2[i])
        rows.append({"pick": pick, "pick_prob": pick_prob,
                     "winner": winner, "correct": correct})
    if not rows:
        return _empty_card()
    vdf = pd.DataFrame(rows)
    out = _roll_up(vdf, "Home" if home_fav.mean() >= 0.5 else "Away") \
        if False else _roll_up_favorite(vdf)
    out["sides"] = {s: _side_block(vdf[vdf["pick"] == s])
                    for s in ("Home", "Away")}
    return out


def _roll_up_favorite(view: pd.DataFrame) -> dict:
    """W/(W+L) roll-up for a favorite-cover card: every row's pick is the
    favorite, so the pick probability IS the favorite's cover probability."""
    ok = view["correct"].notna()
    n = int(ok.sum())
    n_wins = int(view.loc[ok, "correct"].astype(bool).sum()) if n else 0
    return {"n": n, "n_wins": n_wins, "n_losses": n - n_wins,
            "n_pushes": int((view["winner"] == "Push").sum()),
            "predicted_2way": (round(float(view.loc[ok, "pick_prob"].mean()), 6)
                               if n else None),
            "win_rate": round(n_wins / n, 6) if n else None,
            "sides": {}}


def _side_block(side_view: pd.DataFrame) -> dict:
    n = int(side_view["correct"].notna().sum())
    n_wins = int(side_view.loc[side_view["correct"].notna(), "correct"]
                 .astype(bool).sum()) if n else 0
    return {"n": n, "n_pushes": int((side_view["winner"] == "Push").sum()),
            "win_rate": round(n_wins / n, 6) if n else None}


def _empty_card() -> dict:
    return {"n": 0, "n_wins": 0, "n_losses": 0, "n_pushes": 0,
            "predicted_2way": None, "win_rate": None, "sides": {}}


def _roll_up(view: pd.DataFrame, primary_pick: str) -> dict:
    ok = view["correct"].notna()
    n = int(ok.sum())
    n_wins = int(view.loc[ok, "correct"].astype(bool).sum()) if n else 0
    probs = np.where(view["pick"] == primary_pick, view["pick_prob"],
                     1.0 - view["pick_prob"])
    predicted = float(probs[ok.to_numpy()].mean()) if n else None
    return {"n": n, "n_wins": n_wins, "n_losses": n - n_wins,
            "n_pushes": int((view["winner"] == "Push").sum()),
            "predicted_2way": round(predicted, 6) if n else None,
            "win_rate": round(n_wins / n, 6) if n else None,
            "sides": {}}


def _platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    eps = 1e-7
    q = np.clip(p.astype(float), eps, 1 - eps)
    return 1.0 / (1.0 + np.exp(-(a * np.log(q / (1 - q)) + b)))


def calibration_coefficients(monitor: dict | None, family: str,
                             line) -> tuple[float, float] | None:
    """(a, b) from ``calibration_cards`` for one line, or None."""
    if line is None:
        return None
    cards = (monitor or {}).get("calibration_cards") or {}
    entry = (cards.get(family) or {}).get(str(int(line)))
    if not isinstance(entry, dict):
        return None
    primary = entry.get("over") or entry.get("home")
    if not isinstance(primary, dict) or "a" not in primary or "b" not in primary:
        return None
    a, b = _num(primary.get("a")), _num(primary.get("b"))
    return None if a is None or b is None else (a, b)


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if not len(y):
        return float("nan")
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum(float((idx == b).mean())
                     * abs(float(p[idx == b].mean()) - float(y[idx == b].mean()))
                     for b in range(bins) if (idx == b).any()))


def binary_card_metrics(y: np.ndarray, p: np.ndarray,
                        coeff: tuple[float, float] | None = None) -> dict:
    """MLB's card metric block; win rate is CONDITIONAL accuracy (W/(W+L))
    over rows with a real side lean."""
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok].astype(float), np.clip(p[ok].astype(float), 1e-7, 1 - 1e-7)
    n = int(len(y))
    if n == 0:
        return {"n": 0, "win_rate": None, "auc": None, "brier": None,
                "logloss": None, "ece_raw": None, "ece_calibrated": None,
                "predicted_mean": None, "actual_win_rate": None}
    pred = _platt(p, *coeff) if coeff else p
    leaned = np.abs(p - 0.5) > 1e-12
    correct = (p > 0.5) == (y == 1.0)
    win_rate = (round(float(correct[leaned].mean()), 6)
                if leaned.any() else None)
    return {
        "n": n, "win_rate": win_rate, "auc": _auc(y.astype(bool), pred),
        "brier": round(float(np.mean((p - y) ** 2)), 6),
        "logloss": round(float(-np.mean(y * np.log(pred)
                                        + (1 - y) * np.log(1 - pred))), 6),
        "ece_raw": round(_ece(y, p), 6),
        "ece_calibrated": round(_ece(y, pred), 6) if coeff else None,
        "predicted_mean": round(float(p.mean()), 6),
        "actual_win_rate": round(float(y.mean()), 6),
    }


def _over_fair_series(decided: pd.DataFrame) -> pd.Series:
    """2-way P(over) at each game's own fair line, from the grid."""
    values: list[float] = []
    lines = fair_total_lines(decided)
    for i, (_, r) in enumerate(decided.iterrows()):
        p = None
        if not np.isnan(lines[i]):
            over_col, under_col = grid_over_under_cols(lines[i])
            v, u = _num(r.get(over_col)), _num(r.get(under_col))
            if v is not None and u is not None and (v + u) > 0:
                p = v / (v + u)
        values.append(float("nan") if p is None else p)
    return pd.Series(values, index=decided.index, dtype=float)


def winner_cards(decided: pd.DataFrame,
                 monitor: dict | None = None) -> dict:
    """The three binary winner cards in MLB's schema, computed here.

    The shipped monitor winner_cards carry only {n, brier, logloss, ece} and
    in the 20260926 artifact both report the exact no-information values, so
    the cards are computed from the decided rows rather than read.
    """
    empty = {"n": 0, "win_rate": None, "auc": None, "brier": None,
             "logloss": None, "ece_raw": None, "ece_calibrated": None,
             "predicted_mean": None, "actual_win_rate": None}
    if decided is None or not len(decided):
        return {key: dict(empty) for key, _, _ in WINNER_CARDS}

    total = pd.to_numeric(decided.get("total"), errors="coerce")
    over_p = _over_fair_series(decided)
    ok = total.notna() & over_p.notna()
    lines = fair_total_lines(decided)
    med_line = _int_line(np.nanmedian(lines[ok.to_numpy()])) if ok.any() else None
    over = binary_card_metrics(
        (total[ok] > pd.Series(lines, index=decided.index)[ok]).to_numpy(float),
        over_p[ok].to_numpy(float),
        calibration_coefficients(monitor, "totals", med_line))

    rl = runline_history_frame(decided)
    spread = dict(empty)
    if len(rl):
        okr = rl["correct"].notna().to_numpy()
        picked_home = (rl["pick"] == rl["home"]).to_numpy()
        p_pick = np.asarray(
            np.where(picked_home, rl["pick_prob"], 1.0 - rl["pick_prob"]),
            dtype=float)
        covered = np.asarray(
            np.where(picked_home, rl["winner"] == rl["home"],
                     rl["winner"] != rl["home"]), dtype=float)
        y = np.where(okr, covered, np.nan)
        spread = binary_card_metrics(
            y[okr], p_pick[okr],
            calibration_coefficients(
                monitor, "run_line",
                _int_line(pd.to_numeric(rl.loc[rl["correct"].notna(), "line"],
                                        errors="coerce").median())))
        if okr.any():
            spread["win_rate"] = round(float(y[okr].mean()), 6)

    margin = pd.to_numeric(decided.get("margin"), errors="coerce")
    home_p = pd.to_numeric(decided.get("p_home_win_derived"), errors="coerce")
    okh = margin.notna() & home_p.notna()
    derived = binary_card_metrics(
        (margin[okh] > 0).to_numpy(float), home_p[okh].to_numpy(float))
    return {"over_under": over, "run_line": spread, "derived_ml": derived}


def _int_line(value) -> int | None:
    v = _num(value)
    return None if v is None else int(round(v))


def rolling_history_rows(rolling: dict) -> list[dict]:
    """MLB's rolling-history table rows (last 10 points per card)."""
    rows: list[dict] = []
    for key, label, _rule in WINNER_CARDS:
        for pt in ((rolling or {}).get(key) or [])[-10:]:
            rows.append({"Line": label, "Date": pt.get("date", "--"),
                         "ECE-cal": _fmt3(pt.get("ece_calibrated")),
                         "Brier": _fmt3(pt.get("brier")),
                         "Logloss": _fmt3(pt.get("logloss"), 4),
                         "Pred mean": _fmt3(pt.get("predicted_mean")),
                         "n": pt.get("n", 0)})
    return rows[-30:]


def _fmt3(value, digits: int = 3) -> str:
    v = _num(value)
    return "—" if v is None else f"{v:.{digits}f}"


# ---------------------------------------------------------------------------
# Artifact-backed sections (drift / coverage) — MLB's fb-box tables
# ---------------------------------------------------------------------------


def load_run_engine_csv(ds: str, prefix: str) -> pd.DataFrame | None:
    filename = f"{prefix}_{str(ds).replace('-', '')}.csv"
    raw, _ = utils._fetch_bytes(filename, **utils.get_source_config(), sport="nba")
    if raw is None:
        return None
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception:
        return None


def _cell(value) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    return str(value)


def _run_engine_weight_pcts(records: list[dict],
                            weights: dict | None) -> list:
    """Per-row MODEL WEIGHT cells for the run-engine drift table.

    Source of truth: the run-engine drift CSV's own ``weight_pct`` column —
    the DISTRIBUTION model's weights (pooled per-side Poisson LightGBM
    importances of the score regressor, emitted since 2026-10-08). An
    explicit ``weights`` map (test seam) takes precedence per feature; rows
    with no weight of their own render None (the table's '—') — the binary
    moneyline blend's map is never substituted, so the Totals & Run Lines
    page reports the distribution model and nothing else.
    """
    weights = weights or {}
    out = []
    for r in records:
        f = str(r.get("feature", ""))
        w = weights.get(f)
        if w is None:
            w = r.get("weight_pct")
        # NaN is "no weight" (a pandas-read CSV turns absent cells into NaN
        # and would render as the literal 'nan%') — normalize to None so
        # the cell renders as an em-dash and a fully unweighted artifact
        # omits the column entirely.
        if w is None or (isinstance(w, float) and pd.isna(w)):
            w = None
        out.append(w)
    return out


def render_run_engine_drift(frame: pd.DataFrame | None,
                            weights: dict | None = None,
                            served_metadata: dict | None = None) -> None:
    """Run-engine feature drift — MLB's PSI fb-box table, column for column.

    PSI ADJ./SHIFT SE carry the same decision columns as MLB's monitor
    table, and MODEL WEIGHT = the DISTRIBUTION model's own per-feature
    importance shipped in the drift CSV's ``weight_pct`` (pooled per-side
    Poisson LightGBM importances across its home/away score fits, summing
    to 100%). Rows with no weight render '—' — the binary moneyline
    blend's weights are never substituted. The column is omitted entirely
    when the artifact carries no weights at all (legacy artifacts), so the
    table still renders.
    """
    st.markdown("### Run-Engine Feature Drift (PSI)")
    if frame is None or frame.empty:
        st.info("No run-engine drift data for this date "
                "(nba_run_engine_feature_drift_*.csv appears after a "
                "pipeline run).")
        return
    records = frame.to_dict("records")
    # MODEL WEIGHT per row — the DISTRIBUTION model's own weights, shipped
    # in the run-engine drift CSV itself (see module docstring). The binary
    # moneyline blend's shared map is NEVER borrowed: rows without their
    # own weight render '—' (never relabeled moneyline importance). Every
    # cell is formatted by the SAME helper the Model Monitor uses.
    weight_pcts = _run_engine_weight_pcts(records, weights)
    has_weights = any(w is not None for w in weight_pcts)
    weight_header = "<th>MODEL WEIGHT</th>" if has_weights else ""
    rows = []
    for r, w in zip(records, weight_pcts):
        psi = _num(r.get("psi"))
        psi_str = "—" if psi is None else f"{psi:.3f}"
        status = r.get("status", "OK")
        # Decision columns, mirroring MLB's drift table: the status is
        # assigned on the NOISE-ADJUSTED PSI under a 2-SE location gate —
        # show both so raw PSI cannot read self-contradictory.
        psi_adj = _num(r.get("psi_adjusted"))
        shift_se = _num(r.get("shift_se"))
        psi_adj_str = "—" if psi_adj is None else f"{psi_adj:.3f}"
        shift_se_str = "—" if shift_se is None else f"{shift_se:.3f}"
        psi_color = (utils.AMBER if status == "WARN"
                     else utils.RED if status == "ALERT" else utils.TEXT)
        pill_cls = {"OK": "ok", "WARN": "warn", "ALERT": "alert",
                    "INSUFFICIENT": "ok", "STRUCTURAL": "ok"}.get(status, "ok")
        n_base, n_cur = r.get("n_baseline"), r.get("n_current")
        samples = (f" ({n_base}/{n_cur})"
                   if n_base is not None and n_cur is not None else "")
        label = utils.describe_feature(r.get("feature", ""),
                                       served_metadata=served_metadata) \
            or r.get("feature", "")
        # A STRUCTURAL row's reason is the finding (which constant, in
        # both windows); render it under the pill so the table answers
        # "why is this not a verdict" without a caption hunt.
        reason_cell = (
            f"<div style='color:#64748B;font-size:0.72rem;"
            f"font-weight:400;margin-top:1px;'>"
            f"{r.get('structural_reason')}</div>"
        ) if r.get("structural_reason") else ""
        weight_cell = (f"<td>{utils.feature_weight_pct({'weight_pct': w})}</td>"
                       if has_weights else "")
        rows.append(
            f"<tr>"
            f"<td style='color:#E2E8F0;'>{r.get('feature','')}"
            f"<div style='color:#94A3B8;font-size:0.72rem;font-weight:400;"
            f"margin-top:1px;'>{label}</div></td>"
            f"<td>{_cell(r.get('current_mean'))}</td>"
            f"<td>{_cell(r.get('baseline_mean'))}</td>"
            f"<td style='color:{psi_color};font-weight:700;'>{psi_str}</td>"
            f"<td style='color:{psi_color};'>{psi_adj_str}</td>"
            f"<td style='color:#64748B;'>{shift_se_str}</td>"
            f"{weight_cell}"
            f"<td><span class='fb-status-pill {pill_cls}'>{status}</span>"
            f"{reason_cell}"
            f"<span style='color:#64748B;font-size:0.72rem;margin-left:5px;'>"
            f"{samples}</span></td></tr>")
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <table class="fb-table">
            <thead><tr><th>FEATURE</th><th>CURRENT MEAN</th><th>BASELINE MEAN</th>
            <th>PSI</th><th>PSI ADJ.</th><th>SHIFT SE</th>
            {weight_header}<th>STATUS</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        <div style="color:#64748B;font-size:0.78rem;margin-top:6px;">
          Same windows as the moneyline drift; STATUS is assigned on
          PSI ADJ. = raw PSI − sampling-noise floor, escalated only when the
          mean also moved &gt; 2× the location SE (the pooled standard error
          widened 1.5× for within-window clustering — about 3× the SHIFT SE
          column above) (location gate). INSUFFICIENT =
          window too small to judge drift. STRUCTURAL = constant at the
          same value in both windows (cannot drift) — a stable fact,
          not a verdict. MODEL WEIGHT = the distribution
          model's own feature importance (pooled per-side Poisson LightGBM
          importances across its home/away score fits, summing to 100%; '—'
          = no weight for this feature on this artifact).
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_run_engine_coverage(frame: pd.DataFrame | None) -> None:
    """Run-engine feature coverage — MLB's measured/non-null fb-box table.

    Worst-first with NO healthy-tail truncation: every feature-window pair
    stays visible (the NBA page previously capped the healthy tail at 12
    rows while MLB showed them all — 2026-10-08 parity fix).
    """
    st.markdown("### Run-Engine Feature Coverage (non-null / measured)")
    if frame is None or frame.empty:
        st.info("No run-engine coverage data for this date "
                "(nba_run_engine_feature_coverage_*.csv appears after a "
                "pipeline run).")
        return
    cov_sorted = sorted(frame.to_dict("records"),
                        key=lambda r: (_num(r.get("pct_measured")) or 0.0,
                                       str(r.get("feature", ""))))
    n_starved = sum(1 for r in cov_sorted if r.get("status") == "STARVED")
    n_low = sum(1 for r in cov_sorted if r.get("status") == "LOW_COVERAGE")
    sub = (
        f"<span style='color:{utils.RED};font-weight:700;'>{n_starved} "
        f"starved</span> · <span style='color:{utils.AMBER};font-weight:700;'>"
        f"{n_low} low</span>"
        if (n_starved or n_low) else
        "<span style='color:#4ADE80;font-weight:700;'>all windows healthy"
        "</span>")
    st.markdown(
        f"<div style='color:#94A3B8;font-size:0.8rem;margin:-6px 0 10px;'>"
        f"Share of games in each drift window with a real observation per "
        f"feature — {sub}</div>",
        unsafe_allow_html=True)
    rows = []
    for r in cov_sorted:
        status = r.get("status", "OK")
        pct_m = _num(r.get("pct_measured")) or 0.0
        pct_n = _num(r.get("pct_nonnull")) or 0.0
        n_def = int(_num(r.get("n_default_zero")) or 0)
        color = (utils.RED if status == "STARVED"
                 else utils.AMBER if status == "LOW_COVERAGE" else utils.TEXT)
        pill_cls = {"OK": "ok", "LOW_COVERAGE": "warn",
                    "STARVED": "alert"}.get(status, "ok")
        default_cell = (
            f"<div style='color:#94A3B8;font-size:0.72rem;font-weight:400;"
            f"margin-top:1px;'>{n_def} default-zero</div>" if n_def else "")
        rows.append(
            f"<tr>"
            f"<td style='color:#E2E8F0;'>{r.get('feature','')}</td>"
            f"<td>{_cell(r.get('window'))}</td>"
            f"<td>{_cell(r.get('n_games'))}</td>"
            f"<td style='color:{color};font-weight:700;'>{pct_m:.0f}%</td>"
            f"<td>{pct_n:.0f}%{default_cell}</td>"
            f"<td><span class='fb-status-pill {pill_cls}'>{status}</span></td>"
            f"</tr>")
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
          % NON-NULL includes them. STARVED &lt;25% measured,
          LOW_COVERAGE &lt;80%. Rows are listed worst-first — every
          feature-window pair stays visible.
        </div>
        """,
        unsafe_allow_html=True,
    )


def fold_slate_history(monitors: list[dict]) -> dict[str, list[dict]]:
    folded: dict[str, list[dict]] = {}
    for monitor in monitors or []:
        for row in monitor.get("slate_history", []) or []:
            key = str(row.get("game_id", "unknown"))
            folded.setdefault(key, []).append(row)
    return {key: rows[-10:] for key, rows in folded.items()}
