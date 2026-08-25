"""Pure computation + chart builders for the Markets page Diagnostics section.

Read-only over run_engine_markets_<date>.csv — no model, config, or metric
changes; no MC re-runs, no refits. Every compute function returns plain data
(plus an optional warning string) so it is testable without Streamlit; the
page layer only renders what these builders produce.

Design decisions inherited from Phase 3 (not revisited here):
- The OVER side is the calibrated quantity; p_under is its exact mirror
  (1 − p_over), so low p_over IS the under-favored region.
- Per-bucket accuracy charts use the favored-side pick probability
  max(p_over, 1 − p_over).

Offset handling (documented choice): relativized lines (expected_total +
offset) are priced by MONOTONE LOGIT-LINEAR INTERPOLATION between the two
bracketing precomputed grid columns per game — the artifact stays untouched,
and logit-space interpolation preserves the grid's monotone-decreasing shape.
Lines outside [6.5, 12.5] clamp to the nearest edge column.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import altair as alt
import numpy as np
import pandas as pd

TOTAL_GRID = [round(6.5 + 0.5 * i, 1) for i in range(13)]   # 6.5 … 12.5
RUN_COVER_COL = "p_home_cover_1_5"
OFFSET_EDGES = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
BUCKET_EDGES = [50, 55, 60, 65, 70, 75, 101]
BUCKET_LABELS = ["50-55", "55-60", "60-65", "65-70", "70-75", "75+"]


def decided_rows(markets: Optional[pd.DataFrame]) -> pd.DataFrame:
    """OOF rows with known outcomes — everything else is excluded loudly."""
    if markets is None or not len(markets):
        return pd.DataFrame()
    df = markets[(markets.get("kind") == "oof")]
    if "total_runs" in df.columns:
        df = df[df["total_runs"].notna()]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chart 1 — totals distribution fit-check
# ---------------------------------------------------------------------------
def _nb_pmf_scalar(k: int, mu: float, alpha: float) -> float:
    """NB(k; μ, α) via math.lgamma — scipy-free for the dashboard host."""
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    alpha = max(float(alpha), 1e-9)
    n = 1.0 / alpha
    p = n / (n + mu)
    if k < 0:
        return 0.0
    logp = (math.lgamma(k + n) - math.lgamma(n) - math.lgamma(k + 1)
            + n * math.log(p) + k * math.log1p(-p))
    return math.exp(logp)


def total_distribution(decided: pd.DataFrame, kmax: int = 15) -> dict[str, Any]:
    """Observed P(total=k) vs modeled mean game-level NB convolution.

    Modeled marginal = average over games of the convolution of the two
    per-game side marginals NB(λ_side, α_side) — exactly what the Monte
    Carlo samples from, evaluated analytically.
    """
    empty = {"ks": list(range(kmax + 1)), "observed": [], "modeled": [],
             "callouts": {}, "n_games": 0,
             "warning": "No decided games with outcomes in the artifact."}
    need = {"total_runs", "home_expected_runs", "away_expected_runs",
            "alpha_home", "alpha_away"}
    if not len(decided) or not need.issubset(decided.columns):
        return empty
    ks = np.arange(0, kmax + 1)
    observed = [(decided["total_runs"] == k).mean() for k in ks]
    modeled = np.zeros(len(ks))
    lam_h = decided["home_expected_runs"].to_numpy(float)
    lam_a = decided["away_expected_runs"].to_numpy(float)
    a_h = np.maximum(decided["alpha_home"].to_numpy(float), 1e-9)
    a_a = np.maximum(decided["alpha_away"].to_numpy(float), 1e-9)
    for i in range(len(decided)):
        ph = [_nb_pmf_scalar(int(k), lam_h[i], a_h[i]) for k in ks]
        pa = [_nb_pmf_scalar(int(k), lam_a[i], a_a[i]) for k in ks]
        conv = np.convolve(ph, pa)[:len(ks)]
        modeled += conv
        # Tail mass beyond kmax flows into the last bucket implicitly via
        # normalization below; keep raw means (both series share support).
    modeled /= max(len(decided), 1)
    obs_le1 = float(sum(observed[:2]))
    obs_ge10 = float(sum(observed[10:]))
    mod_le1 = float(modeled[:2].sum())
    mod_ge10 = float(modeled[10:].sum())
    return {
        "ks": ks.tolist(),
        "observed": [round(float(v), 5) for v in observed],
        "modeled": [round(float(v), 5) for v in modeled],
        "callouts": {
            "P(total<=1)": {"observed": round(obs_le1, 4),
                            "modeled": round(mod_le1, 4)},
            "P(total>=10)": {"observed": round(obs_ge10, 4),
                             "modeled": round(mod_ge10, 4)},
            "note": ("Per-team tail checks (P(X>=10) home/away) live in the "
                     "fit-check table above; this chart is the TOTALS law."),
        },
        "n_games": int(len(decided)),
        "warning": None,
    }


# ---------------------------------------------------------------------------
# Charts 2–4 — calibration curves
# ---------------------------------------------------------------------------
def _logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    out = np.log(p / (1 - p))
    return out if out.ndim else float(out)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    out = 1.0 / (1.0 + np.exp(-np.asarray(x, float)))
    return out if np.ndim(out) else float(out)


def over_prob_at_lines(df: pd.DataFrame, lines: np.ndarray) -> np.ndarray:
    """Grid-column p_over priced at arbitrary half-step lines via monotone
    logit-linear interpolation; clamped outside [6.5, 12.5]."""
    lines = np.asarray(lines, float)
    grid = np.asarray(TOTAL_GRID)
    lo_idx = np.clip(np.floor((lines - grid[0]) / 0.5).astype(int), 0,
                     len(grid) - 2)
    frac = np.clip((lines - grid[lo_idx]) / 0.5, 0.0, 1.0)
    cols = [f"p_over_{str(g).replace('.', '_')}" for g in TOTAL_GRID]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"markets artifact lacks grid columns: {missing[:3]}…")
    mat = df[cols].to_numpy(float)
    p_lo = mat[np.arange(len(df)), lo_idx]
    p_hi = mat[np.arange(len(df)), lo_idx + 1]
    return np.clip(_sigmoid((1 - frac) * _logit(p_lo) + frac * _logit(p_hi)),
                   0.0, 1.0)


def relativized_pairs(decided: pd.DataFrame,
                      offsets: Optional[list[float]] = None) -> pd.DataFrame:
    """(p_over, did_go_over) pairs at line = expected_total + offset.

    Each offset re-prices every game at ITS OWN shifted line — that is why
    the pooled probability axis spans the full ~0.05–0.95 range.
    """
    offsets = OFFSET_EDGES if offsets is None else offsets
    if not len(decided) or "total_runs" not in decided.columns:
        return pd.DataFrame(columns=["p", "y", "offset"])
    exp_total = (decided["home_expected_runs"].to_numpy(float)
                 + decided["away_expected_runs"].to_numpy(float)) \
        if "expected_total" not in decided.columns \
        else decided["expected_total"].to_numpy(float)
    total = decided["total_runs"].to_numpy(float)
    frames = []
    for off in offsets:
        lines = np.round((exp_total + off) * 2) / 2  # snap to nearest half-step
        p = over_prob_at_lines(decided, lines)
        y = (total >= lines + 0.5).astype(float)
        frames.append(pd.DataFrame({"p": p, "y": y, "offset": off}))
    return pd.concat(frames, ignore_index=True)


def fixed_line_pairs(decided: pd.DataFrame,
                     lines: tuple[float, ...]) -> pd.DataFrame:
    """(p_over, outcome) pairs at fixed published lines, one row per
    (game, line)."""
    if not len(decided) or "total_runs" not in decided.columns:
        return pd.DataFrame(columns=["p", "y", "line"])
    total = decided["total_runs"].to_numpy(float)
    frames = []
    for line in lines:
        col = f"p_over_{str(line).replace('.', '_')}"
        if col not in decided.columns:
            continue
        p = decided[col].to_numpy(float)
        y = (total >= math.ceil(line)).astype(float)
        frames.append(pd.DataFrame({"p": p, "y": y, "line": line}))
    return pd.concat(frames, ignore_index=True) if frames \
        else pd.DataFrame(columns=["p", "y", "line"])


def calibration_curve(pairs: pd.DataFrame, n_bins: int = 20,
                      min_count: int = 30) -> dict[str, Any]:
    """Equal-width reliability bins (dropping bins under min_count)."""
    empty = {"bins": [], "n_pairs": 0, "n_dropped_bins": 0,
             "warning": "No (prediction, outcome) pairs to calibrate."}
    if not len(pairs):
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
# Charts 5–6 — favored-side pick accuracy buckets
# ---------------------------------------------------------------------------
def pick_buckets(p_pick_prob: np.ndarray, hit: np.ndarray,
                 labels: Optional[list[str]] = None) -> dict[str, Any]:
    """Count + hit rate per confidence bucket on the FAVORED-side probability.
    Hit rate is NOT calibration — it is binary pick accuracy per bucket."""
    labels = BUCKET_LABELS if labels is None else labels
    p = np.asarray(p_pick_prob, float)
    h = np.asarray(hit, float)
    ok = np.isfinite(p) & np.isfinite(h)
    p, h = p[ok], h[ok]
    rows, warning = [], None
    if not len(p):
        return {"buckets": [], "n_games": 0,
                "warning": "No decided games available for picks."}
    pct = np.clip(p, 0.5, 1.0) * 100.0
    for i, lab in enumerate(labels):
        lo = BUCKET_EDGES[i]
        hi = BUCKET_EDGES[i + 1]
        m = (pct >= lo) & (pct < hi) if i < len(labels) - 1 else (pct >= lo)
        n = int(m.sum())
        rows.append({
            "bucket": lab,
            "count": n,
            "accuracy": (round(float(h[m].mean()) * 100, 2)
                         if n else None),
        })
    return {"buckets": rows, "n_games": int(len(p)), "warning": warning}


def overs_pick_table(decided: pd.DataFrame, line: float = 8.5) -> dict:
    col = f"p_over_{str(line).replace('.', '_')}"
    if not len(decided) or col not in decided.columns \
            or "total_runs" not in decided.columns:
        return {"buckets": [], "n_games": 0,
                "warning": "Missing over-probability column or outcomes."}
    p = decided[col].to_numpy(float)
    pick_over = p >= 0.5
    hit = (pick_over.astype(float)
           == (decided["total_runs"].to_numpy(float) >= math.ceil(line)).astype(float))
    out = pick_buckets(np.maximum(p, 1 - p), hit.astype(float))
    out["pick_rule"] = f"over if P(over {line}) >= 0.5"
    return out


def runline_pick_table(decided: pd.DataFrame,
                       margin_col: str = RUN_COVER_COL) -> dict:
    if not len(decided) or margin_col not in decided.columns \
            or {"home_score", "away_score"}.difference(decided.columns):
        return {"buckets": [], "n_games": 0,
                "warning": "Missing cover-probability column or outcomes."}
    p = decided[margin_col].to_numpy(float)
    home_covers = ((decided["home_score"] - decided["away_score"]).to_numpy(float)
                   >= 2).astype(float)
    pick_home = p >= 0.5
    hit = (pick_home.astype(float) == home_covers)
    out = pick_buckets(np.maximum(p, 1 - p), hit.astype(float))
    out["pick_rule"] = ("home -1.5 cover if P(home -1.5) >= 0.5, "
                        "else away +1.5")
    return out


# ---------------------------------------------------------------------------
# Altair builders (import-safe: pure functions of their data)
# ---------------------------------------------------------------------------
def chart_distribution(dist: dict) -> alt.Chart:
    df = pd.DataFrame({
        "k": dist["ks"] * 2,
        "series": ["observed"] * len(dist["ks"]) + ["modeled"] * len(dist["ks"]),
        "p": dist["observed"] + dist["modeled"],
    })
    bars = alt.Chart(df[df.series == "observed"]).mark_bar(
        color="#3B82F6", opacity=0.65).encode(
        x=alt.X("k:Q", title="Total runs (home + away)"),
        y=alt.Y("p:Q", title="P(total = k)", axis=alt.Axis(format="%")),
    )
    line = alt.Chart(df[df.series == "modeled"]).mark_line(
        color="#F59E0B", strokeWidth=2.5, point=True).encode(
        x="k:Q", y="p:Q")
    return (bars + line).properties(height=300)


def chart_calibration(curve: dict, title: str,
                      x_domain: Optional[list[float]] = None) -> alt.Chart:
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


def chart_pick_buckets(table: dict, title: str) -> dict:
    """Returns {'chart': alt.Chart, 'table': pd.DataFrame} for dual-axis
    rendering; count bars + accuracy line are layered in the page."""
    tdf = pd.DataFrame(table["buckets"])
    if tdf.empty:
        return {"chart": alt.Chart(pd.DataFrame()).mark_bar(), "table": tdf}
    base = alt.Chart(tdf).encode(
        x=alt.X("bucket:N", sort=list(tdf["bucket"]), title=None))
    bars = base.mark_bar(color="#3B82F6").encode(
        y=alt.Y("count:Q", title="Picks"),
        tooltip=["bucket", "count", "accuracy"])
    acc = base.mark_line(color="#22C55E", strokeWidth=2.5,
                         point=alt.OverlayMarkDef(size=60)).encode(
        y=alt.Y("accuracy:Q", title="Actual hit %"),
        tooltip=["bucket", "count", "accuracy"])
    return {"chart": alt.layer(bars, acc).resolve_scale(y="independent")
            .properties(height=280, title=title),
            "table": tdf}
