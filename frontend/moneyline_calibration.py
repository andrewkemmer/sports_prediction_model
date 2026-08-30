"""Moneyline Calibration page: per-1% favored-probability aggregation and the
merged confidence-vs-accuracy + calibration-curve chart builder.

Pure computation + an Altair spec builder — no Streamlit — so it is testable
directly. The page layer only renders what these builders produce.

Convention (shared with the moneyline calibration curve): every OOF prediction
is taken from the FAVORED side (probability >= 50%) and binned to the nearest
1% predicted probability. Each 1% slice yields one curve point AND one count
bar, so the curve and the bars always align one-to-one on the same x-axis.
"""

from __future__ import annotations

from typing import Optional

import altair as alt
import numpy as np
import pandas as pd

# Favored-side 1% bin width for the moneyline calibration curve.
FAVORED_BIN = 0.01

BLUE = "#3B82F6"
GREEN = "#34D399"
GRAY = "#64748B"


def favored_calibration_pts(hist_curve: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Per-1%-favored-probability calibration frame from the prediction history.

    For each OOF prediction: take max(p, 1-p) (favored side, >= 50%), round to
    the nearest 1%, and aggregate by that bin into one row per populated slice:
    prob (bin center), win_rate (actual win rate of the slice), n (games in the
    slice). ``n`` IS the observation count the merged count bars render — sum(n)
    == total decided games. An empty frame (never an error) is returned when the
    history lacks the prediction/outcome columns, so the page can fall back to
    the 10-point artifact curve.
    """
    cols = {"home_win_prob_model", "correct"}
    if hist_curve is None or hist_curve.empty or not cols <= set(hist_curve.columns):
        return pd.DataFrame(columns=["prob", "win_rate", "n"])
    p = pd.to_numeric(hist_curve["home_win_prob_model"], errors="coerce")
    w = pd.to_numeric(hist_curve["correct"], errors="coerce")
    ok = p.notna() & w.notna()
    fav = np.maximum(p[ok], 1.0 - p[ok])
    fav = (fav / FAVORED_BIN).round() * FAVORED_BIN        # nearest 1%
    return (pd.DataFrame({"prob": fav, "won": w[ok]})
            .groupby("prob")
            .agg(win_rate=("won", "mean"), n=("won", "size"))
            .reset_index())


def chart_favored_calibration(pts: pd.DataFrame,
                              pts_cal: pd.DataFrame) -> dict:
    """Merged confidence-vs-accuracy + calibration-curve chart (moneyline page).

    One chart, no information loss: bars (LEFT 'Games' axis) = observation
    count per predicted-probability bin; the blue actual-rate curve and the
    green deployed Platt map share a RIGHT '%' axis (0-100, independent of the
    count axis); the gray dashed line is the perfect-calibration diagonal. Bars
    and the blue curve use the SAME ``pts`` frame and the SAME 1% x-bins — ``n``
    drives both the bar height and the curve point, so they align one-to-one
    (bar heights sum to the total decided games).

    Returns {'chart': the layered spec, 'bars': the per-bin counts frame (``n``
    per ``prob``), 'n_total': sum of bar heights}. Empty/low-n bins are simply
    absent (no fabricated points) and render without error.
    """
    if pts is None or len(pts) == 0:
        pts = pd.DataFrame(columns=["prob", "win_rate", "n"])
    pts = pts.copy()
    pts["win_rate_pct"] = pts["win_rate"] * 100.0
    x_dom = alt.Scale(domain=[0.45, 1.0])
    y_dom = alt.Scale(domain=[0, 100.0])

    bars = alt.Chart(pts).mark_bar(color=BLUE, opacity=0.30).encode(
        x=alt.X("prob:Q", title="Predicted win probability", scale=x_dom),
        # LEFT axis (single owner): 'Games'. No right-axis title here — the
        # right '%' title belongs to the curves layer below.
        y=alt.Y("n:Q", axis=alt.Axis(title="Games", grid=True)),
        tooltip=[
            alt.Tooltip("prob:Q", title="Predicted", format=".0%"),
            alt.Tooltip("n:Q", title="Games"),
            alt.Tooltip("win_rate_pct:Q", title="Actual win rate %", format=".1f"),
        ],
    )
    model_pts = alt.Chart(pts).mark_line(
        point=alt.OverlayMarkDef(filled=True, size=55), color=BLUE,
        strokeWidth=2.5).encode(
        x=alt.X("prob:Q", scale=x_dom),
        # RIGHT axis (single owner): 'Actual win rate %' for the blue actual-
        # rate curve — the green Platt line below shares this right scale and
        # deliberately carries NO title (avoid the overlapping-title bug).
        y=alt.Y("win_rate_pct:Q",
                axis=alt.Axis(title="Actual win rate %", orient="right",
                              grid=False),
                scale=y_dom),
        tooltip=[
            alt.Tooltip("prob:Q", title="Predicted", format=".0%"),
            alt.Tooltip("win_rate_pct:Q", title="Actual win rate %", format=".1f"),
            alt.Tooltip("n:Q", title="Games"),
        ],
    )
    diag_df = pd.DataFrame({"prob": [0.45, 1.0], "win_rate_pct": [45.0, 100.0]})
    diag = alt.Chart(diag_df).mark_line(
        color=GRAY, strokeDash=[5, 5], strokeWidth=1.5).encode(
        x=alt.X("prob:Q", scale=x_dom),
        # Diagonal maps to the same right scale but renders NO axis/title
        # (axis=None) so it never contributes an overlapping label.
        y=alt.Y("win_rate_pct:Q", axis=None, scale=y_dom),
    )
    layers = [bars, diag, model_pts]
    if pts_cal is not None and len(pts_cal):
        pcal = pts_cal.copy()
        pcal["cal_mean_pct"] = pcal["cal_mean"] * 100.0
        cal_line = alt.Chart(pcal).mark_line(
            point=alt.OverlayMarkDef(filled=True, size=45), color=GREEN,
            strokeDash=[6, 4], strokeWidth=2).encode(
            x=alt.X("prob:Q", scale=x_dom),
            # Aligns with model_pts on the right scale but carries NO title
            # (single source = the curves layer above owns 'Actual win rate %').
            y=alt.Y("cal_mean_pct:Q",
                    axis=alt.Axis(title=None, orient="right")),
            tooltip=[
                alt.Tooltip("prob:Q", title="Raw predicted", format=".0%"),
                alt.Tooltip("cal_mean:Q", title="Calibrated prediction", format=".1%"),
                alt.Tooltip("n:Q", title="Games"),
            ],
        )
        layers.append(cal_line)
    chart = alt.layer(*layers).resolve_scale(
        x="shared", y="independent").properties(
        width="container", height=340)
    return {"chart": chart, "bars": pts, "n_total": int(pts["n"].sum())}