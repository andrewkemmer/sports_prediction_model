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


def chart_calibration_curve(
    pts: pd.DataFrame,
    series: list,
    *,
    x_field: str = "prob",
    x_title: str = "Predicted win probability",
    x_scale: Optional[alt.Scale] = None,
    x_format: str = ".0%",
    n_field: str = "n",
    bar_color: str = BLUE,
    bar_opacity: Optional[float] = 0.30,
    bar_tooltips: Optional[list] = None,
    low_n_field: Optional[str] = None,
    low_n_color: str = GRAY,
    low_n_opacity: float = 0.45,
    y_pct_scale: Optional[alt.Scale] = None,
    diag_x: Optional[list] = None,
    diag_y_pct: Optional[list] = None,
    diag_y_field: Optional[str] = None,
    diag_color: str = GRAY,
    pooled: Optional[dict] = None,
    title: Optional[str] = None,
    height: int = 340,
) -> dict:
    """Shared layered calibration-curve builder — the moneyline 'Calibration
    Curve' grammar, reused VERBATIM by the Game Total Lines diagnostics tab
    so both render as the SAME chart type through the same full-width path
    (width='container', resolve_scale x shared / y independent, one title per
    axis).

    ``pts`` is the per-bin bars frame (columns: ``x_field``, ``n_field``,
    optionally ``low_n_field``) driving the LEFT 'Games' count axis;
    ``series`` is a list of curve-layer specs — each {"data": DataFrame (x +
    y fields), "y_field": %-scale field, "axis": alt.Axis (the first series
    owns the right-axis title; later series pass title=None), "color": str or
    alt.Color (alt.Color shares one legend across layers), "dash",
    "point_size", "stroke_width", "tooltips"} — on the shared RIGHT '%' axis
    (0-100, independent of the count axis). The gray dashed
    perfect-calibration diagonal and the optional amber pooled marker are
    scale-bound to the same axes.

    Returns {'chart': the layered spec, 'bars': the bars frame, 'n_total':
    sum of bar heights}. Empty/low-n bins are simply absent (no fabricated
    points) and render without error.
    """
    if pts is None or len(pts) == 0:
        pts = pd.DataFrame(columns=[x_field, n_field])
    pts = pts.copy()
    if x_scale is None:
        x_scale = alt.Scale(domain=[0.0, 1.0])
    if y_pct_scale is None:
        y_pct_scale = alt.Scale(domain=[0.0, 100.0])
    if bar_tooltips is None:
        bar_tooltips = [
            alt.Tooltip(f"{x_field}:Q", title="Predicted", format=x_format),
            alt.Tooltip(f"{n_field}:Q", title="Games"),
        ]

    def _x(title=alt.Undefined) -> alt.X:
        # title only when explicitly given — absent keys must stay absent
        # (never emit "title": null) so every layer byte-matches the
        # reference moneyline spec.
        kw = {"scale": x_scale}
        if title is not alt.Undefined:
            kw["title"] = title
        return alt.X(f"{x_field}:Q", **kw)

    def _y(field: str, axis=alt.Undefined, scale=alt.Undefined) -> alt.Y:
        # Explicit None (e.g. axis=None on the diagonal) must render as null,
        # while absent keys stay absent (e.g. no scale on the count axis).
        kw = {}
        if axis is not alt.Undefined:
            kw["axis"] = axis
        if scale is not alt.Undefined:
            kw["scale"] = scale
        return alt.Y(f"{field}:Q", **kw)

    # Count bars — LEFT 'Games' axis (the single owner of that title).
    mark_kw = {"color": bar_color}
    if bar_opacity is not None:
        mark_kw["opacity"] = bar_opacity
    bars = alt.Chart(pts).mark_bar(**mark_kw).encode(
        x=_x(title=x_title),
        y=_y(n_field, axis=alt.Axis(title="Games", grid=True)),
        tooltip=bar_tooltips)
    layers = [bars]

    # low-n bars render gray (n < LOW_N), axis-title-less so 'Games' is
    # owned by the main bars layer only (single title per axis).
    if low_n_field is not None:
        low = pts[pts[low_n_field].fillna(False).astype(bool)]
        if not low.empty:
            low_bars = alt.Chart(low).mark_bar(
                color=low_n_color, opacity=low_n_opacity).encode(
                x=_x(), y=_y(n_field, axis=alt.Axis(title=None)),
                tooltip=bar_tooltips)
            layers.append(low_bars)

    # Dashed perfect-calibration diagonal (y = x on the 0-100 % scale),
    # axis-less so it never emits a competing axis title.
    if diag_x is not None and diag_y_pct is not None and diag_y_field:
        diag_df = pd.DataFrame({x_field: diag_x, diag_y_field: diag_y_pct})
        diag = alt.Chart(diag_df).mark_line(
            color=diag_color, strokeDash=[5, 5], strokeWidth=1.5).encode(
            x=_x(), y=_y(diag_y_field, axis=None, scale=y_pct_scale))
        layers.append(diag)

    # Series curves on the shared RIGHT '%' axis. A series whose frame is
    # EMPTY (0 rows) still emits its layer (the moneyline blue curve always
    # exists even with no data) — only a missing frame is skipped.
    for s in series:
        data = s.get("data")
        if data is None:
            continue
        s_kw = {"point": alt.OverlayMarkDef(
            filled=True, size=s.get("point_size", 55)),
            "strokeWidth": s.get("stroke_width", 2.5)}
        color = s.get("color")
        if isinstance(color, str):
            s_kw["color"] = color
        if s.get("dash"):
            s_kw["strokeDash"] = s["dash"]
        y_scale = s.get("y_scale", alt.Undefined)
        layer = alt.Chart(data).mark_line(**s_kw).encode(
            x=_x(), y=_y(s["y_field"], axis=s.get("axis"), scale=y_scale),
            tooltip=s.get("tooltips") or [])
        if color is not None and not isinstance(color, str):
            layer = layer.encode(color=color)
        layers.append(layer)

    # Optional pooled marker (amber diamond) on the same axes — the chart
    # and the Total table row agree about the pooled calibration point.
    if pooled is not None:
        pool_df = pd.DataFrame({x_field: [pooled["x"]],
                                "pct": [pooled["y_pct"]]})
        pm = alt.Chart(pool_df).mark_point(
            shape="diamond", size=150, color="#F59E0B", filled=True).encode(
            x=_x(), y=_y("pct", axis=None, scale=y_pct_scale),
            tooltip=pooled.get("tooltips") or [])
        layers.append(pm)

    props = {"width": "container", "height": height}
    if title is not None:
        props["title"] = title
    chart = alt.layer(*layers).resolve_scale(
        x="shared", y="independent").properties(**props)
    return {"chart": chart, "bars": pts, "n_total": int(pts[n_field].sum())}


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

    Delegates to the shared ``chart_calibration_curve`` builder — the SAME
    chart type the Game Total Lines diagnostics tab renders.

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

    series = [{
        "data": pts,
        "y_field": "win_rate_pct",
        # RIGHT axis (single owner): 'Actual win rate %' for the blue actual-
        # rate curve — the green Platt line below shares this right scale and
        # deliberately carries NO title (avoid the overlapping-title bug).
        "axis": alt.Axis(title="Actual win rate %", orient="right",
                         grid=False),
        "color": BLUE, "dash": None, "point_size": 55, "stroke_width": 2.5,
        "y_scale": y_dom,
        "tooltips": [
            alt.Tooltip("prob:Q", title="Predicted", format=".0%"),
            alt.Tooltip("win_rate_pct:Q", title="Actual win rate %",
                        format=".1f"),
            alt.Tooltip("n:Q", title="Games"),
        ],
    }]
    if pts_cal is not None and len(pts_cal):
        pcal = pts_cal.copy()
        pcal["cal_mean_pct"] = pcal["cal_mean"] * 100.0
        series.append({
            "data": pcal,
            "y_field": "cal_mean_pct",
            "axis": alt.Axis(title=None, orient="right"),
            "color": GREEN, "dash": [6, 4], "point_size": 45,
            "stroke_width": 2,
            "tooltips": [
                alt.Tooltip("prob:Q", title="Raw predicted", format=".0%"),
                alt.Tooltip("cal_mean:Q", title="Calibrated prediction",
                            format=".1%"),
                alt.Tooltip("n:Q", title="Games"),
            ],
        })
    built = chart_calibration_curve(
        pts, series,
        x_field="prob", x_title="Predicted win probability",
        x_scale=x_dom, x_format=".0%", n_field="n",
        bar_color=BLUE, bar_opacity=0.30,
        bar_tooltips=[
            alt.Tooltip("prob:Q", title="Predicted", format=".0%"),
            alt.Tooltip("n:Q", title="Games"),
            alt.Tooltip("win_rate_pct:Q", title="Actual win rate %",
                        format=".1f"),
        ],
        y_pct_scale=y_dom,
        diag_x=[0.45, 1.0], diag_y_pct=[45.0, 100.0],
        diag_y_field="win_rate_pct",
        height=340,
    )
    return {"chart": built["chart"], "bars": pts,
            "n_total": int(pts["n"].sum())}


# ---------------------------------------------------------------------------
# Game Total Lines diagnostics chart — SAME builder as the moneyline curve
# ---------------------------------------------------------------------------

# Fixed x-axis domain for the Game Total Lines calibration chart (the
# probability axis for predicted P(over)). Constant [0.25, 0.75] for ALL
# selections (All and every fixed line) — deliberately not adaptive, so the
# degenerate-domain failure class (dynamic min/max/padding/clamp) cannot
# recur. The dashed perfect-calibration diagonal is scale-bound and renders
# correctly at any domain.
GTL_X_DOMAIN = [0.25, 0.75]


def game_total_line_points(table: dict) -> pd.DataFrame:
    """Win-rate + observed line points for the game-total calibration chart —
    one row per (bin, series) over NON-low-n populated bins only (low-n
    points are dropped: n < LOW_N is not reliable calibration evidence).
    ``pct`` is on the 0-100 no-push 2-way basis. Sorted by (series,
    bin_center) so every series always connects in ascending x order — no
    bent/zig-zag line from an out-of-order or noisy slice."""
    rows = []
    for b in table.get("bins") or []:
        if b.get("observed") is None or b.get("low_n"):
            continue
        rows.append({"bin_center": b.get("bin_center"), "series": "Win rate",
                     "pct": round(b["win_rate"] * 100.0, 4),
                     "count": b["count"]})
        rows.append({"bin_center": b.get("bin_center"), "series": "Observed",
                     "pct": round(b["observed"] * 100.0, 4),
                     "count": b["count"]})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["series", "bin_center"]).reset_index(drop=True)
    return df


def chart_game_total_calibration(
        table: dict, title: str,
        obs_label: str = "Observed % (2-way, no push)") -> dict:
    """Game Total Lines calibration chart — the SAME chart type as the
    moneyline 'Calibration Curve', built through the shared
    ``chart_calibration_curve`` builder: count bars on a LEFT 'Games' axis,
    the observed + win-rate series on a RIGHT '%' axis, the gray dashed
    perfect-calibration diagonal, the amber pooled marker, the fixed x-domain
    [0.25, 0.75], the bottom 'Series' legend, the dynamic title
    'Calibration Curve — Over {line}', and height 480 — so it renders
    full-width exactly like the moneyline page. Low-n (< 30) bins render as
    gray bars and their curve points are dropped; per-series points connect
    in ascending bin order.

    Returns {'chart': the layered spec, 'table': per-bin rows + the pooled
    Total row (share 100%, the amber diamond on the chart)}.
    """
    tdf = pd.DataFrame(table["bins"])
    if tdf.empty:
        return {"chart": alt.Chart(pd.DataFrame()).mark_bar(), "table": tdf}
    chart_df = tdf.copy()
    bar_tip = [
        alt.Tooltip("bin_center:Q", title="Predicted P(over)", format=".3f"),
        alt.Tooltip("count:Q", title="Games"),
        alt.Tooltip("mean_pred:Q", title="Mean predicted", format=".3f"),
        alt.Tooltip("observed:Q", title="Observed", format=".3f"),
        alt.Tooltip("win_rate:Q", title="Win rate", format=".3f"),
    ]
    series = []
    stack = game_total_line_points(table)
    if not stack.empty:
        color_enc = alt.Color(
            "series:N",
            scale=alt.Scale(domain=["Observed", "Win rate"],
                            range=["#22C55E", "#8B5CF6"]),
            legend=alt.Legend(title="Series", orient="bottom",
                              titleAnchor="start", offset=14))
        for sname in ("Observed", "Win rate"):
            sub = stack[stack["series"] == sname]
            series.append({
                "data": sub,
                "y_field": "pct",
                "axis": alt.Axis(
                    title=(obs_label if sname == "Observed" else None),
                    orient="right", grid=False),
                "color": color_enc, "dash": None, "point_size": 60,
                "stroke_width": 2.5,
                "y_scale": alt.Scale(domain=[0.0, 100.0]),
                "tooltips": [
                    alt.Tooltip("bin_center:Q", title="Predicted P(over)",
                                format=".3f"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("pct:Q", title=obs_label, format=".1f"),
                    alt.Tooltip("count:Q", title="Games"),
                ],
            })
    pooled = None
    if (table.get("pooled_pred") is not None
            and table.get("pooled_observed") is not None):
        pooled = {
            "x": table["pooled_pred"],
            "y_pct": round(table["pooled_observed"] * 100.0, 4),
            "tooltips": [
                alt.Tooltip("bin_center:Q", title="Pooled predicted",
                            format=".3f"),
                alt.Tooltip("pct:Q", title="Pooled observed %", format=".1f"),
            ],
        }
    built = chart_calibration_curve(
        chart_df, series,
        x_field="bin_center", x_title="Predicted P(over)",
        x_scale=alt.Scale(domain=GTL_X_DOMAIN, nice=False), x_format=".3f",
        n_field="count", bar_color="#3B82F6", bar_opacity=None,
        bar_tooltips=bar_tip, low_n_field="low_n",
        low_n_color="#94A3B8", low_n_opacity=0.45,
        y_pct_scale=alt.Scale(domain=[0.0, 100.0]),
        diag_x=GTL_X_DOMAIN, diag_y_pct=[25.0, 75.0], diag_y_field="pct",
        pooled=pooled, title=title, height=480,
    )
    # Pooled (Total) table row — the pooled-aggregates summary, share 100%.
    total_row = pd.DataFrame([{
        "bin": "Total", "bin_center": None, "count": int(tdf["count"].sum()),
        "mean_pred": table.get("pooled_pred"),
        "observed": table.get("pooled_observed"),
        "win_rate": table.get("pooled_winrate"),
        "ece": table.get("pooled_ece"), "brier": table.get("pooled_brier"),
        "low_n": False, "share_pct": 100.0,
    }])
    table_df = pd.concat([tdf, total_row], ignore_index=True)
    return {"chart": built["chart"], "table": table_df}
