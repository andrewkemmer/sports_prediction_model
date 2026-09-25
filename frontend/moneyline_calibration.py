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

import math
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd

# The moneyline calibration curve is the SAME chart grammar as the market /
# Game Total Lines diagnostics, so it inherits their minimum-evidence rule
# verbatim rather than restating a second threshold: a bin with n < LOW_N is
# not reliable calibration evidence, so its bars render GRAY as volume
# context and its OBSERVED-rate point is dropped (no fabricated point).
# Imported (not copied) so the two pages cannot drift apart again.
from market_diagnostics import LOW_N  # noqa: E402  (no cycle: that module
# does not import this one)

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
        return pd.DataFrame(columns=["prob", "win_rate", "n", "low_n"])
    p = pd.to_numeric(hist_curve["home_win_prob_model"], errors="coerce")
    w = pd.to_numeric(hist_curve["correct"], errors="coerce")
    ok = p.notna() & w.notna()
    fav = np.maximum(p[ok], 1.0 - p[ok])
    fav = (fav / FAVORED_BIN).round() * FAVORED_BIN        # nearest 1%
    out = (pd.DataFrame({"prob": fav, "won": w[ok]})
           .groupby("prob")
           .agg(win_rate=("won", "mean"), n=("won", "size"))
           .reset_index())
    # Shared low-n flag (same rule + constant as the market-diagnostics
    # charts). Drives BOTH the gray bar treatment and the curve-point drop,
    # so bar and curve can never disagree about which bins are evidence.
    out["low_n"] = (out["n"] < LOW_N) & (out["n"] > 0)
    return out


def favored_oof_calibration_pts(hist_curve: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Build the published calibration series from stored OOF outputs.

    The production writer persists both the raw favored probability and the
    per-fold calibrated probability for every OOF game. This function only
    groups those already-computed OOF values for display; it never refits or
    re-evaluates the production calibration map in the frontend.
    """
    cols = {"home_win_prob_model", "home_win_prob_model_calibrated"}
    if hist_curve is None or hist_curve.empty or not cols <= set(hist_curve.columns):
        return pd.DataFrame(columns=["prob", "cal_mean", "n"])

    raw = pd.to_numeric(hist_curve["home_win_prob_model"], errors="coerce")
    calibrated = pd.to_numeric(
        hist_curve["home_win_prob_model_calibrated"], errors="coerce")
    ok = raw.notna() & calibrated.notna()
    raw_favored = np.maximum(raw[ok], 1.0 - raw[ok])
    calibrated_favored = np.where(raw[ok] >= 0.5,
                                  calibrated[ok], 1.0 - calibrated[ok])
    bins = (raw_favored / FAVORED_BIN).round() * FAVORED_BIN
    return (pd.DataFrame({"prob": bins, "cal_mean": calibrated_favored})
            .groupby("prob")
            .agg(cal_mean=("cal_mean", "mean"), n=("cal_mean", "size"))
            .reset_index())


def deployed_probability(raw: np.ndarray, params: Optional[dict]) -> np.ndarray:
    """Apply the POOLED deployed Platt map: p_cal = sigma(a * logit(p) + b).

    This is the backend's user-facing quantity (3) in MLB's three-probability
    contract: the single global map fitted on ALL OOF pairs and published in
    ``calibration_<date>.json``. It is deliberately NOT the per-fold
    prequential column, which the contract marks "honest for scoring/metrics;
    NEVER display". Returns NaN where the params are unusable so a caller can
    fall back rather than plot a fabricated curve.
    """
    if not params:
        return np.full(len(raw), np.nan)
    try:
        a = float(params["a"])
        b = float(params["b"])
    except (KeyError, TypeError, ValueError):
        return np.full(len(raw), np.nan)
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.full(len(raw), np.nan)
    p = np.clip(np.asarray(raw, dtype=float), 1e-7, 1.0 - 1e-7)
    return 1.0 / (1.0 + np.exp(-(a * np.log(p / (1.0 - p)) + b)))


def favored_deployed_calibration_pts(hist_curve: Optional[pd.DataFrame],
                                     params: Optional[dict]) -> pd.DataFrame:
    """The PUBLISHED calibration series: pooled deployed map over the OOF rows.

    Same favored-side convention and the SAME raw 1% bins as
    ``favored_calibration_pts`` (so the green and blue curves stay aligned
    one-to-one), but the y value is the deployed probability a bettor would
    actually be quoted: the pooled map from the calibration artifact applied
    to each OOF game's raw prediction.

    Unlike the observed rate, this is a deterministic monotone function of the
    raw probability, so it carries no sampling noise and is NOT low-n
    suppressed -- a 1-game bin still has a well-defined published probability.
    """
    if hist_curve is None or hist_curve.empty \
            or "home_win_prob_model" not in hist_curve.columns:
        return pd.DataFrame(columns=["prob", "cal_mean", "n"])
    raw = pd.to_numeric(hist_curve["home_win_prob_model"], errors="coerce")
    ok = raw.notna()
    if not ok.any():
        return pd.DataFrame(columns=["prob", "cal_mean", "n"])
    cal = deployed_probability(raw[ok].to_numpy(), params)
    ok2 = np.isfinite(cal)
    raw_favored = np.maximum(raw[ok].to_numpy(), 1.0 - raw[ok].to_numpy())
    cal_favored = np.where(raw[ok].to_numpy() >= 0.5, cal, 1.0 - cal)
    bins = (raw_favored[ok2] / FAVORED_BIN).round() * FAVORED_BIN
    if len(bins) == 0:
        return pd.DataFrame(columns=["prob", "cal_mean", "n"])
    return (pd.DataFrame({"prob": bins, "cal_mean": cal_favored[ok2]})
            .groupby("prob")
            .agg(cal_mean=("cal_mean", "mean"), n=("cal_mean", "size"))
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
    count_scale: Optional[alt.Scale] = None,
    bar_x_field: Optional[str] = None,
    bar_x2_field: Optional[str] = None,
    x_axis: Optional[alt.Axis] = None,
) -> dict:
    """Shared layered calibration-curve builder — the moneyline 'Calibration
    Curve' grammar, reused VERBATIM by the Game Total Lines diagnostics tab
    so both render as the SAME chart type through the same full-width path
    (width='container', resolve_scale x shared / y independent, one title per
    axis).

    ``pts`` is the per-bin bars frame (columns: ``x_field``, ``n_field``,
    optionally ``low_n_field``) driving the LEFT 'Games' count axis;
    ``series`` is a list of curve-layer specs — each {"data": DataFrame (x +
    y fields), "y_field": %-scale field, "axis": alt.Axis or None (EXACTLY ONE
    series owns the right axis; every other series passes axis=None),
    "y_scale" (pin the shared % domain), "color": str or alt.Color (alt.Color
    shares one legend across layers), "dash", "point_size", "stroke_width",
    "tooltips"} — on the shared RIGHT '%' axis (0-100, independent of the count
    axis). The gray dashed perfect-calibration diagonal and the optional amber
    pooled marker are scale-bound to the same axes.

    Because the composite view resolves y as ``independent``, ANY layer that
    does not pin its own ``y`` domain is auto-scaled to that layer's data
    extent — the single most damaging failure mode here, since a count layer
    and a rate layer then stop sharing a visual scale. ``count_scale`` pins the
    count domain for BOTH bar layers (pass it and the low-n bars stay on the
    same 'Games' scale as the main bars instead of stretching to fill the
    panel). For the same reason, every %-scale layer must receive the same
    explicit ``y_scale`` and every non-owning layer ``axis=None``; see
    ``chart_favored_calibration``.

    ``bar_x_field``/``bar_x2_field`` draw the bars as RANGE bars spanning one
    bin (``x`` = bin lower edge, ``x2`` = bin upper edge) instead of a point on
    a continuous x: a quantitative x gives ``mark_bar`` its ~5px default
    width, which renders narrow bins as thin spikes at any container width,
    while an x/x2 range is resolution-independent. ``x_axis`` rides the bars
    layer — the single owner of the shared x-axis.

    Returns {'chart': the layered spec, 'bars': the bars frame, 'n_total':
    sum of bar heights}. Empty/low-n bins are simply absent (no fabricated
    points) and render without error.
    """
    if pts is None or len(pts) == 0:
        # Keep the low-n column when rebuilding an empty frame: the layer
        # builder below indexes it unconditionally, so dropping it here made
        # an empty/low-n frame raise KeyError instead of rendering nothing.
        _cols = [x_field, n_field]
        if low_n_field and low_n_field not in _cols:
            _cols.append(low_n_field)
        pts = pd.DataFrame(columns=_cols)
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

    # The count domain is shared by BOTH bar layers: under independent y
    # resolution an unpinned layer auto-scales to its own extent, so the low-n
    # bars (a handful of games) drew as full-height spikes. Absent
    # ``count_scale`` keeps the previous auto-scaled behavior.
    count_sc = count_scale if count_scale is not None else alt.Undefined

    def _bar_x(title=alt.Undefined, own_axis: bool = True) -> dict:
        """Bars' x encoding — a bin-wide RANGE when bin edges are supplied.

        The MAIN bars layer is the single owner of the shared x-axis, so
        ``x_axis`` is attached there and nowhere else (``own_axis=False`` for
        the low-n overlay, which must not compete for it).
        """
        kw = {"scale": x_scale}
        if title is not alt.Undefined:
            kw["title"] = title
        if x_axis is not None and own_axis:
            kw["axis"] = x_axis
        if bar_x_field and bar_x2_field:
            return {"x": alt.X(f"{bar_x_field}:Q", **kw),
                    "x2": alt.X2(f"{bar_x2_field}:Q")}
        return {"x": alt.X(f"{x_field}:Q", **kw)}

    # Count bars — LEFT 'Games' axis (the single owner of that title).
    mark_kw = {"color": bar_color}
    if bar_opacity is not None:
        mark_kw["opacity"] = bar_opacity
    bars = alt.Chart(pts).mark_bar(**mark_kw).encode(
        **_bar_x(title=x_title),
        y=_y(n_field, axis=alt.Axis(title="Games", grid=True),
             scale=count_sc),
        tooltip=bar_tooltips)
    layers = [bars]

    # low-n bars render gray (n < LOW_N) on the SAME 'Games' scale as the
    # main bars, and carry no axis at all: the title belongs to the main bars
    # layer only (single title per axis), and an axis-less layer also emits no
    # tick labels, so no second left tick row can overlap it.
    if low_n_field is not None:
        low = pts[pts[low_n_field].fillna(False).astype(bool)]
        if not low.empty:
            low_bars = alt.Chart(low).mark_bar(
                color=low_n_color, opacity=low_n_opacity).encode(
                **_bar_x(own_axis=False),
                y=_y(n_field, axis=None, scale=count_sc),
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

    LOW-N SUPPRESSION (shared with the market-diagnostics / Game Total Lines
    charts, same imported ``LOW_N``): a bin with n < LOW_N renders as a GRAY
    bar — the volume context stays visible at true relative height, and on
    hover — and contributes NO blue point. Without it a single decided game at
    the high-confidence tail is plotted as if it were a rate, which drags the
    curve to 0% or 100% on noise alone. The green deployed-map curve is a
    deterministic function of the raw probability, so it is never suppressed.

    SCALE DISCIPLINE (the y axis is resolved 'independent', so every layer
    that does not pin its own domain silently auto-scales to its own extent).
    That is what made the low-n bars render as full-height spikes — and what
    let the green layer grow a second, overlapping right axis. Both bar layers
    therefore share ONE pinned count scale, and the diagonal, the blue
    observed curve and the green deployed curve share ONE pinned [0, 100]
    scale with exactly one owning axis.

    Delegates to the shared ``chart_calibration_curve`` builder — the SAME
    chart type the Game Total Lines diagnostics tab renders.

    Returns {'chart': the layered spec, 'bars': the per-bin counts frame (``n``
    per ``prob``), 'n_total': sum of bar heights}. Empty/low-n bins are simply
    absent (no fabricated points) and render without error.
    """
    if pts is None or len(pts) == 0:
        pts = pd.DataFrame(columns=["prob", "win_rate", "n", "low_n"])
    pts = pts.copy()
    if "low_n" not in pts.columns:
        # Callers that predate the flag (or a fixture frame) get the shared
        # rule applied here so the guard can never be bypassed.
        _n = (pd.to_numeric(pts["n"], errors="coerce") if "n" in pts.columns
              else pd.Series(np.nan, index=pts.index))
        pts["low_n"] = (_n < LOW_N).fillna(False) & (_n > 0)
    pts["win_rate_pct"] = pts["win_rate"] * 100.0
    x_dom = alt.Scale(domain=[0.45, 1.0])
    y_dom = alt.Scale(domain=[0, 100.0])

    # Range bars: the bins are FAVORED_BIN wide, so each bar spans 90% of its
    # own bin (the 10% remainder keeps adjacent bins legible). Encoding the
    # bin edges as x/x2 also makes the bar resolution-independent: a
    # quantitative x alone gives mark_bar its ~5px default width, which
    # rendered 1% bins as thin spikes that read as a comb rather than a
    # histogram. Clipped so an edge bin never leaves the probability range.
    _half = FAVORED_BIN * 0.45
    pts["bin_lo"] = (pts["prob"] - _half).clip(lower=0.0)
    pts["bin_hi"] = (pts["prob"] + _half).clip(upper=1.0)

    # ONE count scale for BOTH bar layers, pinned from the full per-bin frame
    # (so the main bars and the low-n bars cannot drift). The composite view
    # resolves y as 'independent', so an unpinned layer auto-scales to its own
    # extent: the low-n gray bars (n as low as 1) were drawn against their own
    # max and rendered as full-height spikes stabbing across the rate curve.
    #
    # The top is rounded UP to a whole tick step (~5 ticks) and the scale is
    # NOT left to Vega's 'nice' — that widens BOTH ends, which put a -100
    # 'Games' tick under a count axis that can never be negative. max() over
    # ALL bins (not just the plotted ones) keeps the domain valid for a frame
    # whose plotted rows are few; the 1.0 floor covers empty/thin frames.
    _counts = (pd.to_numeric(pts["n"], errors="coerce")
               if "n" in pts.columns else pd.Series(dtype="float64"))
    _n_max = float(_counts.max()) if len(_counts) else float("nan")
    if not (np.isfinite(_n_max) and _n_max > 0):
        _n_max = 1.0
    _step = max(1.0, 10.0 ** math.floor(math.log10(_n_max / 5.0)))
    count_dom = alt.Scale(
        domain=[0.0, _step * math.ceil(_n_max / _step)], nice=False)

    curve_pts = pts[~pts["low_n"].fillna(False).astype(bool)]
    series = [{
        "data": curve_pts,
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
            # Same pinned [0, 100] scale as the blue observed curve and the
            # diagonal, and NO axis of its own. Left unpinned, this layer got
            # its own auto scale (Vega's default zero=true happened to stretch
            # it to 0-100, so the line sat right by luck) AND emitted a SECOND
            # right axis whose tick labels overlapped the blue axis's. Pinning
            # the scale makes the published line's position a property of the
            # spec instead of an accident of the data's extent; the blue series
            # stays the single owner of the right axis.
            "axis": None,
            "y_scale": y_dom,
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
        bar_color=BLUE, bar_opacity=0.28,
        low_n_field="low_n", low_n_opacity=0.40,
        count_scale=count_dom,
        bar_x_field="bin_lo", bar_x2_field="bin_hi",
        x_axis=alt.Axis(format=".0%", tickCount=7, grid=True),
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
