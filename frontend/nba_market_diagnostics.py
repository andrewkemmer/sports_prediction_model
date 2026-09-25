"""NBA totals/point-spread diagnostics over run-engine artifacts."""
from __future__ import annotations

import io
import math
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import utils

TOTAL_GRID = list(range(180, 281))
SPREAD_GRID = list(range(-20, 21))
HALF_STOP_LINES = [-0.5, 0.5]
DIAG_TABS = ("Distribution", "Relativized", "Pooled Lines", "Game Total Lines", "Spread Lines")
OFFSET_EDGES = (-10.0, -5.0, 0.0, 5.0, 10.0)


def _label(value: float) -> str:
    text = str(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace("-", "m").replace(".", "_")


def _col(base: str, value: float) -> str:
    return f"{base}_{_label(value)}"


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


def _num(value) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _grid_p_over(frame: pd.DataFrame, line: float) -> np.ndarray:
    values = []
    for _, row in frame.iterrows():
        p = _num(row.get(_col("p_over", line)))
        if p is None:
            values.append(np.nan)
        else:
            values.append(p)
    return np.asarray(values, dtype=float)


def _line_value(row, base: str, line: float):
    return _num(row.get(_col(base, line)))


def _pairs(decided: pd.DataFrame, line: float, kind: str) -> dict[str, Any]:
    rows = []
    for _, row in decided.iterrows():
        actual = float(row.total) if kind == "total" else float(row.margin)
        expected = _num(row.get("fair_total" if kind == "total" else "fair_spread"))
        p = _line_value(row, "p_over" if kind == "total" else "p_home_cover", line)
        if p is None or expected is None:
            continue
        if kind == "total":
            target = 1.0 if actual > line else (0.5 if actual == line else 0.0)
        else:
            target = 1.0 if actual > line else (0.5 if actual == line else 0.0)
        rows.append({"predicted": p, "actual": target, "line": line,
                     "offset": float(line) - expected, "date": row.get("gameday")})
    return {"pairs": rows, "n_pairs": len(rows), "warning": None if rows else "No valid priced pairs."}


def total_distribution(decided: pd.DataFrame, kmax: int = 300) -> dict[str, Any]:
    if decided is None or not len(decided):
        return {"warning": "No decided NBA games available for distribution diagnostics.", "n_games": 0, "callouts": {}}
    totals = pd.to_numeric(decided.total, errors="coerce").dropna().astype(int)
    observed = totals[(totals >= 0) & (totals <= kmax)].value_counts().sort_index()
    return {"warning": None, "n_games": int(len(totals)),
            "observed": observed, "callouts": {"P(total<=1)": {"observed": float((totals <= 1).mean())},
                                                 "P(total>=250)": {"observed": float((totals >= 250).mean())}}}


def relativized_pairs(decided: pd.DataFrame, lines=None) -> dict[str, Any]:
    lines = lines or (205, 210, 215, 220, 225, 230, 235, 240)
    pairs = []
    for line in lines:
        pairs.extend(_pairs(decided, line, "total")["pairs"])
    return {"pairs": pairs, "n_pairs": len(pairs), "warning": None if pairs else "No total calibration pairs available."}


def fixed_line_pairs(decided: pd.DataFrame, lines=(220, 225, 230)) -> dict[str, Any]:
    pairs = []
    for line in lines:
        pairs.extend(_pairs(decided, line, "total")["pairs"])
    return {"pairs": pairs, "n_pairs": len(pairs), "warning": None if pairs else "No fixed-line pairs available."}


def _calibration_curve(pair_dict: dict[str, Any], title: str = "Calibration") -> dict[str, Any]:
    pairs = pair_dict.get("pairs", [])
    if not pairs:
        return {"warning": pair_dict.get("warning", "No calibration pairs."), "bins": [], "n_pairs": 0}
    frame = pd.DataFrame(pairs)
    frame = frame[np.isfinite(frame.predicted) & np.isfinite(frame.actual)]
    if frame.empty:
        return {"warning": "No finite calibration pairs.", "bins": [], "n_pairs": 0}
    frame["bin"] = np.clip((frame.predicted * 10).astype(int), 0, 9)
    bins = []
    for key, group in frame.groupby("bin", sort=True):
        bins.append({"mean_pred": float(group.predicted.mean()),
                     "mean_actual": float(group.actual.mean()), "n": int(len(group))})
    return {"warning": None, "bins": bins, "n_pairs": int(len(frame)),
            "n_dropped_bins": 0, "title": title}


def calibration_curve(pair_dict: dict[str, Any], title: str = "Calibration") -> dict[str, Any]:
    return _calibration_curve(pair_dict, title)


def game_total_calibration(decided: pd.DataFrame, line: float | None = None) -> dict[str, Any]:
    if line is None:
        line = _num((decided.get("fair_total") if decided is not None else None).median()) if decided is not None and len(decided) else None
    return _calibration_curve(_pairs(decided, line, "total") if line is not None else {"pairs": []}, "Game total calibration")


def run_line_calibration(decided: pd.DataFrame, magnitude: float = 2.0) -> dict[str, Any]:
    threshold = float(magnitude)
    return _calibration_curve(_pairs(decided, threshold, "spread"), "Point spread calibration")


def _card_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], np.clip(p[ok], 1e-7, 1 - 1e-7)
    if not len(y):
        return {"n": 0, "brier": np.nan, "logloss": np.nan, "ece": np.nan}
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(float((bins == b).mean() * abs(p[bins == b].mean() - y[bins == b].mean()))
              for b in range(10) if (bins == b).any())
    return {"n": int(len(y)), "brier": float(np.mean((p - y) ** 2)),
            "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "ece": float(ece)}


def winner_cards(decided: pd.DataFrame) -> dict[str, Any]:
    """Return three distinct model-card metrics.

    ``total`` is a totals card: its target is ``total > fair_total`` and its
    probability is ``p_over_fair``.  Reusing the home moneyline probability
    there silently scored the wrong event.  Home/away remain the derived
    distribution moneyline, including their coherent three-way tie mass.
    """
    empty = {"n": 0, "brier": np.nan, "logloss": np.nan, "ece": np.nan}
    if decided is None or not len(decided):
        return {"total": dict(empty), "home": dict(empty), "away": dict(empty)}

    margin = pd.to_numeric(decided.get("margin"), errors="coerce")
    home_p = pd.to_numeric(decided.get("p_home_win_derived"), errors="coerce")
    home_ok = margin.notna() & home_p.notna()
    home_y = (margin[home_ok] > 0).astype(float)
    home_result = _card_metrics(home_y, home_p[home_ok])
    away_result = _card_metrics(1.0 - home_y, 1.0 - home_p[home_ok])

    total = pd.to_numeric(decided.get("total"), errors="coerce")
    fair = pd.to_numeric(decided.get("fair_total"), errors="coerce")
    over_fair = pd.to_numeric(decided.get("p_over_fair"), errors="coerce")
    # Be tolerant of hand-built/older frames that persisted the fair-line
    # probability only in the grid column.
    if over_fair.isna().any():
        inferred = []
        for _, row in decided.iterrows():
            value = _num(row.get("p_over_fair"))
            line = _num(row.get("fair_total"))
            if value is None and line is not None:
                value = _num(row.get(_col("p_over", line)))
            inferred.append(value)
        over_fair = pd.Series(inferred, index=decided.index, dtype=float)
    total_ok = total.notna() & fair.notna() & over_fair.notna()
    total_y = (total[total_ok] > fair[total_ok]).astype(float)
    total_result = _card_metrics(total_y, over_fair[total_ok])
    total_result["event"] = "total_over_fair_total"
    home_result["event"] = "home_win"
    away_result["event"] = "away_win"
    return {"total": total_result, "home": home_result, "away": away_result}


def totals_monitor_stats(decided: pd.DataFrame, min_pct: float = 0.0) -> dict[str, Any]:
    rows = []
    for line in (220, 225, 230):
        pairs = _pairs(decided, line, "total")["pairs"]
        if pairs:
            frame = pd.DataFrame(pairs)
            rows.append({"line": line, "n": len(frame), "brier": float(np.mean((frame.predicted - frame.actual) ** 2))})
    return {"rows": rows, "warning": None if rows else "No canonical total metrics available."}


def runline_monitor_stats(decided: pd.DataFrame, min_pct: float = 0.0) -> dict[str, Any]:
    rows = []
    for line in (2, 5):
        pairs = _pairs(decided, line, "spread")["pairs"]
        if pairs:
            frame = pd.DataFrame(pairs)
            rows.append({"line": line, "n": len(frame), "brier": float(np.mean((frame.predicted - frame.actual) ** 2))})
    return {"rows": rows, "warning": None if rows else "No canonical spread metrics available."}


def load_run_engine_csv(ds: str, prefix: str) -> pd.DataFrame | None:
    """Load an NBA-prefixed run-engine drift/coverage CSV."""
    filename = f"{prefix}_{str(ds).replace('-', '')}.csv"
    raw, _ = utils._fetch_bytes(filename, **utils.get_source_config(), sport="nba")
    if raw is None:
        return None
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception:
        return None


def render_run_engine_drift(frame: pd.DataFrame | None) -> None:
    st.markdown("#### Run-Engine Feature Drift")
    if frame is None or frame.empty:
        st.info("No run-engine feature-drift artifact found for this date.")
        return
    st.dataframe(frame, use_container_width=True)


def render_run_engine_coverage(frame: pd.DataFrame | None) -> None:
    st.markdown("#### Run-Engine Feature Coverage")
    if frame is None or frame.empty:
        st.info("No run-engine feature-coverage artifact found for this date.")
        return
    st.dataframe(frame, use_container_width=True)


def fold_slate_history(monitors: list[dict]) -> dict[str, list[dict]]:
    folded: dict[str, list[dict]] = {}
    for monitor in monitors or []:
        for row in monitor.get("slate_history", []) or []:
            key = str(row.get("game_id", "unknown"))
            folded.setdefault(key, []).append(row)
    return {key: rows[-10:] for key, rows in folded.items()}


def chart_calibration(curve: dict[str, Any], title: str = "Calibration") -> Any:
    bins = curve.get("bins", []) if isinstance(curve, dict) else []
    if not bins:
        return None
    frame = pd.DataFrame(bins)
    return alt.Chart(frame).mark_line().encode(
        x="mean_pred:Q", y="mean_actual:Q",
        tooltip=["mean_pred:Q", "mean_actual:Q", "n:Q"]).properties(title=title)


def chart_distribution(distribution: dict[str, Any]) -> Any:
    observed = distribution.get("observed") if isinstance(distribution, dict) else None
    if observed is None or not len(observed):
        return None
    frame = observed.rename_axis("total").reset_index(name="games")
    return alt.Chart(frame).mark_bar().encode(x="total:O", y="games:Q").properties(title="NBA totals distribution")
