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
TOTAL_GRID_LO = TOTAL_GRID[0]
TOTAL_GRID_HI = TOTAL_GRID[-1]
RUN_COVER_COL = "p_home_cover_1_5"
# Per-line run-line grid (mirror of the backend RUN_LINE_GRID_FULL): the
# half-lines (−1.5 … −4.5 by legacy grid) PLUS whole-number alternates
# (−1 … −4). Column names come from rl_cols() — built from the RAW margin,
# so whole 1 and half 1.5 never collide (the totals-grid lesson).
RUN_LINE_GRID_FULL = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def rl_cols(margin: float) -> tuple[str, str, str]:
    """p_rl column names for a run-line margin: (home, push, away).
    MUST match the backend's run_engine.rl_col naming exactly (the
    artifact is canonical): f'{1.0:.1f}' -> '1_0' vs f'{1.5:.1f}' ->
    '1_5', so whole 1 and half 1.5 never collide — p_rl_1_0_home and
    p_rl_1_5_home are distinct columns."""
    key = f"{margin:.1f}".replace(".", "_")
    return (f"p_rl_{key}_home", f"p_rl_{key}_push", f"p_rl_{key}_away")


def rl_legacy_col(margin: float) -> str:
    """Legacy p_home_cover_<margin> column (half-lines only, e.g. 1.5)."""
    return f"p_home_cover_{str(margin).replace('.', '_')}"
# Own-line ('All') predicted-P(over) buckets — 1% increments (40–41 … 59–60,
# 60+). The All view prices each game at its OWN fair line, where re-scaled
# 2-way P(over) ≈ 0.5 BY CONSTRUCTION (the fair line IS the 50/50 anchor), so
# the support is a tight band around 50% (0.47–0.53 on the shipped artifact).
# Predicted is the SAME re-scaled P(over) the Today's Games card displays at
# its default line — never the picked-side max (those agree only on
# over-favored games). Empty buckets render as 0 — a data property, not an
# error.
OWN_LINE_EDGES = list(range(40, 61)) + [101]
OWN_LINE_LABELS = [f"{lo}-{lo + 1}" for lo in range(40, 60)] + ["60+"]

# Low-sample threshold for the totals calibration chart: buckets with
# n < LOW_N games are shaded gray, their win-rate/observed points dropped
# from the chart, and flagged in the table + footnote — never readable as
# strong calibration evidence.
LOW_N = 30


def _gtl_line_points(table: dict) -> pd.DataFrame:
    """Win-rate + observed line points for the moneyline-style game-total
    calibration chart — one row per (bin, series) over NON-low-n populated
    bins only (low-n points are dropped: n < LOW_N is not reliable
    calibration evidence). ``pct`` is on the 0-100 no-push 2-way basis."""
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
    return pd.DataFrame(rows)


def chart_game_total_curve(table: dict, title: str,
                           obs_label: str = "Observed % (2-way, no push)") -> dict:
    """Moneyline-style single chart for the 'Game Total Lines' diagnostics tab
    — count bars + calibration curves + dashed diagonal in ONE chart
    ('chart'), plus the pooled table ('table'). No separate scatter.

    Grammar mirrors the moneyline Calibration Curve page: a continuous
    predicted-P(over) x-axis (bin centers on a 0-1 probability scale — the
    same 5-pt buckets, 0.025…0.975, for a fixed line; the All branch keeps
    its 1-pt own-line buckets ~0.50–0.61), count bars on a LEFT 'Games'
    axis, the observed + picked-side win-rate lines on a shared RIGHT '%'
    axis (0-100), a gray dashed perfect-calibration diagonal, an amber
    pooled marker at the pooled calibration point, and hover per bin (games,
    mean predicted, observed, win rate). Each axis title lives on EXACTLY
    ONE layer (the bars own 'Games' left; the lines own the right '%'
    title; diagonal + pooled marker render axis-less) so labels never
    overlap — the same single-title-per-axis fix as the moneyline curve.

    Win rate = the moneyline-card convention: pick over if P(over) > 50%
    else under; W/(W+L) per bin on the no-push 2-way basis (a 'V' around
    50%; bins below 50% give 1 − observed). Empty bins keep count 0 / None
    stats: bars render zero-height, lines skip them. low_n bins (n < LOW_N)
    render as gray bars and their win-rate/observed points are DROPPED from
    the curves (never readable as reliable calibration). Pooled aggregates
    ride the Total table row, the caption, and the amber pooled marker.
    """
    tdf = pd.DataFrame(table["bins"])
    if tdf.empty:
        return {"chart": alt.Chart(pd.DataFrame()).mark_bar(), "table": tdf}
    chart_df = tdf.copy()
    chart_df["observed_pct"] = chart_df["observed"] * 100.0
    chart_df["win_rate_pct"] = chart_df["win_rate"] * 100.0
    x_dom = alt.Scale(domain=[0.0, 1.0], nice=False)
    y_pct_dom = alt.Scale(domain=[0.0, 100.0])

    # Count bars — LEFT 'Games' axis (the single owner of that title). low_n
    # bins render gray (n < LOW_N suppressed exactly as before).
    bar_tip = [
        alt.Tooltip("bin_center:Q", title="Predicted P(over)", format=".3f"),
        alt.Tooltip("count:Q", title="Games"),
        alt.Tooltip("mean_pred:Q", title="Mean predicted", format=".3f"),
        alt.Tooltip("observed:Q", title="Observed", format=".3f"),
        alt.Tooltip("win_rate:Q", title="Win rate", format=".3f"),
    ]
    bars = alt.Chart(chart_df).mark_bar(color="#3B82F6").encode(
        x=alt.X("bin_center:Q", title="Predicted P(over)", scale=x_dom),
        y=alt.Y("count:Q", axis=alt.Axis(title="Games", grid=True)),
        tooltip=bar_tip)
    bar_layer = bars
    low_df = chart_df[chart_df["low_n"]]
    if not low_df.empty:
        # low-n bars are gray and axis-title-less — "Games" (the left
        # axis title) belongs to the main bars layer ONLY, so independent
        # y-scales never draw it twice (single title per axis fix).
        low_bars = alt.Chart(low_df).mark_bar(
            color="#94A3B8", opacity=0.45).encode(
            x=alt.X("bin_center:Q", scale=x_dom),
            y=alt.Y("count:Q", axis=alt.Axis(title=None)),
            tooltip=bar_tip)
        bar_layer = bars + low_bars

    # Observed + win-rate curves over non-low-n populated bins — RIGHT '%'
    # axis (0-100), the ONLY owner of the obs_label title (single title per
    # axis: no duplicate/overlapping labels). low-n points dropped.
    stack = _gtl_line_points(table)
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
                            scale=alt.Scale(domain=["Observed", "Win rate"],
                                            range=["#22C55E", "#8B5CF6"]),
                            title="Series"),
            tooltip=[
                alt.Tooltip("bin_center:Q", title="Predicted P(over)", format=".3f"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("pct:Q", title=obs_label, format=".1f"),
                alt.Tooltip("count:Q", title="Games")])

    # Dashed perfect-calibration diagonal (y = x on the 0-100 % scale),
    # axis-less so it never emits a competing axis title.
    diag_df = pd.DataFrame({"bin_center": [0.0, 1.0], "pct": [0.0, 100.0]})
    diag = alt.Chart(diag_df).mark_line(
        color="#64748B", strokeDash=[5, 5], strokeWidth=1.5).encode(
        x=alt.X("bin_center:Q", scale=x_dom),
        y=alt.Y("pct:Q", axis=None, scale=y_pct_dom))

    layers = [bar_layer, line_chart, diag]
    # Amber pooled marker on the SAME chart (not a separate scatter) so the
    # chart and the Total table row agree about the pooled calibration point.
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
    # Pooled (Total) table row — the pooled-aggregates summary, share 100%.
    total_row = pd.DataFrame([{
        "bin": "Total", "bin_center": None, "count": int(tdf["count"].sum()),
        "mean_pred": pooled_pred, "observed": pooled_obs,
        "win_rate": table.get("pooled_winrate"),
        "ece": table.get("pooled_ece"), "brier": table.get("pooled_brier"),
        "low_n": False, "share_pct": 100.0,
    }])
    table_df = pd.concat([tdf, total_row], ignore_index=True)
    return {"chart": chart, "table": table_df}


def round_to_half(x: float) -> float:
    """Round to the nearest 0.5, ties away from zero (round half up).

    Totals lines are quoted in half-increments, so a projected total of
    9.3 prices at 9.5 (9.3 * 2 = 18.6 → 19 → 9.5). Uses floor/ceil on
    (2x + 0.5) instead of Python's banker's-rounding round(), and avoids
    float drift on exact halves (8.25 → 16.5 → 17 → 8.5).
    """
    if x < 0:
        return math.ceil(x * 2 - 0.5) / 2.0
    return math.floor(x * 2 + 0.5) / 2.0


def clamp_to_grid(line: float) -> tuple[float, bool]:
    """Clamp a total line into the shipped grid; returns (line, clamped).

    The grid ships half-steps 6.5 … 12.5, so any rounded line inside the
    range is already a real grid line; only lines outside clamp to the
    nearest edge (the caller notes it — never fabricated).
    """
    if line < TOTAL_GRID_LO:
        return TOTAL_GRID_LO, True
    if line > TOTAL_GRID_HI:
        return TOTAL_GRID_HI, True
    return line, False


def grid_over_under_cols(line: float) -> tuple[str, str]:
    """p_over / p_under column names for a (grid) total line, e.g. 9.5 →
    p_over_9_5 / p_under_9_5; 10.0 → p_over_10_0 / p_under_10_0."""
    key = str(line).replace(".", "_")
    return f"p_over_{key}", f"p_under_{key}"


def grid_push_col(line: float) -> str:
    """Explicit p_push column name for a grid total line (post-65b44ec
    artifacts ship P(total == line) directly), e.g. 9.0 → p_push_9_0."""
    return f"p_push_{str(line).replace('.', '_')}"
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


def fair_total_lines(decided: pd.DataFrame) -> np.ndarray:
    """Per-game FAIR total line — the grid argmin of
    |re-scaled P(over) − 0.5| over the shipped grid (6.5 … 12.5 by 0.5),
    where re-scaled P(over) = p_over / (p_over + p_under) conditions out
    the push band (whole-number lines). NaN where a game cannot be priced
    (missing/NaN grid columns or zero denom) — NEVER fabricated.

    Equates to the re-normalized median (where P(over | no push) = 50%)
    snapped to the nearest 0.5 line, i.e. the same "cut at the 50/50 point"
    logic as the run-line cut. Ties in |Δ| pick the LOWER line (strict
    `<` keeps the first ascending match). If the argmin sits on a grid
    boundary it is taken as-is (documented) — a line outside the grid is
    never fabricated.
    """
    n = len(decided)
    best_line = np.full(n, np.nan)
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
        take = valid & (delta < best_delta - 1e-12)  # ties keep lower line
        best_delta[take] = delta[take]
        best_line[take] = line
    return best_line


def fair_total_line_row(row: Any) -> Optional[float]:
    """FAIR total line for a single artifact/slate row (dict or Series).

    Returns None when no grid Over/Under column pair is present+valid with
    a positive sum (the caller falls back to the round-to-half projection
    rather than fabricating a line). Uses the same argmin rule as
    fair_total_lines; ties pick the lower line; a grid-boundary argmin is
    taken verbatim.
    """
    best_line, best_delta = None, None
    for line in TOTAL_GRID:
        over_col, _ = grid_over_under_cols(line)
        _, under_col = grid_over_under_cols(line)
        po = _num(row, over_col)
        pu = _num(row, under_col)
        if po is None or pu is None:
            continue
        denom = po + pu
        delta = abs(po / denom - 0.5)
        if best_line is None or delta < best_delta - 1e-12:
            best_line, best_delta = line, delta
    return best_line


def _rounded_lines(decided: pd.DataFrame) -> np.ndarray:
    """Per-game OWN total line = FAIR line when the grid allows, else the
    round-half-up λ_home + λ_away projection clamped to the grid (legacy
    artifacts / unpriced rows). Shared by pairs, picks, and push
    detection — the fair line is the legal-bookmaker anchor, so pick and
    push logic operate on the same line the card quotes."""
    fair = fair_total_lines(decided)
    exp_h = decided["home_expected_runs"].to_numpy(float)
    exp_a = decided["away_expected_runs"].to_numpy(float)
    fallback = np.array([clamp_to_grid(round_to_half(h + a))[0]
                         for h, a in zip(exp_h, exp_a)])
    return np.where(np.isnan(fair), fallback, fair)


def push_stats(decided: pd.DataFrame) -> dict:
    """Whole-number-line PUSHES: total_runs == the game's rounded total.

    A push is neither a win nor a loss for either side, so win rates must
    exclude these games from BOTH numerator and denominator. Empirically the
    pushed games are UNDER-favored games landing exactly on the line: the
    rounded line sits at/above the expected total (line ≥ λ_home + λ_away),
    so p_under > p_over, and the old scoring counted the push as an UNDER
    win (total == line → not over). Excluding them is why the honest pooled
    win rate runs BELOW the inflated one (2026-08-24 artifact: 56.1% →
    54.1%, ≈2,420 wins/4,314 → ≈2,200 wins/4,066). Only whole-number lines
    can tie (totals are integers; a 9.5 line can't push). Returns counts +
    rate for captions — never fabricated.
    """
    empty = {"n_games": 0, "n_pushes": 0, "push_rate": 0.0}
    if not len(decided) or "total_runs" not in decided.columns:
        return empty
    if ({"home_expected_runs", "away_expected_runs"}
            .difference(decided.columns)):
        return empty
    n = len(decided)
    lines = _rounded_lines(decided)
    total = decided["total_runs"].to_numpy(float)
    n_pushes = int((total == lines).sum())
    return {"n_games": n, "n_pushes": n_pushes,
            "push_rate": round(n_pushes / n, 4) if n else 0.0}


def rounded_total_pairs(decided: pd.DataFrame) -> pd.DataFrame:
    """(p_over, outcome) pairs at each game's OWN rounded total line.

    Mirrors how the Relativized tab prices each game at its own expected
    total, but at the half-step line a bettor actually quotes: nearest 0.5
    of λ_home + λ_away (round half up), clamped to the shipped grid. One
    row per non-push game: PUSHES (total == line, whole-number lines only)
    are excluded from the win-rate pairs — neither wins nor losses (see
    push_stats for the count/rate). Rows whose grid column is missing are
    skipped loudly by the caller's empty state — never fabricated.
    """
    if not len(decided) or "total_runs" not in decided.columns:
        return pd.DataFrame(columns=["p", "y", "line"])
    if ({"home_expected_runs", "away_expected_runs"}
            .difference(decided.columns)):
        return pd.DataFrame(columns=["p", "y", "line"])
    total = decided["total_runs"].to_numpy(float)
    lines = _rounded_lines(decided)
    rows = []
    for i in range(len(decided)):
        line = lines[i]
        if total[i] == line:          # push — excluded from win rates
            continue
        over_col, _ = grid_over_under_cols(line)
        if over_col not in decided.columns:
            continue
        p = float(decided[over_col].iloc[i])
        if pd.isna(p):
            continue
        # Over at a total line means strictly more runs than line + 0.5.
        y = float(total[i] >= line + 0.5)
        rows.append({"p": p, "y": y, "line": line})
    if not rows:
        return pd.DataFrame(columns=["p", "y", "line"])
    return pd.DataFrame(rows)


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


def _bucket_calibration(pred: np.ndarray, event: np.ndarray,
                        edges: list[float],
                        labels: list[str]) -> tuple[list[dict], float, float]:
    """Bucket predicted probabilities + binary outcomes into ``edges``
    (percent units; last bucket open-ended). Returns (bins, pooled_pred,
    pooled_event). EMPTY bins are kept with count 0 / None stats — never
    dropped. ``event`` is the outcome the prediction refers to: the over
    event for a fixed line, the picked-side win for the own-line 'All'
    branch    (so observed-vs-predicted stays apples-to-apples). share_pct =
    % of total observations in the group = count_bin / count_total × 100
    (count_total = all observations passed in — the priced non-push games).

    Extends the moneyline-calibration-card convention to totals: per-bin
    ``win_rate`` = W/(W+L) for the pick rule 'over if P(over) > 50% else
    under' (a 'V' around 50% — bins below 50% give 1 − observed, bins
    above give observed); ``ece`` = |mean_pred − observed| (per-bin, 0–1);
    ``brier`` = mean((prediction − outcome)²) both on the same 2-way
    no-push event basis. ``low_n`` flags bins with n < LOW_N for the
    chart/table. Pooled aggregates returned: pooled_pred, pooled_observed,
    pooled_winrate, pooled_ece (count-weighted over bins),
    pooled_brier (mean over all pairs)."""
    pred_frac = np.clip(np.asarray(pred, float), 0.0, 1.0)
    ev = np.asarray(event, float)
    win = np.where(pred_frac > 0.5, ev, 1.0 - ev)  # over pick / under pick
    pct = pred_frac * 100.0
    n = len(pct)
    bins = []
    for b, lab in enumerate(labels):
        lo, hi = edges[b], edges[b + 1]
        m = (pct >= lo) & (pct < hi) if b < len(labels) - 1 else (pct >= lo)
        cnt = int(m.sum())
        mean_pred = float(pred_frac[m].mean()) if cnt else None
        observed = float(ev[m].mean()) if cnt else None
        bins.append({
            "bin": lab,
            "bin_center": round(float((lo + hi) / 200.0), 3),
            "count": cnt,
            "mean_pred": (round(mean_pred, 4) if cnt else None),
            "observed": (round(observed, 4) if cnt else None),
            "win_rate": (round(float(win[m].mean()), 4) if cnt else None),
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
    return (bins, round(float(pred_frac.mean()), 4),
            round(float(ev.mean()), 4),
            round(float(win.mean()), 4),
            round(pooled_ece, 4),
            round(float(((pred_frac - ev) ** 2).mean()), 4))


def game_total_calibration(decided: pd.DataFrame,
                           line: Optional[float] = None,
                           n_bins: int = 20) -> dict[str, Any]:
    """Calibration table for the 'Game Total Lines' diagnostics tab.

    line=None ('All') → every game priced at its OWN FAIR line (grid argmin
    of |re-scaled P(over) − 0.5|, ties → lower — fair_total_lines):
    predicted = re-scaled 2-way P(over) = p_over / (p_over + p_under) at the
    own line — the SAME value run_engine_card_bits shows as the card's O/U
    Over% at the default line (verified identical, pinned in tests) —
    bucketed at 1 pt (40–41 … 60+) because own-line P(over) hugs 50% by
    construction; observed = over rate (#over / (#over + #under)), pushes
    excluded from both sides.

    line given → ALL games priced at that ONE fixed line: predicted =
    re-scaled 2-way P(over) = p_over / (p_over + p_under); observed = over
    frequency on the same no-push basis (#over / (#over + #under)); 5-pt
    bins over [0, 1] (n_bins).

    Both branches share one code path: pushes (total == whole-number line)
    are excluded from the calibration population — neither wins nor losses
    — and reported as n_pushes / push_rate. share_pct = count_bin /
    count_total × 100 with count_total = priced non-push games. Strict over:
    total > line (total >= line + 0.5).
    """
    empty = {"line": line, "bins": [], "n_games": 0, "n_pushes": 0,
             "push_rate": 0.0, "pooled_pred": None, "pooled_observed": None,
             "pooled_winrate": None, "pooled_ece": None, "pooled_brier": None,
             "warning": "No decided games available for this view."}
    if not len(decided) or "total_runs" not in decided.columns:
        return empty
    if line is None and ({"home_expected_runs", "away_expected_runs"}
                         .difference(decided.columns)):
        # Only the 'All' (own-line) branch needs the expected runs to resolve
        # each game's fair line; a fixed line reads its own grid columns.
        empty["warning"] = "Missing expected-runs columns."
        return empty
    total = decided["total_runs"].to_numpy(float)
    n_all = len(decided)
    pred = np.full(n_all, np.nan)
    event = np.zeros(n_all)
    push = np.zeros(n_all, bool)
    priced = np.zeros(n_all, bool)
    if line is None:
        lines = _rounded_lines(decided)
        for i in range(n_all):
            l = lines[i]
            if np.isnan(l):
                continue
            over_col, under_col = grid_over_under_cols(l)
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
            rso = float(v) / denom
            pred[i] = rso          # re-scaled 2-way P(over) = card's Over%
            priced[i] = True
            if total[i] == l:
                push[i] = True                 # whole lines only
                continue
            event[i] = float(total[i] >= l + 0.5)   # over rate, no-push basis
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
        over = (total >= line + 0.5) & valid & ~push
        event = over.astype(float)
        edges = [round(5.0 * b, 2) for b in range(n_bins + 1)]  # 0, 5, …, 100
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
    bins, pooled_pred, pooled_obs, pooled_winrate, pooled_ece, pooled_brier = (
        _bucket_calibration(pred[ok], event[ok], edges, labels))
    return {"line": line, "bins": bins, "n_games": n, "n_pushes": n_pushes,
            "push_rate": round(n_pushes / n, 4) if n else 0.0,
            "pooled_pred": pooled_pred, "pooled_observed": pooled_obs,
            "pooled_winrate": pooled_winrate, "pooled_ece": pooled_ece,
            "pooled_brier": pooled_brier,
            "warning": None}


# ---------------------------------------------------------------------------
# Charts 5–6 — favored-side pick accuracy buckets
# ---------------------------------------------------------------------------
def pick_buckets(p_pick_prob: np.ndarray, hit: np.ndarray,
                 labels: Optional[list[str]] = None,
                 edges: Optional[list[float]] = None) -> dict[str, Any]:
    """Count + hit rate per confidence bucket on the FAVORED-side probability.
    Hit rate is NOT calibration — it is binary pick accuracy per bucket.

    ``edges``/``labels`` jointly define the buckets: bucket i covers
    [edges[i], edges[i+1]), except the last bucket which is ``>= edges[i]``
    (open-ended). Defaults to the generic 5% buckets (BUCKET_LABELS). Empty
    buckets are kept with count 0 and accuracy None — never dropped."""
    labels = BUCKET_LABELS if labels is None else labels
    edges = BUCKET_EDGES if edges is None else edges
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
        lo = edges[i]
        hi = edges[i + 1]
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
# Prediction-history row builders — game totals & run lines (read-only)
# ---------------------------------------------------------------------------
def _col_or(df: pd.DataFrame, name: str, default: Any = np.nan):
    """Column value when present, else a constant (fixtures may omit it)."""
    return df[name] if name in df.columns else default


def totals_history_frame(decided: pd.DataFrame) -> pd.DataFrame:
    """Per-game rows for the game-totals prediction-history table.

    Each row is priced at the game's OWN total line = the FAIR line — the
    grid argmin of |re-scaled P(over) − 0.5| (the legal 50/50 anchor) —
    the SAME line the diagnostics' totals-picks chart and the card use, so
    the three agree exactly. pick = Over if the 2-way RE-SCALED
    P(over|no push) = p_over/(p_over + p_under) >= 0.5 else Under (raw
    p_over under-states Over on whole-number lines via the push band);
    pick_prob is the favored side's re-scaled probability. winner = Over
    when total_runs > line,
    Under when total_runs < line, Push when total_runs == line
    (whole-number lines only — integer totals can never equal an X.5
    line). Push rows carry correct = NaN so win rates exclude them from
    BOTH numerator and denominator, matching the diagnostics push
    handling. Rows whose grid column is missing/NaN cannot be priced and
    are DROPPED — never fabricated.
    """
    empty = pd.DataFrame(columns=["game_pk", "game_date", "home_score",
                                  "away_score", "total_runs", "line",
                                  "pick", "pick_prob", "winner",
                                  "correct"])
    if not len(decided) or "total_runs" not in decided.columns:
        return empty
    if ({"home_expected_runs", "away_expected_runs"}
            .difference(decided.columns)):
        return empty
    total = decided["total_runs"].to_numpy(float)
    lines = _rounded_lines(decided)
    hs = _col_or(decided, "home_score")
    as_ = _col_or(decided, "away_score")
    gd = _col_or(decided, "game_date")
    pk = _col_or(decided, "game_pk")
    rows = []
    for i in range(len(decided)):
        line = lines[i]
        over_col, under_col = grid_over_under_cols(line)
        if over_col not in decided.columns or under_col not in decided.columns:
            continue
        po = decided[over_col].iloc[i]
        pu = decided[under_col].iloc[i]
        if pd.isna(po) or pd.isna(pu):
            continue
        po, pu = float(po), float(pu)
        denom = po + pu
        if denom <= 0:
            continue
        # 2-way RE-SCALED P(over|no push) — the same quantity the FAIR line
        # is defined on. The raw p_over column under-states Over on whole-number
        # lines (the push band dilutes it below 0.5 even at the 50/50 anchor),
        # which made the split collapse to ~all-Under; pick/prob use the re-
        # scaled value so the split is genuinely 50/50 and the probability is
        # calibrated (pushes folded out, consistent with the monitor card).
        p = po / denom
        pick = "Over" if p >= 0.5 else "Under"
        pick_prob = p if pick == "Over" else 1.0 - p
        t = total[i]
        if t == line:
            winner = "Push"
        else:
            winner = "Over" if t > line else "Under"
        correct = (float(pick == winner) if winner != "Push" else np.nan)
        rows.append({
            "game_pk": pk.iloc[i],
            "game_date": gd.iloc[i],
            "home_score": hs.iloc[i],
            "away_score": as_.iloc[i],
            "total_runs": t,
            "line": line,
            "pick": pick,
            "pick_prob": round(pick_prob, 6),
            "winner": winner,
            "correct": correct,
        })
    if not rows:
        return empty
    return pd.DataFrame(rows)


def runline_history_frame(decided: pd.DataFrame) -> pd.DataFrame:
    """Per-game rows for the run-line (−1.5/+1.5) prediction-history table.

    pick = home −1.5 when p_home_cover_1_5 >= 0.5 else away +1.5
    (pick_prob = the favored side's probability, complement of the shipped
    home-cover column). winner = HOME when home_score − away_score >= 2,
    AWAY otherwise — a 1-run home win is an away +1.5 win. Half-run lines
    never push, so every priced game has a definite winner (no push
    handling, noted in the caption). Rows missing the cover column or a
    NaN cover probability are DROPPED — never fabricated.
    """
    empty = pd.DataFrame(columns=["game_pk", "game_date", "home_score",
                                  "away_score", "pick", "pick_prob",
                                  "winner", "correct"])
    if not len(decided) or RUN_COVER_COL not in decided.columns:
        return empty
    if {"home_score", "away_score"}.difference(decided.columns):
        return empty
    p = decided[RUN_COVER_COL].to_numpy(float)
    margin = (decided["home_score"].to_numpy(float)
              - decided["away_score"].to_numpy(float))
    gd = _col_or(decided, "game_date")
    pk = _col_or(decided, "game_pk")
    rows = []
    for i in range(len(decided)):
        if pd.isna(p[i]):
            continue
        pick = "home" if p[i] >= 0.5 else "away"
        pick_prob = p[i] if pick == "home" else 1.0 - p[i]
        winner = "home" if margin[i] >= 2 else "away"
        rows.append({
            "game_pk": pk.iloc[i],
            "game_date": gd.iloc[i],
            "home_score": decided["home_score"].iloc[i],
            "away_score": decided["away_score"].iloc[i],
            "pick": pick,
            "pick_prob": round(float(pick_prob), 6),
            "winner": winner,
            "correct": float(pick == winner),
        })
    if not rows:
        return empty
    return pd.DataFrame(rows)


def filter_history_frame(frame: pd.DataFrame, start_date: Any,
                         end_date: Any) -> pd.DataFrame:
    """Rows with game_date within [start_date, end_date] (inclusive).

    Pure date filter over the history frames (start/end as date or
    datetime-like); the render layer sorts most-recent-first.
    """
    if not len(frame) or "game_date" not in frame.columns:
        return frame.copy()
    dts = pd.to_datetime(frame["game_date"], errors="coerce").dt.date
    keep = (dts >= start_date) & (dts <= end_date)
    return frame[keep].reset_index(drop=True)


def history_win_rate(frame: pd.DataFrame) -> dict:
    """Header stats: n_games (non-push denominator) + pooled correct rate.

    Push rows carry correct = NaN (totals whole-number-line pushes) and
    are excluded from BOTH numerator and denominator; run-line rows never
    push, so every row counts. Returns {"n_games", "win_rate",
    "n_pushes"} — win_rate is None when no decided (non-push) rows.
    """
    if not len(frame) or "correct" not in frame.columns:
        return {"n_games": 0, "win_rate": None, "n_pushes": 0}
    ok = frame["correct"].notna()
    n = int(ok.sum())
    wins = float(frame.loc[ok, "correct"].sum()) if n else 0.0
    n_pushes = 0
    if "winner" in frame.columns:
        n_pushes = int((frame["winner"] == "Push").sum())
    return {"n_games": n,
            "win_rate": (round(wins / n, 6) if n else None),
            "n_pushes": n_pushes}


# ---------------------------------------------------------------------------
# Run-line cut history + monitor calibration cards — unified 3-way push
# resolution. Everything reads the POST-FIX margin distribution: the
# corrected p_rl_*_home/push/away columns (tie-mass renormalization 2531462
# + home one-run adjustment fdd9187). −0.5 is derived from the corrected
# moneyline (P(favored wins) = P(margin > 0)); there is no p_rl_0_5 column.
# ---------------------------------------------------------------------------
# Favorite-side run-line margins for the cut search and the monitor LINE
# toggle. In MLB, line 0 ≡ −0.5 (integer margins, no ties — identical cover
# sets), so 0 never appears; map_run_line_zero guards any rounding that
# lands on 0.
RUN_GRID_CUT = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# Favorite-side line choices for the run-line monitor card (−0.5 … −4).
RUN_LINE_CHOICES = [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0]
# Cumulative percent-confidence thresholds for the totals calibration card
# — 1-point steps 50-55. At the fair-line own total the pick is always the
# side with P > 50%, so 40/45 were no-ops (every pick qualified) and 55+
# was empty; 1-point steps show the population fall-off vs confidence.
TOTALS_CONF_THRESHOLDS = [50, 51, 52, 53, 54, 55]


def map_run_line_zero(line: float) -> float:
    """Run-line 0 ≡ −0.5 in MLB: margins are integers and there are no
    ties, so margin > 0 and margin >= 1 select identical cover sets. The
    cut search never emits 0 (favored cover at 0.5 is P(favored win) >= 0.5
    by construction); this maps any rounding that lands on 0 to the −0.5
    line magnitude so a 0 line can never be priced/drawn."""
    return 0.5 if abs(line) < 1e-9 else line


def corrected_home_win(decided: pd.DataFrame) -> np.ndarray:
    """Corrected NB home-win probability from the POST-FIX margin
    distribution — never the legacy raw column.

    P'(home win) = P'(margin > 0) = P(margin >= 2) + P'(margin == 1 resolved)
                 = p_rl_1_0_home + p_rl_1_0_push
    (p_rl_1_0_home = P(margin >= 2); p_rl_1_0_push = the resolved +1 band
    on post-fix artifacts, which the home one-run structural adjustment
    fdd9187 moved so pooled P(win) = 0.5084 → 0.5323 vs actual 0.5320).
    Composing the pair is preferred over the shipped p_home_win_derived
    column so a stale artifact (raw moneyline, e.g. 0.4566) cannot feed
    legacy values. Falls back to p_home_win_derived, then p_rl_1_0_home,
    when the +1 band column is absent. NaN where unpricable."""
    n = len(decided)
    nan = np.full(n, np.nan)
    if not n:
        return nan
    if {"p_rl_1_0_home", "p_rl_1_0_push"}.issubset(decided.columns):
        a = decided["p_rl_1_0_home"].to_numpy(float)
        b = decided["p_rl_1_0_push"].to_numpy(float)
        h = np.where(np.isfinite(a) & np.isfinite(b), a + b, np.nan)
        return np.where((h >= 0.0) & (h <= 1.0), h, np.nan)
    if "p_home_win_derived" in decided.columns:
        v = decided["p_home_win_derived"].to_numpy(float)
        return np.where(np.isfinite(v), v, np.nan)
    if "p_rl_1_0_home" in decided.columns:
        v = decided["p_rl_1_0_home"].to_numpy(float)
        return np.where(np.isfinite(v), v, np.nan)
    return nan


def favored_cover_at(decided: pd.DataFrame, line: float,
                     home_win: Optional[np.ndarray] = None):
    """Per-game FAVORED-side cover probability at run-line margin
    ``line`` (magnitude in RUN_GRID_CUT), from the corrected p_rl columns.

    Favored side = the side with corrected P(win) > 0.5 (home_win from
    corrected_home_win; at 0.5 the toss-up defaults home). line 0.5 →
    P(favored win) (≡ outright win). line >= 1 → p_rl_{line}_home when the
    favorite is home (P(margin > L)) / p_rl_{line}_away when away. Returns
    (cover, is_home) float/bool (n,) arrays; cover NaN where unpricable."""
    n = len(decided)
    nan = np.full(n, np.nan)
    if not n:
        return nan, np.zeros(n, dtype=bool)
    hw = corrected_home_win(decided) if home_win is None else home_win
    is_home = np.where(np.isfinite(hw), hw >= 0.5, False)
    if line == 0.5:
        # Cover −0.5 ≡ outright win: P(favored wins) — always >= 0.5 by the
        # favored definition. Shipped for both sides via the corrected win.
        cover = np.where(np.isfinite(hw),
                         np.where(is_home, hw, 1.0 - hw), np.nan)
        return cover, is_home
    key = f"{line:.1f}".replace(".", "_")
    h_col, a_col = f"p_rl_{key}_home", f"p_rl_{key}_away"
    cover = nan.copy()
    if h_col in decided.columns:
        hcv = decided[h_col].to_numpy(float)
        # HOME favorite cover at −L = P(margin > L). For an AWAY favorite the
        # favorite is also quoted at −L, so its cover = P(away wins by > L) =
        # P(margin < −L) — which the home-frame artifact does NOT ship
        # (p_rl_{L}_away = P(margin < L) is the away +L DOG line, not the
        # favorite cover; using it would inflate away cover as L grows). Never
        # fabricate: away-favorite deep favorite lines price as unavailable
        # (NaN), so those games fall back to their reliable −0.5 line.
        cover = np.where(is_home, hcv, np.nan)
        cover = np.where(np.isfinite(cover), cover, np.nan)
    return cover, is_home


def runline_cut_history_frame(decided: pd.DataFrame) -> pd.DataFrame:
    """Cut-line run-line prediction history — per decided OOF game.

    Favored side (corrected NB moneyline) is picked at its CUT line: the
    DEEPEST margin L in {0.5, 1, 1.5, …, 4} with P(favored covers −L or
    +L) >= 0.5. P(cover 0.5) = P(favored win) >= 0.5 by the favored
    definition, so the cut is ALWAYS >= 0.5 and line 0 never occurs —
    pick the favored side at its cut.
    Resolution is the unified 3-way: favored COVERS if its margin > L,
    PUSH if == L (whole lines only — 0.5/1.5/2.5/3.5 never push), LOSS if
    < L. Push rows carry correct = NaN, so W/(W+L) excludes them from BOTH
    numerator and denominator (history_win_rate). Rows the grid cannot
    price are dropped — never fabricated."""
    empty = pd.DataFrame(columns=["game_pk", "game_date", "home_score",
                                  "away_score", "margin", "favored", "cut",
                                  "pick", "pick_prob", "winner", "push",
                                  "correct"])
    if not len(decided) or {"home_score", "away_score"}.difference(
            decided.columns):
        return empty
    if not {"p_rl_1_0_home", "p_rl_1_0_push"}.issubset(decided.columns):
        return empty  # legacy artifact: no p_rl grid → cannot price the cut
    hw = corrected_home_win(decided)
    margin = (decided["home_score"].to_numpy(float)
              - decided["away_score"].to_numpy(float))
    covers, is_home = {}, None
    for L in RUN_GRID_CUT:
        c, is_home = favored_cover_at(decided, L, home_win=hw)
        covers[L] = c
    fav_margin = np.where(is_home, margin, -margin)
    gd = _col_or(decided, "game_date")
    pk = _col_or(decided, "game_pk")
    rows = []
    for i in range(len(decided)):
        if not (np.isfinite(hw[i]) and np.isfinite(margin[i])):
            continue
        cut, cut_cover = 0.5, covers[0.5][i]
        if not np.isfinite(cut_cover):
            continue
        for L in RUN_GRID_CUT[1:]:
            v = covers[L][i]
            if np.isfinite(v) and v >= 0.5:
                cut, cut_cover = L, v
        fav = "home" if is_home[i] else "away"
        fm = fav_margin[i]
        whole = float(cut).is_integer()
        if fm > cut:
            winner = fav
            push = False
        elif whole and abs(fm - cut) < 1e-9:
            winner, push = "Push", True
        else:
            winner = "away" if fav == "home" else "home"
            push = False
        correct = np.nan if push else float(winner == fav)
        rows.append({
            "game_pk": pk.iloc[i], "game_date": gd.iloc[i],
            "home_score": decided["home_score"].iloc[i],
            "away_score": decided["away_score"].iloc[i],
            "margin": margin[i], "favored": fav, "cut": cut,
            "pick": fav, "pick_prob": round(float(cut_cover), 6),
            "winner": winner, "push": push, "correct": correct,
        })
    return pd.DataFrame(rows) if rows else empty


def filter_history_by_side(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    """Filter a pick history frame by the model's pick side.

    side in {"All", "Over", "Under"} for totals frames, {"All", "home",
    "away"} for run-line frames. Never mutates the input; empty/na → the
    frame returned unchanged."""
    s = (str(side) or "All").strip()
    if not len(frame) or s == "All" or "pick" not in frame.columns:
        return frame.copy()
    m = frame["pick"].astype(str) == s
    return frame[m].reset_index(drop=True)


def totals_monitor_stats(decided: pd.DataFrame, min_pct: int = 50,
                         side: str = "All") -> dict:
    """Pooled win-rate calibration for the totals monitor card — the
    scoring-mean diagnostic (Over under-predicting + Under over-predicting
    can net to zero pooled, so the side filter is required).

    Picks Over/Under at each game's OWN rounded total line (the same pick
    rule as the prediction-history view); keeps games whose favored-side
    pick_prob*100 > ``min_pct`` (cumulative — raising the threshold is a
    nested subset); optional side filter {"All","Over","Under"}. Win rate
    is W/(W+L) — pushes (total == whole-number line) excluded from BOTH
    numerator and denominator and folded out of the 2-way display. When
    side == "All" the per-side split is included (Over n / win rate vs
    Under n / win rate)."""
    empty = {"n": 0, "win_rate": None, "n_wins": 0, "n_losses": 0,
             "n_pushes": 0, "side": side, "min_pct": min_pct,
             "sides": {}}
    if not len(decided):
        return empty
    frame = totals_history_frame(decided)
    if not len(frame) or "pick_prob" not in frame.columns:
        return empty
    thr = max(float(min_pct) / 100.0, 0.0)
    kept = frame[frame["pick_prob"] > thr].reset_index(drop=True)
    view = filter_history_by_side(kept, side)
    stats = history_win_rate(view)
    ok = view["correct"].notna()
    n_wins = int(view.loc[ok, "correct"].sum()) if ok.any() else 0
    out = {"n": stats["n_games"], "win_rate": stats["win_rate"],
           "n_wins": n_wins,
           "n_losses": stats["n_games"] - n_wins,
           "n_pushes": stats["n_pushes"], "side": side,
           "min_pct": int(min_pct), "sides": {}}
    if side == "All" and len(kept):
        for s in ("Over", "Under"):
            sub = filter_history_by_side(kept, s)
            st = history_win_rate(sub)
            out["sides"][s] = {"n": st["n_games"],
                               "win_rate": st["win_rate"],
                               "n_pushes": st["n_pushes"]}
    return out


def runline_monitor_stats(decided: pd.DataFrame, line: float) -> dict:
    """Pooled win-rate calibration for the run-line monitor card at the
    toggled favorite-side line ``line`` (magnitude in RUN_GRID_CUT).

    Picks the MONEYLINE FAVORITE each game (NOT a >50% cover pick — its
    cover P is often < 50% on deeper lines; this is a calibration check).
    Resolution is the unified 3-way (favored covers if its margin > L,
    push if == L on whole lines, loss if < L); win rate is 2-way
    re-normalized W/(W+L) with whole-line pushes folded out of both.
    ``cover_pred_mean`` is the RAW predicted cover rate (pushes in the
    denominator); ``predicted_2way`` re-normalizes it to the SAME basis as
    the win rate (whole-line pushes folded out of both sides — half-lines
    are unchanged). Returns pooled stats + per-favored-side split."""
    empty = {"line": line, "n": 0, "n_home": 0, "n_away": 0,
             "n_wins": 0, "n_losses": 0, "n_pushes": 0,
             "cover_pred_mean": None, "win_rate": None, "sides": {}}
    if not len(decided):
        return empty
    if not RUN_GRID_CUT or line not in RUN_GRID_CUT:
        return empty
    margin = (decided["home_score"].to_numpy(float)
              - decided["away_score"].to_numpy(float)) \
        if {"home_score", "away_score"}.issubset(decided.columns) else None
    if margin is None:
        return empty
    cover, is_home = favored_cover_at(decided, line)
    hw = corrected_home_win(decided)
    valid = np.isfinite(hw) & np.isfinite(margin)
    n = int(valid.sum())
    if not n:
        return empty
    fav_margin = np.where(is_home, margin, -margin)
    whole = float(line).is_integer()
    cov = fav_margin[valid] > line
    pushes = (fav_margin[valid] == line) if whole else np.zeros(n, bool)
    losses = ~cov & ~pushes
    n_wins, n_pushes = int(cov.sum()), int(pushes.sum())
    n_losses = int(losses.sum())
    denom = n_wins + n_losses
    # Predicted cover only where the artifact prices it (home favorites at
    # deep lines; both sides at −0.5). Away-favorite deep favorite lines are
    # not shipped → excluded from cover_pred_mean, never fabricated.
    priced_cover = np.isfinite(cover) & valid
    cp = float(cover[priced_cover].mean()) if priced_cover.any() else None
    # 2-way re-normalized predicted cover — the SAME basis as win_rate
    # (W/(W+L)): whole-line pushes folded out of both sides.
    # predicted_2way = P(cover) / (P(cover) + P(dog)) = mean(cover) /
    # (1 − mean(push)) over the priced subset (dog = 1 − cover − push per
    # game; the same re-scaling convention as the totals card's re-scaled
    # 2-way probabilities — ratio preserved, sums to 100%). Half-lines
    # never push, so predicted_2way == raw cover_pred_mean there. Away-
    # favorite deep lines are unpriced (cover NaN) and their push P is
    # unshipped, so they drop out of both sides consistently.
    p2 = None
    if cp is not None:
        if whole:
            pkey = f"{line:.1f}".replace(".", "_")
            pcol = f"p_rl_{pkey}_push"
            pushp = (decided[pcol].to_numpy(float)
                     if pcol in decided.columns else None)
            mp = None
            if pushp is not None:
                pp_ok = np.isfinite(pushp) & priced_cover
                if pp_ok.any():
                    mp = float(pushp[pp_ok].mean())
            if mp is not None and mp < 1.0:
                p2 = cp / (1.0 - mp)
        else:
            p2 = cp
    n_home = int((valid & is_home).sum())
    out = {"line": line, "n": n, "n_home": n_home, "n_away": n - n_home,
           "n_wins": n_wins, "n_losses": n_losses, "n_pushes": n_pushes,
           "cover_pred_mean": (round(cp, 4) if cp is not None else None),
           "predicted_2way": (round(p2, 4) if p2 is not None else None),
           "win_rate": (round(n_wins / denom, 6) if denom else None),
           "cover_available": int(priced_cover.sum()),
           "sides": {}}
    for label, mask in (("home", valid & is_home), ("away", valid & ~is_home)):
        k = int(mask.sum())
        if not k:
            continue
        w = int((fav_margin[mask] > line).sum())
        p = int((fav_margin[mask] == line).sum()) if whole else 0
        l = k - w - p
        out["sides"][label] = {
            "n": k, "n_wins": w, "n_losses": l, "n_pushes": p,
            "win_rate": (round(w / (w + l), 6) if (w + l) else None)}
    return out


def build_team_map(game_level: pd.DataFrame) -> dict:
    """Dual-keyed team map from the game-level features frame.

    Keys are int(StatsAPI game_pk) AND str(ESPN game_id) — the
    145d841 slate-key convention — so a markets-artifact key resolves
    either way (the markets CSV mixes numeric game_pk with ESPN
    game_id slate rows in ONE object-dtype column). Rows missing either
    team name are skipped; duplicate keys keep the first occurrence.
    Returns {} (never raises) when the frame is empty/missing columns.
    """
    out: dict = {}
    if not len(game_level) or {"home_team", "away_team"}.difference(
            game_level.columns):
        return out
    for _, r in game_level.iterrows():
        if pd.isna(r["home_team"]) or pd.isna(r["away_team"]):
            continue
        tup = (str(r["away_team"]), str(r["home_team"]))  # away, home
        pk = r.get("game_pk")
        if pd.notna(pk):
            try:
                out.setdefault(int(float(pk)), tup)
            except (TypeError, ValueError):
                pass
        gid = r.get("game_id")
        if pd.notna(gid):
            out.setdefault(str(gid).strip(), tup)
    return out


def resolve_matchup_teams(team_map: dict, key: Any) -> tuple:
    """(away, home) for one markets-artifact game key.

    The _resolve_slate_key discipline (145d841): the numeric StatsAPI
    game_pk wins (the key may arrive as int, float 778485.0, or the
    object-dtype string '778485.0' from a mixed column); an ESPN
    game_id string (e.g. '20260826_TB@DET') is the fallback. Returns
    ("—", "—") only when the key resolves NEITHER way — the render
    layer's honest placeholder for a genuinely unresolvable row, never
    a fabricated team.
    """
    if key is None or (isinstance(key, float) and pd.isna(key)):
        return ("—", "—")
    if isinstance(key, (int, np.integer)):
        cand = int(key)
    elif isinstance(key, float):
        cand = int(key) if key.is_integer() else None
    else:
        s = str(key).strip()
        try:
            cand = int(float(s))
        except ValueError:
            cand = None
    if cand is not None and cand in team_map:
        return team_map[cand]
    if isinstance(key, str):
        hit = team_map.get(key.strip())
        if hit is not None:
            return hit
    return ("—", "—")


# ---------------------------------------------------------------------------
# Today's Games card enrichment (read-only over the slate rows)
# ---------------------------------------------------------------------------


def _num(row, key: str) -> Optional[float]:
    """Coerce a row value to float; None for missing/NaN/non-numeric."""
    v = row.get(key) if isinstance(row, dict) else getattr(row, key, None)
    try:
        return None if v is None or pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return None


def run_engine_card_bits(game_id: str,
                         slate_map: Optional[dict] = None,
                         line: Optional[float] = None,
                         rl_line: Optional[float] = None) -> Optional[dict]:
    """Run-engine projections for one Today's Games card, joined by
    game_id == slate game_pk (the 145d841 ESPN-id convention).

    The O/U split is priced at the game's OWN rounded total — nearest 0.5
    of home_expected_runs + away_expected_runs (e.g. 4.9 + 4.4 = 9.3 →
    9.5) — pulled from the grid columns at that line (p_over_9_5 /
    p_under_9_5), unless an explicit ``line`` override is given (the
    per-card selector / future market-lines mode): then the split is
    priced at THAT grid line instead, and line_selected records it so the
    card can flag a non-default line. A ``line`` outside the shipped grid
    (or None) falls back to the game's own rounded line — the selector's
    guard, so an invalid/out-of-grid choice can never crash the card.
    Lines outside the shipped grid clamp to the nearest edge with
    clamped=True (the card notes it). Never fabricated: missing
    columns / NaN degrade has_grid to False (quiet 'n/a') and a missing
    slate row returns None (strip omitted). p_away_cover is the exact
    complement of p_home_cover (1 − p) because the artifact ships
    home-cover columns only; the card labels it as such.
    """
    if not slate_map:
        return None
    row = slate_map.get(str(game_id))
    if row is None:
        return None
    proj_home = _num(row, "home_expected_runs")
    proj_away = _num(row, "away_expected_runs")
    p_home_cover = _num(row, RUN_COVER_COL)
    bits: dict[str, Any] = {
        "proj_home": proj_home,
        "proj_away": proj_away,
        "total_line": None,
        "clamped": False,
        "line_selected": None,
        "p_over": None,
        "p_under": None,
        "p_home_cover": p_home_cover,
        "p_away_cover": (None if p_home_cover is None
                         else 1.0 - p_home_cover),
        "rl_line": None,
        "rl_line_default": 1.5,
        "rl_home": None,
        "rl_away": None,
        "rl_push": 0.0,
        "rl_unverified": False,
        "has_grid": False,
    }
    # --- per-card run-line selection (mirrors the O/U line override) ---
    # rl_home/rl_away are the RE-SCALED 2-way display values; the raw 3-way
    # (rl_home_raw / rl_push / rl_away_raw, sums to 1.0) is carried for EV
    # math (a push refunds the stake: EV = payout·P(home) − stake·P(away)
    # + 0·P(push)).
    # Default: the legacy ±1.5 line (the only one pre-p_rl artifacts
    # carry). An explicit rl_line must be a valid grid margin; anything
    # else falls back to 1.5. p_rl_* columns (post-RL-grid artifacts)
    # give the 3-way split (home / push / away); the legacy artifact
    # lacks them, so ±1.5 still resolves via p_home_cover_1_5 and other
    # lines render as unverified (never fabricated).
    use_rl = 1.5
    rl_selected = None
    if rl_line is not None:
        try:
            rl_line = round(float(rl_line), 1)
        except (TypeError, ValueError):
            rl_line = None
        if rl_line in RUN_LINE_GRID_FULL:
            use_rl = rl_line
            rl_selected = rl_line
    rl_home = rl_away = None
    rl_home_raw = rl_away_raw = None
    rl_push = 0.0
    rl_unverified = False
    h_col, push_col, a_col = rl_cols(use_rl)
    if h_col in row and push_col in row and a_col in row:
        vh = _num(row, h_col)
        vp = _num(row, push_col)
        va = _num(row, a_col)
        if vh is not None and va is not None:
            rl_home = rl_home_raw = vh
            rl_away = rl_away_raw = va
            rl_push = vp if vp is not None else 0.0
    elif use_rl == 1.5:
        # Legacy artifact: ±1.5 via p_home_cover_1_5 (complement, no push).
        if p_home_cover is not None:
            rl_home = rl_home_raw = p_home_cover
            rl_away = rl_away_raw = 1.0 - p_home_cover
    else:
        rl_unverified = True
    if proj_home is not None and proj_away is not None:
        # Default own line = FAIR line (grid argmin of |re-scaled P(over)
        # − 0.5|, ties → lower), the legal 50/50 anchor; falls back to the
        # round-half-up λ_home+λ_away projection only when the grid cannot
        # price the row (never fabricated).
        fair = fair_total_line_row(row)
        if fair is not None:
            model_line, clamped = float(fair), False
        else:
            model_line, clamped = clamp_to_grid(
                round_to_half(proj_home + proj_away))
        # Explicit line override: accept only a valid grid line; anything
        # else falls back to the model's own line (never crash the card).
        use_line = model_line
        line_selected = None
        if line is not None:
            try:
                line = round(float(line), 1)
            except (TypeError, ValueError):
                line = None
            if line in TOTAL_GRID:
                use_line = line
                clamped = False
                line_selected = line
        line, clamped = use_line, clamped
        over_col, under_col = grid_over_under_cols(line)
        p_over = _num(row, over_col)
        p_under = _num(row, under_col)
        if p_over is not None and p_under is not None:
            # P(push) for whole-number lines = P(total == line). Post-65b44ec
            # artifacts ship an explicit p_push_<line> column (P(total ==
            # line) from the same MC draws) — ALWAYS prefer it (exact, no
            # neighbor dependency, works at grid edges). Legacy artifacts
            # lack it: fall back to the grid difference against the LOWER
            # neighbor, p_over(L−0.5) − p_over(L) = P(total == L) for
            # integer totals (strict over: p_over(L−0.5) = P(total ≥ L),
            # p_over(L) = P(total ≥ L+1)). The old p_over(L) − p_over(L+0.5)
            # direction is INVERTED: on whole lines both thresholds reduce
            # to total ≥ L+0.5, so it is always 0, and on half-lines it
            # returns the NEIGHBOR line's push (P(total == L+0.5)) — the
            # half-line-shows-push / whole-line-shows-none display bug.
            # Half-lines can never push (totals are integers), so their
            # fallback is exactly 0 — never a neighbor's push.
            p_push = _num(row, grid_push_col(line))
            if p_push is None:
                if line == int(line):      # whole-number line
                    over_prev_col, _ = grid_over_under_cols(line - 0.5)
                    p_over_prev = _num(row, over_prev_col)
                    if p_over_prev is not None:
                        p_push = max(0.0, p_over_prev - p_over)
                else:                      # half-line: integer totals never
                    p_push = 0.0           #   land on x.5, so no push band
            # Re-scale Over/Under so they sum to 100% by folding the push
            # proportionately into each side (a push refunds the bet;
            # sportsbooks price whole-number lines this way). The over:under
            # ratio is preserved: P(over|no push) = P(over)/[P(over)+P(under)],
            # P(under|no push) = P(under)/[P(over)+P(under)]. Half-lines have
            # p_push = 0 so the re-scaled values equal the raw ones. The raw
            # p_over/p_under/p_push stay in the dict — EV math needs all
            # three (EV = payout×P(over) − stake×P(under) + 0×P(push)).
            p_over_raw, p_under_raw, p_push_raw = p_over, p_under, p_push
            denom = (p_over + p_under) if p_push is not None else None
            p_over_disp, p_under_disp = p_over_raw, p_under_raw
            if denom and denom > 0:
                scale = 1.0 / denom
                p_over_disp = p_over_raw * scale
                p_under_disp = p_under_raw * scale
            bits.update({"total_line": line, "clamped": clamped,
                         "line_selected": line_selected,
                         "p_over": p_over_disp, "p_under": p_under_disp,
                         "p_push": p_push_raw, "p_over_raw": p_over_raw,
                         "p_under_raw": p_under_raw, "has_grid": True})
    # Run-line display: RE-SCALED 2-way cover % (push folded proportionately
    # into home/away so they sum to 100% — same convention as the O/U
    # split). The raw 3-way (home/push/away) is never shown on the card but
    # stays available for EV (push refunds the stake). Half-lines have
    # push = 0 → re-scale is a no-op.
    if rl_home is not None and rl_away is not None:
        denom = rl_home + rl_away
        if denom > 0:
            scale = 1.0 / denom
            rl_home, rl_away = rl_home * scale, rl_away * scale
    bits.update({"rl_line": rl_selected, "rl_line_default": 1.5,
                 "rl_home": rl_home, "rl_away": rl_away,
                 "rl_home_raw": rl_home_raw, "rl_away_raw": rl_away_raw,
                 "rl_push": rl_push, "rl_unverified": rl_unverified})
    return bits


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






ACC_Y_AXIS_FLOOR = 75.0   # accuracy axis is in PERCENT units; floor in %
ACC_Y_HEADROOM = 5.0      # points above the largest visible accuracy value


def chart_pick_buckets(table: dict, title: str,
                       total_line: bool = False,
                       acc_y_max: Optional[float] = None) -> dict:
    """Returns {'chart': alt.Chart, 'table': pd.DataFrame} for dual-axis
    rendering; count bars + accuracy line are layered in the page.

    total_line=True overlays a CONSTANT horizontal reference line on the
    accuracy axis at the pooled win rate (table['win_rate'], across ALL
    picks — not a per-bucket point), labeled 'Total (n=…): …%', so the
    overall hit rate is visible on the chart itself rather than only in the
    caption.

    acc_y_max (percent units) floors the accuracy axis domain max so the
    line doesn't ride the top edge when buckets cluster (e.g. ~54–57%): the
    actual max is max(acc_y_max, largest accuracy value + headroom), so a
    future bucket that genuinely exceeds the floor is never clipped. Other
    callers pass the defaults (False / None) → the accuracy axis still gets
    an explicit full 0–100% domain.

    The accuracy line ALWAYS carries a real scale with an explicit domain.
    A regression passed scale=None on the default path, which serializes to
    "scale": null in the vega-lite spec — vega-lite treats that as "disable
    the scale and drop the axis", so the line rendered on the count scale
    with no right-side accuracy axis. Every path now emits a real scale, so
    the independent right-side accuracy axis is always present.
    """
    tdf = pd.DataFrame(table["buckets"])
    if tdf.empty:
        return {"chart": alt.Chart(pd.DataFrame()).mark_bar(), "table": tdf}
    base = alt.Chart(tdf).encode(
        x=alt.X("bucket:N", sort=list(tdf["bucket"]), title=None))
    bars = base.mark_bar(color="#3B82F6").encode(
        y=alt.Y("count:Q", title="Picks"),
        tooltip=["bucket", "count", "accuracy"])
    # Accuracy axis domain (percent units) — ALWAYS a real scale. Never
    # emit scale=None: it serializes to "scale": null, which vega-lite
    # treats as "disable the scale, drop the axis" (the Run-line picks
    # regression). Explicit domain: 0–100% by default (never clips a rate
    # ≤ 100), or the acc_y_max floor with headroom when the caller asks.
    vals = [float(v) for v in tdf["accuracy"].dropna()]
    rate = table.get("win_rate")
    if total_line and isinstance(rate, (int, float)):
        vals.append(float(rate) * 100.0)
    data_max = max(vals) if vals else 0.0
    if acc_y_max is not None:
        y_max = max(float(acc_y_max), data_max + ACC_Y_HEADROOM)
    else:
        y_max = 100.0
    acc_scale = alt.Scale(domain=[0.0, y_max])
    acc = base.mark_line(color="#22C55E", strokeWidth=2.5,
                         point=alt.OverlayMarkDef(size=60)).encode(
        y=alt.Y("accuracy:Q", title="Actual hit %", scale=acc_scale),
        tooltip=["bucket", "count", "accuracy"])
    acc_layers = [acc]
    if total_line:
        rate = table.get("win_rate")
        if isinstance(rate, (int, float)):
            n = int(table.get("n_games") or 0)
            yv = float(rate) * 100.0
            label = f"Total (n={n:,}): {yv:.1f}%"
            line_df = pd.DataFrame({"y": [yv]})
            rule = alt.Chart(line_df).mark_rule(
                color="#F59E0B", strokeDash=[6, 4], strokeWidth=2).encode(
                y=alt.Y("y:Q"))
            last_bucket = tdf["bucket"].iloc[-1]
            txt = alt.Chart(pd.DataFrame({"bucket": [last_bucket],
                                          "y": [yv]})).mark_text(
                text=label, align="right", baseline="bottom",
                color="#FBBF24", fontSize=11, dx=-8, dy=-4).encode(
                x=alt.X("bucket:N", sort=list(tdf["bucket"])),
                y=alt.Y("y:Q"))
            acc_layers += [rule, txt]
    acc_layer = alt.layer(*acc_layers)
    return {"chart": alt.layer(bars, acc_layer).resolve_scale(y="independent")
            .properties(height=280, title=title),
            "table": tdf}


# ---------------------------------------------------------------------------
# Distributional-fit panel rows (pure extraction — no streamlit)
# ---------------------------------------------------------------------------
# The reader side of the fit block written by
# backend/pipeline._run_engine_fit_block. Keys must stay reconciled with
# that writer and with run_engine.py's monitor-embed dict (alpha_home /
# alpha_away curves, dispersion_chi2_per_df, fit_tables, variance_check,
# mc_meta). Both curve shapes are handled: v2 piecewise {form, lam, alpha,
# selection.chosen} and v1 {form, a[, c], selection.chosen}.


def alpha_hat(curve: Optional[dict]) -> tuple[Optional[float], Optional[str]]:
    """Scalar α estimate + chosen form from an α(λ) curve dict.

    Uses the selection-bin alpha (weighted by bin count) when present — the
    same bins the backend fit displayed — falling back to the curve's alpha
    array mean or the parametric ``a`` scalar. Never raises.
    """
    if not isinstance(curve, dict):
        return None, None
    sel = curve.get("selection")
    form = (sel or {}).get("chosen") if isinstance(sel, dict) else None
    if not form:
        form = curve.get("form")
    bins = (sel or {}).get("bins") if isinstance(sel, dict) else None
    if isinstance(bins, list) and bins:
        alphas = [b.get("alpha") for b in bins if isinstance(b, dict)]
        counts = [b.get("count", 1) for b in bins if isinstance(b, dict)]
        if alphas and all(isinstance(a, (int, float)) for a in alphas):
            total = float(sum(counts)) or 1.0
            return (float(sum(a * c for a, c in zip(alphas, counts))) /
                    total), form
    alphas = curve.get("alpha")
    if (isinstance(alphas, list) and alphas
            and all(isinstance(a, (int, float)) for a in alphas)):
        return float(sum(alphas)) / len(alphas), form
    a = curve.get("a")
    if isinstance(a, (int, float)):
        return float(a), form
    return None, form


def fit_tail_rows(tbl: Optional[list]) -> list[tuple[str, float, float]]:
    """Tail rows (k<=1, k>=10) from a per-side fit-check table.

    Handles the backend's labels ("≤1" / "≥10" plus the legacy "<={1}" /
    ">={10}") and both row-key spellings (observed_p/modeled_p and
    observed/modeled).
    """
    if not isinstance(tbl, list):
        return []
    out: list[tuple[str, float, float]] = []
    for r in tbl:
        if not isinstance(r, dict):
            continue
        k = r.get("k")
        if k in ("≤1", "<={1}"):
            label = "k≤1"
        elif k in ("≥10", ">={10}"):
            label = "k≥10"
        else:
            continue
        obs = r.get("observed_p", r.get("observed"))
        mod = r.get("modeled_p", r.get("modeled"))
        if not isinstance(obs, (int, float)) or not isinstance(mod,
                                                               (int, float)):
            continue
        out.append((label, float(obs), float(mod)))
    return out


def mc_caption(mc: Optional[dict]) -> Optional[str]:
    """Monte Carlo metadata line. Formats n_draws with thousands separators
    ONLY when it is numeric (the historical bug: the '--' default string hit
    ``f"{:,.0f}"`` and raised ValueError)."""
    if not isinstance(mc, dict):
        return None
    n = mc.get("n_draws", mc.get("n_samples"))
    n_str = None
    if isinstance(n, (int, float)) and not isinstance(n, bool):
        n_str = (f"{int(n):,}" if float(n).is_integer() else f"{n:,.3f}")
    parts: list[str] = []
    if n_str is not None:
        parts.append(f"{n_str} draws")
    reason = mc.get("reason")
    if isinstance(reason, str) and reason:
        parts.append(f"reason={reason}")
    se = mc.get("mc_se_totals_max")
    if isinstance(se, (int, float)):
        parts.append(f"totals MC se max {se:.4f}")
    return "Monte Carlo: " + " · ".join(parts) if parts else None


def lambda_edge(fit: Optional[dict]) -> Optional[float]:
    """Modeled home-away run differential from the α(λ) curve bins (weighted
    mean of each side's bin mean_lam) — the NB sampler's λ edge, used by the
    run-engine model card. None when either side lacks bin data."""
    if not isinstance(fit, dict):
        return None

    def _mean_lam(curve) -> Optional[float]:
        if not isinstance(curve, dict):
            return None
        bins = (curve.get("selection") or {}).get("bins") \
            if isinstance(curve.get("selection"), dict) else None
        if not isinstance(bins, list) or not bins:
            return None
        lams = [b.get("mean_lam") for b in bins if isinstance(b, dict)]
        counts = [b.get("count", 1) for b in bins if isinstance(b, dict)]
        if not lams or not all(isinstance(x, (int, float)) for x in lams):
            return None
        total = float(sum(counts)) or 1.0
        return float(sum(x * c for x, c in zip(lams, counts))) / total

    lh = _mean_lam(fit.get("alpha_home"))
    la = _mean_lam(fit.get("alpha_away"))
    if lh is None or la is None:
        return None
    return lh - la


def fit_panel_rows(fit: Optional[dict]) -> dict:
    """Reconcile a monitor fit block into render-ready rows. Every access is
    defensive: a fit dict missing EVERY key yields all-default rows (the
    '--' fallbacks), never a crash."""
    fit = fit or {}
    alpha_home, form_home = alpha_hat(fit.get("alpha_home"))
    alpha_away, form_away = alpha_hat(fit.get("alpha_away"))
    chi2 = fit.get("dispersion_chi2_per_df") or {}
    vc = fit.get("variance_check") or {}
    vc_h = vc.get("home") or {}
    vc_a = vc.get("away") or {}
    tables = fit.get("fit_tables") or {}
    tails: dict[str, str] = {}
    for label, key in (("Home", "home"), ("Away", "away")):
        rows = fit_tail_rows(tables.get(key))
        if rows:
            tails[label] = " | ".join(
                f"{lab}: obs={o:.3f} mod={m:.3f}" for lab, o, m in rows)
    fitted_on = None
    for curve in (fit.get("alpha_home"), fit.get("alpha_away")):
        if isinstance(curve, dict) and curve.get("fitted_on"):
            fitted_on = curve.get("fitted_on")
            break
    return {
        "alpha_home": alpha_home,
        "alpha_home_form": form_home,
        "alpha_away": alpha_away,
        "alpha_away_form": form_away,
        "chi2_home": chi2.get("home"),
        "chi2_away": chi2.get("away"),
        "variance_home": (vc_h.get("implied_var"), vc_h.get("observed_var")),
        "variance_away": (vc_a.get("implied_var"), vc_a.get("observed_var")),
        "fitted_on": fitted_on,
        "tails": tails,
        "mc_caption": mc_caption(fit.get("mc_meta")),
    }


