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
TOTALS_PICK_LABELS = ["50-55", "55-60", "60-65", "65+"]


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


def _rounded_lines(decided: pd.DataFrame) -> np.ndarray:
    """Per-game rounded total line (nearest 0.5 of λ_home + λ_away, clamped
    to the shipped grid) — shared by pairs, picks, and push detection."""
    exp_h = decided["home_expected_runs"].to_numpy(float)
    exp_a = decided["away_expected_runs"].to_numpy(float)
    return np.array([clamp_to_grid(round_to_half(h + a))[0]
                     for h, a in zip(exp_h, exp_a)])


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


def totals_pick_table(decided: pd.DataFrame) -> dict:
    """Favored-side pick at each game's rounded total line.

    pick = over if P(over own rounded total) ≥ 0.5 else under; confidence
    = max(p_over, 1 − p_over). Bucketed by that confidence (50–55, 55–60,
    60–65, 65+). PUSHES (total == whole-number line) are excluded from the
    buckets and the pooled win_rate — neither wins nor losses — and reported
    as n_pushes / push_rate. Pushed games are UNDER-favored games landing
    exactly on the line (rounded line at/above the expected total → under
    favored), previously scored as wins, so excluding them LOWERS the
    honest win_rate vs the inflated one. win_rate is the POOLED accuracy
    across every non-push pick: hit rate is NOT calibration, and
    high-confidence buckets are small because every game sits at its own
    line.
    """
    if not len(decided) or "total_runs" not in decided.columns:
        return {"buckets": [], "n_games": 0, "n_pushes": 0,
                "push_rate": 0.0,
                "warning": "Missing outcomes or expected totals."}
    if ({"home_expected_runs", "away_expected_runs"}
            .difference(decided.columns)):
        return {"buckets": [], "n_games": 0, "n_pushes": 0,
                "push_rate": 0.0,
                "warning": "Missing expected-runs columns."}
    total = decided["total_runs"].to_numpy(float)
    lines = _rounded_lines(decided)
    p_arr = np.full(len(decided), np.nan)
    line_arr = np.full(len(decided), np.nan)
    for i in range(len(decided)):
        line = lines[i]
        over_col, _ = grid_over_under_cols(line)
        if over_col in decided.columns:
            v = decided[over_col].iloc[i]
            if not pd.isna(v):
                p_arr[i] = float(v)
                line_arr[i] = line
    ok = ~np.isnan(p_arr)
    # Pushes (total == whole-number line) are UNDER-favored games landing
    # exactly on the line — neither wins nor losses — excluded from the
    # win-rate denominator and the accuracy buckets.
    is_push = (total == line_arr) & ok
    non_push = ok & ~is_push
    n = int(non_push.sum())
    n_pushes = int(is_push.sum())
    if not n:
        return {"buckets": [], "n_games": 0, "n_pushes": n_pushes,
                "push_rate": (round(n_pushes / (n + n_pushes), 4)
                              if (n + n_pushes) else 0.0),
                "warning": "No non-push games with a rounded-total grid "
                            "column."}
    p = p_arr[non_push]
    line_v = line_arr[non_push]
    total_v = total[non_push]
    pick_over = p >= 0.5
    hit = (pick_over.astype(float)
           == (total_v >= line_v + 0.5).astype(float))
    out = pick_buckets(np.maximum(p, 1 - p), hit.astype(float),
                       labels=TOTALS_PICK_LABELS)
    out["n_games"] = n
    out["n_pushes"] = n_pushes
    out["push_rate"] = round(n_pushes / (n + n_pushes), 4)
    out["win_rate"] = round(float(hit.mean()), 4)
    out["pick_rule"] = ("over if P(over own rounded total) >= 0.5, "
                        "else under")
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

    Each row is priced at the game's OWN rounded total — nearest 0.5 of
    λ_home + λ_away, clamped to the shipped grid — the SAME line the
    diagnostics' totals-picks chart uses, so the two agree exactly.
    pick = Over if p_over(line) >= 0.5 else Under (the diagnostics tie
    convention; p_under is the exact mirror 1 − p_over); pick_prob is the
    favored side's probability. winner = Over when total_runs > line,
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
        over_col, _ = grid_over_under_cols(line)
        if over_col not in decided.columns:
            continue
        p = decided[over_col].iloc[i]
        if pd.isna(p):
            continue
        p = float(p)
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
                         slate_map: Optional[dict] = None) -> Optional[dict]:
    """Run-engine projections for one Today's Games card, joined by
    game_id == slate game_pk (the 145d841 ESPN-id convention).

    The O/U split is priced at the game's OWN rounded total — nearest 0.5
    of home_expected_runs + away_expected_runs (e.g. 4.9 + 4.4 = 9.3 →
    9.5) — pulled from the grid columns at that line (p_over_9_5 /
    p_under_9_5). Lines outside the shipped grid clamp to the nearest edge
    with clamped=True (the card notes it). Never fabricated: missing
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
        "p_over": None,
        "p_under": None,
        "p_home_cover": p_home_cover,
        "p_away_cover": (None if p_home_cover is None
                         else 1.0 - p_home_cover),
        "has_grid": False,
    }
    if proj_home is not None and proj_away is not None:
        line, clamped = clamp_to_grid(round_to_half(proj_home + proj_away))
        over_col, under_col = grid_over_under_cols(line)
        p_over = _num(row, over_col)
        p_under = _num(row, under_col)
        if p_over is not None and p_under is not None:
            # P(push) for whole-number lines: P(total == line) via the
            # grid — difference between the p_over at line L (push-
            # inclusive threshold) and at line L+0.5 (strict threshold).
            # Half-lines can never push, so this is always 0 for them.
            p_push = None
            over_next_col, _ = grid_over_under_cols(line + 0.5)
            p_over_next = _num(row, over_next_col)
            if p_over_next is not None:
                p_push = max(0.0, p_over - p_over_next)
            bits.update({"total_line": line, "clamped": clamped,
                         "p_over": p_over, "p_under": p_under,
                         "p_push": p_push, "has_grid": True})
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
