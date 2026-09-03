"""NFL market layer — run-line/totals chain (record-only).

Targets the residual totals top-bin defect (re-baseline 5eb7d5c: seam
totals ECE 0.087, totals top bin pred 0.7739 vs actual 0.5273) that the
sigma record (3480b05) identified as model-vs-market information asymmetry:
the model's derived totals run hot against the offered line at high lines
(sigma record: d=0.34 SE 0.07 in actual_total ~ c + d*(mu_T_hat - line),
monotone mu_T tercile tilt 0.59 low / 0.40 high).

DESIGN RULE (verbatim scope): NO pooled static overlay of any kind — the
disagreement coefficients (c, d) are fitted over the folds (second-level
walk-forward over the OOF val weeks) and transferred to sealed 2025 by
median-of-fold, mirroring the MLB median-fold-rounds refit convention.
The prior pooled-map failure mode (away -0.14 -> +1.45 on sealed transfer,
record 56893d3) is forbidden. The spread side is already sane (seam covers
ECE 0.078) and stays untouched — the delta/2 shift keeps the margin center
mu_H - mu_A unchanged by construction. The derived ML stays a G4-style
coherence report only (the board moneyline is the frozen 12-pool
incumbent — no Platt, no challenge here). No wiring, no engine edits;
moneyline FEATURE_COLUMNS / 12-pool / daily pipeline untouched.

Only fitted thing in this layer: per-fold (c_k, d_k) by OLS on
    actual_total - line ~ c + d*(mu_T_hat - line)
using ONLY the val rows of strictly-prior folds (weeks < k). Warmup folds
(< MIN_PRIOR_ROWS prior rows) use d=1, c=0 — no-shrink, quote the model;
shrinkage is never fabricated.

Product arms (both re-quoted per game):
  A) own-line: the model's fair total (median of the total PMF) with NO
     market blending — honest ECE reported as-is. This arm is exactly the
     era chain on the fixed engine, so it reproduces the re-baseline seam
     numbers (totals ECE 0.087, top bin 0.7739, covers ECE 0.078) — the
     C0 no-shrink machinery check.
  B) shrink-to-line: mu*_T = line + c_k + d_k*(mu_T_hat - line); shift
     both per-side means by delta/2 (mu_H += delta/2, mu_A += delta/2) and
     rebuild the joint PMFs through the engine's public entrypoints
     (era-centered mu in, DN const-sigma rho IPF-tie machinery unchanged).
     Margin center unchanged by construction -> P(cover -L) untouched.

All operations are deterministic (no RNG): identical inputs -> identical
outputs (the G4 determinism pin).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import nfl_joint_engine as je  # noqa: E402  (committed entrypoints only)

# Warmup threshold: a fold with fewer than this many strictly-prior val
# rows gets d=1, c=0 (no-shrink — quote the model, never fabricate
# shrinkage).
MIN_PRIOR_ROWS = 50

# Per-game market record columns (the honest-ECE emitter contract).
MARKET_RECORD_COLUMNS = [
    "game_id", "season", "week_start", "total_line", "spread_line",
    "total", "margin", "mu_T_hat",
    "fair_total_own", "fair_total_shrunk",
    "p_over_own", "p_over_shrunk", "p_cover_own", "p_cover_shrunk",
    "derived_ml_own", "derived_ml_shrunk",
    "y_over", "y_cover", "y_home_win",
    "used_c", "used_d", "is_warmup",
]


# ── Offered-line loading (nflreadpy; 100% spread+total coverage on the
#    pooled OOF + sealed rows — probed) ──────────────────────────────────────

def load_offered_lines(seasons: list[int] | None = None) -> pd.DataFrame:
    """Pull nflreadpy schedule lines -> [game_id, spread_line, total_line].

    Sign convention (locked by corr(spread_line, margin) = +0.446 on the
    pooled rows): positive spread_line = home favored; home covers iff
    margin > spread_line; over iff total > total_line.
    """
    import nflreadpy
    seasons = seasons or [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    sch = nflreadpy.load_schedules(seasons)
    if hasattr(sch, "to_pandas"):
        sch = sch.to_pandas()
    need = {"game_id", "spread_line", "total_line"}
    missing = [c for c in need if c not in sch.columns]
    if missing:
        raise RuntimeError(f"load_offered_lines: schedule missing {missing}")
    out = sch[["game_id", "spread_line", "total_line"]].copy()
    out = out.drop_duplicates(subset=["game_id"], keep="last")
    return out


# ── Market frame (Step 0) ───────────────────────────────────────────────────

def build_market_frame(pooled: pd.DataFrame, sealed: pd.DataFrame,
                       feats: pd.DataFrame, week_map: dict[str, Any],
                       lines: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge era-centered mu + offered lines + actuals into one market frame.

    Every row must have BOTH an era-centered mu (pred_home/pred_away) and an
    offered line (spread_line/total_line) — Step-0 assertion, fail loud.

    Args:
        pooled: era-centered pooled-OOF per-side outputs (game_id,
            pred_home, pred_away, home_score, away_score), n=1091.
        sealed: era-centered sealed 2025 outputs, n=285.
        feats: decided feature frame (game_id, season, gameday, ...).
        week_map: game_id -> week_start (Monday) for the OOF val rows,
            from the SAME generate_weekly_folds geometry the era walk used.
        lines: nflreadpy [game_id, spread_line, total_line].

    Returns (market_pooled, market_sealed) with columns game_id, season,
    week_start (pooled only), pred_home, pred_away, mu_T_hat, total_line,
    spread_line, total, margin, home_score, away_score.
    """
    for name, df in (("pooled", pooled), ("sealed", sealed)):
        need = {"game_id", "pred_home", "pred_away", "home_score",
                "away_score"}
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise RuntimeError(f"build_market_frame: {name} missing {missing}")

    meta = feats[["game_id", "season"]].copy()
    out: dict[str, pd.DataFrame] = {}
    for name, df in (("pooled", pooled), ("sealed", sealed)):
        m = df.drop(columns=["season", "spread_line", "total_line"],
                    errors="ignore").merge(meta, on="game_id", how="left")
        if m["season"].isna().any():
            raise RuntimeError(
                f"build_market_frame: {name} rows missing from the frame "
                "(frame/era-dump mismatch)")
        m = m.merge(lines, on="game_id", how="left")
        m["total"] = m["home_score"] + m["away_score"]
        m["margin"] = m["home_score"] - m["away_score"]
        m["mu_T_hat"] = m["pred_home"] + m["pred_away"]
        if name == "pooled":
            ws = m["game_id"].map(week_map)
            if ws.isna().any():
                raise RuntimeError(
                    "build_market_frame: pooled rows lack a week_start — "
                    "week_map does not cover the era dump (fold geometry "
                    "mismatch)")
            m["week_start"] = ws
        bad_mu = m["pred_home"].isna() | m["pred_away"].isna()
        bad_line = m["spread_line"].isna() | m["total_line"].isna()
        if bad_mu.any() or bad_line.any():
            raise RuntimeError(
                f"build_market_frame: {name} has rows missing mu "
                f"({int(bad_mu.sum())}) or offered lines ({int(bad_line.sum())})")
        out[name] = m.reset_index(drop=True)
    return out["pooled"], out["sealed"]


# ── Second-level fold-disciplined disagreement walk (Step 1) ────────────────

def fit_fold_disciplined_cd(market_pooled: pd.DataFrame,
                            min_prior_rows: int = MIN_PRIOR_ROWS
                            ) -> dict[str, Any]:
    """Fit (c_k, d_k) per fold over the OOF val weeks; evaluate each week
    with its own coefficients. Warmup folds (< min_prior_rows prior rows)
    get d=1, c=0 (no-shrink).

    For each week w (chronological): OLS on
        y = actual_total - total_line,  x = mu_T_hat - total_line
    over ALL val rows of strictly-prior weeks (weeks < w). The fold's own
    rows are NEVER in its fit set (leak-safety assert below). Returns the
    per-fold table, the median-of-fitted-folds (c, d) for sealed transfer,
    the used (game_id -> (c, d)) assignment, and the leak-safety verdict.
    """
    m = market_pooled.copy()
    if "week_start" not in m.columns:
        raise RuntimeError("fit_fold_disciplined_cd: market needs week_start")
    if m["week_start"].isna().any():
        raise RuntimeError("fit_fold_disciplined_cd: NaN week_start present")
    m = m.sort_values(["week_start", "game_id"]).reset_index(drop=True)
    weeks = sorted(m["week_start"].unique())

    fold_rows: list[dict[str, Any]] = []
    used: dict[str, tuple[float, float]] = {}
    leak_ok = True
    for w in weeks:
        prior = m[m["week_start"] < w]
        cur = m[m["week_start"] == w]
        n_prior = int(len(prior))
        if n_prior >= min_prior_rows:
            if prior["week_start"].max() >= w:  # strict-prior only
                leak_ok = False
            x = (prior["pred_home"] + prior["pred_away"]
                 - prior["total_line"]).to_numpy(float)
            y = (prior["total"] - prior["total_line"]).to_numpy(float)
            A = np.column_stack([np.ones(len(x)), x])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            c, d = float(coef[0]), float(coef[1])
            warmup = False
        else:
            c, d, warmup = 0.0, 1.0, True
        c_r, d_r = round(c, 6), round(d, 6)
        fold_rows.append({
            "week_start": w, "n_val": int(len(cur)), "n_prior": n_prior,
            "c": c_r, "d": d_r, "warmup": bool(warmup),
        })
        for gid in cur["game_id"]:
            used[str(gid)] = (c_r, d_r)

    missing = [g for g in m["game_id"] if str(g) not in used]
    if missing:
        raise RuntimeError(
            f"fit_fold_disciplined_cd: {len(missing)} rows unassigned")

    # Leak-safety assert: every fitted fold's fit set is strictly prior.
    for r in fold_rows:
        if r["warmup"]:
            continue
        prior = m[m["week_start"] < r["week_start"]]
        if len(prior) > 0 and prior["week_start"].max() >= r["week_start"]:
            leak_ok = False
    if not leak_ok:
        raise RuntimeError(
            "fit_fold_disciplined_cd: leak detected — a fold's fit set "
            "touched non-strictly-prior rows; STOP")

    fitted = [r for r in fold_rows if not r["warmup"]]
    median_c = (float(np.median([r["c"] for r in fitted])) if fitted
                else 0.0)
    median_d = (float(np.median([r["d"] for r in fitted])) if fitted
                else 1.0)
    return {
        "folds": fold_rows,
        "median_c": round(median_c, 6),
        "median_d": round(median_d, 6),
        "n_folds": int(len(fold_rows)),
        "n_warmup": int(sum(1 for r in fold_rows if r["warmup"])),
        "n_fitted": int(len(fitted)),
        "leak_safe": bool(leak_ok),
        "min_prior_rows": int(min_prior_rows),
        "used_cd": used,
    }


# ── Delta/2 rebuild (Step 2) ────────────────────────────────────────────────

def shift_means(mu_h: float, mu_a: float, line: float, c: float,
                d: float) -> tuple[float, float]:
    """mu*_T = line + c + d*(mu_T_hat - line); shift both sides by delta/2.

    delta = mu*_T - mu_T_hat. The margin center mu_H - mu_A is UNCHANGED by
    construction (delta/2 cancels) -> P(cover -L) and the spread side are
    untouched.
    """
    mu_t = float(mu_h) + float(mu_a)
    mu_star = float(line) + float(c) + float(d) * (mu_t - float(line))
    delta = mu_star - mu_t
    return mu_h + delta / 2.0, mu_a + delta / 2.0


def fair_total(total_pmf: np.ndarray) -> float:
    """The model's fair total = discrete median of the total PMF (smallest k
    with CDF(k) >= 0.5) — no market blending."""
    cdf = np.cumsum(np.asarray(total_pmf, dtype=float))
    return float(np.searchsorted(cdf, 0.5))


def build_arm(market: pd.DataFrame, params: dict[str, Any], p_tie: float,
              shift_mode: str, cd_by_week: dict[Any, tuple[float, float]]
              | None = None, median_cd: tuple[float, float] | None = None
              ) -> pd.DataFrame:
    """Rebuild per-game joints through the engine's public entrypoints.

    shift_mode:
      "none"   — own-line arm: unshifted era-centered mu (the C0 chain).
      "fold"   — pooled shrink arm: per-game (c_k, d_k) by the row's week.
      "median" — sealed shrink arm: median-of-fold (c, d) on every row.

    Returns the per-game market frame (one row per game) with the derived
    joints' p_home/p_away/p_tie/derived_ml plus fair_total, P(over) at the
    offered total_line, P(cover) at the offered spread_line, and the actual
    y_over/y_cover/y_home_win outcomes (the honest-ECE basis).
    """
    rows = market.copy()
    if shift_mode == "fold":
        if cd_by_week is None:
            raise ValueError("build_arm: fold mode needs cd_by_week")
        cd = rows["week_start"].map(cd_by_week)
        if cd.isna().any():
            raise RuntimeError("build_arm: some pooled rows lack a fold (c,d)")
        c = np.array([t[0] for t in cd], dtype=float)
        d = np.array([t[1] for t in cd], dtype=float)
    elif shift_mode == "median":
        if median_cd is None:
            raise ValueError("build_arm: median mode needs median_cd")
        c = np.full(len(rows), median_cd[0])
        d = np.full(len(rows), median_cd[1])
    else:  # "none"
        c = np.zeros(len(rows))
        d = np.ones(len(rows))

    mu_t = rows["pred_home"].to_numpy(float) + rows["pred_away"].to_numpy(float)
    line = rows["total_line"].to_numpy(float)
    mu_star = line + c + d * (mu_t - line)
    delta = mu_star - mu_t
    frame = pd.DataFrame({
        "game_id": rows["game_id"].values,
        "pred_home": rows["pred_home"].to_numpy(float) + delta / 2.0,
        "pred_away": rows["pred_away"].to_numpy(float) + delta / 2.0,
        "home_score": rows["home_score"].to_numpy(float),
        "away_score": rows["away_score"].to_numpy(float),
    })
    pmfs, summ = je.build_joint_pmfs(frame, params, p_tie)
    derived = summ["derived"].copy()
    tot_pmfs = [je.total_pmf_from_joint(J) for J in pmfs]
    mar_pmfs = [je.margin_pmf_from_joint(J) for J in pmfs]

    merge_cols = [c for c in ("game_id", "season", "week_start",
                              "total_line", "spread_line", "total",
                              "margin", "mu_T_hat", "home_score",
                              "away_score") if c in rows.columns]
    out = derived.merge(rows[merge_cols], on="game_id", how="left")
    if len(out) != len(rows):
        raise RuntimeError("build_arm: derived/game_id merge lost rows")
    out["fair_total"] = [fair_total(t) for t in tot_pmfs]
    out["p_over"] = [je.over_prob(t, float(U))
                     for t, U in zip(tot_pmfs, out["total_line"])]
    out["p_cover"] = [je.cover_prob(m, float(L))
                      for m, L in zip(mar_pmfs, out["spread_line"])]
    out["y_over"] = (out["total"] > out["total_line"]).astype(float)
    out["y_cover"] = (out["margin"] > out["spread_line"]).astype(float)
    out["y_home_win"] = (out["home_score"] > out["away_score"]).astype(float)
    out["used_c"] = np.round(c, 6)
    out["used_d"] = np.round(d, 6)
    out["is_warmup"] = (np.abs(d - 1.0) < 1e-12) & (np.abs(c) < 1e-12)
    return out


# ── Calibration tables ──────────────────────────────────────────────────────

def reliability_table(y: np.ndarray, p: np.ndarray,
                      n_bins: int = 10) -> dict[str, Any]:
    """Decile reliability table + ECE (mirror of run_nfl_joint._reliability —
    the exact binning the re-baseline seam numbers were quoted with, so the
    C0 own-line arm reproduces them bit-for-bit)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 20:
        return {"n": int(len(y)), "ece": None, "bins": []}
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0 + 1e-12
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        bins.append({
            "bin": f"[{lo:.2f},{hi:.2f})", "n": int(m.sum()),
            "pred_mean": round(float(p[m].mean()), 4),
            "actual_rate": round(float(y[m].mean()), 4),
        })
    ece = float(np.mean([abs(b["pred_mean"] - b["actual_rate"])
                         for b in bins]))
    return {"n": int(len(y)), "ece": round(ece, 4), "bins": bins}


def totals_calibration(arm: pd.DataFrame, p_col: str = "p_over",
                       y_col: str = "y_over") -> dict[str, Any]:
    """Totals reliability at each game's offered line + top bin + per-line-bin
    calibration."""
    sub = arm.dropna(subset=["total_line"])
    rel = reliability_table(sub[y_col].to_numpy(float),
                            sub[p_col].to_numpy(float))
    top_bin = rel["bins"][-1] if rel["bins"] else {}
    return {
        "n": int(len(sub)),
        "ece": rel["ece"],
        "bins": rel["bins"],
        "top_bin": top_bin,
        "top_bin_gap": round(abs(top_bin.get("pred_mean", 0.0)
                                 - top_bin.get("actual_rate", 0.0)), 4)
        if top_bin else None,
        "line_bins": line_bin_calibration(sub, p_col, y_col),
        "empirical_over_rate": round(float(sub[y_col].mean()), 4),
    }


def covers_calibration(arm: pd.DataFrame, p_col: str = "p_cover",
                       y_col: str = "y_cover") -> dict[str, Any]:
    sub = arm.dropna(subset=["spread_line"])
    rel = reliability_table(sub[y_col].to_numpy(float),
                            sub[p_col].to_numpy(float))
    return {"n": int(len(sub)), "ece": rel["ece"], "bins": rel["bins"],
            "empirical_ats_rate": round(float(sub[y_col].mean()), 4)}


def line_bin_calibration(sub: pd.DataFrame, p_col: str, y_col: str,
                         n_bins: int = 4) -> list[dict[str, Any]]:
    """Per-line-bin calibration: quantile bins of the offered total_line."""
    lines = sub["total_line"].to_numpy(float)
    if len(lines) < n_bins * 2:
        return []
    edges = np.quantile(lines, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = edges[0] - 1e-9, edges[-1] + 1e-9
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lines >= lo) & (lines < hi)
        if m.sum() == 0:
            continue
        bins.append({
            "line_bin": f"[{lo:.1f},{hi:.1f})", "n": int(m.sum()),
            "mean_line": round(float(lines[m].mean()), 2),
            "pred_mean": round(float(sub.loc[m, p_col].mean()), 4),
            "actual_rate": round(float(sub.loc[m, y_col].mean()), 4),
        })
    return bins


def quote_the_line_ece(y_over: np.ndarray) -> float:
    """Quote-the-line baseline: P(over) = 0.5 for every game -> ECE is the
    single-bin |0.5 - observed over rate| (no calibration possible)."""
    y = np.asarray(y_over, dtype=float)
    return round(abs(0.5 - float(y.mean())), 4)


# ── Per-game market record emitter ──────────────────────────────────────────

def market_record_table(own: pd.DataFrame, shrunk: pd.DataFrame
                        ) -> pd.DataFrame:
    """The per-game market record (honest ECE columns): game identity, both
    arms' fair totals / P(over) / P(cover) / raw derived ML, the actual
    outcomes, and the used (c, d) per row (G5 audit trail)."""
    if len(own) != len(shrunk):
        raise RuntimeError("market_record_table: arm row counts differ")
    o = own.rename(columns={
        "fair_total": "fair_total_own", "p_over": "p_over_own",
        "p_cover": "p_cover_own", "derived_ml": "derived_ml_own"})
    s = shrunk.rename(columns={
        "fair_total": "fair_total_shrunk", "p_over": "p_over_shrunk",
        "p_cover": "p_cover_shrunk", "derived_ml": "derived_ml_shrunk"})
    keep_o = ["game_id", "fair_total_own", "p_over_own", "p_cover_own",
              "derived_ml_own"]
    keep_s = ["game_id", "fair_total_shrunk", "p_over_shrunk",
              "p_cover_shrunk", "derived_ml_shrunk"]
    base_cols = [c for c in ("game_id", "season", "week_start",
                             "total_line", "spread_line", "total",
                             "margin", "mu_T_hat", "y_over", "y_cover",
                             "y_home_win") if c in own.columns]
    base = own[base_cols].copy()
    out = (base.merge(o[keep_o], on="game_id", how="left")
               .merge(s[keep_s], on="game_id", how="left"))
    # The used (c, d) audit trail is the SHRUNK arm's per-row coefficients
    # (what was actually applied) — the own arm's are trivially (0, 1).
    out["used_c"] = shrunk["used_c"].to_numpy()
    out["used_d"] = shrunk["used_d"].to_numpy()
    out["is_warmup"] = shrunk["is_warmup"].to_numpy()
    if out[["p_over_own", "p_over_shrunk"]].isna().any().any():
        raise RuntimeError("market_record_table: merge lost rows")
    if "week_start" not in out.columns:
        out["week_start"] = pd.NaT  # sealed rows carry no fold week
    return out[MARKET_RECORD_COLUMNS].reset_index(drop=True)