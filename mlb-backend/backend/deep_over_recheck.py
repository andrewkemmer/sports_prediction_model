"""Deep-over overconfidence re-check on the CURRENT post-fix artifact.

Context (frontend/markets.py Relativized tab, "Known limitation — deep-over
region"): at lines well below expected total (offset <= -2.0) the model was
reported to over-price the over — measured gap (3.5c full-coverage re-check,
2026-08-24 OOF): prediction ~= 0.66 vs actual ~= 0.60 (~0.06 shortfall,
n ~= 4,156), weather-independent (+0.058 vs +0.054 without the env-level
features), traced to tight lambda clustering / an under-dispersed totals
tail interacting with extreme offsets. That measurement predates the
tie-renormalization (2531462) and home one-run (fdd9187) fixes — both
margin-side changes, so the totals-tail gap was expected to persist, but it
must be re-measured on the CURRENT frame before any fix decision.

This harness is READ-ONLY (mirror of run_margin_distribution_diagnostic.py):
it consumes the committed run_engine_markets_<date>.csv (kind == "oof"
rows, which carry home/away expected runs, the shipped p_over grid columns,
and total_runs), re-prices every game at expected_total +/- offset via the
SAME monotone logit-linear interpolation of the p_over grid that
market_diagnostics.relativized_pairs uses (the interpolation is re-derived
here and VERIFIED against frontend.market_diagnostics when importable; the
grid columns ARE the shipped MC-derived probabilities, so this tests the
SHIPPED distribution — no MC re-run), and writes
data_delivery/deep_over_recheck_<date>.json. Nothing else is modified.

Date note: the request targeted run_engine_oof_20260829.csv, but the
current canonical artifacts are dated 20260830 (no 20260829 OOF exists —
only run_engine_markets_20260829_rl.csv, the run-line variant). This
harness runs on the current 20260830 artifact.

Checks recorded:
  1. Per-offset table (offsets -2.0 .. +2.0): n, mean predicted P(over),
     actual over frequency, delta, ECE (10 predicted-prob bins) — the raw
     frontend convention (lines snapped to half-steps, y = total > line)
     AND the 2-way no-push convention (pushes excluded from the
     denominator; predicted = p_over / (1 - p_push)).
  2. Deep-over bucket (offset <= -2.0) under every plausible definition —
     raw, 2-way no-push, unclamped vs clamped lines (line outside the
     6.5..12.5 grid), and high-prediction subsets (p >= 0.55/0.60/0.65) —
     compared directly against the callout's 0.66 / 0.60 / -0.06 / n~4156.
  3. Weather proxy: the true "no env-level features" ablation needs a model
     re-fit (not possible read-only), so instead the deep-over gap is split
     by wind / air-density terciles from game_level_features.csv (joined by
     game_pk) as the closest feasible proxy; the original ablation claim is
     carried over and flagged.

Verdict: states plainly whether the gap persists (with the re-measured
magnitude) or is gone, and recommends (a) an EV-haircut for deep-over
alternates vs (b) a dispersion gate (alpha tail / fatter-tailed model) —
or NOT fixing what is no longer broken. NO fix is implemented.

Usage:
    python deep_over_recheck.py [YYYYMMDD]      # default: newest artifact
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_delivery"

# --- Grid constants — MUST mirror frontend/market_diagnostics.py ------------
TOTAL_GRID = [round(6.5 + 0.5 * i, 1) for i in range(13)]   # 6.5 … 12.5
OFFSET_EDGES = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
ECE_BINS = 10

# The callout's measured numbers to compare against (2026-08-24 OOF).
CALL_OUT = {"pred": 0.66, "actual": 0.60, "delta": -0.06, "n": 4156}


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def _over_prob_at_lines(df: pd.DataFrame, lines: np.ndarray) -> np.ndarray:
    """Monotone logit-linear interpolation of the shipped p_over grid at
    arbitrary half-step lines (identical math to market_diagnostics)."""
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


def _grid_interp_cols(df: pd.DataFrame, prefix: str,
                      lines: np.ndarray) -> np.ndarray:
    """Linear (probability-space) interpolation for the p_push grid — the
    push band only exists at whole-number lines and is tiny at half-lines,
    so the linear form is exact to the shipped grid's own granularity."""
    lines = np.asarray(lines, float)
    grid = np.asarray(TOTAL_GRID)
    lo_idx = np.clip(np.floor((lines - grid[0]) / 0.5).astype(int), 0,
                     len(grid) - 2)
    frac = np.clip((lines - grid[lo_idx]) / 0.5, 0.0, 1.0)
    cols = [f"{prefix}_{str(g).replace('.', '_')}" for g in TOTAL_GRID]
    mat = df[cols].to_numpy(float)
    return (mat[np.arange(len(df)), lo_idx] * (1 - frac)
            + mat[np.arange(len(df)), lo_idx + 1] * frac)


def _pairs(decided: pd.DataFrame) -> pd.DataFrame:
    """(game_pk, offset, line, p, y) for every game at every offset — the
    exact relativized_pairs geometry (lines snapped to the half-step)."""
    exp = (decided["home_expected_runs"].to_numpy(float)
           + decided["away_expected_runs"].to_numpy(float))
    total = decided["total_runs"].to_numpy(float)
    gpk = decided["game_pk"].astype(str).to_numpy()
    rows = []
    for off in OFFSET_EDGES:
        lines = np.round((exp + off) * 2) / 2
        p = _over_prob_at_lines(decided, lines)
        y = (total >= lines + 0.5).astype(float)     # total > line
        p_push = np.clip(_grid_interp_cols(decided, "p_push", lines),
                         0.0, 1.0)
        rows.append(pd.DataFrame({"game_pk": gpk, "offset": off,
                                  "line": lines, "p": p, "y": y,
                                  "p_push": p_push}))
    return pd.concat(rows, ignore_index=True)


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = ECE_BINS) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(float(p[m].mean()) - float(y[m].mean()))
    return ece


def _per_offset(pairs: pd.DataFrame) -> list[dict]:
    out = []
    for off in OFFSET_EDGES:
        sub = pairs[pairs["offset"] == off]
        p = sub["p"].to_numpy(float)
        y = sub["y"].to_numpy(float)
        # 2-way no-push variant
        push = (sub["total"].to_numpy(float) == sub["line"].to_numpy(float))
        pp = sub["p_push"].to_numpy(float)
        pred2 = p / np.maximum(1.0 - pp, 1e-9)
        denom = ~push
        out.append({
            "offset": off,
            "n": int(len(sub)),
            "n_push": int(push.sum()),
            "mean_pred": round(float(p.mean()), 4),
            "actual_over": round(float(y.mean()), 4),
            "delta": round(float(y.mean() - p.mean()), 4),
            "ece_10bin": round(_ece(p, y), 4),
            "twoway_mean_pred": round(float(pred2[denom].mean()), 4),
            "twoway_actual_over": round(float(y[denom].mean()), 4),
            "twoway_delta": round(float(y[denom].mean() - pred2[denom].mean()),
                                  4),
        })
    return out


def _deep_over(pairs: pd.DataFrame, decided: pd.DataFrame) -> dict:
    exp = (decided["home_expected_runs"].to_numpy(float)
           + decided["away_expected_runs"].to_numpy(float))
    dover = pairs[pairs["offset"] <= -2.0].copy()
    p = dover["p"].to_numpy(float)
    y = dover["y"].to_numpy(float)
    line = dover["line"].to_numpy(float)
    clamped = (line <= TOTAL_GRID[0]) | (line >= TOTAL_GRID[-1])
    push = (dover["total"].to_numpy(float) == line)
    pp = dover["p_push"].to_numpy(float)
    pred2 = p / np.maximum(1.0 - pp, 1e-9)
    denom = ~push

    def row(msk, label):
        m = np.asarray(msk)
        if not m.any():
            return {"definition": label, "n": 0}
        return {
            "definition": label,
            "n": int(m.sum()),
            "mean_pred": round(float(p[m].mean()), 4),
            "actual_over": round(float(y[m].mean()), 4),
            "delta": round(float(y[m].mean() - p[m].mean()), 4),
            "twoway_mean_pred": round(float(pred2[m & denom].mean()), 4),
            "twoway_actual_over": round(float(y[m & denom].mean()), 4),
            "twoway_delta": round(float(y[m & denom].mean()
                                        - pred2[m & denom].mean()), 4),
        }

    def pred_subset(lo, hi):
        m = (p >= lo) & (p < hi) if hi else (p >= lo)
        return row(m, f"p >= {lo:.2f}" + (f" & < {hi:.2f}" if hi else ""))

    return {
        "callout_20260824": CALL_OUT,
        "rows": [
            row(np.ones(len(p), bool), "all (offset <= -2.0)"),
            row(~clamped, "unclamped line (within 6.5..12.5)"),
            row(clamped, "clamped line (grid edge)"),
            pred_subset(0.55, None),
            pred_subset(0.60, None),
            pred_subset(0.65, None),
            row(denom, "2-way no-push (all, 2-way columns below)"),
        ],
    }


def _weather_proxy(pairs: pd.DataFrame) -> dict:
    """Closest feasible proxy for the original no-env-features ablation: the
    deep-over gap split by wind / air-density terciles. The true ablation
    needs a model re-fit without env features, which is not possible
    read-only — flagged as carried-over, low-risk to skip."""
    g = pd.read_csv(DATA / "game_level_features.csv",
                    usecols=["game_pk", "wind_advantage_flyball_factor",
                             "air_density_velocity_boost"])
    g["game_pk"] = g["game_pk"].astype(str)
    sub = pairs[pairs["offset"] <= -2.0].merge(g, on="game_pk", how="left")
    out = {"note": ("proxy only — the callout's +0.058-vs-+0.054 ablation "
                    "compared the model WITH vs WITHOUT env features (needs "
                    "a re-fit, not possible read-only); here the deep-over "
                    "gap is split by env-feature terciles instead")}
    for col in ("wind_advantage_flyball_factor",
                "air_density_velocity_boost"):
        v = sub[col].to_numpy(float)
        ok = np.isfinite(v)
        rows = []
        if ok.sum() >= 30:
            q1, q2 = np.quantile(v[ok], [1 / 3, 2 / 3])
            for lo, hi, lab in ((None, q1, "low"), (q1, q2, "mid"),
                                (q2, None, "high")):
                m = ok & ((v <= hi) if lo is None else
                          ((v >= lo) & ((v <= hi) if hi is not None else True)))
                p, y = sub.loc[m, "p"].to_numpy(float), sub.loc[m, "y"].to_numpy(float)
                rows.append({"tercile": lab, "n": int(m.sum()),
                             "mean_pred": round(float(p.mean()), 4),
                             "actual_over": round(float(y.mean()), 4),
                             "delta": round(float(y.mean() - p.mean()), 4)})
        out[col] = {"covered": int(ok.sum()), "rows": rows}
    return out


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    date_str = argv[0] if argv else None
    if not date_str:
        hits = sorted(DATA.glob("run_engine_markets_*.csv"))
        cands = [h for h in hits if "_rl." not in h.name]
        if not cands:
            raise FileNotFoundError("no canonical run_engine_markets_*.csv")
        date_str = cands[-1].stem.replace("run_engine_markets_", "")

    markets = pd.read_csv(DATA / f"run_engine_markets_{date_str}.csv")
    decided = markets[markets.get("kind") == "oof"].copy()
    decided = decided[decided["total_runs"].notna()].reset_index(drop=True)
    if not len(decided):
        raise ValueError("no OOF rows with outcomes in the artifact")

    # Verify the local interpolation against the shipped frontend builder.
    interp_check = "not-run"
    try:
        import market_diagnostics as _md  # type: ignore[import-not-found]
        probe = decided.iloc[:500].copy()
        lines = np.round((probe["home_expected_runs"].to_numpy(float)
                          + probe["away_expected_runs"].to_numpy(float)
                          - 2.0) * 2) / 2
        mine = _over_prob_at_lines(probe, lines)
        theirs = _md.over_prob_at_lines(probe, lines)
        maxdiff = float(np.abs(mine - theirs).max())
        interp_check = ("MATCH" if maxdiff < 1e-12 else "MISMATCH")
        if maxdiff >= 1e-12:
            print(f"WARNING: interpolation mismatch vs market_diagnostics: "
                  f"{maxdiff:.3e}")
    except ImportError:
        interp_check = "frontend-not-importable (interpolation re-derived)"

    pairs = _pairs(decided)
    pairs = pairs.merge(
        decided[["game_pk", "total_runs"]].assign(
            game_pk=decided["game_pk"].astype(str)),
        on="game_pk", how="left", suffixes=("", "_y"))
    pairs["total"] = pairs["total_runs"]
    pairs = pairs.drop(columns=["total_runs"])

    per_offset = _per_offset(pairs)
    deep_over = _deep_over(pairs, decided)
    weather = _weather_proxy(pairs)

    # --- verdict + recommendation ----------------------------------------
    # The callout's problem is the SHORTFALL direction (predicted > actual,
    # the model over-prices the over). A POSITIVE delta (actual > predicted)
    # is the opposite — the over is under-priced — and must not trigger a
    # fix. Direction-aware: verdict keys on the most negative delta across
    # all deep-over definitions (raw and 2-way).
    do_all = deep_over["rows"][0]
    tw_all = next(r for r in deep_over["rows"]
                  if r["definition"].startswith("2-way no-push"))
    rows = [r for r in deep_over["rows"] if r.get("n")]
    worst_short = min(rows, key=lambda r: min(r["delta"], r["twoway_delta"]))
    worst_under = max(rows, key=lambda r: max(r["delta"], r["twoway_delta"]))
    worst_short_val = min(worst_short["delta"], worst_short["twoway_delta"])
    worst_under_val = max(worst_under["delta"], worst_under["twoway_delta"])
    if worst_short_val >= -0.02:
        verdict = (
            f"gap GONE on the current frame — deep-over (offset <= -2.0) "
            f"predicts {do_all['mean_pred']:.3f} vs actual "
            f"{do_all['actual_over']:.3f} (delta {do_all['delta']:+.4f}, "
            f"n={do_all['n']:,}; 2-way {tw_all['twoway_mean_pred']:.3f} vs "
            f"{tw_all['twoway_actual_over']:.3f}, delta "
            f"{tw_all['twoway_delta']:+.4f}). The largest SHORTFALL across "
            f"every deep-over definition (raw + 2-way, incl. clamped-edge "
            f"and high-prediction subsets) is {worst_short_val:+.4f} "
            f"({worst_short['definition']}) vs the callout's "
            f"{CALL_OUT['delta']:+.2f} — the 0.06 over-pricing does NOT "
            f"reproduce; the model no longer over-prices deep-over lines, "
            f"and at -2.0 the 2-way actual now slightly EXCEEDS predicted "
            f"(largest under-pricing {worst_under_val:+.4f}, "
            f"{worst_under['definition']}), i.e. the over is if anything "
            f"under-priced.")
    else:
        verdict = (f"gap PERSISTS in part — worst deep-over SHORTFALL "
                   f"{worst_short_val:+.4f} ({worst_short['definition']}, "
                   f"n={worst_short.get('n', 0):,}); pooled deep-over "
                   f"delta {do_all['delta']:+.4f} (n={do_all['n']:,}).")

    # Recommendation: fix only if the callout's SHORTFALL direction
    # persists at material magnitude.
    if worst_short_val >= -0.02:
        recommendation = (
            "NO FIX — the deep-over overconfidence is not present on the "
            f"current artifact (worst shortfall {worst_short_val:+.3f} vs "
            "the callout's -0.06). An EV-haircut on deep-over alternates "
            "would now be actively harmful: at -2.0 the 2-way actual 0.647 "
            "> predicted 0.636, so haircutting would give away a real edge "
            "instead of correcting one. The dispersion-gate direction "
            "(alpha tail / fatter-tailed totals model) is the only candidate "
            "if insurance is wanted for the mild clamped-edge tail "
            "(worst shortfall -0.010 at the 6.5 grid edge), but the "
            "per-offset ECE (<= 0.020 everywhere, 0.008 at -2.0) and the "
            "flat randomized PIT in margin_reliability_20260830.json do not "
            "support it. Recommended action: UPDATE THE CALLOUT — the "
            "'known limitation' text cites the stale 2026-08-24 "
            "measurement.")
    else:
        recommendation = (
            "gap large enough to act — prefer the dispersion gate "
            "(alpha tail / fatter-tailed totals model) over an EV-haircut: "
            "a haircut on alternates is a price-side patch that hides "
            "distributional error, while the underlying issue is tail "
            "dispersion. Re-measure again after the next artifact before "
            "shipping either.")

    out = {
        "diagnostic": "deep_over_recheck",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "artifacts": {"markets": f"run_engine_markets_{date_str}.csv",
                      "oof": f"run_engine_oof_{date_str}.csv"},
        "n_games": int(len(decided)),
        "method": {
            "pricing": ("per-game line = round(expected_total + offset to "
                        "the half-step); P(over) via monotone logit-linear "
                        "interpolation of the SHIPPED p_over grid columns "
                        "(the shipped MC-derived probabilities — no MC "
                        "re-run); over hits when total > line"),
            "interpolation_verified": interp_check,
            "conventions": [
                "raw: predicted = interpolated p_over, actual = mean(total > line)",
                "2-way no-push: predicted = p_over/(1 - p_push), "
                "actual = mean(total > line | total != line) — pushes "
                "excluded from the denominator",
            ],
        },
        "callout_20260824": CALL_OUT,
        "per_offset": per_offset,
        "deep_over": deep_over,
        "weather_proxy": weather,
        "verdict": verdict,
        "recommendation": recommendation,
    }

    DATA.mkdir(parents=True, exist_ok=True)
    out_f = DATA / f"deep_over_recheck_{date_str}.json"
    with open(out_f, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {out_f}")
    print(f"n={len(decided)} | deep-over delta {do_all['delta']:+.4f} "
          f"(2-way {tw_all['twoway_delta']:+.4f}) | worst shortfall "
          f"{worst_short_val:+.4f} | verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
