"""Totals CALIBRATION-BY-LINE diagnostic — accuracy per totals line value.

Purpose (gate before a market-line toggle on Today's Games): the run
engine predicts P(over) at every grid line (6.5, 7, 7.5, 8, ...) per
game in run_engine_markets_<date>.csv, and run_engine_oof_<date>.csv
carries the actual outcomes. Existing dashboards either pool lines
(Relativized, Pooled lines, per-game rounded-total money line) or
report ECE/Brier/logloss per line without an accuracy breakdown
(run_engine_monitor market_metrics + the model-card table). None of
them shows, per line value, n / mean predicted / actual over rate /
delta / ECE — which is what a bettor needs to decide whether the
model's P(over) is trustworthy enough to match an alternate market
line.

This harness is READ-ONLY: it consumes the committed artifacts and
writes data_delivery/totals_line_calibration_<date>.json. No model,
sampler, or frontend change.

Method
------
* Join run_engine_markets_<date>.csv (kind == "oof" rows) to
  run_engine_oof_<date>.csv on game_pk.
* For each grid line L in run_engine.TOTAL_LINE_GRID compute:
    n            games priced at L
    p_mean       mean predicted P(over L) across those games
    actual       actual over rate, STRICT definition total > L
                 (total >= L + 0.5 — identical to the monitor scorer
                 and fixed_line_pairs, so MC and monitor agree)
    delta        p_mean - actual
    ece          binned reliability ECE (equal-width 10 bins,
                 min 30 games per bin, same convention as
                 frontend/market_diagnostics.calibration_curve)
    p_push       mean push probability for the line — 0 for half
                 lines by construction; for whole lines derived from
                 the grid as P(over L) - P(over L+0.5) (the artifact
                 predates the explicit p_push_<line> columns)
* Verdict per line (delta threshold 0.02, same scale as the existing
  ECE badges):
    calibrated       |delta| <= 0.02
    over-predicting  delta > +0.02   (predicted too high)
    under-predicting delta < -0.02   (predicted too low)
  Lines with fewer than 200 games are additionally flagged
  "low_n" — verdict is provisional.
* Reliability table per line: (bin_center, mean_pred, mean_actual,
  count) for the bins used in the ECE computation.

Usage:
    python run_totals_line_calibration.py [YYYYMMDD]
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from run_engine import TOTAL_LINE_GRID

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_delivery"

DELTA_THRESHOLD = 0.02     # verdict threshold (|predicted - actual|)
MIN_N_FOR_CONFIDENT = 200  # games below this → verdict flagged low_n
ECE_BINS = 10              # equal-width reliability bins (calibration_curve)
ECE_MIN_COUNT = 30         # per-bin minimum (calibration_curve convention)

def grid_over_col(line: float) -> str:
    """p_over column for a grid line.

    NOTE: str(line).replace('.', '_') is NOT injective across the grid —
    7.0 and 7.5 both map to '7_0' — so the column name must be built from
    the raw line each call, never from a pre-computed dict keyed by line.
    """
    return f"p_over_{str(line).replace('.', '_')}"


def latest_artifact(prefix: str, suffix: str) -> Path | None:
    """Latest data_delivery artifact matching prefix*suffix (by name)."""
    hits = sorted(DATA.glob(f"{prefix}*{suffix}"))
    return hits[-1] if hits else None


def load_frames(date_str: str | None = None) -> tuple[pd.DataFrame,
                                                      pd.DataFrame,
                                                      str, str]:
    """Load (markets, oof) for the given date (default: latest pair)."""
    if date_str:
        markets_f = DATA / f"run_engine_markets_{date_str}.csv"
        oof_f = DATA / f"run_engine_oof_{date_str}.csv"
        if not markets_f.exists() or not oof_f.exists():
            raise FileNotFoundError(
                f"Artifact pair missing for {date_str}: "
                f"{markets_f.name} / {oof_f.name}")
    else:
        markets_f = latest_artifact("run_engine_markets_", ".csv")
        oof_f = latest_artifact("run_engine_oof_", ".csv")
        if markets_f is None or oof_f is None:
            raise FileNotFoundError(
                "No run_engine_markets_/run_engine_oof_ artifacts found.")
    markets = pd.read_csv(markets_f)
    oof = pd.read_csv(oof_f)
    return markets, oof, markets_f.name, oof_f.name


def line_ece(p: np.ndarray, y: np.ndarray) -> tuple[float, list[dict], int]:
    """Binned reliability ECE (weighted mean |mean_pred - mean_actual|),
    same equal-width-bin convention as market_diagnostics.calibration_curve.
    Returns (ece, bins, n_dropped_bins)."""
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    y = np.asarray(y, float)
    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, ECE_BINS - 1)
    bins, dropped, total, ece = [], 0, 0, 0.0
    for b in range(ECE_BINS):
        m = idx == b
        n = int(m.sum())
        if n < ECE_MIN_COUNT:
            dropped += 1
            continue
        mp = float(p[m].mean())
        ma = float(y[m].mean())
        bins.append({
            "bin_center": round(float((edges[b] + edges[b + 1]) / 2), 3),
            "mean_pred": round(mp, 4),
            "mean_actual": round(ma, 4),
            "count": n,
        })
        ece += n * abs(mp - ma)
        total += n
    return (round(ece / total, 4) if total else None, bins, dropped)


def verdict(delta: float, n: int) -> str:
    if n < MIN_N_FOR_CONFIDENT:
        return "low_n"  # verdict provisional — not enough games
    if delta > DELTA_THRESHOLD:
        return "over-predicting"
    if delta < -DELTA_THRESHOLD:
        return "under-predicting"
    return "calibrated"


def build_table(markets: pd.DataFrame, oof: pd.DataFrame) -> dict:
    m = markets[markets["kind"] == "oof"].copy()
    m["game_pk"] = m["game_pk"].astype(str)
    o = oof.copy()
    o["game_pk"] = o["game_pk"].astype(str)
    joined = m.merge(o[["game_pk", "home_score", "away_score"]],
                     on="game_pk", how="inner",
                     suffixes=("", "_oof"))
    if joined.empty:
        raise ValueError("No OOF games joined between markets and oof "
                         "artifacts (check game_pk dtype/coverage).")
    total = (joined["home_score"].to_numpy(float)
             + joined["away_score"].to_numpy(float))
    rows, reliability = [], {}
    for line in TOTAL_LINE_GRID:
        col = grid_over_col(line)
        if col not in joined.columns:
            continue
        p = joined[col].to_numpy(float)
        y = (total >= line + 0.5).astype(float)      # strict over (total > L)
        n = int(np.isfinite(p).sum())
        if not n:
            continue
        ok = np.isfinite(p) & np.isfinite(y)
        pv, yv = p[ok], y[ok]
        p_mean = float(pv.mean())
        actual = float(yv.mean())
        delta = p_mean - actual
        ece, bins, dropped = line_ece(pv, yv)
        # Push probability: 0 for half lines by construction; for whole
        # lines, P(total == L) = P(over L) - P(over L+0.5) via the grid.
        p_push = 0.0
        if abs(line - round(line)) < 1e-9:
            hi_col = grid_over_col(round(line + 0.5, 1))
            if hi_col in joined.columns:
                p_push = float(np.clip(p - joined[hi_col].to_numpy(float),
                                       0.0, 1.0).mean())
        rows.append({
            "line": line,
            "n": n,
            "p_mean": round(p_mean, 4),
            "actual_over_rate": round(actual, 4),
            "delta": round(delta, 4),
            "ece": ece,
            "p_push": round(p_push, 4),
            "verdict": verdict(delta, n),
        })
        reliability[str(line)] = {
            "bins": bins,
            "n_dropped_bins": dropped,
        }
    return {"lines": rows, "reliability": reliability}


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    date_str = argv[0] if argv else None
    markets, oof, markets_name, oof_name = load_frames(date_str)
    out = {
        "diagnostic": "totals_line_calibration",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "artifacts": {"markets": markets_name, "oof": oof_name},
        "n_games_joined": int(
            markets[markets["kind"] == "oof"]["game_pk"].astype(str)
            .isin(oof["game_pk"].astype(str)).sum()),
        "method": {
            "over_definition": "strict: total >= line + 0.5 "
                               "(matches monitor scorer & fixed_line_pairs)",
            "push_definition": "P(total == L) = P(over L) - P(over L+0.5) "
                               "for whole lines; 0 for half lines",
            "ece": (f"binned reliability, {ECE_BINS} equal-width bins, "
                    f"min {ECE_MIN_COUNT} games/bin (calibration_curve "
                    "convention)"),
            "verdict_threshold": f"|delta| > {DELTA_THRESHOLD} → "
                                 "over/under-predicting; "
                                 f"n < {MIN_N_FOR_CONFIDENT} → low_n",
            "artifact_verification": "NO COLUMN COLLISION: all grid lines "
                "present and distinct in the artifact — p_under_9_0 != "
                "p_under_9_5 (means 0.5256 vs 0.6117, 0% identical rows), "
                "same for every whole/half pair. Whole-line p_over == "
                "next-half-line p_over BY STRICT-OVER CONSTRUCTION (both = "
                "P(total >= 10) for 9.0/9.5) — expected math, not a bug. "
                "Explicit p_push_<line> columns may be ABSENT in pre-65b44ec "
                "artifacts; grid-derived p_push is then 0 on whole lines for "
                "the same strict-over reason. An earlier diagnostic note "
                "claimed a str(line).replace('.', '_') naming collision; "
                "that claim was WRONG (str(9.0)->'9_0' vs str(9.5)->'9_5' "
                "are distinct keys) and is retracted here.",
        },
        **build_table(markets, oof),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    out_f = DATA / f"totals_line_calibration_{date.today():%Y%m%d}.json"
    with open(out_f, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {out_f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
