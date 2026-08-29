"""Run-line CALIBRATION-BY-LINE diagnostic — accuracy per run-line margin.

Sibling of run_totals_line_calibration.py, gating the per-card run-line
selector: for each run-line margin L in the full grid (half-lines 1.5,
2.5, 3.5 + whole alternates 1, 2, 3, 4) compute, from run_engine_oof
joined to the per-line p_rl_* columns:

    n            games priced at L
    p_home       mean predicted P(home covers −L)  (strict: margin > L)
    actual       actual home-cover rate  (margin > L on real scores)
    delta        p_home − actual
    ece          binned reliability ECE (10 equal-width bins, min 30)
    push_pred    mean predicted P(push) (0 for half-lines by construction)
    push_actual  empirical push rate (margin == L; 0 by construction for
                 half-lines — integer margins can't equal a half-line)
    verdict      calibrated / over-predicting / under-predicting / low_n

Whole lines store the 3-way split in the artifact (home/push/away sum to
1.0 from the MC draws); the gate discipline matches the totals harness
(|delta| <= 0.02 → calibrated; n < 200 → low_n/provisional).

The gate's verdict is what decides whether an alternate line may be
offered in the Today's Games run-line selector: lines the gate marks
"calibrated" are offerable; anything else renders "unverified" on the
card (never fabricated).

Usage:
    python run_rl_line_calibration.py [YYYYMMDD_rl|YYYYMMDD]
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from run_engine import RUN_LINE_GRID_FULL, rl_col

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_delivery"

DELTA_THRESHOLD = 0.02     # verdict threshold (|predicted − actual|)
MIN_N_FOR_CONFIDENT = 200  # below → verdict flagged low_n
ECE_BINS = 10
ECE_MIN_COUNT = 30


def line_ece(p: np.ndarray, y: np.ndarray) -> tuple[float | None, list, int]:
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
        mp, ma = float(p[m].mean()), float(y[m].mean())
        bins.append({"bin_center": round(float((edges[b] + edges[b + 1]) / 2), 3),
                     "mean_pred": round(mp, 4), "mean_actual": round(ma, 4),
                     "count": n})
        ece += n * abs(mp - ma)
        total += n
    return (round(ece / total, 4) if total else None, bins, dropped)


def verdict(delta: float, n: int) -> str:
    if n < MIN_N_FOR_CONFIDENT:
        return "low_n"
    if delta > DELTA_THRESHOLD:
        return "over-predicting"
    if delta < -DELTA_THRESHOLD:
        return "under-predicting"
    return "calibrated"


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    date_str = argv[0] if argv else None
    if not date_str:
        hits = sorted(DATA.glob("run_engine_markets_*_rl.csv"))
        if not hits:
            raise FileNotFoundError(
                "no run_engine_markets_*_rl.csv (run the rl bridge first)")
        date_str = hits[-1].stem.replace("run_engine_markets_", "")
    f = DATA / f"run_engine_markets_{date_str}.csv"
    if not f.exists():
        raise FileNotFoundError(f)
    m = pd.read_csv(f)
    oof = m[m["kind"] == "oof"].reset_index(drop=True)
    margin = (oof["home_score"].to_numpy(float)
              - oof["away_score"].to_numpy(float))
    rows, reliability = [], {}
    for L in RUN_LINE_GRID_FULL:
        hcol = rl_col(L, "home")
        pcol = rl_col(L, "push")
        if hcol not in oof.columns:
            continue
        p = oof[hcol].to_numpy(float)
        y = (margin > L).astype(float)
        ok = np.isfinite(p) & np.isfinite(y)
        pv, yv = p[ok], y[ok]
        n = int(ok.sum())
        if not n:
            continue
        p_mean = float(pv.mean())
        actual = float(yv.mean())
        ece, bins, dropped = line_ece(pv, yv)
        push_pred = (float(oof[pcol].to_numpy(float)[ok].mean())
                     if pcol in oof.columns else 0.0)
        push_actual = float((margin[ok] == L).mean())
        rows.append({
            "line": L, "n": n, "p_home": round(p_mean, 4),
            "actual_home_cover": round(actual, 4),
            "delta": round(p_mean - actual, 4),
            "ece": ece, "push_pred": round(push_pred, 4),
            "push_actual": round(push_actual, 4),
            "verdict": verdict(p_mean - actual, n),
        })
        reliability[str(L)] = {"bins": bins, "n_dropped_bins": dropped}
    out = {
        "diagnostic": "run_line_calibration",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "artifacts": {"markets": f.name},
        "n_games": int(len(oof)),
        "method": {
            "cover_definition": "strict: home covers −L iff margin > L "
                                "(mirrors totals strict-over discipline)",
            "push_definition": "margin == L (whole lines only); 0 by "
                               "construction for half-lines",
            "tie_handling": "margin distribution conditioned on no tie "
                            "(P(margin=0)=0) — the run-engine tie fix; "
                            "away = P(margin < L | no tie)",
            "ece": f"binned reliability, {ECE_BINS} equal-width bins, min "
                   f"{ECE_MIN_COUNT} games/bin",
            "verdict_threshold": f"|delta| > {DELTA_THRESHOLD} → "
                                 f"over/under-predicting; n < "
                                 f"{MIN_N_FOR_CONFIDENT} → low_n",
        },
        "lines": rows, "reliability": reliability,
    }
    DATA.mkdir(parents=True, exist_ok=True)
    out_f = DATA / f"run_line_calibration_{date.today():%Y%m%d}.json"
    with open(out_f, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {out_f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
