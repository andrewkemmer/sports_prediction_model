"""Why the 20261017 dashboard shows almost no spread improvement (2026-10-17).

The delivered meta carries the admitted serving fit (one_run_rebalance, z
3.268, delta_source one_run_rebalance), but the Distribution chart pools the
PMFs each ORIGIN shipped — and the mechanism can only move those where its
own held-out gate admits (and only where the walk-off share admitted first).
This probe extracts the 20261017 artifact from origin/main, re-runs the causal
walk once, records EVERY fit_one_run_spread_rebalance call (share present,
amount, held-out z, refusal reason), and prints:

  * admissions at the generic bar (3.0) vs the pre-registered one-sided
    structural bar (2.5) — i.e. how much of the shortfall is gate-stage,
  * the pooled chart numbers (observed vs shipped model at -2/-1/+1/+2/+3 and
    the one-run split), before/after,
  * the same chart numbers on the previous (20261016) artifact for comparison.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlb-backend" / "backend"))
import distributions as dist          # noqa: E402
import gaussian_family as family      # noqa: E402

TMP = Path(tempfile.gettempdir()) / "mlb17"
ARTIFACT = "run_engine_markets_20261017.csv"


def extract() -> Path:
    TMP.mkdir(parents=True, exist_ok=True)
    path = TMP / ARTIFACT
    if not path.exists():
        blob = subprocess.run(
            ["git", "show", f"origin/main:mlb-backend/data_delivery/{ARTIFACT}"],
            cwd=ROOT, capture_output=True, check=True).stdout
        path.write_bytes(blob)
        print("extracted", path, len(blob) // 1024, "KiB", flush=True)
    return path


def chart(priced: pd.DataFrame, label: str) -> None:
    d = priced.dropna(subset=["home_score", "away_score"]).copy()
    d["m"] = (d.home_score - d.away_score).astype(int)
    print(f"\n--- {label}: pooled chart numbers (n={len(d)}) ---")
    print(f"{'bin':>4} {'actual':>8} {'shipped':>8} {'gap':>9}")
    for v in (-3, -2, -1, 1, 2, 3, 4):
        col = f"p_rl_{v}_0_push" if v > 0 else f"p_rl_{-v}_0_away_push"
        if col not in d:
            continue
        o, m = float((d.m == v).mean()), float(d[col].mean())
        print(f"{v:>4} {o:>8.4f} {m:>8.4f} {o - m:>+9.4f}")
    one = d[d.m.abs() == 1]
    split = float((one.m > 0).mean())
    p1 = float(d["p_rl_1_0_push"].mean())
    m1 = float(d["p_rl_1_0_away_push"].mean())
    print(f"one-run split: observed {split:.4f} / shipped {p1 / (p1 + m1):.4f}"
          f"  (n={len(one)})")


def main() -> None:
    path = extract()
    frame = pd.read_csv(path, float_precision="round_trip", low_memory=False)
    oof = frame[frame["kind"] == "oof"].copy()

    calls = []
    real = dist.fit_one_run_spread_rebalance

    def hook(*a, **k):
        delta, rep = real(*a, **k)
        calls.append({"share": rep.get("share"), "z": rep.get("z"),
                      "reason": rep.get("reason"),
                      "amount": rep.get("amount"),
                      "secondary_residual": rep.get("secondary_residual"),
                      "secondary_capacity": rep.get("secondary_capacity"),
                      "secondary_amount": rep.get("secondary_amount"),
                      "admitted": bool(rep.get("significant"))})
        return delta, rep

    dist.fit_one_run_spread_rebalance = hook
    t0 = time.time()
    priced, bundle = dist.calibrate_market_frame(oof)
    dist.fit_one_run_spread_rebalance = real
    print("walk_seconds:", round(time.time() - t0, 1), flush=True)

    print("\n--- per-origin rebalance calls (final serving fit is the last) ---")
    share_present = [c for c in calls if c["share"] is not None]
    print("calls:", len(calls), "| with an admitted share:",
          len(share_present), "| admitted:", sum(c["admitted"] for c in calls))
    reasons = {}
    for c in calls:
        if not c["admitted"]:
            reasons[c["reason"]] = reasons.get(c["reason"], 0) + 1
    print("refusal reasons:", reasons)
    zs = sorted(c["z"] for c in calls if c.get("z") is not None)
    if zs:
        for bar in (family.LATTICE_GATE_Z, family.LATTICE_STRUCTURAL_Z):
            print(f"  z >= {bar}: {sum(z >= bar for z in zs)}/{len(zs)}")
        print("  z deciles:", [round(z, 2) for z in zs[::max(1, len(zs) // 10)]])
    amounts = [c["amount"] for c in calls if c.get("amount") is not None]
    if amounts:
        print("  amount range:", round(min(amounts), 5), "..",
              round(max(amounts), 5))
    sec = [c for c in calls if c.get("secondary_amount") is not None]
    if sec:
        fired = [c for c in sec if c["secondary_amount"] > 0.]
        print(f"  secondary leg: {len(fired)}/{len(sec)} origins with a -2 "
              f"deficit in their own train actuals")
        if fired:
            amts = sorted(c["secondary_amount"] for c in fired)
            print("  secondary amount range:", round(min(amts), 5), "..",
                  round(max(amts), 5))

    origins = bundle["origins"]
    with_delta = sum(1 for o in origins
                     if (o["maps"]["lattice"]["margin"] or {}).get("delta"))
    print("origins shipping a margin delta:", with_delta, "/", len(origins))

    chart(priced, "20261017 SHIPPED (this walk, == delivered artifact)")
    delivered = pd.read_csv(path, float_precision="round_trip",
                            low_memory=False)
    chart(delivered[delivered["kind"] == "oof"], "20261017 DELIVERED CSV")

    prev = ROOT / "mlb-backend" / "data_delivery" / "run_engine_markets_20261016.csv"
    if prev.exists():
        old = pd.read_csv(prev, float_precision="round_trip", low_memory=False)
        chart(old[old["kind"] == "oof"], "20261016 (pre-rebalance)")

    Path(TMP / "origin_calls.json").write_text(json.dumps(calls, indent=1))
    print("\nsaved", TMP / "origin_calls.json")


if __name__ == "__main__":
    main()
