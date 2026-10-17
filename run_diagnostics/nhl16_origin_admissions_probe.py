"""Why the 20261016 NHL dashboard shows no spread improvement (2026-10-17).

The delivered meta (nhl_run_engine_markets_20261016.meta.json) carries the
empty_net_report: both pairs were sized from actuals (demand 0.0368/0.0251,
supply 0.0248/0.0108) but neither cleared the held-out structural bar
(z 1.739 / 1.492 < 2.5) -> no_transfer_admitted, even on the FULL 2718-game
serving evidence. The Distribution chart, however, pools the PMFs EACH
ORIGIN shipped — per-origin evidence is far thinner than the serving fit's.
This probe re-runs the causal walk once on the delivered OOF rows, records
EVERY fit_empty_net_transfer call (per-pair demand/supply/transfer/z,
refusal reason), and prints:

  * admissions per origin and per pair, plus the z distribution vs both
    bars (generic 3.0, structural 2.5) — where the shortfall comes from,
  * how many origins ship ANY margin delta (transfer vs generic vs
    structural-fallback) so we know what the pooled chart can even see,
  * the pooled chart numbers (observed vs shipped at -4..+4) from this
    walk, from the DELIVERED CSV, and from the previous artifact.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nhl-backend" / "backend"))
import distributions as dist          # noqa: E402
import gaussian_family as family      # noqa: E402

ARTIFACT = ROOT / "nhl-backend" / "data_delivery" / "nhl_run_engine_markets_20261016.csv"


def chart(priced: pd.DataFrame, label: str) -> None:
    d = priced.dropna(subset=["home_score", "away_score"]).copy()
    d["m"] = (d.home_score - d.away_score).astype(int)
    print(f"\n--- {label}: pooled chart numbers (n={len(d)}) ---")
    print(f"{'bin':>4} {'actual':>8} {'shipped':>8} {'gap':>9}")
    for v in (-4, -3, -2, -1, 1, 2, 3, 4):
        col = f"p_push_{v}" if v > 0 else f"p_push_m{-v}"
        if col not in d:
            continue
        o, m = float((d.m == v).mean()), float(d[col].mean())
        print(f"{v:>4} {o:>8.4f} {m:>8.4f} {o - m:>+9.4f}")


def main() -> None:
    frame = pd.read_csv(ARTIFACT, float_precision="round_trip", low_memory=False)
    oof = frame[frame["kind"] == "oof"].copy()

    calls = []
    real = dist.fit_empty_net_transfer

    def hook(*a, **k):
        delta, rep = real(*a, **k)
        pairs = rep.get("pairs") or []
        calls.append({
            "n_evidence": int(len(a[1])) if len(a) > 1 and a[1] is not None else 0,
            "reason": rep.get("reason"),
            "delta_admitted": delta is not None,
            "pairs": [{"donor": p.get("donor"), "recipient": p.get("recipient"),
                       "demand": p.get("demand"), "supply": p.get("supply"),
                       "transfer": p.get("transfer"), "z": p.get("z"),
                       "admitted": p.get("admitted"),
                       "reason": p.get("reason")} for p in pairs],
        })
        return delta, rep

    dist.fit_empty_net_transfer = hook
    t0 = time.time()
    priced, bundle = dist.calibrate_market_frame(oof)
    dist.fit_empty_net_transfer = real
    print("walk_seconds:", round(time.time() - t0, 1), flush=True)

    print("\n--- per-origin empty-net transfer calls (serving fit is last) ---")
    print("calls:", len(calls))
    reasons = {}
    for c in calls:
        reasons[c["reason"]] = reasons.get(c["reason"], 0) + 1
    print("call reasons:", reasons)
    print("deltas admitted:", sum(c["delta_admitted"] for c in calls))

    pair_rows = [p for c in calls for p in c["pairs"]]
    pair_reasons = {}
    for p in pair_rows:
        pair_reasons[p["reason"]] = pair_reasons.get(p["reason"], 0) + 1
    print("pair refusals:", pair_reasons)
    zs = sorted(p["z"] for p in pair_rows if p.get("z") is not None)
    if zs:
        for bar in (family.LATTICE_GATE_Z, family.LATTICE_STRUCTURAL_Z):
            print(f"  pair z >= {bar}: {sum(z >= bar for z in zs)}/{len(zs)}")
        print("  pair z deciles:",
              [round(z, 2) for z in zs[::max(1, len(zs) // 10)]])
    transfers = [p["transfer"] for p in pair_rows if p.get("transfer")]
    if transfers:
        print("  transfer range:", round(min(transfers), 5), "..",
              round(max(transfers), 5))
    zs_by_n = [(c["n_evidence"], p["z"]) for c in calls
               for p in c["pairs"] if p.get("z") is not None]
    if zs_by_n:
        print("  z vs evidence size (last 12):",
              [(n, z) for n, z in zs_by_n[-12:]])

    origins = bundle["origins"]
    with_any = with_transfer = with_structural = with_generic = 0
    for o in origins:
        lat = (o["maps"].get("lattice") or {}).get("margin") or {}
        if lat.get("delta"):
            with_any += 1
            if lat.get("empty_net_report", {}).get("significant"):
                with_transfer += 1
            elif lat.get("structural"):
                with_structural += 1
            else:
                with_generic += 1
    print(f"\norigins shipping a margin delta: {with_any} / {len(origins)}"
          f" (transfer {with_transfer}, structural-fallback {with_structural},"
          f" generic {with_generic})")
    served = (origins[-1]["maps"].get("lattice") or {}).get("margin") or {}
    print("serving fit margin significant:", served.get("significant"),
          "| structural:", served.get("structural"),
          "| gate_z:", served.get("gate_z"),
          "| empty_net reason:",
          (served.get("empty_net_report") or {}).get("reason"))

    chart(priced, "WALK (this run, == what an artifact regenerated now ships)")
    chart(oof, "20261016 DELIVERED CSV")
    prev = ROOT / "nhl-backend" / "data_delivery" / "nhl_run_engine_markets_20261015.csv"
    if prev.exists():
        old = pd.read_csv(prev, float_precision="round_trip", low_memory=False)
        chart(old[old["kind"] == "oof"], "20261015 (previous artifact)")

    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(
        {"calls": calls,
         "origins_with_margin_delta": with_any,
         "origins_total": len(origins)}, indent=1))
    print("\nsaved", out)


if __name__ == "__main__":
    main()
