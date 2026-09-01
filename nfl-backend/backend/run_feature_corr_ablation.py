"""ewm_qb_epa_play_diff / ewm_ypp_diff 0.81-correlation removal arms.

VERDICT EXECUTED — record nfl_feature_corr_ablation_e4aee120a4b8.json
(commit cd3c26b): WITHOUT_QBEPA ADOPT-REMOVE. Dropping
ewm_qb_epa_play_diff beat the 13-pool on SEALED 2025 logloss (−0.0124) AND
AUC (+0.0129) with ECE-cal improving 0.0937 → 0.0656 (under 0.08), pooled
OOF corroborating (−0.0116), so ewm_qb_epa_play_diff was dropped from the
served pool. Note: the market revert (2f79669) changed which twin is
droppable — with the market out, yards-per-play (ewm_ypp_diff) carries the
retained signal, and the WITHOUT_YPP arm (sealed AUC −0.0019) confirmed
ewm_ypp_diff is the keeper and STAYS.

The harness is re-locked to the CURRENT deployed pool (12 features) so a
re-run re-measures the still-live question with the SAME sealed gate as
every prior arm (run_tier1_ablation.adopt_verdict):

    WITH_12         baseline — the deployed 12-feature pool.
    WITHOUT_YPP     11 — drop ewm_ypp_diff (the remaining twin).

The historical WITHOUT_BOTH bounds arm is gone: ewm_qb_epa_play_diff is
already unserved, so it would degenerate to WITHOUT_YPP (the original 4-arm
run is preserved in the committed record). REMOVAL is justified only if the
removal arm beats WITH_12 on SEALED 2025 logloss AND AUC without degrading
ECE-cal, corroborated by pooled OOF. The |r| 0.8055 pair itself is now
unserved — only ewm_ypp_diff remains in the pool — so the report-only gate
can no longer surface it.

Usage (network + nflreadpy needed for the pull):
    python3 run_feature_corr_ablation.py
    python3 run_feature_corr_ablation.py --features <features.csv>
    python3 run_feature_corr_ablation.py --no-record
Artifact: data_delivery/nfl_feature_corr_ablation_<sha>.json
(Harness + tests + record committed together per the close-out convention.)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from run_feature_winpct_ablation import DEPLOYED_12, load_features
from run_tier1_ablation import (MEMBER_NAMES, _frame_sha256, _member_metrics,
                                adopt_verdict)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"

QBEPA = "ewm_qb_epa_play_diff"
YPP = "ewm_ypp_diff"
CORRELATION = 0.8055  # reported by the 2026-09-01 report-only gate


def build_arms(feats: pd.DataFrame) -> dict[str, list[str]]:
    """WITH_12 (deployed pool) and the remaining YPP twin removal (the
    historical WITHOUT_QBEPA / WITHOUT_BOTH arms were executed or
    degenerate — see the module docstring). Columns absent from the frame
    are dropped."""
    base = [c for c in DEPLOYED_12 if c in feats.columns]
    return {
        "WITH_12": base,
        "WITHOUT_YPP": [c for c in base if c != YPP],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=None,
                    help="pre-built features CSV (skips the nflreadpy pull)")
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON record")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    from nfl_moneyline import run_walk_forward

    feats = load_features(args.features)
    arms = build_arms(feats)
    print(f"decided games: {len(feats)} | frame sha256: {_frame_sha256(feats)}", flush=True)
    print(f"twins decided coverage: {QBEPA} "
          f"{100 * float(feats[QBEPA].notna().mean()):.1f}% | {YPP} "
          f"{100 * float(feats[YPP].notna().mean()):.1f}%")
    print(f"reported |r|({QBEPA}, {YPP}) = {CORRELATION}")
    for name, cols in arms.items():
        print(f"{name}: {len(cols)} cols -> {cols}", flush=True)

    results = {}
    for name, cols in arms.items():
        print(f"\n=== running walk-forward arm {name} ({len(cols)} features) ===", flush=True)
        results[name] = run_walk_forward(feats, model_features=cols)

    def _m(rec: dict) -> dict:
        return {k: rec.get(k) for k in ("logloss", "auc", "ece")}

    sealed = {n: _m(results[n]["sealed_2025"]["model_platt"]) for n in arms}
    pooled = {n: _m(results[n]["pooled_preq_2021_2024"]["model_platt"])
              for n in arms}

    # adopt_verdict(baseline, candidate): the removal arm is the CANDIDATE —
    # adopt=True means the removal beats WITH_13 on both sealed axes.
    verdicts = {}
    for n in arms:
        if n == "WITH_12":
            continue
        verdicts[n] = adopt_verdict(sealed["WITH_12"], sealed[n],
                                    pooled["WITH_12"], pooled[n])

    print("\n=== corr-pair removal arms (WITH_12 baseline) ===")
    print("arm             sealed_ll  sealed_auc  sealed_ece  pooled_ll")
    for n in arms:
        s, p = sealed[n], pooled[n]
        print(f"{n:15s} {s['logloss']}  {s['auc']}  {s['ece']}  {p['logloss']}", flush=True)

    member_pooled = {n: _member_metrics(results[n], "members") for n in arms}
    member_sealed = {n: _member_metrics(results[n], "members_sealed") for n in arms}

    def _member_rows(blk: dict[str, dict]) -> None:
        print(f"{'member':12s}" + "".join(f"{n:>17s}" for n in arms))
        for m in MEMBER_NAMES:
            cells = []
            for n in arms:
                e = blk[n].get(m) or {}
                cells.append(f"{e.get('logloss', '--')}/{e.get('auc', '--')}")
            print(f"{m:12s}" + "".join(f"{c:>17s}" for c in cells))

    print("\n=== per-member pooled OOF (logloss/auc) ===")
    _member_rows(member_pooled)
    print("\n=== per-member sealed 2025 (logloss/auc) ===")
    _member_rows(member_sealed)

    for n, v in verdicts.items():
        tag = "ADOPT-REMOVE" if v["adopt"] else "KEEP"
        print(f"\nVERDICT {n}: {tag} (removal vs WITH_12)")
        for r in v["reason"]:
            print("  -", r)

    drops = [n for n, v in verdicts.items() if v["adopt"]]
    if not drops:
        print("\nFINAL: KEEP — no removal beats WITH_12 on both sealed axes; "
              "ewm_ypp_diff stays served on measurement.")
    else:
        print(f"\nFINAL: DROP {drops} — removal(s) beat WITH_12 on sealed "
              "logloss AND AUC (see per-arm verdicts).")

    if args.no_record:
        return 0
    record = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "frame_sha256": _frame_sha256(feats),
        "correlation": CORRELATION,
        "arms": {n: {"features": cols,
                     "sealed_model_platt": sealed[n],
                     "pooled_model_platt": pooled[n],
                     "members": {m: dict(v) for m, v in
                                 (results[n].get("members") or {}).items()},
                     "members_sealed": {m: dict(v) for m, v in
                                 (results[n].get("members_sealed") or {}).items()}}
                 for n, cols in arms.items()},
        "verdicts": verdicts,
        "final": "KEEP BOTH" if not drops else f"DROP {drops}",
    }
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"nfl_feature_corr_ablation_{record['frame_sha256']}.json"
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())