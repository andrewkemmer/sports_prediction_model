"""Tier-3 (officials / roster) feature-family ablation.

Runs the SAME walk-forward + sealed-2025 machinery as the production gate
(``nfl_moneyline.run_walk_forward``) on four arms. (The former market de-vig
family is gone — market-independence policy: no market data anywhere in the
NFL pipeline, so no ablation arm either.)

  WITHOUT — the deployed 13-feature baseline (the 10 v1/v2 features admitted
            2026-08-28 plus the Tier-2 VENUE_3 slices travel_miles_diff,
            altitude_home, prime_time admitted 2026-09-01).
  OFF     — WITHOUT + ref_pen_tend, ref_pace (head-referee crew tendencies).
  ROSTER  — WITHOUT + roster_age_diff, roster_exp_diff (pre-season team means).
  ALL     — WITHOUT + all four Tier-3 candidates.

The Tier-3 candidates are composed by nfl_features.build_features /
build_slate_features but are NOT in FEATURE_COLUMNS (the deployed pool
changes only when this ablation admits something — the Tier-1/Tier-2 rule).
The sealed-2025 hold-out is never touched during fitting (guaranteed by the
shared walk-forward), and the adoption gate is the same rule as Tier-1/2:
an arm must beat WITHOUT on the SEALED hold-out in logloss AND AUC without
degrading ECE-cal; pooled OOF logloss corroborates. Per-member pooled AND
sealed logloss/auc tables are printed and recorded (f01f880 pattern).

Usage (network + nflreadpy needed for the raw pull):
    python3 run_tier3_ablation.py                  # full 2019-2025 window
    python3 run_tier3_ablation.py --features <features.csv>
    python3 run_tier3_ablation.py --no-record      # report only
Artifact: data_delivery/nfl_tier3_ablation_<sha>.json (uncommitted; review
before any commit).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from nfl_features import TIER3_OFF_FEATURES, TIER3_ROSTER_FEATURES
from run_tier1_ablation import (MEMBER_NAMES, WITHOUT_FEATURES, _frame_sha256,
                                _member_metrics, adopt_verdict)
from run_tier2_ablation import VENUE_3_FEATURES

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"

# Deployed baseline = WITHOUT (10) + the admitted Tier-2 slices.
BASELINE_13 = WITHOUT_FEATURES + VENUE_3_FEATURES


def load_features(features_csv: str | None) -> pd.DataFrame:
    """Feature frame: a provided CSV, else the nflreadpy pull + build."""
    if features_csv and Path(features_csv).exists():
        feats = pd.read_csv(features_csv)
        feats["gameday"] = pd.to_datetime(feats["gameday"])
    else:
        from nfl_features import DEFAULT_SEASONS, _load_raw, build_features
        from nfl_moneyline import DECIDED_FRAME
        decided = pd.read_csv(DECIDED_FRAME)
        sched, pbp = _load_raw(DEFAULT_SEASONS)
        feats = build_features(decided, sched, pbp)
    if "home_win" not in feats.columns:
        feats["home_win"] = (feats["home_score"] > feats["away_score"]).astype(int)
    return feats


def build_arms(feats: pd.DataFrame) -> dict[str, list[str]]:
    """Column lists per arm, kept only where the frame carries them."""
    without = [c for c in BASELINE_13 if c in feats.columns]
    off = [c for c in TIER3_OFF_FEATURES if c in feats.columns]
    roster = [c for c in TIER3_ROSTER_FEATURES if c in feats.columns]
    return {"WITHOUT": without, "OFF": without + off,
            "ROSTER": without + roster, "ALL": without + off + roster}


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
    print(f"decided games: {len(feats)} | frame sha256: {_frame_sha256(feats)}")
    print("candidate coverage (decided frame):")
    for name in (TIER3_OFF_FEATURES + TIER3_ROSTER_FEATURES):
        print(f"  {name:22s} "
              f"{100 * float(feats[name].notna().mean()):6.1f}%")
    for name, cols in arms.items():
        print(f"{name}: {len(cols)} cols -> {cols}")

    results = {}
    for name, cols in arms.items():
        print(f"\n=== running walk-forward arm {name} ({len(cols)} features) ===")
        results[name] = run_walk_forward(feats, model_features=cols)

    def _m(rec: dict) -> dict:
        return {k: rec.get(k) for k in ("logloss", "auc", "ece")}

    sealed = {n: _m(results[n]["sealed_2025"]["model_platt"]) for n in arms}
    pooled = {n: _m(results[n]["pooled_preq_2021_2024"]["model_platt"])
              for n in arms}

    pairs = [("OFF", "officials slice"),
             ("ROSTER", "roster slice"),
             ("ALL", "all 4 Tier-3 candidates")]
    verdicts = {n: adopt_verdict(sealed["WITHOUT"], sealed[n],
                                 pooled["WITHOUT"], pooled[n])
                for n, _ in pairs}

    print("\n=== Tier-3 ablation (OFF / ROSTER / ALL vs WITHOUT) ===")
    print("arm           sealed_ll  sealed_auc  sealed_ece  pooled_ll")
    for n in arms:
        s, p = sealed[n], pooled[n]
        print(f"{n:14s} {s['logloss']}  {s['auc']}  {s['ece']}  {p['logloss']}")

    member_pooled = {n: _member_metrics(results[n], "members") for n in arms}
    member_sealed = {n: _member_metrics(results[n], "members_sealed")
                     for n in arms}

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

    for arm, note in pairs:
        v = verdicts[arm]
        print(f"VERDICT ({arm} vs WITHOUT):",
              "ADOPT" if v["adopt"] else "DON'T ADOPT",
              f"({note})", "|", " | ".join(v["reason"]))

    if args.no_record:
        return 0
    record = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "frame_sha256": _frame_sha256(feats),
        "arms": {n: {"features": cols,
                     "sealed_model_platt": sealed[n],
                     "pooled_model_platt": pooled[n],
                     "members": {m: dict(v) for m, v in
                                 (results[n].get("members") or {}).items()},
                     "members_sealed": {m: dict(v) for m, v in
                                 (results[n].get("members_sealed") or {}).items()}}
                 for n, cols in arms.items()},
        "verdicts": {n: verdicts[n] for n, _ in pairs},
    }
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"nfl_tier3_ablation_{record['frame_sha256']}.json"
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())