"""Tier-2 (venue/travel/schedule) feature-family ablation for the NFL moneyline ensemble.

Runs the SAME walk-forward + sealed-2025 machinery as the production gate
(``nfl_moneyline.run_walk_forward``) on three arms:

  WITHOUT  — the admitted 10-feature set the 2026-08-31 gate record shipped
             (minus the ``is_home`` constant anchor, never a model column).
  VENUE    — WITHOUT + all six Tier-2 static venue/travel/schedule candidates
             (travel_miles_diff, timezone_diff, altitude_home, turf_home,
             prime_time, neutral_site).
  VENUE_3  — WITHOUT + the three strongest univariate slices (travel_miles_diff,
             altitude_home, prime_time). Tier-1's lesson: small curated slices
             beat blocks, so the 3-slice arm tests whether a minimal venue
             addition keeps any pooled gain without a sealed regression.

The Tier-2 candidates are static pre-game facts (stadium coordinates,
elevation, timezone, surface, kickoff hour, neutral location) — leak-safe by
construction, and composed by nfl_features.build_features / build_slate_features
but NOT in FEATURE_COLUMNS (the deployed pool stays the 10-feature baseline
until a sealed ablation admits them). The sealed-2025 hold-out is never
touched during fitting (guaranteed by the shared machinery), and the adoption
gate is the same rule as Tier-1/MLB: an arm must beat WITHOUT on the SEALED
hold-out in logloss AND AUC without degrading ECE-cal; pooled OOF logloss
corroborates. A pooled-gain / sealed-loss inversion means DON'T ADOPT.

Usage (Kaggle — network + nflreadpy needed for the raw pull):
    python3 run_tier2_ablation.py                  # full 2019-2025 window
    python3 run_tier2_ablation.py --features <features.csv>
    python3 run_tier2_ablation.py --no-record      # report only
Artifact: data_delivery/nfl_tier2_ablation_<sha>.json (uncommitted; review
before any commit).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from nfl_features import VENUE_FEATURES
from run_tier1_ablation import WITHOUT_FEATURES, _frame_sha256, adopt_verdict

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"

# The 3-slice arm (Tier-1 lesson: small curated slices beat blocks).
VENUE_3_FEATURES = ["travel_miles_diff", "altitude_home", "prime_time"]


def load_features(features_csv: str | None) -> pd.DataFrame:
    """Feature frame: a provided CSV, else the nflreadpy pull + build (Kaggle)."""
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
    """Column lists per arm, kept only where the frame carries them.

    Arms: WITHOUT (deployed 10), VENUE (10 + all 6 Tier-2 candidates),
    VENUE_3 (10 + travel_miles_diff, altitude_home, prime_time)."""
    without = [c for c in WITHOUT_FEATURES if c in feats.columns]
    venue = [c for c in VENUE_FEATURES if c in feats.columns]
    venue3 = [c for c in VENUE_3_FEATURES if c in feats.columns]
    return {"WITHOUT": without, "VENUE": without + venue,
            "VENUE_3": without + venue3}


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
    for name, cols in arms.items():
        print(f"{name}: {len(cols)} cols -> {cols}")

    results = {}
    for name, cols in arms.items():
        print(f"\n=== running walk-forward arm {name} ({len(cols)} features) ===")
        results[name] = run_walk_forward(feats, model_features=cols)

    def _m(rec: dict) -> dict:
        return {k: rec.get(k) for k in ("logloss", "auc", "ece")}

    sealed = {n: _m(results[n]["sealed_2025"]["model_platt"]) for n in arms}
    pooled = {n: _m(results[n]["pooled_preq_2021_2024"]["model_platt"]) for n in arms}

    verdict = adopt_verdict(sealed["WITHOUT"], sealed["VENUE"],
                            pooled["WITHOUT"], pooled["VENUE"])
    verdict_3 = adopt_verdict(sealed["WITHOUT"], sealed["VENUE_3"],
                              pooled["WITHOUT"], pooled["VENUE_3"])

    print("\n=== Tier-2 ablation (VENUE / VENUE_3 vs WITHOUT) ===")
    print("arm           sealed_ll  sealed_auc  sealed_ece  pooled_ll")
    for n in arms:
        s, p = sealed[n], pooled[n]
        print(f"{n:14s} {s['logloss']}  {s['auc']}  {s['ece']}  {p['logloss']}")
    print("\nVERDICT (VENUE-6 vs WITHOUT):",
          "ADOPT" if verdict["adopt"] else "DON'T ADOPT",
          "|", " | ".join(verdict["reason"]))
    print("VERDICT (VENUE_3-3 vs WITHOUT):",
          "ADOPT" if verdict_3["adopt"] else "DON'T ADOPT",
          "|", " | ".join(verdict_3["reason"]))

    if args.no_record:
        return 0
    record = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "frame_sha256": _frame_sha256(feats),
        "arms": {n: {"features": cols,
                     "sealed_model_platt": sealed[n],
                     "pooled_model_platt": pooled[n]}
                 for n, cols in arms.items()},
        "verdict_venue": verdict,
        "verdict_venue_3": verdict_3,
    }
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"nfl_tier2_ablation_{record['frame_sha256']}.json"
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())