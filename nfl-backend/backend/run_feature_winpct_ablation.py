"""win_pct_diff removal arm — evidence test for the only served feature whose
inclusion rests on RESTORATION rather than an ablation-gate verdict.

Baseline (WITH_12):  the deployed 12-feature pool, synced with
FEATURE_COLUMNS (9 v1/v2 + travel_miles_diff, altitude_home, prime_time;
ewm_qb_epa_play_diff was removed 2026-09-01 by the corr-pair twin verdict
cd3c26b, and market_home_implied was admitted then policy-reverted out —
the model stays market-independent, the market is a benchmark, not an input).
Test arm (WITHOUT_11): the same 12 MINUS win_pct_diff.

The sealed-2025 gate is the SAME rule as the Tier-1/2/3 harnesses
(run_tier1_ablation.adopt_verdict): the test arm must beat the baseline on
SEALED logloss AND AUC without degrading ECE-cal for REMOVAL to be
evidence-based ADOPT. If the removal loses (or ties) either sealed axis,
win_pct_diff stays served — now on measurement, not assumption.

Usage (network + nflreadpy needed for the raw pull):
    python3 run_feature_winpct_ablation.py
    python3 run_feature_winpct_ablation.py --features <features.csv>
    python3 run_feature_winpct_ablation.py --no-record
Artifact: data_delivery/nfl_feature_winpct_ablation_<sha>.json
(reviewed before commit; the evidence record is committed with the harness
per the close-out instruction).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from run_tier1_ablation import (MEMBER_NAMES, _frame_sha256, _member_metrics,
                                adopt_verdict)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"

# The deployed pool (FEATURE_COLUMNS minus the is_home anchor) as of the
# 2026-09-01 corr-pair twin verdict (cd3c26b): 12 features — 9 v1/v2 + the
# VENUE_3 Tier-2 admission, minus ewm_qb_epa_play_diff. market_home_implied
# is deliberately NOT here (policy reversal).
DEPLOYED_12 = [
    "elo_diff", "win_pct_diff", "rest_days_diff", "is_dome_home",
    "ewm_net_pts_diff", "ewm_ypp_diff",
    "pace_plays_min_diff", "rest_short_diff", "div_game",
    "travel_miles_diff", "altitude_home", "prime_time",
]
REMOVED = "win_pct_diff"


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
    """WITH_12 (deployed pool incl. win_pct_diff) / WITHOUT_11 (removal)."""
    with_f = [c for c in DEPLOYED_12 if c in feats.columns]
    without_f = [c for c in DEPLOYED_12 if c != REMOVED and c in feats.columns]
    return {"WITH_12": with_f, "WITHOUT_11": without_f}


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
    print(f"win_pct_diff decided coverage: "
          f"{100 * float(feats[REMOVED].notna().mean()):.1f}%")
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

    # adopt_verdict(baseline, candidate): candidate = WITHOUT_11 (the removal)
    verdict = adopt_verdict(sealed["WITH_12"], sealed["WITHOUT_11"],
                            pooled["WITH_12"], pooled["WITHOUT_11"])

    print("\n=== win_pct_diff removal arm (WITH_12 vs WITHOUT_11) ===")
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

    if verdict["adopt"]:
        print("\nVERDICT (REMOVAL vs WITH_12): ADOPT — win_pct_diff removal "
              "beats the 12 on sealed logloss AND AUC; a real evidence basis "
              "exists to drop it.")
    else:
        print("\nVERDICT (REMOVAL vs WITH_12): KEEP — the removal wins neither "
              "sealed axis; win_pct_diff stays served on measurement, not "
              "assumption.")
    for r in verdict["reason"]:
        print("  -", r)

    if args.no_record:
        return 0
    record = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "frame_sha256": _frame_sha256(feats),
        "removed": REMOVED,
        "arms": {n: {"features": cols,
                     "sealed_model_platt": sealed[n],
                     "pooled_model_platt": pooled[n],
                     "members": {m: dict(v) for m, v in
                                 (results[n].get("members") or {}).items()},
                     "members_sealed": {m: dict(v) for m, v in
                                 (results[n].get("members_sealed") or {}).items()}}
                 for n, cols in arms.items()},
        "verdict_removal": verdict,
    }
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"nfl_feature_winpct_ablation_{record['frame_sha256']}.json"
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())