"""Tier-1 (v3) feature-family ablation for the NFL moneyline ensemble.

Runs the SAME walk-forward + sealed-2025 machinery as the production gate
(``nfl_moneyline.run_walk_forward``) on two arms:

  WITHOUT — the pre-Tier-1 admitted model set (the 10 features the 2026-08-31
            gate record shipped, minus the ``is_home`` constant anchor that is
            never fed as a model column).
  WITH    — WITHOUT + the nine Tier-1 candidates (turnover differential,
            ANY/A, sack rate, EPA success rate, explosive-play rate, net
            penalty yards, third-down conversion rate, red-zone TD rate,
            points per drive).

The sealed-2025 hold-out is never touched during fitting (guaranteed by the
shared machinery), and the adoption gate mirrors MLB's
``run_opponent_adjusted_ablation.py`` rule: WITH must beat WITHOUT on the
SEALED hold-out in logloss AND AUC without degrading ECE-cal. A pooled-gain /
sealed-loss inversion means DON'T ADOPT, exactly as the user-specified gate.

Usage (Kaggle — network + nflreadpy needed for the raw pull):
    python3 run_tier1_ablation.py                  # full 2019-2025 window
    python3 run_tier1_ablation.py --features <features.csv>
Artifact: data_delivery/nfl_tier1_ablation_<sha>.json (uncommitted; review
before any commit).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"

# WITHOUT arm = the admitted set shipped by the 2026-08-31 gate record,
# excluding the is_home constant anchor (absorbed by intercept/baseline).
WITHOUT_FEATURES = [
    "elo_diff", "win_pct_diff", "rest_days_diff", "is_dome_home",
    "ewm_net_pts_diff", "ewm_qb_epa_play_diff", "ewm_ypp_diff",
    "pace_plays_min_diff", "rest_short_diff", "div_game",
]

# WITH additions — the Tier-1 (v3) candidates built by nfl_features.
TIER1_FEATURES = [
    "turnover_diff", "any_a_diff", "sack_rate_diff", "success_rate_diff",
    "explosive_rate_diff", "penalty_diff", "third_down_rate_diff",
    "redzone_td_rate_diff", "pts_per_drive_diff",
]

# Gate-admitted Tier-1 subset — the 7 that cleared coverage + redundancy on
# the 2026-08-31 run: excludes any_a_diff (redundant with ypp_diff, r=0.84)
# and pts_per_drive_diff (redundant with ewm_epa_play_diff, r=0.89).
TIER1_ADMITTED = [
    "turnover_diff", "sack_rate_diff", "success_rate_diff",
    "explosive_rate_diff", "penalty_diff", "third_down_rate_diff",
    "redzone_td_rate_diff",
]

# Smallest high-signal slice — the gate's three best univariate discriminators
# (by <2025 AUC): success rate 0.647, third-down 0.591, turnover 0.578. Tests
# whether a minimal, non-redundant Tier-1 addition keeps the pooled gain
# without the sealed regression the 7/9-feature blocks showed.
TIER1_SUBSET = ["success_rate_diff", "third_down_rate_diff", "turnover_diff"]


def _frame_sha256(df: pd.DataFrame) -> str:
    """Content hash of the feature frame (row/col-sorted) for reproducibility."""
    h = hashlib.sha256()
    sorted_df = df.sort_values("game_id").reset_index(drop=True)
    h.update(sorted_df.to_csv(index=False).encode("utf-8"))
    return h.hexdigest()[:12]


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

    Five arms: WITHOUT (deployed 10), WITH (10 + all 9 Tier-1),
    WITH_ADMITTED (10 + the gate-admitted 7 Tier-1), WITH_SUBSET
    (10 + the three strongest discriminators), and TIER1_ONLY (the 7
    admitted Tier-1 features ALONE — tests whether the new family can
    REPLACE the original 10 on its own, not just augment them)."""
    without = [c for c in WITHOUT_FEATURES if c in feats.columns]
    tier1 = [c for c in TIER1_FEATURES if c in feats.columns]
    admitted = [c for c in TIER1_ADMITTED if c in feats.columns]
    subset = [c for c in TIER1_SUBSET if c in feats.columns]
    return {"WITHOUT": without, "WITH": without + tier1,
            "WITH_ADMITTED": without + admitted,
            "WITH_SUBSET": without + subset,
            "TIER1_ONLY": admitted}


def _platt_metrics(rec: dict, key: str) -> dict:
    m = rec[key]["model_platt"]
    return {k: m.get(k) for k in ("logloss", "auc", "ece")}


def adopt_verdict(sealed_without: dict, sealed_with: dict,
                  pooled_without: dict, pooled_with: dict) -> dict:
    """Gated rule (user-specified): WITH wins only on the SEALED hold-out in
    logloss AND AUC without degrading ECE-cal; pooled corroborates."""
    ll_w = sealed_with["logloss"]; ll_o = sealed_without["logloss"]
    auc_w = sealed_with["auc"];     auc_o = sealed_without["auc"]
    ece_w = sealed_with["ece"];     ece_o = sealed_without["ece"]
    sealed_win = (ll_w is not None and ll_o is not None and auc_w is not None
                  and auc_o is not None and ll_w < ll_o and auc_w > auc_o)
    ece_ok = ece_w is None or ece_o is None or ece_w <= ece_o + 1e-9
    pooled_win = (pooled_with["logloss"] is not None
                  and pooled_without["logloss"] is not None
                  and pooled_with["logloss"] < pooled_without["logloss"])
    adopt = bool(sealed_win and ece_ok)
    reason = []
    if not sealed_win:
        reason.append("arm does not beat WITHOUT on sealed logloss AND AUC")
    if not ece_ok:
        reason.append("sealed ECE-cal degraded")
    if not pooled_win:
        reason.append("pooled OOF logloss went the wrong way (corroboration only)")
    return {
        "adopt": adopt,
        "sealed_win": bool(sealed_win),
        "ece_ok": bool(ece_ok),
        "pooled_win": bool(pooled_win),
        "delta": {
            "sealed_logloss": round(ll_w - ll_o, 4) if ll_w is not None and ll_o is not None else None,
            "sealed_auc": round(auc_w - auc_o, 4) if auc_w is not None and auc_o is not None else None,
            "sealed_ece_cal": round(ece_w - ece_o, 4) if ece_w is not None and ece_o is not None else None,
            "pooled_logloss": round(pooled_with["logloss"] - pooled_without["logloss"], 4)
            if pooled_with["logloss"] is not None and pooled_without["logloss"] is not None else None,
        },
        "reason": reason,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=None,
                    help="pre-built features CSV (skips the nflreadpy pull)")
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON record")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    from nfl_moneyline import run_walk_forward
    from nfl_features import run_feature_gate

    feats = load_features(args.features)
    arms = build_arms(feats)
    print(f"decided games: {len(feats)} | frame sha256: {_frame_sha256(feats)}")
    for name, cols in arms.items():
        print(f"{name}: {len(cols)} cols -> {cols}")

    # Admission-gate report for the Tier-1 candidates (informational: the
    # walk-forward is the arbiter; the gate documents coverage/redundancy).
    gate = run_feature_gate(feats)
    tier1_gate = {c: gate["reasons"].get(c) or "admitted" for c in TIER1_FEATURES}

    results = {}
    for name, cols in arms.items():
        print(f"\n=== running walk-forward arm {name} ({len(cols)} features) ===")
        results[name] = run_walk_forward(feats, model_features=cols)

    sealed = {n: _platt_metrics(results[n], "sealed_2025") for n in arms}
    pooled = {n: _platt_metrics(results[n], "pooled_preq_2021_2024") for n in arms}
    # Primary gate: WITHOUT vs WITH (all 9). Informational: WITHOUT vs
    # WITH_ADMITTED (the gate-pruned 7) — did dropping the 2 redundant
    # features keep the pooled gain without the sealed regression?
    verdict = adopt_verdict(sealed["WITHOUT"], sealed["WITH"],
                            pooled["WITHOUT"], pooled["WITH"])
    verdict_admitted = adopt_verdict(sealed["WITHOUT"], sealed["WITH_ADMITTED"],
                                     pooled["WITHOUT"], pooled["WITH_ADMITTED"])
    verdict_subset = adopt_verdict(sealed["WITHOUT"], sealed["WITH_SUBSET"],
                                   pooled["WITHOUT"], pooled["WITH_SUBSET"])
    verdict_tier1_only = adopt_verdict(sealed["WITHOUT"], sealed["TIER1_ONLY"],
                                       pooled["WITHOUT"], pooled["TIER1_ONLY"])

    print("\n=== Tier-1 ablation (WITH / WITH_ADMITTED / WITH_SUBSET / TIER1_ONLY vs WITHOUT) ===")
    print("arm           sealed_ll  sealed_auc  sealed_ece  pooled_ll")
    for n in arms:
        s, p = sealed[n], pooled[n]
        print(f"{n:14s} {s['logloss']}  {s['auc']}  {s['ece']}  {p['logloss']}")
    print("\nVERDICT (WITH all-9 vs WITHOUT):",
          "ADOPT" if verdict["adopt"] else "DON'T ADOPT",
          "|", " | ".join(verdict["reason"]))
    print("VERDICT (WITH_ADMITTED-7 vs WITHOUT):",
          "ADOPT" if verdict_admitted["adopt"] else "DON'T ADOPT",
          "|", " | ".join(verdict_admitted["reason"]))
    print("VERDICT (WITH_SUBSET-3 vs WITHOUT):",
          "ADOPT" if verdict_subset["adopt"] else "DON'T ADOPT",
          "|", " | ".join(verdict_subset["reason"]))
    print("VERDICT (TIER1_ONLY-7 vs WITHOUT):",
          "ADOPT" if verdict_tier1_only["adopt"] else "DON'T ADOPT",
          "|", " | ".join(verdict_tier1_only["reason"]))

    if args.no_record:
        return 0
    record = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "frame_sha256": _frame_sha256(feats),
        "arms": {n: {"features": cols,
                     "sealed_model_platt": sealed[n],
                     "pooled_model_platt": pooled[n]}
                 for n, cols in arms.items()},
        "tier1_gate": tier1_gate,
        "verdict_with": verdict,
        "verdict_with_admitted": verdict_admitted,
        "verdict_with_subset": verdict_subset,
        "verdict_tier1_only": verdict_tier1_only,
    }
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"nfl_tier1_ablation_{record['frame_sha256']}.json"
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
