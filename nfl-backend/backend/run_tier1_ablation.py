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
    """Column lists per arm, kept only where the frame carries them."""
    without = [c for c in WITHOUT_FEATURES if c in feats.columns]
    tier1 = [c for c in TIER1_FEATURES if c in feats.columns]
    return {"WITHOUT": without, "WITH": without + tier1}


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
        reason.append("WITH does not beat WITHOUT on sealed logloss AND AUC")
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
    verdict = adopt_verdict(sealed["WITHOUT"], sealed["WITH"],
                            pooled["WITHOUT"], pooled["WITH"])

    print("\n=== Tier-1 ablation (WITH vs WITHOUT) ===")
    print("arm       sealed_ll  sealed_auc  sealed_ece  pooled_ll")
    for n in arms:
        s, p = sealed[n], pooled[n]
        print(f"{n:9s} {s['logloss']}  {s['auc']}  {s['ece']}  {p['logloss']}")
    print("VERDICT:", "ADOPT" if verdict["adopt"] else "DON'T ADOPT",
          "|", " | ".join(verdict["reason"]))

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
        "verdict": verdict,
    }
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"nfl_tier1_ablation_{record['frame_sha256']}.json"
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
