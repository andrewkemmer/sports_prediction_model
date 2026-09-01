"""Tier-4 (v6) feature-family ablation — game-script / opponent-adjusted /
drive-level / QB-conditional PBP candidates.

Runs the SAME walk-forward + sealed-2025 machinery as the production gate
(``nfl_moneyline.run_walk_forward``) on the deployed 12-feature baseline:

  WITHOUT — the deployed 12-feature pool (FEATURE_COLUMNS minus the is_home
            anchor; ewm_qb_epa_play_diff removed 2026-09-01 by the corr-pair
            twin verdict cd3c26b, market_home_implied policy-deleted).
  GS      — WITHOUT + the 3 game-script features (non-garbage-time EWM twins:
            ewm_qb_epa_play_diff_gs, ewm_net_pts_diff_gs, ewm_ypp_diff_gs;
            nflfastR garbage definition via ``wp`` — never ``vegas_wp``,
            market-independence policy).
  OPPADJ  — WITHOUT + the 3 opponent-adjusted axes
            (ewm_qb_epa_play_diff_oppadj, ewm_net_pts_diff_oppadj,
            ewm_ypp_diff_oppadj).
  GSOPP   — WITHOUT + all 6 (GS + OPPADJ).
  DRIVE   — conditional arm: WITHOUT + ewm_yds_per_drive_diff,
            ewm_epa_per_drive_diff, ewm_qb_epa_per_drive_diff (per-drive
            efficiency). Skipped with a printed reason if any candidate's
            decided coverage < 95%.
  QB      — conditional arm: WITHOUT + ewm_qb_epa_starter_diff (trailing QB
            EPA/play restricted to the announced/recorded starter's plays —
            nflverse schedule qb_id, which matches pbp passer_id). Slate-safe:
            the trailing shift uses only PAST games' recorded starters
            (pending games post no expected starter, verified 2026-09-01, so
            the current game's starter is never assumed or faked). Skipped if
            decided coverage < 95%.

The adoption gate is the SAME rule as Tier-1/2/3 (run_tier1_ablation.
adopt_verdict): an arm must beat WITHOUT on the SEALED 2025 hold-out in
logloss AND AUC without degrading ECE-cal; pooled OOF logloss corroborates.
Per-member pooled AND sealed logloss/auc tables are printed and recorded
(f01f880 pattern). The deployed pool changes ONLY if a verdict says ADOPT —
the candidates are composed-but-unregistered by construction.

Usage (network + nflreadpy needed for the raw pull):
    python3 run_tier4_ablation.py
    python3 run_tier4_ablation.py --features <features.csv>
    python3 run_tier4_ablation.py --no-record      # report only
Artifact: data_delivery/nfl_tier4_ablation_<sha>.json (reviewed before any
commit; the evidence record is committed with the harness per convention).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_tier4 import (TIER4_DRIVE_FEATURES, TIER4_GS_FEATURES,
                       TIER4_OPPADJ_FEATURES, TIER4_PBP_NEEDS,
                       TIER4_QB_FEATURES, compose_tier4_features)
from run_tier1_ablation import (MEMBER_NAMES, _frame_sha256, _member_metrics,
                                adopt_verdict)
from run_feature_winpct_ablation import DEPLOYED_12

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"

# The deployed 12-feature pool (FEATURE_COLUMNS minus is_home), imported from
# run_feature_winpct_ablation so the harness always tests the TRUE served pool.
BASELINE_12 = list(DEPLOYED_12)

COVERAGE_FLOOR = 0.95
CONDITIONAL_ARMS = ("DRIVE", "QB")


def load_features(features_csv: str | None) -> pd.DataFrame:
    """Feature frame: a provided CSV, else the nflreadpy pull + build.

    The pbp keep-list extends ``nfl_features._load_raw`` with the Tier-4
    source columns (wp / qtr / passer_id) so the GS mask and starter
    restriction are computable; the Tier-4 candidates are then composed onto
    the built frame via ``compose_tier4_features``.
    """
    if features_csv and Path(features_csv).exists():
        feats = pd.read_csv(features_csv)
        feats["gameday"] = pd.to_datetime(feats["gameday"])
    else:
        import nflreadpy
        from nfl_features import (DEFAULT_SEASONS, TIER1_NEEDS, _decided_rows,
                                  build_features)
        from nfl_moneyline import DECIDED_FRAME
        seasons = DEFAULT_SEASONS
        sched = nflreadpy.load_schedules(seasons).to_pandas()
        pbp = nflreadpy.load_pbp(seasons)
        keep = [c for c in (("game_id", "posteam", "yards_gained", "epa",
                             "qb_epa", "game_seconds_remaining")
                            + TIER1_NEEDS + TIER4_PBP_NEEDS)
                if c in pbp.columns]
        pbp = pbp.select(keep).to_pandas()
        decided = pd.read_csv(DECIDED_FRAME)
        decided = decided[decided["season"].isin(seasons)]
        feats = build_features(decided, sched, pbp)
        feats = compose_tier4_features(feats, sched, pbp)
        # For the frame sha + decided universe parity with the other harnesses.
        feats["_decided"] = feats["game_id"].isin(set(_decided_rows(sched)["game_id"]))
    if "home_win" not in feats.columns:
        feats["home_win"] = (feats["home_score"] > feats["away_score"]).astype(int)
    return feats


def build_arms(feats: pd.DataFrame) -> dict[str, list[str]]:
    """Column lists per arm, kept only where the frame carries them.

    GS / OPPADJ / GSOPP run unconditionally; DRIVE and QB are CONDITIONAL —
    skipped (with the reason printed by the caller) when any candidate's
    decided coverage is below the 95% floor.
    """
    base = [c for c in BASELINE_12 if c in feats.columns]
    gs = [c for c in TIER4_GS_FEATURES if c in feats.columns]
    opp = [c for c in TIER4_OPPADJ_FEATURES if c in feats.columns]
    drive = [c for c in TIER4_DRIVE_FEATURES if c in feats.columns]
    qb = [c for c in TIER4_QB_FEATURES if c in feats.columns]
    arms = {"WITHOUT": base, "GS": base + gs, "OPPADJ": base + opp,
            "GSOPP": base + gs + opp}
    for name, cands in (("DRIVE", drive), ("QB", qb)):
        if cands and min(float(feats[c].notna().mean()) for c in cands) >= COVERAGE_FLOOR:
            arms[name] = base + cands
    return arms


def _coverage_report(feats: pd.DataFrame) -> None:
    print("Tier-4 candidate coverage (decided frame):")
    for c in (TIER4_GS_FEATURES + TIER4_OPPADJ_FEATURES
              + TIER4_DRIVE_FEATURES + TIER4_QB_FEATURES):
        if c in feats.columns:
            print(f"  {c:32s} {100 * float(feats[c].notna().mean()):6.1f}%")
        else:
            print(f"  {c:32s}   ABSENT")


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
    _coverage_report(feats)
    for name, cols in arms.items():
        print(f"{name}: {len(cols)} cols -> {cols}")
    for name in CONDITIONAL_ARMS:
        if name not in arms:
            print(f"[conditional] {name} skipped: decided coverage below "
                  f"{100 * COVERAGE_FLOOR:.0f}% — see the coverage report")

    results = {}
    for name, cols in arms.items():
        print(f"\n=== running walk-forward arm {name} ({len(cols)} features) ===")
        results[name] = run_walk_forward(feats, model_features=cols)

    def _m(rec: dict) -> dict:
        return {k: rec.get(k) for k in ("logloss", "auc", "ece")}

    sealed = {n: _m(results[n]["sealed_2025"]["model_platt"]) for n in arms}
    pooled = {n: _m(results[n]["pooled_preq_2021_2024"]["model_platt"])
              for n in arms}

    pairs = [("GS", "game-script slice"), ("OPPADJ", "opponent-adjusted slice"),
             ("GSOPP", "GS + OPPADJ")]
    pairs += [(n, {"DRIVE": "drive-level slice",
                   "QB": "QB-conditional slice"}[n])
              for n in CONDITIONAL_ARMS if n in arms]
    verdicts = {n: adopt_verdict(sealed["WITHOUT"], sealed[n],
                                 pooled["WITHOUT"], pooled[n])
                for n, _ in pairs}

    print("\n=== Tier-4 ablation (GS / OPPADJ / GSOPP / DRIVE / QB vs WITHOUT) ===")
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
        "baseline": "deployed 12-feature pool (FEATURE_COLUMNS minus is_home)",
        "notes": ("GS filter = nflfastR garbage-time via wp (NOT vegas_wp — "
                  "market-independence policy); QB-conditional conditions on "
                  "past games' recorded starters and is slate-safe (the "
                  "current game's starter is never assumed — pending games "
                  "post no expected starter)."),
        "arms": {n: {"features": cols,
                     "sealed_model_platt": sealed[n],
                     "pooled_model_platt": pooled[n],
                     "members": {m: dict(v) for m, v in
                                 (results[n].get("members") or {}).items()},
                     "members_sealed": {m: dict(v) for m, v in
                                 (results[n].get("members_sealed") or {}).items()}}
                 for n, cols in arms.items()},
        "skipped_conditional": [n for n in CONDITIONAL_ARMS if n not in arms],
        "verdicts": {n: verdicts[n] for n, _ in pairs},
    }
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"nfl_tier4_ablation_{record['frame_sha256']}.json"
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
