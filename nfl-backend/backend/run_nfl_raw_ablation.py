"""Focused NFL raw-columns ablation — do raw per-side columns recover signal
the diff encoding discards?

The served 12-feature pool is encoded as home-minus-away DIFFS. This harness
asks whether feeding the TREE members the raw per-side values (elo_home,
elo_away, ...) — which the production ladder already computes before diffing
(``nfl_raw_columns``, same strictly-prior PIT rules, zero new leak surface) —
recovers signal the diff encoding throws away. It informs the Tier-5
player-level encoding decision (diff-first vs raw-per-side).

Arms (same decided-game frame as the Tier-4 harness):
  C0         — the deployed 12-feature diff pool (FEATURE_COLUMNS minus the
               is_home anchor), run with the PRODUCTION
               ``nfl_moneyline.run_walk_forward`` (zero divergence).
  RAW_ADDED  — C0 + the 14 raw per-side columns. Member handling mirrors
               MLB's LOGISTIC_USE_RAW_COLS=False: the logistic member (linear
               head) receives ONLY diffs + flags (C0), while the tree members
               (xgb/lgb/rf) AND mlp receive raws + diffs. The linear head is
               z-scored with train mu/sd (StandardScaler fit per fold on the
               train split only, consistent across folds).

Per-member pooled + sealed logloss/auc tables are reported for BOTH arms from
the start (f01f880 tier pattern) — the key read is WHICH member moves when
raws are added. The adoption gate is the SAME rule as Tier-1/2/3/4
(run_tier1_ablation.adopt_verdict): RAW_ADDED must beat C0 on the SEALED 2025
hold-out in logloss AND AUC without degrading ECE-cal; pooled OOF logloss
corroborates. Adoption is a SEPARATE decision — a win keeps the raws
composed-but-unregistered and records the verdict for Tier-5; a loss records
that the diff encoding is not the bottleneck and Tier-5 proceeds diff-first.

Usage (network + nflreadpy needed for the raw pull):
    python3 run_nfl_raw_ablation.py
    python3 run_nfl_raw_ablation.py --features <features.csv>
    python3 run_nfl_raw_ablation.py --arm C0          # single arm (time-boxing)
    python3 run_nfl_raw_ablation.py --no-record       # report only
Artifact: data_delivery/nfl_raw_ablation_<sha>.json (reviewed before any
commit; the evidence record is committed with the harness per convention).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_raw_columns import RAW_PER_SIDE_COLS, compose_raw_columns, raw_coverage
from run_tier1_ablation import (MEMBER_NAMES, _frame_sha256, _member_metrics,
                                adopt_verdict)
from run_feature_winpct_ablation import DEPLOYED_12

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"

# The deployed 12-feature pool (FEATURE_COLUMNS minus is_home), imported so the
# harness always tests the TRUE served pool.
BASELINE_12 = list(DEPLOYED_12)

# Logistic receives ONLY the diff pool (mirror MLB LOGISTIC_USE_RAW_COLS=False);
# the tree members (and mlp) receive the raws too.
TREE_MEMBERS = ("xgboost", "lightgbm", "randomforest", "mlp")
RAW_ARMS = ("C0", "RAW_ADDED")


def load_features(features_csv: str | None) -> pd.DataFrame:
    """Feature frame: a provided CSV, else the nflreadpy pull + build.

    Mirrors run_tier4_ablation.load_features, then attaches the raw per-side
    columns via ``compose_raw_columns`` (the SAME ladder the diffs come from).
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
                            + TIER1_NEEDS)
                if c in pbp.columns]
        pbp = pbp.select(keep).to_pandas()
        decided = pd.read_csv(DECIDED_FRAME)
        decided = decided[decided["season"].isin(seasons)]
        feats = build_features(decided, sched, pbp)
        feats["_decided"] = feats["game_id"].isin(set(_decided_rows(sched)["game_id"]))
        # Attach the raw per-side columns (same ladder the diffs come from).
        feats = compose_raw_columns(feats, sched, pbp)
    if "home_win" not in feats.columns:
        feats["home_win"] = (feats["home_score"] > feats["away_score"]).astype(int)
    return feats


def build_arms(feats: pd.DataFrame) -> dict[str, list[str]]:
    """Column lists per arm, kept only where the frame carries them."""
    base = [c for c in BASELINE_12 if c in feats.columns]
    raws = [c for c in RAW_PER_SIDE_COLS if c in feats.columns]
    return {"C0": base, "RAW_ADDED": base + raws}


def _member_plan(feats: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(tree_cols, logistic_cols): trees get raws + diffs; logistic diffs only."""
    arms = build_arms(feats)
    return arms["RAW_ADDED"], arms["C0"]


def run_walk_forward_masked(feats: pd.DataFrame,
                            tree_cols: list[str],
                            logistic_cols: list[str]) -> dict:
    """Prequential walk-forward with PER-MEMBER feature masks.

    The production ``run_walk_forward`` trains every member on one shared
    column list; this variant trains the tree members (xgb/lgb/rf) and mlp on
    ``tree_cols`` (raws + diffs) and the logistic member on ``logistic_cols``
    (diffs + flags only), then merges the member probabilities into the same
    blend/Platt/adaptive-weight machinery. Everything else mirrors
    ``run_walk_forward`` exactly (sealed-2025 never touches fitting; the
    sealed Platt map is fit only on the pooled pre-holdout OOF). Return dict
    shape is the same key set the tier harnesses consume.
    """
    from nfl_run_engine_legacy_windows import TRAIN_SEASONS, SEALED_SEASON, generate_weekly_folds
    from nfl_moneyline import (TARGET, META_COLS,
                               _adaptive_blend, _elo_logistic_p, _member_weights,
                               _score_member_table, _valid_rows, auc, ece,
                               compute_adaptive_weights, compute_metrics,
                               ensemble_predict,
                               logloss, platt_fit, platt_predict,
                               train_ensemble)

    preq_all = feats[feats["season"].isin(TRAIN_SEASONS)].copy()
    sealed = feats[feats["season"] == SEALED_SEASON].copy()

    tree = [c for c in tree_cols if c in feats.columns]
    logi = [c for c in logistic_cols if c in feats.columns]
    if not tree or not logi:
        raise ValueError("masked walk-forward needs both column sets present")

    # Universe = rows with ALL tree features + target (tree set ⊇ logistic
    # set, so this is the stricter mask — the fair common universe).
    preq = preq_all[_valid_rows(preq_all, tree)].copy()
    sld = sealed[_valid_rows(sealed, tree)].copy()
    folds = generate_weekly_folds(preq)

    order_actual, order_raw, order_elo, ws_list = [], [], [], []
    oof_members: dict[str, list[float]] = {}
    oof_members_cal: dict[str, list[float]] = {}
    cal_pool, raw_pool, elo_pool, y_pool = [], [], [], []

    for f in folds:
        tr, va = f["train"], f["val"]
        yva = va[TARGET].to_numpy(dtype=float)
        try:
            models_tree, _ = train_ensemble(tr, va, features=tree)
            _, members_tree, _ = ensemble_predict(models_tree, va, features=tree)
            models_logi, _ = train_ensemble(tr, va, features=logi)
            _, members_logi, _ = ensemble_predict(models_logi, va, features=logi)
        except Exception as e:  # noqa: BLE001 — fold-level skip, as production
            logger.warning("fold %s ensemble failed: %s", f["week_start"], e)
            continue
        # Merge: trees + mlp from the tree-set ensemble; logistic from the
        # diff-only ensemble (its raw-trained twin is discarded).
        member_probs = {n: members_tree[n] for n in members_tree if n != "logistic"}
        if "logistic" in members_logi:
            member_probs["logistic"] = members_logi["logistic"]
        names = [n for n in MEMBER_NAMES if n in member_probs]
        wts = _member_weights(names)
        blend = np.zeros(len(yva))
        for n in names:
            blend += wts[n] * np.asarray(member_probs[n], dtype=float)
        elo_p = _elo_logistic_p(tr, va, logi)

        lr = None
        if y_pool:
            lr = platt_fit(np.concatenate(raw_pool),
                           np.concatenate(y_pool).astype(int))
            cal_p = platt_predict(blend, lr)
        else:
            cal_p = blend.copy()
        for name, p in member_probs.items():
            p_arr = np.asarray(p, dtype=float)
            oof_members.setdefault(name, []).extend(p_arr.tolist())
            pc = platt_predict(p_arr, lr) if lr is not None else p_arr
            oof_members_cal.setdefault(name, []).extend(pc.tolist())

        order_actual.append(yva)
        order_raw.append(blend)
        order_elo.append(elo_p)
        ws_list.append(f["week_start"])
        cal_pool.append(cal_p)
        raw_pool.append(blend)
        elo_pool.append(elo_p)
        y_pool.append(yva)

    if not y_pool:
        raise RuntimeError("no folds produced ensemble predictions")

    y_po = np.concatenate(y_pool)
    raw_po = np.concatenate(raw_pool)
    cal_po = np.concatenate(cal_pool)
    elo_po = np.concatenate(elo_pool)
    const_p = preq[TARGET].mean()

    pooled = {
        "n": int(len(y_po)),
        "fold_count": len(folds),
        "constant_home_edge": {
            "proba": round(float(const_p), 4),
            "logloss": round(logloss(y_po, np.full_like(y_po, const_p)), 4),
            "auc": round(auc(y_po, np.full_like(y_po, const_p)), 4),
        },
        "elo_logistic": {
            "logloss": round(logloss(y_po, elo_po), 4),
            "auc": round(auc(y_po, elo_po), 4),
        },
        "model_raw": {
            "logloss": round(logloss(y_po, raw_po), 4),
            "auc": round(auc(y_po, raw_po), 4),
        },
        "model_platt": {
            "logloss": round(logloss(y_po, cal_po), 4),
            "auc": round(auc(y_po, cal_po), 4),
            "ece": round(ece(y_po, cal_po), 4),
        },
    }

    adaptive = compute_adaptive_weights(oof_members, y_po)
    members_table = {}
    for name in sorted(set(oof_members)):
        raw_p = np.asarray(oof_members[name], dtype=float)
        entry = {"weight": float(adaptive.get(name, 0.0))}
        if len(raw_p) == len(y_po):
            m = compute_metrics(y_po, raw_p)
            entry.update({k: m[k] for k in ("logloss", "auc", "ece", "brier")})
        if len(oof_members_cal.get(name, [])) == len(y_po):
            mc = compute_metrics(y_po, np.asarray(oof_members_cal[name], dtype=float))
            entry.update({"logloss_calibrated": mc["logloss"],
                          "auc_calibrated": mc["auc"],
                          "ece_calibrated": mc["ece"]})
        members_table[name] = entry

    # ---- SEALED 2025 (fit-only refit on all 2019-2024, never 2025) --------
    models_sealed_tree, _ = train_ensemble(preq, None, features=tree)
    _, sealed_members_tree, _ = ensemble_predict(
        models_sealed_tree, sld, features=tree)
    models_sealed_logi, _ = train_ensemble(preq, None, features=logi)
    _, sealed_members_logi, _ = ensemble_predict(
        models_sealed_logi, sld, features=logi)
    sealed_members = {n: p for n, p in sealed_members_tree.items() if n != "logistic"}
    if "logistic" in sealed_members_logi:
        sealed_members["logistic"] = sealed_members_logi["logistic"]

    sealed_raw = np.zeros(len(sld))
    for n in MEMBER_NAMES:
        if n in sealed_members:
            sealed_raw += adaptive.get(n, 0.0) * np.asarray(sealed_members[n],
                                                            dtype=float)
    sealed_elo = _elo_logistic_p(preq, sld, logi)

    oof_adaptive_blend = _adaptive_blend(oof_members, adaptive, len(y_po))
    platt_sealed = platt_fit(oof_adaptive_blend, y_po.astype(int))
    sealed_cal = platt_predict(sealed_raw, platt_sealed)
    const_sealed = preq[TARGET].mean()
    sealed_members_table = _score_member_table(sld[TARGET].to_numpy(),
                                               sealed_members)

    sealed = {
        "n": int(len(sld)),
        "constant_home_edge": {
            "proba": round(float(const_sealed), 4),
            "logloss": round(logloss(sld[TARGET], np.full(len(sld), const_sealed)), 4),
            "auc": round(auc(sld[TARGET], np.full(len(sld), const_sealed)), 4),
        },
        "elo_logistic": {
            "logloss": round(logloss(sld[TARGET], sealed_elo), 4),
            "auc": round(auc(sld[TARGET], sealed_elo), 4),
        },
        "model_raw": {
            "logloss": round(logloss(sld[TARGET], sealed_raw), 4),
            "auc": round(auc(sld[TARGET], sealed_raw), 4),
        },
        "model_platt": {
            "logloss": round(logloss(sld[TARGET], sealed_cal), 4),
            "auc": round(auc(sld[TARGET], sealed_cal), 4),
            "ece": round(ece(sld[TARGET], sealed_cal), 4),
        },
    }
    _sm = [c for c in META_COLS if c in sld.columns]
    return {
        "fold_geometry": {
            "train_seasons": TRAIN_SEASONS,
            "sealed_season": SEALED_SEASON,
            "fold_count": len(folds),
            "pooled_oof_games": int(len(y_po)),
            "sealed_games": int(len(sld)),
        },
        "pooled_preq_2021_2024": pooled,
        "sealed_2025": sealed,
        "adaptive_weights": adaptive,
        "members": members_table,
        "members_sealed": sealed_members_table,
        "_deployed": {"features": tree, "logistic_features": logi},
    }


def _coverage_report(feats: pd.DataFrame) -> None:
    print("Raw per-side coverage (decided frame):")
    for c, pct in raw_coverage(feats).items():
        print(f"  {c:24s} {pct:6.1f}%")


def _member_mover(blk: dict[str, dict], metric: str) -> dict:
    """Per-member RAW_ADDED − C0 delta on a metric; the biggest mover first."""
    rows = []
    for m in MEMBER_NAMES:
        a = (blk["C0"].get(m) or {}).get(metric)
        b = (blk["RAW_ADDED"].get(m) or {}).get(metric)
        if a is None or b is None:
            continue
        rows.append({"member": m, "c0": a, "raw_added": b,
                     "delta": round(b - a, 4)})
    rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=None,
                    help="pre-built features CSV (skips the nflreadpy pull)")
    ap.add_argument("--arm", choices=RAW_ARMS, default=None,
                    help="run a single arm (time-boxing); default runs both")
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

    tree_cols, logi_cols = _member_plan(feats)
    print(f"logistic sees: {logi_cols} (no raws)")
    print(f"trees/mlp see: {tree_cols}")

    todo = [a for a in RAW_ARMS if args.arm is None or a == args.arm]
    results = {}
    for name in todo:
        print(f"\n=== running walk-forward arm {name} ===")
        if name == "C0":
            results[name] = run_walk_forward(feats, model_features=arms[name])
        else:
            results[name] = run_walk_forward_masked(
                feats, tree_cols=arms[name], logistic_cols=arms["C0"])

    def _m(rec: dict) -> dict:
        return {k: rec.get(k) for k in ("logloss", "auc", "ece")}

    sealed = {n: _m(results[n]["sealed_2025"]["model_platt"]) for n in todo}
    pooled = {n: _m(results[n]["pooled_preq_2021_2024"]["model_platt"])
              for n in todo}

    print("\n=== raw-columns ablation (RAW_ADDED vs C0) ===")
    print("arm           sealed_ll  sealed_auc  sealed_ece  pooled_ll")
    for n in todo:
        s, p = sealed[n], pooled[n]
        print(f"{n:14s} {s['logloss']}  {s['auc']}  {s['ece']}  {p['logloss']}")

    member_pooled = {n: _member_metrics(results[n], "members") for n in todo}
    member_sealed = {n: _member_metrics(results[n], "members_sealed")
                     for n in todo}

    def _member_rows(blk: dict[str, dict]) -> None:
        print(f"{'member':12s}" + "".join(f"{n:>17s}" for n in todo))
        for m in MEMBER_NAMES:
            cells = []
            for n in todo:
                e = blk[n].get(m) or {}
                cells.append(f"{e.get('logloss', '--')}/{e.get('auc', '--')}")
            print(f"{m:12s}" + "".join(f"{c:>17s}" for c in cells))

    print("\n=== per-member pooled OOF (logloss/auc) ===")
    _member_rows(member_pooled)
    print("\n=== per-member sealed 2025 (logloss/auc) ===")
    _member_rows(member_sealed)

    if len(todo) == 2:
        v = adopt_verdict(sealed["C0"], sealed["RAW_ADDED"],
                          pooled["C0"], pooled["RAW_ADDED"])
        print("\nVERDICT (RAW_ADDED vs C0):",
              "ADOPT (keep composed-but-unregistered for Tier-5)"
              if v["adopt"] else
              "DON'T ADOPT (diff encoding is not the bottleneck — "
              "Tier-5 proceeds diff-first)",
              "|", " | ".join(v["reasons"]))
        print("\n=== which member moved (pooled OOF) ===")
        for r in _member_mover(member_pooled, "logloss"):
            print(f"  {r['member']:12s} c0 {r['c0']} -> raw {r['raw_added']} "
                  f"(d {r['delta']:+.4f})")
        print("\n=== which member moved (sealed 2025) ===")
        for r in _member_mover(member_sealed, "logloss"):
            print(f"  {r['member']:12s} c0 {r['c0']} -> raw {r['raw_added']} "
                  f"(d {r['delta']:+.4f})")

    if args.no_record or len(todo) != 2:
        return 0
    verdict = adopt_verdict(sealed["C0"], sealed["RAW_ADDED"],
                            pooled["C0"], pooled["RAW_ADDED"])
    record = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "frame_sha256": _frame_sha256(feats),
        "baseline": "deployed 12-feature pool (FEATURE_COLUMNS minus is_home)",
        "notes": ("Raw per-side columns (composed-but-unregistered) re-emit "
                  "the per-side values the production ladder already computes "
                  "before diffing — same strictly-prior PIT rules, no new "
                  "leak surface. Logistic receives ONLY diffs + flags "
                  "(LOGISTIC_USE_RAW_COLS=False); trees (xgb/lgb/rf) and mlp "
                  "receive raws AND diffs. Linear head z-scored with train "
                  "mu/sd per fold."),
        "raw_columns": RAW_PER_SIDE_COLS,
        "raw_coverage_pct": raw_coverage(feats),
        "member_masks": {"trees_mlp": tree_cols, "logistic": logi_cols},
        "arms": {n: {"features": arms[n],
                     "sealed_model_platt": sealed[n],
                     "pooled_model_platt": pooled[n],
                     "members": {m: dict(v) for m, v in
                                 (results[n].get("members") or {}).items()},
                     "members_sealed": {m: dict(v) for m, v in
                                 (results[n].get("members_sealed") or {}).items()}}
                 for n in todo},
        "verdict": verdict,
    }
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"nfl_raw_ablation_{record['frame_sha256']}.json"
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
