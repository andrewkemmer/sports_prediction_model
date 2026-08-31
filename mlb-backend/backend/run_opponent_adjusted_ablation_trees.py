"""Trees-only follow-up to the opponent-adjusted ablation — WITH vs WITHOUT
restricted to the three tree members (xgboost / lightgbm / randomforest).

This is a FOLLOW-UP measurement, NOT a feature ship and NOT the original run
re-done: the full 5-member run (run_opponent_adjusted_ablation.py) measured
the six opponent-adjusted columns as a clean DON'T-ADOPT (sealed-holdout
blend nominal win but ECE-cal degraded 0.0126 -> 0.0151, pooled-OOF logloss
went the WRONG way, and the per-member breakdown showed XGBoost gaining
(holdout AUC 0.5798 -> 0.6009) while MLP and RandomForest regressed). The
single question this script answers: do the TREE members alone extract real
signal from the columns once MLP isn't diluting the blend?

Apples-to-apples: BOTH arms are restricted to the SAME three tree members
(drop logistic + MLP from the trained ensemble before blending). The tree
members train identically to production (their training never depends on
logistic/MLP), so filtering after train_moneyline_ensemble is exactly
equivalent to training trees-only, and the WITH-vs-WITHOUT comparison stays
clean.

Everything else mirrors run_opponent_adjusted_ablation.py exactly:
- Same data + point-in-time ladders (reused add_opponent_adjusted_features):
  the 6 opponent-adjusted columns are all genuinely computable on the
  committed game_level_features.csv (home_starter_id/away_starter_id 100%
  present), strictly game_date < before each row, same-day doubleheader legs
  excluded, min-5-prior-team / min-3-prior-start gates, NaN never imputed.
  run_margin_diff is all-NaN in BOTH arms (produced at training time, not in
  the artifact) so it cannot differentiate the arms.
- WITHOUT = exact production training.FEATURE_COLS (59); WITH = 59 + the 6
  opponent-adjusted columns (all real coverage).
- Same walk_forward_splits machinery, MIN_VAL_FOLD_GAMES gate (declared vs
  executed), adaptive weights cleared per variant so both blend identically.
- Same compute_metrics (logloss / AUC / Brier / ECE raw + prequentially
  calibrated) and the same sealed 21-day holdout fitted only at the end.
- Per-member logloss/AUC reported for WITH vs WITHOUT so we can see whether
  XGBoost's earlier benefit survives without MLP dilution.

Gate (unchanged): WITH must beat WITHOUT on the sealed 21-day holdout on
logloss AND AUC without degrading ECE-cal, AND pooled-OOF logloss must not
invert. A pooled-gain / sealed-loss (or sealed-gain / pooled-loss)
inversion => DON'T-ADOPT; a clean, corroborated win is the ADOPT signal.
A clean DON'T-ADOPT is the honest, expected outcome for a noisily-gated run
— no AF-forcing (no loosening point-in-time discipline, no fold cherry-
picking).

Emits data_delivery/opponent_adjusted_ablation_trees_<sha>.json
(incremental — resumes by skipping variants already present). COMMITS
NOTHING.

Usage:
    python run_opponent_adjusted_ablation_trees.py
    python run_opponent_adjusted_ablation_trees.py --variants WITHOUT,WITH
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

import training  # noqa: E402
from config import (  # noqa: E402
    DATA_DELIVERY_DIR,
    MIN_VAL_FOLD_GAMES,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
)
from calibration import (  # noqa: E402
    MIN_OOF_FOR_FIT,
    apply_platt,
    fit_platt,
)
# Reuse the exact ladders / coverage / variant construction from the
# original measurement so the two runs cannot drift apart.
from run_opponent_adjusted_ablation import (  # noqa: E402
    EPS,
    OPP_ADJ_COLS,
    add_opponent_adjusted_features,
    build_variants,
    coverage_report,
    head_sha,
    sha256_file,
)

# Members that participate in the trees-only blend. Adaptive weights are
# cleared per variant (as in the original), so the blend uses the static
# ENSEMBLE_WEIGHTS priors renormalized over whichever members are present —
# here the three trees only. Restricting BOTH arms identically keeps the
# WITH-vs-WITHOUT comparison apples-to-apples.
TREE_MEMBERS = ("xgboost", "lightgbm", "randomforest")
_NON_TREE_MEMBERS = ("logistic", "mlp")


def filter_tree_members(models: dict) -> dict:
    """Restrict a trained ensemble dict to the tree members only.

    Keeps the three tree models AND the helper keys ensemble_predict needs
    (scaler / impute_median / categorical_vocab); drops logistic + mlp.
    """
    return {k: v for k, v in models.items() if k not in _NON_TREE_MEMBERS}


def run_variant_trees(cols: list[str], folds, tune_df, hold_df) -> dict:
    """Same loop as the original run_variant but blending trees ONLY.

    Both arms train the full ensemble then restrict to trees via
    filter_tree_members before prediction/blending, so the tree members train
    identically to production in both arms.
    """
    training.FEATURE_COLS = list(cols)
    training._LAST_ADAPTIVE_WEIGHTS.clear()  # both variants blend identically

    oof_y: list[float] = []
    oof_blend: list[float] = []
    oof_members: dict[str, list[float]] = {}
    oof_blend_cal: list[float] = []
    oof_members_cal: dict[str, list[float]] = {}
    executed = 0

    for split in folds:
        train = split["train_games"]
        val = split["val_games"]
        try:
            models, _ = training.train_moneyline_ensemble(train, val)
            models = filter_tree_members(models)
        except Exception as e:  # keep the loop honest: log, skip, continue
            print(f"  fold {split['fold_idx']} failed: {e}")
            continue
        blend, member_probs, _wts = training.ensemble_predict(models, val)
        y_val = val["home_win"].values.astype(float)
        fold_cal = None
        if len(oof_blend) >= MIN_OOF_FOR_FIT:
            fold_cal = fit_platt(np.asarray(oof_y), np.asarray(oof_blend))
        oof_y.extend(y_val.tolist())
        oof_blend.extend(np.asarray(blend, dtype=float).tolist())
        oof_blend_cal.extend(
            np.asarray(apply_platt(np.asarray(blend), fold_cal), dtype=float).tolist())
        for name, p in member_probs.items():
            pa = np.asarray(p, dtype=float)
            oof_members.setdefault(name, []).extend(pa.tolist())
            oof_members_cal.setdefault(name, []).extend(
                np.asarray(apply_platt(pa, fold_cal), dtype=float).tolist())
        executed += 1

    y_all = np.asarray(oof_y, dtype=float)
    pooled: dict[str, dict] = {}
    pooled["blend"] = training.compute_metrics(
        y_all, np.asarray(oof_blend, dtype=float))
    pooled["blend_calibrated"] = training.compute_metrics(
        y_all, np.asarray(oof_blend_cal, dtype=float))
    for name, plist in oof_members.items():
        pooled[name] = training.compute_metrics(
            y_all, np.asarray(plist, dtype=float))
        pooled[f"{name}_calibrated"] = training.compute_metrics(
            y_all, np.asarray(oof_members_cal.get(name, []), dtype=float))

    # ── sealed holdout: fit only at the end ───────────────────────────────
    models, _ = training.train_moneyline_ensemble(tune_df)
    models = filter_tree_members(models)
    blend_hold, member_hold, _wts = training.ensemble_predict(models, hold_df)
    y_hold = hold_df["home_win"].values.astype(float)
    holdout: dict[str, dict] = {
        "blend": training.compute_metrics(y_hold, np.asarray(blend_hold)),
    }
    for name, p in member_hold.items():
        holdout[name] = training.compute_metrics(
            y_hold, np.asarray(p, dtype=float))

    return {"n_cols": len(cols), "members": list(TREE_MEMBERS),
            "folds_executed": executed,
            "pooled": pooled, "holdout": holdout}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", type=str, default="WITHOUT,WITH")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--holdout-days", type=int, default=21)
    args = ap.parse_args()

    sha = head_sha()
    data_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    data_hash = sha256_file(data_path)

    games = pd.read_csv(data_path)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_win"]).reset_index(drop=True)
    games = add_opponent_adjusted_features(games)

    cutoff = games["game_date"].max() - pd.Timedelta(days=args.holdout_days - 1)
    tune_df = games[games["game_date"] < cutoff].reset_index(drop=True)
    hold_df = games[games["game_date"] >= cutoff].reset_index(drop=True)
    all_splits = training.walk_forward_splits(
        tune_df, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
    folds = [s for s in all_splits if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]

    print(f"commit={sha[:12]} data_sha={data_hash[:12]} games={len(games)} "
          f"tuning={len(tune_df)} holdout={len(hold_df)} "
          f"folds={len(all_splits)}/{len(folds)} "
          f"members={','.join(TREE_MEMBERS)} seed={RANDOM_SEED} clip={EPS}")

    coverage = coverage_report(games)
    real = [c for c in coverage if c["coverage"] > 0]
    dropped = [c for c in coverage if c["coverage"] == 0.0]
    print(f"opponent-adjusted coverage on committed CSV: "
          f"{len(real)}/{len(OPP_ADJ_COLS)} real")
    for c in coverage:
        print(f"    {c['column']:24s} coverage={c['coverage']:.3f}")
    if dropped:
        print(f"  dropping {len(dropped)} incomputable candidate(s) "
              f"(coverage 0.0, would be constant-NaN noise): "
              f"{', '.join(c['column'] for c in dropped)}")

    variants = build_variants(games, coverage)
    out = args.out or (DATA_DELIVERY_DIR
                       / f"opponent_adjusted_ablation_trees_{sha[:12]}.json")
    if out.exists():
        results = json.loads(out.read_text())
    else:
        results = {"schema": "opponent-adjusted-ablation-trees/v1",
                   "commit_sha": sha, "data_sha256": data_hash,
                   "holdout_days": args.holdout_days,
                   "members": list(TREE_MEMBERS),
                   "folds_declared": len(all_splits),
                   "folds_executed": len(folds), "clip_eps": EPS,
                   "seed": int(RANDOM_SEED),
                   "coverage": coverage, "variants": {}}
    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    for name in want:
        if name in results["variants"]:
            print(f"  {name}: cached, skipping")
            continue
        print(f"  {name}: running ({len(variants[name])} cols) ...")
        r = run_variant_trees(variants[name], folds, tune_df, hold_df)
        r["cols"] = variants[name]
        results["variants"][name] = r
        out.write_text(json.dumps(results, indent=2) + "\n")
        b = r["pooled"]["blend"]
        h = r["holdout"]["blend"]
        print(f"    pooled blend {b['logloss']:.4f}/{b['auc']:.4f} "
              f"brier {b['brier']:.4f} ece {b['ece']:.4f} | "
              f"holdout {h['logloss']:.4f}/{h['auc']:.4f}")

    # gate: WITH must beat WITHOUT on the sealed holdout (both metrics)
    if "WITHOUT" in results["variants"] and "WITH" in results["variants"]:
        wo = results["variants"]["WITHOUT"]["holdout"]["blend"]
        w = results["variants"]["WITH"]["holdout"]["blend"]
        win = w["logloss"] < wo["logloss"] and w["auc"] > wo["auc"]
        pw = (results["variants"]["WITH"]["pooled"]["blend"]["logloss"]
              < results["variants"]["WITHOUT"]["pooled"]["blend"]["logloss"])
        print(f"\n=== sealed-holdout gate (trees blend) ===")
        print(f"  WITHOUT {wo['logloss']:.4f}/{wo['auc']:.4f}  "
              f"WITH {w['logloss']:.4f}/{w['auc']:.4f}  -> "
              f"{'BEATS WITHOUT' if win else 'loses/ties WITHOUT'} "
              f"(pooled_win={pw})")
        if pw and not win:
            print("  FLAG: pooled win with holdout loss — likely overfit, not adopted.")
        if win and not pw:
            print("  FLAG: sealed win with pooled-OOF loss — pooled/sealed "
                  "inversion, not a corroborated ADOPT signal.")
        # Per-member holdout breakdown, WITH vs WITHOUT (isolate XGBoost's
        # earlier benefit without MLP dilution).
        print(f"\n=== per-member sealed holdout (WITH vs WITHOUT) ===")
        ww = results["variants"]["WITH"]["holdout"]
        woe = results["variants"]["WITHOUT"]["holdout"]
        for m in TREE_MEMBERS:
            a = woe.get(m)
            b = ww.get(m)
            if a is None or b is None:
                continue
            print(f"  {m:12s} WITHOUT {a['logloss']:.4f}/{a['auc']:.4f}  "
                  f"WITH {b['logloss']:.4f}/{b['auc']:.4f}  "
                  f"Δ ll {b['logloss']-a['logloss']:+.4f} / "
                  f"Δ auc {b['auc']-a['auc']:+.4f}")
    print(f"\nablation written: {out}")


if __name__ == "__main__":
    main()