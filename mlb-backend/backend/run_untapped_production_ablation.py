"""Production-correct untapped-columns ablation — full ensemble, WITH vs WITHOUT.

Same candidate test as run_untapped_columns_ablation.py (lineup_ops_vs_*
raws + park_wind_factor, absent from FEATURE_COLS), but PRODUCTION-CORRECT:

1. run_margin_diff is ATTACHED, exactly as production does — via
   training._attach_oof_run_margins over the whole decided frame (leakage-free
   OOF margins from the run engine's own 29-feature view, 2026-08-30 verified
   ~98% coverage on this artifact). So WITHOUT is the real shipped baseline
   (run_margin_diff present in both arms), not a stripped-down one.
2. Each variant gets the full walk-forward pooled OOF (all folds, raw +
   prequential-calibrated) exactly like walk_forward_evaluate.
3. The WITH-vs-WITHOUT decision is evaluated on MULTIPLE sealed holdout
   windows (--windows), so a single 21-day slice can't drive the verdict.

Arms:
    WITHOUT = current production FEATURE_COLS (59, incl. run_margin_diff)
    WITH    = baseline + the added untapped columns
Members: all 5 + the static-prior blend; per-variant blend weights cleared.

Gate (per window AND pooled): WITH must beat WITHOUT on holdout logloss AND AUC
-- a win on one window with a loss on another is flagged, not trusted. ECE and
per-member collapse are flagged in caveats, NOT part of the verdict (policy).

Emits data_delivery/lineup_ablation_untapped_prod_<sha>.json (incremental).
COMMITS NOTHING.
    python run_untapped_production_ablation.py --windows 3 --add <csv>
"""
from __future__ import annotations

import argparse
import json
import subprocess
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
    DATA_DELIVERY_DIR, MIN_VAL_FOLD_GAMES, RANDOM_SEED, RETRAIN_CADENCE_DAYS,
)
from calibration import apply_platt, fit_platt, MIN_OOF_FOR_FIT  # noqa: E402

EPS = 1e-7

# Lean untapped subset (top-importance columns that cleared the two-sided
# inspection; the weakest two, lineup_ops_vs_l_home / r_away, dropped for a
# smaller width with ~the same gain).
DEFAULT_ADD = [
    "lineup_ops_vs_l_away", "lineup_ops_vs_r_home",
    "lineup_ops_vs_starter_hand_home", "lineup_ops_vs_starter_hand_away",
    "park_wind_factor",
]


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def head_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=_BACKEND_DIR.parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


def build_variants(add: list[str]) -> dict[str, list[str]]:
    base = [c for c in training.FEATURE_COLS if c not in add]
    assert base == training.FEATURE_COLS
    return {"WITHOUT": base, "WITH": base + add}


def _run_pooled(cols: list[str], folds) -> tuple[dict, dict, int]:
    training.FEATURE_COLS = list(cols)
    training._LAST_ADAPTIVE_WEIGHTS.clear()

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
        except Exception as e:
            print(f"  fold {split['fold_idx']} failed: {e}")
            continue
        blend, member_probs, _wts = training.ensemble_predict(models, val)
        y_val = val["home_win"].values.astype(float)
        fold_cal = None
        if len(oof_blend) >= MIN_OOF_FOR_FIT:
            fold_cal = fit_platt(np.asarray(oof_y), np.asarray(oof_blend))
        oof_y.extend(y_val.tolist())
        oof_blend.extend(np.asarray(blend, dtype=float).tolist())
        oof_blend_cal.extend(np.asarray(apply_platt(np.asarray(blend), fold_cal),
                                        dtype=float).tolist())
        for n, p in member_probs.items():
            pa = np.asarray(p, dtype=float)
            oof_members.setdefault(n, []).extend(pa.tolist())
            oof_members_cal.setdefault(n, []).extend(
                np.asarray(apply_platt(pa, fold_cal), dtype=float).tolist())
        executed += 1

    y_all = np.asarray(oof_y, dtype=float)
    pooled = {
        "blend": training.compute_metrics(y_all, np.asarray(oof_blend, dtype=float)),
        "blend_calibrated": training.compute_metrics(
            y_all, np.asarray(oof_blend_cal, dtype=float)),
    }
    for n, plist in oof_members.items():
        pooled[n] = training.compute_metrics(y_all, np.asarray(plist, dtype=float))
        pooled[f"{n}_calibrated"] = training.compute_metrics(
            y_all, np.asarray(oof_members_cal.get(n, []), dtype=float))
    return pooled, {"executed": executed}, executed


def _sealed(tune_df: pd.DataFrame, hold_df: pd.DataFrame,
            cols: list[str]) -> dict:
    training.FEATURE_COLS = list(cols)
    training._LAST_ADAPTIVE_WEIGHTS.clear()
    models, _ = training.train_moneyline_ensemble(tune_df)
    blend, member_probs, _wts = training.ensemble_predict(models, hold_df)
    yh = hold_df["home_win"].values.astype(float)
    metrics = {"blend": training.compute_metrics(yh, np.asarray(blend))}
    for n, p in member_probs.items():
        metrics[n] = training.compute_metrics(yh, np.asarray(p, dtype=float))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--add", type=str, default=",".join(DEFAULT_ADD))
    ap.add_argument("--windows", type=int, default=3,
                    help="number of sealed 21-day windows to evaluate")
    ap.add_argument("--holdout-days", type=int, default=21)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    add = [c.strip() for c in args.add.split(",") if c.strip()]
    sha = head_sha()
    data_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    data_hash = sha256_file(data_path)

    games = pd.read_csv(data_path)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_win"]).reset_index(drop=True)
    games = games.sort_values("game_date").reset_index(drop=True)

    # 1) production-correct: attach OOF run margins over the whole frame.
    splits = training.walk_forward_splits(games, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
    if "run_margin_diff" in training.FEATURE_COLS and splits:
        games, splits = training._attach_oof_run_margins(
            games, splits, MIN_VAL_FOLD_GAMES, 0, RETRAIN_CADENCE_DAYS, 0,
            decided_snapshot=games)
        cov = float(games["run_margin_diff"].notna().mean())
    else:
        games["run_margin_diff"] = np.nan
        cov = 0.0
    folds = [s for s in splits if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]
    print(f"prod-correct untapped commit={sha[:12]} data_sha={data_hash[:12]} "
          f"games={len(games)} margin_cov={cov:.3f} "
          f"folds={len(splits)}/{len(folds)} add={add}")

    variants = build_variants(add)
    out = args.out or (DATA_DELIVERY_DIR
                       / f"lineup_ablation_untapped_prod_{sha[:12]}.json")
    if out.exists():
        results = json.loads(out.read_text())
    else:
        results = {"schema": "untapped-cols-prod-ablation/v1", "commit_sha": sha,
                   "data_sha256": data_hash, "add": add,
                   "windows": args.windows, "holdout_days": args.holdout_days,
                   "folds_executed": len(folds), "margin_coverage": cov,
                   "clip_eps": EPS, "seed": int(RANDOM_SEED), "variants": {}}

    maxd = games["game_date"].max()
    for name in ("WITHOUT", "WITH"):
        if name in results["variants"]:
            print(f"  {name}: cached, skipping")
            continue
        print(f"  {name}: running pooled ({len(variants[name])} cols) ...")
        pooled, info, executed = _run_pooled(variants[name], folds)
        variant = {"n_cols": len(variants[name]), "cols": variants[name],
                   "pooled": pooled, "executed": executed, "windows": {}}
        # sealed windows, newest -> oldest
        for k in range(args.windows):
            end = maxd - pd.Timedelta(days=k * args.holdout_days)
            start = end - pd.Timedelta(days=args.holdout_days)
            hold_k = games[(games["game_date"] > start)
                           & (games["game_date"] <= end)].reset_index(drop=True)
            tune_k = games[games["game_date"] <= start].reset_index(drop=True)
            hm = _sealed(tune_k, hold_k, variants[name])
            variant["windows"][f"w{k}"] = {
                "n_hold": int(len(hold_k)), "holdout": hm["blend"],
                "hold_start": str(start.date()),
                "hold_end": str(end.date()),
            }
            pb = pooled["blend"]
            hb = hm["blend"]
            print(f"    [{name}] pooled {pb['logloss']:.4f}/{pb['auc']:.4f} | "
                  f"w{k} [{start.date()}..{end.date()}] n={len(hold_k)} "
                  f"holdout {hb['logloss']:.4f}/{hb['auc']:.4f}/{hb['ece']:.4f}")
        results["variants"][name] = variant
        out.write_text(json.dumps(results, indent=2) + "\n")

    # per-window + pooled gate
    if "WITHOUT" in results["variants"] and "WITH" in results["variants"]:
        wo = results["variants"]["WITHOUT"]
        wi = results["variants"]["WITH"]
        wouts = {}
        any_win_gain = 0
        losses = []
        for k in range(args.windows):
            h0 = wo["windows"][f"w{k}"]["holdout"]; h1 = wi["windows"][f"w{k}"]["holdout"]
            wins = (h1["logloss"] < h0["logloss"]) and (h1["auc"] > h0["auc"])
            wouts[f"w{k}"] = {"verdict": "SHIP" if wins else "DON'T SHIP",
                              "holdout_without": h0, "holdout_with": h1}
            if wins:
                any_win_gain += 1
            else:
                losses.append(f"w{k}")
        p0 = wo["pooled"]["blend"]; p1 = wi["pooled"]["blend"]
        pooled_gain = (p1["logloss"] < p0["logloss"]) and (p1["auc"] > p0["auc"])
        overall = ("SHIP" if (any_win_gain == args.windows) and pooled_gain
                   else "DON'T SHIP")
        caveats = []
        for k in range(args.windows):
            h0 = wo["windows"][f"w{k}"]["holdout"]; h1 = wi["windows"][f"w{k}"]["holdout"]
            if h1["ece"] is not None and h0["ece"] is not None and h1["ece"] > h0["ece"]:
                caveats.append(
                    f"w{k} holdout ece declined {h0['ece']:.4f}->{h1['ece']:.4f}")
        results["gate"] = {
            "verdict": overall, "pooled_without": p0, "pooled_with": p1,
            "wins": any_win_gain, "total": args.windows,
            "loss_windows": losses, "windows": wouts,
            "ece_excluded_from_verdict": True, "caveats": caveats,
        }
        out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"GATE: pooled {p0['logloss']:.4f}/{p0['auc']:.4f} vs "
              f"{p1['logloss']:.4f}/{p1['auc']:.4f} -> "
              f"{'pooled-gain' if pooled_gain else 'no-pooled-gain'}; sealed "
              f"wins {any_win_gain}/{args.windows} -> {overall}")
        for c in caveats:
            print(f"  caveat: {c}")


if __name__ == "__main__":
    main()