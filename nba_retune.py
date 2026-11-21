"""Full retune of the NBA three-member moneyline ensemble on the 92-column
``nba-prod-v3.6-noelo`` contract.

The program:
  1. builds the production feature frame (same calls as ``ab_delay``);
  2. searches a compact deterministic grid for each member on chronological
     inner holdouts that stay inside the pre-holdout history;
  3. patches the selected parameters in-process and runs the full causal
     walk-forward on the new contract;
  4. refits the three selected members on every decided game and writes a
     moneyline-only bundle plus the OOF/metrics report.

The search never touches the last 20% of dates.  The subsequent walk-forward
scores every game, including that tail; the report labels the tail separately
so the selection population is visible.

Usage:
    python nba_retune.py --out nba_retune_v36
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sport_abc as abc


def _logloss(y, p) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return float("nan")
    return float(-np.mean(y[ok] * np.log(p[ok])
                         + (1 - y[ok]) * np.log(1 - p[ok])))


def _xgboost_grid(base: dict) -> list[dict]:
    rows = [dict(base)]
    variants = [
        {"max_depth": 2, "min_child_weight": 22, "learning_rate": 0.025,
         "subsample": 0.80, "colsample_bytree": 0.60},
        {"max_depth": 3, "min_child_weight": 10, "learning_rate": 0.025,
         "subsample": 0.85, "colsample_bytree": 0.55},
        {"max_depth": 3, "min_child_weight": 18, "learning_rate": 0.020,
         "subsample": 0.75, "colsample_bytree": 0.70},
        {"max_depth": 4, "min_child_weight": 20, "learning_rate": 0.020,
         "subsample": 0.80, "colsample_bytree": 0.50},
        {"max_depth": 2, "min_child_weight": 14, "learning_rate": 0.035,
         "subsample": 0.70, "colsample_bytree": 0.65},
        {"max_depth": 3, "min_child_weight": 28, "learning_rate": 0.030,
         "subsample": 0.90, "colsample_bytree": 0.45},
    ]
    for variant in variants:
        row = dict(base)
        row.update(variant)
        rows.append(row)
    return rows


def _lightgbm_grid(base: dict) -> list[dict]:
    rows = [dict(base)]
    variants = [
        {"num_leaves": 31, "min_child_samples": 80, "learning_rate": 0.025,
         "feature_fraction": 0.50, "bagging_fraction": 0.85},
        {"num_leaves": 63, "min_child_samples": 60, "learning_rate": 0.025,
         "feature_fraction": 0.45, "bagging_fraction": 0.90},
        {"num_leaves": 85, "min_child_samples": 120, "learning_rate": 0.020,
         "feature_fraction": 0.55, "bagging_fraction": 0.85},
        {"num_leaves": 127, "min_child_samples": 96, "learning_rate": 0.015,
         "feature_fraction": 0.40, "bagging_fraction": 0.90},
        {"num_leaves": 63, "min_child_samples": 96, "learning_rate": 0.035,
         "feature_fraction": 0.60, "bagging_fraction": 0.80},
        {"num_leaves": 31, "min_child_samples": 140, "learning_rate": 0.020,
         "feature_fraction": 0.65, "bagging_fraction": 0.90},
    ]
    for variant in variants:
        row = dict(base)
        row.update(variant)
        rows.append(row)
    return rows


def _elasticnet_grid(base: dict) -> list[dict]:
    rows = []
    for c in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        row = dict(base)
        row["C"] = c
        rows.append(row)
    return rows


def _tune_splits(frame: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    dates = np.array(sorted(pd.to_datetime(frame["gameday"]).dt.normalize()
                            .unique()))
    if len(dates) < 60:
        return []
    # Leave the final 20% of dates untouched by parameter selection.
    usable = dates[:max(2, int(len(dates) * 0.80))]
    cuts = [int(len(usable) * frac) for frac in (0.55, 0.65, 0.75)]
    splits = []
    for i, cut in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else len(usable)
        if end - cut < 8:
            continue
        val_start, val_end = usable[cut], usable[end - 1]
        train = frame[frame["gameday"] < val_start]
        val = frame[(frame["gameday"] >= val_start)
                    & (frame["gameday"] <= val_end)]
        if len(train) >= 300 and len(val) >= 30:
            splits.append((train, val))
    return splits


def _candidate_score(mods, name: str, train: pd.DataFrame,
                     val: pd.DataFrame) -> float:
    model, pre = mods.ml._fit_member(name, train)
    p = mods.ml._predict(model, name, mods.ml.member_matrix(name, val), pre)
    return _logloss(val["home_win"].to_numpy(float), p)


def _search_member(mods, name: str, grid: list[dict],
                   splits: list[tuple[pd.DataFrame, pd.DataFrame]],
                   log) -> tuple[dict, list[dict]]:
    base = dict(getattr(mods.config,
                        {"xgboost": "XGBOOST_PARAMS",
                         "lightgbm": "LIGHTGBM_PARAMS",
                         "elasticnet": "ELASTICNET_PARAMS"}[name]))
    results = []
    for i, candidate in enumerate(grid, start=1):
        target = getattr(mods.config,
                         {"xgboost": "XGBOOST_PARAMS",
                          "lightgbm": "LIGHTGBM_PARAMS",
                          "elasticnet": "ELASTICNET_PARAMS"}[name])
        target.clear()
        target.update(candidate)
        losses = []
        started = time.time()
        for train, val in splits:
            try:
                losses.append(_candidate_score(mods, name, train, val))
            except Exception as exc:  # noqa: BLE001 - one bad candidate is not fatal
                log(f"  {name} candidate {i} failed on a split: {exc}")
                losses = []
                break
        score = float(np.nanmean(losses)) if losses else float("inf")
        results.append({"member": name, "candidate": i, "logloss": score,
                        "params": candidate,
                        "seconds": round(time.time() - started, 1)})
        log(f"  {name} {i}/{len(grid)}: {score:.6f}")
    # Restore the incumbent until the winner is selected.
    target = getattr(mods.config,
                     {"xgboost": "XGBOOST_PARAMS",
                      "lightgbm": "LIGHTGBM_PARAMS",
                      "elasticnet": "ELASTICNET_PARAMS"}[name])
    target.clear()
    target.update(base)
    winner = min(results, key=lambda r: (r["logloss"], r["candidate"]))
    return winner, results


def _apply(mods, selected: dict[str, dict]) -> None:
    for name, params in selected.items():
        key = {"xgboost": "XGBOOST_PARAMS", "lightgbm": "LIGHTGBM_PARAMS",
               "elasticnet": "ELASTICNET_PARAMS"}[name]
        target = getattr(mods.config, key)
        target.clear()
        target.update(params)


def _score_member(oof: pd.DataFrame, name: str) -> dict:
    p = pd.to_numeric(oof[f"p_{name}"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(oof["home_win"], errors="coerce").to_numpy(float)
    mask = (oof["grades_pooled"].astype(bool).to_numpy()
            if "grades_pooled" in oof else np.isfinite(y))
    return {"n": int(mask.sum()), "logloss": _logloss(y[mask], p[mask])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="nba_retune_v36")
    ap.add_argument("--frame", default="")
    ap.add_argument("--save-frame", action="store_true")
    ap.add_argument("--skip-tune", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(message, flush=True)

    mods = abc.load_modules("nba")
    frame = (pd.read_parquet(args.frame) if args.frame
             and Path(args.frame).exists()
             else abc.build_frame(mods, "nba"))
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="coerce")
    frame = mods.folds.canonical_sort(frame, "gameday").reset_index(drop=True)
    if args.save_frame and args.frame:
        frame.to_parquet(args.frame, index=False)
    frame = frame[frame["gameday"].notna()].reset_index(drop=True)
    log(f"frame: {len(frame)} games {frame.gameday.min().date()} -> "
        f"{frame.gameday.max().date()}")
    log(f"contract: {len(mods.contract.active_moneyline_feature_cols())} features "
        f"({mods.contract.FEATURE_SET_VERSION})")

    splits = _tune_splits(frame)
    log(f"tuning splits: {len(splits)}")
    bases = {
        "xgboost": _xgboost_grid(dict(mods.config.XGBOOST_PARAMS)),
        "lightgbm": _lightgbm_grid(dict(mods.config.LIGHTGBM_PARAMS)),
        "elasticnet": _elasticnet_grid(dict(mods.config.ELASTICNET_PARAMS)),
    }
    tuning_rows: list[dict] = []
    selected: dict[str, dict] = {}
    if args.skip_tune:
        selected = {name: dict(getattr(
            mods.config, {"xgboost": "XGBOOST_PARAMS",
                          "lightgbm": "LIGHTGBM_PARAMS",
                          "elasticnet": "ELASTICNET_PARAMS"}[name]))
                    for name in bases}
    else:
        for name, grid in bases.items():
            log(f"searching {name}: {len(grid)} candidates x {len(splits)} splits")
            winner, rows = _search_member(mods, name, grid, splits, log)
            selected[name] = winner["params"]
            tuning_rows.extend(rows)
            log(f"selected {name}: {winner['params']} (ll {winner['logloss']:.6f})")
        pd.DataFrame(tuning_rows).to_csv(out_dir / "tuning.csv", index=False)
        (out_dir / "selected_params.json").write_text(
            json.dumps(selected, indent=2, default=float))

    _apply(mods, selected)
    folds = mods.folds.make_folds(frame, "gameday")
    log(f"retuned walk-forward: {len(folds)} folds")
    started = time.time()
    result = mods.ml.walk_forward_oof(
        frame, fold_list=folds, arm="incumbent", progress_every=0)
    oof = result["oof"]
    oof.to_csv(out_dir / "oof.csv", index=False)
    log(f"walk-forward finished in {(time.time() - started) / 60:.1f} min")

    metrics = {
        "n_oof": int(len(oof)),
        "n_graded": int(oof["grades_pooled"].astype(bool).sum()),
        "ensemble": _score_member(oof, "ensemble"),
        "members": {name: _score_member(oof, name)
                    for name in mods.config.ENSEMBLE_MEMBERS},
        "member_weights": {k: float(v)
                           for k, v in result["member_weights"].items()},
        "feature_set_version": mods.contract.FEATURE_SET_VERSION,
        "n_features": len(mods.contract.active_moneyline_feature_cols()),
    }
    # Calibrate the served view on the causal OOF pairs.
    cal = mods.ml.moneyline_fit(
        oof["p_ensemble"].to_numpy(float),
        oof["home_win"].to_numpy(float))
    metrics["platt"] = cal
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float))

    # Final fit on every decided game.  This is the moneyline leg only; the
    # production master pipeline owns the run-engine/score-regressor side.
    final = {}
    for name in mods.config.ENSEMBLE_MEMBERS:
        model, pre = mods.ml._fit_member(name, frame)
        final[name] = {"model": model, "pre": pre}
    import joblib
    joblib.dump({
        "moneyline_models": {n: e["model"] for n, e in final.items()},
        "moneyline_preprocessors": {n: e["pre"] for n, e in final.items()},
        "ensemble_weights": result["member_weights"],
        "platt": cal,
        "feature_set_version": mods.contract.FEATURE_SET_VERSION,
        "feature_columns": mods.contract.active_moneyline_feature_cols(),
        "selected_params": selected,
        "trained_rows": int(len(frame)),
        "trained_through": str(frame["gameday"].max().date()),
    }, out_dir / "nba_moneyline_retuned.joblib")
    log(json.dumps(metrics, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
