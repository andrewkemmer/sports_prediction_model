"""Offline LGBM-Poisson run-line tuning harness for the NFL (one-shot).

Follows the NHL protocol (nhl-backend, 2026-09-27): Optuna search over the
ScoreRegressor hyperparameters, objective = pooled per-game Poisson deviance
over folds[:-4]; the sealed last-4 folds are the holdout that confirms or
rejects the winner. The feature matrix is the run line's OWN contract
(features.tree_view over the active moneyline list + team-ID pair) built ONCE
from cached pulls; folds are the production walk-forward folds (7-day windows,
expanding train) so every trial sees exactly the data the pipeline sees.

Writes run_line_tuning_<date>.json with the full verdict record.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import folds as folds_mod  # noqa: E402
import features as feat_mod  # noqa: E402
import ingestion  # noqa: E402
import distributions as dist_mod  # noqa: E402

N_FOLDS_HOLDOUT = 4
N_TRIALS = 32
SEED = config.RANDOM_SEED

_FOLD_LIST = None
_FRAME = None


def poisson_dev(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Per-game Poisson deviance: 2*(y*log(y/mu) - y + mu), with the
    standard convention y*log(y/mu) := 0 when y == 0 (so a blanked score
    contributes 2*mu, penalizing over-prediction exactly as the likelihood
    does). Computed without evaluating log(0) on the zero branch."""
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(mu, float), 1e-6, None)
    yl = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return 2.0 * (yl - y + mu)


def build_frame() -> pd.DataFrame:
    global _FRAME
    if _FRAME is not None:
        return _FRAME
    sched = ingestion.load_schedule(seasons=config.ALL_SEASONS, use_cache=True)
    sched = ingestion.eligible_games(sched)
    sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")
    sched = sched[(sched["gameday"] >= pd.Timestamp("2016-01-01"))
                  & (sched["gameday"] <= pd.Timestamp.today())]
    dec = sched[sched["home_score"].notna() & sched["away_score"].notna()].copy()
    pbp = ingestion.load_pbp(seasons=config.ALL_SEASONS, use_cache=True)
    df = feat_mod.build_game_features(dec, pbp)
    _FRAME = folds_mod.canonical_sort(df, "gameday")
    return _FRAME


def get_folds():
    global _FOLD_LIST
    if _FOLD_LIST is None:
        _FOLD_LIST = folds_mod.make_folds(build_frame(), date_col="gameday")
    return _FOLD_LIST


def eval_folds(folds: list, params: dict) -> dict:
    """Fit both side regressors per fold and pool the per-game deviance."""
    df = build_frame()
    from lightgbm import LGBMRegressor
    dev_h, dev_a, n = [], [], 0
    for fold in folds:
        train, val = df.loc[fold.train_idx], df.loc[fold.val_idx]
        p = dict(params)
        p["random_state"] = SEED
        p["verbose"] = -1
        p["objective"] = "poisson"
        reg = dist_mod.ScoreRegressor.__new__(dist_mod.ScoreRegressor)
        reg.home_model = LGBMRegressor(**p)
        reg.away_model = LGBMRegressor(**p)
        reg.feature_columns = []
        reg.fit(train)
        mu_h, mu_a = reg.predict(val)
        dev_h.append(poisson_dev(val["home_score"].to_numpy(float), mu_h))
        dev_a.append(poisson_dev(val["away_score"].to_numpy(float), mu_a))
        n += len(val)
    dev_h, dev_a = np.concatenate(dev_h), np.concatenate(dev_a)
    return {"n": int(n), "dev_home": float(dev_h.mean()),
            "dev_away": float(dev_a.mean()),
            "dev_mean": float((dev_h.mean() + dev_a.mean()) / 2.0)}


def main() -> None:
    t0 = time.time()
    df = build_frame()
    fold_list = get_folds()
    cut = len(fold_list) - N_FOLDS_HOLDOUT
    train_folds = fold_list[:cut]
    holdout_folds = fold_list[cut:]
    print(f"frame: {len(df)} games, {len(fold_list)} folds "
          f"({len(train_folds)} search / {len(holdout_folds)} sealed holdout)")

    incumbent = dict(config.LIGHTGBM_REG_PARAMS)
    inc_train = eval_folds(train_folds, incumbent)
    inc_hold = eval_folds(holdout_folds, incumbent)
    print(f"incumbent: train dev={inc_train['dev_mean']:.5f} "
          f"holdout dev={inc_hold['dev_mean']:.5f}")

    df_arr = df
    from lightgbm import LGBMRegressor

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 320),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 90),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split",
                                                     0.0, 3.0),
            "learning_rate": trial.suggest_float("learning_rate",
                                                 0.015, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree",
                                                    0.5, 0.95),
        }
        # Fold-by-fold with cumulative reporting so hopeless trials are
        # pruned early (a 102-fold sweep at full length is ~2.5 min/trial).
        dev_h, dev_a, n = [], [], 0
        for step, fold in enumerate(train_folds):
            train, val = df_arr.loc[fold.train_idx], df_arr.loc[fold.val_idx]
            p = dict(params)
            p["random_state"] = SEED
            p["verbose"] = -1
            p["objective"] = "poisson"
            reg = dist_mod.ScoreRegressor.__new__(dist_mod.ScoreRegressor)
            reg.home_model = LGBMRegressor(**p)
            reg.away_model = LGBMRegressor(**p)
            reg.feature_columns = []
            reg.fit(train)
            mu_h, mu_a = reg.predict(val)
            dev_h.append(poisson_dev(val["home_score"].to_numpy(float), mu_h))
            dev_a.append(poisson_dev(val["away_score"].to_numpy(float), mu_a))
            n += len(val)
            running = (float(np.concatenate(dev_h).mean())
                       + float(np.concatenate(dev_a).mean())) / 2.0
            trial.report(running, step)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"pruned at fold {step} (running dev {running:.5f})")
        dev_h, dev_a = np.concatenate(dev_h), np.concatenate(dev_a)
        dev_mean = float((dev_h.mean() + dev_a.mean()) / 2.0)
        trial.set_user_attr("dev_home", float(dev_h.mean()))
        trial.set_user_attr("dev_away", float(dev_a.mean()))
        return dev_mean

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8,
                                           n_warmup_steps=20))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = dict(study.best_params)
    best_train = eval_folds(train_folds, best)
    best_hold = eval_folds(holdout_folds, best)
    print(f"candidate: train dev={best_train['dev_mean']:.5f} "
          f"holdout dev={best_hold['dev_mean']:.5f}")
    print(f"holdout delta: {inc_hold['dev_mean'] - best_hold['dev_mean']:+.5f} "
          f"(positive = candidate better)")
    print("best params:", json.dumps(best, indent=1))

    record = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_games": int(len(df)), "n_folds": len(fold_list),
        "n_search_folds": len(train_folds),
        "n_holdout_folds": len(holdout_folds), "n_trials": N_TRIALS,
        "incumbent": {"params": {k: v for k, v in incumbent.items()
                                 if k not in ("random_state", "verbose")},
                      "train": inc_train, "holdout": inc_hold},
        "candidate": {"params": best, "train": best_train,
                      "holdout": best_hold},
        "holdout_delta_dev": inc_hold["dev_mean"] - best_hold["dev_mean"],
        "adopted": bool(best_hold["dev_mean"] < inc_hold["dev_mean"]),
        "n_pruned": len(study.trials) - len(study.best_trials) - sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = Path(__file__).parent / f"run_line_tuning_{datetime.now():%Y%m%d}.json"
    out.write_text(json.dumps(record, indent=1))
    print("record ->", out.name)


if __name__ == "__main__":
    main()
