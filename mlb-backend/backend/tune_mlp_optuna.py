"""
Optuna tuning for the MLP member of the MLB moneyline ensemble.

Mirrors backend/tune_xgboost_optuna.py and tune_lightgbm_optuna.py
structurally AND mirrors production's MLP training path
(backend/training.py) exactly:

  * Same feature contract   -> training._feature_matrix (full-width
                                FEATURE_COLS, canonical order) via
                                training._prepare_features
  * SAME IMPUTATION PATH    -> train-fold-median imputation ONLY (MLP cannot
                                consume NaN; never switch to a native-NaN
                                route) — val rows filled with TRAIN medians.
  * SAME SCALING            -> sklearn StandardScaler fit on the imputed
                                train matrix, transform applied to val.
  * Same fold generator     -> training.walk_forward_splits (fixed once,
                                reused by every trial).
  * Objective = POOLED out-of-fold logloss (one log_loss over all fold
    predictions concatenated — never a mean of per-fold scores).
  * Per fold: MLP early stopping on its own validation_fraction split
    (the production contract — MLP always early-stops internally).

Study durability:
  * Pass --storage sqlite:///path.db to persist trials; an interrupted run
    resumes where it left off (load_if_exists). Without it the study is
    in-memory.

Hold-out verification (once, after tuning):
  * Last N days (default 21) sealed BEFORE fold generation.
  * Current production config (config.MLP_PARAMS verbatim) vs the Optuna
    winner, both refit on all pre-holdout games (impute + scale on the
    pre-holdout set only) and scored on the sealed holdout.
  * A losing winner is REPORTED, not re-tuned — the current config stays.

Usage:
    python tune_mlp_optuna.py                          # 60 trials
    python tune_mlp_optuna.py --trials 5 --smoke       # sanity run
    python tune_mlp_optuna.py --max-folds 5            # plumbing check
    python tune_mlp_optuna.py --storage sqlite:///mlp.db --jobs 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss, roc_auc_score

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))
try:
    from training import (
        FEATURE_COLS,
        _impute_median,
        _prepare_features,
        walk_forward_splits,
    )
    from config import (
        DATA_DELIVERY_DIR,
        MLP_PARAMS,
        MIN_VAL_FOLD_GAMES,
        RANDOM_SEED,
        RETRAIN_CADENCE_DAYS,
    )
except ImportError:  # pragma: no cover - direct execution fallback
    from backend.training import (
        FEATURE_COLS,
        _impute_median,
        _prepare_features,
        walk_forward_splits,
    )
    from backend.config import (
        DATA_DELIVERY_DIR,
        MLP_PARAMS,
        MIN_VAL_FOLD_GAMES,
        RANDOM_SEED,
        RETRAIN_CADENCE_DAYS,
    )


# Probability clip used everywhere, chosen to match production's
# compute_metrics (training.py clips at 1e-7). The 1e-6 clip used in the
# first study pass differed measurably on degenerate early-fold points
# (pooled logloss 0.79105 vs 0.79879 on the current config) — reconciled
# against walk_forward_evaluate and aligned.
_EPS = 1e-7


# ---------------------------------------------------------------------------
# Data / folds — built through the PRODUCTION feature builders
# ---------------------------------------------------------------------------
def load_games(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.dropna(subset=["home_win"]).reset_index(drop=True)
    return df


def prepare_fold(tr: pd.DataFrame, va: pd.DataFrame) -> dict:
    """The exact production MLP input layout: imputed + StandardScaled
    full-width numeric matrix (MLP never sees team-ID categoricals)."""
    from sklearn.preprocessing import StandardScaler

    X_tr, _, y_tr = _prepare_features(tr)
    X_va, _, y_va = _prepare_features(va)
    X_tr_i, med = _impute_median(X_tr)
    X_va_i, _ = _impute_median(X_va, med)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_i)
    X_va_s = scaler.transform(X_va_i)
    return {
        "X_train": X_tr_s,
        "X_val": X_va_s,
        "y_train": y_tr.astype(float),
        "y_val": y_va.astype(float),
        "medians": med,
    }


# ---------------------------------------------------------------------------
# Fold-level training — production contract: impute + scale + early stopping
# ---------------------------------------------------------------------------
def make_model(params: dict):
    from sklearn.neural_network import MLPClassifier

    clean = {k: v for k, v in params.items() if not k.startswith("_")}
    # Early stopping + seed are never search dimensions: production always
    # early-stops (validation_fraction split) and seeds for reproducibility.
    clean["early_stopping"] = True
    clean["random_state"] = RANDOM_SEED
    return MLPClassifier(**clean)


def fit_fold(params: dict, fold: dict) -> tuple[np.ndarray, int]:
    """Train one model; returns (validation probabilities, epochs used)."""
    model = make_model(params)
    model.fit(fold["X_train"], fold["y_train"])
    n_iter = int(getattr(model, "n_iter_", params.get("max_iter", 300)))
    proba = model.predict_proba(fold["X_val"])[:, 1]
    return proba, n_iter


def base_params(sampled: dict | None) -> dict:
    """Fixed backbone. early_stopping/seed forced in make_model; max_iter
    defaults to the production ceiling when a trial does not sample it."""
    p = {"max_iter": 300}
    if sampled:
        p.update({k: v for k, v in sampled.items()
                  if not k.startswith("_")})
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=DATA_DELIVERY_DIR / "game_level_features.csv")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--holdout-days", type=int, default=21,
                    help="Sealed tail of the schedule (2-3 weeks suggested)")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny run to verify plumbing (4 trials)")
    ap.add_argument("--max-folds", type=int, default=0,
                    help="Cap folds at the N most recent (0 = all); "
                         "plumbing checks only — never for final tuning")
    ap.add_argument("--storage", type=str, default=None,
                    help="Optuna storage URL (e.g. sqlite:///mlp_study.db) "
                         "so interrupted runs resume instead of restarting")
    ap.add_argument("--study-name", type=str, default="mlp_moneyline")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel Optuna workers (fold data is pickled to "
                         "each worker once; 16-32 works well on big hosts)")
    ap.add_argument("--skip-verify", action="store_true",
                    help="Skip the current-config reference + sealed holdout "
                         "(batch/resume runs — the study alone is the point; "
                         "run the final verify with --trials 0)")
    args = ap.parse_args()
    if args.smoke:
        args.trials = 4

    games = load_games(args.data)
    cutoff = games["game_date"].max() - pd.Timedelta(days=args.holdout_days - 1)
    tune_df = games[games["game_date"] < cutoff].reset_index(drop=True)
    hold_df = games[games["game_date"] >= cutoff].reset_index(drop=True)
    print(f"games={len(games)}  tuning={len(tune_df)}  "
          f"HOLDOUT(sealed)={len(hold_df)}  "
          f"[{hold_df['game_date'].min().date()} → {hold_df['game_date'].max().date()}]")

    # Fixed folds — generated once, reused by every trial.
    folds = [
        s for s in walk_forward_splits(
            tune_df, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
        if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES
    ]
    print(f"fixed folds: {len(folds)}  "
          f"(windows {folds[0]['val_start'].date()} → {folds[-1]['val_end'].date()})")
    if args.max_folds > 0:
        folds = folds[-args.max_folds:]
        print(f"  ⚠️ --max-folds={args.max_folds}: plumbing check only — "
              f"scoring the {len(folds)} most recent folds")
    fold_data = [prepare_fold(s["train_games"], s["val_games"]) for s in folds]
    print(f"fold frames: {FEATURE_COLS.__len__() if hasattr(FEATURE_COLS, '__len__') else len(list(FEATURE_COLS))} "
          f"numeric cols, imputed + StandardScaled (train-fold medians only)")

    # ----------------------------- study -----------------------------------
    # Hidden-layer choices are STRING-COded for sqlite-persistent storage
    # (tuples/lists are not stable categorical values across resume), mapped
    # back to sklearn tuples inside the objective.
    _HIDDEN_CODES = {"16": (16,), "32": (32,), "32_16": (32, 16),
                     "64_32": (64, 32), "64_32_16": (64, 32, 16)}

    def objective(trial: optuna.Trial) -> float:
        sampled = {
            "hidden_layer_sizes": _HIDDEN_CODES[trial.suggest_categorical(
                "hidden_layer_sizes", list(_HIDDEN_CODES))],
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", ["adaptive", "constant"]),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical(
                "batch_size", [64, 128, 256]),
            "max_iter": trial.suggest_categorical("max_iter", [300, 600]),
            "validation_fraction": trial.suggest_categorical(
                "validation_fraction", [0.1, 0.15]),
            "n_iter_no_change": trial.suggest_categorical(
                "n_iter_no_change", [10, 15, 20]),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "tanh"]),
        }
        params = base_params(sampled)
        pooled_pred, pooled_y, iters = [], [], []
        for fold in fold_data:
            proba, n_iter = fit_fold(params, fold)
            pooled_pred.append(np.clip(proba, _EPS, 1 - _EPS))
            pooled_y.append(fold["y_val"])
        iters.append(n_iter)
        trial.set_user_attr("mean_n_iter", float(np.mean(iters)))
        trial.set_user_attr("median_n_iter", float(np.median(iters)))
        return log_loss(np.concatenate(pooled_y), np.concatenate(pooled_pred))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=True,
    )
    if args.trials > 0:
        study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs,
                       show_progress_bar=False)
    if args.skip_verify:
        print(f"\nstudy: {len(study.trials)} trials (--skip-verify); "
              f"best so far = {study.best_trial.value:.5f}")
        return

    best = study.best_trial
    bp = dict(best.params)
    print("\n================ STUDY RESULT ================")
    print(f"trials completed     : {len(study.trials)}")
    print(f"best POOLED OOF loss : {best.value:.5f}")
    print(f"best_params          : {bp}")

    # Current-config reference on the SAME fixed folds (pooled, honest).
    cur_pred, cur_y = [], []
    for fold in fold_data:
        proba, _ = fit_fold(base_params(dict(MLP_PARAMS)), fold)
        cur_pred.append(np.clip(proba, _EPS, 1 - _EPS))
        cur_y.append(fold["y_val"])
    cur_y_all = np.concatenate(cur_y)
    cur_ll = log_loss(cur_y_all, np.concatenate(cur_pred))
    cur_auc = roc_auc_score(cur_y_all, np.concatenate(cur_pred))
    print(f"current cfg pooled   : logloss={cur_ll:.5f}  auc={cur_auc:.4f}")
    print(f"tuned   cfg pooled   : logloss={best.value:.5f}")

    # ----------------------- hold-out verification -------------------------
    print("\n============ SEALED HOLDOUT VERIFICATION ============")
    from training import _prepare_features
    from sklearn.preprocessing import StandardScaler
    X_refit, _, refit_y = _prepare_features(tune_df)
    X_hold, _, hold_y = _prepare_features(hold_df)
    X_refit_i, refit_med = _impute_median(X_refit)
    X_hold_i, _ = _impute_median(X_hold, refit_med)
    scaler = StandardScaler()
    X_refit_s = scaler.fit_transform(X_refit_i)
    X_hold_s = scaler.transform(X_hold_i)
    hold_fold = {
        "X_train": X_refit_s, "X_val": X_hold_s,
        "y_train": refit_y.astype(float), "y_val": hold_y.astype(float),
    }

    eps = _EPS
    rows = []

    # Baseline: exact production config (config.MLP_PARAMS verbatim).
    proba_base, _ = fit_fold(base_params(dict(MLP_PARAMS)), hold_fold)
    pc = np.clip(proba_base, eps, 1 - eps)
    rows.append(("current (MLP_PARAMS verbatim)", log_loss(hold_y, pc),
                 roc_auc_score(hold_y, pc)))

    # Winner: same params as the best trial (no round selection — MLP
    # early-stops internally, exactly as production refits do).
    bp_decoded = dict(bp)
    if isinstance(bp.get("hidden_layer_sizes"), str):
        bp_decoded["hidden_layer_sizes"] = _HIDDEN_CODES[bp["hidden_layer_sizes"]]
    proba_tune, _ = fit_fold(base_params(bp_decoded), hold_fold)
    pc = np.clip(proba_tune, eps, 1 - eps)
    rows.append((f"optuna winner (n_iter mean {best.user_attrs.get('mean_n_iter', float('nan')):.0f})",
                 log_loss(hold_y, pc), roc_auc_score(hold_y, pc)))

    w = max(len(n) for n, _, _ in rows)
    print(f"{'config':<{w}} | {'logloss':>8} | {'auc':>7}")
    for n, ll, auc in rows:
        print(f"{n:<{w}} | {ll:8.5f} | {auc:7.4f}")
    (ll_b, ll_t) = rows[0][1], rows[1][1]
    winner = rows[0] if ll_b <= ll_t else rows[1]
    print(f"\nHOLDOUT WINNER: {winner[0]}  "
          f"(Δlogloss={abs(ll_b - ll_t):.5f})")
    if winner[0].startswith("current"):
        print("→ The winner did NOT beat the current config on the sealed "
              "holdout. Current config stays (honesty contract).")


if __name__ == "__main__":
    main()
