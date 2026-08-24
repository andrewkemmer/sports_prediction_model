"""
Optuna tuning for the XGBoost member of the MLB moneyline ensemble.

Methodology (mirrors backend/training.py exactly):
  * Same feature contract      -> training.FEATURE_COLS
  * Same fold generator        -> training.walk_forward_splits
                                    (7-day windows, expanding window,
                                     min 40 validation games per fold)
  * Folds are generated ONCE before tuning and reused for every trial,
    so every candidate is scored on identical date windows.
  * Objective = POOLED out-of-fold logloss: predictions from every fold's
    validation window are concatenated into one array and log_loss is
    computed once (per-fold averaging is never used).
  * Inside each fold, early stopping selects the round count on that
    fold's own validation window (prequential-honest).

Hold-out verification (run once after tuning):
  * The last N days of games (default 21) are sealed BEFORE fold
    generation — no trial ever sees them.
  * Current production config (XGBOOST_PARAMS: depth 5 / 300 rounds /
    lr 0.05, native NaN handling) vs the Optuna winner are both refit on
    all pre-holdout games and scored on the sealed holdout.
  * Tuned round count at refit = median best_iteration observed across
    the winning trial's folds (no holdout labels involved).

Usage:
    python tune_xgboost_optuna.py                        # 50 trials
    python tune_xgboost_optuna.py --trials 5 --smoke     # quick sanity run
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
import xgboost as xgb

# --- import siblings whether run from backend/ or repo root -------------
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))
try:
    from training import FEATURE_COLS, walk_forward_splits
    from config import (
        DATA_DELIVERY_DIR,
        MIN_VAL_FOLD_GAMES,
        RANDOM_SEED,
        RETRAIN_CADENCE_DAYS,
        XGBOOST_PARAMS,
    )
except ImportError:  # pragma: no cover - direct execution fallback
    from backend.training import FEATURE_COLS, walk_forward_splits
    from backend.config import (
        DATA_DELIVERY_DIR,
        MIN_VAL_FOLD_GAMES,
        RANDOM_SEED,
        RETRAIN_CADENCE_DAYS,
        XGBOOST_PARAMS,
    )

EARLY_STOPPING_ROUNDS = 50
MAX_ROUNDS = 2000          # generous cap; early stopping picks the real count


# ---------------------------------------------------------------------------
# Data / folds
# ---------------------------------------------------------------------------
def load_games(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.dropna(subset=["home_win"]).reset_index(drop=True)
    return df


def matrix(df: pd.DataFrame, medians: np.ndarray | None) -> np.ndarray:
    """FEATURE_COLS matrix preserving NaN (medians=None) or median-filled."""
    X = df[[c for c in FEATURE_COLS if c in df.columns]] \
        .reindex(columns=FEATURE_COLS).to_numpy(dtype=float)
    if medians is not None:
        idx = np.where(np.isnan(X))
        X[idx] = np.take(medians, idx[1])
    return X


def train_medians(X: np.ndarray) -> np.ndarray:
    """Column medians from the training rows only. A column that is entirely
    empty within a fold (e.g. 5-start fields in week one) keeps NaN so the
    booster's native handling applies — we never fabricate a fill value."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(X, axis=0)


def make_dmatrix(X: np.ndarray, y: np.ndarray | None) -> xgb.DMatrix:
    return xgb.DMatrix(X, label=y, feature_names=list(FEATURE_COLS))


# ---------------------------------------------------------------------------
# Fold-level training shared by tuning and holdout stages
# ---------------------------------------------------------------------------
def fit_fold(params: dict, X_tr, y_tr, X_va, y_va,
             early_stop: bool) -> tuple[np.ndarray, int]:
    """Train one booster; returns (validation probabilities, rounds used)."""
    dtr = make_dmatrix(X_tr, y_tr)
    dva = make_dmatrix(X_va, y_va)
    clean = {k: v for k, v in params.items() if not k.startswith("_")}
    rounds = MAX_ROUNDS if early_stop else int(params.get("_rounds", 300))
    booster = xgb.train(
        clean,
        dtr,
        num_boost_round=rounds,
        evals=[(dva, "val")],
        early_stopping_rounds=(EARLY_STOPPING_ROUNDS if early_stop else None),
        verbose_eval=False,
    )
    best = int(getattr(booster, "best_iteration", 0)) + 1
    proba = booster.predict(dva, iteration_range=(0, best))
    return proba, best


def base_params(sampled: dict | None) -> dict:
    p = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "seed": RANDOM_SEED,
    }
    if sampled:
        p.update({k: v for k, v in sampled.items() if k != "impute_medians"})
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=DATA_DELIVERY_DIR / "game_level_features.csv")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--holdout-days", type=int, default=21,
                    help="Sealed tail of the schedule (2-3 weeks suggested)")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny run to verify plumbing (4 trials)")
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

    fold_data = []
    for s in folds:
        tr, va = s["train_games"], s["val_games"]
        fold_data.append((
            matrix(tr, None), tr["home_win"].to_numpy(dtype=float),
            matrix(va, None), va["home_win"].to_numpy(dtype=float),
        ))

    # ----------------------------- study -----------------------------------
    def objective(trial: optuna.Trial) -> float:
        sampled = {
            "max_depth": trial.suggest_int("max_depth", 1, 3),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "gamma": trial.suggest_float("gamma", 0.5, 5.0),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
            "eta": trial.suggest_float("eta", 0.01, 0.06),
            "impute_medians": trial.suggest_categorical("impute_medians", [True, False]),
        }
        params = base_params(sampled)
        pooled_pred, pooled_y, iters = [], [], []
        for X_tr, y_tr, X_va, y_va in fold_data:
            if sampled["impute_medians"]:
                med = train_medians(X_tr)
                X_tr_s, X_va_s = matrix_fill(X_tr, med), matrix_fill(X_va, med)
            else:
                X_tr_s, X_va_s = X_tr, X_va
            proba, best = fit_fold(params, X_tr_s, y_tr, X_va_s, y_va,
                                   early_stop=True)
            pooled_pred.append(np.clip(proba, 1e-6, 1 - 1e-6))
            pooled_y.append(y_va)
            iters.append(best)
        trial.set_user_attr("mean_best_iter", float(np.mean(iters)))
        trial.set_user_attr("median_best_iter", float(np.median(iters)))
        return log_loss(np.concatenate(pooled_y), np.concatenate(pooled_pred))

    def matrix_fill(X: np.ndarray, med: np.ndarray) -> np.ndarray:
        Xf = X.copy()
        idx = np.where(np.isnan(Xf))
        Xf[idx] = np.take(med, idx[1])
        return Xf

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best = study.best_trial
    bp = dict(best.params)
    print("\n================ STUDY RESULT ================")
    print(f"trials               : {len(study.trials)}")
    print(f"best POOLED OOF loss : {best.value:.5f}")
    print(f"best_params          : {bp}")

    # Current-config reference on the SAME fixed folds (pooled, honest).
    cur_params = base_params({
        "max_depth": XGBOOST_PARAMS["max_depth"],
        "eta": XGBOOST_PARAMS["learning_rate"],
        "subsample": XGBOOST_PARAMS["subsample"],
        "colsample_bytree": XGBOOST_PARAMS["colsample_bytree"],
        "min_child_weight": 1, "gamma": 0.0,
        "impute_medians": False, "_rounds": XGBOOST_PARAMS["n_estimators"],
    })
    cur_pred, cur_y = [], []
    for X_tr, y_tr, X_va, y_va in fold_data:
        proba, _ = fit_fold(cur_params.copy(), X_tr, y_tr, X_va, y_va,
                            early_stop=False)
        cur_pred.append(np.clip(proba, 1e-6, 1 - 1e-6))
        cur_y.append(y_va)
    cur_ll = log_loss(np.concatenate(cur_y), np.concatenate(cur_pred))
    cur_auc = roc_auc_score(np.concatenate(cur_y), np.concatenate(cur_pred))
    tuned_auc_ctx = None
    print(f"current cfg pooled   : logloss={cur_ll:.5f}  auc={cur_auc:.4f}")
    print(f"tuned   cfg pooled   : logloss={best.value:.5f}")

    # ----------------------- hold-out verification -------------------------
    print("\n============ SEALED HOLDOUT VERIFICATION ============")
    full_X = matrix(tune_df, None)
    full_y = tune_df["home_win"].to_numpy(dtype=float)
    hold_X = matrix(hold_df, None)
    hold_y = hold_df["home_win"].to_numpy(dtype=float)

    # Baseline: exact production config, fixed 300 rounds, no imputation.
    base_hold_params = {k: v for k, v in base_params({
        "max_depth": XGBOOST_PARAMS["max_depth"],
        "eta": XGBOOST_PARAMS["learning_rate"],
        "subsample": XGBOOST_PARAMS["subsample"],
        "colsample_bytree": XGBOOST_PARAMS["colsample_bytree"],
        "_rounds": XGBOOST_PARAMS["n_estimators"],
    }).items()}
    proba_base, _ = fit_fold(base_hold_params, full_X, full_y, hold_X, hold_y,
                             early_stop=False)

    # Winner: refit on all pre-holdout data; rounds = median best_iter the
    # winning trial used across folds (never touches holdout labels).
    rounds = int(max(50, min(best.user_attrs.get("median_best_iter", 300), MAX_ROUNDS)))
    win_params = base_params(bp)
    win_params["_rounds"] = rounds
    if bp["impute_medians"]:
        med = train_medians(full_X)
        fX, hX = matrix_fill(full_X, med), matrix_fill(hold_X, med)
    else:
        fX, hX = full_X, hold_X
    proba_tune, _ = fit_fold(win_params, fX, full_y, hX, hold_y,
                             early_stop=False)

    eps = 1e-6
    rows = []
    for name, p in (("current (depth5/300/lr.05)", proba_base),
                    (f"optuna winner ({rounds}r)", proba_tune)):
        pc = np.clip(p, eps, 1 - eps)
        rows.append((name,
                     log_loss(hold_y, pc),
                     roc_auc_score(hold_y, pc)))
    w = max(len(n) for n, _, _ in rows)
    print(f"{'config':<{w}} | {'logloss':>8} | {'auc':>7}")
    for n, ll, auc in rows:
        print(f"{n:<{w}} | {ll:8.5f} | {auc:7.4f}")
    (ll_b, ll_t) = rows[0][1], rows[1][1]
    winner = rows[0] if ll_b <= ll_t else rows[1]
    print(f"\nHOLDOUT WINNER: {winner[0]}  "
          f"(Δlogloss={abs(ll_b - ll_t):.5f})")
    print(f"winning trial mean best_iter across folds: "
          f"{best.user_attrs.get('mean_best_iter', float('nan')):.1f}")


if __name__ == "__main__":
    main()
