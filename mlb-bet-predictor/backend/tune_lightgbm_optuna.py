"""
Optuna tuning for the LightGBM member of the MLB moneyline ensemble.

Mirrors backend/tune_xgboost_optuna.py structurally AND mirrors production's
LightGBM training path (backend/training.py) exactly:

  * Same feature contract      -> training._feature_matrix (full-width
                                    FEATURE_COLS, canonical order)
  * SAME TEAM-ID ROUTING       -> int home_team_id/away_team_id columns
                                    (UNK_TEAM_ID clamped) appended to every
                                    frame, passed as categorical_feature BY
                                    NAME at EVERY fit — production's exact
                                    contract. Dropping the IDs "for
                                    simplicity" would tune a model we do
                                    not deploy.
  * Imputation is a TRIAL DIMENSION here (unlike the XGBoost tuner):
                                    production LightGBM consumes the raw
                                    NaN-preserving matrix natively today, so
                                    train-median-imputation-vs-native-NaN is
                                    a legitimate, deployable choice.
  * Same fold generator        -> training.walk_forward_splits (fixed once,
                                    reused by every trial).
  * Objective = POOLED out-of-fold logloss (one log_loss over all fold
    predictions concatenated — never a mean of per-fold scores).
  * Per fold: early stopping on that fold's own validation window
    (early_stopping_rounds=20, matching the XGBoost fold trainer).

Study durability:
  * Pass --storage sqlite:///path.db to persist trials; an interrupted run
    resumes where it left off (load_if_exists). Without it the study is
    in-memory.

Hold-out verification (once, after tuning):
  * Last N days (default 21) sealed BEFORE fold generation.
  * Current production config (LIGHTGBM_PARAMS verbatim: depth 5 / 300
    rounds / native NaN) vs the Optuna winner, both refit on all pre-holdout
    games and scored on the sealed holdout. Winner rounds = median
    best_iteration across the winning trial's folds (never sees holdout).
  * A losing winner is REPORTED, not re-tuned.

Usage:
    python tune_lightgbm_optuna.py                         # 50 trials
    python tune_lightgbm_optuna.py --trials 5 --smoke      # sanity run
    python tune_lightgbm_optuna.py --max-folds 5           # plumbing check
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
        TREE_CATEGORICAL_COLS,
        UNK_TEAM_ID,
        _add_team_ids,
        _categorical_matrix,
        _feature_matrix,
        walk_forward_splits,
    )
    from config import (
        DATA_DELIVERY_DIR,
        LIGHTGBM_PARAMS,
        MIN_VAL_FOLD_GAMES,
        RANDOM_SEED,
        RETRAIN_CADENCE_DAYS,
    )
except ImportError:  # pragma: no cover - direct execution fallback
    from backend.training import (
        FEATURE_COLS,
        TREE_CATEGORICAL_COLS,
        UNK_TEAM_ID,
        _add_team_ids,
        _categorical_matrix,
        _feature_matrix,
        walk_forward_splits,
    )
    from backend.config import (
        DATA_DELIVERY_DIR,
        LIGHTGBM_PARAMS,
        MIN_VAL_FOLD_GAMES,
        RANDOM_SEED,
        RETRAIN_CADENCE_DAYS,
    )

EARLY_STOPPING_ROUNDS = 20   # == the XGBoost fold trainer's setting
MAX_ROUNDS = 2000            # generous ceiling; early stopping picks the count


# ---------------------------------------------------------------------------
# Data / folds — built through the PRODUCTION feature builders
# ---------------------------------------------------------------------------
def load_games(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.dropna(subset=["home_win"]).reset_index(drop=True)
    return df


def train_medians(X: np.ndarray) -> np.ndarray:
    """Column medians from the training rows only (never validation rows)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(X, axis=0)


def fill_medians(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    Xf = X.copy()
    idx = np.where(np.isnan(Xf))
    Xf[idx] = np.take(med, idx[1])
    return Xf


def lgbm_frame(X_num: np.ndarray, X_cat: np.ndarray,
               impute: bool, med: np.ndarray | None) -> "pd.DataFrame":
    """The exact production input layout: named FEATURE_COLS (NaN-preserving
    when impute=False, train-median-filled when impute=True) plus INT
    team-ID columns clamped to UNK_TEAM_ID."""
    if impute:
        if med is None:
            raise ValueError("impute=True requires train-fold medians")
        X_num = fill_medians(X_num, med)
    df = pd.DataFrame(X_num, columns=list(FEATURE_COLS))
    for i, c in enumerate(TREE_CATEGORICAL_COLS):
        vals = np.where(X_cat[:, i] < 0, UNK_TEAM_ID, X_cat[:, i]).astype(int)
        df[c] = vals
    return df


def prepare_fold(tr: pd.DataFrame, va: pd.DataFrame) -> dict:
    """Both numeric variants (native NaN + train-median-imputed) and the
    shared categorical matrix, materialized once per fold."""
    tr_ids, va_ids = _add_team_ids(tr), _add_team_ids(va)
    X_num_tr = _feature_matrix(tr_ids)
    X_num_va = _feature_matrix(va_ids)
    med = train_medians(X_num_tr)
    X_cat_tr = _categorical_matrix(tr_ids)
    X_cat_va = _categorical_matrix(va_ids)
    return {
        "frames": {
            False: lgbm_frame(X_num_tr, X_cat_tr, False, None),
            True: lgbm_frame(X_num_tr, X_cat_tr, True, med),
        },
        "val_frames": {
            False: lgbm_frame(X_num_va, X_cat_va, False, None),
            # Validation rows are filled with TRAIN medians only.
            True: lgbm_frame(X_num_va, X_cat_va, True, med),
        },
        "y_train": tr["home_win"].to_numpy(dtype=float),
        "y_val": va["home_win"].to_numpy(dtype=float),
    }


# ---------------------------------------------------------------------------
# Fold-level training — production contract: categorical_feature BY NAME at
# EVERY fit, UNK clamp baked into the frame.
# ---------------------------------------------------------------------------
def make_model(params: dict):
    from lightgbm import LGBMClassifier
    clean = {k: v for k, v in params.items() if not k.startswith("_")}
    return LGBMClassifier(**clean)


def fit_fold(params: dict, fold: dict, impute: bool,
             early_stop: bool) -> tuple[np.ndarray, int]:
    """Train one model; returns (validation probabilities, rounds used)."""
    from lightgbm import early_stopping, log_evaluation

    model = make_model(params)
    tr_frame = fold["frames"][bool(impute)]
    va_frame = fold["val_frames"][bool(impute)]
    if early_stop:
        model.set_params(n_estimators=MAX_ROUNDS)
        model.fit(
            tr_frame, fold["y_train"],
            eval_set=[(va_frame, fold["y_val"])],
            categorical_feature=list(TREE_CATEGORICAL_COLS),
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                       log_evaluation(period=0)],
        )
        best = int(model.best_iteration_ or MAX_ROUNDS)
        proba = model.predict_proba(va_frame, num_iteration=best)[:, 1]
    else:
        model.fit(
            tr_frame, fold["y_train"],
            categorical_feature=list(TREE_CATEGORICAL_COLS),
        )
        best = int(model.n_estimators or params.get("n_estimators") or 100)
        proba = model.predict_proba(va_frame)[:, 1]
    return proba, best


def base_params(sampled: dict | None) -> dict:
    """Fixed backbone. objective stays binary; seed fixed for reproducibility."""
    p = {"objective": "binary", "seed": RANDOM_SEED, "verbose": -1}
    if sampled:
        p.update({k: v for k, v in sampled.items()
                  if k not in ("_rounds", "impute_medians")})
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
    ap.add_argument("--max-folds", type=int, default=0,
                    help="Cap folds at the N most recent (0 = all); "
                         "plumbing checks only — never for final tuning")
    ap.add_argument("--storage", type=str, default=None,
                    help="Optuna storage URL (e.g. sqlite:///lgbm_study.db) "
                         "so interrupted runs resume instead of restarting")
    ap.add_argument("--study-name", type=str, default="lightgbm_moneyline")
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
    print(f"fold frames: {len(FEATURE_COLS)} numeric + {len(TREE_CATEGORICAL_COLS)} "
          f"categorical team-ID cols (categorical_feature by name, every fit)")

    # ----------------------------- study -----------------------------------
    def objective(trial: optuna.Trial) -> float:
        sampled = {
            "num_leaves": trial.suggest_int("num_leaves", 2, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.1, 2.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.8),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06),
            "impute_medians": trial.suggest_categorical("impute_medians", [True, False]),
        }
        params = base_params(sampled)
        # Guardrail made structural: bagging needs a frequency to activate.
        if sampled["bagging_fraction"] < 1.0:
            params["bagging_freq"] = 1
        pooled_pred, pooled_y, iters = [], [], []
        for fold in fold_data:
            proba, best = fit_fold(params, fold, sampled["impute_medians"],
                                   early_stop=True)
            pooled_pred.append(np.clip(proba, 1e-6, 1 - 1e-6))
            pooled_y.append(fold["y_val"])
            iters.append(best)
        trial.set_user_attr("mean_best_iter", float(np.mean(iters)))
        trial.set_user_attr("median_best_iter", float(np.median(iters)))
        return log_loss(np.concatenate(pooled_y), np.concatenate(pooled_pred))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best = study.best_trial
    bp = dict(best.params)
    print("\n================ STUDY RESULT ================")
    print(f"trials completed     : {len(study.trials)}")
    print(f"best POOLED OOF loss : {best.value:.5f}")
    print(f"best_params          : {bp}")

    # Current-config reference on the SAME fixed folds (pooled, honest).
    cur_params = base_params(dict(LIGHTGBM_PARAMS))
    cur_pred, cur_y = [], []
    for fold in fold_data:
        proba, _ = fit_fold(cur_params.copy(), fold, impute=False,
                            early_stop=False)
        cur_pred.append(np.clip(proba, 1e-6, 1 - 1e-6))
        cur_y.append(fold["y_val"])
    cur_y_all = np.concatenate(cur_y)
    cur_ll = log_loss(cur_y_all, np.concatenate(cur_pred))
    cur_auc = roc_auc_score(cur_y_all, np.concatenate(cur_pred))
    print(f"current cfg pooled   : logloss={cur_ll:.5f}  auc={cur_auc:.4f}")
    print(f"tuned   cfg pooled   : logloss={best.value:.5f}")

    # ----------------------- hold-out verification -------------------------
    print("\n============ SEALED HOLDOUT VERIFICATION ============")
    hold_ids = _add_team_ids(hold_df)
    hold_X_num = _feature_matrix(hold_ids)
    hold_X_cat = _categorical_matrix(hold_ids)
    refit_ids = _add_team_ids(tune_df)
    refit_X_num = _feature_matrix(refit_ids)
    refit_med = train_medians(refit_X_num)
    refit_y = tune_df["home_win"].to_numpy(dtype=float)
    hold_y = hold_df["home_win"].to_numpy(dtype=float)

    def holdout_fold(impute: bool) -> dict:
        # Holdout rows filled with REFIT medians when imputing (never their own).
        return {
            "frames": {False: lgbm_frame(refit_X_num, _categorical_matrix(refit_ids),
                                         False, None),
                       True: lgbm_frame(refit_X_num, _categorical_matrix(refit_ids),
                                        True, refit_med)},
            "val_frames": {False: lgbm_frame(hold_X_num, hold_X_cat, False, None),
                           True: lgbm_frame(hold_X_num, hold_X_cat, True, refit_med)},
            "y_train": refit_y, "y_val": hold_y,
        }

    eps = 1e-6
    rows = []

    # Baseline: exact production config, native NaN, fixed 300 rounds.
    proba_base, _ = fit_fold(base_params(dict(LIGHTGBM_PARAMS)),
                             holdout_fold(False), impute=False, early_stop=False)
    pc = np.clip(proba_base, eps, 1 - eps)
    rows.append(("current (LIGHTGBM_PARAMS verbatim)", log_loss(hold_y, pc),
                 roc_auc_score(hold_y, pc)))

    # Winner: rounds = median best_iter the winning trial used across its
    # folds (holdout labels untouched by that choice).
    rounds = int(max(50, min(best.user_attrs.get("median_best_iter", 300), MAX_ROUNDS)))
    win_params = base_params(bp)
    win_params["_rounds"] = rounds
    win_params["n_estimators"] = rounds
    win_impute = bool(bp["impute_medians"])
    if bp["bagging_fraction"] < 1.0:
        win_params["bagging_freq"] = 1
    proba_tune, _ = fit_fold(win_params, holdout_fold(win_impute),
                             impute=win_impute, early_stop=False)
    pc = np.clip(proba_tune, eps, 1 - eps)
    rows.append((f"optuna winner ({rounds}r, impute={win_impute})",
                 log_loss(hold_y, pc), roc_auc_score(hold_y, pc)))

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
