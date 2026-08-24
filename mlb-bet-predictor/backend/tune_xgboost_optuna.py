"""
Optuna tuning for the XGBoost member of the MLB moneyline ensemble.

Methodology (mirrors backend/training.py EXACTLY — same code paths, not
a reimplementation):

  * Same feature contract      -> training._feature_matrix (full-width
                                    FEATURE_COLS, canonical order)
  * SAME TEAM-ID ROUTING       -> training._add_team_ids +
                                    training._categorical_matrix +
                                    training._tree_dataframe. The tuner
                                    trains on the exact DataFrame layout
                                    production uses: numeric FEATURE_COLS
                                    plus home_team_id/away_team_id as
                                    pd.Categorical with the explicit
                                    known-ids + UNK_TEAM_ID category set,
                                    consumed via enable_categorical=True.
                                    A tuned config therefore matches
                                    production by construction — dropping
                                    the ID columns "for simplicity" would
                                    select hyperparameters for a model we
                                    do not deploy.
  * Same imputation            -> production median-imputes the numeric
                                    block for XGBoost every fold
                                    (train-fold medians only); the tuner
                                    does the same unconditionally.
                                    impute-vs-not is NOT a search
                                    dimension because a False winner could
                                    never be deployed faithfully.
  * Same fold generator        -> training.walk_forward_splits
                                    (7-day windows, expanding window,
                                     min 40 validation games per fold)
  * Folds are generated ONCE before tuning and reused for every trial,
    so every candidate is scored on identical date windows.
  * Objective = POOLED out-of-fold logloss: predictions from every fold's
    validation window are concatenated into one array and log_loss is
    computed once (per-fold averaging is never used).
  * Inside each fold, early stopping selects the round count on that
    fold's own validation window (prequential-honest), via the same
    sklearn-API mechanism as production
    (n_estimators=XGBOOST_FOLD_ROUNDS, early_stopping_rounds=
    XGBOOST_EARLY_STOP).

Hold-out verification (run once after tuning):
  * The last N days of games (default 21) are sealed BEFORE fold
    generation — no trial ever sees them.
  * The CURRENT production config (XGBOOST_PARAMS verbatim, sklearn
    defaults for anything it does not set — exactly how a fit-only refit
    runs in production) vs the Optuna winner are both refit on all
    pre-holdout games and scored on the sealed holdout.
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

# --- import siblings whether run from backend/ or repo root -------------
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
        _tree_dataframe,
        walk_forward_splits,
    )
    from config import (
        DATA_DELIVERY_DIR,
        MIN_VAL_FOLD_GAMES,
        RANDOM_SEED,
        RETRAIN_CADENCE_DAYS,
        XGBOOST_PARAMS,
    )
except ImportError:  # pragma: no cover - direct execution fallback
    from backend.training import (
        FEATURE_COLS,
        TREE_CATEGORICAL_COLS,
        UNK_TEAM_ID,
        _add_team_ids,
        _categorical_matrix,
        _feature_matrix,
        _tree_dataframe,
        walk_forward_splits,
    )
    from backend.config import (
        DATA_DELIVERY_DIR,
        MIN_VAL_FOLD_GAMES,
        RANDOM_SEED,
        RETRAIN_CADENCE_DAYS,
        XGBOOST_PARAMS,
    )

MAX_ROUNDS = 2000          # generous cap; early stopping picks the real count


# ---------------------------------------------------------------------------
# Data / folds — built through the PRODUCTION feature builders
# ---------------------------------------------------------------------------
def load_games(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.dropna(subset=["home_win"]).reset_index(drop=True)
    return df


def train_medians(X: np.ndarray) -> np.ndarray:
    """Column medians from the training rows only. A column that is entirely
    empty within a fold (e.g. 5-start fields in week one) keeps NaN so the
    booster's native handling applies — we never fabricate a fill value."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(X, axis=0)


def fill_medians(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    Xf = X.copy()
    idx = np.where(np.isnan(Xf))
    Xf[idx] = np.take(med, idx[1])
    return Xf


def tree_frame(X_num: np.ndarray, X_cat: np.ndarray) -> "pd.DataFrame":
    """The exact production input layout: named numeric FEATURE_COLS plus
    categorical team-ID columns (explicit categories incl. UNK_TEAM_ID).

    Delegates to training._tree_dataframe so any change to production
    routing automatically changes the tuner identically."""
    return _tree_dataframe(X_num, X_cat, list(FEATURE_COLS))


def prepare_fold(tr: pd.DataFrame, va: pd.DataFrame) -> dict:
    """Numeric (median-imputed on train rows) + categorical matrices and the
    ready-to-fit tree DataFrames for one fold — the same transformations the
    production fold trainer applies to XGBoost."""
    tr_ids, va_ids = _add_team_ids(tr), _add_team_ids(va)
    X_num_tr = _feature_matrix(tr_ids)
    X_num_va = _feature_matrix(va_ids)
    med = train_medians(X_num_tr)
    X_num_tr_i, X_num_va_i = fill_medians(X_num_tr, med), fill_medians(X_num_va, med)
    X_cat_tr = _categorical_matrix(tr_ids)
    X_cat_va = _categorical_matrix(va_ids)
    return {
        "train_frame": tree_frame(X_num_tr_i, X_cat_tr),
        "val_frame": tree_frame(X_num_va_i, X_cat_va),
        "y_train": tr["home_win"].to_numpy(dtype=float),
        "y_val": va["home_win"].to_numpy(dtype=float),
    }


# ---------------------------------------------------------------------------
# Fold-level training shared by tuning and holdout stages — mirrors the
# production fold trainer (XGBClassifier + enable_categorical + constructor
# early stopping), never a parallel DMatrix path.
# ---------------------------------------------------------------------------
def make_model(params: dict, early_stop: bool):
    from xgboost import XGBClassifier
    clean = {k: v for k, v in params.items() if not k.startswith("_")}
    if early_stop:
        # 20 == XGBOOST_EARLY_STOP, kept literal so tuning has no path that
        # edits production config constants.
        return XGBClassifier(
            **clean,
            n_estimators=MAX_ROUNDS,
            early_stopping_rounds=20,
        )
    return XGBClassifier(**clean)


def fit_fold(params: dict, fold: dict, early_stop: bool) -> tuple[np.ndarray, int]:
    """Train one model; returns (validation probabilities, rounds used)."""
    model = make_model(params, early_stop)
    model.fit(
        fold["train_frame"], fold["y_train"],
        eval_set=[(fold["val_frame"], fold["y_val"])],
        verbose=False,
    ) if early_stop else model.fit(fold["train_frame"], fold["y_train"], verbose=False)
    if early_stop:
        best = int(model.best_iteration) + 1
        proba = model.predict_proba(
            fold["val_frame"], iteration_range=(0, best))[:, 1]
    else:
        # No early stopping (refit semantics): all configured rounds are used.
        # n_estimators can be None on sklearn defaults (XGBOOST_PARAMS omits it).
        best = int(model.n_estimators or params.get("n_estimators") or 100)
        proba = model.predict_proba(fold["val_frame"])[:, 1]
    return proba, best


def base_params(sampled: dict | None) -> dict:
    """Fixed backbone shared by every candidate. enable_categorical is NOT
    optional — without it the categorical team-ID columns cannot be consumed
    and the tuned config would not match production."""
    p = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "enable_categorical": True,
        "seed": RANDOM_SEED,
    }
    if sampled:
        # "_rounds" is private bookkeeping; enable_categorical is the
        # production routing contract and can never be sampled away.
        p.update({k: v for k, v in sampled.items()
                  if k not in ("_rounds", "enable_categorical")})
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

    # Fixed folds — generated once, reused by every trial. Frames are fully
    # materialized up front (identical for every trial).
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
    n_num, n_cat = len(FEATURE_COLS), len(TREE_CATEGORICAL_COLS)
    print(f"fold frames: {n_num} numeric + {n_cat} categorical team-ID cols "
          f"(production routing via _tree_dataframe)")

    # ----------------------------- study -----------------------------------
    def objective(trial: optuna.Trial) -> float:
        sampled = {
            "max_depth": trial.suggest_int("max_depth", 1, 3),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "gamma": trial.suggest_float("gamma", 0.5, 5.0),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
            "eta": trial.suggest_float("eta", 0.01, 0.06),
        }
        params = base_params(sampled)
        pooled_pred, pooled_y, iters = [], [], []
        for fold in fold_data:
            proba, best = fit_fold(params, fold, early_stop=True)
            pooled_pred.append(np.clip(proba, 1e-6, 1 - 1e-6))
            pooled_y.append(fold["y_val"])
            iters.append(best)
        trial.set_user_attr("mean_best_iter", float(np.mean(iters)))
        trial.set_user_attr("median_best_iter", float(np.median(iters)))
        return log_loss(np.concatenate(pooled_y), np.concatenate(pooled_pred))

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
    # Verbatim XGBOOST_PARAMS — the exact object a production fit-only refit
    # consumes (sklearn defaults for anything it omits). No re-derived or
    # stale hardcodes here.
    cur_params = base_params(dict(XGBOOST_PARAMS))
    cur_pred, cur_y = [], []
    for fold in fold_data:
        proba, _ = fit_fold(cur_params.copy(), fold, early_stop=False)
        cur_pred.append(np.clip(proba, 1e-6, 1 - 1e-6))
        cur_y.append(fold["y_val"])
    cur_ll = log_loss(np.concatenate(cur_y), np.concatenate(cur_pred))
    cur_auc = roc_auc_score(np.concatenate(cur_y), np.concatenate(cur_pred))
    print(f"current cfg pooled   : logloss={cur_ll:.5f}  auc={cur_auc:.4f}")
    print(f"tuned   cfg pooled   : logloss={best.value:.5f}")

    # ----------------------- hold-out verification -------------------------
    print("\n============ SEALED HOLDOUT VERIFICATION ============")
    hold_ids = _add_team_ids(hold_df)
    hold_X_num = _feature_matrix(hold_ids)
    hold_X_cat = _categorical_matrix(hold_ids)

    # Refit inputs: ALL pre-holdout games, production pipeline (median-impute
    # numerics on the refit rows + categorical routing).
    refit_ids = _add_team_ids(tune_df)
    refit_X_num = _feature_matrix(refit_ids)
    med = train_medians(refit_X_num)
    refit_frame = tree_frame(fill_medians(refit_X_num, med),
                             _categorical_matrix(refit_ids))
    # Holdout rows imputed with the REFIT medians (never their own stats).
    hold_frame = tree_frame(fill_medians(hold_X_num, med),
                            _categorical_matrix(hold_ids))
    refit_y = tune_df["home_win"].to_numpy(dtype=float)
    hold_y = hold_df["home_win"].to_numpy(dtype=float)

    # Baseline: exact production config (fit-only refit semantics).
    proba_base, _ = fit_fold(cur_params.copy(),
                             {"train_frame": refit_frame,
                              "val_frame": hold_frame,
                              "y_train": refit_y, "y_val": hold_y},
                             early_stop=False)

    # Winner: rounds = median best_iter the winning trial used across folds
    # (never touches holdout labels).
    rounds = int(max(50, min(best.user_attrs.get("median_best_iter", 300), MAX_ROUNDS)))
    win_params = base_params(bp)
    win_params["_rounds"] = rounds
    win_params["n_estimators"] = rounds
    proba_tune, _ = fit_fold(win_params,
                             {"train_frame": refit_frame,
                              "val_frame": hold_frame,
                              "y_train": refit_y, "y_val": hold_y},
                             early_stop=False)

    eps = 1e-6
    rows = []
    for name, p in (("current (XGBOOST_PARAMS verbatim)", proba_base),
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
