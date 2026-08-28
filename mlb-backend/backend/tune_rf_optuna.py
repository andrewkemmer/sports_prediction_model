"""
Optuna tuning for the RandomForest member of the MLB moneyline ensemble.

Mirrors backend/tune_mlp_optuna.py (and the XGB/LGBM tuners) structurally AND
mirrors production's RF training path (backend/training.py) exactly:

  * Same feature contract   -> training._feature_matrix (full-width
                                FEATURE_COLS, canonical order) via
                                training._prepare_features
  * SAME IMPUTATION PATH    -> train-fold-median imputation ONLY (sklearn
                                trees cannot consume NaN; never switch to a
                                native-NaN route) — val rows filled with
                                TRAIN medians.
  * SAME CATEGORICAL PATH   -> integer team-ID categoricals (home/away) via
                                _add_team_ids, hstacked onto the imputed
                                numeric matrix (RF_WITH_TEAM_IDS=True, the
                                production default).
  * Same fold generator     -> training.walk_forward_splits (fixed once,
                                reused by every trial).
  * MARGIN COLUMN INCLUDED  -> the production harness now attaches OOF
                                run_margin_diff (λ_home − λ_away, run engine
                                READ-ONLY) to fold frames inside
                                walk_forward_evaluate. This tuner calls
                                training._attach_oof_run_margins on the SAME
                                folds so the current-config baseline measures
                                the deployed 65-column matrix — a tuner that
                                dropped the margin would not tie to the
                                deployed state and its verdict could be wrong.
                                Holdout margins come from a fit-only refit on
                                all pre-holdout games (strictly future).
  * Objective = POOLED out-of-fold logloss (one log_loss over all fold
    predictions concatenated — never a mean of per-fold scores).

Study durability:
  * Pass --storage sqlite:///path.db to persist trials; an interrupted run
    resumes where it left off (load_if_exists). Without it the study is
    in-memory. The enriched margin frame is cached under /tmp (keyed by data
    hash + fold geometry) so batched/resumed runs do not re-derive the
    run-engine OOF margins every invocation.

Hold-out verification (once, after tuning):
  * Last N days (default 21) sealed BEFORE fold generation.
  * Current production config (config.RF_PARAMS verbatim) vs the Optuna
    winner, both refit on all pre-holdout games (impute + team IDs on the
    pre-holdout set only) and scored on the sealed holdout.
  * Gate (task rule): adopt ONLY if the winner beats the current config on
    the sealed holdout on logloss OR AUC without degrading calibration
    (ECE). A losing winner is REPORTED, not re-tuned — the current config
    stays (honesty contract).

Usage:
    python tune_rf_optuna.py                          # 75 trials
    python tune_rf_optuna.py --trials 5 --smoke       # sanity run
    python tune_rf_optuna.py --max-folds 5            # plumbing check
    python tune_rf_optuna.py --storage sqlite:///rf.db --jobs 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))
try:
    from training import (
        FEATURE_COLS,
        RF_TREE_CATEGORICAL_COLS,
        TREE_CATEGORICAL_COLS,
        _attach_oof_run_margins,
        _impute_median,
        _prepare_features,
        compute_metrics,
        get_last_margin_rounds,
        walk_forward_splits,
    )
    from config import (
        DATA_DELIVERY_DIR,
        MIN_VAL_FOLD_GAMES,
        RANDOM_SEED,
        RETRAIN_CADENCE_DAYS,
        RF_PARAMS,
    )
except ImportError:  # pragma: no cover - direct execution fallback
    from backend.training import (
        FEATURE_COLS,
        RF_TREE_CATEGORICAL_COLS,
        TREE_CATEGORICAL_COLS,
        _attach_oof_run_margins,
        _impute_median,
        _prepare_features,
        compute_metrics,
        get_last_margin_rounds,
        walk_forward_splits,
    )
    from backend.config import (
        DATA_DELIVERY_DIR,
        MIN_VAL_FOLD_GAMES,
        RANDOM_SEED,
        RETRAIN_CADENCE_DAYS,
        RF_PARAMS,
    )

MARGIN_COL = "run_margin_diff"
# Probability clip used everywhere, chosen to match production's
# compute_metrics (training.py clips at 1e-7).
_EPS = 1e-7


# ---------------------------------------------------------------------------
# Data / folds — through the PRODUCTION feature builders incl. the margin
# ---------------------------------------------------------------------------
def load_games(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.dropna(subset=["home_win"]).reset_index(drop=True)
    return df


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _regen_folds(enriched: pd.DataFrame, min_val_games: int) -> list[dict]:
    return [s for s in walk_forward_splits(
        enriched, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
        if len(s["val_games"]) >= min_val_games]


def prepare_data(holdout_days: int, data_path: Path,
                 max_folds: int = 0, cache_dir: Path | None = None) -> dict:
    """Load + enrich with the shipped run-margin feature, split the sealed
    holdout, and build the fixed folds ONCE (reused by every trial).

    The OOF margin join (training._attach_oof_run_margins) reproduces the
    deployed 65-column matrix: for each fold the run engine trains on that
    fold's train games ONLY and predicts its val games (leakage-free by
    construction, fold geometry asserted). Holdout margins are a fit-only
    refit on all pre-holdout games at the median fold round count.

    Returns a dict with games/tune/hold DataFrames, the fixed folds, the
    median fit rounds, and the uncovered-game count.
    """
    games = load_games(data_path)
    cutoff = games["game_date"].max() - pd.Timedelta(days=holdout_days - 1)
    tune_df = games[games["game_date"] < cutoff].reset_index(drop=True)
    hold_df = games[games["game_date"] >= cutoff].reset_index(drop=True)

    folds = _regen_folds(tune_df, MIN_VAL_FOLD_GAMES)
    if max_folds > 0:
        folds = folds[-max_folds:]
        # Plumbing path (never final tuning): skip the run-engine margin
        # derivation entirely — the margin column is all-NaN and handled by
        # the train-median imputation path, so the folds/feature contract
        # and RF fits are exercised without the per-fold run-engine cost.
        tune_enriched = tune_df.copy()
        tune_enriched[MARGIN_COL] = np.nan
        hold_enriched = hold_df.copy()
        hold_enriched[MARGIN_COL] = np.nan
        rounds: dict[str, int] = {}
        uncov = 0
    else:
        from build_oof_margin import refit_run_margins

        if cache_dir is not None:
            key = hashlib.sha256()
            key.update(_sha256_file(data_path).encode())
            key.update(json.dumps([str(s["val_start"]) for s in folds]).encode())
            h = key.hexdigest()[:16]
            cache = cache_dir / f"rf_tune_enriched_{h}.parquet"
            meta_path = cache.with_suffix(".meta.json")
            if cache.exists() and meta_path.exists():
                meta = json.loads(meta_path.read_text())
                tune_enriched = pd.read_parquet(cache)
                regen = _regen_folds(tune_enriched, MIN_VAL_FOLD_GAMES)
                if ([str(s["val_start"]) for s in regen] == meta["val_starts"]
                        and len(regen) == len(folds)):
                    folds = regen
                    rounds = {k: int(v) for k, v in meta["rounds"].items()}
                    uncov = int(meta["n_uncovered"])
                    print(f"margin cache hit ({len(folds)} folds) — no "
                          f"run-engine re-derivation")
                else:
                    cache.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    tune_enriched, folds, rounds, uncov = _build_margins(
                        tune_df, folds)
            else:
                tune_enriched, folds, rounds, uncov = _build_margins(
                    tune_df, folds)
                tune_enriched.to_parquet(cache)
                meta_path.write_text(json.dumps({
                    "rounds": {k: int(v) for k, v in rounds.items()},
                    "n_uncovered": int(uncov),
                    "val_starts": [str(s["val_start"]) for s in folds],
                }))
        else:
            tune_enriched, folds, rounds, uncov = _build_margins(
                tune_df, folds)

        hold_margins = refit_run_margins(tune_df, hold_df, rounds)
        hold_enriched = hold_df.copy()
        hold_enriched = hold_enriched.drop(
            columns=[MARGIN_COL] if MARGIN_COL in hold_enriched.columns else [])
        hold_enriched = hold_enriched.merge(
            hold_margins[["game_pk", MARGIN_COL]], on="game_pk", how="left")

    covered = float(tune_enriched[MARGIN_COL].notna().mean())
    print(f"games={len(games)}  tuning={len(tune_enriched)}  "
          f"HOLDOUT(sealed)={len(hold_df)}  "
          f"[{hold_df['game_date'].min().date()} → "
          f"{hold_df['game_date'].max().date()}]")
    print(f"fixed folds: {len(folds)} | margin coverage tuning: "
          f"{100 * covered:.1f}% | uncovered_tune_games={uncov} "
          f"(NaN → train-median imputation, the production path) | "
          f"median rounds: {rounds}")
    return {"games": games, "tune": tune_enriched, "hold": hold_enriched,
            "folds": folds, "rounds": rounds, "n_uncovered": int(uncov)}


def _build_margins(tune_df: pd.DataFrame,
                   folds: list[dict]) -> tuple[pd.DataFrame, list[dict],
                                               dict, int]:
    """Attach leakage-free OOF margins on the moneyline's own folds (run
    engine READ-ONLY). Returns (enriched frame, regenerated folds, median
    rounds, uncovered decided games)."""
    print("building OOF run margins on the fixed folds "
          "(run engine READ-ONLY) ...", flush=True)
    tune_enriched, regen = _attach_oof_run_margins(
        tune_df, folds, min_val_games=MIN_VAL_FOLD_GAMES, max_eval_folds=0,
        retrain_cadence_days=RETRAIN_CADENCE_DAYS, min_train_days=0)
    rounds = get_last_margin_rounds() or {}
    covered = set(regen[-1]["val_games"]["game_pk"]) if regen else set()
    all_val = set().union(*(set(s["val_games"]["game_pk"]) for s in regen)) \
        if regen else set()
    uncov = int((~tune_df["game_pk"].isin(all_val)).sum())
    return tune_enriched, regen, rounds, uncov


# ---------------------------------------------------------------------------
# Fold-level training — production contract: impute + team IDs, NO scaling
# ---------------------------------------------------------------------------
def prepare_fold(tr: pd.DataFrame, va: pd.DataFrame) -> dict:
    """The exact production RF input layout: train-median-imputed full-width
    numeric matrix hstacked with integer TEAM-ID categoricals (the same
    matrix production fits X_train_lr_tree on — RF stays on the team pair
    while LGB/XGB get the full TREE_CATEGORICAL_COLS set). RF never sees NaN
    and never sees StandardScaler output (unlike MLP)."""
    X_tr, X_cat_tr, y_tr = _prepare_features(tr)
    X_va, X_cat_va, y_va = _prepare_features(va)
    X_tr_i, med = _impute_median(X_tr)
    X_va_i, _ = _impute_median(X_va, med)
    rf_idx = [TREE_CATEGORICAL_COLS.index(c) for c in RF_TREE_CATEGORICAL_COLS]
    return {
        "X_train": np.hstack([X_tr_i, X_cat_tr[:, rf_idx]]),
        "X_val": np.hstack([X_va_i, X_cat_va[:, rf_idx]]),
        "y_train": y_tr.astype(float),
        "y_val": y_va.astype(float),
        "medians": med,
    }


def make_model(params: dict):
    from sklearn.ensemble import RandomForestClassifier

    clean = {k: v for k, v in params.items() if not k.startswith("_")}
    # Seed + parallelism are never search dimensions: production always
    # seeds for reproducibility and fits with n_jobs=-1.
    clean["random_state"] = RANDOM_SEED
    clean["n_jobs"] = -1
    return RandomForestClassifier(**clean)


def fit_fold(params: dict, fold: dict) -> np.ndarray:
    """Train one RF on the fold's train rows; returns validation probs."""
    model = make_model(params)
    model.fit(fold["X_train"], fold["y_train"])
    return model.predict_proba(fold["X_val"])[:, 1]


def base_params(sampled: dict | None) -> dict:
    """Fixed backbone (current production config). Sampled params override
    the search dimensions only; random_state/n_jobs forced in make_model."""
    p = dict(RF_PARAMS)
    if sampled:
        p.update({k: v for k, v in sampled.items()
                  if not k.startswith("_")})
    return p


# max_features string-codes (sqlite-persistent categorical values — floats
# mixed with strings are not stable across resume), decoded in the objective.
_MAX_FEATURES_CODES = {
    "sqrt": "sqrt", "log2": "log2",
    "0.3": 0.3, "0.4": 0.4, "0.5": 0.5, "0.6": 0.6,
    "0.7": 0.7, "0.8": 0.8,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=DATA_DELIVERY_DIR / "game_level_features.csv")
    ap.add_argument("--trials", type=int, default=75)
    ap.add_argument("--holdout-days", type=int, default=21,
                    help="Sealed tail of the schedule (2-3 weeks suggested)")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny run to verify plumbing (4 trials)")
    ap.add_argument("--max-folds", type=int, default=0,
                    help="Cap folds at the N most recent (0 = all); "
                         "plumbing checks only — never for final tuning")
    ap.add_argument("--storage", type=str, default=None,
                    help="Optuna storage URL (e.g. sqlite:///rf_study.db) "
                         "so interrupted runs resume instead of restarting")
    ap.add_argument("--study-name", type=str, default="rf_moneyline")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel Optuna workers (fold data is pickled to "
                         "each worker once; each RF fit already uses all "
                         "cores, so keep this modest)")
    ap.add_argument("--skip-verify", action="store_true",
                    help="Skip the current-config reference + sealed holdout "
                         "(batch/resume runs — the study alone is the point; "
                         "run the final verify with --trials 0)")
    args = ap.parse_args()
    if args.smoke:
        args.trials = 4

    cache_dir = Path("/tmp") if args.max_folds == 0 else None
    data = prepare_data(args.holdout_days, args.data,
                        max_folds=args.max_folds, cache_dir=cache_dir)
    tune_df, hold_df = data["tune"], data["hold"]
    folds = data["folds"]
    if args.max_folds > 0:
        print(f"  ⚠️ --max-folds={args.max_folds}: plumbing check only — "
              f"scoring the {len(folds)} most recent folds")
    fold_data = [prepare_fold(s["train_games"], s["val_games"]) for s in folds]
    n_cols = fold_data[0]["X_train"].shape[1] if fold_data else 0
    print(f"fold frames: {n_cols} cols ({len(FEATURE_COLS)} numeric + 2 team "
          f"IDs), train-median imputed, margin column "
          f"{'PRESENT' if MARGIN_COL in FEATURE_COLS else 'MISSING'}"
          if MARGIN_COL in FEATURE_COLS else
          f"fold frames: {n_cols} cols ({len(FEATURE_COLS)} numeric + 2 team "
          f"IDs), train-median imputed")

    # ----------------------------- study -----------------------------------
    def objective(trial: optuna.Trial) -> float:
        mf = trial.suggest_categorical("max_features",
                                       list(_MAX_FEATURES_CODES))
        sampled = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 200, 800, step=50),
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 6, 10, 14, 18, 22, 26, 30]),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 20),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 30),
            "max_features": _MAX_FEATURES_CODES[mf],
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
        params = base_params(sampled)
        pooled_pred, pooled_y = [], []
        for fold in fold_data:
            proba = fit_fold(params, fold)
            pooled_pred.append(np.clip(proba, _EPS, 1 - _EPS))
            pooled_y.append(fold["y_val"])
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
    bp_decoded = dict(bp)
    bp_decoded["max_features"] = _MAX_FEATURES_CODES[bp["max_features"]]
    print("\n================ STUDY RESULT ================")
    print(f"trials completed     : {len(study.trials)}")
    print(f"best POOLED OOF loss : {best.value:.5f}")
    print(f"best_params          : {bp_decoded}")

    # Current-config reference on the SAME fixed folds (pooled, honest).
    cur_pred, cur_y = [], []
    for fold in fold_data:
        proba = fit_fold(base_params(None), fold)
        cur_pred.append(np.clip(proba, _EPS, 1 - _EPS))
        cur_y.append(fold["y_val"])
    cur_y_all = np.concatenate(cur_y)
    cur_ll = log_loss(cur_y_all, np.concatenate(cur_pred))
    cur_m = compute_metrics(cur_y_all, np.concatenate(cur_pred))
    print(f"current cfg pooled   : logloss={cur_ll:.5f}  "
          f"auc={cur_m['auc']:.4f}  ece={cur_m['ece']:.4f}")
    print(f"tuned   cfg pooled   : logloss={best.value:.5f}")

    # ----------------------- hold-out verification -------------------------
    print("\n============ SEALED HOLDOUT VERIFICATION ============")
    refit = prepare_fold(tune_df, hold_df)
    hold_y = refit["y_val"]

    rows = []

    def _score(params: dict, label: str) -> None:
        proba = fit_fold(params, refit)
        m = compute_metrics(hold_y, proba)
        rows.append((label, m))

    _score(base_params(None), "current (RF_PARAMS verbatim)")
    _score(base_params(bp_decoded), "optuna winner")

    w = max(len(n) for n, _ in rows)
    print(f"{'config':<{w}} | {'logloss':>8} | {'auc':>7} | {'ece':>7}")
    for n, m in rows:
        print(f"{n:<{w}} | {m['logloss']:8.5f} | {m['auc']:7.4f} | "
              f"{m['ece']:7.4f}")
    (name_c, m_c), (name_w, m_w) = rows
    improves = (m_w["logloss"] < m_c["logloss"]) or (m_w["auc"] > m_c["auc"])
    cal_ok = m_w["ece"] <= m_c["ece"]
    verdict = "ADOPT" if (improves and cal_ok) else "DON'T ADOPT"
    print(f"\nHOLDOUT GATE: winner {'improves' if improves else 'does NOT improve'}"
          f" logloss/AUC, calibration {'OK' if cal_ok else 'DEGRADED'}"
          f" → {verdict}")
    if verdict == "DON'T ADOPT":
        print("→ The winner did NOT clear the sealed-holdout gate. Current "
              "config stays (honesty contract).")
    else:
        print("→ The winner clears the gate — adopt into config.RF_PARAMS "
              "with this provenance block.")


if __name__ == "__main__":
    main()
