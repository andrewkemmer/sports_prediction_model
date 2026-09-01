"""
Optuna re-tuning for the LightGBM member of the MLB moneyline ensemble.

Re-tune on the CURRENT 65-column matrix (the original tune_lightgbm_optuna.py
study was run on the 58-col matrix — since then lineup deltas and the run
margin shipped; the depth-5 / 50-round config is the stale assumption under
test). Structural conventions mirror backend/tune_rf_optuna.py and the
original LGB tuner:

  * Same feature contract   -> training._feature_matrix (full-width
                                FEATURE_COLS, canonical order) — now 65
                                columns including the shipped lineup deltas
                                + run_margin_diff.
  * MARGIN COLUMN INCLUDED  -> the production harness attaches OOF
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
  * NATIVE NaN ONLY         -> production LightGBM consumes the raw
                                NaN-preserving matrix natively. This re-tune
                                drops the original tuner's impute_medians
                                trial dimension — no imputation switching
                                (the guardrail: LGB routes missing values
                                natively, do NOT switch to imputation).
  * SAME TEAM-ID ROUTING    -> int home_team_id/away_team_id columns
                                (UNK_TEAM_ID clamped) appended to every
                                frame, passed as categorical_feature BY
                                NAME at EVERY fit — production's exact
                                contract.
  * Same fold generator     -> training.walk_forward_splits (fixed once,
                                reused by every trial).
  * Objective = POOLED out-of-fold logloss (one log_loss over all fold
    predictions concatenated — never a mean of per-fold scores).
  * Per fold: early stopping on that fold's own validation window
    (early_stopping_rounds=20, MAX_ROUNDS=2000 ceiling) — the re-tune
    deliberately tests the stale 50-round hard cap; the study chooses
    between few-deep and many-shallow.

SEARCH SPACE (refreshed vs the original 58-col tune — the low-signal regime
favored XGB's depth-2 shallow additive style, so depth is OPEN 2–8 and the
data decides; round counts go HIGHER with per-fold early stopping; lambda
regularization is added to police the wider space):
  max_depth 2–8, num_leaves 2–64, min_child_samples 20–200,
  min_gain_to_split 0.0–2.0, bagging_fraction 0.5–1.0 (+ bagging_freq 1–5,
  only active when fraction < 1 — structural guardrail), feature_fraction
  0.5–0.9, learning_rate 0.01–0.1, lambda_l1/l2 (log, 1e-3–10).

Study durability:
  * Pass --storage sqlite:///path.db to persist trials; an interrupted run
    resumes where it left off (load_if_exists). Without it the study is
    in-memory. The enriched margin frame is cached under /tmp (keyed by data
    hash + fold geometry) so batched/resumed runs do not re-derive the
    run-engine OOF margins every invocation.
  * NOTE: the study name is lightgbm_moneyline_65col — never point --storage
    at the ORIGINAL lightgbm_moneyline study (58 cols, impute dimension):
    a resume would silently mix two search spaces.

Hold-out verification (once, after tuning):
  * Last N days (default 21) sealed BEFORE fold generation.
  * Current production config (LIGHTGBM_PARAMS verbatim, native NaN, fixed
    50 rounds — the deployed fit) vs the Optuna winner, both refit on all
    pre-holdout games and scored on the sealed holdout. Winner rounds =
    median best_iteration across the winning trial's folds (never sees
    holdout labels).
  * Gate (task rule): adopt ONLY if the winner beats the current config on
    the sealed holdout on logloss OR AUC without degrading calibration
    (ECE). A losing winner is REPORTED, not re-tuned — the current config
    stays (honesty contract).

Usage:
    python tune_lightgbm_optuna.py                          # 75 trials
    python tune_lightgbm_optuna.py --trials 5 --smoke       # sanity run
    python tune_lightgbm_optuna.py --max-folds 5            # plumbing check
    python tune_lightgbm_optuna.py --storage sqlite:///lgbm.db --jobs 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
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
        TREE_CATEGORICAL_COLS,
        UNK_TEAM_ID,
        _add_team_ids,
        _attach_oof_run_margins,
        _categorical_matrix,
        _feature_matrix,
        compute_metrics,
        get_last_margin_rounds,
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
        _attach_oof_run_margins,
        _categorical_matrix,
        _feature_matrix,
        compute_metrics,
        get_last_margin_rounds,
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
    # Mirror production walk_forward_evaluate: keep folds below min_val_games
    # ONLY when they are the partial tail (is_partial_tail) — dropping them
    # desyncs the tuner's geometry from _attach_oof_run_margins (which
    # regenerates with the tail included) and trips its misaligned-split guard.
    return [s for s in walk_forward_splits(
        enriched, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
        if len(s["val_games"]) >= min_val_games or s.get("is_partial_tail")]


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
        # LightGBM's native NaN routing, so the folds/feature contract and
        # LGB fits are exercised without the per-fold run-engine cost.
        tune_enriched = tune_df.copy()
        tune_enriched[MARGIN_COL] = np.nan
        hold_enriched = hold_df.copy()
        hold_enriched[MARGIN_COL] = np.nan
        # Re-slice the ENRICHED frame so fold frames carry the (all-NaN)
        # margin column — the same 65-col layout as the real path (fold
        # geometry is unchanged: walk_forward_splits is a pure function of
        # game_date/home_win).
        folds = _regen_folds(tune_enriched, MIN_VAL_FOLD_GAMES)[-max_folds:]
        rounds: dict[str, int] = {}
        uncov = 0
    else:
        from build_oof_margin import refit_run_margins

        if cache_dir is not None:
            key = hashlib.sha256()
            key.update(_sha256_file(data_path).encode())
            key.update(json.dumps([str(s["val_start"]) for s in folds]).encode())
            h = key.hexdigest()[:16]
            cache = cache_dir / f"lgbm_tune_enriched_{h}.parquet"
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
          f"(NaN → native LightGBM routing, the production path) | "
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
    all_val = set().union(*(set(s["val_games"]["game_pk"]) for s in regen)) \
        if regen else set()
    uncov = int((~tune_df["game_pk"].isin(all_val)).sum())
    return tune_enriched, regen, rounds, uncov


# ---------------------------------------------------------------------------
# Fold-level training — production contract: native NaN + team IDs by name
# ---------------------------------------------------------------------------
def lgbm_frame(X_num: np.ndarray, X_cat: np.ndarray) -> "pd.DataFrame":
    """The exact production input layout: named FEATURE_COLS (NaN-preserving
    — LightGBM routes missing values natively, never imputed) plus INT
    team-ID columns clamped to UNK_TEAM_ID."""
    df = pd.DataFrame(X_num, columns=list(FEATURE_COLS))
    for i, c in enumerate(TREE_CATEGORICAL_COLS):
        vals = np.where(X_cat[:, i] < 0, UNK_TEAM_ID, X_cat[:, i]).astype(int)
        df[c] = vals
    return df


def prepare_fold(tr: pd.DataFrame, va: pd.DataFrame) -> dict:
    """The exact production LGB input layout: raw NaN-preserving full-width
    FEATURE_COLS frame (65 cols incl. run_margin_diff) + int team-ID
    categorical columns. No imputation — native NaN routing."""
    tr_ids, va_ids = _add_team_ids(tr), _add_team_ids(va)
    return {
        "frames": lgbm_frame(_feature_matrix(tr_ids),
                             _categorical_matrix(tr_ids)),
        "val_frames": lgbm_frame(_feature_matrix(va_ids),
                                 _categorical_matrix(va_ids)),
        "y_train": tr["home_win"].to_numpy(dtype=float),
        "y_val": va["home_win"].to_numpy(dtype=float),
    }


def make_model(params: dict):
    from lightgbm import LGBMClassifier
    clean = {k: v for k, v in params.items() if not k.startswith("_")}
    return LGBMClassifier(**clean)


def fit_fold(params: dict, fold: dict,
             early_stop: bool) -> tuple[np.ndarray, int]:
    """Train one model; returns (validation probabilities, rounds used).
    categorical_feature BY NAME at EVERY fit (tuned folds and plain refits),
    exactly like the production fold trainer."""
    from lightgbm import early_stopping, log_evaluation

    model = make_model(params)
    tr_frame = fold["frames"]
    va_frame = fold["val_frames"]
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
                  if not k.startswith("_")})
    return p


def sample_params(trial: optuna.Trial) -> dict:
    """The refreshed search space (65-col matrix). max_depth is OPEN 2–8
    (the original capped at 5 — the stale assumption to test), round counts
    go higher via early stopping, and lambda regularization is added to
    police the wider space. Native NaN routing only — no impute dimension."""
    sampled = {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "num_leaves": trial.suggest_int("num_leaves", 2, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 2.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    }
    params = base_params(sampled)
    # Structural guardrail: bagging_freq is only legal with
    # bagging_fraction < 1.0 (LightGBM raises otherwise).
    if sampled["bagging_fraction"] >= 1.0:
        params.pop("bagging_freq", None)
    return params


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
                    help="Optuna storage URL (e.g. sqlite:///lgbm_study.db) "
                         "so interrupted runs resume instead of restarting")
    ap.add_argument("--study-name", type=str,
                    default="lightgbm_moneyline_65col",
                    help="Study name. The 65-col re-tune has its own name — "
                         "never resume the original 58-col lightgbm_moneyline "
                         "study (different search space)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel Optuna workers (fold data is pickled to "
                         "each worker once; LGB fits are fast, so this is "
                         "for batch/resume runs on many cores)")
    ap.add_argument("--skip-verify", action="store_true",
                    help="Skip the current-config reference + sealed holdout "
                         "(batch/resume runs — the study alone is the point; "
                         "run the final verify with --trials 0)")
    args = ap.parse_args()
    if args.smoke:
        args.trials = 4

    cache_dir = Path(tempfile.gettempdir()) if args.max_folds == 0 else None
    data = prepare_data(args.holdout_days, args.data,
                        max_folds=args.max_folds, cache_dir=cache_dir)
    tune_df, hold_df = data["tune"], data["hold"]
    folds = data["folds"]
    if args.max_folds > 0:
        print(f"  ⚠️ --max-folds={args.max_folds}: plumbing check only — "
              f"scoring the {len(folds)} most recent folds")
    fold_data = [prepare_fold(s["train_games"], s["val_games"]) for s in folds]
    frame = fold_data[0]["frames"] if fold_data else None
    if frame is not None:
        print(f"fold frames: {len(FEATURE_COLS)} numeric + "
              f"{len(TREE_CATEGORICAL_COLS)} categorical team-ID cols "
              f"({len(frame.columns)} total), native NaN routing, margin "
              f"column {'PRESENT' if MARGIN_COL in FEATURE_COLS else 'MISSING'}")

    # ----------------------------- study -----------------------------------
    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial)
        pooled_pred, pooled_y, iters = [], [], []
        for fold in fold_data:
            proba, best = fit_fold(params, fold, early_stop=True)
            pooled_pred.append(np.clip(proba, _EPS, 1 - _EPS))
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
        proba, _ = fit_fold(base_params(dict(LIGHTGBM_PARAMS)), fold,
                            early_stop=False)
        cur_pred.append(np.clip(proba, _EPS, 1 - _EPS))
        cur_y.append(fold["y_val"])
    cur_y_all = np.concatenate(cur_y)
    cur_m = compute_metrics(cur_y_all, np.concatenate(cur_pred))
    print(f"current cfg pooled   : logloss={cur_m['logloss']:.5f}  "
          f"auc={cur_m['auc']:.4f}  ece={cur_m['ece']:.4f}  "
          f"(LIGHTGBM_PARAMS verbatim, 50r)")
    print(f"tuned   cfg pooled   : logloss={best.value:.5f}")

    # ----------------------- hold-out verification -------------------------
    print("\n============ SEALED HOLDOUT VERIFICATION ============")
    refit = prepare_fold(tune_df, hold_df)
    hold_y = refit["y_val"]

    rows = []

    def _score(params: dict, label: str) -> None:
        proba, _ = fit_fold(params, refit, early_stop=False)
        m = compute_metrics(hold_y, proba)
        rows.append((label, m))

    _score(base_params(dict(LIGHTGBM_PARAMS)),
           "current (LIGHTGBM_PARAMS verbatim, 50r)")
    # Winner: rounds = median best_iter the winning trial used across its
    # folds (holdout labels untouched by that choice).
    rounds = int(max(50, min(best.user_attrs.get("median_best_iter", 300),
                             MAX_ROUNDS)))
    win_params = base_params(bp)
    win_params["n_estimators"] = rounds
    if win_params.get("bagging_fraction", 1.0) >= 1.0:
        win_params.pop("bagging_freq", None)
    _score(win_params, f"optuna winner ({rounds}r)")

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
        print("→ The winner clears the gate — adopt into config.LIGHTGBM_PARAMS "
              "with this provenance block.")
    print(f"winning trial median best_iter across folds: "
          f"{best.user_attrs.get('median_best_iter', float('nan')):.1f}  "
          f"(mean {best.user_attrs.get('mean_best_iter', float('nan')):.1f})")


if __name__ == "__main__":
    main()
