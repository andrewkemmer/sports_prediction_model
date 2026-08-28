"""Leakage-free OOF run-margin feature for the moneyline (READ-ONLY over the
run engine).

run_margin_diff = lam_home - lam_away, where the lambdas come from the run
engine's per-side LightGBM Poisson models trained on ITS OWN 29-feature
levels+env view — computed OUT-OF-FOLD on the MONEYLINE harness's fold split,
so no game's margin ever comes from a model that saw it.

Design contract (2026-08 margin-feature task):
- Folds are SUPPLIED BY THE CALLER — the moneyline harness's walk-forward
  splits (generated once, filtered by MIN_VAL_FOLD_GAMES), NOT the run
  engine's own fold filter.
- Run-engine machinery is reused read-only (build_side_frame /
  _fit_side_model / RUN_LGBM_PARAMS). Nothing in run_engine.py changes; the
  29-feature view and alpha(lambda) stay untouched. (Even if
  ``run_margin_diff`` itself were ever added to FEATURE_COLS,
  derive_run_features drops every ``*_diff`` except park_factor_slug_diff —
  the run view cannot leak it.)
- Early stopping uses the validation fold's targets to pick the ITERATION
  COUNT only — the same per-fold early-stopping convention every moneyline
  member and the run engine's own OOF already use. No target information
  enters a fold's fitted weights w.r.t. its own train rows.
- Games outside any executed fold's val window (small folds below the
  min-val gate) get NO OOF lambda -> NaN -> the moneyline's train-median
  imputation path. The count is reported loudly, never papered over.
- Holdout/sealed/inference margins come from a FIT-ONLY refit at the MEDIAN
  fold round count per side (predict_slate_runs' production convention) —
  strictly-future games cannot leak by construction.

Usage:
    python3 build_oof_margin.py --out /tmp/oof_run_margin.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_engine import (  # noqa: E402
    MAX_ROUNDS,
    RUN_LGBM_PARAMS,
    _fit_side_model,
    build_side_frame,
)

SIDES = (("home", "home_score"), ("away", "away_score"))
MARGIN_COL = "run_margin_diff"
LAM_COLS = ("lam_home", "lam_away")


def _side_frames(games: pd.DataFrame,
                 run_features: list[str] | None = None,
                 dropped: list[str] | None = None,
                 ) -> dict[str, tuple[pd.DataFrame, list[str]]]:
    """Materialize both sides' run-engine model frames once (feature columns
    only — no target information in construction)."""
    return {side: build_side_frame(games, side, run_features=run_features,
                                   dropped=dropped)
            for side, _target in SIDES}


def _fit_predict_side(frames_side: tuple[pd.DataFrame, list[str]],
                      tr: pd.DataFrame, va: pd.DataFrame,
                      target: str, params: dict) -> tuple[np.ndarray, int]:
    """Train one side on the fold's train rows, predict lambda for its val
    rows. Returns (lam, best_iteration)."""
    cols_all = frames_side[1]
    tr_f = tr.reindex(columns=cols_all).astype(float)
    va_f = va.reindex(columns=cols_all).astype(float)
    y_tr = tr[target].to_numpy(dtype=float)
    y_va = va[target].to_numpy(dtype=float)
    _model, lam, best = _fit_side_model(params, tr_f, y_tr, va_f, y_va)
    return lam, int(best)


def oof_run_margins(tune_df: pd.DataFrame, folds: list[dict],
                    run_features: list[str] | None = None,
                    dropped: list[str] | None = None,
                    ) -> tuple[pd.DataFrame, dict[str, int], int]:
    """OOF margins on the caller's folds.

    Returns (margins_df with [game_pk, fold_idx, lam_home, lam_away,
    run_margin_diff], median_best_rounds per side, n_uncovered_games —
    decided tune games that sit in NO executed fold's val window and will be
    NaN-imputed downstream).
    Deterministic: identical folds + seed => identical table.
    """
    games = tune_df[tune_df["home_win"].notna()].reset_index(drop=True)
    frames = _side_frames(games, run_features, dropped)
    params = dict(RUN_LGBM_PARAMS)

    parts: list[pd.DataFrame] = []
    best_iters: dict[str, list[int]] = {s: [] for s, _t in SIDES}
    for split in folds:
        tr, va = split["train_games"], split["val_games"]
        # Fold-boundary guard: every predicted game must lie strictly after
        # the training window. walk_forward_splits guarantees this; asserted
        # here so a future split change cannot silently break the contract.
        tr_end = pd.to_datetime(tr["game_date"]).max()
        va_start = pd.to_datetime(va["game_date"]).min()
        if not (va_start > tr_end):
            raise AssertionError(
                f"fold {split['fold_idx']}: val starts {va_start} <= "
                f"train end {tr_end} — leakage-safe split violated")
        rec = {"game_pk": va["game_pk"].to_numpy(),
               "fold_idx": int(split["fold_idx"])}
        for side, target in SIDES:
            lam, best = _fit_predict_side(frames[side], tr, va, target, params)
            best_iters[side].append(best)
            rec[f"lam_{side}"] = np.round(lam, 5)
        rec[MARGIN_COL] = np.round(rec["lam_home"] - rec["lam_away"], 5)
        parts.append(pd.DataFrame(rec))

    margins = pd.concat(parts, ignore_index=True)
    median_rounds = {
        s: int(np.median(best_iters[s])) if best_iters[s] else MAX_ROUNDS
        for s, _t in SIDES}
    covered = set(margins["game_pk"])
    n_uncovered = int((~games["game_pk"].isin(covered)).sum())
    return margins, median_rounds, n_uncovered


def refit_run_margins(games_all: pd.DataFrame, pred_df: pd.DataFrame,
                      n_rounds: dict[str, int],
                      run_features: list[str] | None = None,
                      dropped: list[str] | None = None,
                      ) -> pd.DataFrame:
    """Fit-only refit of both side models on ``games_all`` at FIXED round
    counts (the production slate convention — predict_slate_runs), then
    predict lambdas + margin for ``pred_df`` rows.

    Used for (a) the sealed-holdout margin (fit on all pre-holdout data) and
    (b) production inference for upcoming games (fit on all decided games —
    strictly future, cannot leak by construction)."""
    frames = _side_frames(games_all, run_features, dropped)
    rec: dict = {"game_pk": pred_df["game_pk"].to_numpy()}
    for side, target in SIDES:
        cols_all = frames[side][1]
        tr_f = games_all.reindex(columns=cols_all).astype(float)
        pr_f = pred_df.reindex(columns=cols_all).astype(float)
        y_tr = games_all[target].to_numpy(dtype=float)
        from lightgbm import LGBMRegressor, log_evaluation
        model = LGBMRegressor(**{**RUN_LGBM_PARAMS, "n_estimators": n_rounds[side]})
        model.fit(tr_f, y_tr, callbacks=[log_evaluation(period=0)])
        lam = np.clip(model.predict(pr_f), 1e-6, None)
        rec[f"lam_{side}"] = np.round(lam, 5)
    rec[MARGIN_COL] = np.round(rec["lam_home"] - rec["lam_away"], 5)
    return pd.DataFrame(rec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/oof_run_margin.parquet"))
    args = ap.parse_args()

    import subprocess
    from config import DATA_DELIVERY_DIR, RETRAIN_CADENCE_DAYS, MIN_VAL_FOLD_GAMES
    from data_ingestion import load_game_features
    import training

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True,
                         cwd=str(_repo_root())).stdout.strip()
    csv_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    games = load_game_features(csv_path)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_win"]).reset_index(drop=True)

    cutoff = games["game_date"].max() - pd.Timedelta(days=20)
    tune_df = games[games["game_date"] < cutoff].reset_index(drop=True)
    hold_df = games[games["game_date"] >= cutoff].reset_index(drop=True)
    all_splits = training.walk_forward_splits(
        tune_df, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
    folds = [s for s in all_splits if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]

    print(f"commit={sha} games={len(games)} tuning={len(tune_df)} "
          f"holdout={len(hold_df)} folds={len(all_splits)}/{len(folds)}")
    margins, rounds, n_uncov = oof_run_margins(tune_df, folds)
    hold_margins = refit_run_margins(tune_df, hold_df, rounds)
    out = pd.concat([margins, hold_margins], ignore_index=True)
    out.to_parquet(args.out)
    print(f"margin rows={len(out)} (oof={len(margins)}, "
          f"holdout={len(hold_margins)}) uncovered_tune_games={n_uncov}")
    print(f"median rounds: {rounds}")
    print(f"margin stats: mean={out[MARGIN_COL].mean():.4f} "
          f"std={out[MARGIN_COL].std():.4f} min={out[MARGIN_COL].min():.2f} "
          f"max={out[MARGIN_COL].max():.2f}")
    print(f"-> {args.out}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    main()
