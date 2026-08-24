"""Run engine Phase 1 — per-team expected-runs models (λ per side).

THE GOLDEN RULE: run models consume LEVELS + ENVIRONMENT only. Diff features
(sp_era_diff ≈ 0 for ace-vs-ace AND bad-vs-bad) carry no information about
scoring LEVEL, so they are excluded — EXCEPT park_factor_slug_diff, which is a
park-context term, and the four engineered interactions that are products of
excluded diffs. The kept list is DERIVED from FEATURE_COLS at call time so new
features flow in (and are logged); only the exclusion RULE lives here.

One regularized LightGBM regressor (objective="poisson") per side, trained on
the SAME fixed walk-forward folds as the moneyline pipeline. Pooled OOF scoring,
baseline comparison vs constant league-mean, Pearson chi-square/dispersion
probe for the Phase-2 Poisson-vs-negative-binomial decision.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from config import (
    DATA_DELIVERY_DIR,
    MIN_VAL_FOLD_GAMES,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
)

logger = logging.getLogger(__name__)

# Excluded despite not ending in _diff: pure matchup term (no level info) and
# interactions whose factors are themselves excluded diffs.
RUN_EXTRA_EXCLUSIONS = {
    "lineup_handedness_matchup_advantage",
    "bullpen_meltdown_risk",          # pitches_diff × whip_diff
    "pitcher_regression_indicator",   # velo_diff × era_diff
    "lineup_depth_multiplier",        # woba_mean_diff × top3_diff
    "ace_efficiency_factor",          # k9_diff × whiff_diff
}
# The one sanctioned _diff survivor: a PARK context multiplier, not a matchup gap.
RUN_DIFF_EXCEPTION = "park_factor_slug_diff"

MAX_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 20  # matches the tuned LightGBM fold convention

RUN_LGBM_PARAMS = {
    "objective": "poisson",
    "learning_rate": 0.05,
    "num_leaves": 8,
    "min_child_samples": 40,
    "min_gain_to_split": 0.5,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}


# ---------------------------------------------------------------------------
# Feature-view derivation
# ---------------------------------------------------------------------------
def derive_run_features(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Derive (run_features, dropped) from FEATURE_COLS by rule:

      drop  f  if f.endswith("_diff") and f != RUN_DIFF_EXCEPTION
         or  f in RUN_EXTRA_EXCLUSIONS

    Everything else flows in automatically (new level/env features included
    without touching this file). Returns both lists so callers log the drops.
    """
    run_feats, dropped = [], []
    for f in feature_cols:
        if f.endswith("_diff") and f != RUN_DIFF_EXCEPTION:
            dropped.append(f)
        elif f in RUN_EXTRA_EXCLUSIONS:
            dropped.append(f)
        else:
            run_feats.append(f)
    return run_feats, dropped


def split_side_view(run_features: list[str],
                    side: str) -> tuple[list[str], list[str]]:
    """Split run features into (side_view, env_shared) for 'home' or 'away'.

    Side columns: *_home / home_* (symmetrically for away). Shared environment
    = everything that belongs to NEITHER side (dome flag, park multiplier,
    weather, is_home) — never the opponent's levels."""
    other = "away" if side == "home" else "home"
    side_cols = [f for f in run_features
                 if f.endswith(f"_{side}") or f.startswith(f"{side}_")]
    other_cols = {f for f in run_features
                  if f.endswith(f"_{other}") or f.startswith(f"{other}_")}
    env_cols = [f for f in run_features
                if f not in side_cols and f not in other_cols]
    return side_cols, env_cols


def build_side_frame(games: pd.DataFrame, side: str,
                     run_features: Optional[list[str]] = None,
                     dropped: Optional[list[str]] = None
                     ) -> tuple[pd.DataFrame, list[str]]:
    """Materialize the side's model frame (levels + environment), preserving
    NaN (LightGBM routes it natively). Logs the derivation once per call."""
    from training import FEATURE_COLS

    feats = list(run_features) if run_features is not None else None
    if feats is None or dropped is None:
        feats, dropped = derive_run_features(list(FEATURE_COLS))
        logger.info("Run engine: %d/%d features kept; dropped %d: %s",
                    len(feats), len(FEATURE_COLS), len(dropped), dropped)
    side_cols, env_cols = split_side_view(feats, side)
    cols = side_cols + env_cols
    frame = games.reindex(columns=cols).astype(float)
    return frame, cols


# ---------------------------------------------------------------------------
# Training / OOF scoring
# ---------------------------------------------------------------------------
def _fit_side_model(params: dict, tr_frame: pd.DataFrame, y_tr: np.ndarray,
                    va_frame: pd.DataFrame, y_va: np.ndarray):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    model = LGBMRegressor(**params)
    model.set_params(n_estimators=MAX_ROUNDS)
    model.fit(
        tr_frame, y_tr,
        eval_set=[(va_frame, y_va)],
        callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                   log_evaluation(period=0)],
    )
    best = int(model.best_iteration_ or MAX_ROUNDS)
    lam = np.clip(model.predict(va_frame, num_iteration=best), 1e-6, None)
    return model, lam, best


def poisson_deviance(y: np.ndarray, lam: np.ndarray) -> float:
    """Mean unit deviance: 2·[y·ln(y/λ) − (y − λ)]; the y=0 term reduces to 2λ."""
    y = np.asarray(y, dtype=float)
    lam = np.clip(np.asarray(lam, dtype=float), 1e-10, None)
    terms = np.empty_like(y)
    pos = y > 0
    terms[pos] = y[pos] * np.log(y[pos] / lam[pos]) - (y[pos] - lam[pos])
    terms[~pos] = lam[~pos]
    return float(2.0 * terms.mean())


def dispersion_ratio(y: np.ndarray, lam: np.ndarray) -> float:
    """Pearson chi-square / df. ≈1 → Poisson variance is adequate;
    clearly >1 → over-dispersion (negative-binomial upgrade territory)."""
    y = np.asarray(y, dtype=float)
    lam = np.clip(np.asarray(lam, dtype=float), 1e-10, None)
    chi2 = float(((y - lam) ** 2 / lam).sum())
    df = max(int(len(y)) - 1, 1)
    return chi2 / df


def run_oof(games: pd.DataFrame,
            retrain_cadence_days: int = RETRAIN_CADENCE_DAYS,
            min_val_games: int = MIN_VAL_FOLD_GAMES,
            ) -> dict[str, Any]:
    """Walk-forward OOF for both side models on the moneyline pipeline's folds.

    Returns rows (one per decided game), per-side metrics, baseline metrics,
    and the dispersion probe."""
    from training import walk_forward_splits

    games = games[games["home_win"].notna()].reset_index(drop=True)
    folds = [
        s for s in walk_forward_splits(games, retrain_cadence_days=retrain_cadence_days)
        if len(s["val_games"]) >= min_val_games
    ]
    logger.info("Run engine: %d walk-forward folds, %d decided games",
                len(folds), len(games))

    frames = {side: build_side_frame(games, side) for side in ("home", "away")}
    params = dict(RUN_LGBM_PARAMS)

    out_rows: list[dict] = []
    metrics: dict[str, dict[str, list[float]]] = {
        s: {"deviance": [], "rmse": [], "mae": []} for s in ("home", "away")}
    base_metrics: dict[str, dict[str, list[float]]] = {
        s: {"deviance": [], "rmse": [], "mae": []} for s in ("home", "away")}

    for split in folds:
        tr, va = split["train_games"], split["val_games"]
        rec_base = {"game_pk": va["game_pk"].to_numpy(),
                    "game_date": pd.to_datetime(va["game_date"]).dt.strftime("%Y-%m-%d"),
                    "fold_idx": split["fold_idx"]}
        for side, target in (("home", "home_score"), ("away", "away_score")):
            _, cols_all = frames[side]
            tr_frame = tr.reindex(columns=cols_all).astype(float)
            va_frame = va.reindex(columns=cols_all).astype(float)
            y_tr = tr[target].to_numpy(dtype=float)
            y_va = va[target].to_numpy(dtype=float)
            _, lam, best = _fit_side_model(params, tr_frame, y_tr, va_frame, y_va)
            key = f"{side}_expected_runs"
            rec_base[key] = np.round(lam, 4)
            rec_base[target] = y_va.astype(int)
            m = metrics[side]
            m["deviance"].append(poisson_deviance(y_va, lam))
            m["rmse"].append(float(np.sqrt(((y_va - lam) ** 2).mean())))
            m["mae"].append(float(np.abs(y_va - lam).mean()))
            # Constant league-mean baseline (TRAIN mean — no val leakage).
            mu = float(y_tr.mean())
            bm = base_metrics[side]
            bm["deviance"].append(poisson_deviance(y_va, np.full(len(y_va), mu)))
            bm["rmse"].append(float(np.sqrt(((y_va - mu) ** 2).mean())))
            bm["mae"].append(float(np.abs(y_va - mu).mean()))
        out_rows.extend(pd.DataFrame(rec_base).to_dict(orient="records"))

    oof = pd.DataFrame(out_rows).sort_values(["game_date", "game_pk"])
    summary: dict[str, Any] = {"n_folds": len(folds), "n_games": len(oof)}
    for side in ("home", "away"):
        summary[f"{side}_model"] = {
            "poisson_deviance" if k == "deviance" else k:
                round(float(np.average(v)), 5)
            for k, v in metrics[side].items()}  # fold-mean; pooled below is exact
        summary[f"{side}_baseline"] = {
            "poisson_deviance" if k == "deviance" else k:
                round(float(np.average(v)), 5)
            for k, v in base_metrics[side].items()}
    # Pooled (exact) recomputation over all OOF rows — never fold averages.
    for side in ("home", "away"):
        y = oof[f"{side}_score"].to_numpy(float)
        lam = oof[f"{side}_expected_runs"].to_numpy(float)
        summary[f"{side}_pooled"] = {
            "poisson_deviance": round(poisson_deviance(y, lam), 5),
            "rmse": round(float(np.sqrt(((y - lam) ** 2).mean())), 5),
            "mae": round(float(np.abs(y - lam).mean()), 5),
        }
        summary[f"{side}_dispersion_ratio"] = round(dispersion_ratio(y, lam), 4)
    return {"oof": oof, "summary": summary}


OOF_COLUMNS = ["game_pk", "game_date", "home_expected_runs",
               "away_expected_runs", "home_score", "away_score"]


def persist_oof(oof: pd.DataFrame, target_date_str: str,
                out_dir: Optional[Path] = None) -> Path:
    """run_engine_oof_<date>.csv — Phase 2's input contract."""
    out_path = (out_dir or DATA_DELIVERY_DIR) / f"run_engine_oof_{target_date_str}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing = [c for c in OOF_COLUMNS if c not in oof.columns]
    if missing:
        raise ValueError(f"OOF frame missing required columns: {missing}")
    tmp = out_path.with_suffix(".csv.tmp")
    oof[OOF_COLUMNS].to_csv(tmp, index=False)
    tmp.replace(out_path)
    logger.info("Run engine OOF: %d rows -> %s", len(oof), out_path.name)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=DATA_DELIVERY_DIR / "game_level_features.csv")
    ap.add_argument("--date", type=str, default=None,
                    help="Artifact date stamp (default: today)")
    args = ap.parse_args()
    import datetime
    date_stamp = args.date or datetime.date.today().strftime("%Y%m%d")

    df = pd.read_csv(args.data)
    result = run_oof(df)

    s = result["summary"]
    print("\n================ RUN ENGINE PHASE 1 ================")
    print(f"folds={s['n_folds']}  oof games={s['n_games']}")
    print(f"\n{'metric':<18}{'home model':>12}{'away model':>12}{'baseline':>10}")
    for metric in ("poisson_deviance", "rmse", "mae"):
        print(f"{metric:<18}"
              f"{s['home_pooled'][metric]:>12.4f}"
              f"{s['away_pooled'][metric]:>12.4f}"
              f"{s['home_baseline'][metric]:>10.4f}"
              f"  (away base {s['away_baseline'][metric]:.4f})")
    for side in ("home", "away"):
        ratio = s[f"{side}_dispersion_ratio"]
        verdict = "Poisson OK" if ratio < 1.3 else (
            "MILD over-dispersion" if ratio < 1.8 else "STRONG over-dispersion")
        print(f"{side} dispersion ratio (Pearson χ²/df): {ratio:.3f} → {verdict}")

    path = persist_oof(result["oof"], date_stamp)
    print(f"\nOOF artifact: {path}")


if __name__ == "__main__":
    main()
