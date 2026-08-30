"""Run engine Phase 1 — per-team expected-runs models (λ per side).

THE GOLDEN RULE: run models consume LEVELS + ENVIRONMENT only. Diff features
(sp_era_diff ≈ 0 for ace-vs-ace AND bad-vs-bad) carry no information about
scoring LEVEL, so they are excluded — EXCEPT park_factor_slug_diff, which is a
park-context term, and the four engineered interactions that are products of
excluded diffs. The kept list is DERIVED from FEATURE_COLS at call time so new
features flow in (and are logged); only the exclusion RULE lives here.

2026-08 run-engine-native keep-list ablation (run_engine_keep_ablation.py,
data_delivery/run_engine_keep_ablation_2bc3ba1*.json) — verdict: the 24
matchup-gap _diff features STAY EXCLUDED (DO NOT SHIP). Restoring them (arm B,
53 cols) improved the count objective on BOTH sides — home Δdev −0.0032 /
ΔRMSE −0.0026; away Δdev −0.0131 / ΔRMSE −0.0084 (pooled OOF, 48 folds /
4,354 games) — but DEGRADED the totals-market calibration the engine sells:
prequential ECE-cal on the 6 reference lines rose 0.0119 → 0.0144 pooled
(over_8_5 0.0109 → 0.0178) and again on the sealed 21-day holdout
(over_7_5/8_5/9_5 +0.008/+0.010/+0.013); tail fit (P(X≤1)/P(X≥10), χ²/df)
unchanged. The moneyline-side audit (feature_audit_3b929cfcf3e2.json)
recommended restore through the binary harness — that result does NOT
transfer to λ: the diff-level signal helps the count model but hurts the
prices. Gate: B must beat A on core metrics on BOTH sides without degrading
market calibration (ECE-cal, tail fit) — core won, calibration leg failed.

One regularized LightGBM regressor (objective="poisson") per side, trained on
the SAME fixed walk-forward folds as the moneyline pipeline. Pooled OOF scoring,
baseline comparison vs constant league-mean, Pearson chi-square/dispersion
probe for the Phase-2 Poisson-vs-negative-binomial decision.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional

import json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from frames import get_decided_frame, fold_signature

from config import (
    AGREEMENT_FILTER_DELTA,
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
    # Phase 2 lineup-delta features (actual starting-9 wOBA vs team season) —
    # matchup/form signal, moneyline-only; excluded so the run engine's
    # raw-only view stays byte-identical (GOLDEN RULE: levels + environment).
    "lineup_actual_woba_delta_home", "lineup_actual_woba_delta_away",
    "lineup_actual_top3_delta_home", "lineup_actual_top3_delta_away",
    "lineup_rest_count_home", "lineup_rest_count_away",
    # Categorical-context columns (venue + the two starter IDs) — moneyline
    # TREE_CATEGORICAL_COLS inputs only, not scoring LEVELS. They are not in
    # FEATURE_COLS today; named here so the rule also drops them if they ever
    # enter a frame the run engine derives from (kept/dropped lists from
    # FEATURE_COLS stay byte-identical — asserted in
    # test_categorical_venue_starters.py).
    "venue", "home_starter_id", "away_starter_id",
}
# The one sanctioned _diff survivor: a PARK context multiplier, not a matchup gap.
RUN_DIFF_EXCEPTION = "park_factor_slug_diff"

# Phase 3.5b — standalone ENVIRONMENT-LEVEL features. These live OUTSIDE
# FEATURE_COLS (the moneyline's list is untouched until its own ablation
# says otherwise); the run engine appends whichever are present in the frame.
RUN_LEVEL_ENV_FEATURES = (
    "park_wind_factor", "air_density_level", "park_factor_slug",
    "dome_is_neutral_game",
)

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
         or  f ends with _delta_home/_delta_away (momentum form deltas)

    Everything else flows in automatically (new level/env features included
    without touching this file). Returns both lists so callers log the drops.
    """
    run_feats, dropped = [], []
    for f in feature_cols:
        if f.endswith("_diff") and f != RUN_DIFF_EXCEPTION:
            dropped.append(f)
        elif f in RUN_EXTRA_EXCLUSIONS:
            dropped.append(f)
        elif f.endswith("_delta_home") or f.endswith("_delta_away"):
            # Momentum form deltas (recent − season baseline) are matchup/form
            # signal, not scoring LEVEL — moneyline-only per the 2026-08
            # momentum feature set. Excluded here so the run engine's raw-only
            # view stays byte-identical (GOLDEN RULE: levels + environment).
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
                     dropped: Optional[list[str]] = None,
                     include_level_env: bool = True,
                     ) -> tuple[pd.DataFrame, list[str]]:
    """Materialize the side's model frame (levels + environment), preserving
    NaN (LightGBM routes it natively). Logs the derivation once per call.

    ``include_level_env=False`` excludes the standalone env-LEVEL columns
    (ablation variant A); present-in-frame level features otherwise append
    automatically, with any missing source warned loudly."""
    from training import FEATURE_COLS

    feats = list(run_features) if run_features is not None else None
    if feats is None or dropped is None:
        feats, dropped = derive_run_features(list(FEATURE_COLS))
        logger.info("Run engine: %d/%d features kept; dropped %d: %s",
                    len(feats), len(FEATURE_COLS), len(dropped), dropped)
    if include_level_env:
        present = [c for c in RUN_LEVEL_ENV_FEATURES if c in games.columns]
        missing = [c for c in RUN_LEVEL_ENV_FEATURES if c not in games.columns]
        if missing:
            logger.warning(
                "Run engine: %d/%d env-level feature columns absent from the "
                "frame (stale artifact?): %s", len(missing),
                len(RUN_LEVEL_ENV_FEATURES), missing)
        feats = feats + present
    else:
        logger.info("Run engine: env-level features EXCLUDED (ablation arm A)")
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
            include_level_env: bool = True,
            run_features: Optional[list[str]] = None,
            dropped: Optional[list[str]] = None,
            decided_snapshot: Optional[pd.DataFrame] = None,
            ) -> dict[str, Any]:
    """Walk-forward OOF for both side models on the moneyline pipeline's folds.

    ``run_features`` / ``dropped`` override the derived keep-list (ablation
    arms B/C/REF exercise the full 58-col view); None → derive_run_features.
    ``decided_snapshot``: pre-computed decided frame (from frames.get_decided_frame)
    captured ONCE after official results, before slate merge.  When provided,
    skips re-derivation from the caller's ``games`` object so the OOF folds
    are built on the SAME canonical frame training used — preventing the
    fold-signature desync when the pipeline frame is mutated by slate
    concatenation, weather, or dome refinement.
    Returns rows (one per decided game), per-side metrics, baseline metrics,
    and the dispersion probe."""
    from training import walk_forward_splits

    games = (decided_snapshot.copy() if decided_snapshot is not None
             else get_decided_frame(games))
    folds = [
        s for s in walk_forward_splits(games, retrain_cadence_days=retrain_cadence_days)
        if len(s["val_games"]) >= min_val_games
    ]
    logger.info("Run engine: %d walk-forward folds, %d decided games",
                len(folds), len(games))

    frames = {
        side: build_side_frame(games, side, run_features=run_features,
                               dropped=dropped,
                               include_level_env=include_level_env)
        for side in ("home", "away")}
    params = dict(RUN_LGBM_PARAMS)

    out_rows: list[dict] = []
    best_iters: dict[str, list[int]] = {s: [] for s in ("home", "away")}
    metrics: dict[str, dict[str, list[float]]] = {
        s: {"deviance": [], "rmse": [], "mae": []} for s in ("home", "away")}
    base_metrics: dict[str, dict[str, list[float]]] = {
        s: {"deviance": [], "rmse": [], "mae": []} for s in ("home", "away")}

    for split in folds:
        tr, va = split["train_games"], split["val_games"]
        # game_id mirrors the moneyline predictions_history convention
        # ({YYYYMMDD}_{away}@{home}) so Phase 2 can merge against it.
        _d = pd.to_datetime(va["game_date"]).dt.strftime("%Y%m%d")
        rec_base = {"game_pk": va["game_pk"].to_numpy(),
                    "game_date": pd.to_datetime(va["game_date"]).dt.strftime("%Y-%m-%d"),
                    "fold_idx": split["fold_idx"],
                    "game_id": (va["game_id"] if "game_id" in va.columns else
                                _d + "_" + va["away_team"] + "@" + va["home_team"]) if
                    ("game_id" in va.columns or {"home_team", "away_team"}.issubset(va.columns))
                    else pd.Series([""] * len(va), index=va.index)}
        for side, target in (("home", "home_score"), ("away", "away_score")):
            _, cols_all = frames[side]
            tr_frame = tr.reindex(columns=cols_all).astype(float)
            va_frame = va.reindex(columns=cols_all).astype(float)
            y_tr = tr[target].to_numpy(dtype=float)
            y_va = va[target].to_numpy(dtype=float)
            _, lam, best = _fit_side_model(params, tr_frame, y_tr, va_frame, y_va)
            best_iters[side].append(best)
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
    # Median early-stopped rounds per side → fixed round count for the final
    # all-data slate models (predict_slate_runs).
    summary["final_fit_rounds"] = {
        s: int(np.median(best_iters[s])) if best_iters[s] else MAX_ROUNDS
        for s in ("home", "away")}
    summary["final_fit_rounds_note"] = (
        "median early-stopping iteration across walk-forward folds; used as "
        "the fixed round count when refitting on ALL decided games for slate λ")
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


# Persisted Phase-1 contract. fold_idx / game_id ride along internally
# (Phase 2's prequential calibration and agreement merge need them) but stay
# out of the CSV schema.
OOF_COLUMNS = ["game_pk", "game_date", "home_expected_runs",
               "away_expected_runs", "home_score", "away_score"]


def persist_oof(oof: pd.DataFrame, target_date_str: str,
                out_dir: Optional[Path] = None) -> Path:
    """run_engine_oof_<date>.csv — Phase 2's input contract."""
    # Identity guard at the persist boundary (f33b569 discipline, extended):
    # rows without a stable game_pk (slate games whose results were filled
    # post-merge) cannot be joined to slate/history and must never ship in
    # the committed OOF artifact. Drop them LOUDLY here so every caller is
    # protected; derive_markets_v3 re-applies the same filter downstream
    # (defense in depth) before any market construction.
    if "game_pk" in oof.columns:
        _bad_identity = oof["game_pk"].isna()
        if _bad_identity.any():
            _dates = (sorted(oof.loc[_bad_identity, "game_date"]
                             .astype(str).unique().tolist())
                      if "game_date" in oof.columns else "n/a")
            logger.warning(
                "persist_oof: dropping %d identity-less OOF row(s) "
                "(null game_pk, dates %s) before writing the shipped "
                "artifact", int(_bad_identity.sum()), _dates)
            oof = oof.loc[~_bad_identity].reset_index(drop=True)
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
# Phase 2 — market derivation (NB marginals + Monte Carlo; NEVER Skellam/Poisson:
# Phase 1 measured χ²/df ≈ 2.2–2.5, and the difference of two NBs has no closed
# form — Monte Carlo is the honest route).
# ---------------------------------------------------------------------------
MARKET_SEED = 42
MC_DRAWS = 10_000
MC_DRAWS_TAIL = 50_000   # bump when tail-sensitive SE > MC_SE_TARGET
MC_SE_TARGET = 5e-3
TOTAL_LINE = 8.5          # reference over line (over = total ≥ 9)
RUN_LINE_MARGIN = 1.5     # home cover = home − away ≥ 2
ALPHA_FLOOR = 1e-6        # α below this ≈ Poisson; sample with huge n instead

# Phase 3 — full line grid so the dashboard toggle prices ANY line offline.
TOTAL_LINE_GRID = [round(6.5 + 0.5 * i, 1) for i in range(13)]   # 6.5 … 12.5
RUN_LINE_GRID = [0.5, 1.5, 2.5, 3.5]   # home-favorite margins (−0.5 … −3.5)
# Per-line run-line grid for the p_rl_* columns: the legacy half-lines PLUS
# the whole-number alternates (−1, −2, −3, −4) sportsbooks also post.
# Convention (settled, mirrors totals): home covers −L iff margin > L
# (strict); margin == L is a PUSH on whole lines only; half-lines can never
# push. On half-lines, margin > L ⇔ margin ≥ L + 0.5 for integer margins,
# so p_rl_<m>_home is byte-identical to the legacy p_home_cover_<m>
# (computed as diff ≥ ceil(m)) — backward compat is exact, no re-fit.
RUN_LINE_GRID_FULL = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # −1 … −4 by 0.5
RUN_LINE_GRID_WHOLE = [1.0, 2.0, 3.0, 4.0]
TOTAL_REF_LINES = (7.5, 8.5, 9.5)      # mandatory multi-line scoring points
RUN_REF_LINES = (1.5, 2.5)

# Phase 3 — α(λ) dispersion model
HOLDOUT_DAYS = 21        # sealed-holdout convention (matches LightGBM gate)
ALPHA_N_BINS = 7
ALPHA_MIN_BIN = 250
ALPHA_CAP = 2.0          # sane max — beyond this variance is degenerate
ALPHA_YEAR_REL_GAP = 0.25  # |Δα|/α between years above this → loud caveat

# Structural home one-run adjustment — MARGIN POST-PROCESSING ONLY, no
# sampler/λ/α(λ) change. The independent-NB marginals put ~10% mass on
# impossible ties; MLB games always resolve, and real ties-after-regulation
# resolve at margin = ±1 (walk-offs), home-weighted. So the tie mass
# P(margin=0) is resolved into the +1 /−1 bands with home share α:
#   P(+1)' = P(+1) + α·P(0),  P(−1)' = P(−1) + (1−α)·P(0),  P(0)' = 0,
# and EVERY other margin stays at its RAW full-basis value (P(≥2) and
# P(≤−2) already match reality; totals are sum-based and byte-identical).
# α fit on pooled OOF to match the +1 target
# (data_delivery/margin_adjustment_ablation_20260829.json): α = 0.744
# yields +1 = 17.40% (actual 17.40%) and all run lines calibrated; the
# seasonal fit ranges 0.657 (2026) / 0.706 (2024) / 0.853 (2025). α may
# exceed the true extra-inning win rate — it absorbs walk-off/bottom-9th
# effects, so it is a calibration parameter, not a mechanism rate.
MARGIN_PLUS1_HOME_SHARE = 0.744

MARKET_COLUMNS = ["game_pk", "game_date", "home_expected_runs",
                  "away_expected_runs", "p_over_8_5", "p_home_cover_1_5",
                  "p_home_win_derived", "home_score", "away_score",
                  "total_runs"]


def fit_alpha(y: np.ndarray, lam: np.ndarray) -> float:
    """Method-of-moments NB dispersion: var = λ̄ + α·λ̄²  ⇒
    α = max((var_obs − λ̄) / λ̄², 0)."""
    y = np.asarray(y, dtype=float)
    lam = np.asarray(lam, dtype=float)
    lam_bar = float(lam.mean())
    var_obs = float(y.var(ddof=0))
    return round(max((var_obs - lam_bar) / (lam_bar ** 2), 0.0), 4)


def _nb_size_prob(mu: np.ndarray | float,
                  alpha: float | np.ndarray) -> tuple[np.ndarray | float, np.ndarray | float]:
    """numpy's NB(n, p): mean n(1−p)/p = μ ⇒ p = n/(n+μ); variance μ + μ²/n
    with n = 1/α. α below ALPHA_FLOOR → huge n (Poisson limit). Accepts
    scalar or per-game column arrays."""
    a = np.asarray(alpha, dtype=float)
    n = np.where(a > ALPHA_FLOOR, 1.0 / np.maximum(a, ALPHA_FLOOR), 1e12)
    p = n / (n + mu)
    return n, p


def nb_pmf(k: np.ndarray, mu: float, alpha: float) -> np.ndarray:
    """NB(μ, α) pmf at integer k ≥ 0 (scipy-free, log-space stable)."""
    from math import lgamma, exp, log
    n = 1.0 / alpha if alpha > ALPHA_FLOOR else 1e12
    out = np.empty(len(k), dtype=float)
    for i, ki in enumerate(k):
        out[i] = exp(lgamma(ki + n) - lgamma(n) - lgamma(ki + 1)
                     + n * log(n / (n + mu)) + ki * log(mu / (n + mu)))
    return out


def fit_check_table(actual: np.ndarray, lam: np.ndarray, alpha: float,
                    kmax: int = 12) -> list[dict]:
    """NB(λ̄, α) implied score distribution vs observed OOF distribution."""
    actual = np.asarray(actual, dtype=int)
    lam_bar = float(np.asarray(lam, dtype=float).mean())
    ks = np.arange(0, kmax + 1)
    modeled = nb_pmf(ks, lam_bar, alpha)
    observed = np.array([(actual == k).mean() for k in ks])
    tail_ge10_m = float(modeled[ks >= 10].sum())
    tail_ge10_o = float((actual >= 10).mean())
    le1_m = float(modeled[ks <= 1].sum())
    le1_o = float((actual <= 1).mean())
    rows = [{"k": int(k), "modeled_p": round(float(m), 4),
             "observed_p": round(float(o), 4)}
            for k, m, o in zip(ks, modeled, observed)]
    rows.append({"k": "≥10", "modeled_p": round(tail_ge10_m, 4),
                 "observed_p": round(tail_ge10_o, 4)})
    rows.append({"k": "≤1", "modeled_p": round(le1_m, 4),
                 "observed_p": round(le1_o, 4)})
    return rows


def _as_alpha_col(alpha: float | np.ndarray, n_games: int) -> np.ndarray:
    """Accept a scalar or per-game α vector; return an (n_games, 1) column."""
    arr = np.asarray(alpha, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n_games, float(arr))
    return np.maximum(arr, ALPHA_FLOOR)[:, None]


# ────────────────────────────────────────────────────────────────────────
# PINNED VERDICT 2026-08-27 — discriminative totals/run-line blending member
# (run_total_blend_ablation.py, data_delivery/total_blend_ablation_20260827.json):
# DON'T ADOPT. A gradient-boosted member trained on the run engine's own
# 29-feature view (levels + environment; the same keep-list derive_run_features
# produces for this module) with heads E[total runs] / P(over 8.5) / P(home
# cover -1.5), blended with the NB sampler's per-line probabilities (fixed
# 50/50 average + prequential L2 logistic stackers) through the shared 45-fold
# geometry + sealed 284 holdout. The GBM member improves POOLED OOF on both
# surfaces (totals ll 0.6843 vs NB 0.6882, run-line 0.6428 vs 0.6488), and the
# stack improves SEALED totals (ll 0.6833 vs 0.6866, ECE-cal 0.0410 vs 0.0779)
# — but every blend variant DEGRADES sealed run-line logloss (best +0.0056 vs
# NB 0.6641), so no variant clears BOTH surfaces' sealed window. The same
# pooled-gain/sealed-loss inversion seen in every prior blend-level gate
# (stack, calibration flip, margin, edge correction). NB-only stays the
# production pricing path; the harness remains for re-testing on a later frame.
# ────────────────────────────────────────────────────────────────────────
def derive_markets_mc(lam_home: np.ndarray, lam_away: np.ndarray,
                      alpha_home: float | np.ndarray,
                      alpha_away: float | np.ndarray,
                      n_draws: int = MC_DRAWS,
                      seed: int = MARKET_SEED) -> dict[str, np.ndarray]:
    """Monte Carlo per game from INDEPENDENT NB(λ, α) marginals.

    α may be scalar (Phase 2) or per-game vectors from the α(λ) curve
    (Phase 3). Chunked over games to cap memory; identical seed ⇒ identical
    output. Prices the FULL line grid (totals 6.5–12.5 half-steps; run lines
    −0.5…−3.5) from the SAME draw matrix, so grid columns are mutually
    consistent and p_over_8_5 equals its legacy value exactly.
    Returns per-game probabilities plus the totals-market MC standard error."""
    rng = np.random.default_rng(seed)
    n_games = len(lam_home)
    p_over = np.empty(n_games)
    p_cover = np.empty(n_games)
    p_win = np.empty(n_games)
    n_lines = len(TOTAL_LINE_GRID)
    n_margins = len(RUN_LINE_GRID)
    grid_over = np.empty((n_games, n_lines))
    grid_cover = np.empty((n_games, n_margins))
    grid_push = np.empty((n_games, n_lines))
    n_rl = len(RUN_LINE_GRID_FULL)
    grid_rl_home = np.empty((n_games, n_rl))   # P(home covers −L) = P(diff > L)
    grid_rl_push = np.empty((n_games, n_rl))   # P(push) = P(diff == L)
    grid_rl_away = np.empty((n_games, n_rl))   # P(away +L) = P(diff < L)
    chunk = max(1, min(n_games, 2_000_000 // max(n_draws, 1)))
    for start in range(0, n_games, chunk):
        end = min(start + chunk, n_games)
        mu_h = np.maximum(np.asarray(lam_home[start:end], float), 1e-6)[:, None]
        mu_a = np.maximum(np.asarray(lam_away[start:end], float), 1e-6)[:, None]
        nh, ph_ = _nb_size_prob(mu_h, _as_alpha_col(alpha_home, n_games)[start:end])
        na, pa = _nb_size_prob(mu_a, _as_alpha_col(alpha_away, n_games)[start:end])
        h = rng.negative_binomial(nh, ph_,
                                  size=(end - start, n_draws)).astype(np.int32)
        a = rng.negative_binomial(na, pa,
                                  size=(end - start, n_draws)).astype(np.int32)
        total = h + a
        diff = h - a
        # Structural home one-run adjustment (replaces the proportional no-tie
        # renormalization of 2531462). Instead of rescaling EVERY nonzero
        # margin by 1/(1−P0) — which inflated P(≥2) 0.3583→0.3989 and caused
        # the +4.1pt line −1 over — the impossible tie mass P(0) is resolved
        # into ±1 home-weighted: P(+1)' = P(+1)+α·P(0), P(−1)' = P(−1)+
        # (1−α)·P(0), P(0)'=0, and every other margin stays at its RAW
        # full-basis value. P(≥2) and P(≤−2) already match actual rates, so
        # they are untouched; totals stay byte-identical.
        p0 = (diff == 0).mean(axis=1)
        p1 = (diff == 1).mean(axis=1)
        pn1 = (diff == -1).mean(axis=1)
        ge2 = (diff >= 2).mean(axis=1)
        ge3 = (diff >= 3).mean(axis=1)
        ge4 = (diff >= 4).mean(axis=1)
        ge5 = (diff >= 5).mean(axis=1)
        eq2 = (diff == 2).mean(axis=1)
        eq3 = (diff == 3).mean(axis=1)
        eq4 = (diff == 4).mean(axis=1)
        push1 = p1 + MARGIN_PLUS1_HOME_SHARE * p0   # resolved +1 band
        # Strict over: total must EXCEED the line (total > line). Using
        # TOTAL_LINE + 0.5 matches the monitor scorer's definition and
        # fixes the push-inclusive bug where int(-(-line//1)) produced
        # the same threshold for whole lines (9.0) and half-lines (8.5).
        p_over[start:end] = (total >= TOTAL_LINE + 0.5).mean(axis=1)
        p_cover[start:end] = ge2   # legacy home cover −1.5 = P'(margin>=2)
        # p_home_win_derived = P'(margin>0) = P(≥2)+P(+1)' (ties resolve home
        # with share α). −0.5 cover IS home win, so it stays equal to
        # p_cover_grid[:, −0.5] (the identity the game-structure sampler's
        # current-arm test pins). PURELY DIAGNOSTIC (see TRACED note below).
        p_win[start:end] = ge2 + push1
    # ------------------------------------------------------------------
    # TRACED 2026-08-27: p_home_win_derived is PURELY DIAGNOSTIC — it does
    # NOT feed production pricing. Consumers: the run-engine monitor's
    # derived-ML card + nb_diagnostic, agreement_vs_moneyline/slate stats
    # (p_run), the persisted markets CSV columns (p_home_win_derived /
    # p_away_win_derived), and the agreement_conflict flag. NO EV or
    # market-table math consumes it: Today's Games ML pick/edge uses the
    # ENSEMBLE home_win_prob_model (training.py edge = model_prob -
    # fair_market_prob), and the O/U + run-line prices use the grid/cover
    # columns only. agreement_conflict is written to the artifact with no
    # downstream consumer altering pricing. Therefore the NB home-edge
    # underweighting (mean P(home) 0.4684 vs actual 0.5354; alpha_away >
    # alpha_home with only a +0.12 lambda edge) is a DIAGNOSTIC-QUALITY
    # issue only — no leakage-free ablation needed; finding recorded and
    # closed. If p_home_win_derived ever feeds pricing, reopen this.
    # ------------------------------------------------------------------
        for j, line in enumerate(TOTAL_LINE_GRID):
            # Strict over (total > line) — matches the monitor scorer's
            # total >= line + 0.5.  The old formula int(-(-line//1))
            # produced threshold = line for whole numbers (e.g. 9.0 → 9),
            # making P(over) push-inclusive (~50% at line 9 instead of
            # ~40%).  For half-lines (8.5) both formulas agree.
            grid_over[start:end, j] = (total >= line + 0.5).mean(axis=1)
        for j, m in enumerate(RUN_LINE_GRID):
            # Legacy margins: P'(margin >= ceil(m)). −0.5 cover IS home win
            # (ge2 + resolved +1); 1.5/2.5/3.5 are P'(>=2/3/4) = raw ge2/ge3/
            # ge4 — unaffected by the tie resolve.
            if m == 0.5:
                grid_cover[start:end, j] = ge2 + push1
            elif m == 1.5:
                grid_cover[start:end, j] = ge2
            elif m == 2.5:
                grid_cover[start:end, j] = ge3
            else:
                grid_cover[start:end, j] = ge4
        for j, line in enumerate(TOTAL_LINE_GRID):
            grid_push[start:end, j] = (total == line).mean(axis=1)
        for j, m in enumerate(RUN_LINE_GRID_FULL):
            # Strict cover: margin > L. Under the structural fix only the +1
            # band changes: P'(>1)=P'(>1.5)=ge2, P'(>2)=P'(>2.5)=ge3,
            # P'(>3)=P'(>3.5)=ge4, P'(>4)=ge5. Push is the resolved +1 band at
            # m=1.0 only (half-lines can never push; m=2/3/4 push = raw
            # P(margin==m)); away is the residual, so home+push+away sums to
            # 1.0 exactly on every line.
            if m <= 1.5:
                grid_rl_home[start:end, j] = ge2
                grid_rl_push[start:end, j] = push1 if m == 1.0 else 0.0
            elif m <= 2.5:
                grid_rl_home[start:end, j] = ge3
                grid_rl_push[start:end, j] = eq2 if m == 2.0 else 0.0
            elif m <= 3.5:
                grid_rl_home[start:end, j] = ge4
                grid_rl_push[start:end, j] = eq3 if m == 3.0 else 0.0
            else:
                grid_rl_home[start:end, j] = ge5
                grid_rl_push[start:end, j] = eq4
            grid_rl_away[start:end, j] = (1.0 - grid_rl_home[start:end, j]
                                          - grid_rl_push[start:end, j])
    mc_se = np.sqrt(p_over * (1 - p_over) / n_draws)
    return {"p_over_8_5": p_over, "p_home_cover_1_5": p_cover,
            "p_home_win_derived": p_win, "mc_se_totals": mc_se,
            "p_over_grid": grid_over, "p_cover_grid": grid_cover,
            "p_push_grid": grid_push,
            "p_rl_home_grid": grid_rl_home, "p_rl_push_grid": grid_rl_push,
            "p_rl_away_grid": grid_rl_away}


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(((np.asarray(p) - np.asarray(y)) ** 2).mean())


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width binned expected calibration error."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, edges[1:-1], right=False)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return round(float(ece), 5)


def prequential_calibrate(y: np.ndarray, p: np.ndarray,
                          fold_idx: np.ndarray) -> np.ndarray:
    """Per-fold Platt maps fitted on PRIOR folds only (post-F2 discipline).
    Folds before MIN_OOF_FOR_FIT games accumulate keep raw probabilities.
    Never composes the global map on top of fold maps."""
    from calibration import apply_platt, fit_platt

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    folds = np.unique(fold_idx)
    history_y, history_p = [], []
    out = p.copy()
    for f in folds:
        m = fold_idx == f
        cal = fit_platt(np.concatenate(history_y) if history_y else [],
                        np.concatenate(history_p) if history_p else [])
        if cal is not None:
            out[m] = apply_platt(p[m], cal)
        history_y.append(y[m])
        history_p.append(p[m])
    return np.clip(out, 1e-6, 1 - 1e-6)


def derive_markets(oof: pd.DataFrame,
                   moneyline_probs: Optional[pd.DataFrame] = None,
                   n_draws: int = MC_DRAWS,
                   seed: int = MARKET_SEED) -> dict[str, Any]:
    """Full Phase-2 layer: fit α per side, NB fit-check, MC markets, honest
    pooled scoring vs base-rate baselines (raw + prequential calibrated),
    agreement groundwork vs the moneyline ensemble."""
    lam_h = oof["home_expected_runs"].to_numpy(float)
    lam_a = oof["away_expected_runs"].to_numpy(float)
    hs = oof["home_score"].to_numpy(float)
    as_ = oof["away_score"].to_numpy(float)

    alpha_home = fit_alpha(hs, lam_h)
    alpha_away = fit_alpha(as_, lam_a)
    logger.warning(
        "Run engine Phase 2: fitted α_home=%.4f α_away=%.4f "
        "(Poisson rejected by Phase-1 dispersion)", alpha_home, alpha_away)

    summary: dict[str, Any] = {
        "alpha_home": alpha_home, "alpha_away": alpha_away,
        "n_draws": n_draws, "seed": seed,
        "total_line": TOTAL_LINE, "run_line_margin": RUN_LINE_MARGIN,
        "fit_check": {
            "home": fit_check_table(hs, lam_h, alpha_home),
            "away": fit_check_table(as_, lam_a, alpha_away),
        },
    }

    mc = derive_markets_mc(lam_h, lam_a, alpha_home, alpha_away,
                           n_draws=n_draws, seed=seed)
    total_runs = hs + as_
    y_over = (total_runs >= 9).astype(float)
    y_cover = ((hs - as_) >= 2).astype(float)
    y_win = (hs > as_).astype(float)

    # MC standard error on example games (metadata honesty).
    se = mc["mc_se_totals"]
    summary["mc_se_examples"] = {
        "first_game": round(float(se[0]), 6),
        "median": round(float(np.median(se)), 6),
        "max": round(float(se.max()), 6),
    }

    def score_market(name: str, p: np.ndarray, y: np.ndarray,
                     fold_idx: np.ndarray) -> dict:
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        base_p = float(np.asarray(y, float).mean())
        p_cal = prequential_calibrate(y, p, fold_idx)
        row = {
            "engine_logloss": round(log_loss(y, p), 5),
            "engine_brier": round(brier_score(y, p), 5),
            "engine_ece_raw": ece_score(y, p),
            "engine_ece_calibrated": ece_score(y, p_cal),
            "engine_logloss_calibrated": round(log_loss(y, p_cal), 5),
            "baseline_rate": round(base_p, 4),
            "baseline_logloss": round(log_loss(y, np.full(len(y), base_p)), 5),
            "baseline_brier": round(brier_score(y, np.full(len(y), base_p)), 5),
        }
        row["beats_baseline_logloss"] = bool(row["engine_logloss"] < row["baseline_logloss"])
        summary[f"market_{name}"] = row
        return row

    fold_idx = oof["fold_idx"].to_numpy()
    score_market("over_8_5", mc["p_over_8_5"], y_over, fold_idx)
    score_market("home_cover_1_5", mc["p_home_cover_1_5"], y_cover, fold_idx)
    score_market("derived_moneyline", mc["p_home_win_derived"], y_win, fold_idx)

    # Agreement groundwork vs the moneyline ensemble (if probabilities given).
    if moneyline_probs is not None and len(moneyline_probs):
        merged = pd.DataFrame({
            "game_id": oof["game_id"].to_numpy(),
            "p_run": mc["p_home_win_derived"],
        }).merge(
            moneyline_probs[["game_id", "home_win_prob_model"]],
            on="game_id", how="inner",
        )
        if len(merged) > 30:
            diff = (merged["p_run"] - merged["home_win_prob_model"]).abs()
            summary["agreement_vs_moneyline"] = {
                "n_merged": int(len(merged)),
                "correlation": round(float(merged["p_run"].corr(
                    merged["home_win_prob_model"])), 4),
                "mean_abs_diff": round(float(diff.mean()), 4),
                "share_gt_0_08": round(float((diff > 0.08).mean()), 4),
                "share_gt_0_10": round(float((diff > 0.10).mean()), 4),
            }

    markets = oof[["game_pk", "game_date", "home_expected_runs",
                   "away_expected_runs"]].copy()
    markets["p_over_8_5"] = np.round(mc["p_over_8_5"], 5)
    markets["p_home_cover_1_5"] = np.round(mc["p_home_cover_1_5"], 5)
    markets["p_home_win_derived"] = np.round(mc["p_home_win_derived"], 5)
    markets["home_score"] = hs.astype(int)
    markets["away_score"] = as_.astype(int)
    markets["total_runs"] = total_runs.astype(int)
    return {"markets": markets, "summary": summary}


# Persisted Phase-3 contract. Grid columns are named p_over_<line> /
# p_push_<line> / p_under_<line> / p_home_cover_<margin> / p_rl_<m>_<side>
# so the dashboard toggle reads exact per-line values with zero frontend
# math. p_push_<line> is P(total == line) from the MC draws (exactly 0.0 on
# half-lines — integer totals never equal x.5); p_rl_<m>_{home,push,away} is
# the 3-way margin split (margin > L cover / == L push / < L away) under the
# post-fix margin distribution. ml_win_prob may be null when the moneyline
# history lacks the game (conflicts then uncomputable, loudly).
def rl_col(m: float, side: str) -> str:
    """Injective p_rl column name for a run-line margin + side.

    The suffix is built from the RAW margin formatted one-decimal (the
    totals-grid lesson — never a pre-computed dict key): f'{1.0:.1f}' ->
    '1.0' -> '1_0' vs f'{1.5:.1f}' -> '1_5', so whole 1.0 vs half 1.5
    stay distinct: p_rl_1_0_home and p_rl_1_5_home are different columns.
    """
    return f"p_rl_{f'{m:.1f}'.replace('.', '_')}_{side}"


MARKET_COLUMNS_V3 = (
    ["game_pk", "game_date", "kind", "home_expected_runs", "away_expected_runs",
     "alpha_home", "alpha_away"]
    + [f"p_over_{str(l).replace('.', '_')}" for l in TOTAL_LINE_GRID]
    + [f"p_push_{str(l).replace('.', '_')}" for l in TOTAL_LINE_GRID]
    + [f"p_under_{str(l).replace('.', '_')}" for l in TOTAL_LINE_GRID]
    + [f"p_home_cover_{str(m).replace('.', '_')}" for m in RUN_LINE_GRID]
    + [rl_col(m, side) for m in RUN_LINE_GRID_FULL
       for side in ("home", "push", "away")]
    + ["p_home_win_derived", "p_away_win_derived",
       "home_score", "away_score", "total_runs",
       "ml_win_prob", "agreement_conflict"]
)
NULLABLE_MARKET_COLUMNS = {"ml_win_prob"}


def persist_markets(markets: pd.DataFrame, target_date_str: str,
                    summary: dict,
                    out_dir: Optional[Path] = None) -> Path:
    out_path = (out_dir or DATA_DELIVERY_DIR) / f"run_engine_markets_{target_date_str}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing = [c for c in MARKET_COLUMNS_V3 if c not in markets.columns]
    if missing:
        raise ValueError(f"markets frame missing required columns: {missing}")
    frame = markets[MARKET_COLUMNS_V3]
    # Decided-target columns must be populated on OOF rows; slate rows are
    # undecided BY DEFINITION and are exempt from exactly those three.
    target_cols = ["home_score", "away_score", "total_runs"]
    bad = [c for c in MARKET_COLUMNS_V3
           if c not in NULLABLE_MARKET_COLUMNS
           and (frame[c].isna().any() if c not in target_cols
                else frame.loc[frame["kind"] == "oof", c].isna().any())]
    if bad:
        raise ValueError(f"markets frame contains NaNs in {bad} — refusing to persist")
    tmp = out_path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(out_path)
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path.with_suffix(".json.tmp"), "w") as f:
        json.dump(summary, f, indent=2)
    meta_path.with_suffix(".json.tmp").replace(meta_path)
    logger.info("Run engine markets: %d rows -> %s (+ meta json)", len(frame), out_path.name)
    return out_path


# ---------------------------------------------------------------------------
# Phase 3 — α(λ) dispersion model. Phase 2's single global α underestimated
# the blowout tail (P(X≥10): 0.052 vs 0.069 home, 0.053 vs 0.084 away)
# because dispersion is heteroskedastic in λ. Here: binned method-of-moments
# → candidate curves (piecewise / linear / power) selected by OUT-OF-BAG tail
# validation, monotone + non-negative + capped. Never a 3rd-order polynomial:
# the ≥10-run tail is only ~250–350 games/side.
# ---------------------------------------------------------------------------
def alpha_bins(y: np.ndarray, lam: np.ndarray,
               n_bins: int = ALPHA_N_BINS,
               min_count: int = ALPHA_MIN_BIN) -> list[dict]:
    """Binned method-of-moments points: quantile bins on λ, underfilled bins
    merged into their nearest neighbor until every bin holds ≥ min_count
    games. Per bin: α = max(0, (Var(y) − mean(λ)) / mean(λ)²)."""
    y = np.asarray(y, float)
    lam = np.asarray(lam, float)
    edges = np.unique(np.quantile(lam, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([lam.min() - 1e-9, lam.max() + 1e-9])
    idx = np.clip(np.digitize(lam, edges[1:-1], right=False), 0, len(edges) - 2)
    groups = [np.where(idx == b)[0] for b in range(len(edges) - 1)]
    # Merge any bin below min_count into its smaller neighbor (loop: merges
    # can cascade).
    while True:
        sizes = [len(g) for g in groups]
        if len(groups) <= 1 or min(sizes) >= min_count:
            break
        i = int(np.argmin(sizes))
        j = i - 1 if i == 0 else (
            i + 1 if i == len(groups) - 1
            else (i - 1 if sizes[i - 1] <= sizes[i + 1] else i + 1))
        lo, hi = min(i, j), max(i, j)
        groups[lo] = np.concatenate([groups[lo], groups[hi]])
        del groups[hi]
    bins = []
    for g in groups:
        if not len(g):
            continue
        mu, var = float(lam[g].mean()), float(y[g].var(ddof=0))
        bins.append({
            "count": int(len(g)),
            "mean_lam": round(mu, 4),
            "alpha": round(max((var - mu) / (mu ** 2), 0.0), 4),
        })
    return sorted(bins, key=lambda b: b["mean_lam"])


def _bin_direction(lams: list[float], alphas: list[float]) -> int:
    """+1 when dispersion rises with λ, −1 when it falls. Data decides per
    side — Phase 2 residuals showed home RISING and away FALLING."""
    if len(lams) < 2 or np.std(alphas) == 0 or np.std(lams) == 0:
        return +1
    corr = np.corrcoef(lams, alphas)[0, 1]
    return -1 if corr < 0 else +1


def _fit_curve_piecewise(bins: list[dict]) -> dict:
    """Weighted isotonic fit through the bin points: monotone in the
    data-chosen direction, count-weighted so big bins dominate, clipped to
    [0, CAP]. Unlike a raw cummax it does NOT propagate one high bin into
    every later point."""
    from sklearn.isotonic import IsotonicRegression

    xs = np.array([b["mean_lam"] for b in bins])
    ys = np.array([max(b["alpha"], 0.0) for b in bins])
    w = np.array([b["count"] for b in bins], dtype=float)
    d = _bin_direction(xs.tolist(), ys.tolist())
    iso = IsotonicRegression(increasing=bool(d > 0), out_of_bounds="clip")
    iso.fit(xs, ys, sample_weight=w)
    grid = np.linspace(float(xs.min()), float(xs.max()), 40)
    vals = np.clip(iso.predict(grid), 0.0, ALPHA_CAP)
    return {"form": "piecewise", "lam": [round(float(v), 5) for v in grid],
            "alpha": [round(float(v), 5) for v in vals],
            "direction": "rising" if d > 0 else "falling"}


def _fit_curve_linear(bins: list[dict]) -> dict:
    xs = np.array([b["mean_lam"] for b in bins])
    ys = np.array([max(b["alpha"], 0.0) for b in bins])
    if len(xs) < 2:   # degenerate: single bin → constant level, no polyfit
        return {"form": "linear", "a": float(ys.mean()), "b": 0.0}
    b_, a_ = np.polyfit(xs, ys, 1)
    # Either slope sign is fine (both monotone); alpha_of clips at [0, CAP].
    return {"form": "linear", "a": float(a_), "b": float(b_)}


def _fit_curve_power(bins: list[dict]) -> dict:
    pos = [b for b in bins if b["alpha"] > 0]
    if len(pos) < 2:
        return _fit_curve_linear(bins)
    xs = np.log(np.array([b["mean_lam"] for b in pos]))
    ys = np.log(np.array([b["alpha"] for b in pos]))
    if len(xs) < 2:
        return _fit_curve_linear(bins)
    c, log_a = np.polyfit(xs, ys, 1)
    return {"form": "power", "a": float(np.exp(log_a)), "c": float(c)}


def alpha_of(lam: np.ndarray, curve: dict) -> np.ndarray:
    """Evaluate the fitted α(λ) — always in [0, ALPHA_CAP]."""
    lam = np.asarray(lam, float)
    form = curve["form"]
    if form == "piecewise":
        out = np.interp(lam, curve["lam"], curve["alpha"])
    elif form == "linear":
        out = curve["a"] + curve["b"] * lam
    else:
        out = curve["a"] * np.power(np.maximum(lam, 1e-9), curve["c"])
    return np.clip(out, 0.0, ALPHA_CAP)


def nb_pmf_matrix(ks: np.ndarray, mu_col: np.ndarray,
                  alpha_col: np.ndarray) -> np.ndarray:
    """Vectorized NB pmf: (n_games, len(ks)). ks ints ≥0; columns (n,1)."""
    from scipy.special import gammaln
    ks = np.asarray(ks, dtype=float)[None, :]
    n_size = 1.0 / np.maximum(alpha_col, ALPHA_FLOOR)
    p = n_size / (n_size + mu_col)
    logpmf = (gammaln(ks + n_size) - gammaln(n_size) - gammaln(ks + 1.0)
              + n_size * np.log(p) + ks * np.log1p(-p))
    return np.exp(logpmf)


def eval_alpha_fit(y: np.ndarray, lam: np.ndarray, alpha: np.ndarray,
                   tail_k: int = 10, kmax: int = 80) -> dict:
    """Validation metrics for an α vector on held-out games: absolute gap in
    P(X≥tail_k) and mean NB log-likelihood (higher is better)."""
    y = np.asarray(y, float)
    mu_col = np.maximum(np.asarray(lam, float), 1e-6)[:, None]
    a_col = np.maximum(np.asarray(alpha, float), ALPHA_FLOOR)[:, None]
    M = nb_pmf_matrix(np.arange(kmax + 1), mu_col, a_col)
    modeled_tail = float(M[:, tail_k:].sum(axis=1).mean())
    obs_tail = float((y >= tail_k).mean())
    loglik = float(np.log(np.maximum(M[np.arange(len(y)), y.astype(int)], 1e-12)).mean())
    return {"tail_gap": round(abs(modeled_tail - obs_tail), 5),
            "modeled_tail": round(modeled_tail, 5),
            "observed_tail": round(obs_tail, 5),
            "loglik": round(loglik, 5)}


def select_alpha_curve(y: np.ndarray, lam: np.ndarray,
                       seed: int = RANDOM_SEED) -> tuple[dict, dict]:
    """Out-of-bag selection among piecewise/linear/power forms.

    Two-fold cross-fit (fit half A → score half B, swap); primary metric =
    |P(X≥10) modeled − observed| on the held-out half, tie-break = mean NB
    log-likelihood. The chosen form is then REFIT on all rows passed here by
    the caller (pre-holdout only). Returns (curve, diagnostics)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    halves = [perm[:len(perm) // 2], perm[len(perm) // 2:]]
    fitters = {"piecewise": _fit_curve_piecewise,
               "linear": _fit_curve_linear,
               "power": _fit_curve_power}
    diag: dict[str, dict] = {}
    for name, fit_fn in fitters.items():
        scores = []
        for fit_idx, eval_idx in ((halves[0], halves[1]), (halves[1], halves[0])):
            curve = fit_fn(alpha_bins(y[fit_idx], lam[fit_idx]))
            ev = eval_alpha_fit(y[eval_idx], lam[eval_idx],
                                alpha_of(lam[eval_idx], curve))
            scores.append(ev)
        diag[name] = {
            "tail_gap_avg": round((scores[0]["tail_gap"] + scores[1]["tail_gap"]) / 2, 5),
            "loglik_avg": round((scores[0]["loglik"] + scores[1]["loglik"]) / 2, 5),
        }
    best = min(fitters, key=lambda n: (diag[n]["tail_gap_avg"], -diag[n]["loglik_avg"]))
    bins = alpha_bins(y, lam)
    if len(bins) < 2:
        # Everything merged into one bin (small samples): every parametric
        # form degenerates to a constant level — ship piecewise directly.
        best = "piecewise"
    curve = fitters[best](bins)
    diag["chosen"] = best
    diag["bins"] = bins
    return curve, diag


def year_effect_check(oof: pd.DataFrame, side: str,
                      rel_gap_flag: float = ALPHA_YEAR_REL_GAP) -> dict:
    """2025-vs-2026 dispersion at the same λ: per-year global MoM α plus a
    common-binned comparison. Reports a relative gap and pools with a loud
    caveat when |Δα|/α exceeds the flag threshold (per-year curves are too
    noisy at ~4k games/side to justify splitting)."""
    years = pd.to_datetime(oof["game_date"]).dt.year
    out: dict[str, Any] = {"rows": []}
    alphas = {}
    for yr in sorted(years.unique()):
        sub = oof[years == yr]
        y = sub[f"{side}_score"].to_numpy(float)
        lam = sub[f"{side}_expected_runs"].to_numpy(float)
        mu = float(lam.mean())
        a = round(max((float(y.var(ddof=0)) - mu) / mu ** 2, 0.0), 4)
        alphas[int(yr)] = a
        out["rows"].append({"year": int(yr), "n_games": int(len(sub)),
                            "alpha": a})
    if len(alphas) >= 2:
        vals = list(alphas.values())
        base = min(vals) or 1e-9
        out["rel_gap"] = round(abs(vals[-1] - vals[0]) / base, 4)
        out["pooled_with_caveat"] = bool(out["rel_gap"] > rel_gap_flag)
        if out["pooled_with_caveat"]:
            logger.warning(
                "Run engine %s side: year effect in dispersion — α by year %s "
                "(rel gap %.0f%% > %.0f%%). Pooling with caveat: per-year "
                "curves are too noisy at this sample size.",
                side, alphas, 100 * out["rel_gap"], 100 * rel_gap_flag)
    return out


def fit_check_table_curve(actual: np.ndarray, lam: np.ndarray,
                          alpha: np.ndarray, kmax: int = 12) -> list[dict]:
    """Fit check under the α(λ) model: the modeled marginal score
    distribution is the game-average of NB(k; λ_i, α(λ_i)) pmfs."""
    actual = np.asarray(actual, int)
    mu_col = np.maximum(np.asarray(lam, float), 1e-6)[:, None]
    a_col = np.maximum(np.asarray(alpha, float), ALPHA_FLOOR)[:, None]
    K = np.arange(0, kmax + 1)
    modeled_k = nb_pmf_matrix(K, mu_col, a_col).mean(axis=0)
    big = nb_pmf_matrix(np.arange(0, 61), mu_col, a_col)
    surv = big[:, ::-1].cumsum(axis=1)[:, ::-1]   # P(X ≥ k) per game
    rows = [{"k": int(k), "modeled_p": round(float(m), 4),
             "observed_p": round(float((actual == k).mean()), 4)}
            for k, m in zip(K, modeled_k)]
    for t in (10, 11, 12):
        rows.append({"k": f"≥{t}", "modeled_p": round(float(surv[:, t].mean()), 4),
                     "observed_p": round(float((actual >= t).mean()), 4)})
    rows.append({"k": "≤1", "modeled_p": round(float(modeled_k[:2].sum()), 4),
                 "observed_p": round(float((actual <= 1).mean()), 4)})
    return rows


def agreement_stats(p_run: np.ndarray, p_ml: np.ndarray,
                    delta: float | None = None) -> dict:
    """Moneyline-vs-derived divergence stats at the configured delta plus the
    standard secondary 0.08/0.10 read."""
    from config import AGREEMENT_FILTER_DELTA
    delta = AGREEMENT_FILTER_DELTA if delta is None else delta
    diff = np.abs(np.asarray(p_run, float) - np.asarray(p_ml, float))
    stats = {
        "delta_primary": delta,
        "n": int(len(diff)),
        "mean_abs_diff": round(float(diff.mean()), 4),
        "share_gt_primary": round(float((diff > delta).mean()), 4),
        "share_gt_0_08": round(float((diff > 0.08).mean()), 4),
        "share_gt_0_10": round(float((diff > 0.10).mean()), 4),
    }
    stats["n_flagged_primary"] = int((diff > delta).sum())
    stats["n_flagged_0_08"] = int((diff > 0.08).sum())
    stats["n_flagged_0_10"] = int((diff > 0.10).sum())
    return stats


def derive_markets_v3(oof: pd.DataFrame,
                      moneyline_probs: Optional[pd.DataFrame] = None,
                      n_draws: int = MC_DRAWS,
                      seed: int = MARKET_SEED,
                      holdout_days: int = HOLDOUT_DAYS,
                     ) -> dict[str, Any]:
    """Phase 3: α(λ) dispersion curves + full-line-grid MC markets + sealed
    holdout gate + agreement groundwork.

    The last `holdout_days` of OOF games are SEALED: no α fitting, binning,
    or form selection ever sees them. Everything is scored pooled OOF and
    again holdout-only, each against constant-base-rate baselines."""
    # OOF rows without a stable identity cannot be joined to slate/history
    # artifacts and must never enter the persisted markets table. Keep this
    # filter at the market-construction boundary so every caller is safe.
    _identity = oof.get("game_pk")
    if _identity is not None:
        _bad_identity = _identity.isna()
        if _bad_identity.any():
            logger.warning(
                "derive_markets_v3: dropping %d identity-less OOF row(s) "
                "before market construction; game_pk=%s game_id=%s",
                int(_bad_identity.sum()),
                oof.loc[_bad_identity, "game_pk"].tolist(),
                oof.loc[_bad_identity, "game_id"].tolist()
                if "game_id" in oof.columns else "<absent>",
            )
            oof = oof.loc[~_bad_identity].reset_index(drop=True)
    dates = pd.to_datetime(oof["game_date"])
    cutoff = dates.max() - pd.Timedelta(days=holdout_days)
    pre_mask = (dates < cutoff).to_numpy()
    hs = oof["home_score"].to_numpy(float)
    as_ = oof["away_score"].to_numpy(float)
    total_runs = hs + as_
    fold_idx = oof["fold_idx"].to_numpy()

    summary: dict[str, Any] = {
        "n_draws": n_draws, "seed": seed,
        "holdout_cutoff": str(cutoff.date()),
        "n_pre": int(pre_mask.sum()), "n_holdout": int((~pre_mask).sum()),
        "line_grid": {"totals": TOTAL_LINE_GRID,
                      "run_lines": [-m for m in RUN_LINE_GRID],
                      "run_lines_full": [-m for m in RUN_LINE_GRID_FULL],
                      "run_lines_whole": [-m for m in RUN_LINE_GRID_WHOLE],
                      "total_ref_lines": list(TOTAL_REF_LINES),
                      "run_ref_lines": [-m for m in RUN_REF_LINES]},
    }

    curves, alpha_cols, single_alpha = {}, {}, {}
    phase2_fc, phase3_fc, variance_check = {}, {}, {}
    for side, target in (("home", "home_score"), ("away", "away_score")):
        y_all = oof[target].to_numpy(float)
        lam_all = oof[f"{side}_expected_runs"].to_numpy(float)
        y_pre, lam_pre = y_all[pre_mask], lam_all[pre_mask]
        curve, diag = select_alpha_curve(y_pre, lam_pre,
                                         seed=seed + (1 if side == "home" else 2))
        curves[side] = curve
        summary[f"alpha_{side}"] = {
            **curve, "selection": diag,
            "fitted_on": "pre-holdout OOF only",
            "cap": ALPHA_CAP, "min_bin_count": ALPHA_MIN_BIN,
        }
        alpha_cols[side] = alpha_of(lam_all, curve)
        single_alpha[side] = fit_alpha(y_pre, lam_pre)  # Phase-2 style, PRE-fit
        # Phase-2 comparison: SAME single-α fit applied to the full OOF set.
        phase2_fc[side] = fit_check_table(y_all, lam_all, single_alpha[side])
        phase3_fc[side] = fit_check_table_curve(y_all, lam_all, alpha_cols[side])
        # var = λ + α·λ² (NB size parameterization n = 1/α).
        var_implied = float((lam_all + alpha_cols[side] * lam_all ** 2).mean())
        variance_check[side] = {
            "implied_var": round(var_implied, 3),
            "observed_var": round(float(y_all.var(ddof=0)), 3),
            "phase2_implied_var": round(float(lam_pre.mean()
                                              + single_alpha[side] * lam_pre.mean() ** 2), 3),
        }
        summary[f"year_effect_{side}"] = year_effect_check(
            oof.iloc[np.where(pre_mask)[0]], side)
    summary["phase2_single_alpha"] = single_alpha
    summary["fit_check_single_alpha"] = phase2_fc
    summary["fit_check_alpha_lambda"] = phase3_fc
    summary["variance_check"] = variance_check

    mc = derive_markets_mc(oof["home_expected_runs"].to_numpy(float),
                           oof["away_expected_runs"].to_numpy(float),
                           alpha_cols["home"], alpha_cols["away"],
                           n_draws=n_draws, seed=seed)
    se = mc["mc_se_totals"]
    used_draws = n_draws
    draw_reason = "default"
    if se.max() > MC_SE_TARGET and n_draws < MC_DRAWS_TAIL:
        used_draws, draw_reason = MC_DRAWS_TAIL, (
            f"SE {se.max():.4f} > {MC_SE_TARGET} at N={n_draws} — bumped")
        mc = derive_markets_mc(oof["home_expected_runs"].to_numpy(float),
                               oof["away_expected_runs"].to_numpy(float),
                               alpha_cols["home"], alpha_cols["away"],
                               n_draws=MC_DRAWS_TAIL, seed=seed)
        se = mc["mc_se_totals"]
    summary["mc_meta"] = {
        "n_draws": used_draws, "requested_draws": n_draws,
        "reason": draw_reason,
        "mc_se_totals_max": round(float(se.max()), 6),
    }

    def line_key_total(line: float) -> str:
        return f"p_over_{str(line).replace('.', '_')}"

    def line_key_margin(m: float) -> str:
        return f"p_home_cover_{str(m).replace('.', '_')}"

    def score_at(name: str, p: np.ndarray, yv: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> None:
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        yv = np.asarray(yv, float)
        base = float(yv.mean())
        row = {
            "engine_logloss": round(log_loss(yv, p, labels=[0.0, 1.0]), 5),
            "engine_brier": round(brier_score(yv, p), 5),
            "engine_ece_raw": ece_score(yv, p),
            "baseline_rate": round(base, 4),
            "baseline_logloss": round(log_loss(
                yv, np.full(len(yv), base), labels=[0.0, 1.0]), 5),
            "baseline_brier": round(brier_score(yv, np.full(len(yv), base)), 5),
        }
        # AUC over the same OOF y/p vectors (additive). Single-class y cannot
        # be rank-scored: auc=None with a loud warning, never a crash. Some
        # sklearn builds raise ValueError, others return NaN — guard both.
        try:
            _auc = float(roc_auc_score(yv, p))
        except ValueError:
            _auc = None
        if _auc is None or not np.isfinite(_auc):
            row["auc"] = None
            logger.warning(
                "score_at(%s): single-class y (%d of %d) — auc=None",
                name, int(yv.sum()), len(yv))
        else:
            row["auc"] = round(_auc, 5)
        p_cal = prequential_calibrate(yv, p, fold_idx)
        row["engine_logloss_calibrated"] = round(
            log_loss(yv, p_cal, labels=[0.0, 1.0]), 5)
        row["engine_ece_calibrated"] = ece_score(yv, p_cal)
        # Pooled mean of the PREQUENTIALLY-CALIBRATED probability vector. This
        # is the "predicted mean" the monitor shows next to base_rate so the
        # calibration spread (mean prediction vs mean outcome) is explicit.
        # Purely additive — no scoring math changed.
        row["predicted_mean"] = round(float(np.mean(p_cal)), 4)
        row["n"] = int(len(yv))  # pooled game count scored for this line
        if mask is not None and mask.any():
            ph = p[mask]
            yh = yv[mask]
            base_h = float(yh.mean())
            h_ll = round(log_loss(yh, ph, labels=[0.0, 1.0]), 5)
            h_bll = round(log_loss(
                yh, np.full(len(yh), base_h), labels=[0.0, 1.0]), 5)
            row["holdout"] = {
                "n": int(mask.sum()),
                "engine_logloss": h_ll,
                "engine_brier": round(brier_score(yh, ph), 5),
                "engine_ece_raw": ece_score(yh, ph),
                "baseline_rate": round(base_h, 4),
                "baseline_logloss": h_bll,
                "baseline_brier": round(brier_score(yh, np.full(len(yh), base_h)), 5),
                "beats_baseline_logloss": bool(h_ll < h_bll),
            }
        row["beats_baseline_logloss"] = bool(row["engine_logloss"] < row["baseline_logloss"])
        summary[f"market_{name}"] = row

    y_over = (total_runs >= 9).astype(float)
    y_win = (hs > as_).astype(float)
    score_at("derived_moneyline", mc["p_home_win_derived"], y_win)
    score_at("derived_moneyline_holdout", mc["p_home_win_derived"], y_win,
             mask=~pre_mask)
    for line in TOTAL_REF_LINES:
        col = TOTAL_LINE_GRID.index(line)
        score_at(f"over_{str(line).replace('.', '_')}",
                 mc["p_over_grid"][:, col], (total_runs >= line + 0.5).astype(float))
        score_at(f"over_{str(line).replace('.', '_')}_holdout",
                 mc["p_over_grid"][:, col], (total_runs >= line + 0.5).astype(float),
                 mask=~pre_mask)
    for m in RUN_REF_LINES:
        col = RUN_LINE_GRID.index(m)
        score_at(f"home_cover_{str(m).replace('.', '_')}",
                 mc["p_cover_grid"][:, col], ((hs - as_) >= m + 0.5).astype(float))
        score_at(f"home_cover_{str(m).replace('.', '_')}_holdout",
                 mc["p_cover_grid"][:, col], ((hs - as_) >= m + 0.5).astype(float),
                 mask=~pre_mask)

    markets = oof[["game_pk", "game_date"]].copy()
    for tc in ("home_team", "away_team"):
        if tc in oof.columns:
            markets[tc] = oof[tc]
    markets["kind"] = "oof"
    markets["home_expected_runs"] = np.round(
        oof["home_expected_runs"].to_numpy(float), 4)
    markets["away_expected_runs"] = np.round(
        oof["away_expected_runs"].to_numpy(float), 4)
    markets["alpha_home"] = np.round(alpha_cols["home"], 4)
    markets["alpha_away"] = np.round(alpha_cols["away"], 4)
    for j, line in enumerate(TOTAL_LINE_GRID):
        key = line_key_total(line)
        markets[key] = np.round(mc["p_over_grid"][:, j], 5)
        markets[key.replace("p_over_", "p_push_")] = np.round(
            mc["p_push_grid"][:, j], 5)
        markets[key.replace("p_over_", "p_under_")] = np.round(
            1 - mc["p_over_grid"][:, j] - mc["p_push_grid"][:, j], 5)
    for j, m in enumerate(RUN_LINE_GRID):
        markets[line_key_margin(m)] = np.round(mc["p_cover_grid"][:, j], 5)
    for j, m in enumerate(RUN_LINE_GRID_FULL):
        # Per-line run-line 3-way split (home covers −L, push, away +L).
        # Sums to 1.0 exactly from the same margin draws; half-lines have
        # push = 0 so home + away = 1.0 there.
        markets[rl_col(m, "home")] = np.round(mc["p_rl_home_grid"][:, j], 5)
        markets[rl_col(m, "push")] = np.round(mc["p_rl_push_grid"][:, j], 5)
        markets[rl_col(m, "away")] = np.round(mc["p_rl_away_grid"][:, j], 5)
    markets["p_home_win_derived"] = np.round(mc["p_home_win_derived"], 5)
    markets["p_away_win_derived"] = np.round(1 - mc["p_home_win_derived"], 5)
    markets["home_score"] = hs.astype(int)
    markets["away_score"] = as_.astype(int)
    markets["total_runs"] = total_runs.astype(int)

    if moneyline_probs is not None and len(moneyline_probs):
        # predictions_history keys on game_id historically; newer frames
        # carry game_pk. Merge on whichever the artifact actually has.
        key = "game_pk" if "game_pk" in moneyline_probs.columns else "game_id"
        merged = pd.DataFrame({
            key: oof[key].to_numpy() if key in oof.columns
            else markets["game_pk"].to_numpy(),
            "p_run": mc["p_home_win_derived"],
        }).merge(moneyline_probs[[key, "home_win_prob_model"]],
                 on=key, how="inner")
        if len(merged) > 30:
            stats = agreement_stats(merged["p_run"], merged["home_win_prob_model"])
            summary["agreement_vs_moneyline"] = stats
            diff_series = ((merged["p_run"] - merged["home_win_prob_model"]).abs()
                           > stats["delta_primary"])
            conflict_map = dict(zip(merged[key], diff_series))
            prob_map = dict(zip(merged[key], merged["home_win_prob_model"]))
            join_key = (markets["game_pk"] if key == "game_pk"
                        else oof["game_id"].to_numpy())
            markets["agreement_conflict"] = [
                bool(conflict_map.get(k, False)) for k in join_key]
            markets["ml_win_prob"] = [prob_map.get(k) for k in join_key]
    if "agreement_conflict" not in markets.columns:
        logger.warning("Run engine: moneyline probabilities unavailable — "
                       "agreement conflicts NOT computed for this run")
        markets["agreement_conflict"] = False
        markets["ml_win_prob"] = np.nan

    return {"markets": markets, "summary": summary}


def load_moneyline_probs(csv_path: Path) -> Optional[pd.DataFrame]:
    """predictions_history columns needed for agreement merges (Phase 2 keys
    on game_id; Phase 3 on game_pk — keep whichever exists)."""
    if not Path(csv_path).exists():
        return None
    df = pd.read_csv(csv_path, usecols=lambda c: c in (
        "game_pk", "game_id", "home_win_prob_model"))
    has_key = "game_pk" in df.columns or "game_id" in df.columns
    return df if (has_key and "home_win_prob_model" in df.columns) else None


def _rounded_total_line(exp_home: float, exp_away: float) -> float:
    """Nearest 0.5 of λ_home + λ_away, clamped to the shipped grid.

    Mirrors the frontend history tables' line assignment
    (market_diagnostics._rounded_lines) EXACTLY so the monitor's win rates
    match the Totals & Run Lines tables — the winner-card cross-check.
    """
    x = float(exp_home) + float(exp_away)
    half = math.floor(x * 2 + 0.5) / 2.0  # round half up, positive side
    if half < TOTAL_LINE_GRID[0]:
        return TOTAL_LINE_GRID[0]
    if half > TOTAL_LINE_GRID[-1]:
        return TOTAL_LINE_GRID[-1]
    return half


def _fair_total_line(row: pd.Series) -> Optional[float]:
    """FAIR total line for one artifact row - grid argmin of
    |re-scaled P(over) - 0.5| over the shipped grid, where re-scaled
    P(over) = p_over / (p_over + p_under) conditions out the push band.

    Mirrors market_diagnostics.fair_total_line_row EXACTLY so the backend
    winner cards and the frontend Totals & Run Lines tables share ONE
    own-total-line definition (the winner-card cross-check). Ties pick the
    LOWER line; a grid-boundary argmin is taken verbatim (never fabricated);
    None when no grid Over/Under pair is present+valid - the caller falls
    back to the round-to-half projection.
    """
    best_line, best_delta = None, None
    for line in TOTAL_LINE_GRID:
        key = str(line).replace(".", "_")
        po = row.get(f"p_over_{key}")
        pu = row.get(f"p_under_{key}")
        try:
            po = None if po is None or pd.isna(po) else float(po)
            pu = None if pu is None or pd.isna(pu) else float(pu)
        except (TypeError, ValueError):
            continue
        if po is None or pu is None or (po + pu) <= 0:
            continue
        delta = abs(po / (po + pu) - 0.5)
        if best_line is None or delta < best_delta - 1e-12:
            best_line, best_delta = line, delta
    return best_line


def _safe_auc(p: np.ndarray, y: np.ndarray) -> Optional[float]:
    """roc_auc_score guarded for degenerate y (mirrors score_at's guard):
    single-class / too-short vectors return None, never a crash or NaN.
    AUC is a rank metric on the RAW probability — the prequential
    calibration step never touches it."""
    try:
        a = float(roc_auc_score(np.asarray(y, float), np.asarray(p, float)))
        return round(a, 5) if np.isfinite(a) else None
    except ValueError:
        return None


def _winner_card_stats(p: np.ndarray, y: np.ndarray,
                       fold_idx: Optional[np.ndarray],
                       hold_mask: np.ndarray,
                       pick_mask: Optional[np.ndarray] = None) -> dict:
    """Favored-side binary card stats (mirrors score_at's metric set).

    p is the favored side's probability (>= 0.5 by construction), y is 1 when
    the pick won. Prequential calibration uses the OOF fold index when
    available; without it (fixtures/tests) calibrated == raw.

    ``pick_mask`` (True = home-pick) adds the by_pick split — home-pick vs
    away-pick n / win_rate / predicted_mean — so every metric is verifiably
    on the PICKED side (never home-side unconditionally).
    """
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    base = float(y.mean())
    if fold_idx is not None and len(fold_idx) == len(y):
        p_cal = prequential_calibrate(y, p, np.asarray(fold_idx, float))
    else:
        p_cal = p.copy()
    row = {
        "n": int(len(y)),
        "actual_win_rate": round(base, 4),
        "win_rate": round(base, 4),
        "predicted_mean": round(float(np.mean(p_cal)), 4),
        "ece_raw": ece_score(y, p),
        "ece_calibrated": ece_score(y, p_cal),
        "brier": round(brier_score(y, p), 5),
        "logloss": round(log_loss(y, p), 5),
        "logloss_calibrated": round(log_loss(y, p_cal), 5),
        "beats_baseline_logloss": bool(
            log_loss(y, p) < log_loss(y, np.full(len(y), base))),
    }
    if pick_mask is not None and len(pick_mask) == len(y):
        pm = np.asarray(pick_mask, bool)
        row["by_pick"] = {}
        for side, m in (("home", pm), ("away", ~pm)):
            if not m.any():
                row["by_pick"][side] = {"n": 0, "win_rate": None,
                                        "predicted_mean": None}
                continue
            row["by_pick"][side] = {
                "n": int(m.sum()),
                "win_rate": round(float(y[m].mean()), 4),
                "predicted_mean": round(float(p_cal[m].mean()), 4),
            }
    if hold_mask is not None and hold_mask.any():
        yh = y[hold_mask]
        ph = p[hold_mask]
        base_h = float(yh.mean())
        row["holdout"] = {
            "n": int(hold_mask.sum()),
            "actual_win_rate": round(base_h, 4),
            "win_rate": round(base_h, 4),
            "predicted_mean": round(float(np.mean(p_cal[hold_mask])), 4),
            "ece_raw": ece_score(yh, ph),
            "ece_calibrated": ece_score(yh, p_cal[hold_mask]),
            "brier": round(brier_score(yh, ph), 5),
            "logloss": round(log_loss(yh, ph), 5),
            "baseline_rate": round(base_h, 4),
            "baseline_logloss": round(log_loss(yh, np.full(len(yh), base_h)), 5),
            "beats_baseline_logloss": bool(
                log_loss(yh, ph) < log_loss(yh, np.full(len(yh), base_h))),
        }
    return row


def compute_winner_cards(markets: pd.DataFrame,
                         oof: Optional[pd.DataFrame] = None,
                         holdout_days: int = HOLDOUT_DAYS) -> dict:
    """Three binary WINNER cards from the OOF markets frame.

    Each game contributes ONE favored-side (p, y) pair — the pick-framing the
    Totals & Run Lines history tables use:

      over_under  pick Over if P(over the game's ASSIGNED rounded total) > 50%
                  else Under; pushes (total == whole-number line) excluded.
      run_line    pick Home -1.5 if P(home cover -1.5) > 50% else Away +1.5;
                  half-run lines never push.
      derived_ml  pick the side with P > 50% (home if P(home win) > 50%,
                  else away) — the run line model's own NB moneyline;
                  the moneyline ensemble rides as a one-line ml_reference.

    Each card carries ``auc`` = roc_auc_score on its (picked-side
    probability, settled outcome) over pooled OOF (pushes excluded by
    construction): over_under ranks the re-scaled pick prob vs the settled
    over/under; run_line ranks the picked side's cover prob vs covered;
    derived_ml ranks the RAW p_home_win vs home/away win (the standard
    moneyline discrimination). The holdout-split AUC is nested in the card's
    ``holdout`` dict when that split has both classes. ``oof`` provides the
    per-game fold_idx for honest prequential calibration when present.
    """
    if markets is None or not len(markets):
        return {}
    df = markets[markets.get("kind") == "oof"].copy()
    if "total_runs" in df.columns:
        df = df[df["total_runs"].notna()]
    if not len(df):
        return {}
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    dates = df["game_date"]
    cutoff = dates.max() - pd.Timedelta(days=holdout_days)
    hold_mask = (dates >= cutoff).to_numpy()

    fold_map: dict = {}
    if oof is not None and "fold_idx" in oof.columns and "game_pk" in oof.columns:
        fold_map = dict(zip(
            pd.to_numeric(oof["game_pk"], errors="coerce"),
            pd.to_numeric(oof["fold_idx"], errors="coerce")))

    def _fold_idx(keys: np.ndarray) -> Optional[np.ndarray]:
        if not fold_map:
            return None
        return np.asarray([fold_map.get(k, np.nan) for k in keys], float)

    hs = df["home_score"].to_numpy(float)
    as_ = df["away_score"].to_numpy(float)
    total = df["total_runs"].to_numpy(float)
    pks = df["game_pk"].to_numpy()

    # --- over_under: per-game FAIR line (grid argmin |re-scaled P(over)
    # - 0.5|, the 50/50 anchor), 3-way push-excluded; pick on the RE-SCALED
    # 2-way probability (raw p_over under-states Over on whole-number lines
    # via the push band) - identical to the frontend history tables. ---
    ou_p, ou_y, ou_keys, ou_dates = [], [], [], []
    for i in range(len(df)):
        line = _fair_total_line(df.iloc[i])
        if line is None:
            line = _rounded_total_line(df["home_expected_runs"].iloc[i],
                                       df["away_expected_runs"].iloc[i])
        over_col = f"p_over_{str(line).replace('.', '_')}"
        under_col = f"p_under_{str(line).replace('.', '_')}"
        if over_col not in df.columns or under_col not in df.columns:
            continue
        po = df[over_col].iloc[i]
        pu = df[under_col].iloc[i]
        if pd.isna(po) or pd.isna(pu):
            continue
        denom = float(po) + float(pu)
        if denom <= 0:
            continue
        if total[i] == line:      # push (whole-number lines only)
            continue
        p = float(po) / denom
        pick_over = p >= 0.5
        went_over = total[i] > line
        ou_p.append(p if pick_over else 1.0 - p)
        ou_y.append(float(pick_over == went_over))
        ou_keys.append(pks[i])
        ou_dates.append(dates.iloc[i])
    ou = None
    if len(ou_p) >= 2:
        ou_arr_p = np.asarray(ou_p)
        ou_arr_y = np.asarray(ou_y)
        ou_hold = np.asarray(pd.to_datetime(ou_dates)) >= cutoff
        ou = _winner_card_stats(ou_arr_p, ou_arr_y,
                                _fold_idx(ou_keys), ou_hold)
        ou["auc"] = _safe_auc(ou_arr_p, ou_arr_y)
        if ou_hold.any():
            ou.setdefault("holdout", {})["auc"] = _safe_auc(
                ou_arr_p[ou_hold], ou_arr_y[ou_hold])

    # --- run_line: -1.5/+1.5, half-run lines never push ---
    if "p_home_cover_1_5" in df.columns:
        rp = df["p_home_cover_1_5"].to_numpy(float)
        ok = np.isfinite(rp)
        margin = hs - as_
        home_covers = (margin >= 2).astype(float)
        pick_home = rp >= 0.5
        hit = (pick_home.astype(float) == home_covers).astype(float)
        fav = np.where(pick_home, rp, 1.0 - rp)
        rl = _winner_card_stats(fav[ok], hit[ok], _fold_idx(pks[ok]),
                                 hold_mask[ok], pick_home[ok])
        rl["auc"] = _safe_auc(fav[ok], hit[ok])
        if hold_mask[ok].any():
            rl.setdefault("holdout", {})["auc"] = _safe_auc(
                fav[ok][hold_mask[ok]], hit[ok][hold_mask[ok]])
    else:
        rl = None

    # --- derived_ml: home vs away (the RUN LINE model's own moneyline) ---
    # SOURCE: p_home_win_derived — the NB Monte-Carlo moneyline derived from
    # the SAME expected-runs/alpha machinery that prices the run-line card.
    # It is a DISTINCT model from the binary moneyline ensemble. Post the
    # tie-mass renormalization + home one-run structural fix the derivation
    # is CALIBRATED (pooled 54.1% vs 53.9%, ECE_raw 0.004; mean P(home)
    # 0.5323 vs actual home-win rate 0.532) — the earlier "underweights the
    # home edge" finding is resolved, not masked. The moneyline ENSEMBLE
    # (ml_win_prob) is kept as a one-line ml_reference so the model
    # comparison stays visible.
    home_won = (hs > as_).astype(float)
    ml = None
    if "p_home_win_derived" in df.columns:
        mp = df["p_home_win_derived"].to_numpy(float)
        ok = np.isfinite(mp)
        pick_home = mp >= 0.5
        hit = (pick_home.astype(float) == home_won).astype(float)
        fav = np.where(pick_home, mp, 1.0 - mp)
        ml = _winner_card_stats(fav[ok], hit[ok], _fold_idx(pks[ok]),
                                hold_mask[ok], pick_home[ok])
        # AUC on the RAW p_home_win vs home/away win (the standard moneyline
        # discrimination) — not the picked-side framing (per monitor spec).
        ml["auc"] = _safe_auc(mp[ok], home_won[ok])
        if hold_mask[ok].any():
            ml.setdefault("holdout", {})["auc"] = _safe_auc(
                mp[ok][hold_mask[ok]], home_won[ok][hold_mask[ok]])
        ml["source"] = "nb_mc_p_home_win_derived"
        # nb_diagnostic: schema-stable record of the NB model finding
        # (equals the card itself now that the card IS the NB moneyline).
        ml["nb_diagnostic"] = {
            "n": ml["n"], "actual_win_rate": ml["actual_win_rate"],
            "predicted_mean": ml["predicted_mean"],
            "ece_raw": ml["ece_raw"], "by_pick": ml["by_pick"],
        }
        # Moneyline ENSEMBLE one-line reference (comparison anchor only).
        ens = (df["ml_win_prob"].to_numpy(float)
               if "ml_win_prob" in df.columns else None)
        if ens is not None and np.isfinite(ens).any():
            eok = np.isfinite(ens)
            epk = ens >= 0.5
            ehit = (epk.astype(float) == home_won).astype(float)
            efav = np.where(epk, ens, 1.0 - ens)
            es = _winner_card_stats(efav[eok], ehit[eok],
                                    _fold_idx(pks[eok]), hold_mask[eok],
                                    epk[eok])
            ml["ml_reference"] = {
                "source": "ml_win_prob", "n": es["n"],
                "win_rate": es["win_rate"],
                "predicted_mean": es["predicted_mean"],
            }

    cards: dict[str, dict] = {}
    for name, card in (("over_under", ou), ("run_line", rl), ("derived_ml", ml)):
        if card is None:
            continue
        cards[name] = card
    return cards


# ---------------------------------------------------------------------------
# Phase 3 — production path: slate λ prediction + daily orchestration.
# ---------------------------------------------------------------------------
def _resolve_slate_key(slate: pd.DataFrame) -> str:
    """Return 'game_pk' if usable, else 'game_id' if usable, else raise.

    Pre-game ESPN slates carry ``game_id`` (e.g. "20260824_SF@BOS") but
    not ``game_pk`` (StatsAPI — only available after first pitch). A
    ``game_pk`` column counts as usable only if it actually holds at least
    one non-null value: a column that exists but is empty/all-NaN (an ESPN
    game_id that never resolved onto the StatsAPI feed) MUST fall back to
    ``game_id`` — otherwise empty keys ride through to persist_markets as
    NaNs and refuse the artifact (the 145d841 discipline).
    """
    for col in ("game_pk", "game_id"):
        if col in slate.columns and slate[col].notna().any():
            return col
    raise KeyError(
        "Slate frame has neither 'game_pk' nor 'game_id' — "
        f"columns: {sorted(slate.columns.tolist())}")


def predict_slate_runs(decided_games: pd.DataFrame, slate_games: pd.DataFrame,
                       final_fit_rounds: dict[str, int],
                       curves: dict[str, dict],
                       n_draws: int = MC_DRAWS,
                       seed: int = MARKET_SEED) -> pd.DataFrame:
    """λ + full market grid for TODAY'S SLATE.

    Side models refit on ALL decided games at fixed rounds (median fold
    early-stopping iteration — no early stopping against the future), then
    priced through the SAME α(λ) curves and MC machinery as OOF rows."""
    if slate_games.empty:
        return pd.DataFrame()
    from lightgbm import LGBMRegressor

    # Guard: ensure required columns exist with an actionable message.
    _required = ["game_date"]
    _missing = [c for c in _required if c not in slate_games.columns]
    if _missing:
        raise KeyError(
            f"Slate frame missing required column(s): {_missing} -- "
            f"available: {sorted(slate_games.columns.tolist())}")

    # Pre-game ESPN slates only have game_id, not game_pk; OOF rows always
    # carry game_pk.  Unify by always emitting game_pk, populating from
    # game_id when game_pk is absent.  A ``game_pk`` column that exists but
    # is empty/all-NaN (an ESPN game_id that never resolved) must fall back
    # to game_id per-row; rows that resolve to NEITHER key are dropped with
    # a loud warning so an unresolvable slate row never reaches
    # persist_markets as a NaN game_pk.
    from pandas import isna as _pd_isna
    _slate_key = _resolve_slate_key(slate_games)
    out = slate_games[[_slate_key, "game_date"]].copy()
    if _slate_key == "game_id":
        out = out.rename(columns={"game_id": "game_pk"})
    # Per-row resolution (the 145d841 discipline): prefer game_pk where it
    # holds a value, else fall back to game_id (a ``game_pk`` column can be
    # present-but-null for an ESPN game_id that never resolved). Emit clean
    # string keys so a numeric game_pk like 900111 never stringifies to the
    # float form '900111.0' inside the markets artifact. Row count is kept
    # fixed here so the model/grid assignments below stay position-aligned;
    # truly-unresolvable rows are dropped (with a loud log) at the end.
    _pks: list[object] = []
    for _i in out.index:
        _v = out.at[_i, "game_pk"]
        if _pd_isna(_v) and "game_id" in slate_games.columns:
            _v = slate_games.at[_i, "game_id"]
        if _v is None or _pd_isna(_v):
            _pks.append(None)
            continue
        if isinstance(_v, float) and _v == int(_v):
            _v = int(_v)
        _pks.append(str(_v))
    out["game_pk"] = _pks
    for tc in ("home_team", "away_team"):
        if tc in slate_games.columns:
            out[tc] = slate_games[tc]
    for side in ("home", "away"):
        _, cols = build_side_frame(decided_games, side)
        tr = decided_games.reindex(columns=cols).astype(float)
        model = LGBMRegressor(**RUN_LGBM_PARAMS)
        model.set_params(n_estimators=int(final_fit_rounds[side]))
        model.fit(tr, decided_games[f"{side}_score"].to_numpy(dtype=float))
        va = slate_games.reindex(columns=cols).astype(float)
        out[f"{side}_expected_runs"] = np.round(
            np.clip(model.predict(va), 1e-6, None), 4)
    alpha_h = alpha_of(out["home_expected_runs"].to_numpy(float), curves["home"])
    alpha_a = alpha_of(out["away_expected_runs"].to_numpy(float), curves["away"])
    out["alpha_home"] = np.round(alpha_h, 4)
    out["alpha_away"] = np.round(alpha_a, 4)
    mc = derive_markets_mc(out["home_expected_runs"].to_numpy(float),
                           out["away_expected_runs"].to_numpy(float),
                           alpha_h, alpha_a, n_draws=n_draws, seed=seed)
    out["kind"] = "slate"
    for j, line in enumerate(TOTAL_LINE_GRID):
        key = f"p_over_{str(line).replace('.', '_')}"
        out[key] = np.round(mc["p_over_grid"][:, j], 5)
        out[key.replace("p_over_", "p_push_")] = np.round(
            mc["p_push_grid"][:, j], 5)
        out[key.replace("p_over_", "p_under_")] = np.round(
            1 - mc["p_over_grid"][:, j] - mc["p_push_grid"][:, j], 5)
    for j, m in enumerate(RUN_LINE_GRID):
        out[f"p_home_cover_{str(m).replace('.', '_')}"] = np.round(
            mc["p_cover_grid"][:, j], 5)
    for j, m in enumerate(RUN_LINE_GRID_FULL):
        out[rl_col(m, "home")] = np.round(mc["p_rl_home_grid"][:, j], 5)
        out[rl_col(m, "push")] = np.round(mc["p_rl_push_grid"][:, j], 5)
        out[rl_col(m, "away")] = np.round(mc["p_rl_away_grid"][:, j], 5)
    out["p_home_win_derived"] = np.round(mc["p_home_win_derived"], 5)
    out["p_away_win_derived"] = np.round(1 - mc["p_home_win_derived"], 5)
    # Undecided by definition; excluded from the NaN-checked numeric contract.
    out["home_score"] = pd.NA
    out["away_score"] = pd.NA
    out["total_runs"] = pd.NA
    out["agreement_conflict"] = False
    out["ml_win_prob"] = np.nan
    # Drop rows still without a resolvable key (loud, honest) — an
    # unresolvable ESPN game_id must never be persisted as a NaN game_pk.
    # Done last so the grid rows above stay position-aligned with slate_games.
    unresolved = out["game_pk"].isna()
    if unresolved.any():
        n = int(unresolved.sum())
        logger.warning(
            "predict_slate_runs: dropping %d unresolvable slate row(s) "
            "(no game_pk/game_id) — excluded from the markets artifact "
            "rather than persisted with a NaN key", n)
        out = out[~unresolved].reset_index(drop=True)
    return out


def compute_rolling_totals_brier(markets: pd.DataFrame,
                                 window_days: int = 30,
                                 min_games_per_day: int = 10) -> dict:
    """Rolling trailing-window Brier of p_over_8_5 vs decided totals — mirrors
    explainability.compute_rolling_brier conventions for the moneyline."""
    empty = {"series": [], "window_days": window_days,
             "min_games_per_day": min_games_per_day}
    if markets is None or not len(markets):
        logger.warning("Rolling totals Brier: no markets history — empty series")
        return empty
    df = markets[markets["kind"] == "oof"].copy()
    df["d"] = pd.to_datetime(df["game_date"])
    df = df.dropna(subset=["total_runs", "p_over_8_5"])
    if not len(df):
        logger.warning("Rolling totals Brier: no decided OOF rows — empty series")
        return empty
    df["brier"] = (df["p_over_8_5"] - (df["total_runs"] >= 9).astype(float)) ** 2
    daily = df.groupby("d").agg(n=("brier", "size"), b=("brier", "mean"))
    daily = daily[daily["n"] >= min_games_per_day]
    series = []
    for d, _ in daily.iterrows():
        win = daily.loc[(daily.index > d - pd.Timedelta(days=window_days))
                        & (daily.index <= d)]
        if len(win):
            series.append({"date": d.strftime("%Y-%m-%d"),
                           "brier": round(float(win["b"].mean()), 5)})
    return {"series": series, "window_days": window_days,
            "min_games_per_day": min_games_per_day,
            "history_mean_brier": round(float(df["brier"].mean()), 5)}


def run_engine_daily(games: pd.DataFrame, target_games: pd.DataFrame,
                     target_date_str: str,
                     n_draws: int = MC_DRAWS,
                     decided_snapshot: Optional[pd.DataFrame] = None,
                     ) -> dict[str, Any]:
    """Full Phase-3 daily pass inside the pipeline.

    OOF re-derivation (α(λ) fitted on PRE-HOLDOUT OOF only) → markets artifact
    with the full line grid → slate λ priced through the same curves →
    agreement conflicts vs predictions_history → rolling totals Brier.
    Returns the monitor-embed block plus written artifact paths."""
    decided = (decided_snapshot.copy() if decided_snapshot is not None
               else get_decided_frame(games))
    result = run_oof(decided, decided_snapshot=decided)
    oof = result["oof"]

    history_path = DATA_DELIVERY_DIR / f"predictions_history_{target_date_str}.csv"
    if not history_path.exists():
        history_path = DATA_DELIVERY_DIR / "predictions_history_latest.csv"
    ml_probs = load_moneyline_probs(history_path)
    if ml_probs is None:
        logger.warning("Run engine daily: %s unavailable — agreement filter "
                       "will report zero coverage", history_path.name)

    mk = derive_markets_v3(oof, moneyline_probs=ml_probs, n_draws=n_draws)
    summary = mk["summary"]
    markets = mk["markets"]
    curves = {s: summary[f"alpha_{s}"] for s in ("home", "away")}

    slate_frame = predict_slate_runs(
        decided, target_games.copy(), result["summary"]["final_fit_rounds"],
        curves, n_draws=n_draws)
    if (not slate_frame.empty and ml_probs is not None
            and "home_win_prob_model" in target_games.columns):
        # Use whichever key exists: game_pk (StatsAPI) or game_id (ESPN).
        _tg_key = _resolve_slate_key(target_games)
        tp_map = dict(zip(target_games[_tg_key],
                          target_games["home_win_prob_model"]))
        # slate_frame already unified to game_pk by predict_slate_runs
        m = slate_frame["game_pk"].map(tp_map)
        known = m.notna()
        if known.any():
            slate_frame.loc[known, "ml_win_prob"] = m[known]
            slate_frame.loc[known, "agreement_conflict"] = (
                (slate_frame.loc[known, "p_home_win_derived"]
                 - slate_frame.loc[known, "ml_win_prob"]).abs()
                > AGREEMENT_FILTER_DELTA)
            slate_stats = agreement_stats(
                slate_frame.loc[known, "p_home_win_derived"].to_numpy(),
                m[known].to_numpy(), delta=AGREEMENT_FILTER_DELTA)
            slate_stats["scope"] = "today_slate"
            summary["agreement_slate"] = slate_stats

    combined = (pd.concat([markets, slate_frame], ignore_index=True)
                if not slate_frame.empty else markets)
    # Persist the markets artifact LAST so a persist failure leaves the OOF +
    # monitor block intact and loudly flags markets_persisted=False (never a
    # silent stale serve). Task 0's row-resolution fix makes this path robust
    # to unresolvable slate rows; this wraps it defensively regardless.
    oof_path = persist_oof(oof, target_date_str)
    markets_persisted = True
    markets_persist_error: Optional[str] = None
    mkt_path: Optional[Path] = None
    try:
        mkt_path = persist_markets(combined, target_date_str, summary)
    except Exception as exc:
        markets_persisted = False
        markets_persist_error = f"{type(exc).__name__}: {exc}"
        logger.error(
            "Run engine: markets persist FAILED — "
            "run_engine_markets_%s.csv NOT written (monitor must flag it): %s",
            target_date_str, exc, exc_info=True)
    totals_brier = compute_rolling_totals_brier(combined)

    s1 = result["summary"]
    holdout_markets = {
        k.replace("market_", ""): v.get("holdout")
        for k, v in summary.items()
        if k.startswith("market_") and isinstance(v, dict) and v.get("holdout")
    }
    monitor_block = {
        "version": "phase3",
        "winner_cards": compute_winner_cards(
            markets, oof, HOLDOUT_DAYS),
        "alpha_home": summary["alpha_home"],
        "alpha_away": summary["alpha_away"],
        "phase2_single_alpha": summary["phase2_single_alpha"],
        "fit_check_single_alpha": summary["fit_check_single_alpha"],
        "fit_check_alpha_lambda": summary["fit_check_alpha_lambda"],
        "variance_check": summary["variance_check"],
        "year_effect_home": summary["year_effect_home"],
        "year_effect_away": summary["year_effect_away"],
        "mc_meta": summary["mc_meta"],
        "line_grid": summary["line_grid"],
        "holdout_gate": {
            "cutoff": summary["holdout_cutoff"],
            "n_pre": summary["n_pre"], "n_holdout": summary["n_holdout"],
            "markets": holdout_markets,
            "totals_beat_baselines_holdout": all(
                bool(v.get("beats_baseline_logloss"))
                for k, v in holdout_markets.items() if k.startswith("over_")),
        },
        "agreement_vs_moneyline": summary.get("agreement_vs_moneyline"),
        "agreement_slate": summary.get("agreement_slate"),
        "agreement_delta": AGREEMENT_FILTER_DELTA,
        "market_metrics": {
            k.replace("market_", ""): v for k, v in summary.items()
            if k.startswith("market_") and isinstance(v, dict)
            and not k.endswith("_holdout")  # holdout data nested in main entry
        },
        "rolling_totals_brier": totals_brier,
        "phase1": {"n_folds": s1["n_folds"], "n_games": s1["n_games"],
                   "dispersion_ratio": {
                       "home": s1["home_dispersion_ratio"],
                       "away": s1["away_dispersion_ratio"]},
                   "final_fit_rounds": s1["final_fit_rounds"]},
    }
    artifacts = [str(oof_path)]
    if mkt_path is not None:
        artifacts.append(str(mkt_path))
        artifacts.append(
            str(DATA_DELIVERY_DIR
                / f"run_engine_markets_{target_date_str}.meta.json"))
    return {"block": monitor_block, "artifacts": artifacts,
            "markets_persisted": markets_persisted,
            "markets_persist_error": markets_persist_error}


def _print_phase2(s: dict[str, Any],
                  fc_key: str = "fit_check_single_alpha") -> None:
    print("\n========== PHASE 2 — NB MARGINALS ==========")
    print(f"α_home={s['alpha_home']}  α_away={s['alpha_away']}  "
          f"(N={s['n_draws']:,} draws, seed={s['seed']})")
    for side in ("home", "away"):
        print(f"\n{side.upper()} fit-check (single global alpha):")
        print(f"{'k':>6}{'modeled':>10}{'observed':>10}")
        for row in s[fc_key][side]:
            print(f"{str(row['k']):>6}{row['modeled_p']:>10.4f}{row['observed_p']:>10.4f}")
    print("\n========== MARKET SCORING (pooled OOF) ==========")
    print(f"{'market':<20}{'logloss':>9}{'brier':>8}{'ECE(raw)':>9}"
          f"{'ECE(cal)':>9}{'base LL':>9}{'ΔLL':>9}")
    for key, name in (("market_over_8_5", "over 8.5"),
                      ("market_home_cover_1_5", "home -1.5"),
                      ("market_derived_moneyline", "derived ML")):
        m = s[key]
        print(f"{name:<20}{m['engine_logloss']:>9.4f}{m['engine_brier']:>8.4f}"
              f"{m['engine_ece_raw']:>9.4f}{m['engine_ece_calibrated']:>9.4f}"
              f"{m['baseline_logloss']:>9.4f}"
              f"{m['engine_logloss'] - m['baseline_logloss']:>9.4f}")
    if "agreement_vs_moneyline" in s:
        a = s["agreement_vs_moneyline"]
        print(f"\nAgreement vs moneyline ensemble (n={a['n']}): "
              f"mean|d|={a['mean_abs_diff']}, flagged@{a['delta_primary']}: "
              f"{a['n_flagged_primary']} ({100*a['share_gt_primary']:.1f}%), "
              f"@0.10: {100*a['share_gt_0_10']:.1f}%")


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

    # Phase 2 consumes Phase 1's OOF λ directly (same process, no retrain).
    history_path = DATA_DELIVERY_DIR / f"predictions_history_{date_stamp}.csv"
    if not history_path.exists():
        history_path = DATA_DELIVERY_DIR / "predictions_history_latest.csv"
    ml_probs = load_moneyline_probs(history_path)

    mk = derive_markets_v3(result["oof"], moneyline_probs=ml_probs)
    oof_path = persist_oof(result["oof"], date_stamp)
    mkt_path = persist_markets(mk["markets"], date_stamp, mk["summary"])

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
    print(f"\nOOF artifact: {oof_path}")
    _print_phase2(mk["summary"])
    s3 = mk["summary"]
    print("\n========== PHASE 3 — α(λ) vs SINGLE α ==========")
    for side in ("home", "away"):
        a2 = s3[f"alpha_{side}"]
        sel = a2["selection"]
        params = {k: v for k, v in a2.items()
                  if k in ("a", "b", "c", "lam", "alpha")}
        print(f"\n{side}: form={a2['form']} params={params}")
        sel_summary = [
            (f, sel[f]["tail_gap_avg"], sel[f]["loglik_avg"])
            for f in ("piecewise", "linear", "power") if f in sel
        ]
        print(f"  OOB selection (tail_gap, loglik): {sel_summary}")
        print(f"  bins: {sel['bins']}")
        print(f"  year effect: {s3[f'year_effect_{side}']}")
        print(f"  variance check: {s3['variance_check'][side]}")
        fc2 = {r["k"]: r for r in s3["fit_check_single_alpha"][side]}
        fc3 = {r["k"]: r for r in s3["fit_check_alpha_lambda"][side]}
        order = list(range(13)) + ["≥10", "≥11", "≥12", "≤1"]
        print(f"  {'k':>5}{'single-a':>10}{'alpha(l)':>10}{'observed':>10}")
        for k in order:
            o = fc2.get(k, fc3.get(k, {})).get("observed_p")
            m2 = fc2.get(k, {}).get("modeled_p")
            m3 = fc3.get(k, {}).get("modeled_p")
            if m2 is None or m3 is None or o is None:
                continue
            print(f"  {str(k):>5}{m2:>10.4f}{m3:>10.4f}{o:>10.4f}")
    print("\n========== MULTI-LINE SCORING (pooled | holdout) ==========")
    print(f"{'market':<22}{'LL':>8}{'Brier':>8}{'ECEcal':>8}{'baseLL':>8}"
          f"{'hLL':>8}{'hBaseLL':>9}{'beats?':>7}")
    for key in sorted(k for k in s3 if k.startswith("market_")):
        m = s3[key]
        if not isinstance(m, dict):
            continue
        h = m.get("holdout") or {}
        name = key.replace("market_", "")
        print(f"{name:<22}{m['engine_logloss']:>8.4f}{m['engine_brier']:>8.4f}"
              f"{m.get('engine_ece_calibrated', float('nan')):>8.4f}"
              f"{m['baseline_logloss']:>8.4f}"
              f"{h.get('engine_logloss', float('nan')):>8.4f}"
              f"{h.get('baseline_logloss', float('nan')):>9.4f}"
              f"{str(h.get('beats_baseline_logloss', '-')):>7}")
    if "agreement_vs_moneyline" in s3:
        print(f"\nAgreement: {s3['agreement_vs_moneyline']}")
    print(f"\nMarkets artifact: {mkt_path}")


if __name__ == "__main__":
    main()
