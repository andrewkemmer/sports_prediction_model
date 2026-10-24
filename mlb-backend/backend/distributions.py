"""Distributions engine (formerly run_engine.py) — per-team expected-runs
models (λ per side).

THE GOLDEN RULE (REVISED 2026-08-30): run models consume LEVELS + ENVIRONMENT
+ the 24 restored matchup-gap _diff features. Diff features (sp_xfip_diff ≈ 0
for ace-vs-ace AND bad-vs-bad) carry no scoring-LEVEL information, so an
earlier rule excluded them all — but the 2026-08-30 A/B on the current frame
(run_engine_cull_diagnostic_20260830.json, 6,829 OOF) showed that restore
helps the prices, not just the count: derived-ML AUC 0.5515→0.5682, spread sd
+27%, share>0.55 up 0.19→0.25, lambda deviance/RMSE down on both sides,
holdout LL 0.6865→0.6846 (the earlier DO-NOT-SHIP verdict was measured on a
stale artifact where calibration degraded 0.0119→0.0144). park_factor_slug_diff
stays kept as a park-context term; the 5 engineered composite interactions and
run_margin_diff (a lambda-derived moneyline-side feature) stay excluded. The
kept list is DERIVED from MONEYLINE_FEATURE_COLS at call time so new features flow in
(and are logged); only the exclusion RULE lives here.

One regularized LightGBM regressor (objective="poisson") per side, trained on
the SAME fixed walk-forward folds as the moneyline pipeline. Pooled OOF scoring,
baseline comparison vs constant league-mean, Pearson chi-square/dispersion
probe for score-mean diagnostics, separate from split-skew variance pricing.

FEATURE PARITY (2026-09-27): the run line's per-side Poisson models train
and price on EXACTLY the binary moneyline's active feature list — the same
served columns, nothing appended. The P1 projection input seam
(attach_projection_levels / sp_proj_era_<opp>, adopted by gate 7e4c529 on
2026-09-04) was REMOVED here; the projection producer (sp_projection.py)
survives as a standalone research module only. The prior P1 arm measured
sealed margin CRPS 2.53856 -> 2.51736 and P(win) SD 0.0409 -> 0.0546, but
run-line/moneyline feature identity is the governing contract.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional

import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from frames import get_decided_frame, fold_signature

import config as config_mod
from config import (
    AGREEMENT_FILTER_DELTA,
    DATA_DELIVERY_DIR,
    MIN_VAL_FOLD_GAMES,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
    SCORE_ELASTICNET_PARAMS,
    SCORE_ENSEMBLE_MEMBERS,
    SCORE_ENSEMBLE_WEIGHTS,
    SCORE_XGBOOST_PARAMS,
)

logger = logging.getLogger(__name__)

# HISTORICAL (2026-08-30 derivation era): moneyline-only composites kept out
# of the run engine's λ view — pure matchup terms with no level info, and
# interactions whose factors are themselves matchup diffs. The legacy rule
# that consumed this set was removed 2026-09-27; the production run view is
# the FULL active moneyline list (build_side_frame), so these columns are
# now SERVED — the side-view split prices each side from its own columns and
# the tree members may use them freely.
RUN_EXTRA_EXCLUSIONS = {
    "lineup_handedness_matchup_advantage_diff",
    "bullpen_meltdown_risk_diff",     # pitches_diff × whip_10g_diff
    # RENAMED 2026-09-27: the three interaction composites take their *_diff
    # names; the run engine never consumed the composites, so the twins ride
    # the same exclusion (matchup signal, not a scoring LEVEL).
    "pitcher_regression_indicator_diff",   # velo_diff × era_diff
    "pitcher_regression_indicator_home",   # within-side velo × era
    "pitcher_regression_indicator_away",
    "lineup_depth_multiplier_diff",        # woba_mean_diff × top3_diff
    "lineup_depth_multiplier_home",
    "lineup_depth_multiplier_away",
    "ace_efficiency_factor_diff",          # k9_diff × whiff_diff
    "ace_efficiency_factor_home",
    "ace_efficiency_factor_away",
    # Phase 2 lineup-delta features (actual starting-9 wOBA vs team season) —
    # matchup/form signal, moneyline-only; excluded so the run engine's
    # raw-only view stays byte-identical (GOLDEN RULE: levels + environment).
    "lineup_actual_woba_delta_home", "lineup_actual_woba_delta_away",
    "lineup_actual_top3_delta_home", "lineup_actual_top3_delta_away",
    "lineup_rest_count_home", "lineup_rest_count_away",
    # Categorical-context columns (venue + the two starter IDs) — moneyline
    # TREE_CATEGORICAL_COLS inputs only, not scoring LEVELS. They are not in
    # MONEYLINE_FEATURE_COLS today; named here so the rule also drops them if they ever
    # enter a frame the run engine derives from (kept/dropped lists from
    # MONEYLINE_FEATURE_COLS stay byte-identical — asserted in
    # test_categorical_venue_starters.py).
    "venue", "home_starter_id", "away_starter_id",
}

# ── Tree-member team-ID categoricals (2026-09-23 ADOPT) ────────────────────
# The moneyline trees have carried the team-ID pair since the categorical
# ablation (training.TREE_CATEGORICAL_COLS); the run engine's Poisson side
# models were numeric-only. The scratch A/B (scratch/runengine_teamid_
# ablation.json, 2026-09-23, 6,811 OOF games / 74 canonical folds, paired
# per-fold deltas) measured BOTH sides improving on the production contract:
#   home: deviance −0.00890 (SE 0.00480)  CRPS −0.00525 (SE 0.00235)
#   away: deviance −0.02061 (SE 0.00669)  CRPS −0.00995 (SE 0.00319)
# derived-ML (diagnostic conversion) logloss −0.00114, AUC +0.00806; the
# trees took ~37k splits on the pair (real usage, not noise), mean |Δλ|
# 0.27 runs, spread +33%. venue_id / starter categories stay EXCLUDED —
# the moneyline's DON'T-ADOPT verdict for that arm stands untested here.
#
# Mechanism: the pair rides ALONGSIDE the numeric contract as LightGBM
# native categoricals (same member mechanism as the moneyline LGB head).
# Frames are built float-only as before; the ID columns are attached by
# _materialize_side_frame at the LAST moment (opt-in via the flag below),
# so every astype(float) consumer site keeps working unchanged and the
# numeric kept/dropped lists stay byte-identical.
RUN_TREE_CATEGORICAL_COLS = ["home_team_id", "away_team_id"]
RUN_WITH_TEAM_IDS = True          # the adopted production config


def _materialize_side_frame(games: pd.DataFrame, cols: list[str],
                            side: str) -> tuple[pd.DataFrame, list[str]]:
    """Final frame materialization — unchanged numeric contract.

    The returned columns are the NUMERIC view exactly as before (the
    kept/dropped contract stays byte-identical); the ID pair is NOT in the
    frame here. Every fitting site reindexes its fold/slate slices on the
    returned column list and float-casts, which would flatten category
    dtypes — so the pair is attached AFTER that by
    :func:`_apply_categorical_ids`. This function only validates the flag's
    preconditions (source columns present or attachable) so a mis-wired
    caller fails loud here rather than silently training numeric-only.
    """
    frame = games.reindex(columns=cols).astype(float)
    return frame, list(cols)


def _apply_categorical_ids(*frames: pd.DataFrame,
                           rows: pd.DataFrame,
                           ) -> None:
    """Attach the team-ID pair as LightGBM categoricals, in place.

    Called at the fitting sites AFTER the float reindex (which would cast a
    categorical back to float) and after the side-view slice has dropped
    the raw ``home_team``/``away_team`` name columns — so the caller must
    pass the PRE-SLICE rows (``rows``) that positionally correspond 1:1 to
    each model frame (the fold's train/val games or the slate). IDs resolve
    from those rows' own ``home_team``/``away_team`` values via
    ``training._add_team_ids``: the mapper is process-stable, so the same
    abbreviation always maps to the same integer across train/val/slate,
    and values it has never seen (plus missing name columns on privacy-
    stripped slate fixtures) map to the reserved UNK slot — the moneyline's
    own UNK contract. In-place: category dtype on both ID columns.
    """
    if not (RUN_WITH_TEAM_IDS and RUN_TREE_CATEGORICAL_COLS):
        return
    if rows is None:
        raise ValueError(
            "_apply_categorical_ids requires rows= (the pre-slice frame) so "
            "team abbreviations can resolve — a positional 1:1 match with "
            "every model frame passed alongside")
    from training import _add_team_ids
    for f in frames:
        if len(f) != len(rows):
            raise ValueError(
                f"frame length {len(f)} != rows length {len(rows)} — the "
                "model frame and its source rows must align positionally")
        ids = _add_team_ids(rows)
        for c in RUN_TREE_CATEGORICAL_COLS:
            f[c] = pd.Categorical(ids[c].to_numpy())
# The one sanctioned _diff survivor: a PARK context multiplier, not a matchup gap.
RUN_DIFF_EXCEPTION = "park_factor_slug_diff"

# 2026-08-30 feature-restore decision (run_engine_cull_diagnostic_20260830.json):
# the 24 matchup-gap _diff features were RESTORED to the served run view —
# the fresh A/B on the 2026-08-30 frame (6,829 OOF) showed restoring them
# improves EVERYTHING: derived-ML AUC 0.5515 -> 0.5682, margin-spread sd +27%,
# the share of wide prices (~>0.55) up 0.19 -> 0.25, home/away lambda deviance
# and RMSE DOWN, holdout logloss 0.6865 -> 0.6846. Two restored features show
# material drift (woba_30g_diff 0.296, lineup_re24_top3_diff 0.104 WARN) but
# still net-improve in the A/B; they stay drift-monitored so
# classify_drift_retention can act. 2026-09-27 twin expansion: the 16 exp2
# per-side twins are the halves of matchup-gap diffs and joined the served
# view with everything else. The frozensets below now carry the HISTORY of
# those exclusions (the legacy rule that consumed them was removed
# 2026-09-27); the production contract is the full active moneyline list.
RUN_EXP2_TWIN_EXCLUSIONS = frozenset({
    "exp2_centered_k_home", "exp2_centered_k_away",
    "exp2_cat_k_fastball_home", "exp2_cat_k_fastball_away",
    "exp2_cat_k_breaking_home", "exp2_cat_k_breaking_away",
    "exp2_cat_k_offspeed_home", "exp2_cat_k_offspeed_away",
    "exp2_cat_xwoba_fastball_home", "exp2_cat_xwoba_fastball_away",
    "exp2_cat_xwoba_breaking_home", "exp2_cat_xwoba_breaking_away",
    "exp2_cat_xwoba_offspeed_home", "exp2_cat_xwoba_offspeed_away",
    "exp2_cat_platoon_k_fastball_home", "exp2_cat_platoon_k_fastball_away",
})

RUN_RESTORED_DIFF_FEATURES = frozenset({
    "win_pct_diff", "elo_diff", "rest_days_diff",
    "sp_xfip_diff", "sp_xfip_5g_diff", "sp_k9_diff", "sp_k9_5g_diff",
    "sp_fbvelo_diff", "sp_fbpct_diff", "sp_whiff_diff", "sp_xwoba_diff",
    "sp_xwoba_vs_l_diff", "lineup_re24_mean_diff", "lineup_re24_top3_diff",
    "lineup_re24_std_diff", "woba_30g_diff", "bullpen_kbb_10g_diff",
    "bullpen_kbb_3g_diff", "bullpen_pitches_3d_diff", "team_barrel_15g_diff",
    "team_hardhit_15g_diff", "team_exitvelo_15g_diff", "travel_fatigue_diff",
    "closer_availability_diff",
})

# The standalone env-LEVEL columns (park_wind_factor, air_density_level,
# park_factor_slug, dome_is_neutral_game) are computed into the artifact frame
# (features.add_env_level_features) and feed the moneyline's DERIVED env
# features, but they are NOT served features and the run engine never
# appends them: since the 2026-09-27 feature-parity contract the run line's
# side models train on EXACTLY the active moneyline list. If an env level
# is ever adopted into the feature set, it joins MONEYLINE_FEATURE_COLS and
# flows to the run line automatically — there is no separate append path.

MAX_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 20  # matches the tuned LightGBM fold convention

# 2026-10-14 NBA-protocol walk-forward tune (scratch/mean_tuning_multisport.
# py, 7,035 priced OOF rows / 84 folds, judged at seeds 42 AND 44 on
# fixed-line proper scores + mu RMSE + derived-ML AUC, winner-card as the
# no-regression check): winner ff05_double_rounds = half-rate feature
# subsampling (colsample 0.8 -> 0.5 + feature_fraction 0.5) with the fixed
# 90-round budget doubled (see RUN_FIXED_FIT_ROUNDS). Deep averaging alone
# UNDERFITS at 90 rounds (deep_average arm: mu_total RMSE 4.4480 -> 4.4613,
# derived-ML AUC 0.5692 -> 0.5629), which is why the budget moves with the
# subsampling rather than before it. Every priced line improves at BOTH
# seeds (totals 8/9/10 logloss -0.0006/-0.0007/-0.0004 and run line 1.5
# -0.0014 at seed 42; totals -0.0008/-0.0009/-0.0005 and run line 1.5
# -0.0004 at seed 44), derived moneyline AUC 0.5692 -> 0.5763 (0.5728 at
# seed 44), logloss 0.6838 -> 0.6824; mu_total RMSE 4.4480 -> 4.4458.
# Record: scratch/mean_tuning_mlb_report.json (scratch/ is gitignored).
RUN_LGBM_PARAMS = {
    "objective": "poisson",
    "learning_rate": 0.03,
    "num_leaves": 8,
    "min_child_samples": 40,
    "min_gain_to_split": 0.5,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.5,
    "feature_fraction": 0.5,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}


# ---------------------------------------------------------------------------
# Objective layer (NBA parity, 2026-10-13)
#
# NBA's distributions module carries an opt-in analytic CRPS location
# objective next to the production Poisson path; MLB's run engine now exposes
# the same surface so all four backends share one objective contract. The
# default stays Poisson -- the NBA walk of 2026-10-11 underperformed, so
# nothing here changes production behavior until a caller passes crps=True.
# ---------------------------------------------------------------------------
class GaussianCRPSObjective:
    """Analytic proper-score location objective at a train-only scale.

    Initial score is supplied explicitly to LightGBM and restored at predict.
    Scale adaptation then uses past OOF via the lattice-CRPS CDF layer.
    """

    def __init__(self, scale):
        self.scale = max(float(scale), 1.)

    def __call__(self, y, prediction):
        from scipy.special import ndtr
        z = (np.asarray(prediction) - y) / self.scale
        gradient = 2. * ndtr(z) - 1.
        hessian = 2. * np.exp(-.5 * z ** 2) / (np.sqrt(2 * np.pi) * self.scale)
        return gradient, np.maximum(hessian, 1e-5)


def _make_reg(scale=12., crps=False, params: Optional[dict] = None):
    """Side regressor builder: MLB's RUN_LGBM_PARAMS + the objective switch."""
    from lightgbm import LGBMRegressor
    built = dict(RUN_LGBM_PARAMS if params is None else params)
    built["objective"] = GaussianCRPSObjective(scale) if crps else "poisson"
    return LGBMRegressor(**built)


def gaussian_crps(y, location, scale):
    """Analytic continuous Gaussian CRPS, useful for objective verification."""
    from scipy.special import ndtr
    scale = np.asarray(scale, float)
    if np.any(scale <= 0):
        raise ValueError("CRPS scale must be positive")
    z = (np.asarray(y, float) - np.asarray(location, float)) / scale
    return scale * (z * (2 * ndtr(z) - 1.)
                    + 2 * np.exp(-.5 * z ** 2) / np.sqrt(2 * np.pi)
                    - 1. / np.sqrt(np.pi))


def _grid_key(prefix: str, value: float | int) -> str:
    """Same line-column key builder NBA carries. Present so the shared
    pricing body is complete here too: MLB's own branch spells its columns
    with rl_col/suffix strings, but the generic branch below resolves through
    this helper -- an undefined name on the non-MLB path is exactly the kind
    of structural gap the 2026-10-13 parity audit closed."""
    s = str(float(value))
    if s.endswith(".0"):
        s = s[:-2]
    return f"{prefix}_{s.replace('-', 'm').replace('.', '_')}"


# ---------------------------------------------------------------------------
# C2 edge expansion — REMOVED (2026-09-27). Retired to monitor-only on
# 2026-09-22 (adoption per the run-engine tuning policy: lr 0.03 x fixed 90
# rounds gives k-hat ~= 1.0 on the OOF basis, so the transform was inert by
# construction); the diagnostic module run_engine_k_edge.py — which fitted
# and published the monitor-only k-hat — was deleted with it once no
# consumer remained. The published probabilities price the RAW lambda pair.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Feature-view derivation
# ---------------------------------------------------------------------------
def _resolve_run_view(
        feature_cols: Optional[list[str]] = None,
) -> tuple[list[str], list[str]]:
    """Resolve the run engine's served view: the active moneyline feature
    list, VERBATIM — every served column flows in, nothing is dropped.

    Single seam used by build_side_frame (and pinned by the tests); the
    production contract has been call-time resolution since the 2026-08-30
    matchup-gap restore, and the historical *_diff-drop rule that once sat
    here was removed 2026-09-27. ``feature_cols`` overrides the list for
    alternative universes (diagnostics only). Returns (features, dropped) —
    ``dropped`` is always empty; it exists so callers can assert the
    no-drop contract loudly.
    """
    if feature_cols is None:
        from training import active_moneyline_feature_cols
        feature_cols = active_moneyline_feature_cols()
    return list(feature_cols), []


# The run engine has one feature contract: the active moneyline list. It is
# resolved at call time so adopted RFE additions/removals automatically apply
# to expected-run fitting, run-line pricing, and monitoring.


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


# Content key of the last run-view resolution already logged at INFO
# (2026-10-05 log review: the identical "active moneyline feature view"
# line printed 6x per run — every resolution returns the same list, so the
# repeats carried no information). Keyed on the RESOLVED CONTENT, not a call
# counter: a view that changes mid-process (adoption between calls) logs
# again as a distinct line. See _log_resolved_view.
_LAST_LOGGED_RUN_VIEW: Optional[tuple[str, ...]] = None


def _log_resolved_view(feats: list[str]) -> None:
    """Log the resolved run-engine feature view once per DISTINCT view.

    ``build_side_frame`` re-resolves the active moneyline list on every call
    (parity contract), and production calls it six times per run with the
    same result — six byte-identical INFO lines the reader cannot tell
    apart. The first resolution of a given view logs at INFO; repeats drop
    to DEBUG; a *different* view logs at INFO again.
    """
    global _LAST_LOGGED_RUN_VIEW
    key = tuple(feats)
    if key == _LAST_LOGGED_RUN_VIEW:
        logger.debug("Run engine: active moneyline feature view unchanged "
                     "(%d features)", len(feats))
        return
    logger.info("Run engine: active moneyline feature view (%d features)",
                len(feats))
    _LAST_LOGGED_RUN_VIEW = key


def build_side_frame(games: pd.DataFrame, side: str,
                     run_features: Optional[list[str]] = None,
                     dropped: Optional[list[str]] = None,
                     strict_feature_parity: bool = False,
                     ) -> tuple[pd.DataFrame, list[str]]:
    """Materialize the side's model frame from the run view, preserving
    NaN (LightGBM routes it natively). Logs the resolved view once per
    DISTINCT resolution (see ``_log_resolved_view`` — identical repeats
    used to print the same line 6x per run).

    The side view is EXACTLY the resolved feature list — nothing appended.
    The former env-LEVEL append path (RUN_LEVEL_ENV_FEATURES /
    include_level_env, the pre-parity 'ablation arm A/B' switch) was
    removed 2026-09-27: the run line trains on exactly what the moneyline
    serves, and any future env feature must join the served list to reach
    the run view."""
    feats = list(run_features) if run_features is not None else None
    if strict_feature_parity or (feats is None and dropped is None):
        # Synchronization contract: every production run model receives exactly
        # the active moneyline feature list. Resolve it at call time so adopted
        # RFE additions/removals automatically affect run-line serving.
        feats, dropped = _resolve_run_view()
        _log_resolved_view(feats)
    elif feats is None:
        raise ValueError("run_features and dropped must be supplied together")
    if strict_feature_parity:
        # Do not apply the legacy side/level/environment split in parity mode:
        # both home and away run regressors receive the same named columns.
        cols = list(feats)
    else:
        side_cols, env_cols = split_side_view(feats, side)
        cols = side_cols + env_cols
        # FEATURE PARITY: the side view carries exactly the served moneyline
        # columns (side split + shared environment). No extra inputs — the
        # former P1 projection append (sp_proj_era_<opp>) was removed
        # 2026-09-27 so both models train on identical features.
    frame, out_cols = _materialize_side_frame(games, cols, side)
    # Record the pair on the returned list (contract metadata for callers
    # that need the full model width) — but NOT in the frame; the fitting
    # sites attach the categorical columns post-reindex via
    # _apply_categorical_ids.
    if RUN_WITH_TEAM_IDS and RUN_TREE_CATEGORICAL_COLS:
        out_cols = list(out_cols) + list(RUN_TREE_CATEGORICAL_COLS)
    return frame, out_cols


# ---------------------------------------------------------------------------
# Training / OOF scoring
# ---------------------------------------------------------------------------
# 2026-10-14: 90 -> 180, the second half of the RUN_LGBM_PARAMS tune above
# (ff05_double_rounds). This pin is the side Poisson models' boosting
# budget; at the incumbent 90 rounds the deep-averaging arm measured
# UNDERFIT (comment above the params dict), so the winner pairs half-rate
# feature subsampling with a doubled budget.
RUN_FIXED_FIT_ROUNDS = 180


NB_DISPERSION_MIN = 1.0
NB_DISPERSION_MAX = 1e4
NB_RAW_CLIP = (-5.0, 12.0)


def estimate_nb_dispersion(y: np.ndarray) -> float:
    """Method-of-moments NB dispersion r from train labels (per side)."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y) & (y >= 0)]
    if len(y) < 2:
        return NB_DISPERSION_MAX
    m, v = float(y.mean()), float(y.var(ddof=1))
    if not np.isfinite(m) or m <= 0 or not np.isfinite(v) or v <= m:
        return NB_DISPERSION_MAX
    return float(np.clip(m * m / (v - m),
                         NB_DISPERSION_MIN, NB_DISPERSION_MAX))


class NegativeBinomialObjective:
    """NB2 NLL grad/hess w.r.t. the raw log-mean score (NBA template)."""

    def __init__(self, dispersion: float) -> None:
        self.r = float(np.clip(dispersion, NB_DISPERSION_MIN,
                               NB_DISPERSION_MAX))

    def __call__(self, y_true, y_pred):
        y = np.asarray(y_true, dtype=float)
        mu = np.exp(np.clip(np.asarray(y_pred, dtype=float), *NB_RAW_CLIP))
        r = self.r
        grad = -y + (y + r) * mu / (mu + r)
        hess = (y + r) * mu * r / (mu + r) ** 2
        return grad, np.maximum(hess, 1e-5)


def _fit_side_model(params: dict, tr_frame: pd.DataFrame, y_tr: np.ndarray,
                    va_frame: pd.DataFrame, y_va: np.ndarray,
                    fixed_rounds: Optional[int] = None,
                    crps: bool = False, nb: bool = False):
    """Fit one side's Poisson model.

    fixed_rounds (run-engine OOF + production path): train exactly that many
    rounds with no eval set -- the validation frame never selects fold model
    complexity (db69935 discipline).

    fixed_rounds=None (build_oof_margin's caller-supplied-fold path ONLY): the
    validation fold early-stops the iteration count, per that module's
    documented margin-feature contract. Not used by run_engine's own OOF.

    crps (default False): NBA-parity opt-in that swaps the Poisson
    likelihood for the analytic Gaussian CRPS location objective with an
    explicit train-mean init score. Production callers never set it.

    nb (default False): trial-only Negative-Binomial challenger (2026-10-17)
    — NB2 log-link with train-fold dispersion, same matrix and rounds.
    """
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    if crps and nb:
        raise ValueError("_fit_side_model takes one challenger at a time")
    model = LGBMRegressor(**params)
    crps_scale = max(float(np.std(np.asarray(y_tr, float))), 1.)
    crps_center = float(np.mean(np.asarray(y_tr, float)))
    fit_kwargs = {}
    nb_log_mean = None
    if crps:
        model.set_params(objective=GaussianCRPSObjective(crps_scale))
        fit_kwargs["init_score"] = np.full(len(y_tr), crps_center)
    if nb:
        nb_r = estimate_nb_dispersion(np.asarray(y_tr, float))
        nb_log_mean = float(np.log(max(float(np.mean(np.asarray(y_tr, float))),
                                        1e-6)))
        model.set_params(objective=NegativeBinomialObjective(nb_r))
        fit_kwargs["init_score"] = np.full(len(y_tr), nb_log_mean)
    if fixed_rounds is not None:
        # OOF uses the same fixed-round refit policy as production. The
        # validation frame remains prediction-only; it must not select the
        # model complexity for the fold being scored.
        model.set_params(n_estimators=int(fixed_rounds))
        model.fit(tr_frame, y_tr, callbacks=[log_evaluation(period=0)],
                  **fit_kwargs)
        best = int(fixed_rounds)
    else:
        model.set_params(n_estimators=MAX_ROUNDS)
        model.fit(
            tr_frame, y_tr,
            eval_set=[(va_frame, y_va)],
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                       log_evaluation(period=0)],
            **fit_kwargs,
        )
        best = int(model.best_iteration_ or MAX_ROUNDS)
    lam = np.clip(model.predict(va_frame, num_iteration=best), 1e-6, None)
    if crps:
        lam = np.clip(lam + crps_center, 1e-6, None)
    if nb:
        assert nb_log_mean is not None
        lo, hi = NB_RAW_CLIP
        lam = np.clip(np.exp(np.clip(lam + nb_log_mean, lo, hi)), 1e-6, None)
    return model, lam, best


def pearson_poisson_adequacy(y: np.ndarray, mu: np.ndarray) -> float:
    """Pearson chi-square / df. ~=1 -> Poisson variance is adequate; >1 means
    over-dispersion the Gaussian variance layer learns (MLB run_engine fit probe)."""
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(mu, float), 1e-9, None)
    ok = np.isfinite(y) & np.isfinite(mu)
    y, mu = y[ok], mu[ok]
    if len(y) < 2:
        return float("nan")
    return float(((y - mu) ** 2 / mu).sum() / len(y))


def _poisson_deviance_mean(y: np.ndarray, mu: np.ndarray) -> float:
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(mu, float), 1e-9, None)
    ok = np.isfinite(y) & np.isfinite(mu)
    y, mu = y[ok], mu[ok]
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return float(2.0 * np.mean(term - (y - mu)))


def run_line_fit_check(oof: pd.DataFrame) -> dict[str, Any]:
    """Pooled fit diagnostics for the run line (NBA/NHL name for the probe
    MLB already publishes through ``run_oof``'s summary).

    Reads the mu columns NBA/NHL carry (``mu_h``/``mu_a``) or MLB's own
    ``home_expected_runs``/``away_expected_runs`` aliases, so the same call
    works on any sport's OOF frame: Poisson-adequacy per side plus deviance/
    RMSE of the model's means against the constant train-mean baseline.
    """
    out: dict[str, Any] = {}
    for side, y_col in (("home", "home_score"), ("away", "away_score")):
        mu_col = "mu_h" if side == "home" else "mu_a"
        alt = f"{side}_expected_runs"
        if mu_col not in oof.columns and alt in oof.columns:
            mu_col = alt
        y = oof[y_col].to_numpy(float)
        mu = oof[mu_col].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(mu)
        y, mu = y[ok], mu[ok]
        base = float(y.mean()) if len(y) else float("nan")
        out[side] = {
            "pearson": round(pearson_poisson_adequacy(y, mu), 4),
            "deviance_model": round(_poisson_deviance_mean(y, mu), 5),
            "deviance_baseline": round(_poisson_deviance_mean(
                y, np.full(len(y), base)), 5),
            "rmse_model": round(float(np.sqrt(np.mean((y - mu) ** 2))), 4),
            "rmse_baseline": round(float(np.sqrt(np.mean((y - base) ** 2))), 4),
            "n": int(len(y)),
        }
    return out


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
    clearly >1 indicates residual variance above the score mean."""
    y = np.asarray(y, dtype=float)
    lam = np.clip(np.asarray(lam, dtype=float), 1e-10, None)
    chi2 = float(((y - lam) ** 2 / lam).sum())
    df = max(int(len(y)) - 1, 1)
    return chi2 / df


def run_oof(games: pd.DataFrame,
            retrain_cadence_days: int = RETRAIN_CADENCE_DAYS,
            min_val_games: int = MIN_VAL_FOLD_GAMES,
            min_train_days: int = 30,
            run_features: Optional[list[str]] = None,
            dropped: Optional[list[str]] = None,
            decided_snapshot: Optional[pd.DataFrame] = None,
            nb: bool = False,
            ) -> dict[str, Any]:
    """Walk-forward OOF for both side models on the moneyline pipeline's folds.

    The 30-day warm-up default matches master_pipeline's moneyline training
    contract; changing it changes the shared OOF population.

    ``run_features`` / ``dropped`` are optional explicit overrides for controlled
    experiments; when omitted, production resolves the active moneyline list.
    ``decided_snapshot``: pre-computed decided frame (from frames.get_decided_frame)
    captured ONCE after official results, before slate merge.  When provided,
    skips re-derivation from the caller's ``games`` object so the OOF folds
    are built on the SAME canonical frame training used — preventing the
    fold-signature desync when the pipeline frame is mutated by slate
    concatenation, weather, or dome refinement.
    ``nb`` (default False): trial-only Negative-Binomial challenger
    (2026-10-17) — NB2 log-link per side with train-fold dispersion.
    Returns rows (one per decided game), per-side metrics, baseline metrics,
    and the dispersion probe."""
    from training import canonical_walk_forward_splits

    source = (decided_snapshot.copy() if decided_snapshot is not None
              else games)
    games, folds = canonical_walk_forward_splits(
        source, retrain_cadence_days=retrain_cadence_days,
        max_eval_folds=0, min_train_days=min_train_days,
        min_val_games=min_val_games,
    )
    logger.info("Run engine: %d walk-forward folds, %d decided games",
                len(folds), len(games))

    frames = {
        side: build_side_frame(games, side, run_features=run_features,
                               dropped=dropped,
                               strict_feature_parity=(run_features is None
                                                      and dropped is None))
        for side in ("home", "away")}
    params = dict(RUN_LGBM_PARAMS)

    out_rows: list[dict] = []
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
                    "home_team": va["home_team"].to_numpy() if "home_team" in va.columns else np.array([pd.NA] * len(va)),
                    "away_team": va["away_team"].to_numpy() if "away_team" in va.columns else np.array([pd.NA] * len(va)),
                    "game_id": (va["game_id"] if "game_id" in va.columns else
                                _d + "_" + va["away_team"] + "@" + va["home_team"]) if
                    ("game_id" in va.columns or {"home_team", "away_team"}.issubset(va.columns))
                    else pd.Series([""] * len(va), index=va.index)}
        for side, target in (("home", "home_score"), ("away", "away_score")):
            _, cols_all = frames[side]
            tr_frame = tr.reindex(columns=cols_all).astype(float)
            va_frame = va.reindex(columns=cols_all).astype(float)
            # Team-ID categoricals (2026-09-23 ADOPT): attach AFTER the float
            # reindex (which would cast a category back to float), resolved
            # from each fold slice's own team abbreviations.
            _apply_categorical_ids(tr_frame, rows=tr)
            _apply_categorical_ids(va_frame, rows=va)
            y_tr = tr[target].to_numpy(dtype=float)
            y_va = va[target].to_numpy(dtype=float)
            _, lam, _ = _fit_side_model(
                params, tr_frame, y_tr, va_frame, y_va,
                fixed_rounds=RUN_FIXED_FIT_ROUNDS, nb=nb,
            )
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
    from training import active_moneyline_feature_cols
    summary: dict[str, Any] = {"n_folds": len(folds), "n_games": len(oof),
                               "feature_contract": {
                                   "mode": "strict_active_moneyline",
                                   "n_features": len(active_moneyline_feature_cols()),
                                   "feature_cols": list(active_moneyline_feature_cols()),
                                   "tree_categorical_cols": (
                                       list(RUN_TREE_CATEGORICAL_COLS)
                                       if RUN_WITH_TEAM_IDS else []),
                               },
                               "postseason_policy": "include"}
    # OOF and production share one explicit model-complexity policy. This
    # prevents the validation frame from selecting a different fit state than
    # the final all-data slate refit and keeps home/away symmetric.
    summary["final_fit_rounds"] = {
        "home": RUN_FIXED_FIT_ROUNDS,
        "away": RUN_FIXED_FIT_ROUNDS,
    }
    summary["final_fit_rounds_note"] = (
        f"fixed {RUN_FIXED_FIT_ROUNDS}-round policy shared by OOF and production "
        "refits; the validation frame never selects fold model complexity")
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


# ---------------------------------------------------------------------------
# Score ensemble + level-2 CDF mixture (NBA structural port).
#
# STRUCTURALLY ADOPTED 2026-10-23 (owner-directed NBA alignment):
# run_engine_daily serves walk_forward_oof_ensemble (earned
# Poisson-deviance blends, full-view elastic net — DIFFS_ONLY=False) plus
# fit_final_ensemble/predict_ensemble_slate for the slate. Trial verdict
# (7,039 games: B z=+1.38 favors the single-LGBM incumbent, C identical
# to B) did NOT clear the 2.5 gate; this is structural parity with
# NBA-C, not a gate promotion. run_oof (single LightGBM) is retained
# for the margin-round derivation and diagnostics only:
#   * ScoreEnsembleRegressor: per-side xgboost / lightgbm / elasticnet
#     regressors on the SAME strict-active-moneyline side view the run
#     engine uses (build_side_frame + _apply_categorical_ids for the tree
#     legs; train-median-imputed + standardized numeric view for the
#     elastic-net leg), predicting home_score / away_score.
#   * walk_forward_oof_ensemble: causal expanding OOF on the shared
#     canonical_walk_forward_splits geometry; fold k blends with weights
#     earned on folds strictly before k (pooled home+away Poisson
#     deviance, plain simplex). Only regular-season rows of
#     non-provisional folds grade the blender — the moneyline grading
#     policy — postseason/provisional rows are scored but never evidence.
#   * Level-2 mixture: each member's mus price their OWN CDF through an
#     origin's fitted maps, and the served predictive CDF is the per-row
#     mixture of the member CDFs with weights earned on prior-fold OOF
#     CRPS (per kind). Column mapping is MLB's home_expected_runs /
#     away_expected_runs (not NBA's mu_h / mu_a).
# ---------------------------------------------------------------------------

#: MLB column mapping for the ensemble OOF (NBA: mu_h / mu_a).
_MLB_MU_H = "home_expected_runs"
_MLB_MU_A = "away_expected_runs"


def _make_score_member(name: str):
    """Construct one score-regression member."""
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**SCORE_XGBOOST_PARAMS)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        params = dict(RUN_LGBM_PARAMS)
        params["objective"] = "poisson"
        return LGBMRegressor(**params)
    if name == "elasticnet":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(**SCORE_ELASTICNET_PARAMS)
    raise KeyError(f"unknown score ensemble member {name!r}")


def _log_blend_mus(P: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Geometric blend with per-row NaN-member skipping.

    Log-space positivity-preserving blend for score means, renormalized
    over the finite members of each row (NBA port).
    """
    P = np.asarray(P, dtype=float)
    W = np.asarray(weights, dtype=float)
    if P.ndim != 2 or P.shape[1] != len(W):
        return np.full(len(P), np.nan)
    mask = np.isfinite(P) & (P > 0)
    active = (W > 0)[None, :] & mask
    logs = np.log(np.where(mask, P, 1.0))
    weighted = np.where(active, W[None, :], 0.0)
    denom = weighted.sum(axis=1)
    z = np.divide((logs * weighted).sum(axis=1), denom,
                  out=np.full(len(P), np.nan), where=denom > 0)
    return np.where(denom > 0, np.exp(z), np.nan)


def compute_score_weights(
    oof_members: dict[str, list[float]],
    y_oof,
) -> dict[str, float]:
    """Earn a deterministic simplex blend by pooled Poisson deviance.

    The objective is evaluated in log space with SLSQP over pooled
    home+away OOF pairs. A member missing on a row is skipped there; the
    remaining finite members are renormalized for that row (NBA port).
    """
    y = np.asarray(y_oof, dtype=float)
    if y.size == 0:
        return {}
    labeled = np.isfinite(y) & (y >= 0)
    if labeled.sum() < 2:
        return {}
    arrays: dict[str, np.ndarray] = {}
    for name, values in oof_members.items():
        if values is None:
            continue
        p = np.asarray(values, dtype=float)
        if p.ndim != 1 or len(p) != len(y):
            continue
        arrays[name] = np.where(np.isfinite(p) & (p > 0), p, np.nan)
    names = sorted(name for name, p in arrays.items()
                   if np.isfinite(p[labeled]).sum() >= 2)
    if not names:
        return {}
    yy = y[labeled]
    P = np.column_stack([arrays[name][labeled] for name in names])
    finite_any = np.isfinite(P).any(axis=1)
    if finite_any.sum() < 2:
        return {}
    P = P[finite_any]
    yy = yy[finite_any]
    Z = np.log(np.clip(P, 1e-6, None))

    def blend(w: np.ndarray) -> np.ndarray:
        active = np.isfinite(P) & (w[None, :] > 0)
        weights = np.where(active, w[None, :], 0.0)
        denom = weights.sum(axis=1)
        z = np.divide((np.where(active, Z, 0.0) * weights).sum(axis=1),
                      denom, out=np.full(len(P), np.nan), where=denom > 0)
        return np.exp(z)

    def objective(w: np.ndarray) -> float:
        return poisson_deviance(yy, blend(w))

    w0 = np.full(len(names), 1.0 / len(names))
    if len(names) == 1:
        return {names[0]: 1.0}
    solo = {name: poisson_deviance(yy, P[:, i])
            for i, name in enumerate(names)}
    best = min(solo, key=solo.get)

    from scipy.optimize import minimize
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(names),
        constraints=({"type": "eq",
                      "fun": lambda w: float(w.sum() - 1.0)},),
        options={"maxiter": 300, "ftol": 1e-9},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return {name: float(name == best) for name in names}
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    weights = weights / weights.sum() if weights.sum() else w0
    if solo[best] < objective(weights) - 1e-12:
        return {name: float(name == best) for name in names}
    rounded = {name: round(float(value), 4)
               for name, value in zip(names, weights)}
    drift = round(1.0 - sum(rounded.values()), 4)
    if drift:
        top = max(names, key=rounded.get)
        rounded[top] = round(rounded[top] + drift, 4)
    return rounded


class _ScoreStandardizer:
    """Train-median imputation + standardization for the elastic-net leg.

    Fit strictly on the training fold; the same fitted transform applies
    to validation and slate frames. Tree legs consume NaN natively.
    """

    def __init__(self) -> None:
        self.medians: pd.Series | None = None
        self.means: pd.Series | None = None
        self.stds: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "_ScoreStandardizer":
        self.medians = X.median(numeric_only=True).fillna(0.0)
        self.means = X.mean(numeric_only=True).fillna(0.0)
        self.stds = X.std(numeric_only=True).replace(0.0, 1.0).fillna(1.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Xf = X.astype(float).fillna(self.medians)
        return ((Xf - self.means) / self.stds).to_numpy(dtype=np.float64)


def _check_score_xgb_categories(model, frame: pd.DataFrame) -> None:
    """XGBoost team-ID columns must stay categorical."""
    booster = model.get_booster()
    types = dict(zip(booster.feature_names, booster.feature_types))
    if any(types.get(c) != "c" for c in RUN_TREE_CATEGORICAL_COLS
           if c in frame):
        raise RuntimeError("XGBoost team columns must remain categorical")


def _score_cat_vocab(games: pd.DataFrame) -> list:
    from training import _add_team_ids, UNK_TEAM_ID
    ids = _add_team_ids(games)
    vocab = set()
    for c in RUN_TREE_CATEGORICAL_COLS:
        if c in ids:
            vocab.update(pd.to_numeric(ids[c], errors="coerce").dropna()
                         .astype(int).tolist())
    vocab.add(int(UNK_TEAM_ID))
    return sorted(vocab)

def _cast_score_categories(frame: pd.DataFrame, vocab: list) -> None:
    from training import UNK_TEAM_ID
    for c in RUN_TREE_CATEGORICAL_COLS:
        if c in frame:
            v = pd.to_numeric(frame[c], errors="coerce")
            v = v.where(v.isin(set(vocab)), int(UNK_TEAM_ID)).astype(int)
            frame[c] = pd.Categorical(v, categories=list(vocab))


class ScoreEnsembleRegressor:
    """Per-side score-mean ensemble; blend weights set from prior OOF."""

    SIDES = ("home", "away")
    TARGETS = {"home": "home_score", "away": "away_score"}

    def __init__(self, cat_vocab: list | None = None) -> None:
        self.members: dict[str, dict[str, tuple[Any, Any]]] = {}
        self.columns: dict[str, list[str]] = {}
        self.scalers: dict[str, _ScoreStandardizer] = {}
        self.weights: dict[str, float] = dict(SCORE_ENSEMBLE_WEIGHTS)
        self.cat_vocab = list(cat_vocab) if cat_vocab is not None else None
        # Diffs-only challenger view (trial-only): the EN's own column
        # list + standardizer per side, so the tree legs' shared frame is
        # never narrowed. Empty unless the flag is on at fit time.
        self.en_columns: dict[str, list[str]] = {}
        self.en_scalers: dict[str, _ScoreStandardizer] = {}
        # Fit-time EN view per side (True = diffs-only). Predict reuses
        # the training geometry even if the trial flag moves between fit
        # and predict — probed 2026-10-17: a 74-wide model fed a 41-wide
        # frame raises in sklearn instead of predicting.
        self.en_active: dict[str, bool] = {}

    def set_weights(self, weights: dict[str, float]) -> "ScoreEnsembleRegressor":
        self.weights = dict(weights)
        return self

    def _matrices(self, games: pd.DataFrame, side: str,
                  fit: bool = False,
                  for_elasticnet: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
        """Tree-frame (categoricals attached) + standardized numeric array.

        With SCORE_EN_ELASTICNET_DIFFS_ONLY and for_elasticnet, the numeric
        array is the side frame minus the raw per-side levels (the NBA arm-B
        view); the tree frame is always the full frame. The view freezes at
        fit time (``en_active``) so predict reuses the training geometry
        even if the trial flag moves between fit and predict.
        """
        frame, _ = build_side_frame(games, side)
        if side not in self.columns:
            self.columns[side] = list(frame.columns)
        frame = frame.reindex(columns=self.columns[side])
        tree_frame = frame.copy()
        _apply_categorical_ids(tree_frame, rows=games)
        if self.cat_vocab is not None:
            _cast_score_categories(tree_frame, self.cat_vocab)
        if fit or side not in self.scalers:
            self.scalers[side] = _ScoreStandardizer().fit(frame)
        # Read through the module (not the from-import binding) so trials
        # and tests can flip the flag without reimporting. At predict time
        # the recorded fit-time view wins over the ambient flag.
        if fit and for_elasticnet:
            self.en_active[side] = bool(
                config_mod.SCORE_EN_ELASTICNET_DIFFS_ONLY)
        if for_elasticnet and not fit and side in self.en_active:
            en_view = self.en_active[side]
        else:
            en_view = bool(for_elasticnet and
                           config_mod.SCORE_EN_ELASTICNET_DIFFS_ONLY)
        if en_view:
            from training import RAW_PER_SIDE_COLS
            raw = set(RAW_PER_SIDE_COLS)
            if side not in self.en_columns:
                self.en_columns[side] = [
                    c for c in self.columns[side] if c not in raw]
            en_frame = frame.reindex(columns=self.en_columns[side])
            if fit or side not in self.en_scalers:
                self.en_scalers[side] = _ScoreStandardizer().fit(en_frame)
            return tree_frame, self.en_scalers[side].transform(en_frame)
        return tree_frame, self.scalers[side].transform(frame)

    def fit(self, games: pd.DataFrame) -> "ScoreEnsembleRegressor":
        for side in self.SIDES:
            y = pd.to_numeric(games[self.TARGETS[side]], errors="coerce")
            valid = y.notna() & (y >= 0)
            if valid.sum() < 2:
                raise ValueError("score ensemble requires settled scores")
            train = games.loc[valid]
            y = y.loc[valid].to_numpy(dtype=float)
            tree_frame, num = self._matrices(train, side, fit=True,
                                              for_elasticnet=True)
            fitted: dict[str, tuple[Any, Any]] = {}
            for name in SCORE_ENSEMBLE_MEMBERS:
                model = _make_score_member(name)
                if name == "elasticnet":
                    model.fit(num, y)
                    # Second slot records the standardizer that produced
                    # num: the EN's own scaler under the diffs-only flag,
                    # else the shared side scaler (identical object then).
                    en_scaler = self.en_scalers.get(side, self.scalers[side])
                    fitted[name] = (model, en_scaler)
                else:
                    if name == "lightgbm":
                        model.set_params(n_estimators=int(RUN_FIXED_FIT_ROUNDS))
                    model.fit(tree_frame, y)
                    if name == "xgboost":
                        _check_score_xgb_categories(model, tree_frame)
                    fitted[name] = (model, None)
            self.members[side] = fitted
        return self

    def predict_member(self, games: pd.DataFrame, name: str,
                       side: str) -> np.ndarray:
        model, _ = self.members[side][name]
        tree_frame, num = self._matrices(
            games, side, for_elasticnet=(name == "elasticnet"))
        X = num if name == "elasticnet" else tree_frame
        return np.clip(np.asarray(model.predict(X), dtype=float), 1e-6, None)

    def predict(self, games: pd.DataFrame,
                weights: dict[str, float] | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
        if not self.members:
            raise RuntimeError("score ensemble is not fitted")
        weights = self.weights if weights is None else weights
        out = []
        for side in self.SIDES:
            cols = []
            for name in SCORE_ENSEMBLE_MEMBERS:
                try:
                    cols.append(self.predict_member(games, name, side))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("score ensemble member %s/%s failed: %s",
                                   name, side, exc)
                    cols.append(np.full(len(games), np.nan))
            P = np.column_stack(cols)
            w = np.asarray([weights.get(name, 0.0)
                            for name in SCORE_ENSEMBLE_MEMBERS], dtype=float)
            out.append(_log_blend_mus(P, w))
        return out[0], out[1]


def walk_forward_oof_ensemble(
    games: pd.DataFrame,
    retrain_cadence_days: int = RETRAIN_CADENCE_DAYS,
    min_val_games: int = MIN_VAL_FOLD_GAMES,
    min_train_days: int = 30,
    progress_every: int = 10,
) -> dict[str, Any]:
    """Causal expanding score-ensemble OOF with prior-fold deviance weights.

    Same folds as run_oof (canonical_walk_forward_splits on the decided
    frame). Fold k blends with weights earned on folds strictly before k.
    Only regular-season rows of non-provisional folds grade the blender.
    """
    from training import canonical_walk_forward_splits, postseason_flag

    decided, splits = canonical_walk_forward_splits(
        games, retrain_cadence_days=retrain_cadence_days,
        max_eval_folds=0, min_train_days=min_train_days,
        min_val_games=min_val_games,
    )
    vocab = _score_cat_vocab(decided)
    parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    prior_weights = dict(SCORE_ENSEMBLE_WEIGHTS)
    pooled: dict[str, list[float]] = {name: []
                                      for name in SCORE_ENSEMBLE_MEMBERS}
    pooled_y: list[float] = []
    n_splits = len(splits)

    for si, split in enumerate(splits):
        tr, va = split["train_games"], split["val_games"]
        fold_idx = split["fold_idx"]
        fold_weights = dict(prior_weights)
        if len(tr) < 2 or len(va) < 1:
            continue
        try:
            ens = ScoreEnsembleRegressor(cat_vocab=vocab).fit(tr)
        except Exception as exc:  # noqa: BLE001
            logger.warning("score-ensemble fold %s failed: %s", fold_idx, exc)
            continue
        member_mu: dict[str, dict[str, np.ndarray]] = {}
        for name in SCORE_ENSEMBLE_MEMBERS:
            member_mu[name] = {}
            for side in ScoreEnsembleRegressor.SIDES:
                try:
                    member_mu[name][side] = ens.predict_member(va, name, side)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("score-ensemble fold %s member %s/%s failed: %s",
                                   fold_idx, name, side, exc)
                    member_mu[name][side] = np.full(len(va), np.nan)
        w = np.asarray([prior_weights.get(name, 0.0)
                        for name in SCORE_ENSEMBLE_MEMBERS], dtype=float)
        mu_h = _log_blend_mus(
            np.column_stack([member_mu[name]["home"]
                             for name in SCORE_ENSEMBLE_MEMBERS]), w)
        mu_a = _log_blend_mus(
            np.column_stack([member_mu[name]["away"]
                             for name in SCORE_ENSEMBLE_MEMBERS]), w)
        y_h = pd.to_numeric(va["home_score"], errors="coerce")
        y_a = pd.to_numeric(va["away_score"], errors="coerce")
        game_pk = (pd.to_numeric(va["game_pk"], errors="coerce")
                   if "game_pk" in va.columns
                   else pd.Series(np.arange(len(va)), index=va.index))
        # Identity parity with run_oof (2026-10-25 off-day: the ensemble
        # OOF dropped home_team/away_team/game_id, so derive_markets_v3
        # priced identity-less markets and persist_markets refused the
        # artifact). Mirror rec_base so markets carry the join contract.
        _d = pd.to_datetime(va["game_date"]).dt.strftime("%Y%m%d")
        if "game_id" in va.columns:
            _gid = va["game_id"]
        elif {"home_team", "away_team"}.issubset(va.columns):
            _gid = _d + "_" + va["away_team"].astype(str) + "@" + va["home_team"].astype(str)
        else:
            _gid = pd.Series([""] * len(va), index=va.index)
        row = pd.DataFrame({
            "game_pk": game_pk.to_numpy(),
            "game_date": pd.to_datetime(va["game_date"]).to_numpy(),
            "fold_idx": fold_idx,
            "home_team": va["home_team"].to_numpy() if "home_team" in va.columns else np.array([pd.NA] * len(va)),
            "away_team": va["away_team"].to_numpy() if "away_team" in va.columns else np.array([pd.NA] * len(va)),
            "game_id": _gid.to_numpy() if hasattr(_gid, "to_numpy") else np.asarray(_gid),
            _MLB_MU_H: mu_h, _MLB_MU_A: mu_a,
            "home_score": y_h.to_numpy(float),
            "away_score": y_a.to_numpy(float),
        })
        for name in SCORE_ENSEMBLE_MEMBERS:
            row[f"{_MLB_MU_H}_{name}"] = member_mu[name]["home"]
            row[f"{_MLB_MU_A}_{name}"] = member_mu[name]["away"]
            row[f"weight_{name}"] = fold_weights.get(name, 0.0)
        provisional = bool(split.get("provisional", False))
        if provisional:
            grades = np.zeros(len(va), dtype=bool)
        elif "game_type" in va.columns:
            grades = ~postseason_flag(va["game_type"].to_numpy())
        else:
            grades = np.ones(len(va), dtype=bool)
        row["grades_blend"] = grades
        for name in SCORE_ENSEMBLE_MEMBERS:
            pooled[name].extend(row[f"{_MLB_MU_H}_{name}"].to_numpy(float)[grades].tolist())
            pooled[name].extend(row[f"{_MLB_MU_A}_{name}"].to_numpy(float)[grades].tolist())
        pooled_y.extend(row.home_score.to_numpy(float)[grades].tolist())
        pooled_y.extend(row.away_score.to_numpy(float)[grades].tolist())
        earned = compute_score_weights(pooled, np.asarray(pooled_y, dtype=float))
        if earned:
            prior_weights = earned
        fold_rows.append({"fold_idx": fold_idx,
                          "val_start": str(pd.to_datetime(split["val_start"]).date()),
                          "val_end": str(pd.to_datetime(split["val_end"]).date()),
                          "n_train": int(len(tr)), "n_val": int(len(va)),
                          "n_grading": int(grades.sum()),
                          "weights": fold_weights})
        parts.append(row)
        if progress_every and (si + 1) % progress_every == 0:
            logger.info("score-ensemble OOF fold %d/%d", si + 1, n_splits)

    oof = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(oof):
        oof = oof.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    return {"oof": oof, "member_weights": prior_weights,
            "fold_table": pd.DataFrame(fold_rows)}


def fit_final_ensemble(games: pd.DataFrame) -> dict:
    """Fit all score members on every decided game for production serving.

    Blend weights are NOT fitted here — the caller serves the walk's
    member_weights (OOF evidence), never an in-sample optimum.
    """
    from training import canonical_walk_forward_splits
    decided, _ = canonical_walk_forward_splits(games)
    base = decided if len(decided) else games
    ens = ScoreEnsembleRegressor(cat_vocab=_score_cat_vocab(base)).fit(base)
    out = dict(ens.members)
    out["cat_vocab"] = list(ens.cat_vocab or [])
    return out


def predict_ensemble_slate(models: dict, slate_games: pd.DataFrame,
                           weights: dict[str, float]
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Blend a pending slate's per-side means with deployed OOF weights."""
    mus = []
    for side in ScoreEnsembleRegressor.SIDES:
        frame, _ = build_side_frame(slate_games, side)
        tree_frame = frame.copy()
        _apply_categorical_ids(tree_frame, rows=slate_games)
        if models.get("cat_vocab"):
            _cast_score_categories(tree_frame, models["cat_vocab"])
        cols = []
        names = []
        for name in SCORE_ENSEMBLE_MEMBERS:
            entry = (models.get(side) or {}).get(name)
            if not entry:
                continue
            model, pre = entry
            try:
                if name == "elasticnet":
                    X = pre.transform(frame)
                else:
                    X = tree_frame
                cols.append((name, np.clip(np.asarray(model.predict(X),
                                                     dtype=float), 1e-6, None)))
                names.append(name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("serving score member %s/%s failed: %s",
                               name, side, exc)
        if not cols:
            mus.append(np.full(len(slate_games), np.nan))
            continue
        P = np.column_stack([c for _, c in cols])
        w = np.asarray([weights.get(name, 0.0) for name in names],
                       dtype=float)
        mus.append(_log_blend_mus(P, w))
    return mus[0], mus[1]


# ---------------------------------------------------------------------------
# Level-2 mixture (Option B): each member's mus price their OWN CDF through
# an origin's fitted maps, and the served predictive CDF is the per-row
# mixture of the member CDFs with weights earned on prior-fold OOF CRPS.
# ---------------------------------------------------------------------------


def _pmf_from_cdf(cdf: np.ndarray) -> np.ndarray:
    cdf = np.asarray(cdf, dtype=float)
    return np.diff(np.column_stack([np.zeros(len(cdf)), cdf]), axis=1)


def _align_pmfs_to_union(pmfs: list[np.ndarray],
                         supports: list[np.ndarray]):
    """Zero-pad member PMFs onto their union integer lattice."""
    lo = min(int(s[0]) for s in supports)
    hi = max(int(s[-1]) for s in supports)
    union = np.arange(lo, hi + 1)
    out = []
    for pmf, sup in zip(pmfs, supports):
        wide = np.zeros((len(pmf), len(union)))
        j0 = int(sup[0]) - lo
        wide[:, j0:j0 + len(sup)] = pmf
        out.append(wide)
    return out, union


def mixture_cdf(member_cdfs: dict[str, tuple[np.ndarray, np.ndarray]],
                weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Per-row CDF mixture over available members, renormalized per row."""
    names = [n for n in member_cdfs if (weights.get(n, 0.0) or 0.0) > 0]
    if not names:
        raise ValueError("mixture needs at least one positively weighted member")
    n_rows = len(member_cdfs[names[0]][1])
    if any(len(member_cdfs[n][1]) != n_rows for n in names):
        raise ValueError("mixture members must share row order and count")
    pmfs, supports, avail = [], [], []
    for name in names:
        sup, cdf = member_cdfs[name]
        ok = np.isfinite(cdf).all(axis=1)
        avail.append(ok)
        pmfs.append(_pmf_from_cdf(np.where(ok[:, None], cdf, 0.0)))
        supports.append(np.asarray(sup))
    aligned, union = _align_pmfs_to_union(pmfs, supports)
    w = np.asarray([float(weights[n]) for n in names], dtype=float)
    pmf_mix = np.zeros((n_rows, len(union)))
    mass = np.zeros(n_rows)
    for wi, pmf, ok in zip(w, aligned, avail):
        pmf_mix[ok] += wi * pmf[ok]
        mass[ok] += wi
    none = mass <= 0
    pmf_mix[~none] /= mass[~none, None]
    cdf_mix = np.cumsum(pmf_mix, axis=1)
    cdf_mix = np.clip(cdf_mix, 0.0, 1.0)
    cdf_mix[none] = np.nan
    return union, cdf_mix


def compute_mixture_weights(blocks) -> dict[str, float]:
    """Earn a deterministic simplex blend by pooled CRPS (NBA port)."""
    per_member: dict[str, list] = {}
    for member_cdfs, y in blocks:
        y = np.asarray(y, dtype=float)
        for name, (support, cdf) in member_cdfs.items():
            cdf = np.asarray(cdf, dtype=float)
            if len(cdf) != len(y):
                raise ValueError("mixture evidence members must share row order")
            ok = np.isfinite(cdf).all(axis=1) & np.isfinite(y)
            per_member.setdefault(name, []).append(
                (cdf[ok], np.asarray(support), y[ok]))
    names = sorted(name for name, parts in per_member.items()
                   if sum(len(p[0]) for p in parts) >= 2)
    if not names:
        return {}
    pooled: dict[str, list] = {name: [] for name in names}
    for member_cdfs, y in blocks:
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(y)
        for name in names:
            _, cdf = member_cdfs.get(name, (None, None))
            if cdf is None:
                ok = np.zeros(len(y), dtype=bool)
                break
            cdf = np.asarray(cdf, dtype=float)
            if len(cdf) != len(y):
                raise ValueError("mixture evidence members must share row order")
            ok = ok & np.isfinite(cdf).all(axis=1)
        if ok.sum() < 1:
            continue
        for name in names:
            support, cdf = member_cdfs[name]
            pooled[name].append((np.asarray(cdf, dtype=float)[ok],
                                 np.asarray(support), y[ok]))
    total = sum(len(p[0]) for p in pooled[names[0]])
    if total < 2:
        return {}
    union = np.arange(
        min(int(p[1][0]) for parts in pooled.values() for p in parts),
        max(int(p[1][-1]) for parts in pooled.values() for p in parts) + 1)
    stacked, yy = [], []
    for name in names:
        parts = pooled[name]
        wide = np.zeros((sum(len(p[0]) for p in parts), len(union)))
        ys = []
        r = 0
        for cdf, support, y in parts:
            pmf = _pmf_from_cdf(cdf)
            j0 = int(support[0]) - int(union[0])
            wide[r:r + len(pmf), j0:j0 + len(support)] = pmf
            ys.append(y)
            r += len(pmf)
        stacked.append(np.cumsum(wide, axis=1))
        yy.append(np.concatenate(ys))
    P = np.stack(stacked)
    yy = yy[0]

    def objective(w: np.ndarray) -> float:
        w = np.clip(np.asarray(w, dtype=float), 0.0, None)
        total_w = w.sum()
        if total_w <= 0:
            return float(np.mean(family.crps(P.mean(axis=0), union, yy)))
        mix = np.tensordot(w / total_w, P, axes=([0], [0]))
        mix = np.clip(mix, 0.0, 1.0)
        return float(np.mean(family.crps(mix, union, yy)))

    w0 = np.full(len(names), 1.0 / len(names))
    if len(names) == 1:
        return {names[0]: 1.0}
    solo = {}
    for i, name in enumerate(names):
        try:
            solo[name] = float(np.mean(family.crps(P[i], union, yy)))
        except ValueError:
            solo[name] = float("inf")
    best = min(solo, key=solo.get)

    from scipy.optimize import minimize
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(names),
        constraints=({"type": "eq",
                      "fun": lambda w: float(w.sum() - 1.0)},),
        options={"maxiter": 300, "ftol": 1e-9},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return {name: float(name == best) for name in names}
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    weights = weights / weights.sum() if weights.sum() else w0
    if solo[best] < objective(weights) - 1e-12:
        return {name: float(name == best) for name in names}
    rounded = {name: round(float(value), 4)
               for name, value in zip(names, weights)}
    drift = round(1.0 - sum(rounded.values()), 4)
    if drift:
        top = max(names, key=rounded.get)
        rounded[top] = round(rounded[top] + drift, 4)
    return rounded


def price_member_cdfs(frame: pd.DataFrame, mus: dict[str, tuple[np.ndarray, np.ndarray]],
                      maps: dict) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Price each member's mus through fitted origin maps, per kind."""
    out: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for name, (mu_h, mu_a) in mus.items():
        mu_h = np.asarray(mu_h, dtype=float)
        mu_a = np.asarray(mu_a, dtype=float)
        ok = (np.isfinite(mu_h) & np.isfinite(mu_a)
              & (mu_h > 0) & (mu_a > 0))
        if not ok.any():
            continue
        part = frame.iloc[np.flatnonzero(ok)].copy()
        part[_MLB_MU_H] = mu_h[ok]
        part[_MLB_MU_A] = mu_a[ok]
        cdfs = _distribution_cdfs(part, maps, sampled=False)
        for kind, (support, cdf) in cdfs.items():
            full = np.full((len(frame), len(support)), np.nan)
            full[np.flatnonzero(ok)] = cdf
            out.setdefault(kind, {})[name] = (np.asarray(support), full)
    return out


def walk_forward_mixture(oof: pd.DataFrame, bundle: dict,
                         kinds: tuple[str, ...] = ("total", "margin"),
                         progress_every: int = 10) -> dict:
    """Causal per-origin CDF mixture over ensemble member mus (NBA port)."""
    members = list(SCORE_ENSEMBLE_MEMBERS)
    have = [m for m in members
            if f"{_MLB_MU_H}_{m}" in oof.columns and f"{_MLB_MU_A}_{m}" in oof.columns]
    if not have:
        raise ValueError("mixture walk needs per-member OOF mus columns")
    if _MLB_MU_H not in oof.columns or _MLB_MU_A not in oof.columns:
        raise ValueError("mixture walk needs blended-single OOF mus columns")
    frame = oof.reset_index(drop=True)
    grades_all = (frame["grades_blend"].to_numpy(bool)
                  if "grades_blend" in frame.columns
                  else np.ones(len(frame), dtype=bool))
    pooled: dict[str, list] = {k: [] for k in kinds}
    pooled_n = {k: 0 for k in kinds}
    prior_weights = {k: {m: 1.0 / len(have) for m in have} for k in kinds}
    parts: list[pd.DataFrame] = []
    weight_rows: list[dict] = []
    origins = bundle.get("origins") or []
    blocks = _distribution_blocks(frame)
    if len(blocks) != len(origins):
        raise ValueError("mixture walk needs one bundle origin per OOF block")
    for bi, ((_, _, mask), origin) in enumerate(zip(blocks, origins)):
        idx = np.flatnonzero(mask)
        part = frame.iloc[idx]
        maps = origin["maps"]
        mus = {m: (part[f"{_MLB_MU_H}_{m}"].to_numpy(float),
                   part[f"{_MLB_MU_A}_{m}"].to_numpy(float)) for m in have}
        single = {}
        ok_single = (np.isfinite(part[_MLB_MU_H].to_numpy(float))
                     & np.isfinite(part[_MLB_MU_A].to_numpy(float)))
        part_y_total = (part.home_score + part.away_score).to_numpy(float) \
            if "home_score" in part.columns else np.full(len(part), np.nan)
        part_y_margin = (part.home_score - part.away_score).to_numpy(float) \
            if "home_score" in part.columns else np.full(len(part), np.nan)
        if ok_single.any():
            single = _distribution_cdfs(part.iloc[np.flatnonzero(ok_single)],
                                        maps, sampled=False)
        priced = price_member_cdfs(part.reset_index(drop=True), mus, maps)
        row = pd.DataFrame({"game_pk": part.game_pk.to_numpy(),
                            "fold_idx": (part.fold_idx.to_numpy()
                                         if "fold_idx" in part.columns else bi)})
        mix_weights = {k: dict(prior_weights[k]) for k in kinds}
        for kind in kinds:
            y = part_y_total if kind == "total" else part_y_margin
            row[f"crps_single_{kind}"] = np.nan
            row[f"crps_mix_{kind}"] = np.nan
            for name in have:
                row[f"weight_mix_{kind}_{name}"] = mix_weights[kind].get(name, 0.0)
            if kind in single:
                sup_s, cdf_s = single[kind]
                sub = np.flatnonzero(ok_single)
                keep = np.isfinite(y[sub])
                if keep.any():
                    full_s = np.full(len(part), np.nan)
                    full_s[sub[keep]] = lattice_crps(
                        cdf_s[keep], sup_s, y[sub[keep]])
                    row[f"crps_single_{kind}"] = full_s
            if kind not in priced:
                continue
            try:
                union, cdf_mix = mixture_cdf(priced[kind], prior_weights[kind])
            except ValueError:
                continue
            finite = np.isfinite(cdf_mix).all(axis=1) & np.isfinite(y)
            if finite.any():
                full_m = np.full(len(part), np.nan)
                full_m[np.flatnonzero(finite)] = lattice_crps(
                    cdf_mix[finite], union, y[finite])
                row[f"crps_mix_{kind}"] = full_m
            g = grades_all[idx] & np.isfinite(y)
            if g.any() and kind in priced:
                keep = np.flatnonzero(g)
                pooled[kind].append(
                    ({name: (sup, cdf[keep])
                      for name, (sup, cdf) in priced[kind].items()
                      if name in have}, y[g]))
            pooled_n[kind] += int(g.sum())
        for kind in kinds:
            if pooled_n[kind] < 2:
                continue
            earned = compute_mixture_weights(pooled[kind])
            if earned:
                prior_weights[kind] = earned
        weight_rows.append({"block": bi,
                            "used": {k: dict(mix_weights[k]) for k in kinds},
                            "next": {k: dict(prior_weights[k])
                                     for k in kinds}})
        parts.append(row)
        if progress_every and (bi + 1) % progress_every == 0:
            logger.info("mixture OOF block %d/%d", bi + 1, len(blocks))
    mix_oof = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    summary = {}
    for kind in kinds:
        s = mix_oof[f"crps_single_{kind}"].to_numpy(float)
        m = mix_oof[f"crps_mix_{kind}"].to_numpy(float)
        ok = np.isfinite(s) & np.isfinite(m)
        summary[kind] = {
            "n": int(ok.sum()),
            "crps_single": round(float(s[ok].mean()), 6) if ok.any() else None,
            "crps_mix": round(float(m[ok].mean()), 6) if ok.any() else None,
            "paired_gain": (round(float((s[ok] - m[ok]).mean()), 6)
                            if ok.any() else None),
            "final_weights": dict(prior_weights[kind]),
        }
    return {"oof": mix_oof, "weights": prior_weights,
            "weight_history": weight_rows, "summary": summary}


def predict_mixture_slate(models: dict, slate_games: pd.DataFrame,
                          serving_maps: dict,
                          weights_l2: dict[str, dict[str, float]],
                          kinds: tuple[str, ...] = ("total", "margin")
                          ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Blend a pending slate's per-member CDFs with deployed OOF weights."""
    mus = {}
    for side in ScoreEnsembleRegressor.SIDES:
        frame, _ = build_side_frame(slate_games, side)
        tree_frame = frame.copy()
        _apply_categorical_ids(tree_frame, rows=slate_games)
        if models.get("cat_vocab"):
            _cast_score_categories(tree_frame, models["cat_vocab"])
        for name in SCORE_ENSEMBLE_MEMBERS:
            entry = (models.get(side) or {}).get(name)
            if not entry:
                continue
            model, pre = entry
            try:
                X = pre.transform(frame) if name == "elasticnet" else tree_frame
                mus.setdefault(name, {})[side] = np.clip(
                    np.asarray(model.predict(X), dtype=float), 1e-6, None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("serving mixture member %s/%s failed: %s",
                               name, side, exc)
    per_member = {}
    for name, sides in mus.items():
        if set(sides) != {"home", "away"}:
            continue
        per_member[name] = (sides["home"], sides["away"])
    priced = price_member_cdfs(slate_games.reset_index(drop=True),
                               per_member, serving_maps)
    out = {}
    for kind in kinds:
        if kind not in priced:
            continue
        try:
            out[kind] = mixture_cdf(priced[kind], weights_l2.get(kind, {}))
        except ValueError:
            continue
    return out


# Persisted Phase-1 contract. fold_idx / game_id ride along internally
# (Phase 2's prequential calibration and agreement merge need them) but stay
# out of the CSV schema.
OOF_COLUMNS = ["game_pk", "game_date", "home_expected_runs",
               "away_expected_runs", "home_score", "away_score", "fold_idx"]


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


def _contract_key(value: Any) -> str:
    """Canonical artifact key for numeric game_pk and string game_id values."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return text


def build_artifact_contract(oof: pd.DataFrame,
                            markets: pd.DataFrame) -> dict[str, Any]:
    """Compare the persisted OOF score frame with market OOF rows.

    This contract is intentionally based on stable keys and the expected
    runs actually priced, not row order. A later frontend can reject an older or
    partially rewritten artifact without guessing from its filename.
    """
    left = oof.copy() if oof is not None else pd.DataFrame()
    right = markets.copy() if markets is not None else pd.DataFrame()
    if "kind" in right.columns:
        right = right[right["kind"].eq("oof")].copy()
    required = {"game_pk", "game_date", "home_expected_runs", "away_expected_runs"}
    missing_left = sorted(required.difference(left.columns))
    missing_right = sorted(required.difference(right.columns))
    if missing_left or missing_right:
        return {"schema": "run-engine-artifact/v1", "alignment_status": "invalid",
                "missing_oof_columns": missing_left,
                "missing_market_columns": missing_right}
    left["_key"] = left["game_pk"].map(_contract_key)
    right["_key"] = right["game_pk"].map(_contract_key)
    left = left[left["_key"].ne("")]
    right = right[right["_key"].ne("")]
    duplicate_oof_keys = int(left["_key"].duplicated().sum())
    duplicate_market_keys = int(right["_key"].duplicated().sum())
    lk, rk = set(left["_key"]), set(right["_key"])
    common = sorted(lk & rk)
    lmap = left.set_index("_key")
    rmap = right.set_index("_key")
    if common:
        # Duplicate keys make .loc[common] return MORE rows than common
        # (every duplicate entry) — the 2026-09-28 persist crash. De-duplicate
        # and position-align via an index intersection instead of a direct
        # label lookup; duplicate_* counts already reported above flag the
        # duplication (aligned requires both == 0, so the contract reads
        # misaligned — but the builder must never CRASH).
        lcommon = lmap.loc[[k for k in common if k in lmap.index]]
        rcommon = rmap.loc[[k for k in common if k in rmap.index]]
        lcommon = lcommon[~lcommon.index.duplicated(keep="first")]
        rcommon = rcommon[~rcommon.index.duplicated(keep="first")]
        shared = lcommon.index.intersection(rcommon.index)
        lcommon = lcommon.loc[shared]
        rcommon = rcommon.loc[shared]
        hdiff = (pd.to_numeric(lcommon["home_expected_runs"], errors="coerce")
                 - pd.to_numeric(rcommon["home_expected_runs"], errors="coerce")).abs()
        adiff = (pd.to_numeric(lcommon["away_expected_runs"], errors="coerce")
                 - pd.to_numeric(rcommon["away_expected_runs"], errors="coerce")).abs()
        max_home = float(hdiff.max()) if len(hdiff) else 0.0
        max_away = float(adiff.max()) if len(adiff) else 0.0
    else:
        max_home = max_away = float("inf")
    left_dates = (pd.to_datetime(lcommon["game_date"], errors="coerce")
                  if common and len(lcommon) else pd.Series(dtype="datetime64[ns]"))
    right_dates = (pd.to_datetime(rcommon["game_date"], errors="coerce")
                   if common and len(rcommon) else pd.Series(dtype="datetime64[ns]"))
    date_mismatch = int((left_dates.reset_index(drop=True).dt.normalize().to_numpy()
                         != right_dates.reset_index(drop=True).dt.normalize().to_numpy()).sum()) if common else 0
    aligned = bool(lk == rk and duplicate_oof_keys == 0
                   and duplicate_market_keys == 0 and date_mismatch == 0
                   and max_home <= 1e-8 and max_away <= 1e-8)
    dates = pd.to_datetime(left["game_date"], errors="coerce")
    canonical = left[["_key", "game_date", "home_expected_runs",
                      "away_expected_runs"]].sort_values("_key").to_csv(
                          index=False, lineterminator="\n")
    return {
        "schema": "run-engine-artifact/v1",
        "alignment_status": "aligned" if aligned else "misaligned",
        "oof_rows": int(len(left)), "market_oof_rows": int(len(right)),
        "missing_market_keys": int(len(lk - rk)),
        "missing_oof_keys": int(len(rk - lk)),
        "duplicate_oof_keys": duplicate_oof_keys,
        "duplicate_market_keys": duplicate_market_keys,
        "date_mismatch_rows": date_mismatch,
        "max_abs_home_expected_runs_diff": round(max_home, 8),
        "max_abs_away_expected_runs_diff": round(max_away, 8),
        "oof_date_min": (dates.min().strftime("%Y-%m-%d") if dates.notna().any() else None),
        "oof_date_max": (dates.max().strftime("%Y-%m-%d") if dates.notna().any() else None),
        "artifact_fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Challenger market derivation: calibrated split-skew Gaussian Monte Carlo.
from gaussian_family import MC_DRAWS, MC_SEED, MC_SE_TARGET
import gaussian_family as family
MARKET_SEED = MC_SEED
HOLDOUT_DAYS = 21
TOTAL_LINE = 8.5
RUN_LINE_MARGIN = 1.5
TOTAL_LINE_GRID = [round(6.5 + .5 * i, 1) for i in range(13)]
RUN_LINE_GRID = [.5, 1.5, 2.5, 3.5]
RUN_LINE_GRID_FULL = [1., 1.5, 2., 2.5, 3., 3.5, 4.]
RUN_LINE_GRID_WHOLE = [1., 2., 3., 4.]
# Diagnostic margin-push extension (2026-10-16): whole-run margins whose
# push mass ships for the Distribution tab only (+-5..10, push legs, no
# home/away/cover legs — the traded run-line grid is unchanged). Blowouts
# reach +-15 while the traded whole-margin grid stops at +-4; the tab
# plots to +-10.
MARGIN_PUSH_EXTENSION = [5., 6., 7., 8., 9., 10.]
TOTAL_REF_LINES = (7.5, 8.5, 9.5)
RUN_REF_LINES = (1.5, 2.5)


def derive_markets_v3(oof, moneyline_probs=None, n_draws=MC_DRAWS,
                      seed=MC_SEED, holdout_days=HOLDOUT_DAYS):
    """Causal split-skew Gaussian markets through calibrated Monte Carlo."""
    if holdout_days != HOLDOUT_DAYS:
        raise ValueError("challenger evaluation uses the fixed 21-day sealed tail")
    if "game_pk" in oof and oof.game_pk.isna().any():
        logger.warning("Dropping %d identity-less OOF rows", int(oof.game_pk.isna().sum()))
        oof = oof[oof.game_pk.notna()].reset_index(drop=True)
    markets, calibration = calibrate_market_frame(oof, n_draws=n_draws, seed=seed)
    markets["kind"] = "oof"
    markets["total_runs"] = markets.home_score + markets.away_score
    markets["agreement_conflict"] = False
    markets["ml_win_prob"] = np.nan
    if moneyline_probs is not None and len(moneyline_probs):
        key = "game_pk" if "game_pk" in moneyline_probs else "game_id"
        if key in markets:
            lookup = moneyline_probs.set_index(key).home_win_prob_model
            markets["ml_win_prob"] = markets[key].map(lookup)
            markets["agreement_conflict"] = (markets.p_home_win_derived - markets.ml_win_prob).abs() > AGREEMENT_FILTER_DELTA
    hold = calibration["holdout"]
    metrics = calibration["oof_metrics"]
    summary = {"family": family.FAMILY, "n_draws": calibration["monte_carlo"]["n_draws"],
               "holdout_cutoff": hold["cutoff"], "n_pre": hold["n_pre"], "n_holdout": hold["n_holdout"],
               "distribution_metrics": metrics, "mc_meta": calibration["monte_carlo"],
               "variance": calibration["maps"]["variance"], "holdout": hold,
               # Lattice fit disclosure (totals + margin fitted independently;
               # magnitude + gate verdict, no delta vectors).
               "lattice_summary": calibration.get("lattice_summary", {}),
               "calibration": {"method": calibration["method"], "calibrators": calibration},
               "line_grid": {"totals": TOTAL_LINE_GRID, "run_lines_full": [-m for m in RUN_LINE_GRID_FULL]}}
    # Per-line model card uses the same issued probabilities, pushes excluded.
    for kind, lines in (("total", TOTAL_REF_LINES), ("margin", RUN_REF_LINES)):
        y = (markets.home_score + markets.away_score if kind == "total" else markets.home_score - markets.away_score).to_numpy(float)
        for line in lines:
            suffix = str(float(line)).replace('.', '_')
            win_col = f"p_over_{suffix}" if kind == "total" else rl_col(line, "home")
            push_col = f"p_push_{suffix}" if kind == "total" else rl_col(line, "push")
            win, push = markets[win_col].to_numpy(float), markets[push_col].to_numpy(float)
            valid = (y != line) & (push < 1.)
            probability = np.clip(win[valid] / (1. - push[valid]), 1e-7, 1.-1e-7)
            target = y[valid] > line
            key = f"market_over_{str(line).replace('.', '_')}" if kind == "total" else f"market_home_cover_{str(line).replace('.', '_')}"
            bins = np.minimum((probability * 10).astype(int), 9)
            ece = sum(float((bins == b).mean()) * abs(float(probability[bins == b].mean() - target[bins == b].mean())) for b in np.unique(bins))
            summary[key] = {"n": int(valid.sum()), "engine_ece_raw": None, "engine_ece_calibrated": ece,
                            "engine_brier": float(np.mean((probability - target) ** 2)),
                            "engine_logloss": float(-np.mean(target*np.log(probability)+(~target)*np.log1p(-probability))),
                            "baseline_logloss": float(log_loss(target, np.full(len(target), np.clip(target.mean(), 1e-7, 1.-1e-7)), labels=[0, 1]))}
    summary["market_derived_moneyline"] = {"n": metrics["derived_moneyline"]["n"],
                                          "engine_ece_raw": None, "engine_ece_calibrated": None,
                                          "engine_brier": metrics["derived_moneyline"]["brier"],
                                          "engine_logloss": metrics["derived_moneyline"]["logloss"]}
    return {"markets": markets, "summary": summary}


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
     "distribution_mean_total", "distribution_mean_margin", "distribution_sd_total", "distribution_sd_margin"]
    + [f"p_over_{str(l).replace('.', '_')}" for l in TOTAL_LINE_GRID]
    + [f"p_push_{str(l).replace('.', '_')}" for l in TOTAL_LINE_GRID]
    + [f"p_under_{str(l).replace('.', '_')}" for l in TOTAL_LINE_GRID]
    + [f"p_home_cover_{str(m).replace('.', '_')}" for m in RUN_LINE_GRID]
    + [rl_col(m, side) for m in RUN_LINE_GRID_FULL
       for side in ("home", "push", "away")]
    + [rl_col(m, side) for m in RUN_LINE_GRID_FULL
       for side in ("away_favorite", "away_push", "home_dog")]
    + ["game_id", "home_team", "away_team",
       "p_home_win_derived", "p_away_win_derived",
       "home_score", "away_score", "total_runs",
       "ml_win_prob", "agreement_conflict", "fold_idx",
     "fair_total", "fair_spread", "p_over_fair", "p_cover_fair",
     "distribution_fit_n", "distribution_fit_end", "p_tie"]
)
NULLABLE_MARKET_COLUMNS = {"ml_win_prob", "fold_idx", "distribution_fit_end",
                           "fair_total", "fair_spread", "p_over_fair", "p_cover_fair",
                           "distribution_fit_n", "p_tie"}


def persist_markets(markets: pd.DataFrame, target_date_str: str,
                    summary: dict,
                    out_dir: Optional[Path] = None) -> Path:
    out_path = (out_dir or DATA_DELIVERY_DIR) / f"run_engine_markets_{target_date_str}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    markets = markets.copy()
    for column in ("fold_idx", "fair_total", "fair_spread", "p_over_fair", "p_cover_fair",
                   "distribution_fit_n", "distribution_fit_end", "p_tie"):
        if column not in markets: markets[column] = np.nan
    # Identity backfill BEFORE the strict contract check (2026-10-25
    # off-day: the OOF score frame carries game_pk/game_date/scores but no
    # team names; the check used to raise "missing game_id/home_team/
    # away_team" before the fill below could derive game_id from game_pk).
    # Derive game_id from the stable game_pk; team names stay NA here and
    # are attached from the decided frame by run_engine_daily before this
    # boundary (legacy direct callers keep the nullable path).
    for _c in ("game_id", "home_team", "away_team"):
        if _c not in markets.columns:
            if _c == "game_id" and "game_pk" in markets.columns:
                markets[_c] = markets["game_pk"].astype(str)
            else:
                markets[_c] = pd.NA
    missing = [c for c in MARKET_COLUMNS_V3 if c not in markets.columns]
    if missing:
        raise ValueError(f"markets frame missing required columns: {missing}")
    # The diagnostic margin-push extension rides along too: exactly its
    # whole-margin push legs (MARGIN_PUSH_EXTENSION, same sampled CDF, never
    # refitted) — not every p_rl_ column the pricer emits (per-game median
    # halves would make the schema data-dependent), like the totals
    # off-grid extras above.
    extension_rl = {rl_col(m, side) for m in MARGIN_PUSH_EXTENSION
                    for side in ("push", "away_push")}
    extra = [c for c in markets
             if (c.startswith(("p_over_", "p_push_", "p_under_"))
                 or c in extension_rl)
             and c not in MARKET_COLUMNS_V3]
    frame = markets[[*MARKET_COLUMNS_V3, *extra]].copy()
    # OOF identity fields are part of the shipped join contract. Older direct
    # callers may not provide them; derive game_id from the stable key, while
    # team names remain nullable only for legacy direct callers. Daily OOF
    # rows must carry the source identity columns before this boundary.
    for _c in ("game_id", "home_team", "away_team"):
        if _c not in frame.columns:
            if _c == "game_id" and "game_pk" in frame.columns:
                frame[_c] = frame["game_pk"].astype(str)
            else:
                frame[_c] = pd.NA
    # Preserve the frontend join key for slate rows. Pre-game rows use the
    # ESPN game_id as game_pk; if the daily path did not carry game_id through,
    # recover it before the strict contract check rather than silently
    # suppressing every Today's Games market selector.
    if "game_id" in frame.columns and "kind" in frame.columns:
        missing_game_id = frame["game_id"].isna() & frame["kind"].eq("slate")
        if missing_game_id.any():
            frame.loc[missing_game_id, "game_id"] = frame.loc[missing_game_id, "game_pk"]

    # Persist the contract even for direct callers that do not use the daily
    # wrapper; the daily path recomputes it over markets + slate before the
    # final persist (see run_engine_daily).
    summary.setdefault("artifact_contract", build_artifact_contract(frame, frame))
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
    # LF-pinned delivery bytes (pandas csv default is CRLF on some stacks).
    frame.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(out_path)
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path.with_suffix(".json.tmp"), "w") as f:
        json.dump(summary, f, indent=2)
    meta_path.with_suffix(".json.tmp").replace(meta_path)
    logger.info("Run engine markets: %d rows -> %s (+ meta json)", len(frame), out_path.name)
    return out_path


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


def brier_score(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece_score(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum((bins == b).mean() * abs(p[bins == b].mean() - y[bins == b].mean()) for b in np.unique(bins)))


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
    if not len(y):
        return {"n": 0, "actual_win_rate": None, "win_rate": None,
                "predicted_mean": None, "ece_raw": None, "ece_calibrated": None,
                "brier": None, "logloss": None, "logloss_calibrated": None,
                "beats_baseline_logloss": None,
                "by_pick": {side: {"n": 0, "win_rate": None, "predicted_mean": None}
                            for side in ("home", "away")}}
    base = float(y.mean())
    p_cal = p.copy()  # already calibrated at issuance; never fit a second card map
    row = {
        "n": int(len(y)),
        "actual_win_rate": round(base, 4),
        "win_rate": round(base, 4),
        "predicted_mean": round(float(np.mean(p_cal)), 4),
        "ece_raw": ece_score(y, p),
        "ece_calibrated": ece_score(y, p_cal),
        "brier": round(brier_score(y, p), 5),
        "logloss": round(log_loss(y, p, labels=[0.0, 1.0]), 5),
        "logloss_calibrated": round(log_loss(y, p_cal, labels=[0.0, 1.0]), 5),
        # Single-class cards (a holdout where every favored pick won, or every
        # lost) are degenerate but valid — never a ValueError in an artifact.
        "beats_baseline_logloss": bool(
            log_loss(y, p, labels=[0.0, 1.0])
            < log_loss(y, np.full(len(y), base), labels=[0.0, 1.0])),
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
            "logloss": round(log_loss(yh, ph, labels=[0.0, 1.0]), 5),
            "baseline_rate": round(base_h, 4),
            "baseline_logloss": round(
                log_loss(yh, np.full(len(yh), base_h), labels=[0.0, 1.0]), 5),
            "beats_baseline_logloss": bool(
                log_loss(yh, ph, labels=[0.0, 1.0])
                < log_loss(yh, np.full(len(yh), base_h), labels=[0.0, 1.0])),
        }
    return row



def _pooled_grid_crps(df: pd.DataFrame, outcome: np.ndarray,
                      prob_base: str, grid, mask=None) -> float | None:
    """Mean grid CRPS over pooled OOF rows (the "Pooled CRPS" cell).

    CRPS = step * mean_rows(sum_lines((P(Y > line) - 1{y > line})^2)) over
    the sport's grid (raw over/cover legs — no 2-way rescale; push mass
    belongs to the CDF). Population: ALL pooled OOF rows with a finite
    outcome and complete grid legs — deliberately NOT lean-gated. None
    when the grid is absent or no row qualifies. Totals grid is
    half-lines (step 0.5, exact for integer runs); the margin grid is the
    traded run-line set (partial CRPS, truncated — documented, identical
    every run, comparable across runs).
    """
    grid = np.asarray(grid, dtype=float)
    if grid.size < 2:
        return None
    step = float(grid[1] - grid[0])
    cols = [f"{prob_base}_{str(float(x)).replace('.', '_')}" for x in grid]
    if any(c not in df.columns for c in cols):
        return None
    P = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = np.asarray(outcome, dtype=float)
    use = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    ok = use & np.isfinite(y) & np.isfinite(P).all(axis=1)
    if not ok.any():
        return None
    gap = P[ok] - (y[ok][:, None] > grid[None, :]).astype(float)
    return round(float(step * (gap ** 2).sum(axis=1).mean()), 4)


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
                  else away) — the run line model's own derived moneyline;
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
        ou["auc_n"] = int(len(ou_arr_p))
        if ou_hold.any():
            ou.setdefault("holdout", {})["auc"] = _safe_auc(
                ou_arr_p[ou_hold], ou_arr_y[ou_hold])
            ou["holdout"]["auc_n"] = int(ou_hold.sum())

    # --- run line: HOME cover at -1.5, half-run lines never push ---
    # HOME-SIDE (2026-10-12): only home-lean rows (P(home covers -1.5) >
    # 50%) are graded — no pick is made on the away side; the card fields
    # and their meanings are unchanged.
    if "p_home_cover_1_5" in df.columns:
        rp = df["p_home_cover_1_5"].to_numpy(float)
        margin = hs - as_
        home_covers = (margin >= 2).astype(float)
        graded = np.isfinite(rp)
        ok = graded & (rp > 0.5)
        rl = _winner_card_stats(rp[ok], home_covers[ok], _fold_idx(pks[ok]),
                                hold_mask[ok], np.ones(int(ok.sum()), bool))
        # AUC ranks the FULL graded pool, not the lean-gated card rows —
        # conditioning on P > 0.5 range-restricts the score to (0.5, 1] and
        # discards the pairs spanning the 50/50 line that carry the
        # discrimination (2026-10-13: NBA 0.7304 full vs 0.6435 gated).
        # ``auc_n`` states the population so it is never read as ``n``.
        rl["auc"] = _safe_auc(rp[graded], home_covers[graded])
        rl["auc_n"] = int(graded.sum())
        if hold_mask[graded].any():
            rl.setdefault("holdout", {})["auc"] = _safe_auc(
                rp[graded][hold_mask[graded]],
                home_covers[graded][hold_mask[graded]])
            rl["holdout"]["auc_n"] = int(hold_mask[graded].sum())
    else:
        rl = None

    # --- derived_ml: HOME win (the RUN LINE model's own moneyline) ---
    # SOURCE: p_home_win_derived — the split-skew Gaussian Monte Carlo moneyline derived from
    # the SAME calibrated margin-CDF machinery that prices the run-line card.
    # It is a DISTINCT model from the binary moneyline ensemble. HOME-SIDE
    # (2026-10-12): only home-lean rows (P(home win) > 50%) are graded; the
    # ensemble reference below is scored on the SAME home-lean population
    # so the comparison stays apples-to-apples.
    home_won = (hs > as_).astype(float)
    ml = None
    if "p_home_win_derived" in df.columns:
        mp = df["p_home_win_derived"].to_numpy(float)
        graded = np.isfinite(mp)
        ok = graded & (mp > 0.5)
        ml = _winner_card_stats(mp[ok], home_won[ok], _fold_idx(pks[ok]),
                                hold_mask[ok], np.ones(int(ok.sum()), bool))
        # AUC on the home-side probability vs the home win (the standard
        # moneyline discrimination) over the FULL graded pool. The card's
        # calibration metrics stay lean-gated; ranking cannot be — see the
        # run-line note above. ``auc_n`` carries the pool size.
        ml["auc"] = _safe_auc(mp[graded], home_won[graded])
        ml["auc_n"] = int(graded.sum())
        if hold_mask[graded].any():
            ml.setdefault("holdout", {})["auc"] = _safe_auc(
                mp[graded][hold_mask[graded]],
                home_won[graded][hold_mask[graded]])
            ml["holdout"]["auc_n"] = int(hold_mask[graded].sum())
        ml["source"] = "split_skew_mc_p_home_win_derived"
        # distribution_diagnostic: schema-stable record of the distribution model finding
        # (equals the card itself now that the card IS the derived moneyline).
        ml["distribution_diagnostic"] = {
            "n": ml["n"], "actual_win_rate": ml["actual_win_rate"],
            "predicted_mean": ml["predicted_mean"],
            "ece_raw": ml["ece_raw"], "by_pick": ml["by_pick"],
        }
        # Moneyline ENSEMBLE one-line reference (comparison anchor only),
        # scored on the SAME home-lean population as the card above.
        ens = (df["ml_win_prob"].to_numpy(float)
               if "ml_win_prob" in df.columns else None)
        if ens is not None and np.isfinite(ens).any():
            eok = np.isfinite(ens) & (ens > 0.5)
            es = _winner_card_stats(ens[eok], home_won[eok],
                                    _fold_idx(pks[eok]), hold_mask[eok],
                                    np.ones(int(eok.sum()), bool))
            ml["ml_reference"] = {
                "source": "ml_win_prob", "n": es["n"],
                "win_rate": es["win_rate"],
                "predicted_mean": es["predicted_mean"],
            }

    # Pooled CRPS (the whole-OOF distribution cell).
    _t = _pooled_grid_crps(df, total, "p_over", TOTAL_LINE_GRID)
    _m = _pooled_grid_crps(df, hs - as_, "p_home_cover", RUN_LINE_GRID)
    if _t is not None and ou is not None:
        ou["crps"] = _t
    for _c in (rl, ml):
        if _m is not None and _c is not None:
            _c["crps"] = _m
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


def run_line_feature_weights(decided_games: pd.DataFrame,
                             final_fit_rounds: dict[str, int]) -> dict[str, float]:
    """The RUN LINE's own per-feature weights: pooled, n-weighted LightGBM
    split GAIN across the two per-side Poisson fits (home-λ + away-λ),
    normalized to sum to 1.0 over the SERVED features.

    The side models are the run line's only feature consumers (the variance layer
    carries no per-feature parameters), so this gain is the run-line model's
    honest importance — the drift table's MODEL WEIGHT column reports it
    instead of the moneyline blend's. The team-ID categorical columns are
    the trees' encoding, not served features: their gain is dropped and the
    rest renormalized (the moneyline's weight_pct convention sums to 100
    across the served list). Computed from the DECIDED frame only — never
    slate-dependent, so an off-day run weights exactly like a game day.
    Diagnostic only: a failure returns {} and never blocks pricing.
    """
    from lightgbm import LGBMRegressor
    gain: dict[str, float] = {}
    try:
        for side in ("home", "away"):
            _, cols = build_side_frame(decided_games, side,
                                       strict_feature_parity=True)
            tr = decided_games.reindex(columns=cols).astype(float)
            _apply_categorical_ids(tr, rows=decided_games)
            m = LGBMRegressor(**RUN_LGBM_PARAMS)
            m.set_params(n_estimators=int(final_fit_rounds[side]))
            m.fit(tr, decided_games[f"{side}_score"].to_numpy(dtype=float))
            gains = m.booster_.feature_importance(importance_type="gain")
            names = list(m.booster_.feature_name())
            n_rows = float(len(tr))
            for f, g in zip(names, gains):
                gain[f] = gain.get(f, 0.0) + float(g) * n_rows
        tot = sum(gain.values())
        weight = {k: v / tot for k, v in gain.items()} if tot > 0 else {}
        id_share = sum(weight.pop(c, 0.0) for c in RUN_TREE_CATEGORICAL_COLS)
        if id_share > 0 and weight:
            kept = sum(weight.values())
            if kept > 0:
                weight = {k: v / kept for k, v in weight.items()}
        return weight
    except Exception as exc:  # diagnostic only — never block pricing
        logger.warning("run_line_feature_weights: unavailable (%s)", exc)
        return {}


def predict_slate_runs(decided_games: pd.DataFrame, slate_games: pd.DataFrame,
                       final_fit_rounds: dict[str, int],
                       n_draws: int = MC_DRAWS,
                       seed: int = MARKET_SEED,
                       calibration: Optional[dict] = None,
                       ensemble_models: Optional[dict] = None,
                       ensemble_weights: Optional[dict[str, float]] = None
                       ) -> pd.DataFrame:
    """λ + full market grid for TODAY'S SLATE.

    Default (legacy) path: side models refit on ALL decided games at fixed
    rounds (median fold early-stopping iteration — no early stopping
    against the future), then priced through the same split-skew
    calibrated Monte Carlo as OOF rows.

    Ensemble path (STRUCTURAL ADOPTION 2026-10-23, NBA-C parity): when
    ensemble_models + ensemble_weights are given, slate means come from
    predict_ensemble_slate (full-view blend at OOF-earned weights) instead
    of the single-LGBM refit. The run-line drift weight stays the
    diagnostic single-model refit so the monitor column keeps its
    definition; the served mus are the blend.
    """
    # Run line's own MODEL WEIGHT (2026-09-27): pooled split-gain from the
    # side models, computed on the DECIDED frame — BEFORE the empty-slate
    # early return, so an off-day run weights exactly like a game day (the
    # 2026-09-29 all-NaN drift-weight regression had it behind the return).
    _rl_weight = run_line_feature_weights(decided_games, final_fit_rounds)
    if slate_games.empty:
        # Off-day: no slate rows to price, but the weights still ride the
        # empty frame's attrs to run_engine_daily's monitor block.
        empty = pd.DataFrame()
        empty.attrs["run_line_feature_weights"] = _rl_weight
        return empty
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
    if "game_id" in slate_games.columns:
        out["game_id"] = slate_games["game_id"].to_numpy()
    if _slate_key == "game_id":
        out = out.rename(columns={"game_id": "game_pk"})
        # Preserve the stable ESPN identity separately for the frontend join.
        # The markets contract uses game_pk for row identity, while Today's
        # Games resolves cards by game_id. Losing this field makes the slate
        # rows NaN at persist_markets and suppresses every card selector.
        out["game_id"] = slate_games["game_id"].to_numpy()
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
    if ensemble_models is not None and ensemble_weights is not None:
        # Served path (2026-10-23 adoption): full-view ensemble blend at
        # OOF-earned weights — the single-LGBM refit below is skipped.
        mu_h, mu_a = predict_ensemble_slate(
            ensemble_models, slate_games, ensemble_weights)
        out["home_expected_runs"] = np.round(np.clip(mu_h, 1e-6, None), 4)
        out["away_expected_runs"] = np.round(np.clip(mu_a, 1e-6, None), 4)
    else:
        for side in ("home", "away"):
            _, cols = build_side_frame(decided_games, side,
                                       strict_feature_parity=True)
            tr = decided_games.reindex(columns=cols).astype(float)
            va = slate_games.reindex(columns=cols).astype(float)
            # Team-ID categoricals (2026-09-23 ADOPT): attach AFTER the float
            # reindex, resolved from each frame's own team abbreviations. Slate
            # rows carry abbreviations (ESPN slates) or nothing at all —
            # _add_team_ids degrades to the reserved UNK slot per the
            # moneyline's own contract.
            _apply_categorical_ids(tr, rows=decided_games)
            _apply_categorical_ids(va, rows=slate_games)
            model = LGBMRegressor(**RUN_LGBM_PARAMS)
            model.set_params(n_estimators=int(final_fit_rounds[side]))
            model.fit(tr, decided_games[f"{side}_score"].to_numpy(dtype=float))
            out[f"{side}_expected_runs"] = np.round(
                np.clip(model.predict(va), 1e-6, None), 4)
    if not calibration or calibration.get("method") != family.METHOD:
        raise ValueError("MLB slate requires a fitted split-skew challenger bundle")
    out["kind"] = "slate"
    out = apply_market_calibration(out, calibration)
    for col in ("home_score", "away_score", "total_runs", "fold_idx", "ml_win_prob"):
        out[col] = np.nan
    out["agreement_conflict"] = False
    unresolved = out.game_pk.isna()
    if unresolved.any():
        logger.warning("Dropping %d unresolved slate identities", int(unresolved.sum()))
        out = out[~unresolved].reset_index(drop=True)
    out.attrs["run_line_feature_weights"] = _rl_weight
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

    Prior-origin split-skew fitting and calibration -> full Monte Carlo grid
    -> slate means priced through the same calibrated family ->
    agreement conflicts vs predictions_history → rolling totals Brier.
    Returns the monitor-embed block plus written artifact paths."""
    decided = (decided_snapshot.copy() if decided_snapshot is not None
               else get_decided_frame(games))
    # FEATURE PARITY (2026-09-27): no attach step — the run line trains and
    # prices on exactly the moneyline's active feature list. (The P1
    # projection attach adopted by gate 7e4c529 was removed here so both
    # models see identical features.)
    # STRUCTURALLY ADOPTED 2026-10-23: the served means come from the
    # full-view score ensemble (earned prior-fold blend weights), not the
    # single-LightGBM run_oof — NBA-C parity (verdict: not a gate pass).
    result = walk_forward_oof_ensemble(decided)
    oof = result["oof"]
    member_weights = dict(result["member_weights"])
    logger.info("score-ensemble serving weights (OOF-earned): %s",
                {k: round(float(v), 4) for k, v in member_weights.items()})

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

    # Serving slate means come from the ensemble fitted on all decided
    # games, blended with the walk's OOF-earned weights — never equal
    # priors, never an in-sample optimum (moneyline fit_final convention).
    _final_ensemble = fit_final_ensemble(decided)
    slate_frame = predict_slate_runs(
        decided, target_games, {"home": RUN_FIXED_FIT_ROUNDS,
                               "away": RUN_FIXED_FIT_ROUNDS},
        n_draws=n_draws,
        calibration=summary.get("calibration", {}).get("calibrators"),
        ensemble_models=_final_ensemble,
        ensemble_weights=member_weights)
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
    if not slate_frame.empty:
        # Duplicate-key guard (2026-09-28 incident): a fallback/off-day slate
        # carrying games the OOF already scored made combined carry both
        # copies; persist_markets' artifact contract then hit mismatched
        # label-lookup lengths and the markets artifact was LOST for the day.
        # The slate row is never the authoritative record for a decided game
        # — the OOF row is — so drop the slate-side duplicates loudly and
        # keep any genuine pre-game rows.
        _oof_keys = set(oof["game_pk"].astype(str))
        _slate_dup = slate_frame["game_pk"].astype(str).isin(_oof_keys)
        if _slate_dup.any():
            logger.warning(
                "Run engine: dropping %d slate row(s) whose game_pk already "
                "exists in the OOF (decided games re-served as slate — the "
                "OOF row is authoritative)", int(_slate_dup.sum()))
            _surviving = slate_frame[~_slate_dup]
            combined = (pd.concat([markets, _surviving], ignore_index=True)
                        if not _surviving.empty else markets)
    summary["artifact_contract"] = build_artifact_contract(oof, combined)
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
    try:
        update_totals_history_store(combined, target_date_str)
    except Exception as exc:  # belt-and-braces: store must never fail the run
        logger.error("Totals history store call-site guard: %s", exc,
                     exc_info=True)
    totals_brier = compute_rolling_totals_brier(combined)

    # Phase-1 monitor shim: the ensemble walk carries member_weights, not
    # the single-model summary (final_fit_rounds/dispersion). Report the
    # walk geometry + served blend; downstream readers must tolerate the
    # ensemble shape (rounds/dispersion None when serving the blend).
    _ft = result.get("fold_table")
    s1 = {"n_folds": int(len(_ft)) if _ft is not None else 0,
          "n_games": int(len(oof)),
          "member_weights": member_weights,
          "final_fit_rounds": {},
          "home_dispersion_ratio": None,
          "away_dispersion_ratio": None}
    holdout_markets = {
        k.replace("market_", ""): v.get("holdout")
        for k, v in summary.items()
        if k.startswith("market_") and isinstance(v, dict) and v.get("holdout")
    }
    monitor_block = {
        "version": "phase3",
        "winner_cards": compute_winner_cards(
            markets, oof, HOLDOUT_DAYS),
        "family": family.FAMILY,
        "variance": summary["variance"],
        # Lattice fit disclosure (totals + margin, see
        # gaussian_family.LATTICE_CONFIG) — summary strips the delta vectors.
        "lattice": summary.get("lattice_summary") or
                   ((summary.get("calibration", {}).get("calibrators", {})
                     or {}).get("lattice_summary") or {}),
        "holdout": summary["holdout"],
        "mc_meta": summary["mc_meta"],
        "calibration": summary.get("calibration"),
        "distribution_metrics": summary.get("distribution_metrics"),
        "line_grid": summary["line_grid"],
        "holdout_gate": {
            "cutoff": summary["holdout_cutoff"],
            "n_pre": summary["n_pre"], "n_holdout": summary["n_holdout"],
            "markets": holdout_markets,
            "totals_beat_baselines_holdout": (
                all(bool(v.get("beats_baseline_logloss"))
                    for k, v in holdout_markets.items() if k.startswith("over_"))
                if any(k.startswith("over_") for k in holdout_markets) else None),
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
        # Run line's OWN feature weights (pooled split-gain, n-weighted,
        # from the two side models that price this board) — consumed by the
        # run-engine drift table's MODEL WEIGHT column so it reports the run
        # line model, not the moneyline blend.
        "feature_weights": slate_frame.attrs.get(
            "run_line_feature_weights", {}) if slate_frame is not None else {},
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


def _print_phase2(summary):
    print("\nSplit-skew Gaussian challenger, causal CRPS calibration + Monte Carlo")
    print(json.dumps(summary.get("distribution_metrics", {}), indent=2))


TOTALS_HISTORY_SCHEMA_VERSION = 1


def _th_line_key(line: float) -> str:
    """Grid line -> column key, e.g. 8.0 -> '8_0', 8.5 -> '8_5'."""
    return str(line).replace(".", "_")


def _th_iso_date(date_str: str) -> str:
    """Artifact date stamp -> ISO date string (20260919 -> 2026-09-19) so
    provenance round-trips CSV as a string, never an int."""
    s = str(date_str)
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def _th_grid_over_under_cols(line: float) -> tuple[str, str]:
    """p_over / p_under column names for a grid total line (frontend parity:
    market_diagnostics.grid_over_under_cols)."""
    key = _th_line_key(line)
    return f"p_over_{key}", f"p_under_{key}"


def _th_round_to_half(x: float) -> float:
    """Round to nearest 0.5, ties away from zero (frontend parity:
    market_diagnostics.round_to_half)."""
    if x < 0:
        return math.ceil(x * 2 - 0.5) / 2.0
    return math.floor(x * 2 + 0.5) / 2.0


def _th_clamp_to_grid(line: float) -> tuple[float, bool]:
    """Clamp into the shipped totals grid; returns (line, clamped)."""
    if line < TOTAL_LINE_GRID[0]:
        return TOTAL_LINE_GRID[0], True
    if line > TOTAL_LINE_GRID[-1]:
        return TOTAL_LINE_GRID[-1], True
    return line, False


def _th_norm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a frozen-store frame for identity comparison: game_pk as
    string (artifact dtype varies across legacy files), line as float,
    sorted by (game_pk, game_date)."""
    out = df.copy()
    out["game_pk"] = out["game_pk"].astype(str)
    out["line"] = out["line"].astype(float)
    return out.sort_values(["game_pk", "game_date"]).reset_index(drop=True)


def _th_price_history_rows_vectorized(
        dec: pd.DataFrame,
        known: set) -> tuple[list[dict], int]:
    """Price every not-yet-frozen decided OOF row in one vectorized pass.

    Same rules as the frontend's totals_history_frame (fair line = grid
    argmin of |re-scaled P(over) - 0.5| with lower-line tie-breaks,
    round-half-up clamped fallback, 2-way re-scaled pick, push-aware
    grading) but O(grid) numpy passes instead of per-row work — the seed
    path prices ~6,800 rows and per-row DataFrame ops are O(n^2) there.
    Returns (rows, n_priced); rows exclude game_pks already in ``known``.
    """
    n = len(dec)
    if not n:
        return [], 0
    lam_total = (pd.to_numeric(dec["home_expected_runs"], errors="coerce")
                 + pd.to_numeric(dec["away_expected_runs"], errors="coerce")
                 ).to_numpy(float)
    total = pd.to_numeric(dec["total_runs"], errors="coerce").to_numpy(float)
    # Fair-line argmin across the grid (ties keep the lower line).
    best_line = np.full(n, np.nan)
    best_delta = np.full(n, np.inf)
    po_map: dict[float, np.ndarray] = {}
    pu_map: dict[float, np.ndarray] = {}
    for line in TOTAL_LINE_GRID:
        over_col, under_col = _th_grid_over_under_cols(line)
        if over_col not in dec.columns or under_col not in dec.columns:
            continue
        po = pd.to_numeric(dec[over_col], errors="coerce").to_numpy(float)
        pu = pd.to_numeric(dec[under_col], errors="coerce").to_numpy(float)
        po_map[line], pu_map[line] = po, pu
        denom = po + pu
        valid = (np.isfinite(po) & np.isfinite(pu) & np.isfinite(denom)
                 & (denom > 0))
        delta = np.full(n, np.inf)
        delta[valid] = np.abs(po[valid] / denom[valid] - 0.5)
        take = valid & (delta < best_delta - 1e-12)
        best_delta[take] = delta[take]
        best_line[take] = line
    # Fallback: round-half-up projection clamped to the grid, where no
    # valid grid pair exists (legacy artifacts / unpriced rows).
    fallback = np.array([
        _th_clamp_to_grid(_th_round_to_half(lt))[0] if np.isfinite(lt)
        else np.nan for lt in lam_total])
    line = np.where(np.isnan(best_line), fallback, best_line)
    # Re-scaled probabilities at each row's selected line.
    po_sel = np.full(n, np.nan)
    pu_sel = np.full(n, np.nan)
    for ln in np.unique(line[np.isfinite(line)]):
        mask = line == ln
        po_sel[mask] = po_map[float(ln)][mask]
        pu_sel[mask] = pu_map[float(ln)][mask]
    denom = po_sel + pu_sel
    priceable = (np.isfinite(total) & np.isfinite(line) & np.isfinite(po_sel)
                 & np.isfinite(pu_sel) & np.isfinite(denom) & (denom > 0))
    if not priceable.any():
        return [], 0
    idx = np.where(priceable)[0]
    p = po_sel[idx] / denom[idx]
    pick = np.where(p >= 0.5, "Over", "Under")
    pick_prob = np.where(p >= 0.5, p, 1.0 - p)
    t_idx, l_idx = total[idx], line[idx]
    winner = np.where(t_idx == l_idx, "Push",
                      np.where(t_idx > l_idx, "Over", "Under"))
    correct = np.where(winner == "Push", np.nan,
                       (pick == winner).astype(float))
    rows: list[dict] = []
    pks = dec["game_pk"].astype(str).to_numpy()
    for j, i in enumerate(idx):
        pk = pks[i]
        if pk in known:
            continue
        rows.append({
            "game_pk": pk,
            "game_date": str(dec["game_date"].iloc[i]),
            "home_score": dec["home_score"].iloc[i],
            "away_score": dec["away_score"].iloc[i],
            "total_runs": int(total[i]) if float(total[i]).is_integer()
            else float(total[i]),
            "line": float(line[i]),
            "pick": str(pick[j]),
            "pick_prob": round(float(pick_prob[j]), 6),
            "winner": str(winner[j]),
            "correct": float(correct[j]) if np.isfinite(correct[j])
            else None,
        })
    return rows, len(rows)


def update_totals_history_store(markets: pd.DataFrame,
                                target_date_str: str) -> Optional[Path]:
    """Append newly-decided OOF games to the frozen totals-history store.

    Store: data_delivery/run_engine_totals_history.csv (+ .meta.json). Rows
    are priced ONCE at first publication (first artifact date in which the
    game_pk appears as a decided OOF row — the walk-forward's prequential
    prediction) and never mutated afterward. Idempotent: re-runs and repulls
    add nothing for game_pks already in the store. A store failure is logged
    and returned via the meta — it must never fail the daily run.
    """
    meta: dict = {"schema_version": TOTALS_HISTORY_SCHEMA_VERSION,
                  "run_date": target_date_str, "updated": False,
                  "seeded": False, "rows_added": 0, "conflicts_skipped": 0,
                  "error": None}
    store_path = DATA_DELIVERY_DIR / "run_engine_totals_history.csv"
    try:
        if store_path.exists():
            store = _th_norm_scores(pd.read_csv(store_path,
                                                low_memory=False))
        else:
            # Seed from every retained dated markets artifact (oldest first
            # so first-publication-wins is deterministic).
            store = pd.DataFrame()
            legacy = sorted(DATA_DELIVERY_DIR.glob(
                "run_engine_markets_2026*.csv"))
            for art in legacy:
                date_str = art.stem.rsplit("_", 1)[-1]
                try:
                    df = pd.read_csv(art, low_memory=False)
                except Exception:
                    continue
                if "kind" not in df.columns:
                    continue
                dec = df[(df["kind"] == "oof")].copy()
                if "total_runs" in dec.columns:
                    dec = dec[dec["total_runs"].notna()]
                known = set(store["game_pk"]) if len(store) else set()
                rows, _ = _th_price_history_rows_vectorized(dec, known)
                for r in rows:
                    r["source_artifact_date"] = _th_iso_date(date_str)
                if rows:
                    store = pd.concat([store, pd.DataFrame(rows)],
                                      ignore_index=True)
                meta["seeded"] = bool(meta["seeded"] or rows)
                meta["rows_added"] += len(rows)
        # Append this run's newly-decided OOF rows (first publication only).
        dec = markets[(markets.get("kind") == "oof")].copy()
        if "total_runs" in dec.columns:
            dec = dec[dec["total_runs"].notna()]
        known = set(store["game_pk"]) if len(store) else set()
        rows, n_priced = _th_price_history_rows_vectorized(dec, known)
        conflicts = n_priced - len(rows)  # priced but already frozen
        for r in rows:
            r["source_artifact_date"] = _th_iso_date(target_date_str)
        if rows:
            store = pd.concat([store, pd.DataFrame(rows)],
                              ignore_index=True)
        meta["rows_added"] += len(rows)
        meta["conflicts_skipped"] = conflicts
        meta["updated"] = bool(meta["rows_added"] or meta["seeded"])
        if len(store):
            store = _th_norm_scores(store)
            tmp = store_path.with_suffix(".csv.tmp")
            store.to_csv(tmp, index=False)
            tmp.replace(store_path)
            meta["n_rows"] = int(len(store))
            meta_path = (DATA_DELIVERY_DIR
                         / "run_engine_totals_history.meta.json")
            meta_path.write_text(json.dumps(meta, indent=2),
                                 encoding="utf-8")
            logger.info("Totals history store: %d rows (%d added this run,"
                        " seeded=%s) -> %s",
                        len(store), meta["rows_added"], meta["seeded"],
                        store_path.name)
        return store_path if len(store) else None
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("Totals history store update FAILED (run continues; "
                     "frontend falls back to legacy re-pricing): %s", exc,
                     exc_info=True)
        try:
            (DATA_DELIVERY_DIR / "run_engine_totals_history.meta.json"
             ).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass
        return None


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
    decided = get_decided_frame(df)
    # Served path (2026-10-23 adoption): full-view ensemble, not run_oof.
    result = walk_forward_oof_ensemble(decided)

    # Phase 2 consumes Phase 1's OOF λ directly (same process, no retrain).
    history_path = DATA_DELIVERY_DIR / f"predictions_history_{date_stamp}.csv"
    if not history_path.exists():
        history_path = DATA_DELIVERY_DIR / "predictions_history_latest.csv"
    ml_probs = load_moneyline_probs(history_path)

    mk = derive_markets_v3(result["oof"], moneyline_probs=ml_probs)
    oof_path = persist_oof(result["oof"], date_stamp)
    mkt_path = persist_markets(mk["markets"], date_stamp, mk["summary"])

    oof_cli = result["oof"]
    print("\n================ RUN ENGINE PHASE 1 (ensemble) ================")
    print(f"folds={len(result['fold_table'])}  oof games={len(oof_cli)}")
    print(f"serving weights: {result['member_weights']}")
    for side, mu_col, y_col in (
            ("home", _MLB_MU_H, "home_score"),
            ("away", _MLB_MU_A, "away_score")):
        y = oof_cli[y_col].to_numpy(float)
        mu = oof_cli[mu_col].to_numpy(float)
        print(f"{side} pooled deviance: {poisson_deviance(y, mu):.5f}")
    print(f"\nOOF artifact: {oof_path}")
    _print_phase2(mk["summary"])
    print(f"Markets artifact: {mkt_path}")


# ---------------------------------------------------------------------------
# Causal whole-distribution layer. Mirrored across sports deliberately: these
# backends run independently in Kaggle; no new shared runtime is required.
# ---------------------------------------------------------------------------
def calibrate_market_frame(df, n_draws=MC_DRAWS, seed=MC_SEED):
    """Dated OOF only: fit split-skew calibration on strictly prior origins."""
    priced, bundle = causal_distribution_frame(df, n_draws=n_draws, seed=seed)
    if priced is not None and len(priced):
        metrics = distribution_score_report(priced, bundle)
        bundle["oof_metrics"] = metrics
        priced.attrs["distribution_metrics"] = metrics
        # The issued-CDF handles served the report; drop them so the
        # runtime-only payload never rides the bundle into delivery
        # attrs further (challenger_bundle is attrs-housed downstream).
        bundle.pop("_issued_cdfs", None)
    return priced, bundle


def apply_market_calibration(df, bundle):
    if not bundle or bundle.get("method") != family.METHOD:
        raise ValueError("serving requires a fitted split-skew challenger bundle")
    mc = bundle.get("monte_carlo", {})
    out = _distribution_price(df, bundle.get("maps", {}),
                              n_draws=mc.get("requested_draws", MC_DRAWS),
                              seed=mc.get("seed", MC_SEED))
    out["distribution_fit_n"] = bundle.get("fit_n", 0)
    out["distribution_fit_end"] = bundle.get("fit_end")
    return out


DISTRIBUTION_SPORT = "mlb"
CRPS_MIN_HISTORY = family.MIN_HISTORY


def _distribution_blocks(frame):
    """Date-ordered, indivisible forecast origins; never trust fold label order."""
    date_col = "game_date" if DISTRIBUTION_SPORT == "mlb" else "gameday"
    fold_col = "fold_idx" if DISTRIBUTION_SPORT == "mlb" else "fold_id"
    dates = pd.to_datetime(frame[date_col], errors="coerce", utc=True).dt.normalize()
    if dates.isna().any():
        raise ValueError("distribution evaluation requires finite forecast dates")
    labels = frame[fold_col] if fold_col in frame else dates
    if labels.isna().any():
        raise ValueError("distribution evaluation requires finite origin labels")
    blocks = []
    for label in pd.unique(labels):
        mask = (labels == label).to_numpy()
        blocks.append((dates[mask].min(), dates[mask].max(), mask))
    blocks.sort(key=lambda b: b[0])
    for previous, current in zip(blocks, blocks[1:]):
        if previous[1] >= current[0]:
            raise ValueError("distribution forecast blocks overlap in calendar time")
    return blocks


def _distribution_inputs(frame):
    hc, ac = (("home_expected_runs", "away_expected_runs")
              if DISTRIBUTION_SPORT == "mlb" else ("mu_h", "mu_a"))
    h, a = frame[hc].to_numpy(float), frame[ac].to_numpy(float)
    if not (np.isfinite(h).all() and np.isfinite(a).all()):
        raise ValueError("non-finite score predictions cannot be priced")
    if np.any(h <= 0.) or np.any(a <= 0.):
        raise ValueError("expected scores must be positive")
    return h, a


def _distribution_support(kind):
    return family.support_for(kind, DISTRIBUTION_SPORT)


def _distribution_design(h, a):
    return family.design(h, a, DISTRIBUTION_SPORT)


def _distribution_raw_cdf(h, a, kind, parameters=None):
    return family.raw_cdf(h, a, kind, DISTRIBUTION_SPORT, parameters)


def _distribution_transform(cdf, design, coefficients):
    return family.transform(cdf, design, coefficients)


def _distribution_no_tie(cdf, support, home_share=None, tie_rate=None):
    # Structural no-tie rule of the margin lattice. The NHL supplies a
    # causal OT/SO home-win share (maps["lattice"]["margin"]
    # ["ot_home_share"], estimated from prior OOF one-goal finals) so the
    # P(M=0) mass is redistributed to +/-1 by OT win probabilities from OOF;
    # None keeps the exact equal split every pre-lattice bundle used. The
    # NFL instead supplies a fitted overtime tie_rate (preregistered
    # ("nfl", "margin", "tie_mass")); None keeps the exact pre-lattice
    # valid tie mass. Other sports ignore tie_rate (passthrough).
    return family.no_tie(cdf, support, DISTRIBUTION_SPORT, home_share,
                         tie_rate=tie_rate)


def lattice_crps(cdf, support, outcome):
    return family.crps(cdf, support, outcome)


def _fit_distribution_map(h, a, outcome, kind, parameters=None):
    return family.fit_map(h, a, outcome, kind, DISTRIBUTION_SPORT,
                          parameters or family.defaults(DISTRIBUTION_SPORT))


def _distribution_cdfs(frame, maps, sampled=False, n_draws=MC_DRAWS,
                       seed=MC_SEED, meta_out=None):
    h, a = _distribution_inputs(frame)
    design = _distribution_design(h, a)
    identity_col = "game_pk" if DISTRIBUTION_SPORT == "mlb" else "game_id"
    if sampled and (identity_col not in frame or frame[identity_col].isna().any()):
        raise ValueError("Monte Carlo pricing requires finite game identities")
    out = {}
    # Integer-lattice corrections ride the maps: totals and margin are fit
    # INDEPENDENTLY (see gaussian_family.LATTICE_CONFIG) and applied AFTER
    # the monotone calibration map and the structural rules, BEFORE Monte
    # Carlo sampling. Stored bundles that predate lattice fitting carry no
    # "lattice" key and price through the exact pre-lattice path (NHL totals
    # keep their structural OT/SO parity from the variance parameters).
    lattice = maps.get("lattice") or {}
    for kind in ("total", "margin"):
        target = "totals" if kind == "total" else "margin"
        support, raw = _distribution_raw_cdf(h, a, kind, maps.get("variance"))
        cdf = _distribution_transform(raw, design, maps.get(kind, [0.] * 9))
        spec = dict(lattice.get(target) or {})
        if kind == "total" and DISTRIBUTION_SPORT == "nhl" and "alpha" not in spec:
            # NHL odd/even parity rides the VARIANCE parameters (structural,
            # fitted from strictly-prior finals), never the CRPS admission
            # gate for the calibration maps: alpha <= 0 (stored bundles
            # predating the key) is a byte-identical no-op.
            variance = maps.get("variance") or family.defaults(DISTRIBUTION_SPORT)
            spec["alpha"] = float(variance.get("total_parity_alpha", 0.))
        if kind == "margin":
            cdf = _distribution_no_tie(cdf, support, spec.get("ot_home_share"),
                                         tie_rate=spec.get("tie_rate"))
        cdf = family.lattice_correct(cdf, support, spec,
                                     DISTRIBUTION_SPORT, target)
        if sampled:
            cdf, meta = family.sample_cdf(cdf, support, frame[identity_col].astype(str),
                                          kind, DISTRIBUTION_SPORT, n_draws, seed)
            if meta_out is not None:
                meta_out.update(meta)
        out[kind] = (support, cdf)
    return out


def _distribution_price(frame, maps, n_draws=MC_DRAWS, seed=MC_SEED, meta_out=None,
                        cdfs_out: dict | None = None):
    """All prices and settlement legs from the same sampled challenger CDFs."""
    mc_meta = {}
    cdfs = _distribution_cdfs(frame, maps, sampled=True, n_draws=n_draws,
                               seed=seed, meta_out=mc_meta)
    if meta_out is not None:
        meta_out.update(mc_meta)
    if cdfs_out is not None:
        # Runtime-only handles for the OOF score report, which rebuilds
        # these same sampled CDFs from the same maps/seed/draws. The
        # report prefers the handles (byte-identical values, no recompute)
        # and recomputes when they are absent, so direct callers are
        # unaffected. Plain lists: numpy arrays in frame attrs break
        # pd.concat (its attrs-equality check cannot compare arrays).
        for kind, (support, cdf) in cdfs.items():
            cdfs_out[kind] = (np.asarray(support).tolist(),
                              np.asarray(cdf).tolist())
    h, a = _distribution_inputs(frame)
    updates = {"mu_margin": h - a, "mu_total": h + a}
    for kind, (support, cdf) in cdfs.items():
        mass = np.diff(np.column_stack([np.zeros(len(frame)), cdf]), axis=1)
        mean = mass @ support
        updates[f"distribution_mean_{kind}"] = mean
        updates[f"distribution_sd_{kind}"] = np.sqrt(np.maximum(mass @ (support ** 2) - mean ** 2, 0.))
        def at(threshold):
            index = int(np.floor(threshold) - support[0])
            if index < 0: return np.zeros(len(frame))
            if index >= len(support): return np.ones(len(frame))
            return cdf[:, index]
        if DISTRIBUTION_SPORT == "mlb":
            lines = (sorted(set([*TOTAL_LINE_GRID, *range(0, 31)])) if kind == "total" else
                     sorted(set([*RUN_LINE_GRID, *RUN_LINE_GRID_FULL,
                                 *[-x for x in RUN_LINE_GRID_FULL], 0.])))
        else:
            original = config.TOTAL_GRID if kind == "total" else config.SPREAD_GRID
            lines = sorted(set([*original, *[float(x) + .5 for x in original],
                                *([] if kind == "total" else config.HALF_STOP_LINES)]))
        half = support[:-1] + .5
        j = np.argmin(np.abs(1. - cdf[:, :-1] - .5), axis=1)
        lines = sorted(set([*lines, *half[j].tolist()]))
        for line in lines:
            upper, lower = at(line), at(np.ceil(line) - 1.)
            over, push, under = 1. - upper, np.maximum(upper - lower, 0.), lower
            if DISTRIBUTION_SPORT == "mlb":
                if kind == "total":
                    suffix = str(float(line)).replace('.', '_')
                    updates[f"p_over_{suffix}"] = over
                    updates[f"p_push_{suffix}"] = push
                    updates[f"p_under_{suffix}"] = under
                elif line >= .5:
                    updates[rl_col(line, "home")] = over
                    updates[rl_col(line, "push")] = push
                    updates[rl_col(line, "away")] = under
                    updates[rl_col(line, "away_favorite")] = at(-line - (1. if float(line).is_integer() else 0.))
                    updates[rl_col(line, "away_push")] = at(-line) - at(np.ceil(-line) - 1.)
                    updates[rl_col(line, "home_dog")] = 1. - at(-line)
                    updates[f"p_home_cover_{str(float(line)).replace('.', '_')}"] = over
            else:
                prefix = "p_over" if kind == "total" else "p_home_cover"
                pp = "p_push_total" if kind == "total" else "p_push"
                updates[_grid_key(prefix, line)] = over
                updates[_grid_key(pp, line)] = push
                updates[_grid_key("p_under" if kind == "total" else "p_away_cover", line)] = under
        if kind == "margin" and DISTRIBUTION_SPORT == "mlb":
            # Diagnostic margin-push extension (2026-10-16,
            # MARGIN_PUSH_EXTENSION): push-only mass for whole margins +-5..10
            # from the SAME sampled CDF (never refitted) so the Distribution
            # tab's modeled line is gapless to +-10. Push-only: the home-side
            # push leg plus the away-side push leg the spread reader resolves
            # — no home/away/cover/favorite/dog legs outside the market grid.
            # Additive setdefault: a future grid change that prices these
            # lines owns them instead.
            for m in MARGIN_PUSH_EXTENSION:
                upper, lower = at(m), at(np.ceil(m) - 1.)
                updates.setdefault(rl_col(m, "push"),
                                   np.maximum(upper - lower, 0.))
                updates.setdefault(rl_col(m, "away_push"),
                                   at(-m) - at(np.ceil(-m) - 1.))
        if kind == "margin":
            updates["p_home_win_derived"] = 1. - at(0.)
            updates["p_tie"] = at(0.) - at(-1.)
            updates["p_away_win_derived"] = at(-1.)
            updates["derived_ml"] = updates["p_home_win_derived"]
        # Nearest half-line to the actual CDF median, not the old integer
        # grid boundary. This remains defined even outside displayed grids.
        half = support[:-1] + .5
        j = np.argmin(np.abs(1. - cdf[:, :-1] - .5), axis=1)
        updates["fair_total" if kind == "total" else "fair_spread"] = half[j]
        updates["p_over_fair" if kind == "total" else "p_cover_fair"] = 1. - cdf[np.arange(len(frame)), j]
    out = pd.concat([frame.drop(columns=list(updates), errors="ignore"),
                     pd.DataFrame(updates, index=frame.index)], axis=1)
    out.attrs["distribution_method"] = family.METHOD
    out.attrs["mc_meta"] = mc_meta
    out.attrs["distribution_objective"] = "discrete_crps"
    if "home_score" in out and "away_score" in out:
        margin = out.home_score - out.away_score
        total = out.home_score + out.away_score
        for col, values in {"y_cover_fair": margin > out.fair_spread,
                            "y_push_spread_fair": margin == out.fair_spread,
                            "y_over_fair": total > out.fair_total,
                            "y_under_fair": total < out.fair_total,
                            "y_push_fair": total == out.fair_total}.items():
            out[col] = np.where(margin.notna() & total.notna(), values.astype(float), np.nan)
    return out


# ---------------------------------------------------------------------------
# MLB one-run "spread spike" rebalance (2026-10-16, PRE-REGISTERED)
# ---------------------------------------------------------------------------
# WHY: on the 20261016 artifact the pooled margin model sits P(M=+1) 0.1433
# against 0.1739 actual (+0.031 short — the +1 spike UNDERFITS the residual)
# and P(M=-1) 0.1255 against 0.1089 (-0.017 long — the -1 spike OVERFITS it).
# The structural walk-off share re-splits each row's TIE mass only
# ((q-1/2)*m0, m0 ~ 0.07 of model mass), so it can never absorb the tilt: the
# defect is the model's own ONE-RUN-FINAL split, P(+1)/(P(+1)+P(-1)) = 0.515
# against the gated pooled share q-hat ~ 0.60. The generic residual fit has
# the amplitude (its delta pins M=+1 at the 0.03 fix cap) but its held-out z
# (2.23 on 20261016) sits under the generic bar, which is why MLB margin has
# shipped structural-only.
#
# MECHANISM — direction and bar declared BEFORE any fitting (this module is
# the disclosure record: the shared family.LATTICE_PREREGISTERED table is
# byte-identical infrastructure mirrored in all four backends, so an
# MLB-only mechanism registers here, next to its code): the ADMITTED walk-off
# share is the target split of the model's one-run-final mass, and the
# rebalance closes the gap with TWO dynamically-sized legs — the primary
#     amt = (share - 1/2) * (p(+1) + p(-1)) / 2    (pooled model masses)
# moved from the funding donors into M=+1 (2026-10-16), and, since
# 2026-10-17, a SECOND recipient at M=-2 (the other spike the residual
# shows underfit: 20261017 pooled obs 0.0881 vs model 0.0785) whose leg is
# sized by ITS OWN SE-shrunk train shortfall — the same evidence function
# that caps donor capacity, read with the opposite sign. Every number is
# DYNAMIC — the primary leg comes from the gated pooled share estimator and
# the fitted masses, the -2 leg from the fitted residual at -2 itself; a
# -2 the actuals show LONG (or flat) receives exact 0. There is no constant
# amplitude anywhere.
#
# FUNDING (the "shift mass out of the neighbouring spread bins" ask, made
# honest): the declared donors (-1, +3 — M=-2 moved to the recipient side
# in the two-recipient extension, since the actuals show it SHORT not long)
# fund in proportion to their own ACTUALS-SUPPORTED excess — each donor's
# draw is capped at the shrunk train residual it can give before flipping
# sign, and the recipients' legs are scaled DOWN to the funding that cap
# actually provides when it falls short (donors never overdraw). A donor the
# actuals show SHORT funds exact 0 (measured 20261016: M=-2 is model 0.0784
# vs 0.0880 actual — the reason -2 is a recipient, not a donor), and the
# unfunded remainder falls to the mass-proportional sink, so the rebalance
# can never widen a bin its own evidence already shows short. Each donor's
# residual/capacity/drawn triple and each recipient's residual/amount ride
# the report.
#
# GATES: one-sided direction per bin (recipient up, donors down), zero-sum by
# construction, forbidden bins untouched, and the per-bin collateral guard
# vetoing any draw that pushes a bin past its own noise floor. Admission runs
# the SAME two-stage gate the walk-off share and the NFL key-number spikes
# use: the generic bar first (LATTICE_GATE_Z on the paired held-out loss),
# then — for this pre-registered, one-sided-direction mechanism only — the
# one-sided structural bar (LATTICE_STRUCTURAL_Z). Measured on the 20261017
# artifact (run_diagnostics/mlb17_origin_admissions.log): the share-sized
# amount clears the generic bar at the serving fit (z 3.27) but per-origin
# evidence is thinner (median z ~2.5), so the generic stage alone admitted
# 10/57 share-bearing origins and the pooled chart barely moved (+0.0015 at
# M=+1); the structural stage takes 36/57. The stage that admitted is
# disclosed per fit as "bar_applied". A refused rebalance returns no delta
# and prices byte-identically to before.
ONE_RUN_REBALANCE_RECIPIENT = 1        # M=+1: the walk-off one-run home win
ONE_RUN_REBALANCE_MIRROR = -1          # M=-1: its away-side mirror
ONE_RUN_REBALANCE_SECONDARY = -2       # M=-2: second recipient (2026-10-17); its
                                       # leg = its own SE-shrunk shortfall, one-sided
ONE_RUN_REBALANCE_DONORS = (-1, 3)     # funding order; capacity-capped


def fit_one_run_spread_rebalance(pmfs, outcomes, support, share,
                                 min_history=family.MIN_HISTORY):
    """Close the model's one-run-final split gap to the admitted walk-off share.

    Pre-registered MLB-only mechanism (module header above): ``share`` is the
    ALREADY-ADMITTED walk-off home-win estimate — without it this is a
    no-op — and the transfer amounts (the share-sized primary leg into M=+1
    plus the residual-sized secondary leg into M=-2), the donor capacities
    and every funding split are read from the fitted evidence, never
    constant. Returns ``(delta | None, report)``: the zero-sum,
    rounded transfer vector or None when the evidence, the direction, the
    collateral guard or the held-out gate refuses, plus the per-bin
    disclosure report.
    """
    report: dict = {"estimator": "one_run_spread_rebalance",
                    "sport": DISTRIBUTION_SPORT,
                    "recipient": ONE_RUN_REBALANCE_RECIPIENT,
                    "secondary": ONE_RUN_REBALANCE_SECONDARY,
                    "donors": list(ONE_RUN_REBALANCE_DONORS),
                    "bar": float(family.LATTICE_GATE_Z),
                    "structural_bar": float(family.LATTICE_STRUCTURAL_Z),
                    "share": None if share is None else round(float(share), 4),
                    "bins": []}
    if share is None:
        report["reason"] = "no_admitted_walkoff_share"
        return None, report
    support = np.asarray(support)
    pmfs = np.empty((0, len(support))) if pmfs is None else np.asarray(pmfs, float)
    outcomes = np.empty(0) if outcomes is None else np.asarray(outcomes, float)
    if pmfs.ndim != 2 or pmfs.shape[1] != len(support) or len(pmfs) != len(outcomes):
        report["reason"] = "misaligned_evidence"
        return None, report
    inside = ((outcomes >= support[0]) & (outcomes <= support[-1])
              & np.isfinite(outcomes))
    n_test = int(family.LATTICE_TEST_FRACTION * len(outcomes))
    if (inside.sum() < int(min_history) or n_test < family.LATTICE_MIN_TEST
            or len(outcomes) - n_test < int(min_history)):
        report["reason"] = "insufficient_held_out_evidence"
        return None, report
    train = np.flatnonzero(inside[:len(outcomes) - n_test])
    hold = (np.flatnonzero(inside[len(outcomes) - n_test:])  # GLOBAL rows: flatnonzero on a tail slice is slice-relative
                   + (len(outcomes) - n_test))
    if len(train) < int(min_history) or len(hold) < family.LATTICE_MIN_TEST:
        report["reason"] = "insufficient_held_out_evidence"
        return None, report
    index = {int(v): i for i, v in enumerate(support)}
    registered = (ONE_RUN_REBALANCE_RECIPIENT, ONE_RUN_REBALANCE_MIRROR,
                  ONE_RUN_REBALANCE_SECONDARY, *ONE_RUN_REBALANCE_DONORS)
    if any(int(v) not in index for v in registered):
        report["reason"] = "support_lacks_registered_bins"
        return None, report
    recipient = index[ONE_RUN_REBALANCE_RECIPIENT]
    mirror = index[ONE_RUN_REBALANCE_MIRROR]
    secondary = index[ONE_RUN_REBALANCE_SECONDARY]
    donors = [index[int(v)] for v in ONE_RUN_REBALANCE_DONORS]
    n_train = max(int(len(train)), 1)
    pbar = np.clip(pmfs[train].mean(axis=0), 0., 1.)
    obs = np.bincount(np.searchsorted(support, outcomes[train]),
                      minlength=len(support)).astype(float) / n_train
    resid = obs - pbar
    se = np.sqrt(pbar * (1. - pbar) / n_train)
    shrunk = family.lattice_local_deltas(pbar, resid, se)
    one_run_mass = float(pbar[recipient] + pbar[mirror])
    tilt = (float(share) - .5) * .5 * one_run_mass
    if not tilt > 0. or one_run_mass <= 0.:
        report["reason"] = "no_positive_tilt"
        return None, report
    # Secondary leg (2026-10-17): M=-2 is the second spike the pooled
    # residual shows underfit. Its amount is the SE-shrunk train shortfall
    # AT -2 itself — one-sided by construction (a -2 the actuals show long
    # or flat gets exact 0), with no magnitude cap — the disclosed
    # "capacity" is the shipped amount.
    secondary_capacity = max(0., float(shrunk[secondary]))
    tilt2 = secondary_capacity
    report["amount"] = round(float(tilt), 6)
    report["secondary_residual"] = round(float(resid[secondary]), 6)
    report["secondary_capacity"] = round(secondary_capacity, 6)
    report["secondary_amount"] = round(float(tilt2), 6)
    report["total_amount"] = round(float(tilt) + float(tilt2), 6)
    report["one_run_mass"] = round(one_run_mass, 6)
    report["model_split"] = round(float(pbar[recipient]) / one_run_mass, 4)
    registered_idx = {recipient, secondary, *donors}
    banned = {index[int(b)] for b in
              (family.lattice_config(DISTRIBUTION_SPORT, "margin")
               or {}).get("forbidden_bins", ()) if int(b) in index}
    sink = np.array([i for i in range(len(support))
                     if i not in registered_idx and i not in banned])
    sink_mass = float(pbar[sink].sum()) if len(sink) else 0.
    # Donor capacity = the shrunk train residual the donor can give before
    # its own sign flips: a donor the actuals show SHORT funds exact 0.
    capacity = {d: max(0., -float(shrunk[d])) for d in donors}
    total_capacity = float(sum(capacity.values()))
    # Funding never overdraws a donor: when the capped capacity cannot
    # cover both legs, the legs scale DOWN together to what the actuals
    # actually fund (the shortfall still falls to the sink below).
    need = float(tilt) + float(tilt2)
    funded = min(need, total_capacity) if total_capacity > 0. else 0.
    delta = np.zeros(len(support))
    delta[recipient] += tilt
    delta[secondary] += tilt2
    drawn = 0.
    for d in donors:
        draw = (funded * capacity[d] / total_capacity) if total_capacity > 0. else 0.
        delta[d] -= draw
        drawn += draw
        report["bins"].append({"bin": int(support[d]),
                               "residual": round(float(resid[d]), 6),
                               "capacity": round(float(capacity[d]), 6),
                               "drawn": round(float(draw), 6)})
    remainder = need - drawn
    if remainder > 0. and sink_mass > 0.:
        delta[sink] -= remainder * pbar[sink] / sink_mass
    report["funded_by_donors"] = round(float(drawn), 6)
    report["funded_by_sink"] = round(float(remainder), 6)
    if pbar.sum() > 0.:
        delta -= delta.sum() * (pbar / pbar.sum())
    for j in banned:
        delta[j] = 0.
    # Per-bin collateral guard: no bin's |residual| may worsen by more than
    # its own 1-sigma binomial noise floor (pooled over all inside rows).
    n_all = max(int(inside.sum()), 1)
    pooled = np.clip(pmfs[inside].mean(axis=0), 0., 1.)
    pooled_obs = np.bincount(np.searchsorted(support, outcomes[inside]),
                             minlength=len(support)).astype(float) / n_all
    se_pool = np.sqrt(pooled * (1. - pooled) / n_all)
    excess = ((np.abs((pooled_obs - pooled) - delta)
               - np.abs(pooled_obs - pooled)) - se_pool)
    report["collateral_max_excess"] = round(float(excess.max()), 6)
    if not (excess <= 0.).all():
        report["reason"] = "collateral_violation"
        return None, report
    indices_hold = np.searchsorted(support, outcomes[hold])
    d = (-2. * delta[indices_hold] + 2. * (pmfs[hold] @ delta)
         + float(delta @ delta))
    ses = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
    z = float(-d.mean() / ses) if ses > 0. else 0.
    report["z"] = round(z, 3)
    report["n_test"] = int(len(hold))
    if d.mean() < 0. and z >= family.LATTICE_GATE_Z:
        report["bar_applied"] = "generic"
    elif d.mean() < 0. and z >= family.LATTICE_STRUCTURAL_Z:
        # Pre-registered TWO-STAGE gate (2026-10-17; the house rule the
        # walk-off share and the NFL key-number spikes already use): the
        # generic bar applies first, and a mechanism whose direction was
        # declared one-sided BEFORE fitting may then take the one-sided
        # structural bar. Nothing else relaxes — the direction is enforced
        # by construction (recipient up; donors down only where their own
        # actuals support it), the collateral guard above is unchanged and
        # the held-out loss must still improve. Measured on the 20261017
        # artifact: 10/57 share-bearing origins clear the generic bar and
        # 36/57 clear the structural one, which is the difference between a
        # chart-level no-op and a visible ±1 correction.
        report["bar_applied"] = "structural"
    else:
        report["reason"] = "held_out_not_significant"
        return None, report
    delta[np.abs(delta) < family.LATTICE_DELTA_EPS] = 0.
    delta = np.round(delta, family.LATTICE_ROUND)
    report["significant"] = True
    report["delta_l1"] = round(float(np.abs(delta).sum()), 6)
    report["delta_max"] = round(float(np.abs(delta).max()), 6)
    return [float(v) for v in delta], report


def _fit_lattice_maps(outcomes, parameters, evidence):
    """Independent per-target lattice fits — totals AND margin.

    Structural terms ride the VARIANCE parameters (NHL totals OT/SO parity
    alpha; the no-tie +/-1 split weight — NHL OT/SO home-win share, MLB
    walk-off home-win share — estimated causally from prior OOF one-goal
    own BASE-model evidence — pmfs of strictly-prior OOF rows, each priced
    with its own origin's maps — per target, behind the held-out
    significance gate. Every value is plain JSON so stored bundles reprice
    byte-identically.
    """
    totals: dict = {}
    margin: dict = {}
    if DISTRIBUTION_SPORT == "nhl":
        alpha = float(parameters.get("total_parity_alpha", 0.))
        if alpha > 0.:
            totals["alpha"] = alpha
    if DISTRIBUTION_SPORT == "mlb":
        # Pre-registered walk-off split (family.LATTICE_PREREGISTERED): the
        # hierarchically pooled share is gated at the generic bar first; only
        # on rejection does the pre-registered one-sided structural bar apply
        # (direction q > 1/2 enforced inside). Disclosure rides the margin
        # spec; unconfirmed shifts keep the exact equal split.
        margin_ev = (evidence or {}).get("margin") or {}
        share, share_rep = family.fit_pooled_home_share(
            margin_ev.get("outcomes"), pmfs=margin_ev.get("pmfs"),
            support=margin_ev.get("support"))
        if share is None:
            share, share_rep = family.fit_pooled_home_share(
                margin_ev.get("outcomes"), pmfs=margin_ev.get("pmfs"),
                support=margin_ev.get("support"),
                bar=family.LATTICE_STRUCTURAL_Z, structural=True)
        if share is not None:
            margin["ot_home_share"] = share
            margin["split_share_report"] = share_rep
    if DISTRIBUTION_SPORT == "nhl":
        # The no-tie +/-1 split weight (NHL OT/SO) keeps the legacy gated
        # one-goal home-share estimate byte-stable (pinned by
        # test_distribution_objective_parity); cold starts and
        # unconfirmed shifts keep the exact equal split.
        margin_ev = (evidence or {}).get("margin") or {}
        share = family.fit_ot_home_share(margin_ev.get("outcomes"),
                                         pmfs=margin_ev.get("pmfs"),
                                         support=margin_ev.get("support"))
        if share is not None:
            margin["ot_home_share"] = share
    for target, kind in (("totals", "total"), ("margin", "margin")):
        ev = (evidence or {}).get(kind) or None
        if ev is None:
            continue
        (totals if target == "totals" else margin).update(
            family.fit_lattice_correction(ev.get("pmfs"), ev.get("outcomes"),
                                          ev.get("support"), DISTRIBUTION_SPORT,
                                          target))
    if DISTRIBUTION_SPORT == "mlb":
        # One-run spread rebalance (2026-10-16, pre-registered in the module
        # header above): the ADMITTED walk-off share is the target split of
        # the model's one-run-final mass, and this lattice transfer closes the
        # remaining gap, funded by the actuals-capacity donors (-1/-2/+3 — a
        # donor the actuals show SHORT funds exact 0). Disclosure always
        # rides the margin spec; a refused rebalance keeps the exact shipped
        # prices, and an admitted one composes with the generic delta.
        rebalance, rebalance_report = fit_one_run_spread_rebalance(
            margin_ev.get("pmfs"), margin_ev.get("outcomes"),
            margin_ev.get("support"), margin.get("ot_home_share"))
        margin["one_run_rebalance"] = rebalance_report
        if rebalance is not None:
            parts = ([margin["delta"]] if "delta" in margin else []) + [rebalance]
            weight = np.clip(np.asarray(margin_ev.get("pmfs"), float).mean(axis=0),
                             0., None)
            margin["delta"] = (family.compose_lattice_deltas(parts, weight)
                               if len(parts) > 1 else rebalance)
            margin["significant"] = True
            margin["delta_source"] = ("generic+one_run_rebalance"
                                      if len(parts) > 1 else "one_run_rebalance")
            fitted = np.asarray(margin_ev.get("support"))
            margin["support"] = [int(fitted[0]), int(fitted[-1])]
    return {"totals": totals, "margin": margin}


def _distribution_fit_maps(h, a, outcomes, lattice_evidence=None):
    parameters = family.fit_variance(h, a, outcomes, DISTRIBUTION_SPORT)
    if DISTRIBUTION_SPORT in ("nba", "nfl") and bool(getattr(config, "DISTRIBUTION_TOTAL_TILT_PACE", False)):
        # Adopted 2026-10-18 (V3a verdict: top-quintile over-230 gap
        # +0.0438 -> +0.0081, CRPS neutral). Totals pace-tilt location
        # correction rides the variance params; raw_cdf shifts the TOTAL
        # location only, so calibration refits against the tilted base.
        parameters.update(family.fit_tilt(h, a, outcomes, DISTRIBUTION_SPORT))
    if DISTRIBUTION_SPORT == "nhl" and bool(getattr(config, "DISTRIBUTION_TOTAL_SHRINK_NHL", False)):
        # 2026-10-19: totals slope-shrinkage (measured pooled-OOF slope of
        # actual totals on mu_total ~0.14 at corr ~0.02 — the spread carries
        # no signal and the fair-line card ran 56.2% predicted vs 50.8%
        # actual). Rides the variance params like the NBA/NFL tilt;
        # raw_cdf shrinks the TOTAL location only, so margin / run-line /
        # derived-ML price byte-identical and calibration refits against
        # the shrunk base.
        parameters.update(family.fit_total_shrinkage(h, a, outcomes, DISTRIBUTION_SPORT))
    return {"variance": parameters,
            **{kind: _fit_distribution_map(h, a, y, kind, parameters)
               for kind, y in outcomes.items()},
            "lattice": _fit_lattice_maps(outcomes, parameters, lattice_evidence)}


def causal_distribution_frame(frame, n_draws=MC_DRAWS, seed=MC_SEED,
                              progress_every: int = 25):
    """Fit distribution mechanics at each origin on PRIOR OOF outcomes only.

    Final serving maps may use all settled past OOF; NEVER apply those maps
    back to OOF rows. An undated frame is refused rather than claimed PIT.
    """
    if frame is None or not len(frame):
        return frame, {"method": family.METHOD, "maps": {}}
    original_index = frame.index
    date_key = "game_date" if DISTRIBUTION_SPORT == "mlb" else "gameday"
    identity_key = "game_pk" if DISTRIBUTION_SPORT == "mlb" else "game_id"
    frame = frame.sort_values([date_key, identity_key], kind="stable")
    h, a = _distribution_inputs(frame)
    outcomes = {"total": frame.home_score.to_numpy(float) + frame.away_score.to_numpy(float),
                "margin": frame.home_score.to_numpy(float) - frame.away_score.to_numpy(float)}
    if not all(np.isfinite(y).all() for y in outcomes.values()):
        raise ValueError("distribution evaluation requires settled finite outcomes")
    if not frame.index.is_unique:
        raise ValueError("distribution evaluation requires unique row indices")
    settled = frame[["home_score", "away_score"]].to_numpy(float)
    if np.any(settled < 0.) or np.any(settled != np.floor(settled)):
        raise ValueError("distribution outcomes must be nonnegative integer final scores")
    blocks = _distribution_blocks(frame)
    date_col = "game_date" if DISTRIBUTION_SPORT == "mlb" else "gameday"
    dates = pd.to_datetime(frame[date_col], utc=True).dt.normalize()
    cutoff = dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    pre_holdout = (dates < cutoff).to_numpy()
    history = np.zeros(len(frame), dtype=bool)
    parts, origins, issued_cdfs = [], [], []
    import time as _time
    _t0 = _time.time()
    n_blocks = len(blocks)
    logger.info("causal_distribution_frame: %d rows over %d origins "
                "(n_draws=%d)", len(frame), n_blocks, n_draws)
    evidence = {"total": [], "margin": []}
    # BASE-model pmfs (post-calibration + structural rules, PRE-lattice) of
    # already-priced prior blocks, per target: the causal evidence the next
    # origin's lattice residual is fit from. Never the row being priced.
    lattice_prior = {"total": [], "margin": []}

    def _lattice_evidence(include_sealed=False):
        # The sealed-holdout tail never trains the maps, so it never trains
        # the lattice fits at in-window origins either; the FINAL serving
        # fit (include_sealed=True) learns from all now-settled rows — the
        # same scope the serving variance/calibration maps use.
        def _select(kind):
            blocks = []
            for pmf, y, sup, pre in lattice_prior[kind]:
                keep = pre if not include_sealed else np.ones(len(pre), dtype=bool)
                if keep.any():
                    blocks.append((pmf[keep], y[keep], sup))
            return family.lattice_evidence(blocks)
        return {kind: _select(kind) for kind in ("total", "margin")}

    for start, end, mask in blocks:
        candidate = _distribution_fit_maps(h[history], a[history],
                                           {k: y[history] for k, y in outcomes.items()},
                                           _lattice_evidence())
        maps = dict(candidate)
        admitted = {}
        for kind in ("total", "margin"):
            differences = np.concatenate(evidence[kind]) if evidence[kind] else np.array([])
            # Paired losses are from previously issued, honest candidate and
            # identity CDFs. No same-origin fit performance selects the map.
            admitted[kind] = bool(len(differences) >= CRPS_MIN_HISTORY
                                  and differences.mean() < 0.)
            if not admitted[kind]: maps[kind] = [0.] * len(maps[kind])
        current = frame.iloc[np.flatnonzero(mask)]
        block_cdfs: dict = {}
        part = _distribution_price(current, maps, n_draws=n_draws, seed=seed,
                                   cdfs_out=block_cdfs)
        issued_cdfs.append(block_cdfs)
        baseline = dict(candidate,
                        total=[0.] * len(candidate["total"]),
                        margin=[0.] * len(candidate["margin"]))
        # The identity baseline prices the BASE family — structural rules
        # from the variance parameters, no lattice delta — so the admission
        # gate asks whether the whole candidate (calibration maps + lattice
        # residual) beats it on prior issued origins.
        baseline.pop("lattice", None)
        cc, bc = _distribution_cdfs(current, candidate), _distribution_cdfs(current, baseline)
        for kind in ("total", "margin"):
            support, cdf = cc[kind]
            difference = (lattice_crps(cdf, support, outcomes[kind][mask])
                          - lattice_crps(bc[kind][1], support, outcomes[kind][mask]))
            evidence[kind].append(difference[pre_holdout[mask]])
        # Accumulate this block's BASE-model pmfs for the NEXT origin's
        # lattice fit (the candidate minus its fitted delta).
        base_maps = dict(candidate)
        base_maps.pop("lattice", None)
        sb = _distribution_cdfs(current, base_maps)
        for kind in ("total", "margin"):
            support_b, cdf_b = sb[kind]
            lattice_prior[kind].append(
                (np.diff(np.column_stack([np.zeros(len(current)), cdf_b]), axis=1),
                 outcomes[kind][mask].copy(), support_b,
                 pre_holdout[mask].copy()))
        part["distribution_fit_n"] = int(history.sum())
        part["distribution_fit_end"] = str(origins[-1]["end"]) if origins else None
        parts.append(part)
        if progress_every and (len(origins) % progress_every == 0
                               or len(origins) == n_blocks):
            logger.info("causal_distribution_frame: origin %d/%d (%.1fs)",
                        len(origins), n_blocks, _time.time() - _t0)
        origins.append({"start": str(start.date()), "end": str(end.date()),
                        "n_prior": int(history.sum()), "n_scored": int(mask.sum()),
                        "maps": maps, "candidate_maps": candidate,
                        "calibration_admitted": admitted})
        history |= mask & pre_holdout
    # Freeze evaluation learning through the sealed tail; serving can learn
    # from all now-settled past games, but cannot rewrite issued OOF prices.
    maps = _distribution_fit_maps(h, a, outcomes, _lattice_evidence(True))
    for kind in ("total", "margin"):
        difference = np.concatenate(evidence[kind])
        if len(difference) < CRPS_MIN_HISTORY or difference.mean() >= 0.:
            maps[kind] = [0.] * len(maps[kind])
    bundle = {"method": family.METHOD, "family": family.FAMILY, "scope": "whole_cdf",
              "sport": DISTRIBUTION_SPORT, "objective": "discrete_crps",
              "maps": maps, "origins": origins, "fit_n": len(frame),
              "lattice_summary": {target: family.lattice_summary(
                  (maps.get("lattice") or {}).get(target) or {})
                  for target in family.LATTICE_TARGETS},
              "fit_end": str(dates.max().date()),
              "monte_carlo": parts[-1].attrs["mc_meta"],
              "holdout": {"cutoff": str(cutoff.date()), "n_pre": int(pre_holdout.sum()),
                          "n_holdout": int((~pre_holdout).sum()),
                          "holdout_days": HOLDOUT_DAYS,
                          "fitted_on": "strictly prior pre-holdout origins"},
              "serving_fit_scope": "all_settled_prior_oof_only",
              "admission": "negative_mean_paired_crps_on_prior_issued_origins"}
    priced = pd.concat(parts).reindex(original_index)
    priced.attrs["mc_meta"] = bundle["monte_carlo"]
    # Runtime-only, parallel to ``origins``: the issued sampled CDFs per
    # block for distribution_score_report. They ride the BUNDLE, never
    # frame attrs: pandas __finalize__ deep-copies attrs on EVERY op
    # (column getitem, iloc, arithmetic), so megabytes of nested CDF
    # lists in attrs make every downstream op O(payload) -- the 2026-10-22
    # remote run spent ~35 min in marketize+calibrate on this alone.
    # Never serialized (writers pluck named keys); calibrate_market_frame
    # drops it after the report.
    bundle["_issued_cdfs"] = issued_cdfs
    return priced, bundle


def distribution_score_report(frame, bundle):
    """Rebuild each origin's SAME CDF for full-support CRPS and binary scores."""
    from sklearn.metrics import roc_auc_score
    values = {"total": [], "margin": []}
    row_losses = {"total": np.empty(len(frame)), "margin": np.empty(len(frame))}
    for (_, _, mask), origin in zip(_distribution_blocks(frame), bundle["origins"]):
        part = frame.iloc[np.flatnonzero(mask)]
        mc = bundle.get("monte_carlo", {})
        cdfs = _distribution_cdfs(part, origin["maps"], sampled=True,
                                  n_draws=mc.get("requested_draws", MC_DRAWS),
                                  seed=mc.get("seed", MC_SEED))
        for kind, (support, cdf) in cdfs.items():
            y = (part.home_score + part.away_score if kind == "total"
                 else part.home_score - part.away_score).to_numpy(float)
            loss = lattice_crps(cdf, support, y)
            values[kind].append(loss)
            row_losses[kind][mask] = loss
    result = {"method": family.METHOD, "roi_verified": False}
    for kind, key, line_key, name in (("total", "p_over_fair", "fair_total", "totals"),
                                      ("margin", "p_cover_fair", "fair_spread", "run_line")):
        p = frame[key].to_numpy(float)
        y = (frame.home_score + frame.away_score if kind == "total"
             else frame.home_score - frame.away_score).to_numpy(float)
        event = y > frame[line_key].to_numpy(float)
        clipped = np.clip(p, 1e-7, 1 - 1e-7)
        ll = float(-np.mean(event * np.log(clipped) + (~event) * np.log1p(-clipped)))
        bins = np.minimum((p * 10).astype(int), 9)
        reliability = []
        error = 0.
        for b in np.unique(bins):
            m = bins == b
            gap = float(p[m].mean() - event[m].mean())
            error += float(m.mean()) * abs(gap)
            reliability.append({"n": int(m.sum()), "predicted": float(p[m].mean()),
                                "actual": float(event[m].mean()), "gap": gap})
        result[name] = {"n": len(p), "n_pushes": 0,
                        "crps": float(np.concatenate(values[kind]).mean()),
                        "auc": float(roc_auc_score(event, p)) if len(np.unique(event)) == 2 else None,
                        "logloss": ll, "logscore": ll,
                        "brier": float(np.mean((p - event) ** 2)),
                        "predicted_mean": float(p.mean()), "actual_win_rate": float(event.mean()),
                        "ece_cover": error, "reliability": reliability,
                        "calibration_in_the_large": float(p.mean() - event.mean()),
                        "threshold_policy": "closest_half_point_to_model_50_percent"}
    probability = frame.p_home_win_derived.to_numpy(float)
    tie = frame.p_tie.to_numpy(float)
    margin = (frame.home_score - frame.away_score).to_numpy(float)
    eligible = margin != 0
    conditional = np.clip(probability[eligible] / np.maximum(1. - tie[eligible], 1e-12), 1e-7, 1 - 1e-7)
    target = margin[eligible] > 0
    result["derived_moneyline"] = {
        "n": int(eligible.sum()), "n_ties": int((~eligible).sum()),
        "auc": float(roc_auc_score(target, conditional)) if len(np.unique(target)) == 2 else None,
        "logloss": float(-np.mean(target * np.log(conditional) + (~target) * np.log1p(-conditional))),
        "brier": float(np.mean((conditional - target) ** 2)),
        "settlement": "conditional_non_tie" if DISTRIBUTION_SPORT == "nfl" else "final_score_no_tie"}
    result["fixed_thresholds"] = {}
    for kind, prefix, lines in (
            ("total", "p_over", {"nba": [210., 220., 230., 240.],
                                  "nfl": [38., 44., 50.], "nhl": [5., 6., 7.],
                                  "mlb": [7.5, 8.5, 9.5]}[DISTRIBUTION_SPORT]),
            ("margin", "p_home_cover", [-1.5, .5, 1.5] if DISTRIBUTION_SPORT in ("mlb", "nhl")
                                       else [-5.5, .5, 5.5])):
        outcome = (frame.home_score + frame.away_score if kind == "total"
                   else frame.home_score - frame.away_score).to_numpy(float)
        for line in lines:
            key = (f"{prefix}_{str(float(line)).replace('.', '_')}" if DISTRIBUTION_SPORT == "mlb"
                   else _grid_key(prefix, line))
            if key not in frame: continue
            p = np.clip(frame[key].to_numpy(float), 1e-7, 1 - 1e-7)
            event = outcome > line
            bins = np.minimum((p * 10).astype(int), 9)
            reliability = []
            for b in np.unique(bins):
                m = bins == b
                reliability.append({"n": int(m.sum()), "predicted": float(p[m].mean()),
                                    "actual": float(event[m].mean())})
            result["fixed_thresholds"][f"{kind}_{line}"] = {
                "n": len(p), "auc": float(roc_auc_score(event, p)) if len(np.unique(event)) == 2 else None,
                "logloss": float(-np.mean(event * np.log(p) + (~event) * np.log1p(-p))),
                "brier": float(np.mean((p - event) ** 2)), "reliability": reliability,
                "event": "strict_exceedance_including_push_as_nonwin"}
    result["conditional_crps"] = {}
    # All strata remain visible including small postseason groups; no exclusion
    # based on how a regime happens to perform.
    if "season" in frame:
        for season in pd.unique(frame.season.dropna()):
            mask = frame.season.eq(season).to_numpy()
            result["conditional_crps"][f"season_{season}"] = {
                "n": int(mask.sum()), **{k: float(v[mask].mean()) for k, v in row_losses.items()}}
    for column in ("is_playoffs", "provisional"):
        if column in frame:
            for flag in (False, True):
                mask = frame[column].fillna(False).astype(bool).eq(flag).to_numpy()
                if mask.any():
                    result["conditional_crps"][f"{column}_{flag}"] = {
                        "n": int(mask.sum()), **{k: float(v[mask].mean()) for k, v in row_losses.items()}}
    if DISTRIBUTION_SPORT in ("nba", "nfl"):
        # V3a tilt watch (NBA adopted 2026-10-18, NFL 2026-10-18): slope-sign
        # guard on the serving fit plus the top-quintile over-line gap panel
        # (over-230 NBA, over-47 NFL), so the bias the tilt was adopted to
        # fix stays watched every run. Absent tilt block (flag off) still
        # reports the gap; other sports skip this.
        _hot_line = 230. if DISTRIBUTION_SPORT == "nba" else 47.
        _hot_key = f"p_over_{int(_hot_line)}"
        tilt, gap = {}, {}
        serving_var = (bundle.get("maps") or {}).get("variance") or {}
        block = serving_var.get("total_tilt") or {}
        if block:
            slopes = []
            for origin in bundle.get("origins") or []:
                ob = ((origin.get("maps") or {}).get("variance") or {}).get("total_tilt") or {}
                if ob.get("slope") is not None:
                    slopes.append(float(ob["slope"]))
            tilt = {"slope": round(float(block["slope"]), 3),
                    "intercept": round(float(block["intercept"]), 3),
                    "sign_ok": bool(float(block["slope"]) > 0),
                    "origin_slopes": [round(s, 3) for s in slopes],
                    "origin_sign_flips": int(sum(1 for s in slopes if s <= 0))}
            if not tilt["sign_ok"]:
                logger.warning("totals tilt slope flipped non-positive (%.3f); "
                               "the pace-tilt correction is suspect", block["slope"])
        if "mu_h" in frame and "mu_a" in frame and _hot_key in frame:
            exp = frame.mu_h.to_numpy(float) + frame.mu_a.to_numpy(float)
            ok = np.isfinite(exp)
            if int(ok.sum()) >= 100:
                q80 = float(np.quantile(exp[ok], 0.8))
                hot = ok & (exp >= q80)
                if int(hot.sum()) >= 20:
                    p = np.clip(frame[_hot_key].to_numpy(float)[hot], 1e-7, 1 - 1e-7)
                    y = ((frame.home_score + frame.away_score).to_numpy(float)[hot] > _hot_line).astype(float)
                    gap = {"n": int(hot.sum()), "q80": round(q80, 2),
                           "predicted": round(float(p.mean()), 4),
                           "actual": round(float(y.mean()), 4),
                           "gap": round(float(y.mean() - p.mean()), 4)}
        result["tilt_watch"] = {"tilt": tilt, "top_quintile_over_230": gap}
    return result


def distribution_probability(frame, bundle, kind, line):
    """Exact win/push/loss at ANY integer/half line from serving CRPS CDF.

    line is a margin threshold, not a signed sportsbook home handicap:
    home handicap -3.5 maps to margin threshold +3.5.
    """
    if kind not in ("total", "margin") or not np.isfinite(line):
        raise ValueError("expected total/margin and finite line")
    mc = bundle.get("monte_carlo", {})
    support, cdf = _distribution_cdfs(frame, bundle.get("maps", {}), sampled=True,
                                     n_draws=mc.get("requested_draws", MC_DRAWS),
                                     seed=mc.get("seed", MC_SEED))[kind]
    def at(value):
        index = int(np.floor(value) - support[0])
        if index < 0: return np.zeros(len(frame))
        if index >= len(support): return np.ones(len(frame))
        return cdf[:, index]
    upper, lower = at(line), at(np.ceil(line) - 1.)
    return {"win": 1. - upper, "push": np.maximum(upper - lower, 0.), "loss": lower}


if __name__ == "__main__":
    main()
