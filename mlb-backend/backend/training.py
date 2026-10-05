"""
Walk-forward training for MLB Bet Predictor.

Implements expanding-window walk-forward splits, multi-target heads
(moneyline, totals, run line), evaluation metrics, and ensemble persistence.
"""
from __future__ import annotations

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from frames import get_decided_frame, fold_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from calibration import (
    is_identity,
    MIN_OOF_FOR_FIT,
    moneyline_apply,
    moneyline_fit,
    should_gate_calibrator,
)
from config import (
    ADAPTIVE_WEIGHT_METRIC,
    BLEND_SPACE,
    DATA_DELIVERY_DIR,
    DATE_FMT,
    ENSEMBLE_FILE,
    ENSEMBLE_WEIGHTS,
    LIGHTGBM_PARAMS,
    MIN_VAL_FOLD_GAMES,
    MODELS_DIR,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
    RFE_CANDIDATE_COLS,
    VERSION_KEY,
    TRAINED_AT_KEY,
    DATA_CUTOFF_KEY,
    ELASTICNET_PARAMS,
    XGBOOST_PARAMS,
)

logger = logging.getLogger(__name__)

# Run-margin feature column (λ_home − λ_away from the run engine's per-side
# Poisson models, out-of-fold on the moneyline's own split). Kept in sync
# with build_oof_margin.MARGIN_COL; the value lives here so feature_metadata
# and the run-engine derivation both stay in one canonical place.
MARGIN_COL = "run_margin_diff"

# Median early-stopping round counts (home/away) from the most recent OOF
# margin build inside walk_forward_evaluate. The slate/inference path reuses
# them for its fit-only refit so a fresh run-engine OOF is not needed every
# day; None until the first walk-forward margin build.
_LAST_MARGIN_ROUNDS: dict | None = None


def set_last_margin_rounds(rounds: dict | None) -> None:
    """Record the median fold round counts used for the OOF margin build."""
    global _LAST_MARGIN_ROUNDS
    _LAST_MARGIN_ROUNDS = dict(rounds) if rounds else None


def get_last_margin_rounds() -> dict | None:
    """Median fold round counts from the most recent margin build (or None)."""
    return dict(_LAST_MARGIN_ROUNDS) if _LAST_MARGIN_ROUNDS else None


# Walk-forward splits produced during the most recent training pass.
# _attach_drift_run_margins reuses these to avoid building splits on
# a different row set (the drift frame can differ from the training
# frame), which would desync fold geometry and trigger the
# _attach_oof_run_margins desync guard.
_LAST_WALK_FORWARD_SPLITS: list = []


def set_last_walk_forward_splits(splits: list) -> None:
    """Record the walk-forward splits from the most recent training pass."""
    global _LAST_WALK_FORWARD_SPLITS
    _LAST_WALK_FORWARD_SPLITS = list(splits)


def get_last_walk_forward_splits() -> list:
    """Walk-forward splits from the most recent training pass (empty list if none)."""
    return list(_LAST_WALK_FORWARD_SPLITS)


# Fold signature of the most recent training pass's canonical decided frame
# (frames.fold_signature). The pipeline's drift/coverage steps assert theirs
# against this so a decided-frame divergence fails loudly at the historically
# desyncing points instead of shifting fold boundaries silently.
_LAST_FOLD_SIGNATURE: str | None = None


def set_last_fold_signature(sig: str | None) -> None:
    """Record the fold signature of the most recent training pass."""
    global _LAST_FOLD_SIGNATURE
    _LAST_FOLD_SIGNATURE = sig


def get_last_fold_signature() -> str | None:
    """Fold signature from the most recent training pass (None if none)."""
    return _LAST_FOLD_SIGNATURE


# Features used for model input — all diff/computed features.
# Diff convention: home − away (positive = home advantage).
#
# Pruned 2026-08-23 (feature_audit):
#   wind/air_density factors         constant 0 (weather backfill covers only
#                                    ~7% of games; starved, not broken —
#                                    re-add when coverage improves)
#   bullpen_ip_diff                  r=0.87 with bullpen_pitches_diff, weaker
#
# Wave-2 ablation candidates (univariate |lift| < ~0.01, need walk-forward
# retrain to confirm): sp_fbpct_diff, team_barrel_diff, lineup_re24_std_diff,
# park_factor_slug_diff, closer_availability_diff, travel_fatigue_diff,
# bullpen_whip_3g_diff, ace_efficiency_factor, pitcher_regression_indicator.
MONEYLINE_FEATURE_COLS = [
    # 1. Baseline (home-field anchor; constant by construction)
    "is_home",
    # 2–4. Core pre-game diffs
    "win_pct_diff",
    "elo_diff",
    "rest_days_diff",
    # 5–8. Starting pitcher diffs (season-to-date + last-5-start)
    "sp_era_diff",
    "sp_era_5g_diff",
    "sp_k9_diff",
    "sp_k9_5g_diff",
    # 9–11. SP stuff diffs (trailing 3-game)
    "sp_fbvelo_diff",
    "sp_fbpct_diff",
    "sp_whiff_diff",
    # 10–11. SP xwOBA diffs
    "sp_xwoba_diff",
    "sp_xwoba_vs_l_diff",
    # 12–14. Lineup RE24 diffs (swapped from lineup wOBA 2026-10-02)
    "lineup_re24_mean_diff",
    "lineup_re24_top3_diff",
    "lineup_re24_std_diff",
    # 15. Team rolling wOBA diff
    "woba_30g_diff",
    # 16–18. Bullpen diffs (whip_diff RENAMED 2026-09-30: the name now
    # carries its window; values unchanged — it was always the 10g form)
    "bullpen_whip_10g_diff",
    "bullpen_whip_3g_diff",
    "bullpen_pitches_diff",
    # 19–21. Team contact form diffs (trailing 15g)
    "team_barrel_diff",
    "team_hardhit_diff",
    "team_exitvelo_diff",
    # 22. Lineup handedness matchup advantage (OPS vs tonight's starter hand)
    "lineup_handedness_matchup_advantage",
    # 23. Travel fatigue (timezone crossings, last 3 days)
    "travel_fatigue_diff",
    # 24. Closer availability
    "closer_availability_diff",
    # 25. Dome neutral flag
    "dome_is_neutral",
    # 26. Park factor
    "park_factor_slug_diff",
    # 27–28. Weather-driven interactions from REAL Open-Meteo observations
    # (validated 2026-08: varied, venue-sane values incl. Coors thin-air).
    # Missing observations stay NULL; dome wind is a valid neutral 0.
    "wind_advantage_flyball_factor",
    "air_density_velocity_boost",
    # 29–32. Derived interaction features (meltdown_diff RENAMED from
    # bullpen_meltdown_risk 2026-09-30; per-side twins added — the
    # within-side product of the family's own factors)
    "bullpen_meltdown_risk_diff",
    "bullpen_meltdown_risk_home",
    "bullpen_meltdown_risk_away",
    # RENAMED 2026-09-27 (structural): every model-side feature ends in
    # _diff. The three interaction composites take their diff names plus
    # per-side twins (the within-side product of their own factors —
    # home − away of the pair equals the diff only up to cross terms, so
    # the pair is the interaction's raw representation, not its halves).
    "pitcher_regression_indicator_diff",
    "pitcher_regression_indicator_home",
    "pitcher_regression_indicator_away",
    "lineup_depth_multiplier_diff",
    "lineup_depth_multiplier_home",
    "lineup_depth_multiplier_away",
    "ace_efficiency_factor_diff",
    "ace_efficiency_factor_home",
    "ace_efficiency_factor_away",
    # 33–56. Raw per-side inputs (home/away pre-differenced values).
    # Gives every member the raw home and away values alongside their diffs,
    # letting tree members discover side-specific thresholds and interactions
    # that a pure home-minus-away diff cannot express.
    # Elo
    "home_elo",
    "away_elo",
    # Win percentage
    "home_win_pct",
    "away_win_pct",    # SP K/9 (season-to-date)
    "sp_k9_home", "sp_k9_away",
    # SP xwOBA allowed (last 6 starts)
    "sp_xwoba_home", "sp_xwoba_away",
    # Team 30-game wOBA
    "woba_30g_home",
    "woba_30g_away",
    # Bullpen 10-game WHIP
    "bullpen_whip_10g_home",
    "bullpen_whip_10g_away",
    # Bullpen 3-game WHIP
    "bullpen_whip_3g_home",
    "bullpen_whip_3g_away",
    # Team barrel% (15-game)
    "team_barrel_15g_home",
    "team_barrel_15g_away",
    # Team exit velocity (15-game)
    "team_exitvelo_15g_home",
    "team_exitvelo_15g_away",
    # Momentum form deltas (recent window − season-to-date baseline, per side)
    # are NOT in the active moneyline set: the 2026-08 ablation measured them
    # negative (WITH loses BOTH pooled OOF and the sealed 21-day holdout vs the
    # 58-column baseline; see run_form_delta_ablation.py and
    # data_delivery/form_delta_ablation_<sha>.json). The 38 *_delta_* columns
    # are still computed and shipped in the artifact (features.py SQL +
    # add_form_delta_features, metadata authored) but excluded from
    # MONEYLINE_FEATURE_COLS — re-enabling is a one-line append here
    # after a re-test on a refreshed artifact.
    # 57–62. Phase 2 lineup-delta features — REMOVED 2026-08-29: train-serve
    # skew fix. These 6 features (lineup_actual_woba_delta_home/away,
    # lineup_actual_top3_delta_home/away, lineup_rest_count_home/away) are
    # populated from post-game ACTUAL lineups in the decided frame (~52-60%
    # measured) but are ALWAYS zero/imputed at prediction time (slate path
    # has only projected lineups). The model learned signal that cannot exist
    # at bet time — a classic train-serve skew / leakage. The prior pruning
    # ablation's LOW_COVERAGE arm lost OOF AUC only because that AUC was
    # inflated by the leaked actuals (false negative). The run engine's
    # 29-feature view already excludes these (RUN_EXTRA_EXCLUSIONS), so no
    # prediction-path change. See the prior ablation:
    # data_delivery/lineup_ablation_<sha>.json; coverage-divergence test
    # failure was the symptom this fixed.
    # 63. Run-engine expected-run margin — SHIPPED 2026-08-26: the ablation
    # cleared the sealed-21-day-holdout gate (holdout logloss 0.6774 → 0.6765,
    # AUC 0.5780 → 0.5786; ECE-cal 0.0626 → 0.0554; pooled OOF logloss
    # 0.6839 → 0.6830 / AUC 0.5669 → 0.5694; see run_margin_ablation.py and
    # data_delivery/margin_ablation_<sha>.json). One column, computed
    # OUT-OF-FOLD on the MONEYLINE'S OWN fold split: λ_home − λ_away from the
    # run engine's per-side LightGBM Poisson models (its unchanged per-side
    # levels+env view), so no game's margin ever comes from a model that saw
    # it. Computed at training time by _attach_oof_run_margins() in
    # walk_forward_evaluate; slate margins come from a fit-only refit on all
    # decided games at the median fold round count (pipeline._attach_slate_
    # run_margins). The run engine itself can never consume it through the
    # moneyline list: run_margin_diff is not a member of
    # MONEYLINE_FEATURE_COLS (and the ablation's LAMBDAS variant showed
    # λ_home/λ_away add nothing beyond the margin).
    # Home-edge interaction ablation (DON'T ADOPT, 2026-08-27): the structural
    # finding that home edge is environment-conditional (+0.27 low-total vs
    # -0.09 high-total) is NOT recoverable through the run engine's expected
    # total. Expected-total variants (expected_total = lam_home+lam_away,
    # run_margin_x_exp_total product, high_expected_total median bucket) added
    # NO separation beyond the main effects at the pooled-OOF logistic
    # pre-check (ΔAUC +0.0003, CI [-0.0008, +0.0010] straddles 0; Δlogloss
    # 0.0000; interaction coef -0.0036), so the heavy walk-forward gate was
    # skipped. The +0.27/-0.09 diagnostic was on ACTUAL totals (post-game); the
    # leakage-free expected total carries far weaker environment signal. See
    # run_home_edge_interaction_ablation.py and
    # data_delivery/home_edge_interaction_ablation_20260827.json.
    # REMOVED 2026-09-17 (structural decoupling): run_margin_diff is the run
    # engine's own model output (lambda_home - lambda_away). Both engines now
    # share the same feature universe, so keeping the run engine's prediction
    # inside the moneyline let the run-line model indirectly feed (and
    # correlate with) the moneyline it shares inputs with. Removal is purely
    # an output-list change: the column stays COMPUTED and drift-monitored
    # (the attach guards in pipeline.py / training.py key on MARGIN_COL
    # membership and now skip cleanly), and the run engine's NB pricing is
    # untouched. The live adopted RFE subset was re-issued
    # 62-col alongside this edit so apply_adopted_subset() keeps validating
    # (an out-of-pool name rejects the WHOLE state at startup, which would
    # silently drop the shipped closer pair from serving width). Width
    # note: the closer pair was PROMOTED into this list 2026-09-17
    # (appended, same trailing position as the adopted state) so the
    # universe is once again the literal production width: 61 -> 63
    # (promotion) -> 62 (this removal). Universe = active serving width =
    # adopted state = 62; the next retrain/persist seals the 62-col
    # matrix, and the bundle-fallback width now retains the closers.
    # 60–67. Experiment #2 matchup candidates — SHIPPED 2026-09-07 per the
    # frozen C+E (moneyline) / D+F (true −1.5 run line) implementation
    # decision. NOT a new selection exercise: all 8 candidates ship for
    # consistency, and NO baseline family was removed (the targeted-removal
    # evidence in the decision only gated adoption; expansion ≠ replacement).
    # sp_xwoba_vs_r_diff was confirmed ABSENT from the original baseline and
    # is deliberately NOT backfilled. Candidates are computed by
    # features.add_exp2_features from the Experiment #2 source layer's
    # PIT-safe columns (league priors, category K%/xwOBA, platoon splits);
    # every input satisfies source_game_date < target_game_date (date-level
    # ASOF, doubleheader-safe), so no new temporal exposure. Formulas frozen
    # from run_exp2_feature_test.add_candidates; adoption evidence:
    # data_delivery/exp2_feature_test_20260907.json +
    # exp2_stability_20260907.json. All 8 end in _diff and are matchup gaps
    # that the run engine's per-side view splits into side columns — the run
    # engine (NB pricing) prices each side from its own columns; the run-line
    # model is the true −1.5 classifier below.
    "exp2_centered_k_diff",
    "exp2_cat_k_fastball_diff",
    "exp2_cat_k_breaking_diff",
    "exp2_cat_k_offspeed_diff",
    "exp2_cat_xwoba_fastball_diff",
    "exp2_cat_xwoba_breaking_diff",
    "exp2_cat_xwoba_offspeed_diff",
    "exp2_cat_platoon_k_fastball_diff",
]

# CORRECTED 2026-09-07: the intended Experiment #2 operation was a feature
# REPLACEMENT, not a pure expansion — ADD C/D (the 8 exp2 candidates above)
# and REMOVE the E/F targeted-removal families. The removal sets resolve
# (from data_delivery/exp2_feature_test_20260907.json 's_families' +
# 'candidate_s_map', the frozen experiment registry) to 6 UNIQUE baseline
# features; sp_k9_diff / sp_k9_5g_diff appear in both S_K and S_PLATOON and
# are counted once. Both models' removal unions are identical, so one list
# serves moneyline and run line alike:
#   S_K        = {sp_k9_diff, sp_k9_5g_diff}          (centered, platoon)
#   S_FB_K     = {sp_fbpct_diff, sp_whiff_diff}       (category K ×3)
#   S_XWOBA    = {sp_xwoba_diff, sp_xwoba_vs_l_diff}  (category xwOBA ×3)
# sp_xwoba_vs_r_diff stays absent (never in the original baseline).
# The raw per-side source columns and every OTHER baseline feature are
# preserved; the run engine's λ view (run_engine.py RUN_FEATURE_COLS) is a
# separate list and is untouched — NB pricing behavior unchanged.
#
# REVERTED 2026-09-30 (user-directed structural alignment): sp_k9_diff
# rejoins the universe — its raw per-side twins sp_k9_home/away re-entered
# serving on 2026-09-27 (adopted state + provenance rework), so serving the
# raw pair while withholding the diff is no longer coherent. Dropping it
# here does NOT rewrite the frozen 09-07 experiment registry
# (exp2_feature_test_20260907.json still records the original 6-name union);
# sp_k9_5g_diff and the four remaining E/F diffs stay removed. This is a
# serving-width change made outside an RFE trial: the adopted state
# (mlb_feature_selection_state.json) was re-issued with the column inserted
# at its canonical universe position, and the width pins moved 100 -> 101.
_EXP2_REMOVALS = [
    "sp_k9_5g_diff",
    "sp_fbpct_diff",
    "sp_whiff_diff",
    "sp_xwoba_diff",
    "sp_xwoba_vs_l_diff",
]
MONEYLINE_FEATURE_COLS = [c for c in MONEYLINE_FEATURE_COLS if c not in _EXP2_REMOVALS]
# Deduplicate (should already be unique but defensive)
MONEYLINE_FEATURE_COLS = list(dict.fromkeys(MONEYLINE_FEATURE_COLS))

# PROMOTED 2026-09-17: closer_available_home/away join the universe so
# the generation list is once again the literal production serving
# width. They reached serving 2026-09-16 via the adopted RFE state
# (ea2c7e0) as candidate columns -- never members of this list -- which
# left the universe (the fallback serving width) two features behind
# the production model. Appended to match the adopted state's trailing
# order, so set_feature_subset's canonical reordering produces zero
# positional churn for the 62-col matrix.
MONEYLINE_FEATURE_COLS += ["closer_available_home", "closer_available_away"]

# EXPANDED 2026-09-27 (structural, mirrors the NHL d83e0c1 per-side twin
# rollout): every served diff family now also exposes its raw home/away
# halves, so the tree members see the levels the matchup gaps summarize.
#   * 12 level twins (rest_days, sp_era_5g, sp_fbvelo_3g, lineup_re24_std,
#     bullpen_pitches_3d, team_hardhit_15g) + the travel twins are the
#     diff pass's OWN input columns — the same strictly-prior source, so
#     home − away == diff by construction and no second derivation can
#     drift.
#   * 16 exp2 twins are the per-side scratch add_exp2_features always
#     computed and dropped; now served under final names from the same
#     frozen arithmetic.
#   * The 6 interaction twins above are the within-side products.
# Twin coverage == diff coverage minus source availability: the exp2
# offspeed family rides the documented 0.567 source floor (twins DOMINATE
# their diff, so a diff can never be better covered than its halves).
# Routing stays untouched: RAW_PER_SIDE_COLS (below) routes all 38 new
# twins tree-only; the logistic member keeps its diffs-only view and the
# run engine's λ view prices each side from its own side columns (split_side_view).
# Universe 62 → 98.
MONEYLINE_FEATURE_COLS += [
    # Raw per-side levels for the remaining served diff families
    "rest_days_home", "rest_days_away",
    "sp_era_5g_home", "sp_era_5g_away",
    "sp_fbvelo_3g_home", "sp_fbvelo_3g_away",
    "lineup_re24_std_home", "lineup_re24_std_away",
    "bullpen_pitches_3d_home", "bullpen_pitches_3d_away",
    "team_hardhit_15g_home", "team_hardhit_15g_away",
    "time_zones_crossed_last_3d_home", "time_zones_crossed_last_3d_away",
    # Experiment #2 per-side halves (the diffs' own scratch, served; the
    # literal contract mirrors features.EXP2_TWIN_COLS — same frozen names,
    # owned here so the serving universe never depends on a builder import)
    "exp2_centered_k_home", "exp2_centered_k_away",
    "exp2_cat_k_fastball_home", "exp2_cat_k_fastball_away",
    "exp2_cat_k_breaking_home", "exp2_cat_k_breaking_away",
    "exp2_cat_k_offspeed_home", "exp2_cat_k_offspeed_away",
    "exp2_cat_xwoba_fastball_home", "exp2_cat_xwoba_fastball_away",
    "exp2_cat_xwoba_breaking_home", "exp2_cat_xwoba_breaking_away",
    "exp2_cat_xwoba_offspeed_home", "exp2_cat_xwoba_offspeed_away",
    "exp2_cat_platoon_k_fastball_home", "exp2_cat_platoon_k_fastball_away",
]

# ── Position-pool adoption (2026-10-03): the pl_[pos] + removals plan ───────
# add 24 (8 pools × home/away/diff), replace 9, remove 7:
#   NEW      24  pl_<pos>_xwoba_{home,away,diff} for c/fb/sb/ss/tb/rf/cf/lf
#                (the plan sheet's 8 pools; pl_dh is generated in the frame
#                 but NOT served)
#   REPLACE   9  the lineup re24 family (mean/top3/std × diff/home/away)
#   REMOVE    7  lineup_depth_multiplier ×3, the plain sp_era trio ×3, and
#                park_factor_slug_diff (built on lineup_re24_top3_diff)
# → 101 − 9 − 7 + 24 = 109. RAW_PER_SIDE 60 − 10 + 16 = 66 (the 10
# departed levels out, the 16 pl levels in); the logistic slice
# 41 − 6 + 8 = 43 — pl levels tree-only, pl diffs join the slice. The
# removed columns remain generated (features.py) and move to
# config.RFE_CANDIDATE_COLS so RFE may re-trial them — the 2026-09-07
# _EXP2_REMOVALS pattern.
_PL_PLAN_REMOVALS = [
    # replace: the lineup re24 family (9)
    "lineup_re24_mean_diff",
    "lineup_re24_top3_diff",
    "lineup_re24_std_diff",
    "lineup_re24_mean_home",
    "lineup_re24_mean_away",
    "lineup_re24_top3_home",
    "lineup_re24_top3_away",
    "lineup_re24_std_home",
    "lineup_re24_std_away",
    # remove: the lineup depth interaction (3) — mean × top3, both re24
    "lineup_depth_multiplier_diff",
    "lineup_depth_multiplier_home",
    "lineup_depth_multiplier_away",
    # remove: the plain SP ERA trio (3) (sp_era_5g_* stays)
    "sp_era_diff",
    "sp_era_home",
    "sp_era_away",
    # remove: park slugging × re24 top3 (the 7th removal — consumes the
    # re24 family that just left)
    "park_factor_slug_diff",
]
MONEYLINE_FEATURE_COLS = [c for c in MONEYLINE_FEATURE_COLS
                          if c not in _PL_PLAN_REMOVALS]
# NEW 2026-10-03: the eight position-pool xwOBA families, each serving its
# two levels (tree-only via RAW_PER_SIDE_COLS) plus the matchup diff.
# pl_dh is generated in the frame but NOT served (the plan sheet lists 8
# pools). Literal names, owned here so the serving universe never depends
# on a builder import (same contract as the exp2 twins above).
MONEYLINE_FEATURE_COLS += [
    "pl_c_xwoba_home", "pl_c_xwoba_away", "pl_c_xwoba_diff",
    "pl_fb_xwoba_home", "pl_fb_xwoba_away", "pl_fb_xwoba_diff",
    "pl_sb_xwoba_home", "pl_sb_xwoba_away", "pl_sb_xwoba_diff",
    "pl_ss_xwoba_home", "pl_ss_xwoba_away", "pl_ss_xwoba_diff",
    "pl_tb_xwoba_home", "pl_tb_xwoba_away", "pl_tb_xwoba_diff",
    "pl_rf_xwoba_home", "pl_rf_xwoba_away", "pl_rf_xwoba_diff",
    "pl_cf_xwoba_home", "pl_cf_xwoba_away", "pl_cf_xwoba_diff",
    "pl_lf_xwoba_home", "pl_lf_xwoba_away", "pl_lf_xwoba_diff",
]
MONEYLINE_FEATURE_COLS = list(dict.fromkeys(MONEYLINE_FEATURE_COLS))

# ── Known feature pool (RFE trial space) ────────────────────────────────────
# KNOWN_FEATURE_COLS = the generation universe plus every RFE candidate
# column (config.RFE_CANDIDATE_COLS, generated PIT-safe, pre-game, not in the
# universe). It is the complete space feature_selection.py may trial:
# removals come from the universe half, additions from the candidate half,
# and an adopted record may promote candidates into serving width — so the
# subset validator must accept the full pool, not just the universe. Order:
# universe first (canonical), then candidates in config group-priority order;
# set_feature_subset always rebuilds subsets in this order so every
# consumer's positional assumptions hold. Nothing is ever REMOVED from this
# pool — RFE verdicts govern serving width, never candidacy.
KNOWN_FEATURE_COLS = list(dict.fromkeys(
    list(MONEYLINE_FEATURE_COLS) + list(RFE_CANDIDATE_COLS)))


# ── Active feature subset (RFE-controlled) ──────────────────────────────────
# MONEYLINE_FEATURE_COLS above is the canonical GENERATION universe and is
# NEVER rebound after import (run_engine.py and other modules import the name
# at load time — rebinding would silently desync them). Serving/training use
# the ACTIVE subset returned by active_moneyline_feature_cols(): identical to
# the universe until feature_selection.apply_adopted_subset() applies an
# adopted RFE record (data_delivery/mlb_feature_selection_state.json) at the
# start of a retrain. Generation-side code (features.py construction, the
# pipeline's universe assertions, run-engine derivation, data_ingestion's
# frame skeleton) keeps reading the literal universe.
_FEATURE_SUBSET: list[str] | None = None


def active_moneyline_feature_cols() -> list[str]:
    """Model-facing feature list: adopted RFE subset, else the full universe."""
    return list(_FEATURE_SUBSET) if _FEATURE_SUBSET is not None \
        else list(MONEYLINE_FEATURE_COLS)


def set_feature_subset(cols: list[str] | None) -> None:
    """Apply (or clear, None) the active feature subset.

    Canonical pool order (KNOWN_FEATURE_COLS: universe then candidates) is
    preserved regardless of ``cols`` ordering, so every consumer's positional
    assumptions hold. Raises on non-pool names — a subset referencing an
    unknown feature is an upstream bug, not something to silently intersect
    away. The pool (not just the universe) is the validator because an
    adopted RFE record may promote candidate columns into serving width.
    """
    global _FEATURE_SUBSET
    if cols is None:
        _FEATURE_SUBSET = None
        return
    unknown = [c for c in cols if c not in KNOWN_FEATURE_COLS]
    if unknown:
        raise ValueError(
            f"feature subset contains non-pool columns: {unknown[:6]}"
            f" (known pool has {len(KNOWN_FEATURE_COLS)} entries)")
    keep = set(cols)
    _FEATURE_SUBSET = [c for c in KNOWN_FEATURE_COLS if c in keep]


def reset_feature_subset() -> None:
    """Clear any active subset (back to the full universe)."""
    global _FEATURE_SUBSET
    _FEATURE_SUBSET = None


# ── Walk-forward splits ─────────────────────────────────────────────────────

def postseason_flag(values) -> np.ndarray:
    """Row mask for postseason games in a game-type-like column.

    MLB's Statcast ``game_type`` codes regular season R and the four
    postseason rounds F/D/L/W (matching ingestion.KEEP_GAME_TYPES, plus the
    StatsAPI schedule-level code P); a loose text match keeps the mask
    honest if the vocabulary ever widens ("postseason"/"playoff"). Frames
    without the column (synthetic test frames) yield an all-False mask —
    regular by construction.
    """
    s = pd.Series(list(values)).astype(str).str.strip().str.upper()
    if s.empty:
        return np.zeros(0, dtype=bool)
    code = s.isin({"F", "D", "L", "W", "P"})
    loose = s.str.contains("POST|PLAY", regex=True, na=False)
    return (code | loose).to_numpy()


def _season_type(is_post: np.ndarray) -> str:
    """Window label from a row-level postseason mask."""
    is_post = np.asarray(is_post, dtype=bool)
    if not is_post.any():
        return "regular"
    if is_post.all():
        return "postseason"
    return "mixed"


def walk_forward_splits(
    games: pd.DataFrame,
    retrain_cadence_days: int = RETRAIN_CADENCE_DAYS,
    max_eval_folds: int = 0,
    min_train_days: int = 0,
) -> list[dict[str, Any]]:
    """Generate expanding-window walk-forward train/val splits.

    Each validation window is `retrain_cadence_days` wide. The training set
    is all games strictly before the validation window start. Windows are
    non-overlapping and chronological.

    Args:
        min_train_days: Skip validation windows that start before this many
            calendar days of history. Prevents tiny-training-set folds from
            polluting pooled metrics with noise (default 0 = no warm-up).

    Returns a list of dicts with keys:
        train_games: DataFrame of training games
        val_games: DataFrame of validation games
        fold_idx: int
        val_start: datetime
        val_end: datetime
        is_partial_tail: True ONLY for the final fold when its window runs
            partially into the frame's tail (short of a full cadence).
            Retention no longer depends on it — every non-empty window is
            retained — but the flag stays as tail provenance
            (still leakage-free: train < val_start).
        season_type: "regular" / "postseason" / "mixed" derived from the
            window's game_type rows (F/D/L/W/P = postseason; frames without
            game_type are "regular").
    """
    if "game_date" not in games.columns:
        raise ValueError("games must have a 'game_date' column")
    if "home_win" not in games.columns:
        logger.warning("walk_forward_splits: no 'home_win' column — cannot split")
        return []

    df = games.dropna(subset=["home_win"]).copy()
    if df.empty:
        logger.warning(
            "walk_forward_splits: all %d rows have NaN home_win — cannot split",
            len(games),
        )
        return []
    df["game_date"] = pd.to_datetime(df["game_date"])
    # Normalize to date-only (strip time) so unique dates represent calendar days,
    # not individual timestamps. Without this, each game with a unique start time
    # becomes its own "date" and 7-day validation windows collapse to 1 game.
    df["game_date"] = df["game_date"].dt.normalize()
    df = df.sort_values("game_date").reset_index(drop=True)

    if df.empty:
        return []

    unique_dates = sorted(df["game_date"].unique())
    if len(unique_dates) < retrain_cadence_days + 1:
        # Not enough data for even one split — use all as train, none as val
        logger.warning(
            "walk_forward_splits: only %d unique dates (need >= %d for cadence %d)",
            len(unique_dates), retrain_cadence_days + 1, retrain_cadence_days,
        )
        return []

    splits = []
    fold_idx = 0

    # Start validation windows after the warm-up period so every fold has
    # enough training history to produce meaningful predictions.
    val_start_idx = max(retrain_cadence_days, min_train_days)
    while val_start_idx < len(unique_dates):
        val_start = unique_dates[val_start_idx]
        val_end_idx = min(val_start_idx + retrain_cadence_days, len(unique_dates))
        val_end = unique_dates[val_end_idx - 1]

        # A FINAL fold whose full cadence window would overrun the frame's
        # max game_date runs PARTIALLY into the tail (val_end = frame max)
        # instead of being dropped. Consumers (walk_forward_evaluate,
        # _attach_oof_run_margins) use this flag to keep that one fold even
        # when it falls under the min-val gate, so the last decided day's
        # OOF predictions surface (the tail rows are never in the final
        # fold's TRAIN set — train stays strictly < val_start — so keeping
        # the partial fold is leakage-free).
        is_partial_tail = val_end_idx < val_start_idx + retrain_cadence_days

        # Training: everything strictly before val_start
        train_mask = df["game_date"] < val_start
        val_mask = (df["game_date"] >= val_start) & (df["game_date"] <= val_end)

        train_games = df[train_mask].copy()
        val_games = df[val_mask].copy()

        if not train_games.empty and not val_games.empty:
            _st = (_season_type(postseason_flag(val_games["game_type"].to_numpy()))
                   if "game_type" in val_games.columns else "regular")
            splits.append({
                "train_games": train_games,
                "val_games": val_games,
                "fold_idx": fold_idx,
                "val_start": val_start,
                "val_end": val_end,
                "is_partial_tail": is_partial_tail,
                "season_type": _st,
            })
            fold_idx += 1

        val_start_idx = val_end_idx

    # Limit to max_eval_folds (most recent folds)
    if max_eval_folds > 0 and len(splits) > max_eval_folds:
        splits = splits[-max_eval_folds:]

    return splits


def canonical_walk_forward_splits(
    games: pd.DataFrame,
    retrain_cadence_days: int = RETRAIN_CADENCE_DAYS,
    max_eval_folds: int = 0,
    min_train_days: int = 0,
    min_val_games: int = MIN_VAL_FOLD_GAMES,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Return the shared decided frame and production OOF folds.

    This is the synchronization contract for moneyline, totals, and run
    engine evaluations: canonical post-game identity filtering, expanding
    weekly windows, and ONE shared retained fold set (2026-10-03
    season-split remediation). Every non-empty window is RETAINED — windows
    under ``min_val_games`` are stamped ``provisional=True`` (fit and
    scored, reported, but excluded from the grading population) instead of
    being dropped, and each window carries a ``season_type`` label.
    Postseason games were always in the training frame (train stays
    strictly prior, so never a leakage fix); what changes is that their
    grades are a separate reporting block rather than silently missing.
    """
    decided = get_decided_frame(games).copy()
    all_splits = walk_forward_splits(
        decided, retrain_cadence_days=retrain_cadence_days,
        max_eval_folds=max_eval_folds, min_train_days=min_train_days,
    )
    kept = []
    for s in all_splits:
        s["provisional"] = len(s["val_games"]) < min_val_games
        kept.append(s)
    return decided, kept


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> dict[str, float]:
    """Compute classification metrics: AUC, Brier, LogLoss, ECE."""
    y_true = np.asarray(y_true)
    y_pred_prob = np.asarray(y_pred_prob)

    # Clip to avoid log(0)
    y_pred_prob = np.clip(y_pred_prob, 1e-7, 1 - 1e-7)

    result = {}
    try:
        result["auc"] = round(float(roc_auc_score(y_true, y_pred_prob)), 4)
    except ValueError:
        result["auc"] = 0.5

    result["brier"] = round(float(brier_score_loss(y_true, y_pred_prob)), 4)
    result["logloss"] = round(float(log_loss(y_true, y_pred_prob)), 4)
    result["ece"] = round(float(_expected_calibration_error(y_true, y_pred_prob)), 4)

    return result


def _expected_calibration_error(
    y_true: np.ndarray, y_pred_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred_prob >= bin_edges[i]) & (y_pred_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_pred_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def calibration_buckets(
    y_true: np.ndarray, y_pred_prob: np.ndarray, n_bins: int = 10
) -> list[dict[str, Any]]:
    """Compute calibration bucket data for the dashboard.

    Observations are taken from the FAVORED team's perspective: each game
    contributes ONE point at probability max(p_home, p_away) ∈ [0.5, 1],
    labeled by whether the favorite actually won. This matches how the
    model is consumed (you bet the pick) and is information-equivalent
    to the home-side view, since (p, y) and (1 − p, 1 − y) are exact
    complements — every metric derived from these buckets mirrors the
    home-side version.
    """
    y_pred_prob = np.asarray(y_pred_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    fav_prob = np.maximum(y_pred_prob, 1.0 - y_pred_prob)
    fav_won = np.where(y_pred_prob >= 0.5, y_true, 1.0 - y_true)
    bin_edges = np.linspace(0.5, 1.0, max(n_bins // 2, 1) + 1)
    buckets = []
    for i in range(len(bin_edges) - 1):
        mask = (fav_prob >= bin_edges[i]) & (fav_prob < bin_edges[i + 1])
        if i == len(bin_edges) - 2:  # include 1.0 in the top bucket
            mask |= fav_prob == bin_edges[i + 1]
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = round(float(fav_prob[mask].mean()), 4)
        mean_actual = round(float(fav_won[mask].mean()), 4)
        gap = round(mean_pred - mean_actual, 4)
        buckets.append({
            "bucket": f"{bin_edges[i]*100:.0f}–{bin_edges[i+1]*100:.0f}%",
            "mean_predicted": mean_pred,
            "mean_actual": mean_actual,
            "count": count,
            "gap": gap,
        })
    return buckets


# ── Moneyline ensemble ─────────────────────────────────────────────────────

def _feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Feature matrix preserving NaN.

    Missing observations stay NULL: XGBoost/LightGBM route them natively and
    zero-filling fabricates signal (a 0 mph fastball, a 0.000 wOBA). Only the
    logistic/MLP members — which cannot consume NaN — get train-median
    imputation, applied at predict time via the medians stored in the models
    dict. Team IDs route through a separate categorical path (LightGBM).

    WIDTH IS AN INVARIANT: the result is ALWAYS len(active_moneyline_feature_cols())
    wide in canonical order (the adopted RFE subset when one is applied, else
    the full MONEYLINE_FEATURE_COLS universe). Columns absent from ``df`` come back as
    all-NaN (never silently dropped) with one loud warning — a narrower-than-
    fit-time matrix is how SHAP attributions went quietly empty on synthetic
    slates, and column-name labels elsewhere assume this exact width/order.
    """
    cols = active_moneyline_feature_cols()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning(
            "Feature matrix: %d/%d expected columns absent (%s%s) — filled as "
            "NULL (tree members route NaN; logistic/mlp impute); investigate "
            "the source frame",
            len(missing), len(cols), ", ".join(missing[:6]),
            " …" if len(missing) > 6 else "",
        )
    return df.reindex(columns=cols).to_numpy(dtype=float)


def _prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract feature matrix, categorical matrix, and target.

    Returns (X_numeric, X_categorical, y). X_categorical carries the full
    TREE_CATEGORICAL_COLS set (team + venue + starter IDs) for tree members
    that use native categorical support (LightGBM/XGBoost).
    """
    df = _add_team_ids(df)
    X = _feature_matrix(df)
    X_cat = _categorical_matrix(df)
    y = df["home_win"].values.astype(float)
    return X, X_cat, y


def _impute_median(
    X: np.ndarray, medians: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN with column medians (fit on train when medians is None).

    All-NaN columns fall back to 0.0 so the logistic member stays usable.
    Returns (imputed_matrix, medians_used).
    """
    X = np.asarray(X, dtype=float)
    if medians is None:
        with np.errstate(all="ignore"):
            medians = np.nanmedian(X, axis=0) if len(X) else np.zeros(X.shape[1])
        medians = np.where(np.isnan(medians), 0.0, medians)
    X = X.copy()
    idx = np.isnan(X)
    X[idx] = np.take(np.asarray(medians, dtype=float), idx.nonzero()[1])
    return X, np.asarray(medians, dtype=float)


# Adaptive blend weights from the most recent walk-forward run. Empty until
# the first walk_forward_evaluate() completes; falls back to the static
# ENSEMBLE_WEIGHTS priors before that.
_LAST_ADAPTIVE_WEIGHTS: dict[str, float] = {}

# CAUSAL XGB round measurements (2026-09-30): per-fold best iterations
# from the early-stopped measurement fits, in fold order. Transfer rule,
# FLOORED 2026-10-05 (run-log remediation): fold k ships the median of
# entries STRICTLY BEFORE k and the deployed refit the median of ALL
# entries — but neither may ship BELOW the static no-evidence budget
# (see _shipped_xgb_rounds; the unfloored transfer shipped early
# production folds at 4/6/9/11/13 rounds). Cleared at each
# walk_forward_evaluate start (the OOF-scoring-from-priors rule, same
# as the weights).
_LAST_XGB_BEST_ROUNDS: list[int] = []


def _causal_xgb_rounds(prior_bests: list[int]) -> int:
    """Causal TRANSFER of the XGB round count from PRIOR measurements only.

    Median of the given best-iteration list (even length: lower median,
    matching statistics.median's behaviour of averaging — kept simple and
    deterministic); falls back to the config priors when no measurements
    exist. Pure transfer — SHIPPED models go through
    _shipped_xgb_rounds, which floors this value at the static budget.
    Pure function so the tests pin the selection rule.
    """
    from config import XGBOOST_FOLD0_ROUNDS, XGBOOST_REFIT_ROUNDS
    if not prior_bests:
        return XGBOOST_FOLD0_ROUNDS
    s = sorted(int(b) for b in prior_bests if b and int(b) > 0)
    if not s:
        return XGBOOST_REFIT_ROUNDS
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) // 2


def _shipped_xgb_rounds(prior_bests: list[int], static_rounds: int) -> int:
    """Shipped XGB round count: causal transfer FLOORED at the static budget.

    2026-10-05 run-log remediation: on the production frame the unfloored
    transfer shipped early folds at 4/6/9/11/13 rounds — fold 0's noisy
    probe best became fold 1's entire prior — and the cumulative median
    lagged the growing train window all walk (~27-36). The underfit
    showed up as the XGB member's collapsed OOF (0.5569 AUC / 0.6853
    log-loss) and its 0.0% earned blend weight. max(transfer, static)
    keeps the honesty contract intact — a fold STILL never selects its
    own rounds; the floor is a static config prior and the transfer can
    only raise a shipped count above it (all 83 measured transfers were
    <= 36, so every shipped model trained at exactly the static budget).

    Measured (83 folds, min_train_days=30, production frame): member OOF
    AUC 0.5569 -> 0.5638, log-loss 0.6853 -> 0.6844; blend log-loss
    0.6825 -> 0.6824, Brier 0.2448 -> 0.2447, AUC held 0.5707; verify
    window (folds 70-82) better on all three (AUC 0.5979 -> 0.5990,
    log-loss 0.6760 -> 0.6759, ECE 0.0228 -> 0.0187); the member
    re-earned 13.65% blend weight with NO weight-policy change. Pure
    function so the tests pin the rule.
    """
    return max(_causal_xgb_rounds(prior_bests), int(static_rounds))

# Post-hoc Platt calibrator from the most recent walk-forward run. Applied
# to live blended probabilities in predict_games(); restored from a cached
# bundle via set_calibration() so cached-model runs stay consistent.
_LAST_CALIBRATOR: dict | None = None

# ── Team ID mapping for tree-member categoricals ──────────────────────────

# Consistent integer IDs from 3-letter team abbreviations. Same team = same
# ID across seasons (verified: Statcast team_id is stable). Built lazily
# from observed data so expansion teams get IDs automatically.
_TEAM_ABBR_TO_ID: dict[str, int] = {}
_TEAM_ID_TO_ABBR: dict[int, str] = {}

def _team_id(abbr: str) -> int:
    """Convert a 3-letter team abbreviation to a stable integer ID.

    Unknown / invalid abbreviations (expansion teams, All-Star rosters,
    missing data) map to UNK_TEAM_ID — a dedicated category with
    near-zero training presence so trees learn a neutral weight for it
    instead of silently aliasing a real team (e.g., 0 = NYY).

    Auto-generated IDs skip UNK_TEAM_ID so the reserved slot can never
    collide with a real team no matter how many abbreviations accumulate.
    """
    if abbr in _TEAM_ABBR_TO_ID:
        return _TEAM_ABBR_TO_ID[abbr]
    if not isinstance(abbr, str) or len(abbr) < 2:
        return UNK_TEAM_ID  # semantic "unknown", not a real team
    tid = len(_TEAM_ABBR_TO_ID)
    if tid >= UNK_TEAM_ID:
        tid += 1  # skip the reserved slot
    _TEAM_ABBR_TO_ID[abbr] = tid
    _TEAM_ID_TO_ABBR[tid] = abbr
    return tid

# Tree-member-only categorical columns. NOT in MONEYLINE_FEATURE_COLS — they get
# native categorical handling in LightGBM (categorical_feature= BY NAME) and
# XGBoost (enable_categorical + pd.Categorical dtype). Logistic/MLP must not
# receive them (one-hot would starve on ~4k rows).
#
# ADOPTED SET (production): the 2 team IDs ONLY. Venue + the two starter IDs
# (venue_id / home_starter_cat_id / away_starter_cat_id) were measured as
# native categoricals in the 2026-08-26 ablation — all available at slate
# time (venue always; starters from probable-pitcher data) — but the gate
# was a clear DON'T ADOPT: the 5-col set degraded sealed-holdout logloss
# (+0.0005) and ECE-cal (+0.0068) vs the team-only baseline. See
# data_delivery/categorical_ablation_<sha>.json. The full set stays available
# as FULL_TREE_CATEGORICAL_COLS for the ablation harness (WITH arm) and any
# future re-test; flip the toggle below to re-enable it (re-ablate first).
TREE_CATEGORICAL_COLS = ["home_team_id", "away_team_id"]

# The full candidate set (teams + venue + starters) measured but NOT adopted.
# Exercise it via the harness / tests by patching TREE_CATEGORICAL_COLS to
# this list (which is what run_categorical_ablation.py does for the WITH arm).
FULL_TREE_CATEGORICAL_COLS = [
    "home_team_id",
    "away_team_id",
    "venue_id",
    "home_starter_cat_id",
    "away_starter_cat_id",
]

# RF keeps ONLY the team-ID pair as integer features (sklearn has no native
# categoricals). Slicing is by INDEX into the active TREE_CATEGORICAL_COLS,
# so the team pair must stay the first two entries (asserted in tests).
RF_TREE_CATEGORICAL_COLS = ["home_team_id", "away_team_id"]

# Dedicated "unknown" category IDs, one per categorical space. Each sits above
# its real category count (teams ~30, venues ~29, starters ~430) and is never
# auto-assigned (mappers skip the whole reserved set below), so an unknown /
# invalid / predict-time value maps to a dedicated near-zero-presence category
# instead of silently aliasing a real one. With near-zero training presence,
# trees learn a neutral weight for it.
UNK_TEAM_ID = 99
UNK_VENUE_ID = 98
UNK_STARTER_ID = 97
_RESERVED_CAT_IDS = frozenset({UNK_TEAM_ID, UNK_VENUE_ID, UNK_STARTER_ID})

# RF ablation toggle: set False to train RandomForest WITHOUT team IDs
# (for measuring the marginal benefit of team IDs on the RF member).
# Default True = team IDs included (the production config).
RF_WITH_TEAM_IDS = True

# Logistic feature-set toggle. Default False: the 2026-08-24 ablation
# (14 most recent walk-forward windows, 1206 pooled OOF games) showed the
# diffs-only set equal-or-better for BOTH the member (logloss 0.6918 vs
# 0.6936, AUC +0.0038, Brier −0.0008) and the blended ensemble (logloss
# 0.6885 vs 0.6888; calibrated metrics also favor diffs-only). Set True
# to restore diffs+raws; predict-time routing auto-detects either bundle.
LOGISTIC_USE_RAW_COLS = False

# The raw home/away per-side columns that mirror existing diff twins.
# Routing note: ONLY the logistic member honors LOGISTIC_USE_RAW_COLS — tree
# members always receive them, and the MLP is untouched by this toggle.
# 2026-09-27: expanded 24 → 60 with the 36 new twins (14 level/travel, 16
# exp2, 6 interaction), so every served diff family's raw halves route
# tree-only exactly like the originals.
RAW_PER_SIDE_COLS = [
    "home_elo", "away_elo",
    "home_win_pct", "away_win_pct",
    "sp_k9_home", "sp_k9_away",
    "sp_xwoba_home", "sp_xwoba_away",
    "woba_30g_home", "woba_30g_away",
    "bullpen_whip_10g_home", "bullpen_whip_10g_away",
    "bullpen_whip_3g_home", "bullpen_whip_3g_away",
    "team_barrel_15g_home", "team_barrel_15g_away",
    "team_exitvelo_15g_home", "team_exitvelo_15g_away",
    # 2026-09-27 twin expansion (all tree-only, mirroring the rule above)
    "rest_days_home", "rest_days_away",
    "sp_era_5g_home", "sp_era_5g_away",
    "sp_fbvelo_3g_home", "sp_fbvelo_3g_away",
    "bullpen_pitches_3d_home", "bullpen_pitches_3d_away",
    "team_hardhit_15g_home", "team_hardhit_15g_away",
    "time_zones_crossed_last_3d_home", "time_zones_crossed_last_3d_away",
    "pitcher_regression_indicator_home", "pitcher_regression_indicator_away",
    "ace_efficiency_factor_home", "ace_efficiency_factor_away",
    # exp2 twins — same frozen names as features.EXP2_TWIN_COLS (literal
    # contract, no builder import)
    "exp2_centered_k_home", "exp2_centered_k_away",
    "exp2_cat_k_fastball_home", "exp2_cat_k_fastball_away",
    "exp2_cat_k_breaking_home", "exp2_cat_k_breaking_away",
    "exp2_cat_k_offspeed_home", "exp2_cat_k_offspeed_away",
    "exp2_cat_xwoba_fastball_home", "exp2_cat_xwoba_fastball_away",
    "exp2_cat_xwoba_breaking_home", "exp2_cat_xwoba_breaking_away",
    "exp2_cat_xwoba_offspeed_home", "exp2_cat_xwoba_offspeed_away",
    "exp2_cat_platoon_k_fastball_home", "exp2_cat_platoon_k_fastball_away",
]

# 2026-10-03 (pl_[pos] plan): the re24 / sp_era / depth level twins left
# this list with their families (universe filter _PL_PLAN_REMOVALS above);
# the 16 pl position-pool levels (8 pools × home/away — no DH) take their
# tree-only seat under the same mirror rule. 60 − 10 + 16 = 66; the
# logistic slice moves 41 → 43.
RAW_PER_SIDE_COLS += [
    "pl_c_xwoba_home", "pl_c_xwoba_away",
    "pl_fb_xwoba_home", "pl_fb_xwoba_away",
    "pl_sb_xwoba_home", "pl_sb_xwoba_away",
    "pl_ss_xwoba_home", "pl_ss_xwoba_away",
    "pl_tb_xwoba_home", "pl_tb_xwoba_away",
    "pl_rf_xwoba_home", "pl_rf_xwoba_away",
    "pl_cf_xwoba_home", "pl_cf_xwoba_away",
    "pl_lf_xwoba_home", "pl_lf_xwoba_away",
]


def _logistic_feature_cols() -> list[str]:
    """Model-facing columns for the logistic member (canonical order).

Diff-only under the default flag; the raw per-side columns remain
available to tree members regardless of this toggle.
"""
    cols = active_moneyline_feature_cols()
    if LOGISTIC_USE_RAW_COLS:
        return list(cols)
    raw = set(RAW_PER_SIDE_COLS)
    return [c for c in cols if c not in raw]


def _logistic_feature_indices() -> list[int]:
    """Indices into the active-width feature matrix for the logistic member."""
    active = active_moneyline_feature_cols()
    return [active.index(c) for c in _logistic_feature_cols()]

# Venue / starter categorical mappers — same lazy auto-ID contract as
# _team_id, each with its own reserved UNK slot.
_VENUE_NAME_TO_ID: dict[str, int] = {}
_VENUE_ID_TO_NAME: dict[int, str] = {}
_STARTER_RAW_TO_ID: dict[object, int] = {}
_STARTER_ID_TO_RAW: dict[int, object] = {}


def _next_unreserved(n: int) -> int:
    """Next auto-ID skipping every reserved UNK slot (97/98/99)."""
    while n in _RESERVED_CAT_IDS:
        n += 1
    return n


def _venue_id(venue: object) -> int:
    """Map a venue name to a stable integer category ID.

    The literal 'Unknown' / blank / non-string values map to UNK_VENUE_ID — a
    dedicated near-zero-presence category, never a real park."""
    if venue in _VENUE_NAME_TO_ID:
        return _VENUE_NAME_TO_ID[venue]
    if not isinstance(venue, str) or not venue.strip() \
            or venue.strip().lower() == "unknown":
        return UNK_VENUE_ID
    name = venue.strip()
    vid = _next_unreserved(len(_VENUE_NAME_TO_ID))
    _VENUE_NAME_TO_ID[name] = vid
    _VENUE_ID_TO_NAME[vid] = name
    return vid


def _starter_id(raw: object) -> int:
    """Map an MLB player ID (e.g. 684007) to a compact categorical index.

    Raw IDs are large sparse ints; remapping to a dense space keeps the
    XGBoost category vocabulary small and stable. Missing / non-numeric /
    predict-time callups never seen in training map to UNK_STARTER_ID."""
    if raw in _STARTER_RAW_TO_ID:
        return _STARTER_RAW_TO_ID[raw]
    try:
        if raw is None or pd.isna(raw):
            return UNK_STARTER_ID
        f = float(raw)
        if not np.isfinite(f) or not f.is_integer():
            return UNK_STARTER_ID
        key = int(f)
        if key < 0:
            return UNK_STARTER_ID
    except (TypeError, ValueError):
        return UNK_STARTER_ID
    if key in _STARTER_RAW_TO_ID:
        return _STARTER_RAW_TO_ID[key]
    sid = _next_unreserved(len(_STARTER_RAW_TO_ID))
    _STARTER_RAW_TO_ID[key] = sid
    _STARTER_ID_TO_RAW[sid] = key
    return sid


def _cat_unk_for(col: str) -> int:
    """The reserved UNK slot for a categorical column name."""
    if col == "venue_id":
        return UNK_VENUE_ID
    if col in ("home_starter_cat_id", "away_starter_cat_id"):
        return UNK_STARTER_ID
    return UNK_TEAM_ID


def _cat_known_ids(col: str) -> list[int]:
    """Explicit XGBoost category vocabulary for a categorical column: every
    ID seen in that space plus its reserved UNK slot."""
    if col == "venue_id":
        return sorted(set(_VENUE_ID_TO_NAME) | {UNK_VENUE_ID})
    if col in ("home_starter_cat_id", "away_starter_cat_id"):
        return sorted(set(_STARTER_ID_TO_RAW) | {UNK_STARTER_ID})
    return sorted(set(_TEAM_ID_TO_ABBR) | {UNK_TEAM_ID})


def _add_team_ids(df: "pd.DataFrame") -> "pd.DataFrame":
    """Attach stable integer categorical IDs for tree-member routing.

    Adds the full TREE_CATEGORICAL_COLS set: team IDs (from home/away_team),
    venue ID (from the venue column) and the two starter category IDs (from
    home/away_starter_id). Frames missing a source column get that column's
    reserved UNK slot (slate rows before probable-pitcher announcements, and
    synthetic test frames) — never an error, never a fabricated real value."""
    import pandas as pd
    df = df.copy()
    df["home_team_id"] = df["home_team"].apply(_team_id)
    df["away_team_id"] = df["away_team"].apply(_team_id)
    df["venue_id"] = (df["venue"].apply(_venue_id)
                       if "venue" in df.columns else UNK_VENUE_ID)
    if "home_starter_id" in df.columns:
        df["home_starter_cat_id"] = df["home_starter_id"].apply(_starter_id)
    else:
        df["home_starter_cat_id"] = UNK_STARTER_ID
    if "away_starter_id" in df.columns:
        df["away_starter_cat_id"] = df["away_starter_id"].apply(_starter_id)
    else:
        df["away_starter_cat_id"] = UNK_STARTER_ID
    # Belt-and-suspenders: no real team abbreviation may map to the
    # reserved UNK slot.  The auto-generation skip prevents this in
    # normal operation; this guard catches corruption before training.
    for abbr, tid in sorted(_TEAM_ABBR_TO_ID.items()):
        if tid == UNK_TEAM_ID:
            raise AssertionError(
                f"UNK_TEAM_ID={UNK_TEAM_ID} collides with real team "
                f"'{abbr}' → {tid}. "
                f"Check _team_id auto-generation logic."
            )
    return df

def _categorical_matrix(df: "pd.DataFrame",
                         cols: "Optional[list[str]]" = None) -> "np.ndarray":
    """Extract categorical-feature matrix (default: the full TREE_CATEGORICAL
    set; pass a subset e.g. RF_TREE_CATEGORICAL_COLS for the RF integer path)."""
    import numpy as np
    use = TREE_CATEGORICAL_COLS if cols is None else cols
    return df[use].to_numpy(dtype=int)


def _tree_dataframe(
    X_num: "np.ndarray",
    X_cat: "np.ndarray",
    numeric_cols: list[str],
    vocabs: "Optional[dict[str, list[int]]]" = None,
) -> "pd.DataFrame":
    """Build a DataFrame with named numeric + categorical columns.

    Numeric columns preserve their names from MONEYLINE_FEATURE_COLS. Categorical
    columns become pandas Categorical with an explicit category set (so
    XGBoost never throws "unseen category" at predict time); LightGBM gets
    plain ints + categorical_feature= by name.

    ``vocabs``: the per-column category vocabulary the model was FIT with
    (stored on the models dict as "categorical_vocab"). When provided,
    any value outside that vocabulary — predict-time newcomers like callup
    starters or expansion venues — is clamped to the column's reserved UNK
    slot instead of auto-assigning a fresh category XGBoost never saw
    (which raises "Found a category not in the training set"). When None
    (training-time frame construction, tuners), categories derive from the
    current global ID maps — safe because those callers populate the maps
    before building the frame.
    """
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(X_num, columns=numeric_cols)
    for i, c in enumerate(TREE_CATEGORICAL_COLS):
        vals = X_cat[:, i].copy()
        unk = _cat_unk_for(c)
        # Safety: clamp any lingering negatives / sub-0 values to this
        # column's UNK slot instead of the first real category.
        vals = np.where(vals < 0, unk, vals)
        vocab = (vocabs or {}).get(c)
        if vocab is not None:
            known = np.asarray(sorted(set(vocab)), dtype=int)
            vals = np.where(np.isin(vals, known), vals, unk)
            df[c] = pd.Categorical(vals, categories=sorted(set(vocab)))
        else:
            df[c] = pd.Categorical(vals, categories=_cat_known_ids(c))
    return df



def compute_adaptive_weights(
    oof_members: dict[str, list[float]], y_oof: np.ndarray
) -> dict[str, float]:
    """Blend weights earned by out-of-sample performance.

    Optimized per-cycle blend on pooled OOF scores (2026-09-21). With
    ADAPTIVE_WEIGHT_METRIC="logloss" (default) the weights come from a
    simplex-constrained SLSQP fit minimizing pooled OOF log-loss, applied
    in LOGIT space (clip -> logit -> weighted mean -> sigmoid); with
    "auc" the probability mean is scored instead. There is NO
    temperature, floor, or cap. A member takes 100% of the weight only
    when its own pooled OOF log-loss beats the optimized blend's;
    otherwise the optimized weights stand. In production the caller
    re-earns these weights after every walk-forward fold (rolling
    per-fold weighting — each fold's blend is weighted by the PRIOR
    folds' OOF evidence only), so this function sees one fold's OOF
    window at a time. The result sums to exactly 1.0 and feeds both
    prediction blending and reporting so the ensemble visibly
    self-corrects as features improve.

    Constrained stacking meta-learner ablation (DON'T ADOPT, 2026-08-27):
    an L2-regularized logistic stack (scipy SLSQP; standardized member
    probs; unconstrained / non-negative / non-negative-sum-to-1 variants)
    fit on the five members' pooled OOF probabilities beat the adaptive
    blend on POOLED OOF (e.g. nonneg ll 0.6786/auc 0.5872 vs adaptive
    0.6799/0.5823) but every variant DEGRADED the sealed 284 holdout
    (logloss 0.6847-0.7476 vs adaptive 0.6804; ECE 0.0701-0.1517 vs
    0.0455) — the pooled-gain/sealed-loss inversion seen in every prior
    blend-level gate. See run_stack_ablation.py and
    data_delivery/stack_ablation_20260827.json. Adaptive renormalization
    stays.
    """
    scores: dict[str, float] = {}
    y = np.asarray(y_oof, dtype=float)
    if len(y) == 0:
        return {}
    for name, preds in oof_members.items():
        if not preds or len(preds) != len(y):
            continue
        m = compute_metrics(y, np.asarray(preds, dtype=float))
        if ADAPTIVE_WEIGHT_METRIC == "auc":
            a = m.get("auc")
            if a is None or not np.isfinite(a):
                continue
            scores[name] = float(a)
        else:
            ll = m.get("logloss")
            if ll is None or not np.isfinite(ll):
                continue
            scores[name] = float(ll)
    if not scores:
        return {}

    # Optimized blend, no gates (2026-09-21). Weights minimize pooled OOF
    # log-loss over the simplex (w >= 0, sum(w) = 1) — the same rehearsal
    # window the weights are graded on — and the blend is pooled in LOGIT
    # space. A member earns the ENTIRE weight only when it outperforms the
    # optimized blend on the same pooled OOF window; otherwise the
    # optimized weights stand. The 73-fold roll-forward verification that
    # accompanies this change measures the honesty of this fit (weights
    # from folds < k, scoring fold k).
    names = sorted(scores)
    arrays = {n: np.asarray(oof_members[n], dtype=float) for n in names}
    y = np.asarray(y_oof, dtype=float)

    def _logloss_of(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    if ADAPTIVE_WEIGHT_METRIC == "auc":
        # AUC is rank-based: optimize the probability mean.
        P = np.column_stack([arrays[n] for n in names])
        blend_loss = lambda w: _logloss_of(P @ w)
    else:
        # Log-loss is optimized in LOGIT space, matching ensemble_predict.
        Z = np.column_stack([
            np.log(np.clip(arrays[n], 1e-7, 1 - 1e-7)
                   / (1 - np.clip(arrays[n], 1e-7, 1 - 1e-7)))
            for n in names])
        blend_loss = lambda w: _logloss_of(1.0 / (1.0 + np.exp(-(Z @ w))))

    if len(names) == 1:
        return {names[0]: 1.0}
    from scipy.optimize import minimize
    w0 = np.full(len(names), 1.0 / len(names))
    res = minimize(blend_loss, w0, method="SLSQP",
                   bounds=[(0.0, 1.0)] * len(names),
                   constraints=({"type": "eq",
                                 "fun": lambda w: float(w.sum() - 1.0)}),
                   options={"maxiter": 300, "ftol": 1e-9})
    if not res.success or not np.all(np.isfinite(res.x)):
        # Fall back to the best single member rather than serve a
        # malformed weight vector.
        best = (max if ADAPTIVE_WEIGHT_METRIC == "auc" else min)(
            names, key=lambda n: scores[n])
        return {n: (1.0 if n == best else 0.0) for n in names}
    w = np.clip(np.asarray(res.x, dtype=float), 0.0, None)
    w = w / w.sum() if w.sum() > 0 else w0

    best_name = (max if ADAPTIVE_WEIGHT_METRIC == "auc" else min)(
        names, key=lambda n: scores[n])
    if ADAPTIVE_WEIGHT_METRIC != "auc" and scores[best_name] < blend_loss(w) - 1e-12:
        return {n: (1.0 if n == best_name else 0.0) for n in names}

    # Round without breaking the exact 1.0 total: give the rounding
    # remainder to the largest weight.
    rounded = {n: round(float(v), 4) for n, v in zip(names, w)}
    drift = round(1.0 - sum(rounded.values()), 4)
    if drift:
        top = max(rounded, key=lambda n: rounded[n])
        rounded[top] = round(rounded[top] + drift, 4)
    return rounded


def _member_weights(member_names: list[str]) -> dict[str, float]:
    """Normalized blend weights for the members that actually trained.

    Prefers the rolling adaptive weights (last fold's OOF log-loss earning)
    when available; falls back to static ENSEMBLE_WEIGHTS priors otherwise;
    falls back to static ENSEMBLE_WEIGHTS priors otherwise (e.g. mid-run or
    before the first full evaluation). Members that failed to train
    contribute 0% and the remainder renormalizes to exactly 1.0.
    """
    names = [n for n in member_names
             if n not in ("scaler", "impute_median", "categorical_vocab")]
    # Earned weights stand as earned (including exact zeros — the OOF
    # evidence already re-admits a member whose fit improves, so serving
    # must not resurrect it behind evaluation's back). The equal-thirds
    # priors apply only when nothing has been earned yet (fold 0 / pre-
    # evaluation serving), where they mean "no evidence: treat members
    # alike" rather than an audit-flavored prior.
    raw = ({n: float(_LAST_ADAPTIVE_WEIGHTS.get(n, 0.0)) for n in names}
           if _LAST_ADAPTIVE_WEIGHTS
           else {n: float(ENSEMBLE_WEIGHTS.get(n, 0.0)) for n in names})
    # Every trained member keeps its key (earned zeros included, value 0.0):
    # callers index weights[name] directly, and zero-weight members must
    # simply contribute nothing rather than raise. Positives renormalize.
    total = sum(raw.values())
    if total <= 0:
        w = 1.0 / max(len(names), 1)
        return {n: w for n in names}
    return {n: v / total for n, v in raw.items()}


def feature_importance_weights(ml_models: dict[str, Any]) -> dict[str, float] | None:
    """Blend-weighted feature importance across ensemble members (sums to 100).

    Each member's importances are normalized internally, then averaged with
    the member's configured ENSEMBLE_WEIGHTS share — so the result answers
    "what fraction of the final blended model rides on this feature?"
    Tree members contribute split-gain importance; linear members (elasticnet)
    contribute |coefficient| scattered from the diff-column slice back to
    active-width positions. Returns None when no member exposes importances.
    """
    members = {n: m for n, m in ml_models.items()
               if n not in ("scaler", "impute_median", "categorical_vocab")}
    if not members:
        return None
    eff = _member_weights(list(members.keys()))
    raw = {n: float(eff.get(n, 0.0)) for n in members}
    total = sum(raw.values())
    if total <= 0:
        raw = {n: 1.0 / len(members) for n in members}
        total = 1.0

    agg = np.zeros(len(active_moneyline_feature_cols()))
    nfc = len(active_moneyline_feature_cols())
    # Linear members (elasticnet) train on the diff-column slice, so their
    # coef_ vector is slice-shaped. Map slice -> active-width indices once;
    # on any routing mismatch skip the scatter (the member then simply does
    # not contribute, exactly as before this mapping existed).
    try:
        linear_idx = _logistic_feature_indices()
    except ValueError:
        linear_idx = None
    contributed = False
    for name, model in members.items():
        try:
            if hasattr(model, "feature_importances_"):
                imp = np.asarray(model.feature_importances_, dtype=float).ravel()
            elif hasattr(model, "coef_"):
                imp = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
                if (linear_idx is not None and len(imp) == len(linear_idx)
                        and len(linear_idx) <= nfc):
                    full = np.zeros(nfc)
                    full[linear_idx] = imp
                    imp = full
            else:
                continue
        except Exception:
            continue
        # Tree members trained with team-ID categoricals have larger
        # feature-importance vectors; trim to numeric active cols only.
        if len(imp) >= nfc:
            imp = imp[:nfc]
        if len(imp) != nfc or imp.sum() <= 0:
            continue
        agg += (raw[name] / total) * (imp / imp.sum())
        contributed = True
    if not contributed or agg.sum() <= 0:
        return None
    return {f: round(float(w), 4) for f, w in zip(active_moneyline_feature_cols(), agg / agg.sum() * 100.0)}


def ensemble_predict(
    ml_models: dict[str, Any], games: pd.DataFrame
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """Weighted-blend prediction plus per-member probabilities and weights.

    Returns (blended_prob, {member_name: prob_vector}, {member_name: weight}).
    Falls back to 0.5 when no member can predict.
    """
    games = _add_team_ids(games)
    X = _feature_matrix(games)
    X_cat = _categorical_matrix(games)
    # RF stays on the team-ID pair only (see train_moneyline_ensemble);
    # LGB/XGB build their own full-width frames below.
    _rf_idx = [TREE_CATEGORICAL_COLS.index(c) for c in RF_TREE_CATEGORICAL_COLS]
    X_tree = np.hstack([X, X_cat[:, _rf_idx]])
    scaler = ml_models.get("scaler")
    medians = ml_models.get("impute_median")

    members: dict[str, np.ndarray] = {}
    for name, model in ml_models.items():
        if name in ("scaler", "impute_median", "categorical_vocab"):
            continue
        try:
            if name in ("elasticnet", "logistic", "mlp"):
                # "logistic"/"mlp" stay routable so a cached pre-roster
                # bundle still serves until the next retrain; the elasticnet
                # member rides the identical scaled diff-slice path.
                Xi, _ = _impute_median(X, medians)
                Xu = scaler.transform(Xi) if scaler is not None else Xi
                if name in ("elasticnet", "logistic"):
                    expected = getattr(model, "n_features_in_", Xu.shape[1])
                    if expected != Xu.shape[1]:
                        idx = _logistic_feature_indices()
                        if len(idx) != expected:
                            raise ValueError(
                                f"{name} member expects {expected} columns "
                                f"but LOGISTIC_USE_RAW_COLS routing yields "
                                f"{len(idx)} — persisted bundle was trained "
                                f"under a different flag value")
                        Xu = Xu[:, idx]
                    Xuse = Xu
                else:
                    Xuse = Xu  # legacy mlp: full matrix
            elif name == "xgboost":
                # Raw NaN frame — the SAME representation the member is
                # fit with (NHL member_matrix parity: both tree members
                # share one frame; XGBoost routes missing values
                # natively). _feature_matrix guarantees full
                # MONEYLINE_FEATURE_COLS width/order.
                # Clamp to the fit-time vocabulary so predict-time newcomers
                # (callup starters etc.) route to UNK instead of crashing
                # XGBoost's "category not in the training set" check.
                Xuse = _tree_dataframe(X, X_cat, active_moneyline_feature_cols(),
                                       vocabs=ml_models.get("categorical_vocab"))
            elif name == "lightgbm":
                import pandas as pd
                num_cols_in_data = active_moneyline_feature_cols()
                _df = pd.DataFrame(X, columns=num_cols_in_data)
                _vocab = ml_models.get("categorical_vocab") or {}
                for i, c_ in enumerate(TREE_CATEGORICAL_COLS):
                    vals = np.where(X_cat[:, i] < 0, _cat_unk_for(c_), X_cat[:, i])
                    v = _vocab.get(c_)
                    if v is not None:
                        known = np.asarray(sorted(set(v)), dtype=int)
                        vals = np.where(np.isin(vals, known), vals, _cat_unk_for(c_))
                    _df[c_] = vals.astype(int)
                Xuse = _df
            elif name == "randomforest":
                # If model was trained without team IDs (ablation), it expects
                # active-subset dimensions. Detect from model's n_features_in_.
                if hasattr(model, "n_features_in_") and model.n_features_in_ == len(active_moneyline_feature_cols()):
                    Xuse = X  # ablation: numeric only
                else:
                    Xuse = X_tree  # production: numeric + int team IDs
            else:
                Xuse = X
            members[name] = model.predict_proba(Xuse)[:, 1]
        except Exception as e:
            logger.warning("Member %s failed to predict: %s", name, e)

    if not members:
        return np.full(len(games), 0.5), {}, {}

    weights = _member_weights(list(members.keys()))
    # Logit-space pooling (2026-09-21): the earned weights are fit by
    # minimizing OOF log-loss of the sigmoid of the weighted logit mean,
    # so serving pools the same way. Members with zero weight drop out.
    active = {n: p for n, p in members.items() if weights.get(n, 0.0) > 0}
    if not active:
        return np.full(len(games), 0.5), members, weights
    tot = sum(weights[n] for n in active)
    z = np.zeros(len(games))
    for name, p in active.items():
        pc = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
        z += (weights[name] / tot) * np.log(pc / (1 - pc))
    blend = 1.0 / (1.0 + np.exp(-z))
    return blend, members, weights


# Candidate roster from the most recent walk_forward_evaluate() run:
# every candidate model with its blend weight and pooled out-of-sample
# AUC/Brier/LogLoss (None if it never produced predictions).
_LAST_ENSEMBLE_INFO: list[dict[str, Any]] = []


def last_ensemble_info() -> list[dict[str, Any]]:
    """Candidate-model report from the most recent walk-forward evaluation."""
    return [dict(e) for e in _LAST_ENSEMBLE_INFO]


# Season-split reporting from the most recent walk_forward_evaluate():
# row counts + four published metric blocks (regular / postseason /
# provisional / all) so the graded headline never hides a scored game.
_LAST_SEASON_SPLIT: dict[str, Any] = {}


def get_last_season_split() -> dict[str, Any]:
    """Season-split counts + metric blocks from the latest walk-forward run."""
    from copy import deepcopy
    return deepcopy(_LAST_SEASON_SPLIT)


def set_calibration(calibrator: dict | None) -> None:
    """Restore the favored-team moneyline calibrator from a persisted bundle.

    Legacy home-space Platt maps are rejected rather than silently applied.
    ``fit_platt`` remains available separately for NB market calibration.
    """
    global _LAST_CALIBRATOR
    if calibrator:
        method = str(calibrator.get("method", ""))
        if method != "favored_platt_floor":
            raise ValueError(
                "unsupported moneyline calibrator in bundle: "
                f"{method or '<missing method>'}; retrain with favored-space calibration"
            )
    _LAST_CALIBRATOR = dict(calibrator) if calibrator else None


def get_last_calibrator() -> dict | None:
    """Calibrator fitted on pooled OOF by the most recent walk-forward run."""
    return dict(_LAST_CALIBRATOR) if _LAST_CALIBRATOR else None


def set_adaptive_weights(weights: dict[str, float] | None) -> None:
    """Restore adaptive blend weights (e.g. from a persisted ensemble bundle)
    so prediction blending matches the model that was actually evaluated."""
    _LAST_ADAPTIVE_WEIGHTS.clear()
    if weights:
        _LAST_ADAPTIVE_WEIGHTS.update({k: float(v) for k, v in weights.items()})


def train_moneyline_ensemble(
    train: pd.DataFrame, val: Optional[pd.DataFrame] = None
) -> tuple[dict[str, Any], dict[str, float]]:
    """Train the moneyline ensemble.

    ``val`` is supplied for walk-forward folds so the boosting members can
    evaluate against a strictly future holdout. When omitted, the function
    performs a fit-only refit on every decided game for the deployed bundle.
    """
    X_train, X_cat_train, y_train = _prepare_features(train)
    X_val = X_cat_val = y_val = None
    if val is not None:
        X_val, X_cat_val, y_val = _prepare_features(val)

    if len(X_train) == 0 or (X_val is not None and len(X_val) == 0):
        raise ValueError("Insufficient training or validation data")

    # Logistic cannot consume NaN — impute with TRAIN-fold medians only
    # (never val medians, which would leak).
    X_train_lr, impute_medians = _impute_median(X_train)
    X_val_scaled = None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_lr)
    if X_val is not None:
        X_val_lr, _ = _impute_median(X_val, impute_medians)
        X_val_scaled = scaler.transform(X_val_lr)

    # Build tree-member feature matrices: numeric diffs + categoricals.
    # LightGBM + XGBoost receive the FULL TREE_CATEGORICAL_COLS set natively
    # (by name / pd.Categorical). RandomForest has no native categoricals and
    # stays on the TEAM-ID pair only (integers — the small cardinality, ~30
    # teams, works fine as ordinal-like bins), so its n_features_in_ stays
    # 65 + 2 and the predict-time routing below can detect it unchanged.
    _rf_idx = [TREE_CATEGORICAL_COLS.index(c) for c in RF_TREE_CATEGORICAL_COLS]
    X_cat_rf_train = X_cat_train[:, _rf_idx]
    X_cat_rf_val = X_cat_val[:, _rf_idx] if X_cat_val is not None else None
    X_train_tree = np.hstack([X_train, X_cat_train])
    X_val_tree = np.hstack([X_val, X_cat_val]) if X_val is not None else None
    # Imputed-numeric + team categoricals (RandomForest: impute NaN, keep IDs).
    X_train_lr_tree = np.hstack([X_train_lr, X_cat_rf_train])
    if X_val is not None:
        X_val_lr_tree = np.hstack([X_val_lr, X_cat_rf_val])
    else:
        X_val_lr_tree = None

    models = {}

    # XGBoost — tuned config: raw NaN frame + early stopping. The XGB
    # member consumes the SAME raw feature matrix LightGBM does (NaN routed
    # natively — the missing-value direction is learned, not imputed),
    # matching the NHL member_matrix routing where both tree members share
    # one frame. Walk-forward folds get n_estimators=2000 +
    # early_stopping_rounds=20 on the val window as a MEASUREMENT probe
    # (~19-26 median rounds at depth 2 — see config provenance); the SHIPPED
    # fold model is the causal refit below. Fit-only refits use
    # XGBOOST_PARAMS directly with no early stopping.
    try:
        from xgboost import XGBClassifier
        from config import (XGBOOST_FOLD_ROUNDS, XGBOOST_EARLY_STOP,
                            XGBOOST_FOLD0_ROUNDS, XGBOOST_REFIT_ROUNDS)
        # XGBoost: named DataFrame with pd.Categorical team-ID columns.
        # enable_categorical=True (in XGBOOST_PARAMS) picks them up natively.
        # Labels mirror _feature_matrix's guaranteed width/order.
        num_cols_in_data = active_moneyline_feature_cols()
        X_train_xgb = _tree_dataframe(X_train, X_cat_train, num_cols_in_data)
        if X_val is not None:
            X_val_xgb = _tree_dataframe(X_val, X_cat_val, num_cols_in_data)
            # CAUSAL FOLD ROUNDS (2026-09-30 PIT review; FLOORED
            # 2026-10-05): the early-stopped fit below is a MEASUREMENT
            # ONLY — its best_iteration enters the causal list and informs
            # STRICTLY LATER folds. The SHIPPED fold model is refit
            # without any eval_set at the causal transfer of PRIOR folds'
            # measurements, FLOORED at the static no-evidence budget
            # (_shipped_xgb_rounds — the unfloored transfer shipped early
            # folds at 4-13 rounds), so the fold's own val window never
            # selects the model that scores it (the build_oof_margin
            # fixed-rounds pattern).
            probe = XGBClassifier(
                **XGBOOST_PARAMS,
                n_estimators=XGBOOST_FOLD_ROUNDS,
                early_stopping_rounds=XGBOOST_EARLY_STOP,
            )
            probe.fit(
                X_train_xgb, y_train,
                eval_set=[(X_val_xgb, y_val)],
                verbose=False,
            )
            try:
                _best = int(getattr(probe, "best_iteration", 0)) + 1
            except Exception:
                _best = 0
            if _best > 0:
                _LAST_XGB_BEST_ROUNDS.append(_best)
            causal_n = _shipped_xgb_rounds(
                _LAST_XGB_BEST_ROUNDS[:-1], XGBOOST_FOLD0_ROUNDS)
            xgb = XGBClassifier(**XGBOOST_PARAMS, n_estimators=causal_n)
            xgb.fit(X_train_xgb, y_train, verbose=False)
        else:
            # Fit-only refit (deployed bundle): ship at the causal
            # transfer of the walk's measured fold best rounds (training-
            # side information only), FLOORED at the static refit budget
            # so the deployed bundle matches the same operating point the
            # walk graded; the static prior also covers cache-refit paths
            # with no walk in the same process.
            refit_n = _shipped_xgb_rounds(
                _LAST_XGB_BEST_ROUNDS, XGBOOST_REFIT_ROUNDS)
            xgb = XGBClassifier(**XGBOOST_PARAMS, n_estimators=refit_n)
            xgb.fit(X_train_xgb, y_train, verbose=False)
        models["xgboost"] = xgb
    except ImportError:
        logger.warning("xgboost not available, skipping XGB member")

    # LightGBM — native categorical support via named columns.
    # DataFrame with int team-ID columns + categorical_feature by NAME.
    try:
        from lightgbm import LGBMClassifier
        import pandas as pd
        num_cols_in_data = active_moneyline_feature_cols()
        lgbm_cols = num_cols_in_data + TREE_CATEGORICAL_COLS
        X_train_lgbm = pd.DataFrame(X_train, columns=num_cols_in_data)
        for i, c in enumerate(TREE_CATEGORICAL_COLS):
            X_train_lgbm[c] = np.where(
                X_cat_train[:, i] < 0, _cat_unk_for(c), X_cat_train[:, i]
            ).astype(int)
        lgbm = LGBMClassifier(**LIGHTGBM_PARAMS)
        if X_val is not None:
            X_val_lgbm = pd.DataFrame(X_val, columns=num_cols_in_data)
            for i, c in enumerate(TREE_CATEGORICAL_COLS):
                X_val_lgbm[c] = np.where(
                    X_cat_val[:, i] < 0, _cat_unk_for(c), X_cat_val[:, i]
                ).astype(int)
            lgbm.fit(X_train_lgbm, y_train, eval_set=[(X_val_lgbm, y_val)],
                     categorical_feature=TREE_CATEGORICAL_COLS)
        else:
            lgbm.fit(X_train_lgbm, y_train,
                     categorical_feature=TREE_CATEGORICAL_COLS)
        models["lightgbm"] = lgbm
    except ImportError:
        logger.warning("lightgbm not available, skipping LGBM member")

    # Logistic Regression
    # LOGISTIC_USE_RAW_COLS routing: slice AFTER imputation+scaling so the
    # stored full-width medians/scaler stay canonical; the model simply sees
    # fewer columns. MLP keeps the full matrix regardless of this toggle.
    # Elastic-net logistic — the linear-family member (2026-09-16 roster).
    # Diff-column slice of the imputed+standardized matrix: the identical
    # input path the former logistic member used (LOGISTIC_USE_RAW_COLS
    # routing via _logistic_feature_indices). The mixed L1/L2 penalty
    # prunes redundant correlated diffs (grid-measured ~6 sigma better
    # than plain-L2 logistic on the 84-fold walk-forward).
    _lr_idx = _logistic_feature_indices()
    enet = LogisticRegression(**ELASTICNET_PARAMS)
    enet.fit(X_train_scaled[:, _lr_idx], y_train)
    models["elasticnet"] = enet
    models["scaler"] = scaler
    models["impute_median"] = impute_medians

    # (RandomForest and MLP members removed from the roster 2026-09-16;
    # legacy bundles carrying them still serve via the predict-time
    # routing in ensemble_predict until the next retrain.)
    # Record the categorical vocabulary the tree members were FIT with (the
    # global ID maps as of this fit, i.e. train+val for fold fits, train for
    # fit-only refits). Predict-time frames clamp unseen values to UNK against
    # this vocabulary, so a callup starter at slate time can never crash
    # XGBoost or alias a training category.
    models["categorical_vocab"] = {
        c: _cat_known_ids(c) for c in TREE_CATEGORICAL_COLS
    }

    # A fit-only refit has no honest holdout metric to report.
    if X_val is None:
        return models, {}

    # Weighted ensemble prediction (weights renormalized over trained members)
    weights = _member_weights(list(models.keys()))
    probs, wts = [], []
    for name, model in models.items():
        if name in ("scaler", "impute_median", "categorical_vocab"):
            continue
        if name == "elasticnet":
            Xuse = X_val_scaled[:, _lr_idx]
        elif name == "xgboost":
            Xuse = X_val_xgb  # DataFrame with pd.Categorical team IDs
        elif name == "randomforest":
            if RF_WITH_TEAM_IDS:
                Xuse = X_val_lr_tree
            else:
                Xuse = X_val_lr  # ablation: numeric only
        elif name == "lightgbm":
            Xuse = X_val_lgbm  # DataFrame with int team IDs + cat names
        else:
            Xuse = X_val
        probs.append(model.predict_proba(Xuse)[:, 1])
        wts.append(weights[name])

    ensemble_prob = np.average(probs, axis=0, weights=wts) if probs else np.full(len(y_val), 0.5)

    metrics = compute_metrics(y_val, ensemble_prob)
    return models, metrics


# ── Totals regression ───────────────────────────────────────────────────────

def _attach_oof_run_margins(
    games: pd.DataFrame,
    splits: list[dict[str, Any]],
    min_val_games: int,
    max_eval_folds: int,
    retrain_cadence_days: int,
    min_train_days: int,
    decided_snapshot: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Compute + attach the out-of-fold run margin on the CALLER'S folds.

    run_margin_diff = λ_home − λ_away, where the λs come from the run
    engine's per-side Poisson models (build_oof_margin.oof_run_margins — the
    run engine's own 29-feature levels+env view, READ-ONLY). For each fold
    the run engine trains on that fold's TRAIN games only and predicts that
    fold's VAL games, so every game's margin comes from a model trained
    strictly before it (fold-boundary asserted inside oof_run_margins).

    Games outside any retained fold's val window (warm-up rows) get NO
    margin → NaN → the moneyline's existing imputation path (trees route
    NaN; logistic/MLP train-median). The coverage is logged loudly, never
    papered over. Retention is unconditional (2026-10-03 season-split
    remediation): every retained window gets margins, matching
    walk_forward_evaluate's executed set — the sync assertion below pins
    the two together.

    Returns (enriched COPY of games, regenerated splits over the enriched
    frame). Fold GEOMETRY is asserted identical to the input splits —
    walk_forward_splits is a pure function of game_date/home_win, so the
    only difference is the attached margin columns.

    Frames without the run-engine inputs (game_pk/home_score/away_score —
    e.g. synthetic test frames) get an all-NaN margin column and a loud
    warning instead of a crash: the production training path always carries
    scores, and silently dropping the column would be worse.
    """
    _missing_cols = {"game_pk", "home_score", "away_score"} - set(games.columns)
    if _missing_cols:
        logger.warning(
            "run_margin_diff: frame lacks %s — margin stays all-NaN (imputed "
            "by existing paths); a production frame must carry these",
            sorted(_missing_cols))
        out = games.copy()
        out[MARGIN_COL] = np.nan
        return out, _regenerate_splits(out, splits, min_val_games,
                                       retrain_cadence_days, max_eval_folds,
                                       min_train_days)

    exec_folds = list(splits)
    if not exec_folds:
        logger.warning("run_margin_diff: no executed folds — margin stays all-NaN")
        out = games.copy()
        out[MARGIN_COL] = np.nan
        return out, splits

    from build_oof_margin import MARGIN_COL as _BOM_MARGIN, oof_run_margins
    assert _BOM_MARGIN == MARGIN_COL
    # Single source of truth for the decided frame (frames.py): the
    # same rules as every other consumer, so the margin build can never
    # see a row set that differs from the drift/coverage/run-engine
    # decided frames. Row-for-row no-op on the canonical training frame.
    decided = decided_snapshot if decided_snapshot is not None else get_decided_frame(games)
    set_last_fold_signature(fold_signature(decided))
    margins, rounds, uncov = oof_run_margins(decided, exec_folds)
    set_last_margin_rounds(rounds)

    out = games.copy()
    out = out.drop(columns=[MARGIN_COL] if MARGIN_COL in out.columns else [])
    out = out.merge(margins[["game_pk", MARGIN_COL]], on="game_pk", how="left")

    regen = _regenerate_splits(out, splits, min_val_games,
                               retrain_cadence_days, max_eval_folds,
                               min_train_days)
    if len(regen) != len(exec_folds) or not all(
        a["fold_idx"] == b["fold_idx"]
        and pd.Timestamp(a["val_start"]) == pd.Timestamp(b["val_start"])
        and a["val_games"]["game_pk"].tolist()
        == b["val_games"]["game_pk"].tolist()
        for a, b in zip(regen, exec_folds)):
        raise AssertionError(
            "run_margin_diff: enriched-frame folds desynced from margin-build "
            "folds (walk_forward_splits changed?) — refusing to train on a "
            "misaligned split")

    covered = float(out[MARGIN_COL].notna().mean())
    logger.info(
        "run_margin_diff: OOF margins attached on the moneyline's own folds "
        "(run engine READ-ONLY, 29-feature view); coverage %.1f%% of %d rows, "
        "%d decided game(s) uncovered (NaN → imputation); median rounds %s",
        100 * covered, len(out), uncov,
        {k: int(v) for k, v in rounds.items()})
    return out, regen


def _regenerate_splits(out: pd.DataFrame, splits: list[dict[str, Any]],
                       min_val_games: int, retrain_cadence_days: int,
                       max_eval_folds: int, min_train_days: int) -> list[dict[str, Any]]:
    """Re-split the (possibly enriched) frame with the caller's geometry
    parameters so every train/val slice carries the attached columns.

    Retention is unconditional (2026-10-03): the same set
    walk_forward_evaluate executes, so the margin-fold-sync assertion holds
    on every window. ``min_val_games`` is kept for call-signature parity
    and no longer filters here; callers re-derive the ``provisional`` flag.
    """
    return walk_forward_splits(
        out, retrain_cadence_days=retrain_cadence_days,
        max_eval_folds=max_eval_folds, min_train_days=min_train_days)


# ── Full walk-forward evaluation ────────────────────────────────────────────

def walk_forward_evaluate(
    games: pd.DataFrame,
    retrain_cadence_days: int = RETRAIN_CADENCE_DAYS,
    max_eval_folds: int = 0,
    force_retrain: bool = False,
    min_train_days: int = 0,
    min_val_games: Optional[int] = None,
    decided_snapshot: Optional[pd.DataFrame] = None,
) -> tuple[dict[str, Any], dict[str, float], pd.DataFrame]:
    """Run full walk-forward evaluation across all splits.

    Every retained window is FIT and SCORED (2026-10-03 season-split
    remediation). Folds with fewer than ``min_val_games`` val games are
    marked ``provisional`` instead of being skipped: their rows still land
    in the returned OOF frame (CSV / predictions_history) but never enter
    the grading population — the headline pooled metrics, the shipped
    Platt map, the prequential calibrator and the adaptive weights all
    earn only from GRADES rows (regular-season rows of non-provisional
    folds). Postseason rows are reported as their own block via
    get_last_season_split() instead of being pooled into — or dropped
    from — the headline. Pass ``min_val_games=0`` to grade every fold
    (used by tests).

    decided_snapshot: Pre-computed decided frame (from frames.get_decided_frame)
            captured ONCE after official results, before slate merge.  Passed
            through so _attach_oof_run_margins records the SAME fold signature
            training saw — preventing the drift-vs-training desync when the
            pipeline frame is later mutated by slate concatenation.

    Returns:
        (best_models, pooled_metrics, all_predictions)
    """
    if min_val_games is None:
        min_val_games = MIN_VAL_FOLD_GAMES

    # OOF scoring must start from the configured priors. Otherwise an earlier
    # run's adaptive weights can change the current run's fold predictions.
    _LAST_ADAPTIVE_WEIGHTS.clear()
    # Season-split report is per-run state too.
    _LAST_SEASON_SPLIT.clear()
    # Same priors rule for the causal XGB round measurements: a stale list
    # from a previous run/frame would leak another run's fold geometry.
    _LAST_XGB_BEST_ROUNDS.clear()
    splits = walk_forward_splits(games, retrain_cadence_days, max_eval_folds, min_train_days)

    # Record the canonical training-frame signature even when the optional
    # run-margin feature is not active.  The margin builder records this too,
    # but relying on that path left the drift guard with training=None on
    # otherwise complete non-margin runs.
    decided_for_signature = (
        decided_snapshot if decided_snapshot is not None else get_decided_frame(games)
    )
    set_last_fold_signature(fold_signature(decided_for_signature))

    # Shipped run-margin feature: attach leakage-free OOF margins on exactly
    # these folds before any training happens, and regenerate the splits over
    # the enriched frame (geometry asserted identical inside). The final
    # fit-only refit below then sees the same column. Skipped with a loud
    # warning when the frame lacks run-engine inputs.
    if MARGIN_COL in active_moneyline_feature_cols() and splits:
        games, splits = _attach_oof_run_margins(
            games, splits, min_val_games, max_eval_folds,
            retrain_cadence_days, min_train_days,
            decided_snapshot=decided_snapshot)

    # Record canonical fold geometry for the later drift attachment.
    set_last_walk_forward_splits(splits)

    if not splits:
        logger.warning("No walk-forward splits generated; training on full data")
        # Fall back to train on everything
        splits = [{
            "train_games": games.dropna(subset=["home_win"]),
            "val_games": games.dropna(subset=["home_win"]).tail(min(50, len(games.dropna(subset=["home_win"])))),
            "fold_idx": 0,
            "val_start": games["game_date"].min(),
            "val_end": games["game_date"].max(),
        }]

    all_preds = []
    fold_metrics_list = []
    oof_members: dict[str, list[float]] = {}
    # Per-member prequential calibrated twins: fold k's map (fitted strictly
    # on PRIOR folds' blend pairs — the same map already applied to the blend)
    # applied to each member's raw probabilities. Honest out-of-sample
    # calibrated diagnostics without fitting anything extra.
    oof_members_cal: dict[str, list[float]] = {}
    oof_y: list[float] = []
    # Blended raw probabilities and their PREQUENTIAL calibrated twins:
    # each fold's calibration comes from a Platt map fitted strictly on
    # PRIOR folds' OOF pairs, so every calibrated point stays honest.
    oof_blend: list[float] = []
    oof_blend_calibrated: list[float] = []

    for split in splits:
        train = split["train_games"]
        val = split["val_games"]

        if len(train) < 10 or len(val) < 5:
            continue
        # Season-split remediation (2026-10-03): no window is skipped for
        # size anymore. Sub-gate windows stay in the run as PROVISIONAL
        # folds — fit, scored, written to the OOF frame — but never enter
        # the grading accumulators below. Leakage contract unchanged: train
        # is strictly < val_start for every fold, so keeping a window is
        # never a leak (the old "tiny postseason folds pollute the pooled
        # scores" concern is answered by grading the population, not by
        # deleting the games).
        provisional = len(val) < min_val_games
        _post = (postseason_flag(val["game_type"].to_numpy())
                 if "game_type" in val.columns
                 else np.zeros(len(val), dtype=bool))
        # Row-level grading mask: regular-season rows of non-provisional
        # folds. Postseason rows and every provisional row are SCORED but
        # never GRADERS — the structural contradiction being removed is
        # n=3 grades (thin Finals tail pooled) vs n=34 not graded.
        grades = (~_post) if not provisional else np.zeros(len(val), dtype=bool)
        season_type = split.get("season_type") or _season_type(_post)
        if provisional:
            logger.info(
                "Fold %d [%s → %s]: PROVISIONAL (%d val games < %d) — "
                "fit + scored, excluded from grading",
                split["fold_idx"], str(split["val_start"])[:10],
                str(split["val_end"])[:10], len(val), min_val_games,
            )

        try:
            ml_models, ml_metrics = train_moneyline_ensemble(train, val)
        except Exception as e:
            logger.warning("Fold %d moneyline training failed: %s", split["fold_idx"], e)
            continue

        logger.info(
            "Fold %d [%s → %s]: train=%d val=%d auc=%.4f brier=%.4f",
            split["fold_idx"],
            str(split["val_start"])[:10], str(split["val_end"])[:10],
            len(train), len(val), ml_metrics.get("auc", 0.5), ml_metrics.get("brier", 0.25),
        )

        # Weighted-blend prediction; keep each member's probabilities so we
        # can score candidates individually out of sample.
        ensemble_prob, member_probs, _wts = ensemble_predict(ml_models, val)
        y_val = val["home_win"].values.tolist()

        # Prequential calibration: fit on everything out-of-sample BEFORE
        # this fold, then transform this fold's predictions.
        fold_cal = None
        if len(oof_blend) >= MIN_OOF_FOR_FIT:
            fold_cal = moneyline_fit(oof_y, oof_blend)
        fold_calibrated = moneyline_apply(ensemble_prob, fold_cal)

        # Grading-gated accumulators: only GRADES rows feed the prequential
        # calibrator, the shipped Platt map, the adaptive weights and the
        # per-member reports. Non-grading rows are still scored and still
        # published — via val_pred/combined below (CSV, predictions_history,
        # calibration curve) and via the season-split blocks.
        oof_y.extend(np.asarray(y_val, dtype=float)[grades].tolist())
        oof_blend.extend(
            np.asarray(ensemble_prob, dtype=float)[grades].tolist())
        oof_blend_calibrated.extend(
            np.asarray(fold_calibrated, dtype=float)[grades].tolist()
        )
        for name, p in member_probs.items():
            p_arr = np.asarray(p, dtype=float)
            pc = np.asarray(moneyline_apply(p_arr, fold_cal), dtype=float)
            oof_members.setdefault(name, []).extend(p_arr[grades].tolist())
            oof_members_cal.setdefault(name, []).extend(pc[grades].tolist())

        # Rolling per-fold blend weighting (2026-09-16 spec): after each
        # fold, re-earn the blend weights from the accumulated PRIOR+current
        # fold OOF member log-loss, so the NEXT fold's blend is weighted by
        # evidence strictly before it (causal — never sees what it scores).
        # Fold 0 blended on the static priors (cleared at run start).
        _rolling = compute_adaptive_weights(oof_members, oof_y) if oof_y else {}
        if _rolling:
            _LAST_ADAPTIVE_WEIGHTS.clear()
            _LAST_ADAPTIVE_WEIGHTS.update(_rolling)

        val_pred = val.copy()
        val_pred["home_win_prob_model"] = ensemble_prob
        val_pred["home_win_prob_model_calibrated"] = np.round(fold_calibrated, 4)
        val_pred["fold_idx"] = split["fold_idx"]
        # Season-split disclosure columns: row-level where the fact is
        # row-level (is_playoffs, grades_pooled), fold-level broadcast
        # where it is a window property (season_type, provisional).
        val_pred["is_playoffs"] = _post
        val_pred["season_type"] = season_type
        val_pred["provisional"] = provisional
        val_pred["grades_pooled"] = grades
        all_preds.append(val_pred)
        fold_metrics_list.append(ml_metrics)

    # Pool metrics across the GRADING population only — regular-season
    # rows of non-provisional folds (2026-10-03 season-split remediation).
    # The full OOF frame, postseason and provisional rows included, is
    # still returned/written; it just doesn't grade. For a run whose every
    # window is under the gate this falls back to the uninformed default —
    # the same pooled number the OLD skip-everything path reported, now
    # with the rows visible instead of absent.
    if all_preds:
        combined = pd.concat(all_preds, ignore_index=True)
        _g = (combined["grades_pooled"].astype(bool).to_numpy()
              if "grades_pooled" in combined.columns
              else np.ones(len(combined), dtype=bool))
        if _g.any():
            pooled = compute_metrics(
                combined["home_win"].values[_g],
                combined["home_win_prob_model"].values[_g])
        else:
            logger.warning(
                "Walk-forward: all %d OOF rows are non-grading "
                "(postseason/provisional) — pooled metrics fall back to the "
                "uninformed default", len(combined))
            pooled = {"auc": 0.5, "brier": 0.25, "logloss": 0.69, "ece": 0.0}
    else:
        combined = pd.DataFrame()
        pooled = {"auc": 0.5, "brier": 0.25, "logloss": 0.69, "ece": 0.0}

    # Post-hoc calibration: fit the shipped Platt map on ALL pooled OOF
    # pairs, and score it against the raw blend using the prequential
    # calibrated predictions (fold k corrected only by folds < k — never
    # self-calibrated). Raw headline metrics stay untouched; calibrated
    # twins ride alongside so dashboards can show both.
    y_oof_all = np.asarray(oof_y, dtype=float) if oof_y else np.empty(0)
    p_raw_all = np.asarray(oof_blend, dtype=float) if oof_blend else np.empty(0)
    p_cal_prequential = (
        np.asarray(oof_blend_calibrated, dtype=float)
        if oof_blend_calibrated else np.empty(0)
    )
    final_calibrator = moneyline_fit(y_oof_all, p_raw_all)
    global _LAST_CALIBRATOR
    _LAST_CALIBRATOR = final_calibrator
    gated_out = False
    if p_cal_prequential.size == len(y_oof_all) and len(y_oof_all) > 0:
        # Score the PREQUENTIAL column directly. Each point was corrected by
        # a map fitted strictly on PRIOR folds — exactly how the deployed
        # final map behaves on unseen games. Composing final_calibrator on
        # top instead evaluates G(platt_k(raw_k)): a double correction no
        # production path ever applies, which flatters the reported numbers
        # (empirically verified: brier 0.2473 double-applied vs 0.2715 honest).
        m_cal = compute_metrics(y_oof_all, p_cal_prequential)
        pooled["brier_calibrated"] = m_cal["brier"]
        pooled["logloss_calibrated"] = m_cal["logloss"]
        pooled["ece_calibrated"] = m_cal["ece"]
        logger.info(
            "Calibration (OOF, prequential): ECE %.4f → %.4f, log-loss %.4f → %.4f",
            pooled.get("ece", 0.0), m_cal["ece"],
            pooled.get("logloss", 0.0), m_cal["logloss"],
        )
        # Deployed-calibrator gate (2026-10-01): the shipped map above is
        # fitted on ALL pooled OOF, so its only honest rehearsal is the
        # prequential column just scored. When that rehearsal is worse than
        # the raw blend on BOTH log-loss and ECE, the map demonstrably hurts
        # every headline metric — ship the raw blend (identity) instead of a
        # harmful correction. Mixed evidence keeps the fitted map (the
        # 2026-08-27 flip-test status quo). Fully reversible: a later run
        # whose prequential column improves ships the fitted map again.
        gated_out, gate_reason = should_gate_calibrator(pooled, m_cal)
        pooled["calibrator_gated_out"] = gated_out
        if gated_out:
            _LAST_CALIBRATOR = None
            logger.warning(
                "Calibration: prequential calibrated metrics worse than raw "
                "(%s) — shipping the raw blend (identity calibrator)",
                gate_reason,
            )

    # Fit the deployed bundle on every decided game. The walk-forward folds
    # remain the only source of honest OOF metrics; no final validation holdout
    # is needed once evaluation is complete.
    full_train = games.dropna(subset=["home_win"])
    if len(full_train) >= 20:
        try:
            best_models, _ = train_moneyline_ensemble(full_train)
        except Exception:
            best_models = {}
    else:
        best_models = {}

    # Rolling per-fold weighting (2026-09-16 spec): weights are re-earned
    # INSIDE the fold loop (after each fold, from OOF member log-loss —
    # see the loop body), so the deployed bundle carries the LAST fold's
    # earned vector — the most recent causal evidence. The old pooled-OOF
    # earning (one softmax over the entire history at run end) is retired.
    y_oof = np.asarray(oof_y, dtype=float)
    adaptive = dict(_LAST_ADAPTIVE_WEIGHTS)
    if adaptive:
        logger.info(
            "Rolling blend weights (last fold earned): %s",
            {k: f"{v:.1%}" for k, v in sorted(adaptive.items())},
        )
    # Shipped-rounds visibility (2026-10-05 log remediation): the run log
    # could not show that folds shipped at 4-13 rounds because the causal
    # transfer was invisible — one summary line makes the measurement,
    # the transfer and the floored ship count auditable per run.
    if _LAST_XGB_BEST_ROUNDS:
        from config import XGBOOST_REFIT_ROUNDS
        _xfer = _causal_xgb_rounds(_LAST_XGB_BEST_ROUNDS)
        logger.info(
            "XGBoost shipped rounds: %d probe measurement(s), causal transfer "
            "%d floored at the static budget — refit ships %d",
            len(_LAST_XGB_BEST_ROUNDS), _xfer,
            max(_xfer, XGBOOST_REFIT_ROUNDS),
        )

    # Candidate-model report: every candidate that ever trained, its blend
    # weight in the deployed ensemble (adaptive when available; absent or
    # failed candidates report 0%), and its own pooled out-of-fold
    # AUC/Brier/LogLoss across all evaluation folds.
    final_members = {
        n for n in (best_models or {})
        if n not in ("scaler", "impute_median", "categorical_vocab")
    }
    raw_w = {
        n: float(adaptive.get(n, 0.0)) if adaptive
        else float(ENSEMBLE_WEIGHTS.get(n, 0.0))
        for n in final_members
    }
    w_total = sum(raw_w.values()) or 1.0
    # Every configured candidate is reported — even ones that failed to train
    # this run (weight 0%, metrics null) — per the ensemble transparency rule.
    roster = list(dict.fromkeys(
        list(ENSEMBLE_WEIGHTS.keys()) + list(oof_members.keys()) + sorted(final_members)
    ))
    _LAST_ENSEMBLE_INFO.clear()
    for name in roster:
        entry: dict[str, Any] = {
            "name": name,
            "weight": round(raw_w[name] / w_total, 4) if name in final_members else 0.0,
        }
        preds = oof_members.get(name)
        if preds and len(preds) == len(y_oof) and len(y_oof) > 0:
            m = compute_metrics(y_oof, np.asarray(preds, dtype=float))
            entry.update({
                "auc": m.get("auc"),
                "brier": m.get("brier"),
                "logloss": m.get("logloss"),
                "n_eval": int(len(y_oof)),
            })
            preds_cal = oof_members_cal.get(name)
            if (preds_cal and len(preds_cal) == len(y_oof)):
                mc = compute_metrics(y_oof, np.asarray(preds_cal, dtype=float))
                entry.update({
                    "brier_calibrated": mc.get("brier"),
                    "logloss_calibrated": mc.get("logloss"),
                    "ece_calibrated": mc.get("ece"),
                })
        else:
            entry.update({"auc": None, "brier": None, "logloss": None, "n_eval": 0})
        _LAST_ENSEMBLE_INFO.append(entry)

    # Season-split reporting (2026-10-03): four published populations so
    # the headline can grade on regular-season folds alone without hiding
    # any scored game. Blocks use the RAW blend — the same quantity as
    # `pooled` — so oof_regular reconciles with the headline exactly, and
    # `sufficient` discloses a block thinner than the gate instead of
    # letting a thin block masquerade as a verdict.
    def _season_block(frame: Optional[pd.DataFrame]) -> dict[str, Any]:
        if frame is None or not len(frame):
            return {"n": 0, "sufficient": False}
        m = compute_metrics(frame["home_win"].to_numpy(dtype=float),
                            frame["home_win_prob_model"].to_numpy(dtype=float))
        return {**m, "n": int(len(frame)),
                "sufficient": bool(len(frame) >= min_val_games)}

    if len(combined) and "grades_pooled" in combined.columns:
        _g = combined["grades_pooled"].astype(bool).to_numpy()
        _p = (combined["is_playoffs"].astype(bool).to_numpy()
              if "is_playoffs" in combined.columns
              else np.zeros(len(combined), dtype=bool))
        _v = (combined["provisional"].astype(bool).to_numpy()
              if "provisional" in combined.columns
              else np.zeros(len(combined), dtype=bool))
    else:
        _g = np.zeros(len(combined), dtype=bool)
        _p = np.zeros(len(combined), dtype=bool)
        _v = np.zeros(len(combined), dtype=bool)
    season_split: dict[str, Any] = {
        "counts": {
            "regular_rows": int((~_p).sum()),
            "postseason_rows": int(_p.sum()),
            "provisional_rows": int(_v.sum()),
            "grading_rows": int(_g.sum()),
            "oof_rows": int(len(combined)),
        },
        "blocks": {
            "oof_regular": _season_block(combined[_g] if len(combined) else None),
            "oof_postseason": _season_block(combined[_p] if len(combined) else None),
            "oof_provisional": _season_block(combined[_v] if len(combined) else None),
            "oof_all": _season_block(combined if len(combined) else None),
        },
    }
    _LAST_SEASON_SPLIT.clear()
    _LAST_SEASON_SPLIT.update(season_split)
    logger.info(
        "Walk-forward season split: %d OOF rows (%d grading / %d "
        "postseason / %d provisional); blocks regular n=%d, postseason "
        "n=%d, provisional n=%d",
        len(combined), int(_g.sum()), int(_p.sum()), int(_v.sum()),
        season_split["blocks"]["oof_regular"]["n"],
        season_split["blocks"]["oof_postseason"]["n"],
        season_split["blocks"]["oof_provisional"]["n"],
    )

    return best_models, pooled, combined


# ── Persistence ─────────────────────────────────────────────────────────────

def persist_ensemble(
    models: dict[str, Any],
    metrics: dict[str, float],
    version: str = "v3.2.1",
    data_cutoff: Optional[str] = None,
) -> Path:
    """Save ensemble models and metadata to joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        VERSION_KEY: version,
        TRAINED_AT_KEY: datetime.now().isoformat(),
        DATA_CUTOFF_KEY: data_cutoff or datetime.now().strftime(DATE_FMT),
    }

    bundle = {
        "models": models,
        "metrics": metrics,
        "metadata": metadata,
        # Earned blend weights ride with the models so a cached-model run
        # predicts with exactly the weighting that was validated.
        "adaptive_weights": dict(_LAST_ADAPTIVE_WEIGHTS),
        # Post-hoc Platt calibrator fitted on pooled OOF; applied by
        # predict_games so published probabilities are calibrated.
        "calibrator": get_last_calibrator(),
        # The exact feature list the members were trained on. Serving-side
        # truth: a cached bundle must always be predicted with its own fit
        # width, even if the adopted RFE subset later changes — see
        # apply_bundle_feature_cols. Bundles persisted before this field
        # existed were trained at universe width (key absent → universe).
        "feature_cols": active_moneyline_feature_cols(),
    }

    path = MODELS_DIR / ENSEMBLE_FILE
    joblib.dump(bundle, path)
    logger.info("Ensemble persisted to %s", path)
    return path


def load_ensemble(path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Load a persisted ensemble bundle."""
    path = path or (MODELS_DIR / ENSEMBLE_FILE)
    if not path.exists():
        return None
    return joblib.load(path)


def apply_bundle_feature_cols(bundle: Optional[dict[str, Any]]) -> None:
    """Make serving width match a loaded bundle's fit width.

    Applies the bundle's recorded feature_cols as the active subset. A bundle
    without the field (pre-RFE format) was trained at universe width and
    resets to it. Invalid recorded lists (universe shrank beneath an old
    bundle, corrupted joblib) degrade loudly to the universe rather than
    crash the daily board — a width mismatch would fail anyway inside the
    members, so the universe fallback is the honest best effort.
    """
    if not bundle:
        reset_feature_subset()
        return
    cols = bundle.get("feature_cols")
    if cols is None:
        reset_feature_subset()
        return
    try:
        set_feature_subset(list(cols))
        logger.info("Serving feature width: %d (bundle-recorded subset)", len(cols))
    except ValueError as e:
        logger.warning(
            "Bundle feature_cols not a valid subset — serving at full "
            "universe width instead (%s)", e)
        reset_feature_subset()


def should_retrain(last_trained: Optional[datetime], cadence_days: int = RETRAIN_CADENCE_DAYS) -> bool:
    """Determine if retraining is needed based on cadence."""
    if last_trained is None:
        return True
    return (datetime.now() - last_trained).days >= cadence_days


def predict_games(
    models: dict[str, Any],
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Apply ensemble models to predict on a set of games.

    Adds columns: home_win_prob_model, away_win_prob_model, model_pick, edge_home, edge_away
    """
    if not models:
        return games

    if games.empty:
        # 0-row board (genuine off-day, 2026-09-28 incident): tree predict()
        # rejects 0-sample matrices — including inside ensemble_predict — so
        # guard BEFORE any model call: stamp the probability columns on the
        # empty frame and return; the artifact writers ship an honest empty
        # board.
        games["home_win_prob_model"] = pd.Series(dtype=float)
        games["away_win_prob_model"] = pd.Series(dtype=float)
        games["model_pick"] = pd.Series(dtype=object)
        return games
    blend, _members, _wts = ensemble_predict(models, games)
    # Post-hoc recalibration: correct blended probabilities before they
    # feed picks/edges. Identity (no-op) when no calibrator is loaded.
    calibrator = get_last_calibrator()
    if not is_identity(calibrator):
        blend = moneyline_apply(blend, calibrator)
    games["home_win_prob_model"] = np.round(blend, 4)

    games["away_win_prob_model"] = 1 - games["home_win_prob_model"]

    # Model pick
    games["model_pick"] = np.where(
        games["home_win_prob_model"] >= 0.5, games["home_team"], games["away_team"]
    )

    # Edge: model_prob - fair_market_prob (vig removed via two-way normalization)
    if "moneyline_home" in games.columns and games["moneyline_home"].notna().any():
        ml_home = games["moneyline_home"].fillna(-110).values
        ml_away = games["moneyline_away"].fillna(-110).values
        fair_home = np.where(ml_home < 0, -ml_home / (-ml_home + 100), 100 / (ml_home + 100))
        fair_away = np.where(ml_away < 0, -ml_away / (-ml_away + 100), 100 / (ml_away + 100))
        # Normalize (remove vig)
        total = fair_home + fair_away
        fair_home_norm = fair_home / total
        fair_away_norm = fair_away / total
        games["edge_home"] = np.round(games["home_win_prob_model"].values - fair_home_norm, 4)
        games["edge_away"] = np.round(games["away_win_prob_model"].values - fair_away_norm, 4)
    else:
        games["edge_home"] = 0.0
        games["edge_away"] = 0.0

    return games


def update_model_history(
    metrics: dict[str, float],
    version: str,
    notes: str = "",
) -> None:
    """Append a row to model_history.json for the Model Monitor page.

    One row per calendar day: a re-run on the same day REPLACES that day's
    row instead of appending, so the Version History table reflects real
    retrains rather than every debugging rerun.
    """
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    history_path = DATA_DELIVERY_DIR / "model_history.json"

    history = []
    if history_path.exists():
        with open(history_path) as f:
            try:
                history = json.load(f)
            except ValueError:
                history = []

    today = datetime.now().strftime("%Y-%m-%d")
    history = [row for row in history if row.get("date") != today]
    history.append({
        "version": version,
        "date": today,
        "auc": metrics.get("auc", 0),
        "brier": metrics.get("brier", 0),
        "logloss": metrics.get("logloss", 0),
        "ece": metrics.get("ece", 0),
        **({"ece_calibrated": metrics["ece_calibrated"]}
           if "ece_calibrated" in metrics else {}),
        "notes": notes,
    })
    history.sort(key=lambda row: str(row.get("date", "")))

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

VERSION_HISTORY_CAP = 20
VERSION_HISTORY_FILE = "model_version_history.json"


def update_model_version_history(
    metrics: dict,
    version: str,
    ensemble_info: Optional[list] = None,
    calibrator: dict | None = None,
) -> Optional[Path]:
    """Append/merge one snapshot row into model_version_history.json.

    Row = version, date, per-member ensemble weights, pooled walk-forward
    metrics (raw + calibrated variants), and the deployed calibration params.
    Re-running the same version REPLACES its row (merge by version). The file
    keeps the last VERSION_HISTORY_CAP rows. Write is atomic (tmp + rename)
    so a crash can never leave a truncated artifact behind; a row is only
    written when ALL inputs are present — never a partial snapshot.
    """
    if not metrics or not version:
        logger.warning("Version history: missing metrics/version — no row written")
        return None
    if not ensemble_info:
        logger.warning(
            "Version history: no ensemble roster for %s — no row written "
            "(weights would be fabricated)", version,
        )
        return None

    weights = {
        str(e["name"]): round(float(e.get("weight") or 0.0), 4)
        for e in ensemble_info
        if e.get("name") is not None and e.get("weight") is not None
    }
    if not weights:
        logger.warning("Version history: roster carries no usable weights — skipped")
        return None

    metric_keys = (
        "auc", "brier", "logloss", "ece",
        "brier_calibrated", "logloss_calibrated", "ece_calibrated",
    )
    row: dict = {
        "version": version,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "weights": weights,
        **{k: metrics[k] for k in metric_keys if k in metrics},
    }
    if isinstance(calibrator, dict) and calibrator.get("a") is not None:
        try:
            row["calibration"] = {
                "a": float(calibrator["a"]),
                "b": float(calibrator["b"]),
                "n": int(calibrator.get("n", 0)),
                "method": str(calibrator.get("method", "platt")),
                **({"floor": float(calibrator["floor"])}
                   if calibrator.get("floor") is not None else {}),
            }
        except (TypeError, ValueError):
            pass  # deployed map stays absent rather than half-recorded

    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / VERSION_HISTORY_FILE
    history: list = []
    if path.exists():
        try:
            with open(path) as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                history = loaded
        except ValueError:
            logger.warning("Version history file corrupt — starting fresh")
            history = []

    history = [r for r in history if r.get("version") != version]
    history.append(row)
    history.sort(key=lambda r: str(r.get("date", "")))
    history = history[-VERSION_HISTORY_CAP:]

    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(history, f, indent=2)
    os.replace(tmp_path, path)  # atomic: readers never see partial JSON
    logger.info("Version history: %d versions recorded (%s latest)", len(history), version)
    return path
