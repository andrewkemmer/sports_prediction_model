"""Central configuration for MLB Bet Predictor backend.

All paths, hyperparameters, seeds, PSI thresholds, and version metadata
keys live here. Import from this module to avoid hardcoding values.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"
MODELS_DIR = DATA_DELIVERY_DIR / "models"

# Multi-sport restructure (Phase A): repo-relative directory holding this
# sport's backend + data_delivery. Mirrored in master_pipeline.py (which
# needs the name before backend/ is importable) and frontend/sports_config.py
# (repo_subdir, the GitHub raw-URL prefix + local fallback dir). Phase C
# renames the directory to mlb-backend/ and flips all three at once; the
# path math above is already relative, so it survives the rename untouched.
SPORT_DIR_NAME = "mlb-backend"
REPO_SUBDIR = SPORT_DIR_NAME  # GitHub raw-URL path prefix for data_delivery

# Full-history weather backfill (default ON): real StatsAPI first pitches +
# strictly-prior Open-Meteo archive observations for EVERY decided game,
# cached by game_pk so each run fetches only games missing from the cache.
# Set MLB_WEATHER_BACKFILL_ALL=0 to keep the old trailing-35-day window only.
WEATHER_BACKFILL_ALL = os.getenv("MLB_WEATHER_BACKFILL_ALL", "1").strip().lower() in ("1", "true", "yes")

# Calibration map applied to the MONEYLINE blend after blending (calibration.py).
# "platt" = today's shipped 2-parameter logistic map; "identity" = publish the
# raw blend uncalibrated (calibrated == raw). Default stays "platt" until the
# blend-level gate (run_calibration_flip_test.py) passes; reversible via the
# CALIBRATION_MODE environment variable. Run-engine calibration is untouched.
CALIBRATION_MODE = os.getenv("CALIBRATION_MODE", "platt").strip().lower()

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
ELO_SEED = 1500  # Starting Elo for every team

# ---------------------------------------------------------------------------
# Elo parameters
# ---------------------------------------------------------------------------
ELO_K = 20  # Update factor
ELO_HOME_ADV = 65  # Home-field advantage in Elo points
ELO_REVERT_FACTOR = 1 / 3  # Season-to-season regression toward mean

# ---------------------------------------------------------------------------
# Feature rolling windows
# ---------------------------------------------------------------------------
WOBA_WINDOW = 30  # Games
BULLPEN_WHIP_WINDOW = 10  # Games
SP_ERA_WINDOW = 30  # Games
SP_K9_WINDOW = 30  # Games

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# Walk-forward FOLD cadence: fold boundaries every N days (weekly on the
# ~7,018-game frame -> ~74 folds). This is GEOMETRY, not a retrain schedule —
# the pipeline retrains the ensemble on EVERY run (no per-day dedup; the
# run-line/run-engine refits per run too). Do not repurpose this constant as
# a monitoring heuristic; the monitor's NEXT RETRAIN card uses
# NEXT_RUN_HEURISTIC_DAYS below.
RETRAIN_CADENCE_DAYS = 7
# Model-monitor "next expected run" heuristic for the NEXT RETRAIN card:
# retrain-every-run decision (2026-09-02) means the next expected run lands
# ~1 day after this run's emission. NOT scheduler-backed (no cron/next_run
# mechanism exists in the repo) — it drives card text only.
NEXT_RUN_HEURISTIC_DAYS = 1
# Walk-forward validation folds below this many games are skipped entirely:
# a handful of games (postseason tails, offseason gaps) produce wild AUC/Brier
# swings (e.g. AUC 0.18 on 11 games) that pollute pooled metrics and the
# adaptive blend weights earned from them.
MIN_VAL_FOLD_GAMES = 40

# Drift-monitor season-seam guard (2026-09-30): the production drift
# baseline is a ~21-day trailing window, so every season's final week is
# compared against mid-September — a cross-season seam. Playoff-roster
# bullpens and eliminated-team call-ups make that seam REGULARLY look like
# drift when it is seasonal (2026-09-29: bullpen_whip_10g_diff z=+2.78 and
# bullpen_whip_10g_away z=-3.37 vs the trailing baseline, BOTH vanishing
# against the same calendar phase of 2024-25: z=+1.94 / -1.28). When a
# feature's location_shift survives the trailing baseline, the monitor
# re-checks against these prior-season same-calendar-month windows and
# labels a clean re-check OK-SEASONAL instead of paging.
DRIFT_PHASE_EXTENSION_MONTHS = (-1, -2)

# Run-engine agreement filter: |moneyline_win_prob − derived_win_prob| above
# this marks a game as a CONFLICT on the dashboard and suppresses it from any
# future recommendation surface (see run_engine.agreement_stats).
AGREEMENT_FILTER_DELTA = 0.08
DEFAULT_MAX_EVAL_FOLDS = 0  # 0 = full history
TRAIN_TEST_SPLIT_RATIO = 0.2  # Not used directly; walk-forward handles splits

# Ensemble members and their blend weights. These are FALLBACK PRIORS used
# only for the cold start (fold 0 / before any OOF exists); after every fold
# the blend re-earns ADAPTIVE weights from the prior fold's out-of-sample
# log-loss (rolling per-fold weighting, 2026-09-16 spec).
# Roster (2026-09-16): xgboost + lightgbm + elastic-net logistic on the
# diff-column slice. RF/logistic/MLP seats removed after the member-audit
# program (MLP convicted at 3.5 sigma; enet beats logistic ~6 sigma solo).
# Equal thirds (2026-09-21): these priors apply ONLY when no OOF evidence
# exists yet (fold 0 of a walk-forward, or before the first evaluation) —
# the blend weights are earned from pooled OOF logloss from the first
# re-earning onward. Equal thirds make the no-evidence starting point
# assumption-free rather than audit-flavored.
ENSEMBLE_WEIGHTS = {
    "xgboost": 0.3333,
    "lightgbm": 0.3333,
    "elasticnet": 0.3334,
}

# Adaptive ensemble weighting: winner-take-all on pooled out-of-fold
# score (2026-09-21). The member with the best pooled OOF log-loss (or
# AUC under metric="auc") takes 100% of the blend weight; the softmax
# TEMPERATURE and the FLOOR/CAP band were removed as dead code — measured
# earned weights (xgb 0.39 / lgbm 0.30 / enet 0.31 at T=0.03) sat strictly
# inside [FLOOR, CAP], so the band never bound; the temperature alone kept
# the near-best member from converging to the weight it had earned.
# Blend-weight objective: "logloss" (softmax over pooled OOF log-loss).
# 2026-09-16: switched from "auc" to "logloss" as part of the rolling
# per-fold weighting spec — on the 84-fold walk-forward the rolling
# logloss-weighted blend measured 0.6817 vs 0.6822 for rolling AUC, and
# per-fold re-earning (fold k weighted by fold k-1 OOF) is now the
# production behavior. The AUC temperature constant is kept for
# reversibility but is unused under "logloss".
ADAPTIVE_WEIGHT_METRIC = "logloss"

# Local BLEND_SPACE experiment retained from the pre-existing worktree change.
# "prob" uses the historical weighted probability mean; "logit" uses the
# weighted logit mean. The production default remains logit in training.py.
BLEND_SPACE = "prob"
# Elastic-net logistic (the linear-family member, 2026-09-16). Mixed L1/L2
# penalty on the diff-column slice; l1_ratio 0.5 is the standard mix, C=0.03
# the grid optimum's strong-regularization edge (grid on the 84-fold
# walk-forward: all 12 cells beat plain-L2 logistic at 5.9-6.7 sigma paired;
# C=0.03 cells led with pooled OOF logloss 0.6849-0.6850 vs 0.6984).
ELASTICNET_PARAMS = {
    "penalty": "elasticnet",
    "l1_ratio": 0.5,
    "C": 0.03,
    "solver": "saga",
    "max_iter": 4000,
    "tol": 1e-4,
    "random_state": 42,
}

# Regularized: Optuna-tuned on 4,144-games/44-fold walk-forward
# (pooled OOF logloss 0.68107 vs 0.69115 for the old depth-5/300-r config).
# Shallow depth + high gamma + subsampling suppress variance in the
# MLB low-signal regime. The fold trainer adds early_stopping_rounds=20
# and n_estimators=2000 (generous ceiling ~50 median rounds at refit) when
# a validation window is available; fit-only refits use the params below
# directly with no early stopping. The member consumes the RAW NaN
# feature matrix (B2a, 2026-10-01): the same frame LightGBM gets,
# routed natively like the NHL tree members — XGB-only median
# imputation was a detour that made the two tree members see
# different data. Optuna tuned on the imputed matrix; the raw
# frame is the parity-correct representation, re-measured below.
# L5 re-tune (2026-09-21): causal 73-fold random search (seeded, 32 draws) on
# the 7,288-game frame + 3-seed confirmation (42/7/2026): pooled OOF member
# logloss 0.6789 -> 0.6770 mean, better on all three seeds (>=0.001 gate).
# Blend impact neutral across seeds (mean -0.0004, sign-mixed) -> adopted per
# the member-strength policy (RF precedent: member gains with blend unharmed).
# L6 re-tune (2026-09-28): 16-draw screen on the 40 most-recent folds +
# full-walk 3-seed confirm of the top-3 (record mlb_tune_l6_2026-09-28.json):
# every screen leader's gain reversed on the 73-fold walk (lightgbm worse on
# 3/3 seeds) -> params CONFIRMED, unchanged. Second consecutive retune the
# production config survives.
XGBOOST_PARAMS = {
    "max_depth": 2,
    "min_child_weight": 12,
    "gamma": 4.0,
    "subsample": 0.5406,
    "colsample_bytree": 0.6382,
    "learning_rate": 0.055,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
    "enable_categorical": True,
}
# RETUNE 2026-09-30 (max_depth 3 -> 1) — adopted, then PROVENANCE-
# CORRECTED 2026-09-30 after the first production run under the new
# regime zero-weighted the member (ensemble AUC 0.585 -> 0.561):
#   WHAT WAS ACTUALLY MEASURED: the retune's "incumbent 0.69876" was
#   depth-3 at the STATIC 50-round refit budget — a strawman the real
#   production incumbent (early-stopped on 2000/20 at its own
#   operating point) never ran at; the early-stopped incumbent had
#   pooled 0.678/AUC 0.586 for nine consecutive production runs
#   (0921-0928, 100% blend weight). The adoption decision (depth-1
#   beats depth-3 AT THE 50-ROUND REFIT POINT) was reproduced and
#   CONFIRMED: 3 seeds, mean 0.69020 vs 0.69818 (AUC 0.5551 vs
#   0.5521) — the depth-1 stump is genuinely right for the deployed
#   refit bundle.
#   WHAT THE PRODUCTION WALK REGIME ACTUALLY SCORES (three-arm
#   experiment, 78 production folds, 7,378 games; plus 3-seed
#   reruns): depth3+causal-rounds 0.68762/AUC 0.5511,
#   depth1+causal-rounds 0.68770/0.5495 (parity, d_ll mean
#   +0.00055, d3 wins 1/3), depth3+val-selected-early-stop
#   0.67671/0.5943. The honest causal regime LOSES ~0.011 logloss /
#   ~0.04 AUC versus the leak-flattered old numbers, at EITHER
#   depth: the pre-0929 0.678/0.586 was never an honest incumbent,
#   so the ensemble's headline drop 0.585 -> 0.561 is the leak fix
#   itself finally measuring honestly — NOT a stump regression.
#   Honest per-fold adaptivity alternatives were tested and
#   rejected: fixed-150 0.7207 (early-fold overfit), internal-split
#   early stop (last-20%/30% of train, min 200/300) selects ~9
#   rounds with seed-spread AUC 0.544-0.556 — no better than the
#   causal prior-fold median.
#   OBLIGATION: the member now legitimately ranks below
#   elasticnet/lightgbm and earns its weight per fold from the
#   SLSQP blend; if it stays at/near 0% across future runs that is
#   the blend telling the truth about the honest regime. Any future
#   retune must re-measure BOTH operating points (walk-causal and
#   refit-static) and compare against the honest numbers here —
#   never against a re-hobbled baseline.
# RETUNE 2026-09-30 PM (max_depth 1 -> 2; OWNER DIRECTIVE: the shipped
# member must not be a depth-1 stump). A depth-family sweep under the
# honest causal walk showed a plain depth flip cannot win: at each
# depth's own measured budget, refit logloss is monotone in depth
# (d1@20 0.68755 / d2@12 0.68776 / d3@11 0.68799 / d4@11 0.68852,
# 3 seeds; the retune commit eecbbd6's "d3 loses ~0.008" compared
# d3@50 — a rounds mismatch, not a depth verdict). A 32-config
# one-axis screen (seed 42) then found depth-2 winners, and a JOINT
# search over the winning axes (subsample x mcw x lr) adopted:
#   max_depth 2, subsample 0.5406, gamma 4.0, lr 0.055 (other params
#   unchanged; the config's own walk probe-median is ~19-26 rounds).
#   Evidence (78 production folds, 7,378 games; seal = folds 70-77,
#   selection on folds 0-69 only):
#   * 3-seed full-walk: walk 0.68706/AUC 0.5540, static refit
#     0.68662/0.5562 — vs depth-1 incumbent 0.68777/0.5506 and
#     0.68755/0.5528 (better on every seed, both operating points).
#   * SEALED member (3 seeds, frozen budget @22): 0.68089/0.5861 vs
#     depth-1 0.68107/0.5842 — the ONLY challenger of five sealed
#     (mcw24+lr0.055 0.68155, joint mcw24 family, NHL-style 0.68977)
#     to beat the incumbent on the untouched window.
#   * Blend: full-walk ens 0.68609/0.5554 vs incumbent 0.68635/0.5547
#     with weights re-earned toward thirds (en .333/lg .333/xgb .333);
#     frozen-weight blend seal 0.67896/0.5873 vs incumbent 0.67854/
#     0.5878 (-0.0004 ll, inside noise; seal AUC 0.5861 vs 0.5842).
#   Gate accounting: no member logloss gate clears in either direction
#   (largest honest delta -0.0009 refit); adoption rests on the owner's
#   no-stump directive plus a consistent member-cell AUC edge (+0.0019
#   to +0.0034 everywhere) and a neutral-to-positive blend. Rejected on
#   the seal: plain d2 (0.68155 seal), single-axis winners, and the
#   NHL-style d2 block (colsample 0.4025 collapses AUC to ~0.54 here).
# L7 FROM-SCRATCH TUNE (2026-09-30 PM, mlb_tune_l7_2026-09-30.json):
# 89-trial seeded search over ALL SIX param dims (incumbent included as
# same-harness reference), selection on folds 0-69 by refit ll. Every
# top-8 margin <0.001; the three depth-eligible finalists REVERSED or
# fell below the gate on the 3-seed full walk (best: -0.00011 refit /
# -0.00032 walk vs the 0.001 gate). CONFIG CONFIRMED — third
# consecutive retune the production block survives (L5 adopted 0921;
# L6, L7 confirmed). The search's one consistent signal — slower lr +
# stronger column/row shrinkage — is already the direction this block
# moved; depth is not the discriminator once shrinkage is right (d1-d4
# all appear in the top 8).
# L7 ADDENDUM (owner directive: no parameter off the table): REFIT/
# FOLD0_ROUNDS re-sealed at the DEPLOYED depth-2 block — @30 beat @50
# by -0.00127 (3/3) on the full walk but LOST the seal 0/3 (-0.00184
# the other way): split verdict = window noise, budget stays 50.
# Overcooking is real (@75/@100 degrade monotonically). EARLY_STOP
# screened {10,20,40} on the causal walk: 20 confirmed (clear of 10,
# tied with 40). The early-stopped probe is measurement-only in the
# causal pipeline (shipped fold model never sees its own val), so the
# patience is a tunable — but it was kept fixed during the main search
# and scrutinized only in the addendum so the search never selected
# its own measurement noise.
# n_estimators ceiling + early-stopping rounds for walk-forward folds.
# Separate from the constructor dict because xgboost 3.2 sklearn API
# requires eval_set when early_stopping_rounds is set, and the full-refit
# path has no validation window.
XGBOOST_FOLD_ROUNDS = 2000
XGBOOST_EARLY_STOP = 20
# CAUSAL FOLD ROUNDS (2026-09-30 PIT review): fold k's SHIPPED XGBoost
# model is fit WITHOUT any eval_set at the median of PRIOR folds'
# measured best iterations — the fold's own val window never selects the
# shipped model's round count (the early-stopped fit is only a
# measurement that later folds may consume). Fold 0 (no prior evidence)
# and any refit without fold measurements use the static priors here.
# Fold0/refit statics cover cold-start paths only; the production walk's
# own probe-median is ~20 rounds at depth 1 and ~19-26 under the adopted
# depth-2 block. The 50-round prior was kept over a 20-round re-pin: the
# 2026-09-30 sealed-window check split the verdict (d1@50 ll 0.68013 vs
# d1@20 0.68107 on folds 70-77; AUC the other way), matching the LGBM
# rounds precedent — a tune-gain must survive the seal, not just the
# OOF walk.
XGBOOST_FOLD0_ROUNDS = 50
XGBOOST_REFIT_ROUNDS = 50
# Optuna-tuned on 4,159-games/44-fold walk-forward (tune_lightgbm_optuna.py,
# 50 trials, pooled OOF logloss 0.68066 vs 0.78465 for the old depth-5/300-r
# config; sealed holdout 2026-08-03→08-23 confirmed: 0.68150/AUC 0.5573 vs
# 0.71444/0.5492). Strongly regularized: tiny leaf count + high min gain +
# heavy bagging suppress variance in the MLB low-signal regime. Native NaN
# routing kept (impute_medians=False won); team-ID categoricals route via
# categorical_feature BY NAME in the fold trainer — unchanged.
# Full-harness rounds test (run_lgb_rounds_test.py,
# lgb_rounds_22e1c5cb7895afe2, 65-col matrix incl. run_margin_diff, sealed
# 284-game holdout 2026-08-05→08-25): 15/17/20-round variants improve
# pooled OOF (logloss 0.6844→0.6837, ECE-cal 0.0085→0.0046) but degrade
# sealed-holdout ECE-cal (0.0566→0.0669 at 17r) → the member-level
# calibration gain does not survive the blend; config unchanged, 50 rounds
# stays.
#
# Fair winner re-verification (verify_lgb_winner.py, 2026-08-26): the
# Colab Optuna winner was re-measured under its OWN early-stopping
# discipline (not the tuner's forced-50 clamp). At its natural count
# (6 rounds, tail-early-stop condition) the winner scored sealed-holdout
# 0.6917 LL / 0.5134 AUC / 0.0401 ECE vs current (LIGHTGBM_PARAMS, 17r)
# 0.6823 / 0.5499 / 0.0548 → the winner still loses logloss and AUC on
# the sealed set. Verdict: DON'T ADOPT; this config stays.
# L5 re-tune (2026-09-21): same causal 73-fold search + 3-seed confirmation:
# pooled OOF member logloss 0.6863 -> 0.6845 mean, better on all three seeds.
# Elastic-net searched too: production C=0.03 / l1_ratio 0.5 confirmed optimal
# on the big frame (best challenger -0.0001, below the 0.001 gate) - unchanged.
LIGHTGBM_PARAMS = {
    "n_estimators": 50,
    "max_depth": 6,
    "num_leaves": 6,
    "min_child_samples": 70,
    "min_gain_to_split": 1.2224,
    "bagging_fraction": 0.4518,
    "bagging_freq": 1,
    "feature_fraction": 0.7632,
    "learning_rate": 0.0332,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}
# MLP member — small neural net with early stopping; the ensemble's
# diversity wildcard whose weight is earned (or starved) by the adaptive
# blend. Params below are the production values (hidden 32x16, alpha 0.01),
# plus the sklearn defaults the fold trainer relied on implicitly. MLP
# cannot consume NaN: train-fold-median imputation + StandardScaler, never
# a native-NaN route.
#
# Optuna study (tune_mlp_optuna.py, 2026-08-25, 75 trials, same 44
# walk-forward folds as production): best pooled OOF logloss 0.70992
# (hidden 32x16, tanh, alpha 1.1e-3, adaptive lr 2.3e-3, batch 256,
# max_iter 600, val 0.1, n_iter_no_change 10) vs current 0.79879 — the
# winner's edge came from the tiny early folds. On the SEALED 21-day
# holdout (refit on all pre-holdout games, the production condition) the
# current config WON: logloss 0.70283 vs 0.71069, AUC 0.5263 vs 0.5066.
# Verdict: NOT ADOPTED — the current config stays (honesty contract).
#
# Harness reconciliation (2026-08-25): per-fold MLP probabilities are
# bit-identical to production's walk_forward_evaluate on the same folds
# (logistic control 0.7052 matches to 4dp; AUC identical), and the pooled
# numbers agree exactly once the tuner clips at the same 1e-7 as
# compute_metrics (the 1e-6 clip had hidden 19 degenerate early-fold
# points: 0.79105 vs 0.79879). The ~0.7076 figure from the v2026.08.24
# run was a different data snapshot (47 folds), not a harness artifact.
MLP_PARAMS = {
    "hidden_layer_sizes": (32, 16),
    "alpha": 0.01,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "max_iter": 300,
    "learning_rate": "constant",
    "learning_rate_init": 0.001,
    "batch_size": "auto",
    "n_iter_no_change": 10,
    "activation": "relu",
    "random_state": RANDOM_SEED,
}

# Random Forest member — bagged trees, decorrelated from boosting errors;
# ~20% of the blend weight before adaptive re-weighting. sklearn trees
# cannot consume NaN: train-fold-median imputation (the same imputed
# matrix logistic/MLP use), plus integer team-ID categoricals when
# RF_WITH_TEAM_IDS is on (the production default).
#
# PROVENANCE (2026-08-31): tuned by backend/tune_rf_optuna.py — 75 Optuna
# trials, pooled OOF logloss objective over 71 walk-forward folds with OOF
# run_margin_diff attached (production-correct), study 'rf_moneyline'
# (sqlite:///rf_study.db).
#   MEMBER GATE (PASSED): winner vs the prior inline defaults (300 trees /
# min_samples_leaf 20): pooled OOF logloss 0.68655 vs 0.68783; sealed
# 21-day holdout (2026-08-09..08-29, n=282) logloss 0.68050 vs 0.68150,
# AUC 0.5783 vs 0.5770, ECE 0.0357 vs 0.0593 — all three improve on both
# views, so the tuner's harness gate printed ADOPT.
#   BLEND GATE (NEUTRAL, adopted per policy): run_rf_tuned_blend_ablation.py
# (production-correct, 3 sealed windows) shows the member gain does NOT move
# the blend — pooled blend ll 0.6836 vs 0.6837 / AUC 0.5657 vs 0.5656
# (slightly better), sealed deltas all within ±0.001 AUC / ±0.0006 ll
# (mixed-sign noise), strict multi-window verdict DON'T SHIP (0/3).
# Policy (user, 2026-08-31): each member as strong as possible as long as
# the blend is not measurably impacted — the tuned RF qualifies, so these
# params are ADOPTED. Trade-off: 800 trees makes the RF fold fit ~2.7x
# slower than 300.
RF_PARAMS = {
    "n_estimators": 800,
    "max_depth": 6,
    "min_samples_leaf": 17,
    "min_samples_split": 6,
    "max_features": "log2",
    "bootstrap": True,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# ---------------------------------------------------------------------------
# Recursive feature elimination (moneyline blend-level RFE) — see
# feature_selection.py. Seeded and deterministic: same data → same verdict.
# ---------------------------------------------------------------------------
RFE_FLOOR = 25                 # never prune below this many features
RFE_AUC_GUARD = 0.003          # adopt step only if pooled AUC drop <= this
RFE_ECE_GUARD = 0.005          # adopt step only if pooled ECE rise <= this
                               # (reverted to the original strict bound on
                               # owner decision 2026-09-15; the 0.010 loosening
                               # vetoed fewer sharpness trades but let real
                               # calibration blows-ups ride — 0.005 trips them,
                               # e.g. the rest_days_diff 0.0138 rise)
RFE_MIN_LOGLOSS_GAIN = 0.0005  # floor on the commit threshold (never commit on
                               # a smaller measured gain, even at huge n)
RFE_NOISE_SIGMA = 1.0          # commit threshold = max(floor, this many
                               # standard errors of the PAIRED per-game
                               # logloss difference (trial - baseline, same
                               # folds/games scored twice). Paired differencing
                               # cancels shared per-game difficulty, so the SE
                               # reflects the CHANGE's noise, not the level's.
                               # 1.0 as of 2026-09-23 (owner decision): parity
                               # with the NFL RFE bar (RFE_COMMIT_SE_MULTIPLE
                               # = 1.0) — the paired construction still makes
                               # junk features self-calibrate a high bar (their
                               # per-game diffs bounce randomly), so the looser
                               # multiple does not admit noise commits.
RFE_MAX_STEPS = 40             # walk-forward scoring evaluations per RFE run
RFE_REDUNDANCY_R = 0.9         # |r| above which two pool features are flagged
                               # redundant_with each other (informational: they
                               # are trialed consecutively so one measured verdict
                               # informs its sibling)
RFE_GRID_MAX_STATES = 16       # grid-mode lattice cap (MLB_RFE_ADDITION_REMOVAL_
                               # GRID_MODE=1): 2^adds x 2^removes states max.
                               # 16 = the 4-adds-or-4-removes shapes (4+0, 3+1,
                               # 2+2, 1+3, 0+4 are all exactly 16 states). Env
                               # override RFE_GRID_MAX_STATES; read ONLY when
                               # grid mode is on — ignored (not required) for
                               # normal/targeted runs. Windowing silently
                               # round-robins adds/removes to fit the cap; the
                               # dropped tail is recorded in the trace and overflow
                               # names still get 1-by-1 targeted trials.
# The RFE candidate pool: generated, PIT-safe columns NOT currently in the
# moneyline universe that feature_selection.py may additionally trial
# (universe members are always in scope). Grouped in trial-priority order;
# every entry is computed pre-game. Blocked by design: identity/metadata
# (game_pk, game_id, game_date, team/venue, start times), targets and
# post-game fields (home_win, scores, total_runs), string-typed cols
# (records, starter hands), and dead all-NaN cols (lineup_rest_count_*).
# Serving risk is handled at adoption time: confirm_slate_coverage() builds
# the real upcoming slate and refuses any adopted col below
# RFE_SLATE_COVERAGE_FLOOR non-null coverage.
RFE_CANDIDATE_COLS = [
    # --- form deltas (5g/10g vs season momentum) (42) ---
    "sp_era_delta_home",
    "sp_k9_delta_home",
    "sp_bb9_delta_home",
    "sp_whip_delta_home",
    "sp_xwoba_delta_home",
    "sp_fbvelo_delta_home",
    "sp_fbpct_delta_home",
    "sp_whiff_delta_home",
    "woba_delta_home",
    "team_iso_delta_home",
    "team_k_rate_delta_home",
    "team_bb_rate_delta_home",
    "team_barrel_delta_home",
    "team_hardhit_delta_home",
    "team_exitvelo_delta_home",
    "bullpen_whip_delta_home",
    "bullpen_era_delta_home",
    "lineup_re24_mean_delta_home",
    "lineup_re24_top3_delta_home",
    "sp_era_delta_away",
    "sp_k9_delta_away",
    "sp_bb9_delta_away",
    "sp_whip_delta_away",
    "sp_xwoba_delta_away",
    "sp_fbvelo_delta_away",
    "sp_fbpct_delta_away",
    "sp_whiff_delta_away",
    "woba_delta_away",
    "team_iso_delta_away",
    "team_k_rate_delta_away",
    "team_bb_rate_delta_away",
    "team_barrel_delta_away",
    "team_hardhit_delta_away",
    "team_exitvelo_delta_away",
    "bullpen_whip_delta_away",
    "bullpen_era_delta_away",
    "lineup_re24_mean_delta_away",
    "lineup_re24_top3_delta_away",
    # lineup_actual_*/lineup_rest_count_* REMOVED 2026-09-26. They were
    # addition-ELIGIBLE here only because feature_selection.CANDIDATE_COLS is
    # "RFE_CANDIDATE_COLS minus MONEYLINE_FEATURE_COLS" -- and these four left
    # the model on 2026-08-29 as a train-serve skew fix. Leaving known-leaked,
    # never-scored columns in the addition pool invites an RFE run to select
    # them on post-game-actual signal. The pipeline no longer computes them.
    # --- diff cols culled from universe, never ablated (S-family cull 2026-09-07) (6) ---
    # sp_k9_diff REMOVED from this list 2026-09-30: readmitted to the
    # MONEYLINE_FEATURE_COLS universe (user-directed structural alignment —
    # its raw twins sp_k9_home/away serve since 2026-09-27), so it left the
    # addition pool by the universe-minus rule below.
    "sp_k9_5g_diff",
    "sp_fbpct_diff",
    "sp_whiff_diff",
    "sp_xwoba_diff",
    "sp_xwoba_vs_l_diff",
    "bullpen_ip_diff",
    # --- raw per-side levels (shadowed by diff-only routing) (76) ---
    "rest_days_home",
    "rest_days_away",
    "sp_bb9_home",
    "sp_whip_home",
    "sp_fip_home",
    "sp_bb9_away",
    "sp_whip_away",
    "sp_fip_away",
    "sp_era_5g_home",
    "sp_k9_5g_home",
    "sp_era_5g_away",
    "sp_k9_5g_away",
    "team_iso_30g_home",
    "team_k_rate_30g_home",
    "team_bb_rate_30g_home",
    "team_iso_30g_away",
    "team_k_rate_30g_away",
    "team_bb_rate_30g_away",
    "bullpen_era_10g_home",
    "bullpen_era_10g_away",
    "sp_fbvelo_3g_home",
    "sp_fbpct_3g_home",
    "sp_whiff_3g_home",
    "sp_xwoba_vs_l_home",
    "sp_xwoba_vs_r_home",
    "sp_fbvelo_3g_away",
    "sp_fbpct_3g_away",
    "sp_whiff_3g_away",
    "sp_xwoba_vs_l_away",
    "sp_xwoba_vs_r_away",
    "team_hardhit_15g_home",
    "team_hardhit_15g_away",
    "opp_lefty_share_home",
    "opp_lefty_share_away",
    "bullpen_pitches_3d_home",
    "bullpen_ip_3d_home",
    "bullpen_pitches_3d_away",
    "bullpen_ip_3d_away",
    "lineup_ops_vs_l_home",
    "lineup_ops_vs_r_home",
    "lineup_ops_vs_l_away",
    "lineup_ops_vs_r_away",
    "lineup_ops_vs_starter_hand_home",
    "lineup_ops_vs_starter_hand_away",
    "time_zones_crossed_last_3d_home",
    "time_zones_crossed_last_3d_away",
    "closer_available_home",
    "closer_available_away",
    "lineup_re24_std_home",
    "lineup_re24_std_away",
    "lineup_il_flag_home",
    "lineup_il_flag_away",
    "lineup_il_flag_diff",
    # Bullpen readiness (2026-09-30): computed + metadata-authored but NOT
    # in the serving width yet — adoption runs the standard ablation gate
    # (3-seed walk + seal) before these can be RFE-selected.
    "bullpen_budget_2d_home",
    "bullpen_budget_2d_away",
    "bp_ready_share_home",
    "bp_ready_share_away",
    "sp_k_pct_cat_fastball_home",
    "sp_k_pct_cat_fastball_away",
    "sp_k_pct_cat_breaking_home",
    "sp_k_pct_cat_breaking_away",
    "sp_k_pct_cat_offspeed_home",
    "sp_k_pct_cat_offspeed_away",
    "sp_usage_cat_fastball_home",
    "sp_usage_cat_fastball_away",
    "sp_usage_cat_breaking_home",
    "sp_usage_cat_breaking_away",
    "sp_usage_cat_offspeed_home",
    "sp_usage_cat_offspeed_away",
    "sp_xwoba_cat_fastball_home",
    "sp_xwoba_cat_fastball_away",
    "sp_xwoba_cat_breaking_home",
    "sp_xwoba_cat_breaking_away",
    "sp_xwoba_cat_offspeed_home",
    "sp_xwoba_cat_offspeed_away",
    "sp_k_pct_fb_vs_l_home",
    "sp_k_pct_fb_vs_l_away",
    "sp_k_pct_fb_vs_r_home",
    "sp_k_pct_fb_vs_r_away",
    "sp_pa_fb_vs_l_home",
    "sp_pa_fb_vs_l_away",
    "sp_pa_fb_vs_r_home",
    "sp_pa_fb_vs_r_away",
    # --- pitch-arsenal / matchup category stats (28) ---
    "league_k_pct_cat_fastball",
    "league_k_pct_cat_breaking",
    "league_k_pct_cat_offspeed",
    "league_xwoba_cat_fastball",
    "league_xwoba_cat_breaking",
    "league_xwoba_cat_offspeed",
    "league_k_pct_fb_vs_l",
    "league_k_pct_fb_vs_r",
    "team_k_pct_cat_fastball_home",
    "team_k_pct_cat_fastball_away",
    "team_k_pct_cat_breaking_home",
    "team_k_pct_cat_breaking_away",
    "team_k_pct_cat_offspeed_home",
    "team_k_pct_cat_offspeed_away",
    "team_xwoba_cat_fastball_home",
    "team_xwoba_cat_fastball_away",
    "team_xwoba_cat_breaking_home",
    "team_xwoba_cat_breaking_away",
    "team_xwoba_cat_offspeed_home",
    "team_xwoba_cat_offspeed_away",
    "team_k_pct_fb_vs_l_home",
    "team_k_pct_fb_vs_l_away",
    "team_k_pct_fb_vs_r_home",
    "team_k_pct_fb_vs_r_away",
    "team_pa_fb_vs_l_home",
    "team_pa_fb_vs_l_away",
    "team_pa_fb_vs_r_home",
    "team_pa_fb_vs_r_away",
    # --- league-context aggregates (1) ---
    "league_k_pct",
    # --- environment/schedule per-side extras (10) ---
    "home_wins",
    "home_losses",
    "away_wins",
    "away_losses",
    "home_run_diff",
    "away_run_diff",
    "park_wind_factor",
    "air_density_level",
    "dome_is_neutral_game",
    "park_factor_slug",
    # --- lineup actuals deltas (posted lineup vs probable) (0) ---
]
RFE_SLATE_COVERAGE_FLOOR = 0.5  # min non-null share on a real upcoming
                                # slate for an adopted candidate feature

# ---------------------------------------------------------------------------
# Coin-flip threshold
# ---------------------------------------------------------------------------
COIN_FLIP_THRESHOLD = 0.02  # |P - 0.5| < this → coin flip

# ---------------------------------------------------------------------------
# PSI thresholds
# ---------------------------------------------------------------------------
PSI_WARN_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25

# ---------------------------------------------------------------------------
# Artifact date format
# ---------------------------------------------------------------------------
DATE_FMT = "%Y%m%d"
DATE_READABLE_FMT = "%B %d, %Y"

# ---------------------------------------------------------------------------
# Version metadata keys
# ---------------------------------------------------------------------------
VERSION_KEY = "VERSION"
TRAINED_AT_KEY = "TRAINED_AT"
DATA_CUTOFF_KEY = "DATA_CUTOFF"

# ---------------------------------------------------------------------------
# Supported sports (MLB primary, others scaffolded)
# ---------------------------------------------------------------------------
SUPPORTED_SPORTS = ["MLB", "NBA", "NHL", "NFL", "CFB", "CBBM", "Tennis"]

# ---------------------------------------------------------------------------
# Tracking strings
# ---------------------------------------------------------------------------
FEATURE_DRIFT = "feature_drift"
TODAYS_GAMES = "todays_games"
POWER_RANKINGS = "power_rankings"
CALIBRATION = "calibration"
MODEL_MONITOR = "model_monitor"
SHAP_GAME = "shap_game"
MODEL_HISTORY = "model_history"
ENSEMBLE_FILE = "ensemble_latest.joblib"
