"""Central configuration for MLB Bet Predictor backend.

All paths, hyperparameters, seeds, PSI thresholds, and version metadata
keys live here. Import from this module to avoid hardcoding values.
"""
import os
from datetime import date, datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Run window — stale-pin guard (2026-10-06 log review, T7)
# ---------------------------------------------------------------------------

def resolve_run_end_date(pinned: str) -> str:
    """Resolve MLB_END_DATE with a stale-pin guard.

    The Kaggle notebook pins a literal ``MLB_END_DATE`` for rebuilds
    ("set ONLY for a backfill") and leaves the pin there afterwards — the
    run target is ``end`` (master_pipeline: "predict the last date in the
    range"), so a pin that resolves before today would freeze the daily
    slate on every later run: the 2026-10-06 rebuild pinned 2026-10-06,
    and from 2026-10-07 on the pipeline would keep re-predicting 10-06.
    The notebook is Kaggle-owned (standing guardrail: never edit it from
    the repo — Kaggle Version 5 restored the active pin the moment we
    commented it out), so the PIPELINE defends itself instead: a pin
    before today extends to TODAY, keeping daily runs fresh. Today is the
    same clock as the default fallback (server-local ``date.today()``).

    A pin that still reaches today, a deliberate future window, and
    malformed input all pass through untouched — Phase 1's ``strptime``
    keeps failing loudly on malformed input instead of silently defaulting,
    and a same-day rebuild pin keeps working exactly as written.
    """
    try:
        pin = datetime.strptime(pinned, "%Y-%m-%d").date()
    except ValueError:
        return pinned
    today = date.today()
    return today.isoformat() if pin < today else pinned


# ---------------------------------------------------------------------------
# Delivery-root reconciliation (2026-10-07 run-log review, T1)
# ---------------------------------------------------------------------------

def ensure_config_root(expected_backend, *, log=print) -> bool:
    """Repoint an already-imported foreign ``config`` at the running clone.

    The daily run can execute in TWO trees at once: if the launch kernel
    imported ``config`` from its own checkout BEFORE master_pipeline put
    the fresh clone's ``backend/`` at ``sys.path[0]``, every later
    ``from config import DATA_DELIVERY_DIR`` resolves against the KERNEL's
    tree while cwd, the run-log tee and Phase 5's staging scan all point
    at the CLONE. The 2026-10-07 MLB run split exactly that way (tee +
    features' lineup root at ``/content/sports_prediction_model`` vs
    artifacts at ``/kaggle/working/sports_prediction_model``):

      * all 15 Step-5 artifacts (todays_games_*, model_monitor_*, ...)
        were written outside every tree Phase 5 scans, so GitHub kept
        serving the PREVIOUS day's board while the pushed log claimed a
        successful delivery (remote had zero ``*20261007*`` files);
      * the lineup capture wrote ``lineups.parquet`` (dated 10-07) to the
        kernel root while features read the clone's stale 10-06 copy, so
        ``pl_slate_20261006.parquet`` shipped for a 10-07 slate and the
        loader (which reads the kernel root) found nothing — all four
        slate sides priced on the marked carry.

    Reconciliation loads ``<expected_backend>/config.py`` BY PATH into
    ``sys.modules['config']`` and purges every already-imported module
    whose ``__file__`` lives in the foreign backend directory (those
    modules hold stale ``from config import ...`` bindings). Modules
    outside that directory — the notebook kernel's own state — are never
    touched.

    Returns True when a foreign config was replaced. Never raises: a
    missing/unreadable clone config leaves the run on its current roots
    (Phase 5's union scan + reported-artifact gate still deliver), and
    the caller announces the outcome into the pushed run log.
    """
    import importlib.util
    import sys

    try:
        expected = Path(expected_backend).resolve()
        target = expected / "config.py"
        current = sys.modules.get("config")
        current_file = getattr(current, "__file__", None)
        foreign_backend: Path | None = None
        if current_file:
            try:
                foreign_backend = Path(current_file).resolve().parent
                if foreign_backend == expected:
                    return False  # already the clone's config — nothing to do
            except OSError:
                foreign_backend = None
        if not target.exists():
            log(f"  ⚠️  config-root reconciliation skipped: {target} not "
                f"found (running on {current_file or 'an unimported config'})")
            return False
        spec = importlib.util.spec_from_file_location("config", target)
        if spec is None or spec.loader is None:
            log(f"  ⚠️  config-root reconciliation skipped: cannot load {target}")
            return False
        module = importlib.util.module_from_spec(spec)
        previous = sys.modules.get("config")
        sys.modules["config"] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 — a broken clone config must
            # not take the run down: restore the module that was there.
            if previous is not None:
                sys.modules["config"] = previous
            else:
                sys.modules.pop("config", None)
            log(f"  ⚠️  config-root reconciliation failed ({exc}) — keeping "
                f"{current_file or 'the current roots'}")
            return False
        purged: list[str] = []
        if foreign_backend is not None:
            for name, mod in list(sys.modules.items()):
                # '__main__' may BE the foreign master_pipeline still
                # executing this very reconciliation — never unload it.
                if name in ("config", "__main__"):
                    continue
                mod_file = getattr(mod, "__file__", None)
                if not mod_file:
                    continue
                try:
                    mod_path = Path(mod_file).resolve()
                except OSError:
                    continue
                if foreign_backend in mod_path.parents:
                    sys.modules.pop(name, None)
                    purged.append(name)
        log(f"  📦 config root reconciled → {target} "
            f"(DATA_DELIVERY_DIR={module.DATA_DELIVERY_DIR})"
            + (f"; purged {len(purged)} foreign module(s): "
               f"{', '.join(sorted(purged))}" if purged else ""))
        return True
    except Exception as exc:  # noqa: BLE001 — reconciliation is best-effort
        log(f"  ⚠️  config-root reconciliation skipped ({exc})")
        return False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"
# Training-process residue (RFE workbooks, tune/analysis dumps) is NOT
# delivery: nothing reads it back, so it writes to this local gitignored dir
# (NHL 2026-09-30 retention-audit pattern, adopted for parity 2026-10-04).
RUN_DIAGNOSTICS_DIR = ROOT_DIR / "run_diagnostics"
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
# Same 109 numeric names, corrected source meanings; rebuild + refit together.
FEATURE_SCHEMA_VERSION = "mlb-v2-observed-statcast-causal-blend"
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
# WEIGHT-POLICY SCREEN (2026-10-05 run-log remediation): every softening
# variant was measured on the production walk (83 folds,
# min_train_days=30) under the same rolling causal protocol — earned-
# weight floors 0.05-0.25, shrink-to-priors 0.1/0.25/0.5, trailing-
# window earning (5/10/20/40 folds) and metric="auc". All neutral to
# negative inside noise: pooled log-loss within 0.0002 and AUC within
# 0.0002 either way; trailing strictly worse (pooled AUC 0.5677-0.5701
# and log-loss 0.6827-0.6834 vs 0.5707/0.6825). NONE adopted — the zero
# xgboost weight was a member-quality defect (underfit shipped rounds,
# see the XGBOOST_FOLD0_ROUNDS addendum), fixed at the shipped-rounds
# layer so the member re-earns its weight honestly. The floor/cap band
# stays removed; weights stay fully earned.
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
# L8 re-tune (2026-10-03): the same FULL RESET that re-tuned LightGBM ran
# XGBoost too (12 dims, both operating points) — params CONFIRMED,
# unchanged. Best candidate screened well (20-fold d -0.0030) and cleared
# the 3-seed causal floor by only 4% (mean d_tune -0.00391 vs a 0.00374
# floor, measured on 8 verify folds vs the floor's 12), and refit-static
# did NOT confirm (-0.00121 vs its own 0.00237 floor, seal positive). The
# full 82-fold tune surface then settled it: the challenger is WORSE on
# all three metrics (logloss +0.00104, brier +0.00049, AUC -0.00296) —
# the earlier causal "gain" was an artifact of the narrow verify surface.
# Blend-level logloss/AUC 0.6835 / 0.567 unchanged either way.
# 2026-10-09 feature-family test (record mlb_xfip_ab_20261009.json): the
# serving set was A/B/C-tested against two replacements —
#   A = the shipped ERA/WHIP serving set (incumbent);
#   B = every ERA-family column (+ the 4 served ERA composites) swapped to
#       point-in-time xFIP analogues built from pitches.parquet, attached
#       LAG-first (no same-day self-inclusion);
#   C = B + bullpen WHIP -> K-BB% (+ the 3 served meltdown composites).
# Paired causal walk (seeds 42/7/2026, min_train_days=30, UNROUNDED OOF
# logloss): B's blend deltas [-0.00010, -0.00005, +0.00021] all ride inside
# the 0.000681 seed-noise floor (pool gate FAIL, seal non-refuting); C at
# seed 42 is +0.00035 worse than A. Params CONFIRMED, unchanged.
# Artifacts: data_delivery/mlb_xfip_ab_20261009.json,
# docs/mlb_xfip_ab_20261009.md.
#
# ADOPTED 2026-10-09 (user-directed): Scenario C is now the production
# feature set. The ERA-family columns became point-in-time xFIP analogues
# (sp_era* -> sp_xfip_*, bullpen_era* -> bullpen_xfip_*) and the bullpen
# WHIP family became K-BB% (bullpen_whip_* -> bullpen_kbb_*), built from
# pitches.parquet over the IDENTICAL windows/shrink/staleness gates and
# LAG-first (no same-day self-inclusion). The 4 served ERA composites
# (pitcher_regression_indicator_*, wind_advantage_flyball_factor) and the
# 3 meltdown composites recompute on the new inputs. This is a rename +
# value swap at the SAME column positions: serving width stays 109 and the
# model-matrix order is unchanged, so the next refit trains the shipped
# members on the xFIP/K-BB% estimators. sp_whip_* (the STARTER WHIP
# family) is untouched — it was not part of the A/B/C test and is not
# served in the 109.
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
    # ── DETERMINISM BLUEPRINT (2026-10-08 full-tune review) ────────────────
    # tree_method='hist' pinned EXPLICITLY (it is the 3.x default, but the
    # blueprint forbids leaving the algorithm to a version-dependent
    # default) and n_jobs=-1: full multi-threaded training, never a
    # 1-thread throttle, with a fixed random_state so run N and run N+1
    # are byte-identical. Verified on this machine: two complete fits of
    # this exact block at n_jobs=-1 produce bit-identical
    # predict_proba, and two full walk-forward baseline walks produced
    # identical pooled + per-member OOF metrics.
    # The blueprint's deterministic_histogram=True is INTENTIONALLY
    # ABSENT: xgboost 3.4.0 removed the parameter upstream (NEWS.md:
    # "we removed the parameter deterministic_histogram and now ...
    # always deterministic"); passing it only emits "Parameters: {
    # deterministic_histogram } are not used" on every fit. Compliance
    # is by construction (hist + fixed seed + verified bit-identity),
    # not by carrying a dead key that warns in production logs.
    "tree_method": "hist",
    "n_jobs": -1,
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
# L9 FROM-SCRATCH TUNE (2026-10-08, record mlb_tune_20261008.json):
# the determinism blueprint was activated FIRST (tree_method='hist' +
# n_jobs=-1 below; two full 83-fold walks proved bit-identical pooled,
# member and weight outputs), then the 3-seed noise floor was measured
# BEFORE any search (member pooled-OOF-logloss range 0.0012, blend
# 0.0003), 32 seeded random draws screened on a 20-fold surface whose
# last 4 folds were SEALED off (truncated-frame geometry), and the top-2
# finalists were verified on the FULL walk under 3 paired seeds.
# Both finalists REVERSED — the L7/L8 narrow-surface pattern again:
#   c1 (d2/mcw5/g2/sub.85/col.75/lr.12/la2/lalpha1)
#     screen d_ll -0.0047 -> full-walk member +0.0022 mean (3/3 worse),
#   c2 (d2/mcw50/g4/sub.65/col.5/lr.12/la1/lalpha1)
#     screen -0.0041 -> full-walk -0.00023 (inside the 0.0012 floor,
#     not 3/3). Member gate failed both; blend (c2 -0.00041) and the
#     seal (c1 -0.0050 / c2 -0.0051 mean member d_ll over the untouched
#     last 4 folds) leaned challenger, but the doctrine grades the
#     member gate. XGBOOST_PARAMS CONFIRMED — second consecutive
#     challenge defeated (L8, L9). Signal logged for a future review:
#     challengers improved member AND blend AUC on every seed (+0.002
#     to +0.008) while losing pooled logloss — this member ranks better
#     than it calibrates; revisit only under an AUC-aware, member-level
#     gate with its own floor, never by weakening the L-series gate.
# n_estimators ceiling + early-stopping rounds for walk-forward folds.
# Separate from the constructor dict because xgboost 3.2 sklearn API
# requires eval_set when early_stopping_rounds is set, and the full-refit
# path has no validation window.
XGBOOST_FOLD_ROUNDS = 2000
XGBOOST_EARLY_STOP = 20
# CAUSAL FOLD ROUNDS (2026-09-30 PIT review): fold k's SHIPPED XGBoost
# model is fit WITHOUT any eval_set at the causal transfer (median) of
# PRIOR folds' measured best iterations — the fold's own val window never
# selects the shipped model's round count (the early-stopped fit is only
# a measurement that later folds may consume). Fold 0 (no prior
# evidence) and any refit without fold measurements use the static
# priors here. The 50-round prior was kept over a 20-round re-pin: the
# 2026-09-30 sealed-window check split the verdict (d1@50 ll 0.68013 vs
# d1@20 0.68107 on folds 70-77; AUC the other way), matching the LGBM
# rounds precedent — a tune-gain must survive the seal, not just the
# OOF walk.
# WALK-SHIP FLOOR ADDENDUM (2026-10-05 run-log remediation): these
# statics are NO LONGER cold-start-only — they floor the causal transfer
# on EVERY shipped path (training._shipped_xgb_rounds). The log deep
# dive found the unfloored transfer shipping early production folds at
# 4/6/9/11/13 rounds (fold 0's noisy probe best became fold 1's entire
# prior) and the cumulative median lagging the growing train window all
# walk (~27-36); the underfit surfaced as the XGB member's collapsed OOF
# (0.5569 AUC / 0.6853 log-loss, worst of the roster) and its 0.0%
# earned blend weight. Measured on the production frame (83 folds,
# min_train_days=30, the exact remote-run protocol):
#   member OOF AUC 0.5569 -> 0.5638, log-loss 0.6853 -> 0.6844;
#   blend log-loss 0.6825 -> 0.6824, Brier 0.2448 -> 0.2447, AUC HELD
#   0.5707; verify window (folds 70-82, never used for selection)
#   better on all three: AUC 0.5979 -> 0.5990, log-loss 0.6760 ->
#   0.6759, ECE 0.0228 -> 0.0187; XGB re-earned 13.65% blend weight
#   with NO weight-policy change (was 0.0%).
# The honesty contract is unchanged: a fold still NEVER selects its own
# rounds — the floor is a static config prior and the transfer can only
# raise a shipped count above it (all 83 measured transfers were <= 36,
# so every shipped model trained at exactly the static 50). The earlier
# comment's "cold-start paths only" framing is superseded; the probe
# early-stopped fit stays measurement-only.
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
# L8 re-tune (2026-10-03): FULL RESET — every model parameter free (13
# dims, incl. n_estimators/reg_alpha/reg_lambda/max_bin/extra_trees, which
# no earlier L-series search touched), machine-stable num_threads=1, TPE
# seed 42 over the 20 most-recent tune folds of the 7,387-game / 86-fold
# frame; the last 4 folds stayed sealed until verification. Doctrine held:
# the 3-seed noise floor (0.00144 pooled OOF) was measured FIRST, and the
# winner had to clear it. It did — paired vs the full incumbent params at
# the same seed, 3/3 seeds negative on BOTH surfaces: mean d_tune -0.00323
# (2.2x the floor), sealed holdout -0.00802. Re-measured on the whole
# 82-fold tune set (seed 42, n=7019 pooled OOF) the member improves on all
# three metrics: logloss 0.68500 -> 0.68344, brier 0.24599 -> 0.24523,
# AUC 0.56216 -> 0.56888. Blend impact neutral (full production walk
# logloss 0.6835 / AUC 0.567 incumbent and challenger alike) -> ADOPTED
# under the member-strength policy (RF 2026-08-31 / L5 precedent: adopt a
# member gain when the blend is unharmed). Record:
# mlb_tune_full_20261003.json.
# What changed in spirit: the old hobble (50 rounds, 6 leaves, min_gain
# 1.2224 blocking most splits) is gone. Capacity now comes from rounds
# (364) and leaves (40) at a 2.9x slower learning rate, and stability from
# sampling discipline instead of pruning — min_gain_to_split down to
# 0.418, bagging 0.39 every 4 rounds, features 0.53, extra_trees.
# L9 FROM-SCRATCH TUNE (2026-10-08, record mlb_tune_20261008.json):
# determinism blueprint first (deterministic/force_col_wise/num_threads=-1
# below; both baseline walks were bit-identical), 3-seed noise floor
# measured BEFORE the search (pooled OOF logloss range 0.0005 — half of
# the L8 frame's 0.00144), 32 seeded draws on a surface whose last 4
# folds were SEALED, top-2 verified on the full walk under 3 paired
# seeds. Both screen leaders REVERSED on the full honest walk: c1
# (250r/depth6/15 leaves/bagging .9/features .9/no extra_trees/lambda 5)
# screen d_ll -0.0006/AUC +0.0046 -> full-walk member +0.0036 mean (3/3
# worse); c2 +0.0002 mean (inside the floor). Member gate failed both;
# LIGHTGBM_PARAMS CONFIRMED — the block survives its fourth challenge
# (L6, L7 confirmed; L8 adopted; L9 confirmed).
LIGHTGBM_PARAMS = {
    "n_estimators": 364,
    "max_depth": 7,
    "num_leaves": 40,
    "min_child_samples": 64,
    "min_gain_to_split": 0.4182845581383924,
    "bagging_fraction": 0.38979006549031003,
    "bagging_freq": 4,
    "feature_fraction": 0.5273799895070881,
    "learning_rate": 0.01131605401240319,
    "reg_alpha": 0.3851388837658578,
    "reg_lambda": 0.0026749993754089075,
    "max_bin": 173,
    "extra_trees": True,
    "random_state": RANDOM_SEED,
    "verbose": -1,
    # ── DETERMINISM BLUEPRINT (2026-10-08 full-tune review) ────────────────
    # deterministic=True pins histogram construction; force_col_wise=True
    # is the explicit builder choice for the many-features arm (109
    # serving features) so the row/col builder selection can never shift
    # between machines or library builds; num_threads=-1 keeps FULL
    # multi-threading (never a 1-thread hobble — the L8 tune's
    # num_threads=1 was a tuner-stability device, not a production
    # setting, and production never shipped it); random_state fixed above.
    # Column-order contract (blueprint item 4) is enforced upstream:
    # training.set_feature_subset() reorders any adopted RFE subset to
    # canonical pool order and training._feature_matrix reindexes to it,
    # so LightGBM sees a strictly identical column order every run.
    # Verified on this machine: two complete fits at num_threads=-1 are
    # bit-identical on predict_proba.
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": -1,
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
    "sp_xfip_delta_home",
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
    "bullpen_kbb_delta_home",
    "bullpen_xfip_delta_home",
    "lineup_re24_mean_delta_home",
    "lineup_re24_top3_delta_home",
    "sp_xfip_delta_away",
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
    "bullpen_kbb_delta_away",
    "bullpen_xfip_delta_away",
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
    # --- universe departures (pl_[pos] + removals plan, 2026-10-03) (14) ---
    # The re24 / depth / sp_xfip (era->xFIP 2026-10-09) / park-factor-slug
    # families left
    # MONEYLINE_FEATURE_COLS for the position-pool xwOBA swap (plan: replace
    # 9, remove 7). They remain generated (features.py) and enter the
    # addition pool so RFE may re-trial them with evidence — the 2026-09-07
    # S-family cull pattern. lineup_re24_std_home/away already sit in the
    # per-side group below.
    "lineup_re24_mean_diff",
    "lineup_re24_top3_diff",
    "lineup_re24_std_diff",
    "lineup_re24_mean_home",
    "lineup_re24_mean_away",
    "lineup_re24_top3_home",
    "lineup_re24_top3_away",
    "lineup_depth_multiplier_diff",
    "lineup_depth_multiplier_home",
    "lineup_depth_multiplier_away",
    "park_factor_slug_diff",
    "sp_xfip_diff",
    "sp_xfip_home",
    "sp_xfip_away",
    # --- raw per-side levels (shadowed by diff-only routing) (76) ---
    "rest_days_home",
    "rest_days_away",
    "sp_bb9_home",
    "sp_whip_home",
    "sp_fip_home",
    "sp_bb9_away",
    "sp_whip_away",
    "sp_fip_away",
    "sp_xfip_5g_home",
    "sp_k9_5g_home",
    "sp_xfip_5g_away",
    "sp_k9_5g_away",
    "team_iso_30g_home",
    "team_k_rate_30g_home",
    "team_bb_rate_30g_home",
    "team_iso_30g_away",
    "team_k_rate_30g_away",
    "team_bb_rate_30g_away",
    "bullpen_xfip_10g_home",
    "bullpen_xfip_10g_away",
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
