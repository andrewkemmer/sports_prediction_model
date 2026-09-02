# MLB blend-state audit — no NFL-style cross-walk contamination (no-change)

**Date:** 2026-09-02 · **HEAD audited:** 05a0440 (origin/main) · **Finding:** MLB is CLEAN — no fix required.

## Background

The NFL codebase carried an A/B-state bug (fixed in `5fd0549`): the fold-loop blend
reads the module-global adaptive weights, which are only WRITTEN at the **end** of a
walk — so in any harness walking arms sequentially in ONE process, every walk after
the first blended with the PREVIOUS arm's adaptive weights instead of the static
`ENSEMBLE_WEIGHTS` priors (proof: two identical consecutive walks measured pooled
ll 0.6312 then 0.6201 pre-fix; post-fix consecutive walks are byte-identical).
Impact was pooled OOF ll/auc surfaces only (~0.011 pooled ll — ~TOL-sized), so
multi-arm pooled verdicts were untrustworthy until re-derived.

## Pattern comparison (MLB vs NFL)

| Item | MLB (`mlb-backend/backend/training.py`) | NFL (pre-fix, `nfl_moneyline.py`) |
|---|---|---|
| Blend-weight store | module global `_LAST_ADAPTIVE_WEIGHTS` | module global `_ADAPTIVE_WEIGHTS` |
| Fold-loop blend read | `ensemble_predict` → `_member_weights` → `source = _LAST_ADAPTIVE_WEIGHTS or ENSEMBLE_WEIGHTS` | same shape (static-prior fallback when empty) |
| Write site | END of `walk_forward_evaluate`: `compute_adaptive_weights(oof_members, y_oof)` → clear + update | END of `run_walk_forward` (line ~1323) |
| **Entry reset** | **YES — `walk_forward_evaluate` clears the global as its first action (since `17b382b`, predates the NFL discovery)** | NO (this was the bug) |
| Harness defense | belt-and-suspenders per-arm clears (`run_*.py`: `training._LAST_ADAPTIVE_WEIGHTS.clear()`); several harnesses also isolate arms via `subprocess` | per-arm clears added only after the fix |

Production single-walk behavior is identical in both (walk 1 always starts from the
static priors). MLB's reset is exactly the entry reset the NFL fix added — MLB's
`walk_forward_evaluate` is the reference the NFL change brought `nfl_moneyline.py`
into line with.

## Empirical proof (reproduced the NFL detection on MLB)

Synthetic 150-day frame (1,144 OOF rows / 21 folds), `walk_forward_evaluate`
called repeatedly in ONE process:

- **Walk 1 vs walk 2 (identical config):** pooled `auc`/`logloss`/`brier`/`ece` +
  calibrated twins **byte-identical** (logloss 0.6901 both; every pooled key equal).
  Per-game `home_win_prob_model` max |Δ| = 1.1e-16 (1 ulp — float noise; a blend
  contamination would move it ~1e-3).
- **Poison test:** setting `_LAST_ADAPTIVE_WEIGHTS = {xgboost: 1.0, others: 0.0}`
  between walks (the exact stale state the NFL bug left behind) changed NOTHING —
  the third walk's pooled surfaces are byte-identical to walk 1. The entry reset
  wiped the poison before the first fold blended.
- **Early-failure reset:** the clear is the walk's first statement, so even a walk
  that dies in `walk_forward_splits` leaves the global empty (regression-pinned).

## Harness architecture

Multi-arm MLB ablations (run_form_delta_ablation, run_lineup_ablation,
run_margin_ablation, run_mlb_margin_k_ablation, run_opponent_adjusted_ablation,
run_feature_pruning_ablation, run_lgb/rf_tuned_blend_ablation, …) run arms through
`walk_forward_evaluate` in one process with per-arm clears — the walk's own entry
reset makes those clears redundant-but-harmless. Several harnesses additionally
isolate arms via `subprocess` (fresh module state). **No MLB multi-arm record needs
re-derivation for this cause.**

## Regression pins

`mlb-backend/backend/test_walk_blend_state.py`:
1. entry reset happens even on an early `walk_forward_splits` failure;
2. consecutive identical walks → byte-identical pooled surfaces, and a poisoned
   global between walks does not change the next walk.

**No code change made** — single-walk/production behavior untouched; the NFL fix
(`5fd0549`) and MLB's long-standing entry reset are now documented as aligned.
