# NFL causal evaluation and fold/refit parity (2026-10-08)

## Scope

Remediates the same structural pitfalls the NHL causal-evaluation commit
(`42027979`) and the MLB Oct-7 remediation (`f414970b`) fixed, found present in
the NFL moneyline during a parity audit authorized by the user. No production
pipeline run, notebook edit, hyperparameter change, or weight-optimizer change
is part of this change. Existing model-facing feature names,
prediction-history CSV columns, and calibration/monitor JSON metric keys are
unchanged; additional metadata keys are backward-compatible. The current
version is **`nfl-prod-v9.8-causal-evaluation`** (was v9.7): the tree
representation changed, so old bundles must not score — rerun features, OOF,
blend/calibration, and the final refit together.

## Pitfall register (audit of NHL/MLB commits vs NFL code)

| Pitfall | NHL/MLB fix | NFL before (defect?) | NFL now |
|---|---|---|---|
| Final weights replayed over OOF published as `p_ensemble` (retrospective flatters headline) | `42027979`, `f414970b` | **PRESENT** — 2026-10-05 "artifact = binary's blend" re-pool, log said "headline metrics grade THE serving blend" | `p_ensemble` stays the rolling causal blend; replay only in `p_ensemble_retrospective`; log labels both |
| Final serving calibrator bypasses the fold-origin gate (ungated all-OOF fit + post-hoc headline gate) | `gated_calibrator` at every origin | **PRESENT** — ungated `moneyline_fit` on all grading rows + dynamic `should_gate_calibrator` comparing headline columns | one nested prior-evidence gate at every fold origin and the final origin; `should_gate_calibrator` retained for history/tests, uncalled (MLB left its def uncalled too) |
| Prequential fold maps fit on non-grading rows (postseason/provisional move maps) | `grades=` mask | **PRESENT** — `prior_mask` was `(fold_ids < fid) & okp` | evidence mask adds `& _grading`; input is the causal blend |
| XGBoost fits team IDs as ordinal numbers (no `enable_categorical`, no category dtype); LightGBM gets no categorical declaration | true pandas categoricals, pinned vocabulary, booster feature-type check | **PRESENT** — int64 team IDs, no `enable_categorical`, no `categorical_feature` | `member_fit_input` converts to `pd.Categorical` with the config vocabulary (UNK slot included); `enable_categorical=True`; LGBM fit declares categoricals by name; fitted booster types CHECKED (`!= "c"` raises); SHAP keeps the vocabulary |
| Fold and final fits drift (different kwargs/representation/order) | shared `_fit_member`, canonical refit order | partial — both used plain `_make_member`, but two divergent inline fit paths, no canonical sort | one shared `_fit_member`; `fit_final_models` canonical-sorts |
| Calibration buckets / `n_games` describe a different population than headline metrics | grading population | **PRESENT** — full frame + `okp.sum()` | grading population in both |
| XGB rounds ledger (mutable cross-run state, val-window early stop) | explicit ledger | **ABSENT by construction** — NFL XGBoost has no early-stop surface at all; fixed `n_estimators` at fold and final (already pinned by tests) | unchanged; recorded in metadata as `xgb_round_policy: fixed_config_budgets_no_early_stop` |

## Evaluation contract (mirrors NHL's table)

| View | Meaning | Allowed use |
|---|---|---|
| `p_ensemble` | Each fold uses only earlier-fold blend evidence | Headlines, selection, calibration, backtest history |
| `p_ensemble_causal` | Identical compatibility alias | Explicit causal diagnostics |
| `p_ensemble_retrospective` | Final learned weights replayed on their fitting OOF member outcomes (== returned `blend_full`) | Diagnostic only, **not OOF** |
| Future slate | Full-history member refits through the shared `_fit_member`, all-prior learned weights, next-origin gated calibration | New forecasts |

The run log's `Published blend:` line states that headlines grade the causal
rolling blend and the final weights replay only in `p_ensemble_retrospective`
(not OOF). Phase 9 logs `moneyline retrospective final-weight replay (NOT
OOF / not selection evidence)` on the same grading population, and the
summary/calibration JSON carry `moneyline_retrospective_not_oof` and
`config_meta.moneyline_evaluation`.

## Calibration contract

`moneyline.gated_calibrator` is the single policy at every origin (NHL
`42027979` port):

1. finite causal-blend probabilities from eligible `grades_pooled` rows;
2. strictly-prior chronological evidence split into the early fit slice and
   the recent holdout slice (`CAL_GATE_HOLDOUT_FRAC = 0.25`,
   `CAL_GATE_MIN_HOLDOUT = 200`, `MIN_OOF_FOR_FIT = 300`);
3. accept only if holdout log-loss beats raw by `CAL_GATE_EPS = 5e-3` nats;
4. refit on all prior evidence; otherwise identity.

Aligned vectors/masks are validated (misaligned raises). The gate audit
(`config_meta.calibration_gate`) records per-fold decisions and the
final-serving verdict; `metrics.calibrator_gated_out` still records whether
identity ships. Serving buckets and `n_games` describe the grading
population.

## Verification

- NFL backend suites: `test_fold_season_split.py` (12 passed) and
  `test_production.py` (script-style; RESULTS line plus new pins for: causal
  headline, retrospective column, nested gate at every origin, grades-masked
  prequential evidence, grading-population buckets, config vocabulary
  categoricals, shared `_fit_member`, canonical refit, `gated_calibrator`
  unit contracts), `test_run_log_tee.py` (published-blend label pins
  updated to the causal contract — the old assertions encoded the faulty
  policy).
- Interface smoke (real xgboost/lightgbm/sklearn, synthetic frames):
  `_fit_member` fits all three families; the booster categorical-type check
  passes; an unseen team label maps to the UNK vocabulary slot and predicts;
  `gated_calibrator` identity/no-gain/misalignment behaviors verified.
- Numbers on this change are structural/honesty fixes; no AUC/logloss gain
  is claimed from code alone. The retrospective-vs-causal gap (if any) is
  now visible in the log instead of hidden inside the headline.

## Remaining limitations

- The nested gate will hold most early-fold maps and possibly the final map
  at identity until enough causal evidence accumulates (`n_fit ≥ 300` plus a
  5e-3 holdout bar). That is the intended conservative policy, not a defect.
- Full production impact (postseason-in-frame games, epa season-boundary
  carry from the prior remediation, and this evaluation change) is measured
  on the next full pipeline run; the Kaggle notebook is untouched.
- LightGBM consumes the team-ID pair as categoricals by name at fit; its
  sklearn predict path replays the same integer codes (config IDs equal
  category codes by construction — the reason the map is versioned config).
