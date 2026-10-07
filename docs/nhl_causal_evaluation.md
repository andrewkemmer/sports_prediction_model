# NHL causal evaluation and fold/refit parity

## Scope and acceptance

Follow-on to the [input repair report](nhl_input_repairs.md), authorized to fix
moneyline evaluation before the user reruns the full pipeline. No production
pipeline run, hyperparameter search, production artifact replacement, or notebook
edit is part of this change. Existing model-facing feature names, prediction-history
CSV columns, calibration/monitor JSON metric keys, bundle name, and pipeline CLI
remain unchanged. Additional diagnostic columns/metadata are backward-compatible.

The current version is **`nhl-prod-v1.3-causal-evaluation`**. Older v1.1/v1.2 bundles
must not score the corrected representation; rerun features, OOF, blend/calibration,
and final refit together. Boxscore caches remain at corrected **v5**.

## Evaluation contract

[Moneyline](../nhl-backend/backend/moneyline.py) no longer overwrites the rolling
OOF ensemble with final weights fitted on those same outcomes.

| View | Meaning | Allowed use |
|---|---|---|
| `p_ensemble` | Each fold uses only earlier-fold blend evidence | Headlines, selection, calibration, backtest history |
| `p_ensemble_causal` | Identical compatibility alias | Explicit causal diagnostics |
| `p_ensemble_retrospective` | Final learned weights replayed on their fitting OOF member outcomes | Diagnostic only, **not OOF** |
| Future slate | Full-history member refits, all-prior learned weights, next-origin gated calibration | New forecasts |

Per-row `weight_xgboost`, `weight_lightgbm`, `weight_elasticnet` and per-fold weights
are saved in local diagnostics. The fold table includes `xgb_rounds`. The walk
returns `xgb_best_rounds` and `final_xgb_rounds` explicitly. The master pipeline
labels causal headlines and retrospective metrics separately in its log/summary;
config and bundle metadata record the policy. A future fold's labels cannot
rewrite that fold's headline probabilities, an earlier map, or learned entering
weights. Final serving weights are not claimed to have been OOF on their own
fitting outcomes.

This preserves the existing eligibility policy: regular-season rows from
non-provisional folds feed blend fitting and headline metrics. Postseason and
provisional rows are still scored and reported separately. The pooled objective
is per-game, not equal-fold weighting; the old equal-vote rationale was removed.
Changing eligibility/window geometry is a separate model-policy experiment.

## Calibration contract

Both prequential maps and the final serving map use the **same**
`gated_calibrator` policy:

1. Use finite causal blend probabilities from eligible `grades_pooled` rows.
2. Split strictly-prior chronological evidence into the existing early fit and
   recent holdout slices (existing minimums, fraction, and improvement bar).
3. Accept a map only if its holdout log loss beats raw by more than the bar.
4. Refit an accepted map on all eligible prior evidence; otherwise use identity.

The final origin no longer bypasses that gate. Postseason/provisional outcomes
cannot move its fit, gate, or subsequent fold maps. Vectors and masks must align;
misaligned inputs raise instead of broadcasting. The gate audit records eligibility,
causal input view, and the final-serving verdict.

Headline calibration buckets and `n_games` now describe the same grading
population as headline metrics. Daily rows and separately reported populations
remain full causal views. AUC comparisons on identical rows remain meaningful:
fold-dependent maps and the favored-probability floor can change ranking/create
ties. A blanket claim that calibration must preserve AUC was removed.

## Fit/refit contract

Both scored folds and final moneyline refits use one shared fit function:

- Train-only preprocessing for elastic net; canonical final row order matches
  fold ordering.
- LightGBM receives the same categorical declarations and no scored-window
  evaluation set in either path. Its fixed tree count is unchanged.
- XGBoost receives true pandas categorical team columns with a fixed configured
  vocabulary, including the unknown slot; vocabulary does not depend on which
  clubs happened to occur in a fold. Actual booster feature types are checked.
- Final XGBoost rounds are the deterministic median of **this walk's** measured
  prior probes (integer average for an even count), with the same no-evidence
  prior as fold zero. No hidden cross-run mutable ledger or implicit 100-tree
  default is consumed. All completed OOF windows are prior to future serving.
- Fit/predict/SHAP preserve the same XGBoost category vocabulary. Joblib bundle
  replay with validation-only/unknown teams is regression-tested.

The probes remain measurement-only: a fold's own early-stop result may inform
later folds or the future final fit, never its scored model.

## Verification

- [Regression tests](../nhl-backend/backend/test_causal_evaluation.py): future-label
  poisoning, explicit entering weights/rounds, excluded-label invariance,
  calibration next-origin equivalence and no-gain identity, alignment validation,
  stable category vocabulary, shared fit arguments, canonical refit order,
  independent-ledger fallback, actual categorical boosters, joblib/slate replay,
  and master wiring.
- Updated [existing PIT tests](../nhl-backend/backend/test_run_engine_pit.py) to
  require causal headlines plus separately labeled retrospective replay, and
  shared LightGBM no-validation-dependency fitting. No test was skipped or
  weakened to hide a failure; previous assertions encoded the old faulty policy.
- [Targeted checks](nhl_causal_evaluation/targeted_tests.txt): **19 passed**.
- [Bounded interface verification](nhl_causal_evaluation/interface_verification.json):
  real XGB/LGBM/elastic-net walk, final ledger refit, unknown-team slate predictions,
  persisted causal history CSV and calibration JSON, two SHAP artifacts, and
  the exact master's learned-ledger write verified through its saved fold CSV.
  Synthetic inputs and small test budgets demonstrate interface correctness,
  **not NHL AUC/log-loss gains or production timing**.
- [Full-suite log](nhl_causal_evaluation/test_results.txt): **587 passed, 1 failed**,
  32 dependency warnings, in 251.49 seconds; preserved pytest exit code **1**.
  The only failure is the pre-existing Kaggle notebook `tqdm` dependency
  assertion. It is not bypassed; the notebook remains Kaggle-owned.
  Compilation and whitespace checks passed. No configured/installed mypy or
  pyright checker is available.

## User pipeline rerun checklist

1. Ensure the existing Kaggle clone/update step pulls the newly pushed `main`.
   No notebook modification is required for these code repairs.
2. Run the **full** master pipeline, not only slate serving or loading the prior
   joblib. Keep any established local/Kaggle delivery policy explicit. Local
   no-push usage is documented in the [README](../nhl-backend/README.md).
3. Confirm the run/bundle config says `nhl-prod-v1.3-causal-evaluation` and
   ingestion uses boxscore v5. Old v4 cache files may remain but are never read.
4. Check the log says headlines use the **causal rolling blend**, retrospective
   replay is **NOT OOF**, and the final calibration audit gives its gate reason.
5. Confirm logged final XGB rounds match the ledger policy and fitted booster
   team fields are categorical. Review PP/faceoff and goalie warm-null coverage.
6. Retain `nhl-backend/run_diagnostics/nhl_oof_moneyline.csv` and
   `nhl_fold_table.csv` locally alongside the run log and summary. They contain
   the member predictions, row weights, budget ledger, and separated blend views
   needed for honest follow-up comparisons; they are deliberately not delivery.
7. Compare causal regular/postseason/provisional metrics separately. The old
   retrospective headline is not an equivalent baseline. Do not claim improved
   AUC/loss from these code fixes alone.

## Remaining limitations

This does not repair the audit's separate run-engine reversed metric arguments,
cross-line probability coherence, historical availability gaps, or non-Elo pending
team-stat contamination. Past tuning choices were already inspected and are not
pristine holdouts. Full NHL accuracy and resource measurements await the user's
pipeline rerun; library-version differences can still affect XGB outcomes.
