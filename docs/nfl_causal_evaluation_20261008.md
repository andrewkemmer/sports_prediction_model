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

## 2026-10-08 run review (pre-remediation baseline) and accuracy audit

The last remote run before these remediations (log delivered in `f2630028`,
22:50–23:09 UTC) is CLEAN mechanically: all seven Phase-14 gates PASS, 239
artifacts, no failed members, no crashes. It is the honest BASELINE the next
(full-remediation) run must be compared against:

| Metric | Baseline (old code) | Next-run expectation |
|---|---|---|
| `oof_regular` (grading, causal) | AUC 0.6977 / logloss 0.6281 (n=1689) | compare THIS to the new causal headline — not the old "raw" line |
| old "moneyline OOF raw" line | AUC 0.6998 / logloss 0.6270 | was the RETROSPECTIVE replay; next run reports causal + retrospective separately |
| `oof_provisional` | AUC 0.6850 / logloss 0.6395 (n=758) | grows: playoff windows now enter as provisional |
| members | xgb 0.6954/0.6290, lgbm 0.6981/0.6285, enet 0.6965/0.6306 | re-earned on the enlarged frame |
| schedule rows | 2704 (REG only) | 2826 (+122 WC/DIV/CON/SB) |
| `postseason_windows` | 0 | real; Jan/Feb windows are thin → provisional |
| epa_* coverage | 87.94–92.08% | higher: season-boundary carry covers 2017+ openers (2016 openers stay NaN — no prior pbp exists) |
| calibrator lines | "final pooled calibrator (fitted; Phase 9 gates it)" + dynamic gate | "final serving calibrator … (nested prior-evidence gate: accepted/identity + reason)" |

### Accuracy audit vs official game data

- Delivered cards store (2,447 rows): `actual_winner` matches the official
  scores 2447/2447; the `correct` invariant holds 1152/1152; probabilities
  sum to 1 on every row (audited earlier this session).
- Delivered `nfl_power_rankings_20261008.csv`: cumulative W-L for all 34
  rows recomputed from the official nflverse schedule (REG, through
  2026-10-05) — **34/34 exact, 0 mismatches**.
- Frame census reconciles exactly with the official schedule (2,703 decided
  = 2,639 REG 2016–2025 + 64 in 2026 through week 4; warmup 256 + core
  2447).
- Coverage gaps in the log are all structural and documented: weather
  71.14% (outdoor stadiums only — dome games have no weather by design),
  rest_days/opener families 93.5–99.4% (openers are NaN, never a priced
  offseason gap), travel/turf 98.4–99.2% (historical venue-resolution
  gaps).

### Drift verdicts (investigated, no action — honest signals)

- `inj_ol_out_home/away` ALERT (psi_adj 0.339/0.294): a REAL location
  shift in the delivered weekly-report + roster-overlay data — current
  window mean 1.37 vs season-phase-matched baseline 2.04 OL designations
  per team-game (the 2026 feed carries thinner OL-unavailability capture
  than prior Septembers; `date_modified` confirms the strict-PIT/weekly
  channel seam at 2025). The model gives the family ~0% blend weight
  (weight_pct 0.0/0.08), so there is no accuracy lever here; suppressing
  the alert would hide a true distribution change, so it stays.
- `pace_plays_min_away` WARN (mean 1.298 vs baseline 1.334, −2.8%): small
  but real location shift at 0.54% blend weight — monitored, no action.

## Final parity pass (2026-10-09): coverage gates, drift visibility, missing finals

A second review round against the newest MLB/NHL commits found four more
structural gaps, all now closed in existing files (no new programs):

- **Ingestion/column coverage gates** (NHL `c9bcc3b8` / MLB column
  contract): `_feature_coverage_gaps` fails the run BEFORE training when a
  served column is absent or wholly unobserved, and
  `_season_openers_ingested` rides the Phase-14 gate dict — every covered
  season must contain its Sep 4–20 opener band, fail-closed on missing
  frame/window, skipping operator mid-season starts. Pinned through the
  helper AND through `_validate_outputs` itself.
- **Drift visibility + season-seam guard** (MLB 2026-09-30 / NHL
  `b698f90b`): `feature_drift` now emits view-labeled summary lines
  ("Feature drift [moneyline]/[run-engine]: …") so a run-log reader can
  see drift and tell the two surfaces apart, and re-measures any WARN/ALERT
  location shift against the same calendar phase of prior years
  (`DRIFT_PHASE_EXTENSION_MONTHS = (-1, -2)`, ±7-day pad): a clean
  re-check relabels the row `OK-SEASONAL` (verdict only — PSI evidence
  untouched). Both call sites pass the full decided pool as `phase_frame`;
  a genuine regime shift stays ALERT (pinned both ways).
- **Missing-final warning** (MLB `2d58210e`): `ingestion.warn_missing_finals`
  returns recent past-gameday games still lacking a score (7-day bound;
  today's slate never claimed) and master logs them warn-only. The live
  2016–2026 schedule currently carries exactly one such row —
  `2026_05_TB_DAL` (2026-10-08), the defect class happening in real time —
  and zero historical past-score nulls, so the bound costs no detection.
- **Dashboard population labels** (MLB `758f745b`): the shared Calibration
  page already reads `n_eval → n_games`, and NFL's writer carries the
  grading population in `n_games` (buckets sum == n_games == 2447 on the
  delivered artifact); `league_total`/`evening_games_league` fall back
  exactly as NHL/NBA do — no label defect. `OK-SEASONAL` renders through
  the existing green `ok` pill default on both drift tables.
- **Artifact self-consistency**: cards store == predictions history ==
  calibration `n_games` == bucket sum (2447) on the delivered v9.7
  artifact; the new writer's grading-pool semantics are test-pinned for
  the next run (v9.8).
- MLB's `75c12cda` weather-fill and `01555b66` single-class/finality bugs
  have no NFL surface: NFL never fills weather (documented structural-NaN
  policy, single build path for train+serve) and every `log_loss` call
  already passes `labels=[0, 1]` behind a single-class guard.

Verification after this pass: production script **450 passed / 0 failed**,
fold+tee 34, feature-name 39, data-delivery 102, calibration/gate-note
smokes 5, monitor smoke and board-render smoke exit 0.

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
