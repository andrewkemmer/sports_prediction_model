# Four-Sport Moneyline & Distribution Model Audit — 2026-10-14

**Scope.** MLB, NFL, NHL, NBA backends: the binary moneyline model (objective:
pooled OOF log-loss minimization; success criteria: higher AUC, lower log-loss,
lower Brier) and the score-distribution / run-engine model (objective: CRPS
minimization, with distribution-derived market probabilities AUC-tracked).
Audited invariants: strictly point-in-time (PIT) feature ingestion and
engineering, professional-grade model mechanics, and effective optimization of
both stated objectives.

**Method.** Full read of the moneyline, folds, features, calibration, evaluation
and distribution modules in all four backends; execution of each backend's PIT /
causality / mechanics regression suites; inspection of the latest delivered run
artifacts (`data_delivery/*calibration*.json`, `*run_engine_monitor_*.json`);
cross-reference of the prior `LEAKAGE_REPORT.json` audit (baseline commit
5b703ca). A separate thread is actively tuning the distribution models
(CRS-location objective port in `mlb/nfl/nhl` `distributions.py`); those
uncommitted files were treated as out of scope and were not modified.

**Verdict.** Both model families are structurally sound and PIT-clean in all
four sports. Two objective-policy gaps were found and repaired (NFL, MLB — see
§5). No leakage defect was found in the current code.

---

## 1. Binary moneyline — objective & optimization

**Objective contract (all 4 sports: PASS).** Blend weights are earned by
simplex-constrained SLSQP minimizing **pooled** OOF binary log-loss, applied in
logit space (clip → logit → weighted mean → sigmoid), matching the serving
blend exactly. No temperature, floor, or cap; a member takes 100% weight only
when its own pooled log-loss beats the optimized blend's.

| Sport | Optimizer | Pooling | Objective guard |
|---|---|---|---|
| MLB | SLSQP simplex, logit space (`training.compute_adaptive_weights`) | pooled OOF | **warns** off-policy (new, §5.2); shipped config pinned by test |
| NFL | SLSQP simplex, logit space (`moneyline.compute_adaptive_weights`) | pooled OOF | **raises** (new, §5.1) |
| NHL | SLSQP simplex, logit space | pooled OOF | raises (pre-existing) |
| NBA | SLSQP simplex, logit space, member caps/floors policy box | pooled OOF, per-row NaN renormalization | raises (pre-existing) |

**Causal walk-forward weighting (PASS).** Every sport re-earns weights only
after a fold is scored, from strictly-prior grading evidence; fold 0 rides the
static 1/3 priors; the fold's own rows never weight the blend that scored them
(weights are snapshotted before the update). The final-weight replay ships as
`p_ensemble_retrospective` and is explicitly labeled NOT OOF.

**XGBoost round policies — no validation-window self-steering (PASS).**

| Sport | Policy | Leak status |
|---|---|---|
| MLB | Shipped fixed-budget fit first; early-stopped probe afterwards is measurement-only; causal median of **prior** folds' best rounds, floored at the static budget | clean |
| NHL | Same probe-after-shipped pattern (`causal_xgb_rounds`), fixed 2026-10-12 | clean |
| NBA | Early-stopping watch set is a chronological **tail of the training fold** (`_early_stop_watch`), then refit on all train rows at the measured budget | clean |
| NFL | Fixed budget from `XGBOOST_PARAMS`, no eval_set at all | clean |

**Grading population (PASS).** NBA/NFL/NHL: only regular-season rows from
windows at or above `MIN_VAL_FOLD_GAMES` feed the blend optimizer and the
calibration fits; postseason and provisional windows are still fit and scored
(and shipped to `*_predictions_history`) but never grade the model. MLB applies
the equivalent window gate.

**Metrics (PASS).** AUC, log-loss, Brier and ECE are computed on OOF in all
four backends (`compute_metrics` / `binary_metrics`), reported raw and
calibrated, plus rolling Brier timelines and winner-card market metrics.

Latest delivered moneyline metrics:

| Sport | Date | AUC | Log-loss | Brier | ECE |
|---|---|---|---|---|---|
| MLB | 2026-10-14 | 0.5750 | 0.6815 | 0.2443 | 0.0072 |
| NFL | 2026-10-13 | 0.6997 | 0.6287 | 0.2194 | 0.0292 |
| NHL | 2026-10-13 | 0.5974 | 0.6742 | 0.2407 | 0.0105 |
| NBA | 2026-10-14 | 0.7333 | 0.6025 | 0.2077 | 0.0227 |

Calibrated == raw in the current runs: in every sport the final serving
calibrator is gated and correctly declined to identity (no out-of-sample gain
over the raw blend on the nested holdout — the documented behavior when the
blend is already near-calibrated).

## 2. Binary moneyline — calibration mechanics

All four sports implement the same two-layer contract (PASS):

1. **Prequential per-fold evaluation layer** — fold *k*'s calibrated OOF
   probabilities come from a favored-space Platt map fitted strictly on folds
   `< k` (NBA: inline loop in `walk_forward_oof`; NHL: helper
   `prequential_fold_calibrators`; NFL: per-fold `gated_calibrator`; MLB:
   prequential OOF twins in `calibration.py`). Each candidate is admitted only
   when a nested chronological holdout on the prior evidence beats the raw
   blend by `CAL_GATE_EPS` nats; otherwise the fold serves uncalibrated.
2. **Final gated serving calibrator** — same eligibility + nested gate,
   evaluated at the next origin on all prior causal OOF pairs; used for the
   slate only, never re-applied to the OOF it was fit on for headline claims.

`fit_platt` degenerate/fallback counts are uncapped and summarized in the run
log in every sport (reviewer-visible).

## 3. Distribution model — CRPS minimization & AUC

**Mechanics (PASS, all 4).** Two-sided asymmetric Gaussian split-skew family
(shared, identical `gaussian_family.py` ×4) with lattice discrete CRPS:

- `walk_forward_oof` fits the Poisson score regressor per fold on strictly
  prior rows and hard-raises if `train.max(date) >= val.min(date)`.
- `causal_distribution_frame` fits variance/skew/calibration maps **at each
  forecast origin on prior OOF outcomes only** and refuses undated frames
  rather than claiming PIT. "Final serving maps may use all settled past OOF;
  NEVER apply those maps back to OOF rows."
- `_distribution_blocks` asserts date-ordered, non-overlapping forecast blocks.
- MC pricing (`sample_cdf`) consults no outcomes; `distribution_score_report`
  rebuilds each origin's SAME CDF for full-support CRPS and the binary market
  scores (log-loss/ECE/Brier/AUC per line).
- Serving admission: `negative_mean_paired_crps_on_prior_issued_origins`
  (`objective: discrete_crps` in every sport's delivered
  `distribution_calibration` block).

**AUC tracking (PASS).** Distribution-derived probabilities
(`p_over_*`, `p_cover_*`, `p_home_win_derived`) are AUC/log-loss/Brier-scored
per market card in every sport's `run_engine_monitor` (winner_cards) and
per-line diagnostics. Latest derived-moneyline AUCs: MLB 0.569, NFL 0.700, NHL
0.589, NBA 0.644. Totals cards run near coin-flip (MLB 0.512, NFL 0.512, NHL
0.533, NBA 0.485) — a known property of totals pricing, monitored rather than
gated.

**In-flight work (noted, out of scope).** Another thread is porting the
opt-in analytic Gaussian-CRPS location objective (`GaussianCRPSObjective`) from
NBA to MLB/NFL/NHL, with objective-parity tests per backend. Production stays
on the Poisson/discrete-CRPS path until a challenger wins a causal walk.

## 4. Point-in-time ingestion & feature engineering

**PASS with evidence, all 4 sports.**

- **NBA/NFL/NHL features**: every trailing primitive is per-team sorted then
  `shift(1)` (rolling/EWM/ewm-halflife), including the opponent-adjusted and
  shift-of-shift ladders — all ride the same causal primitives.
- **MLB features**: pure DuckDB SQL; the documented contract is LAG() first,
  then `ROWS BETWEEN N PRECEDING AND CURRENT` over the shifted values.
- **MLB-01 repair verified in code** (prior LEAKAGE_REPORT finding, high
  severity): `compute_elo_entries` and `compute_season_records` snapshot
  entering-day state per calendar date, so doubleheader legs can never read
  each other's results without verified final timestamps ("Export tie order is
  not event availability").
- **Folds**: chronological walk-forward only; thin windows are retained as
  `provisional` (fit + scored, never grade); no window without training rows.
  Covered by `test_fold_season_split.py` in all four backends.
- **Availability/injury PIT**: NBA resolves slate designations through the
  same game-day-submission resolver history was built with (strictly before
  tipoff, never the box score); NHL has dedicated `test_pit_availability.py`;
  MLB gates IL stints and SP staleness in the feature engine.
- **Serving parity**: slate features are built through the identical feature
  contract with the same resolvers; coverage gaps fail the run loudly
  (`_empty_contract_columns`).

## 5. Remediations applied by this audit

### 5.1 NFL — blend objective guard (behavior-preserving)

`nfl-backend/backend/moneyline.py::compute_adaptive_weights` now raises
`ValueError` when `ADAPTIVE_WEIGHT_METRIC != "logloss"`, mirroring the NBA and
NHL guards. NFL had no guard, so a flipped config value would have silently
re-aimed the production blend optimizer off the stated objective.
`nfl-backend/backend/config.py` now **declares** `ADAPTIVE_WEIGHT_METRIC =
"logloss"` explicitly (it was previously implied by absence — NBA and NHL both
declare it). Regression tests added in
`nfl-backend/backend/test_fold_season_split.py::TestBlendObjectivePolicy`
(pin the shipped value, the refusal, and the simplex pooled-log-loss
behavior).

### 5.2 MLB — off-policy warning + objective pin

MLB retains the historical `ADAPTIVE_WEIGHT_METRIC == "auc"` experiment
branch (documented in the function). The shipped config is `"logloss"`, and
`training.compute_adaptive_weights` now logs a **loud warning** whenever the
metric is anything else, so a config flip cannot silently re-aim the blend
optimizer. The shipped value and the simplex pooled-log-loss behavior are
pinned by the new
`mlb-backend/backend/test_adaptive_weight_policy.py` (production-graph
guard verified: file is test-compliant).

### 5.3 Verification

All suites re-run after the change (see §6). No behavioral change in any
shipped configuration.

## 6. Test evidence (executed 2026-10-14)

| Sport | Suites | Result |
|---|---|---|
| MLB | `test_run_engine_pit`, `test_fold_season_split`, `test_run_engine_contract`, `test_calibration_gate`, `test_gaussian_family`, `test_grid_rfe` | 110 passed |
| MLB (post-fix) | + `test_adaptive_weight_policy`; `check_production_graph.py` | 64 passed; graph OK |
| NFL | `test_fold_season_split`, `test_feature_name_consistency`, `test_gaussian_family` | 22 passed |
| NFL (post-fix) | `test_fold_season_split` (16, incl. 3 new), `test_production` | 16 passed; production suite re-run green |
| NHL | `test_run_engine_pit`, `test_fold_season_split`, `test_pit_availability`, `test_causal_evaluation`, `test_gaussian_family`, `test_grid_rfe` | 260 passed |
| NBA | `test_fold_season_split`, `test_nba_rapm_features`, `test_gaussian_family` | 125 passed |

## 7. Observations & recommendations (non-blocking)

1. **Retire or fully guard the MLB `auc` branch.** NBA/NHL hard-refuse any
   non-logloss metric; MLB now warns, but the cleanest end-state is the same
   hard refusal with the experiment preserved in git history.
2. **Distribution totals AUCs sit near 0.5** in MLB/NBA. Proper-score
   objectives do not optimize AUC directly; if totals AUC becomes a KPI,
   consider a rank-aware challenger evaluated under the same causal walk
   (the in-flight CRPS-location work is the right pattern).
3. **Calibration JSON consistency.** NHL ships a `prequential_gate` audit
   block in its calibration JSON; NBA/NFL/MLB keep the gate audit in
   fold/config metadata. Consider standardizing the key across sports.
4. **Do not ship the in-flight distribution port without its parity tests.**
   The uncommitted `test_distribution_objective_parity.py` files (mlb/nfl/nhl)
   should land with the code they pin.
