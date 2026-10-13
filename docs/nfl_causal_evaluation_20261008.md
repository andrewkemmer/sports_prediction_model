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

## Fourth pass — run-log review of the 2026-10-13 10:23 delivery (Open-Meteo rate limits)

Scope: review the delivered `nfl_pipeline_run_log.txt` (remote `d4665299`)
and remediate what it still shows. Guardrail held: existing files only,
no new programs.

**What the log showed (reconstructed timeline).** The archive sweep ran
448 batches clean for ~18 minutes, then the shared Kaggle egress IP's
quota closed (10:41:35): one batch burned its two quick retries, 47 more
deferred at pace (48/448 deferred). The requeue's pass-local abort line
behaved correctly (12 paced probes attempted, 36 unattempted — the
2026-10-13-01:15 fix held). The recent phase inherited the sweep's
streak, cooled down the designed 240s cap and its single probe
**succeeded** — the 91-day observed window landed (all final gaps are
2025-era archive games, none recent). The same-run top-up then ran its
discovery (2 quick retries + one 60s cooldown + probe), the window had
re-closed within seconds of the recent probe's success, 18/18 gap
batches deferred, its requeue aborted at its own line of 12, and the run
shipped **142/2023 games with NaN weather**, surfaced correctly by the
gap warning (count semantics from the 2026-10-13-01:07 fix) and by the
monitoring coverage rows (`temp_f`/`wind_mph`/`is_precip`/`is_snow`
LOW_COVERAGE at 58.4%/75.0% in both views). Prior seams all fired as
intended: view-labeled drift lines with 2 `OK-SEASONAL` verdicts, the
`inj_ol_out_*` ALERTs staying alert on real shift, `season_openers_ingested
PASS`, and the missing-finals guard staying silent because TB-DAL had
resolved.

**Defect found and fixed.** Both rate-limit emitters inside the run's
**final** recovery pass promised an in-run recovery that cannot exist:
log line 24 ("recover via the requeue, the same-run top-up or the next
run's cache-miss pass") and line 27's requeue abort ("recover via the
same-run top-up and the next run's cache-miss pass") fired from *inside*
the same-run top-up itself — the pass that was aborting. The 142 lost
games healed only on the next run, so a reader of the remote log was
twice told the hole would close in-run. `weather._recovery_note(final_pass)`
now owns that string for both emitters; `fetch_games_weather` marks its
second top-up pass `final_pass=True`, so those lines name only the next
run's cache-miss pass (tagged "this was the run's final recovery pass"),
while the sweep's messages — whose top-up really does follow — keep
naming both paths byte-identically to the shipped log.

**Verified (exit statuses captured).** production script **549 passed /
0 failed** (6 new honesty pins + the extended top-up wiring pin),
log-review suite 15 passed, fold 13, tee exit 0, feature-name 39,
data-delivery 102. Note: `test_requeue_probes_a_sweep_that_ended_past_the_abort_line`
is slow by design (~94s of real `_BATCH_PAUSE_SEC` sleeps across 48
probes; it does not use the `sleeps` fixture) — it passes, and the suite
takes ~153s in total.

**Limitations.** The message fix changes what the NEXT run's log prints
after a final-pass abort; the 142-game NaN hole itself is unchanged
behavior (deliberate bounded recovery — healing stays with the next
run's cache-miss pass, which the persistent `.nfl_cache` makes cheap: 18
batches vs the sweep's 448). No performance claims: this pass is
observability honesty only.

## Fifth pass — weather structural review: full repull, no cache top-off, MLB rate-limit shape

Scope (user directive): *"The nfl pipeline must not use cache files to
top off weather observations. The observations must be optimized for a
full repull every pipeline run and avoid rate limits. Mlb also
accomplishes this successfully. Guardrail: weather must be able to fully
repopulate every pipeline run without triggering rate limits."*
Existing files only; no new programs.

**What MLB does (the guidance).** `mlb-backend/backend/weather.py` has
no cache in its fetch path at all: `fetch_games_weather` builds targets
from the schedule and fetches EVERY one each run through one paced flow
(14-day windows × 15 locations, 1.0s pause), with a single retry ladder
— 7 attempts, `2**attempt` + jitter ≈ 63s, `Retry-After` authoritative —
that "outlasts one full quota reset so a rate-limited batch drains
instead of being lost", plus loud 80%/per-year coverage gates.

**What NFL had.** A `.nfl_cache` parquet gated the fetch (`missing` =
games not in cache), a defer/requeue/abort machinery for shut windows,
a two-tier cooldown breaker, and 14-day windows → **448 requests over
~20 minutes** per run. On Kaggle the repo is cloned fresh each session
(2026-10-13 10:23 run: `NFL_FULL_REPULL=1`, cold cache → full 448-batch
sweep), the shared-IP quota closed at minute 18, and 142/2023 games
shipped NaN after three recovery passes all met the same shut window —
while local runs with a warm cache silently under-fetched. The top-off
could only ever heal runs whose cache survived; the log's recovery
promises (`next run's cache-miss pass`) assume exactly that.

**What changed (`weather.py`, net −377 lines).**
- **Full repull, no cache gating**: `fetch_games_weather` requests every
  eligible game every run; `refresh=` is a documented no-op (kept for
  `build_weather_table`); delivery is what THAT run observed — a failed
  fetch ships NaN and is never backfilled from disk. The cache file is
  still written as provenance history for the `--skip-pull` cache-only
  reader; `ingestion.clear_cache`'s keep-the-weather-cache rationale and
  `build_weather_table`'s usage text were updated to the new contract.
- **MLB drain ladder**: one ladder for 429/502/503/504 and network
  exceptions (7 attempts, 63s worst case, Retry-After authoritative) —
  a rate-limited batch drains instead of dying.
- **Fail-fast brake**: `_DRAIN_FAIL_FAST_AFTER = 2` consecutive
  full-ladder 429 failures engage the brake (logged once); then a
  still-shut window costs ONE request per batch, the paced sweep keeps
  probing, and the first definitive answer re-arms draining. Bounds the
  measured 37-minute flap at 2×63s of sleep + fast-fail batches.
- **Request budget**: `_BATCH_DAYS` 14 → 60 (NFL's own measured option:
  "357 → 111 request batches"); a full repull of a decade-dense schedule
  now costs ~132 requests over ~6–8 minutes vs 448 over ~20 — a third of
  the requests and of the quota exposure. Pace floor stays ≥2.0s
  (~15 req/min); batch size stays 15 (MLB-proven response size).
- **Removed**: defer mode, `_requeue_deferred_batches`, the cooldown/probe
tier, the retry budget, the abort streak, `_recovery_note`. The same-run
top-up remains as the in-run gap recovery (fresh breaker, gaps only);
recovery messages now name only the truth: the same-run top-up and the
next run's full repull.

**Pins.** `test_log_review_20261011.py` rewritten to the new contract (18
tests: ladder geometry, brake engage/re-arm/one-shot disclosure,
gateway-errors-are-not-quota, None-vs-{} semantics, sweep-completes-
under-a-shut-window, top-up gap-only + fresh breaker, warm-cache-still-
full-fetch, no-cache-backfill, refresh-noop, request-budget ≤140, pace
floor, machinery-stays-gone). `test_production.py` updated: ladder pin
(7 calls, [1,2,4,8,16,32]), top-up shape without per-pass kwargs, 429-line
honesty (two real recovery paths, no "requeue"/"cache-miss"), window pins
(60 days with the measured rationale on record).

**Verified (exit statuses captured).** production **546 passed / 0
failed**; log-review **18 passed** (71s — was 153s with the old requeue
sleeps); fold 13, tee exit 0, feature-name 39, data-delivery 102, all
exit 0. Guards probed live: a fully warm cache still triggers a full
fetch of every target; a failed fetch delivers 0 rows with the top-up
re-attempting exactly the gaps; 2×63s + fast-fail brake bounds a storm.

**Limitations.** The next real run is the first to execute the new
shape (~132 requests; drain/brake lines appear only if the shared IP
shuts a window); the previous 142-game hole class is now healed by the
next run's guaranteed full repull rather than by cache-miss targeting.
No performance claims beyond the request-count arithmetic above.
