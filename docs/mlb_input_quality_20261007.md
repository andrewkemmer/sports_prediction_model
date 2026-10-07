# MLB input accuracy, coverage, and causal model review — 2026-10-07

## Scope and acceptance

Follow the committed NHL [input-semantic](nhl_input_repairs.md) and
[causal-evaluation](nhl_causal_evaluation.md) structure: repair observed inputs
before claiming model quality, preserve causal fold-origin evaluation, and move
corrected features and final refits together. The request authorizes smoke tests,
a commit and push, **no new programs**, and retention of full feature coverage.

This change edits existing production/test modules only. The **109 numeric
moneyline features**, feature names, existing three-member roster, model
hyperparameters, pooling/weight optimizer, and artifact filenames remain intact.
Tests extend existing files; scratch harnesses and joblib/OOF outputs are not
committed. Non-executable audit records accompany this report. Production
predictions, feature CSV, and fitted model have **not** been replaced by scratch
measurements.

## Findings and implemented contracts

### 1. Official Statcast semantics, source schema, and identity

[Ingestion](../mlb-backend/backend/ingestion.py) retains `post_home_score`,
`post_away_score`, and `launch_speed_angle`. According to the
[official Savant CSV definitions](https://baseballsavant.mlb.com/csv-docs),
`home_score`/`away_score` are **pre-pitch**, post-score fields are **post-pitch**,
and classification **6 is Barrel**. Dropping the observed fields forced incorrect
last-pitch results and velocity/angle barrel proxies.

[Engineering](../mlb-backend/backend/features.py) now uses post-pitch last scores,
post-score PA deltas, and observed barrel classification rather than the old
98-mph/26–30-degree team proxy (the pitcher proxy used a different 22–36 band).
Unknown classifications/scores remain NULL; even a missing PA score baseline
cannot become zero through DuckDB `GREATEST(NULL, 0)` behavior. These are source
semantic corrections, not a statistical retune. Pitcher `ERA` remains the
existing runs/9 approximation, not newly asserted official earned runs.

Legacy pitch caches without the three observed fields automatically re-pull the
**requested historical range**; engineering's source-schema guard rejects a
legacy source instead of silently synthesizing fields. Alias normalization
coalesces canonical holes and multiple aliases without duplicate columns.
Duplicate engineered game keys abort. [Official results](../mlb-backend/backend/results.py)
never falls back from a real unmatched game PK to date/team, and refuses ambiguous
PK-less doubleheader matches. Unknown score totals are not fabricated as zero.

### 2. Uniform historical/recomputed/slate feature representation

`add_diff_features` previously erased saved weather products when no new fetch
was supplied. Observed wind/density **levels** now reconstruct products with the
current starter differences; partial weather fetches preserve existing levels.
Closed-roof neutral policy is retained, but a density-dependent product cannot
claim an observation without density. Coverage distinguishes closed-roof policy
zeros from measured values. Product-of-diffs composite features are intentionally
not rewritten into home-minus-away products.

[Slate construction](../mlb-backend/backend/data_ingestion.py) now carries the
three-game bullpen WHIP family and chooses each batting lineup's OPS against the
**opposing** probable starter's hand, preferring current schedule evidence over
strictly-prior observed hand. Unannounced starters remain unknown. Historical and
probable starter IDs share categorical aliases (starter/venue categories remain
inactive candidates, not newly enabled predictors).

UTC prior filtering supports pandas nullable strings and timezone-aware inputs,
excludes invalid timestamps, and preserves false start-time provenance. Stable
same-date row order improves reproducibility. First-season slates reset W/L,
leave zero-games win percentage unknown, and apply Elo season reversion.

[Coverage reporting](../mlb-backend/backend/explainability.py) emits missing-column
rows rather than omitting them, counts finite observations only, records invalid
values, and writes a readable empty schema. The master checks the **generation
universe**, not only an adopted serving subset: absent/entirely unobserved history
columns fail before training. Infinite model values fail explicitly; individual
warm-up/availability NULLs are not zero-filled to manufacture 100% coverage.

### 3. Honest blend/calibration evaluation and fit/refit behavior

[Training](../mlb-backend/backend/training.py) retains each fold's entering blend:

| View | Contract |
|---|---|
| `home_win_prob_model` | Causal rolling blend; headlines, selection, calibration, history |
| `home_win_prob_model_causal` | Explicit alias in local OOF diagnostics |
| `home_win_prob_model_retrospective` | Final learned weights replayed on their fitting outcomes; **NOT OOF**, diagnostic only |
| Future slate | Full-history refitted members, all-prior learned weights, final-origin gated map |

Member probabilities and entering row weights are saved in gitignored
`mlb-backend/run_diagnostics/mlb_oof_moneyline.csv`. Final retrospective metrics,
probability policy, and calibration gate audit are separately recorded in
season-split/bundle metadata. Headline reliability buckets and calibration
metrics use the same regular/non-provisional grading mask; postseason and
provisional rows still receive predictions and separate reports.

[Calibration](../mlb-backend/backend/calibration.py) shares NHL's nested gate at
**every** fold and final origin: early prior fit, recent 25% holdout (minimum 200),
minimum 300 fit rows, require >0.005 nats logloss improvement, then refit accepted
maps on all eligible prior rows. Otherwise identity. The previous final-only
aggregate LL/ECE harm check remains a compatibility helper, not deployment policy.

LightGBM's unchanged fixed budget receives no scored-window eval set, matching
final refit. Fold metrics use the actual serving predictor/pooling interface.
Short histories no longer generate train/validation-overlapping fake OOF.
Validation windows of fewer than five games are scored provisionally rather than
silently lost; training warm-up below ten rows is explicitly excluded. Fit or
prediction failure aborts instead of silently dropping a member/fold.
[Feature selection](../mlb-backend/backend/feature_selection.py) isolates/restores
blend and XGB budget state per candidate, learns weights only for later origins
from eligible rows, and honors its requested date cutoff.

### 4. Model structure/parameter verdict

Reviewed the actual operating roster: XGBoost, LightGBM, elastic-net logistic;
team IDs are the active tree categorical fields. XGBoost retains depth 2, gamma 4,
learning rate 0.055, a 50-round minimum and prior-origin probe-budget transfer
(2000-round / 20-early-stop measurement probes never select their own scored
model). LightGBM retains 364 trees, depth 7, 40 leaves. Logistic preprocessing
continues to fit only training rows. Native missing routing and separate linear
representation are retained.

**No parameter, category, feature-subset, or optimizer change was adopted.**
Further tuning against a semantic-stale feature cache would confound the input
repair and reward already-inspected outcomes. Rebuild first; compare candidates
with a genuinely new temporal holdout and the corrected causal policy.

## Measured evidence (not a full corrected-history rebuild)

[Full measurement JSON](mlb_input_quality_20261007/metrics.json),
[109-feature coverage](mlb_input_quality_20261007/coverage.csv), and the
[durable summary record](../mlb-backend/data_delivery/mlb_input_quality_20261007.json).

- Saved cache: **7,397 unique games × 297 columns**, 2024-03-20 through
  2026-10-06; all 109 model features present/observed somewhere, no infinite
  values or duplicate PKs. Lowest repaired feature non-null coverage is 86.89%.
- Raw saved weather coverage was **86.89% wind / 88.24% air**, NOT 26%.
  The defect occurred on recomputation: **25.90% / 26.51%**. Repair restores
  **86.89% / 88.17%**. Five prior dome-density zeros without observed levels are
  deliberately removed; full column retention is not a claim of full row coverage.
- Both full comparisons use the **same corrected causal evaluation**, unchanged
  production statistical budgets, 83 scored windows, **6,879 grading rows** and
  **7,026 total OOF rows** (including two previously lost tiny-window rows).
  Only the weather recomputation arm changes; this does **not** measure the full
  new Statcast/post-score/barrel feature representation.

| Same causal policy, cached features | AUC | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|
| Old weather recomputation | 0.5721 | 0.6821 | 0.2446 | 0.0082 |
| Preserved observed weather | 0.5729 | 0.6820 | 0.2445 | 0.0072 |

The [paired fold-block bootstrap](mlb_input_quality_20261007/paired_uncertainty.json)
(1,000 resamples, seed 42, 74 graded blocks) gives AUC delta **+0.000716** with
95% interval **[-0.001335, +0.002683]**, and LL delta **-0.000098** with interval
**[-0.000594, +0.000464]**. Both cross zero. This is descriptive evidence,
**not a proven gain or a sealed holdout**. The separately retained
[recent 12-fold slice](mlb_input_quality_20261007/recent_slice_metrics.json)
reverses direction: AUC 0.6021 → 0.5988; LL 0.6745 → 0.6747. Do not cherry-pick.
Final calibration is gated to identity in both full arms. Final-weight
retrospective replay (repaired arm AUC 0.5744 / LL 0.6815) is not comparable to
honest OOF and never becomes the headline.

Final rerun wall times were 40.87s / 38.62s with two learner threads, unchanged
statistical budgets. These are observations, **not a benchmarked speedup** or
production runtime forecast. Historical raw rebuilding and weather backfill were
not included.

## Final verification and scope

- [Final backend suite](mlb_input_quality_20261007/test_results.txt): **337 passed**,
  3 warnings, 16.13 seconds, exit 0. Baseline was 310 passed / 3 failures on
  Windows. Live-memory reporting now uses the real Windows process working set;
  Unicode tee fixtures use explicit UTF-8, preserving their assertions.
- [Direct contract CLI](mlb_input_quality_20261007/contract_cli_results.txt):
  **36 passed**. Existing direct test entry now uses pytest to supply isolated
  fixtures; no tests are skipped.
- Compilation and `git diff --check` pass. No configured/installed mypy or
  pyright is available; compilation is not a substitute for static type checking.
- [Official input smoke](mlb_input_quality_20261007/official_source_verification.json):
  real saved 2024-04-01 Savant sample, **4,190 pitches / 14 unique games**,
  exact production source-stage post scores verified before official overlay;
  **one** last-pitch pre-score game was wrong. **61 official barrels vs 20 narrow
  proxy barrels; 41 official barrels missed by that proxy**. Actual
  `features.build_features` produced 14 games and 4,190 PBP rows. Classification
  coverage on BIP: 99.86%; missing remains unknown. The isolated sample lacks
  complete lineup/IL/position ledgers, so its degraded rolling coverage is not
  evidence of full historical enrichment.
- [Interface verification](mlb_input_quality_20261007/interface_verification.json):
  actual full-cache final refit, joblib persist/reload/schema check, live Oct. 7
  four-game schedule, and actual prediction interface produced finite probabilities.
  Bullpen three-game coverage 4/4; handedness matchup coverage 3/4 with one genuine
  TBD starter (not filled). Exact master artifact writers reconciled 6,879
  calibration/bucket rows and 7,026 history rows. Streamlit AppTest rendered actual
  Calibration, Model Monitor, and Today's Games pages from generated local scratch
  artifacts, checked displayed metrics/team content, and found no exceptions.
- The master is notebook-shaped and has package installation, clone/ingestion,
  and automatic push at top level. It was **not imported/executed end-to-end**.
  Its existing writer code and coverage gate were executed via AST extraction to
  avoid those side effects. Interface smoke excludes weather fetch, run-engine
  market regeneration and SHAP generation; missing run-engine scratch artifacts
  correctly warn. This is not a full production delivery/deployment test.
- [Production graph evidence](mlb_input_quality_20261007/production_graph.json):
  tracked changed-file export passes (27 reachable/allowlisted modules). The shared
  workspace guard still exits **1** solely for the user's pre-existing untracked
  research script `measure_blend_views.py`; it is untouched, uncommitted, and not
  allowlisted to hide the failure. No new programs are in this commit.

## Required rollout and remaining accuracy risks

1. Pull the pushed `main` in the established MLB run environment and run the
   **complete** master: requested-history source refresh, feature rebuild,
   causal OOF/blend/gated calibration, and final refit. Schema is
   **`mlb-v2-observed-statcast-causal-blend`**; old-schema serving is rejected.
   First semantic-cache refresh may be costly; no new notebook or program is needed.
2. Confirm all 109 generation columns have finite observed history, inspect
   measured/default/NULL coverage on the slate, and resolve stale/TBD starter
   warnings from real evidence—not neutral imputation.
3. Retain the local OOF diagnostics/run log; compare causal regular,
   postseason and provisional populations separately. Prior published
   final-weight retrospective metrics are not a valid before/after baseline.
4. Measure the newly rebuilt representation before claiming AUC/LL improvement
   or retuning member budgets/blends. Current historical feature CSV/model are
   intentionally not claimed to contain the corrected source representation.

**Unresolved:** no full three-season corrected Statcast rebuild/retrain was run;
schema presence does not prove every inner historical source gap is complete.
Date-only SQL rolling ties do not fully resolve doubleheader chronology; some
slate carried entering-state statistics lag one game. Historical lineup captures
lack universally archived pregame availability timestamps. Pitcher runs/9 is not
official ERA, and the previously rejected experimental outs correction remains
OFF. Partial PA/pitch datasets can still understate aggregates. Source forecasts,
roofs, weather and starter availability need production-run coverage review.
These limitations prevent any assertion that all MLB inputs now have universal
row-level accuracy or coverage. No unknown values were fabricated to satisfy the
coverage guardrail.

## Second-pass structural review (2026-10-07, same day)

A follow-up defect-hunting pass — classic-bug pattern sweeps plus a
data-driven audit of the committed feature CSV — found and fixed five more
issues in existing files. No new programs; hyperparameters and blend policy
remain untouched. The delivery summary JSON carries the machine-readable
record (`second_pass_review`).

### Defects found and fixed

1. **Run-engine single-class crash (9 call sites).** `score_market` and
   `_winner_card_stats` called `log_loss` without `labels=[0.0, 1.0]`.
   Verified empirically in this environment: sklearn **raises**
   `ValueError: y_true contains only one label` on single-class input. A
   holdout where every favored pick won, or an all-over/all-cover window,
   would abort winner-card and market artifact generation mid-run. All nine
   calls now carry explicit labels — the same contract
   `training.compute_metrics` already enforces. Regression tests cover both
   functions with all-ones targets (they fail before the fix).
2. **`merge_result_cache` NaN finality.** `astype(bool)` alone maps NaN to
   `True`, so a fresh partial row with unverifiable `is_final` could win the
   merge and ship stale scores as a verified final. Now `fillna(False)` with
   a stable tie order (cached-then-fresh precedence is deterministic); a new
   test pins both the verified-final-wins and newest-non-final-wins cases.
3. **Fabricated historical start times never repaired.** All **7,397** rows
   of the committed `game_level_features.csv` carry the documented 19:00-UTC
   fallback with `start_time_observed=False`. Consequences: the non-backfill
   history weather gate (`MLB_WEATHER_BACKFILL_ALL=0`) is permanently dead
   behind `observed=False`; within-day PIT ordering collapses to equal
   timestamps (doubleheader ties); evening-game counts and line as-of joins
   read placeholder hours. New `results.refresh_start_times` backfills
   authoritative StatsAPI first pitches for **unobserved rows only** —
   never overwrites an observed timestamp, never marks an unmatched row
   observed, and stops querying once everything is observed. It is wired
   into `run_daily_pipeline` before diffs/weather, best-effort on network
   failure (the default `WEATHER_BACKFILL_ALL=1` weather path already
   fetches real times on demand and is unchanged). **Live-verified** against
   real 2024-04-01 game pks: `744875 → 20:05Z`, `745109 → 22:50Z` replacing
   the fabricated 19:00; fabricated/unmatched pks correctly stayed
   unobserved (honest no-match logged).
4. **Weather gate provenance.** The master's observed-time gate used
   `fillna(True)` — unknown provenance treated as observed, inconsistent
   with `load_game_features`' round-one `fillna(False)` rule — and would
   re-fetch every observed row each run. It now treats unknown as unobserved
   and skips rows already carrying a wind observation (only genuinely
   unweathered observed rows fetch). Behavior-neutral in the default
   backfill mode (this branch is not taken).
5. **Non-stable sorts on ordering-sensitive paths.** Added `kind="stable"`
   to the market-line pick (equal `line_posted_at`), slate doubleheader
   re-key (tied/NaT start times → deterministic ordinal suffixes), the team
   records walk, and the slate export order.

### Data-driven audit of the committed frame

- 109/109 universe columns present; zero duplicate or case-insensitive
  duplicate column names; no outcome columns (`home_score`, `home_win`,
  `total_runs`, `correct`, …) inside the universe.
- `home_win` is **100%** consistent with final scores across all 7,397
  decided rows.
- Leakage scan: max absolute feature–target correlation is **0.114**
  (`win_pct_diff`, followed by `elo_diff` 0.111) — no column leaks the label;
  no all-null features; the only constant column is the by-design `is_home`
  indicator (PSI/drift is constant-safe: `min==max → 0.0`).
- Elo season-revert parity confirmed between the historical
  (`compute_elo_entries`) and slate (`compute_elos_up_to`) paths; pitch-cache
  merge precedence confirmed correct in both directions (newer copy wins).

### Verification

- [Final backend suite](mlb_input_quality_20261007/test_results.txt):
  **342 passed** (337 from round one + 5 new regression tests), 3 warnings,
  exit 0 — the linked log is the second-pass run; round one's own run was
  337 passed.
- Production artifact/frontend interface smoke re-run after the slate/sort
  changes: **exit 0**, identical reconciliation (6,879 grading / 7,026
  history, 4 games scored, 3 pages, 0 exceptions).
- Live StatsAPI start-time refresh: **exit 0** (recorded above).
- `compileall` and `git diff --check`: clean.
- Remaining round-one limitations are unchanged: no full corrected
  three-season rebuild, no end-to-end notebook run, workspace production
  graph still reports only the pre-existing untracked
  `measure_blend_views.py`, no mypy/pyright, and no proven AUC/log-loss
  gain.
