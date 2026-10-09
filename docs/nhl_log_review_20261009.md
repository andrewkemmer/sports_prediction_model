# NHL run-log review — 2026-10-09 (the 15:27 ET delivery run)

**Subject:** the run trained 2026-10-09 14:48:41 → 16:12:27 (log clock; **5,027 s**),
artifacts commit `54c85d54`, run log `nhl_pipeline_run_log.txt` = 442 lines,
`NHL_FULL_REPULL=1`, window 2024-01-01 → 2026-10-09 (675 score dates, 2,857
boxscores, 0 cache hits by design), 14 artifacts written.

**Verdict:** delivery is clean — all **11 Phase-13 gates PASS**, drift is fully
quiet (**0 warnings / 0 alerts / 8 seasonal**, where the 10-08 run logged
6 alerts + 1 warning), coverage reports **100.000% measured on eligible games
with no warm nulls** in all three windows, and the headline matches the
committed calibration artifact. Every documented feature-coverage gap was
re-checked against the shipped artifacts; **one was still open** and is fixed in
this review: the player-pool columns can serve the documented **position-prior
fallback**, and the coverage report counted those fabricated values as
measured — `n_default_zero` was hardcoded to `0` while the dashboard printed
"% MEASURED = real observations only (default-filled values excluded)". Measured
on the run's own drift windows after the fix: **200 default-filled cells in the
baseline window (194 on warm games, worst feature 91.2% measured)** and 21 in
the current window (worst 85.0%) — every one of which the 20261009 CSV reports
as 100.000% with `n_default_zero = 0`.

## What was validated

### Gates, geometry, headline

All 11 Phase-13 gates PASS (same set as 10-08): `ml_probability_bounds`,
`oof_population`, `spread_grid_coherent`, `totals_grid_coherent`,
`markets_has_slate`, `slate_contract_fields`, `fold_geometry`,
`season_openers_ingested`, `oof_reach`, `sealed_holdout_gate`,
`delivery_consistency`.

| measure | run log (Phase 5/8/9) | committed artifact |
|---|---|---|
| AUC (oof_regular, n=2,465) | 0.5959569634 | `nhl_calibration_20261009.json` 0.596 |
| log-loss | 0.6747233520 | 0.6747 |
| Brier | 0.2409625056 | 0.241 |
| ECE | 0.0091339743 | 0.0091 |

Fold geometry: 60 expanding windows, 2,665 OOF rows = 2,465 grading + 168
postseason + 189 provisional, reach 93.3%; the 14 thin windows are the usual
playoff weeks/season ramps (the run's only WARNING, excluded from pooled
metrics by design). Members xgboost 0.67563 / lightgbm 0.67632 / elasticnet
0.67614, earned weights elasticnet 0.443 / xgboost 0.329 / lightgbm 0.228; the
pooled calibrator gated out again (`identity`, `gated_no_gain`, 0 of 60 folds
kept a map) and dispersion is at the Poisson limit (α = 0.0000). Delivery:
`nhl_moneyline_v1_20261009.json` carries a 4-game slate (SEA@DET, NYR@WSH,
PIT@CBJ, ANA@WPG), markets 2,669 rows, 4 game-SHAP files, cards history 2,674
(+10).

### The 10-08 remediations, confirmed live

- **Season-seam drift re-check** (`4c83d9df`, floor `max(30, n_c)`): the 10-08
  run logged `1 warnings, 6 alerts, 0 seasonal`; this run logs
  **`0 warnings, 0 alerts, 8 seasonal`** on the same 68 features. The alerts
  stopped paging and the seam explanation took their place.
- **Boxscore refresh window** (`BOXSCORE_REFRESH_DAYS = 3`): the 10-05 warm-null
  signature (`goalie_sv_pct_diff(2), goalie_gaa_diff(2), …`) has not returned —
  `run_engine_feature_coverage_20261009.csv` is 204/204 `OK`, `n_warm_null = 0`
  in every window.
- **Oct-7 board fix**: the rail still folds the dated NHL families (this run's
  4-game slate is dated 20261009 and the markets/cards families are complete).

## Feature-coverage gap ledger (every gap the documented audits state)

| documented gap | source | status now |
|---|---|---|
| Bullpen/boxscore-family NULLs recurring (the 10-05 9-warm-null goalie outage) | `nhl_log_review_20261008.md` Finding 3 | **holds** — 0 warm nulls in the 10-09 CSV; refresh window in place |
| Boxscore-only families span "93 cached boxscores = 2.4% of history" | `nhl_ingestion_audit_20261008.md` §3 | **closed by data** — this run fetched all 2,857 boxscores; on the rebuilt frame goalie/PP/faceoff families read 97.86–99.47% over the whole 2,857-game pool (goalie = season-opener NaNs only) |
| Cold-start classification "determines debut from each window's own first-seen teams … the 16 cold goalie-diff nulls cannot be assumed genuinely cold" | `nhl_model_audit_20261006.md` §H | **fixed (`289e5292`) and now measured**: `_season_open_mask` looks firsts up in the full pool; every one of the **61** goalie-family null rows in the rebuilt pool (18 of them in the last-60 window) is a season opener, **0 non-opener nulls** |
| "Recent goalie-diff coverage is only 70% … early-season quality resets to missing despite available prior-season performance" | `nhl_model_audit_20261006.md` §D | **by design since the 10-06 input fix** — the workload vote counts same-season starts and an opener resolves an honest NaN (never last season's workhorse). Current window reads 73.33% raw / **100% on the eligible denominator**. The audit's proposed remedy (multi-season goalie skill with prior-season shrinkage + a probabilistic starter model) is a **feature-design request**, recorded for the owner, not a coverage defect |
| "Non-null `pl_*` is NOT proof of measured coverage: defaults can yield 100% non-null output" | `nhl_model_audit_20261006.md` §E | **OPEN → FIXED in this review** (Finding 1) |
| "Honest coverage provenance" (Phase 0 do-first) | `nhl_model_audit_20261006.md` §9 | **partly open → fixed**: the report's `n_default_zero` column was a hardcoded `0`; it now counts default-filled cells, splits them cold/warm, and the run log names them |
| "Review PP/faceoff and goalie warm-null coverage" | `nhl_causal_evaluation.md` | reviewed — 0 warm nulls in any family, all three windows |
| Serving-slate blindness (goalie family 96–98% decided / null on every published card) | `nhl_log_review_20261008.md` §Board and coverage | holds — the slate window exists and reads 100% with the serving pool resolved (8/8 sides) |

## Finding 1 — the pool's position-prior defaults were counted as measured (DEFECT, fixed)

**Root cause.** `features.add_player_pool_features` serves a side with no pool
row the documented position prior (`_POSITION_PRIOR` — deliberately not NaN and
not 0, "the prior is the measured default"), so all 24 `pl_*` columns are
non-null however little evidence stands behind them. The coverage builder then
reported a value as measured whenever it was non-null and wrote
`"n_default_zero": 0` unconditionally, so the fabricated rows passed as
observations and the dashboard's "real observations only" caption was false.

**Evidence (measured, not inferred).**

- The shipped run log itself carries the tell: the Phase-3 pool resolved
  **5,690 sides** where the decided frame has **5,714 side-slots** — 24 sides
  served priors (the 10-08 run had the identical 24: 5,670 of 5,694).
- On the locally rebuilt frame (`.freebuff/nhl_frame_full_src_20261008.parquet`,
  2,855×148 — the Kaggle frame's shape), **12 games** carry priors on *every*
  pool group for *both* sides: 2025-06-04 … 06-17 and 2026-06-02 … 06-14, both
  Stanley-Cup-Final stretches. Six of them (June 2026) sit inside the baseline
  drift window, so they were inside the shipped coverage report.
- The fallback is not only whole-side: replaying the run's own drift windows
  through the fixed report shows per-group holes too — `pl_ppo_r_away` reads
  the PPO right-wing prior (1.518) on **7 of the last 60 games** while the same
  side's centre/defence/left PPO values are real measurements (e.g. 2026-10-01
  SJS@FLA: away `pl_ppo_r_away` 1.518 vs `pl_ppo_c_away` 1.974).
- Shipped artifact, read directly: `run_engine_feature_coverage_20261009.csv`
  has `pct_measured = 100.0` and `n_default_zero = 0` for **all 24** pool
  columns in every window.

| window | features with defaults | default-filled cells (warm / cold) | worst `% measured` | shipped CSV said |
|---|---|---|---|---|
| baseline (250 games) | 24 / 68 | 200 (194 / 6) | 91.2% | 100.0%, `n_default_zero` 0 |
| current (60 games) | 5 / 68 | 21 (14 / 7) | 85.0% | 100.0%, `n_default_zero` 0 |

**Fix (`nhl-backend/backend/monitoring.py`).**

- `_pool_default_mask` recognises a default the same way the builder writes it:
  a `pl_*_{home,away}` value still equal to its position prior, and a `*_diff`
  contaminated when **either** side used the prior for that group. Anything it
  cannot judge (non-pool feature, absent side columns) returns None and the
  null-only arithmetic is unchanged.
- `_coverage_row` now reports **measured = non-null − default-filled**,
  fills `n_default_zero`, and splits defaults into `n_cold_default` (the
  documented warm-up — a debut row served the prior) and `n_warm_default`
  (a hole). `pct_measured_eligible` subtracts both cold halves; `cause` becomes
  `position_prior_default` when defaults, not warm nulls, are the story.
- Status stays on the published thresholds: warm **nulls** still force
  LOW_COVERAGE/STARVED (the 10-05 contract), while defaults move the pill only
  through measured% (<80% LOW_COVERAGE, <25% STARVED) — exactly how MLB treats
  its closed-roof policy zeros, and the reason the dashboard caption is
  unchanged. A pool that silently served priors everywhere reads 0% measured
  and **STARVED**; the 91%-measured partial case is disclosed without crying
  wolf.

**Fix (`nhl-backend/backend/master_pipeline.py`).** The Phase-14 verdict warns,
naming the features, when default-filled values sit on warm games
(`… carry default-filled values counted as UNMEASURED — a warm gap, not a
warm-up: pl_ppo_r_away(9 default-filled), …`), and stays informational when
they are the cold warm-up.

**Fix (`nhl-backend/backend/features.py`).** The pool builder counts its own
fallback and the run log now states it every phase —
`player pool: …, 24/5714 side(s) served >=1 position prior (no pool row)` —
so a silent pool outage is visible in the log without arithmetic.

**Fix (`frontend/nhl_market_diagnostics.py`).** The sub-annotation under
% NON-NULL reads "default-filled" (the NHL's defaults are priors, not zeros);
MLB's caption strings and the status pills are untouched.

**Verification.** New pins in `test_run_engine_pit.py`:
`test_coverage_reports_the_position_prior_pool_fallback` (a frame with no pool
source must read 100% NON-NULL but 0% MEASURED / STARVED / `n_default_zero ==
n_games`, while a frame with a real pool measures every column), and
`test_coverage_verdict_names_default_filled_pool_values` (warm defaults warn
and name the feature; cold defaults stay informational; the cold-null wording
now says "team debut or season opener"). The one pre-existing test whose
fixture stood in for a healthy frame now brings its own pool source
(`_pool_ratings`) — its assertions are unchanged.

## Finding 2 — the verdict line called every cold null a "team debut" (wording, fixed)

Since `289e5292` the goalie family's cold set includes season openers (the
opening-night honest NaN), which is not a team debut. The line now reads
`… cold null(s) by design (team debut or season opener)`, so the 114 cold nulls
this run reported in the current window are labelled truthfully.

## Ops observations (recorded, not changed)

- **Duration 5,027 s** — a third full rebuild in three days (10-08 10:42,
  10-08 21:34, 10-09 14:48), all `NHL_FULL_REPULL=1` at ~1.3 boxscores/s.
  With `SCORE_REFRESH_DAYS` and `BOXSCORE_REFRESH_DAYS` in place an incremental
  run is the cheaper default; the Kaggle notebook still pins `NHL_FULL_REPULL =
  "1"` under the comment *"set once, then remove"*. Flagged for the notebook
  owner (Kaggle-owned per `nhl_model_audit_20261006.md`), not edited here.
- **Injury id bridge 3,365/4,571 = 73.6% matched** — the same decomposition the
  10-08 review measured and cleared (goalies have no MoneyPuck skater rows;
  AHL/call-up skaters have no game in the archive). Unmatched count moved with
  the archive, not with the matcher.
- **wrong-team guard dropped 7,578** (10-08: 7,482) — scales with the pool, as
  expected for a team-change guard.
- Retention pruned the 20260928 artifact family (10 files) against the 10-day
  window anchored on the data-window end; 21 protected + 182 in-window kept.
- The run's frame-level coverage table (over all 2,857 decided games) is the
  full-history view the drift windows are not: goalie 97.86%, PP/faceoff
  99.30%, win% 99.30%, elo 100.00%.

## Verification

- `python -m pytest nhl-backend/backend/ -q` — **608 passed in 10:51** (the
  10-08 baseline was 455 + 151 = 606; +2 new pins), exit 0.
- `python -m pytest frontend/test_nhl_frontend.py
  frontend/test_nhl_pooled_oof_parity.py
  frontend/test_nhl_board_render_smoke.py -q` — **11 passed**.
- `python -m pytest nhl-backend/backend/test_run_engine_pit.py -q -k "coverage or verdict or pool_fallback"`
  — **9 passed** (includes the 2 new pins).
- `python -m py_compile` on every edited module — clean; `git diff --check` clean.
- Real-frame replay of the run's own windows through the fixed report (the
  200 / 21 default-filled cells above), plus a direct read of the shipped CSV
  to establish the before-state.
- No new programs: 4 existing NHL modules, 1 existing frontend page, 1 existing
  test module, 1 document.

## Known limitations

- The shipped artifacts are **not rewritten**. The corrected coverage numbers
  land in `run_engine_feature_coverage_*.csv` on the **next pipeline run**
  (Kaggle); until then the committed 20261009 CSV still shows the pool columns
  at 100% measured.
- The serving-slate window of the *shipped* frame is not replayable locally
  (the 4-game slate frame is Kaggle-side); the pool resolved all 8 serving
  sides per the run log, but a per-group fallback inside the slate can only be
  seen from now on — the new side-count log line reports it every run.
- The local rebuild used for the replays is the 2026-10-08 frame (2,855 rows,
  2,857 decided on the runner); window geometry and default counts match the
  run's own numbers, but the frames are not byte-identical.
- The NHL has no `check_production_graph.py` (MLB-only); production wiring is
  covered by the backend suite's import/contract tests.

## Second-run re-review — the 17:40 ET delivery (artifacts `0ef48389`)

The next NHL run landed and both confirmed the morning remediations and
surfaced their one residual item, now closed.

**Remediation confirmed live.** The rebuilt
`run_engine_feature_coverage_20261009.csv` carries `n_default_zero` /
`n_cold_default` / `n_warm_default` with honest `pct_measured` (the pool
columns read 97.6%, not 100.0%), the Phase-14 verdict names the features,
and the pool line reports the fallback count directly
(`368/5714 side(s) served >=1 position prior`). All 11 Phase-13 gates PASS,
drift 0 warnings / 0 alerts / 8 seasonal, run green in 4,830 s, and the
headline still matches the committed calibration artifact.

**The residual "warm gap" WARNING is two documented boundaries, not a defect
(remediated).** Measured on the run's own windows: the baseline's 24 features
× 6 warm default-filled cells are exactly the six June-2026 Cup-Final games,
and the current window's 5 PPO features are the early-October 5on4 evidence
ramp (7-9 cells on `pl_ppo_r_*`). MoneyPuck's player-game archives are
regular-season-only (the ingestion contract), so past the pool's 45-day
serve window the Stanley Cup Final has no pool — a boundary that recurs every
June (the local 2,855-row frame replays both 2025 and 2026 Cup Finals as 12
defaulted cells per pool feature, every one on a playoff row) — and the ramp
self-heals as power-play ice accumulates past the 900 s per-player floor.
The 2026-10-05-style true warm gap has not recurred in either run since.

- **Fix (`nhl-backend/backend/monitoring.py`).** Warm defaults whose games
  are ALL playoff rows now classify as
  `position_prior_postseason_boundary`; a single regular-season default keeps
  `position_prior_default`. Measured%/status arithmetic is untouched — the
  priors stay UNMEASURED and move the pill only through the published
  thresholds.
- **Fix (`nhl-backend/backend/master_pipeline.py`).** The Phase-14 verdict
  warns only on non-postseason warm defaults (the true gap the 10-05 outage
  trained it on) and discloses the postseason boundary as its own INFO line
  naming the source contract; a mixed window does both — the warning carries
  the boundary count.
- **Pin** (`nhl-backend/backend/test_run_engine_pit.py`, existing file):
  `test_coverage_splits_postseason_pool_boundary_from_warm_gap` — a
  boundary-only window never WARNs and names the boundary; one
  regular-season default keeps `position_prior_default` plus the WARNING; a
  mixed window warns on the hole and discloses the boundary count; a frame
  without `is_playoffs` keeps the old cause. Verified on the real frame by
  replaying it through the fixed report (Cup-Final rows → boundary,
  scattered regular-season group holes → `position_prior_default`).

**Known limitation.** The shipped 19:01 artifacts are not rewritten; the new
cause value lands in the coverage CSV and the verdict wording lands in the
run log on the next pipeline run. The notebook-owned `NHL_FULL_REPULL=1`
pin and the ops observations above are unchanged.

## Third review — the remote 19:01 delivery (`0ef48389`), and one residual
coverage-report defect (remediated)

Re-reviewing the same remote delivery that closed above, this time walking
the whole 428-line log and replaying the run's own drift windows through the
fixed report on the real 2,855-row frame. The delivery itself is confirmed
clean and the second review's boundary remediation is confirmed to land on
the next run; the replay surfaced one residual defect **in the remediation
code itself**, now fixed.

**Delivery re-verified.** All 11 Phase-13 gates PASS; run green in 4,830 s
(`NHL_FULL_REPULL=1`, 2,857/2,857 boxscores fetched, 0 cache hits — the
notebook pin above, unchanged). Drift: 0 warnings / 0 alerts / 8 seasonal on
both views. The three log WARNINGs are the already-triaged classes: the
14-window OOF thin-population line (provisional by design, 189 rows excluded
from grading) and the two warm-default coverage lines (the pre-fix wording —
this run started 17:40 UTC, the boundary split landed 21:28 UTC, so its
output belongs to the next run as stated). New since the 10-05 outage log:
the **pre-game lineup channel is populated** (257 intervals, bound through
2026-10-06), so the "historical `pl_*` availability gap stays OPEN" warning
is gone; the pool line reports `368/5714` prior-served sides and the slate
served `0/8` — every serving side had a real pool row.

**Boundary fix replayed on the real frame (verified to land).** Slicing the
run's own windows (250-game baseline / 60-game current) out of the
2,855-row rebuilt frame and running them through the current
`monitoring.coverage`: the baseline's six warm defaults per pool feature
that sit only on the June-2026 Cup-Final games classify
`position_prior_postseason_boundary` — 18 features (all 12 EVO + `pl_ppo_c_*`
+ `pl_ppo_d_*`) — while the six `pl_ppo_l/r_*` features whose warm defaults
mix Cup-Final rows with regular-season per-group holes keep
`position_prior_default`; the current window's three `pl_ppo_r_*` rows stay
`position_prior_default` (the early-October 5on4 ramp). The next run's
verdict will therefore WARN on the 6 + 3 true-gap features and disclose the
18 boundary features on the INFO line, exactly as the remediation intends.
Window geometry and per-feature default counts match the shipped CSV
cell-for-cell within the two-game frame difference (2,855 local vs 2,857
runner).

**Residual defect (found by the replay, fixed).** The first replay — run
with `from backend import monitoring` (the package-import context every
frontend tool, audit replay, or external consumer uses) — reported **zero
defaults in either window**, i.e. silently the pre-Finding-1 behavior. Root
cause: `_pool_default_mask` resolved its prior with a bare
`from features import _POSITION_PRIOR` inside a try/except. That name
resolves only when `nhl-backend/backend/` is `sys.path[0]` (the production
run and the test suite both put it there), so under a package import it
raised ImportError, the guard swallowed it, and the mask returned None —
reverting the coverage report to *defaults counted as measured*, the exact
2026-10-06 defect the mask exists to prevent, with no signal at all. The
failure mode is the dangerous kind: the report still prints a healthy
measured% and nobody is paged.

- **Fix (`nhl-backend/backend/monitoring.py`).** The mask now reads the
  prior through the module-level `feat_mod` binding (features is imported
  once at the top in *both* run contexts), removing the second import and
  its silent failure path. Verified in both contexts: package import and
  backend-dir script now return identical masks (`pl_evo_c_home` at the
  prior → default; a diff defaults when either side still reads the prior).
- **Pin** (`nhl-backend/backend/test_run_engine_pit.py`, existing file):
  `test_pool_default_mask_reads_prior_through_module_binding` — swapping
  `_POSITION_PRIOR` on the already-imported module changes what the mask
  marks default, proving it consults `feat_mod` rather than its own import;
  a non-pool feature still returns None (unchanged contract).

**Coverage-audit status after this review.** The gap ledger above is
unchanged — every documented gap is closed or holding, and the second
review's residual boundary item is verified to land on the next run. The
only outstanding coverage-audit line is environmental, not code: the
committed 20261009 CSV and log still carry the pre-fix labels until the next
Kaggle run executes, and the notebook-owned `NHL_FULL_REPULL=1` pin (a full
repull every run) remains flagged for the notebook owner, not edited here.

**Verification (this review).** Real-frame replay of both shipped windows
through the fixed report (the 18/6 and 3-feature splits above); the new pin
plus the surrounding coverage/verdict pins **11 passed**;
`py_compile` clean on `monitoring.py`; `git diff --check` clean. The full
backend suite is recorded in the commit message. No new programs: the
replay script lives in the ignored `.freebuff/` diagnostics directory; the
commit touches only the two existing files above plus this document.
