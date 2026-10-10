# NFL EPA position-quality predictive validation and coverage follow-up — 2026-10-10

## Verdict

**No computational defects were found in the `epa_{qb,wr,te,rb}` family — the
causal and arithmetic verification passes every test the NBA RAPM replay
failed.** Prefix-deletion replay shows byte-identical pre-cut features at
three cut dates (1,486 / 2,207 / 2,748 pre-cut games); an independent
reimplementation of the whole chain — opportunity table, rolling-8 windows,
as-of league priors, top-11 cut, shrinkage, per-game weighting — reproduces
all **2,203 real aggregate cells with max difference 0.0**, NaN-for-NaN. One
*tooling* defect outside the family surfaced and is repaired (the OOF
replay's quiet-progress contract divided by zero). Predictive value: the
family's incremental signal is concentrated in the **QB** position; the
pooled family contrast is directionally positive but not a statistical
pass; WR/TE/RB leave-one-out point estimates lean slightly the other way
with intervals crossing zero. **No membership, hyperparameter, blend or
serving change was made or should be made on this replay.**

This review is the first predictive validation of the family (structurally
adopted 2026-09-27 as the MLB lineup-wOBA analogue — position quality for
the projected offensive lineup). It supersedes no prior claim: coverage had
been reported OK, but no paired walk-forward evidence existed for the
family's serving membership.

## Calculation and causal verification — no repairs needed

Static gates, all confirmed in source:

- **Strictly-prior-date candidate pool.** A rating row qualifies only when
  `rating_day < target_day` (calendar dates, not kickoffs): PBP supplies no
  authoritative end-of-game publication time, so a same-day prior kickoff is
  never treated as knowable. Rolling player windows are inclusive through
  the player's own most recent observed game, which then serves only
  strictly-later targets.
- **Point-in-time league priors.** `mu` (pooled EPA/opportunity) and `k`
  (20% of the expanding median rolling-8 opportunity total) are as-of
  joined strictly before each target day via `searchsorted(...,
  side="left") - 1`; a season's first dates honestly carry NaN priors
  rather than a future-anchored mean.
- **Kickoff integrity.** `_kickoff_utc` parses real ET schedule times only;
  ambiguous/nonexistent local times become NaT, and a target game without a
  parseable kickoff is dropped from the aggregate (its features stay NaN —
  never a fabricated permissive timestamp). Injury designations admit only
  `published < kickoff_utc` rows with explicitly zoned timestamps; naive
  timestamps are rejected.
- **Availability overlays only remove, never add**, keyed to the target's
  own `(season, week, team, player)`: the strict-PIT channel
  (2016–2024), the weekly report-cycle channel (Out/IR/Doubtful; Questionable
  stays eligible), and the roster-snapshot channel (RES/INA/CUT/SUS/PUP,
  probe-validated carry rules, the "Achane proof" in ingestion).
- **Season-boundary carry** (300 days) re-fills only the `(game, team)`
  pairs the 21-day fresh window found nothing for — the in-season team the
  fresh window covers never pulls stale rows.

Dynamic evidence ([causal checks](nfl_epa_review_20261010/causal_and_coverage.json)):
rebuild the full decided frame (2,826 games, 2016-09-08 → 2026-10-08, 267
columns) through the production `build_game_features` with cache-only
inputs, then rebuild again with the family's only play-level sources (pbp,
ps) truncated strictly before three cut dates:

| Cut | Pre-cut games | Columns with leaks (of 12) |
|---|---:|---:|
| 2021-11-15 (mid-season) | 1,486 | **0** |
| 2024-09-10 (Week-1 boundary) | 2,207 | **0** |
| 2026-01-05 (season end) | 2,748 | **0** |

Every pre-cut `epa_*` value is byte-identical between the full and truncated
builds — no target-day admission, no future-derived shrinkage, no
order-dependent state.

### Independent arithmetic and timing evidence

[Arithmetic evidence](nfl_epa_review_20261010/arithmetic_checks.json):
independently recompute the 2024 window (277 targets, 2024-09-01 →
2025-01-12) from raw PBP in plain pandas — role-flag opportunity table with
dual-role dedup, FB→RB normalization, per-player rolling-8 sums, the
fresh-window/season-carry pool, the three availability overlays, the top-11
workload cut, the as-of `mu`/`k` shrink, and the per-game weighting — and
compare against the production `epa_opportunity_table` →
`epa_quality_ratings` → `epa_quality_team_agg` chain on identical inputs:
**2,203 matched (game, team, position) cells, 0 key mismatches, 0
NaN-pattern disagreements, max |Δ| = 0.0.** The served diff identity
(`epa_*_diff == home − away`) holds with max residual **0.0** for all four
positions across the whole rebuilt frame.

## Tooling repair (outside the family)

`moneyline.walk_forward_oof(..., progress_every=0)` raised
`ZeroDivisionError` — the modulo progress guard divided before checking the
documented "log nothing" value (MLB's replay accepts 0). Production's
default of 25 never hit it; an audit harness did. The guard now
short-circuits on 0
([moneyline.py](../nfl-backend/backend/moneyline.py#L314)), pinned by a new
test in the existing
[test_fold_season_split.py](../nfl-backend/backend/test_fold_season_split.py)
(`test_walk_forward_accepts_quiet_progress`). Default-path behavior is
unchanged.

## Predictive validation: matched production walk-forward, no storytelling

- Same population and machinery as production: the rebuilt decided frame
  (2016 warmup + 2017–2026 core), one fold list of **200 expanding windows
  computed once and passed to every arm** (identical train/val indices),
  xgboost/lightgbm/elasticnet members, train-fold preprocessing, causal
  rolling blend re-earned per fold, and the pipeline's own Phase-8
  per-fold gated Platt producing the served `p_ensemble_calibrated` path.
- Grading on `grades_pooled` (93 thin windows are provisional and excluded
  exactly as production excludes them): **1,689 grading games on 327
  dates**. In this replay every fold's calibrator gated to identity (the
  documented degenerate-Platt behavior on thin OOF evidence), so the
  calibrated and causal columns coincide.
- Six preregistered arms: as-served; minus the whole 12-column family; and
  minus each position's three columns. Nothing tuned; canonical IDs, labels
  and grading masks are identical across arms.

| Grading population (n=1,689) | AUC ↑ | log loss ↓ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| **As-served (all 12)** | **0.697713** | **0.630125** | **0.220009** | 0.029316 |
| Minus the whole epa family | 0.692849 | 0.632501 | 0.221157 | 0.030237 |
| Minus QB | 0.692189 | 0.632131 | 0.221062 | **0.024815** |
| Minus WR | 0.698834 | 0.629934 | 0.219886 | 0.030496 |
| Minus TE | 0.698586 | 0.629508 | 0.219767 | 0.030332 |
| Minus RB | 0.698457 | 0.629753 | 0.219842 | 0.030885 |

[Paired uncertainty](nfl_epa_review_20261010/predictive_metrics.json), 1,200
shared-date bootstrap replicates (same-night games resampled together),
as-served minus contrast:

- **Family (minus all 12):** AUC **+0.004864**, 95% interval
  **[-0.001268, +0.010995]**; log loss **-0.002376**, interval
  **[-0.005910, +0.001196]**; Brier **-0.001148**, interval
  **[-0.002744, +0.000480]**. All three pooled intervals cross zero.
- **QB:** AUC **+0.005525**, interval **[+0.000480, +0.010588]** —
  excludes zero; log loss -0.002006 [-0.004997, +0.000953]; Brier -0.001053
  [-0.002408, +0.000303] cross zero.
- **WR / TE / RB:** point estimates slightly favor the *smaller* family
  (e.g., WR: AUC -0.001120 [-0.002733, +0.000665]); every interval crosses
  zero — the non-QB positions are not distinguishable on this replay.

**Adoption decision:** keep the family exactly as served — removing it
costs point estimates on all four pooled metrics, and the QB leave-one-out
AUC interval excludes zero — but do **not** claim an established all-family
gain, do **not** remove WR/TE/RB, and do not re-tune anything on this same
replay. These are unadjusted exploratory multi-contrast intervals, not a
pristine holdout or a family-wise guarantee; ECE is slightly *worse* with
the full family (0.0293 vs 0.0248 minus-QB). The appropriate next gate is
a sealed future-season comparison with calibration and postseason strata,
mirroring the NBA review's admission standard.

## Feature-coverage follow-up

[Sliced coverage](nfl_epa_review_20261010/causal_and_coverage.json) over
the rebuilt frame, all 12 columns:

| Slice | n | QB diff | WR diff | TE diff | RB diff |
|---|---:|---:|---:|---:|---:|
| Full history (2016–2026) | 2,826 | 96.67% | 99.40% | 97.84% | 99.40% |
| Core (2017+) | 1,923+636 | 97.15% | 99.96% | 98.40% | 99.95% |
| Week 1, core | 159 | **99.37%** | 100% | **95.60%** | 100% |
| Weeks 1–4, core | 636 | 97.96% | 100% | 97.64% | 100% |
| Season 2026 | 65 | 98.46% | 100% | 98.46% | 100% |

The **Week-1 season-boundary carry** (added 2026-10-08, when the family's
largest structural hole shipped NaN on every opener) holds on the real
frame: Week-1 coverage is 95.6–100%, and the residual TE holes are
team-games with no admissible prior tight-end opportunity, which the family
honestly reports as NaN rather than fabricating. The delivered monitor
artifact (`run_engine_feature_coverage_20261009.csv`) reports all 24 EPA
rows **OK** at 96.4–100% in both windows — reconciling with this rebuild.

Serving surface ([live probes](nfl_epa_review_20261010/live_source_probes.json)):
the delivered week-5 SHAP artifact (`nfl_shap_game_2026_05_SF_SEA.csv`)
carries **nonzero attributions for all 12 columns**, with `epa_qb_diff` the
largest single driver of that game's forecast (shap -0.215 for SEA) — the
family is priced by the shipped model, and the values behind those
attributions are exactly what parts B/C reproduce from raw PBP.

## Live sources and freshness

Bounded probes, fetched 2026-10-10 ~01:40 UTC:

- ESPN scoreboard finals for the last three 2026 game days
  (10-04, 10-05, 10-08): **16/16 completed events match the schedule
  cache's final-score multisets, 0 unmatched.**
- A **fresh nflverse PBP pull** (`nflreadpy.load_pbp([2026])`) at probe
  time: 11,327 rows, identical to the cache; across the latest 16 games,
  per-game play counts and per-game EPA sums are identical
  (**max |Δ| = 0.0**) — the cache feeding this review is the live source.

These are sampled network checks, not full-window freshness proof. The
weekly injury feed remains **pre-merged per player-week with no
publication timestamps** (verified in the cached parquet), so the
report-cycle rule stays the only admissible gate for 2025/2026 — a source
limitation, honestly documented rather than papered over.

## Verification and evidence contract

- Full NFL backend suite: run recorded in
  [backend tests](nfl_epa_review_20261010/backend_tests.txt) with the
  repair above and its pin; no other test changed.
- The frame rebuild, causal replay, arithmetic reproduction and the six
  walk-forward arms were driven by **uncommitted scratch harnesses**;
  curated JSON evidence only is committed under
  [nfl_epa_review_20261010/](nfl_epa_review_20261010/). The machine
  outputs (OOF frames, rebuilt parquet) stay local — inspect the JSON
  claims without trusting prose.
- Production changes are limited to the one-line progress guard in an
  existing module plus its pin in an existing test file. **No new
  programs, no feature-set, model, blend, or pipeline changes.** The
  served 70-column contract, the epa family's membership and every
  hyperparameter are untouched.
- This is historical replay over cached sources, not a guarantee of
  future AUC/loss/Brier performance.
