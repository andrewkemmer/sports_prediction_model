# NBA coverage audit — 2026-10-08 snapshot

## Verdict

**NBA coverage is substantially stronger than NFL's, but not a clean pass.** Source reconciliation, box semantics, fold geometry, and side/diff coherence all measure clean. Three real gaps remain: the center-lineup RAPM segment has persistent (not opener-only) holes; the upcoming slate's availability is archive-only because game-day filings don't exist yet; and the delivered artifacts predate the current feature contract.

This is an audit, not a remediation or model-quality claim. No production predictions, model bundle, or tuning parameters were replaced. Committed files are reports and bounded diagnostic evidence only; heavy parquet checkpoints stay in the ignored `run_diagnostics/` directory.

## Scope and reproduction context

- Code: `bce1c4c35c9fdf64b119558cc1f17cd5c8dc028b`, source feature contract `nba-prod-v2.7-input-causal-parity`.
- Audit window: 2024-01-01 through 2026-10-25 (cache-complete; the production default window is today-minus-900-days = 2024-04-21 through 2026-10-08, i.e. the audit denominators below include ~750 more January–April 2024 warmup games than a default run would ingest; validation geometry is identical either way).
- The audit rebuilt from production interfaces in cache-only mode plus **bounded fresh upstream probes** (five ESPN scoreboard days and one LeagueGameLog day fetched live and compared row-for-row against cache). It did not run the full pipeline, which trains, writes delivery artifacts, and syncs.
- Facts rebuilt: 3,517 schedule games, 6,922 team-game rows, 74,449 player lines, 6,922 event rollup rows, 1,372,130 play-by-play actions in cache. History frame: 3,461 trainable games (2023-24 through 2025-26); slate: 43 pending games (2026-10-20 onward).
- Full historical RAPM rebuild: 294,580 player-date ratings, 7,018 projected lineup team-games, 37,211 PIT designation rows applied. Base rebuild 20.4 s; rating build 79 s in a single-threaded-BLAS audit environment (production config unchanged; an earlier multi-threaded attempt was canceled before its first checkpoint and is recorded as a limitation, not a speed measurement).
- 71 served features and 113 known served/candidate columns audited across: full history, core, regular season, postseason, every season, first-10/30-days-of-season, post-30-days, each team, and the pending slate.

## Prioritized findings

### Medium-High — center lineup segment has real, persistent holes

`pl_rapm_c_*` coverage is the only served family below full coverage:

| Slice | pl_rapm_c_diff | pl_rapm_f_diff | pl_rapm_g_diff |
|---|---:|---:|---:|
| Core (2,629 games) | 91.71% | 97.64% | 97.68% |
| After first 30 days of season | 94.67% | 100% | 100% |
| First 10 days of season | 43.84% | — | — |
| Pending slate | 100% | 100% | 100% |

- The opener hypothesis explains the early-season ramp (43.84% at day 10) but **not** the post-day-30 residual: 5.33% of mature-season team-games still lack a center diff, and the hole is team-dependent — WAS 68.7%, GSW 73.1%, CHA 85.2% across their full slices.
- Mechanism: the eligible pool gates (prior-minutes floor of 50, 30-day appearance recency) applied to the full-feed position listing can leave a team with no eligible centre-listed member, and an empty segment is NaN (correctly refused, never zero-filled). This is source/pool-supported missingness, not structural-by-policy; it should not be monitored as structural without an eligibility mask.
- F/G segments are fine outside openers. Only `is_home` is constant among served core features (the intentional anchor).

### Medium — slate availability is archive-only before filings exist

The PIT designation archive covers 2024-01-01 through 2026-06-13 (the last decided game). The 43 pending games (2026-10-20 through 2026-10-25) have **no game-day filings yet**, because the league publishes them on game day. The slate's `pl_rapm_*` therefore carries **no injury removals at all** — its 100% coverage must not be read as availability-aware.

- The production design handles this: `_slate_designations` re-resolves each pending date through the same filing resolver at serve time and unions live rows over the archive. The audit did not run that resolver for future dates (no filings exist to parse; a pre-submission run is explicitly a documented fallback).
- Archive provenance is sound where it exists: 37,179 `game_day_submission` rows, 32 explicitly flagged `pre_window_fallback`, and **zero rows with a status published at or after tipoff**. 1,385 rows are empty-status sentinels with no publication timestamp (report-not-available, not "available").
- Designation game coverage: 2023-24 832/832, 2024-25 1,313/1,314, 2025-26 1,313/1,315. The 3 unsupported games are listed in the evidence.
- The removal key reads the submission-cutoff `status`; a separate `status_tipoff` field differs on 9,913 rows (late upgrades/downgrades). Using the earlier state is PIT-safe (no leak) but can leave late scratches in pools — an accuracy nuance worth an explicit rule, not a causal violation.
- Name identity: 24,671 of 26,189 removing keys match some player line's name in the log. Unmatched names are typically two-way/non-participating players the report lists; this is identity evidence only, not target-date eligibility.

### Medium — event features mix local PBP sweep with the shipped rollup archive

Raw play-by-play cache covers 2,754 game IDs; the event rollup frame covers all 3,461 trainable games because `load_ingested` absorbed the **committed `nba_event_rollups.parquet` archive for 1,414 team-games (707 games)** the local sweep didn't re-measure. This is the designed provenance split (ephemeral-host continuity), and the rollups are internally consistent (below), but the manifest should say which team-games came from which path. The delivered run reports 1,712,728 PBP actions vs the audit cache's 1,372,130 — the two acquisition paths hold different action slices of the same events.

### Medium — delivered artifacts predate the current contract

`nba_feature_v1_20261008.json` declares `nba-prod-v2.6-rapm`; current source is `nba-prod-v2.7-input-causal-parity`. Fold geometry parity is good (delivered: 56 folds / 2,400 validation games; audit rebuild of the current builder: 56 folds / 2,400 validation games / 2024: 1,085 + 2025: 1,315), but the shipped model and metrics were produced under the older contract. Rebuild features, OOF, and final models together before attributing current-code behavior to delivered forecasts.

### Low — classification and naming notes

- Two finished games have no player lines and are excluded from training by the documented rule: the **2024-12-17 and 2025-12-16 NBA Cup finals**, confirmed absent from the raw LeagueGameLog regular-season source itself (0 rows for `401734908` and `401809839`). Postponed games are excluded from the slate by the postponed rule. This is correct behavior, not a join failure — but the two finals remain unmodeled games.
- `team_stats.ast_per_game` is an all-NaN source column (a game log has no per-game assist column). `_attach_stats` overwrites it with the player-summed `ast` alias before use, so served `ewm_ast_per_game_*` is populated; the dead source column is a naming artifact worth dropping or renaming at the schema layer.
- 14 of 37,211 designation rows carry a report tipoff that differs from the schedule's actual tipoff by up to 1 hour (delays/moves, e.g. LAC@MIA 2024-02-04). Each row's own tipoff is what the PIT check uses; the mismatches are enumerated in evidence.

## Passing coverage and invariants

- **Source reconciliation is complete**: all 6,922 settled team-games have exactly one team-stats row, one event rollup row, and player lines; the season log has zero unmatched game IDs against the schedule. No duplicate games, team rows, or player-games anywhere.
- **Event-vs-box cross-check** (the pipeline's advisory two-source check) over 6,922 team-games: fga, fgm, fg3a, fg3m, fta, ftm, tov, points **exact on all rows**; oreb 6,761 exact (max diff 5), dreb 6,470 (max 10), pf 6,387 (max 3), assists 6,566 (max 22, concentrated in Denver games). Small residual rebound/assist divergence between narration and box is expected per the documented policy; the counts and worst offenders are recorded.
- **Box-derived semantics match independent recomputation exactly** (0 differing rows of 6,922; max error 2.8e-14) for efg_pct, turnover_margin, rebound_margin, off_rating, def_rating, and pace — once computed under the code's own conventions (symmetric box-score possessions averaged across both teams; pace = 48·possessions/(minutes/5), overtime-safe). An initial naive comparator (own-side possessions, no duration normalization) produced spurious mismatches and was corrected; the convention is deliberate and documented in `nba_sources.team_stats_from_log`.
- **Point-in-time discipline** of the rating build: ratings are built per target date over strictly-prior games; 17,874 rating rows were removed by PIT designations with the share guard intact; no duplicate (target, player) ratings. Position joins are complete: 74,449/74,449 player-games carry a position (2023-24 17,767; 2024-25 28,110; 2025-26 28,572).
- **Feature frame**: all 113 known columns exist; no all-null known column in history, core, or slate; zero infinities; zero home-minus-away coherence mismatches across every tested diff/side triple. Linear view width 31, tree view 73 (incl. two categorical IDs).
- **Schedule metadata** (`start_time_utc`, `venue`, `game_state`) complete on all 3,517 rows; ESPN tipoffs are genuine UTC instants (unlike the NFL board writer's ET-labeled UTC).
- **Cache inventory** is complete for the window (1,029/1,029 schedule days, 34/34 season-log units), and bounded fresh probes matched cache exactly: 3 ESPN game days with games (2/1/3 games, identical IDs and values), 2 genuinely empty days, and a 40-row LeagueGameLog day with zero value mismatches on points/minutes/fga/fgm/tov/reb. This validates the cached data at sampled points; it is **not** a full-window freshness proof.

## Verification

- Full NBA backend test suite under `PYTHONUTF8=1`: **569 passed** (54.6 s, single-threaded BLAS). Warnings only (sklearn `penalty` deprecation, pandas `Timestamp.utcnow` deprecation).
- The same suite under Windows' default codepage: 568 passed, **1 failed** — `test_run_log_tee.py::test_torn_plain_line_waits_for_its_terminator`, whose fixture writes a `✅` emoji to a file opened without an encoding. Under cp1252-style codepages the character cannot be written and the assertion sees an empty file. The production tee opens its log with `encoding="utf-8"` and is unaffected; the same test file passes 24/24 under `PYTHONUTF8=1`. Reported as an environment/encoding limitation, not silently passed; no test was weakened or skipped.
- Real interfaces exercised: `ingestion.load_ingested`, `eligible_games`, `trainable_games`, `features.build_game_features`/`build_slate_features`, `player_rapm.build_player_rapm`, `lineup_projection.load_designations`/`apply_pit_designations`/`projected_lineup`/`attach_position_rapm`, `nba_sources.cross_check`, `folds.make_folds`.
- No full estimator refit, no calibration/OOF scoring under v2.7, no live injury-PDF parse, and no frontend/browser validation were run. No static typechecker is installed/configured.

## Evidence

Curated committed evidence lives in [nba_coverage_audit_20261008](nba_coverage_audit_20261008/):

- [Audit results](nba_coverage_audit_20261008/audit_results.json) — source reconciliation, semantics, provenance, fold geometry, coverage summaries
- [Snapshot manifest](nba_coverage_audit_20261008/audit_manifest.json)
- [All feature coverage slices](nba_coverage_audit_20261008/feature_coverage.csv)
- [Event-vs-box cross-check](nba_coverage_audit_20261008/event_box_crosscheck.json)
- [Source team-game reconciliation](nba_coverage_audit_20261008/source_reconciliation.csv)
- [Side/difference coherence](nba_coverage_audit_20261008/side_diff_coherence.csv)
- [Fold geometry](nba_coverage_audit_20261008/fold_geometry.csv)
- [Injury game coverage](nba_coverage_audit_20261008/injury_game_coverage.csv) and [tipoff mismatches](nba_coverage_audit_20261008/injury_tipoff_mismatches.csv)
- [Games without player lines](nba_coverage_audit_20261008/schedule_without_player_lines.csv)
- [Upstream fresh probes](nba_coverage_audit_20261008/upstream_samples.json)
- [RAPM rebuild inventory](nba_coverage_audit_20261008/rapm_inventory.json)
- [Schedule cache inventory](nba_coverage_audit_20261008/schedule_cache_inventory.csv) and [season-log cache inventory](nba_coverage_audit_20261008/season_log_cache_inventory.csv)

Large raw/feature/RAPM parquet checkpoints remain local under the ignored `nba-backend/run_diagnostics/coverage_audit_20261008/` directory and are not committed.
