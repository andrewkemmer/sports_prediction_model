# NBA run-log review — 2026-10-09 delivery (`fb799ece`/`4d9a2f83`)

Scope: the remote NBA delivery (run 17:34–18:32 UTC, status ok in 3487s,
`nba_pipeline_run_log.txt` 253 lines), reviewed line-by-line against the
delivered artifacts, plus the outstanding items of the 2026-10-08 NBA
feature-coverage audit (`docs/nba_coverage_audit_20261008.md`). No new
programs were added.

## Verdict

**The delivery is clean end-to-end — all ten phases green — and it is the
FIRST run shipping the NBA coverage-audit remediation (`62afdd8c`,
pool-eligibility mask, honest source schema, event-rollup provenance).**
Every audit item of Medium or higher severity was verified shipped and
live in this run's artifacts. One log-readability defect surfaced and was
fixed; no data or model defects were found.

Run facts: 3488 schedule games (2024-01-01..2026-10-21, full repull,
1025/1025 days fetched), 3461 trainable games × 71 features, 56 folds /
2400 OOF rows (blend: elasticnet 74.2%, lightgbm 25.8%, xgboost 0.0%),
identity calibrator gated no-gain, 14 slate games scored, 23 artifacts,
rolling Brier 0.2090 over 388 days.

## WARNING-by-WARNING triage

- **All-Star weekend drops** (EAST/WEST 2024, CHK/KEN/SHQ/CAN 2025,
  STARS/WORLD/STRIPES 2026): non-franchise exhibitions with no player
  lines — the documented drop-with-reason class. Benign.
- **27 games with no season-log match / no NBA game id**: 16 are
  2026-10-20/21 games the 2026-27 season log has not published yet
  (pre-season, expected); 11 are postponed games explicitly excluded from
  the slate; the 2 NBA Cup finals (2024-12-17, 2025-12-16) are the
  audit-confirmed unmodeled games absent from the LeagueGameLog source
  itself. All named with reasons in the log. Benign/documented.
- **Position index covers 24%/23% of 2023-24/2024-25 players**: the
  full-feed position listing exists only for 2025-26 (100%); the older
  seasons honestly keep the G/F/C convention under the declared floor.
  Consistent with the audit's "position joins complete" finding.
- **"no filing lists DAL@GSW before tipoff on 2024-01-19: game
  uncovered"**: one of the audit's three designation-unsupported games
  (itself a postponed game). Benign.
- **OOF thin on 16 of 56 windows**: the standing provisional-window class
  (postseason weeks and season ramps), retained + scored, excluded from
  pooled grading. Benign.
- **pl_rapm coverage gap (189/3461: 104 within lookback, 18 within
  recency, 67 beyond)**: the audit's Medium-High center-hole finding, now
  with per-row accounting — see below.
- **10 drift alerts** (elo_home/away, ewm_*, pl_rapm_{c,f,g}_*): the
  current window is the Finals (100% playoffs), composition-matched to 220
  prior playoff rows so the comparison measures drift, not the calendar.
  Playoff-tightened rotations and rating shifts are real movement; the
  alerts are the monitor doing its job.

## Audit items — verification of the shipped remediation

Each finding of the 10-08 audit, verified against this run's log and
artifacts:

- **Center-lineup holes (Medium-High)** → the `_eligible_pl_rapm_*`
  refusal mask ships: `lineup_projection` stamps per-position membership
  flags (empty segment → 0, membership → 1 even when the blend refuses,
  unattributable frame → NaN so a real hole can never be silenced), and
  `monitoring.coverage` classifies STRUCTURAL only when EVERY null on the
  row carries a refusal. The delivered coverage CSV confirms the
  `n_pool_refused` column live with all rows OK on the late-season
  windows; the early-season day-10 windows (the 43.8% case) will now
  classify as pool refusals rather than starved features. Verified
  mechanism against the audit's `center_gap_diagnosis.csv` (e.g. 2024-01-01
  NYK: 2 center-listed members, 0 passing the minutes/recency gates).
- **Slate availability archive-only (Medium)** → the live resolver ships:
  "slate designations resolved live: 122 record(s) across 7 date(s)", with
  per-game pre-submission fallbacks logged against each filing cutoff.
- **Event-rollup provenance split (Medium)** → `_absorb_event_rollup_archive`
  returns the absorbed `game_id|team` keys and the run logs the count
  (silently correct here: the full repull's own sweep covered all 3461
  games, so zero rows were absorbed).
- **Artifacts predate the contract (Medium)** → resolved by the run itself:
  `nba_feature_v1_20261009.json` declares `nba-prod-v2.7-input-causal-parity`.
- **Low items**: the dead `team_stats.ast_per_game` source column is
  overwritten by the player-summed `ast` alias in `_attach_stats` before
  any reader sees it, and the manifest documents `ewm_ast_per_game_*` as
  AST-derived — behavior is correct; renaming the schema column would risk
  delivered-artifact consumers for zero behavioral gain, so it is
  recorded as accepted. The 14 designation tipoff mismatches keep the
  audit's documented policy (each row's own tipoff drives the PIT check).
  The two NBA Cup finals remain the documented unmodeled games.

## Remediation in this commit

**The "projected lineups" inventory line was unreadable after the mask
remediation.** Every refused team-game now emits a `pool_size-0` mask row,
so the whole-frame means behind the line collapsed from the pre-mask
"22 team-game(s), mean pool 19.5" to "6976 team-game(s), mean pool 0.1" —
a healthy run's own log reading like a projection collapse. The line now
splits priced vs pool-refusal rows and reports the means over priced rows
only, e.g. `projected lineups: 6976 team-game(s) (11 priced, 6965
pool-refusal mask rows), mean pool 19.5 / healthy 18.0 over priced rows`.
A source pin keeps the split from regressing.

## Verification

- NBA backend suite under `PYTHONUTF8=1` (from `nba-backend/backend`):
  **584 passed** (583 + the new pin; the audit's run had 569).
- The remediation's own mask tests (`test_nba_injury_lineup.py`,
  `test_nba_rapm_features.py`, additions from `62afdd8c`) pass in the same
  run.
- Delivered artifacts re-read directly for this review: the run-engine
  coverage CSV (all 140 rows OK, `n_pool_refused` populated), the feature
  contract version, the log's designation/absorption/projection lines,
  and the 10-08 run's projection line for the before/after comparison.
- `git diff --check` clean; no new programs — the only code edits are the
  inventory log line in `master_pipeline.py` and its test pin.
