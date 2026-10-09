# NFL run-log review — 2026-10-09 delivery (`b333fccd`)

Scope: the remote NFL delivery committed as `b333fccd` (run 17:33–18:14 UTC,
239 artifacts, `nfl_pipeline_run_log.txt` 460 lines), reviewed line-by-line
against the delivered artifacts, plus the outstanding items of the
2026-10-08 NFL feature-coverage audit (`docs/nfl_coverage_audit_20261008.md`).
No new programs were added; every remediation lands in existing modules.

## Verdict

**The delivery is clean and complete; the run ended green with all eight
schema gates PASS.** This was the first run with `NFL_FULL_REPULL=1` (fresh
decided pool 2016-09-08..2026-10-08, 2826 games) and the first with the
second-pass coverage-audit remediation (`293269a0`) live. Two log classes
needed action, both remediated in this commit:

1. **The rest trio false-alarmed LOW_COVERAGE** — a regression in the NEW
   mask-verified structural classification, not in the data.
2. **A silent 58-game weather hole** — twelve Open-Meteo 429 ladders
   exhausted mid-repull; 97.1% coverage sat above the 80% warn line, so the
   hole printed a single INFO.

Everything else triaged clean: Open-Meteo 429 retry storms recovered (all
other windows fetched fully), the OOF fold-thin warning is the standing
provisional-window class (93 of 200 windows, retained + scored, excluded
from pooled grading), the two drift ALERTs (`inj_ol_out_home/away`) are the
documented cumulative-season class, retention pruned exactly the stale
10-day artifacts, and Phase 11/12/14 all match their artifact counts
(13 slate games, 13 SHAP cards, 2559 OOF rows, 239 artifacts).

## Finding 1 — rest_days structural proof regression (coverage audit item)

The 10-08 artifacts graded `rest_days_{home,away,diff}` STRUCTURAL
("first game of the season"); the 10-09 run graded the same features
LOW_COVERAGE 74.4% (baseline) — three spurious WARNINGs in the Phase 13
summary.

Root cause, proven against the audit's own rebuilt frame
(`run_diagnostics/coverage_audit_20261009/historical_features.parquet`,
2825 games, zero unexplained rest nulls): the hardened mask
(`_ineligible_mask`) computed each team's opener as its first appearance
**within the window slice** and **within each side separately**. Both are
wrong: a mid-season window's first appearance is usually game N>1 carrying
a measured value, and a team whose opener was AWAY is over-flagged at its
first HOME game (~176 measured rows per pool; 352 flags vs 176 true
openers). The exact-equality proof (`nulls == mask`) therefore could not
hold outside opener-dense windows — the 10-08 statuses had come from the
old pct-similarity rule, which the audit itself retired.

Remediation (`monitoring.py`):

- The rest-family mask is now **side-agnostic** (a team's first game on
  EITHER side — matching the ladder's null semantics) and
  **pool-anchored**: `coverage()` takes the canonical decided pool
  (`phase_frame`), computes the opener fact once on it, and reindexes onto
  each window. `master_pipeline.py` passes `phase_frame=game_df` to the
  moneyline coverage; the run-engine writer threads its existing
  `phase_frame` into its coverage call too.
- A fully-measured window (100%) can no longer be labeled STRUCTURAL —
  the override explains absence, never a clean rate.
- The 15-point window-stability gate and every alarm class are RETAINED:
  unexplained nulls (the settled-gate/fetcher-death class) still fail the
  mask and keep the raw threshold alarm; an unstable mask-proven pair
  still grades raw (the weather-truncation protection).

With the fix, the 10-09 windows classify STRUCTURAL again — now because
the pool's own eligibility mask proves the slice, not because two
percentages happened to agree.

## Finding 2 — silent 58-game weather hole (Open-Meteo 429s)

The full repull requested 444 PIT-weather batches (111 windows × 4). Twelve
batches (3 sub-windows × 4) exhausted their 6-retry ladders under 429s;
`PIT weather: 1964/2022 eligible games` ended the run 58 games short
(2025 weeks 6–10). The affected games never enter the weather cache, so the
next run's cache-miss set refetches exactly them — the hole is
self-healing. But 97.1% sat above the `<80%` WARNING line, so nothing in
the log flagged it: the silent-starvation class the coverage audit exists
to surface.

Remediation (`weather.py`):

- **Same-run top-up**: after the first pass over the missing set, one
  second pass refetches only the still-missing games (seconds once the
  rate limit clears), closing recoverable holes inside the delivery that
  opened them. Logged at INFO ("PIT weather top-up: N game(s) still
  missing...").
- **Any-gap visibility**: a partial hole at or above the 80% line now
  WARNINGs with the missing count, example game_ids, and the self-heal
  note ("the next run's cache-miss pass refetches only these games").

## Audit items verified shipped (no action needed)

Confirmed live in this run's log/artifacts:

- **Franchise aliases** (High): `TEAM_ALIASES` normalization at the feature
  boundary — EPA family coverage rose to 96.7–100% (audit: 95.27% QB diff
  with OAK/SD pool losses); no alias-split identities remain in the frame.
- **FTN side attribution** (High): "196044 of 196044 charted play(s) bound
  to possession identity (0 without a PBP key)".
- **NGS Super Bowl weeks** (Medium): "NGS Super Bowl week reconciled: 20
  row(s) moved to the schedule's week number".
- **Board kickoff UTC** (High): `_start_time_utc` now converts
  America/New_York properly; the delivered `nfl_board_20261011.csv` carries
  correct UTC stamps (London game 13:30Z, 1 PM ET games 17:00Z).
- **Postseason in prediction history** (Medium): the history now carries
  the 111 postseason OOF rows (2559 total; audit: 2447 REG-only).
- **Availability provenance** (High): PIT designation rows 49445 ingested
  with the unknown-state handling from `293269a0`; no zero-default
  health claims in the log.

Remaining audit items deliberately NOT touched here (model/definition
work, out of scope for a log review): PBP pace/yards-play play-population
semantics, the ties/moneyline policy decision, venue-timeline unknowns
(35-game 2023 surface cluster), and the README/manifest population drift.

## Verification

- NFL production test script: **521 passed, 0 failed** (10 new checks:
  pool-anchored STRUCTURAL regression, standalone refusal, unexplained-null
  alarm, stability-gate retention, top-up recovery, any-gap WARNING,
  top-up INFO, and three source pins on the phase_frame plumbing).
- Backend pytest: **34 passed**.
- Both CSVs re-read directly: the 10-09 rest rows and their 10-08
  STRUCTURAL predecessors were compared row-by-row from the committed
  artifacts, and the mask arithmetic was reproduced against the audit's
  historical frame before the fix was written.
- `git diff --check` clean; no new programs — edits only in
  `monitoring.py`, `master_pipeline.py`, `weather.py`,
  `test_production.py`, and this doc.
