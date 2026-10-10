# MLB run-log review — 2026-10-10

**Subject:** `mlb-backend/data_delivery/mlb_pipeline_run_log.txt` as delivered on
remote — the **crash delivery** (145 lines, PHASE 1 clean through Phase 2-3, run
2026-10-10 01:29, log commit `6e0447bd`).

**Verdict:** one root cause, one token of fix, and an audit that closes every
other line of the log. The crash is Scenario C's own `bullpen_season` rewrite
(`65a200b7`, 2026-10-09 19:42): the statement body was changed to interpolate
`{_xfip_sql(...)}` helper calls but the `con.execute` at
[features.py:3041](../mlb-backend/backend/features.py#L3041) kept a **plain**
triple-quoted string, so DuckDB parsed the rendered call's literal `"("` →
`ParserException: syntax error at or near "("`, PHASE 2-3 died, and every
statement after line 3041 — the whole rest of `_build_game_level`, including
every downstream Scenario C hunk — never executed on the remote run. The fix is
the missing `f` prefix (`con.execute(f"""…`) and nothing else: a one-token
change to one line, verified by running the **full production `build_features`
end-to-end on real Statcast data**, by a static parse audit of all 129
`con.execute` sites, by the full MLB backend suite, and by a new AST pin that
fires on exactly the crashing line.

Guardrails honored: **no change to the MLB Kaggle run** (the notebook is
Kaggle-owned, T7 of the 2026-10-06 review — untouched; the self-healing
`MLB_END_DATE` pin warning is therefore documented, not suppressed), **no new
programs committed** (one production line edited, one pin appended to the
existing [test_log_review_20261008.py](../mlb-backend/backend/test_log_review_20261008.py),
this document is non-executable; the audit/rebuild harnesses live under
gitignored scratch directories), and **the Kaggle run's code path, feature
width and hyperparameters are unchanged**.

## Root cause

```
File ".../mlb-backend/backend/features.py", line 3041, in _build_game_level
    con.execute("""
duckdb.duckdb.ParserException: Parser Error: syntax error at or near "("
```

Scenario C replaced the bullpen **WHIP** family with **K-BB%** and bullpen
**ERA** with **xFIP**, and the new values are built inside `bullpen_season`
with helper-interpolated expressions (`{_xfip_sql("w.ks", …)}`). In commit
`65a200b7` the surrounding `CREATE TABLE bullpen_season AS …` statement was
written back with a **plain** `"""` string: at runtime the braces render as
literal text, DuckDB sees `… (CASE WHEN w.ip > 0 THEN (13.0 * (` and hits the
call's own `"("` as bare syntax — exactly the reported position. Every other
rewritten statement in the commit correctly uses `f"""`; only this one lost the
prefix.

Fix (the entire production diff):

```python
-    con.execute("""
+    con.execute(f"""
         CREATE TABLE bullpen_season AS
```

## Why the suite could not catch it (the escaped-defect class)

* `test_bullpen_availability` **does** execute `bullpen_season` via its
  `lift()` helper (line ~742) — but `lift()` **regex-renders the
  `{_xfip_sql(...)}` placeholders itself** before running the SQL. The test
  therefore simulates the render production never performs, and passes with or
  without the `f` prefix. This is the same class as the 2026-10-08
  `lift()`-masked checks: a helper that re-implements the call site's contract
  cannot see the call site breaking it.
* `test_run_engine_contract` calls the real `build_features` but expects it to
  fail **early** on its fixture; the pipeline died deep inside `_build_game_level`
  instead — a failure mode outside that test's assertion.
* No committed test executes `_build_game_level`'s own `con.execute` chain
  end-to-end on a real frame, which is why "backend 385 passed" shipped
  alongside a run that crashed at runtime.

**T14 pin (this review, appended to the existing log-review test file):**
walk `features.py`'s AST; any SQL **string literal** handed to `execute()`
that still contains `{` is this defect (f-strings parse as `JoinedStr` and can
never appear as a `Constant`). Verified both directions: passes on the fixed
tree (35 passed, 1 skipped), and **fires on exactly line 3041** when run
against `git show 65a200b7:mlb-backend/backend/features.py`.

## Log line review

| Log line(s) | Finding | Action |
|---|---|---|
| `WARNING MLB_END_DATE=2026-10-09 is stale … extended to 2026-10-10` | Self-healing by design: the guard extends the window so the daily slate cannot freeze and tells the operator to re-date the notebook pin. The pin lives in the **Kaggle notebook** — Kaggle-owned guardrail (T7 2026-10-06), explicitly out of scope this run | Documented, no code change (pinned by `test_log_review_20261006`) |
| `Chunk … attempt 1/3 failed: Error tokenizing data…` / `Connection broken: IncompleteRead…` | Two transient network failures; **both recovered on attempt 2/3** with full pitch counts (164,216 and 141,645). The retry/backoff machinery did its job | None — healthy |
| `missing-finals guard: frame horizon 2026-10-08; no official finals beyond it` | T1 guard from the 10-08 review reporting a clean tail | None |
| `on-IL while batting: 0 player-games (3420 before reconciliation, budget 250)` | Reconciliation working as designed | None |
| `ParserException … PIPELINE CRASH 2026-10-10 01:29:55` | **The defect — fixed (above)** | Fixed + pinned (T14) |
| `UserWarning: To exit: use 'exit'…` after `SystemExit` | Pure crash artifact: `master_pipeline` raises `SystemExit` only in its honest-failure block (`_phase4_error`); a green run never triggers it | Resolves itself with the fix |
| Everything after `Bullpen shrinkage: k = 93.3 pitches…` | **Never ran remotely** — hence no 20261010 artifacts, no new coverage CSV, no drift table | Covered by the end-to-end local execution below |

## Verification: the crashed path now executes, on real data

The strongest possible smoke of the exact statement that died: run the
**production** `build_features(..., validate=True)` locally against a real
Statcast extract (2,427,076 pitches), then the production decided-frame
sequence (`enrich_elo_and_records` → `add_diff_features` →
`add_exp2_features`), on an R-scope mirror of remote ingestion (S/E dropped —
identical scope rule to the run log's `Dropped 117501 … (game_type S/E)`):

| Check | Result |
|---|---|
| Full build incl. line 3041 and everything downstream | **GAME_DF (8333, 201)** all game types; R-scope rebuild **(7073, 290)** after the production enrich/diff/exp2 sequence, horizon 2024-03-20 → 2026-09-04 (remote: 2024-03-20 → 2026-10-08), `validate=True` passed |
| `bullpen_season` (the crashing statement) | Parses **and executes**; earlier hand-verification of its shrink math matches to 1e-9; season opener stays NULL by design (LAG-first, no same-day self-inclusion) |
| Static SQL audit of `features.py` | All 129 `con.execute` sites checked: **zero** non-f-string blocks carrying placeholders; all helper-interpolated statements (`_xfip_sql`, exp2 SP-category) render and parse against DuckDB |
| MLB backend suite | **385 passed, 1 skipped** (unchanged from the crash commit's claim — the fix breaks nothing), plus the new T14 pin |
| Monitor smoke (`test_monitor_smoke.py`, all four sports) | Passes |

## Full feature audit — coverage and accuracy against real game data

### Serving contract

* `active_moneyline_feature_cols()` = **109**, every column present in the
  built frame, **zero** legacy names served (`sp_era_*`, `bullpen_whip_*` all
  gone), **zero** served columns all-NULL.
* The 12 Scenario C renames: **9 served** + 3 intentionally unserved — the
  plain `sp_xfip_home/away/diff` trio sits in `_PL_PLAN_REMOVALS` (the
  2026-10-03 pl-plan removed the plain ERA trio from serving; it remains
  generated and RFE-triable), matching the pre-rename contract exactly.
* New-column checks on the real frame: all 12 generated with **86.7–99.3%**
  non-null coverage (SP rows gated by the documented debut/staleness policy);
  all four `*_diff` identities hold with **max residual 0.0**
  (`diff == home − away` exactly); xFIP range `[-1.5, 16.7]` (37 tail values
  on tiny-IP windows — the same small-window behavior the replaced ERA family
  had); K-BB% range `[-0.097, 0.364]`, zero out-of-`[-1,1]`.
* The new estimators are **genuinely re-estimated, not renamed copies**: vs
  the delivered pre-C frame, `sp_xfip_home` vs `sp_era_home` corr 0.50 /
  `sp_xfip_5g` 0.50 / `sp_xfip_away` 0.41 / `bullpen_kbb_10g` vs
  `bullpen_whip_10g` −0.57 (direction: higher K-BB% ↔ lower WHIP ✓), with
  **0.0%** exact-equal rows in every pair.

### Coverage (production windows and formula)

`compute_feature_coverage` over the serving width with the production window
construction (`cutoff = target − 7d`, `baseline = prior.tail(max(3·|current|, 250))`):

* **214 OK + 4 non-OK** — the 4 being exactly the two weather features × two
  windows, whose local STARVED status is a **harness limitation**: the per-game
  weather cache (`weather_history.parquet`) lives only in the remote run cache
  and cannot be reproduced offline, so no observations attach locally. The
  committed 20261009 artifact (weather attached) reports the same 214 OK with
  those 4 rows as **STRUCTURAL** (closed-roof policy) — 2026-10-09 review
  verified that family directly.
* Every non-weather column: **OK**, including all Scenario C renames.

### Parity vs the delivered remote frame (three-frame attribution)

Local post-C+fix frame vs the committed remote `game_level_features.csv`
(pre-C, last successful run 2026-10-09 17:15) over **7,073 shared games**
(identical game sets within the shared range — zero missing either way):

1. **Official-result columns are bit-exact:** `home_win`, `home_score`,
   `away_score`, `total_runs`, `rest_days_home`, `home_starter_id`,
   `time_zones_crossed_last_3d_home` all **7073/7073 exact**.
2. **Elo is bit-identical under equal iteration order:** replaying
   `compute_elo_entries` on both frames with the same tie-break gives
   `max|Δ| = 0.000000000`. The stored differences (median 0.041, max 15.2)
   trace to a single mechanism, reproduced exactly: the earliest divergence is
   the **2024-04-04 NYM/DET doubleheader**, where each frame's stable
   date-sort preserves its own within-date row order, and the two builds'
   export orders place the twin bill in opposite order. Replaying the remote
   CSV **in its own file order reproduces its stored Elo to 0.000000** —
   remote Elo is self-consistent; only the tie order differs. This is a
   pre-existing property of `enrich_elo_and_records` (date-stable sort over
   the build's row order), untouched by Scenario C (`data_ingestion.py`'s
   commit changes don't alter the Elo/record math — proven by the identical
   replay). Season records (`home_wins`/`home_losses`, 40/28 rows) differ by
   the same doubleheader tie mechanism.
3. **Scenario C's blast radius is exactly what it declared:** a pre-C
   rebuild (worktree at `65a200b7^` = `eb93d2ce`, whose MLB backend is
   value-identical to the code the remote run used — only label-only
   `feature_metadata.py` differs) vs the post-C+fix frame over the same
   inputs gives **263 of 269 shared columns bit-exact**, and the only 6
   differing columns are the **three re-pointed composites and their
   home/away/diff variants** — `pitcher_regression_indicator_*`,
   `bullpen_meltdown_risk_*` — which Scenario C explicitly re-points from the
   ERA/WHIP embeddings to xFIP/K-BB (`wind_advantage_flyball_factor` is
   equally re-pointed but carries no local weather observations to differ
   with). Nothing else moved: **the fix's only effect is executing the
   statement that used to crash.**
4. **Residual local-vs-remote deltas in pitch-fed aggregates are
   extract-level, not code:** in the pre-C-vs-remote comparison (identical
   code era) the remaining differences are confined to pitch-derived
   aggregates at vanishing scale — league rates ≤ 2.7e-4, position-pool xwOBA
   ≤ 0.015, one exit-velo outlier 2.29 on a single game — while official-fed
   columns stay exact. Two **independent local pulls** of the same games are
   value-identical (per-game pitch counts 2,078,494/2,078,494 with zero
   differing games; `pitch_type`, `woba_value`, `events` equal; only
   float32-epsilon noise in `release_speed`), and the pre-C rebuild proves the
   code path identical — so the residual tracks the remote run's own
   2026-10-10 `MLB_FULL_REPULL` extract differing at pitch level from the
   local extracts. It is a data-vintage delta bounded to ≤0.1% of scale, not a
   code defect, and it does not touch serving coverage or column contracts.

### Deliberately not changed

* **The Kaggle notebook / `MLB_END_DATE` pin** — Kaggle-owned (T7); the
  warning is the documented self-healing behavior.
* **Coverage status thresholds and denominators** — remain an owner decision
  (2026-10-09 review, T9).
* **Model, blend, hyperparameters, feature-set version, filenames** — the
  Scenario C adoption itself was "per owner direction" against the A/B
  harness's `KEEP A` verdict ([mlb_xfip_ab_20261009.md](mlb_xfip_ab_20261009.md));
  recorded, not reversed here.
* **Order-tie sensitivity of Elo/records** — documented above with a exact
  reproduction; changing the sort key would alter every shipped Elo value and
  is an owner decision, not a remediation.

## Residual risk

The remote run remains the authority for the next delivery: this fix lets
PHASE 2-3 proceed, after which the 20261010 artifacts, the fresh
`feature_coverage_20261010.csv` (expected: 214 OK + 4 weather STRUCTURAL with
weather attached) and the drift table will be written normally. If the next
remote delivery shows the weather rows as anything other than STRUCTURAL/OK,
or any `MISSING_COLUMN`, that is a new defect — this review's audit bounds
everything else.

---

## Second delivery — the 05:30 run on remote (reviewed the same day)

**Subject:** `mlb-backend/data_delivery/mlb_pipeline_run_log.txt` as delivered
on remote — the **successful re-run** (452 lines, six phases, run 2026-10-10
05:29 → 05:30, commits `dbb6e5b5` features + `c3bbe223` stale cleanup +
`b2edfb29` final log).

**Verdict:** the first delivery's fix worked — PHASE 2-3 now executes
`bullpen_season` and the whole pipeline runs to `ok`, pushing 15 artifacts,
staging 28 files and removing 10 stale ones, every claim confirmed against the
remote commits. The run is clean; **two reporting defects in it are not**, and
both are remediated below. Nothing in the log required touching a model,
threshold, the notebook or a delivered value.

### Claims reconciliation — every count in the log against the delivered artifacts

| Log claim | Artifact check | Verdict |
|---|---|---|
| `Total raw pitches 2292817`, `Dropped 117501 (S/E)`, `Saved 2175316` | `pbp_defense_20261010.parquet` = 2,175,316 rows | ✅ |
| `game_level built: 7402 games`, `✅ Game: (7402, 201)` → 290 cols at the Phase-4 load | delivered frame 7402 rows × 297 cols (the 290/293 counts predate Step 1's weather refresh, which re-exported the CSV) | ✅ |
| `Feature width: RFE subset (109 cols)`, `Applied adopted feature subset: 109/275` | `mlb_feature_selection_state.json` `n_cols=109`, `len(cols)=109` | ✅ |
| `Feature metadata: 109 features written` | `features_metadata_20261010.json` 109 entries, 0 warnings | ✅ |
| `Feature coverage: all 218 feature-window pairs OK (4 STRUCTURAL …)` | CSV: **214 OK + 4 STRUCTURAL** | ❌ **Finding 2** |
| `Feature drift [moneyline]: 109 features, 0 warnings, 0 alerts, 0 seasonal` | CSV: **109/109 INSUFFICIENT** (n_current 14–15) | ❌ **Finding 1** |
| `model_monitor` `drift_summary` warnings 0 / alerts 0 | same 109 INSUFFICIENT rows | ❌ ships with Finding 1 |
| `Run engine OOF: 7031 rows`, `Prediction history written: 7031 games` | both CSVs 7031 rows | ✅ |
| `Run engine markets: 7032 rows` | 7031 `kind=oof` + 1 `kind=slate` (2026-10-10) | ✅ |
| `Rolling Brier: 511 points … 7031 games … mean 0.2445` | `rolling_brier_20261010.json` 511 / 7031 / 0.244496 | ✅ |
| `Version history: 20 versions recorded (v2026.10.10 latest)` | `model_version_history.json` 20 entries, last `v2026.10.10` | ✅ |
| `Upcoming slate built: 1 games`, `SHAP … over 1 games` | `todays_games_20261010.csv` 1 row, `shap_game_…CHW@CLE.csv` | ✅ |
| `Staging 28 files` → `Pushed … 28 files` | `dbb6e5b5` commits **26** (2 staged files — `run_engine_totals_history.csv`, `umpire_stats.csv` — were already identical), `c3bbe223` deletes the **10** named stale files | ✅ (wording: 28 staged, 26 changed) |
| `Final run-log delivery … synced clone to remote tip c3bbe223d5` | `b2edfb29` touches only the log | ✅ log is complete, self-describing |
| Metrics `auc 0.573 brier 0.2445 logloss 0.6821 ece 0.0091` | `model_history.json`: prior run 0.5721 / 0.2446 / 0.6821; six-run band 0.5721–0.5746 | ✅ first Scenario C delivery, inside noise |

### Finding 1 (remediated): a monitor that measures nothing still read as clean

`Feature drift [moneyline]: 109 features, 0 warnings, 0 alerts, 0 seasonal`
(lines 352 and 356, both views) was true and vacuous: **every one of the 109
rows is INSUFFICIENT**, because the trailing 7-day current window holds 14–15
games against the 30-row judge floor (`n_b < 100 or n_c < 30` in
[explainability.py](../mlb-backend/backend/explainability.py#L660)). This is not
new — [drift_status_history.csv](mlb_log_review_20261010/drift_status_history.csv)
shows **seven consecutive deliveries (20261004 → 20261010) with 109/109
INSUFFICIENT** after 20261003's 109 OK, every one of them carrying 0 warnings
and 0 alerts in its drift summary (only the current run's log survives, so the
status column of each delivered CSV is the proof), and
`model_monitor_20261010.json` ships the same green `drift_summary`. A reader of the delivery log cannot distinguish "no drift"
from "no measurement" — the exact failure mode the monitor exists to prevent.

**Remediation (message only, no threshold and no status changed):**

* the summary line now publishes how many rows were judged —
  `109 features, 0 evaluated, 109 INSUFFICIENT, 0 warnings, 0 alerts, …` — and
* when **nothing** was evaluable it additionally emits ONE WARNING that names
  the condition: `0 of 109 features evaluable this run (… ) — drift is
  UNMEASURED, not clean`.
* `drift_summary` in the monitor JSON gains `evaluated` and `insufficient`, so
  the card can no longer show warnings 0 / alerts 0 without the context.

INSUFFICIENT rows themselves still never page (window-size statement, not a
defect); only the all-unevaluated case does. Reproduction of both the delivered
and the remediated lines from the committed frame is recorded in
[monitor_replay.txt](mlb_log_review_20261010/monitor_replay.txt) — the replayed
pre-remediation messages are identical to the delivered ones, so the defect was
in the message, not the run.

### Finding 2 (remediated): a coverage line that contradicts its own CSV

`all 218 feature-window pairs OK (4 STRUCTURAL by declared policy)` — 218
cannot be all-OK when 4 of them are STRUCTURAL; the CSV holds 214 OK + 4
STRUCTURAL. The line now prints what the artifact contains:
`218 feature-window pairs — 214 OK, 4 STRUCTURAL by declared policy, 0 alarms`
(and `218 feature-window pairs — 218 OK, 0 STRUCTURAL, 0 alarms` when there is
no structural row). The STRUCTURAL detail line is unchanged.

### Reviewed, no defect

| Log line(s) | Finding | Action |
|---|---|---|
| `WARNING SP slate staleness gate: 146 pitcher(s) … (pre-stint form)` | Population-level notice: the gate fires only for a ≥10d gap that overlaps an IL stint. Both starters of the day's one game carry non-null lines in `todays_games_20261010.csv` (`sp_xfip` 2.684 / 3.825, `sp_k9` 12.61 / 10.01) — no slot went empty | None |
| `WARNING Calibration: degenerate Platt params (a=-0.33 …)` ×3 + `322 of 6140 …` | Working as designed since the 2026-10-06 review: `a <= 0` refuses an inverting fit, the 3-line cap plus uncapped counters feed the block-delta summary (5.2% of fits, identity fallback) | None |
| `Fold 83 … auc=nan brier=0.2639 [PROVISIONAL]` | AUC is undefined on a 1-game, single-class val set; the fold is fitted+scored and excluded from grading, and the PROVISIONAL marker rides the same line | None — honest value, not an error |
| Two identical 16-line `StatsAPI schedule …` blocks | Two consumers of the schedule: the start-time refresh and the weather start-time lookup. Redundant work (~32 range fetches), no correctness impact | None (documented) |
| `MLB_END_DATE=2026-10-09 is stale …` warning **absent** | The notebook pin was re-dated/unset by the operator, exactly what the first delivery's warning asked for; the self-heal guard remains armed | Resolved upstream, guard untouched |
| `STRUCTURAL … 4 unmeasured row(s) … 63 unmeasured row(s)` (one long line) | Four reason blocks = 2 declared-policy features × 2 windows (wind ← `sp_xfip_diff`, air density ← `sp_fbvelo_diff`); counts reconcile with the CSV (3+1 of 15, 47+16 of 250) | None |
| `Diff features complete: 0 columns added` (Step 1) | Recompute over an already-enriched frame adds nothing; the build itself reported 53 | None |
| `on-IL while batting: 0 player-games`, `missing-finals guard … no official finals beyond it`, `Dome refinement … 0 UNKNOWN roof`, `Ingestion guard … expected ≥2044874 pitches / 6960 decided` | All prior-review guards reporting healthy | None |

### Verification for this section

* Both monitor summary lines were **replayed locally from the committed frame**
  with production's own window construction (`cutoff = target − 7d`, current
  15 games, baseline `prior.tail(max(3·|current|, 250))` = 250) and the active
  109-column serving list: replayed statuses match the committed CSVs exactly
  — **109/109 drift rows and 218/218 coverage rows, 0 disagreements** — and the
  replayed pre-remediation messages are identical to the delivered lines.
  Evidence: [monitor_replay.txt](mlb_log_review_20261010/monitor_replay.txt).
* Full MLB backend suite: **390 passed, 1 skipped** (386 before this review),
  including the four new pins T15/T16 in the existing
  [test_log_review_20261008.py](../mlb-backend/backend/test_log_review_20261008.py):
  an all-INSUFFICIENT drift window must disclose its counts and WARN, a
  judgeable window must stay quiet, the monitor JSON must carry `evaluated` /
  `insufficient`, and the coverage line's counts must reconcile with the CSV it
  describes. [Test output](mlb_xfip_kbb_xwoba_review_20261010/backend_tests.txt).
* Python compilation and `git diff --check` pass. Feature values, models,
  thresholds, calibration, the notebook and every delivered artifact are
  untouched; the feature-side audit of this same delivery is the
  [xFIP / K-BB% / xwOBA review](mlb_xfip_kbb_xwoba_review_20261010.md).

### Deliberately not changed

* **NFL and NHL emit the same drift summary shape**
  (`nfl-backend/backend/monitoring.py`, `nhl-backend/backend/monitoring.py`)
  and would hide the same condition; they were out of scope for an MLB log
  review and are recorded here as the follow-up.
* **Drift window policy** (trailing 7 days against a 30-row floor) stays an
  owner decision — changing it would make the monitor judge offseason windows,
  not just make the line honest.
* **The `auc=nan` token, the schedule double-fetch and the calibration
  identity fallbacks** are behavior, not defects, and are documented above.
* **The negative-xFIP tail** (9 rows, unserved trio) is a feature-value
  decision recorded in the
  [feature review](mlb_xfip_kbb_xwoba_review_20261010.md#findings), not a
  log defect.
