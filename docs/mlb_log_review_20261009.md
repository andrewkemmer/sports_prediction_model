# MLB run-log review — 2026-10-09

**Subject:** `mlb-backend/data_delivery/mlb_pipeline_run_log.txt` as delivered on
remote — **first delivery** (438 lines, run trained 2026-10-09 05:27, artifacts commit `09705665`,
log commit `64b6f899`). The **second delivery** of the same day (431 lines, trained 17:15,
artifacts `2cc8f88f`, log `34b6cddc`) is reviewed at the end of this document.
log commit `64b6f899`, cleanup `7b7cb6d1`, previous HEAD `871c45eb`).
**Verdict:** delivery, the missing-finals guard and the off-day path are all
clean. The run's only WARNING class left unexplained was
`Feature coverage gaps [moneyline]/[run-engine]` on the two weather features —
and it is NOT a data outage: it is two real defects (an exact-midnight
point-in-time sampling hole and a partial cache record that outranked the
complete one) plus a denominator that can never clear the 80% line. Both
defects are root-caused against the frame and the emitters and remediated in
this commit; every gap from the documented 2026-10-08 feature-discrepancy
audit is now either verified-fixed or fixed here. One further defect surfaced
while verifying against the delivered artifacts: the committed
`test_mlb_board_render_smoke.py` fails **on the honest off-day board**
(scenario A expects the newest valid date to be renderable — it ships 0
rows), reproduced unchanged at `54c85d54` and fixed here as **T10**.

Guardrail honored: **no new programs committed** — three production modules
edited ([weather.py](../mlb-backend/backend/weather.py),
[master_pipeline.py](../mlb-backend/backend/master_pipeline.py),
[explainability.py](../mlb-backend/backend/explainability.py)), regression
pins added to the existing
[test_log_review_20261008.py](../mlb-backend/backend/test_log_review_20261008.py),
scratch harnesses kept under `.freebuff/` (uncommitted), this document is
non-executable.

## What was validated (clean)

| Log claim | Remote/artifact reality |
|---|---|
| `📁 Artifacts: 15 files` + Phase 5 `✅ Pushed and remotely verified 26 files` | `git ls-tree origin/main` carries **17** `*20261009*` paths (the 15 reported artifacts + `pbp_defense_20261009.{parquet,meta.json}` staged by Phase 5) — delivery holds |
| `WARNING missing-finals guard: … 2026-10-08 CLE@CWS (849832) ABSENT` | The T1 guard from the 10-08 review fired exactly as designed, and its forecast came true: `predictions_history_20261009.csv` now carries the **2026-10-07 slate (4 games)** that the 10-08 run lost — the forward top-up + 3-day tail refresh recovered it. Only 849832 (10-08) remains outstanding at log time |
| `WARNING No games found for 20261009 (schedule fetch empty) — shipping an EMPTY board` | **Genuine off-day, verified live against StatsAPI:** the schedule holds no game on 2026-10-09 at all (ALDS G3 10-10 `849831`, NLDS G3 10-11 `849809`). `todays_games_20261009.csv` = 0 rows, `SHAP attributions written for 0 games` and `0 posted row(s)` are the same honest emptiness, not an outage |
| `🛡️ Ingestion guard: 2174948 pitches, 7401 decided games (expected ≥2044874 / 6960)` | Frame horizon 2024-03-20 → 2026-10-07, 7,401 decided rows |
| `Feature drift [moneyline]: 109 features, 0 warnings, 0 alerts, 0 seasonal` | No drift regression |
| `Totals history store: 7332 rows (0 added this run, seeded=False)` | Not a stall: the store already reaches **2026-10-07** (its 4 rows were written by the 10-08 22:02 run); nothing has been final-and-unframed since, because the only newer final is the still-missing 849832 |
| `Calibration: … degenerate Platt (311 of 6064 prequential fits, 3-line cap)` + final gate `identity / gated_no_gain` | The documented degenerate-fit counter (256 → 311 of 6,064) and the gated identity map behave as specified; AUC 0.5732 / Brier 0.2446 / LL 0.6821 / ECE 0.0061 reconcile with `calibration_20261009.json` |
| `Fold 82 … PROVISIONAL (19 val games < 40) — excluded from grading` | Postseason window correctly out of the graded pool (143 provisional incl. 105 postseason) |

## The feature-coverage audit: every documented gap, reviewed

The documented audit is the **2026-10-08 follow-up feature-discrepancy audit**
([mlb_log_review_20261008.md](mlb_log_review_20261008.md) → "Follow-up:
feature-discrepancy audit"), cross-read against
[mlb_input_quality_20261007.md](mlb_input_quality_20261007.md) §2 (coverage
reporting contracts). Status against the shipped 10-09 artifacts:

| # | Documented gap | Status in the 2026-10-09 run | Verdict |
|---|---|---|---|
| 1 | Bullpen trailing-window NULLs (115 rows; `LOW_COVERAGE` on the 10-03 DS openers) | `feature_coverage_20261009.csv` has **no** `bullpen_*` row outside `OK` — its only 4 `LOW_COVERAGE` rows are the two weather features × two windows (the 10-06 and 10-07 CSVs still carried 4 bullpen rows each; cleared from 10-08 onward) | **Holds** — T3 fix verified, no action |
| 2 | Indoor-neutral weather fills (214 wind + 188 air closed-roof NULLs) | 1,304 closed-roof rows: **0** with NULL wind, **0** with NULL air, **0** with a non-zero interaction; `park_wind_factor` forced 0 on all of them | **Holds** — T4 fix verified, no action |
| 3 | "Recent open-air weather misses (3 rows: 10-01 air, 10-04 both, 10-05 air) — self-healing" | **Did not self-heal.** `849844` (10-01) and `849823` (10-04) are *still* NULL for air in the shipped frame; `849839` is closed-roof anyway. Five further open-air games joined them (`813022/813023/813032/849848/849851/849838`) | **False claim — root-caused and FIXED here (T6)** |
| 4 | Watch item: "`air_density_velocity_boost` needs pressure, which the StatsAPI gap-filler does not carry — Open-Meteo-archive gaps can still leave that column alone" | Confirmed open: the filler recovered 4/8 gap games and cached **wind-only** records that then blocked the archive retry forever, so air stayed NULL for `813022/849844/849851/849838` | **FIXED here (T7)**: the partial record no longer outranks the complete one |
| 5 | (New, from this review) the `Feature coverage gaps …` WARN line is unreadable for weather features | Every run warns `…/baseline=74% measured` even when all open-air games are observed, because ~19% of the window is closed-roof **policy zeros** that the table correctly refuses to count as observations | **Clarified here (T9)**: the line now prints the open-air-only ratio beside the measured %; status thresholds and CSV schema unchanged |

## Root cause of gap 3 — exact-midnight first pitch (T6)

All 13 weather-less rows of the committed frame are exactly the 13 games whose
**official first pitch is 00:00:00 UTC** — verified one by one against
`gameData.datetime.dateTime` (`8:00 PM` / `5:00 PM` national-TV slots: 2025
NLCS/WS `813022`–`813032`, plus `849823/849838/849839/849844/849848/849851`):

```
frame start_time_utc = 2026-09-30 00:00:00+00:00   official dateTime = 2026-09-30T00:00:00Z
```

The batched sampler keys observations by the **UTC day** of first pitch and
keeps the latest hourly row *strictly before* it. Inside a day that begins at
00:00 there is no such row, so `_hour_prior` returned `-1`, the record came
back `open_meteo_unavailable`, and the game shipped **no** weather at all.
The hour that *is* strictly prior — 23:00Z — sits in the **previous** day's
series, which the fetch already requests (`min_day = min(start.date()) − 1`)
but the lookup never consulted. Reproduced before/after on the real module:
pre-fix `available=False` for all 13; post-fix **13/13 recovered live from
Open-Meteo** (temperature, humidity, wind and pressure all present, e.g.
`849823` 37.7 °C / 17% RH → air density 1.10794).

This is why the 10-08 audit's "the next build's gap-filler picks them up"
claim failed twice over: the StatsAPI filler only ever yields **wind** (the
feed has no humidity → no density, honest by contract) and it refuses an
unmappable direction phrase — 4 of the 8 gap games report
`"5 mph, Varies"`, which produced neither a multiplier nor a cache record
(pinned in T8: refused, never guessed).

## Root cause of gap 4 — a partial record outlives the complete one (T7)

`_attach_weather_history` caches **one record per `game_pk`** and built its
fetch set as "any pk not already cached". The wind-only official fills
(`source=statsapi_gamefeed`, `air_density=NULL`) therefore satisfied the gate
permanently: `849851/849844/849838/813022` were never re-offered to the
archive even after it published their hours, so
`air_density_velocity_boost` stayed NULL on exactly the games the filler had
"recovered". The gate now re-attempts **only** cached records that still lack
an air-density observation; a failed retry keeps the partial (only available
records are ever written back), and the run log reports
`Weather top-up: N/M …` so the completion is visible.

## Reviewed and deliberately NOT changed

* **SP-input NULLs are the other half of the weather-feature NULLs, and they
  are policy, not a gap.** 967 rows lack `sp_era_diff` and 864 lack
  `sp_fbvelo_diff`, which is why 767 of the 771 weather-interaction NULLs are
  input-driven rather than weather-driven. Measured against the frame: **0**
  of those rows has a gap < 10 days — every one is either a season/career
  debut (entering season stats do not exist yet: 411 home rows) or a
  ≥10-day return overlapping an availability stint, i.e. exactly the
  documented SP staleness gate (`_SP_STALE_GAP_DAYS`, features.py, tested by
  `test_bullpen_availability`/`test_lineup_il`). Nulling a stale carry is the
  point; zero-filling it would be fabrication.
* **Coverage status thresholds stay as published.** `LOW_COVERAGE <80%
  measured` counts closed-roof policy zeros as un-measured by design and the
  frontend tooltips document that wording, so the two weather features cannot
  clear 80% on a roof-heavy window. Changing the denominator would silently
  rewrite four dashboards; instead the WARN line now carries the open-air
  ratio (T9), and a threshold-policy change remains an owner decision.
* **No model, feature-set, hyperparameter or artifact-name change.** The 109
  feature universe, blend weights and filenames are untouched.

## Measured effect (recomputed with the production coverage formula)

Committed frame, last-14 (current) and last-250 (baseline) decided rows,
before → after applying the fixed observations to those 13 games:

| window | wind measured | air measured | open-air observed after |
|---|---|---|---|
| current (14) | 64.3% → **71.4%** | 57.1% → **71.4%** | 90.9% (10/11) |
| baseline (250) | 73.2% → **74.0%** | 72.0% → **74.0%** | 92.0% (185/201) |

(The shipped CSV's own baseline window differs by ≤3 rows from this
reconstruction; its `current` window reproduces exactly: 64.3% / 57.1%.)

Residual open-air misses after the fix are **input gaps, not observation
gaps**: 16 baseline rows and 1 current row whose weather *was* observed but
whose `sp_*_diff` is gated to NULL. `air_density_level` itself goes
13 → 0 missing rows, and `park_wind_factor` 4 → 0.

## Remediations

- **T6 — exact-midnight point-in-time sampling.**
  `weather._stadium_weather_utc` takes the previous day's series and consults
  it **only** when first pitch is exactly 00:00:00 UTC *and* the game's own
  day has no strictly-prior row; `fetch_games_weather` passes the previous
  day along. Every other game reads its own day exactly as before — no wider
  staleness window is ever accepted (a night game whose own-day series is
  absent stays unavailable rather than borrowing a 24h-old hour).
- **T7 — partial-record top-up.** `_attach_weather_history` re-attempts
  cached records that lack an air-density observation, preserves them on a
  failed retry, and logs `Weather top-up: N/M`.
- **T8 — official "Varies" wind** stays an honest refusal (no guessed
  direction, no density), pinned so a future "fix" cannot quietly invent one.
- **T9 — coverage WARN readability.** `compute_feature_coverage` appends
  `open-air NN% (N rows, M closed-roof policy zeros excluded)` for any
  feature carrying policy zeros. Status, thresholds and the CSV schema are
  unchanged.
- **T10 — the board smoke's recovery target vs the honest off-day board.**
  `frontend/test_mlb_board_render_smoke.py` scenario A computed its
  expectation as "the newest VALID date ≤ ET-today", but the page's
  `_recovered_board` accepts only a **non-empty** frame (by design — "an
  empty frame must NOT short-circuit the walk"). With
  `todays_games_20261009.csv` shipping **0 rows** (genuine off-day), the
  page correctly stepped back to 20261008 while the smoke demanded
  "October 09" → `AssertionError: recovery must land on the most recent
  valid board (20261009)`. The same expectation class broke on 10-07; the
  smoke now mirrors the page's renderability rule (board rows > 0, else the
  prediction-history rebuild for that date) and pins the **newest
  renderable** board. The honesty assertions (recovery banner, valid-set
  gate, no 0925/0926 content, dead-end path) are untouched.

## Verification

- `python -m pytest mlb-backend/backend/ -q` — **371 passed** (8 new pins in
  the existing `test_log_review_20261008.py`: midnight recovery, own-day
  regression guard, no stale fallback, partial re-attempt, partial
  preservation, complete-record cost pin, "Varies" refusal, open-air WARN).
- The new tests were checked against the **pre-fix** module loaded from
  `git show HEAD:` — the midnight case returns `available=False` there and
  passes now (a real regression pin, not a tautology).
- `python check_production_graph.py` — **OK, 27 modules** (no new files).
- Frontend smokes, each **exit 0**: `test_mlb_board_render_smoke` (5/5
  scenarios — T10, failing with `AssertionError … (20261009)` before the
  fix and reproduced unchanged in a clean worktree at `54c85d54`),
  `test_board_render_smoke`, `test_nhl_board_render_smoke`,
  `test_monitor_smoke`, `test_calibration_smoke`,
  `test_power_rankings_smoke`, plus
  `pytest frontend/test_calibration_gate_note.py` (**2 passed**).
- **Live end-to-end on the 13 defective games**: `fetch_games_weather`
  recovers 13/13 with observed temperature/humidity/pressure against the
  shipped frame's 0/13.
- Coverage effect measured on the committed frame with the production
  formula (table above); frame-level reconciliation of the 13 rows against
  StatsAPI `dateTime` (13/13 exact matches).
- `git diff --check` clean; **no new programs** (3 production modules, 1
  existing test file, 1 document).

## Known limitations

- The committed 10-09 frame is **not** rewritten: the repaired observations
  land on the **next pipeline run**, which will also fetch the still-missing
  `849832` (10-08 CLE@CWS) final. `pitches.parquet` and
  `weather_history.parquet` live outside this repo, so no local rebuild is
  possible.
- The 80%-measured threshold remains unreachable for the two weather
  features on roof-heavy windows (documented above, deliberately not
  changed); the WARN line will still fire for them, now with the open-air
  ratio attached.
- `python -m pytest frontend/` (whole directory in one session) still crashes
  in pytest's capture layer — pre-existing since the 10-08 review; frontend
  tests are run per file. The board smokes additionally re-wrap `sys.stdout`
  at import, so they must be run as **scripts** (`python
  frontend/test_mlb_board_render_smoke.py`), never through pytest — that
  single-file pytest crash is pre-existing too.
- The production pipeline was not re-run (Kaggle/Colab environment); all
  conclusions come from the committed artifacts, the committed run log, and
  live StatsAPI/Open-Meteo queries.


---

## Second delivery — the 17:16 run log on remote (reviewed same day)

**Subject:** the run log appended at `34b6cddc` (431 lines total, run
trained 2026-10-09 17:15, artifacts `2cc8f88f`). **Verdict: clean.**
Every fix from this document's first review is verified landed in the
delivered artifacts, the missing-final and off-day paths are resolved,
and the log's transient Statcast retry recovered byte-identically. The
only WARNING class left standing is the feature-coverage pair — handed
to the same-day [feature-coverage audit](mlb_coverage_audit_20261009.md),
which root-caused it as a fully-explained policy false alarm and
remediates it there (STRUCTURAL classification, this commit). Scope note:
only the delta from the first review is re-argued below; T1–T10 stand as
written.

### Validated clean (first review → second delivery)

| First-review claim or open item | Second-delivery evidence |
|---|---|
| T1: missing finals recover on the next pull | Guard now logs `INFO missing-finals guard: no official schedule rows for 2026-10-09 → 2026-10-09 (frame horizon 2026-10-08)` — `849832` (10-08 CLE@CWS) is recovered end-to-end: frame horizon 2026-10-08, the 10-08 slate present in `predictions_history_20261009.csv` (row `2026-10-08 CWS CLE`, `home_win=0`), `Totals history store: 7333 rows (1 added this run)`, decided games 7401 → 7402 |
| T6: exact-midnight sampling recovered all 13 games | Shipped frame: **13/13** of the named games carry observed weather (`park_wind_factor` + `air_density_level` present; open-air ones carry real interactions) — [midnight_games_recovery.csv](mlb_coverage_audit_20261009/midnight_games_recovery.csv). `Weather fetched: 7402/7402 games from batched observations` on an empty cache |
| T7: partial records no longer block the archive | Fresh cache, `Weather history: 7402/7402 games with observed weather`; frame-wide observation scan finds **0** open-air rows missing `park_wind_factor`/`air_density_level` — the T7 top-up line simply had nothing to repair |
| T9: coverage WARN carries the open-air ratio | Present verbatim in both view warnings: `open-air 92% (12 rows, 3 closed-roof policy zeros excluded)` — now superseded by the STRUCTURAL remediation in the coverage audit |
| T10: board smoke must expect the newest renderable board | `python frontend/test_mlb_board_render_smoke.py` → PASS (5 scenarios) against this delivery |
| Off-day board honest | Live StatsAPI `date=2026-10-09` → `totalGames: 0`; `todays_games_20261009.csv` 0 rows; `SHAP attributions written for 0 games`; `0 posted row(s)` — same honest emptiness, re-verified |
| Ingestion / drift / calibration clean | `Ingestion guard: 2175316 pitches, 7402 decided games (expected ≥2044874 / 6960)` pass; `Feature drift [moneyline]/[run-engine]: 109 features, 0 warnings, 0 alerts, 0 seasonal`; degenerate-Platt counter `231 of 6140` with the documented 3-line cap, final gate `identity / gated_no_gain` |

### New in this log — reviewed, no defect

- **Chunk retry (line 47):** `Chunk 2025-06-24 → 2025-08-22 attempt 1/3
  failed: Error tokenizing data ... Expected 1 fields in line 12, saw 2`
  — a truncated/rate-limited Statcast response. The bounded retry fired
  (`↻ retrying ... attempt 2/3, backoff 1.0s`) and the chunk then
  returned **221,858 pitches — byte-identical to the 05:27 run's chunk**,
  so the retry machinery recovered the full data with no silent loss.
- **`Fold 83 ... val=1 auc=nan brier=0.2955 [PROVISIONAL]`** — a single
  10-08 validation game; AUC is undefined for one sample, the fold is
  fit + scored and honestly excluded from grading, same as the 143/144
  provisional folds before it. No action.
- **Delivery accounting:** Phase 5 staged 26 files; `2cc8f88f` carries
  24 with diffs — `features_metadata_20261009.json` and
  `todays_games_20261009.csv` were already byte-identical on remote
  (deterministic 0-row/109-feature outputs from the 05:27 run).
  `git ls-files` shows 17 `*20261009*` paths; Phase 6 reports `No stale
  files`; the final log delivery landed as `34b6cddc`. Delivery holds.
- **`Calibration: ... degenerate Platt` ×3 and the `WARNING No games
  found for 20261009` line** are the already-documented classes above —
  unchanged behavior, verified again.

### The remaining WARNING class → remediated in the coverage audit

`WARNING Feature coverage gaps [moneyline]/[run-engine]` still fired on
the two weather features (73%/75% measured) even though this run's
frame has **zero** weather-observation gaps: the rows are closed-roof
policy zeros plus SP-staleness-gated NULLs, which the 80%-measured
denominator can never absorb. Full audit, evidence and remediation:
[mlb_coverage_audit_20261009.md](mlb_coverage_audit_20261009.md) —
fully explained rows now classify `STRUCTURAL` (INFO with reason, never
WARNING); one unexplained row keeps the raw alarm. The next run's log
should carry no `Feature coverage gaps` WARNING while observation
coverage holds.

### Verification for this section

- `python -m pytest mlb-backend/backend/ -q` — **377 passed** (6 new
  T11 pins, all failing against the pre-fix module).
- `python check_production_graph.py` — **OK, 27 modules**.
- Frontend smokes, each exit 0: MLB board (5/5), board render, monitor,
  calibration, power rankings, calibration gate note (2 passed).
- Artifact reconciliation of all 218 coverage rows against the
  committed frame; live StatsAPI off-day check.


---

## Third delivery — the 19:08 run, and the notebook traceback

**Subject:** artifacts `129c3fd3`, final log `f4ce14d9` (434 lines, run
trained 2026-10-09 19:08). **Verdict: pipeline and delivery clean — the
first run to ship the coverage audit's `STRUCTURAL` statuses, verified
end-to-end.** The reported `FileNotFoundError` is **not** in the pushed
log (434 lines, no traceback): it comes from the Kaggle notebook's
post-run confirmation block, executed from the LIVE Kaggle copy which
carries a typo'd duplicate of the `repo` literal. Delivery was
unaffected (pipeline exit 0; Phase 5 pushed and remotely verified 25
files).

### Validated clean

| Check | Evidence |
|---|---|
| Coverage remediation live (end-to-end) | Log carries `INFO Feature coverage [moneyline]/[run-engine]: all 218 feature-window pairs OK (4 STRUCTURAL by declared policy)` with the full reasons — **no `Feature coverage gaps` WARNING in either view**. CSVs: 214 `OK` + 4 `STRUCTURAL`, `structural_reason` column present, run-engine ≡ moneyline. This closes the "verify on next run" item from the coverage audit |
| Run otherwise identical to the 17:15 delivery | Same frame (7,402 games / 2,175,316 pitches, horizon 2026-10-08), same folds and metrics, missing-finals guard clean, totals history 7333 (0 added — already current), ingestion guard pass, off-day board honest again (still no 10-09 games) |
| New log class: Open-Meteo 429 retries (lines 199-206) | Two rate-limit backoff ladders (1/6 → 4/6) during the fresh-cache weather refetch of this back-to-back same-day full repull. **Recovered completely**: `Weather fetched: 7402/7402`, wind/air coverage byte-identical to the prior run (6635/6711). The retry machinery worked as designed — no action |

### The reported traceback — root cause

```
FileNotFoundError: [Errno 2] No such file or directory:
    '/kaggle/working/sports_predictio_model'
    ... subprocess.run([...], cwd=repo)   # notebook confirmation block
```

- The traceback is **kernel output** of the notebook session (ipykernel
  cell after "Pipeline completed — artifacts pushed to GitHub by Phase 5
  sync."), not tee'd log content — the pushed log is clean.
- The confirmation block of the live Kaggle copy re-types the `repo`
  literal a **second** time (the pipeline block already defines it) and
  the duplicate was typo'd: `sports_predictio_model` (missing `n`).
  Every **committed** notebook version — including the Kaggle-synced
  Version 5 (`89bd0879`) — is correctly spelled; the live copy has an
  unsaved/unsynced edit.
- Consequence class: a **successful** delivery was rendered a red run
  by an advisory post-run check that duplicated state it did not need
  to redefine.

### Remediation

- **Repo side (this commit).** The notebook is Kaggle-owned — never
  edited from the repo (standing guardrail, T7 of the 2026-10-06
  review). Defense is therefore a content pin in the existing
  [test_log_review_20261008.py](../mlb-backend/backend/test_log_review_20261008.py)
  (**T12**): every `repo = "..."` literal in the notebook must be the
  canonical `/kaggle/working/sports_prediction_model`; any
  `sports_predictio` not followed by `n_model` fails the suite; the
  post-run sanity footer (`Repo HEAD after run:` +
  `Latest dated artifacts:`) must stay. A Kaggle re-upload carrying the
  defect class now fails loudly at the next sync instead of shipping.
- **T13 — the RUN repairs the notebook's path; the notebook program is
  never changed (guardrail).** `github_sync.ensure_notebook_sanity_alias`,
  called at the end of `master_pipeline`'s final-delivery tail (non-fatal,
  Kaggle-only), points the typo'd location at the real clone through a
  directory symlink. The pipeline executes in the same kernel session
  BEFORE the confirmation block, so the notebook's advisory `git log` /
  `git ls-tree` checks now run green and report the ACTUAL repository.
  The live notebook needs **no edit** for runs to end green; the typo
  stays visible there and fixing it on the Kaggle side is optional
  cleanup. Pinned behaviorally: existing path untouched, missing clone
  never fabricated, never fatal, source-pinned to the delivery tail.

### Verification for this section

- `python -m pytest mlb-backend/backend/ -q` — **385 passed, 1 skipped**
  (3 T12 + 6 T13 pins; T12 green on the committed notebook and red
  against a typo'd copy; T13's positive path verified with a patched
  creator — the real-symlink case skips on hosts that refuse symlinks
  (this Windows review host) and runs on Kaggle/Linux).
- `python check_production_graph.py` — **OK, 27 modules**.
- 19:08 artifacts re-read directly: coverage CSVs, monitor JSON parity,
  weather fetch counts, guard/totals lines (table above).

---

## Fourth-delivery review — 2026-10-09 20:40 (`ff7801e9`/`21b6f1a1`)

Two more same-day deliveries landed after the third review: 20:07
(`ce405493`/`95afedc8`) and 20:40 (`ff7801e9`/`21b6f1a1`). The 20:07 run
completed its pipeline but **honestly aborted at delivery**: its Phase 5
push raced this workspace's NFL commit `7e8c4a4e` on `origin/main` and
logged `WARNING Error lines received while fetching: error: failed to push
some refs` — a transient git non-fast-forward, not a data defect. The
20:40 rerun delivered cleanly: **25 files pushed and remotely verified**,
retention clean ("data_delivery holds exactly this run's artifacts"),
DONE with the full run stats (7402 games, 2,175,316 pitches, 290 game +
88 pbp columns).

### Line-by-line verdict

A normalized diff against the reviewed 19:08 log shows the 20:40 run is
materially identical to the third delivery except: **no weather 429
retries at all** (the 19:08 run's Open-Meteo 429 backoff ladders are
gone — clean archive fetch), the expected sync tips
(`129c3fd3` → `ff7801e9`), retention counts, and progress-bar timing.
The four WARNING lines are all previously-reviewed, documented classes:

| Line | Class | Verdict |
|---|---|---|
| `No games found for 20261009 ... EMPTY board` | genuine off-day | `todays_games_20261009.csv` ships header-only; the empty-board contract (never recycling decided games) ships every consumer a 0-row slate it already handles |
| `degenerate Platt params (a=-0.348/-0.230/-0.180) — identity map` ×3 | T4-calibrated class | the INFO companion states `231 of 6140` run-engine prequential fits degenerated to identity (negative slope); countable and pinned since the 10-06 review (test_log_review_20261006 T4) |

The `[MEM]` telemetry block (Start 704 MB → pandas-load peak 3739 MB →
682 MB after DuckDB close) is the documented live-resident instrumentation
in `features.py`, present in every run since 10-05.

### Feature-coverage audit — closure verified, nothing outstanding

`docs/mlb_coverage_audit_20261009.md` closed every finding in its own
commit; this review verified each against the 20:40 artifacts:

- **High (weather LOW_COVERAGE false alarm)** → STRUCTURAL statuses live
  for a **second consecutive run**: `Feature coverage [moneyline]:
  all 218 feature-window pairs OK (4 STRUCTURAL by declared policy)` and
  byte-identical run-engine wording, with the four weather rows carrying
  their declared-policy reasons at INFO — no coverage WARNING anywhere
  in the log.
- **Verified-fixed list (10-08 discrepancies 1–5)** → re-confirmed in the
  20:40 CSVs: no `bullpen_*` row outside OK, no `MISSING_COLUMN`, no
  `n_invalid`, CSV ≡ run-engine CSV ≡ monitor JSON.
- **Low (documentation drift "29 kept features")** → corrected in the
  audit commit; comments now match the single-list rule (109 features).
- Known limitations are non-actionable by design (artifacts never
  rewritten post-delivery; the 80% denominator unchanged; the one-row
  baseline tie boundary pinned by production sort).

### Also verified in this delivery

`973fe7fb` (16:07, xFIP A/B/C verdict + dashboard label coverage) ran
inside the 20:40 run without incident: the serving set is confirmed
unchanged (Scenario C +0.00035 vs shipped, all paired deltas inside the
0.000681 seed-noise floor), and `FEATURE_DESCRIPTIONS` now covers all 109
served MLB names.

### Verification for this section

- `python -m pytest mlb-backend/backend/ -q` — **385 passed, 1 skipped**
  (re-run on the combined tree since `973fe7fb` touched
  `config.py`/`frontend/utils.py` after the third review's run; skip is
  the known Windows symlink case).
- `python check_production_graph.py` — **OK, 27 modules**.
- Frontend smokes: `test_mlb_board_render_smoke` (5/5 scenarios),
  `test_monitor_smoke` (clean) — covering the label-path change.
- 20:40 log and artifacts re-read directly; the 20:07 abort line and the
  19:08-vs-20:40 diff quoted above.
