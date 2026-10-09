# MLB run-log review — 2026-10-09

**Subject:** `mlb-backend/data_delivery/mlb_pipeline_run_log.txt` as delivered on
remote (438 lines, run trained 2026-10-09 05:27, artifacts commit `09705665`,
log commit `64b6f899`, cleanup `7b7cb6d1`, previous HEAD `871c45eb`).
**Verdict:** delivery, the missing-finals guard and the off-day path are all
clean. The run's only WARNING class left unexplained was
`Feature coverage gaps [moneyline]/[run-engine]` on the two weather features —
and it is NOT a data outage: it is two real defects (an exact-midnight
point-in-time sampling hole and a partial cache record that outranked the
complete one) plus a denominator that can never clear the 80% line. Both
defects are root-caused against the frame and the emitters and remediated in
this commit; every gap from the documented 2026-10-08 feature-discrepancy
audit is now either verified-fixed or fixed here.

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

## Verification

- `python -m pytest mlb-backend/backend/ -q` — **371 passed** (8 new pins in
  the existing `test_log_review_20261008.py`: midnight recovery, own-day
  regression guard, no stale fallback, partial re-attempt, partial
  preservation, complete-record cost pin, "Varies" refusal, open-air WARN).
- The new tests were checked against the **pre-fix** module loaded from
  `git show HEAD:` — the midnight case returns `available=False` there and
  passes now (a real regression pin, not a tautology).
- `python check_production_graph.py` — **OK, 27 modules** (no new files).
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
  tests are run per file.
- The production pipeline was not re-run (Kaggle/Colab environment); all
  conclusions come from the committed artifacts, the committed run log, and
  live StatsAPI/Open-Meteo queries.
