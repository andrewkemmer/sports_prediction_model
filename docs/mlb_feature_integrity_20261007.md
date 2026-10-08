# MLB feature ingestion and integrity deep dive — 2026-10-07/08

## Scope and guardrails

The request: validate ALL feature ingestion against actual game data, verify
that feature engineering is computationally correct, and investigate and
remediate any missing data, incorrect data, or structural gaps — with **no new
programs committed**, then commit and push.

Honored as follows: only existing production and test modules were edited
([features.py](../mlb-backend/backend/features.py),
[results.py](../mlb-backend/backend/results.py),
[feature_metadata.py](../mlb-backend/backend/feature_metadata.py),
[test_log_review_20261007.py](../mlb-backend/backend/test_log_review_20261007.py),
[test_mlb_board_render_smoke.py](../frontend/test_mlb_board_render_smoke.py));
regression pins were added to the existing review test; scratch harnesses live
under `.freebuff/` and are **not** committed; this document is non-executable.
The production frame
[game_level_features.csv](../mlb-backend/data_delivery/game_level_features.csv)
was repaired cell-surgically (only intended lines differ, verified byte-wise).

## 1. Official reconciliation — the frame IS the official record

Source of truth: StatsAPI `v1/schedule` (2024-03-01 → 2026-10-31, 8,785
official games) cross-checked against `v1.1/game/{pk}/feed/live`.

* Frame window 2024-03-20 → 2026-10-06: official games **7,399**; frame rows
  **7,397**; the 2 absent are both **Cancelled** (pks 746577, 823490) —
  correctly excluded. Extra rows in frame: **0**.
* Direct score matches: 7,395/7,397. The 92 apparent mismatches were the
  reconciliation fetch's dedup keeping postponed listings; each verified
  against the live feed: `game_date == officialDate` **92/92**, scores equal
  the official finals **92/92**, all Final **92/92**.
* Suspended-game census: exactly **8** games carry `resumedFrom` in the whole
  window — the same 8 found below; no additional same-day resumes exist.

## 2. Computational correctness — independent recomputation, all clean

Every engineered identity was recomputed from raw inputs with 0 mismatches:
score-derived labels (`home_win`, `total_runs`), every generic `_diff` column
(home − away), the 12 product-of-diff composites, the weather composites
(open-air `wind_advantage = park_wind_factor × sp_era_diff`; dome forced 0.0;
missing inputs stay NULL), Elo / season records / win% / run-diff, `rest_days`,
handedness matchup advantage, and cross-artifact agreement with
`predictions_history` and the run-engine OOF frame. The 109-feature universe
equals `feature_metadata` exactly: no inf, no all-NaN column, and the only
constant is the documented `is_home`.

## 3. Defects found and remediated

### Defect 1 — suspended games sealed with their RESUME slot as first pitch

8 rows carried `start_time_utc` = the resume slot, **1–61 days after**
`game_date`, yet `start_time_observed=True`. That mis-keyed the as-of
market-line merge, the Open-Meteo weather sampling ("latest hourly record
strictly before actual UTC first pitch"), and PIT ordering.

**Code.** [results.py](../mlb-backend/backend/results.py):
`fetch_game_start_times` now prefers `resumedFrom` (the real first pitch) over
`gameDate` (the resume slot), and `refresh_start_times` gained a self-healing
`repair` predicate — observed rows whose ET date is *strictly later* than
their `game_date` are re-mapped to the authoritative schedule value. The
predicate is idempotent (for non-suspended games the schedule value is that
same instant, so late West Coast starts are untouched); same-day resumes would
be undetectable by date alone, and the census above proves none exist.

**Data.** The 8 `start_time_utc` cells now hold the official
`resumedFromDateTime`, cross-verified independently from the game feed and the
schedule endpoint (both agree exactly). Re-verified: for **all 7,397 rows**
the ET date of `start_time_utc` equals `game_date`; all rows remain observed,
no nulls.

**Derived weather repaired too.** The per-game weather cache is keyed by
`game_pk`, but was populated from those resume timestamps, so the 8 open-air
rows' weather-derived cells described the wrong day. The four weather columns
(`park_wind_factor`, `air_density_level`, `wind_advantage_flyball_factor`,
`air_density_velocity_boost`) were recomputed for exactly those 8 rows with
the pipeline's own `fetch_games_weather` + `apply_weather_features` at the
true first pitch (`open_meteo_archive`, 8/8 observed). The errors were
material — e.g. pk 745180 `park_wind_factor` −0.0048 → −0.792 and
pk 746755 0.2711 → −0.4038 (sign flip). 27 cells changed, only on those 8
lines.

### Defect 2 — two competing roof truths

The model feature `dome_is_neutral` (static venue map) claimed a closed roof
on **834** self-contradictory rows — all 243 MIN home games (Target Field is
open-air) plus 591 open-roof retractable games — while the weather composites
on the same rows were computed outdoors from the game-accurate
`dome_is_neutral_game` (StatsAPI roof cache: 1,731/1,731 retractable games
resolved, 1,140 closed / 591 open).

**Code.** [features.py](../mlb-backend/backend/features.py):
`DOME_STATUS["MIN"]` corrected 1 → 0; `refine_dome_game_level` now syncs
`dome_is_neutral` to the refined per-game state (one roof truth — its
docstring records why); `add_diff_features` overlays the game-accurate flag
where non-null. [feature_metadata.py](../mlb-backend/backend/feature_metadata.py)
documents the per-game contract.

**Data.** All 834 `dome_is_neutral` cells synced to `dome_is_neutral_game`:
0 contradictions remain, both columns now mark 1,385 games closed, and all
243 MIN home rows are 0.

### Defect 3 — date-sensitive board smoke (structural test gap)

`test_mlb_board_render_smoke.py` scenario C pinned the 2026-09-27 board, but
the rolling 10-day MLB retention window
(`utils._mlb_retention_window`) legitimately drops 0927 once ET today passes
2026-10-07 — at which point the page correctly falls back to the recovery
view and the smoke failed at midnight. The smoke now picks its target from
TODAY's valid set (preferring 0927 while retained, otherwise the newest local
board **with decided rows**, since a same-night board is still LIVE/pre-game
and renders no accuracy line) while asserting the same date-independent
contract: own date, full card count, no recovery banner, decided picks
graded, and no 0925/0926 content.

## 4. Verification

* Backend: **348 passed** — including the 5 new pins in
  [test_log_review_20261007.py](../mlb-backend/backend/test_log_review_20261007.py)
  (resumedFrom preference + sealed-resume-slot refresh idempotency; MIN is
  never a dome; `refine_dome_game_level` syncs the model feature;
  `add_diff_features` prefers the game-accurate roof flag).
* `python backend/check_production_graph.py` — OK, 27 modules.
* Frontend smokes: calibration, monitor, MLB board render (5/5 scenarios),
  power-rankings, NBA and NHL suites — all pass.
* Final frame re-verification: shape 7,397 × 297; `dome_is_neutral ==
  dome_is_neutral_game` everywhere; ET date == `game_date` for every row;
  open-air composite products exact (<1e-9); dome rows carry only the
  by-design forced 0.0; no inf; labels score-consistent.
