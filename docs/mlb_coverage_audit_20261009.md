# MLB feature-coverage audit — 2026-10-09 snapshot

## Verdict

**The delivered coverage table is clean except for one standing false
alarm.** All 109 serving features across both drift windows reconcile
against the committed frame: 214 of 218 feature-window rows are `OK`
(≥93.3% measured), every column is present, and `n_invalid` is 0
everywhere. The moneyline CSV, the run-engine CSV and the monitor JSON
are byte-identical views of the same table. The only non-`OK` rows are
the two weather interactions × two windows — and this audit proves
those four rows are **100% declared policy, not starvation**: the frame
carries zero weather-observation gaps (0 of 6,098 open-air rows lack
`park_wind_factor` or `air_density_level`), every one of the 767 wind /
691 air open-air NULLs sits behind the documented SP-staleness gate, and
every closed-roof row carries its policy zero. The defect is therefore
in the STATUS, not the data: `% measured` counts policy zeros and policy
NULLs in its denominator, so the row could never reach the 80% OK line
and stood `LOW_COVERAGE` on every run — a permanent alarm that trains
viewers to ignore the backstop. Remediated here: a fully explained row
now classifies `STRUCTURAL` with its reason, one unexplained row keeps
the raw alarm.

This is an audit of the 2026-10-09 17:15 run's delivered artifacts plus
the documented 2026-10-08 feature-discrepancy list. No model, feature
set, hyperparameter, threshold or artifact was rewritten; the committed
coverage CSVs still show this run's delivered statuses (artifacts are
the run's record). The status change takes effect on the **next**
pipeline run.

## Scope and reproduction context

- Artifacts: `feature_coverage_20261009.csv`,
  `run_engine_feature_coverage_20261009.csv`,
  `model_monitor_20261009.json`, `game_level_features.csv` as delivered
  by commits `2cc8f88f` (artifacts) / `34b6cddc` (run log, 431 lines,
  trained 2026-10-09 17:15).
- Frame: 7,402 rows × 297 columns, 2024-03-20 → 2026-10-08 (the missing
  `849832` 10-08 final is recovered — frame horizon advanced).
- Windows, recomputed exactly as production does (`cutoff = target −
  7d`, baseline = `prior.tail(max(3·len(current), 250))`): current 15
  games (2026-10-03 → 2026-10-08), baseline 250 games (2026-09-09 →
  2026-10-01).
- Serving width: the adopted 109-feature RFE subset — the SINGLE-LIST
  rule; the run-engine view resolves the same contract (the shipped
  CSVs are identical, verified).
- Every number below is reproduced from the committed artifacts by
  `.freebuff/mlb_cov_audit_evidence.py` (scratch harness, not
  committed); bounded evidence is committed under
  [mlb_coverage_audit_20261009/](mlb_coverage_audit_20261009/).

## Findings

### High — the standing weather `LOW_COVERAGE` alarm is a false alarm that can never clear

| window | shipped | unmeasured rows | explained by | unexplained |
|---|---|---|---|---|
| current (15) | 73.3% measured, `LOW_COVERAGE` | 4 | 3 closed-roof policy zeros + 1 `sp_era_diff`-gated NULL | **0** |
| baseline (250) | 74.8% measured, `LOW_COVERAGE` | 63 (62 in the reconstruction) | 46–47 policy zeros + 16 `sp_*_diff`-gated NULLs | **0** |

- Observation scan over the whole frame: open-air rows 6,098;
  `park_wind_factor` NULL **0**; `air_density_level` NULL **0**; dome
  rows with a NULL weather interaction **0**
  ([audit_summary.json](mlb_coverage_audit_20261009/audit_summary.json)).
  There is no observation gap left anywhere — the T6 exact-midnight and
  T7 partial-record fixes from this morning's review are verified
  landed (all 13 named midnight games carry observed weather:
  [midnight_games_recovery.csv](mlb_coverage_audit_20261009/midnight_games_recovery.csv)).
- Every open-air NULL is the SP-staleness gate doing its job
  ([weather_null_causes.csv](mlb_coverage_audit_20261009/weather_null_causes.csv)):
  `null_open_air_input_present = 0` in both windows — i.e. not one NULL
  row has a valid SP input beside a missing observation, which is the
  exact signature of the 2026-10-07 truncation incident.
- Because `% measured` divides by ALL rows (policy zeros and policy
  NULLs included), the row cannot reach 80% on a roof-heavy window even
  at perfect observation coverage, and the run log carries
  `WARNING Feature coverage gaps` twice per run (moneyline +
  run-engine) forever.

**Remediation (this commit).** `compute_feature_coverage` now classifies
a non-`OK` row of a declared-policy feature as `STRUCTURAL` **only when
every unmeasured row carries a declared reason** — a closed-roof policy
zero (`apply_indoor_neutral_fills` writes it unconditionally) or an
open-air NULL whose governing SP input (`sp_era_diff` for wind,
`sp_fbvelo_diff` for air) is NULL. One unexplained row — an open-air
NULL beside a present input, a broken indoor fill, an inf, a missing
input column to prove the gate — keeps the raw `LOW_COVERAGE`/`STARVED`
thresholds and the WARNING, so sensitivity to real starvation is
unchanged. This is the cross-sport contract the dashboard already
speaks: NFL grades declared-policy absence `STRUCTURAL` (2026-09-29)
and NBA requires "every missing value explained" (2026-10-08); the MLB
Model Monitor panel already renders `STRUCTURAL` with its reason (live
in the NFL monitor artifacts), and the run-engine panel in
[markets.py](../frontend/markets.py) now does too. The numbers never
change — only what the status means. `STRUCTURAL` rows are summarized
at INFO with their reason, never WARNING'd.

### Verified clean — table integrity and parity

- 218 rows = 109 features × 2 windows; `column_present` all true, no
  `MISSING_COLUMN`, `n_invalid` 0 across every row (infinities never
  count as observations).
- `feature_coverage_20261009.csv` ≡ `run_engine_feature_coverage_20261009.csv`
  (identical DataFrames), and the monitor JSON's `feature_coverage`
  list matches the CSV row-for-row — one table, three surfaces.
- Lowest `OK` rows are 93.3% (the `sp_*` input family in the 15-game
  window: one debut/return row, `849827` 10-07 SD — the staleness gate,
  not a data hole). Drift side: `109 features, 0 warnings, 0 alerts,
  0 seasonal` in both views.
- Independent recomputation of all 218 rows reproduces the shipped
  current window **exactly** and the baseline window to within one
  boundary row (±0.4 on 16 rows, all `2026-09-09` tie order —
  production pins the sort to byte-identical prior-run behavior by
  design; see [feature_coverage_reconciliation.csv](mlb_coverage_audit_20261009/feature_coverage_reconciliation.csv)).

### Verified fixed — the documented 2026-10-08 discrepancy list

| # | Documented gap | Status in the 17:15 artifacts |
|---|---|---|
| 1 | Bullpen trailing-window NULLs (115 rows) | No `bullpen_*` row outside `OK` — holds |
| 2 | Indoor-neutral weather fills input-conditioned | 0 dome rows with a NULL weather interaction (7,402/7,402 filled) — holds |
| 3 | "Recent open-air misses self-heal" (false claim) | Root-caused (exact-midnight sampling) and fixed in T6 — **13/13 recovered**, verified in the shipped frame |
| 4 | Partial wind-only cache record blocking the archive | Fixed in T7; this run's fresh cache fetched 7402/7402 observed — no partial residue |
| 5 | Coverage WARN unreadable for weather features | T9 annotation present in the log (`open-air 92% (12 rows, 3 closed-roof policy zeros excluded)`) — and superseded by the STRUCTURAL remediation above |

### Low — documentation drift: "the run engine's 29 kept features"

The run-engine monitoring resolver has shared the moneyline feature
contract since the single-list rule (`run_engine_feature_cols()` →
`active_moneyline_feature_cols()`, today 109 — the log itself prints
`Run engine: active moneyline feature view (109 features)`), but five
comments/docstrings still said "29 kept features"
([explainability.py](../mlb-backend/backend/explainability.py) ×2,
[master_pipeline.py](../mlb-backend/backend/master_pipeline.py),
[markets.py](../frontend/markets.py) ×2). Corrected in this commit —
comments only, no behavior change.

## Remediation

- **`mlb-backend/backend/explainability.py`** — declared-policy map
  (`wind_advantage_flyball_factor ← sp_era_diff`,
  `air_density_velocity_boost ← sp_fbvelo_diff`), row-level
  `_declared_policy_reason` (fail-open: no governing input column → no
  STRUCTURAL), `STRUCTURAL` status + additive `structural_reason` CSV
  column, alarms scoped to `STARVED`/`LOW_COVERAGE`/`MISSING_COLUMN`,
  structural rows summarized at INFO.
- **`frontend/markets.py`** — run-engine coverage panel renders
  `STRUCTURAL` (muted color, ok-pill, reason line, header count) and
  documents it in the footer, mirroring the Model Monitor panel.
- **Pins** — six new tests in the existing
  [test_log_review_20261008.py](../mlb-backend/backend/test_log_review_20261008.py)
  (T11): full explanation → STRUCTURAL with reason and no WARNING; one
  unobserved open-air row → keeps `LOW_COVERAGE` + WARNING; per-feature
  governing input; broken indoor fill stays unexplained; missing input
  column fails open; run-engine wrapper shares the rule. All six fail
  against the pre-fix module (`git show HEAD:...`) — real regression
  pins, not tautologies.
- **No new programs.** Three existing files edited, one document, four
  bounded evidence files.

## Verification

- `python -m pytest mlb-backend/backend/ -q` — **377 passed** (371
  baseline + 6 new T11 pins).
- `python check_production_graph.py` — **OK, 27 modules**.
- Real-frame replay: the shipped 17:15 windows recompute to
  `STRUCTURAL` for all four weather rows with counts
  (3/1, 46–47/16) in the reasons, and the log output switches from
  `WARNING Feature coverage gaps` to
  `INFO ... all 4 feature-window pairs OK (4 STRUCTURAL by declared policy)`.
- Frontend, each exit 0: `test_mlb_board_render_smoke` (5/5 scenarios),
  `test_board_render_smoke`, `test_monitor_smoke`,
  `test_calibration_smoke`, `test_power_rankings_smoke`,
  `pytest frontend/test_calibration_gate_note.py` (2 passed). The
  edited run-engine coverage panel additionally rendered a synthetic
  `STRUCTURAL`/`LOW_COVERAGE`/`OK` frame (reason line, header count,
  empty-state) without exception.
- Live: StatsAPI `schedule?sportId=1&date=2026-10-09` returns
  `totalGames: 0` — the empty-board WARNING class is a genuine off-day,
  not a source outage.

## Known limitations

- The committed `feature_coverage_20261009.csv` and monitor JSON keep
  this run's delivered `LOW_COVERAGE` statuses; the remediation is
  visible from the **next** pipeline run onward (artifacts are never
  rewritten after delivery — same stance as every prior review).
- The 80%-measured denominator is otherwise unchanged: threshold
  policy, tooltips and CSV columns are untouched except the additive
  `structural_reason`.
- The baseline window's one-row tie difference vs an independent
  reconstruction is by design (production pins the sort) and shifts no
  status.
- The pipeline was not re-run in this environment (Kaggle/Colab);
  conclusions come from the committed artifacts, the committed run log,
  and live StatsAPI/Open-Meteo checks.
