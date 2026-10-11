# NHL run-log review — 2026-10-11

Scope: the delivered `nhl-backend/data_delivery/nhl_pipeline_run_log.txt` — 442
lines, run **2026-10-11 05:20:16 → 06:44:11 (5,035s)**, `NHL_FULL_REPULL=1`,
delivered by `6d87483e` ("Update NHL production artifacts") — the second NHL
run of the day (the first, `8a438018`, delivered at 02:20).

**Verdict: 0 ERROR / 0 Traceback; 3 WARNING lines, all classified below; 1
finding remediated (the drift summary never printed its INSUFFICIENT rows);
the same-day rerun delta fully explained; every material claim reconciled
against the delivered artifacts.**

## Log line review — all 3 WARNINGs

| Line | Message (abridged) | Verdict | Disposition |
|---|---|---|---|
| 304 | `OOF validation population is thin on 14 of 60 window(s) (< MIN_VAL_FOLD_GAMES=40) … retained for evidence, but PROVISIONAL: excluded from pooled metrics, blend weights and calibration` | Working as designed — 2025 playoff weeks and the final 2026-10-10 window (n=31) are genuinely thin; provisional rows are disclosed and never grade the blend. Arithmetic reconciles: `2465 grading + 50 provisional-regular + 168 postseason = 2683` OOF rows, provisional 207 = 50 + 157. | None. |
| 413 | `coverage [baseline]: 6 feature(s) carry default-filled values counted as UNMEASURED — a warm gap, not a warm-up (+18 postseason-boundary feature(s))` | The 2026-10-09 remediation working as designed (hole vs documented source boundary split). Reconciles **exactly** against `run_engine_feature_coverage_20261011.csv`: 6 hole + 18 boundary = 24 default-filled features; `98.853%` on the eligible denominator; `159` cold nulls; 0 warm nulls. | None. |
| 417 | `coverage [current]: 3 feature(s) … (pl_ppo_r_away/home/diff)` | Exact vs the same CSV: 3 hole features, `99.607%`, 0 cold nulls, 0 warm nulls. | None. |

## Finding 1 (remediated): the drift summary never printed its INSUFFICIENT rows

Both drift views read:

```
INFO Feature drift [moneyline]: 68 features, 0 warnings, 0 alerts, 8 seasonal (statuses on noise-adjusted PSI; mean noise floor 0.093)
INFO Feature drift [run-engine]: 68 features, 0 warnings, 0 alerts, 8 seasonal (…)
```

while both delivered artifacts — `run_engine_feature_drift_20261011.csv` and
`feature_drift` in `nhl_model_monitor_20261011.json` — hold:

```
68 = 57 OK + 8 OK-SEASONAL + 3 INSUFFICIENT (is_playoffs, back_to_back_away, is_home)
```

`0 + 0 + 8 = 8 ≠ 68`: a reader reconciling the line against the CSV silently
assumes the other 60 rows were OK, when **3 were never judged** (`n_current=60`
against the judge floors — `is_playoffs`/`is_home` are also constant in the
current window, so PSI cannot be binned). This is MLB's T15 defect verbatim:
the line was copied from MLB's `compute_feature_drift` **before** MLB's
2026-10-10 remediation, and the copy never got the fix. The suite could not
catch it because the only pin (`test_feature_drift_logs_view_labeled_summary_lines`)
asserted the line's *prefix*, never its contents.

**Remediation** (`nhl-backend/backend/monitoring.py`, `feature_drift` — message
only; the judge floors `n_baseline < 100 or n_current < 30` and the
never-pages-on-INSUFFICIENT rule are untouched, MLB parity):

```
INFO  Feature drift [%s]: %d features, %d evaluated, %d INSUFFICIENT, %d warnings, %d alerts, %d seasonal (statuses on noise-adjusted PSI; mean noise floor %.3f)
WARN  Feature drift [%s]: 0 of %d features evaluable this run (every row INSUFFICIENT — current/baseline windows under the PSI sample floor) — drift is UNMEASURED, not clean
```

The counts now always sum to the CSV (`65 evaluated + 3 INSUFFICIENT = 68` for
this run), and the all-unevaluated case — the off-season shape that turned
MLB's line clean for seven straight runs — pages instead of reading healthy.
INSUFFICIENT alone still never warns: it is a window-size statement.

**Pins** (appended to `test_run_engine_pit.py`, the file's established
drift-summary pattern):

- `test_drift_summary_publishes_evaluated_and_insufficient_counts` — one
  judged + one dataless column → `2 features, 1 evaluated, 1 INSUFFICIENT`,
  and **no** warning (mixed windows don't page);
- `test_drift_summary_pages_when_nothing_was_judged` — a 10-row current window
  (under the 30-row floor) → `2 features, 0 evaluated, 2 INSUFFICIENT` plus
  exactly one `UNMEASURED, not clean` WARNING carrying the line's prefix.

### Cross-sport note (out of scope, flagged)

`nfl-backend/backend/monitoring.py:627` still prints the pre-T15 format —
NFL's 2026-10-11 log reads `70 features, 0 warnings, 2 alerts, 0 seasonal`
while `run_engine_feature_drift_20261011.csv` holds `66 OK + 2 STRUCTURAL +
2 ALERT`: **2 STRUCTURAL rows invisible** (and INSUFFICIENT would be too).
Same class, same fix; left for a dedicated NFL pass. NBA prints no such
summary line.

## Same-day rerun delta — reviewed, no defect

| | 02:20 run (`8a438018`) | 06:44 run (`6d87483e`) |
|---|---|---|
| feature frame | 2865 decided games | **2875** (+10 finalized between runs) |
| OOF | 2673 (2465 grading / 168 postseason / **197** provisional) | 2683 (2465 / 168 / **207**) |
| final weights | 44.3 / 22.8 / 32.9 | identical |
| `moneyline OOF raw` auc | `0.5959569634053661` | **bit-identical** |
| artifacts | — | 211 common → **206 byte-identical**, 5 changed, 15 added, 13 removed |

Every added game landed in the **last** window (2026-10-04 → 10-10, n=31,
already provisional), so grading rows — and therefore weights and headline
metrics — did not move a digit. The 5 changed artifacts are exactly the
legitimate ones: `models/nhl_ensemble_latest.joblib` (refit on 2875 rows),
`nhl_injury_snapshot_history.parquet` (new snapshot), the run log, and the
cards history (+10 rows). The 13 removed match `retention: removed 13 stale
file(s)` line for line. Deterministic, explained, no action.

## Claims reconciliation — every material count against the artifacts

| Log claim | Artifact | Verdict |
|---|---|---|
| `boxscores resolved: 2875 requested … 2875 fetched`; `feature frame: 2875 decided games, 148 columns` | frame size | ✓ |
| `OOF reach: 2683 of 2875 eligible games validated (93.3%)`; `folds: 60 windows, provisional 14/207, dropped 0` | fold geometry block | ✓ (2683 = 2465 + 50 + 168 as decomposed above) |
| `moneyline OOF complete: … 2465 grading / 168 postseason / 207 provisional, final weights elasticnet=0.443, lightgbm=0.228, xgboost=0.329` | `folds oof blocks` + monitor `ensemble` | ✓ blocks overlap by design; headline n=2465 |
| `Published blend: … replay only in p_ensemble_retrospective (not OOF)` | retrospective line labeled `NOT OOF` | ✓ causal/replay separation explicit |
| `coverage [baseline]: 68/68 features, 98.853% measured on eligible games, 159 cold null(s)`; `[current] 99.607%, 0`; `[serving slate] 100.000%, 0` | `run_engine_feature_coverage_20261011.csv` recomputed: measured ÷ (games − cold null − cold default) | ✓ exact to 3 decimals; 6+18=24 and 3 default-filled features exact |
| `Feature drift [both views]: 68 features, 0 warnings, 0 alerts, 8 seasonal` | CSV + monitor JSON: 57 OK + 8 OK-SEASONAL + **3 INSUFFICIENT** | ⚠ counts didn't sum → **Finding 1 (remediated)** |
| `11 gates … PASS` (PHASE 13) | schema-validation block | ✓ |
| `NHL SHAP attributions written for 3 games`; `slate games: 3` | 3 SHAP artifacts for 20261011 | ✓ |
| `cards history store: 2692 rows (10 added this run)` | `nhl_production_cards_history.csv` changed in delivery | ✓ |
| `retention: … removed 13 stale file(s)` | `git diff 8a438018..6d87483e` removals = 13 (all `*20260930*`) | ✓ |
| `DONE - 14 artifacts written in 5035s` | persistence-phase counter; delivery added 15 files (14 written + delivery bookkeeping) | ✓ within counter semantics |
| 0 `ERROR`, 0 `Traceback`; 3 WARNING lines | grep over the delivered log | ✓ |

Also reviewed, no defect: `feature subset: applied False … 'no adopted state'`
(the RFE universe fallback, by design); `injury id bridge 74.6% matched`
(INFO, name-bridge disclosed with 0 ambiguous); `wrong-team guard dropped
7756` (roster guard counting rows, not games); drift `mean noise floor 0.093`
vs the CSV's `noise_floor` column; `calibrated NB dispersion … POISSON LIMIT`
(alpha=0, disclosed); the coverage `status` column reading OK while the ledger
WARNs (status uses the published thresholds <80% / <25%, the ledger pages on
warm gaps — two different questions, both documented).

## Verification

- Targeted: `python -m pytest test_run_engine_pit.py -k "drift_summary or
  drift_logs_view or coverage_verdict or postseason_pool_boundary"` →
  **6 passed, exit 0** (2 new pins + 4 pre-existing).
- Full NHL suite (all 17 test files, run in three chunks — no single
  shell survives the 600s tool cap, the slow half takes 9:29):
  **462 + 79 + 78 = 619 passed, 0 failed**, exit 0 on every chunk.
- Reconciliation of every table above was executed against the delivered
  artifacts in `nhl-backend/data_delivery/` (pandas/JSON recomputation, not
  eyeballing).

## Deliberately not changed

- Judge floors (`n_baseline < 100`, `n_current < 30`) and
  `INSUFFICIENT never pages` — owner policy, MLB parity; only the *disclosure*
  of what was not judged changed.
- The coverage ledger's `(%d default-filled)` numbers — they reconcile exactly
  with the CSV's `n_default_zero` column (the warm/cold split lives in
  `n_warm_default` / `n_cold_default`, and the headline percentage correctly
  uses the eligible denominator it names).
- The cold-only-defaults disclosure branch (did not fire this run) and the
  monitor JSON's `feature_coverage` card (rows already carry `status`;
  consumers can count) — no summary card added.
- OOF provisional policy, the 11 gates, calibration gates, retention anchors —
  untouched.
- `nfl-backend`'s identical pre-T15 summary line — same defect class, flagged
  above for a dedicated NFL pass, not fixed here.

## Residual risk / next-run expectations

- From the next run on, the drift lines read
  `68 features, 65 evaluated, 3 INSUFFICIENT, 0 warnings, 0 alerts, 8
  seasonal` (this run's numbers) — reviewers should expect the counts.
- On a run where the current window drops under the 30-row floor (NHL
  off-season), the line will now emit the `UNMEASURED, not clean` WARNING —
  correct by design (MLB has paged this way since 2026-10-10), not a
  regression.
- NFL's summary line carries the same hidden-status defect (2 STRUCTURAL
  invisible on its 2026-10-11 delivery); recorded here, remediation deferred
  to an NFL-scoped change.
