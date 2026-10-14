# NFL run-log review — 2026-10-14

Scope: the delivered `nfl-backend/data_delivery/nfl_pipeline_run_log.txt` — 356
lines, run **2026-10-14 02:25:08 → 02:46:39 (1,290.3s)**, `NFL_FULL_REPULL=1`,
delivered by `29a38e75` ("Update NFL production artifacts") via `c0884808`
("Pipeline run log").

**Verdict: 0 ERROR / 0 Traceback; 5 WARNINGs, all classified below; every
material count reconciles against the delivered artifacts; 1 finding
remediated — a pandas `.attrs` propagation defect that deep-copied the
200-origin distribution bundle once per row, costing the run 252s of silent
wall clock (19.5% of the run) with zero effect on any output.**

## Runtime profile — where the 1,290.3s went

| Phase | Elapsed | Share | Verdict |
|---|---|---|---|
| PIT weather archive (102 batches, Open-Meteo) | 244.4s | 19.0% | External pacing, already remediated by `1d248572` (venue dedup cut 136 → 102 batches, byte-identical keys). Zero 429s. |
| Markets: prior-fold calibration (200 origins, MC_DRAWS=50,000) | 337.5s | 26.1% | Bracketed with start/end records (`1d248572`); a real fit, disclosed on its own clock. |
| **Phase 12: monitor write — the silent gap** | **252.1s** | **19.5%** | **Finding 1, remediated.** |
| Moneyline OOF (200 folds) | 300s | 23.3% | Progress bar reports every 25 folds; healthy (1.50s/it). |
| Phase 8 calibration + OOF dumps | 65.9s | 5.1% | Identity calibrator, `gated_no_gain` — disclosed. |
| Ingestion, features, folds, eval, artifacts, monitoring | ~90s | 7.0% | Clean. |

## Finding 1 (remediated): the monitor writer deep-copied the bundle per row

The run printed:

```
2026-10-14 02:42:24,588 INFO cards history store: 2573 rows (0 added this run) -> nfl_production_cards_history.csv
2026-10-14 02:46:36,667 INFO SHAP: no slate games — no attribution cards written
```

**252.1 seconds of unlogged wall clock between two lines of a phase whose
every other write totals ~2s.** It is persistent, not a one-off: 252.1s
(10-14), 211.9s (10-13), 200.6s (10-09), 196.4s, 188.5s — present in every
delivered run since the monitor writer landed.

### Root cause

`master_pipeline` attaches the FULL distribution calibration bundle to the
market frame's `.attrs` before the monitor writer runs:

```python
oof_market_rows.attrs["challenger_bundle"] = market_calibration   # 200 origins
```

and `write_markets_monitor_json` then walked that same frame row by row:

```python
cards = markets_winner_cards(oof_market_rows)          # iterrows over 2,573 rows
"market_metrics": _run_engine_market_metrics(oof_market_rows)   # 7 more full passes
```

pandas propagates `.attrs` into **every row Series that `DataFrame.iterrows`
builds** — a shallow view of the frame still deep-copies its attrs dict into
each of the 2,573 row Series, and `_run_engine_line_pairs` runs that walk
once per market line (5 fixed totals + 1 canonical spread + derived ML =
**7 passes**). The bundle is not small: the delivered
`nfl_calibration_20261013.json` carries **200 origins × {maps,
candidate_maps} × 23 entries = 9,200 floats**, so each of the ~20,584
row-Series constructions deep-copied a ~9,200-float nested structure.

Measured on the exact production shape (2,573 delivered rows + the delivered
200-origin bundle), HEAD's writer vs the remediated one:

| | elapsed | JSON |
|---|---|---|
| HEAD | **54.6s** | — |
| Remediated | **1.1s** | **byte-identical** |

Locally that is 50x; the Kaggle runtime showed the same ratio against its
252s. Nothing reads `.attrs` inside either walker — the three bundle keys the
`fit` block publishes (`variance`, `monte_carlo`, `holdout`) are now read
**once**, above the loop, and the frame handed to the row walkers carries no
attrs at all.

### Why the output cannot change

`.attrs` never participates in a computed value here: it is metadata pandas
copies along, and no card or per-line metric consults it. The A/B above
proves it at the byte level — old and new writers emit identical JSON for
the same inputs. The regression tests pin it structurally instead of by
timing: an attrs-carrying frame and an attrs-free frame must produce
identical `winner_cards` and `market_metrics`, and a spy asserts the walkers
receive an attrs-free frame (an equality-only test would pass even with the
defect restored, since the defect changes no value).

### Cross-sport: NHL carries the same defect

`nhl-backend/backend/monitoring.py`'s `write_markets_monitor_json` has the
identical shape — `markets_winner_cards(oof_market_rows)` and
`_run_engine_market_metrics(oof_market_rows)` on a frame whose attrs
`nhl-master_pipeline.py:786` fills with the same bundle. The current NHL log
does not surface a comparable gap because its Phase 14 brackets absorb it,
but the same 50x applies whenever its bundle grows. **Not remediated here**
(the review's scope is the NFL log); the fix is the same three-line change,
and this document is the handoff.

## Claims reconciliation

| Log claim | Against | Verdict |
|---|---|---|
| `DONE — 11 artifacts in 1290s`; `artifacts written: 11` | Log span 1,290.3s; 11 named artifacts | ✓ |
| `decided games: warmup 267, core (OOF population) 2573` | 267 + 2,573 = 2,840 = `feature frame: 2840 decided games` | ✓ |
| `moneyline OOF: 2573 scored row(s), 1704 grading / 111 postseason / 869 provisional` | oof blocks: 1,704 + 869 = **2,573**; the 111 postseason rows are a **subset of the 869 provisional** (playoff weeks are thin windows), not a third additive bucket — see note below | ✓ reconciles once read as overlapping |
| `Published blend … weights {'elasticnet': '42.4%', 'lightgbm': '21.0%', 'xgboost': '36.6%'}` | `adaptive weights: {"elasticnet": 0.424, "lightgbm": 0.21, "xgboost": 0.366}` | ✓ |
| `Rolling Brier: 556 points … 2573 games … mean 0.2208` | `oof_all` brier 0.22081 | ✓ |
| `folds: {"n_folds": 200 … "provisional_games": 869, "postseason_windows": 36}` | Fold summary block | ✓ |
| `PIT weather: 2023/2023 eligible games` | `PIT weather rows: 2023`; coverage warning suppressed at 100% | ✓ |
| `Feature drift: 70 features, 0 warnings, 2 alerts … 1 coverage` | `2 drift (ALERT/WARN) … 1 coverage (STARVED/LOW)` = inj_ol_out_home/away + is_turf_home | ✓ |

Note (assessed, not remediated): the `1704 grading / 111 postseason / 869
provisional` slash list reads as a partition, but the categories overlap —
adding them gives 2,684 against 2,573 rows. The oof-blocks JSON in the same
run publishes all four populations with their own `n`, so the artifact
reconciles; only the console line invites the mis-add. A wording fix (e.g.
`1704 grading / 869 provisional incl. 111 postseason`) is a one-line change
in `moneyline.py` if a future review wants it.

## WARNING line review — all 5

| Line | Message | Verdict |
|---|---|---|
| Phase 4 | `OOF validation population is thin on 92 of 200 window(s) (< MIN_VAL_FOLD_GAMES=15) … PROVISIONAL` | By design — bye weeks, playoff weeks and week-straddling blocks; provisional rows are excluded from pooled metrics, blend weights and calibration, and the per-window detail ships in `nfl_fold_table.csv`. |
| Phase 13 | `drift inj_ol_out_home ALERT psi_adj=0.456` / `inj_ol_out_away ALERT psi_adj=0.482` | Genuine drift signal — offensive-line availability has moved against the baseline. Noise-adjusted PSI with the null median disclosed; the run continues by design. Worth a human look, not a code defect. |
| Phase 13 | `coverage is_turf_home LOW_COVERAGE 76.7%` | Disclosed venue-metadata gap; the feature stays in the model with its NaN policy. |
| Phase 13 | `full tables: run_engine_feature_drift_20261013.csv …` | Pointer line, not a condition. |

Two features are flagged STRUCTURAL (`is_snow` constant 0, `is_home` constant
1 in both windows) — correctly reported as unable to drift rather than as
clean.

## Verification

- **NFL production suite 566 passed, 0 failed** (561 baseline + 5 new
  regression checks in the existing file — no new programs).
- **NFL pytest 74 passed** (the +9 over the prior 65 is the concurrent
  `test_distribution_objective_parity.py`, not this change).
- A/B against HEAD on the exact delivered rows + delivered bundle: **JSON
  byte-identical, 54.6s → 1.1s**.
- The regression tests were verified to **fail on a reverted fix** (walkers
  re-pointed at the attrs-carrying frame → 565 passed / 1 failed, exit 1) and
  pass after restore — an equality-only pin would not have caught it.
- `kaggle_nfl_run.ipynb` and every `data_delivery/` artifact are untouched.
