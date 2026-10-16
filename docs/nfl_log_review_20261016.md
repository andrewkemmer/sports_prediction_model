# NFL run-log review — 2026-10-16 02:44 delivery

Scope: the delivered `nfl-backend/data_delivery/nfl_pipeline_run_log.txt` — 367
lines, run **2026-10-16 02:44:13 → 03:02:48 (≈1115s)**, `NFL_FULL_REPULL=1`,
delivered by `897163ef` ("Update NFL production artifacts"). This run
supersedes the 2026-10-15 20:43 crash delivery (`6ca8279e`, lattice
`ValueError`, log overwritten) — see Crash note below.

**Verdict: 0 ERROR / 0 Traceback; 4 WARNING lines, all working as designed
except Finding 1 (remediated below: the drift summary hides 2 STRUCTURAL
rows — the exact defect flagged for this dedicated NFL pass in the
2026-10-11 NHL review).**

## Log line review — all 4 WARNINGs

| Line | Message (abridged) | Verdict | Disposition |
|---|---|---|---|
| 212 | `OOF validation population is thin on 92 of 200 window(s) (< 15 games) … PROVISIONAL: excluded from pooled metrics, blend weights and calibration` | Working as designed — playoff/bye/week-straddle weeks; per-window detail in `nfl_fold_table.csv`. 2573 = 1704 grading + 111 postseason + 869 provisional (overlap-aware). | None. |
| 323 | `drift inj_ol_out_home ALERT psi_adj=0.456 …` | Named with evidence (raw/null/median); same pair as the prior run — recurring, disclosed. | None. |
| 324 | `drift inj_ol_out_away ALERT psi_adj=0.482 …` | Same as above. | None. |
| 327 | `coverage is_turf_home LOW_COVERAGE 76.7%` | Ledger pages the hole; `monitoring:` line counts it (`1 coverage STARVED/LOW`). | None. |

## Finding 1 (remediated): the drift summary hides its STRUCTURAL rows

Both drift views read (lines 319-320):

```
Feature drift [moneyline]: 70 features, 0 warnings, 2 alerts, 2 seasonal (… mean noise floor 0.136)
Feature drift [run-engine]: 70 features, 0 warnings, 2 alerts, 2 seasonal (…)
```

against `run_engine_feature_drift_20261015.csv` = **64 OK + 2 OK-SEASONAL +
2 STRUCTURAL (`is_snow`, `is_home` — constant in both windows, cannot drift)
+ 2 ALERT**. `0+2+2 = 6 ≠ 70`: a reader assumes the other 64 were OK, when 2
were stable-by-construction and never judged — and any future INSUFFICIENT
row would vanish the same way. This is MLB's T15 verbatim, pre-flagged for
exactly this pass ("2 STRUCTURAL rows invisible (and INSUFFICIENT would be
too)"). The Phase-13 ledger beside it (`monitoring: 70 features scored, 2
drift, 0 insufficient-window, 2 drift structural, …`) already counts them;
only the headline did not.

**Remediation** (`nfl-backend/backend/monitoring.py`, `feature_drift` —
message only; floors, thresholds, seasonal re-check untouched):

```python
INFO Feature drift [%s]: %d features, %d evaluated, %d STRUCTURAL, %d INSUFFICIENT, %d warnings, %d alerts, %d seasonal (…)
WARN Feature drift [%s]: 0 of %d features evaluable … (%d STRUCTURAL stable constants, %d INSUFFICIENT …) — drift is UNMEASURED, not clean
# WARN fires only when inability to judge leaves nothing evaluated (evaluated == 0 AND insufficient > 0).
# STRUCTURAL alone never pages — a stable constant is a verdict, like OK.
```

From the next run this delivery reads `70 features, 68 evaluated, 2
STRUCTURAL, 0 INSUFFICIENT, 0 warnings, 2 alerts, 2 seasonal` — reviewers
should expect the counts.

**Pins** (`nfl-backend/backend/test_production.py`, script-style `check()`
idiom — no new programs): a STRUCTURAL+judged fixture asserts the counts
sum (`evaluated + STRUCTURAL + INSUFFICIENT == features`) with no UNMEASURED
page; an all-INSUFFICIENT window (10-row current) asserts exactly one
`UNMEASURED, not clean` WARNING. The pre-existing view-label pin
(`args[1]` = view) is unaffected — the view stays the second format arg.

## Crash note (reviewed, superseded — no action)

`6ca8279e` delivered a crash log: `ValueError: lattice delta must align
with the distribution support` from `gaussian_family.py` (mirrored in all
four sports). The lattice-alignment commits (`601bbfed`, `9b49f2f3`) landed
after it, and THIS run completes all 14 phases with `DONE — 223 artifacts`
and bit-identical headline metrics to the pre-crash 22:50 run
(auc 0.69968…, weights 42.4/21.0/36.6 — deterministic full repull). The
crash cause is fixed in-tree; hardening the guard itself belongs to the
active lattice work (probes in flight), not this review.

## Claims reconciliation — spot

| Log claim | Artifact | Verdict |
|---|---|---|
| `decided games: warmup 267, core 2573`; `feature frame: 2840 decided games` | 267 + 2573 = 2840 | ✓ |
| `folds: 200 windows, provisional 92/869, dropped 0`; `last OOF 2026-10-14 fold 199` | fold geometry block | ✓ |
| `moneyline OOF: 2573 scored, 1704 grading / 111 postseason / 869 provisional`; `OOF raw auc 0.69968…` | identical to the 22:50 run — deterministic | ✓ |
| `final serving calibrator: identity (gated_no_gain)` | calibration gate lines | ✓ disclosed |
| `slate games: 1`; `SHAP attributions written for 1 games`; `SHAP cards: 1 written (209 on disk)` | single-game slate served, history retained | ✓ |
| `cards history store: 2573 rows (0 added this run)` | no new cards decided | ✓ assessed, acceptable |
| drift `0 warnings, 2 alerts, 2 seasonal` (both views) | CSV 64 OK + 2 OK-SEASONAL + **2 STRUCTURAL** + 2 ALERT | ⚠ 2 STRUCTURAL invisible → **Finding 1** |
| `Rolling Brier: 556 points (2017-09-07 → 2026-10-12), 2573 games, 0 sparse days, mean 0.2208` | points span full OOF window | ✓ |
| `artifacts written: 223`; `DONE — 223 artifacts in 1115s`; retention pruned 3 stale `*20261004*` | persistence block | ✓ |

## Verification

- `python nfl-backend/backend/test_production.py` → **565 passed, 0 failed**,
  exit 0 (562 baseline + 3 new pins), re-run after the final edit. One
  first-pass pin failure was a test-index bug (format args shifted by the
  new counts), fixed at its cause — the `args[1]`-is-view convention the
  older pin relies on is preserved.
- `pytest test_log_review_20261011.py test_run_log_tee.py` → **43 passed**
  (no old-headline assertions anywhere — grep-verified; neither file
  touches `feature_drift`).

## Deliberately not changed

- PSI floors, WARN/ALERT thresholds, the season-seam re-check, STRUCTURAL
  semantics — disclosure only.
- Whether `inj_ol_out_*` SHOULD alert — a modeling question; the log names
  the evidence.
- The `DtypeWarning: Columns (25) have mixed types` in Phase 14
  (`master_pipeline.py:1679` `pd.read_csv`) — a pandas stderr notice, not a
  log line; fragile typing worth a hygiene pass, but out of scope here.
- The lattice guard (`gaussian_family.py`) — fixed by the lattice commits;
  active work continues elsewhere.
