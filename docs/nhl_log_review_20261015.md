# NHL run-log review — 2026-10-15 evening delivery (2026-10-16 01:25 UTC run)

Scope: the delivered `nhl-backend/data_delivery/nhl_pipeline_run_log.txt` — 364
lines, run **2026-10-16 01:25:35 → ~01:43 (≈1040s)**, `NHL_FULL_REPULL=1`,
committed on remote tip `0cd53129` (prior tip; no NHL commits since `da37c794`).

**Verdict: 0 ERROR / 0 Traceback; 4 WARNING lines, all working as designed
except Finding 1 (remediated below: the drift summary counts 2 alerts without
naming them).**

## Log line review — all 4 WARNINGs

| Line | Message (abridged) | Verdict | Disposition |
|---|---|---|---|
| 239 | `OOF validation population is thin on 14 of 61 window(s) (< MIN_VAL_FOLD_GAMES=40) … retained for evidence, but PROVISIONAL` | Working as designed — playoff weeks and season ramps genuinely thin; 178 provisional rows disclosed, never grading. Arithmetic: `2518 grading + 168 postseason + 178 provisional` with overlap (2707 scored of 2899 eligible, 93.4% reach). | None. |
| 347 | `coverage [baseline]: 6 feature(s) carry default-filled values counted as UNMEASURED — a warm gap, not a warm-up (+18 postseason-boundary)` | The 2026-10-09 remediation working: hole vs documented source boundary split, features named. | None. |
| 351 | `coverage [current]: 3 feature(s) … (pl_ppo_r_away/home/diff)` | Same class, current window; features named. | None. |
| 354 | `coverage [serving slate]: 3 feature(s) … (1/1/2 default-filled)` | Same class, slate window; features named. | None. |

The 2026-10-11 remediation is working: both drift views read `68 features, 64
evaluated, 4 INSUFFICIENT, 0 warnings, 2 alerts, 2 seasonal` — counts sum to
the CSV (64+4=68), reconciled exactly against
`run_engine_feature_drift_20261015.csv`
(60 OK + 2 OK-SEASONAL + 4 INSUFFICIENT + 2 ALERT).

## Finding 1 (remediated): the drift summary counts alerts it never names

Both drift views report `2 alerts`, but no line says which features. The CSV
beside the log holds:

- `goalie_starts_home` — current mean 3.08 vs baseline 40.58, psi_adj 11.07 vs
  noise floor 0.094, location shift, ALERT;
- `goalie_starts_away` — 3.27 vs 40.44, psi_adj 11.21 vs 0.094, ALERT.

Shape assessment: the baseline carries last season's accumulated starts (~40)
while the current window sits in the new season (starts reset, ~3) — the
season-boundary reset the season-seam guard exists to seasonalize, but the
phase re-check could not fire it clean, so ALERT stands. Whether the verdict
is right is owner policy and untouched; the defect is disclosure-only, same
class as the 2026-10-11 INSUFFICIENT finding: a reader reconciling the line
against the CSV must open the file to learn which rows paged, while the
coverage ledger in the SAME phase names its features
(`pl_ppo_r_away(6 default-filled) …`).

**Remediation** (`nhl-backend/backend/monitoring.py`, `feature_drift` —
message only; thresholds, floors, and the INSUFFICIENT-never-pages rule
untouched, MLB parity):

```python
logger.info("Feature drift [%s]: %d paged row(s) — %s", view, len(paged), desc)
# desc: "goalie_starts_home:ALERT (psi_adj 11.069 vs floor 0.094), …" (cap 20)
```

INFO, not WARNING, by design: the warning budget stays exactly as the
2026-10-11 review pinned it (4 WARNING lines), and MLB's summary treats
alerts the same way (counts at INFO). From the next run the log reads e.g.
`Feature drift [moneyline]: 68 features, 64 evaluated, 4 INSUFFICIENT, 0
warnings, 2 alerts, 2 seasonal (…)` followed by the paged-row census.

**Pins** (appended to `test_run_engine_pit.py`, the file's established
drift-summary pattern — no new programs):

- `test_drift_summary_names_paged_alert_rows` — a +50 shift on one column
  yields ALERT; the follow-up INFO names `rest_days_home:ALERT` with
  `psi_adj`/`floor`, and never names the unpaged column;
- `test_drift_summary_silent_on_names_when_nothing_paged` — a clean judged
  window emits no naming line.

## Claims reconciliation — spot

| Log claim | Artifact | Verdict |
|---|---|---|
| `score-date span clipped: 681 dates (dropped 273)`; `score dates resolved: 681 fetched, 0 cache hits` | full-repull shape (`NHL_FULL_REPULL=1`) | ✓ |
| `decided games: core 2899`; `feature frame: 2899 decided games, 148 columns` | frame size | ✓ |
| `folds: 61 windows, provisional 14/178, dropped 0`; `OOF reach: 2707 of 2899 (93.4%)` | fold geometry block | ✓ |
| `moneyline OOF complete: 2707 scored, 2518 grading / 168 postseason / 178 provisional, weights 0.479/0.224/0.297` | `folds oof blocks` + `moneyline OOF rows` lines | ✓ |
| `prequential per-fold calibration: 0 fitted, 61 identity`; `final pooled calibrator: identity (gated_no_gain)` | calibration gate lines | ✓ identity served, disclosed |
| `slate games: 11`; `SHAP attributions written for 11 games` | 11-game 2026-10-15 slate + SHAP set | ✓ |
| `cards history store: 2716 rows (0 added this run)` | `nhl_production_cards_history.csv` unchanged this run — assessed, acceptable (no new cards decided; the 12:39 run carried the day's adds) | None. |
| `11 gates … PASS` (PHASE 13) | schema-validation block | ✓ |
| drift `64 evaluated + 4 INSUFFICIENT = 68` both views | drift CSV 60 OK + 2 OK-SEASONAL + 4 INSUFFICIENT + 2 ALERT | ✓ counts; ⚠ alerts unnamed → **Finding 1** |
| `DONE - 14 artifacts written in 1040s` | persistence-phase counter | ✓ |

## Verification

- Targeted: `pytest test_run_engine_pit.py -k "drift_summary or drift_logs_view
  or drift_calls_receive or regime_shift"` → **7 passed** (2 new + 5
  pre-existing), exit 0.
- Full NHL suite not rerun (the 2026-10-11 review records 619 passed across
  three chunks at ~9.5 min; the change is additive logging after the summary
  emit — no computation, threshold, or artifact path touched).

## Deliberately not changed

- PSI floors, WARN/ALERT thresholds, the season-seam re-check, and
  `INSUFFICIENT never pages` — owner policy / MLB parity; only the naming of
  what paged changed.
- Whether `goalie_starts_*` SHOULD alert at the season seam — a modeling
  question for the owner; the log now surfaces it instead of hiding it.
- `nfl-backend`'s pre-T15 summary line — flagged in the 2026-10-11 review for
  a dedicated NFL pass, not fixed here.
