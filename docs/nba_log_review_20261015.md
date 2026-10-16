# NBA run-log review — 2026-10-15 22:22 delivery

Scope: the delivered `nba-backend/data_delivery/nba_pipeline_run_log.txt` — 256
lines, run **2026-10-15 22:22:39 → 22:52 (~1,673s)**, `NBA_FULL_REPULL=1`,
delivered by `b6869eda` (artifacts) / `fc4ba744` (final log) on remote tip
`5b5ab087` (prior tip; no NBA commits since).

**Verdict: 0 ERROR / 0 Traceback; every WARNING classified below; 1 finding
remediated (the drift headline has no evaluated/INSUFFICIENT slot — the exact
pre-T15 shape remediated twice before); every material count reconciles
against the delivered artifacts.**

## Log line review — WARNINGs

| # | Lines | Message (abridged) | Verdict |
|---|---|---|---|
| 8× | chunks | `dropping <id> (EAST vs WEST / CHK vs KEN / STARS vs WORLD …): not one of the 30 franchises` | By design — All-Star exhibitions unscorable; reason named. |
| 2× | schedule | `27 scheduled game(s) have no season-log match` / `27 game(s) carry no NBA game id` | Same 27 games from two angles; reconciles as 11 postponed + 2 finished without lines + 14 upcoming slate (2026-10-20/21), per the 10-14 decomposition. |
| 2× | positions | `position index covers only 135/572 (24%) / 131/569 (23%), floor 90%; keeps G/F/C` | Disclosed fallback; 2025-26 reads 100%. |
| 1× | designations | `no filing lists DAL@GSW before tipoff on 2024-01-19: game uncovered` | Disclosed single-game archive gap; the 14 slate games' `pre-submission` lines are by design. |
| 1× | folds | `OOF thin on 16 of 56 windows (< 40): PROVISIONAL, excluded from pooled metrics/blend/calibration` | Working as designed. 2400 = 2130 grading + 169 postseason + 101 provisional (overlap-aware). |
| 1× | drift | `newest decided game is 2026-06-13 — 124 day(s) before this run (floor 14); … NOT the slate … UNMEASURED … not new signal` | The 2026-10-11 remediation firing (age 120 → 124d). The 10 alerts below repeat the frozen comparison. |

The 2026-10-14 remediation is working: all three sweeps name ceilings —
schedule `8.99 day/s against a 10.00/s gate (90%)`, play-by-play `4.00
game/s against a 4.00/s gate (100%)`, season log `2.02 slice/s against a
2.00/s gate (101%)`.

## Finding 1 (remediated): the drift headline cannot show an unjudged row

The run printed:

```
feature drift: 71 features, 0 warning(s), 10 alert(s); statuses on noise-adjusted PSI with a location gate [alerts: …]
```

against `nba_run_engine_feature_drift_20261015.csv` = 61 OK + 10 ALERT = 71 —
counts sum today, so nothing is hidden *this* run. But the status enum HAS an
`INSUFFICIENT` state (`monitoring.py`: empty windows, or `n_b < 100` / `n_c <
30`), and the headline has no slot for it: the first small-window run of the
2026-27 season (slate games tip 2026-10-20/21; the window is frozen at a
31-row June tail today) would print `71 features, 0 warning(s), 0 alert(s)`
while judging nothing — the exact pre-T15 shape MLB carried for 7 runs and
NHL fixed on 2026-10-11. Same class, same fix, before it can bite.

**Remediation** (`nba-backend/backend/monitoring.py` — message only;
statuses, windows, floors untouched):

```python
INFO feature drift: %d features, %d evaluated, %d INSUFFICIENT, %d warning(s), %d alert(s); …
WARN feature drift: 0 of %d features evaluable … — drift is UNMEASURED, not clean  # only when nothing judged
```

The dedupe key (`_LAST_LOGGED_DRIFT`, one INFO per distinct verdict across
the feature-report + monitor-JSON passes) carries the new counts, so
identical repeats still collapse to DEBUG. INSUFFICIENT alone still never
warns. From the next run the line reads e.g. `71 features, 71 evaluated, 0
INSUFFICIENT, 0 warning(s), 10 alert(s)` — reviewers should expect the
counts.

**Pins** (appended to `test_log_review_20261011.py` — no new programs):
`test_drift_headline_publishes_evaluated_and_insufficient_counts` (one NaN
column → `2 features, 1 evaluated, 1 INSUFFICIENT`, no page) and
`test_drift_headline_pages_when_nothing_was_judged` (10-row window → `0
evaluated` + exactly one `UNMEASURED, not clean` WARNING).

## Claims reconciliation — spot

| Log claim | Artifact | Verdict |
|---|---|---|
| `NBA facts: 3488 games, 6922 team rows, 74449 player rows, 1712728 actions, 6922 event rows` | 3488 × 2 = 6976 slots; events = 3461 decided × 2 | ✓ byte-identical inputs to the 10-14 run (deterministic full repull) |
| `pl_rapm attached: 3266/3461`; gap `195 = 104 + 19 + 72` | 3266 + 195 = 3461 | ✓ exact |
| `Published blend: 2400 OOF rows; weights 65.2/30.2/4.6`; retrospective `n=2130` | importance shares 0.6522/0.3021/0.0457 | ✓ |
| `feature drift: 71 features, 0 warning(s), 10 alert(s) [alerts: …10 named]` | drift CSV 61 OK + 10 ALERT | ✓ counts sum; ⚠ no INSUFFICIENT slot → **Finding 1** |
| `14 upcoming game(s) scored`; `SHAP for 14 game(s)` | 14 `nba_shap_game_401909*.csv` (088-092, 834-842) | ✓ |
| `projected lineups: 6976 team-game(s) (22 priced, 6954 pool-refusal)`; lineup diffs `11/3488 rows` | per-10-09 split-count convention; SUPERSEDED family, verdict in monitor artifact | ✓ disclosed, not silent |
| `run-line fit check: pearson 1.27 dev 1.28 (baseline 1.46)` | `pearson_poisson_adequacy` = χ²/df (docstring: ≈1 → Poisson variance adequate) — a variance diagnostic, not a correlation; model beats baseline | ✓ assessed, not a defect |
| `rolling Brier: 0.2091 … last day 2026-06-13 0.3435 over 1 game(s)` | single-game tail day disclosed | ✓ |
| `✅ status ok, 1673.48s` | log span 22:22:39 → ~22:52 | ✓ |

Also reviewed, no defect: the `(+2 more)` truncation on the 27-game lists
(decomposition verified 10-14); `lineup_rapm_rest_count_diff … CONSTANT`
(disclosed young-feature shape); `season log 74449` repeating 10-14
(determinism, not staleness).

## Verification

- `test_log_review_20261011.py` → **6 passed** (4 existing + 2 new), exit 0.
- `test_run_log_tee.py` → **24 passed**; no other suite asserts the old
  headline (grep-verified `warning(s)` across NBA tests).
- Full NBA backend suite not rerun (10-11 review records 629 passed; the
  change is log-annotation-only in one emitter plus its pins).

## Deliberately not changed

- Drift statuses, floors, windows, composition matching, the frozen-window
  staleness warning — disclosure only.
- The other thread's active NBA work (`config.py`, `distributions.py`,
  `serving.py`, `test_nba_rapm_features.py`, frontend diagnostics) —
  untouched; this change confines itself to `monitoring.py` + the existing
  review test.
- A truncated 5-line local overwrite of the run log found in the working
  tree (a different checkout's fragment, `E:\…` paths) was restored via
  `git checkout` — not reviewed, not committed.
