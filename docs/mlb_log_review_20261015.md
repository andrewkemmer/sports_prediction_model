# MLB run-log review — 2026-10-15 22:24 delivery

Scope: the delivered `mlb-backend/data_delivery/mlb_pipeline_run_log.txt` — 443 lines,
run **2026-10-15 22:24**, delivered by `8ef658be` (artifacts) / `f9de118b` (final log)
on remote tip `9b49f2f3`.

**Verdict: 0 errors/exceptions; 4 WARNING lines, all working as designed except
Finding 1 (remediated below: the slate board log contradicts its own artifact).**

## Log line review — WARNINGs

| # | Line | Message (abridged) | Verdict |
|---|------|--------------------|---------|
| 1 | 79 | `missing-finals guard: frame horizon 2026-10-14 but 1 official final(s) newer than the frame are ABSENT … 2026-10-15 TB@CLE (849818)` | Guard fired correctly (Savant posting lag). Wording overclaimed — see Finding 1. |
| 2 | 309 | `Calibration: degenerate Platt params (a=-0.2477, n=448) — identity map` | Working as designed since 2026-10-06; non-positive slope ships identity. |
| 3 | 322 | `SP slate staleness gate: 147 pitcher(s) enter 2026-10-15 off a >=10d availability-stint gap` | Gate working; pre-stint form withheld. |
| 4 | 334 | `Rolling Brier: series is STALE — last point 2026-09-27 is 17 day(s) behind the newest decided game 2026-10-14; 30 later game(s) on 14 day(s) stay under the 5-game floor` | The 2026-10-11 remediation firing as designed. Artifact reconciles: 511 points, 6888 of 7037 in-series, 149 excluded on 76 sparse days, tail 30. |

No `ERROR`, `Traceback`, or failed-artifact line. Drift widening (`7d → 21d, 86 rows meets the floor 30`) is the 2026-10-13 remediation working: `feature_drift_20261015.csv` = 100 OK + 9 OK-SEASONAL (0 INSUFFICIENT), `feature_coverage_20261015.csv` = 214 OK + 4 STRUCTURAL (both weather features × both windows, declared policy). Retrospective blend replay labeled NOT OOF; calibrator gated out with identity.

## Finding 1 (remediated): the slate board log contradicts its own artifact

Delivered lines 324-325, verbatim:

```
INFO Upcoming slate built: 2 games for 2026-10-15
INFO No completed games on 20261015 -- built 2-game upcoming slate (pre-game PIT features)
```

Reconciled against `todays_games_20261015.csv` (2 rows):

- `20261015_TB@CLE` — `game_state=post`, `Final`, CLE 3-2, `home_win=1.0`
- `20261015_MIL@LAD` — `game_state=pre`, `Scheduled`

The board is **1 post + 1 pre**, not 0 completed / 2 upcoming pre-game. Root cause is
ordering: the slate line fires when the pitch-frame-derived board is built
(`target_games.empty` → `build_upcoming_slate`), but the official-results overlay
(`apply_official_results`) runs **after** it and promotes TB@CLE to post/Final —
line 333 confirms `Official results applied: 1 games verified/corrected`.
The same ordering made the guard's tail sentence false: it claimed the missing
final "will not resolve into predictions_history / today's record", yet today's
record DID resolve it (scores via overlay). `predictions_history_20261015.csv`
correctly holds 0 rows for 2026-10-15 (newest 2026-10-14) — pitch features were
absent, so history correctly waits for the next top-up + tail refresh.

Same escaped-defect class as T15/T16 and the 2026-10-11 Finding 1: a monitor line
reading complete while covering less than it implies.

**Remediation** (message/disclosure only; no threshold, window, or overlay change):

- `master_pipeline.py` — the pre-overlay slate line now scopes itself:
  `No completed games in the pitch frame on %s -- built %d-game upcoming slate
  (pre-game PIT features; official-results overlay follows)`;
- `master_pipeline.py` — immediately after the slate overlay, a new census line:
  `Today's board post-results: %d post/Final + %d pre/Scheduled (%d rows) --
  reconciles with todays_games CSV` (game_state when present, else home_win);
- `ingestion.py` — the guard tail now reads `will not resolve them into
  predictions_history (pitch features absent; today's board may still carry the
  official final via the results overlay); the next run's forward top-up +
  %d-day tail refresh recovers them into history`.

**Verification:** T19 pins appended to the existing `test_log_review_20261008.py`
(no new programs): guard wording, pre-overlay scope, post-results census wiring.
`test_log_review_20261008.py` 48 passed, 1 skipped; plus `test_log_review_20261007.py`
and `test_run_engine_contract.py`: 120 passed, 1 skipped.

## Claims reconciliation (spot)

| Log claim | Artifact | Verdict |
|---|---|---|
| `Pipeline snapshot: 7408 decided games, hash=e442f83b73f1` | `game_level_features.csv` frame | ✓ |
| `7037 OOF rows = 6879 grading + 158 non-grading (150 provisional incl. 112 postseason, 8 postseason outside)` | `predictions_history_20261015.csv` 7037 rows, 0 from 2026-10-15, newest 2026-10-14 | ✓ (6879+158=7037) |
| `Upcoming slate built: 2 games` + SHAP over 2 games | `shap_game_20261015_MIL@LAD.csv`, `shap_game_20261015_TB@CLE.csv` | ✓ count, ✗ state wording → Finding 1 |
| `Run engine markets: 7039 rows` | `run_engine_markets_20261015.csv` | ✓ |
| `Totals history store: 7339 rows (0 added this run)` | `run_engine_totals_history.csv` (17:54, unchanged by 21:18 run — no new totals decided) | ✓ assessed, acceptable |
| Drift `109 evaluated, 0 INSUFFICIENT` / coverage `214 OK + 4 STRUCTURAL` | both drift/coverage CSVs both views | ✓ |
