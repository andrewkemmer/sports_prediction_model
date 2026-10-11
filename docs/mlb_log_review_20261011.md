# MLB run-log review — 2026-10-11

Scope: the delivered `mlb-backend/data_delivery/mlb_pipeline_run_log.txt` — 453 lines,
run **2026-10-11 05:43**, delivered by `d0e52126` (artifacts) / `f740b886` (stale
removal) / `dc9d2aee` (final log) on remote tip `76ea3e43`.

Companion: `docs/pipeline_log_review_20261011.md` (four-sport) covers the **01:26**
MLB run — evidence: it quotes the `MLB_END_DATE=2026-10-10 is stale` WARNING that the
01:26 log carries (1 match) and this one does not (0 matches), and cites remote tip
`6a1fdd4e80`. This review covers the **05:43** delivery and reconciles every material
count against the delivered artifacts.

**Verdict: 0 errors/exceptions; 7 WARNING lines, all classified below; 1 finding
remediated (the rolling-Brier line read as current while its series was 11 days
stale); 1 apparent anomaly explained and NOT a defect (the same-day rerun delta).**

## Log line review — all 7 WARNINGs

| # | Line | Message (abridged) | Verdict | Disposition |
|---|------|--------------------|---------|-------------|
| 1 | 79 | `missing-finals guard: frame horizon 2026-10-08 but 1 official final(s) newer than the frame are ABSENT … 2026-10-10 CWS@CLE (849831)` | Guard working as designed (T-guard, `test_log_review_20261008`). Savant posting lag: the result went official **between** the 01:26 run (which logged "no finals beyond the frame") and the 05:43 run, so the guard correctly escalated. | None. Recovered by the next run's forward top-up + 3-day tail refresh. |
| 2 | 309 | `Calibration: degenerate Platt params (a=-0.2434, n=448) — identity map` | Working as designed since 2026-10-06: a non-positive slope would invert probabilities, so identity ships. | None. |
| 3 | 322 | `SP slate staleness gate: 146 pitcher(s) enter 2026-10-11 off a >=10d availability-stint gap` | Gate working; pre-stint form withheld rather than served. Today's announced starters are **not** gated: `todays_games_20261011.csv` carries complete `sp_*` lines for both (Misiorowski vs Skubal). | None. |
| 4 | 338 | `Calibration: degenerate Platt params (a=-0.3546, n=374) — identity map` | Same class as #2 (run-engine markets). | None. |
| 5 | 339 | `Calibration: degenerate Platt params (a=-0.3166, n=467) — identity map` | Same class as #2. (Line 343 discloses 305 of 6140 prequential fits degenerated — disclosed, bounded, by design.) | None. |
| 6 | 353 | `Feature drift [moneyline]: 0 of 109 features evaluable … drift is UNMEASURED, not clean` | T15 remediation working. The CSV confirms `feature_drift_20261011.csv` = 109/109 INSUFFICIENT — the **8th consecutive** unmeasured delivery (20261004 → 20261011; 20261003 was the last OK run). Trailing 7-day window sits under the 30-row judge floor now that the regular season ended. | None — judge floors are owner policy; the WARNING is the disclosure. |
| 7 | 358 | `Feature drift [run-engine]: 0 of 109 features evaluable …` | Same class as #6, second view. | None. |

No `ERROR`, `Traceback`, or failed-artifact line appears anywhere in the log.

## Finding 1 (remediated): the rolling-Brier line read current while the series was 11 days stale

Delivered line 334, verbatim:

```
INFO Rolling Brier: 511 points (2024-04-25 → 2026-09-27), 7031 games, 71 sparse days excluded (<5 games/day), mean 0.2442
```

Reconciled against `rolling_brier_20261011.json` and the history frame
(`predictions_history_20261011.csv`, 7031 rows):

- **7031 is `n_games_total`, not the series' game count.** Only **6888** games entered
  any series point (`n_games_in_series`, present in the artifact since its inception —
  the log line simply never printed it); **143 games on 71 sparse days** were excluded.
- **Nothing said where the series ends relative to the data.** The frame's newest
  decided game is **2026-10-08**; the last point is **2026-09-27** — an **11-day** gap
  whose trailing **24 games on 9 days** (the entire postseason tail) all sit under the
  5-game floor and therefore enter **no** point. Every one of those days was silently
  dropped, yet the line read as current through the run date.
- This is the same escaped-defect class as T15/T16 (2026-10-10 review): a monitor line
  reading healthy while covering less than it implies.

**Why the suite could not catch it:** until this review, no test called
`compute_rolling_brier` — the only test reference to it anywhere was the artifact
*filename* in `test_log_review_20261007.py`'s manifest list. The log line's contents
were never asserted.

**Remediation** (`mlb-backend/backend/explainability.py`, `compute_rolling_brier` —
message/disclosure only; no threshold, window, or exclusion change):

- the INFO line now publishes in-series coverage and labels the mean for what it is:
  `Rolling Brier: 511 points (2024-04-25 → 2026-09-27), 6888 of 7031 games in-series (143 excluded on 71 sparse days, <5 games/day), all-game mean 0.2442`
- a **new WARNING** fires whenever decided games exist past the last point, naming the
  gap and the excluded tail (quoted from the real-data run below);
- the artifact gains `excluded_sparse_games`, `frame_newest_game_date`,
  `last_series_date`, `tail_games_after_last_point` so the JSON reconciles with the
  line (`n_games_in_series` was already there);
- `ROLLING_BRIER_MIN_GAMES_PER_DAY = 5` and `ROLLING_BRIER_WINDOW_DAYS = 30` are
  **unchanged** — judge floors are the owner's decision; disclosing what was *not*
  judged is ours (same boundary drawn at T15).

Real-data verification — executing the remediated function against the delivered
7031-row history frame:

```
INFO   Rolling Brier: 511 points (2024-04-25 → 2026-09-27), 6888 of 7031 games in-series (143 excluded on 71 sparse days, <5 games/day), all-game mean 0.2442
WARNING Rolling Brier: series is STALE — last point 2026-09-27 is 11 day(s) behind the newest decided game 2026-10-08; 24 later game(s) on 9 day(s) stay under the 5-game floor and enter NO series point, so the trailing monitor does not describe the current window
```

## Same-day rerun delta — reviewed, NOT a defect

The 01:26 run and the 05:43 run share the identical input frame yet shipped different
fold metrics and different deployed blend weights. That pattern (identical inputs →
different outputs) normally means non-determinism; here it is fully explained by a
config change that landed **between** the runs:

| Evidence | 01:26 run (`59182be6`/`5f624714`) | 05:43 run (`d0e52126`/`dc9d2aee`) |
|---|---|---|
| `Pipeline snapshot` (line 208/209) | `7402 decided games, hash=63bb8dc962ce` | `7402 decided games, hash=63bb8dc962ce` — **identical** |
| Fold lines (`Fold N […] auc=… brier=…`) | 83 lines | 83 lines; **79 differ in AUC, none byte-identical** |
| `Rolling blend weights (last fold earned)` | `{'elasticnet': '32.6%', 'lightgbm': '50.0%', 'xgboost': '17.4%'}` | `{'elasticnet': '44.0%', 'lightgbm': '28.0%', 'xgboost': '28.0%'}` |
| Delivered artifacts carried in both | 252 files → **234 byte-identical, 18 content-changed** | changed set is exactly the model-dependent artifacts (`models/ensemble_latest.joblib`, `calibration_`, `model_history`, `model_version_history`, `predictions_history`, `rolling_brier`, `power_rankings`, `run_engine_*`, `shap_game_*`, `todays_games`, drift CSVs) |

Root cause: **`5b703ca4`** ("MLB: L2 regularization parity … LGBM `reg_lambda`
0.0027→5.0, XGBOOST `reg_lambda` pinned explicitly to 5.0") landed between the two
deliveries — `git merge-base --is-ancestor 59182be6 5b703ca4` confirms the 01:26 run
predates it. Different regularization ⇒ different fold fits ⇒ different last-fold
weights. Two corroborating signals that this is a *config* delta and not
non-determinism:

- 234 of 252 artifacts are **byte-identical** across the runs — everything not
  downstream of model fitting is bit-stable;
- the 18 changed files are exactly the fitted-model outputs and their reports.

No code change; the determinism blueprint (fixed folds/seeds) is intact.

## Claims reconciliation — every material count against the delivered artifacts

| Log claim | Artifact | Verdict |
|---|---|---|
| L208 `Pipeline snapshot: 7402 decided games, hash=63bb8dc962ce` | same hash on both same-day runs | ✓ |
| L316 `7031 OOF rows = 6879 grading + 152 non-grading (144 provisional incl. 106 postseason, 8 postseason outside provisional folds)` | `predictions_history_20261011.csv` = 7031 rows | ✓ (6879+152=7031; non-grading 152 = 144 provisional + 8 outside; postseason 114 = 106 in-provisional + 8 outside) |
| L317 walk-forward `auc 0.5752, brier 0.2443, logloss 0.6815, ece 0.0088, calibrator_gated_out: True` | L313 final-origin gate: `decision: identity, reason: gated_no_gain, n_prior 6879` | ✓ identity gate consistent with the gated-out metric column |
| L319 `Version history: 20 versions recorded (v2026.10.11 latest)` | `model_version_history.json` changed in this delivery | ✓ |
| L324 `Upcoming slate built: 1 games for 2026-10-11` | `todays_games_20261011.csv`: 1 row `20261011_LAD@MIL`, `game_state=pre`, `sp_*` complete both sides, model pick MIL 0.5273 | ✓ |
| L341 `Run engine markets: 7032 rows` | `run_engine_markets_20261011.csv` (changed this delivery) | ✓ |
| L344 `Run-engine monitor: 3 winner cards, rolling 45 days` | `run_engine_monitor_20261011.json` | ✓ |
| L334 `Rolling Brier: 511 points … 7031 games, 71 sparse days excluded, mean 0.2442` | JSON: 511 points, `n_games_total` 7031, `excluded_sparse_days` 71, `history_mean_brier` 0.244243 — **but `n_games_in_series` 6888** | ⚠ numbers reconcile; the line's *implied coverage* did not → **Finding 1 (remediated)** |
| L352/L357 drift `109 features, 0 evaluated, 109 INSUFFICIENT` (both views) | `feature_drift_20261011.csv` 109/109 INSUFFICIENT (8th consecutive) | ✓ |
| L354/L360 coverage `… 63 unmeasured rows: 48 closed-roof policy zeros + 15 open-air NULLs`, `baseline=75%` | `feature_coverage_20261011.csv`: 218 rows = **216 OK + 2 STRUCTURAL** (both baseline windows of the two weather features: `pct_nonnull` 94.0, `pct_measured` 74.8→"75%", `n_default_zero` 48, 250−235=15 NULLs) | ✓ (log's "75%" is `pct_measured`, matching the prior review's line semantics; 10-10 was 214+4 — the two *current*-window rows are now OK) |
| L109 `SP staleness gate: 63501 appearance rows, 2400 flagged stale` | gate counters | ✓ (INFO context for WARNING #3) |
| L79 `frame horizon 2026-10-08 … CWS@CLE (849831) ABSENT` | frame horizon corroborated by L110 (`lag -3d vs data horizon 2026-10-08`) and by `frame_newest_game_date` = 2026-10-08 in the rolling-Brier artifact | ✓ |
| 14 pull chunks → sum 2,292,817 pitches | arithmetic over the log's own chunk lines | ✓ |
| L435/444 `Removing 8 stale files … confirmed` | `f740b886` deletes exactly 8 files (all `*20260930*`) | ✓ |
| Final log delivery L452 | `dc9d2aee` touches only `mlb_pipeline_run_log.txt` | ✓ |

Coverage-STRUCTURAL reconciliation detail (both views identical): the current window of
each weather feature is OK this run; only the two baseline windows remain STRUCTURAL —
declared policy, exhaustively enumerated (48 indoor-neutral zeros + 15 SP-gated NULLs).

## Verification

- `python -m pytest -q -x` in `mlb-backend/backend` → **393 passed, 1 skipped, exit 0**
  (includes the two new pins below).
- **T17 pins** (appended to `test_log_review_20261008.py` per the no-new-test-file
  convention):
  - `test_rolling_brier_line_publishes_in_series_coverage_and_stale_tail` — synthetic
    history with a sparse head, dense block and sparse tail: asserts the INFO line
    prints `60 of 68 games in-series` (and that the gross `", 68 games,"` phrasing
    cannot come back), the STALE WARNING names both dates and the tail counts, and the
    artifact carries all four disclosure keys;
  - `test_rolling_brier_stays_quiet_when_the_series_reaches_the_newest_game` — dense
    final slate: coverage still disclosed, **no** false STALE alarm.
- Real-data execution against the delivered 7031-row history frame (quoted above).

## Deliberately not changed

- `ROLLING_BRIER_MIN_GAMES_PER_DAY = 5` / `ROLLING_BRIER_WINDOW_DAYS = 30` — owner
  policy; only the disclosure of what the floor excludes changed.
- Drift window and judge floors (owner policy) — the UNMEASURED WARNING already
  discloses the consequence every run.
- Degenerate-Platt threshold and identity fallback (by design since 2026-10-06).
- Guard postures: missing-finals **warns, never aborts** (transient Savant lag);
  SP staleness gate withholds rather than guesses.
- `frontend/model_monitor.py` and the monitor JSON's `rolling_brier_meta`: untouched.
  The panel never claims currency — its caption already states the floor and the
  excluded sparse-day count, and the baseline label (`History mean (N games)`) is
  accurate over the full history. The new disclosure keys live in the
  `rolling_brier_*.json` artifact.

## Residual risk / next-run expectations

- The new `series is STALE` WARNING will fire on **every** postseason and off-season
  run while trailing slates stay under the 5-game floor. That is correct-by-design —
  the same posture adopted at T15 (a monitor that cannot describe the present must say
  so) — but reviewers of the next logs should expect it rather than treat it as a
  regression.
- `CWS@CLE (849831)` remains unresolved until the next run's forward top-up + tail
  refresh (guard's own stated recovery path).
- The `MLB_END_DATE` notebook pin (the four-sport review's one open, external action)
  did not trip this run (0 matches vs 1 in the 01:26 log). The pin lives in the
  Kaggle-owned notebook, outside this repo — re-date/unset it there if it recurs.
- `mlb-backend/data/` (pitches.parquet) is not committed, so frame-level claims were
  verified against the delivered `data_delivery` artifacts and the log's own internal
  arithmetic; no raw-pitch recount was possible in this workspace.
