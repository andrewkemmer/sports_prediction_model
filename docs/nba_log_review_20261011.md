# NBA run-log review — 2026-10-11

Scope: the delivered `nba-backend/data_delivery/nba_pipeline_run_log.txt` — 254
lines, run **2026-10-11 05:20:34 → 06:11:35 (3,061s)**, `NBA_FULL_REPULL=1`,
delivered by `8f6f4b2d` ("Update NBA features + predictions") + `323aceac`
("final delivery").

**Verdict: 0 ERROR / 0 Traceback; 14 WARNING lines, all classified below; 2
findings remediated (the drift monitor's off-season window went silently
stale; the dispersion line's "(curve max)" parenthetical reads as a violated
bound); 1 pre-existing environment-dependent test failure repaired; every
material claim reconciled against the delivered artifacts.**

## Log line review — all 14 WARNINGs

| Line | Message (abridged) | Verdict | Disposition |
|---|---|---|---|
| 8× across chunks | `dropping <id> (EAST vs WEST / CHK vs KEN / STARS vs WORLD …): a side is not one of the 30 franchises` | By design — the four 2024 All-Star, three 2025 All-Star (skills-challenge) and four 2026 All-Star-format games are not scoreable by this model; the drop names the reason. | None. |
| schedule | `27 scheduled game(s) have no season-log match …` and `27 game(s) carry no NBA game id …` | Same 27 games seen from two angles; reconciles exactly: **11 postponed** (excluded from the slate, listed at 06:01:19) **+ 2 finished without player lines** (MIL@OKC 2024-12-17, SAS@NYK 2025-12-16 — excluded from training, disclosed as postponed/neutral-site, not a join failure) **+ 14 upcoming slate games** (2026-10-20/21, obviously not yet played). 11+2+14 = 27. | None. |
| position index ×2 | `covers only 135 of 572 2023-24 players (24%, floor 90%); this season keeps the G/F/C convention` (and 2024-25 at 23%) | By design — the position archive predates those seasons; the fallback convention is disclosed rather than silently applied. 2025-26 reads 100%. | None. |
| designations | `no filing lists DAL@GSW before tipoff on 2024-01-19: game uncovered` | Disclosed archive gap for one 2024 game; the 14 upcoming games' `pre-submission` lines are by design (filing windows close near tipoff). | None. |
| folds | `OOF validation population is thin on 16 of 56 window(s) (< MIN_VAL_FOLD_GAMES=40) … PROVISIONAL` | Working as designed — playoff weeks and season ramps; provisional rows never grade the blend. | None. |

## Claims reconciliation — every material count against the artifacts

| Log claim | Against | Verdict |
|---|---|---|
| `NBA facts: 3488 games … 6922 event rows`; `projected lineups: 6976 team-game(s)` | 3488 × 2 = 6976; event rows = 3461 decided × 2 = 6922 | ✓ |
| `pl_rapm features attached: 3266/3461`; `195 lack … 104 + 19 + 72` | 3266 + 195 = 3461; 104+19+72 = 195 | ✓ exact |
| `Published blend: 2400 OOF rows … weights elasticnet 65.2 / lightgbm 30.2 / xgboost 4.6`; retrospective replay `n=2130` | monitor JSON ensemble (0.6522/0.3021/0.0457, n_eval 2130); 2400 = 2130 grading + 169 postseason + 101 provisional | ✓ |
| `sealed holdout gate: … 2389 pre / 11 sealed rows` | 2389 + 11 = 2400 | ✓ |
| `feature drift: 71 features, 0 warning(s), 10 alert(s) [alerts: …]` | `nba_run_engine_feature_drift_20261011.csv`: 71 = 61 OK + 10 ALERT — counts **sum exactly** (no hidden statuses; the T15 class is absent) | ✓ (see Finding 1 for the window) |
| `SHAP attributions written for 14 game(s)`; `14 upcoming game(s) scored` | slate = the 2026-10-20/21 games in the 27-list | ✓ |
| `✅ status ok, 3061.11s` vs log span 05:20:34 → 06:11:35 | 3,061s | ✓ |
| `rolling Brier: 0.2091 … 2400 game(s)`; `designation archive published: 37211 record(s)` = loaded 37,211; `event rollup: 6922` | monitor JSON / archives | ✓ |

Also reviewed, no defect: `alpha_home 0.0040 (curve max 0.0034)` — reconciles
once read as the two different estimators it is (pooled MoM scalar vs
per-row curve diagnostic; see Finding 2 for the wording fix); the lineup
features' `11/3488 rows … CONSTANT` disclosure (the feature is young and
mostly pool-refused historically — disclosed, not silent); the drift
window's widening + composition-matching (both logged, both careful).

## Finding 1 (remediated): the drift monitor's "current" window was silently frozen

The run printed:

```
drift window: only 3 decided game(s) in the last 7 day(s); widened to 37 day(s) for a current window of 31
drift window: current window is 100% is_playoffs against a 22% prior; baseline composition-matched to 220 is_playoffs row(s) …
feature drift: 71 features, 0 warning(s), 10 alert(s); …
```

The window builder did its job — but the window's **newest decided game was
2026-06-13, 120 days before the run**. The same 10 ALERTs (`elo_home`,
`elo_away`, three `ewm_*`, `event_live_tov_rate_home`, four `pl_rapm_*`)
had been printed identically for weeks: the delivered CSVs for
20261006–20261011 carry the same alert set. Nothing in the log said the
comparison was frozen, so a reader reasonably took "10 alerts" as a
statement about the October slate being served — a comparison the June
tail structurally cannot make (off-season roster and trade movement is
invisible to it). Same disclosure family as MLB's T15 / NHL's hidden
INSUFFICIENT: the line omitted the fact that changes its interpretation.

**Remediation** (`nba-backend/backend/monitoring.py`, disclosure only —
statuses, windows and thresholds untouched): `drift_windows` now takes an
optional `as_of` and, when the newest decided game is more than
`DRIFT_STALE_DAYS = 14` before the run, warns:

```
WARNING drift window: newest decided game is 2026-06-13 — 120 day(s) before this run (staleness floor 14); statuses measure that frozen 31-row tail against its baseline, NOT the slate being served. Off-season roster and trade movement is UNMEASURED until play resumes, and recurring alerts repeat the same frozen comparison — they are not new signal
```

14 days clears every in-season gap (the All-Star break is ~1 week; the
longest gap between decided NBA games is days), so the line fires only
when the clock has genuinely stopped.

**Pins** (`nba-backend/backend/test_log_review_20261011.py`, 4 tests):
the stale disclosure (window end, age, "NOT the slate being served",
"UNMEASURED", "not new signal"; windows unchanged); a fresh window stays
quiet; the boundary is exactly `DRIFT_STALE_DAYS` (quiet) vs +1 (warns);
and the default `as_of` is the wall clock.

## Finding 2 (remediated): "(curve max)" reads as a violated bound

`calibrated NB dispersion …: alpha_home 0.0040 (curve max 0.0034)` puts a
number *above* its own parenthetical, which reads as "the fit exceeded its
allowed range". The two numbers are different estimators — the shipped
scalar is the pooled method-of-moments estimate the draw path consumes;
`alpha_*_max` is the per-row maximum under the selected diagnostic curve
(different rows, different λ). Reworded in
`nba-backend/backend/master_pipeline.py` to `(per-row curve max …)` so the
parenthetical cannot be misread as a cap. NBA-only: MLB and NHL print no
such parenthetical (grep-verified).

## Finding 3 (repaired, pre-existing): the torn-line tee pin could not pass on Windows

`test_run_log_tee.py::test_torn_plain_line_waits_for_its_terminator`
failed in isolation on this machine: the test opened its capture file with
the locale default (cp1252 on Windows), and writing `"✅"` raised
`UnicodeEncodeError` — a `ValueError` subclass that `_Tee.write`'s
disk-failure guard swallows by design — so the file stayed empty and the
pin asserted against nothing. Production (`install_run_log_tee`) opens the
tee file `encoding="utf-8", errors="replace"`, so Kaggle deliveries are
unaffected; only the test diverged from the production contract. Fixed by
constructing the file and reading it back under `encoding="utf-8"`. The
same test exists only in NBA's suite (grep-verified across the other three
leagues).

## Verification

- `python -m pytest -q` in `nba-backend/backend` → **629 passed, exit 0**
  (625 baseline + 4 new pins), re-run after the final edit.
- The reconciliation tables above were executed against the delivered
  artifacts (pandas/JSON), not eyeballed.
- The staleness behavior was exercised against the real `drift_windows`
  with a synthetic decided frame and explicit `as_of` clocks — no live
  pipeline run.

## Deliberately not changed

- Drift statuses, judge floors, window construction, composition matching —
  disclosure only; the 10 alerts are a legitimate year-over-year
  playoff-tail signal once the window's age is disclosed.
- The 27-game disclosures listing upcoming slate games — noisy but honest;
  the "not swept" wording is correct for games without play-by-play.
- OOF provisional policy, calibration gates, lineup pool-refusal, the
  position-index fallback — all documented behavior.
- `mlb/nfl/nhl-backend` each carry their own copy of `drift_windows`; the
  same staleness gap exists there (their off-seasons will freeze the
  window too). Flagged for those leagues' next reviews, not fixed here.

## Residual risk / next-run expectations

- From the next run until the 2026-27 schedule produces decided games
  (~2026-10-21+), the log carries the `drift window: newest decided game …
  UNMEASURED` WARNING on every run — correct by design, not a regression.
  Once the season starts, the newest game is days old and the line
  disappears.
- If the same 10 alerts persist INTO the new season with a fresh window,
  that is real signal about the 2026-27 population and belongs to the
  model owner, not this review.
