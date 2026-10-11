# NFL run-log review — 2026-10-11

Scope: the delivered `nfl-backend/data_delivery/nfl_pipeline_run_log.txt` — 656
lines, run **2026-10-11 05:20 → 06:12 (3,170s)**, `NFL_FULL_REPULL=1`, delivered
by `76ea3e43` ("Update NFL production artifacts").

**Verdict: 0 ERROR / 0 Traceback; 185 WARNING lines = 181 Open-Meteo
rate-limit lines (the finding, remediated) + 4 classified below.**

> **Second delivery, same day — reviewed and remediated again.** A later
> run (15:49:51 → 16:16, delivered by `518caae8`/`fff9afc3`) was the first
> with the breaker below in effect; it bounded the storm but still burned
> 21 logged 429 retries and kept tripping the quota. See
> "Second delivery" below for that review and the second-pass remediation
> (429s no longer ladder at all; the sweep is paced at 2.0s/batch).
>
> **Third delivery, same day — reviewed and remediated again.** The evening
> run (18:10:27 → 19:14, `d54ff441`) was the first with both earlier passes
> in effect; the breaker held (9 dead batches, no abort, 2022/2022) but the
> shared-IP window flapped shut for ~37 minutes and cooldown waiting
> stretched a ~26-minute sweep to **51m28s**. See "Third delivery" below
> for the third pass: defer-and-requeue — the archive sweep never sleeps
> on a 429 again.
>
> **Fourth delivery, same day — third pass verified in production.** The
> 20:45 run (`c18a6e37`, first with defer-and-requeue deployed) shows
> **zero** rate-limit activity — no 429, no cooldown, no deferral lines —
> 444/444 batches at the healthy 3.59s/it pace, coverage 2022/2022, and
> the fastest repull since the breaker landed (2,086s vs 3,815s). See
> "Fourth delivery" below.

## Fourth delivery (20:45 run) — third-pass verification

Scope: the delivered `nfl_pipeline_run_log.txt` — 370 lines, run
**2026-10-11 20:45:01 → 21:19:47 (2,086s)**, `NFL_FULL_REPULL=1`, delivered
by `c18a6e37` — the first production run with defer-and-requeue in effect
(third-pass code pushed 20:05 UTC; the run started 20:45 UTC against
`main`).

**Verdict: 0 ERROR / 0 Traceback; ZERO rate-limit activity.** A grep
across the full log for `429`, `deferred`, `requeue`, `cooling down` and
`dead batch` matches **nothing** — no quick retries, no cooldowns, no
deferrals, no requeue, no batch failures.

- Archive sweep: **444/444 batches in 26:32 @ 3.59s/it** — the healthy
  pace (2.0s pause + request time), window open the entire way.
- Recent phase: 6 seconds (21:11:36 → 21:11:42); final coverage
  **2022/2022**; `DONE — 238 artifacts in 2086s` — the fastest full
  repull since the breaker landed (vs 3,815s at 18:10 and 3,170s for
  the 05:20 storm run).
- The remaining 4 WARNING classes are previously classified, not new:
  2017-era OOF-thin windows (designed provisional handling), the OL-
  injury drift ALERTs (`inj_ol_out_home` psi_adj now 0.462, up from
  0.338 — a real model-owner signal, monitoring working), and
  `rest_days_*` LOW_COVERAGE at 74.4%.

### What this does and does not prove

The shared-IP quota never closed during this run, so defer-and-requeue
had nothing to defer — logged nothing, correctly. That verifies the
deployed code's CLEAN-path behavior end to end (identical pace, no
degradation, no spurious machinery output); the closed-window path stays
verified by the four contract tests in `test_log_review_20261011.py`
(defer never sleeps; sweep completes while closed; reopen detected and
requeued in-run; requeue stops at the abort line). Two consecutive runs
now show no storm (18:10: flap handled at 9 dead batches; 20:45: none at
all); if a flap returns, each deferred batch now costs ~one batch
interval instead of a 60–240s cooldown.

## Third delivery (evening 18:10 run) — re-review and third-pass remediation

Scope: the delivered `nfl_pipeline_run_log.txt`, run **2026-10-11
18:10:27 → 19:14 (3,815s)**, `NFL_FULL_REPULL=1`, delivered by `d54ff441`
— the first production run with both earlier passes in effect.

**Verdict: 0 ERROR / 0 Traceback; final coverage 2022/2022 (100%,
same-run). The breaker worked exactly as designed — and that is the
finding.**

### What the second remediation achieved (kept)

- No storm: **2** logged quick retries (vs 21 / 156 before), **9** dead
  batches (vs 25), one in-call cooldown + 9 entry cooldowns, streak peaked
  at 10 — under the 12-line. No abort, no errors, and the same-run top-up
  recovered the 48 gap games in **12 seconds** → `2022/2022`.

### What remained (the finding)

- Open-Meteo's shared-IP window **flapped shut for ~37 minutes anyway**
  (first 429 at 18:25:17, ~9 closures interleaved with healthy
  stretches). Every dead batch still bought an escalating cooldown —
  60 → 120 → 240s, **~30 minutes of pure sleeping** inside the sweep —
  so a ~26-minute sweep stretched to **51m28s** (18:10:27 → 19:01:55).
- Cooldown waiting itself was the remaining wall-clock risk: only the
  12-line stood between a longer block and a stalling run, and a batch
  arriving mid-flap still paid an entry wait plus a probe (the first
  batch of each flap additionally spent its two quick retries).
- No timeouts occurred (the 20s × 3 bound was not exercised).

### Third-pass remediation — defer-and-requeue (`weather.py`, MLB structure)

- **The archive sweep never sleeps on a 429.** `defer_on_429=True`
  (sweep-only mode in `_get_with_retry`) returns the batch immediately —
  the two quick retries survive only for the streak-0 marginal case — and
  `_fetch_batch_range` returns `None` so the sweep **defers** the batch
  into a requeue list. Every subsequent paced batch is itself the probe:
  a reopened window is detected within one batch interval (~4s) instead
  of after a 60–240s cooldown, and the sweep completes at its steady pace
  however long the window stays shut. Deferrals are logged once per phase
  (`N of M … were deferred`), never per batch.
- **One paced re-attempt per deferred batch** after the sweep
  (`_requeue_deferred_batches`), still in defer mode and bounded by
  `_RATE_LIMIT_ABORT_STREAK = 12` consecutive dead batches — under the new
  cost structure that is ~1 minute of probing into a window that is not
  coming back, not ~44 minutes of cooldowns. Recovered series merge
  in-run; the stop warning names both recovery paths.
- **The same-run top-up resets the streak first**, so the run's final
  recovery re-probes a window that may have reopened (the 18:10 run
  recovered all 48 gaps in 12s at the tail of a flap — now guaranteed a
  fresh breaker instead of inheriting a possibly-tripped one).
- **The sweep abort is gone** — superseded, not dropped: under defer mode
  a dead batch costs one paced request, so steady probing replaces both
  the abort and the cooldown as the reopen-detection mechanism. That is
  MLB's keep-sweeping structure, now safe at NFL's volume (MLB never
  needed it: 0 retries in its last 8 runs).
- Unchanged: the cooldown-probe path for every non-sweep caller
  (recent/forecast phases — at most one cooldown wait per call), the
  budget, the timeout cascade, the 2.0s pace, and all MLB parity.

### Expected shape on the next limited run

Zero `cooling down` lines during the archive sweep; at most one deferral
summary + one requeue line per phase. A full repull's weather phase is
~19–26 minutes **regardless of quota flaps**, deferred batches refill
in-run whenever the window reopens before the end, and anything still
missing after the top-up is refetched by the next run's cache-miss pass
(features stay NaN until then — never substituted).

### Verification (third pass)

- `python -m pytest -q` in `nfl-backend/backend` → **46 passed, exit 0**
  (43 + 4 new pins − 1 superseded).
- New pins in `test_log_review_20261011.py`: defer mode never sleeps and
  costs exactly one request per closed-window batch (the streak-0
  marginal case keeps its quick retries; a 200 still closes the window);
  the sweep completes every window while closed (defer → one re-attempt
  each; the only sleeps are pace pauses); a reopened window is detected by
  the next paced batch and the requeue refills deferred windows in-run;
  the requeue stops at the abort line with both recovery paths named
  while the sweep still completes all windows.
- The old "streak at the abort line → sweep sends zero requests" pin was
  replaced — that contract is superseded by defer-and-requeue, documented
  in the test module's docstring.

## Second delivery (15:50 run) — re-review and second-pass remediation

Scope: the delivered `nfl_pipeline_run_log.txt` — 489 lines, run **2026-10-11
15:49:51 → 16:16 (sweep phase 11:29 of ~26 min)**, `NFL_FULL_REPULL=1`,
delivered by `518caae8` ("Update NFL production artifacts") + `fff9afc3` —
the first production run with the 05:20-run breaker in effect.

**Verdict: 0 ERROR / 0 Traceback; 27 WARNINGs = 21 Open-Meteo 429 retries
(the finding, remediated again) + 1 batch-failed + 1 cooldown disclosure
(the breaker working) + 1 OOF-thin + 2 drift ALERT (same real signal as
05:20) + 1 table pointer.**

### What the first remediation achieved (kept)

- The 29-minute storm shape did not recur: **444/444 batches completed in
  11:29** (vs 40:39 with the storm), exactly **one dead batch**, one 60s
  cooldown disclosure, zero errors, full sweep — no abort.
- Timeout exposure, budget and abort from the first pass are untouched and
  remain the backstops.

### What remained (the finding)

- The per-request ladder still fired on 429s: **21 logged retries across 7
  groups** (waits 1.0–4.2s), all landing inside quota minutes that stayed
  shut for ~60s; one batch burned **all six retries (~65s) and died anyway**
  (15:58:26 → 15:59:30), then the cooldown fired. Quick retries inside a
  closed minute are pure waste — the only thing that ever worked was waiting.
- The sweep itself ran at **~39 requests/min** (444 batches in 689s with the
  1.0s pause) and the quota closed intermittently from minute ~8. Open-Meteo's
  limits (600/min, 5000/hr, 10000/day) are enforced **per IP** and Kaggle's
  egress IP is shared with other tenants, so the effective burst budget is
  far below the documented one — and our own pace was the only knob we own.
- No timeouts occurred in this run (the 20s × 3 bound from the first pass
  was not exercised).

### Second-pass remediation

- **429 is a quota event, handled in two stages** (`_get_with_retry`):
  up to `_429_QUICK_RETRIES = 2` short retries ride out a *marginal*
  window (the observed recoveries answered within ~3–9s); a still-closed
  window OPENS the streak and buys **one cooldown (60s, doubling, 4-min
  cap) + one probe** — never a ladder. A call now costs at most **4 requests
  and one cooldown**; batches arriving during a closed window cost **one
  request + the entry cooldown**. Budget, abort, streak-reset-on-definitive
  and the timeout cascade are unchanged.
- **`_BATCH_PAUSE_SEC` 1.0 → 2.0**: the sweep paces to ~23 requests/min
  (halving our contribution to the shared-IP budget); a full repull's 444
  batches take ~19 min — still far below the ~30-min storm cost.
- 502/503/504/network keep the full MLB-parity 7-attempt ladder: gateway
  blips answer quick retries and are not quota events.

### Expected shape on the next limited run

A closed quota minute now costs one `retrying` line + one `cooling down …`
line + one probe per batch, instead of 3–6 ladder lines per batch; the
sweep runs ~19 min at a steady ~23 req/min. If the window stays shut past
12 dead batches (~44 min), the sweep aborts loudly with the same recovery
disclosure as before.

### Verification (second pass)

- `python -m pytest -q` → **43 passed, exit 0** (41 + 2 new pins).
- `python test_production.py` → **511 passed, 0 failed, exit 0**. Two pins
  were updated to the new contract, each with the evidence in-line: the
  429-shape check (now `4 calls, sleeps [1.0, 2.0, 60.0], streak 2` + a
  hermetic limiter reset) and the pace check (`_BATCH_PAUSE_SEC >= 2.0`
  floor; window/size stay MLB-matched).
- New pins in `test_log_review_20261011.py`: dead batches after the first
  cost exactly one request each (5 dead calls → `[4, 1, 1, 1, 1]` requests,
  one cooldown per call, capped waits, streak climbing toward the abort),
  and the sweep pace floor.
- Note for testers: `test_production.py` executes its checks at import and
  `sys.exit(1)`s on failure, so a failing run crashes pytest collection
  with an INTERNALERROR — run it standalone (`python test_production.py`),
  which is how the two contract updates above were surfaced and verified.

## Log line review

| Lines | Message (abridged) | Verdict | Disposition |
|---|---|---|---|
| 12–67+ (156×) | `Open-Meteo 429; retrying in 1.1s…32.1s (n/6)` | **Finding 1 — remediated.** The per-batch ladder re-fired into a quota-closed window for 29 straight minutes (05:30:35 → 05:59:43). | Breaker (cooldown + single probe + budget + abort) below. |
| 25× | `Open-Meteo batch failed (… 429 Client Error …)` | Same finding: 26 dead batches, one per minute — each burned its full 7-attempt ladder. | Same. |
| 395 | `OOF validation population is thin on 93 of 200 window(s) (< MIN_VAL_FOLD_GAMES=15)` | Working as designed — 2017-era weekly windows are genuinely small; disclosure of what the gate excluded. | None. |
| 604–605 | `drift inj_ol_out_home ALERT psi_adj=0.338` / `inj_ol_out_away ALERT psi_adj=0.287` | The drift monitor **working** — a real distributional shift signal on offensive-line injury features (raw 0.408/0.355 vs null ~0.07). | None in this review (model-side follow-up, owner decision). |
| 608 | `full tables: run_engine_feature_drift_20261011.csv, …` | Pointer to the delivered tables. | None. |

## Finding 1 (remediated): the Open-Meteo retry storm — too many attempts into a closed window

### Evidence

| Run (delivered log) | 429 retry lines | dead batches |
|---|---|---|
| 10-04 (`d5d0feb5`) | 18 | 0 |
| 10-07 (`0b496c3d`) | 13 | 0 |
| 10-09 (`b333fccd`) | 72 | 11 |
| 10-10 00:12 (`fffcc312`) | 119 | 16 |
| 10-10 06:35 (`8e8cacfe`) | 0 | 0 |
| **10-11 (`76ea3e43`)** | **156** | **25** |

- The archive sweep fired **444 request batches** (111 windows × 4 of 15
  stadiums) and finished ~94% of them in the first 10 minutes — a sustained
  ~40 requests/min — before the per-minute quota closed at **05:30:35**.
- From then on **one batch died per minute for 29 minutes**: every batch
  re-armed the full 7-attempt ladder (1/2/4/8/16/32s + jitter ≈ 63s of
  sleeping, 7 requests) into a window that answered every one of them with
  429 — **≈182 requests into a closed window** (156 retries + 26 terminal).
- Worst-case timeout exposure of that design: 7 attempts × 45s + 63s of
  backoff ≈ **6.5 minutes per dead batch** — the "times out after multiple
  attempts" class. A stalled (rather than 429) endpoint would have cascaded
  this across the whole sweep with no accounting whatsoever.
- Cost: the storm alone added **~29 of the sweep's 40:39** (and ~55% of the
  53-minute run).
- What worked and is preserved: once the window reopened the same-run
  **top-up refetched the 126 missing games in 14 batches / 20s**, ending
  `2022/2022 eligible games available` — the recovery doctrine stands.

**Why the suite could not catch it:** no NFL test drove `_get_with_retry`
under a 429 sequence; the ladder's own constants were the only pin, and they
are correct *for an isolated blip*. Nothing modeled a quota-closed window
spanning many batches, because nothing in the client knew about windows —
each batch reset to attempt 1 as if nothing had happened.

### Root cause

Per-request resilience (7 attempts, `2**n` + jitter, Retry-After honored —
MLB parity) with **no cross-batch rate-limit awareness**: no cooldown, no
shared state, no attempt budget, and a 45s timeout on every attempt. Volume
did the rest: a full repull's 444-batch sweep runs straight into a
per-minute quota, and every following batch re-laddered.

### Remediation (`nfl-backend/backend/weather.py`, MLB structure)

MLB's `weather.py` is the structural reference — documented-constants block
with rationale, `_get_with_retry` ladder, batched `_fetch_batch_range` /
`_fetch_batched_weather` with pause + window skip, `batch failed` naming the
locations, same-run top-up and coverage warnings. NFL now has all of that
**plus a rate-limit breaker** MLB never needed (MLB logged 0 retries across
its last 8 runs — its per-run request volume is tiny):

- **Cooldown, then one probe.** A batch dead to a retriable status opens the
  window: the next request cools down **60s (one quota minute), doubling per
  consecutive dead batch, capped at 4 minutes**, then probes **once** —
  never a ladder. Any definitive (non-retriable) response resets the streak.
  `_get_with_retry` also prints `cooling down Ns … (consecutive dead
  batches: k)` instead of six ladder lines per batch.
- **Attempt budget.** `_429_RETRY_BUDGET = 40` bounds rate-limited retries
  for the whole process; past it a rate-limited request fails after a
  single attempt, disclosed once (`budget exhausted …`).
- **Sweep abort.** `_RATE_LIMIT_ABORT_STREAK = 12` consecutive dead batches
  abort the remaining archive sweep with a loud count of what was skipped
  and names the recovery paths (same-run top-up + next run's cache-miss
  pass). The line sits at **≈44 minutes of quiet probing** — just above the
  worst block ever observed (this run's 29 minutes), so a block of that size
  still recovers inside the same sweep, while a longer one stops costing the
  run its time budget. 12 = 63s first ladder + 60 + 120 + 10 × 240s.
- **Timeouts bounded.** Per-attempt timeout 45s → **20s** (`_REQUEST_TIMEOUT_SEC`,
  still 4–10× Open-Meteo's normal answer time for a ~400KB batch), and
  **3 consecutive hangs** end the request (`_TIMEOUT_RETRY_LIMIT`, ≈63s with
  backoff) instead of 7 × 45s ≈ 6.5 min — and a timeout-dead batch counts
  toward the streak, so a timeout storm can trip the abort too.
- Untouched (MLB parity): the 7-attempt ladder for isolated blips,
  Retry-After authority, `429/502/503/504` retriable set, 14-day windows,
  15 locations/batch, 1.0s pause, gameless-window skip, fail-fast 4xx,
  `batch failed` location listing, top-up, NaN-never-substituted doctrine.

### Simulation — same storm, old vs new client

Replay of this run's shape (quota closed for the observed 29.5 minutes,
half-way through a 104-batch sweep; mocked clock, mocked network, the **real**
`_fetch_batched_weather` for the new client):

| | requests total | requests into the closed window | dead ladders | sweep | wall |
|---|---|---|---|---|---|
| OLD (ladder-only) | 260 | ≈182 (26 × 7) | 26 | completes | 29.7 min |
| NEW (breaker) | **111** | **≈17** (one 7-ladder, then one probe per cooldown) | 1 ladder + probes, streak never reached 12 | completes, **no abort** | 33.8 min |

−57% total requests, ≈−91% into the closed window, +4 minutes wall (the
price of waiting instead of hammering), and the sweep still finishes — the
29-minute block recovers same-run exactly as designed.

## Claims reconciliation

| Log claim | Against | Verdict |
|---|---|---|
| `PIT weather archive: 111 window(s) x 4 batch(es) … = 444 request batches`, 40:39 @ 5.49s/it | fits ~427 healthy batches ×1.5s + 26 ladder-minutes | ✓ (storm accounts for ~29 of the 40 min) |
| `PIT weather top-up: 126 game(s) still missing … refetching the gaps only` → `14 request batches`, 20s | recovery ran after the quota reopened | ✓ |
| `PIT weather: 2022/2022 eligible games available from strict pre-kickoff hourly records` | final coverage line | ✓ (100% same-run; no NaN substitution) |
| `185 WARNING` total | 156 retries + 25 batch-failed + 1 OOF-thin + 2 drift ALERT + 1 table pointer = 185 | ✓ |
| 0 `ERROR`, 0 `Traceback` | grep over the delivered log | ✓ |
| `schedule rows (date window): 2839` (2016..2026), `238 artifacts`, `DONE — 238 artifacts in 3170s` | internal consistency | ✓ |

## Verification

- `python -m pytest -q` in `nfl-backend/backend` → **41 passed, exit 0**
  (35 baseline + 6 new pins).
- `python test_production.py` → **511 passed, 0 failed, exit 0** (the
  check-script's weather/top-up pins still pass against the new client).
- **New pins** (`test_log_review_20261011.py`, MLB-style):
  - isolated blip keeps the parity ladder (Retry-After authoritative,
    ≤2.5s second wait, streak stays 0, timeout constant used);
  - closed window → cooldown 60 → 120 (cap 240 asserted) + exactly **one**
    probe per call; a 200 resets the streak; no cooldown once open;
  - budget spent → 1 request, **no** retry sleep, one-shot disclosure;
  - 3 consecutive timeouts end the request and open the streak;
  - streak at the abort line → archive sweep sends **zero** archive
    requests and warns with `not attempted` + both recovery paths;
    control run (streak 0) still sweeps;
  - abort boundary is exactly `_RATE_LIMIT_ABORT_STREAK`.
- Storm replay simulation (table above), mocked clock/network, no live API
  calls.

## Deliberately not changed

- The 7-attempt `2**n` ladder for 502/503/504 and network errors
  (MLB-documented design; correct for gateway blips). *(Superseded for
  429s by the second pass below — they no longer ladder at all.)*
- `_BATCH_DAYS = 14`, `_BATCH_SIZE = 15`, gameless-window skip — MLB
  parity, already paced. *(`_BATCH_PAUSE_SEC` was 1.0s here; superseded
  by the second pass below.)*
- Top-up pass, coverage warnings, NaN-never-substituted and cache-miss
  self-heal doctrine — untouched; they are the recovery path the abort
  names.
- The run's other 4 warnings (OOF thin windows, two drift ALERTs, table
  pointer) — monitoring outputs, not defects; the `inj_ol_out_*` ALERT is a
  real signal for the model owner, out of scope here.
- `mlb-backend/weather.py` — same ladder, but 0 retries in its last 8 runs;
  porting the breaker there is an optional follow-up, not a fix.

## Residual risk / next-run expectations

- The next limited run will log `cooling down Ns … (consecutive dead
  batches: k)` instead of walls of retry lines — expected, not a regression.
- If a block ever exceeds ~44 minutes, the sweep aborts with a
  `… not attempted` warning; affected games stay out of the cache and the
  **next run's cache-miss pass refetches exactly them** (features stay NaN
  until then — never substituted).
- The breaker is process-local: every fresh run starts with a clean streak
  and a fresh 40-retry budget (bounded per run, as specified).
- The sweep's healthy pace (~40 requests/min) is what trips the per-minute
  quota on full repulls; the breaker makes tripping cheap. *(Superseded by
  the second pass below: the pace itself was slowed to ~23 req/min.)*
- NBA/NHL do not use Open-Meteo; MLB keeps its own (currently storm-free)
  client — no cross-sport exposure from this change.
