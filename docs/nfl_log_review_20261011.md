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
