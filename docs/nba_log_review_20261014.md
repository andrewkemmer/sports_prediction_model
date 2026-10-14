# NBA run-log review — 2026-10-14

Scope: the delivered `nba-backend/data_delivery/nba_pipeline_run_log.txt` — 255
lines, run **2026-10-14 00:45:37 → 01:03:12 (1,643.5s)**, `NBA_FULL_REPULL=1`,
delivered by `ca48fd10` ("Update NBA features + predictions") + `dca502f0`
("final delivery"). The ingestion layer under review is `5c952541` ("parallel
cold sweeps behind a paced gate halve the pull").

**Verdict: 0 ERROR / 0 Traceback; every material count reconciles against the
delivered artifacts; the runtime optimization is verified sound — identical
output at 2.29x on play-by-play and 1.57x on the schedule. 1 finding
remediated: the run's single largest phase reported a rate without ever naming
the ceiling that rate was measured against, so the log concealed the one fact
that decides whether the sweep can be made faster.**

## Runtime profile — where the 1,643.5s went

| Phase | Elapsed | Share | Rate | Ceiling |
|---|---|---|---|---|
| Play-by-play sweep (3,461 games) | 872.1s | **53.1%** | 3.97 game/s | 4.00/s → **99.2%** |
| Schedule sweep (1,025 ESPN days) | 115.2s | 7.0% | 8.90 day/s | 10.00/s → 89% |
| Season-log pull (34 slices) | 21.0s | 1.3% | 1.62 slice/s | 2.00/s → 81% |
| Features, folds, blend, artifacts | ~635s | 38.6% | — | — |

The three sweeps are rate-gated by a shared turnstile, not by the worker pool.
The pool only decides how many requests are *waiting* when the gate opens.

## Finding 1 (remediated): the largest phase never named its constraint

The run printed:

```
2026-10-14 00:45:37,788 INFO schedule: 1025 day(s) …; 4 worker(s), at most one request start every 0.10s
2026-10-14 00:47:57,510 INFO play-by-play: 3461 candidate game(s), sweeping 3461 on 4 worker(s), at most one request start every 0.25s
play-by-play: 100%|##########| 3461/3461 [14:32<00:00,  3.97game/s, 0042500405]
2026-10-14 01:02:29,618 INFO play-by-play: 3461 fetched, 0 from cache, 0 unavailable
```

The header names the worker count. The tail names the achieved rate. Neither
names the ceiling, and the reader is left to divide the two to discover that
**the play-by-play sweep is 53% of the pipeline and sits at 99.2% of the rate
its own gate permits.** That is the load-bearing fact about this run, and it
decides the only question a reader has: can this be made faster?

The answer is no, not with workers. `NBA_PULL_WORKERS` cannot shorten a sweep
whose gate is already the binding constraint — adding workers just moves the
queue that waits. Only `NBA_PBP_PAUSE_SEC` can shorten it, and lowering that
is a deliberate politeness choice about a third party rather than a knob to
turn blind. Stating the ceiling converts a silent "it is what it is" into a
number a reader can act on, and turns a future regression — the pool no longer
saturating the gate — into a visible shortfall instead of a quiet slowdown.

### The remediation, and the bug it had to avoid

`_gate_note()` now appends the ceiling to all three sweep summaries:

```
play-by-play: 3461 fetched, 0 from cache, 0 unavailable; 3.97 game/s against a 4.00/s gate (99% of ceiling)
schedule: 3488 games over 1025 days (1025 fetched, 0 from cache); 8.90 day/s against a 10.00/s gate (89% of ceiling)
season log: 74449 player row(s) over 34 slice(s); 1.62 slice/s against a 2.00/s gate (81% of ceiling)
```

The season-log pull is worth its own line: at 81% it is *not* gate-bound the
way play-by-play is, because its 34 requests each carry up to 60 days of
parsing. Its ceiling has slack to give; the play-by-play gate does not.

The numerator is the count of units that **reached the turnstile**, not the
count walked. A cache hit is served without a request start, so it must not
sit in the numerator. This is also what makes the figure trustworthy: the
turnstile itself guarantees `elapsed >= paced * interval_sec`, so the
reported rate **cannot exceed its own ceiling**. Counting the walked total
instead would print `8.05 game/s (201% of ceiling)` on a half-warm sweep —
over 100% of a bound that cannot be broken, which is precisely the false
reading that would discredit the number. Six regression tests pin this,
including one that fails against the walked-total numerator by construction
(`test_the_rate_never_exceeds_the_ceiling_it_is_measured_against`,
`test_the_schedule_sweep_counts_days_it_fetched_not_days_it_walked`).

Two further guards: a fully-cached sweep returns an empty note rather than a
millions-per-second figure, and an ungated sweep reports a rate with no
ceiling to claim.

## The optimization itself — verified sound

`5c952541` moved the three cold sweeps onto a bounded worker pool consumed in
input order, behind a turnstile that the pool cannot outrun. The review
confirms the property the commit claims:

- **Output is unchanged.** The serial 10-13 run (`bbedbb8c`) and the parallel
  10-14 run produced **identical** facts: 3,488 games, 74,449 player rows,
  1,712,728 play-by-play actions, 6,922 event rows, and an identical rolling
  Brier of 0.2091 over 2,400 rows.
- **The gate holds.** 3,461 games over 872.1s is 3.97 game/s against a 4.00/s
  ceiling — 99.2%, i.e. the pool saturated the gate rather than starving it,
  with zero non-200 responses.
- **The sweep budget has ample headroom.** 872s actual against a 5,400s
  `PBP_BUDGET` — 16% used, so the budget is not the constraint either.
- **The rollup archive cannot seed the cold sweep.** `read_event_rollup_archive`
  holds team-game *aggregates*, not raw play-by-play, so a cold run's sweep
  cost is structural. Worth knowing before anyone assumes a warm archive
  removes the 872s.

## Cross-commit review of the runtime-optimization series

Four commits constitute the runtime-optimization pass. Each was reviewed
against the delivered logs; three required no change.

| Commit | Sport | Reviewed verdict |
|---|---|---|
| `5c952541` | NBA | Sound. **Finding 1 above remediated** — the sweeps now name their ceiling. |
| `df57af1a` | NHL | Sound. Collapsed a 32-min Phase-3 silent window and parallelized the cold pull with frame-identical outputs (goalie_state 1260.8s → 7.2s). Fully remediated by `931ad8fa`. |
| `931ad8fa` | NHL | Remediation, not a defect source. Fixed the ONE shared bar reporting one chunk's counters as the sweep's (`679/679 … fetched=43`), and added the worker/pause announcements `df57af1a` lacked. NHL reports rates on a live progress line with an eta, and `PULL_CHUNK_PAUSE_SEC = 0.0` means NHL has **no rate gate to name** — the pool is genuinely the limit there, so no ceiling note applies. Correct as shipped. |
| `1d248572` | NFL | Sound. Deduplicated per physical venue (136 → 102 request batches, −25%) with byte-identical delivered keys, and closed the log's structural gaps. The rate-limit machinery was deliberately untouched and is ahead of MLB's. |

**MLB was checked and needs no ceiling note.** It has no parallel cold pools —
its pulls are serial — so there is no turnstile whose ceiling could be
saturated or misreported. Its `PAUSE_SEC` constants pace serial loops, where
the elapsed time is request-bound by construction.

## Claims reconciliation

| Log claim | Against | Verdict |
|---|---|---|
| `NBA facts: 3488 games, 6922 team rows, 74449 player rows, 1712728 play-by-play actions, 6922 event rows` | 3,488 × 2 = 6,976 team-game slots; 6,922 event rows = 3,461 decided × 2 | ✓ |
| `play-by-play: 3461 candidate game(s), sweeping 3461 on 4 worker(s)` → `3461 fetched, 0 from cache, 0 unavailable` | Bar reads 3461/3461; log span 00:47:57,510 → 01:02:29,618 = 872.1s | ✓ |
| `rolling Brier: 0.2091 … 2400 game(s)` | Identical to the serial 10-13 run | ✓ |
| `✅ status ok, 1643.51s` | Log span 00:45:37 → 01:03:12 = 1,643.5s | ✓ exact |
| `27 scheduled game(s) have no season-log match` | 11 postponed + 2 finished without player lines + 14 upcoming slate games (2026-10-20/21) = 27 | ✓ |
| `3 ensemble member(s), 14 upcoming game(s) scored` | The 14 upcoming games within the 27-list | ✓ |

## Verification

- **NBA 653 passed** (647 baseline + 6 new gate-note regression tests).
- **NHL 632 passed** — matches `931ad8fa`'s stated baseline.
- **NFL 65 passed**; script-style production suite **561 passed, 0 failed** —
  matches `1d248572`.
- **MLB 408 passed, 1 skipped**; **frontend 119 passed**.
- No new programs. The remediation is confined to `nba-backend/backend/ingestion.py`
  and its existing test file; the note is a log annotation and never a
  control-flow input.
