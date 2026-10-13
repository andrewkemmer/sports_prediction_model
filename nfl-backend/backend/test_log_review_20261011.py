"""2026-10-11 NFL run-log review: the Open-Meteo rate-limit/timeout breaker.

Evidence (``data_delivery/nfl_pipeline_run_log.txt``, run 2026-10-11
05:30:35 → 05:39:42): **156** ``Open-Meteo 429; retrying`` lines and **25**
``Open-Meteo batch failed`` batches inside a single archive sweep — and the
storms grow run over run (13 → 72 → 119 → 156 retries over 10-07 → 10-11).
The per-request ladder (7 attempts, 2**n + jitter, Retry-After honored —
MLB parity) is right for an isolated blip, but a closed quota window answers
EVERY following batch with 429 too, and each batch re-armed the full ladder
independently: ~7 requests + 63s of sleeps per dead batch, back-to-back, into
a window that could not answer with anything else. Each attempt also carried
a 45s timeout, so a batch that STALLED instead of 429-ing cost up to 7×45s +
63s ≈ 6.5 minutes — the "times out after multiple attempts" class.

Second evidence set (same file, 15:50 run — the first with the breaker):
the breaker bounded the sweep (444/444 batches in 11:29, one dead batch,
one 60s cooldown, zero errors) but the LADDER still burned **21 logged
429 retries** across 7 groups: quick 1/2/4s retries fired into quota
minutes that stayed shut for ~60s, and one batch consumed all six retries
(~65s) and died anyway. Open-Meteo's
limits are per IP (600/min, 5000/hr, 10000/day) and Kaggle's egress IP is
shared, so the sweep also had to slow down.

Third evidence set (same file, evening 18:10 run — the first WITH both
remediations): the breaker behaved exactly as designed — 2 quick retries
on the first 429, one cooldown + one probe per dead batch, 9 dead
batches, streak peaked at 10 (under the 12 line), the top-up recovered
the 48 gap games in 12s, final coverage 2022/2022 — yet the shared-IP
window flapped shut for ~37 minutes and the escalating cooldowns (60 →
240s, ~30 minutes of pure sleep) stretched a ~26-minute sweep to
**51m28s**.  Cooldown waiting itself was the remaining wall-clock risk:
every closed-window batch still bought a 60–240s wait, and only the
abort line stood between a longer block and a stalling run.

Remediation (``nfl-backend/backend/weather.py``, structure mirrors
``mlb-backend/backend/weather.py``'s documented-constants + retry + batch +
top-up layout):

* 429 is a quota event: up to _429_QUICK_RETRIES (2) short retries ride
  out a MARGINAL window (observed recoveries at ~3-9s); a still-closed
  window OPENS the streak and buys one cooldown (60s, doubling per
  consecutive dead batch, 4-minute cap) + one probe — never a ladder.
  This remains the path for non-sweep callers (the recent/forecast
  phases and anything outside the archive sweep);
* third pass (defer-and-requeue): the ARCHIVE SWEEP never sleeps on a
  429 — _get_with_retry(defer_on_429=True) returns immediately (the two
  quick retries survive only for the streak-0 marginal case) and
  _fetch_batch_range returns None so the sweep DEFERS the batch; every
  subsequent paced batch is itself the probe, so a reopened window is
  detected within one batch interval (~4s) instead of after a 60–240s
  cooldown, and the sweep completes at its steady pace however long the
  window stays shut;
* one paced re-attempt per deferred batch follows the sweep
  (_requeue_deferred_batches), bounded by _RATE_LIMIT_ABORT_STREAK (12)
  consecutive dead batches; the same-run top-up resets the streak first
  so the run's final recovery re-probes a possibly-reopened window, and
  the next run's cache-miss pass remains the last resort (features NaN,
  never substituted);
* any definitive (non-retriable) response CLOSES the window (streak reset);
* a per-process 429 budget (40) bounds rate-limited attempts outright —
  past it a rate-limited request fails after one attempt, logged once;
* timeouts: 20s per attempt, and three consecutive hangs end the request
  (≈63s) instead of seven 45s hangs (≈6.5 min per dead batch);
* _BATCH_PAUSE_SEC = 2.0: the sweep paces to ~23 requests/min, halving
  the ~39/min rate at which the shared-IP quota kept closing.

Thresholds are pinned here so a future tuning change is a deliberate,
reviewed edit rather than a silent drift back toward storming.
"""
from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import weather  # noqa: E402


class _Resp:
    """Minimal requests.Response stand-in for the status/header surface."""

    def __init__(self, status_code: int, headers: dict | None = None):
        self.status_code = status_code
        self.headers = headers or {}


@pytest.fixture(autouse=True)
def _clean_rate_limiter():
    weather._reset_rate_limiter()
    yield
    weather._reset_rate_limiter()


@pytest.fixture
def sleeps(monkeypatch):
    """Record every time.sleep the weather module issues (never actually wait)."""
    recorded: list[float] = []
    monkeypatch.setattr(weather.time, "sleep", recorded.append)
    return recorded


def _patch_get(monkeypatch, responder):
    calls: list[int] = []

    def fake_get(url, params=None, timeout=None):
        calls.append(timeout)
        result = responder(len(calls))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(weather.requests, "get", fake_get)
    return calls


def _budget_messages(caplog):
    return [m for m in caplog.messages if "budget exhausted" in m]


def test_isolated_blip_still_uses_the_mlb_parity_ladder(
        monkeypatch, sleeps, caplog):
    """A marginal 429 blip rides out with the two quick retries
    (Retry-After authoritative) and never opens the breaker."""
    seq = [_Resp(429, {"Retry-After": "7"}), _Resp(429), _Resp(200)]
    calls = _patch_get(monkeypatch, lambda n: seq[n - 1])

    with caplog.at_level(logging.WARNING):
        resp = weather._get_with_retry("https://example.invalid", {})

    assert resp.status_code == 200
    assert len(calls) == 3, calls
    assert len(sleeps) == 2, sleeps
    assert sleeps[0] >= 7.0, f"Retry-After must be authoritative: {sleeps}"
    assert 2.0 <= sleeps[1] <= 2.5, sleeps
    assert weather._rate_limit_streak() == 0, "a recovered blip opens nothing"
    assert all(t == weather._REQUEST_TIMEOUT_SEC for t in calls), calls
    # Two quick retries were available; both were consumed and recovered
    # the batch without ever entering the cooldown/probe stage.
    assert weather._rate_limit["retries"] == 2
    assert not any("cooling down" in m for m in caplog.messages)


def test_closed_window_cools_down_then_probes_once(
        monkeypatch, sleeps, caplog):
    """The storm shape from the delivered log: every request answers 429.

    The first call spends its two quick retries (a marginal window might
    reopen), then OPENS the streak, cools down one quota minute and probes
    once — 4 requests total, not a 7-attempt ladder. Afterwards every
    call cools down (60s → 120s → … capped at 240s) and makes exactly
    ONE request — and a definitive answer resets the streak.
    """
    calls = _patch_get(monkeypatch, lambda n: _Resp(429))

    with caplog.at_level(logging.WARNING):
        resp = weather._get_with_retry("https://example.invalid", {})
    assert resp.status_code == 429, "a terminal probe returns the 429"
    assert len(calls) == 2 + 1 + weather._RATE_LIMIT_PROBE_ATTEMPTS, calls
    assert len(sleeps) == 3, sleeps
    assert sleeps[2] == weather._RATE_LIMIT_COOLDOWN_SEC, sleeps
    assert weather._rate_limit_streak() == 2

    sleeps.clear()
    weather._get_with_retry("https://example.invalid", {})
    assert sleeps == [weather._cooldown_wait(2)], (
        f"an entry probe cools down for the quota minute: {sleeps}")
    assert len(calls) == 5, f"one entry probe, never a ladder: {calls}"
    assert weather._rate_limit_streak() == 3

    sleeps.clear()
    reopened = _patch_get(monkeypatch, lambda n: _Resp(200))
    weather._get_with_retry("https://example.invalid", {})
    assert sleeps == [weather._cooldown_wait(3)], (
        f"cooldown doubles per dead batch: {sleeps}")
    assert len(reopened) == 1, reopened
    assert weather._rate_limit_streak() == 0, "a 200 closes the window"

    sleeps.clear()
    weather._get_with_retry("https://example.invalid", {})
    assert sleeps == [], f"window open again, no cooldown owed: {sleeps}"
    assert len(reopened) == 2, reopened

    # Quota math: one minute, doubling, capped at four.
    assert weather._cooldown_wait(1) == 60.0
    assert weather._cooldown_wait(2) == 120.0
    assert weather._cooldown_wait(5) == weather._RATE_LIMIT_COOLDOWN_MAX_SEC

    cooldowns = [m for m in caplog.messages if "cooling down" in m]
    assert len(cooldowns) == 3, cooldowns


def test_429_budget_fails_fast_and_logs_once(monkeypatch, sleeps, caplog):
    """Past the per-run budget a rate-limited request must not ladder at all:
    one attempt, no sleep, and the budget disclosure repeats only once."""
    weather._rate_limit["retries"] = weather._429_RETRY_BUDGET
    calls = _patch_get(monkeypatch, lambda n: _Resp(429))

    with caplog.at_level(logging.WARNING):
        weather._get_with_retry("https://example.invalid", {})
    assert len(calls) == 1, calls
    assert sleeps == [], f"budget-exhausted = fail fast, no retry sleep: {sleeps}"
    assert weather._rate_limit_streak() == 1
    assert len(_budget_messages(caplog)) == 1

    with caplog.at_level(logging.WARNING):
        weather._get_with_retry("https://example.invalid", {})
    assert len(calls) == 2, "the second call is one cooldown probe"
    assert sleeps == [weather._RATE_LIMIT_COOLDOWN_SEC], sleeps
    assert len(_budget_messages(caplog)) == 1, "budget warning is one-shot"


def test_timeout_cascade_ends_after_three_hangs(monkeypatch, sleeps, caplog):
    """A stalled endpoint must not cost 7 x 45s per batch: three hangs end
    the request (~63s with backoff), the batch dies, and the streak accounts
    for it so the breaker can pace/abort the sweep."""
    calls = _patch_get(monkeypatch, lambda n: weather.requests.Timeout("hang"))

    with caplog.at_level(logging.WARNING), pytest.raises(weather.requests.Timeout):
        weather._get_with_retry("https://example.invalid", {})

    assert len(calls) == weather._TIMEOUT_RETRY_LIMIT, calls
    assert len(sleeps) == weather._TIMEOUT_RETRY_LIMIT - 1, sleeps
    assert weather._rate_limit_streak() == 1, (
        "a timeout-dead batch must open the streak, or a timeout storm "
        "would never trip the abort")
    assert all(t == weather._REQUEST_TIMEOUT_SEC for t in calls), calls


def _target(game_id: str, kickoff: str) -> dict:
    return {
        "game_id": game_id,
        "stadium": "Test Stadium",
        "lat": 39.0,
        "lon": -95.0,
        "kickoff_utc": pd.Timestamp(kickoff),
    }


# Kickoffs 30 days apart -> two 14-day archive windows (the days between
# hold no game, so exactly these two windows are planned, as in production).
_TWO_WINDOW_TARGETS = [
    _target("20260812_A@HOME", "2026-08-12T17:00:00Z"),
    _target("20260911_B@HOME", "2026-09-11T17:00:00Z"),
]


def test_defer_mode_never_cools_down_and_costs_one_request_per_batch(
        monkeypatch, sleeps, caplog):
    """The sweep's mode, pinned at the client level (third review): a 429
    in defer mode returns immediately — no entry cooldown, no
    cooldown/probe stage — so no rate-limited request in that mode can ever
    sleep.  The streak-0 marginal case still gets its two quick retries; a
    streak-open batch costs exactly ONE request; a definitive answer closes
    the window."""
    calls = _patch_get(monkeypatch, lambda n: _Resp(429))

    with caplog.at_level(logging.WARNING):
        resp = weather._get_with_retry("https://example.invalid", {},
                                       defer_on_429=True)
    assert resp.status_code == 429
    assert len(calls) == 1 + weather._429_QUICK_RETRIES, (
        f"streak-0 marginal case keeps its quick retries: {calls}")
    assert len(sleeps) == 2, sleeps  # the two quick retries, nothing else
    assert weather._rate_limit_streak() == 1
    assert not any("cooling down" in m for m in caplog.messages)

    per_call = []
    for _ in range(2):
        before = len(calls)
        sleeps.clear()
        weather._get_with_retry("https://example.invalid", {},
                                defer_on_429=True)
        per_call.append(len(calls) - before)
        assert sleeps == [], (
            f"a streak-open defer-mode batch must never sleep: {sleeps}")
    assert per_call == [1, 1], (
        f"a closed-window defer-mode batch costs one request each: {per_call}")
    assert weather._rate_limit_streak() == 3
    # Only the marginal case spent the per-run 429 budget.
    assert weather._rate_limit["retries"] == weather._429_QUICK_RETRIES

    reopened = _patch_get(monkeypatch, lambda n: _Resp(200))
    resp = weather._get_with_retry("https://example.invalid", {},
                                   defer_on_429=True)
    assert resp.status_code == 200 and len(reopened) == 1
    assert weather._rate_limit_streak() == 0, "a 200 closes the window"


def test_sweep_defers_429_batches_and_still_completes(
        monkeypatch, sleeps, caplog):
    """Third-review contract: a closed window must neither stretch nor
    truncate the archive sweep.  Every window is attempted at the sweep's
    pace; 429 batches come back ``None`` and are DEFERRED for one re-attempt
    — never dying into cooldown sleeps — and the requeue gets exactly one
    paced request per deferred batch."""
    state = {"closed": False, "archive": 0}

    def fake_batch_range(locations, start_date, end_date, *, source,
                         defer_on_429=False, **kw):
        if source != weather.OPEN_METEO_ARCHIVE:
            return {}
        state["archive"] += 1
        if state["closed"] and defer_on_429:
            weather._rate_limit["streak"] += 1  # a real 429 opens it
            return None
        weather._rate_limit["streak"] = 0
        return {}

    monkeypatch.setattr(weather, "_fetch_batch_range", fake_batch_range)
    now = datetime(2026, 10, 11, 6, 0, tzinfo=timezone.utc)

    # Control: an open window sweeps both windows normally, deferring nothing.
    with caplog.at_level(logging.INFO):
        weather._fetch_batched_weather(_TWO_WINDOW_TARGETS, now)
    assert state["archive"] == 2, state
    assert not any("deferred" in m for m in caplog.messages)

    # Closed: the sweep STILL completes both windows (one request each) and
    # the requeue re-attempts each exactly once — four requests total, and
    # the only sleeps are the pace pauses, never a cooldown.
    state.update(closed=True, archive=0)
    weather._reset_rate_limiter()
    caplog.clear()
    sleeps.clear()
    with caplog.at_level(logging.INFO):
        weather._fetch_batched_weather(_TWO_WINDOW_TARGETS, now)

    assert state["archive"] == 4, (
        "closed window: sweep completes both windows + one re-attempt each")
    assert weather._rate_limit_streak() == 4, (
        "each deferred request opens one more streak unit")
    assert sleeps == [weather._BATCH_PAUSE_SEC,
                      weather._BATCH_PAUSE_SEC], (
        f"defer mode must never sleep a cooldown: {sleeps}")
    deferred = [m for m in caplog.messages if "were deferred" in m]
    assert len(deferred) == 1 and "2 of 2" in deferred[0], caplog.messages
    requeue = [m for m in caplog.messages if "requeue" in m]
    assert len(requeue) == 1 and "0 of 2" in requeue[0], caplog.messages


def test_requeue_recovers_deferred_batches_when_the_window_reopens(
        monkeypatch, caplog):
    """The deferral's payoff: a window that reopens mid-sweep is noticed by
    the next paced batch (no cooldown required), and the requeue refills
    every deferred window in-run — the same-run top-up never sees them."""
    state = {"deferred": 0, "archive": 0}

    def fake_batch_range(locations, start_date, end_date, *, source,
                         defer_on_429=False, **kw):
        if source != weather.OPEN_METEO_ARCHIVE:
            return {}
        state["archive"] += 1
        if state["deferred"] < 1:
            state["deferred"] += 1
            weather._rate_limit["streak"] += 1
            return None
        weather._rate_limit["streak"] = 0
        return {(str(locations[0][0]), end_date): {"time": []}}

    monkeypatch.setattr(weather, "_fetch_batch_range", fake_batch_range)
    now = datetime(2026, 10, 11, 6, 0, tzinfo=timezone.utc)

    with caplog.at_level(logging.INFO):
        by_key = weather._fetch_batched_weather(_TWO_WINDOW_TARGETS, now)

    # Sweep: window 1 deferred, window 2 answered (streak reset).  Requeue:
    # window 1 recovered.  Three archive requests, no extra sleeps.
    assert state["archive"] == 3, state
    assert weather._rate_limit_streak() == 0, "the reopened window closes it"
    assert {day for (_stadium, day) in by_key} == {date(2026, 8, 24),
                                                  date(2026, 9, 11)}, by_key
    requeue = [m for m in caplog.messages if "requeue" in m]
    assert len(requeue) == 1 and "1 of 1" in requeue[0], caplog.messages


def test_requeue_stops_at_the_abort_line_and_names_the_recovery_paths(
        monkeypatch, caplog):
    """Past _RATE_LIMIT_ABORT_STREAK consecutive dead batches the requeue
    stops probing — a dead batch costs one paced request now, so the line
    bounds the requeue at about a minute of waste — and the warning names
    both recovery paths.  The SWEEP itself is never aborted: it completes
    all 13 windows despite the streak.

    The 2026-10-13 review changed WHICH streak bounds the pass: it used to
    inherit the sweep's, so a sweep that ended past the line tripped the
    check on index 0 and the pass attempted NOTHING.  The line now bounds
    the pass on its own consecutive dead batches — 12 paced probes here —
    while the sweep's 13 dead batches still stand."""
    state = {"archive": 0}

    def fake_batch_range(locations, start_date, end_date, *, source,
                         defer_on_429=False, **kw):
        if source != weather.OPEN_METEO_ARCHIVE:
            return {}
        state["archive"] += 1
        weather._rate_limit["streak"] += 1
        return None

    monkeypatch.setattr(weather, "_fetch_batch_range", fake_batch_range)
    now = datetime(2026, 10, 11, 6, 0, tzinfo=timezone.utc)
    # A game every 15 days -> exactly one per 14-day window, 13 windows.
    targets = [
        _target(
            f"2026{i:02d}01_TEST@HOME",
            (datetime(2026, 1, 1, tzinfo=timezone.utc)
             + timedelta(days=15 * i)).strftime("%Y-%m-%dT%H:%M:%SZ"))
        for i in range(13)
    ]

    with caplog.at_level(logging.INFO):
        weather._fetch_batched_weather(targets, now)

    assert state["archive"] == 13 + weather._RATE_LIMIT_ABORT_STREAK, (
        "the sweep completes every window, then the pass probes exactly "
        f"its own abort line: {state}")
    warnings = [m for m in caplog.messages if "requeue" in m
                and "not re-attempted" in m]
    assert len(warnings) == 1, caplog.messages
    assert "1 deferred" in warnings[0], warnings[0]
    assert "12 consecutive dead batches" in warnings[0], warnings[0]
    assert "top-up" in warnings[0] and "cache-miss" in warnings[0], (
        "the warning must name the recovery paths")
    assert weather._rate_limit_streak() >= weather._RATE_LIMIT_ABORT_STREAK


def test_requeue_probes_a_sweep_that_ended_past_the_abort_line(
        monkeypatch, caplog):
    """2026-10-13 run-log evidence: the archive sweep ended at streak 48
    (48 of 448 batches deferred), the INFO line promised "one paced
    re-attempt follows", and the requeue then tripped its inherited streak
    on index 0 — "48 deferred batch(es) not re-attempted".  The pass must
    actually probe in that state: every deferred batch gets its re-attempt
    and a recovered batch merges, with no abort."""
    weather._reset_rate_limiter()
    weather._rate_limit["streak"] = 48      # the sweep's inheritance
    deferred = [
        ([("Test Stadium", 39.0, -95.0)],
         date(2026, 9, 1) + timedelta(days=i),
         date(2026, 9, 1) + timedelta(days=i))
        for i in range(48)
    ]
    state = {"calls": 0}

    def fake_batch_range(locations, start_date, end_date, *, source,
                         defer_on_429=False, **kw):
        state["calls"] += 1
        weather._rate_limit["streak"] = 0    # a 200 closes the window
        return {(locations[0][0], start_date): {"time": []}}

    monkeypatch.setattr(weather, "_fetch_batch_range", fake_batch_range)
    by_key: dict = {}
    with caplog.at_level(logging.INFO):
        weather._requeue_deferred_batches(deferred, by_key)

    assert state["calls"] == 48, (
        f"every deferred batch must be probed: {state['calls']} of 48")
    assert len(by_key) == 48, by_key
    assert not any("not re-attempted" in m for m in caplog.messages), (
        caplog.messages)
    assert weather._rate_limit_streak() == 0, (
        "a recovered batch closes the window the sweep left open")


def test_recovery_pass_first_batch_takes_the_cooldown_path(monkeypatch):
    """2026-10-13 run-log evidence: the gap-only top-up fired all 18
    batches into a still-shut archive window in 39 seconds (defer mode
    never waits) and recovered 0 of the 142 missing games.  The recovery
    pass's FIRST batch therefore runs the non-defer client — two quick
    retries, one bounded cooldown, one probe — while every later batch
    keeps defer mode so the sweep itself still never sleeps."""
    modes: list[bool] = []

    def fake_batch_range(locations, start_date, end_date, *, source,
                         defer_on_429=False, **kw):
        if source != weather.OPEN_METEO_ARCHIVE:
            return {}
        modes.append(defer_on_429)
        weather._rate_limit["streak"] = 0
        return {}

    monkeypatch.setattr(weather, "_fetch_batch_range", fake_batch_range)
    now = datetime(2026, 10, 11, 6, 0, tzinfo=timezone.utc)

    weather._fetch_batched_weather(_TWO_WINDOW_TARGETS, now)
    assert modes == [True, True], (
        f"the main sweep defers every batch: {modes}")

    modes.clear()
    weather._fetch_batched_weather(_TWO_WINDOW_TARGETS, now,
                                   cooldown_first_batch=True)
    assert modes == [False, True], (
        f"only the recovery pass's first batch may cool down: {modes}")


def test_a_429_is_a_deferred_batch_in_both_modes(monkeypatch, sleeps,
                                                  caplog):
    """Rate-limited is never dead-for-good: ``_fetch_batch_range`` returns
    ``None`` instead of raising into the generic failure handler, which
    returned ``{}`` and silently dropped the batch from the requeue list.
    A non-defer caller gets one explicit line (the sweep stays quiet)."""
    _patch_get(monkeypatch, lambda n: _Resp(429))

    with caplog.at_level(logging.WARNING):
        result = weather._fetch_batch_range(
            [("Test Stadium", 39.0, -95.0)],
            date(2026, 9, 1), date(2026, 9, 14),
            source=weather.OPEN_METEO_ARCHIVE)

    assert result is None, result
    assert not any("batch failed" in m for m in caplog.messages), (
        "a 429 must not be reported as a failed batch: "
        f"{caplog.messages}")
    assert any("rate limited after cooldown+probe" in m
               for m in caplog.messages), caplog.messages
    assert sleeps, "the non-defer path's cooldown must have been recorded"


def test_recent_and_forecast_phases_skip_a_deferred_batch(monkeypatch):
    """The recent/forecast phases receive ``None`` from a rate-limited
    batch; calling ``.items()`` on it would raise straight out of the
    weather phase.  They skip and leave the gap to the cache-miss pass."""
    def always_none(locations, start_date, end_date, *, source, **kw):
        weather._rate_limit["streak"] = 0
        return None

    monkeypatch.setattr(weather, "_fetch_batch_range", always_none)
    now = datetime(2026, 10, 11, 6, 0, tzinfo=timezone.utc)
    targets = [*_TWO_WINDOW_TARGETS,
               _target("20261020_C@HOME", "2026-10-20T17:00:00Z")]

    assert weather._fetch_batched_weather(targets, now) == {}


def test_rate_limited_out_boundary():
    """The abort line is exactly _RATE_LIMIT_ABORT_STREAK dead batches."""
    weather._reset_rate_limiter()
    assert not weather._rate_limited_out()
    weather._rate_limit["streak"] = weather._RATE_LIMIT_ABORT_STREAK - 1
    assert not weather._rate_limited_out()
    weather._rate_limit["streak"] = weather._RATE_LIMIT_ABORT_STREAK
    assert weather._rate_limited_out()


def test_dead_batches_cost_one_request_each_after_the_first(
        monkeypatch, sleeps, caplog):
    """The 15:50-run contract, pinned: a dead quota minute costs ONE
    request per batch, never a re-armed ladder.

    Five back-to-back all-429 calls (the storm shape) must cost exactly
    4 + 4 requests — two quick retries + cooldown + probe on the first,
    then one entry probe each — with at most one cooldown sleep per call,
    every wait capped at the 4-minute ceiling, and the streak climbing
    the whole way.  This is the DEFAULT path (recent/forecast phases,
    top-up, any non-sweep caller); the archive sweep uses defer mode
    instead — see test_defer_mode_never_cools_down_and_costs_one_request.
    """
    calls = _patch_get(monkeypatch, lambda n: _Resp(429))

    with caplog.at_level(logging.WARNING):
        weather._get_with_retry("https://example.invalid", {})          # 1
    assert len(calls) == 4, f"first dead batch: 2 quick + cooldown + probe: {calls}"
    per_call = [len(calls)]
    for _ in range(4):
        before = len(calls)
        sleeps.clear()
        weather._get_with_retry("https://example.invalid", {})
        per_call.append(len(calls) - before)
        assert len(sleeps) == 1, (
            "a closed window costs one cooldown wait, not a ladder: "
            f"{sleeps}")
        assert sleeps[0] <= weather._RATE_LIMIT_COOLDOWN_MAX_SEC, sleeps

    assert per_call == [4, 1, 1, 1, 1], (
        f"dead batches after the first must cost one request each: {per_call}")
    assert weather._rate_limit_streak() == 6, (
        "every dead batch opens one more streak unit toward the abort")
    # Budget accounting: only the FIRST batch spent budget (2 quick retries
    # + 1 cooldown-probe = 3 of _429_RETRY_BUDGET).  Entry probes are
    # already minimal — one request, no retries — and are bounded by the
    # abort streak instead, exactly as in the first remediation.
    assert weather._rate_limit["retries"] == 3, weather._rate_limit
    assert weather._rate_limit["retries"] < weather._429_RETRY_BUDGET


def test_sweep_pace_paces_below_the_observed_trip_rate():
    """The batch pause keeps the sweep under the rate that trips the shared
    IP's quota.

    The 15:50 run measured ~39 requests/min (444 batches in 689s with the
    old 1.0s pause) and the quota closed intermittently from minute ~8.
    Open-Meteo enforces its limits per IP and Kaggle's egress IP is shared,
    so the pace is pinned at ≥ 2.0s (~23 requests/min, ~19 min for a full
    repull) — halving our contribution while staying far below the ~30-min
    storm cost this breaker replaced.
    """
    assert weather._BATCH_PAUSE_SEC >= 2.0, (
        "sweep pacing below 2.0s re-opens the shared-IP 429 bursts "
        "observed on 2026-10-11 — tune deliberately, with a new review")
