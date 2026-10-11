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

Remediation (``nfl-backend/backend/weather.py``, structure mirrors
``mlb-backend/backend/weather.py``'s documented-constants + retry + batch +
top-up layout, which never storms because its per-run request volume is
tiny):

* a batch dead to a retriable status OPENS the window — the next request
  cools down 60s (one quota minute), doubling per consecutive dead batch to
  a 4-minute cap, then probes ONCE instead of laddering;
* any definitive (non-retriable) response CLOSES the window (streak reset);
* a per-process 429 budget (40) bounds rate-limited retries outright — past
  it a rate-limited request fails after one attempt, logged once;
* _RATE_LIMIT_ABORT_STREAK (12) consecutive dead batches abort the
  remaining archive sweep with a count of what was skipped; the same-run
  top-up and the next run's cache-miss pass remain the recovery paths
  (features NaN, never substituted) -- the line sits just above the worst
  block ever observed (29 minutes, this run) so that size still recovers
* timeouts: 20s per attempt, and three consecutive hangs end the request
  (≈63s) instead of seven 45s hangs (≈6.5 min per dead batch).

Thresholds are pinned here so a future tuning change is a deliberate,
reviewed edit rather than a silent drift back toward storming.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
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
    """An isolated 429 keeps the documented 7-attempt ladder (2**n + jitter,
    Retry-After authoritative) and never opens the breaker."""
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
    # 6 retries max were available; only 2 were consumed.
    assert weather._rate_limit["retries"] == 2


def test_closed_window_cools_down_then_probes_once(
        monkeypatch, sleeps, caplog):
    """The storm shape from the delivered log: every request answers 429.

    First call burns the isolated-blip ladder ONCE; afterwards each call
    cools down for the quota window (60s → 120s → … capped at 240s) and
    makes exactly ONE request — and a definitive answer resets the streak.
    """
    calls = _patch_get(monkeypatch, lambda n: _Resp(429))

    with caplog.at_level(logging.WARNING):
        weather._get_with_retry("https://example.invalid", {})
    assert len(calls) == weather._RETRY_ATTEMPTS, calls
    assert len(sleeps) == weather._RETRY_ATTEMPTS - 1, sleeps
    assert weather._rate_limit_streak() == 1

    sleeps.clear()
    weather._get_with_retry("https://example.invalid", {})
    assert sleeps[0] == weather._RATE_LIMIT_COOLDOWN_SEC, sleeps
    assert len(sleeps) == 1, f"a closed window probes once, not a ladder: {sleeps}"
    assert len(calls) == weather._RETRY_ATTEMPTS + 1, calls
    assert weather._rate_limit_streak() == 2

    sleeps.clear()
    reopened = _patch_get(monkeypatch, lambda n: _Resp(200))
    weather._get_with_retry("https://example.invalid", {})
    assert sleeps == [120.0], f"cooldown doubles per dead batch: {sleeps}"
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
    assert len(cooldowns) == 2, cooldowns


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


def test_dead_batch_streak_aborts_the_archive_sweep(monkeypatch, sleeps, caplog):
    """12 consecutive dead batches stop the remaining archive sweep loudly
    instead of re-arming ladders for hundreds of windows; the recovery path
    (same-run top-up + next run's cache-miss pass) is named in the warning."""
    seen: list[str] = []

    def fake_batch_range(locations, start_date, end_date, *, source, **kw):
        seen.append(source)
        return {}

    monkeypatch.setattr(weather, "_fetch_batch_range", fake_batch_range)
    now = datetime(2026, 10, 11, 6, 0, tzinfo=timezone.utc)
    targets = [{
        "game_id": "20260911_TEST@HOME",
        "stadium": "Test Stadium",
        "lat": 39.0,
        "lon": -95.0,
        "kickoff_utc": pd.Timestamp("2026-09-11T17:00:00Z"),
    }]

    # Control: an open window runs the archive sweep normally.
    with caplog.at_level(logging.WARNING):
        weather._fetch_batched_weather(targets, now)
    assert weather.OPEN_METEO_ARCHIVE in seen, "control sweep must run"
    archive_calls = seen.count(weather.OPEN_METEO_ARCHIVE)
    assert archive_calls >= 1

    # Tripped: the sweep stops before its first request and says so.
    weather._reset_rate_limiter()
    weather._rate_limit["streak"] = weather._RATE_LIMIT_ABORT_STREAK
    seen.clear()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        weather._fetch_batched_weather(targets, now)

    assert weather.OPEN_METEO_ARCHIVE not in seen, (
        "an aborted sweep must send no archive requests")
    aborted = [m for m in caplog.messages if "archive sweep aborted" in m]
    assert len(aborted) == 1, caplog.messages
    assert "not attempted" in aborted[0]
    assert "top-up" in aborted[0] and "cache-miss" in aborted[0], (
        "the warning must name the recovery paths")


def test_rate_limited_out_boundary():
    """The abort line is exactly _RATE_LIMIT_ABORT_STREAK dead batches."""
    weather._reset_rate_limiter()
    assert not weather._rate_limited_out()
    weather._rate_limit["streak"] = weather._RATE_LIMIT_ABORT_STREAK - 1
    assert not weather._rate_limited_out()
    weather._rate_limit["streak"] = weather._RATE_LIMIT_ABORT_STREAK
    assert weather._rate_limited_out()
