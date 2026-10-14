"""2026-10-11/13 NFL run-log reviews: Open-Meteo rate limits under a FULL REPULL.

Evidence history (``data_delivery/nfl_pipeline_run_log.txt``):

* 10-07 -> 10-11: storms GROWING run over run — 13 -> 72 -> 119 -> 156
  ``Open-Meteo 429; retrying`` lines across 25 back-to-back dead batches,
  because each batch re-armed its own ladder into a window that could not
  answer with anything else (05:30:35 -> 05:39:42 in the 10-11 05:30 run).
* 10-11 15:50: the first breaker bounded the sweep but burned 21 logged
  429 retries on 1/2/4s waits inside quota minutes that stayed shut.
* 10-11 18:10: cooldown sleeping was itself the wall-clock risk — a
  ~37-minute shared-IP flap stretched a ~26-minute sweep to 51m28s, so the
  archive sweep learned to NEVER sleep on a 429 (defer-and-requeue).
* 10-13 10:23: defer/requeue bounded everything correctly, yet 142/2023
  games still shipped weather-less — the sweep alone was 448 requests over
  ~20 minutes, three recovery passes all met the same shut window, and the
  delivered log's final-pass messages promised a same-run top-up that was
  itself the pass aborting.

Remediation (``nfl-backend/backend/weather.py``, 2026-10-13 structural
review — MLB for structural guidance, guardrail: a full repull every run
WITHOUT triggering rate limits):

* **Full repull, no cache gating**: ``fetch_games_weather`` requests every
  eligible game every run; the cache file is provenance history for the
  --skip-pull reader only and never decides what is fetched, and delivery
  is what THAT run observed (failed fetch -> NaN, never a backfilled row).
* **MLB drain ladder**: one ``2**attempt`` + jitter ladder (7 attempts,
  1+2+4+8+16+32 = 63s, Retry-After authoritative) for every retriable
  status — 63s outlasts one Open-Meteo quota minute, so a rate-limited
  batch DRAINS instead of being lost.
* **Fail-fast brake**: ``_DRAIN_FAIL_FAST_AFTER`` (2) consecutive batches
  that spend the whole ladder on a 429 engage the brake (logged once);
  then a still-shut window costs ONE request per batch, the paced sweep
  keeps probing, and any definitive answer re-arms draining.  Bounds a
  37-minute flap at 2 x 63s of sleep + fast-fail batches.
* **Request budget**: ``_BATCH_DAYS`` 14 -> 60 (already measured: a decade
  of history drops to ~111 request batches), 15 locations per batch,
  ``_BATCH_PAUSE_SEC`` 2.0 — a full repull is ~132 requests over ~6-8
  minutes instead of 448 over ~20.
* **Same-run top-up** remains the in-run gap recovery (fresh breaker,
  gaps only); the next run's full repull is the durable recovery.  No
  requeue pass exists anymore — deferred/braked batches are simply missing
  until one of those two recovers them.
* Timeouts: 20s per attempt, three consecutive hangs end the request.

Thresholds are pinned here so a future tuning change is a deliberate,
reviewed edit rather than a silent drift back toward storming.
"""
from __future__ import annotations

import logging
import sys
import tempfile
import warnings
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


def _brake_messages(caplog):
    return [m for m in caplog.messages if "rate-limit brake" in m]


def test_isolated_blip_rides_the_ladder_with_retry_after_authority(
        monkeypatch, sleeps, caplog):
    """A marginal 429 blip drains through the ladder (Retry-After
    authoritative) and a 200 re-arms — nothing persistent is recorded."""
    seq = [_Resp(429, {"Retry-After": "7"}), _Resp(429), _Resp(200)]
    calls = _patch_get(monkeypatch, lambda n: seq[n - 1])

    with caplog.at_level(logging.WARNING):
        resp = weather._get_with_retry("https://example.invalid", {})

    assert resp.status_code == 200
    assert len(calls) == 3, calls
    assert len(sleeps) == 2, sleeps
    assert sleeps[0] >= 7.0, f"Retry-After must be authoritative: {sleeps}"
    assert 2.0 <= sleeps[1] <= 2.5, sleeps
    assert weather._rate_limit == {"drain_fails": 0, "brake": False}, (
        "a recovered blip records nothing")
    assert all(t == weather._REQUEST_TIMEOUT_SEC for t in calls), calls
    assert not _brake_messages(caplog)


def test_closed_window_drains_one_quota_minute_then_brakes(
        monkeypatch, sleeps, caplog):
    """    The storm shape from the delivered log: every request answers 429.

    Each batch spends MLB's full drain ladder once — 7 requests, waits
    1+2+4+8+16+32 = 63s, which outlasts one quota minute.  Two such
    batches in a row engage the brake (logged once); from then on a 429
    costs exactly ONE request and ZERO sleeps.
    """
    monkeypatch.setattr(weather.random, "uniform", lambda a, b: 0.0)
    calls = _patch_get(monkeypatch, lambda n: _Resp(429))

    with caplog.at_level(logging.WARNING):
        resp = weather._get_with_retry("https://example.invalid", {})
    assert resp.status_code == 429, "a drained batch returns the 429"
    assert len(calls) == weather._RETRY_ATTEMPTS, calls
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0], sleeps
    assert weather._rate_limit["drain_fails"] == 1
    assert not weather._rate_limit["brake"], "one failure is not yet a flap"

    # Second consecutive full-ladder failure engages the brake, once.
    sleeps.clear()
    with caplog.at_level(logging.WARNING):
        weather._get_with_retry("https://example.invalid", {})
    assert len(calls) == 2 * weather._RETRY_ATTEMPTS, calls
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    assert weather._rate_limit["brake"] is True
    assert len(_brake_messages(caplog)) == 1

    # Braked: one request, no sleeps, no repeated disclosure.
    sleeps.clear()
    with caplog.at_level(logging.WARNING):
        weather._get_with_retry("https://example.invalid", {})
    assert len(calls) == 2 * weather._RETRY_ATTEMPTS + 1, (
        "a braked 429 must cost exactly one request")
    assert sleeps == [], f"the brake must never sleep: {sleeps}"
    assert len(_brake_messages(caplog)) == 1, "brake disclosure is one-shot"


def test_a_success_rearms_draining_after_the_brake(monkeypatch, sleeps, caplog):
    """The paced sweep keeps probing under the brake, and the first
    definitive answer re-arms the ladder (drain again, brake clears)."""
    _patch_get(monkeypatch, lambda n: _Resp(429))
    with caplog.at_level(logging.WARNING):
        weather._get_with_retry("https://example.invalid", {})
        weather._get_with_retry("https://example.invalid", {})
    assert weather._rate_limit["brake"] is True

    reopened = _patch_get(monkeypatch, lambda n: _Resp(200))
    sleeps.clear()
    weather._get_with_retry("https://example.invalid", {})
    assert len(reopened) == 1, reopened
    assert weather._rate_limit == {"drain_fails": 0, "brake": False}, (
        "a definitive answer re-arms draining and clears the brake")

    # The next closure drains the FULL ladder again (not fast-fail).
    calls = _patch_get(monkeypatch, lambda n: _Resp(429))
    sleeps.clear()
    weather._get_with_retry("https://example.invalid", {})
    assert len(calls) == weather._RETRY_ATTEMPTS, calls
    assert len(sleeps) == weather._RETRY_ATTEMPTS - 1, sleeps


def test_gateway_errors_share_the_ladder_but_never_the_brake(
        monkeypatch, sleeps, caplog):
    """502s answer the same drain ladder (MLB parity) but are NOT quota
    events: they never count toward, never engage, and never clear the
    rate-limit brake."""
    seq = [_Resp(503)] * (weather._RETRY_ATTEMPTS - 1) + [_Resp(503)]
    calls = _patch_get(monkeypatch, lambda n: seq[min(n - 1, len(seq) - 1)])

    with caplog.at_level(logging.WARNING):
        resp = weather._get_with_retry("https://example.invalid", {})
    assert resp.status_code == 503
    assert len(calls) == weather._RETRY_ATTEMPTS, calls
    assert len(sleeps) == weather._RETRY_ATTEMPTS - 1, sleeps
    assert weather._rate_limit == {"drain_fails": 0, "brake": False}, (
        "gateway trouble is not a quota event")

    # A 200 must NOT be required to clear a brake that was never set.
    _patch_get(monkeypatch, lambda n: _Resp(429))
    weather._get_with_retry("https://example.invalid", {})
    assert weather._rate_limit["drain_fails"] == 1


def test_timeout_cascade_ends_after_three_hangs(monkeypatch, sleeps, caplog):
    """A stalled endpoint must not cost 7 x 20s per batch: three hangs end
    the request (~63s with backoff), the batch dies, and timeouts never
    touch the quota brake (a stalled connection is not a closed window)."""
    calls = _patch_get(monkeypatch, lambda n: weather.requests.Timeout("hang"))

    with caplog.at_level(logging.WARNING), pytest.raises(weather.requests.Timeout):
        weather._get_with_retry("https://example.invalid", {})

    assert len(calls) == weather._TIMEOUT_RETRY_LIMIT, calls
    assert len(sleeps) == weather._TIMEOUT_RETRY_LIMIT - 1, sleeps
    assert weather._rate_limit == {"drain_fails": 0, "brake": False}
    assert all(t == weather._REQUEST_TIMEOUT_SEC for t in calls), calls


def _target(game_id: str, kickoff: str) -> dict:
    return {
        "game_id": game_id,
        "stadium": "Test Stadium",
        "lat": 39.0,
        "lon": -95.0,
        "kickoff_utc": pd.Timestamp(kickoff),
    }


# Kickoffs 80 days apart -> TWO 60-day archive windows (the days between
# hold no game, so exactly these two windows are planned, as in production).
_TWO_WINDOW_TARGETS = [
    _target("20260601_A@HOME", "2026-06-01T17:00:00Z"),
    _target("20260820_B@HOME", "2026-08-20T17:00:00Z"),
]


def test_fetch_batch_range_returns_none_on_429_with_honest_recovery_line(
        monkeypatch, sleeps, caplog):
    """Rate-limited is never dead-for-good: ``None`` (not ``{}``), with a
    line naming the recovery that REALLY exists — the same-run top-up and
    the next run's full repull.  The 10-13 10:23 log's line named a
    \"requeue\" and a \"cache-miss pass\"; neither exists anymore."""
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
    honest = [m for m in caplog.messages if "429 after the drain ladder" in m]
    assert len(honest) == 1, caplog.messages
    assert "same-run top-up" in honest[0], honest[0]
    assert "next run's full repull" in honest[0], honest[0]
    assert "requeue" not in honest[0] and "cache-miss" not in honest[0], (
        "the removed recovery paths must never be promised again: "
        f"{honest[0]}")


def test_fetch_batch_range_is_silent_while_the_brake_is_engaged(
        monkeypatch, sleeps, caplog):
    """One line per fast-failed batch would bury a long closure (the
    10-11 review): once the brake owns the disclosure, terminal 429s stay
    quiet and still return ``None``."""
    _patch_get(monkeypatch, lambda n: _Resp(429))
    with caplog.at_level(logging.WARNING):
        weather._get_with_retry("https://example.invalid", {})
        weather._get_with_retry("https://example.invalid", {})
    assert weather._rate_limit["brake"] is True
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        result = weather._fetch_batch_range(
            [("Test Stadium", 39.0, -95.0)],
            date(2026, 9, 1), date(2026, 9, 14),
            source=weather.OPEN_METEO_ARCHIVE)
    assert result is None
    assert not any("429 after the drain ladder" in m
                   for m in caplog.messages), caplog.messages


def test_fetch_batch_range_generic_failure_returns_empty(monkeypatch, caplog):
    """Non-429 failures (gateway exhaustion, parse errors) return ``{}``
    and are logged as failed batches with their locations named."""
    _patch_get(monkeypatch, lambda n: _Resp(503))

    with caplog.at_level(logging.WARNING):
        result = weather._fetch_batch_range(
            [("Test Stadium", 39.0, -95.0)],
            date(2026, 9, 1), date(2026, 9, 14),
            source=weather.OPEN_METEO_ARCHIVE)

    assert result == {}, result
    assert any("batch failed" in m and "Test Stadium" in m
               for m in caplog.messages), caplog.messages


def test_sweep_completes_under_a_shut_window_leaving_days_missing(
        monkeypatch, caplog):
    """The sweep has no requeue and no defer mode anymore: a window the
    drain ladder cannot save contributes NOTHING to ``by_key`` (its days
    stay missing), the other windows still land, and the pass runs to
    completion with no requeue/abort message anywhere."""
    def fake_batch_range(locations, start_date, end_date, *, source, **kw):
        if source != weather.OPEN_METEO_ARCHIVE:
            return None                      # recent phase: also rate-limited
        if start_date <= date(2026, 6, 1) <= end_date:
            return {("Test Stadium", date(2026, 6, 1)): {"time": []}}
        return None                          # the second window is 429-dead

    monkeypatch.setattr(weather, "_fetch_batch_range", fake_batch_range)
    now = datetime(2026, 10, 13, 6, 0, tzinfo=timezone.utc)

    with caplog.at_level(logging.INFO):
        out = weather._fetch_batched_weather(_TWO_WINDOW_TARGETS, now)

    assert {day for (_stadium, day) in out} == {date(2026, 6, 1)}, (
        "the open window lands; the shut window stays missing")
    assert not any("requeue" in m or "deferred" in m
                   for m in caplog.messages), caplog.messages


def test_a_reopened_window_lands_the_remaining_windows(
        monkeypatch, caplog):
    """The recovery story without a requeue: whatever the sweep could not
    fetch is missing here and recovered by the same-run top-up — which is
    exactly the gap set (this run's missing targets), refetched with a
    fresh breaker.  Simulated end-to-end through fetch_games_weather."""
    fetch_calls: list[list[str]] = []

    def _series(target):
        return {(target["stadium"], target["kickoff_utc"].date()): {
                "time": [target["kickoff_utc"].strftime("%Y-%m-%dT16:00:00Z")],
                "temperature_2m": [70.0], "wind_speed_10m": [5.0],
                "precipitation": [0.0], "snowfall": [0.0],
                "_source": weather.OPEN_METEO_ARCHIVE,
                "_fetched_at_utc": pd.Timestamp("2026-06-03T12:00:00Z")}}

    def fake_fetch(targets, now, **kw):
        fetch_calls.append([t["game_id"] for t in targets])
        # Pass 1 (the sweep) loses B; pass 2 (the top-up) recovers it.
        out = {}
        for t in targets:
            if len(fetch_calls) == 1 and t["game_id"].endswith("B@HOME"):
                continue
            out.update(_series(t))
        return out

    monkeypatch.setattr(weather, "_fetch_batched_weather", fake_fetch)
    gdf = pd.DataFrame([
        {"game_id": "20260601_A@HOME", "season": 2026, "week": 1,
         "gameday": "2026-06-01", "gametime": "13:00", "home_team": "A",
         "away_team": "B", "home_score": 20.0, "away_score": 10.0,
         "game_type": "REG", "roof": "outdoors", "div_game": 0,
         "stadium": "MetLife Stadium", "surface": "grass"},
        {"game_id": "20260602_B@HOME", "season": 2026, "week": 1,
         "gameday": "2026-06-02", "gametime": "13:00", "home_team": "A",
         "away_team": "B", "home_score": 20.0, "away_score": 10.0,
         "game_type": "REG", "roof": "outdoors", "div_game": 0,
         "stadium": "Gillette Stadium", "surface": "grass"},
    ])

    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        got = weather.fetch_games_weather(
            gdf, path=Path(tmp) / "w.parquet",
            now=pd.Timestamp("2026-06-03T12:00:00Z"))

    assert len(fetch_calls) == 2, (
        f"exactly one sweep + one gap-only top-up: {fetch_calls}")
    assert sorted(fetch_calls[0]) == ["20260601_A@HOME", "20260602_B@HOME"]
    assert fetch_calls[1] == ["20260602_B@HOME"], (
        "the top-up refetches ONLY the gaps")
    assert set(got["game_id"]) == {"20260601_A@HOME", "20260602_B@HOME"}


def test_top_up_starts_with_a_fresh_breaker(monkeypatch, caplog):
    """The top-up must re-probe a window that may have reopened: it calls
    _reset_rate_limiter() first, clearing a brake the sweep may have set
    (the 10-11 18:10 run recovered all 48 gaps in 12s after a reset)."""
    resets: list[bool] = []
    original_reset = weather._reset_rate_limiter

    def spy_reset():
        resets.append(True)
        original_reset()

    fetch_calls: list[list[str]] = []

    def _series(target):
        return {(target["stadium"], target["kickoff_utc"].date()): {
                "time": [target["kickoff_utc"].strftime("%Y-%m-%dT16:00:00Z")],
                "temperature_2m": [70.0], "wind_speed_10m": [5.0],
                "precipitation": [0.0], "snowfall": [0.0],
                "_source": weather.OPEN_METEO_ARCHIVE,
                "_fetched_at_utc": pd.Timestamp("2026-06-03T12:00:00Z")}}

    def fake_fetch(targets, now, **kw):
        fetch_calls.append([t["game_id"] for t in targets])
        if len(fetch_calls) == 1:
            # The sweep ends braked and loses one game.
            weather._rate_limit.update(drain_fails=5, brake=True)
            return _series(targets[0])
        assert not weather._rate_limit["brake"], (
            "the top-up must start with a cleared brake")
        return _series(targets[0])

    monkeypatch.setattr(weather, "_reset_rate_limiter", spy_reset)
    monkeypatch.setattr(weather, "_fetch_batched_weather", fake_fetch)
    gdf = pd.DataFrame([
        {"game_id": "TOP_A", "season": 2026, "week": 1,
         "gameday": "2026-06-01", "gametime": "13:00", "home_team": "A",
         "away_team": "B", "home_score": 20.0, "away_score": 10.0,
         "game_type": "REG", "roof": "outdoors", "div_game": 0,
         "stadium": "MetLife Stadium", "surface": "grass"},
        {"game_id": "TOP_B", "season": 2026, "week": 1,
         "gameday": "2026-06-02", "gametime": "13:00", "home_team": "A",
         "away_team": "B", "home_score": 20.0, "away_score": 10.0,
         "game_type": "REG", "roof": "outdoors", "div_game": 0,
         "stadium": "Gillette Stadium", "surface": "grass"},
    ])

    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        weather.fetch_games_weather(
            gdf, path=Path(tmp) / "w.parquet",
            now=pd.Timestamp("2026-06-03T12:00:00Z"))

    assert fetch_calls == [["TOP_A", "TOP_B"], ["TOP_B"]], fetch_calls
    assert len(resets) == 1, "exactly one fresh-breaker reset, before pass 2"
    assert weather._rate_limit == {"drain_fails": 0, "brake": False}


def test_recent_and_forecast_phases_skip_a_rate_limited_batch(monkeypatch):
    """The recent/forecast phases receive ``None`` from a rate-limited
    batch; calling ``.items()`` on it would raise straight out of the
    weather phase.  They skip and leave the gap to the top-up."""
    def always_none(locations, start_date, end_date, *, source, **kw):
        return None

    monkeypatch.setattr(weather, "_fetch_batch_range", always_none)
    now = datetime(2026, 10, 13, 6, 0, tzinfo=timezone.utc)
    targets = [*_TWO_WINDOW_TARGETS,
               _target("20261020_C@HOME", "2026-10-20T17:00:00Z")]

    assert weather._fetch_batched_weather(targets, now) == {}


# ---------------------------------------------------------------------------
# Full repull guardrails (2026-10-13 structural review)
#
# The request: "the nfl pipeline must not use cache files to top off
# weather observations ... optimized for a full repull every pipeline run
# and avoid rate limits ... weather must be able to fully repopulate every
# pipeline run without triggering rate limits."
# ---------------------------------------------------------------------------

def _full_repull_schedule() -> pd.DataFrame:
    return pd.DataFrame([
        {"game_id": "FR_A", "season": 2024, "week": 1,
         "gameday": "2024-09-01", "gametime": "13:00", "home_team": "A",
         "away_team": "B", "home_score": 20.0, "away_score": 10.0,
         "game_type": "REG", "roof": "outdoors", "div_game": 0,
         "stadium": "MetLife Stadium", "surface": "grass"},
        {"game_id": "FR_B", "season": 2024, "week": 1,
         "gameday": "2024-09-02", "gametime": "13:00", "home_team": "A",
         "away_team": "B", "home_score": 20.0, "away_score": 10.0,
         "game_type": "REG", "roof": "outdoors", "div_game": 0,
         "stadium": "Gillette Stadium", "surface": "grass"},
    ])


def _warm_cache(path: Path) -> None:
    """A PIT-valid cached row for BOTH schedule games (the old top-off's
    reason to fetch nothing)."""
    rows = [
        {"game_id": gid, "stadium": stadium,
         "kickoff_utc": pd.Timestamp(f"{day} 17:00:00Z"),
         "weather_time_utc": pd.Timestamp(f"{day} 16:00:00Z"),
         "fetched_at_utc": pd.Timestamp(f"{day} 18:00:00Z"),
         "source": weather.OPEN_METEO_ARCHIVE, "temp_f": 70.0,
         "wind_mph": 5.0, "precip_in": 0.0, "snow_in": 0.0,
         "is_precip": 0.0, "is_snow": 0.0}
        for gid, stadium, day in (("FR_A", "MetLife Stadium", "2024-09-01"),
                                  ("FR_B", "Gillette Stadium", "2024-09-02"))
    ]
    weather._save_cache(path, pd.DataFrame(rows, columns=list(weather.CACHE_COLUMNS)))


def test_a_warm_cache_never_top_off_the_fetch(monkeypatch):
    """THE guardrail: with a fully populated cache, the run still requests
    EVERY eligible game.  (The old design computed ``missing`` from the
    cache and fetched nothing here.)"""
    gdf = _full_repull_schedule()
    fetch_calls: list[list[str]] = []

    def fake_fetch(targets, now, **kw):
        fetch_calls.append([t["game_id"] for t in targets])
        out = {}
        for t in targets:
            out[(t["stadium"], t["kickoff_utc"].date())] = {
                "time": [t["kickoff_utc"].strftime("%Y-%m-%dT16:00:00Z")],
                "temperature_2m": [70.0], "wind_speed_10m": [5.0],
                "precipitation": [0.0], "snowfall": [0.0],
                "_source": weather.OPEN_METEO_ARCHIVE,
                "_fetched_at_utc": pd.Timestamp("2024-09-03T12:00:00Z"),
            }
        return out

    monkeypatch.setattr(weather, "_fetch_batched_weather", fake_fetch)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "w.parquet"
        _warm_cache(path)
        assert len(weather.cached_weather_for_games(gdf, path=path)) == 2, (
            "the cache must be warm for this test to mean anything")
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            got = weather.fetch_games_weather(
                gdf, path=path, now=pd.Timestamp("2024-09-03T12:00:00Z"))

    assert fetch_calls == [["FR_A", "FR_B"]], (
        f"a warm cache must not shrink the fetch: {fetch_calls}")
    assert set(got["game_id"]) == {"FR_A", "FR_B"}


def test_a_failed_fetch_never_backfills_from_the_cache(monkeypatch):
    """Delivery is THIS run's observations only: when the fetch fails the
    run ships NaN for those games even though valid older rows exist on
    disk — the next run's full repull (not the file) recovers them."""
    gdf = _full_repull_schedule()
    fetch_calls: list[list[str]] = []

    def empty_fetch(targets, now, **kw):
        fetch_calls.append([t["game_id"] for t in targets])
        return {}

    monkeypatch.setattr(weather, "_fetch_batched_weather", empty_fetch)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "w.parquet"
        _warm_cache(path)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            got = weather.fetch_games_weather(
                gdf, path=path, now=pd.Timestamp("2024-09-04T12:00:00Z"))
        # The cache file itself still holds the history for --skip-pull.
        assert len(weather.cached_weather_for_games(gdf, path=path)) == 2

    assert len(got) == 0, (
        f"cache rows must never backfill delivery: {list(got.get('game_id', []))}")
    assert len(fetch_calls) == 2, (
        "sweep + gap-only top-up, both attempted: %s" % fetch_calls)


def test_refresh_is_a_documented_noop(monkeypatch):
    """``refresh=`` survives for build_weather_table compatibility; it must
    not change what is fetched — every run fetches everything."""
    gdf = _full_repull_schedule()
    fetch_calls: list[int] = []

    def fake_fetch(targets, now, **kw):
        fetch_calls.append(len(targets))
        return {}

    monkeypatch.setattr(weather, "_fetch_batched_weather", fake_fetch)
    with tempfile.TemporaryDirectory() as tmp:
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            weather.fetch_games_weather(
                gdf, path=Path(tmp) / "a.parquet", refresh=False,
                now=pd.Timestamp("2024-09-03T12:00:00Z"))
            weather.fetch_games_weather(
                gdf, path=Path(tmp) / "b.parquet", refresh=True,
                now=pd.Timestamp("2024-09-03T12:00:00Z"))

    assert fetch_calls == [2, 2, 2, 2], (
        "per run: sweep + gap-only top-up (the fake yields nothing), "
        f"identically with refresh on or off: {fetch_calls}")


def test_full_repul_request_budget_pinned():
    """The rate-limit geometry: 60-day windows x 15-location batches keep
    a decade-long FULL repull near ~132 requests (measured ~111 for the
    production date span) instead of the 448-request 14-day sweep that
    lost 142 games on 2026-10-13."""
    assert weather._BATCH_DAYS == 60, (
        "window width drives the request count; tune deliberately")
    assert weather._BATCH_SIZE == 15, (
        "MLB-proven response size per multi-coordinate request")

    needed = set()
    for yr in range(2016, 2027):
        for off in range(0, 130, 3):        # dense game cluster per season
            needed.add(date(yr, 9, 1) + timedelta(days=off))
    start, end = date(2016, 1, 1), date(2026, 10, 13)
    plan = []
    for off in range(0, (end - start).days + 1, weather._BATCH_DAYS):
        cs = start + timedelta(days=off)
        ce = min(cs + timedelta(days=weather._BATCH_DAYS - 1), end)
        if any(cs <= d <= ce for d in needed):
            for i in range(0, 50, weather._BATCH_SIZE):
                plan.append((cs, ce, i))
    assert len(plan) <= 140, (
        f"a full repull must stay near ~132 requests, got {len(plan)}")


def test_sweep_pace_paces_below_the_observed_trip_rate():
    """The batch pause keeps the sweep under the rate that trips the shared
    IP's quota.

    The 15:50 run measured ~39 requests/min (444 batches in 689s with the
    old 1.0s pause) and the quota closed intermittently from minute ~8.
    Open-Meteo enforces its limits per IP and Kaggle's egress IP is shared,
    so the pace stays pinned at >= 2.0s (~15 requests/min over the much
    shorter 132-batch sweep) — tune deliberately, with a new review.
    """
    assert weather._BATCH_PAUSE_SEC >= 2.0, (
        "sweep pacing below 2.0s re-opens the shared-IP 429 bursts "
        "observed on 2026-10-11 — tune deliberately, with a new review")


def test_the_defer_requeue_machinery_is_gone():
    """The 2026-10-13 structural review replaced defer-and-requeue (three
    recovery passes, 142 games still lost) with the drain ladder + brake
    + full repull.  The removed entry points must not quietly return."""
    for symbol in ("_requeue_deferred_batches", "_recovery_note",
                   "_rate_limited_out", "_cooldown_wait",
                   "_rate_limit_streak"):
        assert not hasattr(weather, symbol), (
            f"{symbol} belongs to the removed defer/requeue design")
    import inspect
    sig = inspect.signature(weather._get_with_retry)
    assert "defer_on_429" not in sig.parameters, sig
    sig = inspect.signature(weather._fetch_batch_range)
    assert "defer_on_429" not in sig.parameters, sig


# ---------------------------------------------------------------------------
# 2026-10-13 second pass: MLB structure parity of the Open-Meteo machinery
#
# The 16:22 production run was CLEAN (136 batches, zero 429s, 2023/2023
# eligible games), so this pass compares the SHAPE of the sweep with MLB's:
# one request per PHYSICAL venue (not per stadium name), every skipped
# window reported after the sweep, and a starved season named loudly even
# under a healthy-looking aggregate.
# ---------------------------------------------------------------------------

def _site_target(game_id: str, kickoff: str, stadium: str,
                 lat: float, lon: float) -> dict:
    return {"game_id": game_id, "stadium": stadium, "lat": lat, "lon": lon,
            "kickoff_utc": pd.Timestamp(kickoff)}


def test_rename_duplicate_stadiums_share_one_coordinate_request(
        monkeypatch, caplog, sleeps):
    """MLB keys its fetch locations ONE entry per physical venue; the NFL
    sweep keyed one per stadium NAME, so every rename pair (Heinz Field /
    Acrisure Stadium is a single coordinate) spent a second copy of quota
    on coordinates already inside the same batch — the 2026-10-13 run's 50
    names were 37 physical sites.  Each request must carry every
    coordinate exactly once, and the series must fan back out to every
    name so the delivered keys are unchanged."""
    requests_sent: list[list[tuple[str, float, float]]] = []

    def fake_batch_range(locations, start_date, end_date, *, source, **kw):
        requests_sent.append(list(locations))
        out = {}
        for name, lat, lon in locations:
            for offset in range((end_date - start_date).days + 1):
                day = start_date + timedelta(days=offset)
                # Content depends ONLY on the coordinates: identical sites
                # must produce identical series whatever they are called.
                out[(name, day)] = {
                    "time": [pd.Timestamp(f"{day} 16:00:00Z")],
                    "temperature_2m": [float(lat)],
                    "wind_speed_10m": [float(lon)],
                    "_source": source,
                    "_fetched_at_utc": pd.Timestamp("2026-10-13T06:00:00Z"),
                }
        return out

    monkeypatch.setattr(weather, "_fetch_batch_range", fake_batch_range)
    targets = [
        _site_target("OLD", "2018-09-02T17:00:00Z", "Heinz Field",
                     40.446667, -80.015833),
        _site_target("NEW", "2024-09-02T17:00:00Z", "Acrisure Stadium",
                     40.446667, -80.015833),   # ONE physical site, new name
        _site_target("NY", "2024-09-03T17:00:00Z", "MetLife Stadium",
                     40.8135, -74.0744),
    ]
    now = datetime(2026, 10, 13, 6, 0, tzinfo=timezone.utc)

    with caplog.at_level(logging.INFO):
        out = weather._fetch_batched_weather(targets, now)

    assert requests_sent, "the sweep must send requests"
    for locations in requests_sent:
        coords = [(lat, lon) for _, lat, lon in locations]
        assert len(coords) == len(set(coords)), (
            f"a duplicate coordinate rode inside one batch: {locations}")
        assert len(locations) == 2, (
            "3 names are 2 physical sites — exactly one request per site, "
            f"got {locations}")
        assert "Acrisure Stadium" not in [name for name, _, _ in locations], (
            "the rename alias must never ship: its coordinate is already "
            "in this request")

    def days_for(stadium):
        return {day for (name, day) in out if name == stadium}

    assert date(2018, 9, 2) in days_for("Heinz Field")
    assert date(2024, 9, 2) in days_for("Acrisure Stadium"), (
        "the alias' game day must exist via the coordinate fan-out")
    assert days_for("Heinz Field") == days_for("Acrisure Stadium"), (
        "both names must see EXACTLY the days the site was fetched for")
    sample = date(2018, 9, 2)
    assert out[("Heinz Field", sample)] == out[("Acrisure Stadium", sample)]
    assert out[("Heinz Field", sample)] is not out[("Acrisure Stadium", sample)], (
        "aliased keys must never share one mutable series object")
    assert any(f"= {len(requests_sent)} request batches" in m
               for m in caplog.messages), (
        "the logged batch total must count the deduplicated requests: "
        f"{caplog.messages}")


def test_gameless_window_skips_are_reported_with_mlb_wording(
        monkeypatch, caplog, sleeps):
    """MLB's sweep logs what it did NOT request — 'Weather batches: skipped
    N gameless window(s) (span) — no Open-Meteo requests sent for them'.
    The NFL log printed only '34 window(s) x 4 batch(es)', leaving the ~28
    off-season spans invisible and the count looking arbitrary."""
    archive_calls: list[tuple[date, date]] = []

    def fake_batch_range(locations, start_date, end_date, *, source, **kw):
        if source == weather.OPEN_METEO_ARCHIVE:
            archive_calls.append((start_date, end_date))
        return {}

    monkeypatch.setattr(weather, "_fetch_batch_range", fake_batch_range)
    now = datetime(2026, 10, 13, 6, 0, tzinfo=timezone.utc)
    targets = [
        _site_target("A", "2026-02-01T17:00:00Z", "Test Stadium",
                     39.0, -95.0),
        _site_target("B", "2026-09-20T17:00:00Z", "Test Stadium",
                     39.0, -95.0),
    ]

    with caplog.at_level(logging.INFO):
        weather._fetch_batched_weather(targets, now)

    skips = [m for m in caplog.messages if "gameless window(s)" in m]
    assert len(skips) == 1, caplog.messages
    assert "skipped 2 gameless window(s)" in skips[0], skips[0]
    assert ("gameless window(s) (2026-01-31→2026-09-20 span) — "
            "no Open-Meteo requests sent for them") in skips[0], skips[0]
    assert len(archive_calls) == 2, (
        "only the two windows that hold games may be requested: "
        f"{archive_calls}")


def test_a_starved_season_is_named_even_under_a_healthy_aggregate(
        monkeypatch, caplog):
    """MLB prints 'Weather coverage <year>: only N/M games observed — that
    season's weather features are starving' per calendar year at >= 40
    games, because a healthy aggregate must not dilute one starved year.
    NFL had only the aggregate gates: a fully-failed 2017 inside 83%
    overall would surface only as the gap note's first five IDs."""
    def fake_fetch(targets, now, **kw):
        out = {}
        for t in targets:
            if not t["game_id"].startswith("G26"):
                continue
            day = t["kickoff_utc"].date()
            out[(t["stadium"], day)] = {
                "time": [pd.Timestamp(f"{day} 16:00:00Z")],
                "temperature_2m": [70.0], "wind_speed_10m": [5.0],
                "precipitation": [0.0], "snowfall": [0.0],
                "_source": weather.OPEN_METEO_ARCHIVE,
                "_fetched_at_utc": pd.Timestamp("2026-10-13T06:00:00Z"),
            }
        return out

    monkeypatch.setattr(weather, "_fetch_batched_weather", fake_fetch)
    rows = []
    for i in range(8):          # season 2017: every game missing
        rows.append({"game_id": f"G17_{i}", "season": 2017, "week": 1,
                     "gameday": "2017-09-10", "gametime": "13:00",
                     "home_team": "A", "away_team": "B",
                     "home_score": 20.0, "away_score": 10.0,
                     "game_type": "REG", "roof": "outdoors", "div_game": 0,
                     "stadium": "MetLife Stadium", "surface": "grass"})
    for i in range(40):         # season 2026: every game observed
        rows.append({"game_id": f"G26_{i}", "season": 2026, "week": 1,
                     "gameday": "2026-09-10", "gametime": "13:00",
                     "home_team": "A", "away_team": "B",
                     "home_score": 20.0, "away_score": 10.0,
                     "game_type": "REG", "roof": "outdoors", "div_game": 0,
                     "stadium": "Gillette Stadium", "surface": "grass"})

    with caplog.at_level(logging.WARNING), \
            tempfile.TemporaryDirectory() as tmp, \
            warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        got = weather.fetch_games_weather(
            pd.DataFrame(rows), path=Path(tmp) / "w.parquet",
            now=pd.Timestamp("2026-10-13T06:00:00Z"))

    assert len(got) == 40, f"only the observed games ship: {len(got)}"
    starving = [m for m in caplog.messages if "starving" in m]
    assert any("PIT weather coverage 2017: only 0/8 games observed" in m
               for m in starving), caplog.messages
    assert not any("2026" in m for m in starving), (
        f"the healthy season must stay unflagged: {starving}")
    assert any("8/48 eligible games lack" in m for m in caplog.messages), (
        "the any-gap note still discloses the hole: "
        f"{caplog.messages}")
    assert not any("coverage only" in m for m in caplog.messages), (
        "83% is above the 80% catastrophic gate — the YEAR gate is the "
        f"one that must report this hole: {caplog.messages}")
