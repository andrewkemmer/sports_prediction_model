"""In-pipeline designation archive refresh (2026-10-24 structural fix).

The training frame must never depend on the static committed
``nba_designations.parquet``: every run repopulates the archive fully under
``NBA_FULL_REPULL=1`` and tops off missing decided days otherwise, before
``load_designations`` reads it. These tests pin that contract with a stubbed
resolver (no network): which days get fetched, what lands in the archive +
delivery copy, and that a resolver failure keeps the old archive instead of
raising into the run.
"""

import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import backfill_injury_designations as bf_mod  # noqa: E402
import master_pipeline as mp  # noqa: E402
import nba_injury_report as ir_mod  # noqa: E402

D1 = date(2026, 10, 20)
D2 = date(2026, 10, 21)
D3 = date(2026, 10, 22)
# D3 is "today": decided days are strictly before it, so D3 itself is
# pending and must never enter the archive (the live slate path owns it).
TODAY = date(2026, 10, 22)


def _frame(rows):
    """Minimal 14-column designation frames, one row per (day, player)."""
    recs = []
    for day, player, status in rows:
        recs.append({
            "gameday": pd.Timestamp(day), "matchup": "AWY@HOME",
            "team": "AWY", "team_full": "Away", "player": player,
            "player_report": player, "status": status, "reason": "",
            "published_at": pd.Timestamp(day), "cutoff_at": pd.Timestamp(day),
            "tipoff_at": pd.Timestamp(day), "provenance": "game_day_submission",
            "status_tipoff": status,
            "published_at_tipoff": pd.Timestamp(day),
        })
    out = pd.DataFrame(recs, columns=ir_mod.DESIGNATION_COLUMNS)
    return out


def _games(days):
    rows = []
    for i, day in enumerate(days):
        rows.append({
            "gameday": pd.Timestamp(day),
            "away_team": "AWY", "home_team": "HOME",
            "start_time_utc": f"{day}T23:00:00Z",
            "game_id": f"g{i}",
        })
    return pd.DataFrame(rows)


def _wire(monkeypatch, tmp_path, calls, by_day):
    """Point the refresh at tmp dirs with a canned resolver."""
    monkeypatch.setattr(bf_mod, "SHARD_DIR", tmp_path / "shards")
    monkeypatch.setattr(bf_mod, "ARCHIVE", tmp_path / "nba_designations.parquet")
    monkeypatch.setattr(bf_mod, "CACHE", tmp_path / "pdfs")
    monkeypatch.setattr(ir_mod, "require_pdf_parser", lambda: "stub")

    def fake(day, cache_dir, games=None, **kwargs):
        calls.append(day)
        return _frame(by_day.get(day, [])).copy()

    monkeypatch.setattr(ir_mod, "game_day_designations", fake)
    monkeypatch.delenv("NBA_FULL_REPULL", raising=False)
    return tmp_path / "delivery"


def test_incremental_topoff_fetches_only_missing_days(monkeypatch, tmp_path):
    calls = []
    delivery = _wire(monkeypatch, tmp_path, calls, {
        D2: [(D2, "Out Player", "Out")],
    })
    # Archive already covers D1; D3 is pending (never archived).
    _frame([(D1, "Old Player", "Out")]).to_parquet(
        bf_mod.ARCHIVE, index=False)
    stats = mp._refresh_designation_archive(
        _games([D1, D2, D3]), delivery_dir=delivery, today=TODAY)
    assert stats["mode"] == "incremental"
    assert calls == [D2]
    assert stats["resolved_days"] == 1
    archive = pd.read_parquet(bf_mod.ARCHIVE)
    assert set(pd.to_datetime(archive.gameday).dt.date) == {D1, D2}
    shipped = pd.read_parquet(delivery / "nba_designations.parquet")
    assert set(pd.to_datetime(shipped.gameday).dt.date) == {D1, D2}
    # The live slate path (not the archive) owns pending D3.
    assert D3 not in set(pd.to_datetime(shipped.gameday).dt.date)


def test_noop_when_archive_covers_everything(monkeypatch, tmp_path):
    calls = []
    delivery = _wire(monkeypatch, tmp_path, calls, {})
    _frame([(D1, "Old Player", "Out")]).to_parquet(
        bf_mod.ARCHIVE, index=False)

    def _boom(*args, **kwargs):
        raise AssertionError("resolver must not be called")

    monkeypatch.setattr(ir_mod, "game_day_designations", _boom)
    stats = mp._refresh_designation_archive(
        _games([D1]), delivery_dir=delivery, today=TODAY)
    assert stats["mode"] == "incremental"
    assert stats["resolved_days"] == 0
    assert calls == []
    assert len(pd.read_parquet(bf_mod.ARCHIVE)) == 1


def test_full_repull_reresolves_covered_days(monkeypatch, tmp_path):
    calls = []
    delivery = _wire(monkeypatch, tmp_path, calls, {
        D1: [(D1, "New Player", "Out")],
        D2: [(D2, "Other Player", "Doubtful")],
    })
    monkeypatch.setenv("NBA_FULL_REPULL", "1")
    _frame([(D1, "Stale Player", "Out")]).to_parquet(
        bf_mod.ARCHIVE, index=False)
    stats = mp._refresh_designation_archive(
        _games([D1, D2, D3]), delivery_dir=delivery, today=TODAY)
    assert stats["mode"] == "full"
    assert sorted(calls) == [D1, D2]
    archive = pd.read_parquet(bf_mod.ARCHIVE)
    players = set(archive.player_report)
    assert players == {"New Player", "Other Player"}
    assert "Stale Player" not in players


def test_resolver_failure_keeps_old_archive(monkeypatch, tmp_path):
    delivery = _wire(monkeypatch, tmp_path, [], {})
    before = _frame([(D1, "Old Player", "Out")])
    before.to_parquet(bf_mod.ARCHIVE, index=False)

    def _boom(*args, **kwargs):
        raise RuntimeError("host refused")

    monkeypatch.setattr(ir_mod, "game_day_designations", _boom)
    stats = mp._refresh_designation_archive(
        _games([D1, D2]), delivery_dir=delivery, today=TODAY)
    assert stats["error"]
    pd.testing.assert_frame_equal(pd.read_parquet(bf_mod.ARCHIVE), before,
                                  check_dtype=False)
    assert not (delivery / "nba_designations.parquet").exists()


def test_workers_resolve_identical_rows(monkeypatch, tmp_path):
    """One worker vs four: the same rows in the same order. The pool only
    overlaps latency; the walks, the request rate and the merge are
    unchanged, so the archive is the serial run's archive."""
    by_day = {D1: [(D1, "Old Player", "Out")],
              D2: [(D2, "Other Player", "Doubtful")]}
    shipped = {}
    for workers in ("1", "4"):
        run = tmp_path / f"w{workers}"
        calls = []
        delivery = _wire(monkeypatch, run, calls, by_day)
        monkeypatch.setenv("NBA_DESIGNATION_WORKERS", workers)
        stats = mp._refresh_designation_archive(
            _games([D1, D2, D3]), delivery_dir=delivery, today=TODAY)
        assert stats["resolved_days"] == 2
        shipped[workers] = pd.read_parquet(
            delivery / "nba_designations.parquet")
    pd.testing.assert_frame_equal(shipped["1"], shipped["4"])


def test_one_bad_day_does_not_stop_the_others(monkeypatch, tmp_path):
    """A day whose resolver raises keeps its prior coverage while the rest
    of the window still resolves - the parallel pool must isolate a failure
    to its own day exactly like the serial loop did."""
    delivery = _wire(monkeypatch, tmp_path, [], {})

    def _flaky(day, cache_dir, games=None, **kwargs):
        if day == D2:
            raise RuntimeError("host refused")
        return _frame([(day, "Good Player", "Out")]).copy()

    monkeypatch.setattr(ir_mod, "game_day_designations", _flaky)
    stats = mp._refresh_designation_archive(
        _games([D1, D2, D3]), delivery_dir=delivery, today=TODAY)
    assert stats["mode"] == "incremental"
    assert stats["resolved_days"] == 1
    archive = pd.read_parquet(bf_mod.ARCHIVE)
    assert set(pd.to_datetime(archive.gameday).dt.date) == {D1}


def test_the_refresh_reports_where_its_wall_clock_went(monkeypatch, tmp_path,
                                                      caplog):
    """A 40-minute phase has to name its own cost in the log it delivers.

    This is a reading, never an input: the same days resolve, in the same
    order, whether or not the phase is timed.  Three figures measure three
    different things and must not be added together - the segments are wall
    clock, the fetch/parse figures are worker-time sums.
    """
    delivery = _wire(monkeypatch, tmp_path, [], {
        D2: [(D2, "Out Player", "Out")],
    })
    with caplog.at_level(logging.INFO):
        stats = mp._refresh_designation_archive(
            _games([D2]), delivery_dir=delivery, today=TODAY)
    timing = stats["timing"]
    assert set(timing) >= {"phase_seconds", "resolve_seconds",
                           "merge_seconds", "write_seconds", "parsed",
                           "parse_seconds", "per_parse_ms", "disk_hits",
                           "fetched", "requests", "network_seconds"}
    assert timing["phase_seconds"] >= 0.0
    assert (timing["resolve_seconds"] + timing["merge_seconds"]
            + timing["write_seconds"] <= timing["phase_seconds"] + 1.0)
    assert "timing:" in caplog.text
    assert "pdfs" in caplog.text


def test_the_resolver_receives_one_shared_parse_cache(monkeypatch, tmp_path):
    """Every day in one refresh resolves through the same run-level filing
    cache, so a filing two days both touch is parsed once."""
    delivery = _wire(monkeypatch, tmp_path, [], {})
    seen = []

    def _spy(day, cache_dir, games=None, **kwargs):
        seen.append(kwargs.get("parsed"))
        return _frame([(day, "P", "Out")]).copy()

    monkeypatch.setattr(ir_mod, "game_day_designations", _spy)
    stats = mp._refresh_designation_archive(
        _games([D1, D2]), delivery_dir=delivery, today=TODAY)
    assert stats["resolved_days"] == 2
    assert len(seen) == 2
    assert seen[0] is not None and seen[0] is seen[1]
