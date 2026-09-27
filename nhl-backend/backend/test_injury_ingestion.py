"""Offline regression tests for timestamped ESPN injury snapshots."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import ingestion as ing  # noqa: E402
import features as feat  # noqa: E402
import injury_stints as ist  # noqa: E402


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def _payload(*, status="Out", include_player=True):
    records = []
    if include_player:
        records.append({
            "id": "espn-1",
            "athlete": {"displayName": "Alex Example"},
            "status": status,
            "date": "2026-09-20T12:00Z",
            "type": {"abbreviation": "INJ"},
            "details": {"returnDate": "2026-09-22"},
            "shortComment": "lower body",
        })
    return {"injuries": [{"team": {"abbreviation": "BOS"},
                           "injuries": records}]}


def _set_clock(monkeypatch, stamps):
    values = iter(pd.to_datetime(value, utc=True) for value in stamps)
    monkeypatch.setattr(ing, "_utc_now", lambda: next(values))


def test_schedule_parser_retains_exact_utc_puck_drop_for_snapshot_filtering():
    row = ing._parse_score_game({
        "id": 123, "season": 20262027, "gameDate": "2026-10-15",
        "startTimeUTC": "2026-10-15T23:00:00Z",
        "homeTeam": {"abbrev": "BOS"}, "awayTeam": {"abbrev": "TOR"},
    })
    assert row["start_time_utc"] == "2026-10-15T23:00:00Z"
    assert "start_time_utc" in ing.SCORE_KEEP


def test_empty_successful_report_is_saved_as_an_explicit_snapshot_marker(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    monkeypatch.setattr(ing, "_utc_now", lambda: pd.Timestamp("2026-09-26T21:00:00Z"))
    with patch("requests.get", return_value=_Response(_payload(include_player=False))) as get:
        out = ing.load_espn_injuries(use_cache=True, snapshot=True)

    assert get.call_count == 1
    assert len(out) == 1
    assert out.iloc[0]["snapshot_marker"]
    assert pd.Timestamp(out.iloc[0]["snapshot_at"]) == pd.Timestamp("2026-09-26T21:00:00Z")
    latest = json.loads((tmp_path / f"espn_injuries_{ing.INJURY_VERSION}_latest.json").read_text())
    assert latest["snapshot_at"] == "2026-09-26T21:00:00+00:00"
    history = pd.read_parquet(tmp_path / f"espn_injuries_{ing.INJURY_VERSION}_history.parquet")
    assert len(history) == 1 and bool(history.iloc[0]["snapshot_marker"])


def test_capture_timestamp_is_local_observation_time_and_preserved_in_history(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    _set_clock(monkeypatch, ["2026-09-26T10:00:00Z", "2026-09-26T10:05:12Z"])
    with patch("requests.get", return_value=_Response(_payload())):
        out = ing.load_espn_injuries(use_cache=True, snapshot=True)

    assert pd.Timestamp(out.iloc[0]["snapshot_at"]) == pd.Timestamp("2026-09-26T10:05:12Z")
    assert out.iloc[0]["status"] == "Out"
    assert not bool(out.iloc[0]["snapshot_marker"])
    history = pd.read_parquet(tmp_path / f"espn_injuries_{ing.INJURY_VERSION}_history.parquet")
    assert pd.Timestamp(history.iloc[0]["snapshot_at"]) == pd.Timestamp("2026-09-26T10:05:12Z")


def test_cache_hit_same_day_does_not_refetch_or_restamp(tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    capture = pd.Timestamp("2026-09-26T10:05:12Z")
    _set_clock(monkeypatch, ["2026-09-26T11:00:00Z"])
    (tmp_path / f"espn_injuries_{ing.INJURY_VERSION}_latest.json").write_text(
        json.dumps({"snapshot_payload": _payload(), "snapshot_at": capture.isoformat()}))

    with patch("requests.get") as get:
        out = ing.load_espn_injuries(use_cache=True, snapshot=True)

    get.assert_not_called()
    assert pd.Timestamp(out.iloc[0]["snapshot_at"]) == capture
    assert len(pd.read_parquet(
        tmp_path / f"espn_injuries_{ing.INJURY_VERSION}_history.parquet")) == 1


def test_stale_same_day_cache_is_refetched_after_the_short_ttl(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    _set_clock(monkeypatch, ["2026-09-26T10:00:00Z", "2026-09-26T10:01:00Z"])
    (tmp_path / f"espn_injuries_{ing.INJURY_VERSION}_latest.json").write_text(
        json.dumps({
            "snapshot_payload": _payload(status="Out"),
            "snapshot_at": "2026-09-26T00:00:00+00:00",
        }))
    with patch("requests.get", return_value=_Response(_payload(status="IR"))) as get:
        out = ing.load_espn_injuries(use_cache=True, snapshot=True)

    get.assert_called_once()
    assert out.iloc[0]["status"] == "IR"
    assert pd.Timestamp(out.iloc[0]["snapshot_at"]) == pd.Timestamp("2026-09-26T10:01:00Z")


def test_stale_legacy_cache_is_refetched_not_given_a_fresh_timestamp(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    _set_clock(monkeypatch, ["2026-09-26T10:00:00Z", "2026-09-26T10:01:00Z"])
    # v1-era raw provider JSON has neither the version-2 envelope nor capture time.
    (tmp_path / f"espn_injuries_{ing.INJURY_VERSION}_latest.json").write_text(
        json.dumps(_payload()))
    with patch("requests.get", return_value=_Response(_payload(status="IR"))) as get:
        out = ing.load_espn_injuries(use_cache=True, snapshot=True)

    assert get.call_count == 1
    assert out.iloc[0]["status"] == "IR"
    assert pd.Timestamp(out.iloc[0]["snapshot_at"]) == pd.Timestamp("2026-09-26T10:01:00Z")


def test_fetch_failure_returns_old_archive_without_stamping_it_fresh(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    old = pd.DataFrame([{
        "player_id": "mp-1", "player_name": "Alex Example", "status": "Out",
        "report_date": "2026-09-01", "snapshot_at": pd.Timestamp("2026-09-01T12:00:00Z"),
        "snapshot_marker": False,
    }])
    old.to_parquet(tmp_path / f"espn_injuries_{ing.INJURY_VERSION}_history.parquet")
    _set_clock(monkeypatch, ["2026-09-26T10:00:00Z"])
    with patch("requests.get", side_effect=RuntimeError("offline")) as get:
        out = ing.load_espn_injuries(use_cache=True, snapshot=True)

    get.assert_called_once()
    assert len(out) == 1
    assert pd.Timestamp(out.iloc[0]["snapshot_at"]) == pd.Timestamp("2026-09-01T12:00:00Z")


def test_invalid_empty_object_is_fetch_failure_not_an_all_clear_report(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    _set_clock(monkeypatch, ["2026-09-26T10:00:00Z"])
    with patch("requests.get", return_value=_Response({})):
        out = ing.load_espn_injuries(use_cache=False, snapshot=True)

    assert out is None
    assert not list(tmp_path.glob("*history.parquet"))


def test_empty_successful_snapshot_is_accepted_as_snapshot_provenance():
    reports = pd.DataFrame([{
        "player_id": None, "status": None,
        "snapshot_at": "2026-09-26T10:00:00Z", "snapshot_marker": True,
    }])
    stints, audit = ist.build_stint_intervals(reports)
    stints.attrs["window_end"] = pd.Timestamp("2026-09-26T10:00:00Z")
    stints.attrs["snapshot_times"] = [pd.Timestamp("2026-09-26T10:00:00Z")]
    ratings = pd.DataFrame({"player_id": ["1"]})
    verdicts = ist.assert_pit(
        ratings, stints, decided_max_date="2026-09-26",
        window_end=stints.attrs["window_end"], snapshot_based=audit["snapshot_based"])
    assert len(stints) == 0
    assert verdicts["snapshot_based"]["ok"]


def test_history_deduplicates_only_exact_capture_times(tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    path = tmp_path / f"espn_injuries_{ing.INJURY_VERSION}_history.parquet"
    first = pd.DataFrame([{
        "player_id": "1", "player_name": "Alex", "status": "Out",
        "snapshot_at": pd.Timestamp("2026-09-26T10:00:00Z"),
        "snapshot_marker": False,
    }])
    second = pd.DataFrame([{
        "player_id": "1", "player_name": "Alex", "status": "IR",
        "snapshot_at": pd.Timestamp("2026-09-26T10:00:00Z"),
        "snapshot_marker": False,
    }])
    third = pd.DataFrame([{
        "player_id": "1", "player_name": "Alex", "status": "IR",
        "snapshot_at": pd.Timestamp("2026-09-26T10:00:01Z"),
        "snapshot_marker": False,
    }])

    ing._append_injury_snapshot(first)
    same_time = ing._append_injury_snapshot(second)
    two_times = ing._append_injury_snapshot(third)

    assert len(same_time) == 1 and same_time.iloc[0]["status"] == "IR"
    assert len(two_times) == 2
    assert pd.to_datetime(two_times["snapshot_at"], utc=True).nunique() == 2
