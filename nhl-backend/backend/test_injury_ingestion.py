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
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
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


def test_transient_edge_block_is_retried_before_history_falls_back(
        tmp_path, monkeypatch):
    """The ESPN edge blocks agents in passing waves (2026-09-28 Kaggle run:
    one 403 with a profile that tested 200 minutes before and after, costing
    BOTH snapshots of the day because a single failed request fell straight
    to previously-captured history). A 403 must be retried and only a
    persistent block falls back to history.
    """
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
    _set_clock(monkeypatch, ["2026-09-26T10:00:00Z", "2026-09-26T10:05:12Z"])
    responses = [_Response(_payload(), status_code=403),
                 _Response(_payload(), status_code=403),
                 _Response(_payload())]
    with patch("requests.get", side_effect=responses) as get:
        out = ing.load_espn_injuries(use_cache=True, snapshot=True)

    assert get.call_count == 3  # two transient blocks, then success
    assert out.iloc[0]["status"] == "Out"
    assert not bool(out.iloc[0]["snapshot_marker"])
    latest = json.loads(
        (tmp_path / f"espn_injuries_{ing.INJURY_VERSION}_latest.json").read_text())
    assert latest["snapshot_at"] == "2026-09-26T10:05:12+00:00"


def test_identity_fallback_rescues_the_snapshot(tmp_path, monkeypatch):
    """The wave can outlast any retry schedule (2026-09-29 15:01: both runs
    lost the ESPN fetch for ~9 minutes — every attempt of all 4 retries 403'd
    in BOTH phases while the same identity tested 200 minutes later). When
    the browser agent's retries are exhausted, ONE last attempt rides the
    library-default identity (no User-Agent header at all — requests drops a
    None value; verified 200 on 2026-09-29 while the browser agent was
    mid-wave). A second, structurally different profile doubles the chance
    the day's snapshot is captured at all."""
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    _set_clock(monkeypatch, ["2026-09-29T15:01:00Z", "2026-09-29T15:02:00Z"])
    monkeypatch.setattr(ing.time, "sleep", lambda _s: None)

    seen_ua = []

    def _get(url, timeout=None, headers=None, **_):
        seen_ua.append((headers or {}).get("User-Agent"))
        if (headers or {}).get("User-Agent") == "Mozilla/5.0":
            return _Response(_payload(), status_code=403)
        return _Response(_payload())  # the library-default identity passes

    with patch("requests.get", side_effect=_get) as get:
        out = ing.load_espn_injuries(use_cache=True, snapshot=True)

    assert get.call_count == 5  # 4 blocked browser-agent attempts + 1 rescue
    assert seen_ua[:4] == ["Mozilla/5.0"] * 4
    assert seen_ua[4] is None, (
        "the fallback must ride the library-default identity: a None "
        "User-Agent is dropped by requests, not sent as the string 'None'")
    assert out.iloc[0]["status"] == "Out"
    latest = json.loads(
        (tmp_path / f"espn_injuries_{ing.INJURY_VERSION}_latest.json").read_text())
    assert latest["snapshot_at"] == "2026-09-29T15:02:00+00:00"


def test_persistent_wave_on_both_identities_still_replays_history(
        tmp_path, monkeypatch):
    """When BOTH identities are blocked for the whole run, the fallback must
    stay bounded and history takes over — the extra identity attempt may not
    turn a transient edge block into an unbounded hang."""
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    monkeypatch.setattr(ing.config, "DATA_DELIVERY_DIR", tmp_path)
    artifact = pd.DataFrame([{
        "player_id": "mp-9", "player_name": "Repo Example", "status": "Out",
        "report_date": "2026-09-20",
        "snapshot_at": pd.Timestamp("2026-09-20T09:00:00Z"),
        "snapshot_marker": False,
    }])
    artifact.to_parquet(tmp_path / ing.INJURY_HISTORY_ARTIFACT)
    _set_clock(monkeypatch, ["2026-09-29T15:01:00Z"])
    monkeypatch.setattr(ing.time, "sleep", lambda _s: None)

    seen_ua = []

    def _get(url, timeout=None, headers=None, **_):
        seen_ua.append((headers or {}).get("User-Agent"))
        return _Response(_payload(), status_code=403)

    with patch("requests.get", side_effect=_get) as get:
        out = ing.load_espn_injuries(use_cache=True, snapshot=True)

    # 4 browser-agent retries + 2 library-default retries, all blocked.
    assert get.call_count == 6
    assert seen_ua[:4] == ["Mozilla/5.0"] * 4
    assert seen_ua[4:] == [None, None]
    assert len(out) == 1
    assert out.iloc[0]["player_id"] == "mp-9"


def test_history_falls_back_to_repo_artifact_when_local_cache_is_absent(
        tmp_path, monkeypatch):
    """A sandbox whose egress to the injury endpoint is blocked (2026-09-28
    Kaggle runs: every retry 403'd while the same profile probed 200 from
    another network) must still replay captured history — the pipeline
    persists it into data_delivery so health is a data artifact, not a
    cache-local one. Local snapshot cache first, artifact second.
    """
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    artifact = pd.DataFrame([{
        "player_id": "mp-9", "player_name": "Repo Example", "status": "Out",
        "report_date": "2026-09-20", "snapshot_at": pd.Timestamp("2026-09-20T09:00:00Z"),
        "snapshot_marker": False,
    }])
    # Patch the config module ingestion ACTUALLY holds (under pytest the
    # backend package import makes it backend.config; the test file's own
    # `import config` would be a different module object).
    monkeypatch.setattr(ing.config, "DATA_DELIVERY_DIR", tmp_path)
    artifact.to_parquet(tmp_path / ing.INJURY_HISTORY_ARTIFACT)

    out = ing._injury_history()

    assert len(out) == 1
    assert out.iloc[0]["player_id"] == "mp-9"
    # And the fetch-failure fallback now returns that history, not None.
    _set_clock(monkeypatch, ["2026-09-26T10:00:00Z"])
    monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
    with patch("requests.get",
               return_value=_Response(_payload(), status_code=403)) as get:
        out = ing.load_espn_injuries(use_cache=True, snapshot=True)
    # 4 browser-agent retries + 2 library-default identity retries, all
    # blocked, then history.
    assert get.call_count == 6
    assert len(out) == 1
    assert out.iloc[0]["player_id"] == "mp-9"
    assert pd.Timestamp(out.iloc[0]["snapshot_at"]) == pd.Timestamp("2026-09-20T09:00:00Z")


def test_history_tiers_union_and_dedupe_on_snapshot_identity(
        tmp_path, monkeypatch):
    """Local cache rows and artifact rows are unioned on (snapshot_at,
    player_id): a sandbox that CAN refresh keeps its new rows alongside the
    artifact's older ones; overlapping rows never double-count a player-day.
    """
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    monkeypatch.setattr(ing.config, "DATA_DELIVERY_DIR", tmp_path)
    stamp = pd.Timestamp("2026-09-20T09:00:00Z")
    shared = {"player_id": "mp-1", "player_name": "Alex Example",
              "status": "Out", "report_date": "2026-09-20",
              "snapshot_at": stamp, "snapshot_marker": False}
    pd.DataFrame([shared]).to_parquet(
        tmp_path / f"espn_injuries_{ing.INJURY_VERSION}_history.parquet")
    pd.DataFrame([shared, {**shared, "player_id": "mp-2",
                           "player_name": "Repo Example"}]).to_parquet(
        tmp_path / ing.INJURY_HISTORY_ARTIFACT)

    out = ing._injury_history()

    assert sorted(out["player_id"]) == ["mp-1", "mp-2"]


def test_export_injury_history_artifact_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    monkeypatch.setattr(ing.config, "DATA_DELIVERY_DIR", tmp_path)
    pd.DataFrame([{
        "player_id": "mp-1", "player_name": "Alex Example", "status": "Out",
        "report_date": "2026-09-20", "snapshot_at": pd.Timestamp("2026-09-20T09:00:00Z"),
        "snapshot_marker": False,
    }]).to_parquet(
        tmp_path / f"espn_injuries_{ing.INJURY_VERSION}_history.parquet")

    name = ing.export_injury_history_artifact()

    assert name == ing.INJURY_HISTORY_ARTIFACT
    restored = pd.read_parquet(tmp_path / ing.INJURY_HISTORY_ARTIFACT)
    assert len(restored) == 1
    assert restored.iloc[0]["player_id"] == "mp-1"


def test_export_grows_the_repo_artifact_never_shrinks_it(tmp_path, monkeypatch):
    """A machine whose capture starts LATER than the artifact's (exactly the
    2026-09-29 Kaggle log: the repo artifact never landed, and a bare
    local-cache overwrite would have let the first exporter erase any older
    carried snapshots) must union with the carried rows, not replace them.
    """
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    monkeypatch.setattr(ing.config, "DATA_DELIVERY_DIR", tmp_path)
    older = pd.DataFrame([{
        "player_id": "mp-old", "player_name": "Carried Example", "status": "IR",
        "report_date": "2026-09-01", "snapshot_at": pd.Timestamp("2026-09-01T12:00:00Z"),
        "snapshot_marker": False,
    }])
    older.to_parquet(tmp_path / ing.INJURY_HISTORY_ARTIFACT)
    pd.DataFrame([{
        "player_id": "mp-new", "player_name": "Local Example", "status": "Out",
        "report_date": "2026-09-25", "snapshot_at": pd.Timestamp("2026-09-25T09:00:00Z"),
        "snapshot_marker": False,
    }]).to_parquet(
        tmp_path / f"espn_injuries_{ing.INJURY_VERSION}_history.parquet")

    ing.export_injury_history_artifact()

    restored = pd.read_parquet(tmp_path / ing.INJURY_HISTORY_ARTIFACT)
    assert sorted(restored["player_id"]) == ["mp-new", "mp-old"]


def test_export_logs_the_snapshot_window(tmp_path, monkeypatch, caplog):
    """The 2026-09-29 log's bare '%d rows' export line hid the archive's real
    coverage; the export must report how many snapshots it carries and the
    window they span, so a one-snapshot artifact is visible in the log.
    """
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    monkeypatch.setattr(ing.config, "DATA_DELIVERY_DIR", tmp_path)
    pd.DataFrame([
        {"player_id": "1", "snapshot_at": pd.Timestamp("2026-09-26T10:00:00Z"),
         "snapshot_marker": False},
        {"player_id": "2", "snapshot_at": pd.Timestamp("2026-09-26T10:00:00Z"),
         "snapshot_marker": False},
        {"player_id": "1", "snapshot_at": pd.Timestamp("2026-09-27T10:00:00Z"),
         "snapshot_marker": False},
    ]).to_parquet(
        tmp_path / f"espn_injuries_{ing.INJURY_VERSION}_history.parquet")

    with caplog.at_level("INFO", logger=ing.__name__):
        ing.export_injury_history_artifact()

    assert any("2 captured snapshot" in r.message for r in caplog.records)
    assert any("2026-09-26" in r.message and "2026-09-27" in r.message
               for r in caplog.records)


def test_export_injury_history_artifact_without_history_writes_nothing(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    monkeypatch.setattr(ing.config, "DATA_DELIVERY_DIR", tmp_path)

    assert ing.export_injury_history_artifact() is None
    assert not (tmp_path / ing.INJURY_HISTORY_ARTIFACT).exists()


def test_persistent_edge_block_still_falls_back_to_captured_history(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    # Isolate the artifact tier too: the repo may genuinely carry captured
    # history now, and this test pins the LOCAL-cache tier in isolation.
    monkeypatch.setattr(ing.config, "DATA_DELIVERY_DIR", tmp_path)
    monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
    old = pd.DataFrame([{
        "player_id": "mp-1", "player_name": "Alex Example", "status": "Out",
        "report_date": "2026-09-01", "snapshot_at": pd.Timestamp("2026-09-01T12:00:00Z"),
        "snapshot_marker": False,
    }])
    old.to_parquet(tmp_path / f"espn_injuries_{ing.INJURY_VERSION}_history.parquet")
    _set_clock(monkeypatch, ["2026-09-26T10:00:00Z"])
    with patch("requests.get",
               return_value=_Response(_payload(), status_code=403)) as get:
        out = ing.load_espn_injuries(use_cache=True, snapshot=True)

    # 4 browser-agent retries + 2 library-default identity retries, then
    # the history fallback — still bounded.
    assert get.call_count == 6
    assert len(out) == 1
    # The fallback is the OLD captured state, never re-stamped as fresh.
    assert pd.Timestamp(out.iloc[0]["snapshot_at"]) == pd.Timestamp("2026-09-01T12:00:00Z")


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
    # Isolate the artifact tier: this test pins the LOCAL-cache fallback.
    monkeypatch.setattr(ing.config, "DATA_DELIVERY_DIR", tmp_path)
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    old = pd.DataFrame([{
        "player_id": "mp-1", "player_name": "Alex Example", "status": "Out",
        "report_date": "2026-09-01", "snapshot_at": pd.Timestamp("2026-09-01T12:00:00Z"),
        "snapshot_marker": False,
    }])
    old.to_parquet(tmp_path / f"espn_injuries_{ing.INJURY_VERSION}_history.parquet")
    _set_clock(monkeypatch, ["2026-09-26T10:00:00Z"])
    monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
    with patch("requests.get", side_effect=RuntimeError("offline")) as get:
        out = ing.load_espn_injuries(use_cache=True, snapshot=True)

    # Hard network failures are also retried (4 browser-agent + 2
    # library-default identity attempts, bounded) before the run settles
    # for previously captured history.
    assert get.call_count == 6
    assert len(out) == 1
    assert pd.Timestamp(out.iloc[0]["snapshot_at"]) == pd.Timestamp("2026-09-01T12:00:00Z")


def test_invalid_empty_object_is_fetch_failure_not_an_all_clear_report(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    # Isolate the artifact tier: with no captured history anywhere, the
    # failure path must return None — never a fabricated all-clear.
    monkeypatch.setattr(ing.config, "DATA_DELIVERY_DIR", tmp_path)
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
