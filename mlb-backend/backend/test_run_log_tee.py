"""Pin the run-log tee + retention protection (2026-10-01).

The Kaggle run's stdout evaporates with the VM; the tee captures it into
one rolling master file (data_delivery/mlb_pipeline_run_log.txt) that
Phase 5 pushes and Phase 6 protects, so the latest run is reviewable from
a plain git pull. Each test pins one contract:

* the tee duplicates console output into the file and keeps the console
  return contract (bytes-written passthrough);
* installation overwrites the previous run's file (one rolling master);
* install is idempotent (no tee-over-tee chains) and degrades to
  console-only when the destination is unwritable;
* Phase 6 retention classifies the log as protected even on a run that
  did not stage it (the dateless-name eviction trap), and stays stale-
  classification clean for genuinely dated artifacts.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import run_log_tee  # noqa: E402
from retention_policy import classify_artifact  # noqa: E402


# ── tee mechanics ───────────────────────────────────────────────────────────

def test_tee_duplicates_into_file_and_returns_console_count(tmp_path, capsys):
    log_file = open(tmp_path / "log.txt", "w")
    console = io.StringIO()
    tee = run_log_tee._Tee(console, log_file)
    n = tee.write("hello tee\n")
    tee.flush()
    tee.close()
    assert console.getvalue() == "hello tee\n"
    assert (tmp_path / "log.txt").read_text() == "hello tee\n"
    # passthrough contract: console's return value survives
    assert n == len("hello tee\n")


def test_install_overwrites_previous_run(tmp_path, monkeypatch):
    log_path = tmp_path / run_log_tee.RUN_LOG_NAME
    log_path.write_text("PREVIOUS RUN CONTENT\n")
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    got = run_log_tee.install_run_log_tee(tmp_path)
    assert got == log_path
    print("NEW RUN line one")
    sys.stderr.write("stderr line\n")
    sys.stdout.flush()
    text = log_path.read_text()
    assert "NEW RUN line one" in text
    assert "stderr line" in text
    assert "PREVIOUS RUN CONTENT" not in text  # overwritten, not appended


def test_install_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    first = run_log_tee.install_run_log_tee(tmp_path)
    assert first is not None
    assert run_log_tee.install_run_log_tee(tmp_path) is None  # no re-tee


def test_install_degrades_to_console_on_unwritable_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    blocked = tmp_path / "not_a_dir"
    blocked.write_text("this is a file, mkdir will fail")
    assert run_log_tee.install_run_log_tee(blocked) is None
    print("console still works")  # must not raise


# ── Phase 6 retention contract ──────────────────────────────────────────────

def test_run_log_is_protected_from_stale_eviction():
    # A run that did NOT stage the log (not in seen) must still keep it:
    # the dateless name is registered as a protected master.
    verdict = classify_artifact(
        "mlb-backend/data_delivery/mlb_pipeline_run_log.txt",
        seen=set(), retention_dates=set(), recent_dates=set(),
        board_dates=set())
    assert verdict == "protected"


def test_dated_artifacts_outside_windows_still_classify_stale():
    # Guard the guard: the exemption is name-specific, not a blanket
    # dateless-file amnesty.
    verdict = classify_artifact(
        "mlb-backend/data_delivery/todays_games_20200101.csv",
        seen=set(), retention_dates=set(), recent_dates=set(),
        board_dates=set())
    assert verdict == "stale"


# ── crash delivery (2026-10-01 nightly failure: the log must outlive the VM) ─

def test_push_log_on_crash_noop_without_token_or_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert run_log_tee.push_log_on_crash(tmp_path / "x.txt", "u", "r") is False
    monkeypatch.setenv("GITHUB_TOKEN", "dummy")
    assert run_log_tee.push_log_on_crash(
        tmp_path / "missing.txt", "u", "r") is False


def test_crash_hook_runs_previous_hook_first_and_skips_systemexit(
        tmp_path, monkeypatch):
    log = tmp_path / run_log_tee.RUN_LOG_NAME
    log.write_text("traceback would be here")
    calls: list = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: calls.append(a))
    monkeypatch.setattr(run_log_tee, "push_log_on_crash",
                        lambda *a, **k: calls.append("push") or True)
    run_log_tee.install_crash_log_pusher(log, "u", "r")
    err = ValueError("boom")
    sys.excepthook(type(err), err, None)
    assert calls[0] is not "push" and calls[0][1] is err  # original hook ran
    assert calls[1] == "push"                              # then delivery
    # SystemExit = deliberate shutdown, not a crash: no push
    calls.clear()
    sys.excepthook(SystemExit, SystemExit(1), None)
    assert "push" not in calls


def test_crash_hook_never_masks_the_original_error(tmp_path, monkeypatch):
    log = tmp_path / run_log_tee.RUN_LOG_NAME
    log.write_text("x")
    monkeypatch.setattr(sys, "excepthook", lambda *a: None)
    monkeypatch.setattr(run_log_tee, "push_log_on_crash",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("network down")))
    run_log_tee.install_crash_log_pusher(log, "u", "r")
    err = ValueError("original")
    sys.excepthook(type(err), err, None)  # must NOT raise despite push failure
