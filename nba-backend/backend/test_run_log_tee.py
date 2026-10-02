"""Pin the run-log tee + retention protection (2026-10-01).

The Kaggle run's stdout evaporates with the VM; the tee captures it into
one rolling master file (data_delivery/nba_pipeline_run_log.txt) that
the publish phase ships and retention_policy protects, so the latest run
is reviewable from a plain git pull. Each test pins one contract:

* the tee duplicates console output into the file and keeps the console
  return contract (bytes-written passthrough);
* tty-ness and carriage returns pass through to the CONSOLE untouched
  (tqdm's graphical black bar), while the FILE stays line-oriented;
* logging handlers are re-pointed at the tee, so INFO/WARNING records
  reach the log (basicConfig captures the pre-tee stream otherwise);
* installation overwrites the previous run's file (one rolling master);
* install is idempotent (no tee-over-tee chains) and degrades to
  console-only when the destination is unwritable;
* retention classifies the log as protected even on a run that did not
  produce it (the dateless-name eviction trap), and stays stale-
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


# ── tee mechanics ───────────────────────────────────────────────────

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
    text = log_path.read_text(encoding="utf-8")
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


# ── tty passthrough + \r contract (tqdm graphical-bar regression) ──

def test_tee_isatty_delegates_to_the_console_stream(tmp_path):
    console = io.StringIO()
    tee = run_log_tee._Tee(console, open(tmp_path / "a.txt", "w"))
    assert tee.isatty() == console.isatty()  # False for a StringIO

    class _FakeTTY:
        def write(self, s):
            return len(s)

        def flush(self):
            pass

        def isatty(self):
            return True

    tty_tee = run_log_tee._Tee(_FakeTTY(), open(tmp_path / "b.txt", "w"))
    assert tty_tee.isatty() is True  # tqdm must see what it saw pre-tee


def test_tee_keeps_raw_carriage_returns_on_console_only(tmp_path):
    console = io.StringIO()
    log_file = open(tmp_path / "log.txt", "w")
    tee = run_log_tee._Tee(console, log_file)
    tee.write(" 50%|#####     | 3/6 [..]\r 83%|########  | 5/6 [..]\r100%| done\n")
    tee.flush()
    tee.close()
    assert "\r" in console.getvalue()  # console keeps tqdm's raw frames
    text = (tmp_path / "log.txt").read_text()
    assert "\r" not in text  # the file stays line-oriented
    assert " 83%|########  | 5/6 [..]" in text  # every frame is its own line


# ── logging rebind (pushed log was prints-only without this) ────────

def test_install_rebinds_logging_handlers_so_records_reach_the_log(
        tmp_path, monkeypatch):
    import logging

    # Model production: basicConfig's handler holds the SAME pre-tee
    # stream object the tee is about to wrap.
    pre_tee_stream = io.StringIO()
    handler = logging.StreamHandler(pre_tee_stream)
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    old_level = root.level
    root.setLevel(logging.INFO)  # production runs basicConfig(level=INFO)
    root.handlers[:] = [handler]
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", pre_tee_stream)
    try:
        log_path = run_log_tee.install_run_log_tee(tmp_path)
        assert log_path is not None
        logging.getLogger().info("fold line the tee must capture")
        logging.getLogger().warning("gate WARNING the tee must capture")
        for h in root.handlers:
            if hasattr(h, "flush"):
                h.flush()
        text = log_path.read_text(encoding="utf-8")
        assert "fold line the tee must capture" in text
        assert "gate WARNING the tee must capture" in text
        # re-pointed in place, not replaced: the console copy survives
        assert isinstance(handler.stream, run_log_tee._Tee)
        assert "fold line the tee must capture" in pre_tee_stream.getvalue()
    finally:
        root.handlers[:] = old_handlers
        root.setLevel(old_level)


def test_install_leaves_harness_owned_handlers_alone(tmp_path, monkeypatch):
    import logging

    # Handlers owned by an outer harness (pytest's log capture, notebook
    # kernels) hold their own stream objects and must NOT be re-pointed:
    # the harness reads them back by identity/attribute after the run.
    foreign = logging.StreamHandler(io.StringIO())
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    root.handlers[:] = [foreign]
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    try:
        assert run_log_tee.install_run_log_tee(tmp_path) is not None
        assert not isinstance(foreign.stream, run_log_tee._Tee)
    finally:
        root.handlers[:] = old_handlers


# ── retention contract ──────────────────────────────────────────────

def test_run_log_is_protected_from_stale_eviction():
    # A run that did NOT stage the log (not in seen) must still keep it:
    # the dateless name is registered as a protected master.
    verdict = classify_artifact(
        "nba-backend/data_delivery/nba_pipeline_run_log.txt",
        seen=set(), retention_dates=set(), recent_dates=set(),
        board_dates=set())
    assert verdict == "protected"


def test_dated_artifacts_outside_windows_still_classify_stale():
    # Guard the guard: the exemption is name-specific, not a blanket
    # dateless-file amnesty.
    verdict = classify_artifact(
        "nba-backend/data_delivery/nba_moneyline_v1_20200101.csv",
        seen=set(), retention_dates=set(), recent_dates=set(),
        board_dates=set())
    assert verdict == "stale"


# ── crash delivery (the log must outlive the VM) ────────────────────

def test_push_log_on_crash_noop_without_token_or_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert run_log_tee.push_log_on_crash(tmp_path / "x.txt") is False
    monkeypatch.setenv("GITHUB_TOKEN", "dummy")
    assert run_log_tee.push_log_on_crash(tmp_path / "missing.txt") is False


def test_auth_remote_url_rejects_non_https_and_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_REPO_URL", raising=False)
    monkeypatch.setattr(run_log_tee.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": "",
                                                       "returncode": 1})())
    assert run_log_tee._auth_remote_url("tok") == ""  # no origin found
    monkeypatch.setenv("GITHUB_REPO_URL", "git@github.com:u/r.git")
    assert run_log_tee._auth_remote_url("tok") == ""  # ssh: token can't bind
    monkeypatch.setenv("GITHUB_REPO_URL", "https://github.com/u/r.git")
    assert run_log_tee._auth_remote_url("tok") == \
        "https://tok@github.com/u/r.git"


def test_crash_hook_runs_previous_hook_first_and_skips_systemexit(
        tmp_path, monkeypatch):
    log = tmp_path / run_log_tee.RUN_LOG_NAME
    log.write_text("traceback would be here")
    calls: list = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: calls.append(a))
    monkeypatch.setattr(run_log_tee, "push_log_on_crash",
                        lambda *a, **k: calls.append("push") or True)
    run_log_tee.install_crash_log_pusher(log)
    err = ValueError("boom")
    sys.excepthook(type(err), err, None)
    assert calls[0] != "push" and calls[0][1] is err  # original hook ran
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
    run_log_tee.install_crash_log_pusher(log)
    err = ValueError("original")
    sys.excepthook(type(err), err, None)  # must NOT raise despite push failure
