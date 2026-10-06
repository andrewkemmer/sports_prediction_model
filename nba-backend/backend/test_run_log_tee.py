"""Pin the run-log tee + retention protection (2026-10-01).

The Kaggle run's stdout evaporates with the VM; the tee captures it into
one rolling master file (data_delivery/nba_pipeline_run_log.txt) that
the publish phase ships and retention_policy protects, so the latest run
is reviewable from a plain git pull. Each test pins one contract:

* the tee duplicates console output into the file and keeps the console
  return contract (bytes-written passthrough);
* tty-ness and carriage returns pass through to the CONSOLE untouched
  (tqdm's graphical black bar), while the FILE collapses CR runs to each
  bar's final frame — one line per bar, never one line per frame (the
  2026-10-06 review found 4,849 of the 5,654 committed lines were
  superseded tqdm frames);
* logging handlers are re-pointed at the tee, so INFO/WARNING records
  reach the log (basicConfig captures the pre-tee stream otherwise);
* installation overwrites the previous run's file (one rolling master);
* install is idempotent (no tee-over-tee chains) and degrades to
  console-only when the destination is unwritable;
* retention classifies the log as protected even on a run that did not
  produce it (the dateless-name eviction trap), and stays stale-
  classification clean for genuinely dated artifacts.
* the 2026-10-06 run-log review remediation stays pinned: CR-frame
  collapse (file side), the published-blend line, the Platt fallback
  evidence with n=, the final pooled calibrator line, the stale
  NBA_END_DATE guard, and the final run-log delivery.
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
    """Console keeps tqdm's raw \r frames; the FILE collapses the CR run
    to ONE line — the bar's FINAL frame (2026-10-06 log review: the old
    \r→\n conversion turned every superseded frame into its own line, 4,849
    of the 5,654 committed log lines; MLB/NHL/NFL CR-collapse parity)."""
    console = io.StringIO()
    log_file = open(tmp_path / "log.txt", "w")
    tee = run_log_tee._Tee(console, log_file)
    tee.write(" 50%|#####     | 3/6 [..]\r 83%|########  | 5/6 [..]\r100%| done\n")
    tee.flush()
    tee.close()
    assert "\r" in console.getvalue()  # console keeps tqdm's raw frames
    text = (tmp_path / "log.txt").read_text()
    assert "\r" not in text  # the file stays line-oriented
    assert text == "100%| done\n"  # ONE line per bar: the final frame only


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


# ── 2026-10-06 run-log review remediation ───────────────────────────

def test_tee_flush_does_not_flood_the_file_with_superseded_frames(tmp_path):
    """An open CR-run commits NOTHING on flush — tqdm flushes after every
    frame, so committing there would land exactly the superseded frames
    the collapse removes."""
    console = io.StringIO()
    tee = run_log_tee._Tee(console, open(tmp_path / "f.txt", "w"))
    tee.write("\r10%|x| 1/9 [..]")
    tee.write("\r50%|x| 5/9 [..]")
    tee.flush()
    tee.flush()
    assert (tmp_path / "f.txt").read_text() == ""  # nothing landed
    assert "50%|x| 5/9 [..]" in console.getvalue()  # console saw the frames
    tee.write("\r100%| done\n")
    tee.close()
    assert (tmp_path / "f.txt").read_text() == "100%| done\n"


def test_tee_close_commits_a_bar_that_ends_without_a_newline(tmp_path):
    """A run that dies (or ends) on a half-written line still lands it —
    close / atexit commit the pending partial."""
    console = io.StringIO()
    tee = run_log_tee._Tee(console, open(tmp_path / "c.txt", "w"))
    tee.write("\rfinal frame")  # no trailing newline
    tee.close()
    assert (tmp_path / "c.txt").read_text() == "final frame\n"


def test_tee_lands_a_log_record_after_a_frame_on_its_own_line(tmp_path):
    """A logging record arriving while a bar is open must open its own
    line — the frame becomes the bar's final line, the record follows
    intact (the old \r→\n conversion glued 9 tqdm+✅ lines in the
    2026-10-06 committed log)."""
    console = io.StringIO()
    tee = run_log_tee._Tee(console, open(tmp_path / "r.txt", "w"))
    tee.write("\r40%|z| 4/9 [..]")  # tqdm frames LEAD each frame with \r
    tee.write("2026-10-06 04:00:00,000 INFO something fetched=7\n")
    tee.close()
    assert (tmp_path / "r.txt").read_text() == (
        "40%|z| 4/9 [..]\n"
        "2026-10-06 04:00:00,000 INFO something fetched=7\n")


def test_torn_plain_line_waits_for_its_terminator(tmp_path):
    """Content without \r reassembles byte-for-byte across write calls —
    multi-arg prints never split across lines."""
    console = io.StringIO()
    tee = run_log_tee._Tee(console, open(tmp_path / "t.txt", "w"))
    tee.write("  ✅ 3461 games, ")
    tee.write("71 features, 56 folds\n")
    tee.close()
    assert (tmp_path / "t.txt").read_text() == (
        "  ✅ 3461 games, 71 features, 56 folds\n")


def test_stale_end_date_pin_extends_to_today_and_passes_others_through(
        monkeypatch):
    """Stale-pin guard (MLB/NHL/NFL parity): a literal NBA_END_DATE pin
    before today must extend to today so the daily slate cannot freeze.
    The notebook is Kaggle-owned (never edited from the repo), so the
    PIPELINE defends itself. Same-day pins, forward-looking windows and
    malformed input all pass through untouched."""
    from datetime import date
    import ingestion
    monkeypatch.delenv("NBA_START_DATE", raising=False)
    monkeypatch.setenv("NBA_END_DATE", "2001-01-01")
    _start, end = ingestion.window()
    assert end == date.today()  # stale → today
    monkeypatch.setenv("NBA_END_DATE", date.today().isoformat())
    _start, end = ingestion.window()
    assert end == date.today()  # same-day passes
    monkeypatch.setenv("NBA_END_DATE", "2099-12-31")
    _start, end = ingestion.window()
    assert end.isoformat() == "2099-12-31"  # forward window passes
    monkeypatch.setenv("NBA_END_DATE", "not-a-date")
    _start, end = ingestion.window()
    assert end == date.today()  # malformed → the no-pin default


def test_stale_pin_guard_warns_where_the_window_resolves():
    """The guard must be WIRED where the run log can see it: window()
    re-announces an extension through the logger so the pushed run log
    states why the window no longer matches the notebook's pin."""
    src = (BACKEND / "ingestion.py").read_text(encoding="utf-8")
    assert "is stale (before today) — extended to" in src
    assert "slate cannot freeze" in src


def test_published_blend_line_names_the_deployed_weights():
    """Published-blend evidence (MLB/NHL parity): the re-pool must
    announce itself in the run log — one line with the applied weights
    and the row count, stating that headline metrics grade THE serving
    blend. The 2026-10-06 log's re-pool left no trace."""
    src = (BACKEND / "moneyline.py").read_text(encoding="utf-8")
    assert "Published blend:" in src, (
        "moneyline.py no longer logs the published-blend re-pool — the "
        "headline metrics' blend claim is unauditable from the run log")
    assert "headline metrics grade THE serving blend" in src


def test_platt_fallbacks_report_n_and_the_block_summary():
    """Calibration evidence (MLB/NHL parity): the delivered log carried
    NO calibration line and fit_platt's identity fallbacks were silent.
    The below-minimum and degenerate paths must warn with n= behind a
    3-line cap, keep uncapped counters, and the published-blend block
    must report the true fallback total."""
    src = (BACKEND / "moneyline.py").read_text(encoding="utf-8")
    assert "degenerate Platt params (a=%s, n=%d)" in src, (
        "the degenerate-Platt warning lost its n= — a degenerate fit's "
        "sample size must be visible in the log")
    assert "OOF games < %d minimum" in src
    assert "fit_platt._fit_total" in src and "fit_platt._degen_total" in src
    assert "fit_platt._min_total" in src
    assert "published-blend prequential Platt fits" in src


def test_final_pooled_calibrator_line_states_the_shipped_map():
    """The run log must state which calibrator ships — fitted parameters
    or identity — next to the phase that fits it (2026-10-06 log review:
    zero calibration lines in the delivered log)."""
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    assert "final pooled calibrator: a=%.4f b=%.4f n=%d method=%s" in src
    assert "final pooled calibrator: identity map" in src  # the None branch speaks too


def test_final_run_log_delivery_pushes_the_complete_log_after_sync():
    """Final delivery (MLB/NHL/NFL parity): _sync_data_delivery copies
    the log into a throwaway clone — a SNAPSHOT taken before its own push
    prints ever reached the file, so every pushed log ended at the
    publish ✅ (2026-10-06 review: 5,654 lines ending there, sync prints
    and the summary absent). The complete log is re-staged and pushed as
    the run's LAST delivery — after the sync, push-gated, never fatal."""
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    assert "Final run-log delivery" in src
    assert "NBA pipeline run log (final delivery" in src
    sync_at = src.index("_sync_data_delivery(config.ROOT_DIR.parent)")
    final_at = src.index("Final run-log delivery")
    assert sync_at < final_at, "the final log push must come after the sync"
    assert "if _log_path and _push_enabled()[0]" in src  # push-gated
    assert "_commit_partial" in src  # tees' pending lines land first
