"""Pin the run-log tee + retention protection (2026-10-01).

The Kaggle run's stdout evaporates with the VM; the tee captures it into
one rolling master file (data_delivery/nfl_pipeline_run_log.txt) that the
end-of-run sync delivers and retention protects, so the latest run is
reviewable from a plain git pull. Each test pins one contract:

* the tee duplicates console output into the file and keeps the console
  return contract (bytes-written passthrough);
* tty-ness and carriage returns pass through to the CONSOLE untouched
  (tqdm's graphical black bar), while the FILE collapses CR runs to each
  bar's final frame — one line per bar, never one line per frame (the
  2026-10-06 review found 860 of 1,431 committed lines were superseded
  tqdm frames);
* logging handlers are re-pointed at the tee, so INFO/WARNING records
  reach the log (basicConfig captures the pre-tee stderr otherwise);
* installation overwrites the previous run's file (one rolling master);
* install is idempotent (no tee-over-tee chains) and degrades to
  console-only when the destination is unwritable;
* the retention policy classifies the log as protected even on a run that
  did not stage it (the dateless-name eviction trap), and stays stale-
  classification clean for genuinely dated artifacts;
* the 2026-10-06 run-log review remediation stays pinned: CR-frame
  collapse (file side), the published-blend line, the degenerate-Platt
  n=, the stale END_DATE guard, and the final run-log delivery.
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


def test_tee_degrades_console_write_on_non_utf8_console(tmp_path):
    # A cp1252 console cannot encode the pipeline's emoji; the
    # tee must degrade the CONSOLE copy (the FILE keeps the exact
    # bytes) instead of crashing the run on the first banner.
    class _Cp1252Console:
        encoding = "cp1252"

        def __init__(self):
            self.written: list[str] = []

        def write(self, s):
            s.encode("cp1252")  # raises on emoji, like the real console
            self.written.append(s)
            return len(s)

        def flush(self):
            pass

        def isatty(self):
            return False

        def fileno(self):
            return 1

    console = _Cp1252Console()
    log_file = open(tmp_path / "log.txt", "w", encoding="utf-8")
    tee = run_log_tee._Tee(console, log_file)
    n = tee.write("📝 emoji banner\n")
    tee.flush()
    tee.close()
    # passthrough contract survives the degraded write
    assert n == len("📝 emoji banner\n")
    assert "📝 emoji banner" in (tmp_path / "log.txt") \
        .read_text(encoding="utf-8")  # file keeps the exact bytes
    assert "? emoji banner" in console.written[-1]  # console degraded


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
    # The tee writes UTF-8 by contract; read it back explicitly
    # (Windows' cp1252 default cannot decode the header emoji).
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


# ── tty passthrough + \r contract (tqdm loading-bar regression) ──────

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
    \r→\n conversion turned every superseded frame into its own line, 860
    of the 1,431 committed log lines; MLB/NHL CR-collapse parity)."""
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


# ── logging rebind (basicConfig captures the pre-tee stderr) ─────────

def test_install_rebinds_logging_handlers_so_records_reach_the_log(
        tmp_path, monkeypatch):
    import logging

    # Model production: basicConfig's handler holds the SAME pre-tee
    # stderr object the tee is about to wrap.
    pre_tee_stderr = io.StringIO()
    handler = logging.StreamHandler(pre_tee_stderr)
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    old_level = root.level
    root.setLevel(logging.INFO)  # production runs basicConfig(level=INFO)
    root.handlers[:] = [handler]
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", pre_tee_stderr)
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
        assert "fold line the tee must capture" in pre_tee_stderr.getvalue()
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


# ── Retention contract ──────────────────────────────────────────────────

def test_run_log_is_protected_from_stale_eviction():
    # A run that did NOT stage the log (not in seen) must still keep it:
    # the dateless name is registered as a protected master.
    verdict = classify_artifact(
        "nfl-backend/data_delivery/nfl_pipeline_run_log.txt",
        seen=set(), retention_dates=set(), recent_dates=set(),
        board_dates=set())
    assert verdict == "protected"


def test_dated_artifacts_outside_windows_still_classify_stale():
    # Guard the guard: the exemption is name-specific, not a blanket
    # dateless-file amnesty.
    verdict = classify_artifact(
        "nfl-backend/data_delivery/nfl_board_20200101.csv",
        seen=set(), retention_dates=set(), recent_dates=set(),
        board_dates=set())
    assert verdict == "stale"


# ── crash delivery (the log must outlive the VM) ───────────────────────

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
    """The glue case the old _is_log_record patch worked around: a logging
    record arriving while a bar is open must open its own line — the
    frame becomes the bar's final line, the record follows intact."""
    console = io.StringIO()
    tee = run_log_tee._Tee(console, open(tmp_path / "r.txt", "w"))
    tee.write("\r40%|z| 4/9 [..]")  # tqdm frames LEAD each frame with \r
    tee.write("2026-10-06 04:00:00,000 INFO something fetched=7\n")
    tee.close()
    assert (tmp_path / "r.txt").read_text() == (
        "40%|z| 4/9 [..]\n"
        "2026-10-06 04:00:00,000 INFO something fetched=7\n")


def test_stale_end_date_pin_extends_to_today_and_passes_others_through():
    """T7 (MLB/NHL parity): a literal NFL_END_DATE pin before today must
    extend to today so the daily slate cannot freeze. The notebook is
    Kaggle-owned (never edited from the repo), so the PIPELINE defends
    itself. Same-day pins, the forward-looking runner window, and
    malformed input all pass through untouched."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import config
    today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    assert config.resolve_run_end_date("2001-01-01") == today  # stale → today
    assert config.resolve_run_end_date(today) == today          # same-day passes
    future = "2099-12-31"
    assert config.resolve_run_end_date(future) == future  # forward window passes
    assert config.resolve_run_end_date("not-a-date") == "not-a-date"  # malformed


def test_end_bounds_and_phase1_wire_the_stale_pin_guard():
    """The guard must be WIRED: _env_end_bounds runs every resolved day
    through config.resolve_run_end_date, and main() re-announces an
    extension through the logger so the pushed run log states why the
    window no longer matches the notebook's pin."""
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    assert "day = config.resolve_run_end_date(day)" in src
    assert "end-date pin %s is stale" in src  # Phase-1 WARNING into the log


def test_published_blend_line_names_the_deployed_weights():
    """T3 (MLB/NHL parity): the published-blend pass must announce itself
    in the run log — one line with the applied weights and the row count,
    stating that headline metrics grade THE serving blend. The 2026-10-06
    log's re-pool left no trace."""
    src = (BACKEND / "moneyline.py").read_text(encoding="utf-8")
    assert "Published blend:" in src, (
        "moneyline.py no longer logs the published-blend re-pool — the "
        "headline metrics' blend claim is unauditable from the run log")
    assert "headline metrics grade THE serving blend" in src


def test_degenerate_platt_warning_carries_the_sample_size():
    """T4 (MLB/NHL parity): the degenerate-Platt warning must report n=,
    fit_platt keeps uncapped _fit_total/_degen_total counters, and the
    8a prequential block reports its own degenerate total behind the
    3-line cap."""
    src = (BACKEND / "moneyline.py").read_text(encoding="utf-8")
    assert "degenerate Platt params (a=%s, n=%d)" in src, (
        "the degenerate-Platt warning lost its n= — a degenerate fit's "
        "sample size must be visible in the log")
    assert "fit_platt._fit_total" in src and "fit_platt._degen_total" in src
    msrc = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    assert "prequential Platt fits degenerated" in msrc, (
        "the prequential block must report the uncapped degenerate total")


def test_final_run_log_delivery_pushes_the_complete_log_after_sync():
    """Final delivery (MLB/NHL parity): _sync_data_delivery stages a
    SNAPSHOT of the log taken before its own push prints, so every pushed
    log ended at the DONE banner (2026-10-06 review: 1,431 lines ending
    at DONE, sync prints absent). The complete log is re-staged and pushed
    as the run's LAST delivery — after sync, token-guarded, never fatal."""
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    assert "Final run-log delivery" in src
    assert "NFL pipeline run log (final delivery)" in src
    sync_at = src.index("_sync_data_delivery(repo_root=")
    final_at = src.index("Final run-log delivery")
    assert sync_at < final_at, "the final log push must come after the sync"
    assert 'os.environ.get("GITHUB_TOKEN"' in src  # token-guarded push
    assert '_env_flag("NFL_NO_PUSH")' in src       # local runs never push
