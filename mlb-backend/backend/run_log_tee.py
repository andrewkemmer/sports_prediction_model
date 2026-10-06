"""Run-log tee: capture the whole pipeline run into ONE rolling master file.

The Kaggle/Colab run's stdout/stderr is the only place phase banners, fold
progress, and WARNING context live — but it evaporates with the VM. Writing
it to ``data_delivery/mlb_pipeline_run_log.txt`` (overwritten every run)
makes the log a pushed artifact like any other, so a reviewer can read the
latest run with a plain git pull instead of a pasted log.

Installed by master_pipeline right after logging.basicConfig (before the
Phase 1 import), so EVERYTHING the run prints — including the tee's own
one-line header naming the log path — lands in the file. Phase 5 stages it
like any other regenerated artifact (its mtime always moves) and Phase 6
keeps it because the name is protected in retention_policy
(EXACT_MASTER_NAMES, dateless master file).

CRASH DELIVERY: a run that dies BEFORE Phase 5 would never push the log —
the traceback would sit on the VM with everything else. install_crash_log_pusher
registers a sys.excepthook that, on any uncaught exception AFTER the default
traceback printer runs, clones the remote tip and pushes just the log file
(subject: 'crash delivery'). Before the copy, _record_failure appends an
explicit crash marker + formatted traceback DIRECTLY to the log file, so the
delivered log always carries its own failure context even when a harness
re-bound sys.stdout/sys.stderr after the tee installed (the default hook
would then print the traceback everywhere EXCEPT the file). Like the tee,
every failure mode degrades silently — a failed crash push must never mask
the original error.

PROGRESS FRAMES (2026-10-06 log review): the file side used to turn every
carriage return into a newline, so each tqdm frame became its own line —
665 of the 1,105-line committed log were superseded progress frames. The
file side now COLLAPSES CR runs: a ``\r`` supersedes the partial line
before it, so a bar lands as ONE line (its final frame) while the console
keeps raw ``\r`` for tqdm's graphical widget.
"""
from __future__ import annotations

import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

RUN_LOG_NAME = "mlb_pipeline_run_log.txt"


class _Tee:
    """File+stream duplicator with an explicit closing contract.

    The console keeps RAW ``\r`` (tqdm's graphical-bar contract — Kaggle
    coalesces \r-frames into the black progress widget); the FILE is
    line-oriented and, since the 2026-10-06 log review, COLLAPSES CR runs:
    every ``\r`` supersedes the partial line before it, so a progress bar
    lands as ONE line — its final frame — instead of one line per frame
    (665 of 1,105 lines in the 2026-10-06 log were superseded frames).
    Content without ``\r`` reassembles exactly as written: a torn write
    (no newline yet) waits for its terminator, so multi-arg prints never
    split across lines.
    """

    def __init__(self, stream, file) -> None:
        self._stream = stream
        self._file = file
        self._partial = ""        # current line assembled so far
        self._in_cr_run = False   # last terminator was \r (partial = a frame)

    def write(self, data: str) -> int:
        orig = data  # the console contract: ALWAYS the caller's bytes, verbatim
        try:
            if data and self._in_cr_run and data[0] != "\r":
                # The CR-run ended at this write boundary: the final frame
                # becomes a line of its own — unless data opens with the
                # newline that completes it. tqdm emits each frame as one
                # write, so no real line can be split by this commit.
                if data[0] == "\n":
                    data = self._partial + data
                elif self._partial:
                    self._file.write(self._partial + "\n")
                self._partial = ""
                self._in_cr_run = False
            if "\r" in data:
                for ch in data:  # rare path — progress-bar frames only
                    if ch == "\r":
                        self._partial = ""  # supersede the frame before it
                        self._in_cr_run = True
                    elif ch == "\n":
                        self._file.write(self._partial + "\n")
                        self._partial = ""
                        self._in_cr_run = False
                    else:
                        self._partial += ch
            else:
                if self._partial:
                    data = self._partial + data  # torn line: wait for \n
                    self._partial = ""
                if "\n" in data:
                    cut = data.rfind("\n")
                    self._file.write(data[:cut + 1])
                    data = data[cut + 1:]
                self._partial = data
            self._file.flush()
        except (OSError, ValueError):
            pass  # disk full/removed mid-run: console keeps working
        return self._stream.write(orig)

    def _commit_partial(self) -> None:
        """Land whatever line is half-written (close / atexit delivery)."""
        try:
            if self._partial:
                self._file.write(self._partial + "\n")
            self._partial = ""
            self._in_cr_run = False
        except (OSError, ValueError):
            pass

    def flush(self) -> None:
        # Deliberately does NOT commit an open CR-run: tqdm flushes after
        # every frame, and committing would flood the file with exactly
        # the superseded frames the collapse removes. Torn plain lines
        # behave like the line-buffered file underneath (flush on \n).
        try:
            self._file.flush()
        except (OSError, ValueError):
            pass
        self._stream.flush()

    def isatty(self) -> bool:
        # Delegate — a hardcoded False flipped tqdm into text mode (the
        # 2026-10-01 regression: Kaggle's graphical black bar became flat
        # pink stderr lines). tqdm must see exactly what it saw pre-tee.
        try:
            return self._stream.isatty()
        except (OSError, ValueError):
            return False

    def fileno(self) -> int:
        return self._stream.fileno()

    def close(self) -> None:  # the FILE is tee-owned; the console never is
        self._commit_partial()  # a run that ends on a frame still lands it
        self._file.close()


def install_run_log_tee(data_delivery_dir: Path) -> Path | None:
    """Start tee-ing sys.stdout/stderr into RUN_LOG_NAME under ``dir``.

    Idempotent per process: if stdout is already a tee, returns None.
    Overwrites any previous run's file (this is the one rolling master
    log, not an accumulation). Returns the log path, or None when the
    destination is unwritable — the pipeline then just prints normally.
    """
    if isinstance(sys.stdout, _Tee):
        return None
    try:
        data_delivery_dir.mkdir(parents=True, exist_ok=True)
        log_path = data_delivery_dir / RUN_LOG_NAME
        # Line-buffered text file; failures fall back to console-only.
        log_file = open(log_path, "w", buffering=1, encoding="utf-8",
                        errors="replace")
    except OSError:
        return None
    # Tee the CURRENT stream objects (whatever harness replaced them with),
    # preserving their targets; the idempotence check above prevents chains.
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_stdout, log_file)
    sys.stderr = _Tee(orig_stderr, log_file)
    # logging.basicConfig ran BEFORE this install, so its StreamHandler
    # captured the PRE-tee stderr and every INFO/WARNING record bypassed
    # the log (the 2026-10-01 push carried banners only — no folds, no
    # weather, no calibration-gate WARNING). Re-point handlers still
    # aimed at the captured console streams; handlers owned by an outer
    # harness (pytest's capture, notebook kernels) hold their own stream
    # objects and are left alone.
    for _h in logging.getLogger().handlers:
        if isinstance(_h, logging.StreamHandler) \
                and _h.stream in (orig_stdout, orig_stderr):
            _h.stream = _Tee(_h.stream, log_file)
    import atexit
    out_tee, err_tee = sys.stdout, sys.stderr  # the _Tee objects just installed

    def _close_log() -> None:
        # Commit any half-written line FIRST — the atexit target is the raw
        # file object, which does not know about the tees' pending partials.
        for _t in (out_tee, err_tee):
            _t._commit_partial()
        log_file.close()

    atexit.register(_close_log)
    print(f"  📝 Run log tee: {log_path} (one rolling master file, "
          f"overwritten every run)")
    return log_path


def push_log_on_crash(log_path: Path, username: str, repo_name: str,
                      branch: str = "main") -> bool:
    """Push ONLY the run log to data_delivery (crash delivery).

    Self-contained so it works no matter how early the pipeline died:
    fresh depth-1 clone of the remote tip, copy the log over the protected
    master name, commit, push_with_retry. Returns True on a confirmed push.
    """
    token = os.getenv("GITHUB_TOKEN", "")
    if not token or not log_path or not Path(log_path).exists():
        return False
    import shutil
    import tempfile
    import git
    from github_sync import push_with_retry
    with tempfile.TemporaryDirectory(prefix="mlb_crash_log_") as td:
        auth_url = (f"https://{token}@github.com/{username}/{repo_name}.git")
        repo = git.Repo.clone_from(auth_url, td, branch=branch, depth=1)
        rel = f"mlb-backend/data_delivery/{RUN_LOG_NAME}"
        dest = Path(td) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log_path, dest)
        repo.index.add([rel])
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        repo.index.commit(f"Pipeline run log (crash delivery, {stamp})")
        push_with_retry(repo, branch, restage=None, attempts=2, log=print)
    return True


def _record_failure(log_path: Path, exc_type, exc, tb) -> None:
    """Append a terminal crash marker + full traceback DIRECTLY to the log.

    The default excepthook prints to whatever ``sys.stderr`` holds at
    crash time. If a harness rebound stdout/stderr AFTER the tee
    installed, that traceback bypasses the file — and the crash delivery
    would ship a log that stops mid-run with no failure line at all.
    Writing straight to the file cannot be re-routed, so every delivered
    crash log states what killed the run. Appends (never truncates) and
    degrades silently: recording must never mask the original error.
    """
    try:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = "".join(traceback.format_exception(exc_type, exc, tb))
        with open(log_path, "a", encoding="utf-8", errors="replace") as f:
            f.write(f"\n  ❌ PIPELINE CRASH {stamp} — "
                    f"{exc_type.__name__}: {exc}\n")
            f.write(body.rstrip() + "\n")
            f.flush()
    except Exception:
        pass  # the marker is a courtesy; the original error is the point


def install_crash_log_pusher(log_path: Path | None, username: str,
                             repo_name: str, branch: str = "main") -> None:
    """On any uncaught exception, print the traceback (the ORIGINAL hook's
    job — never skipped), record it in the log file, then best-effort push
    the captured log."""
    if not log_path:
        return
    previous = sys.excepthook

    def _hook(exc_type, exc, tb) -> None:
        previous(exc_type, exc, tb)  # traceback to console (and tee) first
        if isinstance(exc, SystemExit):
            return  # deliberate exits are not crashes
        # Flush whatever streams the process currently holds, then append
        # the failure marker straight to the file — a stream re-bound by a
        # harness must not be able to keep the traceback out of the copy
        # that crash delivery ships.
        for _s in (sys.stdout, sys.stderr):
            try:
                _s.flush()
            except Exception:
                pass
        try:
            _record_failure(log_path, exc_type, exc, tb)
        except Exception:
            pass  # marker failure must never block the delivery itself
        try:
            if push_log_on_crash(log_path, username, repo_name, branch):
                print("  📝 Crash log pushed to "
                      f"mlb-backend/data_delivery/{RUN_LOG_NAME}")
            else:
                print("  ⚠️  Crash log NOT pushed (no token or no log file)")
        except Exception as push_exc:  # never mask the original error
            print(f"  ⚠️  Crash log push failed: {push_exc}")

    sys.excepthook = _hook
