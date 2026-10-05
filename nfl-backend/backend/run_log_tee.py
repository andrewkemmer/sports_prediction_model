"""Run-log tee: capture the whole pipeline run into ONE rolling master file.

The Kaggle run's stdout/stderr is the only place phase banners, fold
progress, and WARNING context live — but it evaporates with the VM.
Writing it to ``data_delivery/nfl_pipeline_run_log.txt`` (overwritten
every run) makes the log a delivered artifact like any other, so a
reviewer can read the latest run with a plain git pull instead of a
pasted log.

Installed by master_pipeline.main() right at the top (before
argparse), so EVERYTHING the run prints — including the tee's own
one-line header naming the log path — lands in the file. The
end-of-run ``_sync_data_delivery`` stages the complete delivery tree
(``git add -A``), so the log rides along like any regenerated
artifact, and retention_policy protects the dateless master name
(EXACT_MASTER_NAMES) from ``_prune_old_artifacts``.

CRASH DELIVERY: a run that dies before the end-of-run sync would
never push the log — the traceback would sit on the VM with
everything else. install_crash_log_pusher registers a sys.excepthook
that, on any uncaught exception AFTER the default traceback printer
runs, clones the remote tip and pushes just the log (subject:
'crash delivery'). Before the copy, _record_failure appends an
explicit crash marker + formatted traceback DIRECTLY to the log file,
so the delivered log always carries its own failure context even when
a harness re-bound sys.stdout/sys.stderr after the tee installed (the
default hook would then print the traceback everywhere EXCEPT the
file). Like the tee, every failure mode degrades silently — a failed
crash push must never mask the original error.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

RUN_LOG_NAME = "nfl_pipeline_run_log.txt"

# Same delivery identity _sync_data_delivery configures for the
# end-of-run push (env-overridable there via GIT_USER_NAME/EMAIL).
_GIT_NAME = os.environ.get("GIT_USER_NAME", "NFL Production Pipeline")
_GIT_EMAIL = os.environ.get(
    "GIT_USER_EMAIL", "nfl-pipeline@users.noreply.github.com")


class _Tee:
    """File+stream duplicator with an explicit closing contract."""

    def __init__(self, stream, file) -> None:
        self._stream = stream
        self._file = file

    def write(self, data: str) -> int:
        try:
            # The console keeps RAW \r (tqdm's graphical-bar contract —
            # Kaggle coalesces \r-frames into the black progress widget);
            # the FILE is line-oriented, so carriage returns become
            # newlines instead of each frame overwriting the previous one.
            self._file.write(data.replace("\r", "\n"))
            self._file.flush()
        except (OSError, ValueError):
            pass  # disk full/removed mid-run: console keeps working
        try:
            return self._stream.write(data)
        except UnicodeEncodeError:
            # A non-UTF-8 console (a Windows cp1252 terminal) cannot
            # print the pipeline's emoji/box-drawing banners verbatim.
            # The FILE already holds the exact bytes, so degrade the
            # console copy to replaceable characters instead of
            # crashing the run on the first banner — the tee must
            # never be the reason a local run dies (the pre-tee
            # pipeline crashed at its first banner on such a
            # console anyway).
            enc = getattr(self._stream, "encoding", None) or "utf-8"
            safe = data.encode(enc, errors="replace").decode(
                enc, errors="replace")
            return self._stream.write(safe)

    def flush(self) -> None:
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
    tee_stdout = _Tee(orig_stdout, log_file)
    tee_stderr = _Tee(orig_stderr, log_file)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr
    # logging.basicConfig ran BEFORE this install (module import time),
    # so its StreamHandler captured the PRE-tee stderr and every
    # INFO/WARNING record bypassed the log. Re-point handlers still
    # aimed at the captured console streams; handlers owned by an outer
    # harness (pytest's capture, notebook kernels) hold their own stream
    # objects and are left alone.
    #
    # The handler receives the SAME _Tee object as sys.stdout/sys.stderr,
    # not a fresh wrapper of the same console. tqdm recognises a stream
    # by identity (external_write_mode's ``f in (sys.stdout, sys.stderr)``),
    # so a second _Tee around the same console would leave every console
    # handler invisible to it — and a log record would paint itself into
    # the middle of the live bar's row instead of above the bar. One
    # wrapper per stream keeps the identity the clear/write/repaint dance
    # in ingestion.StageProgress depends on.
    for _h in logging.getLogger().handlers:
        if isinstance(_h, logging.StreamHandler):
            if _h.stream is orig_stdout:
                _h.stream = tee_stdout
            elif _h.stream is orig_stderr:
                _h.stream = tee_stderr
    import atexit
    atexit.register(log_file.close)
    print(f"  📝 Run log tee: {log_path} (one rolling master file, "
          f"overwritten every run)")
    return log_path


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    """One raw git call — the same subprocess machinery
    _sync_data_delivery uses (this backend has no GitPython)."""
    return subprocess.run(
        ["git", *args], cwd=str(cwd), check=True,
        capture_output=True, text=True,
    )


def push_log_on_crash(log_path: Path, username: str, repo_name: str,
                      branch: str = "main") -> bool:
    """Push ONLY the run log to data_delivery (crash delivery).

    Self-contained so it works no matter how early the pipeline died:
    fresh depth-1 clone of the remote tip, copy the log over the
    protected master name, commit, push (2 attempts). Returns True on
    a confirmed push. Every failure returns False — the hook's caller
    decides what to print, and the original exception is never masked.
    """
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token or not log_path or not Path(log_path).exists():
        return False
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory(prefix="nfl_crash_log_") as td:
        td = Path(td)
        # Same authenticated URL form as _sync_data_delivery.
        auth_url = (
            "https://x-access-token:"
            f"{quote(token, safe='')}"
            f"@github.com/{username}/{repo_name}.git"
        )
        try:
            _git("clone", "--depth", "1", "--branch", branch,
                 auth_url, str(td), cwd=td)
            rel = f"nfl-backend/data_delivery/{RUN_LOG_NAME}"
            dest = td / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log_path, dest)
            _git("add", "--", rel, cwd=td)
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            _git("-c", f"user.name={_GIT_NAME}",
                 "-c", f"user.email={_GIT_EMAIL}",
                 "commit", "-m",
                 f"NFL pipeline run log (crash delivery, {stamp})",
                 cwd=td)
            for _attempt in (1, 2):
                try:
                    _git("push", "origin", branch, cwd=td)
                    return True
                except subprocess.CalledProcessError:
                    continue  # transient remote rejection — retry once
        except Exception:
            return False
    return False


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
                      f"nfl-backend/data_delivery/{RUN_LOG_NAME}")
            else:
                print("  ⚠️  Crash log NOT pushed (no token or no log file)")
        except Exception as push_exc:  # never mask the original error
            print(f"  ⚠️  Crash log push failed: {push_exc}")

    sys.excepthook = _hook
