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

The tee is intentionally forgiving: every failure mode degrades to plain
console logging (never raises into the pipeline).
"""
from __future__ import annotations

import sys
from pathlib import Path

RUN_LOG_NAME = "mlb_pipeline_run_log.txt"


class _Tee:
    """File+stream duplicator with an explicit closing contract."""

    def __init__(self, stream, file) -> None:
        self._stream = stream
        self._file = file

    def write(self, data: str) -> int:
        try:
            self._file.write(data)
            self._file.flush()
        except (OSError, ValueError):
            pass  # disk full/removed mid-run: console keeps working
        return self._stream.write(data)

    def flush(self) -> None:
        try:
            self._file.flush()
        except (OSError, ValueError):
            pass
        self._stream.flush()

    def isatty(self) -> bool:
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
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    import atexit
    atexit.register(log_file.close)
    print(f"  📝 Run log tee: {log_path} (one rolling master file, "
          f"overwritten every run)")
    return log_path
