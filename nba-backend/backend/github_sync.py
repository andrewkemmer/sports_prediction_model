"""
GitHub sync for the NBA pipeline.

Modeled directly on ``mlb-backend/backend/github_sync.py``, deliberately kept as
a separate module rather than imported from the MLB backend: the production
graph does not import another sport's backend, and a delivery bug in one sport
should not be able to break another's run.

Why this exists at all: the NBA pipeline used to end its delivery phase by
returning ``{"staged_files": [], "skipped": "automatic Git staging/commit/push
disabled"}``. That is a no-op, so on Kaggle - where the working directory is
ephemeral - the 19 artifacts a run produced were written and then discarded at
session end, while ``nba-backend/data_delivery`` stayed empty on ``main``
forever. MLB, NFL and NHL all publish; NBA was the only sport that did not.

The helpers here are the three that make publishing survivable rather than
fragile, all of them responding to a failure mode that actually cost MLB a
delivery:

* :func:`sync_remote_tip` - a reused clone can sit on a stale snapshot, and
  committing on top of a stale parent makes the push a *guaranteed*
  non-fast-forward rejection. Fetch, hard-reset, clean.
* :func:`push_with_retry` - a rejection is a race, not a verdict. Re-sync and
  replay rather than losing the run's artifacts.
* :func:`verify_pushed_paths` - a push that reports success and delivers
  nothing is worse than one that failed, because the run reports success too.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _rejected_pushinfos(info) -> list:
    """PushInfo objects carrying an error/rejection flag (bad = failed ref)."""
    import git

    bad_flags = (git.PushInfo.ERROR | git.PushInfo.REJECTED
                 | git.PushInfo.REMOTE_REJECTED | git.PushInfo.REMOTE_FAILURE)
    return [p for p in info if p.flags & bad_flags]


def sync_remote_tip(repo, branch: str = "main", log: Optional[Callable] = None) -> None:
    """Hard-heal ``repo`` to the CURRENT remote tip of ``branch``.

    Safe for this use case specifically because the sync clone is a throwaway
    whose only local changes are files the caller stages *after* this reset.
    Never call this on a checkout holding work.
    """
    import git

    say = log or (lambda msg: None)
    repo.git.fetch("origin", branch)
    try:
        tip = repo.git.rev_parse(f"origin/{branch}")
    except git.GitCommandError:  # very old git: tracking ref not updated
        tip = repo.git.rev_parse("FETCH_HEAD")
    repo.git.reset("--hard", tip)
    repo.git.clean("-fd")
    say(f"  synced clone to remote tip {tip[:10]} ({branch})")


def verify_pushed_paths(repo, branch: str, paths: list[str]) -> None:
    """Verify the pushed branch actually contains every requested artifact.

    A push can report success and deliver nothing. Since the run reports
    success off the same signal, an unverified push turns a delivery failure
    into a silent one, so this is checked against the remote tree rather than
    trusted.
    """
    repo.git.fetch("origin", branch)
    remote_tip = repo.git.rev_parse(f"origin/{branch}")
    listed = set(repo.git.ls_tree("-r", "--name-only", remote_tip).splitlines())
    missing = sorted(set(paths) - listed)
    if missing:
        raise RuntimeError(
            f"remote verification missing {len(missing)} artifact(s): "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")


def push_with_retry(repo, branch: str, restage: Optional[Callable] = None,
                    attempts: int = 3, log: Optional[Callable] = None) -> None:
    """Push ``branch`` with bounded, self-healing retries; raise on final failure.

    On a non-fast-forward rejection the throwaway clone is re-synced to the new
    remote tip and ``restage()`` replays this run's artifact copy + add +
    commit on top. The replay is safe because the run only ever writes under
    ``nba-backend/data_delivery``.
    """
    say = log or (lambda msg: None)
    restage = restage or (lambda: None)
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        info = repo.remote("origin").push(branch)
        bad = _rejected_pushinfos(info)
        if not bad:
            return
        last_error = RuntimeError(
            f"Push rejected by remote: {[p.summary for p in bad]}")
        if attempt == attempts:
            break
        say(f"  push rejected (attempt {attempt}/{attempts}) - re-syncing to "
            f"the current remote tip and replaying this run")
        sync_remote_tip(repo, branch, log=log)
        restage()
    raise last_error
