"""
GitHub sync for MLB Bet Predictor.

Uses GitPython to clone the repo, copy artifacts into data_delivery/,
commit, and push. Provides robust error handling and supports both
SSH key and PAT authentication in Colab.

Also hosts the shared daily-run sync helpers used by master_pipeline.py
(Phases 5/6): ``sync_remote_tip`` heals a reused sync clone to the current
remote tip, and ``push_with_retry`` pushes with bounded, self-healing
retries so a non-fast-forward race (another actor pushed between our clone
and our push) no longer kills the run's artifact delivery.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def sync_artifacts(
    repo_url: Optional[str] = None,
    data_delivery_dir: Optional[Path] = None,
    branch: str = "main",
) -> dict:
    """Clone repo, copy artifacts, commit, and push.

    Args:
        repo_url: Git remote URL. Falls back to GITHUB_REPO_URL env var.
        data_delivery_dir: Local path to artifacts. Defaults to config.DATA_DELIVERY_DIR.
        branch: Branch to push to.

    Returns:
        Status dict with keys: pushed, commit_sha, staged_files, error
    """
    from config import DATA_DELIVERY_DIR, ROOT_DIR

    repo_url = repo_url or os.environ.get("GITHUB_REPO_URL", "")
    data_delivery_dir = data_delivery_dir or DATA_DELIVERY_DIR

    if not repo_url:
        return {
            "pushed": False,
            "commit_sha": None,
            "staged_files": [],
            "error": "No GITHUB_REPO_URL set. Export it in Colab.",
        }

    # Build auth URL if PAT is provided
    github_token = os.environ.get("GITHUB_TOKEN", "")
    if github_token and repo_url.startswith("https://"):
        # Inject token into URL for push only
        auth_url = repo_url.replace("https://", f"https://{github_token}@")
    else:
        auth_url = repo_url

    tmp_dir = None
    try:
        import git

        tmp_dir = tempfile.mkdtemp(prefix="mlb_sync_")
        logger.info("Cloning %s → %s", repo_url, tmp_dir)

        repo = git.Repo.clone_from(
            auth_url,
            tmp_dir,
            branch=branch,
            depth=1,
        )

        # Copy artifacts into cloned repo's data_delivery
        dest_delivery = Path(tmp_dir) / "data_delivery"
        dest_delivery.mkdir(exist_ok=True)

        # Copy models dir too
        models_src = data_delivery_dir / "models"
        if models_src.exists():
            models_dest = dest_delivery / "models"
            shutil.copytree(models_src, models_dest, dirs_exist_ok=True)

        staged_files = []
        for item in data_delivery_dir.iterdir():
            if item.is_file():
                dest = dest_delivery / item.name
                shutil.copy2(item, dest)
                staged_files.append(f"data_delivery/{item.name}")

        # Stage and commit
        repo.index.add(staged_files)
        commit_msg = f"Update artifacts: {len(staged_files)} files"
        commit = repo.index.commit(commit_msg)

        # Push
        origin = repo.remote("origin")
        origin.push(branch)

        logger.info("Pushed %d files, commit %s", len(staged_files), commit.hexsha[:8])
        return {
            "pushed": True,
            "commit_sha": commit.hexsha,
            "staged_files": staged_files,
            "error": None,
        }

    except ImportError:
        msg = "GitPython not installed. Install with: pip install GitPython"
        logger.error(msg)
        return {"pushed": False, "commit_sha": None, "staged_files": [], "error": msg}

    except Exception as e:
        msg = f"Sync failed: {e}"
        logger.error(msg)
        return {"pushed": False, "commit_sha": None, "staged_files": [], "error": msg}

    finally:
        if tmp_dir and Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Daily-run sync helpers (used by master_pipeline.py Phases 5/6) ─────────


def _rejected_pushinfos(info) -> list:
    """PushInfo objects carrying an error/rejection flag (bad = failed ref)."""
    import git

    bad_flags = (git.PushInfo.ERROR | git.PushInfo.REJECTED
                 | git.PushInfo.REMOTE_REJECTED | git.PushInfo.REMOTE_FAILURE)
    return [p for p in info if p.flags & bad_flags]


def sync_remote_tip(repo, branch: str = "main", log=None) -> None:
    """Hard-heal ``repo`` to the CURRENT remote tip of ``branch``.

    A reused (warm) sync clone can sit on a stale snapshot of the branch —
    the previous run crashed before its cleanup, or a manual push landed
    between runs — and committing on top of a stale parent makes the
    eventual push a guaranteed non-fast-forward rejection. Fetch, then hard
    reset (and drop stray untracked files) so every commit lands on the
    true tip. Safe for the artifact-sync use case: the clone is a throwaway
    whose only local changes are files the caller stages AFTER this reset.
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
    """Verify the pushed branch contains every requested artifact path."""
    import git
    repo.git.fetch("origin", branch)
    remote_tip = repo.git.rev_parse(f"origin/{branch}")
    listed = set(repo.git.ls_tree("-r", "--name-only", remote_tip).splitlines())
    missing = sorted(set(paths) - listed)
    if missing:
        raise RuntimeError(f"remote verification missing artifacts: {missing}")


def push_with_retry(repo, branch: str, restage=None, attempts: int = 3,
                    log=None) -> None:
    """Push ``branch`` with bounded, self-healing retries; raise on final failure.

    On a non-fast-forward rejection the throwaway clone is re-synced to the
    NEW remote tip via :func:`sync_remote_tip` and ``restage()`` replays this
    run's changes (re-copy + re-add + re-commit for Phase 5 artifacts; re-run
    the stale-file rm for Phase 6) on top, then the push is attempted again.
    The replay is always safe because the run only ever touches
    ``<sport>/data_delivery/**`` — replaying it onto any fresh tip is valid.
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
        say(f"  ⚠️  Push rejected (attempt {attempt}/{attempts}) — "
            f"re-syncing to the current remote tip and replaying this run")
        sync_remote_tip(repo, branch, log=log)
        restage()
    raise last_error
