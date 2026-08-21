"""Push daily artifacts to the GitHub ``data_delivery`` folder via GitPython.

``data_delivery`` is the canonical artifact sink: the Streamlit frontend reads
it from raw.githubusercontent.com, so whatever the Colab pipeline pushes is
what the dashboard shows.

Credentials (never hardcoded — environment variables only)
----------------------------------------------------------
Option A — SSH key (recommended for Colab):
    1. In Colab:  ``!ssh-keygen -t ed25519 -f /root/.ssh/id_ed25519 -N '' -C colab``
    2. Add the public key (``!cat /root/.ssh/id_ed25519.pub``) to
       GitHub → Settings → SSH and GPG keys.
    3. Set ``GITHUB_REPO_URL`` to the SSH URL:
       ``git@github.com:<owner>/<repo>.git``

Option B — Personal Access Token (PAT):
    1. GitHub → Settings → Developer settings → Personal access tokens
       → generate a token with ``repo`` scope.
    2. Export it in Colab: ``os.environ["GITHUB_TOKEN"] = "ghp_..."``
    3. Set ``GITHUB_REPO_URL`` to the HTTPS URL:
       ``https://github.com/<owner>/<repo>.git``

The token is injected into the remote URL only for the push and is never
written to disk or committed.
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import date as date_cls
from pathlib import Path
from typing import Optional

from config import DATA_DIR, GITHUB_BRANCH, GITHUB_REPO_URL, REPO_LOCAL_CLONE

logger = logging.getLogger(__name__)


class GitHubSyncError(RuntimeError):
    """Raised when artifact sync fails after retries; never contains secrets."""


def _get_git():
    try:
        import git  # GitPython, lazy
        return git
    except ImportError as exc:  # pragma: no cover
        raise GitHubSyncError(
            "GitPython is required for GitHub sync. Install it with "
            "`pip install GitPython` (in backend/requirements.txt)."
        ) from exc


def resolve_repo_url() -> str:
    """Repo URL from env: GITHUB_REPO_URL, else https://github.com/OWNER/REPO.git."""
    url = GITHUB_REPO_URL.strip()
    if url:
        return url
    owner = os.environ.get("GITHUB_OWNER", "").strip()
    repo = os.environ.get("GITHUB_REPO", "").strip()
    if owner and repo:
        return f"https://github.com/{owner}/{repo}.git"
    raise GitHubSyncError(
        "No GitHub repo configured. Set GITHUB_REPO_URL (or GITHUB_OWNER and "
        "GITHUB_REPO) as environment variables — see README.md for Colab setup."
    )


def _auth_remote(git, repo, url: str, token: Optional[str]) -> None:
    """Configure the origin remote; inject the PAT into the URL when present."""
    if token:
        scheme, rest = url.split("://", 1)
        authed = f"{scheme}://x-access-token:{token}@{rest}"
        repo.remotes.origin.set_url(authed)
    else:
        repo.remotes.origin.set_url(url)


def clone_or_open_repo(local_dir: Path, url: str, token: Optional[str] = None):
    """Clone the repo (or open an existing checkout) into ``local_dir``."""
    git = _get_git()
    if (local_dir / ".git").exists():
        repo = git.Repo(local_dir)
        _auth_remote(git, repo, url, token)
        repo.remotes.origin.fetch()
        repo.git.checkout(GITHUB_BRANCH, force=True)
        repo.remotes.origin.pull()
        return repo

    local_dir.mkdir(parents=True, exist_ok=True)
    if token:
        scheme, rest = url.split("://", 1)
        url = f"{scheme}://x-access-token:{token}@{rest}"
    return git.Repo.clone_from(url, str(local_dir), branch=GITHUB_BRANCH)


def sync_artifacts(
    target_date: date_cls,
    repo_url: Optional[str] = None,
    local_clone: Optional[Path] = None,
    push: bool = True,
) -> dict:
    """Copy artifacts into the repo's ``data_delivery/`` and push them.

    Returns a status dict::

        {
          "pushed": bool,
          "commit_sha": str | None,
          "staged_files": [ ... ],
          "repo_url": str,
          "error": str | None,
        }
    """
    status = {"pushed": False, "commit_sha": None, "staged_files": [], "repo_url": None, "error": None}
    try:
        url = repo_url or resolve_repo_url()
        status["repo_url"] = url
        token = os.environ.get("GITHUB_TOKEN", "").strip() or None
        clone_dir = Path(local_clone or REPO_LOCAL_CLONE)

        repo = clone_or_open_repo(clone_dir, url, token=token)
        repo_delivery = clone_dir / "data_delivery"
        repo_delivery.mkdir(parents=True, exist_ok=True)

        # Copy every artifact produced for this date (CSVs, JSON, models).
        yyyymmdd = f"{target_date:%Y%m%d}"
        staged = []
        for src in DATA_DIR.glob(f"*_{yyyymmdd}.*"):
            dst = repo_delivery / src.name
            shutil.copy2(src, dst)
            staged.append(str(dst.relative_to(clone_dir)))
        if (DATA_DIR / "models" / "ensemble_latest.joblib").exists():
            dst = repo_delivery / "models" / "ensemble_latest.joblib"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(DATA_DIR / "models" / "ensemble_latest.joblib", dst)
            staged.append(str(dst.relative_to(clone_dir)))
        for src in DATA_DIR.glob("shap_game_*.csv"):
            dst = repo_delivery / src.name
            shutil.copy2(src, dst)
            staged.append(str(dst.relative_to(clone_dir)))

        if not staged:
            raise GitHubSyncError(
                f"No artifacts matched *_{yyyymmdd}.* in {DATA_DIR}. "
                "Run the pipeline before syncing."
            )

        repo.git.add("data_delivery")
        commit_msg = f"Daily artifacts for {target_date:%Y-%m-%d} [skip ci]"
        repo.index.commit(commit_msg)
        status["commit_sha"] = repo.head.commit.hexsha[:12]
        status["staged_files"] = staged

        if push:
            repo.git.push("origin", GITHUB_BRANCH, set_upstream=True)
            status["pushed"] = True
        logger.info("Synced %d files to %s (commit %s)", len(staged), url, status["commit_sha"])
        return status

    except Exception as exc:  # noqa: BLE001 - surface a safe, actionable message
        status["error"] = (
            f"GitHub sync failed: {type(exc).__name__}: {exc}. "
            "Check GITHUB_REPO_URL / GITHUB_TOKEN env vars and network access."
        )
        logger.error(status["error"])
        return status


def configure_colab_ssh_instructions() -> str:
    """Human-readable Colab SSH setup steps (shown when sync is skipped)."""
    return (
        "# Colab SSH setup for GitPython push\n"
        "!ssh-keygen -t ed25519 -f /root/.ssh/id_ed25519 -N '' -C colab\n"
        "!cat /root/.ssh/id_ed25519.pub   # add this to GitHub → Settings → SSH keys\n"
        'os.environ["GITHUB_REPO_URL"] = "git@github.com:<owner>/<repo>.git"\n'
    )
