"""
GitHub sync for MLB Bet Predictor.

Uses GitPython to clone the repo, copy artifacts into data_delivery/,
commit, and push. Provides robust error handling and supports both
SSH key and PAT authentication in Colab.
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
