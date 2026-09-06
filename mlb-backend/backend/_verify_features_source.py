"""
Standalone pre-flight check for the Kaggle notebook: confirms the features.py
that will be imported does not contain the known-corrupt duplicated/corrupted
arsenal block, and prints the exact commit/path/files lines around the
previously-failing location so a stale checkout can be caught before DuckDB.

Usage in the Kaggle notebook (before `from features import build_features`):

    import _verify_features_source as vfs
    vfs.check()
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


_BAD_FRAGMENTS = [
    "CREATE TABLE batter_cat_shifted\n                         WHEN pitch_type",
    "CREATE TABLE batter_cat_shifted WHEN pitch_type",
]


def _git(cwd: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True
    )
    return r.stdout.strip() or ""


def check(features_path: Path | None = None) -> None:
    if features_path is None:
        # Assume we are inside backend/ already (Kaggle sys.path injection).
        features_path = Path(__file__).resolve().parent / "features.py"

    repo_dir = features_path.parent.parent
    print("=== REPOSITORY DIAGNOSTIC ===", flush=True)
    print("cwd:", os.getcwd(), flush=True)
    print("repo dir (inferred):", repo_dir, flush=True)
    print("git remote -v:\n" + _git(repo_dir, "remote", "-v"), flush=True)
    print("git rev-parse HEAD:", _git(repo_dir, "rev-parse", "HEAD"), flush=True)
    print("git status --short:\n" + _git(repo_dir, "status", "--short"), flush=True)
    print("git log -1 --oneline:", _git(repo_dir, "log", "-1", "--oneline"), flush=True)

    if not features_path.exists():
        raise SystemExit(f"features.py not found at: {features_path}")

    print("features module path:", features_path, flush=True)

    source = features_path.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines()
    print("\n--- source around once-failing location (1635-1685) ---", flush=True)
    for i in range(1635, 1685):
        if i < len(lines):
            print(f"{i+1}: {lines[i]}", flush=True)

    bad = [f for f in _BAD_FRAGMENTS if f in source]
    if bad:
        raise SystemExit(
            "STALE/CORRUPT features.py detected.\n"
            "The imported features.py still contains the duplicated/corrupted\n"
            "arsenal block. Do NOT execute build_features() against it.\n\n"
            "Detected fragment(s):\n"
            + "\n".join(f"  {repr(f)}" for f in bad)
            + "\n\n"
            f"File: {features_path}\n"
            "Refresh the repo checkout to a commit whose features.py has a single\n"
            "clean arsenal construction path (batter_cat_game2 -> batter_cat_game\n"
            "-> batter_cat_shifted -> batter_cat_rolling -> batter_cat_league,\n"
            "arsenal_usage_game -> ... -> arsenal_usage_league,\n"
            "arsenal_lineup -> arsenal_side).\n"
        )

    print("\nOK: imported features.py does NOT contain the malformed WHEN fragment.", flush=True)
    print("OK: single clean arsenal construction path present.", flush=True)
    sys.exit(0)
