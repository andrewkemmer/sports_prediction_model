"""
MLB Master Pipeline — Single-Script End-to-End for Google Colab.

Paste this entire script into a Colab notebook, configure the credentials
at the top, and hit "Run All". It handles:

  1. Dependency installation
  2. Statcast data ingestion (auto-chunked into 6-day batches)
  3. Multi-tier feature engineering (PIT-compliant)
  4. Dual-level output: game_df + pbp_df
  5. Null handling & compression (CSV + snappy Parquet)
  6. Automated GitHub sync

Usage:
    1. Open Colab → New notebook
    2. Paste this entire script into one cell
    3. Fill in the CONFIG block below
    4. Runtime → Run all
"""
from __future__ import annotations

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — Edit these values before running                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

CONFIG = {
    # ── Date Range ────────────────────────────────────────────────────────────
    "start_date": "2025-04-01",       # Season start
    "end_date":   "2025-08-01",       # Season end (or today)

    # ── GitHub Credentials ────────────────────────────────────────────────────
    "github_username": "andrewkemmer",
    "github_repo":     "sports_prediction_model",
    "github_branch":   "main",
    "github_token":    "",            # Personal Access Token (repo scope)
    "git_email":       "",            # For commit authorship
    "git_name":        "",            # For commit authorship

    # ── Output Paths ──────────────────────────────────────────────────────────
    "output_dir":      "/content/mlb_clean_data",   # Local output
    "data_subdir":     "data",                       # Subdir inside repo for push

    # ── Pipeline Settings ─────────────────────────────────────────────────────
    "statcast_chunk_days": 6,         # Days per API call (6 = safe, avoids timeout)
    "statcast_pause_sec":  2,         # Seconds between API calls
    "checkpoint_dir":      None,      # Set to "/content/drive/MyDrive/mlb_ckpt" for Drive
}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 0: SETUP                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", message=".*does not support indexing.*")

import os
import sys
import subprocess
import time
from datetime import date, datetime, timedelta
from pathlib import Path

def _banner(phase: str, msg: str = ""):
    """Print a clean phase banner."""
    print(f"\n{'━' * 70}")
    print(f"  {phase}" + (f" — {msg}" if msg else ""))
    print(f"{'━' * 70}\n")

def _run(cmd: str, check: bool = True):
    """Run a shell command and print output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  ⚠️  Command failed: {cmd}")
        print(f"      {result.stderr[:500]}")
    return result


_banner("PHASE 0", "Environment Setup")

# Install dependencies (idempotent — skips already-installed)
print("📦 Installing dependencies...")
_deps = (
    "pandas numpy scikit-learn xgboost lightgbm shap joblib "
    "gitpython pybaseball requests tqdm pyarrow"
)
_run(f"pip install -q {_deps}", check=False)
print("  ✅ Dependencies installed")

# Clone repo (if not already present)
_repo_dir = Path(f"/content/{CONFIG['github_repo']}")
if not _repo_dir.exists():
    print(f"📥 Cloning {CONFIG['github_repo']}...")
    _run(f"git clone -q https://github.com/{CONFIG['github_username']}/{CONFIG['github_repo']}.git /content/{CONFIG['github_repo']}")
else:
    print(f"  ℹ️  Repo already cloned at {_repo_dir}")

# Add backend to Python path
sys.path.insert(0, str(_repo_dir / "mlb-bet-predictor" / "backend"))
os.chdir(str(_repo_dir / "mlb-bet-predictor"))
print(f"  📁 Working directory: {os.getcwd()}")

# Suppress pandas warnings inside our modules
import pandas as pd
import numpy as np
pd.set_option("mode.chained_assignment", None)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 1: DATA INGESTION                                                  ║
# ║  Pull raw Statcast pitch-by-pitch data with auto-chunking                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_banner("PHASE 1", "Statcast Data Ingestion")

from tqdm.auto import tqdm
from statcast_pipeline import (
    pull_statcast_data,
    build_game_level_features,
    build_pbp_level_features,
    validate_datasets,
)

start_date = datetime.strptime(CONFIG["start_date"], "%Y-%m-%d").date()
end_date = datetime.strptime(CONFIG["end_date"], "%Y-%m-%d").date()

print(f"📅 Date range: {start_date} → {end_date}")
print(f"   Chunk size: {CONFIG['statcast_chunk_days']} days")
print(f"   Estimated API calls: {((end_date - start_date).days // CONFIG['statcast_chunk_days']) + 1}")
print()

# Pull data with progress tracking
pitches = pull_statcast_data(
    start_date=start_date,
    end_date=end_date,
    checkpoint_dir=CONFIG.get("checkpoint_dir"),
    resume=True,
)

if pitches.empty:
    print("\n❌ FATAL: No Statcast data was pulled. Check your date range and network.")
    print("   Tip: Try a smaller date range first (e.g., 1 month).")
    sys.exit(1)

print(f"\n✅ Phase 1 complete: {len(pitches):,} pitches across "
      f"{pitches['game_pk'].nunique()} games, "
      f"{pitches['game_date'].nunique()} dates")
print(f"   Memory: {pitches.memory_usage(deep=True).sum() / 1e6:.1f} MB")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 2: FEATURE ENGINEERING                                              ║
# ║  Multi-tier PIT-compliant features from raw Statcast stream               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_banner("PHASE 2", "Feature Engineering")

# 2a. Game-level features
print("🏟️  Building game-level features...")
game_df = build_game_level_features(pitches)
print(f"   ✅ Game-level: {game_df.shape[0]} rows × {game_df.shape[1]} columns")

# 2b. Play-by-play features
print("⚾ Building play-by-play features...")
pbp_df = build_pbp_level_features(pitches, game_df)
print(f"   ✅ PBP-level: {pbp_df.shape[0]:,} rows × {pbp_df.shape[1]} columns")

# 2c. Feature tier summary
print("\n📊 Feature Tiers:")
_tiers = {
    "Pitcher Rolling (ERA/K9/WHIP/FIP/xwOBA)": [c for c in game_df.columns if c.startswith("sp_")],
    "Team Offense Rolling (wOBA/ISO/K/BB)": [c for c in game_df.columns if c.startswith("team_") and "woba" in c or "iso" in c or "k_rate" in c or "bb_rate" in c],
    "Bullpen Rolling (WHIP/ERA)": [c for c in game_df.columns if c.startswith("bullpen_")],
    "Market Lines (ML/Total/RunLine)": [c for c in game_df.columns if c.startswith("moneyline_") or c in ("total_line", "run_line_home", "juice")],
    "Context (Rest/Venue/Starter)": [c for c in game_df.columns if "rest" in c or "venue" in c or "starter" in c],
    "PBP Situational (Inning/Score/Bases)": [c for c in pbp_df.columns if c in ("inning", "score_diff", "bases_loaded", "runners_in_scoring_position", "is_risp", "ab_pitch_count", "times_through_order")],
    "PBP Contact Quality (EV/LA/Barrel)": [c for c in pbp_df.columns if c in ("exit_velocity", "launch_angle", "is_barrel", "is_hard_hit")],
}
for tier_name, cols in _tiers.items():
    status = "✅" if cols else "⚠️"
    print(f"   {status} {tier_name}: {len(cols)} features")

print(f"\n✅ Phase 2 complete: {len(game_df.columns) + len(pbp_df.columns)} total features")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 3: VALIDATION                                                      ║
# ║  Zero leakage check, null audit, shape/memory report                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_banner("PHASE 3", "Validation & Quality Check")

validation = validate_datasets(game_df, pbp_df, pitches)

if validation["status"] != "PASS":
    print("\n⚠️  Validation found issues. Review the report above.")
    print("   Continuing with available data...")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 4: NULL HANDLING & COMPRESSION                                     ║
# ║  Clean nulls, save CSV (game) + snappy Parquet (pbp)                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_banner("PHASE 4", "Null Handling & Compression")

output_dir = Path(CONFIG["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

# 4a. Handle nulls in game_df
print("🔧 Cleaning game_df nulls...")
null_before = game_df.isnull().sum().sum()
for col in game_df.columns:
    if game_df[col].dtype in [np.float64, np.float32]:
        median_val = game_df[col].median()
        if pd.notna(median_val):
            game_df[col] = game_df[col].fillna(median_val)
    elif game_df[col].dtype == object:
        game_df[col] = game_df[col].fillna("Unknown")
null_after = game_df.isnull().sum().sum()
print(f"   Nulls: {null_before:,} → {null_after:,}")

# 4b. Handle nulls in pbp_df
print("🔧 Cleaning pbp_df nulls...")
null_before = pbp_df.isnull().sum().sum()
for col in pbp_df.columns:
    if pbp_df[col].dtype in [np.float64, np.float32]:
        median_val = pbp_df[col].median()
        if pd.notna(median_val):
            pbp_df[col] = pbp_df[col].fillna(median_val)
    elif pbp_df[col].dtype == object:
        pbp_df[col] = pbp_df[col].fillna("Unknown")
null_after = pbp_df.isnull().sum().sum()
print(f"   Nulls: {null_before:,} → {null_after:,}")

# 4c. Save game_df as CSV
csv_path = output_dir / "game_level_features.csv"
game_df.to_csv(csv_path, index=False)
csv_size = csv_path.stat().st_size / 1e6
print(f"\n📄 Saved game_df → {csv_path.name} ({csv_size:.1f} MB)")

# 4d. Save pbp_df as snappy Parquet (high compression)
parquet_path = output_dir / "pbp_level_features.parquet"
pbp_df.to_parquet(parquet_path, index=False, compression="snappy")
parquet_size = parquet_path.stat().st_size / 1e6
csv_equivalent = pbp_df.memory_usage(deep=True).sum() / 1e6
compression_ratio = csv_equivalent / parquet_size if parquet_size > 0 else 1
print(f"📄 Saved pbp_df → {parquet_path.name} ({parquet_size:.1f} MB, {compression_ratio:.1f}× compression)")

# 4e. Also save a CSV version of pbp_df for GitHub (if under 100MB)
pbp_csv_path = output_dir / "pbp_level_features.csv"
pbp_df.to_csv(pbp_csv_path, index=False)
pbp_csv_size = pbp_csv_path.stat().st_size / 1e6
print(f"📄 Saved pbp_df CSV → {pbp_csv_path.name} ({pbp_csv_size:.1f} MB)")

if pbp_csv_size > 100:
    print("   ⚠️  PBP CSV exceeds 100MB — GitHub push will use Parquet only")
    # Remove CSV to save space, keep Parquet
    pbp_csv_path.unlink(missing_ok=True)

# 4f. Summary
print(f"\n✅ Phase 4 complete:")
print(f"   game_df:  {game_df.shape[0]} rows × {game_df.shape[1]} cols → CSV ({csv_size:.1f} MB)")
print(f"   pbp_df:   {pbp_df.shape[0]:,} rows × {pbp_df.shape[1]} cols → Parquet ({parquet_size:.1f} MB)")


# ╔══════════════════════════════════════════════════════════════════════════════╝
# ║  PHASE 5: AUTOMATED GITHUB SYNC                                           ║
# ║  Clone → Copy → Commit → Push                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_banner("PHASE 5", "GitHub Sync")

_github_token = CONFIG.get("github_token", "")
_github_user  = CONFIG.get("github_username", "")
_github_repo  = CONFIG.get("github_repo", "")
_github_branch = CONFIG.get("github_branch", "main")
_git_email    = CONFIG.get("git_email", "")
_git_name     = CONFIG.get("git_name", "")

if not _github_token:
    print("⏭️  Skipping GitHub sync — no token configured in CONFIG.")
    print("   To enable: set 'github_token' in the CONFIG block at the top.")
else:
    print(f"🔐 Pushing to: {_github_user}/{_github_repo} ({_github_branch})")

    _sync_dir = Path("/content/mlb_sync_tmp")
    _sync_dir.mkdir(exist_ok=True)

    try:
        import git

        # Build auth URL
        auth_url = f"https://{_github_token}@github.com/{_github_user}/{_github_repo}.git"

        # Clone (shallow)
        print("📥 Cloning repo...")
        repo = git.Repo.clone_from(auth_url, str(_sync_dir), branch=_github_branch, depth=1)

        # Configure git identity
        if _git_email:
            repo.config_writer().set_value("user", "email", _git_email).release()
        if _git_name:
            repo.config_writer().set_value("user", "name", _git_name).release()

        # Create data subdirectory
        data_dir = Path(_sync_dir) / CONFIG["data_subdir"]
        data_dir.mkdir(exist_ok=True)

        # Copy files
        staged = []
        for src_file in [csv_path, parquet_path]:
            if src_file.exists():
                dest = data_dir / src_file.name
                import shutil
                shutil.copy2(src_file, dest)
                staged.append(f"{CONFIG['data_subdir']}/{src_file.name}")

        # Copy CSV of pbp if it exists
        if pbp_csv_path.exists():
            dest = data_dir / pbp_csv_path.name
            shutil.copy2(pbp_csv_path, dest)
            staged.append(f"{CONFIG['data_subdir']}/{pbp_csv_path.name}")

        # Check if anything changed
        repo.index.add(staged)
        diff = repo.index.diff("HEAD")

        if not diff:
            print("ℹ️  No data changes detected — skipping push.")
        else:
            # Commit
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_msg = (
                f"Update MLB features: {timestamp}\n\n"
                f"Files: {', '.join(staged)}\n"
                f"Date range: {CONFIG['start_date']} to {CONFIG['end_date']}\n"
                f"Game-level: {game_df.shape[0]} rows, {game_df.shape[1]} cols\n"
                f"PBP-level: {pbp_df.shape[0]:,} rows, {pbp_df.shape[1]} cols"
            )
            commit = repo.index.commit(commit_msg)
            print(f"📝 Committed: {commit.hexsha[:8]}")

            # Push
            print("🚀 Pushing...")
            origin = repo.remote("origin")
            origin.push(_github_branch)
            print(f"✅ Pushed to {_github_user}/{_github_repo}@{_github_branch}")

    except Exception as e:
        print(f"❌ GitHub sync failed: {e}")
        print("   Check your token, repo name, and network connection.")

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(_sync_dir, ignore_errors=True)
        print("  🧹 Cleaned up temp directory")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  COMPLETE                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_banner("PIPELINE COMPLETE ✅")

print(f"""
Summary:
  📅 Date range:     {CONFIG['start_date']} → {CONFIG['end_date']}
  📊 Games:          {game_df.shape[0]}
  ⚾ Pitches:        {pbp_df.shape[0]:,}
  🏗️  Features:       {len(game_df.columns)} (game) + {len(pbp_df.columns)} (pbp)
  📄 Output:         {csv_path} ({csv_size:.1f} MB)
                     {parquet_path} ({parquet_size:.1f} MB)
  🔒 Validation:     {validation['status']}
  🚀 GitHub:         {"Pushed" if _github_token else "Skipped (no token)"}

Files saved to: {output_dir}
""")
