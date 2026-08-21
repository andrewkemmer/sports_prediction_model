"""
MLB Master Pipeline — Single-Script End-to-End for Google Colab.
"""
from __future__ import annotations

CONFIG = {
    "start_date": "2026-08-01",
    "end_date":   "2026-08-20",
    "github_username": "andrewkemmer",
    "github_repo":     "sports_prediction_model",
    "github_branch":   "main",
    "github_token":    "",
    "git_email":       "",
    "git_name":        "",
    "output_dir":      "/content/mlb_clean_data",
    "data_subdir":     "data",
    "statcast_chunk_days": 6,
    "statcast_pause_sec":  2,
    "checkpoint_dir":      None,
}

import warnings
warnings.filterwarnings("ignore")
import os, sys, subprocess, shutil, time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
pd.set_option("mode.chained_assignment", None)

def _banner(phase, msg=""):
    print(f"\n{'━'*70}\n  {phase} — {msg}\n{'━'*70}\n")

def _run(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"  ⚠️  {cmd}\n      {r.stderr[:300]}")
    return r

_banner("PHASE 0", "Environment Setup")
print("📦 Installing dependencies...")
_run("pip install -q pandas numpy scikit-learn xgboost lightgbm shap joblib gitpython pybaseball requests tqdm pyarrow", check=False)
print("  ✅ Done")

# Always clone fresh
repo_dir = Path(f"/content/{CONFIG['github_repo']}")
if repo_dir.exists():
    print("  🔄 Removing old clone...")
    shutil.rmtree(repo_dir, ignore_errors=True)
print(f"📥 Cloning {CONFIG['github_repo']}...")
_run(f"git clone -q https://github.com/{CONFIG['github_username']}/{CONFIG['github_repo']}.git /content/{CONFIG['github_repo']}")
sys.path.insert(0, str(repo_dir / "mlb-bet-predictor" / "backend"))
os.chdir(str(repo_dir / "mlb-bet-predictor"))
print(f"  📁 {os.getcwd()}")

# Clear any stale Python module caches
for mod in list(sys.modules.keys()):
    if any(x in mod for x in ['statcast', 'pipeline', 'data_ingestion', 'config', 'training', 'explainability']):
        del sys.modules[mod]

_banner("PHASE 1-3", "Chunked Pipeline (pull + features + validate)")
from statcast_pipeline import run_statcast_pipeline
import gc

start = datetime.strptime(CONFIG["start_date"], "%Y-%m-%d").date()
end = datetime.strptime(CONFIG["end_date"], "%Y-%m-%d").date()
print(f"📅 {start} → {end}")
print("  Each feature tier computed in a separate pass (~40 MB peak)")

out_dir = CONFIG.get("output_dir", "/content/mlb_clean_data")
game_df, pbp_df = run_statcast_pipeline(
    start_date=start,
    end_date=end,
    checkpoint_dir=out_dir,
    validate=True,
)

print(f"  ✅ Game: {game_df.shape}")
print(f"  ✅ PBP:  {pbp_df.shape}")
gc.collect()

_banner("PHASE 4", "Compression")
_banner("PHASE 4", "Compression")
output_dir = Path(CONFIG["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
game_df.fillna(game_df.median(numeric_only=True), inplace=True)
pbp_df.fillna(pbp_df.median(numeric_only=True), inplace=True)
csv_path = output_dir / "game_level_features.csv"; game_df.to_csv(csv_path, index=False)
parquet_path = output_dir / "pbp_level_features.parquet"; pbp_df.to_parquet(parquet_path, index=False, compression="snappy")
print(f"  📄 CSV: {csv_path.stat().st_size/1e6:.1f} MB")
print(f"  📄 Parquet: {parquet_path.stat().st_size/1e6:.1f} MB")

_banner("PHASE 5", "GitHub Sync")
token = CONFIG.get("github_token", "")
if not token:
    print("  ⏭️  No token — skipping push")
else:
    try:
        import git
        auth_url = f"https://{token}@github.com/{CONFIG['github_username']}/{CONFIG['github_repo']}.git"
        sync_dir = Path("/content/mlb_sync_tmp")
        repo = git.Repo.clone_from(auth_url, str(sync_dir), branch=CONFIG["github_branch"], depth=1)
        if CONFIG["git_email"]: repo.config_writer().set_value("user","email",CONFIG["git_email"]).release()
        if CONFIG["git_name"]: repo.config_writer().set_value("user","name",CONFIG["git_name"]).release()
        data_dir = sync_dir / CONFIG["data_subdir"]; data_dir.mkdir(exist_ok=True)
        staged = []
        for f in [csv_path, parquet_path]:
            if f.exists(): shutil.copy2(f, data_dir / f.name); staged.append(f"{CONFIG['data_subdir']}/{f.name}")
        repo.index.add(staged)
        if repo.index.diff("HEAD"):
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            repo.index.commit(f"Update MLB features: {ts}")
            repo.remote("origin").push(CONFIG["github_branch"])
            print(f"  ✅ Pushed {len(staged)} files")
        else:
            print("  ℹ️  No changes")
        shutil.rmtree(sync_dir, ignore_errors=True)
    except Exception as e:
        print(f"  ❌ {e}")

_banner("DONE ✅")
print(f"  Games: {game_df.shape[0]}  |  Pitches: {pbp_df.shape[0]:,}  |  Features: {game_df.shape[1]+pbp_df.shape[1]}")
