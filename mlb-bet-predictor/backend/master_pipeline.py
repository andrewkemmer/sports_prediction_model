
import os

CONFIG = {
    "start_date": "2025-04-01",
    "end_date":   "2026-08-20",
    "github_username": "andrewkemmer",
    "github_repo":     "sports_prediction_model",
    "github_branch":   "main",
    "github_token":    "",
    "git_email":       "",
    "git_name":        "",
    "output_dir":      "/content/mlb_clean_data",
    "data_subdir":     "data",
    "statcast_chunk_days": 7,
    "statcast_pause_sec":  2,
}

import warnings
warnings.filterwarnings("ignore")
import os, sys, subprocess, shutil, gc
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
pd.set_option("mode.chained_assignment", None)

def _banner(phase, msg=""):
    print(f"\n{'━'*70}\n  {phase} — {msg}\n{'━'*70}\n")

def _run(cmd, check=True, cwd="/content"):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if check and r.returncode != 0:
        print(f"  ⚠️  {cmd}\n      {r.stderr[:300]}")
    return r

# CRITICAL: escape to a known-good directory FIRST, before anything else.
# If a previous run deleted the cwd, every subprocess will fail with getcwd().
try:
    os.chdir("/content")
except OSError:
    pass  # already there or doesn't exist yet

_banner("PHASE 0", "Environment Setup")
print("📦 Installing dependencies...")
_run("pip install -q pandas numpy scikit-learn xgboost lightgbm shap joblib gitpython pybaseball requests tqdm pyarrow duckdb", check=False)
print("  ✅ Done")

# Clone fresh
repo_dir = Path(f"/content/{CONFIG['github_repo']}")
if repo_dir.exists():
    print("  🔄 Removing old clone...")
    shutil.rmtree(repo_dir, ignore_errors=True)
print(f"📥 Cloning {CONFIG['github_repo']}...")
_run(f"git clone -q https://github.com/{CONFIG['github_username']}/{CONFIG['github_repo']}.git /content/{CONFIG['github_repo']}")
sys.path.insert(0, str(repo_dir / "mlb-bet-predictor" / "backend"))
os.chdir(str(repo_dir / "mlb-bet-predictor"))
print(f"  📁 {os.getcwd()}")

for mod in list(sys.modules.keys()):
    if any(x in mod for x in ['ingestion', 'features', 'pipeline', 'training', 'data_ingestion', 'statcast', 'duckdb']):
        del sys.modules[mod]

# ── Phase 1: Ingestion ──────────────────────────────────────────────────────
_banner("PHASE 1", "Statcast Data Ingestion")
from ingestion import pull_statcast

start = datetime.strptime(CONFIG["start_date"], "%Y-%m-%d").date()
end = datetime.strptime(CONFIG["end_date"], "%Y-%m-%d").date()
out_dir = Path(CONFIG.get("output_dir", "/content/mlb_clean_data"))
out_dir.mkdir(parents=True, exist_ok=True)
pitches_path = out_dir / "pitches.parquet"

print(f"📅 {start} → {end}")
pull_statcast(
    start_date=start, end_date=end, out_path=pitches_path,
    chunk_days=CONFIG.get("statcast_chunk_days", 7),
    pause_sec=CONFIG.get("statcast_pause_sec", 2),
    resume=True,
)
print(f"  ✅ Raw pitches: {pitches_path}")

# ── Phase 2-3: Feature Engineering ──────────────────────────────────────────
_banner("PHASE 2-3", "DuckDB Feature Engineering (pure SQL)")
from features import build_features

game_df, pbp_df = build_features(
    pitches_path=pitches_path,
    output_dir=out_dir,
    validate=True,
)
print(f"  ✅ Game: {game_df.shape}")
print(f"  ✅ PBP:  {pbp_df.shape}")
gc.collect()

# ── Phase 4: Training + Prediction ──────────────────────────────────────────
_banner("PHASE 4", "Training + Prediction")
try:
    from pipeline import run_daily_pipeline
    target = end  # predict the last date in the range
    summary = run_daily_pipeline(
        target_date=target,
        real=True,
        skip_sync=True,
        force_retrain=True,
    )
    print(f"  📊 Status: {summary['status']}")
    if summary.get("metrics"):
        import json
        print(f"  📈 Metrics: {json.dumps(summary['metrics'], indent=2)}")
    if summary.get("artifacts"):
        print(f"  📁 Artifacts: {len(summary['artifacts'])} files")
        for a in summary['artifacts']:
            print(f"    {a}")
    if summary.get("errors"):
        print(f"  ❌ Errors: {summary['errors']}")
except Exception as e:
    print(f"  ❌ Training failed: {e}")

# ── Phase 5: Save Outputs ──────────────────────────────────────────────────
_banner("PHASE 5", "Save Outputs")
game_df.fillna(game_df.median(numeric_only=True), inplace=True)
pbp_df.fillna(pbp_df.median(numeric_only=True), inplace=True)
csv_path = out_dir / "game_level_features.csv"
parquet_path = out_dir / "pbp_level_features.parquet"
game_df.to_csv(csv_path, index=False)
pbp_df.to_parquet(parquet_path, index=False, compression="snappy")
print(f"  📄 CSV: {csv_path.stat().st_size/1e6:.1f} MB")
print(f"  📄 Parquet: {parquet_path.stat().st_size/1e6:.1f} MB")

# ── Phase 6: GitHub Sync ────────────────────────────────────────────────────
_banner("PHASE 6", "GitHub Sync")
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
        data_delivery_dir = sync_dir / "mlb-bet-predictor" / "data_delivery"; data_delivery_dir.mkdir(exist_ok=True)
        staged = []
        # Sync feature files
        for f in [csv_path, parquet_path]:
            if f.exists():
                shutil.copy2(f, data_delivery_dir / f.name)
                staged.append(f"mlb-bet-predictor/data_delivery/{f.name}")
        # Sync training artifacts (pipeline saves to data_delivery/ in CWD)
        data_delivery_local = Path.cwd() / "data_delivery"
        if data_delivery_local.exists():
            for artifact in data_delivery_local.glob("*"):
                if artifact.is_file():
                    shutil.copy2(artifact, data_delivery_dir / artifact.name)
                    staged.append(f"mlb-bet-predictor/data_delivery/{artifact.name}")
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
print(f"  Output: {out_dir}")
