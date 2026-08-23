
import os

def _load_github_token() -> str:
    """Load GitHub token from env or Colab Secrets, never crashing.

    Order:
      1. GITHUB_TOKEN / MY_GITHUB_TOKEN environment variable
         (works everywhere, incl. `python master_pipeline.py` subprocess)
      2. Colab userdata secret 'MY_GITHUB_TOKEN'
         (only available inside a live notebook kernel — guarded so a
         subprocess run doesn't crash with AttributeError)
    Returns "" when unavailable; GitHub sync is then skipped.
    """
    env_tok = (os.environ.get("GITHUB_TOKEN") or os.environ.get("MY_GITHUB_TOKEN") or "").strip()
    if env_tok:
        print("🔑 GitHub token loaded from environment")
        return env_tok
    try:
        from google.colab import userdata
        tok = userdata.get("MY_GITHUB_TOKEN").strip()
        print("🔑 GitHub token loaded from Colab Secrets")
        return tok
    except Exception:
        print("⚠️  No GitHub token (set Colab Secret 'MY_GITHUB_TOKEN' or env GITHUB_TOKEN)")
        print("    → GitHub sync will be skipped")
        return ""

# Safely fetch the token (env var first, then Colab Secrets)
token = _load_github_token()

# Run dates resolve in order:
#   1. Environment variables MLB_START_DATE / MLB_END_DATE (set these in the
#      Colab cell to override a single run without editing this file)
#   2. Repo defaults below — end_date defaults to TODAY so daily runs never
#      go stale and never need a commit just to move the window forward.
#
# Historical re-pull: set MLB_FULL_REPULL=1 to discard the cached
# pitches.parquet and re-download the ENTIRE window cleanly (e.g. after a
# schema/vendor change). Without it, resume logic only tops up missing days.
def _env_date(key: str, fallback: str) -> str:
    val = os.environ.get(key, "").strip()
    return val or fallback

CONFIG = {
    "start_date": _env_date("MLB_START_DATE", "2025-01-01"),
    "end_date":   _env_date("MLB_END_DATE", __import__("datetime").date.today().strftime("%Y-%m-%d")),
    "github_username": "andrewkemmer",
    "github_repo":     "sports_prediction_model",
    "github_branch":   "main",
    "github_token":    token,
    "git_email":       "andrew.kemmer@gmail.com",
    "git_name":        "andrewkemmer",
    "output_dir":      "/content/mlb_clean_data",
    "data_subdir":     "data",
    "statcast_chunk_days": 60,
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
import logging
logging.basicConfig(level=logging.INFO, format="  %(levelname)s %(message)s")
_banner("PHASE 1", "Statcast Data Ingestion")
from ingestion import pull_statcast

start = datetime.strptime(CONFIG["start_date"], "%Y-%m-%d").date()
end = datetime.strptime(CONFIG["end_date"], "%Y-%m-%d").date()
out_dir = Path(CONFIG.get("output_dir", "/content/mlb_clean_data"))
out_dir.mkdir(parents=True, exist_ok=True)
pitches_path = out_dir / "pitches.parquet"

print(f"📅 {start} → {end}")
full_repull = os.environ.get("MLB_FULL_REPULL", "").strip().lower() in ("1", "true", "yes")
if full_repull:
    print("  ♻️  MLB_FULL_REPULL set — discarding cache and re-pulling full history")
pull_statcast(
    start_date=start, end_date=end, out_path=pitches_path,
    chunk_days=CONFIG.get("statcast_chunk_days", 7),
    pause_sec=CONFIG.get("statcast_pause_sec", 2),
    resume=not full_repull,
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

# ── Save features BEFORE training (Phase 4 needs the CSV) ────────────────
_banner("PHASE 3.5", "Save Features")
# NOTE: no fillna here — missing observations ship as true NULLs. Tree models
# handle NaN natively and zero/median fills fabricated signal that poisoned
# PSI drift stats.
csv_path = out_dir / "game_level_features.csv"
parquet_path = out_dir / "pbp_level_features.parquet"
game_df.to_csv(csv_path, index=False)
pbp_df.to_parquet(parquet_path, index=False, compression="snappy")
print(f"  📄 CSV: {csv_path.stat().st_size/1e6:.1f} MB")
print(f"  📄 Parquet: {parquet_path.stat().st_size/1e6:.1f} MB")

# ── Phase 4: Training + Prediction ──────────────────────────────────────────
_banner("PHASE 4", "Training + Prediction")
try:
    from pipeline import run_daily_pipeline
    from data_ingestion import load_game_features

    # Always load via load_game_features — it computes ELO, win_pct,
    # run_diff and maps columns to training.py's FEATURE_COLS format.
    train_games = load_game_features(csv_path)
    print(f"  📋 Training data: {train_games.shape[0]} games, {train_games.shape[1]} features")
    key_feats = ["home_elo", "home_win_pct", "sp_era_30g_home", "woba_30g_home",
                 "bullpen_whip_10g_home", "rest_days_home"]
    cov = ", ".join(
        f"{c}:{train_games[c].notna().mean()*100:.0f}%"
        for c in key_feats if c in train_games.columns
    )
    print(f"  📊 Feature coverage: {cov}")

    target = end  # predict the last date in the range
    summary = run_daily_pipeline(
        target_date=target,
        real=True,
        skip_sync=True,
        force_retrain=True,
        games=train_games,
        pbp_df=pbp_df,  # maps ESPN probable-pitcher names to rolling stat lines
        min_train_days=30,  # warm-up: skip folds trained on < 30 days (~350 games)
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

# ── Phase 5: GitHub Sync ────────────────────────────────────────────────────
_banner("PHASE 5", "GitHub Sync")
token = token or CONFIG.get("github_token", "")
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
        # Wipe old artifacts so only this run's files are pushed — prevents
        # stale SHAP/monitor/calibration files from piling up indefinitely.
        for old in data_delivery_dir.iterdir():
            if old.is_file():
                old.unlink()
        staged = []
        seen = set()

        def _stage(src: Path, rel: str) -> None:
            if rel not in seen:
                seen.add(rel)
                shutil.copy2(src, data_delivery_dir / Path(rel).name)
                staged.append(rel)

        # Sync game-level features CSV (dashboard uses it for final scores).
        # pbp_level_features.parquet is NOT synced: ~7.6 MB per run and
        # nothing in the dashboard reads it. It stays in /content/mlb_clean_data.
        if csv_path.exists():
            _stage(csv_path, f"mlb-bet-predictor/data_delivery/{csv_path.name}")
        # Sync training artifacts (pipeline saves to data_delivery/ in CWD)
        data_delivery_local = Path.cwd() / "data_delivery"
        if data_delivery_local.exists():
            for artifact in data_delivery_local.glob("*"):
                if artifact.is_file():
                    _stage(artifact, f"mlb-bet-predictor/data_delivery/{artifact.name}")
        print(f"  📋 Staging {len(staged)} files:")
        for s in staged:
            print(f"    {s}")
        repo.index.add(staged)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        repo.index.commit(f"Update MLB features + predictions: {ts}")
        repo.remote("origin").push(CONFIG["github_branch"])
        print(f"  ✅ Pushed {len(staged)} files")
        shutil.rmtree(sync_dir, ignore_errors=True)
    except Exception as e:
        print(f"  ❌ {e}")

_banner("DONE ✅")
print(f"  Games: {game_df.shape[0]}  |  Pitches: {pbp_df.shape[0]:,}  |  Features: {game_df.shape[1]+pbp_df.shape[1]}")
print(f"  Output: {out_dir}")
