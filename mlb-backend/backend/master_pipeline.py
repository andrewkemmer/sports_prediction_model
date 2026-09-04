
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

# Multi-sport restructure (Phase A): repo-relative directory holding this
# sport's backend + data_delivery. Needed HERE (not from config) because
# the sys.path/os.chdir lines below run before backend/ is importable.
# Mirrored in backend/config.py (SPORT_DIR_NAME) and frontend/sports_config.py
# (repo_subdir) — Phase C renames the directory to mlb-backend/ and flips all
# three at once.
SPORT_DIR_NAME = "mlb-backend"

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
# shap/xgboost are version-guarded: shap's XGBoost loader needs our
# base_score decode shim for xgboost>=2 UBJSON dumps (verified on
# xgboost 3.2 + shap 0.49; see backend/explainability.py). Upper bounds
# at the next major prevent an untested pairing from silently shipping.
_run('pip install -q pandas numpy scikit-learn "xgboost>=1.7,<4" lightgbm optuna "shap>=0.45,<0.51" joblib gitpython pybaseball requests tqdm pyarrow duckdb', check=False)
print("  ✅ Done")

# Clone fresh
repo_dir = Path(f"/content/{CONFIG['github_repo']}")
if repo_dir.exists():
    print("  🔄 Removing old clone...")
    shutil.rmtree(repo_dir, ignore_errors=True)
print(f"📥 Cloning {CONFIG['github_repo']}...")
_run(f"git clone -q https://github.com/{CONFIG['github_username']}/{CONFIG['github_repo']}.git /content/{CONFIG['github_repo']}")
sys.path.insert(0, str(repo_dir / SPORT_DIR_NAME / "backend"))
os.chdir(str(repo_dir / SPORT_DIR_NAME))
print(f"  📁 {os.getcwd()}")

# Snapshot the artifacts already in the repo's data_delivery (relative path →
# mtime in ns) BEFORE the pipeline writes anything. Phase 5 must stage only
# files THIS run produced or modified; every other file in that folder is a
# stale file that Phase 6 will delete from GitHub.
_preexisting_delivery: dict[str, int] = {}
_preexisting_dir = Path.cwd() / "data_delivery"
if _preexisting_dir.exists():
    for _p in _preexisting_dir.rglob("*"):
        if _p.is_file():
            _preexisting_delivery[_p.relative_to(_preexisting_dir).as_posix()] = _p.stat().st_mtime_ns

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
# The weather-history cache lives beside pitches.parquet (outside the git
# repo) so Colab's per-run artifact sync never stages it.
getattr(os, "environ").setdefault("MLB_CACHE_DIR", str(out_dir))
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

# The DuckDB export omits Elo/season records and ships the rolling team wOBA
# under team_woba_30g_*; enrich the PIT inputs, then recompute the diff
# features so win_pct_diff / elo_diff / woba_30g_diff ship real values
# (spec features 2, 3, 17) instead of all-NaN columns.
from data_ingestion import enrich_elo_and_records
from features import add_diff_features

game_df = enrich_elo_and_records(game_df, rename_team_woba=True)
game_df = add_diff_features(game_df)

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

# ── Phase 3.6: Defense projection (curated Statcast subset) ─────────────────
# Project the defense-relevant Statcast subset (identity, batted-ball,
# fielders, alignment, WIP outcomes) into data_delivery as
# pbp_defense_<date>.parquet + self-documenting metadata; 2024 backfill
# included. F2/F4 of the defense ablation need this wide cache; the lean
# 8-col pbp cache stays untouched so no current consumer breaks.
from build_pbp_defense import main as _build_pbp_defense
sys.argv = ["build_pbp_defense.py",
            "--source", str(pitches_path),
            "--end", CONFIG["end_date"],
            "--backfill-2024"]
try:
    _build_pbp_defense()
except Exception as e:
    print(f"  ⚠️  Defense projection failed (non-fatal): {e}")

# ── Phase 4: Training + Prediction ──────────────────────────────────────────
_banner("PHASE 4", "Training + Prediction")
try:
    from pipeline import run_daily_pipeline
    from data_ingestion import load_game_features

    # Always load via load_game_features — it computes ELO, win_pct,
    # run_diff and maps columns to training.py's FEATURE_COLS format.
    train_games = load_game_features(csv_path)
    print(f"  📋 Training data: {train_games.shape[0]} games, {train_games.shape[1]} features")
    key_feats = ["home_elo", "home_win_pct", "sp_era_5g_home", "woba_30g_home",
                 "bullpen_whip_10g_home", "rest_days_home"]
    cov = ", ".join(
        f"{c}:{train_games[c].notna().mean()*100:.0f}%"
        for c in key_feats if c in train_games.columns
    )
    print(f"  📊 Feature coverage: {cov}")

    # ── PRE-TRAINING SILENT-DATA INGESTION GUARD ────────────────────────────
    # The 08-28 Statcast chunk failure (IncompleteRead on a core-season chunk,
    # came back EMPTY) dropped ~800 games from the decided frame (6,161 vs the
    # expected 6,960) and the pipeline trained anyway. Never let a degraded
    # frame past this point: verify the canonical expected pitch + decided-game
    # counts BEFORE training, and ABORT loudly if either falls short. This is
    # checked against the good-run baseline, so a silent gap (missing chunk,
    # posting failure) can never ship a quietly-worse model.
    from frames import get_decided_frame
    n_pitches = len(pbp_df)          # one row per raw pitch in pbp_level
    n_decided = len(get_decided_frame(train_games))
    _min_pitches = 2_044_874         # good 6,960-game run shipped ≥ this
    _min_decided = 6_960
    print(f"  🛡️  Ingestion guard: {n_pitches} pitches, {n_decided} decided games "
          f"(expected ≥{_min_pitches} pitches / {_min_decided} decided)")
    if n_pitches < _min_pitches or n_decided < _min_decided:
        raise RuntimeError(
            f"Statcast ingestion looks DEGRADED before training: got "
            f"{n_pitches} pitches / {n_decided} decided games, expected ≥ "
            f"{_min_pitches} / {_min_decided}. A core-season chunk likely came "
            f"back empty (see the ingestion abort above). Refusing to train / "
            f"push on this incomplete frame. Re-run after the data gap is "
            f"filled (or set MLB_FULL_REPULL=1 for a clean re-pull)."
        )

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

# ── Phase 5: GitHub Sync — push this run's NEW files first ─────────────────
_banner("PHASE 5", "GitHub Sync — push new artifacts")
token = token or CONFIG.get("github_token", "")
sync_dir = Path("/content/mlb_sync_tmp")

def _git_push_confirmed(repo, branch: str) -> None:
    """Push and verify the remote accepted the new head (raise on failure)."""
    info = repo.remote("origin").push(branch)
    bad = [p for p in info if p.flags & (p.ERROR | p.REJECTED | p.REMOTE_REJECTED | p.REMOTE_FAILURE)]
    if bad:
        raise RuntimeError(f"Push rejected by remote: {[p.summary for p in bad]}")

def _open_sync_repo(token: str, sync_dir: Path):
    """Open the sync clone (or create it), configuring git identity."""
    import git
    auth_url = f"https://{token}@github.com/{CONFIG['github_username']}/{CONFIG['github_repo']}.git"
    if (sync_dir / ".git").exists():
        repo = git.Repo(str(sync_dir))
    else:
        repo = git.Repo.clone_from(auth_url, str(sync_dir), branch=CONFIG["github_branch"], depth=1)
    if CONFIG["git_email"]: repo.config_writer().set_value("user", "email", CONFIG["git_email"]).release()
    if CONFIG["git_name"]:  repo.config_writer().set_value("user", "name", CONFIG["git_name"]).release()
    return repo

staged: list[str] = []
seen: set[str] = set()

if not token:
    print("  ⏭️  No token — skipping push and cleanup")
else:
    try:
        repo = _open_sync_repo(token, sync_dir)
        data_delivery_dir = sync_dir / SPORT_DIR_NAME / "data_delivery"
        data_delivery_dir.mkdir(parents=True, exist_ok=True)

        def _stage(src: Path, rel: str) -> None:
            if rel in seen:
                return
            seen.add(rel)
            dest = data_delivery_dir / rel[len(f"{SPORT_DIR_NAME}/data_delivery/"):]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            staged.append(rel)

        # Sync game-level features CSV (dashboard uses it for final scores).
        # pbp_level_features.parquet is NOT synced: ~7.6 MB per run and
        # nothing in the dashboard reads it. It stays in /content/mlb_clean_data.
        # Two copies can exist: the PRE-weather Phase-3.5 export in out_dir
        # and the pipeline's post-weather re-export in data_delivery/ (which
        # also carries game_id/start_time columns). Stage whichever is NEWER
        # — staging the stale out_dir copy unconditionally used to overwrite
        # the enriched one every run, so shipped weather features stayed at
        # dome-zeros/nulls even when training saw real values in memory.
        _csv_candidates = [
            p for p in (
                Path.cwd() / "data_delivery" / csv_path.name,
                csv_path,
            ) if p.exists()
        ]
        if _csv_candidates:
            _csv_src = max(_csv_candidates, key=lambda p: p.stat().st_mtime)
            _stage(_csv_src, f"{SPORT_DIR_NAME}/data_delivery/{csv_path.name}")
        # Sync every artifact THIS run regenerated in data_delivery/, including
        # the models/ subdir (trained ensemble joblib the dashboard loads). The
        # fresh clone starts with the repo's old files, so compare mtimes to
        # the pre-run snapshot: files this run didn't touch are stale — they
        # are left out of the push and removed from GitHub by Phase 6.
        data_delivery_local = Path.cwd() / "data_delivery"
        if data_delivery_local.exists():
            for artifact in sorted(data_delivery_local.rglob("*")):
                if artifact.is_file():
                    rel_local = artifact.relative_to(data_delivery_local).as_posix()
                    pre_mtime = _preexisting_delivery.get(rel_local)
                    if pre_mtime is not None and artifact.stat().st_mtime_ns <= pre_mtime:
                        continue  # repo file untouched by this run -> stale
                    _stage(artifact, f"{SPORT_DIR_NAME}/data_delivery/{rel_local}")
        print(f"  📋 Staging {len(staged)} files:")
        for s in staged:
            print(f"    {s}")
        if staged:
            repo.index.add(staged)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            repo.index.commit(f"Update MLB features + predictions: {ts}")
            _git_push_confirmed(repo, CONFIG["github_branch"])
            print(f"  ✅ Pushed {len(staged)} files — confirmed on {CONFIG['github_repo']}@{CONFIG['github_branch']}")
        else:
            print("  ⏭️  Nothing new to push")
    except Exception as e:
        print(f"  ❌ {e}")

# ── Phase 6: Stale artifact cleanup — LAST step, after push confirmed ──────
# data_delivery/ on GitHub must contain ONLY this run's refreshed files.
# Everything the run did NOT regenerate (older SHAP/calibration/monitor/
# power-rankings snapshots, superseded models) is deleted here — strictly
# after the new files were pushed AND confirmed, so a failed push can never
# empty the folder, and the repo keeps no redundant/stale blobs.
#
# BUG FIX (Phase 3.5b): the old logic deleted EVERY tracked file not in
# ``seen``, which nuked persistent assets (statsapi_roof_cache.json,
# model_history.json, models/) and same-day artifacts not regenerated by
# this pipeline path.  Fix: (1) an explicit PROTECTED set that cleanup
# never touches, and (2) date-gating — only delete date-stamped files
# whose date is STRICTLY OLDER than the current run's date.
import re as _re
from datetime import date as _date
from datetime import timedelta as _td, timezone as _tz
import datetime as _dt

# SINGLE SOURCE OF TRUTH: the explicit rolling-retention policy
# (retention_policy.py) — the consumer-audit-backed per-family windows, the
# never-delete markers (masters / records / series readers), and the pure
# keep/stale predicate ``classify_artifact``.  Phase 6 derives EVERY rule
# from it; no family tuples are hard-coded here anymore.  The policy
# deliberately REVERSES the old "committed artifacts are never auto-deleted"
# convention for the ALLOWLISTED dated board-artifact families only — never
# for records, masters, or series readers (see the module docstring and the
# audit record data_delivery/mlb_retention_policy_<framesha>.json).
from retention_policy import (
    classify_artifact,
    artifact_date as _artifact_date,
    family_prefixes as _family_prefixes,
    local_name as _basename,
)

# Current run date in YYYYMMDD for date-gating.
_run_date_compact = CONFIG["end_date"].replace("-", "")  # e.g. "20260824"

# Recent-slate protection window: keep todays_games_* and shap_game_*
# for the current run date AND the 2 prior days so that games have time to
# settle before their card snapshots are pruned.  Other dated artifacts
# ride the 48h retention window below.
_now_utc = _dt.datetime.now(_tz.utc)
_RECENT_DATES = {
    (_now_utc - _td(days=i)).strftime("%Y%m%d")
    for i in range(3)  # today, yesterday, 2 days ago
}

# Board-backed retention (doubleheader regression fix): a dated run-engine /
# predictions artifact is kept for ANY date that still has a tracked
# todays_games_<date>.csv board, so a navigable board is never left without
# the RUN ENGINE columns its cards need. Keep them for as long as the board
# itself is tracked (policy: families with board_supported=True).
_BOARD_BACKED_PREFIXES = _family_prefixes("board_supported")

# Rolling 48-hour retention window (GMT-rollover regression fix): keep EVERY
# dated artifact for the current run date AND the previous GMT day. US games
# end up to ~midnight ET (~05:00 GMT next day), so the previous GMT day is
# never "stale" while those games are live — their card RUN ENGINE block
# depends on run_engine_markets / run_engine_oof / predictions_history from
# that day. The old strict same-day rule (`art_date == _run_date_compact`)
# deleted yesterday's artifacts at the 00:00 GMT rollover even while those
# games were still pre-game, silently dropping the RUN ENGINE block. Threshold
# is timedelta(days=1) — never string equality against today.
_RETENTION_DAYS = 1
_run_date_obj = _date(*(int(x) for x in CONFIG["end_date"].split("-")))
_RETENTION_DATES = {(_run_date_obj - _td(days=i)).strftime("%Y%m%d")
                    for i in range(_RETENTION_DAYS + 1)}  # {today, yesterday}

_banner("PHASE 6", "Stale artifact cleanup (final step)")
if not token:
    print("  ⏭️  No token — skipping cleanup")
elif not staged:
    print("  ⏭️  Nothing was pushed this run — skipping cleanup (can't tell stale from new)")
else:
    try:
        repo = _open_sync_repo(token, sync_dir)
        tracked = repo.git.ls_files(f"{SPORT_DIR_NAME}/data_delivery").splitlines()
        # Board-backed retention: every date that still has a tracked
        # todays_games_<date>.csv board keeps its run-engine/predictions
        # artifacts — a navigable board must never lose the RUN ENGINE data
        # its cards need (the 2026-08-29 doubleheader regression).
        board_dates = {d for p in tracked
                       if _basename(p).startswith("todays_games_")
                       for d in [_artifact_date(p)] if d}
        # Classify: protected → keep; in seen → keep; within the 48h
        # retention window (current + previous GMT day) → keep; recent or
        # board-backed slate/run-engine artifacts → keep; otherwise → stale.
        stale = []
        kept_protected = 0
        kept_current = 0
        for p in tracked:
            verdict = classify_artifact(
                p, seen, _RETENTION_DATES, _RECENT_DATES, board_dates)
            if verdict == "seen":
                continue  # this run staged it
            if verdict == "protected":
                kept_protected += 1
                continue
            if verdict == "current":
                kept_current += 1
                continue
            stale.append(p)
        if kept_protected:
            print(f"  🛡️  Kept {kept_protected} protected file(s) (never deleted)")
        if kept_current:
            print(f"  📅 Kept {kept_current} same-day artifact(s)")
        if not stale:
            print("  ✅ No stale files — data_delivery holds exactly this run's artifacts")
        else:
            print(f"  🧹 Removing {len(stale)} stale files:")
            for s in stale:
                print(f"    {s}")
            repo.git.rm(stale)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            repo.index.commit(f"Remove stale data_delivery artifacts: {ts}")
            _git_push_confirmed(repo, CONFIG["github_branch"])
            print(f"  ✅ Removed {len(stale)} stale files — confirmed on {CONFIG['github_repo']}@{CONFIG['github_branch']}")
    except Exception as e:
        print(f"  ❌ Cleanup failed: {e}")

if sync_dir.exists():
    shutil.rmtree(sync_dir, ignore_errors=True)

_banner("DONE ✅")
print(f"  Games: {game_df.shape[0]}  |  Pitches: {pbp_df.shape[0]:,}  |  Features: {game_df.shape[1]+pbp_df.shape[1]}")
print(f"  Output: {out_dir}")
