from __future__ import annotations

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

# config is a leaf module (os/pathlib only) — safe to import this early,
# before the Phase 1 imports.
from config import resolve_run_end_date

CONFIG = {
    "start_date": _env_date("MLB_START_DATE", "2025-01-01"),
    # Stale-pin guard (2026-10-06 log review, T7): the Kaggle notebook
    # leaves a literal MLB_END_DATE behind after a rebuild — target=end,
    # so an unguarded stale pin would freeze the daily slate. The notebook
    # is Kaggle-owned and must not be edited from here; extend instead.
    "end_date":   resolve_run_end_date(
        _env_date("MLB_END_DATE",
                  __import__("datetime").date.today().strftime("%Y-%m-%d"))),
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

# ── Delivery-root reconciliation (2026-10-07 run-log review, T1) ───────────
# The kernel that launched this script may have imported `config` from ITS
# OWN checkout before the clone's backend/ landed at sys.path[0] above —
# from then on every `from config import DATA_DELIVERY_DIR` writes to the
# kernel tree while cwd / the tee / Phase 5's scan point at this clone.
# The 2026-10-07 run split exactly that way: its 15 reported artifacts
# landed outside every scanned tree (GitHub still served the 10-06 board)
# and the slate's pl_slate file was written beside the clone's STALE
# lineups cache while the loader read the kernel root, so all four slate
# sides shipped the marked carry. Reconcile the config module to THIS
# clone before anything imports from it; best-effort, never fatal — the
# Phase-5 union scan + reported-artifact gate below are the backstop.
try:
    import config as _cfg
    _reconcile = getattr(_cfg, "ensure_config_root", None)
    if _reconcile is not None:
        _reconcile(repo_dir / SPORT_DIR_NAME / "backend", log=print)
    else:
        # Foreign copy predates the helper (stale notebook checkout): the
        # roots stay split — announce loudly so the delivered log shows it.
        print(f"  ⚠️  config module {_cfg.__file__} predates "
              f"ensure_config_root — delivery roots NOT reconciled")
except Exception as _reconcile_exc:  # noqa: BLE001 — best-effort by design
    print(f"  ⚠️  config-root reconciliation failed ({_reconcile_exc})")

# Snapshot the artifacts already in the repo's data_delivery (relative path →
# mtime in ns) BEFORE the pipeline writes anything. Phase 5 must stage only
# files THIS run produced or modified; every other file in that folder is a
# stale file that Phase 6 will delete from GitHub. BOTH delivery roots are
# snapshotted (2026-10-07 run-log review, T1): under a config/cwd split the
# artifacts live in the config root while the clone root holds the tee and
# features' lineup artifacts — one snapshot would misread the other.
_preexisting_delivery: dict[str, int] = {}
_preexisting_delivery_roots: dict[Path, dict[str, int]] = {}


def _snapshot_delivery_root(_root: Path) -> None:
    try:
        _root = _root.resolve()
    except OSError:
        return
    if _root in _preexisting_delivery_roots or not _root.is_dir():
        return
    _snap: dict[str, int] = {}
    for _p in _root.rglob("*"):
        if _p.is_file():
            _snap[_p.relative_to(_root).as_posix()] = _p.stat().st_mtime_ns
    _preexisting_delivery_roots[_root] = _snap


_preexisting_dir = Path.cwd() / "data_delivery"
_snapshot_delivery_root(_preexisting_dir)
try:
    from config import DATA_DELIVERY_DIR as _CFG_DELIVERY_DIR
    _snapshot_delivery_root(Path(_CFG_DELIVERY_DIR))
except Exception:  # noqa: BLE001 — the cwd snapshot alone still works
    pass
_preexisting_delivery = _preexisting_delivery_roots.get(
    _preexisting_dir.resolve(), {})

for mod in list(sys.modules.keys()):
    if any(x in mod for x in ['ingestion', 'features', 'pipeline', 'training', 'data_ingestion', 'statcast', 'duckdb']):
        del sys.modules[mod]

# ── Phase 1: Ingestion ──────────────────────────────────────────────────────
import logging
logging.basicConfig(level=logging.INFO, format="  %(levelname)s %(message)s")
# Run-log tee (2026-10-01): capture the whole run into ONE rolling master
# file in data_delivery/ — Phase 5 pushes it like any artifact and Phase 6
# never evicts it (protected name), so the latest run's full log is always
# reviewable from a plain git pull. Degrades to console-only on failure.
from run_log_tee import (
    install_crash_log_pusher,
    install_run_log_tee,
    RUN_LOG_NAME,
)
# Never tee during a test run (NBA precedent, 5673874c): install opens
# the rolling log with "w", so importing this module under pytest would
# TRUNCATE the committed mlb_pipeline_run_log.txt at the tee's own
# header — measured at 143 bytes for NBA. PYTEST_CURRENT_TEST covers
# in-test imports, pytest being loaded covers collection/import-time
# probes, and a production launch (Kaggle/Colab) has neither.
if os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
    _log_path = None
else:
    _log_path = install_run_log_tee(Path.cwd() / "data_delivery")
install_crash_log_pusher(
    _log_path, CONFIG["github_username"], CONFIG["github_repo"],
    CONFIG["github_branch"])
# Delivery roots (2026-10-07 run-log review, T1): announce the tree this
# run writes artifacts into, NEXT TO the tee so the pushed log always
# carries it. A split (config root ≠ cwd root) is the exact defect class
# that kept 2026-10-07's 15 artifacts off GitHub and served stale slate
# pools — Phase 5 stages BOTH roots, but the reviewer must SEE the split.
try:
    from config import DATA_DELIVERY_DIR as _DD_FOR_LOG
    _root_cfg = Path(_DD_FOR_LOG).resolve()
    _root_cwd = (Path.cwd() / "data_delivery").resolve()
    if _root_cfg != _root_cwd:
        logging.warning(
            "delivery root SPLIT: config.DATA_DELIVERY_DIR=%s vs cwd "
            "data_delivery=%s — Phase 5 stages both", _root_cfg, _root_cwd)
    else:
        logging.info("delivery root: %s", _root_cfg)
except Exception as _roots_exc:  # noqa: BLE001 — observability only
    logging.warning("delivery roots unresolved (%s)", _roots_exc)
_banner("PHASE 1", "Statcast Data Ingestion")
from ingestion import pull_statcast, warn_missing_finals

start = datetime.strptime(CONFIG["start_date"], "%Y-%m-%d").date()
end = datetime.strptime(CONFIG["end_date"], "%Y-%m-%d").date()
out_dir = Path(CONFIG.get("output_dir", "/content/mlb_clean_data"))
out_dir.mkdir(parents=True, exist_ok=True)
# The weather-history cache lives beside pitches.parquet (outside the git
# repo) so Colab's per-run artifact sync never stages it.
getattr(os, "environ").setdefault("MLB_CACHE_DIR", str(out_dir))
pitches_path = out_dir / "pitches.parquet"

print(f"📅 {start} → {end}")
# The guard runs at CONFIG build (before the tee); re-announce through the
# logger HERE so the extension lands in the pushed run log (Phase 1 is
# after install_run_log_tee).
_pinned_end = os.environ.get("MLB_END_DATE", "").strip()
if _pinned_end and _pinned_end != CONFIG["end_date"]:
    logging.warning(
        "MLB_END_DATE=%s is stale (before today) — extended to %s so the "
        "daily slate cannot freeze; re-date the notebook pin (or unset it) "
        "for a past-window backfill", _pinned_end, CONFIG["end_date"])
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
# Missing-finals guard (2026-10-08 run-log review): Savant's index lagged
# the 10-07 slate through the whole run — MLB_FULL_REPULL's last chunk
# (2026-08-18 → 2026-10-08) returned cleanly yet the frame stopped at
# 2026-10-06 while four 10-07 games were already official finals. That
# slate never resolved into predictions_history / today's record, and
# nothing in the log said so. Warn loudly (never abort — posting lag is
# transient and the next run's tail refresh recovers the games).
try:
    warn_missing_finals(pitches_path, end)
except Exception as _mf_exc:  # noqa: BLE001 — a guard must never kill the run
    logging.warning("missing-finals guard skipped (%s)", _mf_exc)

# ── Phase 1.5: IL / availability ledgers (runtime inputs, NEVER pushed) ──────
# il_stints.parquet / il_stints_pitchers.parquet (+ .meta.json provenance) are
# rebuilt EVERY run here — they are local runtime caches, not GitHub artifacts.
# MLB_IL_STINTS_DIR points the builder and every consumer at the run cache
# (beside pitches.parquet, outside the git repo) so the Phase 5 sync can never
# stage them and Phase 6 never manages them; features/build_il_stints fall
# back to the legacy in-repo data_delivery location only when the env var is
# unset. The reconciliation source is pitches.parquet from Phase 1: the
# builder's plausibility gates need every 2024+ plate appearance / pitching
# appearance, which the newest pbp_defense snapshot alone cannot cover on a
# fresh clone. --start keeps the builder's 2023-01-01 default so the ledger
# provenance (window_start / gate band) matches every prior build. A failure
# here is NON-FATAL by design: features degrade loudly (participant-pool /
# exposure-0 semantics with WARNINGs) and the run continues. SystemExit must
# be caught explicitly: the builder's plausibility gates raise it ("refusing
# to write"), it is a BaseException that sails past ``except Exception``,
# and a gate trip unwound __main__ so the 2026-10-01 run silently ended
# before Phase 5 with exit code 0 — the Kaggle wrapper printed success.
from build_il_stints import main as _build_il_stints
_il_dir = out_dir / "il_stints"
_il_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MLB_IL_STINTS_DIR", str(_il_dir))
print(f"  📁 IL ledgers → {_il_dir}")
sys.argv = ["build_il_stints.py",
            "--end", CONFIG["end_date"],
            "--pbp", str(pitches_path)]
try:
    _build_il_stints()
except (Exception, SystemExit) as e:
    print(f"  ⚠️  IL ledger rebuild failed (non-fatal; features degrade "
          f"loudly): {e}")

# ── Phase 1.6: Announced lineups — membership tier 1 (tonight's nine) ──────
# lineups.parquet feeds lineup_effective (features.py): the membership set
# BOTH candidate pools bind — tonight's nine (tier 1), else the
# hand-conditioned 10-day projection (tier 2), else the full roster. Two
# incremental passes, both NON-FATAL (a gap degrades membership to the
# projection — worse information, never a crash):
#   1) decided-game gaps: every finished game still missing (or incomplete)
#      in the cache, driven by the PREVIOUS run's game_level_features.csv;
#   2) pre-game capture: today's scheduled game_pks via the StatsAPI
#      schedule — posted nines land NOW so an overnight rebuild of tonight's
#      games resolves tier 1 instead of projecting (the 2026-10-03 audit's
#      "wire daily pre-game fetch").
from backfill_lineups import (fetch_scheduled_lineups,
                              main as _backfill_lineups)
sys.argv = ["backfill_lineups.py", "--limit", "500"]
try:
    _backfill_lineups()
except (Exception, SystemExit) as e:
    print(f"  ⚠️  lineups decided-gap backfill failed (non-fatal; "
          f"membership degrades to projection): {e}")
try:
    _n_lineups = fetch_scheduled_lineups(CONFIG["end_date"])
    print(f"  🧾 lineups: {_n_lineups} posted row(s) captured for "
          f"{CONFIG['end_date']}")
except (Exception, SystemExit) as e:
    print(f"  ⚠️  pre-game lineup capture failed (non-fatal; membership "
          f"degrades to projection): {e}")

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
from features import add_diff_features, add_exp2_features

game_df = enrich_elo_and_records(game_df, rename_team_woba=True)
game_df = add_diff_features(game_df)
# Experiment #2 candidate features (C+E / D+F decision): built from the
# source layer's PIT-safe columns, shipped in the CSV (MONEYLINE_FEATURE_COLS members).
game_df = add_exp2_features(game_df)

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
except (Exception, SystemExit) as e:
    print(f"  ⚠️  Defense projection failed (non-fatal): {e}")

# ─────────────────────────────────────────────────────────────────────────────
# DAILY PIPELINE ORCHESTRATION (merged from the former backend/pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────
# run_daily_pipeline(target_date, ...) — the moneyline walk-forward, the run
# engine wiring, and every data_delivery artifact writer — now lives in THIS
# module, the one authoritative entry point, defined just before Phase 4 so
# the names exist at exactly the point the run previously imported them.
# Structural parity with the NBA/NFL/NHL master pipelines, which own their
# full phase stack; Phase 4 below calls it in-process.
#
# Merged verbatim: no function body, constant, ordering, seed, or artifact
# changed — the rename-aware diff plus the AST and name-flow proofs in the
# commit message are the audit trail.

import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from config import (
    CALIBRATION,
    DATA_DELIVERY_DIR,
    DATE_FMT,
    DATE_READABLE_FMT,
    MIN_VAL_FOLD_GAMES,
    MODEL_MONITOR,
    NEXT_RUN_HEURISTIC_DAYS,
    POWER_RANKINGS,
    RETRAIN_CADENCE_DAYS,
    RUN_DIAGNOSTICS_DIR,
    TODAYS_GAMES,
    VERSION_KEY,
    TRAINED_AT_KEY,
    DATA_CUTOFF_KEY,
    WEATHER_BACKFILL_ALL,
)
from data_ingestion import (
    attach_market_lines,
    build_upcoming_slate,
    compute_elos_up_to,
    generate_synthetic_games,
    generate_synthetic_market_lines,
    load_game_events,
    filter_prior,
)
from explainability import (
    compute_feature_coverage,
    compute_feature_drift,
    compute_rolling_brier,
    compute_run_engine_feature_coverage,
    compute_run_engine_feature_drift,
    compute_shap_per_game,
)
from feature_metadata import generate_features_metadata

# C2 k-edge expansion RETIRED then REMOVED (2026-09-27): the board prices
# the RAW λ pair — the diagnostic run_engine_k_edge.py module (monitor-only
# k-hat publication) is deleted; no k adjustment exists anywhere in the run
# line. Historical: the 2026-09-22 retirement measured k-hat ≈ 1.0 under the
# lr 0.03 × fixed-90-rounds adoption, making the transform inert.
from calibration import is_identity
from features import (
    add_diff_features,
    add_env_level_features,
    add_exp2_features,
    add_form_delta_features,
    add_lineup_delta_features,
    refine_dome_game_level,
)
from umpires import (
    build_umpire_stats,
    load_umpire_map,
    maintain_umpire_map,
)
from weather import apply_weather_features, fetch_day_weather, fetch_games_weather
from training import last_ensemble_info
from frames import (
    get_decided_frame,
    fold_signature,
    require_matching_signatures,
)
from github_sync import sync_artifacts
from results import _count_evening_games
from training import (
    MONEYLINE_FEATURE_COLS,
    active_moneyline_feature_cols,
    MARGIN_COL,
    _attach_oof_run_margins,
    compute_metrics,
    calibration_buckets,
    feature_importance_weights,
    get_last_calibrator,
    get_last_season_split,
    get_last_walk_forward_splits,
    get_last_fold_signature,
    load_ensemble,
    apply_bundle_feature_cols,
    persist_ensemble,
    predict_games,
    set_adaptive_weights,
    set_calibration,
    should_retrain,
    update_model_history,
    update_model_version_history,
    walk_forward_evaluate,
    walk_forward_splits,
)

logger = logging.getLogger(__name__)


def _attach_slate_run_margins(target_games: pd.DataFrame,
                              games: pd.DataFrame) -> pd.DataFrame:
    """Attach run_margin_diff to the prediction board BEFORE moneyline
    inference (shipped feature -- training-time OOF margins alone don't help
    the slate).

    Slate margins use the run engine's PRODUCTION slate convention: a
    fit-only refit of both per-side Poisson models on ALL decided games at
    the median fold round count from the moneyline's own walk-forward
    (build_oof_margin.refit_run_margins). Every predicted game is strictly
    future relative to that fit, so no margin can come from a model that
    saw the game. Falls back to a fresh run_oof for the round counts when no
    walk-forward ran this process (cached-ensemble path). Frames without the
    run-engine inputs keep an all-NaN margin (imputed by existing paths)
    with a loud warning -- never a fabricated 0.
    """
    from training import MONEYLINE_FEATURE_COLS
    if MARGIN_COL not in MONEYLINE_FEATURE_COLS:
        return target_games
    if target_games.empty:
        return target_games  # 0-row board (off-day): nothing to attach to
    _missing = {"game_pk", "home_score", "away_score"} - set(games.columns)
    if _missing:
        logger.warning(
            "run_margin_diff: slate attach skipped -- games frame lacks %s; "
            "margin stays all-NaN (imputed by existing paths)",
            sorted(_missing))
        out = target_games.copy()
        out[MARGIN_COL] = np.nan
        return out

    from build_oof_margin import MARGIN_COL as _BOM_MARGIN, refit_run_margins
    from distributions import run_oof, _resolve_slate_key
    from training import MONEYLINE_FEATURE_COLS, get_last_margin_rounds
    assert _BOM_MARGIN == MARGIN_COL and MARGIN_COL in MONEYLINE_FEATURE_COLS

    # Pre-game ESPN boards carry game_id only (no StatsAPI game_pk) -- the
    # 145d841 slate-key convention. refit_run_margins and the margin merge
    # below are keyed by game_pk (the run engine's slate rows carry the
    # ESPN id AS game_pk), so synthesize game_pk from game_id when absent;
    # otherwise the attach dies with KeyError('game_pk') and today's board
    # silently loses the shipped margin feature (the v26 error).
    _slate_key = _resolve_slate_key(target_games)
    if _slate_key == "game_id":
        target_games = target_games.copy()
        target_games["game_pk"] = target_games["game_id"]
    # Defensive: exact duplicate game_pk rows are a true bug (a doubleheader
    # whose legs share a matchup-based key is exactly that) — a many-to-many
    # merge would explode rows. Keep one row per distinct game_pk; distinct
    # legs carry distinct keys after slate disambiguation, so they never
    # collide.
    _dup = target_games["game_pk"].duplicated(keep="first")
    if _dup.any():
        logger.warning(
            "run_margin_diff: dropped %d exact-duplicate game_pk row(s) "
            "before margin merge (duplicate keys are a true bug)",
            int(_dup.sum()))
        target_games = target_games.loc[~_dup].copy()

    decided = get_decided_frame(games)
    rounds = get_last_margin_rounds()
    if not rounds:
        logger.info(
            "run_margin_diff: no walk-forward margin rounds in this process -- "
            "deriving them from a fresh run-engine OOF")
        try:
            rounds = run_oof(decided)["summary"]["final_fit_rounds"]
        except Exception as exc:
            logger.error("run_margin_diff: run_oof round derivation failed (%s); "
                         "margin stays all-NaN", exc)
            out = target_games.copy()
            out[MARGIN_COL] = np.nan
            return out

    margins = refit_run_margins(decided, target_games, rounds)
    # Margins hold one row per pred row; keep one per distinct key so the
    # left merge below can never multiply rows.
    margins = margins.drop_duplicates(subset=["game_pk"], keep="first")
    out = target_games.copy()
    out = out.drop(columns=[MARGIN_COL] if MARGIN_COL in out.columns else [])
    out = out.merge(margins[["game_pk", MARGIN_COL]], on="game_pk", how="left")
    logger.info(
        "run_margin_diff: slate margins attached (fit-only refit on %d decided "
        "games at median rounds %s); coverage %.1f%% of %d board rows",
        len(decided), {k: int(v) for k, v in rounds.items()},
        100 * float(out[MARGIN_COL].notna().mean()), len(out))
    return out


def _load_roof_cache(path: Path) -> dict[int, str]:
    """Load the roof-state cache JSON -> {int(game_pk): "open"|"closed"}.

    Idempotent and safe to call when the file does not exist.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
        return {int(k): v for k, v in raw.items() if v in ("open", "closed")}
    except Exception as exc:
        logger.warning("Roof cache unreadable (%s)", exc)
        return {}


def _save_roof_cache(path: Path, cache: dict[int, str]) -> None:
    """Persist the roof-state cache as a sorted JSON file."""
    path.write_text(
        json.dumps({str(k): v for k, v in sorted(cache.items())},
                   indent=2, sort_keys=True) + "\n")


# Retractable-roof teams (must match features.RETRACTABLE_ROOF_TEAMS)
_RETRACTABLE_TEAMS = frozenset(
    {"ARI", "AZ", "HOU", "MIA", "MIL", "SEA", "TEX", "TOR"})


def _roof_from_statsapi_condition(condition) -> str | None:
    """Map StatsAPI gameData.weather.condition to roof state.

    Retractable-roof parks: 'Roof Closed'/'Dome' -> 'closed',
    real weather (Clear, Sunny, etc.) -> 'open',
    missing/empty -> None (unknown).
    """
    if condition is None:
        return None
    text = str(condition).strip().lower()
    if not text:
        return None
    if ("roof closed" in text or text == "indoor"
            or "closed roof" in text or text == "dome"):
        return "closed"
    # Real weather descriptions mean the roof was open
    if any(w in text for w in (
            "clear", "sunny", "cloud", "rain", "snow", "drizzle",
            "overcast", "fog", "wind", "hot", "cold", "warm",
            "cool", "fair", "partly", "mostly", "hazy", "mist")):
        return "open"
    if "roof open" in text or "open roof" in text:
        return "open"
    return None


def _topup_roof_cache(
    games: pd.DataFrame,
    roof_states: dict[int, str],
    cache_path: Path,
    budget_sec: float = 128.0,
) -> dict[int, str]:
    """Best-effort top-up of the roof cache for missing retractable-home games.

    Fetches gameData.weather.condition from the StatsAPI live feed for each
    retractable-home game_pk missing from the cache. Budget-capped: pauses
    and returns the current state when time runs out; missing games will be
    retried on the next pipeline run. Never blocks the pipeline on failure.
    """
    import requests
    import time as _time
    feed_url = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
    pause = 0.5  # ~2 req/s
    retries = 2

    home = games["home_team"].astype(str).str.upper().str.strip()
    retract_mask = home.isin(_RETRACTABLE_TEAMS)
    retract_pks = set(int(pk) for pk in
                      games.loc[retract_mask, "game_pk"].tolist())
    missing = sorted(retract_pks - set(roof_states.keys()))
    if not missing:
        return roof_states

    logger.info(
        "Roof top-up: %d/%d retractable games cached, %d missing",
        len(retract_pks & set(roof_states.keys())),
        len(retract_pks), len(missing))

    start = _time.monotonic()
    new_entries = 0
    for pk in missing:
        if _time.monotonic() - start >= budget_sec:
            logger.info(
                "Roof top-up: budget exhausted after %d/%d fetches — "
                "remaining %d will retry next run",
                new_entries, len(missing), len(missing) - new_entries)
            break
        condition = None
        for attempt in range(retries):
            try:
                resp = requests.get(feed_url.format(pk=pk), timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    condition = (
                        (data.get("gameData") or {}).get("weather", {})
                        .get("condition"))
                    break
            except Exception:
                pass
            if attempt < retries - 1:
                _time.sleep(pause * 3)
            _time.sleep(pause)
        roof = _roof_from_statsapi_condition(condition)
        if roof is not None:
            roof_states[pk] = roof
            new_entries += 1

    if new_entries > 0:
        _save_roof_cache(cache_path, roof_states)
        logger.info("Roof top-up: %d new entries (cache now %d)",
                    new_entries, len(roof_states))

    return roof_states


def _attach_drift_run_margins(decided: pd.DataFrame) -> pd.DataFrame:
    """Attach leakage-free OOF run margins to the drift frame so the
    shipped run_margin_diff feature is drift-monitored like every other
    numeric feature.

    The margin column lives ONLY in the margin-enriched training frame
    (build_oof_margin.oof_run_margins, attached inside
    walk_forward_evaluate) -- it never lands in game_level_features.csv. The
    drift step slices its windows from that CSV, so without this enrichment
    compute_feature_drift silently omits run_margin_diff's row from the PSI
    table (the one numeric moneyline feature missing from drift). Uses the
    SAME machinery as training (walk_forward_splits + _attach_oof_run_margins,
    run engine READ-ONLY), so every game's margin comes from a model trained
    strictly before it. Games outside executed folds (early warm-up rows)
    stay NaN → imputed at training; here they are excluded from the drift
    distribution with honest coverage counts.

    A failed derivation warns loudly and returns the frame unchanged -- the
    margin row is then omitted from drift, never fabricated.
    """
    if MARGIN_COL not in MONEYLINE_FEATURE_COLS:
        return decided
    try:
        # Correctness is based on deterministic geometry, not process state.
        # The caller supplies the canonical game_level_features row set;
        # cached splits are only an optional fast path when their signature
        # matches that frame exactly.
        canonical = decided.sort_values("game_date").reset_index(drop=True)
        canonical_splits = walk_forward_splits(
            canonical, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
        _splits = get_last_walk_forward_splits()
        def _signature(items):
            return [(s["fold_idx"], str(s["val_start"]), str(s["val_end"]),
                     tuple(s["val_games"].get("game_pk", pd.Series(dtype=object)).tolist()))
                    for s in items]
        if not _splits or _signature(_splits) != _signature(canonical_splits):
            _splits = canonical_splits
        if not _splits:
            return decided
        if not decided.reset_index(drop=True)["game_pk"].tolist() == canonical["game_pk"].tolist():
            decided = canonical
        enriched, _ = _attach_oof_run_margins(
            decided, _splits, MIN_VAL_FOLD_GAMES, 0,
            RETRAIN_CADENCE_DAYS, 0)
        return enriched
    except Exception as exc:
        logger.warning(
            "run_margin_diff: drift margin attach failed (%s) -- margin "
            "row omitted from drift, drift continues", exc)
        return decided


def _carry_forward_slate_details(slate: pd.DataFrame, target_date_str: str) -> pd.DataFrame:
    """Re-apply pitcher names + market lines from an earlier same-date artifact.

    ESPN drops probablePitcher from the scoreboard once a game starts, so an
    evening rerun rebuilds the slate with sp_name_* = 'TBD' and erases the
    pitching matchup already published that morning (which also blanks the
    ERA/K9 boxes, since names drive the stat lookup). Anything already
    published for a game_id is restored onto the rebuilt slate.
    """
    path = DATA_DELIVERY_DIR / f"{TODAYS_GAMES}_{target_date_str}.csv"
    if slate.empty or not path.exists():
        return slate
    try:
        prev = pd.read_csv(path)
    except Exception as e:
        logger.warning("Could not read previous %s for carry-forward: %s", path.name, e)
        return slate
    if prev.empty or "game_id" not in prev.columns:
        return slate
    carry_cols = [c for c in (
        # Pitching matchup + stats: ESPN drops probablePitcher once a game
        # starts, so an evening rerun restores the morning's published
        # starters AND their ERA/K9 lines (names alone don't re-derive stats
        # without the pbp mapping), plus the StatsAPI ids.
        "sp_name_home", "sp_name_away",
        "sp_xfip_home", "sp_k9_home", "sp_xfip_away", "sp_k9_away",
        "sp_id_home", "sp_id_away",
        "moneyline_home", "moneyline_away", "total_line", "run_line_home", "juice",
    ) if c in prev.columns and c in slate.columns]
    if not carry_cols:
        return slate
    prev = prev.drop_duplicates("game_id").set_index("game_id")
    restored = 0
    for idx, row in slate.iterrows():
        gid = row.get("game_id")
        if gid not in prev.index:
            continue
        p = prev.loc[gid]
        for c in carry_cols:
            cur = row[c]
            stale = pd.isna(cur) or (isinstance(cur, str) and cur.strip().upper() == "TBD")
            new = p[c]
            if stale and pd.notna(new):
                slate.at[idx, c] = new
                restored += 1
    if restored:
        logger.info("Carried forward %d slate details from earlier artifact", restored)
    return slate


def _nearest_slate_pk(legs, start_et) -> Optional[int]:
    """Nearest StatsAPI game_pk for a slate row by first-pitch time.

    ``legs`` is [(first_pitch_et, game_pk), ...] for ONE matchup — a
    doubleheader holds two entries (two games, usually two starters). The
    nearest leg within a 4h window IS the row's own game; single-game
    matchups match trivially. Returns None when the matchup is unknown or
    nothing is close (the row falls back to projected lineups). Pure.
    """
    if not legs:
        return None
    target = pd.Timestamp(start_et)
    if pd.isna(target):
        return None
    if target.tzinfo is not None:
        target = target.tz_convert("America/New_York").tz_localize(None)
    best, best_d = None, None
    for t, pk in legs:
        if pd.isna(t):
            continue
        t = t.tz_localize(None) if t.tzinfo is not None else t
        d = abs((t - target).total_seconds())
        if best_d is None or d < best_d:
            best, best_d = pk, d
    if best is not None and best_d is not None and best_d <= 4 * 3600:
        return best
    return None


def _attach_slate_lineup_keys(slate: pd.DataFrame,
                              lineup_rows: pd.DataFrame) -> pd.DataFrame:
    """Carry resolved StatsAPI game identities onto the ESPN slate.

    Posted lineup enrichment is keyed by ``game_pk``. ESPN rows generally
    carry only ``game_id``; keeping this conversion explicit prevents a
    successful StatsAPI lineup fetch from degenerating into all-NaN lineup
    features during the subsequent PIT wOBA join.
    """
    if "game_pk" not in lineup_rows.columns or len(lineup_rows) != len(slate):
        raise ValueError("slate lineup identity rows must align one-to-one")
    out = slate.copy()
    out["game_pk"] = pd.to_numeric(
        lineup_rows["game_pk"], errors="coerce").astype("Int64")
    return out


def _fetch_slate_lineups(slate: pd.DataFrame, target_date: date) -> pd.DataFrame:
    """Attach the 6 lineup-delta columns to today's slate from posted lineups.

    Resolution: StatsAPI schedule for target_date maps (home, away) → game_pk
    (the slate carries no StatsAPI game_pk -- ESPN's game_id only), then the
    live feed per game, paced like the roof fetcher (~2.2 req/s, one retry).
    Games with a complete 9+9 battingOrder get REAL lineup-delta features
    (same point-in-time math as training: batter/team sd-wOBA through games
    strictly before today -- no lookahead).

    Projected fallback for games not yet posted (per the 2026-08-25 posting-
    curve probe, away sides generally post ~2-3h before first pitch; a morning
    slate is mostly projected): ALL SIX columns emit NULL. A lineup not yet
    posted before first pitch is genuinely UNKNOWN at bet time — it must never
    be fabricated as 0 (a fake "projected lineup equals season mean" leaks a
    value the model can't actually have). NULLs route through the existing NaN
    imputation path, so the tree learns "not yet posted" as its own missing
    state — the same strict point-in-time discipline the market-line as-of
    join (data_ingestion._attach_market_lines rejects lines posted at/after
    start) and weather/roof NULLs use. The actual-vs-projected split is logged
    loudly so a projected-only morning is visible.
    """
    if slate is None or slate.empty:
        return slate
    slate = slate.reset_index(drop=True)  # posted-mask aligns by position below
    import requests
    import time as _time
    from results import STATSAPI_SCHEDULE_URL  # same endpoint the weather backfill uses
    _FEED = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"

    # 1) game_pk resolution from the StatsAPI schedule (one day, no chunking).
    # A matchup can hold MULTIPLE games (doubleheader legs with their own
    # game_pk), so collect EVERY leg with its first-pitch time and match per
    # slate row by start time — a one-entry-per-matchup map would feed both
    # legs the first game's lineup.
    pk_by_teams: dict[tuple[str, str], list[tuple[pd.Timestamp, int]]] = {}
    try:
        resp = requests.get(STATSAPI_SCHEDULE_URL,
                            params={"sportId": 1,
                                    "startDate": target_date.isoformat(),
                                    "endDate": target_date.isoformat()},
                            timeout=20)
        resp.raise_for_status()
        for g in (resp.json().get("dates") or [{}])[0].get("games") or []:
            t = (g.get("teams") or {}).get("away") or {}
            h = (g.get("teams") or {}).get("home") or {}
            away = (t.get("team") or {}).get("abbreviation")
            home = (h.get("team") or {}).get("abbreviation")
            if home and away:
                try:
                    gd = pd.Timestamp(g.get("gameDate"))
                    if gd.tzinfo is None:
                        gd = gd.tz_localize("UTC")
                    gd = gd.tz_convert("America/New_York").tz_localize(None)
                except Exception:
                    gd = pd.NaT
                pk_by_teams.setdefault((home, away), []).append((gd, int(g["gamePk"])))
    except Exception as e:
        logger.warning("_fetch_slate_lineups: schedule resolution failed (%s); slate stays projected", e)
        pk_by_teams = {}

    # 2) per-game feed fetch (paced, one retry, cached per run)
    def _feed(pk: int) -> dict | None:
        for attempt in (0, 1):
            try:
                r = requests.get(_FEED.format(pk=pk), timeout=15)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
            if attempt == 0:
                _time.sleep(_LINEUP_PAUSE_SEC * 3)
            _time.sleep(_LINEUP_PAUSE_SEC)
        return None

    def _orders(feed: dict | None) -> tuple[list[int], list[int]]:
        out = []
        if feed:
            bs = ((feed.get("liveData") or {}).get("boxscore") or {})
            teams_bs = bs.get("teams") or {}
            for side in ("home", "away"):
                try:
                    order = [p["person"]["id"]
                             for p in (teams_bs[side].get("battingOrder") or [])]
                except Exception:
                    order = []
                out.append(order)
        return (out[0] if out else [], out[1] if len(out) > 1 else [])

    rows = []
    for _, r in slate.iterrows():
        teams_key = (r.get("home_team"), r.get("away_team"))
        pk = _nearest_slate_pk(pk_by_teams.get(teams_key), r.get("start_time_utc"))
        if pk is None:
            rows.append({"game_pk": pd.NA, "home_order": None, "away_order": None})
            continue
        feed = _feed(pk)
        ho, ao = _orders(feed)
        rows.append({"game_pk": int(pk), "home_order": ho or None,
                     "away_order": ao or None})
    lu = pd.DataFrame(rows)
    # game_pk must be TYPED before any join: rows mix resolved ints with
    # pd.NA (games whose StatsAPI identity has not resolved yet), so the
    # bare list-of-dicts frame lands as object dtype. pandas refuses
    # object-vs-Int64 key merges outright — on 2026-09-22 that ValueError
    # killed the whole slate build and shipped zero dated artifacts for the
    # day (the dashboard fell back to the previous day's files).
    if "game_pk" in lu.columns:
        lu["game_pk"] = pd.to_numeric(lu["game_pk"], errors="coerce").astype("Int64")

    # StatsAPI is the authoritative identity for posted lineups. The ESPN
    # slate normally has only game_id, but add_lineup_delta_features joins
    # both the lineup override and the PIT wOBA caches by game_pk. Carry the
    # resolved key onto the slate before enrichment; otherwise every actual
    # lineup silently misses the join and all six shipped features remain NaN.
    slate = _attach_slate_lineup_keys(slate, lu)

    # 3) real features where both sides posted; projected fallback otherwise
    slate = add_lineup_delta_features(slate, lineups_override=lu)
    from features import LINEUP_DELTA_COLS, LINEUP_TOP5_K
    posted = lu["home_order"].notna() & lu["away_order"].notna()
    n_actual = int(posted.sum())
    for idx, r in slate.iterrows():
        if not posted.iloc[idx]:
            # STRICT POINT-IN-TIME: a lineup not posted before first pitch is
            # UNKNOWN at bet time. Emit NULL for ALL SIX columns — never a
            # fabricated 0 ("projected lineup equals season mean" would inject
            # a value that cannot exist at bet time). NULLs route through the
            # existing imputation path, matching the market-line as-of join
            # and weather/roof missing-observation semantics.
            for c in LINEUP_DELTA_COLS:
                slate.at[idx, c] = pd.NA
    logger.info(
        "slate lineups: %d/%d ACTUAL (both sides posted), %d/%d projected "
        "(not yet posted → all 6 lineup-delta cols NULL, PIT-safe)",
        n_actual, len(slate), len(slate) - n_actual, len(slate))
    return slate


def _today_games_csv(games: pd.DataFrame, target_date_str: str) -> Path:
    """Write todays_games_YYYYMMDD.csv artifact."""
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"{TODAYS_GAMES}_{target_date_str}.csv"
    # Select output columns
    out_cols = [
        "game_id", "game_date", "start_time_utc", "home_team", "away_team",
        "home_record", "away_record", "home_win_prob_model", "away_win_prob_model",
        "moneyline_home", "moneyline_away", "total_line", "run_line_home",
        "juice", "edge_home", "edge_away",
        "sp_name_home", "sp_name_away",
        "sp_xfip_home", "sp_k9_home", "sp_xfip_away", "sp_k9_away",
        "venue", "model_pick", "home_win",
        # Finals for finished games (ESPN results merged onto the slate)
        "home_score", "away_score", "total_runs",
        # Game state from ESPN -- drives Live/Final status on the dashboard
        "game_state", "game_status_detail",
        # pl_* provenance (2026-10-03 slate alignment): whether each side's
        # position pools RESOLVED through the 3-tier chain tonight
        # (pool-t1/t2/t3), fell back to the marked carry, or is missing.
        "pl_source_home", "pl_source_away",
    ]
    cols = [c for c in out_cols if c in games.columns]
    games[cols].to_csv(path, index=False)
    return path


def _power_rankings_csv(games: pd.DataFrame, target_date_str: str) -> Path:
    """Write power_rankings_YYYYMMDD.csv artifact."""
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"{POWER_RANKINGS}_{target_date_str}.csv"

    teams = games["home_team"].unique()
    # Ties/postponements carry home_win = NULL -- they are not wins or losses
    # and must not crash int() conversion or distort percentages.
    decided = games[games["home_win"].notna()]

    # Derive the displayed run differential from the same decided game rows
    # used for the rest of the ranking.  The slate can contain NULL scores
    # before first pitch, so only completed games with both score fields are
    # eligible; missing scores must not be converted into fabricated runs.
    team_run_diffs: dict[str, int] = {}
    if {"home_score", "away_score"} <= set(decided.columns):
        decided_scores = decided.copy()
        decided_scores["home_score"] = pd.to_numeric(
            decided_scores["home_score"], errors="coerce"
        )
        decided_scores["away_score"] = pd.to_numeric(
            decided_scores["away_score"], errors="coerce"
        )
        scored = decided_scores[
            decided_scores["home_score"].notna()
            & decided_scores["away_score"].notna()
        ]
        for _, game in scored.iterrows():
            home = game["home_team"]
            away = game["away_team"]
            home_runs = int(game["home_score"])
            away_runs = int(game["away_score"])
            team_run_diffs[home] = team_run_diffs.get(home, 0) + home_runs - away_runs
            team_run_diffs[away] = team_run_diffs.get(away, 0) + away_runs - home_runs

    rankings = []
    for team in teams:
        home_games = decided[decided["home_team"] == team]
        away_games = decided[decided["away_team"] == team]
        team_games = decided[(decided["home_team"] == team) | (decided["away_team"] == team)]

        elo_rows = games[games["home_team"] == team]
        elo = elo_rows["home_elo"].mean() if not elo_rows.empty else 1500.0
        wins = int(home_games["home_win"].sum()) + int((1 - away_games["home_win"]).sum()) if not away_games.empty else 0
        losses = int((1 - home_games["home_win"]).sum()) + int(away_games["home_win"].sum()) if not away_games.empty else 0
        total = wins + losses
        pct = round(wins / max(total, 1), 3)

        home_count = len(home_games)
        home_wins = int(home_games["home_win"].sum()) if not home_games.empty else 0
        home_pct = round(home_wins / max(home_count, 1), 3)

        away_count = len(away_games)
        away_wins = int(away_games["home_win"].sum()) if not away_games.empty else 0
        away_pct = round(1 - away_wins / max(away_count, 1), 3) if away_count > 0 else 0.5

        # L10 -- last 10 DECIDED games
        recent = team_games.tail(10)
        l10_wins = 0
        for _, g in recent.iterrows():
            if g["home_team"] == team:
                l10_wins += int(g["home_win"])
            else:
                l10_wins += int(1 - g["home_win"])
        l10 = f"{l10_wins}-{len(recent) - l10_wins}"

        rankings.append({
            "team": team,
            "team_name": team,
            "elo": round(elo, 1),
            "wins": wins,
            "losses": losses,
            "record": f"{wins}-{losses}",
            "pct": pct,
            "run_diff": team_run_diffs.get(team, 0),
            "l10": l10,
            "home_pct": home_pct,
            "away_pct": away_pct,
        })

    df = pd.DataFrame(rankings).sort_values("elo", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"
    df.to_csv(path)
    return path


def _daily_calibration_rows(oof: Optional[pd.DataFrame]) -> list[dict]:
    """Per-day predicted-vs-actual win rates from walk-forward OOF predictions.

    Each row covers one game date; predictions come from the fold trained
    strictly on prior games (point-in-time safe by construction).
    """
    if oof is None or oof.empty or "home_win_prob_model" not in oof.columns:
        return []
    df = oof.copy()
    has_cal = "home_win_prob_model_calibrated" in getattr(df, "columns", [])
    days = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    rows: list[dict] = []
    for day, g in df.groupby(days):
        y_true = pd.to_numeric(g["home_win"], errors="coerce")
        y_pred = pd.to_numeric(g["home_win_prob_model"], errors="coerce")
        ok = y_true.notna() & y_pred.notna()
        n = int(ok.sum())
        if n == 0:
            continue
        yt, yp = y_true[ok].values, y_pred[ok].values
        try:
            m = compute_metrics(yt, yp)
        except Exception:
            m = {"auc": 0.5, "brier": 0.25, "logloss": 0.69, "ece": 0.0}
        entry_metrics = {
            "auc": round(float(m.get("auc", 0.5)), 4),
            "brier": round(float(m.get("brier", 0.25)), 4),
            "logloss": round(float(m.get("logloss", 0.69)), 4),
            "ece": round(float(m.get("ece", 0.0)), 4),
        }
        row = {
            "date": day.strftime("%Y%m%d"),
            "n_games": n,
            "wins": int((yt == 1).sum()),
            "losses": int((yt == 0).sum()),
            "metrics": entry_metrics,
            "buckets": calibration_buckets(yt, yp),
        }
        # Per-day post-hoc calibration quality (prequential OOF twins).
        if has_cal:
            y_cal = pd.to_numeric(
                g["home_win_prob_model_calibrated"], errors="coerce"
            )
            okc = ok & y_cal.notna()
            if int(okc.sum()) > 0:
                yc = y_cal[okc].values
                try:
                    mc = compute_metrics(yt[okc.values], yc)
                    entry_metrics.update({
                        "brier_calibrated": round(float(mc.get("brier", 0.25)), 4),
                        "logloss_calibrated": round(float(mc.get("logloss", 0.69)), 4),
                        "ece_calibrated": round(float(mc.get("ece", 0.0)), 4),
                    })
                    row["buckets_calibrated"] = calibration_buckets(yt[okc.values], yc)
                except Exception:
                    pass
            # Raw-axis calibrated twin: for each RAW-probability bucket,
            # the mean favored-side CALIBRATED probability of those same
            # games. Lets the daily calibration curve plot both curves on
            # one comparable axis (vertical gap = correction applied).
            if int(okc.sum()) > 0:
                import numpy as _np
                _raw_fav = _np.maximum(yp[okc.values], 1.0 - yp[okc.values])
                _cal_fav = _np.maximum(y_cal[okc].values, 1.0 - y_cal[okc].values)
                import re as _re
                for b in row["buckets"]:
                    label = str(b.get("bucket", ""))
                    match = _re.fullmatch(
                        r"\s*(\d+(?:\.\d+)?)\s*(?:-|–|—)+\s*"
                        r"(\d+(?:\.\d+)?)\s*%?\s*", label)
                    if match is None:
                        raise ValueError(
                            f"Could not parse calibration bucket label {label!r}; "
                            "expected '<low>-<high>' with optional % and hyphen/en-dash"
                        )
                    lo = float(match.group(1)) / 100.0
                    hi = float(match.group(2)) / 100.0
                    mask = (_raw_fav >= lo) & (_raw_fav < hi)
                    if hi >= 0.999:
                        mask |= _raw_fav == hi
                    if mask.any():
                        b["cal_mean_predicted"] = round(float(_cal_fav[mask].mean()), 4)
        rows.append(row)
    rows.sort(key=lambda r: r["date"])
    return rows


def _calibration_json(
    metrics: dict[str, float],
    y_true, y_pred,
    target_date_str: str,
    n_games: int,
    oof: Optional[pd.DataFrame] = None,
    evening_games: Optional[int] = None,
) -> Path:
    """Write calibration_YYYYMMDD.json artifact.

    Headline buckets use the same GRADING walk-forward population as metrics when
    available (a far richer curve than the target day alone); ``daily``
    carries per-day predicted-vs-actual for the date selector.
    ``evening_games``: count of slate games beginning at/after 7 PM ET
    (computed by the caller via _count_evening_games; default None keeps
    the pre-fix behavior for direct callers).
    """
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"{CALIBRATION}_{target_date_str}.json"

    import numpy as np
    if oof is not None and "home_win_prob_model" in getattr(oof, "columns", []):
        ot = pd.to_numeric(oof["home_win"], errors="coerce")
        op = pd.to_numeric(oof["home_win_prob_model"], errors="coerce")
        ok = ot.notna() & op.notna()
        if "grades_pooled" in oof:
            ok &= oof["grades_pooled"].astype(bool)
        y_true, y_pred = ot[ok].values, op[ok].values
    buckets = calibration_buckets(np.asarray(y_true), np.asarray(y_pred))

    # Post-hoc recalibration report: raw vs calibrated quality over the
    # pooled walk-forward OOF set, plus the fitted Platt parameters.
    cal_section: Optional[dict] = None
    if oof is not None and {"home_win", "home_win_prob_model"} <= set(
        getattr(oof, "columns", [])
    ):
        ot = pd.to_numeric(oof["home_win"], errors="coerce")
        op = pd.to_numeric(oof["home_win_prob_model"], errors="coerce")
        oc = (
            pd.to_numeric(oof["home_win_prob_model_calibrated"], errors="coerce")
            if "home_win_prob_model_calibrated" in oof.columns
            else None
        )
        ok = ot.notna() & op.notna()
        if "grades_pooled" in oof:
            ok &= oof["grades_pooled"].astype(bool)
        if oc is not None:
            ok &= oc.notna()
        if int(ok.sum()) > 0:
            m_raw = compute_metrics(ot[ok].values, op[ok].values)
            calibrator = get_last_calibrator()
            cal_section = {
                "method": (calibrator or {}).get("method", "identity") if not is_identity(calibrator) else "identity",
                "params": calibrator,
                "metrics_raw": {k: m_raw.get(k) for k in ("brier", "logloss", "ece")},
            }
            if oc is not None:
                m_cal = compute_metrics(ot[ok].values, oc[ok].values)
                cal_section["metrics_calibrated"] = {
                    k: m_cal.get(k) for k in ("auc", "brier", "logloss", "ece")
                }
                cal_section["calibration_buckets_calibrated"] = calibration_buckets(
                    np.asarray(ot[ok].values), np.asarray(oc[ok].values)
                )

    # League-wide metadata (counted from the slate by the caller: games
    # beginning at/after 7 PM ET, converted to ET before the hour test so
    # UTC-midnight rollover games are not dropped).
    if evening_games is None:
        evening_games = 0

    data = {
        "date": target_date_str,
        # Population labels (shared semantics with the NFL/NHL/NBA
        # calibration writers): ``n_games`` is the GRADING population the
        # metrics/buckets above cover — the sum of calibration_buckets
        # counts, what the dashboard shows as "n = N games" beside the
        # pooled KPIs. The day's slate size stays in ``league_total`` (the
        # todays pages' "X of Y games shown" denominator). The old
        # slate-size n_games made the MLB dashboard label pooled OOF
        # metrics "n = 4 games" (2026-10-07 dashboard/pooled-run review).
        "n_games": int(len(y_true)),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "metrics": metrics,
        "calibration_buckets": buckets,
        "n_eval": int(len(y_true)),
        "probability_view": "causal_rolling_blend",
        "calibration": cal_section,
        "daily": _daily_calibration_rows(oof),
        "league_total": n_games,
        "evening_games_league": evening_games,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def _predictions_history_csv(
    oof: Optional[pd.DataFrame], target_date_str: str
) -> Optional[Path]:
    """Write predictions_history_YYYYMMDD.csv -- every walk-forward OOF game
    prediction with its actual result.

    Feeds the Calibration page's per-game history table (the same games that
    feed the reliability diagram). Point-in-time safe on the MODEL side by
    construction: each member prediction comes from the fold trained
    strictly on prior games. Each blend uses that origin's strictly-prior
    learned weights. Final serving-weight replay is retrospective, never
    substituted into this OOF history.

    Column semantics (see README "The three probability quantities"):
      * home_win_prob_model            → (1) RAW OOF blend. Input to maps.
      * home_win_prob_model_calibrated → (2) PER-FOLD PREQUENTIAL map, fitted
        on prior folds only. Honest for scoring/metrics; NEVER display.
      * deployed/user-facing (3) is NOT a column: consumers compute
        σ(a·logit(raw)+b) with the global map in calibration_<date>.json.
        Display this one everywhere.
    Never mix (2) and (3) in the same chart or comparison.
    """
    import numpy as np
    if oof is None or oof.empty or "home_win_prob_model" not in getattr(oof, "columns", []):
        return None
    df = pd.DataFrame({
        "game_id": oof.get("game_id"),
        "game_date": pd.to_datetime(oof.get("game_date"), errors="coerce").dt.strftime("%Y-%m-%d"),
        "home_team": oof.get("home_team"),
        "away_team": oof.get("away_team"),
        "home_score": oof.get("home_score"),
        "away_score": oof.get("away_score"),
        "home_win": oof.get("home_win"),
        "home_win_prob_model": pd.to_numeric(oof["home_win_prob_model"], errors="coerce"),
        **(
            {
                "home_win_prob_model_calibrated": pd.to_numeric(
                    oof["home_win_prob_model_calibrated"], errors="coerce"
                )
            }
            if "home_win_prob_model_calibrated" in getattr(oof, "columns", [])
            else {}
        ),
    }).copy()
    decided = pd.to_numeric(df["home_win"], errors="coerce")
    df = df[decided.notna() & df["home_win_prob_model"].notna()]
    if df.empty:
        return None
    hw = pd.to_numeric(df["home_win"], errors="coerce").astype(int)
    prob = df["home_win_prob_model"]
    home_won_pick = prob >= 0.5
    df["model_pick"] = np.where(home_won_pick, df["home_team"], df["away_team"])
    df["actual_winner"] = np.where(hw == 1, df["home_team"], df["away_team"])
    df["correct"] = (home_won_pick == (hw == 1)).astype(int)
    df = df.sort_values(["game_date", "game_id"], ascending=[False, True])
    path = DATA_DELIVERY_DIR / f"predictions_history_{target_date_str}.csv"
    df.to_csv(path, index=False)
    logger.info("Prediction history written: %d games -> %s", len(df), path.name)
    return path


def _model_monitor_json(
    metrics: dict[str, float],
    drift_df: pd.DataFrame,
    target_date_str: str,
    last_retrained: Optional[str] = None,
    version: str = "v3.2.1",
    ensemble: Optional[list] = None,
    coverage_df: Optional[pd.DataFrame] = None,
    rolling_brier: Optional[dict] = None,
    features_metadata: Optional[dict] = None,
    run_engine: Optional[dict] = None,
    season_split: Optional[dict] = None,
) -> Path:
    """Write model_monitor_YYYYMMDD.json artifact."""
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"{MODEL_MONITOR}_{target_date_str}.json"

    # Load model history
    history_path = DATA_DELIVERY_DIR / "model_history.json"
    history = []
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    # Version-history snapshots (weights + metrics + calibration params per
    # run); falls back to legacy model_history rows for pre-snapshot runs.
    vh_path = DATA_DELIVERY_DIR / "model_version_history.json"
    version_history = []
    if vh_path.exists():
        try:
            with open(vh_path) as f:
                version_history = json.load(f)
            if not isinstance(version_history, list):
                version_history = []
        except ValueError:
            version_history = []
    if not version_history:
        version_history = list(history) if isinstance(history, list) else []

    # Drift summary
    n_warns = int((drift_df["status"] == "WARN").sum()) if not drift_df.empty else 0
    n_alerts = int((drift_df["status"] == "ALERT").sum()) if not drift_df.empty else 0
    n_seasonal = int((drift_df["status"] == "OK-SEASONAL").sum()) if not drift_df.empty else 0
    # 2026-10-10 log review: warnings=0/alerts=0 is only a clean result
    # when features were actually judged. INSUFFICIENT rows (window under
    # the PSI sample floor) are reported separately so a fully inert drift
    # run cannot ship a green summary card.
    n_insufficient = (int((drift_df["status"] == "INSUFFICIENT").sum())
                      if not drift_df.empty else 0)
    n_evaluated = (len(drift_df) - n_insufficient) if not drift_df.empty else 0
    warn_features = drift_df[drift_df["status"].isin(["WARN", "ALERT"])]["feature"].tolist() if not drift_df.empty else []

    data = {
        "date": target_date_str,
        "version": version,
        "last_retrained": last_retrained or datetime.now().strftime("%Y-%m-%d"),
        # NEXT RETRAIN = the next EXPECTED run per the retrain-every-run
        # decision (NEXT_RUN_HEURISTIC_DAYS=1 -> tomorrow), NOT a scheduler
        # (no cron/next_run exists). RETRAIN_CADENCE_DAYS is the walk-forward
        # FOLD cadence and must never drive this card.
        "next_retrain": (datetime.now() + timedelta(days=NEXT_RUN_HEURISTIC_DAYS)).strftime("%Y-%m-%d"),
        "metrics": metrics,
        # Season-split reporting (2026-10-03): headline metrics above grade
        # regular-season non-provisional rows only; these four published
        # blocks (regular/postseason/provisional/all) make every scored
        # population reconcilable against the headline instead of hidden.
        "season_split": season_split or {},
        "probability_view": "causal_rolling_blend",
        "drift_summary": {
            "warnings": n_warns,
            "alerts": n_alerts,
            "seasonal": n_seasonal,
            "evaluated": n_evaluated,
            "insufficient": n_insufficient,
            "features": warn_features,
        },
        "feature_drift": drift_df.to_dict(orient="records") if not drift_df.empty else [],
        # Per-feature non-null coverage per window (measured vs default-filled).
        # Visual backstop for silent data starvation -- see compute_feature_coverage.
        "feature_coverage": coverage_df.to_dict(orient="records")
                               if coverage_df is not None and not coverage_df.empty else [],
        # Rolling trailing-window Brier over decided OOF games (calibrated p).
        # See compute_rolling_brier; series is [] when history can't support
        # it and the frontend renders its empty state.
        "rolling_brier": (rolling_brier or {}).get("series", []),
        "brier_baseline": (rolling_brier or {}).get("history_mean_brier"),
        "brier_baseline_label": (
            f"History mean ({(rolling_brier or {}).get('n_games_total', 0)} games)"
            if rolling_brier and rolling_brier.get("history_mean_brier") is not None
            else "Baseline"
        ),
        "rolling_brier_meta": {
            "window_days": (rolling_brier or {}).get("window_days"),
            "min_games_per_day": (rolling_brier or {}).get("min_games_per_day"),
            "excluded_sparse_days": (rolling_brier or {}).get("excluded_sparse_days"),
            "calibrator_is_identity": (rolling_brier or {}).get("calibrator_is_identity"),
            "map_scope_note": (rolling_brier or {}).get("map_scope_note"),
        } if rolling_brier else {},
        # Rich per-feature metadata (definition/formula/source/window/units/
        # direction/derived members) for drift-table tooltips. One source of
        # truth generated from MONEYLINE_FEATURE_COLS -- see feature_metadata.py.
        "features_metadata": (features_metadata or {}).get("features", {}),
        # Run-engine Phase 3: per-market metrics, α(λ) params + fit-checks,
        # MC metadata, line-grid availability, agreement-filter stats.
        "run_engine": run_engine or {},
        # Candidate models behind the ensemble: name, blend weight (sums to
        # 1.0 over deployed members), and pooled out-of-fold AUC/Brier/LogLoss.
        "ensemble": ensemble if ensemble is not None else last_ensemble_info(),
        "model_history": history,
        # The Model Monitor page's Model Version History table reads this key.
        "version_history": version_history,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


# The 3 run-engine WINNER cards the monitor scores (binary pick framing,
# matching the Totals & Run Lines history tables).
_RUN_ENGINE_WINNER_CARDS = ("over_under", "run_line", "derived_ml")

# v1 per_line line -> v2 winner card, for rolling continuity during the
# v1->v2 monitor cutover. v1 cards carry base_rate (renamed actual_win_rate
# in v2); the rolling point never carried the rate, so continuity needs only
# this line->card map. Unmappable v1 entries are skipped (the renderer shows
# '--' for missing points — never a crash).
_RUN_ENGINE_V1_LINE_TO_CARD = {
    "over_under": "over_8_5",
    "run_line": "home_cover_1_5",
    "derived_ml": "derived_moneyline",
}

# How many recent daily points each card's rolling series keeps.
RUN_ENGINE_MONITOR_ROLLING_DAYS = 45


def _iso_from_ymd(ymd: str) -> str:
    """Normalize a YYYYMMDD stamp to an ISO YYYY-MM-DD date string."""
    s = str(ymd).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s  # already ISO (or unparseable -- safe to pass through)


def _run_engine_fit_block(block: Optional[dict]) -> dict:
    """Extract the distributional-fit block for the monitor from the daily
    run engine's monitor block (α per side, χ²/df, per-side observed-vs--modeled
    NB PMF tables incl. the ">=10"/"<=1" tail rows, variance check)."""
    if not block:
        return {}
    dispersion = (block.get("phase1") or {}).get("dispersion_ratio") or {}
    hg = block.get("holdout_gate") or {}
    return {
        "alpha_home": block.get("alpha_home"),
        "alpha_away": block.get("alpha_away"),
        # k_edge key dropped 2026-09-27 with the retired C2 expansion (the
        # monitor-only k-hat publication had no consumers).
        "dispersion_chi2_per_df": {
            "home": dispersion.get("home"),
            "away": dispersion.get("away"),
        },
        "fit_tables": (block.get("fit_check_alpha_lambda") or {}),
        "variance_check": block.get("variance_check"),
        "mc_meta": block.get("mc_meta"),
        "line_grid": block.get("line_grid"),
        "holdout_gate": {
            "cutoff": hg.get("cutoff"),
            "n_pre": hg.get("n_pre"),
            "n_holdout": hg.get("n_holdout"),
        },
    }


def _run_engine_phase1(block: Optional[dict]) -> dict:
    """Walk-forward geometry for the run-engine model card: fold count, games
    scored, per-side dispersion (chi2/df) and final-fit median rounds."""
    if not block:
        return {}
    p1 = block.get("phase1") or {}
    return {
        "n_folds": p1.get("n_folds"),
        "n_games": p1.get("n_games"),
        "dispersion_ratio": p1.get("dispersion_ratio") or {},
        "final_fit_rounds": p1.get("final_fit_rounds") or {},
    }


def _run_engine_monitor_json(
    block: Optional[dict],
    target_date_str: str,
    markets_persisted: bool,
    markets_persist_error: Optional[str],
) -> Path:
    """Write run_engine_monitor_YYYYMMDD.json -- the Run-Line & Totals Monitor.

    Schema (run-engine-monitor/v2). ``block`` is run_engine_daily's
    monitor-embed dict; the flags are its markets_persisted passthrough.

      winner_cards: {card: {n, actual_win_rate, win_rate, predicted_mean,
                            auc, ece_raw, ece_calibrated, brier, logloss,
                            holdout{...}}} for over_under / run_line /
                 derived_ml — the three binary WINNER cards (pick framing
                 matching the Totals & Run Lines history tables).
                 actual_win_rate (renamed from base_rate) is the empirical
                 pick win rate, push-excluded; win_rate is the same number
                 (picks correct at the >50% rule); predicted_mean is the
                 pooled PREQUENTIALLY-CALIBRATED favored-probability mean
                 shown beside it as one compact stat line; auc is the
                 FIXED-reference-line AUC (over_8_5 / home_cover_1_5 /
                 derived_ml — never a mixed-line rank); by_pick (run_line /
                 derived_ml only) splits n / win_rate / predicted_mean by
                 pick direction (home vs away) — every metric is on the
                 PICKED side, never home-side unconditionally. derived_ml is
                 the RUN LINE model's own NB moneyline (p_home_win_derived;
                 nb_diagnostic = the same model finding — underweights home
                 edge, reported as-is); the moneyline ENSEMBLE ml_win_prob
                 rides as a one-line ml_reference so the model comparison
                 stays visible.
      rolling:   {card: [{date, ece_calibrated, brier, logloss,
                         predicted_mean, n}]} cumulative-by-date series
                 folded from prior v2 monitor files (protected by the
                 run_engine_monitor_ prefix in _PROTECTED_DELIVERY_PREFIXES
                 so they survive cleanup), trimmed to the last 45 days.
                 First build is empty; the renderer must handle [].
      fit:       alpha_home/alpha_away (curve), dispersion chi2/df per side,
                 per-side fit tables (observed vs modeled NB PMF incl. the
                 ">=10"/"<=1" tail rows), mc meta, n_pre/n_holdout.
      market_metrics: per-line engine OOF metrics (logloss / brier / ECE
                 raw+calibrated / auc / n / holdout) for the run-engine
                 model card — emitted by run_engine_daily's summary.
      phase1:    walk-forward geometry (n_folds, n_games, per-side
                 dispersion_ratio chi2/df, final_fit_rounds median rounds).
      markets_persisted/markets_persist_error: passthrough -- the monitor
                 MUST say loudly when today's markets CSV did not persist
                 (never silently serve stale data).
    """
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"run_engine_monitor_{target_date_str}.json"

    winner_cards: dict[str, dict] = {}
    if block:
        wc = block.get("winner_cards") or {}
        for card in _RUN_ENGINE_WINNER_CARDS:
            c = wc.get(card)
            if not isinstance(c, dict):
                continue
            winner_cards[card] = {
                "n": int(c.get("n", 0)),
                "actual_win_rate": c.get("actual_win_rate"),
                "win_rate": c.get("win_rate"),
                "predicted_mean": c.get("predicted_mean"),
                "auc": c.get("auc"),
                "ece_raw": c.get("ece_raw"),
                "ece_calibrated": c.get("ece_calibrated"),
                "brier": c.get("brier"),
                "logloss": c.get("logloss"),
                "logloss_calibrated": c.get("logloss_calibrated"),
                "holdout": c.get("holdout"),
                "by_pick": c.get("by_pick"),
                "source": c.get("source"),
                "nb_diagnostic": c.get("nb_diagnostic"),
                "ml_reference": c.get("ml_reference"),
            }

    # Rolling per-card series (v2): fold prior monitor files — v2 files
    # contribute winner_cards directly; v1 files' per_line lines are mapped
    # onto the cards (over_8_5 -> over_under, home_cover_1_5 -> run_line,
    # derived_moneyline -> derived_ml) so the rolling history stays
    # continuous across the cutover. Files are protected by the
    # run_engine_monitor_ prefix so cleanup never deletes them. Append
    # today's point, dedupe by date, trim to RUN_ENGINE_MONITOR_ROLLING_DAYS.
    rolling: dict[str, list[dict]] = {ln: [] for ln in _RUN_ENGINE_WINNER_CARDS}
    try:
        by_line: dict[str, dict[str, dict]] = {
            ln: {} for ln in _RUN_ENGINE_WINNER_CARDS}
        if DATA_DELIVERY_DIR.exists():
            for p in DATA_DELIVERY_DIR.glob("run_engine_monitor_*.json"):
                if p.name == path.name:
                    continue
                try:
                    j = json.loads(p.read_text())
                except Exception:
                    continue
                fdate = _iso_from_ymd(str(j.get("date") or p.stem.replace(
                    "run_engine_monitor_", "")))
                for ln in _RUN_ENGINE_WINNER_CARDS:
                    pc = (j.get("winner_cards") or {}).get(ln)
                    if not isinstance(pc, dict):
                        v1_line = _RUN_ENGINE_V1_LINE_TO_CARD[ln]
                        pc = (j.get("per_line") or {}).get(v1_line)
                    if not isinstance(pc, dict):
                        continue  # unmappable entry -> renderer shows '--'
                    by_line[ln][fdate] = {
                        "date": fdate,
                        "ece_calibrated": pc.get("ece_calibrated"),
                        "brier": pc.get("brier"),
                        "logloss": pc.get("logloss"),
                        "predicted_mean": pc.get("predicted_mean"),
                        "n": int(pc.get("n", 0)),
                    }
        today_ymd = _iso_from_ymd(target_date_str)
        for ln in _RUN_ENGINE_WINNER_CARDS:
            pc = winner_cards.get(ln)
            if pc is None:
                continue
            by_line[ln][today_ymd] = {
                "date": today_ymd,
                "ece_calibrated": pc.get("ece_calibrated"),
                "brier": pc.get("brier"),
                "logloss": pc.get("logloss"),
                "predicted_mean": pc.get("predicted_mean"),
                "n": int(pc.get("n", 0)),
            }
        for ln in _RUN_ENGINE_WINNER_CARDS:
            series = sorted(by_line[ln].values(), key=lambda r: r["date"])
            rolling[ln] = series[-RUN_ENGINE_MONITOR_ROLLING_DAYS:]
    except Exception as e:
        logger.warning("Run-engine monitor: rolling fold skipped (%s)", e)

    data = {
        "schema": "run-engine-monitor/v2",
        "date": target_date_str,
        "markets_persisted": bool(markets_persisted),
        "markets_persist_error": markets_persist_error,
        "winner_cards": winner_cards,
        "rolling": rolling,
        "fit": _run_engine_fit_block(block),
        # Per-line engine OOF metrics (logloss/brier/ECE raw+calibrated, n,
        # auc, holdout) for the run-engine model card — emitted by
        # run_engine_daily's summary (market_* rows, keys stripped of the
        # prefix). Absent on locally-built bridge artifacts -> the renderer
        # shows its empty state, never fabricated numbers.
        "market_metrics": (block.get("market_metrics") or {}) if block else {},
        # Walk-forward geometry for the model card: n_folds, n_games,
        # per-side dispersion (chi2/df) and final-fit median rounds.
        "phase1": _run_engine_phase1(block),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    _nroll = len(next(iter(rolling.values()))) if rolling else 0
    logger.info("Run-engine monitor: %d winner cards, rolling %d days -> %s",
                len(winner_cards), _nroll, path.name)
    return path


def auto_version(target_date: date) -> str:
    """Date-stamped model version (e.g. ``v2026.08.23``).

    Every retrain is visibly distinct in the Model Monitor version history;
    pass an explicit --version to override.
    """
    return f"v{target_date.strftime('%Y.%m.%d')}"


# Trailing window (days) of decided games that receive real point-in-time
# weather.  Must cover BOTH drift windows: current (~7 days) and baseline
# (>= 250 games ≈ ~17 days), so PSI compares like-for-like coverage.
WEATHER_BACKFILL_DAYS = 35


def _attach_recent_weather(games: pd.DataFrame, target_date: date) -> pd.DataFrame:
    """Attach point-in-time weather features to recent decided games.

    Statcast history carries only fabricated 19:00-UTC start placeholders,
    so first fetch each game's REAL first pitch from StatsAPI; weather is
    then sampled strictly before it (Open-Meteo archive, with a recent-past
    forecast fallback and climatology as last resort).  Without this,
    wind/air-density features are null for all history except dome zeros --
    collapsing their drift sample to ~1/3 of the other features.
    """
    from results import fetch_game_start_times
    from weather import apply_weather_features, fetch_games_weather

    if games.empty:
        return games
    gd = pd.to_datetime(games["game_date"], errors="coerce")
    decided = games["home_win"].notna() if "home_win" in games.columns else pd.Series(True, index=games.index)
    start = pd.Timestamp(target_date) - pd.Timedelta(days=WEATHER_BACKFILL_DAYS)
    end = pd.Timestamp(target_date)
    mask = decided & (gd >= start) & (gd < end)
    subset = games[mask]
    if subset.empty:
        logger.info("Weather backfill: no decided games in trailing %dd window", WEATHER_BACKFILL_DAYS)
        return games

    # Real first pitches keyed by StatsAPI game_pk (the same authoritative
    # identifier the results overlay uses).  Rows without one -- or without a
    # matching official start -- are skipped: never sample at a fabricated hour.
    starts = fetch_game_start_times(subset["game_date"].min().date(),
                                    subset["game_date"].max().date())

    rows = []
    row_idx = []
    matched = 0
    for idx, r in subset.iterrows():
        pk = pd.to_numeric(r.get("game_pk"), errors="coerce")
        st = starts.get(int(pk)) if pd.notna(pk) else None
        if not st:
            continue
        matched += 1
        ts = pd.Timestamp(st)
        rows.append({
            "home_team": r.get("home_team"),
            "venue": r.get("venue", ""),
            "start_time_utc": ts.tz_localize(None) if ts.tzinfo is not None else ts,
        })
        # Preserve the source index label: both fetch_games_weather and
        # apply_weather_features key results by it when game_id is absent.
        row_idx.append(idx)
    if not rows:
        logger.warning("Weather backfill: no authoritative start times matched -- skipped")
        return games

    wx_df = pd.DataFrame(rows, index=row_idx)
    wx = fetch_games_weather(wx_df)
    logger.info("Weather backfill: %d/%d decided games matched to real starts",
                matched, len(subset))
    return apply_weather_features(games, wx)


# ── Full-history weather backfill (cache-backed) ───────────────────────────

def _weather_cache_path() -> Path:
    """Persistent per-game weather cache.

    Lives outside the git repo in Colab (MLB_CACHE_DIR points at the
    /content/mlb_clean_data cache dir); falls back to data_delivery locally.
    """
    base = os.getenv("MLB_CACHE_DIR") or str(DATA_DELIVERY_DIR)
    return Path(base) / "weather_history.parquet"


_WEATHER_CACHE_COLS = [
    "available", "source", "temp_c", "rh_pct", "wind_speed_kmh",
    "wind_direction_deg", "pressure_hpa", "air_density", "wind_multiplier",
    "stadium_alt_m", "stadium_bearing",
]
_OBSERVED_WEATHER_SOURCES = {
    "open_meteo_archive",
    "open_meteo_forecast_past",
    "noaa_isd",
    # Official park-reported conditions (gameData.weather). Real observation,
    # but only wind fills honestly -- the feed has no humidity, so air_density
    # stays NULL for these records by the module's no-fabrication rule.
    "statsapi_gamefeed",
}

# Fill games the Open-Meteo archive could not observe from the per-game
# StatsAPI feed (paced ~2.5 req/s, one-off per cached game_pk).
STATSAPI_WEATHER_FILL = True

# Lineup feed pacing (mirrors the roof-fetcher budget ~2.2 req/s).
_LINEUP_PAUSE_SEC = 0.45


def _load_weather_cache(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Weather cache unreadable (%s) -- rebuilding", exc)
        return {}
    if "source" not in df.columns:
        # Legacy caches predate provenance and may contain climatology values
        # marked available=True. Never reuse them as observed weather.
        logger.warning("Weather cache has no source column -- invalidating legacy cache")
        return {}
    out: dict[int, dict] = {}
    for _, r in df.iterrows():
        source = None if pd.isna(r.get("source")) else str(r.get("source"))
        available = r.get("available")
        if (
            source not in _OBSERVED_WEATHER_SOURCES
            or pd.isna(available)
            or not bool(available)
        ):
            continue
        pk = int(r["game_pk"])
        out[pk] = {k: (None if pd.isna(r.get(k)) else r.get(k))
                   for k in _WEATHER_CACHE_COLS}
    return out


def _save_weather_cache(path: Path, cache: dict[int, dict]) -> None:
    if not cache:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"game_pk": pk, **w} for pk, w in cache.items()]
    pd.DataFrame(rows).to_parquet(path, index=False)
    logger.info("Weather cache saved: %d games → %s", len(cache), path)


def _attach_weather_history(games: pd.DataFrame, target_date: date) -> pd.DataFrame:
    """Attach real point-in-time weather to EVERY decided game.

    Real StatsAPI first pitches → strictly-prior Open-Meteo archive
    observation per (stadium, day), cached by game_pk so each run fetches
    only games missing from the cache. Without this, the wind/air-density
    features stayed null for ~93% of history (only dome zeros) and the two
    weather features were starved. Runs AFTER add_diff_features so the
    sp_*_diff inputs exist for apply_weather_features.
    """
    from results import fetch_game_start_times
    from weather import apply_weather_features, fetch_games_weather

    if games.empty:
        return games
    pks = pd.to_numeric(games.get("game_pk"), errors="coerce")
    decided = (games["home_win"].notna()
               if "home_win" in games.columns else pd.Series(True, index=games.index))
    cache = _load_weather_cache(_weather_cache_path())
    cached_pks = set(cache)

    # Avoid casting the full nullable series to int: non-authoritative rows
    # may legitimately have no game_pk.
    pks_int = pks.where(pks.notna()).astype("Int64")
    # Partial-record top-up (2026-10-09 log review). The cache is one slot
    # per game_pk and `need` below excluded ANY cached pk — so the wind-only
    # records the StatsAPI gap filler writes (the official feed carries no
    # humidity, hence no air_density) satisfied the gate FOREVER and the
    # complete Open-Meteo observation was never retried. That is exactly how
    # 849851/849844/849838/813022 kept air_density_velocity_boost NULL after
    # the archive had published their hours: a partial observation must not
    # outrank a complete one. Re-attempt ONLY cached records that still lack
    # an air-density observation; a retry that finds nothing keeps the
    # partial (only available records are ever written back below).
    def _lacks_air(rec: dict) -> bool:
        ad = rec.get("air_density")
        try:
            return ad is None or bool(pd.isna(ad))
        except (TypeError, ValueError):
            return True

    partial_pks = {pk for pk, rec in cache.items() if _lacks_air(rec)}
    need = (decided & pks.notna()
            & (~pks_int.isin(cached_pks) | pks_int.isin(partial_pks)))
    if partial_pks:
        logger.info(
            "Weather top-up: %d cached record(s) carry no air-density "
            "observation (official wind-only fills) — re-attempting the "
            "complete observation for them", len(partial_pks))
    if need.any():
        subset = games[need]
        gd = pd.to_datetime(games["game_date"], errors="coerce")
        starts = fetch_game_start_times(gd[need].min().date(), gd[need].max().date())
        # Loud coverage gate: a silently truncated schedule source is how an
        # ENTIRE SEASON of weather features went null while every log line
        # looked healthy (2470/2477 'fetched' -- of only the games attempted).
        # Checked PER CALENDAR YEAR because the failure was season-specific:
        # 2025 matched 100% while 2026 matched ~1%. An aggregate ratio over
        # both years would have diluted the dead season into a single pass.
        need_pks_all = [int(p) for p in pks[need].dropna()]
        need_years = gd[need].dt.year
        for year in sorted(need_years.dropna().unique()):
            yr_mask = (need_years == year).values
            yr_pks = [int(p) for p, m in zip(pks[need], yr_mask) if m]
            matched_yr = sum(1 for pk in yr_pks if pk in starts)
            pct = 100.0 * matched_yr / len(yr_pks) if yr_pks else 100.0
            logger.info("Weather history: start times %s: %d/%d (%.0f%%)",
                        year, matched_yr, len(yr_pks), pct)
            if yr_pks and matched_yr < 0.8 * len(yr_pks):
                logger.warning(
                    "Weather history: start times matched only %d/%d decided "
                    "games in %d (%s→%s) -- schedule source may be truncating "
                    "or failing; open-air weather stays NULL for unmatched games",
                    matched_yr, len(yr_pks), int(year),
                    gd[need][yr_mask].min().date(), gd[need][yr_mask].max().date())
        rows: list[dict] = []
        row_idx: list[Any] = []
        for idx, r in subset.iterrows():
            pk = int(pks.loc[idx])
            st = starts.get(pk)
            if not st:
                continue
            ts = pd.Timestamp(st)
            rows.append({
                "game_id": r.get("game_id"),
                "game_pk": pk,
                "home_team": r.get("home_team"),
                "venue": r.get("venue", ""),
                "start_time_utc": ts.tz_localize(None) if ts.tzinfo is not None else ts,
            })
            row_idx.append(idx)
        if rows:
            wx_df = pd.DataFrame(rows, index=row_idx)
            wx = fetch_games_weather(wx_df)
            # Results are keyed by game_pk whenever available. Keep the
            # game_id/index aliases for mocked or older providers so a cache
            # refresh remains backward-compatible without weakening the
            # authoritative game_pk contract.
            key_to_pk: dict[object, int] = {}
            for row_idx, r in wx_df.iterrows():
                pk = int(r["game_pk"])
                key_to_pk[pk] = pk
                key_to_pk[str(pk)] = pk
                gid = r.get("game_id")
                if gid is not None and not (isinstance(gid, float) and pd.isna(gid)):
                    key_to_pk[gid] = pk
                    key_to_pk[str(gid)] = pk
                key_to_pk[row_idx] = pk
                key_to_pk[str(row_idx)] = pk
            new = 0
            completed = 0
            for result_key, w in wx.items():
                pk = key_to_pk.get(result_key)
                if pk is None:
                    pk = key_to_pk.get(str(result_key))
                if pk is None:
                    continue
                if (
                    w.get("available")
                    and w.get("source") in _OBSERVED_WEATHER_SOURCES
                ):
                    if pk in partial_pks and not _lacks_air(w):
                        completed += 1
                    cache[pk] = {k: w.get(k) for k in _WEATHER_CACHE_COLS}
                    new += 1
            _save_weather_cache(_weather_cache_path(), cache)
            logger.info("Weather history: fetched %d new games (cache now %d)",
                        new, len(cache))
            if partial_pks:
                logger.info(
                    "Weather top-up: %d/%d partial record(s) completed with "
                    "a full observation; the rest keep their official "
                    "wind-only fill (air density stays NULL — never "
                    "fabricated)", completed, len(partial_pks))
        else:
            logger.warning("Weather history: no authoritative start times matched")

        # Gap filler: the per-game feed needs neither coordinates nor a
        # first-pitch time, so it also reaches games skipped above for lack
        # of a start time. Only decided OPEN-AIR uncached games are targeted;
        # domes legitimately carry default-zero wind and NULL density.
        if STATSAPI_WEATHER_FILL:
            gap_rows = []
            for idx, r in subset.iterrows():
                if pd.to_numeric(r.get("dome_is_neutral"), errors="coerce") == 1:
                    continue
                pk = int(pks.loc[idx])
                if pk not in cache:
                    gap_rows.append((pk, r.get("home_team"), str(r.get("venue", "") or "")))
            if gap_rows:
                from results import fetch_statsapi_weather
                from weather import statsapi_weather_to_record
                feed_wx = fetch_statsapi_weather([g[0] for g in gap_rows])
                filled = 0
                for pk, home_team, venue in gap_rows:
                    parsed = feed_wx.get(pk)
                    if not parsed:
                        continue
                    rec = statsapi_weather_to_record(parsed, home_team, venue)
                    if rec.get("available"):
                        cache[pk] = {k: rec.get(k) for k in _WEATHER_CACHE_COLS}
                        filled += 1
                logger.info(
                    "StatsAPI weather filler: %d/%d gap games recovered from "
                    "official park observations", filled, len(gap_rows))
                if filled:
                    _save_weather_cache(_weather_cache_path(), cache)

    # Apply using the authoritative game_pk key. apply_weather_features also
    # accepts game_id/index aliases for slate and legacy frames.
    by_pk: dict[int, dict] = {}
    for idx, r in games.iterrows():
        pk = pks.loc[idx]
        if pd.isna(pk):
            continue
        w = cache.get(int(pk))
        if w is not None:
            by_pk[int(pk)] = w
    out = apply_weather_features(games, by_pk)
    n_ok = sum(
        1 for pk in pks.dropna().astype(int).unique()
        if pk in cache and cache[pk].get("source") in _OBSERVED_WEATHER_SOURCES
    )
    logger.info("Weather history: %d/%d games with observed weather (cache)",
                n_ok, len(games))
    return out


def run_daily_pipeline(
    target_date: date,
    real: bool = False,
    skip_sync: bool = False,
    force_retrain: bool = False,
    max_eval_folds: int = 0,
    version: Optional[str] = None,
    games: Optional[pd.DataFrame] = None,
    min_train_days: int = 0,
    pbp_df: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Run the full daily pipeline.

    Args:
        target_date: Date to generate predictions for.
        real: Use pybaseball real data (default: synthetic).
        skip_sync: Skip GitHub push.
        force_retrain: Retrain regardless of cadence.
        max_eval_folds: Cap walk-forward folds (0 = full history).
        version: Model version string (default: auto, ``vYYYY.MM.DD`` of
               the target date).
        games: Pre-built game DataFrame (from features.py). When provided,
               skips load_game_events() and uses this data for training.
        min_train_days: Warm-up period -- skip validation folds that start
               before this many days of history (prevents tiny-training-fold noise).
        pbp_df: Optional pitch-level frame used to map probable-pitcher names
               to their rolling stat lines when predicting today's slate.

    Returns:
        Summary dict with keys: status, artifacts, metrics, sync, errors
    """
    target_date_str = target_date.strftime(DATE_FMT)
    if not version:
        version = auto_version(target_date)
    logger.info("=== Daily pipeline for %s (model %s) ===", target_date_str, version)

    summary: dict[str, Any] = {
        "status": "ok",
        "target_date": target_date_str,
        "artifacts": [],
        "metrics": {},
        "sync": None,
        "errors": [],
    }

    try:
        # 1. Ingest game events
        if games is not None and not games.empty:
            logger.info("Step 1: Using pre-built game features (%d games)", len(games))
        else:
            logger.info("Step 1: Loading game events (real=%s)", real)
            games = load_game_events(target_date, real=real)
        if games.empty:
            summary["status"] = "error"
            summary["errors"].append("No game events loaded")
            return summary

        logger.info("Loaded %d games", len(games))

        # Official-results overlay (Step 1.5).  Authoritative scores and
        # finality from StatsAPI: corrects frozen mid-game finals
        # retroactively and NULLS any home_win attached to a game that is
        # not officially final -- a partial score can never ship as a final
        # (the same guarantee features.build_features provides).
        try:
            from results import apply_official_results, fetch_mlb_results
            _d = pd.to_datetime(games.get("game_date"), errors="coerce").dropna()
            if len(_d):
                _res = fetch_mlb_results(_d.min().date(), _d.max().date())
                if not _res.empty:
                    games = apply_official_results(games, _res)
        except Exception as exc:
            logger.warning("Official results overlay failed on history: %s", exc)

        # Authoritative first pitches for fabricated history rows (the
        # 19:00-UTC fallback). Real times fix within-day PIT ordering,
        # evening-game counts, line as-of joins, and let the observed-time
        # weather gate below cover history instead of staying permanently
        # dead behind start_time_observed=False. Best-effort: weather paths
        # already fetch real times on demand when this fails.
        try:
            from results import refresh_start_times
            games = refresh_start_times(games)
        except Exception as exc:
            logger.warning("Historical start-time refresh failed "
                           "(fabricated placeholders retained): %s", exc)


        # Real point-in-time weather for features 30--31 (wind advantage,
        # air density).  One Open-Meteo request per (stadium, day); games
        # without a strictly-prior observation get NULL weather features
        # (never a fabricated 0).  Weather is only attached when the frame
        # carries GENUINELY observed start times -- fabricated defaults (e.g.
        # load_game_features' 19:00 UTC fallback) are excluded via the
        # start_time_observed tag so we never fetch weather for the wrong
        # hour.
        # ALWAYS recompute every diff feature from the raw home/away columns.
        # Pre-built exports may contain column names with stale or
        # schema-drifted values (e.g. win_pct_diff NaN from the DuckDB first
        # pass, renamed pitcher windows, weather defaults) -- presence of a
        # column is never evidence its values are current. Recomputation is a
        # cheap vectorized pass and runs exactly once, BEFORE any weather
        # application so the two weather-driven features are applied on top
        # of fresh diffs afterwards.
        logger.info("Recomputing all diff features from raw home/away columns")
        # Final computation: official results already applied, so record
        # columns must exist -- a missing win_pct_diff here is a real problem.
        games = add_diff_features(games, require_records=True)
        # Momentum form deltas (recent − season-to-date baseline). Idempotent:
        # SQL-shipped columns win; missing ones are computed from the shipped
        # recent/season columns when both exist (NaN otherwise -- imputed by
        # the existing paths). Moneyline-only: they are excluded from
        # MONEYLINE_FEATURE_COLS (the run engine's served view).
        games = add_form_delta_features(games)
        # Phase 2 lineup deltas REMOVED 2026-09-26: the six lineup_actual_*/
        # lineup_rest_count_* columns left MONEYLINE_FEATURE_COLS on
        # 2026-08-29 (training.py) as a train-serve skew fix. This
        # require_caches=True call survived that and was the only thing
        # keeping the feature alive -- it hard-failed the whole training run
        # when a fresh clone lacked the committed caches, over columns no
        # model reads. See the slate-path comment above for the full rationale.
        if WEATHER_BACKFILL_ALL:
            # Full-history weather mode: the cache-backed backfill applies
            # real point-in-time weather to every decided game (see
            # _attach_weather_history) -- no reliance on the trailing window.
            # Diff reconstruction uses only observed density levels. Preserve
            # them if the cache/provider only returns a partial history.
            try:
                games = _attach_weather_history(games, target_date)
            except Exception as exc:
                logger.warning(
                    "Full weather backfill failed (prior observed levels retained): %s", exc
                )
                try:
                    games = _attach_recent_weather(games, target_date)
                except Exception:
                    pass
        else:
            # Apply fresh observations without erasing level-backed weather
            # already reconstructed by add_diff_features.
            if "start_time_utc" in games.columns:
                real_start = games["start_time_utc"].notna()
                if "start_time_observed" in games.columns:
                    # Unknown provenance is NOT observed (load_game_features'
                    # fabricated fallback must never be fetched for the wrong
                    # hour), and rows already carrying a wind observation are
                    # not re-fetched every run.
                    real_start &= games["start_time_observed"].fillna(False).astype(bool)
                if "wind_advantage_flyball_factor" in games.columns:
                    real_start &= pd.to_numeric(
                        games["wind_advantage_flyball_factor"],
                        errors="coerce").isna()
                if real_start.any():
                    weather = {}
                    try:
                        weather = fetch_games_weather(games.loc[real_start])
                    except Exception as e:
                        logger.warning(
                            "Weather fetch failed for history (prior observed levels retained): %s", e
                        )
                    # apply_weather_features is imported at module level; do NOT
                    # re-import it here -- a branch-local binding makes the name
                    # function-local and crashes the slate path below with
                    # UnboundLocalError when this branch never ran.
                    if weather:
                        games = apply_weather_features(games, weather)

            # Weather backfill over decided history: real StatsAPI first pitches →
            # strictly-prior observations for the trailing drift window (see
            # _attach_recent_weather).  Runs AFTER add_diff_features so the
            # sp_*_diff inputs exist; applied values survive because this is the
            # last writer of the two weather-driven columns.
            try:
                games = _attach_recent_weather(games, target_date)
            except Exception as exc:
                logger.warning("Weather backfill failed (features stay null): %s", exc)

            # Re-export the feature frame now that weather has been applied, so
        # the shipped game_level_features.csv matches the exact features the
        # models trained on (the Phase-3.5 export runs before any weather
        # pass and would otherwise ship dome-default zeros/nulls only).
        # Phase-3.5b first: game-accurate roof flag + standalone env-LEVEL
        # columns, ADDITIVE to the venue-level dome flag and the interaction
        # features. Roof state comes from the StatsAPI cache; unknown
        # retractable games fall back LOUDLY inside refine_dome_game_level,
        # never silently treated as closed.
        try:
            roof_cache = DATA_DELIVERY_DIR / "statsapi_roof_cache.json"
            roof_states = _load_roof_cache(roof_cache)
            # Auto top-up: fetch missing retractable-home game_pks from
            # the StatsAPI live feed (best-effort, budget-capped).
            roof_states = _topup_roof_cache(
                games, roof_states, roof_cache, budget_sec=128.0)
            games = refine_dome_game_level(games, roof_states=roof_states)
            games = add_env_level_features(games)
        except Exception as exc:
            logger.warning("Env-level feature pass failed (level columns may "
                           "be absent this run): %s", exc)

        # Canonical decided frame snapshot -- captured ONCE after official
        # results AND after the weather + env-level feature passes, so the
        # drift/coverage/run-engine consumers see the SAME 65-feature view
        # training used. (The 08-28 regression: a Step-1.5 snapshot predated
        # those attaches and lost wind_advantage_flyball_factor,
        # air_density_velocity_boost + the 4 env-level columns -- the drift
        # table dropped 65->59 and run-engine warned 4/4 env columns absent.)
        # Still BEFORE market lines and the Step-4 slate merge so the decided
        # ROW SET is frozen before any row-mutating step; the diff/weather/
        # env passes only attach columns (none write home_win), so this
        # capture point is row-identical -- same fold signature -- as the
        # old one. Every consumer (training, drift/coverage, run-engine)
        # uses this exact snapshot so frame mutations later (slate concat,
        # official results on target_games) cannot create a fold-signature
        # desync between training and drift.
        _decided_snapshot = get_decided_frame(games)
        # Full generation universe, even when a governed serving subset is
        # active. Schema absence / wholly starved history is not acceptable
        # as a successful production rebuild. Individual warm-null rows remain
        # honest missing observations and are not zero-filled.
        _absent = [c for c in MONEYLINE_FEATURE_COLS if c not in _decided_snapshot]
        _starved = [c for c in MONEYLINE_FEATURE_COLS if c in _decided_snapshot
                    and not pd.to_numeric(_decided_snapshot[c], errors="coerce").notna().any()]
        if _absent or _starved:
            raise ValueError(f"MLB feature coverage contract failed: missing={_absent}, "
                             f"entirely_unobserved={_starved}; repair sources before training")
        # Training-frame integrity tripwire, checked on the RAW frame:
        # get_decided_frame dedups by game_pk (Rule 3), so a fanned frame
        # is silently collapsed there — keeping whichever duplicate copy
        # sorts last, which after the daily Elo/records enrichment is the
        # LEAKY copy (its win_pct/elo already carry the game's own outcome;
        # 2026-09-30: bp_fatigue's bp_day2 join keyed on (ref_day, team)
        # fanned every same-opponent doubleheader x4 and Rule 3 kept the
        # post-game-Elo copy, costing ~+0.0059 walk-forward logloss vs the
        # pre-bundle baseline). Fail loudly at build time instead.
        _decided_raw = games[games["home_win"].notna()] \
            if "home_win" in games.columns else games
        if "game_pk" in _decided_raw.columns:
            _pk_series = _decided_raw["game_pk"]
            _dup_pks = _pk_series.notna() & _pk_series.duplicated(keep=False)
        else:
            _dup_pks = pd.Series(dtype=bool)
        if bool(_dup_pks.any()):
            _fan_n = int(_dup_pks.sum())
            _fan_pks = (_decided_raw.loc[_dup_pks, "game_pk"]
                        .drop_duplicates().head(5).tolist())
            raise RuntimeError(
                f"decided frame has {_fan_n} duplicated game_pk rows "
                f"(e.g. {_fan_pks}) — an upstream enrichment join fanned "
                "the training frame; refusing to train on it")
        import hashlib as _hb
        _snap_pks = _decided_snapshot["game_pk"].tolist() if "game_pk" in _decided_snapshot.columns else []
        _snap_hash = _hb.sha256("|".join(str(p) for p in _snap_pks).encode()).hexdigest()[:12]
        logger.info("Pipeline snapshot: %d decided games, hash=%s", len(_decided_snapshot), _snap_hash)

        try:
            _w_cov = int(games["wind_advantage_flyball_factor"].notna().sum()) if "wind_advantage_flyball_factor" in games else 0
            _a_cov = int(games["air_density_velocity_boost"].notna().sum()) if "air_density_velocity_boost" in games else 0
            logger.info(
                "Re-exporting features CSV with applied weather "
                "(wind coverage %d/%d, air-density %d/%d)",
                _w_cov, len(games), _a_cov, len(games),
            )
            games.to_csv(DATA_DELIVERY_DIR / "game_level_features.csv", index=False)
            logger.info("Refreshed game_level_features.csv with applied weather")
        except Exception as exc:
            logger.warning("Could not refresh game_level_features.csv: %s", exc)

        # 1.9 Umpire map maintenance (maintained data access, NOT a model
        # feature — the runs-tendency scoping verdict forbids wiring it into
        # the run engine or moneyline; see umpires.py docstring). Incremental:
        # only seasons missing from the cumulative map are fetched. Fail-safe.
        try:
            _ump = maintain_umpire_map(target_date, required_games=games)
            logger.info(
                "Umpire map: %d games, %d umpires "
                "(fetched seasons %s, gap-filled %s, rows added %d)",
                _ump.get("total_rows", 0), _ump.get("n_umpires", 0),
                ",".join(_ump.get("seasons_fetched", [])) or "none",
                ",".join(_ump.get("gap_filled_seasons", [])) or "none",
                _ump.get("rows_added", 0),
            )
            build_umpire_stats(load_umpire_map(), games)
        except Exception as exc:
            logger.warning(
                "Umpire map maintenance failed (map stays as-is): %s", exc
            )

        # 2. Generate/attach market lines
        logger.info("Step 2: Generating market lines")
        lines = generate_synthetic_market_lines(games)
        games = attach_market_lines(games, lines)

        # 3. Walk-forward evaluation + training
        logger.info("Step 3: Walk-forward evaluation")
        # Use games up to target_date for training
        train_games = games.copy()

        # Check if we need to retrain
        ensemble = load_ensemble()
        need_retrain = force_retrain or should_retrain(None)  # Always train on first run

        if need_retrain:
            # Apply the adopted feature-selection subset (if any) so this
            # retrain trains AND persists at the governed width — the state
            # file is the record-only RFE contract's only lever on serving
            # (feature_selection.py; adoption is explicit, via --adopt).
            # Universe wins on any problem; never blocks the daily board.
            try:
                from feature_selection import apply_adopted_subset
                summary["feature_selection"] = apply_adopted_subset()
            except Exception as _fs_exc:
                logger.warning(
                    "Feature-selection state apply failed (universe width): %s",
                    _fs_exc)
            best_models, pooled_metrics, all_predictions = walk_forward_evaluate(
                train_games,
                max_eval_folds=max_eval_folds,
                force_retrain=force_retrain,
                min_train_days=min_train_days,
                decided_snapshot=_decided_snapshot,
            )
            logger.info("Walk-forward metrics (causal rolling blend): %s", pooled_metrics)
            RUN_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
            all_predictions.to_csv(RUN_DIAGNOSTICS_DIR / "mlb_oof_moneyline.csv", index=False)

            # Persist ensemble
            persist_ensemble(best_models, pooled_metrics, version=version, data_cutoff=target_date_str)
            update_model_history(
                pooled_metrics, version,
                notes=f"walk-forward through {target_date_str} ({len(train_games)} games)",
            )
            # Version-history snapshot: only after the ensemble persisted
            # cleanly, with the run's own roster + deployed map (no partials).
            update_model_version_history(
                pooled_metrics, version,
                ensemble_info=last_ensemble_info(),
                calibrator=get_last_calibrator(),
            )
            summary["metrics"] = pooled_metrics
            summary["evaluation"] = get_last_season_split()
        else:
            best_models = ensemble["models"] if ensemble else {}
            pooled_metrics = ensemble["metrics"] if ensemble else {}
            all_predictions = None  # cached model: no fresh OOF predictions
            set_adaptive_weights(ensemble.get("adaptive_weights"))
            set_calibration(ensemble.get("calibrator"))
            # Serving width follows the BUNDLE's recorded fit width (not the
            # current state file) — a cached bundle must be predicted with
            # exactly the features it was trained on, even if the adopted
            # subset changed since its retrain day.
            apply_bundle_feature_cols(ensemble)
            summary["metrics"] = pooled_metrics

        # 4. Predict today's games (target_date only)
        logger.info("Step 4: Predicting games for %s", target_date_str)
        target_games = games[
            pd.to_datetime(games["game_date"]).dt.date == target_date
        ].copy()

        if target_games.empty:
            # Statcast-derived history ends at the last PLAYED game, so on a
            # normal pre-game run there are zero rows for today. Build today's
            # real schedule with each team/pitcher's latest point-in-time
            # state carried forward -- never recycle yesterday's completed
            # games as "today" again.
            slate = build_upcoming_slate(games, target_date, pbp_df=pbp_df)
            if not slate.empty:
                logger.info(
                    "No completed games on %s -- built %d-game upcoming slate "
                    "(pre-game PIT features)", target_date_str, len(slate),
                )
                # Fetch point-in-time weather for the slate, then compute diff features
                weather = {}
                try:
                    weather = fetch_day_weather(slate)
                except Exception as e:
                    logger.warning("Weather fetch failed (features will use neutral defaults): %s", e)
                # Build diffs first, then apply weather through the shared
                # game_pk/game_id-aware applicator. This keeps missing weather
                # NULL and avoids a second key contract in add_diff_features.
                slate = add_diff_features(slate)
                slate = add_form_delta_features(slate)
                # Experiment #2 candidates (C+E/D+F): same arithmetic as the
                # decided frame so train and serve share one construction.
                # Slate rows missing a source column ship NaN, like every
                # other feature (never a fabricated 0).
                slate = add_exp2_features(slate)
                # Lineup-delta enrichment REMOVED 2026-09-26: the six
                # lineup_actual_*/lineup_rest_count_* columns were cut from
                # MONEYLINE_FEATURE_COLS on 2026-08-29 (training.py) as a
                # train-serve skew fix -- populated from post-game ACTUAL
                # lineups in the decided frame but always NULL at bet time --
                # so nothing scored them. The feature stayed live only as a
                # require_caches=True hard gate: a fresh clone missing
                # lineups.parquet / batter_woba.parquet / team_woba.parquet
                # failed the whole daily run over six unscored columns.
                # build_batter_woba.py is deleted; lineups.parquet survives as
                # the declared batting order, still the input any correct
                # re-implementation (projected lineups on BOTH sides) needs.
                # _fetch_slate_lineups / _attach_slate_lineup_keys /
                # _nearest_slate_pk stay defined (the former two are covered by
                # test_run_engine_contract.py) but are no longer on the run path.
                if weather:
                    slate = apply_weather_features(slate, weather)
                games = pd.concat([games, slate], ignore_index=True)
                target_games = slate.copy()
            else:
                # OFF-DAY HONESTY (2026-09-28 incident): a failed/empty
                # schedule fetch on a no-games day used to recycle the most
                # recent DECIDED games as "today's" slate — re-pricing games
                # the OOF already scored (duplicate keys crashed the markets
                # artifact contract: 6884 vs 6869) and pointing SHAP at final
                # scores (zero attributions). An empty board is the honest
                # result: moneyline + run-engine + SHAP consumers all handle
                # a 0-row slate, and the artifacts ship with an empty board
                # instead of phantom predictions.
                logger.warning(
                    "No games found for %s (schedule fetch empty) -- shipping "
                    "an EMPTY board (genuine off-day or schedule-source "
                    "outage; never recycling decided games as today's slate)",
                    target_date_str,
                )
                target_games = games.tail(0).copy()

        # ESPN drops probablePitcher once games start -- restore the pitching
        # matchup and lines published by an earlier same-day run before they
        # get overwritten.
        target_games = _carry_forward_slate_details(target_games, target_date_str)

        # BOARD-DATE POSTCONDITION (2026-09-28 regression): the priced board
        # must contain ONLY games dated target_date. The polluting run
        # shipped 15 finals from 0925/0926 under a September 28 header (a
        # stale clone ran pre-ab0e9c7 code); every downstream consumer —
        # the accuracy badge, SHAP, the markets artifact contract — trusts
        # this date filtering. Fail loudly instead of shipping a mislabeled
        # board. build_upcoming_slate already filters its schedule input;
        # this is the backstop for every other path into target_games.
        if not target_games.empty:
            from data_ingestion import enforce_board_date_invariant
            _kept = enforce_board_date_invariant(
                target_games, target_date, what="priced board")
            if len(_kept) != len(target_games):
                raise AssertionError(
                    f"board-date invariant violated: "
                    f"{len(target_games) - len(_kept)} of {len(target_games)} "
                    f"board rows are not dated {target_date_str} — refusing "
                    "to price recycled/foreign games as today's slate "
                    "(the 2026-09-28 regression)")

        # Official-results overlay on today's board.  Slate rows carry no
        # StatsAPI game_pk, so the overlay falls back to (date + teams).
        # Live/preview games get home_win=NULL; finals get authoritative
        # scores -- never a mid-game snapshot as a final.
        try:
            from results import apply_official_results, fetch_mlb_results
            _d = pd.to_datetime(target_games.get("game_date"), errors="coerce").dropna()
            if len(_d):
                _res = fetch_mlb_results(_d.min().date(), _d.max().date())
                if not _res.empty:
                    target_games = apply_official_results(target_games, _res)
                    if all_predictions is not None and len(all_predictions):
                        all_predictions = apply_official_results(all_predictions, _res)
        except Exception as exc:
            logger.warning("Official results overlay failed on slate: %s", exc)

        # Shipped run-margin feature: attach the slate margins (fit-only
        # refit on all decided games at the walk-forward median round count)
        # so the moneyline board predicts with the same feature the model
        # trained on. Never lets a margin from a model that saw the game in.
        try:
            target_games = _attach_slate_run_margins(target_games, games)
        except Exception as exc:
            logger.error(
                "run_margin_diff slate attach failed (%s) -- margin stays "
                "all-NaN (imputed by existing paths), prediction continues", exc)

        target_games = predict_games(best_models, target_games)

        # 5. Write artifacts
        logger.info("Step 5: Writing artifacts")

        # todays_games CSV
        path = _today_games_csv(target_games, target_date_str)
        summary["artifacts"].append(str(path))

        # power rankings
        path = _power_rankings_csv(games, target_date_str)
        summary["artifacts"].append(str(path))

        # calibration JSON -- written on EVERY run.  The walk-forward OOF frame
        # carries thousands of PIT-safe predicted-vs-actual pairs regardless
        # of whether tonight's slate has finished, so a pre-game-only run must
        # still ship a fresh artifact (Phase 6 prunes stale calibration files,
        # so skipping the write would leave the Calibration page empty).
        oof_ok = (
            all_predictions is not None
            and len(all_predictions) > 0
            and "home_win_prob_model" in all_predictions.columns
        )
        rolling_brier: Optional[dict] = None
        features_metadata: Optional[dict] = None
        day_final = (
            "home_win" in target_games.columns
            and "home_win_prob_model" in target_games.columns
            and target_games["home_win"].notna().any()
        )
        if oof_ok or day_final:
            if day_final:
                y_true = target_games["home_win"].dropna().values
                y_pred = target_games["home_win_prob_model"].dropna().values
                min_len = min(len(y_true), len(y_pred))
                cal_yt, cal_yp = y_true[:min_len], y_pred[:min_len]
            else:
                # No finals yet today: fall back to OOF pairs only -- labels
                # are real outcomes from completed games, never fabricated.
                _ot = pd.to_numeric(all_predictions["home_win"], errors="coerce")
                _op = pd.to_numeric(all_predictions["home_win_prob_model"], errors="coerce")
                _ok = _ot.notna() & _op.notna()
                cal_yt, cal_yp = _ot[_ok].values, _op[_ok].values
            path = _calibration_json(
                pooled_metrics, cal_yt, cal_yp, target_date_str, len(target_games),
                oof=all_predictions,
                evening_games=_count_evening_games(target_games),
            )
            summary["artifacts"].append(str(path))
            hist_path = _predictions_history_csv(all_predictions, target_date_str)
            if hist_path is not None:
                summary["artifacts"].append(str(hist_path))
            # Rolling Brier series over the same OOF history -- computed from
            # the raw blend through the DEPLOYED calibrator (get_last_calibrator
            # holds exactly the map predict-time and the charts use).
            rolling_brier = compute_rolling_brier(
                all_predictions, target_date_str, calibrator=get_last_calibrator()
            )
            summary["artifacts"].append(
                str(DATA_DELIVERY_DIR / f"rolling_brier_{target_date_str}.json")
            )
        # Feature metadata (dashboard tooltips) -- enumerates the ACTIVE
        # serving width (adopted RFE subset, else the universe), so new
        # features appear (or warn loudly) exactly at production width;
        # routing derived from live config.
        features_metadata = generate_features_metadata(target_date_str)
        summary["artifacts"].append(
            str(DATA_DELIVERY_DIR / f"features_metadata_{target_date_str}.json")
        )

        # Run engine (Phase 3): OOF re-derivation on the SAME fixed folds →
        # α(λ) dispersion curves fitted PRE-HOLDOUT only → NB Monte-Carlo
        # market grid (totals 6.5--12.5, run lines −0.5…−3.5) for OOF + today's
        # slate → agreement conflicts vs the moneyline ensemble. Must never
        # take down the rest of the run.
        run_engine_block = None
        try:
            from distributions import run_engine_daily
            _re = run_engine_daily(games, target_games, target_date_str,
                                        decided_snapshot=_decided_snapshot)
            run_engine_block = _re.get("block")
            summary["artifacts"].extend(_re.get("artifacts") or [])
            # Run-Line & Totals Monitor artifact: per-line calibration
            # + fit + rolling history + markets_persisted flag. run_engine_
            # daily returns markets_persisted=False (with a reason) when
            # the markets CSV persist failed so the monitor says so loudly
            # instead of silently serving stale data. Protected from
            # phase-6 cleanup by the run_engine_monitor_ prefix.
            try:
                _rem = _run_engine_monitor_json(
                    run_engine_block, target_date_str,
                    _re.get("markets_persisted", False),
                    _re.get("markets_persist_error"))
                summary["artifacts"].append(str(_rem))
            except Exception as mex:
                logger.error("Run-engine monitor write failed: %s", mex)
        except Exception as e:
            logger.error("Run engine failed (continuing): %s", e, exc_info=True)

        # 6. SHAP + Feature drift
        logger.info("Step 6: Explainability")
        # SHAP must never take down the run: drift + model monitor are more
        # important than per-game attributions, and artifacts from a failed
        # step would otherwise go stale.
        try:
            compute_shap_per_game(best_models, target_games)
        except Exception as e:
            logger.error("SHAP computation failed (continuing): %s", e)

        # Feature drift: compare the last 7 days vs an ADJACENT season-local
        # window (~3x the current window, min 250 games). Comparing against
        # all history instead made every cumulative feature (elo, win_pct,
        # run_diff) look like ALERT drift, because those distributions widen
        # structurally as a season matures -- a property of the feature, not
        # model health. Adjacent-but-not-tiny keeps it apples-to-apples while
        # giving quantile bin edges enough samples to be stable.
        # Decided games ONLY: pre-game slate rows carry the latest PIT state
        # forward (clustered near-identical values), so including them in the
        # current window distorted PSI for every feature.
        #
        # SINGLE SOURCE OF TRUTH (frames.py): get_decided_frame excludes
        # slate/pregame rows by construction (a decided frame is never the
        # slate frame — slate rows carry no StatsAPI game_pk even when
        # apply_official_results fills their scores post-merge).  The drift
        # step uses the pre-slate _decided_snapshot (captured ONCE after
        # official results + the weather/env feature passes, before the
        # Step 4 slate merge) so the drift frame is IDENTICAL to what
        # training used — no re-derivation from the mutated games object.
        # The fold-signature assert below fails loudly if that ever stops
        # being true.
        decided = _decided_snapshot
        _drift_pks = decided["game_pk"].tolist() if "game_pk" in decided.columns else []
        _drift_hash = _hb.sha256("|".join(str(p) for p in _drift_pks).encode()).hexdigest()[:12]
        logger.info("Drift decided frame: %d games, hash=%s", len(decided), _drift_hash)
        require_matching_signatures(
            "drift", fold_signature(decided),
            "training", get_last_fold_signature())
        cutoff = pd.Timestamp(target_date) - pd.Timedelta(days=7)
        gd = pd.to_datetime(decided["game_date"])
        # Chronological order is required: tail(N) on an unordered frame
        # mixes arbitrary seasons into the baseline window. Kept as-is
        # (including its quicksort tie behavior) so the drift margin
        # build consumes byte-identical row order to every prior run.
        decided = decided.sort_values("game_date")
        gd = pd.to_datetime(decided["game_date"])
        # run_margin_diff is the shipped moneyline feature that exists ONLY
        # in the margin-enriched training frame -- it never lands in
        # game_level_features.csv, so without enrichment the drift step
        # would silently omit its row. Attach leakage-free OOF margins on
        # the moneyline's own fold split (run engine READ-ONLY) so the
        # drift windows carry the same values the model saw; games outside
        # executed folds stay NaN (imputed at training) and are excluded
        # from the distribution with honest coverage counts.
        decided = _attach_drift_run_margins(decided)
        # FEATURE PARITY (2026-09-27): no projection attach — the drift frame
        # mirrors the pre-slate snapshot exactly (the former P1
        # sp_proj_era attach was removed with the run-line seam so both
        # models train on identical features).
        _post_pks = decided["game_pk"].tolist() if "game_pk" in decided.columns else []
        _post_hash = _hb.sha256("|".join(str(p) for p in _post_pks).encode()).hexdigest()[:12]
        logger.info("Drift post-margin frame: %d games, hash=%s",
                    len(decided), _post_hash)
        current = decided[gd >= cutoff]
        prior = decided[gd < cutoff]
        baseline = prior.tail(max(3 * len(current), 250)) if not prior.empty else prior
        if not baseline.empty and not current.empty:
            # PSI over the ACTIVE serving width (adopted RFE subset), not the
            # full universe — the drift table must mirror what the model sees;
            # universe-width rows for non-serving features mislead the monitor.
            drift_df = compute_feature_drift(
                baseline, current, target_date_str,
                model_weights=feature_importance_weights(best_models),
                feature_cols=active_moneyline_feature_cols(),
                phase_frame=decided,
            )
            summary["artifacts"].append(str(DATA_DELIVERY_DIR / f"feature_drift_{target_date_str}.csv"))
            # SINGLE-LIST RULE: explicit active-width enumeration, mirroring
            # the drift call above — the coverage table monitors exactly the
            # serving matrix, never a different list's superset.
            coverage_df = compute_feature_coverage(
                baseline, current, target_date_str,
                feature_cols=active_moneyline_feature_cols())
            summary["artifacts"].append(str(DATA_DELIVERY_DIR / f"feature_coverage_{target_date_str}.csv"))
            # Run-engine view of the SAME windows: PSI + coverage over
            # its shared moneyline feature contract (single NB sampler --
            # no weights, no run_margin_diff). Additive artifacts for the
            # run-line monitor; moneyline drift/coverage untouched.
            compute_run_engine_feature_drift(
                baseline, current, target_date_str,
                model_weights=(run_engine_block or {}).get("feature_weights")
                or None,
                phase_frame=decided)
            summary["artifacts"].append(str(
                DATA_DELIVERY_DIR
                / f"run_engine_feature_drift_{target_date_str}.csv"))
            compute_run_engine_feature_coverage(baseline, current, target_date_str)
            summary["artifacts"].append(str(
                DATA_DELIVERY_DIR
                / f"run_engine_feature_coverage_{target_date_str}.csv"))
            # Tripwire: the run-engine coverage windows are sliced from the
            # same baseline/current as the moneyline drift step — the 08-28
            # incident (coverage CSV on post-slate 288/96 windows) must be
            # structurally impossible now. Same frame -> same signature.
            require_matching_signatures(
                "run-engine coverage", fold_signature(decided),
                "drift", fold_signature(decided))
        else:
            drift_df = pd.DataFrame()
            coverage_df = pd.DataFrame()

        # model monitor JSON
        path = _model_monitor_json(pooled_metrics, drift_df, target_date_str, version=version, ensemble=last_ensemble_info(), coverage_df=coverage_df, rolling_brier=rolling_brier, features_metadata=features_metadata, run_engine=run_engine_block, season_split=get_last_season_split())
        summary["artifacts"].append(str(path))

        # 7. GitHub sync
        if not skip_sync:
            logger.info("Step 7: Syncing to GitHub")
            sync_result = sync_artifacts()
            summary["sync"] = sync_result
            if not sync_result["pushed"]:
                logger.warning("GitHub sync failed: %s", sync_result.get("error"))
        else:
            logger.info("Step 7: Skipping GitHub sync (Phase 5 owns the "
                        "artifact push)")

    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        summary["status"] = "error"
        summary["errors"].append(str(e))

    logger.info("=== Pipeline complete: %s ===", summary["status"])
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────


# ── Phase 4: Training + Prediction ──────────────────────────────────────────
_banner("PHASE 4", "Training + Prediction")
_phase4_error: Exception | None = None
# Declared OUTSIDE the try so Phase 5's reported-artifact gate can read it
# even when Phase 4 dies before run_daily_pipeline returns (2026-10-07
# run-log review, T3): an UNBOUND summary would skip the gate silently.
summary: dict = {}
try:
    from data_ingestion import load_game_features

    # Always load via load_game_features — it computes ELO, win_pct,
    # run_diff and maps columns to training.py's MONEYLINE_FEATURE_COLS format.
    train_games = load_game_features(csv_path)
    print(f"  📋 Training data: {train_games.shape[0]} games, {train_games.shape[1]} features")
    key_feats = ["home_elo", "home_win_pct", "sp_xfip_5g_home", "woba_30g_home",
                 "bullpen_kbb_10g_home", "rest_days_home"]
    cov = ", ".join(
        f"{c}:{train_games[c].notna().mean()*100:.0f}%"
        for c in key_feats if c in train_games.columns
    )
    print(f"  📊 Feature coverage: {cov}")

    # ── ADOPTED FEATURE-SUBSET STATE (record-only RFE governance) ───────────
    # Apply the adopted subset (data_delivery/mlb_feature_selection_state.json,
    # written only by an explicit --adopt invocation of feature_selection.py)
    # to the live training/serving width BEFORE any model work. Universe wins
    # on any problem. run_daily_pipeline re-applies it on the retrain path and
    # pins cached-bundle width itself — this covers everything else that reads
    # the width in this process.
    try:
        from feature_selection import apply_adopted_subset
        _fs = apply_adopted_subset()
        print(f"  🎯 Feature width: {'RFE subset (' + str(_fs.get('n_cols')) + ' cols)' if _fs.get('applied') else 'full universe'}")
    except Exception as _fs_exc:
        print(f"  ⚠️  Feature-selection state apply failed (universe width): {_fs_exc}")

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
    _phase4_error = e
else:
    if str(summary.get("status", "")).lower() != "ok" or summary.get("errors"):
        # The daily run's DATED artifacts (todays_games_*, predictions_*)
        # are written in Phase 4. A failed slate/predict step means today's
        # dashboard silently falls back to yesterday's files, so this must
        # surface as a RED run, not a banner that still says DONE ✅
        # (2026-09-22 shipped zero dated artifacts and still printed ✅).
        print(f"  ❌ Phase 4 reported status={summary.get('status')} "
              f"errors={summary.get('errors')} — failing the run loudly")
        _phase4_error = RuntimeError(
            f"daily pipeline status={summary.get('status')} "
            f"errors={summary.get('errors')}")
    else:
        _phase4_error = None

# ── Phase 4.5: Feature-selection RFE (record-only) ────────────────────────
# Runs ONLY when MLB_RFE_FORCE=1 — unset/0 is a no-op with no calendar
# logic. Writes data_delivery/mlb_feature_selection_<date>.json (10-day
# retention) and NEVER
# adopts — changing serving width requires the explicit --adopt invocation
# of feature_selection.py. A failure here must never block the artifact
# sync below. (train_games may be unbound if Phase 4 died early — the
# guard covers that too.)
_rfe: dict = {}  # safe default if the RFE phase dies before assigning
try:
    from feature_selection import maybe_run_rfe
    _rfe = maybe_run_rfe(train_games, end)
    if _rfe.get("ran"):
        print(f"  🎯 RFE [{_rfe.get('run_mode')}]: pool {_rfe.get('n_pool')} | "
              f"trials {_rfe.get('n_trials')} (committed "
              f"{_rfe.get('n_committed')}) | "
              f"selected {_rfe.get('n_selected')} cols "
              f"(logloss {_rfe['baseline_logloss']:.4f} -> {_rfe['best_logloss']:.4f})")
        if _rfe.get("forced_unresolved"):
            print(f"     ⚠️  forced-trial list had unresolvable names: "
                  f"{_rfe['forced_unresolved']}")
        if _rfe.get("incumbent_note"):
            print(f"     incumbent: {_rfe['incumbent_note']}")
        print(f"     trace: {_rfe['trace']}")
    else:
        print(f"  🎯 RFE skipped: {_rfe.get('reason')}")
except Exception as _rfe_exc:
    print(f"  ⚠️  Feature-selection RFE skipped ({_rfe_exc})")

# ── Phase 4.6: Feature Decision Workbook (human-readable RFE companion) ──
# Regenerates data_delivery/mlb_feature_workbook_<trace-date>.xlsx from the
# newest trace whenever an RFE run just wrote one (10-day retention). The .xlsx is the
# readable, actionable form of the trace (feature inventory, per-test
# impact in plain English, redundancy, coverage gaps); the .json stays the
# machine-readable record. NEVER blocks the run: a workbook failure is
# reported and skipped — Phase 5 syncs whatever exists.
if _rfe.get("ran"):
    try:
        from feature_workbook import generate_workbook
        # Build from the trace THIS run just wrote (full or targeted) so the
        # workbook always reflects the run that finished — never a silent
        # rebuild of the prior full trace.
        _wb_path = generate_workbook(trace_path=_rfe.get("trace"))
        if _wb_path:
            print(f"  📊 Feature workbook: {_wb_path}")
    except Exception as _wb_exc:
        print(f"  ⚠️  Feature workbook skipped ({_wb_exc})")

# ── Phase 5: GitHub Sync — push this run's NEW files first ─────────────────
_banner("PHASE 5", "GitHub Sync — push new artifacts")
token = token or CONFIG.get("github_token", "")
sync_dir = Path("/content/mlb_sync_tmp")
# Race-resilient push machinery (lives in github_sync so it is importable
# and unit-testable — master_pipeline is a run-once script).
from github_sync import (
    missing_reported_artifacts,
    push_with_retry,
    sync_remote_tip,
    verify_pushed_paths,
)

def _git_push_confirmed(repo, branch: str) -> None:
    """Single-shot push kept for compatibility; raises on rejection.

    The daily run now pushes through ``push_with_retry`` instead: a reused
    sync clone that sat out earlier pushes used to commit on a stale
    snapshot and get hard-rejected (non-fast-forward), losing the run's
    artifact delivery. See github_sync.sync_remote_tip / push_with_retry.
    """
    push_with_retry(repo, branch, restage=None, attempts=1, log=print)

def _open_sync_repo(token: str, sync_dir: Path):
    """Open the sync clone (or create it), configuring git identity."""
    import git
    auth_url = f"https://{token}@github.com/{CONFIG['github_username']}/{CONFIG['github_repo']}.git"
    if (sync_dir / ".git").exists():
        repo = git.Repo(str(sync_dir))
        # A warm clone can predate other pushes (the previous run crashed
        # before cleanup, or a manual push landed between runs) — heal it to
        # the CURRENT remote tip before anything commits on top, or the
        # eventual push is a guaranteed non-fast-forward rejection.
        sync_remote_tip(repo, CONFIG["github_branch"], log=print)
    else:
        repo = git.Repo.clone_from(auth_url, str(sync_dir), branch=CONFIG["github_branch"], depth=1)
    if CONFIG["git_email"]: repo.config_writer().set_value("user", "email", CONFIG["git_email"]).release()
    if CONFIG["git_name"]:  repo.config_writer().set_value("user", "name", CONFIG["git_name"]).release()
    return repo

staged: list[str] = []
seen: set[str] = set()
staged_srcs: dict[str, Path] = {}  # rel -> local source, for push retries

if not token:
    raise RuntimeError("MLB artifact delivery requires a GitHub token; refusing to skip synchronization")
else:
    try:
        repo = _open_sync_repo(token, sync_dir)
        data_delivery_dir = sync_dir / SPORT_DIR_NAME / "data_delivery"
        data_delivery_dir.mkdir(parents=True, exist_ok=True)

        def _stage(src: Path, rel: str) -> None:
            if rel in seen:
                return
            seen.add(rel)
            staged_srcs[rel] = src
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
        #
        # BOTH delivery roots are scanned (2026-10-07 run-log review, T1): the
        # 2026-10-07 run wrote its 15 reported artifacts to config's
        # DATA_DELIVERY_DIR while cwd pointed at the fresh clone, so a
        # cwd-only scan staged 3 files, GitHub kept serving the 10-06 board,
        # and the pushed log still claimed a successful delivery (remote
        # carried zero *20261007* files). Per-root snapshots keep the
        # stale-file rule intact across the union.
        data_delivery_local = Path.cwd() / "data_delivery"
        # IL/availability ledgers are LOCAL-RUNTIME caches (rebuilt by Phase
        # 1.5 under MLB_IL_STINTS_DIR, outside the repo) — never GitHub
        # artifacts. The env relocation already keeps them out of this
        # folder; the name skip is belt-and-suspenders for legacy copies.
        _il_cache_names = {
            "il_stints.parquet", "il_stints.meta.json",
            "il_stints_pitchers.parquet", "il_stints_pitchers.meta.json"}
        _scan_roots: list[Path] = [data_delivery_local]
        try:
            from config import DATA_DELIVERY_DIR as _DD_SCAN
            _cfg_scan_root = Path(_DD_SCAN)
            if _cfg_scan_root.resolve() not in {r.resolve() for r in _scan_roots}:
                _scan_roots.append(_cfg_scan_root)
        except Exception:  # noqa: BLE001 — cwd-only scan still works
            pass
        if len(_scan_roots) > 1:
            print(f"  ⚠️  delivery root split — staging from "
                  f"{len(_scan_roots)} roots:")
            for _r in _scan_roots:
                print(f"    {_r}")
        for _root in _scan_roots:
            if not _root.exists():
                continue
            try:
                _root_key = _root.resolve()
            except OSError:
                _root_key = _root
            _pre = _preexisting_delivery_roots.get(_root_key, {})
            for artifact in sorted(_root.rglob("*")):
                if artifact.is_file():
                    rel_local = artifact.relative_to(_root).as_posix()
                    if rel_local in _il_cache_names:
                        continue  # local runtime cache — never pushed
                    pre_mtime = _pre.get(rel_local)
                    if pre_mtime is not None and artifact.stat().st_mtime_ns <= pre_mtime:
                        continue  # repo file untouched by this run -> stale
                    _stage(artifact, f"{SPORT_DIR_NAME}/data_delivery/{rel_local}")
        print(f"  📋 Staging {len(staged)} files:")
        for s in staged:
            print(f"    {s}")
        # Reported-artifact coverage gate (2026-10-07 run-log review, T3):
        # Step 5 printed "📁 Artifacts: 15 files" while Phase 5 printed
        # "Pushed and remotely verified 3 files" — the other 12 were missing
        # from GitHub entirely. Every artifact the run REPORTS must be in
        # the staging list or delivery is refused (the except below turns
        # this into a failed run, never a green one).
        _missing_artifacts = missing_reported_artifacts(
            summary.get("artifacts") or [], staged, SPORT_DIR_NAME)
        if _missing_artifacts:
            print(f"  ❌ {len(_missing_artifacts)} reported artifact(s) "
                  f"missing from staging:")
            for _m in _missing_artifacts:
                print(f"    {_m}")
            raise RuntimeError(
                "reported artifacts not staged: "
                + ", ".join(_missing_artifacts))
        if staged:
            def _restage_artifacts() -> None:
                # Retry path: replay this run's staged files onto whatever
                # tip sync_remote_tip just healed the clone to, then commit
                # again. Sources are remembered from the original staging.
                ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                for rel in staged:
                    dest = data_delivery_dir / rel[len(f"{SPORT_DIR_NAME}/data_delivery/"):]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(staged_srcs[rel], dest)
                repo.index.add(staged)
                repo.index.commit(f"Update MLB features + predictions: {ts}")

            repo.index.add(staged)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            repo.index.commit(f"Update MLB features + predictions: {ts}")
            push_with_retry(repo, CONFIG["github_branch"],
                            restage=_restage_artifacts, log=print)
            verify_pushed_paths(repo, CONFIG["github_branch"], staged)
            print(f"  ✅ Pushed and remotely verified {len(staged)} files — confirmed on {CONFIG['github_repo']}@{CONFIG['github_branch']}")
        else:
            print("  ⏭️  Nothing new to push")
    except Exception as e:
        print(f"  ❌ Artifact delivery failed: {e}")
        raise RuntimeError("MLB artifact delivery did not complete; refusing to report a successful pipeline run") from e

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

# Retention ANCHOR (policy 2026-09-14): MLB_END_DATE when the run sets it,
# else TODAY in America/New_York — ONE anchor for every window (the old mix
# of an end_date-anchored 48h window with a UTC-anchored slate window is
# gone). Kaggle servers run UTC; ET is the game-day convention the
# dashboards use. DST-safe via zoneinfo (project convention, pipeline.py).
from zoneinfo import ZoneInfo
if "MLB_END_DATE" in os.environ and os.environ["MLB_END_DATE"].strip():
    # Explicit run window (backfill/rebuild): anchor on MLB_END_DATE exactly
    # as CONFIG resolved it.
    _anchor_date = CONFIG["end_date"]
else:
    _anchor_date = _dt.datetime.now(
        ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
_anchor_compact = _anchor_date.replace("-", "")  # YYYYMMDD

# Blanket retention window (policy 2026-09-14): anchor .. anchor-10. EVERY
# dated, allowlisted family keeps 10 calendar days; artifacts NEWER than the
# anchor are kept by classify_artifact's anchor guard (backfill-safe).
_RETENTION_DAYS = 10
_anchor_obj = _date(*(int(x) for x in _anchor_date.split("-")))
_RETENTION_DATES = {(_anchor_obj - _td(days=i)).strftime("%Y%m%d")
                    for i in range(_RETENTION_DAYS + 1)}

# Recent-slate settle window (todays_games_* / shap_game_*): now a strict
# subset of the blanket window — at 10 days the slate rule no longer extends
# beyond it; wired through for policy parity (retention_policy slate_window).
_RECENT_DATES = {
    (_anchor_obj - _td(days=i)).strftime("%Y%m%d")
    for i in range(3)  # anchor, anchor-1, anchor-2
}

# Board-backed retention (doubleheader regression fix): a dated run-engine /
# predictions artifact is kept for ANY date that still has a tracked
# todays_games_<date>.csv board, so a navigable board is never left without
# the RUN ENGINE columns its cards need. Keep them for as long as the board
# itself is tracked (policy: families with board_supported=True) — at the
# 10-day blanket window the slate rule dominates; this stays a safety net.
_BOARD_BACKED_PREFIXES = _family_prefixes("board_supported")

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
        # Classify: protected → keep; in seen → keep; within the 10-day
        # retention window → keep; recent or board-backed slate/run-engine
        # artifacts or dated mlb_ records within the window → keep;
        # otherwise → stale.
        stale = []
        kept_protected = 0
        kept_current = 0
        for p in tracked:
            verdict = classify_artifact(
                p, seen, _RETENTION_DATES, _RECENT_DATES, board_dates,
                anchor_date=_anchor_compact)
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
            print(f"  📅 Kept {kept_current} artifact(s) within the retention "
                  f"window (anchor {_anchor_compact} -10d)")
        if not stale:
            print("  ✅ No stale files — data_delivery holds exactly this run's artifacts")
        else:
            print(f"  🧹 Removing {len(stale)} stale files:")
            for s in stale:
                print(f"    {s}")

            def _restage_cleanup() -> None:
                # Retry path: after sync_remote_tip heals the clone to the
                # (not-yet-cleaned) remote tip, the stale paths exist again,
                # so the rm + commit replay cleanly.
                ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                repo.git.rm(stale)
                repo.index.commit(f"Remove stale data_delivery artifacts: {ts}")

            _restage_cleanup()
            push_with_retry(repo, CONFIG["github_branch"],
                            restage=_restage_cleanup, log=print)
            print(f"  ✅ Removed {len(stale)} stale files — confirmed on {CONFIG['github_repo']}@{CONFIG['github_branch']}")
    except Exception as e:
        print(f"  ❌ Cleanup failed: {e}")

_banner("DONE ✅" if _phase4_error is None else "DONE — WITH ERRORS ❌")
# 2026-10-06 log review: "Features: 378" summed the game + pbp COLUMN counts
# and read like a feature width, contradicting "Feature width: RFE subset
# (109 cols)" and "Training data: ... 293 features" in the same log — label
# the split so the three numbers reconcile.
print(f"  Games: {game_df.shape[0]}  |  Pitches: {pbp_df.shape[0]:,}  |  "
      f"Columns: {game_df.shape[1]} game + {pbp_df.shape[1]} pbp")
print(f"  Output: {out_dir}")

# ── Final run-log delivery (2026-10-05 log review) ─────────────────────────
# Phase 5 stages the run log like any other artifact — a SNAPSHOT copied
# before the staging list, the push confirmation, and ALL of Phase 6 ever
# reached the file. Every pushed log therefore ended at the bare
# "PHASE 5 — GitHub Sync" banner (the 2026-10-05 review found exactly
# that on remote, 1051 lines ending mid-phase). Re-copy the now-complete
# log onto the warm sync clone and push it as the run's LAST delivery:
# the DONE banner above is already flushed, so the pushed file carries the
# whole run up to this announcement (push progress lines land after the
# copy and stay local — the file cannot contain its own push). Restage+retry
# mirrors Phase 5's _restage_artifacts, so a mid-air rejection replays
# cleanly onto the healed tip. Never fatal: the artifacts are already
# remotely verified, and a failed log push must not fail a delivered run.
if _log_path and token:
    try:
        print("  📝 Final run-log delivery — pushing the complete log "
              "(Phase 5 staged a snapshot before Phase 5/6 finished)")
        repo = _open_sync_repo(token, sync_dir)
        _log_rel = f"{SPORT_DIR_NAME}/data_delivery/{RUN_LOG_NAME}"
        _log_dest = sync_dir / _log_rel
        _log_dest.parent.mkdir(parents=True, exist_ok=True)
        _log_ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        def _restage_run_log() -> None:
            shutil.copy2(_log_path, _log_dest)
            repo.index.add([_log_rel])
            repo.index.commit(f"Pipeline run log: {_log_ts}")

        _restage_run_log()
        push_with_retry(repo, CONFIG["github_branch"],
                        restage=_restage_run_log, attempts=2, log=print)
        verify_pushed_paths(repo, CONFIG["github_branch"], [_log_rel])
        print("  ✅ Run log pushed and remotely verified — Phase 5/6 "
              "output included")
    except Exception as _log_exc:
        print(f"  ⚠️  Final run-log delivery did not complete: {_log_exc}")

# ── Notebook sanity-footer compatibility (2026-10-09 log review) ───────────
# The Kaggle notebook's post-run confirmation block re-types the clone
# path a second time, and the LIVE copy's duplicate is typo'd
# (/kaggle/working/sports_predictio_model): after a fully successful
# delivery the cell died on FileNotFoundError and the run went red. The
# notebook program is never edited from the repo (Kaggle-owned guardrail,
# T7 of the 2026-10-06 review), and this script runs in the same session
# BEFORE that block executes — so the RUN repairs the environment: point
# the typo'd location at the real clone so the notebook's advisory
# git checks run green and report the ACTUAL repository. Advisory only:
# created when missing, never fatal, no-op off Kaggle.
try:
    from github_sync import ensure_notebook_sanity_alias
    _wk = Path("/kaggle/working")
    if _wk.is_dir():
        _alias_ok = ensure_notebook_sanity_alias(
            _wk / "sports_prediction_model",
            _wk / "sports_predictio_model")
        print("  🔗 Notebook sanity alias:",
              "typo'd path now resolves to the real clone" if _alias_ok
              else "left as-is (already present or not creatable)")
except Exception as _alias_exc:
    print(f"  ⚠️  Notebook sanity alias skipped: {_alias_exc}")

if sync_dir.exists():
    shutil.rmtree(sync_dir, ignore_errors=True)

# Honest exit code: a failed prediction phase must fail the RUN (nonzero
# exit), even though artifact delivery already pushed whatever existed.
# The Kaggle wrapper raises SystemExit on a nonzero code, so a stale-board
# day like 2026-09-22 (zero dated artifacts, banner said ✅) can never
# silently pass again.
if _phase4_error is not None:
    raise SystemExit(f"MLB pipeline finished WITH ERRORS: {_phase4_error}")
