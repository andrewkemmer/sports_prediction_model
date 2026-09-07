"""Data Delivery regression tests — artifact persistence + frontend contract.

Evidence-first: every test exercises real filesystem persistence or the real
frontend resolution logic (imported from frontend/sports_config.py +
frontend/utils.py). No network is used; the GitHub fetch path is asserted by
source inspection and pattern compatibility, not by live requests.

Run:  python3 test_data_delivery.py        (from nfl-backend/backend/)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent.parent
DD = BACKEND_DIR.parent / "data_delivery"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  ({detail})" if detail and not cond else ""))


# ---------------------------------------------------------------------------
print("\n== 1. Persistence / directory creation ==")
import config  # noqa: E402
check("config.DATA_DELIVERY_DIR is nfl-backend/data_delivery",
      config.DATA_DELIVERY_DIR == REPO_ROOT / "nfl-backend" / "data_delivery",
      str(config.DATA_DELIVERY_DIR))
check("persistence layer creates the output dir (mkdir in main)",
      "out_dir.mkdir(parents=True, exist_ok=True)"
      in (BACKEND_DIR / "master_pipeline.py").read_text())
check("data_delivery dir exists on disk", DD.is_dir())

# ---------------------------------------------------------------------------
print("\n== 2. Frontend contract: directory + filename patterns ==")
_spec = importlib.util.spec_from_file_location(
    "sports_config", REPO_ROOT / "frontend" / "sports_config.py")
sports_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sports_config)
cfg = sports_config.SPORTS["nfl"]
check("frontend repo_subdir is nfl-backend", cfg["repo_subdir"] == "nfl-backend")
check("frontend artifact registry has_run_engine", cfg.get("has_run_engine") is True)
frontend_dir = REPO_ROOT / cfg["repo_subdir"] / "data_delivery"
check("frontend expected directory == backend output directory",
      frontend_dir == config.DATA_DELIVERY_DIR,
      f"{frontend_dir} vs {config.DATA_DELIVERY_DIR}")

PATTERNS = {
    "moneyline_json": "nfl_moneyline_v1_*.json",
    "feature_json": "nfl_feature_v1_*.json",
    "calibration_json": "nfl_calibration_*.json",
    "predictions_history_csv": "nfl_predictions_history_*.csv",
    "power_rankings_csv": "nfl_power_rankings_*.csv",
    "markets_csv": "nfl_run_engine_markets_*.csv",
    "qb_matchup_json": "nfl_qb_matchup_*.json",
}
check("frontend artifact_patterns registry present", len(cfg.get("artifacts", {})) > 0)
for family, pattern in PATTERNS.items():
    produced = sorted(DD.glob(pattern))
    check(f"frontend pattern {pattern!r} matches a produced artifact",
          len(produced) > 0, "no matching file")
    if produced:
        stem = produced[-1].name
        # the frontend's _stamp_suffixes/YYYYMMDD resolution requires an
        # 8-digit date embedded in the filename
        core = stem[len(pattern.split("*")[0]):]
        core = core[:8] if core[:8].isdigit() else ""
        check(f"  {stem} carries a YYYYMMDD date suffix", len(core) == 8)

# ---------------------------------------------------------------------------
print("\n== 3. Required artifacts exist, parse, and satisfy the frontend schema ==")
REQUIRED_JSON = ["nfl_moneyline_v1_*.json", "nfl_calibration_*.json",
                 "nfl_qb_matchup_*.json", "nfl_feature_v1_*.json",
                 "nfl_model_monitor_*.json"]
REQUIRED_CSV = ["nfl_predictions_history_*.csv", "nfl_power_rankings_*.csv",
                "nfl_run_engine_markets_*.csv"]
CARD_FIELDS = {"game_id", "game_date", "home_team", "away_team",
               "home_team_name", "away_team_name", "home_record", "away_record",
               "venue", "game_status", "home_score", "away_score",
               "home_win_prob_model", "away_win_prob_model", "model_pick",
               "model_correct"}
for pattern in REQUIRED_JSON:
    files = sorted(DD.glob(pattern))
    check(f"{pattern} present", bool(files))
    if not files:
        continue
    path = files[-1]
    try:
        rec = json.loads(path.read_text())
        valid = True
    except Exception as exc:  # noqa: BLE001
        check(f"{path.name} parses as JSON", False, str(exc))
        continue
    check(f"{path.name} parses as JSON ({path.stat().st_size:,} bytes)", valid)
    if pattern.startswith("nfl_moneyline"):
        games = rec.get("games") or []
        check("  moneyline games[] present and non-empty", len(games) > 0)
        check("  moneyline card schema superset of NFL_CARD_COLUMNS contract",
              CARD_FIELDS.issubset(games[0].keys()) if games else False,
              str(sorted(CARD_FIELDS - set(games[0].keys()))) if games else "")
        check("  every game carries a finite home_win_prob_model",
              all(isinstance(g.get("home_win_prob_model"), (int, float))
                  and 0.0 <= g["home_win_prob_model"] <= 1.0 for g in games))
    if pattern.startswith("nfl_calibration"):
        check("  calibration metrics/daily/buckets keys present",
              all(k in rec for k in ("metrics", "daily", "calibration_buckets")))
        check("  calibration metrics carry auc/brier/logloss/ece",
              all(k in rec["metrics"] for k in ("auc", "brier", "logloss", "ece")))

for pattern in REQUIRED_CSV:
    files = sorted(DD.glob(pattern))
    check(f"{pattern} present", bool(files))
    if not files:
        continue
    path = files[-1]
    try:
        df = pd.read_csv(path)
        check(f"{path.name} parses as CSV ({path.stat().st_size:,} bytes, "
              f"{len(df)} rows)", True)
    except Exception as exc:  # noqa: BLE001
        check(f"{path.name} parses as CSV", False, str(exc))
        continue
    if "predictions_history" in pattern:
        need = {"game_id", "game_date", "home_team", "away_team", "home_score",
                "away_score", "home_win_prob_model", "model_pick",
                "actual_winner", "correct"}
        check("  predictions history columns superset of frontend reader",
              need.issubset(df.columns), str(sorted(need - set(df.columns))))
    if "power_rankings" in pattern:
        need = {"rank", "team", "team_name", "elo", "wins", "losses", "record"}
        check("  power rankings columns superset of shared page",
              need.issubset(df.columns), str(sorted(need - set(df.columns))))
    if "run_engine_markets" in pattern:
        need = {"kind", "game_id", "mu_h", "mu_a", "fair_spread", "fair_total",
                "p_home_win_derived", "p_away_win_derived"}
        check("  markets columns superset of frontend reader",
              need.issubset(df.columns), str(sorted(need - set(df.columns))))
        check("  markets carries slate rows (current-slate delivery)",
              (df["kind"] == "slate").any())

# ---------------------------------------------------------------------------
print("\n== 4. Current-slate delivery verification ==")
ml_files = sorted(DD.glob("nfl_moneyline_v1_*.json"))
if ml_files:
    rec = json.loads(ml_files[-1].read_text())
    games = rec.get("games") or []
    dates = sorted({g.get("game_date") for g in games})
    check("moneyline record carries a current slate", len(games) > 0)
    check("slate games have unique game_ids",
          len({g.get("game_id") for g in games}) == len(games))
    check("slate game_date values present", bool(dates) and dates[0] is not None)
    check("model_pick present on every game",
          all(g.get("model_pick") in (g.get("home_team"), g.get("away_team"))
              for g in games))
    print(f"    slate: n_games={len(games)}, slate_date={rec.get('slate_date')}, "
          f"first game_date={dates[0] if dates else None}")

# ---------------------------------------------------------------------------
print("\n== 5. Git visibility: serving families must be trackable ==")
import subprocess
def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=str(REPO_ROOT)).stdout

for pattern in REQUIRED_JSON + REQUIRED_CSV + ["models/nfl_ensemble_latest.joblib"]:
    files = sorted((DD).glob(pattern)) if "models/" not in pattern else sorted((DD / "models").glob("*.joblib"))
    if not files:
        check(f"git-trackable: {pattern}", False, "no file produced")
        continue
    rel = files[-1].relative_to(REPO_ROOT).as_posix()
    ignored = subprocess.run(["git", "check-ignore", rel], capture_output=True,
                             text=True, cwd=str(REPO_ROOT)).returncode == 0
    check(f"git-trackable (not ignored): {rel}", not ignored)

oof_ignored = subprocess.run(
    ["git", "check-ignore", "nfl-backend/data_delivery/nfl_oof_moneyline.csv"],
    capture_output=True, text=True, cwd=str(REPO_ROOT)).returncode == 0
check("run internals stay ignored (nfl_oof_*.csv)", oof_ignored)

# The frontend fetch path prefers GitHub raw over local disk; the delivery
# architecture therefore requires these files to be committable. Assert the
# resolution order from the frontend source.
utils_src = (REPO_ROOT / "frontend" / "utils.py").read_text()
check("frontend _fetch_bytes prefers raw.githubusercontent.com then local",
      "raw.githubusercontent.com" in utils_src
      and "return resp.content, \"github\"" in utils_src
      and "return local.read_bytes(), \"local\"" in utils_src)

# ---------------------------------------------------------------------------
print("\n== 6. Persistence failure semantics ==")
mon_src = (BACKEND_DIR / "master_pipeline.py").read_text()
check("pipeline gates completion on schema validation (no silent success)",
      "validation gates failed" in mon_src and "RuntimeError" in mon_src)
check("pipeline hard-fails on zero folds", "PHASE 4 produced ZERO folds" in mon_src)

# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"RESULTS: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL DATA DELIVERY TESTS PASSED")
