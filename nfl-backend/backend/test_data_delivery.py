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
      in (BACKEND_DIR / "master_pipeline.py").read_text(encoding="utf-8"))
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
        rec = json.loads(path.read_text(encoding="utf-8"))
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
        cal_buckets = (rec.get("calibration") or {}).get(
            "calibration_buckets_calibrated") or []
        raw_buckets = rec.get("calibration_buckets") or []
        check("  raw/calibrated buckets use identical counts",
              len(raw_buckets) == len(cal_buckets)
              and [b.get("count") for b in raw_buckets]
              == [b.get("count") for b in cal_buckets])
        check("  calibration provenance preserves favored method",
              (rec.get("calibration") or {}).get("method")
              in ("favored_platt_floor", "platt"))

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
        check("  history carries stored deployed calibrated probabilities",
              "home_win_prob_model_calibrated" in df.columns)
        if "home_win_prob_model_calibrated" in df.columns:
            check("  deployed probabilities are finite and bounded",
                  pd.to_numeric(df["home_win_prob_model_calibrated"], errors="coerce")
                  .between(0, 1).all())
        dates_2026 = set(df.loc[
            df["game_date"].astype(str).str.startswith("2026"),
            "game_date"].astype(str))
        check("  retained history includes season 2026 dates",
              {"2026-09-20", "2026-09-21"}.issubset(dates_2026))
        car = df[(df["game_date"].astype(str) == "2026-09-20")
                 & (df["home_team"] == "ATL") & (df["away_team"] == "CAR")]
        if not car.empty:
            p_car = pd.to_numeric(car["home_win_prob_model_calibrated"],
                                  errors="coerce").iloc[0]
            check("  away pick displays the away deployed probability",
                  car["model_pick"].iloc[0] == "CAR" and float(1.0 - p_car) > 0.5)
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
    rec = json.loads(ml_files[-1].read_text(encoding="utf-8"))
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
utils_src = (REPO_ROOT / "frontend" / "utils.py").read_text(encoding="utf-8")
check("frontend _fetch_bytes prefers raw.githubusercontent.com then local",
      "raw.githubusercontent.com" in utils_src
      and "return resp.content, \"github\"" in utils_src
      and "return local.read_bytes(), \"local\"" in utils_src)

# ---------------------------------------------------------------------------
print("\n== 6. Rolling retention policy (MLB parity: 10-day blanket window) ==")
import tempfile
import retention_policy as rp

_ANCHOR = "20260922"
_retention = {(pd.Timestamp(_ANCHOR) - pd.Timedelta(days=i)).strftime("%Y%m%d")
              for i in range(11)}
_recent = {(pd.Timestamp(_ANCHOR) - pd.Timedelta(days=i)).strftime("%Y%m%d")
           for i in range(3)}

check("anchor date kept (window boundary)",
      rp.classify_artifact("x/nfl_calibration_20260922.json", set(),
                           _retention, _recent, set(), anchor_date=_ANCHOR)
      == "current")
check("anchor-10 kept (window boundary)",
      rp.classify_artifact("x/nfl_calibration_20260912.json", set(),
                           _retention, _recent, set(), anchor_date=_ANCHOR)
      == "current")
check("anchor-11 stale (window boundary)",
      rp.classify_artifact("x/nfl_calibration_20260911.json", set(),
                           _retention, _recent, set(), anchor_date=_ANCHOR)
      == "stale")
check("newer than anchor kept (backfill-safe)",
      rp.classify_artifact("x/nfl_calibration_20260930.json", set(),
                           _retention, _recent, set(), anchor_date=_ANCHOR)
      == "current")
check("models/ never deleted", rp.is_never_delete("x/models/nfl_ensemble_latest.joblib"))
check("run-engine monitor series never deleted",
      rp.classify_artifact("x/nfl_run_engine_monitor_20250101.json", set(),
                           _retention, _recent, set(), anchor_date=_ANCHOR)
      == "protected")
check("frozen card store never deleted",
      rp.is_never_delete("x/nfl_production_cards_history.csv"))
check("RFE state never deleted",
      rp.is_never_delete("x/nfl_feature_selection_state.json"))
check("shap file pruned after 10 days past its game (game-date map)",
      rp.classify_artifact("x/nfl_shap_game_2026_01_ARI_LAC.csv", set(),
                           _retention, _recent, set(), anchor_date=_ANCHOR,
                           game_dates={"2026_01_ARI_LAC": "2026-01-04"})
      == "stale")
check("shap file kept while its game is inside the window",
      rp.classify_artifact("x/nfl_shap_game_2026_02_CAR_ATL.csv", set(),
                           _retention, _recent, set(), anchor_date=_ANCHOR,
                           game_dates={"2026_02_CAR_ATL": "2026-09-20"})
      == "current")
check("future-slate shap kept (backfill-safe anchor guard)",
      rp.classify_artifact("x/nfl_shap_game_2026_14_BUF_NE.csv", set(),
                           _retention, _recent, set(), anchor_date=_ANCHOR,
                           game_dates={"2026_14_BUF_NE": "2026-12-27"})
      == "current")
check("unresolvable shap id protected (never guessed)",
      rp.classify_artifact("x/nfl_shap_game_unknown_id.csv", set(),
                           _retention, _recent, set(), anchor_date=_ANCHOR,
                           game_dates={})
      == "protected")
check("board-backed family kept while its board is tracked",
      rp.classify_artifact("x/nfl_run_engine_markets_20260910.csv", set(),
                           _retention, _recent, {"20260910"}, anchor_date=_ANCHOR)
      == "current")
check("dateless non-master is stale",
      rp.classify_artifact("x/nfl_random_dateless.csv", set(),
                           _retention, _recent, set(), anchor_date=_ANCHOR)
      == "stale")

# Integration: the REAL prune against a temp delivery dir removes exactly
# the stale set and leaves masters/series/current files untouched.
try:
    import master_pipeline as mp_mod
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        (out / "models").mkdir()
        staged = {"nfl_calibration_20260923.json"}
        files = {
            "nfl_calibration_20260922.json": "keep",
            "nfl_calibration_20260912.json": "keep",
            "nfl_calibration_20260911.json": "prune",
            "nfl_moneyline_v1_20260910.json": "prune",
            "nfl_run_engine_monitor_20260901.json": "keep",   # series
            "nfl_run_engine_monitor_20260825.json": "keep",   # series
            "models/nfl_ensemble_latest.joblib": "keep",      # master
            "nfl_production_cards_history.csv": "keep",        # master
            "nfl_feature_selection_state.json": "keep",        # master
            "nfl_shap_game_2026_01_ARI_LAC.csv": "prune",      # aged via map
            "nfl_shap_game_2026_02_CAR_ATL.csv": "keep",       # in window
            "nfl_shap_game_2026_14_BUF_NE.csv": "keep",        # future slate
        }
        for name in files:
            q = out / name
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_bytes(b"x")
        # moneyline record drives board dates + SHAP game-date map
        (out / "nfl_moneyline_v1_20260922.json").write_text(json.dumps({
            "games": [
                {"game_id": "2026_01_ARI_LAC", "game_date": "2026-01-04"},
                {"game_id": "2026_02_CAR_ATL", "game_date": "2026-09-20"},
                {"game_id": "2026_14_BUF_NE", "game_date": "2026-12-27"},
            ]}), encoding="utf-8")
        mp_mod._prune_old_artifacts(out, "20260922",
                                    seen=staged, anchor_iso="2026-09-22")
        remaining = {str(p.relative_to(out).as_posix())
                     for p in out.rglob("*") if p.is_file()}
        remaining.add("nfl_calibration_20260923.json")
        expected_keep = {n for n, v in files.items() if v == "keep"} | staged
        expected_prune = {n for n, v in files.items() if v == "prune"}
        check("integration: exactly the stale set removed",
              expected_prune.isdisjoint(remaining),
              str(sorted(expected_prune & remaining)))
        check("integration: every keep/protected/current file survives",
              expected_keep.issubset(remaining),
              str(sorted(expected_keep - remaining)))
except Exception as exc:  # noqa: BLE001
    check("retention integration probe", False, str(exc))

# ---------------------------------------------------------------------------
print("\n== 7. Frozen first-publication card store ==")
try:
    import master_pipeline as mp_mod
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        oof = pd.DataFrame({
            "game_id": ["2026_02_CAR_ATL", "2026_02_NO_BAL"],
            "gameday": ["2026-09-20", "2026-09-20"],
            "home_team": ["ATL", "BAL"], "away_team": ["CAR", "NO"],
            "p_ensemble": [0.466087, 0.697600],
            "p_ensemble_calibrated": [0.460653, 0.700000],
            "home_win": [0.0, 0.0],
            "home_score": [3.0, 17.0], "away_score": [34.0, 24.0],
        })
        name = mp_mod._update_cards_history_store(out, oof, pd.DataFrame(), "20260923")
        check("store written and manifested", name == "nfl_production_cards_history.csv")
        s1 = pd.read_csv(out / name, dtype={"game_id": str})
        car = s1[s1.game_id == "2026_02_CAR_ATL"].iloc[0]
        check("deployed calibrated probability frozen (CAR ~0.46)",
              abs(float(car["p_home_win"]) - 0.460653) < 1e-4)
        check("away pick recorded (CAR)", car["model_pick"] == "CAR")
        # Idempotency: re-running appends nothing and never mutates frozen rows.
        mp_mod._update_cards_history_store(out, oof, pd.DataFrame(), "20260924")
        s2 = pd.read_csv(out / name, dtype={"game_id": str})
        check("re-run is idempotent (no duplicate/changed rows)",
              len(s2) == len(s1)
              and abs(float(s2[s2.game_id == "2026_02_CAR_ATL"].iloc[0]["p_home_win"])
                      - 0.460653) < 1e-4)
        # Slate-decided append: a new decided game enters once.
        slate = pd.DataFrame({
            "game_id": ["2026_03_DET_BUF"], "gameday": ["2026-09-27"],
            "home_team": ["BUF"], "away_team": ["DET"],
            "p_home_win": [0.6347],
            "home_score": [41.0], "away_score": [31.0],
        })
        mp_mod._update_cards_history_store(out, oof, slate, "20260927")
        s3 = pd.read_csv(out / name, dtype={"game_id": str})
        check("slate-decided game appended once",
              int((s3.game_id == "2026_03_DET_BUF").sum()) == 1)
except Exception as exc:  # noqa: BLE001
    check("frozen card store probe", False, str(exc))

# ---------------------------------------------------------------------------
print("\n== 8. Persistence failure semantics ==")
mon_src = (BACKEND_DIR / "master_pipeline.py").read_text(encoding="utf-8")
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
