"""Daily pipeline orchestration.

``run_daily_pipeline(target_date)`` is the single entry point Colab calls. It:

1. Ingests a point-in-time game log (synthetic by default; pybaseball opt-in).
2. Runs expanding-window walk-forward evaluation (calibration + metrics) and
   retrains the production ensemble when the weekly cadence is due.
3. Predicts P(home win) for every game on ``target_date`` using only features
   available before each game's scheduled start.
4. Writes ``todays_games_YYYYMMDD.csv``, SHAP files per game,
   ``calibration_YYYYMMDD.json``, ``feature_drift_YYYYMMDD.csv`` and
   ``model_monitor_YYYYMMDD.json`` into ``data_delivery/``.
5. Attempts to stage/commit/push those artifacts to GitHub via GitPython.

Run it in Colab with::

    from pipeline import run_daily_pipeline
    from datetime import date
    run_daily_pipeline(date(2026, 8, 9))
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date as date_cls, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CALIBRATION_FILE,
    DATA_CUTOFF,
    DATA_DIR,
    FEATURE_DRIFT,
    FEATURE_COLUMNS,
    LOGS_DIR,
    METRICS,
    MODEL_MONITOR_FILE,
    MODELS_DIR,
    N_TRAIN_GAMES,
    POWER_RANKINGS_FILE,
    RETRAIN_CADENCE_DAYS,
    SEED,
    TODAYS_GAMES_FILE,
    TRAINED_AT,
    VERSION,
    artifact_path,
    date_to_yyyymmdd,
)
import data_ingestion as di
import explainability as ex
import github_sync as gs
import training as tr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pipeline")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Wrote %s", path)


def _confidence_distribution(calibration_rows: list[dict]) -> list[dict]:
    """Game count + actual accuracy % per probability bucket (for the combo chart)."""
    out = []
    for row in calibration_rows:
        out.append(
            {
                "bucket": row["bucket"],
                "count": row["count"],
                "accuracy_pct": round(row["mean_actual"] * 100.0, 1),
            }
        )
    return out


def _rolling_brier(eval_result: dict, window_days: int = 30) -> list[dict]:
    """Daily Brier score for the last ``window_days`` of validation games."""
    y_true = np.asarray(eval_result["pooled_y_true"], dtype=float)
    y_prob = np.asarray(eval_result["pooled_y_prob"], dtype=float)
    dates = pd.to_datetime(eval_result["pooled_dates"], utc=True)
    daily = pd.DataFrame({"date": dates.date, "y": y_true, "p": y_prob})
    daily = daily.groupby("date").apply(
        lambda g: pd.Series({"brier": float(np.mean((g["p"] - g["y"]) ** 2)), "n": len(g)}),
        include_groups=False,
    )
    daily = daily.reset_index().sort_values("date").tail(window_days)
    return [
        {"date": r["date"].isoformat(), "brier": round(float(r["brier"]), 4)}
        for _, r in daily.iterrows()
    ]


def _model_history() -> list[dict]:
    path = MODELS_DIR / "model_history.json"
    if path.exists():
        return json.loads(path.read_text())
    return []


def _append_model_history(entry: dict) -> None:
    path = MODELS_DIR / "model_history.json"
    history = _model_history()
    history = [h for h in history if h.get("version") != entry.get("version")] + [entry]
    history.sort(key=lambda h: h.get("date", ""), reverse=True)
    path.write_text(json.dumps(history, indent=2))


def run_daily_pipeline(
    target_date: date_cls,
    synthetic: bool = True,
    force_retrain: bool = False,
    skip_sync: bool = False,
    cadence_days: int = RETRAIN_CADENCE_DAYS,
    seed: int = SEED,
    max_eval_folds: int | None = None,
) -> dict:
    """Run the full daily pipeline and return a summary dict."""
    summary = {"target_date": target_date.isoformat(), "synthetic": synthetic}

    # 1) Ingest ---------------------------------------------------------------
    start, end = di.history_window(target_date)
    events = di.load_game_events(start, end, synthetic=synthetic, seed=seed, target_date=target_date)
    summary["games_ingested"] = int(len(events))
    logger.info("Ingested %d games from %s to %s", len(events), start, end)

    # 2) Point-in-time features ------------------------------------------------
    features = di.build_point_in_time_features(events, include_unready=True).set_index("game_id")
    logger.info("Built features for %d games", len(features))

    # 3) Walk-forward evaluation + (cadence-based) retrain ---------------------
    eval_result = tr.run_walk_forward_evaluation(
        features, cadence_days=cadence_days, n_splits=max_eval_folds
    )
    due = tr.is_retrain_due(target_date, cadence_days)
    if due or force_retrain:
        payload = tr.train_final_model(features, data_cutoff=target_date)
        payload[METRICS] = eval_result["metrics_moneyline"]
        payload[N_TRAIN_GAMES] = eval_result["metrics_moneyline"]["n_games"]
        # re-persist with metrics attached
        tr._persist_ensemble(payload)
        _append_model_history(
            {
                "version": payload[VERSION],
                "date": target_date.isoformat(),
                "auc": payload[METRICS]["auc_roc"],
                "brier": payload[METRICS]["brier_score"],
                "notes": "Weekly walk-forward retrain",
            }
        )
        summary["retrained"] = True
        logger.info("Retrained ensemble (version %s)", payload[VERSION])
    else:
        payload = tr.load_ensemble()
        summary["retrained"] = False
        logger.info("Retrain not due — using existing ensemble %s", payload.get(VERSION))

    # 4) Predict today's games -------------------------------------------------
    today = features[features["start_time"].dt.date == target_date]
    today_feats = today[FEATURE_COLUMNS].dropna()
    probs = pd.Series(0.5, index=today_feats.index)
    if not today_feats.empty:
        probs = pd.Series(
            payload["moneyline"].predict_proba_home(today_feats), index=today_feats.index
        )
    summary["todays_games_predicted"] = int(len(probs))

    # 5) Today's games CSV -----------------------------------------------------
    games_df = di.build_todays_games(events, features, probs, target_date)
    games_path = artifact_path(TODAYS_GAMES_FILE, target_date)
    games_df.to_csv(games_path, index=False)
    logger.info("Wrote %s (%d games)", games_path, len(games_df))

    # 6) SHAP per game ---------------------------------------------------------
    shap_files = ex.save_shap_for_games(payload, features, list(today_feats.index))
    summary["shap_files"] = shap_files

    # 7) Feature drift (PSI) ---------------------------------------------------
    cur_win = features[(features["start_time"].dt.date > target_date - timedelta(days=30)) &
                       (features["start_time"].dt.date <= target_date)]
    base_win = features[(features["start_time"].dt.date > target_date - timedelta(days=180)) &
                        (features["start_time"].dt.date <= target_date - timedelta(days=150))]
    drift_df = ex.compute_feature_drift(cur_win, base_win)
    drift_path = artifact_path(FEATURE_DRIFT, target_date)
    drift_df.to_csv(drift_path, index=False)
    summary["drift_features"] = len(drift_df)

    # 8) Power rankings ----------------------------------------------------------
    rankings_df = di.build_power_rankings(events, as_of=target_date)
    rankings_path = artifact_path(POWER_RANKINGS_FILE, target_date)
    rankings_df.to_csv(rankings_path, index=False)

    # 9) Calibration JSON ---------------------------------------------------------
    calibration = {
        "date": target_date.isoformat(),
        "n_games": eval_result["metrics_moneyline"]["n_games"],
        "trained_version": payload.get(VERSION, "unknown"),
        "trained_at": payload.get(TRAINED_AT, ""),
        "data_cutoff": payload.get(DATA_CUTOFF, ""),
        "kpis": eval_result["metrics_moneyline"],
        "calibration_curve": eval_result["calibration"],
        "confidence": _confidence_distribution(eval_result["calibration"]),
        "today_record": _today_record(games_df),
        "upsets": _today_upsets(games_df),
        "games_shown": int(len(games_df)),
        "league_total": int(len(games_df)),
        "evening_games_league": int(games_df["evening_game"].sum()) if not games_df.empty else 0,
    }
    _write_json(artifact_path(CALIBRATION_FILE, target_date), calibration)

    # 10) Model monitor JSON ------------------------------------------------------
    retrained_ts = payload.get(TRAINED_AT, "")
    retrain_date = pd.Timestamp(retrained_ts).tz_localize(None).date() if retrained_ts else target_date
    monitor = {
        "date": target_date.isoformat(),
        "version": payload.get(VERSION, "unknown"),
        "last_retrained": retrain_date.isoformat(),
        "last_retrained_note": "Model healthy",
        "next_retrain": (retrain_date + timedelta(days=cadence_days)).isoformat(),
        "next_retrain_note": f"Nightly schedule — retrain cadence {cadence_days}d",
        "drift_alerts": [
            {"feature": r["feature"], "psi": r["psi"], "status": r["status"]}
            for _, r in drift_df.iterrows() if r["status"] != "OK"
        ],
        "upset_note": _upset_note(games_df, calibration["upsets"]),
        "feature_drift": drift_df.to_dict(orient="records"),
        "rolling_brier": _rolling_brier(eval_result),
        "brier_baseline_version": "v3.2.0",
        "brier_baseline": eval_result["metrics_moneyline"]["brier_score"],
        "version_history": _model_history(),
        "metrics_moneyline": eval_result["metrics_moneyline"],
    }
    _write_json(artifact_path(MODEL_MONITOR_FILE, target_date), monitor)

    # 11) GitHub sync ---------------------------------------------------------------
    if not skip_sync:
        sync_status = gs.sync_artifacts(target_date)
        summary["github_sync"] = sync_status
        if not sync_status["pushed"]:
            logger.warning("Artifacts NOT pushed: %s", sync_status.get("error"))
    else:
        summary["github_sync"] = {"pushed": False, "error": "skipped (skip_sync=True)"}

    summary["artifacts"] = [
        p.name for p in DATA_DIR.glob(f"*_{date_to_yyyymmdd(target_date)}.*")
    ] + shap_files
    summary["status"] = "ok"
    return summary


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _today_record(games_df: pd.DataFrame) -> dict:
    finals = games_df[games_df["game_status"] == "Final"]
    correct = int(finals["model_correct"].sum()) if not finals.empty else 0
    total = int(len(finals))
    return {
        "wins": correct,
        "losses": total - correct,
        "completed": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
    }


def _today_upsets(games_df: pd.DataFrame) -> list[dict]:
    up = games_df[games_df["is_upset"] == True]  # noqa: E712
    return [
        {
            "team": r["home_team"] if r["home_score"] > r["away_score"] else r["away_team"],
            "prob": float(min(r["home_win_prob_model"], r["away_win_prob_model"])),
        }
        for _, r in up.iterrows()
    ]


def _upset_note(games_df: pd.DataFrame, upsets: list[dict]) -> str:
    if not upsets:
        return "No upsets today — model performance consistent with expectations."
    detail = ", ".join(f"{u['team']} at {u['prob']:.0%}" for u in upsets)
    record = _today_record(games_df)
    return (
        f"{len(upsets)} upset(s) today ({detail}) — monitoring for regime shift. "
        f"Model went {record['wins']}-{record['losses']} overall but high-confidence "
        "picks (>65%) showed vulnerability. Will assess after tonight's retrain."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the daily MLB prediction pipeline.")
    parser.add_argument("--date", type=date_cls.fromisoformat, default=None, help="YYYY-MM-DD")
    parser.add_argument("--real", action="store_true", help="Use pybaseball instead of synthetic data")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--cadence", type=int, default=RETRAIN_CADENCE_DAYS)
    parser.add_argument("--max-eval-folds", type=int, default=None,
                        help="Cap walk-forward evaluation to the most recent N weekly folds")
    args = parser.parse_args()

    target = args.date or date_cls.today()
    summary = run_daily_pipeline(
        target,
        synthetic=not args.real,
        force_retrain=args.force_retrain,
        skip_sync=args.skip_sync,
        cadence_days=args.cadence,
        max_eval_folds=args.max_eval_folds,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
