"""
Daily pipeline orchestration for MLB Bet Predictor.

run_daily_pipeline(target_date) is the single entry point that:
1. Ingests game events (synthetic or real)
2. Attaches market lines
3. Runs walk-forward evaluation + trains ensemble
4. Predicts today's games
5. Computes SHAP + feature drift
6. Writes all artifacts to data_delivery/
7. Syncs to GitHub (unless skip_sync=True)

CLI:
    python pipeline.py --date 2026-08-09 --real --skip-sync
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .config import (
    CALIBRATION,
    DATA_DELIVERY_DIR,
    DATE_FMT,
    DATE_READABLE_FMT,
    MODEL_MONITOR,
    POWER_RANKINGS,
    RETRAIN_CADENCE_DAYS,
    TODAYS_GAMES,
    VERSION_KEY,
    TRAINED_AT_KEY,
    DATA_CUTOFF_KEY,
)
from .data_ingestion import (
    attach_market_lines,
    compute_elos_up_to,
    generate_synthetic_games,
    generate_synthetic_market_lines,
    load_game_events,
    filter_prior,
)
from .explainability import compute_feature_drift, compute_shap_per_game
from .github_sync import sync_artifacts
from .training import (
    compute_metrics,
    calibration_buckets,
    load_ensemble,
    persist_ensemble,
    predict_games,
    should_retrain,
    update_model_history,
    walk_forward_evaluate,
)

logger = logging.getLogger(__name__)


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
        "sp_era_home", "sp_k9_home", "sp_era_away", "sp_k9_away",
        "venue", "model_pick", "home_win",
    ]
    cols = [c for c in out_cols if c in games.columns]
    games[cols].to_csv(path, index=False)
    return path


def _power_rankings_csv(games: pd.DataFrame, target_date_str: str) -> Path:
    """Write power_rankings_YYYYMMDD.csv artifact."""
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"{POWER_RANKINGS}_{target_date_str}.csv"

    teams = games["home_team"].unique()
    rankings = []
    for team in teams:
        home_games = games[games["home_team"] == team]
        away_games = games[games["away_team"] == team]

        elo = home_games["home_elo"].mean() if not home_games.empty else 1500.0
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

        # L10 (approximate from synthetic data)
        recent = games[
            ((games["home_team"] == team) | (games["away_team"] == team))
        ].tail(10)
        l10_wins = 0
        for _, g in recent.iterrows():
            if g["home_team"] == team:
                l10_wins += int(g.get("home_win", 0))
            else:
                l10_wins += int(1 - g.get("home_win", 0))
        l10 = f"{l10_wins}-{len(recent) - l10_wins}"

        rankings.append({
            "team": team,
            "team_name": team,
            "elo": round(elo, 1),
            "wins": wins,
            "losses": losses,
            "record": f"{wins}-{losses}",
            "pct": pct,
            "run_diff": 0,
            "l10": l10,
            "home_pct": home_pct,
            "away_pct": away_pct,
        })

    df = pd.DataFrame(rankings).sort_values("elo", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"
    df.to_csv(path)
    return path


def _calibration_json(
    metrics: dict[str, float],
    y_true, y_pred,
    target_date_str: str,
    n_games: int,
) -> Path:
    """Write calibration_YYYYMMDD.json artifact."""
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"{CALIBRATION}_{target_date_str}.json"

    import numpy as np
    buckets = calibration_buckets(np.asarray(y_true), np.asarray(y_pred))

    # League-wide metadata
    evening_games = 0
    # (synthetic: count games with start hour >= 19)

    data = {
        "date": target_date_str,
        "n_games": n_games,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "metrics": metrics,
        "calibration_buckets": buckets,
        "league_total": n_games,
        "evening_games_league": evening_games,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def _model_monitor_json(
    metrics: dict[str, float],
    drift_df: pd.DataFrame,
    target_date_str: str,
    last_retrained: Optional[str] = None,
    version: str = "v3.2.1",
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

    # Drift summary
    n_warns = int((drift_df["status"] == "WARN").sum()) if not drift_df.empty else 0
    n_alerts = int((drift_df["status"] == "ALERT").sum()) if not drift_df.empty else 0
    warn_features = drift_df[drift_df["status"].isin(["WARN", "ALERT"])]["feature"].tolist() if not drift_df.empty else []

    data = {
        "date": target_date_str,
        "version": version,
        "last_retrained": last_retrained or datetime.now().strftime("%Y-%m-%d"),
        "next_retrain": (datetime.now() + timedelta(days=RETRAIN_CADENCE_DAYS)).strftime("%Y-%m-%d"),
        "metrics": metrics,
        "drift_summary": {
            "warnings": n_warns,
            "alerts": n_alerts,
            "features": warn_features,
        },
        "feature_drift": drift_df.to_dict(orient="records") if not drift_df.empty else [],
        "model_history": history,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def run_daily_pipeline(
    target_date: date,
    real: bool = False,
    skip_sync: bool = False,
    force_retrain: bool = False,
    max_eval_folds: int = 0,
    version: str = "v3.2.1",
) -> dict[str, Any]:
    """Run the full daily pipeline.

    Args:
        target_date: Date to generate predictions for.
        real: Use pybaseball real data (default: synthetic).
        skip_sync: Skip GitHub push.
        force_retrain: Retrain regardless of cadence.
        max_eval_folds: Cap walk-forward folds (0 = full history).
        version: Model version string.

    Returns:
        Summary dict with keys: status, artifacts, metrics, sync, errors
    """
    target_date_str = target_date.strftime(DATE_FMT)
    logger.info("=== Daily pipeline for %s ===", target_date_str)

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
        logger.info("Step 1: Loading game events (real=%s)", real)
        games = load_game_events(target_date, real=real)
        if games.empty:
            summary["status"] = "error"
            summary["errors"].append("No game events loaded")
            return summary

        logger.info("Loaded %d games", len(games))

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
            best_models, pooled_metrics, all_predictions = walk_forward_evaluate(
                train_games,
                max_eval_folds=max_eval_folds,
                force_retrain=force_retrain,
            )
            logger.info("Walk-forward metrics: %s", pooled_metrics)

            # Persist ensemble
            persist_ensemble(best_models, pooled_metrics, version=version, data_cutoff=target_date_str)
            update_model_history(pooled_metrics, version)
            summary["metrics"] = pooled_metrics
        else:
            best_models = ensemble["models"] if ensemble else {}
            pooled_metrics = ensemble["metrics"] if ensemble else {}
            summary["metrics"] = pooled_metrics

        # 4. Predict today's games (target_date only)
        logger.info("Step 4: Predicting games for %s", target_date_str)
        target_games = games[
            pd.to_datetime(games["game_date"]).dt.date == target_date
        ].copy()

        if target_games.empty:
            # Use all games if no specific target date games
            target_games = games.tail(15).copy()

        target_games = predict_games(best_models, target_games)

        # 5. Write artifacts
        logger.info("Step 5: Writing artifacts")

        # todays_games CSV
        path = _today_games_csv(target_games, target_date_str)
        summary["artifacts"].append(str(path))

        # power rankings
        path = _power_rankings_csv(games, target_date_str)
        summary["artifacts"].append(str(path))

        # calibration JSON
        if "home_win" in target_games.columns and "home_win_prob_model" in target_games.columns:
            y_true = target_games["home_win"].dropna().values
            y_pred = target_games["home_win_prob_model"].dropna().values
            if len(y_true) > 0 and len(y_pred) > 0:
                min_len = min(len(y_true), len(y_pred))
                path = _calibration_json(pooled_metrics, y_true[:min_len], y_pred[:min_len], target_date_str, len(target_games))
                summary["artifacts"].append(str(path))

        # 6. SHAP + Feature drift
        logger.info("Step 6: Explainability")
        compute_shap_per_game(best_models, target_games)

        # Feature drift: compare last 7 days vs all prior
        cutoff = pd.Timestamp(target_date) - pd.Timedelta(days=7)
        current = games[pd.to_datetime(games["game_date"]) >= cutoff]
        baseline = games[pd.to_datetime(games["game_date"]) < cutoff]
        if not baseline.empty and not current.empty:
            drift_df = compute_feature_drift(baseline, current, target_date_str)
            summary["artifacts"].append(str(DATA_DELIVERY_DIR / f"feature_drift_{target_date_str}.csv"))
        else:
            drift_df = pd.DataFrame()

        # model monitor JSON
        path = _model_monitor_json(pooled_metrics, drift_df, target_date_str, version=version)
        summary["artifacts"].append(str(path))

        # 7. GitHub sync
        if not skip_sync:
            logger.info("Step 7: Syncing to GitHub")
            sync_result = sync_artifacts()
            summary["sync"] = sync_result
            if not sync_result["pushed"]:
                logger.warning("GitHub sync failed: %s", sync_result.get("error"))
        else:
            logger.info("Step 7: Skipping GitHub sync")

    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        summary["status"] = "error"
        summary["errors"].append(str(e))

    logger.info("=== Pipeline complete: %s ===", summary["status"])
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLB Bet Predictor Daily Pipeline")
    parser.add_argument(
        "--date",
        type=str,
        default=date.today().strftime(DATE_FMT),
        help="Target date (YYYYMMDD format)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real pybaseball data instead of synthetic",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip GitHub push",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain regardless of cadence",
    )
    parser.add_argument(
        "--max-eval-folds",
        type=int,
        default=0,
        help="Max walk-forward evaluation folds (0 = full history)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v3.2.1",
        help="Model version string",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    target = datetime.strptime(args.date, DATE_FMT).date()

    summary = run_daily_pipeline(
        target_date=target,
        real=args.real,
        skip_sync=args.skip_sync,
        force_retrain=args.force_retrain,
        max_eval_folds=args.max_eval_folds,
        version=args.version,
    )

    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
