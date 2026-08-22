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

from config import (
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
from data_ingestion import (
    attach_market_lines,
    build_upcoming_slate,
    compute_elos_up_to,
    generate_synthetic_games,
    generate_synthetic_market_lines,
    load_game_events,
    filter_prior,
)
from explainability import compute_feature_drift, compute_shap_per_game
from training import last_ensemble_info
from github_sync import sync_artifacts
from training import (
    compute_metrics,
    calibration_buckets,
    feature_importance_weights,
    load_ensemble,
    persist_ensemble,
    predict_games,
    set_adaptive_weights,
    should_retrain,
    update_model_history,
    walk_forward_evaluate,
)

logger = logging.getLogger(__name__)


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
        "sp_name_home", "sp_name_away",
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
        # Finals for finished games (ESPN results merged onto the slate)
        "home_score", "away_score", "total_runs",
    ]
    cols = [c for c in out_cols if c in games.columns]
    games[cols].to_csv(path, index=False)
    return path


def _power_rankings_csv(games: pd.DataFrame, target_date_str: str) -> Path:
    """Write power_rankings_YYYYMMDD.csv artifact."""
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"{POWER_RANKINGS}_{target_date_str}.csv"

    teams = games["home_team"].unique()
    # Ties/postponements carry home_win = NULL — they are not wins or losses
    # and must not crash int() conversion or distort percentages.
    decided = games[games["home_win"].notna()]
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

        # L10 — last 10 DECIDED games
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


def _daily_calibration_rows(oof: Optional[pd.DataFrame]) -> list[dict]:
    """Per-day predicted-vs-actual win rates from walk-forward OOF predictions.

    Each row covers one game date; predictions come from the fold trained
    strictly on prior games (point-in-time safe by construction).
    """
    if oof is None or oof.empty or "home_win_prob_model" not in oof.columns:
        return []
    df = oof.copy()
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
        rows.append({
            "date": day.strftime("%Y%m%d"),
            "n_games": n,
            "wins": int((yt == 1).sum()),
            "losses": int((yt == 0).sum()),
            "metrics": {
                "auc": round(float(m.get("auc", 0.5)), 4),
                "brier": round(float(m.get("brier", 0.25)), 4),
                "logloss": round(float(m.get("logloss", 0.69)), 4),
                "ece": round(float(m.get("ece", 0.0)), 4),
            },
            "buckets": calibration_buckets(yt, yp),
        })
    rows.sort(key=lambda r: r["date"])
    return rows


def _calibration_json(
    metrics: dict[str, float],
    y_true, y_pred,
    target_date_str: str,
    n_games: int,
    oof: Optional[pd.DataFrame] = None,
) -> Path:
    """Write calibration_YYYYMMDD.json artifact.

    Headline buckets use ALL walk-forward out-of-sample predictions when
    available (a far richer curve than the target day alone); ``daily``
    carries per-day predicted-vs-actual for the date selector.
    """
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"{CALIBRATION}_{target_date_str}.json"

    import numpy as np
    if oof is not None and "home_win_prob_model" in getattr(oof, "columns", []):
        ot = pd.to_numeric(oof["home_win"], errors="coerce")
        op = pd.to_numeric(oof["home_win_prob_model"], errors="coerce")
        ok = ot.notna() & op.notna()
        if int(ok.sum()) >= len(y_true):
            y_true, y_pred = ot[ok].values, op[ok].values
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
    """Write predictions_history_YYYYMMDD.csv — every walk-forward OOF game
    prediction with its actual result.

    Feeds the Calibration page's per-game history table (the same games that
    feed the reliability diagram). Point-in-time safe by construction: each
    prediction comes from the fold trained strictly on prior games.
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
        # Candidate models behind the ensemble: name, blend weight (sums to
        # 1.0 over deployed members), and pooled out-of-fold AUC/Brier/LogLoss.
        "ensemble": ensemble if ensemble is not None else last_ensemble_info(),
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
        version: Model version string.
        games: Pre-built game DataFrame (from features.py). When provided,
               skips load_game_events() and uses this data for training.
        min_train_days: Warm-up period — skip validation folds that start
               before this many days of history (prevents tiny-training-fold noise).
        pbp_df: Optional pitch-level frame used to map probable-pitcher names
               to their rolling stat lines when predicting today's slate.

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
                min_train_days=min_train_days,
            )
            logger.info("Walk-forward metrics: %s", pooled_metrics)

            # Persist ensemble
            persist_ensemble(best_models, pooled_metrics, version=version, data_cutoff=target_date_str)
            update_model_history(pooled_metrics, version)
            summary["metrics"] = pooled_metrics
        else:
            best_models = ensemble["models"] if ensemble else {}
            pooled_metrics = ensemble["metrics"] if ensemble else {}
            all_predictions = None  # cached model: no fresh OOF predictions
            set_adaptive_weights(ensemble.get("adaptive_weights"))
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
            # state carried forward — never recycle yesterday's completed
            # games as "today" again.
            slate = build_upcoming_slate(games, target_date, pbp_df=pbp_df)
            if not slate.empty:
                logger.info(
                    "No completed games on %s — built %d-game upcoming slate "
                    "(pre-game PIT features)", target_date_str, len(slate),
                )
                games = pd.concat([games, slate], ignore_index=True)
                target_games = slate.copy()
            else:
                logger.warning(
                    "No games found for %s (schedule fetch empty) — falling "
                    "back to most recent games", target_date_str,
                )
                target_games = games.tail(15).copy()

        # ESPN drops probablePitcher once games start — restore the pitching
        # matchup and lines published by an earlier same-day run before they
        # get overwritten.
        target_games = _carry_forward_slate_details(target_games, target_date_str)

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
                path = _calibration_json(pooled_metrics, y_true[:min_len], y_pred[:min_len], target_date_str, len(target_games), oof=all_predictions)
                summary["artifacts"].append(str(path))
                hist_path = _predictions_history_csv(all_predictions, target_date_str)
                if hist_path is not None:
                    summary["artifacts"].append(str(hist_path))

        # 6. SHAP + Feature drift
        logger.info("Step 6: Explainability")
        compute_shap_per_game(best_models, target_games)

        # Feature drift: compare the last 7 days vs an ADJACENT season-local
        # window (~3x the current window, min 250 games). Comparing against
        # all history instead made every cumulative feature (elo, win_pct,
        # run_diff) look like ALERT drift, because those distributions widen
        # structurally as a season matures — a property of the feature, not
        # model health. Adjacent-but-not-tiny keeps it apples-to-apples while
        # giving quantile bin edges enough samples to be stable.
        # Decided games ONLY: pre-game slate rows carry the latest PIT state
        # forward (clustered near-identical values), so including them in the
        # current window distorted PSI for every feature.
        decided = games[games["home_win"].notna()]
        cutoff = pd.Timestamp(target_date) - pd.Timedelta(days=7)
        gd = pd.to_datetime(decided["game_date"])
        current = decided[gd >= cutoff]
        prior = decided[gd < cutoff]
        baseline = prior.tail(max(3 * len(current), 250)) if not prior.empty else prior
        if not baseline.empty and not current.empty:
            drift_df = compute_feature_drift(
                baseline, current, target_date_str,
                model_weights=feature_importance_weights(best_models),
            )
            summary["artifacts"].append(str(DATA_DELIVERY_DIR / f"feature_drift_{target_date_str}.csv"))
        else:
            drift_df = pd.DataFrame()

        # model monitor JSON
        path = _model_monitor_json(pooled_metrics, drift_df, target_date_str, version=version, ensemble=last_ensemble_info())
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
        "--statcast",
        action="store_true",
        help="Use comprehensive Statcast pipeline (pulls raw pitch data, builds all features)",
    )
    parser.add_argument(
        "--statcast-start",
        type=str,
        default=None,
        help="Statcast start date (YYYY-MM-DD) for --statcast mode",
    )
    parser.add_argument(
        "--statcast-end",
        type=str,
        default=None,
        help="Statcast end date (YYYY-MM-DD) for --statcast mode",
    )
    parser.add_argument(
        "--statcast-checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory for Statcast data (use Google Drive path for Colab)",
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

    if args.statcast:
        # Run Statcast pipeline via ingestion + features modules
        from ingestion import pull_statcast
        from features import build_features
        import tempfile

        start = args.statcast_start or (target - timedelta(days=120)).strftime("%Y-%m-%d")
        end = args.statcast_end or target.strftime("%Y-%m-%d")
        ckpt = Path(args.statcast_checkpoint_dir) if args.statcast_checkpoint_dir else Path(tempfile.mkdtemp())
        ckpt.mkdir(parents=True, exist_ok=True)

        pitches_path = ckpt / "pitches.parquet"
        pull_statcast(start, end, out_path=pitches_path, resume=True)
        game_df, pbp_df = build_features(pitches_path, ckpt)

        print(f"\nStatcast pipeline complete:")
        print(f"  Game-level: {game_df.shape}")
        print(f"  PBP-level:  {pbp_df.shape}")
        return {"status": "ok", "game_level": game_df, "pbp_level": pbp_df}

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
