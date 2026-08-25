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
import os
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
from explainability import compute_feature_coverage, compute_feature_drift, compute_rolling_brier, compute_shap_per_game
from feature_metadata import generate_features_metadata
from calibration import is_identity
from features import (
    add_diff_features,
    add_env_level_features,
    add_form_delta_features,
    refine_dome_game_level,
)
from weather import apply_weather_features, fetch_day_weather, fetch_games_weather
from training import last_ensemble_info
from github_sync import sync_artifacts
from training import (
    compute_metrics,
    calibration_buckets,
    feature_importance_weights,
    get_last_calibrator,
    load_ensemble,
    persist_ensemble,
    predict_games,
    set_adaptive_weights,
    set_calibration,
    should_retrain,
    update_model_history,
    update_model_version_history,
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
        # Pitching matchup + stats: ESPN drops probablePitcher once a game
        # starts, so an evening rerun restores the morning's published
        # starters AND their ERA/K9 lines (names alone don't re-derive stats
        # without the pbp mapping), plus the StatsAPI ids.
        "sp_name_home", "sp_name_away",
        "sp_era_home", "sp_k9_home", "sp_era_away", "sp_k9_away",
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
        # Game state from ESPN — drives Live/Final status on the dashboard
        "game_state", "game_status_detail",
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
                for b in row["buckets"]:
                    lo = float(b["bucket"].split("–")[0]) / 100.0
                    hi = float(b["bucket"].split("–")[1].rstrip("%")) / 100.0
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
        if oc is not None:
            ok &= oc.notna()
        if int(ok.sum()) > 0:
            m_raw = compute_metrics(ot[ok].values, op[ok].values)
            calibrator = get_last_calibrator()
            cal_section = {
                "method": "platt" if not is_identity(calibrator) else "identity",
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

    # League-wide metadata
    evening_games = 0
    # (synthetic: count games with start hour >= 19)

    data = {
        "date": target_date_str,
        "n_games": n_games,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "metrics": metrics,
        "calibration_buckets": buckets,
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
    """Write predictions_history_YYYYMMDD.csv — every walk-forward OOF game
    prediction with its actual result.

    Feeds the Calibration page's per-game history table (the same games that
    feed the reliability diagram). Point-in-time safe by construction: each
    prediction comes from the fold trained strictly on prior games.

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
        # Per-feature non-null coverage per window (measured vs default-filled).
        # Visual backstop for silent data starvation — see compute_feature_coverage.
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
        # truth generated from FEATURE_COLS — see feature_metadata.py.
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
    wind/air-density features are null for all history except dome zeros —
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
    # identifier the results overlay uses).  Rows without one — or without a
    # matching official start — are skipped: never sample at a fabricated hour.
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
        logger.warning("Weather backfill: no authoritative start times matched — skipped")
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
    # but only wind fills honestly — the feed has no humidity, so air_density
    # stays NULL for these records by the module's no-fabrication rule.
    "statsapi_gamefeed",
}

# Fill games the Open-Meteo archive could not observe from the per-game
# StatsAPI feed (paced ~2.5 req/s, one-off per cached game_pk).
STATSAPI_WEATHER_FILL = True


def _load_weather_cache(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Weather cache unreadable (%s) — rebuilding", exc)
        return {}
    if "source" not in df.columns:
        # Legacy caches predate provenance and may contain climatology values
        # marked available=True. Never reuse them as observed weather.
        logger.warning("Weather cache has no source column — invalidating legacy cache")
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
    need = decided & pks.notna() & ~pks_int.isin(cached_pks)
    if need.any():
        subset = games[need]
        gd = pd.to_datetime(games["game_date"], errors="coerce")
        starts = fetch_game_start_times(gd[need].min().date(), gd[need].max().date())
        # Loud coverage gate: a silently truncated schedule source is how an
        # ENTIRE SEASON of weather features went null while every log line
        # looked healthy (2470/2477 'fetched' — of only the games attempted).
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
                    "games in %d (%s→%s) — schedule source may be truncating "
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
                    cache[pk] = {k: w.get(k) for k in _WEATHER_CACHE_COLS}
                    new += 1
            _save_weather_cache(_weather_cache_path(), cache)
            logger.info("Weather history: fetched %d new games (cache now %d)",
                        new, len(cache))
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
        min_train_days: Warm-up period — skip validation folds that start
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
        # not officially final — a partial score can never ship as a final
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

        # Real point-in-time weather for features 30–31 (wind advantage,
        # air density).  One Open-Meteo request per (stadium, day); games
        # without a strictly-prior observation get NULL weather features
        # (never a fabricated 0).  Weather is only attached when the frame
        # carries GENUINELY observed start times — fabricated defaults (e.g.
        # load_game_features' 19:00 UTC fallback) are excluded via the
        # start_time_observed tag so we never fetch weather for the wrong
        # hour.
        # ALWAYS recompute every diff feature from the raw home/away columns.
        # Pre-built exports may contain column names with stale or
        # schema-drifted values (e.g. win_pct_diff NaN from the DuckDB first
        # pass, renamed pitcher windows, weather defaults) — presence of a
        # column is never evidence its values are current. Recomputation is a
        # cheap vectorized pass and runs exactly once, BEFORE any weather
        # application so the two weather-driven features are applied on top
        # of fresh diffs afterwards.
        logger.info("Recomputing all diff features from raw home/away columns")
        # Final computation: official results already applied, so record
        # columns must exist — a missing win_pct_diff here is a real problem.
        games = add_diff_features(games, require_records=True)
        # Momentum form deltas (recent − season-to-date baseline). Idempotent:
        # SQL-shipped columns win; missing ones are computed from the shipped
        # recent/season columns when both exist (NaN otherwise — imputed by
        # the existing paths). Moneyline-only: the run engine excludes
        # *_delta_* columns in derive_run_features.
        games = add_form_delta_features(games)
        if WEATHER_BACKFILL_ALL:
            # Full-history weather mode: the cache-backed backfill applies
            # real point-in-time weather to every decided game (see
            # _attach_weather_history) — no reliance on the trailing window.
            # add_diff_features may assign legacy dome-neutral defaults before
            # the real weather pass. Air density is not safely neutral indoors
            # without an observation, so clear it before applying cache data.
            games["air_density_velocity_boost"] = pd.NA
            try:
                games = _attach_weather_history(games, target_date)
            except Exception as exc:
                logger.warning(
                    "Full weather backfill failed (features stay null): %s", exc
                )
                try:
                    games = _attach_recent_weather(games, target_date)
                except Exception:
                    pass
        else:
            # Apply observed weather separately from diffs: this prevents the
            # legacy dome default from fabricating an air-density value when
            # the provider returns no observation.
            games["air_density_velocity_boost"] = pd.NA
            if "start_time_utc" in games.columns:
                real_start = games["start_time_utc"].notna()
                if "start_time_observed" in games.columns:
                    real_start &= games["start_time_observed"].fillna(True).astype(bool)
                if real_start.any():
                    weather = {}
                    try:
                        weather = fetch_games_weather(games.loc[real_start])
                    except Exception as e:
                        logger.warning(
                            "Weather fetch failed for history (features 30–31 stay NULL): %s", e
                        )
                    # apply_weather_features is imported at module level; do NOT
                    # re-import it here — a branch-local binding makes the name
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
            roof_states = {}
            if roof_cache.exists():
                roof_states = {
                    int(k): v for k, v in json.loads(
                        roof_cache.read_text()).items()
                    if v in ("open", "closed")}
            games = refine_dome_game_level(games, roof_states=roof_states)
            games = add_env_level_features(games)
        except Exception as exc:
            logger.warning("Env-level feature pass failed (level columns may "
                           "be absent this run): %s", exc)
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
        else:
            best_models = ensemble["models"] if ensemble else {}
            pooled_metrics = ensemble["metrics"] if ensemble else {}
            all_predictions = None  # cached model: no fresh OOF predictions
            set_adaptive_weights(ensemble.get("adaptive_weights"))
            set_calibration(ensemble.get("calibrator"))
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
                if weather:
                    slate = apply_weather_features(slate, weather)
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

        # Official-results overlay on today's board.  Slate rows carry no
        # StatsAPI game_pk, so the overlay falls back to (date + teams).
        # Live/preview games get home_win=NULL; finals get authoritative
        # scores — never a mid-game snapshot as a final.
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

        target_games = predict_games(best_models, target_games)

        # 5. Write artifacts
        logger.info("Step 5: Writing artifacts")

        # todays_games CSV
        path = _today_games_csv(target_games, target_date_str)
        summary["artifacts"].append(str(path))

        # power rankings
        path = _power_rankings_csv(games, target_date_str)
        summary["artifacts"].append(str(path))

        # calibration JSON — written on EVERY run.  The walk-forward OOF frame
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
                # No finals yet today: fall back to OOF pairs only — labels
                # are real outcomes from completed games, never fabricated.
                _ot = pd.to_numeric(all_predictions["home_win"], errors="coerce")
                _op = pd.to_numeric(all_predictions["home_win_prob_model"], errors="coerce")
                _ok = _ot.notna() & _op.notna()
                cal_yt, cal_yp = _ot[_ok].values, _op[_ok].values
            path = _calibration_json(pooled_metrics, cal_yt, cal_yp, target_date_str, len(target_games), oof=all_predictions)
            summary["artifacts"].append(str(path))
            hist_path = _predictions_history_csv(all_predictions, target_date_str)
            if hist_path is not None:
                summary["artifacts"].append(str(hist_path))
            # Rolling Brier series over the same OOF history — computed from
            # the raw blend through the DEPLOYED calibrator (get_last_calibrator
            # holds exactly the map predict-time and the charts use).
            rolling_brier = compute_rolling_brier(
                all_predictions, target_date_str, calibrator=get_last_calibrator()
            )
            summary["artifacts"].append(
                str(DATA_DELIVERY_DIR / f"rolling_brier_{target_date_str}.json")
            )
        # Feature metadata (dashboard tooltips) — walks FEATURE_COLS itself so
        # new features appear (or warn loudly); routing derived from live config.
        features_metadata = generate_features_metadata(target_date_str)
        summary["artifacts"].append(
            str(DATA_DELIVERY_DIR / f"features_metadata_{target_date_str}.json")
        )

        # Run engine (Phase 3): OOF re-derivation on the SAME fixed folds →
        # α(λ) dispersion curves fitted PRE-HOLDOUT only → NB Monte-Carlo
        # market grid (totals 6.5–12.5, run lines −0.5…−3.5) for OOF + today's
        # slate → agreement conflicts vs the moneyline ensemble. Must never
        # take down the rest of the run.
        run_engine_block = None
        try:
            from run_engine import run_engine_daily
            _re = run_engine_daily(games, target_games, target_date_str)
            run_engine_block = _re["block"]
            summary["artifacts"].extend(_re["artifacts"])
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
        # structurally as a season matures — a property of the feature, not
        # model health. Adjacent-but-not-tiny keeps it apples-to-apples while
        # giving quantile bin edges enough samples to be stable.
        # Decided games ONLY: pre-game slate rows carry the latest PIT state
        # forward (clustered near-identical values), so including them in the
        # current window distorted PSI for every feature.
        decided = games[games["home_win"].notna()]
        cutoff = pd.Timestamp(target_date) - pd.Timedelta(days=7)
        gd = pd.to_datetime(decided["game_date"])
        # Chronological order is required: tail(N) on an unordered frame
        # mixes arbitrary seasons into the baseline window.
        decided = decided.sort_values("game_date")
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
            coverage_df = compute_feature_coverage(baseline, current, target_date_str)
            summary["artifacts"].append(str(DATA_DELIVERY_DIR / f"feature_coverage_{target_date_str}.csv"))
        else:
            drift_df = pd.DataFrame()
            coverage_df = pd.DataFrame()

        # model monitor JSON
        path = _model_monitor_json(pooled_metrics, drift_df, target_date_str, version=version, ensemble=last_ensemble_info(), coverage_df=coverage_df, rolling_brier=rolling_brier, features_metadata=features_metadata, run_engine=run_engine_block)
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
        default=None,
        help="Model version string (default: auto, vYYYY.MM.DD of the run date)",
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
