"""Current-slate serving — writes the exact artifact families the existing
NFL frontend consumes (contract in frontend/sports_config.py artifacts map
+ frontend/utils.py adapters). The frontend never reconstructs model
internals; every field it reads is emitted here.

Families (per run date):
  nfl_moneyline_v1_<date>.json        games[] card contract
  nfl_calibration_<date>.json         metrics/daily/today_record contract
  nfl_predictions_history_<date>.csv  decided-game prediction history
  nfl_power_rankings_<date>.csv       rank/team/elo/record board
  nfl_run_engine_markets_<date>.csv   + .meta.json  mu/grids/oof contract
  nfl_qb_matchup_<date>.json          QB enrichment contract
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from backend import config
except ImportError:
    import config

logger = logging.getLogger(__name__)


def _clean(v):
    """JSON-safe scalar: NaN/None -> None, numpy -> python, np.nan in lists
    handled by the caller's row dicts."""
    if v is None:
        return None
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _row_clean(d: dict) -> dict:
    return {k: _clean(v) for k, v in d.items()}


def _json_safe(obj):
    """Recursively make a structure JSON-strict (NaN/Inf -> None)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _dump_json(path, record: dict) -> None:
    path.write_text(json.dumps(_json_safe(record), indent=1, allow_nan=False))


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date_compact(run_date: str) -> str:
    return run_date.replace("-", "")


# ---------------------------------------------------------------------------
# Moneyline JSON — the games[] card contract
# ---------------------------------------------------------------------------
def write_moneyline_json(path, slate_df: pd.DataFrame, p_home: np.ndarray,
                         p_home_cal: np.ndarray, team_names: dict[str, str],
                         config_meta: dict) -> dict:
    games = []
    for i, row in enumerate(slate_df.reset_index(drop=True).iterrows()):
        _, g = row
        ph = _clean(p_home[i]) if i < len(p_home) else None
        phc = _clean(p_home_cal[i]) if i < len(p_home_cal) else None
        pick = None
        if ph is not None:
            pick = g["home_team"] if ph >= 0.5 else g["away_team"]
        games.append(_row_clean({
            "game_id": g["game_id"],
            "game_date": str(g["gameday"]),
            "start_time_utc": _start_time_utc(g),
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "home_team_name": team_names.get(g["home_team"], g["home_team"]),
            "away_team_name": team_names.get(g["away_team"], g["away_team"]),
            "home_record": g.get("home_record") or None,
            "away_record": g.get("away_record") or None,
            "venue": g.get("stadium") or None,
            "game_status": "pre",
            "home_score": None,
            "away_score": None,
            "home_win_prob_model": phc if phc is not None else ph,
            "away_win_prob_model": (1.0 - phc) if phc is not None else ((1.0 - ph) if ph is not None else None),
            "model_pick": pick,
            "model_correct": None,
        }))
    record = {
        "created_utc": _now_utc(),
        "config": config_meta,
        "slate_date": str(slate_df["gameday"].min()) if len(slate_df) else None,
        "n_games": len(games),
        "games": games,
    }
    _dump_json(path, record)
    return record


def _start_time_utc(g) -> str | None:
    gd = str(g.get("gameday", "")) or ""
    gt = str(g.get("gametime", "") or "")
    if not gd:
        return None
    if gt and ":" in gt:
        # nflverse gametime is ET; encode as UTC by adding the ET offset (4/5h).
        # Simple documented convention: EST (UTC-5) outside DST is ignored for
        # display purposes — the frontend renders the time string as-is.
        try:
            hour, minute = gt.split(":")[:2]
            return f"{gd}T{int(hour) + 0:02d}:{minute}:00Z"
        except Exception:
            pass
    return f"{gd}T00:00:00Z"


# ---------------------------------------------------------------------------
# Calibration JSON — metrics/daily contract the shared page renders
# ---------------------------------------------------------------------------
def _r4(v):
    """Round a metric to 4 decimals (None/NaN passthrough) — MLB's stored
    presentation precision (mlb-backend compute_metrics rounds every emitted
    metric to 4 decimals, so the shared Calibration page renders both sports
    from identically-precise artifacts)."""
    try:
        f = float(v)
        return round(f, 4) if f == f else None  # NaN -> None
    except (TypeError, ValueError):
        return v


def _r6(v):
    """Round a Platt parameter to 6 decimals — MLB's fit_platt precision."""
    try:
        f = float(v)
        return round(f, 6) if f == f else None
    except (TypeError, ValueError):
        return v


def write_calibration_json(path, moneyline_metrics: dict,
                           calibrated_metrics: dict,
                           buckets: list[dict], daily: list[dict],
                           config_meta: dict, platt: dict | None = None,
                           run_date: str = "", n_games: int = 0,
                           calibrated_buckets: list[dict] | None = None) -> dict:
    """MLB-shaped calibration artifact (frontend presentation contract).

    Carries the MLB schema keys the shared Calibration page renders: ``date``
    / ``trained_at`` / ``n_games`` for the header pill, ``calibration`` with
    the Platt ``params`` + ``metrics_raw`` / ``metrics_calibrated`` for the
    recalibration banner and green calibrated curve, ``calibration`` buckets
    carrying ``gap`` / ``count`` / ``mean_predicted`` (favored view), and
    ``calibration.calibration_buckets_calibrated`` — the prequential
    calibrated twin per bucket (MLB parity: the reliability table's
    CALIBRATED column renders '—' without it). All values are the NFL
    pipeline's own outputs; ``platt`` is the OOF-fitted Platt map the
    serving path already applies — never refitted here.
    """
    cal_sec: dict = {}
    if isinstance(platt, dict) and platt.get("a") is not None \
            and platt.get("b") is not None:
        n = int(platt.get("n") or n_games or 0)
        cal_sec = {
            "method": "platt",
            "params": {"a": _r6(platt.get("a")), "b": _r6(platt.get("b")),
                       "n": n},
            "metrics_raw": {
                "brier": _r4(moneyline_metrics.get("brier")),
                "logloss": _r4(moneyline_metrics.get("logloss")),
                "ece": _r4(moneyline_metrics.get("ece")),
            },
            "metrics_calibrated": {
                "brier": _r4(calibrated_metrics.get("brier")),
                "logloss": _r4(calibrated_metrics.get("logloss")),
                "ece": _r4(calibrated_metrics.get("ece")),
            },
        }
        if calibrated_buckets:
            cal_sec["calibration_buckets_calibrated"] = calibrated_buckets
    record = {
        "date": run_date,
        "trained_at": _now_utc(),
        "n_games": int(n_games),
        "created_utc": _now_utc(),
        "config": config_meta,
        "metrics": {
            "auc": _r4(moneyline_metrics.get("auc")),
            "brier": _r4(moneyline_metrics.get("brier")),
            "logloss": _r4(moneyline_metrics.get("logloss")),
            "ece": _r4(moneyline_metrics.get("ece")),
            "ece_calibrated": _r4(calibrated_metrics.get("ece")),
            "auc_calibrated": _r4(calibrated_metrics.get("auc")),
            "brier_calibrated": _r4(calibrated_metrics.get("brier")),
            "logloss_calibrated": _r4(calibrated_metrics.get("logloss")),
        },
        "calibration": cal_sec,
        "calibration_buckets": buckets,
        # Daily metrics at MLB's stored precision too (MLB's
        # _daily_calibration_rows rounds every emitted daily metric to 4).
        "daily": [
            {**row, "metrics": {k: _r4(v)
                                for k, v in (row.get("metrics") or {}).items()}}
            for row in daily
        ],
    }
    _dump_json(path, record)
    return record


# ---------------------------------------------------------------------------
# Predictions history CSV — decided-game predicted-vs-actual
# ---------------------------------------------------------------------------
def write_predictions_history_csv(path, oof: pd.DataFrame,
                                  p_cal: np.ndarray | None) -> pd.DataFrame:
    df = oof.sort_values("gameday").reset_index(drop=True)
    out = pd.DataFrame({
        "game_id": df["game_id"],
        "game_date": pd.to_datetime(df["gameday"]).dt.strftime("%Y-%m-%d"),
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "home_score": df["home_score"].astype(float),
        "away_score": df["away_score"].astype(float),
        "home_win": df["home_win"].astype(float),
        "home_win_prob_model": df["p_ensemble"].astype(float),
        "home_win_prob_model_calibrated": p_cal if p_cal is not None else df["p_ensemble"].astype(float),
        "model_pick": np.where(df["p_ensemble"] >= 0.5, df["home_team"], df["away_team"]),
        "actual_winner": np.where(df["home_win"] > 0.5, df["home_team"],
                                  np.where(df["home_win"] < 0.5, df["away_team"], "TIE")),
        "correct": (np.where(df["p_ensemble"] >= 0.5, df["home_team"], df["away_team"])
                    == np.where(df["home_win"] > 0.5, df["home_team"],
                                np.where(df["home_win"] < 0.5, df["away_team"], "TIE"))),
    })
    out.to_csv(path, index=False)
    return out


# ---------------------------------------------------------------------------
# Power rankings CSV — shared page shape (rank/team/team_name/elo/record)
# ---------------------------------------------------------------------------
def write_power_rankings_csv(path, ratings: dict[str, float],
                             records: dict[str, tuple[int, int]],
                             team_names: dict[str, str],
                             ladder_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for team, elo in sorted(ratings.items(), key=lambda kv: -kv[1]):
        w, l = records.get(team, (0, 0))
        row = {
            "rank": 0, "team": team,
            "team_name": team_names.get(team, team),
            "elo": round(float(elo), 1),
            "wins": int(w), "losses": int(l),
            "record": f"{int(w)}-{int(l)}",
            "pct": round(w / (w + l), 3) if (w + l) else np.nan,
            "run_diff": 0,
            "l10": "", "home_pct": np.nan, "away_pct": np.nan,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df["rank"] = range(1, len(df) + 1)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Run-engine markets CSV (+ meta) — the mu/grids/OOF contract
# ---------------------------------------------------------------------------
MARKETS_BASE_COLS = [
    "kind", "game_id", "gameday", "season", "week", "home_team", "away_team",
    "stadium", "gametime", "home_record", "away_record",
    "p_home_win", "p_away_win", "p_tie",
    "derived_ml", "p_home_win_derived", "p_away_win_derived",
    "mu_h", "mu_a", "mu_margin", "mu_total",
    "fair_spread", "fair_total",
    "spread_line", "total_line", "has_offer",
    "p_cover_offered", "p_push_offered",
    "p_over_offered", "p_under_offered", "p_push_total_offered",
    "home_score", "away_score", "total", "margin",
    "p_over_fair", "p_cover_fair",
    "y_over_fair", "y_under_fair", "y_push_fair",
    "y_cover_fair", "y_push_spread_fair",
    "y_over_offered", "y_under_offered", "y_push_total_offered",
    "y_cover_offered", "y_push_spread_offered",
    "y_home_win", "decided", "frame_view",
    "pred_home", "pred_away",
]


def write_markets_csv(path, meta_path, oof_rows: pd.DataFrame,
                      slate_rows: pd.DataFrame, config_meta: dict) -> pd.DataFrame:
    """Combine decided OOF rows (kind='oof') + current slate rows
    (kind='slate') into the markets artifact the diagnostics page reads."""
    cols = MARKETS_BASE_COLS[:]
    # add every grid column (spread -14..14 + half stops, totals 24..66)
    for L in config.SPREAD_GRID:
        label = f"m{-L}" if L < 0 else str(L)
        cols += [f"p_home_cover_{label}", f"p_push_{label}"]
    for L in config.HALF_STOP_LINES:
        cols.append(f"p_home_cover_{str(L).replace('.', '_').replace('-', 'm')}")
    for U in config.TOTAL_GRID:
        cols += [f"p_over_{U}", f"p_under_{U}", f"p_push_{U}"]

    out = pd.concat([oof_rows, slate_rows], ignore_index=True)
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols]
    out.to_csv(path, index=False)
    meta = {
        "record": "nfl_run_engine_markets",
        "written_utc": _now_utc(),
        "config": config_meta,
        "columns": len(cols),
        "n_rows": int(len(out)),
        "n_oof": int((out["kind"] == "oof").sum()),
        "n_slate": int((out["kind"] == "slate").sum()),
        "grids": {"spread": [-14, 14], "total": [24, 66],
                  "half_stops": config.HALF_STOP_LINES},
    }
    meta_path.write_text(json.dumps(meta, indent=1))
    return out


# ---------------------------------------------------------------------------
# QB matchup JSON — enrichment contract
# ---------------------------------------------------------------------------
def write_qb_matchup_json(path, qb_df: pd.DataFrame,
                          slate_df: pd.DataFrame) -> dict:
    base = slate_df.reset_index(drop=True)
    games = []
    for i, row in base.iterrows():
        rec = {
            "game_id": row["game_id"],
            "gameday": str(row.get("gameday", row.get("game_date", "")) or ""),
            "home_team": str(row.get("home_team", "") or ""),
            "away_team": str(row.get("away_team", "") or ""),
        }
        for f in config.QB_FIELDS:
            rec[f] = _clean(qb_df.iloc[i][f]) if f in qb_df.columns else None
        games.append(rec)
    record = {
        "created_utc": _now_utc(),
        "n_games": len(games),
        "games": games,
    }
    _dump_json(path, record)
    return record
