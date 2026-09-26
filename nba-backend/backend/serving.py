"""NBA frontend-facing artifact writers and JSON contracts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from backend import config
    from backend import nba_sources as sources
except ImportError:
    import config
    import nba_sources as sources


def _clean(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_safe(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return _clean(value)


def dump_json(path: Path, record: dict) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_record = _safe(record)
    path.write_text(json.dumps(safe_record, indent=1, allow_nan=False, default=str))
    # Return the same strict, JSON-safe shape that was persisted.  Returning
    # the caller's mutable NaN-bearing object would let downstream reporting
    # accidentally reintroduce non-standard JSON values.
    return safe_record


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date(value: Any) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def _prob_array(value: Any, length: int) -> np.ndarray:
    if value is None:
        return np.full(length, np.nan)
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.repeat(array, length)
    if len(array) != length:
        raise ValueError("probability array length does not match slate")
    return array


def _start_time_utc(g) -> str | None:
    """Return the official UTC tipoff, or null when it is unavailable.

    Mirrors the NHL backend's rule exactly, and for the same reason: ``gameday``
    is an Eastern board date, so fabricating midnight UTC for a missing tipoff
    places the instant on the prior Eastern evening and makes a real board date
    look valid. A card with no tipoff is honest; a card with the wrong one is
    not, and the difference is invisible to the reader.
    """
    raw = str(g.get("start_time_utc", "") or "").strip()
    if not raw or ("T" not in raw and " " not in raw):
        return None
    try:
        stamp = pd.Timestamp(raw)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        else:
            stamp = stamp.tz_convert("UTC")
        return stamp.isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError, OverflowError):
        return None


def write_moneyline_json(path, slate: pd.DataFrame, p_home, p_home_cal,
                         team_names=None, config_meta=None,
                         player_leaders=None) -> dict:
    """Write the v1 moneyline/card JSON consumed by ``frontend/utils.py``."""
    team_names = team_names or {}
    frame = slate.reset_index(drop=True)
    raw = _prob_array(p_home, len(frame))
    cal = _prob_array(p_home_cal, len(frame))
    leaders = {}
    if player_leaders is not None and len(player_leaders):
        for _, row in player_leaders.iterrows():
            leaders[str(row.get("game_id"))] = row.to_dict()
    games = []
    for i, (_, game) in enumerate(frame.iterrows()):
        raw_p = _clean(raw[i])
        cal_p = _clean(cal[i])
        shown = cal_p if cal_p is not None else raw_p
        pick = None if shown is None else (game.get("home_team") if shown >= 0.5 else game.get("away_team"))
        hs, as_ = game.get("home_score"), game.get("away_score")
        final = pd.notna(hs) and pd.notna(as_)
        detail = str(game.get("game_status_detail", "") or "")
        postponed = sources.is_postponed_detail(detail)
        row = {
            "game_id": str(game.get("game_id", "")),
            "game_date": _date(game.get("gameday", game.get("game_date"))),
            "start_time_utc": _start_time_utc(game),
            "home_team": game.get("home_team"), "away_team": game.get("away_team"),
            "home_team_name": game.get("home_team_name") or team_names.get(game.get("home_team"), game.get("home_team")),
            "away_team_name": game.get("away_team_name") or team_names.get(game.get("away_team"), game.get("away_team")),
            "home_record": _clean(game.get("home_record")), "away_record": _clean(game.get("away_record")),
            "venue": _clean(game.get("venue")),
            # "Postponed" is a distinct state from "Scheduled", MLB's structure:
            # the game has no tip to predict against and no result to show, and
            # labelling it either way misleads.  When ESPN reschedules it the
            # event keeps its id, moves to the new date and settles there, so it
            # reappears on the card for the day it is actually played.
            "game_status": ("Postponed" if postponed else
                            "Final" if final else
                            (game.get("game_status") or "Scheduled")),
            "game_status_detail": detail,
            "home_score": _clean(hs), "away_score": _clean(as_),
            "home_win_prob_model": shown,
            "away_win_prob_model": None if shown is None else 1 - shown,
            "home_win_prob_model_raw": raw_p,
            "model_pick": pick,
            "model_correct": None,
        }
        player = leaders.get(str(game.get("game_id")), {})
        for field in config.PLAYER_FIELDS:
            row[field] = _clean(player.get(field))
        games.append(_safe(row))
    slate_date_value = frame.get("gameday", frame.get("game_date"))
    slate_date = _date(pd.to_datetime(slate_date_value, errors="coerce").min()) if len(frame) else None
    record = {"created_utc": _now(), "config": config_meta or {},
              "slate_date": slate_date, "n_games": len(games), "games": games}
    return dump_json(Path(path), record)


def write_calibration_json(path, raw_metrics, calibrated_metrics, buckets,
                           daily, config_meta=None, platt=None, run_date="",
                           n_games=0, distribution_calibration=None) -> dict:
    calibration: dict[str, Any] = {}
    if platt and platt.get("a") is not None:
        calibration = {
            "method": platt.get("method", "favored_platt_floor"),
            "params": {"a": platt.get("a"), "b": platt.get("b"), "n": platt.get("n", n_games)},
            "metrics_raw": {k: raw_metrics.get(k) for k in ("brier", "logloss", "ece")},
            "metrics_calibrated": {k: calibrated_metrics.get(k) for k in ("brier", "logloss", "ece")},
        }
    record = {
        "date": run_date, "trained_at": _now(), "created_utc": _now(),
        "config": config_meta or {}, "n_games": int(n_games),
        "metrics": {**{k: raw_metrics.get(k) for k in ("auc", "brier", "logloss", "ece")},
                    **{f"{k}_calibrated": calibrated_metrics.get(k) for k in ("auc", "brier", "logloss", "ece")}},
        "calibration": calibration,
        "distribution_calibration": distribution_calibration or {},
        "calibration_buckets": buckets or [], "daily": daily or [],
    }
    return dump_json(Path(path), record)


def write_predictions_history_csv(path, oof: pd.DataFrame, p_cal=None) -> pd.DataFrame:
    if oof is None or not len(oof):
        pd.DataFrame().to_csv(path, index=False)
        return pd.DataFrame()
    frame = oof.copy()
    if p_cal is not None:
        calibrated = _prob_array(p_cal, len(frame))
    else:
        calibrated = pd.to_numeric(frame.get("p_ensemble"), errors="coerce").to_numpy()
    raw_values = frame.get("p_ensemble")
    raw = (pd.to_numeric(raw_values, errors="coerce").to_numpy()
           if raw_values is not None else np.full(len(frame), np.nan))
    dates = pd.to_datetime(frame.get("gameday"), errors="coerce").dt.strftime("%Y-%m-%d")
    home = frame.get("home_team", pd.Series("", index=frame.index)).fillna("")
    away = frame.get("away_team", pd.Series("", index=frame.index)).fillna("")
    actual = frame.get("home_win", pd.Series(np.nan, index=frame.index))
    pick = np.where(raw >= 0.5, home, away)
    winner = np.where(actual > 0.5, home, np.where(actual < 0.5, away, ""))
    out = pd.DataFrame({
        "game_id": frame.game_id.astype(str), "game_date": dates,
        "home_team": home, "away_team": away,
        "home_score": frame.get("home_score"), "away_score": frame.get("away_score"),
        "home_win": actual,
        "home_win_prob_model": raw, "home_win_prob_model_calibrated": calibrated,
        "model_pick": pick, "actual_winner": winner,
        "correct": (pick == winner) & actual.notna(),
    })
    out.to_csv(path, index=False)
    return out


def write_power_rankings_csv(path, ratings, records, team_names=None,
                             point_diff=None) -> pd.DataFrame:
    names = team_names or {}
    rows = []
    for team, elo in sorted((ratings or {}).items(), key=lambda item: -float(item[1])):
        w, l = (records or {}).get(team, (0, 0))
        diff = int((point_diff or {}).get(team, 0) or 0)
        rows.append({"rank": 0, "team": team, "team_name": names.get(team, team),
                     "elo": round(float(elo), 1), "wins": int(w), "losses": int(l),
                     "record": f"{int(w)}-{int(l)}",
                     "pct": round(w / (w + l), 3) if w + l else np.nan,
                     "run_diff": diff, "point_diff": diff, "l10": "",
                     "home_pct": np.nan, "away_pct": np.nan})
    out = pd.DataFrame(rows)
    if len(out):
        out["rank"] = range(1, len(out) + 1)
    out.to_csv(path, index=False)
    return out


def _label(value: float | int) -> str:
    """Serialize a spread key using the shared NBA grid convention.

    Negative integer lines use ``mN`` (not ``-N``), and decimal points become
    underscores.  Keeping this helper identical to the distribution module is
    what prevents a valid priced grid from being reindexed to NaN columns when
    the serving writer projects the persisted schema.
    """
    number = float(value)
    text = str(number)
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace("-", "m").replace(".", "_")


def _grid_columns() -> list[str]:
    spread = config.SPREAD_GRID
    half = config.HALF_STOP_LINES
    totals = config.TOTAL_GRID
    return ([f"p_home_cover_{_label(x)}" for x in spread]
            + [f"p_away_cover_{_label(x)}" for x in spread]
            + [f"p_push_{_label(x)}" for x in spread]
            + [f"p_home_cover_{_label(x)}" for x in half]
            + [f"p_away_cover_{_label(x)}" for x in half]
            + [f"p_push_{_label(x)}" for x in half]
            + [f"p_over_{x}" for x in totals]
            + [f"p_push_total_{x}" for x in totals]
            + [f"p_under_{x}" for x in totals])


def markets_columns() -> list[str]:
    base = [
        "kind", "game_id", "gameday", "season", "home_team", "away_team",
        "venue", "start_time_utc", "home_record", "away_record",
        "p_home_win", "p_away_win", "p_tie", "derived_ml",
        "p_home_win_derived", "p_away_win_derived", "mu_h", "mu_a",
        "mu_margin", "mu_total", "fair_spread", "fair_total", "spread_line",
        "total_line", "has_offer", "p_cover_offered", "p_push_offered",
        "p_over_offered", "p_under_offered", "p_push_total_offered",
        "home_score", "away_score", "total", "margin", "p_over_fair",
        "p_cover_fair", "y_over_fair", "y_under_fair", "y_push_fair",
        "y_cover_fair", "y_push_spread_fair", "y_home_win", "decided",
        "frame_view", "pred_home", "pred_away",
    ]
    return list(dict.fromkeys(base + _grid_columns()))


def write_markets_csv(path, meta_path, oof_rows, slate_rows, config_meta=None) -> pd.DataFrame:
    pieces = [frame for frame in (oof_rows, slate_rows)
              if frame is not None and len(frame)]
    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    # A merge can leave duplicate labels; the CSV contract is name-unique.
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()].copy()
    # ``reindex`` is what fills the absent grid columns with NaN, and it does
    # it in one block.  Assigning them one at a time first built a frame with
    # ~470 single-column inserts, which pandas reports as "highly fragmented"
    # and charges for on every later column access.
    out = out.reindex(columns=markets_columns())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    meta = {
        "record": "nba_run_engine_markets", "written_utc": _now(),
        "config": config_meta or {}, "columns": len(out.columns), "n_rows": len(out),
        "n_oof": int((out.kind == "oof").sum()) if len(out) else 0,
        "n_slate": int((out.kind == "slate").sum()) if len(out) else 0,
        "grids": {"spread": [min(config.SPREAD_GRID), max(config.SPREAD_GRID)],
                  "total": [min(config.TOTAL_GRID), max(config.TOTAL_GRID)],
                  "half_stops": list(config.HALF_STOP_LINES),
                  "totals_push_namespace": "p_push_total_<U>"},
    }
    dump_json(Path(meta_path), meta)
    return out


def write_player_matchup_json(path, leaders: pd.DataFrame) -> dict:
    games = []
    if leaders is not None:
        for _, row in leaders.iterrows():
            record = {"game_id": row.get("game_id"),
                      "gameday": _date(row.get("gameday", row.get("game_date"))),
                      "home_team": row.get("home_team"), "away_team": row.get("away_team")}
            record.update({field: row.get(field) for field in config.PLAYER_FIELDS})
            games.append(_safe(record))
    return dump_json(Path(path), {"created_utc": _now(), "n_games": len(games), "games": games})


def write_feature_json(path, coverage, config_meta=None, fold_info=None) -> dict:
    try:
        from backend.manifest import FEATURE_MANIFEST
    except ImportError:
        from manifest import FEATURE_MANIFEST
    record = {
        "created_utc": _now(), "feature_set_version": config.FEATURE_SET_VERSION,
        "manifest": FEATURE_MANIFEST,
        "served_columns": config.active_moneyline_feature_cols(),
        "coverage": coverage.to_dict(orient="records") if hasattr(coverage, "to_dict") else coverage,
        "fold_geometry": fold_info or {}, "config": config_meta or {},
    }
    return dump_json(Path(path), record)


def write_production_cards_history(path, cards: pd.DataFrame) -> pd.DataFrame:
    """Persist frozen, first-publication card probabilities for archive view."""
    path = Path(path)
    incoming = cards.copy()
    if incoming.empty:
        return incoming
    if "game_date" not in incoming and "gameday" in incoming:
        incoming["game_date"] = pd.to_datetime(incoming.gameday, errors="coerce").dt.strftime("%Y-%m-%d")
    if "p_home_win" not in incoming:
        incoming["p_home_win"] = incoming.get("p_ensemble_calibrated", incoming.get("home_win_prob_model"))
    if "p_away_win" not in incoming:
        incoming["p_away_win"] = 1 - pd.to_numeric(incoming.p_home_win, errors="coerce")
    incoming["p_home_win"] = pd.to_numeric(incoming.p_home_win, errors="coerce")
    incoming["p_away_win"] = pd.to_numeric(incoming.p_away_win, errors="coerce")
    if "model_pick" not in incoming:
        incoming["model_pick"] = np.where(incoming.p_home_win >= 0.5,
                                           incoming.get("home_team", ""), incoming.get("away_team", ""))
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    out = incoming if existing.empty else pd.concat([existing, incoming], ignore_index=True, sort=False)
    if "game_id" in out:
        # First publication wins.  Re-running the pipeline for a slate that
        # was already archived must never rewrite its frozen production pick.
        out = out.drop_duplicates("game_id", keep="first")
    out.to_csv(path, index=False)
    return out
