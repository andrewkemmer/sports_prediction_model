"""NFL moneyline RFE: record-only, deterministic, and explicitly adopted.

The default production path is unchanged. RFE evaluates the existing NFL
moneyline ensemble on the existing folds; it never changes the active feature
contract unless ``--adopt`` is invoked explicitly.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from backend import config, folds as folds_mod, moneyline, evaluation, features as feat_mod
except ImportError:
    import config, folds as folds_mod, moneyline, evaluation, features as feat_mod

BACKEND = Path(__file__).resolve().parent
DELIVERY = BACKEND.parent / "data_delivery"
STATE_FILE = DELIVERY / "nfl_feature_selection_state.json"
TRACE_PREFIX = "nfl_feature_selection_"


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _list_env(name: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


def _candidate_pool(games: pd.DataFrame) -> list[str]:
    """All valid generated features not already in the production contract."""
    generated = feat_mod.feature_engine_columns(games)
    base = set(config.active_feature_columns()) | set(config.ANCHOR_COLUMNS)
    return [c for c in generated if c not in base]


def _resolve(names: list[str], pool: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    resolved, unresolved = [], []
    for name in names:
        if name in pool:
            resolved.append(name)
        else:
            unresolved.append({"name": name, "nearest": difflib.get_close_matches(name, pool, n=3, cutoff=.4)})
    return resolved, unresolved


def _score(games: pd.DataFrame, folds: list) -> tuple[dict[str, Any], np.ndarray]:
    result = moneyline.walk_forward_oof(games, fold_list=folds)
    oof = result["oof"]
    if oof.empty:
        raise RuntimeError("NFL RFE produced no OOF rows")
    y = oof["home_win"].to_numpy(float)
    p = oof["p_ensemble"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], np.clip(p[ok], 1e-7, 1 - 1e-7)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    metrics = evaluation.binary_metrics(p, y)
    metrics.update({"logloss_se": float(np.std(loss, ddof=1) / np.sqrt(len(loss))) if len(loss) > 1 else 0.0,
                    "games_scored": int(len(loss)),
                    "folds_used": int(oof["fold_id"].nunique())})
    return metrics, loss


def _load_adopted() -> list[str] | None:
    if not STATE_FILE.exists():
        return None
    try:
        rec = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        cols = rec.get("selected_cols")
        return list(cols) if isinstance(cols, list) and cols else None
    except (OSError, json.JSONDecodeError):
        return None


def apply_adopted_subset(available_columns: list[str] | None = None) -> dict[str, Any]:
    """Apply a previously adopted subset; reject unavailable features safely."""
    cols = _load_adopted()
    if not cols:
        config.reset_feature_subset()
        return {"applied": False, "n_cols": len(config.FEATURE_COLUMNS), "reason": "no adopted state"}
    if len(cols) < 1 or (available_columns is not None and
                          any(c not in available_columns for c in cols)):
        config.reset_feature_subset()
        return {"applied": False, "n_cols": len(config.FEATURE_COLUMNS),
                "reason": "adopted feature unavailable in current frame"}
    config.set_feature_subset(cols)
    return {"applied": True, "n_cols": len(cols), "columns": cols}


def _trace_path(day: str, targeted: bool) -> Path:
    suffix = "_targeted" if targeted else ""
    return DELIVERY / f"{TRACE_PREFIX}{day}{suffix}.json"


def run_rfe(games: pd.DataFrame, day: str, max_steps: int = 40) -> dict[str, Any]:
    base = list(config.active_feature_columns())
    pool = list(dict.fromkeys(base + _candidate_pool(games)))
    adds, unresolved_add = _resolve(_list_env("NFL_RFE_ADDITION_MONEYLINE_LIST"), pool)
    removes, unresolved_remove = _resolve(_list_env("NFL_RFE_REMOVAL_MONEYLINE_LIST"), base)
    targeted = bool(adds or removes)
    if (unresolved_add or unresolved_remove) and _truthy("NFL_RFE_ADDITION_REMOVAL_GRID_MODE"):
        raise ValueError(f"unresolvable grid feature names: {unresolved_add + unresolved_remove}")
    folds = folds_mod.make_folds(games, date_col="gameday")
    if not folds:
        raise RuntimeError("NFL RFE has no production folds")
    config.set_feature_subset(base)
    baseline, base_loss = _score(games, folds)
    order = ([{"kind": "remove", "feature": f} for f in removes] +
             [{"kind": "add", "feature": f} for f in adds])
    if not targeted:
        order = ([{"kind": "remove", "feature": f} for f in base] +
                 [{"kind": "add", "feature": f} for f in _candidate_pool(games)])
    if _truthy("NFL_RFE_RETRIAL") and targeted:
        order = order * 1
    max_states_raw = os.environ.get("RFE_GRID_MAX_STATES", "16")
    try:
        max_states = max(2, int(max_states_raw))
    except ValueError:
        max_states = 16
    if _truthy("NFL_RFE_ADDITION_REMOVAL_GRID_MODE") and targeted:
        # The full lattice is intentionally capped; overflow is still tested
        # one-by-one below, never silently discarded.
        from itertools import combinations
        states = []
        for na in range(len(adds) + 1):
            for nr in range(len(removes) + 1):
                for a in combinations(adds, na):
                    for r in combinations(removes, nr):
                        states.append((list(a), list(r)))
        states = states[:max_states]
        order = []
        for a, r in states:
            order.append({"kind": "grid", "adds": a, "removes": r,
                          "feature": "+".join(a) + "/" + ",".join(r)})
    active = list(base)
    best = baseline
    steps = []
    budget = max_steps
    for item in order:
        if budget <= 0:
            break
        kind = item["kind"]
        if kind == "grid":
            trial = [c for c in base if c not in item["removes"]] + list(item["adds"])
        elif kind == "remove":
            if item["feature"] not in active:
                continue
            trial = [c for c in active if c != item["feature"]]
        else:
            if item["feature"] in active:
                continue
            trial = active + [item["feature"]]
        if not trial:
            continue
        config.set_feature_subset(trial)
        try:
            try:
                metrics, losses = _score(games, folds)
            except Exception as exc:  # noqa: BLE001
                # A failed trial must be visible in the trace, not abort the
                # entire record-only analysis and suppress its workbook.
                steps.append({"step": len(steps) + 1, "kind": kind,
                              "feature": item["feature"],
                              "adds": item.get("adds", []),
                              "removes": item.get("removes", []),
                              "n_features": len(trial), "status": "failed",
                              "error": f"{type(exc).__name__}: {exc}",
                              "committed": False})
                budget -= 1
                continue
            diff = losses - base_loss
            se = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
            threshold = max(0.0005, 2.0 * se)
            gain = float(best["logloss"] - metrics["logloss"])
            committed = bool(gain >= threshold and metrics["auc"] >= best["auc"] - .003
                             and metrics["ece"] <= best["ece"] + .005)
            if committed and kind != "grid":
                active = trial
                best = metrics
            steps.append({"step": len(steps) + 1, "kind": kind, "feature": item["feature"],
                          "adds": item.get("adds", []), "removes": item.get("removes", []),
                          "n_features": len(trial), "metrics": metrics,
                          "logloss_gain": gain, "paired_se": se,
                          "commit_threshold": threshold, "committed": committed,
                          "status": "scored"})
        finally:
            config.set_feature_subset(active)
        budget -= 1
    config.reset_feature_subset()
    record = {"schema": "nfl-rfe-v1", "date": day, "created_utc": datetime.utcnow().isoformat() + "Z",
              "run_mode": "targeted_full_history" if targeted else "full_history",
              "targeted": targeted, "n_pool": len(pool), "n_universe": len(base),
              "n_trials": len(steps), "n_failed": sum(s.get("status") == "failed" for s in steps),
              "n_committed": sum(s["committed"] for s in steps),
              "n_selected": len(active), "selected_cols": active,
              "baseline_metrics": baseline, "best_metrics": best, "steps": steps,
              "forced_lists": {"additions": adds, "removals": removes,
                               "unresolved": unresolved_add + unresolved_remove},
              "grid_max_states": max_states}
    DELIVERY.mkdir(parents=True, exist_ok=True)
    path = _trace_path(day.replace("-", ""), targeted)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return {"ran": True, "trace": str(path), **record}


def _score_with_active(*_args):
    raise RuntimeError("unreachable helper")


def maybe_run_rfe(games: pd.DataFrame, day: str) -> dict[str, Any]:
    if not _truthy("NFL_RFE_FORCE"):
        return {"ran": False, "reason": "NFL_RFE_FORCE not set"}
    return run_rfe(games, str(day)[:10])


def adopt(trace_path: str | None = None) -> dict[str, Any]:
    path = Path(trace_path) if trace_path else sorted(DELIVERY.glob(f"{TRACE_PREFIX}*.json"))[-1]
    rec = json.loads(path.read_text(encoding="utf-8"))
    cols = rec.get("selected_cols") or []
    if len(cols) < 1:
        raise ValueError("RFE trace has no selected feature list")
    STATE_FILE.write_text(json.dumps({"schema": "nfl-rfe-state-v1", "source_trace": path.name,
                                      "selected_cols": cols}, indent=2), encoding="utf-8")
    return {"adopted": True, "n_cols": len(cols), "state": str(STATE_FILE)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace")
    ap.add_argument("--adopt", action="store_true")
    args = ap.parse_args()
    print(json.dumps(adopt(args.trace) if args.adopt else {"error": "use pipeline for RFE"}, indent=2))
