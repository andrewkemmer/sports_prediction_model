"""NFL moneyline RFE: record-only, deterministic, and explicitly adopted.

The default production path is unchanged. RFE evaluates the existing NFL
moneyline ensemble on the existing folds; it never changes the active feature
contract unless ``--adopt`` is invoked explicitly.
"""
from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from backend import config, folds as folds_mod, moneyline, evaluation
except ImportError:
    import config
    import folds as folds_mod
    import moneyline
    import evaluation

logger = logging.getLogger(__name__)

BACKEND = Path(__file__).resolve().parent
DELIVERY = BACKEND.parent / "data_delivery"
STATE_FILE = DELIVERY / "nfl_feature_selection_state.json"
TRACE_PREFIX = "nfl_feature_selection_"
WORKBOOK_PREFIX = "nfl_feature_workbook_"


def workbook_filename(trace: dict[str, Any], trace_path: Path | str) -> str:
    """Workbook artifact name derived from its trace.

    Lives beside the trace writer (and its prefix) so the naming contract is
    independent of the workbook writer's openpyxl dependency: a full-history
    sweep and a targeted run on the same day must not overwrite each other, so
    targeted traces carry their run stamp in the name.
    """
    path = Path(trace_path)
    raw_day = str(trace.get("date", ""))[:10]
    try:
        day = datetime.fromisoformat(raw_day).date().isoformat()
    except ValueError:
        day = raw_day if len(raw_day) == 10 and raw_day[4] == "-" else path.stem[-8:]
    targeted = bool(trace.get("targeted")) or "_targeted" in path.stem
    if not targeted:
        return f"{WORKBOOK_PREFIX}{day}.xlsx"
    try:
        stamp = datetime.fromisoformat(
            str(trace.get("created_utc", "")).replace("Z", "+00:00")).strftime("%H%M")
    except ValueError:
        stamp = datetime.now(timezone.utc).strftime("%H%M")
    return f"{WORKBOOK_PREFIX}{day}_targeted_{stamp}.xlsx"


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _list_env(name: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


def _trial_space(games: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """(incumbent, candidates, pool) — all PULLED from the one contract.

    MLB parity: the RFE owns no feature list. Removals come from the
    incumbent serving width (config.MONEYLINE_FEATURE_COLS, or the adopted
    subset), additions only from the declared candidates
    (config.RFE_CANDIDATE_COLS), and the pool is what the subset validator
    accepts (config.KNOWN_FEATURE_COLS). Each is narrowed to columns the
    frame actually generates (MLB's ``pool = [c for c in KNOWN_FEATURE_COLS
    if c in games.columns]``): a declared-but-ungenerated candidate must never
    enter a trial that would score as a no-op and read like evidence.
    """
    incumbent = [c for c in config.active_moneyline_feature_cols()
                 if c in games.columns]
    universe = list(config.MONEYLINE_FEATURE_COLS)
    candidates = [c for c in config.RFE_CANDIDATE_COLS
                  if c not in universe and c in games.columns]
    pool = [c for c in config.KNOWN_FEATURE_COLS if c in games.columns]
    return incumbent, candidates, pool


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
    y = pd.to_numeric(oof["home_win"], errors="coerce").to_numpy(float)
    p = pd.to_numeric(oof["p_ensemble"], errors="coerce").to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], np.clip(p[ok], 1e-7, 1 - 1e-7)
    if len(y) == 0:
        # Every ensemble member failed on every fold (e.g. a matrix a member
        # family refuses to fit). Failing loudly here keeps the caller from
        # comparing a (0,)-shaped loss vector against the baseline losses.
        raise RuntimeError("NFL RFE trial produced no finite OOF predictions")
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    metrics = evaluation.binary_metrics(p, y)
    metrics.update({"logloss_se": float(np.std(loss, ddof=1) / np.sqrt(len(loss))) if len(loss) > 1 else 0.0,
                    "games_scored": int(len(loss)),
                    "folds_used": int(oof["fold_id"].nunique()),
                    "per_fold": _per_fold_rows(oof)})
    return metrics, loss


def _per_fold_rows(oof: pd.DataFrame) -> list[dict[str, Any]]:
    """Fold-scoped trial detail (the decision workbook's Per-Fold sheet)."""
    from sklearn.metrics import log_loss
    rows: list[dict[str, Any]] = []
    for fold_id, grp in oof.groupby("fold_id", sort=True):
        yy = pd.to_numeric(grp["home_win"], errors="coerce").to_numpy(float)
        pp = pd.to_numeric(grp["p_ensemble"], errors="coerce").to_numpy(float)
        m = np.isfinite(yy) & np.isfinite(pp)
        days = pd.to_datetime(grp["gameday"])
        row: dict[str, Any] = {
            "fold_id": int(fold_id),
            "val_window": f"{days.min():%Y-%m-%d} → {days.max():%Y-%m-%d}",
            "n_games": int(m.sum()),
            "logloss": None,
        }
        if m.sum() and len(np.unique(yy[m])) > 1:
            row["logloss"] = round(float(log_loss(
                yy[m], np.clip(pp[m], 1e-7, 1 - 1e-7), labels=[0, 1])), 6)
        rows.append(row)
    return rows


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
    """Apply a previously adopted subset; reject unavailable features safely.

    Validity is judged against the declared pool (config.KNOWN_FEATURE_COLS),
    exactly as config.set_feature_subset does — an adopted record may promote
    candidates into serving width, so the pool (never the universe) validates
    the names. The frame check keeps a record from naming a column the
    current build does not generate.
    """
    universe_n = len(config.MONEYLINE_FEATURE_COLS)
    cols = _load_adopted()
    if not cols:
        config.reset_feature_subset()
        return {"applied": False, "n_cols": universe_n, "reason": "no adopted state"}
    unknown = [c for c in cols if c not in config.KNOWN_FEATURE_COLS]
    if unknown:
        config.reset_feature_subset()
        return {"applied": False, "n_cols": universe_n,
                "reason": f"adopted feature not in the declared pool: {unknown[:4]}"}
    if available_columns is not None and any(c not in available_columns for c in cols):
        config.reset_feature_subset()
        return {"applied": False, "n_cols": universe_n,
                "reason": "adopted feature unavailable in current frame"}
    config.set_feature_subset(cols)
    applied = config.active_moneyline_feature_cols()
    return {"applied": True, "n_cols": len(applied), "columns": applied}


def _trace_path(day: str, targeted: bool) -> Path:
    suffix = "_targeted" if targeted else ""
    return DELIVERY / f"{TRACE_PREFIX}{day}{suffix}.json"


def _feature_context(games: pd.DataFrame) -> dict[str, Any]:
    """Manifest-derived workbook context, recorded once per trace.

    The decision workbook's Master/Production/Candidates/Glossary/Redundancy/
    Coverage sheets render from this block — the workbook stays artifact-only
    (it re-reads nothing and re-derives nothing). All fields come from the
    authoritative feature manifest (manifest.FEATURE_MANIFEST) and the frame
    the sweep actually scored:

      meta      : per-feature Type/Category/Side/Description/Window/Units
      coverage  : per-feature non-null count + pct on the scored frame
      redundancy: every |r| >= 0.9 pair among pool features on the frame
    """
    try:
        from manifest import FEATURE_MANIFEST, CANDIDATE_MANIFEST
    except Exception:  # noqa: BLE001 — artifact-only; never fatal
        FEATURE_MANIFEST = {}  # type: ignore[assignment]
        CANDIDATE_MANIFEST = {}  # type: ignore[assignment]

    _, _, pool = _trial_space(games)
    meta: dict[str, dict[str, Any]] = {}
    for name in pool:
        # Candidates are documented in manifest.CANDIDATE_MANIFEST until an
        # adoption promotes them into FEATURE_MANIFEST's served list; served
        # names always win so an adopted promotion reads consistently.
        entry = FEATURE_MANIFEST.get(name) or CANDIDATE_MANIFEST.get(name) or {}
        lookback = entry.get("lookback", "")
        if isinstance(lookback, int):
            window = f"{lookback} games" if lookback else "static pre-game"
        else:
            window = str(lookback) or "static pre-game"
        # Category/side derived from the representation + name convention —
        # the same diff-vs-level split the manifest's ``representation`` field
        # documents.
        rep = str(entry.get("representation", ""))
        if name.endswith("_home") or name.endswith("_away"):
            side = "Home" if name.endswith("_home") else "Away"
            ftype = "Raw level (per side)"
            category = "Matchup context"
        elif "difference" in rep or name.endswith("_diff"):
            side = "Home − Away (diff)"
            ftype = "Diff"
            category = "Team form"
        else:
            side = "Game-level"
            ftype = "Game-level flag/context"
            category = "Schedule / venue"
        meta[name] = {
            "description": str(entry.get("description", "")),
            "definition": str(entry.get("definition", "")),
            "source": str(entry.get("source", "")),
            "point_in_time_rule": str(entry.get("point_in_time_rule", "")),
            "missing_value_policy": str(entry.get("missing_value_policy", "")),
            "window": window,
            "type": ftype,
            "category": category,
            "side": side,
        }

    # Coverage: share of non-null values per pool feature on the scored frame.
    coverage: dict[str, dict[str, float]] = {}
    n_rows = len(games)
    for name in pool:
        if name in games.columns:
            col = pd.to_numeric(games[name], errors="coerce")
            n_ok = int(col.notna().sum())
        else:
            n_ok = 0
        coverage[name] = {"n": n_ok,
                          "pct": round(100.0 * n_ok / n_rows, 2) if n_rows else 0.0}

    # Redundancy: |r| >= 0.9 pairs among pool features (MLB v2 parity).
    redundancy: list[dict[str, Any]] = []
    cols_present = [c for c in pool if c in games.columns]
    if len(cols_present) >= 2 and n_rows > 2:
        try:
            corr = games[cols_present].apply(
                pd.to_numeric, errors="coerce").corr()
            for i, a in enumerate(cols_present):
                for b in cols_present[i + 1:]:
                    r = corr.loc[a, b]
                    if pd.notna(r) and abs(float(r)) >= 0.9:
                        redundancy.append({"a": a, "b": b,
                                           "r": round(float(r), 4)})
        except Exception as exc:  # noqa: BLE001 — artifact-only
            logger.warning("RFE redundancy scan skipped: %s", exc)

    return {"meta": meta, "coverage": coverage, "redundancy": redundancy,
            "n_rows": int(n_rows)}


def run_rfe(games: pd.DataFrame, day: str, max_steps: int = 40) -> dict[str, Any]:
    base, candidates, pool = _trial_space(games)
    adds, unresolved_add = _resolve(_list_env("NFL_RFE_ADDITION_MONEYLINE_LIST"), candidates)
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
                 [{"kind": "add", "feature": f} for f in candidates])
        if not candidates:
            logger.info("NFL RFE: no declared candidates (config.RFE_CANDIDATE_COLS is "
                        "empty) — the sweep is removals-only over the %d-feature contract",
                        len(base))
    if _truthy("NFL_RFE_RETRIAL"):
        # Retrial pass ("allow all features back in regardless of verdict"):
        # re-offer every pool feature once the sweep has finished. Features
        # still active are skipped by the loop's own guard, so this only costs
        # trials for features the sweep actually committed out.
        order = order + [{"kind": "add", "feature": f} for f in pool]
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
            metrics, losses = _score(games, folds)
            # Paired per-game comparison against the baseline subset: the
            # paired SE is what makes the commit bar fold-noise aware.
            diff = losses - base_loss
            se = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
            threshold = max(0.0005,
                            getattr(config, "RFE_COMMIT_SE_MULTIPLE", 2.0) * se)
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
                          "commit_threshold": threshold, "committed": committed})
            logger.info("NFL RFE trial %d (%s %s, n=%d): committed=%s gain=%+.5f "
                        "threshold=%.5f", len(steps), kind, item["feature"],
                        len(trial), committed, gain, threshold)
        except Exception as exc:  # noqa: BLE001
            # A trial that cannot be scored is RECORDED, never fatal: losing
            # the whole sweep — and with it the trace and the decision
            # workbook — because one trial could not be evaluated is the
            # failure this guards against.
            steps.append({"step": len(steps) + 1, "kind": kind, "feature": item["feature"],
                          "adds": item.get("adds", []), "removes": item.get("removes", []),
                          "n_features": len(trial), "metrics": {},
                          "logloss_gain": None, "paired_se": None,
                          "commit_threshold": None, "committed": False,
                          "error": str(exc)})
            logger.warning("NFL RFE trial %d (%s %s) failed; recorded, not fatal: %s",
                           len(steps), kind, item["feature"], exc)
        finally:
            config.set_feature_subset(active)
        budget -= 1
    config.reset_feature_subset()
    record = {"schema": "nfl-rfe-v2", "date": day, "created_utc": datetime.utcnow().isoformat() + "Z",
              "run_mode": "targeted_full_history" if targeted else "full_history",
              "targeted": targeted, "n_pool": len(pool), "n_universe": len(base),
              "candidate_pool": candidates, "n_candidates": len(candidates),
              "n_trials": len(steps), "n_committed": sum(s["committed"] for s in steps),
              "n_failed": sum(bool(s.get("error")) for s in steps),
              "n_selected": len(active), "selected_cols": active,
              "baseline_metrics": baseline, "best_metrics": best, "steps": steps,
              "forced_lists": {"additions": adds, "removals": removes,
                               "unresolved": unresolved_add + unresolved_remove},
              "grid_max_states": max_states,
              "commit_se_multiple": getattr(config, "RFE_COMMIT_SE_MULTIPLE", 2.0),
              "feature_context": _feature_context(games)}
    DELIVERY.mkdir(parents=True, exist_ok=True)
    path = _trace_path(day.replace("-", ""), targeted)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return {"ran": True, "trace": str(path), **record}


def maybe_run_rfe(games: pd.DataFrame, day: str) -> dict[str, Any]:
    if not _truthy("NFL_RFE_FORCE"):
        return {"ran": False, "reason": "NFL_RFE_FORCE not set"}
    return run_rfe(games, str(day)[:10])


def adopt(trace_path: str | None = None) -> dict[str, Any]:
    # Dated traces only: the glob prefix also matches the STATE file's name
    # ("nfl_feature_selection_state.json"), which sorts last alphabetically —
    # without the date-shape filter, default adopt() would read JSON state as
    # a trace and crash on missing keys.
    def _is_trace(p: Path) -> bool:
        stem = p.stem[len(TRACE_PREFIX):]
        return bool(stem) and (stem[-8:].isdigit() or "_targeted" in stem)
    if trace_path:
        path = Path(trace_path)
    else:
        candidates = sorted(p for p in DELIVERY.glob(f"{TRACE_PREFIX}*.json")
                            if _is_trace(p))
        if not candidates:
            raise FileNotFoundError("no RFE trace in data_delivery to adopt")
        path = candidates[-1]
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
