"""Automated feature selection: blend-level recursive feature elimination.

Primary objective: minimize pooled walk-forward logloss of the moneyline
ENSEMBLE (all five members, each with its own feature view, blended with the
run's adaptive weights). Secondary objectives are guards, not targets: a
pruning step is only committed when pooled AUC does not significantly decline
(<= RFE_AUC_GUARD) and pooled ECE does not significantly rise (<= RFE_ECE_GUARD).

Design decisions (locked):

* BLEND-LEVEL, not per-member. A subset composes with every member's existing
  routing (logistic diff-only slice, tree categoricals, RF/MLP scaling) inside
  training._feature_matrix / ensemble_predict, so the measured metric IS the
  blend's logloss by construction. Per-member RFE would mean five coupled
  searches against one shared objective.
* RANK-ORDERED removal (classic RBE/RFE): features are ranked ONCE per run by
  ensemble feature importance (feature_importance_weights, deterministic under
  fixed seed) and eliminated weakest-first. Re-ranking after every step would
  multiply walk-forward evaluations by the number of remaining features —
  RFE_MAX_STEPS bounds total scored evaluations instead.
* SAME fold geometry as production: walk_forward_splits with RETRAIN_CADENCE_DAYS,
  max_eval_folds=MAX_EVAL_FOLDS. The RFE view can never see folds the daily run
  doesn't, and its pooled logloss is directly comparable to the shipped metric.
* RECORD-ONLY by default: results land in
  data_delivery/mlb_feature_selection_<date>.json (never deleted; feeds
  dashboards and audits). Nothing changes at serving time unless the record is
  explicitly adopted (--adopt), which writes
  data_delivery/mlb_feature_selection_state.json. The daily retrain applies the
  adopted subset with a safe fallback to the full universe.
* Deterministic: RANDOM_SEED throughout; identical data + code → identical
  verdict, step trace, and adoption decision.

Scratch-space runs: `python feature_selection.py --date YYYY-MM-DD`
Adoption:          `python feature_selection.py --date YYYY-MM-DD --adopt`
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config import (
    DEFAULT_MAX_EVAL_FOLDS,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
    RFE_AUC_GUARD,
    RFE_ECE_GUARD,
    RFE_FLOOR,
    RFE_MIN_LOGLOSS_GAIN,
    RFE_MAX_STEPS,
    RFE_NOISE_SIGMA,
)
from explainability import DATA_DELIVERY_DIR
from training import (
    MARGIN_COL,
    MONEYLINE_FEATURE_COLS,
    active_moneyline_feature_cols,
    compute_metrics,
    ensemble_predict,
    feature_importance_weights,
    reset_feature_subset,
    set_feature_subset,
    train_moneyline_ensemble,
    walk_forward_splits,
)

logger = logging.getLogger(__name__)

STATE_FILE = "mlb_feature_selection_state.json"
TRACE_PREFIX = "mlb_feature_selection_"


# ── Fold scoring on the real ensemble ───────────────────────────────────────


def _score_splits(splits: list[dict[str, Any]]) -> dict[str, float]:
    """Train + score the full ensemble on every fold; return pooled metrics.

    Mirrors the daily walk-forward loop's per-fold contract: train each fold's
    members with the val window for early stopping, predict the val window
    with the blended ensemble, pool compute_metrics across all validation
    rows. Evaluated with the CURRENT active subset — callers set it before
    calling. Deterministic under the module seed.
    """
    import numpy as np

    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    n_used = 0
    for split in splits:
        train = split["train_games"]
        val = split["val_games"]
        if train.empty or val.empty:
            continue
        try:
            models, _fold_metrics = train_moneyline_ensemble(train, val)
        except Exception as exc:
            logger.warning("Fold training failed (skipped): %s", exc)
            continue
        if not models:
            continue
        blend, _members, _wts = ensemble_predict(models, val)
        y = val["home_win"].astype(int).to_numpy()
        if len(y) != len(blend):
            continue
        y_all.append(y)
        p_all.append(np.asarray(blend, dtype=float))
        n_used += 1

    if not n_used or not y_all:
        raise RuntimeError("RFE scoring produced no usable folds")

    y_true = np.concatenate(y_all)
    y_prob = np.concatenate(p_all)
    metrics = compute_metrics(y_true, y_prob)
    # Standard error of the pooled logloss estimate: a commit threshold of
    # ~2·SE is what separates real signal from val-window chance (fixed
    # floors cannot — see RFE_NOISE_SIGMA).
    eps = 1e-7
    p = np.clip(y_prob, eps, 1 - eps)
    losses = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    metrics["logloss_se"] = round(float(np.std(losses, ddof=1) / np.sqrt(len(losses))), 6)
    metrics["folds_used"] = n_used
    return metrics


def _splits_and_frame(games: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Attach OOF run margins when the margin feature is active, then split.

    Same geometry contract as the daily walk-forward: when run_margin_diff is
    in the active subset, folds are regenerated over the margin-enriched frame
    (build_oof_margin._attach_oof_run_margins asserts identical geometry).
    max_eval_folds follows the production default (0 = full history) so the
    RFE view can never see fewer folds than the shipped metric.
    """
    splits = walk_forward_splits(
        games, retrain_cadence_days=RETRAIN_CADENCE_DAYS,
        max_eval_folds=DEFAULT_MAX_EVAL_FOLDS)
    if MARGIN_COL not in active_moneyline_feature_cols():
        return games, splits
    from training import _attach_oof_run_margins

    enriched, splits = _attach_oof_run_margins(
        games, splits,
        min_val_games=0, max_eval_folds=DEFAULT_MAX_EVAL_FOLDS,
        retrain_cadence_days=RETRAIN_CADENCE_DAYS, min_train_days=0)
    return enriched, splits


# ── Rank ordering ───────────────────────────────────────────────────────────


def _rank_features(models: dict[str, Any]) -> list[str]:
    """Rank active features weakest-first by ensemble importance.

    feature_importance_weights returns {feature: percent-share}; missing names
    (a member failed to train) rank last — they are the weakest signal by
    definition. Ties break alphabetically for determinism.
    """
    active = active_moneyline_feature_cols()
    weights = feature_importance_weights(models) or {}
    return sorted(
        active,
        key=lambda f: (weights.get(f, 0.0), f),
    )


# ── The RFE walk ────────────────────────────────────────────────────────────


def run_rfe(
    games: pd.DataFrame,
    floor: int = RFE_FLOOR,
    max_steps: int = RFE_MAX_STEPS,
    auc_guard: float = RFE_AUC_GUARD,
    ece_guard: float = RFE_ECE_GUARD,
    min_logloss_gain: float = RFE_MIN_LOGLOSS_GAIN,
    noise_sigma: float = RFE_NOISE_SIGMA,
) -> dict[str, Any]:
    """Run blend-level RFE. Returns the full result record (not yet adopted).

    Steps:
      1. Score the ensemble at full universe width (the incumbent baseline).
      2. Rank features weakest-first via ensemble importance.
      3. Repeatedly remove the weakest remaining feature (down to ``floor``)
         and re-score; commit the removal only if pooled logloss improves by
         at least ``min_logloss_gain`` AND the AUC/ECE guards hold. Otherwise
         restore the feature and strike it from the candidate removal order
         (it carries signal). The strict-improvement threshold is a noise
         floor: with zero tolerance, chance logloss jitter on small val
         windows commits noise-fitting removals (caught in scratch
         verification on synthetic pure-noise targets).
      4. Stop when the floor is reached, the step budget is exhausted, or
         every remaining candidate has been rejected.

    Deterministic end to end under RANDOM_SEED.
    """
    reset_feature_subset()
    universe = list(MONEYLINE_FEATURE_COLS)

    enriched, splits = _splits_and_frame(games)
    if not splits:
        raise RuntimeError("No walk-forward folds for the supplied frame")

    baseline = _score_splits(splits)
    baseline_ll = float(baseline["logloss"])
    # A removal commits only on a gain the noise cannot explain: at least
    # min_logloss_gain AND at least noise_sigma standard errors of the
    # baseline pooled logloss estimate.
    threshold = max(float(min_logloss_gain),
                    noise_sigma * float(baseline.get("logloss_se", 0.0)))

    # Rank once from a full-width ensemble (importances are width-consistent
    # with the universe).
    seed_models, _ = train_moneyline_ensemble(
        splits[0]["train_games"], splits[0]["val_games"])
    removal_order = _rank_features(seed_models)

    active = list(universe)
    steps: list[dict[str, Any]] = []
    rejected: set[str] = set()
    guard_breaches = 0
    best = {
        "cols": list(active),
        "logloss": baseline_ll,
        "metrics": dict(baseline),
    }

    budget = int(max_steps)
    while len(active) > floor and budget > 0:
        candidates = [f for f in removal_order if f in active and f not in rejected]
        if not candidates:
            break  # every remaining feature earned its place
        victim = candidates[0]
        trial = [c for c in active if c != victim]
        set_feature_subset(trial)
        try:
            m = _score_splits(splits)
        except Exception as exc:  # narrower set failed to train at all
            logger.warning("Scoring failed at %d cols (restoring %s): %s",
                           len(trial), victim, exc)
            m = None
        budget -= 1
        record = {
            "step": len(steps) + 1,
            "removed": victim,
            "n_features": len(trial),
            "metrics": m,
            "committed": False,
        }
        if m is not None:
            ll = float(m["logloss"])
            record["logloss"] = ll
            record["logloss_gain"] = round(baseline_ll - ll, 4)
            record["commit_threshold"] = round(threshold, 4)
            record["auc_drop"] = round(float(baseline["auc"]) - float(m["auc"]), 4)
            record["ece_rise"] = round(float(m["ece"]) - float(baseline["ece"]), 4)
            guards_ok = (
                ll <= baseline_ll - threshold
                and record["auc_drop"] <= auc_guard
                and record["ece_rise"] <= ece_guard
            )
            if guards_ok:
                active = trial
                record["committed"] = True
                if ll < best["logloss"]:
                    best = {"cols": list(active), "logloss": ll, "metrics": dict(m)}
            else:
                rejected.add(victim)
                if record["auc_drop"] > auc_guard or record["ece_rise"] > ece_guard:
                    guard_breaches += 1
        else:
            rejected.add(victim)
        steps.append(record)

    reset_feature_subset()
    final = best["cols"] if len(best["cols"]) < len(universe) else list(universe)
    return {
        "baseline_metrics": baseline,
        "best_metrics": best["metrics"],
        "selected_cols": final,
        "n_selected": len(final),
        "n_universe": len(universe),
        "steps": steps,
        "guard_breaches": guard_breaches,
        "floor": floor,
        "max_steps": max_steps,
    }


def confirm_candidate(
    games: pd.DataFrame, cols: list[str],
    alt_cadence: int = 5,
    auc_guard: float = RFE_AUC_GUARD,
    ece_guard: float = RFE_ECE_GUARD,
    min_logloss_gain: float = RFE_MIN_LOGLOSS_GAIN,
    noise_sigma: float = RFE_NOISE_SIGMA,
) -> dict[str, Any]:
    """Confirm a candidate on a DIFFERENT fold geometry before adoption.

    The RFE verdict is earned on production folds (cadence
    RETRAIN_CADENCE_DAYS). Before it may govern serving width it must also
    hold on an alternate expanding-window geometry (cadence ``alt_cadence``):
    a pruning that only helps one specific fold sequence is fold-noise, not
    signal. Both widths are scored on the SAME alternate splits (built once
    at universe width so geometry and OOF-margin enrichment are identical).

    Returns {"holds", "threshold", "universe", "candidate", "alt_cadence"};
    the caller must refuse adoption unless ``holds`` is True. Restores the
    caller's active subset afterwards.
    """
    from training import _attach_oof_run_margins, walk_forward_splits

    saved = list(active_moneyline_feature_cols())
    at_universe = saved == list(MONEYLINE_FEATURE_COLS)
    try:
        reset_feature_subset()
        splits = walk_forward_splits(
            games, retrain_cadence_days=alt_cadence,
            max_eval_folds=DEFAULT_MAX_EVAL_FOLDS)
        if MARGIN_COL in MONEYLINE_FEATURE_COLS:
            games, splits = _attach_oof_run_margins(
                games, splits, 0, DEFAULT_MAX_EVAL_FOLDS, alt_cadence, 0)
        out: dict[str, Any] = {}
        for label, subset in (("universe", None), ("candidate", cols)):
            if subset is None:
                reset_feature_subset()
            else:
                set_feature_subset(subset)
            out[label] = _score_splits(splits)
        threshold = max(float(min_logloss_gain),
                        noise_sigma * float(out["universe"].get("logloss_se", 0.0)))
        holds = (
            out["candidate"]["logloss"]
                <= out["universe"]["logloss"] - threshold
            and float(out["universe"]["auc"]) - float(out["candidate"]["auc"]) <= auc_guard
            and float(out["candidate"]["ece"]) - float(out["universe"]["ece"]) <= ece_guard
        )
        return {"holds": bool(holds), "threshold": round(threshold, 4),
                "universe": out["universe"], "candidate": out["candidate"],
                "alt_cadence": alt_cadence}
    finally:
        if at_universe:
            reset_feature_subset()
        else:
            set_feature_subset(saved)


# ── Trace + state governance ────────────────────────────────────────────────


def _trace_path(day: date) -> Path:
    return DATA_DELIVERY_DIR / f"{TRACE_PREFIX}{day.isoformat()}.json"


def write_trace(day: date, result: dict[str, Any], adopted: bool) -> Path:
    """Write the mlb_feature_selection_<date>.json record (never deleted)."""
    record = {
        "record": "mlb_feature_selection",
        "date": day.isoformat(),
        "generated_at": datetime.now().isoformat(),
        "seed": RANDOM_SEED,
        "objective": "minimize pooled ensemble logloss per fold",
        "guards": {"auc_drop_max": RFE_AUC_GUARD, "ece_rise_max": RFE_ECE_GUARD},
        "universe_cols": list(MONEYLINE_FEATURE_COLS),
        "selected_cols": result["selected_cols"],
        "n_selected": result["n_selected"],
        "baseline_metrics": result["baseline_metrics"],
        "best_metrics": result["best_metrics"],
        "steps": result["steps"],
        "guard_breaches": result.get("guard_breaches", 0),
        "adopted": bool(adopted),
    }
    out = _trace_path(day)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return out


def _state_path() -> Path:
    return DATA_DELIVERY_DIR / STATE_FILE


def load_state() -> Optional[dict[str, Any]]:
    p = _state_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Feature-selection state unreadable (%s) — universe wins", exc)
        return None


def write_state(cols: list[str], trace: dict[str, Any]) -> Path:
    """Record the adopted subset. Never merges — full replacement each adopt."""
    state = {
        "record": "mlb_feature_selection_state",
        "adopted_at": datetime.now().isoformat(),
        "trace_date": trace.get("date"),
        "cols": list(cols),
        "n_cols": len(cols),
        "baseline_metrics": trace.get("baseline_metrics"),
        "best_metrics": trace.get("best_metrics"),
    }
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return p


def apply_adopted_subset() -> dict[str, Any]:
    """Apply the adopted feature subset to the live training/serving width.

    Called by the daily pipeline at process start (before any model work).
    Universe wins on any problem: missing/corrupt state, empty cols, names not
    in the current universe (post-exp2 drift), or fewer than RFE_FLOOR.
    Returns a small report for the run summary; NEVER raises.
    """
    report: dict[str, Any] = {"applied": False, "n_cols": None, "source": "universe"}
    state = load_state()
    if not state:
        return report
    cols = state.get("cols") or []
    valid = [c for c in cols if c in MONEYLINE_FEATURE_COLS]
    if not cols or len(valid) != len(cols) or len(valid) < RFE_FLOOR:
        logger.warning(
            "Adopted feature subset rejected (%d/%d valid, floor %d) — "
            "serving at full universe width",
            len(valid), len(cols), RFE_FLOOR)
        return report
    try:
        set_feature_subset(valid)
        report.update(applied=True, n_cols=len(valid), source="rfe_state")
        logger.info("Applied adopted feature subset: %d/%d features",
                    len(valid), len(MONEYLINE_FEATURE_COLS))
    except ValueError as exc:
        logger.warning("Feature subset apply failed — universe wins: %s", exc)
    return report


def maybe_run_rfe(games: pd.DataFrame, day_or_str: "str | date",
                  weekday: Optional[int] = None) -> dict[str, Any]:
    """Weekly trigger wrapper for the daily pipeline. Fail-soft by contract.

    Runs the full RFE walk on scheduled days only (default: Mondays, weekday
    0) — off-schedule calls are a no-op. MLB_RFE_FORCE=1 forces a run on any
    day (used for manual/scratch verification). Writes the trace record;
    NEVER adopts. Accepts a date object or an ISO string (master_pipeline's
    ``end`` is a datetime.date). Any exception propagates — the caller
    (master_pipeline) catches and continues the daily run without selection.
    """
    day = day_or_str if isinstance(day_or_str, date) \
        else date.fromisoformat(str(day_or_str)[:10])
    dow = day.weekday() if weekday is None else weekday
    force = os.environ.get("MLB_RFE_FORCE", "").strip().lower() in ("1", "true", "yes")
    if dow != 0 and not force:
        return {"ran": False,
                "reason": "not scheduled (Mondays; set MLB_RFE_FORCE=1 to force)"}
    result = run_rfe(games)
    trace = write_trace(day, result, adopted=False)
    return {
        "ran": True,
        "trace": str(trace),
        "n_universe": result["n_universe"],
        "n_selected": result["n_selected"],
        "baseline_logloss": float(result["baseline_metrics"]["logloss"]),
        "best_logloss": float(result["best_metrics"]["logloss"]),
    }


# ── Data loading (mirrors the daily pipeline's frame) ───────────────────────


def load_games_for_date(day: date) -> pd.DataFrame:
    """Assemble the same decided frame the daily pipeline trains on.

    loads raw events through data_ingestion (cache-aware), derives features via
    pipeline's game-level path, and keeps only decided games — the exact
    training population of run_daily_pipeline.
    """
    from data_ingestion import load_game_events
    from frames import get_decided_frame
    import pipeline as _p

    games = load_game_events(day, real=os.environ.get("MLB_REAL", "").lower() in ("1", "true", "yes"))
    games = _p.attach_market_lines(
        games, _p.generate_synthetic_market_lines(games))
    decided = get_decided_frame(games)
    if decided.empty:
        raise RuntimeError(f"No decided games available through {day.isoformat()}")
    return decided


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Blend-level RFE feature selection (moneyline ensemble)")
    parser.add_argument("--date", required=True, help="as-of date YYYY-MM-DD")
    parser.add_argument("--adopt", action="store_true",
                        help="adopt the result: write the state file consumed "
                             "by the daily retrain")
    parser.add_argument("--floor", type=int, default=RFE_FLOOR)
    parser.add_argument("--max-steps", type=int, default=RFE_MAX_STEPS)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    day = date.fromisoformat(args.date)
    games = load_games_for_date(day)
    logger.info("RFE on %d decided games through %s (universe %d)",
                len(games), day.isoformat(), len(MONEYLINE_FEATURE_COLS))

    result = run_rfe(games, floor=args.floor, max_steps=args.max_steps)
    logger.info("RFE: %d -> %d features | logloss %.4f -> %.4f | steps %d",
                result["n_universe"], result["n_selected"],
                float(result["baseline_metrics"]["logloss"]),
                float(result["best_metrics"]["logloss"]),
                len(result["steps"]))

    trace = write_trace(day, result, adopted=args.adopt)
    print(f"trace: {trace}")
    if args.adopt:
        if result["n_selected"] >= result["n_universe"]:
            print("adoption refused: RFE removed nothing — universe remains "
                  "the serving width (nothing to adopt)")
            return 0
        print("confirming candidate on alternate fold geometry "
              "(cadence 5) before adoption …")
        confirm = confirm_candidate(games, result["selected_cols"])
        print(f"  alt-geometry universe  logloss {confirm['universe']['logloss']:.4f} "
              f"auc {confirm['universe']['auc']:.4f} ece {confirm['universe']['ece']:.4f}")
        print(f"  alt-geometry candidate logloss {confirm['candidate']['logloss']:.4f} "
              f"auc {confirm['candidate']['auc']:.4f} ece {confirm['candidate']['ece']:.4f}")
        if not confirm["holds"]:
            print("adoption refused: candidate does not hold on alternate "
                  "fold geometry — verdict is fold-noise, not signal")
            return 1
        state = write_state(result["selected_cols"], {
            "date": day.isoformat(),
            "baseline_metrics": result["baseline_metrics"],
            "best_metrics": result["best_metrics"],
            "alt_geometry_confirmation": {
                "holds": confirm["holds"],
                "universe": confirm["universe"],
                "candidate": confirm["candidate"],
                "alt_cadence": confirm["alt_cadence"],
            },
        })
        print(f"state: {state} — daily retrain will apply this subset")
    else:
        print("record-only run; pass --adopt to change serving width")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
