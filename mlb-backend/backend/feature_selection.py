"""Automated feature selection: blend-level RFE over the FULL known pool.

Primary objective: minimize pooled walk-forward logloss of the moneyline
ENSEMBLE (all five members, each with its own feature view, blended with the
run's adaptive weights). Secondary objectives are guards, not targets: a
commit is accepted only when pooled AUC does not significantly decline
(<= RFE_AUC_GUARD) and pooled ECE does not significantly rise
(<= RFE_ECE_GUARD, CLI-tunable).

Design decisions (locked):

* FULL-POOL SEARCH, importance-ranked. The trial space is
  training.KNOWN_FEATURE_COLS — the 61-feature generation universe
  (removal-eligible) plus config.RFE_CANDIDATE_COLS, the 164 generated
  PIT-safe pre-game candidates (addition-eligible). 225 features, no
  exclusions. One all-folds importance pass (feature_importance_weights
  averaged over a deterministic sample of walk-forward folds, weighted by
  training-window size) ranks every feature; trials proceed down that
  ranking, interleaving removals and additions by rank. Importance is a
  training-set usage statistic — a trial-order PRIOR only. Every verdict is
  measured: pooled full-history logloss vs a noise-calibrated threshold,
  plus the AUC/ECE guards. A wrong ranking wastes budget; it can never
  produce a wrong adoption.
* NO PERMANENT REMOVAL. A rejection deprioritizes a feature to the back of
  the trial queue (re-trialed once untried features are exhausted); an
  adopted record is a serving-width lever, never a deletion, and ``--reset``
  reverts serving width instantly. The incumbent (adopted) subset is
  re-scored against the universe every run and flagged when it stops
  earning its keep.
* REDUNDANCY AS INFORMATION. Features with |r| >= RFE_REDUNDANCY_R are
  trialed consecutively (untried siblings of a trialed feature follow it in
  the same run, budget permitting — else they lead the next run) so one
  measured verdict informs its correlated siblings. The trace records the
  pairs. No cluster machinery.
* FORCED TRIAL LISTS. MLB_RFE_ADDITION_MONEYLINE_LIST /
  MLB_RFE_REMOVAL_MONEYLINE_LIST (comma-separated env vars, blank by
  default) force specific addition/removal trials to the front of the run —
  overriding deprioritization and queue consumption. They force the TEST,
  never the result: every commit still passes the full gates. Names are
  matched exactly against the pool; unresolvable names are reported loudly
  (with nearest pool names) and recorded in the trace.
* SAME fold geometry as production: walk_forward_splits with
  RETRAIN_CADENCE_DAYS. Real runs score FULL HISTORY
  (max_eval_folds=DEFAULT_MAX_EVAL_FOLDS=0). A bounded run
  (--max-eval-folds N) is a smoke test only: loud warning, trace
  annotation, and adoption is REFUSED on a bounded-depth trace.
* QUEUE CONSUMPTION. A feature's full-depth verdict is not re-trialed while
  untried features remain (it is deprioritized, not dropped). Bounded runs
  never consume the queue. MLB_RFE_RETRIAL=1 forces a fresh sweep. Full
  first-pass coverage of the 225-feature pool therefore completes over
  ~6-7 runs at the default RFE_MAX_STEPS=40 budget.
* RECORD-ONLY by default: results land in
  data_delivery/mlb_feature_selection_<date>.json (10-day retention;
  feeds the RFE's cross-run prior-verdict memory and audits). Nothing
  changes at serving time unless the record
  is explicitly adopted (--adopt), which writes
  data_delivery/mlb_feature_selection_state.json after TWO gates: alternate
  fold-geometry confirmation (cadence 5) and slate-coverage verification
  (every adopted candidate must be >= RFE_SLATE_COVERAGE_FLOOR non-null on
  the REAL upcoming slate). The daily retrain applies the adopted subset
  with a safe fallback to the full universe.
* TRIGGER: runs ONLY when MLB_RFE_FORCE=1. Unset/0 = no-op. No calendar.
* SELF-DESCRIBING TRACES: invocation source (cli|pipeline), git SHA, fold
  depth, fold count, games scored, forced lists, per-fold logloss for every
  scored evaluation.
* Deterministic: RANDOM_SEED throughout; identical data + code -> identical
  verdict, step trace, and adoption decision.

Runs:      set MLB_RFE_FORCE=1 in the environment, then
           `python feature_selection.py --date YYYY-MM-DD`
Adoption:  `python feature_selection.py --date YYYY-MM-DD --adopt`
Reset:     `python feature_selection.py --reset`
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from config import (
    DEFAULT_MAX_EVAL_FOLDS,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
    RFE_AUC_GUARD,
    RFE_CANDIDATE_COLS,
    RFE_ECE_GUARD,
    RFE_FLOOR,
    RFE_MIN_LOGLOSS_GAIN,
    RFE_MAX_STEPS,
    RFE_NOISE_SIGMA,
    RFE_REDUNDANCY_R,
    RFE_SLATE_COVERAGE_FLOOR,
)
from explainability import DATA_DELIVERY_DIR
from training import (
    KNOWN_FEATURE_COLS,
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

ENV_FORCE = "MLB_RFE_FORCE"
ENV_RETRIAL = "MLB_RFE_RETRIAL"
ENV_ADDITION_LIST = "MLB_RFE_ADDITION_MONEYLINE_LIST"
ENV_REMOVAL_LIST = "MLB_RFE_REMOVAL_MONEYLINE_LIST"

# Universe members never leave candidacy; CANDIDATE_COLS is the
# addition-eligible half of the pool (defensively re-derived in case a
# config entry ever collides with the universe).
CANDIDATE_COLS = [c for c in RFE_CANDIDATE_COLS if c not in MONEYLINE_FEATURE_COLS]


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _git_sha() -> str:
    """Short SHA of the running checkout ('unknown' outside a git repo)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parent,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _parse_env_list(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _nearest_names(name: str, limit: int = 3) -> list[str]:
    return difflib.get_close_matches(name, KNOWN_FEATURE_COLS, n=limit, cutoff=0.4)


# ── Fold scoring on the real ensemble ───────────────────────────────────────


def _score_splits(splits: list[dict[str, Any]]) -> dict[str, float]:
    """Train + score the full ensemble on every fold; return pooled metrics.

    Mirrors the daily walk-forward loop's per-fold contract: train each fold's
    members with the val window for early stopping, predict the val window
    with the blended ensemble, pool compute_metrics across all validation
    rows. Evaluated with the CURRENT active subset — callers set it before
    calling. Deterministic under the module seed. The pooled record also
    carries per-fold logloss so the trace shows WHERE a width helps or
    hurts, not just the average.
    """
    per_fold: list[dict[str, Any]] = []
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
        eps = 1e-7
        p = np.clip(np.asarray(blend, dtype=float), eps, 1 - eps)
        losses = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        per_fold.append({
            "val_window": _val_window_label(val),
            "n_games": int(len(y)),
            "logloss": round(float(losses.mean()), 4),
        })
        y_all.append(y)
        p_all.append(p)
        n_used += 1

    if not n_used or not y_all:
        raise RuntimeError("RFE scoring produced no usable folds")

    y_true = np.concatenate(y_all)
    y_prob = np.concatenate(p_all)
    metrics = compute_metrics(y_true, y_prob)
    # Standard error of the pooled logloss estimate: a commit threshold of
    # ~2*SE is what separates real signal from val-window chance (fixed
    # floors cannot — see RFE_NOISE_SIGMA).
    eps = 1e-7
    p = np.clip(y_prob, eps, 1 - eps)
    losses = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    metrics["logloss_se"] = round(float(np.std(losses, ddof=1) / np.sqrt(len(losses))), 6)
    metrics["folds_used"] = n_used
    metrics["games_scored"] = int(len(y_true))
    metrics["per_fold"] = per_fold
    return metrics


def _val_window_label(val: pd.DataFrame) -> str:
    d = pd.to_datetime(val["game_date"], errors="coerce").dropna()
    if d.empty:
        return "?"
    return f"{d.min():%Y-%m-%d}..{d.max():%Y-%m-%d}"


def _splits_and_frame(games: pd.DataFrame,
                      max_eval_folds: int = DEFAULT_MAX_EVAL_FOLDS) -> tuple[pd.DataFrame, list]:
    """Attach OOF run margins when the margin feature is active, then split.

    Same geometry contract as the daily walk-forward: when run_margin_diff is
    in the active subset, folds are regenerated over the margin-enriched frame
    (build_oof_margin._attach_oof_run_margins asserts identical geometry).
    The production default (0) scores full history so the RFE view can never
    see fewer folds than the shipped metric; a bounded smoke run may pass N
    to score only the N most recent folds (the trace records the depth so a
    bounded verdict is never mistaken for a full one).
    """
    splits = walk_forward_splits(
        games, retrain_cadence_days=RETRAIN_CADENCE_DAYS,
        max_eval_folds=max_eval_folds)
    if MARGIN_COL not in active_moneyline_feature_cols():
        return games, splits
    from training import _attach_oof_run_margins

    enriched, splits = _attach_oof_run_margins(
        games, splits,
        min_val_games=0, max_eval_folds=max_eval_folds,
        retrain_cadence_days=RETRAIN_CADENCE_DAYS, min_train_days=0)
    return enriched, splits


# ── Importance prior (trial ORDER only — never a verdict) ───────────────────


def _importance_prior(splits: list[dict[str, Any]],
                      sample_step: int = 10) -> dict[str, float]:
    """All-folds ensemble importance, averaged over a deterministic sample.

    feature_importance_weights is a training-set usage statistic (trees:
    split-gain; logistic: |coef| on standardized inputs; blended by member
    weights). Averaging across folds spread through history — weighted by
    training-window size — makes the prior era-robust: features introduced
    mid-history (like the exp2 columns) are not diluted to zero by the early
    folds where they cannot exist. The last fold is always included.
    Subsample stride keeps this at ~1/40th of the cost of a single
    walk-forward evaluation. Returns {feature: mean share} (sums to 1).
    """
    n = len(splits)
    if n == 0:
        return {}
    idxs = list(range(0, n, sample_step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    acc: dict[str, float] = {}
    wsum = 0.0
    for i in idxs:
        sp = splits[i]
        if sp["train_games"].empty or sp["val_games"].empty:
            continue
        try:
            models, _ = train_moneyline_ensemble(sp["train_games"], sp["val_games"])
        except Exception as exc:
            logger.warning("Importance prior fold %d failed: %s", i, exc)
            continue
        if not models:
            continue
        imp = feature_importance_weights(models) or {}
        if not imp:
            continue
        s = sum(imp.values()) or 1.0
        w = float(len(sp["train_games"]))
        for f, v in imp.items():
            acc[f] = acc.get(f, 0.0) + (v / s) * w
        wsum += w
    if wsum <= 0:
        return {}
    return {f: v / wsum for f, v in acc.items()}


# ── Redundancy (informational ordering, not clusters) ───────────────────────


def _correlation_pairs(games: pd.DataFrame) -> list[tuple[str, str, float]]:
    """|r| >= RFE_REDUNDANCY_R pairs within the known pool.

    Computed once on the decided frame; used ONLY to order trials (a
    feature's untried redundant siblings follow it consecutively) and to
    annotate the trace. Constant/all-NaN columns are excluded silently.
    """
    pool = [c for c in KNOWN_FEATURE_COLS if c in games.columns]
    X = games[pool].apply(pd.to_numeric, errors="coerce")
    valid = [c for c in pool if X[c].std(skipna=True) is not None
             and not pd.isna(X[c].std(skipna=True)) and X[c].std(skipna=True) > 0]
    if len(valid) < 2:
        return []
    corr = X[valid].corr().abs()
    pairs: list[tuple[str, str, float]] = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and float(r) >= RFE_REDUNDANCY_R:
                pairs.append((cols[i], cols[j], round(float(r), 3)))
    return pairs


def _family_order(queue: list[str],
                  redundancy: dict[str, list[str]]) -> list[str]:
    """Reorder so a feature's untried redundant siblings are consecutive."""
    seen: set[str] = set()
    out: list[str] = []
    for f in queue:
        if f in seen:
            continue
        block = [f] + [s for s in redundancy.get(f, [])
                       if s not in seen and s in queue]
        out.extend(block)
        seen.update(block)
    return out


# ── The RFE walk ────────────────────────────────────────────────────────────


def run_rfe(
    games: pd.DataFrame,
    floor: int = RFE_FLOOR,
    max_steps: int = RFE_MAX_STEPS,
    auc_guard: float = RFE_AUC_GUARD,
    ece_guard: float = RFE_ECE_GUARD,
    min_logloss_gain: float = RFE_MIN_LOGLOSS_GAIN,
    noise_sigma: float = RFE_NOISE_SIGMA,
    max_eval_folds: int = DEFAULT_MAX_EVAL_FOLDS,
    forced_additions: Optional[list[str]] = None,
    forced_removals: Optional[list[str]] = None,
    retrial: bool = False,
    incumbent_cols: Optional[list[str]] = None,
    prior_verdicts: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Run the importance-ranked, full-pool RFE walk.

    Order of operations:
      1. Score the universe-width ensemble (baseline; also the incumbent
         health reference).
      2. All-folds importance prior ranks the full 225-feature pool.
      3. Trial queue = forced trials (env lists, front) + untried features
         (importance order, redundant siblings consecutive) + previously
         verdict-ed features (deprioritized tail; re-trialed only if budget
         remains). Bounded-depth prior verdicts never constrain a full run.
         TARGETED MODE: if a forced list is supplied, the queue is ONLY the
         listed trials — no ranking pass, no other scoring (fast targeted
         tests; ~minutes instead of hours).
      4. Trials proceed down the queue while budget lasts: a removal trial
         drops the feature from the active set, an addition trial appends
         it; the change commits only if pooled logloss improves by at least
         max(min_logloss_gain, noise_sigma * SE) AND AUC drop <= auc_guard
         AND ECE rise <= ece_guard (vs the running best). Otherwise the
         feature is rejected — deprioritized, never blacklisted.
      5. The incumbent (adopted) subset, if any, is re-scored against the
         universe and flagged when it no longer beats it.

    Deterministic end to end under RANDOM_SEED.
    """
    reset_feature_subset()
    universe = list(MONEYLINE_FEATURE_COLS)
    candidates = list(CANDIDATE_COLS)
    pool = list(KNOWN_FEATURE_COLS)

    enriched, splits = _splits_and_frame(games, max_eval_folds=max_eval_folds)
    if not splits:
        raise RuntimeError("No walk-forward folds for the supplied frame")
    full_depth = int(max_eval_folds) == 0
    folds_available = len(splits)

    baseline = _score_splits(splits)
    baseline_ll = float(baseline["logloss"])
    # A commit is accepted only on a gain the noise cannot explain: at least
    # min_logloss_gain AND at least noise_sigma standard errors of the
    # baseline pooled logloss estimate.
    threshold = max(float(min_logloss_gain),
                    noise_sigma * float(baseline.get("logloss_se", 0.0)))

    # Forced trial lists: strict resolution against the pool; anything
    # unresolvable is reported loudly (never silently dropped).
    forced_add_raw = list(forced_additions or [])
    forced_rm_raw = list(forced_removals or [])
    forced_add = [c for c in forced_add_raw if c in candidates]
    forced_rm = [c for c in forced_rm_raw if c in universe]
    unresolved = {
        "addition": [
            {"name": n, "nearest": _nearest_names(n)}
            for n in forced_add_raw if n not in candidates
        ],
        "removal": [
            {"name": n, "nearest": _nearest_names(n)}
            for n in forced_rm_raw if n not in universe
        ],
    }
    if unresolved["addition"] or unresolved["removal"]:
        logger.warning(
            "FORCED TRIAL LIST has unresolvable names (reported in trace) — "
            "resolvable entries still run in targeted mode: %s",
            {k: [u["name"] for u in v] for k, v in unresolved.items() if v})

    # TARGETED MODE: when the user supplies a forced list, run ONLY those
    # trials (baseline + the listed trials). Nothing else is ranked, queued,
    # or scored — a 2-name list costs baseline + 2 trials instead of a
    # 40-trial full search. Unresolvable names are reported loudly in the
    # trace and skipped; resolvable ones still run (forcing the TEST, never
    # the result).
    targeted = bool(forced_add or forced_rm)
    if not targeted and (forced_add_raw or forced_rm_raw):
        # Defensive: a targeted request where NOTHING resolved would fall
        # through to a silent full search and overwrite the full trace.
        # That is a user error — fail loud, never burn a 7-hour budget by
        # surprise.
        raise RuntimeError(
            "Targeted RFE requested but no forced name resolved against the "
            f"pool (candidates={len(candidates)}, universe={len(universe)}). "
            "Nearest-name suggestions were logged; check the workbook's "
            "Candidates sheet for exact names.")

    # Importance prior + redundancy map: full runs need both (queue order +
    # step annotations); targeted runs need neither — skipped for speed.
    if targeted:
        prior: dict[str, float] = {}
        pairs: list = []
    else:
        prior = _importance_prior(splits)
        pairs = _correlation_pairs(enriched)
    redundancy: dict[str, list[str]] = {}
    for a, b, _r in pairs:
        redundancy.setdefault(a, []).append(b)
        redundancy.setdefault(b, []).append(a)

    # Prior full-depth verdicts: deprioritized tail, dropped entirely under
    # RETRIAL, and never derived from bounded-depth traces (load_prior_verdicts
    # enforces that upstream; the folds_used >= current check re-asserts it).
    carried: dict[str, dict[str, Any]] = {}
    verdicted: set[str] = set()
    if not retrial and full_depth:
        for f, rec in (prior_verdicts or {}).items():
            if int(rec.get("folds_used", 0)) >= folds_available:
                carried[f] = rec
                verdicted.add(f)

    def rank_key(f: str) -> tuple:
        return (-prior.get(f, 0.0), f)

    removals_q = _family_order(sorted(universe, key=rank_key), redundancy)
    additions_q = _family_order(sorted(candidates, key=rank_key), redundancy)

    # Interleave removals and additions by importance rank (round-robin on
    # the two ranked queues, taking whichever head ranks higher next).
    fresh: list[dict[str, Any]] = []
    ri = ai = 0
    while ri < len(removals_q) or ai < len(additions_q):
        take_removal: Optional[bool] = None
        if ri >= len(removals_q):
            take_removal = False
        elif ai >= len(additions_q):
            take_removal = True
        else:
            take_removal = rank_key(removals_q[ri]) <= rank_key(additions_q[ai])
        if take_removal:
            f = removals_q[ri]; ri += 1
            if f not in forced_rm and f not in verdicted:
                fresh.append({"kind": "remove", "feature": f})
        else:
            f = additions_q[ai]; ai += 1
            if f not in forced_add and f not in verdicted:
                fresh.append({"kind": "add", "feature": f})
    tail: list[dict[str, Any]] = [
        {"kind": "remove", "feature": f} for f in removals_q
        if f not in forced_rm and f in verdicted
    ] + [
        {"kind": "add", "feature": f} for f in additions_q
        if f not in forced_add and f in verdicted
    ]
    forced: list[dict[str, Any]] = (
        [{"kind": "remove", "feature": f} for f in forced_rm]
        + [{"kind": "add", "feature": f} for f in forced_add]
    )
    if targeted:
        # Only the requested trials run — nothing is queued, ranked, or
        # scored beyond them.
        order = forced
    else:
        fresh: list[dict[str, Any]] = []
        ri = ai = 0
        while ri < len(removals_q) or ai < len(additions_q):
            take_removal: Optional[bool] = None
            if ri >= len(removals_q):
                take_removal = False
            elif ai >= len(additions_q):
                take_removal = True
            else:
                take_removal = rank_key(removals_q[ri]) <= rank_key(additions_q[ai])
            if take_removal:
                f = removals_q[ri]; ri += 1
                if f not in forced_rm and f not in verdicted:
                    fresh.append({"kind": "remove", "feature": f})
            else:
                f = additions_q[ai]; ai += 1
                if f not in forced_add and f not in verdicted:
                    fresh.append({"kind": "add", "feature": f})
        tail: list[dict[str, Any]] = [
            {"kind": "remove", "feature": f} for f in removals_q
            if f not in forced_rm and f in verdicted
        ] + [
            {"kind": "add", "feature": f} for f in additions_q
            if f not in forced_add and f in verdicted
        ]
        order = forced + fresh + tail

    active = list(universe)
    steps: list[dict[str, Any]] = []
    verdicts: dict[str, dict[str, Any]] = {}
    best = {
        "cols": list(active),
        "logloss": baseline_ll,
        "metrics": dict(baseline),
    }

    budget = int(max_steps)
    for entry in order:
        if budget <= 0:
            break
        f = entry["feature"]
        kind = entry["kind"]
        if kind == "remove":
            if f not in active:
                continue
            trial = [c for c in active if c != f]
        else:
            if f in active:
                continue
            trial = active + [f]
        set_feature_subset(trial)
        try:
            m = _score_splits(splits)
        except Exception as exc:  # the trial width failed to train at all
            logger.warning("Scoring failed at %d cols (%s %s): %s",
                           len(trial), kind, f, exc)
            m = None
        budget -= 1
        record: dict[str, Any] = {
            "step": len(steps) + 1,
            "action": kind,
            "feature": f,
            "n_features": len(trial),
            "metrics": None,
            "per_fold": None,
            "committed": False,
            "redundant_with": redundancy.get(f, []),
            "forced": kind == "remove" and f in forced_rm
                      or kind == "add" and f in forced_add,
        }
        if m is not None:
            ll = float(m["logloss"])
            record["metrics"] = {
                k: v for k, v in m.items() if k != "per_fold"
            }
            record["per_fold"] = m.get("per_fold", [])
            record["logloss"] = ll
            record["logloss_gain"] = round(best["logloss"] - ll, 4)
            record["commit_threshold"] = round(threshold, 4)
            record["auc_drop"] = round(float(best["metrics"]["auc"]) - float(m["auc"]), 4)
            record["ece_rise"] = round(float(m["ece"]) - float(best["metrics"]["ece"]), 4)
            guards_ok = (
                ll <= best["logloss"] - threshold
                and record["auc_drop"] <= auc_guard
                and record["ece_rise"] <= ece_guard
            )
            if guards_ok:
                active = trial
                record["committed"] = True
                verdicts[f] = {"verdict": "committed", "step": record["step"],
                               "folds_used": int(m.get("folds_used", 0)),
                               "logloss_gain": record["logloss_gain"]}
                if ll < best["logloss"]:
                    best = {"cols": list(active), "logloss": ll,
                            "metrics": {k: v for k, v in m.items()
                                        if k != "per_fold"}}
            else:
                verdicts[f] = {"verdict": "rejected", "step": record["step"],
                               "folds_used": int(m.get("folds_used", 0)),
                               "logloss_gain": record["logloss_gain"]}
        else:
            verdicts[f] = {"verdict": "rejected", "step": record["step"],
                           "folds_used": folds_available, "logloss_gain": None}
        steps.append(record)

    reset_feature_subset()

    # Incumbent health: re-score the adopted subset against the universe
    # (governance eval — outside the trial budget).
    incumbent_check: Optional[dict[str, Any]] = None
    if incumbent_cols:
        valid_inc = [c for c in incumbent_cols if c in pool]
        if valid_inc and len(valid_inc) == len(incumbent_cols) \
                and len(valid_inc) >= floor:
            try:
                set_feature_subset(valid_inc)
                im = _score_splits(splits)
                reset_feature_subset()
                incumbent_check = {
                    "metrics": {k: v for k, v in im.items() if k != "per_fold"},
                    "beats_universe": bool(
                        float(im["logloss"]) <= baseline_ll - threshold),
                    "note": ("adopted subset still earns its keep"
                             if float(im["logloss"]) <= baseline_ll - threshold
                             else "regressed vs universe — consider --reset"),
                }
            except Exception as exc:
                reset_feature_subset()
                incumbent_check = {"error": str(exc),
                                   "note": "incumbent re-score failed"}
        else:
            incumbent_check = {"note": "adopted subset invalid vs pool/floor",
                               "n_valid": len(valid_inc)}

    final = best["cols"] if len(best["cols"]) != len(universe) else list(universe)
    return {
        "baseline_metrics": {k: v for k, v in baseline.items()},
        "best_metrics": best["metrics"],
        "selected_cols": final,
        "n_selected": len(final),
        "n_universe": len(universe),
        "n_candidates": len(candidates),
        "n_pool": len(pool),
        "steps": steps,
        "verdicts": verdicts,
        "prior_verdicts": carried,
        "n_prior_carried": len(carried),
        "forced_lists": {
            "addition_resolved": forced_add,
            "removal_resolved": forced_rm,
            "unresolved": unresolved,
        },
        "importance_prior_top": dict(
            sorted(prior.items(), key=lambda kv: -kv[1])[:25]),
        "redundancy_pairs": [list(p) for p in pairs],
        "floor": floor,
        "max_steps": max_steps,
        "max_eval_folds": max_eval_folds,
        "full_depth": full_depth,
        "folds_available": folds_available,
        "retrial": retrial,
        "targeted": targeted,
        "run_mode": ("targeted_full_history" if targeted
                     else ("full_history" if full_depth else "bounded")),
        "targeted_trials": ({"additions": list(forced_add),
                             "removals": list(forced_rm)} if targeted else None),
        "guards": {"auc_drop_max": auc_guard, "ece_rise_max": ece_guard},
        "commit_threshold": round(threshold, 4),
        "incumbent_check": incumbent_check,
    }


def load_prior_verdicts(before_day: date) -> dict[str, dict[str, Any]]:
    """Full-depth verdicts from the newest trace STRICTLY BEFORE ``before_day``.

    Returns {} when no prior full-depth trace exists. Bounded traces never
    contribute — their verdicts were earned on too little data to constrain
    a full run.
    """
    best_path: Optional[Path] = None
    best_day: Optional[date] = None
    if DATA_DELIVERY_DIR.exists():
        for p in DATA_DELIVERY_DIR.glob(f"{TRACE_PREFIX}*.json"):
            try:
                d = date.fromisoformat(p.stem[len(TRACE_PREFIX):])
            except ValueError:
                continue
            if d < before_day and (best_day is None or d > best_day):
                best_day, best_path = d, p
    if best_path is None:
        return {}
    try:
        rec = json.loads(best_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Prior trace unreadable (%s): %s", best_path, exc)
        return {}
    if int(rec.get("max_eval_folds", 0)) != 0:
        return {}  # bounded-depth trace: not usable as prior
    verdicts = rec.get("verdicts") or {}
    out: dict[str, dict[str, Any]] = {}
    for f, v in verdicts.items():
        if isinstance(v, dict) and v.get("folds_used"):
            out[f] = {"verdict": v.get("verdict"),
                      "folds_used": int(v.get("folds_used", 0)),
                      "from_trace": best_path.name}
    return out


def confirm_candidate(
    games: pd.DataFrame, cols: list[str],
    alt_cadence: int = 5,
    auc_guard: float = RFE_AUC_GUARD,
    ece_guard: float = RFE_ECE_GUARD,
    min_logloss_gain: float = RFE_MIN_LOGLOSS_GAIN,
    noise_sigma: float = RFE_NOISE_SIGMA,
    max_eval_folds: int = DEFAULT_MAX_EVAL_FOLDS,
) -> dict[str, Any]:
    """Confirm a candidate on a DIFFERENT fold geometry before adoption.

    The RFE verdict is earned on production folds (cadence
    RETRAIN_CADENCE_DAYS). Before it may govern serving width it must also
    hold on an alternate expanding-window geometry (cadence ``alt_cadence``):
    a subset that only helps one specific fold sequence is fold-noise, not
    signal. Both widths are scored on the SAME alternate splits (built once
    at universe width so geometry and OOF-margin enrichment are identical).

    Returns {"holds", "threshold", "universe", "candidate", "alt_cadence"};
    the caller must refuse adoption unless ``holds`` is True. Restores the
    caller's active subset afterwards.
    """
    from training import _attach_oof_run_margins

    saved = list(active_moneyline_feature_cols())
    at_universe = saved == list(MONEYLINE_FEATURE_COLS)
    try:
        reset_feature_subset()
        splits = walk_forward_splits(
            games, retrain_cadence_days=alt_cadence,
            max_eval_folds=max_eval_folds)
        if MARGIN_COL in cols:
            games, splits = _attach_oof_run_margins(
                games, splits, 0, max_eval_folds, alt_cadence, 0)
        out: dict[str, Any] = {}
        for label, subset in (("universe", None), ("candidate", cols)):
            if subset is None:
                reset_feature_subset()
            else:
                set_feature_subset(subset)
            m = _score_splits(splits)
            out[label] = {k: v for k, v in m.items() if k != "per_fold"}
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


def confirm_slate_coverage(
    games: pd.DataFrame,
    cols: list[str],
    slate_df: Optional[pd.DataFrame] = None,
    target_date: Optional[date] = None,
) -> dict[str, Any]:
    """Adoption gate: candidates must survive on the REAL upcoming slate.

    Builds tomorrow's slate exactly as the serving pipeline does
    (data_ingestion.build_upcoming_slate + the same diff/form-delta/exp2
    builders) and measures non-null coverage for every selected CANDIDATE
    column. Universe members are exempt — they already ship. Refuses when
    any candidate falls below RFE_SLATE_COVERAGE_FLOOR, when the slate
    cannot be built (candidates unprovable), or when the pbp context needed
    for pitcher mapping is absent while pitcher candidates are selected
    (their NaN would be an artifact of the gate, not of serving).

    Pass ``slate_df``/``target_date`` for scratch verification without
    network access.
    """
    candidates_selected = [c for c in cols if c in CANDIDATE_COLS]
    if not candidates_selected:
        return {"verified": True, "note": "universe-only selection — gate not applicable",
                "coverage": {}, "floor": RFE_SLATE_COVERAGE_FLOOR}

    from data_ingestion import build_upcoming_slate
    from features import add_diff_features, add_exp2_features, add_form_delta_features

    if target_date is None:
        d = pd.to_datetime(games["game_date"], errors="coerce").dropna()
        if d.empty:
            return {"verified": False, "reason": "no dates in the decided frame",
                    "coverage": {}, "floor": RFE_SLATE_COVERAGE_FLOOR}
        target_date = (d.max() + timedelta(days=1)).date()

    slate = slate_df
    if slate is None:
        pbp_df = None
        chunk_dir = DATA_DELIVERY_DIR / "pbp_chunks"
        try:
            files = sorted(chunk_dir.glob("pbp_*.parquet"))
            if files:
                pbp_df = pd.concat(
                    (pd.read_parquet(f, columns=["player_name", "pitcher"])
                     for f in files), ignore_index=True)
        except Exception as exc:
            logger.warning("pbp context unavailable for slate gate: %s", exc)
            pbp_df = None
        needs_pbp = any(c.startswith(("sp_", "bullpen_")) for c in candidates_selected)
        if needs_pbp and pbp_df is None:
            return {"verified": False,
                    "reason": ("pitcher candidates selected but pbp context is "
                               "unavailable — coverage would be measured unfairly"),
                    "coverage": {}, "floor": RFE_SLATE_COVERAGE_FLOOR}
        try:
            slate = build_upcoming_slate(games, target_date, pbp_df=pbp_df)
        except Exception as exc:
            return {"verified": False,
                    "reason": f"upcoming slate could not be built: {exc}",
                    "coverage": {}, "floor": RFE_SLATE_COVERAGE_FLOOR}
    if slate is None or slate.empty:
        return {"verified": False,
                "reason": f"no scheduled games found for {target_date.isoformat()} "
                          "(candidates unprovable on an empty slate)",
                "coverage": {}, "floor": RFE_SLATE_COVERAGE_FLOOR,
                "slate_date": target_date.isoformat()}

    slate = add_diff_features(slate)
    slate = add_form_delta_features(slate)
    slate = add_exp2_features(slate)

    coverage = {
        c: round(float(slate[c].notna().mean()), 3)
        if c in slate.columns else 0.0
        for c in candidates_selected
    }
    below = {c: v for c, v in coverage.items() if v < RFE_SLATE_COVERAGE_FLOOR}
    return {
        "verified": not below,
        "coverage": coverage,
        "below_floor": below,
        "floor": RFE_SLATE_COVERAGE_FLOOR,
        "slate_date": target_date.isoformat(),
        "slate_games": int(len(slate)),
    }


# ── Trace + state governance ────────────────────────────────────────────────


def _trace_path(day: date, suffix: str = "") -> Path:
    """Trace record path. A targeted run gets a ``_targeted`` suffix so it
    can never silently overwrite the same-day full trace record (the
    never-deleted full-depth search history is exactly what the RFE's
    cross-run  memory builds on — retention keeps 10 days of traces; the prior-verdict
  memory degrades to a fresh search if all prior traces age out, and
  adoption reads the never-deleted state file, never a trace)."""
    return DATA_DELIVERY_DIR / f"{TRACE_PREFIX}{day.isoformat()}{suffix}.json"


def write_trace(day: date, result: dict[str, Any], adopted: bool,
                invoked_by: str = "cli") -> Path:
    """Write the mlb_feature_selection_<date>.json record (10-day
    retention; adoption reads the never-deleted state file)."""
    record = {
        "record": "mlb_feature_selection",
        "date": day.isoformat(),
        "generated_at": datetime.now().isoformat(),
        "seed": RANDOM_SEED,
        "objective": "minimize pooled ensemble logloss over walk-forward folds",
        "invocation": {
            "invoked_by": invoked_by,
            "git_sha": _git_sha(),
            "max_eval_folds": result.get("max_eval_folds", DEFAULT_MAX_EVAL_FOLDS),
            "run_mode": result.get(
                "run_mode",
                "full_history" if result.get("full_depth") else "BOUNDED"),
            "folds_available": result.get("folds_available"),
            "retrial": result.get("retrial", False),
            "forced_lists": result.get("forced_lists"),
        },
        "guards": {
            "auc_drop_max": result.get("guards", {}).get("auc_drop_max", RFE_AUC_GUARD),
            "ece_rise_max": result.get("guards", {}).get("ece_rise_max", RFE_ECE_GUARD),
            "min_logloss_gain": RFE_MIN_LOGLOSS_GAIN,
            "noise_sigma": RFE_NOISE_SIGMA,
            "commit_threshold": result.get("commit_threshold"),
        },
        "pool": {
            "n_universe": result.get("n_universe"),
            "n_candidates": result.get("n_candidates"),
            "n_pool": result.get("n_pool"),
            "universe_cols": list(MONEYLINE_FEATURE_COLS),
            "importance_prior_top": result.get("importance_prior_top"),
            "redundancy_pairs": result.get("redundancy_pairs"),
        },
        "max_eval_folds": result.get("max_eval_folds", DEFAULT_MAX_EVAL_FOLDS),
        "selected_cols": result["selected_cols"],
        "n_selected": result["n_selected"],
        "baseline_metrics": result["baseline_metrics"],
        "best_metrics": result["best_metrics"],
        "steps": result["steps"],
        "verdicts": result.get("verdicts"),
        "prior_verdicts_carried": result.get("n_prior_carried", 0),
        "incumbent_check": result.get("incumbent_check"),
        "adopted": bool(adopted),
    }
    suffix = "_targeted" if result.get("targeted") else ""
    out = _trace_path(day, suffix)
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
        "alt_geometry_confirmation": trace.get("alt_geometry_confirmation"),
        "slate_coverage": trace.get("slate_coverage"),
    }
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return p


def reset_serving_width() -> dict[str, Any]:
    """Revert serving width to the full universe by removing the state file.

    Serving width is a lever, never a deletion: the pool and every feature's
    candidacy are untouched; the next daily run trains and serves at
    universe width. Safe to call when no state exists.
    """
    p = _state_path()
    existed = p.exists()
    if existed:
        p.unlink()
    return {"reset": True, "state_file_removed": existed,
            "serving_width": "full universe"}


def apply_adopted_subset() -> dict[str, Any]:
    """Apply the adopted feature subset to the live training/serving width.

    Called by the daily pipeline at process start (before any model work).
    Universe wins on any problem: missing/corrupt state, empty cols, names
    not in the known pool (post-expansion drift), or fewer than RFE_FLOOR.
    Returns a small report for the run summary; NEVER raises.
    """
    report: dict[str, Any] = {"applied": False, "n_cols": None, "source": "universe"}
    state = load_state()
    if not state:
        return report
    cols = state.get("cols") or []
    valid = [c for c in cols if c in KNOWN_FEATURE_COLS]
    if not cols or len(valid) != len(cols) or len(valid) < RFE_FLOOR:
        logger.warning(
            "Adopted feature subset rejected (%d/%d valid vs pool, floor %d) — "
            "serving at full universe width",
            len(valid), len(cols), RFE_FLOOR)
        return report
    try:
        set_feature_subset(valid)
        report.update(applied=True, n_cols=len(valid), source="rfe_state")
        logger.info("Applied adopted feature subset: %d/%d pool features",
                    len(valid), len(KNOWN_FEATURE_COLS))
    except ValueError as exc:
        logger.warning("Feature subset apply failed — universe wins: %s", exc)
    return report


def maybe_run_rfe(games: pd.DataFrame, day_or_str: "str | date") -> dict[str, Any]:
    """Pipeline trigger. Runs ONLY when MLB_RFE_FORCE=1; otherwise a no-op.

    No calendar logic: you decide from the notebook when a run happens.
    Writes the trace record; NEVER adopts. Any exception propagates — the
    caller (master_pipeline) catches and continues the daily run without
    selection.
    """
    if not _truthy_env(ENV_FORCE):
        return {"ran": False,
                "reason": "MLB_RFE_FORCE not set — RFE does not run"}
    day = day_or_str if isinstance(day_or_str, date) \
        else date.fromisoformat(str(day_or_str)[:10])
    retrial = _truthy_env(ENV_RETRIAL)
    forced_add = _parse_env_list(os.environ.get(ENV_ADDITION_LIST, ""))
    forced_rm = _parse_env_list(os.environ.get(ENV_REMOVAL_LIST, ""))
    prior = {} if retrial else load_prior_verdicts(day)
    state = load_state()
    result = run_rfe(
        games,
        forced_additions=forced_add,
        forced_removals=forced_rm,
        retrial=retrial,
        incumbent_cols=(state or {}).get("cols"),
        prior_verdicts=prior,
    )
    trace = write_trace(day, result, adopted=False, invoked_by="pipeline")
    inc = result.get("incumbent_check") or {}
    return {
        "ran": True,
        "trace": str(trace),
        "run_mode": result["run_mode"],
        "targeted": result.get("targeted", False),
        "n_pool": result["n_pool"],
        "n_universe": result["n_universe"],
        "n_selected": result["n_selected"],
        "baseline_logloss": float(result["baseline_metrics"]["logloss"]),
        "best_logloss": float(result["best_metrics"]["logloss"]),
        "n_trials": len(result["steps"]),
        "n_committed": sum(1 for s in result["steps"] if s.get("committed")),
        "incumbent_note": inc.get("note"),
        "forced_unresolved": {
            k: [u["name"] for u in v]
            for k, v in result["forced_lists"]["unresolved"].items() if v
        } or None,
    }


# ── Data loading (mirrors the daily pipeline's frame) ───────────────────────


def load_games_for_date(day: date) -> pd.DataFrame:
    """Load the same decided frame the daily pipeline trains on.

    Reads the production feature artifact (data_delivery/game_level_features.csv)
    through data_ingestion.load_game_features — the exact frame master_pipeline
    Phase 4 hands run_daily_pipeline (column mapping, ELO/win-pct/run-diff
    derivation included) — and keeps only decided games via
    frames.get_decided_frame (the canonical decided-frame contract).

    Override the path with MLB_FEATURES_CSV for scratch ablations against a
    different frame (e.g. an older CSV to check verdict stability).
    """
    from data_ingestion import load_game_features
    from frames import get_decided_frame

    csv = Path(os.environ.get("MLB_FEATURES_CSV") or
               (DATA_DELIVERY_DIR / "game_level_features.csv"))
    games = load_game_features(csv)
    decided = get_decided_frame(games)
    if decided.empty:
        raise RuntimeError(
            f"No decided games available through {day.isoformat()} "
            f"(frame: {csv})")
    return decided


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Blend-level RFE over the full known feature pool "
                    "(moneyline ensemble)")
    parser.add_argument("--date", help="as-of date YYYY-MM-DD (required unless --reset)")
    parser.add_argument("--adopt", action="store_true",
                        help="adopt the result: write the state file consumed "
                             "by the daily retrain (gated; full-depth run required)")
    parser.add_argument("--reset", action="store_true",
                        help="revert serving width to the full universe "
                             "(removes the state file; nothing is deleted)")
    parser.add_argument("--floor", type=int, default=RFE_FLOOR)
    parser.add_argument("--max-steps", type=int, default=RFE_MAX_STEPS)
    parser.add_argument("--max-eval-folds", type=int,
                        default=DEFAULT_MAX_EVAL_FOLDS,
                        help="score only the N most recent folds — SMOKE TESTS "
                             "ONLY (0 = full history, the production default; "
                             "a bounded run cannot be adopted)")
    parser.add_argument("--ece-guard", type=float, default=RFE_ECE_GUARD,
                        help="max pooled ECE rise a commit may cause")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.reset:
        report = reset_serving_width()
        print(f"reset: {json.dumps(report)}")
        return 0
    if not args.date:
        parser.error("--date is required unless --reset")

    day = date.fromisoformat(args.date)
    games = load_games_for_date(day)
    logger.info("RFE on %d decided games through %s (pool %d = universe %d + "
                "candidates %d)",
                len(games), day.isoformat(), len(KNOWN_FEATURE_COLS),
                len(MONEYLINE_FEATURE_COLS), len(CANDIDATE_COLS))

    if args.max_eval_folds and int(args.max_eval_folds) > 0:
        print("⚠️  BOUNDED RUN — scoring only the most recent "
              f"{args.max_eval_folds} fold(s). Results are a smoke test: "
              "noisy, queue-nonconsuming, and INELIGIBLE FOR ADOPTION.")

    forced_add = _parse_env_list(os.environ.get(ENV_ADDITION_LIST, ""))
    forced_rm = _parse_env_list(os.environ.get(ENV_REMOVAL_LIST, ""))
    if forced_add or forced_rm:
        print(f"forced trials — additions: {forced_add or 'none'} | "
              f"removals: {forced_rm or 'none'}")

    retrial = _truthy_env(ENV_RETRIAL)
    prior = {} if retrial else load_prior_verdicts(day)
    state = load_state()

    result = run_rfe(
        games, floor=args.floor, max_steps=args.max_steps,
        ece_guard=args.ece_guard, max_eval_folds=args.max_eval_folds,
        forced_additions=forced_add, forced_removals=forced_rm,
        retrial=retrial, incumbent_cols=(state or {}).get("cols"),
        prior_verdicts=prior,
    )
    logger.info("RFE: pool %d | baseline logloss %.4f -> best %.4f | "
                "trials %d | committed %d | selected %d cols",
                result["n_pool"],
                float(result["baseline_metrics"]["logloss"]),
                float(result["best_metrics"]["logloss"]),
                len(result["steps"]),
                sum(1 for s in result["steps"] if s.get("committed")),
                result["n_selected"])

    trace = write_trace(day, result, adopted=args.adopt, invoked_by="cli")
    print(f"trace: {trace}")
    inc = result.get("incumbent_check")
    if inc and inc.get("note"):
        print(f"incumbent: {inc['note']}")

    if args.adopt:
        if not result["full_depth"]:
            print("adoption refused: BOUNDED-DEPTH trace — a subset may only "
                  "be adopted from a full-history run (omit --max-eval-folds)")
            return 1
        if result["n_selected"] >= result["n_universe"] \
                and result["best_metrics"]["logloss"] >= float(
                    result["baseline_metrics"]["logloss"]):
            print("adoption refused: RFE improved nothing — universe remains "
                  "the serving width (nothing to adopt)")
            return 0
        print("confirming candidate on alternate fold geometry "
              "(cadence 5) before adoption …")
        confirm = confirm_candidate(games, result["selected_cols"],
                                    max_eval_folds=args.max_eval_folds)
        print(f"  alt-geometry universe  logloss {confirm['universe']['logloss']:.4f} "
              f"auc {confirm['universe']['auc']:.4f} ece {confirm['universe']['ece']:.4f}")
        print(f"  alt-geometry candidate logloss {confirm['candidate']['logloss']:.4f} "
              f"auc {confirm['candidate']['auc']:.4f} ece {confirm['candidate']['ece']:.4f}")
        if not confirm["holds"]:
            print("adoption refused: candidate does not hold on alternate "
                  "fold geometry — verdict is fold-noise, not signal")
            return 1
        print("verifying candidate coverage on the REAL upcoming slate …")
        slate = confirm_slate_coverage(games, result["selected_cols"])
        print(f"  slate gate: verified={slate.get('verified')} "
              f"({slate.get('slate_games', '?')} games on "
              f"{slate.get('slate_date', '?')})")
        if not slate.get("verified"):
            print(f"adoption refused: slate coverage gate failed — "
                  f"{slate.get('below_floor') or slate.get('reason')}")
            return 1
        state_out = write_state(result["selected_cols"], {
            "date": day.isoformat(),
            "baseline_metrics": result["baseline_metrics"],
            "best_metrics": result["best_metrics"],
            "alt_geometry_confirmation": {
                "holds": confirm["holds"],
                "universe": confirm["universe"],
                "candidate": confirm["candidate"],
                "alt_cadence": confirm["alt_cadence"],
            },
            "slate_coverage": slate,
        })
        print(f"state: {state_out} — daily retrain will apply this subset")
    else:
        print("record-only run; pass --adopt to change serving width "
              "(full-history runs only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
