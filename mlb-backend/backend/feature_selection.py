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
  measured: pooled full-history logloss vs a noise-calibrated threshold
  (paired per-game difference SE — the same folds are scored twice, so the
  bar reflects the CHANGE's noise, not the level's),
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
  targeted runs write mlb_feature_selection_<date>_targeted_<HHMM>.json,
  unique per run), feeding the RFE's cross-run prior-verdict memory and
  audits. Nothing
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
    RFE_GRID_MAX_STATES,
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


def _score_splits(splits: list[dict[str, Any]],
                  return_losses: bool = False):
    """Train + score the full ensemble on every fold; return pooled metrics.

    Mirrors the daily walk-forward loop's per-fold contract: train each fold's
    members with the val window for early stopping, predict the val window
    with the blended ensemble, pool compute_metrics across all validation
    rows. Evaluated with the CURRENT active subset — callers set it before
    calling. Deterministic under the module seed. The pooled record also
    carries per-fold logloss so the trace shows WHERE a width helps or
    hurts, not just the average.

    ``return_losses=True`` additionally returns the concatenated per-game
    loss vector in fold order — the basis of the paired-difference commit
    bar (every call scores the SAME rows in the SAME order, so differences
    against another call's vector are game-aligned by construction).
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
    if return_losses:
        return metrics, losses
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
         max(min_logloss_gain, noise_sigma * PAIRED_SE) — the standard error
         of the per-game logloss DIFFERENCE vs the running champion (same
         folds scored twice; shared difficulty cancels) — AND AUC drop <=
         auc_guard AND ECE rise <= ece_guard. Otherwise the feature is
         rejected — deprioritized, never blacklisted.
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

    baseline, base_losses = _score_splits(splits, return_losses=True)
    baseline_ll = float(baseline["logloss"])
    # A commit is accepted only on a gain the noise cannot explain: at least
    # min_logloss_gain AND at least noise_sigma standard errors of the
    # PAIRED per-game logloss difference (trial vs the current champion —
    # the same folds are scored twice, so shared per-game difficulty cancels
    # and the SE reflects the CHANGE's noise, not the level's). Per-step
    # thresholds are computed in the trial loop; this is the legacy
    # baseline-SE fallback bar for steps where pairing is impossible.
    fallback_threshold = max(float(min_logloss_gain),
                             noise_sigma * float(baseline.get("logloss_se", 0.0)))
    threshold = fallback_threshold
    best_losses = base_losses

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
            m, t_losses = _score_splits(splits, return_losses=True)
        except Exception as exc:  # the trial width failed to train at all
            logger.warning("Scoring failed at %d cols (%s %s): %s",
                           len(trial), kind, f, exc)
            m, t_losses = None, None
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
            # Paired-difference SE vs the CURRENT champion (best_losses):
            # per-game diffs cancel shared game difficulty. Fold-set mismatch
            # (a fold failed for this trial) -> legacy baseline-SE bar.
            if (t_losses is not None and best_losses is not None
                    and len(t_losses) == len(best_losses)):
                diff = t_losses - best_losses
                paired_se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
                step_threshold = max(float(min_logloss_gain),
                                     noise_sigma * paired_se)
                record["paired_se"] = round(paired_se, 6)
            else:
                if t_losses is not None:
                    logger.warning(
                        "Paired SE unavailable (%s %s): fold-set mismatch "
                        "(%d vs %d rows) — legacy baseline-SE bar used",
                        kind, f, len(t_losses),
                        len(best_losses) if best_losses is not None else 0)
                step_threshold = fallback_threshold
            record["commit_threshold"] = round(step_threshold, 4)
            record["auc_drop"] = round(float(best["metrics"]["auc"]) - float(m["auc"]), 4)
            record["ece_rise"] = round(float(m["ece"]) - float(best["metrics"]["ece"]), 4)
            guards_ok = (
                ll <= best["logloss"] - step_threshold
                and record["auc_drop"] <= auc_guard
                and record["ece_rise"] <= ece_guard
            )
            if guards_ok:
                active = trial
                record["committed"] = True
                if t_losses is not None:
                    best_losses = t_losses
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
                im, inc_losses = _score_splits(splits, return_losses=True)
                reset_feature_subset()
                # Same paired bar as the trial loop: the incumbent must beat
                # the universe by a REAL margin, judged on per-game diffs.
                if (inc_losses is not None and base_losses is not None
                        and len(inc_losses) == len(base_losses)):
                    diff = inc_losses - base_losses
                    inc_se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
                    inc_threshold = max(float(min_logloss_gain),
                                        noise_sigma * inc_se)
                else:
                    inc_threshold = fallback_threshold
                incumbent_check = {
                    "metrics": {k: v for k, v in im.items() if k != "per_fold"},
                    "bar_basis": "paired_diff_2sigma",
                    "beats_universe": bool(
                        float(im["logloss"]) <= baseline_ll - inc_threshold),
                    "note": ("adopted subset still earns its keep"
                             if float(im["logloss"]) <= baseline_ll - inc_threshold
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
        "bar_basis": "paired_diff_2sigma",
        "commit_threshold": round(threshold, 4),  # legacy fallback bar (baseline SE)
        "incumbent_check": incumbent_check,
    }


def load_prior_verdicts(before_day: date) -> dict[str, dict[str, Any]]:
    """Full-depth verdicts from the newest trace STRICTLY BEFORE ``before_day``.

    Returns {} when no prior full-depth trace exists. Bounded traces never
    contribute — their verdicts were earned on too little data to constrain
    a full run. TARGETED traces are skipped explicitly: they only cover
    their forced features, so they must never stand in for the full-run
    verdict memory.
    """
    best_path: Optional[Path] = None
    best_day: Optional[date] = None
    if DATA_DELIVERY_DIR.exists():
        for p in DATA_DELIVERY_DIR.glob(f"{TRACE_PREFIX}*.json"):
            if "_targeted" in p.stem:
                continue  # targeted probe — never the full-run memory
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


# ── Grid mode: all subset combinations of forced adds × removals ────────────

ENV_GRID_MODE = "MLB_RFE_ADDITION_REMOVAL_GRID_MODE"
ENV_GRID_MAX_STATES = "RFE_GRID_MAX_STATES"

# Worker-process globals (Windows spawn: state arrives via the initializer,
# NOT via closures or the fork-inherited module state).
_GRID_CTX: dict[str, Any] = {}


def _grid_worker_init(enriched: pd.DataFrame,
                      splits: list[dict[str, Any]]) -> None:
    """ProcessPoolExecutor initializer: hand the frame + splits to a worker.

    Windows spawns a fresh interpreter per worker, so the parent's module
    globals (including the active feature subset) do not exist here. Each
    worker gets the enriched frame and the FIXED splits once, then serves
    many state-scoring tasks. Trees are pinned to one thread per worker —
    parallelism comes from the process pool, not from BLAS inside it.
    """
    _GRID_CTX["enriched"] = enriched
    _GRID_CTX["splits"] = splits
    reset_feature_subset()


def _grid_worker_score(cols: list[str]) -> dict[str, Any]:
    """Score ONE grid state: a specific feature list on the shared splits.

    Returns pooled metrics + the per-game loss vector (game-aligned with
    every other state by fold order — the paired bar's foundation).
    """
    from training import set_feature_subset as _set, reset_feature_subset as _reset
    enriched: pd.DataFrame = _GRID_CTX["enriched"]
    splits: list[dict[str, Any]] = _GRID_CTX["splits"]
    _set(cols)
    try:
        m, losses = _score_splits(splits, return_losses=True)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        _reset()
    if losses is None:
        return {"ok": False, "error": "no loss vector (fold-set mismatch)"}
    return {"ok": True, "metrics": {k: v for k, v in m.items()
                                    if k != "per_fold"},
            "per_fold": m.get("per_fold", []),
            "losses": losses}


def _grid_window(forced_add: list[str], forced_rm: list[str],
                 max_states: int, prior: dict[str, float]) -> tuple[list[str], list[str], list[str]]:
    """Silently shrink the add/remove lists to fit the state cap.

    Round-robin: A1, R1, A2, R2, ... — extend the window with the next name
    from whichever side still has overflow ONLY while the resulting lattice
    (2^adds x 2^removes) stays <= max_states. Single-name lists grow to 4;
    balanced lists stop at 2+2. Overflow names still get 1-by-1 targeted
    trials after the grid, so every name the user listed is honored — the
    cap shapes the LATTICE, never silently drops a test.
    """
    def _key(f: str) -> tuple:
        return (-prior.get(f, 0.0), f)

    adds = sorted(forced_add, key=_key)
    removes = sorted(forced_rm, key=_key)
    win_a: list[str] = []
    win_r: list[str] = []
    tail: list[str] = []
    ia = ir = 0
    while True:
        progressed = False
        if ia < len(adds):
            cand_a = 2 ** (len(win_a) + 1) * 2 ** len(win_r)
            if cand_a <= max_states:
                win_a.append(adds[ia])
                ia += 1
                progressed = True
            else:
                tail.append(adds[ia])
                ia += 1
                progressed = True
        if ir < len(removes):
            cand_r = 2 ** len(win_a) * 2 ** (len(win_r) + 1)
            if cand_r <= max_states:
                win_r.append(removes[ir])
                ir += 1
                progressed = True
            else:
                tail.append(removes[ir])
                ir += 1
                progressed = True
        if not progressed:
            break
    return win_a, win_r, tail


def _grid_edge_verdict(name_from: str, name_to: str,
                       from_losses: Optional[np.ndarray],
                       to_losses: Optional[np.ndarray],
                       from_metrics: Optional[dict[str, Any]],
                       to_metrics: Optional[dict[str, Any]],
                       min_logloss_gain: float, noise_sigma: float,
                       auc_guard: float, ece_guard: float,
                       fallback_threshold: float) -> dict[str, Any]:
    """One paired-difference verdict between two grid states.

    The EXACT same bar a normal run applies to a trial: the commit threshold
    is max(RFE_MIN_LOGLOSS_GAIN, noise_sigma * paired SE) computed on the
    per-game logloss difference, plus the pooled AUC/ECE guards. Here both
    sides are fixed grid states, so the reference is the state named in the
    edge — usually the baseline.
    """
    edge: dict[str, Any] = {"from": name_from, "to": name_to}
    if to_metrics is None or from_metrics is None:
        edge.update(committed=False, verdict="not scored (state failed)")
        return edge
    gain = round(float(from_metrics["logloss"]) - float(to_metrics["logloss"]), 4)
    edge["logloss_gain"] = gain
    if (to_losses is not None and from_losses is not None
            and len(to_losses) == len(from_losses)):
        diff = to_losses - from_losses
        paired_se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
        threshold = max(float(min_logloss_gain), noise_sigma * paired_se)
        edge["paired_se"] = round(paired_se, 6)
        edge["bar_basis"] = "paired_diff_2sigma"
    else:
        threshold = fallback_threshold
        edge["bar_basis"] = "baseline_se_fallback"
    edge["commit_threshold"] = round(threshold, 4)
    edge["auc_drop"] = round(float(from_metrics["auc"]) - float(to_metrics["auc"]), 4)
    edge["ece_rise"] = round(float(to_metrics["ece"]) - float(from_metrics["ece"]), 4)
    edge["committed"] = bool(
        float(to_metrics["logloss"]) <= float(from_metrics["logloss"]) - threshold
        and edge["auc_drop"] <= auc_guard
        and edge["ece_rise"] <= ece_guard)
    edge["verdict"] = (
        f"COMMITTED — {gain:+.4f} clears the {threshold:.4f} noise bar"
        if edge["committed"] else
        f"not committed — gain {gain:+.4f} vs the {threshold:.4f} noise bar")
    return edge


def run_grid_rfe(
    games: pd.DataFrame,
    forced_additions: list[str],
    forced_removals: list[str],
    max_eval_folds: int = DEFAULT_MAX_EVAL_FOLDS,
    auc_guard: float = RFE_AUC_GUARD,
    ece_guard: float = RFE_ECE_GUARD,
    min_logloss_gain: float = RFE_MIN_LOGLOSS_GAIN,
    noise_sigma: float = RFE_NOISE_SIGMA,
    incumbent_cols: Optional[list[str]] = None,
    floor: int = RFE_FLOOR,
    max_workers: Optional[int] = None,
) -> dict[str, Any]:
    """Grid mode: score EVERY subset combination of the forced lists.

    With forced additions [A1, A2] and removals [R1, R2] the lattice is
    {∅,A1,A2,A1+A2} × {∅,R1,R2,R1+R2} = 16 states — baseline, each single,
    each pair, and the crossed cells. All states are scored ONCE each (in
    parallel, full-history walk-forward, identical geometry to production),
    and every edge verdict is a paired per-game logloss difference between
    two stored loss vectors — the same bar, floor, and guards as the
    sequential engine. Reference for "gain" is the BASELINE state (∅,∅),
    not a moving champion: the grid exists to answer joint questions
    ("remove R1+R2 together?", "does A1 only help once R1 is gone?") that
    1-by-1 trials cannot. Per crossed cell the trace records an interaction
    statistic: joint_gain − (part_a_gain + part_b_gain) — negative = the
    features are substitutes sharing one signal, positive = they help each
    other.

    Windowing: if 2^adds × 2^removes exceeds RFE_GRID_MAX_STATES (env
    RFE_GRID_MAX_STATES, default 16), the lists are silently round-robined
    down to fit (A1, R1, A2, R2, ...); the dropped tail is recorded and each
    overflow name still receives its normal 1-by-1 targeted trial.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    universe = list(MONEYLINE_FEATURE_COLS)
    candidates = list(CANDIDATE_COLS)
    pool = list(KNOWN_FEATURE_COLS)
    forced_add = [c for c in forced_additions if c in candidates]
    forced_rm = [c for c in forced_removals if c in universe]
    unresolved = {
        "addition": [{"name": n, "nearest": _nearest_names(n)}
                     for n in forced_additions if n not in candidates],
        "removal": [{"name": n, "nearest": _nearest_names(n)}
                    for n in forced_removals if n not in universe],
    }
    if not forced_add and not forced_rm:
        raise RuntimeError(
            "Grid RFE requested but no forced name resolved against the "
            f"pool (candidates={len(candidates)}, universe={len(universe)}). "
            "Nearest-name suggestions were logged; check the workbook's "
            "Candidates sheet for exact names.")
    if unresolved["addition"] or unresolved["removal"]:
        # Grid lattices are built from exact names — a typo must never
        # silently shrink the experiment (strict, unlike targeted mode).
        raise RuntimeError(
            "Grid RFE requested with unresolvable forced names — refusing to "
            f"run a shrunken lattice: {unresolved}")

    # Cap resolution: config default 16; a REAL env value overrides it.
    # Empty/"0"/"null"/garbage env values fall back to the default rather
    # than zeroing the cap (a 0 cap would abort every grid).
    raw_cap = (os.environ.get(ENV_GRID_MAX_STATES) or "").strip()
    try:
        max_states = int(raw_cap) if raw_cap else int(RFE_GRID_MAX_STATES)
    except ValueError:
        logger.warning("RFE_GRID_MAX_STATES=%r is not an integer — "
                       "using the default %d", raw_cap, RFE_GRID_MAX_STATES)
        max_states = int(RFE_GRID_MAX_STATES)
    if max_states <= 0:  # "0"-style env values fall back to the default
        max_states = int(RFE_GRID_MAX_STATES)
    max_states = max(2, max_states)  # a grid is at least baseline + one state

    enriched, splits = _splits_and_frame(games, max_eval_folds=max_eval_folds)
    if not splits:
        raise RuntimeError("No walk-forward folds for the supplied frame")
    full_depth = int(max_eval_folds) == 0
    folds_available = len(splits)

    # Importance prior for canonical state ordering + windowing decisions.
    prior = _importance_prior(splits)
    win_add, win_rm, overflow = _grid_window(forced_add, forced_rm,
                                             max_states, prior)
    lattice_states = 2 ** len(win_add) * 2 ** len(win_rm)
    if lattice_states > max_states:  # defensive; windowing should prevent this
        raise RuntimeError(
            f"Grid lattice ({lattice_states} states) exceeds the state cap "
            f"({max_states}) — refusing to run an oversized grid")

    # Lattice: EVERY subset of the windowed adds × EVERY subset of the
    # windowed removes (2^a × 2^r states — 2+2 is exactly the 16-state grid
    # the default cap is sized for). Canonical order: fewer changes first,
    # then importance rank (desc) then name within a count class — two runs
    # with identical inputs build byte-identical state lists.
    from itertools import combinations

    def _ranked(subset: list[str]) -> list[str]:
        return sorted(subset, key=lambda x: (-prior.get(x, 0.0), x))

    def _all_subsets(items: list[str]) -> list[tuple[str, ...]]:
        out: list[tuple[str, ...]] = []
        for k in range(len(items) + 1):
            out.extend(combinations(items, k))
        return out

    add_subsets = _all_subsets(_ranked(win_add))
    rm_subsets = _all_subsets(_ranked(win_rm))
    states: list[dict[str, Any]] = []
    for a_cols in add_subsets:
        for r_cols in rm_subsets:
            cols = [c for c in universe if c not in r_cols] + list(a_cols)
            states.append({
                "id": f"S{len(states) + 1:02d}",
                "adds_applied": list(a_cols),
                "removes_applied": list(r_cols),
                "cols": cols,
                "n_features": len(cols),
            })
    n_states = len(states)

    logger.info("Grid RFE: %d states (%d adds x %d removes window, cap %d, "
                "overflow %d) on %d folds",
                n_states, len(win_add), len(win_rm), max_states,
                len(overflow), folds_available)

    # ── Score every state once, in parallel ────────────────────────────────
    # The adopted incumbent subset (if any) joins the pool as an extra task:
    # the same governance re-score a normal run performs, now free-riding on
    # the same worker fan-out.
    INCUMBENT_KEY = ("__incumbent__",)
    tasks: list[tuple[tuple, list[str], str]] = [
        ((tuple(st["adds_applied"]), tuple(st["removes_applied"])),
         st["cols"], st["id"])
        for st in states]
    valid_inc: Optional[list[str]] = None
    if incumbent_cols:
        cand_inc = [c for c in incumbent_cols if c in pool]
        if (cand_inc and len(cand_inc) == len(incumbent_cols)
                and len(cand_inc) >= floor):
            valid_inc = cand_inc
            tasks.append((INCUMBENT_KEY, valid_inc, "__incumbent__"))
    losses_by_key: dict[tuple, Optional[np.ndarray]] = {}
    metrics_by_key: dict[tuple, Optional[dict[str, Any]]] = {}
    per_fold_by_key: dict[tuple, list] = {}
    errors_by_key: dict[tuple, str] = {}
    workers = max_workers or max(1, min(len(tasks), (os.cpu_count() or 2) - 1))
    # Pin BLAS/LightGBM threads to 1 in the PARENT environment so every
    # spawned worker inherits it (their LightGBM imports initialize OpenMP
    # before the worker initializer runs). Saved and restored so a pipeline
    # caller's later phases keep their normal threading.
    _thread_vars = ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    _saved_threads = {v: os.environ.get(v) for v in _thread_vars}
    for v in _thread_vars:
        os.environ[v] = "1"
    try:
        with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_grid_worker_init,
                initargs=(enriched, splits)) as ex:
            futs = {ex.submit(_grid_worker_score, cols): (key, label)
                    for key, cols, label in tasks}
            for fut in as_completed(futs):
                key, label = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:  # worker died (OOM, native crash)
                    res = {"ok": False, "error": f"worker failed: {exc}"}
                if res.get("ok"):
                    losses_by_key[key] = res["losses"]
                    metrics_by_key[key] = res["metrics"]
                    per_fold_by_key[key] = res["per_fold"]
                else:
                    losses_by_key[key] = None
                    metrics_by_key[key] = None
                    per_fold_by_key[key] = []
                    errors_by_key[key] = res.get("error", "unknown")
                    logger.warning("Grid task %s failed: %s", label,
                                   errors_by_key[key])
    finally:
        for v, old in _saved_threads.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old

    base_key = ((), ())
    base_metrics = metrics_by_key.get(base_key)
    base_losses = losses_by_key.get(base_key)
    if base_metrics is None:
        raise RuntimeError(
            "Grid baseline state (∅,∅) failed to score — cannot anchor the "
            f"lattice ({errors_by_key.get(base_key, 'unknown')})")
    baseline_ll = float(base_metrics["logloss"])
    fallback_threshold = max(float(min_logloss_gain),
                             noise_sigma * float(base_metrics.get("logloss_se", 0.0)))

    # ── Incumbent health: re-score the adopted subset against the universe ─
    # (governance eval — same paired bar as a normal run's incumbent check;
    # the grid cannot silently strand an adopted subset it outperformed.)
    incumbent_check: Optional[dict[str, Any]] = None
    if incumbent_cols:
        if valid_inc is None:
            incumbent_check = {"note": "adopted subset invalid vs pool/floor",
                               "n_valid": len([c for c in incumbent_cols
                                               if c in pool])}
        elif metrics_by_key.get(INCUMBENT_KEY) is not None:
            im = metrics_by_key[INCUMBENT_KEY]
            il = losses_by_key[INCUMBENT_KEY]
            if (il is not None and base_losses is not None
                    and len(il) == len(base_losses)):
                inc_diff = il - base_losses
                inc_se = float(np.std(inc_diff, ddof=1) / np.sqrt(len(inc_diff)))
                inc_threshold = max(float(min_logloss_gain), noise_sigma * inc_se)
            else:
                inc_threshold = fallback_threshold
            beats = float(im["logloss"]) <= baseline_ll - inc_threshold
            incumbent_check = {
                "metrics": {k: v for k, v in im.items()},
                "bar_basis": "paired_diff_2sigma",
                "beats_universe": bool(beats),
                "note": ("adopted subset still earns its keep" if beats
                         else "regressed vs universe — consider --reset"),
            }
        else:
            incumbent_check = {"error": errors_by_key.get(INCUMBENT_KEY, "unknown"),
                               "note": "incumbent re-score failed"}

    # ── Fill state records (gains are vs the BASELINE state) ───────────────
    for st in states:
        key = (tuple(st["adds_applied"]), tuple(st["removes_applied"]))
        m = metrics_by_key.get(key)
        st["metrics"] = m
        st["per_fold"] = per_fold_by_key.get(key, [])
        if m is None:
            st["error"] = errors_by_key.get(key, "unknown")
            st["logloss_gain"] = None
            continue
        st["logloss_gain"] = round(baseline_ll - float(m["logloss"]), 4)

    def _gain(a: tuple[str, ...], r: tuple[str, ...]) -> Optional[float]:
        m = metrics_by_key.get((a, r))
        return None if m is None else round(baseline_ll - float(m["logloss"]), 4)

    # ── Edge verdicts. Two families, both judged with the standard bar:
    #    (1) DIRECT vs baseline for every state — the grid's primary
    #        question ("does this whole list beat production?") and the
    #        basis of commit/winner selection;
    #    (2) CHAINED parent→child edges along the add-chain (fixed removal
    #        column) and removal-chain (fixed add row) — the 1-by-1 marginal
    #        view ("does A1 help ON TOP of B−R1?").
    def _edge_kind(a_delta: int, r_delta: int) -> str:
        if a_delta == 1 and r_delta == 0:
            return "add"
        if a_delta == 0 and r_delta == 1:
            return "remove"
        return "joint"

    edges: list[dict[str, Any]] = []
    direct_committed: dict[str, bool] = {}
    for st in states:
        key = (tuple(st["adds_applied"]), tuple(st["removes_applied"]))
        if key == ((), ()):
            continue  # baseline is the reference, never an edge
        edge = _grid_edge_verdict(
            "baseline", _state_label(key),
            base_losses, losses_by_key.get(key),
            base_metrics, metrics_by_key.get(key),
            min_logloss_gain, noise_sigma, auc_guard, ece_guard,
            fallback_threshold)
        edge["kind"] = _edge_kind(len(key[0]), len(key[1]))
        edge["reference"] = "baseline"
        edges.append(edge)
        direct_committed[_state_label(key)] = bool(edge["committed"])
    chained: list[dict[str, Any]] = []
    # Hasse (single-feature-delta) edges: every state differs from each of
    # its one-feature-lighter parents by exactly one add or one remove —
    # the 1-by-1 marginal view ("does A1 help ON TOP of B−R1?") for every
    # context the lattice contains.
    state_key_set = {(tuple(st["adds_applied"]),
                      tuple(st["removes_applied"])) for st in states}
    for st in states:
        a_key = tuple(st["adds_applied"])
        r_key = tuple(st["removes_applied"])
        if not a_key and not r_key:
            continue
        child = (a_key, r_key)
        for f in a_key:  # parent without this add
            p_key = (tuple(x for x in a_key if x != f), r_key)
            if p_key not in state_key_set:
                continue
            e = _grid_edge_verdict(
                _state_label(p_key), _state_label(child),
                losses_by_key.get(p_key), losses_by_key.get(child),
                metrics_by_key.get(p_key), metrics_by_key.get(child),
                min_logloss_gain, noise_sigma, auc_guard, ece_guard,
                fallback_threshold)
            e["kind"] = "add"
            e["feature"] = f
            e["reference"] = "parent_state"
            chained.append(e)
        for f in r_key:  # parent without this removal
            p_key = (a_key, tuple(x for x in r_key if x != f))
            if p_key not in state_key_set:
                continue
            e = _grid_edge_verdict(
                _state_label(p_key), _state_label(child),
                losses_by_key.get(p_key), losses_by_key.get(child),
                metrics_by_key.get(p_key), metrics_by_key.get(child),
                min_logloss_gain, noise_sigma, auc_guard, ece_guard,
                fallback_threshold)
            e["kind"] = "remove"
            e["feature"] = f
            e["reference"] = "parent_state"
            chained.append(e)
    # Dedupe by (from, to): the direct (baseline-referenced) edge wins.
    seen_edges: set[tuple[str, str]] = {(e["from"], e["to"]) for e in edges}
    for e in chained:
        k = (e["from"], e["to"])
        if k not in seen_edges:
            seen_edges.add(k)
            edges.append(e)

    # ── Interaction statistics for every crossed cell (both sides nonempty) ─
    # joint(A,R) − gain(A) − gain(R): covers every subset pair, including the
    # joint-adds cell (A={A1,A2}) and the joint-removals cell (R={R1,R2}).
    interactions: list[dict[str, Any]] = []
    for a_sub in add_subsets:
        if not a_sub:
            continue
        for r_sub in rm_subsets:
            if not r_sub:
                continue
            ga = _gain(a_sub, ())
            gr = _gain((), r_sub)
            gj = _gain(a_sub, r_sub)
            if None in (ga, gr, gj):
                continue
            joint_gain = round(gj - (ga + gr), 4)
            interactions.append({
                "adds": list(a_sub),
                "removes": list(r_sub),
                "gain_add_only": ga,
                "gain_remove_only": gr,
                "gain_joint": gj,
                "interaction_vs_parts": joint_gain,
                "reading": ("substitutes — they share one signal (joint ≈ parts)"
                            if abs(joint_gain) < 0.0005 else
                            "synergy — they help each other (joint > parts)"
                            if joint_gain > 0 else
                            "interference — together they overfit (joint < parts)"),
            })

    # ── Winner: best state whose DIRECT-vs-baseline edge commits under the
    #    STANDARD bar — identical math to a normal run, no extra knob ────────
    winner: Optional[dict[str, Any]] = None
    committed_states = [
        st for st in states
        if st["metrics"] is not None
        and direct_committed.get(_state_label(
            (tuple(st["adds_applied"]), tuple(st["removes_applied"]))))
    ]
    if committed_states:
        w = min(committed_states, key=lambda s: float(s["metrics"]["logloss"]))
        winner = {"state_id": w["id"],
                  "adds_applied": w["adds_applied"],
                  "removes_applied": w["removes_applied"],
                  "cols": w["cols"],
                  "n_features": w["n_features"],
                  "logloss": float(w["metrics"]["logloss"]),
                  "logloss_gain": w["logloss_gain"],
                  "note": "best grid state clearing the standard noise bar "
                          "vs baseline — adoption still requires alt-geometry "
                          "confirmation + slate coverage via --adopt"}

    # ── Overflow names: honored with normal 1-by-1 targeted trials ─────────
    overflow_trials: list[dict[str, Any]] = []
    if overflow:
        try:
            for f in overflow:
                if f in win_add or f in win_rm:
                    continue
                kind = "add" if f in forced_add else "remove"
                trial = (list(universe) + [f]) if kind == "add" \
                    else [c for c in universe if c != f]
                set_feature_subset(trial)
                try:
                    m, t_losses = _score_splits(splits, return_losses=True)
                except Exception as exc:
                    logger.warning("Overflow trial failed (%s %s): %s",
                                   kind, f, exc)
                    m, t_losses = None, None
                rec: dict[str, Any] = {"feature": f, "action": kind,
                                       "n_features": len(trial),
                                       "metrics": None, "committed": False}
                if m is not None:
                    rec["metrics"] = {k: v for k, v in m.items()
                                      if k != "per_fold"}
                    rec["logloss_gain"] = round(baseline_ll - float(m["logloss"]), 4)
                    edge = _grid_edge_verdict(
                        "baseline", f"overflow:{kind}:{f}",
                        base_losses, t_losses, base_metrics, rec["metrics"],
                        min_logloss_gain, noise_sigma, auc_guard, ece_guard,
                        fallback_threshold)
                    rec["committed"] = edge["committed"]
                    rec["commit_threshold"] = edge["commit_threshold"]
                    rec["paired_se"] = edge.get("paired_se")
                overflow_trials.append(rec)
        finally:
            reset_feature_subset()

    reset_feature_subset()

    selected = (winner["cols"] if winner
                else list(universe))
    return {
        "baseline_metrics": {k: v for k, v in base_metrics.items()},
        "best_metrics": (dict(metrics_by_key[(tuple(winner["adds_applied"]),
                                              tuple(winner["removes_applied"]))])
                         if winner else {k: v for k, v in base_metrics.items()}),
        "selected_cols": selected,
        "n_selected": len(selected),
        "n_universe": len(universe),
        "n_candidates": len(candidates),
        "n_pool": len(pool),
        "steps": [],  # no sequential steps in grid mode
        "grid": {
            "max_states": max_states,
            "window": {"adds": win_add, "removes": win_rm},
            "grid_truncated": overflow,
            "n_states": n_states,
            "states": [
                {**{k: st[k] for k in ("id", "adds_applied", "removes_applied",
                                       "cols", "n_features", "metrics",
                                       "per_fold", "logloss_gain")},
                 **({"error": st["error"]} if "error" in st else {})}
                for st in states],
            "edges": edges,
            "interactions": interactions,
            "winner": winner,
            "overflow_trials": overflow_trials,
            "workers": workers,
        },
        "verdicts": {},
        "prior_verdicts": {},
        "n_prior_carried": 0,
        "forced_lists": {
            "addition_resolved": forced_add,
            "removal_resolved": forced_rm,
            "unresolved": unresolved,
        },
        "importance_prior_top": dict(
            sorted(prior.items(), key=lambda kv: -kv[1])[:25]),
        "redundancy_pairs": [],
        "floor": floor,
        "max_steps": 0,
        "max_eval_folds": max_eval_folds,
        "full_depth": full_depth,
        "folds_available": folds_available,
        "retrial": False,
        "targeted": True,
        "grid_mode": True,
        "run_mode": ("targeted_grid_full_history" if full_depth
                     else "targeted_grid_bounded"),
        "targeted_trials": {"additions": list(forced_add),
                            "removals": list(forced_rm)},
        "guards": {"auc_drop_max": auc_guard, "ece_rise_max": ece_guard},
        "bar_basis": "paired_diff_2sigma",
        "commit_threshold": round(fallback_threshold, 4),
        "incumbent_check": incumbent_check,
    }


def _state_label(key: tuple[tuple[str, ...], tuple[str, ...]]) -> str:
    """Human-readable grid state label: adds/removes applied."""
    a, r = key
    parts = []
    if a:
        parts.append("+" + "+".join(a))
    if r:
        parts.append("−" + "−".join(r))
    return "baseline" if not parts else " ".join(parts)


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
            m, ls = _score_splits(splits, return_losses=True)
            out[label] = {k: v for k, v in m.items() if k != "per_fold"}
            out[label + "_losses"] = ls
        u_losses = out.pop("universe_losses")
        c_losses = out.pop("candidate_losses")
        # Paired bar (same alternate folds scored twice) with a legacy
        # baseline-SE fallback on fold-set mismatch.
        if (u_losses is not None and c_losses is not None
                and len(u_losses) == len(c_losses)):
            diff = c_losses - u_losses
            paired_se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
            threshold = max(float(min_logloss_gain), noise_sigma * paired_se)
            bar_basis = "paired_diff_2sigma"
        else:
            threshold = max(float(min_logloss_gain),
                            noise_sigma * float(out["universe"].get("logloss_se", 0.0)))
            bar_basis = "baseline_se_fallback"
        holds = (
            out["candidate"]["logloss"]
                <= out["universe"]["logloss"] - threshold
            and float(out["universe"]["auc"]) - float(out["candidate"]["auc"]) <= auc_guard
            and float(out["candidate"]["ece"]) - float(out["universe"]["ece"]) <= ece_guard
        )
        return {"holds": bool(holds), "threshold": round(threshold, 4),
                "bar_basis": bar_basis,
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


def _targeted_suffix() -> str:
    """``_targeted_<HHMM>`` (ET; UTC fallback when zoneinfo is unavailable)
    — unique per run so same-day targeted runs never overwrite each other,
    and never colliding with the authoritative per-day full-trace name."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now = datetime.utcnow()
    return f"_targeted_{now.strftime('%H%M')}"


def _trace_path(day: date, suffix: str = "") -> Path:
    """Trace record path.

    - Full runs keep the authoritative per-day name
      (``mlb_feature_selection_<date>.json``): a same-day full rerun
      overwrites by design (it recomputes the whole search; git history
      retains the prior version).
    - Targeted runs get ``_targeted_<HHMM>`` (ET) — unique per run, so two
      targeted runs on the same day never overwrite each other, and a
      targeted run can never overwrite the full trace.

    Retention: traces are dated ``mlb_`` records on the 10-day window; the
    prior-verdict memory degrades to a fresh search if all traces age out,
    and adoption reads the never-deleted state file, never a trace."""
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
        "bar_basis": result.get("bar_basis"),
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
        "grid": result.get("grid"),
        "verdicts": result.get("verdicts"),
        "prior_verdicts_carried": result.get("n_prior_carried", 0),
        "incumbent_check": result.get("incumbent_check"),
        "adopted": bool(adopted),
    }
    suffix = _targeted_suffix() if result.get("targeted") else ""
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
    # GRID MODE: all subset combinations of the forced lists, judged against
    # the baseline with the standard bar. The flag alone (both lists empty)
    # silently falls through to normal mode — never an error.
    grid_mode = _truthy_env(ENV_GRID_MODE)
    if grid_mode and not (forced_add or forced_rm):
        grid_mode = False
    if grid_mode:
        state = load_state()
        result = run_grid_rfe(
            games, forced_add, forced_rm,
            incumbent_cols=(state or {}).get("cols"),
        )
        trace = write_trace(day, result, adopted=False, invoked_by="pipeline")
        grid = result.get("grid", {})
        n_committed = sum(1 for e in grid.get("edges", []) if e.get("committed")) \
            + sum(1 for t in grid.get("overflow_trials", []) if t.get("committed"))
        return {
            "ran": True,
            "trace": str(trace),
            "run_mode": result["run_mode"],
            "grid_mode": True,
            "n_states": grid.get("n_states", 0),
            "n_edges": len(grid.get("edges", [])),
            "winner": (grid.get("winner") or {}).get("state_id"),
            "n_pool": result["n_pool"],
            "n_universe": result["n_universe"],
            "n_selected": result["n_selected"],
            "baseline_logloss": float(result["baseline_metrics"]["logloss"]),
            "best_logloss": float(result["best_metrics"]["logloss"]),
            "n_trials": 0,
            "n_committed": n_committed,
            "incumbent_note": (result.get("incumbent_check") or {}).get("note"),
            "forced_unresolved": None,
        }
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
    grid_mode = _truthy_env(ENV_GRID_MODE)
    if grid_mode and not (forced_add or forced_rm):
        print("grid mode requested but both forced lists are empty — "
              "running normal mode")
        grid_mode = False
    if grid_mode:
        print("GRID MODE — scoring every subset combination of the forced "
              "lists in parallel (judged vs baseline, standard noise bar)")
    prior = {} if retrial else load_prior_verdicts(day)
    state = load_state()

    if grid_mode:
        result = run_grid_rfe(
            games, forced_add, forced_rm, max_eval_folds=args.max_eval_folds,
            ece_guard=args.ece_guard,
            incumbent_cols=(state or {}).get("cols"), floor=args.floor,
        )
    else:
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
    if result.get("grid_mode"):
        grid = result.get("grid", {})
        for it in grid.get("interactions", []):
            logger.info("Grid interaction [%s | %s]: %+.4f (%s)",
                        "+".join(it["adds"]), "-".join(it["removes"]),
                        it["interaction_vs_parts"], it["reading"])
        w = grid.get("winner")
        if w:
            logger.info("Grid winner: %s (%s) logloss %.4f (gain %+.4f)",
                        w["state_id"], _state_label(
                            (tuple(w["adds_applied"]),
                             tuple(w["removes_applied"]))),
                        w["logloss"], w["logloss_gain"])
        else:
            logger.info("Grid winner: none — no state cleared the standard "
                        "noise bar vs baseline")

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
