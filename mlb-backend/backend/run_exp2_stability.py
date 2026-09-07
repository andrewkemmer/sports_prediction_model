"""Experiment #2 stability pass — fold-level ΔAUC, sign counts, fold-cluster
bootstrap CI, and P(ΔAUC <= 0) for the pre-specified promising candidates,
on BOTH targets. Read-only for production; writes
data_delivery/exp2_stability_<date>.json.

Re-runs the needed arms capturing per-fold OOF predictions (the main
registry stores only pooled metrics). Same fold geometry, same trainer,
same prequential calibration as run_exp2_feature_test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_exp2_feature_test import (CANDIDATES, CANDIDATE_S, DATA, DATE_TAG,
                                   S_FAMILIES, _worker_init, _worker_task,
                                   add_candidates, arm_metrics, base_cols_of,
                                   build_frame, eval_arm_ml)
from training import compute_metrics

OUT = DATA / f"exp2_stability_{DATE_TAG}.json"

# Pre-specified stability set: the candidates that cleared the expansion
# screen (positive pooled ΔAUC on either target) + their removal pairs.
STABILITY_CANDS = [
    "exp2_cat_k_breaking_diff",     # best ML, positive RL
    "exp2_cat_k_fastball_diff",     # positive both (small)
    "exp2_centered_k_diff",         # positive both (small)
    "exp2_cat_platoon_k_fastball_diff",  # ML positive, RL negative
    "exp2_cat_xwoba_fastball_diff",  # ML positive, RL negative
]
TARGETS = ["home_win", "rl_cover"]


def fold_delta_table(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                     fold_ids: np.ndarray, n_boot: int = 2000,
                     seed: int = 42) -> dict:
    """Per-fold ΔAUC (a − b), sign counts, fold-cluster bootstrap CI, and
    P(ΔAUC <= 0) from a normal approximation over fold-mean deltas."""
    from sklearn.metrics import roc_auc_score
    folds = np.unique(fold_ids)
    d = []
    for f in folds:
        m = fold_ids == f
        if len(np.unique(y[m])) < 2:
            continue
        d.append(roc_auc_score(y[m], p_a[m]) - roc_auc_score(y[m], p_b[m]))
    d = np.array(d)
    rng = np.random.default_rng(seed)
    boots = [float(d[rng.integers(0, len(d), len(d))].mean())
             for _ in range(n_boot)]
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    mu, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    from math import erf, sqrt
    p_le0 = 0.5 * (1 + erf((0 - mu) / (se * sqrt(2)))) if se > 0 else 0.5
    return {
        "folds": int(len(d)),
        "mean_d_auc": round(mu, 5),
        "folds_positive": int((d > 0).sum()),
        "folds_negative": int((d < 0).sum()),
        "ci95_d_auc": [round(ci[0], 5), round(ci[1], 5)],
        "p_delta_le_0": round(float(p_le0), 4),
    }


def main() -> None:
    tune_ml, tune_rl, folds_by_target, _cov = build_frame()
    base_cols = base_cols_of()
    fam_cols = {fam: [c for c in base_cols if c not in sfx]
                for fam, sfx in S_FAMILIES.items()}

    # arms to (re)run with per-fold capture: F per target + each candidate's
    # expansion + its targeted-removal pair.
    wanted: dict[str, tuple[list[str], str]] = {}
    for t in TARGETS:
        wanted[f"{t}:F"] = (base_cols, t)
    for cand in STABILITY_CANDS:
        fam = CANDIDATE_S[cand]
        for t in TARGETS:
            wanted[f"{t}:F+{cand}"] = (base_cols + [cand], t)
            wanted[f"{t}:F-{fam}+{cand}"] = (fam_cols[fam] + [cand], t)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    # Resumable: per-arm checkpoints survive timeout kills across invocations.
    ckpt_path = DATA / f"exp2_stability_ckpt_{DATE_TAG}.json"
    fold_store: dict[str, dict] = {}
    if ckpt_path.exists():
        fold_store = json.loads(ckpt_path.read_text())
        print(f"resuming with {len(fold_store)} checkpointed arms", flush=True)
    pending = {k: v for k, v in wanted.items() if k not in fold_store}
    with ProcessPoolExecutor(max_workers=4, initializer=_worker_init,
                             initargs=({"home_win": tune_ml,
                                        "rl_cover": tune_rl},
                                       folds_by_target)) as ex:
        futs = {ex.submit(_worker_task, (k, k, cols, t)): k
                for k, (cols, t) in pending.items()}
        for fut in as_completed(futs):
            key, _n, _c, _t, res, rt = fut.result()
            fold_store[key] = {
                "y": res["y"].tolist(), "p": res["p"].tolist(),
                "fold_dates": res["fold_dates"],
                "fold_aucs": res["fold_aucs"].tolist(),
            }
            ckpt_path.write_text(json.dumps(fold_store))
            print(f"  [{rt:.0f}s] {key}", flush=True)

    out: dict = {"date_tag": DATE_TAG, "candidates": {}}
    # Per-game fold ids: expand the per-fold val_end dates over each fold's
    # validation games, replicating eval_arm_ml's fold-skip logic exactly.
    from run_exp2_feature_test import MIN_VAL_FOLD_GAMES

    def expand_fold_ids(target: str, stored_dates: list[str]) -> np.ndarray:
        ids, di = [], 0
        for split in folds_by_target[target]:
            train, val = split["train_games"], split["val_games"]
            if len(train) < 10 or len(val) < 5:
                continue
            if len(val) < MIN_VAL_FOLD_GAMES and not split.get("is_partial_tail"):
                continue
            n = len(val)
            assert di < len(stored_dates), "stored fold dates exhausted"
            ids.extend([stored_dates[di]] * n)
            di += 1
        assert di == len(stored_dates), "stored fold dates unconsumed"
        return np.array(ids)

    for t in TARGETS:
        fdF = fold_store[f"{t}:F"]["fold_dates"]
        fold_ids_F = expand_fold_ids(t, fdF)
        yF = np.array(fold_store[f"{t}:F"]["y"])
        pF = np.array(fold_store[f"{t}:F"]["p"])
        assert len(yF) == len(fold_ids_F), "fold-id expansion mismatch"
        for cand in STABILITY_CANDS:
            key = f"{t}:F+{cand}"
            if key not in fold_store:
                continue
            pX = np.array(fold_store[key]["p"])
            assert fold_store[key]["fold_dates"] == fdF, "fold alignment drifted"
            assert len(pX) == len(yF), "row count mismatch vs baseline"
            fam = CANDIDATE_S[cand]
            krem = f"{t}:F-{fam}+{cand}"
            rec = {
                "expansion": fold_delta_table(yF, pX, pF, fold_ids_F),
                "metrics_expansion": compute_metrics(yF, pX),
                "metrics_base": compute_metrics(yF, pF),
            }
            if krem in fold_store:
                pR = np.array(fold_store[krem]["p"])
                rec["removal"] = fold_delta_table(yF, pR, pF, fold_ids_F)
                rec["metrics_removal"] = compute_metrics(yF, pR)
            out["candidates"].setdefault(cand, {})[t] = rec

    OUT.write_text(json.dumps(out, indent=1, default=str))
    ckpt_path.unlink(missing_ok=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
