"""Standalone identity/Platt/isotonic calibration replay.

This module deliberately does not modify the production pipeline or
calibration math. It consumes a saved OOF CSV with columns ``game_date``,
``home_win``, and ``home_win_prob_model``. Optional ``fold``/``fold_idx``
columns preserve the original chronological fold boundaries; without them,
the harness uses chronological chunks and records that fallback explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression

from calibration import apply_platt, fit_platt

EPS = 1e-7
HOLDOUT_START = pd.Timestamp("2026-08-05")
HOLDOUT_END = pd.Timestamp("2026-08-25")
MIN_VALIDATION_SLICE = 300


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            total += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=float)
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = 0.5
    return {"logloss": float(log_loss(y, p)), "auc": auc,
            "brier": float(brier_score_loss(y, p)), "ece": _ece(y, p)}


def fit_isotonic(y: np.ndarray, p: np.ndarray) -> dict[str, Any] | None:
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return None
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(p[ok], y[ok])
    return {"model": model, "min": float(np.min(p[ok])), "max": float(np.max(p[ok]))}


def apply_isotonic(p: np.ndarray, fitted: dict[str, Any] | None) -> np.ndarray:
    if fitted is None:
        return np.asarray(p, dtype=float)
    clipped = np.clip(np.asarray(p, dtype=float), fitted["min"], fitted["max"])
    return np.asarray(fitted["model"].predict(clipped), dtype=float)


def _fit_candidate(name: str, y: np.ndarray, p: np.ndarray) -> Any:
    if name == "identity":
        return None
    if name == "platt":
        return fit_platt(y, p)
    if name == "isotonic":
        return fit_isotonic(y, p)
    raise ValueError(f"unknown candidate: {name}")


def _apply_candidate(name: str, p: np.ndarray, fitted: Any) -> np.ndarray:
    if name == "identity":
        return np.asarray(p, dtype=float)
    if name == "platt":
        return np.asarray(apply_platt(p, fitted), dtype=float)
    return apply_isotonic(p, fitted)


def candidate_names(n_prior: int) -> list[str]:
    if n_prior < 300:
        return ["identity"]
    if n_prior < 1000:
        return ["identity", "platt"]
    return ["identity", "platt", "isotonic"]


def conditional_replay(y: np.ndarray, p: np.ndarray, folds: list[np.ndarray]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    out = np.full(len(p), np.nan, dtype=float)
    decisions: list[dict[str, Any]] = []
    prior_indices: list[int] = []
    for fold_no, val_idx in enumerate(folds):
        pool = np.asarray(prior_indices, dtype=int)
        names = candidate_names(len(pool))
        if len(pool) < 300:
            winner = "identity"
        else:
            eval_idx = pool[-min(MIN_VALIDATION_SLICE, len(pool)):]
            train_idx = pool[:-len(eval_idx)] if len(pool) > len(eval_idx) else pool
            if len(train_idx) < 2 or len(np.unique(y[train_idx])) < 2:
                winner = "identity"
            else:
                scores = {}
                for name in names:
                    fitted = _fit_candidate(name, y[train_idx], p[train_idx])
                    scores[name] = metrics(y[eval_idx], _apply_candidate(name, p[eval_idx], fitted))["logloss"]
                winner = min(scores, key=scores.get)
            decisions.append({"fold": fold_no, "prior_n": len(pool), "candidates": names, "winner": winner})
        if winner == "identity":
            fitted = None
        else:
            fitted = _fit_candidate(winner, y[pool], p[pool])
        out[val_idx] = _apply_candidate(winner, p[val_idx], fitted)
        prior_indices.extend(val_idx.tolist())
    return out, decisions


def _folds(df: pd.DataFrame, tuning: pd.DataFrame) -> list[np.ndarray]:
    if "fold" in tuning.columns:
        groups = [g for _, g in tuning.groupby("fold", sort=True)]
    elif "fold_idx" in tuning.columns:
        groups = [g for _, g in tuning.groupby("fold_idx", sort=True)]
    else:
        groups = list(np.array_split(tuning.sort_values("game_date").index.to_numpy(), max(1, len(tuning) // 90)))
    return [g.to_numpy(dtype=int) if isinstance(g, pd.Index) else np.asarray(g, dtype=int) for g in groups if len(g)]


def replay(df: pd.DataFrame) -> dict[str, Any]:
    required = {"game_date", "home_win", "home_win_prob_model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OOF CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["home_win"] = pd.to_numeric(df["home_win"], errors="coerce")
    df["home_win_prob_model"] = pd.to_numeric(df["home_win_prob_model"], errors="coerce")
    df = df.dropna(subset=list(required)).sort_values("game_date").reset_index(drop=True)
    hold = df[df["game_date"].between(HOLDOUT_START, HOLDOUT_END)].copy()
    tuning = df[df["game_date"] < HOLDOUT_START].copy()
    if hold.empty:
        raise ValueError("sealed holdout is empty")
    y_t, p_t = tuning["home_win"].to_numpy(), tuning["home_win_prob_model"].to_numpy()
    y_h, p_h = hold["home_win"].to_numpy(), hold["home_win_prob_model"].to_numpy()
    folds = _folds(df, tuning)
    conditional, decisions = conditional_replay(y_t, p_t, folds)
    # Fit unconditional maps only on tuning OOF, never holdout.
    platt = _fit_candidate("platt", y_t, p_t)
    iso = _fit_candidate("isotonic", y_t, p_t)
    variants = {
        "identity": p_h,
        "unconditional_platt": _apply_candidate("platt", p_h, platt),
        "conditional": np.repeat(float(np.mean(y_t)), len(p_h)),
        "isotonic": _apply_candidate("isotonic", p_h, iso),
    }
    # Conditional replay is only defined for tuning folds; holdout receives
    # the final winner selected using prior tuning OOF and never holdout data.
    final_names = candidate_names(len(y_t))
    eval_idx = np.arange(max(0, len(y_t) - min(MIN_VALIDATION_SLICE, len(y_t))), len(y_t))
    if len(y_t) >= 300 and len(eval_idx) and len(np.unique(y_t[:eval_idx[0]])) > 1:
        scores = {}
        for name in final_names:
            fitted = _fit_candidate(name, y_t[:eval_idx[0]], p_t[:eval_idx[0]])
            scores[name] = metrics(y_t[eval_idx], _apply_candidate(name, p_t[eval_idx], fitted))["logloss"]
        final_winner = min(scores, key=scores.get)
    else:
        final_winner = "identity"
    final_fit = _fit_candidate(final_winner, y_t, p_t) if final_winner != "identity" else None
    variants["conditional"] = _apply_candidate(final_winner, p_h, final_fit)
    report = {name: metrics(y_h, pred) for name, pred in variants.items()}
    base = report["unconditional_platt"]
    cond = report["conditional"]
    adopt = cond["ece"] < base["ece"] and cond["logloss"] <= base["logloss"]
    return {"schema": "calibration-ablation/v1", "holdout_start": str(HOLDOUT_START.date()),
            "holdout_end": str(HOLDOUT_END.date()), "holdout_n": len(hold),
            "tuning_n": len(tuning), "fold_decisions": decisions,
            "final_winner": final_winner, "variants": report,
            "gate": {"verdict": "ADOPT" if adopt else "DON'T ADOPT",
                     "reason": "conditional improves ECE without logloss degradation" if adopt else "conditional did not clear both holdout gates"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--target-date", type=str, default=None,
                        help="Pipeline target date (YYYY-MM-DD); defaults to today")
    args = parser.parse_args()
    result = replay(pd.read_csv(args.csv))
    target = args.target_date or date.today().isoformat()
    compact = target.replace("-", "")
    out = args.out or args.csv.with_name(f"calibration_ablation_{compact}.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    for name, scores in result["variants"].items():
        print(f"{name}: logloss={scores['logloss']:.4f} auc={scores['auc']:.4f} ece={scores['ece']:.4f}")
    print(f"GATE: {result['gate']['verdict']}")


if __name__ == "__main__":
    main()
