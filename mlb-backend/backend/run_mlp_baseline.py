"""MLP baseline rerun on the committed game_level_features.csv.

v2 (refreshed snapshot): the daily Colab pipeline (2026-08-25,
MLB_FULL_REPULL) regenerated the data — 4,451 games with weather features
applied (was 4,441). The old baseline (baseline_mlp_6508bda10fb2.json) was
deleted by pipeline cleanup. This script re-locks the baseline on the new
snapshot:

  * Data     : committed data_delivery/game_level_features.csv; SHA-256 is
               recorded in the output.
  * Folds    : identical machinery to the tuner / production —
               walk_forward_splits(RETRAIN_CADENCE_DAYS) filtered by
               MIN_VAL_FOLD_GAMES. Whatever the engine produces is ACCEPTED
               (declared vs executed are both recorded); old fold counts are
               NOT forced. On the refreshed snapshot the engine declares 48
               and executes 44 (4 skipped for <40 val games).
  * Members  : DECISION (recorded in metadata) — the baseline is defined as
               logistic + MLP ONLY. xgb/lgbm/rf are skipped via an import
               patch; member OOF metrics come from each member's own
               probabilities and are independent of the other members'
               presence, and the reconciliation proved harness-vs-production
               parity under exactly this configuration.
  * Clip     : 1e-7 everywhere (matches training.compute_metrics).
  * Per fold : impute on train -> scale on train -> transform val. The
               harness path is checked bit-identical (max|dp| == 0.0) against
               the production path (train_moneyline_ensemble +
               ensemble_predict) for BOTH members.
  * Holdout  : the sealed 21-day tail is never touched during fitting; both
               configs (current MLP_PARAMS and the Optuna winner) are refit
               on ALL pre-holdout games and scored on the holdout last.

Acceptance gates (tolerance 2e-3) reference the OLD snapshot's numbers
(from the reconciled harness cross-check on 4,441 games / 44 folds):
  logistic pooled:  logloss ~0.7052,  auc ~0.5437
  mlp pooled:       logloss ~0.7988,  auc ~0.5296
  per-fold max|dp| vs harness == 0.0 (bit-identical)
  holdout:          current ~0.70283/0.5263, winner ~0.71069/0.5066
Numbers will shift slightly on the refreshed snapshot — drift is EXPECTED and
reported, never silently adjusted. A 6th gate checks the decision-relevant
outcome is preserved: holdout current beats winner (lower logloss, higher
AUC).

Emits data_delivery/baseline_mlp_<sha>.json and prints a PASS/FAIL gate list.
A failing gate is FLAGGED, never silently fixed.

Usage:
    python run_mlp_baseline.py            # full baseline
    python run_mlp_baseline.py --out /tmp/baseline.json
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

# --- Baseline definition: logistic + MLP only -------------------------------
# Skip the tree members via import interception (they import lazily inside
# train_moneyline_ensemble). Member OOF metrics are per-member, so this does
# not touch logistic/MLP numbers — proven by the reconciliation.
sys.modules["lightgbm"] = None
sys.modules["xgboost"] = None
import sklearn.ensemble  # noqa: E402


class _FastRF:
    """Cheap stand-in so RandomForestClassifier import succeeds but costs ~0."""

    def __init__(self, **kw):
        pass

    def fit(self, X, y):
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full((len(X), 2), 0.0)
        p[:, 0] = 1 - self.p_
        p[:, 1] = self.p_
        return p


sklearn.ensemble.RandomForestClassifier = _FastRF

from training import (  # noqa: E402
    walk_forward_splits,
    walk_forward_evaluate,
    train_moneyline_ensemble,
    ensemble_predict,
    _LAST_ENSEMBLE_INFO,
    _prepare_features,
    _impute_median,
)
from config import (  # noqa: E402
    DATA_DELIVERY_DIR,
    MIN_VAL_FOLD_GAMES,
    MLP_PARAMS,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
)
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import tune_mlp_optuna as tune  # noqa: E402

EPS = 1e-7  # production compute_metrics clip
GATE_TOL = 2e-3

# Optuna winner (study mlp_moneyline_v2, 75 trials, 2026-08-25, best pooled
# OOF 0.70992) — embedded so the baseline is reproducible without the
# ephemeral sqlite study file. Provenance: tune_mlp_optuna.py.
WINNER_PARAMS = {
    "hidden_layer_sizes": (32, 16),
    "alpha": 0.001092224832249551,
    "learning_rate": "adaptive",
    "learning_rate_init": 0.002257204643542392,
    "batch_size": 256,
    "max_iter": 600,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "activation": "tanh",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def head_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=_BACKEND_DIR.parent,
        ).stdout.strip()
        return out
    except Exception:
        return "unknown"


def harness_fold(s, member: str):
    """Tuner-path per-fold fit (impute train -> scale train -> transform val)."""
    X_tr, _, y_tr = _prepare_features(s["train_games"])
    X_va, _, y_va = _prepare_features(s["val_games"])
    X_tr_i, med = _impute_median(X_tr)
    X_va_i, _ = _impute_median(X_va, med)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_i)
    X_va_s = sc.transform(X_va_i)
    if member == "logistic":
        from training import _logistic_feature_indices
        idx = _logistic_feature_indices()
        m = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        m.fit(X_tr_s[:, idx], y_tr)
        return m.predict_proba(X_va_s[:, idx])[:, 1], y_va
    m = MLPClassifier(**MLP_PARAMS)
    m.fit(X_tr_s, y_tr)
    return m.predict_proba(X_va_s)[:, 1], y_va


def pooled_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(probs, EPS, 1 - EPS)
    return {
        "logloss": round(float(log_loss(y, p)), 6),
        "auc": round(float(roc_auc_score(y, p)), 6),
        "n": int(len(y)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSON path (default data_delivery/baseline_mlp_<sha>.json)")
    ap.add_argument("--holdout-days", type=int, default=21)
    args = ap.parse_args()

    sha = head_sha()
    data_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    data_hash = sha256_file(data_path)

    games = pd.read_csv(data_path)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_win"]).reset_index(drop=True)
    cutoff = games["game_date"].max() - pd.Timedelta(days=args.holdout_days - 1)
    tune_df = games[games["game_date"] < cutoff].reset_index(drop=True)
    hold_df = games[games["game_date"] >= cutoff].reset_index(drop=True)

    all_splits = walk_forward_splits(tune_df, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
    folds = [s for s in all_splits if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]
    n_declared, n_executed = len(all_splits), len(folds)
    # Whatever the engine produces is accepted — old fold counts are NOT forced.
    print(f"commit={sha[:12]}  data_sha={data_hash[:12]}  games={len(games)}  "
          f"tuning={len(tune_df)}  holdout={len(hold_df)}  "
          f"folds declared={n_declared} executed={n_executed}")

    # ---------- per-fold bit-identical check + pooled (harness vs prod) -----
    per_fold = []
    pooled = {"logistic": {"harness": [], "prod": []},
              "mlp": {"harness": [], "prod": []}}
    y_all, log_h, log_p = [], [], []
    mlp_h, mlp_p = [], []
    winner_p = []
    for i, s in enumerate(folds):
        tr, va = s["train_games"], s["val_games"]
        y_va = va["home_win"].values.astype(float)
        y_all.append(y_va)
        # harness path
        p_h_log, _ = harness_fold(s, "logistic")
        p_h_mlp, _ = harness_fold(s, "mlp")
        log_h.append(p_h_log); mlp_h.append(p_h_mlp)
        # production path
        models, _ = train_moneyline_ensemble(tr, va)
        member_probs = ensemble_predict(models, va)[1]
        p_p_log = member_probs["logistic"]
        p_p_mlp = member_probs["mlp"]
        log_p.append(p_p_log); mlp_p.append(p_p_mlp)
        # winner (harness path) for the pooled-OOF record
        X_tr, _, y_tr = _prepare_features(tr)
        X_va, _, _ = _prepare_features(va)
        X_tr_i, med = _impute_median(X_tr)
        X_va_i, _ = _impute_median(X_va, med)
        sc = StandardScaler()
        mw = MLPClassifier(**dict(WINNER_PARAMS, random_state=RANDOM_SEED,
                                  early_stopping=True))
        mw.fit(sc.fit_transform(X_tr_i), y_tr)
        winner_p.append(mw.predict_proba(sc.transform(X_va_i))[:, 1])
        per_fold.append({
            "fold": i,
            "n": int(len(va)),
            "max_dp_logistic": float(np.max(np.abs(p_h_log - p_p_log))),
            "max_dp_mlp": float(np.max(np.abs(p_h_mlp - p_p_mlp))),
        })

    y_all = np.concatenate(y_all)
    pooled["logistic"]["harness"] = pooled_metrics(np.concatenate(log_h), y_all)
    pooled["logistic"]["prod"] = pooled_metrics(np.concatenate(log_p), y_all)
    pooled["mlp"]["harness"] = pooled_metrics(np.concatenate(mlp_h), y_all)
    pooled["mlp"]["prod"] = pooled_metrics(np.concatenate(mlp_p), y_all)
    winner_pooled = pooled_metrics(np.concatenate(winner_p), y_all)

    # ---------- canonical production numbers (walk_forward_evaluate) --------
    walk_forward_evaluate(tune_df)
    prod_info = {e["name"]: e for e in _LAST_ENSEMBLE_INFO}
    canonical = {
        name: {"logloss": prod_info[name]["logloss"], "auc": prod_info[name]["auc"],
               "n": prod_info[name]["n_eval"]}
        for name in ("logistic", "mlp")
    }

    # ---------- sealed holdout (LAST — never touched during fitting) --------
    X_refit, _, refit_y = _prepare_features(tune_df)
    X_hold, _, hold_y = _prepare_features(hold_df)
    X_refit_i, med = _impute_median(X_refit)
    X_hold_i, _ = _impute_median(X_hold, med)
    sc = StandardScaler()
    X_refit_s = sc.fit_transform(X_refit_i)
    X_hold_s = sc.transform(X_hold_i)
    holdout = {}
    for label, params in (("current", dict(MLP_PARAMS)),
                          ("winner", dict(WINNER_PARAMS, random_state=RANDOM_SEED,
                                          early_stopping=True))):
        m = MLPClassifier(**params)
        m.fit(X_refit_s, refit_y)
        p = np.clip(m.predict_proba(X_hold_s)[:, 1], EPS, 1 - EPS)
        holdout[label] = {
            "logloss": round(float(log_loss(hold_y, p)), 6),
            "auc": round(float(roc_auc_score(hold_y, p)), 6),
            "n": int(len(hold_y)),
        }

    # ---------- gates --------------------------------------------------------
    def gate(name, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return {"name": name, "pass": bool(ok), "detail": detail}

    gates = []
    gates.append(gate("logistic pooled ~0.7052/0.5437",
                      abs(canonical["logistic"]["logloss"] - 0.7052) < GATE_TOL
                      and abs(canonical["logistic"]["auc"] - 0.5437) < GATE_TOL,
                      f"logloss={canonical['logistic']['logloss']} "
                      f"auc={canonical['logistic']['auc']}"))
    gates.append(gate("mlp pooled ~0.7988/0.5296",
                      abs(canonical["mlp"]["logloss"] - 0.7988) < GATE_TOL
                      and abs(canonical["mlp"]["auc"] - 0.5296) < GATE_TOL,
                      f"logloss={canonical['mlp']['logloss']} "
                      f"auc={canonical['mlp']['auc']}"))
    max_dp_log = max(f["max_dp_logistic"] for f in per_fold)
    max_dp_mlp = max(f["max_dp_mlp"] for f in per_fold)
    gates.append(gate("per-fold max|dp| == 0.0 (bit-identical)",
                      max_dp_log == 0.0 and max_dp_mlp == 0.0,
                      f"logistic={max_dp_log:.3e} mlp={max_dp_mlp:.3e}"))
    gates.append(gate("holdout current ~0.70283/0.5263",
                      abs(holdout["current"]["logloss"] - 0.70283) < GATE_TOL
                      and abs(holdout["current"]["auc"] - 0.5263) < GATE_TOL,
                      f"logloss={holdout['current']['logloss']} "
                      f"auc={holdout['current']['auc']}"))
    gates.append(gate("holdout winner ~0.71069/0.5066",
                      abs(holdout["winner"]["logloss"] - 0.71069) < GATE_TOL
                      and abs(holdout["winner"]["auc"] - 0.5066) < GATE_TOL,
                      f"logloss={holdout['winner']['logloss']} "
                      f"auc={holdout['winner']['auc']}"))
    hc, hw = holdout["current"], holdout["winner"]
    gates.append(gate("holdout outcome preserved (current beats winner)",
                      hc["logloss"] < hw["logloss"] and hc["auc"] > hw["auc"],
                      f"current {hc['logloss']}/{hc['auc']} vs "
                      f"winner {hw['logloss']}/{hw['auc']}"))

    def b64(a: np.ndarray) -> str:
        return base64.b64encode(np.asarray(a, dtype=np.float64).tobytes()).decode()

    result = {
        "schema": "mlp-baseline/v2",
        "commit_sha": sha,
        "data_file": str(data_path.relative_to(_BACKEND_DIR.parent)),
        "data_sha256": data_hash,
        "n_folds_declared": n_declared,
        "n_folds_executed": n_executed,
        "reference_snapshot": ("baseline_mlp_6508bda10fb2.json (old 4,441-game "
                               "snapshot / 44 folds) — gate targets reference "
                               "those numbers; drift on the refreshed "
                               "4,451-game CSV is expected and reported"),
        "n_games": int(len(games)),
        "n_tuning": int(len(tune_df)),
        "n_holdout": int(len(hold_df)),
        "clip_eps": EPS,
        "seed": int(RANDOM_SEED),
        "retrain_cadence_days": int(RETRAIN_CADENCE_DAYS),
        "min_val_fold_games": int(MIN_VAL_FOLD_GAMES),
        "members": ["logistic", "mlp"],
        "members_decision": ("baseline defined as logistic+MLP only; "
                             "xgb/lgbm/rf skipped via import patch — member "
                             "OOF metrics are per-member and independent of "
                             "the other members' presence (proven by the "
                             "reconciliation)"),
        "fold_machinery": "impute on train -> scale on train -> transform val "
                          "(tune_mlp_optuna.prepare_fold / production "
                          "train_moneyline_ensemble)",
        "pooled": pooled,
        "canonical_walk_forward_evaluate": canonical,
        "winner_pooled_oof": winner_pooled,
        "winner_params": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in WINNER_PARAMS.items()},
        "holdout": holdout,
        "per_fold": per_fold,
        "pooled_probs_b64": {
            "y": b64(y_all),
            "logistic_harness": b64(np.concatenate(log_h)),
            "logistic_prod": b64(np.concatenate(log_p)),
            "mlp_harness": b64(np.concatenate(mlp_h)),
            "mlp_prod": b64(np.concatenate(mlp_p)),
            "mlp_winner": b64(np.concatenate(winner_p)),
        },
        "gates": gates,
        "all_gates_passed": all(g["pass"] for g in gates),
    }

    out = args.out or (DATA_DELIVERY_DIR / f"baseline_mlp_{sha[:12]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nbaseline written: {out}")
    print(f"ALL GATES: {'PASS' if result['all_gates_passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
