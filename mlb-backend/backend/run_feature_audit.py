"""Feature keep-list audit — measure whether the 29 dropped features (24
matchup-gap _diff + 5 engineered composites) carry moneyline signal.

Measurement task (NOT tuning): nothing is changed silently. Reuses the
locked baseline harness exactly (refreshed 4,451-game CSV, 48 declared / 44
executed folds, clip 1e-7, RANDOM_SEED=42, sealed 21-day holdout fitted only
at the end).

Variants (column subsets of FEATURE_COLS):
  A    : the engine's current keep-list (29 kept — run-engine view)
  B    : A + all 24 dropped _diff features            (53 cols)
  C    : A + the 5 engineered composites              (34 cols)
  D    : A + 4 curated diffs (sp_xwoba_diff, woba_30g_diff,
         lineup_woba_mean_diff, elo_diff)             (33 cols)
  REF  : full FEATURE_COLS (58) — the LOCKED baseline control. Note: the
         locked baseline (baseline_mlp_d3fc4e913adf.json) trains the MLP on
         all 58 FEATURE_COLS; the 29/58 keep-list governs the RUN ENGINE
         (per-side Poisson), not the moneyline. REF reproduces the locked
         numbers; A is the run-engine view applied to the moneyline.

Member: MLP is the decision member (its full-matrix training reproduces the
locked baseline at REF; the baseline's holdout gate is MLP-only). Logistic is
reported as secondary on the variant subset (its separate 34-col raw/diff
routing is out of scope and would confound the comparison at subset views).

Gate (per the task): a variant is recommended only if it beats A on the
SEALED holdout on BOTH log loss and AUC. A pooled win with a holdout loss is
flagged as likely overfit, never adopted.

Collinearity: for each restored feature, max |corr| vs the 29 kept features
(median-imputed tuning pool); |corr| > 0.8 is flagged.

Emits data_delivery/feature_audit_<sha>.json (incremental — resumes by
skipping variants already present). COMMITS NOTHING — review first.

Usage:
    python run_feature_audit.py
    python run_feature_audit.py --variants A,B,REF
"""
from __future__ import annotations

import argparse
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

# Baseline definition: logistic + MLP only (xgb/lgbm/rf skipped; member
# metrics are per-member and independent of the other members' presence).
sys.modules["lightgbm"] = None
sys.modules["xgboost"] = None
import sklearn.ensemble  # noqa: E402


class _FastRF:
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
    FEATURE_COLS,
    walk_forward_splits,
    _prepare_features,
    _impute_median,
)
from run_engine import derive_run_features  # noqa: E402
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

EPS = 1e-7

CURATED_DIFFS = ["sp_xwoba_diff", "woba_30g_diff",
                 "lineup_woba_mean_diff", "elo_diff"]


def build_variants() -> dict[str, list[str]]:
    kept, dropped = derive_run_features(list(FEATURE_COLS))
    diffs = [d for d in dropped if d.endswith("_diff")]
    composites = [d for d in dropped if not d.endswith("_diff")]
    assert len(kept) == 29 and len(diffs) == 24 and len(composites) == 5
    return {
        "A": list(kept),
        "B": list(kept) + diffs,
        "C": list(kept) + composites,
        "D": list(kept) + CURATED_DIFFS,
        "REF": list(FEATURE_COLS),
    }


def sha256_file(path: Path) -> str:
    h = hashlib_new = __import__("hashlib").sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def head_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=_BACKEND_DIR.parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


def col_idx(cols: list[str]) -> np.ndarray:
    return np.array([FEATURE_COLS.index(c) for c in cols], dtype=int)


def pooled_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    pc = np.clip(p, EPS, 1 - EPS)
    return {"logloss": round(float(log_loss(y, pc)), 6),
            "auc": round(float(roc_auc_score(y, pc)), 6),
            "n": int(len(y))}


def run_variant(cols: list[str], folds, tune_df, hold_df,
                holdout_days: int) -> dict:
    idx = col_idx(cols)
    # ---- pooled OOF (MLP primary + logistic secondary on the subset) ------
    mlp_p, log_p, ys = [], [], []
    for s in folds:
        X_tr, _, y_tr = _prepare_features(s["train_games"])
        X_va, _, y_va = _prepare_features(s["val_games"])
        X_tr_i, med = _impute_median(X_tr)
        X_va_i, _ = _impute_median(X_va, med)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_i)[:, idx]
        X_va_s = sc.transform(X_va_i)[:, idx]
        m = MLPClassifier(**MLP_PARAMS)
        m.fit(X_tr_s, y_tr)
        mlp_p.append(m.predict_proba(X_va_s)[:, 1])
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        lr.fit(X_tr_s, y_tr)
        log_p.append(lr.predict_proba(X_va_s)[:, 1])
        ys.append(y_va)
    y_all = np.concatenate(ys)
    pooled = {
        "mlp": pooled_metrics(np.concatenate(mlp_p), y_all),
        "logistic": pooled_metrics(np.concatenate(log_p), y_all),
    }
    # ---- sealed holdout (fitted only at the end) ---------------------------
    X_refit, _, refit_y = _prepare_features(tune_df)
    X_hold, _, hold_y = _prepare_features(hold_df)
    X_refit_i, med = _impute_median(X_refit)
    X_hold_i, _ = _impute_median(X_hold, med)
    sc = StandardScaler()
    X_refit_s = sc.fit_transform(X_refit_i)[:, idx]
    X_hold_s = sc.transform(X_hold_i)[:, idx]
    holdout = {}
    for label, params in (("mlp", dict(MLP_PARAMS)),):
        m = MLPClassifier(**params)
        m.fit(X_refit_s, refit_y)
        p = np.clip(m.predict_proba(X_hold_s)[:, 1], EPS, 1 - EPS)
        holdout[label] = {"logloss": round(float(log_loss(hold_y, p)), 6),
                          "auc": round(float(roc_auc_score(hold_y, p)), 6),
                          "n": int(len(hold_y))}
    return {"n_cols": len(cols), "pooled": pooled, "holdout": holdout}


def collinearity(cols_kept: list[str], cols_restored: list[str],
                 X_imputed: np.ndarray) -> list[dict]:
    out = []
    keep_idx = col_idx(cols_kept)
    Xk = X_imputed[:, keep_idx]
    for f in cols_restored:
        j = FEATURE_COLS.index(f)
        x = X_imputed[:, j]
        # drop constant columns before corr
        mask = (Xk.std(axis=0) > 0) & (np.std(x) > 0)
        if not mask.any():
            out.append({"feature": f, "max_abs_corr": None,
                        "vs": None, "flag": False})
            continue
        corrs = np.corrcoef(Xk[:, mask].T, x)[:-1, -1]
        corrs = np.nan_to_num(corrs, nan=0.0)
        i = int(np.argmax(np.abs(corrs)))
        mx = float(corrs[i])
        out.append({"feature": f, "max_abs_corr": round(abs(mx), 3),
                    "vs": cols_kept[int(np.where(mask)[0][i])],
                    "flag": abs(mx) > 0.8})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", type=str, default="A,B,C,D,REF")
    ap.add_argument("--out", type=Path, default=None)
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
    print(f"commit={sha[:12]} data_sha={data_hash[:12]} games={len(games)} "
          f"tuning={len(tune_df)} holdout={len(hold_df)} "
          f"folds={len(all_splits)}/{len(folds)}")

    variants = build_variants()
    print(f"variants: " + ", ".join(f"{k}({len(v)} cols)" for k, v in variants.items()))

    # collinearity on the median-imputed tuning pool (restored vs kept)
    X_all, _, _ = _prepare_features(tune_df)
    X_imp, _ = _impute_median(X_all)
    kept, dropped = derive_run_features(list(FEATURE_COLS))
    restored = dropped  # all 29 dropped features are the restored candidates
    collin = collinearity(kept, restored, X_imp)
    n_flag = sum(1 for c in collin if c["flag"])
    print(f"collinearity: {len(collin)} restored features vs 29 kept; "
          f"{n_flag} flagged |corr|>0.8")

    # incremental resume
    out = args.out or (DATA_DELIVERY_DIR / f"feature_audit_{sha[:12]}.json")
    if out.exists():
        results = json.loads(out.read_text())
    else:
        results = {"schema": "feature-audit/v1", "commit_sha": sha,
                   "data_sha256": data_hash, "holdout_days": args.holdout_days,
                   "folds_declared": len(all_splits),
                   "folds_executed": len(folds), "clip_eps": EPS,
                   "seed": int(RANDOM_SEED), "collinearity": collin,
                   "variants": {}}
    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    for name in want:
        if name in results["variants"]:
            print(f"  {name}: cached, skipping")
            continue
        print(f"  {name}: running ({len(variants[name])} cols) ...")
        r = run_variant(variants[name], folds, tune_df, hold_df,
                        args.holdout_days)
        r["cols"] = variants[name]
        results["variants"][name] = r
        out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"    pooled mlp={r['pooled']['mlp']} log={r['pooled']['logistic']} "
              f"holdout mlp={r['holdout']['mlp']}")

    # gate summary
    A = results["variants"]["A"]["holdout"]["mlp"]
    print("\n=== sealed-holdout gate (MLP) vs A ===")
    for name in ("B", "C", "D", "REF"):
        if name not in results["variants"]:
            continue
        v = results["variants"][name]["holdout"]["mlp"]
        win = v["logloss"] < A["logloss"] and v["auc"] > A["auc"]
        pooled_win = (results["variants"][name]["pooled"]["mlp"]["logloss"]
                      < results["variants"]["A"]["pooled"]["mlp"]["logloss"])
        print(f"  {name:<3} holdout {v['logloss']:.4f}/{v['auc']:.4f} vs A "
              f"{A['logloss']:.4f}/{A['auc']:.4f} -> "
              f"{'BEATS A' if win else 'loses/ties A'} "
              f"(pooled_win={pooled_win})")
    print(f"\naudit written: {out}")


if __name__ == "__main__":
    main()
