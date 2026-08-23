"""Feature audit for the MLB moneyline ensemble.

Scores every FEATURE_COLS entry on the shipped artifacts:
  * univariate out-of-fold AUC (direction-aware) — does the feature separate
    winners from losers on its own, on games the model never trained on?
  * per-fold stability of that signal (sd across folds)
  * coverage (fraction non-null)
  * value ranges / outlier counts (clip / transform candidates)
  * pairwise correlations (redundancy clusters)
  * per-day walk-forward AUC stability (regime / boundary detection)

Run from the repo root:
    PYTHONPATH=backend python3 backend/feature_audit.py [path/to/game_level_features.csv]

Writes feature_audit_<date>.csv next to the features file and prints a
ranked summary. Pure analysis — trains nothing, modifies nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from training import FEATURE_COLS, walk_forward_splits

DATA_DELIVERY = Path(__file__).resolve().parents[1] / "data_delivery"
CALIBRATION = DATA_DELIVERY / "calibration_20260823.json"


def _signed_auc(y: np.ndarray, x: np.ndarray) -> float | None:
    """AUC with direction: >0.5 means higher x → more likely home win."""
    ok = ~np.isnan(x)
    if ok.sum() < 20 or np.unique(x[ok]).size < 2:
        return None
    a = roc_auc_score(y[ok], x[ok])
    return float(a)


def audit_features(games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for f in FEATURE_COLS:
        if f not in games.columns:
            rows.append({"feature": f, "missing_in_frame": True})
            continue
        y = games["home_win"].to_numpy(float)
        x = pd.to_numeric(games[f], errors="coerce").to_numpy(float)
        cov = float(np.isfinite(x).mean())
        if cov == 0:
            rows.append({"feature": f, "coverage": 0.0, "constant": None})
            continue
        xv = x[np.isfinite(x)]
        const = bool(np.unique(xv).size < 2)
        # Univariate AUC on the pooled OOF set (all val games across folds).
        auc = _signed_auc(y, x)
        # Per-fold stability of the same signal.
        fold_aucs = []
        for split in walk_forward_splits(games):
            v = split["val_games"]
            if len(v) < 40:
                continue
            yv = v["home_win"].to_numpy(float)
            xv_f = pd.to_numeric(v[f], errors="coerce").to_numpy(float)
            fa = _signed_auc(yv, xv_f)
            if fa is not None:
                fold_aucs.append(fa)
        fold_auc = float(np.mean(fold_aucs)) if fold_aucs else None
        fold_sd = float(np.std(fold_aucs)) if len(fold_aucs) > 2 else None
        # Outlier / clip candidates (robust: 3 * MAD beyond the median).
        med = float(np.median(xv))
        mad = float(np.median(np.abs(xv - med))) or 1e-9
        n_out = int((np.abs(xv - med) > 3 * mad).sum())
        rows.append({
            "feature": f,
            "missing_in_frame": False,
            "coverage": round(cov, 4),
            "constant": const,
            "pooled_oof_auc": round(auc, 4) if auc is not None else None,
            "fold_mean_auc": round(fold_auc, 4) if fold_auc is not None else None,
            "fold_sd_auc": round(fold_sd, 4) if fold_sd is not None else None,
            "n_folds": len(fold_aucs),
            "p1": round(float(np.percentile(xv, 1)), 3),
            "p50": round(med, 3),
            "p99": round(float(np.percentile(xv, 99)), 3),
            "min": round(float(xv.min()), 3),
            "max": round(float(xv.max()), 3),
            "n_outliers_3mad": n_out,
            "outlier_pct": round(100.0 * n_out / len(xv), 2),
        })
    return pd.DataFrame(rows)


def redundancy(games: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURE_COLS if c in games.columns]
    m = games[cols].apply(pd.to_numeric, errors="coerce")
    corr = m.corr()
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= 0.6:
                pairs.append({"feature_a": cols[i], "feature_b": cols[j],
                              "corr": round(float(r), 3)})
    return pd.DataFrame(sorted(pairs, key=lambda p: -abs(p["corr"])))


def daily_auc_stability() -> pd.DataFrame:
    if not CALIBRATION.exists():
        return pd.DataFrame()
    cal = json.loads(CALIBRATION.read_text())
    daily = cal.get("daily", [])
    rows = []
    for d in daily:
        m = d.get("metrics") or {}
        a = m.get("auc")
        if a is None:
            continue
        rows.append({"date": d.get("date"), "auc": float(a), "n": d.get("n_games")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["month"] = df["date"].str[:6]
    return df


def main() -> None:
    feat_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DELIVERY / "game_level_features.csv"
    games = pd.read_csv(feat_path)
    print(f"Loaded {len(games)} games from {feat_path}\n")

    rep = audit_features(games)
    order = ["feature", "pooled_oof_auc", "fold_mean_auc", "fold_sd_auc",
             "coverage", "constant", "p1", "p50", "p99", "min", "max",
             "outlier_pct", "n_folds"]
    rep_sorted = rep.sort_values("pooled_oof_auc", ascending=False)
    print("=" * 100)
    print("PER-FEATURE UNIVARIATE OOF AUDIT (ranked by pooled AUC; 0.5 = no lift)")
    print("=" * 100)
    print(rep_sorted[order].to_string(index=False, float_format=lambda v: f"{v:g}"))

    print("\n" + "=" * 100)
    print("REDUNDANCY — pairwise |corr| >= 0.6 (drop candidates)")
    print("=" * 100)
    red = redundancy(games)
    print(red.to_string(index=False) if not red.empty else "(none)")

    print("\n" + "=" * 100)
    print("PER-DAY WALK-FORWARD AUC STABILITY (from calibration daily)")
    print("=" * 100)
    dag = daily_auc_stability()
    if not dag.empty:
        summ = dag.groupby("month").agg(
            n_days=("auc", "size"), mean_auc=("auc", "mean"),
            min_auc=("auc", "min"), max_auc=("auc", "max"))
        print(summ.round(4).to_string())
        print(f"\nOverall: mean={dag['auc'].mean():.4f} sd={dag['auc'].std():.4f} "
              f"n_days={len(dag)} | worst day {dag.loc[dag['auc'].idxmin(), 'date']} "
              f"({dag['auc'].min():.4f})")

    out = DATA_DELIVERY / f"feature_audit_{pd.Timestamp.now():%Y%m%d}.csv"
    rep_sorted[order].to_csv(out, index=False)
    print(f"\nReport written: {out}")


if __name__ == "__main__":
    main()
