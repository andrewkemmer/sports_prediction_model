"""run_pitcher_stats_ablation.py — CORRECTED vs CURRENT pitcher-stat features.

Measurement task (NOT a cleanup): the moneyline was trained on the current
(partially biased) sp_era/sp_k9 semantics — missing force_out/FC outs and
dropped intent_walk/truncated_pa PAs (see audit_pitcher_era_k9.py and
features.build_features(corrected_outs=True)). 'Correct' math may help or
hurt out-of-sample; only measurement decides. Production features.py is
NOT changed unless this gate says ADOPT.

Variants
--------
CURRENT   — data_delivery/game_level_features.csv (the production artifact)
CORRECTED — a frame rebuilt from the same Statcast history with
            build_features(..., corrected_outs=True). Only pitcher-stat
            columns differ (asserted column-by-column before modeling).

Protocol (two-family screen, mirrors the repo's locked conventions)
-------------------------------------------------------------------
- Folds: training.walk_forward_splits on the tuning pool, filtered by
  MIN_VAL_FOLD_GAMES. Geometry is a pure function of game_date/home_win,
  so identical across variants — asserted row-for-row (game_pk lists).
- Members: LightGBM + standardized logistic ONLY (the two-family screen).
  Identical folds/seeds (RANDOM_SEED) for both variants.
- Metrics: compute_metrics (clip 1e-7) — logloss / AUC / Brier / ECE —
  pooled per family, plus per-game logloss for the significance tests.
- Significance: paired Diebold-Mariano (HAC-1 variance) + paired t-test
  on per-game logloss (CORRECTED − CURRENT), per family.
- Sealed 21-day holdout: fit-only refit on the whole tuning pool AFTER
  the fold loop; never touched during fold fitting.

Emits data_delivery/pitcher_stats_ablation_<sha>.json. COMMITS NOTHING.

Usage
-----
    python run_pitcher_stats_ablation.py
    python run_pitcher_stats_ablation.py --current-csv A.csv --corrected-csv B.csv
    python run_pitcher_stats_ablation.py --limit-folds 4   # quick check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

if "resource" not in sys.modules:  # POSIX-only module, stub on Windows
    _res = types.ModuleType("resource")
    _res.RUSAGE_SELF = 0
    _res.getrusage = lambda who: type("RU", (), {"ru_maxrss": 0})()
    sys.modules["resource"] = _res

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

import training  # noqa: E402
from features import add_lineup_delta_features  # noqa: E402
from config import (  # noqa: E402
    DATA_DELIVERY_DIR,
    LIGHTGBM_PARAMS,
    MIN_VAL_FOLD_GAMES,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
)

EPS = 1e-7
MEMBERS = ["lightgbm", "logistic"]


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
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


def load_frame(path: Path) -> pd.DataFrame:
    g = pd.read_csv(path)
    g["game_date"] = pd.to_datetime(g["game_date"])
    g = g.dropna(subset=["home_win"]).reset_index(drop=True)
    if "game_id" not in g.columns:
        g.insert(0, "game_id",
                 (g["game_date"].dt.strftime("%Y%m%d")
                  + "_" + g["away_team"].astype(str)
                  + "@" + g["home_team"].astype(str)))
    return add_lineup_delta_features(g)


def split_hold(games: pd.DataFrame, holdout_days: int):
    cutoff = games["game_date"].max() - pd.Timedelta(days=holdout_days - 1)
    tune = games[games["game_date"] < cutoff].reset_index(drop=True)
    hold = games[games["game_date"] >= cutoff].reset_index(drop=True)
    return tune, hold


def folds_for(tune: pd.DataFrame, limit: int = 0):
    all_splits = training.walk_forward_splits(
        tune, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
    folds = [s for s in all_splits if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]
    if limit:
        folds = folds[:limit]
    return folds


def train_two_family(train: pd.DataFrame, val: pd.DataFrame | None):
    """LGB + standardized logistic only — mirrors train_moneyline_ensemble's
    per-member paths exactly (same imputation/scaling/routing)."""
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_train, X_cat_train, y_train = training._prepare_features(train)
    X_train_lr, impute_medians = training._impute_median(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_lr)

    models: dict = {}

    # LightGBM — native categoricals by name.
    num_cols = list(training.FEATURE_COLS)
    X_tr_lgb = pd.DataFrame(X_train, columns=num_cols)
    for i, c in enumerate(training.TREE_CATEGORICAL_COLS):
        X_tr_lgb[c] = np.where(
            X_cat_train[:, i] < 0, training._cat_unk_for(c), X_cat_train[:, i]
        ).astype(int)
    lgbm = LGBMClassifier(**LIGHTGBM_PARAMS)
    if val is not None:
        X_val, X_cat_val, y_val = training._prepare_features(val)
        X_val_lr, _ = training._impute_median(X_val, impute_medians)
        X_val_lgb = pd.DataFrame(X_val, columns=num_cols)
        for i, c in enumerate(training.TREE_CATEGORICAL_COLS):
            X_val_lgb[c] = np.where(
                X_cat_val[:, i] < 0, training._cat_unk_for(c), X_cat_val[:, i]
            ).astype(int)
        lgbm.fit(X_tr_lgb, y_train, eval_set=[(X_val_lgb, y_val)],
                 categorical_feature=training.TREE_CATEGORICAL_COLS)
        X_val_scaled = scaler.transform(X_val_lr)
    else:
        lgbm.fit(X_tr_lgb, y_train,
                 categorical_feature=training.TREE_CATEGORICAL_COLS)
        X_val_scaled = None
    models["lightgbm"] = lgbm

    # Logistic — train-median imputed + standardized, raw-col routing.
    _lr_idx = training._logistic_feature_indices()
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X_train_scaled[:, _lr_idx], y_train)
    models["logistic"] = lr
    models["scaler"] = scaler
    models["impute_median"] = impute_medians
    models["_lr_idx"] = _lr_idx

    if val is None:
        return models, None
    probs: dict[str, np.ndarray] = {}
    probs["lightgbm"] = lgbm.predict_proba(X_val_lgb)[:, 1]
    probs["logistic"] = lr.predict_proba(X_val_scaled[:, _lr_idx])[:, 1]
    return models, (probs, np.asarray(y_val, dtype=float))


def per_game_ll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def dm_test(d: np.ndarray) -> dict:
    """Diebold-Mariano on per-game ll diffs d (corrected − current), HAC-1."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 30 or d.std() == 0:
        return {"stat": None, "p": None, "n": int(n),
                "note": "too few games or zero variance"}
    mu = d.mean()
    gam0 = d.var()
    gam1 = np.mean(d[:-1] * d[1:])
    var_hac = max(gam0 + 2 * gam1, 1e-12)
    stat = mu / np.sqrt(var_hac / n)
    from scipy import stats as st
    p = 2 * (1 - st.norm.cdf(abs(stat)))
    return {"stat": round(float(stat), 4), "p": round(float(p), 4),
            "n": int(n)}


def paired_t(d: np.ndarray) -> dict:
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 3 or d.std() == 0:
        return {"stat": None, "p": None, "n": int(n)}
    from scipy import stats as st
    t, p = st.ttest_1samp(d, 0.0)
    return {"stat": round(float(t), 4), "p": round(float(p), 4),
            "n": int(n)}


def run_variant(frame_csv: Path, ref_folds, ref_tune_df, ref_hold_df,
                limit_folds: int, label: str) -> dict:
    """Each variant trains on ITS OWN frame's fold rows (geometry is shared
    with the reference variant's folds — asserted, then only used for
    training here)."""
    games = load_frame(frame_csv)
    tune, hold = split_hold(games, 21)
    assert len(tune) == len(ref_tune_df) and len(hold) == len(ref_hold_df)
    vfolds = folds_for(tune, limit_folds)
    assert len(vfolds) == len(ref_folds)
    for a, b in zip(ref_folds, vfolds):
        assert pd.Timestamp(a["val_start"]) == pd.Timestamp(b["val_start"])
        assert a["val_games"]["game_pk"].tolist() == b["val_games"]["game_pk"].tolist(), \
            f"{label}: fold geometry desynced"

    oof: dict[str, list] = {m: [] for m in MEMBERS}
    oof_y: list = []
    executed = 0
    for s in vfolds:
        tr = s["train_games"].reset_index(drop=True)
        va = s["val_games"].reset_index(drop=True)
        try:
            _, res = train_two_family(tr, va)
        except Exception as e:
            print(f"  fold {s['fold_idx']} failed: {e}", flush=True)
            continue
        if res is None:
            continue
        probs, yv = res
        for m in MEMBERS:
            oof[m].extend(probs[m])
        oof_y.extend(yv)
        executed += 1

    y_all = np.asarray(oof_y, dtype=float)
    pooled = {}
    for m in MEMBERS:
        pooled[m] = training.compute_metrics(y_all, np.asarray(oof[m]))

    # sealed holdout: fit-only refit on full tune
    models, _ = train_two_family(tune, None)
    hold_out = {}
    Xh, Xch, yh = training._prepare_features(hold)
    num_cols = list(training.FEATURE_COLS)
    Xh_lr, _ = training._impute_median(Xh, models["impute_median"])
    Xh_scaled = models["scaler"].transform(Xh_lr)
    Xh_lgb = pd.DataFrame(Xh, columns=num_cols)
    for i, c in enumerate(training.TREE_CATEGORICAL_COLS):
        Xh_lgb[c] = np.where(Xch[:, i] < 0, training._cat_unk_for(c),
                             Xch[:, i]).astype(int)
    for m in MEMBERS:
        if m == "lightgbm":
            p = models[m].predict_proba(Xh_lgb)[:, 1]
        else:
            p = models[m].predict_proba(Xh_scaled[:, models["_lr_idx"]])[:, 1]
        hold_out[m] = training.compute_metrics(
            np.asarray(yh, dtype=float), p)
        hold_out[f"{m}_per_game_ll"] = per_game_ll(
            np.asarray(yh, dtype=float), p).tolist()

    return ({"label": label, "n_games": int(len(y_all)),
             "folds_executed": executed, "pooled": pooled,
             "holdout": {k: v for k, v in hold_out.items()
                         if not k.endswith("_per_game_ll")},
             "holdout_per_game_ll": hold_out},
            y_all, oof)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--current-csv", type=Path,
                    default=DATA_DELIVERY_DIR / "game_level_features.csv")
    ap.add_argument("--corrected-csv", type=Path,
                    default=DATA_DELIVERY_DIR / "game_level_features_corrected.csv")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--holdout-days", type=int, default=21)
    ap.add_argument("--limit-folds", type=int, default=0)
    args = ap.parse_args()

    out = args.out or (DATA_DELIVERY_DIR /
                       f"pitcher_stats_ablation_{head_sha()[:12]}.json")

    cur = load_frame(args.current_csv)
    cor = load_frame(args.corrected_csv)
    assert len(cur) == len(cor), (len(cur), len(cor))
    assert (cur["game_id"].values == cor["game_id"].values).all(), \
        "frames must contain the SAME games (same order)"
    assert cur["home_win"].equals(cor["home_win"]), "labels differ"
    assert cur["game_date"].equals(cor["game_date"]), "dates differ"

    # Column-diff audit: only pitcher-stat columns may differ.
    numeric = cur.select_dtypes(include=[np.number]).columns
    differing = []
    for c in numeric:
        a = pd.to_numeric(cur[c], errors="coerce").fillna(0.0)
        b = pd.to_numeric(cor[c], errors="coerce").fillna(0.0)
        if (a - b).abs().max() > 1e-9:
            differing.append(c)
    # ace_efficiency_factor / pitcher_regression_indicator are products of
    # pitcher-stat columns (features.py:2415/2424) — part of the expected diff.
    pit_cols = [c for c in differing
                if c.startswith(("sp_", "bullpen")) or "sp_" in c
                or c in ("ace_efficiency_factor", "pitcher_regression_indicator")]
    print(f"[frame] {len(cur)} games | columns differing: {len(differing)}")
    for c in differing:
        mark = " <-- non-pitcher col DIFFERS (unexpected!)" if c not in pit_cols else ""
        print(f"    {c}{mark}")
    assert set(differing) == set(pit_cols), \
        f"non-pitcher columns differ unexpectedly: {set(differing)-set(pit_cols)}"

    tune, hold = split_hold(cur, args.holdout_days)
    folds = folds_for(tune, args.limit_folds)
    print(f"[protocol] {len(folds)} folds | tune {len(tune)} | "
          f"holdout {len(hold)} ({args.holdout_days}d) | seed {RANDOM_SEED}",
          flush=True)

    variants = {}
    pergame = {}
    for label, csv in (("CURRENT", args.current_csv),
                       ("CORRECTED", args.corrected_csv)):
        print(f"── {label} walk-forward (LGB + logistic) ──", flush=True)
        var, y, oof = run_variant(csv, folds, tune, hold,
                                  args.limit_folds, label)
        variants[label] = var
        pg = {m: np.asarray(oof[m]) for m in MEMBERS}
        yarr = np.asarray(y, dtype=float)
        pergame[label] = {
            "y": yarr.tolist(),
            "lightgbm_ll": per_game_ll(yarr, pg["lightgbm"]).tolist(),
            "logistic_ll": per_game_ll(yarr, pg["logistic"]).tolist(),
        }

    print("\n=== POOLED OOF (two-family, identical folds) ===")
    hdr = f"{'member':10s} {'pooled ll cur→cor':26s} {'AUC cur→cor':22s} {'ECE cur→cor':18s}"
    print(hdr)
    sig = {}
    for m in MEMBERS:
        cur_m = variants["CURRENT"]["pooled"][m]
        cor_m = variants["CORRECTED"]["pooled"][m]
        d = (np.asarray(pergame["CORRECTED"][f"{m}_ll"])
             - np.asarray(pergame["CURRENT"][f"{m}_ll"]))
        dm = dm_test(d)
        tt = paired_t(d)
        sig[m] = {"dm": dm, "t": tt,
                  "mean_delta_ll": round(float(d.mean()), 6),
                  "pooled_ll_delta": round(float(cor_m["logloss"] - cur_m["logloss"]), 4)}
        print(f"{m:10s} {cur_m['logloss']} → {cor_m['logloss']}  "
              f"{cur_m['auc']} → {cor_m['auc']}  {cur_m['ece']} → {cor_m['ece']}")
        print(f"            per-game ll Δ=+{d.mean():+.6f} (corrected−current) | "
              f"DM stat {dm['stat']} p={dm['p']} | paired-t stat {tt['stat']} p={tt['p']}")

    print("\n=== SEALED HOLDOUT (21d) ===")
    for m in MEMBERS:
        ch = variants["CURRENT"]["holdout"][m]
        oh = variants["CORRECTED"]["holdout"][m]
        d = (np.asarray(variants["CORRECTED"]["holdout_per_game_ll"][f"{m}_per_game_ll"])
             - np.asarray(variants["CURRENT"]["holdout_per_game_ll"][f"{m}_per_game_ll"]))
        print(f"{m:10s} ll {ch['logloss']} → {oh['logloss']} | auc {ch['auc']} → {oh['auc']} | "
              f"ece {ch['ece']} → {oh['ece']} | ll Δ={d.mean():+.6f}")

    report = {
        "task": "pitcher-stats corrected-outs ablation (two-family screen)",
        "head": head_sha(),
        "frames": {"current": str(args.current_csv),
                   "current_sha": sha256_file(args.current_csv),
                   "corrected": str(args.corrected_csv),
                   "corrected_sha": sha256_file(args.corrected_csv),
                   "differing_columns": differing},
        "protocol": {"members": MEMBERS, "holdout_days": args.holdout_days,
                     "folds": len(folds), "seed": RANDOM_SEED},
        "variants": variants,
        "significance": sig,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport → {out}")


if __name__ == "__main__":
    main()