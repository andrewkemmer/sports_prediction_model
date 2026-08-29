"""NFL moneyline v1 — the feature-admission model gate (NO adoption unless the
sealed window earns it).

Builds on the v1 admitted features (``nfl_features.py``): elo_diff,
form_diff_pts, rest_days_diff, ypp_diff, is_dome_home, and is_home (the
home-edge anchor — a constant, so it is carried by the baselines/intercept and
NOT fed as a model column). Target = home_win (home_score > away_score).

Discipline (MLB retrospective):
- START SIMPLE: a single LightGBM with modest regularization + a logistic
  reference arm. NO 5-member weighted ensemble (that is a later upgrade to
  gate, not the v1 baseline). No interactions; no new features.
- STRICT point-in-time: every feature is already leakage-safe (v1 gate;
  ``team_stats_ladder`` asserts per-team strict gameday monotonicity). At the
  model entry point we additionally assert walk-forward folds never train on a
  row at/after the fold's validation week, and that season 2025 (the SEALED
  hold-out) never appears in any pre-sealed fit or calibration map.
- Prequential weekly-cadence folds over 2019-2024 (warm-up = first two full
  seasons, 2019+2020, are never validated): pooled OOF logloss/AUC/ECE, plus
  an honestly-nested Platt twin.
- SEALED hold-out: ALL of 2025, model fitted on 2019-2024 only, calibrated by
  a Platt map fitted only on pre-holdout OOF (2021-2024 pooled).
- Baselines to beat (sealed): (a) constant home-edge (NFL home win rate),
  (b) elo-only logistic (cheapest signal). ADOPT requires the model to beat
  BOTH on the SEALED window in logloss AND AUC with a sane ECE. A pooled-gain
  / sealed-loss inversion means DON'T ADOPT.

Artifact: data_delivery/nfl_moneyline_v1_<date>.json — fold geometry,
per-arm pooled + sealed tables (raw + Platt twins), baselines, verdict+reason.
No predictions artifacts are written yet (that is the NEXT task, only if ADOPT).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"
DECIDED_FRAME = DATA_DELIVERY_DIR / "nfl_game_level_features.csv"

# model inputs = the v1 numeric feature set (is_home is a constant anchor ->
# absorbed by intercept/baseline, not fed as a column)
V1_FEATURES = ["elo_diff", "form_diff_pts", "rest_days_diff", "ypp_diff",
               "is_dome_home"]
TARGET = "home_win"

WARMUP_SEASONS = [2018]
TRAIN_SEASONS = list(range(2019, 2025))   # pre-sealed training window: 2019..2024
VAL_SEASONS = [2021, 2022, 2023, 2024]    # prequential validation (2-season warm-up)
SEALED_SEASON = 2025

DATE_FMT = "%Y%m%d"
RECORD_TEMPLATE = f"nfl_moneyline_v1_{{date}}.json"

# LightGBM hyperparams (modest regularization, deterministic)
LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 40,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "seed": 7,
    "verbosity": -1,
    "force_row_wise": True,
    "boost_from_average": True,
    "n_jobs": 1,
}
NUM_BOOST_ROUND = 300
EARLY_STOPPING = 30

ECE_BINS = 10
ECE_MAX = 0.08          # "sane" calibration bar for adoption
PROB_EPS = 1e-6


# ---------------------------------------------------------------------------
# Metrics (pure)
# ---------------------------------------------------------------------------
def clip_p(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.clip(p, PROB_EPS, 1.0 - PROB_EPS)


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = clip_p(np.asarray(p, dtype=float))
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    n = len(p)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        total += (m.sum() / n) * abs(acc - conf)
    return float(total)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC via the v1 gate's rank statistic (ties -> 0.5)."""
    from nfl_features import univariate_auc
    return float(univariate_auc(np.asarray(y), np.asarray(p)))


# ---------------------------------------------------------------------------
# Platt calibration (pure, via sklearn; fit maps are sealed-off-holdout)
# ---------------------------------------------------------------------------
def platt_fit(p: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    x = np.log(clip_p(p) / (1 - clip_p(p))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6)          # essentially unregularized Platt map
    lr.fit(x, y)
    return lr


def platt_predict(p: np.ndarray, lr) -> np.ndarray:
    x = np.log(clip_p(p) / (1 - clip_p(p))).reshape(-1, 1)
    return lr.predict_proba(x)[:, 1]


# ---------------------------------------------------------------------------
# Walk-forward fold generation (pure, leakage-asserted)
# ---------------------------------------------------------------------------
def _week_start(dates: pd.Series) -> pd.Series:
    """Monday of each date's calendar week (NFL weeks are Mon-Sun)."""
    d = pd.to_datetime(dates)
    return d - pd.to_timedelta(d.dt.weekday, unit="D")


def generate_weekly_folds(preq: pd.DataFrame,
                          val_seasons: list[int] | None = None) -> list[dict]:
    """Prequential weekly folds over ``preq`` (2019-2024 decided games).

    Each fold validates all games in one calendar week (Mon-Sun) of a
    validation season; its train set is EVERY game with gameday strictly
    before that week (so a fold can never see its own or any future week).

    LEAKAGE ASSERTION: for every fold, max(train.gameday) < min(val.gameday).
    """
    val_seasons = val_seasons or VAL_SEASONS
    g = preq.copy()
    g["gameday"] = pd.to_datetime(g["gameday"], errors="coerce")
    g = g.sort_values("gameday").reset_index(drop=True)
    g["week_start"] = _week_start(g["gameday"])
    g["val_season"] = g["season"].isin(val_seasons)
    folds = []
    for mon, idx in g[g["val_season"]].groupby("week_start")["week_start"].groups.items():
        val = g.loc[idx]
        train = g[g["gameday"] < mon]
        if len(val) == 0 or len(train) == 0:
            continue
        tr_max = train["gameday"].max()
        va_min = val["gameday"].min()
        if not (tr_max < mon <= va_min):
            raise AssertionError(
                f"fold week {mon}: train max {tr_max} not strictly before "
                f"val min {va_min} -> future-week leak")
        folds.append({"week_start": mon, "train": train.copy(), "val": val.copy()})
    folds.sort(key=lambda f: f["week_start"])
    return folds


# ---------------------------------------------------------------------------
# Model arms
# ---------------------------------------------------------------------------
def _valid_rows(df: pd.DataFrame) -> np.ndarray:
    return df[V1_FEATURES + [TARGET]].notna().all(axis=1).to_numpy()


def fit_predict_lgbm(Xtr: np.ndarray, ytr: np.ndarray,
                     Xte: np.ndarray, early_stopping_val=None) -> np.ndarray:
    """Single LightGBM; by default fixed rounds (NO early-stop on a fold's own
    labels, which would leak that fold: early_stopping_val is only for tests)."""
    import lightgbm as lgb
    dtr = lgb.Dataset(Xtr, ytr)
    callbacks = [lgb.log_evaluation(0)]
    if early_stopping_val is not None:
        Xv, yv = early_stopping_val
        dv = lgb.Dataset(Xv, yv, reference=dtr)
        bst = lgb.train(dict(LGB_PARAMS), dtr, num_boost_round=NUM_BOOST_ROUND,
                        valid_sets=[dv], callbacks=[lgb.log_evaluation(0),
                                                    lgb.early_stopping(EARLY_STOPPING)])
        return bst.predict(Xte)
    bst = lgb.train(dict(LGB_PARAMS), dtr, num_boost_round=NUM_BOOST_ROUND,
                    callbacks=callbacks)
    return bst.predict(Xte)


def _platt_on(ppool: list[np.ndarray], ypool: list[np.ndarray]) -> object:
    return platt_fit(np.concatenate(ppool), np.concatenate(ypool).astype(int))


def run_walk_forward(feats: pd.DataFrame) -> dict:
    """Prequential fold evaluation over 2019-2024 + sealed 2025 evaluation.

    Returns per-arm pooled tables + sealed tables (raw + Platt twins) and the
    adoption verdict. No training ever sees 2025; the sealed Platt map is fit
    only on the pooled pre-holdout OOF (2021-2024), never 2025.
    """
    preq_all = feats[feats["season"].isin(TRAIN_SEASONS)].copy()
    sealed = feats[feats["season"] == SEALED_SEASON].copy()

    # Universe for a fair comparison: rows with all v1 numeric features + target.
    preq = preq_all[_valid_rows(preq_all)].copy()
    sld = sealed[_valid_rows(sealed)].copy()

    folds = generate_weekly_folds(preq)          # asserts no future-week leak

    # NOTE: warm-up (2019-2020) is never validated -> folds cover 2021-2024.
    Xcol = V1_FEATURES
    ycol = TARGET

    # ---- per-fold store for nested (honest) preq Platt twin ----
    order_actual, order_raw, order_elo, ws_list = [], [], [], []

    for f in folds:
        tr, va = f["train"], f["val"]
        Xtr = tr[Xcol].to_numpy(dtype=float)
        ytr = tr[ycol].to_numpy(dtype=float)
        Xva = va[Xcol].to_numpy(dtype=float)
        yva = va[ycol].to_numpy(dtype=float)
        # model: single LightGBM, early stop on the fold's train itself? No —
        # early-stop on validation WOULD let the fold's own labels shape the
        # model. Use a holdout-free default: fixed rounds (no early stopping).
        raw = fit_predict_lgbm(Xtr, ytr, Xva)
        # elo-only logistic reference
        from sklearn.linear_model import LogisticRegression
        elo = LogisticRegression(max_iter=1000)
        elo.fit(Xtr[:, V1_FEATURES.index("elo_diff")].reshape(-1, 1), ytr)
        elo_p = elo.predict_proba(
            Xva[:, V1_FEATURES.index("elo_diff")].reshape(-1, 1))[:, 1]

        order_actual.append(yva)
        order_raw.append(raw)
        order_elo.append(elo_p)
        ws_list.append(f["week_start"])

    # ---- nested Platt twin for the preq window (honest per-fold) ----
    cal_pool, raw_pool, elo_pool, y_pool = [], [], [], []
    cal_cum = []
    for i, (ya, raw) in enumerate(zip(order_actual, order_raw)):
        # fit Platt on all STRICTLY-EARLIER folds' OOF, apply to this fold
        if i == 0:
            cal_p = raw.copy()
        else:
            pp = [r for (_, r) in cal_cum]
            yy = [y_ for (y_, _) in cal_cum]
            lr = platt_fit(np.concatenate(pp), np.concatenate(yy).astype(int))
            cal_p = platt_predict(raw, lr)
        cal_pool.append(cal_p)
        raw_pool.append(raw)
        elo_pool.append(order_elo[i])
        y_pool.append(ya)
        cal_cum.append((ya, raw))

    y_po = np.concatenate(y_pool)
    raw_po = np.concatenate(raw_pool)
    cal_po = np.concatenate(cal_pool)
    elo_po = np.concatenate(elo_pool)

    # constant home-edge baseline fit on pre-holdout (2019-2024) only
    const_p = preq[ycol].mean()

    pooled = {
        "n": int(len(y_po)),
        "fold_count": len(folds),
        "constant_home_edge": {
            "proba": round(float(const_p), 4),
            "logloss": round(logloss(y_po, np.full_like(y_po, const_p)), 4),
            "auc": round(auc(y_po, np.full_like(y_po, const_p)), 4),
        },
        "elo_logistic": {
            "logloss": round(logloss(y_po, elo_po), 4),
            "auc": round(auc(y_po, elo_po), 4),
        },
        "model_raw": {
            "logloss": round(logloss(y_po, raw_po), 4),
            "auc": round(auc(y_po, raw_po), 4),
        },
        "model_platt": {
            "logloss": round(logloss(y_po, cal_po), 4),
            "auc": round(auc(y_po, cal_po), 4),
            "ece": round(ece(y_po, cal_po), 4),
        },
    }

    # ---- SEALED 2025 ----
    # model fitted on ALL 2019-2024 (no fold) -> predict 2025
    Xtr = preq[Xcol].to_numpy(dtype=float)
    ytr = preq[ycol].to_numpy(dtype=float)
    Xse = sld[Xcol].to_numpy(dtype=float)
    yse = sld[ycol].to_numpy(dtype=float)

    sealed_raw = fit_predict_lgbm(Xtr, ytr, Xse)
    from sklearn.linear_model import LogisticRegression
    elo_sealed = LogisticRegression(max_iter=1000)
    elo_sealed.fit(Xtr[:, V1_FEATURES.index("elo_diff")].reshape(-1, 1), ytr)
    sealed_elo = elo_sealed.predict_proba(
        Xse[:, V1_FEATURES.index("elo_diff")].reshape(-1, 1))[:, 1]

    # Platt twin for the sealed window: fit on pooled pre-holdout OOF only
    platt_sealed = platt_fit(raw_po, y_po.astype(int))
    sealed_cal = platt_predict(sealed_raw, platt_sealed)

    const_sealed = ytr.mean()
    sealed = {
        "n": int(len(yse)),
        "constant_home_edge": {
            "proba": round(float(const_sealed), 4),
            "logloss": round(logloss(yse, np.full_like(yse, const_sealed)), 4),
            "auc": round(auc(yse, np.full_like(yse, const_sealed)), 4),
        },
        "elo_logistic": {
            "logloss": round(logloss(yse, sealed_elo), 4),
            "auc": round(auc(yse, sealed_elo), 4),
        },
        "model_raw": {
            "logloss": round(logloss(yse, sealed_raw), 4),
            "auc": round(auc(yse, sealed_raw), 4),
        },
        "model_platt": {
            "logloss": round(logloss(yse, sealed_cal), 4),
            "auc": round(auc(yse, sealed_cal), 4),
            "ece": round(ece(yse, sealed_cal), 4),
        },
    }

    verdict = adopt_decision(pooled, sealed)
    return {
        "fold_geometry": {
            "train_seasons": TRAIN_SEASONS,
            "val_seasons": VAL_SEASONS,
            "sealed_season": SEALED_SEASON,
            "fold_count": len(folds),
            "pooled_oof_games": int(len(y_po)),
            "sealed_games": int(len(yse)),
            "preq_weeks": [str(f["week_start"].date()) for f in folds],
        },
        "pooled_preq_2021_2024": pooled,
        "sealed_2025": sealed,
        "verdict": verdict,
    }


def adopt_decision(pooled: dict, sealed: dict) -> dict:
    """ADOPT only if the calibrated model beats BOTH baselines on the SEALED
    window in both logloss and AUC, with sane ECE. A pooled-gain/sealed-loss
    inversion -> DON'T ADOPT (the pattern every MLB gate hit)."""
    m_ll = sealed["model_platt"]["logloss"]
    m_auc = sealed["model_platt"]["auc"]
    m_ece = sealed["model_platt"]["ece"]
    elo_ll = sealed["elo_logistic"]["logloss"]
    elo_auc = sealed["elo_logistic"]["auc"]
    c_ll = sealed["constant_home_edge"]["logloss"]
    c_auc = sealed["constant_home_edge"]["auc"]

    beats_elo = (m_ll < elo_ll) and (m_auc > elo_auc)
    beats_const = (m_ll < c_ll) and (m_auc > c_auc)
    sane_ece = m_ece <= ECE_MAX

    # pooled-vs-sealed inversion warning (informational unless sealed fails)
    pm_ll = pooled["model_platt"]["logloss"]
    pe_ll = pooled["elo_logistic"]["logloss"]
    pc_ll = pooled["constant_home_edge"]["logloss"]
    pooled_wing = pm_ll < min(pe_ll, pc_ll)
    sealed_wing = m_ll < min(elo_ll, c_ll)

    adopt = bool(beats_elo and beats_const and sane_ece)
    reasons = []
    if not beats_elo:
        reasons.append(f"sealed logloss {m_ll} / auc {m_auc} not both better "
                       f"than elo-logistic ({elo_ll} / {elo_auc})")
    if not beats_const:
        reasons.append(f"sealed logloss {m_ll} / auc {m_auc} not both better "
                       f"than constant home-edge ({c_ll} / {c_auc})")
    if not sane_ece:
        reasons.append(f"sealed ECE {m_ece} > {ECE_MAX} (not sane)")
    if not adopt and (pooled_wing and not sealed_wing):
        reasons.append("pooled-gain / sealed-loss inversion -> DON'T ADOPT")
    elif adopt and (sealed_wing and not pooled_wing):
        reasons.append("note: model wins sealed but slightly worse pooled (watch)")

    return {
        "adopt": adopt,
        "sealed_beats_elo": bool(beats_elo),
        "sealed_beats_constant": bool(beats_const),
        "sane_ece": bool(sane_ece),
        "pooled_gain_sealed_loss_inversion": bool(pooled_wing and not sealed_wing),
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def pull_and_run(out_dir: Path | None = None,
                 write_record: bool = True,
                 features_csv: Path | None = None) -> dict:
    from nfl_features import _load_raw, build_features, DEFAULT_SEASONS
    out_dir = Path(out_dir) if out_dir is not None else DATA_DELIVERY_DIR

    if features_csv is not None and Path(features_csv).exists():
        logger.info("Loading pre-computed features from %s", features_csv)
        feats = pd.read_csv(features_csv)
        feats["gameday"] = pd.to_datetime(feats["gameday"])
        if TARGET not in feats.columns:
            feats[TARGET] = (feats["home_score"] > feats["away_score"]).astype(int)
    else:
        if not DECIDED_FRAME.exists():
            raise FileNotFoundError(
                f"{DECIDED_FRAME} absent — run `python3 nfl_game_frame.py` first")
        decided = pd.read_csv(DECIDED_FRAME)
        logger.info("Computing v1 features over %s seasons", DEFAULT_SEASONS)
        schedule, pbp = _load_raw(DEFAULT_SEASONS)
        feats = build_features(decided, schedule, pbp)
        feats[TARGET] = (feats["home_score"] > feats["away_score"]).astype(int)

    # sealed isolation assertion: no 2025 row may be used in any pre-sealed fit
    # (guaranteed by construction in run_walk_forward; re-assert here loudly)
    if SEALED_SEASON not in TRAIN_SEASONS and SEALED_SEASON not in VAL_SEASONS:
        assert not feats[feats["season"] == SEALED_SEASON].empty

    result = run_walk_forward(feats)

    if write_record:
        record = {
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "config": {
                "features": V1_FEATURES,
                "excluded_constant_anchor": "is_home",
                "model": "LightGBM (single, modest reg) + Platt twin",
                "reference_arm": "logistic (full-fitted, elo-only for cheap signal)",
                "baselines": ["constant home-edge", "elo-only logistic"],
                "lgb_params": {k: v for k, v in LGB_PARAMS.items()},
                "num_boost_round": NUM_BOOST_ROUND,
                "ece_bins": ECE_BINS, "ece_max": ECE_MAX,
                "leakage": ("features strictly-trailing (v1 gate); folds assert "
                            "train.gameday < week_start; 2025 never in any "
                            "pre-sealed fit or calibration map"),
            },
            **result,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        rec_path = out_dir / RECORD_TEMPLATE.format(date=datetime.now().strftime(DATE_FMT))
        with open(rec_path, "w") as fh:
            json.dump(record, fh, indent=2)
        result["record"] = str(rec_path)

    _print_report(result)
    return result


def _print_report(result: dict) -> None:
    print("\n=== NFL moneyline v1 gate ===")
    print("pooled OOF (2021-2024):", result["fold_geometry"]["pooled_oof_games"],
          "games,", result["fold_geometry"]["fold_count"], "folds")
    print(format_table("sealed_2025", result["sealed_2025"]))
    print("VERDICT:", "ADOPT" if result["verdict"]["adopt"] else "DO NOT ADOPT")
    for r in result["verdict"]["reasons"]:
        print("  -", r)


def format_table(window: str, arms: dict) -> str:
    lines = [f"\n{window}:"]
    lines.append(f"  {'arm':20s} {'logloss':>9s} {'auc':>7s} {'ece':>6s}")
    for name in ("constant_home_edge", "elo_logistic", "model_raw", "model_platt"):
        a = arms[name]
        lines.append(f"  {name:20s} {a['logloss']:9.4f} {a['auc']:7.4f} "
                     f"{a.get('ece', float('nan')):6.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        description="Run the NFL moneyline v1 walk-forward + sealed gate (no adopt "
                    "unless the sealed window earns it).")
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON record")
    ap.add_argument("--features-csv", type=Path, default=None,
                    help="path to pre-computed features CSV (skips nflreadpy download)")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    pull_and_run(write_record=not args.no_record, features_csv=args.features_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())