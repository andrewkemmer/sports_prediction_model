"""Home-edge × expected-total interaction ablation — WITH vs WITHOUT on the moneyline.

Context (structural finding): the run engine's home edge is environment-
conditional — +0.27 in LOW-total games vs -0.09 in HIGH-total games — but the
moneyline's run_margin_diff (its #1-weighted feature) carries no totals context.

Candidate features (all from the run engine's per-game OOF lambdas, leakage-free
— never the actual total, which is post-game):
    expected_total          = lam_home + lam_away   (run-engine OOF expected runs)
    run_margin_x_exp_total  = run_margin_diff * expected_total   (continuous product)
    high_expected_total     = median-split indicator on expected_total (OOF-only median)

Two-tone protocol:
1. CHEAP OOF PRE-CHECK (logistic, pooled OOF): does the interaction add separation
   beyond the main effects (run_margin_diff + expected_total)? If not, record the
   pre-check as the DON'T ADOPT evidence and SKIP the heavy walk-forward. If it
   does, proceed to the full blend-level ablation.
2. FULL ABLATION (only if the pre-check clears): WITH (product + bucket) vs
   WITHOUT (production 65 cols) through the full walk-forward protocol
   (shared 45-fold CADENCE-7 geometry, pooled OOF + sealed 284 holdout,
   logloss/AUC/ECE, adaptive weights re-earned) — the same discipline as
   run_margin_ablation / categoricals / LGB-rounds. Run engine READ-ONLY;
   the OOF margin + expected-total build is cached.

Gate (full ablation): WITH must beat WITHOUT on sealed-holdout logloss AND AUC
without degrading sealed ECE, and pooled OOF must not be lost.

Emits data_delivery/home_edge_interaction_ablation_<date>.json (date-stamped).
COMMITS NOTHING.

Usage:
    python run_home_edge_interaction_ablation.py --precheck-only   # cheap decision
    python run_home_edge_interaction_ablation.py --skip-precheck   # force full gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

import training  # noqa: E402
import run_margin_ablation as rma  # noqa: E402
from build_oof_margin import LAM_COLS, MARGIN_COL  # noqa: E402
from config import DATA_DELIVERY_DIR, RANDOM_SEED  # noqa: E402

EPS = 1e-7

# Candidate columns built from the run engine's OOF lambdas.
EXP_TOTAL = "expected_total"
PROD_COL = "run_margin_x_exp_total"
BUCKET_COL = "high_expected_total"
_CANDIDATES = [EXP_TOTAL, PROD_COL, BUCKET_COL]


def build_pooled_oof(tune_enriched: pd.DataFrame, folds) -> pd.DataFrame:
    """Concatenate every executed fold's val games over the ENRICHED tuning
    frame — the same pooled-OOF row set run_margin_ablation scores (contains
    lam_home/lam_away/run_margin_diff per game)."""
    rows = [s["val_games"] for s in folds if len(s["val_games"]) > 0]
    return pd.concat(rows, ignore_index=True) if rows else tune_enriched.iloc[0:0]


def add_candidates(df: pd.DataFrame, exp_median: float) -> pd.DataFrame:
    """Attach the candidate columns (leak-free). df must carry lam_home/lam_away.
    exp_median is the pooled-OOF expected-total median (leak-free context)."""
    df = df.copy()
    df[EXP_TOTAL] = df["lam_home"] + df["lam_away"]
    df[PROD_COL] = df[MARGIN_COL] * df[EXP_TOTAL]
    df[BUCKET_COL] = (df[EXP_TOTAL] >= exp_median).astype(float)
    return df


def _logistic(y: np.ndarray, X: np.ndarray, seed: int = RANDOM_SEED):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=2000, random_state=seed)
    m.fit(X, y)
    p = m.predict_proba(X)[:, 1]
    p = np.clip(p, EPS, 1 - EPS)
    return m, p


def _ll(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return 0.5


def precheck(oof: pd.DataFrame) -> dict:
    """Logistic on pooled OOF: main effects vs main + interaction (+ bucket).
    Reports the interaction coefficient sign and ΔAUC/Δlogloss of adding it."""
    rng = np.random.default_rng(RANDOM_SEED)
    y = oof["home_win"].values.astype(float)
    X_main = np.column_stack([
        oof[MARGIN_COL].values.astype(float),
        oof[EXP_TOTAL].values.astype(float),
    ])
    X_inter = np.column_stack([X_main, oof[PROD_COL].values.astype(float)])
    X_all = np.column_stack([X_inter, oof[BUCKET_COL].values.astype(float)])

    m_main, p_main = _logistic(y, X_main)
    m_inter, p_inter = _logistic(y, X_inter)
    m_all, p_all = _logistic(y, X_all)

    coef_inter = float(m_inter.coef_[0][2])  # coefficient on the product term
    # sign against the marginal logistic drift for interpretability
    base_ll, base_auc = _ll(y, p_main), _auc(y, p_main)
    inter_ll, inter_auc = _ll(y, p_inter), _auc(y, p_inter)
    # WITH bucket as well
    all_ll, all_auc = _ll(y, p_all), _auc(y, p_all)

    # Paired bootstrap CIs on ΔAUC(logistic +interaction vs main-only)
    n = len(y)
    dAUC = []
    dLL = []
    for _ in range(300):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        _, pb_main = _logistic(yb, X_main[idx], seed=int(rng.integers(1, 2**31)))
        _, pb_inter = _logistic(yb, X_inter[idx], seed=int(rng.integers(1, 2**31)))
        dAUC.append(_auc(yb, pb_inter) - _auc(yb, pb_main))
        dLL.append(_ll(yb, pb_inter) - _ll(yb, pb_main))

    def ci(a):
        lo, hi = np.percentile(a, [2.5, 97.5])
        return round(float(lo), 4), round(float(hi), 4)

    return {
        "gate": "PRECHECK",
        "n_pooled_oof": int(n),
        "main_effects_logloss": round(base_ll, 4),
        "main_effects_auc": round(base_auc, 4),
        "plus_interaction_logloss": round(inter_ll, 4),
        "plus_interaction_auc": round(inter_auc, 4),
        "plus_interaction_and_bucket_logloss": round(all_ll, 4),
        "plus_interaction_and_bucket_auc": round(all_auc, 4),
        "interaction_coef_sign": ("positive" if coef_inter > 0 else
                                  "negative" if coef_inter < 0 else "zero"),
        "interaction_coef": round(float(coef_inter), 5),
        "delta_auc_add_interaction": round(inter_auc - base_auc, 4),
        "delta_logloss_add_interaction": round(inter_ll - base_ll, 4),
        "delta_auc_ci_2_5_97_5": ci(dAUC),
        "delta_logloss_ci_2_5_97_5": ci(dLL),
        "expected_total_present_in_feature_cols": any(
            c in training.FEATURE_COLS for c in _CANDIDATES),
    }


def _verdict_from_precheck(pr: dict) -> tuple[str, str]:
    lo, hi = pr["delta_auc_ci_2_5_97_5"]
    # Proceed to the full gate only when the interaction's AUC gain excludes 0
    # and it does not hurt logloss. Other cases: DON'T ADOPT at the pre-check.
    auc_gain = pr["delta_auc_add_interaction"]
    ll_delta = pr["delta_logloss_add_interaction"]
    if (lo > 0 and ll_delta <= 0.001):
        return ("PROCEED_TO_FULL", "interaction AUC CI excludes 0 and logloss not worse")
    if auc_gain >= 0.002 and ll_delta <= 0.0005:
        return ("PROCEED_TO_FULL", "interaction AUC gain >= 0.002 with logloss ≤ +0.0005")
    return ("DON'T ADOPT",
            f"interaction adds no clear separation beyond main effects "
            f"(ΔAUC={auc_gain:+.4f} CI {lo:+.4f}/{hi:+.4f}, Δlogloss={ll_delta:+.4f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--precheck-only", action="store_true",
                    help="run the cheap logistic pre-check and stop (decide before the heavy gate)")
    ap.add_argument("--skip-precheck", action="store_true",
                    help="force the full ablation even if the pre-check is thin")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true", help="3 folds, /tmp, gate skipped")
    args = ap.parse_args()

    sha = rma.head_sha()
    (games, tune_enriched, hold_df, folds, _margins,
     hold_margins, rounds, uncov) = rma.prepare_data(holdout_days=21)
    if args.smoke:
        folds = folds[:3]

    oof = build_pooled_oof(tune_enriched, folds)
    if oof.empty:
        raise SystemExit("no pooled OOF rows")

    # Leak-free expected-total median computed on the pooled OOF tuning rows only.
    pool_lams = oof["lam_home"].values + oof["lam_away"].values
    exp_median = float(np.median(pool_lams))
    tune_enriched = add_candidates(tune_enriched, exp_median)
    hold_enriched = add_candidates(rma.attach(hold_df, hold_margins), exp_median)
    oof = add_candidates(oof, exp_median)

    print(f"commit={sha[:12]} games={len(games)} tuning={len(tune_enriched)} "
          f"holdout={len(hold_df)} folds={len(folds)} seed={RANDOM_SEED}")
    print(f"expected-total median (OOF): {exp_median:.3f} | margin coverage "
          f"tuning {100*tune_enriched[MARGIN_COL].notna().mean():.1f}%")

    pr = precheck(oof)
    print("\n── OOF logistic pre-check ──")
    print(f"  main      : logloss {pr['main_effects_logloss']:.4f} AUC {pr['main_effects_auc']:.4f}")
    print(f"  +inter    : logloss {pr['plus_interaction_logloss']:.4f} AUC {pr['plus_interaction_auc']:.4f}")
    print(f"  +int+bkt  : logloss {pr['plus_interaction_and_bucket_logloss']:.4f} AUC {pr['plus_interaction_and_bucket_auc']:.4f}")
    print(f"  interaction coef {pr['interaction_coef']:+.5f} ({pr['interaction_coef_sign']})")
    print(f"  ΔAUC(+inter) {pr['delta_auc_add_interaction']:+.4f} CI "
          f"{pr['delta_auc_ci_2_5_97_5'][0]:+.4f}/{pr['delta_auc_ci_2_5_97_5'][1]:+.4f}"
          f" | Δlogloss {pr['delta_logloss_add_interaction']:+.4f}")

    verdict, reason = _verdict_from_precheck(pr)

    target = pd.Timestamp.now().date().isoformat()
    compact = target.replace("-", "")
    out = args.out or (DATA_DELIVERY_DIR /
                       f"home_edge_interaction_ablation_{compact}.json")

    results = {
        "schema": "home-edge-interaction-ablation/v1",
        "commit_sha": sha,
        "data": "data_delivery/game_level_features.csv",
        "seed": int(RANDOM_SEED),
        "expected_total_source": "run_engine per-side Poisson OOF lambdas "
                                 "(lam_home+lam_away) on the moneyline's own folds, "
                                 "leakage-free; never the actual total",
        "expected_total_median_oof": round(exp_median, 4),
        "candidate_cols": _CANDIDATES,
        "precheck": pr,
        "holdout_days": 21,
        "sealed_window": "[2026-08-05 .. 2026-08-25] (last 284 games by date)",
        "verdict": verdict,
        "verdict_reason": reason,
    }

    if verdict == "PROCEED_TO_FULL" and not args.skip_precheck and not args.smoke:
        print(f"\n→ {verdict}: {reason}\nRunning FULL blend-level ablation ...", flush=True)
        full = _run_full_ablation(folds, tune_enriched, hold_enriched,
                                  out_partial=Path(str(out) + ".partial.json"),
                                  smoke=args.smoke)
        results["full_ablation"] = full
        wi, wo = full.get("WITH"), full.get("WITHOUT")
        if wi and wo:
            hw_i, hw_o = wi["holdout"]["blend"], wo["holdout"]["blend"]
            improve = (hw_i["logloss"] < hw_o["logloss"]) and (hw_i["auc"] >= hw_o["auc"])
            ece_i = wi["holdout"].get("blend_calibrated", hw_i).get("ece", 1)
            ece_o = wo["holdout"].get("blend_calibrated", hw_o).get("ece", 1)
            po_i = wi["pooled"]["blend"]["logloss"]
            po_o = wo["pooled"]["blend"]["logloss"]
            adopt = improve and ece_i <= ece_o and po_i <= po_o + 0.0005
            results["verdict"] = "ADOPT" if adopt else "DON'T ADOPT"
            results["adopt_reason"] = (
                "WITH beats WITHOUT on sealed logloss AND AUC, sealed ECE not "
                "degraded, pooled OOF logloss not lost" if adopt else
                f"sealed gate not cleared (logloss {hw_i['logloss']} vs "
                f"{hw_o['logloss']}, AUC {hw_i['auc']} vs {hw_o['auc']}, "
                f"ECE-cal {ece_i} vs {ece_o})")

    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nVERDICT: {results['verdict']} — {results.get('verdict_reason', reason)}")
    print(f"record -> {out}")


def _run_full_ablation(folds, tune_enriched, hold_enriched, out_partial,
                       smoke: bool = False) -> dict:
    """WITH (product + bucket) vs WITHOUT (production) through the blend protocol."""
    base = [c for c in training.FEATURE_COLS if c not in _CANDIDATES]
    cols = {
        "WITHOUT": list(base),
        "WITH": list(base) + list(_CANDIDATES),
    }
    full = {}
    for name in ("WITHOUT", "WITH"):
        r = rma.run_variant(cols[name], folds, tune_enriched, hold_enriched,
                            partial_path=out_partial)
        r["cols"] = cols[name]
        full[name] = r
        b, h = r["pooled"]["blend"], r["holdout"]["blend"]
        print(f"  {name}: pooled {b['logloss']:.4f}/{b['auc']:.4f} "
              f"ece {b['ece']:.4f} | holdout {h['logloss']:.4f}/{h['auc']:.4f} "
              f"ece {h['ece']:.4f}", flush=True)
    return full


if __name__ == "__main__":
    main()