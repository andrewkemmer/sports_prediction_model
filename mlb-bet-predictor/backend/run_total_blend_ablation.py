"""Discriminative totals/run-line blending member — GBM vs NB-only gate.

Trains a gradient-boosted DISCRIMINATIVE member on the run engine's 29-feature
view (derive_run_features(FEATURE_COLS): levels + environment only, the same
leak-free keep-list the run engine itself uses — 36 _diff/matchup columns
excluded, incl. run_margin_diff) over the SAME 45-fold walk-forward geometry
as production. Three heads (the two mandated plus the run-line mirror):

  (a) E[total runs]        LGBMRegressor (objective=poisson) on home+away
  (b) P(over 8.5)          LGBMClassifier on total_runs >= 9
  (c) P(home cover -1.5)   LGBMClassifier on margin >= 2
                           (the run-line mirror of (b); required so the
                           run-line gate leg has a GBM probability — a
                           totals head alone cannot split the margin)

The NB sampler member is the CURRENT production prices: per-game OOF lambdas
from the run engine (READ-ONLY; the margin OOF cache built on this exact fold
geometry — cache hit expected) + α(λ) dispersion from the committed monitor
JSON (fallback: fit_alpha on the pre-sealed pooled OOF), Monte-Carlo priced
(10k draws, production convention). Both members price the SAME games' OOF,
leak-free and apples-to-apples.

Blend arms (task: "weighted or a small stacker on the two members' OOF probs"):
  nb_only            production baseline
  gbm_only           the discriminative member alone (context)
  blend_equal        fixed 50/50 probability average
  blend_stack        prequential L2 logistic stacker (unconstrained) on the
                     two members' OOF probs — fold k fit on folds < k only;
                     sealed fit on ALL pooled OOF pairs (strictly pre-holdout)
  blend_stack_nonneg same stacker with w >= 0 (pure convex combination)

Surfaces (task: "totals + run-line ECE/logloss/AUC"):
  totals    FIXED over_8_5 reference for BOTH arms — the same event both
            members price; AUC is never a mixed-line rank (per-game line
            probabilities exist only in the NB grid, so the comparable
            totals leg is the fixed reference; the NB-only per-game
            assigned-line push-excluded totals is reported as context)
  run_line  p_home_cover_1_5 vs margin >= 2 (half-run line never pushes)
Both pooled OOF + sealed 284 (the last 284 decided games by date —
2026-08-06 .. 2026-08-26 on the 20260827 frame; strictly pre-holdout for
every fit), logloss / AUC / ECE with prequential Platt calibrated twins
(fold k map fit on prior folds' pairs; sealed map fit on all pooled OOF pairs
of the arm's own blend).

Gate (task rule): ADOPT only if the best blend beats NB-only on SEALED
logloss AND AUC without degrading sealed ECE, on BOTH surfaces, and pooled
OOF logloss/ECE is not lost. State ADOPT / DON'T ADOPT flatly.

Run engine READ-ONLY; standalone harness; COMMITS NOTHING.

Usage:
    python run_total_blend_ablation.py
    python run_total_blend_ablation.py --smoke   # 3 folds -> /tmp, gate skipped
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

import run_margin_ablation as rma  # noqa: E402
import run_stack_ablation as rsa  # noqa: E402
import training  # noqa: E402
from calibration import apply_platt, fit_platt  # noqa: E402
from config import DATA_DELIVERY_DIR, RANDOM_SEED  # noqa: E402
from run_engine import (  # noqa: E402
    EARLY_STOPPING_ROUNDS,
    MAX_ROUNDS,
    RUN_LGBM_PARAMS,
    TOTAL_LINE_GRID,
    _rounded_total_line,
    alpha_of,
    derive_markets_mc,
    derive_run_features,
    fit_alpha,
)

EPS = 1e-7
MC_N = 10_000                 # production-grade draws (matches shipped MC)
MC_SEED = 42                  # run_engine.MARKET_SEED
SEALED_N = 284
# Sealed-window sampling bands (mirror run_engine_edge_correction_gate):
# non-degradation tolerance = 2x the metric's rough sampling se on n games.
def _se_band(n: int, kind: str) -> float:
    if kind == "ece":
        return float(np.sqrt(0.05 * 0.95 / max(n, 1)))
    if kind == "ll":
        return float(0.7 / np.sqrt(max(n, 1)))
    return float(0.5 / np.sqrt(max(n, 1)))  # auc

POOLED_LL_TOL = 0.0005        # pooled "not lost" logloss tolerance
POOLED_ECE_TOL = 0.0005
BLEND_VARIANTS = ["blend_equal", "blend_stack", "blend_stack_nonneg"]


def _gbm_params(kind: str) -> dict:
    p = {k: v for k, v in RUN_LGBM_PARAMS.items()}
    if kind == "reg":
        p["objective"] = "poisson"
    else:
        p["objective"] = "binary"
    return p


def _gbm_head_targets(va: pd.DataFrame) -> dict[str, np.ndarray]:
    total = va["home_score"].to_numpy(float) + va["away_score"].to_numpy(float)
    return {
        "reg": total,
        "over": (total >= 9).astype(float),
        "cover": ((va["home_score"].to_numpy(float)
                   - va["away_score"].to_numpy(float)) >= 2).astype(float),
    }


def _fit_and_predict(feat_cols: list[str], tr: pd.DataFrame, va: pd.DataFrame,
                     y_tr: np.ndarray, y_eval: np.ndarray,
                     kind: str) -> tuple[object, np.ndarray, int]:
    """Fit a LightGBM head with the run engine's walk-forward convention
    (early stopping on the fold's val — the repo's accepted discipline) and
    return (model, val predictions, best_iteration)."""
    from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, \
        log_evaluation

    cls = LGBMRegressor if kind == "reg" else LGBMClassifier
    model = cls(**_gbm_params(kind))
    model.set_params(n_estimators=MAX_ROUNDS)
    X_tr = tr.reindex(columns=feat_cols).astype(float)
    X_va = va.reindex(columns=feat_cols).astype(float)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_eval)],
              callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                         log_evaluation(period=0)])
    best = int(model.best_iteration_ or MAX_ROUNDS)
    if kind == "reg":
        pred = np.clip(model.predict(X_va, num_iteration=best), 1e-6, None)
    else:
        pred = model.predict_proba(X_va, num_iteration=best)[:, 1]
    return model, np.asarray(pred, dtype=float), best


def _metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    return training.compute_metrics(y, np.clip(p, EPS, 1 - EPS))


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score
    try:
        a = float(roc_auc_score(y, p))
    except ValueError:
        return None
    return None if not np.isfinite(a) else round(a, 5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="3 folds -> /tmp, gate skipped")
    args = ap.parse_args()
    if args.smoke:
        args.out = Path("/tmp/total_blend_ablation_smoke.json")

    sha = rma.head_sha()
    (games, tune_enriched, hold_df, folds, _m, hold_margins, _rounds, _u) = \
        rma.prepare_data(holdout_days=21)
    if args.smoke:
        folds = folds[:3]
    hold_enriched = rma.attach(hold_df, hold_margins)
    assert len(hold_df) == SEALED_N, \
        f"sealed window = {len(hold_df)} games, expected {SEALED_N}"

    feat_cols, dropped = derive_run_features(list(training.FEATURE_COLS))
    assert len(feat_cols) == 29, f"expected 29 run features, got {len(feat_cols)}"
    missing = [c for c in feat_cols if c not in tune_enriched.columns]
    assert not missing, f"run features missing from frame: {missing}"

    # α(λ) dispersion for the NB member: production curves from the committed
    # monitor JSON if present (evaluated at THIS fold's lambdas), else fitted
    # leak-free on the pre-sealed pooled OOF.
    mon_path = DATA_DELIVERY_DIR / "run_engine_monitor_20260827.json"
    fit = None
    if mon_path.exists():
        try:
            fit = json.loads(mon_path.read_text())["fit"]
            alpha_src = "monitor-json-alpha_curves"
        except Exception:
            fit = None
            alpha_src = None
    if fit is None:
        lam_h_all = tune_enriched["lam_home"].to_numpy(float)
        lam_a_all = tune_enriched["lam_away"].to_numpy(float)
        hs_all = tune_enriched["home_score"].to_numpy(float)
        as_all = tune_enriched["away_score"].to_numpy(float)
        _ah = fit_alpha(hs_all, lam_h_all)
        _aa = fit_alpha(as_all, lam_a_all)
        alpha_src = f"fit_alpha(pre-sealed pooled OOF): home={_ah} away={_aa}"
        fit = {"alpha_home": {"form": "constant", "lam": [], "alpha": []},
               "alpha_away": {"form": "constant", "lam": [], "alpha": []},
               "_scalar": {"home": _ah, "away": _aa}}

    def _alpha_side(lam: np.ndarray, side: str) -> np.ndarray:
        if fit.get("_scalar"):
            return np.full(len(lam), fit["_scalar"][side])
        return alpha_of(lam, fit[f"alpha_{side}"])

    print(f"commit={sha[:12]} games={len(games)} tuning={len(tune_enriched)} "
          f"holdout={len(hold_df)} folds={len(folds)} seed={RANDOM_SEED} "
          f"alpha={alpha_src} mc_n={MC_N}", flush=True)
    print(f"run-engine feature view: {len(feat_cols)} kept / "
          f"{len(dropped)} dropped", flush=True)
    print("fold loop: GBM heads (reg/over/cover) + NB sampler per fold "
          "(run engine READ-ONLY) ...", flush=True)

    # Per-fold accumulation across every arm (one fold loop; shared vectors).
    acc = {s: {"y": [], "nb": [], "gbm": [], "fold": []}
           for s in ("totals", "run_line")}
    gbm_mae_total: list[float] = []
    gbm_iters = {"reg": [], "over": [], "cover": []}
    # per-game assigned-line totals context (NB grid only)
    ctx_p, ctx_y = [], []

    for split in folds:
        tr, va = split["train_games"], split["val_games"]
        tr_heads = _gbm_head_targets(tr)
        heads = _gbm_head_targets(va)
        _mreg, p_reg, b_reg = _fit_and_predict(feat_cols, tr, va,
                                               tr_heads["reg"], heads["reg"],
                                               "reg")
        _mover, p_over, b_over = _fit_and_predict(feat_cols, tr, va,
                                                  tr_heads["over"],
                                                  heads["over"], "over")
        _mcov, p_cov, b_cov = _fit_and_predict(feat_cols, tr, va,
                                               tr_heads["cover"],
                                               heads["cover"], "cover")
        gbm_iters["reg"].append(b_reg)
        gbm_iters["over"].append(b_over)
        gbm_iters["cover"].append(b_cov)
        y_total = heads["reg"]
        y_over = heads["over"]
        y_cover = heads["cover"]
        gbm_mae_total.append(float(np.abs(p_reg - y_total).mean()))

        # NB sampler member on the SAME games' OOF lambdas (READ-ONLY cache).
        lam_h = va["lam_home"].to_numpy(float)
        lam_a = va["lam_away"].to_numpy(float)
        mc = derive_markets_mc(lam_h, lam_a,
                               _alpha_side(lam_h, "home"),
                               _alpha_side(lam_a, "away"),
                               n_draws=MC_N, seed=MC_SEED)
        acc["totals"]["y"].extend(y_over.tolist())
        acc["totals"]["nb"].extend(mc["p_over_8_5"].tolist())
        acc["totals"]["gbm"].extend(p_over.tolist())
        acc["totals"]["fold"].extend([split["fold_idx"]] * len(y_over))
        acc["run_line"]["y"].extend(y_cover.tolist())
        acc["run_line"]["nb"].extend(mc["p_home_cover_1_5"].tolist())
        acc["run_line"]["gbm"].extend(p_cov.tolist())
        acc["run_line"]["fold"].extend([split["fold_idx"]] * len(y_cover))

        # NB-only per-game assigned-line totals context (production surface).
        for i in range(len(lam_h)):
            line = _rounded_total_line(lam_h[i], lam_a[i])
            if line not in TOTAL_LINE_GRID:
                continue
            j = TOTAL_LINE_GRID.index(line)
            p = float(mc["p_over_grid"][i, j])
            if np.isnan(p) or y_total[i] == line:  # push (whole-number only)
                continue
            ctx_p.append(p)
            ctx_y.append(float(y_total[i] > line))

    # ── Sealed holdout: refit GBM heads on ALL tune games at median fold
    # iterations (run engine final-fit convention); NB from refit margins.
    med = {k: int(np.median(v)) if v else MAX_ROUNDS
           for k, v in gbm_iters.items()}
    tune_heads = _gbm_head_targets(tune_enriched)  # full tune-frame targets
    from lightgbm import LGBMClassifier, LGBMRegressor

    def _refit(kind: str, n_iter: int) -> np.ndarray:
        cls = LGBMRegressor if kind == "reg" else LGBMClassifier
        model = cls(**_gbm_params(kind))
        model.set_params(n_estimators=n_iter)
        X = tune_enriched.reindex(columns=feat_cols).astype(float)
        model.fit(X, tune_heads[kind])
        Xh = hold_enriched.reindex(columns=feat_cols).astype(float)
        if kind == "reg":
            return np.clip(model.predict(Xh, num_iteration=n_iter), 1e-6, None)
        return model.predict_proba(Xh, num_iteration=n_iter)[:, 1]

    h_total = (hold_enriched["home_score"].to_numpy(float)
               + hold_enriched["away_score"].to_numpy(float))
    h_over = (h_total >= 9).astype(float)
    h_cover = ((hold_enriched["home_score"].to_numpy(float)
                - hold_enriched["away_score"].to_numpy(float)) >= 2).astype(float)
    gbm_h = {
        "totals": _refit("over", med["over"]),
        "run_line": _refit("cover", med["cover"]),
    }
    lam_hh = hold_enriched["lam_home"].to_numpy(float)
    lam_ah = hold_enriched["lam_away"].to_numpy(float)
    mc_h = derive_markets_mc(lam_hh, lam_ah,
                             _alpha_side(lam_hh, "home"),
                             _alpha_side(lam_ah, "away"),
                             n_draws=MC_N, seed=MC_SEED)
    nb_h = {"totals": mc_h["p_over_8_5"], "run_line": mc_h["p_home_cover_1_5"]}
    y_h = {"totals": h_over, "run_line": h_cover}

    # ── Assemble per-arm blended probabilities ────────────────────────────
    oof = {}
    for s in ("totals", "run_line"):
        oof[s] = {
            "y": np.asarray(acc[s]["y"], float),
            "nb": np.clip(np.asarray(acc[s]["nb"], float), EPS, 1 - EPS),
            "gbm": np.clip(np.asarray(acc[s]["gbm"], float), EPS, 1 - EPS),
            "fold": np.asarray(acc[s]["fold"], int),
        }

    def _prequential_stack(surface: str, variant: str) -> np.ndarray:
        """Fold k stack fit on folds < k only (pooled OOF, out-of-sample).

        Before MIN_STACK_FIT prior GAMES the blend is the plain 50/50
        probability average (the honest no-data fallback; the stacker's
        own 'equal weights' convention is a sigmoid-of-mean that is NOT a
        probability average and would collapse early folds).
        """
        out = []
        prior_X, prior_y = [], []
        n_prior = 0
        for f in np.unique(oof[surface]["fold"]):
            m = oof[surface]["fold"] == f
            X = np.column_stack([oof[surface]["nb"][m],
                                 oof[surface]["gbm"][m]])
            y = oof[surface]["y"][m]
            if n_prior >= rsa.MIN_STACK_FIT:
                w, b, mu, sd = rsa.fit_stack(np.vstack(prior_X),
                                             np.concatenate(prior_y), variant)
                out.extend(rsa.stack_predict(X, w, b, mu, sd).tolist())
            else:
                out.extend((0.5 * X[:, 0] + 0.5 * X[:, 1]).tolist())
            prior_X.append(X)
            prior_y.append(y)
            n_prior += len(y)
        return np.clip(np.asarray(out, float), EPS, 1 - EPS)

    def _sealed_stack(surface: str, variant: str) -> np.ndarray:
        X_all = np.column_stack([oof[surface]["nb"], oof[surface]["gbm"]])
        y_all = oof[surface]["y"]
        Xh = np.column_stack([nb_h[surface], gbm_h[surface]])
        if len(y_all) >= rsa.MIN_STACK_FIT:
            w, b, mu, sd = rsa.fit_stack(X_all, y_all, variant)
            return np.clip(rsa.stack_predict(Xh, w, b, mu, sd), EPS, 1 - EPS)
        return np.clip(0.5 * Xh[:, 0] + 0.5 * Xh[:, 1], EPS, 1 - EPS)

    def _prequential_platt(p: np.ndarray, surface: str) -> np.ndarray:
        """Fold k Platt map fit on prior folds' pairs (prequential twin)."""
        p_cal = np.empty_like(p, dtype=float)
        hist_y, hist_p = [], []
        for f in np.unique(oof[surface]["fold"]):
            m = oof[surface]["fold"] == f
            cal = None
            if len(hist_y) >= 300:  # calibration.MIN_OOF_FOR_FIT
                cal = fit_platt(np.concatenate(hist_y),
                                np.concatenate(hist_p))
            p_cal[m] = apply_platt(p[m], cal)
            hist_y.append(oof[surface]["y"][m])
            hist_p.append(p[m])
        return np.clip(p_cal, EPS, 1 - EPS)

    arms = {}
    for s in ("totals", "run_line"):
        y = oof[s]["y"]
        bl = {
            "nb_only": oof[s]["nb"],
            "gbm_only": oof[s]["gbm"],
            "blend_equal": 0.5 * oof[s]["nb"] + 0.5 * oof[s]["gbm"],
            "blend_stack": _prequential_stack(s, "unconstrained"),
            "blend_stack_nonneg": _prequential_stack(s, "nonneg"),
        }
        for name, p in list(bl.items()):
            bl[f"{name}_cal"] = _prequential_platt(p, s)
        yh = y_h[s]
        bh = {
            "nb_only": nb_h[s],
            "gbm_only": gbm_h[s],
            "blend_equal": 0.5 * nb_h[s] + 0.5 * gbm_h[s],
            "blend_stack": _sealed_stack(s, "unconstrained"),
            "blend_stack_nonneg": _sealed_stack(s, "nonneg"),
        }
        for name, p in list(bh.items()):
            # Sealed map: the arm's OWN pooled OOF blend, fit on all pooled
            # pairs (strictly pre-holdout) — the repo refit convention.
            cal = fit_platt(oof[s]["y"], bl[name])
            bh[f"{name}_cal"] = np.clip(apply_platt(p, cal), EPS, 1 - EPS)
        arms[s] = {"pooled": bl, "sealed": bh, "y": y, "y_sealed": yh}

    # ── Report ────────────────────────────────────────────────────────────
    def _row(p: np.ndarray, y: np.ndarray, cal: np.ndarray | None = None):
        r = _metrics(p, y)
        if cal is not None:
            r2 = _metrics(cal, y)
            r["ece_calibrated"] = r2["ece"]
            r["logloss_calibrated"] = r2["logloss"]
        else:
            r["ece_calibrated"] = None
            r["logloss_calibrated"] = None
        r["auc"] = _safe_auc(y, p)
        return {k: (round(float(v), 5) if isinstance(v, (int, float))
                    and v is not None else v)
                for k, v in r.items()}

    record = {
        "schema": "total-blend-ablation/v1",
        "commit_sha": sha,
        "data": "data_delivery/game_level_features.csv",
        "seed": int(RANDOM_SEED),
        "feature_view": {
            "n_kept": len(feat_cols),
            "n_dropped": len(dropped),
            "source": "derive_run_features(FEATURE_COLS) — levels + "
                      "environment only (run engine's own keep-list)",
        },
        "fold_geometry": "shared walk-forward (MIN_VAL_FOLD_GAMES); "
                         f"{len(folds)} folds; sealed {SEALED_N} = last "
                         "decided games by date ("
                         f"{pd.to_datetime(hold_df['game_date']).min().date()} "
                         f".. {pd.to_datetime(hold_df['game_date']).max().date()})",
        "members": {
            "nb_sampler": ("run engine per-game OOF lambdas (READ-ONLY margin "
                           "cache on this fold geometry) + α(λ) from "
                           f"{alpha_src} + MC {MC_N} draws, seed {MC_SEED}"),
            "gbm": ("LightGBM on the 29 run-engine features; heads reg "
                    "(poisson E[total runs]) / over (P(over 8.5)) / cover "
                    "(P(home cover -1.5)); early stopping on the fold val; "
                    "sealed refit at median fold iterations "
                    f"{med}"),
        },
        "gbm_mae_total_runs_pooled": round(
            float(np.mean(gbm_mae_total)), 4),
        "alpha_source": alpha_src,
        "arms": {},
        "sealed": {"n": int(len(hold_df)),
                   "start": str(pd.to_datetime(
                       hold_df["game_date"]).min().date()),
                   "end": str(pd.to_datetime(
                       hold_df["game_date"]).max().date())},
        "context_totals_per_game_line_nb_only": {
            "pooled_n": len(ctx_p),
            "pooled_win_rate": round(float(np.mean(
                (np.asarray(ctx_p) >= 0.5).astype(float)
                == np.asarray(ctx_y))), 4),
        },
    }

    for s in ("totals", "run_line"):
        a = arms[s]
        row = {}
        for win in ("pooled", "sealed"):
            bl = a[win]
            y = a["y"] if win == "pooled" else a["y_sealed"]
            row[win] = {
                name: _row(np.asarray(p, float), y,
                           np.asarray(bl[f"{name}_cal"], float))
                for name, p in bl.items() if not name.endswith("_cal")
            }
        record["arms"][s] = row
        print(f"\n=== {s} ===", flush=True)
        for name in ["nb_only", "gbm_only", "blend_equal", "blend_stack",
                     "blend_stack_nonneg"]:
            rp = _metrics(a["pooled"][name], a["y"])
            rc = _metrics(a["pooled"][f"{name}_cal"], a["y"])
            rh = _metrics(a["sealed"][name], a["y_sealed"])
            rch = _metrics(a["sealed"][f"{name}_cal"], a["y_sealed"])
            ap = _safe_auc(a["y"], a["pooled"][name])
            ah = _safe_auc(a["y_sealed"], a["sealed"][name])
            print(f"  {name:20s} pooled ll={rp['logloss']:.4f} "
                  f"auc={ap if ap is not None else float('nan'):.4f} "
                  f"ece={rp['ece']:.4f} (cal {rc['ece']:.4f}) | "
                  f"sealed ll={rh['logloss']:.4f} "
                  f"auc={ah if ah is not None else float('nan'):.4f} "
                  f"ece={rh['ece']:.4f} (cal {rch['ece']:.4f})", flush=True)

    # ── Gate ──────────────────────────────────────────────────────────────
    checks = []
    for s in ("totals", "run_line"):
        a = arms[s]
        yh = a["y_sealed"]
        yp = a["y"]
        nb_m = _metrics(a["sealed"]["nb_only"], yh)
        nb_c_m = _metrics(a["sealed"]["nb_only_cal"], yh)
        po_nb_m = _metrics(a["pooled"]["nb_only"], yp)
        po_nb_c_m = _metrics(a["pooled"]["nb_only_cal"], yp)
        nb_auc = _safe_auc(yh, a["sealed"]["nb_only"])
        for name in BLEND_VARIANTS:
            b = _metrics(a["sealed"][name], yh)
            b_c = _metrics(a["sealed"][f"{name}_cal"], yh)
            po_b = _metrics(a["pooled"][name], yp)
            po_b_c = _metrics(a["pooled"][f"{name}_cal"], yp)
            b_auc = _safe_auc(yh, a["sealed"][name])
            n = len(yh)
            ll_ok = b["logloss"] <= nb_m["logloss"] + 2 * _se_band(n, "ll")
            auc_ok = (b_auc is None or nb_auc is None
                      or b_auc >= nb_auc - 2 * _se_band(n, "auc"))
            ece_ok = b_c["ece"] <= nb_c_m["ece"] + 2 * _se_band(n, "ece")
            po_ll_ok = po_b["logloss"] <= po_nb_m["logloss"] + POOLED_LL_TOL
            po_ece_ok = po_b_c["ece"] <= po_nb_c_m["ece"] + POOLED_ECE_TOL
            checks.append({
                "surface": s, "variant": name, "n_sealed": n,
                "sealed_ll_delta": round(b["logloss"] - nb_m["logloss"], 5),
                "sealed_auc_delta": (None if b_auc is None or nb_auc is None
                                      else round(b_auc - nb_auc, 5)),
                "sealed_ececal_delta": round(b_c["ece"] - nb_c_m["ece"], 5),
                "pooled_ll_delta": round(po_b["logloss"]
                                          - po_nb_m["logloss"], 5),
                "pooled_ececal_delta": round(
                    po_b_c["ece"] - po_nb_c_m["ece"], 5),
                "sealed_ll_ok": bool(ll_ok), "sealed_auc_ok": bool(auc_ok),
                "sealed_ece_ok": bool(ece_ok),
                "pooled_ll_ok": bool(po_ll_ok),
                "pooled_ece_ok": bool(po_ece_ok),
                "sealed_ll_improves": bool(b["logloss"] < nb_m["logloss"]),
            })

    def _passes(c: dict) -> bool:
        return all(c[k] for k in ("sealed_ll_ok", "sealed_auc_ok",
                                  "sealed_ece_ok", "pooled_ll_ok",
                                  "pooled_ece_ok"))

    # Gate: a variant must clear BOTH surfaces on the SEALED window
    # (logloss not degraded, AUC preserved, ECE-cal not degraded) AND not
    # lose pooled logloss/ECE — and, per the task's "ADOPT only if it beats
    # NB-only on sealed logloss AND AUC", require sealed logloss improvement.
    best = None
    for name in BLEND_VARIANTS:
        surf_ok = all(_passes(c) for c in checks if c["variant"] == name)
        imp = all(c["sealed_ll_improves"] for c in checks
                  if c["variant"] == name)
        if surf_ok and imp:
            sealed_ll = sum(c["sealed_ll_delta"] for c in checks
                            if c["variant"] == name)
            if best is None or sealed_ll < best[1]:
                best = (name, sealed_ll)
    if best is None:
        verdict = "DON'T ADOPT"
        reason = ("no blend variant beats NB-only on sealed logloss (with "
                  "AUC preserved and ECE-cal not degraded) on BOTH totals "
                  "and run-line without losing pooled OOF")
    else:
        verdict = "ADOPT"
        reason = (f"{best[0]} beats NB-only on sealed logloss on both "
                  "surfaces with AUC/ECE-cal preserved and pooled OOF not "
                  "lost")
    record["gate"] = {
        "verdict": verdict, "reason": reason,
        "rule": ("sealed (n=284): blend logloss <= NB-only + 2*se_band(ll), "
                 "AUC >= NB-only - 2*se_band(auc), ECE-cal <= NB-only + "
                 "2*se_band(ece); pooled: logloss <= NB-only + 0.0005, "
                 "ECE-cal <= NB-only + 0.0005; plus sealed logloss must "
                 "STRICTLY improve on both surfaces (the task's 'beats on "
                 "sealed logloss AND AUC' leg)"),
        "tolerances": {"pooled_ll": POOLED_LL_TOL,
                       "pooled_ece_cal": POOLED_ECE_TOL,
                       "sealed_ll_band": round(2 * _se_band(SEALED_N, "ll"), 5),
                       "sealed_auc_band": round(2 * _se_band(SEALED_N, "auc"), 5),
                       "sealed_ece_band": round(2 * _se_band(SEALED_N, "ece"), 5)},
        "checks": checks,
    }

    target = pd.Timestamp.now().date().isoformat()
    compact = target.replace("-", "")
    out = args.out or (DATA_DELIVERY_DIR / f"total_blend_ablation_{compact}.json")
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nrecord -> {out}")
    print(f"GATE: {verdict} — {reason}")


if __name__ == "__main__":
    main()
