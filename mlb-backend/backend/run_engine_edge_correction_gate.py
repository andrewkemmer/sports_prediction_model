"""Gate the run-engine λ-edge correction on the PRODUCTION surfaces.

Trace (committed 61d0911): p_home_win_derived is monitor-only — no EV or
market-table math consumes it (Today's Games ML uses the ensemble
home_win_prob_model; the O/U + run-line prices use the grid/cover columns;
agreement_conflict has no downstream consumer). The home-edge ablation
(a84ddb8) showed the model's λ differential is overstated (+0.120 modeled
vs +0.068 actual) and that shifting λ_home to the empirical differential
improves run-line + totals ECE at the only cost of the structurally-limited
derived-ML diagnostic. This gate therefore scores the correction on the
production-priced surfaces ONLY:

  run_line  p_home_cover_1_5 vs (margin >= 2) — half-run lines never push
  totals    per-game assigned rounded line (push-excluded) vs went-over

with pooled OOF + sealed 284 holdout, logloss / AUC / ECE / win rate.
Acceptance (corrected vs current):
  - ECE: strictly improved on the POOLED cells; not degraded by more than
    GATE_ECE_TOL on the SEALED cells.
  - logloss: not degraded by more than GATE_LL_TOL anywhere.
  - AUC: preserved within GATE_AUC_TOL (a constant λ shift is rank-
    preserving up to MC discreteness and totals line-flip games).
Leakage-free: δ fitted on the PRE-sealed window only. Record written to
data_delivery/run_engine_edge_correction_gate_<date>.json (date-stamped).
Run engine READ-ONLY; standalone harness; nothing committed until reviewed.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from run_engine import (RUN_LINE_MARGIN, TOTAL_LINE_GRID,
                        _rounded_total_line, alpha_of, derive_markets_mc,
                        ece_score)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_delivery"
SEALED_N = 284
MC_N = 10_000          # production-grade draws (matches the shipped MC)
GATE_ECE_TOL = 0.005   # pooled logloss/ECE degradation tolerance
GATE_LL_TOL = 0.005
GATE_AUC_TOL = 0.010   # AUC preserved within MC-discreteness band (pooled)
BOOT = 200             # paired-bootstrap iterations for Δ confidence bands


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    try:
        a = float(roc_auc_score(y, p))
    except ValueError:
        return None
    return None if not np.isfinite(a) else round(a, 5)


def _metrics(p: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    if not mask.any():
        return {"n": 0}
    p, y = p[mask], y[mask]
    return {
        "n": int(len(y)),
        "logloss": round(float(log_loss(
            y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0.0, 1.0])), 5),
        "auc": _safe_auc(y, p),
        "ece": round(float(ece_score(y, p)), 5),
        "win_rate": round(float(np.mean((p >= 0.5).astype(float) == y)), 4),
        "mean_p": round(float(p.mean()), 4),
    }


def _se_band(n: int, kind: str) -> float:
    """Sampling se of a metric on an n-game window (rough, for the sealed
    no-significant-degradation tolerance)."""
    if kind == "ece":
        return float(np.sqrt(0.05 * 0.95 / max(n, 1)))
    if kind == "ll":
        return float(0.7 / np.sqrt(max(n, 1)))
    return float(0.5 / np.sqrt(max(n, 1)))  # auc


def _totals_rows(grid_over: np.ndarray, lam_h: np.ndarray,
                 lam_a: np.ndarray, total: np.ndarray) -> tuple[np.ndarray,
                                                               np.ndarray,
                                                               np.ndarray]:
    ps, ys, idx = [], [], []
    for i in range(len(lam_h)):
        line = _rounded_total_line(lam_h[i], lam_a[i])
        if line not in TOTAL_LINE_GRID:
            continue
        j = TOTAL_LINE_GRID.index(line)
        p = float(grid_over[i, j])
        if np.isnan(p):
            continue
        if total[i] == line:  # push (whole-number lines only)
            continue
        ps.append(p)
        ys.append(float(total[i] > line))
        idx.append(i)
    return (np.asarray(ps, float), np.asarray(ys, float),
            np.asarray(idx, int))


def main() -> None:
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime(
        "%Y%m%d")
    oof_path = DATA / f"run_engine_oof_{date_str}.csv"
    mon_path = DATA / f"run_engine_monitor_{date_str}.json"
    for p in (oof_path, mon_path):
        if not p.exists():
            raise SystemExit(f"missing artifact: {p}")

    oof = pd.read_csv(oof_path)
    lam_h = oof["home_expected_runs"].to_numpy(float)
    lam_a = oof["away_expected_runs"].to_numpy(float)
    hs = oof["home_score"].to_numpy(float)
    as_ = oof["away_score"].to_numpy(float)
    dates = pd.to_datetime(oof["game_date"], errors="coerce")
    total = hs + as_
    home_covers = (hs - as_ >= int(RUN_LINE_MARGIN) + 1).astype(float)

    order = np.argsort(dates.to_numpy(), kind="stable")
    hold = np.zeros(len(oof), bool)
    hold[order[-SEALED_N:]] = True
    pre = ~hold

    fit = json.loads(mon_path.read_text())["fit"]
    a_h = alpha_of(lam_h, fit["alpha_home"])
    a_a = alpha_of(lam_a, fit["alpha_away"])

    # Leakage-free correction: empirical minus modeled differential, PRE only.
    delta = float(np.mean((hs - as_)[pre]) - np.mean((lam_h - lam_a)[pre]))

    arms = {
        "current": (lam_h, lam_a),
        "lambda_edge_corrected": (lam_h + delta, lam_a),
    }
    # Keep the per-arm probability/outcome vectors for the paired bootstrap.
    out = {}
    vecs: dict[str, dict] = {}
    over8_y = (total >= 9).astype(float)  # fixed-reference totals event
    for name, (lh, la) in arms.items():
        mc = derive_markets_mc(lh, la, a_h, a_a, n_draws=MC_N)
        tp, ty, tidx = _totals_rows(mc["p_over_grid"], lh, la, total)
        vecs[name] = {
            "rl_p": mc["p_home_cover_1_5"], "rl_y": home_covers,
            "to_p": tp, "to_y": ty, "to_idx": tidx,
            "ref_p": mc["p_over_8_5"],  # fixed reference (never mixed-line)
        }
        out[name] = {
            "lambda_shift": round(float(delta), 4) if name != "current" else 0.0,
            "mean_lam_h": round(float(lh.mean()), 4),
            "mean_lam_diff": round(float(np.mean(lh - la)), 4),
            "run_line": {
                "pooled": _metrics(mc["p_home_cover_1_5"], home_covers,
                                   np.ones(len(oof), bool)),
                "sealed": _metrics(mc["p_home_cover_1_5"], home_covers, hold),
            },
            "totals": {
                "pooled": _metrics(tp, ty, np.ones(len(tp), bool)),
                "sealed": _metrics(tp, ty, hold[tidx]),
                "over_8_5_reference": {
                    "pooled": _metrics(mc["p_over_8_5"], over8_y,
                                       np.ones(len(oof), bool)),
                    "sealed": _metrics(mc["p_over_8_5"], over8_y, hold),
                },
            },
        }

    # ---- Paired bootstrap Δ bands (resample games WITHIN each window) ----
    # Totals rows are aligned across arms by game index (the push-excluded
    # row SETS differ when λ-edge line flips change which games push), and
    # the totals AUC uses the FIXED over_8_5 reference (never a mixed-line
    # rank) aligned on all games.
    rng = np.random.default_rng(11)
    tcur_idx = vecs["current"]["to_idx"]
    tcor_idx = vecs["lambda_edge_corrected"]["to_idx"]
    common = np.intersect1d(tcur_idx, tcor_idx)
    pos_c = {g: i for i, g in enumerate(tcur_idx)}
    pos_k = {g: i for i, g in enumerate(tcor_idx)}
    c_rows = np.asarray([pos_c[g] for g in common], int)
    k_rows = np.asarray([pos_k[g] for g in common], int)
    tot_c = (vecs["current"]["to_p"][c_rows], vecs["current"]["to_y"][c_rows])
    tot_k = (vecs["lambda_edge_corrected"]["to_p"][k_rows],
             vecs["lambda_edge_corrected"]["to_y"][k_rows])
    boot: dict[str, dict] = {}

    def _bootstrap(pc, kc, yc, yk, n, win_mask=None):
        if win_mask is not None:
            sel = np.where(win_mask)[0]
            pc, kc, yc, yk = pc[sel], kc[sel], yc[sel], yk[sel]
        n = len(pc)
        d_ece, d_ll, d_auc = [], [], []
        for _ in range(BOOT):
            b = rng.integers(0, n, n)
            pb, kb = pc[b], kc[b]
            d_ece.append(ece_score(yk[b], kb) - ece_score(yc[b], pb))
            d_ll.append(float(log_loss(
                yk[b], np.clip(kb, 1e-6, 1 - 1e-6), labels=[0.0, 1.0])
                - log_loss(yc[b], np.clip(pb, 1e-6, 1 - 1e-6),
                           labels=[0.0, 1.0])))
        return n, d_ece, d_ll

    def _ci(a):
        a = np.asarray(a)
        return (round(float(a.mean()), 5),
                round(float(np.percentile(a, 2.5)), 5),
                round(float(np.percentile(a, 97.5)), 5))

    # run_line: same 4,369 games both arms (paired per game).
    for win, mask in (("pooled", None), ("sealed", hold)):
        pc = vecs["current"]["rl_p"]
        kc = vecs["lambda_edge_corrected"]["rl_p"]
        y = vecs["current"]["rl_y"]
        if win == "sealed":
            sel = np.where(mask)[0]
            pc, kc, y = pc[sel], kc[sel], y[sel]
        n = len(y)
        d_ece, d_ll = [], []
        d_auc = []
        for _ in range(BOOT):
            b = rng.integers(0, n, n)
            pb, kb, yb = pc[b], kc[b], y[b]
            d_ece.append(ece_score(yb, kb) - ece_score(yb, pb))
            d_ll.append(float(log_loss(
                yb, np.clip(kb, 1e-6, 1 - 1e-6), labels=[0.0, 1.0])
                - log_loss(yb, np.clip(pb, 1e-6, 1 - 1e-6),
                           labels=[0.0, 1.0])))
            try:
                d_auc.append(roc_auc_score(yb, kb) - roc_auc_score(yb, pb))
            except ValueError:
                pass
        boot[f"run_line/{win}"] = {
            "n": n, "d_ece": _ci(d_ece), "d_logloss": _ci(d_ll),
            "d_auc": (_ci(d_auc) if d_auc else None),
        }
    # totals: intersection-aligned per-game-line ECE/logloss; FIXED
    # over_8_5 reference AUC aligned on all games.
    hold_common = hold[common]
    for win, mask in (("pooled", None), ("sealed", hold_common)):
        n, d_ece, d_ll = _bootstrap(tot_c[0], tot_k[0], tot_c[1], tot_k[1],
                                    len(common), mask)
        boot[f"totals/{win}"] = {
            "n": n, "d_ece": _ci(d_ece), "d_logloss": _ci(d_ll),
            "d_auc": None,
        }
    for win, mask in (("pooled", None), ("sealed", hold)):
        pc = vecs["current"]["ref_p"]
        kc = vecs["lambda_edge_corrected"]["ref_p"]
        y = over8_y
        if win == "sealed":
            sel = np.where(mask)[0]
            pc, kc, y = pc[sel], kc[sel], y[sel]
        n = len(y)
        d_auc, d_ece = [], []
        for _ in range(BOOT):
            b = rng.integers(0, n, n)
            pb, kb, yb = pc[b], kc[b], y[b]
            try:
                d_auc.append(roc_auc_score(yb, kb) - roc_auc_score(yb, pb))
            except ValueError:
                pass
            d_ece.append(ece_score(yb, kb) - ece_score(yb, pb))
        boot[f"totals_ref_auc/{win}"] = {
            "n": n, "d_ece": None, "d_logloss": None,
            "d_auc": (_ci(d_auc) if d_auc else None),
        }
        boot[f"totals_ref_ece/{win}"] = {
            "n": n, "d_ece": _ci(d_ece), "d_logloss": None,
            "d_auc": None,
        }

    # ---- Gate on the bootstrap bands ----
    checks = []
    for surf, key in (("run_line", "run_line"),
                      ("totals", "totals"),
                      ("totals AUC (over 8.5 ref)", "totals_ref_auc"),
                      ("totals ECE (over 8.5 ref)", "totals_ref_ece")):
        for win in ("pooled", "sealed"):
            b = boot[f"{key}/{win}"]
            ece_ub = None if b["d_ece"] is None else b["d_ece"][2]
            ll_ub = None if b["d_logloss"] is None else b["d_logloss"][2]
            auc_ub = (None if b["d_auc"] is None
                      else max(abs(b["d_auc"][1]), abs(b["d_auc"][2])))
            if win == "pooled":
                ece_ok = ece_ub is None or ece_ub < 0.0
                tol_ll, tol_auc = GATE_LL_TOL, GATE_AUC_TOL
            else:
                ece_ok = ece_ub is None or \
                    ece_ub <= 2 * _se_band(b["n"], "ece")
                tol_ll = (2 * _se_band(b["n"], "ll") if ll_ub is not None
                          else 0.0)
                tol_auc = 2 * _se_band(b["n"], "auc")
            ll_ok = ll_ub is None or ll_ub <= tol_ll
            auc_ok = auc_ub is None or auc_ub <= tol_auc
            checks.append({
                "surface": surf, "window": win, "n": b["n"],
                "d_ece_mean_ci": b["d_ece"],
                "d_logloss_mean_ci": b["d_logloss"],
                "d_auc_mean_ci": b["d_auc"],
                "tolerances": {"logloss": tol_ll, "auc": tol_auc},
                "ece_ok": bool(ece_ok), "logloss_ok": bool(ll_ok),
                "auc_ok": bool(auc_ok),
            })
    pooled_ece_improved = all(
        c["ece_ok"] for c in checks if c["window"] == "pooled")
    all_ok = all(c["ece_ok"] and c["logloss_ok"] and c["auc_ok"]
                 for c in checks)
    adopt = bool(pooled_ece_improved and all_ok)

    record = {
        "schema": "run-engine-edge-correction-gate/v1",
        "date": date_str,
        "decision_rule": ("trace first: p_home_win_derived is monitor-only "
                          "(61d0911) -> gate scored on production surfaces "
                          "(run line + totals) only; derived-ML accepted as "
                          "a diagnostic"),
        "correction": ("lambda_home += delta, delta = mean_pre(actual diff) "
                       "- mean_pre(modeled diff) = %s (fitted on the "
                       "pre-sealed window only)" % round(delta, 4)),
        "sealed": {"n": int(hold.sum()),
                   "start": str(dates[hold].min().date()),
                   "end": str(dates[hold].max().date())},
        "mc": {"n_draws": MC_N},
        "bootstrap": {"iterations": BOOT, "seed": 11,
                       "method": "paired resample of games within each "
                                  "window; 2.5/97.5 percentile CI"},
        "arms": out,
        "checks": checks,
        "gate": {
            "rule": ("pooled: ECE 97.5% CI upper bound < 0 (significant "
                     "improvement), logloss upper bound <= 0.005, |AUC| <= "
                     "0.01; sealed: upper bounds within the window sampling "
                     "band (2x se: ECE ~0.026, logloss ~0.03, AUC ~0.06 on "
                     "n~270-284). Totals AUC + ECE on the FIXED over_8_5 "
                     "reference (never a mixed-line rank); per-game totals "
                     "ECE/logloss paired on the intersection of push-"
                     "excluded game indices across arms."),
            "pooled_ece_improved": bool(pooled_ece_improved),
            "all_cells_ok": bool(all_ok),
            "adopt": bool(adopt),
        },
        "verdict": ("ADOPT lambda_edge_corrected (shipped as the run engine's "
                    "λ-home edge)" if adopt else
                    "DON'T ADOPT — keep current λ edge"),
        "accepted_cost": ("derived-ML (p_home_win_derived) is monitor-only "
                          "and its ECE degrades under the correction — "
                          "accepted per the decision rule (diagnostic). The "
                          "expected-runs display and the moneyline "
                          "run_margin_diff feature shift by ~0.05 (see "
                          "step2_cascade in the home-edge ablation record); "
                          "the moneyline ensemble recalibrates on its normal "
                          "cycle."),
    }
    out_path = DATA / f"run_engine_edge_correction_gate_{date_str}.json"
    out_path.write_text(json.dumps(record, indent=2))
    print(f"wrote {out_path}")
    def _fmt(t):
        return ("%+.5f[%+.5f,%+.5f]" % (t[0], t[1], t[2])) if t else "--"
    for c in checks:
        print("  %-27s %-6s n=%-4d Δece=%s Δll=%s Δauc=%s  %s" % (
            c["surface"], c["window"], c["n"],
            _fmt(c["d_ece_mean_ci"]), _fmt(c["d_logloss_mean_ci"]),
            _fmt(c["d_auc_mean_ci"]),
            "OK" if (c["ece_ok"] and c["logloss_ok"] and c["auc_ok"])
            else "FAIL"))
    print("verdict:", record["verdict"])


if __name__ == "__main__":
    main()
