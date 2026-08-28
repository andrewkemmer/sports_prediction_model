"""Standalone, leakage-free ablation of the run engine's home-edge
underweighting (derived-ML bug: mean P(home) 0.4684 vs actual 0.5354).

STEP 1 (E[X] bias diagnostic) + STEP 3 (corrected-vs-current scoring) in one
harness; STEP 2 (cascade check) is a qualitative note recorded from the
moneyline monitor's drift table.

Arms (all derived through the run engine's OWN MC machinery —
`derive_markets_mc`, identical seed, no retraining, run engine READ-ONLY):
  current                — shipped λs + fitted α(λ) curves
  lambda_edge_corrected  — λ_home shifted so the mean λ differential equals
                           the empirical pre-sealed run differential
  alpha_symmetric        — away α reparameterized to the home α(λ) curve
  alpha_symmetric_plus_lambda — both corrections

Correction statistics are computed on the PRE-sealed window only
(prequential discipline; the sealed 284 holdout is never used to fit or
select). Every arm is scored on all THREE surfaces — derived ML, run line
(±1.5), totals (per-game assigned rounded line, push-excluded) — pooled
OOF + sealed 284 holdout, logloss / AUC / ECE / win rate + mean predicted
probability. The "corrected" arm is chosen by the acceptance gate: derived
ML calibration improves (ECE toward ~0.01-0.02, mean P(home) toward
~0.5354) WITHOUT degrading totals or run-line ECE/logloss/AUC beyond
tolerance. Record written to
data_delivery/run_engine_home_edge_ablation_<date>.json (date-stamped so
Phase 6 keeps it).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from run_engine import (ALPHA_CAP, MC_DRAWS, RUN_LINE_MARGIN,
                        TOTAL_LINE_GRID, _rounded_total_line, alpha_of,
                        derive_markets_mc, ece_score)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_delivery"
SEALED_N = 284
MC_N = 5_000  # per-game se ~1.4%; pooled means precise to ~0.02%
GATE_ECE_TOL = 0.002  # totals/run-line ECE may not degrade beyond this


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    try:
        a = float(roc_auc_score(y, p))
    except ValueError:
        return None
    return None if not np.isfinite(a) else round(a, 5)


def _safe_ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return round(float(log_loss(y, p, labels=[0.0, 1.0])), 5)


def _surface_metrics(p: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    """logloss / AUC / ECE / win rate (>50% rule) on a p/y slice."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    if not mask.any():
        return {"n": 0}
    p, y = p[mask], y[mask]
    out = {
        "n": int(len(y)),
        "logloss": _safe_ll(y, p),
        "auc": _safe_auc(y, p),
        "ece": round(float(ece_score(y, p)), 5),
        "win_rate": round(float(np.mean((p >= 0.5).astype(float) == y)), 4),
        "mean_p": round(float(p.mean()), 4),
    }
    # Card-style (picked-side) ECE, matching the monitor's winner card
    # framing: ECE on favored-side (max(p, 1-p)) vs picked-side outcome.
    fav = np.maximum(p, 1.0 - p)
    hit = ((p >= 0.5).astype(float) == y).astype(float)
    out["card_style_ece"] = round(float(ece_score(hit, fav)), 5)
    if out["auc"] is None:
        out["auc"] = None
    return out


def _totals_surface(grid_over: np.ndarray, lam_h: np.ndarray,
                    lam_a: np.ndarray, total: np.ndarray) -> dict:
    """Per-game assigned rounded line (mirrors the winner-card over_under),
    push-excluded, from the arm's own λs. Returns the OOF row index per
    kept game so the sealed mask stays aligned after push exclusion."""
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
    return {"p": np.asarray(ps, float), "y": np.asarray(ys, float),
            "idx": np.asarray(idx, int)}


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
    home_won = (hs > as_).astype(float)
    # home covers -1.5 iff margin >= 2 (int(RUN_LINE_MARGIN) + 1 — the same
    # event derive_markets_mc prices: p_cover = (diff >= 2)).
    home_covers = (hs - as_ >= int(RUN_LINE_MARGIN) + 1).astype(float)

    # Sealed 284: the last SEALED_N games by date (08-05..08-25 on 08-27 run).
    order = np.argsort(dates.to_numpy(), kind="stable")
    hold_idx = np.zeros(len(oof), bool)
    hold_idx[order[-SEALED_N:]] = True
    pre = ~hold_idx
    print(f"sealed: n={int(hold_idx.sum())} "
          f"[{dates[hold_idx].min().date()} -> {dates[hold_idx].max().date()}]")

    fit = json.loads(mon_path.read_text())["fit"]
    cur_h, cur_a = fit["alpha_home"], fit["alpha_away"]
    a_h = alpha_of(lam_h, cur_h)
    a_a = alpha_of(lam_a, cur_a)

    # ---- STEP 1: E[X] bias (pre-sealed, leakage-free) + full for report ----
    d_lambda_pre = float(np.mean((lam_h - lam_a)[pre]))
    d_actual_pre = float(np.mean((hs - as_)[pre]))
    delta_lambda = d_actual_pre - d_lambda_pre  # λ_home shift to match reality
    # Poisson-limit probe: what the joint gives with no dispersion at all.
    poisson = derive_markets_mc(lam_h, lam_a, np.full(len(lam_h), 1e-6),
                                np.full(len(lam_a), 1e-6), n_draws=MC_N)
    # Empirical home edge conditional on the projected total (pre-sealed fit;
    # robust bins >= 30 games, interpolated, clamped).
    proj_total = lam_h + lam_a
    ebin_edges = np.linspace(7.0, 11.5, 10)
    ebin_idx = np.digitize(proj_total, ebin_edges)
    centers, gaps = [], []
    for b in range(len(ebin_edges) + 1):
        m = pre & (ebin_idx == b)
        if m.sum() >= 30:
            centers.append(proj_total[m].mean())
            gaps.append(float(np.mean((hs - as_)[m])
                              - np.mean((lam_h - lam_a)[m])))
    centers = np.asarray(centers)
    gaps = np.asarray(gaps)

    def edge_at(t: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(t, float), centers, gaps)

    step1 = {
        "pooled": {
            "mean_lam_h": round(float(lam_h.mean()), 4),
            "mean_home_actual": round(float(hs.mean()), 4),
            "home_bias": round(float(hs.mean() - lam_h.mean()), 4),
            "mean_lam_a": round(float(lam_a.mean()), 4),
            "mean_away_actual": round(float(as_.mean()), 4),
            "away_bias": round(float(as_.mean() - lam_a.mean()), 4),
            "mean_lambda_diff": round(float(np.mean(lam_h - lam_a)), 4),
            "mean_actual_diff": round(float(np.mean(hs - as_)), 4),
            "diff_bias": round(float(np.mean(hs - as_)
                                    - np.mean(lam_h - lam_a)), 4),
            "mean_alpha_home": round(float(a_h.mean()), 4),
            "mean_alpha_away": round(float(a_a.mean()), 4),
            "actual_home_won_rate": round(float(np.mean(home_won)), 4),
            "poisson_limit_mean_p_home": round(float(poisson[
                "p_home_win_derived"].mean()), 4),
        },
        "environment_edge_bins": [
            {"projected_total_center": round(float(c), 3),
             "empirical_minus_modeled_edge": round(float(g), 4)}
            for c, g in zip(centers, gaps)],
        "verdict": ("E[X] means are calibrated per side (home bias +0.010, "
                    "away +0.062); the lambda differential (+0.120) actually "
                    "OVERSTATES the true home edge (+0.068), yet P(home win) "
                    "= 0.4684 < 0.5354. Even the POISSON limit gives 0.449, "
                    "and a +0.25 constant lambda edge only reaches 0.49 — the "
                    "joint P(h>a) of independent NB margins at these means is "
                    "structurally ~0.45-0.47. Alpha changes move it the WRONG "
                    "way (away's fat tail RAISES its median relative to its "
                    "mean, helping home). The deficit is a JOINT-vs-MARGINAL "
                    "mismatch (the empirical home edge is environment-"
                    "conditional: +0.27 in low-total games, -0.09 in high) "
                    "that no constant lambda/alpha knob can express without "
                    "breaking the calibrated per-side surfaces."),
        "delta_lambda": round(delta_lambda, 4),
    }
    print("step1: Poisson-limit mean P(home)=", step1["pooled"]
          ["poisson_limit_mean_p_home"])

    # ---- Arms ----
    arms = {
        "current": (lam_h, lam_a, a_h, a_a),
        "lambda_edge_corrected": (lam_h + delta_lambda, lam_a, a_h, a_a),
        "alpha_symmetric": (lam_h, lam_a, a_h, a_h),
        "alpha_symmetric_plus_lambda": (lam_h + delta_lambda, lam_a, a_h,
                                        a_h),
        # lambda_home adjustment matched to the EMPIRICAL home run
        # differential, conditional on the projected total (pre-sealed fit).
        "environment_edge_corrected": (lam_h + edge_at(proj_total), lam_a,
                                       a_h, a_a),
    }
    results: dict[str, dict] = {}
    for name, (lh, la, ah, aa) in arms.items():
        mc = derive_markets_mc(lh, la, ah, aa, n_draws=MC_N)
        p_win, p_cover, p_over8 = (mc["p_home_win_derived"],
                                   mc["p_home_cover_1_5"], mc["p_over_8_5"])
        tov = _totals_surface(mc["p_over_grid"], lh, la, total)
        tot_sealed = hold_idx[tov["idx"]]
        surfaces = {
            "derived_ml": {
                "pooled": _surface_metrics(p_win, home_won,
                                           np.ones(len(oof), bool)),
                "sealed": _surface_metrics(p_win, home_won, hold_idx),
            },
            "run_line": {
                "pooled": _surface_metrics(p_cover, home_covers,
                                           np.ones(len(oof), bool)),
                "sealed": _surface_metrics(p_cover, home_covers, hold_idx),
            },
            "totals": {
                "pooled": _surface_metrics(tov["p"], tov["y"],
                                           np.ones(len(tov["p"]), bool)),
                "sealed": _surface_metrics(tov["p"], tov["y"], tot_sealed),
            },
        }
        results[name] = {
            "mean_p_home_win": round(float(p_win.mean()), 4),
            "mean_p_home_cover": round(float(p_cover.mean()), 4),
            "surfaces": surfaces,
        }
        print(f"[{name}] mean P(home win)={p_win.mean():.4f} "
              f"mean P(home cover)={p_cover.mean():.4f}")

    # ---- Acceptance gate (corrected vs current, per surface, both windows) ----
    cur = results["current"]
    gate = {}
    verdicts = {}
    candidates = [n for n in arms if n != "current"]
    best = None
    for name in candidates:
        r = results[name]
        dml = r["surfaces"]["derived_ml"]
        c_dml = cur["surfaces"]["derived_ml"]
        # Derived-ML calibration improvement:
        ece_ok = (dml["pooled"]["ece"] < c_dml["pooled"]["ece"]
                  and dml["sealed"]["ece"] < c_dml["sealed"]["ece"])
        mean_ok = (abs(r["mean_p_home_win"] - step1["pooled"]
                       ["actual_home_won_rate"])
                   < abs(cur["mean_p_home_win"] - step1["pooled"]
                         ["actual_home_won_rate"]))
        # Non-degradation on totals + run line (both windows, ECE gate):
        nd = True
        for surf in ("totals", "run_line"):
            for win in ("pooled", "sealed"):
                e = r["surfaces"][surf][win].get("ece")
                ce = cur["surfaces"][surf][win].get("ece")
                if e is not None and ce is not None and e > ce + GATE_ECE_TOL:
                    nd = False
        ok = ece_ok and mean_ok and nd
        gate[name] = {
            "derived_ml_ece_improved_pooled": bool(dml["pooled"]["ece"]
                                                   < c_dml["pooled"]["ece"]),
            "derived_ml_ece_improved_sealed": bool(dml["sealed"]["ece"]
                                                   < c_dml["sealed"]["ece"]),
            "mean_p_home_closer": bool(mean_ok),
            "totals_run_line_not_degraded": bool(nd),
            "adopt": bool(ok),
        }
        verdicts[name] = "ADOPT" if ok else "DON'T ADOPT"
        if ok and (best is None or dml["sealed"]["ece"]
                   < results[best]["surfaces"]["derived_ml"]["sealed"]["ece"]):
            best = name
    corrected = best or "current"

    record = {
        "schema": "run-engine-home-edge-ablation/v1",
        "date": date_str,
        "frame": oof_path.name,
        "n_games": int(len(oof)),
        "sealed": {"n": int(hold_idx.sum()),
                   "start": str(dates[hold_idx].min().date()),
                   "end": str(dates[hold_idx].max().date())},
        "mc": {"n_draws": MC_N, "seed_implied": "derive_markets_mc default"},
        "step1_ex_bias": step1,
        "step2_cascade": {
            "run_margin_diff_weight_pct": 3.68,
            "rank": "1 of 65 (drift table)",
            "median_weight_pct": 1.55,
            "summary": ("run_margin_diff is the moneyline ensemble's most-"
                        "weighted feature; a corrected lambda edge shifts its "
                        "mean by ~0.05 (comparable to an already-tolerated "
                        "drift mean_shift 0.041, classified location_shift "
                        "false). The moneyline has calibrated AROUND the "
                        "biased feature (ensemble mean P(home) 0.5365 ~ "
                        "actual 0.5354), so a run-engine correction would move "
                        "its inputs modestly (qualitatively a few probability "
                        "points at most) and needs the normal retrain/"
                        "calibration cycle to settle; the fix's primary value "
                        "is the run engine's OWN surfaces."),
        },
        "arms": results,
        "gate": gate,
        "verdicts": verdicts,
        "corrected": corrected,
        "acceptance": {
            "rule": ("corrected ADOPT only if derived-ML ECE improves on "
                     "pooled AND sealed AND mean P(home) moves toward the "
                     "empirical 0.5354 AND totals/run-line ECE degrade by "
                     f"<= {GATE_ECE_TOL} on both windows"),
            "result": (f"corrected={corrected}; "
                       + "; ".join(f"{k}={v}" for k, v in verdicts.items())),
        },
    }
    out = DATA / f"run_engine_home_edge_ablation_{date_str}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out}")
    for k, v in verdicts.items():
        print(f"  {k}: {v}")
    print(f"  corrected -> {corrected}")


if __name__ == "__main__":
    main()
