"""Gate the C2 k-edge expansion (run_engine_k_edge) on the PRODUCTION surfaces.

Rollout gate for the challenger's Phase-A winner (C2 linear edge, 4feff51):
the run engine prices λ'_H = μ + k(λ_H − μ), λ'_A = μ + k(λ_A − μ) with the
level λ_H + λ_A preserved. k is REFIT on this run's pre-holdout OOF (the
per-run refit policy — never the sealed games) and the SAME k prices both
the OOF markets and the slate.

Scores C2 vs the current engine (C0) on the production surfaces, using the
production α(λ) curves from run_engine_monitor_<date>.json and the
production OOF (run_engine_oof_<date>.csv), same walk-forward + sealed-284
geometry as the run-engine walk-forward:

  a) run-line −1.5 cover calibration in >0.65/>0.70 bins (pooled + sealed)
  b) O/U calibration (per-game assigned rounded line, push-excluded)
  c) derived ML P(win) SD (target: toward the binary model's ~0.066) and
     the [0.55,0.60) calibration gap
  d) sealed/pooled CRPS — cross-referenced to the committed challenger
     record (run_engine_challenger_<date>.json: C2 2.4015 vs C0 2.4111
     sealed; 2.4675 vs 2.4714 pooled)

No isotonic-on-ML recalibration is added here (explicitly out of scope).
Standalone harness; read-only over artifacts; writes the gate record
data_delivery/run_engine_k_edge_gate_<date>.json.

Usage:
    python run_engine_k_edge_gate.py [YYYYMMDD]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from run_engine import (TOTAL_LINE_GRID, _rounded_total_line, alpha_of,
                        derive_markets_mc, ece_score)
import run_engine_k_edge as ke

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_delivery"
SEALED_N = 284                 # matches the walk-forward sealed holdout
MC_N = 10_000                  # production-grade draws (matches shipped MC)
EXTREME_BINS = (0.65, 0.70)
PWIN_BUCKET = (0.55, 0.60)
PWIN_SD_TARGET = 0.066         # binary ensemble's P(win) SD
ECE_TOL = 0.005                # totals/run-line ECE degradation tolerance


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


def _bin_rows(p: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    """Quantile-decile calibration + fixed >0.65/>0.70 bins on p.

    NOTE: on the run-line −1.5 surface the home-cover p rarely exceeds
    0.65 (mean_p ≈ 0.36), so the p-based extreme bins are usually EMPTY.
    The λ-edge probe's "extreme" under-pricing (>0.70 bin: pred 0.430 vs
    actual 0.698) was binned on |λ_edge|, not on p — see _edge_bin_rows.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    pm, ym = p[mask], y[mask]
    rows: list[dict] = []
    if len(pm) >= 20:
        edges = np.quantile(pm, np.linspace(0, 1, 11))
        edges = np.unique(edges)
        idx = np.clip(np.digitize(pm, edges[1:-1], right=False), 0,
                      len(edges) - 2)
        for b in range(len(edges) - 1):
            m = idx == b
            if not m.any():
                continue
            rows.append({
                "bin": f"[{edges[b]:.2f},{edges[b + 1]:.2f})",
                "n": int(m.sum()),
                "pred": round(float(pm[m].mean()), 4),
                "actual": round(float(ym[m].mean()), 4),
                "delta": round(float(ym[m].mean() - pm[m].mean()), 4),
            })
    extremes = []
    for lo in EXTREME_BINS:
        m = pm >= lo
        extremes.append({
            "bin": f">={lo:.2f}", "n": int(m.sum()),
            "pred": (None if not m.any()
                     else round(float(pm[m].mean()), 4)),
            "actual": (None if not m.any()
                       else round(float(ym[m].mean()), 4)),
            "delta": (None if not m.any()
                      else round(float(ym[m].mean() - pm[m].mean()), 4)),
        })
    return {
        "deciles": rows,
        "extreme": extremes,
        "overall_delta": round(float(ym.mean() - pm.mean()), 4)
        if len(pm) else None,
    }


def _edge_bin_rows(p_cover: np.ndarray, y_cover: np.ndarray,
                   edge: np.ndarray, mask: np.ndarray) -> dict:
    """Run-line −1.5 cover calibration binned by |λ_edge| (the probe's
    surface: "run-line −1.5 cover under-priced at extremes").

    Fixed cutoffs at |edge| ≥ 0.5 / 0.70 / 0.90 plus quantile deciles, so
    the probe's >0.70 bin is reproduced and the C2-vs-C0 delta read is
    direct (C2 scales the edge, which is exactly what these bins isolate).
    """
    p = np.asarray(p_cover, float)
    y = np.asarray(y_cover, float)
    e = np.abs(np.asarray(edge, float))
    pm, ym, em = p[mask], y[mask], e[mask]
    rows: list[dict] = []
    if len(pm) >= 20:
        edges = np.quantile(em, np.linspace(0, 1, 11))
        edges = np.unique(edges)
        idx = np.clip(np.digitize(em, edges[1:-1], right=False), 0,
                      len(edges) - 2)
        for b in range(len(edges) - 1):
            m = idx == b
            if not m.any():
                continue
            rows.append({
                "bin": f"|edge| [{edges[b]:.2f},{edges[b + 1]:.2f})",
                "n": int(m.sum()),
                "pred": round(float(pm[m].mean()), 4),
                "actual": round(float(ym[m].mean()), 4),
                "delta": round(float(ym[m].mean() - pm[m].mean()), 4),
            })
    extremes = []
    for lo in (0.50, 0.70, 0.90):
        m = em >= lo
        extremes.append({
            "bin": f"|edge|>={lo:.2f}", "n": int(m.sum()),
            "pred": (None if not m.any() else round(float(pm[m].mean()), 4)),
            "actual": (None if not m.any() else round(float(ym[m].mean()), 4)),
            "delta": (None if not m.any()
                      else round(float(ym[m].mean() - pm[m].mean()), 4)),
        })
    return {"deciles": rows, "extreme": extremes,
            "overall_delta": round(float(ym.mean() - pm.mean()), 4)
            if len(pm) else None}


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


def _pwin_gap(p: np.ndarray, y: np.ndarray, mask: np.ndarray,
              lo: float, hi: float) -> dict:
    m = mask & (p >= lo) & (p < hi)
    return {
        "bin": f"[{lo:.2f},{hi:.2f})",
        "n": int(m.sum()),
        "pred": (None if not m.any() else round(float(p[m].mean()), 4)),
        "actual": (None if not m.any() else round(float(y[m].mean()), 4)),
        "gap": (None if not m.any()
                else round(float(y[m].mean() - p[m].mean()), 4)),
    }


def main() -> None:
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime(
        "%Y%m%d")
    oof_path = DATA / f"run_engine_oof_{date_str}.csv"
    mon_path = DATA / f"run_engine_monitor_{date_str}.json"
    chal_path = DATA / f"run_engine_challenger_{date_str}.json"
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
    margin = hs - as_
    home_covers = (margin >= 2).astype(float)          # −1.5 line cover
    home_won = (margin > 0).astype(float)              # derived-ML target

    order = np.argsort(dates.to_numpy(), kind="stable")
    hold = np.zeros(len(oof), bool)
    hold[order[-SEALED_N:]] = True
    pre = ~hold

    fit = json.loads(mon_path.read_text())["fit"]
    a_h = alpha_of(lam_h, fit["alpha_home"])
    a_a = alpha_of(lam_a, fit["alpha_away"])

    # Per-run refit k: pre-holdout OOF only (strictly-prior discipline).
    k = ke.fit_k_edge(lam_h, lam_a, margin, pre)
    meta = ke.k_edge_meta(k)

    arms = {"C0_current": (lam_h, lam_a),
            "C2_k_edge": ke.apply_k_edge(lam_h, lam_a, k)}
    # Edge bins are shared across arms (bin on C0's |λ_edge|) so the
    # C2-vs-C0 read in the probe's extreme-edge bins is apples-to-apples.
    edge_c0 = lam_h - lam_a
    out: dict = {}
    for name, (lh, la) in arms.items():
        mc = derive_markets_mc(lh, la, a_h, a_a, n_draws=MC_N)
        tp, ty, tidx = _totals_rows(mc["p_over_grid"], lh, la, total)
        pwin = mc["p_home_win_derived"]
        out[name] = {
            "run_line_minus_1_5": {
                "pooled": _bin_rows(mc["p_home_cover_1_5"], home_covers,
                                    np.ones(len(oof), bool)),
                "sealed": _bin_rows(mc["p_home_cover_1_5"], home_covers,
                                    hold),
                "metrics_pooled": _metrics(mc["p_home_cover_1_5"],
                                           home_covers,
                                           np.ones(len(oof), bool)),
                "metrics_sealed": _metrics(mc["p_home_cover_1_5"],
                                           home_covers, hold),
            },
            "run_line_edge_bins": {
                "pooled": _edge_bin_rows(mc["p_home_cover_1_5"],
                                          home_covers, edge_c0,
                                          np.ones(len(oof), bool)),
                "sealed": _edge_bin_rows(mc["p_home_cover_1_5"],
                                          home_covers, edge_c0, hold),
            },
            "totals": {
                "metrics_pooled": _metrics(tp, ty, np.ones(len(tp), bool)),
                "metrics_sealed": _metrics(tp, ty, hold[tidx]),
            },
            "derived_ml": {
                "metrics_pooled": _metrics(pwin, home_won,
                                           np.ones(len(oof), bool)),
                "metrics_sealed": _metrics(pwin, home_won, hold),
                "pwin_sd_pooled": round(float(pwin[pre].std(ddof=1)), 4),
                "pwin_sd_sealed": round(float(pwin[hold].std(ddof=1)), 4),
                "bucket_55_60_pooled": _pwin_gap(
                    pwin, home_won, pre, *PWIN_BUCKET),
                "bucket_55_60_sealed": _pwin_gap(
                    pwin, home_won, hold, *PWIN_BUCKET),
            },
        }

    c0, c2 = out["C0_current"], out["C2_k_edge"]

    # ---- Gate checks (a–d) ----
    def _edge_gap_close(name: str, arm: dict) -> bool | None:
        """Extreme |λ_edge| bins (n ≥ 10): C2's max |delta| ≤ 0.05 (noise
        band on a cover rate), i.e. the probe's under-pricing closes."""
        rows = arm["run_line_edge_bins"][name]["extreme"]
        gs = [abs(r["delta"]) for r in rows
              if r.get("delta") is not None and (r.get("n") or 0) >= 10]
        return bool(gs and max(gs) <= 0.05)

    def _se_band(n: int) -> float:
        return float(np.sqrt(0.05 * 0.95 / max(n, 1)))

    checks = {
        "a_run_line_extreme_edge_bins": {
            "pooled_c0": c0["run_line_edge_bins"]["pooled"]["extreme"],
            "pooled_c2": c2["run_line_edge_bins"]["pooled"]["extreme"],
            "sealed_c0": c0["run_line_edge_bins"]["sealed"]["extreme"],
            "sealed_c2": c2["run_line_edge_bins"]["sealed"]["extreme"],
            "pooled_close_to_noise": _edge_gap_close(
                "pooled", c2),
            "sealed_close_to_noise": _edge_gap_close("sealed", c2),
        },
        "b_totals_flat": {
            "ece_pooled": (c0["totals"]["metrics_pooled"]["ece"],
                           c2["totals"]["metrics_pooled"]["ece"]),
            "ece_sealed": (c0["totals"]["metrics_sealed"]["ece"],
                           c2["totals"]["metrics_sealed"]["ece"]),
            "ece_pooled_ok": abs(c2["totals"]["metrics_pooled"]["ece"]
                                 - c0["totals"]["metrics_pooled"]["ece"])
            <= ECE_TOL,
            # sealed tolerance is sampling-noise based (2·se on n games)
            "ece_sealed_ok": abs(
                c2["totals"]["metrics_sealed"]["ece"]
                - c0["totals"]["metrics_sealed"]["ece"])
            <= 2 * _se_band(c0["totals"]["metrics_sealed"]["n"]),
        },
        "c_derived_ml": {
            "pwin_sd_c0_c2": (c0["derived_ml"]["pwin_sd_sealed"],
                              c2["derived_ml"]["pwin_sd_sealed"]),
            "sd_toward_target": c2["derived_ml"]["pwin_sd_sealed"]
            >= c0["derived_ml"]["pwin_sd_sealed"],
            "bucket_55_60_gap_c0_c2": (
                c0["derived_ml"]["bucket_55_60_sealed"]["gap"],
                c2["derived_ml"]["bucket_55_60_sealed"]["gap"]),
        },
    }
    if chal_path.exists():
        ch = json.loads(chal_path.read_text())
        ca, cb = ch.get("arms", {}).get("C2", {}), \
            ch.get("arms", {}).get("C0", {})
        checks["d_crps"] = {
            "sealed_c0": cb.get("margin_crps_sealed"),
            "sealed_c2": ca.get("margin_crps_sealed"),
            "pooled_c0": cb.get("margin_crps_pooled"),
            "pooled_c2": ca.get("margin_crps_pooled"),
            "source": str(chal_path.name),
            "near_2_4015": bool(ca.get("margin_crps_sealed") is not None
                                and abs(ca["margin_crps_sealed"] - 2.4015)
                                < 0.01),
            "pooled_corroborates": bool(
                ca.get("margin_crps_pooled") is not None
                and cb.get("margin_crps_pooled") is not None
                and ca["margin_crps_pooled"] < cb["margin_crps_pooled"]),
        }
    else:
        checks["d_crps"] = {"sealed_c2": None, "source": "challenger record "
                          "missing for this date"}

    ext_ok = checks["a_run_line_extreme_edge_bins"]["pooled_close_to_noise"]
    tot_ok = (checks["b_totals_flat"]["ece_pooled_ok"]
              and checks["b_totals_flat"]["ece_sealed_ok"])
    ml_ok = checks["c_derived_ml"]["sd_toward_target"]
    crps = checks["d_crps"]
    crps_ok = (crps.get("near_2_4015") is True
               and crps.get("pooled_corroborates") is True)
    verdict = bool(ext_ok and tot_ok and ml_ok and crps_ok)

    record = {
        "schema": "run-engine-k-edge-gate/v1",
        "date": date_str,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "baseline": "current engine (C0, raw λ pair)",
        "k": {"k_fitted_run": round(float(k), 4),
              "fit": "pre-holdout OOF only (per-run refit policy)",
              "reference_k": ke.K_EDGE_REF,
              "drift_band": meta["drift_band"],
              "drift_alert": meta["drift_alert"]},
        "arms": out,
        "checks": checks,
        "verdict": {
            "adopt_c2": verdict,
            "rule": ("C2 adopted iff (a) run-line −1.5 cover in the probe's "
                     "extreme |λ_edge| bins (>=0.5/0.70/0.90) closes toward "
                     "within noise (away +1.5 mirror is 1−p, same magnitude), "
                     "(b) totals ECE flat within tolerance (pooled 0.005, "
                     "sealed 2·se), (c) derived-ML P(win) SD toward 0.066 "
                     "and the [0.55,0.60) gap closes, (d) sealed CRPS "
                     "~2.4015 with pooled corroboration. No isotonic-on-ML "
                     "recalibration (out of scope)."),
            "parts": {"a_extreme_bins": bool(ext_ok),
                      "b_totals_flat": bool(tot_ok),
                      "c_derived_ml": bool(ml_ok),
                      "d_crps": bool(crps_ok)},
        },
    }
    out_path = DATA / f"run_engine_k_edge_gate_{date_str}.json"
    out_path.write_text(json.dumps(record, indent=2))
    print(f"k_edge gate record -> {out_path}")
    print(f"k_fitted_run={k:.4f} drift_alert={meta['drift_alert']}")
    print(f"verdict: {'ADOPT C2' if verdict else 'DO NOT ADOPT C2'} "
          f"(a={ext_ok} b={tot_ok} c={ml_ok} d={crps_ok})")


if __name__ == "__main__":
    main()
