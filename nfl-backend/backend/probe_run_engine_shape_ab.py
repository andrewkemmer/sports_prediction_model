"""Probe: run-engine under-recovery Variant A/B — response-shape fix.

READ-ONLY A/B against the pinned 1,376-row decided OOF store (the margin
audit c1a7c12 measured the run engine recovers only 59-67% of the true
quality-margin spread). No production/model change — this probe decides
whether ANY shape variant (sigma stretch / center-gap stretch / additive
quality-uncertainty variance) restores recovered spread to ~100% WITHOUT
degrading totals or derived-ML calibration, and routes a pre-registered
verdict. Produces one JSON on stdout.

Sources (all committed artifacts, no fitting of any production model):
  - nfl_run_engine_markets_20260904.csv  kind==oof rows: pred_home/
    pred_away (the unrounded per-side mu pair), p_home_win_derived,
    p_over_offered/p_cover_offered + y_* outcomes, spread_line/total_line,
    actuals, frame_view (pooled/sealed).
  - the canonical decided feature frame via load_features (the margin
    probe's read-only cache path) for the recovery features
    (elo_diff / ewm_net_pts_diff / win_pct_diff / ewm_ypp_diff).

Variant machinery = the SLATE EMITTER's own joint (nfl_joint_engine
build_joint_pmfs — exact 76x76 calibrated DN joints; no analytic
shortcut, so V0 through the same code path reproduces the artifact
bit-consistently). Variants only change the params (V1: per-side const
sigma x k — which scales BOTH margin and totals sigma, the honest DN
stretch) or the mu pair (V2: gap scale around the pair mean, which
preserves the total mean exactly) or both sigmas by an additive
quality-uncertainty term (V3: per-side sigma' = sqrt(sigma0^2 + c*mu_g^2),
the task's last-resort arm).

Gate (pre-registered, per spec):
  leg1  mean recovered-spread ratio over the 4 quality features in [90,110]
  leg2  derived-ML ll/ece within +/-0.002 of the pins BOTH views
        (pooled 0.6365/0.0435, sealed 0.6535/0.1009)
  leg3  totals ECE not worse than the measured V0 value (pooled + sealed)
  leg4  double-emit byte-identical (external double run; sha256 recorded)
GATE_PASS = a variant clears legs 1-3 (and determinism holds); the chosen
adoption candidate = passing variant with smallest |param - 1| (parsimony,
then V1 < V2 < V3 tie-break). GATE_FAIL = record the full variant table.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nfl_market_engine as M  # noqa: E402
import nfl_slate_engine as SE  # noqa: E402
from nfl_joint_engine import (build_joint_pmfs, cover_prob,  # noqa: E402
                              margin_pmf_from_joint, over_prob,
                              total_pmf_from_joint)
from nfl_moneyline import compute_metrics  # noqa: E402

DD = Path(__file__).resolve().parent.parent / "data_delivery"
DATE = "20260904"

# The 4 recovery features the margin audit's sextile table used.
RECOVERY_FEATURES = ["elo_diff", "ewm_net_pts_diff", "win_pct_diff",
                     "ewm_ypp_diff"]

# Record pins the probe must reproduce (run_nfl_markets_backfill
# RECORD_PINS — the task's R0 sanity gate).
PINS = {
    "pooled": {"totals_ece": 0.087, "covers_ece": 0.078,
               "derived_ml": {"logloss": 0.6365, "auc": 0.695,
                              "ece": 0.0435, "brier": 0.2221}},
    "sealed": {"totals_ece": 0.1547, "covers_ece": 0.1145,
               "derived_ml": {"logloss": 0.6535, "auc": 0.6782,
                              "ece": 0.1009, "brier": 0.2299}},
}
PIN_TOL = {"ece": 0.001, "ll": 0.0005, "auc": 0.0005, "brier": 0.0005}

# Variant grids (spec: V1 k in 1.00..1.50 step 0.05; V2 gap scale; V3 c
# only if V1/V2 cannot hit the band).
V1_KS = [round(1.00 + 0.05 * i, 2) for i in range(11)]
V2_KS = [round(1.00 + 0.05 * i, 2) for i in range(17)]   # 1.00 .. 1.80
V3_CS = [0.10, 0.50, 1.00]

# Gate bands (spec).
RECOVERY_BAND = (90.0, 110.0)
ML_TOL = 0.002


def _sextile_recovery(df: pd.DataFrame, feat: str,
                      pred_col: str) -> dict[str, Any] | None:
    """Audit methodology (MLB 39c865e mirror, probe_margin_disagreement):
    mean margin per sextile of the feature; recovery % = predicted
    sextile-5-minus-0 spread / actual spread."""
    if feat not in df.columns or df[feat].notna().sum() < 12:
        return None
    d = df[["margin", pred_col, feat]].dropna().copy()
    try:
        d["_q"] = pd.qcut(d[feat], 6, labels=False, duplicates="drop")
    except ValueError:
        return None
    rows = []
    for q in sorted(d["_q"].dropna().unique()):
        sub = d[d["_q"] == q]
        rows.append({"sextile": int(q), "n": int(len(sub)),
                     "actual_margin": round(float(sub["margin"].mean()), 3),
                     "pred_margin": round(float(sub[pred_col].mean()), 3)})
    if len(rows) < 2:
        return None
    act = rows[0]["actual_margin"] - rows[-1]["actual_margin"]
    rep = rows[0]["pred_margin"] - rows[-1]["pred_margin"]
    return {
        "actual_spread": round(act, 3),
        "pred_spread": round(rep, 3),
        "recovery_pct": round(100.0 * rep / act, 1) if act else None,
    }


def _variant_pass(rows: pd.DataFrame, params: dict[str, Any], p_tie: float,
                  name: str, k: float, ret_pwin: bool = False
                  ) -> tuple[dict[str, Any], np.ndarray | None]:
    """Rebuild exact 76x76 joints under the variant shape and score every
    metric per view (derived-ML, totals/cover ECE at the offered lines) +
    the sextile recovery on the variant pred margin.

    Returns (out, pwin_vector) — pwin_vector only when ``ret_pwin`` (V0
    fidelity check), else None."""
    pmfs, summ = build_joint_pmfs(
        rows[["game_id", "pred_home", "pred_away"]], params, p_tie)
    derived = summ["derived"]
    marg = [margin_pmf_from_joint(J) for J in pmfs]
    tot = [total_pmf_from_joint(J) for J in pmfs]
    v = rows.copy()
    pwin = derived["derived_ml"].to_numpy(float)
    v["p_win_v"] = pwin
    L0 = rows["spread_line"].to_numpy(float)
    U0 = rows["total_line"].to_numpy(float)
    v["p_over_v"] = np.array(
        [over_prob(t, float(U)) if np.isfinite(U) else np.nan
         for t, U in zip(tot, U0)])
    v["p_cover_v"] = np.array(
        [cover_prob(m, float(L)) if np.isfinite(L) else np.nan
         for m, L in zip(marg, L0)])
    v["pred_margin_v"] = v["pred_home"] - v["pred_away"]

    out: dict[str, Any] = {"name": name, "param": k}
    # NOTE: no elapsed/timing field in the output — determinism requires
    # byte-identical double emits; timing goes to stderr only.
    for view in ("pooled", "sealed"):
        sub = v[v["frame_view"] == view].copy()
        ml = compute_metrics(sub["y_home_win"].to_numpy(float),
                             sub["p_win_v"].to_numpy(float))
        t_ece = M.totals_calibration(sub, p_col="p_over_v",
                                     y_col="y_over_offered")["ece"]
        c_ece = M.covers_calibration(sub, p_col="p_cover_v",
                                     y_col="y_cover_offered")["ece"]
        out[view] = {
            "n": int(len(sub)),
            "derived_ml": ml,
            "totals_ece": (round(float(t_ece), 4)
                           if t_ece is not None else None),
            "covers_ece": (round(float(c_ece), 4)
                           if c_ece is not None else None),
        }
    rec = {}
    for feat in RECOVERY_FEATURES:
        r = _sextile_recovery(v, feat, "pred_margin_v")
        if r is not None:
            rec[feat] = r
    ratios = [r["recovery_pct"] for r in rec.values()
              if r.get("recovery_pct") is not None]
    out["recovery"] = rec
    out["recovery_mean_pct"] = (
        round(float(np.mean(ratios)), 1) if ratios else None)
    return out, (pwin if ret_pwin else None)


def _gate_legs(pass_row: dict[str, Any], v0_totals: dict[str, float]
               ) -> dict[str, Any]:
    """Pre-registered legs. pass_row = a variant pass output."""
    rm = pass_row.get("recovery_mean_pct")
    leg1 = rm is not None and RECOVERY_BAND[0] <= rm <= RECOVERY_BAND[1]
    ml_deltas = {}
    for view in ("pooled", "sealed"):
        for k in ("logloss", "ece"):
            ml_deltas[f"{view}_{k}"] = round(
                float(pass_row[view]["derived_ml"][k]) - PINS[view]["derived_ml"][k], 4)
    leg2 = all(abs(d) <= ML_TOL for d in ml_deltas.values())
    t_deltas = {view: round(float(pass_row[view]["totals_ece"])
                            - v0_totals[view], 4)
                for view in ("pooled", "sealed")
                if pass_row[view]["totals_ece"] is not None}
    leg3 = all(d <= 0.0 for d in t_deltas.values())
    return {
        "leg1_recovery_90_110": bool(leg1),
        "recovery_mean_pct": rm,
        "leg2_derived_ml_plusminus_0_002": bool(leg2),
        "ml_deltas_vs_pins": ml_deltas,
        "leg3_totals_ece_not_worse": bool(leg3),
        "totals_ece_delta_vs_v0": t_deltas,
        "all_legs": bool(leg1 and leg2 and leg3),
    }


def main() -> int:
    mk = pd.read_csv(DD / f"nfl_run_engine_markets_{DATE}.csv")
    oof = mk[mk["kind"] == "oof"].copy()
    if len(oof) != 1376:
        raise RuntimeError(f"oof rows {len(oof)} != 1,376 — STOP")

    # Recovery features from the canonical decided frame (margin probe path).
    # load_features prints a build banner to stdout when it builds from
    # nflreadpy — redirect it so the probe's stdout stays pure JSON.
    try:
        import contextlib
        import io
        from run_nfl_margin_ablation import load_features
        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf):
            feats = load_features(None)[["game_id"] + RECOVERY_FEATURES]
        if _buf.getvalue().strip():
            print(_buf.getvalue().strip(), file=sys.stderr)
        oof = oof.merge(feats, on="game_id", how="left")
    except Exception as exc:  # noqa: BLE001 — honest degradation, never silent
        print(f"WARN: recovery features unavailable ({exc}); recovery "
              "legs will be None", file=sys.stderr)
        for f in RECOVERY_FEATURES:
            oof[f] = np.nan

    y = (oof["home_score"] > oof["away_score"]).astype(float).to_numpy()
    oof["y_home_win"] = y
    oof["margin"] = (oof["home_score"] - oof["away_score"]).to_numpy()
    oof = oof.reset_index(drop=True)

    out: dict[str, Any] = {
        "record": "nfl_run_engine_diagnostics_shape_ab",
        "frame_sha256": "3e8c8a510f04",
        "date": DATE,
        "universe": {"oof_rows": int(len(oof)),
                     "pooled": int((oof["frame_view"] == "pooled").sum()),
                     "sealed": int((oof["frame_view"] == "sealed").sum()),
                     "recovery_feature_coverage": {
                         f: int(oof[f].notna().sum()) for f in RECOVERY_FEATURES},
                     "sigma_margin_pinned": round(float(np.sqrt(
                         SE.PINNED_SIGMA_HOME ** 2 + SE.PINNED_SIGMA_AWAY ** 2
                         - 2 * SE.PINNED_RHO * SE.PINNED_SIGMA_HOME
                         * SE.PINNED_SIGMA_AWAY)), 4),
                     "sigma_total_pinned": round(float(np.sqrt(
                         SE.PINNED_SIGMA_HOME ** 2 + SE.PINNED_SIGMA_AWAY ** 2
                         + 2 * SE.PINNED_RHO * SE.PINNED_SIGMA_HOME
                         * SE.PINNED_SIGMA_AWAY)), 4)},
    }

    # ── R0 sanity gate (V0 from the committed artifact columns) ────────────
    r0: dict[str, Any] = {}
    v0_totals: dict[str, float] = {}
    for view in ("pooled", "sealed"):
        m = oof["frame_view"] == view
        ml = compute_metrics(y[m], oof["p_home_win_derived"].to_numpy()[m])
        t_ece = M.totals_calibration(oof[m], p_col="p_over_offered",
                                     y_col="y_over_offered")["ece"]
        c_ece = M.covers_calibration(oof[m], p_col="p_cover_offered",
                                     y_col="y_cover_offered")["ece"]
        r0[view] = {"derived_ml_artifact": ml,
                    "totals_ece_artifact": (round(float(t_ece), 4)
                                            if t_ece is not None else None),
                    "covers_ece_artifact": (round(float(c_ece), 4)
                                            if c_ece is not None else None)}
        v0_totals[view] = float(t_ece)
        pin = PINS[view]
        for k, tol in (("logloss", PIN_TOL["ll"]), ("auc", PIN_TOL["auc"]),
                       ("ece", PIN_TOL["ece"]), ("brier", PIN_TOL["brier"])):
            if abs(ml[k] - pin["derived_ml"][k]) > tol:
                r0["fail"] = f"{view} derived-ML {k} {ml[k]} != pin {pin['derived_ml'][k]}"
        if abs(float(t_ece) - pin["totals_ece"]) > PIN_TOL["ece"]:
            r0["fail"] = f"{view} totals {t_ece} != pin {pin['totals_ece']}"
        if abs(float(c_ece) - pin["covers_ece"]) > PIN_TOL["ece"]:
            r0["fail"] = f"{view} covers {c_ece} != pin {pin['covers_ece']}"
    r0["pass"] = "fail" not in r0
    out["r0_gate"] = r0
    if not r0["pass"]:
        out["verdict"] = "R0_FAIL — artifact pins not reproduced; STOP"
        print(json.dumps(out, indent=2, default=str))
        return 2

    # ── V0 machinery fidelity: rebuild joints at pinned params ─────────────
    params0 = SE.pinned_joint_params()
    v0, pwin0 = _variant_pass(oof, params0, SE.PINNED_P_TIE,
                              "V0_identity", 1.0, ret_pwin=True)
    max_diff = float(np.abs(pwin0 - oof["p_home_win_derived"].to_numpy()).max())
    out["r0_gate"]["joint_rebuild_fidelity"] = {
        "max_abs_diff_derived_ml_vs_artifact": round(max_diff, 8),
        "metric_ll_pooled_rebuilt": v0["pooled"]["derived_ml"]["logloss"],
        "metric_ece_pooled_rebuilt": v0["pooled"]["derived_ml"]["ece"],
    }

    # ── Variants ────────────────────────────────────────────────────────────
    variants: dict[str, Any] = {}
    variants["V0_identity"] = {
        "grid": [1.0], "rows": [v0],
        "mechanism": "pinned params, identity mu — must reproduce the pins"}
    t0 = time.time()

    v1_rows = []
    for k in V1_KS:
        p = SE.pinned_joint_params()
        p["sigma_h"] = {"spec": "const", "sigma0": SE.PINNED_SIGMA_HOME * k,
                        "q": 0.0}
        p["sigma_a"] = {"spec": "const", "sigma0": SE.PINNED_SIGMA_AWAY * k,
                        "q": 0.0}
        v1_rows.append(_variant_pass(oof, p, SE.PINNED_P_TIE,
                                     "V1_sigma_stretch", k)[0])
        print(f"  V1 k={k} done ({time.time()-t0:.0f}s)", file=sys.stderr)
    variants["V1_sigma_stretch"] = {"grid": V1_KS, "rows": v1_rows,
                                    "mechanism": ("per-side const sigma x k "
                                                  "(scales BOTH margin and "
                                                  "totals sigma — honest DN "
                                                  "stretch; mu pair untouched)")}

    v2_rows = []
    for k in V2_KS:
        g = oof.copy()
        m = (g["pred_home"] + g["pred_away"]) / 2.0
        g["pred_home"] = m + k * (g["pred_home"] - m)
        g["pred_away"] = m + k * (g["pred_away"] - m)
        v2_rows.append(_variant_pass(g, SE.pinned_joint_params(),
                                     SE.PINNED_P_TIE, "V2_gap_stretch", k)[0])
        print(f"  V2 k={k} done ({time.time()-t0:.0f}s)", file=sys.stderr)
    variants["V2_gap_stretch"] = {
        "grid": V2_KS, "rows": v2_rows,
        "mechanism": ("gap scale k around the pair mean: mu'_h = m + k(mu_h - m), "
                      "mu'_a = m + k(mu_a - m) — total mean preserved exactly, "
                      "joint params pinned")}

    v3_rows = []
    for c in V3_CS:
        p = SE.pinned_joint_params()
        gap = (oof["pred_home"] - oof["pred_away"]).to_numpy(float)
        add_h = (c * gap ** 2).astype(float)
        add_a = add_h.copy()
        p["sigma_h"] = {"spec": "const",
                        "sigma0": float(np.sqrt(SE.PINNED_SIGMA_HOME ** 2
                                                + add_h.mean())), "q": 0.0}
        p["sigma_a"] = {"spec": "const",
                        "sigma0": float(np.sqrt(SE.PINNED_SIGMA_AWAY ** 2
                                                + add_a.mean())), "q": 0.0}
        # NOTE: per-game variance needs per-game sigma; the const-const joint
        # cannot carry it — this arm is defined with a GLOBAL additive
        # variance term (documented in the record) and reported for
        # completeness; the per-game form is out of the pinned-joint schema.
        v3_rows.append(_variant_pass(oof, p, SE.PINNED_P_TIE,
                                     "V3_global_quality_variance", c)[0])
        print(f"  V3 c={c} done ({time.time()-t0:.0f}s)", file=sys.stderr)
    variants["V3_quality_variance"] = {
        "grid": V3_CS, "rows": v3_rows,
        "mechanism": ("global additive variance: sigma0' = sqrt(sigma0^2 + "
                      "c*mean(mu_gap^2)) — the pinned const-const joint "
                      "cannot carry per-game sigma; last-resort arm")}
    out["variants"] = variants

    # ── Gate routing ────────────────────────────────────────────────────────
    gate_rows = []
    for fam in ("V1_sigma_stretch", "V2_gap_stretch", "V3_quality_variance"):
        for r in variants[fam]["rows"]:
            legs = _gate_legs(r, v0_totals)
            gate_rows.append({"family": fam, "param": r["param"],
                              **legs,
                              "recovery_mean_pct": r.get("recovery_mean_pct"),
                              "pooled_ll": r["pooled"]["derived_ml"]["logloss"],
                              "pooled_ece": r["pooled"]["derived_ml"]["ece"],
                              "sealed_ll": r["sealed"]["derived_ml"]["logloss"],
                              "sealed_ece": r["sealed"]["derived_ml"]["ece"],
                              "totals_ece_pooled": r["pooled"]["totals_ece"],
                              "totals_ece_sealed": r["sealed"]["totals_ece"]})
    passing = [g for g in gate_rows if g["all_legs"]]
    out["gate"] = {
        "rows": gate_rows,
        "passing_variants": [{k: g[k] for k in ("family", "param")}
                             for g in passing],
        "verdict": "GATE_PASS" if passing else "GATE_FAIL",
    }
    if passing:
        # Parsimony: smallest |param - 1|, then V1 < V2 < V3 tie-break.
        order = {"V1_sigma_stretch": 0, "V2_gap_stretch": 1,
                 "V3_quality_variance": 2}
        best = min(passing, key=lambda g: (abs(g["param"] - 1.0),
                                           order[g["family"]]))
        out["gate"]["adoption_candidate"] = best
        out["gate"]["adoption_note"] = ("parameter from the recorded fit, "
                                        "never hardcoded — wire the variant "
                                        "as the production shape")
    else:
        out["gate"]["follow_ons"] = [
            "no shape variant clears the gate: recovery cannot be restored "
            "to 90-110% within the +/-0.002 derived-ML band / totals "
            "constraint — the sigma-compression is not a shape parameter, "
            "the mu response is (V2 moves it but pays calibration)",
            "the remaining lever is the NFL-P1 opposing-quality LEVEL input "
            "arm (audit recommendation) — a feature-level response fix, not "
            "a post-hoc shape transform",
            "revisit with 2026 decided rows (~272/yr) — the diagnostic gets "
            "stronger and the sealed band legs are the least certain",
        ]
    out["determinism"] = {"sha256": None, "note": ("external double run; "
                                                   "sha256 filled by the "
                                                   "runner")}

    raw = json.dumps(out, indent=2, default=str)
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    out["determinism"]["sha256"] = h
    raw = json.dumps(out, indent=2, default=str)
    print(raw)
    print(f"SHA256: {h}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())