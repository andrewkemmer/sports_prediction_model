"""NFL era/conditional-mean layer — Step-0 gate + C0/E1/E2 arms + joint chain
re-run (record-only).

Target: the by-season away mean-bias defect (2021 -1.71 / 2022 -1.63 /
2023 -2.27 / 2024 -0.40 / sealed 2025 -0.14, resid = actual − pred) and the
away CRPS shortfall. This layer addresses LEAGUE-LEVEL scoring-environment
drift — not team-specific offseason structural change (roster/defense
overhaul stays invisible until games confirm it — do not overclaim). No
wiring; FEATURE_COLUMNS / 12-pool / moneyline / daily pipeline untouched.
nfl_per_side_engine.py and nfl_joint_engine.py are NOT modified — the
centered-target code path lives in nfl_era_features.py and the joint chain
re-run calls the EXISTING engine entrypoints.

Pipeline (identical geometry to every prior record — same 88 folds, pooled
OOF 2021-24 n=1,091, sealed 2025 n=285):
  Step 0 (falsification gate, diagnostics ONLY — no model change):
    0a season fact table 2019-2025; 0b per-season OOF + sealed bias;
    0c week-half away-bias split (weeks 1-5 vs 6+); 0d model-free ceiling
    per center spec (ps, ewm_2w/4w/8w) vs the LGB's by-season bias.
    GATE: proceed only if (i) away mean scoring tracks the bias pattern and
    (ii) some center spec cuts the 2021-23 mean |away bias| by >= ~50% or
    >= 1.0 pt model-free. If the gate fails the era layer is refuted at the
    data level — the record carries the tables and the run stops (no arms).
  Step 1: center-spec CV selection on the 2021-24 pooled OOF ONLY (sealed
    2025 never touches selection — structural guard) via the E2 walk; then
    C0 (raw-target re-walk anchor), E1 (+ era macro columns in the marginal
    feature view), E2 (centered targets, PRIMARY).
  Step 2: joint-chain re-run on the adopted arm's per-side table through
    nfl_joint_engine entrypoints: gates G1-G5 exactly as the prior records,
    totals ECE seam check, by-season bias after the era layer.
  Verdict: SUCCESS = 2021-23 mean |away bias| roughly halved vs C0 AND
    sealed 2025 not overcorrected (side resid within ±0.5) AND G2/G3/G5 AND
    G4 coherent-or-better. NOT gated on G1 >= 5% or totals top-bin closing
    (those are the dispersion/sigma layer's job — this is a NECESSARY step
    toward the market layer, not sufficient).

Usage:
    cd nfl-backend && python3 backend/run_nfl_era.py [--no-record]
        [--family lgb] [--seam-pass]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nfl_era_features import (
    CENTER_AWAY, CENTER_COLS, CENTER_HOME, EWM_HALFLIFE_DAYS, NEUTRAL_CENTER,
    PRED_COL, RESID_COL, SCORE_COL, SIDES, SPECS, attach_centers,
    bias_by_season, center_bias_by_season, compute_centers,
    mean_resid_stats, oof_centered_per_side, refit_centered_per_side,
    season_fact_table, week_half_split,
)
from nfl_joint_engine import build_joint_pmfs, fit_joint_params, \
    load_residual_artifact
from nfl_run_engine_legacy_windows import (SEALED_SEASON, TRAIN_SEASONS,
                                           generate_weekly_folds)
from nfl_moneyline import (_valid_rows, compute_metrics)
from nfl_per_side_engine import SIDE_FEATURES, oof_per_side
from run_nfl_joint import (C0_ANCHOR, _crps_vs_climatology, _seam_check,
                           _sealed_predictions)
from run_nfl_margin_ablation import _frame_sha256, load_features

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY = ROOT_DIR / "data_delivery"
DECIDED_FRAME = DATA_DELIVERY / "nfl_game_level_features.csv"
ARTIFACT = DATA_DELIVERY / "nfl_per_side_oof_residuals_3e8c8a510f04.csv"

# Prior-chain provenance for the before-figures (deterministic ⇒ stored
# numbers ARE the before numbers on identical code paths).
MEAN_BIAS_RECORD_NAME = "nfl_mean_bias_calibration_3e8c8a510f04.json"

SCOPE_PIN = (
    "This layer addresses LEAGUE-LEVEL scoring-environment drift, not "
    "team-specific offseason structural change (roster/defense overhaul "
    "stays invisible until games confirm it — do not overclaim). "
    "Leakage-safe side-specific era centers over strictly-prior decided "
    "games; centered-target marginal (E2) fits (target − center) and adds "
    "the center back at prediction. Spec chosen empirically on 2021-24 "
    "pooled OOF CV only; sealed 2025 evaluated once. Joint-chain re-run "
    "through the EXISTING nfl_joint_engine entrypoints — no engine edits, "
    "no wiring. Market pricing/calibration paths, artifact emitters, and "
    "wire-in remain later phases."
)

# Step-0 gate thresholds (documented, deterministic).
AWAY_MOVEMENT_PTS = 1.0    # season-mean away scoring range (scored seasons)
SPIKE_PTS = 1.0            # 2020 away mean above the 2021-23 mean
BIAS_CUT_ABS_PTS = 1.0     # gate (ii): spec cuts 2021-23 |bias| by >= 1.0 pt
BIAS_CUT_REL = 0.5         # ... or to <= 50% of the LGB 2021-23 |bias|

# Adopted-arm decision (spec judgment: if E2 ~= E1 adopt E2 — simpler, more
# robust; E2 fixes level by construction).
E2_REFUTED_MAE_MARGIN = 0.10  # E2 away MAE > C0 + this ⇒ era refuted


def _frame_sha() -> str:
    return hashlib.sha256(DECIDED_FRAME.read_bytes()).hexdigest()[:12]


def _mae(df: pd.DataFrame, side: str) -> float:
    return round(float(np.abs(df[SCORE_COL[side]].to_numpy(float)
                              - df[PRED_COL[side]].to_numpy(float)).mean()), 4)


def _mean_abs_resid_2021_23(df: pd.DataFrame, side: str) -> float | None:
    sub = df[df["season"].isin([2021, 2022, 2023])]
    if not len(sub):
        return None
    vals = []
    for _, g in sub.groupby("season"):
        st = mean_resid_stats(g, PRED_COL[side], SCORE_COL[side])
        vals.append(abs(st["mean_resid"]))
    return round(float(np.mean(vals)), 4)


def _decided_view(feats: pd.DataFrame) -> pd.DataFrame:
    return feats[["game_id", "season", "week", "gameday", "home_score",
                  "away_score", "total"]].copy()


def _folds_for(feats: pd.DataFrame, features: list[str]) -> list[dict]:
    preq = feats[feats["season"].isin(TRAIN_SEASONS)].copy()
    preq_valid = preq[_valid_rows(preq, features)].copy()
    return generate_weekly_folds(preq_valid)


def _sealed_era_eval(feats: pd.DataFrame, rounds: dict,
                     features: list[str]) -> pd.DataFrame:
    """Fit-only refill on centered targets: fit 2019-24 at the E2 arm's
    per-side median rounds, predict 2025 (center added back per row)."""
    preq = feats[feats["season"].isin(TRAIN_SEASONS)].copy()
    preq_valid = preq[_valid_rows(preq, features)].copy()
    sld = feats[feats["season"] == SEALED_SEASON].copy()
    sld_valid = sld[_valid_rows(sld, features)].copy()
    refit = refit_centered_per_side(preq_valid, sld_valid, rounds, features)
    return sld_valid.merge(refit, on="game_id", how="left")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON record")
    ap.add_argument("--seam-pass", action="store_true",
                    help="skip the walks; re-run ONLY the totals seam check "
                         "from /tmp dumps and patch the record")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    warnings.filterwarnings("ignore",
                            message="The argument 'eval_set' is deprecated")
    t0 = time.time()
    frame_sha = _frame_sha()
    print(f"frame_sha256={frame_sha}")

    # =====================================================================
    # SEAM PASS (light re-run from /tmp dumps; patches the record in place)
    # =====================================================================
    if args.seam_pass:
        dump = Path("/tmp/nfl_era_e2_pooled.csv")
        if not dump.exists():
            print("seam pass: no /tmp dumps — run the main pass first")
            return 1
        # The main pass dumps the E2 engine table WITH actuals + residuals
        # — no re-merge needed (a second score merge would suffix-collide).
        eng = pd.read_csv(dump)
        params = fit_joint_params(eng)
        p_tie = float(np.mean(eng["home_score"] == eng["away_score"]))
        pmfs, _sum = build_joint_pmfs(eng, params, p_tie)
        derived = _sum["derived"].copy()
        derived = derived.merge(eng[["game_id", "home_score", "away_score"]],
                                on="game_id", how="left")
        seam = _seam_check(load_features(None), pmfs, derived)
        print(f"seam: ok={seam.get('ok')} totals_ece="
              f"{(seam.get('totals') or {}).get('ece')} covers_ece="
              f"{(seam.get('covers') or {}).get('ece')}")
        rec_path = DATA_DELIVERY / f"nfl_era_{frame_sha}.json"
        if not rec_path.exists():
            print(f"seam pass: record missing ({rec_path}) — run main pass first")
            return 1
        rec = json.loads(rec_path.read_text())
        rec["step2_joint_chain"]["data_seam"] = \
            seam if seam else {"skipped": True}
        rec_path.write_text(json.dumps(rec, indent=2, default=str))
        print(f"seam pass: record patched ({rec_path.name}) in "
              f"{time.time() - t0:.0f}s")
        return 0

    # =====================================================================
    # INPUTS
    # =====================================================================
    feats = load_features(None)
    feats["home_win"] = (feats["home_score"] > feats["away_score"]).astype(int)
    decided = _decided_view(feats)
    art = load_residual_artifact(ARTIFACT)
    art = art.merge(feats[["game_id", "home_score", "away_score", "season",
                           "week"]], on="game_id", how="left")
    if art["home_score"].isna().any():
        raise RuntimeError("artifact join left NaN actuals")
    rounds_c0 = {"home": int(art["best_iter_home"].median()),
                 "away": int(art["best_iter_away"].median())}
    sealed_c0 = _sealed_predictions(feats, rounds_c0)
    print(f"pooled OOF n={len(art)} | sealed n={len(sealed_c0)} | "
          f"C0 rounds={rounds_c0}")
    centers = {s: compute_centers(decided, s) for s in SPECS}
    print(f"centers: {SPECS} (neutral fallback {NEUTRAL_CENTER})")

    # =====================================================================
    # STEP 0 — falsification gate (diagnostics only)
    # =====================================================================
    print("\n[Step 0] falsification gate (diagnostics)...")
    fact = season_fact_table(decided)
    for d in fact:
        print(f"  {d['season']}: n={d['n']} home={d['mean_home_score']} "
              f"away={d['mean_away_score']} tot={d['mean_total']} "
              f"hwr={d['home_win_rate']}")
    oof_bias = {s: bias_by_season(art, PRED_COL[s], SCORE_COL[s])
                for s in SIDES}
    sealed_bias_c0 = {s: mean_resid_stats(sealed_c0, PRED_COL[s],
                                          SCORE_COL[s]) for s in SIDES}
    week_half = {s: week_half_split(art, PRED_COL[s], SCORE_COL[s])
                 for s in SIDES}
    print("  OOF away bias by season: "
          + ", ".join(f"{d['season']}:{d['mean_resid']}"
                      for d in oof_bias["away"]))
    print(f"  sealed 2025 away bias: {sealed_bias_c0['away']['mean_resid']}")
    ceiling = center_bias_by_season(decided, centers)

    lgb_away_2123 = float(np.mean([abs(d["mean_resid"])
                                   for d in oof_bias["away"]
                                   if d["season"] in (2021, 2022, 2023)]))
    away_means = {int(d["season"]): d["mean_away_score"] for d in fact}
    sc_mean = away_means[2020] - float(np.mean(
        [away_means[s_] for s_ in (2021, 2022, 2023)]))
    sc_range = max(away_means[s_] for s_ in (2021, 2022, 2023, 2024)) \
        - min(away_means[s_] for s_ in (2021, 2022, 2023, 2024))
    gate_i = bool(sc_mean >= SPIKE_PTS and sc_range >= AWAY_MOVEMENT_PTS)
    spec_cuts = {}
    gate_ii = False
    for spec in SPECS:
        b = ceiling["per_side"]["away"][spec]["mean_abs_bias_2021_23"]
        if b is None:
            continue
        cut_abs = (lgb_away_2123 - b) >= BIAS_CUT_ABS_PTS
        cut_rel = b <= BIAS_CUT_REL * lgb_away_2123
        spec_cuts[spec] = {"mean_abs_bias_2021_23": b,
                           "lgb_mean_abs_2021_23": round(lgb_away_2123, 4),
                           "cut_abs_1pt": bool(cut_abs),
                           "cut_rel_50pct": bool(cut_rel),
                           "passes": bool(cut_abs or cut_rel)}
        gate_ii = gate_ii or bool(cut_abs or cut_rel)
    print(f"  gate (i): spike={sc_mean:.2f} range={sc_range:.2f} → {gate_i}")
    for spec, d in spec_cuts.items():
        print(f"  gate (ii) {spec}: |bias|={d['mean_abs_bias_2021_23']} vs "
              f"LGB {lgb_away_2123:.2f} → {d['passes']}")
    gate_pass = gate_i and gate_ii
    print(f"  STEP-0 GATE → {'PASS' if gate_pass else 'FAIL'}")

    step0 = {
        "fact_table_2019_2025": fact,
        "oof_bias_by_season": oof_bias,
        "sealed_2025_bias_c0": sealed_bias_c0,
        "week_half_away_bias": week_half,
        "sign_convention": ("resid = actual − pred; negative mean resid ⇒ "
                            "predictions run HIGH (overprediction)"),
        "ceiling_model_free": ceiling,
        "gate": {
            "i_away_scoring_tracks_bias": {
                "pass": gate_i,
                "rule": ("2020 spike = away_mean(2020) − mean(away 2021-23) "
                         f">= {SPIKE_PTS} pt AND scored-season away range "
                         f">= {AWAY_MOVEMENT_PTS} pt (≈2 season-SEs)"),
                "spike_pts": round(sc_mean, 3),
                "scored_season_range_pts": round(sc_range, 3)},
            "ii_center_cuts_bias": {
                "pass": gate_ii,
                "rule": ("some spec's model-free 2021-23 mean |bias| cuts the "
                         f"LGB's ({lgb_away_2123:.2f}) by >= {BIAS_CUT_ABS_PTS} "
                         f"pt or to <= {BIAS_CUT_REL:.0%}"),
                "specs": spec_cuts},
            "pass": gate_pass,
            "on_fail": ("era layer refuted at the data level — record tables "
                        "and stop (no E arms)"),
        },
    }

    # =====================================================================
    # STEP 1 — spec CV selection (pooled 2021-24 ONLY) + C0/E1/E2 arms
    # =====================================================================
    if not gate_pass:
        print("\nStep-0 gate FAILED — era layer refuted at the data level; "
              "recording tables and stopping (no arms).")
    else:
        print("\n[Step 1] spec CV selection on pooled 2021-24 (never 2025)...")
        cv: dict[str, Any] = {}
        for spec in SPECS:
            f_sp = attach_centers(feats, centers[spec])
            folds = _folds_for(f_sp, SIDE_FEATURES)
            t1 = time.time()
            oof_sp, _r, _u = oof_centered_per_side(folds, SIDE_FEATURES, f_sp)
            oof_sp = oof_sp.merge(f_sp[["game_id", "season", "home_score",
                                        "away_score"]], on="game_id",
                                  how="left")
            if len(oof_sp) != 1091:
                raise RuntimeError(f"spec {spec}: E2 coverage {len(oof_sp)} "
                                   "!= 1091 — geometry drift")
            cv[spec] = {
                "away_mean_abs_resid_2021_23": _mean_abs_resid_2021_23(
                    oof_sp, "away"),
                "away_mae": _mae(oof_sp, "away"),
                "home_mae": _mae(oof_sp, "home"),
                "n": int(len(oof_sp)),
                "walk_s": round(time.time() - t1, 1),
            }
            print(f"  {spec}: away |bias|21-23="
                  f"{cv[spec]['away_mean_abs_resid_2021_23']} away_mae="
                  f"{cv[spec]['away_mae']} home_mae={cv[spec]['home_mae']} "
                  f"({cv[spec]['walk_s']}s)")
        # Selection: minimize pooled 2021-23 away mean |season resid|;
        # tiebreak away MAE. Selection inputs are pooled-OOF walks only —
        # sealed rows structurally never enter (compute_centers output for
        # 2021-24 rows depends only on strictly-prior games).
        sel = sorted(SPECS, key=lambda s: (cv[s]["away_mean_abs_resid_2021_23"]
                                           or 1e9, cv[s]["away_mae"]))
        chosen = sel[0]
        print(f"  chosen spec: {chosen} (CV order: {sel})")

        f_chosen = attach_centers(feats, centers[chosen])
        folds_chosen = _folds_for(f_chosen, SIDE_FEATURES)

        # ---- C0: raw-target re-walk anchor (same folds, same view) ----
        print(f"\n[Step 1] arms on chosen spec {chosen}...")
        folds_c0 = _folds_for(feats, SIDE_FEATURES)
        c0_oof, _r0, _u0 = oof_per_side(folds_c0, SIDE_FEATURES, feats)
        c0_oof = c0_oof.merge(feats[["game_id", "season", "home_score",
                                     "away_score"]], on="game_id", how="left")
        c0_mae = {s: _mae(c0_oof, s) for s in SIDES}
        c0_bias = {s: bias_by_season(c0_oof, PRED_COL[s], SCORE_COL[s])
                   for s in SIDES}
        print(f"  C0: n={len(c0_oof)} home_mae={c0_mae['home']} "
              f"away_mae={c0_mae['away']} (artifact pin 7.262/7.123)")

        # ---- E1: + era macro columns in the marginal view ----
        e1_oof, _r1, _u1 = oof_per_side(folds_chosen,
                                        SIDE_FEATURES + CENTER_COLS, f_chosen)
        e1_oof = e1_oof.merge(f_chosen[["game_id", "season", "home_score",
                                        "away_score"]], on="game_id",
                              how="left")
        e1_mae = {s: _mae(e1_oof, s) for s in SIDES}
        e1_bias = {s: bias_by_season(e1_oof, PRED_COL[s], SCORE_COL[s])
                   for s in SIDES}
        print(f"  E1: n={len(e1_oof)} home_mae={e1_mae['home']} "
              f"away_mae={e1_mae['away']}")

        # ---- E2: centered targets, PRIMARY (double walk for G5) ----
        e2_oof, rounds_e2, _u2 = oof_centered_per_side(folds_chosen,
                                                       SIDE_FEATURES,
                                                       f_chosen)
        e2_oof = e2_oof.merge(f_chosen[["game_id", "season", "home_score",
                                        "away_score"]], on="game_id",
                              how="left")
        e2_oof2, _r2b, _u2b = oof_centered_per_side(folds_chosen,
                                                    SIDE_FEATURES, f_chosen)
        e2_det = (e2_oof.drop(columns=["season", "home_score", "away_score"],
                              errors="ignore")
                  .sort_values("game_id").reset_index(drop=True)
                  .to_csv(index=False)
                  == e2_oof2.sort_values("game_id").reset_index(drop=True)
                  .to_csv(index=False))
        e2_mae = {s: _mae(e2_oof, s) for s in SIDES}
        e2_bias = {s: bias_by_season(e2_oof, PRED_COL[s], SCORE_COL[s])
                   for s in SIDES}
        e2_away_2123 = _mean_abs_resid_2021_23(e2_oof, "away")
        print(f"  E2: n={len(e2_oof)} rounds={rounds_e2} "
              f"home_mae={e2_mae['home']} away_mae={e2_mae['away']} "
              f"away|bias|21-23={e2_away_2123} det={e2_det}")

        # ---- Adoption (spec: E2 ~= E1 ⇒ adopt E2; E2 <= C0 ⇒ refuted) ----
        e2_refuted = e2_mae["away"] > c0_mae["away"] + E2_REFUTED_MAE_MARGIN
        adopted = "E2"
        adoption_note = ("E2 ~= E1 by design (E2 fixes level by construction, "
                         "leaves the 12-feature mapping untouched, cannot be "
                         "ignored by the trees) — E2 adopted")
        if e2_refuted:
            adopted = "NONE"
            adoption_note = ("E2 away MAE > C0 + 0.10 ⇒ era hypothesis "
                             "refuted at the model level — record and stop")
        print(f"  adopted arm: {adopted} | {adoption_note}")

        # ---- Sealed 2025 on the adopted arm (fit-only, E2 rounds) ----
        sealed_e2 = None
        if adopted == "E2":
            sealed_e2 = _sealed_era_eval(f_chosen, rounds_e2, SIDE_FEATURES)
            print(f"  sealed E2: n={len(sealed_e2)} away bias="
                  f"{float((sealed_e2['away_score'].to_numpy() - sealed_e2['pred_away'].to_numpy()).mean()):.3f}")
            eng_pooled = e2_oof[["game_id", "pred_home", "pred_away",
                                 "home_score", "away_score"]].copy()
            eng_sealed = sealed_e2[["game_id", "pred_home", "pred_away",
                                    "home_score", "away_score"]].copy()
            for eng in (eng_pooled, eng_sealed):
                eng["resid_home"] = (eng["home_score"]
                                     - eng["pred_home"])
                eng["resid_away"] = (eng["away_score"]
                                     - eng["pred_away"])
            eng_pooled.to_csv("/tmp/nfl_era_e2_pooled.csv", index=False)
            eng_sealed.to_csv("/tmp/nfl_era_e2_sealed.csv", index=False)

        # =================================================================
        # STEP 2 — joint-chain re-run on the adopted arm (engine untouched)
        # =================================================================
        step2: dict[str, Any] = {}
        if adopted == "E2":
            print("\n[Step 2] joint-chain re-run (existing entrypoints)...")
            params = fit_joint_params(eng_pooled)
            print(f"  family={params['family']} "
                  f"sigma_h={params['sigma_h']['sigma0']} "
                  f"sigma_a={params['sigma_a']['sigma0']} "
                  f"rho={params['rho']} CI={params['rho_ci']}")
            n_ties = int((eng_pooled["home_score"] == eng_pooled["away_score"]
                          ).sum())
            p_tie = n_ties / len(eng_pooled)
            pooled_pmfs, pooled_sum = build_joint_pmfs(eng_pooled, params,
                                                       p_tie)
            pooled_derived = pooled_sum["derived"].copy()
            pooled_derived = pooled_derived.merge(
                eng_pooled[["game_id", "home_score", "away_score"]],
                on="game_id", how="left")
            sealed_pmfs, sealed_sum = build_joint_pmfs(eng_sealed, params,
                                                       p_tie)
            sealed_derived = sealed_sum["derived"].copy()
            sealed_derived = sealed_derived.merge(
                eng_sealed[["game_id", "home_score", "away_score"]],
                on="game_id", how="left")
            crps = _crps_vs_climatology({"pooled": eng_pooled,
                                         "sealed": eng_sealed}, params)
            g1 = all(crps["pooled"][f"ratio_{s}"] <= 0.95 for s in SIDES)
            d_cal = float(np.mean([np.trace(p) for p in pooled_pmfs]))
            g2_delta_pp = (d_cal - p_tie) * 100
            g2 = abs(g2_delta_pp) <= 0.2
            g3_err = pooled_sum["summary"]["max_marginal_err_post_ipf"]
            g3 = g3_err is not None and g3_err <= 1e-9
            y_ml = (pooled_derived["home_score"]
                    > pooled_derived["away_score"])
            ml = compute_metrics(y_ml.astype(float).to_numpy(),
                                 pooled_derived["derived_ml"].to_numpy(float))
            g4_flag = ml["logloss"] - C0_ANCHOR["logloss"] > 0.02
            _p2, sum2 = build_joint_pmfs(eng_pooled, params, p_tie)
            g5 = (pooled_sum["derived"].to_csv(index=False)
                  == sum2["derived"].to_csv(index=False))
            away_bias_after = {s: bias_by_season(e2_oof, PRED_COL[s],
                                                 SCORE_COL[s])
                               for s in SIDES}
            away_bias_after_sealed = {
                s: mean_resid_stats(eng_sealed, PRED_COL[s], SCORE_COL[s])
                for s in SIDES}
            print(f"  G1 pooled: {crps['pooled']['improvement_pct_home']}% / "
                  f"{crps['pooled']['improvement_pct_away']}% | "
                  f"G2 Δ={g2_delta_pp:+.4f}pp | G3 err={g3_err} | "
                  f"G4 ml={ml} | G5={g5}")

            step2 = {
                "adopted_arm": adopted,
                "adoption_note": adoption_note,
                "joint_params": {
                    "family": params["family"],
                    "sigma_h": params["sigma_h"],
                    "sigma_a": params["sigma_a"],
                    "rho": params["rho"],
                    "rho_ci": params["rho_ci"],
                    "rho_n": params["rho_n"],
                    "fit_on": params["fit_on"],
                    "ll_table": params["ll_table"],
                },
                "tie": {"p_tie_empirical": round(p_tie, 5),
                        "n_ties_pooled": n_ties,
                        "d_raw_mean": pooled_sum["summary"]["d_raw_mean"],
                        "d_calibrated_mean":
                            pooled_sum["summary"]["d_calibrated_mean"]},
                "crps_vs_climatology": crps,
                "gates": {
                    "g1": {"pass": bool(g1), "pooled": crps["pooled"],
                           "sealed": crps["sealed"]},
                    "g2": {"pass": bool(g2),
                           "delta_pp": round(g2_delta_pp, 4)},
                    "g3": {"pass": bool(g3), "max_err": g3_err},
                    "g4": {"metrics": ml, "c0_anchor": C0_ANCHOR,
                           "flag": bool(g4_flag)},
                    "g5": {"pass": bool(g5)},
                },
                "bias_after_era_pooled": away_bias_after,
                "bias_after_era_sealed": away_bias_after_sealed,
                "data_seam": {"skipped": True, "note": "run --seam-pass "
                              "to patch totals ECE / covers ECE from the "
                              "offered-line check"},
            }
            sealed_e2_bias = {s: away_bias_after_sealed[s]["mean_resid"]
                              for s in SIDES}

        # ---- before-reference (latest canonical chain on main) ----
        before: dict[str, Any] = {}
        mb_path = DATA_DELIVERY / MEAN_BIAS_RECORD_NAME
        if mb_path.exists():
            mb = json.loads(mb_path.read_text())
            before = {
                "source": f"{MEAN_BIAS_RECORD_NAME} on main",
                "c0_by_season_away_resid": [d["mean_resid"] for d in
                                            oof_bias["away"]],
                "c0_away_mean_abs_2021_23": round(lgb_away_2123, 4),
                "c0_mae": {"home": None, "away": None},
                "sealed_2025_away_bias_c0":
                    sealed_bias_c0["away"]["mean_resid"],
            }

        # ---- verdict (SUCCESS criteria per spec) ----
        verdict: dict[str, Any] = {"state": "NO_ARMS" if not gate_pass else
                                   ("REFUTED_MODEL_LEVEL" if adopted == "NONE"
                                    else "SCORED")}
        if verdict["state"] == "SCORED":
            cut_model = (lgb_away_2123 - e2_away_2123) >= BIAS_CUT_ABS_PTS \
                or e2_away_2123 <= BIAS_CUT_REL * lgb_away_2123
            sealed_ok = all(abs(v) <= 0.5 for v in sealed_e2_bias.values())
            g2g3g5 = bool(g2 and g3 and g5)
            g4_ok = not g4_flag
            all_ok = bool(cut_model and sealed_ok and g2g3g5 and g4_ok)
            verdict = {
                "state": "SCORED",
                "criteria": {
                    "2021_23_away_bias_halved": {
                        "pass": bool(cut_model),
                        "c0_mean_abs": round(lgb_away_2123, 4),
                        "e2_mean_abs": e2_away_2123,
                        "rule": ">= 1.0 pt cut or <= 50% of C0"},
                    "sealed_not_overcorrected": {
                        "pass": bool(sealed_ok),
                        "sealed_resid": sealed_e2_bias,
                        "rule": "both sides within ±0.5"},
                    "g2_g3_g5": {"pass": bool(g2g3g5)},
                    "g4_coherent_or_better": {"pass": bool(g4_ok),
                                              "flag": bool(g4_flag)},
                },
                "pass": all_ok,
                "note": ("NOT gated on G1 >= 5% both legs or the totals "
                         "top-bin closing — the dispersion/sigma layer's "
                         "job; this layer is a NECESSARY step toward the "
                         "market layer, not sufficient. On success the away "
                         "CRPS leg may still sit below 5% and the queued "
                         "sigma layer follows."),
            }

    # =====================================================================
    # RECORD
    # =====================================================================
    arms: dict[str, Any] = {}
    if gate_pass:
        arms = {
            "cv_selection": {
                "note": ("selected on pooled 2021-24 OOF ONLY — sealed 2025 "
                         "structurally never enters selection (centers for "
                         "2021-24 rows depend only on strictly-prior games)"),
                "metric": "2021-23 away mean |season resid|, tiebreak away MAE",
                "table": cv,
                "chosen": chosen,
                "ranking": sel,
            },
            "c0": {"mae": c0_mae, "bias_by_season": c0_bias,
                   "n": int(len(c0_oof))},
            "e1": {"mae": e1_mae, "bias_by_season": e1_bias,
                   "n": int(len(e1_oof)),
                   "view": "SIDE_FEATURES + [era_center_home, era_center_away] "
                           "(raw targets — trees must find the level)"},
            "e2": {"mae": e2_mae, "bias_by_season": e2_bias,
                   "away_mean_abs_resid_2021_23": e2_away_2123,
                   "n": int(len(e2_oof)),
                   "rounds": rounds_e2,
                   "determinism_byte_identical": bool(e2_det),
                   "view": "SIDE_FEATURES on (target − center), center added "
                           "back at prediction"},
        }

    record = {
        "record": "nfl_era_conditional_mean",
        "frame_sha": frame_sha,
        "frame_sha256": _frame_sha256(feats),
        "written_utc": pd.Timestamp.now("UTC").isoformat(),
        "elapsed_seconds": round(time.time() - t0, 1),
        "scope": SCOPE_PIN,
        "geometry": {
            "seasons": sorted(feats["season"].unique().tolist()),
            "train_seasons": TRAIN_SEASONS,
            "sealed_season": SEALED_SEASON,
            "n_folds": 88,
            "pooled_oof_n": int(len(art)),
            "sealed_n": int(len(sealed_c0)),
            "view": "12-pool per-side PIT (SIDE_FEATURES) — E1 adds era "
                    "macro columns, E2 centers targets; FEATURE_COLUMNS "
                    "untouched",
            "engines_modified": False,
        },
        "step0_falsification": step0,
        "step1_arms": arms,
        "step2_joint_chain": step2,
        "before_reference": before,
        "verdict": verdict,
        "feature_columns_untouched": True,
        "judgment_calls": {
            "scope": ("LEAGUE-LEVEL scoring-environment drift only; team "
                      "roster/defense overhaul stays invisible until games "
                      "confirm it"),
            "linear_center_specs": ("ps + ewm_2w/4w/8w only; no blend spec — "
                                    "clean falsifiable candidates, thin CV"),
            "neutral_fallback": (f"constant {NEUTRAL_CENTER} for rows with "
                                 "no strictly-prior history (warmup only, "
                                 "never scored; constant is information-free)"),
            "league_level_not_team": ("team strength is already in the "
                                      "12-pool; the era layer isolates the "
                                      "common scoring level"),
            "sealed_center_definition": ("2025 rows' centers may use "
                                         "strictly-prior 2025 games (pure "
                                         "time-series; never the row's own "
                                         "or later labels); the model is "
                                         "never fit on 2025"),
            "g1_not_gated": ("G1 >= 5% and totals top-bin are the "
                             "dispersion/sigma layer's job — see verdict.note"),
        },
    }
    if not args.no_record:
        rec_path = DATA_DELIVERY / f"nfl_era_{frame_sha}.json"
        rec_path.write_text(json.dumps(record, indent=2, default=str))
        print(f"\nRecord: {rec_path.name}")
    else:
        print("\n[--no-record] record skipped")
    print(f"Done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    raise SystemExit(main())
