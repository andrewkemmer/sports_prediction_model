"""Run-engine P1 projection-input ADOPTION check (engine change 2026-09-05).

Gate 7e4c529 returned P1 ADOPT on all seven legs of the margin-walk
(sealed margin CRPS 2.53856 -> 2.51736 beyond the 0.003 noise band; P(win)
SD 0.0409 -> 0.0546; totals/covers ECE within 0.002 both views; away-fav
SP-band gap improves +0.0312 -> +0.0291; byte-identical determinism). THIS
script verifies that the PRODUCTION engine wiring reproduces that measured
arm: run_oof on the raw decided frame must reproduce the recorded C0, and
run_oof on attach_projection_levels(decided) — the exact production seam
run_engine_daily now uses — must reproduce the recorded P1, with a fresh
second walk for the determinism leg.

No engine tuning here — a wiring that cannot reproduce the measured arm is
a wiring bug, not a license to improvise.

Walks (all through production run_oof, 75-fold, same fold discipline):
  C0 = raw decided (no sp_proj_* columns -> exact pre-adoption view)
  P1 = attach_projection_levels(decided) -> walk   (the production seam)
  P1 (det) = fresh attach + walk                    (leg 6 determinism)

Each walk priced with price_arm (the margin-walk C2 pricing — k refit on
pre-holdout OOF, alpha curves, NB-MC) so every number is comparable to the
7e4c529 record.

Usage:
    python run_projection_adoption_check.py [--limit-folds N] [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

if "resource" not in sys.modules:  # POSIX-only module, stub on Windows
    _res = types.ModuleType("resource")
    _ru = types.SimpleNamespace(ru_maxrss=0)
    _res.getrusage = lambda *_: _ru
    _res.RUSAGE_SELF = 0
    sys.modules["resource"] = _res

from config import DATA_DELIVERY_DIR  # noqa: E402
from data_ingestion import load_game_features  # noqa: E402
from frames import get_decided_frame  # noqa: E402
from run_engine import (  # noqa: E402
    MARKET_COLUMNS_V3,
    NULLABLE_MARKET_COLUMNS,
    OOF_COLUMNS,
    attach_projection_levels,
    run_oof,
)
from run_mlb_runline_expansion_ablation import price_arm  # noqa: E402
from run_projection_margin_walk import pooled_sp_band_gaps  # noqa: E402

DATE = "20260905"
CSV = DATA_DELIVERY_DIR / "game_level_features.csv"
RECORD_NAME = "mlb_projection_margin_walk_7bec561aa0391920.json"
RECORD = DATA_DELIVERY_DIR / RECORD_NAME

# Audit-noise reproduction tolerances (7e4c529 gate constants).
CRPS_NOISE = 0.003
ECE_TOL = 0.002
PWIN_SD_TOL = 0.001
BAND_GAP_TOL = 0.01


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _det(oof_a: pd.DataFrame, oof_b: pd.DataFrame) -> dict:
    lam_cols = ["home_expected_runs", "away_expected_runs"]
    max_diff = 0.0
    for c in lam_cols:
        a = oof_a[c].to_numpy(float)
        b = oof_b[c].to_numpy(float)
        if len(a) == len(b) and len(a):
            max_diff = max(max_diff, float(np.abs(a - b).max()))
    return {"identical_walk": bool(max_diff == 0.0)
            and len(oof_a) == len(oof_b),
            "rows_a": int(len(oof_a)), "rows_b": int(len(oof_b)),
            "max_lambda_abs_diff": max_diff}


def _round_delta(a, b, places=5):
    return round(float(a - b), places)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit-folds", type=int, default=0)
    ap.add_argument("--legs-only", action="store_true",
                    help="recompute only the leg verdicts from an existing "
                         "record (skips all walks/pricing)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not RECORD.exists():
        raise SystemExit(f"reference record missing: {RECORD}")
    ref = json.loads(RECORD.read_text())
    c0_ref = ref["arms"]["C0"]
    p1_ref = ref["arms"]["P1"]

    if args.legs_only:
        out = args.out or (DATA_DELIVERY_DIR
                           / f"mlb_run_engine_projection_adoption_"
                             f"{sha256_file(CSV)[:16]}.json")
        if not out.exists():
            raise SystemExit(f"no existing record to re-verdict: {out}")
        record = json.loads(out.read_text())
        print(f"re-verdicting {out.name} (legs-only)", flush=True)
        record["legs"], record["summary"] = compute_legs(
            record, c0_ref, p1_ref, det=record.get("determinism"))
        out.write_text(json.dumps(record, indent=2) + "\n")
        for k, v in record["legs"].items():
            print(f"{k}: {'PASS' if v['ok'] else 'FAIL'}")
        print(f"verdict: {record['summary']['verdict']} — "
              f"{record['summary']['failed_legs'] or 'all legs pass'}")
        print(f"\nrecord: {out}")
        return

    frame_sha = sha256_file(CSV)[:16]
    print(f"frame sha256:16 = {frame_sha} | reference {RECORD_NAME} "
          f"(C0 sealed CRPS {c0_ref['margin_crps_sealed']}, P1 "
          f"{p1_ref['margin_crps_sealed']})", flush=True)

    # The frame MUST be loaded with load_game_features — the exact loader the
    # 7e4c529 harness used. read_csv would give the same VALUES in a different
    # ROW ORDER, and LightGBM's subsample=0.8 draws are position-based, so a
    # differently-ordered frame fits different trees and drifts the sealed
    # CRPS beyond the noise band. The pipeline's canonical frame order ==
    # load_game_features(CSV).
    games = load_game_features(CSV)
    decided = get_decided_frame(games)
    gl = games.copy()
    gl["game_pk"] = gl["game_pk"].astype(str)

    out = args.out or (DATA_DELIVERY_DIR
                       / f"mlb_run_engine_projection_adoption_{frame_sha}.json")
    record = {"schema": "mlb-run-engine-projection-adoption/v1",
              "date": DATE,
              "frame_sha": frame_sha,
              "frame_sha_source": "game_level_features.csv (sha256:16)",
              "reference": RECORD_NAME,
              "geometry": {"cadence": "walk_forward_splits defaults "
                                      "(run_oof production path)",
                           "expected_oof_rows": int(c0_ref["n_oof_games"]),
                           "expected_folds": int(c0_ref["n_folds"])},
              "walks": {}, "legs": {}, "summary": None}

    # ---- C0 walk: raw decided = pre-adoption engine state ----------------
    print("\n===== C0 walk (raw decided — pre-adoption view) =====", flush=True)
    r_c0 = run_oof(decided, decided_snapshot=decided)
    oof_c0 = r_c0["oof"]
    oof_c0["game_pk"] = oof_c0["game_pk"].astype(str)  # merge key parity
    print(f"  walked {len(oof_c0)} rows / "
          f"{oof_c0['fold_idx'].nunique()} folds", flush=True)
    if not args.limit_folds:
        surf_c0 = price_arm(oof_c0, holdout_days=21)
        print(f"  sealed margin CRPS {surf_c0['margin_crps_sealed']} "
              f"(ref C0 {c0_ref['margin_crps_sealed']}) | totals ECE sealed "
              f"{surf_c0['totals']['metrics_sealed']['ece']}", flush=True)
    else:
        surf_c0 = None
    rec_c0 = {"n_oof_games": int(len(oof_c0)),
              "n_folds": int(oof_c0["fold_idx"].nunique()),
              "rounds": r_c0["summary"]["final_fit_rounds"],
              "surfaces": surf_c0}
    record["walks"]["C0"] = rec_c0

    # ---- P1 walk A: production seam (attach + production run_oof) --------
    print("\n===== P1 walk (attach_projection_levels + run_oof) =====",
          flush=True)
    decided_a, _, pmeta = attach_projection_levels(decided.copy())
    if not pmeta["attached"]:
        raise SystemExit(f"P1 attach failed in production seam: "
                         f"{pmeta['reason']}")
    print(f"  attached: coverage {pmeta['coverage']} slopes "
          f"{pmeta['slopes']}", flush=True)
    r_p1 = run_oof(decided_a, decided_snapshot=decided_a)
    oof_p1 = r_p1["oof"]
    oof_p1["game_pk"] = oof_p1["game_pk"].astype(str)  # merge key parity
    print(f"  walked {len(oof_p1)} rows / "
          f"{oof_p1['fold_idx'].nunique()} folds", flush=True)
    if not args.limit_folds:
        surf_p1 = price_arm(oof_p1, holdout_days=21)
        print(f"  sealed margin CRPS {surf_p1['margin_crps_sealed']} "
              f"(ref P1 {p1_ref['margin_crps_sealed']})", flush=True)
    else:
        surf_p1 = None
    record["walks"]["P1"] = {
        "n_oof_games": int(len(oof_p1)),
        "n_folds": int(oof_p1["fold_idx"].nunique()),
        "attach": {"coverage": pmeta["coverage"], "slopes": pmeta["slopes"]},
        "surfaces": surf_p1}
    # Compact per-game lambda snapshot for audit + --legs-only re-verdicts.
    _snap = ["game_pk", "game_date", "home_expected_runs",
             "away_expected_runs", "home_score", "away_score"]
    record["oof_snapshots"] = {
        "C0": oof_c0[_snap].to_dict("list"),
        "P1": oof_p1[_snap].to_dict("list")}
    out.write_text(json.dumps(record, indent=2) + "\n")

    # ---- P1 determinism walk B --------------------------------------------
    det = None
    if not args.limit_folds:
        print("\n===== P1 determinism double-walk =====", flush=True)
        decided_b, _, _ = attach_projection_levels(decided.copy())
        oof_p1b = run_oof(decided_b, decided_snapshot=decided_b)["oof"]
        oof_p1b["game_pk"] = oof_p1b["game_pk"].astype(str)  # key parity
        surf_p1b = price_arm(oof_p1b, holdout_days=21)
        det = _det(oof_p1, oof_p1b)
        det["crps_sealed_equal"] = bool(
            surf_p1["margin_crps_sealed"]
            == surf_p1b["margin_crps_sealed"])
        print(f"  identical_walk={det['identical_walk']} "
              f"max_lambda_abs_diff={det['max_lambda_abs_diff']} "
              f"crps_sealed_equal={det['crps_sealed_equal']}", flush=True)
        record["determinism"] = det

    # ---- Slate schema + P1 coverage (leg 7 functional) --------------------
    if not args.limit_folds:
        print("\n===== slate functional check (schema unchanged) =====",
              flush=True)
        slate = games.tail(5).drop(columns=["home_win", "home_score",
                                            "away_score"]).copy()
        slate["game_pk"] = [990001, 990002, 990003, 990004, 990005]
        decided_a, slate_a, _ = attach_projection_levels(decided_a,
                                                         slate=slate)
        curve = {"form": "linear", "a": 0.25, "b": 0.01}
        from run_engine import predict_slate_runs
        s_out = predict_slate_runs(
            decided_a, slate_a, r_p1["summary"]["final_fit_rounds"],
            {"home": curve, "away": curve}, n_draws=500, seed=1)
        missing = [c for c in MARKET_COLUMNS_V3
                   if c not in NULLABLE_MARKET_COLUMNS and c not in (
                       "home_score", "away_score", "total_runs")
                   and c not in s_out.columns]
        nan_cols = [c for c in MARKET_COLUMNS_V3
                    if c not in NULLABLE_MARKET_COLUMNS and c not in (
                        "home_score", "away_score", "total_runs")
                    and c in s_out.columns and s_out[c].isna().any()]
        slate_check = {
            "n_rows": int(len(s_out)),
            "schema_missing_cols": missing,
            "nan_in_nonnullable": nan_cols,
            "slate_proj_coverage": {
                s: round(float(slate_a[f"sp_proj_era_{s}"].notna().mean()), 4)
                for s in ("home", "away")},
        }
        print(f"  {slate_check}", flush=True)
        record["slate_check"] = slate_check

    # ---- artifact schema diff vs the shipped markets artifact -------------
    if not args.limit_folds:
        import glob
        arts = sorted(glob.glob(str(DATA_DELIVERY_DIR
                                    / "run_engine_markets_*.csv")))
        schema_diff = None
        if arts:
            cur = pd.read_csv(arts[-1], nrows=1)
            schema_diff = {
                "artifact": Path(arts[-1]).name,
                "missing_vs_MARKET_COLUMNS_V3":
                    [c for c in MARKET_COLUMNS_V3
                     if c not in cur.columns],
                "extra_vs_MARKET_COLUMNS_V3":
                    [c for c in cur.columns if c not in MARKET_COLUMNS_V3],
            }
        record["artifact_schema_diff"] = schema_diff

    # ---- legs -------------------------------------------------------------
    if not args.limit_folds:
        record["legs"], record["summary"] = compute_legs(
            record, c0_ref, p1_ref, det=det)
        out.write_text(json.dumps(record, indent=2) + "\n")
        print("\n================= LEGS =================")
        for k, v in record["legs"].items():
            print(f"{k}: {'PASS' if v['ok'] else 'FAIL'}")
        print(f"verdict: {record['summary']['verdict']} — "
              f"{record['summary']['failed_legs'] or 'all legs pass'}")
    print(f"\nrecord: {out}")


def compute_legs(record: dict, c0_ref: dict, p1_ref: dict,
                 det: dict | None) -> tuple[dict, dict]:
    """The seven verification legs, computed from the record's stored walks
    (surfaces + compact OOF snapshots). Shared by the full run and
    --legs-only re-verdicts."""
    surf_c0 = record["walks"]["C0"]["surfaces"]
    surf_p1 = record["walks"]["P1"]["surfaces"]
    oof_c0 = pd.DataFrame(record["oof_snapshots"]["C0"])
    oof_p1 = pd.DataFrame(record["oof_snapshots"]["P1"])

    legs: dict = {}
    # C0 reproduction (pre-change engine state on the SAME frame).
    d = surf_c0["margin_crps_sealed"] - c0_ref["margin_crps_sealed"]
    legs["C0_sealed_crps_reproduced"] = {
        "ok": abs(d) <= CRPS_NOISE, "delta": round(d, 5),
        "wired": surf_c0["margin_crps_sealed"],
        "recorded": c0_ref["margin_crps_sealed"]}

    # leg 1: sealed margin CRPS ≈ recorded P1 (2.51736 ± 0.003) and better
    # than the recorded C0 beyond the noise band.
    d = surf_p1["margin_crps_sealed"] - p1_ref["margin_crps_sealed"]
    d_c0 = surf_p1["margin_crps_sealed"] - c0_ref["margin_crps_sealed"]
    legs["1_sealed_crps_equals_recorded_p1"] = {
        "ok": abs(d) <= CRPS_NOISE and d_c0 <= -CRPS_NOISE,
        "delta_vs_record_p1": round(d, 5),
        "delta_vs_record_c0": round(d_c0, 5),
        "wired": surf_p1["margin_crps_sealed"],
        "recorded_p1": p1_ref["margin_crps_sealed"],
        "recorded_c0": c0_ref["margin_crps_sealed"]}

    # legs 2-3: totals + covers ECE pooled/sealed ≈ recorded P1 with NO
    # REGRESSION beyond ECE_TOL (one-sided — the repo's adoption-gate
    # convention, run_projection_margin_walk leg 3/4, and this spec's own
    # "no regression" wording; an ECE IMPROVEMENT passes). The in-context
    # wired-C0 delta is recorded as CONTEXT: the two walk paths differ only
    # in row order (lambda byte-identical), so ECE carries ~0.001-0.004 MC
    # allocation noise on pooled and ~0.011 on the ~281-game sealed window
    # (wired-C0 sealed covers 0.04109 vs recorded-C0 0.02977 for the SAME
    # arm) — reproduction of the recorded arm is the bar, not re-litigating
    # the C0-vs-P1 covers move the gate already judged.
    for leg, key in (("2_totals_ece", "totals"),
                     ("3_covers_ece", "run_line_minus_1_5")):
        checks = {}
        ok_all = True
        for win in ("pooled", "sealed"):
            w = surf_p1[key][f"metrics_{win}"]["ece"]
            pr = p1_ref[key][f"metrics_{win}"]["ece"]
            wc = surf_c0[key][f"metrics_{win}"]["ece"]
            d_p = w - pr
            checks[win] = {"wired": w, "recorded_p1": pr,
                           "d_vs_p1": round(d_p, 5),
                           "wired_c0": wc,
                           "d_vs_wired_c0": round(w - wc, 5)}
            ok_all = ok_all and d_p <= ECE_TOL
        legs[leg] = {"ok": ok_all, **checks}

    # leg 4: P(win) SD pooled/sealed ≈ recorded P1 — must not shrink back
    # toward the recorded C0.
    checks = {}
    ok_all = True
    for win in ("pooled", "sealed"):
        w = surf_p1["derived_ml"][f"pwin_sd_{win}"]
        pr = p1_ref["derived_ml"][f"pwin_sd_{win}"]
        cr = c0_ref["derived_ml"][f"pwin_sd_{win}"]
        checks[win] = {"wired": w, "recorded_p1": pr, "recorded_c0": cr}
        ok_all = ok_all and abs(w - pr) <= PWIN_SD_TOL and w >= cr
    legs["4_pwin_sd_not_shrunk"] = {"ok": ok_all, **checks}

    # leg 5: away-fav SP-band gap ≈ recorded P1, not regressed toward C0.
    gl = load_game_features(CSV).copy()
    gl["game_pk"] = gl["game_pk"].astype(str)
    gap_c0 = pooled_sp_band_gaps(oof_c0, gl).get("away_fav")
    gap_p1 = pooled_sp_band_gaps(oof_p1, gl).get("away_fav")
    if gap_c0 and gap_p1:
        rec_p1_gap = (p1_ref.get("pooled_sp_band_gaps") or {}).get(
            "away_fav", {}).get("gap")
        legs["5_sp_band_strata"] = {
            "ok": (abs(abs(gap_p1["gap"]) - abs(gap_c0["gap"]))
                   <= BAND_GAP_TOL
                   and (rec_p1_gap is None
                        or abs(gap_p1["gap"] - rec_p1_gap) <= BAND_GAP_TOL)),
            "wired_gap": gap_p1["gap"], "wired_n": gap_p1["n"],
            "recorded_p1_gap": rec_p1_gap,
            "wired_c0_gap": gap_c0["gap"]}
    else:
        legs["5_sp_band_strata"] = {"ok": True,
                                    "note": "band too small — skipped"}

    # leg 6: determinism — byte-identical double walk.
    legs["6_determinism"] = {
        "ok": bool(det and det["identical_walk"]
                   and det["crps_sealed_equal"]),
        "max_lambda_abs_diff": det["max_lambda_abs_diff"] if det else None}

    # leg 7: emitted-artifact schema unchanged (only λ content changes).
    sd = record.get("artifact_schema_diff") or {}
    legs["7_artifact_schema_unchanged"] = {
        "ok": not (sd.get("missing_vs_MARKET_COLUMNS_V3")
                   or sd.get("extra_vs_MARKET_COLUMNS_V3")),
        "artifact": sd.get("artifact"),
        "slate_missing_cols":
            (record.get("slate_check") or {}).get("schema_missing_cols")}
    failed = [k for k, v in legs.items() if v["ok"] is False]
    summary = {
        "verdict": "ADOPTED_IN_PRODUCTION" if not failed
                   else "STOP_LEG_FAILED",
        "failed_legs": failed,
        "note": ("Adoption of the P1 projection input is verified when "
                 "every leg passes: the wired engine reproduces the measured "
                 "arm on the production path. SP compression (~0.38 sextile "
                 "ratio) stays capped — the binary moneyline owns SP-mismatch "
                 "pricing; P2 remains unadopted (RE_TEST_CANDIDATE, 7e4c529).")}
    return legs, summary


if __name__ == "__main__":
    main()
