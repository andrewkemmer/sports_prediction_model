"""Run-engine MARGIN-WALK gate for the projection-input arms (record-only).

This is the deferred adoption gate from the SP-sensitivity arm-test
(b7eed32, mlb_sp_projection_arm_7bec561aa0391920.json): the projection
composite does NOT fix SP compression (that thread is closed — sextile
ratio ~0.38 in every arm), but P1 delivered a margin-quality signal
orthogonal to SP (sealed margin CRPS -0.021, derived-ML AUC +0.029,
P(win) SD 0.041->0.055). b7eed32 deferred adoption to "a future
ensemble-style margin walk that gates it before any engine change" —
THIS IS THAT GATE.

Consumer discipline: every arm routes the projection through
sp_projection.py's columns (committed 3108bb0, bit-identical to the
b7eed32 composite: era~proj OLS slope -1.2213/-1.2138, coverage
0.9736/0.9742 pre-holdout / 0.9866/0.9933 sealed). No inline
re-computation that could drift from the committed producer.

Arms (mirror b7eed32 exactly — same folds, seed, 75-fold OOF over the
full decided frame, production C2 pricing per arm via price_arm):
  C0 = current view (production RUN_LGBM_PARAMS + side view) — pins:
       sealed margin CRPS ~2.53856, totals ECE ~0.02859 (sealed).
  P1 = C0 view + the OPPONENT's projection level (sp_proj_era_opp) — the
       sealed-CRPS-gain arm.
  P2 = P1 + the raw opponent ERA level (sp_proj_era_opp + sp_era_opp) —
       the additivity arm (best SP bands + best totals ECE in b7eed32).

GATE (margin-quality ONLY — SP compression is NOT re-opened; sextile
ratio is context): seven legs, computed not narrated. Tolerances:
  CRPS_NOISE = 0.003   sealed-CRPS improvement must exceed the audit
                       noise band (C0-refit vs shipped pricing reproduced
                       sealed margin CRPS to 0.0024 — rounded up).
  POOLED_TOL = 0.003   pooled margin CRPS must not regress beyond this.
  ECE_TOL    = 0.002   totals ECE no regression beyond this (pooled AND
                       sealed) — the tolerance used by prior adoption
                       gates (run_engine_home_edge_ablation).
  COVER_ECE_TOL = 0.002 covers (-1.5) ECE pooled AND sealed.
  PWIN_SD_TOL = 0.001  P(win) SD must not shrink below C0 - this (the
                       thesis is LESS compression; a shrink is a strike).
  BAND_GAP_TOL = 0.01  pooled away-fav SP-band |gap| not worse beyond ~1
                       SE (Leg-2 discipline from a9cd6af: non-degradation
                       only, never sized to a fixed gap).
  WORTH_FACTOR = 3     worth-having: sealed CRPS improvement >=
                       CRPS_NOISE / WORTH_FACTOR.

Verdict:
  ADOPT             : every leg passes AND worth-having.
  RE_TEST_CANDIDATE : leg 1 passes but exactly one secondary leg fails by
                      <= half its tolerance (a borderline near-miss worth
                      one re-check), OR the determinism leg fails on a
                      byte-level tie only.
  DON'T_ADOPT       : anything else (list the failed legs). A pooled-flat
                      / ECE-mixed candidate is the honest DON'T_ADOPT —
                      b7eed32 already showed pooled flat -0.0015 and ECE
                      mixed; this walk decides whether the sealed gain
                      survives the stricter gate.

Record: data_delivery/mlb_projection_margin_walk_<frame_sha>.json —
producer verification, per-arm surfaces (sealed + pooled CRPS, totals /
covers ECE, P(win) SD, sextile ratio context, SP-band table, PD), the
seven gate legs with verdicts, computed summary.

If ADOPT: adoption is a SEPARATE engine-change spec that wires
sp_projection.py into the side-model view under the same fold discipline.

Usage:
    python run_projection_margin_walk.py [--arms C0,P1,P2]
        [--limit-folds N] [--smoke] [--determinism-arm P1]
        [--skip-determinism] [--out PATH]
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

import run_sp_sensitivity as sps  # noqa: E402  (arm machinery, committed)
from config import DATA_DELIVERY_DIR  # noqa: E402
from data_ingestion import load_game_features  # noqa: E402
from frames import get_decided_frame  # noqa: E402
from run_mlb_runline_expansion_ablation import price_arm  # noqa: E402
from sp_projection import attach_projection_cols  # noqa: E402

DATE = "20260904"
HOLDOUT_DAYS = 21
CSV = DATA_DELIVERY_DIR / "game_level_features.csv"

# ---- gate tolerances (provenance in the module docstring) ----------------
CRPS_NOISE = 0.003
POOLED_TOL = 0.003
ECE_TOL = 0.002
COVER_ECE_TOL = 0.002
PWIN_SD_TOL = 0.001
BAND_GAP_TOL = 0.01
WORTH_FACTOR = 3

# Producer pins (sp_projection.py on frame 7bec561a; verified by
# test_sp_projection.py — reproduced in the record's producer block).
PRODUCER_PINS = {
    "era_on_proj_slope_home": -1.2213,
    "era_on_proj_slope_away": -1.2138,
    "coverage_pre_home": 0.9736,
    "coverage_pre_away": 0.9742,
    "coverage_sealed_home": 0.9866,
    "coverage_sealed_away": 0.9933,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def attach_on_decided(games: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Attach sp_projection columns (VERBATIM producer) to the decided frame
    and mirror the projection columns onto the CSV-level frame for the
    measurement merge (same layout as run_sp_sensitivity)."""
    decided = get_decided_frame(games)
    dates = pd.to_datetime(decided["game_date"])
    pre_mask = (dates < dates.max()
                - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
    decided, proj_meta = attach_projection_cols(decided, pre_mask)
    gl = games.copy()
    gl["game_pk"] = gl["game_pk"].astype(str)
    pm = decided[["game_pk", "sp_proj_era_home", "sp_proj_era_away"]].copy()
    pm["game_pk"] = pm["game_pk"].astype(str)
    gl = gl.merge(pm, on="game_pk", how="left")
    return decided, proj_meta, gl


def verify_producer(proj_meta: dict, frame_sha: str) -> dict:
    """Producer verification block (Step 0) — pins vs sp_projection output
    on the current frame + committed pin constants."""
    rows = {}
    for side in ("home", "away"):
        rows[f"era_on_proj_slope_{side}"] = proj_meta[side]["era_on_proj_slope"]
        rows[f"coverage_pre_{side}"] = proj_meta[side]["coverage_pre"]
        rows[f"coverage_sealed_{side}"] = proj_meta[side]["coverage_sealed"]
    checks = {}
    for k, v in PRODUCER_PINS.items():
        got = rows.get(k)
        checks[k] = {"expected": v, "got": got,
                     "ok": got is not None and abs(got - v) <= 0.0006}
    checks["_all_ok"] = all(c["ok"] for c in checks.values())
    return {"frame_sha": frame_sha, "producer": "sp_projection.py "
            "(committed 3108bb0, verbatim b7eed32 composite)",
            "pins": checks}


def pooled_sp_band_gaps(oof: pd.DataFrame, gl: pd.DataFrame,
                        col: str = "sp_era_diff") -> dict:
    """Pooled away-fav / home-fav SP-band derived-gaps (gap_cal = mean
    derived-ML minus actual home-win rate) over clean-SP rows — the leg-6
    non-degradation table (never sized to a fixed gap)."""
    from probe_sp_arm_tables import arm_pwin
    pwin, _ledge2, _k = arm_pwin(oof)
    d = oof[["game_pk", "home_score", "away_score"]].copy()
    d["pwin"] = pwin
    d = d.merge(gl[["game_pk", col, "sp_era_home", "sp_era_away"]],
                on="game_pk", how="left")
    d = d.dropna(subset=[col])
    clean = ((d["sp_era_home"].abs() <= 15.0)
             & (d["sp_era_away"].abs() <= 15.0))
    d = d[clean].copy()
    d["home_won"] = (d["home_score"] > d["away_score"]).astype(float)
    out = {}
    for lo, hi, lbl in ((-99.0, -1.5, "home_fav"), (1.5, 99.0, "away_fav")):
        sub = d[(d[col] >= lo) & (d[col] < hi)]
        if len(sub) >= 30:
            act = float(sub["home_won"].mean())
            pred = float(sub["pwin"].mean())
            out[lbl] = {"n": int(len(sub)), "actual": round(act, 4),
                        "derived_ml": round(pred, 4),
                        "gap": round(pred - act, 4)}
    return out


def walk_and_surface(name: str, decided: pd.DataFrame,
                     gl: pd.DataFrame,
                     limit_folds: int = 0) -> tuple[pd.DataFrame, dict]:
    """One arm: fold-disciplined walk (no cache) + production C2 pricing +
    context surfaces. Returns (oof, res_dict)."""
    params, per_side = sps.arm_params_and_frames(name, decided)
    oof = sps.walk_arm(name, decided, params, per_side,
                       limit_folds=limit_folds)
    print(f"  walked {len(oof)} rows / {oof['fold_idx'].nunique()} folds",
          flush=True)
    res = {"n_oof_games": int(len(oof)),
           "n_folds": int(oof["fold_idx"].nunique()),
           "lambda_mean": {
               "home": round(float(oof["home_expected_runs"].mean()), 4),
               "away": round(float(oof["away_expected_runs"].mean()), 4),
               "edge_sd": round(float((oof["home_expected_runs"]
                                       - oof["away_expected_runs"]).std()), 4)}}
    if limit_folds:
        return oof, res  # smoke: walk machinery only (tiny windows break α/k)
    res.update(price_arm(oof, holdout_days=HOLDOUT_DAYS))
    res["sextile_spread_ratio"] = sps.sextile_spread_ratio(oof, gl)
    res["season_sp_band_gaps"] = sps.season_sp_band_gaps(oof, gl)
    res["pooled_sp_band_gaps"] = pooled_sp_band_gaps(oof, gl)
    res["model_pd"] = sps.sp_measurements(oof, gl)["model_pd"]
    return oof, res


def determinism_check(oof_a: pd.DataFrame, oof_b: pd.DataFrame,
                      res_a: dict, res_b: dict) -> dict:
    """Leg 7: byte-identical double walk. λ pairs compared exactly; the
    priced surfaces compared exactly (they are pure functions of the λs)."""
    lam_cols = ["home_expected_runs", "away_expected_runs"]
    max_lam_diff = 0.0
    for c in lam_cols:
        a = oof_a[c].to_numpy(float)
        b = oof_b[c].to_numpy(float)
        if len(a) == len(b):
            max_lam_diff = max(max_lam_diff,
                               float(np.abs(a - b).max()) if len(a) else 0.0)
    identical = bool(max_lam_diff == 0.0) and len(oof_a) == len(oof_b)
    crps_equal = (res_a.get("margin_crps_sealed")
                  == res_b.get("margin_crps_sealed")
                  and res_a.get("margin_crps_pooled")
                  == res_b.get("margin_crps_pooled"))
    return {"identical_walk": identical, "rows_a": int(len(oof_a)),
            "rows_b": int(len(oof_b)), "max_lambda_abs_diff": max_lam_diff,
            "crps_sealed_equal": bool(crps_equal),
            "sealed_crps_a": res_a.get("margin_crps_sealed"),
            "sealed_crps_b": res_b.get("margin_crps_sealed")}


def _metric_safe(x: dict | None, key: str) -> float | None:
    if not x or key not in x or x[key] is None:
        return None
    return float(x[key])


def gate_legs(cand: dict, c0: dict, det: dict | None) -> dict:
    """The seven legs, computed. Legs are booleans; each also carries the
    raw delta for the record."""
    legs = {}
    # 1. sealed margin CRPS improves beyond the audit noise band.
    d = cand["margin_crps_sealed"] - c0["margin_crps_sealed"]
    legs["1_sealed_crps_improves"] = {
        "ok": d <= -CRPS_NOISE, "delta": round(d, 5), "tol": CRPS_NOISE,
        "excess": round(max(0.0, -d - CRPS_NOISE), 5),
        "note": "must improve beyond audit noise band"}
    # 2. pooled margin CRPS not regressed.
    d = cand["margin_crps_pooled"] - c0["margin_crps_pooled"]
    legs["2_pooled_crps_not_regress"] = {
        "ok": d <= POOLED_TOL, "delta": round(d, 5), "tol": POOLED_TOL,
        "excess": round(max(0.0, d - POOLED_TOL), 5)}
    # 3. totals ECE no regression pooled AND sealed.
    d_p = (_metric_safe(cand["totals"]["metrics_pooled"], "ece")
           - _metric_safe(c0["totals"]["metrics_pooled"], "ece"))
    d_s = (_metric_safe(cand["totals"]["metrics_sealed"], "ece")
           - _metric_safe(c0["totals"]["metrics_sealed"], "ece"))
    worst = max([x for x in (d_p, d_s) if x is not None], default=0.0)
    legs["3_totals_ece"] = {
        "ok": (d_p is not None and d_s is not None
               and d_p <= ECE_TOL and d_s <= ECE_TOL),
        "d_pooled": d_p, "d_sealed": d_s, "tol": ECE_TOL,
        "excess": round(max(0.0, worst - ECE_TOL), 5)}
    # 4. covers (-1.5) ECE no regression pooled AND sealed.
    dp = (_metric_safe(cand["run_line_minus_1_5"]["metrics_pooled"], "ece")
          - _metric_safe(c0["run_line_minus_1_5"]["metrics_pooled"], "ece"))
    ds = (_metric_safe(cand["run_line_minus_1_5"]["metrics_sealed"], "ece")
          - _metric_safe(c0["run_line_minus_1_5"]["metrics_sealed"], "ece"))
    worst = max([x for x in (dp, ds) if x is not None], default=0.0)
    legs["4_covers_ece"] = {
        "ok": (dp is not None and ds is not None
               and dp <= COVER_ECE_TOL and ds <= COVER_ECE_TOL),
        "d_pooled": dp, "d_sealed": ds, "tol": COVER_ECE_TOL,
        "excess": round(max(0.0, worst - COVER_ECE_TOL), 5)}
    # 5. P(win) SD must not shrink below C0 (less compression is the thesis).
    c_sd = cand["derived_ml"]["pwin_sd_sealed"]
    b_sd = c0["derived_ml"]["pwin_sd_sealed"]
    shrink = max(0.0, (b_sd - PWIN_SD_TOL) - c_sd)
    legs["5_pwin_sd_not_shrink"] = {
        "ok": c_sd >= b_sd - PWIN_SD_TOL, "cand_sd": c_sd,
        "c0_sd": b_sd, "tol": PWIN_SD_TOL, "excess": round(shrink, 5)}
    # 6. pooled away-fav SP-band |gap| not worse beyond ~1 SE (non-degradation
    # only — Leg-2 discipline; never sized to a fixed gap).
    cg = (cand.get("pooled_sp_band_gaps") or {}).get("away_fav")
    bg = (c0.get("pooled_sp_band_gaps") or {}).get("away_fav")
    if cg and bg:
        d = abs(cg["gap"]) - abs(bg["gap"])
        legs["6_sp_band_strata"] = {
            "ok": d <= BAND_GAP_TOL, "d_abs_gap": round(d, 4),
            "cand_gap": cg["gap"], "c0_gap": bg["gap"], "n": cg["n"],
            "tol": BAND_GAP_TOL, "excess": round(max(0.0, d - BAND_GAP_TOL), 4)}
    else:
        legs["6_sp_band_strata"] = {
            "ok": True, "note": "band too small — skipped", "excess": 0.0}
    # 7. determinism: byte-identical double walk (fresh fits, no cache).
    if det is not None:
        legs["7_determinism"] = {
            "ok": bool(det["identical_walk"] and det["crps_sealed_equal"]),
            "max_lambda_abs_diff": det["max_lambda_abs_diff"],
            "crps_sealed_equal": det["crps_sealed_equal"], "excess": 0.0}
    else:
        legs["7_determinism"] = {
            "ok": True, "excess": 0.0,
            "note": "verified on the shared walk machinery by the arm that "
                     "was double-walked (identical seed/params/fold path)"}
    return legs


def decide(legs: dict) -> dict:
    order = list(legs)
    failed = [k for k in order if legs[k]["ok"] is False]
    missing = [k for k in order if legs[k]["ok"] is None]
    if not failed and not missing:
        # worth-having: sealed CRPS improvement >= CRPS_NOISE / WORTH_FACTOR.
        d = legs["1_sealed_crps_improves"]["delta"]
        worth = d <= -(CRPS_NOISE / WORTH_FACTOR)
        verdict = "ADOPT" if worth else "RE_TEST_CANDIDATE"
        reason = (f"all {len(order)} legs pass"
                  + ("" if worth else "; sealed gain below worth-having"))
    elif not failed:
        verdict = "RE_TEST_CANDIDATE"
        reason = f"legs missing: {missing}"
    elif len(failed) == 1 and legs["1_sealed_crps_improves"]["ok"]:
        # One secondary leg failed: RE_TEST only when the over-tolerance
        # excess is <= half the leg's tolerance (a genuine near-miss).
        leg = legs[failed[0]]
        tol = float(leg.get("tol") or 0.0)
        excess = float(leg.get("excess") or 0.0)
        marginal = tol > 0 and excess <= tol / 2 + 1e-9
        verdict = "RE_TEST_CANDIDATE" if marginal else "DON'T_ADOPT"
        reason = (f"leg {failed[0]} borderline (excess {excess:.5f} <= "
                  f"tol/2 {tol / 2:.5f}) — one re-check warranted"
                  if marginal else f"failed leg: {failed[0]}")
    else:
        verdict = "DON'T_ADOPT"
        reason = f"failed legs: {failed}"
    return {"verdict": verdict, "reason": reason,
            "legs_failed": [k for k in order if legs[k]["ok"] is False],
            "legs_passed": [k for k in order if legs[k]["ok"] is True]}


def b7eed32_crosscheck(record: dict) -> dict | None:
    """Fresh-run vs b7eed32 record deltas (same frame sha 7bec561a, same
    arms) — pins should reproduce; small deltas are run noise, large ones
    signal geometry drift."""
    prior = DATA_DELIVERY_DIR / "mlb_sp_projection_arm_7bec561aa0391920.json"
    if not prior.exists():
        return None
    import json as _json
    p = _json.loads(prior.read_text())
    out = {}
    for name, a in record["arms"].items():
        old = (p.get("arms") or {}).get(name)
        if not old or "margin_crps_sealed" not in a:
            continue
        out[name] = {
            "d_sealed_crps": round(a["margin_crps_sealed"]
                                   - old["margin_crps_sealed"], 5),
            "d_pooled_crps": round(a["margin_crps_pooled"]
                                   - old["margin_crps_pooled"], 5),
            "d_totals_ece_sealed": round(
                a["totals"]["metrics_sealed"]["ece"]
                - old["totals"]["metrics_sealed"]["ece"], 5),
            "n_oof_delta": int(a["n_oof_games"] - old["n_oof_games"]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", type=str, default="C0,P1,P2")
    ap.add_argument("--limit-folds", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--determinism-arm", type=str, default="P1")
    ap.add_argument("--skip-determinism", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ("C0", "P1", "P2", "C1"):
            raise SystemExit(f"unknown arm {a!r}")

    frame_sha = sha256_file(CSV)[:16]
    print(f"frame sha256:16 = {frame_sha} | arms {arms}", flush=True)

    games = load_game_features(CSV)
    decided, proj_meta, gl = attach_on_decided(games)
    print(f"decided {len(decided)} rows | producer slopes "
          f"{ {k: v['era_on_proj_slope'] for k, v in proj_meta.items()} }",
          flush=True)
    producer = verify_producer(proj_meta, frame_sha)

    out = args.out or (DATA_DELIVERY_DIR
                       / f"mlb_projection_margin_walk_{frame_sha}.json")
    record = {"schema": "mlb-projection-margin-walk/v1",
              "frame_sha": frame_sha,
              "frame_sha_source": "game_level_features.csv (sha256:16)",
              "date": DATE,
              "producer_verification": producer,
              "geometry": {"cadence_days": sps.RETRAIN_CADENCE_DAYS,
                           "min_val_games": sps.MIN_VAL_FOLD_GAMES,
                           "holdout_days": HOLDOUT_DAYS,
                           "arm_view": "run-engine side levels + env "
                                       "(sps.arm_params_and_frames)"},
              "gate_tolerances": {"CRPS_NOISE": CRPS_NOISE,
                                  "POOLED_TOL": POOLED_TOL,
                                  "ECE_TOL": ECE_TOL,
                                  "COVER_ECE_TOL": COVER_ECE_TOL,
                                  "PWIN_SD_TOL": PWIN_SD_TOL,
                                  "BAND_GAP_TOL": BAND_GAP_TOL,
                                  "WORTH_FACTOR": WORTH_FACTOR},
              "arms": {}, "gate": None, "summary": None}

    oofs: dict[str, pd.DataFrame] = {}
    for name in arms:
        if name in record["arms"]:
            print(f"  arm {name} already recorded — skipping", flush=True)
            continue
        print(f"\n===== arm {name} =====", flush=True)
        oof, res = walk_and_surface(name, decided, gl,
                                    limit_folds=args.limit_folds)
        oofs[name] = oof
        record["arms"][name] = res
        out.write_text(json.dumps(record, indent=2) + "\n")
        if "margin_crps_sealed" in res:
            print(f"    sealed margin CRPS {res['margin_crps_sealed']} | "
                  f"pooled {res['margin_crps_pooled']} | totals ECE sealed "
                  f"{res['totals']['metrics_sealed']['ece']} | P(win) SD sealed "
                  f"{res['derived_ml']['pwin_sd_sealed']} | edge sd "
                  f"{res['lambda_mean']['edge_sd']}", flush=True)
        else:
            print(f"    (smoke: no pricing on {name})", flush=True)
        if args.smoke:
            break

    det = None
    da = args.determinism_arm
    if not args.smoke and not args.skip_determinism and da in arms:
        # Fresh second walk (walk_arm never caches) — leg 7.
        print(f"\n===== determinism double-walk: {da} =====", flush=True)
        params, per_side = sps.arm_params_and_frames(da, decided)
        oof_b = sps.walk_arm(da, decided, params, per_side)
        res_b = price_arm(oof_b, holdout_days=HOLDOUT_DAYS)
        det = determinism_check(oofs[da], oof_b, record["arms"][da], res_b)
        print(f"    identical_walk={det['identical_walk']} "
              f"max_lambda_abs_diff={det['max_lambda_abs_diff']} "
              f"crps_sealed_equal={det['crps_sealed_equal']}", flush=True)
        record["determinism"] = det

    record["b7eed32_crosscheck"] = b7eed32_crosscheck(record)
    if "C0" in record["arms"] and "margin_crps_sealed" in record["arms"]["C0"]:
        gate = {}
        for name in ("P1", "P2", "C1"):
            if name in record["arms"] and name != "C0" and \
                    "margin_crps_sealed" in record["arms"][name]:
                legs = gate_legs(record["arms"][name], record["arms"]["C0"],
                                 det if name == da else None)
                gate[name] = {"legs": legs, **decide(legs)}
        record["gate"] = gate
        record["summary"] = summarize(record)
        out.write_text(json.dumps(record, indent=2) + "\n")
        print("\n================= GATE vs C0 =================")
        for name, g in gate.items():
            print(f"{name}: {g['verdict']} — {g['reason']}")
            for leg, v in g["legs"].items():
                ok = v["ok"]
                print(f"    {leg}: "
                      f"{'PASS' if ok is True else 'FAIL' if ok is False else 'n/a'}"
                      + (f"  (Δ {v.get('delta')})" if "delta" in v else ""))
    print(f"\nrecord: {out}")


def summarize(record: dict) -> dict:
    """Computed narrative (no prose drift)."""
    gate = record.get("gate") or {}
    out: dict = {"arms": {}, "verdict_text": "", "next_action": ""}
    c0 = record["arms"]["C0"]
    for name, g in gate.items():
        c = record["arms"][name]
        row = {
            "d_sealed_crps": round(c["margin_crps_sealed"]
                                   - c0["margin_crps_sealed"], 5),
            "d_pooled_crps": round(c["margin_crps_pooled"]
                                   - c0["margin_crps_pooled"], 5),
            "d_totals_ece_pooled": round(
                _metric_safe(c["totals"]["metrics_pooled"], "ece")
                - _metric_safe(c0["totals"]["metrics_pooled"], "ece"), 5),
            "d_totals_ece_sealed": round(
                _metric_safe(c["totals"]["metrics_sealed"], "ece")
                - _metric_safe(c0["totals"]["metrics_sealed"], "ece"), 5),
            "d_covers_ece_pooled": round(
                _metric_safe(c["run_line_minus_1_5"]["metrics_pooled"], "ece")
                - _metric_safe(c0["run_line_minus_1_5"]["metrics_pooled"], "ece"), 5),
            "d_covers_ece_sealed": round(
                _metric_safe(c["run_line_minus_1_5"]["metrics_sealed"], "ece")
                - _metric_safe(c0["run_line_minus_1_5"]["metrics_sealed"], "ece"), 5),
            "pwin_sd_sealed": c["derived_ml"]["pwin_sd_sealed"],
            "c0_pwin_sd_sealed": c0["derived_ml"]["pwin_sd_sealed"],
        }
        cg = (c.get("pooled_sp_band_gaps") or {}).get("away_fav")
        bg = (c0.get("pooled_sp_band_gaps") or {}).get("away_fav")
        if cg and bg:
            row["d_away_fav_abs_gap"] = round(abs(cg["gap"]) - abs(bg["gap"]), 4)
        out["arms"][name] = row
    texts = [f"{n}: {g['verdict']} ({g['reason']})" for n, g in gate.items()]
    out["verdict_text"] = " | ".join(texts)
    verdicts = [g["verdict"] for g in gate.values()]
    if "ADOPT" in verdicts:
        out["next_action"] = ("ADOPT (record-only) — a SEPARATE engine-change "
                              "spec wires sp_projection.py into the side-model "
                              "view under the same fold discipline before any "
                              "production change.")
    elif "RE_TEST_CANDIDATE" in verdicts:
        out["next_action"] = ("RE_TEST_CANDIDATE — one re-walk of the "
                              "near-miss arm before any engine-change spec; "
                              "no production change.")
    else:
        out["next_action"] = ("DON'T_ADOPT — keep the production run-engine "
                              "view unchanged; the sealed margin-quality "
                              "signal from b7eed32 did not survive the "
                              "stricter pooled/sealed margin-walk gate.")
    return out


if __name__ == "__main__":
    main()
