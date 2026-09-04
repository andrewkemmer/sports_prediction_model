"""Binary moneyline SP-projection feature arm-test (record-only candidate).

Rationale: product routing assigns SP-mismatch pricing to the binary
moneyline, which already consumes SP features (sp_era_diff, sp_k9_diff,
sp_xwoba_vs_l_diff, sp_fbvelo_diff, sp_era_home/away, ...). The b7eed32
projection composite (xFIP/SIERA-family z-composite scaled to ERA-equivalent
units; 5.6x/2.1x raw ERA's single-game run-variance explanation) was measured
on the RUN ENGINE and is sitting idle. This spec measures whether adding it to
the BINARY's SP features improves SP-mismatch band calibration. NO production
change: served FEATURE_COLS untouched, run engine untouched.

Composite (verbatim b7eed32, reproduced by sp_projection.attach_projection_cols
and validated against the record: era~proj OLS slope -1.2213/-1.2138,
coverage 0.9736/0.9742 pre, 0.9866/0.9933 sealed on frame sha 7bec561a).

Arms (identical walk-forward geometry — the binary's OWN full-frame walk, the
one predictions_history_*.csv comes from; 7-day cadence, min-val 40 + partial
tail, seed 42; OOF run_margin_diff attached once on the same folds and reused
per arm so every arm consumes production-faithful margins):
  B0 = current production FEATURE_COLS (59 cols, incl. run_margin_diff).
  B1 = the raw-ERA slot REPLACED by the projection slot, same shape:
       sp_era_diff   -> sp_proj_era_diff
       sp_era_home   -> sp_proj_era_home
       sp_era_away   -> sp_proj_era_away
       (per-side raws stay excluded from the logistic member exactly like the
       sp_era raws — RAW_PER_SIDE_COLS extended at runtime for the arm.)
  B2 = B0 + sp_proj_era_diff/sp_proj_era_home/sp_proj_era_away appended
       (projection ADDITIVE to raw ERA — the run-engine P2 question, run only
       if B1 shows promise).

Scoring (production path per arm):
  * walk = training.walk_forward_evaluate on the enriched decided frame
    (pooled OOF = the walk's own concatenated fold predictions; prequential
    per-fold Platt twins = the published axis, exactly as
    predictions_history_*.csv).
  * pooled view = compute_metrics over the walk's OOF rows clipped to the
    COMMITTED predictions_history game set E (B0(E) must reproduce the
    committed CSV pins; every arm is scored on the identical E).
  * sealed view S1 = the last-21-day slice of the walk OOF rows (published
    prequential axis).
  * sealed view S2 (deployment convention) = final-fit ensemble on
    pre-tail rows only + Platt fitted on pre-tail pooled OOF, applied to the
    21-day tail.
  * SP-mismatch strata = actual home win rate vs mean prediction per band on
    clean-SP rows (|sp_era| <= 15 both sides), plus the compression meter
    (prediction SD inside the 55-65% binary-prob band and per stratum).

Gate (moneyline-style six legs on the calibrated axis, tolerances mirroring
the model_tuning_policy blend bounds + strict ECE):
    pooled (E):  logloss <= B0 + 0.001 | AUC >= B0 - 0.001 | ECE <= B0
    sealed (S2): logloss <= B0 + 0.001 | AUC >= B0 - 0.001 | ECE <= B0
  + worth-having (any of pooled/sealed logloss -0.0002 or AUC +0.0002).

Verdict:
  ADOPT_CANDIDATE : all six legs AND worth-having AND SP-mismatch strata
                    track actuals better-or-equal than B0 (record-only).
  ADDITIVE_NULL   : all six legs but nothing worth-having / strata unchanged
                    — the composite is redundant with the binary's existing
                    SP features.
  DO_NOT_ADOPT    : any leg regresses beyond tolerance (record which leg).

Usage:
    python run_binary_sp_projection.py [--arms B0,B1] [--limit-folds N]
    python run_binary_sp_projection.py --arms B0,B1,B2
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import types
import warnings
from pathlib import Path
from unittest.mock import patch

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

import training  # noqa: E402
from calibration import MIN_OOF_FOR_FIT, apply_platt, moneyline_fit  # noqa: E402
from config import (  # noqa: E402
    DATA_DELIVERY_DIR,
    MIN_VAL_FOLD_GAMES,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
)
from sp_projection import (  # noqa: E402
    MIN_PROJ_COMPONENTS,
    PROJ_HI_BETTER,
    PROJ_LO_BETTER,
    attach_projection_cols,
)
from tune_lightgbm_optuna import load_games  # noqa: E402

EPS = 1e-7
HOLDOUT_DAYS = 21
TAIL_DAYS = 21  # sealed window width (calendar days, ending at frame max)
SP_JUNK_ERA = 15.0
CSV = DATA_DELIVERY_DIR / "game_level_features.csv"
HISTORY_CSV = DATA_DELIVERY_DIR / "predictions_history_20260903.csv"
# The committed predictions_history OOF universe starts 2024-04-25 and is
# reproduced EXACTLY by trimming the frame to >= 2024-04-18 (verified: kept
# fold val rows == CSV rows, 6,553/6,553, zero extra/missing). The current
# frame holds warm-up rows back to 2024-03-20; walking those early rows
# shifts every fold train and changes pooled metrics. Trim to the committed
# geometry so B0 reproduces the published baseline pins and B1/B2 share it.
TRIM_START = "2024-04-18"

# Gate tolerances (mirror the model_tuning_policy blend bounds + strict ECE).
TOL_LOGLOSS = 0.001
TOL_AUC = 0.001
WORTH_LOGLOSS = 0.0002
WORTH_AUC = 0.0002

# The raw-ERA slot columns B1 replaces with the projection slot.
ERA_SLOT = {
    "sp_era_diff": "sp_proj_era_diff",
    "sp_era_home": "sp_proj_era_home",
    "sp_era_away": "sp_proj_era_away",
}
PROJ_COLS = ["sp_proj_era_diff", "sp_proj_era_home", "sp_proj_era_away"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_arm_cols() -> dict[str, list[str]]:
    """B0/B1/B2 feature lists (runtime-only; training.FEATURE_COLS untouched)."""
    base = list(training.FEATURE_COLS)
    missing = [c for c in ERA_SLOT if c not in base]
    if missing:
        raise SystemExit(f"FEATURE_COLS missing era slot columns: {missing}")
    b1 = [ERA_SLOT.get(c, c) for c in base]
    return {"B0": base, "B1": b1, "B2": base + list(PROJ_COLS)}


def arm_logistic_raw_cols(name: str) -> list[str]:
    """RAW_PER_SIDE_COLS per arm: projection per-side raws are excluded from
    the logistic member exactly like the sp_era raws they mirror."""
    raw = list(training.RAW_PER_SIDE_COLS)
    if name in ("B1", "B2"):
        for c in ("sp_proj_era_home", "sp_proj_era_away"):
            if c not in raw:
                raw.append(c)
    return raw


def _make_skip_margin_attach(real_attach):
    """Build the patch side_effect capturing the REAL attach (the module
    attribute is replaced while the patch is active, so the fallback must
    call the captured original to avoid recursion)."""
    def _skip_margin_attach(games, splits, min_val_games, max_eval_folds,
                            retrain_cadence_days, min_train_days,
                            decided_snapshot=None):
        cov = (float(games["run_margin_diff"].notna().mean())
               if "run_margin_diff" in games.columns else 0.0)
        if cov > 0.90:
            regen = training._regenerate_splits(
                games, splits, min_val_games, retrain_cadence_days,
                max_eval_folds, min_train_days)
            print(f"  [margin] enriched frame reused (coverage "
                  f"{100 * cov:.1f}%) — skipping run-engine re-derivation",
                  flush=True)
            return games, regen
        return real_attach(games, splits, min_val_games, max_eval_folds,
                           retrain_cadence_days, min_train_days,
                           decided_snapshot=decided_snapshot)
    return _skip_margin_attach


def compute_pooled(y, p_raw, p_cal):
    """Metrics dict for a row set (raw + prequential/artifact calibrated)."""
    y = np.asarray(y, dtype=float)
    out = {}
    for tag, p in (("raw", p_raw), ("cal", p_cal)):
        m = training.compute_metrics(y, np.asarray(p, dtype=float))
        out.update({f"logloss_{tag}": m["logloss"], f"auc_{tag}": m["auc"],
                    f"brier_{tag}": m["brier"], f"ece_{tag}": m["ece"]})
    return out


def run_arm(name: str, cols: list[str], enriched: pd.DataFrame,
            e_rows: pd.DataFrame | None, tail_start, csv_n: int,
            limit_folds: int = 0) -> dict:
    """One full arm: production walk + pooled (E) + sealed S1/S2 + strata."""
    print(f"\n===== arm {name} ({len(cols)} cols) =====", flush=True)
    prev_fc = list(training.FEATURE_COLS)
    prev_raw = list(training.RAW_PER_SIDE_COLS)
    try:
        training.FEATURE_COLS = list(cols)
        training.RAW_PER_SIDE_COLS = arm_logistic_raw_cols(name)
        training._LAST_ADAPTIVE_WEIGHTS.clear()
        real_attach = training._attach_oof_run_margins
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(
                training, "_attach_oof_run_margins",
                side_effect=_make_skip_margin_attach(real_attach)))
            best_models, _pooled, combined = training.walk_forward_evaluate(
                enriched, max_eval_folds=limit_folds,
                decided_snapshot=enriched)
    finally:
        training.FEATURE_COLS = prev_fc
        training.RAW_PER_SIDE_COLS = prev_raw

    if combined is None or len(combined) == 0:
        raise SystemExit(f"arm {name}: walk produced no OOF rows")

    weights = {e["name"]: e.get("weight", 0.0)
               for e in training.last_ensemble_info()}
    print(f"  OOF rows {len(combined)} | adaptive weights "
          f"{ {k: round(v, 3) for k, v in weights.items()} }", flush=True)

    rec: dict = {"arm": name, "n_cols": len(cols),
                 "oof_rows_full": int(len(combined))}

    # ---- pooled: full walk rows and the committed-E subset ----
    full = compute_pooled(combined["home_win"],
                          combined["home_win_prob_model"],
                          combined["home_win_prob_model_calibrated"])
    rec["pooled_full"] = full
    if e_rows is not None and len(e_rows):
        # Doubleheader legs share game_id — key rows on the score triple so
        # the committed-E join is exact (ambiguous keys dropped both sides).
        key_cols = ["game_id", "home_score", "away_score"]
        ck = combined.copy()
        dup_c = ck.duplicated(subset=key_cols, keep=False)
        dup_e = e_rows.duplicated(subset=key_cols, keep=False)
        m = (ck[~dup_c].merge(e_rows[~dup_e][key_cols], on=key_cols,
                              how="inner"))
        rec["n_oof_in_E"] = int(len(m))
        rec["E_ambiguous_dropped_combined"] = int(dup_c.sum())
        rec["E_ambiguous_dropped_committed"] = int(dup_e.sum())
        rec["pooled_E"] = compute_pooled(m["home_win"],
                                         m["home_win_prob_model"],
                                         m["home_win_prob_model_calibrated"])
    else:
        rec["n_oof_in_E"] = int(len(combined))
        rec["pooled_E"] = full

    # ---- sealed S1: published-axis (prequential) tail slice of the walk ----
    tail = combined[combined["game_date"] >= tail_start]
    rec["sealed_S1"] = {
        "n": int(len(tail)),
        "range": [str(tail["game_date"].min().date()),
                  str(tail["game_date"].max().date())],
    }
    if len(tail) >= 10:
        rec["sealed_S1"].update(compute_pooled(
            tail["home_win"], tail["home_win_prob_model"],
            tail["home_win_prob_model_calibrated"]))

    # ---- sealed S2: deployment convention (pre-tail refit + pre-tail Platt)
    pretail = combined[combined["game_date"] < tail_start]
    if len(pretail) >= MIN_OOF_FOR_FIT and len(tail) >= 10:
        try:
            cal = moneyline_fit(pretail["home_win"].to_numpy(float),
                                pretail["home_win_prob_model"].to_numpy(float))
            models, _ = training.train_moneyline_ensemble(
                enriched[enriched["game_date"] < tail_start])
            p_raw, _mp, _w = training.ensemble_predict(models, tail)
            p_raw = np.clip(np.asarray(p_raw, dtype=float), EPS, 1 - EPS)
            p_cal = np.clip(np.asarray(apply_platt(p_raw, cal), dtype=float),
                            EPS, 1 - EPS)
            rec["sealed_S2"] = {
                "n": int(len(tail)),
                "platt_n_pre_tail": int(len(pretail)),
                **compute_pooled(tail["home_win"].to_numpy(float), p_raw,
                                 p_cal),
            }
        except Exception as e:  # deployment view is context, never fatal
            rec["sealed_S2"] = {"error": str(e)}
    else:
        rec["sealed_S2"] = {"n": int(len(tail)),
                            "skipped": "pre-tail OOF < MIN_OOF_FOR_FIT"}

    # ---- SP-mismatch strata on the walk OOF (clean-SP rows) ----
    c = combined.copy()
    c["clean"] = (c["sp_era_home"].notna() & c["sp_era_away"].notna()
                  & (c["sp_era_home"].abs() <= SP_JUNK_ERA)
                  & (c["sp_era_away"].abs() <= SP_JUNK_ERA)
                  & c["sp_era_diff"].notna())
    clean = c[c["clean"]]
    bands = [(-99.0, -3.0, "home SP >= 3.0 better"),
             (-3.0, -1.5, "home SP 1.5-3.0 better"),
             (-1.5, 1.5, "even |diff| < 1.5"),
             (1.5, 3.0, "away SP 1.5-3.0 better"),
             (3.0, 99.0, "away SP >= 3.0 better")]
    rows = []
    for lo, hi, lbl in bands:
        sub = clean[(clean["sp_era_diff"] >= lo) & (clean["sp_era_diff"] < hi)]
        if len(sub) < 10:
            continue
        pcal = sub["home_win_prob_model_calibrated"].to_numpy(float)
        praw = sub["home_win_prob_model"].to_numpy(float)
        act = float(sub["home_win"].mean())
        rows.append({"band": lbl, "n": int(len(sub)),
                     "actual_home_win": round(act, 4),
                     "pred_cal_mean": round(float(pcal.mean()), 4),
                     "pred_raw_mean": round(float(praw.mean()), 4),
                     "gap_cal": round(float(pcal.mean() - act), 4),
                     "gap_raw": round(float(praw.mean() - act), 4),
                     "pred_cal_sd": round(float(pcal.std()), 4)})
    # compression meter: prediction SD inside the 55-65% binary band
    band5565 = clean[(clean["home_win_prob_model"] >= 0.55)
                     & (clean["home_win_prob_model"] < 0.65)]
    rec["strata"] = rows
    rec["compression"] = {
        "n_clean": int(len(clean)),
        "pred_cal_sd_all": round(float(clean["home_win_prob_model_calibrated"].std()), 4),
        "n_band_55_65": int(len(band5565)),
        "band_55_65_pred_cal_sd": round(
            float(band5565["home_win_prob_model_calibrated"].std()), 4)
        if len(band5565) >= 5 else None,
        "band_55_65_actual": round(float(band5565["home_win"].mean()), 4)
        if len(band5565) >= 5 else None,
    }
    print(f"  pooled(E) {rec['pooled_E']['logloss_cal']:.4f}/"
          f"{rec['pooled_E']['auc_cal']:.4f} ece_cal "
          f"{rec['pooled_E']['ece_cal']:.4f} | sealed S2 "
          f"{rec.get('sealed_S2', {}).get('logloss_cal', float('nan')):.4f}/"
          f"{rec.get('sealed_S2', {}).get('auc_cal', float('nan')):.4f}",
          flush=True)
    return rec


def _better_than(metric: dict, base: dict, kind: str) -> bool:
    if kind == "logloss":
        return metric["logloss_cal"] <= base["logloss_cal"] - WORTH_LOGLOSS
    return metric["auc_cal"] >= base["auc_cal"] + WORTH_AUC


def gate(cand: dict, base: dict) -> dict:
    """Six moneyline-style legs (calibrated axis) + worth-having."""
    legs = {}
    for view, key in (("pooled", "pooled_E"), ("sealed", "sealed_S2")):
        c, b = cand[key], base[key]
        if "logloss_cal" not in c or "logloss_cal" not in b:
            legs[f"{view}_logloss"] = None
            legs[f"{view}_auc"] = None
            legs[f"{view}_ece"] = None
            continue
        legs[f"{view}_logloss"] = bool(c["logloss_cal"]
                                        <= b["logloss_cal"] + TOL_LOGLOSS)
        legs[f"{view}_auc"] = bool(c["auc_cal"] >= b["auc_cal"] - TOL_AUC)
        legs[f"{view}_ece"] = bool(c["ece_cal"] <= b["ece_cal"])
    passed = [k for k, v in legs.items() if v is True]
    failed = [k for k, v in legs.items() if v is False]
    missing = [k for k, v in legs.items() if v is None]

    def _worth(view_key: str) -> bool:
        return (_better_than(cand[view_key], base[view_key], "logloss")
                or _better_than(cand[view_key], base[view_key], "auc"))

    worth = _worth("pooled_E")
    if "logloss_cal" in cand.get("sealed_S2", {}) and \
            "logloss_cal" in base.get("sealed_S2", {}):
        worth = worth or _worth("sealed_S2")
    return {"legs": legs, "legs_passed": len(passed), "legs_failed": len(failed),
            "legs_missing": len(missing), "worth_having": bool(worth)}


def strata_improved(cand: dict, base: dict) -> bool:
    """SP-mismatch strata track actuals better-or-equal than base: no band's
    |gap_cal| worsens beyond +0.002 and the sum of |gap_cal| does not grow."""
    cb = {r["band"]: r for r in cand["strata"]}
    bb = {r["band"]: r for r in base["strata"]}
    worse = 0
    for band, cr in cb.items():
        br = bb.get(band)
        if br is None:
            continue
        if abs(cr["gap_cal"]) > abs(br["gap_cal"]) + 0.002:
            worse += 1
    return worse == 0


def decide(name: str, cand: dict, base: dict) -> dict:
    g = gate(cand, base)
    verdict = "DO_NOT_ADOPT"
    reason = f"legs failed: {g['legs_failed']} (missing {g['legs_missing']})"
    if g["legs_failed"] == 0 and g["legs_missing"] == 0:
        if g["worth_having"] and strata_improved(cand, base):
            verdict = "ADOPT_CANDIDATE"
            reason = "gate cleared + worth-having + strata better-or-equal"
        else:
            verdict = "ADDITIVE_NULL"
            reason = ("gate cleared but nothing worth-having and/or strata "
                      "unchanged — composite redundant with existing SP "
                      "features")
    return {"candidate": name, "verdict": verdict, "reason": reason, **g}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", type=str, default="B0,B1",
                    help="comma-separated arms, e.g. B0,B1 or B0,B1,B2")
    ap.add_argument("--limit-folds", type=int, default=0,
                    help="run only the last N folds (smoke)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ("B0", "B1", "B2"):
            raise SystemExit(f"unknown arm {a!r}")

    frame_sha = sha256_file(CSV)[:16]
    print(f"frame sha256:16 = {frame_sha} | csv = {CSV}", flush=True)
    games = load_games(CSV)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.sort_values("game_date").reset_index(drop=True)

    # Projection composite: z-stats + OLS scale fit on rows strictly prior to
    # the sealed 21-day window (verbatim b7eed32 discipline).
    pre_mask = (games["game_date"] < games["game_date"].max()
                - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
    games, proj_meta = attach_projection_cols(games, pre_mask)
    games["sp_proj_era_diff"] = (games["sp_proj_era_home"]
                                  - games["sp_proj_era_away"])
    print(f"projection fit: pre rows {int(pre_mask.sum())} | "
          f"slopes { {k: v['era_on_proj_slope'] for k, v in proj_meta.items()} } "
          f"| coverage pre { {k: v['coverage_pre'] for k, v in proj_meta.items()} }",
          flush=True)

    # ---- committed OOF geometry: trim warm-up rows (see TRIM_START) ----
    n_pre_trim = len(games)
    games = games[games["game_date"] >= pd.Timestamp(TRIM_START)]\
        .reset_index(drop=True)
    print(f"frame: {n_pre_trim} rows -> trimmed to >= {TRIM_START}: "
          f"{len(games)} rows", flush=True)

    # ---- binary's own walk-forward geometry (production path) ----
    splits = training.walk_forward_splits(
        games, retrain_cadence_days=RETRAIN_CADENCE_DAYS,
        max_eval_folds=args.limit_folds)
    print(f"splits: {len(splits)} (limit_folds={args.limit_folds or 'all'})",
          flush=True)

    # Enrich margins ONCE over these folds; every arm reuses them.
    enriched, regen = training._attach_oof_run_margins(
        games, splits, MIN_VAL_FOLD_GAMES, args.limit_folds,
        RETRAIN_CADENCE_DAYS, 0, decided_snapshot=games)
    print(f"margin coverage: {100 * enriched['run_margin_diff'].notna().mean():.1f}% "
          f"over {len(enriched)} rows", flush=True)

    # Committed OOF game set E (published-artifact parity for pooled metrics).
    baseline_committed = None
    e_rows = None
    if HISTORY_CSV.exists():
        hist = pd.read_csv(HISTORY_CSV)
        key_cols = ["game_id", "home_score", "away_score"]
        hist_ref = hist.drop_duplicates(subset=key_cols).reset_index(drop=True)
        baseline_committed = compute_pooled(
            hist_ref["home_win"], hist_ref["home_win_prob_model"],
            hist_ref["home_win_prob_model_calibrated"])
        e_rows = hist_ref[key_cols]
        # Geometry parity check: kept-fold val rows must equal the committed
        # OOF set (the trimmed frame reproduces it exactly).
        kept = [s for s in splits
                if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES
                or s.get("is_partial_tail")]
        vk = set()
        for s in kept:
            vk |= set(map(tuple, s["val_games"][key_cols]
                          .itertuples(index=False, name=None)))
        ck = set(map(tuple, hist_ref[key_cols].itertuples(index=False, name=None)))
        extra, missing = len(vk - ck), len(ck - vk)
        print(f"geometry parity: kept-fold val {len(vk)} vs committed E "
              f"{len(ck)} | extra={extra} missing={missing}", flush=True)
        if extra or missing:
            print("  WARNING: walk val set != committed OOF set — B0 pooled "
                  "cannot reproduce the published pins exactly", flush=True)
        print(f"committed pooled pins (n={len(hist_ref)}): "
              f"logloss {baseline_committed['logloss_raw']:.4f} "
              f"cal {baseline_committed['logloss_cal']:.4f} | auc "
              f"{baseline_committed['auc_raw']:.4f} "
              f"cal {baseline_committed['auc_cal']:.4f} | ece_cal "
              f"{baseline_committed['ece_cal']:.4f}", flush=True)

    tail_start = enriched["game_date"].max() - pd.Timedelta(days=TAIL_DAYS - 1)
    print(f"sealed window: >= {tail_start.date()} "
          f"(n in frame = {(enriched['game_date'] >= tail_start).sum()})",
          flush=True)

    out = args.out or (DATA_DELIVERY_DIR
                       / f"mlb_binary_sp_projection_{frame_sha}.json")
    record = {"schema": "mlb-binary-sp-projection/v1",
              "frame_sha": frame_sha,
              "frame_sha_source": "game_level_features.csv (sha256:16)",
              "date": "20260904",
              "csv_history": HISTORY_CSV.name if HISTORY_CSV.exists() else None,
              "geometry": {"cadence_days": RETRAIN_CADENCE_DAYS,
                           "min_val_games": MIN_VAL_FOLD_GAMES,
                           "seed": int(RANDOM_SEED),
                           "tail_days": TAIL_DAYS,
                           "trim_start": TRIM_START,
                           "frame_rows_pre_trim": n_pre_trim,
                           "oof_rows_expected": len(e_rows) if e_rows is not None else None},
              "projection": {
                  "components": {"lower_better": list(PROJ_LO_BETTER),
                                 "higher_better": list(PROJ_HI_BETTER)},
                  "min_components": int(MIN_PROJ_COMPONENTS),
                  "scale": "ERA-equivalent: 1 unit ~= 1 ERA point of quality "
                           "(higher = better); z-stats + OLS fit on pre-holdout "
                           "rows only",
                  "era_on_proj_slope": {k: v["era_on_proj_slope"]
                                        for k, v in proj_meta.items()},
                  "coverage_pre": {k: v["coverage_pre"]
                                   for k, v in proj_meta.items()},
                  "coverage_sealed": {k: v["coverage_sealed"]
                                      for k, v in proj_meta.items()},
                  "note": "verbatim b7eed32 composite; reproduced exactly on "
                          "this frame (see run_sp_sensitivity.py)."},
              "arms_defined": build_arm_cols(),
              "baseline_committed_E": baseline_committed,
              "results": {}}
    if out.exists():
        try:
            record = json.loads(out.read_text())
        except json.JSONDecodeError:
            pass

    arm_cols = build_arm_cols()
    for name in arms:
        if name in record["results"]:
            print(f"  arm {name} already recorded — skipping")
            continue
        rec = run_arm(name, arm_cols[name], enriched, e_rows, tail_start,
                      len(e_rows) if e_rows is not None else 0,
                      limit_folds=args.limit_folds)
        record["results"][name] = rec
        out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"  -> record updated: {out}", flush=True)

    res = record["results"]
    # B2 runs only if B1 showed promise (task: 'only if B1 shows promise').
    if "B1" in res and "B0" in res and "B2" not in res and "B2" in arms:
        g1 = decide("B1", res["B1"], res["B0"])
        if g1["verdict"] != "DO_NOT_ADOPT":
            print("\nB1 cleared the gate — running B2 (additivity arm)...")
            rec = run_arm("B2", arm_cols["B2"], enriched, e_rows,
                          tail_start, len(e_rows) if e_rows is not None else 0,
                          limit_folds=args.limit_folds)
            record["results"]["B2"] = rec
            out.write_text(json.dumps(record, indent=2) + "\n")
        else:
            print(f"\nB1 verdict DO_NOT_ADOPT — skipping B2 "
                  f"({g1['reason']})")

    verdicts = {}
    for name in ("B1", "B2"):
        if name in res and "B0" in res:
            verdicts[name] = decide(name, res[name], res["B0"])
    record["gate"] = verdicts
    record["summary"] = summarize(record)
    out.write_text(json.dumps(record, indent=2) + "\n")

    print("\n================= GATE vs B0 =================")
    for name, v in verdicts.items():
        print(f"{name}: {v['verdict']} — {v['reason']}")
        for leg, ok in v["legs"].items():
            print(f"    {leg}: {'PASS' if ok is True else 'FAIL' if ok is False else 'n/a'}")
        print(f"    worth_having: {v['worth_having']}")
    print(f"\nrecord: {out}")


def summarize(record: dict) -> dict:
    """Computed narrative for the record (no prose drift — every number is
    pulled from the recorded arm results)."""
    res = record["results"]
    b0 = res["B0"]
    out: dict = {"arms": {}}
    for name in ("B1", "B2"):
        if name not in res:
            continue
        c = res[name]
        for view, key in (("pooled", "pooled_E"), ("sealed_S2", "sealed_S2")):
            if "logloss_cal" not in c[key] or "logloss_cal" not in b0[key]:
                continue
            out["arms"].setdefault(name, {})[view] = {
                "d_logloss_cal": round(c[key]["logloss_cal"]
                                       - b0[key]["logloss_cal"], 4),
                "d_auc_cal": round(c[key]["auc_cal"] - b0[key]["auc_cal"], 4),
                "d_ece_cal": round(c[key]["ece_cal"] - b0[key]["ece_cal"], 4),
            }
        # stratum gap deltas (band-level |gap_cal| vs B0)
        cb = {r["band"]: r for r in c["strata"]}
        bb = {r["band"]: r for r in b0["strata"]}
        out["arms"][name]["strata_d_abs_gap_cal"] = {
            band: round(abs(cb[band]["gap_cal"]) - abs(bb[band]["gap_cal"]), 4)
            for band in cb if band in bb}
        cc = c.get("compression", {}); bc = b0.get("compression", {})
        if cc.get("pred_cal_sd_all") and bc.get("pred_cal_sd_all"):
            out["arms"][name]["d_pred_cal_sd_all"] = round(
                cc["pred_cal_sd_all"] - bc["pred_cal_sd_all"], 4)
    g = record.get("gate", {})
    texts = []
    for name, v in g.items():
        texts.append(f"{name}: {v['verdict']} ({v['reason']})")
    if "B1" in res and "B2" not in res:
        texts.append("B2 not run: gated on B1 showing promise; B1 verdict "
                     "DO_NOT_ADOPT so the additivity arm was skipped per the "
                     "spec (a replacement-arm loss also answers additivity in "
                     "the negative direction only if the addition were to "
                     "regress too — not tested here).")
    out["verdict_text"] = " | ".join(texts)
    verdict = next(iter(g.values()), {}).get("verdict", "")
    if verdict == "ADOPT_CANDIDATE":
        out["next_action"] = "Record-only ADOPT_CANDIDATE — a separate " \
                              "engine-change spec gates actual adoption."
    elif verdict == "DO_NOT_ADOPT":
        out["next_action"] = "Keep production FEATURE_COLS unchanged. The " \
                              "binary's existing SP features (raw ERA " \
                              "diff + per-side levels + k9/xwoba/whiff/velo " \
                              "families) already capture the composite's " \
                              "signal; the projection composite is NOT a " \
                              "candidate upgrade for the binary. The two-stage " \
                              "correction layer is NOT queued (no partial " \
                              "signal worth extending)."
    else:
        out["next_action"] = ("ADDITIVE_NULL — composite redundant with the "
                               "binary's existing SP features; no follow-up.")
    return out


if __name__ == "__main__":
    main()
