"""Run-margin ablation — WITH vs WITHOUT ``run_margin_diff`` on the moneyline.

Measurement for the 2026-08 margin feature task (NOT tuning, NOT wiring):
does ONE column — λ_home − λ_away from the run engine's per-side Poisson
models, computed OUT-OF-FOLD on THE MONEYLINE'S OWN fold split — help the
ensemble? Nothing active changes until the sealed-holdout gate says ship;
FEATURE_COLS stays 64 throughout this harness.

Design (mirrors run_lineup_ablation.py exactly, plus the margin join):
- Data: committed data_delivery/game_level_features.csv + production
  add_lineup_delta_features() (the shipped 64-col matrix).
- Folds: walk_forward_splits on the tuning pool, filtered by
  MIN_VAL_FOLD_GAMES — generated ONCE and shared by the margin builder and
  every variant arm (apples-to-apples).
- Margin: build_oof_margin.oof_run_margins() trains the run engine READ-ONLY
  per fold on that fold's TRAIN games only; each game's margin comes from a
  model trained strictly before it (fold-boundary asserted). Holdout margins:
  fit-only refit on ALL tuning games at the median fold round count
  (production slate convention) — strictly future, cannot leak.
- Join: margins merged by game_pk; games outside any executed fold get NaN →
  the moneyline's existing imputation path (trees route NaN; logistic/MLP
  train-median). Coverage reported loudly, never papered over.
- Variants: WITHOUT = production 64 cols; WITH = 64 + run_margin_diff;
  LAMBDAS = WITH + lam_home + lam_away (REPORT-ONLY context per the task —
  the primary decision is margin-only).
- Members: all 5 + static-prior blend (adaptive weights cleared per variant);
  prequential Platt calibration on prior folds' pairs only; clip 1e-7.
- Sealed 21-day holdout refit AFTER the fold loop, once per variant. Its
  calibration leg (ECE-cal) applies the Platt scaling fit on ALL pooled OOF
  pairs — strictly earlier data, no holdout information.

Gate (task rule): SHIP only if the blend improves the SEALED holdout on
logloss OR AUC without degrading calibration (ECE-cal of the calibrated
blend ≤ WITHOUT's). Pooled wins alone never adopt (the MLP-tune lesson).

Emits data_delivery/margin_ablation_<sha>.json (resumable: completed variants
are cached; in-flight variants checkpoint per fold into <out>.partial.json so
an interrupted run resumes without recomputing finished folds). COMMITS
NOTHING.

Usage:
    python run_margin_ablation.py
    python run_margin_ablation.py --variants WITHOUT,WITH,LAMBDAS
    python run_margin_ablation.py --smoke   # 3 folds -> /tmp, gate skipped
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

import training  # noqa: E402
from build_oof_margin import (  # noqa: E402
    LAM_COLS,
    MARGIN_COL,
    oof_run_margins,
    refit_run_margins,
)
from config import (  # noqa: E402
    DATA_DELIVERY_DIR,
    MIN_VAL_FOLD_GAMES,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
)
from features import add_lineup_delta_features  # noqa: E402
from calibration import apply_platt, fit_platt, MIN_OOF_FOR_FIT  # noqa: E402

EPS = 1e-7


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def head_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=_BACKEND_DIR.parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


def build_variants(margin_cols: list[str]) -> dict[str, list[str]]:
    """WITHOUT = production 64; WITH = 64 + margin; LAMBDAS = WITH + λ levels."""
    base = [c for c in training.FEATURE_COLS if c not in margin_cols]
    assert len(base) == 64, f"expected 64 production columns, got {len(base)}"
    return {
        "WITHOUT": list(base),
        "WITH": list(base) + [MARGIN_COL],
        "LAMBDAS": list(base) + [MARGIN_COL] + list(LAM_COLS),
    }


def redundancy_report(games: pd.DataFrame) -> dict:
    """|Spearman ρ| of the margin vs the strongest existing matchup columns —
    the 'is this just elo_diff again?' read the task asked for."""
    out: dict[str, float] = {}
    if MARGIN_COL not in games.columns:
        return out
    m = games[MARGIN_COL]
    for c in ("elo_diff", "win_pct_diff", "home_run_diff", "sp_era_diff"):
        if c in games.columns:
            out[c] = round(float(m.corr(games[c], method="spearman")), 4)
    return out


def run_variant(cols: list[str], folds, tune_df, hold_df,
                partial_path: Path | None = None) -> dict:
    training.FEATURE_COLS = list(cols)
    training._LAST_ADAPTIVE_WEIGHTS.clear()  # both variants blend identically

    # ── resume support: reload completed folds from the partial checkpoint ──
    done: dict[int, dict] = {}
    if partial_path and partial_path.exists():
        try:
            for rec in json.loads(partial_path.read_text()).get("folds", []):
                done[rec["fold_idx"]] = rec
        except Exception:
            done = {}

    def _flush(state: dict) -> None:
        if partial_path:
            partial_path.write_text(json.dumps(state))

    state: dict = {"cols_n": len(cols), "folds": [done[i] for i in sorted(done)]}
    oof_y: list[float] = []
    oof_blend: list[float] = []
    oof_members: dict[str, list[float]] = {}
    oof_blend_cal: list[float] = []
    oof_members_cal: dict[str, list[float]] = {}

    def _accumulate(rec: dict) -> None:
        oof_y.extend(rec["y"])
        oof_blend.extend(rec["blend"])
        oof_blend_cal.extend(rec["blend_cal"])
        for name, plist in rec["members"].items():
            oof_members.setdefault(name, []).extend(plist)
        for name, plist in rec["members_cal"].items():
            oof_members_cal.setdefault(name, []).extend(plist)

    for rec in state["folds"]:
        _accumulate(rec)

    executed = len(done)
    for split in folds:
        fi = split["fold_idx"]
        if fi in done:
            continue
        train, val = split["train_games"], split["val_games"]
        try:
            models, _ = training.train_moneyline_ensemble(train, val)
        except Exception as e:  # keep the loop honest: log, skip, continue
            print(f"  fold {fi} failed: {e}", flush=True)
            continue
        blend, member_probs, _wts = training.ensemble_predict(models, val)
        y_val = val["home_win"].values.astype(float)
        fold_cal = None
        if len(oof_y) >= MIN_OOF_FOR_FIT:
            fold_cal = fit_platt(np.asarray(oof_y), np.asarray(oof_blend))
        pa = {n: np.asarray(p, dtype=float) for n, p in member_probs.items()}
        rec = {
            "fold_idx": int(fi),
            "y": y_val.tolist(),
            "blend": np.asarray(blend, dtype=float).tolist(),
            "blend_cal": np.asarray(
                apply_platt(np.asarray(blend, dtype=float), fold_cal),
                dtype=float).tolist(),
            "members": {n: p.tolist() for n, p in pa.items()},
            "members_cal": {
                n: np.asarray(apply_platt(p, fold_cal), dtype=float).tolist()
                for n, p in pa.items()},
        }
        _accumulate(rec)
        done[fi] = rec
        state["folds"] = [done[i] for i in sorted(done)]
        _flush(state)
        executed += 1
        print(f"  fold {fi}: n_val={len(y_val)} "
              f"blend_ll={training.compute_metrics(np.array(oof_y), np.array(oof_blend))['logloss']:.4f}",
              flush=True)

    y_all = np.asarray(oof_y, dtype=float)
    pooled: dict[str, dict] = {}
    pooled["blend"] = training.compute_metrics(
        y_all, np.asarray(oof_blend, dtype=float))
    pooled["blend_calibrated"] = training.compute_metrics(
        y_all, np.asarray(oof_blend_cal, dtype=float))
    for name, plist in oof_members.items():
        pooled[name] = training.compute_metrics(
            y_all, np.asarray(plist, dtype=float))
        pooled[f"{name}_calibrated"] = training.compute_metrics(
            y_all, np.asarray(oof_members_cal.get(name, []), dtype=float))

    # ── sealed holdout: fit only at the end ───────────────────────────────
    models, _ = training.train_moneyline_ensemble(tune_df)
    blend_hold, member_hold, _wts = training.ensemble_predict(models, hold_df)
    y_hold = hold_df["home_win"].values.astype(float)
    # Calibration leg: Platt fit on the FULL pooled OOF (all strictly-
    # pre-holdout pairs), applied to the holdout blend — the same "refit on
    # all prior data" convention the sealed-holdout models themselves use.
    full_cal = fit_platt(y_all, np.asarray(oof_blend, dtype=float))
    holdout: dict[str, dict] = {
        "blend": training.compute_metrics(y_hold, np.asarray(blend_hold)),
        "blend_calibrated": training.compute_metrics(
            y_hold, np.asarray(apply_platt(
                np.asarray(blend_hold, dtype=float), full_cal))),
    }
    for name, p in member_hold.items():
        holdout[name] = training.compute_metrics(
            y_hold, np.asarray(p, dtype=float))

    if partial_path and partial_path.exists():
        partial_path.unlink()
    return {"n_cols": len(cols), "folds_executed": executed,
            "pooled": pooled, "holdout": holdout}


def prepare_data(holdout_days: int, limit_folds: int = 0):
    """Load + enrich the committed snapshot, split holdout, generate the
    fold GEOMETRY once, build the leakage-free margin table on it (run
    engine READ-ONLY), then REGENERATE the same folds over the ENRICHED
    frame so every variant arm actually carries the margin column.

    Fold geometry is a pure function of game_date/home_win, so the
    regenerated splits are row-for-row identical to the ones the margin was
    built on — asserted below so a future walk_forward_splits change cannot
    silently desync the two."""
    import hashlib

    data_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    games = pd.read_csv(data_path)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_win"]).reset_index(drop=True)
    games = add_lineup_delta_features(games)

    cutoff = games["game_date"].max() - pd.Timedelta(days=holdout_days - 1)
    tune_df = games[games["game_date"] < cutoff].reset_index(drop=True)
    hold_df = games[games["game_date"] >= cutoff].reset_index(drop=True)
    all_splits = training.walk_forward_splits(
        tune_df, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
    folds = [s for s in all_splits if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]
    if limit_folds:
        folds = folds[:limit_folds]

    # Deterministic margin cache key: data hash + fold geometry.
    h = hashlib.sha256()
    h.update(sha256_file(data_path).encode())
    h.update(json.dumps([str(s["val_start"]) for s in folds]).encode())
    cache = (Path(tempfile.gettempdir())
             / f"margin_oof_cache_{h.hexdigest()[:16]}.parquet")

    if cache.exists():
        margins = pd.read_parquet(cache)
        meta = json.loads(cache.with_suffix(".meta.json").read_text())
        uncov = int(meta.pop("n_uncovered"))
        rounds = {k: int(v) for k, v in meta.items()}
    else:
        print("building OOF margins on the moneyline folds "
              "(run engine READ-ONLY) ...", flush=True)
        margins, _rounds_raw, uncov = oof_run_margins(tune_df, folds)
        rounds = {k: int(v) for k, v in _rounds_raw.items()}
        margins.to_parquet(cache)
        meta = {**{k: str(v) for k, v in rounds.items()},
                "n_uncovered": int(uncov)}
        cache.with_suffix(".meta.json").write_text(json.dumps(meta))

    hold_margins = refit_run_margins(tune_df, hold_df, rounds)

    # Regenerate the SAME fold geometry over the ENRICHED tuning frame so
    # train/val frames carry the joined margin columns.
    tune_enriched = attach(tune_df, margins)
    enriched_splits = [
        s for s in training.walk_forward_splits(
            tune_enriched, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
        if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]
    if limit_folds:
        enriched_splits = enriched_splits[:limit_folds]
    assert len(enriched_splits) == len(folds) and all(
        a["fold_idx"] == b["fold_idx"]
        and pd.Timestamp(a["val_start"]) == pd.Timestamp(b["val_start"])
        and a["val_games"]["game_pk"].tolist()
        == b["val_games"]["game_pk"].tolist()
        for a, b in zip(folds, enriched_splits)), \
        "enriched-frame folds desynced from margin-build folds"
    return (games, tune_enriched, hold_df, enriched_splits, margins,
            hold_margins, rounds, uncov)


def attach(df: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    """Left-join the margin columns by game_pk; uncovered rows stay NaN."""
    df = df.drop(columns=[c for c in [MARGIN_COL, *LAM_COLS]
                          if c in df.columns]).copy()
    return df.merge(margins[["game_pk", MARGIN_COL, *LAM_COLS]],
                    on="game_pk", how="left")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", type=str, default="WITHOUT,WITH")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--holdout-days", type=int, default=21)
    ap.add_argument("--limit-folds", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="3 folds, output to /tmp, gate skipped")
    ap.add_argument("--target-date", type=str, default=None,
                    help="Pipeline target date (YYYY-MM-DD); defaults to today")
    args = ap.parse_args()
    if args.smoke:
        args.limit_folds = min(args.limit_folds or 3, 3)
        args.out = Path("/tmp/margin_ablation_smoke.json")
        args.variants = "WITHOUT,WITH"

    sha = head_sha()
    (games, tune_enriched, hold_df, folds, _margins, hold_margins,
     rounds, uncov) = prepare_data(args.holdout_days, args.limit_folds)

    hold_enriched = attach(hold_df, hold_margins)
    covered = float(tune_enriched[MARGIN_COL].notna().mean())
    print(f"commit={sha[:12]} games={len(games)} tuning={len(tune_enriched)} "
          f"holdout={len(hold_df)} folds={len(folds)} seed={RANDOM_SEED} "
          f"clip={EPS}")
    print(f"margin coverage: tuning {covered:.1%} | uncovered_tune_games="
          f"{uncov} (NaN → existing imputation path)")
    print(f"median fold rounds: home={rounds.get('home')} "
          f"away={rounds.get('away')}")
    red = redundancy_report(tune_enriched)
    if red:
        print(f"redundancy |ρ|(margin, ·): {red}")

    variants = build_variants([MARGIN_COL] + list(LAM_COLS))
    target = args.target_date or pd.Timestamp.now().date().isoformat()
    compact_target = target.replace("-", "")
    out = args.out or (DATA_DELIVERY_DIR /
                       f"margin_ablation_{compact_target}.json")
    if out.exists():
        results = json.loads(out.read_text())
    else:
        results = {"schema": "margin-ablation/v1", "commit_sha": sha,
                   "data_sha256": sha256_file(
                       DATA_DELIVERY_DIR / "game_level_features.csv"),
                   "holdout_days": args.holdout_days,
                   "folds_executed": len(folds),
                   "margin_col": MARGIN_COL,
                   "margin_source": "run_engine per-side Poisson, OOF on the "
                                    "moneyline's own folds (read-only)",
                   "margin_coverage_tuning": round(covered, 4),
                   "uncovered_tune_games": int(uncov),
                   "median_fit_rounds": {k: int(v) for k, v in rounds.items()
                                         if k != "n_uncovered"},
                   "redundancy_spearman": red,
                   "clip_eps": EPS, "seed": int(RANDOM_SEED),
                   "variants": {}}

    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    for name in want:
        if name in results["variants"]:
            print(f"  {name}: cached, skipping")
            continue
        print(f"  {name}: running ({len(variants[name])} cols) ...", flush=True)
        r = run_variant(variants[name], folds, tune_enriched, hold_enriched,
                        partial_path=Path(str(out) + ".partial.json"))
        r["cols"] = variants[name]
        results["variants"][name] = r
        out.write_text(json.dumps(results, indent=2) + "\n")
        b, h = r["pooled"]["blend"], r["holdout"]["blend"]
        print(f"    pooled blend {b['logloss']:.4f}/{b['auc']:.4f} "
              f"brier {b['brier']:.4f} ece {b['ece']:.4f} | "
              f"holdout {h['logloss']:.4f}/{h['auc']:.4f} ece {h['ece']:.4f}",
              flush=True)

    # gate: sealed holdout — improve logloss OR AUC without ECE degradation
    if (not args.smoke and "WITHOUT" in results["variants"]
            and "WITH" in results["variants"]):
        wo = results["variants"]["WITHOUT"]["holdout"]
        wi = results["variants"]["WITH"]["holdout"]
        wo_c, wi_c = wo.get("blend_calibrated", wo["blend"]), \
            wi.get("blend_calibrated", wi["blend"])
        b_wo, b_wi = wo["blend"], wi["blend"]
        improves = ((b_wi["logloss"] < b_wo["logloss"])
                    or (b_wi["auc"] > b_wo["auc"]))
        cal_ok = wi_c["ece"] <= wo_c["ece"]
        results["gate"] = {
            "verdict": "SHIP" if (improves and cal_ok) else "DON'T SHIP",
            "rule": "sealed-holdout blend improves logloss OR AUC without "
                    "ECE-cal (Platt-on-prior-OOF) degradation; pooled wins "
                    "alone never adopt",
            "holdout_without": b_wo,
            "holdout_with": b_wi,
            "holdout_without_cal": wo_c,
            "holdout_with_cal": wi_c,
        }
        out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"GATE: holdout WITHOUT {b_wo['logloss']:.4f}/"
              f"{b_wo['auc']:.4f} ece-cal {wo_c['ece']:.4f} vs WITH "
              f"{b_wi['logloss']:.4f}/{b_wi['auc']:.4f} ece-cal "
              f"{wi_c['ece']:.4f} → {results['gate']['verdict']}")


if __name__ == "__main__":
    main()
