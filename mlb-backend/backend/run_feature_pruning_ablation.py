"""Feature-pruning ablation — backward-elimination on the sealed window.

Goal: determine whether a LEANER moneyline feature set (out of the current
65) improves or preserves sealed-window logloss/AUC/ECE. This is
SUBTRACTION (drop features), not addition — no new columns, no model math,
no calibration changes.

Arms:
  CURRENT  = production 65-col FEATURE_COLS (baseline)
  INTERACTIONS = drop the 7 hand-crafted interaction features
  LOW_COVERAGE = drop the 6 sub-90%-coverage lineup_actual_*/rest_count_*
  CORRELATED   = drop the correlated-variant features (verified |r|>0.8 pairs)
  ALL      = INTERACTIONS + LOW_COVERAGE + CORRELATED
run_margin_diff is OUT OF SCOPE — it is retained in every arm (its
decoupling question is a separate task). is_home and dome_is_neutral stay
(tiny anchors).

Design (mirrors run_margin_ablation.py / run_home_edge_interaction_ablation.py):
- Data: committed data_delivery/game_level_features.csv + production
  add_lineup_delta_features() (the shipped 65-col matrix).
- Folds: walk_forward_splits on the tuning pool, filtered by
  MIN_VAL_FOLD_GAMES — generated ONCE and shared by every arm
  (apples-to-apples). Fold geometry is a pure function of game_date/home_win.
- Members: all 5 + static-prior blend per arm (adaptive weights re-earned
  per arm via training._LAST_ADAPTIVE_WEIGHTS.clear()); prequential Platt on
  prior folds' pairs only; clip 1e-7. Member OOF probabilities accumulate so
  every arm's pooled + sealed numbers are apples-to-apples.
- Sealed 21-day holdout refit AFTER the fold loop, once per surviving arm.
  Its calibration leg (ECE-cal) applies the Platt scaling fit on ALL pooled
  OOF pairs — strictly earlier data, no holdout information.

STEP 3 (cheap pre-check): score every prune arm on pooled OOF first. If NO
prune arm improves pooled logloss AND AUC vs CURRENT, record that and SKIP
the sealed run (same rule as the home-edge interaction ablation).

STEP 4 (full gate, only if a prune arm clears STEP 3): run the sealed-284
evaluation for the surviving arms. ADOPT only if sealed logloss AND AUC
improve with ECE not degraded, AND pooled does not regress (a pooled-gain /
sealed-loss inversion = DON'T ADOPT, as with every prior blend gate).

Emits data_delivery/feature_pruning_ablation_<date>.json (resumable: completed
arms are cached; in-flight arms checkpoint per fold into <out>.partial.json).
COMMITS NOTHING — if ADOPT, the production change is the FEATURE_COLS edit +
normal retrain cycle (never hand-edited weights).

Usage:
    python3 run_feature_pruning_ablation.py
    python3 run_feature_pruning_ablation.py --arms CURRENT,ALL
    python3 run_feature_pruning_ablation.py --smoke   # 3 folds -> /tmp, gates skipped
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

import training  # noqa: E402
from config import (  # noqa: E402
    DATA_DELIVERY_DIR,
    MIN_VAL_FOLD_GAMES,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
)
from features import add_lineup_delta_features  # noqa: E402
from calibration import apply_platt, fit_platt, MIN_OOF_FOR_FIT  # noqa: E402
from build_oof_margin import (  # noqa: E402
    MARGIN_COL,
    oof_run_margins,
    refit_run_margins,
)

EPS = 1e-7

# rr========================================================================
# STEP 1 — feature classification + candidate prune sets (verified numbers)
# Proposed removal lists are hypotheses; the correlation/coverage matrix
# below is computed from the pre-sealed decided frame at run start and stored
# in the record so the classification is auditable.
# ---------------------------------------------------------------------------

# (a) hand-crafted interaction features
PRUNE_INTERACTIONS = [
    "bullpen_meltdown_risk",
    "pitcher_regression_indicator",
    "lineup_depth_multiplier",
    "ace_efficiency_factor",
    "wind_advantage_flyball_factor",
    "air_density_velocity_boost",
    "park_factor_slug_diff",
]

# (c) low-coverage (<90% on the decided frame) — verified 62-64% measured
PRUNE_LOW_COVERAGE = [
    "lineup_actual_woba_delta_home",
    "lineup_actual_woba_delta_away",
    "lineup_actual_top3_delta_home",
    "lineup_actual_top3_delta_away",
    "lineup_rest_count_home",
    "lineup_rest_count_away",
]

# (b) correlated variants of one signal — verified |r|>0.8 pairs on the
# decided frame: win_pct_diff~elo_diff (0.881) and team_hardhit_diff~
# team_exitvelo_diff (0.819). Drop the lower-weight member of each pair
# (win_pct_diff 2.02 vs elo_diff 3.53; team_exitvelo_diff 1.42 vs
# team_hardhit_diff 1.93).
PRUNE_CORRELATED = [
    "win_pct_diff",
    "team_exitvelo_diff",
]

# Combined: interactions + low-coverage + correlated (disjoint by construction)
PRUNE_ALL = sorted(set(PRUNE_INTERACTIONS) | set(PRUNE_LOW_COVERAGE)
                   | set(PRUNE_CORRELATED))


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


def build_arms(prune_sets: dict[str, list[str]]) -> dict[str, list[str]]:
    """CURRENT = full 65; each prune arm = CURRENT minus its prune set.

    run_margin_diff, is_home, dome_is_neutral are never pruned.
    """
    base = list(training.FEATURE_COLS)
    assert len(base) == 65, f"expected 65 production columns, got {len(base)}"
    arms = {"CURRENT": base}
    for name, drop in prune_sets.items():
        # defensive: never drop the anchors / out-of-scope features
        protected = {"run_margin_diff", "is_home", "dome_is_neutral"}
        clean = [c for c in drop if c not in protected]
        arms[name] = [c for c in base if c not in clean]
    return arms


# ---------------------------------------------------------------------------
# STEP 1 preprocessing: coverage + correlation matrix on the pre-sealed frame
# ---------------------------------------------------------------------------
def redundancy_report(decided: pd.DataFrame, cols: list[str]) -> dict:
    """Coverage per feature + top |r|>0.8 correlated pairs on the decided frame."""
    mf = decided[cols].apply(pd.to_numeric, errors="coerce")
    coverage = {c: round(float(mf[c].notna().mean()), 4) for c in cols}
    imp = mf.fillna(mf.median())
    corr = imp.corr()
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = float(corr.loc[a, b])
            if abs(r) > 0.8:
                pairs.append([a, b, round(float(r), 4)])
    pairs.sort(key=lambda x: -abs(x[2]))
    return {"coverage": coverage,
            "feature_n": len(cols),
            "corr_gt_0_8_pairs": pairs}


def monitor_importance_report(corr_pairs: list[list],
                              monitor_path: Path | None = None) -> dict:
    """Per-feature blend weights + weight-approx-zero flags from the SHIPPED
    model monitor (STEP 1 read — feature importances/weights; nothing fitted
    here, strictly a read of the committed artifact).

    ``weight_pct`` is the adaptive blend-weight share each feature earned in
    the production walk-forward (the monitor's feature_drift table). Flags
    features whose weight is approx zero (<0.5%) AND that sit in an |r|>0.8
    correlation pair with a higher-weighted sibling — the "dead weight
    carried by a correlated twin" signal the pruning task asks to surface.
    """
    if monitor_path is None:
        cands = sorted(DATA_DELIVERY_DIR.glob("model_monitor_*.json"))
        monitor_path = cands[-1] if cands else None
    if monitor_path is None:
        return {"source": None, "feature_n": 0, "weights_pct": {},
                "weight_zero_in_corr_pair": []}
    with open(monitor_path) as f:
        mon = json.load(f)
    rows = mon.get("feature_drift", []) or []
    weights = {}
    for r in rows:
        w = r.get("weight_pct")
        if w is not None:
            weights[str(r.get("feature"))] = float(w)

    pair_features = {f for pair in corr_pairs for f in pair[:2]}
    zero_in_pair = []
    for feat in sorted(pair_features):
        w = weights.get(feat, 0.0) or 0.0
        siblings = [b if a == feat else a for a, b, _ in corr_pairs
                    if feat in (a, b)]
        if w < 0.5 and any((weights.get(s, 0.0) or 0.0) > w
                           for s in siblings):
            zero_in_pair.append(feat)

    return {
        "source": monitor_path.name,
        "feature_n": len(weights),
        "weights_pct": {f: round(float(w), 4)
                         for f, w in sorted(weights.items())},
        "weight_zero_in_corr_pair": zero_in_pair,
        "note": "weight_pct = adaptive blend-weight share from the shipped "
                "model_monitor feature_drift; weight≈0 = <0.5% AND in an "
                "|r|>0.8 pair with a higher-weighted sibling",
    }


# ---------------------------------------------------------------------------
# Shared single (per-arm) fold loop + sealed refit (mirrors run_margin_ablation)
# ---------------------------------------------------------------------------
def run_arm(cols: list[str], folds, tune_df, hold_df,
            partial_path: Path | None = None) -> dict:
    training.FEATURE_COLS = list(cols)
    training._LAST_ADAPTIVE_WEIGHTS.clear()  # weights re-earned per arm

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
    oof_blend_cal: list[float] = []

    def _accumulate(rec: dict) -> None:
        oof_y.extend(rec["y"])
        oof_blend.extend(rec["blend"])
        oof_blend_cal.extend(rec["blend_cal"])

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
        blend, _member_probs, _wts = training.ensemble_predict(models, val)
        y_val = val["home_win"].values.astype(float)
        fold_cal = None
        if len(oof_y) >= MIN_OOF_FOR_FIT:
            fold_cal = fit_platt(np.asarray(oof_y), np.asarray(oof_blend))
        rec = {
            "fold_idx": int(fi),
            "y": y_val.tolist(),
            "blend": np.asarray(blend, dtype=float).tolist(),
            "blend_cal": np.asarray(
                apply_platt(np.asarray(blend, dtype=float), fold_cal),
                dtype=float).tolist(),
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
    pooled = {
        "blend": training.compute_metrics(y_all, np.asarray(oof_blend, dtype=float)),
        "blend_calibrated": training.compute_metrics(
            y_all, np.asarray(oof_blend_cal, dtype=float)),
    }

    # ── sealed holdout: fit only at the end ───────────────────────────────
    models, _ = training.train_moneyline_ensemble(tune_df)
    blend_hold, _member_hold, _wts = training.ensemble_predict(models, hold_df)
    y_hold = hold_df["home_win"].values.astype(float)
    full_cal = fit_platt(y_all, np.asarray(oof_blend, dtype=float))
    holdout = {
        "blend": training.compute_metrics(y_hold, np.asarray(blend_hold)),
        "blend_calibrated": training.compute_metrics(
            y_hold, np.asarray(apply_platt(
                np.asarray(blend_hold, dtype=float), full_cal))),
    }

    if partial_path and partial_path.exists():
        partial_path.unlink()
    return {"n_cols": len(cols), "folds_executed": executed,
            "pooled": pooled, "holdout": holdout}


def prepare_data(holdout_days: int, limit_folds: int = 0):
    """Load + enrich the committed snapshot, split holdout, generate the fold
    GEOMETRY ONCE, build the leakage-free margin table on it (run engine
    READ-ONLY — the same prepare_data + margin cache as run_margin_ablation),
    then REGENERATE the same folds over the ENRICHED frame so every arm
    carries the real production ``run_margin_diff`` column (the CURRENT arm
    matches what shipped).

    Fold geometry is a pure function of game_date/home_win, so the
    regenerated splits are row-for-row identical to the ones the margin was
    built on — asserted so a future walk_forward_splits change cannot
    silently desync them. The margin cache key embeds the data hash + fold
    geometry, so it is shared across runs/arms (apples-to-apples).
    """
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

    h = hashlib.sha256()
    h.update(sha256_file(data_path).encode())
    h.update(json.dumps([str(s["val_start"]) for s in folds]).encode())
    cache = Path("/tmp") / f"prune_margin_oof_cache_{h.hexdigest()[:16]}.parquet"

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
    # train/val frames carry the joined margin column.
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
    # step-1 correlation/coverage read uses the enriched frame (has margin)
    games = tune_enriched
    return games, tune_enriched, hold_df, enriched_splits, hold_margins, uncov


def attach(df: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    """Left-join the margin column by game_pk; uncovered rows stay NaN."""
    df = df.drop(columns=[c for c in [MARGIN_COL] if c in df.columns]).copy()
    return df.merge(margins[["game_pk", MARGIN_COL]], on="game_pk", how="left")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", type=str, default="CURRENT,INTERACTIONS,LOW_COVERAGE,CORRELATED,ALL")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--holdout-days", type=int, default=21)
    ap.add_argument("--limit-folds", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="3 folds, output to /tmp, gates skipped")
    ap.add_argument("--target-date", type=str, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.limit_folds = min(args.limit_folds or 3, 3)
        args.out = Path("/tmp/feature_pruning_ablation_smoke.json")
        args.arms = "CURRENT,ALL"

    sha = head_sha()
    (games, tune_df, hold_df, folds, hold_margins, uncov) = \
        prepare_data(args.holdout_days, args.limit_folds)
    hold_enriched = attach(hold_df, hold_margins)

    # STEP 1 read — coverage + correlation on the pre-sealed enriched frame
    # (now carries run_margin_diff, so all 65 cols read).
    base = list(training.FEATURE_COLS)
    read_cols = [c for c in base if c != "run_margin_diff"]
    step1 = redundancy_report(games, read_cols)

    covered = float(tune_df[MARGIN_COL].notna().mean())
    print(f"commit={sha[:12]} games={len(games)} tuning={len(tune_df)} "
          f"holdout={len(hold_enriched)} folds={len(folds)} seed={RANDOM_SEED} clip={EPS}")
    print(f"margin coverage: tuning {covered:.1%} | uncovered_tune_games={uncov}")
    lwp = sorted((k, v) for k, v in step1["coverage"].items() if v < 0.90)
    print("features <90% coverage:", ", ".join(f"{k}={v:.0%}" for k, v in lwp) or "none")
    print(f"|r|>0.8 pairs: {len(step1['corr_gt_0_8_pairs'])}")

    arms = build_arms({
        "INTERACTIONS": PRUNE_INTERACTIONS,
        "LOW_COVERAGE": PRUNE_LOW_COVERAGE,
        "CORRELATED": PRUNE_CORRELATED,
        "ALL": PRUNE_ALL,
    })
    target = args.target_date or pd.Timestamp.now().date().isoformat()
    compact_target = target.replace("-", "")
    out = args.out or (DATA_DELIVERY_DIR / f"feature_pruning_ablation_{compact_target}.json")
    if out.exists():
        results = json.loads(out.read_text())
    else:
        results = {
            "schema": "feature-pruning-ablation/v1",
            "commit_sha": sha,
            "data_sha256": sha256_file(DATA_DELIVERY_DIR / "game_level_features.csv"),
            "holdout_days": args.holdout_days,
            "folds_executed": len(folds),
            "margin_col": MARGIN_COL,
            "margin_source": "run_engine per-side Poisson, OOF on the "
                             "moneyline's own folds (read-only)",
            "margin_coverage_tuning": round(float(covered), 4),
            "uncovered_tune_games": int(uncov),
            "step1": step1,
            "prune_sets": {
                "PRUNE_INTERACTIONS": PRUNE_INTERACTIONS,
                "PRUNE_LOW_COVERAGE": PRUNE_LOW_COVERAGE,
                "PRUNE_CORRELATED": PRUNE_CORRELATED,
                "PRUNE_ALL": PRUNE_ALL,
            },
            "arms": {},
        }

    # STEP 1 addendum — feature importances/blend weights from the shipped
    # model monitor (read-only), merged additively so an existing record is
    # enriched without re-running cached arms.
    if "model_monitor" not in results["step1"]:
        results["step1"]["model_monitor"] = monitor_importance_report(
            step1["corr_gt_0_8_pairs"])
        out.write_text(json.dumps(results, indent=2) + "\n")
        src = results["step1"]["model_monitor"].get("source")
        zc = results["step1"]["model_monitor"].get("weight_zero_in_corr_pair", [])
        print(f"  step1: added model_monitor importance/weight table "
              f"(source={src}, weight-zero-in-corr-pair={zc})")

    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    for name in want:
        if name in results["arms"]:
            print(f"  {name}: cached, skipping")
            continue
        print(f"  {name}: running ({len(arms[name])} cols) ...", flush=True)
        r = run_arm(arms[name], folds, tune_df, hold_enriched,
                    partial_path=Path(str(out) + ".partial.json"))
        r["cols"] = arms[name]
        results["arms"][name] = r
        out.write_text(json.dumps(results, indent=2) + "\n")
        b = r["pooled"]["blend"]
        print(f"    pooled blend {b['logloss']:.4f}/{b['auc']:.4f} "
              f"ece {b['ece']:.4f}", flush=True)

    # ── STEP 3: cheap pooled pre-check — does ANY prune arm beat current ──
    cur = results["arms"].get("CURRENT")
    if cur and not args.smoke:
        cur_p = cur["pooled"]["blend"]
        cleared: list[str] = []
        pooled_verdict: dict[str, dict] = {}
        for name in want:
            if name == "CURRENT" or name not in results["arms"]:
                continue
            p = results["arms"][name]["pooled"]["blend"]
            better = (p["logloss"] < cur_p["logloss"]
                      and p["auc"] > cur_p["auc"])
            pooled_verdict[name] = {
                "logloss": p["logloss"], "auc": p["auc"],
                "better_than_current_both": bool(better),
            }
            if better:
                cleared.append(name)
        results["step3"] = {
            "rule": "prune arm improves pooled logloss AND auc vs CURRENT "
                    "→ proceeds to sealed 284",
            "current_pooled": {k: cur_p[k] for k in ("logloss", "auc", "ece", "brier")},
            "arms": pooled_verdict,
        }
        print(f"\nSTEP 3 pooled pre-check: {len(cleared)} arm(s) clear -> "
              f"{cleared if cleared else 'none — sealed skipped'}")
        if not cleared:
            results["verdict"] = {
                "decision": "DON'T ADOPT",
                "reason": "no prune arm improves pooled logloss AND auc vs current; "
                          "sealed run skipped (existing feature set is at/above ceiling)",
            }
            out.write_text(json.dumps(results, indent=2) + "\n")
            print("VERDICT: DON'T ADOPT — no pooled signal, sealed skipped")
            return

        # ── STEP 4: sealed 284 for the surviving arms ─────────────────────
        cur_h = cur["holdout"]
        sealed_verdict: dict[str, dict] = {}
        adopt_candidates = []
        for name in cleared:
            a = results["arms"][name]
            h = a["holdout"]
            hc = h.get("blend_calibrated", h["blend"])
            cur_hc = cur_h.get("blend_calibrated", cur_h["blend"])
            improves = (h["blend"]["logloss"] < cur_h["blend"]["logloss"]
                        and h["blend"]["auc"] > cur_h["blend"]["auc"])
            ece_ok = hc["ece"] <= cur_hc["ece"] + 1e-9
            pooled_not_regressed = (
                a["pooled"]["blend"]["logloss"] <= cur_p["logloss"]
                and a["pooled"]["blend"]["auc"] >= cur_p["auc"])
            sealed_verdict[name] = {
                "sealed_logloss": h["blend"]["logloss"],
                "sealed_auc": h["blend"]["auc"],
                "sealed_ece_cal": hc["ece"],
                "current_sealed_logloss": cur_h["blend"]["logloss"],
                "current_sealed_auc": cur_h["blend"]["auc"],
                "current_sealed_ece_cal": cur_hc["ece"],
                "improves_sealed_both": bool(improves),
                "ece_not_degraded": bool(ece_ok),
                "pooled_not_regressed": bool(pooled_not_regressed),
                "adopt": bool(improves and ece_ok and pooled_not_regressed),
            }
            if sealed_verdict[name]["adopt"]:
                adopt_candidates.append(name)
        results["step4"] = {"rule": "sealed logloss AND auc improve, ece not "
                                    "degraded, pooled not regressed → ADOPT",
                            "arms": sealed_verdict}
        decision = "ADOPT" if adopt_candidates else "DON'T ADOPT"
        reason = (f"sealed 284 improvement on {adopt_candidates}" if adopt_candidates
                  else "no prune arm improves sealed logloss AND auc without "
                       "degrading ece and regressing pooled; pooled-gain/sealed-loss "
                       "inversion observed in prior gates")
        results["verdict"] = {"decision": decision, "reason": reason,
                              "adopt_candidates": adopt_candidates}
        out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nSTEP 4 sealed verdict: {decision} -> "
              f"{adopt_candidates if adopt_candidates else 'none'}")
        for name in cleared:
            v = sealed_verdict[name]
            print(f"  {name}: sealed {v['sealed_logloss']:.4f}/{v['sealed_auc']:.4f} "
                  f"ece-cal {v['sealed_ece_cal']:.4f} "
                  f"(current {v['current_sealed_logloss']:.4f}/"
                  f"{v['current_sealed_auc']:.4f} ece-cal "
                  f"{v['current_sealed_ece_cal']:.4f}) -> "
                  f"{'ADOPT' if v['adopt'] else 'no'}")


if __name__ == "__main__":
    main()