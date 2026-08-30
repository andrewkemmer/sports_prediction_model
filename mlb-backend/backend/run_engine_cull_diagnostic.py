"""run_engine_cull_diagnostic.py — READ-ONLY feature-cull diagnostic (untracked).

Question: the run engine's 29-feature view drops 30 of 59 FEATURE_COLS by a
STATIC rule (derive_run_features: *_diff except park_factor_slug_diff, the 5
RUN_EXTRA_EXCLUSIONS composites). The noise-adjusted PSI drift monitor
(compute_run_engine_feature_drift) is a MONITOR, not a selector — it never
culls. The derived ML (P(home win) from the margin distribution) is
compressed (0.50-0.60, AUC ~0.55); is the aggressive cull a cause?

This harness answers it on the CURRENT post-fix frame:
  * arm A    = 29 kept (production)                       [29 cols + env]
  * arm B    = A + 24 matchup-gap _diff features          [53 cols + env]
  * arm FULL = A + 24 diffs + 5 composites                [58 cols + env]
  (run_margin_diff — FEATURE_COLS #59 — is structurally excluded from the
   run engine by the same *_diff rule and is absent from the frame; it is a
   moneyline-side λ-derived feature. See the report's "denominator" note.)

Per arm: pooled per-side Poisson deviance / RMSE / MAE (same protocol as
run_engine_keep_ablation) + the derived-moneyline market leg (AUC, ECE,
logloss, probability spread) via derive_markets_v3 — the SAME market path
the shipped run uses, so the compression question is answered with the
shipped α(λ)/MC machinery.

Drift: per-feature noise-adjusted PSI on the SAME baseline/current windows
the pipeline uses (current = last 7 days of decided games, baseline =
tail(max(3x, 250)) of the prior history) — computed for ALL 59 features
(the shipped monitor only ever sees the 29 kept ones).

Importance: per-feature LightGBM gain from a single global fit per side on
the 58-col arm (same RUN_LGBM_PARAMS; early-stopped on the last 20% by
date). Labeled GLOBAL-FIT (not walk-forward) — a ranking diagnostic, not a
selection rule.

False-positive-cull flag: a DROPPED feature whose gain importance is at or
above the kept-feature median while its noise-adjusted PSI is at/below its
noise floor → the static rule culled signal with no measured drift.

Usage (each arm is one invocation, JSON cached under /tmp):
    python3 run_engine_cull_diagnostic.py --arm A|B|FULL
    python3 run_engine_cull_diagnostic.py --drift        (PSI for all 59)
    python3 run_engine_cull_diagnostic.py --importance   (global-fit gain)
    python3 run_engine_cull_diagnostic.py --report       (merge + verdict)

Writes (untracked): data_delivery/run_engine_cull_diagnostic_<stamp>.json.
ZERO changes to run_engine.py, the sampler, λ/α(λ), training.py, frontend.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DELIVERY_DIR  # noqa: E402
from data_ingestion import load_game_features  # noqa: E402
from explainability import (  # noqa: E402
    classify_drift_retention,
    compute_feature_drift,
)
from frames import get_decided_frame  # noqa: E402
from run_engine import (  # noqa: E402
    EARLY_STOPPING_ROUNDS,
    MAX_ROUNDS,
    RUN_LGBM_PARAMS,
    derive_markets_v3,
    derive_run_features,
    run_oof,
)
from training import FEATURE_COLS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = DATA_DELIVERY_DIR
TMP = Path("/tmp")
STAMP = "20260830"
ARM_LABELS = {"A": "A (29 kept)", "B": "B (29+24 diffs)",
              "FULL": "FULL (58: kept+diffs+composites)"}


def load_decided() -> pd.DataFrame:
    df = load_game_features(DATA / "game_level_features.csv")
    decided = get_decided_frame(df)
    missing = [c for c in FEATURE_COLS if c not in decided.columns]
    if missing:
        print(f"note: FEATURE_COLS absent from frame: {missing} "
              f"(run_margin_diff is expected — moneyline-only, λ-derived)")
    return decided


def build_arms() -> dict[str, tuple[list[str], list[str]]]:
    kept, dropped = derive_run_features(list(FEATURE_COLS))
    diffs = [d for d in dropped if d.endswith("_diff")]
    composites = [d for d in dropped if not d.endswith("_diff")]
    assert len(kept) == 29 and len(diffs) == 25 and len(composites) == 5, (
        f"unexpected selection: kept={len(kept)} diffs={len(diffs)} "
        f"composites={len(composites)}")
    # run_margin_diff is a moneyline-side λ-derived feature: excluded from
    # the run engine by the same *_diff rule and absent from the frame.
    assert "run_margin_diff" in diffs and "run_margin_diff" not in kept
    usable_diffs = [d for d in diffs if d != "run_margin_diff"]
    return {
        "A": (list(kept), []),
        "B": (list(kept) + usable_diffs, usable_diffs),
        "FULL": (list(kept) + usable_diffs + composites,
                 usable_diffs + composites),
    }


def _market_leg(oof: pd.DataFrame) -> dict:
    mk = derive_markets_v3(oof, n_draws=10_000)
    s = mk["summary"]
    dm = s["market_derived_moneyline"]
    dmh = s.get("market_derived_moneyline_holdout", {})
    p = mk["markets"]["p_home_win_derived"].to_numpy(float)
    q = np.nanpercentile(p, [5, 25, 50, 75, 95])
    return {
        "auc": dm["auc"],
        "logloss": dm["engine_logloss"],
        "logloss_calibrated": dm["engine_logloss_calibrated"],
        "ece_raw": dm["engine_ece_raw"],
        "ece_cal": dm["engine_ece_calibrated"],
        "base_rate": dm["baseline_rate"],
        "beats_baseline": dm["beats_baseline_logloss"],
        "spread": {
            "min": round(float(p.min()), 4), "max": round(float(p.max()), 4),
            "sd": round(float(p.std(ddof=0)), 4),
            "p5": round(float(q[0]), 4), "p25": round(float(q[1]), 4),
            "p50": round(float(q[2]), 4), "p75": round(float(q[3]), 4),
            "p95": round(float(q[4]), 4),
            "share_lt_0_45": round(float((p < 0.45).mean()), 4),
            "share_gt_0_55": round(float((p > 0.55).mean()), 4),
        },
        "n": dm["n"],
        "n_pre": s.get("n_pre"), "n_holdout": s.get("n_holdout"),
        "holdout_logloss": dmh.get("engine_logloss"),
        "holdout_n": (dmh.get("holdout") or {}).get("n"),
    }


def run_arm(arm: str) -> None:
    t0 = time.time()
    feats, dropped = build_arms()[arm]
    decided = load_decided()
    print(f"=== ARM {arm} ({ARM_LABELS[arm]}, {len(feats)} cols, "
          f"{len(decided)} decided) ===", flush=True)
    result = run_oof(decided, run_features=feats, dropped=dropped)
    oof, summary = result["oof"], result["summary"]
    record = {
        "arm": arm, "label": ARM_LABELS[arm], "n_feature_cols": len(feats),
        "kept": feats, "dropped": dropped,
        "n_folds": summary["n_folds"], "n_games": summary["n_games"],
        "per_side": {
            side: {"poisson_deviance": summary[f"{side}_pooled"]["poisson_deviance"],
                   "rmse": summary[f"{side}_pooled"]["rmse"],
                   "mae": summary[f"{side}_pooled"]["mae"],
                   "dispersion_ratio": summary[f"{side}_dispersion_ratio"]}
            for side in ("home", "away")},
        "market_derived_moneyline": _market_leg(oof),
    }
    out = TMP / f"run_engine_cull_{arm}.json"
    out.write_text(json.dumps(record, indent=2))
    oof.to_parquet(TMP / f"run_engine_cull_{arm}.parquet")
    print(f"arm {arm}: {time.time() - t0:.0f}s, {len(oof)} OOF rows -> {out}",
          flush=True)


def run_drift() -> None:
    decided = load_decided()
    decided = decided.sort_values("game_date")
    gd = pd.to_datetime(decided["game_date"])
    target = gd.max()
    cutoff = target - pd.Timedelta(days=7)
    current = decided[gd >= cutoff]
    prior = decided[gd < cutoff]
    baseline = prior.tail(max(3 * len(current), 250))
    cols = [c for c in FEATURE_COLS if c in decided.columns]
    print(f"drift windows: baseline={len(baseline)} (>= {baseline['game_date'].min()}.."
          f"{baseline['game_date'].max()}), current={len(current)} "
          f"({current['game_date'].min()}..{current['game_date'].max()}) "
          f"; {len(cols)}/59 features measured "
          f"(run_margin_diff absent from frame -> INSUFFICIENT)", flush=True)
    df = compute_feature_drift(
        baseline, current, target_date_str=STAMP,
        feature_cols=cols, out_name="run_engine_cull_drift.csv")
    df.to_csv(TMP / "run_engine_cull_drift.csv", index=False)
    print(df[["feature", "psi_adjusted", "noise_floor", "status"]]
          .sort_values("psi_adjusted", ascending=False).to_string(index=False),
          flush=True)


def run_importance() -> None:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    decided = load_decided()
    feats, dropped = build_arms()["FULL"]
    # Same env-level append as build_side_frame (production path).
    present = [c for c in
               ("park_wind_factor", "air_density_level", "park_factor_slug",
                "dome_is_neutral_game") if c in decided.columns]
    cols = feats + present
    decided = decided.sort_values("game_date")
    n_tr = int(len(decided) * 0.8)
    tr, va = decided.iloc[:n_tr], decided.iloc[n_tr:]
    gains = {}
    for side, target in (("home", "home_score"), ("away", "away_score")):
        m = LGBMRegressor(**RUN_LGBM_PARAMS)
        m.set_params(n_estimators=MAX_ROUNDS)
        m.fit(tr[cols].astype(float), tr[target].astype(float),
              eval_set=[(va[cols].astype(float), va[target].astype(float))],
              callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                         log_evaluation(period=0)])
        imp = m.feature_importances_
        best = int(m.best_iteration_ or MAX_ROUNDS)
        for c, g in zip(cols, imp):
            gains.setdefault(c, []).append(float(g))
        print(f"side {side}: best_iter={best}, {len(cols)} cols", flush=True)
    rows = []
    for c, gs in gains.items():
        mean = float(np.mean(gs))
        rows.append({"feature": c, "gain_home": round(gs[0], 3),
                     "gain_away": round(gs[1], 3), "gain_mean": round(mean, 3)})
    out = pd.DataFrame(rows).sort_values("gain_mean", ascending=False)
    out.to_csv(TMP / "run_engine_cull_importance.csv", index=False)
    print(out.to_string(index=False), flush=True)


def _flag_table() -> pd.DataFrame:
    drift = pd.read_csv(TMP / "run_engine_cull_drift.csv")
    imp = pd.read_csv(TMP / "run_engine_cull_importance.csv")
    kept, dropped = derive_run_features(list(FEATURE_COLS))
    usable_diffs = [d for d in dropped if d.endswith("_diff")
                    and d != "run_margin_diff"]
    composites = [d for d in dropped if not d.endswith("_diff")]
    culled = usable_diffs + composites
    t = drift.merge(imp, on="feature", how="left")
    t["culled_by_rule"] = t["feature"].isin(culled)
    t["kept"] = t["feature"].isin(kept)
    kept_med = float(t.loc[t["kept"], "gain_mean"].median())
    t["flag"] = np.where(
        t["culled_by_rule"]
        & t.apply(lambda r: classify_drift_retention(
            r["gain_mean"], r["psi_adjusted"], r["noise_floor"], kept_med),
            axis=1),
        "FLAG", "")
    t["kept_median_gain"] = round(kept_med, 3)
    t["gain_ge_kept_median"] = t["gain_mean"] >= kept_med
    t["drift_within_noise"] = t["psi_adjusted"] <= t["noise_floor"]
    return t[["feature", "kept", "culled_by_rule", "gain_mean",
              "psi_adjusted", "noise_floor", "status",
              "gain_ge_kept_median", "drift_within_noise", "flag"]]


def report() -> None:
    recs = {a: json.loads((TMP / f"run_engine_cull_{a}.json").read_text())
            for a in ("A", "B", "FULL")}
    tab = _flag_table()
    flagged = tab[tab["flag"] == "FLAG"]
    print("\n=== PER-SIDE CORE (pooled OOF) ===")
    for a in ("A", "B", "FULL"):
        r = recs[a]
        print(f"{ARM_LABELS[a]:28s} "
              + "  ".join(f"{s}:dev={r['per_side'][s]['poisson_deviance']:.4f} "
                          f"rmse={r['per_side'][s]['rmse']:.4f}"
                          for s in ("home", "away")))
    print("\n=== DERIVED MONEYLINE (pooled OOF) ===")
    for a in ("A", "B", "FULL"):
        m = recs[a]["market_derived_moneyline"]
        s = m["spread"]
        print(f"{ARM_LABELS[a]:28s} auc={m['auc']} ece_raw={m['ece_raw']:.4f} "
              f"ll={m['logloss']:.4f} (cal {m['logloss_calibrated']:.4f}) "
              f"sd={s['sd']:.4f} p5={s['p5']:.3f} p95={s['p95']:.3f} "
              f"share<0.45={s['share_lt_0_45']:.3f} "
              f"share>0.55={s['share_gt_0_55']:.3f}")
    print("\n=== FALSE-POSITIVE CULL FLAGS ===")
    if len(flagged):
        print(flagged.to_string(index=False))
    else:
        print("none: no culled feature combines kept-median gain with "
              "within-noise drift")
    # B/FULL vs A gate (same decision rule as run_engine_keep_ablation).
    lines = []
    for a in ("B", "FULL"):
        core_ok = all(
            recs[a]["per_side"][s]["poisson_deviance"]
            <= recs["A"]["per_side"][s]["poisson_deviance"] + 1e-9
            and recs[a]["per_side"][s]["rmse"]
            <= recs["A"]["per_side"][s]["rmse"] + 1e-9
            for s in ("home", "away"))
        lines.append(f"{ARM_LABELS[a]}: core B-vs-A {'OK' if core_ok else 'REGRESSED'}")
    print("\n" + "\n".join(lines))
    recs["_flag_table"] = tab.to_dict(orient="records")
    recs["_flags"] = flagged.to_dict(orient="records")
    out = DATA / f"run_engine_cull_diagnostic_{STAMP}.json"
    out.write_text(json.dumps(recs, indent=2))
    print(f"\nrecord -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "FULL"])
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--importance", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.arm:
        run_arm(args.arm)
    if args.drift:
        run_drift()
    if args.importance:
        run_importance()
    if args.report:
        report()


if __name__ == "__main__":
    main()
