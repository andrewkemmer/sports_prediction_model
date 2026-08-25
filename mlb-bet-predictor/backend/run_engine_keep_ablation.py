"""Run-engine-native keep-list ablation: restore the 24 _diff features?

The moneyline-side audit (feature_audit_3b929cfcf3e2.json, committed 2bc3ba1
as a measurement record) recommended restoring all 24 matchup-gap _diff
features — but it was measured through the MONEYLINE harness (binary
logloss/AUC, MLP decision member). The run engine is a per-side LightGBM
Poisson expected-runs model where the "levels only" rule is about λ; this
script settles the question with the run engine's OWN objective and the
market-level calibration of the prices it sells.

Variants (all with the shipped env-LEVEL features, GOLDEN RULE arms):
    A   = current (29 kept: levels + env)                       [29 cols]
    B   = A + 24 _diff matchup-gap features                     [53 cols]
    C   = A + 5 engineered composites                           [34 cols]
    REF = A + 24 diffs + 5 composites (lineup_actual_* stay OUT) [58 cols]

Same protocol as the 3.5c ablation: identical walk-forward folds (48 declared
/ 44 executed on the 4,451-game snapshot), pooled OOF, per-side home/away,
deterministic seed, all arms sharing the same folds. Per side × variant:
Poisson deviance, RMSE, MAE, χ²/df, α̂, P(X≤1) / P(X≥10) modeled vs observed.
Market leg: pooled OOF logloss/Brier/ECE-raw/ECE-cal vs base-rate on the
reference lines (over 7.5/8.5/9.5, home cover −1.5/−2.5, derived moneyline),
plus the sealed 21-day holdout read per line (derive_markets_v3).

DECISION RULE (3.5c standard, run-engine version): restore the 24 diffs ONLY
if B beats A on deviance AND RMSE on BOTH sides (no one-sided or near-the-
money regression) without degrading market calibration (mean ECE-cal on the
reference lines) or the deep-tail fit. C and REF are context only.

Artifacts cached at /tmp/run_engine_keep_ablation_<arm>.parquet/.json so each
arm runs as a separate invocation; `compare` emits the decision table.

Usage:
    python3 run_engine_keep_ablation.py --arm A|B|C|REF|compare
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DELIVERY_DIR  # noqa: E402
from run_engine import (  # noqa: E402
    derive_markets_v3,
    derive_run_features,
    dispersion_ratio,
    fit_alpha,
    fit_check_table,
    poisson_deviance,
    run_oof,
)
from training import FEATURE_COLS  # noqa: E402

TMP = Path("/tmp")
OUT_DIR = DATA_DELIVERY_DIR
ARM_LABELS = {"A": "A (29 kept)", "B": "A + 24 diffs",
              "C": "A + 5 composites", "REF": "58 cols (A+B+C)"}


def head_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=Path(__file__).resolve().parent.parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_games() -> pd.DataFrame:
    """Load the 4,451-game snapshot; env-LEVEL features are already applied
    in the refreshed CSV (post-FULL_REPULL) — assert the coverage floors."""
    csv_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    df = pd.read_csv(csv_path)
    assert len(df) == 4451, f"expected 4,451 games, got {len(df)}"
    for c in ("park_wind_factor", "park_factor_slug", "dome_is_neutral_game"):
        cov = int(pd.to_numeric(df[c], errors="coerce").notna().sum())
        assert cov / len(df) >= 0.99, f"{c} coverage {cov}/{len(df)}"
    return df


def build_variants() -> dict[str, list[str]]:
    kept, dropped = derive_run_features(list(FEATURE_COLS))
    diffs = [d for d in dropped if d.endswith("_diff")]
    # The 5 engineered composites (RUN_EXTRA_EXCLUSIONS minus the 6
    # moneyline-scoped lineup_actual_* columns, which stay out of REF by
    # design — their own ablation shipped them moneyline-only).
    composites = [d for d in dropped
                  if not d.endswith("_diff") and d not in (
                      "lineup_actual_woba_delta_home",
                      "lineup_actual_woba_delta_away",
                      "lineup_actual_top3_delta_home",
                      "lineup_actual_top3_delta_away",
                      "lineup_rest_count_home", "lineup_rest_count_away")]
    assert len(kept) == 29 and len(diffs) == 24 and len(composites) == 5, (
        f"unexpected keep-list: kept={len(kept)} diffs={len(diffs)} "
        f"composites={len(composites)}")
    return {
        "A": (list(kept), []),
        "B": (list(kept) + diffs, diffs),
        "C": (list(kept) + composites, composites),
        "REF": (list(kept) + diffs + composites, diffs + composites),
    }


def _tails(oof: pd.DataFrame) -> dict:
    """Pooled NB tail check per side at the arm's own fitted alpha."""
    out = {}
    for side in ("home", "away"):
        y = oof[f"{side}_score"].to_numpy(float)
        lam = oof[f"{side}_expected_runs"].to_numpy(float)
        alpha = fit_alpha(y, lam)
        rows = {r["k"]: r for r in fit_check_table(y, lam, alpha)}
        out[side] = {
            "alpha": round(float(alpha), 4),
            "chi2_df": round(dispersion_ratio(y, lam), 4),
            "p_le_1_model": round(rows["≤1"]["modeled_p"], 4),
            "p_le_1_obs": round(rows["≤1"]["observed_p"], 4),
            "tail10_model": round(float(
                y.size and sum(r["modeled_p"] for k, r in rows.items()
                               if isinstance(k, int) and k >= 10)), 4),
            "tail10_obs": round(float((y >= 10).mean()), 4),
        }
    return out


def run_arm(arm: str) -> None:
    t0 = time.time()
    feats, dropped = build_variants()[arm]
    print(f"\n=== ARM {arm} — {ARM_LABELS[arm]} ({len(feats)} cols) ===", flush=True)
    games = load_games()
    result = run_oof(games, run_features=feats, dropped=dropped)
    oof, summary = result["oof"], result["summary"]
    summary["arm"] = arm
    summary["label"] = ARM_LABELS[arm]
    summary["n_feature_cols"] = len(feats)
    summary["kept"] = feats
    summary["dropped"] = dropped
    summary["tails"] = _tails(oof)
    # Market leg (α(λ) curves + MC, sealed 21-day holdout per line).
    mk = derive_markets_v3(oof)
    summary["markets"] = {k: v for k, v in mk["summary"].items()
                          if k.startswith("market_")}
    oof.to_parquet(TMP / f"run_engine_keep_ablation_{arm}.parquet")
    (TMP / f"run_engine_keep_ablation_{arm}.json").write_text(
        json.dumps(summary, indent=2))
    print(f"arm {arm}: done in {time.time() - t0:.0f}s "
          f"(folds={summary['n_folds']}, games={summary['n_games']})", flush=True)


def _market_table(s: dict) -> pd.DataFrame:
    rows = []
    for k in sorted(s["markets"]):
        if k.endswith("_holdout"):
            continue
        m = s["markets"][k]
        h = m.get("holdout")
        rows.append({
            "line": k.replace("market_", ""),
            "logloss": m["engine_logloss"],
            "brier": m["engine_brier"],
            "ece_raw": m["engine_ece_raw"],
            "ece_cal": m["engine_ece_calibrated"],
            "base_rate": m["baseline_rate"],
            "base_ll": m["baseline_logloss"],
            "hold_ll": h["engine_logloss"] if h else None,
            "hold_ece_cal": (ece_cal_of(h) if h else None),
            "hold_n": h["n"] if h else None,
        })
    return pd.DataFrame(rows)


def ece_cal_of(h: dict) -> float:
    # holdout rows carry raw ECE only; recompute calibrated is overkill here —
    # pooled ECE-cal is the decision leg, holdout is the tie-breaker.
    return round(float(h["engine_ece_raw"]), 5)


def _gate(summaries: dict[str, dict]) -> list[str]:
    """B-vs-A on core metrics (both sides), market ECE-cal, tail fit."""
    a, b = summaries["A"], summaries["B"]
    lines = []
    core_ok = True
    for side in ("home", "away"):
        pa, pb = a[f"{side}_pooled"], b[f"{side}_pooled"]
        d_dev = pb["poisson_deviance"] - pa["poisson_deviance"]
        d_rmse = pb["rmse"] - pa["rmse"]
        ok = d_dev <= 0 and d_rmse <= 0
        core_ok = core_ok and ok
        lines.append(
            f"{side}: Δdev={d_dev:+.4f} ΔRMSE={d_rmse:+.4f} "
            f"{'OK' if ok else 'REGRESSED'}")
    cal_lines = ["over_7_5", "over_8_5", "over_9_5", "home_cover_1_5",
                 "home_cover_2_5", "derived_moneyline"]
    ece_a = np.mean([a["markets"][f"market_{k}"]["engine_ece_calibrated"]
                     for k in cal_lines])
    ece_b = np.mean([b["markets"][f"market_{k}"]["engine_ece_calibrated"]
                     for k in cal_lines])
    cal_ok = ece_b <= ece_a + 0.002
    lines.append(f"mean ECE-cal (6 ref lines): {ece_a:.4f} -> {ece_b:.4f} "
                 f"{'OK' if cal_ok else 'DEGRADED'}")
    tail_ok = True
    for side in ("home", "away"):
        ta, tb = a["tails"][side], b["tails"][side]
        ga = abs(ta["tail10_model"] - ta["tail10_obs"])
        gb = abs(tb["tail10_model"] - tb["tail10_obs"])
        ok = gb <= ga * 1.05
        tail_ok = tail_ok and ok
        lines.append(f"{side} tail10 |gap|: {ga:.3f} -> {gb:.3f} "
                     f"{'OK' if ok else 'REGRESSED'}")
    verdict = (core_ok and cal_ok and tail_ok)
    lines.append(f"VERDICT: {'SHIP the 24 diffs' if verdict else 'DO NOT SHIP'}")
    return lines


def compare() -> None:
    print("\n============= RUN-ENGINE KEEP-LIST ABLATION =============")
    summaries = {}
    for arm in ("A", "B", "C", "REF"):
        p = TMP / f"run_engine_keep_ablation_{arm}.json"
        if not p.exists():
            print(f"missing {p} — run --arm {arm} first")
            return
        summaries[arm] = json.loads(p.read_text())
    # 1) Core per-side table
    rows = []
    for arm in ("A", "B", "C", "REF"):
        s = summaries[arm]
        for side in ("home", "away"):
            p = s[f"{side}_pooled"]
            t = s["tails"][side]
            rows.append({
                "arm": arm, "side": side,
                "deviance": p["poisson_deviance"], "rmse": p["rmse"],
                "mae": p["mae"], "chi2_df": t["chi2_df"], "alpha_hat": t["alpha"],
                "P(<=1) m/o": f"{t['p_le_1_model']:.3f}/{t['p_le_1_obs']:.3f}",
                "P(>=10) m/o": f"{t['tail10_model']:.3f}/{t['tail10_obs']:.3f}",
            })
    print("\n--- PER-SIDE CORE METRICS (pooled OOF) ---")
    print(pd.DataFrame(rows).to_string(index=False))
    # 2) Market leg
    print("\n--- MARKET LEG (pooled OOF; holdout ll in parens) ---")
    m_rows = []
    for arm in ("A", "B", "C", "REF"):
        s = summaries[arm]
        for k in sorted(s["markets"]):
            if k.endswith("_holdout"):
                continue
            m = s["markets"][k]
            h = m.get("holdout")
            m_rows.append({
                "arm": arm, "line": k.replace("market_", ""),
                "logloss": m["engine_logloss"], "brier": m["engine_brier"],
                "ece_raw": m["engine_ece_raw"], "ece_cal": m["engine_ece_calibrated"],
                "base_ll": m["baseline_logloss"],
                "hold_ll": h["engine_logloss"] if h else None,
                "hold_n": h["n"] if h else None,
            })
    print(pd.DataFrame(m_rows).to_string(index=False))
    # 3) Decision
    print("\n--- DECISION RULE (B vs A) ---")
    for line in _gate(summaries):
        print(line)
    # 4) Persist the combined record
    record = {
        "head_sha": head_sha(),
        "data_file": str(DATA_DELIVERY_DIR / "game_level_features.csv"),
        "data_sha256": sha256_file(DATA_DELIVERY_DIR / "game_level_features.csv"),
        "n_games": summaries["A"]["n_games"],
        "n_folds": summaries["A"]["n_folds"],
        "arms": {arm: summaries[arm] for arm in ("A", "B", "C", "REF")},
        "gate": _gate(summaries),
    }
    out = OUT_DIR / f"run_engine_keep_ablation_{record['head_sha']}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\nrecord -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["A", "B", "C", "REF", "compare"],
                    default="compare")
    args = ap.parse_args()
    if args.arm == "compare":
        compare()
    else:
        run_arm(args.arm)


if __name__ == "__main__":
    main()
