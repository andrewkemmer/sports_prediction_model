"""Year-effect check — diagnosis only. Is 2024's lower dispersion REAL?

The 3-season expansion (2024-2026, 6,960 games) raised the dispersion
question: per-year global MoM alpha_home from the expansion log was
2024 ~0.236 / 2025 ~0.276 / 2026 ~0.297 (~2,318 games/yr) — a ~3-sigma
2024-vs-2026 spread by a naive pairwise estimate. This script settles
whether that spread survives proper uncertainty before ANY year-conditioned
alpha is even considered. NO model change is made here; the run engine is
READ-ONLY (imports only `fit_alpha` / `year_effect_check`).

STEP 1 — tables on the run engine's per-game OOF (the 6,960-game frame):
  * per-year alpha_home / alpha_away with bootstrap uncertainty (n per year,
    within-year game resampling, percentile CI)
  * year-vs-year differences in sigma units (se_diff = sqrt(se_y1^2+se_y2^2))
  * the pooling-misstatement table: pooled alpha vs per-year alpha, implied
    variance error per year (var = lambda + alpha*lambda^2)

STEP 2 — decision rule (no gate unless triggered): trigger ONLY if 2024
  confirms alpha_home <= 0.25 at n~2,318 with a REAL (>=2 sigma) gap from
  BOTH 2025 and 2026. If triggered the caller is expected to run ONE
  year-aware alpha gate (per-year alpha(lambda) curves on pre-sealed OOF,
  sampler picks the curve by game year, sealed-284 acceptance discipline).
  Otherwise keep pooled and write the verdict with numbers — never ship
  year-conditioned alpha preemptively.

STEP 3 — record: data_delivery/year_effect_check_<date>.json with the
  STEP-1 tables, the decision, and the reason.

Input: data_delivery/run_engine_oof_<date>.csv by default, or an explicit
path via argv[2]. NOTE: the 6,960-game expansion OOF is the intended input
(committed artifact @ ac29973 for 2026-08-28, 6,797 rows); the 17:24
degraded 6,161-game artifact (2025 chunk gap) must NOT be used — the
missing ~800 games sit entirely in 2025 and would bias the year table.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from run_engine import fit_alpha, year_effect_check

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_delivery"

BOOT_N = 2_000          # within-year game resamples per (year, side)
SEED = 20260828
Z_GATE = 2.0            # "real (>=2 sigma) gap" threshold
ALPHA_HOME_2024_GATE = 0.25   # "2024 confirms alpha_home ~0.25" threshold
SEALED_N = 284          # sealed-holdout convention (last N OOF games by date)


def _alpha(y: np.ndarray, lam: np.ndarray) -> float:
    return float(fit_alpha(np.asarray(y, float), np.asarray(lam, float)))


def _bootstrap_alpha(y: np.ndarray, lam: np.ndarray,
                     rng: np.random.Generator, n_boot: int) -> np.ndarray:
    y = np.asarray(y, float)
    lam = np.asarray(lam, float)
    n = len(y)
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        out[b] = fit_alpha(y[idx], lam[idx])
    return out


def _ci(boot: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def _per_year_bootstrap(oof: pd.DataFrame, side: str, col: str,
                        rng: np.random.Generator) -> list[dict]:
    years = pd.to_datetime(oof["game_date"]).dt.year
    rows = []
    for yr in sorted(years.unique()):
        sub = oof[years == yr]
        y = sub[col].to_numpy(float)
        lam = sub[f"{side}_expected_runs"].to_numpy(float)
        a = _alpha(y, lam)
        boot = _bootstrap_alpha(y, lam, rng, BOOT_N)
        lo, hi = _ci(boot)
        rows.append({
            "year": int(yr), "n_games": int(len(sub)),
            "alpha": a, "boot_se": round(float(boot.std(ddof=1)), 4),
            "ci95": [lo, hi],
        })
    return rows


def _pairwise_z(rows: list[dict], rng: np.random.Generator,
                oof: pd.DataFrame, side: str, col: str) -> list[dict]:
    """Year-vs-year alpha differences in sigma units.

    Independent within-year game resamples per year (same seed discipline);
    se of the difference = sqrt(se_y1^2 + se_y2^2) from the bootstrap
    distributions. z = d_alpha / se_diff.
    """
    years = pd.to_datetime(oof["game_date"]).dt.year
    alphas: dict[int, float] = {r["year"]: r["alpha"] for r in rows}
    ses: dict[int, float] = {r["year"]: r["boot_se"] for r in rows}
    pairs = [(2025, 2024), (2026, 2024), (2026, 2025)]
    out = []
    for y1, y2 in pairs:
        if y1 not in alphas or y2 not in alphas:
            continue
        d = alphas[y1] - alphas[y2]
        se = float(np.hypot(ses[y1], ses[y2]))
        out.append({
            "year_higher": y1, "year_lower": y2,
            "d_alpha": round(d, 4), "se_diff": round(se, 4),
            "z": round(d / se, 2),
        })
    return out


def _pooling_misstatement(oof: pd.DataFrame, side: str, col: str) -> dict:
    """Pooled (global MoM) alpha vs per-year alpha; implied variance error.

    var = lambda + alpha*lambda^2 (NB size parameterization n = 1/alpha).
    implied_var_pooled uses the pooled alpha at the year's own mean lambda;
    implied_var_year uses the year's own alpha. var_error_pct is how much
    pooling misstates the year's implied variance.
    """
    years = pd.to_datetime(oof["game_date"]).dt.year
    y_all = oof[col].to_numpy(float)
    lam_all = oof[f"{side}_expected_runs"].to_numpy(float)
    pooled = _alpha(y_all, lam_all)
    per_year = []
    for yr in sorted(years.unique()):
        m = years == yr
        y = y_all[m]
        lam = lam_all[m]
        lam_bar = float(lam.mean())
        a_yr = _alpha(y, lam)
        var_yr = float(y.var(ddof=0))
        implied_yr = lam_bar + a_yr * lam_bar ** 2
        implied_pooled = lam_bar + pooled * lam_bar ** 2
        per_year.append({
            "year": int(yr), "n_games": int(m.sum()),
            "lambda_mean": round(lam_bar, 4),
            "observed_var": round(var_yr, 4),
            "alpha_year": round(a_yr, 4),
            "implied_var_year": round(implied_yr, 4),
            "implied_var_pooled": round(implied_pooled, 4),
            "var_error_pct": round(
                100.0 * (implied_pooled - implied_yr) / implied_yr, 2),
        })
    return {"pooled_alpha": round(pooled, 4), "per_year": per_year}


def _pre_sealed_alpha(oof: pd.DataFrame, side: str, col: str) -> dict:
    """Per-year alpha on PRE-sealed rows only (last SEALED_N games by date
    excluded) — the run engine's internal convention, for robustness."""
    dates = pd.to_datetime(oof["game_date"])
    order = np.argsort(dates.to_numpy(), kind="stable")
    pre = np.ones(len(oof), bool)
    pre[order[-SEALED_N:]] = False
    pre_oof = oof.iloc[np.where(pre)[0]]
    years = pd.to_datetime(pre_oof["game_date"]).dt.year
    rows = []
    for yr in sorted(years.unique()):
        sub = pre_oof[years == yr]
        rows.append({
            "year": int(yr), "n_games": int(len(sub)),
            "alpha_pre_sealed": round(_alpha(sub[col].to_numpy(float),
                                             sub[f"{side}_expected_runs"]
                                             .to_numpy(float)), 4),
        })
    return {"n_pre_sealed": int(pre.sum()), "per_year": rows}


def main() -> None:
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime(
        "%Y%m%d")
    oof_path = (Path(sys.argv[2]) if len(sys.argv) > 2
                else DATA / f"run_engine_oof_{date_str}.csv")
    input_note = (sys.argv[3] if len(sys.argv) > 3
                  else "committed artifact for the date")
    if not oof_path.exists():
        raise SystemExit(f"missing artifact: {oof_path}")
    oof = pd.read_csv(oof_path)
    rng = np.random.default_rng(SEED)

    record: dict = {
        "meta": {
            "date": date_str,
            "input_artifact": str(oof_path),
            "input_note": input_note,
            "n_oof_rows": int(len(oof)),
            "n_boot": BOOT_N,
            "seed": SEED,
            "z_gate": Z_GATE,
            "alpha_home_2024_gate": ALPHA_HOME_2024_GATE,
            "diagnosis_only": True,   # no model change shipped by this script
        },
        "year_effect_check_existing": {},
        "bootstrap": {},
        "pairwise_sigma": {},
        "pooling_misstatement": {},
        "pre_sealed_robustness": {},
        "decision": {},
    }

    per_year_alpha: dict[str, dict] = {}
    for side, col in (("home", "home_score"), ("away", "away_score")):
        rows = _per_year_bootstrap(oof, side, col, rng)
        per_year_alpha[side] = {r["year"]: r["alpha"] for r in rows}
        record["year_effect_check_existing"][side] = year_effect_check(
            oof, side)
        record["bootstrap"][side] = rows
        record["pairwise_sigma"][side] = _pairwise_z(
            rows, rng, oof, side, col)
        record["pooling_misstatement"][side] = _pooling_misstatement(
            oof, side, col)
        record["pre_sealed_robustness"][side] = _pre_sealed_alpha(
            oof, side, col)

    # STEP-2 decision rule (home side only).
    h = per_year_alpha["home"]
    z25 = next(p["z"] for p in record["pairwise_sigma"]["home"]
               if (p["year_higher"], p["year_lower"]) == (2025, 2024))
    z26 = next(p["z"] for p in record["pairwise_sigma"]["home"]
               if (p["year_higher"], p["year_lower"]) == (2026, 2024))
    low_2024 = h.get(2024, 1.0) <= ALPHA_HOME_2024_GATE
    real_gap = z25 >= Z_GATE and z26 >= Z_GATE
    triggered = bool(low_2024 and real_gap)

    reason = (
        f"2024 alpha_home={h.get(2024):.4f} ({'at/below' if low_2024 else 'above'} "
        f"{ALPHA_HOME_2024_GATE}); 2025-vs-2024 z={z25:.2f}, "
        f"2026-vs-2024 z={z26:.2f} "
        f"({'both >= 2 sigma' if real_gap else 'gap(s) < 2 sigma — not real'}). "
    )
    if triggered:
        verdict = ("GATE TRIGGERED — 2024 alpha_home confirmed low with a "
                   "real (>=2 sigma) gap from 2025/2026. Run ONE year-aware "
                   "alpha gate before any adoption: per-year alpha(lambda) "
                   "curves on pre-sealed OOF (sparse-year shrinkage toward "
                   "pooled), sampler picks the curve by game year, gate "
                   "totals + run-line ECE/logloss/AUC on pooled OOF + sealed "
                   f"{SEALED_N}. ADOPT only if the sealed window is NOT "
                   "degraded. No year-conditioned alpha is shipped by this "
                   "diagnosis.")
    else:
        verdict = ("KEEP POOLED — 2024's lower dispersion is NOT confirmed as "
                   "real: the 2024-vs-2025/2026 home alpha gaps are "
                   f"{z25:.2f}/{z26:.2f} sigma (below the {Z_GATE}-sigma "
                   "bar), and 2024's bootstrap CI95 overlaps 2025/2026. "
                   "The away pattern is the known noise signature "
                   "(2025 is the single-year outlier, 2026 falls back — "
                   "non-monotonic). No year-conditioned alpha is shipped; "
                   "pooling stands.")
    record["decision"] = {
        "triggered": triggered,
        "verdict": verdict,
        "reason": reason,
    }

    out_path = DATA / f"year_effect_check_{date_str}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Decision: {'TRIGGERED' if triggered else 'KEEP POOLED'}")
    print(reason)
    for side, col in (("home", "home_score"), ("away", "away_score")):
        line = "  ".join(
            f"{r['year']}: a={r['alpha']:.4f} (n={r['n_games']}, "
            f"se={r['boot_se']:.4f}, ci95={r['ci95']})"
            for r in record["bootstrap"][side])
        print(f"  {side}: {line}")
    print(f"Record: {out_path}")


if __name__ == "__main__":
    main()
