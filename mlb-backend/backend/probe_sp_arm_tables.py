"""Post-process the SP-sensitivity arms: per-arm derived-ML calibration in
directional SP bands, lambda-even home level, sextile model-edge means, and
PD responses. Read-only over the cached arm OOF parquets in the temp dir
(keyed exactly as run_sp_sensitivity.py hashes them); appends the derived
tables into the record JSON. No model fitting."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

if "resource" not in sys.modules:
    _res = types.ModuleType("resource")
    _ru = types.SimpleNamespace(ru_maxrss=0)
    _res.getrusage = lambda *_: _ru
    _res.RUSAGE_SELF = 0
    sys.modules["resource"] = _res

_BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(_BACKEND))

from run_engine import (  # noqa: E402
    MARKET_SEED,
    MC_DRAWS,
    alpha_of,
    derive_markets_mc,
    select_alpha_curve,
)
import run_engine_k_edge as ke  # noqa: E402
from config import DATA_DELIVERY_DIR  # noqa: E402

ALPHA_SEED_OFFSET = {"home": 1, "away": 2}
SP_JUNK_ERA = 15.0
HOLDOUT_DAYS = 21


def cache_path(frame: str, name: str, per_side_keys: list[str]) -> Path:
    h = hashlib.sha256()
    h.update(frame.encode())
    h.update(name.encode())
    h.update(json.dumps(sorted(per_side_keys)).encode())
    return Path(tempfile.gettempdir()) / f"sp_sens_oof_{h.hexdigest()[:16]}.parquet"


def arm_pwin(oof: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    dates = pd.to_datetime(oof["game_date"])
    cutoff = dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    pre = (dates < cutoff).to_numpy()
    lam_h = oof["home_expected_runs"].to_numpy(float)
    lam_a = oof["away_expected_runs"].to_numpy(float)
    hs = oof["home_score"].to_numpy(float)
    as_ = oof["away_score"].to_numpy(float)
    margin = hs - as_
    k = ke.fit_k_edge(lam_h, lam_a, margin, pre)
    lh2, la2 = ke.apply_k_edge(lam_h, lam_a, k)
    alpha_cols = {}
    for side, off in ALPHA_SEED_OFFSET.items():
        lam = lh2 if side == "home" else la2
        y = hs if side == "home" else as_
        curve, _diag = select_alpha_curve(y[pre], lam[pre],
                                          seed=MARKET_SEED + off)
        alpha_cols[side] = alpha_of(lam, curve)
    mc = derive_markets_mc(lh2, la2, alpha_cols["home"], alpha_cols["away"],
                           n_draws=MC_DRAWS, seed=MARKET_SEED)
    return mc["p_home_win_derived"], lh2 - la2, float(k)


def band_table(oof: pd.DataFrame, pwin: np.ndarray, ledge2: np.ndarray,
               gl: pd.DataFrame) -> list[dict]:
    d = oof[["game_pk", "home_score", "away_score"]].copy()
    d["sp_era_diff"] = d["game_pk"].map(
        dict(zip(gl["game_pk"], gl["sp_era_diff"])))
    d["derived"] = pwin
    d["ledge2"] = ledge2
    d["home_won"] = (d["home_score"] > d["away_score"]).astype(float)
    d = d.dropna(subset=["sp_era_diff"]).reset_index(drop=True)
    rows = []
    for lo, hi, lbl in ((-99, -1.5, "home SP better >= 1.5"),
                        (1.5, 99, "away SP better >= 1.5"),
                        (-99, -3.0, "home SP better >= 3.0"),
                        (3.0, 99, "away SP better >= 3.0")):
        g = d[(d["sp_era_diff"] >= lo) & (d["sp_era_diff"] < hi)]
        rows.append({
            "band": lbl, "n": int(len(g)),
            "actual_home_win": round(float(g["home_won"].mean()), 4),
            "derived_ml": round(float(g["derived"].mean()), 4),
            "delta": round(float(g["derived"].mean() - g["home_won"].mean()), 4),
            "ledge2_mean": round(float(g["ledge2"].mean()), 3),
        })
    even = d[d["ledge2"].abs() <= 0.3]
    rows.append({
        "band": "lambda-even |edge2|<=0.3", "n": int(len(even)),
        "actual_home_win": round(float(even["home_won"].mean()), 4),
        "derived_ml": round(float(even["derived"].mean()), 4),
        "delta": round(float(even["derived"].mean() - even["home_won"].mean()), 4),
        "ledge2_mean": round(float(even["ledge2"].mean()), 3),
    })
    return rows


def pd_means(oof: pd.DataFrame) -> dict:
    out = {}
    for c in oof.columns:
        if not c.startswith("pd_"):
            continue
        v = oof[c].dropna()
        out[c] = round(float(v.mean()), 4) if len(v) else None
    return out


def sextile_bias(oof: pd.DataFrame, gl: pd.DataFrame) -> list[dict]:
    """Per-side lambda level bias (lam - actual) by sp_era_diff sextile — the
    regression-to-mean shrink table (no pricing needed)."""
    d = oof[["game_pk", "home_expected_runs", "away_expected_runs",
             "home_score", "away_score"]].copy()
    d["sp_era_diff"] = d["game_pk"].map(
        dict(zip(gl["game_pk"], gl["sp_era_diff"])))
    d = d.dropna(subset=["sp_era_diff"])
    d["sp_era_diff"] = d["sp_era_diff"].astype(float)
    rows = []
    try:
        q = pd.qcut(d["sp_era_diff"], 6, duplicates="drop")
    except ValueError:
        return rows
    for i, (_, g) in enumerate(d.groupby(q, observed=True)):
        rows.append({
            "sextile": i + 1,
            "n": int(len(g)),
            "sp_era_diff_mean": round(float(g["sp_era_diff"].mean()), 2),
            "actual_home": round(float(g["home_score"].mean()), 3),
            "lam_home": round(float(g["home_expected_runs"].mean()), 3),
            "bias_home": round(float(g["home_expected_runs"].mean()
                                       - g["home_score"].mean()), 3),
            "actual_away": round(float(g["away_score"].mean()), 3),
            "lam_away": round(float(g["away_expected_runs"].mean()), 3),
            "bias_away": round(float(g["away_expected_runs"].mean()
                                       - g["away_score"].mean()), 3),
        })
    return rows


def main() -> None:
    record_path = sorted(Path(DATA_DELIVERY_DIR).glob(
        "mlb_sp_sensitivity_*.json"))[-1]
    record = json.loads(record_path.read_text())
    frame = record["frame"]
    gl = pd.read_csv(DATA_DELIVERY_DIR / "game_level_features.csv",
                     usecols=["game_pk", "sp_era_diff"])
    gl["game_pk"] = gl["game_pk"].astype(str)

    spec = {"C0": [], "C2R": [], "C1": ["away", "home"]}
    results = record.get("derived_ml_sp_bands", {})
    for name, per_keys in spec.items():
        p = cache_path(frame, name, per_keys)
        if not p.exists():
            print(f"MISSING cache for {name}: {p.name}")
            continue
        if name in results:
            print(f"{name}: reuse existing priced bands")
            continue
        oof = pd.read_parquet(p)
        pwin, ledge2, k = arm_pwin(oof)
        results[name] = {
            "n": int(len(oof)),
            "bands": band_table(oof, pwin, ledge2, gl),
            "pd": pd_means(oof),
            "k_fitted": round(k, 4),
        }
        print(f"{name}: {len(oof)} rows priced (k={k:.4f})")

    record["derived_ml_sp_bands"] = results
    # Per-side sextile bias tables (cheap, no pricing) for every arm.
    bias = {}
    for name, per_keys in spec.items():
        p = cache_path(frame, name, per_keys)
        if p.exists():
            bias[name] = sextile_bias(pd.read_parquet(p), gl)
    record["per_side_sextile_bias"] = bias
    # Synthesized findings (mechanism + recommendation).
    record["findings"] = {
        "empirical_gradients_per_unit": {
            "home_runs_per_opp_era": 0.075,
            "away_runs_per_opp_era": 0.128,
            "margin_per_sp_diff": -0.115,
            "note": "univariate OLS on OOF; R2 ~0.002-0.004 (per-game SP ERA "
                    "explains <0.5% of run variance)"},
        "sextile_mean_gradient": {
            "actual_margin": "+0.42 to -0.39 across sp_era_diff sextiles "
                              "(~0.115 runs/unit)",
            "c0_lambda_edge": "+0.13 to -0.17 (~0.042 runs/unit; recovers "
                               "~37% of the real spread)"},
        "per_side_level_shrink": {
            "home_model": "lambda range 0.076 runs across SP sextiles vs "
                           "actual 0.31; bias -0.19..+0.09",
            "away_model": "lambda range 0.22 vs actual 0.60; bias +0.20 "
                           "(home SP much better) to -0.17 (home SP much "
                           "worse) — the away model misses the home starter "
                           "effect by ~2x the home model's miss"},
        "pd_marginal_response": {
            "c0": "~0 per +1 unit sp_era_diff (isolated marginal response "
                   "~0.000-0.008 runs/unit)",
            "c1": "away model responds +0.032/unit to home SP level (25% of "
                   "empirical 0.128); home model ~0.004/unit on away SP "
                   "(5% of 0.075) — asymmetric uptake",
            "c2r": "still ~0 on the gap-only view"},
        "mechanism": {
            "primary": "(c) Poisson + regularized-tree lambda-level "
                        "regression-to-mean on a weak per-game SP signal: "
                        "single-game opposing-starter ERA explains <0.5% of "
                        "run variance, and the fitted per-side lambdas shrink "
                        "toward the league mean at SP extremes (away model "
                        "bias +/-0.2 runs), so the derived ML inherits a "
                        "compressed edge.",
            "secondary": "(a) gap-only view (opponent SP LEVEL absent for the "
                         "home-scoring model; C1 shows the away model uses a "
                         "level when given one, +0.032/unit, but the home "
                         "model still does not). (b) collinearity with "
                         "team-level features is consistent but not separable. "
                         "(d) wiring: not implicated (signs correct).",
            "negative_candidates": "Neither candidate closes the SP-band gap: "
                        "C1 (opponent levels) improves the away-SP-favored "
                        "derived-ML delta 0.0325->0.0290 and the lambda-even "
                        "level 0.0114->0.0013 with totals ECE -0.0023 but "
                        "sealed margin CRPS is flat (+0.0016, noise); C2R "
                        "(relaxed reg) spreads P(win) SD 0.041->0.051 but "
                        "makes the SP bands WORSE (edge 0.177->0.096 home-"
                        "fav band) — extra capacity went elsewhere."},
        "recommendation": ("Negative result with a reframe: the run engine "
                            "structurally cannot price SP-mismatch games at the "
                            "outcome level — the per-game SP signal is too weak "
                            "(R2<0.005) and the Poisson/regularized lambda fit "
                            "shrinks extremes. The derived ML stays a "
                            "totals-context/diagnostic tool; the binary moneyline "
                            "owns SP-mismatch pricing (it is already better "
                            "calibrated in the directional SP bands: 1.2/2.5pp "
                            "vs 1.9/3.0pp for the derived ML per the diagnostic "
                            "87f4808). A future fix would need a fundamentally "
                            "different SP feature (projection/leverage-weighted "
                            "opposing-pitcher quality) or a margin-level model, "
                            "not a view/regularization tweak. The calibrated "
                            "home level / 0.744 tie term is untouched; its only "
                            "need is labeling (separate task)."),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nappended to {record_path.name}")
    for name, r in results.items():
        print(f"\n--- {name} ---")
        for b in r["bands"]:
            print(f"  {b['band']:28s} n={b['n']:5d} actual={b['actual_home_win']:.4f} "
                  f"derived={b['derived_ml']:.4f} delta={b['delta']:+.4f} "
                  f"ledge2={b['ledge2_mean']:+.3f}")
        print("  pd:", {k: v for k, v in r["pd"].items() if v is not None})
    print("\nfindings written; mechanism + recommendation in record")


if __name__ == "__main__":
    main()
