"""Run-engine ABLATION GATE for the Phase-3.5b env-LEVEL feature additions.

Same fixed walk-forward folds, pooled OOF, everything identical except the
standalone environment-LEVEL columns:

    arm WITHOUT : raw-only view exactly as Phase 3 shipped it
    arm WITH    : + park_wind_factor, air_density_level, park_factor_slug,
                  dome_is_neutral_game (game-accurate roof state)

Reported per side: Poisson deviance, RMSE, MAE, Pearson χ²/df dispersion,
fit-check tails P(X≤1) / P(X≥10) (NB(λ, α̂) with α̂ fitted per arm).

DECISION RULE (from the task): ship if deviance/RMSE improve-or-hold AND the
deep-tail fit improves-or-holds. Near-the-money regression ⇒ do not ship.

Artifacts are written to /tmp (this is an analysis gate, not a pipeline run);
the shipped artifacts keep coming from run_engine_daily in the pipeline.

Usage:
    python3 ablate_env_levels.py [--data path.csv] [--skip-without]
Results are cached at /tmp/ablate_env_levels_<arm>.parquet + .json so each
arm can be run as a separate invocation.
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
from features import (  # noqa: E402
    add_env_level_features,
    refine_dome_game_level,
)
from run_engine import (  # noqa: E402
    dispersion_ratio,
    fit_alpha,
    fit_check_table,
    poisson_deviance,
    run_oof,
)

TMP = Path("/tmp")


def load_games() -> tuple[pd.DataFrame, dict]:
    """Load the feature frame and attach the GAME-level roof flag.

    Returns (games, dome_report). The venue-level dome_is_neutral column is
    never modified; refine_dome_game_level adds dome_is_neutral_game alongside
    it using the StatsAPI roof cache."""
    csv_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    df = pd.read_csv(csv_path)
    cache_path = DATA_DELIVERY_DIR / "statsapi_roof_cache.json"
    roof_states = {}
    if cache_path.exists():
        roof_states = {
            int(k): v for k, v in json.loads(cache_path.read_text()).items()
            if v in ("open", "closed")}
        # Retractable parks report REAL outdoor conditions when OPEN ("Clear"
        # + temp/wind); a null entry WITH a real observation classifies open.
        wx_path = Path("/tmp/statsapi_weather_obs.json")
        if wx_path.exists():
            obs = json.loads(wx_path.read_text())
            for pk, cond in obs.items():
                if roof_states.get(int(pk)) is None and cond == "open":
                    roof_states[int(pk)] = "open"
    before = df["dome_is_neutral"].astype(float)
    df = refine_dome_game_level(df, roof_states=roof_states)
    after = df["dome_is_neutral_game"].astype(float)
    # Production-matching env-level fill: the deployed pipeline calls
    # add_env_level_features right after the dome refinement, which fills
    # park_wind_factor / air_density_level from the committed weather cache
    # (data_delivery/weather_history.parquet). Without this the ablation
    # would measure the stale sparse CSV instead of the full-coverage
    # features the run engine actually trains on.
    df = add_env_level_features(df)
    report = {
        "n_games": int(len(df)),
        "roof_states_known": len(roof_states),
        "venue_flagged_dome": int((before == 1).sum()),
        "still_closed_after_refine": int((after == 1).sum()),
        "affected_open_now": int(((before == 1) & (after == 0)).sum()),
    }
    return df, report


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
            "p_le_1_model": round(rows["≤1"]["modeled_p"], 4),
            "p_le_1_obs": round(rows["≤1"]["observed_p"], 4),
            "tail10_model": round(float(
                y.size and sum(r["modeled_p"] for k, r in rows.items()
                               if isinstance(k, int) and k >= 10)), 4),
            "tail10_obs": round(float((y >= 10).mean()), 4),
        }
    return out


def run_arm(games: pd.DataFrame, include_level_env: bool, tag: str) -> None:
    t0 = time.time()
    print(f"\n=== ARM {'WITH' if include_level_env else 'WITHOUT'} "
          f"env-level features ===", flush=True)
    result = run_oof(games, include_level_env=include_level_env)
    oof, summary = result["oof"], result["summary"]
    summary["tails"] = _tails(oof)
    summary["include_level_env"] = include_level_env
    oof.to_parquet(TMP / f"ablate_env_levels_{tag}.parquet")
    (TMP / f"ablate_env_levels_{tag}.json").write_text(json.dumps(summary))
    print(f"arm {tag}: done in {time.time() - t0:.0f}s "
          f"(folds={summary['n_folds']}, games={summary['n_games']})", flush=True)


def compare() -> None:
    print("\n================= ABLATION GATE =================")
    rows = []
    tails = {}
    for tag, label in (("without", "WITHOUT"), ("with", "WITH")):
        s = json.loads((TMP / f"ablate_env_levels_{tag}.json").read_text())
        tails[label] = s["tails"]
        for side in ("home", "away"):
            p = s[f"{side}_pooled"]
            rows.append({
                "arm": label, "side": side,
                "deviance": p["poisson_deviance"], "rmse": p["rmse"],
                "mae": p["mae"],
                "dispersion_chi2_df": s[f"{side}_dispersion_ratio"],
                "alpha_hat": s["tails"][side]["alpha"],
                "P(X<=1) model/obs":
                    f"{s['tails'][side]['p_le_1_model']:.3f} / "
                    f"{s['tails'][side]['p_le_1_obs']:.3f}",
                "P(X>=10) model/obs":
                    f"{s['tails'][side]['tail10_model']:.3f} / "
                    f"{s['tails'][side]['tail10_obs']:.3f}",
            })
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))

    # Decision rule
    verdict_rows = []
    ship = True
    for side in ("home", "away"):
        w = [r for r in rows if r["arm"] == "WITH" and r["side"] == side][0]
        wo = [r for r in rows if r["arm"] == "WITHOUT" and r["side"] == side][0]
        d_dev = w["deviance"] - wo["deviance"]
        d_rmse = w["rmse"] - wo["rmse"]
        core_ok = d_dev <= 0 or d_rmse <= 0
        tw = tails["WITH"][side]
        two = tails["WITHOUT"][side]
        tail_gap_w = abs(tw["tail10_model"] - tw["tail10_obs"])
        tail_gap_wo = abs(two["tail10_model"] - two["tail10_obs"])
        tail_ok = tail_gap_w <= tail_gap_wo * 1.05
        ok = core_ok and tail_ok
        ship = ship and ok
        verdict_rows.append(f"{side}: Δdev={d_dev:+.4f} ΔRMSE={d_rmse:+.4f} "
                            f"core={'OK' if core_ok else 'REGRESSED'} | "
                            f"|tail10 gap| {tail_gap_wo:.3f}->{tail_gap_w:.3f} "
                            f"tail={'OK' if tail_ok else 'REGRESSED'} "
                            f"=> {'SHIP' if ok else 'DO NOT SHIP'}")
    print("\n--- DECISION RULE ---")
    for v in verdict_rows:
        print(v)
    print(f"\nVERDICT: {'SHIP the env-level features' if ship else 'DO NOT SHIP'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--arm", choices=["with", "without", "both", "compare"],
                    default="both")
    args = ap.parse_args()

    if args.arm == "compare":
        compare()
        return

    games, dome_report = load_games()
    if args.data is not None:
        games = pd.read_csv(args.data)
    print("DOME REPORT:", json.dumps(dome_report))
    have = [c for c in ("park_wind_factor", "air_density_level",
                        "park_factor_slug") if c in games.columns]
    missing = [c for c in ("park_wind_factor", "air_density_level",
                           "park_factor_slug", "dome_is_neutral_game")
               if c not in games.columns]
    print(f"env-level columns present: {have}; missing: {missing}")
    for c in have:
        col = pd.to_numeric(games[c], errors="coerce")
        print(f"  {c}: populated {int(col.notna().sum())}/{len(games)}")

    if args.arm in ("both", "without"):
        run_arm(games, include_level_env=False, tag="without")
    if args.arm in ("both", "with"):
        run_arm(games, include_level_env=True, tag="with")
    if (TMP / "ablate_env_levels_with.json").exists() and \
       (TMP / "ablate_env_levels_without.json").exists():
        compare()


if __name__ == "__main__":
    main()
