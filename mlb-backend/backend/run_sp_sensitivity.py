"""MLB run-engine SP-sensitivity audit + candidate-arm measurement.
Record-only — production engine (run_engine.py, side models, pricing) is
byte-untouched. Triggered by the derived-ML diagnostic (87f4808, verdict A):
the run engine's lambda-edge swings only ~0.45 runs across the SP spectrum
while actual home win rate swings ~7-8pp, and the derived ML inherits the
compression.

Step 1 (feature view): what SP signal does each side model actually see?
Step 2 (measure): marginal lambda sensitivity per unit SP vs the empirical
  target (actual runs/margin on the same SP units), plus feature
  importances.
Step 3 (root cause): rank (a) missing/dropped, (b) collinearity dilution,
  (c) regularization flattening, (d) wiring.
Step 4 (candidate): same folds/seed. C1 = opponent pitching LEVELS appended
  per side (the gap-only view is the structural suspect); C2R = relaxed
  regularization on the production view. Does the lambda-edge gradient close
  toward the empirical target, and does derived-ML calibration in the
  SP-mismatch bands track actuals without regressing totals/derived-ML?

Arms (75-fold walk-forward, cadence 7, min-val 40, seed 42, 6,885 OOF rows):
  C0  = production view (53 levels+env+restored diffs) + production
        RUN_LGBM_PARAMS — validated against the shipped run_engine_oof file.
  C1  = C0 view + the OPPONENT's pitching LEVEL columns appended to each
        side frame (away SP/bullpen into the home-scoring model and vice
        versa), production params.
  C2R = C0 view, relaxed regularization (num_leaves 31, min_child_samples 10,
        min_gain 0.0, subsample/colsample 1.0) — poisson objective, lr, and
        early stopping unchanged.
  P1  = C0 view + the OPPONENT's PROJECTION-QUALITY SP level
        (sp_proj_era_away into the home-scoring model and vice versa) — the
        C1 generalization with a stronger input: a z-composite of the frame's
        Statcast-derived trailing components (FIP, xwOBA, WHIP, BB9, K9-5g,
        whiff-3g, velo-3g), scaled to ERA-equivalent units (1 unit ~= 1 ERA
        point of quality). Replaces raw ERA as the opponent signal.
  P2  = P1 + the raw opponent ERA LEVEL too (sp_proj_era_opp + sp_era_opp):
        is the projection additive to raw ERA or redundant with it? (The raw
        sp_era_diff GAP remains in the shared env of every arm — it is one of
        the 24 restored diffs.)
  P3  = C0 view + opponent projection level AND OWN-side projection level
        (sp_proj_era_opp + sp_proj_era_own): projection as both own-side
        context and opponent signal.

Pricing reuses the run-line expansion harness's C2 layer verbatim
(ke.fit_k_edge/apply_k_edge on pre-holdout OOF; select_alpha_curve/alpha_of
on expanded pre-holdout lambdas; derive_markets_mc NB Monte Carlo) so
totals/derived-ML/margin surfaces are production-faithful per arm. The
calibrated home tie-resolution term (MARGIN_PLUS1_HOME_SHARE) is untouched.

Usage:
    python run_sp_sensitivity.py [--arms C0,C1,C2R] [--smoke] [--limit-folds N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
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

from config import (  # noqa: E402
    DATA_DELIVERY_DIR,
    MIN_VAL_FOLD_GAMES,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
)
from data_ingestion import load_game_features  # noqa: E402
from frames import get_decided_frame  # noqa: E402
from run_engine import (  # noqa: E402
    HOLDOUT_DAYS,
    RUN_LGBM_PARAMS,
    _fit_side_model,
    build_side_frame,
    derive_run_features,
)
import run_engine_k_edge as ke  # noqa: E402
from run_mlb_runline_expansion_ablation import price_arm  # noqa: E402
from training import FEATURE_COLS, walk_forward_splits  # noqa: E402

DATE = "20260903"

# Opponent-pitching LEVEL palette appended per side in C1 (all present in the
# frame; mirrors the pitcher features the moneyline diffs are built from).
OPP_PITCH_COLS = [
    "sp_era", "sp_era_5g", "sp_k9", "sp_k9_5g", "sp_xwoba",
    "sp_xwoba_vs_l", "sp_whiff_3g", "sp_fbvelo_3g", "sp_fbpct_3g",
    "bullpen_whip_10g", "bullpen_whip_3g", "closer_available",
]

# Projection composite components (all PIT-safe trailing windows in the
# frame): lower-is-better and higher-is-better sets. The composite is the
# mean of z-scored components (higher = better pitching), z-stats fit on the
# PRE-HOLDOUT rows only, then scaled to ERA-equivalent units via the
# ERA~composite OLS slope so +1 unit ~= 1 ERA point of quality.
PROJ_LO_BETTER = ["sp_fip", "sp_xwoba", "sp_whip", "sp_bb9"]
PROJ_HI_BETTER = ["sp_k9_5g", "sp_whiff_3g", "sp_fbvelo_3g"]
MIN_PROJ_COMPONENTS = 3

RELAXED_PARAMS = dict(RUN_LGBM_PARAMS)
RELAXED_PARAMS.update({
    "num_leaves": 31,
    "min_child_samples": 10,
    "min_gain_to_split": 0.0,
    "subsample": 1.0,
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
})

SP_JUNK_ERA = 15.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(beta[1]), 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _side_base_cols(games: pd.DataFrame, side: str) -> list[str]:
    feats, _ = derive_run_features(list(FEATURE_COLS))
    return build_side_frame(games, side, run_features=list(feats),
                            dropped=[])[1]


def build_projection_cols(games: pd.DataFrame,
                          pre_mask: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Add sp_proj_era_{home,away} (projection composite in ERA-equivalent
    units) to a copy of the frame, fit on pre-holdout rows only. Returns
    (frame_with_cols, meta). +1 unit ~= 1 ERA point of quality; higher is
    better pitching. Coverage = fraction of OOF rows with a valid projection."""
    df = games.copy()
    meta: dict = {}
    for side in ("home", "away"):
        lo = [f"{c}_{side}" for c in PROJ_LO_BETTER]
        hi = [f"{c}_{side}" for c in PROJ_HI_BETTER]
        z = pd.DataFrame(index=df.index)
        for c in lo + hi:
            mu, sd = df.loc[pre_mask, c].mean(), df.loc[pre_mask, c].std()
            z[c] = (df[c] - mu) / sd
        n_comp = z[lo + hi].notna().sum(axis=1)
        comp = (-z[lo].sum(axis=1, min_count=1)
                + z[hi].sum(axis=1, min_count=1))
        comp = comp.where(n_comp >= MIN_PROJ_COMPONENTS)
        comp = comp / n_comp.where(n_comp >= MIN_PROJ_COMPONENTS)
        df[f"sp_proj_{side}"] = comp
        # ERA-equivalent scale: OLS sp_era ~ comp on pre-holdout, junk out.
        era = f"sp_era_{side}"
        cal = df.loc[pre_mask & comp.notna() & df[era].notna()
                     & (df[era].abs() <= SP_JUNK_ERA)]
        X = np.column_stack([np.ones(len(cal)), cal[f"sp_proj_{side}"].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, cal[era].to_numpy(), rcond=None)
        slope = float(beta[1])
        df[f"sp_proj_era_{side}"] = comp / abs(slope) if slope else comp
        meta[side] = {"era_on_proj_slope": round(slope, 4),
                      "coverage_pre": round(float(comp[pre_mask].notna().mean()), 4),
                      "coverage_sealed": round(float(comp[~pre_mask].notna().mean()), 4)}
    return df, meta


def arm_params_and_frames(name: str, games: pd.DataFrame):
    """Return (params, per_side | None) for an arm. per_side maps side ->
    full column list (production side cols + any arm extras)."""
    feats, _ = derive_run_features(list(FEATURE_COLS))
    if name == "C0":
        return dict(RUN_LGBM_PARAMS), None
    if name == "C2R":
        return dict(RELAXED_PARAMS), None
    if name == "C1":
        per_side = {}
        for side in ("home", "away"):
            cols = build_side_frame(games, side, run_features=list(feats),
                                    dropped=[])[1]
            opp = "away" if side == "home" else "home"
            extras = [c.rsplit("_", 1)[0] + f"_{opp}" for c in OPP_PITCH_COLS]
            extras = [c for c in extras if c in games.columns and c not in cols]
            per_side[side] = list(cols) + extras
        return dict(RUN_LGBM_PARAMS), per_side
    if name in ("P1", "P2", "P3"):
        per_side = {}
        for side in ("home", "away"):
            cols = _side_base_cols(games, side)
            opp = "away" if side == "home" else "home"
            extras = [f"sp_proj_era_{opp}"]
            if name == "P2":
                extras.append(f"sp_era_{opp}")
            if name == "P3":
                extras.append(f"sp_proj_era_{side}")
            extras = [c for c in extras if c in games.columns and c not in cols]
            per_side[side] = list(cols) + extras
        return dict(RUN_LGBM_PARAMS), per_side
    raise SystemExit(f"unknown arm {name!r}")


def _frame_cols(games: pd.DataFrame, side: str,
                per_side: dict | None) -> list[str]:
    if per_side is not None:
        return per_side[side]
    return build_side_frame(games, side, run_features=[])[1]


def walk_arm(name: str, decided: pd.DataFrame, params: dict,
             per_side: dict | None, limit_folds: int = 0) -> pd.DataFrame:
    """75-fold walk: per-game lambda pair + +1-unit PD deltas (computed
    against the fitted fold model) for the SP feature each side can see."""
    folds = [s for s in walk_forward_splits(
        decided, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
        if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]
    if limit_folds:
        folds = folds[:limit_folds]
    rows = []
    for split in folds:
        tr, va = split["train_games"], split["val_games"]
        rec = {
            "game_pk": va["game_pk"].to_numpy(),
            "game_date": pd.to_datetime(va["game_date"]).dt.strftime("%Y-%m-%d"),
            "fold_idx": np.full(len(va), split["fold_idx"]),
            "home_score": va["home_score"].to_numpy(dtype=float),
            "away_score": va["away_score"].to_numpy(dtype=float),
        }
        for side, target in (("home", "home_score"), ("away", "away_score")):
            cols = _frame_cols(decided, side, per_side)
            tr_frame = tr.reindex(columns=cols).astype(float)
            va_frame = va.reindex(columns=cols).astype(float)
            model, lam, best = _fit_side_model(
                params, tr_frame, tr[target].to_numpy(float),
                va_frame, va[target].to_numpy(float))
            rec[f"{side}_expected_runs"] = np.round(lam, 4)
            opp = "away" if side == "home" else "home"
            targets = {"sp_era_diff": "sp_era_diff",
                       f"sp_era_{opp}": f"sp_era_{opp}",
                       f"sp_proj_era_{opp}": f"sp_proj_era_{opp}"}
            for col in targets:
                if col not in cols:
                    rec[f"pd_{col}_{side}"] = np.full(len(va), np.nan)
                    continue
                pert = va_frame.copy()
                pert[col] = pert[col] + 1.0
                lam_p = np.clip(model.predict(pert, num_iteration=best),
                                1e-6, None)
                rec[f"pd_{col}_{side}"] = (lam_p - lam)
        rows.append(pd.DataFrame(rec))
    oof = pd.concat(rows, ignore_index=True)
    oof["game_pk"] = oof["game_pk"].astype(str)
    return oof


def sp_measurements(oof: pd.DataFrame, gl: pd.DataFrame) -> dict:
    """Empirical target + model response, all on the C0 OOF row basis.
    Returns: empirical OLS gradients (actual runs/margin per SP unit),
    sextile means of actual margin vs model lambda edge, and the PD-derived
    model per-unit responses for the arm."""
    d = oof[["game_pk", "game_date", "home_expected_runs",
             "away_expected_runs", "home_score", "away_score"]].merge(
        gl[["game_pk", "sp_era_diff", "sp_era_home", "sp_era_away",
            "sp_proj_era_home", "sp_proj_era_away"]],
        on="game_pk", how="left")
    d = d.dropna(subset=["sp_era_home", "sp_era_away"])
    d = d[(d["sp_era_home"].abs() <= SP_JUNK_ERA) &
          (d["sp_era_away"].abs() <= SP_JUNK_ERA)].copy()
    d["margin"] = d["home_score"] - d["away_score"]
    d["ledge"] = d["home_expected_runs"] - d["away_expected_runs"]
    d["home_won"] = (d["home_score"] > d["away_score"]).astype(float)

    emp = {}
    b, r2 = _ols(d["home_score"].values, d["sp_era_away"].values)
    emp["home_runs_per_opp_era_unit"] = {"slope": round(b, 4), "r2": round(r2, 4)}
    b, r2 = _ols(d["away_score"].values, d["sp_era_home"].values)
    emp["away_runs_per_opp_era_unit"] = {"slope": round(b, 4), "r2": round(r2, 4)}
    b, r2 = _ols(d["margin"].values, d["sp_era_diff"].values)
    emp["margin_per_sp_era_diff_unit"] = {"slope": round(b, 4), "r2": round(r2, 4)}

    # Sextile means on sp_era_diff: actual vs model edge per arm row.
    try:
        q = pd.qcut(d["sp_era_diff"], 6, duplicates="drop")
    except ValueError:
        q = None
    sext = []
    if q is not None:
        for i, (_, g) in enumerate(d.groupby(q, observed=True)):
            sext.append({
                "bin": f"sextile {i + 1}",
                "n": int(len(g)),
                "sp_era_diff_mean": round(float(g["sp_era_diff"].mean()), 3),
                "actual_margin_mean": round(float(g["margin"].mean()), 3),
                "actual_home_win": round(float(g["home_won"].mean()), 4),
                "model_ledge_mean": round(float(g["ledge"].mean()), 3),
            })
    # Projection-space empirical targets (the bar the P-arms must approach,
    # in the SAME feature units the model sees: ERA-equivalent projection).
    b, r2 = _ols(d["home_score"].values, d["sp_proj_era_away"].values)
    emp["home_runs_per_opp_proj_era_unit"] = {"slope": round(b, 4), "r2": round(r2, 4)}
    b, r2 = _ols(d["away_score"].values, d["sp_proj_era_home"].values)
    emp["away_runs_per_opp_proj_era_unit"] = {"slope": round(b, 4), "r2": round(r2, 4)}

    model = {}
    for pdcol, lbl in (("pd_sp_era_diff_home", "lambda_home"),
                       ("pd_sp_era_diff_away", "lambda_away"),
                       ("pd_sp_era_away_home", "lambda_home_opp_era"),
                       ("pd_sp_era_home_away", "lambda_away_opp_era"),
                       ("pd_sp_proj_era_away_home", "lambda_home_opp_proj"),
                       ("pd_sp_proj_era_home_away", "lambda_away_opp_proj")):
        if pdcol in oof.columns:
            v = oof[pdcol]
            v = v[v.notna()]
            model[lbl] = round(float(v.mean()), 4) if len(v) else None
        else:
            model[lbl] = None
    return {"empirical": emp, "sextiles": sext, "model_pd": model}


def sextile_spread_ratio(oof: pd.DataFrame, gl: pd.DataFrame) -> dict | None:
    """Structural compression per arm: sextile-mean spread of the model's
    lambda edge vs the actual margin spread on sp_era_diff sextiles.
    Ratio toward 1.0 = the model's edge tracks reality's spread."""
    d = oof[["game_pk", "home_expected_runs", "away_expected_runs",
             "home_score", "away_score"]].merge(
        gl[["game_pk", "sp_era_diff"]], on="game_pk", how="left")
    d = d.dropna(subset=["sp_era_diff"])
    d["margin"] = d["home_score"] - d["away_score"]
    d["ledge"] = d["home_expected_runs"] - d["away_expected_runs"]
    try:
        q = pd.qcut(d["sp_era_diff"], 6, duplicates="drop")
    except ValueError:
        return None
    grp = d.groupby(q, observed=True)
    act = grp["margin"].mean()
    mod = grp["ledge"].mean()
    act_spread = float(act.max() - act.min())
    mod_spread = float(mod.max() - mod.min())
    return {
        "actual_margin_sextile_spread": round(act_spread, 3),
        "model_ledge_sextile_spread": round(mod_spread, 3),
        "ratio": round(mod_spread / act_spread, 3) if act_spread else None,
    }


def season_sp_band_gaps(oof: pd.DataFrame, gl: pd.DataFrame) -> list[dict]:
    """Per-season derived-ML vs actual home win in the directional SP bands
    (home SP >= 1.5 better / away SP >= 1.5 better), pooled AND per
    season-partial — the Leg-2 "reality wobbles" discipline (a9cd6af):
    compare against each season's actuals, never a fixed 2-3pp gap.
    Uses the SAME C2 pricing as price_arm (k refit on pre-holdout OOF,
    alpha curves, NB MC) via probe_sp_arm_tables.arm_pwin."""
    from probe_sp_arm_tables import arm_pwin
    pwin, _ledge2, k = arm_pwin(oof)
    d = oof[["game_pk", "game_date", "home_score", "away_score"]].copy()
    d["pwin"] = pwin
    d["season"] = pd.to_datetime(d["game_date"]).dt.year.astype(str)
    d = d.merge(gl[["game_pk", "sp_era_diff"]], on="game_pk", how="left")
    d = d.dropna(subset=["sp_era_diff"])
    d["home_won"] = (d["home_score"] > d["away_score"]).astype(float)
    rows = []
    for season, g in d.groupby("season"):
        for lo, hi, lbl in ((-99, -1.5, "home SP better >= 1.5"),
                            (1.5, 99, "away SP better >= 1.5"),
                            (-1.5, 1.5, "even |diff| < 1.5")):
            sub = g[(g["sp_era_diff"] >= lo) & (g["sp_era_diff"] < hi)]
            if len(sub) >= 10:
                rows.append({
                    "season": season, "band": lbl, "n": int(len(sub)),
                    "actual_home_win": round(float(sub["home_won"].mean()), 4),
                    "derived_ml": round(float(sub["pwin"].mean()), 4),
                    "gap": round(float(sub["pwin"].mean()
                                        - sub["home_won"].mean()), 4),
                })
    rows.append({"season": "all", "k_fitted": round(float(k), 4)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", type=str, default="C0,C1,P1,P2,P3")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit-folds", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    games = load_game_features(data_path)
    decided = get_decided_frame(games)
    frame_sha = sha256_file(data_path)[:16]
    gl = pd.read_csv(data_path)
    gl["game_pk"] = gl["game_pk"].astype(str)

    # Projection composite columns (P-arms): z-stats + ERA-scale fit on the
    # PRE-HOLDOUT rows only (strictly prior to the sealed window).
    dates = pd.to_datetime(decided["game_date"])
    pre_mask = (dates < dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
    decided, proj_meta = build_projection_cols(decided, pre_mask)
    print(f"frame={frame_sha} decided={len(decided)} | proj meta "
          f"{proj_meta}", flush=True)
    # Attach projection columns to gl for the measurement merge.
    pm = decided[["game_pk", "sp_proj_era_home", "sp_proj_era_away"]].copy()
    pm["game_pk"] = pm["game_pk"].astype(str)
    gl = gl.merge(pm, on="game_pk", how="left")

    out = args.out or (DATA_DELIVERY_DIR
                       / f"mlb_sp_projection_arm_{frame_sha}.json")
    record = (json.loads(out.read_text()) if out.exists() else
              {"schema": "mlb-sp-projection-arm/v1", "frame": frame_sha,
               "frame_sha_source": "game_level_features.csv (sha256:16)",
               "date": DATE,
               "geometry": {"cadence_days": RETRAIN_CADENCE_DAYS,
                            "min_val_games": MIN_VAL_FOLD_GAMES,
                            "seed": RANDOM_SEED},
               "projection": {
                   "components": {"lower_better": PROJ_LO_BETTER,
                                   "higher_better": PROJ_HI_BETTER},
                   "min_components": MIN_PROJ_COMPONENTS,
                   "scale": "ERA-equivalent: 1 unit ~= 1 ERA point of "
                             "quality (higher = better); fit on pre-holdout",
                   "era_on_proj_slope": {k: v["era_on_proj_slope"]
                                          for k, v in proj_meta.items()},
                   "coverage_pre": {k: v["coverage_pre"]
                                     for k, v in proj_meta.items()},
                   "coverage_sealed": {k: v["coverage_sealed"]
                                        for k, v in proj_meta.items()},
                   "note": "No pitches.parquet locally; composite built from "
                           "the frame's PIT-safe Statcast-derived trailing "
                           "columns (FIP/xwOBA/WHIP/BB9/K9-5g/whiff-3g/velo-3g) "
                           "— the xFIP/SIERA-family fallback."},
               "arms": {}})

    oofs: dict[str, pd.DataFrame] = {}
    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        params, per_side = arm_params_and_frames(name, decided)
        print(f"\n=== arm {name} ===", flush=True)
        h = hashlib.sha256()
        h.update(frame_sha.encode())
        h.update(name.encode())
        h.update(json.dumps(sorted((per_side or {}).keys())).encode())
        key = h.hexdigest()[:16]
        cache = Path(tempfile.gettempdir()) / f"sp_sens_oof_{key}.parquet"
        if cache.exists() and not args.limit_folds:
            oof = pd.read_parquet(cache)
            print(f"  cache hit {cache.name} ({len(oof)} rows)", flush=True)
        else:
            oof = walk_arm(name, decided, params, per_side,
                           limit_folds=args.limit_folds)
            if not args.limit_folds:
                oof.to_parquet(cache)
            print(f"  walked {len(oof)} rows, "
                  f"{oof['fold_idx'].nunique()} folds", flush=True)
        oofs[name] = oof

        # C0 validation vs the shipped run-engine OOF artifact.
        if name == "C0":
            shipped = pd.read_csv(DATA_DELIVERY_DIR / f"run_engine_oof_{DATE}.csv")
            shipped["game_pk"] = shipped["game_pk"].astype(str)
            m = oof.merge(shipped[["game_pk", "home_expected_runs",
                                   "away_expected_runs"]],
                          on="game_pk", suffixes=("", "_ship"))
            m = m.dropna(subset=["home_expected_runs_ship"])
            if len(m):
                print(f"  C0-vs-shipped overlap {len(m)}: lambda MAE home "
                      f"{float(np.abs(m['home_expected_runs'] - m['home_expected_runs_ship']).mean()):.5f} "
                      f"away {float(np.abs(m['away_expected_runs'] - m['away_expected_runs_ship']).mean()):.5f}",
                      flush=True)
        if args.smoke:
            continue
        res = price_arm(oof, holdout_days=HOLDOUT_DAYS)
        res["n_oof_games"] = int(len(oof))
        res["n_folds"] = int(oof["fold_idx"].nunique())
        res["lambda_mean"] = {
            "home": round(float(oof["home_expected_runs"].mean()), 4),
            "away": round(float(oof["away_expected_runs"].mean()), 4),
            "edge_sd": round(float((oof["home_expected_runs"] -
                                    oof["away_expected_runs"]).std()), 4),
        }
        res["sextile_spread_ratio"] = sextile_spread_ratio(oof, gl)
        res["season_sp_band_gaps"] = season_sp_band_gaps(oof, gl)
        res["model_pd"] = sp_measurements(oof, gl)["model_pd"]
        record["arms"][name] = res
        out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"    sealed margin CRPS {res['margin_crps_sealed']} | "
              f"totals sealed ECE {res['totals']['metrics_sealed']['ece']} | "
              f"P(win) SD {res['derived_ml']['pwin_sd_sealed']} | "
              f"edge sd {res['lambda_mean']['edge_sd']} | "
              f"sextile ratio {res['sextile_spread_ratio']['ratio'] if res['sextile_spread_ratio'] else None}",
              flush=True)

    # Empirical targets + C0 model response (single measurement block).
    if "C0" in oofs and not args.smoke:
        meas = sp_measurements(oofs["C0"], gl)
        record["measurements"] = meas
        out.write_text(json.dumps(record, indent=2) + "\n")
        print("\n=== empirical vs model (C0 basis) ===")
        for k, v in meas["empirical"].items():
            print(f"  {k}: slope {v['slope']:+.3f} r2 {v['r2']:.3f}")
        for r in meas["sextiles"]:
            print(f"  {r['bin']}: sp {r['sp_era_diff_mean']:+.2f} | "
                  f"actual margin {r['actual_margin_mean']:+.3f} | "
                  f"model edge {r['model_ledge_mean']:+.3f} | "
                  f"home win {r['actual_home_win']:.3f}")
        print("  model PD:", meas["model_pd"])


if __name__ == "__main__":
    main()
