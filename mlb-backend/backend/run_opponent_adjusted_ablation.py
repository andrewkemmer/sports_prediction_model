"""Opponent-adjusted pitcher + team talent ablation — WITH vs WITHOUT on the
moneyline ensemble (gated measurement, option 3).

Measurement task (NOT tuning, NOT a silent feature ship): does removing
schedule-strength noise from the team/pitcher signal — which today is mostly
raw home-vs-away *_diff columns — help the ensemble out of sample? Nothing is
changed silently; the feature set ships only if it clears the sealed-holdout
gate, and the WITH arm measures only the genuinely, leak-safely computable
subset on the committed artifact.

Design (mirrors the locked run_form_delta_ablation.py conventions exactly):
- Data: committed data_delivery/game_level_features.csv (6,992 decided
  games; sha256 recorded). Decided games only; add_opponent_adjusted_features
  computes the opponent-adjusted ladders that are computable on this artifact.
- Honest-on-artifact caveat: both candidate families ARE reconstructable
  point-in-time on the committed CSV (verified: home_starter_id /
  away_starter_id are 100% present; team identity + scores are present for
  every prior game), so the WITH arm measures the full candidate set below.
  A candidate that could NOT be computed would be reported with coverage 0.0
  and dropped from WITH (constant-NaN columns add noise, never signal).
  Note: run_margin_diff is in FEATURE_COLS but is NOT in the raw CSV (it is
  produced at training time by walk_forward_evaluate._attach_oof_run_margins);
  in this harness it is all-NaN in BOTH arms (median-imputed constant), so it
  cannot differentiate the arms — the WITH-vs-WITHOUT delta stays clean.
- Candidates (both families, strictly point-in-time — a game may use only
  rows with game_date STRICTLY before its own date; same-day doubleheader
  legs are excluded from every ladder):

  Family 1 — opponent-adjusted team talent (SRS-style lap): per team, a
  trailing window of per-game run margins (chronologically trailing, decaying
  by recency via a fixed trailing window), expressed relative to the average
  trailing strength of the teams faced — a simple SRS/tiered-strength lap.
  Home and away sides, plus the home−away adjusted diff:
      home_team_talent_adj, away_team_talent_adj, team_talent_adj_diff
  (window=10 team games, min 5 prior to be non-NaN).

  Family 2 — opponent-adjusted starting-pitcher quality: per home/away
  starter, a chronologically trailing series of the runs allowed by his team
  in games he started (proxy: opponent team runs; the CSV has no
  per-pitcher runs split, documented), adjusted for the trailing raw
  strength of the lineups he faced (schedule-strength, one SRS lap); the
  moneyline arms get the pair:
      home_sp_adj, away_sp_adj, sp_adj_diff
  (window=8 starts, min 3 prior to be non-NaN). The pitcher ladder IS
  reconstructable: starter ids are 100% present and every prior game's
  opponent runs + opponent team talent are computable.

- Variants: WITHOUT = the exact production training.FEATURE_COLS baseline
  (asserted 59, incl. run_margin_diff); WITH = 59 + the opponent-adjusted
  columns with coverage > 0. Nothing in training.py / features.py /
  pipeline.py is edited — the gate swaps training.FEATURE_COLS at run time.
- Folds: walk_forward_splits on the tuning pool with RETRAIN_CADENCE_DAYS,
  filtered by MIN_VAL_FOLD_GAMES (same machinery as walk_forward_evaluate:
  declared vs executed recorded).
- Members: all 5 (xgb/lgbm/rf/logistic/mlp) + the static-prior blend
  (adaptive weights cleared before each variant so both variants blend
  identically).
- Metrics: compute_metrics (clip 1e-7) — logloss / AUC / Brier / ECE — raw
  and prequential-calibrated (fit_platt on prior folds' blend pairs only,
  exactly as walk_forward_evaluate does).
- Sealed 21-day holdout: refit fit-only on the whole tuning pool AFTER the
  fold loop; never touched during fold fitting.

Gate (task rule, identical to the template and the repo's documented prior
gates): CONCLUDE "adopt candidates into FEATURE_COLS" ONLY if WITH beats
WITHOUT on the sealed 21-day holdout on logloss AND AUC without degrading
ECE-cal. A pooled win with a holdout loss → DON'T ADOPT (the pooled-gain /
sealed-loss inversion this repo has hit repeatedly — see the margin, form
delta and home-edge records).

Emits data_delivery/opponent_adjusted_ablation_<sha>.json (incremental —
resumes by skipping variants already present). COMMITS NOTHING.

Usage:
    python run_opponent_adjusted_ablation.py
    python run_opponent_adjusted_ablation.py --variants WITHOUT,WITH
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
from calibration import apply_platt, fit_platt, MIN_OOF_FOR_FIT  # noqa: E402

EPS = 1e-7

# Candidate columns produced by add_opponent_adjusted_features.
OPP_ADJ_COLS = [
    "home_team_talent_adj",
    "away_team_talent_adj",
    "team_talent_adj_diff",
    "home_sp_adj",
    "away_sp_adj",
    "sp_adj_diff",
]
TEAM_WINDOW = 10   # trailing team games used for the talent ladder
TEAM_MIN = 5       # min prior team games before a ladder value is real
SP_WINDOW = 8      # trailing starts used for the pitcher ladder
SP_MIN = 3         # min prior starts before a ladder value is real


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


# ── Leak-safe trailing ladders (pure, testable in isolation) ───────────────

def trailing_team_ladders(side: pd.DataFrame,
                          window: int = TEAM_WINDOW,
                          min_games: int = TEAM_MIN) -> tuple[dict, dict]:
    """Pure: per-team trailing raw strength and opponent-adjusted talent.

    ``side`` must carry ``gidx``, ``date`` (datetime64), ``team``, ``opp``,
    ``margin``. ONLY rows with ``date`` STRICTLY before the current row's
    date contribute to that row's ladder (same-day games — doubleheader
    legs — are excluded, so nothing after first pitch can leak).

    Returns ``(raw_map, adj_map)`` keyed by ``(team, gidx)``:
      raw = mean of trailing run margins (window, min_games gate);
      adj = raw − mean of each prior opponent's OWN raw strength at the
            date that opponent was faced (one SRS/tiered-strength lap).
    A row with insufficient prior history gets NaN (never imputed).
    """
    side = side.sort_values(["date", "gidx"]).reset_index(drop=True)
    rows = side.to_dict("records")
    hist: dict[str, list] = {}
    raw: dict[tuple, float] = {}
    adj: dict[tuple, float] = {}
    # pass 1: raw strength (date-strict)
    for r in rows:
        t, d, gi = r["team"], r["date"], r["gidx"]
        prior = [h for h in hist.get(t, []) if h[0] < d]
        win = prior[-window:]
        raw[(t, gi)] = (float(np.mean([h[2] for h in win]))
                        if len(win) >= min_games else np.nan)
        hist.setdefault(t, []).append((d, gi, r["margin"], r["opp"]))
    # pass 2: opponent-adjusted lap (needs opponents' raw at their rows)
    hist2: dict[str, list] = {}
    for r in rows:
        t, d, gi = r["team"], r["date"], r["gidx"]
        prior = [h for h in hist2.get(t, []) if h[0] < d]
        win = prior[-window:]
        if len(win) >= min_games:
            opp_vals = np.asarray([raw.get((h[3], h[1]), np.nan)
                                   for h in win], dtype=float)
            if np.isfinite(opp_vals).sum() >= min_games:
                adj[(t, gi)] = float(np.mean([h[2] for h in win])
                                     - np.nanmean(opp_vals))
            else:
                adj[(t, gi)] = np.nan
        else:
            adj[(t, gi)] = np.nan
        hist2.setdefault(t, []).append((d, gi, r["margin"], r["opp"]))
    return raw, adj


def trailing_pitcher_ladders(side: pd.DataFrame,
                             team_raw: dict,
                             window: int = SP_WINDOW,
                             min_games: int = SP_MIN) -> dict:
    """Pure: per-starter trailing quality adjusted for opponents faced.

    ``side`` must additionally carry ``starter`` and ``opp_runs`` (the runs
    the starter's team allowed in that game — proxy for starter quality; the
    CSV has no per-pitcher runs split, so this is the documented proxy).
    Only starts with ``date`` STRICTLY before the current row's date count.
    ``team_raw`` is the (team, gidx) → trailing raw-strength map from
    trailing_team_ladders: the quality of each lineup faced at that date
    (schedule-strength, one SRS lap).

    Returns ``{(starter, gidx): adj}`` where
      adj = mean(trailing opp_runs) − mean(opponent raw strength at those
            prior starts); NaN with insufficient prior starts (never imputed).
    """
    hist: dict[str, list] = {}
    adj: dict[tuple, float] = {}
    for r in side.sort_values(["date", "gidx"]).to_dict("records"):
        s = r["starter"]
        try:
            if pd.isna(s):
                continue
        except (TypeError, ValueError):
            pass
        d, gi = r["date"], r["gidx"]
        prior = [h for h in hist.get(s, []) if h[0] < d]
        win = prior[-window:]
        if len(win) >= min_games:
            opp_strength = np.asarray([team_raw.get((h[2], h[1]), np.nan)
                                       for h in win], dtype=float)
            if np.isfinite(opp_strength).sum() >= min_games:
                adj[(s, gi)] = float(np.mean([h[3] for h in win])
                                     - np.nanmean(opp_strength))
            else:
                adj[(s, gi)] = np.nan
        else:
            adj[(s, gi)] = np.nan
        hist.setdefault(s, []).append((d, gi, r["opp"], r["opp_runs"]))
    return adj


def add_opponent_adjusted_features(
        games: pd.DataFrame,
        team_window: int = TEAM_WINDOW,
        team_min: int = TEAM_MIN,
        sp_window: int = SP_WINDOW,
        sp_min: int = SP_MIN) -> pd.DataFrame:
    """Add the six opponent-adjusted columns to a game-level frame.

    Leak-safe by construction: every ladder value for a game uses only rows
    with game_date strictly before that game's date. Rows without enough
    prior history keep NaN (honest coverage; never imputed here).
    """
    df = games.copy()
    n = len(df)
    dates = pd.to_datetime(df["game_date"]).values
    home = pd.DataFrame({
        "gidx": np.arange(n),
        "date": dates,
        "team": df["home_team"].values,
        "opp": df["away_team"].values,
        "margin": (df["home_score"] - df["away_score"]).values.astype(float),
        "starter": df["home_starter_id"].values,
        "opp_runs": df["away_score"].values.astype(float),
    })
    away = pd.DataFrame({
        "gidx": np.arange(n),
        "date": dates,
        "team": df["away_team"].values,
        "opp": df["home_team"].values,
        "margin": (df["away_score"] - df["home_score"]).values.astype(float),
        "starter": df["away_starter_id"].values,
        "opp_runs": df["home_score"].values.astype(float),
    })
    side = pd.concat([home, away], ignore_index=True)
    raw_map, team_adj = trailing_team_ladders(side, team_window, team_min)
    sp_adj = trailing_pitcher_ladders(side, raw_map, sp_window, sp_min)

    home_teams = df["home_team"].tolist()
    away_teams = df["away_team"].tolist()
    home_starters = df["home_starter_id"].tolist()
    away_starters = df["away_starter_id"].tolist()
    idx = list(range(n))
    df["home_team_talent_adj"] = [team_adj.get((t, i), np.nan)
                                  for i, t in zip(idx, home_teams)]
    df["away_team_talent_adj"] = [team_adj.get((t, i), np.nan)
                                  for i, t in zip(idx, away_teams)]
    df["team_talent_adj_diff"] = (df["home_team_talent_adj"]
                                  - df["away_team_talent_adj"])
    df["home_sp_adj"] = [sp_adj.get((s, i), np.nan)
                         for i, s in zip(idx, home_starters)]
    df["away_sp_adj"] = [sp_adj.get((s, i), np.nan)
                         for i, s in zip(idx, away_starters)]
    df["sp_adj_diff"] = df["home_sp_adj"] - df["away_sp_adj"]
    return df


def coverage_report(games: pd.DataFrame) -> list[dict]:
    out = []
    for c in OPP_ADJ_COLS:
        if c not in games.columns:
            out.append({"column": c, "present": False, "coverage": 0.0})
            continue
        cov = float(games[c].notna().mean())
        out.append({"column": c, "present": True, "coverage": round(cov, 4)})
    return out


def build_variants(games: pd.DataFrame,
                   coverage: list[dict]) -> dict[str, list[str]]:
    """WITHOUT = exact production FEATURE_COLS (59, incl. run_margin_diff);
    WITH = 59 + the opponent-adjusted columns with real coverage (> 0)."""
    base = list(training.FEATURE_COLS)
    assert len(base) == 59, (
        f"expected 59 production FEATURE_COLS, got {len(base)} — "
        f"sync this harness with training.py before measuring")
    cov_by_col = {c["column"]: c["coverage"] for c in coverage}
    computable = [c for c in OPP_ADJ_COLS if cov_by_col.get(c, 0.0) > 0]
    return {
        "WITHOUT": base,
        "WITH": base + computable,
    }


def run_variant(cols: list[str], folds, tune_df, hold_df) -> dict:
    training.FEATURE_COLS = list(cols)
    training._LAST_ADAPTIVE_WEIGHTS.clear()  # both variants blend identically

    oof_y: list[float] = []
    oof_blend: list[float] = []
    oof_members: dict[str, list[float]] = {}
    oof_blend_cal: list[float] = []
    oof_members_cal: dict[str, list[float]] = {}
    executed = 0

    for split in folds:
        train = split["train_games"]
        val = split["val_games"]
        try:
            models, _ = training.train_moneyline_ensemble(train, val)
        except Exception as e:  # keep the loop honest: log, skip, continue
            print(f"  fold {split['fold_idx']} failed: {e}")
            continue
        blend, member_probs, _wts = training.ensemble_predict(models, val)
        y_val = val["home_win"].values.astype(float)
        fold_cal = None
        if len(oof_blend) >= MIN_OOF_FOR_FIT:
            fold_cal = fit_platt(np.asarray(oof_y), np.asarray(oof_blend))
        oof_y.extend(y_val.tolist())
        oof_blend.extend(np.asarray(blend, dtype=float).tolist())
        oof_blend_cal.extend(
            np.asarray(apply_platt(np.asarray(blend), fold_cal), dtype=float).tolist())
        for name, p in member_probs.items():
            pa = np.asarray(p, dtype=float)
            oof_members.setdefault(name, []).extend(pa.tolist())
            oof_members_cal.setdefault(name, []).extend(
                np.asarray(apply_platt(pa, fold_cal), dtype=float).tolist())
        executed += 1

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
    holdout: dict[str, dict] = {
        "blend": training.compute_metrics(y_hold, np.asarray(blend_hold)),
    }
    for name, p in member_hold.items():
        holdout[name] = training.compute_metrics(
            y_hold, np.asarray(p, dtype=float))

    return {"n_cols": len(cols), "folds_executed": executed,
            "pooled": pooled, "holdout": holdout}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", type=str, default="WITHOUT,WITH")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--holdout-days", type=int, default=21)
    args = ap.parse_args()

    sha = head_sha()
    data_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    data_hash = sha256_file(data_path)

    games = pd.read_csv(data_path)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_win"]).reset_index(drop=True)
    games = add_opponent_adjusted_features(games)

    cutoff = games["game_date"].max() - pd.Timedelta(days=args.holdout_days - 1)
    tune_df = games[games["game_date"] < cutoff].reset_index(drop=True)
    hold_df = games[games["game_date"] >= cutoff].reset_index(drop=True)
    all_splits = training.walk_forward_splits(
        tune_df, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
    folds = [s for s in all_splits if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]

    print(f"commit={sha[:12]} data_sha={data_hash[:12]} games={len(games)} "
          f"tuning={len(tune_df)} holdout={len(hold_df)} "
          f"folds={len(all_splits)}/{len(folds)} seed={RANDOM_SEED} clip={EPS}")

    coverage = coverage_report(games)
    real = [c for c in coverage if c["coverage"] > 0]
    dropped = [c for c in coverage if c["coverage"] == 0.0]
    print(f"opponent-adjusted coverage on committed CSV: "
          f"{len(real)}/{len(OPP_ADJ_COLS)} real")
    for c in coverage:
        print(f"    {c['column']:24s} coverage={c['coverage']:.3f}")
    if dropped:
        print(f"  dropping {len(dropped)} incomputable candidate(s) "
              f"(coverage 0.0, would be constant-NaN noise): "
              f"{', '.join(c['column'] for c in dropped)}")

    variants = build_variants(games, coverage)
    out = args.out or (DATA_DELIVERY_DIR
                       / f"opponent_adjusted_ablation_{sha[:12]}.json")
    if out.exists():
        results = json.loads(out.read_text())
    else:
        results = {"schema": "opponent-adjusted-ablation/v1",
                   "commit_sha": sha, "data_sha256": data_hash,
                   "holdout_days": args.holdout_days,
                   "folds_declared": len(all_splits),
                   "folds_executed": len(folds), "clip_eps": EPS,
                   "seed": int(RANDOM_SEED),
                   "ladder_params": {"team_window": TEAM_WINDOW,
                                     "team_min_games": TEAM_MIN,
                                     "sp_window": SP_WINDOW,
                                     "sp_min_games": SP_MIN},
                   "coverage": coverage, "variants": {}}
    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    for name in want:
        if name in results["variants"]:
            print(f"  {name}: cached, skipping")
            continue
        print(f"  {name}: running ({len(variants[name])} cols) ...")
        r = run_variant(variants[name], folds, tune_df, hold_df)
        r["cols"] = variants[name]
        results["variants"][name] = r
        out.write_text(json.dumps(results, indent=2) + "\n")
        b = r["pooled"]["blend"]
        h = r["holdout"]["blend"]
        print(f"    pooled blend {b['logloss']:.4f}/{b['auc']:.4f} "
              f"brier {b['brier']:.4f} ece {b['ece']:.4f} | "
              f"holdout {h['logloss']:.4f}/{h['auc']:.4f}")

    # gate: WITH must beat WITHOUT on the sealed holdout (both metrics)
    if "WITHOUT" in results["variants"] and "WITH" in results["variants"]:
        wo = results["variants"]["WITHOUT"]["holdout"]["blend"]
        w = results["variants"]["WITH"]["holdout"]["blend"]
        win = w["logloss"] < wo["logloss"] and w["auc"] > wo["auc"]
        pw = (results["variants"]["WITH"]["pooled"]["blend"]["logloss"]
              < results["variants"]["WITHOUT"]["pooled"]["blend"]["logloss"])
        print(f"\n=== sealed-holdout gate (blend) ===")
        print(f"  WITHOUT {wo['logloss']:.4f}/{wo['auc']:.4f}  "
              f"WITH {w['logloss']:.4f}/{w['auc']:.4f}  -> "
              f"{'BEATS WITHOUT' if win else 'loses/ties WITHOUT'} "
              f"(pooled_win={pw})")
        if pw and not win:
            print("  FLAG: pooled win with holdout loss — likely overfit, not adopted.")
    print(f"\nablation written: {out}")


if __name__ == "__main__":
    main()
