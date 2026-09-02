"""Run-line DEFENSIVE-feature expansion ablation — re-test the four
defensive feature families on the run engine's corrected C2 base.

Reconstruction of the cloud-only harness (the original lived only on the
Kaggle checkout and was lost; this is the same protocol run against the
LOCAL committed caches — no Statcast repull):

  cache   = data_delivery/pbp_defense_20260901.parquet (committed, zstd)
  games   = data_delivery/game_level_features.csv (committed)

Families (identical to the moneyline defense ablation v3, ablation_defense.py,
so the run-line verdicts are comparable to that record):
  F1 TEAM FORM DEFENSE   -- team_runs_allowed_10g/30g (per side)
  F2 BATTED-BALL ALLOWED -- opp_exitvelo/barrel_pct/gb_pct/hardhit_pct/ld_pct
                           15g+30g (per side)
  F3 DEFENSE TREND       -- short-minus-long of the F1/F2 cores (per side)
  F4 POSITION-SPLIT      -- def_if_30g / def_of_30g / def_catcher_30g
                           (per side)
F5 STARTER-CONDITIONED is EXCLUDED: its coverage measured 61.6%, below the
95% family-coverage floor (documented in the 2026-09-01 run-line defense
record). PIT: every ladder uses only pbp rows with game_date < the target
game's date (1-day publication lag respected by construction).

Arms, same geometry as the production run-engine walk-forward
(RETRAIN_CADENCE_DAYS=7, MIN_VAL_FOLD_GAMES=40, RANDOM_SEED=42, decided
frame ~7,018, sealed-21d holdout):
    C0    = current 53-feature C2 base (baseline)
    +F1..+F4 = C0 + one family each (per-side columns only; trees derive
               the diffs implicitly, and the run engine's side-view split
               routes *_home/*_away columns to the correct side).

C2 layer ACTIVE in every arm: per arm, k is REFIT on that arm's pre-holdout
OOF only, the lambda pair is expanded with apply_k_edge, and the alpha(l)
curves are fit on the arm's EXPANDED pre-holdout lambdas (the
derive_markets_v3 path) before NB-MC pricing and CRPS. Sealed never sees k.

Per arm (pooled + sealed):
  MARGIN (primary): CRPS on the full margin distribution; run-line -1.5
  cover calibration in p-deciles + |edge| >= 0.5/0.70/0.90 bins.
  TOTALS: O/U calibration by assigned line + ECE + CRPS on the sum.
  DERIVED ML: calibration + P(win) SD (target 0.066) + [0.55,0.60) gap.

Gate (task discipline): a family ADOPTs iff it wins C0 on sealed CRPS AND
per-line calibration (extreme-bin |delta| closes within noise) AND totals
stay flat within tolerance AND pooled corroborates.

Record: data_delivery/mlb_runline_defense_<frame>.json (frame = data hash),
written after each arm (resumable). Per-arm OOF cached under
/tmp/runline_def_oof_<key>.parquet. COMMITS NOTHING.

Usage:
    python run_mlb_runline_defense_ablation.py --arms C0,F1,F2,F3,F4
    python run_mlb_runline_defense_ablation.py --smoke   # 12 folds, /tmp
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
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))
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
from run_engine import derive_run_features, run_oof  # noqa: E402
from training import FEATURE_COLS  # noqa: E402

# Reuse the exact C2 pricing layer + gate helpers from the feature-expansion
# harness (identical surfaces: margin CRPS, run-line -1.5 cover, totals,
# derived-ML P(win) SD).
from run_mlb_runline_expansion_ablation import (  # noqa: E402
    ECE_TOL,
    PWIN_BUCKET,
    PWIN_SD_TARGET,
    _edge_gap_close,
    head_sha,
    price_arm,
    sha256_file,
)
# Same family definitions as the moneyline defense ablation v3.
from ablation_defense import (  # noqa: E402
    F1_PAIRS,
    F2_PAIRS,
    F3_PAIRS,
    F4_PAIRS,
    build_f1_f3_f5,
    build_f2_f4,
)

F5_EXCLUDED_REASON = (
    "starter-conditioned defense coverage measured 61.6% on the 2026-09-01 "
    "frame — below the 95% family-coverage floor; excluded from this run "
    "(matches the moneyline defense v3 record, def F5)."
)

ARM_LABELS = {
    "F1": [f"{b}_{s}" for b in ("team_runs_allowed_10g", "team_runs_allowed_30g")
           for s in ("home", "away")],
    "F2": sorted({f"{b}_{s}" for b, _ in F2_PAIRS for s in ("home", "away")}),
    "F3": sorted({f"trend_{short}_{s}" for short, _ in F3_PAIRS
                  for s in ("home", "away")}),
    "F4": [f"{b}_{s}" for b, _ in F4_PAIRS for s in ("home", "away")],
}

FAMILY_DESCRIPTIONS = {
    "F1": "team_runs_allowed_10g/30g (team form defense)",
    "F2": "opp batted-ball 15g/30g (exit velo, barrel%, GB%, HH%, LD%)",
    "F3": "defense trend (15g minus 30g of the F1/F2 cores)",
    "F4": "position-split (IF/OF/catcher), rolling 30g",
}


def build_defense_ladders(decided: pd.DataFrame, wide: pd.DataFrame
                          ) -> pd.DataFrame:
    """Attach the per-game defensive ladder columns onto the decided frame.

    Source is ALWAYS the wide cache when present: it starts 2024-03-20 (the
    training window start), so early folds have defense features exactly as
    the baseline covers them. The 1-day publication lag is respected by
    construction (all ladders use game_date < target date only).
    """
    out = decided.copy()  # carry ALL decided columns (run_oof needs home_win etc.)
    f135 = build_f1_f3_f5(wide, decided)
    f24 = build_f2_f4(wide, decided) if wide is not None else None

    # F1 + F3 short/long bases from the f135 ladder.
    f1_cols = [c for c in ARM_LABELS["F1"] if c not in out.columns]
    f1_src = f135[["game_pk"] + f1_cols]
    out = out.merge(f1_src.drop_duplicates("game_pk"), on="game_pk", how="left")

    # F2 + F4 bases from the wide ladder.
    f2_cols = [c for c in ARM_LABELS["F2"] if f24 is not None and c in f24.columns
               and c not in out.columns]
    f4_cols = [c for c in ARM_LABELS["F4"] if f24 is not None and c in f24.columns
               and c not in out.columns]
    if f24 is not None and (f2_cols or f4_cols):
        out = out.merge(f24[["game_pk"] + f2_cols + f4_cols].drop_duplicates("game_pk"),
                        on="game_pk", how="left")

    # F3 trends = short minus long, per side (F1 cores from f135, F2 cores
    # from f24). NaN propagates when either window is missing — correct.
    for short, long in F3_PAIRS:
        for s in ("home", "away"):
            sc, lc = f"{short}_{s}", f"{long}_{s}"
            if sc in out.columns and lc in out.columns:
                out[f"trend_{sc}"] = out[sc] - out[lc]
    return out


def arm_features() -> dict[str, list[str]]:
    """Arm feature lists: C0 = the production 53-feature run view;
    F1..F4 = C0 + that family's per-side columns."""
    kept, _dropped = derive_run_features(list(FEATURE_COLS))
    assert len(kept) == 53, f"production run view must be 53, got {len(kept)}"
    arms: dict[str, list[str]] = {"C0": list(kept)}
    for tag in ("F1", "F2", "F3", "F4"):
        arms[tag] = list(kept) + ARM_LABELS[tag]
    return arms


def arm_drop_terms(arm_feats: list[str]) -> list[str]:
    """The 'dropped' list run_oof expects: every FEATURE_COLS col not in the
    arm (informational only; the derive is skipped when run_features is
    given)."""
    return [c for c in FEATURE_COLS if c not in arm_feats]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", type=str, default="C0,F1,F2,F3,F4")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--holdout-days", type=int, default=21)
    ap.add_argument("--limit-folds", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="12 folds, /tmp output, gate skipped")
    args = ap.parse_args()
    if args.smoke:
        args.limit_folds = min(args.limit_folds or 12, 12)
        args.out = Path(tempfile.gettempdir()) / "mlb_runline_defense_smoke.json"
        args.arms = "C0,F1"

    sha = head_sha()
    data_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    games = load_game_features(data_path)
    decided = get_decided_frame(games)
    frame_sha = sha256_file(data_path)
    frame = frame_sha[:16]
    print(f"commit={sha[:12]} frame={frame} decided_games={len(decided)} "
          f"holdout_days={args.holdout_days}")

    wide_path = sorted(DATA_DELIVERY_DIR.glob("pbp_defense_*.parquet"))
    if not wide_path:
        raise SystemExit("pbp_defense_*.parquet cache MISSING — cannot build "
                         "F2/F4 (F1/F3 would also lose 2024 coverage)")
    wide = pd.read_parquet(wide_path[-1])
    wide["game_date"] = pd.to_datetime(wide["game_date"]).dt.normalize()
    print(f"wide defense cache: {wide_path[-1].name} rows={len(wide)} "
          f"range={wide['game_date'].min().date()}..{wide['game_date'].max().date()}")

    print("building PIT ladders (game_date < target) ...", flush=True)
    ladder_cache = (Path(tempfile.gettempdir())
                    / f"decided_def_{frame}.parquet")
    if ladder_cache.exists():
        decided_def = pd.read_parquet(ladder_cache)
        print(f"  ladder cache hit ({len(decided_def)} rows)", flush=True)
    else:
        decided_def = build_defense_ladders(decided, wide)
        decided_def.to_parquet(ladder_cache)
        print(f"  ladders built + cached ({len(decided_def)} rows)",
              flush=True)

    # Per-family coverage on the built frame (same floor rule as v3).
    coverage = {}
    for tag in ("F1", "F2", "F3", "F4"):
        cols = [c for c in ARM_LABELS[tag] if c in decided_def.columns]
        if cols:
            coverage[tag] = round(float(decided_def[cols].notna().all(axis=1).mean()), 4)
        else:
            coverage[tag] = 0.0
    print("family coverage:", coverage, flush=True)
    print(f"F5 excluded: {F5_EXCLUDED_REASON}", flush=True)

    feats = arm_features()
    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in want:
        if a not in feats:
            raise SystemExit(f"unknown arm {a!r}; known: {sorted(feats)}")

    out = args.out or (DATA_DELIVERY_DIR / f"mlb_runline_defense_{frame}.json")
    if out.exists():
        record = json.loads(out.read_text())
    else:
        record = {"schema": "mlb-runline-defense/v1",
                  "commit_sha": sha, "frame": frame,
                  "data_sha256": frame_sha,
                  "holdout_days": args.holdout_days,
                  "seed": int(RANDOM_SEED),
                  "n_decided": int(len(decided)),
                  "families": {"F1": "team_runs_allowed_10g/30g per side",
                               "F2": "opp batted-ball 15g/30g per side",
                               "F3": "trend (15g-30g) of F1/F2 cores per side",
                               "F4": "position-split IF/OF/catcher 30g per side"},
                  "F5": {"excluded": True, "reason": F5_EXCLUDED_REASON},
                  "family_coverage": coverage,
                  "pit_rule": "game_date < target game date (1-day publication lag)",
                  "cache": wide_path[-1].name,
                  "c2": "ACTIVE in every arm (per-run k refit on the arm's "
                        "pre-holdout OOF; alpha curves fit on expanded "
                        "pre-holdout lambdas)",
                  "arms": {}}
        out.write_text(json.dumps(record, indent=2) + "\n")

    # Re-surface coverage in the record when resuming (fresh metadata).
    record["family_coverage"] = coverage
    record.setdefault("F5", {}).update({"excluded": True,
                                        "reason": F5_EXCLUDED_REASON})

    for name in want:
        if name in record["arms"]:
            print(f"  arm {name} already recorded — skipping")
            continue
        arm_feats = feats[name]
        added = sorted(set(arm_feats) - set(feats["C0"]))
        print(f"\n  {name}: {len(arm_feats)} features (base + {added}) — "
              f"run-engine walk-forward + C2 pricing ...", flush=True)

        h = hashlib.sha256()
        h.update(frame.encode())
        h.update(json.dumps(arm_feats).encode())
        key = h.hexdigest()[:16]
        cache = (Path(tempfile.gettempdir()) / f"runline_def_oof_{key}.parquet")
        if cache.exists():
            oof = pd.read_parquet(cache)
        else:
            oof = run_oof(decided_def, run_features=arm_feats,
                          dropped=arm_drop_terms(arm_feats),
                          retrain_cadence_days=RETRAIN_CADENCE_DAYS,
                          min_val_games=MIN_VAL_FOLD_GAMES)["oof"]
            oof["fold_idx"] = oof["fold_idx"].astype(int)
            oof.to_parquet(cache)  # ALWAYS the FULL walk-forward OOF
        if args.smoke:
            oof = oof.tail(min(len(oof), args.limit_folds * 90))
        res = price_arm(oof, holdout_days=args.holdout_days)
        res["n_folds_oof"] = int(oof["fold_idx"].nunique())
        res["n_oof_games"] = int(len(oof))
        res["features_added"] = added
        res["family_coverage"] = coverage.get(name, None)

        record["arms"][name] = res
        out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"    folds={res['n_folds_oof']} games={res['n_oof_games']} "
              f"| sealed margin CRPS {res['margin_crps_sealed']} "
              f"(pooled {res['margin_crps_pooled']}) | totals CRPS "
              f"{res['totals']['crps_sealed']} | P(win) SD "
              f"{res['derived_ml']['pwin_sd_sealed']} | k="
              f"{res['k']['k_fitted_run']}", flush=True)

    # ---- gate: per-family verdict vs C0 (mirrors expansion protocol) ----
    if not args.smoke and "C0" in record["arms"]:
        c0 = record["arms"]["C0"]
        adopt, reject = [], []
        for name in ("F1", "F2", "F3", "F4"):
            if name not in record["arms"]:
                continue
            a = record["arms"][name]
            scaled_win = (a["margin_crps_sealed"] is not None and
                          c0["margin_crps_sealed"] is not None and
                          a["margin_crps_sealed"] < c0["margin_crps_sealed"])
            pooled_corr = (a["margin_crps_pooled"] is not None and
                           c0["margin_crps_pooled"] is not None and
                           a["margin_crps_pooled"] < c0["margin_crps_pooled"])
            line_ok = _edge_gap_close(a, "pooled") and \
                _edge_gap_close(a, "sealed") is not False
            totals_ok = (abs(a["totals"]["metrics_pooled"]["ece"]
                             - c0["totals"]["metrics_pooled"]["ece"])
                         <= ECE_TOL)
            ml_ok = a["derived_ml"]["pwin_sd_sealed"] >= \
                c0["derived_ml"]["pwin_sd_sealed"]
            verdict = bool(scaled_win and pooled_corr and line_ok and
                           totals_ok and ml_ok)
            (adopt if verdict else reject).append(name)
            record.setdefault("gate", {})[name] = {
                "family": FAMILY_DESCRIPTIONS[name],
                "verdict": "ADOPT" if verdict else "DON'T ADOPT",
                "sealed_crps_delta": round(
                    (a["margin_crps_sealed"] - c0["margin_crps_sealed"])
                    if None not in (a["margin_crps_sealed"],
                                    c0["margin_crps_sealed"]) else None, 5),
                "pooled_crps_delta": round(
                    (a["margin_crps_pooled"] - c0["margin_crps_pooled"])
                    if None not in (a["margin_crps_pooled"],
                                    c0["margin_crps_pooled"]) else None, 5),
                "sealed_win": bool(scaled_win),
                "pooled_corroborates": bool(pooled_corr),
                "line_extreme_ok": bool(line_ok),
                "totals_ece_delta": round(
                    a["totals"]["metrics_pooled"]["ece"]
                    - c0["totals"]["metrics_pooled"]["ece"], 5),
                "totals_ok": bool(totals_ok),
                "derived_ml_pwin_sd": a["derived_ml"]["pwin_sd_sealed"],
                "derived_ml_ok": bool(ml_ok),
                "coverage": coverage.get(name),
            }
        record.setdefault("gate", {})["summary"] = {
            "adopted": adopt,
            "rejected": reject,
            "rule": ("ADOPT iff wins sealed CRPS AND pooled corroborates AND "
                     "run-line extreme-edge bins close to within noise AND "
                     "totals ECE flat within 0.005 AND derived-ML P(win) SD "
                     "not regressed; pooled-only wins never adopt"),
            "vs_moneyline_v3": ("Families identical to ablation_defense v3 "
                                "(F1 team form, F2 batted-ball, F3 trend, "
                                "F4 position-split; F5 excluded <95% floor)")}
        out.write_text(json.dumps(record, indent=2) + "\n")
        print("\n================= GATE =================")
        print("adopted:", adopt or "none", "| rejected:", reject)
        for name, g in record["gate"].items():
            if isinstance(g, dict) and "family" in g:
                print(f"  {name} {FAMILY_DESCRIPTIONS[name]}: {g['verdict']} "
                      f"(sealed ΔCRPS {g['sealed_crps_delta']}, "
                      f"pooled ΔCRPS {g['pooled_crps_delta']})")


if __name__ == "__main__":
    main()