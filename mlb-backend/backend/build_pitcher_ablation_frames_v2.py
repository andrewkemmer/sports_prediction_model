"""build_pitcher_ablation_frames_v2.py — production-exact ablation frames.

v2 fix: the v1 frames were rebuilt ENTIRELY from the pitches parquet, which
silently dropped 24 non-pitcher columns (raw per-side elo / win_pct / woba,
standings-derived cols) and 90 games — those features come from team /
standing / weather sources, not pitch-by-pitch. The ensemble gate then ran
on a degraded frame (6/59 FEATURE_COLS NULL-filled, 100% of elo_diff /
win_pct_diff / woba_30g_diff NaN).

This builder keeps the PRODUCTION frame (data_delivery/game_level_features.csv,
179 cols, 7,006 games — every non-pitcher feature production-exact) and
splices in ONLY the 70 pitcher-stat columns (sp_* / bullpen* / *_sp_* /
pitcher_regression_indicator / ace_efficiency_factor) recomputed from the
same pitches parquet under CURRENT vs CORRECTED outs semantics.

Result: two frames identical to production except in the 70 pitcher-stat
columns, which differ between the arms ONLY by the outs-map fix — the true
apples-to-apples CURRENT vs CORRECTED gate input.

Games present in production but with no pitch coverage in the pull get NaN
pitcher stats in BOTH arms symmetrically (count reported loudly) — trees
route NaN and logistic/MLP impute, the standard production path.

Usage:
    python build_pitcher_ablation_frames_v2.py --pitches C:/tmp/pitches_full/pitches.parquet \
        --out-dir ../data_delivery

Writes:
    game_level_features_corrected.csv        (CORRECTED variant)
    game_level_features_raw_current.csv      (CURRENT variant, spliced)
    pitcher_ablation_frames_<sha>.json       (splice/diff evidence)

COMMITS NOTHING; production features.py semantics untouched (flag off).
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

from config import DATA_DELIVERY_DIR  # noqa: E402


def _stub_resource() -> None:
    import types
    if "resource" not in sys.modules:
        res = types.ModuleType("resource")
        res.RUSAGE_SELF = 0
        res.getrusage = lambda who: type("RU", (), {"ru_maxrss": 0})()
        sys.modules["resource"] = res


def head_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=_BACKEND_DIR.parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


def pitcher_cols_of(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if (c.startswith("sp_") or c.startswith("bullpen")
                or "_sp_" in c
                or c in ("pitcher_regression_indicator",
                         "ace_efficiency_factor"))]


def splice(prod: pd.DataFrame, recomputed: pd.DataFrame,
           pit_cols: list[str], label: str) -> tuple[pd.DataFrame, int]:
    """Replace prod's pitcher columns with recomputed values by game_pk.

    Returns (spliced frame, n_games_without_pitch_coverage)."""
    out = prod.copy()
    key = "game_pk"
    assert key in out.columns and key in recomputed.columns
    rec = recomputed[[key] + pit_cols].copy()
    rec = rec.drop_duplicates(subset=[key])
    n_cover = rec[key].nunique()
    merged = out.drop(columns=pit_cols).merge(rec, on=key, how="left")
    # restore column order: pitcher cols back at their original positions
    order = [c for c in out.columns if c not in pit_cols]
    for i, c in enumerate(out.columns):
        if c in pit_cols:
            order.insert(i, c)
    merged = merged[order]
    uncovered = int(merged[pit_cols[0]].isna().sum())
    print(f"[{label}] spliced {len(pit_cols)} pitcher cols "
          f"({n_cover} games covered, {uncovered} games NaN pitcher stats "
          f"in BOTH arms)", flush=True)
    return merged, uncovered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitches", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    _stub_resource()
    from features import build_features

    out_dir = args.out_dir or DATA_DELIVERY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cur_csv = out_dir / "game_level_features_raw_current.csv"
    cor_csv = out_dir / "game_level_features_corrected.csv"
    prod_csv = DATA_DELIVERY_DIR / "game_level_features.csv"
    tmp = Path(tempfile.mkdtemp(prefix="pitchersplice_"))

    prod = pd.read_csv(prod_csv)

    # Filter R/P (regular + postseason) like production; the unmetered
    # Savant export includes spring (S) and winter-league (L/F/W/D) rows.
    pitches = pd.read_parquet(args.pitches)
    if "game_type" in pitches.columns:
        n0 = len(pitches)
        pitches = pitches[pitches["game_type"].isin(["R", "P"])]
        if len(pitches) != n0:
            print(f"[frame] filtered {n0 - len(pitches):,} non-R/P pitches",
                  flush=True)
            clean = tmp / "pitches_regular.parquet"
            pitches.to_parquet(clean, index=False)
            pitches_path = clean
        else:
            pitches_path = args.pitches
    else:
        pitches_path = args.pitches

    print("Building CURRENT recompute (corrected_outs=False) ...", flush=True)
    df_cur, _ = build_features(pitches_path, tmp / "cur", corrected_outs=False)
    print("Building CORRECTED recompute (corrected_outs=True) ...", flush=True)
    df_cor, _ = build_features(pitches_path, tmp / "cor", corrected_outs=True)

    pit_cols = pitcher_cols_of(df_cur)
    assert pitcher_cols_of(df_cor) == pit_cols, "pitcher col sets diverge"
    missing_in_prod = [c for c in pit_cols if c not in prod.columns]
    assert not missing_in_prod, f"pitcher cols absent from prod: {missing_in_prod}"

    spl_cur, uncov_cur = splice(prod, df_cur, pit_cols, "CURRENT")
    spl_cor, uncov_cor = splice(prod, df_cor, pit_cols, "CORRECTED")
    assert uncov_cur == uncov_cor, "asymmetric pitch coverage between arms"

    # ── audit: only the pitcher cols differ between the two arms ──────────
    numeric = [c for c in spl_cur.columns
               if pd.api.types.is_numeric_dtype(spl_cur[c])]
    differ = []
    for c in numeric:
        if c in pit_cols:
            continue
        a = pd.to_numeric(spl_cur[c], errors="coerce").astype(float).fillna(0.0)
        b = pd.to_numeric(spl_cor[c], errors="coerce").astype(float).fillna(0.0)
        if (a - b).abs().max() > 1e-9:
            differ.append(c)
    assert not differ, f"non-pitcher columns differ between arms: {differ}"
    # ── audit: non-pitcher cols of the CURRENT arm equal production ───────
    nprod = [c for c in numeric if c not in pit_cols]
    drift = []
    for c in nprod:
        a = pd.to_numeric(spl_cur[c], errors="coerce").astype(float).fillna(0.0)
        b = pd.to_numeric(prod[c], errors="coerce").astype(float).fillna(0.0)
        if len(a) == len(b) and (a - b).abs().max() > 1e-9:
            drift.append(c)
    assert not drift, f"CURRENT arm drifted from production: {drift}"
    # pitcher col values themselves may drift (coverage) — compare only
    # shape + coverage stats, report, don't assert.

    spl_cur.to_csv(cur_csv, index=False)
    spl_cor.to_csv(cor_csv, index=False)

    ev = {
        "head": head_sha(),
        "pitches": str(args.pitches),
        "games_production": int(len(prod)),
        "games_spliced_cover": int(len(prod) - uncov_cur),
        "games_nan_pitcher_both_arms": int(uncov_cur),
        "pitcher_stat_columns": pit_cols,
        "non_pitcher_columns_identical_to_production": True,
        "arms_differ_only_in_pitcher_cols": True,
    }
    out = out_dir / f"pitcher_ablation_frames_{head_sha()[:12]}.json"
    out.write_text(json.dumps(ev, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote:\n  {cur_csv}\n  {cor_csv}\n  {out}")


if __name__ == "__main__":
    main()