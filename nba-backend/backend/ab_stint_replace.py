"""Replacement A/B: incumbent game-RAPM nine vs stint-derived columns.

Arms (SAME frame, SAME folds, SAME members, no retuning; PIT throughout):
  I    incumbent : baseline + 9 game-RAPM (pl_rapm_*) -- what ships today.
  S9   stint nets: baseline - 9 game-RAPM + 9 stint nets (pl_srapm_net_*).
  S15  stint full : baseline - 9 game-RAPM + all 15 stint cols (+6 off/def).

Paired verdicts: S9 vs I (better estimation of the same quantities?) and
S15 vs I (nets + O/D decomposition worth it?). S15 vs S9 isolates the six
splits and comes free. Paired per-game log-loss, block-bootstrapped by
gameday (4000 draws), same gate logic as ab_stint_rapm.py.

Per-arm OOFs are saved (reuse for grading/stacking without retraining).

Usage: python ab_stint_replace.py [--stints PATH] [--limit-dates N]
Writes run_diagnostics/ab_stint_replace_{oof_I,oof_S9,oof_S15}.parquet,
  ab_stint_replace_paired_{S9vsI,S15vsI,S15vsS9}.csv, ab_stint_replace_verdict.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("ab_stint_replace")

import ab_stint_rapm as ab
import lineup_projection as proj

STINT_NETS = [c for c in proj.STINT_RAPM_FEATURES if "_net_" in c]
STINT_ALL = list(proj.STINT_RAPM_FEATURES)
assert len(STINT_NETS) == 9, STINT_NETS
assert len(STINT_ALL) == 15, STINT_ALL

OUTDIR = Path(__file__).resolve().parent.parent.parent / "run_diagnostics"


def paired_stats(base_oof: pd.DataFrame, treat_oof: pd.DataFrame,
                 label_b: str, label_t: str, seed: int) -> dict:
    base_oof = base_oof[base_oof.grades_pooled].copy()
    treat_oof = treat_oof[treat_oof.grades_pooled].copy()
    merged = (base_oof[["game_id", "gameday", "home_win",
                        "p_ensemble_calibrated"]]
              .rename(columns={"p_ensemble_calibrated": "p_b"})
              .merge(treat_oof[["game_id", "p_ensemble_calibrated"]]
                     .rename(columns={"p_ensemble_calibrated": "p_t"}),
                     on="game_id", how="inner"))
    y = merged.home_win.to_numpy(float)
    eps = 1e-12
    pb = np.clip(merged.p_b.to_numpy(float), eps, 1 - eps)
    pt = np.clip(merged.p_t.to_numpy(float), eps, 1 - eps)
    d = -(y * np.log(pt) + (1 - y) * np.log(1 - pt)) \
        + (y * np.log(pb) + (1 - y) * np.log(1 - pb))
    n = len(d)
    mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
    rng = np.random.default_rng(seed)
    blocks = [part.index.to_numpy() for _, part in merged.groupby("gameday")]
    boot = np.array([d[np.concatenate([blocks[i] for i in
                     rng.integers(0, len(blocks), len(blocks))])].mean()
                     for _ in range(4000)])
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    return {"label_b": label_b, "label_t": label_t, "n": n,
            "mean_delta": mean, "se": se, "t": mean / se if se else float("nan"),
            "ci95": [lo, hi], "distinguishable": bool(lo * hi > 0),
            "treat_better_share": float((d < 0).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stints", default=str(ab.DEFAULT_STINTS))
    ap.add_argument("--limit-dates", type=int, default=0)
    ap.add_argument("--rerun", action="store_true",
                    help="retrain even if arm OOF exists")
    args = ap.parse_args()
    import config
    import features as feat_mod
    import folds as folds_mod
    import ingestion
    import moneyline as ml_mod

    oof_paths = {a: OUTDIR / f"ab_stint_replace_oof_{a}.parquet"
                 for a in ("I", "S9", "S15")}
    logger.info("loading the fact set from cache")
    facts = ingestion.load_ingested(allow_download=False)
    games = facts.games
    settled = ingestion.trainable_games(games)
    game_df = feat_mod.build_game_features(settled, facts.team_stats,
                                           facts.team_events)
    game_df = folds_mod.canonical_sort(game_df, "gameday")
    base = game_df.copy()
    if args.limit_dates:
        keep = sorted(base.gameday.unique())[:args.limit_dates]
        base = base[base.gameday.isin(keep)].copy()
        games = games[pd.to_datetime(games.gameday).isin(
            pd.to_datetime(keep))].copy()

    designations = proj.load_designations(ab.CACHE)
    if designations is None:
        designations = proj.load_designations(config.DATA_DELIVERY_DIR)
    positions = pd.concat([
        ingestion._fetch_positions(season)
        for season in sorted({ingestion.season_label(d.date())
                              for d in pd.to_datetime(games.gameday).dropna()})
    ], ignore_index=True)
    ratings, name_by_id = ab.build_game_ratings(
        base, designations, team_stats=facts.team_stats,
        player_stats=facts.player_stats, positions=positions)
    rated = ratings.rename(columns={"target_date": "gameday"})
    rated["gameday"] = pd.to_datetime(rated.gameday)
    if "is_available" not in rated.columns:
        rated["is_available"] = True
    rated = proj.apply_pit_designations(rated, designations, name_by_id)
    aggregates = proj.projected_lineup(rated, games=base)
    enriched9 = proj.attach_position_rapm(base, aggregates)
    enriched15, _ = ab.build_stint_enriched(base, ratings, designations,
                                            name_by_id, args.stints)
    missing = [c for c in STINT_ALL if c not in enriched15.columns]
    if missing:
        raise RuntimeError(f"stint features missing: {missing}")

    slim = base.drop(columns=[c for c in ab.POSITION_FEATURES
                              if c in base.columns])
    slim = slim.drop(columns=[c for c in STINT_ALL if c in slim.columns])
    joined = slim.merge(
        enriched9[["gameday", "home_team", "away_team", *ab.POSITION_FEATURES]]
        .drop_duplicates(subset=["gameday", "home_team", "away_team"]),
        on=["gameday", "home_team", "away_team"], how="left")
    joined = joined.merge(
        enriched15[["gameday", "home_team", "away_team", *STINT_ALL]]
        .drop_duplicates(subset=["gameday", "home_team", "away_team"]),
        on=["gameday", "home_team", "away_team"], how="left")
    fold_list = folds_mod.make_folds(joined)
    logger.info("folds: %d", len(fold_list))
    baseline = [c for c in config.MONEYLINE_FEATURE_COLS
                if c not in set(ab.POSITION_FEATURES)]
    arms = [("I", baseline + ab.POSITION_FEATURES),
            ("S9", baseline + STINT_NETS),
            ("S15", baseline + STINT_ALL)]

    oofs, metrics = {}, {}
    for name, columns in arms:
        if oof_paths[name].exists() and not args.rerun:
            logger.info("arm %s OOF exists, loading", name)
            oof = pd.read_parquet(oof_paths[name])
            grading = oof[oof.grades_pooled.astype(bool)]
            metrics[name] = ab.score(
                grading.p_ensemble_calibrated.to_numpy(float),
                grading.home_win.to_numpy(float))
        else:
            m, oof = ab.run_arm(name, columns, joined, fold_list, ml_mod,
                                config)
            metrics[name] = m
            oof.to_parquet(oof_paths[name], index=False)
        oofs[name] = oof
        logger.info("arm %s weights: %s", name,
                    {k: round(v, 4) for k, v in
                     getattr(oof, "attrs", {}).get("member_weights", {}).items()} \
                    if hasattr(oof, "attrs") else "")
    config.reset_feature_subset()

    for key in ("logloss", "accuracy", "brier", "auc", "ece"):
        row = f"  {key:<10}" + "".join(
            f" {metrics[a][key]:>10.5f}" if key in metrics[a] else f" {'-':>10}"
            for a in ("I", "S9", "S15"))
        print(row)
    verdict = {"date": pd.Timestamp.today().date().isoformat(),
               "arms": {a: {k: metrics[a][k] for k in
                             ("logloss", "accuracy", "brier", "auc", "ece")
                             if k in metrics[a]} for a in metrics},
               "paired": {}}
    for b, t, tag in (("I", "S9", "S9vsI"), ("I", "S15", "S15vsI"),
                      ("S9", "S15", "S15vsS9")):
        st = paired_stats(oofs[b], oofs[t], b, t, config.RANDOM_SEED)
        verdict["paired"][tag] = st
        print(f"\n=== paired log-loss: {t} minus {b}")
        print(f"  n={st['n']} mean {st['mean_delta']:+.5f} "
              f"se {st['se']:.5f} t {st['t']:+.2f} "
              f"CI [{st['ci95'][0]:+.5f}, {st['ci95'][1]:+.5f}] "
              f"{'DISTINGUISHABLE' if st['distinguishable'] else 'not distinguishable'} "
              f"(treat better {100*st['treat_better_share']:.1f}%)")
        pd.DataFrame({"delta": [st["mean_delta"]]}).to_csv(
            OUTDIR / f"ab_stint_replace_paired_{tag}.csv", index=False)
    (OUTDIR / "ab_stint_replace_verdict.json").write_text(
        json.dumps(verdict, indent=1))
    logger.info("done")


if __name__ == "__main__":
    main()
