"""Holdout A/B: baseline vs baseline+9 (game RAPM) vs +9+15 (stint O/D RAPM).

The question is not whether fifteen columns beat nothing - the nine
game-level features already passed that gate in ``ab_position_rapm.py``.
The question is what the stint offense/defense ratings add OVER the nine:
arm 3 differs from arm 2 in the fifteen ``pl_srapm_*`` columns and nothing
else. Arm 1 (legacy contract minus the nine) anchors both.

Same honesty rules as the nine's harness: the SAME folds built once and
passed to all arms, the SAME game frame with columns joined on, no
retuning between arms, point-in-time ratings throughout, and a PAIRED
per-game comparison at the end (block-bootstrapped by date), because two
point estimates are not a result.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("ab_stint")

CACHE = Path(os.environ.get("NBA_CACHE_DIR")
             or os.path.expanduser("~/.cache/sports_prediction_model/nba"))
ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STINTS = ROOT / "run_diagnostics" / "nba_stints.parquet"

POSITION_FEATURES = [f"pl_rapm_{p}_{side}" for p in ("c", "f", "g")
                     for side in ("away", "home", "diff")]


def build_game_ratings(games, designations, team_stats=None, player_stats=None,
                       positions=None):
    """Game-level ratings frame (with target_date), mirroring the nine's
    harness so the arms differ in features and nothing else."""
    import lineup_projection as proj
    import player_rapm as rapm_mod
    from ab_position_rapm import build_player_games
    if player_stats is None:
        games_frame, name_by_id = build_player_games()
    else:
        games_frame = rapm_mod.prepare_player_games(player_stats, positions)
        name_by_id = dict(zip(rapm_mod._player_id_str(player_stats.player_id),
                              player_stats.player_name.astype(str)))
    dates = pd.Series(sorted(pd.to_datetime(games.gameday).dropna().unique()))
    logger.info("building game-level ratings for %d target dates",
                len(dates))
    started = time.time()
    import ingestion
    ratings = rapm_mod.build_player_rapm(
        games_frame, target_dates=dates, team_stats=team_stats,
        roster_history=ingestion.load_roster_history(CACHE))
    logger.info("game ratings: %d rows in %.0fs", len(ratings),
                time.time() - started)
    return ratings, name_by_id


def build_stint_enriched(games, ratings, designations, name_by_id,
                         stints_path):
    """Fifteen stint features over the same games, same pools, same day."""
    import lineup_projection as proj
    import stint_rapm as srapm_mod
    import config
    stints = pd.read_parquet(stints_path)
    logger.info("stints: %d rows, %d games", len(stints),
                stints.game_key.nunique())
    started = time.time()
    stint_ratings = srapm_mod.build_stint_rapm(
        ratings, stints, lam=config.STINT_RAPM_LAMBDA)
    logger.info("stint ratings: %d rows in %.0fs", len(stint_ratings),
                time.time() - started)
    stint_ratings = stint_ratings.rename(columns={"target_date": "gameday"})
    stint_ratings["gameday"] = pd.to_datetime(stint_ratings.gameday)
    if "is_available" not in stint_ratings.columns:
        stint_ratings["is_available"] = True
    stint_ratings = proj.apply_pit_designations(stint_ratings, designations,
                                                name_by_id)
    aggregates = proj.projected_lineup(stint_ratings, games=games)
    logger.info("stint aggregates: %d team-games", len(aggregates))
    return proj.attach_stint_rapm(games, aggregates), aggregates


def score(p, y):
    from evaluation import binary_metrics, ece
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(p)
    p, y = p[keep], y[keep]
    metrics = binary_metrics(p, y)
    metrics["ece"] = ece(p, y)
    metrics["n"] = int(len(p))
    return metrics


def run_arm(name, columns, joined, fold_list, ml_mod, config):
    missing = [c for c in columns if c not in joined.columns]
    if missing:
        logger.warning("%s: dropping %d absent column(s)", name,
                       len(missing))
        columns = [c for c in columns if c in joined.columns]
    config.set_feature_subset(columns)
    started = time.time()
    result = ml_mod.walk_forward_oof(joined, fold_list=fold_list,
                                     progress_every=0)
    oof = result["oof"]
    if oof.empty or any(oof[f"p_{m}"].isna().any()
                        for m in config.ENSEMBLE_MEMBERS):
        config.reset_feature_subset()
        raise RuntimeError(f"{name}: incomplete member OOF; A/B is invalid")
    grading = oof[oof.grades_pooled.astype(bool)]
    metrics = score(grading.p_ensemble_calibrated.to_numpy(float),
                    grading.home_win.to_numpy(float))
    print(f"\n=== {name}: {len(columns)} features, "
          f"{len(oof)} OOF rows, {time.time()-started:.0f}s")
    for key in ("logloss", "accuracy", "brier", "auc", "ece", "n"):
        if key in metrics:
            print(f"  {key:<10} {metrics[key]:.5f}")
    weights = result["member_weights"]
    print(f"  weights   {({k: round(v, 4) for k, v in weights.items()})}")
    return metrics, oof


def paired(report_a, report_b, label_a, label_b, config):
    base_oof, treat_oof = report_a[1], report_b[1]
    if (base_oof.game_id.duplicated().any()
            or not base_oof.game_id.equals(treat_oof.game_id)
            or not base_oof.home_win.equals(treat_oof.home_win)
            or not base_oof.grades_pooled.equals(treat_oof.grades_pooled)):
        raise RuntimeError("A/B OOF populations differ; paired scoring refused")
    base_oof = base_oof[base_oof.grades_pooled].copy()
    treat_oof = treat_oof[treat_oof.grades_pooled].copy()
    base_oof["p_ensemble"] = base_oof.p_ensemble_calibrated
    treat_oof["p_ensemble"] = treat_oof.p_ensemble_calibrated
    merged = (base_oof[["game_id", "gameday", "home_win", "p_ensemble"]]
              .rename(columns={"p_ensemble": "p_ensemble_b"})
              .merge(treat_oof[["game_id", "p_ensemble"]]
                     .rename(columns={"p_ensemble": "p_ensemble_t"}),
                     on="game_id", how="inner"))
    y = merged.home_win.to_numpy(float)
    p_b = merged.p_ensemble_b.to_numpy(float)
    p_t = merged.p_ensemble_t.to_numpy(float)
    eps = 1e-12

    def per_game(p):
        p = np.clip(p, eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    d = per_game(p_t) - per_game(p_b)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    print(f"\n=== paired log-loss: {label_b} minus {label_a}")
    if se == 0:
        print("IDENTICAL predictions - the features changed nothing")
        return merged
    rng = np.random.default_rng(config.RANDOM_SEED)
    blocks = [part.index.to_numpy() for _, part in merged.groupby("gameday")]
    boot = np.array([
        d[np.concatenate([blocks[i] for i in
                          rng.integers(0, len(blocks), len(blocks))])].mean()
        for _ in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  n = {n}")
    print(f"  mean delta        {mean:+.5f}")
    print(f"  standard error    {se:.5f}")
    print(f"  t statistic       {mean/se:+.2f}")
    print(f"  95% bootstrap CI  [{lo:+.5f}, {hi:+.5f}]")
    print(f"  verdict: {'distinguishable from zero' if lo * hi > 0 else 'NOT distinguishable from zero'}")
    print(f"  treatment better on {int((d < 0).sum())} of {n} games "
          f"({100*(d<0).mean():.1f}%)")
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stints", default=str(DEFAULT_STINTS))
    ap.add_argument("--limit-dates", type=int, default=0,
                    help="debug: build ratings for the first N dates only")
    args = ap.parse_args()
    import config
    import evaluation  # noqa: F401
    import features as feat_mod
    import folds as folds_mod
    import ingestion
    import moneyline as ml_mod
    import lineup_projection as proj

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

    designations = proj.load_designations(CACHE)
    if designations is None:
        designations = proj.load_designations(config.DATA_DELIVERY_DIR)

    positions = pd.concat([
        ingestion._fetch_positions(season)
        for season in sorted({ingestion.season_label(d.date())
                              for d in pd.to_datetime(games.gameday).dropna()})
    ], ignore_index=True)

    ratings, name_by_id = build_game_ratings(
        base, designations, team_stats=facts.team_stats,
        player_stats=facts.player_stats, positions=positions)
    rated = ratings.rename(columns={"target_date": "gameday"})
    rated["gameday"] = pd.to_datetime(rated.gameday)
    if "is_available" not in rated.columns:
        rated["is_available"] = True
    rated = proj.apply_pit_designations(rated, designations, name_by_id)
    aggregates = proj.projected_lineup(rated, games=base)
    enriched9 = proj.attach_position_rapm(base, aggregates)

    enriched15, _ = build_stint_enriched(base, ratings, designations,
                                         name_by_id, args.stints)

    import stint_rapm as srapm_mod  # for the feature list only
    del srapm_mod
    stint_features = list(proj.STINT_RAPM_FEATURES)
    missing = [c for c in stint_features if c not in enriched15.columns]
    if missing:
        raise RuntimeError(f"stint features missing: {missing}")

    base = base.drop(columns=[c for c in POSITION_FEATURES
                              if c in base.columns])
    base = base.drop(columns=[c for c in stint_features
                              if c in base.columns])
    joined = base.merge(
        enriched9[["gameday", "home_team", "away_team", *POSITION_FEATURES]]
        .drop_duplicates(subset=["gameday", "home_team", "away_team"]),
        on=["gameday", "home_team", "away_team"], how="left")
    joined = joined.merge(
        enriched15[["gameday", "home_team", "away_team", *stint_features]]
        .drop_duplicates(subset=["gameday", "home_team", "away_team"]),
        on=["gameday", "home_team", "away_team"], how="left")

    print("\n=== coverage of the fifteen stint features")
    for column in stint_features:
        values = pd.to_numeric(joined[column], errors="coerce")
        populated = int(values.notna().sum())
        distinct = int(values.dropna().nunique())
        print(f"  {column:<24} {populated:5d}/{len(joined):5d} rows  "
              f"({100*populated/len(joined):5.1f}%)  distinct {distinct:5d}"
              f"{'   CONSTANT' if distinct <= 1 else ''}")

    fold_list = folds_mod.make_folds(joined)
    baseline = [c for c in config.MONEYLINE_FEATURE_COLS
                if c not in set(POSITION_FEATURES)]
    arms = [("baseline", baseline),
            ("position_rapm", baseline + POSITION_FEATURES),
            ("stint_rapm", baseline + POSITION_FEATURES + stint_features)]
    results = {}
    for name, columns in arms:
        results[name] = run_arm(name, columns, joined, fold_list, ml_mod,
                                config)
    config.reset_feature_subset()

    b, t = results["baseline"][0], results["position_rapm"][0]
    s = results["stint_rapm"][0]
    print("\n=== side by side")
    print(f"  {'metric':<10} {'baseline':>10} {'pos9':>10} {'stint15':>10}")
    for key in ("logloss", "accuracy", "brier", "auc", "ece"):
        row = f"  {key:<10}"
        for m in (b, t, s):
            row += f" {m[key]:>10.5f}" if key in m else f" {'-':>10}"
        print(row)

    m1 = paired(results["position_rapm"], results["stint_rapm"],
                "position_rapm", "stint_rapm", config)
    m1.to_csv(Path(config.RUN_DIAGNOSTICS_DIR) / "ab_stint_paired_oof.csv",
              index=False)
    paired(results["baseline"], results["stint_rapm"],
           "baseline", "stint_rapm", config)
    # The nine's own gate receipt: what position_rapm adds over baseline,
    # paired on the same games. Without it the harness can only say the
    # fifteen add nothing over the nine, never whether the nine add
    # anything over nothing.
    paired(results["baseline"], results["position_rapm"],
           "baseline", "position_rapm", config)


if __name__ == "__main__":
    main()
