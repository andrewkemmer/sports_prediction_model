"""Holdout A/B: baseline contract vs baseline + position-segmented RAPM.

Runs the identical walk-forward over the identical games with one difference -
whether the nine ``pl_rapm_*`` columns are in the contract - and reports the
out-of-fold metrics side by side.

What makes this an honest comparison rather than two numbers:

* the SAME folds, built once and passed to both runs, so neither arm gets a
  warmer or cooler validation window;
* the SAME game frame, with the extra columns joined on rather than rebuilt,
  so the arms differ in the features and nothing else;
* every model hyperparameter untouched - no retuning for the treatment arm,
  which would let a nine-column arm win on tuning rather than on signal;
* the feature build is point-in-time by construction (strictly prior rating
  rows, last pre-tipoff designation), and the designation source is applied
  only to games it covers.

Run:
    python ab_position_rapm.py
"""
from __future__ import annotations

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
logger = logging.getLogger("ab")

CACHE = Path(os.environ.get("NBA_CACHE_DIR")
             or os.path.expanduser("~/.cache/sports_prediction_model/nba"))

POSITION_FEATURES = [f"pl_rapm_{p}_{side}" for p in ("c", "f", "g")
                     for side in ("away", "home", "diff")]


def build_player_games():
    """The rated player-games frame plus a (team, name) -> player_id map.

    Reads the whole-season files ONLY. The cache directory also holds 60-day
    slices from earlier runs, and concatenating both counts every game twice.

    The name map exists because the injury report names players ("First Last")
    while the rating frame identifies them by id, and the designation filter
    has to reach the player rather than the team.
    """
    import player_rapm as rapm_mod
    paths = sorted({*CACHE.glob("season_logs/log_*_Regular_Season.parquet"),
                    *CACHE.glob("season_logs/log_*_Playoffs.parquet")})
    frames = [pd.read_parquet(f) for f in paths]
    log = pd.concat(frames, ignore_index=True)
    log = log.drop_duplicates(subset=["nba_game_id", "player_id", "SEASON_ID"],
                              keep="first")
    log["gameday"] = pd.to_datetime(log["gameday"], errors="coerce")
    id_by_name: dict = {}
    name_by_id: dict = {}
    for pid, pname, team in zip(rapm_mod._player_id_str(log.player_id),
                                log.player_name.astype(str), log.team.astype(str)):
        id_by_name.setdefault((team, pname), pid)
        name_by_id.setdefault(pid, pname)
    seasons = sorted({f.stem.rsplit("_", 1)[-1] for f in
                      CACHE.glob("positions/positions_*.parquet")})
    positions = []
    for season in seasons:
        # Newest first: v3's collapsed cell follows the league's own primary,
        # v2's follows the local convention but still carries the listing the
        # segments read, and v1 - still on disk from before the listing existed
        # - has no listing at all. The fallbacks are outage paths, so the first
        # file present wins and none of them is silently upgraded.
        path = next((CACHE / "positions" / f"positions_v{v}_{season}.parquet"
                     for v in (3, 2)
                     if (CACHE / "positions"
                         / f"positions_v{v}_{season}.parquet").exists()),
                    CACHE / "positions" / f"positions_{season}.parquet")
        if path.exists():
            frame = pd.read_parquet(path)
            frame["season"] = season
            positions.append(frame)
    frame = rapm_mod.prepare_player_games(
        log, pd.concat(positions, ignore_index=True))
    return frame, name_by_id


def _flip_name(name: str) -> str:
    """'First Last' -> 'Last, First', the form the injury report uses."""
    if "," in name:
        last, first = name.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return name


def build_position_features(games: pd.DataFrame,
                            designations: pd.DataFrame | None,
                            team_stats: pd.DataFrame | None = None,
                            player_stats: pd.DataFrame | None = None,
                            positions: pd.DataFrame | None = None,
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The nine features, for every decided game in the window.

    The pool is built for EVERY date in the frame rather than one target, which
    is what makes these usable: a column populated on 1 row in 2,779 is a
    column the model cannot learn from, and that was the blocker.
    """
    import lineup_projection as proj
    import player_rapm as rapm_mod
    if player_stats is None:
        games_frame, name_by_id = build_player_games()
    else:
        # The production fact population, not a separate regular-season-only
        # cache sweep: playoff evidence must enter the same ratings as serving.
        games_frame = rapm_mod.prepare_player_games(player_stats, positions)
        name_by_id = dict(zip(rapm_mod._player_id_str(player_stats.player_id),
                              player_stats.player_name.astype(str)))
    dates = pd.Series(sorted(pd.to_datetime(games.gameday).dropna().unique()))
    logger.info("building ratings for %d target dates", len(dates))
    started = time.time()
    import ingestion
    ratings = rapm_mod.build_player_rapm(games_frame, target_dates=dates,
                                         team_stats=team_stats,
                                         roster_history=ingestion.load_roster_history(CACHE))
    ratings = ratings.rename(columns={"target_date": "gameday"})
    ratings["gameday"] = pd.to_datetime(ratings.gameday)
    if "is_available" not in ratings.columns:
        ratings["is_available"] = True
    logger.info("ratings: %d rows in %.0fs", len(ratings), time.time() - started)

    # Apply the official designations through the SHARED production filter.
    # The removal contract (Out/Doubtful/Recovery -> removed from THAT game's
    # pool only; rating survives on every game he played) lives in
    # ``lineup_projection.apply_pit_designations`` so this harness and the
    # nightly run cannot drift into scoring a different feature. The original
    # inline version of this block is where the team-level-key bug lived -
    # 39,285 whole-roster removals, plausible aggregates, quietly disabled
    # feature - which is exactly why the definition is now in one place.
    ratings = proj.apply_pit_designations(ratings, designations, name_by_id)

    started = time.time()
    aggregates = proj.projected_lineup(ratings, games=games)
    logger.info("aggregates: %d team-games in %.0fs", len(aggregates),
                time.time() - started)
    return proj.attach_position_rapm(games, aggregates), aggregates


def score(p: np.ndarray, y: np.ndarray) -> dict:
    from evaluation import binary_metrics, ece
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(p)
    p, y = p[keep], y[keep]
    metrics = binary_metrics(p, y)
    metrics["ece"] = ece(p, y)
    metrics["n"] = int(len(p))
    return metrics


def main() -> None:
    import config
    import evaluation
    import features as feat_mod
    import folds as folds_mod
    import ingestion
    import moneyline as ml_mod

    logger.info("loading the fact set from cache")
    facts = ingestion.load_ingested(allow_download=False)
    games = facts.games
    settled = ingestion.trainable_games(games)
    logger.info("settled eligible games: %d", len(settled))
    game_df = feat_mod.build_game_features(settled, facts.team_stats,
                                           facts.team_events)
    game_df = folds_mod.canonical_sort(game_df, "gameday")
    base = game_df.copy()

    # Every backfilled shard, through the loader the production path uses, so
    # the arm measured here is the arm that ships. This file used to name one
    # window's file outright, which meant a re-run backfill extending the
    # archive was invisible to the experiment that justifies the features.
    import lineup_projection as proj_mod
    designations = proj_mod.load_designations(CACHE)
    if designations is None:
        designations = proj_mod.load_designations(config.DATA_DELIVERY_DIR)
    logger.info("designations: %s",
                "none - full pool retained" if designations is None
                else f"{len(designations)} rows")

    logger.info("building the nine position features")
    positions = pd.concat([
        ingestion._fetch_positions(season)
        for season in sorted({ingestion.season_label(d.date())
                              for d in pd.to_datetime(games.gameday).dropna()})
    ], ignore_index=True)
    enriched, aggregates = build_position_features(
        base, designations, team_stats=facts.team_stats,
        player_stats=facts.player_stats, positions=positions)

    missing = [c for c in POSITION_FEATURES if c not in enriched.columns]
    if missing:
        raise RuntimeError(f"position features missing: {missing}")

    # ``build_game_features`` already creates the nine columns (the ladder
    # derives them from KNOWN_FEATURE_COLS), all-NaN. Drop them before the
    # merge: keeping them under the bare names while the real values land in
    # suffixed copies is exactly how an A/B scores empty features with no
    # error - and the first run of this harness did precisely that.
    base = base.drop(columns=[c for c in POSITION_FEATURES
                              if c in base.columns])
    joined = base.merge(
        enriched[["gameday", "home_team", "away_team", *POSITION_FEATURES]]
        .drop_duplicates(subset=["gameday", "home_team", "away_team"]),
        on=["gameday", "home_team", "away_team"], how="left")

    print("\n=== coverage of the nine features on the decided frame")
    for column in POSITION_FEATURES:
        values = pd.to_numeric(joined[column], errors="coerce")
        populated = int(values.notna().sum())
        distinct = int(values.dropna().nunique())
        print(f"  {column:<16} {populated:5d}/{len(joined):5d} rows  "
              f"({100*populated/len(joined):5.1f}%)  distinct {distinct:5d}"
              f"{'   CONSTANT' if distinct <= 1 else ''}")

    fold_list = folds_mod.make_folds(joined)
    logger.info("folds: %d", len(fold_list))

    # The baseline is the LEGACY contract - MONEYLINE_FEATURE_COLS minus the
    # nine - because once the features are promoted they live in that list,
    # and "baseline plus the nine" would compare the features against
    # themselves. This keeps the harness's question well-posed before AND
    # after promotion: what do the nine add over the contract that spawned
    # them?
    baseline = [c for c in config.MONEYLINE_FEATURE_COLS
                if c not in set(POSITION_FEATURES)]
    results = {}
    for name, columns in (("baseline", baseline),
                          ("position_rapm", baseline + POSITION_FEATURES)):
        missing_cols = [c for c in columns if c not in joined.columns]
        if missing_cols:
            logger.warning("%s: dropping %d absent column(s)", name,
                           len(missing_cols))
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
        print(f"  grading rows: {len(grading)}; all-row diagnostic: "
              f"{score(oof.p_ensemble_calibrated, oof.home_win)}")
        results[name] = (metrics, oof)
        print(f"\n=== {name}: {len(columns)} features, "
              f"{len(oof)} OOF rows, {time.time()-started:.0f}s")
        for key in ("logloss", "accuracy", "brier", "auc", "ece", "n"):
            if key in metrics:
                print(f"  {key:<10} {metrics[key]:.5f}")
        weights = result["member_weights"]
        print(f"  weights   {({k: round(v, 4) for k, v in weights.items()})}")

    config.reset_feature_subset()
    b, t = results["baseline"][0], results["position_rapm"][0]
    print("\n=== side by side (position_rapm minus baseline)")
    print(f"  {'metric':<10} {'baseline':>10} {'position_rapm':>12} {'delta':>10}")
    for key in ("logloss", "accuracy", "brier", "auc", "ece"):
        if key in b and key in t:
            print(f"  {key:<10} {b[key]:>10.5f} {t[key]:>12.5f} "
                  f"{t[key]-b[key]:>+10.5f}")

    # Two point estimates are not a result. The arms are scored on the SAME
    # games, so the honest comparison is PAIRED: per-game log-loss difference,
    # its standard error, and a bootstrap interval on the mean. A mean delta of
    # -0.0007 means nothing if the per-game deltas are an order of magnitude
    # wider, which on 2,000 games they usually are.
    base_oof = results["baseline"][1]
    treat_oof = results["position_rapm"][1]
    if (base_oof.game_id.duplicated().any()
            or not base_oof.game_id.equals(treat_oof.game_id)
            or not base_oof.home_win.equals(treat_oof.home_win)
            or not base_oof.grades_pooled.equals(treat_oof.grades_pooled)):
        raise RuntimeError("A/B OOF populations differ; paired scoring refused")
    base_oof = base_oof[base_oof.grades_pooled].copy()
    treat_oof = treat_oof[treat_oof.grades_pooled].copy()
    # Score the probability actually served, not a final-weight replay.
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
    if se == 0:
        # Identical arms score identical losses on every game. That is a
        # verdict about the harness (the columns carry nothing the model
        # used), not a measurement to divide by.
        print("\nthe two arms produced IDENTICAL predictions - the features "
              "did not reach the model or changed nothing")
        return
    rng = np.random.default_rng(config.RANDOM_SEED)
    # Resample game dates together: same-night outcomes share injuries,
    # travel and league conditions and are not independent observations.
    blocks = [part.index.to_numpy() for _, part in merged.groupby("gameday")]
    boot = np.array([
        d[np.concatenate([blocks[i] for i in
                          rng.integers(0, len(blocks), len(blocks))])].mean()
        for _ in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n=== paired per-game log-loss difference (treatment - baseline)")
    print(f"  n = {n}")
    print(f"  mean delta        {mean:+.5f}")
    print(f"  standard error    {se:.5f}")
    print(f"  t statistic       {mean/se:+.2f}")
    print(f"  95% bootstrap CI  [{lo:+.5f}, {hi:+.5f}]")
    verdict = ("distinguishable from zero" if lo * hi > 0
               else "NOT distinguishable from zero")
    print(f"  verdict: {verdict}")
    print(f"  treatment better on {int((d < 0).sum())} of {n} games "
          f"({100*(d<0).mean():.1f}%)")

    output = Path(config.RUN_DIAGNOSTICS_DIR) / "ab_paired_oof.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"\n  paired OOF written to {output}")


if __name__ == "__main__":
    main()
