"""Three-arm DARKO pts/100: nine blended-DPM segments vs nine pts/100 vs both.

  A (control) : 9 x pl_dpm_* from total dpm (production config: trailing
                minutes, pool as-is, gates on, waterfall).
  B (treat)   : 9 x pl_pts100_* from x_pts_100 (DARKO expected points per
                100, offensive output), built by the IDENTICAL production
                blend with only the rating column swapped.
  C (treat)   : 18 = A nine + B nine.

Same folds (built once on A), same frame, same designations, same
hyperparameters. Paired verdicts A/B and A/C with gameday-blocked CIs.
Adopt bar: paired t <= -2.5 with tail slices not regressing.

Run where the cache lives:
    python nba-backend/backend/ab_darko_pts100.py
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
logger = logging.getLogger("ab_darko_pts100")

from ab_position_rapm import (  # noqa: E402
    POSITION_FEATURES,
    score,
)
from ab_split_od import _paired  # noqa: E402


def main() -> None:
    import config
    import folds as folds_mod
    import features as feat_mod
    import ingestion
    import lineup_projection as proj_mod
    import moneyline as ml_mod

    # Production config: NO overrides. (ab_split_od forced expected-minutes
    # + widened pool; both were rejected afterward. This arm tests the rating
    # column only.)
    logger.info("loading the fact set from cache (allow_download=False)")
    facts = ingestion.load_ingested(allow_download=False)
    games = facts.games
    settled = ingestion.trainable_games(games)
    logger.info("settled eligible games: %d", len(settled))
    game_df = feat_mod.build_game_features(
        settled, facts.team_stats, facts.team_events)
    game_df = folds_mod.canonical_sort(game_df, "gameday")
    base = game_df.copy()

    designations = proj_mod.load_designations(
        Path(os.environ.get("NBA_CACHE_DIR")
             or os.path.expanduser("~/.cache/sports_prediction_model/nba")))
    if designations is None:
        designations = proj_mod.load_designations(config.DATA_DELIVERY_DIR)

    positions = pd.concat([
        ingestion._fetch_positions(season)
        for season in sorted({ingestion.season_label(d.date())
                              for d in pd.to_datetime(games.gameday).dropna()})
    ], ignore_index=True)

    import player_rapm as rapm_mod
    import darko_source
    games_frame = rapm_mod.prepare_player_games(
        facts.player_stats, positions)
    dates = pd.Series(sorted(pd.to_datetime(settled.gameday).dropna().unique()))
    logger.info("building ratings for %d target dates", len(dates))
    ratings = rapm_mod.build_player_rapm(
        games_frame, target_dates=dates, team_stats=facts.team_stats,
        roster_history=ingestion.load_roster_history())
    ids = rapm_mod._player_key(ratings["player_id"]).dropna().unique()
    history = darko_source.fetch_darko_history(ids)
    ratings = rapm_mod.apply_darko_rating(ratings, history)
    if "x_pts_100" not in ratings.columns:
        raise RuntimeError("ratings carry no x_pts_100; apply_darko_rating did not attach it")
    logger.info("x_pts_100 coverage: %.3f",
                float(ratings["x_pts_100"].notna().mean()))

    name_by_id = dict(zip(
        rapm_mod._player_id_str(facts.player_stats.player_id),
        facts.player_stats.player_name.astype(str)))

    rated_total = ratings.rename(columns={"target_date": "gameday"}) \
        if "target_date" not in ratings.columns else ratings
    rated_total = rated_total.copy()
    rated_total["gameday"] = pd.to_datetime(rated_total["gameday"])
    if "is_available" not in rated_total.columns:
        rated_total["is_available"] = True
    rated_total = proj_mod.apply_pit_designations(
        rated_total, designations, name_by_id)

    # A: production path, total DPM.
    frame0, _ = proj_mod.project_lineup_features(
        rated_total, games=base, designations=designations,
        name_by_id=name_by_id, reweight="waterfall")

    # B: identical blend, x_pts_100 swapped in where present (incumbent
    # fallback otherwise -- same contract as the o_dpm/d_dpm swap).
    swapped = ratings.copy()
    vals = pd.to_numeric(swapped["x_pts_100"], errors="coerce")
    swapped["rapm_shrunk"] = vals.where(vals.notna(), swapped["rapm_shrunk"])
    filt = proj_mod.apply_pit_designations(
        swapped.rename(columns={"target_date": "gameday"})
        .assign(gameday=lambda df: pd.to_datetime(df["gameday"])),
        designations, name_by_id)
    agg = proj_mod.projected_lineup(filt, games=base, reweight="waterfall")
    attached = proj_mod.attach_position_rapm(base, agg)
    rename = {c: c.replace("pl_dpm_", "pl_pts100_", 1)
              for c in POSITION_FEATURES if c in attached.columns}
    attached = attached.rename(columns=rename)
    keep = (["gameday", "home_team", "away_team"]
            + [c for c in rename.values() if c in attached.columns])
    pts_frame = attached[keep].drop_duplicates(
        subset=["gameday", "home_team", "away_team"])
    pts_cols = [c for c in rename.values() if c in pts_frame.columns]
    cov = float(pd.to_numeric(pts_frame["pl_pts100_c_diff"],
                              errors="coerce").notna().mean()) \
        if "pl_pts100_c_diff" in pts_frame.columns else 0.0
    logger.info("pl_pts100_c_diff coverage %.3f", cov)

    key = ["gameday", "home_team", "away_team"]
    # B: control frame minus the nine totals, plus the nine pts/100.
    fb_b = frame0.drop(columns=[c for c in POSITION_FEATURES
                                if c in frame0.columns], errors="ignore")
    fb_b = fb_b.merge(pts_frame, on=key, how="left")
    # C: control nine + pts/100 nine.
    fb_c = frame0.merge(pts_frame, on=key, how="left",
                        suffixes=("", "_dup"))
    fb_c = fb_c.drop(columns=[c for c in fb_c.columns if c.endswith("_dup")])

    arms = {"A_total9": frame0, "B_pts100_9": fb_b, "C_18": fb_c}
    for name, fb in arms.items():
        fp = Path(config.RUN_DIAGNOSTICS_DIR) / f"ab_darko_pts100_{name}_features.parquet"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fb.to_parquet(fp, index=False)
        logger.info("%s: rows %d", name, len(fb))

    fold_list = folds_mod.make_folds(arms["A_total9"])
    logger.info("folds: %d", len(fold_list))
    baseline_cols = [c for c in config.MONEYLINE_FEATURE_COLS
                     if c not in set(POSITION_FEATURES)]
    arm_cols = {"A_total9": baseline_cols + POSITION_FEATURES,
                "B_pts100_9": baseline_cols + pts_cols,
                "C_18": baseline_cols + POSITION_FEATURES + pts_cols}
    results = {}
    for name in ("A_total9", "B_pts100_9", "C_18"):
        joined = arms[name]
        cols = [c for c in arm_cols[name] if c in joined.columns]
        config.set_feature_subset(cols)
        started = time.time()
        result = ml_mod.walk_forward_oof(joined, fold_list=fold_list,
                                         progress_every=0)
        oof = result["oof"]
        grading = oof[oof.grades_pooled.astype(bool)]
        metrics = score(grading.p_ensemble_calibrated.to_numpy(float),
                        grading.home_win.to_numpy(float))
        weights = result.get("member_weights", {})
        results[name] = (metrics, oof)
        print(f"\n=== {name}: {len(cols)} features, {len(oof)} OOF, "
              f"{len(grading)} grading, {time.time() - started:.0f}s")
        for k in ("logloss", "accuracy", "brier", "auc", "ece", "n"):
            if k in metrics:
                print(f"  {k:<10} {metrics[k]:.5f}")
        print(f"  weights   {({kk: round(float(v), 4) for kk, v in weights.items()})}")
        config.reset_feature_subset()

    outs = {}
    for name in ("A_total9", "B_pts100_9", "C_18"):
        o = results[name][1]
        o = o[o.grades_pooled].copy()
        outs[name] = o[["game_id", "gameday", "home_win",
                        "p_ensemble_calibrated"]]
    merged = (outs["A_total9"]
              .rename(columns={"p_ensemble_calibrated": "p_A"})
              .merge(outs["B_pts100_9"][["game_id", "p_ensemble_calibrated"]]
                     .rename(columns={"p_ensemble_calibrated": "p_B"}),
                     on="game_id", how="inner")
              .merge(outs["C_18"][["game_id", "p_ensemble_calibrated"]]
                     .rename(columns={"p_ensemble_calibrated": "p_C"}),
                     on="game_id", how="inner"))
    y = merged.home_win.to_numpy(float)
    gd = pd.to_datetime(merged.gameday)
    _paired(y, merged["p_A"].to_numpy(float), merged["p_B"].to_numpy(float),
            gd, "B_pts100_9 - A_total9", seed=config.RANDOM_SEED)
    _paired(y, merged["p_A"].to_numpy(float), merged["p_C"].to_numpy(float),
            gd, "C_18 - A_total9", seed=config.RANDOM_SEED)

    out = Path(config.RUN_DIAGNOSTICS_DIR) / "ab_darko_pts100_paired_oof.csv"
    merged.to_csv(out, index=False)
    print(f"\n  paired OOF written to {out}")
    print("\nAdopt bar: paired t <= -2.5 with star-out/tail slices not regressing.")


if __name__ == "__main__":
    main()
