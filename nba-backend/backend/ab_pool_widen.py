"""Holdout B/D: legacy trailing-gates pool vs DARKO-widened pool.

  B (control): `PLAYER_POOL_WIDEN_DARKO = False` (trailing 50-min floor +
               30-day recency only).
  D (treatment): widened - same-day Darko x_minutes >= PLAYER_POOL_WIDEN_MIN_X
               admits debutants / call-ups / mid-season returnees.

Both arms run under the shipping opportunity weights
(`PLAYER_USE_DARKO_EXPECTED_MINUTES = True`, the code default) and
`reweight="waterfall"`. Same folds (built once on B), same game frame, same
ratings, same designations, same hyperparameters.

Run where the cache lives:
    python nba-backend/backend/ab_pool_widen.py
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
logger = logging.getLogger("ab_pool_widen")

from ab_position_rapm import (  # noqa: E402
    POSITION_FEATURES,
    score,
)


def _paired(y, p_base, p_treat, gamedays, label, seed=42):
    eps = 1e-12

    def per_game(p):
        p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    d = per_game(p_treat) - per_game(p_base)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    t = mean / se if se and np.isfinite(se) and se != 0 else float("nan")
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"gameday": pd.to_datetime(gamedays), "_i": np.arange(n)})
    blocks = [part["_i"].to_numpy() for _, part in df.groupby("gameday")]
    boot = np.array([d[np.concatenate(
        [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])].mean()
        for _ in range(4000)])
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    win = int((d < 0).sum())
    print(f"\n=== paired delta ({label}), n={n}")
    print(f"  mean {mean:+.5f}  se {se:.5f}  t {t:+.2f}  "
          f"95% CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  verdict: {'distinguishable' if lo * hi > 0 else 'NOT distinguishable'}")
    print(f"  treat better on {win}/{n} ({100 * win / max(n, 1):.1f}%)")
    return {"label": label, "n": n, "mean": mean, "se": se, "t": t,
            "lo": lo, "hi": hi, "win": win}


def main() -> None:
    import config
    import folds as folds_mod
    import features as feat_mod
    import ingestion
    import lineup_projection as proj_mod
    import moneyline as ml_mod

    config.PLAYER_USE_DARKO_EXPECTED_MINUTES = True
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
    logger.info("designations: %s",
                "none" if designations is None else f"{len(designations)} rows")

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
    logger.info("x coverage: %.3f",
                float(ratings["darko_x_minutes"].notna().mean()))

    name_by_id = dict(zip(
        rapm_mod._player_id_str(facts.player_stats.player_id),
        facts.player_stats.player_name.astype(str)))

    treat_cols = list(POSITION_FEATURES) + list(proj_mod.STAR_OUT_FEATURES)
    arms = {}
    for name, widen in (("B_legacy_pool", False), ("D_widened_pool", True)):
        config.PLAYER_POOL_WIDEN_DARKO = widen
        rated = ratings.rename(columns={"target_date": "gameday"})
        rated["gameday"] = pd.to_datetime(rated["gameday"])
        if "is_available" not in rated.columns:
            rated["is_available"] = True
        frame, aggregates = proj_mod.project_lineup_features(
            rated, games=base, designations=designations,
            name_by_id=name_by_id, reweight="waterfall")
        fb = frame.drop(columns=[c for c in treat_cols if c in frame.columns],
                        errors="ignore")
        key = ["gameday", "home_team", "away_team"]
        fb = fb.merge(
            frame[key + [c for c in treat_cols if c in frame.columns]]
            .drop_duplicates(subset=key),
            on=key, how="left")
        arms[name] = fb
        fp = Path(config.RUN_DIAGNOSTICS_DIR) / f"ab_pool_widen_{name}_features.parquet"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fb.to_parquet(fp, index=False)
        cov = float(pd.to_numeric(
            fb["pl_dpm_c_diff"], errors="coerce").notna().mean())
        logger.info("%s (widen=%s): pl_dpm_c_diff coverage %.3f, rows %d",
                    name, widen, cov, len(fb))

    fold_list = folds_mod.make_folds(arms["B_legacy_pool"])
    logger.info("folds: %d", len(fold_list))
    baseline_cols = [c for c in config.MONEYLINE_FEATURE_COLS
                     if c not in set(POSITION_FEATURES)]
    results = {}
    for name in ("B_legacy_pool", "D_widened_pool"):
        joined = arms[name]
        cols = baseline_cols + POSITION_FEATURES
        cols = [c for c in cols if c in joined.columns]
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

    b, d = results["B_legacy_pool"][0], results["D_widened_pool"][0]
    print("\n=== side by side (D minus B)")
    for k in ("logloss", "accuracy", "brier", "auc", "ece"):
        if k in b and k in d:
            print(f"  {k:<10} {b[k]:>10.5f} {d[k]:>12.5f} {d[k] - b[k]:>+10.5f}")

    bo, do = results["B_legacy_pool"][1], results["D_widened_pool"][1]
    bo = bo[bo.grades_pooled].copy()
    do = do[do.grades_pooled].copy()
    merged = (bo[["game_id", "gameday", "home_win", "p_ensemble_calibrated"]]
              .rename(columns={"p_ensemble_calibrated": "p_B"})
              .merge(do[["game_id", "p_ensemble_calibrated"]]
                     .rename(columns={"p_ensemble_calibrated": "p_D"}),
                     on="game_id", how="inner"))
    y = merged.home_win.to_numpy(float)
    _paired(y, merged["p_B"].to_numpy(float), merged["p_D"].to_numpy(float),
            pd.to_datetime(merged.gameday), "D_widened - B_legacy",
            seed=config.RANDOM_SEED)

    # Widening-intensity slice: games where D's features actually moved.
    fb_ = pd.read_parquet(
        Path(config.RUN_DIAGNOSTICS_DIR) / "ab_pool_widen_D_widened_pool_features.parquet",
        columns=["game_id", "pl_dpm_c_diff", "pl_dpm_f_diff", "pl_dpm_g_diff"])
    fb_["game_id"] = fb_["game_id"].astype(str)
    merged["game_id"] = merged["game_id"].astype(str)
    sm = merged.merge(fb_, on="game_id", how="left")
    yg = sm.home_win.to_numpy(float)
    eps = 1e-12

    def _pg(p):
        p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
        return -(yg * np.log(p) + (1 - yg) * np.log(1 - p))

    d_all = _pg(sm["p_D"].to_numpy(float)) - _pg(sm["p_B"].to_numpy(float))
    moved = (pd.to_numeric(sm["pl_dpm_c_diff"], errors="coerce").notna()
             | pd.to_numeric(sm["pl_dpm_f_diff"], errors="coerce").notna()
             | pd.to_numeric(sm["pl_dpm_g_diff"], errors="coerce").notna()).to_numpy()
    # Games priced only because widening admitted someone: compare B NaN vs D finite.
    fbB = pd.read_parquet(
        Path(config.RUN_DIAGNOSTICS_DIR) / "ab_pool_widen_B_legacy_pool_features.parquet",
        columns=["game_id", "pl_dpm_c_diff", "pl_dpm_f_diff", "pl_dpm_g_diff"])
    fbB["game_id"] = fbB["game_id"].astype(str)
    cm = merged.merge(fbB, on="game_id", how="left", suffixes=("", "_B")).merge(
        fb_, on="game_id", how="left", suffixes=("", "_D"))
    widened_only = np.zeros(len(cm), dtype=bool)
    for pos in ("c", "f", "g"):
        bcol, dcol = f"pl_dpm_{pos}_diff", f"pl_dpm_{pos}_diff_D"
        if bcol in cm.columns and dcol in cm.columns:
            bv = pd.to_numeric(cm[bcol], errors="coerce")
            dv = pd.to_numeric(cm[dcol], errors="coerce")
            widened_only |= (bv.isna() & dv.notna()).to_numpy()
    dd = _pg(cm["p_D"].to_numpy(float)) - _pg(cm["p_B"].to_numpy(float))
    n = len(dd)
    print(f"\n--- widened-only games (NaN in B, finite in D): "
          f"{int(widened_only.sum())}/{n}")
    if widened_only.sum() >= 30:
        s = dd[widened_only]
        print(f"    mean {s.mean():+.5f} se {s.std(ddof=1) / np.sqrt(len(s)):.5f} "
              f"t {s.mean() / (s.std(ddof=1) / np.sqrt(len(s))):+.2f}")
    out = Path(config.RUN_DIAGNOSTICS_DIR) / "ab_pool_widen_paired_oof.csv"
    merged.to_csv(out, index=False)
    print(f"\n  paired OOF written to {out}")
    print("\nAdopt bar: D-B paired t <= -2.5 with the widened-only slice not worse.")


if __name__ == "__main__":
    main()
