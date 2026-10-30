"""Six-arm pool ablation: membership gates x opportunity weights.

  A (incumbent) : trailing weights, legacy pool (floor + recency, widen off).
  B (old B)      : expected weights, legacy pool.
  D (widen)      : expected weights, widened pool (x >= 5).
  E (no floor)   : expected + widen, prior-minutes floor -> 0.
  F (no recency) : expected + widen, 30-day recency gate off.
  G (no gates)   : expected + widen, both gates off.

All arms `reweight="waterfall"`, same folds (built once on A), same frame,
same ratings, same designations, same hyperparameters.

Run where the cache lives:
    python nba-backend/backend/ab_pool_gates.py
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
logger = logging.getLogger("ab_pool_gates")

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
    return mean, se, t, lo, hi, win, n


def main() -> None:
    import config
    import folds as folds_mod
    import features as feat_mod
    import ingestion
    import lineup_projection as proj_mod
    import moneyline as ml_mod

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
    # (name, expected, widen, no_floor, no_recency)
    spec = (
        ("A_incumbent", False, False, False, False),
        ("B_expected", True, False, False, False),
        ("D_widen", True, True, False, False),
        ("E_nofloor", True, True, True, False),
        ("F_norecency", True, True, False, True),
        ("G_nogates", True, True, True, True),
    )
    arms, covs = {}, {}
    for name, exp, widen, nofloor, norec in spec:
        config.PLAYER_USE_DARKO_EXPECTED_MINUTES = exp
        config.PLAYER_POOL_WIDEN_DARKO = widen
        config.PLAYER_POOL_IGNORE_FLOOR = nofloor
        config.PLAYER_POOL_IGNORE_RECENCY = norec
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
        fp = Path(config.RUN_DIAGNOSTICS_DIR) / f"ab_pool_gates_{name}_features.parquet"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fb.to_parquet(fp, index=False)
        cov = float(pd.to_numeric(
            fb["pl_dpm_c_diff"], errors="coerce").notna().mean())
        covs[name] = cov
        logger.info("%s (exp=%s widen=%s nofloor=%s norec=%s): c cov %.3f, rows %d",
                    name, exp, widen, nofloor, norec, cov, len(fb))

    fold_list = folds_mod.make_folds(arms["A_incumbent"])
    logger.info("folds: %d", len(fold_list))
    baseline_cols = [c for c in config.MONEYLINE_FEATURE_COLS
                     if c not in set(POSITION_FEATURES)]
    results = {}
    names = [s[0] for s in spec]
    for name in names:
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

    print("\n=== side by side (pooled grading logloss)")
    print(f"  {'metric':<10} " + " ".join(f"{n:>16}" for n in names))
    for k in ("logloss", "brier", "auc", "ece"):
        print(f"  {k:<10} " + " ".join(
            f"{results[n][0].get(k, float('nan')):>16.5f}" for n in names))
    print("  c-coverage  " + " ".join(f"{covs[n]:>16.3f}" for n in names))

    oofs = {}
    for n in names:
        o = results[n][1]
        o = o[o.grades_pooled].copy()
        oofs[n] = o[["game_id", "gameday", "home_win", "p_ensemble_calibrated"]]
    merged = oofs[names[0]].rename(columns={"p_ensemble_calibrated": f"p_{names[0]}"})
    for n in names[1:]:
        merged = merged.merge(
            oofs[n][["game_id", "p_ensemble_calibrated"]].rename(
                columns={"p_ensemble_calibrated": f"p_{n}"}),
            on="game_id", how="inner")
    y = merged.home_win.to_numpy(float)
    gd = pd.to_datetime(merged.gameday)
    for lb, lt in (("A_incumbent", "B_expected"), ("A_incumbent", "D_widen"),
                   ("B_expected", "D_widen"), ("D_widen", "E_nofloor"),
                   ("D_widen", "F_norecency"), ("D_widen", "G_nogates"),
                   ("A_incumbent", "G_nogates")):
        _paired(y, merged[f"p_{lb}"].to_numpy(float),
                merged[f"p_{lt}"].to_numpy(float), gd, f"{lt} - {lb}",
                seed=config.RANDOM_SEED)
    out = Path(config.RUN_DIAGNOSTICS_DIR) / "ab_pool_gates_paired_oof.csv"
    merged.to_csv(out, index=False)
    print(f"\n  paired OOF written to {out}")
    for n in names:
        p = Path(config.RUN_DIAGNOSTICS_DIR) / f"ab_pool_gates_{n}_oof.csv"
        results[n][1].to_csv(p, index=False)
    print(f"  per-arm OOFs written to {config.RUN_DIAGNOSTICS_DIR}")
    print("\nAdopt bar: any gate-removal arm needs paired t <= -2.5 vs D_widen "
          "with star-out/tail slices not regressing.")


if __name__ == "__main__":
    main()
