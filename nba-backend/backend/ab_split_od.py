"""Two-arm O/D split: nine total-DPM segments vs eighteen O/D segments.

  J0 (control, D config): 9 x pl_dpm_* from total dpm.
  J  (treatment)        : 18 x pl_odpm_*/pl_ddpm_* from o_dpm/d_dpm, built by
                         the IDENTICAL production blend with only the rating
                         column swapped (dpm = o+d to 1e-4, so replace nests
                         the incumbent for the linear member).

Both arms: expected weights, widened pool, gates on, waterfall, star-out
frozen on total DPM (availability control, not part of this question).
Same folds (built once on J0), same frame, same designations, same
hyperparameters.

Run where the cache lives:
    python nba-backend/backend/ab_split_od.py
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
logger = logging.getLogger("ab_split_od")

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

    config.PLAYER_USE_DARKO_EXPECTED_MINUTES = True
    config.PLAYER_POOL_WIDEN_DARKO = True
    config.PLAYER_POOL_IGNORE_FLOOR = False
    config.PLAYER_POOL_IGNORE_RECENCY = False

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
    for c in ("o_dpm", "d_dpm"):
        logger.info("%s coverage: %.3f", c,
                    float(ratings[c].notna().mean()) if c in ratings.columns else -1.0)

    name_by_id = dict(zip(
        rapm_mod._player_id_str(facts.player_stats.player_id),
        facts.player_stats.player_name.astype(str)))

    def prepared(swapped: pd.DataFrame) -> pd.DataFrame:
        rated = swapped.rename(columns={"target_date": "gameday"})
        rated["gameday"] = pd.to_datetime(rated["gameday"])
        if "is_available" not in rated.columns:
            rated["is_available"] = True
        return proj_mod.apply_pit_designations(rated, designations, name_by_id)

    rated_total = prepared(ratings.rename(columns={"target_date": "gameday"})
                           if "target_date" not in ratings.columns else ratings)
    # Control: production path, total DPM (nine + frozen star-out).
    frame0, agg0 = proj_mod.project_lineup_features(
        rated_total, games=base, designations=designations,
        name_by_id=name_by_id, reweight="waterfall")

    od_frames = {}
    for tag, col in (("odpm", "o_dpm"), ("ddpm", "d_dpm")):
        if col not in ratings.columns:
            raise RuntimeError(f"ratings carry no {col}; apply_darko_rating did not attach it")
        swapped = ratings.copy()
        vals = pd.to_numeric(swapped[col], errors="coerce")
        swapped["rapm_shrunk"] = vals.where(vals.notna(), swapped["rapm_shrunk"])
        filt = proj_mod.apply_pit_designations(
            swapped.rename(columns={"target_date": "gameday"})
            .assign(gameday=lambda df: pd.to_datetime(df["gameday"])),
            designations, name_by_id)
        agg = proj_mod.projected_lineup(filt, games=base, reweight="waterfall")
        attached = proj_mod.attach_position_rapm(base, agg)
        rename = {c: c.replace("pl_dpm_", f"pl_{tag}_", 1)
                  for c in POSITION_FEATURES if c in attached.columns}
        attached = attached.rename(columns=rename)
        keep = (["gameday", "home_team", "away_team"]
                + [c for c in rename.values() if c in attached.columns])
        od_frames[tag] = attached[keep].drop_duplicates(
            subset=["gameday", "home_team", "away_team"])
        cov = float(pd.to_numeric(
            attached[f"pl_{tag}_c_diff"], errors="coerce").notna().mean())
        logger.info("%s: pl_%s_c_diff coverage %.3f", tag, tag, cov)

    od_cols = list(config.SPLIT_OD_RAPM_FEATURE_COLS)
    missing = ([c for c in od_cols if c.startswith("pl_odpm_")
                and c not in od_frames["odpm"].columns]
               + [c for c in od_cols if c.startswith("pl_ddpm_")
                  and c not in od_frames["ddpm"].columns])
    if missing:
        raise RuntimeError(f"O/D attach missing columns: {missing}")

    key = ["gameday", "home_team", "away_team"]
    # Treatment: control frame minus the nine totals, plus the eighteen.
    fb_treat = frame0.drop(columns=[c for c in POSITION_FEATURES
                                    if c in frame0.columns], errors="ignore")
    fb_treat = fb_treat.merge(od_frames["odpm"], on=key, how="left")
    fb_treat = fb_treat.merge(od_frames["ddpm"], on=key, how="left",
                              suffixes=("", "_dup"))
    dupes = [c for c in fb_treat.columns if c.endswith("_dup")]
    fb_treat = fb_treat.drop(columns=dupes)
    arms = {"J0_total9": frame0, "J_split18": fb_treat}
    for name, fb in arms.items():
        fp = Path(config.RUN_DIAGNOSTICS_DIR) / f"ab_split_od_{name}_features.parquet"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fb.to_parquet(fp, index=False)
        logger.info("%s: rows %d", name, len(fb))

    fold_list = folds_mod.make_folds(arms["J0_total9"])
    logger.info("folds: %d", len(fold_list))
    baseline_cols = [c for c in config.MONEYLINE_FEATURE_COLS
                     if c not in set(POSITION_FEATURES)]
    arm_cols = {"J0_total9": baseline_cols + POSITION_FEATURES,
                "J_split18": baseline_cols + od_cols}
    results = {}
    for name in ("J0_total9", "J_split18"):
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

    b, t = results["J0_total9"][0], results["J_split18"][0]
    print("\n=== side by side (J_split18 minus J0_total9)")
    for k in ("logloss", "accuracy", "brier", "auc", "ece"):
        if k in b and k in t:
            print(f"  {k:<10} {b[k]:>10.5f} {t[k]:>12.5f} {t[k] - b[k]:>+10.5f}")

    bo, to = results["J0_total9"][1], results["J_split18"][1]
    bo = bo[bo.grades_pooled].copy()
    to = to[to.grades_pooled].copy()
    merged = (bo[["game_id", "gameday", "home_win", "p_ensemble_calibrated"]]
              .rename(columns={"p_ensemble_calibrated": "p_J0"})
              .merge(to[["game_id", "p_ensemble_calibrated"]]
                     .rename(columns={"p_ensemble_calibrated": "p_J"}),
                     on="game_id", how="inner"))
    y = merged.home_win.to_numpy(float)
    _paired(y, merged["p_J0"].to_numpy(float), merged["p_J"].to_numpy(float),
            pd.to_datetime(merged.gameday), "J_split18 - J0_total9",
            seed=config.RANDOM_SEED)

    # Slices: star-out games + low-confidence decile (baseline probs).
    sm = merged.merge(
        arms["J0_total9"][["game_id", "star_out_weight_share_diff"]]
        .assign(game_id=lambda df: df["game_id"].astype(str)),
        left_on=merged["game_id"].astype(str), right_on="game_id",
        how="left", suffixes=("", "_f"))
    yg = sm.home_win.to_numpy(float)
    eps = 1e-12

    def _pg(p):
        p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
        return -(yg * np.log(p) + (1 - yg) * np.log(1 - p))

    d_all = _pg(sm["p_J"].to_numpy(float)) - _pg(sm["p_J0"].to_numpy(float))
    share = pd.to_numeric(sm["star_out_weight_share_diff"],
                          errors="coerce").fillna(0.0).abs().to_numpy() > 1e-9
    conf = np.abs(sm["p_J0"].to_numpy(float) - 0.5)
    lo10 = float(np.quantile(conf, 0.10))
    for sname, mask in (("starout_nonzero", share),
                        ("lowconf_decile", conf <= lo10)):
        dd = d_all[mask]
        if len(dd) < 30:
            print(f"\n--- slice {sname}: n={len(dd)} too thin, skipped")
            continue
        m, se = float(dd.mean()), float(dd.std(ddof=1) / np.sqrt(len(dd)))
        print(f"\n--- slice {sname}: n={len(dd)} mean={m:+.5f} se={se:.5f} "
              f"t={m / se:+.2f} J better {int((dd < 0).sum())}/{len(dd)} "
              f"({100 * (dd < 0).mean():.1f}%)")

    # Diagnostic (IN-SAMPLE, not OOF): full-sample standardized logistic fit
    # per arm to read whether the elastic net splits coef_O vs coef_D.
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        full = arms["J_split18"]
        gmask = results["J_split18"][1].set_index("game_id")["grades_pooled"]
        gmask.index = gmask.index.astype(str)
        rows = full[full["game_id"].astype(str).isin(
            gmask[gmask.astype(bool)].index)].copy()
        yy = merged.set_index(merged["game_id"].astype(str)).loc[
            rows["game_id"].astype(str)]["home_win"].to_numpy(float)
        for aname, acols in (("J0", POSITION_FEATURES), ("J", od_cols)):
            cols = [c for c in acols if c in rows.columns]
            XX = pd.to_numeric(rows[cols].stack(),
                               errors="coerce").unstack().fillna(0.0).to_numpy()
            Xs = StandardScaler().fit_transform(XX)
            clf = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.03,
                                     solver="saga", max_iter=4000, tol=1e-4,
                                     random_state=config.RANDOM_SEED)
            clf.fit(Xs, yy)
            print(f"\n--- in-sample standardized logit coefs ({aname})")
            for c, w in sorted(zip(cols, clf.coef_[0]), key=lambda kv: -abs(kv[1])):
                print(f"    {c:22s} {w:+.4f}")
    except Exception as exc:  # noqa: BLE001 - diagnostic, never fatal
        print(f"\n(coef diagnostic skipped: {type(exc).__name__}: {exc})")

    out = Path(config.RUN_DIAGNOSTICS_DIR) / "ab_split_od_paired_oof.csv"
    merged.to_csv(out, index=False)
    print(f"\n  paired OOF written to {out}")
    print("\nAdopt bar: J-J0 paired t <= -2.5 with coef_O != coef_D and "
          "star-out/tail slices not regressing.")


if __name__ == "__main__":
    main()
