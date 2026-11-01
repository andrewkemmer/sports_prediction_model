"""Three-arm O-PTS: expected-points levels vs the crossmatch rates.

  A (control) : the served v3.4 contract - eleven crossmatch
                (pl_o_dpm x3, pl_d_dpm x3, pace triplet, pl_net_o x2) plus
                the six star_out_*.
  B (additive): A + six pl_o_pts_*/star_out_pts_* (team expected points =
                SUM over healthy eligible of final opportunity weight x
                per-minute scoring rate; star-out leg = expected points of
                the OUT top-three reference members).
  C (replace) : A minus the eleven (pl_o_dpm/pl_d_dpm/star_out_impact/
                pl_net_o) plus the six - pace triplet and weight-share stay.

All three arms share ONE projection (the production path,
reweight="waterfall"); only the attached column set differs. Same folds
(built once on A), same frame, same designations, same hyperparameters.

Adopt bar: paired t <= -2.5 on per-game logloss with gameday-blocked
bootstrap CI excluding zero, star-out/tail slices not regressing.

FROZEN 2026-11-01: the B arm was adopted into the served contract by owner directive (full-window t -2.04, edge-vs-close t -2.40 - under the bar; C rejected harder). Post-adoption arm A IS the old arm B, so a re-run of this harness measures nothing (B-A == 0 by construction). This file and ab_o_pts_verdict.json are the trial record; do not re-verify against the new contract.

Run where the cache lives:
    python nba-backend/backend/ab_o_pts.py
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
logger = logging.getLogger("ab_o_pts")

#: The eleven columns arm C removes: the six pace-scaled OFF/DEF legs, the
#: two cross-matched nets, and the three star-out impact legs. The pace
#: triplet and star_out_weight_share stay in every arm.
REPLACE_11 = (
    [f"pl_o_dpm_{s}" for s in ("away", "home", "diff")]
    + [f"pl_d_dpm_{s}" for s in ("away", "home", "diff")]
    + ["pl_net_o_home", "pl_net_o_away"]
    + [f"star_out_impact_{s}" for s in ("away", "home", "diff")])


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
    from ab_position_rapm import score

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
    for c in ("x_pts_100", "x_pace", "o_dpm", "d_dpm"):
        logger.info("%s coverage: %.3f", c,
                    float(ratings[c].notna().mean()) if c in ratings.columns else -1.0)

    name_by_id = dict(zip(
        rapm_mod._player_id_str(facts.player_stats.player_id),
        facts.player_stats.player_name.astype(str)))

    if "target_date" in ratings.columns:
        rated = ratings.rename(columns={"target_date": "gameday"})
    else:
        rated = ratings.copy()
    rated["gameday"] = pd.to_datetime(rated["gameday"])
    if "is_available" not in rated.columns:
        rated["is_available"] = True

    # A: production path (contract frame + aggregates share one projection).
    frame_a, agg0 = proj_mod.project_lineup_features(
        rated, games=base, designations=designations,
        name_by_id=name_by_id, reweight="waterfall")
    missing_agg = [c for c in ("team_o_pts", "star_out_pts")
                   if c not in agg0.columns]
    if missing_agg:
        raise RuntimeError(f"projection carries no {missing_agg}; "
                           "the O-PTS aggregates did not attach")
    for c in ("team_o_pts", "star_out_pts"):
        logger.info("%s aggregate coverage: %.3f",
                    c, float(pd.to_numeric(
                        agg0[c], errors="coerce").notna().mean()))

    # B: control frame plus the six (same join, same grain).
    frame_b = proj_mod.attach_o_pts_features(frame_a, agg0)
    o_pts_cols = [c for c in config.O_PTS_FEATURE_COLS if c in frame_b.columns]
    if len(o_pts_cols) != 6:
        raise RuntimeError(f"O-PTS attach published {len(o_pts_cols)}/6: "
                           f"{o_pts_cols}")
    for c in o_pts_cols:
        logger.info("%s coverage: %.3f", c,
                    float(pd.to_numeric(frame_b[c],
                                        errors="coerce").notna().mean()))

    # C: control minus the eleven, plus the six.
    key = ["gameday", "home_team", "away_team"]
    keep_b = (key + [c for c in o_pts_cols if c in frame_b.columns])
    pts_only = frame_b[keep_b].drop_duplicates(
        subset=["gameday", "home_team", "away_team"])
    frame_c = frame_a.drop(columns=[c for c in REPLACE_11
                                    if c in frame_a.columns], errors="ignore")
    frame_c = frame_c.merge(pts_only, on=key, how="left",
                            suffixes=("", "_dup"))
    frame_c = frame_c.drop(columns=[c for c in frame_c.columns
                                    if c.endswith("_dup")])
    gone = [c for c in REPLACE_11 if c in frame_c.columns]
    if gone:
        raise RuntimeError(f"C arm still carries replaced columns: {gone}")

    arms = {"A_contract": frame_a, "B_plus6": frame_b, "C_replace": frame_c}
    for name, fb in arms.items():
        fp = Path(config.RUN_DIAGNOSTICS_DIR) / f"ab_o_pts_{name}_features.parquet"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fb.to_parquet(fp, index=False)
        logger.info("%s: rows %d", name, len(fb))

    fold_list = folds_mod.make_folds(arms["A_contract"])
    logger.info("folds: %d", len(fold_list))
    a_cols = [c for c in config.MONEYLINE_FEATURE_COLS if c in frame_a.columns]
    arm_cols = {"A_contract": a_cols,
                "B_plus6": a_cols + o_pts_cols,
                "C_replace": ([c for c in a_cols if c not in set(REPLACE_11)]
                               + o_pts_cols)}
    results = {}
    for name in ("A_contract", "B_plus6", "C_replace"):
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

    base_m = results["A_contract"][0]
    print("\n=== side by side (treat minus A_contract)")
    for name in ("B_plus6", "C_replace"):
        tm = results[name][0]
        print(f"--- {name}")
        for k in ("logloss", "accuracy", "brier", "auc", "ece"):
            if k in base_m and k in tm:
                print(f"  {k:<10} {base_m[k]:>10.5f} {tm[k]:>12.5f} "
                      f"{tm[k] - base_m[k]:>+10.5f}")

    outs = {}
    for name in ("A_contract", "B_plus6", "C_replace"):
        o = results[name][1]
        o = o[o.grades_pooled].copy()
        outs[name] = o[["game_id", "gameday", "home_win",
                        "p_ensemble_calibrated"]]
    merged = (outs["A_contract"]
              .rename(columns={"p_ensemble_calibrated": "p_A"})
              .merge(outs["B_plus6"][["game_id", "p_ensemble_calibrated"]]
                     .rename(columns={"p_ensemble_calibrated": "p_B"}),
                     on="game_id", how="inner")
              .merge(outs["C_replace"][["game_id", "p_ensemble_calibrated"]]
                     .rename(columns={"p_ensemble_calibrated": "p_C"}),
                     on="game_id", how="inner"))
    y = merged.home_win.to_numpy(float)
    gd = pd.to_datetime(merged.gameday)
    _paired(y, merged["p_A"].to_numpy(float), merged["p_B"].to_numpy(float),
            gd, "B_plus6 - A_contract", seed=config.RANDOM_SEED)
    _paired(y, merged["p_A"].to_numpy(float), merged["p_C"].to_numpy(float),
            gd, "C_replace - A_contract", seed=config.RANDOM_SEED)

    # Slices: star-out games (weight-share and pts legs), low-confidence
    # decile on the control probs. The C arm drops star_out_impact, so the
    # star-out slices are its load-bearing check.
    sm = merged.merge(
        arms["A_contract"][["game_id", "star_out_weight_share_diff"]]
        .assign(game_id=lambda df: df["game_id"].astype(str)),
        left_on=merged["game_id"].astype(str), right_on="game_id",
        how="left", suffixes=("", "_f"))
    sm = sm.merge(
        arms["B_plus6"][["game_id", "star_out_pts_diff"]]
        .assign(game_id=lambda df: df["game_id"].astype(str))
        .rename(columns={"game_id": "game_id_p"}),
        left_on=sm["game_id"].astype(str), right_on="game_id_p",
        how="left")
    yg = sm.home_win.to_numpy(float)
    eps = 1e-12

    def _pg(p):
        p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
        return -(yg * np.log(p) + (1 - yg) * np.log(1 - p))

    d_b = _pg(sm["p_B"].to_numpy(float)) - _pg(sm["p_A"].to_numpy(float))
    d_c = _pg(sm["p_C"].to_numpy(float)) - _pg(sm["p_A"].to_numpy(float))
    share = pd.to_numeric(sm["star_out_weight_share_diff"],
                          errors="coerce").fillna(0.0).abs().to_numpy() > 1e-9
    ptsout = pd.to_numeric(sm["star_out_pts_diff"],
                           errors="coerce").fillna(0.0).abs().to_numpy() > 1e-9
    conf = np.abs(sm["p_A"].to_numpy(float) - 0.5)
    lo10 = float(np.quantile(conf, 0.10))
    for sname, mask in (("starout_nonzero", share),
                        ("starout_pts_nonzero", ptsout),
                        ("lowconf_decile", conf <= lo10)):
        for tname, dd in ((f"B/{sname}", d_b[mask]),
                          (f"C/{sname}", d_c[mask])):
            if len(dd) < 30:
                print(f"\n--- slice {tname}: n={len(dd)} too thin, skipped")
                continue
            m, se = float(dd.mean()), float(dd.std(ddof=1) / np.sqrt(len(dd)))
            print(f"\n--- slice {tname}: n={len(dd)} mean={m:+.5f} "
                  f"se={se:.5f} t={m / se:+.2f} "
                  f"treat better {int((dd < 0).sum())}/{len(dd)} "
                  f"({100 * (dd < 0).mean():.1f}%)")

    # Diagnostic (IN-SAMPLE, not OOF): full-sample standardized logistic fit
    # per arm to read whether the points legs earn weight beside the rates.
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        full_b = arms["B_plus6"]
        gmask = results["B_plus6"][1].set_index("game_id")["grades_pooled"]
        gmask.index = gmask.index.astype(str)
        rows = full_b[full_b["game_id"].astype(str).isin(
            gmask[gmask.astype(bool)].index)].copy()
        yy = merged.set_index(merged["game_id"].astype(str)).loc[
            rows["game_id"].astype(str)]["home_win"].to_numpy(float)
        for aname, acols in (("B_new6", list(config.O_PTS_FEATURE_COLS)),
                             ("B_replaced11", list(REPLACE_11))):
            cols = [c for c in acols if c in rows.columns]
            XX = pd.to_numeric(rows[cols].stack(),
                               errors="coerce").unstack().fillna(0.0).to_numpy()
            Xs = StandardScaler().fit_transform(XX)
            clf = LogisticRegression(penalty="elasticnet", l1_ratio=0.5,
                                     C=0.03, solver="saga", max_iter=4000,
                                     tol=1e-4,
                                     random_state=config.RANDOM_SEED)
            clf.fit(Xs, yy)
            print(f"\n--- in-sample standardized logit coefs ({aname})")
            for c, w in sorted(zip(cols, clf.coef_[0]),
                               key=lambda kv: -abs(kv[1])):
                print(f"    {c:28s} {w:+.4f}")
    except Exception as exc:  # noqa: BLE001 - diagnostic, never fatal
        print(f"\n(coef diagnostic skipped: {type(exc).__name__}: {exc})")

    out = Path(config.RUN_DIAGNOSTICS_DIR) / "ab_o_pts_paired_oof.csv"
    merged.to_csv(out, index=False)
    print(f"\n  paired OOF written to {out}")
    print("\nAdopt bar: B-A or C-A paired t <= -2.5 with star-out/tail "
          "slices not regressing.")


if __name__ == "__main__":
    main()
