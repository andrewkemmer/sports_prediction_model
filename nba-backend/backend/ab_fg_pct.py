"""Two-arm FG%: team expected field-goal percentage vs the v3.5 contract.

  A (control) : the served v3.5 contract (85 columns).
  B (additive): A + three pl_fg_pct_{away,home,diff} (minutes-weighted MEAN
                of per-player DARKO x_fg_pct over the priced healthy
                eligible members at final waterfall weights).

Design decisions (2026-11-01 discussion):
* Weights are expected minutes ONLY. Pace did unit-conversion work in O-PTS
  (per-100 -> per-minute); a rate needs no conversion, and within-team
  pace is ~constant so a pace factor would wash out in normalization.
* Denominator is priced-members-only weight. The _tblend zero-fill works
  for DPM (replacement level ~0) but 0% is not replacement level for
  shooting; unpriced members leave numerator AND denominator.
* No star leg in v1. The healthy-pool rate already prices the absence, and
  the contract already describes it three times over (weight_share, impact,
  star_out_pts). The star leg is pre-registered as a conditional follow-up:
  build it only if B clears the gate but the star-out slice fails.
* x_fg3_pct stays its own later arm; x_ft_pct skipped (high stability,
  tiny spreads, low expected signal).

Both arms share ONE projection (the production path, reweight="waterfall");
only the attached column set differs. Same folds (built once on A), same
frame, same designations, same hyperparameters.

Adopt bar: paired t <= -2.5 on per-game logloss with gameday-blocked
bootstrap CI excluding zero, star-out/tail slices not regressing.

Run where the cache lives:
    python nba-backend/backend/ab_fg_pct.py
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
logger = logging.getLogger("ab_fg_pct")

from ab_o_pts import _paired


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
    for c in ("x_fg_pct", "x_pts_100", "x_pace", "o_dpm", "d_dpm"):
        logger.info("%s coverage: %.3f", c,
                    float(ratings[c].notna().mean()) if c in ratings.columns else -1.0)
    if "x_fg_pct" not in ratings.columns:
        raise RuntimeError("ratings carry no x_fg_pct; apply_darko_rating did not attach it")

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
    if "team_fg_pct" not in agg0.columns:
        raise RuntimeError("projection carries no team_fg_pct; "
                           "the FG aggregate did not attach")
    logger.info("team_fg_pct aggregate coverage: %.3f",
                float(pd.to_numeric(
                    agg0["team_fg_pct"], errors="coerce").notna().mean()))
    logger.info("team_fg_pct aggregate mean: %.4f",
                float(pd.to_numeric(
                    agg0["team_fg_pct"], errors="coerce").mean()))

    # B: control frame plus the three (same join, same grain).
    frame_b = proj_mod.attach_fg_pct_features(frame_a, agg0)
    fg_cols = [c for c in config.FG_PCT_FEATURE_COLS if c in frame_b.columns]
    if len(fg_cols) != 3:
        raise RuntimeError(f"FG attach published {len(fg_cols)}/3: {fg_cols}")
    for c in fg_cols:
        logger.info("%s coverage: %.3f", c,
                    float(pd.to_numeric(frame_b[c],
                                        errors="coerce").notna().mean()))

    arms = {"A_contract": frame_a, "B_plus3": frame_b}
    fp = Path(config.RUN_DIAGNOSTICS_DIR) / "ab_fg_pct_B_features.parquet"
    fp.parent.mkdir(parents=True, exist_ok=True)
    frame_b.to_parquet(fp, index=False)
    logger.info("B_plus3: rows %d", len(frame_b))

    fold_list = folds_mod.make_folds(arms["A_contract"])
    logger.info("folds: %d", len(fold_list))
    a_cols = [c for c in config.MONEYLINE_FEATURE_COLS if c in frame_a.columns]
    arm_cols = {"A_contract": a_cols,
                "B_plus3": a_cols + fg_cols}
    results = {}
    for name in ("A_contract", "B_plus3"):
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
    treat_m = results["B_plus3"][0]
    print("\n=== side by side (B_plus3 minus A_contract)")
    for k in ("logloss", "accuracy", "brier", "auc", "ece"):
        if k in base_m and k in treat_m:
            print(f"  {k:<10} {base_m[k]:>10.5f} {treat_m[k]:>12.5f} "
                  f"{treat_m[k] - base_m[k]:>+10.5f}")

    outs = {}
    for name in ("A_contract", "B_plus3"):
        o = results[name][1]
        o = o[o.grades_pooled].copy()
        outs[name] = o[["game_id", "gameday", "home_win",
                        "p_ensemble_calibrated"]]
    merged = (outs["A_contract"]
              .rename(columns={"p_ensemble_calibrated": "p_A"})
              .merge(outs["B_plus3"][["game_id", "p_ensemble_calibrated"]]
                     .rename(columns={"p_ensemble_calibrated": "p_B"}),
                     on="game_id", how="inner"))
    y = merged.home_win.to_numpy(float)
    gd = pd.to_datetime(merged.gameday)
    mean, se, t, lo, hi, win, n = _paired(
        y, merged["p_A"].to_numpy(float), merged["p_B"].to_numpy(float),
        gd, "B_plus3 - A_contract", seed=config.RANDOM_SEED)

    # Slices: star-out games, low-confidence decile on the control probs,
    # and the top decile of |pl_fg_pct_diff| - where the feature should act.
    sm = merged.merge(
        arms["A_contract"][["game_id", "star_out_weight_share_diff"]]
        .assign(game_id=lambda df: df["game_id"].astype(str)),
        left_on=merged["game_id"].astype(str), right_on="game_id",
        how="left", suffixes=("", "_f"))
    sm = sm.merge(
        arms["B_plus3"][["game_id", "pl_fg_pct_diff"]]
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
    share = pd.to_numeric(sm["star_out_weight_share_diff"],
                          errors="coerce").fillna(0.0).abs().to_numpy() > 1e-9
    conf = np.abs(sm["p_A"].to_numpy(float) - 0.5)
    lo10 = float(np.quantile(conf, 0.10))
    fgspread = pd.to_numeric(sm["pl_fg_pct_diff"],
                             errors="coerce").abs().to_numpy()
    fghi = float(np.quantile(fgspread[np.isfinite(fgspread)], 0.90))
    slice_masks = {"starout_nonzero": share,
                   "lowconf_decile": conf <= lo10,
                   "fgspread_topdecile": fgspread >= fghi}
    slice_out = {}
    for sname, mask in slice_masks.items():
        dd = d_b[mask]
        if len(dd) < 30:
            print(f"\n--- slice B/{sname}: n={len(dd)} too thin, skipped")
            continue
        m, se_s = float(dd.mean()), float(dd.std(ddof=1) / np.sqrt(len(dd)))
        print(f"\n--- slice B/{sname}: n={len(dd)} mean={m:+.5f} "
              f"se={se_s:.5f} t={m / se_s:+.2f} "
              f"treat better {int((dd < 0).sum())}/{len(dd)} "
              f"({100 * (dd < 0).mean():.1f}%)")
        slice_out[sname] = {"n": len(dd), "mean": m, "se": se_s,
                            "t": m / se_s,
                            "better_share": float((dd < 0).mean())}

    # Diagnostic (IN-SAMPLE, not OOF): full-sample standardized logistic fit
    # to read whether the FG legs earn weight beside the rates.
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        full_b = arms["B_plus3"]
        gmask = results["B_plus3"][1].set_index("game_id")["grades_pooled"]
        gmask.index = gmask.index.astype(str)
        rows = full_b[full_b["game_id"].astype(str).isin(
            gmask[gmask.astype(bool)].index)].copy()
        yy = merged.set_index(merged["game_id"].astype(str)).loc[
            rows["game_id"].astype(str)]["home_win"].to_numpy(float)
        cols = [c for c in config.FG_PCT_FEATURE_COLS if c in rows.columns]
        XX = pd.to_numeric(rows[cols].stack(),
                           errors="coerce").unstack().fillna(0.0).to_numpy()
        Xs = StandardScaler().fit_transform(XX)
        clf = LogisticRegression(penalty="elasticnet", l1_ratio=0.5,
                                 C=0.03, solver="saga", max_iter=4000,
                                 tol=1e-4,
                                 random_state=config.RANDOM_SEED)
        clf.fit(Xs, yy)
        print("\n--- in-sample standardized logit coefs (B_new3)")
        for c, w in sorted(zip(cols, clf.coef_[0]),
                           key=lambda kv: -abs(kv[1])):
            print(f"    {c:28s} {w:+.4f}")
    except Exception as exc:  # noqa: BLE001 - diagnostic, never fatal
        print(f"\n(coef diagnostic skipped: {type(exc).__name__}: {exc})")

    out = Path(config.RUN_DIAGNOSTICS_DIR) / "ab_fg_pct_paired_oof.csv"
    merged.to_csv(out, index=False)
    print(f"\n  paired OOF written to {out}")
    verdict = {
        "date": "2026-11-01",
        "arms": {"A": "v3.5 contract (85)",
                 "B": "A + 3 pl_fg_pct (minutes-weighted mean, priced-only denominator, no star leg)"},
        "metrics": {k: {"A": results["A_contract"][0].get(k),
                        "B": results["B_plus3"][0].get(k)}
                    for k in ("logloss", "brier", "auc", "ece")},
        "paired_B_minus_A": {"mean": mean, "se": se, "t": t,
                             "ci95": [lo, hi], "better_share": win / max(n, 1),
                             "n": n},
        "slices": slice_out,
        "bar": -2.5,
        "adopted": bool(t <= -2.5),
    }
    import json
    vp = Path(config.RUN_DIAGNOSTICS_DIR) / "ab_fg_pct_verdict.json"
    json.dump(verdict, open(vp, "w"), indent=2)
    print(f"  verdict written to {vp}")
    print("\nAdopt bar: B-A paired t <= -2.5 with star-out/tail "
          "slices not regressing.")


if __name__ == "__main__":
    main()
