"""NFL production master pipeline — the ONE authoritative entry point.

Sequence (spec section 31):
  1. configuration         8. ensemble/calibration
  2. eligible ingestion    9. evaluation/diagnostics
  3. point-in-time features 10. final full-history fit
  4. fold generation       11. current-slate serving
  5. moneyline OOF         12. artifact persistence
  6. run-line OOF          13. schema validation
  7. totals OOF            14. monitoring

Zero dependency on obsolete NFL research/production modules. Market-free.
Run:  python3 master_pipeline.py            (from nfl-backend/backend/)
      python3 master_pipeline.py --skip-pull   (use cached nflverse pulls)
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config  # noqa: E402
import ingestion  # noqa: E402
import features as feat_mod  # noqa: E402
import folds as folds_mod  # noqa: E402
import moneyline as ml_mod  # noqa: E402
import distributions as dist_mod  # noqa: E402
import evaluation as eval_mod  # noqa: E402
import serving as serve_mod  # noqa: E402
import qb_enrichment  # noqa: E402
import monitoring  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("nfl_master_pipeline")


def _banner(phase: str, msg: str = "") -> None:
    print(f"\n{'━' * 70}\n  {phase} — {msg}\n{'━' * 70}\n", flush=True)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="NFL production master pipeline")
    ap.add_argument("--skip-pull", action="store_true",
                    help="use cached nflverse pulls (no network)")
    ap.add_argument("--out-dir", default=None,
                    help="override artifact output directory")
    args = ap.parse_args(argv)

    t0 = time.time()
    out_dir = Path(args.out_dir) if args.out_dir else config.DATA_DELIVERY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_c = run_date.replace("-", "")
    config_meta = {
        "feature_set_version": config.FEATURE_SET_VERSION,
        "warmup_seasons": config.WARMUP_SEASONS,
        "oof_first_season": config.OOF_FIRST_SEASON,
        "retrain_cadence_days": config.RETRAIN_CADENCE_DAYS,
        "ensemble_members": config.ENSEMBLE_MEMBERS,
        "random_seed": config.RANDOM_SEED,
        "market_independence": True,
    }

    # ── 2. Eligible game ingestion ────────────────────────────────────────
    _banner("PHASE 2", "nflverse ingestion")
    schedule = ingestion.load_schedule(use_cache=not args.skip_pull and False)
    schedule = ingestion.eligible_games(schedule)
    logger.info("schedule rows (eligible seasons): %d", len(schedule))
    pbp = ingestion.load_pbp(use_cache=True)
    logger.info("pbp rows: %s", 0 if pbp is None else len(pbp))

    decided_all = schedule[schedule["home_score"].notna()
                           & schedule["away_score"].notna()].copy()
    warmup = decided_all[pd.to_numeric(decided_all["season"]) < config.OOF_FIRST_SEASON]
    core = decided_all[pd.to_numeric(decided_all["season"]) >= config.OOF_FIRST_SEASON]
    logger.info("decided games: warmup %d, core (OOF population) %d",
                len(warmup), len(core))

    # ── 3. Point-in-time features ─────────────────────────────────────────
    _banner("PHASE 3", "point-in-time feature engine")
    game_df = feat_mod.build_game_features(decided_all, pbp)
    game_df = game_df.sort_values("gameday").reset_index(drop=True)
    logger.info("feature frame: %d decided games, %d columns",
                len(game_df), game_df.shape[1])
    cov = feat_mod.feature_coverage_report(game_df)
    logger.info("feature coverage:\n%s", cov.to_string(index=False))

    # ── 4. Fold generation ────────────────────────────────────────────────
    _banner("PHASE 4", "walk-forward fold generation")
    fold_list = folds_mod.make_folds(game_df, date_col="gameday")
    fold_info = folds_mod.fold_summary(fold_list)
    fold_tbl = folds_mod.fold_table(game_df, fold_list, date_col="gameday")
    if not fold_list:
        raise RuntimeError(
            "PHASE 4 produced ZERO folds — walk-forward OOF cannot proceed. "
            "Check eligible-game ingestion and OOF_FIRST_SEASON geometry.")
    first_val_season = pd.to_numeric(
        game_df.loc[fold_list[0].val_idx, "season"]).min() if fold_list else None
    if first_val_season != config.OOF_FIRST_SEASON:
        raise RuntimeError(
            f"fold geometry violation: first validation season {first_val_season} "
            f"!= OOF_FIRST_SEASON {config.OOF_FIRST_SEASON}")
    # Population reconciliation gate: every eligible 2019+ game must appear
    # in exactly one validation window, and nothing else may.
    _val_ids = [gid for f in fold_list for gid in game_df.loc[f.val_idx, "game_id"]]
    _core_pop = int((pd.to_numeric(game_df["season"]) >= config.OOF_FIRST_SEASON).sum())
    if len(_val_ids) != len(set(_val_ids)):
        raise RuntimeError(
            f"fold population violation: {len(_val_ids) - len(set(_val_ids))} "
            "duplicated validation game IDs across folds")
    if len(set(_val_ids)) != _core_pop:
        raise RuntimeError(
            f"fold population violation: {len(set(_val_ids))} unique validation "
            f"game IDs != {_core_pop} eligible {config.OOF_FIRST_SEASON}+ games")
    # Meaningful, visible Phase 4 report (print → stdout, same stream as the
    # phase banners; logger goes to stderr and was easy to miss).
    gd_dates = pd.to_datetime(game_df["gameday"])
    print(f"  eligible settled games : {len(game_df)}")
    print(f"  game date range        : {gd_dates.min().date()} .. {gd_dates.max().date()}")
    print(f"  warmup seasons         : {config.WARMUP_SEASONS} (training-only)")
    print(f"  first OOF validation   : {fold_list[0].val_start.date()} "
          f"(fold 0, n_val={len(fold_list[0].val_idx)})")
    print(f"  last OOF validation    : {fold_list[-1].val_end.date()} "
          f"(fold {fold_list[-1].fold_id}, n_val={len(fold_list[-1].val_idx)})")
    print(f"  validation windows     : {len(fold_list)} "
          f"(7-calendar-day, non-overlapping, expanding training)")
    print(f"  training observations  : {fold_info['min_train']} (first) .. "
          f"{fold_info['max_train']} (last)")
    print(f"  validation observations: {fold_info['total_val_games']}")
    nv_arr = fold_tbl["n_validation"].to_numpy()
    print(f"  n_validation min/max/mean/median: {int(nv_arr.min())} / {int(nv_arr.max())} / "
          f"{round(float(nv_arr.mean()), 3)} / {float(np.median(nv_arr))} "
          "(windows are 7 CALENDAR days — game counts vary per window)")
    print("  complete n_validation distribution:")
    for k, v in fold_tbl["n_validation"].value_counts().sort_index().items():
        print(f"    {int(k):3d} games: {int(v)} folds")
    print("  representative folds (first 3 / last 3):")
    show = list(range(min(3, len(fold_tbl)))) + \
        list(range(max(3, len(fold_tbl) - 3), len(fold_tbl)))
    for i in show:
        r = fold_tbl.iloc[i]
        print(f"    fold {int(r['fold_id']):3d}  train_end={r['train_end_date']}  "
              f"val=[{r['validation_start']} .. {r['validation_end']}]  "
              f"n_train={int(r['n_train']):5d}  n_val={int(r['n_validation'])}")
    # Persist the per-fold table so Phase 4 output is inspectable post-run.
    fold_tbl.to_csv(out_dir / "nfl_fold_table.csv", index=False)
    logger.info("folds: %s", json.dumps(fold_info))

    # ── 5. Moneyline OOF ──────────────────────────────────────────────────
    _banner("PHASE 5", "moneyline walk-forward OOF")
    ml = ml_mod.walk_forward_oof(game_df, fold_list=fold_list)
    oof_ml = ml["oof"]
    weights = ml["member_weights"]
    logger.info("moneyline OOF rows: %d; adaptive weights: %s",
                len(oof_ml), json.dumps({k: round(v, 3) for k, v in weights.items()}))

    # join targets/sides onto the OOF rows for evaluation/serving. The OOF
    # frame already carries home_win from walk_forward_oof — exclude it from
    # the key frame so the merge never suffixes the target column.
    key = game_df[["game_id", "gameday", "season", "week", "home_team",
                   "away_team", "home_score", "away_score", "margin",
                   "total", "stadium", "gametime",
                   "home_record", "away_record"]].copy()
    key["gameday"] = pd.to_datetime(key["gameday"])
    oof_ml["gameday"] = pd.to_datetime(oof_ml["gameday"])
    oof_ml = oof_ml.merge(key, on=["game_id", "gameday", "season"], how="left")

    # ── 6/7. Run-line + totals OOF (joint distribution model) ─────────────
    _banner("PHASE 6-7", "margin/total distribution OOF")
    dist = dist_mod.walk_forward_oof(game_df, fold_list=fold_list)
    oof_dist = dist["oof"]

    # pooled sigma calibration from OOF residuals
    sig = dist_mod.calibrate_sigma(oof_dist["resid_margin"].to_numpy(),
                                   oof_dist["resid_total"].to_numpy())
    logger.info("calibrated sigma: margin %.3f, total %.3f",
                sig["sigma_margin"], sig["sigma_total"])

    # ── 8. Ensemble calibration (moneyline Platt on OOF) ──────────────────
    _banner("PHASE 8", "calibration")
    y_oof = oof_ml["home_win"].to_numpy(float)
    p_ens = oof_ml["p_ensemble"].to_numpy(float)
    okp = np.isfinite(p_ens)
    platt = ml_mod.fit_platt(p_ens[okp], y_oof[okp])
    logger.info("Platt (OOF-fit): a=%.4f b=%.4f", platt["a"], platt["b"])
    oof_ml["p_ensemble_calibrated"] = np.nan
    oof_ml.loc[okp, "p_ensemble_calibrated"] = ml_mod.apply_platt(
        p_ens[okp], platt)

    # ── 9. Evaluation ─────────────────────────────────────────────────────
    _banner("PHASE 9", "evaluation / diagnostics")
    raw_m = eval_mod.binary_metrics(oof_ml["p_ensemble"], y_oof)
    cal_m = eval_mod.binary_metrics(oof_ml["p_ensemble_calibrated"], y_oof)
    logger.info("moneyline OOF raw:    %s", json.dumps(raw_m))
    logger.info("moneyline OOF calib:  %s", json.dumps(cal_m))
    member_rows = monitoring.ensemble_table(oof_ml, weights)
    for r in member_rows:
        logger.info("  member %-13s w=%.3f auc=%.4f brier=%.4f",
                    r["name"], r["weight"], r["auc"] or np.nan, r["brier"] or np.nan)

    dist_metrics = eval_mod.distribution_metrics(
        oof_dist.merge(
            oof_ml[["game_id"]].assign(_k=1), on="game_id", how="inner"
        ).drop(columns=["_k"]) if len(oof_ml) else oof_dist,
        sig["sigma_margin"], sig["sigma_total"])
    logger.info("run-line OOF:  %s", json.dumps(dist_metrics["run_line"]))
    logger.info("totals OOF:    %s", json.dumps(dist_metrics["totals"]))
    margin_cal_tbl = eval_mod.margin_calibration_table(oof_dist, sig["sigma_margin"])
    total_cal_tbl = eval_mod.total_calibration_table(oof_dist, sig["sigma_total"])

    # daily calibration rows (frontend contract)
    daily = []
    oof_ml["gd_date"] = pd.to_datetime(oof_ml["gameday"]).dt.strftime("%Y%m%d")
    for day, grp in oof_ml.groupby("gd_date"):
        m = eval_mod.binary_metrics(grp["p_ensemble_calibrated"], grp["home_win"])
        daily.append({
            "date": day, "n_games": m["n"],
            "wins": int(((grp["p_ensemble_calibrated"] >= 0.5)
                         == (grp["home_win"] > 0.5)).sum()),
            "losses": int(m["n"] - ((grp["p_ensemble_calibrated"] >= 0.5)
                                    == (grp["home_win"] > 0.5)).sum()),
            "metrics": {k: m[k] for k in ("auc", "brier", "logloss", "ece")},
            "buckets": eval_mod.calibration_buckets(
                grp["p_ensemble_calibrated"], grp["home_win"]),
        })

    # ── 10. Final full-history refit ──────────────────────────────────────
    _banner("PHASE 10", "final full-history refit")
    final_models, _ = ml_mod.fit_final_models(game_df)
    final_reg = dist_mod.fit_final(game_df)

    # ── 11. Current-slate serving ─────────────────────────────────────────
    _banner("PHASE 11", "current-slate serving")
    slate = feat_mod.build_slate_features(schedule, pbp)
    if len(slate):
        slate = slate.sort_values("gameday").reset_index(drop=True)
        p_home = ml_mod.predict_slate(final_models, slate, weights)
        # serve through the SAME Platt map fitted on OOF
        p_home_cal = ml_mod.apply_platt(p_home, platt) if np.isfinite(p_home).any() else p_home
        slate["mu_h"], slate["mu_a"] = final_reg.predict(slate)
        slate = dist_mod.apply_distribution(slate, sig["sigma_margin"],
                                            sig["sigma_total"])
        slate["p_home_win"] = p_home_cal
        slate["p_away_win"] = 1.0 - p_home_cal
        slate["p_tie"] = slate["p_push_0"]
        slate["derived_ml"] = slate["p_home_win_derived"]
        slate["kind"] = "slate"
        slate["decided"] = False
        slate["frame_view"] = "slate"
        slate["pred_home"] = slate["mu_h"]
        slate["pred_away"] = slate["mu_a"]
        # QB enrichment (display only)
        qb_stats = {}
        try:
            from nflreadpy import load_player_stats
            for s in sorted(set(int(x) for x in slate["season"].unique())):
                ps = load_player_stats(s)
                df = ps.to_pandas() if hasattr(ps, "to_pandas") else ps
                qb_stats[s] = df
        except Exception as exc:  # noqa: BLE001
            logger.warning("QB stats pull failed: %s", exc)
        qb_df = qb_enrichment.enrich_slate(slate, qb_stats)
    else:
        slate = pd.DataFrame()
        qb_df = pd.DataFrame()
        p_home = np.array([])
        p_home_cal = np.array([])
    logger.info("slate games: %d", len(slate))

    # decided OOF rows for the markets artifact (same schema as slate rows)
    oof_market_rows = _build_oof_market_rows(oof_ml, oof_dist, sig)

    # ── 12. Artifact persistence ──────────────────────────────────────────
    _banner("PHASE 12", "artifact persistence")
    artifacts: list[str] = []
    if len(slate):
        p = out_dir / config.MONEYLINE_JSON.format(date=date_c)
        serve_mod.write_moneyline_json(p, slate, p_home, p_home_cal,
                                       _team_names(), config_meta)
        artifacts.append(p.name)

        p = out_dir / config.QB_MATCHUP_JSON.format(date=date_c)
        serve_mod.write_qb_matchup_json(p, qb_df, slate)
        artifacts.append(p.name)

    p = out_dir / config.CALIBRATION_JSON.format(date=date_c)
    # MLB convention: the pooled reliability buckets are built from the RAW
    # blend (the table renders MEAN PREDICTED (RAW)) with the calibrated
    # twin carried separately in calibration.calibration_buckets_calibrated.
    serve_mod.write_calibration_json(
        p, raw_m, cal_m,
        eval_mod.calibration_buckets(oof_ml["p_ensemble"], y_oof),
        daily, config_meta, platt=platt, run_date=date_c, n_games=int(okp.sum()),
        calibrated_buckets=eval_mod.calibration_buckets(
            oof_ml["p_ensemble_calibrated"], y_oof))
    artifacts.append(p.name)

    p = out_dir / config.PREDICTIONS_HISTORY_CSV.format(date=date_c)
    serve_mod.write_predictions_history_csv(p, oof_ml,
                                            oof_ml["p_ensemble_calibrated"].to_numpy())
    artifacts.append(p.name)

    p = out_dir / config.POWER_RANKINGS_CSV.format(date=date_c)
    _write_power_rankings(p, game_df)
    artifacts.append(p.name)

    p = out_dir / config.MARKETS_CSV.format(date=date_c)
    mp = out_dir / config.MARKETS_META_JSON.format(date=date_c)
    serve_mod.write_markets_csv(p, mp, oof_market_rows, slate, config_meta)
    artifacts.append(p.name)

    # OOF stores (model artifacts under data_delivery/models/)
    p = out_dir / "nfl_oof_moneyline.csv"
    oof_ml.to_csv(p, index=False)
    artifacts.append(p.name)
    p = out_dir / "nfl_oof_distribution.csv"
    oof_dist.to_csv(p, index=False)
    artifacts.append(p.name)

    # feature manifest record
    p = out_dir / config.FEATURE_JSON.format(date=date_c)
    _write_feature_json(p, cov, config_meta, fold_info)
    artifacts.append(p.name)

    # model bundle (joblib)
    import joblib
    bundle = {
        "moneyline_models": {k: v["model"] for k, v in final_models.items()},
        "moneyline_preprocessors": {k: v["pre"] for k, v in final_models.items()},
        "ensemble_weights": weights,
        "platt": platt,
        "score_regressor": final_reg,
        "sigma": sig,
        "feature_set_version": config.FEATURE_SET_VERSION,
        "feature_columns": config.FEATURE_COLUMNS,
        "trained_utc": _now_utc(),
        "config": config_meta,
    }
    joblib.dump(bundle, config.MODEL_BUNDLE)
    artifacts.append(str(config.MODEL_BUNDLE.name))

    # ── 14. Monitoring ───────────────────────────────────────────────────
    _banner("PHASE 14", "monitoring")
    recent = game_df.tail(60)
    drift = monitoring.feature_drift(game_df, recent, weights=weights)
    cov_rows = monitoring.coverage(game_df)
    rb = monitoring.rolling_brier(oof_ml)
    baseline = float(1.0 - y_oof.mean())  # constant always-predict-home baseline Brier
    p = out_dir / config.MODEL_MONITOR_JSON.format(date=date_c)
    monitoring.write_monitor_json(p, date_c, drift, cov_rows, member_rows,
                                  rb, baseline, config_meta, fold_info,
                                  metrics=cal_m, platt=platt)
    artifacts.append(p.name)

    # ── 13. Schema validation (gates) ─────────────────────────────────────
    _banner("PHASE 13", "schema validation")
    gates = _validate_outputs(out_dir, date_c, oof_ml, slate, fold_info)
    for name, ok in gates.items():
        logger.info("gate %-28s %s", name, "PASS" if ok else "FAIL")
    if not all(gates.values()):
        failed = [k for k, v in gates.items() if not v]
        raise RuntimeError(f"validation gates failed: {failed}")

    # retention: keep only the newest 3 dated copies of each family
    _prune_old_artifacts(out_dir, date_c)

    _banner("DONE", f"{len(artifacts)} artifacts in {time.time() - t0:.0f}s")
    summary = {
        "status": "ok",
        "run_date": run_date,
        "artifacts": artifacts,
        "moneyline_oof": raw_m,
        "moneyline_oof_calibrated": cal_m,
        "run_line_oof": dist_metrics["run_line"],
        "totals_oof": dist_metrics["totals"],
        "weights": weights,
        "folds": fold_info,
        "n_slate": int(len(slate)),
    }
    (out_dir / "nfl_pipeline_summary.json").write_text(
        json.dumps(summary, indent=1, default=str))
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _team_names() -> dict[str, str]:
    try:
        return ingestion.load_team_names()
    except Exception:
        return {}


def _build_oof_market_rows(oof_ml: pd.DataFrame, oof_dist: pd.DataFrame,
                           sig: dict) -> pd.DataFrame:
    """Decided OOF rows in the markets schema: distribution grids from the
    OOF mu pair + honest outcomes (y_*), plus the calibrated moneyline."""
    m = oof_ml[["game_id", "gameday", "season", "week", "home_team",
                "away_team", "stadium", "gametime", "home_record",
                "away_record", "home_score", "away_score", "margin",
                "total", "home_win", "p_ensemble",
                "p_ensemble_calibrated"]].copy()
    d = oof_dist[["game_id", "mu_h", "mu_a"]].copy()
    df = m.merge(d, on="game_id", how="inner")
    if not len(df):
        return pd.DataFrame()
    df = dist_mod.apply_distribution(df, sig["sigma_margin"], sig["sigma_total"])
    # the markets schema's p_home_win is the CALIBRATED moneyline (falling
    # back to the raw blend when calibration is unavailable for a row)
    df["p_home_win"] = df["p_ensemble_calibrated"].where(
        df["p_ensemble_calibrated"].notna(), df["p_ensemble"])
    df["p_away_win"] = 1.0 - df["p_home_win"]
    df["p_tie"] = df["p_push_0"]
    df["derived_ml"] = df["p_home_win_derived"]
    df["kind"] = "oof"
    df["decided"] = True
    df["frame_view"] = "oof"
    df["pred_home"] = df["mu_h"]
    df["pred_away"] = df["mu_a"]
    df["spread_line"] = np.nan
    df["total_line"] = np.nan
    df["has_offer"] = False
    df["p_cover_offered"] = np.nan
    df["p_push_offered"] = np.nan
    df["p_over_offered"] = np.nan
    df["p_under_offered"] = np.nan
    df["p_push_total_offered"] = np.nan
    # honest outcomes at the fair lines
    def _outcome_row(r) -> dict:
        fair_s = r.fair_spread
        fair_t = r.fair_total
        eps = 1e-12
        y_cover = 1.0 if r.margin > fair_s else 0.0
        y_push_s = 1.0 if r.margin == fair_s else 0.0
        y_over = 1.0 if r.total > fair_t else 0.0
        y_push_t = 1.0 if r.total == fair_t else 0.0
        return {
            "y_over_fair": y_over, "y_under_fair": 1.0 - y_over - y_push_t,
            "y_push_fair": y_push_t,
            "y_cover_fair": y_cover, "y_push_spread_fair": y_push_s,
            "y_home_win": float(r.home_win),
        }
    outs = [_outcome_row(r) for r in df.itertuples(index=False)]
    for c in outs[0]:
        df[c] = [o[c] for o in outs]
    return df


def _write_power_rankings(path: Path, game_df: pd.DataFrame) -> None:
    """Elo-based power rankings from the feature engine's state."""
    ev = feat_mod.team_events(game_df)
    _, ratings = feat_mod._elo_apply(ev)
    rec = ev.groupby("team").agg(
        wins=("team_win", lambda s: float((s == 1).sum())),
        losses=("team_win", lambda s: float((s == 0).sum())),
    )
    names = _team_names()
    rows = []
    for team, elo in sorted(ratings.items(), key=lambda kv: -kv[1]):
        w = int(rec.loc[team, "wins"]) if team in rec.index else 0
        l = int(rec.loc[team, "losses"]) if team in rec.index else 0
        rows.append({"rank": 0, "team": team, "team_name": names.get(team, team),
                     "elo": round(float(elo), 1), "wins": w, "losses": l,
                     "record": f"{w}-{l}",
                     "pct": round(w / (w + l), 3) if (w + l) else np.nan,
                     "run_diff": 0, "l10": "", "home_pct": np.nan,
                     "away_pct": np.nan})
    df = pd.DataFrame(rows)
    df["rank"] = range(1, len(df) + 1)
    df.to_csv(path, index=False)


def _write_feature_json(path: Path, cov: pd.DataFrame, config_meta: dict,
                        fold_info: dict) -> None:
    from manifest import FEATURE_MANIFEST
    record = {
        "created_utc": _now_utc(),
        "feature_set_version": config.FEATURE_SET_VERSION,
        "manifest": FEATURE_MANIFEST,
        "served_columns": config.FEATURE_COLUMNS,
        "coverage": cov.to_dict(orient="records"),
        "fold_geometry": fold_info,
        "config": config_meta,
    }
    serve_mod._dump_json(path, record)


def _validate_outputs(out_dir: Path, date_c: str, oof_ml: pd.DataFrame,
                      slate: pd.DataFrame, fold_info: dict) -> dict:
    """Schema/coherence gates over the written artifacts."""
    gates: dict[str, bool] = {}
    # moneyline probability coherence
    p = oof_ml["p_ensemble_calibrated"].to_numpy(float)
    p = p[np.isfinite(p)]
    gates["ml_probability_bounds"] = bool(((p >= 0) & (p <= 1)).all())
    gates["oof_population"] = bool(
        (pd.to_numeric(oof_ml["season"]) >= config.OOF_FIRST_SEASON).all())
    # distribution coherence on the OOF market rows
    p = out_dir / config.MARKETS_CSV.format(date=date_c)
    if p.exists():
        mk = pd.read_csv(p)
        ok = True
        for L in config.SPREAD_GRID:
            label = f"m{-L}" if L < 0 else str(L)
            h, pu = mk.get(f"p_home_cover_{label}"), mk.get(f"p_push_{label}")
            if h is None or pu is None:
                ok = False
                break
            s = (h.fillna(0) + pu.fillna(0)).clip(0, 1)
            if not ((s <= 1.0 + 1e-9)).all():
                ok = False
                break
        gates["spread_grid_coherent"] = ok
        ok = True
        for U in config.TOTAL_GRID:
            o, u, pu = (mk.get(f"p_over_{U}"), mk.get(f"p_under_{U}"),
                        mk.get(f"p_push_{U}"))
            if o is None or u is None or pu is None:
                ok = False
                break
            s = (o.fillna(0) + u.fillna(0) + pu.fillna(0))
            if not ((np.abs(s - 1.0) < 1e-6)).all():
                ok = False
                break
        gates["totals_grid_coherent"] = ok
        gates["markets_has_slate"] = bool((mk["kind"] == "slate").any())
    else:
        gates["spread_grid_coherent"] = False
        gates["totals_grid_coherent"] = False
        gates["markets_has_slate"] = False
    # serving contract fields on the slate rows
    need = {"game_id", "gameday", "home_team", "away_team", "mu_h", "mu_a",
            "fair_spread", "fair_total", "p_home_win_derived",
            "p_away_win_derived"}
    gates["slate_contract_fields"] = need.issubset(slate.columns) if len(slate) else True
    gates["fold_geometry"] = fold_info.get("n_folds", 0) > 0
    return gates


def _prune_old_artifacts(out_dir: Path, date_c: str) -> None:
    """Keep the newest KEEP dated copies per family; never touch non-dated
    files or other sports' directories."""
    KEEP = 3
    families = [
        config.MONEYLINE_JSON, config.CALIBRATION_JSON,
        config.PREDICTIONS_HISTORY_CSV, config.POWER_RANKINGS_CSV,
        config.MARKETS_CSV, config.MARKETS_META_JSON,
        config.QB_MATCHUP_JSON, config.FEATURE_JSON,
        config.MODEL_MONITOR_JSON,
    ]
    for template in families:
        prefix = template.split("{")[0]
        ext = template.split("}")[1]
        dated = sorted(out_dir.glob(f"{prefix}*{ext}"))
        for old in dated[:-KEEP]:
            try:
                old.unlink()
                logger.info("pruned stale artifact %s", old.name)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
