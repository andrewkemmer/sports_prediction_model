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
import importlib.metadata
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.parse import quote

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


def _env_date(primary: str, legacy: str, fallback: str) -> str:
    """Read canonical date controls, retaining the old season aliases."""
    raw = (os.environ.get(primary) or os.environ.get(legacy) or fallback).strip()
    try:
        return date.fromisoformat(raw[:10]).isoformat()
    except ValueError:
        # A legacy integer season remains valid and maps to the season start.
        try:
            return date(int(raw), 1, 1).isoformat()
        except (TypeError, ValueError):
            raise SystemExit(f"{primary}/{legacy} must be YYYY-MM-DD or a season") from None


def _env_end_bounds() -> tuple[str, str]:
    """(seasons-range end, gameday window end) from the end controls.

    A legacy integer season alias (``NFL_END_SEASON=2026``) bounds the
    SEASONS at that year but the gameday window at that season's calendar
    tail (Feb 28 of the following year) — an NFL postseason runs into
    January/February, and capping at Dec 31 silently drops the season's
    own playoff games (the 20260919 run lost the 31 games of Jan 2027).
    An explicit ``NFL_END_DATE`` bounds both literally; the default bounds
    both at today (ET).
    """
    raw = (os.environ.get("NFL_END_DATE") or os.environ.get("NFL_END_SEASON") or "").strip()
    if raw.isdigit() and len(raw) == 4:
        year = int(raw)
        return (date(year, 12, 31).isoformat(), date(year + 1, 2, 28).isoformat())
    # Single timezone for BOTH window bounds and the artifact run-date stamp:
    # ET (the league's operational clock). Mixing ET here with UTC in
    # run_date produced a one-day stamp skew — a 03:50-UTC run (23:50 ET on
    # the 21st) windowed data through 2026-09-21 but stamped artifacts
    # _20260922, and the serving pass found no unplayed in-window game while
    # the artifact date claimed a day the window never covered.
    day = _env_date("NFL_END_DATE", "NFL_END_SEASON",
                    datetime.now(ZoneInfo("America/New_York")).date().isoformat())
    return (day, day)


def _env_flag(name: str) -> bool:
    """Truthy env flag (1/true/yes) — mirrors MLB_FULL_REPULL parsing."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _library_stack() -> dict[str, str]:
    """Versions of the libraries whose numerics shape the ensemble verdict.

    The blend weights are earned by SLSQP on member OOF log-losses, and the
    members themselves are seeded-but-environment-sensitive (xgboost booster
    numerics, scipy SLSQP iteration path). Two environments on the same data
    can therefore ship different weight vectors from the same code. Recording
    the stack per run makes every weight verdict interpretable against the
    exact libraries that earned it — drift shows up in the artifact instead
    of being discoverable only by cross-run forensics.
    """
    stack = {}
    for dist, key in (("scipy", "scipy"), ("scikit-learn", "sklearn"),
                      ("xgboost", "xgboost"), ("lightgbm", "lightgbm"),
                      ("pandas", "pandas"), ("numpy", "numpy")):
        try:
            stack[key] = importlib.metadata.version(dist)
        except Exception:  # noqa: BLE001 — absent optional dist must never block a run
            stack[key] = "unknown"
    return stack


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
    # Artifact stamp on the SAME ET clock as the data window (see
    # _env_end_bounds): keeps run_date, the window, and the slate day aligned.
    run_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    date_c = run_date.replace("-", "")

    # MLB-style date controls. Warm-up remains 2018 by default; OOF still
    # begins in config.OOF_FIRST_SEASON. Legacy *_SEASON aliases are accepted.
    full_repull = _env_flag("NFL_FULL_REPULL")
    start_date = _env_date("NFL_START_DATE", "NFL_START_SEASON", "2018-01-01")
    end_date, window_end = _env_end_bounds()
    if start_date > window_end:
        raise SystemExit(f"invalid date window: {start_date} > {window_end}")
    seasons = list(range(int(start_date[:4]), int(end_date[:4]) + 1))
    logger.info("nflverse date window: %s..%s (seasons %d..%d)",
                start_date, window_end, seasons[0], seasons[-1])
    if full_repull:
        ingestion.clear_cache()
        logger.info("NFL_FULL_REPULL=1 — nflverse cache cleared for full rebuild")
    logger.info("library stack: %s", json.dumps(_library_stack()))
    config_meta = {
        "feature_set_version": config.FEATURE_SET_VERSION,
        "warmup_seasons": config.WARMUP_SEASONS,
        "oof_first_season": config.OOF_FIRST_SEASON,
        "retrain_cadence_days": config.RETRAIN_CADENCE_DAYS,
        "min_val_fold_games": config.MIN_VAL_FOLD_GAMES,
        "game_types": sorted(config.GAME_TYPES),
        "ensemble_members": config.ENSEMBLE_MEMBERS,
        "ensemble_blend": {
            "method": "rolling_logloss_slsqp",
            "space": "logit",
            "prior": "fold0_static_thirds",
            "reearn": "per_fold_pooled_oof_strictly_prior",
            "gates": "none",
        },
        # The exact libraries that earned this run's blend weights (see
        # _library_stack): member numerics and the SLSQP path are
        # environment-sensitive, so weight verdicts are only interpretable
        # against the stack that produced them.
        "library_stack": _library_stack(),
        "run_line_model": {
            "home_model": "lightgbm_poisson",
            "away_model": "lightgbm_poisson",
            "distribution": "negative_binomial",
            "simulation": "monte_carlo",
            "mc_draws": dist_mod.MC_DRAWS,
            "feature_contract": "binary_moneyline",
            "feature_columns": list(config.active_moneyline_feature_cols()),
        },
        "random_seed": config.RANDOM_SEED,
        "market_independence": True,
    }

    # ── 2. Eligible game ingestion ────────────────────────────────────────
    _banner("PHASE 2", "nflverse ingestion")
    # Fresh schedule each run unless --skip-pull; pbp uses per-season cache
    # with incremental top-up (full repull above cleared it).
    schedule = ingestion.load_schedule(
        seasons=seasons,
        use_cache=args.skip_pull and not full_repull,
    )
    schedule = ingestion.eligible_games(schedule)
    schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce")
    # The configured date window bounds the complete delivered game
    # population, including scheduled dashboard games. This keeps NFL_END_DATE
    # structurally aligned with MLB: games on the inclusive end date remain,
    # while future games are not emitted into the dashboard artifact.
    schedule = schedule[
        (schedule["gameday"] >= pd.Timestamp(start_date))
        & (schedule["gameday"] <= pd.Timestamp(window_end))].copy()
    logger.info("schedule rows (date window): %d", len(schedule))
    pbp = ingestion.load_pbp(seasons=seasons, use_cache=not full_repull)
    logger.info("pbp rows: %s", 0 if pbp is None else len(pbp))
    # Skill-position usage, tracking efficiency, and availability (candidate
    # sources + pre-game facts). Trailing windows need a warmup season, so the
    # pull extends one season back (same pattern as the slate QB enrichment).
    ps = ingestion.load_player_stats(
        seasons=[seasons[0] - 1] + seasons, use_cache=not full_repull)
    ngs = ingestion.load_nextgen(
        seasons=[seasons[0] - 1] + seasons, use_cache=not full_repull)
    injuries = ingestion.load_injuries(seasons=seasons,
                                       use_cache=not full_repull)
    logger.info("player stats rows: %s | ngs rows: %s | injury report rows: %s",
                0 if ps is None else len(ps),
                0 if ngs is None else len(ngs),
                0 if injuries is None else len(injuries))

    decided_all = schedule[schedule["home_score"].notna()
                           & schedule["away_score"].notna()].copy()
    warmup = decided_all[pd.to_numeric(decided_all["season"]) < config.OOF_FIRST_SEASON]
    core = decided_all[pd.to_numeric(decided_all["season"]) >= config.OOF_FIRST_SEASON]
    logger.info("decided games: warmup %d, core (OOF population) %d",
                len(warmup), len(core))

    # ── 3. Point-in-time features ─────────────────────────────────────────
    _banner("PHASE 3", "point-in-time feature engine")
    game_df = feat_mod.build_game_features(decided_all, pbp, ps=ps, ngs=ngs,
                                           inj=injuries)
    game_df = game_df.sort_values("gameday").reset_index(drop=True)
    logger.info("feature frame: %d decided games, %d columns",
                len(game_df), game_df.shape[1])
    cov = feat_mod.feature_coverage_report(game_df)
    logger.info("feature coverage:\n%s", cov.to_string(index=False))

    # Apply only an explicitly adopted RFE state. Ordinary runs retain the
    # full production list; a trial never changes serving width.
    try:
        from feature_selection import apply_adopted_subset
        logger.info("feature subset: %s", apply_adopted_subset(list(game_df.columns)))
    except Exception as exc:
        logger.warning("feature-selection state ignored: %s", exc)

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
    # MLB parity: the minimum validation gate intentionally leaves some core
    # games outside scored OOF folds. Those games remain in the expanding
    # history and may train later folds; no full-population equality guard is
    # applied here.
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

    # NFL-specific negative-binomial dispersion is estimated from the
    # walk-forward score predictions. The Poisson LightGBM means remain the
    # regression output; NB alpha controls the count-distribution variance.
    sig = dist_mod.calibrate_dispersion(oof_dist)
    logger.info("calibrated NB dispersion: alpha_home %.6f, alpha_away %.6f",
                sig["alpha_home"], sig["alpha_away"])

    # ── 8. Ensemble calibration (prequential OOF Platt, FAVORED space) ──────────────────
    _banner("PHASE 8", "calibration")
    y_oof = oof_ml["home_win"].to_numpy(float)
    p_ens = oof_ml["p_ensemble"].to_numpy(float)
    okp = np.isfinite(p_ens)

    # MLB structural parity — two distinct calibration layers:
    #
    #   1. PREQUENTIAL per-fold OOF calibration (the honest evaluation layer):
    #      fold k's calibrated predictions come from a favored-space Platt map
    #      fitted strictly on folds 0..k-1's OOF pairs. Fold 0 has no prior
    #      OOF, so its calibrated twin IS the raw blend (identity). The green
    #      calibration-curve lever, the CALIBRATED reliability column, and
    #      the calibrated KPIs all render from this layer — they now match
    #      the OOF-fold calibration leverage exactly.
    #
    #   2. POOLED final calibrator (the serving layer): one favored-space map
    #      fit on ALL OOF pairs, applied ONLY to tonight's slate — never used
    #      to score its own fitting population. Today's Game cards therefore
    #      still exactly match the production run (final calibrated
    #      probabilities), while the OOF layer stays honest.
    #
    # All fits/applications are in FAVORED-team space (p_fav = max(p, 1-p),
    # the side with probability > 50%), never home-team space — matching
    # MLB's favored_platt_floor contract end to end.

    # 8a. Per-fold PREQUENTIAL calibrated twins (evaluation layer).
    fold_calibrators: dict[int, dict | None] = {}
    cal_fitted = 0
    cal_identity = 0
    p_cal_prequential = np.full(len(oof_ml), np.nan)
    fold_ids = oof_ml["fold_id"].to_numpy()
    for fold in fold_list:
        val_mask = fold_ids == fold.fold_id
        prior_mask = (fold_ids < fold.fold_id) & okp
        if prior_mask.sum() >= 2:
            fold_cal = ml_mod.moneyline_fit(p_ens[prior_mask], y_oof[prior_mask])
        else:
            fold_cal = None  # no prior evidence yet — identity for fold 0
        fold_calibrators[int(fold.fold_id)] = fold_cal
        if fold_cal is None:
            cal_identity += 1
        else:
            cal_fitted += 1
        if val_mask.any() and okp[val_mask].any():
            p_cal_prequential[val_mask] = ml_mod.moneyline_apply(
                p_ens[val_mask], fold_cal)
    # Guard every row (NaN-safe identity fallback for un-scored rows).
    need_cal = okp & np.isnan(p_cal_prequential)
    p_cal_prequential[need_cal] = p_ens[need_cal]  # identity fallback
    oof_ml["p_ensemble_calibrated"] = p_cal_prequential
    # per-member prequential calibrated twins (MLB parity; diagnostics only)
    for name in config.ENSEMBLE_MEMBERS:
        col = f"p_{name}"
        if col in oof_ml.columns:
            pm = pd.to_numeric(oof_ml[col], errors="coerce").to_numpy(float)
            twin = np.full(len(oof_ml), np.nan)
            for fold in fold_list:
                val_mask = fold_ids == fold.fold_id
                if val_mask.any():
                    twin[val_mask] = ml_mod.moneyline_apply(
                        pm[val_mask], fold_calibrators[int(fold.fold_id)])
            oof_ml[f"p_{name}_calibrated"] = twin
    logger.info("prequential per-fold calibration: %d fitted, %d identity",
                cal_fitted, cal_identity)

    # 8b. POOLED final calibrator — the serving layer (never used to score
    # its own fitting population). Identical favored-space guardrails apply.
    platt = ml_mod.moneyline_fit(p_ens[okp], y_oof[okp])
    if platt is not None:
        logger.info("final pooled calibrator: a=%.4f b=%.4f n=%d method=%s",
                    platt["a"], platt["b"], platt["n"], platt["method"])
    else:
        logger.info("final pooled calibrator: identity (raw blend is served)")

    # ── 9. Evaluation ─────────────────────────────────────────────────────
    _banner("PHASE 9", "evaluation / diagnostics")
    # Headline calibrated metrics are the PREQUENTIAL twins (honest per-fold
    # leverage), not pooled self-calibration (MLB parity).
    raw_m = eval_mod.binary_metrics(oof_ml["p_ensemble"], y_oof)
    cal_m = eval_mod.binary_metrics(oof_ml["p_ensemble_calibrated"], y_oof)
    logger.info("moneyline OOF raw:    %s", json.dumps(raw_m))
    logger.info("moneyline OOF calib:  %s", json.dumps(cal_m))
    member_rows = monitoring.ensemble_table(oof_ml, weights)
    for r in member_rows:
        logger.info("  member %-13s w=%.3f auc=%.4f brier=%.4f",
                    r["name"], r["weight"], r["auc"] or np.nan, r["brier"] or np.nan)

    dist_metrics = eval_mod.nb_distribution_metrics(
        oof_dist.merge(
            oof_ml[["game_id"]].assign(_k=1), on="game_id", how="inner"
        ).drop(columns=["_k"]) if len(oof_ml) else oof_dist,
        sig)
    logger.info("run-line OOF:  %s", json.dumps(dist_metrics["run_line"]))
    logger.info("totals OOF:    %s", json.dumps(dist_metrics["totals"]))
    margin_cal_tbl = []
    total_cal_tbl = []

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
    slate = feat_mod.build_slate_features(schedule, pbp, ps=ps, ngs=ngs,
                                          inj=injuries)
    if len(slate):
        slate = slate.sort_values("gameday").reset_index(drop=True)
        p_home = ml_mod.predict_slate(final_models, slate, weights)
        # serve through the SAME POOLED favored-space calibrator the OOF fit
        # produced (the serving layer) — gated by CALIBRATION_MODE; identity
        # mode or a None/degenerate calibrator publishes the raw blend.
        p_home_cal = (ml_mod.moneyline_apply(p_home, platt)
                      if np.isfinite(p_home).any() else p_home)
        slate["mu_h"], slate["mu_a"] = final_reg.predict(slate)
        slate = dist_mod.apply_distribution(slate, sig)
        slate["p_home_win"] = p_home_cal
        slate["p_away_win"] = 1.0 - p_home_cal
        slate["p_tie"] = slate["p_push_0"]
        slate["derived_ml"] = slate["p_home_win_derived"]
        slate["kind"] = "slate"
        slate["decided"] = False
        slate["frame_view"] = "slate"
        slate["pred_home"] = slate["mu_h"]
        slate["pred_away"] = slate["mu_a"]
        # QB enrichment (display only). The trailing window walks back
        # through PRIOR seasons (a week-1 slate's last-10 window is the end
        # of the previous season), so load the season before the slate too;
        # a failed/absent season degrades to missing fields, never an error.
        qb_stats = {}
        try:
            from nflreadpy import load_player_stats
            seasons = sorted(set(int(x) for x in slate["season"].unique()))
            for s in [seasons[0] - 1] + seasons:
                try:
                    ps = load_player_stats(s)
                    df = ps.to_pandas() if hasattr(ps, "to_pandas") else ps
                    qb_stats[s] = df
                except Exception as exc:  # noqa: BLE001
                    logger.warning("player stats pull failed for %s: %s", s, exc)
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
    # Calibrate every published total and run-line cut separately using only
    # prior folds for OOF rows; the final maps are reused for tonight's slate.
    oof_market_rows, market_calibration = dist_mod.calibrate_market_frame(oof_market_rows)
    if len(slate):
        slate = dist_mod.apply_market_calibration(slate, market_calibration)

    # ── 4.5. Record-only RFE + workbook ───────────────────────────────────
    _rfe: dict = {"ran": False, "reason": "NFL_RFE_FORCE not set"}
    try:
        from feature_selection import maybe_run_rfe
        _rfe = maybe_run_rfe(game_df, end_date)
        if _rfe.get("ran"):
            logger.info("RFE: mode=%s trials=%s selected=%s trace=%s",
                        _rfe.get("run_mode"), _rfe.get("n_trials"),
                        _rfe.get("n_selected"), _rfe.get("trace"))
            from feature_workbook import generate_workbook
            _rfe["workbook"] = generate_workbook(trace_path=_rfe.get("trace"))
            logger.info("RFE workbook: %s", _rfe["workbook"] or "not written")
    except Exception as exc:
        # Non-fatal by design (a selection hiccup must never block artifact
        # delivery) but never silent: the traceback goes to the log, and the
        # run summary records WHY — otherwise a crashed sweep is reported as
        # "NFL_RFE_FORCE not set" and the missing workbook looks intentional.
        _rfe["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("NFL RFE skipped (non-fatal): %s", exc, exc_info=True)

    # ── 12. Artifact persistence ──────────────────────────────────────────
    _banner("PHASE 12", "artifact persistence")
    artifacts: list[str] = []
    # RFE evidence travels with the run: the sweep trace plus the decision
    # workbook (both absent unless NFL_RFE_FORCE was set).
    for _rfe_path in (_rfe.get("trace"), _rfe.get("workbook")):
        if _rfe_path:
            artifacts.append(Path(_rfe_path).name)
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
        calibrated_buckets=eval_mod.calibration_buckets_pair(
            oof_ml["p_ensemble"], oof_ml["p_ensemble_calibrated"], y_oof),
        distribution_calibration=market_calibration)
    artifacts.append(p.name)

    p = out_dir / config.PREDICTIONS_HISTORY_CSV.format(date=date_c)
    serve_mod.write_predictions_history_csv(p, oof_ml,
                                            oof_ml["p_ensemble_calibrated"].to_numpy())
    artifacts.append(p.name)

    # Frozen first-publication card store (MLB parity with the adopted
    # Game-Totals run_engine_totals_history pattern): every game's
    # PRODUCTION prediction is priced ONCE and never mutated. Historical
    # cards serve from this store — they must never revert to OOF re-prices
    # (the walk-forward column is an evaluation view, not what was served).
    _card_store = _update_cards_history_store(out_dir, oof_ml, slate, date_c)
    if _card_store:
        artifacts.append(_card_store)

    p = out_dir / config.POWER_RANKINGS_CSV.format(date=date_c)
    _write_power_rankings(p, game_df)
    artifacts.append(p.name)

    p = out_dir / config.MARKETS_CSV.format(date=date_c)
    mp = out_dir / config.MARKETS_META_JSON.format(date=date_c)
    serve_mod.write_markets_csv(p, mp, oof_market_rows, slate, config_meta)
    artifacts.append(p.name)

    # Run-Line & Totals Monitor — winner cards from this run's decided OOF
    # store + today's slate_history point (the frontend folds the dated
    # monitors' histories into the rolling table). ml_reference is the
    # shared moneyline ensemble's pooled win rate (the derived-ML card's
    # comparison anchor), computed from THIS run's moneyline OOF store.
    ml_ref: dict = {}
    if "p_ensemble" in oof_ml.columns and "home_win" in oof_ml.columns:
        prob = pd.to_numeric(oof_ml["p_ensemble"], errors="coerce")
        yv = pd.to_numeric(oof_ml["home_win"], errors="coerce")
        ok = prob.notna() & yv.isin([0, 1])
        prob, yv = prob[ok], yv[ok].astype(int)
        if len(prob):
            pick_home = prob > 0.5
            win = float(((pick_home & (yv == 1))
                         | (~pick_home & (yv == 0))).mean())
            ml_ref = {"source": "ml_win_prob", "n": int(len(prob)),
                      "win_rate": round(win, 4),
                      "predicted_mean": round(float(prob.mean()), 4)}
    p = out_dir / config.MARKETS_MONITOR_JSON.format(date=date_c)
    monitoring.write_markets_monitor_json(p, date_c, oof_market_rows,
                                          config_meta, ml_reference=ml_ref)
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
        "distribution": sig,
        "market_calibration": market_calibration,
        "feature_set_version": config.FEATURE_SET_VERSION,
        "feature_columns": config.active_moneyline_feature_cols(),
        "trained_utc": _now_utc(),
        "config": config_meta,
    }
    joblib.dump(bundle, config.MODEL_BUNDLE)
    artifacts.append(str(config.MODEL_BUNDLE.name))

    # ── 14. Monitoring ───────────────────────────────────────────────────
    _banner("PHASE 14", "monitoring")
    recent = game_df.tail(60)
    feature_weights = monitoring.feature_importance_weights(
        final_models, weights, feature_frame=game_df)
    drift = monitoring.feature_drift(game_df, recent, weights=feature_weights)
    cov_rows = monitoring.coverage(game_df)
    run_drift_name, run_cov_name = monitoring.write_run_engine_feature_artifacts(
        out_dir, date_c, game_df, recent, weights=feature_weights)
    artifacts.extend([run_drift_name, run_cov_name])
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

    # retention: enforce the rolling-retention policy (retention_policy.py;
    # MLB parity 10-day blanket window). Files staged by THIS run are "seen"
    # and never touched; the anchor is the run's end date (NFL_END_DATE when
    # set, else today ET) — identical anchor semantics to MLB's Phase 6.
    _prune_old_artifacts(out_dir, date_c, seen=set(artifacts),
                         anchor_iso=end_date)

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
        "rfe": _rfe,
    }
    (out_dir / "nfl_pipeline_summary.json").write_text(
        json.dumps(summary, indent=1, default=str))

    # The Kaggle wrapper intentionally remains MLB-shaped: it runs this
    # pipeline and expects the pipeline itself to deliver data_delivery/.
    # Sync the complete NFL directory dynamically; Git's existing ignore rules
    # determine which generated files are production-deliverable.
    sync_result = _sync_data_delivery(repo_root=BACKEND_DIR.parent.parent)
    if sync_result["staged_files"]:
        print(
            f"NFL artifacts pushed and remotely verified: "
            f"{len(sync_result['staged_files'])} files"
        )
    else:
        print("NFL artifact sync: nothing new to push")
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sync_data_delivery(repo_root: Path, branch: str = "main") -> dict:
    """Commit and push the complete NFL delivery tree without static file lists.

    The Kaggle launcher is deliberately unchanged from the MLB pattern: the
    model process owns artifact delivery and raises on a failed push. Only the
    NFL data_delivery subtree is ever staged; model/training logic is untouched.
    """
    delivery_rel = Path("nfl-backend") / "data_delivery"
    delivery_dir = repo_root / delivery_rel
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required for NFL artifact delivery")
    if not delivery_dir.is_dir():
        raise RuntimeError(f"NFL delivery directory does not exist: {delivery_dir}")

    auth_url = (
        "https://x-access-token:"
        f"{quote(token, safe='')}"
        "@github.com/andrewkemmer/sports_prediction_model.git"
    )
    def git(*args: str, capture: bool = False):
        return subprocess.run(
            ["git", *args], cwd=repo_root, check=True,
            capture_output=capture, text=True,
        )

    git("remote", "set-url", "origin", auth_url)
    git("config", "user.name", os.environ.get("GIT_USER_NAME", "NFL Production Pipeline"))
    git("config", "user.email", os.environ.get("GIT_USER_EMAIL", "nfl-pipeline@users.noreply.github.com"))
    last_error = None
    for attempt in range(1, 4):
        try:
            # Stage dynamically; no artifact family names are maintained here.
            git("reset")
            git("add", "-A", "--", delivery_rel.as_posix())
            staged = git("diff", "--cached", "--name-only", capture=True).stdout.splitlines()
            if any(not p.startswith(f"{delivery_rel.as_posix()}/") for p in staged):
                raise RuntimeError(f"NFL delivery scope violation: {staged}")

            if staged:
                git("commit", "-m", "Update NFL production artifacts")

            # The worktree is clean before rebase, so a concurrent push can be
            # healed without stashing or touching any non-NFL path.
            git("fetch", "origin", branch)
            git("rebase", f"origin/{branch}")
            git("push", "origin", branch)
            git("fetch", "origin", branch)
            remote = git(
                "ls-tree", "-r", "--name-only", f"origin/{branch}",
                delivery_rel.as_posix(), capture=True,
            ).stdout.splitlines()
            if not any(p.startswith(f"{delivery_rel.as_posix()}/") for p in remote):
                raise RuntimeError("remote NFL delivery directory is empty")
            return {"staged_files": staged}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 3:
                break
            logger.warning("NFL artifact sync attempt %d failed; retrying: %s", attempt, exc)
    raise RuntimeError(f"NFL artifact delivery failed after 3 attempts: {last_error}")


def _team_names() -> dict[str, str]:
    try:
        return ingestion.load_team_names()
    except Exception:
        return {}


def _build_oof_market_rows(oof_ml: pd.DataFrame, oof_dist: pd.DataFrame,
                           sig: dict) -> pd.DataFrame:
    """Decided OOF rows in the markets schema: distribution grids from the
    OOF mu pair + honest outcomes (y_*), plus the calibrated moneyline."""
    m = oof_ml[["game_id", "gameday", "season", "week", "fold_id",
                "home_team", "away_team", "stadium", "gametime", "home_record",
                "away_record", "home_score", "away_score", "margin",
                "total", "home_win", "p_ensemble",
                "p_ensemble_calibrated"]].copy()
    d = oof_dist[["game_id", "mu_h", "mu_a"]].copy()
    df = m.merge(d, on="game_id", how="inner")
    if not len(df):
        return pd.DataFrame()
    df = dist_mod.apply_distribution(df, sig)
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
        "served_columns": config.active_moneyline_feature_cols(),
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
        # A zero-game slate is a LEGITIMATE daily state, not a delivery
        # failure: when the window ends on a day with no unplayed in-window
        # games (e.g. Tue/Wed in-season, or the late-UTC run at 03:50 ET where
        # ET-tomorrow's games fall outside the window), the markets CSV is
        # OOF-only by design and the serving writers already emit empty
        # games[] records. The gate must only fail when slate rows SHOULD
        # exist — i.e. the serving pass built them — but the written artifact
        # lost them. len(slate) is passed in precisely for that distinction.
        gates["markets_has_slate"] = (bool((mk["kind"] == "slate").any())
                                      if len(slate) else True)
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


def _update_cards_history_store(out_dir: Path, oof_ml: pd.DataFrame,
                                slate: pd.DataFrame, date_c: str) -> str | None:
    """Append newly-decided games to the frozen card store (once).

    MLB parity (run_engine.update_totals_history_store): rows are priced at
    FIRST PUBLICATION and never mutated afterward. Seeding rebuilds from the
    retained dated predictions-history family (oldest first), whose rows ARE
    the production predictions as published and graded; subsequent runs
    append only game_ids the store has never seen. Returns the filename
    appended to the artifact manifest, or None on any failure (the store
    must never fail the run).
    """
    store_path = out_dir / "nfl_production_cards_history.csv"
    meta_path = out_dir / "nfl_production_cards_history.meta.json"
    cols = ["game_id", "game_date", "start_time_utc", "home_team", "away_team",
            "home_team_name", "away_team_name", "home_record", "away_record",
            "venue", "p_home_win", "p_away_win", "model_pick", "correct",
            "home_score", "away_score", "actual_winner", "game_status",
            "source_artifact_date"]
    try:
        known: set[str] = set()
        if store_path.exists():
            store = pd.read_csv(store_path, dtype={"game_id": str})
            known = set(store["game_id"].astype(str))
        else:
            # Seed from every retained dated history artifact (oldest first
            # so first-publication-wins is deterministic).
            frames = []
            for art in sorted(out_dir.glob("nfl_predictions_history_*.csv")):
                try:
                    df = pd.read_csv(art, dtype={"game_id": str})
                except Exception:
                    continue
                if df.empty or "game_id" not in df.columns:
                    continue
                src = art.stem.rsplit("_", 1)[-1]
                df = df[~df["game_id"].astype(str).isin(known)]
                known.update(df["game_id"].astype(str))
                df["source_artifact_date"] = src
                frames.append(df)
            store = (pd.concat(frames, ignore_index=True) if frames
                     else pd.DataFrame())
        added = 0
        if oof_ml is not None and len(oof_ml) and "game_id" in oof_ml.columns:
            dec = oof_ml[oof_ml["game_id"].astype(str).isin(known) == False].copy()
            dec = dec[dec[["home_score", "away_score"]].notna().all(axis=1)] \
                if {"home_score", "away_score"}.issubset(dec.columns) else dec
            if len(dec):
                ph = pd.to_numeric(dec["p_ensemble_calibrated"], errors="coerce") \
                    if "p_ensemble_calibrated" in dec.columns \
                    else pd.to_numeric(dec["p_ensemble"], errors="coerce")
                pr = pd.to_numeric(dec["p_ensemble"], errors="coerce")
                pick = np.where(ph >= 0.5, dec["home_team"], dec["away_team"])
                winner = np.where(dec["home_win"] > 0.5, dec["home_team"],
                                  np.where(dec["home_win"] < 0.5, dec["away_team"], "TIE"))
                out = pd.DataFrame({
                    "game_id": dec["game_id"].astype(str),
                    "game_date": pd.to_datetime(dec["gameday"]).dt.strftime("%Y-%m-%d"),
                    "home_team": dec["home_team"], "away_team": dec["away_team"],
                    "p_home_win": ph.round(6), "p_away_win": (1.0 - ph).round(6),
                    "model_pick": pick,
                    "correct": np.where(ph >= 0.5, dec["home_team"], dec["away_team"])
                    == winner,
                    "home_score": dec["home_score"], "away_score": dec["away_score"],
                    "actual_winner": winner,
                    "game_status": "Final",
                    "source_artifact_date": date_c,
                })
                out = out[~out["game_id"].isin(set(store["game_id"].astype(str)))] \
                    if len(store) else out
                added += len(out)
                store = pd.concat([store, out], ignore_index=True)
        if slate is not None and len(slate):
            dec_s = slate[slate[["home_score", "away_score"]].notna().all(axis=1)] \
                if {"home_score", "away_score"}.issubset(slate.columns) \
                else pd.DataFrame()
            if len(dec_s):
                dec_s = dec_s[~dec_s["game_id"].astype(str)
                              .isin(set(store["game_id"].astype(str)))] \
                    if len(store) else dec_s
                if len(dec_s):
                    ph = pd.to_numeric(dec_s["p_home_win"], errors="coerce")
                    hw = (pd.to_numeric(dec_s["home_score"], errors="coerce")
                          > pd.to_numeric(dec_s["away_score"], errors="coerce"))
                    winner = np.where(hw, dec_s["home_team"],
                                      np.where(~hw & (pd.to_numeric(dec_s["away_score"], errors="coerce")
                                                      > pd.to_numeric(dec_s["home_score"], errors="coerce")),
                                               dec_s["away_team"], "TIE"))
                    out = pd.DataFrame({
                        "game_id": dec_s["game_id"].astype(str),
                        "game_date": pd.to_datetime(dec_s["gameday"]).dt.strftime("%Y-%m-%d"),
                        "home_team": dec_s["home_team"], "away_team": dec_s["away_team"],
                        "p_home_win": ph.round(6), "p_away_win": (1.0 - ph).round(6),
                        "model_pick": np.where(ph >= 0.5, dec_s["home_team"],
                                               dec_s["away_team"]),
                        "correct": np.where(ph >= 0.5, dec_s["home_team"],
                                            dec_s["away_team"]) == winner,
                        "home_score": dec_s["home_score"],
                        "away_score": dec_s["away_score"],
                        "actual_winner": winner,
                        "game_status": "Final",
                        "source_artifact_date": date_c,
                    })
                    added += len(out)
                    store = pd.concat([store, out], ignore_index=True)
        if not len(store):
            return None
        store = store.sort_values(["game_date", "game_id"]).reset_index(drop=True)
        tmp = store_path.with_suffix(".csv.tmp")
        store.to_csv(tmp, index=False)
        tmp.replace(store_path)
        meta_path.write_text(json.dumps(
            {"run_date": date_c, "rows_added": added, "n_rows": int(len(store))},
            indent=2), encoding="utf-8")
        logger.info("cards history store: %d rows (%d added this run) -> %s",
                    len(store), added, store_path.name)
        return store_path.name
    except Exception as exc:  # noqa: BLE001 — the store must never fail the run
        logger.error("cards history store update FAILED (run continues): %s",
                     exc, exc_info=True)
        return None


def _prune_old_artifacts(out_dir: Path, date_c: str, seen: set | None = None,
                         anchor_iso: str | None = None) -> None:
    """Enforce the rolling-retention policy (retention_policy.py).

    MLB parity: blanket 10-day window anchor..anchor-10 (anchor = the run's
    end date — NFL_END_DATE when set, else today ET), never-delete masters
    and series readers untouched, backfill-safe anchor guard, board-backed
    safety net, and SHAP files aged through the game_id -> game_date map.
    Replaces the old newest-3-copies count rule. Pure-policy enforcement:
    every removal flows into the artifact sync's scoped git add as a forward
    commit, so git history retains every blob.
    """
    import retention_policy as rp

    seen = seen or set()
    anchor = (anchor_iso or date_c).replace("-", "")
    anchor_obj = datetime.strptime(anchor, "%Y%m%d").date()
    retention_dates = {(anchor_obj - timedelta(days=i)).strftime("%Y%m%d")
                       for i in range(11)}
    recent_dates = {(anchor_obj - timedelta(days=i)).strftime("%Y%m%d")
                    for i in range(3)}

    # Board dates: every date a navigable board exists (moneyline games[] +
    # the retained history family) — board-backed families keep those dates.
    # The moneyline games[] also seed the SHAP game-date map (board games are
    # authoritative game_id -> game_date rows).
    board_dates: set[str] = set()
    game_dates: dict[str, str] = {}
    for rec in out_dir.glob("nfl_moneyline_v1_*.json"):
        try:
            for g in json.loads(rec.read_text(encoding="utf-8")).get("games", []):
                d = str(g.get("game_date", ""))[:10].replace("-", "")
                if len(d) == 8 and d.isdigit():
                    board_dates.add(d)
                    gid = str(g.get("game_id", ""))
                    if gid:
                        game_dates.setdefault(gid, d)
        except Exception:
            continue
    for hist in out_dir.glob("nfl_predictions_history_*.csv"):
        try:
            for d in pd.read_csv(hist, usecols=["game_date"])["game_date"] \
                    .dropna().astype(str):
                d = d[:10].replace("-", "")
                if len(d) == 8 and d.isdigit():
                    board_dates.add(d)
        except Exception:
            continue

    # SHAP aging map continued: game_id -> YYYYMMDD (NFL ids embed season+week,
    # not dates — moneyline boards above, then the frozen card store, then the
    # newest history artifact; unresolvable ids stay protected, never guessed).
    cards = out_dir / "nfl_production_cards_history.csv"
    hist_sources = ([cards] if cards.exists() else []) \
        + sorted(out_dir.glob("nfl_predictions_history_*.csv"), reverse=True)
    for src in hist_sources:
        try:
            df = pd.read_csv(src, usecols=lambda c: c in ("game_id", "game_date"))
        except Exception:
            continue
        if df.empty or not {"game_id", "game_date"}.issubset(df.columns):
            continue
        for gid, gd in zip(df["game_id"].astype(str),
                           df["game_date"].astype(str)):
            d = gd[:10].replace("-", "")
            if len(d) == 8 and d.isdigit() and gid not in game_dates:
                game_dates[gid] = d
        if cards.exists() and src == cards:
            break

    stale: list[Path] = []
    kept_protected = 0
    kept_current = 0
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file() or p.name.startswith("~$"):
            continue
        rel = str(p.relative_to(out_dir.parent))
        verdict = rp.classify_artifact(
            rel, seen, retention_dates, recent_dates, board_dates,
            anchor_date=anchor, game_dates=game_dates)
        if verdict == "seen":
            continue
        if verdict == "protected":
            kept_protected += 1
            continue
        if verdict == "current":
            kept_current += 1
            continue
        stale.append(p)
    if kept_protected:
        logger.info("retention: kept %d protected file(s)", kept_protected)
    logger.info("retention: kept %d artifact(s) within the window (anchor %s -10d)",
                kept_current, anchor)
    if stale:
        for old in stale:
            try:
                old.unlink()
                logger.info("retention: pruned stale artifact %s", old.name)
            except OSError:
                pass
        logger.info("retention: removed %d stale file(s)", len(stale))
    else:
        logger.info("retention: no stale files")


if __name__ == "__main__":
    sys.exit(main())
