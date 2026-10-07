"""NHL production master pipeline — the ONE authoritative entry point.

Structural mirror of the NFL master pipeline (MLB lineage), phased:

  1. configuration         8. ensemble/calibration
  2. eligible ingestion    9. evaluation/diagnostics
  3. point-in-time features 10. final full-history fit
  4. fold generation       11. current-slate serving
  5. moneyline OOF         12. artifact persistence
  6. run-line OOF          13. schema validation
  7. totals OOF            14. monitoring (+ retention + delivery)

Zero dependency on any other sport's modules. Market-free.
Run:  python3 master_pipeline.py            (from nhl-backend/backend/)
      python3 master_pipeline.py --skip-pull (use cached NHL API pulls)
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

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
import monitoring  # noqa: E402

# Logs go to STDOUT, the same stream as the phase banners. They used to go to
# stderr while _banner printed to stdout, and the two are separate file
# descriptors, so a merged capture (2>&1) interleaved them arbitrarily: in a
# real run the "folds:", "prequential per-fold calibration" and "moneyline OOF
# raw" records all appeared under the PREVIOUS phase's banner, because they
# belong to phases 4, 8 and 9. Reading such a log sends you to the wrong phase
# with three phases of work misplaced. One stream means one true order, and it
# also means `> run.log` captures the log instead of silently dropping every
# record to the terminal. Nothing parses this stream.
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("nhl_master_pipeline")

# The four published OOF population blocks (fold_info keys) — filled by
# _oof_blocks after Phase 5's walk-forward, placeholder-setdefaulted at
# fold generation so no reader ever misses the key.
_OOF_BLOCK_KEYS = ("oof_regular", "oof_postseason", "oof_provisional",
                   "oof_all")


def _log_coverage_verdict(cov_rows: list[dict]) -> None:
    """One honest line per coverage window: measured, cold (by design), warm.

    Only WARM nulls are defects. A reader who sees a coverage percentage with
    no such split cannot tell "the season started" from "something is broken",
    and the goalie family once read 96-98% on a decided pool that was
    publishing nulls for every slate game.
    """
    for window in sorted({r.get("window", "?") for r in cov_rows}):
        rows = [r for r in cov_rows if r.get("window") == window]
        if not rows:
            continue
        warm = [r for r in rows if int(r.get("n_warm_null") or 0) > 0]
        cold = sum(int(r.get("n_cold_null") or 0) for r in rows)
        total = sum(int(r.get("n_measured") or 0) for r in rows)
        elig = sum(int(r.get("n_games") or 0) - int(r.get("n_cold_null") or 0)
                   for r in rows)
        pct = round(100.0 * total / elig, 3) if elig else 0.0
        logger.info("coverage [%s]: %d/%d features, %.3f%% measured on eligible "
                    "games, %d cold null(s) by design (team debut)",
                    window, len(rows), len(rows), pct, cold)
        if warm:
            detail = ", ".join(
                f"{r['feature']}({int(r['n_warm_null'])} {r.get('cause', 'null')})"
                for r in warm[:8])
            logger.warning("coverage [%s]: %d feature(s) with WARM nulls — a "
                           "defect, not a cold start: %s", window, len(warm), detail)
        else:
            logger.info("coverage [%s]: no warm nulls — every warm game measured",
                        window)


def _banner(phase: str, msg: str = "") -> None:
    print(f"\n{'-' * 70}\n  {phase} - {msg}\n{'-' * 70}\n", flush=True)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_date(primary: str, legacy: str, fallback: str) -> str:
    raw = (os.environ.get(primary) or os.environ.get(legacy) or fallback).strip()
    try:
        return date.fromisoformat(raw[:10]).isoformat()
    except ValueError:
        try:
            return date(int(raw), 1, 1).isoformat()
        except (TypeError, ValueError):
            raise SystemExit(f"{primary}/{legacy} must be YYYY-MM-DD or a season") from None


def _env_end_bounds() -> tuple[str, str]:
    """(seasons-range end, gameday window end) from the end controls.

    An NHL season runs Oct..Jun, so a legacy integer season alias bounds the
    gameday window at Jul 15 of the following year; an explicit
    ``NHL_END_DATE`` bounds both literally; the default bounds both at
    today (ET) — the league's operational clock, the same single-timezone
    discipline the NFL pipeline applies to artifact stamping.

    Stale-pin guard (2026-10-06 NHL log review, T7 — MLB parity): the
    Kaggle notebook leaves a literal ``NHL_END_DATE`` behind after a
    rebuild, and that pin bounds BOTH the seasons range and the gameday
    window — unguarded, every later run would freeze on the pinned day's
    slate. A literal pin before today extends to today via
    ``config.resolve_run_end_date`` (logged as a WARNING by the caller,
    through the run-log tee); the 4-digit season-alias branch returns
    above the guard and is never rewritten.
    """
    raw = (os.environ.get("NHL_END_DATE") or os.environ.get("NHL_END_SEASON") or "").strip()
    if raw.isdigit() and len(raw) == 4:
        year = int(raw)
        return (date(year, 12, 31).isoformat(), date(year + 1, 7, 15).isoformat())
    day = _env_date("NHL_END_DATE", "NHL_END_SEASON",
                    datetime.now(ZoneInfo("America/New_York")).date().isoformat())
    day = config.resolve_run_end_date(day)
    return (day, day)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _library_stack() -> dict[str, str]:
    stack = {}
    for dist, key in (("scipy", "scipy"), ("scikit-learn", "sklearn"),
                      ("xgboost", "xgboost"), ("lightgbm", "lightgbm"),
                      ("pandas", "pandas"), ("numpy", "numpy")):
        try:
            stack[key] = importlib.metadata.version(dist)
        except Exception:  # noqa: BLE001
            stack[key] = "unknown"
    return stack


def main(argv: list[str] | None = None) -> int:
    # Run-log tee (2026-10-02, MLB/NFL parity — run_log_tee.py):
    # capture the whole Kaggle run into ONE rolling master file in
    # nhl-backend/data_delivery/. The end-of-run _sync_data_delivery
    # stages the complete delivery tree (git add -A), so the log rides
    # along like any artifact, and retention_policy protects the
    # dateless master name from the prune — the latest run's full log
    # is reviewable from a plain git pull. A run that dies before the
    # sync pushes just the log (crash delivery). Degrades to
    # console-only on failure. Installed before argparse so EVERYTHING
    # the run prints lands in the file.
    from run_log_tee import (
        install_crash_log_pusher,
        install_run_log_tee,
    )
    # Never tee during a test run (NBA 5673874c / MLB / NFL parity):
    # install opens the rolling log with "w", so a pytest run that reaches
    # main() would TRUNCATE the committed nhl_pipeline_run_log.txt at the
    # tee's own header — the only record of the latest production run.
    # PYTEST_CURRENT_TEST covers in-test calls, pytest being loaded covers
    # collection-time probes, and a production launch (Kaggle) has neither.
    if os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
        _log_path = None
    else:
        _log_path = install_run_log_tee(config.DATA_DELIVERY_DIR)
    # Same hardcoded coordinates _sync_data_delivery pushes to.
    install_crash_log_pusher(
        _log_path, "andrewkemmer", "sports_prediction_model")

    ap = argparse.ArgumentParser(description="NHL production master pipeline")
    ap.add_argument("--skip-pull", action="store_true",
                    help="use cached NHL API pulls (no network)")
    ap.add_argument("--out-dir", default=None,
                    help="override artifact output directory")
    args = ap.parse_args(argv)

    t0 = time.time()
    out_dir = Path(args.out_dir) if args.out_dir else config.DATA_DELIVERY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Configuration ──────────────────────────────────────────────
    _banner("PHASE 1", "configuration (date window, repull, library stack)")
    run_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    date_c = run_date.replace("-", "")

    full_repull = _env_flag("NHL_FULL_REPULL")
    start_date = _env_date("NHL_START_DATE", "NHL_START_SEASON", "2024-01-01")
    end_date, window_end = _env_end_bounds()
    # The guard runs inside _env_end_bounds (before the logger call here,
    # but AFTER install_run_log_tee above), so re-announcing through the
    # logger is what lands the extension in the pushed run log — a reader
    # sees why the window no longer matches the notebook's pin.
    _pinned_end = (os.environ.get("NHL_END_DATE")
                   or os.environ.get("NHL_END_SEASON") or "").strip()
    if _pinned_end and not (_pinned_end.isdigit() and len(_pinned_end) == 4) \
            and _pinned_end != end_date:
        logger.warning(
            "NHL end-date pin %s is stale (before today) — extended to %s so "
            "the daily slate and retention anchor cannot freeze; re-date the "
            "notebook pin (or unset it) for a past-window backfill",
            _pinned_end, end_date)
    if start_date > window_end:
        raise SystemExit(f"invalid date window: {start_date} > {window_end}")
    seasons = list(range(int(start_date[:4]), int(end_date[:4]) + 1))
    logger.info("NHL API date window: %s..%s (seasons %d..%d)",
                start_date, window_end, seasons[0], seasons[-1])
    if full_repull:
        ingestion.clear_cache()
        logger.info("NHL_FULL_REPULL=1 — NHL API cache cleared for full rebuild")
    logger.info("library stack: %s", json.dumps(_library_stack()))
    config_meta = {
        "feature_set_version": config.FEATURE_SET_VERSION,
        "warmup_days": config.WARMUP_DAYS,
        "first_season": config.NHL_FIRST_SEASON,
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
    _banner("PHASE 2", "official NHL API ingestion")
    # Build the per-day date span per configured season (Oct 1 .. Jul 15),
    # then union the bounded operational serving tail.  ``season_dates``
    # intentionally starts on Oct 1, so a late-September regular-season
    # slate would otherwise never be requested even when the run/window is
    # explicitly set to those dates.
    dates: set[str] = set()
    for season in seasons:
        dates.update(ingestion.season_dates(season))
    serving_start = max(pd.Timestamp(start_date), pd.Timestamp(run_date))
    serving_end = pd.Timestamp(window_end)
    if serving_start <= serving_end:
        dates.update(pd.date_range(serving_start, serving_end, freq="D")
                     .strftime("%Y-%m-%d"))
    dates = sorted(dates)
    # Never request a date the window filter below will discard anyway. A run
    # builds season spans for every season in range, so a September run also
    # asks for the whole not-yet-started season (288 dates today). Those pages
    # carry the published but UNPLAYED schedule, so their games have null
    # scores, never reach decided_all, and are dropped by the gameday <=
    # window_end filter regardless — the clip is provably behaviour-preserving
    # and saves a round trip per day, growing with every season. The real cost
    # of that span is invisible in the log, which is what made the cold-cache
    # pull below read as a hang.
    horizon = pd.Timestamp(window_end)
    requested = len(dates)
    dates = [d for d in dates if pd.Timestamp(d) <= horizon]
    if len(dates) < requested:
        logger.info("score-date span clipped to the %s window: %d dates "
                    "(dropped %d dates the window would discard anyway)",
                    window_end, len(dates), requested - len(dates))
    schedule = ingestion.load_score_dates(dates, use_cache=args.skip_pull and not full_repull)
    schedule = ingestion.eligible_games(schedule)
    schedule["gameday"] = pd.to_datetime(schedule["game_date"], errors="coerce")
    schedule = schedule[
        (schedule["gameday"] >= pd.Timestamp(start_date))
        & (schedule["gameday"] <= pd.Timestamp(window_end))].copy()
    logger.info("schedule rows (date window): %d", len(schedule))

    decided_all = schedule[schedule["home_score"].notna()
                           & schedule["away_score"].notna()].copy()
    core = decided_all[pd.to_numeric(decided_all["season"]) >= config.OOF_FIRST_SEASON]
    logger.info("decided games: core (OOF population) %d", len(core))

    # ── 3. Point-in-time features ─────────────────────────────────────────
    _banner("PHASE 3", "point-in-time feature engine")
    decided_ids = decided_all["game_id"].astype(str).tolist()
    # The gameday mapping only groups the pull into 60-day windows for
    # progress reporting (MLB's _chunked_statcast shape); load_boxscores
    # restores the caller's id order, so the frame is unchanged by it.
    gameday_by_id = dict(zip(decided_all["game_id"].astype(str),
                             pd.to_datetime(decided_all["gameday"], errors="coerce")))
    boxscores = ingestion.load_boxscores(
        decided_ids, use_cache=args.skip_pull and not full_repull,
        gameday_by_id=gameday_by_id)
    logger.info("boxscore rows: %d", len(boxscores))
    game_df = feat_mod.build_game_features(decided_all, boxscores)
    # Canonical (date_col, game_id) order: the one order every fold index is
    # valid for. See folds.canonical_sort for why a single-column sort is not
    # enough — fold labels are positional and the tree members are
    # row-order sensitive under a fixed seed.
    game_df = folds_mod.canonical_sort(game_df, "gameday")
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
    fold_diag: dict = {}
    fold_list = folds_mod.make_folds(game_df, date_col="gameday",
                                     diagnostics=fold_diag)
    fold_info = folds_mod.fold_summary(fold_list)
    fold_info.update(fold_diag)
    for _block_key in _OOF_BLOCK_KEYS:
        fold_info.setdefault(_block_key, {"n": 0, "sufficient": False})
    fold_tbl = folds_mod.fold_table(game_df, fold_list, date_col="gameday")
    if not fold_list:
        raise RuntimeError(
            "PHASE 4 produced ZERO folds — walk-forward OOF cannot proceed. "
            "Check eligible-game ingestion and the 30-day warm-up geometry.")
    # Warm-up gate: the first validation window must start at least
    # WARMUP_DAYS after the first game date (the MLB-style warm-up contract).
    first_game_date = pd.to_datetime(game_df["gameday"]).min().normalize()
    if fold_list[0].val_start < first_game_date + pd.Timedelta(days=config.WARMUP_DAYS):
        raise RuntimeError(
            f"fold geometry violation: first validation start "
            f"{fold_list[0].val_start.date()} is within the "
            f"{config.WARMUP_DAYS}-day warm-up (first game "
            f"{first_game_date.date()})")
    first_val_season = pd.to_numeric(
        game_df.loc[fold_list[0].val_idx, "season"]).min() if fold_list else None
    if first_val_season != config.OOF_FIRST_SEASON:
        raise RuntimeError(
            f"fold geometry violation: first validation season {first_val_season} "
            f"!= OOF_FIRST_SEASON {config.OOF_FIRST_SEASON}")
    gd_dates = pd.to_datetime(game_df["gameday"])
    # Logged, not printed. This block is the run's central PIT claim - the
    # fold geometry every later phase rests on - and it was the only phase
    # output that carried no timestamp AND did not flush. Its position in a
    # captured run was a side effect of the logger call below flushing the
    # shared stdout buffer, not of the block itself: stdout is block-buffered
    # on a pipe (exactly the Kaggle subprocess), so these lines surface when
    # something ELSE happens to flush. `logging` writes to stdout too (see the
    # module header's one-stream rule) and its handler flushes every record,
    # so a single record here is both timestamped and self-flushing.
    #
    # The last window also names its ordinal next to its id. "fold 45" beside
    # Phase 5's "fold 46/46" reads as a contradiction unless you already know
    # one is a 0-based id and the other a count.
    logger.info("fold geometry:\n%s", "\n".join([
        f"  eligible settled games : {len(game_df)}",
        f"  game date range        : {gd_dates.min().date()} .. "
        f"{gd_dates.max().date()}",
        f"  warm-up                : {config.WARMUP_DAYS} days (MLB-style; "
        f"first OOF validation {fold_list[0].val_start.date()})",
        f"  last OOF validation    : {fold_list[-1].val_end.date()} "
        f"(last of {len(fold_list)} windows, fold_id {fold_list[-1].fold_id}, "
        f"n_val={len(fold_list[-1].val_idx)})",
        f"  validation windows     : {len(fold_list)} "
        f"(7-observed-date, non-overlapping, expanding training)",
        f"  training observations  : {fold_info['min_train']} (first) .. "
        f"{fold_info['max_train']} (last)",
        f"  validation observations: {fold_info['total_val_games']}",
    ]))
    # Fresh-clone safety (2026-10-01 03:35 Kaggle run): this write happens
    # in Phase 4, BEFORE the Phase 13 block that mkdirs the diagnostics dir.
    # run_diagnostics/ is deliberately gitignored (retention audit), so a
    # fresh clone does not have it and to_csv fails with "Cannot save file
    # into a non-existent directory" — mid-run, after the full refetch.
    config.RUN_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    fold_tbl.to_csv(config.RUN_DIAGNOSTICS_DIR / "nhl_fold_table.csv",
                    index=False)
    # The four oof_* blocks are computed only AFTER Phase 5's walk-forward
    # (fold_info.update(_oof_blocks(...)) below); the setdefaults above
    # exist so later readers always find the keys. Logging them HERE
    # printed {"n": 0, "sufficient": false} four times — reading as
    # "no OOF population" on a run that published 2,635 scored rows
    # (2026-10-06 log review). Log the geometry now; the real blocks are
    # logged where they are computed.
    logger.info("folds: %s (oof_* blocks: computed after Phase 5)",
                json.dumps({k: v for k, v in fold_info.items()
                            if k not in _OOF_BLOCK_KEYS}))

    # Disclosure (2026-09-30): windows whose validation population falls under
    # MIN_VAL_FOLD_GAMES are RETAINED, not skipped — the old skip silently
    # removed the playoff and season-ramp stretches from the OOF population
    # (366 of 2795 core games trained on but never validated). What cannot
    # happen is the OLD silent drop: the disclosure logged ONCE at fold
    # generation (folds.make_folds, bounded sample — 2026-10-05 run-log
    # review found this block re-logging the same verdict a second time,
    # lines 7003/7014 of nhl_pipeline_run_log.txt, the first an 840-char
    # full join of all 14 windows) and the reach gate below turn a
    # shrinking population into a loud one.
    reach = (fold_info["total_val_games"] / len(game_df)
             if len(game_df) else 0.0)
    logger.info("OOF reach: %d of %d eligible games validated (%.1f%%)",
                fold_info["total_val_games"], len(game_df), 100.0 * reach)
    if reach < 0.90:
        raise RuntimeError(
            f"OOF reach gate failed: only {fold_info['total_val_games']} of "
            f"{len(game_df)} eligible games were OOF-validated ({100*reach:.1f}% "
            f"< 90%). A walk-forward that skips windows hands the metrics a "
            "population that excludes exactly the games it serves.")

    # ── 5. Moneyline OOF ──────────────────────────────────────────────────
    _banner("PHASE 5", "moneyline walk-forward OOF")
    ml = ml_mod.walk_forward_oof(game_df, fold_list=fold_list)
    oof_ml = ml["oof"]
    weights = ml["member_weights"]
    # Phase 4's geometry table cannot contain budgets/weights until the walk
    # has earned them. Enrich the SAME diagnostic file after Phase 5.
    learned_cols = [c for c in ml["fold_table"].columns
                    if c == "xgb_rounds" or c.startswith("weight_")]
    fold_tbl = fold_tbl.merge(ml["fold_table"][["fold_id", *learned_cols]],
                              on="fold_id", how="left", validate="one_to_one")
    fold_tbl.to_csv(config.RUN_DIAGNOSTICS_DIR / "nhl_fold_table.csv", index=False)
    logger.info("moneyline OOF rows: %d; adaptive weights: %s",
                len(oof_ml), json.dumps({k: round(v, 3) for k, v in weights.items()}))

    key = game_df[["game_id", "gameday", "season", "home_team",
                   "away_team", "home_score", "away_score", "margin",
                   "total", "venue", "start_time_utc",
                   "home_record", "away_record"]].copy()
    key["gameday"] = pd.to_datetime(key["gameday"])
    oof_ml["gameday"] = pd.to_datetime(oof_ml["gameday"])
    oof_ml = oof_ml.merge(key, on=["game_id", "gameday", "season"], how="left")

    # ── 6/7. Run-line + totals OOF (joint distribution model) ─────────────
    _banner("PHASE 6-7", "goal margin/total distribution OOF")
    # The SAME fold_list object the moneyline just walked. Expected-scoring
    # folds and moneyline folds are one geometry (MLB parity) — logged so a
    # future divergence is visible in the run log rather than silent.
    dist = dist_mod.walk_forward_oof(game_df, fold_list=fold_list)
    oof_dist = dist["oof"]
    fcontract = dist.get("feature_contract", {})
    # Report the FITTED width next to the declared one. Both are in the
    # contract, but only the declared count was logged, so a frame that
    # resolved a narrower matrix read exactly like a healthy run and the
    # train/serve-skew audit this contract exists for was invisible in the log.
    logger.info("run-line feature contract: %s, %d active moneyline feature(s) "
                "+ %d tree categorical(s) = %d fitted column(s), resolved via %s",
                fcontract.get("mode"), fcontract.get("n_features"),
                len(fcontract.get("tree_categorical_cols", [])),
                fcontract.get("n_features_fitted", fcontract.get("n_features")),
                fcontract.get("resolved_via"))
    logger.info("run-line OOF folds: %d (shared with the moneyline walk-forward)",
                dist.get("n_folds", len(fold_list)))
    sig = dist_mod.calibrate_dispersion(oof_dist)
    # Report the VERDICT, not just the numbers. `calibrate_dispersion` decides
    # whether the fit sits at the Poisson limit — now evaluated over the
    # per-row α vector under the fitted α(λ) curve (MLB's curve layer), not
    # just the pooled scalar. The fit check (Pearson adequacy + deviance/RMSE
    # vs the constant league-mean baseline) makes the Poisson verdict MEASURED
    # every run, the way MLB's run-engine diagnostics do.
    _fit_check = dist_mod.run_line_fit_check(oof_dist)
    logger.info("run-line fit check (MLB diagnostics shape): "
                "home pearson %.4f dev %.5f (baseline %.5f) | away pearson %.4f "
                "dev %.5f (baseline %.5f)",
                _fit_check["home"]["pearson"], _fit_check["home"]["deviance_model"],
                _fit_check["home"]["deviance_baseline"],
                _fit_check["away"]["pearson"], _fit_check["away"]["deviance_model"],
                _fit_check["away"]["deviance_baseline"])
    logger.info("calibrated NB dispersion (MLB pooled method-of-moments): "
                "alpha_home %.4f (max %.4f), alpha_away %.4f (max %.4f) - %s",
                sig["alpha_home"], sig.get("alpha_home_max", sig["alpha_home"]),
                sig["alpha_away"], sig.get("alpha_away_max", sig["alpha_away"]),
                "POISSON LIMIT: no over-dispersion to model, the NB term is "
                "inactive and scoring is Poisson" if sig.get("poisson_limit")
                else "over-dispersed fit active")
    # Name the gate scope (MLB v3 parity): which rows the alpha layer was
    # allowed to see, and how many recent rows are sealed away from it.
    _hg = sig.get("holdout") or {}
    if _hg.get("cutoff"):
        logger.info("sealed holdout gate: alpha fitted on %s "
                    "(cutoff %s; %d pre / %d sealed rows)",
                    _hg.get("fitted_on"), _hg.get("cutoff"),
                    _hg.get("n_pre", 0), _hg.get("n_holdout", 0))

    # ── 8. Ensemble calibration (prequential OOF Platt, FAVORED space) ────
    _banner("PHASE 8", "calibration")
    y_oof = oof_ml["home_win"].to_numpy(float)
    p_ens = oof_ml["p_ensemble"].to_numpy(float)
    okp = np.isfinite(p_ens)
    _grading = (oof_ml["grades_pooled"].to_numpy(bool)
                if "grades_pooled" in oof_ml
                else np.ones(len(oof_ml), dtype=bool))
    config_meta["moneyline_evaluation"] = {
        "headline_view": "causal_rolling_blend",
        "calibration_view": "gated_prequential_causal_blend",
        "training_population": "grades_pooled",
        "retrospective_column": "p_ensemble_retrospective",
        "retrospective_is_oof": False,
        "final_xgb_rounds": ml["final_xgb_rounds"],
        "xgb_best_rounds": ml["xgb_best_rounds"],
        "xgb_round_policy": "median_prior_probe_rounds",
    }

    # Two distinct calibration layers (MLB structural parity):
    #   1. PREQUENTIAL per-fold OOF calibration (evaluation layer): fold k's
    #      calibrated predictions come from a favored-space Platt map fitted
    #      strictly on folds 0..k-1's OOF pairs, and applied ONLY when a
    #      nested holdout on that prior evidence beats the raw blend
    #      out-of-sample (2026-10-01 remediation). Ungated, a 2-parameter
    #      Platt on rolling near-calibrated evidence adds variance, not
    #      signal: the ungated layer moved logloss 0.67583->0.67731 and ECE
    #      0.01036->0.01466 on this same OOF while the pooled calibrator
    #      sat at a=1.0124 b=-0.0069.
    #   2. FINAL serving calibrator: the same eligibility and nested gate
    #      evaluated at the next origin on all prior causal OOF pairs.
    #      Retrospective final-weight replay never fits or grades a map.
    fold_ids = oof_ml["fold_id"].to_numpy()
    fold_calibrators, cal_audit = ml_mod.prequential_fold_calibrators(
        p_ens, y_oof, fold_ids, grades=_grading)
    cal_identity_ids = sorted(fid for fid, cal
                              in fold_calibrators.items() if cal is None)
    cal_fitted = len(fold_calibrators) - len(cal_identity_ids)
    cal_identity = len(cal_identity_ids)
    p_cal_prequential = np.full(len(oof_ml), np.nan)
    for fold in fold_list:
        val_mask = fold_ids == fold.fold_id
        if val_mask.any() and okp[val_mask].any():
            p_cal_prequential[val_mask] = ml_mod.moneyline_apply(
                p_ens[val_mask], fold_calibrators[int(fold.fold_id)])
    need_cal = okp & np.isnan(p_cal_prequential)
    p_cal_prequential[need_cal] = p_ens[need_cal]  # identity fallback
    oof_ml["p_ensemble_calibrated"] = p_cal_prequential
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
    # Name the identity folds, and say why. A bare count cannot distinguish
    # the expected thin start (fold 0 has no prior OOF rows at all, and the
    # earliest windows hold a handful of games) from calibration failing on
    # folds scattered through the run - which would be a real defect. Their
    # probabilities are served uncalibrated, so the count is part of what
    # the calibrated OOF metrics below are measured over. Identity now has
    # two causes: the thin start (no fittable prior evidence) and the
    # nested-holdout gate below (prior evidence existed, but its most
    # recent slice said the map adds variance). Name both populations.
    _gated_ids = sorted(r["fold_id"] for r in cal_audit["folds"]
                        if r["reason"] == "gated_no_gain")
    _thin_ids = [f for f in cal_identity_ids if f not in set(_gated_ids)]
    logger.info("prequential per-fold calibration: %d fitted, %d identity%s",
                cal_fitted, cal_identity,
                f" (folds {cal_identity_ids}: served uncalibrated — thin "
                f"start {_thin_ids} lacks fittable prior OOF rows; gated "
                f"{_gated_ids} had prior evidence but no out-of-sample "
                f"gain)" if cal_identity_ids else "")
    logger.info("prequential calib gate (nested holdout): %d of %d folds "
                "kept a map — candidate fitted on the earlier %.0f%% of "
                "prior OOF evidence, scored against raw on the most recent "
                "slice (>= %d rows), kept only if out-of-sample logloss "
                "beat raw by > %.0e nats",
                cal_audit["n_fitted"], cal_audit["n_folds"],
                100.0 * (1.0 - config.CAL_GATE_HOLDOUT_FRAC),
                config.CAL_GATE_MIN_HOLDOUT, config.CAL_GATE_EPS)

    # The SHIPPED pooled calibrator is fitted on the GRADING population only
    # (regular-season rows from windows at or above MIN_VAL_FOLD_GAMES).
    # Postseason and provisional rows stay in ``oof_ml`` -- scored, shipped,
    # reported below as their own blocks -- but they must not move a map that
    # then serves regular season.
    _g_ok = okp & _grading
    fold_info.update(_oof_blocks(oof_ml, _grading))
    logger.info("folds oof blocks: %s",
                json.dumps({k: fold_info[k] for k in _OOF_BLOCK_KEYS}))
    platt, final_cal_audit = ml_mod.gated_calibrator(p_ens, y_oof, grades=_grading)
    cal_audit["final_serving"] = final_cal_audit
    if platt is not None:
        logger.info("final pooled calibrator: a=%.4f b=%.4f n=%d method=%s",
                    platt["a"], platt["b"], platt["n"], platt["method"])
    else:
        logger.info("final pooled calibrator: identity (raw blend is served); "
                    "same causal gate: %s", final_cal_audit["reason"])

    # ── 9. Evaluation ─────────────────────────────────────────────────────
    _banner("PHASE 9", "evaluation / diagnostics")
    # Headline pooled metrics describe the GRADING population; the full
    # frame (postseason + provisional included) is still written to the OOF
    # store and to *_predictions_history, and is reported alongside as
    # oof_postseason / oof_provisional / oof_all.
    raw_m = eval_mod.binary_metrics(oof_ml["p_ensemble"][_grading],
                                    y_oof[_grading])
    # Deployed-calibrator gate decision (MLB parity): True when the nested
    # prior-evidence gate declined the pooled map and this run serves the
    # raw blend — the calibration artifact records it as method="identity".
    raw_m["calibrator_gated_out"] = bool(platt is None)
    cal_m = eval_mod.binary_metrics(oof_ml["p_ensemble_calibrated"][_grading],
                                    y_oof[_grading])
    logger.info("moneyline OOF raw:    %s (causal rolling blend)", json.dumps(raw_m))
    retrospective_m = eval_mod.binary_metrics(
        oof_ml["p_ensemble_retrospective"][_grading], y_oof[_grading])
    logger.info("moneyline retrospective final-weight replay (NOT OOF / "
                "not selection evidence): %s", json.dumps(retrospective_m))
    # Name the layer. `cal_m` is the PREQUENTIAL per-fold map (fold k fitted on
    # folds < k only), NOT the pooled calibrator that serves tonight's slate
    # (logged one phase earlier). Different fold maps can change ranking;
    # raw/calibrated AUC on these same rows ARE comparable. The favored
    # probability floor can also create ties even under a single map.
    logger.info("moneyline OOF calib:  %s (prequential per-fold layer; the "
                "pooled calibrator is what serves the slate)", json.dumps(cal_m))
    # Disclosure, not a gate (2026-09-30; layer gated 2026-10-01): the
    # per-fold map is fitted on strictly-prior evidence, so it cannot be
    # gated on its own fold's outcome (that would read the window it scores).
    # Since 2026-10-01 each fold's map is additionally applied only when a
    # nested holdout on its prior evidence beat the raw blend out-of-sample,
    # so this warning firing means the gate's own holdout did not catch a
    # variance-adding map. When the layer still measures WORSE than raw —
    # the 2026-09-30 run moved logloss 0.6747->0.6762 and ECE 0.0098->
    # 0.0137 on the same rows while the pooled serving calibrator sat
    # near-identity (a=0.968 b=0.026) — say so loudly. This is disclosure,
    # not a retrospective gate; final serving uses the same nested prior-
    # evidence gate at its own origin, never this headline comparison.
    _worse = [m for m in ("logloss", "brier", "ece")
              if isinstance(raw_m.get(m), (int, float))
              and isinstance(cal_m.get(m), (int, float))
              and cal_m[m] > raw_m[m]]
    if _worse:
        logger.warning(
            "prequential calibration layer measures WORSE than raw on %s "
            "(%s) — per-fold Platt fit on rolling prior evidence is adding "
            "variance without signal; final serving uses the same prior-"
            "evidence gate at the next origin",
            " and ".join(_worse),
            ", ".join(f"{m} {raw_m[m]:.5f}->{cal_m[m]:.5f}" for m in _worse))
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

    # daily calibration rows (frontend contract) — MLB's
    # _daily_calibration_rows semantics: the RAW series owns the ``metrics``
    # raw keys and ``buckets``, and the prequential per-fold twin rides the
    # ``*_calibrated`` metric keys plus ``buckets_calibrated`` (the shared
    # page's raw→calibrated KPI arrows and the MLB-identical row shape). The
    # old builder scored the CALIBRATED series under the raw keys and had no
    # twin at all, so the daily row could never show the calibrated side.
    daily = []
    oof_ml["gd_date"] = pd.to_datetime(oof_ml["gameday"]).dt.strftime("%Y%m%d")
    for day, grp in oof_ml.groupby("gd_date"):
        m = eval_mod.binary_metrics(grp["p_ensemble"], grp["home_win"])
        mc = eval_mod.binary_metrics(grp["p_ensemble_calibrated"],
                                     grp["home_win"])
        n_correct = int(((grp["p_ensemble_calibrated"] >= 0.5)
                         == (grp["home_win"] > 0.5)).sum())
        daily.append({
            "date": day, "n_games": m["n"],
            "wins": n_correct,
            "losses": int(m["n"] - n_correct),
            "metrics": {
                **{k: m[k] for k in ("auc", "brier", "logloss", "ece")},
                "brier_calibrated": mc["brier"],
                "logloss_calibrated": mc["logloss"],
                "ece_calibrated": mc["ece"],
            },
            "buckets": eval_mod.calibration_buckets(
                grp["p_ensemble"], grp["home_win"]),
            "buckets_calibrated": eval_mod.calibration_buckets(
                grp["p_ensemble_calibrated"], grp["home_win"]),
        })

    # ── 10. Final full-history refit ──────────────────────────────────────
    _banner("PHASE 10", "final full-history refit")
    final_models, _ = ml_mod.fit_final_models(game_df,
                                             xgb_best_rounds=ml["xgb_best_rounds"])
    final_reg = dist_mod.fit_final(game_df)
    # This phase builds the models that ACTUALLY SERVE tonight. It used to log
    # nothing at all, so a silently degraded refit (an all-NaN regressor, a
    # member that failed to fit) left no trace in the run log even though the
    # shipped joblib was already written. Report what was fitted and on what.
    _fitted = sorted(final_models) if hasattr(final_models, "__iter__") else []
    logger.info("final refit: %d moneyline member(s) %s + 1 distribution "
                "regressor on %d decided games",
                len(_fitted), _fitted or "(none)", len(game_df))
    if not _fitted:
        logger.warning("final refit produced NO moneyline members — the shipped "
                       "joblib cannot score a slate")

    # ── 11. Current-slate serving ─────────────────────────────────────────
    _banner("PHASE 11", "current-slate serving")
    slate = feat_mod.build_slate_features(schedule, boxscores)
    if len(slate):
        slate = slate.sort_values("gameday").reset_index(drop=True)
        p_home = ml_mod.predict_slate(final_models, slate, weights)
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
        goalie_df = slate[[c for c in config.GOALIE_FIELDS
                           if c in slate.columns]].reindex(columns=config.GOALIE_FIELDS)
    else:
        slate = pd.DataFrame()
        goalie_df = pd.DataFrame()
        p_home = np.array([])
        p_home_cal = np.array([])
    logger.info("slate games: %d", len(slate))

    # decided OOF rows for the markets artifact (same schema as slate rows).
    # mc_meta records the derivation's simulation resolution and whether the
    # SE-guard bumped it (MLB mc_meta parity) — it rides the markets meta.
    _mc_meta: dict = {}
    oof_market_rows = _build_oof_market_rows(oof_ml, oof_dist, sig,
                                             mc_meta_out=_mc_meta)
    oof_market_rows, market_calibration = dist_mod.calibrate_market_frame(oof_market_rows)
    # Sealed-holdout tail (MLB derive_markets_v3 parity): the last
    # HOLDOUT_DAYS of OOF rows are stamped frame_view='sealed' so the
    # monitor's winner cards and per-line metrics score them separately.
    # The sealed window never fit the alpha layer above (pre-holdout-only
    # gate in calibrate_dispersion) and never fit the final market-line
    # calibrators, so its scores are the engine's honest recent-form
    # evaluation instead of the window validating itself.
    if len(oof_market_rows) and "gameday" in oof_market_rows.columns:
        _mdates = pd.to_datetime(oof_market_rows["gameday"], errors="coerce")
        if _mdates.notna().any():
            _cutoff = (_mdates.max().normalize()
                       - pd.Timedelta(days=dist_mod.HOLDOUT_DAYS))
            oof_market_rows.loc[_mdates >= _cutoff, "frame_view"] = "sealed"
    if len(slate):
        slate = dist_mod.apply_market_calibration(slate, market_calibration)

    # ── 4.5. Record-only RFE + workbook ───────────────────────────────────
    # Stamp with run_date, NOT end_date. `end_date` is the NHL API window
    # bound and operators routinely push it past today (NHL_END_DATE=2026-09-29
    # on the 2026-09-25 run) so the slate covers upcoming games — that is a
    # LOOKAHEAD, not the run's own date. Using it here wrote
    # nhl_feature_selection_20260929.json / nhl_feature_workbook_2026-09-29.xlsx
    # four days in the future, the only NHL artifacts disagreeing with the
    # `date_c` every other artifact uses. MLB has a single date variable for
    # the window, the RFE trace and the artifact stamps, so the two cannot
    # diverge there; matching that means the RFE trace carries the run date.
    _rfe: dict = {"ran": False, "reason": "NHL_RFE_FORCE not set"}
    try:
        from feature_selection import maybe_run_rfe
        _rfe = maybe_run_rfe(game_df, run_date)
        if _rfe.get("ran"):
            logger.info("RFE: mode=%s trials=%s selected=%s trace=%s",
                        _rfe.get("run_mode"), _rfe.get("n_trials"),
                        _rfe.get("n_selected"), _rfe.get("trace"))
            from feature_workbook import generate_workbook
            _rfe["workbook"] = generate_workbook(trace_path=_rfe.get("trace"))
            logger.info("RFE workbook: %s", _rfe["workbook"] or "not written")
    except Exception as exc:
        _rfe["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("NHL RFE skipped (non-fatal): %s", exc, exc_info=True)

    # ── 12. Artifact persistence ──────────────────────────────────────────
    _banner("PHASE 12", "artifact persistence")
    artifacts: list[str] = []
    for _rfe_path in (_rfe.get("trace"), _rfe.get("workbook")):
        if _rfe_path:
            artifacts.append(Path(_rfe_path).name)
    if len(slate):
        p = out_dir / config.MONEYLINE_JSON.format(date=date_c)
        serve_mod.write_moneyline_json(p, slate, p_home, p_home_cal,
                                       _team_names(), config_meta,
                                       run_date=run_date)
        artifacts.append(p.name)

        p = out_dir / config.GOALIE_MATCHUP_JSON.format(date=date_c)
        serve_mod.write_goalie_matchup_json(p, goalie_df, slate,
                                             run_date=run_date)
        artifacts.append(p.name)

    p = out_dir / config.CALIBRATION_JSON.format(date=date_c)
    cal_rec = serve_mod.write_calibration_json(
        p, raw_m, cal_m,
        eval_mod.calibration_buckets(p_ens[_grading], y_oof[_grading]),
        daily, config_meta, platt=platt, run_date=date_c, n_games=int(_g_ok.sum()),
        calibrated_buckets=eval_mod.calibration_buckets_pair(
            p_ens[_grading], p_cal_prequential[_grading], y_oof[_grading]),
        distribution_calibration=market_calibration,
        prequential_gate=cal_audit)
    artifacts.append(p.name)

    p = out_dir / config.PREDICTIONS_HISTORY_CSV.format(date=date_c)
    serve_mod.write_predictions_history_csv(p, oof_ml,
                                            oof_ml["p_ensemble_calibrated"].to_numpy())
    artifacts.append(p.name)

    _card_store = _update_cards_history_store(out_dir, oof_ml, slate, date_c)
    if _card_store:
        artifacts.append(_card_store)

    # Repo-carried injury history: persist this run's captured ESPN snapshots
    # into data_delivery so a later sandbox run whose egress to the injury
    # endpoint is blocked (2026-09-28 Kaggle 403s) still replays real captured
    # state. Health is a data artifact, not a cache-local one.
    _inj_artifact = ingestion.export_injury_history_artifact()
    if _inj_artifact:
        artifacts.append(_inj_artifact)

    p = out_dir / config.POWER_RANKINGS_CSV.format(date=date_c)
    _write_power_rankings(p, game_df)
    artifacts.append(p.name)

    p = out_dir / config.MARKETS_CSV.format(date=date_c)
    mp_path = out_dir / config.MARKETS_META_JSON.format(date=date_c)
    serve_mod.write_markets_csv(p, mp_path, oof_market_rows, slate, config_meta,
                                mc_meta=_mc_meta, run_line_fit_check=_fit_check)
    artifacts.append(p.name)

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

    # Training/diagnostic residue is NOT delivery (2026-09-30 retention
    # audit): the OOF member/distribution stores and the fold table are
    # model-training dumps nothing reads back, so they write to the local
    # gitignored run_diagnostics/ dir instead of data_delivery/ — the
    # delivery tree carries serving artifacts and cumulative serving state
    # only. (The run summary below stays a delivery-local operational record.)
    config.RUN_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    oof_ml.to_csv(config.RUN_DIAGNOSTICS_DIR / "nhl_oof_moneyline.csv",
                  index=False)
    oof_dist.to_csv(config.RUN_DIAGNOSTICS_DIR / "nhl_oof_distribution.csv",
                    index=False)

    p = out_dir / config.FEATURE_JSON.format(date=date_c)
    _write_feature_json(p, cov, config_meta, fold_info)
    artifacts.append(p.name)

    # Per-game SHAP attributions for the current slate (display only).
    try:
        from shap_explain import compute_nhl_shap_per_game
        bundle_view = {
            "moneyline_models": {k: v["model"] for k, v in final_models.items()},
            "moneyline_preprocessors": {k: v["pre"] for k, v in final_models.items()},
            "ensemble_weights": weights,
        }
        if len(slate):
            slate_serving = slate.copy()
            slate_serving["home_win_prob_model"] = p_home_cal
            n_shap = compute_nhl_shap_per_game(bundle_view, slate_serving, out_dir)
            logger.info("SHAP attributions written: %d", n_shap)
    except Exception as exc:  # noqa: BLE001 — SHAP is display-only
        logger.warning("SHAP attribution pass skipped (non-fatal): %s", exc)

    # model bundle (joblib)
    import joblib
    bundle = {
        "moneyline_models": {k: v["model"] for k, v in final_models.items()},
        "moneyline_preprocessors": {k: v["pre"] for k, v in final_models.items()},
        "ensemble_weights": weights,
        "platt": platt,
        "moneyline_fit_policy": config_meta["moneyline_evaluation"],
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

    # The drift comparison's windows, sliced here ONCE (MLB trailing-tail
    # geometry via monitoring.drift_windows): "current" is the last N decided
    # games, "baseline" the tail of history immediately preceding them. One
    # slice so the drift step, the coverage companion, and the CSV writer all
    # describe identical populations. MONITORING ONLY — the training path is
    # untouched (expanding walk-forward over the full pool).
    drift_baseline, recent = monitoring.drift_windows(game_df)
    feature_weights = monitoring.feature_importance_weights(
        final_models, weights, feature_frame=game_df)

    # ── 13. Schema validation (gates) ─────────────────────────────────────
    # Gates run BEFORE monitoring on purpose. This module's own header argues
    # that one STDOUT stream must show one TRUE order; the gates were still
    # printed under "PHASE 14 - monitoring" because the block was written
    # first, so every run log showed 14 -> 13 and the coverage verdict read as
    # if it belonged to the unvalidated delivery. It also meant the drift /
    # coverage CSVs and the monitor JSON were written, counted into
    # `artifacts` and could be synced before anything had validated them. Both
    # blocks read only pre-Phase-12 state, so ordering them numerically is
    # free: a failed gate now aborts BEFORE monitoring writes a word.
    _banner("PHASE 13", "schema validation")
    gates = _validate_outputs(out_dir, date_c, oof_ml, slate, fold_info,
                              sig=sig, n_eligible=len(game_df))
    for name, ok in gates.items():
        logger.info("gate %-28s %s", name, "PASS" if ok else "FAIL")
    if not all(gates.values()):
        failed = [k for k, v in gates.items() if not v]
        raise RuntimeError(f"validation gates failed: {failed}")

    # ── 14. Monitoring ────────────────────────────────────────────────────
    _banner("PHASE 14", "monitoring")
    drift = monitoring.feature_drift(drift_baseline, recent,
                                     weights=feature_weights)
    # The coverage windows are the drift windows, structurally (MLB parity):
    # sliced ONCE above from the same game_df both steps read — the same
    # guaranteed-shared-frames property MLB enforced after its 08-28 incident
    # ("coverage CSV on post-slate windows the drift never saw must be
    # structurally impossible"). The report's ``current`` window is exactly
    # the population each PSI row describes, and ``baseline`` is the trailing
    # tail behind it (same recent era — see monitoring.drift_windows).
    cov_rows = monitoring.coverage(drift_baseline, slate_df=slate,
                                   current_df=recent)
    # The Phase 3 table is a single coverage_pct per feature, which counts a
    # team's FIRST game (no prior history exists — cold nulls, by design) the
    # same as a genuine defect. A run logging 14 features at "99.28%" reads as
    # a data problem when the honest verdict is that every warm game is
    # measured. This is the split, and it was computed here all along but
    # written only to the artifact — invisible to whoever is watching a run.
    _log_coverage_verdict(cov_rows)
    # The writer gets the SAME ``drift_baseline`` slice cov_rows above was
    # computed on (2026-10-06 log review: it still received the full
    # ``game_df``, so the coverage CSV baseline read 2,827 games / 285 warm
    # nulls while the log verdict and the monitor JSON — both on the
    # 250-game drift tail — read "no warm nulls"; the drift CSV disagreed
    # with the JSON drift the same way, n_baseline 2,827 vs 250). One frame,
    # all four tables: log verdict, JSON drift/coverage, CSV drift/coverage.
    run_drift_name, run_cov_name = monitoring.write_run_engine_feature_artifacts(
        out_dir, date_c, drift_baseline, recent, weights=feature_weights,
        slate_df=slate)
    artifacts.extend([run_drift_name, run_cov_name])
    rb = monitoring.rolling_brier(oof_ml)
    baseline = float(1.0 - y_oof.mean())
    p = out_dir / config.MODEL_MONITOR_JSON.format(date=date_c)
    # The shared Model Monitor page's TOTAL row reads the artifact's headline
    # `metrics` block (the same numbers the Calibration KPI cards show), so
    # hand it the block write_calibration_json just persisted -- the same dict,
    # not a second computation of the same numbers. The key was missing
    # entirely before, so the TOTAL (blended ensemble) row rendered em-dashes
    # next to fully-populated member rows.
    monitoring.write_monitor_json(p, date_c, drift, cov_rows, member_rows,
                                  rb, baseline, config_meta, fold_info,
                                  metrics=cal_rec["metrics"], platt=platt)
    artifacts.append(p.name)

    # retention: enforce the rolling-retention policy (retention_policy.py).
    _prune_old_artifacts(out_dir, date_c, seen=set(artifacts),
                         anchor_iso=end_date)

    # "written", not "produced": the manifest counts files this run put on
    # disk, which is NOT what delivery pushed. 2026-09-26 wrote 15 and pushed
    # 14 - nhl_production_cards_history.csv had 0 new rows, so its bytes were
    # unchanged and git had nothing to commit for it. Counting the manifest
    # made the run look like it had dropped an artifact.
    _banner("DONE", f"{len(artifacts)} artifacts written in "
                    f"{time.time() - t0:.0f}s")
    summary = {
        "status": "ok",
        "run_date": run_date,
        "artifacts": artifacts,
        "moneyline_oof": raw_m,
        "moneyline_oof_calibrated": cal_m,
        "moneyline_retrospective_not_oof": retrospective_m,
        "moneyline_evaluation": config_meta["moneyline_evaluation"],
        "run_line_oof": dist_metrics["run_line"],
        "totals_oof": dist_metrics["totals"],
        "weights": weights,
        "folds": fold_info,
        "n_slate": int(len(slate)),
        "rfe": _rfe,
    }
    (out_dir / "nhl_pipeline_summary.json").write_text(
        json.dumps(summary, indent=1, default=str))

    sync_result = _sync_data_delivery(repo_root=BACKEND_DIR.parent.parent)
    if sync_result["staged_files"]:
        print(
            f"NHL artifacts pushed and remotely verified: "
            f"{len(sync_result['staged_files'])} files"
        )
    else:
        print("NHL artifact sync: nothing new to push")

    # ── Final run-log delivery (2026-10-06 NHL log review, MLB parity) ──
    # _sync_data_delivery stages the run log like any other artifact — a
    # SNAPSHOT taken before the push confirmation and THIS announcement
    # ever reached the file, so every pushed log ends at the DONE banner
    # (the 2026-10-06 review found exactly that on remote: 7,262 lines
    # ending at DONE, the sync prints absent). Re-stage the now-complete
    # log and push it as the run's LAST delivery: the DONE banner above is
    # already flushed, so the staged file carries the whole run up to this
    # announcement — the file cannot contain its own push (git output
    # after the add lands in the working copy only). Never fatal: the
    # artifacts are already remotely verified, and a failed log push must
    # not fail a delivered run.
    if _log_path and os.environ.get("GITHUB_TOKEN", "").strip() \
            and not _env_flag("NHL_NO_PUSH"):
        try:
            print("  📝 Final run-log delivery — pushing the complete log "
                  "(the sync staged a snapshot before sync finished)")
            _rel = _log_path.relative_to(BACKEND_DIR.parent.parent).as_posix()

            def _git_log(*args: str):
                return subprocess.run(
                    ["git", *args], cwd=str(BACKEND_DIR.parent.parent),
                    check=True, capture_output=True, text=True)

            for _attempt in (1, 2):
                try:
                    _git_log("add", "--", _rel)
                    _git_log("-c", "user.name=NHL Production Pipeline",
                             "-c", "user.email=nhl-pipeline@users.noreply.github.com",
                             "commit", "-m", "NHL pipeline run log (final delivery)")
                    _git_log("push", "origin", "main")
                    break
                except subprocess.CalledProcessError:
                    if _attempt == 2:
                        raise
                    _git_log("pull", "--rebase", "origin", "main")
            print("  ✅ Run log pushed and remotely verified — sync output "
                  "included")
        except Exception as _log_exc:  # noqa: BLE001
            print(f"  ⚠️  Final run-log delivery did not complete: {_log_exc}")
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sync_data_delivery(repo_root: Path, branch: str = "main") -> dict:
    """Commit and push the complete NHL delivery tree without static file lists.

    Local-run guard: NHL_NO_PUSH=1 (or an absent GITHUB_TOKEN) skips the git
    delivery entirely. Only the NHL data_delivery subtree is ever staged.
    """
    delivery_rel = Path("nhl-backend") / "data_delivery"
    delivery_dir = repo_root / delivery_rel
    if _env_flag("NHL_NO_PUSH"):
        logger.info("NHL_NO_PUSH=1 - artifact delivery skipped (local run)")
        return {"staged_files": [], "skipped": "NHL_NO_PUSH=1"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        logger.warning("GITHUB_TOKEN absent - artifact delivery skipped (local run)")
        return {"staged_files": [], "skipped": "GITHUB_TOKEN absent"}
    if not delivery_dir.is_dir():
        raise RuntimeError(f"NHL delivery directory does not exist: {delivery_dir}")

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
    git("config", "user.name", os.environ.get("GIT_USER_NAME", "NHL Production Pipeline"))
    git("config", "user.email", os.environ.get("GIT_USER_EMAIL", "nhl-pipeline@users.noreply.github.com"))
    last_error = None
    for attempt in range(1, 4):
        try:
            git("reset")
            git("add", "-A", "--", delivery_rel.as_posix())
            staged = git("diff", "--cached", "--name-only", capture=True).stdout.splitlines()
            if any(not p.startswith(f"{delivery_rel.as_posix()}/") for p in staged):
                raise RuntimeError(f"NHL delivery scope violation: {staged}")

            if staged:
                git("commit", "-m", "Update NHL production artifacts")

            git("fetch", "origin", branch)
            git("rebase", f"origin/{branch}")
            git("push", "origin", branch)
            git("fetch", "origin", branch)
            remote = git(
                "ls-tree", "-r", "--name-only", f"origin/{branch}",
                delivery_rel.as_posix(), capture=True,
            ).stdout.splitlines()
            if not any(p.startswith(f"{delivery_rel.as_posix()}/") for p in remote):
                raise RuntimeError("remote NHL delivery directory is empty")
            return {"staged_files": staged}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 3:
                break
            logger.warning("NHL artifact sync attempt %d failed; retrying: %s", attempt, exc)
    raise RuntimeError(f"NHL artifact delivery failed after 3 attempts: {last_error}")


def _team_names() -> dict[str, str]:
    try:
        return ingestion.load_team_names()
    except Exception:
        return {}


def _build_oof_market_rows(oof_ml: pd.DataFrame, oof_dist: pd.DataFrame,
                           sig: dict,
                           mc_meta_out: dict | None = None) -> pd.DataFrame:
    """Decided OOF rows in the markets schema: distribution grids from the
    OOF mu pair + honest outcomes (y_*), plus the calibrated moneyline."""
    m = oof_ml[["game_id", "gameday", "season", "fold_id",
                "home_team", "away_team", "venue", "start_time_utc",
                "home_record", "away_record", "home_score", "away_score",
                "margin", "total", "home_win", "p_ensemble",
                "p_ensemble_calibrated"]].copy()
    d = oof_dist[["game_id", "mu_h", "mu_a"]].copy()
    df = m.merge(d, on="game_id", how="inner")
    if not len(df):
        return pd.DataFrame()
    df = dist_mod.apply_distribution(df, sig, meta_out=mc_meta_out)
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

    def _outcome_row(r) -> dict:
        fair_s = r.fair_spread
        fair_t = r.fair_total
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
    if outs:
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


def _oof_blocks(oof: pd.DataFrame, grading: np.ndarray) -> dict:
    """Four scored populations, all published.

    ``oof_regular``      regular-season rows from windows at or above
                         ``MIN_VAL_FOLD_GAMES`` — the GRADING set that
                         defines the headline metrics, the blend weights and
                         the pooled calibrator.
    ``oof_postseason``   every playoff row: fit and scored, never a grader.
    ``oof_provisional``  every row in a window under the gate.
    ``oof_all``          the whole OOF frame, so the split hides nothing and
                         any block can be reconciled against it.

    ``sufficient`` says whether the block clears ``MIN_VAL_FOLD_GAMES``, so a
    thin block is disclosed as thin rather than mistaken for a verdict.
    """
    keys = ("oof_regular", "oof_postseason", "oof_provisional", "oof_all")
    if oof is None or not len(oof):
        return {k: {"n": 0, "sufficient": False} for k in keys}
    gate = int(getattr(config, "MIN_VAL_FOLD_GAMES", 40))

    def _block(frame: pd.DataFrame) -> dict:
        if frame is None or not len(frame):
            return {"n": 0, "sufficient": False}
        metrics = eval_mod.binary_metrics(frame["p_ensemble_calibrated"],
                                          frame["home_win"])
        return {**metrics, "n": int(len(frame)),
                "sufficient": bool(len(frame) >= gate)}

    post = (oof[oof["is_playoffs"].astype(bool)]
            if "is_playoffs" in oof else oof.iloc[:0])
    prov = (oof[oof["provisional"].astype(bool)]
            if "provisional" in oof else oof.iloc[:0])
    return {
        "oof_regular": _block(oof[grading]),
        "oof_postseason": _block(post),
        "oof_provisional": _block(prov),
        "oof_all": _block(oof),
    }


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
                      slate: pd.DataFrame, fold_info: dict,
                      sig: dict | None = None,
                      n_eligible: int = 0) -> dict:
    """Schema/coherence gates over the written artifacts.

    ``n_eligible`` is the caller's eligible settled-game count (the 2026-09-30
    15:43 run crashed with NameError: game_df here — a main()-local name
    referenced from a helper that never receives it).
    """
    gates: dict[str, bool] = {}
    p = oof_ml["p_ensemble_calibrated"].to_numpy(float)
    p = p[np.isfinite(p)]
    gates["ml_probability_bounds"] = bool(((p >= 0) & (p <= 1)).all())
    gates["oof_population"] = bool(
        (pd.to_numeric(oof_ml["season"]) >= config.OOF_FIRST_SEASON).all())
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
                        mk.get(f"p_push_total_{U}"))
            if o is None or u is None or pu is None:
                ok = False
                break
            s = (o.fillna(0) + u.fillna(0) + pu.fillna(0))
            if not ((np.abs(s - 1.0) < 1e-6)).all():
                ok = False
                break
        gates["totals_grid_coherent"] = ok
        gates["markets_has_slate"] = (bool((mk["kind"] == "slate").any())
                                      if len(slate) else True)
    else:
        gates["spread_grid_coherent"] = False
        gates["totals_grid_coherent"] = False
        gates["markets_has_slate"] = False
    need = {"game_id", "gameday", "home_team", "away_team", "mu_h", "mu_a",
            "fair_spread", "fair_total", "p_home_win_derived",
            "p_away_win_derived"}
    gates["slate_contract_fields"] = need.issubset(slate.columns) if len(slate) else True
    gates["fold_geometry"] = fold_info.get("n_folds", 0) > 0
    # OOF reach (2026-09-30): the validation population must cover the
    # eligible games. The MIN_VAL_FOLD_GAMES skip once removed 366 of 2795
    # core games — every playoff stretch — from OOF without a single warning.
    gates["oof_reach"] = bool(
        n_eligible
        and fold_info.get("total_val_games", 0) >= 0.90 * n_eligible)
    # The dispersion fit must have run under the sealed-holdout gate: a
    # record without a cutoff means the alpha layer saw the whole OOF
    # window (an undated frame reaching production), which is exactly the
    # silent regression this gate exists to catch.
    gates["sealed_holdout_gate"] = bool(((sig or {}).get("holdout") or {}).get("cutoff"))
    # Delivery consistency (2026-10-02 postmortem): commit cf7ecc0
    # ("Rotate NHL 20260925 data-delivery artifacts", 2026-09-27)
    # deleted the 20260925 markets/predictions/monitor/coverage family
    # while leaving nhl_run_engine_markets_20260925.meta.json behind —
    # an orphaned manifest describing 2440 rows of markets that no
    # longer existed. Every schema gate still passed because each gate
    # reads files that ARE present; nothing checked a manifest against
    # its payload, and the retention window (anchor 20261001 -10d =
    # 20260921) never authorized the deletion. Pair every *.meta.json
    # with its sibling data file (and every serving board CSV with its
    # meta) so a half-deleted delivery fails the next run loudly
    # instead of shipping silently.
    orphans: list[str] = []
    for meta in sorted(out_dir.glob("*.meta.json")):
        stem = meta.name[: -len(".meta.json")]
        siblings = [f for f in out_dir.glob(stem + ".*")
                    if f.name != meta.name]
        if not siblings:
            orphans.append(meta.name)
    for board in sorted(out_dir.glob("nhl_run_engine_markets_*.csv")):
        # The board's manifest is <stem>.meta.json — the .csv
        # suffix is REPLACED, not appended (a board named
        # nhl_run_engine_markets_<d>.csv pairs with
        # nhl_run_engine_markets_<d>.meta.json).
        if not (out_dir / board.with_suffix(".meta.json").name).exists():
            orphans.append(board.name)
    gates["delivery_consistency"] = not orphans
    if orphans:
        logger.warning("delivery consistency: %d orphaned artifact(s): %s",
                       len(orphans), ", ".join(orphans[:8]))
    return gates


# Serving-era boundary for the frozen card store: the first published
# slate's game date (nhl_moneyline_v1_20260926.json's 2026-09-29 slate —
# the day the NHL serving horizon started publishing pre-game prices).
# Post-era games may only enter the store at their PUBLISHED price; OOF
# walk-forward seeding is confined to the pre-serving-era population (NFL
# store parity: "OOF seed is era-confined — post-2026-09-26 games never
# enter via OOF").
NHL_SERVING_ERA_GAME_DATE = 20260929


def _update_cards_history_store(out_dir: Path, oof_ml: pd.DataFrame,
                                slate: pd.DataFrame, date_c: str) -> str | None:
    """Append newly-decided games to the frozen card store (once).

    MLB parity (run_engine.update_totals_history_store): rows are priced at
    FIRST PUBLICATION and never mutated afterward. Returns the filename
    appended to the artifact manifest, or None on any failure.
    """
    store_path = out_dir / "nhl_production_cards_history.csv"
    meta_path = out_dir / "nhl_production_cards_history.meta.json"
    try:
        known: set[str] = set()
        if store_path.exists():
            store = pd.read_csv(store_path, dtype={"game_id": str})
            known = set(store["game_id"].astype(str))
        else:
            frames = []
            for art in sorted(out_dir.glob("nhl_predictions_history_*.csv")):
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
        # PUBLISHED LEDGER + OOF ERA GATE (NFL store parity: "the frozen-
        # price sources run FIRST … the OOF seed runs last, era-confined" —
        # the 2026-09-28 lesson: the old OOF-first order priced settle-
        # between-runs games at walk-forward re-prices instead of the
        # production prices the board had already published). The dated
        # ``nhl_moneyline_v1_*`` records ARE this pipeline's board ledger:
        # the serving horizon's own pre-game price per game (the newest
        # retained publication — exactly what the dashboard's dated board
        # serves). A decided game the ledger covers is priced FROM THE
        # LEDGER (the OOF re-price never touches the store row); a game it
        # never covered may enter via OOF only when the game PREDATES the
        # serving era — a post-era game with no publication never enters
        # the store.
        ledger: dict[str, tuple[float, str]] = {}
        for art in sorted(out_dir.glob("nhl_moneyline_v1_*.json"), reverse=True):
            try:
                rec = json.loads(art.read_text())
            except Exception:
                continue
            for g in (rec.get("games") if isinstance(rec, dict) else None) or []:
                if not isinstance(g, dict):
                    continue
                gid = str(g.get("game_id") or "")
                try:
                    lph = float(g.get("home_win_prob_model"))
                except (TypeError, ValueError):
                    continue
                if gid and gid not in ledger:
                    ledger[gid] = (lph, str(g.get("model_pick") or "").strip())
        if oof_ml is not None and len(oof_ml) and "game_id" in oof_ml.columns:
            dec = oof_ml[oof_ml["game_id"].astype(str).isin(known) == False].copy()  # noqa: E712
            dec = dec[dec[["home_score", "away_score"]].notna().all(axis=1)] \
                if {"home_score", "away_score"}.issubset(dec.columns) else dec
            if len(dec):
                gids = dec["game_id"].astype(str)
                pub_ph = gids.map({k: v[0] for k, v in ledger.items()})
                pub_pick = gids.map({k: v[1] for k, v in ledger.items()})
                oof_ph = pd.to_numeric(dec["p_ensemble_calibrated"], errors="coerce") \
                    if "p_ensemble_calibrated" in dec.columns \
                    else pd.to_numeric(dec["p_ensemble"], errors="coerce")
                gdates = pd.to_datetime(dec["gameday"], errors="coerce")
                post_era = ((gdates.dt.year * 10000 + gdates.dt.month * 100
                             + gdates.dt.day) >= NHL_SERVING_ERA_GAME_DATE)
                use_pub = pub_ph.notna()
                keep = (use_pub | ~post_era).to_numpy()
                dec = dec[keep]
                pub_ph, pub_pick = pub_ph[keep], pub_pick[keep]
                oof_ph, use_pub = oof_ph[keep], use_pub[keep]
                if len(dec):
                    # Published price first; the OOF walk-forward value only
                    # where the ledger has no row AND the game is pre-era.
                    ph = pub_ph.where(use_pub, oof_ph)
                    derived = np.where(ph >= 0.5, dec["home_team"], dec["away_team"])
                    _lp = pub_pick.fillna("").to_numpy()
                    pick = np.where(use_pub.to_numpy() & (_lp != ""), _lp, derived)
                    winner = np.where(dec["home_win"] > 0.5, dec["home_team"],
                                      np.where(dec["home_win"] < 0.5, dec["away_team"], "TIE"))
                    out = pd.DataFrame({
                        "game_id": dec["game_id"].astype(str),
                        "game_date": pd.to_datetime(dec["gameday"]).dt.strftime("%Y-%m-%d"),
                        "home_team": dec["home_team"],
                        "away_team": dec["away_team"],
                        "p_home_win": ph.round(6), "p_away_win": (1.0 - ph).round(6),
                        "model_pick": pick,
                        "correct": pick == winner,
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
    end date — NHL_END_DATE when set, else today ET), never-delete masters
    and series readers untouched, backfill-safe anchor guard, board-backed
    safety net taken over TRACKED BOARDS (the serving
    nhl_run_engine_markets_<date>.csv slate — MLB's todays_games_<date>.csv),
    so a board-backed family ages out with the 10-day board window instead of
    being reprieved by the decided population. NHL SHAP files still age by
    their GAME date, which needs the game-date map because the official NHL
    numeric game id carries no date of its own.
    """
    import retention_policy as rp

    # `classify_artifact` matches `seen` against the path RELATIVE TO
    # out_dir.parent ("data_delivery/<name>"), but `artifacts` is a manifest
    # of BARE names (`p.name`), so `rel in seen` was False for every single
    # file and the "seen - staged by this run" verdict was unreachable. The
    # 10-day window is anchored on end_date, which operators push past today
    # (2026-09-29 on the 2026-09-25 run) and which a BACKFILL sets to a
    # horizon far beyond its own run date — so a backfill's freshly written
    # artifacts fell outside the window and were unlinked seconds after being
    # written, while the run log and the summary still listed them. Normalize
    # both sides to one posix shape here; the shared predicate stays the only
    # decision-maker.
    _seen_names = {str(s).replace("\\", "/").rsplit("/", 1)[-1]
                   for s in (seen or set())}
    _dd = out_dir.name.replace("\\", "/")
    seen = set(_seen_names) | {f"{_dd}/{n}" for n in _seen_names}
    anchor = (anchor_iso or date_c).replace("-", "")
    anchor_obj = datetime.strptime(anchor, "%Y%m%d").date()
    retention_dates = {(anchor_obj - timedelta(days=i)).strftime("%Y%m%d")
                       for i in range(11)}
    recent_dates = {(anchor_obj - timedelta(days=i)).strftime("%Y%m%d")
                    for i in range(3)}

    board_dates: set[str] = set()
    game_dates: dict[str, str] = {}
    # Board-backed retention, MLB structure (mlb-backend Phase 6): board_dates
    # is the set of dates that still have a tracked BOARD artifact. MLB's
    # board is todays_games_<date>.csv; the NHL's is the serving slate,
    # nhl_run_engine_markets_<date>.csv. Those boards ride the blanket window
    # themselves, so board_dates is a strict SUBSET of retention_dates and the
    # board-backed rule is the safety net MLB documents it as ("at the 10-day
    # blanket window the slate rule dominates; this stays a safety net").
    #
    # It used to be built from the GAME dates inside the moneyline record and
    # the predictions history. That made the rule total rather than a net: the
    # NHL plays on most days, so a run-dated nhl_predictions_history_<d>.csv or
    # nhl_run_engine_markets_<d>.csv whose OWN date happened to be a game day
    # was kept forever - measured 257 and 313 days past the anchor, where MLB
    # prunes at 10. Those game dates are still needed, but only to age SHAP
    # cards, which carry an official NHL numeric game id and no date at all.
    for board in out_dir.glob("nhl_run_engine_markets_*.csv"):
        board_day = rp.artifact_date(board.name)
        if board_day:
            board_dates.add(board_day)
    for rec in out_dir.glob("nhl_moneyline_v1_*.json"):
        try:
            for g in json.loads(rec.read_text(encoding="utf-8")).get("games", []):
                d = str(g.get("game_date", ""))[:10].replace("-", "")
                gid = str(g.get("game_id", ""))
                if len(d) == 8 and d.isdigit() and gid:
                    game_dates.setdefault(gid, d)
        except Exception:
            continue

    stale: list[Path] = []
    kept_protected = 0
    kept_current = 0
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file() or p.name.startswith("~$"):
            continue
        # posix, so the key matches `seen` on Windows as well as posix.
        rel = p.relative_to(out_dir.parent).as_posix()
        old_name = p.name
        art_date = rp.artifact_date(rel)
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
        if art_date is None and not old_name.startswith("nhl_shap_game_"):
            # Tripwire (2026-09-29 13:14 postmortem): a DATELESS file that is
            # neither seen-run, protected, nor windowed just classified "stale"
            # by having no parseable date. The leave ledger was one, and the
            # pruner erased it from disk right after Phases 3/11 had loaded it
            # — retention silently destroyed its own pipeline's input. Every
            # future dateless file must be announced loudly here so a master
            # that misses EXACT_MASTER_NAMES is surfaced the run it appears,
            # instead of vanishing quietly. Classifying stale is unchanged:
            # the fix for a dateless master is registering it, not guessing.
            # (SHAP files are excluded: official numeric game ids carry no
            # date token by design — they age through the game-date map.)
            logger.warning(
                "retention: STALE DATELESS file %s (no date token, not "
                "protected, not staged this run) — deleting it is correct "
                "only if this file is neither cumulative state nor a live "
                "pipeline source; otherwise register it in "
                "retention_policy.EXACT_MASTER_NAMES", old_name)
        stale.append(p)
    if kept_protected:
        logger.info("retention: kept %d protected file(s)", kept_protected)
    logger.info("retention: kept %d artifact(s) within the window (anchor %s "
                "-10d; the anchor is the DATA WINDOW END, which a backfill "
                "can push past today - files newer than it are never touched)",
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
