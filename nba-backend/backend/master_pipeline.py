"""The single authoritative NBA production pipeline.

Run from ``nba-backend/backend``.  Every artifact and delivery path is NBA-only;
no other sport backend is imported.

Data comes from three upstreams, each asked for the one thing it is good at, in
the same division of labour MLB already runs: ESPN's scoreboard supplies the
SCHEDULE (the only one of the three that can report a game nobody has played
yet), stats.nba.com's ``LeagueGameLog`` supplies the FEATURES (one request per
season returns every player line of that season), and stats.nba.com's
``playbyplayv3`` supplies the PLAY-BY-PLAY (one request per game, counted into a
per-team event rollup that the feature ladder reads).  There is no fallback
route between them: a missing source stops the run and says which one, because a
feature set quietly assembled from a different set of games each run is not a
feature set.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from backend import (calibration as calibration_mod, config,
                        distributions as dist_mod, evaluation,
                        feature_selection, feature_workbook, features as feat_mod,
                        folds as folds_mod, ingestion, monitoring,
                        moneyline as ml_mod, player_enrichment, progress,
                        retention_policy, serving)
except ImportError:
    import config
    import distributions as dist_mod
    import evaluation
    import feature_selection
    import feature_workbook
    import features as feat_mod
    import folds as folds_mod
    import ingestion
    import monitoring
    import moneyline as ml_mod
    import player_enrichment
    import progress
    import retention_policy
    import serving

logger = logging.getLogger("nba_master_pipeline")
# ``stream=sys.stdout``, matching MLB's ``master_pipeline``. Not a style
# choice: a log stream a host may drop is how a run ends up silent while it
# works, and the run this was written for produced no visible output for ten
# minutes because nothing had said it had started.
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(message)s")

# Run-log tee (2026-10-01): capture the whole run into ONE rolling master
# file in data_delivery/ — the publish phase stages the whole delivery
# directory so the log ships like any artifact, and retention_policy
# protects the dateless name from eviction, so the latest run's full log
# is always reviewable from a plain git pull. A run that dies before the
# publish phase gets a crash-delivery push of just the log. Degrades to
# console-only on failure.
try:
    from backend.run_log_tee import (
        install_crash_log_pusher,
        install_run_log_tee,
        RUN_LOG_NAME,
    )
except ImportError:
    from run_log_tee import (
        install_crash_log_pusher,
        install_run_log_tee,
        RUN_LOG_NAME,
    )
# The tee installs at import so a run - even one that dies before the
# publish phase - carries its log. The test suite imports this module to
# reach its functions - and test_nba_rapm_features imports it at MODULE
# scope, i.e. during collection - and under pytest installing would
# TRUNCATE the committed rolling log at its own banner (measured
# 2026-10-04: one pytest batch left nba_pipeline_run_log.txt at 143
# bytes), so the install is skipped whenever a test run is in progress:
# PYTEST_CURRENT_TEST covers the in-test imports, pytest being loaded
# covers collection, and a production launch has neither.
if os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
    _log_path = None
else:
    _log_path = install_run_log_tee(config.DATA_DELIVERY_DIR)
    install_crash_log_pusher(_log_path)


# The steps of a run, in order, for the phase bar.  Named once here so the bar
# and the log agree, and so a run that dies halfway says which half it reached.
# These are descriptions, not control flow: the order below is the order the
# calls in ``run`` already happen in, and adding a name changes no result.
PHASES = ("ingest schedule", "ingest features", "ingest play-by-play",
          "features", "walk-forward", "final fit", "serve",
          "feature report", "monitor", "publish")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _flag(name: str) -> bool:
    return str(os.environ.get(name, "")).lower() in {"1", "true", "yes"}


def _config_meta(facts=None) -> dict:
    """The provenance block every published artifact carries.

    ``source`` is the route this run actually resolved, and ``player_rows`` the
    player lines behind it, so an artifact can be traced to the endpoint that
    answered rather than to a nominal upstream that may not have been read.
    """
    manifest = getattr(facts, "manifest", None) or {}
    source = manifest.get("source_route") or ingestion.SOURCE_ID
    tables = manifest.get("tables", {})
    frames = manifest.get("frame_sources", {})
    return {"sport": "nba", "feature_set_version": config.FEATURE_SET_VERSION,
            "seed": config.RANDOM_SEED, "warmup_days": config.WARMUP_DAYS,
            "cadence_days": config.RETRAIN_CADENCE_DAYS,
            "min_val_fold_games": config.MIN_VAL_FOLD_GAMES,
            "members": list(config.ENSEMBLE_MEMBERS), "market_free": True,
            "source": source,
            "player_rows": int(tables.get("player_stats", 0)),
            # Per-frame provenance, so an artifact says which upstream filled
            # each part of it rather than naming one vendor for everything.
            "frame_sources": dict(frames),
            "play_by_play_rows": int(tables.get("play_by_play", 0)),
            "event_rows": int(tables.get("team_events", 0))}


def _stamp_sealed_tail(oof_markets: pd.DataFrame) -> pd.DataFrame:
    """Stamp frame_view='sealed' on the last HOLDOUT_DAYS of OOF rows.

    Sealed-holdout tail (NHL 62d00fd / MLB derive_markets_v3 parity): the
    sealed window never fit the alpha layer (the pre-holdout-only gate
    inside ``calibrate_dispersion``), so its market scores are the engine's
    honest recent-form evaluation instead of the window validating itself.
    The per-line Platt calibrators are prequential — each fold's map fits
    prior folds only — so they were already clean for every row including
    the sealed tail. An undated frame (or absent gameday column) is
    returned untouched, exactly as the ungated dispersion fit treats it.
    """
    if oof_markets is None or not len(oof_markets) \
            or "gameday" not in oof_markets.columns:
        return oof_markets
    dates = pd.to_datetime(oof_markets["gameday"], errors="coerce")
    if not dates.notna().any():
        return oof_markets
    cutoff = (dates.max().normalize()
              - pd.Timedelta(days=dist_mod.HOLDOUT_DAYS))
    out = oof_markets.copy()
    out.loc[dates >= cutoff, "frame_view"] = "sealed"
    return out


def _oof_block(frame: pd.DataFrame) -> dict:
    """Metrics for ONE scored population of the OOF frame.

    The walk-forward publishes four populations instead of one pooled number:
    ``oof_regular`` (regular-season rows from windows at or above
    ``MIN_VAL_FOLD_GAMES`` — the GRADING set that defines the headline
    metrics, the blend weights and the Platt map), ``oof_postseason`` (the
    held-out read on the regime every fold trains on — train is strictly
    prior with no season-type filter — but which none of them graded before
    2026-10-03), ``oof_provisional`` (rows in a window under the gate) and
    ``oof_all`` (everything, so the split hides nothing). ``sufficient``
    reports whether the block clears the same gate, so a thin block is
    disclosed as thin rather than mistaken for a verdict.
    """
    if frame is None or not len(frame):
        return {"n": 0, "sufficient": False}
    metrics = evaluation.binary_metrics(
        frame.p_ensemble_calibrated.to_numpy(float),
        frame.home_win.to_numpy(float))
    return {**metrics, "n": int(len(frame)),
            "sufficient": bool(len(frame) >= config.MIN_VAL_FOLD_GAMES)}


def _merge_oof_metadata(oof: pd.DataFrame, game_df: pd.DataFrame) -> pd.DataFrame:
    if oof is None or not len(oof):
        return pd.DataFrame() if oof is None else oof
    if "game_id" not in oof.columns:
        return pd.DataFrame()
    base = oof.copy()
    # Preserve the OOF identity key; only metadata columns that would collide
    # with the richer game frame are replaced during the merge. ``is_playoffs``
    # is listed because walk_forward_oof emits its own copy for the grading
    # split -- the game frame's version is the feature the model was fed, so
    # it wins and the OOF CSV carries exactly one column under that name.
    for col in ("home_win", "gameday", "season", "is_playoffs"):
        if col in base and col in game_df:
            base = base.drop(columns=[col])
    metadata = game_df.copy()
    metadata["game_id"] = metadata.game_id.astype(str)
    base["game_id"] = base.game_id.astype(str)
    return base.merge(metadata, on="game_id", how="left", suffixes=("", "_game"))


def _marketize(base: pd.DataFrame, dist: pd.DataFrame, kind: str,
                distribution_params: dict[str, Any] | None = None,
                mc_meta_out: dict | None = None) -> pd.DataFrame:
    """Attach a complete NBA score/line distribution to a base game frame.

    ``distribution_params`` carries the OOF-estimated dispersion into the
    slate pricing path; omitting it would silently fall back to Poisson even
    after the production fit learned overdispersion. ``mc_meta_out`` receives
    the derivation's mc_meta transparency block when the grid is expanded
    here (MLB mc_meta parity; the OOF path passes it, the slate path — whose
    grid rides the calibrated OOF bundle — reuses the OOF resolution).
    """
    if base is None or not len(base) or dist is None or not len(dist):
        return pd.DataFrame()
    out = base.copy()
    dist_frame = dist.copy()
    if "game_id" in dist_frame:
        dist_frame["game_id"] = dist_frame.game_id.astype(str)
        out["game_id"] = out.game_id.astype(str)
        # Distribution frames already carry metadata in the normal pipeline;
        # retain only pricing columns for a stable merge.
        keep = [c for c in dist_frame.columns if c not in out.columns or c == "game_id"]
        dist_frame = dist_frame[keep]
        out = out.merge(dist_frame, on="game_id", how="inner", suffixes=("", "_dist"))
    # A distribution input may contain only OOF means (the normal path) or
    # be the slate frame itself.  In both cases expand it into the complete
    # line grid when no priced spread column is present yet.
    has_grid = any(str(c).startswith("p_home_cover_") for c in out.columns)
    if "mu_h" not in out or "mu_a" not in out or not has_grid:
        out = dist_mod.apply_distribution(out, distribution_params or {},
                                          meta_out=mc_meta_out)
    out["kind"] = kind
    out["decided"] = kind == "oof"
    out["frame_view"] = kind
    raw_ml = out.get("p_ensemble", out.get("p_home_win", np.nan))
    out["p_home_win"] = out.get("p_ensemble_calibrated", raw_ml)
    out["p_away_win"] = 1 - pd.to_numeric(out.p_home_win, errors="coerce")
    out["derived_ml"] = out.get("p_home_win_derived", np.nan)
    out["pred_home"] = out.get("mu_h", np.nan)
    out["pred_away"] = out.get("mu_a", np.nan)
    out["spread_line"] = np.nan
    out["total_line"] = np.nan
    out["has_offer"] = False
    for col in ("p_cover_offered", "p_push_offered", "p_over_offered",
                "p_under_offered", "p_push_total_offered"):
        out[col] = np.nan
    if "margin" in out:
        out["y_cover_fair"] = (pd.to_numeric(out.margin, errors="coerce") >
                               pd.to_numeric(out.fair_spread, errors="coerce")).astype(float)
        out["y_push_spread_fair"] = (pd.to_numeric(out.margin, errors="coerce") ==
                                     pd.to_numeric(out.fair_spread, errors="coerce")).astype(float)
        out["y_over_fair"] = (pd.to_numeric(out.total, errors="coerce") >
                              pd.to_numeric(out.fair_total, errors="coerce")).astype(float)
        out["y_push_fair"] = (pd.to_numeric(out.total, errors="coerce") ==
                              pd.to_numeric(out.fair_total, errors="coerce")).astype(float)
        out["y_under_fair"] = 1 - out.y_over_fair - out.y_push_fair
        out["y_home_win"] = out.get("home_win", np.nan)
    return out


def _build_injury_stints(facts, games: pd.DataFrame) -> pd.DataFrame | None:
    """Build the injury-stint table for the run, or None when it cannot bind.

    Mirrors MLB's arrangement: a dated record table collapsed into intervals,
    then reconciled against observed player-games. The reconciliation is the
    part that makes the rest safe - a player with a player-game row was in the
    game, so no absence may span it - and it can only SHORTEN an interval, so it
    can never release a real absence back into a projection.

    Returns None rather than an empty frame when the table cannot be built, and
    the caller degrades to the unfiltered pool. An injury filter that silently
    does not bind is indistinguishable from a league where nobody is hurt, so
    the two states must not be represented the same way.
    """
    import injury_stints as stints_mod
    teams = sorted(set(games.home_team.astype(str)) |
                   set(games.away_team.astype(str)))
    availability = ingestion.fetch_availability(teams)
    if availability is None or not len(availability):
        logger.warning("no availability records resolved; the projected "
                       "lineups fall back to the UNFILTERED player pool")
        return None
    rosters = [(date.today(), availability)]
    # Record BEFORE deriving stints. The history is the instrument: the roster
    # endpoint publishes current state only, so without an append-only store
    # every question about how a status behaved is permanently unanswerable.
    ingestion.record_injury_snapshot(availability, snapshot_date=date.today())
    records = stints_mod.records_from_rosters(rosters)
    if not len(records):
        # Everyone is healthy. That is a real state, not a missing one, and it
        # is reported as such rather than as a failure to fetch.
        logger.info("no active injury records for %d team(s); the projected "
                    "lineups are unfiltered by injury", len(teams))
        return None
    stints = stints_mod.build_stints(records)
    if not len(stints):
        return None

    # Reconcile against the observed player log. Without this a status that is
    # never cleared suppresses a player indefinitely, and the symptom is a
    # projection that is quietly missing a starter.
    appearances = facts.player_stats[["player_id", "gameday"]].copy() \
        if facts.player_stats is not None and len(facts.player_stats) \
        and {"player_id", "gameday"} <= set(facts.player_stats.columns) \
        else pd.DataFrame()
    if len(appearances):
        appearances["player_id"] = appearances.player_id.astype(str)
        appearances["gameday"] = pd.to_datetime(appearances.gameday,
                                                errors="coerce")
        before = len(stints)
        stints = stints_mod.reconcile_with_appearances(
            stints, appearances.dropna(subset=["gameday"]))
        if len(stints) != before:
            logger.info("appearance reconciliation closed %d stint(s) the "
                        "feed left open", before - len(stints))
    return stints


def _write_player_rapm(out: Path, date_c: str, facts, games: pd.DataFrame,
                      stints=None) -> str | None:
    """Build and write the player-level RAPM ratings, or report why not.

    Returns the artifact name, or None when the ratings could not be built at
    all. This never raises into the run: the ratings are published alongside
    the contract and feed nothing, so a failure to produce them is a missing
    artifact rather than a failed run. The failure is logged loudly anyway,
    because a silently-absent ratings file is indistinguishable from a league
    where nobody played.
    """
    import player_rapm as rapm_mod
    try:
        slate_dates = pd.to_datetime(games.gameday, errors="coerce").dropna()
        target = slate_dates.max() if len(slate_dates) else None
        if target is None:
            logger.warning("player RAPM skipped: the schedule has no usable date")
            return None
        seasons = sorted({ingestion.season_label(d.date())
                          for d in slate_dates})
        positions = pd.concat(
            [ingestion._fetch_positions(season) for season in seasons],
            ignore_index=True) if seasons else pd.DataFrame()
        if not len(positions):
            logger.warning("player RAPM skipped: no positions resolved for %s; "
                           "every rating would fall back to an unsegmented "
                           "prior", ", ".join(seasons) or "the window")
            return None
        teams = sorted(set(games.home_team.astype(str)) |
                       set(games.away_team.astype(str)))
        games_frame = rapm_mod.prepare_player_games(facts.player_stats, positions)
        if not len(games_frame):
            logger.warning("player RAPM skipped: the player log has no "
                           "rows to rate")
            return None
        # Availability is ANNOTATED, not applied. The rating is computed for
        # every player regardless of injury, and the removal happens at pool
        # construction, where a replacement inherits the vacated slot.
        # Multiplying the rating here instead would leave the player in the
        # pool dragging the mean toward zero.
        if stints is not None:
            import injury_stints as stints_mod
            ratings = rapm_mod.build_player_rapm(
                games_frame, target_dates=pd.Series([target]),
                team_stats=facts.team_stats)
            ratings = stints_mod.annotate_availability(ratings, stints)
        else:
            ratings = rapm_mod.build_player_rapm(
                games_frame, target_dates=pd.Series([target]),
                team_stats=facts.team_stats)
            ratings["is_available"] = True
        if not len(ratings):
            logger.warning("player RAPM skipped: no player had strictly-prior "
                           "evidence as of %s", target.date())
            return None
        # The rating row's own date doubles as the pool's gameday, so the
        # projected-lineup join and the artifact agree on one column. The
        # rename happens AFTER the artifact is written, because the CSV keeps
        # ``target_date`` as its label.
        path = out / config.PLAYER_RAPM_CSV.format(date=date_c)
        ratings.assign(target_date=pd.to_datetime(ratings.target_date)
                       .dt.strftime("%Y-%m-%d")).to_csv(path, index=False)

        ratings = ratings.rename(columns={"target_date": "gameday"})
        ratings["gameday"] = pd.to_datetime(ratings.gameday)
        if "team" not in ratings.columns:
            ratings["team"] = ""

        # The team-level aggregate: widen the pool, then filter, then rank.
        import lineup_projection as proj_mod
        aggregates = proj_mod.projected_lineup(
            ratings, games=games, stints=stints)
        if len(aggregates):
            agg_path = out / config.PLAYER_RAPM_AGG_CSV.format(date=date_c)
            aggregates.assign(gameday=pd.to_datetime(aggregates.gameday)
                              .dt.strftime("%Y-%m-%d")).to_csv(agg_path,
                                                              index=False)
            logger.info("projected lineups: %d team-game(s), mean pool %.1f, "
                        "mean healthy %.1f", len(aggregates),
                        aggregates.pool_size.mean(), aggregates.healthy_size.mean())
            # The seven diff features are attached to a COPY of the slate for
            # Reporting only. They are NOT added to MONEYLINE_FEATURE_COLS:
            # that changes the model and needs its own holdout gate. The
            # family is SUPERSEDED by the nine pl_rapm_* features, which are the
            # same idea built across the whole decided frame the way MLB's
            # lineup_agg is - and this per-target-date version covers one row
            # per run, which is why it was never promoted. The verdict ships
            # INSIDE the monitor artifact so the decision is visible without
            # the run log, not scrawled on a console nobody archives.
            import lineup_projection as slate_proj
            coverage = slate_proj.feature_coverage(
                slate_proj.attach_to_slate(games, aggregates))
            for _, row in coverage.iterrows():
                logger.info("  %-32s %d/%d rows  distinct %d%s",
                            row["feature"], int(row["populated"]),
                            int(row["rows"]), int(row["distinct_values"]),
                            "  CONSTANT" if row["constant"] else "")
            populated = int(coverage.populated.max()) if len(coverage) else 0
            rows = int(coverage.rows.max()) if len(coverage) else 0
            _superseded_lineup_family = {
                "status": "SUPERSEDED",
                "note": ("per-target-date projected-lineup diffs; superseded "
                         "by the nine pl_rapm_* features built across the whole "
                         "decided frame. Never promoted to the serving "
                         "contract, so no model depends on it."),
                "populated_rows": populated,
                "frame_rows": rows,
                "features": [str(f) for f in coverage.feature]
                            if len(coverage) else [],
            }
            try:
                (out / "nba_projected_lineup_status.json").write_text(
                    json.dumps(_superseded_lineup_family, indent=1))
            except OSError as exc:
                logger.warning("projected-lineup status not written (%s)", exc)
        return path.name
    except Exception as exc:  # noqa: BLE001
        logger.warning("player RAPM ratings not written (%s); the run continues "
                       "without them", exc)
        return None


def _concat_frames(frames, what: str) -> pd.DataFrame:
    """``pd.concat`` that reads "no frames" as an empty frame, not a ValueError.

    ``pd.concat([])`` raises ``ValueError: No objects to concatenate``. That is
    a poor answer to the question being asked - "what is in the cache?" is a
    question with an empty answer on a cold cache, and it is the ONLY answer on
    a first run - and it arrives as an exception that unwinds a phase whose
    caller then degrades a whole promoted feature family to NaN. So the empty
    case is answered here, with the reason logged, and the caller gets a frame
    it can test.
    """
    kept = [f for f in frames if f is not None and len(f)]
    if not kept:
        logger.warning("%s: nothing to read", what)
        return pd.DataFrame()
    return pd.concat(kept, ignore_index=True)


def _cache_root(cache_dir: Path | None = None) -> Path:
    """The one cache root, resolved through ``ingestion``.

    This used to read ``ingestion.CACHE_DIR``, which ingestion does not
    define - it defines ``CACHE_DIR_ENV`` and resolves the root in
    ``_cache_dir()``. The attribute lookup raised, the except arm swallowed it
    to ``None``, and every cache read below quietly became a read of nothing.
    That is the shape of bug that hides: a misspelled accessor and a broad
    ``except ImportError`` in the same function, so a genuine miss is
    indistinguishable from a deliberately empty cache.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    return Path(ingestion._cache_dir())


def _empty_contract_columns(frame: pd.DataFrame | None) -> list[str]:
    """Contract columns that no row in ``frame`` carries.

    ``n_features`` counts the CONTRACT, so a family that failed to build still
    reports its full width: the run in which all nine ``pl_epm`` columns went
    missing published "41 features" and a green tick, and the only trace of the
    loss was one WARNING line among four hundred. This is the counter-weight -
    a column that is in the contract and in no row is named in the summary the
    run publishes, so the fact travels with the artifact instead of scrolling
    past in a log.
    """
    if frame is None or not len(frame):
        return []
    return [col for col in config.active_moneyline_feature_cols()
            if col in frame.columns and not frame[col].notna().any()]


def _build_position_rapm_features(facts, games: pd.DataFrame,
                                 cache_dir: Path | None = None
                                 ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """The nine ``pl_rapm_*`` features over the WHOLE decided frame, plus slate.

    This is MLB's ``lineup_agg`` construction, translated: ratings are built
    for EVERY decided game date (not one target date - the "1 of 2,779 rows"
    blocker), the position pools are projected per (game, team), and the nine
    position-segmented columns are attached with sides retained -
    ``pl_rapm_{c,f,g}_{away,home,diff}`` - the same shape as MLB's
    ``lineup_woba_mean_{home,away}`` family, segmented by position instead of
    averaged over one pool.

    POINT-IN-TIME, in both directions the contract requires:

    * every rating row is fitted over games STRICTLY BEFORE its target date
      (the guarantee ``player_rapm._prior_for`` and ``_SeasonDesign.advance``
      implement and the audit recomputes), so a player's rating for game G
      never includes game G;
    * a player designated Out/Doubtful/Recovery in the last pre-tipoff report
      is removed from THAT game's pool only. Every other game he appears in
      keeps his rating, so his absence does not erase the rolling lagged
      rating his prior games earned him - the removal is a fact about one
      game, the rating is a fact about the games he played.

    Returns ``(frame_features, slate_features)``: the nine columns attached
    to the decided frame and to the pending slate, or ``(None, None)`` when
    the inputs cannot support the build - a missing artifact family is a
    degraded run, not a failed one.
    """
    import lineup_projection as proj_mod
    import player_rapm as rapm_mod

    seasons = sorted({rapm_mod._season_of(d) for d in games.gameday.dropna()})
    seasons = [s for s in seasons if s]
    # Positions are FETCHED, not read out of the cache directory. The old glob
    # made this phase depend on a later phase having already written the table
    # (the ratings writer, which runs at serve time), so the first run after any
    # cache wipe - a full repull, a fresh Kaggle kernel - found nothing and
    # shipped all nine promoted features as NaN while reporting the contract's
    # full 41 columns. ``_fetch_positions`` is cache-first, so a warm cache
    # costs nothing, and it is the same accessor the ratings writer calls, so
    # the two paths cannot disagree about what a position is.
    positions = _concat_frames(
        [ingestion._fetch_positions(season) for season in seasons],
        f"positions for {', '.join(seasons) or 'the window'}")
    if not len(positions):
        logger.warning("pl_rapm features skipped: no position table; every "
                       "rating would fall back to an unsegmented prior")
        return None, None
    games_frame = rapm_mod.prepare_player_games(facts.player_stats, positions)
    if not len(games_frame):
        logger.warning("pl_rapm features skipped: the player log has no "
                       "rateable rows")
        return None, None

    # id -> "First Last", the form the injury report files players under.
    name_by_id: dict = {}
    log = facts.player_stats
    for pid, pname in zip(rapm_mod._player_id_str(log.player_id),
                          log.player_name.astype(str)):
        name_by_id.setdefault(str(pid), str(pname))

    # The PIT designations: the official report's last filing strictly before
    # each tipoff, as backfilled across the decided window. Read from cache;
    # fetching the whole archive inside a run is a backfill job, not a serve
    # job. Every shard is read, never one of them - see
    # ``lineup_projection.load_designations`` for what picking a single file
    # did to the coverage. No artifact at all means no removals, and that is
    # logged rather than left to look like a healthy league.
    root = _cache_root(cache_dir)
    designations = proj_mod.load_designations(root)
    if designations is None:
        # The archive also ships IN the delivery directory (committed to the
        # repo, the way MLB ships il_stints.parquet), so a cloud run whose
        # machine cache is empty still binds the PIT removal. Delivery and
        # cache are different trees on Kaggle; checking both is what makes
        # the shipping archive more than a local convenience.
        delivery_root = Path(config.DATA_DELIVERY_DIR)
        designations = proj_mod.load_designations(delivery_root)
        if designations is not None:
            logger.info("PIT designations loaded from the delivery archive "
                        "at %s (machine cache had none)", delivery_root)
    if designations is None:
        logger.warning("no nba_designations_*.parquet in %s - the injury "
                       "removal CANNOT bind and the pools are UNFILTERED for "
                       "this run; run backfill_injury_designations.py to "
                       "restore it", root)

    decided_mask = games.home_score.notna() & games.away_score.notna()
    decided = games[decided_mask]
    pending = games[~decided_mask]

    # SLATE PARITY. Pending games resolve availability through the SAME
    # resolver history was built with: nba_injury_report.
    # game_day_designations reads the league game-day submission filing,
    # strictly before tipoff - never a late scratch, never the box
    # score. Two failure directions this closes: without it, a slate
    # date past the archive's last shard gets NO removal (the feature
    # silently off on exactly the games being bet); with a different
    # source it could read something a historical game never had.
    # Filings after this run's moment do not exist yet, so a
    # pre-submission run resolves the flagged pre-window fallback, and
    # re-running after the submission lands on the identical answer
    # history used.
    pending_dates = sorted({pd.Timestamp(d).date() for d in
                            pd.to_datetime(pending.gameday,
                                           errors="coerce").dropna().unique()})
    if pending_dates:
        designations = _slate_designations(designations, pending_dates,
                                           _pending_schedule(pending))

    dates = pd.Series(sorted(pd.to_datetime(decided.gameday).dropna().unique())
                      + sorted(pd.to_datetime(pending.gameday).dropna().unique()))
    ratings = rapm_mod.build_player_rapm(games_frame,
                                        target_dates=pd.Series(dates),
                                        team_stats=facts.team_stats)
    ratings = ratings.rename(columns={"target_date": "gameday"})
    ratings["gameday"] = pd.to_datetime(ratings.gameday)
    if "is_available" not in ratings.columns:
        ratings["is_available"] = True
    ratings = proj_mod.apply_pit_designations(ratings, designations, name_by_id)

    aggregates = proj_mod.projected_lineup(ratings, games=games)
    if not len(aggregates):
        logger.warning("pl_rapm features skipped: no team-game aggregates")
        return None, None

    def _attach(frame: pd.DataFrame) -> pd.DataFrame | None:
        if frame is None or not len(frame):
            return None
        return proj_mod.attach_position_rapm(frame, aggregates)

    return _attach(decided), _attach(pending)


SLATE_PDF_CACHE = Path(os.path.expanduser(
    "~/.cache/sports_prediction_model/nba/injury_reports"))


def _pending_schedule(pending: pd.DataFrame) -> dict:
    """``{date: [(matchup, tipoff_ET), ...]}`` for the slate being served.

    The same schedule card the backfill hands its resolver: away@home with
    the tip converted from UTC to Eastern, so a pending game is resolved by
    its identity instead of by whichever filing happens to mention it -
    history and slate pass the resolver the same ``games`` argument.  A day
    missing a start time simply falls back to filing discovery inside the
    resolver.
    """
    schedule: dict = {}
    needed = ("gameday", "away_team", "home_team", "start_time_utc")
    if pending is None or not len(pending) or not all(
            c in pending.columns for c in needed):
        return schedule
    for gameday, away, home, start in zip(pending.gameday,
                                          pending.away_team,
                                          pending.home_team,
                                          pending.start_time_utc):
        day = pd.to_datetime(gameday, errors="coerce")
        if pd.isna(day):
            continue
        try:
            tip = (pd.Timestamp(start, tz="UTC")
                   .tz_convert("America/New_York")
                   .tz_localize(None).to_pydatetime())
        except Exception:  # noqa: BLE001 - no time: filing discovery
            continue
        schedule.setdefault(day.date(), []).append((f"{away}@{home}", tip))
    return schedule


def _slate_designations(designations, dates, schedule=None):
    """Resolve pending game days with the one resolver history used.

    ``nba_injury_report.game_day_designations`` - the game-day submission
    window, first filing at/after its close, strictly before tipoff - for
    each pending date, given the slate's own schedule card (``schedule``,
    the same ``games`` argument the backfill passes history), unioned with
    the backfilled archive. The archive is authoritative for the past;
    this is the same table extended to the present, so train and serve
    differ only in which days have happened.

    Never raises: a fetch failure or an empty result leaves the archive
    untouched and logs, because an unfiltered pool is a degraded run, not
    a failed one. A date with no slate (no games, no filings) contributes
    nothing by construction.
    """
    import nba_injury_report as ir_mod
    rows = []
    for day in dates:
        try:
            frame = ir_mod.game_day_designations(
                day, SLATE_PDF_CACHE,
                games=(schedule or {}).get(day) or None, timeout=10)
        except Exception as exc:  # noqa: BLE001 - degrade loudly, not fatal
            logger.warning("slate designations for %s not resolved (%s); "
                           "that date relies on the archive", day, exc)
            continue
        if len(frame):
            rows.append(frame)
    if ir_mod.http_403_count:
        # fetch_report's contract: a run whose counter ends high must not
        # be trusted as complete - a rate-limited filing answers exactly
        # like an empty archive, so say the count where the result is read.
        logger.warning("injury-report host answered %d 403(s) resolving "
                       "the slate; designation coverage may be incomplete",
                       ir_mod.http_403_count)
    if not rows:
        logger.info("no live slate designations for %s - archive only "
                    "(pre-submission run or no filings yet)",
                    ", ".join(str(d) for d in dates))
        return designations
    live = pd.concat(rows, ignore_index=True)
    filled = int((live.status.astype(str).str.strip() != "").sum())
    logger.info("slate designations resolved live: %d record(s) across "
                "%d date(s), %d with a submission status", len(live),
                len(rows), filled)
    if designations is None or not len(designations):
        return live
    # Live rows FIRST: for a pending date the run-time read of the
    # submission is the fresher authority, and dedupe keeps the first.
    combined = pd.concat([live, designations], ignore_index=True)
    subset = [c for c in ("gameday", "team", "player")
              if c in combined.columns]
    return combined.drop_duplicates(subset=subset or None)


def _power_state(game_df: pd.DataFrame):
    events = feat_mod.team_events(game_df)
    _, ratings = feat_mod._elo_apply(events)
    decided = events[events.team_win.notna()]
    records = {}
    for team, group in decided.groupby("team"):
        records[team] = (int((group.team_win >= 0.5).sum()),
                         int((group.team_win < 0.5).sum()))
    point_diff = decided.groupby("team").net_from_team.sum().to_dict()
    return ratings, records, point_diff


def _validate_slate_contract(slate: pd.DataFrame,
                             slate_markets: pd.DataFrame) -> None:
    """Validate the two distinct pending-slate contracts before publication.

    The raw slate carries identity and model/score predictions.  Fair lines
    are produced by the distribution marketizer and therefore live on
    ``slate_markets``; ``away_win_prob_model`` is synthesized by the serving
    writer from the home probability rather than persisted on the slate.
    """
    if slate is None or not len(slate):
        return
    slate_required = {
        "game_id", "home_team", "away_team", "home_win_prob_model", "mu_h", "mu_a",
    }
    missing = slate_required - set(slate.columns)
    if missing:
        raise RuntimeError(f"NBA slate contract missing {sorted(missing)}")
    market_required = {"game_id", "fair_spread", "fair_total"}
    missing_markets = market_required - set(slate_markets.columns)
    if missing_markets:
        raise RuntimeError(
            f"NBA slate market contract missing {sorted(missing_markets)}")

    slate_ids = slate.game_id.astype(str)
    market_ids = slate_markets.game_id.astype(str)
    if slate_ids.duplicated().any() or market_ids.duplicated().any():
        raise RuntimeError("NBA slate contract contains duplicate game_id values")
    if set(slate_ids) != set(market_ids):
        raise RuntimeError("NBA slate and market contract game_id sets differ")
    for column in ("home_win_prob_model", "mu_h", "mu_a"):
        values = pd.to_numeric(slate[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"NBA slate contract has non-finite {column}")
    for column in ("fair_spread", "fair_total"):
        values = pd.to_numeric(slate_markets[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"NBA slate market contract has non-finite {column}")


def _prune(out_dir: Path, date_c: str, seen: set) -> None:
    anchor = datetime.strptime(date_c, "%Y%m%d").date()
    dates = {(anchor - pd.Timedelta(days=i).to_pytimedelta()).strftime("%Y%m%d")
             for i in range(10)}
    board = {date_c}
    for path in out_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(out_dir).as_posix()
        if retention_policy.classify_artifact(rel, seen, dates, dates, board,
                                               date_c, {}) == "stale":
            try:
                path.unlink()
            except OSError:
                pass


#: Delivery is on by default when a token is present, and off without one, so a
#: workstation run syncs nothing and says so while a Kaggle run - which has a
#: token - publishes. ``NBA_PUSH=0`` forces the skip; ``NBA_PUSH=1`` forces the
#: attempt even without a token, which then fails loudly rather than quietly.
PUSH_ENV = "NBA_PUSH"
REPO_URL_ENV = "GITHUB_REPO_URL"
TOKEN_ENV = "GITHUB_TOKEN"
BRANCH = "main"
DELIVERY_REL = "nba-backend/data_delivery"


def _push_enabled() -> tuple[bool, str]:
    """Whether to publish, and the honest reason if not."""
    raw = str(os.environ.get(PUSH_ENV, "")).strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False, f"{PUSH_ENV} is set to {raw!r}"
    if not str(os.environ.get(TOKEN_ENV, "")).strip():
        if raw in {"1", "true", "yes", "on"}:
            return True, ""  # asked for explicitly: let it fail loudly
        return False, (f"no {TOKEN_ENV} in the environment, so there is "
                       f"nothing to authenticate a push with")
    return True, ""


def _remote_url(repo_root: Path) -> str:
    """The push remote: an explicit override, else this checkout's origin.

    Read from the checkout rather than hardcoded, so the pipeline publishes to
    whatever it was cloned from - which on Kaggle is already the right repo.
    """
    override = str(os.environ.get(REPO_URL_ENV, "")).strip()
    if override:
        return override
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "remote", "get-url",
                              "origin"], capture_output=True, text=True,
                             timeout=30)
        url = out.stdout.strip()
        if out.returncode == 0 and url:
            return url
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("could not read the origin remote: %s", exc)
    return ""


def feature_importance_decomposition(
        final_models: dict[str, dict],
        weights: dict[str, float]) -> dict | None:
    """Blend-weighted feature importance, with the blend taken apart.

    The drift table's MODEL WEIGHT column used to publish an explicit
    ``{feature: 0.0}`` - a correct sum wearing a wrong answer, which every
    dashboard downstream reads as "no feature carries any of the model".
    Mirrors MLB's ``feature_importance_weights``: each member's importances
    are normalised internally, then averaged with the member's share of the
    ensemble blend.  Tree members contribute split-gain importance; the
    elastic-net member contributes |coefficient| scattered from its scaled,
    diff-sliced matrix back to active-column positions, weighted by the
    preprocessor's own stds so a zero-variance column can never smuggle its
    importance up to an unrelated feature.

    The MODEL WEIGHT column is a blend-weighted average, so when one member
    holds ~all the weight the column IS that member's importance profile
    and the others are absent from it entirely - measured on 2026-10-01, the
    drift table published ``elo_diff`` at 78.627% while the elastic net held
    82.56% of the blend with 94.4% of its own mass on that column and the
    trees' own profiles put it at 9-16%. A reader could only reconstruct
    that arithmetic by rerunning the pipeline, so the decomposition ships
    beside the column instead:

    * ``model_weight`` - the published column itself (percentages, sums
      to 100), exactly what ``feature_importance_weights`` returns;
    * ``member_shares`` - each contributing member's normalised share of
      the served blend (a member at share 0 IS recorded: being multiplied
      out of the column is the finding, not a reason to disappear);
    * ``member_profiles`` - each contributing member's OWN importance
      profile (percentages, sums to 100), so
      ``model_weight[f] = sum(member_shares[m] * member_profiles[m][f])``
      can be verified by hand.

    Returns None when no member exposes importances - the caller then omits
    the column rather than publishing a fabricated zero.
    """
    cols = config.active_moneyline_feature_cols()
    nfc = len(cols)
    agg = np.zeros(nfc)
    contributors: dict[str, float] = {}
    profiles: dict[str, dict[str, float]] = {}
    raw = {name: max(float(weights.get(name, 0.0)), 0.0) for name in final_models}
    total = sum(raw.values())
    slice_cols = feat_mod.linear_feature_columns()
    contributed = False
    for name, entry in final_models.items():
        model = entry.get("model") if isinstance(entry, dict) else entry
        pre = entry.get("pre") if isinstance(entry, dict) else None
        share = raw[name] / total if total > 0 else 1.0 / max(len(final_models), 1)
        try:
            if hasattr(model, "feature_importances_"):
                imp = np.asarray(model.feature_importances_, dtype=float).ravel()
            elif hasattr(model, "coef_"):
                coef = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
                imp = coef
                if name in ml_mod.LINEAR_MEMBERS:
                    # coef_ is slice-shaped (diff columns only, standardised).
                    # |mean|*|std| maps each slice position back to its active
                    # column in the model's own fitted geometry.
                    if (pre is None or getattr(pre, "stds", None) is None
                            or len(coef) != len(slice_cols)):
                        continue
                    scale = (pd.to_numeric(pre.stds, errors="coerce")
                             .reindex(slice_cols).to_numpy(dtype=float))
                    if (not np.all(np.isfinite(scale)) or (scale <= 0).any()):
                        continue
                    full = np.zeros(nfc)
                    index = {c: i for i, c in enumerate(cols)}
                    for col, importance in zip(slice_cols, coef * scale):
                        if col in index:
                            full[index[col]] = importance
                    imp = full
            else:
                continue
            # Tree members trained with team-ID categoricals carry a longer
            # vector; the categoricals sit AFTER the numeric active columns,
            # so the trim keeps exactly the serving width.
            if len(imp) > nfc:
                imp = imp[:nfc]
            if len(imp) != nfc or imp.sum() <= 0 or not np.all(np.isfinite(imp)):
                continue
        except Exception:  # noqa: BLE001 - one opaque member cannot kill the report
            continue
        profile = imp / imp.sum()
        agg += share * profile
        contributed = True
        # Record which members actually reached the published column. A
        # member at share 0 is multiplied out before the sum, so when the
        # blend is concentrated the "MODEL WEIGHT" column is one member's
        # opinion wearing the model's name - measured on 2026-09-29, where
        # the monitor published elo_diff at 91% while xgboost's and
        # lightgbm's own importances put it at 13% and 16% respectively.
        # The zero-share member keeps its own profile in the disclosure:
        # the trees' 9-16% is precisely the evidence that the column is
        # the elastic net's opinion, not the ensemble's.
        contributors[name] = round(share, 4)
        profiles[name] = {c: round(float(v), 4)
                          for c, v in zip(cols, profile * 100.0)}
    if not contributed or agg.sum() <= 0:
        return None
    model_weight = {c: round(float(w), 4)
                    for c, w in zip(cols, agg / agg.sum() * 100.0)}
    return {"model_weight": model_weight,
            "member_shares": contributors,
            "member_profiles": profiles}


def feature_importance_weights(final_models: dict[str, dict],
                               weights: dict[str, float]) -> dict[str, float] | None:
    """The MODEL WEIGHT column: the blended view of the decomposition.

    Thin wrapper over :func:`feature_importance_decomposition` for callers
    that want only the published column; the drift table and every existing
    consumer keep this exact contract (percentages summing to 100, None when
    no member exposes importances).
    """
    decomp = feature_importance_decomposition(final_models, weights)
    return decomp["model_weight"] if decomp else None


def _sync_data_delivery(repo_root: Path) -> dict:
    """Publish this run's artifacts to ``nba-backend/data_delivery`` on main.

    Mirrors MLB's Phase 5: a throwaway clone, the run's artifacts copied in,
    one commit, a push that retries on rejection, and a verification of the
    remote tree before the run is allowed to call itself a success.

    This used to be a no-op returning ``staged_files: []``, which meant a
    Kaggle run wrote its 19 artifacts into an ephemeral working directory and
    lost them at session end while ``data_delivery`` on ``main`` stayed empty.
    NBA was the only sport not publishing.

    One deliberate difference from MLB: the whole delivery directory is staged,
    not just files whose mtime moved. MLB needs the mtime filter because it
    ships a directory that accumulates across runs; NBA's ``_prune`` runs
    immediately before this call and has already deleted every artifact the run
    did not regenerate, so the directory's contents *are* this run's set.
    """
    delivery = DELIVERY_REL
    enabled, reason = _push_enabled()
    if not enabled:
        return {"pushed": False, "staged_files": [], "delivery_scope": delivery,
                "skipped": reason}

    source = config.DATA_DELIVERY_DIR
    if not source.exists():
        return {"pushed": False, "staged_files": [], "delivery_scope": delivery,
                "skipped": f"{source} does not exist; nothing to publish"}

    repo_url = _remote_url(repo_root)
    if not repo_url:
        raise RuntimeError(
            "NBA artifact delivery needs a remote to push to. Set "
            f"{REPO_URL_ENV} or clone with an origin remote; refusing to "
            "report a successful run whose artifacts were not published.")

    token = str(os.environ.get(TOKEN_ENV, "")).strip()
    auth_url = (repo_url.replace("https://", f"https://{token}@")
                if token and repo_url.startswith("https://") else repo_url)

    import git
    from github_sync import (push_with_retry, sync_remote_tip,
                             verify_pushed_paths)

    artifacts = sorted(p for p in source.rglob("*") if p.is_file())
    if not artifacts:
        return {"pushed": False, "staged_files": [], "delivery_scope": delivery,
                "skipped": "this run produced no artifacts to publish"}

    tmp_dir = tempfile.mkdtemp(prefix="nba_sync_")
    try:
        repo = git.Repo.clone_from(auth_url, tmp_dir, branch=BRANCH, depth=1)
        dest_root = Path(tmp_dir) / DELIVERY_REL
        dest_root.mkdir(parents=True, exist_ok=True)
        staged: list[str] = []
        for artifact in artifacts:
            rel = artifact.relative_to(source).as_posix()
            dest = dest_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact, dest)
            staged.append(f"{delivery}/{rel}")

        def _restage() -> None:
            for rel in staged:
                dest = Path(tmp_dir) / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / rel[len(delivery) + 1:], dest)
            repo.index.add(staged)
            repo.index.commit(
                f"Update NBA features + predictions: "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")

        repo.index.add(staged)
        repo.index.commit(f"Update NBA features + predictions: "
                          f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
        push_with_retry(repo, BRANCH, restage=_restage, log=print)
        verify_pushed_paths(repo, BRANCH, staged)
        return {"pushed": True, "staged_files": staged,
                "delivery_scope": delivery,
                "commit": repo.head.commit.hexsha}
    except Exception as exc:  # noqa: BLE001 - reported, then fatal below
        raise RuntimeError(
            f"NBA artifact delivery did not complete: {exc}. The artifacts are "
            f"intact under {source} on this machine; refusing to report a "
            f"successful run whose delivery failed.") from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run(run_date: str | None = None, out_dir: str | Path | None = None,
        skip_pull: bool = False) -> dict:
    # PARSER PREFLIGHT, before any work: without pdfplumber not ONE
    # game-day filing parses, so every pending date silently falls back to
    # the archive and a slate date past the archive ships UNFILTERED while
    # the run still reports ok (2026-10-04 lost all 696 filings this way
    # and filed the run green). Fail at second zero with the install line
    # instead of after an hour of ingest and fit.
    import nba_injury_report as ir_mod
    ir_mod.require_pdf_parser()
    started = time.time()
    out = Path(out_dir) if out_dir else config.DATA_DELIVERY_DIR
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    run_day = run_date or datetime.now().strftime("%Y-%m-%d")
    date_c = run_day.replace("-", "")
    # Display only.  If this never draws (no tty, no tqdm, NBA_PROGRESS=0) the
    # run below is byte-for-byte the run it was before it existed.
    prog = progress.phases(PHASES)

    def _step(name: str, done: str = "") -> None:
        """Banner a phase, tick the bar, and print MLB's ``✅`` result line.

        One call, so a phase cannot be announced without being counted and the
        banner cannot disagree with the bar about how far the run got.  The
        ``✅`` carries the numbers, which is what makes the log readable
        afterwards: a run that took ten minutes should leave behind ten minutes
        worth of evidence, not a single line at the end.
        """
        progress.banner(name)
        prog.advance(name)
        if done:
            progress.ok(done)

    progress.banner("PHASE 1-3  NBA data acquisition - ESPN schedule, "
                    "stats.nba.com features, play-by-play")
    facts = ingestion.load_ingested(
        use_cache=not skip_pull,
        allow_download=not skip_pull,
    )
    progress.ok(f"facts: {len(facts.games)} games, "
                f"{len(facts.player_stats)} player rows, "
                f"{len(facts.team_events)} event rows, "
                f"{len(facts.play_by_play)} play-by-play actions")
    # The three ingest phases are one call - ``load_ingested`` owns all three
    # upstreams - so the three ingest labels are ticked together here. They are
    # separate labels because the three upstreams fail independently, and a run
    # that died during the play-by-play sweep should say so rather than
    # reporting that feature-building failed.
    for _name in PHASES[:3]:
        prog.advance(_name)
    games = ingestion.eligible_games(facts.games)
    # Settled is not "has a score": it is "has a score AND player lines". A
    # finished game nobody in the season log played - a postponement, the NBA
    # Cup final, an all-star game - would otherwise be a training row whose
    # every feature is NaN, and a model fits NaN rather than rejecting it.
    settled = ingestion.trainable_games(games)
    pending = games[games.home_score.isna() | games.away_score.isna()].copy()
    # The slate is "games with no result yet", and a postponed game has no
    # result *because it is not being played*. Leaving it in makes the run
    # report games nobody is going to see, which is what put a January 2025
    # postponement on a board dated October 2026.  The same rule as
    # ``build_slate_features``, and deliberately so: the count this line
    # reports and the rows that get written have to agree.
    if "game_status_detail" in pending.columns and len(pending):
        postponed = pending.game_status_detail.astype(str).map(
            ingestion.sources.is_postponed_detail)
        if postponed.any():
            # The count and the id list have to agree: a head(5) here
            # once reported "11 postponed games" against five ids, which
            # reads as six games vanishing into a truncated log line.
            # Every id is listed, bounded so a pathological slate cannot
            # write an unbounded line.
            ids = sorted(pending.loc[postponed, "game_id"].astype(str))
            shown = ", ".join(ids[:25])
            if len(ids) > 25:
                shown += f" (+{len(ids) - 25} more)"
            logger.info("excluding %d postponed game(s) from the slate: %s",
                        len(ids), shown)
            pending = pending[~postponed].copy()
    if len(settled) < max(10, config.MIN_VAL_FOLD_GAMES):
        raise RuntimeError("NBA window has too few settled eligible games for walk-forward training")

    game_df = feat_mod.build_game_features(settled, facts.team_stats,
                                           facts.team_events)
    # The nine position-segmented lineup features, built across the WHOLE
    # decided frame the way MLB's lineup_agg is, and attached to the slate for
    # scoring. Before the walk-forward: the features are part of the contract
    # now, so the OOF metrics the run publishes are the metrics OF the
    # contract, not of a contract missing its newest columns. A build failure
    # degrades to NaN columns (imputed downstream) under a loud warning; it
    # never aborts the run, because a missing feature family is a worse
    # artifact, not a broken one.
    try:
        _pl_frame, _pl_slate = _build_position_rapm_features(facts, games)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pl_rapm feature build failed (%s); the nine columns "
                       "will be NaN on this run", exc)
        _pl_frame = _pl_slate = None
    if _pl_frame is not None and len(_pl_frame):
        # ``build_game_features`` pre-creates the nine columns (all-NaN, the
        # always-create rule in _attach_contract). Drop them first: merged on
        # top of them the bare names would keep the NaN versions and the real
        # values would land in suffixed copies nobody reads - the contract
        # carrying empty features with no error, which is the exact failure
        # the first A/B run exhibited.
        game_df = game_df.drop(columns=[
            c for c in config.PLAYER_RAPM_POSITION_FEATURE_COLS
            if c in game_df.columns])
        game_df = game_df.merge(
            _pl_frame[["game_id"] + config.PLAYER_RAPM_POSITION_FEATURE_COLS],
            on="game_id", how="left")
        _pl_attached = int(
            game_df[config.PLAYER_RAPM_POSITION_FEATURE_COLS[0]]
            .notna().sum())
        logger.info("pl_rapm features attached: %d/%d decided rows carry them",
                    _pl_attached, len(game_df))
        if _pl_attached < len(game_df):
            # An unattached row is NaN on all nine position columns, and
            # a bare fraction does not say WHY. The pool gates
            # (PLAYER_EPM_POOL_LOOKBACK_DAYS / RECENCY_DAYS / MIN_MINUTES)
            # have no in-season evidence at the start of a season - a
            # player's last appearances live in the PRIOR season,
            # outside the lookback - so bucketing the gap by days into
            # its season separates that structural hole from a genuine
            # build regression.
            _gaps = game_df[game_df[
                config.PLAYER_RAPM_POSITION_FEATURE_COLS[0]].isna()]
            _gap_days = pd.to_datetime(_gaps.gameday, errors="coerce")
            _all_days = pd.to_datetime(game_df.gameday, errors="coerce")
            _labels = pd.Series(
                [ingestion.season_label(d.date()) if pd.notna(d) else "?"
                 for d in _all_days], index=game_df.index)
            _season_start = _all_days.groupby(_labels).min()
            _gap_age = (_gap_days - pd.to_datetime(
                _labels.loc[_gaps.index].map(_season_start),
                errors="coerce")).dt.days
            _lookback = int(config.PLAYER_EPM_POOL_LOOKBACK_DAYS)
            _recency = int(config.PLAYER_EPM_RECENCY_DAYS)
            _n_look = int((_gap_age < _lookback).sum())
            _n_rec = int(((_gap_age >= _lookback)
                          & (_gap_age < _recency)).sum())
            _n_rest = int(len(_gaps) - _n_look - _n_rec)
            logger.info(
                "pl_rapm coverage gap: %d/%d decided rows lack position-rapm "
                "features; by days into their season: %d within the "
                "%d-day pool lookback, %d more within the %d-day "
                "recency window, %d beyond it - early-season rows are "
                "structural, not a build failure",
                len(_gaps), len(game_df), _n_look, _lookback,
                _n_rec, _recency, _n_rest)
    # Canonical (date_col, game_id) order: the one order every fold index is
    # valid for. See folds.canonical_sort for why a single-column sort is not
    # enough — fold labels are positional, and the tree members are
    # row-order sensitive under a fixed seed.
    game_df = folds_mod.canonical_sort(game_df, "gameday")
    fold_diag: dict = {}
    fold_list = folds_mod.make_folds(game_df, diagnostics=fold_diag)
    fold_info = folds_mod.fold_summary(fold_list)
    fold_info.update(fold_diag)
    for _block_key in ("oof_regular", "oof_postseason", "oof_provisional",
                       "oof_all"):
        fold_info.setdefault(_block_key, {"n": 0, "sufficient": False})
    if not fold_list:
        raise RuntimeError("NBA walk-forward produced no eligible folds after 30-day warm-up")
    _step("features", f"{len(game_df)} games, "
                      f"{len(config.active_moneyline_feature_cols())} features, "
                      f"{len(fold_list)} folds")

    # The slate frame is DATA, not model output - it needs only the schedule
    # and the stat ladders, so it is built before training and survives a
    # training failure. MLB's phase-4 shape: the model block is contained, a
    # failure inside it is recorded and the run still ships every artifact
    # the surviving phases can produce (power rankings, ratings, player RAPM,
    # the feature contract, drift/coverage) and exits with status "failed"
    # so the scheduler files it red. A crash here used to abort ``run``
    # outright, and a run that trains on Monday publishes nothing on Monday -
    # the board went stale even though the data side of the pipeline was
    # perfectly healthy.
    slate = (feat_mod.build_slate_features(games, facts.team_stats,
                                           facts.team_events)
             if len(pending) else pd.DataFrame())
    if len(slate) and _pl_slate is not None and len(_pl_slate):
        # The upcoming games get their own projections - including the PIT
        # injury removal, which binds hardest here: the slate is exactly the
        # "upcoming game" the designation names. Same pre-drop as game_df:
        # the slate's own NaN placeholders must not win the name collision.
        slate = slate.drop(columns=[
            c for c in config.PLAYER_RAPM_POSITION_FEATURE_COLS
            if c in slate.columns])
        slate = slate.merge(
            _pl_slate[["game_id"] + config.PLAYER_RAPM_POSITION_FEATURE_COLS],
            on="game_id", how="left")

    phase_error: Exception | None = None
    ml: dict | None = None
    ml_oof = platt = dist_oof = dispersion = None
    oof_markets = market_calibration = None
    _fit_check: dict | None = None
    _mc_meta: dict = {}
    final_models: dict = {}
    final_reg = None
    slate_markets = pd.DataFrame()
    leaders = pd.DataFrame()
    try:
        ml = ml_mod.walk_forward_oof(game_df, fold_list=fold_list)
        ml_oof = _merge_oof_metadata(ml["oof"], game_df)
        if not len(ml_oof):
            raise RuntimeError("NBA moneyline walk-forward produced no OOF rows")
        # The SHIPPED Platt map is fitted on the GRADING population only:
        # regular-season rows from windows at or above MIN_VAL_FOLD_GAMES.
        # Fitting it on every OOF row would let 169 postseason rows and 16
        # provisional windows move a calibrator that then serves regular
        # season — the exact train/score mismatch this split removes.
        _grades = (ml_oof["grades_pooled"].astype(bool)
                   if "grades_pooled" in ml_oof
                   else pd.Series(True, index=ml_oof.index))
        _grade = ml_oof[_grades]
        platt = ml_mod.moneyline_fit(_grade.p_ensemble.to_numpy(float),
                                     _grade.home_win.to_numpy(float))
        _post = (ml_oof[ml_oof.is_playoffs.astype(bool)]
                 if "is_playoffs" in ml_oof else ml_oof.iloc[:0])
        fold_info["oof_regular"] = _oof_block(_grade)
        fold_info["oof_postseason"] = _oof_block(_post)
        fold_info["oof_provisional"] = _oof_block(
            ml_oof[ml_oof.provisional.astype(bool)]
            if "provisional" in ml_oof else ml_oof.iloc[:0])
        fold_info["oof_all"] = _oof_block(ml_oof)
        if "p_ensemble_calibrated" not in ml_oof:
            ml_oof["p_ensemble_calibrated"] = ml_mod.moneyline_apply(
                ml_oof.p_ensemble.to_numpy(float), platt)
        ml_oof["p_ensemble_calibrated"] = ml_oof.p_ensemble_calibrated.fillna(
            ml_oof.p_ensemble)

        dist = dist_mod.walk_forward_oof(game_df, fold_list=fold_list)
        dist_oof = _merge_oof_metadata(dist["oof"], game_df)
        dispersion = dist_mod.calibrate_dispersion(dist_oof)
        # The run line's fit is now MEASURED every run the way MLB's run
        # engine reports its own: the Pearson Poisson-adequacy probe plus
        # pooled deviance/RMSE against the constant league-mean baseline.
        # "Poisson limit" stops being an unmeasured assertion — the probe
        # either confirms adequate Poisson variance or names the
        # over-dispersion the NB term should absorb.
        _fit_check = dist_mod.run_line_fit_check(dist_oof)
        logger.info("run-line fit check: "
                    "home pearson %.4f dev %.5f (baseline %.5f) | away pearson %.4f "
                    "dev %.5f (baseline %.5f)",
                    _fit_check["home"]["pearson"], _fit_check["home"]["deviance_model"],
                    _fit_check["home"]["deviance_baseline"],
                    _fit_check["away"]["pearson"], _fit_check["away"]["deviance_model"],
                    _fit_check["away"]["deviance_baseline"])
        logger.info("calibrated NB dispersion (MLB pooled method-of-moments): "
                    "alpha_home %.4f (curve max %.4f), alpha_away %.4f "
                    "(curve max %.4f) - %s",
                    dispersion["alpha_home"],
                    dispersion.get("alpha_home_max", dispersion["alpha_home"]),
                    dispersion["alpha_away"],
                    dispersion.get("alpha_away_max", dispersion["alpha_away"]),
                    "POISSON LIMIT: no over-dispersion to model, the NB term is "
                    "inactive and scoring is Poisson" if dispersion.get("poisson_limit")
                    else "over-dispersed fit active")
        # Name the gate scope (NHL 62d00fd / MLB v3 parity): which rows the
        # alpha layer was allowed to see, and how many recent rows are
        # sealed away from it.
        _hg = dispersion.get("holdout") or {}
        if _hg.get("cutoff"):
            logger.info("sealed holdout gate: alpha fitted on %s "
                        "(cutoff %s; %d pre / %d sealed rows)",
                        _hg.get("fitted_on"), _hg.get("cutoff"),
                        _hg.get("n_pre", 0), _hg.get("n_holdout", 0))
        # mc_meta rides the markets meta: the derivation records its own MC
        # resolution and whether the SE-guard bumped it (MLB mc_meta parity).
        oof_markets = _marketize(ml_oof, dist_oof, "oof", dispersion,
                                 mc_meta_out=_mc_meta)
        oof_markets, market_calibration = dist_mod.calibrate_market_frame(oof_markets)
        oof_markets = _stamp_sealed_tail(oof_markets)
        _step("walk-forward", f"{len(ml_oof)} out-of-fold rows over "
                              f"{len(fold_list)} folds")

        final_models, _ = ml_mod.fit_final_models(game_df)
        final_reg = dist_mod.fit_final(game_df)
        if len(slate):
            slate["home_win_prob_model"] = ml_mod.predict_slate(
                final_models, slate, ml["member_weights"])
            slate["p_ensemble"] = slate.home_win_prob_model
            slate["p_ensemble_calibrated"] = ml_mod.moneyline_apply(
                slate.home_win_prob_model.to_numpy(float), platt)
            slate["mu_h"], slate["mu_a"] = final_reg.predict(slate)
            slate_markets = _marketize(slate, slate, "slate", dispersion)
            slate_markets = dist_mod.apply_market_calibration(slate_markets,
                                                               market_calibration)
            leaders = player_enrichment.build_player_leader(slate, facts.player_stats, games)
            slate = slate.merge(leaders[["game_id", *config.PLAYER_FIELDS]],
                                on="game_id", how="left", suffixes=("", "_leader"))
        _step("final fit", f"{len(final_models)} ensemble member(s), "
                           f"{len(slate)} upcoming game(s) scored")
    except Exception as exc:  # noqa: BLE001 - recorded, the run still ships
        phase_error = exc
        logger.error("phase 4 (training + prediction) failed: %s - the run "
                     "ships its remaining artifacts and exits failed", exc)
    if phase_error is None and not len(final_models):
        phase_error = RuntimeError("final fit produced no ensemble members")

    artifacts: list[str] = []
    cal_metrics: dict = {}
    if phase_error is None:
        # Everything from here to the monitor needs a trained model. On a
        # training failure these families are skipped under one loud line -
        # the same trade MLB's phase-4 containment makes - while the data
        # families below still ship.
        p_ml = out / config.MONEYLINE_JSON.format(date=date_c)
        serving.write_moneyline_json(
            p_ml, slate,
            slate.get("home_win_prob_model", pd.Series(dtype=float)),
            slate.get("p_ensemble_calibrated", pd.Series(dtype=float)),
            facts.team_names, _config_meta(facts), leaders)
        artifacts.append(p_ml.name)
        p_player = out / config.PLAYER_MATCHUP_JSON.format(date=date_c)
        serving.write_player_matchup_json(p_player, leaders)
        artifacts.append(p_player.name)

        # Headline pooled metrics describe the GRADING population. The full
        # frame (including postseason + provisional rows) is still written to
        # nba_oof_moneyline.csv / *_predictions_history and is reported
        # separately as oof_postseason — see _oof_season_split below.
        raw_metrics = evaluation.binary_metrics(_grade.p_ensemble.to_numpy(float),
                                                _grade.home_win.to_numpy(float))
        cal_metrics = evaluation.binary_metrics(_grade.p_ensemble_calibrated.to_numpy(float),
                                                _grade.home_win.to_numpy(float))
        buckets = evaluation.calibration_buckets(_grade.p_ensemble.to_numpy(float),
                                                  _grade.home_win.to_numpy(float))
        p_cal = out / config.CALIBRATION_JSON.format(date=date_c)
        serving.write_calibration_json(p_cal, raw_metrics, cal_metrics, buckets, [],
                                       _config_meta(facts), platt, run_day,
                                       len(_grade),
                                       market_calibration)
        artifacts.append(p_cal.name)
        p_hist = out / config.PREDICTIONS_HISTORY_CSV.format(date=date_c)
        serving.write_predictions_history_csv(p_hist, ml_oof,
                                              ml_oof.p_ensemble_calibrated.to_numpy(float))
        artifacts.append(p_hist.name)
        # Stable OOF stores support audits without making the frontend depend on
        # an unfiltered in-memory frame.
        ml_oof.to_csv(out / "nba_oof_moneyline.csv", index=False)
        dist_oof.to_csv(out / "nba_oof_distribution.csv", index=False)
        artifacts += ["nba_oof_moneyline.csv", "nba_oof_distribution.csv"]
    folds_mod.fold_table(game_df, fold_list).to_csv(out / "nba_fold_table.csv", index=False)
    artifacts.append("nba_fold_table.csv")

    if phase_error is None:
        p_markets = out / config.MARKETS_CSV.format(date=date_c)
        serving.write_markets_csv(p_markets,
                                  out / config.MARKETS_META_JSON.format(date=date_c),
                                  oof_markets, slate_markets, _config_meta(facts),
                                  mc_meta=_mc_meta,
                                  run_line_fit_check=_fit_check)
        artifacts += [p_markets.name,
                      (out / config.MARKETS_META_JSON.format(date=date_c)).name]

    ratings, records, point_diff = _power_state(settled)
    p_rank = out / config.POWER_RANKINGS_CSV.format(date=date_c)
    serving.write_power_rankings_csv(p_rank, ratings, records, facts.team_names, point_diff)
    artifacts.append(p_rank.name)

    # Player-level RAPM, shrunk toward a position-segmented prior.
    # Published ALONGSIDE the feature contract and deliberately not fed to the
    # ensemble, for the same reason NHL's player ratings are not: adding a
    # column to MONEYLINE_FEATURE_COLS changes the model and needs its own
    # holdout validation, which is a separate decision from building the
    # rating. The rating is PIT (strictly prior rows, season-partitioned), so
    # it is safe to publish now and to promote later behind a gate.
    rapm_name = _write_player_rapm(out, date_c, facts, games,
                                 stints=_build_injury_stints(facts, games))
    if rapm_name:
        artifacts.append(rapm_name)
    coverage = feat_mod.feature_coverage_report(game_df)
    p_feat = out / config.FEATURE_JSON.format(date=date_c)
    serving.write_feature_json(p_feat, coverage, _config_meta(facts), fold_info)
    artifacts.append(p_feat.name)

    # Per-game SHAP attributions for the current slate (display only).  NHL
    # ships these and its game cards render them; NBA's cards have an
    # expander wired to a file family this pipeline never produced.  Built
    # from the same fitted members the bundle persists, so a card's chart
    # explains the model that actually serves it.  Display-only and wrapped:
    # a SHAP failure degrades to cards without expanders, never a red run.
    try:
        from shap_explain import compute_nba_shap_per_game
        bundle_view = {
            "moneyline_models": {name: entry["model"] for name, entry in final_models.items()},
            "moneyline_preprocessors": {name: entry["pre"] for name, entry in final_models.items()},
            "ensemble_weights": ml["member_weights"] if ml else {},
        }
        if phase_error is None and len(slate):
            slate_serving = slate.copy()
            slate_serving["home_win_prob_model"] = slate.get("p_ensemble_calibrated")
            n_shap = compute_nba_shap_per_game(bundle_view, slate_serving, out)
            if n_shap:
                artifacts.extend(
                    f"{config.SHAP_GAME_PREFIX}_{gid}.csv"
                    for gid in slate["game_id"].astype(str))
            logger.info("SHAP attributions written: %d", n_shap)
    except Exception as exc:  # noqa: BLE001 - display-only, never fatal
        logger.warning("SHAP attribution pass skipped (non-fatal): %s", exc)

    _step("serve", f"{len(artifacts)} artifact(s) written to {out}")

    selection = feature_selection.run_rfe(game_df, out, date_c)
    selection_name = f"nba_feature_selection_{date_c}.json"
    workbook_name = feature_workbook.write_feature_workbook(out, date_c, selection)
    if workbook_name:
        artifacts.append(workbook_name)
    artifacts.append(selection_name)
    # Drift on MLB's window pair: a recent tail against its like-for-like
    # prior, never the whole history against itself - comparing a
    # playoff-heavy 60-row tail to a full season paged eleven features whose
    # means had not moved.  Weights are the members' real blend-weighted
    # importances, not a table of zeros.
    imp_decomp = (feature_importance_decomposition(
        final_models, ml["member_weights"]) if ml else None)
    imp_weights = imp_decomp["model_weight"] if imp_decomp else None
    # Say out loud which members the MODEL WEIGHT column actually averages.
    # A concentrated blend means the column is one member's profile, and a
    # reader comparing it against per-member importances (elasticnet put
    # elo_diff at 94% while the trees put it at 13%/16% on 2026-09-29, and
    # 94.4% against the trees' 9%/6% on 2026-10-01) would otherwise have no
    # way to tell a real concentration from a reporting artifact. The
    # per-member elo_diff importance rides the log line so the decomposition
    # is visible without opening the artifact.
    if ml and imp_decomp:
        _shares = imp_decomp["member_shares"]
        _profiles = imp_decomp["member_profiles"]
        _elo = {n: p.get("elo_diff") for n, p in _profiles.items()}
        logger.info("feature importance weights are blend-weighted across "
                    "%d member(s) with shares %s; each member's own "
                    "elo_diff importance %s - a concentrated share means "
                    "the MODEL WEIGHT column is that member's profile",
                    len(_shares), _shares, _elo)
    drift_baseline, drift_current = monitoring.drift_windows(game_df)
    drift_names = monitoring.write_run_engine_feature_artifacts(
        out, date_c, drift_baseline, drift_current, imp_weights)
    artifacts.extend(drift_names)
    _step("feature report", f"selection {selection_name}, "
                            f"{len(drift_names)} drift/coverage file(s)")

    import joblib
    if phase_error is None:
        # The persisted bundle IS the serving model. A failed fit must never
        # overwrite yesterday's good bundle with an empty-member dict - the
        # board would keep "working" against a model that predicts nothing.
        bundle = {
            "moneyline_models": {name: entry["model"] for name, entry in final_models.items()},
            "moneyline_preprocessors": {name: entry["pre"] for name, entry in final_models.items()},
            "ensemble_weights": ml["member_weights"], "platt": platt,
            "score_regressor": final_reg, "distribution": dispersion,
            "market_calibration": market_calibration,
            "feature_set_version": config.FEATURE_SET_VERSION,
            "feature_columns": config.active_moneyline_feature_cols(),
            "trained_utc": _now(), "config": _config_meta(facts),
        }
        model_path = model_dir / "nba_ensemble_latest.joblib"
        joblib.dump(bundle, model_path)
        artifacts.append(str(model_path.relative_to(out)))
    else:
        logger.warning("model bundle NOT rewritten: the previously served "
                       "ensemble stays in place for serving")

    members = (monitoring.ensemble_table(ml_oof, ml["member_weights"])
               if ml_oof is not None and ml else [])
    drift = monitoring.feature_drift(drift_baseline, drift_current, imp_weights)
    cov = monitoring.coverage(drift_baseline, drift_current)
    brier = monitoring.rolling_brier(ml_oof) if ml_oof is not None else []
    # The old line printed the LAST DAY's Brier under an aggregate-sounding
    # label; the two 2026-09-29 runs read 0.2589 -> 0.2713 and looked like a
    # regression while both numbers were ONE game and the same-window
    # games-weighted mean had IMPROVED. brier_headline names the last day
    # separately and leads with the games-weighted mean so the log stops
    # inviting that false alarm.
    brier_head = monitoring.brier_headline(brier)
    latest_brier = brier_head["summary"] if brier else "n/a"
    baseline_brier = (float(1 - ml_oof.home_win.mean())
                      if ml_oof is not None and len(ml_oof) else None)
    monitoring.write_monitor_json(
        out / config.MODEL_MONITOR_JSON.format(date=date_c), date_c, drift, cov,
        members, brier,
        baseline_brier, _config_meta(facts), fold_info,
        cal_metrics, platt, feature_importance=imp_decomp)
    artifacts.append(config.MODEL_MONITOR_JSON.format(date=date_c))
    monitoring.write_run_engine_monitor(
        out / config.MARKETS_MONITOR_JSON.format(date=date_c), date_c,
        (evaluation.nb_distribution_metrics(dist_oof, dispersion)
         if dist_oof is not None and dispersion is not None else {}),
        oof_markets if oof_markets is not None else {},
        market_calibration, _config_meta(facts), dispersion=dispersion)
    artifacts.append(config.MARKETS_MONITOR_JSON.format(date=date_c))

    if len(slate):
        cards = slate.copy()
        cards["p_home_win"] = pd.to_numeric(cards.get("p_ensemble_calibrated"), errors="coerce")
        cards["p_away_win"] = 1 - cards.p_home_win
        serving.write_production_cards_history(
            out / "nba_production_cards_history.csv", cards)
        artifacts.append("nba_production_cards_history.csv")

    if len(slate) and len(slate_markets):
        _validate_slate_contract(slate, slate_markets)
    _step("monitor", f"rolling Brier: {latest_brier}, "
                     f"{len(members)} ensemble member(s)")

    empty_frame = _empty_contract_columns(game_df)
    empty_slate = _empty_contract_columns(slate)
    if empty_frame or empty_slate:
        logger.error(
            "contract features carrying no value: frame %s | slate %s - "
            "anything listed is in the published contract and in no row, so "
            "the model was fitted (and the slate scored) without it",
            ", ".join(empty_frame) or "none", ", ".join(empty_slate) or "none")
    summary = {"status": "failed" if phase_error else "ok",
               "errors": [str(phase_error)] if phase_error else [],
               "run_date": run_day, "artifacts": artifacts,
               "weights": ml["member_weights"], "folds": fold_info,
               "n_settled": len(settled), "n_slate": len(slate),
               "n_features": len(config.active_moneyline_feature_cols()),
               "features_empty": {"frame": empty_frame, "slate": empty_slate},
               "play_by_play": {
                   "actions": int(len(facts.play_by_play)),
                   "event_rows": int(len(facts.team_events)),
                   "games_covered": int(facts.team_events.game_id.nunique())
                   if len(facts.team_events) else 0,
                   "cross_check": facts.manifest.get("play_by_play", {}),
               },
               "elapsed_seconds": round(time.time() - started, 2),
               "source_manifest": facts.manifest}
    # The PIT designation archive travels with the repo, the way MLB ships
    # ``il_stints.parquet``: a fresh Kaggle clone starts from an EMPTY machine
    # cache, and a designation archive that exists only there means every
    # cloud run projects UNFILTERED lineups - the injury removal silently off
    # for the run that most needs it. The delivery directory is the one
    # artifact sink the sync phase already publishes, so the consolidated
    # table lives here too, rewritten each run and never pruned.
    try:
        import lineup_projection as designation_mod
        archive_root = _cache_root()
        designations = designation_mod.load_designations(archive_root)
        if designations is None:
            # This run itself may have loaded from the delivery copy; the
            # archive to publish is whatever the shards or that copy hold.
            designations = designation_mod.load_designations(
                Path(config.DATA_DELIVERY_DIR))
        if designations is not None and len(designations):
            designations.drop_duplicates(
                subset=[c for c in ("gameday", "team", "player")
                        if c in designations.columns]).to_parquet(
                out / "nba_designations.parquet", index=False)
            logger.info("designation archive published: %d record(s)",
                       len(designations))
    except Exception as exc:  # noqa: BLE001 - publication is best-effort
        logger.warning("designation archive not published (%s); the PIT "
                       "removal falls back to the machine cache only", exc)
    # The event rollup archive: the union of every team-game rollup this run
    # produced plus whatever earlier runs shipped, republished into the
    # delivery so the next run - on this machine or an ephemeral cloud one -
    # starts its event features from measurements instead of a cold cache.
    # Same pattern as the designation archive above, which is what made the
    # PIT injury removal survive cloud runs. The per-game parquet cache is
    # machine-local; the sweep budget's work should compound, not evaporate
    # with the host - which is exactly what the 2026-09-29 runs showed (every
    # cloud run re-fetched the same 1,500 newest games and the 2024 band
    # stayed forward-filled forever).
    try:
        archive = ingestion.read_event_rollup_archive(
            Path(config.DATA_DELIVERY_DIR))
        fresh = facts.team_events
        base = archive if archive is not None and len(archive) else pd.DataFrame()
        if fresh is not None and len(fresh):
            fresh = fresh.copy()
            fresh["game_id"] = fresh.game_id.astype(str)
            fresh["team"] = fresh.team.astype(str)
            if not base.empty:
                keep = ~fresh.game_id.str.cat(fresh.team, sep="|").isin(
                    set(base.game_id.str.cat(base.team, sep="|")))
                fresh = fresh[keep]
            union = (pd.concat([base, fresh], ignore_index=True)
                     if not base.empty else fresh)
        else:
            union = base
        if len(union):
            union.to_parquet(out / ingestion.EVENT_ROLLUP_ARCHIVE, index=False)
            logger.info("event rollup archive published: %d team-game(s)",
                        len(union))
    except Exception as rollup_exc:  # noqa: BLE001 - publication is best-effort
        logger.warning("event rollup archive not published (%s); the next "
                       "run's event coverage falls back to this machine's "
                       "cache alone", rollup_exc)
    _prune(out, date_c, set(artifacts))
    # Write the summary BEFORE the sync, the order NFL and NHL already use.
    # _prune has just deleted every artifact this run did not regenerate -
    # the previous run's summary with it - so a summary written after the
    # sync never ships: the ephemeral Kaggle session drops it at session end
    # and main keeps serving the stale copy from the last run whose summary
    # happened to be committed (the 2026-10-01 run exposed main frozen at
    # 2026-09-27 while five later runs pushed fresh artifacts around it).
    # Written here, the summary ships with this run's own delivery; the sync
    # result is attached to the in-memory summary for stdout only.
    (out / "nba_pipeline_summary.json").write_text(
        json.dumps(summary, indent=1, default=str))
    # The publish step prints BEFORE the sync stages the delivery, because
    # the sync commits the log file itself: anything printed after it exists
    # only on this machine, which is why every delivered log ended mid-bar at
    # "monitor 9/10" with no completion line (both 2026-10-02 runs - the
    # banner, the final tick, and this result line all landed after the
    # staged snapshot). The sync RESULT stays stdout-only under the same
    # rule the summary's "sync" key already accepts.
    _step("publish", f"status {summary['status']}, "
                     f"{summary['elapsed_seconds']}s")
    sync = _sync_data_delivery(config.ROOT_DIR.parent)
    summary["sync"] = sync
    prog.close()
    return summary


def main(argv=None):
    """CLI entry: run, print the summary, and exit non-zero on failure.

    The summary JSON carries ``status: "failed"`` for a machine reader; the
    non-zero exit is for the scheduler wrapper, which must not file a red run
    as a green one.
    """
    parser = argparse.ArgumentParser(description="NBA production pipeline")
    parser.add_argument("--run-date", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--skip-pull", action="store_true")
    args = parser.parse_args(argv)
    summary = run(args.run_date, args.out_dir, args.skip_pull)
    print(json.dumps(summary, indent=1, default=str))
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
