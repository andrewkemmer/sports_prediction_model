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


def _merge_oof_metadata(oof: pd.DataFrame, game_df: pd.DataFrame) -> pd.DataFrame:
    if oof is None or not len(oof):
        return pd.DataFrame() if oof is None else oof
    if "game_id" not in oof.columns:
        return pd.DataFrame()
    base = oof.copy()
    # Preserve the OOF identity key; only metadata columns that would collide
    # with the richer game frame are replaced during the merge.
    for col in ("home_win", "gameday", "season"):
        if col in base and col in game_df:
            base = base.drop(columns=[col])
    metadata = game_df.copy()
    metadata["game_id"] = metadata.game_id.astype(str)
    base["game_id"] = base.game_id.astype(str)
    return base.merge(metadata, on="game_id", how="left", suffixes=("", "_game"))


def _marketize(base: pd.DataFrame, dist: pd.DataFrame, kind: str,
                distribution_params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Attach a complete NBA score/line distribution to a base game frame.

    ``distribution_params`` carries the OOF-estimated dispersion into the
    slate pricing path; omitting it would silently fall back to Poisson even
    after the production fit learned overdispersion.
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
        out = dist_mod.apply_distribution(out, distribution_params or {})
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


def _write_player_ts(out: Path, date_c: str, facts, games: pd.DataFrame,
                     stints=None) -> str | None:
    """Build and write the player-level TS ratings, or report why not.

    Returns the artifact name, or None when the ratings could not be built at
    all. This never raises into the run: the ratings are published alongside
    the contract and feed nothing, so a failure to produce them is a missing
    artifact rather than a failed run. The failure is logged loudly anyway,
    because a silently-absent ratings file is indistinguishable from a league
    where nobody played.
    """
    import player_ts as ts_mod
    try:
        slate_dates = pd.to_datetime(games.gameday, errors="coerce").dropna()
        target = slate_dates.max() if len(slate_dates) else None
        if target is None:
            logger.warning("player TS skipped: the schedule has no usable date")
            return None
        seasons = sorted({ingestion.season_label(d.date())
                          for d in slate_dates})
        positions = pd.concat(
            [ingestion._fetch_positions(season) for season in seasons],
            ignore_index=True) if seasons else pd.DataFrame()
        if not len(positions):
            logger.warning("player TS skipped: no positions resolved for %s; "
                           "every rating would fall back to an unsegmented "
                           "prior", ", ".join(seasons) or "the window")
            return None
        teams = sorted(set(games.home_team.astype(str)) |
                       set(games.away_team.astype(str)))
        games_frame = ts_mod.prepare_player_games(facts.player_stats, positions)
        if not len(games_frame):
            logger.warning("player TS skipped: the player log has no "
                           "points/fga/fta rows to rate")
            return None
        # Availability is ANNOTATED, not applied. The rating is computed for
        # every player regardless of injury, and the removal happens at pool
        # construction, where a replacement inherits the vacated slot.
        # Multiplying the rating here instead would leave the player in the
        # pool dragging the mean toward zero.
        if stints is not None:
            import injury_stints as stints_mod
            ratings = ts_mod.build_player_ts(
                games_frame, target_dates=pd.Series([target]))
            ratings = stints_mod.annotate_availability(ratings, stints)
        else:
            ratings = ts_mod.build_player_ts(
                games_frame, target_dates=pd.Series([target]))
            ratings["is_available"] = True
        if not len(ratings):
            logger.warning("player TS skipped: no player had strictly-prior "
                           "evidence as of %s", target.date())
            return None
        # The rating row's own date doubles as the pool's gameday, so the
        # projected-lineup join and the artifact agree on one column. The
        # rename happens AFTER the artifact is written, because the CSV keeps
        # ``target_date`` as its label.
        path = out / config.PLAYER_TS_CSV.format(date=date_c)
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
            agg_path = out / config.PLAYER_TS_AGG_CSV.format(date=date_c)
            aggregates.assign(gameday=pd.to_datetime(aggregates.gameday)
                              .dt.strftime("%Y-%m-%d")).to_csv(agg_path,
                                                              index=False)
            logger.info("projected lineups: %d team-game(s), mean pool %.1f, "
                        "mean healthy %.1f", len(aggregates),
                        aggregates.pool_size.mean(), aggregates.healthy_size.mean())
            # The seven diff features are attached to a COPY of the slate for
            # reporting only. They are NOT added to MONEYLINE_FEATURE_COLS:
            # that changes the model and needs its own holdout gate. The
            # coverage report is what makes that decision safe to take later -
            # a feature that is constant or unpopulated is visible NOW rather
            # than after it has been trained on for a month.
            import lineup_projection as slate_proj
            coverage = slate_proj.feature_coverage(
                slate_proj.attach_to_slate(games, aggregates))
            for _, row in coverage.iterrows():
                logger.info("  %-32s %d/%d rows  distinct %d%s",
                            row["feature"], int(row["populated"]),
                            int(row["rows"]), int(row["distinct_values"]),
                            "  CONSTANT" if row["constant"] else "")
            # Stated plainly because it is the thing standing between these
            # features and promotion: the pool is built for ONE target date, so
            # only that game's row carries them. MLB builds lineup_agg across
            # the whole decided frame; until this does the same, a column
            # populated on 1 row in 2,779 is a column the model cannot use.
            populated = int(coverage.populated.max()) if len(coverage) else 0
            rows = int(coverage.rows.max()) if len(coverage) else 0
            if rows and populated < rows:
                logger.warning(
                    "lineup_ts features cover %d of %d game rows - the pool is "
                    "built for a single target date. Promoting these to "
                    "MONEYLINE_FEATURE_COLS requires building the ratings "
                    "across the whole decided frame first, the way MLB's "
                    "lineup_agg is.", populated, rows)
        return path.name
    except Exception as exc:  # noqa: BLE001
        logger.warning("player TS ratings not written (%s); the run continues "
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
    reports its full width: the run in which all nine ``pl_ts`` columns went
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


def _build_position_ts_features(facts, games: pd.DataFrame,
                                cache_dir: Path | None = None
                                ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """The nine ``pl_ts_*`` features over the WHOLE decided frame, plus slate.

    This is MLB's ``lineup_agg`` construction, translated: ratings are built
    for EVERY decided game date (not one target date - the "1 of 2,779 rows"
    blocker), the position pools are projected per (game, team), and the nine
    position-segmented columns are attached with sides retained -
    ``pl_ts_{c,f,g}_{away,home,diff}`` - the same shape as MLB's
    ``lineup_woba_mean_{home,away}`` family, segmented by position instead of
    averaged over one pool.

    POINT-IN-TIME, in both directions the contract requires:

    * every rating row is summed over games STRICTLY BEFORE its target date
      (the guarantee ``player_ts._prior_for`` implements and the audit
      recomputes), so a player's rating for game G never includes game G;
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
    import player_ts as ts_mod

    seasons = sorted({ts_mod._season_of(d) for d in games.gameday.dropna()})
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
        logger.warning("pl_ts features skipped: no position table; every "
                       "rating would fall back to an unsegmented prior")
        return None, None
    games_frame = ts_mod.prepare_player_games(facts.player_stats, positions)
    if not len(games_frame):
        logger.warning("pl_ts features skipped: the player log has no "
                       "rateable rows")
        return None, None

    # id -> "First Last", the form the injury report files players under.
    name_by_id: dict = {}
    log = facts.player_stats
    for pid, pname in zip(ts_mod._player_id_str(log.player_id),
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
        logger.warning("no nba_designations_*.parquet in %s - the injury "
                       "removal CANNOT bind and the pools are UNFILTERED for "
                       "this run; run backfill_injury_designations.py to "
                       "restore it", root)

    decided_mask = games.home_score.notna() & games.away_score.notna()
    decided = games[decided_mask]
    pending = games[~decided_mask]

    dates = pd.Series(sorted(pd.to_datetime(decided.gameday).dropna().unique())
                      + sorted(pd.to_datetime(pending.gameday).dropna().unique()))
    ratings = ts_mod.build_player_ts(games_frame, target_dates=pd.Series(dates))
    ratings = ratings.rename(columns={"target_date": "gameday"})
    ratings["gameday"] = pd.to_datetime(ratings.gameday)
    if "is_available" not in ratings.columns:
        ratings["is_available"] = True
    ratings = proj_mod.apply_pit_designations(ratings, designations, name_by_id)

    aggregates = proj_mod.projected_lineup(ratings, games=games)
    if not len(aggregates):
        logger.warning("pl_ts features skipped: no team-game aggregates")
        return None, None

    def _attach(frame: pd.DataFrame) -> pd.DataFrame | None:
        if frame is None or not len(frame):
            return None
        return proj_mod.attach_position_ts(frame, aggregates)

    return _attach(decided), _attach(pending)


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


def feature_importance_weights(final_models: dict[str, dict],
                               weights: dict[str, float]) -> dict[str, float] | None:
    """Blend-weighted feature importance across the ensemble (sums to 100).

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

    Returns None when no member exposes importances - the caller then omits
    the column rather than publishing a fabricated zero.
    """
    cols = config.active_moneyline_feature_cols()
    nfc = len(cols)
    agg = np.zeros(nfc)
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
        agg += share * (imp / imp.sum())
        contributed = True
    if not contributed or agg.sum() <= 0:
        return None
    return {c: round(float(w), 4) for c, w in zip(cols, agg / agg.sum() * 100.0)}


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
            logger.info("excluding %d postponed game(s) from the slate: %s",
                        int(postponed.sum()),
                        ", ".join(sorted(pending.loc[postponed, "game_id"]
                                         .astype(str).head(5))))
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
        _pl_frame, _pl_slate = _build_position_ts_features(facts, games)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pl_ts feature build failed (%s); the nine columns "
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
            c for c in config.PLAYER_TS_POSITION_FEATURE_COLS
            if c in game_df.columns])
        game_df = game_df.merge(
            _pl_frame[["game_id"] + config.PLAYER_TS_POSITION_FEATURE_COLS],
            on="game_id", how="left")
        logger.info("pl_ts features attached: %d/%d decided rows carry them",
                    int(game_df[config.PLAYER_TS_POSITION_FEATURE_COLS[0]]
                        .notna().sum()), len(game_df))
    # Canonical (date_col, game_id) order: the one order every fold index is
    # valid for. See folds.canonical_sort for why a single-column sort is not
    # enough — fold labels are positional, and the tree members are
    # row-order sensitive under a fixed seed.
    game_df = folds_mod.canonical_sort(game_df, "gameday")
    fold_list = folds_mod.make_folds(game_df)
    fold_info = folds_mod.fold_summary(fold_list)
    if not fold_list:
        raise RuntimeError("NBA walk-forward produced no eligible folds after 30-day warm-up")
    _step("features", f"{len(game_df)} games, "
                      f"{len(config.active_moneyline_feature_cols())} features, "
                      f"{len(fold_list)} folds")

    # The slate frame is DATA, not model output - it needs only the schedule
    # and the stat ladders, so it is built before training and survives a
    # training failure. MLB's phase-4 shape: the model block is contained, a
    # failure inside it is recorded and the run still ships every artifact
    # the surviving phases can produce (power rankings, ratings, player TS,
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
            c for c in config.PLAYER_TS_POSITION_FEATURE_COLS
            if c in slate.columns])
        slate = slate.merge(
            _pl_slate[["game_id"] + config.PLAYER_TS_POSITION_FEATURE_COLS],
            on="game_id", how="left")

    phase_error: Exception | None = None
    ml: dict | None = None
    ml_oof = platt = dist_oof = dispersion = None
    oof_markets = market_calibration = None
    final_models: dict = {}
    final_reg = None
    slate_markets = pd.DataFrame()
    leaders = pd.DataFrame()
    try:
        ml = ml_mod.walk_forward_oof(game_df, fold_list=fold_list)
        ml_oof = _merge_oof_metadata(ml["oof"], game_df)
        if not len(ml_oof):
            raise RuntimeError("NBA moneyline walk-forward produced no OOF rows")
        platt = ml_mod.moneyline_fit(ml_oof.p_ensemble.to_numpy(float),
                                     ml_oof.home_win.to_numpy(float))
        if "p_ensemble_calibrated" not in ml_oof:
            ml_oof["p_ensemble_calibrated"] = ml_mod.moneyline_apply(
                ml_oof.p_ensemble.to_numpy(float), platt)
        ml_oof["p_ensemble_calibrated"] = ml_oof.p_ensemble_calibrated.fillna(
            ml_oof.p_ensemble)

        dist = dist_mod.walk_forward_oof(game_df, fold_list=fold_list)
        dist_oof = _merge_oof_metadata(dist["oof"], game_df)
        dispersion = dist_mod.calibrate_dispersion(dist_oof)
        oof_markets = _marketize(ml_oof, dist_oof, "oof", dispersion)
        oof_markets, market_calibration = dist_mod.calibrate_market_frame(oof_markets)
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

        raw_metrics = evaluation.binary_metrics(ml_oof.p_ensemble.to_numpy(float),
                                                ml_oof.home_win.to_numpy(float))
        cal_metrics = evaluation.binary_metrics(ml_oof.p_ensemble_calibrated.to_numpy(float),
                                                ml_oof.home_win.to_numpy(float))
        buckets = evaluation.calibration_buckets(ml_oof.p_ensemble.to_numpy(float),
                                                  ml_oof.home_win.to_numpy(float))
        p_cal = out / config.CALIBRATION_JSON.format(date=date_c)
        serving.write_calibration_json(p_cal, raw_metrics, cal_metrics, buckets, [],
                                       _config_meta(facts), platt, run_day, len(ml_oof),
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
                                  oof_markets, slate_markets, _config_meta(facts))
        artifacts += [p_markets.name,
                      (out / config.MARKETS_META_JSON.format(date=date_c)).name]

    ratings, records, point_diff = _power_state(settled)
    p_rank = out / config.POWER_RANKINGS_CSV.format(date=date_c)
    serving.write_power_rankings_csv(p_rank, ratings, records, facts.team_names, point_diff)
    artifacts.append(p_rank.name)

    # Player-level True Shooting, shrunk to a position-segmented league prior.
    # Published ALONGSIDE the feature contract and deliberately not fed to the
    # ensemble, for the same reason NHL's player ratings are not: adding a
    # column to MONEYLINE_FEATURE_COLS changes the model and needs its own
    # holdout validation, which is a separate decision from building the
    # rating. The rating is PIT (strictly prior rows, season-partitioned), so
    # it is safe to publish now and to promote later behind a gate.
    ts_name = _write_player_ts(out, date_c, facts, games,
                               stints=_build_injury_stints(facts, games))
    if ts_name:
        artifacts.append(ts_name)
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
    imp_weights = (feature_importance_weights(final_models, ml["member_weights"])
                   if ml else None)
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
    latest_brier = f"{brier[-1]['brier']:.4f}" if brier else "n/a"
    baseline_brier = (float(1 - ml_oof.home_win.mean())
                      if ml_oof is not None and len(ml_oof) else None)
    monitoring.write_monitor_json(
        out / config.MODEL_MONITOR_JSON.format(date=date_c), date_c, drift, cov,
        members, brier,
        baseline_brier, _config_meta(facts), fold_info,
        cal_metrics, platt)
    artifacts.append(config.MODEL_MONITOR_JSON.format(date=date_c))
    monitoring.write_run_engine_monitor(
        out / config.MARKETS_MONITOR_JSON.format(date=date_c), date_c,
        (evaluation.nb_distribution_metrics(dist_oof, dispersion)
         if dist_oof is not None and dispersion is not None else {}),
        oof_markets if oof_markets is not None else {},
        market_calibration, _config_meta(facts))
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
    _step("monitor", f"rolling Brier {latest_brier} over {len(brier)} day(s), "
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
    _prune(out, date_c, set(artifacts))
    sync = _sync_data_delivery(config.ROOT_DIR.parent)
    summary["sync"] = sync
    (out / "nba_pipeline_summary.json").write_text(
        json.dumps(summary, indent=1, default=str))
    _step("publish", f"status {summary['status']}, "
                     f"{summary['elapsed_seconds']}s, sync "
                     f"{sync.get('pushed', sync.get('status', 'n/a'))}")
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
