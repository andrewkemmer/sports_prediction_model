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
from datetime import datetime, timezone
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
    slate = (feat_mod.build_slate_features(games, facts.team_stats,
                                           facts.team_events)
             if len(pending) else pd.DataFrame())
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
    else:
        slate_markets = pd.DataFrame()
        leaders = pd.DataFrame()
    _step("final fit", f"{len(final_models)} ensemble member(s), "
                       f"{len(slate)} upcoming game(s) scored")

    artifacts: list[str] = []
    p_ml = out / config.MONEYLINE_JSON.format(date=date_c)
    serving.write_moneyline_json(
        p_ml, slate,
        slate.get("home_win_prob_model", pd.Series(dtype=float)),
        slate.get("p_ensemble_calibrated", pd.Series(dtype=float)),         facts.team_names, _config_meta(facts), leaders)
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
    coverage = feat_mod.feature_coverage_report(game_df)
    p_feat = out / config.FEATURE_JSON.format(date=date_c)
    serving.write_feature_json(p_feat, coverage, _config_meta(facts), fold_info)
    artifacts.append(p_feat.name)
    _step("serve", f"{len(artifacts)} artifact(s) written to {out}")

    selection = feature_selection.run_rfe(game_df, out, date_c)
    selection_name = f"nba_feature_selection_{date_c}.json"
    workbook_name = feature_workbook.write_feature_workbook(out, date_c, selection)
    if workbook_name:
        artifacts.append(workbook_name)
    artifacts.append(selection_name)
    drift_names = monitoring.write_run_engine_feature_artifacts(
        out, date_c, game_df, game_df.tail(min(60, len(game_df))),
        {feature: 0.0 for feature in config.active_moneyline_feature_cols()})
    artifacts.extend(drift_names)
    _step("feature report", f"selection {selection_name}, "
                            f"{len(drift_names)} drift/coverage file(s)")

    import joblib
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

    members = monitoring.ensemble_table(ml_oof, ml["member_weights"])
    drift = monitoring.feature_drift(game_df, game_df.tail(min(60, len(game_df))))
    cov = monitoring.coverage(game_df)
    brier = monitoring.rolling_brier(ml_oof)
    latest_brier = f"{brier[-1]['brier']:.4f}" if brier else "n/a"
    monitoring.write_monitor_json(
        out / config.MODEL_MONITOR_JSON.format(date=date_c), date_c, drift, cov,
        members, brier,
        float(1 - ml_oof.home_win.mean()), _config_meta(facts), fold_info,
        cal_metrics, platt)
    artifacts.append(config.MODEL_MONITOR_JSON.format(date=date_c))
    monitoring.write_run_engine_monitor(
        out / config.MARKETS_MONITOR_JSON.format(date=date_c), date_c,
        evaluation.nb_distribution_metrics(dist_oof, dispersion), oof_markets,
        market_calibration, _config_meta(facts))
    artifacts.append(config.MARKETS_MONITOR_JSON.format(date=date_c))

    if len(slate):
        cards = slate.copy()
        cards["p_home_win"] = pd.to_numeric(cards.get("p_ensemble_calibrated"), errors="coerce")
        cards["p_away_win"] = 1 - cards.p_home_win
        serving.write_production_cards_history(
            out / "nba_production_cards_history.csv", cards)
        artifacts.append("nba_production_cards_history.csv")

    if len(slate):
        _validate_slate_contract(slate, slate_markets)
    _step("monitor", f"rolling Brier {latest_brier} over {len(brier)} day(s), "
                     f"{len(members)} ensemble member(s)")

    summary = {"status": "ok", "run_date": run_day, "artifacts": artifacts,
               "weights": ml["member_weights"], "folds": fold_info,
               "n_settled": len(settled), "n_slate": len(slate),
               "n_features": len(config.active_moneyline_feature_cols()),
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
    parser = argparse.ArgumentParser(description="NBA production pipeline")
    parser.add_argument("--run-date", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--skip-pull", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.run_date, args.out_dir, args.skip_pull),
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
