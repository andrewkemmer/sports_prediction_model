"""2026-10-07 run-log review: remediations for the defects found on remote.

The committed ``mlb-backend/data_delivery/mlb_pipeline_run_log.txt``
(411 lines, run pushed at a5586ad0) was reviewed line by line against its
emitters AND against the remote tree it claimed to deliver. The MODEL
numbers reconciled clean — the log's ``Walk-forward metrics`` (auc 0.5722
/ brier 0.2445 / logloss 0.682) equal model_monitor_20261006's published
blend lineage, the calibration counters (256 of 6064 degenerate Platt
fits, 3-line WARN cap) and the published-blend / PROVISIONAL / stale-pin
lines from the 2026-10-05/06 reviews are all present — the DELIVERY did
not:

**Evidence (log vs. remote tree):**
  * Step 5 printed ``📁 Artifacts: 15 files`` (todays_games_20261007.csv,
    model_monitor_20261007.json, ...) and Phase 5 printed ``Staging 3
    files`` + ``Pushed and remotely verified 3 files`` — yet
    ``git ls-tree origin/main`` carries ZERO ``*20261007*`` files: the
    remote still serves the 10-06 board. The 3-file staging list is
    exactly {tee log, game_level_features.csv, pl_slate_20261006.parquet}
    — the paths reachable from ``Path.cwd()/data_delivery`` plus the one
    special-cased csv candidate.
  * The tee, ``player_positions.parquet`` and features' lineup root are
    under ``/content/sports_prediction_model`` while every Step-5
    artifact, pbp_defense and the ensemble are under
    ``/kaggle/working/sports_prediction_model``: the run executed in two
    trees because ``config`` (imported at master_pipeline:48, BEFORE the
    clone's backend/ lands at ``sys.path[0]``) resolved against the
    kernel's checkout. Phase 5 scanned only the clone tree.
  * The slate half of the same split: ``slate pl_* pools: 4 team-games
    ... → pl_slate_20261006.parquet`` on a run whose slate is
    ``Upcoming slate built: 4 games for 2026-10-07``, and
    ``slate pl_* sources (home/away): {'carry/carry': 4}`` — features
    exported beside the clone's STALE lineups copy (game_pks 849819 /
    849826 dated 10-06; remote's lineups.parquet confirms) while
    ``_load_slate_pl`` read the kernel root. The resolved pools the
    export computed were never served; all four sides priced carry.
  * The season-split line read like a partition — ``7024 OOF rows (6879
    grading / 107 postseason / 137 provisional); blocks regular n=6879``
    — but the counts overlap (99 postseason rows sit inside provisional
    folds) and sum to 7,123 on a 7,024-row frame, and the block labeled
    ``regular`` is built from the GRADING mask.

**Remediations:**
  T1 — ``config.ensure_config_root`` repoints a foreign ``config`` at the
       running clone (by-path load) and purges modules imported from the
       foreign backend dir; master_pipeline calls it in Phase 0 and the
       tee-adjacent observability line announces the delivery root (or
       the SPLIT) into the pushed log.
  T2 — Phase 5 snapshots and scans BOTH delivery roots (per-root mtime
       gates), so artifacts written to either tree are staged.
  T3 — ``github_sync.missing_reported_artifacts``: every artifact Step 5
       REPORTED must be in the staging list or Phase 5 raises — a green
       log can no longer claim delivery that GitHub does not have.
  T4 — ``_load_slate_pl`` searches the config root, cwd root and the
       features tree; serving from an alternate root warns, and a
       NEARBY-DATED sibling with the exact file absent warns about the
       mis-dated export instead of silently serving carry.
  T5 — the walk-forward season-split line states the partition
       (grading + non-grading = OOF) and the overlaps explicitly.

Convention: source/AST pins for the run-once script's inline changes,
behavioral tests for everything importable (config, github_sync,
data_ingestion) — same style as test_log_review_{20260929,20261005,20261006}.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import config  # noqa: E402
import data_ingestion  # noqa: E402
from github_sync import missing_reported_artifacts  # noqa: E402

MASTER_SRC = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
TRAINING_SRC = (BACKEND / "training.py").read_text(encoding="utf-8")


# ── T1: config-root reconciliation ──────────────────────────────────────────

def _load_module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_foreign_config_is_replaced_by_the_clone_copy(tmp_path, capsys):
    """A config imported from a foreign backend dir is swapped for the
    real one BY PATH, and sibling modules from that dir are purged (they
    hold stale ``from config import DATA_DELIVERY_DIR`` bindings)."""
    foreign_backend = tmp_path / "backend"
    foreign_backend.mkdir()
    (foreign_backend / "config.py").write_text(
        "from pathlib import Path\n"
        f"ROOT_DIR = Path(r'{tmp_path}')\n"
        "DATA_DELIVERY_DIR = ROOT_DIR / 'data_delivery'\n",
        encoding="utf-8")
    (foreign_backend / "foreign_helper.py").write_text(
        "from config import DATA_DELIVERY_DIR\n", encoding="utf-8")

    original = sys.modules.get("config")
    foreign_cfg = _load_module_from(foreign_backend / "config.py", "config")
    foreign_helper = _load_module_from(
        foreign_backend / "foreign_helper.py", "foreign_helper")
    assert foreign_cfg.DATA_DELIVERY_DIR != config.DATA_DELIVERY_DIR
    try:
        changed = config.ensure_config_root(BACKEND, log=lambda *_: None)
    finally:
        restored = sys.modules.get("config")
        if original is not None:
            sys.modules["config"] = original
        else:
            sys.modules.pop("config", None)
        sys.modules.pop("foreign_helper", None)

    assert changed is True
    assert restored is not None
    assert Path(restored.__file__).resolve() == (BACKEND / "config.py").resolve()
    assert restored.DATA_DELIVERY_DIR == config.DATA_DELIVERY_DIR
    # the foreign sibling module held a binding to the OLD config — purged
    assert sys.modules.get("foreign_helper") is None
    assert foreign_helper is not None  # it really was loaded beforehand


def test_anchored_config_is_left_alone():
    """The normal case (config already IS the running backend's copy) must
    be a no-op — same module object back, no reload churn."""
    before = sys.modules.get("config")
    changed = config.ensure_config_root(BACKEND, log=lambda *_: None)
    assert changed is False
    assert sys.modules.get("config") is before


def test_missing_clone_config_never_raises(tmp_path):
    """Best-effort by design: an absent clone config keeps the current
    roots instead of killing the run before Phase 1."""
    changed = config.ensure_config_root(
        tmp_path / "nope", log=lambda *_: None)
    assert changed is False


def test_reconciliation_never_unloads_main():
    """When the foreign backend dir IS the executing master_pipeline
    (``__main__``), the purge must skip it — unloading the running script
    mid-reconciliation would be catastrophic."""
    src = (BACKEND / "config.py").read_text(encoding="utf-8")
    assert 'name in ("config", "__main__")' in src


def test_master_pipeline_reconciles_before_snapshot_and_scans_both_roots():
    """Source pins for the run-once script (it executes on import, so its
    inline wiring is pinned as text, ordered like the run executes)."""
    assert "ensure_config_root" in MASTER_SRC
    # reconciliation runs in Phase 0, BEFORE the pre-run snapshot
    assert MASTER_SRC.index("ensure_config_root") < MASTER_SRC.index(
        "_preexisting_delivery_roots: dict[Path, dict[str, int]] = {}")
    # both roots are snapshotted pre-run ...
    assert "_snapshot_delivery_root(_preexisting_dir)" in MASTER_SRC
    assert "_snapshot_delivery_root(Path(_CFG_DELIVERY_DIR))" in MASTER_SRC
    # ... and both are scanned at Phase 5 with per-root mtime gates
    assert '_scan_roots: list[Path] = [data_delivery_local]' in MASTER_SRC
    assert "_DD_SCAN" in MASTER_SRC
    assert "_preexisting_delivery_roots.get(_root_key, {})" in MASTER_SRC
    # the split is announced into the PUSHED log next to the tee
    assert "delivery root SPLIT" in MASTER_SRC


# ── T3: reported-artifact coverage gate ─────────────────────────────────────

_REPORTED_15 = [
    "/kaggle/working/sports_prediction_model/mlb-backend/data_delivery/"
    f"todays_games_2026100{i}.csv" for i in (7,)
] + [
    "/kaggle/working/sports_prediction_model/mlb-backend/data_delivery/"
    + n for n in (
        "power_rankings_20261007.csv", "calibration_20261007.json",
        "predictions_history_20261007.csv", "rolling_brier_20261007.json",
        "features_metadata_20261007.json", "run_engine_oof_20261007.csv",
        "run_engine_markets_20261007.csv",
        "run_engine_markets_20261007.meta.json",
        "run_engine_monitor_20261007.json", "feature_drift_20261007.csv",
        "feature_coverage_20261007.csv",
        "run_engine_feature_drift_20261007.csv",
        "run_engine_feature_coverage_20261007.csv",
        "model_monitor_20261007.json")
]


def test_delivered_run_would_have_been_refused():
    """Replay of 2026-10-07: 15 reported artifacts, the 3 files Phase 5
    actually staged — every reported artifact must come back missing."""
    staged = [
        "mlb-backend/data_delivery/game_level_features.csv",
        "mlb-backend/data_delivery/mlb_pipeline_run_log.txt",
        "mlb-backend/data_delivery/pl_slate_20261006.parquet",
    ]
    missing = missing_reported_artifacts(_REPORTED_15, staged, "mlb-backend")
    assert missing == _REPORTED_15
    assert len(missing) == 15


def test_complete_staging_reports_nothing_missing():
    staged = [f"mlb-backend/data_delivery/{Path(p).name}" for p in _REPORTED_15]
    assert missing_reported_artifacts(
        _REPORTED_15, staged, "mlb-backend") == []


def test_subdir_and_odd_separators_still_match():
    staged = [
        "mlb-backend/data_delivery/models/ensemble_latest.joblib",
        "mlb-backend/data_delivery\\todays_games_20261007.csv",
    ]
    reported = [
        "/kaggle/working/x/mlb-backend/data_delivery/models/"
        "ensemble_latest.joblib",
        "C:\\run\\data_delivery\\todays_games_20261007.csv",
    ]
    assert missing_reported_artifacts(reported, staged, "mlb-backend") == []


def test_empty_or_unreported_never_blocks():
    assert missing_reported_artifacts([], ["mlb-backend/data_delivery/x"],
                                      "mlb-backend") == []
    assert missing_reported_artifacts(None, None, "mlb-backend") == []


def test_phase5_gate_runs_before_the_commit():
    """The raise must precede ``repo.index.commit`` — a refused delivery
    aborts before anything lands on the remote."""
    gate = MASTER_SRC.index("missing_reported_artifacts(")
    commit = MASTER_SRC.index("repo.index.commit(f\"Update MLB features")
    assert gate < commit
    assert "reported artifacts not staged" in MASTER_SRC


# ── T4: slate pl_* artifact lookup ──────────────────────────────────────────

def _write_slate_parquet(directory: Path, date_str: str,
                         rows=2) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "game_date": [pd.Timestamp(date_str)] * rows,
        "game_pk": list(range(849800, 849800 + rows)),
        "team": [f"T{i}" for i in range(rows)],
        "pl_tier": [2] * rows,
    })
    path = directory / f"pl_slate_{pd.Timestamp(date_str):%Y%m%d}.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def split_roots(tmp_path, monkeypatch):
    """Config root (empty), cwd root (empty), features root — the 2026-10-07
    split layout, all three pointing at tmp dirs."""
    cfg_root = tmp_path / "cfg_dd"
    cwd_root = tmp_path / "cwd_dd"
    feat_root = tmp_path / "feat_dd"
    for r in (cfg_root, cwd_root, feat_root):
        r.mkdir()
    monkeypatch.setattr(data_ingestion, "DATA_DELIVERY_DIR", cfg_root)
    import features
    monkeypatch.setattr(features, "_lineup_base_dir", lambda: feat_root)
    monkeypatch.chdir(tmp_path)  # cwd/data_delivery does not exist here
    return cfg_root, cwd_root, feat_root


def test_exact_artifact_at_primary_root_serves_silently(split_roots, caplog):
    cfg_root, _cwd, _feat = split_roots
    _write_slate_parquet(cfg_root, "2026-10-07")
    with caplog.at_level(logging.WARNING, logger="data_ingestion"):
        out = data_ingestion._load_slate_pl(
            pd.Timestamp("2026-10-07").date())
    assert len(out) == 2
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_artifact_in_features_root_serves_with_a_split_warning(
        split_roots, caplog):
    """The export writes beside the lineups cache; the loader must find it
    there — and SAY the roots are split instead of serving carry."""
    _cfg, _cwd, feat_root = split_roots
    _write_slate_parquet(feat_root, "2026-10-07")
    with caplog.at_level(logging.WARNING, logger="data_ingestion"):
        out = data_ingestion._load_slate_pl(
            pd.Timestamp("2026-10-07").date())
    assert len(out) == 2  # served, not carried
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    assert any("alternate delivery root" in w for w in warnings)


def test_mis_dated_sibling_names_the_defect_and_serves_carry(
        split_roots, caplog):
    """Replay of the delivered log: target 10-07, only pl_slate_20261006
    exists — the miss is diagnosed (nearest sibling + day offset) instead
    of a silent ``{}`` that prices carry for the whole slate."""
    cfg_root, _cwd, _feat = split_roots
    _write_slate_parquet(cfg_root, "2026-10-06")
    with caplog.at_level(logging.WARNING, logger="data_ingestion"):
        out = data_ingestion._load_slate_pl(
            pd.Timestamp("2026-10-07").date())
    assert out == {}
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    assert any("pl_slate_20261006.parquet" in w and "1 day(s) off" in w
               for w in warnings)


def test_no_slate_artifact_anywhere_stays_silent(split_roots, caplog):
    """Offseason / empty export: no near-dated sibling anywhere → no noise."""
    split_roots
    with caplog.at_level(logging.WARNING, logger="data_ingestion"):
        out = data_ingestion._load_slate_pl(
            pd.Timestamp("2026-01-15").date())
    assert out == {}
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_explicit_base_is_searched_alone(tmp_path, caplog):
    """A caller-passed base (tests, single-root callers) is intent — never
    widened to the other roots."""
    target = tmp_path / "only"
    _write_slate_parquet(target, "2026-10-07")
    out = data_ingestion._load_slate_pl(
        pd.Timestamp("2026-10-07").date(), base=target)
    assert len(out) == 2
    missing = data_ingestion._load_slate_pl(
        pd.Timestamp("2026-10-08").date(), base=target)
    assert missing == {}
    # the base is the ONLY root searched — and its own near-dated sibling
    # still gets the mis-dated diagnosis
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    assert any("not found in 1 root(s)" in w for w in warnings)
    assert not any("alternate delivery root" in w for w in warnings)


# ── T5: walk-forward season-split line ──────────────────────────────────────

def test_season_split_line_partitions_and_discloses_overlap():
    """The emitter must state grading + non-grading = OOF with the
    postseason/provisional overlap — the old line's three counts summed to
    7,123 on a 7,024-row frame and labeled the grading block 'regular'."""
    assert "blocks OVERLAP" in TRAINING_SRC
    assert "= %d grading + %d " in TRAINING_SRC
    assert "non-grading (%d provisional incl. %d postseason, %d " in TRAINING_SRC
    assert "outside provisional folds); blocks OVERLAP" in TRAINING_SRC
    # the old format STRING (quoted, so the review comment above the
    # emitter may still quote the historical line in prose)
    assert '"blocks regular n=%d"' not in TRAINING_SRC


def test_season_split_overlap_arithmetic_reconciles_the_delivered_log():
    """Replay the delivered counts through the disclosed identities:
    6879 grading + 145 non-grading = 7024, and 137 provisional (incl. 99
    postseason) + 8 postseason outside provisional = 145."""
    total, grading, provisional, postseason = 7024, 6879, 137, 107
    post_in_prov = 99
    post_outside = postseason - post_in_prov
    non_grading = total - grading
    assert grading + non_grading == total
    assert provisional + post_outside == non_grading
    assert provisional >= post_in_prov >= 0


# ── T6: calibration artifact population labels (dashboard/pooled-run parity) ─
# 2026-10-07 dashboard review (a continuation of this review day): every
# headline number the dashboards show reconciled with the production pooled
# OOF run — calibration metrics == model_monitor metrics == model_history's
# newest entry == an independent recomputation over predictions_history's
# 7,026 rows (auc 0.5714 / logloss 0.6823 vs the graded headline 0.5712 /
# 0.6823 on 6,879 rows). EXCEPT the population label: calibration_*.json
# carried n_games = the DAY'S SLATE size (4) while the NFL/NHL/NBA writers
# carry the grading pool there, so the shared Calibration page labeled
# pooled KPIs "n = 4 games" beside an AUC graded on 6,879 pooled OOF games.
# The writer now publishes the GRADING population in n_games (== n_eval ==
# the calibration_buckets count sum) and keeps the slate size in
# league_total — the todays pages' "X of Y games shown" denominator.

def test_calibration_json_labels_the_grading_population_not_the_slate():
    """Source pin for the run-once script's artifact writer (import executes
    the pipeline, so the contract is pinned as text): n_games must be the
    graded scoring population — the y_true AFTER the grades_pooled mask,
    the same population the metrics and buckets cover — while league_total
    keeps the caller's slate count."""
    seg_start = MASTER_SRC.index("def _calibration_json(")
    seg = MASTER_SRC[seg_start:MASTER_SRC.index(
        "def _predictions_history_csv(", seg_start)]
    assert '"n_games": int(len(y_true))' in seg, (
        "calibration n_games must be the grading population (len(y_true) "
        "after the grades_pooled mask), not the day's slate size")
    assert '"n_eval": int(len(y_true))' in seg, (
        "n_eval must stay the grading population — the dashboard resolves "
        "the pooled label from it")
    assert '"league_total": n_games' in seg, (
        "league_total must keep the caller's slate count (the todays "
        "pages' 'X of Y games shown' denominator)")
    # y_true IS the graded population: the mask must gate it upstream.
    assert 'ok &= oof["grades_pooled"].astype(bool)' in seg


# ── T7: first-pitch provenance for suspended-and-resumed games ──────────────
# 2026-10-07 full deep dive (ingestion validated game-by-game against MLB's
# official schedule + live feeds: 7,397/7,397 played R/F/D/L/W games present,
# game_date == officialDate, scores == official finals, cancelled games
# correctly absent). ONE ingestion defect surfaced: the schedule's
# ``gameDate`` is the RESUME slot for a suspended game, and
# results.refresh_start_times sealed it as the game's "observed first pitch"
# — 8 of 7,397 history rows carried slots 1-61 days after their game,
# mis-keying the market-line as-of merge (post-start lines could attach),
# weather sampling and within-day PIT ordering. The fetch now prefers
# ``resumedFrom`` (the real first pitch, present exactly for resumed games)
# and the refresh self-heals sealed resume slots (a resume slot is always
# strictly later than its game date; a legitimate late start maps back to
# the same instant and is untouched).

def test_fetch_game_start_times_prefers_the_resumed_from_first_pitch(
        monkeypatch):
    import results as results_mod

    payload = {"dates": [{"date": "2024-08-26", "games": [
        {"gamePk": 746942, "gameDate": "2024-08-26T18:05:00Z",
         "resumedFrom": "2024-06-26T23:10:00Z"},
        {"gamePk": 746943, "gameDate": "2024-08-26T23:10:00Z"},
    ]}]}

    class _Resp:
        status_code = 200
        def raise_for_status(self):
            return None
        def json(self):
            return payload

    monkeypatch.setattr(results_mod.requests, "get",
                        lambda *a, **k: _Resp())
    times = results_mod.fetch_game_start_times(
        date(2024, 8, 26), date(2024, 8, 26))
    assert times[746942] == "2024-06-26T23:10:00Z", (
        "a resumed game must carry its FIRST pitch (resumedFrom), not the "
        "resume slot in gameDate")
    assert times[746943] == "2024-08-26T23:10:00Z", (
        "a normal game must keep its gameDate first pitch")


def test_refresh_start_times_restores_a_sealed_resume_slot(monkeypatch):
    import results as results_mod
    from results import refresh_start_times

    calls = []
    def fake(start, end):
        calls.append((start, end))
        return {746942: "2024-06-26T23:10:00Z",
                900001: "2026-08-01T23:05:00Z"}
    monkeypatch.setattr(results_mod, "fetch_game_start_times", fake)
    frame = pd.DataFrame({
        "game_pk": [746942, 900001],
        "game_date": ["2024-06-26", "2026-08-01"],
        # Row 0 sealed the RESUME slot (observed=True, ET date 2024-08-26,
        # 61 days after its game); row 1 is a legitimate late start whose
        # value already IS its first pitch.
        "start_time_utc": pd.to_datetime(
            ["2024-08-26 18:05", "2026-08-02 03:05"], utc=True),
        "start_time_observed": [True, True],
    })
    out = refresh_start_times(frame)
    assert out.loc[0, "start_time_utc"] == pd.Timestamp(
        "2024-06-26T23:10:00Z"), (
        "a sealed resume slot must be restored to the game's real first "
        "pitch even though it was marked observed")
    assert bool(out.loc[0, "start_time_observed"]), (
        "the repaired row stays observed — the value is authoritative now")
    assert out.loc[1, "start_time_utc"] == pd.Timestamp(
        "2026-08-02 03:05", tz="UTC"), (
        "a legitimate late/midnight start maps to its own instant and is "
        "never rewritten")
    # Idempotent: repaired frame has no later-than-game rows -> no re-fetch.
    again = refresh_start_times(out)
    assert len(calls) == 1, "a coherent fully-observed frame must not " \
                           "re-query the schedule"
    assert again.start_time_utc.equals(out.start_time_utc)


# ── T8: one roof truth for the model feature and the weather composites ────
# 2026-10-07 full deep dive: the model universe consumed the STATIC venue
# map while the game-accurate ``dome_is_neutral_game`` (StatsAPI roof
# cache, 1,731/1,731 retractable-home games resolved) was computed but
# never fed to the model. 834 rows were self-contradictory: 243 MIN home
# games (Target Field — open-air since 2010) plus 591 open-roof retractable
# games carried dome_is_neutral=1 ("fixed dome/closed roof") while the
# wind/air composites on the SAME rows were computed as outdoor. The model
# feature now carries the game-accurate state (metadata contract: "1 if
# home park is a fixed dome/closed roof, 0 if open-air").

def test_min_is_never_a_dome():
    import features
    assert features.DOME_STATUS["MIN"] == 0, (
        "Target Field is open-air — the model must never claim a closed "
        "roof for MIN")
    assert "MIN" in features.OPEN_AIR_MISLABELED, (
        "stale frames carrying the old mislabel must still be corrected "
        "by the game-level refinement")


def test_refine_dome_syncs_the_model_feature_to_the_game_state():
    import features
    frame = pd.DataFrame({
        "game_pk": [11, 12, 13, 14],
        "home_team": ["MIN", "SEA", "TB", "HOU"],
        # Stale pre-fix venue values: every row claims a closed roof.
        "dome_is_neutral": [1.0, 1.0, 1.0, 1.0],
    })
    out = features.refine_dome_game_level(
        frame, roof_states={12: "open", 14: "closed"})
    # MIN open-air correction; SEA observed open; TB fixed dome; HOU closed.
    assert out["dome_is_neutral_game"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert out["dome_is_neutral"].tolist() == [0.0, 0.0, 1.0, 1.0], (
        "the model feature must carry the SAME game-accurate roof state "
        "the weather composites use — one roof truth")


def test_add_diff_features_prefers_the_game_accurate_roof_flag():
    import features
    frame = pd.DataFrame({
        "game_pk": [1, 2, 3],
        "home_team": ["MIN", "SEA", "TB"],
        "away_team": ["CLE", "HOU", "BOS"],
        "dome_is_neutral_game": [0.0, 0.0, np.nan],
    })
    out = features.add_diff_features(frame)
    # MIN/SEA: observed game state wins over the static prior; TB: no game
    # state -> the fixed-dome venue prior stands.
    assert out["dome_is_neutral"].tolist() == [0.0, 0.0, 1.0]
