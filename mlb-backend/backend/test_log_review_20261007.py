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
from pathlib import Path

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
