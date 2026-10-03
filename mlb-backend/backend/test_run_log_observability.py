"""Pin the run-log observability fixes found reviewing the 2026-09-26 run.

The Kaggle log looked healthy end to end, but three of its lines were lying
by omission. Each test here pins the specific guarantee so the defect
cannot come back quietly:

* ``_chunked_statcast`` abandoned chunks that exhausted all three retries
  without emitting a single terminal line -- the 09-26 run dropped three
  chunks (2024-01-01/2024-02-29, 2024-12-26/2025-02-23,
  2025-12-21/2026-02-18) and the log showed only "retrying" lines, which
  is indistinguishable from a recovered chunk. Every abandoned chunk must
  now log something, at a level matched to how many games could have been
  in the window.
* ``build_features`` computed all 35 diff features, and then
  ``master_pipeline`` recomputed all 35 after Elo/record enrichment, so
  the run did the work twice, threw the first pass away, and logged
  "Computing 35 diff features" twice in a row.
* ``build_features_metadata`` warned about authored features that were
  merely RFE candidates rather than selected ones, which is a healthy
  state, not rot. A truly orphaned name must still warn.
"""
from __future__ import annotations

import ast
import datetime as dt
import logging
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import feature_metadata  # noqa: E402
import features  # noqa: E402
import ingestion  # noqa: E402
import training  # noqa: E402


# ── A. abandoned Statcast chunks must never be silent ────────────────────────

class _Capture(logging.Handler):
    """Collect (level, message) for a module logger without touching config."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[tuple[str, str]] = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))

    @property
    def levels(self) -> list[str]:
        return [lvl for lvl, _ in self.records]

    def text(self) -> str:
        return "\n".join(msg for _, msg in self.records)


def _capture_report(chunk_start: dt.date, chunk_end: dt.date) -> _Capture:
    cap = _Capture()
    logger = ingestion.logger
    prev_level = logger.level
    logger.addHandler(cap)
    logger.setLevel(logging.DEBUG)
    try:
        ingestion._report_exhausted_chunk(chunk_start, chunk_end, "empty response")
    finally:
        logger.removeHandler(cap)
        logger.setLevel(prev_level)
    return cap


# The three windows the 2026-09-26 production run dropped without a trace.
@pytest.mark.parametrize(
    "start,end",
    [
        (dt.date(2024, 1, 1), dt.date(2024, 2, 29)),
        (dt.date(2024, 12, 26), dt.date(2025, 2, 23)),
        (dt.date(2025, 12, 21), dt.date(2026, 2, 18)),
    ],
)
def test_offseason_abandoned_chunk_is_no_longer_silent(start, end):
    """The exact regression: these three logged NOTHING on 09-26."""
    cap = _capture_report(start, end)
    assert cap.records, f"{start} → {end} abandoned silently"
    assert cap.levels == ["WARNING"]


@pytest.mark.parametrize(
    "start,end,label",
    [
        (dt.date(2024, 10, 1), dt.date(2024, 10, 30), "postseason October"),
        (dt.date(2026, 3, 1), dt.date(2026, 3, 30), "season-opener March"),
    ],
)
def test_game_bearing_edge_months_warn_but_do_not_abort(start, end, label):
    """October and early March DO carry real games, so a persistent empty
    there is possible data loss. Warn -- but do not halt the daily run over
    a window that is legitimately empty in most seasons."""
    assert not ingestion._is_past_dated_core_season_chunk(start, end), label
    ingestion._abort_on_exhausted_core_season_empty_chunk(start, end, "empty response")

    cap = _capture_report(start, end)
    assert cap.levels == ["WARNING"], label
    assert "abandoned" in cap.text(), label


def test_core_season_still_aborts_before_reporting():
    """The hard guard must keep winning: Apr-Sep persistent empty raises."""
    with pytest.raises(RuntimeError, match="Refusing to proceed"):
        ingestion._abort_on_exhausted_core_season_empty_chunk(
            dt.date(2024, 7, 1), dt.date(2024, 7, 30), "empty response"
        )


def test_future_dated_chunk_is_quiet():
    """A wholly future-dated window cannot hold completed games."""
    today = dt.date.today()
    cap = _capture_report(
        today + dt.timedelta(days=5), today + dt.timedelta(days=40)
    )
    assert cap.levels == ["DEBUG"]


def test_report_names_the_chunk_and_the_attempt_count():
    """A reviewer must be able to tell WHICH window was lost and after how
    many tries -- that is the whole point of the terminal line."""
    cap = _capture_report(dt.date(2024, 1, 1), dt.date(2024, 2, 29))
    text = cap.text()
    assert "2024-01-01" in text and "2024-02-29" in text
    assert str(ingestion.CHUNK_RETRIES) in text
    assert "abandoned" in text


def test_every_exhausted_chunk_path_calls_the_reporter():
    """_chunked_statcast must route its give-up through the reporter, so a
    future refactor cannot reintroduce the silent branch."""
    src = (BACKEND / "ingestion.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_chunked_statcast"
    )
    called = {
        n.func.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_report_exhausted_chunk" in called
    assert "_warn_if_core_season_chunk_empty" not in called


# ── B. diff features are derived exactly once per run ─────────────────────────

def test_build_features_no_longer_computes_diffs():
    """master_pipeline re-derives diffs after Elo/record enrichment, so
    computing them here too built 35 columns twice and discarded one pass.

    Matched on the AST, not on text: the function's comment legitimately
    names ``add_diff_features()`` when telling callers to invoke it.
    """
    src = (BACKEND / "features.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "build_features"
    )
    calls = {
        n.func.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "add_diff_features" not in calls, (
        "build_features must not derive diff features; each caller does it "
        "once, after its own raw inputs are final"
    )


def test_each_caller_derives_diffs_itself():
    """Both production callers must still produce a diff-complete frame
    (both now live in master_pipeline.py)."""
    mp = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    assert re.search(r"^game_df = add_diff_features\(game_df\)$", mp, re.M), (
        "master_pipeline must derive diffs after enrich_elo_and_records"
    )

    # The Phase 2-3 caller consumes build_features output directly (the
    # former pipeline.py --statcast branch merged into the same module) and
    # must derive diffs itself, after its own raw inputs are final.
    build_call = "game_df, pbp_df = build_features("
    assert build_call in mp
    after = mp.split(build_call, 1)[1]
    assert re.search(r"^game_df = add_diff_features\(game_df\)$", after, re.M), (
        "diffs must be re-derived after the Phase 2-3 build_features call"
    )


def test_master_pipeline_still_imports_add_diff_features():
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    assert re.search(r"from features import \([^)]*add_diff_features", src, re.S)


# ── C. the staleness warning targets orphans, not RFE candidates ─────────────

def test_unselected_rfe_candidates_do_not_warn():
    """The 09-26 run emitted this warning for lineup_il_flag_{home,away,diff}
    purely because they are candidates rather than selected. That is a
    healthy state and the warning is a false positive."""
    _, warnings = feature_metadata.build_features_metadata()
    assert warnings == [], f"unexpected metadata warnings: {warnings}"


@pytest.mark.parametrize(
    "name", ["lineup_il_flag_home", "lineup_il_flag_away", "lineup_il_flag_diff"]
)
def test_il_flag_candidates_are_known_pool_not_universe(name):
    """They must be pool members (so no warning) without being served."""
    assert name in training.KNOWN_FEATURE_COLS, name
    assert name not in training.MONEYLINE_FEATURE_COLS, name


def test_genuinely_orphaned_entry_still_warns(monkeypatch):
    """Dropping the pool exemption must NOT silence real rot: a name the
    pool no longer knows at all (renamed or deleted) must still warn."""
    monkeypatch.setitem(
        feature_metadata._RICH,
        "totally_orphan_feature",
        {
            "summary": "x", "definition": "x", "formula": "x", "source": "x",
            "window": "x", "units": "x", "direction": "x",
        },
    )
    _, warnings = feature_metadata.build_features_metadata()
    hits = [w for w in warnings if "totally_orphan_feature" in w]
    assert len(hits) == 1, f"orphaned entry must warn exactly once, got {warnings}"


def test_metadata_still_covers_the_full_serving_width():
    """The fix narrows the WARNING, not the metadata itself."""
    meta, _ = feature_metadata.build_features_metadata()
    assert len(meta) == len(training.MONEYLINE_FEATURE_COLS) == 109
    assert all(row.get("tooltip") for row in meta.values())


# ── D. the diff pass must report its MEASURED column count ───────────────────

def _diff_output() -> tuple[set, set, pd.DataFrame]:
    """Run add_diff_features on an identity-only frame and return
    (input_cols, created_cols, out_frame). Every diff column is emitted even
    when its raw inputs are absent (they land as NaN), so this is the full
    produced set without needing a real decided frame."""
    frame = pd.DataFrame({
        "game_pk": [1, 2],
        "game_date": ["2026-01-01", "2026-01-02"],
        "home_team": ["A", "A"],
        "away_team": ["B", "B"],
    })
    before = set(frame.columns)
    out = features.add_diff_features(frame)
    return before, set(out.columns) - before, out


def test_diff_pass_creates_fifty_eight_columns():
    """67 on the identity frame: the 36 the pass always created plus the
    2026-09-27 twin expansion minus the 2 travel twins, PLUS the 2
    bullpen_meltdown_risk per-side twins added 2026-09-30 (renames are
    in-place; the 12 level twins are absent inputs here, so they are
    created NULL like their diffs; the 6 interaction twins and the 2
    travel twins are new columns; the 16 exp2 twins belong to
    add_exp2_features), PLUS the 9 pl_<pos>_xwoba position-pool diffs
    added 2026-10-02 (absent inputs here → created NULL like the rest).
    58 -> 67 with that family."""
    _, created, _ = _diff_output()
    assert len(created) == 67, sorted(created)


def test_created_set_carries_the_il_flag_diff():
    """The column that caused the drift must be in the produced set."""
    _, created, _ = _diff_output()
    assert "lineup_il_flag_diff" in created
    # the eight non-`*_diff` interaction features are easy to forget —
    # renamed 2026-09-27 to their *_diff names, twins included
    extras = {
        "lineup_handedness_matchup_advantage", "dome_is_neutral",
        "wind_advantage_flyball_factor", "air_density_velocity_boost",
        "bullpen_meltdown_risk_diff", "pitcher_regression_indicator_diff",
        "lineup_depth_multiplier_diff", "ace_efficiency_factor_diff",
        "pitcher_regression_indicator_home", "pitcher_regression_indicator_away",
        "lineup_depth_multiplier_home", "lineup_depth_multiplier_away",
        "ace_efficiency_factor_home", "ace_efficiency_factor_away",
    }
    assert extras <= created, sorted(extras - created)


def test_reported_count_matches_the_actual_count(caplog):
    """The regression itself: the log hardcoded 35 while creating 36, so
    every run under-reported its own work and nobody noticed for a week.
    Assert the emitted number EQUALS the real delta, so re-hardcoding fails.
    """
    import logging

    with caplog.at_level(logging.INFO, logger="features"):
        before, created, _ = _diff_output()
    assert before is not None
    msgs = [r.getMessage() for r in caplog.records
            if "columns added" in r.getMessage()]
    assert len(msgs) == 1, msgs
    reported = int(msgs[0].rsplit(":", 1)[1].strip().split()[0])
    assert reported == len(created), (
        f"log claims {reported} but {len(created)} columns were created"
    )


def test_no_hardcoded_column_count_remains():
    """Neither diff log line may carry a literal count again."""
    src = (BACKEND / "features.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "add_diff_features"
    )
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "info"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        text = str(node.args[0].value)
        if "columns added" in text:
            # every %d must be fed a Name, never a numeric literal
            for arg in node.args[1:]:
                assert not isinstance(arg, ast.Constant), (
                    f"hardcoded count in: {text!r}"
                )
