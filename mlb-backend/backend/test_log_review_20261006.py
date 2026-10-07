"""2026-10-06 run-log review: remediations for the defects found on remote.

The committed ``mlb-backend/data_delivery/mlb_pipeline_run_log.txt`` was
reviewed line by line (1,105 lines, run pushed at c5086447) and its defects
were confirmed against the emitting source. The NUMBERS reconciled clean —
calibration/model_monitor_20261006 headline (auc 0.5746 / brier 0.2442 /
logloss 0.6814) equals the log's ``Walk-forward metrics`` equals the
published deployed-weight blend (best member 0.5733), so the log's metrics
needed no remediation — the LOG LINES did:

T1 — 665 of 1,105 lines (60%) were superseded tqdm progress frames: the
     tee's file side turned EVERY ``\\r`` into a newline, so each bar
     landed as ~46-60 frame lines. The file side now COLLAPSES CR runs
     (each ``\\r`` supersedes the partial line before it) — a bar lands
     as ONE line, its final frame; the console keeps raw ``\\r``.
T2 — ``Run-engine monitoring view: 109 active moneyline features`` printed
     TWICE per run (drift + coverage re-resolve), byte-identical — the
     exact R7 defect class distributions fixed for ``build_side_frame``
     that explainability's resolver never got. Now once per DISTINCT view.
T3 — the published-blend pass (6e224a42) left NO trace in the log: a
     reviewer could not tell whether the headline metrics graded the
     rolling training-time blend or the deployed bundle's blend. One INFO
     now states the applied deployed weights and the OOF row count.
T4 — three ``degenerate Platt params (a=...)`` WARNINGs were visible but
     the counter is capped at 3 process-wide: the log could not say how
     many fits degenerated, on how many games, or from which block. The
     warning now carries ``n=``; fit_platt keeps UNCAPPED totals
     (_fit_total/_degen_total) and run_engine_daily reports the block's
     own "N of M" before returning.
T5 — a grep for ``auc=`` swept in PROVISIONAL folds' numbers as if they
     graded (the exclusion lived only on the separate line above). The
     metric line itself now carries "[PROVISIONAL — excluded from grading]".
T6 — the DONE banner printed ``Features: 378`` (game + pbp COLUMN sum),
     contradicting ``Feature width: RFE subset (109 cols)`` and
     ``Training data: ... 293 features`` in the same log; Step 7 printed
     ``Skipping GitHub sync`` every run with no hint that Phase 5 owns
     the push. The banner labels the split; the skip line says why.
T7 — Kaggle Version 5 (89bd0879) restored the notebook's ACTIVE
     MLB_FULL_REPULL=1 / MLB_END_DATE="2026-10-06" pins, which reverted
     R6's comment-state pins (two tests red at origin/main before this
     review). Per directive the notebook is Kaggle-owned and must not be
     touched — the PIPELINE now runs against it as-is: the run log opens
     with "♻️ MLB_FULL_REPULL set" (a forced repull that completes, and
     T1 keeps its progress frames to one line each), and target=end means
     a left-behind END_DATE pin would freeze the daily slate on every
     later run. config.resolve_run_end_date extends a stale pin to today
     (logged loudly after the tee install), while same-day/future pins,
     backfill intent and malformed input pass through unchanged. The two
     superseded notebook tests in test_log_review_20261005 were re-pointed
     at documentation-only contracts.

Convention: source/AST pins for emitter changes, behavioral tests where
the machinery is importable (tee, calibration, view resolvers) — the same
style as test_run_log_observability / test_log_review_{20260929,20261005}.
"""
from __future__ import annotations

import ast
import io
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import run_log_tee  # noqa: E402
import calibration  # noqa: E402


# ── T1: CR-run collapse — one line per progress bar ─────────────────────────

def test_a_whole_bar_lands_as_one_file_line(tmp_path):
    """46 frames of one statcast chunk → exactly ONE line in the file
    (the 2026-10-06 log wrote all 46, ×14 bars = 665 frame lines)."""
    log_file = open(tmp_path / "log.txt", "w", encoding="utf-8")
    tee = run_log_tee._Tee(io.StringIO(), log_file)
    for i in range(46):
        tee.write(f"\r{i}/46")
    tee.write("\n")
    tee.write("  INFO     → 164216 pitches\n")
    tee.close()
    lines = (tmp_path / "log.txt").read_text(encoding="utf-8").splitlines()
    assert lines == ["45/46", "  INFO     → 164216 pitches"]


def test_console_passthrough_stays_byte_identical(tmp_path):
    """The collapse is FILE-side only: the console (and therefore tqdm's
    graphical widget) must still see every raw frame."""
    console = io.StringIO()
    tee = run_log_tee._Tee(console, open(tmp_path / "log.txt", "w"))
    frames = "".join(f"\r{i}/5" for i in range(5))
    n = tee.write(frames)
    tee.close()
    assert console.getvalue() == frames
    assert n == len(frames)  # bytes-written passthrough contract


def test_the_old_replace_behavior_is_gone():
    src = (BACKEND / "run_log_tee.py").read_text(encoding="utf-8")
    assert 'replace("\\r", "\\n")' not in src, (
        "one newline per CR is the defect: it is what turned each tqdm "
        "frame into its own line (665 of 1,105 log lines)")


# ── T2: run-engine monitoring view logs once per distinct view ──────────────

def test_monitor_view_line_logs_once_per_distinct_view(caplog):
    import explainability

    explainability._LAST_LOGGED_MON_VIEW = None
    with caplog.at_level(logging.INFO, logger="explainability"):
        explainability._log_resolved_mon_view(["a", "b", "c"])  # drift call
        explainability._log_resolved_mon_view(["a", "b", "c"])  # coverage call
    msgs = [r.getMessage() for r in caplog.records
            if "Run-engine monitoring view" in r.getMessage()]
    assert len(msgs) == 1, (
        "two identical resolutions must produce ONE INFO line — the review "
        "found this printed twice per run as byte-identical twins")

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="explainability"):
        explainability._log_resolved_mon_view(["a", "b", "c"])  # repeat
        explainability._log_resolved_mon_view(["a", "b", "d"])  # changed view
        explainability._log_resolved_mon_view(["a", "b", "d"])  # repeat
    msgs = [r.getMessage() for r in caplog.records
            if "Run-engine monitoring view" in r.getMessage()]
    assert len(msgs) == 1 and "3 active moneyline features" in msgs[0], (
        "a view that CHANGES mid-process must log again — the key is the "
        "resolved content, not a call counter", msgs)


def test_run_engine_feature_cols_routes_through_the_helper():
    src = (BACKEND / "explainability.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "run_engine_feature_cols")
    body = ast.get_source_segment(src, fn) or ""
    assert "_log_resolved_mon_view(feats)" in body, (
        "run_engine_feature_cols must route its resolution log through "
        "_log_resolved_mon_view")
    assert 'logger.info("Run-engine monitoring view' not in body, (
        "logging directly again would resurrect the duplicate line")


# ── T3: the published blend states its weights in the log ───────────────────

def test_published_blend_pass_logs_its_evidence_line():
    src = (BACKEND / "training.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "walk_forward_evaluate")
    fn_src = ast.get_source_segment(src, fn) or ""
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call)
             and any(isinstance(a, ast.Constant)
                     and isinstance(a.value, str)
                     and a.value.startswith("Published blend:")
                     for a in n.args)]
    assert calls, (
        "the published-blend pass must announce itself — the 2026-10-06 "
        "log could not show whether the headline metrics graded the "
        "rolling training-time blend or the deployed bundle's blend")
    fmt = next(a.value for a in calls[0].args
               if isinstance(a, ast.Constant) and isinstance(a.value, str))
    assert "deployed weights" in fmt and "THE serving blend" in fmt
    assert "strictly-prior" in fmt and "NOT OOF" in fmt
    # ordering: announced AFTER the re-pool with the deployed weights and
    # BEFORE the calibration/metrics lines the announcement describes
    # (match the FORMAT STRING — the pass's comment banner also starts
    # with "Published blend:")
    assert (fn_src.index("_member_weights(sorted(oof_members))")
            < fn_src.index("Published blend: %d OOF rows")
            < fn_src.index("Calibration (OOF, prequential)"))


# ── T4: degenerate Platt fits are attributable and countable ────────────────

def test_degenerate_warning_carries_the_fit_size(caplog, monkeypatch):
    """A degenerate fit must say HOW MANY games it saw — and count itself
    on the UNCAPPED total the block summary reports."""
    n = calibration.MIN_OOF_FOR_FIT + 50
    rng = np.random.default_rng(7)
    p = rng.uniform(0.05, 0.95, n)
    y = (p < 0.5).astype(float)  # perfectly anti-correlated → a < 0
    monkeypatch.setattr(calibration.fit_platt, "_degen_logged", 0,
                        raising=False)
    monkeypatch.setattr(calibration.fit_platt, "_degen_total", 0,
                        raising=False)
    monkeypatch.setattr(calibration.fit_platt, "_fit_total", 0,
                        raising=False)
    with caplog.at_level(logging.WARNING, logger="calibration"):
        assert calibration.fit_platt(y, p) is None  # identity fallback
    msgs = [r.getMessage() for r in caplog.records
            if "degenerate Platt" in r.getMessage()]
    assert len(msgs) == 1 and f"n={n}" in msgs[0], (
        "the warning must carry the fit size — 'a=...' alone told the "
        "reviewer nothing about how thin the fit was", msgs)
    assert calibration.fit_platt._degen_total == 1  # uncapped ...
    assert calibration.fit_platt._fit_total == 1     # ... with denominator
    # the cap still works: only the first 3 degenerates print (1 done),
    # while the uncapped total keeps counting every one
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="calibration"):
        for _ in range(3):
            assert calibration.fit_platt(y, p) is None
    assert len([r for r in caplog.records
                if "degenerate Platt" in r.getMessage()]) == 2, (
        "the WARNING line is capped at 3 process-wide — fits 2 and 3 "
        "print here, the 4th must be silent", caplog.records)
    assert calibration.fit_platt._degen_total == 4  # counted regardless
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="calibration"):
        assert calibration.fit_platt(y, p) is None  # 5th — past the cap
    assert not [r for r in caplog.records
                if "degenerate Platt" in r.getMessage()]
    assert calibration.fit_platt._degen_total == 5


def test_run_engine_block_reports_its_own_degenerate_count():
    src = (BACKEND / "distributions.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "run_engine_daily")
    seg = ast.get_source_segment(src, fn) or ""
    assert "prequential Platt fits degenerated" in seg, (
        "run_engine_daily must report the block's uncapped degenerate "
        "total — the capped WARNING line alone hides the true count")
    # counters snapshotted BEFORE any fit and the summary must run inside
    # this function (before its return), not left to a caller
    assert seg.index("_degen0 = getattr") < seg.index("run_oof(decided")
    assert seg.index("prequential Platt fits degenerated") < seg.rindex(
        "return {\"block\": monitor_block")


# ── T5: provisional folds mark their own metric line ────────────────────────

def test_fold_metric_line_marks_provisional_folds():
    src = (BACKEND / "training.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and any(isinstance(a, ast.Constant)
                     and isinstance(a.value, str)
                     and "train=%d val=%d auc=%.4f brier=%.4f%s"
                     in a.value
                     for a in n.args)]
    assert calls, (
        "the fold metric line must carry a suffix slot — a grep for "
        "'auc=' used to sweep in excluded folds' numbers as graded")
    call_src = ast.unparse(calls[0])
    assert "provisional" in call_src and "excluded from grading" in call_src, (
        "the marker must ride the METRIC line itself, not only the "
        "separate PROVISIONAL line printed before training", call_src)


# ── T7: the pipeline runs successfully against the Kaggle notebook as-is ───

def test_stale_pinned_end_date_is_extended_to_today():
    """The notebook pins MLB_END_DATE="2026-10-06" for a rebuild and
    leaves it there; target=end, so an unguarded stale pin would make every
    later run re-predict that date. The pipeline defends itself."""
    from datetime import date, timedelta

    import config

    today = date.today().isoformat()
    stale = (date.today() - timedelta(days=3)).isoformat()
    assert config.resolve_run_end_date(stale) == today, (
        "a pin before today must extend to today — otherwise the daily "
        "slate freezes on the pinned date")
    assert config.resolve_run_end_date(today) == today  # same-day pin kept
    assert config.resolve_run_end_date("2099-01-01") == "2099-01-01", (
        "a deliberate future window is out of the guard's scope")
    assert config.resolve_run_end_date("not-a-date") == "not-a-date", (
        "malformed input passes through — Phase 1's strptime must keep "
        "failing loudly instead of silently defaulting to today")


def test_pipeline_resolves_end_date_through_the_guard_and_logs_it():
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    assert "resolve_run_end_date(" in src, (
        "CONFIG[end_date] must resolve through the guard — target=end, so "
        "an unguarded stale pin freezes the daily slate")
    # The extension is decided at CONFIG build (pre-tee), so it must be
    # re-announced AFTER install_run_log_tee or the pushed log would never
    # show why its window differs from the notebook's pin.
    i_tee = src.index("install_run_log_tee(")
    i_warn = src.index("is stale (before today)")
    assert i_tee < i_warn, "the staleness WARNING must land in the run log"


def test_full_repull_stays_a_honored_rebuild_switch():
    """The Kaggle-owned notebook sets MLB_FULL_REPULL=1 for rebuilds; the
    pipeline honors it (the 2026-10-06 run completed with the forced
    repull — gameless-window skips still applied), and T1's CR collapse
    keeps that run's progress frames to one line per bar."""
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    assert 'os.environ.get("MLB_FULL_REPULL"' in src
    assert "MLB_FULL_REPULL set — discarding cache" in src


# ── T6: DONE banner counts columns, Step 7 says who owns the push ───────────

def test_done_banner_labels_the_column_split():
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    assert "Columns: {game_df.shape[1]} game + {pbp_df.shape[1]} pbp" in src
    assert "Features: {game_df.shape[1]+pbp_df.shape[1]}" not in src, (
        "summing game + pbp columns under the label 'Features' read like a "
        "feature width (378) and contradicted 'Feature width: RFE subset "
        "(109 cols)' in the same log")


def test_step7_skip_line_says_who_owns_the_push():
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    assert 'logger.info("Step 7: Skipping GitHub sync (Phase 5 owns the "' in src, (
        "the daily run always skips Step 7 (skip_sync=True) while Phase 5 "
        "pushes — the bare 'Skipping GitHub sync' read as a broken run")
