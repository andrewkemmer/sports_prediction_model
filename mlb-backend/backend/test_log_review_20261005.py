"""2026-10-05 run-log review: remediations for the defects found on remote.

The committed ``mlb-backend/data_delivery/mlb_pipeline_run_log.txt`` was
reviewed line by line (1,051 lines at HEAD ae929bf3) and its defects were
confirmed against the emitting source. Each fix is pinned here:

R1 — crash delivery could ship a log with NO failure context: the default
     excepthook prints the traceback to whatever ``sys.stderr`` holds at
     crash time, so a stream rebound by a harness after the tee installed
     routes the traceback everywhere EXCEPT the file. ``_record_failure``
     now appends an explicit crash marker + formatted traceback straight
     to the file BEFORE push_log_on_crash copies it.
R2 — all six ``[MEM]`` lines printed the identical peak number (5955 MB
     x6): ``ru_maxrss`` is a high-water mark. ``_mem_mb`` now reads the
     LIVE resident set from /proc/self/statm, peak only as fallback.
R3 — two ~7 KB single-line column dumps (6,953 / 6,994 chars) polluted
     the log. Both call sites now log a column COUNT.
R4 — the moneyline and run-engine monitoring views emit the SAME unlabeled
     lines from the same functions, so "Feature drift: 109 features..." /
     "Feature coverage gaps: ..." appeared as indistinguishable twins.
     Both functions now carry a ``view`` label (moneyline vs run-engine).
R5 — Phase 5 stages the log as a SNAPSHOT before its own output and all of
     Phase 6 reach the file, so every pushed log ended at the bare
     "PHASE 5 — GitHub Sync" banner. master_pipeline now re-pushes the
     complete log as the run's LAST delivery, after the DONE banner.
R6 — kaggle_mlb_run.ipynb pinned MLB_FULL_REPULL=1 ("set once, then
     remove") and a stale MLB_END_DATE=2026-09-29, forcing a full
     Statcast re-pull on every daily run. Both are now commented out;
     the 2024-01-01 start (training history) stays.

Plus the guard: installing the tee opens the rolling log with "w", so a
pytest import of master_pipeline would TRUNCATE the committed log (the
NBA incident, 5673874c: 143 bytes). The install is now pytest-skipped.

Convention: source/AST pins for emitter changes, behavioral tests where
the machinery is importable (run_log_tee, explainability) — the same
style as test_run_log_observability / test_log_review_20260929.
"""
from __future__ import annotations

import ast
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
REPO_ROOT = BACKEND.parent.parent
DD = BACKEND.parent / "data_delivery"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import run_log_tee  # noqa: E402
from retention_policy import classify_artifact  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_view_junk():
    """Drift/coverage smoke calls write probe CSVs into data_delivery;
    remove them (and any stale ones) around every test."""
    yield
    for junk in DD.glob("_t_*.csv"):
        junk.unlink(missing_ok=True)
    for name in ("run_engine_feature_drift_2099-01-01.csv",
                 "run_engine_feature_coverage_2099-01-01.csv"):
        (DD / name).unlink(missing_ok=True)


# ── R1: every crash delivery carries its own failure context ────────────────

def test_record_failure_appends_marker_without_truncating(tmp_path):
    log = tmp_path / run_log_name()
    log.write_text("partial run output\n", encoding="utf-8")
    err = ValueError("kaput")
    run_log_tee._record_failure(log, type(err), err, None)
    text = log.read_text(encoding="utf-8")
    assert "PIPELINE CRASH" in text
    assert "ValueError: kaput" in text
    assert "partial run output" in text  # append, never truncate


def test_crash_hook_records_failure_into_the_file_before_pushing(
        tmp_path, monkeypatch):
    """The marker must be ON DISK by the time push_log_on_crash copies the
    file — a delivery whose local copy lacks the failure line is the exact
    defect this guards."""
    log = tmp_path / run_log_name()
    log.write_text("run so far\n", encoding="utf-8")
    monkeypatch.setattr(sys, "excepthook", lambda *a: None)
    seen: dict = {}

    def _fake_push(*a, **k):
        seen["content"] = log.read_text(encoding="utf-8")
        return True

    monkeypatch.setattr(run_log_tee, "push_log_on_crash", _fake_push)
    run_log_tee.install_crash_log_pusher(log, "u", "r")
    try:
        raise RuntimeError("phase 3 exploded")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())  # what the interpreter would do
    assert "run so far" in seen["content"]
    assert "PIPELINE CRASH" in seen["content"]
    assert "phase 3 exploded" in seen["content"]


def test_crash_hook_still_skips_systemexit_and_never_masks(tmp_path,
                                                           monkeypatch):
    """Guard the guard: SystemExit stays a non-crash, and a marker-write
    failure must never raise out of the hook."""
    log = tmp_path / run_log_name()
    log.write_text("x", encoding="utf-8")
    calls: list = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: None)
    monkeypatch.setattr(run_log_tee, "push_log_on_crash",
                        lambda *a, **k: calls.append("push") or True)
    monkeypatch.setattr(run_log_tee, "_record_failure",
                        lambda *a, **k: (_ for _ in ()).throw(
                            OSError("disk gone")))
    run_log_tee.install_crash_log_pusher(log, "u", "r")
    err = ValueError("original")
    sys.excepthook(type(err), err, None)  # must NOT raise
    assert calls == ["push"]  # delivery still attempted after marker OSError
    calls.clear()
    sys.excepthook(SystemExit, SystemExit(1), None)
    assert "push" not in calls


def run_log_name() -> str:
    return run_log_tee.RUN_LOG_NAME


# ── R2: [MEM] checkpoints read the live resident set, not the peak ──────────

def test_mem_mb_reads_current_rss_from_proc_statm():
    src = (BACKEND / "features.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_mem_mb")
    seg = ast.get_source_segment(src, fn)
    assert "/proc/self/statm" in seg, (
        "current RSS must come from /proc/self/statm — ru_maxrss alone "
        "prints the same peak on every checkpoint (5955 MB x6)")
    # statm is the primary route; getrusage (ru_maxrss peak) survives
    # only as the non-Linux fallback — compare against the CALL, not the
    # docstring mention
    assert seg.index("/proc/self/statm") < seg.index("resource.getrusage")
    assert "ru_maxrss" in seg  # fallback still present


def test_mem_mb_returns_a_positive_float():
    import features  # noqa: E402 — existing suite imports features too
    val = features._mem_mb()
    assert isinstance(val, float) and val > 0


# ── R3: column COUNTS instead of full column-list dumps ─────────────────────

def test_data_ingestion_logs_column_counts_not_column_lists():
    src = (BACKEND / "data_ingestion.py").read_text(encoding="utf-8")
    assert "list(df.columns)" not in src, (
        "a full column list in a log call is the ~7 KB single-line dump "
        "class the review flagged (6,953 / 6,994 chars)")
    assert "columns: %s" not in src
    assert "Loaded %d games from %s (%d columns)" in src
    assert "Feature mapping complete: %d games, %d columns" in src


# ── R4: drift/coverage log lines name their view ────────────────────────────

def _frames():
    rng = np.random.default_rng(5)

    def rows(n, start):
        d = pd.date_range(start, periods=n, freq="D")
        return pd.DataFrame({"game_date": d, "f_view": rng.normal(0, 1, n)})

    return rows(150, "2026-08-01"), rows(120, "2026-09-20")


def test_drift_and_coverage_lines_carry_the_view_label(caplog):
    import explainability

    base, cur = _frames()
    with caplog.at_level(logging.INFO, logger="explainability"):
        explainability.compute_feature_drift(
            base, cur, "2099-01-01", feature_cols=["f_view"],
            out_name="_t_view_drift.csv")
        explainability.compute_feature_coverage(
            base, cur, "2099-01-01", feature_cols=["f_view"],
            out_name="_t_view_cov.csv")
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Feature drift [moneyline]:") for m in msgs), msgs


def test_run_engine_wrappers_label_their_lines_run_engine(caplog):
    import explainability

    base, cur = _frames()
    orig = explainability.run_engine_feature_cols
    monkey_cols = ["f_view"]
    explainability.run_engine_feature_cols = lambda: list(monkey_cols)
    try:
        with caplog.at_level(logging.INFO, logger="explainability"):
            explainability.compute_run_engine_feature_drift(
                base, cur, "2099-01-01")
            explainability.compute_run_engine_feature_coverage(
                base, cur, "2099-01-01")
    finally:
        explainability.run_engine_feature_cols = orig
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Feature drift [run-engine]:") for m in msgs), (
        "run-engine drift must be distinguishable from moneyline drift", msgs)
    assert any(m.startswith("Feature coverage") and "[run-engine]"
               in m for m in msgs), msgs


def test_view_defaults_to_moneyline_for_existing_callers():
    src = (BACKEND / "explainability.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for fname in ("compute_feature_drift", "compute_feature_coverage"):
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == fname)
        args = [a.arg for a in fn.args.args]
        assert "view" in args, f"{fname} lost its view label"
        assert fn.args.defaults and any(
            isinstance(d, ast.Constant) and d.value == "moneyline"
            for d in fn.args.defaults), (
            f"{fname} view must default to moneyline so every existing "
            "call site stays labeled")


# ── R5: the complete log is pushed as the run's last delivery ───────────────

def test_final_log_delivery_runs_after_done_banner_and_before_cleanup():
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    i_banner = src.index('_banner("DONE')
    i_copy = src.index("shutil.copy2(_log_path, _log_dest)")
    i_rmtree = src.index("shutil.rmtree(sync_dir, ignore_errors=True)")
    i_exit = src.index("MLB pipeline finished WITH ERRORS")
    assert i_banner < i_copy, (
        "the DONE banner must be flushed into the log BEFORE the final "
        "copy, or the pushed log cannot show the run completed")
    assert i_copy < i_rmtree, (
        "the final delivery needs the warm sync clone — push before the "
        "clone is removed")
    assert i_copy < i_exit, (
        "an errored run must still deliver its log before SystemExit")
    # retry-safe and remotely verified, mirroring Phase 5's own push
    assert "restage=_restage_run_log" in src
    assert 'verify_pushed_paths(repo, CONFIG["github_branch"], [_log_rel])' in src


def test_phase5_staging_still_includes_the_log_and_retention_protects_it():
    """The final delivery is additive: Phase 5 keeps staging the log (so a
    crash between Phase 5 and the final push still ships a partial log),
    and Phase 6 still never evicts it."""
    verdict = classify_artifact(
        "mlb-backend/data_delivery/mlb_pipeline_run_log.txt",
        seen=set(), retention_dates=set(), recent_dates=set(),
        board_dates=set())
    assert verdict == "protected"


# ── D7: a pytest import must never truncate the committed rolling log ───────

def test_tee_install_is_guarded_against_pytest():
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    i_guard = src.index(
        'if os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:')
    i_install = src.index("install_run_log_tee(Path.cwd()")
    assert i_guard < i_install, (
        "the pytest guard must sit above the tee install — installing "
        "opens the rolling log with 'w' and truncates it")


# ── R6: the notebook's daily-run environment overrides ──────────────────────

def _notebook_run_option_lines(nb, needle: str) -> list[str]:
    src = nb["cells"][0]["source"]
    text = "".join(src) if isinstance(src, list) else src
    return [l.strip() for l in text.splitlines() if needle in l]


def test_notebook_daily_run_does_not_force_full_repull():
    nb = json.loads(
        (REPO_ROOT / "kaggle_mlb_run.ipynb").read_text(encoding="utf-8"))
    lines = [l for l in _notebook_run_option_lines(nb, "MLB_FULL_REPULL")
             if "os.environ" in l]
    assert lines, "notebook must still document MLB_FULL_REPULL"
    for line in lines:
        assert line.startswith("#"), (
            "MLB_FULL_REPULL must be commented out for daily runs — this "
            "pin discarded the chunk cache and re-pulled full Statcast "
            "history on EVERY run (log line: '♻️ MLB_FULL_REPULL set')")


def test_notebook_daily_run_does_not_pin_a_stale_end_date():
    nb = json.loads(
        (REPO_ROOT / "kaggle_mlb_run.ipynb").read_text(encoding="utf-8"))
    end_lines = [l for l in _notebook_run_option_lines(nb, "MLB_END_DATE")
                 if "os.environ" in l]
    assert end_lines, "notebook must still document MLB_END_DATE"
    for line in end_lines:
        assert line.startswith("#"), (
            "MLB_END_DATE must be commented out for daily runs — a pinned "
            "date goes stale (2026-09-29 committed while runs ran "
            "2026-10-04) and the default already resolves to today")
    # the training-history start stays active
    start_lines = [l for l in _notebook_run_option_lines(nb, "MLB_START_DATE")
                   if "os.environ" in l]
    assert start_lines and not start_lines[0].startswith("#"), (
        "MLB_START_DATE=2024-01-01 is live training history — the "
        "pipeline default (2025-01-01) would silently halve it")
