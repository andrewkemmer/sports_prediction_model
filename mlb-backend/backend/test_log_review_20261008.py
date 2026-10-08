"""2026-10-08 review: the frame-vs-official-tail guard (missing 10-07 slate).

Evidence from the committed run log (pushed at 9afdac73's successor, run
trained 2026-10-08 05:44Z): the run was a ``MLB_FULL_REPULL`` whose last
chunk (``2026-08-18 → 2026-10-08``) returned cleanly with 168,080 pitches
and zero retries — yet the frame horizon stayed ``2026-10-06`` while four
2026-10-07 games (CLE@CWS, LAD@ATL, TB@NYY, MIL@SD — all official Finals
before the run's last pitch) were absent from ``pitches.parquet``. Savant
posting lag, not a query bug. Consequences that went UNLOGGED:

  * ``predictions_history_20261008.csv`` tops out at 2026-10-06 — the
    10-07 slate never resolved into predictions history / today's record,
    so a whole board's results silently vanished from the dashboards;
  * the IL ledger even printed ``lag -2d vs data horizon 2026-10-06``
    without anyone connecting it to missing finals;
  * ``_abort_on_exhausted_core_season_empty_chunk`` could not catch it —
    the chunk was NON-empty, just stale.

Remediation: ``ingestion.warn_missing_finals`` runs in Phase 1 right after
the frame lands, compares the frame's newest game against the official
StatsAPI schedule for the tail window, and WARNs (never aborts — posting
lag is transient and the next run's forward top-up + REFRESH_TAIL_DAYS
refresh recovers the games) when finals are absent.

Convention: behavioral tests for the importable module (ingestion),
source pins for the run-once script (master_pipeline) — same style as
test_log_review_{20260929,20261005,20261007}.
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import ingestion  # noqa: E402
from ingestion import warn_missing_finals  # noqa: E402

MASTER_SRC = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")

GUARD_TAG = "missing-finals guard"


# ── Fixtures / helpers ──────────────────────────────────────────────────────

def _frame(tmp_path: Path) -> Path:
    """A tiny pitch frame whose horizon is 2026-10-06 (the observed case)."""
    p = tmp_path / "pitches.parquet"
    pd.DataFrame({
        "game_date": pd.to_datetime(["2026-10-05", "2026-10-06", "2026-10-06"]),
        "game_pk": [100, 200, 300],
    }).to_parquet(p, index=False)
    return p


def _official(rows: list[dict], monkeypatch, capture: list | None = None):
    """Stand-in for results.fetch_mlb_results with the real column contract."""
    cols = ["game_pk", "game_date", "home_score", "away_score", "home_win",
            "is_final", "home_team", "away_team"]
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows], columns=cols)

    def _fake(start_date, end_date, timeout: int = 20):
        if capture is not None:
            capture.append((start_date, end_date))
        return df

    import results
    monkeypatch.setattr(results, "fetch_mlb_results", _fake)
    return df


# ── T1: the guard flags finals absent from the frame ────────────────────────

def test_guard_flags_finals_newer_than_the_frame(tmp_path, monkeypatch, caplog):
    """The 10-07 defect, replayed: finals exist officially, frame lags."""
    path = _frame(tmp_path)
    _official([
        # An official FINAL newer than the frame — must be flagged.
        {"game_pk": 849833, "game_date": "2026-10-07", "is_final": True,
         "home_score": 4.0, "away_score": 2.0, "home_win": 1.0,
         "home_team": "CWS", "away_team": "CLE"},
        {"game_pk": 849827, "game_date": "2026-10-07", "is_final": True,
         "home_score": 3.0, "away_score": 5.0, "home_win": 0.0,
         "home_team": "SD", "away_team": "MIL"},
        # Not final yet (the run date's in-progress slate) — must NOT flag.
        {"game_pk": 849850, "game_date": "2026-10-08", "is_final": False},
    ], monkeypatch)

    with caplog.at_level(logging.WARNING, logger="ingestion"):
        rows = warn_missing_finals(path, date(2026, 10, 8))

    assert [r["game_pk"] for r in rows] == [849833, 849827], (
        "guard must return exactly the official finals absent from the frame")
    assert all(r["game_date"] == "2026-10-07" for r in rows)
    assert rows[0]["away_team"] == "CLE" and rows[0]["home_team"] == "CWS"

    text = caplog.text
    assert GUARD_TAG in text, "the warning must be tagged for log review"
    assert "ABSENT from pitches.parquet" in text
    assert "2026-10-06" in text, "warning must name the lagging frame horizon"
    assert "849833" in text and "CLE@CWS" in text, (
        "each missing game must be listed with pk and matchup")
    assert "recovers them" in text, "the note must say how the gap self-heals"
    # Non-final games and in-frame games never leak into the list.
    assert "849850" not in text


def test_guard_queries_only_the_tail_window(tmp_path, monkeypatch):
    """Cost/semantics pin: the fetch spans horizon+1 → end, nothing wider."""
    path = _frame(tmp_path)
    seen: list = []
    _official([{"game_pk": 1, "game_date": "2026-10-07", "is_final": False}],
              monkeypatch, capture=seen)

    warn_missing_finals(path, date(2026, 10, 8))

    assert seen == [(date(2026, 10, 7), date(2026, 10, 8))], (
        "guard must fetch only frame_horizon+1 .. end_date")


def test_guard_silent_when_the_frame_covers_the_run_window(tmp_path, monkeypatch):
    """Horizon == end (Savant current): no schedule call, no warning."""
    path = _frame(tmp_path)

    def _boom(*_a, **_k):
        raise AssertionError("must not fetch the schedule when nothing can lag")

    import results
    monkeypatch.setattr(results, "fetch_mlb_results", _boom)

    assert warn_missing_finals(path, date(2026, 10, 6)) == []
    assert warn_missing_finals(path, date(2026, 10, 5)) == []


def test_guard_silent_when_the_final_is_already_in_the_frame(tmp_path,
                                                             monkeypatch,
                                                             caplog):
    """Suspended-and-resumed listings (final under a later date while the
    pitches are already framed under the original date) must not warn."""
    path = _frame(tmp_path)
    _official([
        # Same pk as an in-frame game (game_pk 200) under a later date.
        {"game_pk": 200, "game_date": "2026-10-07", "is_final": True,
         "home_team": "NYY", "away_team": "TB"},
    ], monkeypatch)

    with caplog.at_level(logging.WARNING, logger="ingestion"):
        rows = warn_missing_finals(path, date(2026, 10, 8))

    assert rows == []
    assert GUARD_TAG not in caplog.text or "ABSENT" not in caplog.text


def test_guard_never_raises_on_a_degraded_frame(tmp_path, caplog):
    """A guard must degrade to a warning — it may not kill the run."""
    missing = tmp_path / "nope.parquet"
    with caplog.at_level(logging.WARNING, logger="ingestion"):
        assert warn_missing_finals(missing, date(2026, 10, 8)) == []
    assert GUARD_TAG in caplog.text

    # Unreadable parquet (garbage bytes) — same contract.
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not a parquet file")
    with caplog.at_level(logging.WARNING, logger="ingestion"):
        assert warn_missing_finals(bad, date(2026, 10, 8)) == []


# ── T2: Phase 1 wiring (source pins — master_pipeline is run-once) ──────────

def test_phase1_runs_the_guard_right_after_the_frame_lands():
    """The guard fires AFTER the pull and BEFORE Phase 1.5 / feature build,
    so its warning lands in the pushed run log before training starts."""
    assert "from ingestion import pull_statcast, warn_missing_finals" in MASTER_SRC, (
        "Phase 1 must import the guard with the pull")
    pull_i = MASTER_SRC.index('print(f"  ✅ Raw pitches: {pitches_path}")')
    call_i = MASTER_SRC.index("warn_missing_finals(pitches_path, end)", pull_i)
    phase15_i = MASTER_SRC.index("Phase 1.5", pull_i)
    assert pull_i < call_i < phase15_i, (
        "the missing-finals guard must run between the pull and Phase 1.5")


def test_phase1_guard_is_warn_only_and_never_kills_the_run():
    """Posting lag is transient: the guard may warn, never abort. Its own
    failure is swallowed into a logging.warning (observability, not a
    dependency)."""
    pull_i = MASTER_SRC.index('print(f"  ✅ Raw pitches: {pitches_path}")')
    phase15_i = MASTER_SRC.index("Phase 1.5", pull_i)
    block = MASTER_SRC[pull_i:phase15_i]
    assert "try:" in block and "except Exception" in block, (
        "guard invocation must be wrapped so a defect cannot stop Phase 1")
    assert "missing-finals guard skipped" in block
    # The guard runs UNCONDITIONALLY on every pull — not behind a flag a
    # routine run could skip (unlike the MLB_FULL_REPULL optionals).
    call_lines = [ln.strip() for ln in block.splitlines()
                  if "warn_missing_finals(pitches_path, end)" in ln]
    assert call_lines == ["warn_missing_finals(pitches_path, end)"], (
        "the guard must be a bare call in Phase 1, not conditionally gated")
