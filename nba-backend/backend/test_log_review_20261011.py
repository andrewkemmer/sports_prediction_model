"""2026-10-11 NBA run-log review: the frozen off-season drift window.

Evidence (``data_delivery/nba_pipeline_run_log.txt`` + the delivered drift
CSVs): the run printed ``feature drift: 71 features, 0 warning(s), 10
alert(s)`` on a "current" window whose newest decided game was **2026-06-13**
— 120 days before the run.  The window builder did its job (widened 7 → 45
days to hold 31 rows, composition-matched the baseline to 220 postseason
rows), but nothing in the log said the window was frozen: the same 10 ALERTs
had been printed identically for weeks (20261006..20261011 CSVs carry the
same alert set), and a reader reasonably took "10 alerts" as a statement
about the October slate the comparison never measured.  Off-season roster
and trade movement is exactly what this frozen tail cannot see.

Remediation (``nba-backend/backend/monitoring.py``, disclosure only —
statuses, windows and thresholds untouched): when the current window's
newest decided game is more than ``DRIFT_STALE_DAYS`` (14) before the run,
``drift_windows`` warns with the window end, the age, and the fact a reader
needs — recurring alerts repeat the same frozen comparison and are not new
signal.  14 days clears every in-season gap (the All-Star break is ~1 week)
so the line fires only when the clock has genuinely stopped.

Also reviewed, no defect: 14 WARNINGs all classified (8 All-Star drops,
27 no-season-log/no-NBA-id disclosures = 11 postponed + 2 finished-no-lines
+ 14 upcoming slate, 2 position-index fallbacks, 1 designation gap, 1
OOF-thin); drift CSV 61 OK + 10 ALERT = 71 exactly (no hidden statuses);
``alpha_home 0.0040 (curve max 0.0034)`` reconciles once the parenthetical
is read as the per-row diagnostic it is — wording clarified in the same
commit (``per-row curve max``).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import monitoring as mon  # noqa: E402


def _decided(newest: str, n_recent: int = 40, n_prior: int = 200) -> pd.DataFrame:
    """A decided-games frame: a recent tail ending ``newest`` plus prior rows."""
    recent = pd.date_range(end=newest, periods=n_recent, freq="D")
    prior = pd.date_range(end=pd.Timestamp(newest) - pd.Timedelta(days=10),
                          periods=n_prior, freq="D")
    days = list(prior) + list(recent)
    return pd.DataFrame({
        "gameday": [d.date() for d in days],
        "is_playoffs": [1.0] * len(days),
        "elo_diff": [float(i % 7) - 3.0 for i in range(len(days))],
    })


@pytest.fixture
def stale_logs(caplog):
    def run(frame, as_of):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=mon.logger.name):
            baseline, current = mon.drift_windows(
                frame, days=7, min_baseline=20, min_current=10, as_of=as_of)
        return baseline, current, caplog.messages
    return run


def test_stale_window_discloses_the_frozen_tail(stale_logs):
    """The 2026-10-11 shape: newest decided game 120 days old."""
    _, current, messages = stale_logs(_decided("2026-06-13"),
                                      as_of=pd.Timestamp("2026-10-11"))
    stale = [m for m in messages if "frozen" in m and "UNMEASURED" in m]
    assert len(stale) == 1, messages
    assert "2026-06-13" in stale[0], stale[0]
    assert "not new signal" in stale[0], stale[0]
    assert "NOT the slate being served" in stale[0], stale[0]
    # The windows themselves are unchanged — disclosure only: the widened
    # window stops growing at min_current=10.
    assert len(current) == 10, len(current)


def test_fresh_window_stays_quiet(stale_logs):
    """In-season: the newest decided game is days old — no staleness line."""
    _, _, messages = stale_logs(_decided("2026-06-13"),
                                as_of=pd.Timestamp("2026-06-15"))
    assert not [m for m in messages if "frozen" in m], messages


def test_staleness_boundary_is_exactly_the_floor(stale_logs):
    """Age == DRIFT_STALE_DAYS stays quiet; one day more warns."""
    frame = _decided("2026-06-13")
    newest = pd.Timestamp("2026-06-13")
    _, _, messages = stale_logs(
        frame, as_of=newest + pd.Timedelta(days=mon.DRIFT_STALE_DAYS))
    assert not [m for m in messages if "frozen" in m], messages

    _, _, messages = stale_logs(
        frame, as_of=newest + pd.Timedelta(days=mon.DRIFT_STALE_DAYS + 1))
    stale = [m for m in messages if "frozen" in m]
    assert len(stale) == 1, messages


def test_drift_headline_publishes_evaluated_and_insufficient_counts(caplog):
    """2026-10-15 log review (MLB T15 / NHL 2026-10-11 parity): the headline
    read ``71 features, 0 warning(s), 10 alert(s)`` with no evaluated or
    INSUFFICIENT slot — the status enum HAS an INSUFFICIENT state, so a
    future small-window run would print 0 warnings/alerts while judging
    nothing and read clean. Counts must reconcile with the CSV; a judged
    window must NOT page."""
    import numpy as np
    mon._LAST_LOGGED_DRIFT = None
    rng = np.random.default_rng(7)
    cols = ["elo_diff", "win_pct_diff"]
    baseline = pd.DataFrame(rng.normal(0.0, 1.0, (300, len(cols))), columns=cols)
    current = pd.DataFrame(rng.normal(0.0, 1.0, (40, len(cols))), columns=cols)
    current["win_pct_diff"] = np.nan  # no current-window data → INSUFFICIENT
    with caplog.at_level(logging.INFO, logger=mon.logger.name):
        rows = mon.feature_drift(baseline, current, feature_cols=cols)
    assert [r["status"] for r in rows].count("INSUFFICIENT") == 1
    summary = [m for m in caplog.messages if m.startswith("feature drift:")]
    assert len(summary) == 1, caplog.messages
    assert "2 features, 1 evaluated, 1 INSUFFICIENT" in summary[0], summary[0]
    assert not [m for m in caplog.messages if "UNMEASURED" in m], (
        "a judged window must not page — INSUFFICIENT alone never warns")


def test_drift_headline_pages_when_nothing_was_judged(caplog):
    """The inert-monitor case: a current window under the 30-row floor
    leaves EVERY row INSUFFICIENT, and the headline must say drift is
    UNMEASURED at WARNING — an inert monitor must never read healthy."""
    import numpy as np
    mon._LAST_LOGGED_DRIFT = None
    rng = np.random.default_rng(9)
    cols = ["elo_diff", "win_pct_diff"]
    baseline = pd.DataFrame(rng.normal(0.0, 1.0, (300, len(cols))), columns=cols)
    current = pd.DataFrame(rng.normal(0.0, 1.0, (10, len(cols))), columns=cols)
    with caplog.at_level(logging.INFO, logger=mon.logger.name):
        mon.feature_drift(baseline, current, feature_cols=cols)
    summary = [m for m in caplog.messages if m.startswith("feature drift:")
               and "INSUFFICIENT," in m]
    assert len(summary) == 1, caplog.messages
    assert "2 features, 0 evaluated, 2 INSUFFICIENT" in summary[0], summary[0]
    paged = [m for m in caplog.messages if "UNMEASURED, not clean" in m]
    assert len(paged) == 1, caplog.messages


def test_default_as_of_is_the_wall_clock(stale_logs, monkeypatch):
    """Callers that pass no as_of judge staleness against now()."""
    frozen = pd.Timestamp("2026-10-11 09:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: frozen))
    _, _, messages = stale_logs(_decided("2026-06-13"), as_of=None)
    assert len([m for m in messages if "frozen" in m]) == 1, messages
