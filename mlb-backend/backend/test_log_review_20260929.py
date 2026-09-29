"""2026-09-30 log review: remediations for the two defects found in the
2026-09-29 production log.

D1 — the full-history re-pull burned CHUNK_RETRIES + backoff on three
     deep-offseason windows that provably contain zero MLB games, then
     reported them as possible data loss. The pull now skips known
     gameless windows before the first attempt.
D2 — the drift monitor's ~21-day trailing baseline crosses the season
     seam every final week, so REGULAR seasonal movement (playoff bullpen
     usage, eliminated-team call-ups) pages as WARN/ALERT. Measured
     2026-09-29: bullpen_whip_diff z=+2.78 and bullpen_whip_10g_away
     z=-3.37 vs trailing, BOTH vanishing against the same calendar phase
     of 2024-25 (z=+1.94 / -1.28). A location shift that survives the
     trailing baseline is now re-checked against prior-season
     same-calendar-phase windows and labeled OK-SEASONAL when clean.
"""
import ast
import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
import ingestion

BACKEND = Path(__file__).resolve().parent

# ── D1: known gameless windows are skipped before the first attempt ──────────


def test_known_gameless_windows_cover_the_20260929_storm():
    """The three windows the 09-29 log retried must all classify gameless."""
    assert ingestion._is_known_gameless_window(
        dt.date(2024, 1, 1), dt.date(2024, 2, 29))
    assert ingestion._is_known_gameless_window(
        dt.date(2024, 12, 26), dt.date(2025, 2, 23))
    assert ingestion._is_known_gameless_window(
        dt.date(2025, 12, 21), dt.date(2026, 2, 18))


def test_game_bearing_windows_are_never_skipped():
    """October postseason, late-March openers, and the mid-season core must
    all stay OUTSIDE the gameless set — a wrong entry here would silently
    drop real games instead of merely burning retries."""
    assert not ingestion._is_known_gameless_window(
        dt.date(2024, 10, 1), dt.date(2024, 10, 30))
    assert not ingestion._is_known_gameless_window(
        dt.date(2026, 3, 1), dt.date(2026, 3, 30))
    assert not ingestion._is_known_gameless_window(
        dt.date(2024, 7, 1), dt.date(2024, 7, 30))
    assert not ingestion._is_known_gameless_window(
        dt.date(2025, 2, 24), dt.date(2025, 4, 24))  # straddles window end


def test_gameless_skip_is_inside_the_chunk_loop():
    """Structural pin: _chunked_statcast must consult the gameless helper
    BEFORE attempting statcast, so the skip cannot regress silently."""
    src = (BACKEND / "ingestion.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "_chunked_statcast")
    names = {n.func.id for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_is_known_gameless_window" in names
    # and it must appear before the statcast call in source order
    order = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            order.append(n.func.id)
    assert order.index("_is_known_gameless_window") < order.index("statcast")


def test_config_knob_exists():
    assert config.DRIFT_PHASE_EXTENSION_MONTHS == (-1, -2)


# ── D2: season-seam reclassification to OK-SEASONAL ──────────────────────────

_COLS = ["f_seam", "f_regime", "f_quiet", "f_soon", "g"]


def _frames():
    """Baseline/current with three seam stories in one call:

    f_seam   shifts NOW and by the same amount one year ago  -> OK-SEASONAL
    f_regime shifts NOW but was FLAT one year ago            -> stays ALERT
    f_quiet  never moves                                     -> OK
    Returns (baseline, current, phase_frame) — the phase frame carries the
    PRIOR-SEASON same-calendar-phase rows (what production passes as the
    full decided frame).
    """
    rng = np.random.default_rng(11)

    def rows(start, days):
        d = pd.date_range(start, periods=days, freq="D")
        n = 10 * days
        data = {"game_date": np.repeat(d.values, 10)}
        for c in _COLS:
            data[c] = rng.normal(0.0, 1.0, n)
        return pd.DataFrame(data)

    cur = rows(pd.Timestamp("2026-09-16"), 12)
    base = rows(pd.Timestamp("2026-08-25"), 22)
    # phase frame: SAME calendar phase one year earlier (mirrors the
    # current window's width so the phase layer clears its 100-row floor,
    # as the production decided frame does with ~90 games/week x 2 yrs)
    phase = pd.concat([rows(pd.Timestamp("2025-09-16"), 12),
                       rows(pd.Timestamp("2025-08-20"), 16)],
                      ignore_index=True)
    # f_seam: +2.0 in the current window AND +2.0 in last year's same phase
    cur["f_seam"] += 2.0
    phase.loc[phase.game_date.dt.year == 2025, "f_seam"] += 2.0
    # f_regime: +2.0 now, nothing last year
    cur["f_regime"] += 2.0
    return base, cur, phase


def test_season_seam_reclassification():
    from explainability import compute_feature_drift
    base, cur, phase = _frames()
    df = compute_feature_drift(base, cur, "2099-01-01",
                               out_name="_t_seam_drift.csv",
                               feature_cols=_COLS, phase_frame=phase)
    row = df.set_index("feature")
    assert row.loc["f_seam", "status"] == "OK-SEASONAL", (
        "seasonal shift must not page")
    assert row.loc["f_regime", "status"] == "ALERT", (
        "a true regime break must still page")
    assert row.loc["f_quiet", "status"] == "OK"


def test_ok_seasonal_requires_enough_phase_rows():
    """No paging change when the phase window is too small to judge:
    with no prior-season rows at all, the re-check cannot fire and the
    trailing verdict stands."""
    from explainability import compute_feature_drift
    base, cur, phase = _frames()
    # drop all 2025 rows -> phase extension finds nothing
    phase = phase[phase.game_date.dt.year == 2026]
    df = compute_feature_drift(base, cur, "2099-01-01",
                               out_name="_t_seam_drift2.csv",
                               feature_cols=_COLS, phase_frame=phase)
    row = df.set_index("feature")
    assert row.loc["f_seam", "status"] in ("WARN", "ALERT")


def test_ok_seasonal_is_backward_compatible_without_phase_frame():
    """Callers that omit phase_frame (tests, historical scripts) get the
    old behavior exactly — the re-check finds no phase rows in the
    baseline-only frame and never reclassifies."""
    from explainability import compute_feature_drift
    base, cur, _phase = _frames()
    df = compute_feature_drift(base, cur, "2099-01-01",
                               out_name="_t_seam_drift3.csv",
                               feature_cols=_COLS)
    row = df.set_index("feature")
    assert row.loc["f_seam", "status"] in ("WARN", "ALERT")


def test_run_engine_drift_inherits_the_guard(monkeypatch):
    """The run-engine drift view calls the same machinery on the same
    windows, so its CSV carries the same OK-SEASONAL semantics."""
    import explainability
    from explainability import compute_run_engine_feature_drift
    monkeypatch.setattr(explainability, "run_engine_feature_cols",
                        lambda: list(_COLS))
    base, cur, phase = _frames()
    df = compute_run_engine_feature_drift(base, cur, "2099-01-01",
                                          phase_frame=phase)
    assert {"OK-SEASONAL", "ALERT", "OK"} <= set(df["status"])
    junk = BACKEND.parent / "data_delivery" / "run_engine_feature_drift_2099-01-01.csv"
    if junk.exists():
        junk.unlink()
