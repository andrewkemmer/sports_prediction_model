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
FEATURES_SRC = (BACKEND / "features.py").read_text(encoding="utf-8")

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


# ── T3: bullpen empty-window fill (2026-10-08 feature-discrepancy review) ───
# A team with NO outings inside the trailing-3d window threw ZERO bullpen
# pitches — a known fact (rest days, the All-Star break, postseason byes,
# opener gaps). The bp_fatigue LEFT JOIN turned the empty window into NULL
# and erased the whole bullpen family on exactly those rows (115 rows /
# 1.6% of the frame; all four 2026-10-03 DS openers carried 7/11 coverage
# into the monitor). The fill is scoped to "team has an earlier game in the
# frame" — a team with no prior game keeps NULL.

def _bp_fatigue_sql() -> str:
    """The PRODUCTION bp_fatigue statement, extracted from features.py so
    the test exercises the shipped SQL rather than a copy of it."""
    start = FEATURES_SRC.index("CREATE TABLE bp_fatigue AS")
    end = FEATURES_SRC.index('"""', start)
    return FEATURES_SRC[start:end]


def test_bp_fatigue_empty_window_is_zero_not_null():
    import duckdb

    con = duckdb.connect()
    con.execute("CREATE TABLE pitches (game_pk INTEGER, game_date DATE, "
                "home_team VARCHAR, away_team VARCHAR)")
    con.execute("CREATE TABLE bp_daily (day DATE, team VARCHAR, "
                "pitches INTEGER, ip DOUBLE)")
    con.execute("CREATE TABLE bp_day2 (game_pk INTEGER, home_pitches_2d INTEGER, "
                "home_ready_2d INTEGER, away_pitches_2d INTEGER, "
                "away_ready_2d INTEGER)")
    con.executemany("INSERT INTO pitches VALUES (?, ?, ?, ?)", [
        (1, "2026-03-27", "AAA", "BBB"),  # opener: no earlier game either side
        (2, "2026-04-01", "AAA", "BBB"),  # AAA rested (prior game, empty window)
        (3, "2026-07-18", "AAA", "BBB"),  # ASB return: pre-break work only
    ])
    con.executemany("INSERT INTO bp_daily VALUES (?, ?, ?, ?)", [
        ("2026-03-31", "BBB", 40, 4.0),   # inside game 2's window → passes through
        ("2026-07-12", "AAA", 30, 3.0),   # before game 3's window
        ("2026-07-11", "BBB", 25, 2.5),   # before game 3's window
    ])
    con.execute(_bp_fatigue_sql())
    rows = con.execute(
        "SELECT game_pk, bullpen_pitches_3d_home, bullpen_ip_3d_home, "
        "       bullpen_pitches_3d_away, bullpen_ip_3d_away "
        "FROM bp_fatigue ORDER BY game_pk").fetchall()
    got = {r[0]: r for r in rows}

    # Opener: no earlier game → NULL preserved (non-vacuity: the fill is
    # scoped to "has prior game", never a blanket COALESCE).
    assert got[1][1] is None and got[1][3] is None, got[1]
    # In-window work passes through untouched (non-vacuity).
    assert got[2][3] == 40 and abs(got[2][4] - 4.0) < 1e-9, got[2]
    # Empty window + earlier game → known 0 (home rested; game 3 both sides).
    assert got[2][1] == 0 and abs(got[2][2] - 0.0) < 1e-9, got[2]
    assert got[3][1] == 0 and got[3][3] == 0, got[3]


# ── T4: indoor-neutral fill (unconditional, wired into BOTH passes) ─────────

def test_indoor_neutral_fill_is_unconditional_and_open_air_stays_null():
    """Closed roof → both weather interactions are 0 whatever the
    pitcher-side inputs say (the wind multiplier is structurally 0 indoors
    and air effects are policy-neutral). Open-air rows and unknown-roof
    rows are never touched — a missing outdoor observation stays NULL."""
    import numpy as np
    import features

    df = pd.DataFrame({
        "dome_is_neutral_game": [1.0, 1.0, 0.0, np.nan],
        "wind_advantage_flyball_factor": [np.nan, 0.5, np.nan, np.nan],
        "air_density_velocity_boost": [np.nan, np.nan, np.nan, 0.3],
    })
    out = features.apply_indoor_neutral_fills(df)
    assert out["wind_advantage_flyball_factor"].tolist()[:2] == [0.0, 0.0]
    assert np.isnan(out["wind_advantage_flyball_factor"].iloc[2])
    assert np.isnan(out["wind_advantage_flyball_factor"].iloc[3])
    assert out["air_density_velocity_boost"].tolist()[:2] == [0.0, 0.0]
    assert np.isnan(out["air_density_velocity_boost"].iloc[2])
    assert out["air_density_velocity_boost"].iloc[3] == 0.3

    # Venue-prior fallback when the game-state column is absent.
    venue = pd.DataFrame({
        "dome_is_neutral": [1.0, 0.0],
        "wind_advantage_flyball_factor": [np.nan, np.nan],
    })
    vout = features.apply_indoor_neutral_fills(venue)
    assert vout["wind_advantage_flyball_factor"].iloc[0] == 0.0
    assert np.isnan(vout["wind_advantage_flyball_factor"].iloc[1])


def test_indoor_fill_is_wired_into_both_passes_and_old_conditionals_gone():
    """add_diff_features builds the interactions and add_env_level_features
    runs LAST (refine → env) — both must route through the shared helper,
    because the env pass's old conditional version was the pipeline's final
    word and re-NaN'd rows the build had already zeroed."""
    src = FEATURES_SRC
    assert src.count("apply_indoor_neutral_fills(df)") == 2, (
        "expected exactly two production call sites (add_diff_features + "
        "add_env_level_features)")
    # The old input-conditioned / forced-NaN lines must be gone.
    assert 'df.loc[dome_flag & _era_ok' not in src
    assert 'df.loc[dome_flag & ~_density_ok' not in src
    assert 'df.loc[_closed & df["sp_era_diff"].notna()' not in src
