"""2026-10-08 review: the frame-vs-official-tail guard (missing 10-07 slate).

PLUS (2026-10-09, no new test file per the no-new-programs guardrail) the
T6/T7/T8 section: the weather half of the feature-coverage audit. T6 pins
the exact-midnight first-pitch defect (13 games shipped NO weather), T7
pins the partial wind-only cache record top-up, T8 pins the honest refusal
to invent a wind direction from an official "Varies" report.

PLUS (same day, feature-accuracy pass over the same run log): venue
defects T5 — the static team→park map had no arm for the Statcast codes
AZ/ATH (486 home rows shipped venue='Unknown') and pinned TB's park to
Steinbrenner Field for EVERY season (164 wrong labels: Tropicana hosted
2024 and 2026+) plus the matching dome defect (all 81 open-air 2025
Steinbrenner games claimed a closed roof, zeroing the weather
interactions). Verified against the official StatsAPI schedule.

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

PLUS (2026-10-09, coverage-audit pass over the same run log) the T11
section: the standing weather LOW_COVERAGE alarm classifies STRUCTURAL
only when EVERY unmeasured row is explained by a declared policy
(closed-roof zero / SP-staleness-gated NULL); one unobserved open-air
row beside a valid SP input keeps the raw alarm (the 2026-10-07
truncation class).

PLUS (2026-10-09, third-delivery pass) the T12 notebook pins: the 19:08
run's artifacts and log were perfect (first run carrying the STRUCTURAL
statuses), but the notebook SESSION died after delivery with
FileNotFoundError on a typo'd duplicate of the ``repo`` literal in the
confirmation block of the LIVE Kaggle copy
(``/kaggle/working/sports_predictio_model``). The notebook is
Kaggle-owned (T7, test_log_review_20261006) — never edited from the
repo — so the repo-side defense is a content pin: a Kaggle re-upload
carrying this defect class fails the suite instead of shipping silently.
PLUS T13 (the resolution under the no-change-to-the-notebook guardrail):
the RUN repairs the typo'd path itself — master_pipeline's final-delivery
tail calls github_sync.ensure_notebook_sanity_alias, which points the
typo'd location at the real clone through a directory symlink before the
notebook's confirmation block executes, so sessions end green with the
advisory git checks reporting the ACTUAL repository. The notebook
program is never modified; the pins remain the defense for the
committed copy.

Convention: behavioral tests for the importable module (ingestion),
source pins for the run-once script (master_pipeline) — same style as
test_log_review_{20260929,20261005,20261007}.
"""
from __future__ import annotations

import ast
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
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
    assert 'df.loc[_closed & df["sp_xfip_diff"].notna()' not in src


# ── T5: venue accuracy + TB's season-dependent dome (2026-10-08 review) ─────
# Two production defects found by reconciling the shipped frame against
# the official StatsAPI schedule:
#   1. VENUE_MAP had no arm for the Statcast team codes AZ/ATH (it was
#      keyed ARI/OAK), so the venues CASE fell to ELSE 'Unknown' for all
#      486 AZ/ATH home rows (2024–2026) — venue display/metadata coverage
#      hole, UNK category in venue_id.
#   2. VENUE_MAP["TB"] was a static "Steinbrenner Field": wrong park name
#      for 2024/2026 (Tropicana Field) AND wrong dome prior for the 2025
#      Steinbrenner season — refine_dome_game_level treated TB as a
#      BLANKET fixed dome, so all 81 open-air 2025 games got
#      dome_is_neutral_game=1 → park_wind_factor and both weather
#      interactions forced to 0 for games the weather composites had
#      measured for real.


def test_venue_case_sql_is_season_aware_and_covers_statcast_codes():
    """Execute the PRODUCTION venue CASE (features._venue_case_sql) on a
    tiny table: every Statcast code resolves to the official park for the
    game's season; only unknown teams reach 'Unknown'."""
    import duckdb
    import features

    # Key-set invariant: the map must cover every real Statcast code —
    # the root cause of the 486 'Unknown' rows was a missing key.
    missing = features.REAL_TEAM_CODES - set(features.VENUE_MAP)
    assert not missing, f"VENUE_MAP missing Statcast codes: {sorted(missing)}"
    assert {"ARI", "OAK"} <= set(features.VENUE_MAP)  # StatsAPI aliases kept

    con = duckdb.connect()
    con.execute("CREATE TABLE pitches (game_pk INTEGER, game_date DATE, "
                "home_team VARCHAR)")
    con.executemany("INSERT INTO pitches VALUES (?, ?, ?)", [
        (1, "2024-03-28", "TB"),   # Tropicana (fixed dome)
        (2, "2025-04-01", "TB"),   # Steinbrenner (open air)
        (3, "2026-04-01", "TB"),   # Tropicana again
        (4, "2024-04-01", "ATH"),  # Oakland Coliseum (ATH code, 2024)
        (5, "2025-04-01", "ATH"),  # Sutter Health Park
        (6, "2026-04-01", "AZ"),   # Chase Field (Statcast code)
        (7, "2026-04-01", "SEA"),  # static arm still works
        (8, "2026-04-01", "ZZZ"),  # unknown team → Unknown
    ])
    rows = con.execute(
        f"SELECT game_pk, home_team, {features._venue_case_sql()} AS venue "
        "FROM pitches ORDER BY game_pk").fetchall()
    venue = {r[0]: r[2] for r in rows}
    assert venue[1] == "Tropicana Field"
    assert venue[2] == "Steinbrenner Field"
    assert venue[3] == "Tropicana Field"
    assert venue[4] == "Oakland Coliseum"
    assert venue[5] == "Sutter Health Park"
    assert venue[6] == "Chase Field"
    assert venue[7] == "T-Mobile Park"
    assert venue[8] == "Unknown"


def test_tb_dome_is_season_aware_in_prior_and_refinement():
    """TB home games resolve dome=1 for the Tropicana seasons (2024,
    2026+) and dome=0 for the open-air 2025 Steinbrenner season, in BOTH
    layers: the DOME_STATUS prior inside add_diff_features and the
    game-accurate refinement (refine_dome_game_level)."""
    import features

    df = pd.DataFrame({
        "game_date": ["2024-04-01", "2025-04-01", "2026-04-01"],
        "home_team": ["TB", "TB", "TB"],
        "away_team": ["NYY", "NYY", "NYY"],
        "home_win": [1.0, 0.0, 1.0],
    })
    prior = features.add_diff_features(df.copy())
    assert prior["dome_is_neutral"].tolist() == [1.0, 0.0, 1.0], (
        "the static TB=1 prior must be corrected for the 2025 season")

    refined = features.refine_dome_game_level(prior.copy())
    assert refined["dome_is_neutral_game"].tolist() == [1.0, 0.0, 1.0]
    # The model feature is SYNCED to the game-accurate state.
    assert refined["dome_is_neutral"].tolist() == [1.0, 0.0, 1.0]

    # Non-TB fixed-dome behavior is unchanged (regression guard), and the
    # helper is the single source of truth for the fixed-dome branch.
    assert features._fixed_dome_game_value(
        "TB", {"game_date": "2025-07-01"}) == 0.0
    assert features._fixed_dome_game_value(
        "TB", {"game_date": "2026-07-01"}) == 1.0
    assert features._fixed_dome_game_value(
        "TB", {"game_date": None}) == 1.0  # unknown date → fixed dome
    assert features.TB_OPEN_AIR_YEARS == frozenset({2025})
    assert features.FIXED_DOME_TEAMS == frozenset({"TB"})


# ── T6: exact-midnight first pitch → weather from the PREVIOUS day ─────────
# 13 games through 2026-10-07 have an OFFICIAL first pitch at exactly
# 00:00:00 UTC (8:00 PM ET / 5:00 PM PT — the national-TV slot; verified
# against StatsAPI ``gameData.datetime.dateTime`` for every one of them).
# Inside such a game's own UTC-day series there is no hourly row STRICTLY
# before first pitch, so the point-in-time rule found nothing and those
# games shipped NO weather at all — they are exactly the 13 rows missing
# ``air_density_level`` in the committed frame, and the reason the
# 2026-10-08 audit's "recent open-air misses self-heal" claim did NOT hold
# (2 of its 3 named games are still NULL in the 2026-10-09 run).

def _day_series(day: str, *, temp: float, rh: float, wind: float,
                direction: float, source: str = "open_meteo_archive") -> dict:
    """A full 24-hour GMT series for one stadium-day."""
    return {
        "time": [f"{day}T{h:02d}:00" for h in range(24)],
        "_source": source,
        "temperature_2m": [temp] * 24,
        "relative_humidity_2m": [rh] * 24,
        "wind_speed_10m": [wind] * 24,
        "wind_direction_10m": [direction] * 24,
        "surface_pressure": [1013.0] * 24,
    }


def _patch_batch(monkeypatch, series: dict):
    import weather
    monkeypatch.setattr(
        weather, "_fetch_batched_weather",
        lambda locations, start_date, end_date, needed_days=None: series)


def _game(pk: int, start: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "game_pk": pk, "home_team": "NYY", "venue": "Yankee Stadium",
        "start_time_utc": start,
    }])


def test_exact_midnight_first_pitch_reads_the_previous_days_hour(monkeypatch):
    """The 13-game defect, replayed: first pitch at 00:00:00 UTC picks the
    previous day's 23:00Z row — the hour that IS strictly prior — instead of
    shipping nothing."""
    import weather
    own = _day_series("2026-09-30", temp=30.0, rh=40.0, wind=10.0, direction=10.0)
    prev = _day_series("2026-09-29", temp=7.0, rh=80.0, wind=20.0, direction=190.0)
    _patch_batch(monkeypatch, {
        ("NYY", date(2026, 9, 30)): own,
        ("NYY", date(2026, 9, 29)): prev,
    })

    rec = weather.fetch_games_weather(_game(849848, "2026-09-30 00:00:00"))[849848]

    assert rec["available"] is True, "exact-midnight game must get weather"
    # Own day holds only 00:00..23:00 — nothing strictly before 00:00:00.
    assert rec["temp_c"] == 7.0, "must come from the previous day's 23:00Z row"
    assert rec["source"] == "open_meteo_archive"
    # Both weather interactions become computable (the coverage gap itself).
    assert not np.isnan(rec["wind_multiplier"]), rec
    assert not np.isnan(rec["air_density"]), rec


def test_other_starts_still_read_their_own_day(monkeypatch):
    """Regression guard: the previous day is consulted ONLY for an exact
    00:00:00 first pitch — every other game is byte-identical to before and
    never inherits a stale hour."""
    import weather
    own = _day_series("2026-09-30", temp=30.0, rh=40.0, wind=10.0, direction=10.0)
    prev = _day_series("2026-09-29", temp=7.0, rh=80.0, wind=20.0, direction=190.0)
    _patch_batch(monkeypatch, {
        ("NYY", date(2026, 9, 30)): own,
        ("NYY", date(2026, 9, 29)): prev,
    })

    # 23:05Z start → the own-day 23:00Z row, never the previous day's.
    rec = weather.fetch_games_weather(_game(1, "2026-09-30 23:05:00"))[1]
    assert rec["available"] is True
    assert rec["temp_c"] == 30.0, "own-day observation must win"

    # 00:05Z start → the own-day 00:00Z row (a real strictly-prior hour).
    rec = weather.fetch_games_weather(_game(2, "2026-09-30 00:05:00"))[2]
    assert rec["available"] is True
    assert rec["temp_c"] == 30.0


def test_previous_day_never_fills_a_non_midnight_or_empty_lookup(monkeypatch):
    """Two honesty pins: (a) an own-day series that is simply ABSENT is never
    papered over with a previous-day hour (that would be up to 24h stale for
    a night game); (b) no series at all stays unavailable."""
    import weather
    prev = _day_series("2026-09-29", temp=7.0, rh=80.0, wind=20.0, direction=190.0)
    _patch_batch(monkeypatch, {("NYY", date(2026, 9, 29)): prev})

    # Midnight start + own day missing → previous day is exactly right.
    ok = weather.fetch_games_weather(_game(3, "2026-09-30 00:00:00"))[3]
    assert ok["available"] is True and ok["temp_c"] == 7.0

    # Night start + own day missing → unavailable, never the stale prev day.
    stale = weather.fetch_games_weather(_game(4, "2026-09-30 23:05:00"))[4]
    assert stale["available"] is False
    assert stale["source"] == "open_meteo_unavailable"

    # Nothing fetched for either day → unavailable.
    _patch_batch(monkeypatch, {})
    none = weather.fetch_games_weather(_game(5, "2026-09-30 00:00:00"))[5]
    assert none["available"] is False


# ── T7: a partial (wind-only) cache record must not block the full one ─────
# The StatsAPI gap filler caches an official observation with wind but no
# humidity — hence no ``air_density`` — and the cache is ONE slot per
# game_pk. ``need`` excluded ANY cached pk, so that partial satisfied the
# gate forever and the complete Open-Meteo observation was never retried:
# 849851/849844/849838/813022 kept ``air_density_velocity_boost`` NULL in
# the 2026-10-09 run even though the archive had published their hours.
# master_pipeline is a RUN-ONCE script (importing it executes the phases and
# opens the committed run log with "w"), so the shipped function is executed
# from its AST instead — the same convention this file already uses for its
# source pins, but behaviorally exercised.

def _exec_weather_history(cache_path: Path) -> dict:
    tree = ast.parse(MASTER_SRC)
    want_fn = {"_attach_weather_history", "_load_weather_cache",
               "_save_weather_cache", "_weather_cache_path"}
    want_assign = {"_WEATHER_CACHE_COLS", "_OBSERVED_WEATHER_SOURCES",
                   "STATSAPI_WEATHER_FILL"}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want_fn:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in want_assign
                for t in node.targets):
            nodes.append(node)
    found = {n.name for n in nodes if isinstance(n, ast.FunctionDef)}
    assert found == want_fn, f"shipped helpers changed: {found ^ want_fn}"
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    ns = {
        "pd": pd, "np": np, "Path": Path, "date": date,
        "os": __import__("os"),
        "logger": logging.getLogger("mp_extract"),
    }
    exec(compile(module, str(BACKEND / "master_pipeline.py"), "exec"), ns)
    ns["_weather_cache_path"] = lambda: cache_path
    return ns


def _wind_only_record(pk: int) -> dict:
    """What the StatsAPI filler caches: real wind, no humidity/pressure."""
    return {
        "game_pk": pk, "available": True, "source": "statsapi_gamefeed",
        "temp_c": 20.0, "rh_pct": None, "wind_speed_kmh": 8.0,
        "wind_direction_deg": 10.0, "pressure_hpa": None,
        "air_density": None, "wind_multiplier": 0.2,
        "stadium_alt_m": 9.0, "stadium_bearing": 10.0,
    }


def _full_record(pk: int) -> dict:
    return {
        "game_pk": pk, "available": True, "source": "open_meteo_archive",
        "temp_c": 7.0, "rh_pct": 80.0, "wind_speed_kmh": 20.0,
        "wind_direction_deg": 190.0, "pressure_hpa": 1012.0,
        "air_density": 1.18, "wind_multiplier": -0.35,
        "stadium_alt_m": 9.0, "stadium_bearing": 10.0,
    }


def _games_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "game_pk": [100], "game_date": ["2026-09-30"],
        "home_team": ["NYY"], "away_team": ["BOS"],
        "venue": ["Yankee Stadium"], "home_win": [1.0],
        "start_time_utc": ["2026-09-30 00:00:00"],
        "sp_xfip_diff": [0.5], "sp_fbvelo_diff": [1.0],
        "dome_is_neutral": [0.0],
    })


def test_partial_wind_only_record_is_re_attempted_and_completed(
        tmp_path, monkeypatch, caplog):
    import results, weather
    cache_path = tmp_path / "weather_history.parquet"
    pd.DataFrame([_wind_only_record(100)]).to_parquet(cache_path, index=False)
    ns = _exec_weather_history(cache_path)

    monkeypatch.setattr(
        results, "fetch_game_start_times",
        lambda a, b: {100: datetime(2026, 9, 30, 0, 0)})
    attempted: list = []

    def _fake_fetch(df):
        attempted.extend(int(p) for p in df["game_pk"])
        return {100: _full_record(100)}

    monkeypatch.setattr(weather, "fetch_games_weather", _fake_fetch)

    with caplog.at_level(logging.INFO, logger="mp_extract"):
        ns["_attach_weather_history"](_games_frame(), date(2026, 10, 9))

    assert attempted == [100], (
        "a cached record WITHOUT an air-density observation must be "
        "re-attempted — otherwise the official wind-only fill blocks the "
        "complete one forever")
    stored = pd.read_parquet(cache_path).iloc[0]
    assert stored["source"] == "open_meteo_archive"
    assert float(stored["air_density"]) == 1.18, "full observation must replace the partial"
    assert "Weather top-up: 1/1" in caplog.text, (
        "the completion must be visible in the run log")


def test_failed_top_up_keeps_the_partial_record(tmp_path, monkeypatch, caplog):
    """Non-destruction pin: a retry that finds nothing may NEVER erase the
    official wind-only fill (only available records are written back)."""
    import results, weather
    cache_path = tmp_path / "weather_history.parquet"
    pd.DataFrame([_wind_only_record(100)]).to_parquet(cache_path, index=False)
    ns = _exec_weather_history(cache_path)

    monkeypatch.setattr(
        results, "fetch_game_start_times",
        lambda a, b: {100: datetime(2026, 9, 30, 0, 0)})
    monkeypatch.setattr(
        weather, "fetch_games_weather",
        lambda df: {100: {"available": False, "source": "open_meteo_unavailable"}})

    with caplog.at_level(logging.INFO, logger="mp_extract"):
        ns["_attach_weather_history"](_games_frame(), date(2026, 10, 9))

    stored = pd.read_parquet(cache_path).iloc[0]
    assert stored["source"] == "statsapi_gamefeed", "partial must survive a failed retry"
    assert float(stored["wind_multiplier"]) == 0.2
    assert pd.isna(stored["air_density"])
    assert "Weather top-up: 0/1" in caplog.text


def test_complete_records_are_never_flagged_for_top_up(tmp_path, monkeypatch):
    """Cost pin: a full observation must NOT be re-fetched every run — only
    genuinely partial records enter the retry set."""
    import results, weather
    cache_path = tmp_path / "weather_history.parquet"
    pd.DataFrame([_full_record(100)]).to_parquet(cache_path, index=False)
    ns = _exec_weather_history(cache_path)

    monkeypatch.setattr(
        results, "fetch_game_start_times",
        lambda a, b: (_ for _ in ()).throw(AssertionError("no fetch expected")))
    monkeypatch.setattr(
        weather, "fetch_games_weather",
        lambda df: (_ for _ in ()).throw(AssertionError("no fetch expected")))

    out = ns["_attach_weather_history"](_games_frame(), date(2026, 10, 9))
    assert not out.empty
    assert pd.read_parquet(cache_path).iloc[0]["source"] == "open_meteo_archive"


# ── T8: the official "Varies" wind report stays an honest refusal ─────────
# Four of the eight open-air gap games (813023, 813032, 849823, 849848) are
# reported by the park as e.g. "5 mph, Varies": a real observation with NO
# fixed direction, so no wind multiplier can be computed — and the feed has
# no humidity either, so no air density. The record is refused (never a
# guessed direction), which is why those four are covered by T6/T7's
# archive retry rather than by the filler.

def test_official_varies_wind_is_refused_not_guessed():
    import weather
    parsed = {"temp_f": 68.0, "wind_mph": 5.0,
              "wind_text": "5 mph, Varies", "condition": "Cloudy"}
    rec = weather.statsapi_weather_to_record(parsed, "NYY", "Yankee Stadium")
    assert rec["available"] is False
    assert rec["source"] == "statsapi_gamefeed_unusable"
    assert rec["wind_multiplier"] is None and rec["air_density"] is None

    # A directional report still fills wind (air density honestly stays NULL
    # — the official feed carries no humidity, see T7).
    directional = {"temp_f": 84.0, "wind_mph": 5.0,
                   "wind_text": "5 mph, L To R", "condition": "Clear"}
    ok = weather.statsapi_weather_to_record(directional, "NYY", "Yankee Stadium")
    assert ok["available"] is True
    assert ok["wind_multiplier"] is not None
    assert ok["air_density"] is None


# ── T9: the coverage warning must explain closed-roof policy zeros ─────────
# The two weather features can never reach the 80% measured OK line on a
# roof-heavy window: ~19% of games are closed-roof rows whose value is a
# POLICY zero (correctly never counted as an observation). After T6/T7 the
# residual gap on those rows is defaults + staleness-gated SP inputs — so
# the line now prints the open-air-only ratio too. Status/threshold and the
# CSV schema were deliberately UNCHANGED at T9 time (the frontend tooltips
# document "LOW_COVERAGE <80% measured"); this is additive observability
# only. SUPERSEDED IN PART by T11 below (2026-10-09 coverage audit): a row
# whose every unmeasured value is declared policy now classifies
# STRUCTURAL instead — these T9 rows carry no SP input column to prove the
# gate, so they keep their alarm exactly as asserted here.

def test_coverage_warning_reports_the_open_air_ratio(tmp_path, monkeypatch,
                                                      caplog):
    import explainability
    monkeypatch.setattr(explainability, "DATA_DELIVERY_DIR", tmp_path)
    frame = pd.DataFrame({
        # 3 closed-roof policy zeros + 5 open-air observations + 2 open-air
        # rows with no observation (or a gated SP input).
        "wind_advantage_flyball_factor": [0.0, 0.0, 0.0, 0.5, 0.5,
                                          0.5, 0.5, 0.5, np.nan, np.nan],
        "dome_is_neutral_game": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # A low-coverage feature with NO policy zeros: no open-air clause.
        "home_elo": [1.0] * 6 + [np.nan] * 4,
    })

    with caplog.at_level(logging.WARNING, logger="explainability"):
        cov = explainability.compute_feature_coverage(
            frame, frame, "20990103",
            feature_cols=["wind_advantage_flyball_factor", "home_elo"])

    row = cov[(cov.feature == "wind_advantage_flyball_factor")
              & (cov.window == "current")].iloc[0]
    # Status semantics are untouched — the alert still fires at 50% measured.
    assert row.status == "LOW_COVERAGE" and row.pct_measured == 50.0
    assert int(row.n_default_zero) == 3 and int(row.n_measured) == 5

    text = caplog.text
    assert "wind_advantage_flyball_factor/current=50% measured" in text
    assert "open-air 71%" in text, "5 of 7 open-air rows observed"
    assert "3 closed-roof policy zeros excluded" in text
    # No defaults → no clause for that feature (exactly two occurrences:
    # one per window of the weather feature, none for home_elo).
    assert "home_elo/current=60% measured" in text
    assert text.count("open-air") == 2
    assert text.count("closed-roof policy zeros excluded") == 2
    line = next(ln for ln in text.splitlines()
                if "Feature coverage gaps" in ln)
    home_segs = [s for s in line.split("; ") if s.startswith("home_elo/")]
    assert len(home_segs) == 2 and all("open-air" not in s for s in home_segs), (
        "only policy-zero features earn the open-air clause")


# ── T11: the standing weather alarm → STRUCTURAL, only when FULLY explained ─
# 2026-10-09 coverage audit: after T6/T7 the shipped frame has ZERO
# weather-observation gaps (park_wind_factor/air_density_level are NULL on
# 0 rows; every open-air NULL is an SP-staleness-gated input), yet the two
# weather features stood LOW_COVERAGE 73%/75% on EVERY run because the 80%
# line counts closed-roof policy zeros and policy NULLs in its denominator.
# A permanent alarm trains viewers to ignore it — the exact failure mode
# the backstop exists to prevent. Remediation (cross-sport contract: NBA
# 2026-10-08, NFL 2026-09-29; the frontend already renders STRUCTURAL):
# classify STRUCTURAL WITH a reason only when every unmeasured row carries
# a declared-policy explanation. A weather OUTAGE has the opposite
# signature — an open-air NULL beside a PRESENT SP input — and keeps the
# raw LOW_COVERAGE/STARVED thresholds, so starvation sensitivity is
# unchanged. Numbers never change; only what the status MEANS.

def _explained_frame() -> pd.DataFrame:
    """10 rows: 3 closed-roof policy zeros, 5 open-air observations,
    2 open-air NULLs whose SP inputs are gated (staleness/debut)."""
    return pd.DataFrame({
        "wind_advantage_flyball_factor":
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, np.nan, np.nan],
        "air_density_velocity_boost":
            [0.0, 0.0, 0.0, -0.1, -0.1, -0.1, -0.1, -0.1, np.nan, np.nan],
        "dome_is_neutral_game": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0],
        "sp_xfip_diff": [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1,
                        np.nan, np.nan],
        "sp_fbvelo_diff": [0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4,
                           np.nan, np.nan],
    })


def test_fully_explained_weather_window_is_structural_and_never_pages(
        tmp_path, monkeypatch, caplog):
    import explainability
    monkeypatch.setattr(explainability, "DATA_DELIVERY_DIR", tmp_path)

    with caplog.at_level(logging.INFO, logger="explainability"):
        cov = explainability.compute_feature_coverage(
            _explained_frame(), _explained_frame(), "20990111",
            feature_cols=["wind_advantage_flyball_factor",
                          "air_density_velocity_boost"])

    for col, input_col in (("wind_advantage_flyball_factor", "sp_xfip_diff"),
                           ("air_density_velocity_boost", "sp_fbvelo_diff")):
        for window in ("current", "baseline"):
            row = cov[(cov.feature == col) & (cov.window == window)].iloc[0]
            assert row.status == "STRUCTURAL", (col, window, row.status)
            # The numbers are untouched — STRUCTURAL changes what the
            # status means, never the measured counts.
            assert row.pct_measured == 50.0
            assert int(row.n_default_zero) == 3 and int(row.n_measured) == 5
            assert int(row.n_nonnull) == 8
            reason = str(row.structural_reason)
            assert "3 closed-roof policy zero" in reason
            assert f"2 open-air NULL(s) gated by a missing {input_col}" in reason

    text = caplog.text
    assert "Feature coverage gaps" not in text, (
        "fully-explained policy rows must never page")
    assert "STRUCTURAL (declared missing-value policy" in text
    assert "SP staleness gate" in text

    # The reason rides along in the CSV (additive column) and in the rows
    # the monitor JSON serializes.
    out = pd.read_csv(tmp_path / "feature_coverage_20990111.csv")
    assert "structural_reason" in out.columns
    srow = out[(out.feature == "wind_advantage_flyball_factor")
               & (out.window == "current")].iloc[0]
    assert srow.status == "STRUCTURAL" and "sp_xfip_diff" in str(srow.structural_reason)


def test_an_unobserved_open_air_row_keeps_the_raw_alarm(tmp_path, monkeypatch,
                                                         caplog):
    """The starvation signature: an open-air row with a PRESENT SP input
    and no observation cannot be policy — the row is the 2026-10-07
    truncation class and must keep LOW_COVERAGE (and the WARNING)."""
    import explainability
    monkeypatch.setattr(explainability, "DATA_DELIVERY_DIR", tmp_path)
    frame = _explained_frame()
    # Row 9: inputs present, weather missing → unobserved, not gated.
    frame.loc[9, "sp_xfip_diff"] = 0.1
    frame.loc[9, "sp_fbvelo_diff"] = 0.2

    with caplog.at_level(logging.WARNING, logger="explainability"):
        cov = explainability.compute_feature_coverage(
            frame, frame, "20990112",
            feature_cols=["wind_advantage_flyball_factor",
                          "air_density_velocity_boost"])

    row = cov[(cov.feature == "wind_advantage_flyball_factor")
              & (cov.window == "current")].iloc[0]
    assert row.status == "LOW_COVERAGE", "one unexplained row kills STRUCTURAL"
    assert pd.isna(row.structural_reason), "no reason may dangle beside an alarm"
    assert "Feature coverage gaps" in caplog.text
    assert "wind_advantage_flyball_factor/current=50% measured" in caplog.text


def test_structural_is_per_feature_and_gated_on_its_declared_input(
        tmp_path, monkeypatch):
    """wind is governed by sp_xfip_diff, air by sp_fbvelo_diff: the same
    two NULL rows can be policy for one feature and an outage for the other."""
    import explainability
    monkeypatch.setattr(explainability, "DATA_DELIVERY_DIR", tmp_path)
    frame = _explained_frame()
    # Inputs missing for era only: wind NULLs are gated (explained),
    # air NULLs sit beside a present input (unobserved → alarm).
    frame["sp_fbvelo_diff"] = [0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4,
                               0.4, 0.4]

    cov = explainability.compute_feature_coverage(
        frame, frame, "20990113",
        feature_cols=["wind_advantage_flyball_factor",
                      "air_density_velocity_boost"])
    wind = cov[(cov.feature == "wind_advantage_flyball_factor")
               & (cov.window == "current")].iloc[0]
    air = cov[(cov.feature == "air_density_velocity_boost")
              & (cov.window == "current")].iloc[0]
    assert wind.status == "STRUCTURAL"
    assert "missing sp_xfip_diff" in str(wind.structural_reason)
    assert air.status == "LOW_COVERAGE"
    assert pd.isna(air.structural_reason)


def test_a_broken_indoor_fill_stays_unexplained(tmp_path, monkeypatch):
    """A closed-roof row whose value is NULL means apply_indoor_neutral_
    fills did NOT run — a build hole, never declared policy (policy writes
    the zero unconditionally). It keeps the alarm."""
    import explainability
    monkeypatch.setattr(explainability, "DATA_DELIVERY_DIR", tmp_path)
    frame = _explained_frame()
    frame.loc[0, "wind_advantage_flyball_factor"] = np.nan  # dome row, no zero

    cov = explainability.compute_feature_coverage(
        frame, frame, "20990114",
        feature_cols=["wind_advantage_flyball_factor",
                      "air_density_velocity_boost"])
    wind = cov[(cov.feature == "wind_advantage_flyball_factor")
               & (cov.window == "current")].iloc[0]
    air = cov[(cov.feature == "air_density_velocity_boost")
              & (cov.window == "current")].iloc[0]
    assert wind.status == "LOW_COVERAGE"  # the broken fill alarms
    assert air.status == "STRUCTURAL"     # air's policy set is complete


def test_missing_governing_input_column_fails_open(tmp_path, monkeypatch):
    """Without the SP input column there is no proof of the gate — the
    threshold alarm stays (fail-open, never fail-silent). This is also
    what keeps T9's input-less mini frame paging as before."""
    import explainability
    monkeypatch.setattr(explainability, "DATA_DELIVERY_DIR", tmp_path)
    frame = _explained_frame().drop(columns=["sp_xfip_diff", "sp_fbvelo_diff"])

    cov = explainability.compute_feature_coverage(
        frame, frame, "20990115",
        feature_cols=["wind_advantage_flyball_factor"])
    wind = cov[(cov.feature == "wind_advantage_flyball_factor")
               & (cov.window == "current")].iloc[0]
    assert wind.status == "LOW_COVERAGE"
    assert pd.isna(wind.structural_reason)


def test_run_engine_coverage_shares_the_structural_rule(tmp_path, monkeypatch):
    """The run-engine wrapper forwards the SAME windows through the SAME
    function (single-list rule) — its CSV must carry the same status and
    reason, not a separate classification."""
    import explainability
    monkeypatch.setattr(explainability, "DATA_DELIVERY_DIR", tmp_path)

    explainability.compute_run_engine_feature_coverage(
        _explained_frame(), _explained_frame(), "20990116")

    out = pd.read_csv(tmp_path / "run_engine_feature_coverage_20990116.csv")
    w = out[(out.feature == "wind_advantage_flyball_factor")
            & (out.window == "current")].iloc[0]
    assert w.status == "STRUCTURAL"
    assert "declared missing-value policy" in str(w.structural_reason)
    assert "structural_reason" in out.columns


# ── T12: the Kaggle notebook's repo path + post-run sanity footer ──────────
# 2026-10-09 third delivery (artifacts 129c3fd3, log f4ce14d9): the run
# itself was CLEAN — the first to ship the T11 STRUCTURAL statuses — but
# the notebook session ended red AFTER delivery:
#   FileNotFoundError: No such file or directory:
#       '/kaggle/working/sports_predictio_model'
# in the confirmation block's ``subprocess.run(..., cwd=repo)``. The live
# Kaggle copy's confirmation block re-typed the ``repo`` literal a SECOND
# time and typo'd it; every committed notebook version (through Kaggle
# Version 5, 89bd0879) is correctly spelled, and the pushed run log is
# clean — delivery was unaffected (pipeline exit 0; Phase 5 pushed and
# remotely verified 25 files). Standing guardrail (T7,
# test_log_review_20261006): the notebook is Kaggle-owned and is never
# edited from the repo, so the defense is a content pin — a Kaggle
# re-upload carrying the defect must fail this suite.

_MLB_NB = BACKEND.parent.parent / "kaggle_mlb_run.ipynb"
_CANONICAL_REPO = "/kaggle/working/sports_prediction_model"


def _mlb_notebook_source() -> str:
    import json
    nb = json.loads(_MLB_NB.read_text(encoding="utf-8"))
    parts = []
    for cell in nb["cells"]:
        src = cell["source"]
        parts.append("".join(src) if isinstance(src, list) else src)
    return "\n".join(parts)


def test_every_notebook_repo_literal_is_canonical():
    """The 2026-10-09 incident: the confirmation block DUPLICATES the
    ``repo`` literal instead of reusing the pipeline block's value, and a
    typo in that duplicate (sports_predictio_model) crashed the cell
    AFTER a successful delivery. Every ``repo = "..."`` literal in the
    notebook must be the canonical clone path — one drift anywhere and
    the sanity footer dies on a directory that does not exist."""
    import re
    literals = re.findall(r'repo = "([^"]*)"', _mlb_notebook_source())
    assert literals, "the notebook must keep a repo path for its sanity footer"
    assert set(literals) == {_CANONICAL_REPO}, (
        "every repo literal in the Kaggle notebook must be the canonical "
        f"clone path; found {literals}")


def test_notebook_has_no_typo_class_repo_path():
    """The exact defect string from the 19:08 session, pinned: any
    'sports_predictio' that is not followed by 'n_model' is this typo."""
    import re
    bad = re.findall(r"sports_predictio(?!n_model)", _mlb_notebook_source())
    assert not bad, (
        "typo'd sports_prediction_model path in the Kaggle notebook — "
        "the 2026-10-09 FileNotFoundError class (fix on the Kaggle side)")


def test_notebook_keeps_the_post_run_sanity_footer():
    """The footer the crash silenced must stay: HEAD after the run + the
    newest dated artifacts on origin/main, plus the typo-safe runner."""
    src = _mlb_notebook_source()
    assert "Repo HEAD after run:" in src
    assert "Latest dated artifacts:" in src
    assert '["git", "log", "--oneline", "-1"]' in src


# ── T13: the RUN repairs the notebook's typo'd path — the notebook itself
# is never changed (guardrail: no change to the MLB Kaggle run program).
# master_pipeline executes in the same kernel session BEFORE the notebook's
# confirmation block, so at the end of delivery it may repair the
# ENVIRONMENT: github_sync.ensure_notebook_sanity_alias points the typo'd
# location at the real clone through a directory symlink, and the
# notebook's advisory git checks (git log / git ls-tree with cwd=repo) run
# green against the ACTUAL repository. Advisory by construction: existing
# paths untouched, missing clones never fabricated, never fatal.

def test_sanity_alias_points_the_typo_path_at_the_real_clone(tmp_path, monkeypatch):
    """Positive path: missing alias + real clone -> alias created and the
    notebook's cwd=alias git checks resolve into the real repo. The
    symlink call is patched because this test host may lack symlink
    privilege (Windows); a real-symlink case runs where the host allows.
    """
    import github_sync
    clone = tmp_path / "sports_prediction_model"
    (clone / ".git").mkdir(parents=True)
    alias = tmp_path / "sports_predictio_model"

    def _fake_symlink(self, target, target_is_directory=False):
        assert Path(target) == clone.resolve()
        assert target_is_directory is True
        self.mkdir()

    monkeypatch.setattr(type(alias), "symlink_to", _fake_symlink)
    assert github_sync.ensure_notebook_sanity_alias(clone, alias) is True
    assert alias.is_dir(), "the typo'd path must resolve as a directory"
    # Idempotent: a second run never touches the existing path.
    assert github_sync.ensure_notebook_sanity_alias(clone, alias) is False


def test_sanity_alias_real_symlink_where_the_host_allows(tmp_path):
    """The production mechanism itself (Kaggle is Linux, where this always
    works): a real directory symlink that git could resolve."""
    import github_sync
    clone = tmp_path / "c"
    (clone / ".git").mkdir(parents=True)
    alias = tmp_path / "a"
    created = github_sync.ensure_notebook_sanity_alias(clone, alias)
    if not created:
        pytest.skip("host refuses symlinks (Windows privilege) — patched "
                    "positive case covers the logic")
    assert alias.is_dir() and (alias / ".git").is_dir()


def test_sanity_alias_never_touches_an_existing_path(tmp_path):
    from github_sync import ensure_notebook_sanity_alias
    clone = tmp_path / "c"
    (clone / ".git").mkdir(parents=True)
    alias = tmp_path / "a"
    alias.mkdir()
    (alias / "precious.txt").write_text("keep", encoding="utf-8")
    assert ensure_notebook_sanity_alias(clone, alias) is False
    assert (alias / "precious.txt").read_text(encoding="utf-8") == "keep"


def test_sanity_alias_never_fabricates_a_missing_clone(tmp_path):
    from github_sync import ensure_notebook_sanity_alias
    clone = tmp_path / "c"
    clone.mkdir()  # no .git — not a repo clone
    alias = tmp_path / "a"
    assert ensure_notebook_sanity_alias(clone, alias) is False
    assert not alias.exists() and not alias.is_symlink()


def test_sanity_alias_is_never_fatal(tmp_path, monkeypatch):
    """No symlink privilege / read-only volume / any failure: return False,
    raise nothing — the step can never fail a delivered run."""
    import github_sync
    clone = tmp_path / "c"
    (clone / ".git").mkdir(parents=True)
    alias = tmp_path / "a"

    def _boom(*_a, **_k):
        raise OSError("no symlink here")

    monkeypatch.setattr(type(alias), "symlink_to", _boom)
    assert github_sync.ensure_notebook_sanity_alias(clone, alias) is False
    assert not alias.exists()


def test_master_pipeline_owns_the_notebook_typo_repair():
    """Source pin: the repair lives in the RUN (not the notebook), runs
    after the final log delivery and before the honest-exit block, names
    the observed typo path, and is wrapped so it can never fail the run.
    """
    tail = MASTER_SRC[MASTER_SRC.index("Final run-log delivery"):]
    j = tail.index("ensure_notebook_sanity_alias")
    end = tail.index("_phase4_error is not None")
    assert j < end, "the alias step must run before the honest-exit block"
    block = tail[:end]
    assert 'Path("/kaggle/working")' in block
    assert "sports_predictio_model" in block
    assert "try:" in block and "except Exception" in block, (
        "the alias step must be non-fatal")
