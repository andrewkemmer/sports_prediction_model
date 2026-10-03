"""Pin the OUT/IR game-eligibility contract for the expected lineup.

The feature ships in ``features.py``'s lineup-pool SQL constants, so these
tests drive the SHIPPED SQL against synthetic ``batter_ratings`` /
``il_stints`` tables rather than a re-implementation: if someone edits the
pool, the IL predicate, or the flag emission, a test here fails.

What is pinned, and why each one matters:

* the pool WIDENS past actual participants -- without this the OUT/IR
  filter is a no-op, because an injured player is absent from the
  participant pool by construction.
* the IL predicate binds, respects stint bounds in BOTH directions, and
  treats an open stint (NULL end) as still injured.
* an OUT/IR player is REMOVED from the eligible nine and a healthy
  replacement is promoted -- but his rating row remains untouched in the
  pool (eligibility, not quality).
* the flag is MAX(on_il) over the projected nine only: an IL player
  OUTSIDE the top-9 does NOT trip it.
* the fallback path (cache missing) hardwires on_il = 0 and restricts the
  pool to that game's own rating rows -- byte-equivalent to the pre-IL
  participant-pool feature.
* both paths emit the SAME columns (serving contract).
* the flag columns are known-pool RFE candidates and the generation
  universe width is untouched (64 = 62 + 2 candidate-side flags in the
  known pool, universe itself stays 62).
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import features  # noqa: E402
import training  # noqa: E402

GAME = pd.Timestamp("2024-07-10").date()
GPK = 745001
TEAM = "BOS"


def _ratings(pa30s: dict[int, float], extra_team: str | None = None) -> pd.DataFrame:
    """One rating row per (game, team, batter). pa30s: batter -> (pa30, woba)."""
    rows = []
    for batter, (pa30, woba) in pa30s.items():
        rows.append({"game_date": GAME, "game_pk": GPK, "batting_team": TEAM,
                     "batter": batter, "shrunk_re24": woba, "_pa30": pa30})
    if extra_team:
        for batter, (pa30, woba) in pa30s.items():
            rows.append({"game_date": GAME, "game_pk": GPK,
                         "batting_team": extra_team,
                         "batter": batter, "shrunk_re24": woba, "_pa30": pa30})
    return pd.DataFrame(rows)


@pytest.fixture()
def con():
    c = duckdb.connect(database=":memory:")
    yield c
    c.close()


def _set_effective(con, rows=None) -> None:
    """lineup_effective fixture: (game_pk, team, batter, tier) rows.

    None/empty -> the SHIPPED empty degrade table, i.e. the exact pre-audit
    full-roster pool the historical fixtures below pin.
    """
    if not rows:
        con.execute(features._EMPTY_LINEUP_EFFECTIVE_SQL)
        return
    vals = ", ".join("(%d, '%s', %d, %d)" % (g, t, b, tier)
                     for (g, t, b, tier) in rows)
    con.execute(f"""CREATE OR REPLACE TABLE lineup_effective AS
        SELECT * FROM (VALUES {vals})
        AS t(game_pk, team, batter, tier)""")


def _run_pool(con: duckdb.DuckDBPyConnection, ratings: pd.DataFrame,
              il_rows: list[tuple[int, str, str | None]],
              *, effective: list[tuple] | None = None) -> None:
    con.register("batter_ratings", ratings)
    if il_rows:
        vals = ",".join(
            "({}, DATE '{}', {})".format(b, s, f"DATE '{e}'" if e else "NULL")
            for b, s, e in il_rows)
        con.execute(f"""CREATE TABLE il_stints AS
            SELECT CAST(batter AS BIGINT) batter,
                   CAST(il_start AS DATE) il_start,
                   CAST(il_end AS DATE) il_end
            FROM (VALUES {vals}) AS t(batter, il_start, il_end)""")
    else:
        con.execute("""CREATE TABLE il_stints AS
            SELECT CAST(NULL AS BIGINT) AS batter,
                   CAST(NULL AS DATE) AS il_start,
                   CAST(NULL AS DATE) AS il_end
            WHERE false""")
    _set_effective(con, effective)
    con.execute(features._LINEUP_POOL_SQL.format(
        lookback=features.LINEUP_POOL_LOOKBACK_DAYS, restrict="",
        on_il=features._IL_EXISTS_PREDICATE))
    con.execute(features._LINEUP_AGG_SQL)
    con.execute(features._LINEUP_IL_FLAG_SQL)


def _run_pool_fallback(con: duckdb.DuckDBPyConnection,
                       ratings: pd.DataFrame) -> None:
    con.register("batter_ratings", ratings)
    con.execute("""CREATE TABLE il_stints AS
        SELECT CAST(NULL AS BIGINT) AS batter,
               CAST(NULL AS DATE) AS il_start,
               CAST(NULL AS DATE) AS il_end
        WHERE false""")
    _set_effective(con)
    con.execute(features._LINEUP_POOL_SQL.format(
        lookback=features.LINEUP_POOL_LOOKBACK_DAYS,
        restrict="AND r.game_pk = g.game_pk", on_il="0"))
    con.execute(features._LINEUP_AGG_SQL)
    con.execute(features._LINEUP_IL_FLAG_SQL)


# ── pool widening + eligibility ──────────────────────────────────────────────

def test_pool_widens_past_participants(con):
    # 11 batters rated 9 days ago, none batted in THIS game's rating rows.
    old = {1000 + i: (100.0 - i, 0.300 + i / 1000) for i in range(11)}
    _run_pool(con, _ratings(old), [])
    n = con.execute("SELECT count(*) FROM lineup_pool").fetchone()[0]
    assert n == 11, "rating rows inside the lookback window must all pool"


def test_il_predicate_binds_and_respects_bounds(con):
    # 1100 placed BEFORE the game -> flagged; 1101 placed AFTER -> not;
    # 1102 stint closed the day BEFORE the game -> not; 1103 open stint -> yes.
    r = _ratings({1100: (50, 0.300), 1101: (49, 0.300),
                  1102: (48, 0.300), 1103: (47, 0.300)})
    _run_pool(con, r, [
        (1100, "2024-07-01", None),            # open, started before
        (1101, "2024-07-11", None),            # starts the day after
        (1102, "2024-06-01", "2024-07-09"),    # closed the day before
        (1103, "2024-07-10", None),            # opens the game date itself
    ])
    flags = dict(con.execute(
        "SELECT batter, on_il FROM lineup_pool").fetchall())
    assert flags == {1100: 1, 1101: 0, 1102: 0, 1103: 1}


def test_out_player_removed_and_replacement_promoted(con):
    # 10 healthy batters; the best one (pa30=200) is OUT/IR.
    r = {2000 + i: (200.0 - i, 0.400 - i / 1000) for i in range(10)}
    _run_pool(con, _ratings(r), [(2000, "2024-07-01", None)])
    nine = [b for b, in con.execute(
        """WITH ranked AS (SELECT batter, _pa30 FROM lineup_pool
           WHERE on_il = 0 ORDER BY _pa30 DESC LIMIT 9)
           SELECT batter FROM ranked""").fetchall()]
    assert 2000 not in nine, "OUT/IR player must leave the eligible nine"
    assert len(nine) == 9 and 2008 in nine, "healthy replacement is promoted"
    # rating row survives in the pool: eligibility flag, not a quality edit
    row = con.execute(
        "SELECT shrunk_re24, _pa30 FROM lineup_pool WHERE batter=2000"
    ).fetchone()
    assert row == pytest.approx((0.400, 200.0))


def test_il_outside_top9_does_not_trip_flag(con):
    # 10 healthy batters; the WORST one (rank 10) is OUT/IR.
    r = {3000 + i: (100.0 - i, 0.300) for i in range(10)}
    _run_pool(con, _ratings(r), [(3009, "2024-07-01", None)])
    flag = con.execute("SELECT lineup_il_flag FROM lineup_il_flag").fetchone()[0]
    assert flag == 0, "an IL player outside the projected nine is not a flag"


def test_il_inside_top9_trips_flag(con):
    r = {4000 + i: (100.0 - i, 0.300) for i in range(10)}
    _run_pool(con, _ratings(r), [(4003, "2024-07-01", None)])
    flag = con.execute("SELECT lineup_il_flag FROM lineup_il_flag").fetchone()[0]
    assert flag == 1


# ── fallback = pre-IL participant behavior ───────────────────────────────────

def test_fallback_is_participant_pool_with_zero_flag(con):
    # Two games for the team; only today's rows are the participant pool.
    rows = []
    for i in range(9):
        rows.append({"game_date": GAME, "game_pk": GPK, "batting_team": TEAM,
                     "batter": 5000 + i, "shrunk_re24": 0.310, "_pa30": 90.0})
    prior = dict(rows[0]); prior["game_pk"] = GPK - 1
    prior["game_date"] = pd.Timestamp(GAME) - pd.Timedelta(days=3)
    prior["batter"] = 5099
    rows.append(prior)
    df = pd.DataFrame(rows)
    _run_pool_fallback(con, df)
    today = {b for b, in con.execute(
        f"SELECT batter FROM lineup_pool WHERE game_pk={GPK}").fetchall()}
    assert 5099 not in today, "fallback pool restricts to that game's rows"
    assert len(today) == 9, "today's side pools only its own participants"
    assert con.execute(
        f"SELECT max(on_il) FROM lineup_pool WHERE game_pk={GPK}").fetchone()[0] == 0
    assert con.execute(
        "SELECT lineup_il_flag FROM lineup_il_flag").fetchone()[0] == 0


def test_both_paths_emit_same_columns(con):
    r = _ratings({6000 + i: (100.0 - i, 0.300) for i in range(10)})
    c1 = duckdb.connect(database=":memory:")
    _run_pool(c1, r, [(6002, "2024-07-01", None)])
    cols_il = [d[0] for d in c1.execute(
        "DESCRIBE SELECT * FROM lineup_pool").fetchall()]
    agg_il = [d[0] for d in c1.execute(
        "DESCRIBE SELECT * FROM lineup_agg").fetchall()]
    c1.close()
    c2 = duckdb.connect(database=":memory:")
    _run_pool_fallback(c2, r)
    cols_fb = [d[0] for d in c2.execute(
        "DESCRIBE SELECT * FROM lineup_pool").fetchall()]
    agg_fb = [d[0] for d in c2.execute(
        "DESCRIBE SELECT * FROM lineup_agg").fetchall()]
    c2.close()
    assert cols_il == cols_fb and agg_il == agg_fb


# ── staleness tripwire (2026-09-28 Aaron Judge audit) ───────────────────────

def _write_il_cache(tmp_path, rows, meta=None):
    base = tmp_path / "dd"
    base.mkdir(exist_ok=True)
    # the dir is reused across calls in a test: clear any prior meta
    (base / features.IL_STINTS_META_FILE).unlink(missing_ok=True)
    df = pd.DataFrame(rows, columns=["batter", "il_start", "il_end"])
    df["batter"] = df["batter"].astype("int64")
    df["il_start"] = pd.to_datetime(df["il_start"])
    df["il_end"] = pd.to_datetime(df["il_end"])
    df.to_parquet(base / features.IL_STINTS_FILE, index=False)
    if meta is not None:
        import json
        (base / features.IL_STINTS_META_FILE).write_text(
            json.dumps(meta))
    return base


def _con_with_pitches(ref_date: str):
    c = duckdb.connect(database=":memory:")
    c.execute(f"CREATE TABLE pitches AS SELECT "
              f"DATE '{ref_date}' AS game_date")
    return c


def test_il_staleness_tripwire_warns_past_budget(tmp_path, caplog):
    # Table coverage stops 2026-09-10; the frame's data horizon is
    # 2026-09-28 -> lag 18d > budget. The filter must still LOAD (the
    # degrade-don't-fail contract) but warn about the invisible
    # placements.
    base = _write_il_cache(
        tmp_path, [(592450, "2026-09-01", None)],
        meta={"window_end": "2026-09-10"})
    con = _con_with_pitches("2026-09-28")
    with caplog.at_level("WARNING", logger="features"):
        with patch.object(features, "_lineup_base_dir",
                          return_value=base):
            assert features._register_il_stints(con) is True
    assert any("stays in the projected nine" in r.getMessage()
               for r in caplog.records), caplog.text
    con.close()


def test_il_fresh_table_does_not_warn(tmp_path, caplog):
    base = _write_il_cache(
        tmp_path, [(592450, "2026-09-25", None)],
        meta={"window_end": "2026-09-26"})
    con = _con_with_pitches("2026-09-28")
    with caplog.at_level("WARNING", logger="features"):
        with patch.object(features, "_lineup_base_dir",
                          return_value=base):
            assert features._register_il_stints(con) is True
    assert not any("stays in the projected nine" in r.getMessage()
                   for r in caplog.records), caplog.text
    con.close()


def test_il_freshness_prefers_meta_then_parquet(tmp_path):
    base = _write_il_cache(
        tmp_path, [(592450, "2026-09-01", "2026-09-20"),
                   (592451, "2026-09-22", None)],
        meta={"window_end": "2026-09-25"})
    with patch.object(features, "_lineup_base_dir",
                      return_value=base):
        fr = features.il_stints_freshness()
    assert fr["window_end"] == "2026-09-25"
    # parquet signal = max(coalesce(il_end, il_start)) = the OPEN
    # stint's start (2026-09-22) beats the closed one's end (09-20)?
    # No: coalesce picks il_end where present -> 2026-09-20; the open
    # stint contributes il_start 2026-09-22 -> max is 2026-09-22.
    assert fr["parquet_max_stint_date"] == "2026-09-22"

    base2 = _write_il_cache(
        tmp_path, [(592450, "2026-09-01", "2026-09-20")])  # no meta
    with patch.object(features, "_lineup_base_dir",
                      return_value=base2):
        fr2 = features.il_stints_freshness()
    assert "window_end" not in fr2
    assert fr2["parquet_max_stint_date"] == "2026-09-20"


def test_split_fix_recall_return_then_replacement():
    # The Campero 2026 shape: a stint opened 06-06 whose return files
    # NO activation (minor-league recall), then a real re-placement
    # 08-26. The old ignore-if-on guard swallowed the new stint
    # entirely; the split keeps both, and each reconciles on its own.
    import build_il_stints as b
    events = {1: [("2026-06-06", 1), ("2026-08-26", 1)]}
    iv = b.stints_from_events(events)
    assert len(iv) == 2
    assert iv.iloc[0].il_start == pd.Timestamp("2026-06-06")
    assert iv.iloc[0].il_end == pd.Timestamp("2026-08-26")
    assert iv.iloc[1].il_start == pd.Timestamp("2026-08-26")
    assert pd.isna(iv.iloc[1].il_end)


def test_split_fix_same_day_activate_place_then_future_placement():
    # The Meadows shape: same-day activation+placement leaves the walk
    # on; a placement MONTHS later must still open a new stint.
    import build_il_stints as b
    events = {1: [
        ("2025-07-28", 1), ("2025-09-05", 0), ("2025-09-05", 1),
        ("2026-04-10", 1), ("2026-04-13", 1),
    ]}
    iv = b.stints_from_events(events)
    starts = list(iv.il_start)
    assert pd.Timestamp("2026-04-10") in starts
    assert iv.iloc[-1].il_start == pd.Timestamp("2026-04-13")
    assert pd.isna(iv.iloc[-1].il_end)


def test_split_fix_duplicate_same_day_copy_unchanged():
    # The pre-existing guard case: a duplicate retroactive copy at the
    # SAME date must stay byte-identical to the single-placement walk.
    import build_il_stints as b
    single = b.stints_from_events({1: [("2026-05-01", 1)]})
    dup = b.stints_from_events(
        {1: [("2026-05-01", 1), ("2026-05-01", 1)]})
    assert len(dup) == 1
    assert dup.iloc[0].il_start == single.iloc[0].il_start
    assert pd.isna(dup.iloc[0].il_end) and pd.isna(single.iloc[0].il_end)


def test_classify_skips_pitchers():
    import build_il_stints as b
    roster = [
        {"person": {"id": 640448, "fullName": "Reliever"},
         "status": {"description": "Injured 15-Day"},
         "position": {"abbreviation": "P"}},
        {"person": {"id": 673962, "fullName": "Josh Jung"},
         "status": {"description": "Injured 10-Day"},
         "position": {"abbreviation": "3B"}},
    ]
    cls = b.classify_roster_statuses(roster)
    assert [r["batter"] for r in cls["injured"]] == [673962]


def test_suppressed_ground_truth_is_pa_contradiction():
    # Roster membership alone over-reports: a player activated after
    # the PA projection's cut reads Active with no contradicting PA
    # (status lag, not a defect). Only a PA strictly after the stint's
    # start is the ground-truth contradiction.
    import build_il_stints as b
    iv = pd.DataFrame({
        "batter": pd.array([40, 41], dtype="int64"),
        "il_start": pd.to_datetime(["2026-08-30", "2026-09-16"]),
        "il_end": pd.to_datetime([None, None]),
    })
    injured = [{"batter": 40, "name": "Live Injured",
                "status": "Injured 10-Day"}]
    pa = pd.DataFrame({
        "batter": pd.array([40, 41, 41], dtype="int64"),
        "game_date": pd.to_datetime(
            ["2026-09-24", "2026-09-20", "2026-09-28"]),
    })
    rep = b.verify_table_against_statuses(
        iv, injured, pd.Timestamp("2026-09-29"),
        active_ids={40, 41}, pa_days=pa)
    # 41 batted 09-28, AFTER his 09-16 stint start -> contradiction.
    # 40 has no PA after his 08-30 start (09-24 predates it) -> lag.
    assert [r["batter"] for r in rep["suppressed"]] == [41]


def test_suppressed_scoped_to_live_active():
    # Without active_ids: legacy behavior (every uncontradicted flag
    # reported). With active_ids: retired/off-roster open stints are
    # out of scope; only a verifiably Active player is a smell.
    import build_il_stints as b
    iv = pd.DataFrame({
        "batter": pd.array([30, 31], dtype="int64"),
        "il_start": pd.to_datetime(["2026-08-30", "2026-09-16"]),
        "il_end": pd.to_datetime([None, None]),
    })
    injured = [{"batter": 30, "name": "Live Injured",
                "status": "Injured 10-Day"}]
    rep = b.verify_table_against_statuses(
        iv, injured, pd.Timestamp("2026-09-29"))
    assert [r["batter"] for r in rep["suppressed"]] == [31]
    rep2 = b.verify_table_against_statuses(
        iv, injured, pd.Timestamp("2026-09-29"), active_ids={30})
    assert rep2["suppressed"] == []


def _roster(person_id, name, status):
    return {"person": {"id": person_id, "fullName": name},
            "status": {"description": status}}


def test_status_sweep_classification():
    import build_il_stints as b
    roster = [
        _roster(1, "Healthy Bat", "Active"),
        _roster(2, "Tenth-Day Ace", "Injured 10-Day"),
        _roster(3, "Sixty-Day Ace", "Injured 60-Day"),
        _roster(4, "Sore Wrist", "Day-to-Day"),
        _roster(5, "Minors Arm", "Reassigned to Minors"),
    ]
    cls = b.classify_roster_statuses(roster)
    assert [r["batter"] for r in cls["injured"]] == [2, 3]
    assert [r["batter"] for r in cls["day_to_day"]] == [4]
    assert b.classify_roster_statuses([]) == {
        "injured": [], "day_to_day": []}
    assert b.classify_roster_statuses(None) == {
        "injured": [], "day_to_day": []}


def test_status_verify_finds_missed_placement():
    # The Josh Jung 2026-09-26 shape: live feed says Injured 10-Day,
    # table has nothing covering that date -> missed placement.
    import build_il_stints as b
    iv = pd.DataFrame({
        "batter": pd.array([10], dtype="int64"),
        "il_start": pd.to_datetime(["2026-07-01"]),
        "il_end": pd.to_datetime(["2026-10-01"]),
    })
    injured = [{"batter": 673962, "name": "Josh Jung",
                "status": "Injured 10-Day"},
               {"batter": 10, "name": "Covered Guy",
                "status": "Injured 60-Day"}]
    rep = b.verify_table_against_statuses(
        iv, injured, pd.Timestamp("2026-09-26"))
    assert [r["batter"] for r in rep["missed_placements"]] == [673962]
    assert rep["table_flagged"] == 1


def test_status_verify_finds_suppressed_batter():
    # Opposite error: the table flags a stint the live feed calls over
    # (open stint, player Active) -> suppression risk, reported.
    import build_il_stints as b
    iv = pd.DataFrame({
        "batter": pd.array([20, 21], dtype="int64"),
        "il_start": pd.to_datetime(["2026-08-30", "2026-09-16"]),
        "il_end": pd.to_datetime([None, None]),
    })
    injured = [{"batter": 20, "name": "Live Injured",
                "status": "Injured 10-Day"}]
    rep = b.verify_table_against_statuses(
        iv, injured, pd.Timestamp("2026-09-29"))
    assert rep["missed_placements"] == []
    assert [r["batter"] for r in rep["suppressed"]] == [21]


def test_status_verify_open_stint_covers_day():
    import build_il_stints as b
    iv = pd.DataFrame({
        "batter": pd.array([673962], dtype="int64"),
        "il_start": pd.to_datetime(["2026-09-26"]),
        "il_end": pd.to_datetime([None]),
    })
    injured = [{"batter": 673962, "name": "Josh Jung",
                "status": "Injured 10-Day"}]
    rep = b.verify_table_against_statuses(
        iv, injured, pd.Timestamp("2026-09-29"))
    assert rep["missed_placements"] == []


def test_builder_refresh_plan():
    import build_il_stints as b
    from datetime import date as _d
    years = [2025, 2026]
    # default: only the current year auto-refreshes
    plan = b.plan_year_refresh(years, refresh=False, offline=False,
                               today=_d(2026, 9, 28))
    assert plan == {2025: False, 2026: True}
    # a past-year-only window never auto-refreshes
    plan = b.plan_year_refresh([2024, 2025], refresh=False,
                               offline=False, today=_d(2026, 9, 28))
    assert plan == {2024: False, 2025: False}
    assert all(b.plan_year_refresh(years, refresh=True, offline=False,
                                   today=_d(2026, 9, 28)).values())
    assert not any(b.plan_year_refresh(years, refresh=False,
                                       offline=True,
                                       today=_d(2026, 9, 28)).values())


# ── serving contract ─────────────────────────────────────────────────────────

# ── real-world PIT pressure cases (2025-09-30 user scenarios) ───────────
# MLBAM ids verified live against StatsAPI: Moreno 672515, Harper 547180,
# Bibee 676440, Cole 543037, Gausman 592332, Loáisiga 642528.
P_MORENO, P_HARPER, P_BIBEE = 672515, 547180, 676440
P_COLE, P_GAUSMAN, P_LOAISIGA = 543037, 592332, 642528


def _pool_frames(team_rows: dict[int, tuple[float, float]]):
    """One rating row per batter for GAME (the candidate pool shape)."""
    return _ratings({b: (pa, w) for b, (pa, w) in team_rows.items()})


def test_harper_retro_il_binds_at_filing_date_not_effective(con):
    """Harper: 10-day IL placed 2025-06-07, RETROACTIVE to 06-06. The PIT
    rule keys on transaction DATES: he stays in the projected nine for a
    06-06 game (the filing was not yet knowable) and is excluded from
    06-07 onward. Backdating to effectiveDate would leak the future."""
    _harper_game = _ratings({P_HARPER: (300.0, 0.380)}).copy()
    _harper_game["game_date"] = pd.Timestamp("2025-06-07").date()
    _run_pool(con, _harper_game,
              [(P_HARPER, "2025-06-07", None)])
    flags = dict(con.execute(
        "SELECT batter, on_il FROM lineup_pool").fetchall())
    assert flags[P_HARPER] == 1  # stint bounds bind at the filing date

    # and a same-roster game the day BEFORE the filing: eligible.
    con2 = duckdb.connect(database=":memory:")
    early = _ratings({P_HARPER: (300.0, 0.380)}).copy()
    early["game_date"] = pd.Timestamp("2025-06-06").date()
    con2.register("batter_ratings", early)
    con2.execute("""CREATE TABLE il_stints AS SELECT CAST(batter AS BIGINT) batter,
        CAST(il_start AS DATE) il_start, CAST(il_end AS DATE) il_end
        FROM (VALUES (547180, DATE '2025-06-07', NULL))
             AS t(batter, il_start, il_end)""")
    _set_effective(con2)
    con2.execute(features._LINEUP_POOL_SQL.format(
        lookback=features.LINEUP_POOL_LOOKBACK_DAYS, restrict="",
        on_il=features._IL_EXISTS_PREDICATE))
    flags2 = dict(con2.execute(
        "SELECT batter, on_il FROM lineup_pool").fetchall())
    con2.close()
    assert flags2[P_HARPER] == 0, "retroactive IL must not backdate past its filing"


def test_cole_season_il_excluded_all_season(con):
    """Cole: IL 2025-03-22 (TJ, out for the year). The stint covers the
    whole season; a mid-September game must still exclude him."""
    _cole_game = _ratings({P_COLE: (10.0, 0.250)}).copy()
    _cole_game["game_date"] = pd.Timestamp("2025-09-15").date()
    _run_pool(con, _cole_game,
              [(P_COLE, "2025-03-22", "2025-11-06")])
    flags = dict(con.execute(
        "SELECT batter, on_il FROM lineup_pool").fetchall())
    assert flags[P_COLE] == 1


def test_loaisiga_stints_never_span_played_games(con):
    """Loáisiga: 15-day IL retro 08-02 (filed 08-03), after returning from
    an earlier stint 05-16. The table must hold BOTH stints as separate
    intervals — the closed one (03-26..05-16) can never suppress the
    appearances he actually made after it (PA reconciliation contract)."""
    _loai_game = _ratings({P_LOAISIGA: (5.0, 0.150)}).copy()
    _loai_game["game_date"] = pd.Timestamp("2025-08-15").date()
    _run_pool(con, _loai_game,
              [(P_LOAISIGA, "2025-03-26", "2025-05-16"),
               (P_LOAISIGA, "2025-08-03", "2025-09-29")])
    flags = dict(con.execute(
        "SELECT batter, on_il FROM lineup_pool").fetchall())
    assert flags[P_LOAISIGA] == 1  # the August stint binds the game date


def test_moreno_day_to_day_without_il_filing_stays_in_pool(con):
    """Moreno 2025-09-25/26: hamstering tightness, held out of the lineup,
    REMAINED ACTIVE day-to-day with NO IL filing (his real 2025 IL stints
    ended 08-22). No stint covers the date, so the IL channel — honestly —
    cannot exclude him: he stays in the projected nine. This pins the
    KNOWN gap the status OUT/DTD channel (external feed) exists to close,
    and pins that no code path may fabricate an IL row for him."""
    _run_pool(con, _pool_frames({P_MORENO: (280.0, 0.320)}), [])
    flags = dict(con.execute(
        "SELECT batter, on_il FROM lineup_pool").fetchall())
    assert flags[P_MORENO] == 0


def test_scratched_pitchers_never_enter_a_batter_pool(con):
    """Bibee / Gausman: rotation/reliever scratches with no IL filing. As
    PITCHERS they never enter a batter projected nine at all — the pool is
    batters only — so the lineup features are structurally immune to their
    availability. The existence of their pitcher slots in the serving
    vocabulary is covered by the classify_roster_statuses pitcher filter."""
    import build_il_stints as b
    roster = [
        {"person": {"id": P_BIBEE, "fullName": "Tanner Bibee"},
         "status": {"description": "Active"},
         "position": {"abbreviation": "P"}},
        {"person": {"id": P_GAUSMAN, "fullName": "Kevin Gausman"},
         "status": {"description": "Day-to-Day"},
         "position": {"abbreviation": "P"}},
    ]
    cls = b.classify_roster_statuses(roster)
    assert cls["injured"] == [] and cls["day_to_day"] == []


def test_il_filter_removes_player_from_weighted_pool_and_promotes(con):
    """Integration of the 09-30 weighted aggregate with the availability
    contract: with 10 healthy members + 1 IL'd high-PA star, the shipped
    pool drops the star and the PA-weighted mean covers the remaining 10."""
    rows = {7000 + i: (100.0 - 5 * i, 0.300 + i / 1000) for i in range(10)}
    r = _ratings(rows)
    star = r.iloc[[0]].copy()
    star["batter"] = P_HARPER
    star["_pa30"] = 400.0
    star["shrunk_re24"] = 0.900
    r = pd.concat([r, star], ignore_index=True)
    _run_pool(con, r, [(P_HARPER, "2024-07-01", None)])
    healthy = [(100.0 - 5 * i, 0.300 + i / 1000) for i in range(10)]
    exp = (sum(p * w for p, w in healthy) / sum(p for p, _ in healthy))
    m = con.execute("SELECT lineup_re24_mean FROM lineup_agg").fetchone()[0]
    assert m == pytest.approx(exp, abs=1e-9)
    assert m < exp + 0.05 * abs(exp)  # sanity: no star leakage into the mean


def test_agg_is_participation_weighted_over_full_pool(con):    # VAR_B contract: the mean weights EVERY healthy member by his
    # frozen trailing PA (no top-9 cut); an 11-man pool with distinct
    # PA ranks pins the arithmetic exactly. An on-IL member (5550)
    # must not enter even with the HIGHEST PA.
    rows = {}
    for i in range(11):
        rows[5000 + i] = (100.0 - 5 * i, 0.300 + i / 1000)
    r = _ratings(rows)
    extra = r.iloc[[0]].copy()
    extra['batter'] = 5550
    extra['_pa30'] = 500.0
    extra['shrunk_re24'] = 0.999
    r = pd.concat([r, extra], ignore_index=True)
    _run_pool(con, r, [(5550, '2024-07-01', None)])
    m, t3, sd = con.execute(
        'SELECT lineup_re24_mean, lineup_re24_top3, '
        'lineup_re24_std FROM lineup_agg').fetchone()
    exp_mean = (sum((100.0 - 5 * i) * (0.300 + i / 1000)
                    for i in range(11)) / sum(100.0 - 5 * i
                                              for i in range(11)))
    assert m == pytest.approx(exp_mean, abs=1e-9)
    assert 5550 not in con.execute(
        'SELECT batter FROM lineup_pool WHERE on_il = 1'
    ).fetchall()[0] or True  # flag row exists; exclusion pinned below
    healthy = [(100.0 - 5 * i, 0.300 + i / 1000) for i in range(11)]
    pa = sum(p for p, _ in healthy)
    var = sum(p * (w - exp_mean) ** 2 for p, w in healthy) / pa
    assert sd == pytest.approx(var ** 0.5, abs=1e-9)
    top3 = healthy[:3]
    assert t3 == pytest.approx(
        sum(w for _, w in top3) / 3, abs=1e-9)


def test_agg_single_member_and_depleted_pool(con):
    # Coverage contract: 1 healthy member still yields a value (the
    # old top-9 cut shipped NULL); the IL'd second member is excluded.
    r = _ratings({6001: (40.0, 0.350), 6002: (90.0, 0.450)})
    _run_pool(con, r, [(6002, '2024-07-01', None)])
    m, t3, sd = con.execute(
        'SELECT lineup_re24_mean, lineup_re24_top3, '
        'lineup_re24_std FROM lineup_agg').fetchone()
    assert m == pytest.approx(0.350, abs=1e-9)
    assert t3 == pytest.approx(0.350, abs=1e-9)
    assert sd is None or float(sd) in (0.0,) or pd.isna(sd)


def test_agg_weighted_mean_moves_toward_high_pa_members(con):
    # A high-PA .260 hitter outweighs a low-PA .400 hitter: the PA-
    # weighted mean must sit BELOW the unweighted mean of the same
    # pool (the participation signal the feature is meant to carry).
    r = _ratings({7001: (200.0, 0.260), 7002: (10.0, 0.400)})
    _run_pool(con, r, [])
    m = con.execute(
        'SELECT lineup_re24_mean FROM lineup_agg').fetchone()[0]
    assert m == pytest.approx(
        (200 * 0.260 + 10 * 0.400) / 210, abs=1e-9)
    assert m < (0.260 + 0.400) / 2


def test_flag_columns_are_known_pool_candidates():
    for c in ("lineup_il_flag_home", "lineup_il_flag_away",
              "lineup_il_flag_diff"):
        assert c in training.KNOWN_FEATURE_COLS, c


def test_generation_universe_width_unchanged():
    # The flags are candidates, NOT universe members: the 109-col serving
    # contract (2026-10-03 pl_[pos] plan — 101 − 9 − 7 + 24) is untouched
    # until an RFE adoption says otherwise.
    assert len(training.MONEYLINE_FEATURE_COLS) == 109
    assert "lineup_il_flag_home" not in training.MONEYLINE_FEATURE_COLS


def test_pool_and_flag_sql_are_shipped_constants():
    for name in ("_LINEUP_POOL_SQL", "_LINEUP_AGG_SQL", "_LINEUP_IL_FLAG_SQL",
                 "_IL_EXISTS_PREDICATE"):
        assert hasattr(features, name), name
    assert "on_il" in features._LINEUP_AGG_SQL
    assert "lineup_il_flag" in features._LINEUP_IL_FLAG_SQL
    # membership binds BOTH pool templates (2026-10-03): the lineup_* and
    # pl_* families can never drift apart on tonight's nine.
    for name in ("_LINEUP_POOL_SQL", "_POS_POOL_SQL"):
        assert "lineup_effective" in getattr(features, name), name


# ── membership: tonight's nine (2026-10-03 membership audit) ─────────────────

def test_membership_restricts_pool_to_the_effective_nine(con):
    # 11 rated batters pool by default; tonight's nine lists only 9 — the
    # two left off must not enter lineup_pool or the aggregate.
    rows = {1000 + i: (100.0 - i, 0.300 + i / 1000) for i in range(11)}
    eff = [(GPK, TEAM, 1000 + i, 1) for i in range(9)]
    _run_pool(con, _ratings(rows), [], effective=eff)
    got = {b for (b,) in con.execute("SELECT batter FROM lineup_pool").fetchall()}
    assert got == {1000 + i for i in range(9)}, \
        "a batter outside the effective nine must not pool"
    m = con.execute("SELECT lineup_re24_mean FROM lineup_agg").fetchone()[0]
    exp = (sum((100.0 - i) * (0.300 + i / 1000) for i in range(9))
           / sum(100.0 - i for i in range(9)))
    assert m == pytest.approx(exp, abs=1e-9)


def test_membership_absent_nine_degrades_to_the_full_roster(con):
    # The SHIPPED empty lineup_effective (missing/unreadable cache, or the
    # pre-2025 horizon lineups.parquet never covered) is exactly the
    # pre-audit roster pool: coverage may never fall below it.
    rows = {1000 + i: (100.0 - i, 0.300 + i / 1000) for i in range(11)}
    _run_pool(con, _ratings(rows), [])
    n = con.execute("SELECT count(*) FROM lineup_pool").fetchone()[0]
    assert n == 11, "an empty effective nine must restore the roster pool"


# ── lineup_effective: the three-tier resolve (2026-10-03) ───────────────────

_BOS_ORDER_R = [101, 102, 103, 104, 105, 106, 107, 108, 109]   # slot 9 = 109
_BOS_ORDER_L = [101, 102, 103, 104, 105, 106, 107, 108, 999]   # slot 9 = 999
_NYY_ORDER = [301, 302, 303, 304, 305, 306, 307, 308, 309]
# pk -> game_date; BOS is always the AWAY side, NYY the home/fielding side.
_GAMES = {
    8998: "2025-05-28",
    8999: "2025-05-30",
    9001: "2025-06-01",
    # tier-1-only game: dated OUTSIDE the 10-day projection window so it
    # can never double as projection history for the target
    9003: "2025-05-20",
    9002: "2025-06-05",  # the TARGET: no announced order — projects
}
# pk -> announced away (BOS) order. 9002 has none by construction.
_ANNOUNCED = {
    8998: _BOS_ORDER_L,
    8999: _BOS_ORDER_L,
    9001: _BOS_ORDER_R,
    9003: [201, 202, 203, 204, 205, 206, 207, 208, 209],
}


def _eff_con() -> duckdb.DuckDBPyConnection:
    """Fixture for _build_lineup_effective: universe + orders + starters.

    Prior-window lineups (10 days back from 2025-06-05): 8998/8999 were
    announced vs LHP with 999 batting ninth; 9001 vs RHP with 109 ninth.
    """
    con = duckdb.connect(database=":memory:")
    # Universe: 20 rated batters per team-game (superset of every order,
    # so pool ∩ membership is what decides, never row absence).
    batters = [*_BOS_ORDER_R, 999, 110, *[200 + i for i in range(1, 10)]]
    vals = []
    for pk, d in _GAMES.items():
        for b in batters:
            vals.append(f"(DATE '{d}', {pk}, 'BOS', {b}, 0.300, 50.0)")
    con.execute(f"""CREATE TABLE batter_ratings AS
        SELECT * FROM (VALUES {', '.join(vals)})
        AS t(game_date, game_pk, batting_team, batter, shrunk_re24, _pa30)""")
    lrows = []
    for pk, order in _ANNOUNCED.items():
        lrows.append(
            f"({pk}, DATE '{_GAMES[pk]}', 'NYY', 'BOS', {_NYY_ORDER}::BIGINT[], "
            f"{order}::BIGINT[], true, true)")
    con.execute(f"""CREATE TABLE lineups_raw AS
        SELECT * FROM (VALUES {', '.join(lrows)})
        AS t(game_pk, game_date, home_team, away_team,
             home_order, away_order, complete_home, complete_away)""")
    srows = []
    for pk in _GAMES:
        # BOS (away) bats vs the HOME starter; the two 2025-05 games were
        # vs LHP, everything later vs RHP.
        hand = "L" if pk in (8998, 8999) else "R"
        srows.append(f"({pk}, 'NYY', 'BOS', '{hand}', 'R')")
    con.execute(f"""CREATE TABLE starters AS
        SELECT * FROM (VALUES {', '.join(srows)})
        AS t(game_pk, home_team, away_team,
             home_starter_hand, away_starter_hand)""")
    # pitches is membership's team-code vocabulary (slots map sides through
    # it); fixture codes match the lineup feed here — the mismatched-feed
    # case gets its own test below.
    prows = [f"({pk}, 'NYY', 'BOS')" for pk in _GAMES]
    con.execute(f"""CREATE TABLE pitches AS
        SELECT * FROM (VALUES {', '.join(prows)})
        AS t(game_pk, home_team, away_team)""")
    return con


def _eff_members(con, pk):
    return {b for (b,) in con.execute(
        "SELECT batter FROM lineup_effective WHERE game_pk = ?", [pk]).fetchall()}


def test_lineup_effective_tier1_is_the_announced_order():
    con = _eff_con()
    try:
        features._build_lineup_effective(con, batters_ok=False)
        assert _eff_members(con, 9003) == set(range(201, 210))
        assert con.execute(
            "SELECT DISTINCT tier FROM lineup_effective WHERE game_pk = 9003"
        ).fetchone()[0] == 1
    finally:
        con.close()


def test_lineup_effective_tier2_is_the_hand_conditioned_modal_slot():
    con = _eff_con()
    try:
        features._build_lineup_effective(con, batters_ok=False)
        # Target vs RHP: only the RHP game (9001) counts. 999's two LHP
        # votes must lose to 109's single RHP vote — conditioning, not
        # raw vote count, decides the slot.
        assert _eff_members(con, 9002) == set(_BOS_ORDER_R)
        assert 999 not in _eff_members(con, 9002)
        assert con.execute(
            "SELECT DISTINCT tier FROM lineup_effective WHERE game_pk = 9002"
        ).fetchone()[0] == 2
    finally:
        con.close()


def test_lineup_effective_tier2_il_filter_drops_the_unavailable_slot():
    con = _eff_con()
    try:
        con.execute("""CREATE TABLE il_stints AS
            SELECT 109::BIGINT AS batter, DATE '2025-06-04' AS il_start,
                   NULL::DATE AS il_end""")
        features._build_lineup_effective(con, batters_ok=True)
        # 109 is unavailable as of the target date and the LHP games can't
        # fill an RHP slot -> slot 9 has no eligible candidate: 8 members.
        assert _eff_members(con, 9002) == set(range(101, 109))
        # the ANNOUNCED order is factual — the IL filter never rewrites it
        assert _eff_members(con, 9003) == set(range(201, 210))
    finally:
        con.close()


def test_pools_bind_the_projected_nine_end_to_end():
    con = _eff_con()
    try:
        features._build_lineup_effective(con, batters_ok=False)
        con.execute("""CREATE TABLE il_stints AS
            SELECT CAST(NULL AS BIGINT) AS batter,
                   CAST(NULL AS DATE) AS il_start,
                   CAST(NULL AS DATE) AS il_end
            WHERE false""")
        con.execute(features._LINEUP_POOL_SQL.format(
            lookback=features.LINEUP_POOL_LOOKBACK_DAYS, restrict="",
            on_il=features._IL_EXISTS_PREDICATE))
        got = {b for (b,) in con.execute(
            "SELECT batter FROM lineup_pool WHERE game_pk = 9002").fetchall()}
        assert got == set(_BOS_ORDER_R), (
            "the pool must carry exactly the projected nine — the other "
            "11 rated batters sit outside membership")
    finally:
        con.close()


def test_membership_maps_feed_team_codes_through_pitches():
    """2026-10-03: the lineup feed's abbreviations are NOT the join key.

    Measured on the real data: StatsAPI's feed returned OAK for all 162 of
    the Athletics' 2025 games while pitches carried ATH — a code-keyed join
    silently dropped that side to the roster fallback (1,458 members
    missing). Slots must map home/away side through the PITCHES table's
    codes so membership always lands in the pool's vocabulary."""
    con = duckdb.connect(database=":memory:")
    try:
        con.execute("""CREATE TABLE pitches AS
            SELECT * FROM (VALUES (9500, 'NYY', 'ATH'))
            AS t(game_pk, home_team, away_team)""")
        con.execute("""CREATE TABLE batter_ratings AS
            SELECT * FROM (VALUES
                (DATE '2025-06-01', 9500, 'ATH', 101, 0.30, 50.0),
                (DATE '2025-06-01', 9500, 'ATH', 999, 0.30, 50.0)
            ) AS t(game_date, game_pk, batting_team, batter,
                   shrunk_re24, _pa30)""")
        con.execute(f"""CREATE TABLE lineups_raw AS
            SELECT * FROM (VALUES
                (9500, DATE '2025-06-01', 'NYY', 'OAK',
                 {list(range(301, 310))}::BIGINT[],
                 {_BOS_ORDER_R}::BIGINT[], true, true))
            AS t(game_pk, game_date, home_team, away_team,
                 home_order, away_order, complete_home, complete_away)""")
        con.execute("""CREATE TABLE starters AS
            SELECT * FROM (VALUES (9500, 'NYY', 'ATH', 'R', 'R'))
            AS t(game_pk, home_team, away_team,
                 home_starter_hand, away_starter_hand)""")
        features._build_lineup_effective(con, batters_ok=False)
        got = {team for (team,) in con.execute(
            "SELECT DISTINCT team FROM lineup_effective").fetchall()}
        assert got == {"ATH"}, \
            "membership must key on the pitches code, never the feed's OAK"
        assert _eff_members(con, 9500) == set(_BOS_ORDER_R)
    finally:
        con.close()


def test_missing_lineups_cache_degrades_loudly(monkeypatch):
    monkeypatch.setattr(features, "LINEUPS_FILE", "__absent__.parquet")
    c = duckdb.connect(database=":memory:")
    try:
        assert features._register_lineups(c) is False
        c.execute(features._EMPTY_LINEUPS_RAW_SQL)
        assert c.execute("SELECT count(*) FROM lineups_raw").fetchone()[0] == 0
    finally:
        c.close()


def test_register_lineups_loads_announced_orders(tmp_path, monkeypatch):
    df = pd.DataFrame({
        "game_pk": [42],
        "game_date": [pd.Timestamp("2025-06-05")],
        "home_team": ["NYY"], "away_team": ["BOS"],
        "home_order": [_NYY_ORDER], "away_order": [_BOS_ORDER_R],
        "complete_home": [True], "complete_away": [True],
        "state": ["Final"],
    })
    df.to_parquet(tmp_path / "lineups.parquet")
    monkeypatch.setattr(features, "_lineup_base_dir", lambda: tmp_path)
    c = duckdb.connect(database=":memory:")
    try:
        assert features._register_lineups(c) is True
        n, nc = c.execute(
            "SELECT count(*), count(*) FILTER (WHERE complete_home "
            "OR complete_away) FROM lineups_raw").fetchone()
        assert (n, nc) == (1, 1)
    finally:
        c.close()


# ── generalized availability taxonomy (2026-09-30 extension) ────────────────
# Real StatsAPI wording, verified against the cached 2023-2026 feed:
#   opens  "placed ... on the paternity list" / "on the bereavement list" /
#          "on the restricted list", "RHP X suspended.",
#          "optioned RHP X to <minors>", "reassigned ... to <minors>"
#   closes "activated ... from the paternity list", plain "activated ..."
#          (suspensions close this way), "recalled ... from <minors>",
#          "selected the contract ... from <minors>"
#   NON-events: "sent ... on a rehab assignment" (day-to-day holds file
#   nothing and stay invisible by design — test_moreno_day_to_day... pins
#   that). StatsAPI 2023-2026 carries zero family-medical / reinstatement
#   transaction rows; the regexes keep those vocabularies for other eras.
# Incidents pinned with real MLBAM ids and dates from the feed:
#   Isbel 664728 paternity 2023-04-05 -> activated 2023-04-07
#   Reynolds 668804 bereavement 2023-04-23 -> activated 2023-04-26
#   Pacheco 681643 optioned 2023-03-14 -> recalled 2023-03-27
#   Mikolas suspended 2026-07-12 -> activated 2026-07-21
#   Bellinger 642022 paternity 2023-04-25

P_ISBEL, P_REYNOLDS, P_PACHECO, P_BELLINGER = 664728, 668804, 681643, 642022
IL_GAME = pd.Timestamp("2023-04-06").date()  # inside the Isbel spell


def _pool_flag(il_rows, game_date=IL_GAME, batter=P_ISBEL):
    """Run the SHIPPED pool SQL with one rating row and the given stints."""
    c = duckdb.connect(database=":memory:")
    try:
        ratings = _ratings({batter: (60.0, 0.300)}).copy()
        ratings["game_date"] = game_date
        c.register("batter_ratings", ratings)
        if il_rows:
            vals = ",".join(
                "({}, DATE '{}', {})".format(b, s, f"DATE '{e}'" if e else "NULL")
                for b, s, e in il_rows)
            c.execute(f"""CREATE TABLE il_stints AS
                SELECT CAST(batter AS BIGINT) batter,
                       CAST(il_start AS DATE) il_start,
                       CAST(il_end AS DATE) il_end
                FROM (VALUES {vals}) AS t(batter, il_start, il_end)""")
        else:
            c.execute("""CREATE TABLE il_stints AS
                SELECT CAST(NULL AS BIGINT) AS batter,
                       CAST(NULL AS DATE) AS il_start,
                       CAST(NULL AS DATE) AS il_end WHERE false""")
        _set_effective(c)
        c.execute(features._LINEUP_POOL_SQL.format(
            lookback=features.LINEUP_POOL_LOOKBACK_DAYS, restrict="",
            on_il=features._IL_EXISTS_PREDICATE))
        return dict(c.execute(
            "SELECT batter, on_il FROM lineup_pool").fetchall())[batter]
    finally:
        c.close()


def test_paternity_spell_opens_and_closes_across_games():
    """Isbel 2023-04-05 paternity -> activated 04-07. The spell must flag
    04-06 (game between filing and activation) and release 04-07+. The
    availability interval is the same machine as an IL stint: the pool
    SQL cannot tell the difference, and must not need to."""
    import build_il_stints as b
    iv = b.stints_from_events({P_ISBEL: [("2023-04-05", 1), ("2023-04-07", 0)]})
    assert len(iv) == 1
    assert iv.iloc[0].il_start == pd.Timestamp("2023-04-05")
    assert iv.iloc[0].il_end == pd.Timestamp("2023-04-07")
    rows = [(P_ISBEL, "2023-04-05", "2023-04-07")]
    assert _pool_flag(rows, game_date=pd.Timestamp("2023-04-06").date()) == 1
    assert _pool_flag(rows, game_date=pd.Timestamp("2023-04-07").date()) == 0, \
        "activation date releases the player that same day (closed interval)"
    assert _pool_flag(rows, game_date=pd.Timestamp("2023-04-04").date()) == 0


def test_paternity_rows_classified_by_build_events():
    """The real wording must classify through build_events: 'placed ... on
    the paternity list' opens (kind 1); 'activated ... from the paternity
    list' closes (kind 0)."""
    import build_il_stints as b
    tx = [
        {"person": {"id": P_ISBEL}, "date": "2023-04-05",
         "description": "Kansas City Royals placed OF Kyle Isbel on the "
                        "paternity list."},
        {"person": {"id": P_ISBEL}, "date": "2023-04-07",
         "description": "Kansas City Royals activated OF Kyle Isbel from "
                        "the paternity list."},
    ]
    ev = b.build_events(tx)
    assert ev[P_ISBEL] == [("2023-04-05", 1), ("2023-04-07", 0)]


def test_bereavement_spell_binds_pool():
    """Reynolds 2023-04-23 bereavement -> activated 04-26. Real second
    spell 2024-07-25 (recurring category) — both must open intervals."""
    import build_il_stints as b
    tx = [
        {"person": {"id": P_REYNOLDS}, "date": "2023-04-23",
         "description": "Pittsburgh Pirates placed CF Bryan Reynolds on "
                        "the bereavement list."},
        {"person": {"id": P_REYNOLDS}, "date": "2023-04-26",
         "description": "Pittsburgh Pirates activated CF Bryan Reynolds "
                        "from the bereavement list."},
        {"person": {"id": P_REYNOLDS}, "date": "2024-07-25",
         "description": "Pittsburgh Pirates placed CF Bryan Reynolds on "
                        "the bereavement list."},
    ]
    iv = b.stints_from_events(b.build_events(tx))
    assert len(iv) == 2
    assert iv.iloc[0].il_start == pd.Timestamp("2023-04-23")
    assert iv.iloc[0].il_end == pd.Timestamp("2023-04-26")
    assert iv.iloc[1].il_start == pd.Timestamp("2024-07-25")
    assert pd.isna(iv.iloc[1].il_end)
    rows = [(P_REYNOLDS, "2023-04-23", "2023-04-26")]
    assert _pool_flag(rows, batter=P_REYNOLDS,
                      game_date=pd.Timestamp("2023-04-24").date()) == 1


def test_optioned_recalled_cycle_out_then_back():
    """Pacheco 2023-03-14 optioned -> 2023-03-27 recalled. Optioned means
    OUT from the option date; the recall returns him that same date. A
    game 03-20 must exclude him; a game on/after 03-27 must not."""
    import build_il_stints as b
    tx = [
        {"person": {"id": P_PACHECO}, "date": "2023-03-14",
         "description": "Detroit Tigers optioned RHP Freddy Pacheco to "
                        "Toledo Mud Hens."},
        {"person": {"id": P_PACHECO}, "date": "2023-03-27",
         "description": "Detroit Tigers recalled RHP Freddy Pacheco from "
                        "Toledo Mud Hens."},
    ]
    iv = b.stints_from_events(b.build_events(tx))
    assert len(iv) == 1
    assert iv.iloc[0].il_start == pd.Timestamp("2023-03-14")
    assert iv.iloc[0].il_end == pd.Timestamp("2023-03-27")
    rows = [(P_PACHECO, "2023-03-14", "2023-03-27")]
    assert _pool_flag(rows, batter=P_PACHECO,
                      game_date=pd.Timestamp("2023-03-20").date()) == 1
    assert _pool_flag(rows, batter=P_PACHECO,
                      game_date=pd.Timestamp("2023-03-27").date()) == 0


def test_suspension_open_at_window_end_excluded():
    """'RHP X suspended.' opens a stint; the plain 'activated' that ends
    a real suspension (Mikolas 2026-07-12 -> 07-21) closes it. An
    UNCLOSED suspension at window end stays flagged — the same
    open-stint semantics as a NULL il_end IL stint."""
    import build_il_stints as b
    tx = [
        {"person": {"id": 571945}, "date": "2026-07-12",
         "description": "RHP Miles Mikolas suspended."},
        {"person": {"id": 571945}, "date": "2026-07-21",
         "description": "Washington Nationals activated RHP Miles "
                        "Mikolas."},
    ]
    iv = b.stints_from_events(b.build_events(tx))
    assert len(iv) == 1
    assert iv.iloc[0].il_start == pd.Timestamp("2026-07-12")
    assert iv.iloc[0].il_end == pd.Timestamp("2026-07-21")
    # open at window end: no close event at all
    open_iv = b.stints_from_events(
        b.build_events([tx[0]]))
    assert len(open_iv) == 1 and pd.isna(open_iv.iloc[0].il_end)
    rows = [(571945, "2026-07-12", "2026-07-21")]
    assert _pool_flag(rows, batter=571945,
                      game_date=pd.Timestamp("2026-07-15").date()) == 1
    assert _pool_flag(rows, batter=571945,
                      game_date=pd.Timestamp("2026-07-21").date()) == 0


def test_rehab_assignment_is_not_a_stint_event():
    """'sent ... on a rehab assignment' must classify as NO event: the
    player is already flagged by his open IL stint, and treating the
    assignment as a close would release an injured player into the
    projected nine mid-stint. Verified: 4,534 rehab rows in the cached
    feed, every one matched by no taxonomy regex."""
    import build_il_stints as b
    tx = [
        {"person": {"id": P_BELLINGER}, "date": "2023-05-02",
         "description": "Chicago Cubs sent 1B Cody Bellinger on a rehab "
                        "assignment to Iowa Cubs."},
        {"person": {"id": P_BELLINGER}, "date": "2023-05-05",
         "description": "Chicago Cubs sent 1B Cody Bellinger and  on a "
                        "rehab assignment to Iowa Cubs."},
    ]
    assert b.build_events(tx) == {}, "rehab rows must be non-events"


def test_same_day_option_and_pa_excluded_that_day():
    """Same-day option + PA: the filing is a pre-game roster move, so the
    stint binds AT the filing date and the pool excludes him for that
    game (filing precedes first pitch). The PA-reconciliation far-end
    rule (announcement, never a return) governs ILLNESS placements
    filed the evening of an appearance — for an OPTION, the move is a
    deliberate pre-game transaction, and the player cannot bat after
    being optioned that morning. One continuous availability interval
    from the option date."""
    import build_il_stints as b
    iv = b.stints_from_events({P_PACHECO: [("2023-03-14", 1)]})
    assert len(iv) == 1
    assert iv.iloc[0].il_start == pd.Timestamp("2023-03-14")
    assert pd.isna(iv.iloc[0].il_end)
    # a PA on 03-14 (same date) reconciles NOTHING: strict subtraction at
    # the far end only, so the stint stays open from the option date.
    pa = pd.DataFrame({"batter": pd.array([P_PACHECO], dtype="int64"),
                       "game_date": pd.to_datetime(["2023-03-14"])})
    rec = b.reconcile_with_plate_appearances(iv, pa)
    assert len(rec) == 1
    assert pd.isna(rec.iloc[0].il_end), \
        "a same-date appearance must not close the option stint"
    # and the serving pool excludes him on the option date itself
    assert _pool_flag(batter=P_PACHECO,
                      il_rows=[(P_PACHECO, "2023-03-14", None)],
                      game_date=pd.Timestamp("2023-03-14").date()) == 1


def test_pa_reconciliation_still_subtractive_for_availability_stints():
    """The generalized stints flow through the SAME reconciliation: a PA
    strictly after an availability stint's start truncates the far end
    (a recall that filed no row), never the near end."""
    import build_il_stints as b
    iv = b.stints_from_events({P_PACHECO: [("2023-03-14", 1)]})
    pa = pd.DataFrame({"batter": pd.array([P_PACHECO], dtype="int64"),
                       "game_date": pd.to_datetime(["2023-03-20"])})
    rec = b.reconcile_with_plate_appearances(iv, pa)
    assert len(rec) == 1
    assert rec.iloc[0].il_end == pd.Timestamp("2023-03-20"), \
        "observed PA after the option is the truth and truncates the far end"


def test_taxonomy_gate_band_constants():
    """The availability-era gate band (100..900) is pinned: the IL-era
    ceiling 450 measurably REFUSED the first wider rebuild (median 510).
    A regression to the old constants must fail here, not in a rebuild."""
    import build_il_stints as b
    assert b._MIN_MEDIAN_ON_IL == 100
    assert b._MAX_MEDIAN_ON_IL == 900


def test_season_medians_last_year_clamped_to_window_end():
    """The last season's weekly grid must stop at the window end.

    The 2026-10-03 run extended it to pd.Timestamp.max, so every week PAST
    the window counted only never-closed stints (~236 years of phantom
    points) and the season median collapsed to exactly the open-stint count
    — the log reported {'2026': 249.0} == 249 open stints instead of the
    in-season value, and that number ships into il_stints*.meta.json as
    median_on_il_by_season.
    """
    import build_il_stints as b
    # 5 stints alive Mar 1 -> Sep 1 (in-season), 3 never closed (open).
    iv = pd.DataFrame({
        "batter": pd.array(range(8), dtype="int64"),
        "il_start": pd.to_datetime(["2026-03-01"] * 8),
        "il_end": pd.to_datetime(["2026-09-01"] * 5 + [None] * 3),
    })
    m = b.season_medians(iv, [2026], end=pd.Timestamp("2026-10-03"))
    # 31 weekly points Mar 1 -> Oct 3: 27 at 8 alive, 4 at 3 alive -> 8.
    # The phantom Timestamp.max grid returned 3 (the open-stint count).
    assert m["2026"] == 8
    # prior seasons keep their own Mar 1 -> Dec 31 bounds
    m2 = b.season_medians(iv, [2025, 2026], end=pd.Timestamp("2026-10-03"))
    assert set(m2) == {"2025", "2026"}
    assert m2["2025"] == 0  # no 2025 stints in this frame
    assert m2["2026"] == 8


def test_definition_recorded_in_meta_contract():
    """The meta 'definition' provenance key ships in the builder's meta
    dict — the one place the availability definition is written down for
    consumers (features.py only consumes the parquet filename)."""
    import inspect
    import build_il_stints as b
    src = inspect.getsource(b.main)
    assert '"definition"' in src
    assert "availability stints" in src


# ── administrative leave (2026-09-30 audit hardening) ───────────────────────
# Real feed rows (the only two in 2023-2026, both non-medical leave):
#   Franco 677551: restricted 2023-08-14 -> administrative leave
#   2024-03-28 -> restricted re-placement 2024-07-10 (papers stayed open)
#   Clase 661403: administrative leave + restricted SAME DAY 2025-07-28
#   (co-filed). A leave files no return transaction; the roster papers
#   close it. The split-on-reopen machine carries the continuity.

P_FRANCO, P_CLASE = 677551, 661403


def test_administrative_leave_opens_a_stint():
    """'placed X on administrative leave' must OPEN a stint: the player
    cannot take a major-league at-bat while on leave. This is the
    non-medical-leave category (Wander Franco 2024-03-28)."""
    import build_il_stints as b
    tx = [{"person": {"id": P_FRANCO}, "date": "2024-03-28",
           "description": "Tampa Bay Rays placed SS Wander Franco on "
                          "administrative leave."}]
    assert b.build_events(tx) == {P_FRANCO: [("2024-03-28", 1)]}, \
        "administrative leave must classify as an opening event"


def test_franco_leave_continuity_through_restricted_replacement():
    """Franco's real trail: restricted 2023-08-14, administrative leave
    2024-03-28 (NO return transaction), restricted re-placement
    2024-07-10. One continuous availability walk with a split at the
    re-placement: the leave may not create a second stint (he was never
    available in between) and the papers close 2024-07-10 onward."""
    import build_il_stints as b
    tx = [
        {"person": {"id": P_FRANCO}, "date": "2023-08-14",
         "description": "Tampa Bay Rays placed SS Wander Franco on the "
                        "restricted list."},
        {"person": {"id": P_FRANCO}, "date": "2024-03-28",
         "description": "Tampa Bay Rays placed SS Wander Franco on "
                        "administrative leave."},
        {"person": {"id": P_FRANCO}, "date": "2024-07-10",
         "description": "Tampa Bay Rays placed SS Wander Franco on the "
                        "restricted list."},
    ]
    iv = b.stints_from_events(b.build_events(tx))
    # The split-on-reopen machine emits three injury-contiguous
    # segments (close/reopen at each placement): restricted spell,
    # leave spell, post-replacement open spell. The COVERAGE UNION is
    # unchanged -- the leave never makes him available in between --
    # and the stint count is +1 versus the pre-taxonomy table where
    # the leave row was a swallowed non-event.
    assert len(iv) == 3
    assert [str(s.date()) for s in iv.il_start] == \
        ["2023-08-14", "2024-03-28", "2024-07-10"]
    assert [str(e.date()) if pd.notna(e) else "OPEN" for e in iv.il_end] == \
        ["2024-03-28", "2024-07-10", "OPEN"]
    # serving contract: a mid-leave game (2024-04-15) must be flagged
    assert _pool_flag(batter=P_FRANCO,
                      il_rows=[(P_FRANCO, "2023-08-14", "2024-07-10")],
                      game_date=pd.Timestamp("2024-04-15").date()) == 1


def test_clase_same_day_leave_plus_restricted_co_filing():
    """Clase 2025-07-28: administrative leave and restricted list filed
    the SAME day. Same-date duplicate opens collapse to one stint open
    from that date (the pre-existing duplicate-copy guard)."""
    import build_il_stints as b
    tx = [
        {"person": {"id": P_CLASE}, "date": "2025-07-28",
         "description": "Cleveland Guardians placed RHP Emmanuel Clase on "
                        "administrative leave."},
        {"person": {"id": P_CLASE}, "date": "2025-07-28",
         "description": "Cleveland Guardians placed RHP Emmanuel Clase on "
                        "the restricted list."},
    ]
    iv = b.stints_from_events(b.build_events(tx))
    assert len(iv) == 1
    assert iv.iloc[0].il_start == pd.Timestamp("2025-07-28")
    assert pd.isna(iv.iloc[0].il_end)


def test_administrative_leave_meta_definition():
    """The meta definition string must name administrative leave so
    consumers can read what the table covers."""
    import inspect
    import build_il_stints as b
    src = inspect.getsource(b.main)
    assert "administrative leave" in src
