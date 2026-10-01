"""Bullpen availability (2026-09-30): the bullpen family gets the same
player-availability layer the lineup family has — as a FILTER, not new
features (2026-09-30 directive).

Pins:
  * il_stints_pitchers.parquet registers; a missing ledger degrades LOUDLY
    to all-pitched-innings semantics (pre-availability behavior)
  * innings by an unavailable arm are EXCLUDED from bullpen_raw, so every
    served bullpen feature (bullpen_whip_10g, bullpen_whip_3g, era_10g and
    everything derived from them) inherits tonight's availability
  * PIT rule: innings ON the placement date and ON the return date stay
    available (strictly-between predicate)
  * bullpen_whip_diff is RENAMED bullpen_whip_10g_diff (values unchanged)
  * bullpen_meltdown_risk is RENAMED ..._risk_diff; per-side twins exist
    as the within-side product of the family's own factors
  * the serving contract carries exactly the family: 100 cols, NO
    exposed_share / il_count extras
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import features  # noqa: E402
import training  # noqa: E402

BACKEND = Path(__file__).resolve().parent


# ── contract-level pins ──────────────────────────────────────────────────────

def test_contract_carries_the_renamed_bullpen_family():
    cols = set(training.MONEYLINE_FEATURE_COLS)
    for c in ("bullpen_whip_10g_diff", "bullpen_whip_3g_diff",
              "bullpen_pitches_diff", "bullpen_meltdown_risk_diff",
              "bullpen_meltdown_risk_home", "bullpen_meltdown_risk_away"):
        assert c in cols, f"{c} missing from the serving contract"
    assert "bullpen_whip_diff" not in cols
    assert "bullpen_meltdown_risk" not in cols
    # The availability mechanism is a filter, NOT extra model columns.
    assert not [c for c in cols if "exposed_share" in c or "il_count" in c]
    assert len(training.MONEYLINE_FEATURE_COLS) == 101


def test_add_diff_features_emits_the_renamed_whip_diff(tmp_path, monkeypatch):
    """The rename is value-identical: 10g_diff == home - away, always."""
    rng = np.random.default_rng(5)
    n = 40
    df = pd.DataFrame({
        "home_team": ["BOS"] * n,
        "away_team": ["NYY"] * n,
        "game_pk": np.arange(n),
        "bullpen_whip_10g_home": rng.uniform(0.8, 1.8, n),
        "bullpen_whip_10g_away": rng.uniform(0.8, 1.8, n),
        "bullpen_whip_3g_home": rng.uniform(0.8, 1.8, n),
        "bullpen_whip_3g_away": rng.uniform(0.8, 1.8, n),
        "bullpen_pitches_3d_home": rng.uniform(0, 300, n),
        "bullpen_pitches_3d_away": rng.uniform(0, 300, n),
        "bullpen_ip_3d_home": rng.uniform(0, 12, n),
        "bullpen_ip_3d_away": rng.uniform(0, 12, n),
    })
    out = features.add_diff_features(df.copy())
    d = out["bullpen_whip_10g_diff"]
    expected = out["bullpen_whip_10g_home"] - out["bullpen_whip_10g_away"]
    assert np.allclose(d, expected, equal_nan=True)
    assert "bullpen_whip_diff" not in out.columns
    # meltdown twins are within-side products of the family's own factors
    assert np.allclose(
        out["bullpen_meltdown_risk_home"],
        out["bullpen_pitches_3d_home"] * out["bullpen_whip_10g_home"],
        equal_nan=True)
    assert np.allclose(
        out["bullpen_meltdown_risk_away"],
        out["bullpen_pitches_3d_away"] * out["bullpen_whip_10g_away"],
        equal_nan=True)
    assert np.allclose(
        out["bullpen_meltdown_risk_diff"],
        out["bullpen_pitches_diff"] * out["bullpen_whip_10g_diff"],
        equal_nan=True)


# ── availability semantics, exercised through real DuckDB SQL ───────────────

# bullpen_raw with the availability predicate bound as production binds it.
def _bullpen_raw_sql(ledger_ok: bool) -> str:
    arm_out = (features._BP_ARM_UNAVAILABLE_SQL if ledger_ok else "FALSE")
    return f"""
        CREATE TABLE bullpen_raw AS
        WITH reliever_events AS (
            SELECT CAST(p.game_date AS DATE) AS game_date,
                   p.game_pk,
                   p.pitcher,
                   CASE WHEN p.inning_topbot = 'Top' THEN p.home_team
                        ELSE p.away_team END AS fielding_team,
                   p.events
            FROM pitches p
            JOIN starters s ON p.game_pk = s.game_pk
            WHERE (s.home_starter_id IS NULL OR p.pitcher != s.home_starter_id)
              AND (s.away_starter_id IS NULL OR p.pitcher != s.away_starter_id)
              AND p.events IN ('single','double','triple','home_run',
                  'strikeout','strikeout_double_play','walk','hit_by_pitch',
                  'field_out','field_error','fielders_choice',
                  'fielders_choice_out','grounded_into_double_play',
                  'double_play','triple_play','sac_fly','sac_bunt',
                  'sac_fly_double_play','catcher_interf',
                  'batter_interference','force_out',
                  'sacrifice_bunt_double_play')
              AND NOT {arm_out}
        )
        SELECT game_date, game_pk, fielding_team AS team,
            SUM(CASE events
                    WHEN 'field_out' THEN 1
                    WHEN 'strikeout' THEN 1
                    WHEN 'strikeout_double_play' THEN 2
                    WHEN 'grounded_into_double_play' THEN 2
                    WHEN 'double_play' THEN 2
                    WHEN 'triple_play' THEN 3
                    WHEN 'sac_fly' THEN 1
                    WHEN 'sac_bunt' THEN 1
                    WHEN 'fielders_choice_out' THEN 1
                    WHEN 'sac_fly_double_play' THEN 2
                    WHEN 'sacrifice_bunt_double_play' THEN 2
                    ELSE 0 END) / 3.0 AS bullpen_ip,
            SUM(CASE WHEN events IN ('strikeout',
                    'strikeout_double_play') THEN 1 ELSE 0 END) AS bullpen_ks,
            SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS bullpen_bbs,
            SUM(CASE WHEN events IN ('single','double','triple',
                'home_run') THEN 1 ELSE 0 END) AS bullpen_hits
        FROM reliever_events
        GROUP BY game_date, game_pk, fielding_team
    """


def _con_with(relievers: list[dict], ilp: pd.DataFrame | None):
    con = duckdb.connect(database=":memory:")
    con.register("pitches", pd.DataFrame({
        "game_date": pd.to_datetime([r["game_date"] for r in relievers]),
        "game_pk": [r["game_pk"] for r in relievers],
        "pitcher": [r["pitcher"] for r in relievers],
        "inning_topbot": ["Top"] * len(relievers),
        "home_team": [r["team"] for r in relievers],
        "away_team": ["OPP"] * len(relievers),
        "events": ["strikeout"] * len(relievers),
    }))
    # ONE starters row per distinct game_pk — a per-reliever row here would
    # fan the JOIN out and double every arm's innings.
    gp = sorted({r["game_pk"] for r in relievers})
    con.register("starters", pd.DataFrame({
        "game_pk": gp,
        "home_starter_id": [None] * len(gp),
        "away_starter_id": [None] * len(gp),
    }))
    con.register("games", pd.DataFrame({
        "game_pk": gp,
        "home_team": [relievers[0]["team"]] * len(gp),
        "away_team": ["OPP"] * len(gp),
    }))
    if ilp is not None:
        con.register("il_stints_pitchers", ilp)
    return con


def test_unavailable_arm_innings_excluded_from_family_inputs():
    # reliever 101: unavailable strictly between 06-08 and 07-01 → his
    # 06-10..06-14 innings leave bullpen_raw; 202 stays all window.
    relievers = [
        *[{ "game_date": f"2026-06-{d:02d}", "game_pk": 1000 + d,
            "team": "BOS", "pitcher": 101}
          for d in range(10, 15)],
        *[{ "game_date": f"2026-06-{d:02d}", "game_pk": 1000 + d,
            "team": "BOS", "pitcher": 202}
          for d in range(10, 15)],
    ]
    ilp = pd.DataFrame({
        "batter": [101], "il_start": [pd.Timestamp("2026-06-08")],
        "il_end": [pd.Timestamp("2026-07-01")],
    })
    con = _con_with(relievers, ilp)
    con.execute(_bullpen_raw_sql(ledger_ok=True))
    rows = con.execute(
        "SELECT count(*) n, count(DISTINCT game_pk) games, "
        "SUM(bullpen_ks) ks FROM bullpen_raw").fetchone()
    # only the healthy arm survives: 5 games, his 5 Ks, none from 101
    assert rows[0] == 5 and rows[1] == 5 and rows[2] == 5

    # same data without the ledger → pre-availability semantics
    con2 = _con_with(relievers, None)
    con2.execute(_bullpen_raw_sql(ledger_ok=False))
    rows2 = con2.execute(
        "SELECT count(*), SUM(bullpen_ks) FROM bullpen_raw").fetchone()
    assert rows2[0] == 5 and rows2[1] == 10  # both arms' Ks count


def test_placement_and_return_day_innings_stay_available():
    """PIT rule: innings ON the placement date (announcement evening) and
    ON the return date (the appearance that closes the stint) are an
    available arm's — unavailable strictly BETWEEN the stint dates."""
    relievers = [{"game_date": d, "game_pk": i + 1, "team": "BOS",
                  "pitcher": 101}
                 for i, d in enumerate(
                     ["2026-06-08", "2026-06-20", "2026-06-25"])]
    ilp = pd.DataFrame({
        "batter": [101], "il_start": [pd.Timestamp("2026-06-08")],
        "il_end": [pd.Timestamp("2026-06-25")],
    })
    con = _con_with(relievers, ilp)
    con.execute(_bullpen_raw_sql(ledger_ok=True))
    dates = [pd.Timestamp(r[0]) for r in con.execute(
        "SELECT game_date FROM bullpen_raw ORDER BY game_date").fetchall()]
    assert pd.Timestamp("2026-06-08") in dates   # placement day: available
    assert pd.Timestamp("2026-06-25") in dates   # return day: available
    assert pd.Timestamp("2026-06-20") not in dates  # strictly between: out


def test_helper_registers_and_degrades_loudly(tmp_path, caplog):
    """_register_il_stints_pitchers: True with the file, False + warning
    without it — never an exception."""
    import logging
    good = tmp_path / "il_stints_pitchers.parquet"
    pd.DataFrame({"batter": [1], "il_start": [pd.Timestamp("2026-01-01")],
                  "il_end": [pd.NaT]}).to_parquet(good, index=False)
    con = duckdb.connect(database=":memory:")
    with patch_ledger_dir(tmp_path):
        assert features._register_il_stints_pitchers(con) is True
        assert con.execute("SELECT count(*) FROM "
                           "il_stints_pitchers").fetchone()[0] == 1
    missing = tmp_path / "empty"
    missing.mkdir()
    with patch_ledger_dir(missing):
        caplog.set_level(logging.WARNING)
        assert features._register_il_stints_pitchers(con) is False
        assert any("degrades" in r.message for r in caplog.records)


def patch_ledger_dir(d):
    return patch.object(features, "_lineup_base_dir", return_value=d)


def _mk_pbp(tmp_path, name, bulk_days, cameo_days):
    """Pitch-level pbp fixture: a bulk day = 3 innings x 8 pitches (a real
    appearance under the n_innings>=2 / n_pitches>=15 rule); a cameo day =
    1 inning x 8 pitches (token, does not close a stint)."""
    rows = []
    for d, inn in [*((x, 3) for x in bulk_days),
                   *((x, 1) for x in cameo_days)]:
        for i in range(inn):
            for _ in range(8):
                rows.append({"game_date": pd.Timestamp(d), "pitcher": 555,
                             "inning": i + 1, "events": "strikeout",
                             "game_pk": 1})
    p = tmp_path / name
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


# ── over-budget (spent-arm) exclusion (2026-09-30) ─────────────────────

_SPENT_SQL = """
    CREATE OR REPLACE TABLE bp_spent AS
    SELECT o.team, o.pitcher, o.game_date AS game_date,
           h.last_heavy AS heavy_outing
    FROM bp_outing o
    JOIN (
        SELECT team, pitcher, game_date,
               MAX(CASE WHEN n_pitches >= 35 THEN game_date END) OVER (
                   PARTITION BY team, pitcher ORDER BY game_date
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                   AS last_heavy
        FROM bp_outing
    ) h ON h.team = o.team
       AND h.pitcher = o.pitcher
       AND h.game_date = o.game_date
    WHERE h.last_heavy IS NOT NULL
      AND o.game_date > h.last_heavy
      AND o.game_date <= CAST(h.last_heavy AS DATE)
          + INTERVAL '2 DAY'
"""

_SPENT_RAW = """
    SELECT COUNT(*) AS n
    FROM bp_outing o
    WHERE NOT EXISTS (
        SELECT 1 FROM bp_spent sp
        WHERE sp.team = o.team
          AND sp.pitcher = o.pitcher
          AND sp.game_date = CAST(o.game_date AS DATE))
"""


def test_spent_asof_window_and_rest_refresh():
    """AS-OF semantics, 2-day window: a game row is excluded when the arm's
    PRIOR outing (before that row) was >= 35 pitches within 2 days —
    regardless of what came after. Rows thrown by the heavy outing itself
    stay; day 3+ is rested again (arms return 18%+ by day 3, so the window
    is the strict availability fact only)."""
    con = duckdb.connect(database=":memory:")
    con.register("bp_outing", pd.DataFrame({
        "game_date": pd.to_datetime([
            "2026-06-02",                  # arm 7 light: kept
            "2026-06-08",                  # arm 7 HEAVY: kept (own outing)
            "2026-06-09",                  # +1d: excluded
            "2026-06-10",                  # +2d (window edge): excluded
            "2026-06-11",                  # +3d: rested, kept
        ]),
        "game_pk": [1, 2, 3, 4, 5],
        "team": ["BOS"] * 5,
        "pitcher": [7, 7, 7, 7, 7],
        "n_pitches": [10, 50, 10, 10, 10],
    }))
    con.execute(_SPENT_SQL)
    kept = con.execute(_SPENT_RAW).fetchone()[0]
    assert kept == 3  # 06-02, 06-08, 06-11 (first rested day)


def test_spent_boundary_is_35_and_team_scoped():
    con = duckdb.connect(database=":memory:")
    con.register("bp_outing", pd.DataFrame({
        "game_date": pd.to_datetime(
            ["2026-06-01", "2026-06-03", "2026-06-01", "2026-06-03"]),
        "game_pk": [1, 2, 3, 4],
        "team": ["BOS", "BOS", "NYY", "NYY"],
        "pitcher": [9, 9, 9, 9],
        "n_pitches": [50, 10, 50, 10],
    }))
    con.execute(_SPENT_SQL)
    spent = con.execute("SELECT team, game_date FROM bp_spent ORDER BY team")
    rows = spent.fetchall()
    # both teams' 06-03 rows are post-heavy within 4d
    assert len(rows) == 2 and {r[0] for r in rows} == {"BOS", "NYY"}
    kept = con.execute(_SPENT_RAW).fetchone()[0]
    assert kept == 2  # only the two 06-01 heavy outings themselves remain


# ── readiness likelihood (2026-09-30): budget + ready-share semantics ──────

def _day2_con(arms, ilp):
    """Register a minimal bp_outing-shaped table; arms = (day_offset, team,
    pitcher, n_pitches) tuples relative to the reference day 2026-06-10."""
    import datetime
    base = datetime.date(2026, 6, 10)
    con = duckdb.connect(database=":memory:")
    con.register("bp_outing", pd.DataFrame({
        "game_date": [base - datetime.timedelta(days=a[0]) for a in arms],
        "game_pk": [1000 + i for i in range(len(arms))],
        "team": [a[1] for a in arms],
        "pitcher": [a[2] for a in arms],
        "n_pitches": [a[3] for a in arms],
    }))
    if ilp is None:
        ilp = pd.DataFrame({
            "batter": pd.Series([], dtype="int64"),
            "il_start": pd.Series([], dtype="datetime64[ns]"),
            "il_end": pd.Series([], dtype="datetime64[ns]"),
        })
    con.register("il_stints_pitchers", ilp)
    return con

_DAY2_SQL = """
    WITH d2 AS (
        SELECT o.team, o.game_date AS day,
               SUM(o.n_pitches) AS pitches_2d,
               SUM(o.n_pitches * o.ready_p) AS ready_pitches_2d
        FROM (
            SELECT o.game_date, o.team, o.pitcher, o.n_pitches,
                   CASE
                     WHEN EXISTS (SELECT 1 FROM il_stints_pitchers i
                                  WHERE i.batter = o.pitcher
                                    AND o.game_date > CAST(i.il_start AS DATE)
                                    AND (i.il_end IS NULL
                                         OR o.game_date < CAST(i.il_end AS DATE)))
                         THEN 0.0
                     WHEN o.n_pitches < 20 THEN 0.25
                     WHEN o.n_pitches < 35 THEN 0.13
                     ELSE 0.007 END AS ready_p
            FROM bp_outing o) o
        GROUP BY 1, 2)
    SELECT SUM(pitches_2d) AS budget,
           SUM(ready_pitches_2d) / NULLIF(SUM(pitches_2d), 0) / 0.25 AS share
    FROM d2
    WHERE day < DATE '2026-06-10'
      AND day >= DATE '2026-06-10' - INTERVAL 2 DAY
"""


def test_budget_2d_sums_the_prior_two_days_only():
    con = _day2_con([
        (0, "BOS", 101, 10),   # ref day itself: EXCLUDED (no next-day info)
        (1, "BOS", 102, 15),   # yesterday: included
        (2, "BOS", 103, 20),   # two days ago: included
        (3, "BOS", 104, 99),   # three days ago: EXCLUDED
    ], ilp=None)
    row = con.execute(_DAY2_SQL).fetchone()
    # 15 pitches -> light (0.25), 20 -> medium (0.13): calibrated buckets
    assert row[0] == 35
    assert row[1] == pytest.approx((15 * 0.25 + 20 * 0.13) / 35 / 0.25)


def test_ready_share_staircase_and_normalization():
    # all-light (10, 15 pitches) -> share 1.0; all-heavy (40) -> 0.028;
    # one of each mixes proportionally.
    light = _day2_con([(1, "BOS", 1, 10), (1, "BOS", 2, 15)], ilp=None)
    assert light.execute(_DAY2_SQL).fetchone()[1] == pytest.approx(1.0)
    heavy = _day2_con([(1, "BOS", 1, 40)], ilp=None)
    assert heavy.execute(_DAY2_SQL).fetchone()[1] == pytest.approx(
        0.007 / 0.25)
    mix = _day2_con([(1, "BOS", 1, 10), (1, "BOS", 2, 40)], ilp=None)
    expected = (10 * 0.25 + 40 * 0.007) / 50 / 0.25
    assert mix.execute(_DAY2_SQL).fetchone()[1] == pytest.approx(expected)


def test_ready_share_ledger_arm_is_zero_only_as_of_outing_date():
    # arm 7 pitched 06-09 with a stint open 06-01..06-05: HISTORICAL stint
    # must NOT zero him (any-stint matching was the original defect); an
    # arm pitching strictly INSIDE his stint dates gets 0 (and should not
    # exist in reality — the appearance-based reconciliation prevents it).
    ilp = pd.DataFrame({
        "batter": [7, 8],
        "il_start": [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-02")],
        "il_end": [pd.Timestamp("2026-06-05"), pd.Timestamp("2026-06-09")],
    })
    con = _day2_con([
        (1, "BOS", 7, 20),   # 06-09: after his stint closed -> full weight
        (1, "BOS", 8, 20),   # 06-09: strictly after il_end -> full weight
    ], ilp=ilp)
    row = con.execute(_DAY2_SQL).fetchone()
    assert row[1] == pytest.approx(0.13 / 0.25)  # both scored by pitch count


def test_exporter_status_labels_and_degradation():
    ilp = pd.DataFrame({
        "batter": [9],
        "il_start": [pd.Timestamp("2026-06-09")],
        "il_end": [pd.NaT],
    })
    con = _day2_con([
        (1, "BOS", 9, 40),    # inside open stint -> ON_LEDGER
        (1, "BOS", 10, 40),   # heavy -> UNLIKELY
        (1, "BOS", 11, 25),   # mid -> DOUBTFUL
        (1, "BOS", 12, 10),   # light -> LIKELY
    ], ilp=ilp)
    import features as f
    df = f.export_bullpen_availability(con, "2026-06-10")
    s = df.set_index("pitcher").next_day_status
    assert s[9] == "ON_LEDGER" and s[10].startswith("UNLIKELY")
    assert s[11].startswith("DOUBTFUL") and s[12].startswith("LIKELY")
    assert df.set_index("pitcher").p_available[9] == 0.0


def test_pitcher_builder_reconciles_and_splits(tmp_path):
    """The pitcher pipeline: a placed pitcher whose next appearance is a
    real bulk game has his stint closed by that appearance."""
    import build_il_stints as m
    pbp = _mk_pbp(tmp_path, "pbp.parquet",
                  ["2026-06-09"], ["2026-06-01", "2026-06-20"])
    tx = [
        {"description": "BOS placed RHP Ace on the 15-day injured list.",
         "person": {"id": 555}, "date": "2026-06-02"},
    ]
    iv, stats = m.build_pitcher_stints(tx, pbp, pd.Timestamp("2026-06-30"))
    assert stats["reconciled"] is True
    # the 06-09 bulk appearance (3 innings, 24 pitches) closes the stint
    assert len(iv) == 1
    assert iv.iloc[0]["il_end"] == pd.Timestamp("2026-06-09")
    # a token cameo (1 inning, 8 pitches) does NOT close it
    pbp2 = _mk_pbp(tmp_path, "pbp2.parquet",
                   [], ["2026-06-01", "2026-06-09"])
    iv2, _ = m.build_pitcher_stints(tx, pbp2, pd.Timestamp("2026-06-30"))
    assert iv2.iloc[0]["il_end"] is pd.NaT or pd.isna(iv2.iloc[0]["il_end"])


# ── SP staleness gate (2026-09-30): the starting-pitcher analogue ───────────
# Same directive: availability enters the SP stat chain as an IN-FILTER, not
# new features. An appearance whose gap to the pitcher's previous appearance
# is >= _SP_STALE_GAP_DAYS AND overlaps a generalized-availability stint is
# STALE — every SP stat sourced from that appearance ships NULL, and the
# upcoming slate withholds a stale return starter's carried line entirely.


def _mk_ledger(tmp_path, rows):
    p = tmp_path / "il_stints_pitchers.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def test_sp_gate_flags_post_stint_return_not_normal_turns(tmp_path):
    # pitcher 101: 06-01 app, stint 06-03..06-25, 06-20 app -> gap 19d
    # overlapping the stint -> STALE. Pitcher 202: identical calendar, no
    # stint -> the 19d rest is NOT an availability fact. Pitcher 303: open
    # stint from 06-05 but only a 5d turn (06-01, 06-06) -> within normal
    # rotation band, not gated.
    from datetime import date as _d  # noqa: F401  (symmetry with slate tests)
    games = pd.DataFrame({
        "game_date": pd.to_datetime(
            ["2026-06-01", "2026-06-20", "2026-06-01", "2026-06-20",
             "2026-06-01", "2026-06-06"]),
        "game_pk": [1, 2, 3, 4, 5, 6],
        "pitcher": [101, 101, 202, 202, 303, 303],
    })
    ilp = pd.DataFrame({
        "batter": [101, 303],
        "il_start": [pd.Timestamp("2026-06-03"), pd.Timestamp("2026-06-05")],
        "il_end": [pd.Timestamp("2026-06-25"), pd.NaT],
    })
    _mk_ledger(tmp_path, ilp)
    con = duckdb.connect(database=":memory:")
    with patch_ledger_dir(tmp_path):
        con.register("pgs_reg", games)
        con.execute("CREATE TABLE pitcher_game_stats AS "
                    "SELECT CAST(game_date AS DATE) AS game_date, game_pk, "
                    "pitcher FROM pgs_reg")
        assert features._register_sp_staleness_gate(con) is True
        rows = {r[0]: r[1] for r in con.execute(
            "SELECT game_pk, sp_stale FROM pitcher_stale").fetchall()}
    assert rows[2] is True    # 101 return: gap 19d across the stint
    assert rows[1] is False   # 101 pre-stint start stays available
    assert rows[3] is False and rows[4] is False  # rest control: no stint
    assert rows[5] is False and rows[6] is False


def test_sp_gate_missing_ledger_degrades_loudly_and_ungated(tmp_path, caplog):
    import logging
    games = pd.DataFrame({
        "game_date": pd.to_datetime(["2026-06-01", "2026-06-20"]),
        "game_pk": [1, 2], "pitcher": [101, 101],
    })
    con = duckdb.connect(database=":memory:")
    with patch_ledger_dir(tmp_path):  # dir WITHOUT the ledger parquet
        caplog.set_level(logging.WARNING)
        con.register("pgs_reg", games)
        con.execute("CREATE TABLE pitcher_game_stats AS "
                    "SELECT CAST(game_date AS DATE) AS game_date, game_pk, "
                    "pitcher FROM pgs_reg")
        assert features._register_sp_staleness_gate(con) is False
        assert any("degrades" in r.message or "missing" in r.message
                   for r in caplog.records)
        # empty gate table: every consumer LEFT JOIN yields NULL sp_stale,
        # i.e. the exact pre-gate (ungated) semantics
        n, stale = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(CASE WHEN sp_stale THEN 1 "
            "ELSE 0 END), 0) FROM pitcher_stale").fetchone()
    assert n == 0 and stale == 0


def test_slate_withholds_stale_return_starter(tmp_path, caplog):
    """End-to-end: the upcoming slate must NOT serve a stale return starter's
    pre-stint carried line (the latest-non-null picker would), while a
    healthy announced starter carries normally."""
    import logging
    from data_ingestion import build_upcoming_slate
    from datetime import date
    # pbp frame: 101's latest app is a 16d-gap return inside his stint;
    # 202's latest app is a normal 6d turn.
    pbp = pd.DataFrame({
        "game_date": pd.to_datetime(
            ["2026-05-20", "2026-06-05", "2026-06-20", "2026-06-26"]),
        "game_pk": [1, 2, 3, 4],
        "pitcher": [101, 101, 202, 202],
        "events": ["strikeout"] * 4,
    })
    ilp = pd.DataFrame({
        "batter": [101],
        "il_start": [pd.Timestamp("2026-05-22")],
        "il_end": [pd.NaT],  # returned via paperwork the feed never filed
    })
    _mk_ledger(tmp_path, ilp)
    hist = pd.DataFrame([{
        "game_date": pd.Timestamp("2026-06-26"),
        "game_id": "20260626_AWAY@HOME", "home_team": "HOME",
        "away_team": "AWAY", "home_win": 1.0, "home_score": 5,
        "away_score": 3, "home_starter_id": 101, "away_starter_id": 202,
        "sp_k9_home": 8.0, "sp_k9_away": 7.0,
        "sp_era_home": 3.0, "sp_era_away": 4.0,
    }])
    schedule = pd.DataFrame([{
        "game_id": "20260628_AWAY@HOME", "game_date": "2026-06-28",
        "start_time_utc": "2026-06-28T18:00:00", "home_team": "HOME",
        "away_team": "AWAY", "venue": "Test Park", "sp_id_home": 101,
        "sp_id_away": 202, "sp_name_home": "Return Ace",
        "sp_name_away": "Healthy Arm",
    }])
    schedule["game_date"] = pd.to_datetime(schedule["game_date"])
    schedule["start_time_utc"] = pd.to_datetime(schedule["start_time_utc"])
    with patch_ledger_dir(tmp_path):
        caplog.set_level(logging.WARNING)
        slate = build_upcoming_slate(hist, date(2026, 6, 28), pbp_df=pbp,
                                     schedule_df=schedule)
    assert len(slate) == 1
    # stale return: SP columns withheld (NaN), team evidence still priced
    assert pd.isna(slate.loc[0, "sp_k9_home"])
    assert pd.isna(slate.loc[0, "sp_era_home"])
    # healthy starter: carried line intact
    assert slate.loc[0, "sp_k9_away"] == 7.0
    assert slate.loc[0, "sp_era_away"] == 4.0
    assert any("stale" in r.message.lower() for r in caplog.records)
    # observability: ONE summary entry for the gated side — the generic
    # unmapped loop must not double-count a mapped-and-gated starter nor
    # misreport him as "absent or unmappable" (2026-09-29 run logged
    # "2 of 8 starter slots" for a single gated side)
    summary = [r.message for r in caplog.records
               if "Slate pitcher resolution" in r.message]
    assert summary == [
        "Slate pitcher resolution: 1 of 2 starter slots unresolved — those "
        "sides price without SP/exp2 features "
        "(20260628_AWAY@HOME:home (stale return))"], summary
    assert not [r for r in caplog.records
                if "absent or unmappable" in r.message]


def test_no_pbp_frame_degrades_slate_gate_inertly(caplog):
    """Synthetic-history runs and pbp-less tests must keep working with the
    gate loudly inert (no gating, no exception, slate still built)."""
    import logging
    from data_ingestion import build_upcoming_slate
    from datetime import date
    hist = pd.DataFrame([{
        "game_date": pd.Timestamp("2026-06-26"),
        "game_id": "20260626_AWAY@HOME", "home_team": "HOME",
        "away_team": "AWAY", "home_win": 1.0, "home_score": 5,
        "away_score": 3, "home_starter_id": 101, "away_starter_id": 202,
        "sp_k9_home": 8.0, "sp_k9_away": 7.0,
    }])
    schedule = pd.DataFrame([{
        "game_id": "20260628_AWAY@HOME", "game_date": "2026-06-28",
        "start_time_utc": "2026-06-28T18:00:00", "home_team": "HOME",
        "away_team": "AWAY", "venue": "Test Park", "sp_id_home": 101,
        "sp_id_away": 202, "sp_name_home": "H", "sp_name_away": "A",
    }])
    schedule["game_date"] = pd.to_datetime(schedule["game_date"])
    schedule["start_time_utc"] = pd.to_datetime(schedule["start_time_utc"])
    caplog.set_level(logging.WARNING)
    slate = build_upcoming_slate(hist, date(2026, 6, 28),
                                 schedule_df=schedule)
    assert len(slate) == 1
    assert slate.loc[0, "sp_k9_home"] == 8.0  # carried: gate did not fire
    assert any("SP slate staleness gate unavailable" in r.message
               for r in caplog.records)


def test_sp_gate_is_infilter_only_and_wired_upstream():
    """No new serving columns; the gate table is built before the first SP
    window and consumed by all three per-pitcher sources; cleanup drops it."""
    cols = training.MONEYLINE_FEATURE_COLS
    assert len(cols) == 101
    assert not [c for c in cols if "stale" in c or "gap" in c]
    src = (BACKEND / "features.py").read_text(encoding="utf-8")
    # orchestrator: gate built from pitcher_game_stats BEFORE pitcher_shifted
    assert src.index("_register_sp_staleness_gate(con)") < src.index(
        "CREATE TABLE pitcher_shifted AS")
    # all three per-pitcher sources consume the gate
    for tbl in ("pitcher_season_features", "pitcher_features",
                "pitcher_stuff"):
        seg = src[src.index(f"CREATE TABLE {tbl} AS"):]
        assert "LEFT JOIN pitcher_stale st" in seg[:2000], tbl
        assert "CASE WHEN st.sp_stale THEN NULL" in seg[:2000], tbl
    # cleanup list drops the gate table
    assert '"pitcher_stale"' in src


# ── Bullpen opportunity-weighted shrinkage (2026-09-30, adopted by owner) ──
# NBA player_ts structural alignment: rate windows shrink toward the PIT
# league reliever prior with k = 20% of the mean reliever-season pitch count
# (data-derived, floor 20). Volumes never shrink. Degenerate prior ships raw.


def _shrink_chain(con, features_src, k_pitches: float | None = None,
                  k_ip: float | None = None, lg_whip: float | None = None,
                  lg_era: float | None = None):
    """Build the production shrink-prior + rolling/season tables on ``con``
    from the LIFTED shipped SQL. k_pitches None = keep the shipped data-
    derived prior; pass (k_pitches, k_ip, lg_whip, lg_era) to override the
    prior row analytically after the shipped derivation runs."""
    def lift(name):
        j0 = features_src.index(f"CREATE TABLE {name} AS")
        return features_src[j0:features_src.index('"""', j0)]
    # The prior SQL reads bp_outing + the two filters; stub them minimally
    # (empty ledger/spent = NOT EXISTS trivially true) so the SHIPPED
    # derivation (population alignment, COALESCE floor, GREATEST) executes.
    if not con.execute("SELECT COUNT(*) FROM duckdb_tables() "
                       "WHERE table_name = 'bp_outing'").fetchone()[0]:
        con.register("bo_reg", pd.DataFrame({
            "game_date": pd.to_datetime(["2026-04-01"] * 2),
            "game_pk": [1, 2], "team": ["TB", "LAD"],
            "pitcher": [11, 22], "n_pitches": [400.0, 500.0],
        }))
        con.execute("CREATE TABLE bp_outing AS SELECT "
                    "CAST(game_date AS DATE) AS game_date, game_pk, team, "
                    "pitcher, n_pitches FROM bo_reg")
        con.execute("CREATE TABLE il_stints_pitchers AS SELECT "
                    "CAST(NULL AS BIGINT) AS batter, "
                    "CAST(NULL AS TIMESTAMP) AS il_start, "
                    "CAST(NULL AS TIMESTAMP) AS il_end "
                    "WHERE FALSE")
        con.execute("CREATE TABLE bp_spent AS SELECT "
                    "CAST(NULL AS VARCHAR) AS team, "
                    "CAST(NULL AS BIGINT) AS pitcher, "
                    "CAST(NULL AS DATE) AS game_date, "
                    "CAST(NULL AS DATE) AS heavy_outing WHERE FALSE")
    con.execute(lift("bp_shrink_prior")
                 .replace("{_k_arm_out}",
                          features._BP_LEDGER_EXCLUDE_ROW_SQL.format(
                              alias="o", date="o.game_date"))
                 .replace("{_BP_MIN_SEASON_PITCHES}", "30")
                 .replace("{_BP_K_FLOOR}", "20.0")
                 .replace("{_BP_SHRINK_FRACTION}", "0.20")
                 .replace("{_BP_PITCHES_PER_IP}", "15.5"))
    if k_pitches is not None:
        con.execute(f"CREATE OR REPLACE TABLE bp_shrink_prior AS SELECT "
                    f"season, game_date, {k_pitches} AS k_pitches, "
                    f"{k_ip} AS k_ip, {lg_whip} AS lg_whip, "
                    f"{lg_era} AS lg_era FROM bp_shrink_prior")
    con.execute(lift("bullpen_rolling"))
    con.execute(lift("bullpen_season"))


def test_shrinkage_pulls_thin_window_toward_prior_and_leaves_thick_one():
    src = (BACKEND / "features.py").read_text(encoding="utf-8")
    con = duckdb.connect(database=":memory:")
    # ONE shifted row per team at the prior's date: the w10/w3 window
    # arithmetic is pre-existing tested code; what is under test is the
    # BLEND. With a single row each window collapses to that row, so the
    # expected value is analytic. TB: 2 IP window (thin, raw WHIP 2.50).
    # LAD: 60 IP window (thick, raw 1.50). Prior: k = 6 IP, lg_whip 1.30.
    con.register("bs_reg", pd.DataFrame({
        "game_date": pd.to_datetime(["2026-04-03"] * 2),
        "game_pk": [1, 2], "team": ["TB", "LAD"],
        "_s_bbs": [2.0, 30.0], "_s_hits": [3.0, 60.0],
        "_s_ip": [2.0, 60.0], "_s_runs": [1.0, 30.0],
    }))
    con.execute("CREATE TABLE bullpen_shifted AS "
                "SELECT CAST(game_date AS DATE) AS game_date, game_pk, "
                "team, _s_bbs, _s_hits, _s_ip, _s_runs FROM bs_reg")
    # bullpen_season re-derives its own LAGs from bullpen_raw: minimal rows.
    con.register("br_reg", pd.DataFrame({
        "game_date": pd.to_datetime(["2026-04-03"] * 2),
        "game_pk": [1, 2], "team": ["TB", "LAD"],
        "bullpen_bbs": [2.0, 30.0], "bullpen_hits": [3.0, 60.0],
        "bullpen_ip": [2.0, 60.0], "bullpen_runs": [1.0, 30.0],
    }))
    con.execute("CREATE TABLE bullpen_raw AS "
                "SELECT CAST(game_date AS DATE) AS game_date, game_pk, "
                "team, bullpen_bbs, bullpen_hits, bullpen_ip, bullpen_runs "
                "FROM br_reg")
    # analytic prior override after the shipped derivation: k = 6 IP,
    # lg_whip 1.30, lg_era 4.20
    _shrink_chain(con, src, k_pitches=93.0, k_ip=6.0,
                  lg_whip=1.30, lg_era=4.20)
    out = con.execute("SELECT team, bullpen_whip_3g, bullpen_whip_10g "
                      "FROM bullpen_rolling ORDER BY team").fetchall()
    got = {t: (w3, w10) for t, w3, w10 in out}
    # TB: raw = (2+3)/2 = 2.50; w = 2/(2+6) = 0.25
    exp_tb = 0.25 * 2.50 + 0.75 * 1.30
    assert abs(got["TB"][0] - exp_tb) < 1e-9
    assert abs(got["TB"][1] - exp_tb) < 1e-9
    # LAD: raw = 90/60 = 1.50; w = 60/66
    exp_lad = (60.0 / 66.0) * 1.50 + (6.0 / 66.0) * 1.30
    assert abs(got["LAD"][0] - exp_lad) < 1e-9
    # direction: the thin window moved toward the prior, the thick barely moved
    assert got["TB"][0] < 2.50 and abs(exp_lad - 1.50) < 0.02


def test_shrinkage_volumes_and_degenerate_prior_paths():
    """k=NULL/0 ships raw rates (CASE guard); volume columns carry no CASE
    (never shrunk); cleanup drops bp_shrink_prior; the prior's k derivation
    is population-aligned and floor-guarded."""
    src = (BACKEND / "features.py").read_text(encoding="utf-8")
    con = duckdb.connect(database=":memory:")
    con.register("bs_reg", pd.DataFrame({
        "game_date": pd.to_datetime(["2026-04-03"] * 2),
        "game_pk": [1, 2], "team": ["TB", "LAD"],
        "_s_bbs": [2.0, 30.0], "_s_hits": [3.0, 60.0],
        "_s_ip": [2.0, 60.0], "_s_runs": [1.0, 30.0],
    }))
    con.execute("CREATE TABLE bullpen_shifted AS "
                "SELECT CAST(game_date AS DATE) AS game_date, game_pk, "
                "team, _s_bbs, _s_hits, _s_ip, _s_runs FROM bs_reg")
    # degenerate prior override: k_ip = 0 -> the CASE guard ships raw rates
    con.register("br_reg", pd.DataFrame({
        "game_date": pd.to_datetime(["2026-04-03"] * 2),
        "game_pk": [1, 2], "team": ["TB", "LAD"],
        "bullpen_bbs": [2.0, 30.0], "bullpen_hits": [3.0, 60.0],
        "bullpen_ip": [2.0, 60.0], "bullpen_runs": [1.0, 30.0],
    }))
    con.execute("CREATE TABLE bullpen_raw AS "
                "SELECT CAST(game_date AS DATE) AS game_date, game_pk, "
                "team, bullpen_bbs, bullpen_hits, bullpen_ip, bullpen_runs "
                "FROM br_reg")
    _shrink_chain(con, src, k_pitches=0.0, k_ip=0.0,
                  lg_whip=1.30, lg_era=4.20)
    row = con.execute("SELECT bullpen_whip_3g, bullpen_whip_10g "
                      "FROM bullpen_rolling WHERE team = 'TB'").fetchone()
    # single row: both windows collapse to raw (2+3)/2 = 2.50, and the
    # k_ip = 0 guard ships them UNSHRUNK despite lg_whip = 1.30
    assert abs(row[0] - 2.50) < 1e-9 and abs(row[1] - 2.50) < 1e-9
    # volumes never shrink: no CASE wraps the workload expressions
    roll = src[src.index("CREATE TABLE bullpen_rolling AS"):
               src.index('"""', src.index("CREATE TABLE bullpen_rolling AS"))]
    assert "bullpen_pitches" not in roll and "budget" not in roll
    # cleanup drops the prior table
    assert '"bp_shrink_prior"' in src
    # k derivation: population-aligned (ledger + spent filters) + floor —
    # the ledger predicate is bound via the conditional fragment, never a
    # hard table reference (the 2026-09-30 sweep deletion turned a hard
    # reference into a CatalogException)
    pr = src[src.index("CREATE TABLE bp_shrink_prior AS"):
             src.index('"""', src.index("CREATE TABLE bp_shrink_prior AS"))]
    assert "{_k_arm_out}" in pr and "bp_spent" in pr
    assert "COALESCE" in pr and "GREATEST" in pr
    assert "il_stints_pitchers" in features._BP_LEDGER_EXCLUDE_ROW_SQL
    assert "il_stints_pitchers" in features._BP_LEDGER_EXCLUDE_ASOF_SQL
    d2 = src[src.index("CREATE TABLE bp_day2 AS"):
             src.index('"""', src.index("CREATE TABLE bp_day2 AS"))]
    assert "{_d2_ledger}" in d2


def test_bp_fstring_constants_resolve():
    """Every {_BP_*} placeholder in features.py f-strings must exist as a
    module-level constant, with the production-calibrated values pinned.

    Regression guard for the 2026-09-30 Kaggle failure: the spent-window
    tighten commit deleted _BP_SPENT_PITCHES while editing the comment
    block above it. imports and py_compile stayed green (the f-string only
    evaluates the name when _build_game_level RUNS), and no test executed
    the SQL (smokes inline constants manually) — so only the production
    build could catch it. This test catches it locally instead."""
    src = (BACKEND / "features.py").read_text(encoding="utf-8")
    placeholders = set(re.findall(r"\{_BP_[A-Z_0-9]+\}", src))
    assert placeholders, "expected _BP_* f-string placeholders in features.py"
    missing = [p for p in sorted(placeholders)
               if not hasattr(features, p[1:-1])]
    assert not missing, (
        f"f-string placeholders without module constants: {missing}")
    # the production calibration values themselves, so a silent semantic
    # drift (not just a deletion) fails here too
    assert features._BP_SPENT_PITCHES == 35
    assert features._BP_SPENT_LOOKBACK_DAYS == 2


def test_pitcher_availability_ledger_is_a_local_runtime_cache(
        tmp_path, monkeypatch):
    """The ledger is a LOCAL runtime cache — rebuilt by Phase 1.5 of every
    daily run, never shipped as a GitHub artifact — and every consumer
    degrades LOUDLY without it — never a CatalogException.

    2026-09-30 (rev 1): the retention sweep deleted the ledger the repo
    shipped; the guard then pinned the file as a committed runtime input.
    Same-day (rev 2), the owner moved the ledgers OUT of the repo entirely:
    master_pipeline Phase 1.5 rebuilds them every run under MLB_IL_STINTS_DIR
    (outside the git repo), Phase 5 staging skips the four names, and git
    ignores them. A fresh clone therefore has NO ledger file — and that is
    correct, because the run rebuilds it before Phase 2-3 consumes it. This
    test now pins the new contract instead of a committed blob: the rebuild
    must be wired into the pipeline, staging must skip the names, the env
    resolver must work, and the two formerly-unguarded statements must still
    execute on a ledger-less connection (bp_day2's staircase runs unfiltered;
    the shrink prior's k derives from the unfiltered population; both loudly
    degraded)."""
    mp_src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    assert "build_il_stints import main as _build_il_stints" in mp_src, (
        "Phase 1.5 must rebuild the availability ledgers every run — a clone "
        "without the file and without the rebuild would ship degraded "
        "availability features silently")
    assert "MLB_IL_STINTS_DIR" in mp_src
    # Phase 5 staging skips the four cache names (belt-and-suspenders; the
    # env relocation already keeps them out of data_delivery).
    assert "rel_local in _il_cache_names" in mp_src
    for _name in ("il_stints.parquet", "il_stints.meta.json",
                  "il_stints_pitchers.parquet", "il_stints_pitchers.meta.json"):
        assert f'"{_name}"' in mp_src, _name
    # the resolver: env wins, legacy data_delivery is the fallback
    monkeypatch.setenv("MLB_IL_STINTS_DIR", str(tmp_path))
    assert features.il_stints_dir() == tmp_path
    monkeypatch.delenv("MLB_IL_STINTS_DIR", raising=False)
    assert features.il_stints_dir() == BACKEND.parent / "data_delivery"

    # every ledger reference must be conditional: with NO ledger table
    # registered, the formerly-unguarded statements still execute
    src = (BACKEND / "features.py").read_text(encoding="utf-8")

    def lift(name):
        j0 = src.index(f"CREATE TABLE {name} AS")
        return src[j0:src.index('"""', j0)]

    con = duckdb.connect(database=":memory:")
    con.register("bo_reg", pd.DataFrame({
        "game_date": pd.to_datetime(["2026-06-08", "2026-06-09"]),
        "game_pk": [1, 2], "team": ["BOS", "BOS"],
        "pitcher": [101, 101], "n_pitches": [400.0, 30.0],
    }))
    con.execute("CREATE TABLE bp_outing AS SELECT "
                "CAST(game_date AS DATE) AS game_date, game_pk, team, "
                "pitcher, n_pitches FROM bo_reg")
    con.register("br_reg", pd.DataFrame({
        "game_date": pd.to_datetime(["2026-06-09"]),
        "game_pk": [2], "team": ["BOS"],
        "bullpen_bbs": [2.0], "bullpen_hits": [3.0],
        "bullpen_ip": [2.0], "bullpen_runs": [1.0],
    }))
    con.execute("CREATE TABLE bullpen_raw AS SELECT "
                "CAST(game_date AS DATE) AS game_date, game_pk, team, "
                "bullpen_bbs, bullpen_hits, bullpen_ip, bullpen_runs "
                "FROM br_reg")
    con.register("g_reg", pd.DataFrame({
        "game_date": pd.to_datetime(["2026-06-10", "2026-06-10"]),
        "game_pk": [10, 10],
        "home_team": ["BOS", "NYY"], "away_team": ["NYY", "BOS"],
    }))
    con.execute("CREATE TABLE game_days AS SELECT DISTINCT game_pk, "
                "CAST(game_date AS DATE) AS gd, home_team, away_team "
                "FROM g_reg")
    # the ledger-conditional fragments bind as FALSE when _pitchers_ok is
    # False — exactly what features.py interpolates without a ledger
    _k_arm_out = "FALSE"
    _d2_ledger = "FALSE"
    con.execute("CREATE OR REPLACE TABLE bp_shrink_prior AS "
                "WITH arm_season AS ("
                "  SELECT pitcher, EXTRACT(YEAR FROM game_date) AS season, "
                "  SUM(n_pitches) AS pitches FROM bp_outing "
                "  WHERE " + _k_arm_out + " GROUP BY 1,2 "
                "  HAVING SUM(n_pitches) >= 30), "
                "k AS (SELECT GREATEST(20.0, 0.20 * COALESCE("
                "    (SELECT AVG(pitches) FROM arm_season), 20.0)) AS k_pitches, "
                "  GREATEST(20.0, 0.20 * COALESCE("
                "    (SELECT AVG(pitches) FROM arm_season), 20.0)) / 15.5 "
                "  AS k_ip FROM (SELECT 1) anchor) "
                "SELECT k_pitches, k_ip FROM k")
    krow = con.execute("SELECT k_pitches, k_ip FROM bp_shrink_prior") \
        .fetchone()
    assert krow == (20.0, 20.0 / 15.5)  # floor k via the COALESCE path
    con.execute(f"""
        CREATE TABLE bp_day2 AS
        WITH d2 AS (
            SELECT g.game_pk, o.team,
                   SUM(o.n_pitches) AS pitches_2d,
                   SUM(o.n_pitches * o.ready_p) AS ready_pitches_2d
            FROM (
                SELECT o.game_date, o.team, o.pitcher, o.n_pitches,
                       CASE WHEN {_d2_ledger} THEN 0.0
                            WHEN o.n_pitches < 20 THEN 0.25
                            WHEN o.n_pitches < 35 THEN 0.13
                            ELSE 0.007 END AS ready_p
                FROM bp_outing o) o
            JOIN game_days g
              ON (o.team = g.home_team OR o.team = g.away_team)
             AND o.game_date = g.gd - INTERVAL 1 DAY
            GROUP BY 1, 2)
        SELECT g2.gd AS ref_day, g2.game_pk, g2.home_team, g2.away_team,
               hh.pitches_2d AS home_pitches_2d,
               hh.ready_pitches_2d AS home_ready_2d
        FROM game_days g2
        LEFT JOIN d2 hh ON hh.game_pk = g2.game_pk
                       AND hh.team = g2.home_team
    """)
    n, pitches, ready = con.execute(
        "SELECT COUNT(*), SUM(home_pitches_2d), SUM(home_ready_2d) "
        "FROM bp_day2").fetchone()
    # shipped semantics (corrected 2026-09-30): d2 groups per (game_pk,
    # side) and each outing joins only its single consuming game-day, so
    # the 06-09 outing (30 pitches) counts ONCE per game — its staircase
    # ran UNFILTERED (no ledger -> FALSE predicate, no 0.0 override):
    # 30 * 0.13 = 3.9 ready. The fixture's two game rows are BOS-home
    # mirrors, so the per-game budget of 30/3.9 legitimately appears on
    # both rows (SUM = 60.0/7.8). The OLD per-(team, day) grouping would
    # have double-counted the outing INTO one 60-pitch group; the
    # double-count detector lives in
    # test_bp_day2_no_doubleheader_fan_and_no_double_count.
    assert n == 2 and pitches == 60.0 and abs(ready - 7.8) < 1e-6


def test_bp_day2_no_doubleheader_fan_and_no_double_count():
    """Execution regression for the 2026-09-30 game_level fan (production
    shipped 164 game_pks x 4 duplicate rows; +0.0059 walk-forward logloss).

    Two stacked bugs, both fixed in bp_day2/bp_fatigue and both asserted
    here against the LIFTED production SQL on a same-opponent doubleheader
    fixture (pk 100/101 on 06-10, BOS home both legs):
      1. FAN: bp_fatigue joined bp_day2 on (ref_day, team) — both DH legs
         matched both bp_day2 rows per side, fanning game_level x4.
         Fix: join by game_pk (+ side).
      2. DOUBLE-COUNT: the inner d2 window joined each outing to EVERY
         following game in its prior-2-day window and grouped per
         (team, outing-date), so a 30+20 pitch prior day shipped as 100
         (x2 here, x3 before a twin bill). Fix: scope each outing to its
         single consuming game-day (gd-1) and group per (game_pk, team)."""
    src = (BACKEND / "features.py").read_text(encoding="utf-8")

    def lift(name, start):
        j0 = src.index(f"CREATE TABLE {name} AS", start)
        return src[j0:src.index('\"\"\"', j0)]

    con = duckdb.connect(database=":memory:")
    # Same-opponent twin bill 06-10 + a BOS game 06-09 (feeds the DH's
    # prior-day bullpen state) + a BOS game 06-11 (back-to-back consumer).
    con.register("p_reg", pd.DataFrame({
        "game_pk": [90, 100, 101, 110],
        "game_date": pd.to_datetime(
            ["2026-06-09", "2026-06-10", "2026-06-10", "2026-06-11"]),
        "home_team": ["BOS"] * 4, "away_team": ["NYY"] * 4,
        "inning": [1] * 4, "inning_topbot": ["Top"] * 4,
        "pitcher": [700] * 4, "events": ["strikeout"] * 4,
        "at_bat_number": [1] * 4, "pitch_number": [1] * 4,
    }))
    con.execute("CREATE TABLE pitches AS SELECT CAST(game_date AS DATE) AS "
                "game_date, game_pk, home_team, away_team, inning, "
                "inning_topbot, pitcher, events, at_bat_number, pitch_number "
                "FROM p_reg")
    con.execute("CREATE TABLE starters AS SELECT game_pk, 700 AS "
                "home_starter_id, 800 AS away_starter_id FROM pitches")
    con.register("bo_reg", pd.DataFrame({
        "game_date": pd.to_datetime(
            ["2026-06-09", "2026-06-09", "2026-06-10"]),
        "game_pk": [90, 90, 100], "team": ["BOS"] * 3,
        "pitcher": [900, 901, 900], "n_pitches": [30.0, 20.0, 40.0],
    }))
    con.execute("CREATE TABLE bp_outing AS SELECT "
                "CAST(game_date AS DATE) AS game_date, game_pk, team, "
                "pitcher, n_pitches FROM bo_reg")

    # bp_day2 with the ledger fragment bound FALSE (no ledger table)
    con.execute(lift("bp_daily", 0).replace("{_BP_SPENT_PITCHES}", "35"))
    d2_sql = lift("bp_day2", 0).replace("{_d2_ledger}", "FALSE")
    con.execute(d2_sql)
    con.execute(lift("bp_fatigue", 0))

    # Fix 2: one row per consuming game_pk (every game, LEFT JOIN from the
    # games list — pk 90 has no prior-day outing -> NULLs); the 06-09 pair
    # (30 + 20 pitches) counts ONCE per DH leg. Staircase: 30 -> 0.13,
    # 20 -> 0.13 (the <20 tier is 0.25), 40 -> 0.007.
    rows = con.execute("SELECT game_pk, home_pitches_2d, home_ready_2d, "
                       "away_pitches_2d FROM bp_day2 "
                       "ORDER BY game_pk").fetchall()
    assert rows == [
        (90, None, None, None),
        (100, 50.0, 30 * 0.13 + 20 * 0.13, None),
        (101, 50.0, 30 * 0.13 + 20 * 0.13, None),
        (110, 40.0, 40 * 0.007, None),
    ], f"bp_day2 fanned or double-counted: {rows}"

    # Fix 1: exactly one bp_fatigue row per game_pk — the two DH legs keep
    # IDENTICAL prior-day budgets but remain distinct rows (no x4 fan).
    fat = con.execute(
        "SELECT game_pk, bullpen_budget_2d_home, bp_ready_share_home, "
        "bullpen_budget_2d_away FROM bp_fatigue ORDER BY game_pk"
    ).fetchall()
    assert [f[0] for f in fat] == [90, 100, 101, 110], \
        f"bp_fatigue row set fanned: {fat}"
    by_pk = {f[0]: f for f in fat}
    assert by_pk[100][1] == 50.0 and by_pk[101][1] == 50.0
    assert abs(by_pk[100][2] - (6.5 / 50.0 / 0.25)) < 1e-9
    assert by_pk[100][1] == by_pk[101][1]
    # back-to-back consumer (06-11) sees ONLY the 06-10 outing (40 pitches)
    # — the 06-09 pair is outside its single consuming game-day slice.
    assert by_pk[110][1] == 40.0, f"double-count survived: {by_pk[110]}"
    # away side stays NULL on this fixture (NYY never consumed a home row)
    assert all(f[3] is None for f in fat), f"away side leaked: {fat}"


def test_decided_frame_dup_tripwire_present():
    """master_pipeline must refuse to train on a fanned decided frame.

    get_decided_frame's Rule 3 dedups by game_pk (latest game_date wins),
    which would SILENTLY collapse a fanned frame while keeping whichever
    duplicate copy sorts last — after the daily Elo/records enrichment
    that is the leaky post-game-snapshot copy. The tripwire checks the
    RAW frame instead and raises before training."""
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    assert "duplicated game_pk rows" in src
    assert "refusing to train on it" in src
    tw = src[src.index("_decided_raw = games"):
             src.index("refusing to train on it")]
    assert 'games["home_win"].notna()' in tw, (
        "tripwire must inspect the RAW decided rows, not the "
        "Rule-3-deduped snapshot")


def test_recent_pitcher_era_windows_and_prior_league_are_point_in_time():
    """Older pitcher evidence excludes the recent five; league prior excludes
    every game on the current calendar date."""
    src = (BACKEND / "features.py").read_text(encoding="utf-8")

    def lift(name):
        j0 = src.index(f"CREATE TABLE {name} AS")
        statement = src[j0:src.index('"""', j0)]
        return statement.replace(
            "{_SP_ERA_5G_SHRINK_IP}", str(features._SP_ERA_5G_SHRINK_IP))

    con = duckdb.connect(database=":memory:")
    shifted = pd.DataFrame({
        "game_date": pd.date_range("2026-01-01", periods=7),
        "game_pk": range(1, 8),
        "pitcher": [101] * 7,
        "_s_runs": [None, 2.0, 0.0, 1.0, 3.0, 0.0, 2.0],
        "_s_ip": [None, 3.0, 6.0, 3.0, 6.0, 3.0, 6.0],
        "_s_ks": [None, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
    })
    con.register("shifted_fixture", shifted)
    con.execute("CREATE TABLE pitcher_shifted AS SELECT * FROM shifted_fixture")
    con.execute(lift("pitcher_5g_rolling"))
    recent_runs, recent_ip, older_runs, older_ip = con.execute("""
        SELECT _roll5_runs, _roll5_ip, _older_runs, _older_ip
        FROM pitcher_5g_rolling WHERE game_pk = 7
    """).fetchone()
    assert (recent_runs, older_runs) == (6.0, 2.0)
    assert (recent_ip, older_ip) == (24.0, 3.0)

    raw = pd.DataFrame({
        "game_date": pd.to_datetime([
            "2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02",
            "2026-01-03",
        ]),
        "runs": [1.0, 2.0, 0.0, 6.0, 2.0],
        "ip": [3.0, 6.0, 3.0, 6.0, 3.0],
    })
    con.register("raw_fixture", raw)
    con.execute("CREATE TABLE pitcher_game_stats AS SELECT * FROM raw_fixture")
    con.execute(lift("pitcher_era_league"))
    league = dict(con.execute("""
        SELECT game_date, league_era_prior FROM pitcher_era_league
    """).fetchall())
    assert league[pd.Timestamp("2026-01-01").date()] is None
    assert league[pd.Timestamp("2026-01-02").date()] == pytest.approx(3.0)
    # Jan 2 contributes 6 runs / 9 IP; Jan 3 sees only the prior dates.
    assert league[pd.Timestamp("2026-01-03").date()] == pytest.approx(4.5)


def test_gated_sp_tables_execute_end_to_end(tmp_path):
    """The three gate-consuming SP statements must EXECUTE on DuckDB, not
    just parse.

    Production failed twice in one day on exactly this class: a build-time
    NameError (f-string constant deleted) and a Binder ambiguity (the gate
    join made bare window partitions ambiguous) — both invisible to
    imports, py_compile, and every fixture-less test because no test ever
    RAN the SQL. This lifts each statement from features.py verbatim,
    executes it against minimal fixtures, and asserts the gating behavior
    survives real compilation: clean rows produce values, stale rows NULL."""
    src = (BACKEND / "features.py").read_text(encoding="utf-8")

    def lift(name):
        j0 = src.index(f"CREATE TABLE {name} AS")
        statement = src[j0:src.index('"""', j0)]
        return statement.replace(
            "{_SP_ERA_5G_SHRINK_IP}", str(features._SP_ERA_5G_SHRINK_IP))

    apps = pd.DataFrame({
        "game_date": pd.to_datetime(["2026-05-01", "2026-06-20",
                                     "2026-05-01", "2026-05-20",
                                     "2026-06-30"]),
        "game_pk": [1, 2, 3, 4, 5],
        "pitcher": [101, 101, 202, 202, 303],
    })
    ilp = pd.DataFrame({
        "batter": [101], "il_start": [pd.Timestamp("2026-05-03")],
        "il_end": [pd.Timestamp("2026-06-10")],
    })
    _mk_ledger(tmp_path, ilp)
    con = duckdb.connect(database=":memory:")
    with patch_ledger_dir(tmp_path):
        con.register("pgs_reg", apps)
        con.execute("CREATE TABLE pitcher_game_stats AS "
                    "SELECT CAST(game_date AS DATE) AS game_date, game_pk, "
                    "pitcher FROM pgs_reg")
        assert features._register_sp_staleness_gate(con) is True
        # pitcher_season_features inputs (minimal shifted aggregates)
        con.execute("""
            CREATE TABLE pitcher_season_rolling AS
            SELECT game_date, game_pk, pitcher,
                   5.0 AS _s_runs_s, 40.0 AS _s_ks_s, 40.0 AS _s_ip_s
            FROM pitcher_game_stats
        """)
        con.execute("""
            CREATE TABLE pitcher_5g_rolling AS
            SELECT game_date, game_pk, pitcher,
                   5.0 AS _roll5_runs, 40.0 AS _roll5_ks, 40.0 AS _roll5_ip,
                   CASE WHEN pitcher = 202 THEN 10.0 ELSE 0.0 END AS _older_runs,
                   CASE WHEN pitcher = 202 THEN 20.0 ELSE 0.0 END AS _older_ip
            FROM pitcher_game_stats
        """)
        con.execute("""
            CREATE TABLE pitcher_era_league AS
            SELECT game_date,
                   CASE WHEN pitcher = 303 THEN NULL ELSE 4.0 END AS league_era_prior
            FROM pitcher_game_stats
        """)
        # pitcher_features inputs (minimal trailing-window aggregates)
        con.execute("""
            CREATE TABLE pitcher_rolling AS
            SELECT game_date, game_pk, pitcher,
                   5.0 AS _roll_bbs, 20.0 AS _roll_hits, 2.0 AS _roll_hrs,
                   1.0 AS _roll_hbps, 40.0 AS _roll_ks, 40.0 AS _roll_ip,
                   0.310 AS _roll_xwoba
            FROM pitcher_game_stats
        """)
        # pitcher_stuff inputs (minimal shifted per-start aggregates)
        con.execute("""
            CREATE TABLE pitcher_stuff_raw AS
            SELECT game_date, game_pk, pitcher,
                   EXTRACT(YEAR FROM game_date) AS season,
                   93.0 AS _s_fb_velo, 0.55 AS _s_fb_pct, 90 AS _s_n,
                   20.0 AS _s_whiffs, 0.300 AS _s_xl, 0.290 AS _s_xr,
                   93.0 AS _s_fb_velo_s, 0.55 AS _s_fb_pct_s, 90 AS _s_n_s,
                   20.0 AS _s_whiffs_s
            FROM pitcher_game_stats
        """)
        con.execute(lift("pitcher_season_features"))
        con.execute(lift("pitcher_features"))
        con.execute(lift("pitcher_stuff"))

    # game_pk 2 = 101's stale return start; game_pk 4 = 202's 19d-rest
    # control start (long gap, NO stint -> must stay ungated). The remaining
    # clean rows pin league-prior and raw-rate fallback behavior.
    seas = con.execute("""
        SELECT game_pk, pitcher, sp_k9, sp_era_5g FROM pitcher_season_features
        WHERE game_pk IN (1, 2, 4, 5)
    """).fetchall()
    feats = con.execute("""
        SELECT pitcher, sp_whip_30g FROM pitcher_features
        WHERE game_pk IN (2, 4)
    """).fetchall()
    stuff = con.execute("""
        SELECT pitcher, sp_fbvelo_3g, sp_whiff_3g FROM pitcher_stuff
        WHERE game_pk IN (2, 4)
    """).fetchall()
    seas = {game_pk: (pitcher, k9, e5)
            for game_pk, pitcher, k9, e5 in seas}
    feats = {p: w for p, w in feats}
    stuff = {p: (v, wf) for p, v, wf in stuff}
    # 101 clean row: older history absent, so blend the recent 1.125 runs/9
    # with the league prior 4.0 using 30 pseudo innings.
    assert seas[1] == (101, 9.0, pytest.approx(33.0 / 14.0))
    # 101 stale return: every SP metric stays NULL despite available priors.
    assert seas[2] == (101, None, None)
    assert feats[101] is None
    assert stuff[101] == (None, None)
    # 202: recent 1.125 is blended toward older personal rate 4.5.
    assert seas[4] == (202, 9.0, pytest.approx(18.0 / 7.0))
    assert feats[202] is not None
    assert stuff[202][0] == 93.0
    # 303 has neither older personal exposure nor a prior league rate, so
    # the raw recent estimate is preserved.
    assert seas[5] == (303, 9.0, pytest.approx(1.125))
