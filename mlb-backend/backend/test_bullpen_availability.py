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
    assert len(training.MONEYLINE_FEATURE_COLS) == 100


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
