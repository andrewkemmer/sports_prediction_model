"""Pre-game lineup identification (2026-10-24, Sep-16 NYY).

ONE source, ONE rule, both paths. Training pools and the slate six deltas
must resolve tonight's nine from the same pre-game capture
(lineups.parquet) under a provenance-only trust predicate:

    complete_* AND (pit_valid IS NOT FALSE OR source is
    'backfill'/'actuals'/NULL)

Deliberately NOT gated on agreement with who actually started: a late
scratch makes the official posted nine differ from the starters, and the
model at bet time only ever sees the posted nine. Posted-vs-actual
agreement (validated_*) is an AUDIT signal, never a membership gate.

Pinned here:
  * the pandas twin (_tier1_eligible_lineups) matches the SQL tier-1 rule on
    every provenance combination, raw and normalized — and validated_* is
    IGNORED by both (FALSE still binds);
  * the SQL itself (via _build_lineup_effective) binds exactly the twin's
    sides — DuckDB conformance, not just a Python mirror of itself;
  * the state gate: ONLY a Preview feed may claim pre-game provenance —
    Live never qualifies (not pre-game point-in-time, even with a pre-pitch
    timestamp), Final least of all (constructor + normalize + register
    paths agree). Voided rows still bind tier 1 as the announced-nine
    corpus; they just never carry PIT privilege;
  * the Sep-16 shape (announced nine with 2 false inclusions / 2 false
    exclusions) still BINDS tier 1 as the official posted nine, with the
    mismatch recorded as audit;
  * the slate reads the pre-game capture (load_pregame_orders) — a
    serve-time live feed read is post-pitch leakage and must never return.
    NULLs ship only for genuinely unknowable nines (unposted, or a capture
    that cannot prove pre-pitch).
"""
from __future__ import annotations

import inspect
import itertools
import random
import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import backfill_lineups as bl  # noqa: E402
import features  # noqa: E402
from features import _tier1_eligible_lineups  # noqa: E402

DAY = "2026-09-16"

# (complete, pit_valid, source) — every provenance combination that decides
# tier 1. validated_* is randomized alongside but must NEVER decide.
COMBOS = list(itertools.product(
    [True, False, None],
    [True, False, None],
    ["pregame", "backfill", "actuals", None],
))

_TRUSTED_BASE = {"complete": True, "pit": True, "source": "pregame"}


def _sql_rule(complete, pit, source) -> bool:
    """The order_sides tier-1 predicate, in Python (test oracle)."""
    return (bool(complete) and ((pit is not False)
                                or source == "backfill"
                                or source == "actuals"
                                or source is None))


def _side_frame(cycling: str):
    """162-row frame: `cycling` side runs the provenance matrix.

    The other side is pinned trusted. validated_* cycles True/False/None
    independently (seeded) — expectations must not depend on it.
    """
    rng = random.Random(20261024)
    recs = []
    for i, combo in enumerate(COMBOS):
        c, p, s = combo
        home = dict(_TRUSTED_BASE)
        away = dict(_TRUSTED_BASE)
        (home if cycling == "home" else away).update(
            {"complete": c, "pit": p, "source": s})
        # The cache row carries ONE pit/source pair; the SQL gates each side
        # by the row pair — pin the row pair to the cycling side's pair.
        row_pit, row_src = p, s
        vh = rng.choice([True, False, None])
        va = rng.choice([True, False, None])
        recs.append({
            "game_pk": (1000 if cycling == "home" else 2000) + i,
            "game_date": pd.Timestamp(DAY),
            "home_team": f"H{i}", "away_team": f"A{i}",
            "home_order": (list(range(10 * i + 1, 10 * i + 10))
                           if home["complete"] else None),
            "away_order": (list(range(10000 + 10 * i + 1, 10000 + 10 * i + 10))
                           if away["complete"] else None),
            "complete_home": home["complete"],
            "complete_away": away["complete"],
            "validated_home": vh, "validated_away": va,
            "pit_valid": row_pit, "source": row_src,
            "home_starter_hand": "R", "away_starter_hand": "L",
            "exp_home": _sql_rule(home["complete"], row_pit, row_src),
            "exp_away": _sql_rule(away["complete"], row_pit, row_src),
        })
    df = pd.DataFrame(recs)
    # pandas>=3 infers a `str` dtype for the source column and stores the
    # NULL combos as NaN — but the SQL reads them as NULL and the oracle
    # treats None as NULL. Normalize every provenance cell to None-or-value
    # so all three layers (frame, twin, SQL) see the same NULLs.
    for col in ("complete_home", "complete_away", "validated_home",
                "validated_away", "pit_valid", "source"):
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    return (df.drop(columns=["exp_home", "exp_away"]),
            df["exp_home"], df["exp_away"])


def test_tier1_twin_matches_sql_rule_on_every_provenance_combo():
    """Raw + normalized frames: the pandas twin binds exactly the SQL sides,
    and validated_* never decides (FALSE binds — the late-scratch case)."""
    for cycling in ("home", "away"):
        df, exp_h, exp_a = _side_frame(cycling)
        for shape, frame in (("raw", df), ("normalized", bl._normalize(df))):
            trusted = _tier1_eligible_lineups(frame)
            got_h = trusted["home_order"].notna()
            got_a = trusted["away_order"].notna()
            if shape == "normalized":
                # _normalize folds NULLs: source->backfill, complete/pit
                # None->False, validated None->NA (ignored either way).
                exp_h = frame.apply(
                    lambda r: _sql_rule(r["complete_home"], r["pit_valid"],
                                        r["source"]), axis=1)
                exp_a = frame.apply(
                    lambda r: _sql_rule(r["complete_away"], r["pit_valid"],
                                        r["source"]), axis=1)
            bad_h = frame.loc[got_h != exp_h]
            bad_a = frame.loc[got_a != exp_a]
            assert bad_h.empty and bad_a.empty, \
                (f"{shape}/{cycling}: twin disagrees with SQL on "
                 f"{len(bad_h)} home + {len(bad_a)} away sides")
            # validated=FALSE sides still bind when provenance holds
            # (complete taken from the row — only provenance + validated
            # interact here)
            prov_ok = frame.apply(
                lambda r: _sql_rule(r["complete_home"], r["pit_valid"],
                                     r["source"]),
                axis=1)
            falsely_quarantined = frame.loc[
                (frame["validated_home"] == False)  # noqa: E712
                & prov_ok & ~got_h]
            assert falsely_quarantined.empty, \
                "validated=FALSE must never quarantine a provenanced nine"
            for idx, r in frame.iterrows():
                if exp_h.iloc[idx]:
                    assert list(trusted.loc[idx, "home_order"]) == \
                        list(r["home_order"])
                else:
                    assert trusted.loc[idx, "home_order"] is None
                if exp_a.iloc[idx]:
                    assert list(trusted.loc[idx, "away_order"]) == \
                        list(r["away_order"])
                else:
                    assert trusted.loc[idx, "away_order"] is None


def _conformance_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    # pitches must EXIST (slots LEFT JOINs it) but stays empty: every test
    # game is slate-like (no played frame), so teams resolve from the
    # lineups codes — exactly the tonight-slate shape.
    con.execute("CREATE TABLE pitches AS SELECT "
                "CAST(NULL AS BIGINT) AS game_pk, "
                "CAST(NULL AS VARCHAR) AS home_team, "
                "CAST(NULL AS VARCHAR) AS away_team WHERE false")
    con.execute("CREATE TABLE starters AS SELECT "
                "CAST(NULL AS BIGINT) AS game_pk, "
                "CAST(NULL AS VARCHAR) AS home_team, "
                "CAST(NULL AS VARCHAR) AS away_team, "
                "CAST(NULL AS VARCHAR) AS home_starter_hand, "
                "CAST(NULL AS VARCHAR) AS away_starter_hand WHERE false")
    return con


def test_sql_tier1_binds_exactly_the_trusted_sides():
    """DuckDB conformance: order_sides binds a side IFF provenance holds.

    Same date for every row, so no 10-day projection history exists and
    tier 2 is provably empty — every bound batter is tier 1 by construction.
    """
    df_h, _, _ = _side_frame("home")
    df_a, _, _ = _side_frame("away")
    df = pd.concat([df_h, df_a], ignore_index=True)
    con = _conformance_con()
    try:
        con.execute("CREATE TABLE lineups_raw AS SELECT * FROM df")
        # universe: one row per team-game (batter arbitrary, never pooled here)
        uni = []
        for r in df.itertuples():
            uni.append((DAY, r.game_pk, r.home_team, r.home_order[0]
                        if r.home_order else -1))
            uni.append((DAY, r.game_pk, r.away_team, r.away_order[0]
                        if r.away_order else -1))
        con.execute(
            "CREATE TABLE batter_ratings AS SELECT * FROM "
            "(VALUES " + ", ".join(
                f"(DATE '{d}', {pk}, '{t}', {b}, 0.300, 50.0)"
                for d, pk, t, b in uni) + ") AS t(game_date, game_pk, "
            "batting_team, batter, shrunk_re24, _pa30)")
        features._build_lineup_effective(con, batters_ok=False)
        by_team: dict = {}
        for pk, team, batter, tier in con.execute(
                "SELECT game_pk, team, batter, tier FROM lineup_effective"
        ).fetchall():
            by_team.setdefault((pk, team), []).append((batter, tier))
        assert not con.execute(
            "SELECT count(*) FROM lineup_effective WHERE tier = 2"
        ).fetchone()[0], "same-date frame must have no tier-2 rows"
        n_checked = 0
        for idx, r in df.reset_index(drop=True).iterrows():
            home_exp = _sql_rule(r["complete_home"], r["pit_valid"],
                                 r["source"])
            away_exp = _sql_rule(r["complete_away"], r["pit_valid"],
                                 r["source"])
            for team, order, exp in ((r["home_team"], r["home_order"], home_exp),
                                     (r["away_team"], r["away_order"], away_exp)):
                members = {b for b, t in by_team.get((r["game_pk"], team), [])
                           if t == 1}
                if exp:
                    assert members == set(order), \
                        f"pk={r['game_pk']} {team}: tier 1 must be the nine"
                else:
                    assert not (members & set(order or [])), \
                        f"pk={r['game_pk']} {team}: unprovenanced side must " \
                        f"not bind tier 1"
                n_checked += 1
        assert n_checked == 2 * len(df)
    finally:
        con.close()


def test_sep16_shape_binds_the_official_posted_nine():
    """The incident, re-decided (2026-10-24): announced nine with 2 in / 2
    out vs the first trip STILL BINDS tier 1 — it is the official posted
    nine, scratches included. The mismatch is audit output, not a gate."""
    announced = [501, 502, 503, 504, 505, 506, 507, 508, 509]
    actual = [601, 602, 503, 504, 505, 506, 507, 508, 509]
    trips = {(4242, "home"): actual,
             (4242, "away"): [701, 702, 703, 704, 705, 706, 707, 708, 709]}
    vh, va = bl.validate_orders(announced, trips[(4242, "away")], trips, 4242)
    assert (vh, va) == (False, True)  # the audit verdict is recorded...
    row = {"game_pk": 4242, "game_date": pd.Timestamp(DAY),
           "home_team": "NYY", "away_team": "BOS",
           "home_order": announced, "away_order": trips[(4242, "away")],
           "complete_home": True, "complete_away": True,
           "validated_home": vh, "validated_away": va,  # ...but binds anyway
           "pit_valid": False, "source": "backfill"}
    trusted = _tier1_eligible_lineups(pd.DataFrame([row]))
    assert list(trusted.loc[0, "home_order"]) == announced
    assert list(trusted.loc[0, "away_order"]) == trips[(4242, "away")]
    con = _conformance_con()
    try:
        con.execute("CREATE TABLE lineups_raw AS SELECT * FROM "
                    "(SELECT * FROM (VALUES (4242, DATE '2026-09-16', 'NYY', "
                    "'BOS', [501,502,503,504,505,506,507,508,509]::BIGINT[], "
                    "[701,702,703,704,705,706,707,708,709]::BIGINT[], "
                    "true, true, false, 'backfill', false, true, "
                    "'R', 'L')) AS t("
                    "game_pk, game_date, home_team, away_team, home_order, "
                    "away_order, complete_home, complete_away, pit_valid, "
                    "source, validated_home, validated_away, "
                    "home_starter_hand, away_starter_hand))")
        con.execute("CREATE TABLE batter_ratings AS SELECT * FROM (VALUES "
                    "(DATE '2026-09-16', 4242, 'NYY', 501, 0.300, 50.0), "
                    "(DATE '2026-09-16', 4242, 'BOS', 701, 0.300, 50.0)) "
                    "AS t(game_date, game_pk, batting_team, batter, "
                    "shrunk_re24, _pa30)")
        features._build_lineup_effective(con, batters_ok=False)
        nyy = {b for (b,) in con.execute(
            "SELECT batter FROM lineup_effective "
            "WHERE game_pk = 4242 AND team = 'NYY'").fetchall()}
        assert nyy == set(announced), \
            "the official posted nine binds tier 1 despite the mismatch"
    finally:
        con.close()


def test_register_lineups_voids_contradictory_pregame_claims(tmp_path,
                                                             monkeypatch):
    """_register_lineups applies the provenance void on the SQL path, so the
    pools and the pandas twin can never disagree on tier 1."""
    import duckdb
    feed = {"gameData": {"status": {"abstractGameState": "Preview"}},
            "liveData": {"boxscore": {"teams": {
                "home": {"battingOrder": [501, 502, 503, 504, 505, 506,
                                             507, 508, 509]},
                "away": {"battingOrder": [701, 702, 703, 704, 705, 706,
                                             707, 708, 709]}}}}}
    row = bl.order_row(4242, DAY, "NYY", "BOS", feed, state="Preview",
                       captured_utc="2026-09-16T18:00:00+00:00",
                       first_pitch_utc="2026-09-16T23:05:00+00:00",
                       source="pregame")
    assert row["source"] == "pregame" and row["pit_valid"] is True
    # a Live read is NEVER pre-game point-in-time — even with a pre-pitch
    # feed-attested timestamp, it voids on every path (constructor here,
    # register below). It still binds tier 1 as the announced-nine corpus;
    # it just never carries PIT privilege.
    live_feed = {"gameData": {"status": {"abstractGameState": "Live"}},
                 "metaData": {"timeStamp": "20260916_155450"},
                 "liveData": {"boxscore": {"teams": {
                     "home": {"battingOrder": [501, 502, 503, 504, 505,
                                                  506, 507, 508, 509]},
                     "away": {"battingOrder": [701, 702, 703, 704, 705,
                                                  706, 707, 708, 709]}}}}}
    live_row = bl.order_row(
        4243, DAY, "NYY", "BOS", live_feed,
        captured_utc="2026-09-16T15:54:50+00:00",
        first_pitch_utc="2026-09-16T17:42:43+00:00", source="pregame",
        feed_timecode="20260916_155450")
    assert live_row["source"] == "backfill" \
        and live_row["pit_valid"] is False
    df = pd.DataFrame([row, live_row])
    df.loc[0, "state"] = "Final"  # hand-inject the contradiction
    df.to_parquet(tmp_path / "lineups.parquet", index=False)
    monkeypatch.setattr(features, "_lineup_base_dir", lambda: tmp_path)
    con = duckdb.connect(database=":memory:")
    try:
        assert features._register_lineups(con) is True
        got = {r[0]: (r[1], r[2]) for r in con.execute(
            "SELECT game_pk, source, pit_valid FROM lineups_raw").fetchall()}
        assert got[4242] == ("backfill", False)  # Final voids...
        assert got[4243] == ("backfill", False)  # ...and Live voids too
        # and the twin agrees on the same loaded frame: voided rows bind
        # as the grandfathered announced-nine corpus
        frame = con.execute("SELECT * FROM lineups_raw").df()
        trusted = _tier1_eligible_lineups(frame)
        assert (trusted["home_order"].notna()).all()
        assert (trusted["away_order"].notna()).all()
    finally:
        con.close()


# ── slate reads the pre-game capture, never a live feed ──────────────────────

class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _schedule_payload():
    def game(pk, home, away, start):
        return {"gamePk": pk, "gameDate": start,
                "teams": {"home": {"team": {"abbreviation": home}},
                          "away": {"team": {"abbreviation": away}}}}
    return {"dates": [{"games": [
        game(9001, "BOS", "NYY", "2026-09-16T19:05:00-04:00"),
        game(9002, "BOS", "NYY", "2026-09-16T19:10:00-04:00"),
        game(9003, "BOS", "NYY", "2026-09-16T19:15:00-04:00"),
    ]}]}


def _slate_frame():
    return pd.DataFrame([
        {"game_id": "g1", "home_team": "BOS", "away_team": "NYY",
         "game_date": pd.Timestamp(DAY),
         "start_time_utc": pd.Timestamp("2026-09-16 19:05:00")},
        {"game_id": "g2", "home_team": "BOS", "away_team": "NYY",
         "game_date": pd.Timestamp(DAY),
         "start_time_utc": pd.Timestamp("2026-09-16 19:10:00")},
        {"game_id": "g3", "home_team": "BOS", "away_team": "NYY",
         "game_date": pd.Timestamp(DAY),
         "start_time_utc": pd.Timestamp("2026-09-16 19:15:00")},
        # g4 is a different matchup entirely (identity never resolves) —
        # nothing may leak into it from the other games' posted nines.
        {"game_id": "g4", "home_team": "LAD", "away_team": "SF",
         "game_date": pd.Timestamp(DAY),
         "start_time_utc": pd.Timestamp("2026-09-16 20:05:00")},
    ])


def test_slate_reads_pregame_capture_never_a_live_feed(tmp_path, monkeypatch):
    """results._fetch_slate_lineups: a validated-FALSE side still enriches
    (audit, not a gate); unposted and unprovable captures ship NULL; and no
    feed/live URL is ever fetched (post-pitch leakage vector)."""
    import results

    monkeypatch.setattr(bl, "DATA_DELIVERY_DIR", tmp_path)
    g1_home = list(range(101, 110))
    g1_away = list(range(201, 210))
    g2_home = list(range(111, 120))  # validated FALSE — still the posted nine
    g2_away = list(range(211, 220))
    g3_home = list(range(121, 130))  # post-scheduled Preview capture
    g3_away = list(range(221, 230))  # (delayed game) — unprovable, NULLs
    rows = []
    for pk, ho, ao, vh, cap in (
            (9001, g1_home, g1_away, None, "2026-09-16T14:00:00+00:00"),
            (9002, g2_home, g2_away, False, "2026-09-16T14:00:00+00:00"),
            (9003, g3_home, g3_away, None, "2026-09-17T01:00:00+00:00")):
        rows.append(bl.order_row(
            pk, DAY, "BOS", "NYY",
            {"gameData": {"status": {"abstractGameState": "Preview"}},
             "liveData": {"boxscore": {"teams": {
                 "home": {"battingOrder": ho},
                 "away": {"battingOrder": ao}}}}},
            captured_utc=cap,
            first_pitch_utc="2026-09-16T23:05:00+00:00",
            source="pregame",
            validated_home=vh, validated_away=None))
    bl.upsert_lineups(pd.DataFrame(rows),
                      out_path=tmp_path / "lineups.parquet")

    calls: list = []

    def fake_get(url, **kw):
        calls.append(url)
        assert "feed/live" not in url, \
            "slate must never read a serve-time live feed"
        return _FakeResp(_schedule_payload())

    monkeypatch.setattr("requests.get", fake_get)
    day = pd.Timestamp(DAY)
    bat_ids = (g1_home + g1_away + g2_home + g2_away + g3_home + g3_away)
    features._lineup_cache.clear()
    features._lineup_cache.update({
        "lineups": pd.DataFrame(),
        "batter": pd.DataFrame({
            "season": [2026] * len(bat_ids),
            "game_date": [day] * len(bat_ids),
            "batter": bat_ids,
            "sd_woba": [0.30 + 0.001 * (b % 50) for b in bat_ids],
            "prior_pa": [100] * len(bat_ids),
        }),
        "team": pd.DataFrame({
            "season": [2026, 2026],
            "game_date": [day, day],
            "team": ["BOS", "NYY"],
            "sd_woba": [0.32, 0.33],
            "top3_woba": [0.35, 0.36],
            "top5_ids": [f"{g1_home[:5]}", f"{g1_away[:5]}"],
        }),
    })
    try:
        out = results._fetch_slate_lineups(_slate_frame(), date(2026, 9, 16))
    finally:
        features._lineup_cache.clear()
    assert calls and all("schedule" in u for u in calls)
    assert list(out["game_pk"].astype("Int64")[:3]) == [9001, 9002, 9003]
    assert pd.isna(out.loc[3, "game_pk"])
    six = ["lineup_actual_woba_delta_home", "lineup_actual_woba_delta_away",
           "lineup_actual_top3_delta_home", "lineup_actual_top3_delta_away",
           "lineup_rest_count_home", "lineup_rest_count_away"]
    assert out.loc[0, six].notna().all(), \
        "trusted pre-game nine must enrich all six deltas"
    assert out.loc[1, six].notna().all(), \
        "validated=FALSE is audit-only — the official posted nine enriches"
    assert out.loc[2, six].isna().all(), \
        "a capture that cannot prove pre-pitch ships NULL, never fabricated"
    assert out.loc[3, six].isna().all(), \
        "an unposted game ships NULL — never a neighboring posted nine"


def test_snapshot_repair_and_slate_capture_are_the_same_artifact(tmp_path,
                                                                  monkeypatch):
    """Interchange proof: a snapshot-restored training row and a slate
    capture of the same game are indistinguishable to the trust rule —
    same orders, same provenance shape, same tier-1 binding."""
    import results

    monkeypatch.setattr(bl, "DATA_DELIVERY_DIR", tmp_path)
    p = tmp_path / "lineups.parquet"
    # legacy grandfathered row (the corruptible read)
    bl.upsert_lineups(pd.DataFrame([bl.order_row(
        9001, DAY, "BOS", "NYY",
        {"gameData": {"status": {"abstractGameState": "Final"}},
         "liveData": {"boxscore": {"teams": {
             "home": {"battingOrder": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
             "away": {"battingOrder": [11, 12, 13, 14, 15, 16, 17,
                                          18, 19]}}}}})]), out_path=p)
    # training repair from the archived snapshot (mocked transport)
    snap = {"gameData": {"status": {"abstractGameState": "Preview"},
                            "teams": {"home": {"abbreviation": "BOS"},
                                      "away": {"abbreviation": "NYY"}}},
              "metaData": {"timeStamp": "20260916_155450"},
              "liveData": {"boxscore": {"teams": {
                  "home": {"battingOrder": [21, 22, 23, 24, 25, 26, 27,
                                              28, 29]},
                  "away": {"battingOrder": [31, 32, 33, 34, 35, 36, 37,
                                              38, 39]}}}}}
    monkeypatch.setattr(bl, "fetch_timecodes",
                          lambda pk: ["20260916_155450"])
    monkeypatch.setattr(bl, "fetch_snapshot",
                          lambda pk, tc, pause_sec=0.15: (snap, ""))
    monkeypatch.setattr(results, "fetch_game_start_times",
                          lambda a, b: {9001: "2026-09-16T17:40:00Z"})
    pd.DataFrame([{"game_pk": 9001, "game_date": DAY,
                    "home_team": "BOS", "away_team": "NYY",
                    "home_win": 1}]).to_csv(tmp_path / "game_level_features.csv",
                                             index=False)
    assert bl.backfill_pregame_snapshots(
        limit=5, csv_path=tmp_path / "game_level_features.csv",
        out_path=p) == 1
    # slate capture of the same game (mocked live edge, same orders)
    slate_row = bl.order_row(
        9001, DAY, "BOS", "NYY", snap,
        captured_utc=bl.parse_feed_timecode("20260916_155450"),
        first_pitch_utc="2026-09-16T17:40:00Z", source="pregame",
        feed_timecode="20260916_155450")
    repaired = pd.read_parquet(p).loc[lambda d: d.game_pk == 9001]
    for col in ("home_order", "away_order", "source", "pit_valid",
                "complete_home", "complete_away"):
        r, s = repaired.iloc[0][col], slate_row[col]
        assert list(r) == list(s) if isinstance(s, list) else r == s, col
    # and both bind tier 1 identically
    twin = _tier1_eligible_lineups
    assert twin(repaired)[["home_order", "away_order"]].notna().all().all()
    assert twin(pd.DataFrame([slate_row]))[["home_order",
                                            "away_order"]].notna().all().all()


def test_both_slate_paths_read_the_pregame_cache():
    """Static parity: neither _fetch_slate_lineups copy touches a live feed;
    both delegate orders to load_pregame_orders (one source, one rule)."""
    import results

    bodies = {}
    bodies["results"] = inspect.getsource(results._fetch_slate_lineups)
    text = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    start = text.index("def _fetch_slate_lineups")
    end = text.index("\n\n\ndef ", start)
    bodies["master_pipeline"] = text[start:end]
    for name, body in bodies.items():
        assert "load_pregame_orders" in body, name
        assert "feed/live" not in body, \
            f"{name}._fetch_slate_lineups must not read a live feed"
