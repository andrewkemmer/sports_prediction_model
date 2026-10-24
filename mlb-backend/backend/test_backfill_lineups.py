"""Lineup cache maintenance (2026-10-03 membership audit).

lineups.parquet is membership TIER 1 (tonight's nine): features.py resolves
lineup_effective from it, so the cache's write rules decide what the pools
can price. Pinned here:

  * order_row — pure feed -> row shape; complete_* is TRUE only on an
    exact 9-man side, and a missing feed is an honest incomplete row;
  * upsert_lineups — latest-write merge by game_pk with the NO-DOWNGRADE
    rule: an incomplete pre-game post can never erase a known nine;
  * complete_game_pks — the COMPLETE-only resume set, so fetch_error /
    partial rows retry every run instead of freezing a game at
    projected-only membership forever.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import backfill_lineups as bl  # noqa: E402


def _feed(home_ids, away_ids, state="Final"):
    return {
        "gameData": {"status": {"abstractGameState": state}},
        "liveData": {"boxscore": {"teams": {
            "home": {"battingOrder": list(home_ids)},
            "away": {"battingOrder": list(away_ids)},
        }}},
    }


_NINE_H = [301, 302, 303, 304, 305, 306, 307, 308, 309]
_NINE_A = [101, 102, 103, 104, 105, 106, 107, 108, 109]


def test_order_row_requires_an_exact_nine_per_side():
    row = bl.order_row(42, "2025-06-05", "NYY", "BOS",
                       _feed(_NINE_H, _NINE_A))
    assert row["complete_home"] and row["complete_away"]
    assert row["home_order"] == _NINE_H and row["away_order"] == _NINE_A
    assert row["state"] == "Final"
    # short side -> honest INCOMPLETE row (home_order withheld, not padded)
    row = bl.order_row(43, "2025-06-05", "NYY", "BOS",
                       _feed(_NINE_H[:-1], _NINE_A))
    assert row["complete_home"] is False and row["complete_away"] is True
    assert row["home_order"] is None
    # no feed (fetch error) -> nothing fabricated, caller's state kept
    row = bl.order_row(44, "2025-06-05", "NYY", "BOS", None,
                       state="fetch_error: timeout")
    assert row["complete_home"] is False and row["complete_away"] is False
    assert row["home_order"] is None and row["away_order"] is None
    assert row["state"] == "fetch_error: timeout"


def test_order_row_never_raises_on_malformed_feed():
    row = bl.order_row(45, "2025-06-05", "NYY", "BOS",
                       {"liveData": {"boxscore": {"teams": {
                           "home": {"battingOrder": "not-a-list"}}}}})
    assert row["complete_home"] is False
    assert row["state"] == "unknown"


def _path(tmp_path) -> Path:
    return tmp_path / "lineups.parquet"


def test_upsert_appends_new_games(tmp_path):
    p = _path(tmp_path)
    n = bl.upsert_lineups(pd.DataFrame([
        bl.order_row(1, "2025-06-01", "NYY", "BOS", _feed(_NINE_H, _NINE_A)),
        bl.order_row(2, "2025-06-02", "NYY", "BOS", _feed(_NINE_H, _NINE_A)),
    ]), out_path=p)
    assert n == 2
    df = pd.read_parquet(p)
    assert set(df.game_pk) == {1, 2}
    assert df.complete_home.all() and df.complete_away.all()


def test_upsert_never_downgrades_a_known_nine(tmp_path):
    p = _path(tmp_path)
    bl.upsert_lineups(pd.DataFrame([
        bl.order_row(1, "2025-06-01", "NYY", "BOS", _feed(_NINE_H, _NINE_A)),
    ]), out_path=p)
    # a pre-game re-fetch BEFORE the sides post must not erase the nine
    n = bl.upsert_lineups(pd.DataFrame([
        bl.order_row(1, "2025-06-01", "NYY", "BOS", None,
                     state="Preview"),
    ]), out_path=p)
    assert n == 0
    df = pd.read_parquet(p)
    row = df.loc[df.game_pk == 1].iloc[0]
    assert row.complete_home and row.complete_away
    assert list(row.home_order) == _NINE_H
    assert row.state == "Final"


def test_upsert_upgrades_an_incomplete_row(tmp_path):
    p = _path(tmp_path)
    bl.upsert_lineups(pd.DataFrame([
        bl.order_row(1, "2025-06-01", "NYY", "BOS", None, state="Preview"),
    ]), out_path=p)
    assert bl.complete_game_pks(p) == set()
    n = bl.upsert_lineups(pd.DataFrame([
        bl.order_row(1, "2025-06-01", "NYY", "BOS", _feed(_NINE_H, _NINE_A)),
    ]), out_path=p)
    assert n == 1
    assert bl.complete_game_pks(p) == {1}


def test_complete_resume_set_excludes_partial_and_error_rows(tmp_path):
    p = _path(tmp_path)
    bl.upsert_lineups(pd.DataFrame([
        bl.order_row(1, "2025-06-01", "NYY", "BOS", _feed(_NINE_H, _NINE_A)),
        bl.order_row(2, "2025-06-02", "NYY", "BOS", _feed(_NINE_H, [])),
        bl.order_row(3, "2025-06-03", "NYY", "BOS", None,
                     state="fetch_error: timeout"),
    ]), out_path=p)
    assert bl.complete_game_pks(p) == {1}, \
        "only a 9+9 row may leave the resume set; the rest retry next run"


def test_missing_cache_resume_set_is_empty(tmp_path):
    assert bl.complete_game_pks(_path(tmp_path)) == set()


# ── trust + validation (2026-10-23, Sep-16 NYY case) ─────────────────────────

def _row(pk, date="2026-09-16", home_ids=_NINE_H, away_ids=_NINE_A, **kw):
    return bl.order_row(pk, date, "NYY", "BOS", _feed(home_ids, away_ids),
                        **kw)


# September 2026 Yankees evidence pin (2026-10-26).
#
# These fixtures replay the batting-order IDs and resolved player names
# observed in the archived pre-pitch snapshots. They run offline: the
# constants carry the evidence, while the exercised selectors/gates verify
# the snapshot is strictly pre-pitch and the orders survive the current
# Live-feed provenance gate unchanged (as backfill-grade reference data).
_PINNED_SEP_2026_NYY = (
    {
        "game_pk": 823654,
        "game_date": "2026-09-15",
        "home": "MIN",
        "away": "NYY",
        "snapshot": "20260915_231837",
        "first_pitch": "2026-09-15T23:40:00+00:00",
        "home_order": [807712, 701785, 605137, 665019, 668904,
                         686797, 605170, 691181, 668885],
        "home_names": ["Luke Keaschall", "Kaelen Culpepper",
                         "Josh Bell", "Kody Clemens", "Royce Lewis",
                         "Brooks Lee", "Victor Caratini",
                         "Emmanuel Rodriguez", "Austin Martin"],
        "away_order": [700250, 641355, 671218, 671277, 682987,
                         806146, 683011, 645305, 676609],
        "away_names": ["Ben Rice", "Cody Bellinger", "Heliot Ramos",
                         "Luis Garc\u00eda Jr.", "Spencer Jones",
                         "George Lombard Jr.", "Anthony Volpe",
                         "Ali S\u00e1nchez", "Jos\u00e9 Caballero"],
    },
    {
        "game_pk": 823655,
        "game_date": "2026-09-16",
        "home": "MIN",
        "away": "NYY",
        "snapshot": "20260916_171821",
        "first_pitch": "2026-09-16T17:40:00+00:00",
        "home_order": [807712, 668885, 680777, 605137, 665019,
                         668904, 691181, 668952, 805346],
        "home_names": ["Luke Keaschall", "Austin Martin",
                         "Ryan Jeffers", "Josh Bell", "Kody Clemens",
                         "Royce Lewis", "Emmanuel Rodriguez",
                         "Ryan Kreidler", "Ben Ross"],
        "away_order": [700250, 641355, 592450, 671277, 682987,
                         806146, 641857, 669224, 683011],
        "away_names": ["Ben Rice", "Cody Bellinger", "Aaron Judge",
                         "Luis Garc\u00eda Jr.", "Spencer Jones",
                         "George Lombard Jr.", "Ryan McMahon",
                         "Austin Wells", "Anthony Volpe"],
    },
    {
        "game_pk": 825032,
        "game_date": "2026-09-18",
        "home": "AZ",
        "away": "NYY",
        "snapshot": "20260919_013952",
        "first_pitch": "2026-09-19T01:40:00+00:00",
        "home_order": [663457, 606466, 672515, 672695, 656976,
                         571448, 682998, 660821, 694374],
        "home_names": ["Lars Nootbaar", "Ketel Marte",
                         "Gabriel Moreno", "Geraldo Perdomo",
                         "Pavin Smith", "Nolan Arenado",
                         "Corbin Carroll", "Jes\u00fas S\u00e1nchez",
                         "Tim Tawa"],
        "away_order": [502671, 641355, 671218, 700250, 642708,
                         682987, 806146, 669224, 683011],
        "away_names": ["Paul Goldschmidt", "Cody Bellinger",
                         "Heliot Ramos", "Ben Rice", "Amed Rosario",
                         "Spencer Jones", "George Lombard Jr.",
                         "Austin Wells", "Anthony Volpe"],
    },
)


def _pinned_archived_feed(case):
    def _players(order, names):
        return {
            f"ID{pid}": {"person": {"id": pid, "fullName": name}}
            for pid, name in zip(order, names)
        }

    return {
        "gameData": {
            "status": {"abstractGameState": "Live"},
            "teams": {
                "home": {"abbreviation": case["home"]},
                "away": {"abbreviation": case["away"]},
            },
        },
        "metaData": {"timeStamp": case["snapshot"]},
        "liveData": {"boxscore": {"teams": {
            "home": {"battingOrder": list(case["home_order"]),
                       "players": _players(case["home_order"],
                                            case["home_names"])},
            "away": {"battingOrder": list(case["away_order"]),
                       "players": _players(case["away_order"],
                                            case["away_names"])},
        }}},
    }


def test_archived_sep_2026_nyy_lineups_are_pinned():
    for case in _PINNED_SEP_2026_NYY:
        feed = _pinned_archived_feed(case)
        parsed = bl.parse_batting_orders(feed)
        assert parsed["home"] == case["home_order"]
        assert parsed["away"] == case["away_order"]
        assert [parsed["home_players"][pid] for pid in case["home_order"]] == \
            case["home_names"]
        assert [parsed["away_players"][pid] for pid in case["away_order"]] == \
            case["away_names"]
        assert bl.select_pregame_timecode([case["snapshot"]],
                                          case["first_pitch"]) == case["snapshot"]
        row = bl.order_row(case["game_pk"], case["game_date"],
                           case["home"], case["away"], feed,
                           captured_utc=bl.parse_feed_timecode(case["snapshot"]),
                           first_pitch_utc=case["first_pitch"],
                           source="pregame",
                           feed_timecode=case["snapshot"])
        # The archived feeds report Live state, so the present gate keeps
        # these snapshots as backfill-grade reference evidence. The pinned
        # nine must nevertheless survive unchanged.
        assert row["state"] == "Live"
        assert row["source"] == "backfill"
        assert row["pit_valid"] is False
        assert row["complete_home"] and row["complete_away"]
        assert list(row["home_order"]) == case["home_order"]
        assert list(row["away_order"]) == case["away_order"]


def test_validate_orders_tristate():
    trips = {(7, "home"): list(_NINE_H), (7, "away"): list(_NINE_A)}
    # agreement on both sides
    assert bl.validate_orders(_NINE_H, _NINE_A, trips, 7) == (True, True)
    # Sep-16 shape: 2 false inclusions + 2 false exclusions on one side
    bad_home = [901, 902, *_NINE_H[2:]]
    assert bl.validate_orders(bad_home, _NINE_A, trips, 7) == (False, True)
    # no first-trip coverage -> cannot check (NULL, never a sticky FALSE)
    assert bl.validate_orders(_NINE_H, _NINE_A, {}, 7) == (None, None)
    assert bl.validate_orders(_NINE_H, _NINE_A, {(7, "home"): list(_NINE_H)},
                              7) == (True, None)
    # missing stored order -> cannot check that side
    assert bl.validate_orders(None, _NINE_A, trips, 7) == (None, True)


def test_order_row_state_gate_voids_postgame_pregame_claims():
    # only a Preview feed may claim pre-game provenance — a post-game read
    # keeps its orders as backfill-grade, never pre-game provenance.
    pre = _row(1, state="Preview", captured_utc="2026-09-16T18:00:00+00:00",
                first_pitch_utc="2026-09-16T23:05:00+00:00",
                source="pregame")
    assert pre["source"] == "pregame" and pre["pit_valid"] is True
    for state in ("Final", "Live", "unknown"):
        post = _row(1, state=state,
                    captured_utc="2026-09-16T18:00:00+00:00",
                    first_pitch_utc="2026-09-16T23:05:00+00:00",
                    source="pregame")
        assert post["source"] == "backfill" and post["pit_valid"] is False
        assert list(post["home_order"]) == _NINE_H  # orders kept


def test_provenance_void_applies_on_every_normalize_path():
    # the safety net behind the constructor gate: hand-built frames with a
    # self-contradicting claim void to backfill-grade, orders untouched.
    df = pd.DataFrame([{
        "game_pk": 1, "game_date": "2026-09-16", "home_team": "NYY",
        "away_team": "BOS", "home_order": _NINE_H, "away_order": _NINE_A,
        "complete_home": True, "complete_away": False,
        "state": "Final", "source": "pregame", "pit_valid": True,
        "validated_home": None, "validated_away": None,
    }])
    out = bl.apply_provenance_void(df)
    assert out.loc[0, "source"] == "backfill"
    assert bool(out.loc[0, "pit_valid"]) is False
    assert list(out.loc[0, "home_order"]) == _NINE_H
    # a genuine pre-game claim is untouched; without a recorded state there
    # is no contradiction to judge (the pit_valid flag decides, as in SQL)
    df2 = df.copy()
    df2.loc[0, "state"] = "Preview"
    assert bl.apply_provenance_void(df2).loc[0, "source"] == "pregame"
    df3 = df.drop(columns=["state"])
    assert bl.apply_provenance_void(df3).loc[0, "source"] == "pregame"


def test_upsert_never_clobbers_pit_valid_pregame_row(tmp_path):
    p = _path(tmp_path)
    morning = _row(1, state="Preview",
                   captured_utc="2026-09-16T18:00:00+00:00",
                   first_pitch_utc="2026-09-16T23:05:00+00:00",
                   source="pregame")
    assert morning["pit_valid"] is True
    bl.upsert_lineups(pd.DataFrame([morning]), out_path=p)
    # an evening re-run re-reads the post-game field with contradicting
    # orders — a Final-feed read can only land as backfill-grade, and the
    # morning nine must survive it.
    late = _row(1, state="Final",
                home_ids=list(reversed(_NINE_H)),
                away_ids=list(reversed(_NINE_A)),
                captured_utc="2026-09-17T02:00:00+00:00",
                first_pitch_utc="2026-09-16T23:05:00+00:00",
                source="pregame")
    assert late["source"] == "backfill" and late["pit_valid"] is False
    assert bl.upsert_lineups(pd.DataFrame([late]), out_path=p) == 0
    row = pd.read_parquet(p).loc[lambda d: d.game_pk == 1].iloc[0]
    assert list(row.home_order) == _NINE_H
    assert list(row.away_order) == _NINE_A
    assert row["source"] == "pregame" and bool(row["pit_valid"])
    # a NEWER pit-valid pre-game capture still upgrades (latest wins among
    # the trustworthy).
    update = _row(1, state="Preview", home_ids=[*list(_NINE_H[:8]), 999],
                  captured_utc="2026-09-16T19:00:00+00:00",
                  first_pitch_utc="2026-09-16T23:05:00+00:00",
                  source="pregame")
    assert bl.upsert_lineups(pd.DataFrame([update]), out_path=p) == 1
    row = pd.read_parquet(p).loc[lambda d: d.game_pk == 1].iloc[0]
    assert list(row.home_order)[-1] == 999


def _pbp_trip(game_pk, home_ids, away_ids):
    rows = []
    n = 1
    for side, ids in (("Bot", home_ids), ("Top", away_ids)):
        for b in ids:
            rows.append({"game_pk": game_pk, "inning_topbot": side,
                         "at_bat_number": n, "batter": b})
            n += 1
    return rows


def test_main_stamps_only_decisive_sides_and_never_rewrites_orders(
        tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(bl, "DATA_DELIVERY_DIR", tmp_path)
    day = "2026-09-16"
    # game 1 agrees; game 2 is the Sep-16 shape (2 swapped on the home side);
    # game 3 has no pbp coverage (stays NULL, retries later).
    g2_home = [901, 902, *_NINE_H[2:]]
    cache = pd.DataFrame([
        _row(1, date=day),
        _row(2, date=day, home_ids=g2_home),
        _row(3, date=day),
    ])
    # game 4: a hand-built self-contradicting claim ('pregame' from a Final
    # feed — the constructor gate can no longer produce this, the normalize
    # safety net must) with agreeing orders.
    g4 = pd.DataFrame([{
        "game_pk": 4, "game_date": day, "home_team": "NYY",
        "away_team": "BOS", "home_order": _NINE_H, "away_order": _NINE_A,
        "complete_home": True, "complete_away": True, "state": "Final",
        "captured_utc": "2026-09-16T18:00:00+00:00",
        "first_pitch_utc": "2026-09-16T23:05:00+00:00",
        "source": "pregame", "pit_valid": True,
        "validated_home": None, "validated_away": None,
        "home_starter_hand": "R", "away_starter_hand": "L",
    }])
    cache = pd.concat([cache, g4], ignore_index=True)
    bl.upsert_lineups(cache, out_path=tmp_path / "lineups.parquet")
    pd.DataFrame([
        {"game_pk": 1, "game_date": day, "home_team": "NYY",
         "away_team": "BOS", "home_win": 1},
        {"game_pk": 2, "game_date": day, "home_team": "NYY",
         "away_team": "BOS", "home_win": 0},
        {"game_pk": 3, "game_date": day, "home_team": "NYY",
         "away_team": "BOS", "home_win": 1},
        {"game_pk": 4, "game_date": day, "home_team": "NYY",
         "away_team": "BOS", "home_win": 1},
    ]).to_csv(tmp_path / "game_level_features.csv", index=False)
    pd.DataFrame(
        _pbp_trip(1, _NINE_H, _NINE_A)
        + _pbp_trip(2, _NINE_H, _NINE_A)
        + _pbp_trip(4, _NINE_H, _NINE_A)
    ).to_parquet(tmp_path / "pbp.parquet", index=False)
    monkeypatch.setattr(sys, "argv", ["backfill_lineups.py", "--limit", "10",
                                        "--csv", str(tmp_path / "game_level_features.csv"),
                                        "--pbp", str(tmp_path / "pbp.parquet")])
    bl.main()
    df = pd.read_parquet(tmp_path / "lineups.parquet")
    r1 = df.loc[df.game_pk == 1].iloc[0]
    assert (bool(r1.validated_home), bool(r1.validated_away)) == (True, True)
    r2 = df.loc[df.game_pk == 2].iloc[0]
    assert (bool(r2.validated_home), bool(r2.validated_away)) == (False, True)
    r3 = df.loc[df.game_pk == 3].iloc[0]
    assert pd.isna(r3.validated_home) and pd.isna(r3.validated_away)
    # orders are NEVER rewritten by validation (row selection, not repair)
    assert list(r2.home_order) == g2_home
    assert list(r1.home_order) == _NINE_H
    r4 = df.loc[df.game_pk == 4].iloc[0]
    assert r4["source"] == "backfill" and bool(r4["pit_valid"]) is False
    assert list(r4.home_order) == _NINE_H  # orders kept, claim voided
    assert (bool(r4.validated_home), bool(r4.validated_away)) == (True, True)


def test_main_accepts_fresh_games_frame(tmp_path, monkeypatch):
    """The full-repull path passes decided games enumerated from fresh
    pitches — the audit pass must work with NO game_level_features.csv on
    disk, proving zero dependence on the stored game list."""
    monkeypatch.setattr(bl, "DATA_DELIVERY_DIR", tmp_path)
    day = "2026-09-16"
    g2_home = [901, 902, *_NINE_H[2:]]
    cache = pd.DataFrame([
        _row(1, date=day),
        _row(2, date=day, home_ids=g2_home),
    ])
    bl.upsert_lineups(cache, out_path=tmp_path / "lineups.parquet")
    pd.DataFrame(
        _pbp_trip(1, _NINE_H, _NINE_A)
        + _pbp_trip(2, _NINE_H, _NINE_A)
    ).to_parquet(tmp_path / "pbp.parquet", index=False)
    games = pd.DataFrame([
        {"game_pk": 1, "game_date": day, "home_team": "NYY",
         "away_team": "BOS", "home_win": 1},
        {"game_pk": 2, "game_date": day, "home_team": "NYY",
         "away_team": "BOS", "home_win": 0},
    ])
    monkeypatch.setattr(sys, "argv", ["backfill_lineups.py", "--limit", "10",
                                        "--pbp", str(tmp_path / "pbp.parquet")])
    bl.main(games=games)
    assert not (tmp_path / "game_level_features.csv").exists()
    df = pd.read_parquet(tmp_path / "lineups.parquet")
    r1 = df.loc[df.game_pk == 1].iloc[0]
    assert (bool(r1.validated_home), bool(r1.validated_away)) == (True, True)
    r2 = df.loc[df.game_pk == 2].iloc[0]
    assert (bool(r2.validated_home), bool(r2.validated_away)) == (False, True)
    assert list(r2.home_order) == g2_home  # orders never rewritten


def test_decided_games_from_pitches(tmp_path):
    """Fresh-pitch enumeration: finals in (with the winner), everything
    else out — the full-repull game list without stored history."""
    day = "2026-09-16"
    pd.DataFrame([
        # decided home win
        {"game_pk": 1, "game_date": day, "home_team": "NYY",
         "away_team": "BOS", "post_home_score": 5, "post_away_score": 3},
        # decided away win
        {"game_pk": 2, "game_date": day, "home_team": "NYY",
         "away_team": "BOS", "post_home_score": 1, "post_away_score": 4},
        # in progress: no post scores
        {"game_pk": 3, "game_date": day, "home_team": "NYY",
         "away_team": "BOS", "post_home_score": None,
         "post_away_score": None},
    ]).to_parquet(tmp_path / "pitches.parquet", index=False)
    out = bl.decided_games_from_pitches(tmp_path / "pitches.parquet")
    assert set(out["game_pk"]) == {1, 2}
    by_pk = out.set_index("game_pk")
    assert bool(by_pk.loc[1, "home_win"]) is True
    assert bool(by_pk.loc[2, "home_win"]) is False
    assert set(by_pk.loc[1, ["home_team", "away_team"]]) == {"NYY", "BOS"}
    with pytest.raises(FileNotFoundError):
        bl.decided_games_from_pitches(tmp_path / "nope.parquet")


def test_state_column_survives_normalize_round_trip(tmp_path):
    p = _path(tmp_path)
    bl.upsert_lineups(pd.DataFrame([_row(1)]), out_path=p)
    assert pd.read_parquet(p).loc[0, "state"] == "Final"


def test_parse_feed_timecode():
    assert bl.parse_feed_timecode("20260916_155450") == \
        "2026-09-16T15:54:50+00:00"
    assert bl.parse_feed_timecode(None) is None
    assert bl.parse_feed_timecode("not-a-timecode") is None
    assert bl.parse_feed_timecode("") is None


def test_select_pregame_timecode_picks_latest_strictly_before_first_pitch():
    # the real Sep-16 timeline: snapshots 12:00/15:54/17:41, first pitch 17:42:43
    fp = "2026-09-16T17:42:43+00:00"
    tcs = ["20260916_120000", "20260916_155450", "20260916_174112"]
    assert bl.select_pregame_timecode(tcs, fp) == "20260916_174112"
    # strict: a snapshot AT first pitch is already post-pitch knowledge
    assert bl.select_pregame_timecode(["20260916_174243"], fp) is None
    assert bl.select_pregame_timecode([], fp) is None
    assert bl.select_pregame_timecode(tcs, None) is None
    assert bl.select_pregame_timecode(tcs, "garbage") is None
    # unparseable entries are skipped, never selected
    assert bl.select_pregame_timecode(["garbage", "20260916_120000"],
                                      fp) == "20260916_120000"


def _snap_feed(ho, ao, home="BOS", away="NYY", state="Preview",
               tc="20260916_155450"):
    return {"gameData": {"status": {"abstractGameState": state},
                           "teams": {"home": {"abbreviation": home},
                                     "away": {"abbreviation": away}}},
            "metaData": {"timeStamp": tc},
            "liveData": {"boxscore": {"teams": {
                "home": {"battingOrder": ho},
                "away": {"battingOrder": ao}}}}}


def test_snapshot_order_row_carries_feed_attested_provenance():
    row = bl.order_row(9001, "2026-09-16", "BOS", "NYY",
                       _snap_feed(_NINE_H, _NINE_A),
                       captured_utc="2026-09-16T15:54:50+00:00",
                       first_pitch_utc="2026-09-16T17:40:00+00:00",
                       source="pregame", feed_timecode="20260916_155450")
    assert row["source"] == "pregame" and row["pit_valid"] is True
    assert row["feed_timecode"] == "20260916_155450"
    assert row["state"] == "Preview"
    assert list(row["home_order"]) == _NINE_H


def test_backfill_pregame_snapshots_repairs_only_unproven_rows(
        tmp_path, monkeypatch):
    import results
    # game 1: legacy backfill-grade (repair me); game 2: already PIT-valid
    # (hands off); game 3: no qualifying snapshot (untouched, counted).
    seed = pd.DataFrame([
        _row(1, date="2026-09-16", home_starter_hand="R",
             away_starter_hand="L"),
        _row(2, date="2026-09-16", state="Preview",
             captured_utc="2026-09-16T14:00:00+00:00",
             first_pitch_utc="2026-09-16T23:05:00+00:00",
             source="pregame"),
        _row(3, date="2026-09-16"),
    ])
    bl.upsert_lineups(seed, out_path=_path(tmp_path))
    pd.DataFrame([
        {"game_pk": 1, "game_date": "2026-09-16", "home_team": "NYY",
         "away_team": "BOS", "home_win": 1},
        {"game_pk": 2, "game_date": "2026-09-16", "home_team": "NYY",
         "away_team": "BOS", "home_win": 0},
        {"game_pk": 3, "game_date": "2026-09-16", "home_team": "NYY",
         "away_team": "BOS", "home_win": 1},
    ]).to_csv(tmp_path / "game_level_features.csv", index=False)
    snap9_h = [901, 902, 903, 904, 905, 906, 907, 908, 909]
    monkeypatch.setattr(bl, "fetch_timecodes",
                          lambda pk: (["20260916_120000", "20260916_155450"]
                                     if pk == 1 else []))
    monkeypatch.setattr(bl, "fetch_snapshot",
                          lambda pk, tc, pause_sec=0.15: (
                              _snap_feed(snap9_h, _NINE_A), ""))
    monkeypatch.setattr(results, "fetch_game_start_times",
                          lambda a, b: {1: "2026-09-16T17:40:00+00:00",
                                       2: "2026-09-16T17:40:00+00:00",
                                       3: "2026-09-16T17:40:00+00:00"})
    monkeypatch.setattr(bl, "DATA_DELIVERY_DIR", tmp_path)
    import sys
    monkeypatch.setattr(sys, "argv", ["backfill_lineups.py"])
    n = bl.backfill_pregame_snapshots(
        limit=10, csv_path=tmp_path / "game_level_features.csv",
        out_path=_path(tmp_path))
    assert n == 1
    df = pd.read_parquet(_path(tmp_path))
    r1 = df.loc[df.game_pk == 1].iloc[0]
    assert r1["source"] == "pregame" and bool(r1["pit_valid"])
    assert r1["feed_timecode"] == "20260916_155450"
    assert list(r1.home_order) == snap9_h  # official snapshot wins
    assert r1["home_starter_hand"] == "R"  # hands preserved across repair
    r2 = df.loc[df.game_pk == 2].iloc[0]
    assert list(r2.home_order) == _NINE_H  # PIT-valid row untouched
    r3 = df.loc[df.game_pk == 3].iloc[0]
    assert r3["source"] == "backfill"  # no snapshot: stays grandfathered


def test_load_pregame_orders_reads_only_requested_pks(tmp_path, monkeypatch):
    monkeypatch.setattr(bl, "DATA_DELIVERY_DIR", tmp_path)
    assert bl.load_pregame_orders([1]).empty  # absent cache, never raises
    bl.upsert_lineups(pd.DataFrame([_row(1), _row(2)]), out_path=_path(tmp_path))
    got = bl.load_pregame_orders([2, 999])
    assert set(got.game_pk) == {2}
    assert list(got.iloc[0].home_order) == _NINE_H
    for c in ("source", "pit_valid", "validated_home", "validated_away",
              "complete_home", "complete_away"):
        assert c in got.columns, c


def _snapshot_csv(path, pks):
    pd.DataFrame([
        {"game_pk": pk, "game_date": "2026-09-16", "home_team": "NYY",
         "away_team": "BOS", "home_win": 1}
        for pk in pks
    ]).to_csv(path, index=False)


def test_snapshot_repair_limit_none_processes_all_eligible_games_in_order(
        tmp_path, monkeypatch):
    import results
    seed = pd.DataFrame([
        _row(1),
        _row(2, state="Preview",
              captured_utc="2026-09-16T14:00:00+00:00",
              first_pitch_utc="2026-09-16T23:05:00+00:00",
              source="pregame"),
        _row(3),
        _row(4),
    ])
    bl.upsert_lineups(seed, out_path=_path(tmp_path))
    _snapshot_csv(tmp_path / "game_level_features.csv", [1, 2, 3, 4])

    def fake_snapshot(pk, tc, pause_sec=0.15):
        home = [900 + pk * 10 + i for i in range(9)]
        return _snap_feed(home, _NINE_A, tc=tc), ""

    monkeypatch.setattr(bl, "fetch_timecodes",
                          lambda pk: (["20260916_120000", "20260916_155450"]
                                     if pk in (1, 4) else []))
    monkeypatch.setattr(bl, "fetch_snapshot", fake_snapshot)
    monkeypatch.setattr(results, "fetch_game_start_times",
                          lambda a, b: {pk: "2026-09-16T17:40:00+00:00"
                                       for pk in (1, 2, 3, 4)})
    n = bl.backfill_pregame_snapshots(
        limit=None, csv_path=tmp_path / "game_level_features.csv",
        out_path=_path(tmp_path), max_workers=2)
    assert n == 2
    df = pd.read_parquet(_path(tmp_path))
    # Input order is preserved even though two workers check games.
    assert df.sort_values("game_pk").loc[
        lambda d: d.game_pk.isin([1, 4]), "feed_timecode"].tolist() == \
        ["20260916_155450", "20260916_155450"]
    r2 = df.loc[df.game_pk == 2].iloc[0]
    assert r2["source"] == "pregame" and bool(r2["pit_valid"])
    assert list(r2.home_order) == _NINE_H  # already-proven row untouched


def test_snapshot_full_rebuild_reprocesses_provenance_and_preserves_audit(
        tmp_path, monkeypatch):
    import results
    seed = pd.DataFrame([_row(
        1, state="Preview", captured_utc="2026-09-16T14:00:00+00:00",
        first_pitch_utc="2026-09-16T23:05:00+00:00", source="pregame",
        validated_home=True, validated_away=False,
        home_starter_hand="R", away_starter_hand="L")])
    bl.upsert_lineups(seed, out_path=_path(tmp_path))
    _snapshot_csv(tmp_path / "game_level_features.csv", [1])
    repaired_home = [901, 902, 903, 904, 905, 906, 907, 908, 909]
    monkeypatch.setattr(bl, "fetch_timecodes",
                          lambda pk: ["20260916_120000", "20260916_155450"])
    monkeypatch.setattr(bl, "fetch_snapshot",
                          lambda pk, tc, pause_sec=0.15: (
                              _snap_feed(repaired_home, _NINE_A, tc=tc), ""))
    monkeypatch.setattr(results, "fetch_game_start_times",
                          lambda a, b: {1: "2026-09-16T17:40:00+00:00"})
    n = bl.backfill_pregame_snapshots(
        limit=None, csv_path=tmp_path / "game_level_features.csv",
        out_path=_path(tmp_path), full_rebuild=True, max_workers=2)
    assert n == 1
    row = pd.read_parquet(_path(tmp_path)).iloc[0]
    assert list(row.home_order) == repaired_home
    assert row["source"] == "pregame" and bool(row["pit_valid"])
    assert bool(row.validated_home) is True
    assert bool(row.validated_away) is False
    assert row["home_starter_hand"] == "R"
    assert row["away_starter_hand"] == "L"


def test_snapshot_repair_forwards_sparse_fields(tmp_path, monkeypatch):
    import results
    bl.upsert_lineups(pd.DataFrame([_row(1)]), out_path=_path(tmp_path))
    _snapshot_csv(tmp_path / "game_level_features.csv", [1])
    seen = {}

    def fake_snapshot(pk, tc, pause_sec=0.15, fields=None):
        seen["fields"] = fields
        return _snap_feed(_NINE_H, _NINE_A, tc=tc), ""

    monkeypatch.setattr(bl, "fetch_timecodes",
                          lambda pk: ["20260916_155450"])
    monkeypatch.setattr(bl, "fetch_snapshot", fake_snapshot)
    monkeypatch.setattr(results, "fetch_game_start_times",
                          lambda a, b: {1: "2026-09-16T17:40:00+00:00"})
    n = bl.backfill_pregame_snapshots(
        limit=1, csv_path=tmp_path / "game_level_features.csv",
        out_path=_path(tmp_path), snapshot_fields="CUSTOM_FIELDS")
    assert n == 1
    assert seen["fields"] == "CUSTOM_FIELDS"


def test_snapshot_repair_names_every_grandfathered_game(tmp_path, monkeypatch,
                                                       capsys):
    import results
    bl.upsert_lineups(pd.DataFrame([_row(1), _row(3)]),
                      out_path=_path(tmp_path))
    _snapshot_csv(tmp_path / "game_level_features.csv", [1, 3])
    monkeypatch.setattr(bl, "fetch_timecodes",
                          lambda pk: ["20260916_155450"] if pk == 1 else [])
    monkeypatch.setattr(bl, "fetch_snapshot",
                          lambda pk, tc, pause_sec=0.15: (None, "boom"))
    monkeypatch.setattr(results, "fetch_game_start_times",
                          lambda a, b: {1: "2026-09-16T17:40:00+00:00",
                                       3: "2026-09-16T17:40:00+00:00"})
    n = bl.backfill_pregame_snapshots(
        limit=None, csv_path=tmp_path / "game_level_features.csv",
        out_path=_path(tmp_path))
    assert n == 0
    out = capsys.readouterr().out
    assert "grandfathered game_pks by reason" in out
    assert "fetch_failed=[1]" in out
    assert "no_snapshot=[3]" in out


def test_snapshot_full_rebuild_does_not_list_retained_proven_games(
        tmp_path, monkeypatch, capsys):
    import results
    bl.upsert_lineups(pd.DataFrame([
        _row(1),
        _row(2, state="Preview",
              captured_utc="2026-09-16T14:00:00+00:00",
              first_pitch_utc="2026-09-16T23:05:00+00:00",
              source="pregame"),
    ]), out_path=_path(tmp_path))
    _snapshot_csv(tmp_path / "game_level_features.csv", [1, 2])
    monkeypatch.setattr(bl, "fetch_timecodes",
                          lambda pk: ["20260916_155450"])
    monkeypatch.setattr(bl, "fetch_snapshot",
                          lambda pk, tc, pause_sec=0.15: (
                              (None, "boom") if pk == 2
                              else (_snap_feed(_NINE_H, _NINE_A, tc=tc), "")))
    monkeypatch.setattr(results, "fetch_game_start_times",
                          lambda a, b: {1: "2026-09-16T17:40:00+00:00",
                                       2: "2026-09-16T17:40:00+00:00"})
    n = bl.backfill_pregame_snapshots(
        limit=None, csv_path=tmp_path / "game_level_features.csv",
        out_path=_path(tmp_path), full_rebuild=True)
    assert n == 1
    out = capsys.readouterr().out
    assert "1 snapshot fetches failed" in out
    assert "fetch_failed=[2]" not in out


# ── boxscore actuals (2026-10-26, training truth) ────────────────────────────

_BOX_ACTUAL_H = [701, 702, 703, 704, 705, 706, 707, 708, 709]
_BOX_ACTUAL_A = [601, 602, 603, 604, 605, 606, 607, 608, 609]


def _boxscore(home_ids=_BOX_ACTUAL_H, away_ids=_BOX_ACTUAL_A):
    # battingOrder is the starting nine; `batters` carries subs/pinch
    # hitters that must NEVER enter training orders.
    return {"teams": {
        "home": {"battingOrder": list(home_ids),
                   "batters": list(home_ids) + [999]},
        "away": {"battingOrder": list(away_ids),
                   "batters": list(away_ids) + [998, 997]}}}


def test_boxscore_nines_reads_batting_order_only():
    nines = bl.boxscore_nines(_boxscore())
    assert nines["home"] == _BOX_ACTUAL_H
    assert nines["away"] == _BOX_ACTUAL_A
    assert 999 not in nines["home"] and 998 not in nines["away"]
    assert bl.boxscore_nines(None) == {"home": [], "away": []}
    assert bl.boxscore_nines({"teams": {"home": {}}})["home"] == []


def _actuals_games(*pks):
    return pd.DataFrame([
        {"game_pk": pk, "game_date": "2026-09-16", "home_team": "NYY",
         "away_team": "BOS"} for pk in pks])


def test_backfill_actual_nines_upserts_actuals_source(tmp_path, monkeypatch):
    monkeypatch.setattr(bl, "fetch_boxscore",
                          lambda pk, pause_sec=0.15: (_boxscore(), ""))
    n = bl.backfill_actual_nines(_actuals_games(11, 12), limit=10,
                                  out_path=_path(tmp_path), max_workers=1)
    assert n == 2
    df = pd.read_parquet(_path(tmp_path))
    assert set(df.game_pk) == {11, 12}
    r = df.loc[df.game_pk == 11].iloc[0]
    assert r["source"] == "actuals" and bool(r["pit_valid"]) is False
    assert r["state"] == "Final"
    assert list(r.home_order) == _BOX_ACTUAL_H
    assert list(r.away_order) == _BOX_ACTUAL_A
    # incremental re-run: actuals-complete games are skipped, not re-pulled
    calls = []
    monkeypatch.setattr(bl, "fetch_boxscore",
                          lambda pk, pause_sec=0.15: (calls.append(pk),
                                                     (_boxscore(), ""))[1])
    n = bl.backfill_actual_nines(_actuals_games(11, 12, 13), limit=10,
                                  out_path=_path(tmp_path), max_workers=1)
    assert n == 1 and calls == [13]


def test_upsert_actuals_rows_win_and_survive(tmp_path):
    p = _path(tmp_path)
    stale = bl.order_row(21, "2026-09-16", "NYY", "BOS",
                         _feed(_NINE_H, _NINE_A))
    truth = bl.actuals_row(21, "2026-09-16", "NYY", "BOS",
                           {"home": _BOX_ACTUAL_H, "away": _BOX_ACTUAL_A})
    assert truth["source"] == "actuals" and bool(truth["pit_valid"]) is False
    bl.upsert_lineups(pd.DataFrame([stale]), out_path=p)
    bl.upsert_lineups(pd.DataFrame([truth]), out_path=p)
    df = pd.read_parquet(p)
    assert list(df.loc[df.game_pk == 21].iloc[0].home_order) == _BOX_ACTUAL_H
    # a later stale write must not clobber training truth
    bl.upsert_lineups(pd.DataFrame([stale]), out_path=p)
    df = pd.read_parquet(p)
    r = df.loc[df.game_pk == 21].iloc[0]
    assert r["source"] == "actuals"
    assert list(r.home_order) == _BOX_ACTUAL_H
    # actuals refresh still wins (idempotent re-pull)
    truth2 = dict(truth)
    truth2["away_order"] = list(_BOX_ACTUAL_A)
    bl.upsert_lineups(pd.DataFrame([truth2]), out_path=p)
    df = pd.read_parquet(p)
    assert df.loc[df.game_pk == 21].iloc[0]["source"] == "actuals"


def test_scheduled_capture_is_pregame(tmp_path, monkeypatch):
    import requests
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    far = (now + timedelta(hours=3)).isoformat()
    near = (now + timedelta(minutes=2)).isoformat()

    def fake_get(url, params=None, timeout=None):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"dates": [{"games": [
                    _sched_game(31, far), _sched_game(32, near)]}]}
        return R()

    def fake_feed(pk, pause_sec=0.15):
        return _feed(_NINE_H, _NINE_A, state="Preview"), ""

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(bl, "fetch_feed", fake_feed)
    n = bl.fetch_scheduled_lineups("2026-09-16",
                                    out_path=tmp_path / "slate.parquet")
    assert n == 2
    df = pd.read_parquet(tmp_path / "slate.parquet")
    assert set(df.game_pk) == {31, 32}
    assert (df["source"] == "pregame").all()
    assert df["pit_valid"].astype(bool).all()


def _sched_game(pk, game_date):
    return {"gamePk": pk, "gameDate": game_date,
            "teams": {"away": {"team": {"abbreviation": "BOS"}},
                      "home": {"team": {"abbreviation": "NYY"}}}}


def test_full_repull_wiring_uses_no_stored_history():
    """MLB_FULL_REPULL=1 must replicate from sources, never stored files.

    Pins the master_pipeline wiring (2026-10-26 remediation): the IL ledger
    rebuild re-pulls every transaction year (--refresh, not the run cache's
    il_tx JSONs), and the lineup audit pass reads this run's freshly-pulled
    pitches (--pbp) instead of the repo's data_delivery/pitches.parquet —
    which does not exist on a fresh clone, where the audit would otherwise
    silently defer and the full rebuild would validate nothing.
    """
    mp_src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    assert '["--refresh"] if full_repull else []' in mp_src, (
        "Phase 1.5 must re-pull IL transactions under MLB_FULL_REPULL")
    assert '"--pbp", str(pitches_path)' in mp_src, (
        "the lineup audit pass must read this run's fresh pitches")


