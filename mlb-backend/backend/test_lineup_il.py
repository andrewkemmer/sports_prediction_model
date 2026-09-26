"""Pin the injured-list expected-lineup contract.

The feature ships in ``features.py``'s ``lineup_agg``, so these tests drive
the SHIPPED SQL constants against a synthetic ``batter_ratings`` table rather
than a re-implementation: if someone edits the pool or the IL predicate, this
fails. Everything here is hermetic (in-memory DuckDB, no data_delivery read)
except the two parity/retention checks, which assert the serving contract
rather than a computed value.

What is pinned, and why each one matters:

* the pool WIDENS past actual participants -- without this the IL filter is a
  no-op, because an injured player is absent from the participant pool by
  construction. This is the change that makes the treatment possible at all.
* the IL predicate binds, respects stint bounds in BOTH directions, and
  treats an open stint (NULL end) as still injured.
* a replacement is promoted, i.e. the vacated slot is refilled from the
  healthy tail rather than left short.
* the participant fallback keeps the injured player -- the documented
  degradation when the cache is missing, so it must not silently start
  filtering.
* both SQL variants emit the SAME columns: the serving contract.
* DFA / outright-send close a stint, rehab does not, and a trade or waiver
  claim does not (the receiving team inherits the IL spot).
* the feature is in BOTH model widths, and both models resolve one list.
* the table is exempt from the dated-artifact cleanup that would delete it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import build_il_stints  # noqa: E402
import features  # noqa: E402
import retention_policy  # noqa: E402
import run_engine  # noqa: E402
import training  # noqa: E402

GAME = pd.Timestamp("2024-07-10")
GPK = 745001
TEAM = "BOS"
# The 8 columns whose VALUES this change moves, plus the composite derived
# from them. Every one must stay in the serving width.
SWAPPED = (
    "lineup_woba_mean_home", "lineup_woba_mean_away",
    "lineup_woba_top3_home", "lineup_woba_top3_away",
    "lineup_woba_mean_diff", "lineup_woba_top3_diff",
    "lineup_woba_std_diff", "lineup_depth_multiplier",
)
EXPECTED_COLS = ["game_date", "game_pk", "batting_team",
                 "lineup_woba_mean", "lineup_woba_top3", "lineup_woba_std"]


def _con(ratings: pd.DataFrame, stints: pd.DataFrame | None = None
         ) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.register("ratings_src", ratings)
    con.execute("""
        CREATE TABLE batter_ratings AS
        SELECT CAST(game_date AS DATE) AS game_date, CAST(game_pk AS BIGINT)
                   AS game_pk, batting_team, CAST(batter AS BIGINT) AS batter,
               shrunk_woba, _pa30
        FROM ratings_src
    """)
    if stints is not None:
        con.register("stints_src", stints)
        con.execute("""
            CREATE TABLE il_stints AS
            SELECT CAST(batter AS BIGINT) AS batter,
                   CAST(il_start AS DATE) AS il_start,
                   CAST(il_end AS DATE) AS il_end
            FROM stints_src
        """)
    else:
        # The shipped SQL always has the table (possibly empty); mirror that
        # so the "no stints at all" case exercises the real plan.
        con.execute("CREATE TABLE il_stints (batter BIGINT, il_start DATE, "
                    "il_end DATE)")
    return con


def _agg(sql: str, ratings: pd.DataFrame,
         stints: pd.DataFrame | None) -> pd.DataFrame:
    con = _con(ratings, stints)
    try:
        con.execute(sql)
        return con.execute("SELECT * FROM lineup_agg").df()
    finally:
        con.close()


def _roster(ratings, stints):
    return _agg(features._LINEUP_AGG_ROSTER.format(
        lookback=features.LINEUP_POOL_LOOKBACK_DAYS), ratings, stints)


def _participants(ratings, stints):
    return _agg(features._LINEUP_AGG_PARTICIPANTS, ratings, stints)


def _ratings(rows) -> pd.DataFrame:
    """rows: (game_date, game_pk, batter, shrunk_woba, _pa30).

    ``game_pk`` matters: ``batter_ratings`` carries the game a rating came
    FROM, and the target game-side universe is the distinct
    (game_date, game_pk, batting_team) of those rows. A rating row therefore
    creates its own game-side -- handing the backup the target game's pk
    would invent a second game-side for the same game.
    """
    return pd.DataFrame(
        [{"game_date": d, "game_pk": pk, "batting_team": TEAM, "batter": b,
          "shrunk_woba": w, "_pa30": pa} for d, pk, b, w, pa in rows])


def _ten(backup_pa30: int = 400) -> pd.DataFrame:
    """Ten candidates.

    Nine of them batted in THIS game: batter 100+i carries wOBA 0.30+0.01i and
    a trailing PA count of 500-10i, so he is the NINTH by playing time even
    though he is the best hitter of the nine -- the ordering the top-9 cut
    actually uses (top-3 is by playing time too, not by wOBA).

    The tenth, batter 200, rated three days earlier and did NOT bat in this
    game (a call-up who missed the previous one). ``backup_pa30`` sets where
    he ranks: 600 puts him INSIDE the projected nine (proving the pool
    widened), 400 leaves him tenth (so only the IL filter can promote him,
    which keeps the two effects separable).
    """
    rows = [(GAME, GPK, 100 + i, 0.30 + 0.01 * i, 500 - 10 * i)
            for i in range(9)]
    rows.append((GAME - pd.Timedelta(days=3), GPK - 1, 200, 0.25, backup_pa30))
    return _ratings(rows)


def _target(out: pd.DataFrame) -> pd.Series:
    """The GPK game-side row.

    The aggregate has one row per (game_date, game_pk, batting_team) in
    batter_ratings, and the backup's own rating row is a game-side of its
    own -- so select the target game explicitly rather than trusting row
    order.
    """
    rows = out[out.game_pk == GPK]
    assert len(rows) == 1, f"expected exactly one {GPK} game-side, got {len(rows)}"
    return rows.iloc[0]


def _participants_mean() -> float:
    return sum(0.30 + 0.01 * i for i in range(9)) / 9


def _with_backup_mean() -> float:
    """Mean once batter 108 is out and the backup (0.25) is in."""
    return (0.25 + sum(0.30 + 0.01 * i for i in range(8))) / 9


# ── the pool must widen, or the filter can never bind ──────────────────────

def test_pool_widens_past_actual_participants():
    """Batter 200 never batted in this game yet ranks FIRST by playing time,
    so the widened pool must project him into the nine. Under the
    participant pool he is invisible, which is exactly why an IL filter over
    that pool is a no-op."""
    out = _roster(_ten(backup_pa30=600), None)
    got = _target(out)
    # Top 9 by _pa30: 200 (600) then 100..107 -- batter 108 is displaced.
    expected = (0.25 + sum(0.30 + 0.01 * i for i in range(8))) / 9
    assert got["lineup_woba_mean"] == pytest.approx(expected)
    assert got["lineup_woba_mean"] < _participants_mean()


def test_participant_pool_excludes_the_non_participant():
    """Same data through the fallback: only the 9 who batted are candidates,
    so the backup cannot displace anyone."""
    out = _participants(_ten(backup_pa30=600), None)
    assert _target(out)["lineup_woba_mean"] == pytest.approx(
        _participants_mean())


# ── the IL predicate ───────────────────────────────────────────────────────

def test_il_player_is_replaced_by_the_healthy_backup():
    """Batter 108 (the 9th) is on the IL. He leaves the top nine and the
    next healthy candidate (#200) takes the slot -- the feature reports a
    depleted roster, it does not pad or fabricate a full-strength nine."""
    ratings = _ten()
    stints = pd.DataFrame([{"batter": 108, "il_start": GAME - pd.Timedelta(days=2),
                            "il_end": GAME + pd.Timedelta(days=5)}])
    out = _roster(ratings, stints)
    got = _target(out)
    assert got["lineup_woba_mean"] == pytest.approx(_with_backup_mean())
    # No padding: with 9 healthy candidates the count stays 9, and the
    # shortfall case is covered by test_thin_roster_is_not_padded.
    assert got["lineup_woba_std"] > 0


def test_thin_roster_is_not_padded():
    """Eight candidates, two on the IL -> the mean of the SIX healthy ones.
    Padding would fabricate a full-strength nine."""
    ratings = _ratings([(GAME, GPK, 100 + i, 0.30 + 0.01 * i, 500 - i)
                        for i in range(8)])
    stints = pd.DataFrame([{"batter": 106, "il_start": GAME - pd.Timedelta(days=1),
                            "il_end": pd.NaT},
                           {"batter": 107, "il_start": GAME - pd.Timedelta(days=1),
                            "il_end": pd.NaT}])
    out = _roster(ratings, stints)
    assert _target(out)["lineup_woba_mean"] == pytest.approx(
        sum(0.30 + 0.01 * i for i in range(6)) / 6)


def test_open_stint_counts_as_injured():
    """NULL il_end means the stint never closed -> still injured. Testing
    only for a NULL join key here flagged 59.6% of all participants."""
    ratings = _ten()
    stints = pd.DataFrame([{"batter": 108, "il_start": GAME - pd.Timedelta(days=2),
                            "il_end": pd.NaT}])
    out = _roster(ratings, stints)
    assert _target(out)["lineup_woba_mean"] == pytest.approx(_with_backup_mean())


@pytest.mark.parametrize("start,end,still_out", [
    (GAME - pd.Timedelta(days=1), GAME + pd.Timedelta(days=1), True),   # active
    (GAME - pd.Timedelta(days=30), GAME - pd.Timedelta(days=1), False),  # back
    (GAME - pd.Timedelta(days=30), GAME, False),                         # back ON the day: a
                                                                       # game-date stint has
                                                                       # ended (PIT: an
                                                                       # activation on game
                                                                       # day means he plays)
    (GAME + pd.Timedelta(days=1), GAME + pd.Timedelta(days=5), False),   # not yet
])
def test_stint_bounds_respected(start, end, still_out):
    ratings = _ten()
    stints = pd.DataFrame([{"batter": 108, "il_start": start, "il_end": end}])
    out = _roster(ratings, stints)
    got = _target(out)["lineup_woba_mean"]
    if still_out:
        assert got == pytest.approx(_with_backup_mean())
    else:
        assert got == pytest.approx(_participants_mean())


def test_fallback_does_not_filter():
    """The degradation path must be the UNFILTERED participant pool. If it
    ever started filtering, a missing cache would look like a working build
    with a different (undocumented) semantics."""
    ratings = _ten()
    stints = pd.DataFrame([{"batter": 108, "il_start": GAME - pd.Timedelta(days=2),
                            "il_end": GAME + pd.Timedelta(days=5)}])
    out = _participants(ratings, stints)
    assert _target(out)["lineup_woba_mean"] == pytest.approx(
        _participants_mean())


def test_both_variants_emit_the_same_schema():
    """The serving contract: same columns, same order, whatever the path."""
    ratings, stints = _ten(), None
    assert list(_participants(ratings, stints).columns) == EXPECTED_COLS
    assert list(_roster(ratings, stints).columns) == EXPECTED_COLS


# ── the three-state stint machine ─────────────────────────────────────────

def _machine(events):
    return build_il_stints.stints_from_events(
        build_il_stints.build_events(events))


def _tx(desc, date="2024-07-01", pid=1):
    return {"description": desc, "date": date, "person": {"id": pid}}


def test_activation_closes_a_stint():
    iv = _machine([_tx("Boston Red Sox placed LF A on the 10-day injured list."),
                   _tx("Boston Red Sox activated LF A.", "2024-07-20")])
    assert len(iv) == 1
    assert iv.il_start.iloc[0] == pd.Timestamp("2024-07-01")
    assert iv.il_end.iloc[0] == pd.Timestamp("2024-07-20")


def test_bare_activation_still_closes():
    """Only 2,037 of 3,589 activations repeat the words 'injured list';
    matching closes on /injur/ discards 1,552 real closings."""
    iv = _machine([_tx("Boston Red Sox placed LF A on the 10-day injured list."),
                   _tx("Boston Red Sox activated RHP A.", "2024-07-20")])
    assert len(iv) == 1 and pd.notna(iv.il_end.iloc[0])


def test_designated_for_assignment_closes_a_stint():
    """A club routinely DFA's an IL player to clear a 40-man spot, and MLB
    files NO activation because he never came back. Without this the player
    stays 'on the IL' for years (measured: 7 real leaguers, 3,653 PA)."""
    iv = _machine([_tx("Boston Red Sox placed LF A on the 10-day injured list."),
                   _tx("Boston Red Sox designated LF A for assignment.",
                       "2024-07-20")])
    assert len(iv) == 1 and iv.il_end.iloc[0] == pd.Timestamp("2024-07-20")


def test_outright_send_closes_a_stint():
    iv = _machine([_tx("Boston Red Sox placed LF A on the 10-day injured list."),
                   _tx("Boston Red Sox sent LF A outright to Worcester Red Sox.",
                       "2024-07-20")])
    assert len(iv) == 1 and pd.notna(iv.il_end.iloc[0])


def test_rehab_does_not_close_a_stint():
    """A player on rehab is unavailable to the active roster, which is the
    only question this table is asked."""
    iv = _machine([_tx("Boston Red Sox placed LF A on the 10-day injured list."),
                   _tx("Boston Red Sox sent LF A on a rehab assignment to "
                       "Worcester Red Sox.", "2024-07-20")])
    assert len(iv) == 1 and pd.isna(iv.il_end.iloc[0])


@pytest.mark.parametrize("desc", [
    "Baltimore Orioles claimed RF A off waivers from Atlanta Braves.",
    "New York Yankees traded LF A to Boston Red Sox.",
])
def test_roster_moves_do_not_close_a_stint(desc):
    """The receiving team INHERITS the IL spot, so a claim or trade is not
    a return to health."""
    iv = _machine([_tx("Atlanta Braves placed RF A on the 10-day injured list.",
                       "2024-07-01"),
                   _tx(desc, "2024-07-20")])
    assert len(iv) == 1 and pd.isna(iv.il_end.iloc[0])


def test_escalation_does_not_open_a_second_stint():
    """A 10-day -> 60-day escalation and a retroactive re-filing are extra
    placement records for ONE stint; pairing them against one activation
    left phantoms open and inflated the league-wide count to ~1,117."""
    iv = _machine([
        _tx("Atlanta Braves placed RF A on the 10-day injured list.", "2024-07-01"),
        _tx("Atlanta Braves placed RF A on the 60-day injured list.", "2024-07-20"),
        _tx("Atlanta Braves activated RF A.", "2024-08-15")])
    assert len(iv) == 1
    assert iv.il_start.iloc[0] == pd.Timestamp("2024-07-01")
    assert iv.il_end.iloc[0] == pd.Timestamp("2024-08-15")


def test_state_machine_never_dips_below_zero():
    """An activation with no open stint (e.g. its placement was filed under
    wording we do not match) must be ignored, not emit a negative stint."""
    iv = _machine([_tx("Boston Red Sox activated LF A.", "2024-07-20")])
    assert iv.empty or (iv.il_end >= iv.il_start).all()


# ── reconciliation against observed plate appearances ───────────────────────
#
# The shipped A/B came back net negative and the cause was not the feature:
# 3,522 player-games where the table said "on the IL" and the batter had a
# plate appearance. The transactions feed never closes a player whose return
# is filed as a minor-league recall, so one interval ran three years over a
# batter who played 320 games inside it.

def _pa(*rows):
    return pd.DataFrame(rows, columns=["batter", "game_date"])


def _ivs(*rows):
    return pd.DataFrame(rows, columns=["batter", "il_start", "il_end"])


def test_appearance_after_placement_closes_an_unclosed_stint():
    """No activation row ever arrives for a player who rehabbed repeatedly
    and shuttled to the minors; his appearance is the only return signal."""
    iv = _ivs([1, pd.Timestamp("2023-05-19"), pd.NaT])
    out = build_il_stints.reconcile_with_plate_appearances(
        iv, _pa([1, pd.Timestamp("2026-05-16")]))
    assert len(out) == 1
    assert out.il_end.iloc[0] == pd.Timestamp("2026-05-16")


def test_appearance_truncates_a_late_transaction_close():
    iv = _ivs([1, pd.Timestamp("2024-06-01"), pd.Timestamp("2024-09-01")])
    out = build_il_stints.reconcile_with_plate_appearances(
        iv, _pa([1, pd.Timestamp("2024-06-20")]))
    assert out.il_end.iloc[0] == pd.Timestamp("2024-06-20")


def test_earlier_transaction_close_is_left_alone():
    """Reconciling must never move a close EARLIER: the batter played inside
    the stint, so the stint was real and ended when the club activated him."""
    iv = _ivs([1, pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-10")])
    out = build_il_stints.reconcile_with_plate_appearances(
        iv, _pa([1, pd.Timestamp("2024-06-20")]))
    assert out.il_end.iloc[0] == pd.Timestamp("2024-06-10")


def test_same_day_appearance_does_not_close_the_stint():
    """Clubs file the IL transaction the same evening a player is hurt, so a
    plate appearance ON the placement date is the announcement, not a return.
    All 19 residual cases in the real table are exactly this."""
    iv = _ivs([1, pd.Timestamp("2024-06-01"), pd.NaT])
    out = build_il_stints.reconcile_with_plate_appearances(
        iv, _pa([1, pd.Timestamp("2024-06-01")]))
    assert out.il_end.iloc[0] is pd.NaT or pd.isna(out.il_end.iloc[0])


def test_reconcile_never_releases_a_date_the_batter_did_not_play():
    """The rule is strictly subtractive: it can only shorten an interval at a
    date where the batter was in the lineup, so a genuine absence is never
    handed back to the projected nine."""
    iv = _ivs([1, pd.Timestamp("2024-06-01"), pd.Timestamp("2024-07-01")])
    absent = pd.Timestamp("2024-06-15")
    out = build_il_stints.reconcile_with_plate_appearances(
        iv, _pa([1, pd.Timestamp("2024-08-01")]))
    assert out.il_start.iloc[0] <= absent < out.il_end.iloc[0]


def test_reconcile_drops_a_stint_collapsed_onto_its_own_start():
    iv = _ivs([1, pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-01")])
    out = build_il_stints.reconcile_with_plate_appearances(
        iv, _pa([1, pd.Timestamp("2024-06-02")]))
    assert out.empty


def test_reconcile_keeps_players_with_no_observed_appearances():
    """A batter outside the pitch window is not evidence of anything; his
    stint must survive untouched rather than being closed by a lookup miss."""
    iv = _ivs([7, pd.Timestamp("2024-06-01"), pd.NaT])
    out = build_il_stints.reconcile_with_plate_appearances(
        iv, _pa([1, pd.Timestamp("2024-06-20")]))
    assert len(out) == 1 and pd.isna(out.il_end.iloc[0])


def test_reconcile_is_a_noop_without_appearances():
    iv = _ivs([1, pd.Timestamp("2024-06-01"), pd.NaT])
    out = build_il_stints.reconcile_with_plate_appearances(iv, _pa())
    assert len(out) == 1 and pd.isna(out.il_end.iloc[0])


def test_residual_counter_finds_the_defect_and_the_fix_clears_it():
    """The tripwire. It is the reason this class of bug cannot come back
    silently: a table that claims a batter is hurt while he bats fails the
    build instead of quietly degrading the projection."""
    iv = _ivs([1, pd.Timestamp("2024-06-01"), pd.NaT])
    pa = _pa([1, pd.Timestamp("2024-06-20")], [1, pd.Timestamp("2024-07-20")])
    assert build_il_stints.residual_on_il_player_games(iv, pa) == 2
    fixed = build_il_stints.reconcile_with_plate_appearances(iv, pa)
    assert build_il_stints.residual_on_il_player_games(fixed, pa) == 0


def test_residual_counter_counts_a_player_game_once():
    """Overlapping intervals must not double-count, or the tripwire fires on
    a table that is merely redundant (this bug inflated the count 3,522 ->
    3,813 before it was found)."""
    iv = _ivs([1, pd.Timestamp("2024-06-01"), pd.Timestamp("2024-08-01")],
              [1, pd.Timestamp("2024-07-01"), pd.NaT])
    pa = _pa([1, pd.Timestamp("2024-07-15")])
    assert build_il_stints.residual_on_il_player_games(iv, pa) == 1


def test_residual_counter_ignores_same_day_appearances():
    iv = _ivs([1, pd.Timestamp("2024-06-01"), pd.NaT])
    pa = _pa([1, pd.Timestamp("2024-06-01")])
    assert build_il_stints.residual_on_il_player_games(iv, pa) == 0


def test_stale_pitch_projection_is_refused():
    """A projection that stops months early makes reconciliation a silent
    no-op AND blinds the residual counter, which measures against the same
    file -- the table would look clean while every stint after the cut stayed
    unclosed. This is the one way the defect can come back unobserved."""
    pa = _pa([1, pd.Timestamp("2026-09-24")])
    build_il_stints.check_pbp_covers(
        "pbp.parquet", pa, pd.Timestamp("2026-09-25"))  # current: fine
    with pytest.raises(SystemExit):
        build_il_stints.check_pbp_covers(
            "pbp.parquet", pa, pd.Timestamp("2026-12-31"))
    with pytest.raises(SystemExit):
        build_il_stints.check_pbp_covers("pbp.parquet", _pa(),
                                         pd.Timestamp("2026-09-25"))


def test_shipped_table_has_no_residual_defect():
    """The committed table itself, read from data_delivery, against the pitch
    projection. Skips when either artifact is absent (fresh clone)."""
    import config
    dd = Path(config.DATA_DELIVERY_DIR)
    pbp = sorted(dd.glob("pbp_defense_*.parquet"))
    if not (dd / "il_stints.parquet").exists() or not pbp:
        pytest.skip("il_stints.parquet or the pitch projection is absent")
    iv = pd.read_parquet(dd / "il_stints.parquet")
    pa = build_il_stints.load_plate_appearances(pbp[-1])
    residual = build_il_stints.residual_on_il_player_games(iv, pa)
    assert residual <= build_il_stints._MAX_RESIDUAL_ON_IL_PGAMES, (
        f"{residual} player-games claim on-IL while batting")


def test_shipped_table_stays_inside_the_plausibility_band():
    import json
    import config
    dd = Path(config.DATA_DELIVERY_DIR)
    if not (dd / "il_stints.meta.json").exists():
        pytest.skip("il_stints.meta.json is absent")
    meta = json.loads((dd / "il_stints.meta.json").read_text())
    lo, hi = meta["gate_band"]
    assert lo <= meta["median_on_il_weekly"] <= hi
    assert meta["reconciled_against"], "the table was built without reconciling"


# ── serving contract ───────────────────────────────────────────────────────

def test_swapped_columns_are_in_the_moneyline_width():
    assert all(c in training.MONEYLINE_FEATURE_COLS for c in SWAPPED)


def test_run_line_resolves_the_same_feature_list():
    """The run line must receive exactly the moneyline list -- that is what
    makes this feature a change to BOTH models rather than one.

    Exercised through the real ``build_side_frame`` in strict-parity mode
    (the mode run_engine.py:2474 drives for both Poisson regressors), not by
    grepping the source: a future edit that drops the parity call would make
    the widths diverge silently, and this fails instead.
    """
    from feature_selection import apply_adopted_subset

    apply_adopted_subset()
    cols = training.active_moneyline_feature_cols()
    frame = pd.DataFrame([{c: 0.0 for c in cols}])
    used = {}
    for side in ("home", "away"):
        built, used_side = run_engine.build_side_frame(
            frame, side, strict_feature_parity=True)
        used[side] = used_side
        assert all(c in used_side for c in SWAPPED), (
            f"{side} run regressor is missing a swapped column")
    assert used["home"] == used["away"], (
        "the home and away run regressors must receive the same list")
    assert set(cols) <= set(used["home"]), (
        "the run line dropped a moneyline feature")


def test_il_table_is_exempt_from_dated_cleanup():
    """A dateless name with no exemption is classified stale and git rm'd on
    the next daily run -- which does not fail loudly, it silently reverts
    every expected-lineup feature to the unfiltered pool."""
    for name in (features.IL_STINTS_FILE, features.IL_STINTS_META_FILE):
        assert retention_policy.is_never_delete(
            f"mlb-backend/data_delivery/{name}"), name
        assert retention_policy.artifact_date(
            f"mlb-backend/data_delivery/{name}") is None, name
        assert retention_policy.classify_artifact(
            f"mlb-backend/data_delivery/{name}", set(), set(), set(), set()
        ) == "protected", name
