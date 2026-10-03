"""Tests for the NHL player-level EVO/PPO rating engine.

Mirrors ``mlb-backend/backend/test_lineup_il.py``'s approach: drive the shipped
functions with small synthetic frames whose correct answer is computable by
hand, so a regression shows up as a wrong NUMBER rather than a raised error.

The contract under test (production v1.1, game-grain):

* ``_rolling_player_window`` is a trailing INCLUSIVE sum per (player,
  situation): the current source game's own stats are part of its rating. PIT
  is NOT enforced here — it is enforced once, at the pool join, by requiring a
  candidate's source game date to be strictly earlier than the target date
  (``injury_stints._expand_to_games``).
* Rates are xG per SECOND inside ``shrink_rate``; per-60 values multiply by
  ``SECONDS_PER_HOUR``. MoneyPuck icetime is seconds, not minutes.
* Missing history stays UNKNOWN: no fallback k constant, no fallback league
  rate. A cell without strictly prior evidence is NaN or absent.
* Injuries never alter a rating row. ``build_player_ratings`` accepts an
  injuries frame for compatibility and reports ``no_pool_exclusion_only``;
  the binary exclusion happens per target game in
  ``injury_stints.team_game_rates``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import player_ratings as pr  # noqa: E402

EVO = pr.SITUATION_EVO
PPO = pr.SITUATION_PPO


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_games(rows):
    """rows: (player_id, date, situation, position, xg, ice_seconds)."""
    return pd.DataFrame([
        {
            "player_id": str(p), "player_name": f"P{p}", "team": "T",
            "game_date": pd.Timestamp(dt), "game_id": f"G{p}_{dt}_{s}",
            "position": pos, "situation": s, "season": pd.Timestamp(dt).year,
            "xg": float(xg), "ice_seconds": float(ice),
        }
        for p, dt, s, pos, xg, ice in rows
    ])


@pytest.fixture
def two_season_two_situation():
    """Two players, two seasons, both situations. Hand-computable.

    Ice time is set so each situation's 2023 season total is EXACTLY 10,000s
    across the two players, making the prior-season k table clean:
    k = 0.20 * (mean player-season ice) with one player-season per situation.
    """
    rows = []
    for player, mult in (("1", 1.0), ("2", 3.0)):
        for year in (2023, 2024):
            for sit, ice in ((EVO, 8_000.0), (PPO, 2_000.0)):
                rows.append((player, f"{year}-04-20", sit, "C",
                             100.0 * mult * (1.0 if sit == EVO else 2.0), ice))
    return make_games(rows)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
class TestNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("C", "C"), ("c", "C"), ("Center", "C"), ("centre", "C"),
        ("L", "L"), ("Left", "L"), ("R", "R"), ("D", "D"),
        ("Defense", "D"), ("Defenceman", "D"),
    ])
    def test_known_positions(self, raw, expected):
        assert pr.normalize_position(raw) == expected

    @pytest.mark.parametrize("raw", ["Team Level", "", None, "G", "F", "unknown"])
    def test_unknown_positions_return_none_not_a_guess(self, raw):
        # Guessing would hand an unmapped code the defenceman prior, the lowest
        # of the four, silently shrinking it hardest toward the worst mean.
        assert pr.normalize_position(raw) is None

    def test_team_level_rows_are_dropped_and_counted(self):
        games = make_games([("1", "2024-04-20", EVO, "C", 10.0, 1000.0)])
        games.loc[len(games)] = {
            "player_id": "T", "player_name": "NYR", "team": "NYR",
            "game_date": pd.Timestamp("2024-04-20"), "game_id": "G_T",
            "position": "Team Level", "situation": EVO, "season": 2024,
            "xg": 2.5, "ice_seconds": 3600.0,
        }
        prepared, audit = pr.prepare_player_games(games, id_col="player_id")
        assert audit["dropped_unknown_position"] == 1
        assert len(prepared) == 1
        assert prepared["position"].tolist() == ["C"]

    @pytest.mark.parametrize("date,expected", [
        ("2024-01-15", 2023),   # January belongs to the 2023-24 season
        ("2024-06-30", 2023),   # June still the 2023-24 season (playoffs)
        ("2024-07-01", 2024),   # July starts the 2024-25 label year
        ("2024-10-01", 2024),
    ])
    def test_season_ids_follow_nhl_season_convention(self, date, expected):
        got = pr._season_ids(pd.Series([pd.Timestamp(date)]))
        assert int(got.iloc[0]) == expected

    def test_season_is_always_derived_from_dates_not_source_label(self):
        """A mismatched source season label must not survive preparation.

        MoneyPuck labels seasons by start year, but accepting the label would
        silently move a game into the wrong prior-season k table.
        """
        games = make_games([("1", "2024-01-15", EVO, "C", 10.0, 1000.0)])
        games["season"] = 1999  # wrong on purpose
        prepared, _ = pr.prepare_player_games(games, id_col="player_id")
        assert int(prepared["season"].iloc[0]) == 2023

    def test_zero_ice_rows_are_kept(self):
        """A skater can dress without any 5on4 time; the appearance still
        occupies one of the trailing played-game slots for that situation."""
        games = make_games([("1", "2024-04-20", PPO, "C", 0.0, 0.0)])
        prepared, audit = pr.prepare_player_games(games, id_col="player_id")
        assert audit["dropped_bad_values"] == 0
        assert len(prepared) == 1

    def test_negative_or_missing_values_are_dropped_and_counted(self):
        games = make_games([
            ("1", "2024-04-20", EVO, "C", 10.0, 1000.0),
            ("2", "2024-04-20", EVO, "C", 10.0, -1.0),      # negative ice
            ("3", "2024-04-20", EVO, "C", float("nan"), 1000.0),
            ("4", "2024-04-20", EVO, "C", 10.0, float("nan")),
        ])
        _, audit = pr.prepare_player_games(games, id_col="player_id")
        assert audit["dropped_bad_values"] == 3
        assert audit["rows_out"] == 1


# ---------------------------------------------------------------------------
# The trailing inclusive window — game-grain v1.1
# ---------------------------------------------------------------------------
class TestRollingWindow:
    def test_window_is_inclusive_of_the_current_source_game(
            self, two_season_two_situation):
        """The rating row is a POST-GAME candidate rating.

        Its own game's stats are part of the sum. PIT is enforced later by the
        pool join (source date strictly before target date), not by shifting
        here — a shift would drop the latest completed game from every target.
        """
        prepared, _ = pr.prepare_player_games(
            two_season_two_situation, id_col="player_id")
        rolled = pr._rolling_player_window(prepared, window=1)
        first = rolled[rolled["game_date"] == pd.Timestamp("2023-04-20")]
        # window=1: every row's sum is exactly its own game.
        assert (first["prior_xg"] == first["xg"]).all()
        assert (first["prior_ice_seconds"] == first["ice_seconds"]).all()
        assert (first["prior_rows"] == 1).all()

    def test_window_never_crosses_situations(self, two_season_two_situation):
        """REGRESSION (adapted): rolling must group by (player, situation).

        At game grain each player has one row per situation per date, so a
        player-only group would hand a player's even-strength rating their own
        power-play minutes. Detected here by the magnitudes being wildly
        different per situation.
        """
        prepared, _ = pr.prepare_player_games(
            two_season_two_situation, id_col="player_id")
        rolled = pr._rolling_player_window(prepared, window=1)
        last = rolled[rolled["game_date"] == pd.Timestamp("2024-04-20")]
        evo = last[last["situation"] == EVO].set_index("player_id")["prior_xg"]
        ppo = last[last["situation"] == PPO].set_index("player_id")["prior_xg"]
        # window=1 -> own game only. Player 1 -> 100/200, player 2 -> 3x that,
        # WITHIN each situation. A swap would put 200/600 on the EVO rows.
        assert evo["1"] == 100.0 and evo["2"] == 300.0
        assert ppo["1"] == 200.0 and ppo["2"] == 600.0
        assert evo["1"] != ppo["1"]

    def test_window_spans_multiple_rows_inclusive_and_caps(self):
        rows = [("1", f"2024-04-{day:02d}", EVO, "C", 10.0, 1000.0)
                for day in (10, 11, 12, 13, 14)]
        prepared, _ = pr.prepare_player_games(make_games(rows), id_col="player_id")
        rolled = pr._rolling_player_window(prepared, window=2)
        by_day = rolled.set_index("game_date")
        # Inclusive trailing window of 2 played games.
        assert by_day.loc[pd.Timestamp("2024-04-10"), "prior_xg"] == 10.0  # self
        assert by_day.loc[pd.Timestamp("2024-04-11"), "prior_xg"] == 20.0  # 2 rows
        assert by_day.loc[pd.Timestamp("2024-04-14"), "prior_xg"] == 20.0  # caps
        assert by_day.loc[pd.Timestamp("2024-04-10"), "prior_rows"] == 1
        assert by_day.loc[pd.Timestamp("2024-04-14"), "prior_rows"] == 2

    def test_window_alignment_survives_interleaved_groups(self):
        """REGRESSION (adapted): grouped .rolling() returns GROUP order.

        With one group key a player's rows are contiguous in both orders and
        the bug hides. With two keys they interleave, so a positional reset
        would write each player's sums onto the wrong rows.
        """
        rows = []
        for player, mult in (("1", 1.0), ("2", 3.0)):
            for day in (10, 11, 12):
                rows.append((player, f"2024-04-{day:02d}", EVO, "C",
                             10.0 * mult, 1_000.0))
        prepared, _ = pr.prepare_player_games(make_games(rows), id_col="player_id")
        rolled = pr._rolling_player_window(prepared, window=3)
        last = rolled[rolled["game_date"] == pd.Timestamp("2024-04-12")]
        got = last.set_index("player_id")["prior_xg"]
        assert got["1"] == 30.0
        assert got["2"] == 90.0

    def test_pool_date_is_the_source_game_date(self):
        rows = [("1", "2024-04-10", EVO, "C", 10.0, 1000.0),
                ("1", "2024-04-11", EVO, "C", 10.0, 1000.0)]
        prepared, _ = pr.prepare_player_games(make_games(rows), id_col="player_id")
        rolled = pr._rolling_player_window(prepared, window=2)
        # No synthesized availability date: the strict source-date gate at the
        # pool join needs the real game date.
        assert (rolled["pool_date"] == rolled["game_date"]).all()

    @pytest.mark.parametrize("window", [0, -1])
    def test_window_below_one_raises(self, window):
        prepared, _ = pr.prepare_player_games(
            make_games([("1", "2024-04-10", EVO, "C", 10.0, 1000.0)]),
            id_col="player_id")
        with pytest.raises(ValueError):
            pr._rolling_player_window(prepared, window=window)


# ---------------------------------------------------------------------------
# k: the 20%-of-a-prior-season rule, strictly prior to as_of
# ---------------------------------------------------------------------------
class TestShrinkStrength:
    def test_k_is_twenty_percent_of_a_prior_player_season(
            self, two_season_two_situation):
        prepared, _ = pr.prepare_player_games(
            two_season_two_situation, id_col="player_id")
        table = pr.season_ice_time_table(prepared, as_of="2024-10-01")
        # k is per (position, SITUATION) cell, and only COMPLETED PRIOR seasons
        # (source season < as_of season) contribute: the 2023 season alone.
        assert table[(EVO, "C")] == pytest.approx(0.20 * 8_000.0)
        assert table[(PPO, "C")] == pytest.approx(0.20 * 2_000.0)

    def test_the_as_of_season_itself_is_excluded_in_full(
            self, two_season_two_situation):
        """A season cannot tune its own shrinkage strength."""
        prepared, _ = pr.prepare_player_games(
            two_season_two_situation, id_col="player_id")
        # as_of during the 2023 season: 2023 is the target season -> excluded,
        # and no earlier season exists -> no evidence at all.
        assert pr.season_ice_time_table(prepared, as_of="2023-04-21") == {}
        # as_of during the 2024 season: only 2023 contributes.
        table = pr.season_ice_time_table(prepared, as_of="2024-04-21")
        assert table[(EVO, "C")] == pytest.approx(0.20 * 8_000.0)

    def test_k_is_averaged_per_player_season_not_per_player(self):
        """A 3-game cameo must not count as a full season.

        Averaging per player (total / n_seasons, then mean across players)
        under-shrank every rating on real data; the unit is a player-SEASON.
        """
        rows = [("1", "2024-04-20", EVO, "C", 1.0, 8_000.0)]
        # Nine players with a single 100s season, plus one with 8,000s.
        for i in range(9):
            rows.append((str(100 + i), "2024-04-20", EVO, "C", 0.1, 100.0))
        prepared, _ = pr.prepare_player_games(make_games(rows), id_col="player_id")
        table = pr.season_ice_time_table(prepared, as_of="2025-10-01")
        expected = 0.20 * (8_000.0 + 9 * 100.0) / 10
        assert table[(EVO, "C")] == pytest.approx(expected)

    def test_participation_floor_excludes_short_seasons(self):
        rows = [("1", "2024-04-20", EVO, "C", 1.0, 8_000.0)]
        for i in range(9):
            rows.append((str(100 + i), "2024-04-20", EVO, "C", 0.1, 100.0))
        prepared, _ = pr.prepare_player_games(make_games(rows), id_col="player_id")
        table = pr.season_ice_time_table(
            prepared, as_of="2025-10-01", participation_floor_seconds=5_000.0)
        assert table[(EVO, "C")] == pytest.approx(0.20 * 8_000.0)

    def test_cells_without_prior_evidence_are_omitted_not_filled(self):
        """No fallback constant exists any more.

        A cell with no strictly prior completed-season evidence must be ABSENT
        (the engine then leaves k and the shrunk rate NaN) rather than filled
        from a full-history constant that would leak future data backwards.
        """
        prepared, _ = pr.prepare_player_games(
            make_games([("1", "2024-04-20", EVO, "C", 1.0, 8_000.0)]),
            id_col="player_id")
        # as_of inside the same NHL season (2023-24): the season containing
        # as_of is excluded in full -> no evidence at all yet.
        assert pr.season_ice_time_table(prepared, as_of="2024-04-21") == {}
        # A cell that never occurs is simply absent.
        table = pr.season_ice_time_table(prepared, as_of="2024-10-01")
        assert (EVO, "C") in table
        assert (PPO, "C") not in table
        assert (EVO, "D") not in table

    def test_invalid_as_of_raises(self):
        prepared, _ = pr.prepare_player_games(
            make_games([("1", "2024-04-20", EVO, "C", 1.0, 8_000.0)]),
            id_col="player_id")
        with pytest.raises(ValueError):
            pr.season_ice_time_table(prepared, as_of="not-a-date")

    def test_empty_input_returns_an_empty_table(self):
        assert pr.season_ice_time_table(pd.DataFrame(), as_of="2025-10-01") == {}


# ---------------------------------------------------------------------------
# The shrinkage formula
# ---------------------------------------------------------------------------
class TestShrinkRate:
    def test_returns_a_per_second_rate(self):
        # xg=0, ice=0, k=2000 -> exactly the league rate, in xG per SECOND.
        got = pr.shrink_rate(0.0, 0.0, 0.5, 2_000.0)
        assert got == pytest.approx(0.5 / pr.SECONDS_PER_HOUR)

    def test_per60_conversion_happens_in_the_caller(self):
        """build_player_ratings multiplies by 3600; shrink_rate must not."""
        per_second = pr.shrink_rate(0.0, 0.0, 0.5, 2_000.0)
        assert per_second * pr.SECONDS_PER_HOUR == pytest.approx(0.5)

    def test_a_full_season_of_evidence_dominates(self):
        # A player far above the prior stays far above it.
        assert (pr.shrink_rate(100.0, 1_000.0, 0.5, 2_000.0)
                > 0.5 / pr.SECONDS_PER_HOUR)

    def test_bayesian_arm_weight_is_k_over_ice_plus_k(self):
        ice, xg, mu60, k = 4_000.0, 40.0, 0.5, 2_000.0
        got = pr.shrink_rate(xg, ice, mu60, k, arm="bayesian")
        expected = (xg + (mu60 / pr.SECONDS_PER_HOUR) * k) / (ice + k)
        assert got == pytest.approx(expected)
        # ...and that is 1/3 prior weight, 2/3 evidence.
        assert (k / (ice + k)) == pytest.approx(1 / 3)

    def test_ramp_arm_is_raw_at_or_above_k(self):
        """The ramp arm (gated 2026-10-02, not adopted): at ice >= k the
        prior has ZERO pull — the rating is the player's own rate."""
        ice, xg, mu60, k = 4_000.0, 40.0, 0.5, 2_000.0
        assert ice >= k
        got = pr.shrink_rate(xg, ice, mu60, k, arm="ramp")
        assert got == pytest.approx(xg / ice)
        # Below k the own weight is exactly ice/k (here 1,000/2,000 = 50%).
        below = pr.shrink_rate(20.0, 1_000.0, mu60, k, arm="ramp")
        mu = mu60 / pr.SECONDS_PER_HOUR
        w = (below - mu) / (20.0 / 1_000.0 - mu)
        assert w == pytest.approx(0.5)

    def test_shrinks_toward_the_prior_from_both_directions(self):
        """The invariant is BETWEEN-ness: raw -> prior, never past it."""
        mu = 0.5 / pr.SECONDS_PER_HOUR
        for xg in (1.0, 40.0, 100.0):
            ice = 1_000.0
            raw = xg / ice
            shrunk = pr.shrink_rate(xg, ice, 0.5, 2_000.0)
            assert min(raw, mu) <= shrunk <= max(raw, mu)
            # And it moves toward the prior, not away from it.
            assert abs(shrunk - mu) <= abs(raw - mu)

    @pytest.mark.parametrize("args", [
        (float("nan"), 100.0, 0.5, 2_000.0),
        (1.0, float("nan"), 0.5, 2_000.0),
        (1.0, 100.0, float("nan"), 2_000.0),
        (1.0, 100.0, 0.5, float("nan")),
    ])
    def test_non_finite_inputs_are_nan_not_an_exception(self, args):
        assert np.isnan(pr.shrink_rate(*args))

    def test_zero_denominator_is_nan(self):
        assert np.isnan(pr.shrink_rate(0.0, 0.0, 0.5, 0.0))


# ---------------------------------------------------------------------------
# League prior: strictly prior calendar dates, no fallback constants
# ---------------------------------------------------------------------------
class TestLeaguePrior:
    def test_prior_is_position_segmented(self):
        """The whole reason this module exists: one mean is not enough.

        A defenceman generates ~25% of a winger's even-strength rate, so a
        league-wide prior would over-rate every defenceman by ~2.7x. Two dates
        are needed so the second has real prior evidence rather than NaN.
        """
        rows = []
        for pos, mult in (("C", 3.0), ("D", 0.5)):
            for i in range(4):
                # Each player appears on BOTH dates, so the second date has
                # real prior evidence. Ids are distinct per position (a reused
                # id would be collapsed by the per-player dedup and silently
                # drop half the rows).
                for day in ("04-20", "04-27"):
                    rows.append((f"{pos}{i}", f"2024-{day}", EVO, pos,
                                 30.0 * mult, 1_000.0))
        prepared, _ = pr.prepare_player_games(make_games(rows), id_col="player_id")
        lp = pr.league_prior_table(prepared)
        rates = (lp[lp["game_date"] == pd.Timestamp("2024-04-27")]
                 .set_index("position")["league_rate_per60"])
        # C: 90 xG / 1000s = 324 per 60; D: 15 / 1000s = 54 per 60.
        assert rates["C"] / rates["D"] == pytest.approx(6.0, rel=1e-6)

    def test_prior_excludes_the_entire_current_date(self):
        """A date's own games cannot enter its own prior.

        Calendar-day grain cannot prove within-day event/publish ordering, so
        the whole source date is subtracted from the cumulative sum.
        """
        rows = [("1", "2024-04-20", EVO, "C", 10.0, 1_000.0),
                ("2", "2024-04-20", EVO, "C", 10.0, 1_000.0),
                ("1", "2024-04-21", EVO, "C", 10.0, 1_000.0),
                ("2", "2024-04-21", EVO, "C", 10.0, 1_000.0)]
        prepared, _ = pr.prepare_player_games(make_games(rows), id_col="player_id")
        lp = pr.league_prior_table(prepared)
        got = lp.set_index("game_date")["league_rate_per60"]
        # 04-20: nothing earlier -> unknown, not a fallback constant.
        assert pd.isna(got[pd.Timestamp("2024-04-20")])
        # 04-21: only 04-20's 20 xG / 2000s -> 36.0 per 60.
        assert got[pd.Timestamp("2024-04-21")] == pytest.approx(36.0)

    def test_first_date_is_unknown_not_a_fallback_constant(
            self, two_season_two_situation):
        prepared, _ = pr.prepare_player_games(
            two_season_two_situation, id_col="player_id")
        lp = pr.league_prior_table(prepared)
        first = lp[lp["game_date"] == pd.Timestamp("2023-04-20")]
        assert first["league_rate_per60"].isna().all()
        second = lp[lp["game_date"] == pd.Timestamp("2024-04-20")]
        assert second["league_rate_per60"].notna().all()

    def test_empty_input_returns_an_empty_table(self):
        assert len(pr.league_prior_table(pd.DataFrame())) == 0


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
class TestBuildPlayerRatings:
    def test_produces_one_row_per_player_situation_date(
            self, two_season_two_situation):
        ratings, audit = pr.build_player_ratings(
            two_season_two_situation, id_col="player_id", window=1)
        assert audit["players"] == 2
        assert len(ratings) == 2 * 2 * 2      # players x situations x seasons
        assert set(ratings["metric"]) == {"EVO", "PPO"}

    def test_output_columns_are_present(self, two_season_two_situation):
        ratings, _ = pr.build_player_ratings(
            two_season_two_situation, id_col="player_id")
        for col in pr.OUTPUT_COLUMNS:
            assert col in ratings.columns, col

    def test_first_season_is_unknown_not_fallback_filled(
            self, two_season_two_situation):
        """The v1.1 contract: missing history stays NaN.

        The 2023 rows have no prior season for k and no earlier date for the
        league rate, so their shrunk rate is genuinely unknown. The 2024 rows
        have both (2023 season for k, 2023-04-20 for the league rate).
        """
        ratings, _ = pr.build_player_ratings(
            two_season_two_situation, id_col="player_id")
        first = ratings[ratings["game_date"] == pd.Timestamp("2023-04-20")]
        assert first["shrunk_rate_per60"].isna().all()
        second = ratings[ratings["game_date"] == pd.Timestamp("2024-04-20")]
        assert second["shrunk_rate_per60"].notna().all()
        # The 2024 rating is a finite per-60 number (the fixture's inflated xG
        # values put it in the tens-to-hundreds; only finiteness is pinned).
        assert np.isfinite(second["shrunk_rate_per60"].astype(float)).all()
        assert (second["shrunk_rate_per60"] > 0).all()

    def test_pool_date_equals_source_game_date(self, two_season_two_situation):
        ratings, _ = pr.build_player_ratings(
            two_season_two_situation, id_col="player_id")
        assert (ratings["pool_date"] == ratings["game_date"]).all()

    def test_no_injuries_means_final_equals_shrunk(self, two_season_two_situation):
        ratings, audit = pr.build_player_ratings(
            two_season_two_situation, id_col="player_id")
        assert audit["injuries_applied"] == "no"
        # equal_nan: the first season is unknown, so both columns are NaN and
        # NaN != NaN would fail a plain equality check for the right reason.
        assert np.allclose(ratings["final_rate_per60"].astype(float),
                           ratings["shrunk_rate_per60"].astype(float),
                           equal_nan=True)
        assert (ratings["injury_multiplier"] == 1.0).all()
        assert (ratings["availability_status"] == "healthy").all()

    def test_injuries_never_alter_a_rating_row(self, two_season_two_situation):
        """v1.1: injuries are accepted but never applied here.

        The binary exclusion lives in injury_stints.team_game_rates, once per
        TARGET game. Prior-game ratings are never touched — an injured player
        keeps the rolling rating earned by the games he played.
        """
        injuries = pd.DataFrame([
            {"player_id": "1", "status": "Out", "report_date": "2024-04-19"},
        ])
        ratings, audit = pr.build_player_ratings(
            two_season_two_situation, injuries=injuries, id_col="player_id")
        assert audit["injuries_rows"] == 1
        assert audit["injuries_applied"] == "no_pool_exclusion_only"
        assert (ratings["injury_multiplier"] == 1.0).all()
        assert (ratings["availability_status"] == "healthy").all()
        assert np.allclose(ratings["final_rate_per60"].astype(float),
                           ratings["shrunk_rate_per60"].astype(float),
                           equal_nan=True)

    def test_injuries_are_not_applied_even_when_after_the_report_date(
            self, two_season_two_situation):
        injuries = pd.DataFrame([
            {"player_id": "1", "status": "Out", "report_date": "2023-04-19"},
        ])
        ratings, audit = pr.build_player_ratings(
            two_season_two_situation, injuries=injuries, id_col="player_id")
        # Player 1's 2023 and 2024 rows would both be post-report; nothing may
        # change regardless — exclusion is the pool builder's job.
        assert audit["injuries_applied"] == "no_pool_exclusion_only"
        assert (ratings["injury_multiplier"] == 1.0).all()

    def test_tz_aware_injury_dates_are_accepted(self, two_season_two_situation):
        """ESPN ships report_date as ``2026-09-22T20:08Z``; ratings are naive.

        Mixing the two used to raise outright, and only on real data.
        """
        injuries = pd.DataFrame([
            {"player_id": "1", "status": "Doubtful",
             "report_date": "2024-04-19T20:08Z"},
        ])
        ratings, audit = pr.build_player_ratings(
            two_season_two_situation, injuries=injuries, id_col="player_id")
        assert audit["injuries_applied"] == "no_pool_exclusion_only"
        assert len(ratings) == 8

    def test_unmatched_and_keyless_injuries_also_change_nothing(
            self, two_season_two_situation):
        unmatched = pd.DataFrame([
            {"player_id": "999", "status": "Out", "report_date": "2024-04-19"},
        ])
        ratings, audit = pr.build_player_ratings(
            two_season_two_situation, injuries=unmatched, id_col="player_id")
        assert audit["injuries_applied"] == "no_pool_exclusion_only"
        assert (ratings["injury_multiplier"] == 1.0).all()

        keyless = pd.DataFrame([{"status": "Out"}])
        ratings, audit = pr.build_player_ratings(
            two_season_two_situation, injuries=keyless, id_col="player_id")
        assert audit["injuries_applied"] == "no_pool_exclusion_only"
        assert (ratings["injury_multiplier"] == 1.0).all()

    def test_as_of_drops_the_current_and_later_rows(self, two_season_two_situation):
        ratings, audit = pr.build_player_ratings(
            two_season_two_situation, id_col="player_id",
            as_of="2024-04-20")
        assert ratings["game_date"].max() < pd.Timestamp("2024-04-20")
        assert audit["rows_after_as_of"] < audit["rows_out"]

    def test_invalid_as_of_raises(self, two_season_two_situation):
        with pytest.raises(ValueError):
            pr.build_player_ratings(
                two_season_two_situation, id_col="player_id", as_of="garbage")

    def test_empty_input_returns_empty_output_and_audit(self):
        ratings, audit = pr.build_player_ratings(pd.DataFrame(), id_col="player_id")
        assert len(ratings) == 0
        assert audit["rows"] == 0
        assert audit["players"] == 0

    def test_audit_accounts_for_every_input_row(self):
        games = make_games([("1", "2024-04-20", EVO, "C", 10.0, 1_000.0)])
        games.loc[len(games)] = {
            "player_id": "2", "player_name": "P2", "team": "T",
            "game_date": pd.Timestamp("2024-04-20"), "game_id": "G2",
            "position": "C", "situation": "4on5", "season": 2024,
            "xg": 1.0, "ice_seconds": 500.0,
        }
        _, audit = pr.build_player_ratings(games, id_col="player_id")
        assert audit["rows_in"] == 2
        assert (audit["dropped_bad_situation"]
                + audit["dropped_unknown_position"]
                + audit["dropped_no_player_id"]
                + audit["dropped_bad_values"]
                + audit["dropped_duplicate"]) == audit["rows_in"] - audit["rows_out"]

    def test_duplicate_rows_are_dropped_and_counted(self):
        """A dedup that is not counted is a silent row loss."""
        games = make_games([("1", "2024-04-20", EVO, "C", 10.0, 1_000.0)])
        dup = games.iloc[[0]].copy()
        dup["xg"] = 20.0
        both = pd.concat([games, dup], ignore_index=True)
        prepared, audit = pr.prepare_player_games(both, id_col="player_id")
        assert audit["rows_in"] == 2
        assert audit["dropped_duplicate"] == 1
        assert audit["rows_out"] == 1
        # keep="last" wins.
        assert prepared["xg"].iloc[0] == 20.0


# ---------------------------------------------------------------------------
# Serving shape
# ---------------------------------------------------------------------------
class TestLatestRatings:
    def test_one_row_per_player_situation(self, two_season_two_situation):
        ratings, _ = pr.build_player_ratings(two_season_two_situation,
                                             id_col="player_id")
        latest = pr.latest_ratings(ratings)
        assert len(latest) == 2 * 2          # players x situations
        assert latest["game_date"].nunique() == 1
        assert latest["game_date"].iloc[0] == pd.Timestamp("2024-04-20")

    def test_on_or_before_filter(self, two_season_two_situation):
        ratings, _ = pr.build_player_ratings(two_season_two_situation,
                                             id_col="player_id")
        latest = pr.latest_ratings(ratings, on_or_before="2024-04-19")
        assert (latest["game_date"] == pd.Timestamp("2023-04-20")).all()

    def test_empty_is_safe(self):
        assert len(pr.latest_ratings(pd.DataFrame())) == 0
