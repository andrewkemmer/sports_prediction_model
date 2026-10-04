"""Tests for the player-level Regularized Adjusted Plus-Minus rating.

The fit is linear algebra; what is worth testing is every place the rating
can be quietly wrong while still producing numbers: a sign convention that
prices one side of the game into the other, a one-sided game that feeds the
unadjusted fit, a prior that counts a DNP as a game played, a running total
that gets summed instead of read, a player id that joins in its float
spelling, and a season boundary that lets last season leak into this one's
rating. Each of those produced a plausible frame during development, which
is why each gets its own test.

The fixture convention: ``_row`` describes one player's box-score line;
end-to-end rating tests pair two teams per ``game_id`` so ``y`` (the home
margin) exists, because a game priced with only one side is skipped by
design. A 48-minute player carries ``share = 1.0``, so eff-games per game
is exactly 1 and the shrinkage arithmetic reads in game counts.

Everything here runs offline. The live probes that established what
stats.nba.com will and will not answer for a position are pinned in
``TestPositionSource``; reproducing their conclusions is what stops a
future adapter change from silently reintroducing a request the feed
rejects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
import nba_sources as src
import player_rapm as rapm


def _row(player, day, plus_minus, fga, fta=0, tov=0, minutes=48,
         position="G", season="2024-25", team=None, points=None,
         game_id=None):
    row = {
        "player_id": player, "gameday": day, "season": season,
        "plus_minus": plus_minus, "fga": fga, "fta": fta, "tov": tov,
        "minutes": minutes, "position": position,
        "team": player if team is None else team,
    }
    if points is not None:
        row["points"] = points
    if game_id is not None:
        row["game_id"] = game_id
    return row


def _frame(rows):
    return pd.DataFrame(rows)


def _prepared(rows):
    return rapm.prepare_player_games(_frame(rows))


class TestParticipationShare:
    """The design-matrix entry: MIN/48, clipped at one, zero for a DNP."""

    def test_a_full_game_is_one_and_an_ot_game_is_still_one(self):
        got = rapm.participation_share(pd.Series([48.0, 72.0, 0.0, -5.0]))
        assert list(got) == [1.0, 1.0, 0.0, 0.0]

    def test_minutes_scale_linearly_below_the_full_game(self):
        got = rapm.participation_share(pd.Series([24.0, 12.0]))
        assert got.iloc[0] == pytest.approx(0.5)
        assert got.iloc[1] == pytest.approx(0.25)

    def test_missing_minutes_reads_as_zero_participation_not_nan(self):
        """NaN minutes is a row that tells us nothing, not a mystery weight.

        A NaN would propagate into the design matrix and poison the whole
        game's equation; zero simply leaves the player out of it.
        """
        got = rapm.participation_share(pd.Series([np.nan]))
        assert float(got.iloc[0]) == 0.0

    def test_a_zero_minute_row_carries_no_share(self):
        """A DNP row is gated on MINUTES, not on the feed's PM column.

        The feed can carry a stale nonzero plus-minus on a DNP row (120
        such rows exist in the 2023-24..2025-26 logs); a zero-minute row
        must contribute nothing to either side of the rating, and it is
        not a game the player PLAYED either.
        """
        games = _prepared([
            _row("a", "2024-11-01", 10, 10, 0),
            _row("a", "2024-11-02", 999, 10, 0, minutes=0),
        ])
        dnp = games[games.gameday == "2024-11-02"].iloc[0]
        assert dnp.share == 0.0
        ratings = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2024-11-03"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        assert row.prior_games == 1
        assert row.prior_minutes == pytest.approx(48)
        assert row.prior_eff == pytest.approx(1.0)

    def test_teammate_shares_sum_to_the_games_share_of_minutes(self):
        """Two teammates splitting 48 minutes carry shares summing to 1.

        That identity is what makes each solved beta a per-game average in
        points: every row of the design weighs the game's margin exactly
        once, so no player's participation is counted twice or missed.
        """
        games = _prepared([
            _row("x", "2024-11-01", 10, 10, 0, team="T", minutes=36),
            _row("y", "2024-11-01", 20, 10, 0, team="T", minutes=12),
        ])
        assert games.share.sum() == pytest.approx(1.0)


def _two_team_game(day="2024-11-01", game_id="g1", home_pts=110.0,
                   away_pts=90.0):
    """One game, both sides, real box points for the fallback margin."""
    return _frame([
        _row("h1", day, 20, 10, 0, team="HOME", points=home_pts,
             game_id=game_id, minutes=48),
        _row("a1", day, 10, 10, 0, team="AWAY", points=away_pts,
             game_id=game_id, minutes=48),
    ])


class TestTeamGameEntries:
    """ONE row per game: both sides in the same equation, same y, opposite
    sign. The two-row-per-game design was built and REJECTED during
    development (it rated Huerter +10 and LeBron -2, no star in the top
    eight) - these tests pin the shape that replaced it.
    """

    def test_both_sides_carry_the_same_margin_with_opposite_signs(self):
        entries = rapm.team_game_entries(_two_team_game())
        assert len(entries) == 2
        assert set(entries.sign) == {1.0, -1.0}
        # y is one game-level quantity on both rows; only sign differs.
        assert entries.y.nunique() == 1
        # Box fallback: the lexicographically first team ("AWAY") stands in
        # as the nominal home side, so y is ITS margin: 90 - 110.
        assert float(entries.y.iloc[0]) == pytest.approx(-20.0)
        signs = dict(zip(entries.player_id, entries.sign))
        assert signs["a1"] == 1.0
        assert signs["h1"] == -1.0

    def test_the_official_team_table_supplies_true_sides(self):
        """With team_stats the sides come from is_home, not from spelling.

        The production path passes the official table, so home court is
        real - which is what makes the intercept's ~+1.7 reading mean
        something instead of "which team sorts first".
        """
        ts = pd.DataFrame({
            "gameday": ["2024-11-01", "2024-11-01"],
            "team": ["HOME", "AWAY"],
            "net_points": [20.0, -20.0],
            "is_home": [True, False],
        })
        entries = rapm.team_game_entries(_two_team_game(), team_stats=ts)
        signs = dict(zip(entries.player_id, entries.sign))
        ys = dict(zip(entries.player_id, entries.y))
        assert signs["h1"] == 1.0 and signs["a1"] == -1.0
        # Both rows read the HOME margin: +20 for home, -(-20) for away.
        assert ys["h1"] == pytest.approx(20.0)
        assert ys["a1"] == pytest.approx(20.0)

    def test_a_one_sided_game_yields_no_margin_and_no_fit(self):
        """A game priced with one team is the UNADJUSTED fit, so it is
        skipped rather than half-priced: the opponent would have nowhere
        to put his share and it would land on the margin as if one roster
        caused all of it."""
        entries = rapm.team_game_entries(_frame([
            _row("h1", "2024-11-01", 20, 10, 0, team="HOME", points=110,
                 game_id="g1"),
        ]))
        assert entries.y.isna().all()
        assert rapm.fit_rapm(entries) is None

    def test_dnps_do_not_enter_the_design(self):
        entries = rapm.team_game_entries(_frame([
            _row("h1", "2024-11-01", 20, 10, 0, team="HOME", points=110,
                 game_id="g1", minutes=48),
            _row("a1", "2024-11-01", 10, 10, 0, team="AWAY", points=90,
                 game_id="g1", minutes=48),
            _row("h6", "2024-11-01", 0, 0, 0, team="HOME", points=0,
                 game_id="g1", minutes=0),
        ]))
        assert "h6" not in set(entries.player_id)
        assert len(entries) == 2

    def test_the_game_key_accepts_the_feeds_id_spellings(self):
        """The raw feed spells its id nba_game_id; pairing must not care.

        Without an id the pair falls back to the gameday, which is only
        sound for a one-game day - so the id spellings are load-bearing
        for every multi-game night.
        """
        frame = _two_team_game().rename(columns={"game_id": "nba_game_id"})
        entries = rapm.team_game_entries(frame)
        assert (entries.game_key == "g1").all()


class TestSignedDesign:
    """The fit: signed shares, both sides one equation, unridged intercept."""

    @staticmethod
    def _home_win_games():
        """Home blows out with the star on the floor, loses when he sits.

        Shares must vary ACROSS GAMES: with constant minutes every beta
        column is a scalar copy of the intercept and the free intercept
        column explains the whole margin (every beta exactly zero, which
        is correct but tests nothing). Varying the split also gives the
        attribution something to price - the two teammates play the SAME
        total minutes in complementary games, so their difference is
        which minutes coincided with winning.
        """
        rows = []
        for index in range(6):
            heavy = index < 3
            day = f"2024-11-{index + 1:02d}"
            gid = f"g{index}"
            margin = 40.0 if heavy else -20.0
            star_min, scrub_min = (44.0, 4.0) if heavy else (4.0, 44.0)
            home_pts = 100.0 + margin / 2
            away_pts = 100.0 - margin / 2
            rows.append(_row("star", day, 0, 0, team="HOME",
                             minutes=star_min, points=home_pts, game_id=gid))
            rows.append(_row("scrub", day, 0, 0, team="HOME",
                             minutes=scrub_min, points=0.0, game_id=gid))
            rows.append(_row("avg1", day, 0, 0, team="AWAY", minutes=24,
                             points=away_pts / 2, game_id=gid))
            rows.append(_row("avg2", day, 0, 0, team="AWAY", minutes=24,
                             points=away_pts / 2, game_id=gid))
        return rapm.prepare_player_games(_frame(rows))

    def test_swapping_the_sides_flips_no_beta(self):
        """Sign symmetry: nothing about the betas depends on which side is
        called home. Flipping sign AND margin together leaves the outer
        product and x*y unchanged - only the intercept's READING moves,
        and it absorbs that by flipping with them."""
        entries = rapm.team_game_entries(self._home_win_games())
        beta, icpt = rapm.fit_rapm(entries)
        flipped = entries.copy()
        flipped["sign"] = -flipped.sign
        flipped["y"] = -flipped.y
        beta2, icpt2 = rapm.fit_rapm(flipped)
        assert np.allclose(beta.values, beta2.values, atol=1e-9)
        assert icpt2 == pytest.approx(-icpt)

    def test_the_winner_rates_positive_and_minutes_split_the_credit(self):
        """Attribution: credit follows the minutes that coincided with
        winning.

        The star plays 44 minutes in the three blowouts and 4 in the
        three losses; the scrub plays the mirror image. Their TOTAL
        minutes are equal, so only WHOSE minutes matched the results can
        separate them - and it does: the star above zero, the scrub below
        it. The two visitors play identical minutes in every game and
        rate at zero, which is what "no differential signal" looks like.
        """
        entries = rapm.team_game_entries(self._home_win_games())
        beta, _ = rapm.fit_rapm(entries)
        assert beta["star"] > 0
        assert beta["scrub"] < 0
        assert beta["star"] > beta["scrub"]
        assert abs(beta["avg1"]) < 0.05 and abs(beta["avg2"]) < 0.05

    def test_the_intercept_reads_the_mean_margin_unridged(self):
        """Home court is an extra column that is NEVER ridged: lambda sits
        on players only, so the intercept keeps the game's mean margin
        (-10 here: three +40 wins and three -20 losses average -10 in the
        box fallback's nominal-home reading) instead of being priced as if
        it were a player and dragged toward zero."""
        entries = rapm.team_game_entries(self._home_win_games())
        _, icpt = rapm.fit_rapm(entries)
        assert icpt == pytest.approx(-10.0)

    def test_the_raw_league_mean_is_exactly_zero(self):
        """The ridge's minimum-norm solution, centered on the players this
        fit evidenced - the property every downstream prior leans on."""
        entries = rapm.team_game_entries(self._home_win_games())
        beta, _ = rapm.fit_rapm(entries)
        assert float(beta.mean()) == pytest.approx(0.0, abs=1e-9)

    def test_a_fit_folds_in_each_game_exactly_once_strictly_prior(self):
        """Point-in-time at the solve: a game on the target day is not
        evidence for that day's rating. The day pointer only moves
        forward, so no game is ever folded in twice."""
        entries = rapm.team_game_entries(self._home_win_games())
        design = rapm._SeasonDesign(entries)
        design.advance(pd.Timestamp("2024-11-01"))
        assert design.games == 0        # the 11-01 game is NOT yet evidence
        design.advance(pd.Timestamp("2024-11-03"))
        assert design.games == 2        # 11-01 and 11-02
        design.advance(pd.Timestamp("2024-11-30"))
        assert design.games == 6        # all six, each exactly once
        assert design.solve() is not None


class TestPlayerIdSpelling:
    def test_a_float_id_loses_its_decimal_tail(self):
        """The season log's player_id is a float column.

        A bare ``astype(str)`` yields ``"1628983.0"`` while the position
        table is keyed on the integer id the feed publishes, so the two
        never join - and the frame still produces ratings, just with no
        position and therefore no prior. The failure is invisible in the
        output.
        """
        got = rapm._player_id_str(pd.Series([1628983.0, "2544", 203076.0]))
        assert list(got) == ["1628983", "2544", "203076"]

    def test_prepared_rows_carry_the_clean_id(self):
        out = rapm.prepare_player_games(
            _frame([_row(1628983.0, "2024-11-01", 20, 10, 4)]))
        assert list(out.player_id) == ["1628983"]


class TestRAPMShrinkage:
    def test_a_zero_evidence_player_lands_exactly_on_the_league_prior(self):
        """The whole point of the prior, stated as its limiting case."""
        got = rapm.shrunk_rapm(pd.Series([0.0]), pd.Series([0.0]),
                              pd.Series([5.5]), pd.Series([100.0]))
        assert float(got.iloc[0]) == pytest.approx(5.5)

    def test_evidence_moves_the_rating_off_the_prior(self):
        # (eff * raw + lg * k) / (eff + k) = (10*20 + 5.5*10) / 20
        got = rapm.shrunk_rapm(pd.Series([10.0]), pd.Series([20.0]),
                              pd.Series([5.5]), pd.Series([10.0]))
        assert float(got.iloc[0]) == pytest.approx(12.75)

    def test_a_player_without_a_fitted_beta_shrinks_all_the_way(self):
        """No beta in the window is no evidence, whatever ``prior_eff``
        says (rows whose games carried no margin): his rating IS the
        position prior rather than a number pulled toward it."""
        got = rapm.shrunk_rapm(pd.Series([4.0]), pd.Series([np.nan]),
                              pd.Series([5.5]), pd.Series([10.0]))
        assert float(got.iloc[0]) == pytest.approx(5.5)

    def test_an_empty_position_cell_stays_empty(self):
        """A NaN league cell keeps the rating NaN rather than manufacturing
        one from nothing."""
        got = rapm.shrunk_rapm(pd.Series([4.0]), pd.Series([20.0]),
                              pd.Series([np.nan]), pd.Series([10.0]))
        assert got.isna().all()

    def test_the_rating_stays_in_points_per_game_units(self):
        """Both sides of the mix are points per game, so no scale factor
        hides anywhere - k = 0 returns the player's own raw beta, and a
        6-point rating reads as 6, not 600."""
        got = float(rapm.shrunk_rapm(pd.Series([100.0]), pd.Series([6.0]),
                                    pd.Series([5.5]),
                                    pd.Series([0.0])).iloc[0])
        assert got == pytest.approx(6.0)
        assert 0 < got < 10

    def test_more_evidence_means_less_prior_weight(self):
        """Both players post a 30-point raw beta; the one with more
        evidence sits nearer it.

        The prior's job is to pull a THIN record toward the league, so as
        evidence accumulates the rating should approach the player's own
        beta - and since 30 is above the 5.5 prior, "closer to own beta"
        means "further from the league".
        """
        thin = float(rapm.shrunk_rapm(pd.Series([1.0]), pd.Series([30.0]),
                                     pd.Series([5.5]),
                                     pd.Series([10.0])).iloc[0])
        thick = float(rapm.shrunk_rapm(pd.Series([40.0]), pd.Series([30.0]),
                                      pd.Series([5.5]),
                                      pd.Series([10.0])).iloc[0])
        assert thin < thick
        assert abs(thin - 5.5) < abs(thick - 5.5)
        assert abs(thick - 30.0) < abs(thin - 30.0)


class TestSeasonEffTable:

    def test_k_is_twenty_percent_of_the_mean_player_season(self):
        """The MLB/NHL convention, carried across by its fraction.

        MLB's fixed 120-PA prior is 20% of a 600-PA season, so the
        fraction is the portable part and the season length is
        sport-specific. In eff-games a 48-minute game is exactly 1, so
        the mean player-season reads as a game count.
        """
        games = _prepared([
            _row("a", "2024-11-01", 20, 10, 4),
            _row("a", "2024-11-02", 20, 10, 4),
            _row("b", "2024-11-01", 20, 10, 4),
        ])
        table = rapm.season_eff_table(games, shrink_fraction=0.20)
        per_season = (games.assign(eff=games.share ** 2)
                      .groupby(["player_id", "season"]).eff.sum())
        # Two player-seasons, not three rows: "a"'s two November games are
        # one season of evidence, which is the whole point of the unit.
        assert len(per_season) == 2
        assert table["G"] == pytest.approx(0.20 * per_season.mean())

    def test_the_unit_is_a_player_season_not_a_player(self):
        """Averaging per player would weight a two-game cameo like a full season.

        This is the correction NHL documents at length: dividing by seasons
        first collapses the reference season, which would make k several
        times too small and under-shrink every rating. The two units are
        computed side by side here so the test says which one this
        implements rather than only what it does not.
        """
        rows = [_row("a", "2024-11-01", 20, 10, 4),
                _row("a", "2024-11-02", 20, 10, 4),
                _row("a", "2024-11-03", 20, 10, 4),
                # A distinct day: a player plays at most one game
                # a day, so the 2023-24 row needs its own gameday
                # to survive the player-game dedupe.
                _row("a", "2024-04-01", 20, 10, 4, season="2023-24"),
                _row("b", "2024-11-01", 20, 10, 4)]
        games = _prepared(rows)
        table = rapm.season_eff_table(games, shrink_fraction=0.20)

        # Per PLAYER-SEASON, eff-games: a contributes seasons of 3 and 1,
        # b one season of 1 - mean (3 + 1 + 1) / 3.
        per_player_season = (3 + 1 + 1) / 3
        assert table["G"] == pytest.approx(0.20 * per_player_season)
        # Per PLAYER, dividing each career by its season count first: a is
        # (3 + 1)/2 = 2 and b is 1, so the mean is 1.5 - a different
        # number.
        per_player = ((3 + 1) / 2 + 1) / 2
        assert per_player != pytest.approx(per_player_season)
        assert table["G"] != pytest.approx(0.20 * per_player)

    def test_through_season_excludes_the_season_in_progress(self):
        """NHL discipline: a season cannot tune its own shrinkage strength.

        An in-progress season contributes PARTIAL player-seasons to the
        whole-frame mean, which drags k down as the season accumulates -
        shrinkage silently weakens game by game. Deriving the mean from
        completed prior seasons only removes both the self-tuning and the
        drift; the partial season here (ten games of a full one) is the
        case that bites in production.
        """
        rows = []
        for index in range(82):
            day = (pd.Timestamp("2023-10-01")
                   + pd.Timedelta(days=index)).strftime("%Y-%m-%d")
            rows.append(_row("a", day, 20, 10, 4, season="2023-24"))
        for day in range(1, 11):
            rows.append(_row("b", f"2024-11-{day:02d}", 20, 10, 4,
                             season="2024-25"))
        games = _prepared(rows)
        # 48-minute games: eff-games per game is exactly 1, so the
        # player-seasons are 82 and 10 games of evidence.
        # Whole-frame mean: two player-seasons, one of them only ten games
        # long, so the in-progress season drags the mean down.
        whole = rapm.season_eff_table(games, shrink_fraction=0.20)
        assert whole["G"] == pytest.approx(0.20 * (82 + 10) / 2)
        # Through 2024-25: only the completed 2023-24 season enters, so k
        # is markedly stronger for it.
        pIT = rapm.season_eff_table(games, shrink_fraction=0.20,
                                     through_season="2024-25")
        assert pIT["G"] == pytest.approx(0.20 * 82)
        assert pIT["G"] > whole["G"]

    def test_an_empty_position_cell_keeps_a_usable_default(self):
        """A cell with no data must not raise at lookup time.

        If it raised, one unpopulated position would take the whole rating
        build down rather than degrading just that cell.
        """
        table = rapm.season_eff_table(
            _prepared([_row("a", "2024-11-01", 20, 10, 4)]))
        for position in config.PLAYER_EPM_POSITIONS:
            assert position in table
            assert table[position] > 0
        assert table["C"] == config.PLAYER_RAPM_FALLBACK_K_EFF

    def test_no_data_yields_a_complete_table_rather_than_nothing(self):
        for games in (None, pd.DataFrame()):
            table = rapm.season_eff_table(games)
            assert set(table) == set(config.PLAYER_EPM_POSITIONS)


class TestPositionAssignment:
    def test_a_two_position_player_lands_in_exactly_one_cell(self):
        """Cells are the denominators of the league prior.

        A player counted in two cells is counted twice in the league mean
        and the cells stop being comparable. 52 of 569 players in 2024-25
        are listed at both G and F, so this runs on a fifth of the league.
        """
        assigned = src.assign_positions(
            {"G": {"1", "2"}, "F": {"2", "3"}, "C": {"4"}})
        assert assigned == {"1": "G", "2": "G", "3": "F", "4": "C"}

    def test_every_listed_player_is_assigned_exactly_once(self):
        assigned = src.assign_positions({"G": {"1"}, "F": {"1", "2"},
                                         "C": {"2"}})
        assert sorted(assigned) == ["1", "2"]

    def test_a_position_the_feed_never_lists_is_simply_absent(self):
        frame = src.positions_frame({"G": {"1"}})
        assert list(frame.columns) == ["player_id", "position", "positions"]
        assert list(frame.player_id) == ["1"]
        assert list(frame.positions) == ["G"]

    def test_the_leagues_own_primary_overrides_the_convention(self):
        """``C-F`` is a centre who also plays forward, not a coin flip.

        The convention files him as F. Embiid, Towns, Holmgren, Hartenstein,
        Bitadze and Wendell Carter Jr. are all ``C-F`` in 2025-26 and all
        starters at centre - the same labelling artifact that left
        ``pl_rapm_c_*`` thin. The index is a READING and the convention is a
        convention, so the reading wins.
        """
        assigned = src.assign_positions(
            {"G": {"1"}, "F": {"2", "3"}, "C": {"2", "4"}},
            {"2": "C", "3": "F"})
        assert assigned == {"1": "G", "2": "C", "3": "F", "4": "C"}

    def test_the_primary_never_puts_a_player_in_two_cells(self):
        """The invariant that makes the override safe.

        The prior divides by these cells, so an override that added a cell
        instead of moving one would double-count the league mean. It only
        REWRITES, so every player is still in exactly one.
        """
        assigned = src.assign_positions(
            {"G": {"1", "2"}, "F": {"2"}, "C": {"2", "3"}}, {"2": "C"})
        assert assigned == {"1": "G", "2": "C", "3": "C"}
        assert sorted(assigned) == ["1", "2", "3"]

    def test_a_player_the_index_omits_keeps_the_convention(self):
        """A partial index degrades per player, not wholesale.

        The index is trustworthy for the live season only (23% coverage on a
        back season, measured), so a season ends up mixing two conventions by
        design - and a player the index says nothing about must still get a
        cell rather than falling out of the prior.
        """
        assigned = src.assign_positions(
            {"G": {"1", "2"}, "F": {"2", "3"}}, {"2": "F"})
        assert assigned == {"1": "G", "2": "F", "3": "F"}

    def test_an_index_naming_a_position_the_feed_did_not_list_is_ignored(self):
        """Trust the reading only where the two sources agree the player plays.

        The prior cell has to be a cell this season's filter pull actually
        produced, or ``LEAGUE_TABLE_GONE`` has a denominator nothing was
        counted against. The filter pull is what defines which cells exist;
        the index only chooses between them.
        """
        assigned = src.assign_positions(
            {"G": {"1"}, "F": {"2"}}, {"2": "C"})
        assert assigned == {"1": "G", "2": "F"}

    def test_the_index_reads_the_first_letter_of_a_compound_code(self):
        """``POSITION`` is ordered; the first letter is the league's primary.

        ``C-F`` and ``F-C`` are the same two letters in different orders and
        must NOT collapse to the same answer - that difference is the entire
        reason this source is being read at all.
        """
        payload = {"resultSets": [{"headers": ["PERSON_ID", "POSITION"],
                                   "rowSet": [[1, "C-F"], [2, "F-C"],
                                              [3, "G"], [4, "G-F"]]}]}
        assert src.primary_positions(payload) == {
            1: "C", 2: "F", 3: "G", 4: "G"}

    def test_a_junk_or_absent_position_reads_as_no_index_at_all(self):
        """Never a partial answer: half a primary is two conventions at once."""
        assert src.primary_positions(None) == {}
        assert src.primary_positions({}) == {}
        assert src.primary_positions({"resultSets": []}) == {}
        assert src.primary_positions({"resultSets": [{"headers": [],
                                                     "rowSet": []}]}) == {}
        # An unknown code has no cell in the prior table, so it is dropped
        # rather than inventing a fourth position.
        assert src.primary_positions(
            {"resultSets": [{"headers": ["PERSON_ID", "POSITION"],
                             "rowSet": [[1, "X"]]}]}) == {}

    def test_the_primary_moves_the_cell_but_not_the_listing(self):
        """``position`` follows the league; ``positions`` stays the full set.

        The segments ask "who does this club field who the league lists at
        centre", and that is indifferent to which of them the league calls
        primary. Rewriting the listing too would be a different, worse change.
        """
        frame = src.positions_frame(
            {"G": {"1"}, "F": {"2"}, "C": {"2"}}, {"2": "C"})
        assert dict(zip(frame.player_id, frame.position)) == {
            "1": "G", "2": "C"}
        assert dict(zip(frame.player_id, frame.positions)) == {
            "1": "G", "2": "F|C"}

    def test_no_positions_yields_a_well_formed_empty_frame(self):
        frame = src.positions_frame({})
        assert frame.empty
        assert list(frame.columns) == ["player_id", "position", "positions"]

    def test_the_full_listing_rides_beside_the_collapsed_cell(self):
        """One label for the prior, every label for the segments.

        The forward-centre ("2") is the whole point: the prior divides by
        exactly one cell so no player is counted twice, while the team
        segments need him in BOTH the F and C groups or a roster whose
        centers are all listed F-C ships pl_rapm_c as NaN - the 62%
        baseline coverage the monitor flags LOW_COVERAGE.
        """
        frame = src.positions_frame({"G": {"1"}, "F": {"1", "2"},
                                     "C": {"2", "3"}})
        assert dict(zip(frame.player_id, frame.position)) == {
            "1": "G", "2": "F", "3": "C"}
        assert dict(zip(frame.player_id, frame.positions)) == {
            "1": "G|F", "2": "F|C", "3": "C"}

    def test_the_listing_survives_prepare_and_build(self):
        """The segment reads ``positions`` off the RATING row, so both stages
        must carry it: prepare joins the labels per (player, season), build
        rebuilds the roster from raw columns. A drop anywhere upstream is
        invisible except as NaN ``pl_rapm_c`` at serve time, so the whole path
        is pinned together rather than each hop alone.
        """
        stats = _frame([
            _row("a", "2024-11-01", 20, 10, 4, position="F"),
            _row("b", "2024-11-01", 10, 10, 4, position="F"),
        ]).drop(columns=["position"])  # production player_stats has no label
        positions = src.positions_frame({"F": {"a", "b"}, "C": {"b"}})
        games = rapm.prepare_player_games(stats, positions)
        assert dict(zip(games.player_id, games.position)) == {"a": "F", "b": "F"}
        assert dict(zip(games.player_id, games.positions)) == {
            "a": "F", "b": "F|C"}

        ratings = rapm.build_player_rapm(games, target_dates=["2024-11-02"])
        listed = dict(zip(ratings.player_id, ratings.positions))
        assert listed["b"] == "F|C"
        # The collapsed cell the prior divided by never moved.
        assert dict(zip(ratings.player_id, ratings.position))["b"] == "F"


class TestPositionSource:
    def test_only_three_positions_are_asked_for(self):
        """Pinned because the five-way split is NOT available from this source.

        Measured against the live endpoint, stats.nba.com's
        ``PlayerPosition`` filter answers G, F and C and returns HTTP 400
        for PG/SG/SF/PF and for compound codes like ``G-F``. Adding a
        fourth code would change what the prior can mean, so it has to be
        a deliberate edit rather than a discovery at runtime.
        """
        assert config.PLAYER_EPM_POSITIONS == ("G", "F", "C")

    def test_the_query_carries_the_position_and_a_full_parameter_set(self):
        query = src.position_query("2024-25", "C")
        assert "PlayerPosition=C" in query
        assert "Season=2024-25" in query
        # An incomplete set answers 500, which would look like an upstream
        # outage rather than a missing argument.
        for required in ("LeagueID=00", "PerMode=PerGame",
                         "MeasureType=Base", "SeasonType=Regular+Season"):
            assert required in query

    def test_players_at_position_reads_the_id_column(self):
        payload = {"resultSets": [{"headers": ["PLAYER_ID", "PLAYER_NAME"],
                                   "rowSet": [[1, "A"], [2, "B"]]}]}
        assert src.players_at_position(payload) == {1, 2}

    def test_a_payload_without_the_column_yields_nothing_rather_than_raising(
            self):
        payload = {"resultSets": [{"headers": ["TEAM_ID"], "rowSet": [[1]]}]}
        assert src.players_at_position(payload) == set()
        assert src.players_at_position(None) == set()


class TestPositionJoin:
    """The position join is keyed on the season, and that is load-bearing.

    Joining on ``player_id`` alone matched one game against that player's
    row in every season file, duplicating 92% of the rating frame. The
    rating's RATE is a ratio and survived, which is exactly why it went
    unnoticed - while ``prior_plays``, ``prior_games``, the shrinkage
    weight and both pool gates were quietly inflated up to 3x.
    """

    @staticmethod
    def _stats(n_games=3):
        return pd.DataFrame({
            "player_id": [1628983.0] * n_games,
            "gameday": pd.to_datetime(
                ["2023-11-01", "2023-11-03", "2023-11-05"]),
            "plus_minus": [10.0, 15.0, 5.0],
            "fga": [15.0, 20.0, 8.0],
            "fta": [5.0, 6.0, 2.0],
            "tov": [1.0, 1.0, 1.0],
            "minutes": [32.0, 36.0, 28.0],
        })

    def test_a_player_in_three_seasons_yields_one_row_per_game(self):
        stats = self._stats()
        positions = pd.DataFrame({
            "player_id": ["1628983"] * 3,
            "position": ["G", "G", "F"],
            "season": ["2023-24", "2024-25", "2025-26"],
        })
        frame = rapm.prepare_player_games(stats, positions)
        assert len(frame) == 3
        assert frame.gameday.nunique() == 3

    def test_a_season_specific_position_is_matched_to_its_own_season(self):
        stats = self._stats()
        positions = pd.DataFrame({
            "player_id": ["1628983", "1628983"],
            "position": ["F", "G"],
            "season": ["2023-24", "2025-26"],
        })
        frame = rapm.prepare_player_games(stats, positions)
        # every game here is 2023-24, so the 2025-26 row must not apply
        assert set(frame.position) == {"F"}

    def test_one_row_per_player_per_game_holds(self):
        stats = self._stats()
        positions = pd.DataFrame({
            "player_id": ["1628983"] * 2,
            "position": ["G", "C"],
            "season": ["2023-24", "2023-24"],
        })
        frame = rapm.prepare_player_games(stats, positions)
        assert len(frame.drop_duplicates(["player_id", "gameday"])) \
            == len(frame)

    def test_a_player_season_listed_twice_keeps_one_cell(self):
        """Two positions in ONE season must not fan the player out either."""
        stats = self._stats()
        positions = pd.DataFrame({
            "player_id": ["1628983", "1628983"],
            "position": ["F", "G"],
            "season": ["2023-24", "2023-24"],
        })
        frame = rapm.prepare_player_games(stats, positions)
        assert len(frame) == 3
        assert frame.position.nunique() == 1

    def test_without_a_season_column_the_join_still_dedupes(self):
        stats = self._stats()
        positions = pd.DataFrame({
            "player_id": ["1628983", "1628983"],
            "position": ["G", "F"],
        })
        frame = rapm.prepare_player_games(stats, positions)
        assert len(frame) == 3


class TestAvailability:
    def test_status_words_map_to_availability(self):
        assert src.availability_status("Out") == "out"
        assert src.availability_status("Day-To-Day") == "day_to_day"
        assert src.availability_status("Active") == "healthy"
        assert src.availability_status(None) == "healthy"

    def test_the_official_designations_survive_normalisation(self):
        """``Questionable`` used to normalise to ``out`` on a substring match.

        That was right for ESPN and wrong for the official filing: over
        2025-26 a player designated Questionable before tipoff played 34
        times in 52. Folding it into ``out`` deletes a third of a real
        player's appearances and files the result as an absence.
        """
        for designation in ("Out", "Doubtful", "Recovery", "Questionable",
                            "Available", "Probable"):
            assert src.availability_status(designation) == designation.lower()

    def test_normalisation_is_idempotent_over_the_official_vocabulary(self):
        """The injury table stores a normalized status and the stint builder
        normalizes again, so a second pass must not move a player."""
        for designation in ("Out", "Questionable", "Probable", "Available"):
            once = src.availability_status(designation)
            assert src.availability_status(once) == once

    def test_an_unknown_status_resolves_healthy_rather_than_penalising(self):
        """An unrecognised word must not silently dock a player.

        The raw string is carried alongside so the unknown value stays
        visible in the artifact instead of resolving invisibly to healthy.
        """
        assert src.availability_status("Felt a little under the weather") \
            == "healthy"

    def test_the_weighted_table_has_no_caller_left(self):
        """The injury path decided for binary REMOVAL (a false flag drops the
        player from the pool and a replacement inherits the slot), so the
        weighted form of the same vocabulary was a second answer to a
        question the pipeline no longer asks - kept alive only by its own
        tests. The measured rates remain in
        ``config.PLAYER_EPM_DESIGNATION_PLAY_RATE``; the multiplier is gone
        rather than kept as an audited corpse."""
        assert not hasattr(src, "availability_multiplier")
        assert config.PLAYER_EPM_DESIGNATION_PLAY_RATE["out"] == 0.0
        assert config.PLAYER_EPM_DESIGNATION_PLAY_RATE["available"] == 0.817

    def test_an_empty_injuries_list_is_healthy_not_missing(self):
        """The common case, and the one that must not read as no data.

        ESPN populates the array sparsely by design - 7 of 76 rostered
        players in a four-team sample carried an entry - so empty is
        normal.
        """
        payload = {"athletes": [{"id": "1", "displayName": "A",
                                 "injuries": []}]}
        assert src.roster_availability(payload, team="BOS") == [
            ("1", "BOS", "healthy", "", None)]

    def test_an_injury_entry_sets_the_status_and_keeps_the_raw_text(self):
        payload = {"athletes": [{"id": "1", "displayName": "A",
                                 "injuries": [{"status": {"name": "Out"}}]}]}
        assert src.roster_availability(payload, team="BOS") == [
            ("1", "BOS", "out", "Out", None)]

    def test_the_real_espn_shape_is_a_bare_status_string(self):
        """Pinned against the live feed, which sends ``status`` as a string.

        Measured 2024-25: every injury entry on five rosters was exactly
        ``{"status": "Out", "date": "..."}`` with no nested ``name``. A
        parser written for the nested shape reads ``status.name`` off a
        string, finds nothing, and reports every player healthy - the gate
        stays open and every check downstream still passes.
        """
        payload = {"athletes": [{"id": "6430", "displayName": "Jimmy Butler",
                                 "injuries": [{"status": "Out",
                                               "date": "2026-08-24T15:23Z"}]}]}
        assert src.roster_availability(payload, team="GSW") == [
            ("6430", "GSW", "out", "Out", "2026-08-24T15:23Z")]

    def test_a_bare_day_to_day_string_is_halved(self):
        payload = {"athletes": [{"id": "1", "displayName": "A",
                                 "injuries": [{"status": "Day-To-Day",
                                               "date": "2026-09-02T16:32Z"}]}]}
        assert src.roster_availability(payload)[0][2] == "day_to_day"

    def test_the_published_date_is_carried_because_it_is_the_pit_floor(self):
        """The endpoint publishes current state only, so there is no history.

        What makes a snapshot usable is that each record carries the moment
        it was published. Dropping it would force the PIT floor to fall
        back to the snapshot's own date, which is the leak the column
        exists to prevent.
        """
        payload = {"athletes": [{"id": "1", "displayName": "A",
                                 "injuries": [{"status": "Out",
                                               "date": "2026-08-24T15:23Z"}]}]}
        assert src.roster_availability(payload)[0][4] == "2026-08-24T15:23Z"

    def test_the_status_mapping_is_idempotent(self):
        """Re-normalizing an already-normalized status must not change it.

        The injury table stores a normalized status and the stint builder
        normalizes it again. Without idempotence "day_to_day" falls through
        every branch and resolves to "healthy", so a partial absence is
        silently dropped and the player stays in the projected lineup.
        """
        for status in ("out", "day_to_day", "healthy"):
            once = src.availability_status(status)
            assert src.availability_status(once) == once

    def test_an_athlete_with_no_id_is_skipped(self):
        payload = {"athletes": [{"displayName": "A", "injuries": []}]}
        assert src.roster_availability(payload) == []


class TestPointInTimeDiscipline:
    def _season_games(self):
        return _frame([
            _row("a", "2024-11-01", 10, 5, 0),
            _row("a", "2024-11-02", 30, 10, 0),
            _row("a", "2024-11-03", 20, 20, 0),
        ])

    def test_a_games_own_line_never_enters_its_own_prior(self):
        """A rating dated on a player's own game must not see that game.

        The player's opener is the clean case: with the target day
        excluded there is nothing before it, so the prior is zero however
        much he was on the court - and the rating collapses to the league
        prior, which is the only defensible answer for a player with no
        evidence.
        """
        games = rapm.prepare_player_games(self._season_games())
        ratings = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2024-11-01"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        assert row.prior_eff == 0
        assert row.prior_minutes == 0
        assert row.prior_games == 0

    def test_a_player_with_no_evidence_is_still_emitted(self):
        """"No rating yet" and "never in the data" are different facts.

        A projection that cannot tell them apart will happily project a
        player who does not exist, so the row is present with a zero prior
        rather than dropped.
        """
        games = rapm.prepare_player_games(self._season_games())
        ratings = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2024-11-01"]))
        assert list(ratings.player_id) == ["a"]
        assert pd.isna(ratings.iloc[0].rapm_raw)

    def test_a_rating_does_not_see_the_target_game(self):
        games = rapm.prepare_player_games(self._season_games())
        ratings = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2024-11-02"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        # Only the opener is prior; the target day's line is not.
        assert row.prior_eff == pytest.approx(1.0)
        assert row.prior_minutes == pytest.approx(48)

    def test_a_rating_does_not_carry_the_previous_season(self):
        """The boundary the season partition exists to hold.

        A carry-over is invisible in the output: a player with a full
        season of evidence looks equally well-rated either way.
        """
        games = rapm.prepare_player_games(_frame([
            _row("a", "2024-06-01", 40, 20, 0, season="2023-24"),
            _row("a", "2024-11-01", 10, 5, 0, season="2024-25"),
        ]))
        ratings = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2024-11-02"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        # Only the 2024-25 game counts: one 48-minute game of evidence.
        assert row.prior_eff == pytest.approx(1.0)
        assert row.prior_games == 1

    def test_the_prior_is_a_running_total_read_not_summed(self):
        """Regression: summing the running total grows with the square of the
        season and returns a number several times the player's real
        plus-minus."""
        games = rapm.prepare_player_games(self._season_games())
        ratings = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2024-11-04"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        # 3 games of 48 minutes each - a plain sum, not a sum of prefixes.
        assert row.prior_eff == pytest.approx(3.0)
        assert row.prior_minutes == pytest.approx(144)

    def test_duplicate_player_games_do_not_double_count(self):
        """Overlapping fetch windows repeat a player-game, and a duplicate
        would double both sides of the rate - which looks like a much
        better performance than it was."""
        games = rapm.prepare_player_games(_frame([
            _row("a", "2024-11-01", 10, 5, 0),
            _row("a", "2024-11-01", 10, 5, 0),
            _row("a", "2024-11-05", 4, 2, 0),
        ]))
        assert len(games) == 2
        ratings = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2024-11-06"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        assert row.prior_eff == pytest.approx(2.0)
        assert row.prior_minutes == pytest.approx(96)

    def test_a_season_spanning_new_year_is_one_season_not_two(self):
        """An NBA season runs October to June, so it crosses New Year.

        Deriving the season from the calendar year would split every season
        in half at 1 January and hand each half a thin prior.
        """
        games = rapm.prepare_player_games(_frame([
            _row("a", "2024-12-30", 10, 5, 0, season="2024-25"),
            _row("a", "2025-01-02", 10, 5, 0, season="2024-25"),
        ]))
        assert set(games.season) == {"2024-25"}


class TestPositionSegmentedPrior:
    """The position prior is the mean raw beta among the SAME fit's
    evidenced members at that position: shrink target and thing shrunk
    come out of one solve, so they can never disagree about scale or
    about which date they belong to."""

    @staticmethod
    def _two_position_league():
        """A centre-heavy home side versus a full-minute visitor, with the
        minute split shifting across nights.

        The shift is load-bearing: constant shares make every beta column
        a scalar copy of the intercept, the free intercept explains the
        whole margin, and every beta solves to exactly zero - correct,
        but a fixture that cannot tell two positions apart.
        """
        rows = []
        for index, margin in enumerate((20.0, 6.0, 32.0)):
            day = f"2024-11-{index + 1:02d}"
            gid = f"p{index}"
            c_min = (44.0, 44.0, 8.0)[index]
            g_min = (4.0, 4.0, 40.0)[index]
            rows.append(_row("c", day, 0, 0, position="C", team="AA",
                             minutes=c_min, points=100.0 + margin,
                             game_id=gid))
            rows.append(_row("g", day, 0, 0, position="G", team="AA",
                             minutes=g_min, points=0.0, game_id=gid))
            rows.append(_row("d", day, 0, 0, position="C", team="BB",
                             minutes=48.0, points=100.0, game_id=gid))
        return rapm.prepare_player_games(_frame(rows))

    def _ratings(self):
        return rapm.build_player_rapm(
            self._two_position_league(),
            target_dates=pd.Series(["2024-11-04"]))

    def test_the_league_mean_differs_by_position(self):
        """The reason the prior is segmented at all.

        One league mean would rate this guard as an above-average centre
        and that centre as a below-average guard - the NHL defenceman
        problem in a different sport.
        """
        ratings = self._ratings()
        lg = dict(zip(ratings.position, ratings.lg_rapm))
        assert lg["G"] > lg["C"]
        raw = dict(zip(ratings.player_id, ratings.rapm_raw))
        assert raw["g"] != raw["c"]

    def test_a_rating_is_pulled_toward_its_own_position_not_the_league(self):
        ratings = self._ratings()
        row_c = ratings[ratings.player_id == "c"].iloc[0]
        # Between the player's own raw beta and HIS position's mean, never
        # past either.
        assert ((row_c.rapm_shrunk - row_c.lg_rapm)
                * (row_c.rapm_raw - row_c.lg_rapm) > 0)
        assert (abs(row_c.rapm_shrunk - row_c.lg_rapm)
                < abs(row_c.rapm_raw - row_c.lg_rapm))
        # The sole evidenced guard IS his position's mean, so his rating
        # stays exactly on his raw beta - a single unsegmented league mean
        # would drag him toward the centre cell instead.
        row_g = ratings[ratings.player_id == "g"].iloc[0]
        assert row_g.rapm_shrunk == pytest.approx(row_g.rapm_raw)
        assert row_g.rapm_shrunk == pytest.approx(row_g.lg_rapm)

    def test_prior_strength_is_defined_for_every_position_in_play(self):
        ratings = self._ratings()
        k = dict(zip(ratings.position, ratings.k_eff))
        assert set(k) == {"G", "C"}
        assert all(value > 0 for value in k.values())


class TestShrinkageBehaviour:
    def test_a_one_game_rookie_is_pulled_back_toward_the_league(self):
        """The case the prior exists for, as an end-to-end assertion.

        A single game's beta is almost pure noise. The veterans here split
        twenty alternating wins and losses (net margin zero, so the
        league mean sits at 0 by construction), the rookie arrives with
        one 40-point night, and the rating must carry some - not all - of
        it onward. With lg == 0 the shrinkage reads exactly
        ``raw / (1 + k)``, which pins the arithmetic as well as the
        direction.
        """
        rows = []
        for index in range(20):
            day = f"2024-11-{index + 1:02d}"
            gid = f"v{index}"
            home_pts, away_pts = ((110.0, 100.0) if index % 2 == 0
                                  else (100.0, 110.0))
            rows.append(_row("vet1", day, 0, 0, team="V1", minutes=48,
                             points=home_pts, game_id=gid))
            rows.append(_row("vet2", day, 0, 0, team="V2", minutes=48,
                             points=away_pts, game_id=gid))
        rows.append(_row("rookie", "2024-11-21", 0, 0, team="R",
                         minutes=48, points=140.0, game_id="r0"))
        rows.append(_row("opp", "2024-11-21", 0, 0, team="O",
                         minutes=48, points=100.0, game_id="r0"))
        games = rapm.prepare_player_games(_frame(rows))
        ratings = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2024-12-01"]))
        rookie = ratings[ratings.player_id == "rookie"].iloc[0]
        assert rookie.rapm_raw > 0
        assert rookie.lg_rapm == pytest.approx(0.0, abs=1e-9)
        # Moved off the raw beta, toward the league, but not all the way -
        # some of it is real, and with a zero league mean the exact
        # shrinkage is raw * eff / (eff + k).
        assert 0 < rookie.rapm_shrunk < rookie.rapm_raw
        assert rookie.rapm_shrunk == pytest.approx(
            rookie.rapm_raw * rookie.prior_eff
            / (rookie.prior_eff + rookie.k_eff))

    def test_shrinkage_preserves_rank_but_reduces_spread(self):
        """Both halves matter: a prior that reorders players is broken, and
        one that does not reduce spread is not doing anything."""
        rows = []
        for index, (player, margin) in enumerate(
                zip("abcde", (40, 32, 24, 16, 8))):
            day = f"2024-11-{index + 1:02d}"
            gid = f"r{index}"
            rows.append(_row(player, day, 0, 0, team="W", minutes=48,
                             points=100.0 + margin, game_id=gid))
            rows.append(_row(f"o{index}", day, 0, 0, team="L", minutes=48,
                             points=100.0, game_id=gid))
        games = rapm.prepare_player_games(_frame(rows))
        ratings = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2024-12-01"]))
        rated = ratings.dropna(subset=["rapm_raw"])
        # All ten evidenced players: five protagonists and five foils.
        assert len(rated) == 10
        assert rated.rapm_raw.corr(rated.rapm_shrunk, method="spearman") > 0.9
        assert rated.rapm_shrunk.std() < rated.rapm_raw.std()

    def test_k_follows_the_target_season_from_completed_priors(self):
        """The per-season strength, end to end.

        A 2025-26 target's k is 20% of the 2024-25 mean only - the
        2025-26 player-seasons, including games AFTER the target date,
        never enter it. A 2024-25 target has no completed prior season in
        the frame, so it keeps the whole-frame mean (the pre-2026-10-01
        behavior) rather than a constant that ignores the frame's own
        season scale. Eff-games per 48-minute game is 1, so the counts
        read directly: five games and twenty games.
        """
        rows = []
        for day in range(1, 6):
            rows.append(_row("a", f"2024-11-0{day}", 0, 0))
        for day in range(1, 21):
            rows.append(_row("b", f"2025-11-{day:02d}", 0, 0,
                             season="2025-26"))
        games = rapm.prepare_player_games(_frame(rows))
        # 2024-25's only player-season is five games; the 2025-26 mean
        # never enters the 2025-26 target's strength.
        late = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2025-11-03"]))
        assert late.k_eff.iloc[0] == pytest.approx(0.20 * 5)
        # The earliest season keeps the whole-frame mean: both player-
        # seasons, 5 and 20 games, averaged.
        early = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2024-11-03"]))
        assert early.k_eff.iloc[0] == pytest.approx(0.20 * (5 + 20) / 2)
        assert early.k_eff.iloc[0] != late.k_eff.iloc[0]

    def test_the_rating_carries_no_availability_multiplier(self):
        """The multiplier is gone, and its absence is the point.

        Zero-weighting an injured rating leaves the player IN the pool
        dragging the mean toward zero. Removal happens at pool construction
        instead, so a rating is a property of the player and nothing else.
        The raw status label is carried for callers that want it; the
        rating itself is byte-identical either way.
        """
        games = rapm.prepare_player_games(_frame([
            _row("a", "2024-11-01", 10, 5, 0, team="A", points=110,
                 game_id="g1"),
            _row("b", "2024-11-01", 10, 5, 0, team="B", points=90,
                 game_id="g1"),
        ]))
        availability = pd.DataFrame({"player_id": ["a"], "status": ["out"]})
        ratings = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2024-11-02"]),
            availability=availability)
        assert "availability_multiplier" not in ratings.columns
        assert "availability" not in ratings.columns
        # The rating is byte-identical with and without the injury input: a
        # player still HAS an RAPM whether or not he is dressing tonight.
        # Only the pool changes.
        without = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2024-11-02"]))
        assert ratings.iloc[0].rapm_shrunk == pytest.approx(
            without.iloc[0].rapm_shrunk)


class TestPlayerRapmWithoutInputs:
    @pytest.mark.parametrize("stats", [None, pd.DataFrame()])
    def test_no_log_yields_an_empty_frame_not_none(self, stats):
        """A caller must be able to tell no data from a failed pull."""
        out = rapm.prepare_player_games(stats)
        assert out is not None and out.empty

    def test_a_log_without_the_rating_columns_yields_nothing(self):
        out = rapm.prepare_player_games(pd.DataFrame(
            {"player_id": ["a"], "gameday": ["2024-11-01"],
             "plus_minus": [10]}))
        assert out.empty

    def test_a_frame_with_no_resolved_positions_refuses_to_rate(self):
        """Rating without a position would mean an UNSEGMENTED prior.

        That is the single-league-mean case the segmentation exists to
        avoid, so it is refused rather than quietly produced.
        """
        games = rapm.prepare_player_games(_frame([_row("a", "2024-11-01",
                                                      10, 5, 0)]))
        games["position"] = np.nan
        assert rapm.build_player_rapm(
            games, target_dates=pd.Series(["2024-11-02"])).empty

    def test_no_prior_date_yields_an_empty_frame(self):
        games = rapm.prepare_player_games(_frame([_row("a", "2024-11-01",
                                                      10, 5, 0)]))
        assert rapm.build_player_rapm(
            games, target_dates=pd.Series([])).empty

    def test_a_target_before_any_game_yields_nothing(self):
        games = rapm.prepare_player_games(_frame([_row("a", "2024-11-01",
                                                      10, 5, 0)]))
        assert rapm.build_player_rapm(
            games, target_dates=pd.Series(["2024-10-01"])).empty


class TestEvidenceSeasonFallback:
    """A target whose OWN season has no game yet fits from the last one.

    Two seasons read differently on purpose: the ROSTER side is non-strict
    (the players taking the floor tonight are in the frame to be rated,
    even though none has played yet) and the FIT side is strict (a game on
    the target day cannot be evidence for that day's rating). They diverge
    exactly on a season's first day - the roster is this season's, the
    solve falls back to last season's - so opening night rates against the
    completed season's fit rather than against nothing, and day 2 onward
    is strictly in-season. Every fallback row is still strictly before the
    target: the point-in-time floor is untouched.
    """

    @staticmethod
    def _two_seasons():
        rows = []
        for index, day in enumerate(("2024-11-01", "2024-11-05")):
            gid = f"f{index}"
            rows.append(_row("a", day, 0, 0, team="A", minutes=48,
                             points=110.0, game_id=gid, season="2024-25"))
            rows.append(_row("z", day, 0, 0, team="Z", minutes=48,
                             points=90.0, game_id=gid, season="2024-25"))
        return rapm.prepare_player_games(_frame(rows))

    def test_the_first_game_of_a_season_is_rated_from_the_completed_one(self):
        ratings = rapm.build_player_rapm(
            self._two_seasons(), target_dates=pd.Series(["2025-10-22"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        assert row.prior_eff == pytest.approx(2.0)  # both completed games
        assert row.prior_games == 2
        # The fit fell back WITH the prior, so there is a real shrink
        # target: a rating, not a rating-shaped NaN.
        assert pd.notna(row.rapm_shrunk)

    def test_the_fallback_stops_once_the_own_season_has_evidence(self):
        rows = [
            _row("a", "2024-11-01", 0, 0, team="A", minutes=48,
                 points=110.0, game_id="f0", season="2024-25"),
            _row("z", "2024-11-01", 0, 0, team="Z", minutes=48,
                 points=90.0, game_id="f0", season="2024-25"),
            _row("a", "2025-10-01", 0, 0, team="A", minutes=48,
                 points=105.0, game_id="s0", season="2025-26"),
            _row("o", "2025-10-01", 0, 0, team="O", minutes=48,
                 points=100.0, game_id="s0", season="2025-26"),
        ]
        games = rapm.prepare_player_games(_frame(rows))
        ratings = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2025-10-22"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        # The partition still holds: one in-season game, and only that one.
        assert row.prior_eff == pytest.approx(1.0)
        assert row.prior_games == 1

    def test_the_fallback_picks_the_most_recent_eligible_season(self):
        """Three seasons, own season absent, target after all of them:
        the fallback must compare DATES and take the most recent eligible
        season, not the first season the index happens to hold.

        The original loop took ``.max()`` of the boolean eligibility mask -
        np.True_ for any season with a pre-target row - so the first key in
        the index was unbeatable and the OLDEST season won every fallback.
        Two-season fixtures could not see it (first eligible coincides with
        most recent), but every preseason target does: measured on the
        2026-10-04 delivery, the player board's priors matched 2023-24
        minutes for 550/550 players while the slate rated 2026-27 games.
        """
        rows = [
            _row("a", "2024-01-15", 0, 0, minutes=48, season="2023-24",
                 game_id="o0"),
            _row("a", "2025-01-15", 0, 0, minutes=48, season="2024-25",
                 game_id="o1"),
            _row("a", "2026-01-15", 0, 0, minutes=48, season="2025-26",
                 game_id="o2"),
        ]
        games = rapm.prepare_player_games(_frame(rows))
        index = rapm._season_evidence_index(games)
        # The index is chronological - first key oldest - exactly the
        # production frame's order, and exactly the buggy loop's trap.
        assert list(index) == ["2023-24", "2024-25", "2025-26"]
        assert rapm._evidence_season(
            games, pd.Timestamp("2026-10-20"), index, strict=True) == "2025-26"
        assert rapm._evidence_season(
            games, pd.Timestamp("2026-10-20"), index, strict=False) == "2025-26"
        ratings = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2026-10-20"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        # The prior is the MOST RECENT season's game, never the oldest's.
        assert row.prior_games == 1
        assert row.prior_minutes == pytest.approx(48.0)
        assert row.prior_eff == pytest.approx(1.0)

    def test_opening_night_with_no_prior_season_is_honestly_empty(self):
        """A decided opening-night game with no season to fall back to
        rates from NOTHING - no prior, no league cell, NaN rating.

        Not a rating wearing a prior's clothes: the row still exists (the
        players ARE in the frame to be rated), but nothing is invented.
        """
        rows = [
            _row("a", "2025-10-22", 0, 0, team="A", minutes=48,
                 points=110.0, game_id="o1", season="2025-26"),
            _row("b", "2025-10-22", 0, 0, team="B", minutes=48,
                 points=90.0, game_id="o1", season="2025-26"),
        ]
        games = rapm.prepare_player_games(_frame(rows))
        ratings = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2025-10-22"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        assert row.prior_eff == 0 and row.prior_minutes == 0
        assert row.prior_games == 0
        assert pd.isna(row.rapm_raw)
        assert pd.isna(row.rapm_shrunk)

    def test_opening_night_shrinks_toward_the_prior_seasons_fit(self):
        """Game 1 gets the treatment game 2 already had.

        The veteran played last season, so opening night his zero OWN
        prior mixes against last season's fit: his rating exists from the
        first tip. The debutant has no prior-season row at all, so his
        rating IS the position prior carried out of that fit - zero prior
        plus a real shrink target lands exactly on it.
        """
        rows = [
            _row("c", "2024-11-01", 0, 0, team="A", minutes=48,
                 points=110.0, game_id="f0", season="2024-25"),
            _row("z", "2024-11-01", 0, 0, team="Z", minutes=48,
                 points=90.0, game_id="f0", season="2024-25"),
            _row("c", "2025-10-22", 0, 0, team="A", minutes=48,
                 points=100.0, game_id="o1", season="2025-26"),
            _row("d", "2025-10-22", 0, 0, team="B", minutes=48,
                 points=90.0, game_id="o1", season="2025-26"),
        ]
        games = rapm.prepare_player_games(_frame(rows))
        ratings = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2025-10-22"]))
        debut = ratings[ratings.player_id == "d"].iloc[0]
        assert debut.prior_eff == 0 and debut.prior_games == 0
        assert pd.isna(debut.rapm_raw)
        assert pd.notna(debut.lg_rapm)   # the borrowed fit's cell exists
        assert debut.rapm_shrunk == pytest.approx(debut.lg_rapm)
        vet = ratings[ratings.player_id == "c"].iloc[0]
        assert vet.prior_eff == pytest.approx(1.0)   # last season's game
        assert pd.notna(vet.rapm_raw)
        # His rating mixes his (thin, own-0) evidence against the same fit:
        # the shrinkage arithmetic itself, whatever the fixture's numbers.
        expected = ((vet.prior_eff * vet.rapm_raw
                     + vet.lg_rapm * vet.k_eff)
                    / (vet.prior_eff + vet.k_eff))
        assert vet.rapm_shrunk == pytest.approx(expected)

    def test_season_boundary_day_two_is_in_season_not_carryover(self):
        """Day 2+ rates strictly in-season: prior-season form is dropped.

        Pins the carryover answer the drift report's STARVED/LOW pl_rapm
        coverage depends on: early-season thinness is the shrinkage doing
        its job, not missing evidence. The partition exists so a new
        season's rating never blends the previous one's tail - the same
        convention MLB states for its season-to-date ERA/K/9 LAGs.
        """
        games = rapm.prepare_player_games(_frame([
            _row("a", "2025-04-01", 40, 20, 0, season="2024-25"),
            _row("a", "2025-10-21", 10, 5, 0, season="2025-26"),
        ]))
        ratings = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2025-10-23"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        assert row.prior_eff == pytest.approx(1.0)
        assert row.prior_games == 1

    def test_the_fallback_never_reads_a_game_on_or_after_the_target(self):
        """Point-in-time is about WHEN, not about which season."""
        games = rapm.prepare_player_games(_frame([
            _row("a", "2024-11-01", 0, 0, minutes=48, season="2024-25"),
            _row("a", "2025-10-01", 0, 0, minutes=48, season="2025-26"),
        ]))
        ratings = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2025-10-22"]))
        # The 10-01 game is in the target's OWN season and strictly before
        # it: the fallback must not prefer last season just because it is
        # further away.
        assert ratings[ratings.player_id == "a"].iloc[0].prior_eff \
            == pytest.approx(1.0)
        # ...and a target BEFORE that game falls back to the completed
        # season, still strictly earlier.
        early = rapm.build_player_rapm(
            games, target_dates=pd.Series(["2025-09-30"]))
        assert early[early.player_id == "a"].iloc[0].prior_eff \
            == pytest.approx(1.0)
        assert early[early.player_id == "a"].iloc[0].prior_games == 1

    def test_a_target_before_every_game_still_rates_nothing(self):
        """No season has evidence before the league's first game, so the
        fallback has nothing to offer and must not invent a prior."""
        assert rapm.build_player_rapm(
            self._two_seasons(),
            target_dates=pd.Series(["2024-10-01"])).empty

    def test_the_fit_and_the_prior_draw_the_same_evidence(self):
        """The shrink target and what it shrinks must come out of ONE
        season's solve: the build's raw rating is the fit's own beta for
        the strict evidence season, and lg_rapm is the mean of exactly
        those betas."""
        games = self._two_seasons()
        target = pd.Timestamp("2025-10-22")
        ratings = rapm.build_player_rapm(
            games, target_dates=pd.Series([target]))
        season = rapm._evidence_season(games, target, strict=True)
        entries = rapm.team_game_entries(games)
        beta, _ = rapm.fit_rapm(entries[entries.season == season],
                                target=target)
        row = ratings[ratings.player_id == "a"].iloc[0]
        assert row.rapm_raw == pytest.approx(beta["a"])
        assert row.lg_rapm == pytest.approx(float(beta.mean()))
