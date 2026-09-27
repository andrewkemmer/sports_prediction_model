"""Tests for the official-report ingester and the de-dup invariant.

The two properties worth a test here are both invisible when they break. A
duplicated player-game row doubles a rating's sample without raising anything.
A designation read from the wrong filing is still a real designation from a
real filing, so a lookahead leak looks exactly like a correct answer - the
feature trains, the numbers look fine, and the model is simply better-informed
than anyone betting on it.
"""
from datetime import date, datetime

import pandas as pd
import pytest

import ingestion
import nba_injury_report as ir


# ---------------------------------------------------------------------------
# Filename eras
# ---------------------------------------------------------------------------


def test_hourly_stamp_before_the_format_change():
    when = datetime(2025, 11, 15, 22, 0)
    assert ir.report_stamp(when) == "2025-11-15_10PM"


def test_quarter_hourly_stamp_after_the_format_change():
    when = datetime(2026, 1, 10, 18, 45)
    assert ir.report_stamp(when) == "2026-01-10_06_45PM"


def test_midnight_and_noon():
    assert ir.report_stamp(datetime(2026, 1, 10, 0, 0)) == "2026-01-10_12_00AM"
    assert ir.report_stamp(datetime(2026, 1, 10, 12, 0)) == "2026-01-10_12_00PM"


def test_stamp_uses_a_hyphen_not_a_space():
    # "Injury Report_" with a space answers 403, which is indistinguishable
    # from "no such report" and is how this archive was nearly written off.
    assert "-" in ir.REPORT_URL
    assert " " not in ir.REPORT_URL


# ---------------------------------------------------------------------------
# The twelve-hour clock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("07:00 (ET)", 19),      # an evening game read as 7am finds no history
    ("10:30 (ET)", 22),
    ("01:00 (ET)", 13),      # early afternoon
    ("12:00 (ET)", 12),
])
def test_tipoff_resolves_to_afternoon(text, expected):
    tip = ir.tipoff_et(date(2026, 1, 10), text)
    assert tip.hour == expected
    assert tip.minute == int(text.split(":")[1].split()[0][:2])


def test_tipoff_keeps_its_date():
    assert ir.tipoff_et(date(2026, 1, 10), "08:00 (ET)").date() == date(2026, 1, 10)


# ---------------------------------------------------------------------------
# The point-in-time rule
# ---------------------------------------------------------------------------


def test_designation_at_tipoff_is_not_usable_before_tipoff():
    """The strictness that makes this point-in-time.

    A filing published at the instant of tipoff cannot gate that game. If the
    comparison were ``<=`` a status learned at 7:00:00 would gate a 7:00
    game - one record of lookahead per game, invisible in the output.
    """
    tip = datetime(2026, 1, 10, 19, 0)
    at_tipoff = datetime(2026, 1, 10, 19, 0)
    before = datetime(2026, 1, 10, 18, 45)
    assert before < tip
    assert not (at_tipoff < tip)


def test_every_filing_considered_is_before_tipoff_by_construction():
    """The walk-back cannot return a post-tipoff filing, so the guard is
    structural rather than a filter that could be forgotten."""
    tip = datetime(2026, 1, 10, 19, 0)
    for when in ir._steps(tip.date(), tip, 24):
        assert when < tip
        assert when.date() == tip.date()


def test_walk_back_stops_at_the_day_boundary():
    tip = datetime(2026, 1, 10, 0, 15)
    moments = list(ir._steps(tip.date(), tip, 24))
    assert all(m.date() == date(2026, 1, 10) for m in moments)


def test_walk_back_step_width_matches_the_era():
    old = list(ir._steps(date(2025, 11, 15), datetime(2025, 11, 15, 20, 0), 3))
    new = list(ir._steps(date(2026, 1, 10), datetime(2026, 1, 10, 20, 0), 3))
    assert (old[0] - old[1]) == pd.Timedelta(minutes=60)
    assert (new[0] - new[1]) == pd.Timedelta(minutes=15)


# ---------------------------------------------------------------------------
# Designation vocabulary
# ---------------------------------------------------------------------------


def test_vocabulary_is_the_six_the_league_actually_files():
    assert set(ir.DESIGNATIONS) == {
        "Out", "Available", "Doubtful", "Questionable", "Probable", "Recovery"}


def test_the_will_not_play_set_is_out_and_recovery():
    """``Recovery`` is filed by the league for players who will not dress, so
    it sits with ``Out`` on that basis rather than because it was measured -
    it has zero observations at tipoff."""
    assert ir.WILL_NOT_PLAY == ("Out", "Recovery")


def test_doubtful_is_an_absence_even_though_it_is_not_will_not_play():
    """Doubtful joined the out set on evidence (0 for 5), not on the wording
    of ``WILL_NOT_PLAY``. The two lists answer different questions and are not
    expected to agree."""
    import config
    assert config.PLAYER_TS_STATUS_TREATMENT["doubtful"] == "absent"


# ---------------------------------------------------------------------------
# The out set
# ---------------------------------------------------------------------------


def test_out_set_is_out_doubtful_and_recovery():
    """Doubtful joins Out on evidence, not on word order: it went 0 for 5."""
    assert ir.availability_state("Out") == ir.ABSENT
    assert ir.availability_state("Doubtful") == ir.ABSENT
    assert ir.availability_state("Recovery") == ir.ABSENT


def test_questionable_is_pooled_into_available():
    """Deliberate, and the reason is sample size rather than play rate.

    Questionable played 34 of 52 - genuinely uncertain - but 52 observations
    is 0.4% of a season with a +/-13 point interval, and a state whose only
    parameter is estimated from that is a parameter fitted to almost nothing.
    The accepted cost is that roughly a third of those appearances carry
    0.816 instead of 0.654.
    """
    assert ir.availability_state("Questionable") == ir.AVAILABLE


def test_available_and_probable_are_available():
    assert ir.availability_state("Available") == ir.AVAILABLE
    assert ir.availability_state("Probable") == ir.AVAILABLE


def test_states_partition_the_vocabulary():
    seen = {ir.availability_state(d) for d in ir.DESIGNATIONS}
    assert seen == {ir.ABSENT, ir.AVAILABLE}


def test_the_pooled_rate_is_not_the_thin_questionable_rate():
    """If this ever reads 0.654 the pooling has been undone by a well-meaning
    edit, and the 0.4%-sample estimate is back."""
    import config
    rates = config.PLAYER_TS_DESIGNATION_PLAY_RATE
    assert rates["questionable"] == rates["available"] == rates["probable"]
    assert rates["questionable"] != 0.654


def test_the_pooled_rate_is_actually_the_pooled_rate():
    """The provenance, recomputed.

    The constant was once written as the Available+Questionable blend and
    described as the three-way pool, which is the kind of drift that leaves a
    number nobody can reproduce. Pinning the arithmetic means an edit to the
    constant has to edit this too.
    """
    import config
    played = 2006 + 34 + 23        # Available, Questionable, Probable
    total = 2448 + 52 + 24
    pooled = round(played / total, 3)
    assert config.PLAYER_TS_DESIGNATION_PLAY_RATE["available"] == pooled
    assert pooled == 0.817


def test_an_unknown_designation_is_unknown_not_healthy():
    """Defaulting to available turns an unfamiliar word into a no-op filter."""
    assert ir.availability_state("Fifty-Fifty") == ir.UNKNOWN
    assert ir.availability_state("") == ir.UNKNOWN
    assert ir.availability_state(None) == ir.UNKNOWN


def test_espn_day_to_day_is_not_nba_vocabulary():
    """ESPN's word for this league. Accepting it would let a mapper treat a
    two-status snapshot and a six-status official filing as one vocabulary."""
    assert ir.availability_state("Day-To-Day") == ir.UNKNOWN


def test_config_mirrors_the_module_mapping():
    import config
    for designation, state in ir.DESIGNATION_STATE.items():
        assert config.PLAYER_TS_STATUS_TREATMENT[designation] == state


def test_measured_play_rates_are_between_zero_and_one():
    import config
    assert {d.lower() for d in ir.DESIGNATIONS} == \
        set(config.PLAYER_TS_DESIGNATION_PLAY_RATE)
    for rate in config.PLAYER_TS_DESIGNATION_PLAY_RATE.values():
        assert 0.0 <= rate <= 1.0


def test_play_rates_are_ordered_as_the_states_suggest():
    import config
    rates = config.PLAYER_TS_DESIGNATION_PLAY_RATE
    assert rates["out"] < rates["questionable"]
    assert rates["questionable"] <= 1.0


def test_the_partial_placeholder_constant_is_gone():
    """``PLAYER_TS_DAY_TO_DAY_PLAY_RATE`` described a partial bucket that no
    designation produces any more. Keeping a number named after a retired
    state is how an obsolete assumption outlives the decision that killed it.
    """
    import config
    assert not hasattr(config, "PLAYER_TS_DAY_TO_DAY_PLAY_RATE")


# ---------------------------------------------------------------------------
# The four tables must not drift apart
# ---------------------------------------------------------------------------


def test_every_table_agrees_about_every_designation():
    """Three places describe the same vocabulary: the module's state map,
    ``config.PLAYER_TS_STATUS_TREATMENT``, and the measured play rates.
    Nothing forces them to agree, so nothing would notice when they stop -
    which is how a rating ends up halving a player the config calls absent
    and the source calls available. (The source's old weighted multiplier
    was removed once the pipeline decided unavailability is a REMOVAL, not a
    weight; the rates themselves live only in the config table now.)"""
    import config
    import nba_sources as sources
    for designation in ir.DESIGNATIONS:
        key = designation.lower()
        normalized = sources.availability_status(designation)
        assert normalized == key
        assert ir.availability_state(designation) == \
            config.PLAYER_TS_STATUS_TREATMENT[key]
        assert sources.availability_status(designation) in \
            config.PLAYER_TS_DESIGNATION_PLAY_RATE


def test_a_state_implies_the_right_weight_ordering():
    """absent < available, and every available designation is weighted
    identically. This is the property the whole mapping exists to express, so
    it is asserted as ordering rather than as six separate constants."""
    import config
    rates = config.PLAYER_TS_DESIGNATION_PLAY_RATE
    absent = [rates[d] for d in ("out", "doubtful", "recovery")]
    available = [rates[d] for d in ("questionable", "available", "probable")]
    assert max(absent) < min(available)
    assert len(set(available)) == 1


# ---------------------------------------------------------------------------
# De-duplication
# ---------------------------------------------------------------------------


def _log(rows):
    return pd.DataFrame(rows, columns=["nba_game_id", "player_id", "SEASON_ID",
                                       "points"])


def test_dedupe_keeps_one_row_per_player_game():
    frame = _log([
        ("0022500001", 2544.0, "22025", 30),
        ("0022500001", 2544.0, "22025", 30),   # the same line twice
        ("0022500001", 201939.0, "22025", 12),
    ])
    out = ingestion._dedupe_player_games(frame)
    assert len(out) == 2


def test_dedupe_leaves_a_clean_frame_untouched():
    frame = _log([
        ("0022500001", 2544.0, "22025", 30),
        ("0022500001", 201939.0, "22025", 12),
    ])
    assert len(ingestion._dedupe_player_games(frame)) == 2


def test_same_player_two_games_is_not_a_duplicate():
    frame = _log([
        ("0022500001", 2544.0, "22025", 30),
        ("0022500002", 2544.0, "22025", 22),
    ])
    assert len(ingestion._dedupe_player_games(frame)) == 2


def test_player_id_dtype_does_not_defeat_the_key():
    """player_id arrives as a float column. Stringifying it the naive way
    gives '1628983.0' and joins nothing, so the key must be taken as-is."""
    frame = _log([
        ("0022500001", 1628983.0, "22025", 4),
        ("0022500001", 1628983.0, "22025", 4),
    ])
    assert len(ingestion._dedupe_player_games(frame)) == 1


def test_dedupe_passes_through_a_frame_it_cannot_key():
    frame = pd.DataFrame({"points": [1, 2]})
    assert len(ingestion._dedupe_player_games(frame)) == 2
