"""Tests for the official-report ingester and the de-dup invariant.

The two properties worth a test here are both invisible when they break. A
duplicated player-game row doubles a rating's sample without raising anything.
A designation read from the wrong filing is still a real designation from a
real filing, so a lookahead leak looks exactly like a correct answer - the
feature trains, the numbers look fine, and the model is simply better-informed
than anyone betting on it.
"""
from datetime import date, datetime
from pathlib import Path

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
    assert config.PLAYER_EPM_STATUS_TREATMENT["doubtful"] == "absent"


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
    rates = config.PLAYER_EPM_DESIGNATION_PLAY_RATE
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
    assert config.PLAYER_EPM_DESIGNATION_PLAY_RATE["available"] == pooled
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
        assert config.PLAYER_EPM_STATUS_TREATMENT[designation] == state


def test_measured_play_rates_are_between_zero_and_one():
    import config
    assert {d.lower() for d in ir.DESIGNATIONS} == \
        set(config.PLAYER_EPM_DESIGNATION_PLAY_RATE)
    for rate in config.PLAYER_EPM_DESIGNATION_PLAY_RATE.values():
        assert 0.0 <= rate <= 1.0


def test_play_rates_are_ordered_as_the_states_suggest():
    import config
    rates = config.PLAYER_EPM_DESIGNATION_PLAY_RATE
    assert rates["out"] < rates["questionable"]
    assert rates["questionable"] <= 1.0


def test_the_partial_placeholder_constant_is_gone():
    """``PLAYER_EPM_DAY_TO_DAY_PLAY_RATE`` described a partial bucket that no
    designation produces any more. Keeping a number named after a retired
    state is how an obsolete assumption outlives the decision that killed it.
    """
    import config
    assert not hasattr(config, "PLAYER_EPM_DAY_TO_DAY_PLAY_RATE")


# ---------------------------------------------------------------------------
# The four tables must not drift apart
# ---------------------------------------------------------------------------


def test_every_table_agrees_about_every_designation():
    """Three places describe the same vocabulary: the module's state map,
    ``config.PLAYER_EPM_STATUS_TREATMENT``, and the measured play rates.
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
            config.PLAYER_EPM_STATUS_TREATMENT[key]
        assert sources.availability_status(designation) in \
            config.PLAYER_EPM_DESIGNATION_PLAY_RATE


def test_a_state_implies_the_right_weight_ordering():
    """absent < available, and every available designation is weighted
    identically. This is the property the whole mapping exists to express, so
    it is asserted as ordering rather than as six separate constants."""
    import config
    rates = config.PLAYER_EPM_DESIGNATION_PLAY_RATE
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


# ---------------------------------------------------------------------------
# The game-day submission: the moment availability is read at
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("home,tip,expected", [
    # Evening tips: window closes 13:00 in the HOME zone, expressed in ET.
    ("BOS", datetime(2026, 1, 10, 19, 0), datetime(2026, 1, 10, 13, 0)),
    ("UTA", datetime(2026, 1, 10, 21, 0), datetime(2026, 1, 10, 15, 0)),
    ("SAC", datetime(2026, 1, 10, 22, 30), datetime(2026, 1, 10, 16, 0)),
    # Arizona ignores DST: same wall-clock local close, later ET in June.
    ("PHX", datetime(2026, 1, 10, 21, 30), datetime(2026, 1, 10, 15, 0)),
    ("PHX", datetime(2026, 6, 10, 21, 30), datetime(2026, 6, 10, 16, 0)),
    # Central homes close at 13:00 local = 14:00 ET (the arena map once
    # said Eastern for CHI/MIL, moving their cutoff an hour EARLY - before
    # the league's own close - so a pre-submission filing gated the game).
    ("CHI", datetime(2026, 1, 10, 20, 0), datetime(2026, 1, 10, 14, 0)),
    ("MIL", datetime(2026, 1, 10, 20, 0), datetime(2026, 1, 10, 14, 0)),
    ("MIL", datetime(2026, 6, 10, 20, 0), datetime(2026, 6, 10, 14, 0)),
    # Tips at/before 17:00 local use the 8-10 a.m. window instead.
    ("SAC", datetime(2026, 1, 10, 15, 30), datetime(2026, 1, 10, 13, 0)),
    ("BOS", datetime(2026, 1, 10, 13, 0), datetime(2026, 1, 10, 10, 0)),
])
def test_the_window_closes_at_one_local_in_the_home_zone(home, tip, expected):
    """The league's rule, zone by zone: 11-1 local (8-10 for early tips).

    Read in ET because that is what the filing stamps are in - and the
    conversion must go through the calendar (DST), not a fixed offset, or
    every summer cutoff lands an hour off and the wrong filing gates.
    """
    window = ir.submission_window(tip.date(), tip, home)
    assert window.cutoff == expected
    assert window.cutoff < window.tipoff


@pytest.mark.parametrize("home,tip_hour,early", [
    ("BOS", 19, False),
    ("NYK", 13, True),
    ("SAC", 13, True),
    ("PHX", 13, True),
    # 19:00 ET in Denver/Utah is exactly 5 p.m. local - "5 p.m. or
    # earlier" puts it in the MORNING window, per the policy sentence.
    ("UTA", 19, True),
    ("DEN", 19, True),
    ("UTA", 20, False),
    ("DEN", 21, False),
])
def test_an_early_tip_selects_the_morning_window(home, tip_hour, early):
    tip = datetime(2026, 1, 10, tip_hour, 0)
    window = ir.submission_window(tip.date(), tip, home)
    assert window.early_tip is early
    assert window.window == ("08:00-10:00 local" if early
                             else "11:00-13:00 local")


def test_no_arena_window_reaches_tipoff():
    """A cutoff at/after tip-off would read post-game filings on a real
    schedule; sweep every arena, both seasons, every tip from 14:00 ET on
    (no NBA game tips earlier in any arena - the earliest real tips are
    12:00 ET for eastern homes and 12:00 PT for western ones)."""
    for home in ir.ARENA_TIMEZONE:
        for month in (1, 6):
            for hour in range(14, 24):
                tip = datetime(2026, month, 15, hour, 0)
                window = ir.submission_window(tip.date(), tip, home)
                assert window.cutoff < window.tipoff, (home, tip)


def test_an_unknown_home_degrades_to_eastern():
    """An unlisted club must not lose its game to a KeyError - and Eastern
    is the conservative direction: a western home mis-zoned closes the
    window EARLIER, so an older filing is read, never a newer one."""
    window = ir.submission_window(date(2026, 1, 10),
                                  datetime(2026, 1, 10, 19, 0), "ZZZ")
    assert window.cutoff == datetime(2026, 1, 10, 13, 0)


# ---------------------------------------------------------------------------
# Filing selection: first at/after the close, strictly before tip-off
# ---------------------------------------------------------------------------


class _FakeArchive:
    """A set of filing stamps that "exist", with no network and no PDFs.
    ``fetch_report`` is replaced by stamp membership (its only contract for
    the selection walks), and ``parse_report`` by a table, so every test
    below is deterministic and offline.
    """

    def __init__(self, stamps, records_by_stamp=None):
        self.stamps = set(stamps)
        self.records = records_by_stamp or {}
        self.fetches = []

    def fetch(self, when, cache_dir, timeout=30):
        self.fetches.append(when)
        if ir.report_stamp(when) in self.stamps:
            return cache_dir / ir._stamp_path(when)
        return None

    def parse(self, path, published_at):
        return self.records.get(Path(path).name, [])

    def install(self, monkeypatch):
        monkeypatch.setattr(ir, "fetch_report", self.fetch)
        monkeypatch.setattr(ir, "parse_report", self.parse)



def _submission_window(tip_hour=19):
    tip = datetime(2026, 1, 10, tip_hour, 0)
    return ir.submission_window(tip.date(), tip, "BOS")


def test_the_first_filing_at_or_after_the_close_is_chosen(monkeypatch, tmp_path):
    archive = _FakeArchive({"2026-01-10_12_00PM", "2026-01-10_01_00PM",
                            "2026-01-10_06_15PM"})
    archive.install(monkeypatch)
    chosen = ir.submission_filing(_submission_window(), tmp_path)
    assert chosen is not None
    filing, provenance = chosen
    # 13:00 ET is exactly the close: at/after counts.
    assert filing == datetime(2026, 1, 10, 13, 0)
    assert provenance == ir.PROVENANCE_SUBMISSION


def test_a_gap_after_the_close_walks_forward_to_the_next_filing(monkeypatch,
                                                                tmp_path):
    archive = _FakeArchive({"2026-01-10_01_30PM",  # 13:00 does not exist
                            "2026-01-10_06_15PM"})
    archive.install(monkeypatch)
    filing, provenance = ir.submission_filing(_submission_window(), tmp_path)
    assert filing == datetime(2026, 1, 10, 13, 30)
    assert provenance == ir.PROVENANCE_SUBMISSION


def test_no_filing_after_the_close_falls_back_to_the_last_before_it(monkeypatch,
                                                                    tmp_path):
    """The honest degrade: strictly older than the close, flagged - never
    the evening's filings, which would smuggle the late scratches in.

    (A filing that exists BETWEEN close and tip-off is the submission by
    definition, even an evening one - so the fallback case here is a day
    the league published nothing in that band at all.)
    """
    archive = _FakeArchive({"2026-01-10_12_00PM", "2026-01-10_12_45PM"})
    archive.install(monkeypatch)
    filing, provenance = ir.submission_filing(_submission_window(), tmp_path)
    assert filing == datetime(2026, 1, 10, 12, 45)
    assert provenance == ir.PROVENANCE_PRE_WINDOW
    assert filing < datetime(2026, 1, 10, 13, 0)


def test_no_filing_at_all_leaves_the_game_uncovered(monkeypatch, tmp_path):
    archive = _FakeArchive(set())
    archive.install(monkeypatch)
    assert ir.submission_filing(_submission_window(), tmp_path) is None


def test_a_future_stamp_is_never_sent_to_the_host(monkeypatch, tmp_path):
    """The network chokepoint behind the 2026-10-04 log repair: a stamp
    at or after this run's moment has no filing to fetch, so fetch_report
    must answer None WITHOUT a request. That run's pending slate walked
    16 days into its own future and the host rate-limited it - a 403 that
    reads exactly like a blocked archive."""
    import urllib.error
    monkeypatch.setattr(ir, "_now", lambda: datetime(2026, 10, 4, 16, 0))
    monkeypatch.setattr(ir, "FETCH_PAUSE_SEC", 0)
    attempts = []

    def _refused(request, timeout=30):
        attempts.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found",
                                     {}, None)

    monkeypatch.setattr(ir.urllib.request, "urlopen", _refused)
    before = ir.http_403_count
    assert ir.fetch_report(datetime(2026, 10, 20, 10, 0), tmp_path) is None
    assert attempts == [], "the future must never reach the host"
    assert ir.http_403_count == before
    # The guard is future-only, not an offline mode: a stamp already
    # published still goes to the host (404 -> None, as a miss reads).
    assert ir.fetch_report(datetime(2026, 10, 4, 15, 0), tmp_path) is None
    assert len(attempts) == 1


# ---------------------------------------------------------------------------
# The resolver: what reaches the feature, and what never does
# ---------------------------------------------------------------------------


def _record(when, player, status, team="Celtics", matchup="NYK@BOS",
            game_time="07:00"):
    return ir.Designation(game_date=date(2026, 1, 10),
                          game_time_et=game_time, matchup=matchup,
                          team=team, player=player, status=status,
                          reason="", published_at=when)


def _resolver_archive():
    """Filings for a 19:00 BOS home game: seed, submission, late scratch."""
    submission = datetime(2026, 1, 10, 13, 0)
    late = datetime(2026, 1, 10, 18, 15)
    seed = datetime(2026, 1, 10, 12, 0)
    records = {
        # The seed (discovery) carries the slate and the pre-submission state.
        ir._stamp_path(seed): [_record(seed, "Tatum, Jayson", "Questionable"),
                               _record(seed, "Brown, Jaylen", "Out"),
                               _record(seed, "Holiday, Jrue", "Out")],
        # The submission filing: Brown is cleared mid-day, Holiday stays Out.
        ir._stamp_path(submission): [
            _record(submission, "Tatum, Jayson", "Questionable"),
            _record(submission, "Brown, Jaylen", "Available"),
            _record(submission, "Holiday, Jrue", "Out")],
        # After the close: Tatum is a late scratch (18:15 for a 19:00 tip).
        ir._stamp_path(late): [
            _record(late, "Tatum, Jayson", "Out"),
            _record(late, "Brown, Jaylen", "Available"),
            _record(late, "Holiday, Jrue", "Out")],
    }
    archive = _FakeArchive({ir.report_stamp(s)
                             for s in (seed, submission, late)}, records)
    return archive


def _resolve(monkeypatch, tmp_path):
    archive = _resolver_archive()
    archive.install(monkeypatch)
    return ir.game_day_designations(date(2026, 1, 10), tmp_path)


def test_the_submission_status_wins_over_the_late_scratch(monkeypatch, tmp_path):
    """THE rule, asserted end to end.

    Tatum flips Questionable -> Out at 18:15 for a 19:00 tip. The slate
    cannot know that at the submission moment, so the feature must not
    either: ``status`` is what the league had posted by the close, and the
    scratch lives only in ``status_tipoff`` where nothing that gates a pool
    reads it.
    """
    frame = _resolve(monkeypatch, tmp_path)
    tatum = frame[frame.player_report == "Tatum, Jayson"].iloc[0]
    assert tatum.status == "Questionable"
    assert tatum.status_tipoff == "Out"
    assert tatum.provenance == ir.PROVENANCE_SUBMISSION
    assert pd.Timestamp(tatum.published_at) < pd.Timestamp(tatum.tipoff_at)


def test_a_clearance_inside_the_submission_window_binds(monkeypatch, tmp_path):
    """Brown went Out (day before) -> Available in the 13:00 submission.

    Mid-day movement IS knowable at the cutoff - that is what the game-day
    submission is for - so the removal must be lifted.
    """
    frame = _resolve(monkeypatch, tmp_path)
    brown = frame[frame.player_report == "Brown, Jaylen"].iloc[0]
    assert brown.status == "Available"


def test_the_resolver_is_reproducible_so_history_and_slate_agree(monkeypatch,
                                                                 tmp_path):
    """Parity by construction: the output depends only on the filings, never
    on when the resolver runs - so the historical frame and a pending slate
    resolve the same game to the same answer."""
    first = _resolve(monkeypatch, tmp_path)
    second = ir.game_day_designations(date(2026, 1, 10), tmp_path)
    pd.testing.assert_frame_equal(first, second)


def test_discovery_finds_the_slate_from_any_same_day_filing(monkeypatch,
                                                            tmp_path):
    archive = _resolver_archive()
    archive.install(monkeypatch)
    slate = ir._discover_slate(date(2026, 1, 10), tmp_path, 72)
    assert slate == {"NYK@BOS": datetime(2026, 1, 10, 19, 0)}


def test_a_day_with_no_filings_has_no_slate_and_costs_a_handful_of_probes(monkeypatch,
                                                                         tmp_path):
    """Off days cost a bounded number of probes: the league files nothing
    with no games, and sweeping every stamp across a two-year backfill
    would be tens of thousands of requests that cannot contribute a row."""
    archive = _FakeArchive(set())
    archive.install(monkeypatch)
    assert ir._discover_slate(date(2026, 1, 10), tmp_path, 72) == {}
    assert len(archive.fetches) == len(ir._SEED_PROBE_HOURS)


def test_no_status_reaches_the_feature_from_after_the_close(monkeypatch,
                                                           tmp_path):
    """A second, independent tripwire on the same guarantee: every row that
    carries a removal status was published in [close, tipoff) - never
    before the close's own filing and never at/after tip-off."""
    frame = _resolve(monkeypatch, tmp_path)
    carrying = frame[frame.status.astype(str).str.lower()
                     .isin({"out", "doubtful", "recovery"})]
    assert len(carrying) >= 1  # Holiday: Out AT the submission, not later
    for row in carrying.itertuples():
        published = pd.Timestamp(row.published_at)
        assert pd.Timestamp(row.cutoff_at) <= published
        assert published < pd.Timestamp(row.tipoff_at)


def test_a_game_absent_from_the_first_post_close_filing_walks_forward(monkeypatch,
                                                                     tmp_path):
    """The first snapshot at/after the close can predate this game's
    inclusion: clubs file up TO the close and the league merges as they
    arrive (a placeholder-only filing parses to no rows at all).  The
    game's submission is the first filing, still strictly before tip-off,
    that actually carries it - not the first filing that exists."""
    records = {
        ir._stamp_path(datetime(2026, 1, 10, 13, 0)): [],  # no rows yet
        ir._stamp_path(datetime(2026, 1, 10, 13, 15)): [
            _record(datetime(2026, 1, 10, 13, 15),
                    "Tatum, Jayson", "Questionable")],
    }
    archive = _FakeArchive({"2026-01-10_12_00PM", "2026-01-10_01_00PM",
                            "2026-01-10_01_15PM"}, records)
    archive.install(monkeypatch)
    frame = ir.game_day_designations(
        date(2026, 1, 10), tmp_path,
        games=[("NYK@BOS", datetime(2026, 1, 10, 19, 0))])
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row.status == "Questionable"
    assert row.published_at == "2026-01-10T13:15:00"
    assert row.provenance == ir.PROVENANCE_SUBMISSION
    assert pd.Timestamp(row.published_at) < pd.Timestamp(row.tipoff_at)


def test_a_game_never_listed_is_reported_uncovered_not_invented(monkeypatch,
                                                                tmp_path,
                                                                caplog):
    """A game the league never lists (a real 2026-05-21/23 ECF anomaly:
    every filing of the day carried only the other conference's game)
    must yield NO rows and a warning - availability is never invented
    from silence, for history or for the slate."""
    other = _record(datetime(2026, 1, 10, 13, 0), "Gilgeous-Alexander, Shai",
                    "Questionable", matchup="BOS@OKC", team="Celtics")
    archive = _FakeArchive({"2026-01-10_12_00PM", "2026-01-10_01_00PM",
                            "2026-01-10_06_15PM"},
                           {ir._stamp_path(datetime(2026, 1, 10, 13, 0)):
                            [other]})
    archive.install(monkeypatch)
    frame = ir.game_day_designations(
        date(2026, 1, 10), tmp_path,
        games=[("NYK@BOS", datetime(2026, 1, 10, 19, 0))])
    assert not len(frame)
    assert "game uncovered" in caplog.text


def test_a_future_game_is_pre_submission_not_uncovered(monkeypatch, tmp_path,
                                                       caplog):
    """2026-10-04: three games on a 2026-10-20 slate were logged WARNING
    "game uncovered" while every probe hit a stamp 16 days in the future.
    A game whose submission window has not closed has nothing to file
    yet: no rows and an INFO - never the archive-failed warning."""
    import logging
    monkeypatch.setattr(ir, "_now", lambda: datetime(2026, 10, 4, 16, 0))
    archive = _FakeArchive(set())
    archive.install(monkeypatch)
    caplog.set_level(logging.INFO)
    frame = ir.game_day_designations(
        date(2026, 10, 20), tmp_path,
        games=[("BOS@DET", datetime(2026, 10, 20, 13, 0)),
               ("PHI@NYK", datetime(2026, 10, 20, 19, 0)),
               ("OKC@SAS", datetime(2026, 10, 20, 22, 0))])
    assert not len(frame)
    assert "game uncovered" not in caplog.text
    assert caplog.text.count("pre-submission") == 3


def test_a_past_game_with_no_filing_keeps_the_uncovered_warning(
        monkeypatch, tmp_path, caplog):
    """The demotion is bounded by the submission window: a past game the
    archive failed (the 2024-01-19 DAL@GSW gap in this run's own log)
    must keep its WARNING - silence about a real coverage hole would be
    worse than the noise it removes."""
    import logging
    monkeypatch.setattr(ir, "_now", lambda: datetime(2026, 10, 4, 16, 0))
    archive = _FakeArchive(set())
    archive.install(monkeypatch)
    caplog.set_level(logging.INFO)
    frame = ir.game_day_designations(
        date(2026, 1, 10), tmp_path,
        games=[("NYK@BOS", datetime(2026, 1, 10, 19, 0))])
    assert not len(frame)
    warned = [r for r in caplog.records if r.levelno >= logging.WARNING
              and "game uncovered" in r.getMessage()]
    assert warned, "a past uncovered game must still warn"


def test_rows_that_vanish_after_the_close_fall_back_to_the_last_before(monkeypatch,
                                                                     tmp_path):
    """2025-01-11: the league's morning filings carried HOU@ATL (and
    CHA@LAC, SAS@LAL) with player rows, then every filing after the
    window close dropped them.  The submission band carries nothing, so
    the game reads from the last filing STRICTLY BEFORE the close -
    older information, honestly flagged, never an invented "healthy"."""
    morning = datetime(2026, 1, 10, 12, 0)   # pre-close, has the game
    after_close = datetime(2026, 1, 10, 13, 0)  # close: exists, no game
    late = datetime(2026, 1, 10, 18, 15)     # exists, no game either
    records = {
        ir._stamp_path(morning): [
            _record(morning, "Tatum, Jayson", "Out")],
        ir._stamp_path(after_close): [
            _record(after_close, "Someone, Else", "Out",
                    matchup="MIA@NYK")],
        ir._stamp_path(late): [
            _record(late, "Someone, Else", "Out", matchup="MIA@NYK")],
    }
    archive = _FakeArchive({ir.report_stamp(s)
                             for s in (morning, after_close, late)}, records)
    archive.install(monkeypatch)
    frame = ir.game_day_designations(
        date(2026, 1, 10), tmp_path,
        games=[("NYK@BOS", datetime(2026, 1, 10, 19, 0))])
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row.status == "Out"
    assert row.provenance == ir.PROVENANCE_PRE_WINDOW
    assert row.published_at == "2026-01-10T12:00:00"
    assert pd.Timestamp(row.published_at) < pd.Timestamp(row.cutoff_at)
    assert pd.Timestamp(row.published_at) < pd.Timestamp(row.tipoff_at)


def test_require_pdf_parser_is_a_hard_error_when_the_parser_is_gone(
        monkeypatch):
    """2026-10-04: a runner without pdfplumber turned every filing into an
    ``unparseable`` warning and let the run report ok anyway; the check
    must raise, so an entrypoint can refuse to start without it. The
    pinned-install repair runs first - when it cannot help, the error
    still fires and says why."""
    import importlib.util
    attempts = []
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(
        ir, "_install_pinned_parser",
        lambda: attempts.append("tried") or "simulated: no network")
    with pytest.raises(RuntimeError, match="pdfplumber") as err:
        ir.require_pdf_parser()
    assert attempts == ["tried"], "the pinned install must be attempted"
    assert "automatic install failed" in str(err.value)


def test_require_pdf_parser_repairs_a_stale_runner_then_passes(monkeypatch):
    """2026-10-04 14:49: the Kaggle notebook copy predated the fix, so the
    environment had no pdfplumber and the guard went red at second zero
    with nothing shipped. The pinned install must run once and turn that
    stale runner green instead of only failing."""
    import importlib.util
    state = {"installed": False}
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name: object() if state["installed"] else None)

    def _repair():
        state["installed"] = True
        return ""

    monkeypatch.setattr(ir, "_install_pinned_parser", _repair)
    ir.require_pdf_parser()  # must not raise: the repair took
    assert state["installed"], "the pinned install must have been tried"


def test_require_pdf_parser_passes_where_the_parser_is_installed():
    # The suite parses real filings, so this environment has it; this is
    # the healthy path of the guard above.
    ir.require_pdf_parser()
