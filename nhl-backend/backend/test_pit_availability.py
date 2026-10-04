"""Falsifiability tests for the PIT pre-game availability channel.

Every test here is written to FAIL if a specific leakage mode regresses:

* a 2019 article must be refused for a 2025 game (identity, not timing);
* an article published at/after puck drop must be refused (late scratches);
* a single-game scratch must bind that game and NOT the player's return game;
* team attribution must not collapse both sides into one bucket;
* a game with no pre-game evidence must read as a GAP, never as "healthy";
* the emitted stints must flow through the slate's own engine unchanged.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from backend import pit_availability as pa
from backend import injury_stints


def _game(date_str, away="Utah Mammoth", home="New York Rangers",
          game_id="G1"):
    return {
        "game_id": game_id,
        "away_name": away,
        "home_name": home,
        "start": f"{date_str}T22:00:00Z",
    }


ARTICLE_TWO_TEAMS = """
<html><body>
<div>Mammoth projected lineup</div>
<div>Clayton Keller -- Nick Schmaltz</div>
<div>Scratched: Liam O'Brien, Kailer Yamamoto</div>
<div>Injured: Maveric Lamoureux (shoulder)</div>
<div>Rangers projected lineup</div>
<div>Gabe Perreault -- Mika Zibanejad</div>
<div>Scratched: Matt Rempe</div>
<div>Injured: Joonas Korpisalo (lower body)</div>
</body></html>
"""


class TestIdentityNotJustTiming:
    def test_a_2019_article_is_refused_for_a_2025_game(self):
        """The archive returns every preview for a matchup across seasons.

        A 2019 article published years BEFORE the 2025 puck drop has a huge
        positive lead time, so a timing-only gate would ACCEPT it and attach
        a different game's absences. Identity must refuse it.
        """
        game = _game("2025-01-02")
        pub_2019 = datetime(2019, 12, 23, 19, 17, 49, tzinfo=timezone.utc)
        rep = pa.parse_pregame_article(ARTICLE_TWO_TEAMS, game,
                                       "http://archive/x", pub_2019)
        # lead time is ~5 years (positive!) yet it must not bind.
        assert rep.lead_time_minutes > 0
        assert rep.coverage == "NO_EVIDENCE"
        assert rep.absences == []
        assert any("identity" in n for n in rep.notes)

    def test_matching_date_binds(self):
        game = _game("2025-01-02")
        pub = datetime(2025, 1, 2, 19, 21, 31, tzinfo=timezone.utc)
        rep = pa.parse_pregame_article(ARTICLE_TWO_TEAMS, game,
                                       "http://x", pub)
        assert rep.coverage == "PREVIEW"
        assert len(rep.absences) == 5


class TestLateScratchesNeverBind:
    @pytest.mark.parametrize("pub_iso", [
        "2025-01-02T22:00:00Z",   # exactly at puck drop
        "2025-01-02T22:05:00Z",   # after puck drop
    ])
    def test_published_at_or_after_puck_drop_is_refused(self, pub_iso):
        game = _game("2025-01-02")
        pub = datetime.fromisoformat(pub_iso.replace("Z", "+00:00"))
        rep = pa.parse_pregame_article(ARTICLE_TWO_TEAMS, game,
                                       "http://x", pub)
        assert rep.coverage == "NO_EVIDENCE"
        assert rep.absences == []

    def test_published_before_puck_drop_binds(self):
        game = _game("2025-01-02")
        pub = datetime(2025, 1, 2, 13, 24, 0, tzinfo=timezone.utc)
        rep = pa.parse_pregame_article(ARTICLE_TWO_TEAMS, game,
                                       "http://x", pub)
        assert rep.lead_time_minutes > 0
        assert len(rep.absences) == 5


class TestGameScopedNoReturnGameOverreach:
    def test_scratch_binds_this_game_but_not_the_next(self):
        """A healthy scratch must not exclude the player's return game.

        The interval is [puck_drop - 8h, puck_drop + 1s]; the return game a
        day later sits strictly after the end, so it must not be covered.
        """
        game = _game("2025-01-02", game_id="G1")
        pub = datetime(2025, 1, 2, 13, 24, 0, tzinfo=timezone.utc)
        rep = pa.parse_pregame_article(ARTICLE_TWO_TEAMS, game,
                                       "http://x", pub)
        events = pa.build_pregame_events([rep])
        assert len(events) == 5
        stints = pa.build_pregame_stints(events)

        # resolve the player the same way the engine does
        name_col = "player_name"
        key = injury_stints.normalise_player_name("Matt Rempe")
        # identity resolution maps name -> rating id when ratings exist; here
        # the unresolved key is the raw name.
        assert injury_stints.is_unavailable(
            stints, "Matt Rempe",
            datetime(2025, 1, 2, 22, 0, 0, tzinfo=timezone.utc),
            strict_start=True) or injury_stints.is_unavailable(
            stints, key,
            datetime(2025, 1, 2, 22, 0, 0, tzinfo=timezone.utc),
            strict_start=True), "scratch must bind its own game"

        # The RETURN game the next day must NOT be covered.
        assert not injury_stints.is_unavailable(
            stints, "Matt Rempe",
            datetime(2025, 1, 3, 22, 0, 0, tzinfo=timezone.utc),
            strict_start=True), "scratch must not exclude the return game"


class TestTeamAttribution:
    def test_both_teams_are_distinguished(self):
        game = _game("2025-01-02")
        pub = datetime(2025, 1, 2, 13, 24, 0, tzinfo=timezone.utc)
        rep = pa.parse_pregame_article(ARTICLE_TWO_TEAMS, game,
                                       "http://x", pub)
        by_name = {a.player_name: a for a in rep.absences}
        assert by_name["Liam O'Brien"].team != by_name["Matt Rempe"].team
        assert "Mammoth" in by_name["Liam O'Brien"].team
        assert "Rangers" in by_name["Matt Rempe"].team

    def test_body_part_detail_is_captured(self):
        game = _game("2025-01-02")
        pub = datetime(2025, 1, 2, 13, 24, 0, tzinfo=timezone.utc)
        rep = pa.parse_pregame_article(ARTICLE_TWO_TEAMS, game,
                                       "http://x", pub)
        by_name = {a.player_name: a for a in rep.absences}
        assert by_name["Maveric Lamoureux"].detail == "shoulder"
        assert by_name["Joonas Korpisalo"].detail == "lower body"

    def test_suspended_category_is_parsed(self):
        html = ("Suspended: Connor Hellebuyck")
        game = _game("2025-01-02")
        pub = datetime(2025, 1, 2, 13, 24, 0, tzinfo=timezone.utc)
        rep = pa.parse_pregame_article(html, game, "http://x", pub)
        cats = {a.category for a in rep.absences}
        assert "suspended" in cats


class TestCoverageGate:
    def test_no_article_is_a_gap_not_healthy(self):
        rep = pa.PregameReport(
            game_id="G9", puck_drop_utc=datetime(2025, 1, 2, 22, 0,
                                                 tzinfo=timezone.utc),
            source_url="", date_published_utc=None, lead_time_minutes=None)
        cov = pa.coverage_report([rep])
        assert cov.loc[0, "coverage"] == "NO_EVIDENCE"
        assert cov.loc[0, "coverage"] != "HEALTHY"

    def test_preview_with_no_absences_is_labelled_not_gap(self):
        game = _game("2025-01-02")
        pub = datetime(2025, 1, 2, 13, 24, 0, tzinfo=timezone.utc)
        rep = pa.parse_pregame_article("<html><body>nothing</body></html>",
                                       game, "http://x", pub)
        assert rep.coverage == "PREVIEW_NO_ABSENCES"


class TestRealArticleLayout:
    def test_names_on_following_line_are_captured(self):
        """NHL renders ``Scratched:`` and the names on separate lines.

        A parser that only reads inline payloads silently reports
        "no absences" for every real article -- the exact false-clean this
        whole channel exists to prevent.
        """
        html = """
        <html><body>
        <div>Mammoth projected lineup</div>
        <div>Scratched:</div>
        <div>Liam O'Brien, Kailer Yamamoto</div>
        <div>Injured:</div>
        <div>Maveric Lamoureux (shoulder)</div>
        <div>Rangers projected lineup</div>
        <div>Scratched:</div>
        <div>Matt Rempe</div>
        </body></html>
        """
        game = _game("2025-01-02")
        pub = datetime(2025, 1, 2, 13, 24, 0, tzinfo=timezone.utc)
        rep = pa.parse_pregame_article(html, game, "http://x", pub)
        names = {a.player_name for a in rep.absences}
        assert names == {"Liam O'Brien", "Kailer Yamamoto",
                         "Maveric Lamoureux", "Matt Rempe"}
        by_name = {a.player_name: a for a in rep.absences}
        assert by_name["Maveric Lamoureux"].detail == "shoulder"
        assert "Mammoth" in by_name["Liam O'Brien"].team
        assert "Rangers" in by_name["Matt Rempe"].team


class TestParserDoesNotDrainThePage:
    def test_sidebar_and_footer_are_not_absences(self):
        """A 'Scratched:' list must stop at the paragraph break.

        Skipping blank lines drains the sidebar ('Latest News' headlines) and
        the footer into bogus absences, which would exclude random players
        from the pool -- a wrong exclusion is worse than no exclusion.
        """
        html = """
        <html><body>
        <div>Mammoth projected lineup</div>
        <div>Scratched:</div>
        <div>Liam O'Brien, Kailer Yamamoto</div>

        <div>Injured:</div>
        <div>Maveric Lamoureux (shoulder)</div>

        <div>Latest News</div>
        <div>Canucks rally in third period</div>
        <div>Red Wings donate Zamboni to Michigan rink</div>
        <div>Copyright Policy Your Privacy Choices Careers</div>
        </body></html>
        """
        game = _game("2025-01-02")
        pub = datetime(2025, 1, 2, 13, 24, 0, tzinfo=timezone.utc)
        rep = pa.parse_pregame_article(html, game, "http://x", pub)
        names = {a.player_name for a in rep.absences}
        assert names == {"Liam O'Brien", "Kailer Yamamoto",
                         "Maveric Lamoureux"}
        assert "Canucks rally" not in names
        assert "Red Wings donate Zamboni to Michigan rink" not in names


class TestPipelineScheduleCompat:
    def test_abbrev_and_start_time_utc_resolve_to_a_slug(self):
        """The pipeline's schedule rows are abbreviations + ``start_time_utc``.

        A slug builder that only knows full names returns no candidates for
        those rows, so the backfill silently finds nothing and the repull
        reproduces the exact gap it was meant to close.
        """
        row = {"game_id": "2026020036", "away_team": "UTA", "home_team": "NYR",
               "start_time_utc": "2026-10-04T22:00:00Z"}
        assert pa.team_slug("UTA") == "utah-mammoth"
        cands = pa.preview_slug_candidates(row)
        assert cands[0].endswith(
            "utah-mammoth-new-york-rangers-game-preview-october-4-2026")

    def test_full_name_rows_still_resolve(self):
        row = {"away_name": "Utah Mammoth", "home_name": "New York Rangers",
               "start": "2026-10-04T22:00:00Z"}
        assert pa.preview_slug_candidates(row)[0].endswith(
            "utah-mammoth-new-york-rangers-game-preview-october-4-2026")


class TestCoverageGateFailsLoudly:
    def _rep(self, gid, coverage):
        return pa.PregameReport(
            game_id=gid,
            puck_drop_utc=datetime(2025, 1, 2, 22, 0, tzinfo=timezone.utc),
            source_url="http://x" if coverage != "NO_EVIDENCE" else "",
            date_published_utc=(datetime(2025, 1, 2, 13, 0, tzinfo=timezone.utc)
                                if coverage != "NO_EVIDENCE" else None),
            lead_time_minutes=540.0 if coverage != "NO_EVIDENCE" else None,
            coverage=coverage,
        )

    def test_uncovered_game_reads_as_gap_not_healthy(self, tmp_path, monkeypatch):
        reports = [self._rep("G1", "PREVIEW"),
                   self._rep("G2", "NO_EVIDENCE")]
        st = pa.coverage_gate(reports, write_status=False)
        assert st["by_label"].get("NO_EVIDENCE") == 1
        assert st["uncovered_game_ids"] == ["G2"]
        # the loud signal: coverage fraction reflects the gap
        assert st["games_with_evidence"] == 1
        assert st["coverage_fraction"] == 0.5

    def test_thin_coverage_warns_but_does_not_abort(self, caplog):
        """The gate FAILS OPEN: a repull must complete. Loud, not fatal."""
        import logging
        reports = [self._rep("G1", "PREVIEW")] + [
            self._rep(f"G{i}", "NO_EVIDENCE") for i in range(2, 10)]
        with caplog.at_level(logging.WARNING):
            st = pa.coverage_gate(reports, write_status=False)
        assert st["ok"] is False              # flagged NOT ok...
        assert st["coverage_fraction"] < 0.60
        # ...but it returns normally (no exception) and logs loudly.
        assert any("THIN" in r.message for r in caplog.records)
        assert len(st["uncovered_game_ids"]) == 8

    def test_good_coverage_is_ok_and_quiet(self, caplog):
        import logging
        reports = [self._rep(f"G{i}", "PREVIEW") for i in range(10)]
        with caplog.at_level(logging.WARNING):
            st = pa.coverage_gate(reports, write_status=False)
        assert st["ok"] is True
        assert not any("THIN" in r.message for r in caplog.records)

    def test_zero_games_is_ok_not_a_crash(self):
        assert pa.coverage_gate([], write_status=False)["ok"] is True

    def test_status_artifact_is_written(self, tmp_path, monkeypatch):
        import json
        from backend import config as _cfg
        # The gate writes via ``from . import config`` (backend.config), so
        # patch THAT module's delivery dir, not a top-level 'config'.
        monkeypatch.setattr(_cfg, "DATA_DELIVERY_DIR", tmp_path)
        reports = [self._rep("G1", "PREVIEW"), self._rep("G2", "NO_EVIDENCE")]
        pa.coverage_gate(reports, write_status=True)
        path = tmp_path / pa.COVERAGE_STATUS_ARTIFACT
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["uncovered_game_ids"] == ["G2"]


class _FakeResp:
    def __init__(self, status_code=200, text="", headers=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}


class _FakeSession:
    """Records fetches; serves scripted responses. No network."""
    def __init__(self, responses):
        self.responses = list(responses)  # list of _FakeResp, popped in order
        self.calls = []

    def get(self, url, timeout=None, allow_redirects=None, **kw):
        self.calls.append(url)
        return self.responses.pop(0) if self.responses else _FakeResp(404)


class TestDailyFaceoffSource:
    def test_designations_map_to_availability(self):
        assert pa.df_availability("OUT") == pa.DF_OUT
        assert pa.df_availability("IR") == pa.DF_OUT
        assert pa.df_availability("DTD") == pa.DF_OUT
        assert pa.df_availability("GTD") == pa.DF_UNKNOWN
        assert pa.df_availability(None) == pa.DF_UNKNOWN
        assert pa.df_availability("Healthy") == pa.DF_AVAILABLE

    def test_parses_players_and_status_badges(self):
        html = '''
        <div class="card"><img alt="Mika Zibanejad"/><span class="bg-red-500">IR</span></div>
        <div class="card"><img alt="Artemi Panarin"/></div>
        <div class="card"><img alt="Adam Fox"/><span class="bg-red-500">GTD</span></div>
        <img alt="Daily Faceoff"/>
        '''
        rows = pa.parse_dailyfaceoff(html, team="NYR")
        by_name = {r["player_name"]: r for r in rows}
        assert "Mika Zibanejad" in by_name
        assert by_name["Mika Zibanejad"]["availability"] == pa.DF_OUT
        # a player with no badge is in the lineup -> available, never excluded
        assert by_name["Artemi Panarin"]["availability"] == pa.DF_AVAILABLE
        assert by_name["Adam Fox"]["availability"] == pa.DF_UNKNOWN
        # site/team chrome is not a player
        assert "Daily Faceoff" not in by_name


class TestRateLimitBackoff:
    def test_retries_429_then_succeeds(self):
        import logging
        sess = _FakeSession([_FakeResp(429), _FakeResp(200, "ok")])
        out = pa.fetch_with_backoff(sess, "http://x", base_delay=0.01)
        assert out == "ok"
        assert len(sess.calls) == 2

    def test_gives_up_after_max_retries(self):
        sess = _FakeSession([_FakeResp(429)] * 5)
        out = pa.fetch_with_backoff(sess, "http://x", max_retries=2, base_delay=0.01)
        assert out is None
        assert len(sess.calls) == 2

    def test_hard_404_returns_none_without_retry(self):
        sess = _FakeSession([_FakeResp(404)])
        assert pa.fetch_with_backoff(sess, "http://x", base_delay=0.01) is None
        assert len(sess.calls) == 1


class TestResumableBackfill:
    def test_backfill_skips_games_already_frozen(self, tmp_path, monkeypatch):
        from backend import config as _cfg
        monkeypatch.setattr(_cfg, "DATA_DELIVERY_DIR", tmp_path)
        # Freeze one game's events into the artifact first.
        frozen = pd.DataFrame([{
            "player_name": "Matt Rempe", "player_id": None, "team": "Rangers",
            "announced_at_utc": pd.Timestamp("2025-01-02 13:00:00"),
            "returned_at_utc": pd.Timestamp("2025-01-02 22:00:01"),
            "game_id": "G1", "category": "scratched", "detail": "undisclosed",
            "source_url": "http://x",
        }])
        frozen.to_parquet(tmp_path / pa.PREGAME_AVAILABILITY_ARTIFACT, index=False)

        sess = _FakeSession([_FakeResp(200, "<html></html>")])
        games = [{"game_id": "G1", "away_team": "NYR", "home_team": "BOS",
                  "start_time_utc": "2025-01-02T22:00:00Z"},
                 {"game_id": "G2", "away_team": "NYR", "home_team": "BOS",
                  "start_time_utc": "2025-01-03T22:00:00Z"}]
        pa.backfill(games, sess, write_artifact=True, resume=True)
        # G1 (january-2) is frozen -> never re-fetched; only G2 (january-3)
        # hits the network (both slug candidates). Resume holds.
        assert not any("january-2" in c for c in sess.calls), \
            "a frozen game must never be re-crawled"
        assert any("january-3" in c for c in sess.calls), \
            "an uncovered game must be fetched"

    def test_no_resume_refetches_everything(self, tmp_path, monkeypatch):
        from backend import config as _cfg
        monkeypatch.setattr(_cfg, "DATA_DELIVERY_DIR", tmp_path)
        sess = _FakeSession([_FakeResp(404)] * 10)
        games = [{"game_id": "G1", "away_team": "NYR", "home_team": "BOS",
                  "start_time_utc": "2025-01-02T22:00:00Z"}]
        pa.backfill(games, sess, write_artifact=False, resume=False)
        assert len(sess.calls) == 2  # both slug candidates tried


class TestReproducibility:
    def test_replay_is_deterministic(self):
        """Same frozen artifact -> identical stints every run."""
        game = _game("2025-01-02")
        pub = datetime(2025, 1, 2, 13, 24, 0, tzinfo=timezone.utc)
        rep = pa.parse_pregame_article(ARTICLE_TWO_TEAMS, game, "http://x", pub)
        ev = pa.build_pregame_events([rep])
        s1 = pa.build_pregame_stints(ev.copy())
        s2 = pa.build_pregame_stints(ev.copy())
        pd.testing.assert_frame_equal(
            s1[["player_id", "stint_start", "stint_end"]].reset_index(drop=True),
            s2[["player_id", "stint_start", "stint_end"]].reset_index(drop=True))

    def test_load_path_makes_no_network_calls(self):
        """The replay path reads parquet only; a re-run must never crawl."""
        import inspect
        src = inspect.getsource(pa.load_pregame_stints)
        assert "requests" not in src and "fetch_preview" not in src
        src2 = inspect.getsource(pa.load_pregame_events)
        assert "requests" not in src2 and "fetch_preview" not in src2

    def test_boxscore_signal_is_deterministic(self):
        games = pd.DataFrame([
            {"game_id": "G1", "game_date": "2025-01-02", "team": "NYR"},
            {"game_id": "G2", "game_date": "2025-01-03", "team": "NYR"},
        ])
        app = pd.DataFrame([
            {"game_id": "G1", "game_date": "2025-01-02", "team": "NYR",
             "player_id": "P1"},
        ])
        r1 = pa.boxscore_absence_signal(app, games)
        r2 = pa.boxscore_absence_signal(app, games)
        pd.testing.assert_frame_equal(r1, r2)


class TestBoxscorePITDiscipline:
    def test_absence_shifts_forward_never_onto_revealing_game(self):
        """A boxscore reveals its own game post-hoc; it must never flag that
        game, only the NEXT one. Otherwise a late scratch leaks into the very
        game it was unknowable for."""
        games = pd.DataFrame([
            {"game_id": "G1", "game_date": "2025-01-02", "team": "NYR"},
            {"game_id": "G2", "game_date": "2025-01-03", "team": "NYR"},
        ])
        # P1 dressed in G1's boxscore; P2 did not.
        app = pd.DataFrame([
            {"game_id": "G1", "game_date": "2025-01-02", "team": "NYR",
             "player_id": "P1"},
        ])
        sig = pa.boxscore_absence_signal(app, games)
        # The signal is about G2 (shifted forward), never G1.
        assert set(sig["game_id"]) <= {"G2"}
        assert "G1" not in set(sig["game_id"])

    def test_no_boxscore_side_produces_no_absence_claim(self):
        """A side with no boxscore must be unknown, not 'everyone absent'."""
        games = pd.DataFrame([
            {"game_id": "G1", "game_date": "2025-01-02", "team": "NYR"},
            {"game_id": "G2", "game_date": "2025-01-03", "team": "NYR"},
        ])
        app = pd.DataFrame(columns=["game_id", "game_date", "team", "player_id"])
        sig = pa.boxscore_absence_signal(app, games)
        assert len(sig) == 0


class TestBoxscoreAudit:
    def test_returns_appearance_contradiction(self):
        """A pre-game 'out' who then appears is a review item (a return the
        preview could not see), surfaced by the reproducible witness."""
        events = pd.DataFrame([
            {"game_id": "G1", "player_name": "Matt Rempe"},
            {"game_id": "G1", "player_name": "Liam O'Brien"},
        ])
        app = pd.DataFrame([
            {"game_id": "G1", "player_name": "Matt Rempe"},  # appeared!
        ])
        rep = pa.verify_pregame_vs_boxscores(events, app)
        by_name = {r.player_name: r for r in rep.itertuples(index=False)}
        assert by_name["Matt Rempe"].appeared is True
        assert by_name["Liam O'Brien"].appeared is False


class TestStructuralIdentity:
    def test_stints_use_the_slate_engine_contract(self):
        game = _game("2025-01-02")
        pub = datetime(2025, 1, 2, 13, 24, 0, tzinfo=timezone.utc)
        rep = pa.parse_pregame_article(ARTICLE_TWO_TEAMS, game,
                                       "http://x", pub)
        events = pa.build_pregame_events([rep])
        stints = pa.build_pregame_stints(events)
        assert injury_stints.OUT_PLAYER in stints.columns
        assert injury_stints.STINT_START in stints.columns
        assert injury_stints.STINT_END in stints.columns
        assert stints.attrs.get("snapshot_based") is True
        # every interval is game-scoped and closes itself
        assert stints[injury_stints.STINT_END].notna().all()

    def test_events_are_strictly_pre_puck_drop(self):
        game = _game("2025-01-02")
        puck = datetime(2025, 1, 2, 22, 0, 0, tzinfo=timezone.utc)
        pub = datetime(2025, 1, 2, 13, 24, 0, tzinfo=timezone.utc)
        rep = pa.parse_pregame_article(ARTICLE_TWO_TEAMS, game,
                                       "http://x", pub)
        events = pa.build_pregame_events([rep])
        for ann in events["announced_at_utc"]:
            assert pd.Timestamp(ann) < puck
        for end in events["returned_at_utc"]:
            assert pd.Timestamp(end) > puck


class TestTkachukPressureTest:
    """Pressure test: Matthew Tkachuk, Florida, 2025-26.

    The real timeline (82 games): 47 out (sports-hernia/adductor surgery, Aug
    22 2025), then RETURN on Jan 19 2026 vs SJ, play 30 games, then 5 out
    (paternity — a NON-MEDICAL absence) starting Apr 7 2026, then return for
    the playoffs. 47 + 30 + 5 = 82.

    This exercises BOTH absence channels and the return. The single most
    important assertion is game 48 (the Jan 19 debut): the resolver must mark
    him IN via the pre-game source, where a bare carry-forward marks him OUT.
    That is the return-gap regression guard.
    """

    @staticmethod
    def _timeline():
        dates = pd.date_range("2025-10-09", periods=82, freq="D")
        games = pd.DataFrame({
            "game_id": [f"G{i:03d}" for i in range(1, 83)],
            "game_date": dates,
            "team": ["FLA"] * 82,
        })
        # Tkachuk DRESSED in games 48..77 inclusive = 30 games.
        appearances = pd.DataFrame({
            "game_id": [f"G{i:03d}" for i in range(48, 78)],
            "game_date": dates[47:77],
            "team": ["FLA"] * 30,
            "player_id": ["tkachuk"] * 30,
        })
        # Pre-game source naming the three events carry-forward cannot get right:
        #   G001 -> out (announced pre-season surgery)
        #   G048 -> IN  (return / debut)
        #   G078 -> out (paternity, a scratch on the Apr 7 card)
        pre_game = pd.DataFrame({
            "game_id": ["G001", "G048", "G078"],
            "player_id": ["tkachuk"] * 3,
            "team": ["FLA"] * 3,
            "status": ["out", "in", "out"],
        })
        return games, appearances, pre_game

    def test_return_game_is_marked_in_via_pre_game_source(self):
        """Game 48 (Jan 19 debut): must be IN, not a carried-forward false OUT."""
        games, appearances, pre_game = self._timeline()
        res = pa.resolve_availability(pre_game, appearances, games)
        g48 = res[res["game_id"] == "G048"].iloc[0]
        assert g48["availability"] == pa.AV_IN
        assert g48["source"] == pa.SRC_PRE_GAME

    def test_without_pre_game_the_return_would_be_falsely_excluded(self):
        """Guards the resolver is load-bearing: bare carry-forward marks G48 OUT."""
        games, appearances, pre_game = self._timeline()
        res = pa.resolve_availability(None, appearances, games)
        g48 = res[res["game_id"] == "G048"].iloc[0]
        assert g48["availability"] == pa.AV_OUT   # the false exclusion
        assert g48["source"] == pa.SRC_CARRY_FORWARD

    def test_injury_absence_games_are_out(self):
        """Games 1-47 (surgery): all OUT — G1 via pre-game, G2-47 via carry-forward."""
        games, appearances, pre_game = self._timeline()
        res = pa.resolve_availability(pre_game, appearances, games)
        first = res[res["game_id"] == "G001"].iloc[0]
        assert first["availability"] == pa.AV_OUT
        assert first["source"] == pa.SRC_PRE_GAME   # no in-season G-1 -> pre-game carries it
        mid = res[res["game_id"] == "G025"].iloc[0]
        assert mid["availability"] == pa.AV_OUT
        assert mid["source"] == pa.SRC_CARRY_FORWARD
        assert res[res["game_id"] == "G047"].iloc[0]["availability"] == pa.AV_OUT

    def test_non_medical_paternity_absence_is_out(self):
        """Games 78-82 (paternity, NON-medical): OUT — G78 via pre-game scratch."""
        games, appearances, pre_game = self._timeline()
        res = pa.resolve_availability(pre_game, appearances, games)
        onset = res[res["game_id"] == "G078"].iloc[0]
        assert onset["availability"] == pa.AV_OUT
        assert onset["source"] == pa.SRC_PRE_GAME   # the Apr 7 card named the scratch
        tail = res[res["game_id"] == "G082"].iloc[0]
        assert tail["availability"] == pa.AV_OUT
        assert tail["source"] == pa.SRC_CARRY_FORWARD

    def test_new_absence_without_pre_game_does_not_leak_as_out(self):
        """A fresh absence (played last game, absent now) must NOT leak as OUT.

        With no pre-game card the forward-shift sees he played G077 -> expected
        available (IN). The paternity absence is only captured where the pre-game
        card names it; where it is missing, the resolver errs toward available,
        which is the PIT-correct side (a late scratch is unknowable pre-game).
        """
        games, appearances, pre_game = self._timeline()
        res = pa.resolve_availability(None, appearances, games)
        onset = res[res["game_id"] == "G078"].iloc[0]
        assert onset["availability"] == pa.AV_IN       # NOT out -> no leakage
        assert onset["source"] == pa.SRC_APPEARANCE    # carried from G077

    def test_played_streak_is_in(self):
        """Games 49-77 (the 30-game span he played): IN via appearance."""
        games, appearances, pre_game = self._timeline()
        res = pa.resolve_availability(pre_game, appearances, games)
        span = res[res["game_id"].isin([f"G{i:03d}" for i in range(49, 78)])]
        assert set(span["availability"]) == {pa.AV_IN}
        assert set(span["source"]) == {pa.SRC_APPEARANCE}

    def test_total_out_games_match_the_real_52(self):
        """47 injury + 5 personal = 52 absences; 30 appearances. Matches reality."""
        games, appearances, pre_game = self._timeline()
        res = pa.resolve_availability(pre_game, appearances, games)
        out = res[res["availability"] == pa.AV_OUT]
        ins = res[res["availability"] == pa.AV_IN]
        assert len(out) == 52   # 47 + 5
        assert len(ins) == 30
        assert len(res) == 82
        # with the pre-game card present, every game resolves (no unknowns)
        assert res["availability"].isin([pa.AV_UNKNOWN]).sum() == 0

    def test_return_gap_is_closed_only_where_pre_game_reaches(self):
        """The return is IN with the pre-game source; OUT without it."""
        games, appearances, pre_game = self._timeline()
        with_pre = pa.resolve_availability(pre_game, appearances, games)
        without = pa.resolve_availability(None, appearances, games)
        assert with_pre[with_pre["game_id"] == "G048"].iloc[0]["availability"] == pa.AV_IN
        assert without[without["game_id"] == "G048"].iloc[0]["availability"] == pa.AV_OUT

    def test_deterministic_across_runs(self):
        """Same inputs -> identical output. Freeze-and-replay reproducibility."""
        games, appearances, pre_game = self._timeline()
        r1 = pa.resolve_availability(pre_game, appearances, games).sort_values(
            ["game_id", "player_id"]).reset_index(drop=True)
        r2 = pa.resolve_availability(pre_game, appearances, games).sort_values(
            ["game_id", "player_id"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(r1, r2)
