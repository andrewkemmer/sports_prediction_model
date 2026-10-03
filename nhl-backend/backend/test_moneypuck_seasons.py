"""The MoneyPuck season set must be derived from the calendar, not listed.

Regression for the 2026-10-03 incident: ``config.PLAYER_RATING_SEASONS`` was a
literal ending at 2025, and ``load_moneypuck_player_games`` carried its own
literal ``defaults = [2021..2025]``. Both stopped at 2025 while the calendar
said 2026, so the 2026-27 archive was never fetched. Training still graded 13
real 2026-27 games (those come from NHL-API boxscores), but every ``pl_*``
column was rated from 2025-26 rows or from a position prior — so Kane, traded
to Chicago over the summer, was invisible to the Chicago pool while the model
was scoring Chicago's games.

Three separate guards, because the incident had three causes:

1. The season set is a function of the date and always reaches the season in
   progress. An end-year literal rots the moment the season rolls over.
2. ``current_nhl_season`` agrees with ``player_ratings._season_ids`` — two
   season conventions that disagree would put the rating on one side of a
   boundary and the pool on the other.
3. The loader's own default derives from config, so no second literal can
   reappear behind the caller's back.

Plus the cache rule the fix introduced: a finished season's archive is
immutable and caches forever, but the live season gains games every game day,
so serving its cached copy would freeze ratings at the day it was written —
silently, with no error. That is tested by asserting the live season never
reads the cache when a fetch is possible.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # noqa: E402
import ingestion as ing  # noqa: E402
import player_ratings as pr  # noqa: E402


class TestSeasonDerivation:
    def test_no_hardcoded_season_literal_survives(self):
        """The old gate must be gone, not merely extended."""
        assert not hasattr(config, "PLAYER_RATING_SEASONS"), (
            "PLAYER_RATING_SEASONS is a season gate; use player_rating_seasons()")

    def test_reaches_the_season_in_progress(self):
        """A season set that stops short of today is the whole bug."""
        seasons = config.player_rating_seasons(date(2026, 10, 3))
        assert seasons[-1] == 2026
        assert 2026 in seasons

    def test_season_is_derived_not_stored(self):
        """A stored tuple is stale the moment the calendar moves."""
        assert config.player_rating_seasons(date(2027, 10, 5))[-1] == 2027
        assert config.player_rating_seasons(date(2031, 11, 1))[-1] == 2031

    def test_starts_at_the_first_published_archive(self):
        seasons = config.player_rating_seasons(date(2026, 10, 3))
        assert seasons[0] == config.PLAYER_RATING_FIRST_SEASON == 2008

    @pytest.mark.parametrize("day,expected", [
        ("2026-10-03", 2026),   # in season
        ("2026-04-15", 2025),   # late in a season that started 2025
        ("2026-07-01", 2026),   # first day of a new season year
        ("2026-06-30", 2025),   # last day of the old one
        ("2027-01-15", 2026),   # mid-season, January still counts as 2026
    ])
    def test_boundary_matches_player_ratings(self, day, expected):
        """config and player_ratings must share ONE season convention."""
        assert config.current_nhl_season(day) == expected
        scored = int(pr._season_ids(pd.Series([pd.Timestamp(day)])).iloc[0])
        assert config.current_nhl_season(day) == scored, (
            f"season conventions disagree on {day}: config="
            f"{config.current_nhl_season(day)} _season_ids={scored}")

    def test_default_season_list_is_the_derived_one(self):
        """The loader's own fallback must not reintroduce a literal."""
        source = (HERE / "ingestion.py").read_text(encoding="utf-8")
        assert "defaults = list(config.player_rating_seasons())" in source


def _zip_bytes(rows: list[dict]) -> bytes:
    """Build an in-memory ZIP holding one CSV with the required columns."""
    import io
    import zipfile

    frame = pd.DataFrame(rows)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("players.csv", frame.to_csv(index=False))
    return buf.getvalue()


REQUIRED = ["playerId", "name", "gameId", "season", "playerTeam", "gameDate",
            "position", "situation", "icetime", "I_F_xGoals"]


def _row(season: int, day: str) -> dict:
    return {"playerId": "8474141", "name": "Patrick Kane",
            "gameId": f"{season}020001", "season": season, "playerTeam": "CHI",
            "gameDate": day.replace("-", ""), "position": "R",
            "situation": "5on5", "icetime": 600.0, "I_F_xGoals": 0.25}


class TestLiveSeasonCache:
    """The in-progress season must never be served from a stale cache."""

    @pytest.fixture()
    def cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ing, "CACHE_DIR", tmp_path)
        return tmp_path

    def _seed(self, cache_dir: Path, season: int, rows: list[dict]) -> Path:
        path = cache_dir / f"moneypuck_player_games_{ing.MP_PLAYER_GAME_VERSION}_{season}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        return path

    def test_finished_season_is_served_from_cache(self, cache_dir, monkeypatch):
        """Immutable archives should not be re-fetched."""
        rows = [_row(2025, "2025-10-09")]
        self._seed(cache_dir, 2025, rows)

        def explode(*_args, **_kwargs):
            raise AssertionError("a finished season must not re-download")

        monkeypatch.setattr(ing, "_download_moneypuck_player_game_archive", explode)
        out = ing.load_moneypuck_player_games(seasons=[2025], use_cache=True)
        assert out is not None and len(out) == 1
        assert int(out["season"].iloc[0]) == 2025

    def test_live_season_refetches_instead_of_serving_cache(
            self, cache_dir, monkeypatch):
        """Cached live-season rows would freeze ratings at the write date."""
        stale = [_row(2026, "2026-10-01")]
        self._seed(cache_dir, 2026, stale)
        fresh = [_row(2026, "2026-10-01"), _row(2026, "2026-10-02")]
        calls = []

        def download(url, seasons, required):
            calls.append((url, tuple(sorted(seasons))))
            return pd.DataFrame(fresh)[required]

        monkeypatch.setattr(ing, "_download_moneypuck_player_game_archive", download)
        out = ing.load_moneypuck_player_games(seasons=[2026], use_cache=True)
        assert calls, "the live season must re-fetch every run"
        assert len(out) == 2, "the stale cached copy must not win"
        assert pd.to_datetime(out["gameDate"].astype(str),
                              format="%Y%m%d").max() == pd.Timestamp("2026-10-02")

    def test_live_season_fetch_failure_falls_back_to_cache(
            self, cache_dir, monkeypatch):
        """Offline runs keep the last good copy rather than losing the season."""
        cached = [_row(2026, "2026-10-01")]
        self._seed(cache_dir, 2026, cached)

        def download(*_args, **_kwargs):
            raise RuntimeError("network down")

        monkeypatch.setattr(ing, "_download_moneypuck_player_game_archive", download)
        out = ing.load_moneypuck_player_games(seasons=[2026], use_cache=True)
        assert out is not None and len(out) == 1

    def test_unpublished_current_season_degrades_instead_of_dying(
            self, cache_dir, monkeypatch):
        """A season with no archive yet must not take all 24 pl_* columns down.

        Before opening night MoneyPuck simply has no rows for the new season.
        The old code treated that as a hole in history and returned None,
        which drops every pool column to a position prior.
        """
        history = [_row(2025, "2025-10-09")]
        self._seed(cache_dir, 2025, history)

        def download(url, seasons, required):
            if 2026 in seasons:
                raise ValueError("archive has no regular-season skater rows for 2026")
            return pd.DataFrame(history)[required]

        monkeypatch.setattr(ing, "_download_moneypuck_player_game_archive", download)
        out = ing.load_moneypuck_player_games(seasons=[2025, 2026], use_cache=True)
        assert out is not None, "an unpublished season must not null the family"
        assert int(out["season"].iloc[0]) == 2025

    @pytest.mark.parametrize("season", [2022, 2025])
    def test_missing_FINISHED_season_still_fails_loudly(
            self, cache_dir, monkeypatch, season):
        """A hole in completed history is a real failure and must stay one.

        Parametrized over both branches: 2022 comes from the combined
        2008_to_2024 archive, 2025 from a per-season ZIP — the latter is the
        branch the live-season exemption was added to, so it is the one that
        could be over-broadened into swallowing real failures.
        """
        def download(*_args, **_kwargs):
            raise ValueError("archive unavailable")

        monkeypatch.setattr(ing, "_download_moneypuck_player_game_archive", download)
        assert ing.load_moneypuck_player_games(seasons=[season],
                                                use_cache=True) is None

    def test_exemption_is_bounded_by_the_live_year(self):
        """Only the live year may take the unpublished path.

        The loader branches on ``season >= config.current_nhl_season()``. If
        that were ever widened to something like "recent seasons", a broken
        2025 archive would silently degrade to priors instead of failing.
        """
        live = config.current_nhl_season()
        finished = [s for s in config.player_rating_seasons() if s < live]
        assert finished == list(range(config.PLAYER_RATING_FIRST_SEASON, live)), (
            "every season strictly below the live year is finished history")
        assert config.player_rating_seasons()[-1] == live, (
            "the derived set must end at the live year, never short of it")
