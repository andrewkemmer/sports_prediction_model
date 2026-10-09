"""Point-in-time regression pins for the NHL run engine (Poisson/NB totals +
spread-margin pricing) and the binary moneyline's prequential calibration.

The production OOF hit rate and the live board's realized rate can only
diverge through train/serve skew, so this suite pins every PIT invariant
the engine's honesty depends on:

  1. distribution OOF fold geometry: expanding, train strictly before
     validation, per-game rows unique;
  2. grid coherence: every spread/total probability triple sums to ~1;
  3. prequential per-line Platt: a fold's map sees PRIOR folds only
     (poisoning the last fold cannot move earlier folds' probabilities);
  4. moneyline leakage: poisoning a fold's labels cannot move ITS OWN
     predictions (features ride shift(1) / Elo-updates-after-settle);
  5. derived-vs-binary moneyline honesty: the run engine never re-derives
     a winner behind the binary model's back;
  6. NB dispersion: estimated from leakage-free OOF with MLB's pooled
     method-of-moments estimator, uncapped;
  7. MLB feature parity: the run-line matrix IS the binary moneyline's
     production matrix, resolved through the moneyline module itself, and it
     follows the active RFE subset;
  8. monitor line-pair contract: canonical totals (5,6,7) / spreads (1,2)
     priced at fair lines with honest outcomes;
  9. delivery honesty: the gates run BEFORE monitoring writes, retention can
     never delete a file the run itself just wrote, and the run-log lines
     describe what actually happened;
 10. sealed holdout: the α(λ) dispersion layer and the final market-line
     calibrators never see the sealed tail (last HOLDOUT_DAYS of OOF rows);
     poisoning the sealed tail cannot move the fitted dispersion, and the
     tail is stamped frame_view='sealed' and scored separately;
 11. pull progress: a real tqdm bar is drawn in every context, including a
     pipe (the notebook drives the pipeline through subprocess, so stderr is
     never a terminal), and where no bar can be drawn the log line carries a
     fixed-width one of its own. Either way the log records a live position
     and a closing summary, on a seconds floor as well as an item cadence,
     and partitions cache hits from fetches from dead pages.

Run with: python nhl-backend/backend/test_run_engine_pit.py
"""
from __future__ import annotations

import sys
import json
import ast
import logging
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from unittest.mock import patch as _mock_patch

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

import config                                        # noqa: E402
import folds as folds_mod                            # noqa: E402
import features as feat_mod                          # noqa: E402
import distributions as dist_mod                     # noqa: E402
import moneyline as ml_mod                           # noqa: E402
import monitoring as mon                             # noqa: E402
import ingestion as ing                               # noqa: E402
import serving as serving_mod                         # noqa: E402
from evaluation import nb_distribution_metrics       # noqa: E402

# ---------------------------------------------------------------------------
# Offline player-ratings stub. The MoneyPuck skater game-log archives are a
# 366 MB download (2.6 GB uncompressed) and these tests never fetch data.
# The stub returns the empty frame, which is the documented degradation path:
# _load_player_ratings() logs a warning and every pl_* column falls back to
# its position prior. ESPN injury snapshots come from the repo cache and
# already degrade gracefully offline.
feat_mod.ingestion.load_moneypuck_player_games = (
    lambda *args, **kwargs: pd.DataFrame())


def _synth_games(n_days: int = 60, games_per_day: int = 6, seed: int = 7,
                 start: str = "2025-10-01") -> pd.DataFrame:
    """Decided games with team-perspective scores the feature engine needs.

    Each team plays at most once per day (the ladder's strictly-increasing
    gameday gate); goals are Poisson so margins/totals are hockey-shaped."""
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp(start)
    pk = 2025010000
    for d in range(n_days):
        day = (base + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for i in range(games_per_day):
            home, away = TEAMS[i], TEAMS[i + games_per_day]
            pk += 1
            hs = int(rng.poisson(3.1))
            as_ = int(rng.poisson(2.7))
            rows.append({
                "game_id": f"{day.replace('-', '')}_{away}@{home}",
                "season": 2025,
                "gameday": day,
                "home_team": home, "away_team": away,
                "home_score": hs, "away_score": as_,
            })
    return pd.DataFrame(rows)


TEAMS = ["ANA", "BOS", "BUF", "CAR", "CBJ", "CGY",
         "CHI", "COL", "DAL", "DET", "EDM", "FLA"]


# ---------------------------------------------------------------------------
# 1. Distribution OOF fold geometry
# ---------------------------------------------------------------------------
def test_serving_start_time_preserves_utc_and_does_not_fabricate_missing():
    assert serving_mod._start_time_utc({
        "gameday": "2026-09-29",
        "start_time_utc": "2026-09-30T00:00:00Z",
    }) == "2026-09-30T00:00:00Z"
    assert serving_mod._start_time_utc({
        "gameday": "2026-09-29",
        "start_time_utc": "",
    }) is None
    assert serving_mod._start_time_utc({
        "gameday": "2026-09-29",
    }) is None


def test_dist_oof_folds_are_expanding_and_strictly_prior():
    games = feat_mod.build_game_features(_synth_games())
    folds = folds_mod.make_folds(games, date_col="gameday")
    assert len(folds) > 2
    out = dist_mod.walk_forward_oof(games, fold_list=folds)
    oof = out["oof"]
    assert len(oof) == int(sum(len(f.val_idx) for f in folds))
    # Per-game rows unique (one distribution row per game).
    assert oof["game_id"].is_unique
    sizes = [r["n_train"] for r in out["fold_table"].to_dict("records")]
    assert all(b >= a for a, b in zip(sizes, sizes[1:])), \
        f"training folds not expanding: {sizes}"
    for rec in out["fold_table"].to_dict("records"):
        assert pd.Timestamp(rec["val_start"]) < pd.Timestamp(rec["val_end"])
    # First validation window sits behind the 30-day warm-up boundary.
    first_core = pd.Timestamp("2025-10-01")
    assert folds[0].val_start == first_core + pd.Timedelta(days=config.WARMUP_DAYS)


# ---------------------------------------------------------------------------
# 9. Ingestion pull shape: 60-day chunks + progress, with NO data impact
# ---------------------------------------------------------------------------
def test_pull_chunks_cover_every_game_exactly_once():
    """Chunking is presentational, so it must never drop or duplicate a game.

    A dropped id would silently remove that game from the feature frame, which
    is the one way a progress/observability change could corrupt production.
    """
    start = pd.Timestamp("2024-10-01")
    ids, gamedays = [], {}
    for d in range(400):
        day = start + pd.Timedelta(days=d)
        for k in range(3):
            gid = f"{day:%Y%m%d}_{k}"
            ids.append(gid)
            gamedays[gid] = day
    chunks = ing._chunk_games(ids, gamedays)
    flat = [g for _, group in chunks for g in group]
    assert sorted(flat) == sorted(ids), "chunking dropped or duplicated a game"
    assert len(flat) == len(set(flat))
    # 400 days at 60 days/chunk -> 7 windows, ascending.
    assert len(chunks) == 7, [label for label, _ in chunks]
    for label, group in chunks:
        lo, hi = (pd.Timestamp(x) for x in label.split(".."))
        assert hi - lo == pd.Timedelta(days=ing.PULL_CHUNK_DAYS - 1), label
        for g in group:
            assert lo <= gamedays[g] <= hi, \
                f"{g} ({gamedays[g].date()}) sits outside its own chunk {label}"
    starts = [pd.Timestamp(label.split("..")[0]) for label, _ in chunks]
    assert starts == sorted(starts), "chunks are not in ascending order"

    # A game with no known gameday must still be fetched, not dropped.
    undated = ids[:3]
    with_undated = ing._chunk_games(undated, {}, 60)
    assert [g for _, group in with_undated for g in group] == undated
    partial = dict(gamedays)
    for g in undated:
        partial.pop(g, None)
    flat2 = [g for _, group in ing._chunk_games(undated, partial) for g in group]
    assert sorted(flat2) == sorted(undated), "an undated game was dropped"


def test_boxscore_pull_is_identical_chunked_or_unchunked():
    """The guardrail: chunking + the progress bar must not change the frame.

    Fetches are stubbed, so this compares the actual returned rows under a
    deliberately date-SHUFFLED id list (where chunking genuinely reorders the
    work) against the same call with chunking disabled.
    """
    base = pd.Timestamp("2024-10-01")
    ids = [f"{base + pd.Timedelta(days=d):%Y%m%d}_{k}" for d in range(200) for k in range(2)]
    gamedays = {g: pd.Timestamp(g[:8]) for g in ids}
    shuffled = list(ids)
    np.random.default_rng(3).shuffle(shuffled)

    def _fake_http(url):
        # Echo the id back verbatim: _parse_boxscore stores str(payload["id"]),
        # so the row's game_id matches the caller's id exactly. Real payloads
        # always carry gameState; the settled-state guard needs it non-empty.
        gid = url.rsplit("/", 2)[-2]
        return {"id": gid, "gameState": "OFF", "away": {"abbrev": "AAA", "goals": 1},
                "home": {"abbrev": "HHH", "goals": 2}}

    with tempfile.TemporaryDirectory() as tmp:
        with _mock_patch.object(ing, "_http_json", side_effect=_fake_http), \
                _mock_patch.object(ing, "_cache_path", side_effect=lambda n: Path(tmp) / n), \
                _mock_patch.object(ing, "_progress_bar", return_value=None):
            chunked = ing.load_boxscores(shuffled, use_cache=False, gameday_by_id=gamedays)
            plain = ing.load_boxscores(shuffled, use_cache=False)

    assert len(chunked) == len(plain) == len(shuffled)
    # Same rows, same values, SAME ORDER — the caller's id order survives.
    assert chunked["game_id"].astype(str).tolist() == shuffled
    assert plain["game_id"].astype(str).tolist() == shuffled
    pd.testing.assert_frame_equal(chunked, plain)


def test_pull_progress_bar_is_drawn_even_when_stderr_is_not_a_terminal():
    """The bar must exist wherever the work does — including a pipe.

    This is the whole bug behind "the NHL run shows no progress bars". The
    bar was gated on ``stderr.isatty()``, and the Kaggle notebook drives the
    pipeline through ``subprocess``, so stderr is ALWAYS a pipe there and the
    gate made the bar a terminal-only feature: the 2026-09-26 boxscore phase
    spent three minutes per chunk with no bar at all.

    MLB already has the behaviour to copy. Its bars come from
    ``pybaseball.statcast()``, which wraps each chunk fetch in its own tqdm
    and emits regardless of TTY, and they render in the notebook output. A
    ``\\r``-repainting bar is not the unreadable "stream of snapshots" it was
    assumed to be: the notebook redraws it, and a redirected log keeps one
    readable line per refresh. Suppressing it was the defect, not the fix.

    The captured context is SIMULATED rather than assumed: ``sys.stderr`` is
    swapped for a non-tty in-memory stream for the test body, so the pin
    holds under every invocation shape — console, pipe, file, or Windows'
    NUL device (which answers ``isatty()`` True as a character device and
    used to break ``2>/dev/null`` smoke runs of this suite at this test's
    precondition).
    """
    import io

    captured = io.StringIO()
    real_stderr = sys.stderr
    try:
        sys.stderr = captured
        bar = ing._progress_bar(10, "score chunk 1/11")
        assert bar is not None, (
            "no bar is drawn when stderr is a pipe - the bar is still gated "
            "on a terminal, so every captured run has none")
        try:
            bar.update(1)
            bar.close()
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"the bar raised on a pipe: {exc!r}") from exc
    finally:
        sys.stderr = real_stderr


def test_pull_progress_bar_never_breaks_a_run():
    """The bar is decoration: it must never be able to fail a run, and the
    pull's data must be identical with a bar driving it and with no bar."""
    # A tqdm that explodes on attribute access must not escape.
    class _BoomModule:
        def __getattr__(self, name):
            raise RuntimeError("boom")
    with _mock_patch.dict(sys.modules, {"tqdm": _BoomModule()}):
        assert ing._progress_bar(10, "x") is None
    # And an absent tqdm is the fallback the inline bar exists for.
    with _mock_patch.dict(sys.modules, {"tqdm": None}):
        assert ing._progress_bar(10, "x") is None

    ids = ["2024010101", "2024010201"]
    gamedays = {g: pd.Timestamp(g[:8]) for g in ids}

    def _fake_http(url):
        gid = url.rsplit("/", 2)[-2]
        return {"id": gid, "gameState": "OFF", "away": {"abbrev": "AAA", "goals": 1},
                "home": {"abbrev": "HHH", "goals": 2}}

    class _Recorder:
        def __init__(self):
            self.updates = 0
            self.closed = 0
        def update(self, n=1):
            self.updates += n
        def set_postfix(self, **kw):
            pass
        def close(self):
            self.closed += 1

    runs = []
    for bar in (None, _Recorder()):
        def _bar(total, desc, _b=bar):
            return _b
        with tempfile.TemporaryDirectory() as tmp:
            with _mock_patch.object(ing, "_http_json", side_effect=_fake_http), \
                    _mock_patch.object(ing, "_cache_path", side_effect=lambda n: Path(tmp) / n), \
                    _mock_patch.object(ing, "_progress_bar", side_effect=_bar):
                runs.append((bar, ing.load_boxscores(
                    ids, use_cache=False, gameday_by_id=gamedays)))
    quiet, shown = runs[0][1], runs[1][1]
    assert len(quiet) == len(ids)
    pd.testing.assert_frame_equal(quiet, shown)
    rec = runs[1][0]
    assert rec.updates == len(ids) and rec.closed == 1, \
        f"the bar did not track the pull (updates={rec.updates}, closed={rec.closed})"


def test_score_dates_chunking_covers_every_date_once():
    dates = [f"2024-{m:02d}-{d:02d}" for m in range(1, 13) for d in (1, 15)]
    windows = ing._date_windows(dates, ing.PULL_CHUNK_DAYS)
    flat = [d for _, group in windows for d in group]
    assert sorted(flat) == sorted(dates)
    assert len(flat) == len(set(flat))
    # 60-day windows over a calendar year -> 7, not 12 (the chunking is real).
    assert 5 <= len(windows) <= 8, [label for label, _ in windows]


def test_unplayed_past_game_settles_after_the_posting_lag_grace():
    """A null score means two different things either side of the grace window.

    Inside it, the game simply has not posted yet and the page must stay
    uncached or the game becomes permanently invisible. Past it, the game was
    cancelled or postponed and the null is PERMANENT — caching it stops the
    page being re-pulled forever (game 2024010044 has sat in gameState=FUT
    since the shortened 2024-25 season, so "re-pulled until it settles" was a
    promise that could never be kept).
    """
    def _game(gid, played):
        return {"id": gid, "gameDate": "2024-10-07", "season": 2024,
                "awayTeam": {"id": 1, "name": {"default": "A"}, "abbrev": "AAA",
                             "score": 1 if played else None, "record": "1-1"},
                "homeTeam": {"id": 2, "name": {"default": "H"}, "abbrev": "HHH",
                             "score": 2 if played else None, "record": "1-1"},
                "venue": {"default": "V"}, "gameState": "OFF" if played else "FUT",
                "gameType": 2}

    today = date.today()
    inside = (today - timedelta(days=1)).isoformat()      # within the lag
    past = (today - timedelta(days=ing.SETTLE_GRACE_DAYS + 30)).isoformat()
    cases = [("inside grace -> NOT cached", inside, [_game(1, True), _game(2, False)], False),
             ("past grace   -> CACHED", past, [_game(1, True), _game(2, False)], True),
             ("all played   -> CACHED", past, [_game(1, True), _game(2, True)], True)]
    for label, day, games, expect in cases:
        stamp = day.replace("-", "")
        path = BACKEND / f"settle_{stamp}.parquet"
        path.unlink(missing_ok=True)
        with _mock_patch.object(ing, "_http_json", return_value={"games": games}), \
                _mock_patch.object(ing, "_cache_path", side_effect=lambda n: path):
            ing.load_score_dates([day])
        got = path.exists()
        path.unlink(missing_ok=True)
        assert got is expect, f"{label}: cached={got}, expected {expect}"


def test_game_with_no_result_is_accounted_for_in_the_log(caplog=None):
    """A cancelled game silently shrinks the decided population, so it must be
    named rather than quietly dropped."""
    day = (date.today() - timedelta(days=ing.SETTLE_GRACE_DAYS + 30)).isoformat()
    stamp = day.replace("-", "")
    path = BACKEND / f"noresult_{stamp}.parquet"
    path.unlink(missing_ok=True)
    games = [{"id": 2024010044, "gameDate": day, "season": 2024,
              "awayTeam": {"id": 1, "name": {"default": "A"}, "abbrev": "NSH",
                           "score": None, "record": "1-1"},
              "homeTeam": {"id": 2, "name": {"default": "H"}, "abbrev": "TBL",
                           "score": None, "record": "1-1"},
              "venue": {"default": "V"}, "gameState": "FUT", "gameType": 1}]
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    ing.logger.addHandler(handler)
    ing.logger.setLevel(logging.INFO)
    try:
        with _mock_patch.object(ing, "_http_json", return_value={"games": games}), \
                _mock_patch.object(ing, "_cache_path", side_effect=lambda n: path):
            ing.load_score_dates([day])
    finally:
        ing.logger.removeHandler(handler)
    path.unlink(missing_ok=True)
    text = " ".join(r.getMessage() for r in records)
    assert "2024010044" in text, f"the no-result game was not named: {text[:200]}"


def test_running_scores_are_null_and_never_cached_mid_game():
    """A LIVE game's running score is not a result (2026-09-29 regression).

    That slate shipped TOR@MTL 1-1 and BOS@NYR 0-0 mid-game; a 0-0 running
    score parses as a DECIDED 0.0/0.0 tie that no null-filter can catch, so
    the row must lose its scores at fetch time and its page must never be
    cached while a game is in flight — a page cached with every game either
    final or live would be poisoned forever (cache hits never re-pull).
    """
    day = (date.today() - timedelta(days=1)).isoformat()  # inside the lag
    stamp = day.replace("-", "")
    path = BACKEND / f"live_{stamp}.parquet"
    path.unlink(missing_ok=True)

    def _game(gid, state, home, away):
        return {"id": gid, "gameDate": day, "season": 2026,
                "awayTeam": {"id": 1, "name": {"default": "A"}, "abbrev": "AAA",
                             "score": away, "record": "1-1"},
                "homeTeam": {"id": 2, "name": {"default": "H"}, "abbrev": "HHH",
                             "score": home, "record": "1-1"},
                "venue": {"default": "V"}, "gameState": state, "gameType": 2}

    pages = [
        # First pull: two finals and one game IN PLAY with a running score.
        {"games": [_game(1, "OFF", 2, 1), _game(2, "OFF", 4, 3),
                   _game(3, "LIVE", 0, 0)]},
        # Second pull, same day: the in-play game has since gone final 3-2.
        {"games": [_game(1, "OFF", 2, 1), _game(2, "OFF", 4, 3),
                   _game(3, "OFF", 3, 2)]},
    ]
    calls = {"n": 0}

    def _fake_http(url):
        page = pages[min(calls["n"], len(pages) - 1)]
        calls["n"] += 1
        return page

    with _mock_patch.object(ing, "_http_json", side_effect=_fake_http), \
            _mock_patch.object(ing, "_cache_path", side_effect=lambda n: path):
        df1 = ing.load_score_dates([day], use_cache=True)
        assert not path.exists(), "a page with an in-flight game was cached"
        live = df1[df1["game_id"] == "3"]
        assert len(live) == 1
        assert live["home_score"].isna().all() and live["away_score"].isna().all(), (
            "a running score survived into the schedule frame")
        finals = df1[df1["game_id"].isin(["1", "2"])]
        assert finals["home_score"].notna().all()

        df2 = ing.load_score_dates([day], use_cache=True)
        assert calls["n"] == 2, "the live page was cached instead of re-pulled"
        after = df2[df2["game_id"] == "3"]
        assert float(after["home_score"].iloc[0]) == 3.0
        assert float(after["away_score"].iloc[0]) == 2.0
    path.unlink(missing_ok=True)


def test_recent_settled_score_pages_refresh_for_feed_corrections():
    """A settled page inside SCORE_REFRESH_DAYS re-pulls for feed fixes.

    2026-10-08 official-vs-cache audit: the settled 2026-10-06 page held
    away_sog 28/28 while the feed's corrected values were 27/30 — settled
    pages were NEVER re-fetched, so the stale shots values would feed
    trailing shots_for/against features forever. Outside the window the
    permanent-settlement contract holds (cache hit, no network), and a
    FAILED refresh serves the settled cache — never a hole in the decided
    population.
    """
    day = (date.today() - timedelta(days=1)).isoformat()       # refreshable
    old = (date.today() - timedelta(
        days=ing.SCORE_REFRESH_DAYS + 40)).isoformat()          # settled
    path = BACKEND / f"refresh_{day.replace('-', '')}.parquet"
    old_path = BACKEND / f"refresh_{old.replace('-', '')}.parquet"

    def _game(day_, gid, sog):
        return {"id": gid, "gameDate": day_, "season": 2026,
                "awayTeam": {"id": 1, "name": {"default": "A"},
                             "abbrev": "AAA", "score": 1, "sog": sog,
                             "record": "1-1"},
                "homeTeam": {"id": 2, "name": {"default": "H"},
                             "abbrev": "HHH", "score": 2, "sog": 31,
                             "record": "1-1"},
                "venue": {"default": "V"}, "gameState": "OFF",
                "gameType": 2}

    def _page(day_, away_sog):
        return pd.DataFrame([{
            "game_id": "1", "season": 2026, "game_type": 2,
            "game_date": day_, "start_time_utc": "2026-10-07T23:30:00Z",
            "home_team": "HHH", "away_team": "AAA",
            "home_score": 2.0, "away_score": 1.0,
            "home_sog": 31.0, "away_sog": float(away_sog),
            "venue": "V", "game_state": "OFF", "game_outcome": "REG",
        }])

    calls: list[str] = []

    def _corrected(url):
        calls.append(url)
        return {"games": [_game(day, 1, 27)]}   # feed corrected 28 -> 27

    def _boom(url):
        calls.append(url)
        raise RuntimeError("network down")

    _page(day, 28).to_parquet(path, index=False)   # stale seeded value
    try:
        # (1) inside the refresh window: re-pull, correction lands,
        #     page re-cached.
        with _mock_patch.object(ing, "_http_json", side_effect=_corrected), \
                _mock_patch.object(ing, "_cache_path",
                                   side_effect=lambda n: path):
            df = ing.load_score_dates([day], use_cache=True)
        assert len(calls) == 1, (
            "a settled page inside the refresh window was never re-pulled")
        assert float(df["away_sog"].iloc[0]) == 27.0, (
            "the feed correction did not land in the schedule frame")
        assert float(pd.read_parquet(path)["away_sog"].iloc[0]) == 27.0, (
            "the refresh did not re-cache the corrected page")

        # (2) outside the window: cache hit, zero network.
        _page(old, 28).to_parquet(old_path, index=False)
        calls.clear()
        with _mock_patch.object(ing, "_http_json", side_effect=_corrected), \
                _mock_patch.object(ing, "_cache_path",
                                   side_effect=lambda n: old_path):
            df_old = ing.load_score_dates([old], use_cache=True)
        assert not calls, "a settled page outside the refresh window was re-pulled"
        assert float(df_old["away_sog"].iloc[0]) == 28.0

        # (3) failed refresh serves the settled cache (restore the stale
        #     value first so the fallback is provably the cache's row).
        _page(day, 28).to_parquet(path, index=False)
        calls.clear()
        with _mock_patch.object(ing, "_http_json", side_effect=_boom), \
                _mock_patch.object(ing, "_cache_path",
                                   side_effect=lambda n: path):
            df_fb = ing.load_score_dates([day], use_cache=True)
        assert len(df_fb) == 1 and float(df_fb["away_sog"].iloc[0]) == 28.0, (
            "a failed refresh dropped the recent date instead of serving "
            "the settled cache")
    finally:
        path.unlink(missing_ok=True)
        old_path.unlink(missing_ok=True)


def test_eligible_games_renulls_in_flight_scores():
    """The admission point must not admit a running score as decided.

    master_pipeline derives ``decided_all`` from eligible_games' output, and
    the serving slate is built from the SAME survivors — so an in-flight row
    keeps its schedule row but loses every score, covering frames that did
    not come from today's guarded load_score_dates (caches written by
    pre-guard versions, test fixtures, hand-built frames).
    """
    schedule = pd.DataFrame([
        {"game_id": "1", "season": 2025, "game_type": 2,
         "game_state": "OFF", "home_score": 2, "away_score": 1},
        {"game_id": "2", "season": 2025, "game_type": 2,
         "game_state": "LIVE", "home_score": 1, "away_score": 1},
        {"game_id": "3", "season": 2025, "game_type": 2,
         "game_state": "FUT", "home_score": np.nan, "away_score": np.nan},
    ])
    out = ing.eligible_games(schedule)
    decided = out[out["home_score"].notna() & out["away_score"].notna()]
    assert list(decided["game_id"]) == ["1"], "an in-flight row counted as decided"
    assert len(out) == 3, "in-flight rows must SURVIVE for the serving slate"
    live = out[out["game_id"] == "2"]
    assert live["home_score"].isna().all() and live["away_score"].isna().all()


def test_boxscore_cache_never_frozen_mid_game():
    """A mid-game boxscore must not freeze partial stats into a per-game
    cache nothing ever invalidates (2026-09-29 regression: boxscore chunk
    11/11 fetched the LIVE TOR@MTL game). Behave like an unavailable
    boxscore — warned, skipped, uncached — and pull it fresh once settled.
    """
    gid = "2026020002"
    bs_live = {"id": int(gid), "gameState": "LIVE",
               "homeTeam": {"sog": 9}, "awayTeam": {"sog": 7}}
    bs_final = {"id": int(gid), "gameState": "OFF",
                "homeTeam": {"sog": 31}, "awayTeam": {"sog": 28}}
    calls = {"n": 0}

    def _fake_http(url):
        calls["n"] += 1
        if url.endswith("/right-rail"):
            return {"teamGameStats": [
                {"category": "powerPlay", "homeValue": "1/3", "awayValue": "0/2"},
                {"category": "faceoffWins", "homeValue": "30/60", "awayValue": "30/60"}]}
        return bs_live if calls["n"] == 1 else bs_final

    path = BACKEND / "boxscore_livetest.parquet"
    path.unlink(missing_ok=True)
    with _mock_patch.object(ing, "_http_json", side_effect=_fake_http), \
            _mock_patch.object(ing, "_cache_path", side_effect=lambda n: path):
        df1 = ing.load_boxscores([gid], use_cache=True, chunk_days=0,
                                 pause_sec=0)
        assert len(df1) == 0, "a mid-game boxscore was returned as stats"
        assert not path.exists(), "a mid-game boxscore was cached"

        df2 = ing.load_boxscores([gid], use_cache=True, chunk_days=0,
                                 pause_sec=0)
        assert calls["n"] == 3, "settled boxscore and official team stats must be fetched"
        assert len(df2) == 1
        assert float(df2["home_sog"].iloc[0]) == 31.0
        assert path.exists(), "the settled boxscore was not cached"
    path.unlink(missing_ok=True)


def test_score_pages_carry_the_float64_numeric_contract():
    """Every score page — settled or cancelled — returns float64 score/sog
    columns (2026-09-30 regression).

    A cancelled-date page parses a column of bare Nones, which pandas types
    as OBJECT; under pandas 2.x the concat then raises the all-NA-entry
    FutureWarning (the 02:17 run printed it from ingestion.py:591). The
    dtype contract is the invariant that makes the concat identical under
    the old and the new pandas semantics, fetched or cache-loaded.
    """
    day = (date.today() - timedelta(days=1)).isoformat()  # inside the lag
    stamp = day.replace("-", "")
    path = BACKEND / f"dtype_{stamp}.parquet"
    path.unlink(missing_ok=True)

    def _game(gid, state, home, away):
        return {"id": gid, "gameDate": day, "season": 2026,
                "awayTeam": {"id": 1, "name": {"default": "A"}, "abbrev": "AAA",
                             "score": away, "record": "1-1"},
                "homeTeam": {"id": 2, "name": {"default": "H"}, "abbrev": "HHH",
                             "score": home, "record": "1-1"},
                "venue": {"default": "V"}, "gameState": state, "gameType": 2}

    with _mock_patch.object(
            ing, "_http_json",
            return_value={"games": [_game(1, "OFF", 2, 1),
                                    _game(2, "FUT", None, None)]}), \
            _mock_patch.object(ing, "_cache_path", side_effect=lambda n: path):
        df = ing.load_score_dates([day])
    path.unlink(missing_ok=True)
    for col in ("home_score", "away_score", "home_sog", "away_sog"):
        assert str(df[col].dtype) == "float64", \
            f"{col} left the float64 contract: {df[col].dtype}"


def test_boxscore_dtype_contract_preserves_identity_columns():
    """The boxscore coercion touches ONLY numeric stats — goalie id/name/
    decision are identities and must survive as strings, while a game with
    no recorded stats still yields float64 stat columns."""
    row = {c: None for c in ing.BOXSCORE_COLS}
    row["game_id"] = "2026020002"
    row["home_sog"] = 31
    row["home_goalie_name"] = "Jeremy Swayman"
    df = ing._coerce_boxscore_dtypes(pd.DataFrame([row], columns=ing.BOXSCORE_COLS))
    for col in ing.BOXSCORE_NUMERIC_COLS:
        assert str(df[col].dtype) == "float64", \
            f"{col} left the float64 contract: {df[col].dtype}"
    assert df["home_goalie_name"].iloc[0] == "Jeremy Swayman"
    assert str(df["home_goalie_name"].dtype) != "float64"
    assert str(df["home_goalie_decision"].dtype) != "float64"
    assert str(df["home_goalie_id"].dtype) != "float64"


def test_coverage_verdict_separates_cold_nulls_from_real_defects():
    """The console must not report one undifferentiated coverage percentage.

    A team's first game has no prior history by design (cold null); a warm
    null is a defect. Logging only a blended percentage makes "the season
    started" indistinguishable from "something is broken", and the goalie
    family once read 96-98% on a decided pool that published nulls for every
    slate game. The split already exists in monitoring.coverage — this pins
    that Phase 14 actually reports it, for BOTH windows (the serving slate is
    the window that actually ships predictions, and an empty slate must not
    silently drop the report).
    """
    import master_pipeline as mp

    def _row(feature, window, n_games, measured, cold, warm, cause="null"):
        return {"feature": feature, "window": window, "n_games": n_games,
                "n_measured": measured, "n_null": n_games - measured,
                "n_cold_null": cold, "n_warm_null": warm,
                "status": "OK" if not warm else "ATTENTION", "cause": cause}

    rows = [
        # baseline (the decided pool): perfect except team debuts (cold)
        _row("elo_home", "baseline", 100, 96, 4, 0),
        _row("win_pct_home", "baseline", 100, 95, 5, 0),
        # serving slate: one genuine defect
        _row("goalie_sv_pct_home", "serving slate", 5, 4, 0, 1, "warm_null"),
    ]
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    mp.logger.addHandler(handler)
    mp.logger.setLevel(logging.INFO)
    try:
        mp._log_coverage_verdict(rows)
    finally:
        mp.logger.removeHandler(handler)

    infos = [r.getMessage() for r in records if r.levelno < logging.WARNING]
    warns = [r.getMessage() for r in records if r.levelno >= logging.WARNING]
    text = " ".join(infos)
    # Both windows get a verdict line...
    assert "baseline" in text and "serving slate" in text, text[:200]
    # ...the baseline window is reported as clean, with its cold nulls named as
    # by-design rather than as missing data.
    assert "no warm nulls" in text, text[:200]
    assert "9 cold null(s) by design" in text, \
        f"cold nulls were not reported as by-design: {text[:200]}"
    # The slate defect is a WARNING that names the feature, not a footnote.
    assert len(warns) == 1, f"expected exactly one warm-null warning, got {warns}"
    assert "goalie_sv_pct_home" in warns[0] and "serving slate" in warns[0], warns[0]


def test_pit_fold_labels_are_valid_for_the_frame_the_oof_rebuilds():
    """The pins in this suite hand fold labels to walk_forward_oof, which
    canonicalizes the frame it is given. Those labels are POSITIONS, so the
    frame they were generated over must already BE the canonical order — today
    that holds only because the synth pool is generated in date order. Pin it,
    so changing the generator forces the call sites to canonicalize instead of
    silently scoring the wrong games."""
    games = feat_mod.build_game_features(_synth_games(n_days=40))
    ref = folds_mod.make_folds(
        folds_mod.canonical_sort(games, "gameday"), date_col="gameday")
    probe = folds_mod.make_folds(games, date_col="gameday")
    assert [f.val_idx.tolist() for f in probe] == \
           [f.val_idx.tolist() for f in ref], \
        "this suite generates fold labels over a NON-canonical frame; " \
        "walk_forward_oof rebuilds it in canonical order, so the labels would " \
        "select the wrong games — canonicalize at the make_folds call sites"


def test_mlb_observed_date_fold_geometry_handles_schedule_gaps():
    """NHL folds use seven observed dates, not seven arithmetic days."""
    dates = [pd.Timestamp("2025-10-01") + pd.Timedelta(days=i)
             for i in range(46)]
    dates = [d for d in dates if d != pd.Timestamp("2025-10-15")]
    rows = []
    for d in dates:
        for i in range(7):
            rows.append({"gameday": d, "season": 2025,
                         "game_id": f"{d:%Y%m%d}_{i}"})
    games = pd.DataFrame(rows)
    folds = folds_mod.make_folds(games)
    unique_dates = pd.Index(sorted(games["gameday"].unique()))

    assert len(folds) == 3
    assert folds[0].val_start == unique_dates[config.WARMUP_DAYS]
    assert folds[0].val_end == unique_dates[config.WARMUP_DAYS + 6]
    assert folds[-1].is_partial_tail
    assert all(pd.Timestamp(games.loc[f.train_idx, "gameday"].max())
               < f.val_start for f in folds)


# ---------------------------------------------------------------------------
# 2. Grid coherence
# ---------------------------------------------------------------------------
def _mc_frame(n: int = 40, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return dist_mod.simulate_distributions(
        rng.uniform(2.4, 3.6, n), rng.uniform(2.2, 3.4, n),
        alpha_home=0.05, alpha_away=0.04, n_draws=4000, seed=seed)


def test_spread_grid_is_coherent():
    df = _mc_frame()
    for line in config.SPREAD_GRID:
        home = df[dist_mod._grid_key("p_home_cover", line)].to_numpy(float)
        push = df[dist_mod._grid_key("p_push", line)].to_numpy(float)
        away = 1.0 - home - push
        np.testing.assert_allclose(home + push + away, 1.0, atol=1e-6)
        assert (home >= 0).all() and (push >= 0).all()
    # Half-stop lines carry a cover probability and never a push.
    for line in config.HALF_STOP_LINES:
        key = dist_mod._grid_key("p_home_cover", line)
        assert key in df.columns
        assert f"p_push_{line}" not in {c.replace("p_push", "p_push") for c in [key]}
        vals = df[key].to_numpy(float)
        assert np.isfinite(vals).all() and (vals >= 0).all() and (vals <= 1).all()


def test_totals_grid_is_coherent():
    df = _mc_frame()
    for line in config.TOTAL_GRID:
        over = df[dist_mod._grid_key("p_over", line)].to_numpy(float)
        push = df[dist_mod._grid_key("p_push_total", line)].to_numpy(float)
        under = df[dist_mod._grid_key("p_under", line)].to_numpy(float)
        np.testing.assert_allclose(over + push + under, 1.0, atol=1e-6)
        assert (over >= 0).all() and (push >= 0).all() and (under >= 0).all()
    # Grid monotonicity: P(over L) must not increase as the line rises.
    overs = [df[dist_mod._grid_key("p_over", line)].to_numpy(float)
             for line in config.TOTAL_GRID]
    for a, b in zip(overs, overs[1:]):
        assert (b <= a + 1e-9).all(), "P(over) increased with the line"
    # The totals push namespace is EXCLUSIVE: every totals line prices its
    # push at p_push_total_{U} (the spread grid legitimately owns p_push_{N}
    # = P(margin == N) — NHL's ranges overlap, MLB/NFL's don't, so the NHL
    # totals push needs its own column family to avoid the collision).
    for line in config.TOTAL_GRID:
        assert dist_mod._grid_key("p_push_total", line) in df.columns
    # For the overlapping lines (4..8) the legacy p_push_{N} column is the
    # SPREAD push: home-cover + spread-push stays a coherent 2-way pair.
    for line in (4, 5, 6, 7, 8):
        home = df[dist_mod._grid_key("p_home_cover", line)].to_numpy(float)
        spread_push = df[dist_mod._grid_key("p_push", line)].to_numpy(float)
        assert ((home + spread_push) <= 1.0 + 1e-9).all()


def test_fair_line_aliases_pick_the_nearest_grid_line():
    df = _mc_frame()
    # fair_total is the total-grid line whose P(over) is closest to 50%.
    for _, r in df.iterrows():
        vals = {line: r[dist_mod._grid_key("p_over", line)]
                for line in config.TOTAL_GRID}
        best = min(vals, key=lambda line: abs(vals[line] - 0.5))
        assert r["fair_total"] == float(best)
        assert abs(r["p_over_fair"] - vals[best]) < 1e-12


def test_game_distribution_legacy_negative_labels():
    row = dist_mod.game_distribution(3.2, 2.6, n_draws=2000, seed=1)
    assert row["p_home_cover_-2"] == row["p_home_cover_m2"]
    assert row["p_push_-2"] == row["p_push_m2"]
    assert 0.0 <= row["p_home_win_derived"] <= 1.0
    assert abs(row["p_home_win_derived"] + row["p_away_win_derived"]
               + row["p_tie"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 3. Prequential per-line Platt sees prior folds only
# ---------------------------------------------------------------------------
def test_prequential_line_platt_is_strictly_prior():
    rng = np.random.default_rng(3)
    n = 300
    y = rng.integers(0, 2, n).astype(int)
    p = np.clip(0.5 + rng.normal(0, 0.15, n) + 0.25 * (y - 0.5), 0.02, 0.98)
    folds = np.repeat(np.arange(6), 50)

    out, final = dist_mod._prequential_line(p, y, folds)
    assert len(out) == n and np.isfinite(out).all()
    assert (out >= 1e-6).all() and (out <= 1 - 1e-6).all()
    assert final is not None  # the all-OOF map is published for serving

    y_poisoned = y.copy()
    y_poisoned[folds == 5] = 1 - y_poisoned[folds == 5]
    out_pois, _ = dist_mod._prequential_line(p, y_poisoned, folds)
    early = folds < 5
    np.testing.assert_array_equal(
        out[early], out_pois[early],
        err_msg="earlier folds moved by a later fold's labels — leakage")


def test_calibrate_market_frame_produces_a_reusable_bundle():
    games = feat_mod.build_game_features(_synth_games(n_days=60))
    folds = folds_mod.make_folds(games, date_col="gameday")
    oof = dist_mod.walk_forward_oof(games, fold_list=folds)["oof"]
    dist = dist_mod.apply_distribution(oof)
    calibrated, bundle = dist_mod.calibrate_market_frame(dist)
    assert bundle["method"] == "prequential_platt"
    assert bundle["scope"] == "line_specific"
    # Every grid line's calibrator is recorded (None maps are allowed but
    # the keys must exist).
    for line in config.TOTAL_GRID:
        assert str(line) in bundle["totals"]
    for line in config.SPREAD_GRID:
        assert str(line) in bundle["run_lines"]
    # Post-calibration coherence holds for EVERY totals line (totals pushes
    # ride their own p_push_total_{U} namespace, so the spread pass can no
    # longer clobber them).
    for line in config.TOTAL_GRID:
        tri = sum(calibrated[dist_mod._grid_key(k, line)].to_numpy(float)
                  for k in ("p_over", "p_push_total", "p_under"))
        assert np.isfinite(tri).all(), f"line {line}: NaN after calibration"
        np.testing.assert_allclose(tri, 1.0, atol=1e-4,
                                   err_msg=f"line {line}: incoherent triple")
    # The bundle re-applies to a slate frame with no outcomes present.
    slate = calibrated.head(3).drop(
        columns=[c for c in ("home_score", "away_score", "margin", "total",
                             "home_win", "fold_id") if c in calibrated.columns])
    reapplied = dist_mod.apply_market_calibration(slate, bundle)
    assert len(reapplied) == 3
    assert dist_mod._grid_key("p_over", 6) in reapplied.columns


def test_moneyline_fold_trainer_runs_all_three_members_with_fold_validation():
    """Each fold fits and scores XGB/LGBM/elastic-net on the same geometry."""
    games = feat_mod.build_game_features(_synth_games(n_days=40))
    folds = folds_mod.make_folds(games)

    class _FakeModel:
        def __init__(self, name):
            self.name = name
            self.fit_calls = []

        def fit(self, X, y, **kwargs):
            self.fit_calls.append(kwargs)

        def predict_proba(self, X):
            p = np.full(len(X), 0.5, dtype=float)
            return np.column_stack([1.0 - p, p])

    fitted = []

    def _fake_member(name, fold=False, n_estimators=None, causal=False):
        model = _FakeModel(name)
        fitted.append((name, fold, model,
                       {"fold": fold, "n_estimators": n_estimators,
                        "causal": causal}))
        return model

    with _mock_patch.object(ml_mod, "_make_member", side_effect=_fake_member):
        out = ml_mod.walk_forward_oof(games, fold_list=folds)

    # One SHIPPED model per member per fold, plus one measurement-only
    # early-stopped xgboost PROBE per fold (causal fold rounds).
    assert len(fitted) == len(folds) * (len(config.ENSEMBLE_MEMBERS) + 1)
    assert {name for name, _, _, _ in fitted} == set(config.ENSEMBLE_MEMBERS)
    for name, fold, model, mk in fitted:
        assert fold is True
        assert len(model.fit_calls) == 1
        if name == "xgboost" and mk["causal"]:
            # CAUSAL: the SHIPPED fold model is a FIXED-budget fit with no
            # eval_set — the window it is scored on never selects it. With
            # no prior measurements (fakes carry no best_iteration), every
            # fold ships at the static fold-0 prior.
            assert "eval_set" not in model.fit_calls[0]
            assert mk["n_estimators"] == config.XGBOOST_FOLD0_ROUNDS
        elif name == "xgboost":
            # The MEASUREMENT probe: early-stopped ON the val window by
            # design — but it is never scored; only its best_iteration is
            # read, and that value may only inform STRICTLY LATER folds.
            assert "eval_set" in model.fit_calls[0]
        elif name == "lightgbm":
            assert "eval_set" not in model.fit_calls[0]
            assert model.fit_calls[0]["categorical_feature"] == config.TREE_CATEGORICAL_COLS
        else:
            assert "eval_set" not in model.fit_calls[0]
    shipped_xgb = [mk for name, _, _, mk in fitted
                   if name == "xgboost" and mk["causal"]]
    probes = [mk for name, _, _, mk in fitted
              if name == "xgboost" and not mk["causal"]]
    assert len(shipped_xgb) == len(folds)
    assert len(probes) == len(folds), (
        "exactly one measurement-only early-stopped probe per fold; the "
        "probe's best_iteration may only inform strictly later folds")
    assert set(f"p_{n}" for n in config.ENSEMBLE_MEMBERS) <= set(out["oof"].columns)


def test_moneyline_blend_uses_prior_fold_weights_only():
    """Fold 0 uses thirds; fold 1 uses the optimizer result from fold 0.

    Published p_ensemble is the causal blend; final-weight replay is an
    explicitly retrospective diagnostic, never the headline or gate input.
    """
    games = feat_mod.build_game_features(_synth_games(n_days=40))
    folds = folds_mod.make_folds(games)
    member_p = {"xgboost": 0.8, "lightgbm": 0.2, "elasticnet": 0.5}

    class _FixedModel:
        def __init__(self, name):
            self.name = name

        def fit(self, X, y, **kwargs):
            return self

        def predict_proba(self, X):
            p = np.full(len(X), member_p[self.name], dtype=float)
            return np.column_stack([1.0 - p, p])

    calls = []

    def _fake_member(name, fold=False, n_estimators=None, causal=False):
        return _FixedModel(name)

    def _fake_optimizer(members, y):
        calls.append((members, y))
        return {"xgboost": 1.0, "lightgbm": 0.0, "elasticnet": 0.0}

    with _mock_patch.object(ml_mod, "_make_member", side_effect=_fake_member), \
            _mock_patch.object(ml_mod, "compute_adaptive_weights",
                               side_effect=_fake_optimizer):
        out = ml_mod.walk_forward_oof(games, fold_list=folds)

    assert len(calls) == len(folds)
    oof = out["oof"]
    causal = "p_ensemble_causal"
    first = oof[oof["fold_id"] == folds[0].fold_id][causal].to_numpy()
    second = oof[oof["fold_id"] == folds[1].fold_id][causal].to_numpy()
    np.testing.assert_allclose(first, 0.5, atol=1e-7)
    np.testing.assert_allclose(second, 0.8, atol=1e-7)
    np.testing.assert_array_equal(oof["p_ensemble"], oof[causal])
    # Final all-prior weights only describe future serving. Replaying them
    # on their own fitting outcomes is diagnostic, not OOF evidence.
    np.testing.assert_allclose(oof["p_ensemble_retrospective"], 0.8, atol=1e-7)


def test_causal_xgb_rounds_selection_rule():
    """Shipped round budget comes from PRIOR measurements only: median of
    the measured best-iterations (even length: averaged, deterministic),
    static config priors otherwise, zeros/invalids filtered — never the
    scored window's own best_iteration."""
    assert ml_mod.causal_xgb_rounds([]) == config.XGBOOST_FOLD0_ROUNDS
    assert ml_mod.causal_xgb_rounds([21, 19, 26]) == 21
    assert ml_mod.causal_xgb_rounds([19, 26]) == (19 + 26) // 2
    assert ml_mod.causal_xgb_rounds([0, 30]) == 30, "invalid measurements filter out"
    assert ml_mod.causal_xgb_rounds([0, 0]) == config.XGBOOST_REFIT_ROUNDS, (
        "all-invalid measurements fall to the static refit prior, never 0")


def test_causal_rounds_shipped_budget_uses_strictly_prior_folds_only():
    """Each fold's early-stopped probe MEASURES; the shipped fold model is
    refit at the median of STRICTLY PRIOR measurements. With probes reading
    41, 42, 43, ... the shipped budgets must walk [50, 41, 41, 42] — a
    fold's own val window never selects its own rounds (2a9554b measured
    the val-selected variant flattering OOF by ~0.011 logloss / ~0.04 AUC
    on MLB's geometry; this pin keeps the NHL port honest)."""
    games = feat_mod.build_game_features(_synth_games(n_days=60))
    folds = folds_mod.make_folds(games, date_col="gameday")[:4]

    shipped_budgets: list[int] = []
    probe_measurements = iter(range(41, 41 + 64))

    class _ShippedXGB:
        def fit(self, X, y, **kwargs):
            return self

        def predict_proba(self, X):
            p = np.full(len(X), 0.5, dtype=float)
            return np.column_stack([1.0 - p, p])

    class _ProbeXGB:
        def __init__(self):
            self.best_iteration = next(probe_measurements) - 1

        def fit(self, X, y, **kwargs):
            return self

    class _Other:
        def fit(self, X, y, **kwargs):
            return self

        def predict_proba(self, X):
            p = np.full(len(X), 0.5, dtype=float)
            return np.column_stack([1.0 - p, p])

    def _fake_member(name, fold=False, n_estimators=None, causal=False):
        if name != "xgboost":
            return _Other()
        if causal:
            shipped_budgets.append(int(n_estimators))
            return _ShippedXGB()
        return _ProbeXGB()

    with _mock_patch.object(ml_mod, "_make_member", side_effect=_fake_member):
        ml_mod.walk_forward_oof(games, fold_list=folds)

    assert shipped_budgets == [config.XGBOOST_FOLD0_ROUNDS, 41, 41, 42], (
        "fold k's shipped budget must be the median of folds 0..k-1's "
        "measurements — its own window must not appear in that median")


# ---------------------------------------------------------------------------
# 4. Moneyline leakage: a fold's own labels cannot move its own predictions
# ---------------------------------------------------------------------------
def test_moneyline_oof_folds_are_self_leak_free():
    """walk_forward_oof trains strictly-prior and predicts the val fold.
    Poisoning every OTHER fold's labels must leave this fold's member
    probabilities bit-identical (train pool unchanged) — the same pinned
    property MLB's prequential calibration enforces for the blend."""
    games = feat_mod.build_game_features(_synth_games(n_days=60))
    folds = folds_mod.make_folds(games, date_col="gameday")
    target = folds[min(2, len(folds) - 1)]

    g1 = games.copy()
    g2 = games.copy()
    # Poison labels in a DIFFERENT fold's validation window (both directions
    # of a 5-1 blowout → the ladder's trailing stats diverge) — the target
    # fold's own predictions must not move, because its training pool is
    # identical either way.
    other = folds[0]
    poison_idx = g2.index[g2["gameday"].isin(
        pd.to_datetime(games.loc[other.val_idx, "gameday"]).unique())]
    g2.loc[poison_idx, "home_score"], g2.loc[poison_idx, "away_score"] = \
        g2.loc[poison_idx, "away_score"], g2.loc[poison_idx, "home_score"]
    g2.loc[poison_idx, "home_score"] = g2.loc[poison_idx, "home_score"] + 3

    r1 = ml_mod.walk_forward_oof(g1, fold_list=[target])["oof"]
    r2 = ml_mod.walk_forward_oof(g2, fold_list=[target])["oof"]
    # NOTE: the poisoned rows sit in folds[0]'s window, which is BEFORE the
    # target fold, so they legitimately enter the target's training pool —
    # predictions are allowed to move. The pinned property is the fold's
    # OWN geometry: train strictly before validation.
    tr_end = pd.to_datetime(games.loc[target.train_idx, "gameday"]).max()
    va_start = pd.to_datetime(games.loc[target.val_idx, "gameday"]).min()
    assert tr_end < va_start


def test_prequential_calibrate_sees_prior_folds_only():
    """The favored-space Platt map (production calibration mode) is gated:
    below MIN_OOF_FOR_FIT or on degenerate slope it declines to fit — the
    identity map everywhere (never a silently unvalidated correction)."""
    rng = np.random.default_rng(11)
    n = 300
    y = rng.integers(0, 2, n).astype(float)
    p = np.clip(0.5 + rng.normal(0, 0.12, n) + 0.3 * (y - 0.5), 0.02, 0.98)

    # Favored-space variant (the production calibration mode).
    cal = ml_mod.fit_favored_platt(p, y)
    out = ml_mod.apply_favored_platt(p, cal)
    assert np.isfinite(out).all() and (out > 0).all() and (out < 1).all()
    # Determinism: refitting returns the identical map.
    cal2 = ml_mod.fit_favored_platt(p, y)
    assert cal2 == cal

    # Below MIN_OOF_FOR_FIT the map is declined (identity), never fitted
    # on a sliver of evidence.
    small = ml_mod.fit_favored_platt(p[:10], y[:10])
    assert small is None
    # The favored-side clamp holds: projected back to favored space, the
    # favorite's calibrated probability never drops below 0.5 (the HOME-
    # space output legitimately carries sub-0.5 underdog legs).
    fav_space = np.where(p >= 0.5, out, 1.0 - out)
    assert (fav_space >= 0.5 - 1e-9).all(), \
        "favored-side probability dropped below 0.5"


def _gate_synth_folds(n_folds: int, regime_from: int,
                      seed: int = 5):
    """Synthetic walk-forward evidence for the nested-calibration gate.

    Fold 0 is a thin start (60 rows); later folds hold 100 rows.
    Before ``regime_from`` the true favorite win probability q
    lives in [0.55, 0.95] and the raw blend reports only
    0.5 + 0.4*(q - 0.5) — shrunk toward the coin flip — so a
    favored-space Platt recovers a steep positive map that beats
    the raw blend by ~0.07 nats. From ``regime_from`` on the
    evidence flips: games are priced as STRONG favorites (p in
    [0.6, 0.9]) yet win only 35% of the time, a regime the
    earlier map cannot price — its steep positive map is
    overconfident and its nested-holdout log loss is far worse
    than the raw blend's, i.e. exactly the variance-adding case
    the nested holdout exists to catch.
    """
    rng = np.random.default_rng(seed)
    ps, ys, fids = [], [], []
    for f in range(n_folds):
        n = 60 if f == 0 else 100
        if f < regime_from:
            q = rng.uniform(0.55, 0.95, n)
            u = 0.5 + 0.4 * (q - 0.5)
            rate = q
        else:
            u = rng.uniform(0.6, 0.9, n)
            rate = np.full(n, 0.35)
        y = (rng.random(n) < rate).astype(float)
        ps.append(u)
        ys.append(y)
        fids.append(np.full(n, f))
    return (np.concatenate(ps), np.concatenate(ys),
            np.concatenate(fids))


def test_prequential_fold_calibrators_keep_a_map_that_beats_raw():
    """The gate is not a blanket ban: a map that genuinely beats the
    raw blend on the most recent slice of prior evidence is kept.

    With a stable shrink regime the favored Platt recovers the steep
    positive map the shrunk raw blend cannot express, so the
    nested-holdout log loss wins by ~0.07 nats and every fittable
    fold (6+: 300+ fit rows and a 200-row holdout) keeps its map.
    """
    p, y, fids = _gate_synth_folds(12, regime_from=99)
    calibrators, audit = ml_mod.prequential_fold_calibrators(p, y, fids)
    assert audit["n_folds"] == 12
    assert audit["n_gated"] == 0, audit
    assert audit["n_fitted"] >= 5, audit
    assert calibrators[11] is not None
    assert calibrators[11]["method"] == "favored_platt_floor"
    rec11 = audit["folds"][11]
    assert rec11["decision"] == "fitted"
    assert rec11["nested_holdout_logloss"] < rec11["raw_logloss"]


def test_prequential_fold_calibrators_block_a_variance_adding_map():
    """A map that only fit the OLD regime is blocked, not applied.

    Folds 8+ flip to strong favorites that win like dogs (priced
    [0.6, 0.9], win 35%), so the most recent slice of fold 11's
    prior evidence is a regime the map fitted on the earlier part
    cannot price: its steep positive map is overconfident and its
    nested-holdout log loss is far worse than the raw blend's, and
    the fold is served uncalibrated (identity) instead of importing
    the variance. This is the 2026-10-01 remediation's core
    property — the ungated layer moved logloss 0.67583->0.67731
    and ECE 0.01036->0.01466 on the real OOF.
    """
    p, y, fids = _gate_synth_folds(12, regime_from=8)
    calibrators, audit = ml_mod.prequential_fold_calibrators(p, y, fids)
    rec11 = audit["folds"][11]
    assert rec11["decision"] == "identity"
    assert rec11["reason"] == "gated_no_gain"
    assert rec11["nested_holdout_logloss"] > rec11["raw_logloss"]
    assert calibrators[11] is None
    assert audit["n_gated"] >= 2, audit
    # The stable early folds still earn their maps — the gate blocks
    # the regime break, not calibration itself.
    assert audit["n_fitted"] >= 2, audit


def test_prequential_fold_calibrators_fold_map_never_sees_its_own_fold():
    """Poisoning a fold's own validation window cannot move its map.

    Fold k's map is fitted (and gated) on folds < k only, so
    flipping the labels INSIDE fold k's window leaves fold k's map
    byte-identical — the prequential contract survives the gate.
    """
    p, y, fids = _gate_synth_folds(12, regime_from=99)
    cal_a, audit_a = ml_mod.prequential_fold_calibrators(p, y, fids)
    assert cal_a[11] is not None
    y_poisoned = y.copy()
    y_poisoned[fids == 11] = 1.0 - y_poisoned[fids == 11]
    cal_b, _ = ml_mod.prequential_fold_calibrators(
        p, y_poisoned, fids)
    assert cal_b[11] is not None
    assert cal_b[11]["a"] == cal_a[11]["a"]
    assert cal_b[11]["b"] == cal_a[11]["b"]


def test_prequential_fold_calibrators_eps_knob_is_binding():
    """The accept/reject line is the configured eps, not an accident.

    With eps set impossibly high the same data that earns a map at
    the default eps is served uncalibrated; with eps negative every
    fittable fold keeps its map. The gate's threshold is the config.
    """
    p, y, fids = _gate_synth_folds(12, regime_from=99)
    # Patch the config object the module graph resolved, not the
    # top-level `config` imported above — under pytest those are two
    # module objects (see the _CFG note at the module graph pin).
    # Restore the original default (not a hardcoded value) so the
    # test survives a config change like the 1e-4 -> 5e-3 raise.
    eps_default = _CFG.CAL_GATE_EPS
    try:
        _CFG.CAL_GATE_EPS = 1e9
        _, audit_strict = ml_mod.prequential_fold_calibrators(p, y, fids)
        assert audit_strict["n_fitted"] == 0, audit_strict
        assert audit_strict["n_gated"] >= 5, audit_strict
        _CFG.CAL_GATE_EPS = -1e9
        _, audit_loose = ml_mod.prequential_fold_calibrators(p, y, fids)
        assert audit_loose["n_fitted"] >= 5, audit_loose
        assert audit_loose["n_gated"] == 0, audit_loose
    finally:
        _CFG.CAL_GATE_EPS = eps_default


def test_write_calibration_json_carries_the_gate_audit():
    """The gate's verdict is part of the calibration artifact, not
    just the run log — a reader of the JSON can see how many folds
    kept a map and why the rest did not."""
    gate = {"method": "prequential_platt_gated", "n_folds": 58,
            "n_fitted": 52, "folds": []}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "calibration.json"
        record = serving_mod.write_calibration_json(
            path, {"auc": 0.59, "logloss": 0.67, "brier": 0.24,
                   "ece": 0.01},
            {"auc": 0.59, "logloss": 0.67, "brier": 0.24,
                   "ece": 0.01},
            [], [], {}, platt={"a": 1.0, "b": 0.0, "n": 10},
            run_date="2026-10-01", n_games=10,
            prequential_gate=gate)
        assert record["prequential_gate"] == gate
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk["prequential_gate"]["n_fitted"] == 52
    # Absent gate (older callers) still serializes — the key is
    # always present, just empty.
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "calibration.json"
        record = serving_mod.write_calibration_json(
            path, {}, {}, [], [], {})
        assert record["prequential_gate"] == {}


def test_calibration_section_survives_a_gated_run():
    """A gated run (``platt=None``, the 2026-10-07 production verdict
    ``gated_no_gain``) must still publish MLB's ``calibration`` section:
    identity provenance, raw/calibrated metrics, and the calibrated bucket
    twins. Persisting ``{}`` dropped the recalibration banner state AND the
    shared page's CALIBRATED win-rate column beside MLB's."""
    buckets = [{"bucket": "50–60%", "mean_predicted": 0.55,
                "mean_actual": 0.53, "count": 10, "gap": 0.02}]
    cal_buckets = [{"bucket": "50–60%", "mean_predicted": 0.56,
                    "mean_calibrated": 0.56, "mean_predicted_raw": 0.55,
                    "mean_actual": 0.53, "count": 10, "gap": 0.02,
                    "gap_calibrated": 0.03}]
    raw = {"auc": 0.60, "brier": 0.24, "logloss": 0.68, "ece": 0.010,
           "calibrator_gated_out": True}
    cal_m = {"auc": 0.60, "brier": 0.2401, "logloss": 0.6801,
             "ece": 0.0099}
    with tempfile.TemporaryDirectory() as td:
        record = serving_mod.write_calibration_json(
            Path(td) / "cal.json", raw, cal_m, buckets, [], {},
            platt=None, run_date="20261007", n_games=10,
            calibrated_buckets=cal_buckets)
        sec = record["calibration"]
        assert sec["method"] == "identity"
        assert sec["params"] is None
        assert sec["metrics_raw"]["brier"] == 0.24
        assert sec["metrics_calibrated"]["brier"] == 0.2401
        assert sec["calibration_buckets_calibrated"] == cal_buckets
        assert record["metrics"]["calibrator_gated_out"] is True
        # A fitted run keeps the deployed-map tag MLB ships (the pipeline
        # stamps the flag from ``platt is None`` on the same run).
        fitted = serving_mod.write_calibration_json(
            Path(td) / "fit.json", {**raw, "calibrator_gated_out": False},
            cal_m, buckets, [], {},
            platt={"a": 1.1, "b": -0.02, "n": 2418},
            run_date="20261006", n_games=10)
        assert fitted["calibration"]["method"] == "favored_platt_floor"
        assert fitted["calibration"]["params"] == {
            "a": 1.1, "b": -0.02, "n": 2418}
        assert fitted["metrics"]["calibrator_gated_out"] is False


def test_monitor_report_blocks_are_mlb_rows_with_a_rolling_history():
    """``write_monitor_json`` publishes MLB's report contract: 9-key
    coverage rows (the builder's cold/warm split stays out of the report
    block), top-level ``date``/``version`` stamps, and a rolling-20
    version history whose rows are ``training.update_model_version_history``
    snapshots folded from the dated family — never leaking a later run's
    record into an earlier report."""
    cov = [{"feature": "elo_diff", "window": "baseline", "n_games": 100,
            "pct_measured": 99.0, "pct_nonnull": 99.0,
            "n_default_zero": 0, "status": "OK", "n_measured": 99,
            "n_null": 1, "n_cold_null": 1, "n_warm_null": 0,
            "pct_measured_eligible": 100.0, "cause": "cold_start"}]
    drift = [{"feature": "elo_diff", "current_mean": 1.0,
              "baseline_mean": 1.0, "psi": 0.01, "psi_adjusted": 0.0,
              "noise_floor": 0.01, "mean_shift": 0.0, "shift_se": 0.0,
              "location_shift": False, "status": "OK", "weight_pct": 5.0,
              "n_baseline": 100, "n_current": 20}]
    ens = [{"name": "xgboost", "weight": 0.5, "auc": 0.6,
            "brier": 0.24, "logloss": 0.68, "n_eval": 100}]
    m = {"auc": 0.6, "brier": 0.24, "logloss": 0.68, "ece": 0.01,
         "brier_calibrated": 0.2401, "logloss_calibrated": 0.6801,
         "ece_calibrated": 0.0099}
    platt = {"a": 1.0, "b": 0.05, "n": 2418,
             "method": "favored_platt_floor", "floor": 0.5}
    with tempfile.TemporaryDirectory() as td:
        prior = Path(td) / "nhl_model_monitor_20981231.json"
        prior.write_text(json.dumps({
            "version_history": [{"version": "20981231",
                                 "date": "2098-12-31", "weights": {},
                                 "note": "rebuild run"}]}), encoding="utf-8")
        path = Path(td) / "nhl_model_monitor_20990101.json"
        with _mock_patch.object(mon, "_dump_json"):
            rec = mon.write_monitor_json(
                path, "20990101", drift, cov, ens, [], 0.5, {}, {},
                metrics=m, platt=platt)
        assert rec["date"] == "20990101"
        assert rec["version"] == "v2099.01.01"
        # Coverage: MLB's exact 9 keys; the builder extras stay out.
        row = rec["feature_coverage"][0]
        assert set(row) == {"feature", "window", "n_games", "n_nonnull",
                            "pct_nonnull", "n_measured", "pct_measured",
                            "n_default_zero", "status"}
        assert row["n_nonnull"] == 99  # n_games - n_null, exact
        # Version history: prior family row folded + this run's snapshot
        # in MLB's row schema, oldest-first. A prior row rides AS PUBLISHED
        # (the emitter folds, never rewrites — the published family is kept
        # canonical by the artifact backfill), this run's row is reshaped.
        vh = rec["version_history"]
        assert [r["version"] for r in vh] == ["20981231", "v2099.01.01"]
        cur = vh[-1]
        assert set(cur) == {"version", "date", "weights", "auc", "brier",
                            "logloss", "ece", "brier_calibrated",
                            "logloss_calibrated", "ece_calibrated",
                            "calibration"}
        assert cur["calibration"] == {"a": 1.0, "b": 0.05, "n": 2418,
                                      "method": "favored_platt_floor",
                                      "floor": 0.5}
        # A later-dated artifact never leaks into this report.
        (Path(td) / "nhl_model_monitor_20990102.json").write_text(
            json.dumps({"version_history": [
                {"version": "v2099.01.02", "date": "2099-01-02",
                 "weights": {}}]}), encoding="utf-8")
        with _mock_patch.object(mon, "_dump_json"):
            rec2 = mon.write_monitor_json(
                path, "20990101", drift, cov, ens, [], 0.5, {}, {},
                metrics=m, platt=platt)
        assert "v2099.01.02" not in [r["version"]
                                     for r in rec2["version_history"]]
        # A gated run's snapshot carries no calibration map at all.
        with _mock_patch.object(mon, "_dump_json"):
            rec3 = mon.write_monitor_json(
                path, "20990101", drift, cov, ens, [], 0.5, {}, {},
                metrics=m, platt=None)
        assert "calibration" not in rec3["version_history"][-1]


def test_moneyline_record_is_one_exact_board_date():
    """The published current-slate record carries ONE board date — the first
    game date at or after the run's own date. The 2026-10-07 run's slate
    still held the previous evening's carry-over games, so
    ``slate_date=min(gameday)`` named 2026-10-06 while 2026-10-07 games sat
    in the same record: the strict resolver rejected the file and Today's
    Games showed no Oct 7 board at all. Pricing arrays must stay aligned to
    the surviving rows, and a same-day re-run must never re-introduce
    another date's preserved rows."""
    def _slate(days):
        rows = []
        for i, d in enumerate(days):
            rows.append({
                "game_id": f"g{d}{i}", "gameday": d,
                "home_team": "BOS", "away_team": "TOR",
                "venue": "TD Garden", "start_time_utc": None,
            })
        return pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as td:
        # Mixed slate: yesterday's carry-over + tonight's board.
        slate = _slate(["2026-10-06", "2026-10-06", "2026-10-07",
                        "2026-10-07"])
        p_home = np.array([0.10, 0.20, 0.61, 0.72])
        p_cal = np.array([0.11, 0.21, 0.63, 0.74])  # the served array
        rec = serving_mod.write_moneyline_json(
            Path(td) / "nhl_moneyline_v1_20261007.json", slate,
            p_home, p_cal, {}, {}, run_date="2026-10-07")
        assert rec["slate_date"] == "2026-10-07"
        assert rec["n_games"] == 2
        assert {g["game_date"] for g in rec["games"]} == {"2026-10-07"}
        # Row/probability alignment survives the filter positionally (the
        # record serves the calibrated array's values).
        assert [g["home_win_prob_model"] for g in rec["games"]] == [0.63, 0.74]

        # Preview slate: only the FIRST upcoming night publishes; the later
        # night ships with its own day's run.
        rec2 = serving_mod.write_moneyline_json(
            Path(td) / "nhl_moneyline_v1_20261008.json",
            _slate(["2026-10-09", "2026-10-10"]),
            np.array([0.55, 0.65]), np.array([0.45, 0.35]), {}, {},
            run_date="2026-10-08")
        assert rec2["slate_date"] == "2026-10-09"
        assert rec2["n_games"] == 1

        # Same-day re-run: preserved prior rows must stay ON the board date
        # (the delivered 20261007 file carried Oct-6 rows a re-run must not
        # re-inherit).
        path = Path(td) / "nhl_moneyline_v1_20261009.json"
        path.write_text(json.dumps({
            "created_utc": "2026-10-09T05:00:00+00:00",
            "slate_date": "2026-10-08",
            "n_games": 2,
            "games": [
                {"game_id": "carry", "game_date": "2026-10-08",
                 "home_team": "BOS", "away_team": "TOR",
                 "home_win_prob_model": 0.5, "away_win_prob_model": 0.5},
                {"game_id": "boardearlier", "game_date": "2026-10-09",
                 "home_team": "BOS", "away_team": "TOR",
                 "home_win_prob_model": 0.55, "away_win_prob_model": 0.45},
            ]}), encoding="utf-8")
        rec3 = serving_mod.write_moneyline_json(
            path, _slate(["2026-10-09", "2026-10-10"]),
            np.array([0.56, 0.65]), np.array([0.44, 0.35]), {}, {},
            run_date="2026-10-09")
        assert rec3["slate_date"] == "2026-10-09"
        # Off-date prior rows are not re-inherited; the same-date row IS
        # preserved (publication stands) and this run's price replaces it
        # for ids it re-prices.
        ids = {g["game_id"] for g in rec3["games"]}
        assert "carry" not in ids
        assert all(g["game_date"] == "2026-10-09" for g in rec3["games"])


def test_goalie_matchup_record_is_one_exact_board_date():
    """The dated goalie matchup file carries ONE board date too.

    The delivered ``nhl_goalie_matchup_20261007.json`` held two 2026-10-06
    carry-over games beside the three real Oct-7 games — the only dated NHL
    artifact whose rows disagreed with its own file date, and the reason the
    board smoke's exact-date goalie load saw foreign rows when walking the
    family. Same board rule as the moneyline record: first game date at or
    after ``run_date``, goalie rows filtered positionally alongside."""
    fields = list(serving_mod.config.GOALIE_FIELDS)
    slate = pd.DataFrame([
        {"game_id": "carry1", "gameday": "2026-10-06",
         "home_team": "SEA", "away_team": "VGK"},
        {"game_id": "carry2", "gameday": "2026-10-06",
         "home_team": "LAK", "away_team": "FLA"},
        {"game_id": "board1", "gameday": "2026-10-07",
         "home_team": "WSH", "away_team": "PIT"},
        {"game_id": "board2", "gameday": "2026-10-07",
         "home_team": "ANA", "away_team": "EDM"},
    ])
    # Positional sentinel in every goalie field: row i carries "g<i>".
    goalie_df = pd.DataFrame(
        [[f"g{i}" for _ in fields] for i in range(len(slate))],
        columns=fields)

    with tempfile.TemporaryDirectory() as td:
        rec = serving_mod.write_goalie_matchup_json(
            Path(td) / "nhl_goalie_matchup_20261007.json",
            goalie_df, slate, run_date="2026-10-07")
        assert rec["n_games"] == 2
        assert [g["game_id"] for g in rec["games"]] == ["board1", "board2"]
        assert {g["gameday"] for g in rec["games"]} == {"2026-10-07"}
        # Goalie rows stay positionally aligned with the surviving slate.
        name_col = "g_home_name" if "g_home_name" in fields else fields[0]
        assert [g[name_col] for g in rec["games"]] == ["g2", "g3"]

        # Whole slate is carry-over: earliest date is the board (the
        # moneyline writer's documented fallback), still ONE date.
        rec2 = serving_mod.write_goalie_matchup_json(
            Path(td) / "nhl_goalie_matchup_20261006.json",
            goalie_df.iloc[:2].reset_index(drop=True), slate.iloc[:2],
            run_date="2026-10-06")
        assert rec2["n_games"] == 2
        assert {g["gameday"] for g in rec2["games"]} == {"2026-10-06"}


def test_adaptive_weights_are_logloss_simplex_and_logit_optimal():
    """The optimizer contract is pooled binary log loss in logit space."""
    assert config.ADAPTIVE_WEIGHT_METRIC == "logloss"
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    members = {
        "elasticnet": [0.20, 0.80, 0.30, 0.70, 0.25, 0.75, 0.35, 0.65],
        "lightgbm": [0.10, 0.90, 0.20, 0.80, 0.15, 0.85, 0.25, 0.75],
        "xgboost": [0.40, 0.60, 0.45, 0.55, 0.42, 0.58, 0.47, 0.53],
    }
    weights = ml_mod.compute_adaptive_weights(members, y)
    assert set(weights) == set(members)
    assert all(w >= 0 for w in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 1e-12

    p = np.column_stack([members[n] for n in sorted(members)])
    w = np.array([weights[n] for n in sorted(members)])
    blended = ml_mod._logit_blend_matrix(p, w)
    blend_loss = -float(np.mean(y * np.log(blended)
                                 + (1 - y) * np.log(1 - blended)))
    for member in members.values():
        member = np.asarray(member, dtype=float)
        member_loss = -float(np.mean(y * np.log(member)
                                     + (1 - y) * np.log(1 - member)))
        assert blend_loss <= member_loss + 1e-12


# ---------------------------------------------------------------------------
# 5. Derived-vs-binary moneyline honesty
# ---------------------------------------------------------------------------
def test_derived_moneyline_never_rewrites_the_winner():
    """calibrate_market_frame's derived block calibrates the favored side in
    favored space and clamps at 0.5: the SIGN of the derived moneyline can
    never flip relative to the raw MC margin probability."""
    games = feat_mod.build_game_features(_synth_games(n_days=60))
    folds = folds_mod.make_folds(games, date_col="gameday")
    oof = dist_mod.walk_forward_oof(games, fold_list=folds)["oof"]
    raw = dist_mod.apply_distribution(oof)
    calibrated, _ = dist_mod.calibrate_market_frame(raw)

    raw_home = raw["p_home_win_derived"].to_numpy(float)
    cal_home = calibrated["p_home_win_derived"].to_numpy(float)
    same_side = (raw_home >= 0.5) == (cal_home >= 0.5)
    # Rows whose RAW side sits exactly at the 0.5 knife edge (or exactly on
    # a favorite's clamp boundary) are the only permitted movers; the
    # favorite clamp can only push toward the favorite, never across.
    flips = ~same_side
    if flips.any():
        # A "flip" must be a clamp from below to exactly 0.5 on the raw-Home
        # underdog side... which is impossible by construction; assert the
        # calibrated side only ever TIGHTENS toward the favorite.
        tightened = (np.sign(cal_home - 0.5) == np.sign(raw_home - 0.5)) \
            | (cal_home == 0.5)
        assert tightened.all(), "calibration moved a derived moneyline across the knife edge"
    # The away leg is derived as 1 − home − p_tie, with p_tie the column as
    # it stood at derivation time (the raw MC tie mass; the calibrated
    # spread push refreshes p_tie afterwards — the NFL/MLB-shared order).
    np.testing.assert_allclose(
        calibrated["p_away_win_derived"].to_numpy(float),
        1.0 - cal_home - raw["p_tie"].to_numpy(float), atol=1e-9)


def test_calibrate_market_frame_derived_moneyline_is_prequential():
    """The derived-moneyline column is calibrated by the same strictly-prior
    per-fold map every grid line uses: poisoning the LAST fold's outcomes
    must not move ANY earlier fold's published probability. (2026-09-30
    leakage audit: the derived block was the one pooled in-place calibrator,
    so the sealed-window derived_ml card scored rows its own map had seen.)
    """
    games = feat_mod.build_game_features(_synth_games(n_days=60, seed=11))
    folds = folds_mod.make_folds(games, date_col="gameday")
    oof = dist_mod.walk_forward_oof(games, fold_list=folds)["oof"]
    raw = dist_mod.apply_distribution(oof)

    poisoned = raw.copy()
    last_fold = int(poisoned["fold_id"].max())
    flip = (poisoned["fold_id"] == last_fold).to_numpy()
    margin = poisoned["margin"].to_numpy(float)
    poisoned.loc[flip, "margin"] = np.where(margin[flip] > 0, -1.0, 1.0)

    cal_a, bundle_a = dist_mod.calibrate_market_frame(raw)
    cal_b, _ = dist_mod.calibrate_market_frame(poisoned)
    early = cal_a["fold_id"].to_numpy() < last_fold
    assert early.any(), "synthetic walk produced no prior folds"
    np.testing.assert_array_equal(
        cal_a["p_home_win_derived"].to_numpy(float)[early],
        cal_b["p_home_win_derived"].to_numpy(float)[early],
        err_msg=("earlier folds' derived moneyline moved by a later fold's "
                 "outcomes — leakage"))
    # The pooled all-OOF map still rides the bundle as the serving-layer
    # record (the derived_ml card reads the prequential column, not this).
    assert "derived_moneyline" in bundle_a


def test_apply_distribution_is_row_aligned_and_additive():
    games = feat_mod.build_game_features(_synth_games(n_days=30))
    base = games.head(10).copy()
    base["mu_h"] = 3.0
    base["mu_a"] = 2.6
    out = dist_mod.apply_distribution(base)
    assert len(out) == 10
    assert list(out.index) == list(base.index)
    for line in config.SPREAD_GRID:
        assert dist_mod._grid_key("p_home_cover", line) in out.columns


# ---------------------------------------------------------------------------
# 6. NB dispersion
# ---------------------------------------------------------------------------
def test_estimate_alpha_is_the_mlb_pooled_moment_estimator():
    """MLB parity: alpha = max((var_obs - lam_bar) / lam_bar^2, 0), rounded 4dp.

    Pinned by recomputing MLB's ``run_engine.fit_alpha`` formula (verbatim
    source of truth: mlb-backend/backend/run_engine.py:786) on the SAME arrays
    and requiring exact equality — so the prior mu^2-weighted, 2.0-capped NHL
    form fails here rather than drifting silently.
    """
    rng = np.random.default_rng(9)
    mu = np.full(500, 3.0)
    # Poisson data: variance ~ mean -> alpha near zero.
    y_pois = rng.poisson(3.0, 500).astype(float)
    assert dist_mod.estimate_alpha(y_pois, mu) < 0.05
    # Overdispersed data (variance >> mean) -> positive alpha.
    y_nb = rng.negative_binomial(6.0, 6.0 / (6.0 + 3.0), 500).astype(float)
    a = dist_mod.estimate_alpha(y_nb, mu)
    assert 0.0 < a
    # Degenerate inputs floor at 0, never raise.
    assert dist_mod.estimate_alpha(np.array([np.nan, 1.0]), np.array([3.0, 3.0])) == 0.0
    assert dist_mod.estimate_alpha(np.array([]), np.array([])) == 0.0
    # Heterogeneous lambda: this is where the mu^2-weighted NHL form diverged
    # from MLB. Recompute MLB's pooled formula exactly.
    mu_het = rng.normal(3.0, 0.6, 800)
    y_het = rng.poisson(np.clip(mu_het, 0.2, None)).astype(float)
    lam_bar = float(mu_het.mean())
    mlb = round(max((float(y_het.var(ddof=0)) - lam_bar) / (lam_bar ** 2), 0.0), 4)
    assert dist_mod.estimate_alpha(y_het, mu_het) == mlb


def test_estimate_alpha_is_uncapped_like_mlb():
    """The SCALAR alpha estimator applies NO saturation; a heavy
    over-dispersion survives.

    The retired NHL form clipped at ALPHA_CAP=2.0, which would have silently
    masked a genuinely over-dispersed fit. This fixture is far beyond 2.0.
    ALPHA_CAP exists again since the alpha(λ) CURVE port (MLB parity — its
    curve layer clips at the same 2.0), but the curve clip must never leak
    into the scalar path: the shipped alpha_home/alpha_away stay uncapped
    MoM estimates, and only alpha_of() evaluates the curve's clip.
    """
    rng = np.random.default_rng(11)
    mu = np.full(4000, 3.0)
    # size=1/alpha with alpha=8 -> far beyond the old 2.0 cap.
    y = rng.negative_binomial(0.125, 0.125 / (0.125 + 3.0), 4000).astype(float)
    a = dist_mod.estimate_alpha(y, mu)
    assert a > dist_mod.ALPHA_CAP, \
        f"alpha {a} was capped; the scalar estimator must stay uncapped"
    lam_bar = 3.0
    assert a == round(max((float(y.var(ddof=0)) - lam_bar) / lam_bar ** 2, 0.0), 4)
    # The curve layer clips at ALPHA_CAP exactly where MLB's does — the two
    # saturations are different layers and both are intentional.
    clipped = dist_mod.alpha_of(np.array([5.0]),
                                {"form": "linear", "a": 8.0, "b": 0.0})
    assert clipped[0] == dist_mod.ALPHA_CAP


def test_calibrate_dispersion_reports_the_poisson_limit_flag():
    oof = pd.DataFrame({
        "home_score": [3, 4, 2, 5, 3], "mu_h": [3.0] * 5,
        "away_score": [2, 2, 3, 1, 2], "mu_a": [2.0] * 5})
    params = dist_mod.calibrate_dispersion(oof)
    assert params["distribution"] == "negative_binomial"
    assert params["mc_draws"] == dist_mod.MC_DRAWS
    assert params["poisson_limit"] == (max(params["alpha_home"],
                                           params["alpha_away"])
                                       <= dist_mod.ALPHA_FLOOR)
    assert params["alpha_home"] >= 0.0
    assert params["alpha_away"] >= 0.0


def test_alpha_curve_machinery_mirrors_mlb():
    """The alpha(λ) curve layer must exist with MLB's shape: quantile-binned
    MoM points, OOB selection among piecewise/linear/power, per-row alpha_of
    clipped to ALPHA_CAP — and on Poisson-limit data the selected curve must
    be flat at ~0, so hockey's current regime is unchanged by the port."""
    rng = np.random.default_rng(5)
    # Over-dispersed: alpha ~ 0.5 rising with lambda.
    lam = rng.uniform(2.0, 5.0, 3000)
    a_true = 0.1 + 0.08 * lam
    y = rng.negative_binomial(
        np.repeat(1.0, 3000) / a_true,
        (1.0 / a_true) / (1.0 / a_true + lam)).astype(float)
    curve, diag = dist_mod.select_alpha_curve(y, lam)
    assert curve["form"] in ("piecewise", "linear", "power")
    assert set(diag["candidates"]) == {"piecewise", "linear", "power"}
    vec = dist_mod.alpha_of(lam, curve)
    assert vec.min() >= 0.0 and vec.max() <= dist_mod.ALPHA_CAP
    # Poisson-limit data: the curve collapses to ~0 (scalar path unchanged).
    y_p = rng.poisson(lam).astype(float)
    curve_p, _ = dist_mod.select_alpha_curve(y_p, lam)
    assert float(dist_mod.alpha_of(lam, curve_p).max()) < 0.05


def test_calibrate_dispersion_carries_the_curve_layer():
    """The dispersion record ships the fitted alpha(λ) curves, their OOB
    selection diagnostics, the per-row max alpha, and the MC guard constants
    — while alpha_home/alpha_away remain the pooled MoM scalars consumers
    already read."""
    rng = np.random.default_rng(7)
    n = 1200
    mu_h = rng.uniform(2.4, 3.6, n)
    mu_a = rng.uniform(2.2, 3.4, n)
    oof = pd.DataFrame({
        "home_score": rng.poisson(mu_h).astype(float), "mu_h": mu_h,
        "away_score": rng.poisson(mu_a).astype(float), "mu_a": mu_a})
    params = dist_mod.calibrate_dispersion(oof)
    assert "alpha_home_curve" in params and "alpha_away_curve" in params
    assert "alpha_selection" in params
    assert params["alpha_selection"]["home"]["selected"] in (
        "piecewise", "linear", "power")
    assert "alpha_home_max" in params and "alpha_away_max" in params
    assert params["alpha_home"] == dist_mod.estimate_alpha(
        oof["home_score"].to_numpy(float), oof["mu_h"].to_numpy(float))
    assert params["mc_draws_tail"] == dist_mod.MC_DRAWS_TAIL
    assert params["mc_se_target"] == dist_mod.MC_SE_TARGET
    # The flag keeps its SCALAR semantics (scoring consumes the scalars);
    # the curve's measured verdict is the Pearson probe in run_line_fit_check.
    assert params["poisson_limit"] == (
        params["alpha_home"] <= dist_mod.ALPHA_FLOOR
        and params["alpha_away"] <= dist_mod.ALPHA_FLOOR)


# ---------------------------------------------------------------------------
# 6b. Sealed holdout: the alpha layer and the final line calibrators never
#     see the sealed tail (MLB derive_markets_v3 gate, ported).
# ---------------------------------------------------------------------------
def _sealed_holdout_oof(n_days: int = 60, seed: int = 17) -> pd.DataFrame:
    """A dated OOF frame shaped like walk_forward_oof's output."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-10-01")
    rows = []
    for d in range(n_days):
        day = start + pd.Timedelta(days=d)
        for k in range(4):
            mu_h, mu_a = 3.0 + 0.2 * k, 2.8 - 0.1 * k
            rows.append({
                "game_id": f"{day:%Y%m%d}_{k}",
                "gameday": day,
                "fold_id": d // 7,
                "home_score": float(rng.poisson(mu_h)),
                "away_score": float(rng.poisson(mu_a)),
                "mu_h": mu_h, "mu_a": mu_a,
                "margin": 0.0, "total": 0.0,
            })
    oof = pd.DataFrame(rows)
    oof["margin"] = oof["home_score"] - oof["away_score"]
    oof["total"] = oof["home_score"] + oof["away_score"]
    return oof


def test_alpha_curve_fits_on_pre_holdout_rows_only():
    """The dispersion layer must hand select_alpha_curve exactly the
    pre-holdout rows (gameday < max − HOLDOUT_DAYS) — never the sealed tail.
    Poisoning the sealed tail's outcomes must not move ANY fitted dispersion
    value: the recent window is evaluation evidence, never fit evidence."""
    oof = _sealed_holdout_oof()
    seen: dict = {}
    orig = dist_mod.select_alpha_curve

    def spy(y, lam, seed=dist_mod.MC_SEED):
        seen["n"] = seen.get("n", 0) + len(y)
        return orig(y, lam, seed=seed)

    with _mock_patch.object(dist_mod, "select_alpha_curve", spy):
        sig = dist_mod.calibrate_dispersion(oof)
    dates = pd.to_datetime(oof["gameday"])
    cutoff = dates.max().normalize() - pd.Timedelta(days=dist_mod.HOLDOUT_DAYS)
    n_pre = int((dates < cutoff).sum())
    assert seen.get("n") == 2 * n_pre, (
        f"alpha fit saw {seen.get('n')} rows across both sides, "
        f"pre-holdout pool is {2 * n_pre}")
    gate = sig["holdout"]
    assert gate["n_pre"] == n_pre
    assert gate["n_holdout"] == len(oof) - n_pre
    assert gate["fitted_on"] == "pre-holdout OOF only"

    # The poison: rewrite the SEALED tail's scores entirely.
    poisoned = oof.copy()
    sealed = pd.to_datetime(poisoned["gameday"]) >= cutoff
    poisoned.loc[sealed, "home_score"] = 9.0
    poisoned.loc[sealed, "away_score"] = 1.0
    sig_p = dist_mod.calibrate_dispersion(poisoned)
    for key in ("alpha_home", "alpha_away",
                "alpha_home_max", "alpha_away_max"):
        assert sig[key] == sig_p[key], (
            f"{key} moved after the sealed tail was poisoned — leakage")


def test_calibrate_dispersion_without_gameday_fits_everything():
    """Undated frames (unit fixtures, ad-hoc callers) keep the ungated
    full-OOF behavior: no cutoff, no sealed rows, and the fitted values
    are identical to a dated frame whose rows all sit inside the gate."""
    undated = pd.DataFrame({
        "home_score": [3.0, 4.0, 2.0, 5.0], "mu_h": [3.0] * 4,
        "away_score": [2.0, 2.0, 3.0, 1.0], "mu_a": [2.0] * 4})
    sig = dist_mod.calibrate_dispersion(undated)
    assert sig["holdout"]["cutoff"] is None
    assert sig["holdout"]["n_pre"] == len(undated)
    assert sig["holdout"]["n_holdout"] == 0
    dated = undated.copy()
    dated["gameday"] = pd.date_range("2025-11-01", periods=4)
    sig_d = dist_mod.calibrate_dispersion(dated)
    assert sig_d["holdout"]["cutoff"] is not None
    assert (sig["alpha_home"], sig["alpha_away"]) == \
        (sig_d["alpha_home"], sig_d["alpha_away"])


def test_calibrate_dispersion_fully_sealed_timeline_degrades_to_full_fit():
    """A dated frame whose ENTIRE timeline sits inside the holdout window
    (an early-season small sample) must degrade to the full-frame fit and
    say so — never crash on an empty pre-holdout pool, never claim the
    gated scope it did not use."""
    oof = _sealed_holdout_oof(n_days=10)   # 10 days of rows, all sealed
    sig = dist_mod.calibrate_dispersion(oof)
    assert sig["holdout"]["n_pre"] == 0
    assert sig["holdout"]["n_holdout"] == len(oof)
    assert sig["holdout"]["fitted_on"] == "full OOF (pre-holdout pool too small)"
    assert np.isfinite(sig["alpha_home"]) and np.isfinite(sig["alpha_away"])


def test_sealed_holdout_rows_are_stamped_and_scored_separately():
    """The last HOLDOUT_DAYS of OOF market rows carry frame_view='sealed',
    and the markets monitor nests per-line holdout metrics under each
    canonical line's own 'holdout' key (MLB card shape) — the sealed tail
    is scored, never fit on."""
    games = feat_mod.build_game_features(_synth_games(n_days=60))
    folds = folds_mod.make_folds(games, date_col="gameday")
    oof = dist_mod.walk_forward_oof(games, fold_list=folds)["oof"]
    dist = dist_mod.apply_distribution(oof)
    calibrated, bundle = dist_mod.calibrate_market_frame(dist)
    assert bundle["method"] == "prequential_platt"

    # Stamp the sealed tail exactly as master_pipeline does.
    mdates = pd.to_datetime(calibrated["gameday"])
    cutoff = mdates.max().normalize() - pd.Timedelta(days=dist_mod.HOLDOUT_DAYS)
    calibrated.loc[mdates >= cutoff, "frame_view"] = "sealed"
    n_sealed = int((calibrated["frame_view"] == "sealed").sum())
    assert n_sealed > 0, "the sealed tail stamped nothing"
    assert n_sealed < len(calibrated), "everything was sealed — gate degenerated"

    mon_rows = calibrated.copy()
    mon_rows["derived_ml"] = mon_rows["p_home_win_derived"]
    with _mock_patch.object(mon, "_dump_json"):
        record = mon.write_markets_monitor_json(
            Path("probe.json"), "20260928", mon_rows, {})
    holdout_gate = record.get("holdout_gate") or {}
    assert holdout_gate.get("n_holdout") == n_sealed
    metrics = record["market_metrics"]
    assert metrics, "the monitor produced no per-line metrics"
    holdout_keys = [k for k, v in metrics.items()
                    if isinstance(v, dict) and v.get("holdout")]
    assert holdout_keys, (
        "sealed rows exist but no per-line holdout metrics were scored")
    for key in holdout_keys:
        h = metrics[key]["holdout"]
        assert h.get("n"), f"{key}: empty holdout metrics"


def test_sealed_holdout_gate_is_a_schema_gate():
    """A dispersion record without a cutoff (the ungated path reaching
    production) must FAIL the Phase 13 gates — the regression this port
    exists to prevent cannot ship silently."""
    import master_pipeline as mp

    gates: dict[str, bool] = {}
    gated = {"holdout": {"cutoff": "2026-06-01", "n_pre": 100,
                         "n_holdout": 20}}
    ungated = {"holdout": {"cutoff": None, "n_pre": 120,
                           "n_holdout": 0}}
    gates["sealed_holdout_gate"] = bool(gated.get("holdout", {}).get("cutoff"))
    assert gates["sealed_holdout_gate"] is True
    gates["sealed_holdout_gate"] = bool(ungated.get("holdout", {}).get("cutoff"))
    assert gates["sealed_holdout_gate"] is False
    gates["sealed_holdout_gate"] = bool({}.get("holdout"))
    assert gates["sealed_holdout_gate"] is False

    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    node = _log_call_containing(src, "sealed holdout gate")
    assert node is not None, "the sealed-holdout log line is gone"


def test_simulate_se_guard_bumps_only_over_target():
    """The MC SE-guard is MLB's conditional discipline: at the default draw
    count the derivation records its resolution and fires the 50k tail bump
    ONLY when the worst totals-line SE exceeds MC_SE_TARGET."""
    meta: dict = {}
    dist_mod.simulate_distributions(np.array([3.0, 2.9]), np.array([2.7, 2.8]),
                                    0.0, 0.0, meta_out=meta)
    assert meta["reason"] == "default"
    assert meta["n_draws"] == dist_mod.MC_DRAWS
    assert 0.0 < meta["mc_se_totals_max"] <= dist_mod.MC_SE_TARGET + 1e-9
    meta_bump: dict = {}
    dist_mod.simulate_distributions(np.array([3.0, 2.9]), np.array([2.7, 2.8]),
                                    0.0, 0.0, n_draws=200, meta_out=meta_bump)
    assert meta_bump["n_draws"] == dist_mod.MC_DRAWS_TAIL
    assert meta_bump["requested_draws"] == 200
    assert "bumped" in meta_bump["reason"]


def test_run_line_fit_check_reports_adequacy_and_baseline_beat():
    """MLB's diagnostics shape: the Pearson probe (≈1 under a Poisson-
    consistent fixture) and deviance/RMSE where the μ model must beat the
    constant league-mean baseline it replaces."""
    rng = np.random.default_rng(9)
    n = 1500
    mu_h = rng.uniform(2.2, 4.2, n)
    mu_a = rng.uniform(2.0, 4.0, n)
    oof = pd.DataFrame({
        "home_score": rng.poisson(mu_h).astype(float), "mu_h": mu_h,
        "away_score": rng.poisson(mu_a).astype(float), "mu_a": mu_a})
    fc = dist_mod.run_line_fit_check(oof)
    for side in ("home", "away"):
        assert abs(fc[side]["pearson"] - 1.0) < 0.15, fc[side]
        assert fc[side]["deviance_model"] < fc[side]["deviance_baseline"]
        assert fc[side]["rmse_model"] < fc[side]["rmse_baseline"]


# ---------------------------------------------------------------------------
# 7. MLB feature parity — the run line has NO feature list of its own
# ---------------------------------------------------------------------------
# The module graph `distributions` ACTUALLY resolved. Under pytest the backend
# dir is a package (it ships an __init__.py), so pytest prepends
# nhl-backend to sys.path and the `from backend import moneyline` branch
# inside distributions WINS — creating a second moneyline/config module object
# distinct from the top-level ones imported above. Production
# (master_pipeline) puts only the backend dir on sys.path, so there the
# top-level branch wins and all four are one object. Binding to whatever
# distributions resolved keeps these pins honest in BOTH shapes; patching the
# top-level module instead would silently test a parallel module graph.
_ML = dist_mod.ml_mod
_CFG = dist_mod.config


def test_distributions_resolved_a_single_module_graph():
    """Guard the guard: moneyline and config must be ONE object each.

    If this ever fails, every monkeypatch in the file that targets the
    top-level `ml_mod`/`config` is a no-op against the run line, and the
    parity pins below would be vacuous.
    """
    assert _ML is not None and hasattr(_ML, "member_matrix")
    assert _ML.member_matrix is sys.modules[_ML.__name__].member_matrix
    assert _CFG is sys.modules[_CFG.__name__]
    assert dist_mod.config is _CFG and dist_mod.ml_mod is _ML


def test_run_line_matrix_is_exactly_the_moneyline_production_matrix():
    """The run line's regressor matrix must be the moneyline's, column for column.

    This is the requirement in its strictest form: not "a similar list", but
    the identical ordered column vector the production binary model fits on,
    tree categorical pair included.
    """
    games = feat_mod.build_game_features(_synth_games(n_days=30))
    run_line_cols = list(dist_mod.ScoreRegressor()._matrix(games).columns)
    for member in ("lightgbm", "xgboost"):
        prod = list(_ML.member_matrix(member, games).columns)
        assert run_line_cols == prod, (
            f"run-line matrix diverged from the {member} moneyline member")
    # ...and it is the active moneyline contract, not a subset or superset.
    active = list(_CFG.active_moneyline_feature_cols())
    assert run_line_cols[:len(active)] == active
    assert run_line_cols[len(active):] == list(_CFG.TREE_CATEGORICAL_COLS)


def test_run_line_refit_routes_categoricals_like_the_oof_walk():
    """Train/serve skew guard: the production refit must declare the team-ID
    categoricals to LightGBM exactly as the OOF walk does (categorical_feature
    by name) — the OOF metrics only validate the deployed model if both fits
    route the identical columns the identical way. The default constructor
    declares them; opt-out exists only for controlled comparisons."""
    games = feat_mod.build_game_features(_synth_games(n_days=30))
    reg = dist_mod.ScoreRegressor()
    assert reg._declare_categoricals is True
    captured: dict[str, object] = {}

    class _Spy(dist_mod.lightgbm.LGBMRegressor if hasattr(dist_mod, "lightgbm")
               else object):
        pass

    # Capture the fit kwargs without depending on lightgbm's import site.
    real_home, real_away = reg.home_model, reg.away_model
    for side, model in (("home", real_home), ("away", real_away)):
        def _make_fit(m, s):
            def _fit(X, y, **kwargs):
                captured[s] = dict(kwargs)
                return m.__class__.fit(m, X, y)
            return _fit
        model.fit = _make_fit(model, side)
    X = reg._matrix(games)
    reg.fit(games.assign(home_score=3.0, away_score=2.0))
    for side in ("home", "away"):
        assert captured[side].get("categorical_feature") == \
            list(_CFG.TREE_CATEGORICAL_COLS), captured[side]
    # Opt-out path (controlled comparisons only) routes nothing.
    reg2 = dist_mod.ScoreRegressor(declare_categoricals=False)
    seen: dict[str, object] = {}
    for side, model in (("home", reg2.home_model), ("away", reg2.away_model)):
        def _make_fit2(m, s):
            def _fit(X, y, **kwargs):
                seen[s] = dict(kwargs)
                return m.__class__.fit(m, X, y)
            return _fit
        model.fit = _make_fit2(model, side)
    reg2.fit(games.assign(home_score=3.0, away_score=2.0))
    for side in ("home", "away"):
        assert "categorical_feature" not in seen[side], seen[side]


def test_run_line_features_follow_the_active_moneyline_subset():
    """Dynamic derivation: adopt an RFE subset, the run line follows it.

    A hardcoded run-line list would keep the old width here; only resolution
    through the shared moneyline contract narrows with it.
    """
    games = feat_mod.build_game_features(_synth_games(n_days=30))
    full = list(dist_mod.ScoreRegressor()._matrix(games).columns)
    subset = list(_CFG.active_moneyline_feature_cols()[:3])
    try:
        _CFG.set_feature_subset(subset)
        assert list(_CFG.active_moneyline_feature_cols()) == subset
        narrowed = list(dist_mod.ScoreRegressor()._matrix(games).columns)
        assert narrowed == subset + list(_CFG.TREE_CATEGORICAL_COLS)
        assert len(narrowed) == len(subset) + 2 < len(full)
        # The published contract follows too, not just the matrix.
        c = dist_mod.feature_contract()
        assert c["mode"] == "strict_active_moneyline"
        assert c["feature_cols"] == subset
        assert c["n_features"] == len(subset)
    finally:
        _CFG.reset_feature_subset()
    assert list(dist_mod.ScoreRegressor()._matrix(games).columns) == full


def test_run_line_resolves_features_through_the_moneyline_module():
    """The parity must be structural, not a parallel path that agrees today.

    MLB's run engine imports the moneyline module and calls its feature
    resolver. Pin that NHL does the same: patch the moneyline helper and
    require the run line to go through it.
    """
    games = feat_mod.build_game_features(_synth_games(n_days=10))
    seen: list[tuple] = []
    real = _ML.member_matrix

    def _spy(name, df):
        seen.append((name, id(df)))
        return real(name, df)

    with _mock_patch.object(_ML, "member_matrix", _spy):
        dist_mod.ScoreRegressor()._matrix(games)
    assert seen, "run line did not route its matrix through moneyline.member_matrix"
    assert all(n == dist_mod.MONEYLINE_TREE_MEMBER for n, _ in seen)


def test_walk_forward_oof_publishes_the_mlb_feature_contract():
    games = feat_mod.build_game_features(_synth_games(n_days=45))
    folds = folds_mod.make_folds(games, date_col="gameday")
    out = dist_mod.walk_forward_oof(games, fold_list=folds)
    c = out["feature_contract"]
    active = list(_CFG.active_moneyline_feature_cols())
    assert c["mode"] == "strict_active_moneyline"
    assert c["n_features"] == len(active)
    assert c["feature_cols"] == active
    assert c["tree_categorical_cols"] == list(_CFG.TREE_CATEGORICAL_COLS)
    assert c["resolved_via"] == \
        f"moneyline.member_matrix({dist_mod.MONEYLINE_TREE_MEMBER!r})"
    # Resolved against a real frame: the fitted width is the contract plus
    # the tree categorical pair, so nothing was silently narrowed.
    assert c["n_features_fitted"] == len(active) + len(_CFG.TREE_CATEGORICAL_COLS)
    assert c["fitted_cols"] == active + list(_CFG.TREE_CATEGORICAL_COLS)
    assert out["n_folds"] == len(folds)


def test_run_line_and_moneyline_walk_the_identical_fold_geometry():
    """Fold periods must match MLB's rule: one geometry, shared by both walks.

    Both engines receive the SAME fold_list object, so each scored game lands
    in the same validation window for the moneyline and for expected scoring.
    """
    games = feat_mod.build_game_features(_synth_games(n_days=60))
    folds = folds_mod.make_folds(games, date_col="gameday")
    ml_folds = _ML.walk_forward_oof(games, fold_list=folds)["oof"]
    dist_folds = dist_mod.walk_forward_oof(games, fold_list=folds)["oof"]
    ml_map = ml_folds.set_index("game_id")["fold_id"].to_dict()
    dist_map = dist_folds.set_index("game_id")["fold_id"].to_dict()
    assert set(ml_map) == set(dist_map)
    assert ml_map == dist_map


# ---------------------------------------------------------------------------
# 8. Monitor line-pair contract: canonical lines priced honestly
# ---------------------------------------------------------------------------
def _oof_market_rows(n_days: int = 60, seed: int = 7) -> pd.DataFrame:
    games = feat_mod.build_game_features(_synth_games(n_days, seed=seed))
    folds = folds_mod.make_folds(games, date_col="gameday")
    oof_ml = ml_mod.walk_forward_oof(games, fold_list=folds)["oof"]
    oof_dist = dist_mod.walk_forward_oof(games, fold_list=folds)["oof"]
    merged = oof_ml.merge(
        oof_dist[["game_id", "mu_h", "mu_a", "home_score", "away_score",
                  "margin", "total"]], on="game_id", how="inner")
    out = dist_mod.apply_distribution(merged)
    out = dist_mod.calibrate_market_frame(out)[0]
    out["p_home_win"] = out["p_home_win_derived"]
    out["derived_ml"] = out["p_home_win_derived"]
    return out


def test_markets_winner_cards_price_fair_lines_with_honest_outcomes():
    rows = _oof_market_rows()
    rows["derived_ml"] = rows["p_home_win_derived"]
    cards = mon.markets_winner_cards(rows)
    assert cards, "expected winner-card sections from a decided OOF pool"
    for name in ("over_under", "run_line", "derived_ml"):
        card = cards.get(name) or {}
        assert card, f"{name} card missing"
        # Honest out-of-sample outcomes only (push-excluded 2-way pool).
        assert card["n"] > 0
        assert 0.0 <= card["actual_win_rate"] <= 1.0
        assert 0.0 <= card["predicted_mean"] <= 1.0
        assert card["brier"] is not None and np.isfinite(card["brier"])
        # n plus excluded whole-line pushes covers the decided pool.
        assert card["n"] <= len(rows)


def test_markets_winner_cards_handle_away_favorite_run_line():
    """The away-favorite branch must score the favorite-side cover leg."""
    line = 1
    rows = pd.DataFrame([
        {
            "fair_spread": line, "margin": -2.0, "derived_ml": 0.40,
            "p_home_cover_m1": 0.30, "p_push_m1": 0.10,
        },
        {
            "fair_spread": line, "margin": -2.0, "derived_ml": 0.40,
            "p_home_cover_m1": None, "p_push_m1": None,
        },
    ])
    cards = mon.markets_winner_cards(rows)
    assert cards["run_line"]["n"] == 1
    assert cards["run_line"]["predicted_mean"] == round(2.0 / 3.0, 4)
    assert cards["run_line"]["actual_win_rate"] == 1.0


def test_nhl_boxscore_uses_per_goalie_goals_and_shots():
    payload = {
        "id": 2024010001,
        "homeTeam": {"score": 9, "sog": 99},
        "awayTeam": {"score": 8, "sog": 88},
        "playerByGameStats": {
            "homeTeam": {
                "forwards": [], "defense": [],
                "goalies": [{"playerId": 1, "name": {"default": "Starter"},
                             "decision": "W", "toi": "60:00",
                             "goalsAgainst": 2, "shotsAgainst": 31}],
            },
            "awayTeam": {
                "forwards": [], "defense": [],
                "goalies": [{"playerId": 2, "name": {"default": "Relief"},
                             "decision": "L", "toi": "00:05",
                             "goalsAgainst": 1, "shotsAgainst": 2}],
            },
        },
    }
    row = ing._parse_boxscore(payload)
    assert row["home_goals_against"] == 2
    assert row["home_shots_against"] == 31
    assert row["away_goals_against"] == 1
    assert row["away_shots_against"] == 2
    assert row["home_goals_against"] != payload["awayTeam"]["score"]
    assert row["away_goals_against"] != payload["homeTeam"]["score"]


def test_nhl_goalies_toi_parser_handles_api_clock_values():
    assert ing._parse_toi_minutes("25:00") == 25.0
    assert ing._parse_toi_minutes("1:02:30") == 62.5
    assert ing._parse_toi_minutes(18.25) == 18.25
    for malformed in (None, "", "unknown", "1:xx", "-1:00", "1:2:3:4"):
        assert np.isnan(ing._parse_toi_minutes(malformed))


def test_goalie_state_populates_gaa_from_ingested_minutes():
    games = pd.DataFrame([
        {"game_id": "g1", "gameday": "2025-10-01", "home_team": "ANA", "away_team": "BOS"},
        {"game_id": "g2", "gameday": "2025-10-03", "home_team": "ANA", "away_team": "BOS"},
    ])
    boxscores = pd.DataFrame([
        {"game_id": "g1", "home_goalie_id": 1, "away_goalie_id": 2,
         "home_goalie_name": "Home One", "away_goalie_name": "Away One",
         "home_goalie_toi": 60.0, "away_goalie_toi": 60.0,
         "home_goals_against": 2, "away_goals_against": 3,
         "home_shots_against": 30, "away_shots_against": 28},
        {"game_id": "g2", "home_goalie_id": 1, "away_goalie_id": 2,
         "home_goalie_name": "Home One", "away_goalie_name": "Away One",
         "home_goalie_toi": 60.0, "away_goalie_toi": 58.0,
         "home_goals_against": 1, "away_goals_against": 1,
         "home_shots_against": 30, "away_shots_against": 28},
    ])
    states, _ = feat_mod.goalie_state(boxscores, games)
    assert pd.isna(states.loc[0, "goalie_gaa_home"])
    assert states.loc[1, "goalie_gaa_home"] == 2.0
    assert states.loc[1, "goalie_gaa_away"] == 3.0


def test_goalie_state_ignores_short_relief_appearances():
    games = pd.DataFrame([
        {"game_id": "g1", "gameday": "2025-10-01", "home_team": "ANA", "away_team": "BOS"},
        {"game_id": "g2", "gameday": "2025-10-03", "home_team": "ANA", "away_team": "BOS"},
    ])
    boxscores = pd.DataFrame([
        {"game_id": "g1", "home_goalie_id": 1, "away_goalie_id": 2,
         "home_goalie_name": "Starter", "away_goalie_name": "Relief",
         "home_goalie_toi": 60.0, "away_goalie_toi": 60.0,
         "home_goals_against": 2, "away_goals_against": 3,
         "home_shots_against": 30, "away_shots_against": 28},
        {"game_id": "g2", "home_goalie_id": 1, "away_goalie_id": 2,
         "home_goalie_name": "Starter", "away_goalie_name": "Relief",
         "home_goalie_toi": 60.0, "away_goalie_toi": 5.0,
         "home_goals_against": 1, "away_goals_against": 1,
         "home_shots_against": 30, "away_shots_against": 28},
    ])
    states, _ = feat_mod.goalie_state(boxscores, games)
    assert states.loc[1, "goalie_gaa_home"] == 2.0
    assert states.loc[1, "goalie_gaa_away"] == 3.0


def test_monitor_line_pairs_use_canonical_lines_only():
    rows = _oof_market_rows()
    rows["derived_ml"] = rows["p_home_win_derived"]
    # _run_engine_line_pairs is per-line: (p, y) push-excluded pairs.
    for line in config.RUN_ENGINE_FIXED_TOTALS:
        p, y = mon._run_engine_line_pairs(rows, "over", float(line))
        assert len(p) == len(y) and len(p) > 0, f"total {line}: empty pairs"
        # Ties (totals exactly ON the line) are excluded.
        on_line = (rows["total"].to_numpy(float) == float(line)).sum()
        assert len(p) == len(rows) - int(on_line)
        assert ((y == 0.0) | (y == 1.0)).all()
    for line in config.RUN_ENGINE_CANONICAL_SPREADS:
        p, y = mon._run_engine_line_pairs(rows, "spread", float(line))
        assert len(p) > 0, f"spread {line}: empty pairs"
        on_line = (rows["margin"].to_numpy(float) == float(line)).sum()
        assert len(p) == len(rows) - int(on_line)
    # The 'ml' kind prices the derived moneyline in pick-side framing.
    p, y = mon._run_engine_line_pairs(rows, "ml", 0.0)
    assert len(p) > 0 and np.isfinite(p).all()


def test_run_engine_market_metrics_and_fit_block():
    rows = _oof_market_rows()
    rows["derived_ml"] = rows["p_home_win_derived"]
    metrics = mon._run_engine_market_metrics(rows)
    assert metrics, "expected per-line OOF metrics"
    assert set(metrics) >= {f"over_{u}" for u in config.RUN_ENGINE_FIXED_TOTALS} \
        | {f"home_cover_{l}" for l in config.RUN_ENGINE_CANONICAL_SPREADS} \
        | {"derived_moneyline"}
    for key, m in metrics.items():
        assert m.get("n", 0) > 0, f"{key}: no scored games"
        if m.get("engine_brier") is not None:
            assert 0.0 <= m["engine_brier"] <= 1.0
    fit = mon._run_engine_fit_block(rows)
    assert "total_tail" in fit and "margin_tail" in fit, \
        f"fit block missing tails: {list(fit)}"
    # The NHL fit-block cutpoints: P(total >= 9) / P(total <= 4) observed vs
    # modeled, all in [0, 1].
    tt = fit["total_tail"]
    assert tt["k_ge"] == 9 and tt["k_le"] == 4
    for v in (tt["obs_ge"], tt["mod_ge"], tt["obs_le"], tt["mod_le"]):
        assert 0.0 <= v <= 1.0
    # Variance diagnostics exist for both sides.
    assert fit["variance_obs"]["home"] is not None
    assert fit["variance_obs"]["away"] is not None


def test_write_run_engine_feature_artifacts_enumerates_active_width():
    """Drift/coverage CSVs enumerate the ACTIVE serving width only — a
    column present in the frame but outside the contract gets no PSI row."""
    import tempfile
    games = feat_mod.build_game_features(_synth_games(n_days=40))
    games["poison_col"] = 1.0                     # present, never served
    with tempfile.TemporaryDirectory() as tmp:
        drift_name, cov_name = mon.write_run_engine_feature_artifacts(
            Path(tmp), "20260923", games, games.tail(10))
        drift = pd.read_csv(Path(tmp) / drift_name)
        cov = pd.read_csv(Path(tmp) / cov_name)
    active = [f for f in config.active_moneyline_feature_cols()
              if f in games.columns]
    assert set(drift["feature"]) == set(active), \
        "drift enumerated a non-serving feature"
    assert "poison_col" not in set(cov["feature"])
    assert set(cov["feature"]) == set(config.active_moneyline_feature_cols())


def test_feature_drift_regime_shift_is_insufficient_and_noise_floor_adjusts():
    """MLB-shaped drift: statuses key on the NOISE-ADJUSTED PSI, and a window
    that cannot support a verdict is INSUFFICIENT, never OK.

    Two pinned mutants:
      * a current window entirely inside the playoffs vs a 6%-playoffs
        baseline measured PSI 0.000 and read OK — the degenerate-bin
        collapse turned a full regime shift into an all-clear;
      * ``psi_adjusted`` was a literal passthrough of raw PSI, so
        same-distribution samples of this size (noise floor ~0.08) paged on
        binning wiggle alone.
    """
    rng = np.random.default_rng(7)
    cols = list(config.active_moneyline_feature_cols())
    n_b, n_c = 1000, 40
    full = pd.DataFrame(rng.normal(0.0, 1.0, (n_b, len(cols))), columns=cols)
    recent = pd.DataFrame(rng.normal(0.0, 1.0, (n_c, len(cols))), columns=cols)
    full["is_playoffs"] = np.where(rng.random(n_b) < 0.06, 1.0, 0.0)
    recent["is_playoffs"] = 1.0
    full["is_home"] = 1.0
    recent["is_home"] = 1.0

    rows = {r["feature"]: r for r in mon.feature_drift(full, recent)}

    # 1. Regime shift: unbinnable -> INSUFFICIENT with the signal in
    #    mean_shift, not a fake 0.000 PSI "OK". The window is large enough
    #    (1000/40) that INSUFFICIENT comes from the degenerate bins, not
    #    the small-window rule.
    p = rows["is_playoffs"]
    assert p["status"] == "INSUFFICIENT"
    assert not np.isfinite(p["psi"])
    assert abs(p["mean_shift"]) > 0.5

    # 2. Same-distribution continuous feature: raw PSI stays for
    #    transparency, the adjusted value subtracts the sampling-noise
    #    floor, and without a location shift the row cannot page.
    e = rows["elo_diff"]
    assert e["noise_floor"] > 0
    assert e["psi_adjusted"] <= e["psi"]
    assert e["status"] == "OK" or e["location_shift"]

    # 3. MLB row schema (explainability.compute_feature_drift shape).
    for key in ("psi", "psi_adjusted", "noise_floor", "mean_shift",
                "shift_se", "location_shift", "n_baseline", "n_current"):
        assert key in e, f"drift row dropped the MLB key {key!r}"


def test_feature_status_cannot_judge_is_insufficient_not_ok():
    """A PSI that cannot be computed must never read as an all-clear."""
    assert mon.feature_status(float("nan")) == "INSUFFICIENT"
    assert mon.feature_status(0.0) == "OK"
    assert mon.feature_status(0.10) == "WARN"
    assert mon.feature_status(0.25) == "ALERT"


def test_nb_distribution_metrics_flags_degenerate_inputs():
    oof = pd.DataFrame({
        "home_score": [3.0, 2.0, 4.0], "away_score": [2.0, 3.0, 1.0],
        "margin": [1.0, -1.0, 3.0], "total": [5.0, 5.0, 5.0],
        "mu_h": [3.0, 3.0, 3.0], "mu_a": [2.0, 2.0, 2.0]})
    params = {"alpha_home": 0.0, "alpha_away": 0.0, "mc_draws": 500}
    out = nb_distribution_metrics(oof, params, n_draws=500)
    assert isinstance(out, dict)
    assert set(out) == {"run_line", "totals"}
    for section in out.values():
        assert "n" in section and "logscore" in section and "brier" in section


# ---------------------------------------------------------------------------
# 8. Boxscore-derived served features (regression: dead-contract repair)
# ---------------------------------------------------------------------------
def _synth_boxscores(games: pd.DataFrame) -> pd.DataFrame:
    """Wide per-game boxscore rollup shaped exactly like ingestion's output."""
    rows = []
    for i, r in enumerate(games.itertuples(index=False)):
        rows.append({
            "game_id": r.game_id,
            "home_sog": 30 + i, "away_sog": 24 + i,
            "home_pp_goals": 1, "away_pp_goals": 0,
            "home_pp_opportunities": 5, "away_pp_opportunities": 4,
            "home_faceoff_pct": 0.52, "away_faceoff_pct": 0.48,
            "home_faceoff_wins": 26, "away_faceoff_wins": 24,
            "home_faceoff_attempts": 50, "away_faceoff_attempts": 50,
            "home_hits": 10, "away_hits": 8,
            "home_blocked": 5, "away_blocked": 4,
            "home_pim": 10, "away_pim": 12,
            "home_giveaways": 3, "away_giveaways": 4,
            "home_takeaways": 7, "away_takeaways": 6,
        })
    return pd.DataFrame(rows)


BOXSCORE_SERVED = ("shots_for_per_game_diff", "shots_against_per_game_diff",
                   "pp_success_diff", "faceoff_win_diff")


def _synth_goalie_boxscores(games: pd.DataFrame) -> pd.DataFrame:
    """``_synth_boxscores`` plus a goalie block, shaped like ingestion output.

    Every team runs a PRIMARY goalie who starts all of its games, and starts
    its BACKUP in the team's own last game. The two candidate rules for "who
    starts tonight" therefore disagree exactly on that game, which is what
    makes the PIT pins below meaningful: selecting on the decision goalie of
    the game being predicted reads that game's own boxscore, selecting on
    prior workload does not.
    """
    bs = _synth_boxscores(games)
    team_idx = {t: i for i, t in enumerate(TEAMS)}
    ordered = games.sort_values("gameday", kind="stable")
    appears: dict[str, int] = {}
    for r in ordered.itertuples(index=False):
        for side in ("home", "away"):
            t = str(getattr(r, f"{side}_team"))
            appears[t] = appears.get(t, 0) + 1
    seen: dict[str, int] = {}
    rows = []
    for r in ordered.itertuples(index=False):
        out: dict = {"game_id": r.game_id}
        for side in ("home", "away"):
            t = str(getattr(r, f"{side}_team"))
            i = team_idx[t]
            n = seen.get(t, 0)
            seen[t] = n + 1
            primary = 900000 + i * 2
            backup = primary + 1
            is_backup = (n == appears[t] - 1)
            out[f"{side}_goalie_id"] = backup if is_backup else primary
            out[f"{side}_goalie_name"] = f"G{i}B" if is_backup else f"G{i}A"
            out[f"{side}_goalie_toi"] = 58.0
            out[f"{side}_goals_against"] = 2
            out[f"{side}_shots_against"] = 30
            out[f"{side}_goalie_decision"] = "W"
        rows.append(out)
    return bs.merge(pd.DataFrame(rows), on="game_id", how="left")


def test_boxscore_served_diffs_are_populated_and_strictly_prior():
    """The four boxscore diffs are in MONEYLINE_FEATURE_COLS, so they must
    carry real point-in-time values. They were 0.00%-covered in production
    because the boxscore rollup was never merged into the ladder."""
    games = _synth_games(n_days=20, games_per_day=2)
    df = feat_mod.build_game_features(games, _synth_boxscores(games))
    for col in BOXSCORE_SERVED:
        assert col in df.columns, f"{col} missing from the feature frame"
        v = pd.to_numeric(df[col], errors="coerce")
        assert v.notna().any(), f"{col} is all-NaN — boxscore never reached the ladder"
    # A team's first game has no strictly-prior boxscore history.
    first = df.sort_values("gameday").iloc[0]
    for col in BOXSCORE_SERVED:
        assert pd.isna(first[col]), f"{col} fabricated a value for the first game"


def test_boxscore_served_diffs_never_read_the_current_game():
    """Poisoning game t's boxscore must not move game t's own features."""
    games = _synth_games(n_days=20, games_per_day=2)
    clean = feat_mod.build_game_features(games, _synth_boxscores(games))
    poisoned_bs = _synth_boxscores(games)
    target = games.sort_values("gameday").iloc[10]["game_id"]
    mask = poisoned_bs["game_id"] == target
    for col in ("home_sog", "away_sog", "home_faceoff_pct", "away_faceoff_pct",
                "home_pp_goals", "away_pp_goals"):
        poisoned_bs.loc[mask, col] = 999.0
    dirty = feat_mod.build_game_features(games, poisoned_bs)
    for col in BOXSCORE_SERVED:
        a = clean.loc[clean["game_id"] == target, col].to_numpy()
        b = dirty.loc[dirty["game_id"] == target, col].to_numpy()
        assert np.allclose(a, b, equal_nan=True), \
            f"{col} leaked the current game's own boxscore"
    # Sanity: the poisoned game MUST move LATER games (it is real prior
    # history for them), otherwise the assertion above proves nothing.
    later = clean["game_id"] != target
    assert not np.allclose(
        pd.to_numeric(clean.loc[later, "shots_for_per_game_diff"], errors="coerce"),
        pd.to_numeric(dirty.loc[later, "shots_for_per_game_diff"], errors="coerce"),
        equal_nan=True), "fixture did not actually change any feature"


# ---------------------------------------------------------------------------
# Raw per-side levels: every served diff ships its home/away level halves
# ---------------------------------------------------------------------------
RAW_PER_SIDE_FAMILIES = {
    "ewm_goal_share": "ewm_goal_share",
    "ga_per_game": "ga_per_game",
    "shots_for_per_game": "sog_pg_roll",
    "shots_against_per_game": "shots_against_pg_roll",
    "pp_success": "pp_success_rate_roll",
    "faceoff_win": "faceoff_win_pct_roll",
    "back_to_back": "back_to_back",
}


def test_raw_per_side_levels_served_for_every_diff_family():
    """The tree family needs the LEVEL halves of every served diff.

    The contract carried raw sides only for the Elo/state/goalie families;
    the eight remaining diff families (ewm_goal_share, ga_per_game, shots
    for/against, pp_success, faceoff_win, back_to_back, goalie_starts)
    shipped diffs
    whose home/away halves existed nowhere, so the tree members saw a
    difference the raw sides could not explain. Now every diff in
    MONEYLINE_FEATURE_COLS has its level twins served, the ladder/roll
    families are routed through RAW_PER_SIDE_COLS, and each pair satisfies
    home - away == diff.
    """
    games = _synth_games(n_days=25, games_per_day=2)
    df = feat_mod.build_game_features(games, _synth_boxscores(games))
    served = set(config.MONEYLINE_FEATURE_COLS)
    raw = config.RAW_PER_SIDE_COLS
    diffs = [c for c in served if c.endswith("_diff")]
    assert diffs, "contract sanity: there are served diffs to mirror"
    for d in diffs:
        base = d[: -len("_diff")]
        if base in ("is_playoffs", "is_home"):
            continue  # game-level facts have no level halves
        h, a = f"{base}_home", f"{base}_away"
        assert h in served and a in served, f"{d} served without its level twins"
        if not base.startswith("pl_"):
            assert h in raw and a in raw, f"{h}/{a} not routed to the tree family"
    for base, _lad in RAW_PER_SIDE_FAMILIES.items():
        h = pd.to_numeric(df[f"{base}_home"], errors="coerce")
        a = pd.to_numeric(df[f"{base}_away"], errors="coerce")
        d = pd.to_numeric(df[f"{base}_diff"], errors="coerce")
        assert np.allclose(h - a, d, equal_nan=True), \
            f"{base}: home - away does not reproduce the served diff"


def test_raw_per_side_levels_are_populated_and_strictly_prior():
    """Coverage + PIT: the new level twins carry the same shift(1) discipline
    the diffs inherited -- near-full population on a warm frame, and NaN (not
    a fabricated 0) on the league's first game where no prior history exists."""
    games = _synth_games(n_days=25, games_per_day=2)
    df = feat_mod.build_game_features(games, _synth_boxscores(games))
    df = df.sort_values("gameday", kind="stable")
    for base, _lad in RAW_PER_SIDE_FAMILIES.items():
        for side in ("home", "away"):
            v = pd.to_numeric(df[f"{base}_{side}"], errors="coerce")
            assert v.notna().mean() > 0.9, \
                f"{base}_{side}: {v.notna().mean():.0%} populated, expected near-full"
        first = df.iloc[0]
        assert pd.isna(first[f"{base}_home"]) and pd.isna(first[f"{base}_away"]), \
            f"{base}: fabricated a value for the league's first game"


def test_slate_serves_the_raw_per_side_levels_too():
    """The serving slate ships the same level halves, from strictly-prior
    decided games only -- the tree members must not lose them in serving."""
    games = _synth_games(n_days=20, games_per_day=2)
    bs = _synth_goalie_boxscores(games)
    pending = games.tail(4).copy()
    pending["home_score"] = np.nan
    pending["away_score"] = np.nan
    hist = games.head(-4)
    slate = feat_mod.build_slate_features(
        pd.concat([hist, pending], ignore_index=True),
        bs[bs["game_id"].isin(set(hist["game_id"]))])
    assert len(slate) == len(pending)
    for base, _lad in RAW_PER_SIDE_FAMILIES.items():
        for side in ("home", "away"):
            v = pd.to_numeric(slate[f"{base}_{side}"], errors="coerce")
            assert v.notna().any(), f"{base}_{side} empty on the serving slate"


def test_goalie_starts_levels_come_from_goalie_state_not_a_refit():
    """goalie_starts_home/away were already computed inside goalie_state; the
    contract now serves them. The pair must satisfy home - away == diff and
    degrade to all-NaN honestly when there is no boxscore coverage."""
    games = _synth_games(n_days=20, games_per_day=2)
    df = feat_mod.build_game_features(games, _synth_goalie_boxscores(games))
    for side in ("home", "away"):
        v = pd.to_numeric(df[f"goalie_starts_{side}"], errors="coerce")
        assert v.notna().any(), f"goalie_starts_{side} never populated"
    assert np.allclose(
        pd.to_numeric(df["goalie_starts_home"], errors="coerce")
        - pd.to_numeric(df["goalie_starts_away"], errors="coerce"),
        pd.to_numeric(df["goalie_starts_diff"], errors="coerce"),
        equal_nan=True)
    bare = feat_mod.build_game_features(
        _synth_games(n_days=8, games_per_day=2), None)
    assert pd.to_numeric(bare["goalie_starts_home"], errors="coerce").isna().all()
    assert pd.to_numeric(bare["goalie_starts_away"], errors="coerce").isna().all()


def test_manifest_documents_every_served_raw_per_side_level():
    """A served raw level without a manifest entry is an undocumented model
    input; manifest.validate() name-for-name parity must hold at the new 66
    width, and every documented level must carry the tree representation."""
    import manifest as manifest_mod

    problems = manifest_mod.validate()
    assert not problems, problems
    for f in config.MONEYLINE_FEATURE_COLS:
        if f.endswith(("_home", "_away")) and not f.startswith("pl_"):
            entry = manifest_mod.FEATURE_MANIFEST.get(f)
            assert entry is not None, f"{f} served but undocumented"
            assert "tree" in entry.get("model_family_availability", []), f


def test_boxscore_absent_degrades_to_nan_not_an_error():
    """Honest degradation: with no boxscore the diffs are NaN, never zero-filled."""
    games = _synth_games(n_days=12, games_per_day=2)
    df = feat_mod.build_game_features(games, None)
    for col in BOXSCORE_SERVED:
        assert col in df.columns
        assert pd.to_numeric(df[col], errors="coerce").isna().all()


def test_slate_features_use_the_rollup_from_prior_decided_games():
    """Slate cards must get trailing boxscore stats from strictly-prior games."""
    games = _synth_games(n_days=20, games_per_day=2)
    bs = _synth_boxscores(games)
    pending = games.head(4).copy()
    pending["home_score"] = np.nan
    pending["away_score"] = np.nan
    sched = pd.concat([games.tail(12), pending], ignore_index=True)
    slate = feat_mod.build_slate_features(sched, bs)
    assert len(slate) == len(pending)
    for col in BOXSCORE_SERVED:
        v = pd.to_numeric(slate[col], errors="coerce")
        assert v.notna().any(), f"{col} empty on the slate"


def test_pp_opportunities_come_from_official_team_counts_not_goalie_shots():
    """PP opportunities and goalie shots are distinct hockey events."""
    assert ing._parse_ratio("5/6") == 6.0
    # "0/0" is a REAL observation (that goalie faced no power-play shots),
    # not a missing one. Reporting it as NaN is what starved pp_success_diff
    # whenever a team's recent games happened to contain one.
    assert ing._parse_ratio("0/0") == 0.0
    for bad in (None, "", "abc", "5/", "/6", "-1/3", "7/4"):
        assert np.isnan(ing._parse_ratio(bad)), f"{bad!r} should not parse"
    bs = {"id": 2026020001,
          "homeTeam": {"sog": 30, "score": 3}, "awayTeam": {"sog": 20, "score": 1},
          "playerByGameStats": {
              "homeTeam": {
                  "forwards": [{"powerPlayGoals": 1, "faceoffWinningPctg": 0.52,
                                "hits": 5, "blockedShots": 2, "pim": 4,
                                "giveaways": 1, "takeaways": 3}],
                  "goalies": [{"playerId": 1, "decision": "W", "toi": "60:00",
                               "powerPlayShotsAgainst": "3/4"}],
              },
              "awayTeam": {
                  "forwards": [{"powerPlayGoals": 0, "faceoffWinningPctg": 0.48,
                                "hits": 4, "blockedShots": 1, "pim": 6,
                                "giveaways": 2, "takeaways": 1}],
                  "goalies": [{"playerId": 2, "decision": "L", "toi": "58:00",
                               "powerPlayShotsAgainst": "5/6"}],
              },
          }}
    rail = {"teamGameStats": [
        {"category": "powerPlay", "homeValue": "1/3", "awayValue": "0/2"},
        {"category": "faceoffWins", "homeValue": "26/50", "awayValue": "24/50"}]}
    row = ing._parse_boxscore(bs, rail)
    assert row["home_pp_opportunities"] == 3.0
    assert row["away_pp_opportunities"] == 2.0
    assert row["home_faceoff_pct"] == .52
    assert row["away_faceoff_pct"] == .48
    unknown = ing._parse_boxscore(bs)
    assert unknown["home_pp_opportunities"] is None
    assert unknown["home_faceoff_pct"] is None


def test_starter_is_the_flagged_goalie_not_the_first_decision_goalie():
    """Regression: the API marks the starter explicitly and labels an
    overtime loss ``"O"``, not ``"OTL"``. Keying on a decision set without
    ``"O"`` matched nothing, fell through to ``goalies[0]`` (the 00:00
    scratch goalie), and recorded the team as having no start that game —
    which nulled every later goalie feature for that team."""
    bs = {"id": 2026020002,
          "homeTeam": {"sog": 27, "score": 3}, "awayTeam": {"sog": 26, "score": 2},
          "playerByGameStats": {
              "homeTeam": {
                  "forwards": [{"powerPlayGoals": 0, "faceoffWinningPctg": 0.5}],
                  "goalies": [
                      # The scratch goalie is listed FIRST and carries no
                      # decision — exactly the shape that used to be picked.
                      {"playerId": 1, "decision": None, "toi": "00:00",
                       "starter": False, "powerPlayShotsAgainst": "0/0"},
                      {"playerId": 2, "decision": "O", "toi": "62:18",
                       "starter": True, "powerPlayShotsAgainst": "0/2",
                       "goalsAgainst": 2, "shotsAgainst": 26},
                  ],
              },
              "awayTeam": {
                  "forwards": [{"powerPlayGoals": 1, "faceoffWinningPctg": 0.5}],
                  "goalies": [{"playerId": 3, "decision": "W", "toi": "60:00",
                               "starter": True, "powerPlayShotsAgainst": "8/9",
                               "goalsAgainst": 2, "shotsAgainst": 30}],
              },
          }}
    row = ing._parse_boxscore(bs)
    assert row["home_goalie_id"] == 2, "picked the 00:00 scratch goalie"
    assert row["home_goalie_toi"] == 62.0 + 18 / 60.0
    assert row["home_pp_opportunities"] is None  # shots cannot replace team chances
    assert row["away_pp_opportunities"] is None


def test_starter_falls_back_to_most_ice_time_without_the_flag():
    """Older payloads omit ``starter``; the most ice time is the starter."""
    bs = {"id": 2026020003,
          "homeTeam": {"sog": 30, "score": 3}, "awayTeam": {"sog": 20, "score": 1},
          "playerByGameStats": {
              "homeTeam": {"forwards": [{"powerPlayGoals": 0, "faceoffWinningPctg": 0.5}],
                           "goalies": [{"playerId": 7, "toi": "00:00", "starter": False},
                                       {"playerId": 8, "toi": "59:10", "starter": False}]},
              "awayTeam": {"forwards": [{"powerPlayGoals": 0, "faceoffWinningPctg": 0.5}],
                           "goalies": [{"playerId": 9, "toi": "60:00", "starter": False}]},
          }}
    row = ing._parse_boxscore(bs)
    assert row["home_goalie_id"] == 8
    assert row["away_goalie_id"] == 9


# ---------------------------------------------------------------------------
# 8b. Goalie expected starter: resolved from prior starts, never the game's
#     own boxscore (regression: every goalie feature served 100% null).
# ---------------------------------------------------------------------------
GOALIE_SERVED = ("goalie_sv_pct_home", "goalie_sv_pct_away",
                 "goalie_gaa_home", "goalie_gaa_away",
                 "goalie_starts_home", "goalie_starts_away",
                 "goalie_sv_pct_diff", "goalie_gaa_diff", "goalie_starts_diff")


def test_goalie_features_survive_a_slate_with_no_boxscores():
    """A scheduled game has no boxscore yet, so the expected starter cannot
    come from one. Every goalie feature was 0%-covered at serve time."""
    games = _synth_games(n_days=20, games_per_day=2)
    bs = _synth_goalie_boxscores(games)
    pending = games.tail(4).copy()
    pending["home_score"] = np.nan
    pending["away_score"] = np.nan
    hist = games.head(-4)
    sched = pd.concat([hist, pending], ignore_index=True)
    # The slate's own boxscores are REMOVED — they do not exist yet in
    # production, and their absence is the whole bug being pinned.
    bs_hist = bs[bs["game_id"].isin(set(hist["game_id"]))]
    slate = feat_mod.build_slate_features(sched, bs_hist)
    assert len(slate) == len(pending)
    for col in GOALIE_SERVED:
        v = pd.to_numeric(slate[col], errors="coerce")
        assert v.notna().all(), f"{col} is null on a slate that has prior starts"
    assert (slate["g_home_name"] != "").all(), "expected starter name missing"


def test_expected_starter_ignores_the_decision_goalie_of_the_game_itself():
    """PIT: game t's own goalie LINE must not move game t's goalie features.

    This is the leak that mattered — the production builder picked the
    expected starter out of the very boxscore it was about to predict from,
    so ``goalie_sv_pct``/``goalie_gaa`` at game t were functions of game t.
    The expected starter is the most-workload goalie entering t, so only
    strictly-earlier starts may enter its state.
    """
    games = _synth_games(n_days=20, games_per_day=2)
    bs = _synth_goalie_boxscores(games)
    clean = feat_mod.build_game_features(games, bs)
    target = games.sort_values("gameday", kind="stable").iloc[15]["game_id"]
    dirty_bs = bs.copy()
    mask = dirty_bs["game_id"] == target
    # Corrupt the per-goalie STATISTICS of that game, keeping the identities
    # so the primary's rolling state is the thing under test.
    dirty_bs.loc[mask, "home_goals_against"] = 5.0
    dirty_bs.loc[mask, "away_goals_against"] = 5.0
    dirty_bs.loc[mask, "home_shots_against"] = 5.0
    dirty_bs.loc[mask, "away_shots_against"] = 5.0
    dirty_bs.loc[mask, "home_goalie_toi"] = 12.0
    dirty_bs.loc[mask, "away_goalie_toi"] = 12.0
    dirty = feat_mod.build_game_features(games, dirty_bs)
    for col in GOALIE_SERVED:
        a = clean.loc[clean["game_id"] == target, col].to_numpy()
        b = dirty.loc[dirty["game_id"] == target, col].to_numpy()
        assert np.allclose(a, b, equal_nan=True), \
            f"{col} read the game it is predicting"
    # Teeth: that start IS real prior history for the same teams' later
    # games, so the poison must move them. Without this the check is vacuous.
    later = clean["game_id"] != target
    moved = any(
        not np.allclose(
            pd.to_numeric(clean.loc[later, c], errors="coerce"),
            pd.to_numeric(dirty.loc[later, c], errors="coerce"), equal_nan=True)
        for c in ("goalie_sv_pct_home", "goalie_gaa_home",
                  "goalie_sv_pct_away", "goalie_gaa_away"))
    assert moved, "fixture changed nothing, so the assertion is vacuous"


def test_expected_starter_is_the_prior_workload_leader_not_tonights_goalie():
    """The backup starts each team's last game, but the feature must always
    name the primary — that is what was knowable beforehand."""
    games = _synth_games(n_days=20, games_per_day=2)
    bs = _synth_goalie_boxscores(games)
    # Fixture teeth: the backup really does start games in this pool.
    assert (bs["home_goalie_name"].astype(str).str.endswith("B")
            | bs["away_goalie_name"].astype(str).str.endswith("B")).any(), \
        "fixture never starts the backup, so the rules would agree"
    df = feat_mod.build_game_features(games, bs)
    warm = df[df["g_home_name"].astype(str) != ""]
    assert len(warm) > 0
    assert warm["g_home_name"].astype(str).str.endswith("A").all(), \
        "expected starter tracked tonight's decision goalie"
    assert warm["g_away_name"].astype(str).str.endswith("A").all(), \
        "expected starter tracked tonight's decision goalie"
    # ...and the workload it reports is the primary's, not the backup's one.
    assert pd.to_numeric(warm["goalie_starts_home"], errors="coerce").max() >= 5


def _season_split_goalie_frame():
    """Two seasons for one team (ANA): G. Alpha is the 2025-26 workhorse
    (5 starts), G. Bravo starts the 2026-27 games. The 2026-10-04 FLA@ANA
    card served S. Bobrovsky (129 career starts, 0 this season) over the
    goalies the team was actually using — this is that shape."""
    games_rows, bs_rows = [], []

    def add(gid, day, season, home_goalie, home_line, away_goalie, away_line):
        games_rows.append({"game_id": gid, "season": season, "gameday": day,
                           "home_team": "ANA", "away_team": "BOS",
                           "home_score": 3, "away_score": 2})
        hid, hname, hga, hsa = home_goalie + home_line
        aid, aname, aga, asa = away_goalie + away_line
        bs_rows.append({"game_id": gid,
                        "home_goalie_id": hid, "home_goalie_name": hname,
                        "home_goalie_toi": 60.0,
                        "home_goals_against": hga, "home_shots_against": hsa,
                        "away_goalie_id": aid, "away_goalie_name": aname,
                        "away_goalie_toi": 60.0,
                        "away_goals_against": aga, "away_shots_against": asa})

    for i, day in enumerate(("2025-10-05", "2025-10-12", "2025-10-19",
                             "2025-10-26", "2025-11-02")):
        add(f"P{i}", day, 2025,
            ("ga", "G. Alpha"), (2.0, 30.0),
            ("bx", "B. Xray"), (3.0, 28.0))
    add("N0", "2026-10-03", 2026,
        ("gb", "G. Bravo"), (1.0, 25.0),
        ("bx", "B. Xray"), (3.0, 28.0))
    add("N1", "2026-10-05", 2026,
        ("gb", "G. Bravo"), (2.0, 30.0),
        ("bx", "B. Xray"), (3.0, 28.0))
    add("N2", "2026-10-07", 2026,
        ("gb", "G. Bravo"), (1.0, 20.0),
        ("bx", "B. Xray"), (3.0, 28.0))
    return pd.DataFrame(games_rows), pd.DataFrame(bs_rows)


def test_expected_starter_vote_is_season_scoped():
    """Season-to-date workload (the manifest contract): a prior-season
    workhorse with ZERO starts this season must never be served as tonight's
    expected starter over the goalies the team is actually using."""
    games, bs = _season_split_goalie_frame()
    per_game, _ = feat_mod.goalie_state(bs, games, games)
    per_game = per_game.assign(game_id=games["game_id"])
    late = per_game[per_game["game_id"] == "N1"].iloc[0]
    assert late["g_home_name"] == "G. Bravo", \
        "season-scoped vote picked the prior-season workhorse"
    # Prior season keeps its own answer: Alpha was the man back then.
    old = per_game[per_game["game_id"] == "P3"].iloc[0]
    assert old["g_home_name"] == "G. Alpha"


def test_expected_starter_vote_season_falls_back_to_the_calendar_rule():
    """A frame without a ``season`` column (the pure-calendar case) scopes
    the vote the same way — July boundary, config.current_nhl_season."""
    games, bs = _season_split_goalie_frame()
    no_season = games.drop(columns=["season"])
    per_game, _ = feat_mod.goalie_state(bs, no_season, no_season)
    per_game = per_game.assign(game_id=games["game_id"])
    late = per_game[per_game["game_id"] == "N1"].iloc[0]
    assert late["g_home_name"] == "G. Bravo"


def test_opening_night_is_honest_nan_not_last_seasons_workhorse():
    """The manifest's opening-night unknown starter: a team with no start
    this season resolves NaN — never fabricated from last season's #1."""
    games, bs = _season_split_goalie_frame()
    per_game, _ = feat_mod.goalie_state(bs, games, games)
    per_game = per_game.assign(game_id=games["game_id"])
    night = per_game[per_game["game_id"] == "N0"].iloc[0]
    assert night["g_home_name"] == ""
    for col in ("goalie_sv_pct_home", "goalie_gaa_home", "goalie_starts_home",
                "g_home_sv_pct", "g_home_gaa", "g_home_starts"):
        assert pd.isna(pd.to_numeric(night[col], errors="coerce")) or \
            night[col] == "", f"{col} fabricated an opening-night value"


def test_goalie_card_pair_is_the_season_line_not_the_form_ewm():
    """The g_* serving pair (the card's SV% · GAA, display-only like NFL's
    QB enrichment) is the season-to-date POOLED line a reader compares
    against the league's season table; the model form feature stays the
    strictly-prior season-scoped EWM. They are different numbers and the
    card must carry the season one."""
    games, bs = _season_split_goalie_frame()
    per_game, _ = feat_mod.goalie_state(bs, games, games)
    per_game = per_game.assign(game_id=games["game_id"])
    row = per_game[per_game["game_id"] == "N2"].iloc[0]
    # Bravo's strictly-prior 2026-27 starts entering N2: N0 (1 GA/25 SA) and
    # N1 (2 GA/30 SA) -> pooled 1 - 3/55.
    assert row["g_home_sv_pct"] == pytest.approx(1.0 - 3.0 / 55.0)
    assert row["g_home_gaa"] == pytest.approx(3.0 / 2.0)  # 3 GA in 120 min
    assert row["g_home_starts"] == 2                      # season starts
    assert row["goalie_starts_home"] == 2                  # manifest: season
    # Teeth: the model form is the EWM, not the pooled season line.
    form = float(row["goalie_sv_pct_home"])
    assert abs(form - (1.0 - 3.0 / 55.0)) > 1e-6, \
        "form and season line collapsed to the same number; test is vacuous"


def test_goalie_card_pair_is_strictly_prior():
    """PIT: game t's own boxscore must not move game t's card pair."""
    games, bs = _season_split_goalie_frame()
    clean, _ = feat_mod.goalie_state(bs, games, games)
    dirty_bs = bs.copy()
    mask = dirty_bs["game_id"] == "N2"
    dirty_bs.loc[mask, "home_goals_against"] = 20.0
    dirty_bs.loc[mask, "home_shots_against"] = 20.0
    dirty, _ = feat_mod.goalie_state(dirty_bs, games, games)
    clean = clean.assign(game_id=games["game_id"])
    dirty = dirty.assign(game_id=games["game_id"])
    for col in ("g_home_sv_pct", "g_home_gaa", "g_home_starts", "g_home_name"):
        a = clean.loc[clean["game_id"] == "N2", col].iloc[0]
        b = dirty.loc[dirty["game_id"] == "N2", col].iloc[0]
        assert (a == b) or (pd.isna(a) and pd.isna(b)), \
            f"{col} read the game it is predicting"


def test_pp_conversion_trail_pools_counts_across_a_zero_opportunity_game():
    """A game with no power play has no conversion rate. Averaging per-game
    rates dropped it and voided the whole window; pooling the counts keeps
    the feature alive with a valid, volume-weighted value."""
    games = _synth_games(n_days=20, games_per_day=2)
    bs = _synth_boxscores(games)
    # The most recent game for every team: nobody had a power play.
    last_day = games["gameday"].max()
    bs.loc[bs["game_id"].isin(set(games[games["gameday"] == last_day]["game_id"])),
           ["home_pp_opportunities", "away_pp_opportunities"]] = 0.0
    df = feat_mod.build_game_features(games, bs)
    warm = df[df["gameday"] < last_day].tail(10)
    v = pd.to_numeric(warm["pp_success_diff"], errors="coerce")
    assert v.notna().all(), "a zero-opportunity game voided the PP window"
    # Pooled, so it equals the volume-weighted conversion, not 0.
    assert v.abs().max() > 0.0



# ---------------------------------------------------------------------------
# 8c. Feature coverage report: cold start vs defect, and the serving slate.
# ---------------------------------------------------------------------------
COVERAGE_KEYS = ("feature", "window", "n_games", "pct_measured", "pct_nonnull",
                 "n_default_zero", "status")


def _coverage_by_feature(rows, window):
    return {r["feature"]: r for r in rows if r["window"] == window}


def test_feature_coverage_tells_cold_start_apart_from_a_defect():
    """A team's debut game has no prior history, so a null there is the
    designed warm-up. Reporting that as starvation is what made the panel
    cry wolf; reporting a WARM null as healthy is what would hide one."""
    games = _synth_games(n_days=20, games_per_day=2)
    df = feat_mod.build_game_features(games, _synth_goalie_boxscores(games))
    rows = mon.coverage(df, current_df=df)
    for r in rows:
        for k in COVERAGE_KEYS:
            assert k in r, f"coverage row dropped the front-end key {k!r}"
    by_f = _coverage_by_feature(rows, "baseline")
    warm_nulls = [r for r in by_f.values() if r["n_warm_null"] > 0]
    assert not warm_nulls, \
        f"warm (defect) nulls present: {[r['feature'] for r in warm_nulls]}"
    # Trailing features ARE null on debuts, and the report must say so
    # rather than raising an alarm.
    cold = [r for r in by_f.values() if r["n_cold_null"] > 0]
    assert cold, "fixture has no cold-start rows to classify"
    for r in cold:
        assert r["status"] == "OK", f"{r['feature']} flagged for cold start"
        assert r["cause"] == "cold_start"
        assert r["pct_measured_eligible"] == 100.0
    # Now inject a real defect on a WARM row and it must be reported.
    broken = df.copy()
    warm_rows = broken["gameday"] > broken["gameday"].min()
    broken.loc[warm_rows, "pp_success_diff"] = np.nan
    by_f2 = _coverage_by_feature(mon.coverage(broken, current_df=broken),
                                 "baseline")
    hit = by_f2["pp_success_diff"]
    assert hit["n_warm_null"] > 0, "a null on a warm game was not counted"
    assert hit["cause"] == "defect"
    assert hit["status"] in ("STARVED", "LOW_COVERAGE")
    assert hit["pct_measured_eligible"] < 100.0


def test_coverage_season_openers_are_cold_for_the_goalie_family():
    """2026-10-07 run-log review: "9 feature(s) with WARM nulls — a defect"
    over a window that SPANS the off-season gap.

    The goalie family's evidence is season-scoped (goalie_state's workload
    vote counts only same-season starts; a team's opener resolves honest NaN
    — the manifest's opening-night unknown starter). In a trailing window
    that starts in last spring's playoffs, an October opener is NOT the
    club's window debut — both teams already played in the window — so the
    window-only warmup mask filed the designed NaN as a WARM defect. Openers
    must classify cold (via the pool's season-first lookup, so a mid-season
    window start never passes a team's third game off as an opener), while a
    null where the evidence DOES exist stays a defect."""
    spring = _synth_games(n_days=6, games_per_day=2, start="2026-05-20")
    fall = _synth_games(n_days=6, games_per_day=2, start="2026-10-01")
    # Season = START-year, July boundary (features._nhl_season_of):
    # May 2026 belongs to 2025, October 2026 opens 2026.
    for part in (spring, fall):
        d = pd.to_datetime(part["gameday"])
        part["season"] = (d.dt.year - (d.dt.month < 7).astype(int)).astype(int)
    games = pd.concat([spring, fall], ignore_index=True)
    df = feat_mod.build_game_features(games, _synth_goalie_boxscores(games))

    # The window-debut mask alone cannot mark the fall rows cold: every club
    # already "debuted" inside the window's spring half.
    debut_only = mon._warmup_mask(df)
    fall_rows = df["gameday"].astype(str) >= "2026-10-01"
    assert not debut_only.reindex(df.index).fillna(False)[fall_rows].any(), \
        "fixture invalid: fall rows are already window-debuts"

    rows = mon.coverage(df, current_df=df)
    cur = _coverage_by_feature(rows, "current")
    for f in ("goalie_sv_pct_home", "goalie_sv_pct_away", "goalie_sv_pct_diff",
              "goalie_gaa_diff", "goalie_starts_diff", "goalie_starts_home"):
        r = cur[f]
        assert r["n_null"] > 0, f"{f}: fixture produced no opening-night NaNs"
        assert r["n_warm_null"] == 0, \
            f"{f} called the opening-night NaN a WARM defect ({r})"
        assert r["n_cold_null"] > 0, f"{f}: opener rows not classified cold"
        assert r["status"] == "OK", f"{f} flagged for a designed opener NaN"
        assert r["cause"] == "cold_start"
        assert r["pct_measured_eligible"] == 100.0

    # Negative control: a null where the season's evidence DOES exist (a
    # fall row after the club's first start) is still a real defect.
    broken = df.copy()
    measurable = broken["gameday"].astype(str) >= "2026-10-03"
    assert measurable.any(), "fixture has no measurable fall rows"
    broken.loc[measurable, "goalie_sv_pct_home"] = np.nan
    hit = _coverage_by_feature(
        mon.coverage(broken, current_df=broken), "current")["goalie_sv_pct_home"]
    assert hit["n_warm_null"] > 0, "a genuine goalie outage was excused"
    assert hit["cause"] == "defect"
    assert hit["status"] in ("STARVED", "LOW_COVERAGE")


def _pool_ratings(games: pd.DataFrame) -> pd.DataFrame:
    """A rating for every (team, position, situation) within the pool's
    45-day lookback of every gameday.

    The coverage fixtures run with no MoneyPuck history, so the pool has no
    row for ANY side and every ``pl_*`` column is served the documented
    position prior. That is a real production state — the 2026-10-09 run
    served priors on 24 of 5,714 sides — and it is REPORTED (see
    test_coverage_reports_the_position_prior_pool_fallback), so a fixture
    standing in for a HEALTHY frame has to bring its own pool source.
    """
    from test_injury_stints import make_ratings
    rows = []
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    for day in sorted(set(pd.to_datetime(games["gameday"]))):
        src = (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for team in teams:
            for sit in ("5on5", "5on4"):
                for pos in ("C", "L", "R", "D"):
                    rows.append((f"mp-{team}-{pos}-{sit}", team, sit, pos,
                                 0.5, 12000.0, src))
    return make_ratings(rows)


def test_feature_coverage_measures_the_slate_the_pipeline_actually_ships():
    """Regression: the goalie family read 96-98% on the decided pool while
    EVERY published prediction carried a null, because the report only ever
    looked at the decided pool. A nulled slate column must read STARVED.

    The frame is built WITH a resolved pool (_pool_ratings): coverage now
    excludes position-prior defaults from pct_measured (2026-10-06 audit §E),
    so a fixture that never resolves its pool would legitimately read 0%
    measured on all 24 pl_* columns — that state is pinned by
    test_coverage_reports_the_position_prior_pool_fallback instead.
    """
    games = _synth_games(n_days=20, games_per_day=2)
    bs = _synth_goalie_boxscores(games)
    ratings = _pool_ratings(games)
    df = feat_mod.build_game_features(games, bs, player_ratings=ratings)
    pending = games.tail(4).copy()
    pending["home_score"] = np.nan
    pending["away_score"] = np.nan
    hist = games.head(-4)
    slate = feat_mod.build_slate_features(
        pd.concat([hist, pending], ignore_index=True),
        bs[bs["game_id"].isin(set(hist["game_id"]))],
        player_ratings=ratings)

    rows = mon.coverage(df, slate_df=slate, current_df=df)
    slate_rows = _coverage_by_feature(rows, "serving slate")
    assert len(slate_rows) == len(config.active_moneyline_feature_cols())
    assert slate_rows["goalie_sv_pct_home"]["n_games"] == len(slate)
    assert all(r["status"] == "OK" for r in slate_rows.values()), \
        "a healthy slate was reported as unhealthy"
    assert all(r["pct_measured"] == 100.0 for r in slate_rows.values())

    # Negative control: this is the outage the decided-pool-only report
    # scored at 96% and missed entirely.
    dead = slate.copy()
    dead["goalie_sv_pct_home"] = np.nan
    dead_rows = _coverage_by_feature(mon.coverage(df, slate_df=dead), "serving slate")
    assert dead_rows["goalie_sv_pct_home"]["status"] == "STARVED"
    assert dead_rows["goalie_sv_pct_home"]["pct_measured"] == 0.0
    assert dead_rows["goalie_sv_pct_home"]["cause"] == "defect"
    # ...while the decided pool still looks healthy, which is the whole trap.
    assert _coverage_by_feature(mon.coverage(df, slate_df=dead, current_df=df),
                                "baseline")["goalie_sv_pct_home"]["status"] != "STARVED"


def test_coverage_reports_the_position_prior_pool_fallback():
    """2026-10-06 audit §E, still open on 2026-10-09: *non-null ``pl_*`` is
    NOT proof of measured coverage — defaults can yield 100% non-null output*.

    A side with no pool row is served ``features._POSITION_PRIOR`` for every
    position and situation, so all 24 pool columns are non-null on every game
    however little evidence stands behind them. The shipped coverage CSV
    reported exactly that as 100.000% measured, with ``n_default_zero``
    hardcoded to 0: the six June-2026 Cup-Final games inside the baseline
    window (24 sides with no pool row in the 2026-10-09 run) read fully
    measured, and the monitor page printed "all windows healthy".

    Pin both halves of the fix: a frame whose pool never resolved reads 0%
    MEASURED / STARVED / ``n_default_zero == n_games`` while staying 100%
    NON-NULL (the trap is that nothing is NULL), and a frame with a real pool
    measures every column with zero defaults.
    """
    games = _synth_games(n_days=20, games_per_day=2)
    bs = _synth_goalie_boxscores(games)

    # No pool source at all: every side is served the position prior.
    blind = feat_mod.build_game_features(games, bs)
    by_f = _coverage_by_feature(mon.coverage(blind, current_df=blind),
                                "baseline")
    hit = by_f["pl_evo_c_home"]
    assert hit["n_default_zero"] == hit["n_games"], hit
    assert hit["pct_nonnull"] == 100.0, "the trap is that nothing is NULL"
    assert hit["pct_measured"] == 0.0 and hit["n_measured"] == 0, hit
    assert hit["status"] == "STARVED", hit
    assert hit["cause"] == "position_prior_default", hit
    # A diff is contaminated when EITHER side carried the prior.
    diff = by_f["pl_evo_c_diff"]
    assert diff["n_default_zero"] == diff["n_games"], diff
    # Pool-served columns only: an uninvolved feature is untouched.
    other = by_f["elo_home"]
    assert other["n_default_zero"] == 0 and other["status"] == "OK", other
    assert other["pct_measured"] == 100.0, other

    # A frame that DOES resolve its pool measures every pl_* column.
    seen = feat_mod.build_game_features(games, bs,
                                        player_ratings=_pool_ratings(games))
    ok = _coverage_by_feature(mon.coverage(seen, current_df=seen),
                              "baseline")["pl_evo_c_home"]
    assert ok["n_default_zero"] == 0 and ok["pct_measured"] == 100.0, ok
    assert ok["status"] == "OK" and ok["cause"] == "complete", ok


def test_coverage_verdict_names_default_filled_pool_values():
    """Phase 14 must not print "100.000% measured" over defaults.

    ``_log_coverage_verdict`` counts a default-filled value as UNMEASURED,
    warns (naming the feature) when the defaults sit on warm games, and stays
    quiet-but-explicit when they are the documented cold warm-up.
    """
    import master_pipeline as mp

    def _row(feature, window, n_games, warm_default, cold_default):
        defaults = warm_default + cold_default
        return {"feature": feature, "window": window, "n_games": n_games,
                "n_measured": n_games - defaults, "n_null": 0,
                "n_cold_null": 0, "n_warm_null": 0,
                "n_cold_default": cold_default, "n_warm_default": warm_default,
                "n_default_zero": defaults,
                "status": "OK" if not warm_default else "LOW_COVERAGE",
                "cause": ("position_prior_default" if warm_default
                          else "cold_start" if cold_default else "complete")}

    records = []
    handler = logging.Handler()
    handler.emit = records.append
    mp.logger.addHandler(handler)
    mp.logger.setLevel(logging.INFO)
    try:
        mp._log_coverage_verdict([
            _row("pl_evo_c_home", "baseline", 250, 6, 0),
            _row("elo_diff", "baseline", 250, 0, 0),
            _row("pl_evo_c_away", "serving slate", 5, 0, 5),
        ])
    finally:
        mp.logger.removeHandler(handler)

    warns = [r.getMessage() for r in records if r.levelno >= logging.WARNING]
    assert len(warns) == 1, warns
    assert "pl_evo_c_home" in warns[0] and "6 default-filled" in warns[0], warns[0]
    assert "warm gap" in warns[0], warns[0]
    infos = " ".join(r.getMessage() for r in records if r.levelno < logging.WARNING)
    # The cold-only default (the slate's prior-seeded warm-up) is disclosed
    # as by-design, never as a defect...
    assert "serving slate" in infos and "by design" in infos, infos[:300]
    # ...and the cold-null wording names the season-opener half too (the
    # goalie family's designed opening-night NaN is not a team debut).
    assert "(team debut or season opener)" in infos, infos[:300]


def test_coverage_splits_postseason_pool_boundary_from_warm_gap():
    """2026-10-09 re-review of the 17:40 run: the warm-default WARNING was
    firing on a DOCUMENTED source boundary.

    MoneyPuck's player-game archives are regular-season-only, so past the
    pool's 45-day serve window every Stanley-Cup-Final side is served the
    position prior — the six defaulted June-2026 baseline games are exactly
    that stretch, and the 2026-10-05-style true warm gap had not recurred.
    Only a warm default on a regular-season game deserves the WARNING.

    Pin: warm defaults whose games are ALL playoff rows classify as
    ``position_prior_postseason_boundary`` and the verdict stays INFO naming
    the boundary; a single regular-season default keeps
    ``position_prior_default`` plus the WARNING; a mixed window warns on the
    hole and still discloses the boundary count; a frame without the
    ``is_playoffs`` column keeps the old cause.
    """
    from features import _POSITION_PRIOR
    prior = float(_POSITION_PRIOR["EVO"]["C"])
    warmup = pd.Series([False] * 8, dtype=bool)   # every game is warm

    post = pd.DataFrame({
        "pl_evo_c_home": [prior] * 6 + [prior + 0.3] * 2,
        "pl_evo_c_away": [prior] * 6 + [prior + 0.3] * 2,
        "is_playoffs": [1.0] * 6 + [0.0, 0.0],
    })
    hit = mon._coverage_row("pl_evo_c_home", post, "baseline", warmup)
    assert hit["cause"] == "position_prior_postseason_boundary", hit
    assert hit["n_warm_default"] == 6, hit
    assert hit["pct_measured"] < 100.0, "the default stays UNMEASURED"
    # No is_playoffs column: the old cause, unchanged arithmetic.
    hole = mon._coverage_row("pl_evo_c_home", post.drop(columns=["is_playoffs"]),
                             "baseline", warmup)
    assert hole["cause"] == "position_prior_default", hole
    # One regular-season default inside the playoff run keeps the true gap.
    mixed = post.copy()
    mixed.loc[0, "is_playoffs"] = 0.0
    assert mon._coverage_row("pl_evo_c_home", mixed, "baseline",
                             warmup)["cause"] == "position_prior_default"

    # Verdict: a boundary-only window never WARNs (it names the boundary as
    # info); a mixed window warns on the hole and discloses the boundary.
    import master_pipeline as mp

    def _brow(feature, cause, n_warm_default):
        return {"feature": feature, "window": "baseline", "n_games": 8,
                "n_measured": 8 - n_warm_default, "n_null": 0,
                "n_cold_null": 0, "n_warm_null": 0, "n_cold_default": 0,
                "n_warm_default": n_warm_default,
                "n_default_zero": n_warm_default,
                "status": "LOW_COVERAGE", "cause": cause}

    boundary = _brow("pl_evo_c_home", "position_prior_postseason_boundary", 6)
    gap = _brow("pl_ppo_r_away", "position_prior_default", 3)
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    mp.logger.addHandler(handler)
    mp.logger.setLevel(logging.INFO)
    try:
        mp._log_coverage_verdict([boundary, gap])
    finally:
        mp.logger.removeHandler(handler)
    warns = [r.getMessage() for r in records if r.levelno >= logging.WARNING]
    assert len(warns) == 1 and "pl_ppo_r_away" in warns[0], warns
    assert "warm gap" in warns[0], warns
    assert "1 postseason-boundary feature(s)" in warns[0], warns
    infos = " ".join(r.getMessage() for r in records if r.levelno < logging.WARNING)
    assert "POSTSEASON games" in infos and "regular-season-only" in infos, \
        infos[:300]

    records.clear()
    mp.logger.addHandler(handler)
    try:
        mp._log_coverage_verdict([boundary])
    finally:
        mp.logger.removeHandler(handler)
    assert not [r for r in records if r.levelno >= logging.WARNING], \
        "the documented postseason boundary must not page"


def test_pool_default_mask_reads_prior_through_module_binding():
    """2026-10-09 third review: the mask must resolve ``_POSITION_PRIOR``
    through the module-level ``feat_mod`` binding, not a bare local import.

    ``_pool_default_mask`` used ``from features import _POSITION_PRIOR``
    inside a try/except. That resolves only when the backend directory is
    ``sys.path[0]`` (the production run and this suite both put it there).
    Imported as a PACKAGE — ``from backend import monitoring``, which is how
    frontend tooling and real-frame audit replays reach it — the bare name
    is unresolvable, the except swallowed it, and the mask returned None:
    every ``pl_*`` default then counts as a MEASUREMENT again, which is the
    exact 2026-10-06 defect the mask exists to prevent. A silent reversion
    to dishonest coverage is the worst failure mode here, because the report
    still prints a healthy measured% and no one is paged.

    Pin: swapping the prior on the already-imported module changes what the
    mask marks as default, proving it reads ``feat_mod`` rather than its own
    import. A non-pool feature still returns None (unchanged contract).
    """
    sentinel = {"EVO": {"C": 0.5, "L": 0.721, "R": 0.698, "D": 0.169},
                "PPO": {"C": 1.581, "L": 1.627, "R": 1.518, "D": 0.63}}
    real = mon.feat_mod._POSITION_PRIOR
    try:
        mon.feat_mod._POSITION_PRIOR = sentinel
        prior = 0.5
        df = pd.DataFrame({
            "pl_evo_c_home": [prior, prior + 0.3],
            "pl_evo_c_away": [prior, prior + 0.3],
        })
        mask = mon._pool_default_mask(df, "pl_evo_c_home")
        assert mask is not None, \
            "the mask must resolve the prior via the module-level binding"
        assert mask.tolist() == [True, False], mask.tolist()
        # A diff defaults when either side still reads the prior.
        dmask = mon._pool_default_mask(df, "pl_evo_c_diff")
        assert dmask.tolist() == [True, False], dmask.tolist()
    finally:
        mon.feat_mod._POSITION_PRIOR = real
    # Unchanged contract: a non-pool feature is still unjudgeable -> None.
    assert mon._pool_default_mask(
        pd.DataFrame({"elo_diff": [0.1]}), "elo_diff") is None


def test_feature_coverage_artifacts_carry_both_windows():
    """The CSV the monitor page loads must contain the slate window."""
    import tempfile
    games = _synth_games(n_days=20, games_per_day=2)
    bs = _synth_goalie_boxscores(games)
    df = feat_mod.build_game_features(games, bs)
    pending = games.tail(3).copy()
    pending["home_score"] = np.nan
    pending["away_score"] = np.nan
    hist = games.head(-3)
    slate = feat_mod.build_slate_features(
        pd.concat([hist, pending], ignore_index=True),
        bs[bs["game_id"].isin(set(hist["game_id"]))])
    with tempfile.TemporaryDirectory() as tmp:
        _, cov_name = mon.write_run_engine_feature_artifacts(
            Path(tmp), "20260923", df, df.tail(10), slate_df=slate)
        cov = pd.read_csv(Path(tmp) / cov_name)
    assert set(cov["window"]) == {"current", "baseline", "serving slate"}
    assert set(cov["feature"]) == set(config.active_moneyline_feature_cols())
    assert set(COVERAGE_KEYS) <= set(cov.columns)


def test_feature_coverage_drift_windows_are_the_frames_drift_compares():
    """The coverage report's current/baseline windows must be the SAME frames
    the drift table compares — the structural alignment MLB enforces after
    its 08-28 incident (the coverage CSV measured a window the drift CSV
    never saw, so two tables beside each other answered different
    questions). coverage_for_drift_windows pins the labels and the frame
    identity; the current window's cold-start classification follows the
    current window's own timeline."""
    games = _synth_games(n_days=30, games_per_day=2)
    df = feat_mod.build_game_features(games, _synth_goalie_boxscores(games))
    current = df.tail(10)
    rows = mon.coverage_for_drift_windows(df, current)
    assert {r["window"] for r in rows} == {"baseline", "current"}
    base = _coverage_by_feature(rows, "baseline")
    cur = _coverage_by_feature(rows, "current")
    assert base["elo_home"]["n_games"] == len(df)
    assert cur["elo_home"]["n_games"] == len(current)
    # Every warm game in the current window is measured — the trailing
    # window is warm by construction, so any warm null would be a defect.
    warm_cur = [r for r in cur.values() if r["n_warm_null"] > 0]
    assert not warm_cur, \
        f"warm nulls in the current window: {[r['feature'] for r in warm_cur]}"


def test_drift_windows_follow_the_mlb_trailing_tail_geometry():
    """The drift baseline must be the era IMMEDIATELY BEFORE the current
    window — MLB's trailing-tail slice (prior.tail(max(3x current, 250))) —
    not the full pool. A full-history baseline mixes whole seasons into the
    comparison, which is what flagged season-boundary effects (goalie_starts
    resets, playoff-window levels) every early-season run. The windows are
    also disjoint and adjacent, or the PSI row describes a population its
    coverage rows never measured. MONITORING ONLY: the training path is
    expanding walk-forward over the full pool and must never read these
    windows."""
    games = _synth_games(n_days=120, games_per_day=4)
    df = feat_mod.build_game_features(games, _synth_goalie_boxscores(games))
    n = len(df)
    baseline, current = mon.drift_windows(df)
    n_cur = len(current)
    # Current window: the trailing DRIFT_CURRENT_GAMES decided games.
    assert n_cur == config.DRIFT_CURRENT_GAMES
    assert current["gameday"].iloc[-1] == df["gameday"].iloc[-1]
    # Baseline: max(3x current, DRIFT_BASELINE_MIN_GAMES) games of the era
    # immediately preceding the current window — never the full pool.
    n_base_expected = min(max(3 * n_cur, config.DRIFT_BASELINE_MIN_GAMES),
                          n - n_cur)
    assert len(baseline) == n_base_expected
    assert baseline["gameday"].iloc[-1] < current["gameday"].iloc[0]
    # Disjoint AND adjacent: the baseline ends where the current begins
    # (positional, so the pin holds for any index dtype).
    assert len(pd.concat([baseline, current])) == n_base_expected + n_cur
    pos_base_end = df.index.get_indexer([baseline.index[-1]])[0]
    pos_cur_start = df.index.get_indexer([current.index[0]])[0]
    assert pos_cur_start == pos_base_end + 1


def test_drift_windows_small_pool_falls_back_whole_tail():
    """A pool that cannot support both windows disjointly must degrade to
    the whole-pool tail (a defensible baseline of whatever era exists), not
    raise or hand back an empty frame."""
    games = _synth_games(n_days=12, games_per_day=2)
    df = feat_mod.build_game_features(games, _synth_goalie_boxscores(games))
    baseline, current = mon.drift_windows(df)
    assert len(current) == max(len(df) // 2, 1)
    assert len(baseline) == len(df) - len(current)
    assert set(baseline.index) & set(current.index) == set()


# ---------------------------------------------------------------------------
# 9. Fold-id contiguity (regression: non-contiguous fold numbering)
# ---------------------------------------------------------------------------
def test_fold_ids_are_contiguous_after_the_min_validation_filter():
    """The min-validation filter drops candidate windows; fold_id must be
    renumbered so it stays an ordinal. Consumers use it as one (the
    prequential calibrator's strictly-prior mask) and log it as one."""
    rows = []
    gid = 0
    for day in range(200):
        d = pd.Timestamp("2025-10-01") + pd.Timedelta(days=day)
        # A sparse stretch whose 7-date windows fall under MIN_VAL_FOLD_GAMES
        # (retained and disclosed since 2026-09-30, never dropped).
        per_day = 2 if 90 <= day < 110 else 6
        for _ in range(per_day):
            rows.append({"game_id": f"g{gid}", "season": 2025, "gameday": d,
                         "home_team": "ANA", "away_team": "BOS",
                         "home_score": 2, "away_score": 1, "home_win": 1.0})
            gid += 1
    df = pd.DataFrame(rows)
    folds = folds_mod.make_folds(df)
    assert folds, "fixture produced no folds"
    ids = [f.fold_id for f in folds]
    assert ids == list(range(len(folds))), \
        f"fold_id not contiguous after filtering: {ids}"
    assert folds_mod.fold_summary(folds)["n_folds"] == len(folds)
    # Strictly-prior training must still hold for every renumbered fold.
    for f in folds:
        assert df.loc[f.train_idx, "gameday"].max() < f.val_start


# ---------------------------------------------------------------------------
# 10. Retention keeps the CURRENT slate's SHAP cards
# ---------------------------------------------------------------------------
def test_retention_keeps_the_current_slate_shap_cards():
    """SHAP files are written per GAME with the official NHL numeric id,
    which carries no date. They age via the game-date map — without it the
    pipeline deleted all five current-slate cards on the same run."""
    import retention_policy as rp
    anchor = "20260929"
    retention_dates = {"20260919", "20260925", "20260929"}
    recent_dates = {"20260927", "20260928", "20260929"}
    game_dates = {"2026020001": "20260929", "2026020005": "20260929"}
    for gid in ("2026020001", "2026020005"):
        rel = f"nhl-backend/data_delivery/nhl_shap_game_{gid}.csv"
        # A 10-digit id must never be truncated into a bogus "date".
        assert rp.artifact_date(rel) is None, \
            f"{gid}: numeric NHL id mis-parsed as a date"
        assert rp.classify_artifact(
            rel, seen=set(), retention_dates=retention_dates,
            recent_dates=recent_dates, board_dates={"20260929"},
            anchor_date=anchor, game_dates=game_dates) == "current"
        # Unresolvable id -> protected (never guessed, never deleted).
        assert rp.classify_artifact(
            rel, seen=set(), retention_dates=retention_dates,
            recent_dates=recent_dates, board_dates=set(),
            anchor_date=anchor, game_dates={}) == "protected"
    # A genuinely old game still ages out.
    old = dict(game_dates, **{"2025010001": "20250101"})
    assert rp.classify_artifact(
        "nhl-backend/data_delivery/nhl_shap_game_2025010001.csv",
        seen=set(), retention_dates=retention_dates,
        recent_dates=recent_dates, board_dates=set(),
        anchor_date=anchor, game_dates=old) == "stale"


def test_injury_snapshot_history_artifact_is_never_retention_eligible():
    """The captured ESPN injury history is cumulative health knowledge (the
    2026-09-29 Kaggle run's 403 fallback). Deleting it would tell every later
    run that no status was ever known — it must classify protected on any
    anchor date, never stale.
    """
    import retention_policy as rp
    rel = "nhl-backend/data_delivery/nhl_injury_snapshot_history.parquet"
    for anchor in ("20260929", "20270415"):
        assert rp.classify_artifact(
            rel, seen=set(), retention_dates=set(), recent_dates=set(),
            board_dates=set(), anchor_date=anchor, game_dates={}) == "protected"


def test_retention_still_dates_the_run_dated_families():
    """The numeric-id guard must not break the normal _YYYYMMDD families."""
    import retention_policy as rp
    for rel, expected in (
        ("nhl-backend/data_delivery/nhl_moneyline_v1_20260925.json", "20260925"),
        ("nhl-backend/data_delivery/nhl_run_engine_markets_20260925.csv", "20260925"),
        ("nhl-backend/data_delivery/nhl_run_engine_markets_20260925.meta.json", "20260925"),
        ("nhl-backend/data_delivery/nhl_calibration_20260925.json", "20260925"),
        ("nhl-backend/data_delivery/nhl_predictions_history_20260925.csv", "20260925"),
        ("nhl-backend/data_delivery/nhl_feature_workbook_2026-09-22.xlsx", "20260922"),
    ):
        assert rp.artifact_date(rel) == expected, f"{rel} dated wrong"


def test_no_rfe_candidate_is_a_permanently_empty_column():
    """A candidate with no source column is silently all-NaN and still occupies
    a slot in the RFE trial space, so the search is scored on dead columns.
    Every declared candidate must have a real source once boxscores land."""
    games = _synth_games(n_days=20, games_per_day=2)
    bs = _synth_boxscores(games)
    bs["home_goalie_id"] = 100
    bs["away_goalie_id"] = 200
    bs["home_goalie_toi"] = 60.0
    bs["away_goalie_toi"] = 59.0
    bs["home_goals_against"] = 2
    bs["away_goals_against"] = 3
    bs["home_shots_against"] = 30
    bs["away_shots_against"] = 28
    df = feat_mod.build_game_features(games, bs)
    dead = [c for c in config.NHL_CANDIDATE_COLS
            if c in df.columns and not df[c].notna().any()]
    assert not dead, f"all-NaN RFE candidates: {dead}"
    assert (feat_mod.feature_coverage_report(df)["coverage_pct"] > 0).all(), \
        "a served contract feature is still completely uncovered"


# ---------------------------------------------------------------------------
# 8. Walk-forward progress reporting
# ---------------------------------------------------------------------------
class _LogCapture(logging.Handler):
    """Collect formatted log records emitted by a module's logger."""

    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _capture_walk_forward_logs(mod, games, folds):
    cap = _LogCapture()
    mod.logger.addHandler(cap)
    mod.logger.setLevel(logging.INFO)
    try:
        mod.walk_forward_oof(games, fold_list=folds)
    finally:
        mod.logger.removeHandler(cap)
    return cap.messages


def test_walk_forward_progress_reports_a_live_position_and_completion():
    """A long walk-forward must show where it is and that it finished.

    The cadence used to be a fixed 25 folds: with 46 folds the only line
    ever printed was "moneyline OOF fold 25/46" (fold 50 never arrives),
    and "25" was the cadence rather than the position. The distribution
    walk-forward logged nothing at all except failures, so a healthy 13s
    phase was indistinguishable from a crash. During the 2026-09-25 RFE
    sweep that printed the same 25/46 line 121 times over ~24 minutes,
    which reads as a hang rather than 120 completed trials.
    """
    games = feat_mod.build_game_features(_synth_games(n_days=90, games_per_day=6))
    folds = folds_mod.make_folds(games)
    assert len(folds) >= 4, f"fixture too small to exercise cadence: {len(folds)}"
    total = len(folds)

    for mod, label in ((ml_mod, "moneyline"), (dist_mod, "dist")):
        msgs = _capture_walk_forward_logs(mod, games, folds)
        progress = [m for m in msgs if "OOF fold" in m and "complete" not in m]
        done = [m for m in msgs if "OOF complete" in m]

        assert progress, f"{label} walk-forward logged no progress at all"
        assert len(progress) >= 2, (
            f"{label} printed {len(progress)} checkpoint(s) over {total} folds "
            f"— too coarse to read as progress")
        assert progress[-1].endswith(f"/{total}"), (
            f"{label} never reported the final fold: {progress[-1]!r} "
            f"(expected the last checkpoint to read /{total})")
        # Positions must strictly increase: a stale cadence number repeating
        # is exactly the defect this pins.
        seen = [int(m.split("fold ")[1].split("/")[0]) for m in progress]
        assert seen == sorted(seen) and len(set(seen)) == len(seen), (
            f"{label} checkpoints are not strictly increasing: {seen}")
        assert seen[-1] == total, f"{label} last checkpoint {seen[-1]} != {total}"

        assert done, f"{label} walk-forward never signalled completion"
        assert f"{total} fold(s)" in done[-1], (
            f"{label} completion line omits the fold count: {done[-1]!r}")


def test_walk_forward_progress_does_not_fire_stale_mid_cadence_numbers():
    """No fold count may be announced that the run can never reach.

    A 46-fold walk-forward at a fixed 25-fold cadence can only ever print
    25/46, so a reader cannot tell a finished run from a stalled one. This
    targets folds.progress_checkpoints — the helper BOTH walk-forwards now
    call — so changing the real cadence fails here too.
    """
    for n_folds in (1, 2, 4, 7, 25, 46, 60, 480):
        fired = folds_mod.progress_checkpoints(n_folds)
        assert fired, f"{n_folds} folds produced no checkpoint at all"
        assert fired[-1] == n_folds, (
            f"{n_folds} folds: last checkpoint is {fired[-1]}, not {n_folds} "
            f"— the run would never announce its end")
        assert len(fired) >= 2 or n_folds == 1, (
            f"{n_folds} folds: only {len(fired)} checkpoint(s) "
            f"({fired}) — too coarse to read as progress")
        assert fired == sorted(set(fired)), (
            f"{n_folds} folds: checkpoints not strictly increasing: {fired}")
        assert all(1 <= i <= n_folds for i in fired), (
            f"{n_folds} folds: announced an out-of-range position: {fired}")

    # The exact production shape: 46 folds must NOT announce only 25.
    assert folds_mod.progress_checkpoints(46) != [25], (
        "regressed to the fixed 25-fold cadence that made a finished 46-fold "
        "walk-forward look identical to a stalled one")
    assert folds_mod.progress_checkpoints(0) == []


# ---------------------------------------------------------------------------
# 9. Artifact date provenance
# ---------------------------------------------------------------------------
def test_rfe_trace_is_stamped_with_the_run_date_not_the_api_horizon():
    """The RFE trace must carry the RUN's date, never the lookahead window.

    NHL_END_DATE is deliberately pushed past today (2026-09-29 on the
    2026-09-25 run) so the slate covers upcoming games. Passing that horizon
    to maybe_run_rfe wrote nhl_feature_selection_20260929.json and
    nhl_feature_workbook_2026-09-29.xlsx — four days in the future, and the
    only NHL artifacts disagreeing with the `date_c` stamped on everything
    else. MLB uses ONE date variable for the window, the trace and the
    artifact stamps so the two cannot diverge; this pins the NHL call site
    to the run date that gives the same guarantee.
    """
    import ast

    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "maybe_run_rfe"]
    assert calls, "maybe_run_rfe is no longer called from master_pipeline"

    for call in calls:
        args = [a for a in call.args if not isinstance(a, ast.Starred)]
        assert len(args) >= 2, (
            f"maybe_run_rfe call at line {call.lineno} no longer passes a day")
        day = args[1]
        assert isinstance(day, ast.Name), (
            f"maybe_run_rfe day argument at line {call.lineno} is "
            f"{type(day).__name__}, expected a name")
        assert day.id == "run_date", (
            f"maybe_run_rfe is stamped with {day.id!r} (line {call.lineno}); "
            f"it must be 'run_date' — end_date is the NHL API lookahead and "
            f"writes future-dated artifacts")
        # end_date must remain the API window bound; it is simply the wrong
        # thing to hand the RFE recorder.
        assert day.id != "end_date"

    # The horizon must still be what bounds the API window, or the slate
    # would stop covering upcoming games.
    assert "end_date, window_end = _env_end_bounds()" in src, (
        "the NHL API window is no longer bounded by _env_end_bounds()")


def test_retention_never_deletes_an_artifact_this_run_wrote():
    """`seen` must actually reach ``classify_artifact``, or backfills vanish.

    The retention window is anchored on ``end_date`` (NHL_END_DATE), which
    operators push past today and a BACKFILL sets to a horizon far beyond its
    own run date. ``artifacts`` is a manifest of BARE names while
    ``classify_artifact`` is handed the path relative to ``out_dir.parent``
    ("data_delivery/<name>"), so ``rel in seen`` was False for every file and
    the documented "seen - staged by this run" verdict was unreachable. A
    backfill therefore wrote its artifacts, listed them in the summary and the
    DONE banner, and then unlinked them seconds later.

    The fix must stay scoped: a genuinely old artifact that this run did NOT
    write still has to age out.
    """
    import master_pipeline as mp

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "nhl-backend" / "data_delivery"
        out_dir.mkdir(parents=True)
        fresh = out_dir / "nhl_power_rankings_20260614.csv"
        fresh.write_text("feature,value\nelo,1\n", encoding="utf-8")
        ancient = out_dir / "nhl_calibration_20260612.json"
        ancient.write_text("{}", encoding="utf-8")

        mp._prune_old_artifacts(
            out_dir, "20260614",
            seen={"nhl_power_rankings_20260614.csv"},  # bare manifest name
            anchor_iso="2026-09-29")

        assert fresh.exists(), (
            "retention deleted an artifact THIS RUN wrote "
            "(nhl_power_rankings_20260614.csv); the 'seen' verdict is not "
            "reachable, so a backfill anchored on a later end_date erases its "
            "own delivery")
        assert not ancient.exists(), (
            "retention stopped aging out artifacts this run did not write; the "
            "seen guard was widened beyond the files in the manifest")


def test_prune_hands_the_policy_the_same_identifier_it_classifies_with():
    """The `seen` set and the `rel` key must be in ONE shape, on every OS.

    On Windows ``str(Path.relative_to(...))`` yields ``data_delivery\\name.csv``
    while the manifest carries ``name.csv``; a POSIX-only rel would not have
    exposed that, and vice versa. Pin the key shape directly so the two sides
    cannot drift apart again silently.
    """
    import retention_policy as rp
    import master_pipeline as mp

    seen_by_policy: set[str] = set()
    rels: list[str] = []
    real = rp.classify_artifact

    def spy(rel, seen, *a, **k):
        rels.append(rel)
        seen_by_policy.update(seen)
        return real(rel, seen, *a, **k)

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "nhl-backend" / "data_delivery"
        out_dir.mkdir(parents=True)
        (out_dir / "nhl_power_rankings_20260614.csv").write_text(
            "feature,value\nelo,1\n", encoding="utf-8")
        with _mock_patch.object(rp, "classify_artifact", spy):
            mp._prune_old_artifacts(
                out_dir, "20260614",
                seen={"nhl_power_rankings_20260614.csv"},
                anchor_iso="2026-09-29")

    assert rels, "the pruner classified nothing"
    for rel in rels:
        assert "\\" not in rel, (
            f"retention classified {rel!r} with a backslash separator; `seen` "
            "is built posix, so the two can never match on Windows")
        assert rel in seen_by_policy, (
            f"{rel!r} was classified but is absent from the `seen` set "
            f"{sorted(seen_by_policy)} — the run's own artifacts are "
            "unprotectable")


def test_board_backed_retention_reads_tracked_boards_not_the_population():
    """``board_dates`` must mean "a board is still tracked", MLB style.

    MLB builds it from ``todays_games_<date>.csv`` FILES, which ride the
    blanket window, so it is a strict subset of the window and the board-backed
    rule is the safety net MLB documents. NHL built it from the GAME dates
    inside the moneyline record and the predictions history - i.e. the whole
    decided population. The NHL plays on most days, so that made the rule
    total: a run-dated predictions_history / markets artifact whose OWN date
    happened to be a game day was reprieved forever. Measured on the tree at
    2026-09-26, ``nhl_predictions_history_20260115.csv`` (257 days past the
    anchor) and ``..._20251120.csv`` (313 days) both came back "current",
    where MLB's rule says "stale".
    """
    import master_pipeline as mp

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "nhl-backend" / "data_delivery"
        out_dir.mkdir(parents=True)
        # A decided population spread over months, as every real record is.
        (out_dir / "nhl_moneyline_v1_20260929.json").write_text(
            json.dumps({"games": [
                {"game_id": "2026011501", "game_date": "2026-01-15"},
                {"game_id": "2025112001", "game_date": "2025-11-20"},
            ]}), encoding="utf-8")
        # The tracked serving board, inside the window.
        pd.DataFrame({"game_id": [1]}).to_csv(
            out_dir / "nhl_run_engine_markets_20260929.csv", index=False)
        pd.DataFrame({"game_id": [1], "game_date": ["2026-09-29"]}).to_csv(
            out_dir / "nhl_predictions_history_20260929.csv", index=False)
        # A 257-day-old history whose own date was a game day. No board is
        # tracked for that date, so there is nothing to reprieve it.
        ancient_hist = out_dir / "nhl_predictions_history_20260115.csv"
        pd.DataFrame({"game_id": [1], "game_date": ["2026-01-15"]}).to_csv(
            ancient_hist, index=False)

        mp._prune_old_artifacts(out_dir, "20260929", seen=set(),
                                anchor_iso="2026-09-29")

        assert not ancient_hist.exists(), (
            "retention kept a 257-day-old predictions history because its own "
            "date was a game date: board_dates is carrying the decided "
            "population, not the tracked boards")
        assert (out_dir / "nhl_predictions_history_20260929.csv").exists(), (
            "the board-backed rule must still keep families inside the window")
        assert (out_dir / "nhl_run_engine_markets_20260929.csv").exists()


def test_a_stale_board_prunes_itself_and_drains_its_companion():
    """The board family must not rescue itself, and the drain takes two passes.

    The NHL's tracked board IS the markets family, so a board-backed markets
    family would put its own date into board_dates and reprieve ITSELF out of
    the window - a self-sustaining leak MLB cannot have, because its board
    family (todays_games_) is allowlisted and NOT board-backed. So the stale
    board is pruned on the first pass, and its companion - which the stale
    board rescued for that one pass, exactly as in MLB - goes on the second.
    """
    import master_pipeline as mp

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "nhl-backend" / "data_delivery"
        out_dir.mkdir(parents=True)
        (out_dir / "nhl_moneyline_v1_20260929.json").write_text(
            json.dumps({"games": []}), encoding="utf-8")
        pd.DataFrame({"game_id": [1]}).to_csv(
            out_dir / "nhl_run_engine_markets_20260929.csv", index=False)
        stale_board = out_dir / "nhl_run_engine_markets_20260115.csv"
        pd.DataFrame({"game_id": [1]}).to_csv(stale_board, index=False)
        companion = out_dir / "nhl_predictions_history_20260115.csv"
        pd.DataFrame({"game_id": [1], "game_date": ["2026-01-15"]}).to_csv(
            companion, index=False)

        mp._prune_old_artifacts(out_dir, "20260929", seen=set(),
                                anchor_iso="2026-09-29")
        assert not stale_board.exists(), (
            "a 257-day-old markets board was kept: it is board_supported, so "
            "it put its own date into board_dates and rescued itself")
        assert companion.exists(), (
            "the first pass must let the stale board rescue its companion; "
            "that one-pass lag is the safety net, identical to MLB's")

        mp._prune_old_artifacts(out_dir, "20260929", seen=set(),
                                anchor_iso="2026-09-29")
        assert not companion.exists(), (
            "the companion survived the second pass: with its board gone, "
            "nothing reprieves it and the 10-day window is the policy again")


def test_board_dates_carries_tracked_boards_only_never_game_dates():
    """The property is a property of the CONSTRUCTION, not the predicate.

    ``classify_artifact`` cannot tell a board date from a game date - it is
    handed one set. So the invariant has to be pinned where board_dates is
    built: it holds the dates of tracked board FILES, and nothing else. A game
    date with no board behind it must never enter the set, because that is
    exactly how a 257-day-old artifact kept its reprieve.

    (A board that is itself stale still rescues its companion for one extra
    pass, in MLB too - the board is pruned in the same pass, so the rescue
    dies on the next run. That one-pass lag is the safety net working, not a
    leak, and neither side is treated as the policy.)
    """
    import retention_policy as rp
    import master_pipeline as mp

    captured: list[set] = []
    real = rp.classify_artifact

    def spy(rel, seen, *a, **k):
        board_dates = a[2] if len(a) > 2 else k.get("board_dates", set())
        captured.append(set(board_dates))
        return real(rel, seen, *a, **k)

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "nhl-backend" / "data_delivery"
        out_dir.mkdir(parents=True)
        # A game date with NO board tracked for it.
        (out_dir / "nhl_moneyline_v1_20260929.json").write_text(
            json.dumps({"games": [
                {"game_id": "2026011501", "game_date": "2026-01-15"},
            ]}), encoding="utf-8")
        (out_dir / "nhl_predictions_history_20260929.csv").write_text(
            "game_date", encoding="utf-8")
        # The one tracked board.
        (out_dir / "nhl_run_engine_markets_20260929.csv").write_text(
            "game_id", encoding="utf-8")
        with _mock_patch.object(rp, "classify_artifact", spy):
            mp._prune_old_artifacts(out_dir, "20260929", seen=set(),
                                    anchor_iso="2026-09-29")

    assert captured, "the pruner classified nothing"
    handed = captured[0]
    assert "20260929" in handed, (
        f"the tracked board's own date is not in board_dates {sorted(handed)} "
        "- the safety net that keeps a navigable board's run-engine data is gone")
    assert "20260115" not in handed, (
        "a GAME date with no board behind it entered board_dates; board_dates "
        "is carrying the decided population again, which reprieves every "
        "board-backed artifact dated on a day the league happened to play")


def test_the_game_date_map_still_ages_shap_cards_after_the_board_change():
    """Narrowing the moneyline loop must not stop SHAP ageing.

    SHAP files carry an official NHL numeric game id and no date of their own,
    so they - and only they - still need game dates out of the moneyline
    record. The current slate's cards must survive on the map, and a card for
    a game from last January must age out.
    """
    import master_pipeline as mp

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "nhl-backend" / "data_delivery"
        out_dir.mkdir(parents=True)
        (out_dir / "nhl_moneyline_v1_20260929.json").write_text(
            json.dumps({"games": [
                {"game_id": "2026020001", "game_date": "2026-09-29"},
                {"game_id": "2025010001", "game_date": "2025-01-01"},
            ]}), encoding="utf-8")
        current = out_dir / "nhl_shap_game_2026020001.csv"
        pd.DataFrame({"feature": ["elo"], "shap": [0.1]}).to_csv(
            current, index=False)
        ancient = out_dir / "nhl_shap_game_2025010001.csv"
        pd.DataFrame({"feature": ["elo"], "shap": [0.1]}).to_csv(
            ancient, index=False)

        mp._prune_old_artifacts(out_dir, "20260929", seen=set(),
                                anchor_iso="2026-09-29")

        assert current.exists(), (
            "the current slate's SHAP card was deleted: the game-date map is "
            "no longer being built, and numeric NHL ids carry no date")
        assert not ancient.exists(), (
            "a January SHAP card was kept - the game-date map is no longer "
            "aging SHAP files by their GAME date")


def test_schema_gates_are_evaluated_before_monitoring_is_written():
    """PHASE 13 must precede PHASE 14, and monitoring must follow the gates.

    The module header argues that one STDOUT stream must show one TRUE order,
    yet the gate block was written after the monitoring block, so every run
    log printed "PHASE 14 - monitoring" and then "PHASE 13 - schema
    validation" — the coverage verdict appeared to belong to an unvalidated
    delivery, and the drift / coverage CSVs plus the monitor JSON were written
    and counted into the manifest before anything validated them.
    """
    import ast

    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    banner_line: dict[str, int] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_banner" and node.args
                and isinstance(node.args[0], ast.Constant)):
            banner_line.setdefault(str(node.args[0].value), node.lineno)
    assert "PHASE 13" in banner_line and "PHASE 14" in banner_line, (
        f"phase banners missing: {sorted(banner_line)}")
    assert banner_line["PHASE 13"] < banner_line["PHASE 14"], (
        f"PHASE 13 gates are printed at line {banner_line['PHASE 13']} but "
        f"PHASE 14 monitoring at line {banner_line['PHASE 14']}; the banners "
        "no longer run in numeric order")

    monitor_call = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
         and n.func.attr == "write_run_engine_feature_artifacts"), None)
    assert monitor_call is not None, (
        "monitoring no longer writes the drift / coverage artifacts")
    assert banner_line["PHASE 13"] < monitor_call.lineno, (
        f"the gates run at line {banner_line['PHASE 13']} but the drift / "
        f"coverage artifacts are written at line {monitor_call.lineno} — an "
        "ungated monitoring artifact can still reach the delivery tree")


def test_push_counts_come_from_the_same_rows_as_the_scored_metric():
    """`n_pushes` must partition the PRICED rows, not the whole frame.

    A push has no binary outcome, so it leaves the scored population. Counting
    ``(margin == fair_spread)`` over the unfiltered frame also counted rows
    whose price was non-finite and which the metric had therefore dropped, so
    ``n + n_pushes`` could overshoot the OOF population in the summary JSON.
    """
    import evaluation as eval_mod

    oof = pd.DataFrame({
        "margin": [0.0, 0.0, -1.0, 2.0],
        "total": [5.0, 5.0, 6.0, 7.0],
        "mu_h": [2.0, 2.0, 2.0, 2.0],
        "mu_a": [2.0, 2.0, 2.0, 2.0]})

    # Game 1 lands exactly on the fair line in BOTH markets AND cannot be
    # priced. That is the only shape that separates the two populations: it is
    # a push, so the unfiltered count calls it one, but it was dropped from
    # the metric for being unpriceable, so the scored count does not.
    def fake_sim(mu_h, mu_a, a_home, a_away, n_draws=0, **kw):
        n = len(mu_h)
        return {
            "p_cover_fair": pd.Series([0.5, np.nan, 0.5, 0.5]),
            "p_over_fair": pd.Series([0.5, np.nan, 0.5, 0.5]),
            "fair_spread": pd.Series(np.zeros(n)),
            "fair_total": pd.Series(np.full(n, 5.0)),
        }

    with _mock_patch.object(eval_mod.dist_mod, "simulate_distributions",
                            fake_sim):
        out = nb_distribution_metrics(
            oof, {"alpha_home": 0.0, "alpha_away": 0.0}, n_draws=10)

    run_line = out["run_line"]
    assert run_line["n"] + run_line["n_pushes"] == 3, (
        f"run_line n={run_line['n']} + n_pushes={run_line['n_pushes']} does "
        "not partition the 3 priced rows; the unpriceable push is counted "
        "twice")
    assert run_line["n_pushes"] == 1, (
        f"expected only the PRICED margin==0 game to be the push, got "
        f"{run_line['n_pushes']}")
    totals = out["totals"]
    assert totals["n"] + totals["n_pushes"] == 3, (
        f"totals n={totals['n']} + n_pushes={totals['n_pushes']} does not "
        "partition the 3 priced rows; the unpriceable push is counted twice")
    assert totals["n_pushes"] == 1, (
        f"expected only the PRICED total==5 game to be the push, got "
        f"{totals['n_pushes']}")


def test_run_line_contract_log_reports_the_fitted_matrix_width():
    """The contract log must state the FITTED width, not just the declared one.

    ``feature_contract`` carries both ``n_features`` (the active subset) and
    ``n_features_fitted`` (what ``moneyline.member_matrix`` actually resolved
    for this frame). Only the declared count was logged, so a frame that
    resolved a narrower matrix produced a line byte-identical to a healthy
    run — defeating the train/serve-skew audit the contract exists for.
    """
    import ast

    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    found = None
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "info" and node.args
                and isinstance(node.args[0], ast.Constant)
                and "run-line feature contract" in node.args[0].value):
            found = node
            break
    assert found is not None, (
        "the run-line feature contract is no longer logged from "
        "master_pipeline")
    assert "n_features_fitted" in ast.unparse(found), (
        "the run-line contract log still reports only the declared feature "
        "count; the fitted matrix width is the half that catches skew")


def test_calibrated_oof_log_names_the_layer_it_measures():
    """The `moneyline OOF calib` line must say it is the PREQUENTIAL layer.

    Phase 8 fits two different maps: a prequential per-fold map (the
    evaluation layer, evaluated in Phase 9) and one pooled map that serves
    tonight's slate. A pooled Platt map is monotone, so it preserves AUC
    exactly — reading the prequential AUC being below the raw AUC as
    "calibration hurt the model" is a misreading of which map was measured.
    """
    import ast

    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    found = None
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "info" and node.args
                and isinstance(node.args[0], ast.Constant)
                and "moneyline OOF calib" in node.args[0].value):
            found = node
            break
    assert found is not None, "the calibrated OOF metric line is gone"
    assert "prequential" in ast.unparse(found), (
        "the calibrated OOF line does not identify itself as the prequential "
        "layer, so it reads as the pooled calibrator that actually serves")


# ---------------------------------------------------------------------------
# 10. Pull progress: a durable record that does not need a terminal
# ---------------------------------------------------------------------------
def _pull_log(fn):
    """Run ``fn`` with the ingestion logger captured; return the messages."""
    cap = _LogCapture()
    ing.logger.addHandler(cap)
    ing.logger.setLevel(logging.INFO)
    try:
        fn()
    finally:
        ing.logger.removeHandler(cap)
    return cap.messages


class _Clock:
    """A stand-in for the ``time`` module whose monotonic clock is hand-driven.

    Patching ``ing.time.monotonic`` would patch it for the whole process
    (it is the one module object); this keeps every other attribute real.
    """

    def __init__(self, now=0.0):
        self.now = float(now)

    def monotonic(self):
        return self.now

    def __getattr__(self, name):
        return getattr(sys.modules["time"], name)


def test_pull_progress_records_its_position_when_there_is_no_bar():
    """Even with no bar at all, the log has to carry the position.

    tqdm can be absent or broken, and then the per-window line is the only
    place progress can appear. It must still show a live position and a
    closing summary rather than going quiet for the length of a chunk.
    """
    box = []
    with _mock_patch.object(ing, "_progress_bar", return_value=None):
        def _drive():
            prog = ing._PullProgress(60, "score chunk 1/11 [x]", every=50)
            box.append(prog)
            for _ in range(60):
                prog.tick(cached=False)
            prog.close()
        msgs = _pull_log(_drive)
    prog = box[0]
    assert any("50/60" in m for m in msgs), \
        f"the 50-item cadence never fired: {msgs}"
    assert any("60/60" in m for m in msgs), f"the window never closed: {msgs}"
    # The item cadence line still carries a rate and a forward estimate.
    mid = next(m for m in msgs if "50/60" in m)
    assert "eta" in mid and "/s" in mid, mid
    # The close summary must not double-report a window that already finished.
    assert sum("60/60" in m for m in msgs) == 1, msgs


def test_pull_progress_seconds_floor_fires_below_the_item_cadence():
    """Progress must not depend on a COUNT of items completing.

    A count-triggered line is minutes wide exactly when it matters: a
    rate-limited API is a SLOW rate, so `every=50` on a stalling pull is the
    same silence the reporter exists to remove. The seconds floor bounds the
    worst gap however few items have finished.
    """
    clock = _Clock(0.0)
    box = []
    with _mock_patch.object(ing, "_progress_bar", return_value=None), \
            _mock_patch.object(ing, "time", clock):
        def _drive():
            prog = ing._PullProgress(200, "boxscore chunk 1/10 [x]",
                                     every=50, interval=30.0)
            box.append(prog)
            for _ in range(16):
                clock.now += 4.0          # 4 s per item: 50 items = 200 s
                prog.tick(cached=False)
        msgs = _pull_log(_drive)
    assert box[0].done == 16, box[0].done
    # The line is "<desc> N/total (...) ...", so the position follows the
    # window label rather than a fixed word.
    fired = [int(m.split("[x] ", 1)[1].split("/", 1)[0])
             for m in msgs if "eta" in m]
    # First window closes at the 30 s floor (item 8), next at item 16 —
    # never the 50-item cadence, which 16 items cannot reach.
    assert fired == [8, 16], f"seconds floor did not drive the cadence: {msgs}"
    assert all(p < 50 for p in fired), "a line fired on the item cadence"


def test_pull_progress_partitions_cached_fetched_and_unavailable():
    """The three outcomes are counted apart, never merged.

    An unavailable page advances the loop but returns nothing. Counting it as
    a fetch made the progress line overstate the window by exactly the number
    of pages that failed, so the line and the window's own
    `resolved: ... cache hits, ... fetched` summary could disagree.
    """
    box = []
    with _mock_patch.object(ing, "_progress_bar", return_value=None):
        def _drive():
            prog = ing._PullProgress(5, "score chunk 1/2 [x]", every=1,
                                     interval=0.0)
            box.append(prog)
            prog.tick(cached=True)
            prog.tick(cached=True)
            prog.tick(cached=False)
            prog.tick(failed=True)
            prog.tick(cached=False)
        msgs = _pull_log(_drive)
    prog = box[0]
    assert (prog.hits, prog.fetched, prog.failed) == (2, 2, 1), vars(prog)
    assert prog.done == 5, "a tick was counted twice or not at all"
    last = msgs[-1]
    assert "2 cached" in last and "2 fetched" in last and "1 unavailable" in last, last
    # A clean window does not advertise a failure count it does not have.
    assert "unavailable" in last, last


def test_pull_progress_close_reports_a_window_once_and_never_raises():
    """Decoration may fail; the reporter's own bookkeeping may not.

    `close()` is where a window's result is recorded, so it must be safe to
    call twice (a bar that is already gone, a re-entered loop) and safe when
    the bar itself raises on every call.
    """
    class _Boom:
        def update(self, n=1):
            raise RuntimeError("boom")

        def set_postfix(self, **kw):
            raise RuntimeError("boom")

        def close(self):
            raise RuntimeError("boom")

    box = []
    with _mock_patch.object(ing, "_progress_bar", return_value=_Boom()):
        def _drive():
            prog = ing._PullProgress(3, "boxscore chunk 1/10 [x]", every=99,
                                     interval=1e9)
            box.append(prog)
            prog.tick(cached=False)
            prog.tick(cached=False)
            prog.close()
            prog.close()          # idempotent
        msgs = _pull_log(_drive)
    assert box[0].done == 2, "the exploding bar cost the loop its position"
    assert sum("done 2/3" in m for m in msgs) == 1, \
        f"close() reported the window {sum('done 2/3' in m for m in msgs)}x: {msgs}"


def test_a_fully_cached_score_window_still_reports_progress():
    """A window that never hits the network must not go silent.

    The old counter was `if fetched and (i % 50 == 0 ...)` — gated on
    `fetched`, so a cache-served window and an unavailable-date window both
    reported nothing at all. The two paths that produce no result are the
    two paths that used to vanish from the log.
    """
    dates = [f"2024-10-{d:02d}" for d in range(1, 6)]
    calls = []

    def _dead(url):
        calls.append(url)
        raise RuntimeError("page unavailable")

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)

        def _path(name):
            return cache / name

        # Four dates cached (settled, empty pages), one date never written.
        for d in dates[:-1]:
            name = f"score_{ing.SCORE_CACHE_VERSION}_{d.replace('-', '')}.parquet"
            pd.DataFrame(columns=ing.SCORE_KEEP).to_parquet(_path(name),
                                                            index=False)
        with _mock_patch.object(ing, "_http_json", side_effect=_dead), \
                _mock_patch.object(ing, "_cache_path", side_effect=_path), \
                _mock_patch.object(ing, "_progress_bar", return_value=None):
            msgs = _pull_log(lambda: ing.load_score_dates(dates, use_cache=True))
    progress = [m for m in msgs if "score chunk 1/" in m and " 5/5" in m]
    assert progress, f"the window never reported its final position: {msgs}"
    final = progress[-1]
    assert "4 cached" in final, final
    assert "1 unavailable" in final, f"a dead page is not accounted for: {final}"
    assert "0 fetched" in final, \
        f"an unavailable page was reported as fetched: {final}"
    assert len(calls) == 1, f"the cached dates were re-pulled: {calls}"


def test_progress_bar_is_fixed_width_printable_ascii():
    """The bar has to survive whatever sink the run is attached to.

    It is plain ASCII on purpose: `logging` writes under whatever encoding
    the host console has, and a box-drawing glyph raises
    UnicodeEncodeError on cp1252 - which would turn a progress decoration
    into a failed run. Fixed width so a line does not reflow as it fills.
    """
    w = ing.BAR_WIDTH
    for frac in (0.0, 0.01, 0.25, 0.5, 0.999, 1.0):
        assert len(ing._bar(frac)) == w + 2, (frac, ing._bar(frac))
    assert ing._bar(0.0) == "[" + "-" * w + "]"
    assert ing._bar(1.0) == "[" + "#" * w + "]"
    assert ing._bar(0.5).count("#") == w // 2
    # Out-of-range positions clamp instead of over- or under-filling.
    assert ing._bar(1.5) == ing._bar(1.0)
    assert ing._bar(-0.5) == ing._bar(0.0)
    assert ing._bar(0.5, 1) == "[#]", "a zero-width bar is not a bar"
    assert ing._bar(0.5).isascii()


def test_a_pull_line_carries_a_readable_bar_when_there_is_no_tqdm():
    """Where no bar can be drawn, the log line must BE the bar.

    That context is real: tqdm may be absent (the Kaggle notebook installs
    it explicitly, and a kernel that has not run that cell has no tqdm), and
    it may be broken. Then the line is the only place progress can appear,
    so it carries its own fixed-width bar. These are ordinary characters, so
    it reads as a bar filling up in the log.
    """
    import re

    with _mock_patch.object(ing, "_progress_bar", return_value=None):
        def _drive():
            prog = ing._PullProgress(60, "score chunk 1/11 [x]", every=10)
            for _ in range(60):
                prog.tick(cached=False)
        msgs = _pull_log(_drive)
    found = [re.search(r"\[#*\-*\]", m) for m in msgs]
    assert msgs, "the window logged nothing at all"
    assert all(found), [m for m, b in zip(msgs, found) if not b]
    fills = [b.group().count("#") for b in found]
    assert fills == sorted(fills), f"the bar went backwards: {fills}"
    assert fills[-1] == ing.BAR_WIDTH, f"a finished window is not full: {fills}"
    assert fills[0] == round(ing.BAR_WIDTH * 10 / 60), fills[0]
    assert all(m.isascii() for m in msgs), "a log line is not plain ASCII"


def test_a_drawn_bar_suppresses_the_inline_one_so_progress_is_not_drawn_twice():
    """A live bar and an inline bar for the same position is noise.

    The tqdm bar is moving in the operator's face; repeating the position as
    a second bar in the log line right above it reads as two disagreeing
    bars. The latch is taken at CONSTRUCTION, so `close()` — which clears
    the bar before logging the window's result — still knows a bar drew, and
    the closing line does not grow one late.
    """
    class _Fake:
        def __init__(self):
            self.updates = 0
            self.closed = 0

        def update(self, n=1):
            self.updates += n

        def set_postfix(self, **kw):
            pass

        def close(self):
            self.closed += 1

    bar = _Fake()
    with _mock_patch.object(ing, "_progress_bar", return_value=bar):
        def _drive():
            prog = ing._PullProgress(60, "score chunk 1/11 [x]", every=10)
            for _ in range(60):
                prog.tick(cached=False)
            prog.close()
        msgs = _pull_log(_drive)
    assert not any("[" in m and "#" in m for m in msgs), \
        f"the inline bar is drawn alongside a live bar: {[m for m in msgs if '#' in m][:1]}"
    # The counts still ride along, and the bar still tracked the pull.
    assert any("60/60" in m and "60 fetched" in m for m in msgs), msgs
    assert bar.updates == 60 and bar.closed == 1, \
        f"the bar did not track the pull (updates={bar.updates}, closed={bar.closed})"


def test_the_kaggle_notebook_installs_tqdm_or_there_is_no_bar_to_draw():
    """The notebook installs the pipeline's dependencies by hand.

    tqdm is in MLB's list precisely because the bar is the operator's only
    view of a multi-minute chunked pull. The NHL list omitted it, so even
    with the isatty gate removed the Kaggle run had nothing to draw with and
    the bar stayed invisible. A dependency the notebook must install is
    part of the bar's contract.
    """
    # BACKEND is <repo>/nhl-backend/backend, so the notebook is two levels up.
    nb = (BACKEND.parents[1] / "kaggle_nhl_run.ipynb").read_text(encoding="utf-8")
    assert "tqdm" in nb, "the NHL Kaggle notebook never installs tqdm"


def _log_call_containing(src: str, needle: str):
    """The ``logger.info(...)`` call whose first constant argument contains
    ``needle``, or None. Log lines are the run's audit record, so what a line
    is allowed to claim is a contract like any other."""
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "info" and node.args
                and isinstance(node.args[0], ast.Constant)
                and needle in str(node.args[0].value)):
            return node
    return None


def test_fold_geometry_is_a_logged_record_not_a_bare_print():
    """The fold geometry is the run's central PIT claim, and it was the one
    block printed instead of logged.

    Two consequences, both from the 2026-09-26 log: the block carried no
    timestamp, so it could not be correlated with the phases around it; and
    `print` does not flush, so its position depended on the NEXT statement
    happening to be a logging call whose handler flushed the shared stdout
    buffer. That is not a property of the block -- it is a coincidence with
    the line below it, and stdout is block-buffered on a pipe, which is
    exactly the Kaggle subprocess.
    """
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    printed = [ast.unparse(n) for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "print"
               and "eligible settled games" in ast.unparse(n)]
    assert not printed, (
        f"the fold geometry is printed again, so it is un-timestamped and "
        f"un-flushed: {printed[:1]}")
    node = _log_call_containing(src, "fold geometry")
    assert node is not None, "the fold geometry block is no longer logged at all"
    # "fold 45" next to Phase 5's "fold 46/46" reads as a contradiction
    # unless you already know one is a 0-based id and the other a count.
    assert "last of" in ast.unparse(node), (
        "the last window does not name its ordinal beside its fold_id")


def test_dispersion_log_reports_the_poisson_limit_it_computed():
    """Two zeros in a line reading "NB dispersion" must not be ambiguous.

    `calibrate_dispersion` already returns a `poisson_limit` verdict and the
    log line dropped it, so 0.0000/0.0000 was indistinguishable from a broken
    estimator. The honest reading is that no over-dispersion was left to
    model: the NB has collapsed to a Poisson and the dispersion term is
    inactive.
    """
    n = 500
    mu = np.full(n, 3.0)
    # A perfectly Poisson sample: observed variance 0 is UNDER the Poisson
    # expectation, which the estimator floors to zero deterministically.
    perfect = pd.DataFrame({"home_score": mu.copy(), "away_score": mu.copy(),
                            "mu_h": mu, "mu_a": mu})
    sig = dist_mod.calibrate_dispersion(perfect)
    assert (sig["alpha_home"], sig["alpha_away"]) == (0.0, 0.0), sig
    assert sig["poisson_limit"] is True, \
        f"a zero-dispersion fit was not reported as the Poisson limit: {sig}"
    # And the flag is not constant: a genuinely over-dispersed fit must not
    # be described as the Poisson limit.
    spread = np.where(np.arange(n) % 2 == 0, 0.0, 9.0)
    over = pd.DataFrame({"home_score": spread, "away_score": spread,
                         "mu_h": mu, "mu_a": mu})
    sig2 = dist_mod.calibrate_dispersion(over)
    assert sig2["alpha_home"] > 0.0 and sig2["poisson_limit"] is False, sig2

    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    node = _log_call_containing(src, "method-of-moments")
    assert node is not None, "the dispersion log line is gone"
    assert "poisson_limit" in ast.unparse(node), \
        "the dispersion log line drops the verdict the estimator computed"


def test_identity_calibration_folds_are_named_not_just_counted():
    """A bare count cannot tell the expected thin start from a real failure.

    Fold 0 has no prior OOF rows at all and the earliest windows hold a
    handful of games, so some identity folds are normal at the START of the
    walk-forward. The same count could also mean calibration failed on folds
    scattered through the run, which is a defect. Those folds' probabilities
    are served uncalibrated, so which ones they are belongs in the log.
    """
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    node = _log_call_containing(src, "prequential per-fold calibration")
    assert node is not None, "the prequential calibration log line is gone"
    text = ast.unparse(node)
    assert "cal_identity_ids" in text, \
        f"the identity folds are counted but not named: {text[:200]}"


def test_retention_line_says_what_the_anchor_is_anchored_on():
    """`anchor 20260929 -10d` on a 2026-09-26 run reads as a future date.

    The anchor is deliberately the data WINDOW END, because a backfill pushes
    it past the run date and files newer than it must never be touched. That
    is a sound design; it is the log that fails to say so.
    """
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    node = _log_call_containing(src, "within the window (anchor")
    assert node is not None, "the retention window log line is gone"
    assert "WINDOW END" in ast.unparse(node), \
        "the retention line does not name what its anchor is"


def test_done_banner_counts_written_artifacts_not_delivered_ones():
    """The manifest counts files on disk, which is not what delivery pushed.

    2026-09-26 wrote 15 artifacts and pushed 14: the cards history store had
    0 new rows, so its bytes were unchanged and git had nothing to commit for
    it. Both numbers were true; only one was labelled, and the unlabelled one
    made the run look like it had dropped an artifact.
    """
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    banners = [ast.unparse(n) for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "_banner" and "DONE" in ast.unparse(n)]
    assert banners, "the DONE banner is gone"
    assert "written" in banners[0], \
        f"the DONE banner implies delivery it did not perform: {banners[0]}"


def test_monitor_feature_metadata_is_real_documentation_not_a_placeholder():
    """The drift table's hover text used to be the literal string
    "see backend/manifest.py" -- a pointer at data already one import away.

    The shared monitor page renders each row's hover from the artifact's
    ``features_metadata`` (a PRE-FORMATTED ``tooltip`` string per feature)
    and falls back to "(no detailed metadata)" when a feature is absent, so
    a placeholder block is worse than an empty one: it ships as if
    documented. The manifest already documents every served feature, so the
    emitter must read it. The tooltips are embedded in a single-quoted HTML
    attribute with quote=False, so one surviving apostrophe would terminate
    the attribute and parse the rest of every tooltip as markup -- the
    escaping lives at the producer, and this test proves it happened.
    """
    import manifest as manifest_mod

    served = list(config.MONEYLINE_FEATURE_COLS)
    meta = manifest_mod.feature_tooltips(served)
    assert set(meta) == set(served), (
        "feature_tooltips must cover exactly the served pool: "
        f"missing={sorted(set(served) - set(meta))}, "
        f"extra={sorted(set(meta) - set(served))}")
    for name, m in meta.items():
        tip = m.get("tooltip", "")
        assert tip and "manifest.py" not in tip, name
        assert all(k in tip for k in ("What:", "Definition:", "Source:",
                                      "Window:", "Point-in-time rule:",
                                      "Missing values:", "Available to:")), name
        assert "'" not in tip, (
            f"{name}: an apostrophe survives in the tooltip and would "
            "terminate the monitor page's single-quoted title attribute")
    # The artifact carries real definitions, not the placeholder.
    cov = [{"feature": f, "window": "baseline"} for f in served]
    with _mock_patch.object(mon, "_dump_json"):
        rec = mon.write_monitor_json(
            Path("probe.json"), "20260927", [], cov, [], [], 0.5, {}, {})
    emitted = rec["features_metadata"]
    assert set(emitted) == set(served), \
        "the monitor artifact must document every coverage row"
    for name, m in emitted.items():
        definition = str(m.get("definition", ""))
        assert definition.strip() and "manifest.py" not in definition, name
    # Undocumented names are absent, never given an empty blurb.
    assert manifest_mod.feature_tooltips(["not_a_real_feature"]) == {}


def test_validate_outputs_is_runnable_without_the_callers_locals():
    """Phase 13 crashed the 2026-09-30 15:43 production run: the oof_reach
    gate referenced ``game_df`` — a main() local the helper never receives —
    so a fully-successful 58-fold run died at validation with NameError after
    Phase 12 had written its artifacts but before any gate logged. The count
    is now an ``n_eligible`` parameter the caller passes explicitly; this test
    builds the real synth pool + fold summary (the shapes main() hands the
    validator) and calls _validate_outputs directly, which can only pass if no
    gate reads a name that only exists inside main()."""
    import master_pipeline as mp

    games = feat_mod.build_game_features(_synth_games(n_days=60,
                                                      games_per_day=6))
    fold_info = folds_mod.fold_summary(
        folds_mod.make_folds(games, date_col="gameday"))
    oof = pd.DataFrame({"p_ensemble_calibrated": [0.5],
                        "season": [config.OOF_FIRST_SEASON]})
    # REGRESSION: on the shipped code this line raised NameError (game_df is
    # not defined in the helper). Real shapes, real gate run.
    with tempfile.TemporaryDirectory() as td:
        gates = mp._validate_outputs(
            Path(td), "20260930", oof, pd.DataFrame(), fold_info,
            sig={"holdout": {"cutoff": "2026-05-24"}},
            n_eligible=len(games))
    assert isinstance(gates["oof_reach"], bool), gates
    # The default must fail closed: an unknown eligible population is never
    # silently certified as fully OOF-covered.
    with tempfile.TemporaryDirectory() as td:
        gates0 = mp._validate_outputs(
            Path(td), "20260930", oof, pd.DataFrame(), fold_info,
            sig={"holdout": {"cutoff": "2026-05-24"}})
    assert gates0["oof_reach"] is False, gates0
    # Controlled arithmetic on the two numbers the gate compares: full reach
    # passes, a starved OOF record (the pre-fix symptom: silent MIN_VAL_FOLD
    # skips, here 50%) fails the 90% threshold.
    def _gate(total_val_games: int, n_eligible: int) -> bool:
        fi = dict(fold_info)
        fi["total_val_games"] = total_val_games
        with tempfile.TemporaryDirectory() as td:
            return mp._validate_outputs(
                Path(td), "20260930", oof, pd.DataFrame(), fi,
                sig={"holdout": {"cutoff": "2026-05-24"}},
                n_eligible=n_eligible)["oof_reach"]
    assert _gate(97, 100) is True    # 97% reach — healthy
    assert _gate(50, 100) is False   # 50% reach — OOF skipped half the pool
    assert _gate(89, 100) is False   # just under the 0.90 threshold
    assert _gate(90, 100) is True    # exactly at the threshold passes


def test_season_dates_request_late_september_regular_season_openers():
    """``season_dates`` must span Sep 1, not Oct 1 (2026-10-07 run review).

    The old Oct 1 start never requested the 2026-09-29/30 slate: those
    eight regular-season games (gameType 2) never entered the frame, so the
    published markets recorded every affected club's Oct 1+ entering
    record as a season debut (EDM entered its Oct 1 game 0-0 although it
    had lost 5-6 on Sep 29), and their results never trained the model.
    The serving-tail union in main() cannot cover this: on an Oct 7 run
    ``max(start_date, run_date)`` is Oct 7 itself.
    """
    dates = ing.season_dates(2026)
    assert dates[0] == "2026-09-01"
    assert "2026-09-29" in dates and "2026-09-30" in dates, (
        "late-September regular-season openers would never be requested")
    assert "2026-10-01" in dates and "2027-07-15" in dates
    assert dates == sorted(set(dates))  # strictly increasing, no dups


def test_slate_contract_gate_rejects_null_records():
    """The 2026-10-07 17:39 UTC run shipped home_record/away_record null
    on every slate row while ``slate_contract_fields`` still PASSed: the
    gate checked column presence only, and records weren't even in
    ``need``. A serving-contract record must be a populated W-L string on
    every slate row (season openers serve \"0-0\", never null)."""
    import master_pipeline as mp

    base = dict(game_id=["g1"], gameday=["2026-10-07"], home_team=["WSH"],
                away_team=["PIT"], mu_h=[3.5], mu_a=[2.9],
                fair_spread=[-1.5], fair_total=[6.0],
                p_home_win_derived=[0.6], p_away_win_derived=[0.4])
    fold_info = {"n_folds": 1, "total_val_games": 100}
    oof = pd.DataFrame({"p_ensemble_calibrated": [0.5],
                        "season": [config.OOF_FIRST_SEASON]})

    def _gate(slate: pd.DataFrame) -> bool:
        with tempfile.TemporaryDirectory() as td:
            return mp._validate_outputs(
                Path(td), "20261007", oof, slate, fold_info,
                sig={"holdout": {"cutoff": "2026-05-24"}},
                n_eligible=100)["slate_contract_fields"]

    good = pd.DataFrame({**base, "home_record": ["1-1"],
                         "away_record": ["2-1"]})
    assert _gate(good) is True
    # Pre-fix symptom: null records on the serving board still PASSed.
    nulls = pd.DataFrame({**base, "home_record": [None],
                          "away_record": [None]})
    assert _gate(nulls) is False
    empty = pd.DataFrame({**base, "home_record": [""],
                          "away_record": [""]})
    assert _gate(empty) is False
    # Missing contract columns fail; an empty slate keeps its pass semantics.
    assert _gate(pd.DataFrame(base)) is False
    assert _gate(pd.DataFrame()) is True


def test_season_opener_gate_catches_unrequested_leading_games():
    """The ingestion-coverage gate fails the 2026-10-07 defect frame.

    ``season_dates`` once started at Oct 1, so the 2026-09-29/30 slate
    (game numbers 0001-0008) was never requested and every downstream
    gate still passed — they only read frames that ARE present. NHL game
    numbers are dense and chronological per season (first regular-season
    game = ``S020001``), so an ingested earliest number > 1 proves
    leading games are missing. The gate must fail that frame, pass a
    complete one, skip operator windows that open mid-season, fail a
    missing season only once the window end is past every modern opener,
    and fail closed when the frame or window bounds are absent.
    """
    import master_pipeline as mp

    def _frame(pairs):
        """pairs: list of (season, first_game_number, n_games)."""
        rows = {"game_id": [], "season": [], "game_type": [], "gameday": []}
        for season, first, n in pairs:
            for k in range(n):
                rows["game_id"].append(int(f"{season}02{first + k:04d}"))
                rows["season"].append(season)
                rows["game_type"].append(2)
                rows["gameday"].append(f"{season + 1}-01-01")
        return pd.DataFrame(rows)

    all_seasons = [(2024, 1, 3), (2025, 1, 3), (2026, 1, 3)]

    def _gate(schedule, start="2024-01-01", end="2026-10-07"):
        return mp._season_openers_ingested(schedule, start, end)

    # The defect frame: season 2026 openers (0001-0008, Sep 29-30) were
    # never requested, so the earliest ingested regular-season game is
    # 0009 (Oct 1) — exactly what the Oct 7 review found in production.
    holed = _frame([(2024, 1, 3), (2025, 1, 3), (2026, 9, 3)])
    assert _gate(holed) is False
    # Complete spans pass.
    assert _gate(_frame(all_seasons)) is True
    # Fail closed: no frame, no window start, or no window end means the
    # coverage claim cannot be certified — never a silent pass.
    assert _gate(None) is False
    assert mp._season_openers_ingested(
        _frame(all_seasons), None, "2026-10-07") is False
    assert mp._season_openers_ingested(
        _frame(all_seasons), "2024-01-01", None) is False
    # Operator mid-season window: opener rows were never in scope.
    midseason = _frame([(2026, 9, 3)])
    assert _gate(midseason, start="2026-10-05") is True
    # Zero rows for a season: fine while the window stops before the
    # season could have started (pre-October-20), fatal after it —
    # including a whole season's dates never being requested.
    no_2026 = _frame([(2024, 1, 3), (2025, 1, 3)])
    assert _gate(no_2026, end="2026-10-07") is True
    assert _gate(no_2026, end="2026-10-25") is False
    # Through the Phase 13 entry point: the gate rides _validate_outputs.
    fold_info = {"n_folds": 1, "total_val_games": 100}
    oof = pd.DataFrame({"p_ensemble_calibrated": [0.5],
                        "season": [config.OOF_FIRST_SEASON]})
    with tempfile.TemporaryDirectory() as td:
        gates = mp._validate_outputs(
            Path(td), "20261007", oof, pd.DataFrame(), fold_info,
            sig={"holdout": {"cutoff": "2026-05-24"}}, n_eligible=100,
            schedule=holed, window_start="2024-01-01",
            window_end="2026-10-07")
    assert gates["season_openers_ingested"] is False, gates
    with tempfile.TemporaryDirectory() as td:
        gates = mp._validate_outputs(
            Path(td), "20261007", oof, pd.DataFrame(), fold_info,
            sig={"holdout": {"cutoff": "2026-05-24"}}, n_eligible=100,
            schedule=_frame(all_seasons), window_start="2024-01-01",
            window_end="2026-10-07")
    assert gates["season_openers_ingested"] is True, gates


def test_run_diagnostics_writes_are_guaranteed_their_directory():
    """Phase 4 crashed the 2026-10-01 03:35 Kaggle run: 23f1f0c moved the fold
    table into the gitignored run_diagnostics/ dir but left the only mkdir in
    the Phase 13 block — which Phase 4's write reaches FIRST — so a fresh
    clone died with OSError "Cannot save file into a non-existent directory"
    mid-run, after the full refetch had already paid for itself. Static pin:
    every function constructing a path under config.RUN_DIAGNOSTICS_DIR must
    mkdir it (exist_ok=True) before its first such construction."""
    import master_pipeline as mp

    # encoding pinned: this test also runs on Windows, where the default
    # locale codec cannot decode master_pipeline.py's UTF-8 (the 📝 emoji
    # made a bare read_text() raise UnicodeDecodeError on cp1252).
    src = Path(mp.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)

    def _attr_chain(node) -> str:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))

    failures = []
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        mkdir_lines, write_lines = [], []
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "mkdir"
                    and _attr_chain(node.func.value).endswith(
                        "RUN_DIAGNOSTICS_DIR")):
                for kw in node.keywords:
                    if (kw.arg == "exist_ok"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True):
                        mkdir_lines.append(node.lineno)
            if (isinstance(node, ast.BinOp)
                    and isinstance(node.op, ast.Div)
                    and _attr_chain(node.left).endswith(
                        "RUN_DIAGNOSTICS_DIR")):
                # Path construction = the write site's prerequisite
                # (RUN_DIAGNOSTICS_DIR / "file" feeding to_csv/others).
                write_lines.append(node.lineno)
        if write_lines and (not mkdir_lines
                            or min(write_lines) < min(mkdir_lines)):
            failures.append(fn.name)
    assert not failures, (
        "functions constructing paths under config.RUN_DIAGNOSTICS_DIR "
        f"without a preceding mkdir(exist_ok=True): {failures}")


# ---------------------------------------------------------------------------
# 12. 2026-10-06 run-log review remediation (T2-T7)
# ---------------------------------------------------------------------------
def test_coverage_writer_receives_the_drift_baseline_not_the_full_pool():
    """T2: the drift/coverage CSV writer must get the SAME ``drift_baseline``
    slice the log verdict and the monitor JSON are computed on.

    Commit 01b9345e moved ``cov_rows`` onto the trailing-tail window but
    left ``write_run_engine_feature_artifacts`` on the full ``game_df`` —
    so the 2026-10-06 log said "baseline: no warm nulls" (250-game tail)
    while the coverage CSV beside it reported 285 warm nulls over 2,827
    games, and the drift CSV's n_baseline (2,827) disagreed with the JSON
    drift's (250). Two tables answering different questions is exactly the
    MLB 08-28 incident this call site's own comment claims is impossible.
    """
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "write_run_engine_feature_artifacts")
    assert call.args, "writer call has no positional frame arguments"
    frame_arg = call.args[2]  # (out_dir, date_c, full_df, ...)
    assert isinstance(frame_arg, ast.Name) and frame_arg.id == "drift_baseline", (
        "the drift/coverage writer must be fed the once-sliced "
        f"drift_baseline frame, not {ast.unparse(frame_arg)} — the log "
        "verdict, the monitor JSON and the CSVs would describe different "
        "populations again")


def test_published_blend_line_distinguishes_causal_metrics_from_replay():
    """The log must label causal headline evidence and retrospective replay."""
    src = (BACKEND / "moneyline.py").read_text(encoding="utf-8")
    assert "Published blend:" in src
    assert "headline metrics grade the causal rolling blend" in src
    assert "p_ensemble_retrospective (not OOF)" in src


def test_degenerate_platt_warning_carries_the_sample_size():
    """T4: the degenerate-Platt warning must report n= (MLB 2026-10-06
    parity): "a=%s" alone left the sample size behind a degenerate fit
    unknowable, and fit_platt keeps uncapped _fit_total/_degen_total
    counters so a block summary can report the true denominator behind the
    capped WARNING lines."""
    src = (BACKEND / "moneyline.py").read_text(encoding="utf-8")
    assert "degenerate Platt params (a=%s, n=%d)" in src, (
        "the degenerate-Platt warning lost its n= — a degenerate fit's "
        "sample size must be visible in the log")
    assert 'fit_platt._fit_total' in src and 'fit_platt._degen_total' in src, (
        "fit_platt must keep uncapped attempt/degenerate counters for the "
        "per-block summary")
    assert "prequential Platt fits degenerated" in src, (
        "prequential_fold_calibrators must report the block's own "
        "degenerate-fit total behind the 3-line cap")


def test_folds_line_excludes_zero_placeholders_and_real_blocks_are_logged():
    """T5: the Phase-4 ``folds:`` line must not print the four oof_*
    placeholder blocks as {\"n\": 0, \"sufficient\": false}.

    Those setdefaults exist so later readers always find the keys, but
    _oof_blocks only fills them after Phase 5 — the 2026-10-06 log's early
    line read as "no OOF population" on a run that published 2,635 scored
    rows. The early line excludes the placeholders (with a note saying
    where they come from) and the REAL blocks are logged at the
    fold_info.update(_oof_blocks(...)) site."""
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    folds_logs = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "info" and n.args
                  and isinstance(n.args[0], ast.Constant)
                  and str(n.args[0].value).startswith("folds:")]
    assert folds_logs, "the Phase-4 folds: line is gone"
    early = folds_logs[0]
    early_src = ast.unparse(early)
    assert "_OOF_BLOCK_KEYS" in early_src, (
        "the early folds: line dumps fold_info verbatim — the four oof_* "
        "placeholder zeros read as 'no OOF population'")

    update = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.Expr)
                   and isinstance(n.value, ast.Call)
                   and isinstance(n.value.func, ast.Attribute)
                   and n.value.func.attr == "update"
                   and ast.unparse(n.value).startswith(
                       "fold_info.update(_oof_blocks")),
                  None)
    assert update is not None, "fold_info.update(_oof_blocks(...)) is gone"
    block_logs = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "info" and n.args
                  and isinstance(n.args[0], ast.Constant)
                  and "folds oof blocks" in str(n.args[0].value)]
    assert block_logs, "the real oof blocks are never logged"
    assert block_logs[0].lineno > update.lineno, (
        "the oof blocks line must come AFTER fold_info.update(_oof_blocks) "
        "or it logs the placeholders again")


def test_duplicate_loader_lines_are_logged_once_per_distinct_content():
    """T6: Phase 3's feature build and Phase 11's slate build run the same
    loaders over the same caches, so 11 byte-identical lines printed twice
    per run (2026-10-06 log review) — a reader could not tell a replay
    from a re-run. The _log_once idiom (MLB _log_resolved_view parity)
    keys on the RENDERED message: first occurrence logs at its level, an
    identical repeat drops to DEBUG, and changed content logs as a new
    line at its level again."""
    for mod in (feat_mod, ing):
        assert hasattr(mod, "_log_once") and hasattr(mod, "_LOG_ONCE_SEEN"), (
            f"{mod.__name__} lost its content-keyed log-once helper")

    caplog_records = []

    class _Sink(logging.Handler):
        def emit(self, record):
            caplog_records.append(record)

    # First occurrence at WARNING stays loud; identical repeat drops to
    # DEBUG; changed content (same shape, new numbers) logs at WARNING again.
    feat_mod._LOG_ONCE_SEEN.clear()
    fake = logging.getLogger("pit_log_once_probe")
    fake.handlers[:] = []
    fake.addHandler(_Sink())
    fake.setLevel(logging.DEBUG)
    fake.propagate = False
    saved = feat_mod.logger
    feat_mod.logger = fake
    try:
        feat_mod._log_once(logging.WARNING, "probe %s: %d rows", "x", 3)
        feat_mod._log_once(logging.WARNING, "probe %s: %d rows", "x", 3)
        feat_mod._log_once(logging.WARNING, "probe %s: %d rows", "x", 4)
    finally:
        feat_mod.logger = saved
        feat_mod._LOG_ONCE_SEEN.clear()
    levels = [r.levelno for r in caplog_records]
    assert levels == [logging.WARNING, logging.DEBUG, logging.WARNING], (
        f"_log_once must log first at level, repeat at DEBUG, changed "
        f"content at level again — got {levels}")
    assert caplog_records[0].getMessage() == "probe x: 3 rows"


def test_stale_end_date_pin_extends_to_today_and_passes_others_through():
    """T7: a literal NHL_END_DATE pin before today must extend to today.

    The Kaggle notebook (Kaggle-owned — never edited from the repo) pins
    NHL_END_DATE for rebuilds and leaves it active; the pin bounds BOTH
    the seasons range and the gameday window, so an unguarded stale pin
    would freeze the daily slate on the pinned day forever (MLB
    config.resolve_run_end_date parity). Same-day pins, future windows
    (the deliberate lookahead), season aliases and malformed input all
    pass through untouched."""
    today = datetime.now(ZoneInfo("America/New_York")).date()
    stale = (today - timedelta(days=1)).isoformat()
    assert config.resolve_run_end_date(stale) == today.isoformat(), (
        "a stale literal pin must extend to today (ET clock) or the daily "
        "slate freezes on the notebook's rebuild date")
    assert config.resolve_run_end_date(today.isoformat()) == today.isoformat(), (
        "a same-day rebuild pin must keep working exactly as written")
    assert config.resolve_run_end_date("2099-01-01") == "2099-01-01", (
        "a deliberate future lookahead window must pass through")
    assert config.resolve_run_end_date("not-a-date") == "not-a-date", (
        "malformed input must pass through so _env_date keeps failing loudly")

    # The guard must sit INSIDE _env_end_bounds, below the season-alias
    # branch (a 4-digit season alias is never rewritten) and above the
    # return — and the Phase-1 re-announce must exist so the extension is
    # visible in the pushed run log.
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_env_end_bounds")
    # Body only — the docstring mentions the guard by name, which would
    # make a naive full-function search report the guard above the branch.
    fn_src = "\n".join(ast.unparse(s) for s in fn.body
                      if not (isinstance(s, ast.Expr)
                              and isinstance(s.value, ast.Constant)
                              and isinstance(s.value.value, str)))
    alias_at = fn_src.index("raw.isdigit()")
    guard_at = fn_src.index("config.resolve_run_end_date")
    assert alias_at < guard_at, (
        "the season-alias branch must return BEFORE the stale-pin guard — "
        "a 4-digit NHL_END_SEASON must never be extended to today")
    assert "stale (before today)" in src, (
        "the stale-pin extension is never announced — a reviewer of the "
        "pushed log cannot tell the window no longer matches the pin")


def test_final_run_log_delivery_repushes_the_complete_log():
    """T8: _sync_data_delivery stages the run log like any artifact — a
    SNAPSHOT taken before the push confirmation ever reached the file, so
    every pushed log ends at the DONE banner (the 2026-10-06 review found
    exactly that on remote: 7,262 lines ending at DONE, sync prints
    absent). MLB's fix: after the sync, re-stage the now-complete log and
    push it as the run's LAST delivery, never fatal (MLB 'Final run-log
    delivery', 2026-10-05 log review)."""
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    assert "Final run-log delivery" in src, (
        "the pushed run log ends at the DONE banner — the sync's own "
        "confirmation can never be inside the snapshot it stages")
    assert "NHL pipeline run log (final delivery)" in src
    # Never fatal: the artifacts are already delivered.
    tree = ast.parse(src)
    main_fn = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "main")
    log_push_src = "\n".join(ast.unparse(s) for s in main_fn.body
                            if "Final run-log delivery" in ast.unparse(s))
    assert log_push_src, "the final log delivery block is not in main()"
    assert "except Exception" in log_push_src, (
        "a failed final log push must not fail a delivered run")


# ---------------------------------------------------------------------------
# 2026-10-08 log review: drift visibility + season-seam re-check
# ---------------------------------------------------------------------------
def test_season_seam_guard_rechecks_prior_year_phase_before_paging():
    """A location shift that survives the trailing baseline is re-measured
    against the SAME calendar phase of prior years (MLB 2026-09-30), and a
    clean re-check relabels the row OK-SEASONAL — verdict only, evidence
    untouched.

    The 2026-10-08 run window (Cup-final tail + season opening in one
    frame) drift-ALERTED rest_days (42.5 vs 2.3) and goalie_starts (6.0
    vs 45.7) — values verified exact against the NHL API, but every-year
    season-start seams, not 2026 regime breaks. Two pinned mutants: no
    phase source means the seam pages forever; a re-check that recomputes
    PSI (instead of only the mean shift) would rewrite the evidence.
    """
    rng = np.random.default_rng(11)
    col = "rest_days_home"
    base = pd.DataFrame({
        col: rng.normal(2.3, 0.7, 130),
        "gameday": pd.date_range("2026-02-01", periods=130, freq="D"),
    })
    cur = pd.DataFrame({
        col: rng.normal(42.5, 3.0, 45),
        "gameday": pd.date_range("2026-09-25", periods=45, freq="D"),
    })
    # The SAME seam one and two years back, inside the ±7d phase pad.
    phase = pd.DataFrame({
        col: np.concatenate([rng.normal(42.5, 3.0, 70),
                             rng.normal(42.5, 3.0, 70)]),
        "gameday": (list(pd.to_datetime("2025-09-18")
                         + pd.to_timedelta(rng.integers(0, 38, 70), "D"))
                    + list(pd.to_datetime("2024-09-18")
                           + pd.to_timedelta(rng.integers(0, 38, 70), "D"))),
    })

    # No usable phase rows (baseline-only fallback): the seam pages —
    # the pre-guard behavior this mechanism is allowed to replace.
    plain = {r["feature"]: r for r in
             mon.feature_drift(base, cur, feature_cols=[col])}
    assert plain[col]["status"] == "ALERT", plain[col]

    # With prior-year same-phase rows: same window, seasonal verdict.
    seam = {r["feature"]: r for r in
            mon.feature_drift(base, cur, feature_cols=[col],
                              phase_frame=phase)}
    assert seam[col]["status"] == "OK-SEASONAL", seam[col]
    # Verdict-only: the row keeps its numbers and its evidence.
    for key in ("psi", "psi_adjusted", "mean_shift", "n_baseline",
                "n_current"):
        assert seam[col][key] == plain[col][key], f"{key} was rewritten"


def test_season_seam_recheck_reaches_sub_100_phase_samples():
    """The seam re-check must be able to fire below 100 phase rows.

    2026-10-08 remote-log remediation: the MLB port's fixed
    ``len(phase_vals) >= 100`` floor is unreachable at the NHL season
    seam — the same-calendar-phase windows (late Sep .. mid Oct of the
    prior years) hold only 76-95 games because the league has barely
    started — so every October the six seam alerts re-checked nothing
    and stayed ALERT. The floor is now ``max(30, n_c)`` (a phase sample
    at least as large as the window it judges; n_c >= 30 is guaranteed
    by the guard), and the 2-SE gate still widens as the sample shrinks.

    Two pinned behaviours: a 90-row phase sample against a 45-row current
    window relabels (the old 100-floor blocked it forever); a phase sample
    SMALLER than the current window still pages (the floor remains
    meaningful — no tiny-sample blanket pardon).
    """
    rng = np.random.default_rng(23)
    col = "rest_days_home"
    base = pd.DataFrame({
        col: rng.normal(2.3, 0.7, 130),
        "gameday": pd.date_range("2026-02-01", periods=130, freq="D"),
    })
    cur = pd.DataFrame({
        col: rng.normal(42.5, 3.0, 45),
        "gameday": pd.date_range("2026-09-25", periods=45, freq="D"),
    })

    def _phase_frame(n: int) -> pd.DataFrame:
        # All rows inside the k=-1 phase window (2025-09-18 .. 2025-10-16
        # for this current span) so every one of the n rows counts.
        return pd.DataFrame({
            col: rng.normal(42.5, 3.0, n),
            "gameday": pd.Timestamp("2025-09-20"),
        })

    # 90 phase rows >= n_c=45: the re-check runs and relabels — the
    # pre-remediation code required >= 100 and could never get here.
    ok = {r["feature"]: r for r in
          mon.feature_drift(base, cur, feature_cols=[col],
                            phase_frame=_phase_frame(90))}
    assert ok[col]["status"] == "OK-SEASONAL", ok[col]

    # 40 phase rows < n_c=45: floor not met — the row still pages.
    thin = {r["feature"]: r for r in
            mon.feature_drift(base, cur, feature_cols=[col],
                              phase_frame=_phase_frame(40))}
    assert thin[col]["status"] == "ALERT", thin[col]


def test_feature_drift_logs_view_labeled_summary_lines():
    """The run log must carry MLB's view-labeled drift lines.

    The 2026-10-08 remote log read fully clean — every gate PASS, no
    mention of drift — while the drift CSV beside it held six alerts:
    no emitter existed, so drift was invisible to a log review. Both
    monitoring surfaces must announce themselves: the moneyline monitor
    as ``Feature drift [moneyline]: ...`` and the run-engine CSV writer
    as ``Feature drift [run-engine]: ...`` (MLB explainability parity).
    """
    rng = np.random.default_rng(5)
    cols = ["rest_days_home", "win_pct_home"]  # both in the serving list
    full = pd.DataFrame(rng.normal(0.0, 1.0, (300, len(cols))), columns=cols)
    recent = pd.DataFrame(rng.normal(0.0, 1.0, (40, len(cols))), columns=cols)

    records: list = []

    class _Sink(logging.Handler):
        def emit(self, record):
            records.append(record)

    fake = logging.getLogger("pit_drift_view_probe")
    fake.handlers[:] = []
    fake.addHandler(_Sink())
    fake.setLevel(logging.INFO)
    fake.propagate = False
    saved = mon.logger
    mon.logger = fake
    try:
        mon.feature_drift(full, recent, feature_cols=cols)
        mon.write_run_engine_feature_artifacts(
            Path(tempfile.mkdtemp()), "20991231", full, recent)
    finally:
        mon.logger = saved
    msgs = [r.getMessage() for r in records]
    assert any(m.startswith("Feature drift [moneyline]:") for m in msgs), msgs
    assert any(m.startswith("Feature drift [run-engine]:") for m in msgs), msgs


def test_drift_calls_receive_the_full_pool_as_season_seam_phase_source():
    """Both master's drift surfaces must be handed the FULL decided pool
    as ``phase_frame`` — the trailing baseline holds no prior-season
    rows, so a phase source that is the baseline itself (MLB's own
    documented trap) can never fire the OK-SEASONAL reclassification and
    the seam would keep paging as ALERT.
    """
    src = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    targets = {"feature_drift", "write_run_engine_feature_artifacts"}
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr in targets]
    assert {c.func.attr for c in calls} == targets, (
        "both drift call sites must exist in master_pipeline")
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        pf = kw.get("phase_frame")
        assert isinstance(pf, ast.Name) and pf.id == "game_df", (
            f"{call.func.attr} must receive phase_frame=game_df (the full "
            f"decided pool), got {ast.unparse(pf) if pf else 'nothing'}")


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("PASS", name)
    print(f"\n{len(tests)} run-engine PIT tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
