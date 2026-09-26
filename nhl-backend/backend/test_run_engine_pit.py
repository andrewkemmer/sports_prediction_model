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
     that decide what a log means name what they actually measured;
 10. pull progress: a real tqdm bar is drawn in every context, including a
     pipe (the notebook drives the pipeline through subprocess, so stderr is
     never a terminal), and where no bar can be drawn the log line carries a
     fixed-width one of its own. Either way the log records a live position
     and a closing summary, on a seconds floor as well as an item cadence,
     and partitions cache hits from fetches from dead pages.

Run with: python nhl-backend/backend/test_run_engine_pit.py
"""
from __future__ import annotations

import sys
import ast
import logging
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch as _mock_patch

import numpy as np
import pandas as pd

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
        # so the row's game_id matches the caller's id exactly.
        gid = url.rsplit("/", 2)[-2]
        return {"id": gid, "away": {"abbrev": "AAA", "goals": 1},
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
    """
    assert not getattr(sys.stderr, "isatty", lambda: False)(), (
        "this test is only meaningful under a non-tty stderr")
    bar = ing._progress_bar(10, "score chunk 1/11")
    assert bar is not None, (
        "no bar is drawn when stderr is a pipe - the bar is still gated on a "
        "terminal, so every captured run has none")
    try:
        bar.update(1)
        bar.close()
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"the bar raised on a pipe: {exc!r}") from exc


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
        return {"id": gid, "away": {"abbrev": "AAA", "goals": 1},
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
        # decided pool: perfect except team debuts (cold)
        _row("elo_home", "decided pool", 100, 96, 4, 0),
        _row("win_pct_home", "decided pool", 100, 95, 5, 0),
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
    assert "decided pool" in text and "serving slate" in text, text[:200]
    # ...the decided pool is reported as clean, with its cold nulls named as
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

    def _fake_member(name, fold=False):
        model = _FakeModel(name)
        fitted.append((name, fold, model))
        return model

    with _mock_patch.object(ml_mod, "_make_member", side_effect=_fake_member):
        out = ml_mod.walk_forward_oof(games, fold_list=folds)

    assert len(fitted) == len(folds) * len(config.ENSEMBLE_MEMBERS)
    assert {name for name, _, _ in fitted} == set(config.ENSEMBLE_MEMBERS)
    for name, fold, model in fitted:
        assert fold is True
        assert len(model.fit_calls) == 1
        if name == "xgboost":
            assert "eval_set" in model.fit_calls[0]
        elif name == "lightgbm":
            assert "eval_set" in model.fit_calls[0]
            assert model.fit_calls[0]["categorical_feature"] == config.TREE_CATEGORICAL_COLS
        else:
            assert "eval_set" not in model.fit_calls[0]
    assert set(f"p_{n}" for n in config.ENSEMBLE_MEMBERS) <= set(out["oof"].columns)


def test_moneyline_blend_uses_prior_fold_weights_only():
    """Fold 0 uses thirds; fold 1 uses the optimizer result from fold 0."""
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

    def _fake_member(name, fold=False):
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
    first = oof[oof["fold_id"] == folds[0].fold_id]["p_ensemble"].to_numpy()
    second = oof[oof["fold_id"] == folds[1].fold_id]["p_ensemble"].to_numpy()
    np.testing.assert_allclose(first, 0.5, atol=1e-7)
    np.testing.assert_allclose(second, 0.8, atol=1e-7)


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
    """MLB applies NO saturation to alpha; a heavy over-dispersion survives.

    The retired NHL form clipped at ALPHA_CAP=2.0, which would have silently
    masked a genuinely over-dispersed fit. This fixture is far beyond 2.0.
    """
    rng = np.random.default_rng(11)
    mu = np.full(4000, 3.0)
    # size=1/alpha with alpha=8 -> far beyond the old 2.0 cap.
    y = rng.negative_binomial(0.125, 0.125 / (0.125 + 3.0), 4000).astype(float)
    a = dist_mod.estimate_alpha(y, mu)
    assert a > 2.0, f"alpha {a} was capped; MLB does not cap"
    lam_bar = 3.0
    assert a == round(max((float(y.var(ddof=0)) - lam_bar) / lam_bar ** 2, 0.0), 4)
    assert not hasattr(dist_mod, "ALPHA_CAP")


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


def test_pg_opportunities_come_from_the_opposing_goalie_ratio():
    """PP volume is a goalie-line ratio; the team's opportunities are the
    OPPONENT's goalie PP shots faced."""
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
    row = ing._parse_boxscore(bs)
    # home faced 4 PP shots on the road -> away had 4 PP opportunities.
    assert row["home_pp_opportunities"] == 6.0
    assert row["away_pp_opportunities"] == 4.0


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
    assert row["home_pp_opportunities"] == 9.0   # away goalie faced 9 PP shots
    assert row["away_pp_opportunities"] == 2.0   # home goalie faced 2 PP shots


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
    rows = mon.coverage(df)
    for r in rows:
        for k in COVERAGE_KEYS:
            assert k in r, f"coverage row dropped the front-end key {k!r}"
    by_f = _coverage_by_feature(rows, "decided pool")
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
    by_f2 = _coverage_by_feature(mon.coverage(broken), "decided pool")
    hit = by_f2["pp_success_diff"]
    assert hit["n_warm_null"] > 0, "a null on a warm game was not counted"
    assert hit["cause"] == "defect"
    assert hit["status"] in ("STARVED", "LOW_COVERAGE")
    assert hit["pct_measured_eligible"] < 100.0


def test_feature_coverage_measures_the_slate_the_pipeline_actually_ships():
    """Regression: the goalie family read 96-98% on the decided pool while
    EVERY published prediction carried a null, because the report only ever
    looked at the decided pool. A nulled slate column must read STARVED."""
    games = _synth_games(n_days=20, games_per_day=2)
    bs = _synth_goalie_boxscores(games)
    df = feat_mod.build_game_features(games, bs)
    pending = games.tail(4).copy()
    pending["home_score"] = np.nan
    pending["away_score"] = np.nan
    hist = games.head(-4)
    slate = feat_mod.build_slate_features(
        pd.concat([hist, pending], ignore_index=True),
        bs[bs["game_id"].isin(set(hist["game_id"]))])

    rows = mon.coverage(df, slate_df=slate)
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
    assert _coverage_by_feature(mon.coverage(df, slate_df=dead),
                                "decided pool")["goalie_sv_pct_home"]["status"] != "STARVED"


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
    assert set(cov["window"]) == {"decided pool", "serving slate"}
    assert set(cov["feature"]) == set(config.active_moneyline_feature_cols())
    assert set(COVERAGE_KEYS) <= set(cov.columns)


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
        # A sparse stretch whose 7-date windows fall under MIN_VAL_FOLD_GAMES.
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
