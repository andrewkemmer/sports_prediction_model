"""Hier OOF pricing is point-in-time by construction, never by convention.

Pins the _nba_hier_cdfs settled/unsettled split with a stubbed hier module
(the real one fits LightGBM trees; the contract under test is routing):
* settled rows (known outcome) price from block-refit PIT mus, and the pit
  frame rides into price_cdfs -- bundle trees never touch them;
* unsettled rows (slate/live) price from the full-frame bundle mus;
* a settled row without PIT coverage RAISES (pre-coverage history must be
  excluded up front, never priced with full-frame mus);
* a below-floor frame (no PIT walk can exist) prices from the bundle and
  stamps pit_floor_shortfall; the same frame above the floor raises;
* an all-unsettled frame prices bundle-only with pit=None.
"""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# Capture the real module BEFORE any test swaps a stub into sys.modules.
import nba_hier_mc as _REAL_HIER

_calls = {}


@pytest.fixture(autouse=True)
def _restore_hier_module():
    """Never let this file's stub escape into the rest of the session."""
    saved = sys.modules.get("nba_hier_mc")
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("nba_hier_mc", None)
        else:
            sys.modules["nba_hier_mc"] = saved


def _install_stub(*, pit_rows=None, prepared=True):
    """Stub nba_hier_mc in sys.modules; record price_cdfs arguments."""
    stub = types.ModuleType("nba_hier_mc")
    stub.load_bundle = lambda path=None: {"models": {"KR": {"r": 5.0},
                                                     "KE": {"r": 6.0}},
                                          "weight_tables": None}
    stub.ensure_tables = lambda frame, bundle: bundle
    stub.predict_step1 = lambda frame, bundle: (
        np.full(len(frame), 111.0), np.full(len(frame), 109.0))

    pit_df = None
    if pit_rows is not None:
        pit_df = pd.DataFrame(
            {"game_id": [g for g, *_ in pit_rows],
             "mu_h": [r[1] for r in pit_rows], "mu_a": [r[2] for r in pit_rows],
             "mu_KR": [r[3] for r in pit_rows],
             "mu_KE": [r[4] for r in pit_rows],
             "r_R": [5.0] * len(pit_rows), "r_E": [6.0] * len(pit_rows)})
    stub.lookup_pit_mus = (
        (lambda ids: None) if not prepared
        else (lambda ids: pd.DataFrame({"game_id": list(ids)}).merge(
            pit_df if pit_df is not None
            else pd.DataFrame(columns=["game_id", "mu_h", "mu_a",
                                       "mu_KR", "mu_KE", "r_R", "r_E"]),
            on="game_id", how="left")))

    def fake_price_cdfs(frame, mu_h, mu_a, bundle, n_draws=0, seed=0,
                        meta_out=None, pit=None):
        _calls["mu_h"] = np.asarray(mu_h, dtype=float)
        _calls["mu_a"] = np.asarray(mu_a, dtype=float)
        _calls["pit"] = None if pit is None else pit.copy()
        _calls["pit_none"] = pit is None
        return {"total": ([0], [1.0] * len(frame)),
                "margin": ([0], [1.0] * len(frame))}

    stub.price_cdfs = fake_price_cdfs
    sys.modules["nba_hier_mc"] = stub
    _calls.clear()


def _frame():
    return pd.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "gameday": pd.to_datetime(["2024-01-01"] * 3),
        "home_score": [110.0, np.nan, 105.0],
        "away_score": [100.0, np.nan, 108.0],
        "mu_h": [0.0, 0.0, 0.0], "mu_a": [0.0, 0.0, 0.0],
    })


def _hier_on(monkeypatch):
    import distributions
    # Patch the module object distributions itself reads: a bare ``import
    # config`` can resolve to a second copy on a multi-rootdir sys.path.
    monkeypatch.setattr(distributions.config, "NBA_DISTRIBUTION_ENGINE",
                        "hier_mc", raising=False)
    monkeypatch.setattr(distributions, "DISTRIBUTION_SPORT", "nba")
    return distributions


def test_settled_rows_use_pit_mus_not_bundle(monkeypatch):
    dist = _hier_on(monkeypatch)
    _install_stub(pit_rows=[("g1", 115.0, 104.0, 3.0, -1.0),
                            ("g3", 106.0, 107.0, 1.0, 0.5)])
    frame = _frame()
    dist._nba_hier_cdfs(frame, 100, 7, None)
    # Settled g1/g3 carry PIT mus; unsettled g2 carries bundle mus (111/109).
    assert _calls["mu_h"].tolist() == [115.0, 111.0, 106.0]
    assert _calls["mu_a"].tolist() == [104.0, 109.0, 107.0]
    assert _calls["pit"] is not None
    assert _calls["pit"]["game_id"].tolist() == ["g1", "g3"]
    assert _calls["pit"]["mu_KR"].tolist() == [3.0, 1.0]


def test_settled_row_without_pit_coverage_raises(monkeypatch):
    dist = _hier_on(monkeypatch)
    _install_stub(pit_rows=[("g1", 115.0, 104.0, 3.0, -1.0)])
    try:
        dist._nba_hier_cdfs(_frame(), 100, 7, None)
    except RuntimeError as exc:
        assert "without PIT mus" in str(exc)
    else:
        raise AssertionError("pre-coverage settled row priced without PIT mus")


def test_short_frame_without_walk_prices_bundle_and_discloses_it(monkeypatch):
    """Below the walk floor no PIT walk can exist (a fixture, a short
    backtest). With the champion fallback gone, refusing would mean hier can
    never price a short window: it prices from the bundle, and stamps
    pit_floor_shortfall so the run never reads as point-in-time.
    """
    dist = _hier_on(monkeypatch)
    _install_stub(pit_rows=None, prepared=False)
    meta = {}
    dist._nba_hier_cdfs(_frame(), 100, 7, meta)
    assert _calls["mu_h"].tolist() == [111.0, 111.0, 111.0]
    assert _calls["pit_none"] is True
    assert meta["pit_floor_shortfall"] == 2


def test_frame_above_the_walk_floor_without_walk_raises(monkeypatch):
    """A settled frame that clears the floor always yields walk blocks, so a
    missing walk there is an internal inconsistency: fail, never leak."""
    dist = _hier_on(monkeypatch)
    _install_stub(pit_rows=None, prepared=False)
    n = 1200
    frame = pd.DataFrame({
        "game_id": [f"g{i}" for i in range(n)],
        "gameday": pd.to_datetime(["2024-01-01"] * n),
        "home_score": np.full(n, 110.0), "away_score": np.full(n, 100.0),
        "mu_h": np.zeros(n), "mu_a": np.zeros(n)})
    with pytest.raises(RuntimeError, match="walk floor"):
        dist._nba_hier_cdfs(frame, 100, 7, {})
    assert _calls == {}, "hier priced rows despite having no PIT walk"


def test_all_unsettled_frame_is_bundle_only(monkeypatch):
    dist = _hier_on(monkeypatch)
    _install_stub(pit_rows=None)
    frame = _frame().iloc[[1]].copy()
    dist._nba_hier_cdfs(frame, 100, 7, None)
    assert _calls["pit_none"] is True
    assert _calls["mu_h"].tolist() == [111.0]


# ------------------------------------------------- walk-level PIT invariant
def test_walk_mus_never_see_the_game_they_score(monkeypatch):
    """Every covered game's mus come from a fit whose history ends BEFORE it.

    The walk is stubbed only at the tree boundary: _fit_core reports the last
    training date it was handed, and predict_* echo that date back as the
    "mu". A game priced at or before its own date would be lookahead.
    """
    hier_real = _REAL_HIER

    hier_real._HIST_CACHE.clear()
    hier_real._MUS_CACHE.clear()
    n = 1500
    dates = pd.date_range("2023-10-01", periods=n, freq="D")
    frame = pd.DataFrame({
        "game_id": [f"g{i:04d}" for i in range(n)],
        "gameday": dates,
        "home_score": 110, "away_score": 108,
    })

    epoch = pd.Timestamp("2000-01-01")

    def _days(ts):
        return float((pd.Timestamp(ts) - epoch).days)

    def fake_fit_core(tr, *a, **k):
        return {"train_end": pd.to_datetime(tr["gameday"]).max()}

    def fake_predict_step1(va, fb, *a, **k):
        # mu carries the last TRAINING day: a game priced from a fit that
        # saw it would show mu_h >= its own day.
        v = _days(fb["train_end"])
        return np.full(len(va), v), np.full(len(va), v)

    def fake_predict_residuals(va, mu_h, mu_a, fb, *a, **k):
        v = _days(fb["train_end"])
        return (np.full(len(va), v), np.full(len(va), v), 5.0, 6.0)

    monkeypatch.setattr(hier_real, "_fit_core", fake_fit_core)
    monkeypatch.setattr(hier_real, "predict_step1", fake_predict_step1)
    monkeypatch.setattr(hier_real, "predict_residuals", fake_predict_residuals)

    pit = hier_real.prepare_pit_mus(frame)
    assert pit is not None and len(pit) > 0
    got = pit.set_index("game_id").join(
        frame.set_index("game_id")[["gameday"]])
    game_day = got["gameday"].map(_days)
    # mu holds the last training day; it must strictly precede the game it
    # prices -- no block ever trains on its own games.
    assert (got["mu_h"] < game_day).all()
    assert (got["mu_KR"] < game_day).all()
    assert pit["r_R"].notna().all() and pit["r_E"].notna().all()
    # The first 10% (seed) has no prior history: absent, never infilled.
    assert not set(frame["game_id"].iloc[:600]) & set(pit["game_id"])
    # Lookup is a pure join onto the prepared store.
    j = hier_real.lookup_pit_mus(["g0700", "nope"])
    assert j["mu_h"].notna().tolist() == [True, False]
    _install_stub(pit_rows=[])

def test_walk_drops_uncovered_settled_rows_and_records_them(monkeypatch):
    """The walk prices only PIT-covered OOF rows, and says how many it dropped.

    Stubbed at the hier boundary (no trees fitted), so this pins the WALK's
    bookkeeping: pre-coverage settled rows leave the priced frame entirely
    (never NaN rows, never full-frame prices), and the count plus the date
    span land in the bundle meta the run log and markets meta publish.
    """
    dist = _hier_on(monkeypatch)
    rng = np.random.default_rng(11)
    n = 60
    frame = pd.DataFrame({
        "game_id": [f"g{i:03d}" for i in range(n)],
        "gameday": pd.date_range("2024-01-01", periods=n, freq="D"),
        "home_score": rng.integers(95, 125, n).astype(float),
        "away_score": rng.integers(95, 125, n).astype(float),
    })
    frame["mu_h"] = 110. + rng.normal(0, 2, n)
    frame["mu_a"] = 108. + rng.normal(0, 2, n)
    frame.index = [f"row{i:03d}" for i in range(n)]

    margin = np.arange(-110, 111, dtype=float)
    total = np.arange(0, 401, dtype=float)
    covered = set(frame["game_id"].iloc[6:])       # 6 pre-coverage rows

    stub = types.ModuleType("nba_hier_mc")
    stub.load_bundle = lambda path=None: {"models": {"KR": {"r": 5.0},
                                                     "KE": {"r": 6.0}},
                                          "weight_tables": None}
    stub.ensure_tables = lambda f, b: b
    stub.predict_step1 = lambda f, b: (np.full(len(f), 110.),
                                       np.full(len(f), 108.))
    stub.MIN_HISTORY_ROWS = 1200

    def fake_lookup(ids):
        ids = [str(g) for g in ids]
        if not covered & set(ids):
            return None
        return pd.DataFrame({
            "game_id": ids,
            "mu_h": np.where([g in covered for g in ids], 110., np.nan),
            "mu_a": np.where([g in covered for g in ids], 108., np.nan),
            "mu_KR": np.where([g in covered for g in ids], 0., np.nan),
            "mu_KE": np.where([g in covered for g in ids], 0., np.nan),
            "r_R": np.where([g in covered for g in ids], 5., np.nan),
            "r_E": np.where([g in covered for g in ids], 6., np.nan)})

    def fake_price(frame, mu_h, mu_a, bundle, n_draws=0, seed=0, meta_out=None,
                   pit=None):
        rows = len(frame)
        _calls["pit"] = pit
        cdf_m = np.tile(np.clip((margin + 20.) / 40., 0., 1.), (rows, 1))
        cdf_t = np.tile(np.clip((total - 200.) / 40., 0., 1.), (rows, 1))
        return {"margin": (margin, cdf_m), "total": (total, cdf_t)}

    stub.lookup_pit_mus = fake_lookup
    stub.prepare_pit_mus = lambda f, progress_every=0: pd.DataFrame(
        {"game_id": sorted(covered)})
    stub.price_cdfs = fake_price
    sys.modules["nba_hier_mc"] = stub
    _calls.clear()

    priced, bundle = dist.causal_distribution_frame(frame, n_draws=200)

    assert bundle.get("pit_excluded") == {"n": 6, "first": "2024-01-01",
                                          "last": "2024-01-06"}
    # Dropped rows are GONE from the priced frame, not left as NaN rows.
    assert len(priced) == n - 6
    assert not set(priced.index) & set(frame.index[:6])
    # And every row that WAS priced came with PIT pricing inputs.
    assert _calls["pit"] is not None and len(_calls["pit"]) > 0


# ------------------------------------- champion merge owns the price columns
def test_champion_merge_overwrites_the_old_engines_derived_moneyline():
    """The base OOF frame arrives pre-priced by the split-skew walk.

    A merge that skips columns already present on the base would let the
    champion write only game_id: the grid would come from the champion while
    p_home_win_derived/derived_ml/p_tie stayed on the old engine under a
    champion label. Every column the champion prices must win.
    """
    base = pd.DataFrame({
        "game_id": ["g1", "g2"],
        "p_home_win_derived": [0.50, 0.55],   # old engine's numbers
        "derived_ml": [0.50, 0.55],
        "p_tie": [0.03, 0.03],
        "y_home_win": [1, 0],})
    prices = pd.DataFrame({
        "game_id": ["g1", "g2"],
        "p_home_win_derived": [0.61, 0.42],   # champion's CDF at -1/0
        "derived_ml": [0.61, 0.42],
        "p_tie": [0.02, 0.02],
        "p_home_cover_0": [0.61, 0.42],
        "mu_h": [112.0, 108.0], "mu_a": [108.0, 112.0],})
    out = _REAL_HIER._merge_champion_prices(base, prices).set_index("game_id")
    assert out.p_home_win_derived.tolist() == [0.61, 0.42]
    assert out.derived_ml.tolist() == [0.61, 0.42]
    assert out.p_tie.tolist() == [0.02, 0.02]
    assert out.p_home_cover_0.tolist() == [0.61, 0.42]
    # Base columns the champion does not price survive the merge untouched.
    assert out.y_home_win.tolist() == [1, 0]
