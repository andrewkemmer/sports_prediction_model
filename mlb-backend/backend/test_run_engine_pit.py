"""Point-in-time regression pins for the run engine (Poisson/NB totals +
spread-margin pricing).

The production OOF hit rate (~51% on favored-side cards) and the live
board's realized rate can only diverge through train/serve skew, so this
suite pins every PIT invariant the engine's honesty depends on:

  1. fold geometry: expanding, train strictly before validation;
  2. prequential Platt: a fold's map sees PRIOR folds only (poisoning the
     last fold cannot move earlier folds' probabilities);
  3. sealed holdout: α(λ) curve fitting never sees the holdout window;
  4. k-edge: fit on the pre-holdout mask only (poisoning sealed margins
     cannot move k); OOF markets and the slate board reprice through the
     SAME k (the wrapper seam), and the level λ_H+λ_A is preserved.

Run with: python mlb-backend/backend/test_run_engine_pit.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch as _mock_patch

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

import run_engine as re            # noqa: E402
import run_engine_k_edge as kx     # noqa: E402  (installs wrappers on import)
from training import canonical_walk_forward_splits  # noqa: E402


def _synth_oof(n_days: int = 40, games_per_day: int = 6, seed: int = 7
               ) -> pd.DataFrame:
    """Deterministic synthetic decided/OOF frame with the columns the run
    engine's PIT machinery needs (folds, holdout masks, k-edge fits)."""
    rng = np.random.default_rng(seed)
    rows = []
    pk = 5000
    base = pd.Timestamp("2026-07-01")
    for d in range(n_days):
        day = (base + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for g in range(games_per_day):
            pk += 1
            lam_h = 4.4 + rng.normal(0, 0.4)
            lam_a = 4.3 + rng.normal(0, 0.4)
            margin = rng.normal((lam_h - lam_a) * 1.2, 3.2)
            hs = int(max(0, round(lam_h + margin / 2)))
            as_ = int(max(0, round(lam_a - margin / 2)))
            rows.append({
                "game_pk": pk,
                "game_id": f"{day.replace('-', '')}_A{g}@H{g}",
                "game_date": day,
                "home_team": "HOM", "away_team": "AWY",
                "home_expected_runs": round(lam_h, 4),
                "away_expected_runs": round(lam_a, 4),
                "home_score": hs, "away_score": as_,
                "total_runs": hs + as_,
                "home_win": float(hs > as_),
                "fold_idx": d // 7,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Fold geometry
# ---------------------------------------------------------------------------
def test_run_engine_folds_are_expanding_and_strictly_prior():
    """Every fold's training games end strictly before its validation
    window starts, and training volume grows monotonically (expanding)."""
    oof = _synth_oof()
    decided, folds = canonical_walk_forward_splits(
        oof, retrain_cadence_days=7, max_eval_folds=0,
        min_train_days=0, min_val_games=1)
    assert len(folds) > 3, f"expected multiple folds, got {len(folds)}"
    sizes = [len(s["train_games"]) for s in folds]
    assert all(b >= a for a, b in zip(sizes, sizes[1:])), \
        f"training folds not expanding: {sizes}"
    for s in folds:
        tr_end = pd.to_datetime(s["train_games"]["game_date"]).max()
        va_start = pd.to_datetime(s["val_games"]["game_date"]).min()
        assert tr_end < va_start, (
            f"fold {s.get('fold_idx')}: train ends {tr_end.date()} on/after "
            f"validation start {va_start.date()}")


# ---------------------------------------------------------------------------
# 2. Prequential calibration sees prior folds only
# ---------------------------------------------------------------------------
def test_prequential_calibration_is_strictly_prior():
    """Poisoning the LAST fold's labels must not move any earlier fold's
    calibrated probability — each fold's Platt map is fit on prior folds
    only (post-F2 discipline shared with the binary moneyline)."""
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300).astype(float)
    p = np.clip(0.5 + rng.normal(0, 0.15, 300) + 0.25 * (y - 0.5), 0.02, 0.98)
    fold_idx = np.repeat(np.arange(6), 50)

    cal = re.prequential_calibrate(y, p, fold_idx)
    assert len(cal) == len(y) and np.isfinite(cal).all()
    assert (cal >= 1e-6).all() and (cal <= 1 - 1e-6).all()

    y_poisoned = y.copy()
    y_poisoned[fold_idx == 5] = 1.0 - y_poisoned[fold_idx == 5]
    cal_pois = re.prequential_calibrate(y_poisoned, p, fold_idx)
    early = fold_idx < 5
    np.testing.assert_array_equal(cal[early], cal_pois[early],
                                  err_msg="earlier folds moved by a later "
                                          "fold's labels — leakage")

    # Same property for the favored-side variant used by run-line markets.
    cal_f = re.prequential_favored_calibrate(y, p, fold_idx)
    cal_fp = re.prequential_favored_calibrate(y_poisoned, p, fold_idx)
    np.testing.assert_array_equal(cal_f[early], cal_fp[early])


# ---------------------------------------------------------------------------
# 3. Sealed holdout: α(λ) curves never see the holdout window
# ---------------------------------------------------------------------------
def test_alpha_curves_fit_on_pre_holdout_rows_only():
    """derive_markets_v3 must hand select_alpha_curve exactly the
    pre-holdout rows (dates < max − HOLDOUT_DAYS) — never the sealed tail."""
    oof = _synth_oof(n_days=40)
    seen: dict = {}
    orig = re.select_alpha_curve

    def spy(y, lam, seed=re.RANDOM_SEED):
        seen["n"] = len(y)
        return orig(y, lam, seed=seed)

    with _mock_patch.object(re, "select_alpha_curve", spy):
        re.derive_markets_v3(oof.copy())
    dates = pd.to_datetime(oof["game_date"])
    cutoff = dates.max() - pd.Timedelta(days=re.HOLDOUT_DAYS)
    n_pre = int((dates < cutoff).to_numpy().sum())
    assert seen.get("n") == n_pre, (
        f"α fit saw {seen.get('n')} rows, pre-holdout pool is {n_pre}")


# ---------------------------------------------------------------------------
# 4. k-edge: masked fit + one shared k across OOF and slate
# ---------------------------------------------------------------------------
def test_k_edge_fit_uses_only_pre_holdout_games():
    """fit_k_edge must be an OLS slope on the masked slice only: poisoning
    every SEALED game's margin by +1000 runs cannot move k by a whisker."""
    oof = _synth_oof(n_days=40)
    mask = kx.k_edge_holdout_mask(oof)
    dates = pd.to_datetime(oof["game_date"])
    cutoff = dates.max() - pd.Timedelta(days=re.HOLDOUT_DAYS)
    assert mask.sum() == int((dates < cutoff).to_numpy().sum())
    assert not mask[dates >= cutoff].any()

    lam_h = oof["home_expected_runs"].to_numpy(float)
    lam_a = oof["away_expected_runs"].to_numpy(float)
    margin = (oof["home_score"] - oof["away_score"]).to_numpy(float)
    k0 = kx.fit_k_edge(lam_h, lam_a, margin, mask)
    poisoned = margin.copy()
    poisoned[~mask] += 1000.0
    k1 = kx.fit_k_edge(lam_h, lam_a, poisoned, mask)
    assert k0 == k1, f"k moved from {k0} to {k1} on sealed-row poison"


def test_k_edge_oof_markets_reprice_through_the_fitted_k():
    """With an explicit k_edge, the OOF markets artifact must carry the
    EXPANDED λ pair (level preserved, edge scaled by k) — the same λ state
    the α curves and NB MC consumed."""
    oof = _synth_oof()
    k = 1.8
    mk = kx.derive_markets_v3(oof.copy(), k_edge=k)
    lh2, la2 = kx.apply_k_edge(
        oof["home_expected_runs"].to_numpy(float),
        oof["away_expected_runs"].to_numpy(float), k)
    markets = mk["markets"]
    np.testing.assert_allclose(
        markets["home_expected_runs"].to_numpy(float),
        np.round(lh2, 4), atol=1e-9)
    np.testing.assert_allclose(
        markets["away_expected_runs"].to_numpy(float),
        np.round(la2, 4), atol=1e-9)
    level_in = (oof["home_expected_runs"] + oof["away_expected_runs"]).to_numpy()
    level_out = (markets["home_expected_runs"]
                 + markets["away_expected_runs"]).to_numpy()
    np.testing.assert_allclose(level_out, level_in, atol=1e-6,
                               err_msg="k-edge changed the total (level)")
    assert mk["summary"]["k_edge"]["k"] == k
    assert mk["summary"]["k_edge"]["production_used"] is True


def test_k_edge_slate_reprices_through_the_same_k():
    """The daily seam must price the slate board from the SAME expanded λ
    pair as the OOF markets: expanded slate λ == level + k·(raw λ − level)
    of the raw-priced board, α columns follow the expanded λ, and the grid
    actually moves (the seam is not a no-op)."""
    decided = _synth_oof(n_days=30)
    slate = decided.head(3).drop(
        columns=["home_score", "away_score", "total_runs", "home_win",
                 "fold_idx"]).copy()
    slate["game_date"] = "2026-08-12"
    curves = {"home": {"form": "linear", "a": 0.3, "b": 0.02},
              "away": {"form": "linear", "a": 0.3, "b": 0.02}}
    rounds = {"home": 10, "away": 10}

    kx._K_EDGE_ACTIVE = None
    try:
        raw = kx.predict_slate_runs(decided, slate, rounds, curves,
                                    n_draws=2000, seed=1)
        kx._K_EDGE_ACTIVE = 1.8
        expanded = kx.predict_slate_runs(decided, slate, rounds, curves,
                                         n_draws=2000, seed=1)
    finally:
        kx._K_EDGE_ACTIVE = None

    mu = ((raw["home_expected_runs"].to_numpy(float)
           + raw["away_expected_runs"].to_numpy(float)) / 2.0)
    want_h = mu + 1.8 * (raw["home_expected_runs"].to_numpy(float) - mu)
    want_a = mu + 1.8 * (raw["away_expected_runs"].to_numpy(float) - mu)
    np.testing.assert_allclose(
        expanded["home_expected_runs"].to_numpy(float),
        np.round(want_h, 4), atol=1e-9)
    np.testing.assert_allclose(
        expanded["away_expected_runs"].to_numpy(float),
        np.round(want_a, 4), atol=1e-9)
    # α columns priced from the EXPANDED λ through the same curves
    np.testing.assert_allclose(
        expanded["alpha_home"].to_numpy(float),
        np.round(re.alpha_of(want_h, curves["home"]), 4), atol=1e-9)
    # the seam must actually move prices (k=1.8 is far from identity)
    assert not np.allclose(expanded["p_over_8_5"].to_numpy(float),
                           raw["p_over_8_5"].to_numpy(float)), \
        "slate grid identical with and without k-edge — seam is a no-op"


def test_k_edge_daily_cache_rejected_on_identity_mismatch():
    """The daily seam's module-level OOF cache may only seed the k fit when
    its identity (rows + date span) matches THIS run's decided frame. A
    cache left by a different frame (e.g. the run_margin_diff fallback's
    run_oof call) must be discarded and k fit on the fresh walk instead."""
    cache = _synth_oof(n_days=40)
    foreign_decided = _synth_oof(n_days=20, games_per_day=6, seed=9)
    fresh_walk = _synth_oof(n_days=20, games_per_day=6, seed=11)
    expected_k = kx.fit_k_edge(
        fresh_walk["home_expected_runs"].to_numpy(float),
        fresh_walk["away_expected_runs"].to_numpy(float),
        (fresh_walk["home_score"] - fresh_walk["away_score"]).to_numpy(float),
        kx.k_edge_holdout_mask(fresh_walk))

    calls = {"fresh_walk": 0}

    def _fake_run_oof(*_a, **_k):
        calls["fresh_walk"] += 1
        return {"oof": fresh_walk}

    def _fake_daily(*_a, **_k):
        return {"block": {}}

    kx._DAILY_OOF_CACHE = cache
    try:
        with _mock_patch.object(kx, "_orig_run_oof", _fake_run_oof), \
                _mock_patch.object(kx, "_orig_run_engine_daily", _fake_daily), \
                _mock_patch.object(re, "persist_oof",
                                   lambda *a, **k: Path("/tmp/noop.csv")):
            res = kx.run_engine_daily(
                None, None, "20260922", decided_snapshot=foreign_decided)
        assert calls["fresh_walk"] == 1, "cache should be discarded; fresh " \
            "walk must seed the k fit"
        assert res["block"]["k_edge"]["k"] == round(expected_k, 4)
    finally:
        kx._DAILY_OOF_CACHE = None


def test_k_edge_daily_cache_accepted_when_identity_matches():
    """When the cached OOF matches the run's decided frame, the seam keeps
    the zero-extra-walk optimization: no fresh run_oof, and k equals the
    fit on the cache's pre-holdout mask."""
    cache = _synth_oof(n_days=40)
    expected_k = kx.fit_k_edge(
        cache["home_expected_runs"].to_numpy(float),
        cache["away_expected_runs"].to_numpy(float),
        (cache["home_score"] - cache["away_score"]).to_numpy(float),
        kx.k_edge_holdout_mask(cache))

    calls = {"fresh_walk": 0}

    def _boom(*_a, **_k):
        calls["fresh_walk"] += 1
        raise AssertionError("fresh walk must NOT run when cache matches")

    def _fake_daily(*_a, **_k):
        return {"block": {}}

    kx._DAILY_OOF_CACHE = cache.copy()
    try:
        with _mock_patch.object(kx, "_orig_run_oof", _boom), \
                _mock_patch.object(kx, "_orig_run_engine_daily", _fake_daily), \
                _mock_patch.object(re, "persist_oof",
                                   lambda *a, **k: Path("/tmp/noop.csv")):
            res = kx.run_engine_daily(
                None, None, "20260922", decided_snapshot=cache.copy())
        assert calls["fresh_walk"] == 0
        assert res["block"]["k_edge"]["k"] == round(expected_k, 4)
    finally:
        kx._DAILY_OOF_CACHE = None


# ---------------------------------------------------------------------------
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
