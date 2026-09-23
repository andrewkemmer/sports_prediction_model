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
     SAME k (the explicit-arm wrapper seam), and the level λ_H+λ_A is
     preserved;
  5. k-edge MONITOR-ONLY retirement (2026-09-22): the daily seam fits and
     publishes the diagnostic k-hat but NEVER applies it — production
     prices the RAW λ pair (production_used is False; the board-level
     outcome is identical to k=1).

Run with: python mlb-backend/backend/test_run_engine_pit.py
"""
from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# 5. k-edge MONITOR-ONLY retirement (2026-09-22): fitted k published,
#    never applied — production prices the RAW lambda pair
# ---------------------------------------------------------------------------
def _fake_daily_cm():
    """Stub the underlying daily body so k-fit-focused tests don't need a
    real markets pass (mirrors the older k-edge daily tests)."""
    return _mock_patch.object(kx, "_orig_run_engine_daily",
                              lambda *a, **k: {"block": {}})


def test_k_edge_daily_fits_and_publishes_k_without_applying_it():
    """THE retirement property: the monitor-only daily seam fits the
    diagnostic k-hat AFTER the daily pass on the daily's own OOF (cached by
    _wrapped_run_oof — no second walk) and publishes it into
    block["k_edge"] with production_used=False / mode="monitor_only",
    while the daily body runs on the RAW lambdas — _K_EDGE_ACTIVE must be
    None during the pass and reset afterwards."""
    oof = _synth_oof(n_days=40)
    expected_k = kx.fit_k_edge(
        oof["home_expected_runs"].to_numpy(float),
        oof["away_expected_runs"].to_numpy(float),
        (oof["home_score"] - oof["away_score"]).to_numpy(float),
        kx.k_edge_holdout_mask(oof))
    seen = {"active": []}

    def _spy_daily(*_a, **_k):
        seen["active"].append(kx._K_EDGE_ACTIVE)
        # the real daily's _wrapped_run_oof caches the OOF during the pass
        kx._DAILY_OOF_CACHE = oof
        return {"block": {}}

    def _boom(*_a, **_k):
        raise AssertionError(
            "monitor-only k fit must NOT run a second walk-forward")

    kx._DAILY_OOF_CACHE = None
    try:
        with _mock_patch.object(kx, "_orig_run_oof", _boom), \
                _mock_patch.object(kx, "_orig_run_engine_daily", _spy_daily):
            res = kx.run_engine_daily(None, None, "20260922",
                                      decided_snapshot=oof.copy())
        meta = res["block"]["k_edge"]
        assert seen["active"] == [None], (
            "daily body must run with _K_EDGE_ACTIVE=None (no expansion)")
        assert kx._K_EDGE_ACTIVE is None, "seam must reset after the run"
        assert meta["k"] == round(expected_k, 4)
        assert meta["production_used"] is False
        assert meta["mode"] == "monitor_only"
        assert meta["basis"] == "daily-oof"
        assert meta["sampling_se"] is not None and meta["sampling_se"] > 0
        assert "drift_alert" in meta  # monitor contract keys survive
    finally:
        kx._DAILY_OOF_CACHE = None


def test_k_edge_daily_provided_walk_fallback_when_no_cache():
    """When the daily left no cached OOF, the explicit _fake_walk (test
    harness stand-in) seeds the fit and is labeled basis="provided-walk"."""
    oof = _synth_oof(n_days=40)
    expected_k = kx.fit_k_edge(
        oof["home_expected_runs"].to_numpy(float),
        oof["away_expected_runs"].to_numpy(float),
        (oof["home_score"] - oof["away_score"]).to_numpy(float),
        kx.k_edge_holdout_mask(oof))
    kx._DAILY_OOF_CACHE = None
    try:
        with _fake_daily_cm():
            res = kx.run_engine_daily(None, None, "20260922",
                                      decided_snapshot=oof.copy(),
                                      _fake_walk=oof)
        meta = res["block"]["k_edge"]
        assert meta["mode"] == "monitor_only"
        assert meta["k"] == round(expected_k, 4)
        assert meta["basis"] == "provided-walk"
        assert meta["production_used"] is False
    finally:
        kx._DAILY_OOF_CACHE = None


def test_k_edge_daily_no_oof_skips_k_record_without_crash():
    """No cached OOF and no provided walk: the k monitor skips the run with
    a warning — no k_edge record, no crash (a monitoring gap must never
    take down the daily pass)."""
    kx._DAILY_OOF_CACHE = None
    try:
        with _fake_daily_cm():
            res = kx.run_engine_daily(
                None, None, "20260922",
                decided_snapshot=_synth_oof(n_days=5))
        assert "k_edge" not in res["block"]
    finally:
        kx._DAILY_OOF_CACHE = None


def test_k_edge_explicit_arm_still_reprices_both_sides():
    """The explicit k_edge arm (offline A/B only) keeps the original
    guarantees: OOF markets and the slate board price through the SAME
    expanded lambda pair, level preserved, meta marks production_used."""
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
    level_in = (oof["home_expected_runs"]
                + oof["away_expected_runs"]).to_numpy()
    level_out = (markets["home_expected_runs"]
                 + markets["away_expected_runs"]).to_numpy()
    np.testing.assert_allclose(level_out, level_in, atol=1e-6,
                               err_msg="k-edge changed the total (level)")
    assert mk["summary"]["k_edge"]["k"] == k
    assert mk["summary"]["k_edge"]["production_used"] is True


# ---------------------------------------------------------------------------
# 6. Frozen Game-Totals prediction-history store (first publication, PIT)
# ---------------------------------------------------------------------------
def _th_synthetic_oof_frame(game_pk=123, *, p80=(0.46314, 0.46267),
                            p85=(0.37, 0.63), p90=(0.25, 0.70),
                            total_runs=9):
    """One decided OOF row with a small valid totals grid (8.0/8.5/9.0)."""
    return pd.DataFrame([{
        "game_pk": game_pk, "game_date": "2026-09-18", "kind": "oof",
        "home_score": 5, "away_score": 4, "total_runs": total_runs,
        "home_expected_runs": 4.0, "away_expected_runs": 4.3,
        "p_over_8_0": p80[0], "p_under_8_0": p80[1],
        "p_over_8_5": p85[0], "p_under_8_5": p85[1],
        "p_over_9_0": p90[0], "p_under_9_0": p90[1],
    }])


def _th_with_tmp_store(fn):
    """Run fn(tmp_path) with the store pointed at a hermetic tmp dir."""
    import tempfile
    original = re.DATA_DELIVERY_DIR
    tmp = Path(tempfile.mkdtemp(prefix="th_store_"))
    re.DATA_DELIVERY_DIR = tmp
    try:
        fn(tmp)
    finally:
        re.DATA_DELIVERY_DIR = original


def test_totals_history_store_prices_fair_line_and_rescaled_pick():
    # Re-scaled P(over|8.0) = 0.50025 -> fair line 8.0, Over pick; total 9
    # beats the line -> winner Over, correct 1.0; provenance ISO-dated.
    def body(tmp):
        out = re.update_totals_history_store(
            _th_synthetic_oof_frame(), "20260919")
        assert out is not None and out.exists()
        store = pd.read_csv(out, dtype={"game_pk": str})
        assert len(store) == 1
        r = store.iloc[0]
        assert r["game_pk"] == "123"
        assert float(r["line"]) == 8.0
        assert r["pick"] == "Over"
        assert abs(float(r["pick_prob"]) - 0.500254) < 1e-5
        assert r["winner"] == "Over"
        assert float(r["correct"]) == 1.0
        assert r["source_artifact_date"] == "2026-09-19"
    _th_with_tmp_store(body)


def test_totals_history_store_push_grading():
    # total_runs == whole-number fair line -> Push, correct excluded (NaN).
    def body(tmp):
        re.update_totals_history_store(
            _th_synthetic_oof_frame(total_runs=8), "20260919")
        store = pd.read_csv(re.DATA_DELIVERY_DIR
                            / "run_engine_totals_history.csv")
        r = store.iloc[0]
        assert r["winner"] == "Push"
        assert pd.isna(r["correct"])
    _th_with_tmp_store(body)


def test_totals_history_store_first_publication_wins_and_idempotent():
    # Once frozen, later runs (even with different prices) never mutate the
    # row; re-runs add nothing; only NEW game_pks append with own provenance.
    def body(tmp):
        re.update_totals_history_store(
            _th_synthetic_oof_frame(game_pk=123), "20260919")
        re.update_totals_history_store(
            _th_synthetic_oof_frame(game_pk=123, p80=(0.30, 0.70),
                                    p85=(0.20, 0.80), p90=(0.10, 0.90)),
            "20260922")
        store = pd.read_csv(re.DATA_DELIVERY_DIR
                            / "run_engine_totals_history.csv",
                            dtype={"game_pk": str})
        assert len(store) == 1
        assert abs(float(store.iloc[0]["pick_prob"]) - 0.500254) < 1e-5
        assert store.iloc[0]["source_artifact_date"] == "2026-09-19"
        re.update_totals_history_store(
            _th_synthetic_oof_frame(game_pk=456), "20260922")
        store = pd.read_csv(re.DATA_DELIVERY_DIR
                            / "run_engine_totals_history.csv",
                            dtype={"game_pk": str})
        assert len(store) == 2
        assert set(store["game_pk"]) == {"123", "456"}
        src = dict(zip(store["game_pk"], store["source_artifact_date"]))
        assert src == {"123": "2026-09-19", "456": "2026-09-22"}
    _th_with_tmp_store(body)


def test_totals_history_store_pk_dtype_normalized_and_never_raises():
    # Store pk stored as string; an int-pk artifact row must not duplicate.
    # A corrupt store must return None (logged), never raise.
    def body(tmp):
        re.update_totals_history_store(
            _th_synthetic_oof_frame(game_pk=123), "20260919")
        re.update_totals_history_store(
            _th_synthetic_oof_frame(game_pk=123), "20260920")  # int pk
        store = pd.read_csv(re.DATA_DELIVERY_DIR
                            / "run_engine_totals_history.csv")
        assert len(store) == 1
        (re.DATA_DELIVERY_DIR
         / "run_engine_totals_history.csv").write_text("not,a,store\n1,2")
        out = re.update_totals_history_store(
            _th_synthetic_oof_frame(game_pk=999), "20260921")
        assert out is None
        import json as _json
        meta = _json.loads((re.DATA_DELIVERY_DIR
                            / "run_engine_totals_history.meta.json")
                           .read_text(encoding="utf-8"))
        assert meta.get("error")
    _th_with_tmp_store(body)


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
