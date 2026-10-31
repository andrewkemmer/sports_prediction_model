"""Score-means stability section (2026-10-18): unit contract.

The pipeline summary must answer \"did the new mus break derived ML?\"
on common decided games every run — the forensics the 2026-10-17 ensemble
promotion required by hand (-2.8 bps AUC / +6.9 bps LL, benign but
unwatched). Never fatal: missing prior, no overlap, or any error yields
a status dict, never an exception.
"""
import numpy as np
import pandas as pd

import master_pipeline as mp


def _frame(n=300, seed=4, mu_shift=0.0):
    rng = np.random.default_rng(seed)
    h = rng.normal(113., 5., n) + mu_shift
    a = rng.normal(110., 5., n) + mu_shift
    y = (rng.normal(h - a, 11.) > 0).astype(float)
    p = 1. / (1. + np.exp(-(h - a) / 8.))
    return pd.DataFrame({
        "game_id": [f"g{i}" for i in range(n)], "mu_h": h, "mu_a": a,
        "p_home_win_derived": np.clip(p, 1e-6, 1 - 1e-6),
        "y_home_win": y})


def test_common_games_delta_detects_a_mu_move(tmp_path):
    cur = _frame(mu_shift=1.0)
    _frame(mu_shift=0.0).to_csv(tmp_path / "nba_run_engine_markets_20261016.csv",
                                index=False)
    got = mp._score_means_stability(cur, tmp_path, "20261017")
    assert got["status"] == "ok"
    assert got["n_common"] == 300
    assert got["mean_abs_dmu_home"] == 1.0
    assert got["mean_abs_dmu_away"] == 1.0
    assert "derived_auc_delta_bps" in got
    assert "derived_logloss_delta_bps" in got


def test_no_prior_delivery_is_a_status_not_an_error(tmp_path):
    got = mp._score_means_stability(_frame(), tmp_path, "20261017")
    assert got == {"status": "no_prior_delivery"}


def test_insufficient_overlap_is_reported(tmp_path):
    _frame(n=10).to_csv(tmp_path / "nba_run_engine_markets_20261016.csv",
                        index=False)
    got = mp._score_means_stability(_frame(n=10), tmp_path, "20261017")
    assert got["status"] == "insufficient_overlap"


def test_missing_columns_never_raise(tmp_path):
    got = mp._score_means_stability(pd.DataFrame({"a": [1]}), tmp_path,
                                    "20261017")
    assert got["status"] == "missing_columns"
    assert mp._score_means_stability(None, tmp_path, "20261017") == \
        {"status": "no_current_oof"}


def _marketize_pair():
    rng = np.random.default_rng(11)
    n = 6
    hs = rng.poisson(112, n).astype(float)
    aws = rng.poisson(108, n).astype(float)
    base = pd.DataFrame({
        "game_id": [f"g{i}" for i in range(n)],
        "gameday": pd.date_range("2025-01-01", periods=n),
        "p_ensemble": np.full(n, 0.6),
        "home_score": hs, "away_score": aws,
        "home_win": (hs > aws).astype(float),
        "margin": hs - aws, "total": hs + aws})
    dist = pd.DataFrame({
        "game_id": [f"g{i}" for i in range(n)],
        "mu_h": np.clip(hs + rng.normal(0, 3, n), 1, None),
        "mu_a": np.clip(aws + rng.normal(0, 3, n), 1, None),
        "home_score": hs, "away_score": aws})
    return base, dist


def _frozen_pub(path, slate_date, created, probs):
    import json as _json
    games = [{"game_id": gid,
              "home_win_prob_model": shown,
              "home_win_prob_model_raw": raw}
             for gid, (shown, raw) in probs.items()]
    path.write_text(_json.dumps({
        "created_utc": created, "slate_date": slate_date, "games": games}),
                    encoding="utf-8")


def test_moneyline_freeze_is_first_published_wins(tmp_path):
    # An evening re-run must not re-price a game already published for the
    # same slate date: the earliest publication's numbers are the locked
    # pre-tip price. Never-published games price fresh.
    _frozen_pub(tmp_path / "nba_moneyline_v1_20261030.json",
                 "2026-10-30", "2026-10-30T10:00:00Z",
                 {"g1": (0.62, 0.60)})
    _frozen_pub(tmp_path / "nba_moneyline_v1_20261029.json",
                 "2026-10-29", "2026-10-29T10:00:00Z",
                 {"g1": (0.10, 0.10)})  # other date: must not match
    slate = pd.DataFrame({
        "game_id": ["g1", "g2"],
        "gameday": pd.to_datetime(["2026-10-30", "2026-10-30"]),
        "game_state": ["in", "pre"],
        "game_status_detail": ["8:33 - 3rd Quarter", "7:30 PM ET"],
        "p_ensemble": [0.99, 0.55],
        "p_ensemble_calibrated": [0.98, 0.54],
        "home_win_prob_model": [0.98, 0.54]})
    out = mp._apply_moneyline_freeze(slate, tmp_path)
    assert out.loc[out.game_id == "g1", "p_ensemble_calibrated"].iloc[0] \
        == 0.62
    assert out.loc[out.game_id == "g1", "p_ensemble"].iloc[0] == 0.60
    assert out.loc[out.game_id == "g1", "home_win_prob_model"].iloc[0] \
        == 0.62
    row2 = out.loc[out.game_id == "g2"].iloc[0]
    assert row2["p_ensemble_calibrated"] == 0.54  # fresh, never published


def test_moneyline_freeze_with_no_prior_publication_is_a_noop(tmp_path):
    slate = pd.DataFrame({
        "game_id": ["g9"],
        "gameday": pd.to_datetime(["2026-10-30"]),
        "p_ensemble": [0.55],
        "p_ensemble_calibrated": [0.54],
        "home_win_prob_model": [0.54]})
    out = mp._apply_moneyline_freeze(slate, tmp_path)
    assert out.loc[0, "p_ensemble_calibrated"] == 0.54


def test_markets_freeze_is_first_published_legs(tmp_path):
    # The strip beside the Live badge locks too: a re-run carries the first
    # publication's fair totals/means/grid legs for already-published games
    # (refetch variance cannot wiggle a live game), while never-published
    # games price fresh. No prior file: strict no-op.
    prior = pd.DataFrame({
        "kind": ["slate", "oof"],
        "game_id": ["g1", "g0"],
        "gameday": ["2026-10-30", "2026-10-29"],
        "fair_total": [224.5, 220.0],
        "mu_h": [112.0, 110.0],
        "p_over_220": [0.51, 0.50],
        "spread_line": [-2.5, -1.5],
        "p_cover_offered": [0.50, 0.50]})
    prior.to_csv(tmp_path / "nba_run_engine_markets_20261030.csv", index=False)
    slate_markets = pd.DataFrame({
        "kind": ["slate", "slate"],
        "game_id": ["g1", "g2"],
        "gameday": pd.to_datetime(["2026-10-30", "2026-10-30"]),
        "fair_total": [999.0, 230.0],
        "mu_h": [999.0, 115.0],
        "p_over_220": [0.99, 0.52],
        "spread_line": [-9.5, -3.5],
        "p_cover_offered": [0.99, 0.53]})
    out = mp._apply_markets_freeze(slate_markets, tmp_path)
    live = out.loc[out.game_id == "g1"].iloc[0]
    assert live["fair_total"] == 224.5
    assert live["mu_h"] == 112.0
    assert live["p_over_220"] == 0.51
    # Live market lines stay fresh - they describe the market, not the model.
    assert live["spread_line"] == -9.5
    assert live["p_cover_offered"] == 0.99
    fresh = out.loc[out.game_id == "g2"].iloc[0]
    assert fresh["fair_total"] == 230.0


def test_marketize_expand_false_defers_identical_pricing():
    # Runtime-only speedup contract: expand=False skips the identity-map
    # Monte Carlo pass (the caller reprices right after), and pricing
    # later yields byte-identical columns to pricing inside.
    import distributions as dist_mod
    base, dist = _marketize_pair()
    full = mp._marketize(base, dist, "oof")
    deferred = mp._marketize(base, dist, "oof", expand=False)
    # The deferred frame carries the merge + bookkeeping but no prices.
    assert not any(str(c).startswith("p_home_cover_")
                   for c in deferred.columns)
    assert "y_cover_fair" not in deferred.columns
    for col in ("game_id", "kind", "decided", "frame_view", "p_home_win",
                "mu_h", "mu_a"):
        pd.testing.assert_series_equal(deferred[col], full[col],
                                       check_names=False)
    priced = dist_mod.apply_distribution(deferred)
    pd.testing.assert_frame_equal(
        priced.sort_index(axis=1), full.sort_index(axis=1),
        check_dtype=False)
