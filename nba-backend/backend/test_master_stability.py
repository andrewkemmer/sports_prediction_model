"""Score-means stability section (2026-10-18): unit contract.

The pipeline summary must answer \"did the new mus break derived ML?\"
on common decided games every run — the forensics the 2026-10-17 ensemble
promotion required by hand (-2.8 bps AUC / +6.9 bps LL, benign but
unwatched). Never fatal: missing prior, no overlap, or any error yields
a status dict, never an exception.
"""
import numpy as np
import pandas as pd
import pytest

import master_pipeline as mp
import serving


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


def _slate_card(probs, state, detail):
    return pd.DataFrame({
        "game_id": ["g1", "g2"],
        "gameday": pd.to_datetime(["2026-10-30", "2026-10-30"]),
        "home_team": ["BOS", "NYK"], "away_team": ["LAL", "MIA"],
        "game_state": [state, "pre"],
        "game_status_detail": [detail, "7:30 PM ET"],
        "start_time_utc": ["2026-10-30T23:00:00Z"] * 2,
        "home_win_prob_model": list(probs)})


def test_moneyline_card_locks_only_after_tipoff(tmp_path):
    # NHL parity (2026-11-18): the lock lives in the writer, which reads the
    # card it is about to replace - "the last number published before tip",
    # a value that is always on disk. A pre-tip game re-prices every run.
    card = tmp_path / "nba_moneyline_v1_20261030.json"
    first = _slate_card((0.62, 0.54), "in", "8:33 - 3rd Quarter")
    serving.write_moneyline_json(card, first, first.home_win_prob_model,
                                 first.home_win_prob_model)
    second = _slate_card((0.90, 0.60), "in", "8:33 - 3rd Quarter")
    record = serving.write_moneyline_json(card, second,
                                          second.home_win_prob_model,
                                          second.home_win_prob_model)
    by_id = {g["game_id"]: g for g in record["games"]}
    assert by_id["g1"]["home_win_prob_model"] == pytest.approx(0.62)
    assert by_id["g1"]["away_win_prob_model"] == pytest.approx(0.38)
    assert by_id["g1"]["home_win_prob_model_raw"] == pytest.approx(0.62)
    assert by_id["g2"]["home_win_prob_model"] == pytest.approx(0.60)


def test_moneyline_card_with_no_prior_publication_prices_fresh(tmp_path):
    card = tmp_path / "nba_moneyline_v1_20261030.json"
    slate = _slate_card((0.62, 0.54), "in", "8:33 - 3rd Quarter")
    record = serving.write_moneyline_json(card, slate,
                                          slate.home_win_prob_model,
                                          slate.home_win_prob_model)
    assert record["games"][0]["home_win_prob_model"] == pytest.approx(0.62)


def _strip(fair_total, mu_h, p_over, spread, offered):
    return pd.DataFrame({
        "kind": ["slate", "slate"],
        "game_id": ["g1", "g2"],
        "gameday": ["2026-10-30", "2026-10-30"],
        "game_state": ["in", "pre"],
        "game_status_detail": ["8:33 - 3rd Quarter", "7:30 PM ET"],
        "start_time_utc": ["2026-10-30T23:00:00Z"] * 2,
        "fair_total": fair_total, "mu_h": mu_h, "p_over_220": p_over,
        "spread_line": spread, "p_cover_offered": offered})


def test_markets_strip_locks_legs_only_after_tipoff(tmp_path):
    # The strip beside the Live badge locks too, on the same boundary: a
    # started game keeps the legs it last published before tip, so refetch
    # variance cannot wiggle a live card; a pre-tip game re-prices.
    path = tmp_path / "nba_run_engine_markets_20261030.csv"
    meta = tmp_path / "meta.json"
    serving.write_markets_csv(
        path, meta, pd.DataFrame(),
        _strip([224.5, 230.0], [112.0, 115.0], [0.51, 0.52],
               [-2.5, -3.5], [0.50, 0.53]))
    serving.write_markets_csv(
        path, meta, pd.DataFrame(),
        _strip([999.0, 231.0], [999.0, 115.5], [0.99, 0.52],
               [-9.5, -4.5], [0.99, 0.54]))
    out = pd.read_csv(path).set_index("game_id")
    live = out.loc["g1"]
    assert live["fair_total"] == pytest.approx(224.5)
    assert live["mu_h"] == pytest.approx(112.0)
    assert live["p_over_220"] == pytest.approx(0.51)
    # Live market lines stay fresh - they describe the market, not the model.
    assert live["spread_line"] == pytest.approx(-9.5)
    assert live["p_cover_offered"] == pytest.approx(0.99)
    assert out.loc["g2", "fair_total"] == pytest.approx(231.0)
    assert out.loc["g2", "mu_h"] == pytest.approx(115.5)


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



def test_the_card_stamps_and_ages_each_book_at_publish(tmp_path):
    """2026-11-18: the display pair and the offset input are different books
    with different clocks, and the age is measured at THIS write, not at the
    fetch that filled the frame."""
    now = pd.Timestamp.now(tz="UTC")
    tab = (now - pd.Timedelta(minutes=20)).isoformat()
    nov = (now - pd.Timedelta(minutes=4)).isoformat()
    slate = pd.DataFrame([{
        "game_id": "g1", "gameday": pd.Timestamp("2026-10-30"),
        "home_team": "BOS", "away_team": "LAL",
        "game_state": "pre", "game_status_detail": "7:30 PM ET",
        "start_time_utc": "2099-01-01T00:00:00Z",
        "home_win_prob_model": 0.6,
        "market_home_odds": 1.8, "market_away_odds": 2.1,
        "market_book": "novig", "price_utc": tab, "market_price_utc": nov}])
    rec = serving.write_moneyline_json(tmp_path / "c.json", slate,
                                       slate.home_win_prob_model,
                                       slate.home_win_prob_model)
    g = rec["games"][0]
    assert g["market_price_utc"].startswith(nov[:19])
    assert g["offset_price_utc"].startswith(tab[:19])
    assert g["market_stale_minutes"] == pytest.approx(4.0, abs=0.5)
    assert g["offset_stale_minutes"] == pytest.approx(20.0, abs=0.5)
