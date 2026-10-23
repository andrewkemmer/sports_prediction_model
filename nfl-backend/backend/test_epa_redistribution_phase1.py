"""Phase-1 availability redistribution tests (2026-10-22, challenger).

Covers the pure helper ``_phase1_redistribute_tpg`` only: same-position
proportional absorption of a removed player's trailing ``_tpg``, partial
conservation (skill alpha 0.7 / QB 1.0), p95-style caps with waterfall,
remainder vanishes. The helper must never touch rates (epa_q) -- the
caller keeps ``_epa_per_game = adjusted_tpg * epa_q``.
"""
import numpy as np
import pandas as pd

try:
    from backend import features as feat_mod
except ImportError:
    import features as feat_mod


def _surv(rows):
    return pd.DataFrame(rows)


def _donors(rows):
    return pd.DataFrame(rows)


def test_proportional_split_no_cap():
    surv = _surv([
        {"player_id": "W2", "position": "WR", "_tpg": 6.0},
        {"player_id": "W3", "position": "WR", "_tpg": 3.0},
    ])
    don = _donors([{"player_id": "W1", "position": "WR", "_tpg": 9.0}])
    got = feat_mod._phase1_redistribute_tpg(surv, don, caps=None)
    # pool = 0.7*9 = 6.3 split 6:3 -> +4.2 / +2.1
    assert np.isclose(got.iloc[0], 10.2)
    assert np.isclose(got.iloc[1], 5.1)
    # vanished fraction: team total rises by pool only, not by full 9
    assert np.isclose(got.sum() - surv["_tpg"].sum(), 6.3)


def test_qb_full_conservation():
    surv = _surv([{"player_id": "QB2", "position": "QB", "_tpg": 20.0}])
    don = _donors([{"player_id": "QB1", "position": "QB", "_tpg": 30.0}])
    got = feat_mod._phase1_redistribute_tpg(surv, don, caps=None)
    assert np.isclose(got.iloc[0], 50.0)


def test_cap_waterfall_and_vanish():
    surv = _surv([
        {"player_id": "W2", "position": "WR", "_tpg": 8.0},
        {"player_id": "W3", "position": "WR", "_tpg": 7.0},
    ])
    don = _donors([{"player_id": "W1", "position": "WR", "_tpg": 20.0}])
    # pool = 0.7*20 = 14; proportional +6.53/+7.47 would push both over
    # cap 9, so waterfall fills both to 9 and the rest vanishes.
    got = feat_mod._phase1_redistribute_tpg(surv, don, caps={"WR": 9.0})
    assert np.isclose(got.iloc[0], 9.0)
    assert np.isclose(got.iloc[1], 9.0)
    assert got.sum() - surv["_tpg"].sum() < 14.0  # capped: part vanished


def test_no_same_position_survivor_vanishes():
    surv = _surv([{"player_id": "T1", "position": "TE", "_tpg": 5.0}])
    don = _donors([{"player_id": "W1", "position": "WR", "_tpg": 9.0}])
    got = feat_mod._phase1_redistribute_tpg(surv, don, caps=None)
    assert np.isclose(got.iloc[0], 5.0)  # TE untouched by WR absence (phase 2)


def test_zero_missing_is_noop():
    surv = _surv([{"player_id": "W2", "position": "WR", "_tpg": 6.0}])
    don = _donors([])
    got = feat_mod._phase1_redistribute_tpg(surv, don, caps={"WR": 9.0})
    assert np.isclose(got.iloc[0], 6.0)


def _wired_frames():
    hist = pd.DataFrame([
        {"game_id": "H1", "team": "HOME", "player_id": "W1",
         "position": "WR", "gameday": pd.Timestamp("2024-09-01"),
         "kickoff_utc": pd.Timestamp("2024-09-01 17:00Z"),
         "epa": 2.7, "opp": 9.0, "_num": 2.7, "_den": 9.0,
         "_n_games": 1.0},
        {"game_id": "H1", "team": "HOME", "player_id": "W2",
         "position": "WR", "gameday": pd.Timestamp("2024-09-01"),
         "kickoff_utc": pd.Timestamp("2024-09-01 17:00Z"),
         "epa": 0.6, "opp": 6.0, "_num": 0.6, "_den": 6.0,
         "_n_games": 1.0},
        {"game_id": "H1", "team": "HOME", "player_id": "W3",
         "position": "WR", "gameday": pd.Timestamp("2024-09-01"),
         "kickoff_utc": pd.Timestamp("2024-09-01 17:00Z"),
         "epa": 0.15, "opp": 3.0, "_num": 0.15, "_den": 3.0,
         "_n_games": 1.0},
        {"game_id": "H1", "team": "AWAY", "player_id": "W9",
         "position": "WR", "gameday": pd.Timestamp("2024-09-01"),
         "kickoff_utc": pd.Timestamp("2024-09-01 17:00Z"),
         "epa": 1.0, "opp": 5.0, "_num": 1.0, "_den": 5.0,
         "_n_games": 1.0},
    ])
    games = pd.DataFrame([{"game_id": "T1", "gameday": "2024-09-08",
                           "gametime": "13:00", "home_team": "HOME",
                           "away_team": "AWAY"}])
    inj = pd.DataFrame([{"game_id": "T1", "team": "HOME",
                         "player_id": "W1", "availability_weight": 0.0,
                         "published": "2024-09-08T15:00:00Z"}])
    return hist, games, inj


def _wr_epa(agg):
    return float(agg[(agg["game_id"] == "T1") & (agg["team"] == "HOME")
                     & (agg["position"] == "WR")]["epa_q"].iloc[0])


def test_wired_flag_off_is_baseline():
    hist, games, inj = _wired_frames()
    agg = feat_mod.epa_quality_team_agg(hist, games, inj)
    assert np.isfinite(_wr_epa(agg))


def test_wired_flag_on_lifts_star_out():
    hist, games, inj = _wired_frames()
    base = _wr_epa(feat_mod.epa_quality_team_agg(hist, games, inj))
    ph1 = _wr_epa(feat_mod.epa_quality_team_agg(
        hist, games, inj, redistribute_phase1=True,
        phase1_caps={"WR": 9.0, "QB": 35.0, "TE": 7.0, "RB": 20.0}))
    assert ph1 > base
    # full star volume would overshoot: partial conservation lands between
    no_out = _wr_epa(feat_mod.epa_quality_team_agg(hist, games, None))
    assert base < ph1 < no_out


def test_caps_estimator_is_p95():
    hist, _, _ = _wired_frames()
    caps = feat_mod._phase1_tpg_caps(hist)
    assert np.isclose(caps["WR"],
                      float(np.quantile([9.0, 6.0, 3.0, 5.0], 0.95)))


def test_phase2_wr_donor_spills_to_te():
    surv = _surv([
        {"player_id": "W2", "position": "WR", "_tpg": 6.0},
        {"player_id": "T1", "position": "TE", "_tpg": 4.0},
        {"player_id": "R1", "position": "RB", "_tpg": 10.0},
    ])
    don = _donors([{"player_id": "W1", "position": "WR", "_tpg": 10.0}])
    got = feat_mod._phase2_redistribute_tpg(surv, don, caps=None)
    # matrix WR row 0.61/0.22/0.08 of 10 -> +6.1/+2.2/+0.8, single
    # survivor per position takes it all (no caps)
    assert np.isclose(got.loc[0], 12.1)
    assert np.isclose(got.loc[1], 6.2)
    assert np.isclose(got.loc[2], 10.8)


def test_phase2_rb_donor_splits_rb_wr():
    surv = _surv([
        {"player_id": "W2", "position": "WR", "_tpg": 5.0},
        {"player_id": "R2", "position": "RB", "_tpg": 8.0},
    ])
    don = _donors([{"player_id": "R1", "position": "RB", "_tpg": 12.0}])
    got = feat_mod._phase2_redistribute_tpg(surv, don, caps=None)
    # RB row: WR 0.38*12=4.56, RB 0.29*12=3.48, TE share vanishes (no TE)
    assert np.isclose(got.loc[0], 9.56)
    assert np.isclose(got.loc[1], 11.48)


def test_phase2_qb_falls_back_to_full_takeover():
    surv = _surv([{"player_id": "QB2", "position": "QB", "_tpg": 20.0}])
    don = _donors([{"player_id": "QB1", "position": "QB", "_tpg": 30.0}])
    got = feat_mod._phase2_redistribute_tpg(surv, don, caps=None)
    assert np.isclose(got.iloc[0], 50.0)


def test_phase2_vanish_is_one_minus_row():
    surv = _surv([{"player_id": "W2", "position": "WR", "_tpg": 6.0}])
    don = _donors([{"player_id": "W1", "position": "WR", "_tpg": 10.0}])
    got = feat_mod._phase2_redistribute_tpg(surv, don, caps=None)
    # WR row sums to 0.91 -> only 6.1 of 10 reappears (TE/RB shares vanish)
    assert np.isclose(got.iloc[0], 12.1)


def test_phase2_wired_lifts_te_on_wr_out():
    hist, games, inj = _wired_frames()
    te_hist = pd.DataFrame([{
        "game_id": "H1", "team": "HOME", "player_id": "T1",
        "position": "TE", "gameday": pd.Timestamp("2024-09-01"),
        "kickoff_utc": pd.Timestamp("2024-09-01 17:00Z"),
        "epa": 1.0, "opp": 5.0, "_num": 1.0, "_den": 5.0, "_n_games": 1.0}])
    hist = pd.concat([hist, te_hist], ignore_index=True)
    base = feat_mod.epa_quality_team_agg(hist, games, inj)
    ph2 = feat_mod.epa_quality_team_agg(
        hist, games, inj, redistribute_phase1=True,
        phase1_caps={"WR": 9.0, "TE": 8.0, "RB": 22.0, "QB": 45.0},
        spill_matrix=feat_mod.PHASE2_SPILL_ESTIMATE)

    def _pos(agg, pos):
        return float(agg[(agg["game_id"] == "T1") & (agg["team"] == "HOME")
                         & (agg["position"] == pos)]["epa_q"].iloc[0])
    # phase-1 would leave TE untouched; phase-2 spills WR volume into TE
    assert _pos(ph2, "TE") > _pos(base, "TE")
    assert _pos(ph2, "WR") > _pos(base, "WR")


def test_default_caps_are_strictly_prior_for_each_target():
    hist, games, inj = _wired_frames()
    later = hist.iloc[[0]].copy()
    later["game_id"] = "H2"
    later["gameday"] = pd.Timestamp("2024-09-15")
    later["kickoff_utc"] = pd.Timestamp("2024-09-15 17:00Z")
    later["_den"] = 0.1
    later["_n_games"] = 1.0
    later["opp"] = 0.1
    later = pd.concat([
        later.assign(player_id=f"future_{i}") for i in range(100)
    ], ignore_index=True)
    combined = pd.concat([hist, later], ignore_index=True)
    next_game = games.assign(game_id="T2", gameday="2024-09-22")
    targets = pd.concat([games, next_game], ignore_index=True)
    reports = pd.concat([
        inj, inj.assign(game_id="T2", published="2024-09-22T15:00:00Z")
    ], ignore_index=True)
    multi = feat_mod.epa_quality_team_agg(
        combined, targets, reports, redistribute_phase1=True,
        phase1_alpha=feat_mod.PHASE1_ALPHA_ADOPTED)
    for gid, day in (("T1", "2024-09-08"), ("T2", "2024-09-22")):
        prior = combined[combined.gameday < pd.Timestamp(day)]
        expected = feat_mod.epa_quality_team_agg(
            prior, targets[targets.game_id == gid], reports,
            redistribute_phase1=True,
            phase1_caps=feat_mod._phase1_tpg_caps(prior),
            phase1_alpha=feat_mod.PHASE1_ALPHA_ADOPTED)
        pd.testing.assert_frame_equal(
            multi[multi.game_id == gid].reset_index(drop=True),
            expected.reset_index(drop=True))


def test_default_caps_ignore_target_day_history():
    hist, games, inj = _wired_frames()
    same_day = pd.concat([
        hist.iloc[[0]].assign(
            player_id=f"same_day_{i}", game_id="SAME_DAY",
            gameday=pd.Timestamp("2024-09-08"),
            kickoff_utc=pd.Timestamp("2024-09-08 12:00Z"),
            _den=0.1, _n_games=1.0, opp=0.1)
        for i in range(100)
    ], ignore_index=True)
    kwargs = dict(redistribute_phase1=True,
                  phase1_alpha=feat_mod.PHASE1_ALPHA_ADOPTED)
    expected = feat_mod.epa_quality_team_agg(hist, games, inj, **kwargs)
    actual = feat_mod.epa_quality_team_agg(
        pd.concat([hist, same_day], ignore_index=True), games, inj, **kwargs)
    pd.testing.assert_frame_equal(actual, expected)


def test_explicit_caps_override_estimation():
    hist, games, inj = _wired_frames()
    actual = feat_mod.epa_quality_team_agg(
        hist, games, inj, redistribute_phase1=True,
        phase1_caps={"WR": 6.0},
        phase1_alpha=feat_mod.PHASE1_ALPHA_ADOPTED)
    uncapped = feat_mod.epa_quality_team_agg(
        hist, games, inj, redistribute_phase1=True, phase1_caps={},
        phase1_alpha=feat_mod.PHASE1_ALPHA_ADOPTED)
    assert _wr_epa(actual) < _wr_epa(uncapped)


def test_rate_frozen_call_pattern():
    # Helper returns volume only; EPA math stays rate x adjusted volume.
    surv = _surv([
        {"player_id": "W2", "position": "WR", "_tpg": 6.0, "epa_q": 0.30},
        {"player_id": "W3", "position": "WR", "_tpg": 3.0, "epa_q": 0.10},
    ])
    don = _donors([{"player_id": "W1", "position": "WR", "_tpg": 9.0}])
    adj = feat_mod._phase1_redistribute_tpg(surv, don, caps=None)
    epa = (adj.to_numpy() * surv["epa_q"].to_numpy()).sum()
    assert np.isclose(epa, 10.2 * 0.30 + 5.1 * 0.10)
    # donor rate never enters: same volume from a star or a scrub
    # redistributes identically (efficiency lives in survivors' epa_q).
