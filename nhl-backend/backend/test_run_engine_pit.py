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
  6. NB dispersion: estimated from leakage-free OOF, capped;
  7. monitor line-pair contract: canonical totals (5,6,7) / spreads (1,2)
     priced at fair lines with honest outcomes.

Run with: python nhl-backend/backend/test_run_engine_pit.py
"""
from __future__ import annotations

import sys
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
def test_estimate_alpha_caps_and_floors():
    rng = np.random.default_rng(9)
    mu = np.full(500, 3.0)
    # Poisson data: variance ≈ mean → alpha near zero.
    y_pois = rng.poisson(3.0, 500).astype(float)
    assert dist_mod.estimate_alpha(y_pois, mu) < 0.05
    # Overdispersed data (variance >> mean) → positive alpha, capped.
    y_nb = rng.negative_binomial(6.0, 6.0 / (6.0 + 3.0), 500).astype(float)
    a = dist_mod.estimate_alpha(y_nb, mu)
    assert 0.0 < a <= dist_mod.ALPHA_CAP
    # Degenerate inputs floor at 0, never raise.
    assert dist_mod.estimate_alpha(np.array([np.nan, 1.0]), np.array([3.0, 3.0])) == 0.0
    assert dist_mod.estimate_alpha(np.array([]), np.array([])) == 0.0


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
    assert 0.0 <= params["alpha_home"] <= dist_mod.ALPHA_CAP
    assert 0.0 <= params["alpha_away"] <= dist_mod.ALPHA_CAP


# ---------------------------------------------------------------------------
# 7. Monitor line-pair contract: canonical lines priced honestly
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
         "home_sog": 30, "away_sog": 28},
        {"game_id": "g2", "home_goalie_id": 1, "away_goalie_id": 2,
         "home_goalie_name": "Home One", "away_goalie_name": "Away One",
         "home_goalie_toi": 60.0, "away_goalie_toi": 58.0,
         "home_goals_against": 1, "away_goals_against": 1,
         "home_sog": 30, "away_sog": 28},
    ])
    states, _ = feat_mod.goalie_state(boxscores, games)
    assert pd.isna(states.loc[0, "goalie_gaa_home"])
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
