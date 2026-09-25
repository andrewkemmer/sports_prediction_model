"""Focused offline contract tests for the standalone NBA backend.

Run explicitly with ``python -m pytest nba-backend/backend/test_nba_contract.py``.
The fixtures are synthetic and never read or write a production warehouse.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
REPO_ROOT = BACKEND.parent.parent
sys.path.insert(0, str(BACKEND))

import config  # noqa: E402
import distributions as dist_mod  # noqa: E402
import features as feat_mod  # noqa: E402
import folds as folds_mod  # noqa: E402
import ingestion as ing  # noqa: E402
import master_pipeline as pipeline_mod  # noqa: E402
import moneyline as ml_mod  # noqa: E402
import player_enrichment as players_mod  # noqa: E402
import retention_policy  # noqa: E402
import serving  # noqa: E402

TEAMS = list(config.NBA_TEAM_ID)


def _games(n_days: int = 42, games_per_day: int = 6, start: str = "2024-10-01") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    base = pd.Timestamp(start)
    for day in range(n_days):
        date = base + pd.Timedelta(days=day)
        for i in range(games_per_day):
            home, away = TEAMS[(day * games_per_day + i) % len(TEAMS)], TEAMS[(day * games_per_day + i + 7) % len(TEAMS)]
            hs = int(rng.poisson(112))
            aw = int(rng.poisson(108))
            rows.append({
                "game_id": f"{date:%Y%m%d}_{i}", "season": 2024,
                "gameday": date, "game_type": config.GAME_TYPE_REG,
                "home_team": home, "away_team": away,
                "home_score": hs, "away_score": aw,
            })
    return pd.DataFrame(rows)


def test_config_is_pinned_and_standalone() -> None:
    assert config.NBA_DATASET_REF == "wyattowalsh/basketball"
    assert config.NBA_DATASET_VERSION == "238"
    assert config.RANDOM_SEED == config.NUMPY_SEED == 42
    assert config.BLEND_SPACE == "logit"
    assert config.ENSEMBLE_MEMBERS == ["xgboost", "lightgbm", "elasticnet"]
    assert config.WARMUP_DAYS == 30
    assert config.RETRAIN_CADENCE_DAYS == 7
    assert config.MIN_VAL_FOLD_GAMES == 40
    assert config.NBA_TEAM_ID == {abbr: i for i, abbr in enumerate(TEAMS)}


def test_source_discovery_handles_nested_kaggle_exports(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    dataset = input_root / "generated-basketball-slug"
    parquet = dataset / "parquet" / "season_year=2024"
    parquet.mkdir(parents=True)
    (parquet / "dim_game.parquet").write_bytes(b"fixture")
    (parquet / "fact_game_result.parquet").write_bytes(b"fixture")
    assert ing.discover_warehouse([input_root]) == dataset

    partitioned = tmp_path / "partitioned-input" / "basketball"
    table_dir = partitioned / "parquet" / "dim_game" / "season_year=2024"
    table_dir.mkdir(parents=True)
    (table_dir / "part-000.parquet").write_bytes(b"fixture")
    assert ing.discover_warehouse([partitioned.parent]) == partitioned

    sql_dataset = tmp_path / "sql-input" / "basketball"
    sql_dataset.mkdir(parents=True)
    sql_file = sql_dataset / "nba.duckdb"
    sql_file.write_bytes(b"fixture")
    assert ing.discover_warehouse([sql_dataset.parent]) == sql_file


def test_missing_source_error_rejects_literal_none(tmp_path: Path) -> None:
    with patch.object(ing, "_source_root", return_value=None):
        with pytest.raises(FileNotFoundError, match="Do not pass None"):
            ing.load_dataset("None", use_cache=False)


def test_folds_are_observed_date_expanding_and_prior_only() -> None:
    games = _games(n_days=46)
    games = games[games.gameday != pd.Timestamp("2024-10-15")]
    folds = folds_mod.make_folds(games)
    assert folds
    unique_dates = sorted(pd.to_datetime(games.gameday).unique())
    assert folds[0].val_start == pd.Timestamp(unique_dates[config.WARMUP_DAYS])
    assert folds[0].val_end == pd.Timestamp(unique_dates[config.WARMUP_DAYS + 6])
    for prior, current in zip(folds, folds[1:]):
        assert len(current.train_idx) >= len(prior.train_idx)
    for fold in folds:
        assert games.loc[fold.train_idx, "gameday"].max() < fold.val_start
        assert games.loc[fold.val_idx, "gameday"].min() >= fold.val_start
    assert folds[-1].is_partial_tail


def test_oof_metadata_merge_preserves_game_identity() -> None:
    oof = pd.DataFrame({
        "game_id": ["g1"], "home_win": [1.0], "gameday": [pd.Timestamp("2024-10-01")],
        "p_ensemble": [0.6],
    })
    games = pd.DataFrame({
        "game_id": ["g1"], "home_win": [0.0], "gameday": [pd.Timestamp("2024-10-02")],
        "season": [2024], "home_team": ["ATL"],
    })
    merged = pipeline_mod._merge_oof_metadata(oof, games)
    assert merged.loc[0, "game_id"] == "g1"
    assert merged.loc[0, "home_team"] == "ATL"
    assert merged.loc[0, "home_win"] == 0.0


def test_pending_slate_gate_validates_marketized_fair_lines() -> None:
    slate = pd.DataFrame({
        "game_id": ["g1"], "home_team": ["ATL"], "away_team": ["BOS"],
        "home_win_prob_model": [0.61], "mu_h": [112.0], "mu_a": [108.0],
    })
    markets = pd.DataFrame({
        "game_id": ["g1"], "fair_spread": [2.0], "fair_total": [220.0],
    })
    pipeline_mod._validate_slate_contract(slate, markets)
    with pytest.raises(RuntimeError, match="non-finite fair_total"):
        pipeline_mod._validate_slate_contract(
            slate, markets.assign(fair_total=[np.nan]))


def test_features_are_invariant_to_future_results() -> None:
    games = _games(n_days=36)
    baseline = feat_mod.build_game_features(games.iloc[:180].copy())
    poisoned = games.copy()
    mask = pd.to_datetime(poisoned.gameday) > pd.Timestamp("2024-11-15")
    poisoned.loc[mask, "home_score"] = 999
    poisoned.loc[mask, "away_score"] = 1
    after = feat_mod.build_game_features(poisoned)
    cols = ["elo_diff", "win_pct_diff", "rest_days_diff", "nba_points_for_pg_roll_diff"]
    before = baseline.sort_values("game_id").reset_index(drop=True)
    after = after[after.game_id.isin(before.game_id)].sort_values("game_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(before[cols], after[cols], check_exact=False)


def _distribution_frame() -> pd.DataFrame:
    return dist_mod.simulate_distributions(
        np.array([115.0, 108.0]), np.array([109.0, 112.0]),
        alpha_home=0.04, alpha_away=0.03, n_draws=500, seed=42,
    )


def test_distribution_grid_coherence_and_push_namespace() -> None:
    frame = _distribution_frame()
    for line in config.SPREAD_GRID:
        h = frame[dist_mod._grid_key("p_home_cover", line)].to_numpy(float)
        p = frame[dist_mod._grid_key("p_push", line)].to_numpy(float)
        a = frame[dist_mod._grid_key("p_away_cover", line)].to_numpy(float)
        np.testing.assert_allclose(h + p + a, 1.0, atol=1e-12)
    for line in config.HALF_STOP_LINES:
        assert dist_mod._grid_key("p_push", line) in frame
        np.testing.assert_allclose(frame[dist_mod._grid_key("p_push", line)], 0.0)
    for line in config.TOTAL_GRID:
        o = frame[dist_mod._grid_key("p_over", line)].to_numpy(float)
        p = frame[dist_mod._grid_key("p_push_total", line)].to_numpy(float)
        u = frame[dist_mod._grid_key("p_under", line)].to_numpy(float)
        np.testing.assert_allclose(o + p + u, 1.0, atol=1e-12)
    np.testing.assert_allclose(
        frame.p_home_win_derived + frame.p_away_win_derived + frame.p_tie,
        1.0, atol=1e-12,
    )
    assert (frame.p_tie >= 0).all()


def test_distribution_prequential_poisoning_is_prior_only() -> None:
    rng = np.random.default_rng(3)
    n = 360
    p = np.clip(0.5 + rng.normal(0, 0.15, n), 0.02, 0.98)
    y = rng.integers(0, 2, n).astype(float)
    folds = np.repeat(np.arange(6), 60)
    calibrated, final = dist_mod._prequential(p, y, folds)
    poisoned = y.copy()
    poisoned[folds == 5] = 1 - poisoned[folds == 5]
    poisoned_out, _ = dist_mod._prequential(p, poisoned, folds)
    np.testing.assert_array_equal(calibrated[folds < 5], poisoned_out[folds < 5])
    assert final is not None


def test_moneyline_prequential_fold_table_records_used_weights() -> None:
    games = feat_mod.build_game_features(_games(n_days=40))
    folds = folds_mod.make_folds(games)
    calls: list[dict] = []

    class Model:
        def __init__(self, name: str): self.name = name
        def fit(self, X, y, **kwargs): return self
        def predict_proba(self, X):
            values = {"xgboost": 0.8, "lightgbm": 0.6, "elasticnet": 0.55}[self.name]
            p = np.full(len(X), values)
            return np.column_stack([1 - p, p])

    def fake_member(name, *args, **kwargs): return Model(name)
    def fake_weights(members, y):
        calls.append(members)
        return {"xgboost": 1.0, "lightgbm": 0.0, "elasticnet": 0.0}

    with patch.object(ml_mod, "_make_member", side_effect=fake_member), \
         patch.object(ml_mod, "compute_adaptive_weights", side_effect=fake_weights):
        result = ml_mod.walk_forward_oof(games, fold_list=folds, progress_every=0)
    table = result["fold_table"]
    assert table.iloc[0].weights == config.ENSEMBLE_WEIGHTS
    assert table.iloc[1].weights == {"xgboost": 1.0, "lightgbm": 0.0, "elasticnet": 0.0}
    assert calls


def test_moneyline_missing_members_do_not_poison_healthy_members() -> None:
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    members = {
        "xgboost": [0.1, np.nan, 0.2, np.nan, 0.3, np.nan, 0.4, np.nan],
        "lightgbm": [0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5],
        "elasticnet": [np.nan] * 8,
    }
    weights = ml_mod.compute_adaptive_weights(members, y)
    # The all-NaN member is omitted rather than diluting the healthy one.
    assert weights["xgboost"] > 0
    assert weights["lightgbm"] >= 0
    assert "elasticnet" not in weights
    assert abs(sum(weights.values()) - 1) < 1e-9

    # A partially available member is skipped only on its missing rows; the
    # remaining member is renormalized per row and never becomes NaN.
    blended = ml_mod._logit_blend_matrix(
        np.array([[0.8, 0.2], [np.nan, 0.8], [0.1, np.nan]]),
        np.array([0.5, 0.5]),
    )
    np.testing.assert_allclose(blended, [0.5, 0.8, 0.1], atol=1e-12)


def test_player_leader_uses_prior_five_and_fails_closed() -> None:
    games = _games(n_days=8, games_per_day=1)
    games = games[games.home_team == "ATL"].copy()
    if len(games) < 6:
        games = _games(n_days=12, games_per_day=1)
        games["home_team"] = "ATL"
        games["away_team"] = "BOS"
    game_ids = games.game_id.astype(str).tolist()[:6]
    target = games.iloc[5]
    stats = []
    for i, gid in enumerate(game_ids[:5]):
        stats.append({"game_id": gid, "gameday": games.iloc[i].gameday,
                      "team": "ATL", "player_id": "qualified", "player_name": "Qualified",
                      "points": 30, "assists": 7, "minutes": 36})
        stats.append({"game_id": gid, "gameday": games.iloc[i].gameday,
                      "team": "ATL", "player_id": "short", "player_name": "Short",
                      "points": 50, "assists": 2, "minutes": 4})
    result = players_mod._player_for_team(pd.DataFrame(stats), "ATL", target.gameday, games)
    assert result["name"] == "Qualified"
    assert result["games"] == 5
    unqualified = players_mod._player_for_team(
        pd.DataFrame([r for r in stats if r["player_id"] == "short"]),
        "ATL", target.gameday, games)
    assert unqualified == {}


def test_ingestion_canonical_aliases_join_reordered_results(tmp_path: Path) -> None:
    source = tmp_path / "warehouse"
    source.mkdir()
    (source / "dim_team.csv").write_text("id,team_abbreviation,full_name\n" + "\n".join(
        f"{i},{abbr},{abbr} Club" for i, abbr in enumerate(TEAMS)) + "\n")
    game_rows = []
    for i, abbr in enumerate(TEAMS):
        other = TEAMS[(i + 1) % len(TEAMS)]
        game_rows.append({"game_id": f"002240{i:04d}", "game_date": "2024-10-01",
                          "season_year": "2024", "season_type": "Regular Season",
                          "home_team_id": i, "visitor_team_id": (i + 1) % len(TEAMS)})
    pd.DataFrame(game_rows).to_csv(source / "dim_game.csv", index=False)
    result_rows = []
    for row in reversed(game_rows):
        i = int(row["home_team_id"])
        result_rows.append({"game_id": row["game_id"], "wl_home": "W",
                            "pts_home": 110 + i, "pts_away": 100 + i})
    pd.DataFrame(result_rows).to_csv(source / "fact_game_result.csv", index=False)
    team_rows = []
    for row in game_rows:
        for team_id in (row["home_team_id"], row["visitor_team_id"]):
            team_rows.append({"game_id": row["game_id"], "team_id": team_id,
                              "pts": 100 + int(team_id), "ast": 25, "reb": 45,
                              "tov": 12, "oreb": 10, "dreb": 35, "stl": 7, "blk": 4,
                              "fgm": 40, "fga": 88, "fg3m": 12, "fg3a": 35,
                              "ftm": 16, "fta": 20})
    pd.DataFrame(team_rows).to_csv(source / "fact_box_score_traditional_team.csv", index=False)
    player_rows = []
    for n, row in enumerate(game_rows):
        player_rows.append({"game_id": row["game_id"], "player_id": 100 + n,
                            "team_id": row["home_team_id"], "pts": 30,
                            "ast": 5, "min": "36:00"})
    pd.DataFrame(player_rows).to_csv(source / "fact_box_score_traditional_player.csv", index=False)
    (source / "dim_player.csv").write_text("player_id,first_name,family_name\n100,Ada,Lovelace\n")
    wh = ing.load_dataset(source, use_cache=False)
    assert len(wh.games) == 30
    assert set(wh.games.home_team) | set(wh.games.away_team) == set(TEAMS)
    assert ing._game_id("0022400000") == "0022400000"
    first = wh.games[wh.games.game_id == "0022400000"].iloc[0]
    assert first.home_score == 110
    assert first.away_score == 100
    assert first.game_type == config.GAME_TYPE_REG
    assert wh.team_stats.team.nunique() == 30
    assert wh.player_stats.player_name.iloc[0] == "Ada Lovelace"
    assert wh.manifest["dataset_version"] == "238"
    assert wh.manifest["schemas"]["games"]["columns"]


def test_ingestion_fails_loudly_on_missing_team_coverage(tmp_path: Path) -> None:
    source = tmp_path / "warehouse"
    source.mkdir()
    pd.DataFrame([{"id": 0, "team_abbreviation": "ATL"}]).to_csv(source / "dim_team.csv", index=False)
    pd.DataFrame([{"game_id": "g", "game_date": "2024-10-01", "season_year": 2024,
                   "home_team_id": 0, "visitor_team_id": 0, "home_score": 1, "away_score": 0}]).to_csv(source / "dim_game.csv", index=False)
    try:
        ing.load_dataset(source, use_cache=False)
    except RuntimeError as exc:
        assert "current-team coverage" in str(exc)
    else:
        raise AssertionError("missing team coverage was silently accepted")


def test_serving_grid_is_unique_strict_and_frozen(tmp_path: Path) -> None:
    assert len(serving.markets_columns()) == len(set(serving.markets_columns()))
    assert "p_push_total_220" in serving.markets_columns()
    assert "p_push_20" in serving.markets_columns()
    path = tmp_path / "cards.csv"
    first = pd.DataFrame([{"game_id": "g1", "gameday": "2024-10-01", "p_home_win": .6}])
    second = pd.DataFrame([{"game_id": "g1", "gameday": "2024-10-01", "p_home_win": .2},
                           {"game_id": "g2", "gameday": "2024-10-02", "p_home_win": .7}])
    serving.write_production_cards_history(path, first)
    serving.write_production_cards_history(path, second)
    frozen = pd.read_csv(path)
    assert frozen[frozen.game_id == "g1"].p_home_win.iloc[0] == .6
    record = serving.dump_json(tmp_path / "strict.json", {"x": np.nan, "y": np.float64(2)})
    assert json.loads((tmp_path / "strict.json").read_text())["y"] == 2
    assert record["x"] is None


def test_retention_keeps_nba_masters_and_dated_families() -> None:
    assert retention_policy.classify_artifact("nba_production_cards_history.csv", set(), set(), set(), set()) == "protected"
    assert retention_policy.classify_artifact("nba_moneyline_v1_20240101.json", set(), {"20240101"}, set(), set()) == "current"
    assert retention_policy.classify_artifact("nba_moneyline_v1_20200101.json", set(), set(), set(), set()) == "stale"
