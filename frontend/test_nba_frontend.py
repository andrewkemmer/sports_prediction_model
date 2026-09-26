"""Focused NBA frontend contract tests.

These tests use synthetic, temporary artifacts only.  The fixture backs up any
pre-existing files with the same names and restores them byte-for-byte after
each test; it never stages production data or contacts GitHub.
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

FRONTEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = FRONTEND_DIR.parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import sports_config  # noqa: E402
import utils  # noqa: E402
import nba_market_diagnostics as nba_md  # noqa: E402
import nba_slate_view as nba_sv  # noqa: E402
import nba_todays_page  # noqa: E402

NBA_DD = REPO_ROOT / "nba-backend" / "data_delivery"
DATE = "20990101"
DATE_ISO = "2099-01-01"
GAME_ID = "0029900001"
STAGED_NAMES = (
    f"nba_moneyline_v1_{DATE}.json",
    f"nba_player_leader_matchup_{DATE}.json",
    f"nba_calibration_{DATE}.json",
    f"nba_predictions_history_{DATE}.csv",
    f"nba_power_rankings_{DATE}.csv",
    f"nba_model_monitor_{DATE}.json",
    f"nba_run_engine_markets_{DATE}.csv",
    f"nba_run_engine_monitor_{DATE}.json",
)


def _all_text(app: AppTest) -> str:
    chunks: list[str] = []
    for attr in (
        "markdown", "info", "warning", "caption", "success", "error",
        "title", "header", "subheader", "text",
    ):
        for element in getattr(app, attr, []):
            try:
                chunks.append(str(element.value))
            except Exception:
                pass
    return "\n".join(chunks)


def _assert_clean(app: AppTest) -> None:
    if app.exception:
        details = "\n".join(str(getattr(item, "value", item)) for item in app.exception)
        raise AssertionError(f"Streamlit page raised:\n{details}")


def _run_page(filename: str) -> AppTest:
    app = AppTest.from_file(str(FRONTEND_DIR / filename), default_timeout=60)
    app.session_state["sport"] = "nba"
    app.session_state["gh_owner"] = ""
    app.session_state["gh_repo"] = ""
    app.session_state["gh_branch"] = "main"
    app.run()
    _assert_clean(app)
    return app


def _moneyline_record() -> dict:
    return {
        "created_utc": "2099-01-01T00:00:00Z",
        "slate_date": DATE_ISO,
        "n_games": 1,
        "games": [{
            "game_id": GAME_ID,
            "game_date": DATE_ISO,
            "start_time_utc": "2099-01-01T23:30:00Z",
            "home_team": "ATL",
            "away_team": "BOS",
            "home_team_name": "Atlanta Hawks",
            "away_team_name": "Boston Celtics",
            "home_record": "20-10",
            "away_record": "18-12",
            "venue": "State Farm Arena",
            "game_status": "Scheduled",
            "home_win_prob_model": 0.64,
            "away_win_prob_model": 0.36,
            "model_pick": "ATL",
            "p_home_name": "Qualified Star",
            "p_home_ppg": 31.2,
            "p_home_apg": 7.1,
            "p_home_games": 5,
            "p_away_name": None,
            "p_away_ppg": None,
            "p_away_apg": None,
            "p_away_games": 0,
        }],
    }


def _player_record() -> dict:
    return {
        "created_utc": "2099-01-01T00:00:00Z",
        "n_games": 1,
        "games": [{
            "game_id": GAME_ID,
            "gameday": DATE_ISO,
            "home_team": "ATL",
            "away_team": "BOS",
            "p_home_name": "Qualified Star",
            "p_home_ppg": 31.2,
            "p_home_apg": 7.1,
            "p_home_games": 5,
            "p_away_name": None,
            "p_away_ppg": None,
            "p_away_apg": None,
            "p_away_games": 0,
        }],
    }


def _calibration_record() -> dict:
    buckets = [
        {"bucket": "60%-68%", "mean_predicted": 0.64,
         "mean_actual": 0.63, "count": 2, "gap": 0.01},
        {"bucket": "70%-78%", "mean_predicted": 0.74,
         "mean_actual": 0.75, "count": 3, "gap": -0.01},
    ]
    calibrated = [
        {**row, "mean_predicted": min(0.99, row["mean_predicted"] + 0.01)}
        for row in buckets
    ]
    return {
        "date": DATE,
        "trained_at": "2099-01-01T00:00:00Z",
        "n_games": 5,
        "metrics": {
            "auc": 0.71, "brier": 0.21, "logloss": 0.61, "ece": 0.03,
            "brier_calibrated": 0.20, "logloss_calibrated": 0.60,
            "ece_calibrated": 0.02,
        },
        "calibration_buckets": buckets,
        "calibration": {
            "method": "platt",
            "params": {"a": 1.1, "b": 0.02, "n": 5},
            "metrics_raw": {"brier": 0.21, "logloss": 0.61, "ece": 0.03},
            "metrics_calibrated": {"brier": 0.20, "logloss": 0.60, "ece": 0.02},
            "calibration_buckets_calibrated": calibrated,
        },
        "daily": [],
    }


def _history_frame() -> pd.DataFrame:
    rows = []
    for i, (home, away, p, correct) in enumerate([
        ("ATL", "BOS", 0.64, True),
        ("BOS", "NYK", 0.72, True),
        ("LAL", "DEN", 0.76, False),
        ("DEN", "PHX", 0.81, True),
        ("MIA", "CHI", 0.68, True),
    ]):
        rows.append({
            "game_id": f"history-{i}",
            "game_date": DATE_ISO,
            "home_team": home,
            "away_team": away,
            "home_score": 112 if correct else 101,
            "away_score": 105 if correct else 110,
            "home_win": 1 if correct else 0,
            "home_win_prob_model": p,
            "home_win_prob_model_calibrated": min(0.95, p + 0.01),
            "away_win_prob_model": 1 - p,
            "model_pick": home,
            "actual_winner": home if correct else away,
            "correct": correct,
            "game_status": "Final",
        })
    return pd.DataFrame(rows)


def _markets_frame() -> pd.DataFrame:
    common = {
        "game_id": GAME_ID,
        "gameday": DATE_ISO,
        "season": 2099,
        "home_team": "ATL",
        "away_team": "BOS",
        "venue": "State Farm Arena",
        "mu_h": 112.0,
        "mu_a": 108.0,
        "mu_margin": 4.0,
        "mu_total": 220.0,
        "fair_spread": 2.0,
        "fair_total": 220.0,
        "p_tie": 0.01,
        "p_home_win_derived": 0.55,
        "p_away_win_derived": 0.44,
        "p_home_win": 0.55,
        "p_away_win": 0.44,
        "derived_ml": 0.55,
        "p_over_fair": 0.52,
    }
    for line, (home, push, away) in nba_sv.spread_columns().items():
        common[f"p_home_cover_{nba_sv._label(line)}"] = home
        common[f"p_push_{nba_sv._label(line)}"] = push
        common[f"p_away_cover_{nba_sv._label(line)}"] = away
    for line, (over, under, push) in nba_sv.total_columns().items():
        key = int(line) if float(line).is_integer() else line
        common[f"p_over_{key}"] = over
        common[f"p_under_{key}"] = under
        common[f"p_push_total_{key}"] = push

    oof = {
        **common,
        "kind": "oof",
        "decided": True,
        "home_score": 112,
        "away_score": 108,
        "total": 220,
        "margin": 4,
    }
    slate = {
        **common,
        "kind": "slate",
        "decided": False,
        "home_score": None,
        "away_score": None,
        "total": None,
        "margin": None,
    }
    return pd.DataFrame([oof, slate])


def _rankings_frame() -> pd.DataFrame:
    teams = [
        "BOS", "DEN", "OKC", "MIN", "LAL", "GSW", "MIL", "PHX", "IND",
        "MIA", "ATL", "NYK", "CLE", "ORL", "DAL", "BKN",
    ]
    rows = []
    for rank, team in enumerate(teams, start=1):
        wins = max(1, 30 - rank)
        losses = rank
        rows.append({
            "rank": rank,
            "team": team,
            "team_name": utils.NBA_TEAM_NAMES.get(team, team),
            "elo": 1600 - rank * 5,
            "wins": wins,
            "losses": losses,
            "record": f"{wins}-{losses}",
            "pct": wins / (wins + losses),
            "run_diff": rank,
            "point_diff": rank,
            "l10": "5-5",
            "home_pct": 0.6,
            "away_pct": 0.5,
        })
    return pd.DataFrame(rows)


def _monitor_record() -> dict:
    return {
        "last_retrained": DATE_ISO,
        "last_retrained_note": "Fresh model trained this run",
        "next_retrain": "2099-01-02",
        "next_retrain_note": "next expected run in 1 day(s)",
        "upset_note": "Five synthetic OOF games scored.",
        "feature_drift": [{
            "feature": "elo_diff", "current_mean": 12.0,
            "baseline_mean": 8.0, "psi": 0.2, "psi_adjusted": 0.18,
            "weight_pct": 45.0, "n_baseline": 100, "n_current": 20,
            "status": "WARN",
        }],
        "features_metadata": {
            "elo_diff": {
                "definition": "pre-game Elo gap",
                "tooltip": "Point-in-time home Elo minus away Elo.",
            }
        },
        "feature_coverage": [{
            "feature": "elo_diff", "window": "decided pool", "n_games": 5,
            "pct_measured": 100.0, "pct_nonnull": 100.0,
            "n_default_zero": 0, "status": "OK",
        }],
        "ensemble": [
            {"name": "xgboost", "weight": 0.5, "auc": 0.71,
             "brier": 0.21, "logloss": 0.61, "n_eval": 5},
            {"name": "lightgbm", "weight": 0.5, "auc": 0.70,
             "brier": 0.22, "logloss": 0.62, "n_eval": 5},
        ],
        "rolling_brier": [
            {"date": DATE_ISO, "brier": 0.21},
            {"date": "2099-01-02", "brier": 0.20},
        ],
        "rolling_brier_meta": {"window_days": 30, "min_games_per_day": 1},
        "brier_baseline": 0.23,
        "brier_baseline_label": "Constant home edge",
        "version_history": [{
            "version": DATE, "date": DATE_ISO,
            "weights": {"xgboost": 0.5, "lightgbm": 0.5, "elasticnet": 0.0},
            "auc": 0.71, "logloss": 0.61, "ece_calibrated": 0.02,
            "calibration": {"a": 1.1, "b": 0.02},
        }],
        "market_metrics": {"220": {"brier": 0.2, "n": 5}},
    }


@contextmanager
def _staged_nba_artifacts(monkeypatch: pytest.MonkeyPatch):
    """Stage synthetic files while preserving any pre-existing local files."""
    for key in ("GITHUB_OWNER", "GITHUB_REPO", "GITHUB_BRANCH"):
        monkeypatch.setenv(key, "")
    dd = NBA_DD
    original_exists = dd.exists()
    backups: dict[Path, bytes] = {}
    created: list[Path] = []
    dd.mkdir(parents=True, exist_ok=True)

    def write(name: str, data: bytes) -> None:
        path = dd / name
        if path.exists():
            backups[path] = path.read_bytes()
        path.write_bytes(data)
        created.append(path)

    write(f"nba_moneyline_v1_{DATE}.json", json.dumps(_moneyline_record()).encode())
    write(f"nba_player_leader_matchup_{DATE}.json", json.dumps(_player_record()).encode())
    write(f"nba_calibration_{DATE}.json", json.dumps(_calibration_record()).encode())
    write(f"nba_predictions_history_{DATE}.csv", _history_frame().to_csv(index=False).encode())
    write(f"nba_power_rankings_{DATE}.csv", _rankings_frame().to_csv(index=False).encode())
    write(f"nba_model_monitor_{DATE}.json", json.dumps(_monitor_record()).encode())
    write(f"nba_run_engine_markets_{DATE}.csv", _markets_frame().to_csv(index=False).encode())
    write(f"nba_run_engine_monitor_{DATE}.json", json.dumps(_monitor_record()).encode())
    st.cache_data.clear()
    try:
        yield dd
    finally:
        st.cache_data.clear()
        for path in reversed(created):
            try:
                if path in backups:
                    path.write_bytes(backups.pop(path))
                else:
                    path.unlink()
            except FileNotFoundError:
                pass
        if not original_exists:
            try:
                dd.rmdir()
            except OSError:
                pass


@pytest.fixture
def nba_artifacts(monkeypatch: pytest.MonkeyPatch):
    with _staged_nba_artifacts(monkeypatch):
        yield


def test_nba_registry_and_adapter_contracts() -> None:
    assert sports_config.active_page_url_paths("nba") == [
        "todays-games", "power-rankings", "calibration", "model-monitor", "markets"
    ]
    assert sports_config.data_delivery_dir("nba").name == "data_delivery"
    assert sports_config.resolve_sport("nba")["repo_subdir"] == "nba-backend"
    assert nba_md._label(-20) == "m20"
    assert nba_md._label(-0.5) == "m0_5"

    record = _moneyline_record()
    frame = utils.nba_moneyline_to_frame(record)
    assert list(frame.columns) == utils.NBA_CARD_COLUMNS
    assert frame.loc[0, "home_team"] == "ATL"
    assert frame.loc[0, "home_win_prob_model"] == 0.64
    assert frame.loc[0, "p_home_name"] == "Qualified Star"
    assert pd.isna(frame.loc[0, "p_away_name"])


def test_player_empty_state_renders_em_dash() -> None:
    html = nba_todays_page._player_matchup_html({
        "p_home_name": "Qualified Star", "p_home_ppg": 31.2, "p_home_apg": 7.1,
        "p_away_name": None, "p_away_ppg": None, "p_away_apg": None,
    })
    assert "Qualified Star" in html
    assert "PPG 31.2" in html
    assert "PPG —" in html and "APG —" in html


def test_nba_todays_games_app_dispatches_and_renders_player(nba_artifacts) -> None:
    app = _run_page("todays_games.py")
    text = _all_text(app)
    assert "Atlanta Hawks" in text and "Boston Celtics" in text
    assert "Qualified Star" in text
    assert "PPG —" in text
    assert "O/U" in text
    assert not [item for item in app.exception]


def test_nba_power_rankings_tab_renders_point_diff(nba_artifacts) -> None:
    app = _run_page("power_rankings.py")
    text = _all_text(app)
    assert "Power Rankings" in text
    assert "POINT DIFF" in text
    assert "BOS" in text
    assert "BKN" not in text  # row 16 is outside the top-15 view


def test_nba_calibration_tab_renders_kpis_and_curve(nba_artifacts) -> None:
    app = _run_page("model_calibration.py")
    text = _all_text(app)
    assert "Model Calibration Dashboard" in text
    assert "AUC-ROC" in text and "BRIER SCORE" in text
    assert "Calibration Curve" in text
    assert "Prediction History" in text
    assert len(app.get("vega_lite_chart")) >= 1


def test_nba_monitor_tab_uses_nba_config_and_renders_sections(nba_artifacts) -> None:
    app = _run_page("model_monitor.py")
    text = _all_text(app)
    assert "Model & Data Drift Monitor" in text
    assert "Feature Drift Analysis" in text
    assert "Feature Coverage" in text
    assert "Model Ensemble" in text
    assert "Rolling Brier Score" in text
    assert "Model Version History" in text
    assert "config unavailable" not in text
    assert "max depth 3" in text and "lr 0.1097" in text
    assert len(app.get("vega_lite_chart")) >= 1


def test_nba_markets_tab_dispatches_to_nba_page(nba_artifacts) -> None:
    app = _run_page("markets.py")
    text = _all_text(app)
    assert "Point Spread" in text
    assert "Diagnostics" in text
    assert "Totals Monitor" in text
    assert "NBA model diagnostics" in text


def test_all_nba_tabs_have_honest_missing_artifact_states(monkeypatch) -> None:
    # The repository currently has no production NBA delivery directory.  If
    # a developer has one, temporarily hide only the fixture names this suite
    # owns; this test still asserts the page-level empty states below.
    with _staged_nba_artifacts(monkeypatch):
        for name in STAGED_NAMES:
            path = NBA_DD / name
            if path.exists():
                path.unlink()
        st.cache_data.clear()
        expectations = {
            "todays_games.py": "No NBA per-game moneyline rows available.",
            "power_rankings.py": "No power rankings found",
            "model_calibration.py": "No calibration artifacts found",
            "model_monitor.py": "No model monitor artifacts found",
            "markets.py": "No usable NBA run-engine markets artifact",
        }
        for filename, needle in expectations.items():
            app = _run_page(filename)
            assert needle in _all_text(app), (filename, _all_text(app))


class TestCardStartTimeAndMatchup:
    """The two fields a today's-games card cannot be read without.

    Both were delivered as columns that were always null, and the adapter
    covered the gap by fabricating midnight UTC - which, because ``game_date``
    is an Eastern *board* date, rendered as 7:00 PM ET on the previous evening.
    Every unslotted game therefore showed a confident, wrong start.
    """

    @staticmethod
    def _frame(**overrides) -> pd.DataFrame:
        game = {"game_id": GAME_ID, "game_date": DATE_ISO,
                "home_team": "ATL", "away_team": "BOS",
                "home_win_prob_model": 0.64, "game_status": "Scheduled"}
        game.update(overrides)
        return utils.nba_moneyline_to_frame(
            {"created_utc": "2099-01-01T00:00:00Z", "slate_date": DATE_ISO,
             "n_games": 1, "games": [game]})

    def test_a_real_tipoff_is_carried_through(self):
        frame = self._frame(start_time_utc="2099-01-01T23:30:00Z")
        assert frame.start_time_utc.iloc[0] == "2099-01-01T23:30:00Z"

    def test_a_missing_tipoff_is_not_invented(self):
        """Midnight UTC on an Eastern board date is 7 PM the night before."""
        assert self._frame().start_time_utc.iloc[0] is None

    def test_a_missing_tipoff_renders_as_pregame_not_as_a_time(self):
        assert nba_todays_page._start_time_et(None) == ""
        assert nba_todays_page._start_time_et("") == ""

    def test_a_real_tipoff_renders_in_eastern(self):
        assert nba_todays_page._start_time_et(
            "2099-01-01T23:30:00Z") == "6:30 PM ET"

    def test_the_card_shows_pregame_when_the_start_is_unknown(self):
        card = nba_todays_page._nba_mirror_card_html(
            self._frame().iloc[0], None, "")
        assert "PREGAME" in card

    def test_an_unknown_start_is_not_counted_as_an_evening_game(self):
        assert utils._is_evening_start(None) is False
        assert utils._is_evening_start("") is False

    def test_both_sides_player_matchup_reaches_the_card(self):
        row = {"p_home_name": "Home Star", "p_home_ppg": 30.0, "p_home_apg": 8.0,
               "p_away_name": "Away Star", "p_away_ppg": 22.0, "p_away_apg": 5.0}
        html = nba_todays_page._player_matchup_html(row)
        assert "Home Star" in html and "Away Star" in html
        assert "30.0" in html and "22.0" in html
        assert "8.0" in html and "5.0" in html

    def test_the_matchup_keeps_mlb_two_stat_geometry(self):
        html = nba_todays_page._player_matchup_html(
            {"p_home_name": "A", "p_away_name": "B"})
        assert html.count("fb-pitcher") == 3   # wrapper + one per side
        assert html.count("pstats") == 2

    def test_a_missing_player_renders_a_box_rather_than_crashing(self):
        html = nba_todays_page._player_matchup_html({})
        assert "fb-pitchers" in html

    def test_the_venue_reaches_the_card(self):
        card = nba_todays_page._nba_mirror_card_html(
            self._frame(venue="Little Caesars Arena").iloc[0], None, "")
        assert "Little Caesars Arena" in card
