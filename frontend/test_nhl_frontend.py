"""Focused NHL frontend date-boundary regressions.

The fixtures are synthetic and local-only; no production artifact is changed.
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

FRONTEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = FRONTEND_DIR.parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import utils  # noqa: E402

NHL_DD = REPO_ROOT / "nhl-backend" / "data_delivery"
DATE = "20990101"
DATE_ISO = "2099-01-01"


def _game(game_id: str, home: str, away: str, kickoff: str | None, ph: float) -> dict:
    return {
        "game_id": game_id,
        "game_date": DATE_ISO,
        "start_time_utc": kickoff,
        "home_team": home,
        "away_team": away,
        "home_team_name": home,
        "away_team_name": away,
        "home_record": "1-0",
        "away_record": "0-1",
        "venue": "Test Arena",
        "game_status": "pre",
        "home_score": None,
        "away_score": None,
        "home_win_prob_model": ph,
        "away_win_prob_model": 1.0 - ph,
        "model_pick": home if ph >= 0.5 else away,
        "model_correct": None,
    }


def _record(games: list[dict]) -> dict:
    return {
        "created_utc": "2098-12-31T12:00:00Z",
        "slate_date": DATE_ISO,
        "n_games": len(games),
        "games": games,
    }


def _five_game_record() -> dict:
    return _record([
        _game("2099010001", "CAR", "FLA", "2099-01-01T21:00:00Z", 0.55),
        _game("2099010002", "TOR", "MTL", "2099-01-01T23:00:00Z", 0.48),
        _game("2099010003", "BOS", "NYR", "2099-01-02T00:00:00Z", 0.60),
        _game("2099010004", "EDM", "VAN", "2099-01-02T02:00:00Z", 0.57),
        _game("2099010005", "VGK", "CHI", "2099-01-02T02:30:00Z", 0.62),
    ])


def _patch_resolver(monkeypatch, record: dict) -> None:
    raw = json.dumps(record).encode("utf-8")
    monkeypatch.setattr(
        utils, "_family_dated_dates",
        lambda *args, **kwargs: [DATE],
    )
    monkeypatch.setattr(
        utils, "_fetch_bytes",
        lambda *args, **kwargs: (raw, "fixture"),
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


@contextmanager
def _staged_nhl_record():
    NHL_DD.mkdir(parents=True, exist_ok=True)
    path = NHL_DD / f"nhl_moneyline_v1_{DATE}.json"
    existed = path.exists()
    previous = path.read_bytes() if existed else None
    path.write_text(json.dumps(_five_game_record()), encoding="utf-8")
    st.cache_data.clear()
    try:
        yield
    finally:
        st.cache_data.clear()
        if previous is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(previous)


def test_nhl_resolver_accepts_real_utc_midnight_rollover(monkeypatch):
    record = _record([
        _game("rollover", "BOS", "NYR", "2099-01-02T00:00:00Z", 0.60),
    ])
    _patch_resolver(monkeypatch, record)

    resolved = utils._nhl_current_slate_record("owner", "repo", "main")

    assert resolved["slate_date"] == DATE_ISO
    assert len(resolved["games"]) == 1
    assert utils._start_time_et_date("2099-01-02T00:00:00Z") == date(2099, 1, 1)
    assert utils.start_time_et("2099-01-02T00:00:00Z") == "7:00 PM ET"
    assert utils._is_evening_start("2099-01-02T00:00:00Z")


def test_nhl_resolver_rejects_date_only_fallback_and_missing_kickoff(monkeypatch):
    for kickoff in ("2099-01-01T00:00:00Z", None):
        _patch_resolver(
            monkeypatch,
            _record([_game("fallback", "BOS", "NYR", kickoff, 0.60)]),
        )
        assert utils._nhl_current_slate_record("owner", "repo", "main") == {}


def test_nhl_adapter_preserves_missing_kickoff_as_missing():
    frame = utils.nhl_moneyline_to_frame(_record([
        _game("missing", "BOS", "NYR", None, 0.60),
    ]))
    assert len(frame) == 1
    assert pd.isna(frame.loc[0, "start_time_utc"])


def test_nhl_first_visit_renders_all_rollover_cards():
    with _staged_nhl_record():
        app = AppTest.from_file(
            str(FRONTEND_DIR / "todays_games.py"), default_timeout=60
        )
        app.session_state["sport"] = "nhl"
        app.session_state["gh_owner"] = ""
        app.session_state["gh_repo"] = ""
        app.session_state["gh_branch"] = "main"
        app.run()

        assert not list(app.exception)
        text = _all_text(app)
        assert app.session_state["selected_date"] == DATE
        assert "5 of 5 games shown" in text
        for matchup in ("CAR", "FLA", "TOR", "MTL", "BOS", "NYR",
                        "EDM", "VAN", "VGK", "CHI"):
            assert matchup in text
        assert text.count("7:00 PM ET") >= 1
        assert "No confirmed current NHL slate artifact" not in text
