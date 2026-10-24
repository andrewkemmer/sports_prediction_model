"""Focused NHL frontend date-boundary regressions.

The fixtures are synthetic and local-only; no production artifact is changed.
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

FRONTEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = FRONTEND_DIR.parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import nhl_slate_view as nsv  # noqa: E402
import utils  # noqa: E402
import nhl_todays_page  # noqa: E402

NHL_DD = REPO_ROOT / "nhl-backend" / "data_delivery"
DATE = "20990101"
DATE_ISO = "2099-01-01"
# A slate whose date is PAST in ET — the production state the day after a
# pipeline window ends: the strict current-slate resolver refuses it (a past
# slate_date is not a live board), so no current artifact resolves while the
# published family member still serves the archive board.
ARCHIVE_ISO = (date.today() - timedelta(days=1)).isoformat()
ARCHIVE_DATE = ARCHIVE_ISO.replace("-", "")


def test_price_total_and_spread_renormalize_integer_lines_to_a_coherent_quote():
    """Raw integer-line artifact columns can carry pre-normalization mass
    (documented at half_stop_pair); a quoted three-way must sum to 1."""
    row = {"p_over_5": 0.40, "p_under_5": 0.52, "p_push_total_5": 0.08}
    o, u, p = nsv.price_total(row, 5)
    assert o + u + p == pytest.approx(1.0)
    assert o == pytest.approx(0.40)      # already coherent: unchanged

    row2 = {"p_over_5": 0.44, "p_under_5": 0.57, "p_push_total_5": 0.08}  # 1.09
    o, u, p = nsv.price_total(row2, 5)
    assert o + u + p == pytest.approx(1.0)

    row3 = {"p_home_cover_2": 0.55, "p_push_2": 0.06}    # implied away 0.39
    h, p, a = nsv.price_spread(row3, 2)
    assert h + p + a == pytest.approx(1.0)

    row4 = {"p_home_cover_2": 0.70, "p_push_2": 0.40}    # overlap: implied away < 0
    h, p, a = nsv.price_spread(row4, 2)
    assert h + p + a == pytest.approx(1.0)
    assert a >= 0.0 and p >= 0.0 and h >= 0.0


def test_price_total_half_point_is_coherent_by_construction():
    row = {"p_over_5": 0.55}
    o, u, p = nsv.price_total(row, 5.5)
    assert o == pytest.approx(0.55)
    assert u == pytest.approx(0.45)
    assert p == 0.0
    assert o + u + p == pytest.approx(1.0)


def test_card_spans_quote_the_2way_push_folded_basis_with_a_push_note():
    """MLB card convention: the O/U and RL spans quote the two decided sides
    on the 2-WAY push-folded basis (they sum to 100%) and carry the push as
    its own grey note — raw three-way side shares read as a broken quote
    (Over 55% / Under 22% + 23% push). The ±0.5 stop was already 2-way by
    construction; integer totals and integer spreads were not."""
    row = {
        "mu_h": 3.2, "mu_a": 2.9, "fair_total": 5.0, "fair_spread": 1.0,
        "p_over_5": 0.42, "p_under_5": 0.35, "p_push_total_5": 0.23,
        "p_home_cover_1": 0.32, "p_push_1": 0.23,
        "p_home_cover_0": 0.60, "p_push_0": 0.07,
    }
    html = nsv.runengine_html(row, "CAR", "FLA", total_line=5.0,
                              home_spread=-1.0)
    import re as _re

    # O/U: 2-way over = 0.42/(0.42+0.35) = 54.5% -> 55%; under = 45%.
    m_ou = _re.search(r"O/U 5: Over (\d+)% / Under (\d+)%", html)
    assert m_ou, html
    assert int(m_ou.group(1)) + int(m_ou.group(2)) == 100, html
    assert "23% push" in html
    # RL (home quoted -1 -> threshold +1): 2-way home cover =
    # 0.32/(0.32+0.45) = 41.6% -> 42%; away = 58%.
    m_rl = _re.search(r"RL: CAR −1 (\d+)%", html)
    assert m_rl, html
    assert "FLA +1" in html
    assert "23% push" in html.split("RL:")[1]
    # The ±0.5 stop remains 2-way by construction (raw vs derived pair).
    html_hs = nsv.runengine_html(row, "CAR", "FLA", half_stop=True)
    assert "−0.5" in html_hs and "+0.5" in html_hs and "run-ML" in html_hs


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


def _archive_record() -> dict:
    """A two-game slate dated YESTERDAY — the day-after-a-window-end state."""
    games = [
        _game("2800010001", "BOS", "NYR", f"{ARCHIVE_ISO}T00:30:00Z", 0.60),
        _game("2800010002", "EDM", "VAN", f"{ARCHIVE_ISO}T01:30:00Z", 0.55),
    ]
    for g in games:
        g["game_date"] = ARCHIVE_ISO
    record = _record(games)
    record["created_utc"] = f"{ARCHIVE_ISO}T12:00:00Z"
    record["slate_date"] = ARCHIVE_ISO
    return record


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
    with _stage_family_record(DATE, _five_game_record()):
        yield


@contextmanager
def _stage_family_record(day: str, record: dict):
    """Write ``record`` as the ``nhl_moneyline_v1_<day>`` family member and
    restore the directory on the way out (the fixtures never touch a
    production artifact that already exists)."""
    NHL_DD.mkdir(parents=True, exist_ok=True)
    path = NHL_DD / f"nhl_moneyline_v1_{day}.json"
    existed = path.exists()
    previous = path.read_bytes() if existed else None
    path.write_text(json.dumps(record), encoding="utf-8")
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


def test_nhl_missing_current_slate_renders_the_newest_board_with_no_notice(
        monkeypatch):
    """2026-10-13 regression: the pipeline window ended Oct 12, so the
    strict current-slate resolver returned nothing (a past slate_date is
    not a live board) and the dashboard printed "No confirmed current NHL
    slate artifact is available" directly above an otherwise complete
    Oct 12 board. MLB/NBA/NFL carry no such notice in this state — a first
    visit with no live artifact just resolves to the newest board the store
    holds, and the board's own header strip is the page header (no page
    title either, NFL's documented MLB rule)."""
    monkeypatch.setattr(utils, "_nhl_current_slate_record",
                        lambda *args, **kwargs: {})
    monkeypatch.setattr(utils, "valid_dates",
                        lambda *args, **kwargs: (ARCHIVE_DATE,))
    with _stage_family_record(ARCHIVE_DATE, _archive_record()):
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
        assert "No confirmed current NHL slate artifact" not in text
        # No page title either — the header strip is the page header.
        assert "NHL — Moneyline" not in text
        # The archive board itself still renders, on the seeded date.
        assert app.session_state["selected_date"] == ARCHIVE_DATE
        assert "2 of 2 games shown" in text
        for matchup in ("BOS", "NYR", "EDM", "VAN"):
            assert matchup in text


# ---------------------------------------------------------------------------
# Totals-law modeled-PMF namespace regressions (2026-10-14).
#
# The chart's "modeled" series must read ONLY the totals push family
# (p_push_total_<k>). The old cross-family fallback fell back to the shared
# p_push_<k> columns — on current artifacts the MARGIN/run-line push — so the
# NHL line priced P(margin=1) = 22.5% and P(margin=2) = 12.8% as if they were
# P(total=1)/P(total=2), the phantom 22%/13% spike at totals 1-2 (and the NFL
# line the same at 0-14). Same overlap class as the Brier-315 artifact bug
# nhl_market_diagnostics already documents.
# ---------------------------------------------------------------------------

def test_shipped_total_pmf_never_reads_the_margin_push_family():
    import numpy as np

    import market_diagnostics as diag

    frame = pd.DataFrame({
        "p_push_1": [0.225, 0.225],        # MARGIN push (must never be read)
        "p_push_2": [0.128, 0.128],
        "p_push_total_4": [0.12, 0.12],    # totals push (the real family)
    })
    for namespace in ("nhl", "nfl"):
        values = diag.shipped_total_pmf(frame, [1, 2, 4], namespace)
        assert np.isnan(values[0]) and np.isnan(values[1]), namespace
        assert values[2] == pytest.approx(0.12), namespace
    # MLB reads its own p_push_<k> family — unchanged by the fallback removal.
    mlb = pd.DataFrame({"p_push_1_0": [0.02, 0.02]})
    assert diag.shipped_total_pmf(mlb, [1], "mlb")[0] == pytest.approx(0.02)


def test_nhl_totals_distribution_gaps_off_grid_and_calls_boundary_tails():
    """modeled[] gaps where the artifact ships no totals push (0-3, 13-15),
    and the callouts come from the boundary columns: p_under_5 = P(total<5) =
    P(total<=4), p_over_9 = P(total>9) = P(total>=10) — never a sum over a
    gapped series (nan) or the margin family (0.572 / wrong)."""
    import nhl_market_diagnostics as nd

    n = 4
    frame = pd.DataFrame({
        "total": [3, 5, 5, 9],
        **{f"p_push_total_{k}": [0.10] * n for k in range(4, 13)},
        "p_under_5": [0.13] * n,
        "p_over_9": [0.07] * n,
        "p_push_1": [0.22] * n,   # margin family present but off-limits
        "p_push_2": [0.13] * n,
    })
    dist = nd.total_distribution(frame)
    assert dist["warning"] is None
    assert all(v != v for v in dist["modeled"][:4])      # 0-3: NaN gap
    assert all(v != v for v in dist["modeled"][13:])      # 13-15: NaN gap
    assert dist["modeled"][4] == pytest.approx(0.10)      # on-grid unchanged
    c = dist["callouts"]
    assert c["P(total<=4)"]["modeled"] == pytest.approx(0.13)
    assert c["P(total>=10)"]["modeled"] == pytest.approx(0.07)
    assert c["P(total<=4)"]["observed"] == pytest.approx(0.25)  # total=3 only


def test_nfl_totals_distribution_ignores_the_spread_push_family():
    """NFL: p_push_0..14 is the spread/margin family and p_push_15..23 does
    not exist — the modeled line must gap across 0-23 rather than plot the
    spread pushes as totals, and the tail callouts read p_under_36/p_over_59."""
    import nfl_market_diagnostics as nd

    n = 3
    frame = pd.DataFrame({
        "total": [41, 44, 47],
        **{f"p_push_total_{k}": [0.05] * n for k in range(24, 67)},
        "p_push_1": [0.11] * n,
        "p_push_2": [0.09] * n,
        "p_under_36": [0.30] * n,
        "p_over_59": [0.08] * n,
    })
    dist = nd.total_distribution(frame)
    assert dist["warning"] is None
    assert all(v != v for v in dist["modeled"][:24])     # 0-23: NaN gap
    assert dist["modeled"][24] == pytest.approx(0.05)     # on-grid unchanged
    c = dist["callouts"]
    assert c["P(total<=35)"]["modeled"] == pytest.approx(0.30)
    assert c["P(total>=60)"]["modeled"] == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# Full-support totals-PMF remediation (2026-10-14): the emitter now ships
# p_push_total_<k> over the priced support, so the modeled line spans every
# total that settled — the sub-grid band (NHL finals <= 3 are 13.3% of the
# store; NFL <= 23 is 4.7% and >= 67 another 7.3%) can no longer be a gap.
# ---------------------------------------------------------------------------

def test_nhl_totals_distribution_draws_the_shipped_full_support_pmf():
    """When the artifact ships the integer PMF mass beyond the market grid
    the modeled line spans it; fixtures shipping only the grid (older
    artifacts) still gap honestly (tests above)."""
    import nhl_market_diagnostics as nd

    n = 3
    frame = pd.DataFrame({
        "total": [2, 6, 14],
        **{f"p_push_total_{k}": [0.05] * n for k in range(0, 16)},
        "p_under_5": [0.10] * n,
        "p_over_9": [0.06] * n,
        "p_push_1": [0.22] * n,   # margin family — still off-limits
    })
    dist = nd.total_distribution(frame)
    assert dist["warning"] is None
    assert all(v == v for v in dist["modeled"]), "full-support line has gaps"
    assert dist["modeled"][2] == pytest.approx(0.05)     # sub-grid mass drawn
    assert dist["modeled"][14] == pytest.approx(0.05)    # above-grid mass drawn
    assert dist["modeled"][5] == pytest.approx(0.05)     # grid unchanged
    # Boundary callouts stay the exact tail authority.
    c = dist["callouts"]
    assert c["P(total<=4)"]["modeled"] == pytest.approx(0.10)
    assert c["P(total>=10)"]["modeled"] == pytest.approx(0.06)


def test_nhl_distribution_chart_spans_every_settled_total():
    """A real final beyond the default span (the 16-goal game in the store)
    is never silently dropped from the fit check — the chart's ks extend to
    the observed maximum."""
    import nhl_market_diagnostics as nd

    frame = pd.DataFrame({"total": [5, 16], "p_push_total_5": [0.10, 0.10]})
    dist = nd.total_distribution(frame)
    assert dist["warning"] is None
    assert len(dist["ks"]) == 17                      # 0..16
    assert dist["observed"][16] == pytest.approx(0.5)
    assert dist["callouts"]["P(total>=10)"]["observed"] == pytest.approx(0.5)


def test_nfl_totals_distribution_draws_full_support_and_counts_high_tails():
    """NFL: the full-support PMF covers both off-grid bands, the 80-point
    final (beyond the old 66 cap) is on-chart with its bar, and the
    P(total>=60) observed callout counts it instead of stopping at 66."""
    import nfl_market_diagnostics as nd

    n = 3
    frame = pd.DataFrame({
        "total": [20, 45, 80],
        **{f"p_push_total_{k}": [0.02] * n for k in range(0, 81)},
        "p_under_36": [0.30] * n,
        "p_over_59": [0.10] * n,
    })
    dist = nd.total_distribution(frame)
    assert dist["warning"] is None
    assert all(v == v for v in dist["modeled"]), "full-support line has gaps"
    assert dist["modeled"][20] == pytest.approx(0.02)
    assert dist["modeled"][80] == pytest.approx(0.02)
    assert len(dist["ks"]) == 81                       # 0..80
    assert dist["observed"][80] == pytest.approx(1 / 3, abs=1e-5)
    assert dist["callouts"]["P(total>=60)"]["observed"] == pytest.approx(
        1 / 3, abs=1e-4)   # callouts round to 4 decimals
    assert dist["callouts"]["P(total<=35)"]["observed"] == pytest.approx(
        1 / 3, abs=1e-4)


def test_distribution_noise_band_math_gaps_and_spec():
    """The Distribution tab shades a 95% binomial band around the modeled
    line: per-bin p-hat ± 1.96·SE over the plotted game count; NaN-gap
    bins (unshipped mass) carry no band; the chart layers area+bar+line."""
    import numpy as np

    import market_diagnostics as diag

    dist = {"ks": [4, 5, 6], "observed": [0.2, 0.5, 0.3],
            "modeled": [0.25, float("nan"), 0.09],
            "callouts": {}, "n_games": 100, "warning": None}
    band = diag.distribution_noise_band(dist)
    assert list(band.columns) == ["k", "lo", "hi"]
    assert list(band["k"]) == [4, 6]          # NaN bin skipped
    se = (0.25 * 0.75 / 100) ** 0.5
    assert band["lo"].iloc[0] == pytest.approx(max(0., 0.25 - 1.96 * se))
    assert band["hi"].iloc[0] == pytest.approx(min(1., 0.25 + 1.96 * se))
    assert (band["lo"] <= band["hi"]).all()
    # Degenerate inputs carry no band but never crash the builder.
    assert diag.distribution_noise_band({**dist, "n_games": 0}).empty
    assert diag.distribution_noise_band({}).empty
    spec = json.loads(diag.chart_distribution(dist).to_json())
    marks = [layer["mark"]["type"] for layer in spec["layer"]]
    assert marks == ["area", "bar", "line"]
    area = spec["layer"][0]
    assert set(area["encoding"]) == {"x", "y", "y2"}


def test_nfl_nhl_spread_dicts_feed_the_shared_noise_band():
    """The NFL/NHL Distribution tabs reuse the shared chart builder: their
    spread dicts (NaN gaps outside the shipped push family) must produce a
    band with rows exactly at the shipped bins, and an area+bar+line spec."""
    import numpy as np

    import market_diagnostics as diag
    import nfl_market_diagnostics as nfl
    import nhl_market_diagnostics as nhl

    rng = np.random.default_rng(0)
    margins = rng.integers(-20, 21, size=400).astype(float)
    for mod, lo, hi in ((nfl, -14, 14), (nhl, -8, 8)):
        cols = {}
        for k in range(lo, hi + 1):
            tag = f"p_push_m{-k}" if k < 0 else f"p_push_{k}"
            cols[tag] = np.full(400, 0.03)
        frame = pd.DataFrame({"margin": margins, **cols})
        dist = mod.spread_distribution(frame)
        assert dist["warning"] is None and dist["n_games"] == 400
        finite = {k for k, v in zip(dist["ks"], dist["modeled"]) if v == v}
        assert finite == set(range(lo, hi + 1))
        band = diag.distribution_noise_band(dist)
        assert list(band["k"]) == sorted(finite)
        assert ((band["lo"] <= 0.03) & (0.03 <= band["hi"])).all()
        spec = json.loads(diag.chart_distribution(dist).to_json())
        assert [layer["mark"]["type"]
                for layer in spec["layer"]] == ["area", "bar", "line"]


# ---------------------------------------------------------------------------
# NFL like-with-like spread tails (2026-10-16): the ±10 modeled callouts
# are the exact shipped cover+push composites over FULL support — P(≥10) =
# mean(p_home_cover_10 + p_push_10), P(≤−10) = 1 − mean(p_home_cover_m10)
# — against the full-tail observed rates (blowouts beyond ±14 count on
# both sides). Absent columns render None (the caption shows '--').
# ---------------------------------------------------------------------------

def test_nfl_spread_tail_callouts_use_cover_push_composites():
    import nfl_market_diagnostics as nd

    n = 5
    frame = pd.DataFrame({
        "margin": [-20, -10, 0, 10, 25],
        **{(f"p_push_m{-k}" if k < 0 else f"p_push_{k}"): [0.03] * n
           for k in range(-14, 15)},
        "p_home_cover_10": [0.10] * n,
        "p_push_10": [0.02] * n,
        "p_home_cover_m10": [0.80] * n,
    })
    dist = nd.spread_distribution(frame)
    assert dist["warning"] is None
    c = dist["callouts"]
    # Observed full tails count the blowouts beyond ±14.
    assert c["P(margin<=-10)"]["observed"] == pytest.approx(0.4)
    assert c["P(margin>=10)"]["observed"] == pytest.approx(0.4)
    # Modeled tails are the composites — not a slice of the ±14 line.
    assert c["P(margin>=10)"]["modeled"] == pytest.approx(0.12)
    assert c["P(margin<=-10)"]["modeled"] == pytest.approx(0.20)
    # The per-bin line still reads the margin push family.
    assert dist["modeled"][dist["ks"].index(10)] == pytest.approx(0.02)
    # Absent cover/push columns degrade to None, never a crash.
    bare = pd.DataFrame({
        "margin": [-20, -10, 0, 10, 25],
        **{(f"p_push_m{-k}" if k < 0 else f"p_push_{k}"): [0.03] * n
           for k in range(-14, 15)},
    })
    bare_c = nd.spread_distribution(bare)["callouts"]
    assert bare_c["P(margin<=-10)"]["modeled"] is None
    assert bare_c["P(margin>=10)"]["modeled"] is None
    assert bare_c["P(margin<=-10)"]["observed"] == pytest.approx(0.4)


def test_nfl_spread_never_reads_totals_mass_in_the_margin_namespace():
    """Writer-hygiene mirror: even a legacy artifact carrying totals mass
    as p_push_24..66 (the pre-split dup) cannot move the spread chart —
    only SPREAD_GRID bins are finite and the tails stay cover+push."""
    import nfl_market_diagnostics as nd

    n = 4
    frame = pd.DataFrame({
        "margin": [-11, -3, 7, 18],
        **{(f"p_push_m{-k}" if k < 0 else f"p_push_{k}"): [0.03] * n
           for k in range(-14, 15)},
        **{f"p_push_{u}": [0.99] * n for u in range(24, 67)},
        "p_home_cover_10": [0.10] * n,
        "p_push_10": [0.02] * n,
        "p_home_cover_m10": [0.80] * n,
    })
    dist = nd.spread_distribution(frame)
    assert dist["warning"] is None
    finite = {k for k, v in zip(dist["ks"], dist["modeled"]) if v == v}
    assert finite == set(range(-14, 15))
    assert dist["modeled"][dist["ks"].index(24)] != \
        dist["modeled"][dist["ks"].index(24)]  # NaN: dup unread
    c = dist["callouts"]
    assert c["P(margin>=10)"]["modeled"] == pytest.approx(0.12)
    assert c["P(margin<=-10)"]["modeled"] == pytest.approx(0.20)


def test_nfl_totals_cards_read_the_totals_push_namespace_never_the_margin_one():
    """The totals 3-way push leg resolves ``p_push_total_<U>``. The shared
    ``p_push_<U>`` name is the MARGIN family's (and the pre-split totals
    dup's), so margin mass at a totals line can never move a totals card —
    the cross-family read the writer-hygiene gate exists to prevent."""
    import nfl_market_diagnostics as nd

    def frame(margin_mass: float, totals_mass: float) -> pd.DataFrame:
        return pd.DataFrame({
            "total": [27.0, 30.0], "fair_total": [24.0, 27.0],
            "p_over_fair": [0.48, 0.48],
            "p_push_24": [margin_mass, 0.02],
            "p_push_27": [0.02, margin_mass],
            "p_push_total_24": [totals_mass, 0.02],
            "p_push_total_27": [0.02, totals_mass],
        })

    under = nd.totals_monitor_stats(frame(0.02, 0.02))
    assert under["n"] == 2
    # Margin-namespace mass at a totals line is unread: a wrong read would
    # flip both picks to Over (0.02 -> 0.45 turns p2 0.49 into 0.87).
    assert nd.totals_monitor_stats(frame(0.45, 0.02)) == under
    # The totals namespace IS the leg the card reads.
    assert nd.totals_monitor_stats(frame(0.02, 0.30)) != under


def test_nfl_slate_totals_read_the_totals_push_namespace_never_the_margin_one():
    """The slate box's totals push band resolves ``p_push_total_<U>``: the
    shared ``p_push_<U>`` name is the MARGIN family's, so margin mass at a
    totals line can never be served as the totals push (and a line the
    artifact does not ship stays None, never borrowed)."""
    import nfl_slate_view as sv

    cols = sv.total_columns()
    for U in sv.TOTAL_GRID:
        assert cols[U][2] == f"p_push_total_{U}"
    shipped = {"p_over_44": 0.46, "p_under_44": 0.51,
               "p_push_44": 0.99, "p_push_total_44": 0.03}
    assert sv.price_total(shipped, 44) == pytest.approx((0.46, 0.51, 0.03))
    legacy_only = {"p_over_44": 0.46, "p_under_44": 0.51, "p_push_44": 0.99}
    assert sv.price_total(legacy_only, 44) == (0.46, 0.51, None)


def test_nfl_slate_gate_resolves_the_totals_push_namespace():
    """The shared grid gate (utils._run_engine_grid_usable) must accept the
    canonical totals namespace — requiring the legacy p_push_<U> name marked
    every post-split NFL artifact unusable — while still preferring the
    canonical column over a legacy dup that rides the same rows."""
    import pandas as pd

    import utils

    def frame(totals_push: str | None) -> pd.DataFrame:
        grid = {}
        for line in range(-14, 15):
            label = f"m{-line}" if line < 0 else str(line)
            grid[f"p_home_cover_{label}"] = 0.48
            grid[f"p_push_{label}"] = 0.02
        for line in range(24, 67):
            grid[f"p_over_{line}"] = 0.48
            grid[f"p_under_{line}"] = 0.48
            grid[f"p_push_total_{line}"] = 0.04
            if totals_push == "legacy":
                grid[f"p_push_{line}"] = 0.99
        if totals_push == "none":
            grid = {k: v for k, v in grid.items()
                    if not k.startswith("p_push_total_")}
        return pd.DataFrame([dict(grid, kind="slate",
                                  gameday="2026-10-15", game_id="g1")])

    assert utils._run_engine_grid_usable(frame("canonical"), "nfl") is True
    # A legacy dup on the same rows cannot out-vote the canonical column.
    assert utils._run_engine_grid_usable(frame("legacy"), "nfl") is True
    # Pre-split artifacts (no totals namespace at all) keep their dup read.
    no_ns = frame("none")
    assert utils._run_engine_grid_usable(no_ns, "nfl") is False
    legacy_only = frame("none")
    for line in range(24, 67):
        legacy_only[f"p_push_{line}"] = 0.04
    assert utils._run_engine_grid_usable(legacy_only, "nfl") is True
    # A totals line the artifact does not price at all stays unusable.
    missing = frame("canonical")
    missing["p_over_44"] = None
    assert utils._run_engine_grid_usable(missing, "nfl") is False


def test_nfl_spread_reads_the_push_extension_to_the_blowout_tails():
    """Current artifacts fill the modeled line over −30..+30 via the
    push-only extension (the +24..+30 overlap is margin mass under the name
    it shares with the totals grid, which reads its own p_push_total_<U>);
    pre-extension artifacts keep their gaps there — while the tail
    composites hold in both vintages (never a slice of the per-bin line)."""
    import nfl_market_diagnostics as nd

    n = 3
    full = pd.DataFrame({
        "margin": [0, 25, -25],
        **{(f"p_push_m{-k}" if k < 0 else f"p_push_{k}"): [0.02] * n
           for k in range(-30, 31)},
        **{f"p_push_total_{k}": [0.05] * n for k in range(24, 31)},
        "p_home_cover_10": [0.10] * n,
        "p_push_10": [0.02] * n,
        "p_home_cover_m10": [0.80] * n,
    })
    dist = nd.spread_distribution(full)
    assert dist["warning"] is None
    assert dist["modeled"][dist["ks"].index(23)] == pytest.approx(0.02)
    assert dist["modeled"][dist["ks"].index(-25)] == pytest.approx(0.02)
    # The positive arm ships THROUGH the totals grid: 24..30 plot the
    # extension's own mass, never the totals family's 0.05.
    for k in range(24, 31):
        assert dist["modeled"][dist["ks"].index(k)] == pytest.approx(0.02), k
    # ...and the totals family stays the only thing a totals read resolves.
    over = {"total": [30.0], "fair_total": [24.0], "p_over_fair": [0.5],
            "p_over_24": [0.5], "p_under_24": [0.45],
            "p_push_24": [0.02], "p_push_total_24": [0.05]}
    assert nd.totals_monitor_stats(pd.DataFrame(over))["n"] == 1
    legacy = pd.DataFrame({
        "margin": [0, 25, -25],
        **{(f"p_push_m{-k}" if k < 0 else f"p_push_{k}"): [0.02] * n
           for k in range(-14, 15)},
        "p_home_cover_10": [0.10] * n,
        "p_push_10": [0.02] * n,
        "p_home_cover_m10": [0.80] * n,
    })
    old = nd.spread_distribution(legacy)
    assert old["modeled"][old["ks"].index(10)] == pytest.approx(0.02)
    assert old["modeled"][old["ks"].index(25)] != \
        old["modeled"][old["ks"].index(25)]  # NaN: legacy gap kept
    assert old["callouts"]["P(margin>=10)"]["modeled"] == pytest.approx(0.12)
    assert old["callouts"]["P(margin<=-10)"]["modeled"] == pytest.approx(0.20)
    # Legacy totals dup without the extension edge (pre-split vintage): its
    # p_push_24..30 are totals mass, so the overlap stays gaps.
    dup = pd.DataFrame({
        "margin": [0, 25, -25],
        **{(f"p_push_m{-k}" if k < 0 else f"p_push_{k}"): [0.02] * n
           for k in range(-14, 15)},
        **{f"p_push_{u}": [0.99] * n for u in range(24, 31)},
        "p_home_cover_10": [0.10] * n,
        "p_push_10": [0.02] * n,
        "p_home_cover_m10": [0.80] * n,
    })
    dup_dist = nd.spread_distribution(dup)
    assert dup_dist["modeled"][dup_dist["ks"].index(25)] != \
        dup_dist["modeled"][dup_dist["ks"].index(25)]  # NaN: dup unread


# ---------------------------------------------------------------------------
# Home / Away distribution toggles (2026-10-16): the per-side scoring law.
#
# The Distribution view toggles Total / Spread / Home / Away on all four
# sports. The Home / Away views plot the observed per-side finals PLUS the
# model's OWN per-side Gaussian as the modeled line: the artifacts ship no
# per-team score PMF column, but the bivariate scoring core the backend
# prices from has a plain Gaussian marginal per side, so the line is
# DERIVED from shipped parameters only (side mean split from the calibrated
# distribution_mean_total / distribution_mean_margin, variance = the
# monitor's fit.variance factor × that mean). Without a usable fit block it
# falls back to the all-gap line — never borrowed from another push
# namespace, never refitted on outcomes.
# ---------------------------------------------------------------------------

def test_team_score_distribution_without_a_fit_block_stays_observed_only():
    """No calibrated mean columns and no fit.variance block -> the line
    stays the all-gap line: unavailable, never borrowed from another push
    namespace."""
    import market_diagnostics as diag

    frame = pd.DataFrame({
        "home_score": [5, 0, 3, 9],
        "away_score": [2, 4, 3, 1],
    })
    dist = diag.team_score_distribution(frame, "home")
    assert dist["warning"] is None
    # A blank home side (0 runs) is real mass, so the span anchors at 0;
    # the kmax floor (15) keeps the totals-view span shape.
    assert dist["ks"][0] == 0 and dist["ks"][-1] == 15
    assert dist["observed"][0] == pytest.approx(0.25)   # one 0-run home side
    assert dist["observed"][5] == pytest.approx(0.25)
    assert dist["observed"][9] == pytest.approx(0.25)
    assert all(v != v for v in dist["modeled"])         # all gaps, never borrowed
    c = dist["callouts"]
    assert c["mean"]["observed"] == pytest.approx(4.25)  # (5+0+3+9)/4
    assert c["P(home == 0)"]["observed"] == pytest.approx(0.25)
    # margins are +3, -4, 0, +8 -> home wins 2 of 4 (the tie is no win)
    assert c["P(home wins)"]["observed"] == pytest.approx(0.5)
    assert c["mean"]["modeled"] is None
    assert "never derived or refitted" in c["note"]
    away = diag.team_score_distribution(frame, "away")
    assert away["callouts"]["P(away wins)"]["observed"] == pytest.approx(0.25)
    # A missing score column warns loudly, never fabricates.
    bare = pd.DataFrame({"home_score": [5, 0]})
    assert "away_score" in diag.team_score_distribution(bare, "home")["warning"]
    with pytest.raises(ValueError):
        diag.team_score_distribution(frame, "total")


def test_team_score_distribution_grid_spans_the_settled_scores():
    """The Home/Away grid spans every side score that actually settled —
    and does NOT pad leading zeros when a sport's sides never approach 0
    (NHL goals 3..6 start at 3, the NBA points span starts at its floor).
    NFL/NHL re-export MLB's helper, so the behavior is shared."""
    import nfl_market_diagnostics as nfl
    import nhl_market_diagnostics as nhl

    frame = pd.DataFrame({"home_score": [3, 4, 6], "away_score": [3, 4, 4]})
    for mod in (nhl, nfl):
        home = mod.team_score_distribution(frame, "home")
        assert home["warning"] is None
        assert home["ks"] == [3, 4, 5, 6], mod.__name__
        assert home["observed"][home["ks"].index(6)] == \
            pytest.approx(1 / 3, abs=1e-5)   # series rounds to 5 decimals
        # tied finals (3-3, 4-4) count as a win for neither side
        assert home["callouts"]["P(home wins)"]["observed"] == \
            pytest.approx(1 / 3, abs=1e-4)   # callouts round to 4
        away = mod.team_score_distribution(frame, "away")
        assert away["callouts"]["P(away wins)"]["observed"] == pytest.approx(0.0)
        # every modeled bin is a gap — no totals/margin mass is ever
        # borrowed to fabricate a per-team PMF line.
        assert all(v != v for v in home["modeled"])


# --- The per-side modeled line (the model's own Gaussian marginal) --------

# The monitor's shipped fit.variance block, verbatim in shape.
SIDE_FIT_VARIANCE = {"home_variance": 1.0, "away_variance": 1.0,
                     "correlation": -0.01, "total_skew": 0.07,
                     "margin_skew": -0.04}


def _side_frame() -> pd.DataFrame:
    """A decided frame with the calibrated mean columns the artifact ships.
    total = 7 and margin = 1 on every row -> side means home 4, away 3."""
    return pd.DataFrame({
        "home_score": [5, 0, 3, 9],
        "away_score": [2, 4, 3, 1],
        "distribution_mean_total": [7.0, 7.0, 7.0, 7.0],
        "distribution_mean_margin": [1.0, 1.0, 1.0, 1.0],
        "p_home_win_derived": [0.55] * 4,
        "p_away_win_derived": [0.45] * 4,
    })


def test_side_line_is_the_models_own_gaussian_from_shipped_parameters():
    """With the monitor's fit block the orange line is the model's own
    per-side Gaussian: side mean = the calibrated split (total ± margin)/2,
    variance = the shipped factor × that mean. The lattice mass is NOT
    renormalized — sub-zero mass stays off-grid and is disclosed, which is
    what keeps home + away = the calibrated total mean."""
    import market_diagnostics as diag

    frame = _side_frame()
    home = diag.team_score_distribution(frame, "home",
                                        variance=SIDE_FIT_VARIANCE)
    away = diag.team_score_distribution(frame, "away",
                                        variance=SIDE_FIT_VARIANCE)
    assert home["warning"] is None and away["warning"] is None
    assert home["ks"] == list(range(16))
    line = home["modeled"]
    assert all(v == v for v in line) and all(v >= 0. for v in line)
    hc, ac = home["callouts"], away["callouts"]
    # side means are the shipped calibrated splits: (7+1)/2 and (7-1)/2
    assert hc["mean"]["modeled"] == pytest.approx(4.0, abs=1e-9)
    assert ac["mean"]["modeled"] == pytest.approx(3.0, abs=1e-9)
    # the identity that the non-renormalized lattice preserves exactly
    assert (hc["mean"]["modeled"] + ac["mean"]["modeled"]) == pytest.approx(
        float(frame["distribution_mean_total"].mean()), abs=1e-9)
    # P(side wins) stays the EXACT shipped margin authority, not a
    # Gaussian tail re-derived in the frontend.
    assert hc["P(home wins)"]["modeled"] == pytest.approx(0.55)
    assert ac["P(away wins)"]["modeled"] == pytest.approx(0.45)
    # Sub-zero mass is disclosed and left off-grid (not renormalized away):
    # plotted mass + off-grid mass = 1 for a grid that reaches the tail.
    off = hc["off_grid"]["modeled"]
    assert 0.010 < off < 0.015            # mean 4, sd 2 -> Phi(-2.25)
    assert sum(line) + off == pytest.approx(1.0, abs=1e-3)
    assert "never refitted" in hc["note"]


def test_side_line_is_derived_never_fitted_on_outcomes():
    """Changing the finals must NOT move the modeled line — it comes from
    shipped parameters, so it cannot be a refit on outcomes."""
    import market_diagnostics as diag

    frame = _side_frame()
    base = diag.team_score_distribution(frame, "home",
                                        variance=SIDE_FIT_VARIANCE)
    flipped = frame.copy()
    flipped["home_score"] = [0, 0, 0, 0]     # absurd shutout finals
    flipped["away_score"] = [12, 12, 12, 12]
    moved = diag.team_score_distribution(flipped, "home",
                                         variance=SIDE_FIT_VARIANCE)
    assert (moved["callouts"]["mean"]["modeled"]
            == base["callouts"]["mean"]["modeled"])
    shared = dict(zip(base["ks"], base["modeled"]))
    for k, p in zip(moved["ks"], moved["modeled"]):
        if k in shared:
            assert p == pytest.approx(shared[k], abs=1e-12), k
    # The shipped variance factor IS the width knob.
    wide = diag.team_score_distribution(frame, "home",
                                        variance={**SIDE_FIT_VARIANCE,
                                                  "home_variance": 4.0})
    assert max(wide["modeled"]) < max(base["modeled"])
    assert (wide["callouts"]["mean"]["modeled"]
            == pytest.approx(base["callouts"]["mean"]["modeled"]))


def test_side_line_is_shared_by_the_nfl_and_nhl_re_exports():
    """NFL/NHL re-export MLB's helper, so the modeled line — and the
    all-gap fallback without a fit block — is identical on all three."""
    import market_diagnostics as diag
    import nfl_market_diagnostics as nfl
    import nhl_market_diagnostics as nhl

    frame = _side_frame()
    ref = diag.team_score_distribution(frame, "home",
                                       variance=SIDE_FIT_VARIANCE)
    for mod in (nfl, nhl):
        got = mod.team_score_distribution(frame, "home",
                                          variance=SIDE_FIT_VARIANCE)
        assert got["modeled"] == ref["modeled"], mod.__name__
        assert got["callouts"]["mean"] == ref["callouts"]["mean"]
        assert not diag.distribution_noise_band(got).empty, mod.__name__
        # no fit block -> the all-gap fallback, band suppressed
        gap = mod.team_score_distribution(frame, "home")
        assert all(v != v for v in gap["modeled"]), mod.__name__
        assert diag.distribution_noise_band(gap).empty, mod.__name__


def test_the_day_is_sorted_earliest_puck_drop_first():
    """Cards render chronologically (NBA parity): a later game listed first
    in the artifact still renders after the earlier game, and a row with a
    missing puck drop renders last instead of first."""
    frame = pd.DataFrame([
        {"game_id": "late", "start_time_utc": "2099-01-01T23:30:00Z"},
        {"game_id": "early", "start_time_utc": "2099-01-01T19:00:00Z"},
        {"game_id": "unknown", "start_time_utc": None},
        {"game_id": "mid", "start_time_utc": "2099-01-01T21:00:00Z"},
    ])
    ordered = nhl_todays_page._sort_by_puck_drop(frame)
    assert ordered.game_id.tolist() == ["early", "mid", "late", "unknown"]


def test_puck_drop_ties_keep_artifact_order():
    frame = pd.DataFrame([
        {"game_id": "b", "start_time_utc": "2099-01-01T19:00:00Z"},
        {"game_id": "a", "start_time_utc": "2099-01-01T19:00:00Z"},
    ])
    assert nhl_todays_page._sort_by_puck_drop(frame).game_id.tolist() == ["b", "a"]


def test_side_line_chart_layers_the_band_behind_the_line():
    """With a real modeled line the Home/Away chart layers exactly like the
    Total/Spread charts: band, bars, line."""
    import market_diagnostics as diag

    dist = diag.team_score_distribution(_side_frame(), "away",
                                        variance=SIDE_FIT_VARIANCE)
    spec = json.loads(diag.chart_distribution(
        dist, x_title="Away runs scored",
        y_title="P(away runs = k)").to_json())
    assert [layer["mark"]["type"] for layer in spec["layer"]] == \
        ["area", "bar", "line"]
    band = diag.distribution_noise_band(dist)
    assert len(band) == len(dist["ks"])
    assert (band["hi"] >= band["lo"]).all()
    # a real (non-degenerate) interval at the peak of the line
    peak = int(np.argmax(dist["modeled"]))
    row = band.iloc[peak]
    assert row["hi"] > row["lo"]
