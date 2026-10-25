"""Pitching+ trial family (2026-10-24): derivation + interaction contracts.

The quality/interaction replacements enter as SHADOW TRIAL CANDIDATES,
not serving features until external agreement and game-model validation
pass. Production feature generation uses three independent learned models;
the legacy fixed-weight proxy is tested separately as a comparator. Pinned here:

  * derivation: strictly-prior trailing windows (a game's own pitches never
    score its own trailing value), 100-centering (elite > 100 > poor),
    NULL discipline (debut / cameo / empty-league / stale / ledgered =
    NULL, never a fabricated 100 or 0),
  * add_diff_features: exact product/diff formulas for all 18 trial
    columns, twin NULL propagation, dome 0.0 vs outdoor-NULL wind
    discipline,
  * contracts: all 18 in KNOWN_FEATURE_COLS + RFE_CANDIDATE_COLS, NONE in
    the active serving width, metadata authored for all 18 with no orphan
    warning,
  * weather.apply_weather_features: wind x pitching+ legs mirror the
    xFIP leg (fetched / dome / missing-observation discipline).

Run with: python mlb-backend/backend/test_pitchingplus_trial.py
"""
from __future__ import annotations

import io
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

import duckdb

import features
import training
import weather as weather_mod
from config import RFE_CANDIDATE_COLS
from feature_metadata import _RICH, build_features_metadata
from feature_selection import CANDIDATE_COLS

# ── the 18 trial columns ──────────────────────────────────────────────────
PP_COLS = [
    "sp_pitchingplus_5g_diff",
    "sp_pitchingplus_5g_home",
    "sp_pitchingplus_5g_away",
    "pitcher_regression_indicator_pitchingplus_diff",
    "pitcher_regression_indicator_pitchingplus_home",
    "pitcher_regression_indicator_pitchingplus_away",
    "wind_advantage_flyball_factor_pitchingplus_diff",
    "wind_advantage_flyball_factor_pitchingplus_home",
    "wind_advantage_flyball_factor_pitchingplus_away",
    "bullpen_pitchingplus_10g_diff",
    "bullpen_pitchingplus_10g_home",
    "bullpen_pitchingplus_10g_away",
    "bullpen_pitchingplus_3g_diff",
    "bullpen_pitchingplus_3g_home",
    "bullpen_pitchingplus_3g_away",
    "bullpen_meltdown_risk_pitchingplus_diff",
    "bullpen_meltdown_risk_pitchingplus_home",
    "bullpen_meltdown_risk_pitchingplus_away",
]
assert len(PP_COLS) == 18


# ── synthetic Statcast frame ──────────────────────────────────────────────
N_GAMES = 20  # 4 appearances/game clears the 30-appearance league floor
# from game 9 on, so mid-frame trailings score on a real PIT baseline.
# Proof games (ledger/cameo/stale) all sit in the SCORED region (ix >= 8)
# so NULLs/exclusions prove the gate, not the empty league.
COLLAPSE_IX = 14  # 0-based game index of SP_H's collapse outing
DEBUT_IX = 19
CAMEO_IX = 11
LEDGER_IX = 10
STALE_IX = 12


def _synthetic_pitches() -> pd.DataFrame:
    """20 games, two contrasting starters + relievers, one collapse.

    SP_H (101): elite traits except a terrible COLLAPSE_IX outing (PIT
    proof: that game's own trailing must still read elite). SP_A (202):
    poor traits throughout. Reliever 301 (good) is ledgered on LEDGER_IX
    (availability proof); 304 throws a 2-pitch cameo in CAMEO_IX
    (cameo-NULL proof); 303 debuts in DEBUT_IX (debut-NULL proof).
    """
    rng = np.random.default_rng(11)
    rows = []
    base = date(2026, 4, 1)
    game_pks = [1000 + i for i in range(N_GAMES)]
    # Count cycle covering all three v2 states (ahead / even / behind).
    _counts = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2),
               (3, 2), (1, 0), (2, 0), (2, 1), (3, 1)]

    def _traits(elite: bool):
        if elite:
            return dict(velo=97.0, spin=2400.0, px=8.0, pz=12.0, ext=6.8,
                        cmd="edge")
        return dict(velo=90.0, spin=1800.0, px=4.0, pz=6.0, ext=5.8,
                    cmd="meat")

    def _loc(cmd: str, i: int):
        if cmd == "edge":
            return (0.8 if i % 2 else -0.8, 2.5)
        return (0.0, 2.5)

    def _row(day, gpk, pitcher, topbot, t, i):
        px, pz = _loc(t["cmd"], i)
        b, s = _counts[i % len(_counts)]
        # Value layer for the v4 location leg: the LEAGUE prices edge
        # cells high and meat cells low, so elite (edge) out-credits poor
        # (meat) once the price pool clears its floors. The pitcher's own
        # value never enters his credit — only his locations do.
        if t["cmd"] == "edge":
            pvalue = 0.03 + rng.normal(0, 0.02)
        else:
            pvalue = -0.03 + rng.normal(0, 0.02)
        return dict(
            game_date=pd.Timestamp(day), game_pk=gpk, pitcher=pitcher,
            home_team="NYY", away_team="BOS", inning_topbot=topbot,
            pitch_type="FF", balls=b, strikes=s,
            stand=("L" if i % 2 else "R"),
            delta_pitcher_run_exp=pvalue,
            sz_top=3.5, sz_bot=1.5,
            release_speed=t["velo"] + rng.normal(0, 0.4),
            spin_rate=t["spin"] + rng.normal(0, 30),
            pfx_x=t["px"] + rng.normal(0, 0.3),
            pfx_z=t["pz"] + rng.normal(0, 0.3),
            release_extension=t["ext"] + rng.normal(0, 0.05),
            events="strikeout" if i % 3 == 2 else None,
            plate_x=px + rng.normal(0, 0.05),
            plate_z=pz + rng.normal(0, 0.05))

    for g in range(N_GAMES):
        day = base + timedelta(days=g)
        gpk = game_pks[g]
        # starters: H=101 elite except the collapse game, A=202 poor always
        for pitcher, elite, n in ((101, g != COLLAPSE_IX, 90),
                                  (202, False, 90)):
            t = _traits(elite)
            topbot = "Top" if pitcher == 101 else "Bottom"
            for i in range(n):
                rows.append(_row(day, gpk, pitcher, topbot, t, i))
        # relievers (team NYY pen): 301 good, 302 poor
        for pitcher, elite, n in ((301, True, 15), (302, False, 15)):
            t = _traits(elite)
            for i in range(n):
                rows.append(_row(day, gpk, pitcher, "Bottom", t, i))
        if g == CAMEO_IX:
            t = _traits(True)
            for i in range(2):  # cameo: 2 pitches -> NULL appearance
                rows.append(_row(day, gpk, 304, "Bottom", t, i))
        if g == DEBUT_IX:
            t = _traits(True)
            for i in range(90):  # debut: first appearance -> NULL trailing
                rows.append(_row(day, gpk, 303, "Top", t, i))
    return pd.DataFrame(rows)


def _build_pp_con() -> duckdb.DuckDBPyConnection:
    """Minimal con carrying exactly what _build_pitchingplus_features needs."""
    con = duckdb.connect()
    pitches = _synthetic_pitches()
    con.register("pitches", pitches)
    base = date(2026, 4, 1)
    game_pks = [1000 + i for i in range(N_GAMES)]
    starters = pd.DataFrame([
        {"game_pk": gpk, "home_starter_id": 101, "away_starter_id": 202,
         "home_starter_hand": "R", "away_starter_hand": "R"}
        for gpk in game_pks])
    con.register("starters", starters)
    # bp_outing: reliever rows only (cameo 304 included at 2 pitches —
    # the builder weights by appearance scoring, not by this table).
    outing = []
    for g in range(N_GAMES):
        day = base + timedelta(days=g)
        gpk = game_pks[g]
        outing.append({"game_date": day, "game_pk": gpk, "team": "NYY",
                       "pitcher": 301, "n_pitches": 15})
        outing.append({"game_date": day, "game_pk": gpk, "team": "NYY",
                       "pitcher": 302, "n_pitches": 15})
        if g == CAMEO_IX:
            outing.append({"game_date": day, "game_pk": gpk, "team": "NYY",
                           "pitcher": 304, "n_pitches": 2})
    con.register("bp_outing", pd.DataFrame(outing))
    con.execute("CREATE TABLE bp_spent (team VARCHAR, pitcher BIGINT, "
                "game_date DATE, heavy_outing DATE)")
    # 301 is unavailable on LEDGER_IX: the house rule prices the placement
    # day's own innings as available, so the stint must strictly straddle
    # the game date (game_date > il_start AND game_date < il_end).
    ledger_day = base + timedelta(days=LEDGER_IX)
    con.execute("CREATE TEMP TABLE il_stints_pitchers AS SELECT "
                f"301 AS batter, DATE '{ledger_day - timedelta(days=1)}' "
                f"AS il_start, DATE '{ledger_day + timedelta(days=1)}' "
                "AS il_end")
    # SP_H goes stale in STALE_IX (gap + stint overlap asserted by hand).
    stale = pd.DataFrame([
        {"game_date": base + timedelta(days=STALE_IX),
         "game_pk": game_pks[STALE_IX],
         "pitcher": 101, "sp_stale": True}])
    con.register("pitcher_stale", stale)
    prior = pd.DataFrame([
        {"season": 2026, "game_date": base + timedelta(days=g),
         "k_pitches": 93.0}
        for g in range(N_GAMES)])
    con.register("bp_shrink_prior", prior)
    return con


def test_derivation_pit_scale_null() -> None:
    con = _build_pp_con()
    features._build_pitchingplus_proxy(con, True)

    app = con.execute("SELECT * FROM pp_appearance").fetchdf()
    sp5 = con.execute("SELECT * FROM pp_sp_5g").fetchdf()

    def _app(pitcher: int, gpk: int) -> float | None:
        v = app[(app.pitcher == pitcher) & (app.game_pk == gpk)]
        assert len(v) == 1
        return v.iloc[0]["pitching_plus_app"]

    def _sp5(pitcher: int, gpk: int) -> float | None:
        v = sp5[(sp5.pitcher == pitcher) & (sp5.game_pk == gpk)]
        assert len(v) == 1
        return v.iloc[0]["sp_pitchingplus_5g"]

    collapse_gpk = 1000 + COLLAPSE_IX
    # Scale: elite reads above 100, poor reads below 100.
    assert _sp5(101, collapse_gpk) > 100.0 > _sp5(202, collapse_gpk)
    # PIT: the collapse is IN that game's appearance but NOT its trailing.
    assert _app(101, collapse_gpk) < 100.0 < _sp5(101, collapse_gpk)
    # Debut (303) and opener (game 1) trailings are NULL.
    assert pd.isna(_sp5(303, 1000 + DEBUT_IX))
    assert pd.isna(_sp5(101, 1000))
    # Cameo (304, 2 pitches) appearance is NULL.
    assert pd.isna(_app(304, 1000 + CAMEO_IX))
    # Staleness: SP_H trailing is NULL in the stale game.
    assert pd.isna(_sp5(101, 1000 + STALE_IX))

    pen = con.execute("SELECT * FROM pp_pen_game").fetchdf()
    g_ledger = pen[pen.game_pk == 1000 + LEDGER_IX].iloc[0]
    # Ledgered 301 excluded from that game's pool: only poor 302 scores.
    assert g_ledger["pen_pitches_scored"] == 15
    assert g_ledger["pen_pp"] < 100.0
    g_cameo = pen[pen.game_pk == 1000 + CAMEO_IX].iloc[0]
    # Cameo 304 contributes no scored pitches in its game either.
    assert g_cameo["pen_pitches_scored"] == 30

    roll = con.execute("SELECT * FROM pp_bullpen_rolling").fetchdf()
    assert roll["bullpen_pitchingplus_10g"].notna().any()
    assert roll["bullpen_pitchingplus_3g"].notna().any()
    con.close()


def _lv_row(day, gpk, pitcher, px, pz, b, s, pvalue):
    return {
        "game_date": pd.Timestamp(day), "game_pk": gpk, "pitcher": pitcher,
        "home_team": "NYY", "away_team": "BOS", "inning_topbot": "Top",
        "pitch_type": "FF", "balls": b, "strikes": s, "stand": "R",
        "delta_pitcher_run_exp": pvalue,
        "sz_top": 3.5, "sz_bot": 1.5,
        "release_speed": 96.0, "spin_rate": 2300.0,
        "pfx_x": 8.0, "pfx_z": 12.0, "release_extension": 6.5,
        "plate_x": px, "plate_z": pz}


def _lv_con(df) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.register("pitches", df)
    con.execute("CREATE TABLE starters (game_pk BIGINT, home_starter_id "
                "BIGINT, away_starter_id BIGINT)")
    con.execute("CREATE TABLE bp_outing (game_date DATE, game_pk BIGINT, "
                "team VARCHAR, pitcher BIGINT, n_pitches BIGINT)")
    con.execute("CREATE TABLE bp_spent (team VARCHAR, pitcher BIGINT, "
                "game_date DATE, heavy_outing DATE)")
    con.execute("CREATE TEMP TABLE pitcher_stale (game_date DATE, "
                "game_pk BIGINT, pitcher BIGINT, sp_stale BOOLEAN)")
    con.execute("CREATE TABLE bp_shrink_prior (season INTEGER, "
                "game_date DATE, k_pitches DOUBLE)")
    return con


def test_locval_pricing() -> None:
    """v4 location mechanics: day 1 prices two cells in one context
    (+0.10 vs -0.10, context mean 0.0); day 2 appearances in each cell
    credit +(300*0.10)/(300+10) / -(300*0.10)/(300+10) (shrink-10 prior
    toward context; cells are singletons here so smoothing is identity).
    NULL-count and NULL-location pitches score zero weight but still
    count in n_pitches."""
    day1, day2 = date(2026, 5, 1), date(2026, 5, 2)
    rows = []
    for i in range(300):  # price pool: one context, two cells
        rows.append(_lv_row(day1, 9001, 900, 0.8, 2.5, 0, 2, 0.10))
        rows.append(_lv_row(day1, 9001, 900, 0.0, 2.5, 0, 2, -0.10))
    for i in range(10):  # pitcher 501 lives in the good cell
        rows.append(_lv_row(day2, 9002, 501, 0.8, 2.5, 0, 2, 0.0))
    for i in range(10):  # pitcher 502 lives in the bad cell
        rows.append(_lv_row(day2, 9002, 502, 0.0, 2.5, 0, 2, 0.0))
    for i in range(5):  # NULL count: unscored, still counted
        rows.append(_lv_row(day2, 9002, 501, 0.8, 2.5, None, None, 0.0))
    for i in range(3):  # NULL location: unscored, still counted
        rows.append(_lv_row(day2, 9002, 501, None, None, 0, 2, 0.0))
    con = _lv_con(pd.DataFrame(rows))
    features._build_pitchingplus_proxy(con, False)
    raw = con.execute("SELECT * FROM pp_appearance_raw").fetchdf()
    good = raw[raw.pitcher == 501].iloc[0]
    bad = raw[raw.pitcher == 502].iloc[0]
    assert good["n_pitches"] == 18 and bad["n_pitches"] == 10
    assert good["loc_val"] > 0.0 > bad["loc_val"]
    assert abs(good["loc_val"] - 30.0 / 310.0) < 1e-9
    assert abs(bad["loc_val"] + 30.0 / 310.0) < 1e-9
    w = con.execute("SELECT SUM(credit_w) AS w, COUNT(*) AS n "
                    "FROM pp_pitch_credit WHERE pitcher = 501").fetchdf()
    assert w.iloc[0]["w"] == 10.0 and w.iloc[0]["n"] == 18
    lg = con.execute("SELECT cell, credit FROM pp_lv_league").fetchdf()
    assert (lg["credit"] > 0).any() and (lg["credit"] < 0).any()
    con.close()


def test_locval_credit_signs() -> None:
    """v4 credit mechanics: elite locations (edge cells, priced high)
    out-credit poor locations (meat cells, priced low) on the full
    synthetic frame once the context floors clear."""
    con = _build_pp_con()
    features._build_pitchingplus_proxy(con, True)
    cred = con.execute("SELECT * FROM pp_pitch_credit").fetchdf()
    assert (cred["credit_w"] > 0).any()
    elite = cred[(cred.pitcher == 101)
                 & (cred.credit.notna())].copy()
    assert len(elite) > 0
    poor = cred[(cred.pitcher == 202)
                & (cred.credit.notna())].copy()
    assert len(poor) > 0
    assert elite["credit"].mean() > poor["credit"].mean()
    lg = con.execute("SELECT * FROM pp_lv_league").fetchdf()
    assert lg["credit"].notna().any()
    con.close()


def test_empty_league_ships_null() -> None:
    """A single-date frame has no prior league row: NULL, never 100."""
    con = duckdb.connect()
    df = _synthetic_pitches()
    one_day = df[df.game_date == df.game_date.min()].copy()
    con.register("pitches", one_day)
    con.execute("CREATE TABLE bp_outing (game_date DATE, game_pk BIGINT, "
                "team VARCHAR, pitcher BIGINT, n_pitches BIGINT)")
    con.execute("CREATE TABLE bp_spent (team VARCHAR, pitcher BIGINT, "
                "game_date DATE, heavy_outing DATE)")
    con.execute("CREATE TEMP TABLE pitcher_stale (game_date DATE, "
                "game_pk BIGINT, pitcher BIGINT, sp_stale BOOLEAN)")
    con.execute("CREATE TABLE bp_shrink_prior (season INTEGER, "
                "game_date DATE, k_pitches DOUBLE)")
    features._build_pitchingplus_proxy(con, False)
    app = con.execute("SELECT * FROM pp_appearance").fetchdf()
    assert app["pitching_plus_app"].isna().all()
    con.close()


def test_location_leg_populates() -> None:
    """v4 location leg: once the appearance league floor clears, raw
    credits z-score into a scoring location_plus (not just the stuff leg
    carrying the composite)."""
    con = _build_pp_con()
    features._build_pitchingplus_proxy(con, True)
    n_loc_scored = con.execute(
        "SELECT COUNT(*) FROM pp_appearance_scored "
        "WHERE location_plus IS NOT NULL").fetchone()[0]
    assert n_loc_scored > 0
    n_no_league = con.execute(
        "SELECT COUNT(*) FROM pp_appearance_raw r "
        "LEFT JOIN pp_league lg ON lg.season = r.season "
        "AND lg.game_date = r.game_date "
        "WHERE lg.lg_mean_locval IS NULL").fetchone()[0]
    # Once the league floor clears, location priors exist for every date.
    assert n_no_league < con.execute(
        "SELECT COUNT(*) FROM pp_appearance_raw").fetchone()[0]
    con.close()


# ── add_diff_features: the 18 trial columns ───────────────────────────────
def _pp_frame(n: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(23)

    def sides(base, lo, hi):
        return {f"{base}_home": np.round(rng.uniform(lo, hi, n), 4),
                f"{base}_away": np.round(rng.uniform(lo, hi, n), 4)}

    d: dict[str, object] = {}
    d.update(sides("sp_pitchingplus_5g", 95.0, 110.0))
    d.update(sides("bullpen_pitchingplus_10g", 95.0, 105.0))
    d.update(sides("bullpen_pitchingplus_3g", 95.0, 105.0))
    d.update(sides("sp_fbvelo_3g", 90.0, 99.0))
    d.update(sides("bullpen_pitches_3d", 20.0, 80.0))
    d.update({
        "home_team": ["NYY"] * n,
        "away_team": ["BOS"] * n,
        "park_wind_factor": [0.8] * n,
    })
    df = pd.DataFrame(d)
    df.loc[2, "sp_pitchingplus_5g_away"] = np.nan  # NULL propagation
    return df


def test_diff_family_formulas() -> None:
    out = features.add_diff_features(_pp_frame())
    h = out["sp_pitchingplus_5g_home"].to_numpy(dtype=float)
    a = out["sp_pitchingplus_5g_away"].to_numpy(dtype=float)
    np.testing.assert_allclose(out["sp_pitchingplus_5g_diff"].to_numpy(),
                               h - a, equal_nan=True)
    q10h = out["bullpen_pitchingplus_10g_home"].to_numpy(dtype=float)
    q10a = out["bullpen_pitchingplus_10g_away"].to_numpy(dtype=float)
    np.testing.assert_allclose(out["bullpen_pitchingplus_10g_diff"].to_numpy(),
                               q10h - q10a, equal_nan=True)
    q3h = out["bullpen_pitchingplus_3g_home"].to_numpy(dtype=float)
    q3a = out["bullpen_pitchingplus_3g_away"].to_numpy(dtype=float)
    np.testing.assert_allclose(out["bullpen_pitchingplus_3g_diff"].to_numpy(),
                               q3h - q3a, equal_nan=True)

    # regression_pp: velo x pitching+ (diff + within-side twins).
    v = (out["sp_fbvelo_3g_home"] - out["sp_fbvelo_3g_away"]).to_numpy()
    np.testing.assert_allclose(
        out["pitcher_regression_indicator_pitchingplus_diff"].to_numpy(),
        v * out["sp_pitchingplus_5g_diff"].to_numpy(), equal_nan=True)
    np.testing.assert_allclose(
        out["pitcher_regression_indicator_pitchingplus_home"].to_numpy(),
        out["sp_fbvelo_3g_home"].to_numpy() * h, equal_nan=True)
    np.testing.assert_allclose(
        out["pitcher_regression_indicator_pitchingplus_away"].to_numpy(),
        out["sp_fbvelo_3g_away"].to_numpy() * a, equal_nan=True)

    # meltdown_pp: workload x pitching+ (diff + within-side twins).
    p = (out["bullpen_pitches_3d_home"]
         - out["bullpen_pitches_3d_away"]).to_numpy()
    np.testing.assert_allclose(
        out["bullpen_meltdown_risk_pitchingplus_diff"].to_numpy(),
        p * out["bullpen_pitchingplus_10g_diff"].to_numpy(), equal_nan=True)
    np.testing.assert_allclose(
        out["bullpen_meltdown_risk_pitchingplus_home"].to_numpy(),
        out["bullpen_pitches_3d_home"].to_numpy() * q10h, equal_nan=True)
    np.testing.assert_allclose(
        out["bullpen_meltdown_risk_pitchingplus_away"].to_numpy(),
        out["bullpen_pitches_3d_away"].to_numpy() * q10a, equal_nan=True)

    # wind_pp: multiplier x pitching+ legs; the holed row stays NULL
    # (outdoor missing observation is never fabricated).
    np.testing.assert_allclose(
        out["wind_advantage_flyball_factor_pitchingplus_diff"].to_numpy(),
        0.8 * (h - a), equal_nan=True)
    np.testing.assert_allclose(
        out["wind_advantage_flyball_factor_pitchingplus_home"].to_numpy(),
        0.8 * h, equal_nan=True)
    assert np.isnan(out["wind_advantage_flyball_factor_pitchingplus_away"].iloc[2])

    # NULL propagation: the holed away leg NULLs every dependent column
    # (meltdown_pp rides the bullpen leg, untouched by this hole).
    for col in ("sp_pitchingplus_5g_diff",
                "pitcher_regression_indicator_pitchingplus_diff",
                "pitcher_regression_indicator_pitchingplus_away",
                "wind_advantage_flyball_factor_pitchingplus_diff",
                "wind_advantage_flyball_factor_pitchingplus_away"):
        assert np.isnan(out[col].iloc[2]), col


def test_dome_forces_pp_wind_zero() -> None:
    """Closed roof (TB) is a KNOWN zero for the whole pp wind triplet —
    even the leg whose pitching+ input is missing (indoor-neutral is a
    policy on the roof state, matching the xFIP leg)."""
    df = _pp_frame()
    df["home_team"] = "TB"
    out = features.add_diff_features(df)
    for col in ("wind_advantage_flyball_factor_pitchingplus_diff",
                "wind_advantage_flyball_factor_pitchingplus_home",
                "wind_advantage_flyball_factor_pitchingplus_away"):
        assert (out[col] == 0.0).all(), col


# ── contracts: trial-only, never served ───────────────────────────────────
def test_trial_registration_contracts() -> None:
    for name in PP_COLS:
        assert name in training.KNOWN_FEATURE_COLS, name
        assert name in RFE_CANDIDATE_COLS, name
        assert name in CANDIDATE_COLS, name
        assert name in _RICH, name
    served = set(training.active_moneyline_feature_cols())
    assert not (set(PP_COLS) & served), \
        set(PP_COLS) & served


def test_metadata_no_orphans() -> None:
    _, warnings = build_features_metadata()
    assert not [w for w in warnings if "pitchingplus" in w or "_pp_" in w]


# ── weather fill path ─────────────────────────────────────────────────────
def test_weather_pp_fill() -> None:
    # NOTE: the lookup falls back to the frame index, so weather keys
    # must not collide with it (same discipline as the incumbent legs).
    df = pd.DataFrame({
        "game_pk": [101, 102, 103],
        "dome_is_neutral": [0, 1, 0],
        "sp_pitchingplus_5g_diff": [4.0, 4.0, np.nan],
        "sp_pitchingplus_5g_home": [104.0, 104.0, 104.0],
        "sp_pitchingplus_5g_away": [100.0, 100.0, np.nan],
    })
    wx = {101: {"available": True, "wind_multiplier": 1.5,
                "air_density": 1.2}}
    out = weather_mod.apply_weather_features(df, wx)
    assert out["wind_advantage_flyball_factor_pitchingplus_diff"].iloc[0] == 1.5 * 4.0
    assert out["wind_advantage_flyball_factor_pitchingplus_home"].iloc[0] == 1.5 * 104.0
    assert out["wind_advantage_flyball_factor_pitchingplus_away"].iloc[0] == 1.5 * 100.0
    # Dome without fetch: valid 0 where the input exists.
    assert out["wind_advantage_flyball_factor_pitchingplus_diff"].iloc[1] == 0.0
    assert out["wind_advantage_flyball_factor_pitchingplus_home"].iloc[1] == 0.0
    assert out["wind_advantage_flyball_factor_pitchingplus_away"].iloc[1] == 0.0
    # Outdoor without fetch: NULL preserved, never fabricated.
    assert np.isnan(out["wind_advantage_flyball_factor_pitchingplus_diff"].iloc[2])
    assert np.isnan(out["wind_advantage_flyball_factor_pitchingplus_away"].iloc[2])


# ── v5 pitch grain + primary differentials + per-type disclosure ─────────
def _pp_mixed_frame() -> pd.DataFrame:
    """Two-date frame: pitcher 701 mixes FF (primary, 8/game) with SL
    (4/game); pitcher 702 throws SL only (no primary → NULL differentials)."""
    rows = []
    for day, gpk in ((date(2026, 4, 1), 5001), (date(2026, 4, 2), 5002)):
        for i in range(12):
            rows.append({
                "game_date": pd.Timestamp(day), "game_pk": gpk,
                "pitcher": 701,
                "pitch_type": "FF" if i < 8 else "SL",
                "release_speed": 95.0, "spin_rate": 2200.0,
                "pfx_x": 6.0, "pfx_z": 10.0, "release_extension": 6.4,
                "balls": 0, "strikes": 0, "stand": "R",
                "plate_x": 0.5, "plate_z": 2.5,
                "sz_top": 3.5, "sz_bot": 1.5,
                "delta_pitcher_run_exp": 0.0})
        for i in range(6):
            rows.append({
                "game_date": pd.Timestamp(day), "game_pk": gpk,
                "pitcher": 702, "pitch_type": "SL",
                "release_speed": 84.0, "spin_rate": 2400.0,
                "pfx_x": -8.0, "pfx_z": 4.0, "release_extension": 6.0,
                "balls": 1, "strikes": 1, "stand": "L",
                "plate_x": -0.5, "plate_z": 2.0,
                "sz_top": 3.5, "sz_bot": 1.5,
                "delta_pitcher_run_exp": 0.0})
    return pd.DataFrame(rows)


def _pp_mixed_con() -> object:
    con = duckdb.connect()
    con.register("pitches", _pp_mixed_frame())
    con.execute("CREATE TABLE bp_outing (game_date DATE, game_pk BIGINT, "
                "team VARCHAR, pitcher BIGINT, n_pitches BIGINT)")
    con.execute("CREATE TABLE bp_spent (team VARCHAR, pitcher BIGINT, "
                "game_date DATE, heavy_outing DATE)")
    con.execute("CREATE TEMP TABLE pitcher_stale (game_date DATE, "
                "game_pk BIGINT, pitcher BIGINT, sp_stale BOOLEAN)")
    con.execute("CREATE TABLE bp_shrink_prior (season INTEGER, "
                "game_date DATE, k_pitches DOUBLE)")
    features._build_pitchingplus_proxy(con, False)
    return con


def test_pitch_grain_split_break_and_release() -> None:
    """v5 grain: one stuff row per pitch; vertical/horizontal break are
    separate traits (no move_mag residue); release columns exist."""
    con = _pp_mixed_con()
    n_pitch = con.execute(
        "SELECT COUNT(*) FROM pitches WHERE pitcher IS NOT NULL").fetchone()[0]
    n_rows = con.execute("SELECT COUNT(*) FROM pp_pitch_stuff").fetchone()[0]
    assert n_rows == n_pitch == 36, (n_rows, n_pitch)
    cols = {r[0] for r in con.execute("DESCRIBE pp_pitch_stuff").fetchall()}
    assert {"velo", "vbreak", "hbreak", "spin", "ext", "relx",
            "relz", "dvelo", "dhb", "dvb"} <= cols, cols
    assert "move_mag" not in cols, "v4 appearance trait must be gone"
    scored = con.execute(
        "SELECT COUNT(*), COUNT(stuff_pitch) FROM pp_pitch_stuff_scored"
        ).fetchone()
    assert scored[0] == n_pitch, "one scored row per pitch"
    con.close()


def test_primary_differentials_secondaries_only() -> None:
    """FG usage-defined primary: 701's SL pitches carry dvelo/dhb/dvb
    off his FF means; his FF pitches and 702's SL-only outings (no
    primary) carry NULL — raw traits still score."""
    con = _pp_mixed_con()
    prim = con.execute(
        "SELECT pitcher, primary_fb FROM pp_app_primary_fb").fetchdf()
    assert (prim[prim.pitcher == 701]["primary_fb"] == "FF").all(), prim
    assert 702 not in set(prim["pitcher"]), "no fastball → no reference"
    got = con.execute(
        "SELECT pitcher, pitch_type, COUNT(*) AS n, "
        "COUNT(dvelo) AS nd, COUNT(dhb) AS nh, COUNT(dvb) AS nv "
        "FROM pp_pitch_stuff GROUP BY 1, 2").fetchdf()
    sl701 = got[(got.pitcher == 701) & (got.pitch_type == "SL")].iloc[0]
    assert sl701["n"] == 8 and sl701["nd"] == 8, sl701.to_dict()
    assert sl701["nh"] == 8 and sl701["nv"] == 8
    ff701 = got[(got.pitcher == 701) & (got.pitch_type == "FF")].iloc[0]
    assert ff701["nd"] == 0, "primary pitches are the reference"
    sl702 = got[got.pitcher == 702].iloc[0]
    assert sl702["nd"] == 0, "no primary → NULL differentials"
    con.close()


def test_per_type_disclosure_structure() -> None:
    """The FG-table disclosure has a pitch_type to group by; the per-type
    aggregation runs with correct counts even when the league floor
    leaves values NULL on a tiny frame."""
    con = _pp_mixed_con()
    cols = {r[0] for r in
            con.execute("DESCRIBE pp_pitch_stuff_scored").fetchall()}
    assert "pitch_type" in cols and "stuff_pitch" in cols
    rows = con.execute(
        "SELECT COALESCE(pitch_type, 'unknown') AS pt, COUNT(*) AS n, "
        "AVG(stuff_pitch) AS mean FROM pp_pitch_stuff_scored "
        "GROUP BY 1 ORDER BY 2 DESC").fetchdf()
    assert set(rows["pt"]) == {"FF", "SL"}
    assert rows.set_index("pt").loc["FF", "n"] == 16
    assert rows.set_index("pt").loc["SL", "n"] == 20
    con.close()


def test_sp_trailing_weights_by_innings_not_pitches() -> None:
    """The per-game quality average weights by recorded IP only, without
    shrinkage. Missing innings must not be fabricated from pitch counts."""
    con = duckdb.connect()
    con.execute("CREATE TABLE pp_appearance (game_date DATE, season INTEGER, "
                "game_pk BIGINT, pitcher BIGINT, n_pitches BIGINT, "
                "app_ip DOUBLE, pitching_plus_app DOUBLE)")
    con.execute("CREATE TEMP TABLE pitcher_stale (game_date DATE, "
                "game_pk BIGINT, pitcher BIGINT, sp_stale BOOLEAN)")
    con.execute("INSERT INTO pp_appearance VALUES "
                "('2026-04-01', 2026, 6001, 801, 15, 1.0, 110.0), "
                "('2026-04-07', 2026, 6002, 801, 140, 9.0, 90.0), "
                "('2026-04-13', 2026, 6003, 801, 80, 5.0, 100.0), "
                "('2026-04-01', 2026, 6001, 802, 155, NULL, 100.0), "
                "('2026-04-07', 2026, 6002, 802, 80, 5.0, 100.0)")
    con.execute(features._pp_sp_5g_sql())
    got = {r[0]: r[1] for r in con.execute(
        "SELECT pitcher, sp_pitchingplus_5g FROM pp_sp_5g "
        "WHERE game_pk = 6003 OR (pitcher = 802 AND game_pk = 6002)"
        ).fetchall()}
    # 801: (110*1 + 90*9)/(1+9) = 92, with no pseudo-IP at 100.
    assert abs(got[801] - 92.0) < 1e-9, got
    # 802's only prior appearance has no recorded IP: do not infer it.
    assert got[802] is None, got
    con.close()


def _learned_frame() -> pd.DataFrame:
    """Three causal origins, with a year boundary and an offseason gap."""
    rng = np.random.default_rng(71)
    days = np.concatenate([
        np.repeat(pd.date_range('2024-04-01', periods=120).to_numpy(), 150),
        np.repeat(pd.date_range('2024-09-01', periods=30).to_numpy(), 160),
        np.repeat(pd.date_range('2025-03-01', periods=10).to_numpy(), 100),
        np.repeat(pd.date_range('2025-04-01', periods=10).to_numpy(), 100)])
    n = len(days)
    velo = rng.uniform(85, 100, n)
    px = rng.uniform(-1.8, 1.8, n)
    z = rng.uniform(-.2, 1.2, n)
    return pd.DataFrame({
        'game_date': days, 'game_pk': np.arange(n), 'pitcher': np.arange(n) % 40,
        'game_type': 'R',
        'pitch_type': np.where(np.arange(n) % 2, 'FF', 'SL'),
        'pitch_code': np.arange(n) % 2, 'throw_hand': 1., 'batter_hand': np.arange(n) % 2,
        'release_speed': velo, 'spin': rng.uniform(1900, 2600, n),
        'arm_hb': rng.uniform(-1, 1, n), 'pfx_z': rng.uniform(0, 1.5, n),
        'arm_relx': 2., 'release_pos_z': 6., 'release_extension': 6.5,
        'dvelo': np.nan, 'arm_dhb': np.nan, 'dvb': np.nan, 'axis_deviation': np.nan,
        'balls': np.arange(n) % 4, 'strikes': np.arange(n) % 3,
        'arm_plate_x': px, 'zone_z': z,
        'delta_pitcher_run_exp': .005 * (velo - 92.) + .08 * (abs(px) < .8)
                                  - .04 * abs(z - .5) + rng.normal(0, .015, n)})


def test_learned_models_temporal_isolation_and_scale() -> None:
    f = _learned_frame()
    baseline, diag = features._pp_fit_pitch_models(f)
    # No training before July 2024; no fabricated warmup average.
    assert baseline[pd.to_datetime(f.game_date) < '2024-07-01'].pitching_plus_app.isna().all()
    scored = pd.to_datetime(f.game_date) >= '2025-01-01'
    assert baseline.loc[scored, ['stuff_plus', 'location_plus', 'pitching_plus_app']].notna().all().all()
    fitted = [d for d in diag if d.get('status') == 'fitted']
    # January uses September calibration, rather than dropping spring's grades.
    january = [d for d in fitted if d['origin'] == '2025-01-01']
    assert len(january) == 3
    assert all(d['calibration_last_date'] == '2024-09-30' for d in january)
    for d in fitted:
        assert d['train_last_date'] < d['calibration_last_date'] < d['origin']
        assert d['scale_sd'] > 0
        assert d['test_mse'] < d['test_constant_mse']
    # Outcomes anywhere in a scored quarter cannot change its own grades;
    # later-quarter training may change, but Q1 must be byte-stable.
    changed = f.copy()
    changed.loc[scored, 'delta_pitcher_run_exp'] = 999.
    changed['description'] = 'hit_into_play'
    changed['woba_value'] = 10000.
    changed['estimated_woba_using_speedangle'] = 10000.
    # With no valid TRAIN BIP mapping, future contact cannot invent one.
    # (Separate target test covers a populated train mapping.)
    changed.loc[scored, 'pitcher'] += 10000  # IDs are not learned inputs.
    changed.loc[scored, 'game_pk'] += 100000
    got, _ = features._pp_fit_pitch_models(changed)
    q1 = pd.to_datetime(f.game_date).between('2025-01-01', '2025-03-31')
    for col in ['stuff_plus', 'location_plus', 'pitching_plus_app']:
        np.testing.assert_array_equal(baseline.loc[q1, col], got.loc[q1, col])
    # Component separation: scoring locations cannot affect Stuff+;
    # scoring physical traits cannot affect Location+ at a frozen origin.
    changed_inputs = f.copy()
    changed_inputs.loc[q1, 'arm_plate_x'] += .4
    changed_inputs.loc[q1, 'zone_z'] += .3
    changed_inputs.loc[q1, 'release_speed'] += 15.
    # Change one component at a time so each assertion isolates its input set.
    location_only = f.copy()
    location_only.loc[q1, ['arm_plate_x', 'zone_z']] = changed_inputs.loc[q1, ['arm_plate_x', 'zone_z']]
    location_result, _ = features._pp_fit_pitch_models(location_only)
    physical_only = f.copy()
    physical_only.loc[q1, 'release_speed'] = changed_inputs.loc[q1, 'release_speed']
    physical_result, _ = features._pp_fit_pitch_models(physical_only)
    np.testing.assert_array_equal(baseline.loc[q1, 'stuff_plus'], location_result.loc[q1, 'stuff_plus'])
    np.testing.assert_array_equal(baseline.loc[q1, 'location_plus'], physical_result.loc[q1, 'location_plus'])
    # Overall quality is a separate nonlinear model, not a half-and-half proxy.
    assert not np.allclose(baseline.loc[scored, 'pitching_plus_app'],
                           baseline.loc[scored, ['stuff_plus', 'location_plus']].mean(axis=1))
    # Physical-only grades are directionally correct on the known synthetic RV.
    assert baseline.loc[q1, 'stuff_plus'].corr(f.loc[q1, 'release_speed']) > .9


def test_learned_missing_tracking_and_target_discipline() -> None:
    f = _learned_frame()
    ix = f.index[pd.to_datetime(f.game_date) >= '2025-01-01'][:3]
    f.loc[ix[0], 'release_speed'] = np.nan
    f.loc[ix[1], 'zone_z'] = np.nan
    f.loc[ix[2], 'delta_pitcher_run_exp'] = np.nan
    out, _ = features._pp_fit_pitch_models(f)
    assert pd.isna(out.loc[ix[0], 'pitching_plus_app'])
    assert pd.isna(out.loc[ix[1], 'pitching_plus_app'])
    # A pitch can be graded without an observed outcome.
    assert np.isfinite(out.loc[ix[2], 'pitching_plus_app'])
    assert np.isfinite(out.loc[ix[0], 'location_plus'])
    assert np.isfinite(out.loc[ix[1], 'stuff_plus'])


def test_learned_builder_ip_staff_availability_and_calendar() -> None:
    from unittest.mock import patch
    days = pd.date_range('2025-04-01', periods=12)
    pitches, stats, outings = [], [], []
    for i, day in enumerate(days):
        # Game six has no relief at all, but MUST occupy a team-game slot.
        arms = [101] if i == 5 else [101, 201, 202]
        for arm in arms:
            for pitch in range(3):
                pitches.append({'game_date': day, 'game_pk': i + 1, 'pitcher': arm,
                                'pitch_type': 'FF', 'home_team': 'NYY', 'away_team': 'BOS'})
            ip = 5. if arm == 101 else 1. if arm == 201 else 3.
            stats.append({'game_pk': i + 1, 'pitcher': arm, 'ip': ip})
            if arm != 101:
                outings.append({'game_date': day, 'game_pk': i + 1, 'pitcher': arm,
                                'team': 'NYY', 'n_pitches': 10 if arm == 201 else 30})
    c = duckdb.connect()
    c.register('pitches', pd.DataFrame(pitches))
    c.register('pitcher_game_stats', pd.DataFrame(stats))
    c.register('bp_outing', pd.DataFrame(outings))
    c.execute('CREATE TABLE pitcher_stale (game_date DATE, game_pk BIGINT, pitcher BIGINT, sp_stale BOOLEAN)')
    c.execute("CREATE TABLE il_stints_pitchers AS SELECT 201 AS batter, DATE '2025-04-03' AS il_start, DATE '2025-04-07' AS il_end UNION ALL SELECT 202, DATE '2025-04-10', DATE '2025-04-15'")

    def scorer(f):
        s = f[['game_date', 'game_pk', 'pitcher', 'pitch_type']].copy()
        for name in ['stuff_plus', 'location_plus', 'pitching_plus_app']:
            s[name] = s.pitcher.map({101: 105., 201: 110., 202: 90.})
        return s, []

    with patch.object(features, '_pp_fit_pitch_models', side_effect=scorer):
        features._build_pitchingplus_features(c, True)
    got = c.execute("SELECT * FROM pp_bullpen_rolling WHERE team='NYY' ORDER BY game_date").fetchdf()
    # On game six the pen doesn't pitch, but its PRIOR quality still exists.
    assert got.loc[5, 'bullpen_pitchingplus_3g'] == 90.
    # Game ten: returned 201's historical innings rejoin the available pool.
    assert got.loc[9, 'bullpen_pitchingplus_10g'] == 95.  # (110*1+90*3)/4
    # Game eleven: 202 is now IL; its older healthy innings must be excluded.
    assert got.loc[10, 'bullpen_pitchingplus_10g'] == 110.
    assert got.loc[10, 'bullpen_pitchingplus_3g'] == 110.
    # An arm with a heavy outing is excluded as of target, then returns.
    c.execute('DROP TABLE pp_bullpen_rolling')
    heavy = pd.DataFrame(outings)
    heavy.loc[(heavy.game_pk == 8) & (heavy.pitcher == 201), 'n_pitches'] = 40
    c.unregister('bp_outing')
    c.register('bp_outing', heavy)
    c.execute(features._pp_bullpen_sql(True))
    rows = c.execute("SELECT * FROM pp_bullpen_rolling WHERE team='NYY' ORDER BY game_date").fetchdf()
    assert rows.loc[9, 'bullpen_pitchingplus_10g'] == 90.
    assert rows.loc[10, 'bullpen_pitchingplus_10g'] == 110.
    assert not got.duplicated(['game_pk', 'team']).any()
    c.close()


def test_bullpen_three_games_not_three_relief_appearances() -> None:
    c = duckdb.connect()
    c.execute("CREATE TABLE pitches AS SELECT DATE '2025-04-01' + CAST(i AS INTEGER) AS game_date, i+1 AS game_pk, 'NYY' AS home_team, 'BOS' AS away_team FROM range(5) t(i)")
    c.execute("CREATE TABLE bp_outing AS SELECT DATE '2025-04-01' AS game_date, 1 AS game_pk, 'NYY' AS team, 1 AS pitcher, 10 AS n_pitches")
    c.execute('CREATE TABLE pp_appearance AS SELECT 1 AS game_pk, 1 AS pitcher, 1.0 AS app_ip, 120.0 AS pitching_plus_app')
    c.execute(features._pp_bullpen_sql())
    r = c.execute("SELECT bullpen_pitchingplus_3g, bullpen_pitchingplus_10g FROM pp_bullpen_rolling WHERE game_pk=5 AND team='NYY'").fetchone()
    assert r == (None, 120.), r
    c.close()


def test_sp_same_date_and_five_appearance_boundaries() -> None:
    c = duckdb.connect()
    c.execute("CREATE TABLE pp_appearance AS SELECT DATE '2025-04-01' + CAST(i AS INTEGER) AS game_date, i+1 AS game_pk, 1 AS pitcher, 1.0 AS app_ip, CASE WHEN i=0 THEN 200.0 ELSE 100.0 END AS pitching_plus_app FROM range(8) t(i)")
    c.execute("INSERT INTO pp_appearance VALUES ('2025-04-08', 80, 1, 1., 900.)")
    c.execute('CREATE TABLE pitcher_stale (game_date DATE, game_pk BIGINT, pitcher BIGINT, sp_stale BOOLEAN)')
    c.execute(features._pp_sp_5g_sql())
    r = c.execute("SELECT sp_pitchingplus_5g FROM pp_sp_5g WHERE game_date='2025-04-08'").fetchall()
    assert r == [(100.,), (100.,)], r
    c.close()


def test_learned_inputs_primary_differentials_and_deterministic_order() -> None:
    f = _pp_mixed_frame()
    f['at_bat_number'] = np.arange(len(f)) // 3
    f['pitch_number'] = np.arange(len(f)) % 3 + 1
    f['p_throws'] = 'R'
    c = duckdb.connect()
    c.register('pitches', f.sample(frac=1, random_state=7))
    got = features._pp_pitch_inputs(c)
    c.unregister('pitches')
    c.register('pitches', f.sample(frac=1, random_state=41))
    again = features._pp_pitch_inputs(c)
    pd.testing.assert_frame_equal(got, again)
    secondary = got[(got.pitcher == 701) & got.pitch_type.eq('SL')]
    assert len(secondary) == 8
    assert (secondary.primary_fb == 'FF').all()
    assert secondary[['dvelo', 'arm_dhb', 'dvb']].notna().all().all()
    assert got.loc[got.pitcher.eq(702), ['dvelo', 'arm_dhb', 'dvb']].isna().all().all()
    assert got.loc[got.pitch_type.eq('FF'), 'dvelo'].isna().all()
    np.testing.assert_allclose(got.arm_hb, -got.pfx_x)
    np.testing.assert_allclose(got.zone_z, (got.plate_z - got.sz_bot) / (got.sz_top - got.sz_bot))
    c.close()


def test_expected_contact_target_train_only_mapping_and_missingness() -> None:
    n = 720
    count = np.arange(n) % 12
    woba = (np.arange(n) // 12) % 3
    raw = .3 + .002 * count - .8 * woba
    f = pd.DataFrame({'balls': count // 3, 'strikes': count % 3,
                      'description': 'hit_into_play', 'woba_value': woba.astype(float),
                      'estimated_woba_using_speedangle': .4,
                      'delta_pitcher_run_exp': raw})
    fit = np.arange(n) < 600
    got, info = features._pp_expected_contact_target(f, fit)
    assert info['status'] == 'expected_contact'
    np.testing.assert_allclose(got, .3 + .002 * count - .8 * .4, atol=1e-12)
    # Future realized contact outcomes cannot alter the learned pricing.
    changed = f.copy()
    changed.loc[~fit, 'woba_value'] = 200.
    changed.loc[~fit, 'delta_pitcher_run_exp'] = -500.
    future, future_info = features._pp_expected_contact_target(changed, fit)
    np.testing.assert_array_equal(got, future)
    assert info == future_info
    # Missing xwOBA falls back to observed RV; missing observed RV stays NULL.
    f.loc[700, 'estimated_woba_using_speedangle'] = np.nan
    f.loc[701, 'delta_pitcher_run_exp'] = np.nan
    f.loc[702, 'description'] = 'called_strike'
    missing, _ = features._pp_expected_contact_target(f, fit)
    assert missing[700] == raw[700]
    assert pd.isna(missing[701])
    assert missing[702] == raw[702]


def test_contact_labels_never_enter_scored_quarter_inference() -> None:
    f = _learned_frame()
    f['description'] = 'hit_into_play'
    f['woba_value'] = (.35 - f.delta_pitcher_run_exp) / .8
    f['estimated_woba_using_speedangle'] = f.woba_value
    # This creates a populated, exactly known training contact mapping.
    first, diagnostics = features._pp_fit_pitch_models(f)
    q1 = pd.to_datetime(f.game_date).between('2025-01-01', '2025-03-31')
    changed = f.copy()
    changed.loc[q1, ['woba_value', 'estimated_woba_using_speedangle',
                     'delta_pitcher_run_exp']] = 500.
    second, _ = features._pp_fit_pitch_models(changed)
    for name in ['stuff_plus', 'location_plus', 'pitching_plus_app']:
        np.testing.assert_array_equal(first.loc[q1, name], second.loc[q1, name])
        assert first.loc[q1, name].notna().all()
    assert all(d['target_method']['status'] == 'expected_contact'
               for d in diagnostics if d['status'] == 'fitted')


def test_reference_scale_is_prior_pitcher_population_not_pitch_sd() -> None:
    # Equal weight for each qualified pitcher, even with different volume.
    pitchers = np.repeat(np.arange(40), np.arange(100, 140))
    raw = pitchers.astype(float) * .001
    # Add within-arm pitch-location variation that must NOT define the scale.
    for arm in range(40):
        mask = pitchers == arm
        noise = np.linspace(-.1, .1, mask.sum())
        raw[mask] += noise - noise.mean()
    types = np.repeat('FF', len(raw))
    mu, sd, _, n = features._pp_grade_reference(raw, pitchers, types, False)
    grades = pd.DataFrame({'arm': pitchers, 'grade': 100 + 10 * (raw - mu) / sd}).groupby('arm').grade.mean()
    assert n == 40
    np.testing.assert_allclose(grades.mean(), 100.)
    np.testing.assert_allclose(grades.std(ddof=1), 10.)
    assert sd < raw.std()  # intentional pitcher-reference vs pitch dispersion
    # No trustworthy prior reference -> warmup, not an invented denominator.
    a, b, _, n = features._pp_grade_reference(raw[:200], pitchers[:200], types[:200], False)
    assert np.isnan(a) and np.isnan(b) and n < 30


def test_axis_proxy_uses_catcher_coordinate_orientation() -> None:
    f = _pp_mixed_frame().iloc[:3].copy()
    f['pfx_x'] = [-1., 1., 0.]
    f['pfx_z'] = [1., 1., 0.]
    f['spin_axis'] = [225., 135., 359.]
    f['p_throws'] = ['R', 'L', 'R']
    c = duckdb.connect()
    c.register('pitches', f)
    got = features._pp_pitch_inputs(c)
    assert (got.axis_deviation.iloc[:2] == 0.).all()
    assert pd.isna(got.axis_deviation.iloc[2])
    c.close()


def test_promotion_gate_rejects_rfe_adoption_and_existing_state() -> None:
    from unittest.mock import patch
    import feature_selection as fs
    import pytest
    with pytest.raises(ValueError, match='Pitching\\+ promotion blocked'):
        fs.write_state(PP_COLS, {})
    cols = list(training.MONEYLINE_FEATURE_COLS) + PP_COLS
    with patch.object(fs, 'load_state', return_value={'cols': cols}):
        assert fs.apply_adopted_subset()['applied'] is False


if __name__ == "__main__":
    test_derivation_pit_scale_null()
    print("derivation PIT/scale/NULL ok")
    test_empty_league_ships_null()
    print("empty league NULL ok")
    test_locval_pricing()
    print("locval pricing ok")
    test_locval_credit_signs()
    print("locval credit signs ok")
    test_location_leg_populates()
    print("location leg populates ok")
    test_diff_family_formulas()
    print("diff family formulas ok")
    test_dome_forces_pp_wind_zero()
    print("dome pp wind zero ok")
    test_trial_registration_contracts()
    print("trial registration contracts ok")
    test_metadata_no_orphans()
    print("metadata orphans ok")
    test_weather_pp_fill()
    print("weather pp fill ok")
    test_pitch_grain_split_break_and_release()
    print("pitch grain split break and release ok")
    test_primary_differentials_secondaries_only()
    print("primary differentials secondaries only ok")
    test_per_type_disclosure_structure()
    print("per type disclosure structure ok")
    test_sp_trailing_weights_by_innings_not_pitches()
    print("sp trailing weights by innings not pitches ok")
    test_learned_models_temporal_isolation_and_scale()
    test_learned_missing_tracking_and_target_discipline()
    test_learned_builder_ip_staff_availability_and_calendar()
    test_bullpen_three_games_not_three_relief_appearances()
    test_sp_same_date_and_five_appearance_boundaries()
    test_promotion_gate_rejects_rfe_adoption_and_existing_state()
    test_learned_inputs_primary_differentials_and_deterministic_order()
    test_expected_contact_target_train_only_mapping_and_missingness()
    test_reference_scale_is_prior_pitcher_population_not_pitch_sd()
    test_contact_labels_never_enter_scored_quarter_inference()
    test_axis_proxy_uses_catcher_coordinate_orientation()
    print("ALL PITCHING+ TRIAL TESTS PASSED")
