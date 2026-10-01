"""Pin the exp2 zero-prior SP fallback (2026-10-01 coverage remediation).

Run-log gap: exp2_cat_k_offspeed_diff / exp2_cat_xwoba_offspeed_diff served
57% measured (baseline 64%). Attribution over the 2026-09-30 frame: 100% of
gap rows have at least one starter with ZERO prior tracked PAs of that
category this season (team-side gaps: 1.4%). Under the old arithmetic those
SP cells shipped NaN and rode per-fold median imputation — fabricating
"league-average offspeed" for the matchup product on ~43% of offspeed rows.

The fallback (mirrors the 6b3b3e2 shrinkage philosophy at the feature layer):
  * rate (sp_k_pct_cat / sp_xwoba_cat / sp_k_pct_fb_vs) with zero prior PAs
    -> the strictly-prior league rate for that date, so the SP deviation
    factor is exactly 0 = no information (an observation gap is never
    evidence of ability);
  * usage (sp_usage_cat) with zero prior PAs -> 0.0, a KNOWN ZERO of tracked
    usage (the user's intuition, adopted);
  * day-1 dates (league prior itself empty) stay NULL via 0 * NaN
    propagation;
  * team-side gaps still propagate as NaN (never fabricate the opponent).

These tests execute the emitted SQL templates (features._EXP2_SP_CAT_PRELIM_SQL
/ _EXP2_SP_CAT_SQL / _EXP2_SP_FBHAND_SQL) on tiny in-memory DuckDB inputs —
no 2.2M-pitch rebuild required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import features  # noqa: E402


def _con_with(daily: pd.DataFrame, league: pd.DataFrame,
              tot_daily: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """Register the exp2_sp_cat SQL's input tables with fabricated rows,
    including the two cumulative windows exactly as production builds them."""
    con = duckdb.connect(database=":memory:")
    con.register("exp2_sp_cat_daily", daily)
    con.register("exp2_league_cat", league)
    con.register("exp2_sp_cat_tot_daily", tot_daily)
    con.execute("""
        CREATE TABLE exp2_sp_cat_cum AS
        SELECT *,
            SUM(k_n) OVER w AS k_thru,
            SUM(pa_n) OVER w AS pa_thru,
            SUM(xwoba_num) OVER w AS xwo_thru,
            SUM(xwoba_n) OVER w AS xwon_thru
        FROM exp2_sp_cat_daily
        WINDOW w AS (PARTITION BY pitcher, season, pitch_cat ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)
    con.execute("""
        CREATE TABLE exp2_sp_cat_tot_cum AS
        SELECT *, SUM(pa_n) OVER w AS pa_thru
        FROM exp2_sp_cat_tot_daily
        WINDOW w AS (PARTITION BY pitcher, season ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)
    return con


def _run_sp_cat(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    con.execute(features._EXP2_SP_CAT_PRELIM_SQL)
    con.execute(features._EXP2_SP_CAT_SQL)
    return con.execute(
        "SELECT * FROM exp2_sp_cat "
        "ORDER BY game_date, pitcher, pitch_cat").df()


# P: pitcher 7 pitched once on 2026-04-02 (offspeed PAs only); the league
#    published a prior on 2026-04-01. On 2026-04-05 (target date) his
#    offspeed cum exists, but breaking/fastball have no rows at all.
# P: pitcher 7 started twice — 2026-04-02 (offspeed PAs only) and
#    2026-04-05 (fastball PAs only). The league table publishes one LAG-
#    shifted row per date-with-category-PAs: the value AT date D is the
#    cumulative through D-1 (same convention as the served league join).
#    Prelim driver rows exist per (date pitched, pitcher) x 3 categories.
_DAILY = pd.DataFrame({
    "game_date": pd.to_datetime(["2026-04-02", "2026-04-05"]),
    "pitcher": [7, 7],
    "season": [2026, 2026],
    "pitch_cat": ["offspeed", "fastball"],
    "k_n": [2.0, 1.0],
    "pa_n": [10.0, 4.0],
    "xwoba_num": [3.0, 1.0],
    "xwoba_n": [10.0, 4.0],
})

_LEAGUE = pd.DataFrame({
    "game_date": pd.to_datetime(["2026-04-01"] * 3),
    "pitch_cat": ["fastball", "breaking", "offspeed"],
    "league_k_pct_cat": [0.225, 0.240, 0.205],
    "league_xwoba_cat": [0.330, 0.315, 0.300],
})

_TOT_DAILY = pd.DataFrame({
    "game_date": pd.to_datetime(["2026-04-02", "2026-04-05"]),
    "pitcher": [7, 7],
    "season": [2026, 2026],
    "pa_n": [10.0, 4.0],
})


def test_zero_prior_rate_falls_back_to_league():
    con = _con_with(_DAILY, _LEAGUE, _TOT_DAILY)
    out = _run_sp_cat(con)
    row = out[(out.pitcher == 7) & (out.pitch_cat == "breaking")
              & (out.game_date == pd.Timestamp("2026-04-05"))].iloc[0]
    # zero prior breaking PAs -> strictly-prior league rate (deviation 0)
    assert row.sp_k_pct_cat == pytest.approx(0.240)
    assert row.sp_xwoba_cat == pytest.approx(0.315)


def test_zero_prior_usage_ships_known_zero():
    con = _con_with(_DAILY, _LEAGUE, _TOT_DAILY)
    out = _run_sp_cat(con)
    row = out[(out.pitcher == 7) & (out.pitch_cat == "breaking")
              & (out.game_date == pd.Timestamp("2026-04-05"))].iloc[0]
    assert row.sp_usage_cat == pytest.approx(0.0)
    # no-edge collapse: the exp2 side product is usage * dev * dev = 0
    assert row.sp_usage_cat * (row.sp_k_pct_cat - 0.240) \
        * (row.sp_k_pct_cat - 0.240) == pytest.approx(0.0)


def test_known_prior_cell_is_untouched():
    con = _con_with(_DAILY, _LEAGUE, _TOT_DAILY)
    out = _run_sp_cat(con)
    row = out[(out.pitcher == 7) & (out.pitch_cat == "offspeed")
              & (out.game_date == pd.Timestamp("2026-04-05"))].iloc[0]
    assert row.sp_k_pct_cat == pytest.approx(0.2)   # 2/10, personal
    assert row.sp_xwoba_cat == pytest.approx(0.3)   # 3/10, personal
    assert row.sp_usage_cat == pytest.approx(1.0)   # 10/10 tracked PAs


def test_day_one_no_league_prior_stays_null():
    league_no_rows = _LEAGUE[_LEAGUE.game_date > pd.Timestamp("2026-04-09")]
    con = _con_with(_DAILY, league_no_rows, _TOT_DAILY)
    out = _run_sp_cat(con)
    row = out[(out.pitcher == 7) & (out.pitch_cat == "breaking")
              & (out.game_date == pd.Timestamp("2026-04-05"))].iloc[0]
    assert pd.isna(row.sp_k_pct_cat) and pd.isna(row.sp_xwoba_cat)
    # usage still collapses to its known zero; the rate side carries the
    # NULL that keeps the served diff NaN on day-1 rows
    assert row.sp_usage_cat == pytest.approx(0.0)


def test_league_prior_is_strictly_prior_via_lag_shifted_rows():
    """The league table's value AT date D already excludes D (LAG-shifted
    publication), so the ASOF '>=' join is exactly strictly-prior: the
    freshest row at-or-before the game date wins, and gap dates get the
    latest published prior instead of going stale-NULL."""
    league_later = pd.concat([_LEAGUE, pd.DataFrame({
        "game_date": pd.to_datetime(["2026-04-05"] * 3),
        "pitch_cat": ["fastball", "breaking", "offspeed"],
        "league_k_pct_cat": [0.900, 0.910, 0.920],
        "league_xwoba_cat": [0.800, 0.810, 0.820],
    })], ignore_index=True)
    con = _con_with(_DAILY, league_later, _TOT_DAILY)
    out = _run_sp_cat(con)
    row = out[(out.pitcher == 7) & (out.pitch_cat == "breaking")
              & (out.game_date == pd.Timestamp("2026-04-05"))].iloc[0]
    # zero prior breaking -> the 04-05 league row, whose VALUE is the
    # through-04-04 cumulative (never the game's own league PAs)
    assert row.sp_k_pct_cat == pytest.approx(0.910)


def test_pitchers_own_game_date_row_is_strictly_prior():
    """On the date the pitcher pitches, his own game's PAs are excluded:
    the served cell is the cum strictly BEFORE that date (ASOF '>'), so a
    first-career start serves the league fallback, not its own counts."""
    con = _con_with(_DAILY, _LEAGUE, _TOT_DAILY)
    out = _run_sp_cat(con)
    row = out[(out.pitcher == 7) & (out.pitch_cat == "offspeed")
              & (out.game_date == pd.Timestamp("2026-04-02"))].iloc[0]
    # his own 04-02 offspeed PAs (2/10) cannot inform the 04-02 cell
    assert row.sp_k_pct_cat == pytest.approx(0.205)  # league prior
    assert row.sp_xwoba_cat == pytest.approx(0.300)
    assert row.sp_usage_cat == pytest.approx(0.0)


def test_fbhand_zero_prior_falls_back_to_league_stand_rate():
    """Platoon handedness cells get the same fallback (L/R league prior).
    Pitcher 7 faced L batters on 04-02 (1/4 K) and again on 04-05 (1/2 K);
    he has never recorded an R-stand PA."""
    con = duckdb.connect(database=":memory:")
    con.register("exp2_sp_fbhand_daily", pd.DataFrame({
        "game_date": pd.to_datetime(["2026-04-02", "2026-04-05"]),
        "pitcher": [7, 7], "season": [2026, 2026], "stand": ["L", "L"],
        "k_n": [1.0, 1.0], "pa_n": [4.0, 2.0],
    }))
    con.register("exp2_league_fbhand", pd.DataFrame({
        "game_date": pd.to_datetime(["2026-04-01", "2026-04-01"]),
        "stand": ["L", "R"],
        "league_k_pct_fb_vs": [0.230, 0.210],
    }))
    con.execute("""
        CREATE TABLE exp2_sp_fbhand_cum AS
        SELECT *,
            SUM(k_n) OVER w AS k_thru,
            SUM(pa_n) OVER w AS pa_thru
        FROM exp2_sp_fbhand_daily
        WINDOW w AS (PARTITION BY pitcher, season, stand ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)
    con.execute(features._EXP2_SP_FBHAND_SQL)
    out = con.execute(
        "SELECT * FROM exp2_sp_fbhand ORDER BY game_date, stand").df()
    # On 04-05, his L cum strictly before that date is the 04-02 cell (1/4):
    l_row = out[(out.stand == "L")
                & (out.game_date == pd.Timestamp("2026-04-05"))].iloc[0]
    assert l_row.sp_k_pct_fb_vs == pytest.approx(0.25)   # personal prior
    # R: zero prior stand PAs -> strictly-prior league rate for that stand
    r_row = out[(out.stand == "R")
                & (out.game_date == pd.Timestamp("2026-04-05"))].iloc[0]
    assert r_row.sp_k_pct_fb_vs == pytest.approx(0.210)
    # First-career date (04-02) serves league even for the stand he DID pitch
    l_first = out[(out.stand == "L")
                  & (out.game_date == pd.Timestamp("2026-04-02"))].iloc[0]
    assert l_first.sp_k_pct_fb_vs == pytest.approx(0.230)


def test_team_side_gaps_still_propagate_in_add_exp2():
    """The fallback never invents opponent history: a missing team cell keeps
    the served diff NaN even when both SP sides are fully populated."""
    df = pd.DataFrame({
        "league_k_pct": [0.22] * 2,
        "league_k_pct_fb_vs_l": [0.23] * 2,
        "league_k_pct_fb_vs_r": [0.21] * 2,
        **{f"league_k_pct_cat_{c}": [0.22] * 2
           for c in ("fastball", "breaking", "offspeed")},
        **{f"league_xwoba_cat_{c}": [0.32] * 2
           for c in ("fastball", "breaking", "offspeed")},
    })
    for s in ("home", "away"):
        df[f"sp_k9_{s}"] = [8.0, 8.0]
        df[f"team_k_rate_30g_{s}"] = [0.22, 0.22]
        df[f"opp_lefty_share_{s}"] = [0.5, 0.5]
        for c in ("fastball", "breaking", "offspeed"):
            df[f"sp_usage_cat_{c}_{s}"] = [0.3, 0.3]
            df[f"sp_k_pct_cat_{c}_{s}"] = [0.25, 0.25]
            df[f"sp_xwoba_cat_{c}_{s}"] = [0.31, 0.31]
            df[f"team_k_pct_cat_{c}_{s}"] = [0.22, 0.22]
            df[f"team_xwoba_cat_{c}_{s}"] = [0.32, 0.32]
        df[f"sp_k_pct_fb_vs_l_{s}"] = [0.24, 0.24]
        df[f"sp_k_pct_fb_vs_r_{s}"] = [0.22, 0.22]
        df[f"team_k_pct_fb_vs_l_{s}"] = [0.23, 0.23]
        df[f"team_k_pct_fb_vs_r_{s}"] = [0.21, 0.21]
        df[f"sp_usage_cat_fastball_{s}"] = [0.5, 0.5]
    df.loc[0, "team_k_pct_cat_offspeed_away"] = float("nan")
    out = features.add_exp2_features(df)
    assert out["exp2_cat_k_offspeed_diff"].isna().tolist() == [True, False]


# ── creation-order regression (2026-10-01 nightly failure) ────────────────

def _statement_execution_order() -> list[tuple[int, str]]:
    """(lineno, sql-first-line) for every con.execute(CREATE TABLE ...) and
    every con.execute(MODULE_CONSTANT) statement inside
    features._build_game_level, in source order — the execution order."""
    import ast
    tree = ast.parse(Path(features.__file__).read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "_build_game_level")
    out: list[tuple[int, str]] = []
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "execute"):
            continue
        arg = node.args[0] if node.args else None
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            sql = arg.value
        elif isinstance(arg, ast.JoinedStr):
            # f-string statement: the CREATE TABLE header lives in the first
            # constant segment(s); interpolations come later in the body.
            sql = "".join(v.value for v in arg.values
                          if isinstance(v, ast.Constant)
                          and isinstance(v.value, str))
        elif isinstance(arg, ast.Name):
            sql = getattr(features, arg.id, "")
        else:
            continue
        for line in sql.splitlines():
            s = line.strip()
            if s.upper().startswith("CREATE TABLE "):
                out.append((node.lineno, s.split()[2]))
                break
    return sorted(out)


def test_league_fbhand_created_before_its_consumers():
    """The 2026-10-01 nightly run died with CatalogException: the
    zero-prior fallback made _EXP2_SP_FBHAND_SQL consume exp2_league_fbhand
    while its CREATE lived LATER in the build. Pinned by simulated
    execution order (module-constant statements resolve to their SQL),
    not raw text offsets."""
    order = _statement_execution_order()
    created = {name: i for i, (_, name) in enumerate(order)}
    assert "exp2_league_fbhand" in created
    for consumer in ("exp2_sp_fbhand", "game_level"):
        assert consumer in created, f"{consumer} not found in order scan"
        assert created[consumer] > created["exp2_league_fbhand"], (
            f"{consumer} executes before exp2_league_fbhand is created")
