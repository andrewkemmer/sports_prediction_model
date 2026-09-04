"""Probe: NFL PBP player-cohort feature diagnostic (READ-ONLY, record-only).

Tests whether six 'cohort structure' families add signal BEYOND the
12-pool's existing PBP aggregates (ewm_ypp_diff, ewm_net_pts_diff) and
elo, using the QB-strata protocol (796802c): strata evidence +
pre-registered in-sample ceiling bar; no pipeline is built unless a
family survives.

The six families describe offense-vs-defense UNIT-TYPE structure that
level-diff aggregates cannot see (a team's pass EPA says nothing about
*whose* unit generates it):

  F1 pass_epa      EPA/play on pass plays (OFF) and allowed vs opponent
                   pass plays (DEF)
  F2 rush_epa      EPA/play on run plays (OFF) / allowed (DEF)
  F3 pressure      OL pressure-rate ALLOWED by the starter's protection
                   (PFR advstats: pressures per dropback, OFF) and DL
                   pressure GENERATED (DEF = the opponent's OL rate in
                   the same game)
  F4 big_play      share of scrimmage plays that are big: run > 10 yds,
                   or pass with air_yards >= 20 (downfield attempt;
                   air_yards NaN -> yards_gained >= 20)
  F5 redzone_td    red-zone TD rate per possession drive (drive with a
                   play at yardline_100 <= 20 that ends in a TD)
  F6 third_down    mean EPA on 3rd-down plays (OFF) / allowed (DEF)

As-of construction (strictly-prior, no retro-fit): each side's weekly
stat feeds an ewm halflife-2 series over the team's PRIOR weeks in the
same season; floor >= 4 prior weeks, else the team's prior-season
(S-1) weekly mean (>= 5 weeks), else the league-mean carry (global
mean over the whole window — QB-strata precedent, documented).

Game-level pairing (per family f): with asof_off/asof_def per side,
    M_h = asof_off(home, f) - asof_def(away, f)   (home O vs away D)
    M_a = asof_off(away, f) - asof_def(home, f)
    cohort mismatch = max(|M_h|, |M_a|)

Phases: universe (1,376 decided OOF games, same as c1a7c12), weekly
stats + as-of, redundancy gate (R2 of {M_h, M_a} on elo_diff /
ewm_ypp_diff / ewm_net_pts_diff; R2 > 0.95 -> STOP that family),
strata (top-quartile mismatch vs rest), in-sample ceiling models
(M0 = logit(p); Mf = logit(p) + M_h + M_a), pre-registered verdicts:

  GO              redundancy R2 <= 0.95 AND strata logloss gap >= 0.008
                  (mismatch games harder) AND ceiling logloss delta
                  <= -0.004 (or R2 >= 0.02).
  RE_TEST_CANDIDATE  redundancy ok and (strata gap in [0.002, 0.008)
                  or ceiling delta in (-0.004, -0.002]).
  STOP            redundant (R2 > 0.95), strata gap < 0.002, or ceiling
                  delta >= -0.002.

Decided OOF rows ONLY (never slate/serve rows). Deterministic: the
record JSON is the pin; double-run byte-identical.

Usage (caches on E:/tmp when run from this machine's E: drive):
    cd /e/tmp && PYTHONUTF8=1 python "G:/My Drive/.../probe_pbp_cohorts.py"
Env overrides: NFL_PBP_DIR, NFL_PFR_DIR, NFL_FEATURES_DIR (default
"/tmp", resolved per current drive) — same convention as
probe_qb_strata.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

DD = Path(__file__).resolve().parent.parent / "data_delivery"
DATE = "20260904"
PBP_SEASONS = list(range(2020, 2026))     # 2020 for 2021 early-week fallback
PFR_SEASONS = list(range(2020, 2026))

PBP_DIR = os.environ.get("NFL_PBP_DIR", "/tmp")
PFR_DIR = os.environ.get("NFL_PFR_DIR", "/tmp")
FEATURES_DIR = os.environ.get("NFL_FEATURES_DIR", "/tmp")

ASOF_HALFLIFE = 2
ASOF_FLOOR_PRIOR = 4
PRIOR_SEASON_FLOOR = 5
REDUNDANCY_R2_STOP = 0.95
STRATA_GAP_GO = 0.008
STRATA_GAP_RETEST = 0.002
CEILING_GO = -0.004
CEILING_RETEST = -0.002

FAMILIES = ["F1_pass_epa", "F2_rush_epa", "F3_pressure", "F4_big_play",
            "F5_redzone_td", "F6_third_down"]
REDUNDANCY_REGRESSORS = ["elo_diff", "ewm_ypp_diff", "ewm_net_pts_diff"]
SIDE_COLS = ["game_id", "season", "week", "team", "off_val", "def_val"]


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _ece10(y: np.ndarray, p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    n = len(p)
    total = 0.0
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        total += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(total)


# ---------------------------------------------------------------------------
# Weekly stats per family (OFF and DEF per team-week)
# ---------------------------------------------------------------------------
def _load_pbp(seasons: list[int]) -> pd.DataFrame:
    cols = ["game_id", "season", "week", "posteam", "defteam", "play_type",
            "epa", "air_yards", "yards_gained", "yardline_100", "down",
            "touchdown", "two_point_attempt", "fixed_drive"]
    parts = []
    for yr in seasons:
        pf = Path(f"{PBP_DIR}/nfl_pbp_{yr}.parquet")
        if pf.exists():
            parts.append(pd.read_parquet(pf, columns=cols))
        else:
            import nflreadpy
            pbp = nflreadpy.load_pbp([yr])
            if hasattr(pbp, "to_pandas"):
                pbp = pbp.to_pandas()
            pbp = pbp[cols]
            try:
                pf.parent.mkdir(parents=True, exist_ok=True)
                pbp.to_parquet(pf)
            except Exception:  # noqa: BLE001 - cache write must not block
                pass
            parts.append(pbp)
    return pd.concat(parts, ignore_index=True)


def _load_pfr(seasons: list[int]) -> pd.DataFrame:
    cols = ["game_id", "season", "week", "team", "opponent",
            "times_pressured", "times_pressured_pct"]
    parts = []
    for yr in seasons:
        pf = Path(f"{PFR_DIR}/nfl_pfr_advstats_{yr}.parquet")
        if pf.exists():
            parts.append(pd.read_parquet(pf, columns=cols))
        else:
            import nflreadpy
            pfr = nflreadpy.load_pfr_advstats([yr])
            if hasattr(pfr, "to_pandas"):
                pfr = pfr.to_pandas()
            pfr = pfr[cols]
            try:
                pf.parent.mkdir(parents=True, exist_ok=True)
                pfr.to_parquet(pf)
            except Exception:  # noqa: BLE001
                pass
            parts.append(pfr)
    return pd.concat(parts, ignore_index=True)


def weekly_stats_pbp(pbp: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """F1/F2/F4/F5/F6 per (game, team, side) weekly values from PBP."""
    sp = pbp[(pbp["play_type"].isin(["pass", "run"]))
             & (pbp["two_point_attempt"] != 1)].copy()

    def _off(group: pd.DataFrame, stat: str) -> pd.DataFrame:
        """Per (game, season, week, posteam) value of an OFF stat."""
        return group.groupby(["game_id", "season", "week", "posteam"],
                             sort=False).agg(val=(stat, "mean")).reset_index()

    def _def(group: pd.DataFrame, stat: str) -> pd.DataFrame:
        g = group.groupby(["game_id", "season", "week", "defteam"],
                          sort=False).agg(val=(stat, "mean")).reset_index()
        return g.rename(columns={"defteam": "team"})

    out: dict[str, pd.DataFrame] = {}

    pass_p = sp[sp["play_type"] == "pass"]
    run_p = sp[sp["play_type"] == "run"]

    # F1 pass EPA/play
    ep = pass_p[pass_p["epa"].notna()]
    f1o = _off(ep, "epa"); f1o["val"] = f1o["val"].astype(float)
    f1d = _def(ep, "epa"); f1d["val"] = f1d["val"].astype(float)
    out["F1_pass_epa"] = pd.concat(
        [f1o.rename(columns={"posteam": "team", "val": "off_val"}),
         f1d.rename(columns={"val": "def_val"})], ignore_index=True)
    # F2 rush EPA/play
    rp = run_p[run_p["epa"].notna()]
    f2o = _off(rp, "epa"); f2d = _def(rp, "epa")
    out["F2_rush_epa"] = pd.concat(
        [f2o.rename(columns={"posteam": "team", "val": "off_val"}),
         f2d.rename(columns={"val": "def_val"})], ignore_index=True)

    # F4 big-play share (no epa filter; big = pass air>=20 or gain>=20
    # when air NaN; run > 10 yds).  NaN air_yards on passes is not big
    # unless yards_gained >= 20 (documented air-yards context rule).
    spb = sp.copy()
    spb["big"] = np.where(
        spb["play_type"] == "pass",
        (spb["air_yards"].fillna(-1) >= 20) |
        (spb["air_yards"].isna() & (spb["yards_gained"] >= 20)),
        spb["yards_gained"] > 10)
    denom = spb.groupby(["game_id", "season", "week", "posteam"],
                        sort=False).size().reset_index(name="n")
    num = (spb.groupby(["game_id", "season", "week", "posteam"],
                       sort=False)["big"].sum().reset_index(name="nbig"))
    f4o = denom.merge(num, on=["game_id", "season", "week", "posteam"])
    f4o["off_val"] = f4o["nbig"] / f4o["n"]
    f4o = f4o.rename(columns={"posteam": "team"})
    dden = spb.groupby(["game_id", "season", "week", "defteam"],
                       sort=False).size().reset_index(name="n")
    dnum = (spb.groupby(["game_id", "season", "week", "defteam"],
                        sort=False)["big"].sum().reset_index(name="nbig"))
    f4d = dden.merge(dnum, on=["game_id", "season", "week", "defteam"])
    f4d["def_val"] = f4d["nbig"] / f4d["n"]
    f4d = f4d.rename(columns={"defteam": "team"})
    out["F4_big_play"] = pd.concat(
        [f4o[["game_id", "season", "week", "team", "off_val"]],
         f4d[["game_id", "season", "week", "team", "def_val"]]],
        ignore_index=True)

    # F5 red-zone TD rate per possession drive.  A drive = (game, posteam,
    # fixed_drive).  RZ = any scrimmage play at yardline_100 <= 20.
    dr = sp[["game_id", "season", "week", "posteam", "defteam",
             "fixed_drive", "yardline_100", "touchdown"]].copy()
    dr["rz"] = dr["yardline_100"] <= 20
    dr["td"] = dr["touchdown"] == 1
    dg = (dr.groupby(["game_id", "season", "week", "posteam", "fixed_drive"],
                     sort=False)
          .agg(has_rz=("rz", "any"), has_td=("td", "any"),
               defteam=("defteam", "first"))
          .reset_index())
    drive_off = (dg[dg["has_rz"]].groupby(["game_id", "season", "week",
                                           "posteam"], sort=False)
                 .agg(td_rate=("has_td", "mean")).reset_index()
                 .rename(columns={"posteam": "team", "td_rate": "off_val"}))
    drive_def = (dg[dg["has_rz"]].groupby(["game_id", "season", "week",
                                           "defteam"], sort=False)
                 .agg(td_rate=("has_td", "mean")).reset_index()
                 .rename(columns={"defteam": "team", "td_rate": "def_val"}))
    out["F5_redzone_td"] = pd.concat([drive_off, drive_def],
                                     ignore_index=True)

    # F6 third-down EPA/play
    td3 = sp[(sp["down"] == 3) & (sp["epa"].notna())]
    f6o = _off(td3, "epa"); f6d = _def(td3, "epa")
    out["F6_third_down"] = pd.concat(
        [f6o.rename(columns={"posteam": "team", "val": "off_val"}),
         f6d.rename(columns={"val": "def_val"})], ignore_index=True)

    # normalize side frames to a single row per (game, season, week, team)
    for fam in out:
        f = out[fam]
        off = f[f["off_val"].notna()][["game_id", "season", "week", "team",
                                       "off_val"]]
        dff = f[f["def_val"].notna()][["game_id", "season", "week", "team",
                                       "def_val"]]
        m = off.merge(dff, on=["game_id", "season", "week", "team"], how="outer")
        out[fam] = m[SIDE_COLS]
    return out


def weekly_stats_pressure(pfr: pd.DataFrame) -> pd.DataFrame:
    """F3 from PFR advstats: per (game, team) the STARTING QB's protection
    row (max implied dropbacks = times_pressured / pressure rate) as the
    OL pressure-rate ALLOWED.  DEF (DL generated) = the opponent's OL
    allowed rate in the same game (self-join on team <-> opponent)."""
    p = pfr.copy()
    p["dropbacks"] = np.where(
        (p["times_pressured_pct"].fillna(0) > 0) & p["times_pressured"].notna(),
        p["times_pressured"] / p["times_pressured_pct"], 0.0)
    # a row is a real protection sample when it carries a rate or pressures;
    # truly empty backup rows (all NaN/0) carry no rate and are dropped
    p = p[(p["times_pressured_pct"].notna())
          | (p["times_pressured"].fillna(0) > 0)].copy()
    # the starter = the row with the most implied dropbacks per game-team
    p = p.sort_values(["game_id", "team", "dropbacks"],
                      ascending=[True, True, False])
    st = (p.groupby(["game_id", "season", "week", "team"], sort=False)
          .head(1).reset_index(drop=True))
    off = st[["game_id", "season", "week", "team", "times_pressured_pct",
              "opponent"]]
    off = off.rename(columns={"times_pressured_pct": "off_val"})
    # DEF: the opposing team's OL rate from that same game
    defv = off.rename(columns={"team": "opponent", "opponent": "team",
                               "off_val": "def_val"})
    m = off[["game_id", "team", "off_val"]].merge(
        defv[["game_id", "team", "def_val"]], on=["game_id", "team"],
        how="outer")
    m = m.merge(st[["game_id", "season", "week", "team"]].drop_duplicates(),
                on=["game_id", "team"], how="left")
    return m[SIDE_COLS]


# ---------------------------------------------------------------------------
# As-of (strictly-prior ewm halflife-2 with fallback chain)
# ---------------------------------------------------------------------------
def _ewm_prior(series: pd.Series, halflife: int) -> tuple[pd.Series, pd.Series]:
    """ewm(halflife, adjust=False).mean() over strictly-prior values.

    Returns (prior_ewm, prior_count) aligned to the input series order —
    the ewm is computed over the cumulative series and shifted by one, so
    the value at week w uses only weeks < w.
    """
    e = series.ewm(halflife=halflife, adjust=False).mean()
    count = series.expanding(min_periods=1).count()
    return e.shift(1), count.shift(1).fillna(0).astype(int)


def add_asof(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add off_asof / def_asof per (team, season, week): within-season
    ewm halflife 2 with floor >= ASOF_FLOOR_PRIOR prior weeks; else the
    team's S-1 weekly mean (>= PRIOR_SEASON_FLOOR weeks); else the
    global league mean (whole window — QB-strata precedent).

    Returns the weekly frame with the two extra columns plus the
    prior-count columns used for coverage reporting."""
    w = weekly.sort_values(["team", "season", "week"]).reset_index(drop=True)
    w = w.copy()

    for side in ("off_val", "def_val"):
        if side not in w.columns:
            continue
        by_g = w.groupby(["team", "season"], sort=False)[side]
        # group-aware ewm + shift: the value at week w uses weeks < w only
        ewm_g = by_g.transform(
            lambda s: s.ewm(halflife=ASOF_HALFLIFE, adjust=False)
            .mean().shift(1))
        cnt_g = by_g.transform(
            lambda s: s.expanding(min_periods=1).count().shift(1).fillna(0))
        w[f"{side}_ewm"] = ewm_g
        w[f"{side}_prior_n"] = cnt_g.astype(int)

    # prior-season means (per team, value usable in season S+1)
    ps = (w.groupby(["team", "season"], sort=False)[["off_val", "def_val"]]
          .agg(["count", "mean"]).reset_index())
    ps.columns = ["team", "season", "off_n", "off_m", "def_n", "def_m"]
    ps["season"] = ps["season"] + 1
    w = w.merge(ps, on=["team", "season"], how="left")

    league_off = float(w["off_val"].mean())
    league_def = float(w["def_val"].mean())

    def _chain(row: pd.Series, side: str, ncol: str, mcol: str) -> float:
        if row[f"{side}_prior_n"] >= ASOF_FLOOR_PRIOR and \
                pd.notna(row[f"{side}_ewm"]):
            return float(row[f"{side}_ewm"])
        if row[ncol] >= PRIOR_SEASON_FLOOR and pd.notna(row[mcol]):
            return float(row[mcol])
        return league_off if side == "off_val" else league_def

    w["off_asof"] = w.apply(lambda r: _chain(r, "off_val", "off_n", "off_m"),
                            axis=1)
    w["def_asof"] = w.apply(lambda r: _chain(r, "def_val", "def_n", "def_m"),
                            axis=1)
    return w


# ---------------------------------------------------------------------------
# Game-level pairing + gates
# ---------------------------------------------------------------------------
def pair_covariates(weekly: pd.DataFrame,
                    universe: pd.DataFrame) -> pd.DataFrame:
    """Join the as-of weekly values to the universe and build the pairing
    covariates M_h / M_a / mismatch for one family."""
    cols = ["game_id", "team", "off_asof", "def_asof"]
    aw = weekly[cols]
    home = aw.rename(columns={"team": "home_team", "off_asof": "h_off",
                              "def_asof": "h_def"})
    away = aw.rename(columns={"team": "away_team", "off_asof": "a_off",
                              "def_asof": "a_def"})
    u = universe[["game_id", "home_team", "away_team"]].copy()
    u = u.merge(home, on=["game_id", "home_team"], how="left")
    u = u.merge(away, on=["game_id", "away_team"], how="left")
    u["M_h"] = u["h_off"] - u["a_def"]
    u["M_a"] = u["a_off"] - u["h_def"]
    u["mismatch"] = np.maximum(u["M_h"].abs(), u["M_a"].abs())
    return u


def redundancy_r2(cov: pd.DataFrame, feats: pd.DataFrame) -> dict:
    """R2 of M_h and M_a each regressed on the three served inputs."""
    from numpy.linalg import lstsq
    joined = cov.merge(feats, on="game_id", how="left")

    def _r2(col: str) -> float | None:
        m = joined[[col] + REDUNDANCY_REGRESSORS].dropna()
        if len(m) < 30 or m[REDUNDANCY_REGRESSORS].notna().all(axis=1).sum() < 30:
            return None
        X = np.column_stack([np.ones(len(m))] +
                            [m[c].to_numpy(float) for c in REDUNDANCY_REGRESSORS])
        y = m[col].to_numpy(float)
        beta, *_ = lstsq(X, y, rcond=None)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        ss_res = float(((y - X @ beta) ** 2).sum())
        return round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0

    return {"r2_M_h": _r2("M_h"), "r2_M_a": _r2("M_a"),
            "regressors": REDUNDANCY_REGRESSORS,
            "n_joined": int(joined["M_h"].notna().sum())}


def strata_rows(cov: pd.DataFrame) -> list[dict]:
    """Top-quartile mismatch vs the rest (mirrors QB _stratum_rows)."""
    q75 = float(cov["mismatch"].quantile(0.75))
    cov = cov.copy()
    cov["hi"] = cov["mismatch"] >= q75

    def _row(mask: pd.Series, label: str) -> dict:
        s = cov[mask]
        if len(s) == 0:
            return {"label": label, "n": 0}
        return {
            "label": label, "n": int(len(s)),
            "mean_platt_pred": round(float(s["binary"].mean()), 4),
            "actual_home_win": round(float(s["home_win"].mean()), 4),
            "logloss": round(_logloss(s["home_win"].to_numpy(float),
                                      s["binary"].to_numpy(float)), 4),
            "ece": round(_ece10(s["home_win"].to_numpy(float),
                                s["binary"].to_numpy(float)), 4),
            "mean_derived_ml": round(float(s["derived"].mean()), 4),
            "mean_abs_binary_derived": round(
                float((s["binary"] - s["derived"]).abs().mean()), 4),
            "mean_mismatch": round(float(s["mismatch"].mean()), 4),
        }

    return [_row(cov["hi"], "top_quartile_mismatch"),
            _row(~cov["hi"], "rest")]


def ceiling_models(cov: pd.DataFrame) -> dict:
    """M0 = logit(p); Mf = logit(p) + M_h + M_a (in-sample, same rows)."""
    from sklearn.linear_model import LogisticRegression
    m = cov.dropna(subset=["binary", "M_h", "M_a"]).copy()
    y = m["home_win"].to_numpy(float)
    p = np.clip(m["binary"].to_numpy(float), 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p))

    def _fit(xcols: list[str]) -> tuple[float, float, list[float]]:
        X = np.column_stack([lg] + [m[c].to_numpy(float) for c in xcols])
        lr = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)
        pp = lr.predict_proba(X)[:, 1]
        resid = y - pp
        r2 = 0.0
        if xcols:
            Xs = np.column_stack([np.ones(len(m))] +
                                 [m[c].to_numpy(float) for c in xcols])
            beta, *_ = np.linalg.lstsq(Xs, resid, rcond=None)
            ss_tot = float(((resid - resid.mean()) ** 2).sum())
            ss_res = float(((resid - Xs @ beta) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return _logloss(y, pp), max(r2, 0.0), \
            [float(c) for c in lr.coef_[0][1:]]

    ll0, r2_0, _ = _fit([])
    llf, r2_f, coefs = _fit(["M_h", "M_a"])
    return {
        "n": int(len(m)),
        "M0_logit_p_only_ll": round(ll0, 4),
        "Mf_plus_pairing_ll": round(llf, 4),
        "delta_ll_vs_M0": round(llf - ll0, 4),
        "r2_of_y_minus_p": round(r2_f, 4),
        "coef_M_h": round(coefs[0], 4) if len(coefs) > 0 else None,
        "coef_M_a": round(coefs[1], 4) if len(coefs) > 1 else None,
        "coef_sign_M_h": "positive" if len(coefs) > 0 and coefs[0] > 0
        else ("negative" if len(coefs) > 0 else None),
        "coef_sign_M_a": "positive" if len(coefs) > 1 and coefs[1] > 0
        else ("negative" if len(coefs) > 1 else None),
        "label": ("IN-SAMPLE explanatory fit with as-of unit values on the "
                  "same games the model consumes — an UPPER BOUND; "
                  "serve-time noise lowers achievable gain"),
    }


def pre_registered_verdict(redundancy: dict, strata: list[dict],
                           ceiling: dict, label: str) -> dict:
    """Phase-4 routing per family (thresholds pre-registered in the
    module docstring; the runner computes, it does not narrate)."""
    r2s = [v for v in (redundancy.get("r2_M_h"), redundancy.get("r2_M_a"))
           if v is not None]
    redundant = bool(r2s and max(r2s) > REDUNDANCY_R2_STOP)
    hi = next((r for r in strata if r["label"] == "top_quartile_mismatch"),
              {})
    rest = next((r for r in strata if r["label"] == "rest"), {})
    strata_gap = (hi.get("logloss", 0.0) - rest.get("logloss", 0.0)
                  if hi.get("n") and rest.get("n") else None)
    delta = ceiling.get("delta_ll_vs_M0")
    r2 = ceiling.get("r2_of_y_minus_p", 0.0)

    if redundant:
        verdict = "STOP"
        reason = f"redundant: R2 {max(r2s):.3f} > 0.95 vs served inputs"
    elif (strata_gap is not None and strata_gap >= STRATA_GAP_GO
          and delta is not None and (delta <= CEILING_GO or r2 >= 0.02)):
        verdict = "GO"
        reason = f"strata gap {strata_gap:.4f} >= 0.008 and ceiling delta " \
                 f"{delta:.4f} <= -0.004 (or R2 {r2:.4f} >= 0.02)"
    elif (strata_gap is not None and delta is not None
          and (STRATA_GAP_RETEST <= strata_gap < STRATA_GAP_GO
               or CEILING_GO < delta <= CEILING_RETEST)):
        verdict = "RE_TEST_CANDIDATE"
        reason = (f"strata gap {strata_gap:.4f} in [0.002, 0.008) or "
                  f"ceiling delta {delta:.4f} in (-0.004, -0.002]")
    else:
        verdict = "STOP"
        reason = (f"strata gap {strata_gap if strata_gap is not None else 'n/a'}"
                  f" < 0.002 and/or ceiling delta "
                  f"{delta if delta is not None else 'n/a'} >= -0.002")
    return {"family": label, "verdict": verdict, "reason": reason,
            "redundant": redundant,
            "redundancy_r2_max": round(max(r2s), 4) if r2s else None,
            "strata_logloss_gap_topq_minus_rest":
            round(strata_gap, 4) if strata_gap is not None else None,
            "ceiling_delta_ll": delta,
            "rules": {
                "GO": ("R2 <= 0.95 AND strata gap >= 0.008 AND ceiling "
                       "delta <= -0.004 (or R2 >= 0.02)"),
                "RE_TEST_CANDIDATE": ("R2 <= 0.95 and strata gap in "
                                      "[0.002, 0.008) or ceiling delta in "
                                      "(-0.004, -0.002]"),
                "STOP": "redundant (R2 > 0.95) or ceiling < 0.002"}}


# ---------------------------------------------------------------------------
def main() -> int:
    # ---- universe (identical to probe_qb_strata / probe_margin) ----------
    mk = pd.read_csv(DD / f"nfl_run_engine_markets_{DATE}.csv")
    oof = mk[mk["kind"] == "oof"].copy()
    hist = pd.read_csv(DD / f"nfl_predictions_history_{DATE}.csv")
    oof = oof.rename(columns={"pred_home": "re_pred_home",
                              "pred_away": "re_pred_away"})
    oof = oof[["game_id", "p_home_win_derived"]]
    df = oof.merge(hist[["game_id", "season", "week", "home_team",
                         "away_team", "home_score", "away_score",
                         "home_win_prob_model",
                         "home_win_prob_model_calibrated"]],
                   on="game_id", how="left")
    df["binary"] = df["home_win_prob_model_calibrated"]
    df["derived"] = df["p_home_win_derived"]
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(float)
    assert len(df) == 1376, f"universe {len(df)} != 1376"
    out: dict = {"universe": {"n": int(len(df)),
                              "n_binary_nonnull":
                              int(df["binary"].notna().sum()),
                              "seasons": {int(k): int(v) for k, v in
                                          df["season"].value_counts().items()}},
                 "as_of_convention": {
                     "ewm_halflife": ASOF_HALFLIFE,
                     "floor_prior_weeks": ASOF_FLOOR_PRIOR,
                     "prior_season_fallback_floor_weeks": PRIOR_SEASON_FLOOR,
                     "fallback_chain": ("within-season ewm(hl=2) over prior "
                                        "weeks >= 4 -> prior-season (S-1) "
                                        "weekly mean >= 5 weeks -> global "
                                        "league-mean carry")}}

    # ---- features frame for the redundancy gate (12-pool served inputs) --
    from run_nfl_margin_ablation import DECIDED_FRAME, _frame_sha256
    decided = pd.read_csv(DECIDED_FRAME)
    sha = _frame_sha256(decided)
    feat_cache = Path(f"{FEATURES_DIR}/nfl_features_{sha}.parquet")
    feats = None
    if feat_cache.exists():
        fc = pd.read_parquet(feat_cache)
        if all(c in fc.columns for c in REDUNDANCY_REGRESSORS):
            feats = fc[["game_id"] + REDUNDANCY_REGRESSORS]
    if feats is None:
        from run_nfl_margin_ablation import load_features
        lf = load_features(None)
        feats = lf[["game_id"] + REDUNDANCY_REGRESSORS]
    out["features"] = {"frame_sha_prefix": sha,
                       "cache_path": str(feat_cache),
                       "n_rows": int(len(feats))}

    # ---- weekly stats + as-of -------------------------------------------------
    pbp = _load_pbp(PBP_SEASONS)
    pfr = _load_pfr(PFR_SEASONS)
    weekly_all = weekly_stats_pbp(pbp)
    weekly_all["F3_pressure"] = weekly_stats_pressure(pfr)

    fam_results = {}
    pred_cols = df[["game_id", "binary", "derived", "home_win"]]
    for fam in FAMILIES:
        wk = add_asof(weekly_all[fam])
        cov = pair_covariates(wk, df)
        cov = cov.merge(pred_cols, on="game_id", how="left")
        red = redundancy_r2(cov, feats)
        strata = strata_rows(cov)
        ceiling = ceiling_models(cov)
        verdict = pre_registered_verdict(red, strata, ceiling, fam)
        fam_results[fam] = {
            "weekly_team_weeks": int(len(wk)),
            "coverage": {
                "n_paired": int(cov["M_h"].notna().sum()),
                "pct_of_universe": round(
                    float(cov["M_h"].notna().mean() * 100), 1),
                "asof_prior_lt_floor_pct": round(float(
                    (wk["off_val_prior_n"] < ASOF_FLOOR_PRIOR).mean()
                    * 100), 2),
            },
            "strata": strata,
            "ceiling": ceiling,
            "verdict": verdict,
        }
    out["families"] = fam_results
    out["family_definitions"] = {
        "F1_pass_epa": "OFF mean EPA/pass play; DEF mean EPA allowed on "
                       "opponent pass plays",
        "F2_rush_epa": "OFF mean EPA/run play; DEF mean EPA allowed on "
                       "opponent run plays",
        "F3_pressure": "OFF OL pressure-rate allowed (PFR starter row: "
                       "pressures per dropback); DEF DL pressure generated "
                       "= opponent's OL rate in the same game",
        "F4_big_play": "OFF share of scrimmage plays that are big (run > "
                       "10 yds; pass air_yards >= 20, or yards >= 20 when "
                       "air_yards NaN); DEF mirrored on opponent plays",
        "F5_redzone_td": "OFF red-zone TD rate per possession drive (RZ = "
                         "yardline_100 <= 20); DEF mirrored",
        "F6_third_down": "OFF mean EPA on 3rd-down plays; DEF mean EPA "
                         "allowed on opponent 3rd-down plays",
    }
    out["pairing"] = {
        "M_h": "asof_off(home) - asof_def(away)",
        "M_a": "asof_off(away) - asof_def(home)",
        "mismatch": "max(|M_h|, |M_a|)",
        "ceiling_model": "M0 = logit(p); Mf = logit(p) + M_h + M_a "
                         "(in-sample, same 1,376 rows)"}

    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
