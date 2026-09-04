"""Probe: NFL QB-mismatch strata diagnostic (READ-ONLY, record-only).

Step 1 of the QB/availability feature program: tests whether binary
moneyline OOF errors concentrate in QB-strata games BEFORE any QB
feature pipeline is built. Decided OOF rows ONLY (the shared 1,376-game
universe of c1a7c12 / probe_margin_disagreement.py); never slate rows.

Facts (retrospective, from cached nflreadpy PBP 2020-2025):
  - actual starter per (game, team) = the passer with the most dropbacks
    (qb_dropback == 1, passer_player_id non-null); tie -> first passer.
  - BACKUP flag (operational def): starter differs from the team's
    starter in its most recent PRIOR game in the SAME season; week-1 /
    no-prior -> compare to the prior season's primary starter (by most
    starts; tie -> the last game's starter). Sensitivity row: season
    primary-by-most-starts rule.
  - as-of QB quality (strictly-prior, no look-ahead): per starter, per
    game, EPA/play (mean qb_epa over his dropbacks). For game g use the
    starter's rolling mean over his prior games THAT season (floor >= 3
    prior games), else his prior-season per-game mean (>= 5 games),
    else the league mean. qb_gap = home_asof - away_asof.

Phase 2 strata tables (backup / high-gap |qb_gap|>=0.10 / interaction),
Phase 3 in-sample ceiling models M0-M3 (retrospective-actual-starter
facts — an EXPLICIT upper bound, not an adoption-gain claim), Phase 4
pre-registered verdict routing (GO / STOP-BINARY / STOP-ALL).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DD = Path(__file__).resolve().parent.parent / "data_delivery"
DATE = "20260904"
# Season-parquet cache dir (repo convention: /tmp; override for tests or
# cross-drive layouts via NFL_PBP_DIR). Missing seasons are fetched via
# nflreadpy and written here.
PBP_CACHE = os.environ.get("NFL_PBP_DIR", "/tmp") + "/nfl_pbp_{yr}.parquet"
PBP_SEASONS = list(range(2020, 2026))     # 2020 for 2021 week-1 fallback

HIGH_GAP_THRESHOLD = 0.10
ASOF_FLOOR_PRIOR = 3
PRIOR_SEASON_FLOOR = 5


def _load_pbp(seasons: list[int]) -> pd.DataFrame:
    cols = ["game_id", "season", "week", "posteam", "passer_player_id",
            "passer_player_name", "qb_epa", "qb_dropback", "play_id"]
    parts = []
    for yr in seasons:
        pf = Path(PBP_CACHE.format(yr=yr))
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


def build_starter_tables(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-(game, team) actual-starter + game-level EPA/play table."""
    db = pbp[(pbp["qb_dropback"] == 1) & (pbp["passer_player_id"].notna())]
    db = db.copy()
    db["passer_player_id"] = db["passer_player_id"].astype(str)
    db = db.sort_values(["game_id", "play_id"])
    g = (db.groupby(["game_id", "season", "week", "posteam",
                     "passer_player_id"], sort=False)
           .agg(n_dropback=("qb_epa", "size"),
                first_play=("play_id", "first"),
                epa_mean=("qb_epa", "mean"))
           .reset_index())
    # starter per team-game: most dropbacks; tie -> first passer by play
    g = g.sort_values(["game_id", "n_dropback", "first_play"],
                      ascending=[True, False, True])
    st = (g.groupby(["game_id", "season", "week", "posteam"], sort=False)
           .head(1).reset_index(drop=True))
    return st


def season_primary(st: pd.DataFrame) -> pd.DataFrame:
    """Per (team, season): primary starter by most starts; tie -> the
    starter of the team's last game that season."""
    counts = (st.groupby(["posteam", "season", "passer_player_id"])
              .size().reset_index(name="starts"))
    idx = counts.sort_values(["posteam", "season", "starts"],
                             ascending=[True, True, False])
    prim = idx.groupby(["posteam", "season"]).head(1).reset_index(drop=True)
    return prim[["posteam", "season", "passer_player_id"]]


def add_backup_flags(st: pd.DataFrame,
                     prim: pd.DataFrame) -> pd.DataFrame:
    """Operational backup flag + season-primary sensitivity flag."""
    st = st.copy()
    st = st.sort_values(["posteam", "season", "week"]).reset_index(drop=True)
    prev = (st.groupby(["posteam", "season"], sort=False)
              ["passer_player_id"].shift(1))
    # first game of a team's season -> prior-season primary comparison
    st["_first_of_season"] = prev.isna()
    st["prev_starter"] = prev
    pm = prim.set_index(["posteam", "season"])
    st["prev_primary"] = st.apply(
        lambda r: pm.loc[(r["posteam"], r["season"] - 1),
                         "passer_player_id"]
        if (r["posteam"], r["season"] - 1) in pm.index else None,
        axis=1)
    st["backup_op"] = np.where(
        st["_first_of_season"],
        st["passer_player_id"] != st["prev_primary"],
        st["passer_player_id"] != st["prev_starter"])
    st["backup_season_primary"] = st.apply(
        lambda r: r["passer_player_id"]
        != pm.loc[(r["posteam"], r["season"]), "passer_player_id"]
        if (r["posteam"], r["season"]) in pm.index else False,
        axis=1)
    return st


def add_asof_quality(st: pd.DataFrame) -> pd.DataFrame:
    """Strictly-prior rolling EPA/play per starter + prior-season fallback
    + league mean. Columns on st: prior_n, prior_mean, prev_season_mean,
    asof_epa."""
    st = st.copy()
    st = st.sort_values(["posteam", "season", "passer_player_id", "week"])
    grp = st.groupby(["posteam", "season", "passer_player_id"],
                     sort=False)["epa_mean"]
    cs = grp.cumsum()
    cc = grp.cumcount()
    st["prior_sum"] = cs - st["epa_mean"]
    st["prior_n"] = cc
    with np.errstate(invalid="ignore"):
        st["prior_mean"] = st["prior_sum"] / st["prior_n"].where(
            st["prior_n"] > 0)
    # prior-season per-passer mean over his starter games (>= 5 games)
    ps = (st.groupby(["posteam", "passer_player_id"])["epa_mean"]
            .agg(["count", "mean"]).reset_index())
    ps.columns = ["posteam", "passer_player_id", "ps_n", "ps_mean"]
    st = st.merge(ps, on=["posteam", "passer_player_id"], how="left")
    st["prev_season_mean"] = np.where(
        (st["ps_n"] >= PRIOR_SEASON_FLOOR), st["ps_mean"], np.nan)
    # NOTE: ps aggregates over ALL seasons; for a season-S fallback we need
    # the passer's S-1 mean specifically — computed in main() per universe
    # row instead (kept simple here; the primary path is the within-season
    # rolling mean which covers the vast majority of rows).
    league = float(st["epa_mean"].mean())
    st["league_mean"] = league
    return st


def main() -> int:
    # ---- universe (identical to probe_margin_disagreement) --------------
    mk = pd.read_csv(DD / f"nfl_run_engine_markets_{DATE}.csv")
    oof = mk[mk["kind"] == "oof"].copy()
    hist = pd.read_csv(DD / f"nfl_predictions_history_{DATE}.csv")
    oof = oof.rename(columns={"pred_home": "re_pred_home",
                              "pred_away": "re_pred_away"})
    # keep only the markets-unique columns (scores/teams/season/week come
    # from history — both CSVs carry them, so a full merge would suffix-rename)
    oof = oof[["game_id", "re_pred_home", "re_pred_away",
               "p_home_win_derived"]]
    df = oof.merge(
        hist[["game_id", "season", "week", "home_team", "away_team",
              "home_score", "away_score", "home_win_prob_model",
              "home_win_prob_model_calibrated"]],
        on="game_id", how="left")
    df["binary"] = df["home_win_prob_model_calibrated"]
    df["raw"] = df["home_win_prob_model"]
    df["derived"] = df["p_home_win_derived"]
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(float)
    n_universe = int(len(df))
    assert n_universe == 1376, f"universe {n_universe} != 1376 (c1a7c12)"
    out: dict = {"universe": {"n": n_universe,
                              "n_binary_nonnull": int(df["binary"].notna().sum()),
                              "seasons": {int(k): int(v) for k, v in
                                          df["season"].value_counts().items()}}}

    # ---- pbp facts --------------------------------------------------------
    pbp = _load_pbp(PBP_SEASONS)
    st = build_starter_tables(pbp)
    prim = season_primary(st)
    st = add_backup_flags(st, prim)
    st = add_asof_quality(st)

    # per (team, season, passer) prior-season per-game mean — S-1 of the
    # passer's own season (strictly prior season, >= 5 games)
    st_sorted = st.sort_values(["posteam", "season", "passer_player_id",
                                "week"])
    ps_prev = (st_sorted.groupby(["posteam", "season", "passer_player_id"])
               ["epa_mean"].agg(["count", "mean"]).reset_index())
    ps_prev["season"] = ps_prev["season"] + 1     # value usable in S+1
    ps_prev.columns = ["posteam", "season", "passer_player_id",
                       "ps_prev_n", "ps_prev_mean"]
    st = st.merge(ps_prev, on=["posteam", "season", "passer_player_id"],
                  how="left")
    league = float(st["epa_mean"].mean())

    def _asof(row: pd.Series) -> float:
        if row["prior_n"] >= ASOF_FLOOR_PRIOR and pd.notna(row["prior_mean"]):
            return float(row["prior_mean"])
        if row["ps_prev_n"] >= PRIOR_SEASON_FLOOR and \
                pd.notna(row["ps_prev_mean"]):
            return float(row["ps_prev_mean"])
        return league

    st["asof_epa"] = st.apply(_asof, axis=1)

    # ---- join facts to the universe ---------------------------------------
    key = ["game_id", "posteam"]
    home = st.rename(columns={"posteam": "team"})[
        ["game_id", "team", "passer_player_id", "backup_op",
         "backup_season_primary", "asof_epa"]]
    home.columns = ["game_id", "team", "h_starter", "h_backup_op",
                    "h_backup_sp", "h_asof"]
    away = st.rename(columns={"posteam": "team"})[
        ["game_id", "team", "passer_player_id", "backup_op",
         "backup_season_primary", "asof_epa"]]
    away.columns = ["game_id", "team", "a_starter", "a_backup_op",
                    "a_backup_sp", "a_asof"]
    df = df.merge(home, left_on=["game_id", "home_team"],
                  right_on=["game_id", "team"], how="left").drop(
                      columns=["team"])
    df = df.merge(away, left_on=["game_id", "away_team"],
                  right_on=["game_id", "team"], how="left").drop(
                      columns=["team"])
    df["backup"] = (df["h_backup_op"] | df["a_backup_op"]).fillna(False)
    df["backup_season_primary"] = (df["h_backup_sp"] | df["a_backup_sp"])
    df["qb_gap"] = df["h_asof"] - df["a_asof"]
    abs_gap_q75 = float(df["qb_gap"].abs().quantile(0.75))

    out["facts"] = {
        "pbp_seasons": PBP_SEASONS,
        "starter_rows": int(len(st)),
        "games_with_starter_both_teams": int(
            df[["h_starter", "a_starter"]].notna().all(axis=1).sum()),
        "games_missing_any_starter": int(
            (~df[["h_starter", "a_starter"]].notna().all(axis=1)).sum()),
        "backup_op_games": int(df["backup"].sum()),
        "backup_op_frac": round(float(df["backup"].mean()), 4),
        "backup_season_primary_games": int(df["backup_season_primary"].sum()),
        "backup_season_primary_frac": round(
            float(df["backup_season_primary"].mean()), 4),
        "qb_gap_nonnull": int(df["qb_gap"].notna().sum()),
        "qb_gap_mean": round(float(df["qb_gap"].mean()), 4) if
        df["qb_gap"].notna().any() else None,
        "qb_gap_std": round(float(df["qb_gap"].std()), 4) if
        df["qb_gap"].notna().sum() > 1 else None,
        "high_gap_n": int((df["qb_gap"].abs() >= HIGH_GAP_THRESHOLD).sum()),
        "abs_gap_q75": round(abs_gap_q75, 4),
        "league_epa_mean": round(league, 4),
        "note": ("starter facts are RETROSPECTIVE actuals from PBP — "
                 "permitted only for this ceiling diagnostic; a production "
                 "build must use serve-time EXPECTED starters (train/serve "
                 "skew is the recorded risk)"),
    }
    df["high_gap"] = df["qb_gap"].abs() >= HIGH_GAP_THRESHOLD

    # ---- Phase 2 strata table --------------------------------------------
    def _stratum_rows(mask: pd.Series, label: str) -> dict:
        s = df[mask]
        c = df[~mask]
        if len(s) == 0:
            return {"label": label, "n": 0}
        g = {"label": label, "n": int(len(s)),
             "mean_platt_pred": round(float(s["binary"].mean()), 4),
             "actual_home_win": round(float(s["home_win"].mean()), 4),
             "logloss": round(_logloss(s["home_win"].to_numpy(float),
                                       s["binary"].to_numpy(float)), 4),
             "logloss_complement": round(_logloss(
                 c["home_win"].to_numpy(float),
                 c["binary"].to_numpy(float)), 4) if len(c) else None,
             "ece": round(_ece10(s["home_win"].to_numpy(float),
                                 s["binary"].to_numpy(float)), 4),
             "mean_derived_ml": round(float(s["derived"].mean()), 4),
             "mean_abs_binary_derived": round(
                 float((s["binary"] - s["derived"]).abs().mean()), 4),
             "hi_conf_0_65": _hi_conf(s, 0.65),
             "hi_conf_0_70": _hi_conf(s, 0.70)}
        return g

    def _hi_conf(s: pd.DataFrame, th: float) -> dict:
        m = s["binary"] >= th
        if m.sum() == 0:
            return {"n": 0}
        sub = s[m]
        return {"n": int(len(sub)),
                "pred_vs_actual_gap_pp": round(
                    float((sub["binary"].mean() - sub["home_win"].mean())
                          * 100.0), 2)}

    df["high_gap_topq"] = df["qb_gap"].abs() >= abs_gap_q75
    rows = [_stratum_rows(~df["backup"], "non_backup"),
            _stratum_rows(df["backup"], "backup"),
            _stratum_rows(~df["high_gap"].fillna(False), "low_gap"),
            _stratum_rows(df["high_gap"].fillna(False), "high_gap"),
            _stratum_rows(~df["high_gap_topq"].fillna(False),
                          "low_gap_top_quartile_sensitivity"),
            _stratum_rows(df["high_gap_topq"].fillna(False),
                          "high_gap_top_quartile_sensitivity")]
    # interaction (only where n allows)
    for bk, bk_lab in ((False, "non-backup"), (True, "backup")):
        for hg, hg_lab in ((False, "low-gap"), (True, "high-gap")):
            m = (df["backup"] == bk) & (df["high_gap"].fillna(False) == hg)
            if m.sum() >= 30:
                rows.append(_stratum_rows(m, f"{bk_lab} x {hg_lab}"))
    out["strata"] = rows

    # ---- Phase 3 ceiling models (in-sample, retrospective) ----------------
    from sklearn.linear_model import LogisticRegression
    m = df.dropna(subset=["binary"]).copy()
    m["backup_f"] = m["backup"].astype(float)
    m["qb_gap_f"] = m["qb_gap"].fillna(0.0)
    m["has_gap"] = m["qb_gap"].notna().astype(float)
    y = m["home_win"].to_numpy(float)
    lg = np.log(np.clip(m["binary"].to_numpy(float), 1e-6, 1 - 1e-6)
                / (1 - np.clip(m["binary"].to_numpy(float), 1e-6, 1 - 1e-6)))

    def _fit_m(xcols: list[str]) -> tuple[float, float, float]:
        X = np.column_stack([lg] + [m[c].to_numpy(float) for c in xcols])
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(X, y)
        p = lr.predict_proba(X)[:, 1]
        # R2 of (y - p) regressed on the covariates (0 for M0)
        resid = y - p
        if not xcols:
            return _logloss(y, p), 0.0, 0.0
        cov = np.column_stack([m[c].to_numpy(float) for c in xcols])
        Xs = np.column_stack([np.ones(len(cov)), cov])
        beta, *_ = np.linalg.lstsq(Xs, resid, rcond=None)
        ss_tot = float(((resid - resid.mean()) ** 2).sum())
        ss_res = float(((resid - Xs @ beta) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return _logloss(y, p), max(r2, 0.0), float(beta[0])
    m0 = _fit_m([])
    m1 = _fit_m(["backup_f"])
    m2 = _fit_m(["qb_gap_f"])
    m3 = _fit_m(["backup_f", "qb_gap_f"])
    # qb_gap coefficient sign from a direct logit(y ~ logit(p) + qb_gap)
    X2 = np.column_stack([lg, m["qb_gap_f"].to_numpy(float)])
    lr2 = LogisticRegression(C=1e6, max_iter=1000).fit(X2, y)
    out["ceiling"] = {
        "label": ("RETROSPECTIVE actual-starter facts + IN-SAMPLE "
                  "explanatory fit on the same 1,376 rows — an UPPER "
                  "BOUND, not a predicted adoption gain. A production "
                  "build must use serve-time EXPECTED starters "
                  "(depth-chart/timestamp-derived), which are noisy and "
                  "arrive later; achievable gain will be lower."),
        "n": int(len(m)),
        "M0_logit_p_only": {"logloss": round(m0[0], 4), "r2_of_resid": 0.0},
        "M1_plus_backup": {"logloss": round(m1[0], 4),
                           "r2_of_y_minus_p": round(m1[1], 4)},
        "M2_plus_qb_gap": {"logloss": round(m2[0], 4),
                           "r2_of_y_minus_p": round(m2[1], 4)},
        "M3_plus_both": {"logloss": round(m3[0], 4),
                         "r2_of_y_minus_p": round(m3[1], 4)},
        "delta_ll_M2_vs_M0": round(m2[0] - m0[0], 4),
        "delta_ll_M3_vs_M0": round(m3[0] - m0[0], 4),
        "qb_gap_coef_sign": "positive" if lr2.coef_[0][1] > 0 else "negative",
        "qb_gap_coef_value": round(float(lr2.coef_[0][1]), 4),
        "sign_interpretation": ("positive => the binary UNDER-prices home "
                                "QB advantage / OVER-prices away — the "
                                "direction a future feature must have"),
    }

    # ---- Phase 4 pre-registered verdict ----------------------------------
    bk_row = next(r for r in rows if r["label"] == "backup")
    backup_ll_delta = (bk_row["logloss"] - bk_row["logloss_complement"]
                       if bk_row["logloss_complement"] is not None else 0.0)
    delta_m2 = out["ceiling"]["delta_ll_M2_vs_M0"]
    delta_m3 = out["ceiling"]["delta_ll_M3_vs_M0"]
    r2_m3 = out["ceiling"]["M3_plus_both"]["r2_of_y_minus_p"]
    sensible_sign = out["ceiling"]["qb_gap_coef_sign"] == "positive"
    go_gate = (bk_row["n"] >= 100 and backup_ll_delta >= 0.008
               and (min(delta_m2, delta_m3) <= -0.004 or r2_m3 >= 0.02)
               and sensible_sign)
    concentration = backup_ll_delta >= 0.004
    if go_gate:
        verdict = "GO"
    elif concentration:
        verdict = "STOP-BINARY"
    else:
        verdict = "STOP-ALL"
    out["verdict"] = {
        "pre_registered_rules": {
            "GO": ("backup stratum n >= 100 AND its logloss >= +0.008 worse "
                   "than complement AND M2/M3 in-sample logloss delta >= "
                   "0.004 (or R2 >= 0.02) with a sensible coefficient "
                   "direction"),
            "STOP-BINARY": ("errors concentrate in backup/high-gap strata "
                            "but the ceiling covariate is weak (delta < "
                            "0.004)"),
            "STOP-ALL": "no concentration anywhere"},
        "backup_n": bk_row["n"],
        "backup_logloss_delta_vs_complement": round(backup_ll_delta, 4),
        "delta_ll_M2_vs_M0": delta_m2,
        "delta_ll_M3_vs_M0": delta_m3,
        "r2_M3": r2_m3,
        "qb_gap_coef_sign": out["ceiling"]["qb_gap_coef_sign"],
        "verdict": verdict,
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
