"""Edge-vs-close slice for the O-PTS arms (saved OOF + Stats Insider TAB lines).

Reads the one-snapshot-per-game TAB decimal H2H odds scraped by
scrape_statsinsider_nba.py (si_nba_{2023,2024,2025}.csv), de-vigs each game
multiplicatively to a fair home prob (same de-vig the market_rundown
consensus uses per book), joins to the saved A/B/C paired OOF, and prints
the logloss/brier/auc/ece view for arm A and arm B restricted to
|p_A - mkt| > 3% (repo edge convention), plus paired B-A / C-A on that slice.

Join hygiene: SI's date_utc is the preview-publish stamp (usually gameday - 1,
mixed ISO formats), NOT tipoff, so dates never join. The anchor is
(home, away, final home score, final away score) against the graded frame's
ESPN-sourced actuals (stage 1), with |si_date - gameday| <= 2 days as a guard
against cross-season score collisions. SI's stored finals are wrong for ~6%
of games, so stage 2 recovers those by (teams, si_date in {gameday-1, gameday})
with a strict exactly-one-candidate rule (NBA pairs never meet on consecutive
days, so the window is collision-free -- verified zero multi-candidate
games). Final scores are join keys only -- they never enter any feature or
probability. Early-published previews (Christmas/Cup games stamped up to 14d
early) are a different estimand (edge vs opener, not close) and stay excluded.

Usage: python ab_o_pts_edge.py   (from nba-backend/backend)
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

import config
from evaluation import binary_metrics
from ab_o_pts import _paired

EDGE = 0.03
MAX_OFF_DAYS = 2


def _load_si() -> pd.DataFrame:
    d = Path(config.RUN_DIAGNOSTICS_DIR)
    frames = []
    for season in (2023, 2024, 2025):
        p = d / f"si_nba_{season}.csv"
        if not p.exists():
            raise SystemExit(f"missing {p} -- run scrape_statsinsider_nba.py first")
        frames.append(pd.read_csv(p))
    si = pd.concat(frames, ignore_index=True)
    print(f"SI rows: {len(si)}", flush=True)
    priced = si.dropna(subset=["home_odds_dec", "away_odds_dec"]).copy()
    print(f"SI priced rows: {len(priced)} "
          f"({len(priced) / max(len(si), 1):.1%})", flush=True)
    h = 1.0 / priced["home_odds_dec"].astype(float)
    a = 1.0 / priced["away_odds_dec"].astype(float)
    priced["mkt"] = h / (h + a)
    print(f"SI median overround: {(h + a).median():.4f}", flush=True)
    # Mixed ISO formats in the wild ("...+00:00" and naive); format='mixed'
    # handles both. Naive stamps are UTC (SI is an AU source, midday UTC =
    # evening AEDT publish, morning US).
    priced["si_date"] = pd.to_datetime(
        priced["date_utc"], format="mixed", utc=True).dt.date.astype(str)
    assert priced["si_date"].notna().all(), "unparseable SI date_utc values"
    return priced


def _load_oof() -> pd.DataFrame:
    d = Path(config.RUN_DIAGNOSTICS_DIR)
    oof = pd.read_csv(d / "ab_o_pts_paired_oof.csv")
    oof["game_id"] = oof["game_id"].astype(str)
    feat = pd.read_parquet(
        d / "ab_o_pts_A_contract_features.parquet",
        columns=["game_id", "gameday", "home_team", "away_team",
                 "home_score", "away_score", "home_win"])
    feat["game_id"] = feat["game_id"].astype(str)
    m = oof.merge(feat, on="game_id", how="left",
                  suffixes=("", "_feat"),
                  validate="one_to_one")
    assert m["home_team"].notna().all(), "OOF games missing from features"
    m["gameday"] = pd.to_datetime(m["gameday"]).dt.date.astype(str)
    return m


def _join(si: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    # Stage 1: score-anchored (close-vintage guard).
    cand = si.merge(
        m[["game_id", "gameday", "home_team", "away_team",
           "home_score", "away_score"]],
        left_on=["home_abv", "away_abv",
                 "final_home_score", "final_away_score"],
        right_on=["home_team", "away_team", "home_score", "away_score"],
        how="inner")
    print(f"stage-1 score-matching SI rows: {len(cand)}", flush=True)
    cand["off"] = (pd.to_datetime(cand["si_date"])
                   - pd.to_datetime(cand["gameday"])).dt.days.abs()
    print(f"stage-1 beyond {MAX_OFF_DAYS}d guard (dropped, early "
          f"openers): {int((cand['off'] > MAX_OFF_DAYS).sum())}", flush=True)
    cand = cand[cand["off"] <= MAX_OFF_DAYS]
    cand = cand.sort_values("off").drop_duplicates("game_id", keep="first")
    assert not (cand.groupby("game_id").size() > 1).any()
    got = set(cand["game_id"])
    print(f"stage-1 matched games: {len(got)}", flush=True)

    # Stage 2: SI-error recovery -- SI's stored finals disagree with actuals,
    # so anchor on (teams, close-vintage publish window) instead. The window
    # is collision-free for NBA scheduling; enforced, not assumed.
    rest = m[~m["game_id"].isin(got)].copy()
    rest["gd_m1"] = (pd.to_datetime(rest["gameday"])
                     - pd.Timedelta(days=1)).dt.date.astype(str)
    c2 = si.merge(rest[["game_id", "gameday", "gd_m1",
                        "home_team", "away_team"]],
                  left_on=["home_abv", "away_abv"],
                  right_on=["home_team", "away_team"], how="inner")
    c2 = c2[(c2["si_date"] == c2["gameday"])
            | (c2["si_date"] == c2["gd_m1"])]
    n_cand = c2.groupby("game_id").size()
    assert not (n_cand > 1).any(), \
        f"stage-2 collisions: {n_cand[n_cand > 1].index.tolist()}"
    c2 = c2.drop_duplicates("game_id", keep="first")
    print(f"stage-2 recovered games: {len(c2)} "
          f"(of {len(rest)} stage-1 misses)", flush=True)
    c2["off"] = (pd.to_datetime(c2["si_date"])
                 - pd.to_datetime(c2["gameday"])).dt.days.abs()

    both = pd.concat([
        cand.assign(src="score")[["game_id", "mkt", "si_date", "off", "src",
                                  "home_odds_dec", "away_odds_dec"]],
        c2.assign(src="teams_date")[["game_id", "mkt", "si_date", "off", "src",
                                     "home_odds_dec", "away_odds_dec"]],
    ])
    j = m.merge(both, on="game_id", how="left", validate="one_to_one")
    print(f"matched grading games: {j['mkt'].notna().sum()}/{len(m)}",
          flush=True)
    if j["mkt"].isna().any():
        print(j[j["mkt"].isna()][
            ["game_id", "gameday", "away_team", "home_team",
              "away_score", "home_score"]].to_string(index=False))
    return j.dropna(subset=["mkt"]).reset_index(drop=True)


def _view(df: pd.DataFrame, tag: str) -> None:
    y = df["home_win"].to_numpy(float)
    print(f"\n--- {tag}: n={len(df)} ---")
    for arm in ("A", "B", "C"):
        mt = binary_metrics(df[f"p_{arm}"].to_numpy(float), y)
        print(f"  {arm}: logloss {mt['logloss']:.5f}  brier {mt['brier']:.5f}  "
              f"auc {mt['auc']:.5f}  ece {mt['ece']:.5f}")
    mk = binary_metrics(df["mkt"].to_numpy(float), y)
    print(f"  mkt: logloss {mk['logloss']:.5f}  brier {mk['brier']:.5f}  "
          f"auc {mk['auc']:.5f}  ece {mk['ece']:.5f}")


def main() -> None:
    si = _load_si()
    m = _load_oof()
    j = _join(si, m)
    j["edge_A"] = j["p_A"] - j["mkt"]
    print(f"\nedge>3% rate: {(j['edge_A'].abs() > EDGE).mean():.1%} "
          f"(by join src: "
          + ", ".join(f"{s}={(j['src'] == s).mean():.1%}"
                      for s in ("score", "teams_date")) + ")",
          flush=True)
    _view(j, "all matched")
    sub = j[j["edge_A"].abs() > EDGE].reset_index(drop=True)
    _view(sub, f"edge>3% (|p_A - mkt| > {EDGE})")
    y = sub["home_win"].to_numpy(float)
    _paired(y, sub["p_A"].to_numpy(float), sub["p_B"].to_numpy(float),
            sub["gameday"], "B_plus6 - A_contract (edge>3%)")
    _paired(y, sub["p_A"].to_numpy(float), sub["p_C"].to_numpy(float),
            sub["gameday"], "C_replace - A_contract (edge>3%)")
    out = Path(config.RUN_DIAGNOSTICS_DIR) / "ab_o_pts_edge_paired_oof.csv"
    j[["game_id", "gameday", "home_win", "p_A", "p_B", "p_C", "mkt",
       "edge_A", "src"]].to_csv(out, index=False)
    print(f"\n  edge-joined OOF written to {out}")


if __name__ == "__main__":
    main()
