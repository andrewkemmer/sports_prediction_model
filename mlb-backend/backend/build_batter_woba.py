"""Build point-in-time batter/team season-to-date wOBA tables from Statcast pbp.

Phase 2 (lineup-delta moneyline features) — the "hard part" with no lookahead.

Inputs:  data_delivery/pbp_chunks/pbp_*.parquet (lean Statcast pbp, one row per
         pitch; each PA ends with a non-null `events` value). Chunks cover
         2025-03-18 → 2026-08-24 (regular season + 2025 postseason).

Outputs:
  data_delivery/batter_woba.parquet
    One row per (season, game_date, batter) the batter PLAYED. Columns:
      season, game_date, batter, sd_woba, prior_pa, last_team
    sd_woba  = season-to-date wOBA through games STRICTLY BEFORE game_date
               (same-season only — the season opener's row excludes all prior
               seasons; same discipline as the momentum season-baseline fix).
    prior_pa = cumulative PAs before game_date (min-PA guard input).
    last_team= the team the batter batted for on the previous date they played
               (team membership as of game_date, no lookahead).
  data_delivery/team_woba.parquet
    One row per (season, game_date, team). Columns:
      season, game_date, team, sd_woba, prior_pa, top3_woba
    sd_woba   = team season-to-date wOBA through prior dates (exact, from the
                team's own PAs).
    top3_woba = mean sd_woba of the team's top-3 regulars (prior_pa >= 20, on
                the team as of game_date) through prior dates. NaN when fewer
                than 3 regulars qualify.
    top5_ids  = JSON list of the team's top-5 regulars' batter IDs (prior_pa
                >= 50, on the team as of game_date), by sd_woba — the
                lineup_rest_count input. Empty JSON list when fewer than 5.

wOBA weights: MLB official 2024 fixed weights, documented here and used for
both seasons (the task calls for fixed, documented weights):
  0.690 uBB, 0.722 HBP, 0.888 1B, 1.271 2B, 1.616 3B, 2.101 HR
  wOBA = (wBB*uBB + wHBP*HBP + w1B*1B + w2B*2B + w3B*3B + wHR*HR)
         / (AB + BB - IBB + SF + HBP)
  with AB = PA - BB - HBP - SF - SH - CI, so the denominator collapses to
  PA - IBB - SH - CI (only PA, IBB, SH, CI need counting — no AB subtleties).

Spring-training/exhibition pbp (game_type S/E/L) is excluded; regular season
and postseason (R/D/F/W) are kept, matching the training CSV's 2025 range
(2025-03-18 → 2025-11, incl. the 2025 postseason).

Usage:
    python build_batter_woba.py          # rebuild both tables (idempotent)
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd

DATA_DELIVERY_DIR = Path(__file__).resolve().parent.parent / "data_delivery"

# ── wOBA weights (MLB official 2024 fixed weights) ────────────────────────────
W_BB, W_HBP, W_1B, W_2B, W_3B, W_HR = 0.690, 0.722, 0.888, 1.271, 1.616, 2.101

# event → component bucket (uBB/1b/2b/3b/hr/ibb/hbp/sf/sh/ci; None = out/
# other, which contributes to PA but not the wOBA numerator). Every PA ends
# with exactly one event. Names are this feed's pybaseball/Statcast spellings
# (sac_fly / sac_bunt / intent_walk / catcher_interf).
_EVENT_COMPONENTS = {
    "single": "1b", "double": "2b", "triple": "3b", "home_run": "hr",
    "walk": "uBB", "intent_walk": "ibb", "hit_by_pitch": "hbp",
    "sac_fly": "sf", "sac_fly_double_play": "sf",
    "sac_bunt": "sh",
    "catcher_interf": "ci",
}

# game types to keep (regular season + postseason; drop spring/exhibition)
_KEEP_GAME_TYPES = ("R", "D", "F", "W")


def _load_pbp() -> pd.DataFrame:
    files = sorted(glob.glob(str(DATA_DELIVERY_DIR / "pbp_chunks" / "pbp_*.parquet")))
    if not files:
        sys.exit(f"no pbp chunks in {DATA_DELIVERY_DIR / 'pbp_chunks'}")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(_KEEP_GAME_TYPES)]
    df = df[df["events"].notna()].copy()  # one row per PA (final pitch)
    df["season"] = df["game_date"].dt.year
    return df


def _pa_level(df: pd.DataFrame) -> pd.DataFrame:
    """Per-PA rows → component counts; batting team from inning_topbot."""
    comp = df["events"].map(lambda e: _EVENT_COMPONENTS.get(e))
    out = pd.DataFrame({"season": df["season"].values,
                        "game_date": df["game_date"].values,
                        "batter": df["batter"].values})
    for col in ("uBB", "hbp", "1b", "2b", "3b", "hr", "ibb", "sh", "ci"):
        # .values: out is RangeIndex but comp carries df's (gappy) index —
        # pandas would silently align and null out mismatched rows
        out[col] = (comp == col).astype("int8").values
    out["team"] = pd.Series(
        df["away_team"].where(df["inning_topbot"] == "Top", df["home_team"]).values,
        index=out.index)
    return out


def _point_in_time(agg: pd.DataFrame, value_cols: list[str],
                   count_col: str) -> pd.DataFrame:
    """Prefix sums over STRICTLY earlier dates, per (season, key).

    agg: one row per (season, key..., game_date) with value_cols + count_col.
    Returns one row per (season, key..., game_date) with prior_<v> and
    prior_<count> = sums over dates < game_date (all of a date's games count
    as "before" the next date; nothing from prior seasons leaks in because the
    season partition is the outermost group).
    """
    keys = [c for c in agg.columns
            if c not in ("season", "game_date", *value_cols, count_col)]
    per_date = (agg.groupby(["season", *keys, "game_date"], as_index=False)[
        value_cols + [count_col]].sum())
    per_date = per_date.sort_values(["season", *keys, "game_date"])
    g = per_date.groupby(["season", *keys], sort=False)
    for v in value_cols:
        per_date[f"prior_{v}"] = g[v].cumsum() - per_date[v]
    per_date[f"prior_{count_col}"] = g[count_col].cumsum() - per_date[count_col]
    return per_date


def build() -> None:
    df = _load_pbp()
    print(f"pbp PAs: {len(df):,}")
    pa = _pa_level(df)

    # ── per (season, date, batter) components ────────────────────────────────
    batter_day = pa.groupby(["season", "game_date", "batter"], as_index=False).agg(
        uBB=("uBB", "sum"), hbp=("hbp", "sum"),
        **{k: (k, "sum") for k in ("1b", "2b", "3b", "hr", "ibb", "sh", "ci")},
        pa=("uBB", "size"),
        last_team=("team", "last"))
    batter_day["num"] = (
        W_BB * batter_day["uBB"] + W_HBP * batter_day["hbp"]
        + W_1B * batter_day["1b"] + W_2B * batter_day["2b"]
        + W_3B * batter_day["3b"] + W_HR * batter_day["hr"])
    batter_day["den"] = (batter_day["pa"] - batter_day["ibb"]
                         - batter_day["sh"] - batter_day["ci"])

    b_prior = _point_in_time(
        batter_day[["season", "game_date", "batter", "num", "den", "pa"]],
        ["num", "den"], "pa")
    # last-known team: the team they batted for on the previous date they played
    lt = (batter_day.sort_values(["season", "batter", "game_date"])
          .groupby(["season", "batter"])["last_team"].shift(1)
          .rename("last_team"))
    b_prior = b_prior.sort_values(["season", "batter", "game_date"])
    b_prior["last_team"] = lt.values
    b_prior["sd_woba"] = (b_prior["prior_num"] / b_prior["prior_den"]
                          .where(b_prior["prior_den"] > 0))
    b_out = b_prior[["season", "game_date", "batter", "sd_woba", "prior_pa",
                     "last_team"]].copy()
    b_out = b_out.sort_values(["season", "game_date", "batter"]).reset_index(drop=True)
    b_out.to_parquet(DATA_DELIVERY_DIR / "batter_woba.parquet", index=False)
    print(f"batter_woba.parquet: {len(b_out):,} rows "
          f"({b_out['sd_woba'].notna().mean():.1%} non-null sd_woba)")

    # ── team point-in-time (exact team wOBA from the team's own PAs) ─────────
    t_agg = pa.groupby(["season", "game_date", "team"], as_index=False).agg(
        num=("uBB", lambda s: W_BB * s.sum()),
        hbp=("hbp", "sum"),
        **{k: (k, "sum") for k in ("1b", "2b", "3b", "hr", "ibb", "sh", "ci")})
    t_agg["num"] = t_agg["num"] + W_HBP * t_agg["hbp"] + W_1B * t_agg["1b"] \
        + W_2B * t_agg["2b"] + W_3B * t_agg["3b"] + W_HR * t_agg["hr"]
    # PA count = number of PA rows in the group
    t_pa = pa.groupby(["season", "game_date", "team"], as_index=False).size()
    t_agg = t_agg.merge(t_pa.rename(columns={"size": "pa"}),
                        on=["season", "game_date", "team"], how="left")
    t_agg["den"] = t_agg["pa"] - t_agg["ibb"] - t_agg["sh"] - t_agg["ci"]
    t_prior = _point_in_time(
        t_agg[["season", "game_date", "team", "num", "den", "pa"]],
        ["num", "den"], "pa")
    t_prior["sd_woba"] = (t_prior["prior_num"] / t_prior["prior_den"]
                          .where(t_prior["prior_den"] > 0))

    # ── team top-3 / top-5 regulars as of each date (no lookahead) ───────────
    # For every (season, team, date) take each regular's MOST RECENT sd_woba
    # before that date (merge_asof per batter — a single merge_asof against
    # all regulars would return only the most recent row, i.e. one batter),
    # then rank the team's regulars by wOBA for that date.
    def _top_regulars(b_out: pd.DataFrame, t_prior: pd.DataFrame,
                      pa_floor: int, k: int):
        """{(season, team, date): [(batter_id, woba), ...]} top-k by wOBA."""
        reg = b_out[(b_out["prior_pa"] >= pa_floor)] \
            .dropna(subset=["sd_woba", "last_team"])
        reg = reg.rename(columns={"game_date": "reg_date",
                                  "sd_woba": "reg_woba"})
        out: dict[tuple, list[tuple[int, float]]] = {}
        for (season, team), g in t_prior.groupby(["season", "team"], sort=False):
            dates = g["game_date"].drop_duplicates() \
                .sort_values().reset_index(drop=True)
            ddf = pd.DataFrame({"game_date": dates})
            long_rows = []
            for batter, br in reg[(reg["season"] == season)
                                  & (reg["last_team"] == team)].groupby("batter"):
                m = pd.merge_asof(
                    ddf, br[["reg_date", "reg_woba"]].sort_values("reg_date"),
                    left_on="game_date", right_on="reg_date",
                    direction="backward")
                m["batter"] = batter
                long_rows.append(m[["game_date", "batter", "reg_woba"]])
            if not long_rows:
                continue
            long_df = pd.concat(long_rows, ignore_index=True)
            for date, g2 in long_df.groupby("game_date"):
                pairs = [(int(b), w) for b, w in zip(g2["batter"], g2["reg_woba"])
                         if pd.notna(w)]
                pairs.sort(key=lambda t: -t[1])
                out[(season, team, date)] = pairs[:k]
        return out

    top3 = _top_regulars(b_out, t_prior, pa_floor=20, k=3)
    top5 = _top_regulars(b_out, t_prior, pa_floor=50, k=5)
    t_out = t_prior.sort_values(["season", "game_date", "team"]).copy()
    t_out["top3_woba"] = pd.NA
    t_out["top5_ids"] = "[]"
    for (season, team, date), pairs in top3.items():
        mask = ((t_out["season"] == season) & (t_out["team"] == team)
                & (t_out["game_date"] == date))
        ws = [w for _, w in pairs]
        if len(ws) >= 3:
            t_out.loc[mask, "top3_woba"] = sum(ws[:3]) / 3
    for (season, team, date), pairs in top5.items():
        mask = ((t_out["season"] == season) & (t_out["team"] == team)
                & (t_out["game_date"] == date))
        t_out.loc[mask, "top5_ids"] = json.dumps([b for b, _ in pairs])
    t_out["top3_woba"] = pd.to_numeric(t_out["top3_woba"], errors="coerce")
    t_out = t_out.sort_values(["season", "game_date", "team"]).reset_index(drop=True)
    t_out.to_parquet(DATA_DELIVERY_DIR / "team_woba.parquet", index=False)
    print(f"team_woba.parquet: {len(t_out):,} rows "
          f"({t_out['top3_woba'].notna().mean():.1%} non-null top3_woba)")


if __name__ == "__main__":
    build()
