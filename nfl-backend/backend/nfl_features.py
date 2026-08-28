"""NFL feature engineering v1 — leakage-safe raw candidates + admission gate.

Builds on the committed game-level frame produced by ``nfl_game_frame.py``
(``nfl-backend/data_delivery/nfl_game_level_features.csv``). This module is
the feature-ADMISSION stage only: it proposes raw candidates, audits coverage
and (point-in-time) leakage, and gates them into a v1 set. It does NOT train
any model — walk-forward / ensemble / sealed-holdout is the NEXT task.

Leakage discipline (MLB retrospective lessons — "raw-not-clever, pre-game
coverage rule, gated entry, no model-output-as-input"):
- Every trailing feature is a function ONLY of that team's games with
  ``gameday`` STRICTLY BEFORE the target game. Enforced by chronological sort
  + per-team window shift + an explicit per-team strict-monotonicity assertion
  in :func:`team_stats_ladder`.
- No feature uses market lines, model probabilities, or later results. No
  hand-multiplied "risk" interactions, no injury reports (not reliably final
  12h pre-kickoff), no weather.

12h-pre-kickoff availability assumption (stated): a feature counts as
"available 12h pre-kickoff" iff it is non-null and depends only on completed
prior games or a static venue/prior fact. Nothing here depends on live
intraday state, so availability == non-null coverage for every candidate.

Sources
-------
- Game-level frame: committed ``nfl_game_frame.py`` output (2019-2024 decided).
- Schedule (for ``roof`` -> ``is_dome_home``): nflreadpy ``load_schedules``.
- Play-by-play (for net yards/play): nflreadpy ``load_pbp``.
- ``WARMUP_SEASONS`` (2018) is pulled for the SAME sources purely so the first
  2019 games have clean trailing priors; only the 2019-2024 decided games are
  scored/reported. 2025 is not in the frame at all (untouched future holdout).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent          # nfl-backend/
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"
DECIDED_FRAME = DATA_DELIVERY_DIR / "nfl_game_level_features.csv"

WARMUP_SEASONS = [2018]                          # trailing priors only
CORE_SEASONS = list(range(2019, 2025))           # 2019..2024 scored/reported
DEFAULT_SEASONS = WARMUP_SEASONS + CORE_SEASONS

# ELO (prior + update rule, fully specified / reproducible)
ELO_PRIOR = 1500.0
ELO_K = 32.0                                     # standard logistic gain
ELO_SCALE = 400.0

# trailing windows
FORM_WINDOW = 4       # net pts/game window
WINPCT_WINDOW = 12    # trailing win% window
YPP_WINDOW = 5        # net yards/play window

# admission gate
COVERAGE_FLOOR = 0.95
# The redundancy bar is the reporting bar the user specified ("|r| > 0.8"):
# a feature ~83% correlated with a slightly-stronger one is redundant enough to
# prune (measured elo_diff ~ win_pct_diff r = 0.826 -> keep elo, drop win_pct).
CORR_REDUNDANCY = 0.80
DISC_BAND = 0.05                                  # |auc - 0.5| below = ~random

DATE_FMT = "%Y%m%d"
RECORD_TEMPLATE = f"nfl_feature_v1_{{date}}.json"

# deterministic keep-order for redundant pairs (lower = kept first)
FEATURE_PRIORITY = {
    "elo_diff": 0, "ypp_diff": 1, "form_diff_pts": 2, "win_pct_diff": 3,
    "rest_days_diff": 4, "is_dome_home": 5,
}

FEATURE_COLUMNS = [
    "elo_diff", "form_diff_pts", "win_pct_diff", "rest_days_diff",
    "ypp_diff", "is_dome_home", "is_home",
]

CANONICAL_SOURCE = {
    "elo_diff": "ELO prior 1500, K=32, strictly-prior games",
    "form_diff_pts": "trailing net pts/game (last 4)",
    "win_pct_diff": "trailing win% (last 12)",
    "rest_days_diff": "days since each team's prior game",
    "ypp_diff": "trailing net yards/play (last 5, from pbp)",
    "is_dome_home": "home venue roof (nflverse schedule field)",
    "is_home": "constant anchor for the home edge",
}


# ---------------------------------------------------------------------------
# Core primitives (pure; testable without network)
# ---------------------------------------------------------------------------
def team_events(game: pd.DataFrame) -> pd.DataFrame:
    """Long-form one-row-per-(team,game) view used by all trailing features.

    Adds, from the team's perspective: ``team``, ``opponent``, ``is_home``,
    ``net_from_team`` (score diff, + for that team), ``team_win`` (1/0/0.5 tie).
    """
    required = ["game_id", "season", "week", "gameday", "home_team",
                "away_team", "home_score", "away_score"]
    missing = [c for c in required if c not in game.columns]
    if missing:
        raise ValueError(f"team_events: missing columns {missing}")

    gd = pd.to_datetime(game["gameday"], errors="coerce")
    home = pd.DataFrame({
        "game_id": game["game_id"], "season": game["season"], "week": game["week"],
        "gameday": gd, "team": game["home_team"], "opponent": game["away_team"],
        "is_home": True,
        "for": game["home_score"].astype(float), "against": game["away_score"].astype(float),
    })
    away = pd.DataFrame({
        "game_id": game["game_id"], "season": game["season"], "week": game["week"],
        "gameday": gd, "team": game["away_team"], "opponent": game["home_team"],
        "is_home": False,
        "for": game["away_score"].astype(float), "against": game["home_score"].astype(float),
    })
    ev = pd.concat([home, away], ignore_index=True)
    ev["net_from_team"] = ev["for"] - ev["against"]
    ev["team_win"] = np.select(
        [ev["for"] > ev["against"], ev["for"] < ev["against"]],
        [1.0, 0.0], default=0.5)
    return ev


def compute_elo(events: pd.DataFrame) -> pd.DataFrame:
    """Attach ``elo_entering`` to each (team,event) row: the team's rating at
    kickoff, from ONLY games strictly before this game's gameday.

    Update rule: expected = 1/(1+10**((r_opp - r_self)/400));  r += K*(actual-exp).
    actual = win(1)/loss(0)/tie(0.5). Prior = 1500. Iterated strictly
    chronologically, so a future game can never feed an earlier rating.
    """
    K, prior, scale = ELO_K, ELO_PRIOR, ELO_SCALE
    ev = events.sort_values(["gameday", "game_id", "is_home"]).reset_index(drop=True)
    rating: dict = {}
    entering: dict = {}
    for game_id, rows in ev.groupby("game_id", sort=False):
        rows = list(rows.itertuples(index=False))
        a, b = rows[0], rows[1] if len(rows) == 2 else rows[0]
        ra, rb = rating.get(a.team, prior), rating.get(b.team, prior)
        entering[(game_id, a.team)] = ra
        entering[(game_id, b.team)] = rb
        exp_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / scale))
        exp_b = 1.0 / (1.0 + 10.0 ** ((ra - rb) / scale))
        rating[a.team] = ra + K * (a.team_win - exp_a)
        rating[b.team] = rb + K * (b.team_win - exp_b)
    ev = ev.copy()
    ev["elo_entering"] = ev.apply(
        lambda r: entering.get((r["game_id"], r["team"]), prior), axis=1)
    return ev


def _trailing_per_team(srt: pd.DataFrame, value_col: str, window: int) -> np.ndarray:
    """Per-team windowed mean of ``value_col`` over STRICTLY-PRIOR games.

    ``srt`` must be sorted by (team, gameday, game_id). Rolling mean per team,
    then a per-team shift(1) drops the current row, so each value is the mean
    over that team's prior ``window`` games only. Returned in ``srt`` row order.
    """
    roll = srt.groupby("team", sort=False)[value_col].rolling(
        window, min_periods=1).mean()
    roll = roll.groupby(level=0).shift(1)
    return roll.reset_index(level=0, drop=True).to_numpy()


def team_stats_ladder(events: pd.DataFrame,
                      ypp_game: pd.DataFrame | None = None) -> pd.DataFrame:
    """For every (game_id, team): elo_entering, form_pts (prior net pts/gm),
    win_pct (prior), rest_days (days since the team's previous game), ypp
    (prior net yards/play). ``ypp_game``: (game_id, team, total_yards, n_plays).

    LEAKAGE GATE: after sorting by (team, gameday, game_id), gameday must be
    strictly increasing within each team. Combined with the per-team shift, no
    future game can touch any row's trailing statistics. This is asserted.
    """
    ev = events.copy()
    if ypp_game is not None:
        ygp = ypp_game.rename(columns={"total_yards": "tot_yd", "n_plays": "npl"})
        ygp["ypp_game"] = ygp["tot_yd"] / ygp["npl"].replace(0, np.nan)
        ev = ev.merge(ygp.drop(columns=["tot_yd", "npl"]),
                      on=["game_id", "team"], how="left")

    srt = ev.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)

    diffs = srt.groupby("team", sort=False)["gameday"].diff()
    bad = srt.loc[(diffs.notna()) & (diffs <= pd.Timedelta(0))]
    if len(bad):
        raise AssertionError(
            f"team_stats_ladder: team gameday not strictly increasing -> trailing "
            f"features could reference non-prior games ({len(bad)} rows)")

    srt["form_pts"] = _trailing_per_team(srt, "net_from_team", FORM_WINDOW)
    srt["win_pct"] = _trailing_per_team(srt, "team_win", WINPCT_WINDOW)
    srt["rest_days"] = srt.groupby("team", sort=False)["gameday"].diff().dt.days
    if "ypp_game" in srt.columns:
        srt["ypp"] = _trailing_per_team(srt, "ypp_game", YPP_WINDOW)
    else:
        srt["ypp"] = np.nan
    return srt


# ---------------------------------------------------------------------------
# Feature composition
# ---------------------------------------------------------------------------
def _home_minus_away(ladder: pd.DataFrame, game_ids: pd.Index,
                     col: str) -> np.ndarray:
    home = ladder[ladder["is_home"]].set_index("game_id")[col]
    away = ladder[~ladder["is_home"]].set_index("game_id")[col]
    return (home.reindex(game_ids) - away.reindex(game_ids)).to_numpy()


def build_features(decided: pd.DataFrame,
                   schedule: pd.DataFrame | None = None,
                   pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compose candidate feature columns on a decided game frame.

    The trailing/ELO ladder is computed over ALL decided games present in the
    *schedule* (warmup + core, e.g. 2018-2024) so the earliest scored games get
    real priors; each ladder value is attached to its ``game_id`` row, shifted
    so it uses only strictly-prior games. ``decided`` (2019-2024) is the frame
    scored/reported.
    """
    # --- full decided timeline across warmup+core seasons, from the schedule ---
    sched = schedule.copy() if schedule is not None else decided.copy()
    for c in ("home_score", "away_score"):
        if c in sched.columns:
            sched[c] = pd.to_numeric(sched[c], errors="coerce")
    full = sched[pd.to_numeric(sched["home_score"], errors="coerce").notna() &
                 pd.to_numeric(sched["away_score"], errors="coerce").notna()].copy()

    events = compute_elo(team_events(full))
    ypp_game = None
    if pbp is not None and {"yards_gained", "posteam"}.issubset(pbp.columns):
        p = pbp[["game_id", "posteam", "yards_gained"]].dropna(subset=["posteam"])
        p = p.groupby(["game_id", "posteam"], as_index=False).agg(
            total_yards=("yards_gained", "sum"), n_plays=("yards_gained", "count"))
        ypp_game = p.rename(columns={"posteam": "team"})
    ladder = team_stats_ladder(events, ypp_game)

    df = decided.copy()
    # --- dome flag directly from the feed's roof (dome/closed = indoor) ---
    if schedule is not None and "roof" in schedule.columns:
        roof = schedule[["game_id", "roof"]].drop_duplicates("game_id")
        df = df.merge(roof, on="game_id", how="left")
    if "roof" not in df.columns:
        df["roof"] = np.nan
    df["is_dome_home"] = np.where(
        df["roof"].isin(["dome", "closed"]), 1.0,
        np.where(df["roof"].isin(["outdoors"]), 0.0, np.nan))

    gids = df["game_id"]
    df["is_home"] = 1.0                                   # anchor for the home edge
    df["elo_diff"] = _home_minus_away(ladder, gids, "elo_entering")
    df["form_diff_pts"] = _home_minus_away(ladder, gids, "form_pts")
    df["win_pct_diff"] = _home_minus_away(ladder, gids, "win_pct")
    df["rest_days_diff"] = _home_minus_away(ladder, gids, "rest_days")
    df["ypp_diff"] = _home_minus_away(ladder, gids, "ypp")
    return df


# ---------------------------------------------------------------------------
# Audit + admission gate
# ---------------------------------------------------------------------------
def audit_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """% non-null per candidate + 12h-availability (== coverage, per the stated
    assumption that every candidate is a function of prior games or a static
    venue/prior fact)."""
    rows = {}
    for f in FEATURE_COLUMNS:
        cov = float(df[f].notna().mean())
        rows[f] = {
            "coverage_pct": round(100 * cov, 2),
            "available_12h_pct": round(100 * cov, 2),
            "source": CANONICAL_SOURCE[f],
        }
    return pd.DataFrame(rows).T


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (shared with scipy.stats.rankdata 'average')."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty(n, dtype=np.intp)
    inv[sorter] = np.arange(n)
    xs = x[sorter]
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks[inv]


def univariate_auc(home_win: np.ndarray, feature: np.ndarray) -> float:
    """P(feature of a win > feature of a loss), mean tie=0.5. AUC>0.5 ->
    higher feature -> home win. NaN if either class absent."""
    y = np.asarray(home_win)
    x = np.asarray(feature, dtype=float)
    mask = ~np.isnan(x) & np.isfinite(x)
    y, x = y[mask], x[mask]
    pos = x[y == 1]
    neg = x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = _rankdata(np.concatenate([pos, neg]))
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) \
        / (len(pos) * len(neg))


def audit_correlation(df: pd.DataFrame) -> pd.DataFrame:
    num = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return num.corr()


def _strong_pairs(corr: pd.DataFrame) -> list[dict]:
    seen = set()
    out = []
    for f in corr.columns:
        for g in corr.columns:
            if f >= g or (f, g) in seen or (g, f) in seen:
                continue
            v = corr[f][g]
            if pd.notna(v) and abs(v) > 0.8:
                seen.add((f, g))
                out.append({"feat_a": f, "feat_b": g, "corr": round(float(v), 4)})
    return out


def run_feature_gate(df: pd.DataFrame) -> dict:
    """Coverage floor, then redundant-pair / near-random-AUC pruning (no model)."""
    covered = [f for f in FEATURE_COLUMNS
               if float(df[f].notna().mean()) >= COVERAGE_FLOOR]

    pre = df[df["season"] < 2025]               # 2025 is the untouched holdout
    y = (pre["home_score"] > pre["away_score"]).astype(int).to_numpy()
    auc = {}
    for f in FEATURE_COLUMNS:
        x = pre[f].to_numpy()
        if pre[f].nunique(dropna=True) <= 1:    # constant -> no discriminative info
            auc[f] = float("nan")
        else:
            auc[f] = univariate_auc(y, x)
    corr = audit_correlation(df)

    v1 = list(FEATURE_COLUMNS)
    reasons = {}
    # R0 coverage floor
    for f in FEATURE_COLUMNS:
        if f not in covered:
            if f in v1:
                v1.remove(f)
            reasons[f] = (f"coverage {float(df[f].notna().mean()):.1%} below "
                          f"{COVERAGE_FLOOR:.0%} floor")
    # R1 near-random AND redundant with a stronger feature -> prune
    for f in list(v1):
        if f == "is_home" or f not in auc or pd.isna(auc[f]):
            continue
        if abs(auc[f] - 0.5) < DISC_BAND:
            for g in v1:
                if g == f or g == "is_home" or g not in auc or pd.isna(auc[g]):
                    continue
                if abs(corr[f][g]) > CORR_REDUNDANCY and \
                        abs(auc[g] - 0.5) > abs(auc[f] - 0.5) + 1e-9:
                    if f in v1:
                        v1.remove(f)
                        reasons[f] = (f"auc {auc[f]:.3f} ~ random and |r| "
                                      f"{abs(corr[f][g]):.2f} with {g} "
                                      f"(stronger discriminator)")
                    break
    # R2 redundant pair (|r| large, similar discrimination) -> keep one
    kept = set()
    for f in list(v1):
        if f in kept:
            continue
        for g in list(v1):
            if g == f or g in kept or f not in auc or g not in auc or \
                    pd.isna(auc[f]) or pd.isna(auc[g]):
                continue
            if abs(corr[f][g]) > CORR_REDUNDANCY and \
                    abs(abs(auc[f] - 0.5) - abs(auc[g] - 0.5)) < DISC_BAND:
                disc_f, disc_g = abs(auc[f] - 0.5), abs(auc[g] - 0.5)
                strong, weak = f, g
                if disc_g > disc_f:
                    strong, weak = g, f
                elif disc_f == disc_g and \
                        FEATURE_PRIORITY.get(weak, 99) < FEATURE_PRIORITY.get(strong, 99):
                    strong, weak = weak, strong
                if weak in v1:
                    v1.remove(weak)
                    reasons[weak] = (f"redundant with {strong} (|r| "
                                     f"{abs(corr[f][g]):.2f}, similar discrimination); "
                                     f"keep one")
                kept.add(strong)
                kept.add(weak)

    is_home_in = "is_home" in v1          # constant anchor stays in v1 set
    return {
        "covered_features": covered,
        "v1_features": v1,
        "kept_home_anchor": is_home_in,
        "audit_coverage": audit_coverage(df).to_dict(orient="index"),
        "univariate_auc": {k: (None if pd.isna(v) else round(float(v), 4))
                           for k, v in auc.items()},
        "correlation_pairs_over_0_8": _strong_pairs(corr),
        "reasons": reasons,
        "dropped": [f for f in FEATURE_COLUMNS if f not in v1],
    }


# ---------------------------------------------------------------------------
# Loaders + orchestration
# ---------------------------------------------------------------------------
def _load_raw(seasons: list[int]):
    import nflreadpy
    sched = nflreadpy.load_schedules(seasons).to_pandas()
    pbp = nflreadpy.load_pbp(seasons)
    cols = [c for c in ["game_id", "posteam", "yards_gained"] if c in pbp.columns]
    pbp = pbp.select(cols).to_pandas()
    return sched, pbp


def pull_and_build(out_dir: Path | None = None,
                   write_record: bool = True) -> dict:
    out_dir = Path(out_dir) if out_dir is not None else DATA_DELIVERY_DIR
    if not DECIDED_FRAME.exists():
        raise FileNotFoundError(
            f"{DECIDED_FRAME} absent — run `python3 nfl_game_frame.py` first")
    decided = pd.read_csv(DECIDED_FRAME)

    logger.info("Loading nflreadpy schedule+pbp (warmup+core): %s", DEFAULT_SEASONS)
    schedule, pbp = _load_raw(DEFAULT_SEASONS)
    feats = build_features(decided, schedule, pbp)

    result = run_feature_gate(feats)

    if write_record:
        record = {
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "build": {
                "core_seasons": CORE_SEASONS,
                "warmup_seasons": WARMUP_SEASONS,
                "decided_games": int(len(feats)),
                "decided_frame": str(DECIDED_FRAME),
                "elo_prior": ELO_PRIOR, "elo_k": ELO_K,
                "windows": {"form": FORM_WINDOW, "win_pct": WINPCT_WINDOW,
                            "ypp": YPP_WINDOW},
                "leakage_rule": ("every trailing feature uses only games with "
                                 "gameday strictly before the target (asserted in "
                                 "code: team_stats_ladder strict monotonicity)."),
                "holdout": "2025 is not in the decided frame; all AUC computed on "
                           "seasons < 2025.",
            },
            "feature_admission": result,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        rec_path = out_dir / RECORD_TEMPLATE.format(date=datetime.now().strftime(DATE_FMT))
        with open(rec_path, "w") as fh:
            json.dump(record, fh, indent=2)
        result["record"] = str(rec_path)

    _print_report(feats, result)
    return result


def _print_report(feats: pd.DataFrame, result: dict) -> None:
    print("\n=== NFL feature admission v1 (no model) ===")
    print(f"decided games scored: {len(feats)}")
    cov = pd.DataFrame(result["audit_coverage"]).T
    print("\ncoverage / 12h availability:")
    print(cov[["coverage_pct", "available_12h_pct", "source"]].to_string())
    print("\nstrong correlation pairs (|r| > 0.8):")
    for p in result.get("correlation_pairs_over_0_8", []):
        print(f"  {p['feat_a']} ~ {p['feat_b']}: r={p['corr']}")
    print("\nunivariate AUC (seasons < 2025):")
    for f, v in result["univariate_auc"].items():
        print(f"  {f:16s} {v if v is not None else 'n/a'}")
    print("\nv1 features:", result["v1_features"])
    print("dropped:", result.get("dropped"))
    for f, r in result.get("reasons", {}).items():
        print(f"  drop {f}: {r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        description="Build + gate NFL feature candidates v1 (no model).")
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON record")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    pull_and_build(write_record=not args.no_record)
    return 0


if __name__ == "__main__":
    sys.exit(main())