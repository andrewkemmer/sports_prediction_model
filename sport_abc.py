"""Causal A/B/C moneyline trials for NHL, NFL and MLB.

Arms
----
A  incumbent moneyline, no market offset, full active contract.
B  A minus every Elo and win-percentage name the contract serves.  The
   contracts spell these two ways, so both are dropped:
   ``elo_diff``/``elo_home``/``elo_away``/``win_pct_diff``/``win_pct_home``/
   ``win_pct_away`` (NBA/NHL/NFL) and ``home_elo``/``away_elo``/
   ``home_win_pct``/``away_win_pct`` (MLB's raw per-side twins).  MLB used to
   keep four of these because only the diff spellings were listed.
C  B plus a team-level rating.  For NBA/NHL/NFL the rating is arithmetic on
   columns B already serves; for MLB it is a fitted two-sided run model
   (``team_run_ratings``), because no served column contains a
   opponent-adjusted offence/defence split in runs.

C ratings
---------
NBA  ``5 * (pl_o_dpm + pl_d_dpm) / (projected_game_pace_total / 100)``,
     i.e. the existing team-DARKO formula.
NHL  mean of the eight served player rates (``pl_evo_*`` and ``pl_ppo_*``
     across C/L/R/D).
NFL  mean of the four served player EPA rates minus trailing defensive EPA
     allowed (lower defensive EPA is better).
MLB  seven fitted columns from a causal ridge of runs scored on side rows:
     ``off_rating_{home,away,diff}``, ``def_rating_{home,away,diff}`` and
     ``net_rating_diff``.  Signed so higher is better.

The ratings are trial-only columns.  They are registered in the process-local
known pool and the side levels are routed tree-only, exactly like the existing
trial families; no production contract is changed.

Usage
-----
    python sport_abc.py --sport mlb --out sport_abc/mlb
    python sport_abc.py --sport nhl --frame /kaggle/working/nhl.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DROP = {
    # NBA / NHL / NFL spellings
    "elo_diff", "elo_home", "elo_away",
    "win_pct_diff", "win_pct_home", "win_pct_away",
    # MLB spells the raw per-side twins this way
    "home_elo", "away_elo", "home_win_pct", "away_win_pct",
}
RATING_COLS = ["team_rating_diff", "team_rating_home", "team_rating_away"]
#: MLB's fitted two-sided run-model columns (kept local to avoid importing the
#: backend module at harness import time; ``team_run_ratings`` owns the fit).
RUN_RATING_COLS = [
    "off_rating_diff", "off_rating_home", "off_rating_away",
    "def_rating_diff", "def_rating_home", "def_rating_away",
    "net_rating_diff",
]
RUN_SIDE_RATING_COLS = [
    "off_rating_home", "off_rating_away",
    "def_rating_home", "def_rating_away",
]
ARMS = ("A", "B", "C")
EPS = 1e-7


def load_modules(sport: str) -> types.SimpleNamespace:
    """Import one backend and expose its contract/evaluation entry points."""
    if sport not in ("nba", "nhl", "nfl", "mlb"):
        raise SystemExit(f"unknown sport {sport!r}")
    backend = ROOT / f"{sport}-backend" / "backend"
    if not backend.is_dir():
        raise SystemExit(f"backend not found: {backend}")
    sys.path.insert(0, str(backend))
    if sport == "mlb":
        import config  # noqa: E402
        import features  # noqa: E402
        import frames  # noqa: E402
        import training  # noqa: E402

        # MLB has no folds.py: its walk-forward splits live in training.py.
        return types.SimpleNamespace(
            config=config, feat=features, folds=None, frames=frames,
            ml=training, contract=training, sport=sport)
    import config  # noqa: E402
    import features  # noqa: E402
    import folds  # noqa: E402
    import moneyline  # noqa: E402

    return types.SimpleNamespace(
        config=config, feat=features, folds=folds, ml=moneyline,
        contract=config, sport=sport)


def _build_nfl_frame(mods) -> pd.DataFrame:
    """The production NFL feature frame, built with the phase-1 A/B calls."""
    import ingestion
    import weather as weather_mod

    config, feat, folds_mod = mods.config, mods.feat, mods.folds
    seasons = list(getattr(config, "ALL_SEASONS", range(2016, 2027)))
    schedule = ingestion.load_schedule(seasons=seasons, use_cache=True)
    schedule = ingestion.eligible_games(schedule)
    schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce")
    # nflverse publishes the whole upcoming season.  Rows without finals are
    # not decided games: their margin is NaN, but build_game_features turns
    # ``margin > 0`` into home_win=0, so leaving them in the frame grades
    # unplayed games as away wins.  Keep only finals on or before today.
    before = len(schedule)
    schedule = schedule[schedule["gameday"].notna()].copy()
    schedule = schedule[
        schedule["gameday"] <= pd.Timestamp.today().normalize()].copy()
    for col in ("home_score", "away_score"):
        schedule[col] = pd.to_numeric(schedule[col], errors="coerce")
    schedule = schedule[schedule["home_score"].notna()
                        & schedule["away_score"].notna()].copy()
    print(f"NFL decided frame: {len(schedule)} of {before} scheduled rows "
          f"through {pd.Timestamp.today().date()}", flush=True)
    try:
        pit_weather = weather_mod.cached_weather_for_games(schedule)
    except Exception as exc:  # noqa: BLE001 - weather is allowed to stay NaN
        pit_weather = pd.DataFrame()
        print(f"weather unavailable, stays NaN: {exc}", flush=True)

    pbp = ingestion.load_pbp(seasons=seasons, use_cache=True)
    ps = ingestion.load_player_stats(
        seasons=[seasons[0] - 1] + seasons, use_cache=True)
    ngs = ingestion.load_nextgen(
        seasons=[seasons[0] - 1] + seasons, use_cache=True)
    snaps = ingestion.load_snap_counts(
        seasons=ingestion.snap_count_seasons(seasons), use_cache=True)
    ftn = ingestion.load_ftn_charting(seasons=seasons, use_cache=True)
    injuries = ingestion.load_injuries_pit(
        schedule, seasons=seasons, use_cache=True, refresh_upcoming=False)
    weekly_injuries = ingestion.load_injuries_weekly(
        seasons=seasons, use_cache=True)
    crosswalk = ingestion.load_player_id_crosswalk(use_cache=True)
    weekly_rosters = ingestion.load_weekly_rosters(
        seasons=seasons, use_cache=True)
    roster_unavailable = ingestion.roster_unavailable_table(weekly_rosters)

    frame = feat.build_game_features(
        schedule, pbp=pbp, ps=ps, ngs=ngs, snaps=snaps, ftn=ftn,
        weather=pit_weather, injuries=injuries,
        weekly_injuries=weekly_injuries, crosswalk=crosswalk,
        roster_unavailable=roster_unavailable, redistribute_phase1=False)
    return folds_mod.canonical_sort(frame, "gameday").reset_index(drop=True)


def _build_mlb_frame(mods) -> pd.DataFrame:
    """The committed decided MLB frame, with the same accessor serving uses."""
    path = ROOT / "mlb-backend" / "data_delivery" / "game_level_features.csv"
    if not path.exists():
        raise SystemExit(f"MLB decided frame not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame = mods.frames.get_decided_frame(frame)
    frame["gameday"] = frame["game_date"]
    if "season" not in frame.columns:
        frame["season"] = frame["game_date"].dt.year
    return frame.sort_values("gameday", kind="mergesort").reset_index(drop=True)


def build_frame(mods, sport: str) -> pd.DataFrame:
    if sport in ("nba", "nhl"):
        import ab_delay as ad
        frame = ad.build_frame(mods, sport)
    elif sport == "nfl":
        frame = _build_nfl_frame(mods)
    else:
        frame = _build_mlb_frame(mods)
    if "gameday" not in frame.columns:
        raise SystemExit("feature frame has no gameday")
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="coerce")
    if getattr(frame["gameday"].dt, "tz", None) is not None:
        frame["gameday"] = frame["gameday"].dt.tz_localize(None)
    if mods.folds is not None:
        frame = mods.folds.canonical_sort(frame, "gameday")
    else:
        frame = frame.sort_values("gameday", kind="mergesort")
    frame = frame.reset_index(drop=True)
    frame = frame[frame["gameday"].notna()].reset_index(drop=True)
    if not len(frame):
        raise SystemExit("feature frame is empty")
    return frame


def _mean_columns(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [c for c in columns if c in frame.columns]
    if not available:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    parts = [pd.to_numeric(frame[c], errors="coerce") for c in available]
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def attach_team_rating(frame: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Attach the trial team rating triple without touching a contract."""
    out = frame.copy()
    if sport == "nba":
        if all(c in out.columns for c in
               ("team_darko_home", "team_darko_away", "team_darko_diff")):
            for src, dst in zip(
                    ("team_darko_home", "team_darko_away", "team_darko_diff"),
                    RATING_COLS[1:] + RATING_COLS[:1]):
                out[dst] = pd.to_numeric(out[src], errors="coerce")
        else:
            pace = pd.to_numeric(
                out.get("projected_game_pace_total", np.nan),
                errors="coerce") / 100.0
            for side in ("home", "away"):
                off = pd.to_numeric(out.get(f"pl_o_dpm_{side}", np.nan),
                                    errors="coerce")
                dfn = pd.to_numeric(out.get(f"pl_d_dpm_{side}", np.nan),
                                    errors="coerce")
                total = off + dfn
                out[f"team_rating_{side}"] = np.where(
                    np.isfinite(pace) & (pace > 0) & np.isfinite(total),
                    5.0 * total / pace, np.nan)
    elif sport == "nhl":
        cols = [f"pl_evo_{p}_{side}" for p in ("c", "l", "r", "d")
                for side in ("home", "away")]
        cols += [f"pl_ppo_{p}_{side}" for p in ("c", "l", "r", "d")
                 for side in ("home", "away")]
        for side in ("home", "away"):
            side_cols = [c for c in cols if c.endswith(f"_{side}")]
            out[f"team_rating_{side}"] = _mean_columns(out, side_cols)
    elif sport == "nfl":
        for side in ("home", "away"):
            off = _mean_columns(
                out, [f"pl_epa_{p}_{side}" for p in ("qb", "wr", "te", "rb")])
            defn = pd.to_numeric(
                out.get(f"pbp_def_epa_play_ewm_{side}", np.nan),
                errors="coerce")
            out[f"team_rating_{side}"] = off - defn
    elif sport == "mlb":
        # Fitted, not recombined: the run model refits on a causal grid and
        # returns its own seven columns.  There is no ``team_rating_*`` for
        # MLB; arm C is the run-model family alone.
        from team_run_ratings import attach_run_ratings  # noqa: E402
        return attach_run_ratings(out)
    else:
        raise SystemExit(f"no team-rating definition for {sport}")

    out["team_rating_diff"] = (pd.to_numeric(out["team_rating_home"],
                                             errors="coerce")
                               - pd.to_numeric(out["team_rating_away"],
                                               errors="coerce"))
    missing = [c for c in RATING_COLS if c not in out.columns]
    if missing:
        raise SystemExit(f"team-rating build is missing {missing}")
    coverage = {c: float(out[c].notna().mean()) for c in RATING_COLS}
    if coverage["team_rating_home"] <= 0.0:
        raise SystemExit(f"team-rating build is empty: {coverage}")
    print("team rating coverage: "
          + ", ".join(f"{c} {v:.1%}" for c, v in coverage.items()),
          flush=True)
    return out


def register_trial_columns(mods, sport: str) -> None:
    """Make the trial rating legal for subset validation and member routing."""
    contract = mods.contract
    known = getattr(contract, "KNOWN_FEATURE_COLS")
    columns = RUN_RATING_COLS if sport == "mlb" else RATING_COLS
    for col in columns:
        if col not in known:
            known.append(col)
    raw = getattr(contract, "RAW_PER_SIDE_COLS")
    side = (set(RUN_SIDE_RATING_COLS) if sport == "mlb"
            else {"team_rating_home", "team_rating_away"})
    if isinstance(raw, set):
        contract.RAW_PER_SIDE_COLS = frozenset(set(raw) | side)
    elif isinstance(raw, frozenset):
        contract.RAW_PER_SIDE_COLS = frozenset(set(raw) | side)
    else:
        for col in side:
            if col not in raw:
                raw.append(col)


def arm_columns(mods, sport: str) -> dict[str, list[str]]:
    active = list(mods.contract.active_moneyline_feature_cols())
    base = [c for c in active if c not in DROP]
    extra = RUN_RATING_COLS if sport == "mlb" else RATING_COLS
    return {
        "A": active,
        "B": base,
        "C": base + [c for c in extra if c not in base],
    }


def _set_arm(mods, columns: list[str] | None) -> None:
    if columns is None:
        mods.contract.reset_feature_subset()
    else:
        mods.contract.set_feature_subset(columns)


def _walk(sport: str, mods, frame: pd.DataFrame, folds: list,
          max_folds: int = 0) -> pd.DataFrame:
    if sport == "mlb":
        _models, _metrics, oof = mods.ml.walk_forward_evaluate(
            frame, force_retrain=False, max_eval_folds=max_folds)
        out = oof.copy()
        out["gameday"] = pd.to_datetime(
            out.get("game_date", out.get("gameday")), errors="coerce")
        p = out.get("home_win_prob_model")
        if p is None:
            p = out.get("home_win_prob_model_calibrated")
        out["p"] = pd.to_numeric(p, errors="coerce")
        out["p_calibrated"] = pd.to_numeric(
            out.get("home_win_prob_model_calibrated", out["p"]),
            errors="coerce")
    else:
        kwargs = dict(fold_list=folds, progress_every=0)
        if sport == "nba":
            kwargs["arm"] = "incumbent"
        result = mods.ml.walk_forward_oof(frame, **kwargs)
        out = result["oof"].copy()
        out["p"] = pd.to_numeric(out["p_ensemble"], errors="coerce")
        out["p_calibrated"] = pd.to_numeric(
            out.get("p_ensemble_calibrated", out["p"]), errors="coerce")
    out["game_id"] = out["game_id"].astype(str)
    out["home_win"] = pd.to_numeric(out["home_win"], errors="coerce")
    out["grades_pooled"] = (
        out["grades_pooled"].astype(bool) if "grades_pooled" in out
        else out["home_win"].notna())
    keep = ["game_id", "gameday", "home_win", "fold_id", "grades_pooled",
            "p", "p_calibrated"]
    return out[[c for c in keep if c in out.columns]].copy()


def _score(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if not len(y):
        return {"n": 0, "auc": np.nan, "brier": np.nan,
                "logloss": np.nan, "accuracy": np.nan}
    out = {
        "n": int(len(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy": float(np.mean((p >= 0.5) == (y >= 0.5))),
        "auc": np.nan,
    }
    if len(np.unique(y)) == 2:
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score(y, p))
    return out


def _pair_delta(merged: pd.DataFrame, left: str, right: str,
                reps: int, seed: int) -> dict:
    """Weekly-cluster bootstrap of right minus left on grading rows."""
    sub = merged[merged["grades_A"]].copy()
    y = pd.to_numeric(sub["home_win"], errors="coerce").to_numpy(float)
    pl = pd.to_numeric(sub[f"p_{left}"], errors="coerce").to_numpy(float)
    pr = pd.to_numeric(sub[f"p_{right}"], errors="coerce").to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(pl) & np.isfinite(pr)
    sub, y, pl, pr = sub.loc[ok], y[ok], pl[ok], pr[ok]
    if not len(sub):
        return {"n": 0}

    def ll(yy, pp):
        pp = np.clip(pp, EPS, 1 - EPS)
        return -(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))

    d = ll(y, pr) - ll(y, pl)
    weeks = (pd.to_datetime(sub["gameday"], errors="coerce")
             .dt.to_period("W-SUN").astype(str).to_numpy())
    uniq = np.array(sorted(set(weeks)))
    idx = {w: np.flatnonzero(weeks == w) for w in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(reps)):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx[w] for w in take])
        draws.append(float(d[rows].mean()))
    draws = np.asarray(draws, float)
    return {
        "n": int(len(d)),
        "delta": float(d.mean()),
        "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan,
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "better_share": float(np.mean(draws < 0)),
        "n_boot": int(len(draws)),
    }


def evaluate(sport: str, mods, frame: pd.DataFrame, out_dir: Path,
             reps: int, seed: int, max_folds: int = 0) -> dict:
    if sport == "mlb":
        folds = []
        print(f"{sport}: {len(frame)} games, training.walk_forward_evaluate",
              flush=True)
    else:
        folds = mods.folds.make_folds(frame, "gameday")
        if max_folds:
            folds = folds[:max_folds]
        print(f"{sport}: {len(frame)} games, {len(folds)} folds", flush=True)
    arms = arm_columns(mods, sport)
    print("arms: " + ", ".join(f"{k}={len(v)}" for k, v in arms.items()),
          flush=True)

    pieces: list[pd.DataFrame] = []
    meta: dict[str, dict] = {}
    for label in ARMS:
        started = time.time()
        _set_arm(mods, None if label == "A" else arms[label])
        try:
            active = list(mods.contract.active_moneyline_feature_cols())
            print(f"[{label}] {len(active)} active features", flush=True)
            oof = _walk(sport, mods, frame, folds, max_folds=max_folds)
        finally:
            _set_arm(mods, None)
        oof = oof.rename(columns={
            "p": f"p_{label}", "p_calibrated": f"p_{label}_cal",
            "grades_pooled": f"grades_{label}",
        })
        oof.to_csv(out_dir / f"oof_{label}.csv", index=False)
        pieces.append(oof)
        meta[label] = {
            "n_features": len(active),
            "n_oof": int(len(oof)),
            "n_graded": int(oof[f"grades_{label}"].sum()),
            "seconds": round(time.time() - started, 1),
        }
        print(f"[{label}] done {meta[label]}", flush=True)

    # Merge only the arm-specific columns.  Carrying fold_id (present in the
    # NHL/NFL/NBA OOFs but not MLB's combined frame) through repeated outer
    # merges creates fold_id_dup twice and pandas rejects it.  Labels and the
    # date come from arm A; every arm is aligned on game_id alone.
    merged = pieces[0][["game_id", "gameday", "home_win",
                        "p_A", "p_A_cal", "grades_A"]].copy()
    for label, part in zip(ARMS[1:], pieces[1:]):
        merged = merged.merge(
            part[["game_id", f"p_{label}", f"p_{label}_cal",
                  f"grades_{label}"]],
            on="game_id", how="outer")
    merged["grades_A"] = merged["grades_A"].fillna(False).astype(bool)
    merged.to_csv(out_dir / "oof_merged.csv", index=False)

    rows = []
    masks = {
        "all": merged["grades_A"].to_numpy(bool),
        "band_40_60": (merged["grades_A"].to_numpy(bool)
                       & merged["p_A"].between(0.40, 0.60).to_numpy()),
        "band_50_60": (merged["grades_A"].to_numpy(bool)
                       & merged["p_A"].between(0.50, 0.60).to_numpy()),
    }
    for subset, mask in masks.items():
        for arm in ARMS:
            score = _score(merged.loc[mask, f"p_{arm}"].to_numpy(float),
                           merged.loc[mask, "home_win"].to_numpy(float))
            rows.append({"subset": subset, "arm": arm, **score})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    deltas = []
    for left, right in (("A", "B"), ("B", "C"), ("A", "C")):
        row = _pair_delta(merged, left, right, reps, seed)
        row.update({"left": left, "right": right})
        deltas.append(row)
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(out_dir / "deltas.csv", index=False)

    summary = {
        "sport": sport,
        "frame": {
            "games": int(len(frame)),
            "first": str(frame["gameday"].min().date()),
            "last": str(frame["gameday"].max().date()),
            "folds": int(len(folds)) if folds else None,
        },
        "drop": sorted(DROP),
        "drop_present": sorted(DROP & set(arms["A"])),
        "drop_absent": sorted(DROP - set(arms["A"])),
        "rating_columns": (RUN_RATING_COLS if sport == "mlb" else RATING_COLS),
        "arms": meta,
        "metrics": metrics.to_dict(orient="records"),
        "paired_deltas": deltas.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sport", required=True,
                    choices=("nba", "nhl", "nfl", "mlb"))
    ap.add_argument("--out", default="")
    ap.add_argument("--frame", default="", help="optional cached frame")
    ap.add_argument("--save-frame", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-folds", type=int, default=0,
                    help="smoke only: first N folds")
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore", category=UserWarning)
    out_dir = Path(args.out or f"sport_abc/{args.sport}")
    out_dir.mkdir(parents=True, exist_ok=True)
    mods = load_modules(args.sport)

    if args.frame and Path(args.frame).exists():
        frame = pd.read_parquet(args.frame)
        print(f"loaded frame cache {args.frame}: {frame.shape}", flush=True)
    else:
        started = time.time()
        frame = build_frame(mods, args.sport)
        print(f"frame built in {(time.time() - started) / 60:.1f} min: "
              f"{frame.shape}", flush=True)
        if args.save_frame:
            frame.to_parquet(Path(args.frame or out_dir / "frame.parquet"),
                             index=False)

    frame = attach_team_rating(frame, args.sport)
    register_trial_columns(mods, args.sport)
    evaluate(args.sport, mods, frame, out_dir, args.bootstrap, args.seed,
             max_folds=args.max_folds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
