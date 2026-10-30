"""Holdout A/B/C: trailing vs DARKO expected-minutes weights, waterfall vs vanish.

  A (incumbent production): trailing `prior_minutes_per_game`, reweight="waterfall".
  B (treatment):             `darko_x_minutes` where finite and > 0 else trailing, reweight="waterfall".
  C (vanish):                same weights as B, reweight="legacy" (no redistribution).

Everything else held constant: same folds (built once on A), same game frame,
same ratings (DARKO DPM swap + x_minutes attached once), same
`apply_pit_designations` removal, same hyperparameters. Ranking moves with the
flag (top-k cut by `_w` when expected present) -- documented as part of the
B treatment, not a separate arm.

Full cumulative walk-forward production pipeline: load_ingested ->
build_game_features -> build_player_rapm + DARKO attach -> project_lineup_features
(production path, nine pl_dpm_* + six star_out_*) -> walk_forward_oof per arm
with identical folds -> paired per-game log-loss with gameday-blocked bootstrap.

Run where the cache lives:
    python nba-backend/backend/ab_expected_minutes.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("ab_expected_minutes")

from ab_position_rapm import (  # noqa: E402
    POSITION_FEATURES,
    score,
)


def _paired_delta(y, p_base, p_treat, gamedays, label_b, label_t, seed=42):
    eps = 1e-12

    def per_game(p):
        p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    d = per_game(p_treat) - per_game(p_base)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    t = mean / se if se and np.isfinite(se) and se != 0 else float("nan")
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"gameday": pd.to_datetime(gamedays), "_i": np.arange(n)})
    blocks = [part["_i"].to_numpy() for _, part in df.groupby("gameday")]
    boot = np.array([d[np.concatenate(
        [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])].mean()
        for _ in range(4000)])
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    win = int((d < 0).sum())
    print(f"\n=== paired delta ({label_t} - {label_b}), n={n}")
    print(f"  mean {mean:+.5f}  se {se:.5f}  t {t:+.2f}  "
          f"95% CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  verdict: {'distinguishable' if lo * hi > 0 else 'NOT distinguishable'}")
    print(f"  {label_t} better on {win}/{n} ({100 * win / max(n, 1):.1f}%)")
    return {"label": f"{label_t}-{label_b}", "n": n, "mean": mean,
            "se": se, "t": t, "lo": lo, "hi": hi, "win": win}


def main() -> None:
    import config
    import folds as folds_mod
    import features as feat_mod
    import ingestion
    import lineup_projection as proj_mod
    import moneyline as ml_mod

    logger.info("loading the fact set from cache (allow_download=False)")
    facts = ingestion.load_ingested(allow_download=False)
    games = facts.games
    settled = ingestion.trainable_games(games)
    logger.info("settled eligible games: %d", len(settled))
    game_df = feat_mod.build_game_features(
        settled, facts.team_stats, facts.team_events)
    game_df = folds_mod.canonical_sort(game_df, "gameday")
    base = game_df.copy()

    designations = proj_mod.load_designations(
        Path(os.environ.get("NBA_CACHE_DIR")
             or os.path.expanduser("~/.cache/sports_prediction_model/nba")))
    if designations is None:
        designations = proj_mod.load_designations(config.DATA_DELIVERY_DIR)
    logger.info("designations: %s",
                "none" if designations is None else f"{len(designations)} rows")

    positions = pd.concat([
        ingestion._fetch_positions(season)
        for season in sorted({ingestion.season_label(d.date())
                              for d in pd.to_datetime(games.gameday).dropna()})
    ], ignore_index=True)

    import player_rapm as rapm_mod
    import darko_source
    games_frame = rapm_mod.prepare_player_games(
        facts.player_stats, positions)
    dates = pd.Series(sorted(pd.to_datetime(settled.gameday).dropna().unique()))
    logger.info("building ratings for %d target dates", len(dates))
    ratings = rapm_mod.build_player_rapm(
        games_frame, target_dates=dates, team_stats=facts.team_stats,
        roster_history=ingestion.load_roster_history())
    ids = rapm_mod._player_key(ratings["player_id"]).dropna().unique()
    history = darko_source.fetch_darko_history(ids)
    ratings = rapm_mod.apply_darko_rating(ratings, history)
    logger.info("x coverage: %.3f",
                float(ratings["darko_x_minutes"].notna().mean()))

    name_by_id = dict(zip(
        rapm_mod._player_id_str(facts.player_stats.player_id),
        facts.player_stats.player_name.astype(str)))

    treat_cols = list(POSITION_FEATURES) + list(proj_mod.STAR_OUT_FEATURES)
    arms_spec = (
        ("A_trailing_waterfall", False, "waterfall"),
        ("B_expected_waterfall", True, "waterfall"),
        ("C_expected_vanish", True, "legacy"),
    )

    arms = {}
    for name, flag, reweight in arms_spec:
        config.PLAYER_USE_DARKO_EXPECTED_MINUTES = flag
        rated = ratings.rename(columns={"target_date": "gameday"})
        rated["gameday"] = pd.to_datetime(rated["gameday"])
        if "is_available" not in rated.columns:
            rated["is_available"] = True
        # Production path: designations -> projection -> nine + six.
        frame, aggregates = proj_mod.project_lineup_features(
            rated, games=base, designations=designations,
            name_by_id=name_by_id, reweight=reweight)
        # base already carries stale/NaN copies of these columns from the
        # ladder; drop before the join so the arm's values land bare.
        fb = frame.drop(columns=[c for c in treat_cols if c in frame.columns],
                        errors="ignore")
        # project_lineup_features attaches onto `games=base`, which already
        # has the ladder columns; re-merge from `frame` keyed on game.
        key = ["gameday", "home_team", "away_team"]
        fb = fb.merge(
            frame[key + [c for c in treat_cols if c in frame.columns]]
            .drop_duplicates(subset=key),
            on=key, how="left")
        arms[name] = fb
        # Persist the arm's feature frame so post-hoc slices (star-out,
        # low-confidence tail) can join predictions without rebuilding.
        frame_path = Path(config.RUN_DIAGNOSTICS_DIR) / f"ab_expected_minutes_{name}_features.parquet"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        fb.to_parquet(frame_path, index=False)
        logger.info("%s: feature frame written to %s", name, frame_path)
        cov = float(pd.to_numeric(
            fb["pl_dpm_c_diff"], errors="coerce").notna().mean())
        logger.info("%s (expected=%s reweight=%s): pl_dpm_c_diff coverage %.3f, "
                    "rows %d", name, flag, reweight, cov, len(fb))

    # Same folds for all arms, built once on the incumbent arm.
    fold_list = folds_mod.make_folds(arms["A_trailing_waterfall"])
    logger.info("folds: %d", len(fold_list))
    baseline_cols = [c for c in config.MONEYLINE_FEATURE_COLS
                     if c not in set(POSITION_FEATURES)]
    results = {}
    for name, _, _ in arms_spec:
        joined = arms[name]
        cols = baseline_cols + POSITION_FEATURES
        cols = [c for c in cols if c in joined.columns]
        config.set_feature_subset(cols)
        started = time.time()
        result = ml_mod.walk_forward_oof(joined, fold_list=fold_list,
                                         progress_every=0)
        oof = result["oof"]
        grading = oof[oof.grades_pooled.astype(bool)]
        metrics = score(grading.p_ensemble_calibrated.to_numpy(float),
                        grading.home_win.to_numpy(float))
        weights = result.get("member_weights", {})
        results[name] = (metrics, oof)
        print(f"\n=== {name}: {len(cols)} features, {len(oof)} OOF, "
              f"{len(grading)} grading, {time.time() - started:.0f}s")
        for key in ("logloss", "accuracy", "brier", "auc", "ece", "n"):
            if key in metrics:
                print(f"  {key:<10} {metrics[key]:.5f}")
        print(f"  weights   {({k: round(float(v), 4) for k, v in weights.items()})}")
        config.reset_feature_subset()

    names = [n for n, _, _ in arms_spec]
    print("\n=== side by side (pooled grading logloss)")
    hdr = f"  {'metric':<10} " + " ".join(f"{n:>22}" for n in names)
    print(hdr)
    for key in ("logloss", "accuracy", "brier", "auc", "ece"):
        row = f"  {key:<10} "
        for n in names:
            m = results[n][0]
            row += f"{m.get(key, float('nan')):>22.5f} " if key in m else f"{'n/a':>22} "
        print(row)
        if key == "logloss":
            base = results[names[0]][0].get(key)
            for n in names[1:]:
                v = results[names[1]][0].get(key) if False else results[n][0].get(key)
                if base is not None and v is not None and np.isfinite(base) and np.isfinite(v):
                    print(f"    delta {n}-{names[0]}: {v - base:+.5f}")

    # Paired per-game log-loss on the intersection of graded games.
    oofs = {}
    for n in names:
        o = results[n][1]
        o = o[o.grades_pooled].copy()
        oofs[n] = o[["game_id", "gameday", "home_win", "p_ensemble_calibrated"]]
    merged = oofs[names[0]].rename(columns={"p_ensemble_calibrated": f"p_{names[0]}"})
    for n in names[1:]:
        merged = merged.merge(
            oofs[n][["game_id", "p_ensemble_calibrated"]].rename(
                columns={"p_ensemble_calibrated": f"p_{n}"}),
            on="game_id", how="inner")
    y = merged.home_win.to_numpy(float)
    gd = pd.to_datetime(merged.gameday)
    out_rows = []
    pairs = [(names[0], names[1]), (names[1], names[2]), (names[0], names[2])]
    for lb, lt in pairs:
        r = _paired_delta(y, merged[f"p_{lb}"].to_numpy(float),
                          merged[f"p_{lt}"].to_numpy(float), gd, lb, lt,
                          seed=config.RANDOM_SEED)
        out_rows.append(r)
    out = Path(config.RUN_DIAGNOSTICS_DIR) / "ab_expected_minutes_abc_oof.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"\n  paired OOF (all three arms, inner join) written to {out}")
    for n in names:
        p = Path(config.RUN_DIAGNOSTICS_DIR) / f"ab_expected_minutes_{n}_oof.csv"
        results[n][1].to_csv(p, index=False)
        print(f"  {n} full OOF written to {p}")
    # Star-out + confidence slices, joined from the persisted arm frames.
    try:
        feat_b = pd.read_parquet(
            Path(config.RUN_DIAGNOSTICS_DIR) / "ab_expected_minutes_B_expected_waterfall_features.parquet",
            columns=["game_id", "star_out_weight_share_diff",
                     "star_out_impact_diff"])
        sm = merged.merge(feat_b, on="game_id", how="left")
        share = pd.to_numeric(sm["star_out_weight_share_diff"], errors="coerce").fillna(0.0)
        impact = pd.to_numeric(sm["star_out_impact_diff"], errors="coerce").fillna(0.0)
        yg = sm.home_win.to_numpy(float)
        gdg = pd.to_datetime(sm.gameday)
        eps2 = 1e-12

        def _pg(p):
            p = np.clip(np.asarray(p, dtype=float), eps2, 1 - eps2)
            return -(yg * np.log(p) + (1 - yg) * np.log(1 - p))

        dBA_g = _pg(sm["p_B_expected_waterfall"].to_numpy(float)) - _pg(sm["p_A_trailing_waterfall"].to_numpy(float))
        dCB_g = _pg(sm["p_C_expected_vanish"].to_numpy(float)) - _pg(sm["p_B_expected_waterfall"].to_numpy(float))
        conf_b = np.abs(sm["p_A_trailing_waterfall"].to_numpy(float) - 0.5)
        lo10 = float(np.quantile(conf_b, 0.10))
        slices = {
            "starout_nonzero_share": (share.abs() > 1e-9).to_numpy(),
            "starout_nonzero_impact": (impact.abs() > 1e-9).to_numpy(),
            "lowconf_decile": (conf_b <= lo10),
        }
        for sname, mask in slices.items():
            idx = np.where(mask)[0]
            if len(idx) < 30:
                print(f"\n--- slice {sname}: n={len(idx)} too thin, skipped")
                continue
            for lbl, dd in (("B-A", dBA_g[idx]), ("C-B", dCB_g[idx])):
                n = len(dd)
                m = float(dd.mean())
                se = float(dd.std(ddof=1) / np.sqrt(n))
                t = m / se if se else float("nan")
                print(f"\n--- slice {sname} {lbl}: n={n} mean={m:+.5f} se={se:.5f} t={t:+.2f} "
                      f"win={int((dd < 0).sum())}/{n} ({100 * (dd < 0).mean():.1f}%)")
    except Exception as exc:  # noqa: BLE001 - slices are diagnostic, never fatal
        print(f"\n(slice analysis skipped: {type(exc).__name__}: {exc})")
    print("\nAdopt bar: B-A paired t <= -2.5 with star-out slice and tail not "
          "regressing; then C-vs-B decides waterfall-vs-vanish on x-weights.")


if __name__ == "__main__":
    main()
