"""Walk-forward retrain arm K: trailing own-error location correction.

K0 (control): production distribution (flag off) on a fresh score-ensemble
  walk-forward built from the local cache.
K1 (treatment): identical score-ensemble OOF, distribution with
  DISTRIBUTION_LOCATION_BIAS_CORRECT=True (per-team trailing bias block,
  margin-preserving shift, calibration refits against the corrected base).

Same folds for the score stage (one OOF, shared); distribution origins are
deterministic given the frame, so both arms see identical origin splits.
Graded at SI market total lines (oof_with_market.csv join on game_id):
two-way push-excluded logloss, edge/agreement slices, |edge| ladder, AUC,
gameday-blocked bootstrap paired K1-K0.

Adopt bar: K1-K0 paired 95% CI excludes 0 (K1 better) AND agreement slice
must not regress AND 10%+ bucket improves AND AUC not regressing.
Otherwise GPC stands and the correction stays a trial.

Run where the cache lives (hours; background + log):
    nohup python nba-backend/backend/ab_location_retrain.py > nba-backend/run_diagnostics/ab_location_retrain.log 2>&1 &
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
logger = logging.getLogger("ab_location_retrain")


def _paired(y, p_base, p_treat, gamedays, label, seed=42):
    eps = 1e-12

    def per_game(p):
        p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    d = per_game(p_treat) - per_game(p_base)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"gameday": pd.to_datetime(gamedays), "_i": np.arange(n)})
    blocks = [part["_i"].to_numpy() for _, part in df.groupby("gameday")]
    boot = np.array([d[np.concatenate(
        [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])].mean()
        for _ in range(2000)])
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    print(f"\n=== paired delta ({label}), n={n}", flush=True)
    print(f"  mean {mean:+.5f}  se {se:.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]", flush=True)
    print(f"  verdict: {'distinguishable' if lo * hi > 0 else 'NOT distinguishable'}", flush=True)
    return mean, se, lo, hi, n


def _grade_at_line(priced, market, tag):
    j = priced.merge(market, on="game_id", how="inner")
    # priced may already carry gameday (passed through the causal frame),
    # colliding with market's -> gameday_x/gameday_y. Coalesce to one.
    if "gameday" not in j.columns:
        for _gx, _gy in (("gameday_x", "gameday_y"),):
            if _gx in j.columns:
                j["gameday"] = j[_gx].where(j[_gx].notna(), j.get(_gy))
                break
    print(f"{tag}: priced={len(priced)} market-join={len(j)}", flush=True)
    j["total"] = j["home_score"] + j["away_score"]
    j["U"] = pd.to_numeric(j["total_line"], errors="coerce")

    def lab(u):
        return str(int(u)) if float(u).is_integer() else f"{int(u)}_5"

    j["p_raw"] = j.apply(lambda r: r.get("p_over_" + lab(r["U"])), axis=1)
    j = j[np.isfinite(pd.to_numeric(j["p_raw"], errors="coerce"))].copy()
    whole = (j["U"] % 1 == 0)
    push = np.zeros(len(j))
    pv = j[whole].apply(lambda r: r.get("p_push_total_" + lab(r["U"]), np.nan), axis=1).to_numpy(float)
    push[np.where(whole)[0]] = pv
    j["push"] = push
    j["p_model"] = np.where(whole, j["p_raw"] / (j["p_raw"] + (1 - j["p_raw"] - j["push"])), j["p_raw"])
    j["push_game"] = (j["total"] == j["U"])
    ev = j[~j["push_game"]].copy()
    ev["y"] = (ev["total"] > ev["U"]).astype(float)
    ev["p_mkt"] = 0.5
    ev["ae"] = (ev["p_model"] - 0.5).abs()
    clip = lambda p: np.clip(p, 1e-9, 1 - 1e-9)
    ev["llm"] = -(ev["y"] * np.log(clip(ev["p_model"])) + (1 - ev["y"]) * np.log(1 - clip(ev["p_model"])))
    ev["llk"] = -(ev["y"] * np.log(0.5) + (1 - ev["y"]) * np.log(0.5))
    ev["d"] = ev["llm"] - ev["llk"]
    return ev


def main() -> None:
    import config
    import folds as folds_mod
    import features as feat_mod
    import ingestion
    import distributions as dist_mod

    t0 = time.time()
    logger.info("loading the fact set from cache (allow_download=False)")
    facts = ingestion.load_ingested(allow_download=False)
    games = facts.games
    settled = ingestion.trainable_games(games)
    logger.info("settled eligible games: %d", len(settled))
    game_df = feat_mod.build_game_features(
        settled, facts.team_stats, facts.team_events)
    game_df = folds_mod.canonical_sort(game_df, "gameday")
    logger.info("feature frame: %d rows", len(game_df))

    ckpt = Path(config.RUN_DIAGNOSTICS_DIR) / "ab_location_retrain_score_oof.parquet"
    if ckpt.exists():
        logger.info("score-OOF checkpoint found (%s) -- resuming", ckpt)
        frame = pd.read_parquet(ckpt)
    else:
        logger.info("score-ensemble walk-forward (shared by both arms)")
        ens = dist_mod.walk_forward_oof_ensemble(game_df, progress_every=10)
        oof = ens["oof"]
        logger.info("score OOF: %d rows, weights %s (%.0fs)",
                    len(oof), ens.get("member_weights"), time.time() - t0)
        # team keys for the trial block (score OOF keeps game_id only)
        teams = game_df[["game_id", "gameday", "home_team", "away_team"]].copy()
        teams["game_id"] = teams["game_id"].astype(str)
        oof["game_id"] = oof["game_id"].astype(str)
        frame = oof.merge(teams.drop_duplicates("game_id"), on="game_id",
                          how="left", suffixes=("", "_t"))
        if "gameday_t" in frame.columns:
            frame = frame.drop(columns=[c for c in frame.columns if c.endswith("_t")])
        logger.info("team coverage: %.3f", frame["home_team"].notna().mean())
        frame = frame.sort_values(["gameday", "game_id"], kind="stable").reset_index(drop=True)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(ckpt, index=False)
        logger.info("score-OOF checkpoint written (%s)", ckpt)

    results = {}
    for name, flag in (("K0_control", False), ("K1_locbias", True)):
        config.DISTRIBUTION_LOCATION_BIAS_CORRECT = flag
        s1 = time.time()
        priced, bundle = dist_mod.calibrate_market_frame(frame)
        # priced carries only model outputs; reattach keys positionally-
        # free via the frame index (concat().reindex(original_index)
        # preserves frame.index, which canonical_sort leaves non-reset).
        for _kc in ("game_id", "gameday"):
            if _kc not in priced.columns and _kc in frame.columns:
                priced[_kc] = frame.loc[priced.index, _kc].to_numpy()
        maps = (bundle or {}).get("maps", {})
        has_block = "loc_bias" in (maps.get("variance") or {})
        logger.info("%s: flag=%s block_in_maps=%s rows=%d (%.0fs)",
                    name, flag, has_block, len(priced), time.time() - s1)
        fp = Path(config.RUN_DIAGNOSTICS_DIR) / f"ab_location_retrain_{name}_priced.parquet"
        fp.parent.mkdir(parents=True, exist_ok=True)
        keep = [c for c in priced.columns if c.startswith("p_over_")
                or c.startswith("p_push_total_") or c in (
            "game_id", "gameday", "home_score", "away_score", "fair_total",
            "distribution_sd_total", "mu_h", "mu_a")]
        priced[keep].to_parquet(fp, index=False)
        results[name] = priced
        config.DISTRIBUTION_LOCATION_BIAS_CORRECT = False

    market = pd.read_csv(Path(config.RUN_DIAGNOSTICS_DIR) / "oof_with_market.csv",
                         usecols=["game_id", "gameday", "total_line"])
    market["game_id"] = market["game_id"].astype(str)
    ev0 = _grade_at_line(results["K0_control"], market, "K0")
    ev1 = _grade_at_line(results["K1_locbias"], market, "K1")
    m = ev0[["game_id", "llm", "p_model", "ae"]].rename(
        columns={"llm": "ll0", "p_model": "p0", "ae": "ae0"}).merge(
        ev1[["game_id", "gameday", "y", "llm", "p_model", "ae"]].rename(
        columns={"llm": "ll1", "p_model": "p1", "ae": "ae1"}),
        on="game_id", how="inner")
    print(f"\npaired population: n={len(m)}", flush=True)
    y = m["y"].to_numpy(float)
    _paired(y, m["p0"].to_numpy(float), m["p1"].to_numpy(float),
            m["gameday"], "K1 - K0 (negative = K1 better)")
    for sname, mask in (("edge>3%", m["ae0"] > 0.03), ("edge<=3%", m["ae0"] <= 0.03)):
        mm = m[mask]
        dd = (mm["ll1"] - mm["ll0"]).to_numpy()
        print(f"--- {sname}: n={len(mm)} mean={dd.mean():+.5f} "
              f"K1 better {int((dd < 0).sum())}/{len(mm)}", flush=True)
    print("--- |edge| ladder (K1-K0, negative = K1 better):", flush=True)
    for lo, hi in [(0, .015), (.015, .03), (.03, .06), (.06, .10), (.10, 1)]:
        mm = m[(m["ae0"] > lo) & (m["ae0"] <= hi)]
        if len(mm) == 0:
            continue
        dd = (mm["ll1"] - mm["ll0"]).to_numpy()
        print(f"  {lo:.3f}-{hi:.3f}: n={len(mm):4d} mean={dd.mean():+.5f}", flush=True)
    from sklearn.metrics import roc_auc_score
    for cn in ("p0", "p1"):
        try:
            auc = roc_auc_score(y, m[cn].to_numpy(float))
        except Exception:
            auc = float("nan")
        print(f"  AUC {cn}={auc:.4f}", flush=True)
    out = Path(config.RUN_DIAGNOSTICS_DIR) / "ab_location_retrain_paired_oof.csv"
    m.to_csv(out, index=False)
    print(f"\n  paired OOF written to {out} total {(time.time()-t0)/3600:.1f}h", flush=True)
    print("Adopt bar: K1-K0 paired 95% CI excludes 0 (K1 better) AND agreement "
          "must not regress AND 10%+ bucket improves AND AUC not regressing.", flush=True)


if __name__ == "__main__":
    main()
