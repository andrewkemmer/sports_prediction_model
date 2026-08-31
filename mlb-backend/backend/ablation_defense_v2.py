"""DEFENSIVE EXPANSION ABLATION v2 — richer families, new conditions (MLB
moneyline). Measurement task; the production model / FEATURE_COLS stay
untouched. v1's verdict (DON'T ADOPT) stands until superseded by this run.

Extends the v1 harness (ablation_defense.py — ladder / PIT machinery reused,
not rebuilt) with the families v2 asks for. CRITICAL honest-on-artifact note:
the COMMITTED pbp cache has exactly 8 columns
(game_pk, game_date, home_team, away_team, inning_topbot, batter, events,
game_type). There is NO launch_speed / launch_angle / bb_type / hit_location /
fielder / description / runner_on, and no SB/CS/passed-ball/wild-pitch events.
Therefore:
  * F2 BATTED-BALL ALLOWED (exit velo / barrel% / GB% / hard-hit% / LD%) is
    UNCOMPUTABLE here — coverage 0.0, excluded (would need Statcast batted-ball
    data this repo does not commit).
  * F4 POSITION-SPLIT (IF/OF/catcher) is UNCOMPUTABLE — no bb_type /
    hit_location for IF/OF, no passed-ball/SB/CS or catcher throw data for the
    catcher split; only catcher_interf exists and is already folded into the
    F1 errors metric. Coverage 0.0, excluded.
Both exclusions are reported in the coverage/pre-screen tables exactly as the
honesty contract requires (a constant/never-present candidate is never
included; it would add noise, not signal).
  * F1 TEAM BASE and F3 TREND: reused from v1 (def_eff / errors / DP, 30g
    means + 15g-60g trends).
  * F5 DEFENSE-BEHIND-THE-STARTER (NEW, genuinely buildable): the F1 side
    metrics recomputed over the games STARTED BY TONIGHT'S starter
    (home/away_starter_id are 100% present), per-starter trailing ladder in
    STARTS (window 10, min 5) strictly before game date.
So the two families the briefing expected to be NEW (F2/F4) are not measurable
on this committed artifact; F5 is the new measurable signal.

FAMILIES (per side home/away + home-minus-away diff):
  F1 team base   : RAW_COLS     (9)   -- defensive efficiency/errors/DP, 30g
  F3 trend/lead  : TREND_COLS   (9)   -- 15g - 60g of F1
  F5 behind-st   : STARTER_COLS (9)   -- F1 recomputed over the starter's starts
  F2 / F4        : coverage 0.0 -- EXCLUDED (batted-ball / position-split)

CONDITIONS (surviving blocks only; F2/F4 never reach the walk-forward):
  C0 baseline (production FEATURE_COLS, asserted 59)
  C1 +F1 | C4 +F3 | C6 +F5 | C7 +F1+F3+F5
  NESTED CONTRASTS (HOW does defense help?):
    C6 vs C1  does starter-conditioning beat the aggregate?
    C4 vs C1  do trends add over levels?
    C7 vs each.

PROTOCOL (unchanged from v1 / the task):
  1. PIT + coverage report per family; 2024 absent still unfixed.
  2. Pre-screen (LightGBM + standardized logistic) per block F1/F3/F5 on the
     baseline OOF residuals; drop blocks both proxies reject.
  3. Walk-forward (identical folds/seed 42) on surviving conditions with both
     proxies; per-family per-game logloss delta + DM / paired-t vs baseline;
     nested contrasts with the same tests; tree-vs-linear divergence noted.
  4. Collinearity vs elo / win_pct / bullpen / sp_fip family.
  5. Winner selection on the ensemble-weighted validation metric; if nothing
     beats baseline -> DON'T ADOPT, stop. Else promote the single winner (or
     top-2 on family disagreement) to the full 5-model ensemble and evaluate
     ONCE on the sealed 21-day holdout.

Emits data_delivery/ablation_defense_v2_<sha>.json. COMMITS NOTHING.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

from config import (  # noqa: E402
    DATA_DELIVERY_DIR,
    MIN_VAL_FOLD_GAMES,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
)
from scipy.stats import ttest_rel  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from ablation_defense import (  # noqa: E402
    pbp_defensive_aggregates,
    add_defensive_features,
    add_starter_defensive_features,
    walk_forward_proxies,
    prescreen,
    dm_pvalue,
    _per_game_logloss,
    coverage_report,
    collinearity_report,
    proxy_blend_share,
    run_ensemble_variant,
    RAW_COLS,
    TREND_COLS,
    STARTER_COLS,
    sha256_file,
    head_sha,
    EPS,
)

import training  # noqa: E402  (FEATURE_COLS baseline for C0)

F1 = list(RAW_COLS)
F3 = list(TREND_COLS)
F5 = list(STARTER_COLS)
F2: list[str] = []  # batted-ball allowed — coverage 0, excluded
F4: list[str] = []  # position-split — coverage 0, excluded

BLOCK_OF: dict[str, list[str]] = {"C1": F1, "C4": F3, "C6": F5}
FAM_OF: dict[str, str] = {"C1": "F1", "C4": "F3", "C6": "F5"}
COND_BLOCKS: dict[str, list[str]] = {
    "C1": F1, "C4": F3, "C6": F5, "C7": F1 + F3 + F5,
}
NESTED: list[tuple[str, str, str]] = [
    ("C6", "C1", "starter-conditioning vs aggregate"),
    ("C4", "C1", "trends vs levels"),
    ("C7", "C1", "all vs base"),
    ("C7", "C6", "all vs base+starter"),
    ("C7", "C4", "all vs base+trends"),
]


def _significance(m: pd.DataFrame, base_oof: pd.DataFrame,
                  pcol: str) -> dict:
    b = base_oof.set_index("row_id").loc[m["row_id"].values]
    ll_base = _per_game_logloss(m["y"].values, b[pcol].values)
    ll_cond = _per_game_logloss(m["y"].values, m[pcol].values)
    d = ll_base - ll_cond
    out = {"n": int(len(m)), "delta": round(float(d.mean()), 6),
           "dm_p": round(dm_pvalue(d), 4)}
    tt = ttest_rel(ll_base, ll_cond)
    out["t_p"] = round(float(tt.pvalue), 4) if np.isfinite(tt.pvalue) else None
    return out


def _contrast(a: pd.DataFrame, b: pd.DataFrame, games: pd.DataFrame,
              a_cols: list[str], b_cols: list[str], pcol: str) -> dict:
    """Paired delta of cond A logloss minus cond B on their common, fully
    observable subset. Positive delta => A wins (B logloss bigger)."""
    keys = ["row_id", "y", pcol]
    aa = a[keys].dropna(subset=[pcol])
    bb = b[keys].dropna(subset=[pcol])
    mm = aa.merge(bb, on=["row_id"], suffixes=("_a", "_b"))
    g = games.loc[mm["row_id"].values]
    if a_cols:
        ok = g.reindex(columns=a_cols).notna().all(axis=1).values
        mm = mm[ok]
    if b_cols:
        ok = g.loc[mm["row_id"].values].reindex(columns=b_cols).notna().all(axis=1).values
        mm = mm[ok]
    if len(mm) < 60:
        return {"n": int(len(mm)), "delta": None, "dm_p": None, "t_p": None}
    la = _per_game_logloss(mm["y_a"].values, mm[f"{pcol}_a"].values)
    lb = _per_game_logloss(mm["y_b"].values, mm[f"{pcol}_b"].values)
    d = lb - la  # positive => A better
    out = {"n": int(len(mm)), "delta": round(float(d.mean()), 6),
           "dm_p": round(dm_pvalue(d), 4)}
    tt = ttest_rel(lb, la)
    out["t_p"] = round(float(tt.pvalue), 4) if np.isfinite(tt.pvalue) else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout-days", type=int, default=21)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-ensemble", action="store_true")
    args = ap.parse_args()

    sha = head_sha()
    data_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    data_hash = sha256_file(data_path)
    pbp_files = sorted(DATA_DELIVERY_DIR.glob("pbp_chunks/pbp_*.parquet"))

    games = pd.read_csv(data_path)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_win"]).reset_index(drop=True)
    pbp = pd.concat([pd.read_parquet(f) for f in pbp_files], ignore_index=True)
    per_game = pbp_defensive_aggregates(pbp)
    games = add_defensive_features(games, per_game)   # F1 + F3
    games = add_starter_defensive_features(games, per_game)  # F5
    games["row_id"] = np.arange(len(games))

    base_cols = list(training.FEATURE_COLS)
    assert len(base_cols) == 59, (
        f"expected 59 production FEATURE_COLS, got {len(base_cols)}")

    tree_share, lin_share = proxy_blend_share()
    cutoff = games["game_date"].max() - pd.Timedelta(days=args.holdout_days - 1)
    tune_df = games[games["game_date"] < cutoff].reset_index(drop=True)
    hold_df = games[games["game_date"] >= cutoff].reset_index(drop=True)
    all_splits = training.walk_forward_splits(
        tune_df, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
    folds = [s for s in all_splits
             if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]

    print(f"v2 commit={sha[:12]} data_sha={data_hash[:12]} games={len(games)} "
          f"tuning={len(tune_df)} holdout={len(hold_df)} "
          f"folds={len(all_splits)}/{len(folds)} seed={RANDOM_SEED} "
          f"proxy_blend={tree_share}/{lin_share}")

    all_new = RAW_COLS + TREND_COLS + STARTER_COLS
    cov = coverage_report(games, all_new)
    print("defensive column coverage (pbp cache starts 2025-03-18; 2024 "
          "absent):")
    for c in cov:
        print(f"    {c['column']:18s} coverage={c['coverage']:.3f}")

    out = args.out or (DATA_DELIVERY_DIR / f"ablation_defense_v2_{sha[:12]}.json")
    if out.exists():
        results = json.loads(out.read_text())
    else:
        results = {
            "schema": "defense-ablation/v2", "commit_sha": sha,
            "data_sha256": data_hash, "holdout_days": args.holdout_days,
            "pbp_chunks": len(pbp_files), "pbp_columns": sorted(pbp.columns),
            "folds_declared": len(all_splits), "folds_executed": len(folds),
            "clip_eps": EPS, "seed": int(RANDOM_SEED),
            "blend": {"tree": tree_share, "linear": lin_share},
            "excluded_families": {
                "F2": "batted-ball allowed — launch_speed/launch_angle/bb_type "
                      "absent from the committed pbp cache; coverage 0.0",
                "F4": "position-split — no bb_type/hit_location/fielder ids and "
                      "no passed-ball/SB/CS; coverage 0.0",
            },
            "families": {"F1": F1, "F3": F3, "F5": F5},
            "coverage": cov,
            "blocks": {"F1": len(F1), "F3": len(F3), "F5": len(F5),
                       "F2": 0, "F4": 0},
            "conditions": {"C1": len(F1), "C4": len(F3), "C6": len(F5),
                           "C7": len(F1) + len(F3) + len(F5)},
            "prescreen": {}, "walkforward": {}, "contrasts": {},
            "collinearity": collinearity_report(games, all_new, base_cols),
        }
        out.write_text(json.dumps(results, indent=2) + "\n")

    base_oof_path = DATA_DELIVERY_DIR / "ablation_defense_v2_base_oof.parquet"
    base_oof = (pd.read_parquet(base_oof_path)
                if base_oof_path.exists() else None)

    # ── 2) pre-screen + baseline OOF (one-time) ────────────────────────────
    if base_oof is None:
        print("\n[C0] baseline proxy walk-forward (residual frame) ...")
        base_oof = walk_forward_proxies(folds, games, base_cols)
        base_oof.to_parquet(base_oof_path)
        for fam, fam_cols in (("F1", F1), ("F3", F3), ("F5", F5)):
            res = prescreen(fam_cols, base_oof, games)
            results["prescreen"][fam] = res
            print(f"  pre-screen {fam}: survived={res['survived']} "
                  f"{ {k: v for k, v in res['per_proxy'].items()} }")
        results["prescreen"]["F2"] = {"survived": False,
                                      "coverage": 0.0,
                                      "reason": results["excluded_families"]["F2"]}
        results["prescreen"]["F4"] = {"survived": False,
                                      "coverage": 0.0,
                                      "reason": results["excluded_families"]["F4"]}
        out.write_text(json.dumps(results, indent=2) + "\n")

    survivors = [c for c in ("C1", "C4", "C6")
                 if results["prescreen"][FAM_OF[c]]["survived"]]
    run_conds = ["C0"] + survivors
    if survivors:
        run_conds.append("C7")
    print(f"conditions proceeding to walk-forward: {run_conds}")

    # ── 3) per-condition two-proxy walk-forward ────────────────────────────
    oofs: dict[str, pd.DataFrame] = {"C0": base_oof}
    # C0 pooled aggregates (needed for the winner selection; base_oof is the
    # baseline residual frame). Populate once, resume-safe.
    if "C0" not in results["walkforward"]:
        c0w: dict = {"cols": base_cols}
        for tag, pcol in (("lgb", "p_lgb"), ("lr", "p_lr")):
            m = base_oof.dropna(subset=[pcol])
            c0w[f"{tag}_n"] = int(len(m))
            c0w[f"{tag}_logloss"] = round(float(np.mean(
                _per_game_logloss(m["y"].values, m[pcol].values))), 4)
            try:
                c0w[f"{tag}_auc"] = round(float(
                    roc_auc_score(m["y"].values, m[pcol].values)), 4)
            except ValueError:
                c0w[f"{tag}_auc"] = 0.5
        results["walkforward"]["C0"] = c0w
        out.write_text(json.dumps(results, indent=2) + "\n")
    for cond in run_conds:
        if cond == "C0":
            continue
        if cond in results["walkforward"]:
            if (base_oof_path).exists():
                oofs[cond] = walk_forward_proxies(folds, games,
                                                  base_cols + COND_BLOCKS[cond])
            continue
        cols = base_cols + COND_BLOCKS[cond]
        print(f"\n[{cond}] two-proxy walk-forward ({len(cols)} cols) ...")
        oof = walk_forward_proxies(folds, games, cols)
        oofs[cond] = oof
        wf = {"cols": cols}
        for tag, pcol in (("lgb", "p_lgb"), ("lr", "p_lr")):
            m = oof.dropna(subset=[pcol])
            wf[f"{tag}_n"] = int(len(m))
            wf[f"{tag}_logloss"] = round(float(np.mean(
                _per_game_logloss(m["y"].values, m[pcol].values))), 4)
            try:
                wf[f"{tag}_auc"] = round(float(roc_auc_score(
                    m["y"].values, m[pcol].values)), 4)
            except ValueError:
                wf[f"{tag}_auc"] = 0.5
            print(f"    {tag} ll={wf[f'{tag}_logloss']} auc={wf[f'{tag}_auc']}")
        results["walkforward"][cond] = wf
        out.write_text(json.dumps(results, indent=2) + "\n")

    # significance vs C0 on the treated subsets
    for cond in run_conds:
        if cond == "C0":
            continue
        oof = oofs[cond]
        def_cols = COND_BLOCKS[cond]
        g = games.loc[oof["row_id"].values]
        present = g.reindex(columns=def_cols).notna().all(axis=1).values
        sig = {}
        for tag, pcol in (("lgb", "p_lgb"), ("lr", "p_lr")):
            m = oof[present].dropna(subset=[pcol])
            sig[tag] = _significance(m, base_oof, pcol)
            print(f"  {cond} {tag} tot_n={sig[tag]['n']} "
                  f"delta={sig[tag]['delta']} dm_p={sig[tag]['dm_p']} "
                  f"t_p={sig[tag]['t_p']}")
        results["walkforward"][cond]["significance"] = sig
    out.write_text(json.dumps(results, indent=2) + "\n")

    # ── nested contrasts ──────────────────────────────────────────────────
    if not results["contrasts"]:
        for a, b, label in NESTED:
            if a not in oofs:
                oofs[a] = (base_oof if a == "C0" else walk_forward_proxies(
                    folds, games, base_cols + COND_BLOCKS[a]))
            if b not in oofs:
                oofs[b] = (base_oof if b == "C0" else walk_forward_proxies(
                    folds, games, base_cols + COND_BLOCKS[b]))
            a_cols = [] if a == "C0" else COND_BLOCKS[a]
            b_cols = [] if b == "C0" else COND_BLOCKS[b]
            entry = {"label": label}
            for tag, pcol in (("lgb", "p_lgb"), ("lr", "p_lr")):
                entry[tag] = _contrast(oofs[a], oofs[b], games, a_cols,
                                       b_cols, pcol)
            results["contrasts"][f"{a}_vs_{b}"] = entry
            print(f"  contrast {a} vs {b} ({label}): "
                  f"lgb_delta={entry['lgb']['delta']} "
                  f"dm_p={entry['lgb']['dm_p']} | "
                  f"lr_delta={entry['lr']['delta']} "
                  f"dm_p={entry['lr']['dm_p']}")
        out.write_text(json.dumps(results, indent=2) + "\n")

    # ── 5) winner selection on the ensemble-weighted validation metric ────
    if not results.get("winner"):
        scores = {}
        for cond in run_conds:
            w = results["walkforward"][cond]
            ll = tree_share * w["lgb_logloss"] + lin_share * w["lr_logloss"]
            scores[cond] = round(float(ll), 6)
        base_ll = scores["C0"]
        cands = {c: scores[c] for c in run_conds if c != "C0"}
        best = min(cands, key=cands.get)
        tree_best = min(run_conds[1:],
                        key=lambda c: results["walkforward"][c]["lgb_logloss"])
        lin_best = min(run_conds[1:],
                       key=lambda c: results["walkforward"][c]["lr_logloss"])
        winners = []
        if cands[best] < base_ll:
            winners.append(best)
        if tree_best != lin_best and tree_best in cands and lin_best in cands:
            for c in (tree_best, lin_best):
                if c not in winners and cands[c] < base_ll:
                    winners.append(c)
        results["winner"] = {
            "blend_scores": scores, "base_blend_ll": base_ll,
            "selected": winners, "tree_family_best": tree_best,
            "linear_family_best": lin_best,
            "tree_vs_linear_agree": tree_best == lin_best,
            "none_beat_baseline": not winners,
        }
        print(f"\nwinner selection (blend {tree_share}/{lin_share}): "
              f"scores={scores} -> "
              f"selected={winners or ['(none — keep baseline)']}")
        out.write_text(json.dumps(results, indent=2) + "\n")

    # ── 5) full 5-member ensemble on baseline + winner, sealed gate ───────
    if (not args.skip_ensemble and results.get("winner", {}).get("selected")
            and not results["ensemble"]):
        arms = {"C0": base_cols}
        for w in results["winner"]["selected"]:
            arms[w] = base_cols + COND_BLOCKS[w]
        print(f"\nfull 5-member ensemble arms: {list(arms)}")
        for name, cols in arms.items():
            print(f"  [{name}] {len(cols)} cols, {len(folds)} folds ...")
            r = run_ensemble_variant(cols, folds, tune_df, hold_df)
            r["cols"] = cols
            results["ensemble"][name] = r
            b, h = r["pooled"]["blend"], r["holdout"]["blend"]
            print(f"    pooled {b['logloss']:.4f}/{b['auc']:.4f} ece_cal "
                  f"{r['pooled']['blend_calibrated']['ece']:.4f} | holdout "
                  f"{h['logloss']:.4f}/{h['auc']:.4f}")
            out.write_text(json.dumps(results, indent=2) + "\n")
        c0 = results["ensemble"]["C0"]
        gate = {}
        for w in results["winner"]["selected"]:
            wv = results["ensemble"][w]
            h0, hw = c0["holdout"]["blend"], wv["holdout"]["blend"]
            p0, pw = c0["pooled"], wv["pooled"]
            e0 = c0["pooled"]["blend_calibrated"]["ece"]
            ew = wv["pooled"]["blend_calibrated"]["ece"]
            win = (hw["logloss"] < h0["logloss"] and hw["auc"] > h0["auc"]
                   and ew <= e0)
            pooled_ok = pw["blend"]["logloss"] < p0["blend"]["logloss"]
            gate[w] = {"holdout": {"base": h0, "with": hw},
                       "pooled_invert": not pooled_ok,
                       "ece_cal": {"base": e0, "with": ew, "degraded": ew > e0},
                       "adopt": bool(win and pooled_ok)}
            verdict = "ADOPT" if gate[w]["adopt"] else "DON'T ADOPT"
            print(f"\n=== sealed gate [{w}] vs C0: {verdict} ===")
        results["gate"] = gate
        out.write_text(json.dumps(results, indent=2) + "\n")

    print(f"\nablation v2 written: {out}")


if __name__ == "__main__":
    main()