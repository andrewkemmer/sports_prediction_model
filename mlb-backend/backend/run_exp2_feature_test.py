"""Experiment #2 — feature expansion + targeted-removal test (read-only for production).

Candidates (FROZEN, constructed from production source columns only):

  1  exp2_centered_k_diff        (sp_k9 − league_k_pct)·(opp_k_rate − league_k_pct),
                                 home − away. NOTE: production's starter K column is
                                 K/9 (sp_k9) and the offense K column is K/PA
                                 (team_k_rate_30g); league prior is K/PA
                                 (league_k_pct). The mixed-denominator caveat is
                                 recorded in the registry (frozen before results).
  2-4  exp2_cat_k_{cat}_diff     sp_usage_cat · (sp_k_cat − lg_k_cat) ·
                                 (opp_k_cat − lg_k_cat), home − away
  5-7  exp2_cat_xwoba_{cat}_diff sp_usage_cat · (sp_xwoba_cat − lg_xwoba_cat) ·
                                 (opp_xwoba_cat − lg_xwoba_cat), home − away
                                 (directionality: higher xwOBA = worse for the
                                 pitcher; product is the same amplification form
                                 as the K family — no sign flip)
  8  exp2_cat_platoon_k_fastball_diff
        opp_L_share·(sp_fb_vs_L − lg_fb_L) + opp_R_share·(sp_fb_vs_R − lg_fb_R)
        × same-weighted opp K vs FB centered × sp_fastball_usage, home − away

Targets:
  home_win : production binary moneyline via the production 5-member ensemble,
             production walk-forward folds, prequential calibration —
             identical to walk_forward_evaluate's fold loop, with the OOF
             run-margin attach (feature #59) precomputed ONCE
             (deterministic, candidate-independent).
  rl_cover : TRUE −1.5 run line (home_runs − away_runs ≥ 2). Production has
             no discriminative −1.5 model (legacy train_run_line_model
             collapses its target to home_win — excluded per the brief), so
             the run-line model is the SAME ensemble trainer/hyperparameters/
             folds/prequential calibration with the target swapped to the
             true −1.5 outcome (the frame's home_win column carries y_rl;
             fold geometry is a pure function of game_date + non-null target
             so boundaries are byte-identical).

Removal families (frozen BEFORE results; verified against FEATURE_COLS):
  S_K       = {sp_k9_diff, sp_k9_5g_diff}                    (both in 59)
  S_FB_K    = {sp_fbpct_diff, sp_whiff_diff}                 (both in 59;
              the directly-overlapping fastball-pitch-quality pair; sp_k9
              stays — all-pitch K-rate, removed only in S_K)
  S_XWOBA   = {sp_xwoba_diff, sp_xwoba_vs_l_diff}            (both in 59;
              sp_xwoba_vs_r_diff does NOT exist as a baseline feature)
  S_PLATOON = S_K                                            (per brief)

Arm registry is written after EVERY arm (resumable) to
data_delivery/exp2_feature_test_<date>.json. The sealed 21-day holdout is
never read: the experiment frame is the pre-holdout snapshot
(max_date − 21d), identical to the run-engine research convention.

Usage:
    python run_exp2_feature_test.py            # run/resume all arms
    python run_exp2_feature_test.py --report   # summarize the registry
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import training
from build_oof_margin import oof_run_margins
from training import (MIN_VAL_FOLD_GAMES, RETRAIN_CADENCE_DAYS,
                      walk_forward_splits)
from run_engine import HOLDOUT_DAYS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_delivery"
DATE_TAG = "20260907"
REGISTRY = DATA / f"exp2_feature_test_{DATE_TAG}.json"

CATEGORIES = ["fastball", "breaking", "offspeed"]
CANDIDATES = ["exp2_centered_k_diff"] + [
    f"exp2_cat_k_{c}_diff" for c in CATEGORIES
] + [
    f"exp2_cat_xwoba_{c}_diff" for c in CATEGORIES
] + ["exp2_cat_platoon_k_fastball_diff"]

S_FAMILIES = {
    "S_K": ["sp_k9_diff", "sp_k9_5g_diff"],
    "S_FB_K": ["sp_fbpct_diff", "sp_whiff_diff"],
    "S_XWOBA": ["sp_xwoba_diff", "sp_xwoba_vs_l_diff"],
    "S_PLATOON": ["sp_k9_diff", "sp_k9_5g_diff"],
}
CANDIDATE_S = {
    "exp2_centered_k_diff": "S_K",
    "exp2_cat_k_fastball_diff": "S_FB_K",
    "exp2_cat_k_breaking_diff": "S_FB_K",
    "exp2_cat_k_offspeed_diff": "S_FB_K",
    "exp2_cat_xwoba_fastball_diff": "S_XWOBA",
    "exp2_cat_xwoba_breaking_diff": "S_XWOBA",
    "exp2_cat_xwoba_offspeed_diff": "S_XWOBA",
    "exp2_cat_platoon_k_fastball_diff": "S_PLATOON",
}


# ── Candidate construction (arithmetic on existing production columns) ──────
def add_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list]:
    df = df.copy()
    lg_k = df["league_k_pct"]

    # 1. centered K — production K-rate columns: SP = sp_k9 (K/9), opp =
    #    team_k_rate_30g (offense K/PA), league = league_k_pct (K/PA).
    for side in ("home", "away"):
        sp_c = df[f"sp_k9_{side}"] - lg_k
        opp_c = df[f"team_k_rate_30g_{side}"] - lg_k
        df[f"_side_{side}_centered_k"] = sp_c * opp_c
    df["exp2_centered_k_diff"] = (df["_side_home_centered_k"]
                                  - df["_side_away_centered_k"])

    # 2-4 / 5-7: category K and xwOBA
    for cat in CATEGORIES:
        lg_kc = df[f"league_k_pct_cat_{cat}"]
        lg_xc = df[f"league_xwoba_cat_{cat}"]
        for side in ("home", "away"):
            df[f"_side_{side}_cat_k_{cat}"] = (
                df[f"sp_usage_cat_{cat}_{side}"]
                * (df[f"sp_k_pct_cat_{cat}_{side}"] - lg_kc)
                * (df[f"team_k_pct_cat_{cat}_{side}"] - lg_kc))
            df[f"_side_{side}_cat_xwoba_{cat}"] = (
                df[f"sp_usage_cat_{cat}_{side}"]
                * (df[f"sp_xwoba_cat_{cat}_{side}"] - lg_xc)
                * (df[f"team_xwoba_cat_{cat}_{side}"] - lg_xc))
        df[f"exp2_cat_k_{cat}_diff"] = (df[f"_side_home_cat_k_{cat}"]
                                        - df[f"_side_away_cat_k_{cat}"])
        df[f"exp2_cat_xwoba_{cat}_diff"] = (df[f"_side_home_cat_xwoba_{cat}"]
                                            - df[f"_side_away_cat_xwoba_{cat}"])

    # 8: platoon fastball K — for side s, the OPPOSING lineup is the other
    #    team's offense: its L share is opp_lefty_share_<other> and its K-vs-FB
    #    by hand is team_k_pct_fb_vs_<hand>_<other>.
    def _platoon(side: str) -> pd.Series:
        other = "away" if side == "home" else "home"
        lsh = df[f"opp_lefty_share_{other}"]
        rsh = 1.0 - lsh
        lg_l = df["league_k_pct_fb_vs_l"]
        lg_r = df["league_k_pct_fb_vs_r"]
        sp_c = (lsh * (df[f"sp_k_pct_fb_vs_l_{side}"] - lg_l)
                + rsh * (df[f"sp_k_pct_fb_vs_r_{side}"] - lg_r))
        opp_c = (lsh * (df[f"team_k_pct_fb_vs_l_{other}"] - lg_l)
                 + rsh * (df[f"team_k_pct_fb_vs_r_{other}"] - lg_r))
        return sp_c * opp_c * df[f"sp_usage_cat_fastball_{side}"]

    df["exp2_cat_platoon_k_fastball_diff"] = _platoon("home") - _platoon("away")

    cov = {c: {"coverage": round(float(df[c].notna().mean()), 4),
               "mean": round(float(df[c].mean()), 5),
               "std": round(float(df[c].std()), 5),
               "min": round(float(df[c].min()), 5),
               "max": round(float(df[c].max()), 5)}
           for c in CANDIDATES if c in df.columns}
    missing = [c for c in CANDIDATES if c not in df.columns]
    return df, cov, missing


# ── Arm evaluation (production-identical fold loop) ─────────────────────────
def eval_arm_ml(cols: list[str], folds, frame: pd.DataFrame) -> dict:
    """Mirror of walk_forward_evaluate's fold loop with the OOF run-margin
    attach PRECOMPUTED (candidate-independent). training.FEATURE_COLS is set
    per arm; prequential calibration follows the production path."""
    training.FEATURE_COLS = list(cols)
    training._LAST_ADAPTIVE_WEIGHTS.clear()
    oof_y, oof_blend, oof_blend_cal = [], [], []
    fold_aucs, fold_dates = [], []
    for split in folds:
        train, val = split["train_games"], split["val_games"]
        if len(train) < 10 or len(val) < 5:
            continue
        if len(val) < MIN_VAL_FOLD_GAMES and not split.get("is_partial_tail"):
            continue
        models, _m = training.train_moneyline_ensemble(train, val)
        blend, _mp, _w = training.ensemble_predict(models, val)
        y_val = val["home_win"].to_numpy(float)
        fold_cal = None
        if len(oof_blend) >= training.MIN_OOF_FOR_FIT:
            fold_cal = training.moneyline_fit(np.array(oof_y),
                                              np.array(oof_blend))
        cal = training.moneyline_apply(np.asarray(blend, float), fold_cal)
        fold_aucs.append(float(training.compute_metrics(y_val, blend)["auc"]))
        fold_dates.append(str(pd.Timestamp(split["val_end"]).date()))
        oof_y.extend(y_val.tolist())
        oof_blend.extend(np.asarray(blend, float).tolist())
        oof_blend_cal.extend(np.asarray(cal, float).tolist())
    return {"y": np.array(oof_y), "p": np.array(oof_blend),
            "p_cal": np.array(oof_blend_cal),
            "fold_aucs": np.array(fold_aucs), "fold_dates": fold_dates}


def arm_metrics(res: dict) -> dict:
    from training import compute_metrics
    y, p, pc = res["y"], res["p"], res["p_cal"]
    fa = np.asarray(res["fold_aucs"], float)
    fd = np.asarray(res["fold_dates"])
    mid = fd[len(fd) // 2]
    early = fd <= mid
    return {"pooled": compute_metrics(y, p),
            "pooled_cal": compute_metrics(y, pc),
            "fold_aucs_mean": round(float(fa.mean()), 4),
            "early_fold_auc_mean": round(float(fa[early].mean()), 4),
            "late_fold_auc_mean": round(float(fa[~early].mean()), 4),
            "fold_dates_first": str(fd[0]), "fold_dates_last": str(fd[-1])}


# ── Worker plumbing (4 parallel arms; globals set once per worker) ──────────
_G: dict = {}


def _worker_init(frames: dict, folds_by_target: dict) -> None:
    _G["frames"] = frames
    _G["folds"] = folds_by_target


def _worker_task(job: tuple) -> tuple:
    key, name, cols, target = job
    t0 = time.time()
    res = eval_arm_ml(cols, _G["folds"][target], _G["frames"][target])
    return key, name, cols, target, res, round(time.time() - t0, 1)


# ── Main ────────────────────────────────────────────────────────────────────
def build_frame() -> tuple[pd.DataFrame, pd.DataFrame, list, dict]:
    games = pd.read_csv(DATA / "game_level_features.csv", low_memory=False)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_win"]).reset_index(drop=True)
    cutoff = games["game_date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    tune = games[games["game_date"] < cutoff].reset_index(drop=True)
    print(f"frame={len(games)}  research snapshot (pre-holdout)={len(tune)} "
          f"cutoff={cutoff.date()}  holdout UNTOUCHED", flush=True)

    tune, cov, missing = add_candidates(tune)
    if missing:
        raise SystemExit(f"candidate construction failed for {missing}")
    print("candidate coverage:", json.dumps(cov, indent=1), flush=True)

    folds = [s for s in walk_forward_splits(
        tune, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
        if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]
    print(f"folds={len(folds)}", flush=True)

    # Feature #59 (run_margin_diff) — OOF attach precomputed ONCE on the
    # frozen folds; candidate-independent (run-engine 29-feature view).
    margins, _med, _nun = oof_run_margins(tune, folds)
    mmap = margins.set_index("game_pk")[["run_margin_diff"]]
    tune = tune.merge(mmap, on="game_pk", how="left")
    print(f"attached run_margin_diff coverage="
          f"{float(tune['run_margin_diff'].notna().mean()):.3f}", flush=True)

    # TRUE −1.5 target: swap y (fold geometry is a pure function of
    # game_date + non-null target ⇒ byte-identical boundaries). Rebuild the
    # folds ON THE SWAPPED FRAME so every fold slice carries y_rl (the fold
    # loop trains/predicts from split["train_games"]/["val_games"]).
    tune_rl = tune.copy()
    tune_rl["home_win"] = ((tune_rl["home_score"] - tune_rl["away_score"])
                           >= 2).astype(float)
    folds_rl = [s for s in walk_forward_splits(
        tune_rl, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
        if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]
    assert [str(pd.Timestamp(s["val_start"]).date()) for s in folds_rl] == \
           [str(pd.Timestamp(s["val_start"]).date()) for s in folds], \
        "RL fold geometry drifted"
    return tune, tune_rl, {"home_win": folds, "rl_cover": folds_rl}, cov


def all_arm_specs(base_cols: list[str]) -> list[tuple[str, list[str], str]]:
    """Every pre-specified arm (name, cols, target) in frozen order."""
    specs: list[tuple[str, list[str], str]] = []
    for target in ("home_win", "rl_cover"):
        specs.append((f"{target}:F", base_cols, target))
        for cand in CANDIDATES:
            specs.append((f"{target}:F+{cand}", base_cols + [cand], target))
        fam_cols = {}
        for fam, sfx in S_FAMILIES.items():
            if fam == "S_PLATOON":
                fam_cols[fam] = [c for c in base_cols if c not in sfx]
                continue
            fam_cols[fam] = [c for c in base_cols if c not in sfx]
        for cand in CANDIDATES:
            fam = CANDIDATE_S[cand]
            fc = fam_cols[fam]
            specs.append((f"{target}:F-{fam}", fc, target))
            specs.append((f"{target}:F-{fam}+{cand}", fc + [cand], target))
        fb, br, os_ = (f"exp2_cat_k_{c}_diff" for c in CATEGORIES)
        xfb, xbr, xos = (f"exp2_cat_xwoba_{c}_diff" for c in CATEGORIES)
        ck, pk = "exp2_centered_k_diff", "exp2_cat_platoon_k_fastball_diff"
        ladder = {
            "Kladder": [[fb], [fb, br], [fb, br, os_]],
            "xwladder": [[xfb], [xfb, xbr], [xfb, xbr, xos]],
            "cent_fb": [[ck], [ck, fb]],
            "fb_plat": [[fb], [fb, pk]],
        }
        for fam_name, sets in ladder.items():
            for i, extra in enumerate(sets, 1):
                specs.append((f"{target}:{fam_name}#{i}",
                              base_cols + extra, target))
    return specs


def report() -> None:
    reg = json.loads(REGISTRY.read_text())
    arms = reg["arms"]
    print(f"n_snapshot={reg['n_snapshot']} folds={reg['n_folds']} "
          f"arms={len(arms)}")
    base = {t: arms[f"{t}:F"]["metrics"]["pooled"] for t in
            ("home_win", "rl_cover") if f"{t}:F" in arms}
    rows = []
    for key, rec in arms.items():
        if "metrics" not in rec:
            if "alias_of" in rec:
                tgt_alias = rec["alias_of"]
                tgt_alias = f"{key.split(':')[0]}:{tgt_alias.split(':', 1)[1]}"
                rec = arms.get(tgt_alias, {})
            if "metrics" not in rec:
                continue
        t, name = key.split(":", 1)
        m = rec["metrics"]["pooled"]
        mc = rec["metrics"].get("pooled_cal", m)
        b = base.get(t)
        rows.append({
            "target": t, "arm": name, "auc": m["auc"],
            "d_auc": round(m["auc"] - b["auc"], 4) if b else None,
            "ll": m["logloss"],
            "d_ll": round(m["logloss"] - b["logloss"], 4) if b else None,
            "brier": m["brier"],
            "d_brier": round(m["brier"] - b["brier"], 4) if b else None,
            "ece": m.get("ece"),
            "d_ece": round(mc.get("ece", 0) - m.get("ece", 0), 4),
            "early": rec["metrics"].get("early_fold_auc_mean"),
            "late": rec["metrics"].get("late_fold_auc_mean"),
        })
    df = pd.DataFrame(rows).sort_values(["target", "d_auc"],
                                        ascending=[True, False])
    print(df.to_string(index=False))


def base_cols_of() -> list[str]:
    return list(training.FEATURE_COLS)


def main() -> None:
    tune_ml, tune_rl, folds_by_target, cov = build_frame()
    folds = folds_by_target["home_win"]
    base_cols = list(training.FEATURE_COLS)
    assert len(base_cols) == 59 and not any(
        c.startswith("exp2_") for c in base_cols), "FEATURE_COLS drifted"

    registry: dict = {
        "date_tag": DATE_TAG,
        "n_snapshot": int(len(tune_ml)),
        "n_folds": len(folds),
        "fold_signature": [str(pd.Timestamp(s["val_start"]).date())
                           for s in folds],
        "candidate_coverage": cov,
        "s_families": S_FAMILIES,
        "candidate_s_map": CANDIDATE_S,
        "run_line_model_note": (
            "true -1.5 classifier: same ensemble trainer/params/folds with "
            "target swapped to (home_runs - away_runs >= 2); legacy "
            "train_run_line_model excluded (collapses target to home_win)"),
        "mixed_denominator_note": (
            "candidate 1 uses sp_k9 (K/9) for SP and team_k_rate_30g (K/PA) "
            "for opp + league_k_pct (K/PA) — frozen mapping before results"),
        "arms": {},
    }
    if REGISTRY.exists():
        old = json.loads(REGISTRY.read_text())
        if old.get("fold_signature") == registry["fold_signature"]:
            registry["arms"] = old.get("arms", {})
        else:
            print("fold signature changed — starting a fresh registry")

    specs = all_arm_specs(base_cols)
    print(f"total arm specs: {len(specs)}", flush=True)

    # dedupe identical (target, col-set) — reuse the first arm's result
    by_sig: dict = {}
    pending: list[tuple] = []
    for key, cols, target in specs:
        if key in registry["arms"] and "metrics" in registry["arms"][key]:
            by_sig[(target, tuple(sorted(cols)))] = key
            continue
        sig = (target, tuple(sorted(cols)))
        if sig in by_sig:
            print(f"  [dedupe] {key} == {by_sig[sig]}", flush=True)
            registry["arms"][key] = {"alias_of": by_sig[sig]}
            continue
        by_sig[sig] = key
        pending.append((key, key, cols, target))
    done = sum(1 for k, v in registry["arms"].items() if "metrics" in v)
    print(f"cached={done}  pending={len(pending)}", flush=True)
    REGISTRY.write_text(json.dumps(registry, indent=1, default=str))

    if pending:
        with ProcessPoolExecutor(
                max_workers=4, initializer=_worker_init,
                initargs=({"home_win": tune_ml, "rl_cover": tune_rl},
                          folds_by_target)) as ex:
            futs = {ex.submit(_worker_task, job): job[0] for job in pending}
            for fut in as_completed(futs):
                key, name, cols, target, res, rt = fut.result()
                registry["arms"][key] = {
                    "cols_n": len(cols),
                    "extra_cols": [c for c in cols if c not in base_cols],
                    "removed_cols": [c for c in base_cols if c not in cols],
                    "metrics": arm_metrics(res),
                    "runtime_s": rt,
                }
                # resolve aliases pointing at this arm
                for k2, v2 in registry["arms"].items():
                    if v2.get("alias_of") == key:
                        v2["metrics"] = registry["arms"][key]["metrics"]
                        v2.update({k: v for k, v in
                                   registry["arms"][key].items()
                                   if k != "metrics"})
                        v2["alias_of"] = key
                m = registry["arms"][key]["metrics"]["pooled"]
                print(f"  [{rt:.0f}s] {key}: auc={m['auc']} "
                      f"ll={m['logloss']}", flush=True)
                REGISTRY.write_text(json.dumps(registry, indent=1,
                                               default=str))

    REGISTRY.write_text(json.dumps(registry, indent=1, default=str))
    n_done = sum(1 for v in registry["arms"].values() if "metrics" in v)
    print(f"ALL ARMS COMPLETE ({n_done}/{len(specs)})", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
