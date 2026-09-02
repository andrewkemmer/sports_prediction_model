"""Tier-5 (v7, PLAYER-LEVEL) expected-QB-starter identity ablation - NFL binary moneyline.

FIRST PLAYER-LEVEL EXPANSION (2026-09-02). Runs on the served 12-pool under
the unified within-run tolerance gate (MLB-shaped; the ONE shared
``nfl_moneyline.tolerance_verdict``, called verbatim - ll/auc/ece x
pooled/sealed, six conditions, TOL_LL 0.012 / TOL_AUC 0.016 / ECE_TOL 0.01).

HYPOTHESIS: team-level trailing stats (ewm_ypp_diff, ewm_net_pts_diff) are
stale in QB-change games (~10-15% of games) - those windows describe the
OTHER quarterback. Expected-starter identity/skill features (published
pre-game depth-chart QB1) add signal exactly there. The team-level QB-EPA
composite was already rejected (corr 0.8055 with ewm_ypp_diff, DON'T ADOPT
cd3c26b); this family tests starter-IDENTITY conditioning - a different
hypothesis. Pooled/sealed MARGINAL metrics will read "within tolerance, no
clear win" by dilution - the DECISION SURFACE is the conditional QB-change
table + worth-having bar, not the pooled average.

ARMS (identical 88-fold geometry, seed 42, within-run re-trained baselines,
Platt-calibrated pooled + sealed surfaces - the shared run_walk_forward
machinery, no mid-ablation tuning):
    C0 = served 12-pool (FEATURE_COLUMNS minus the is_home anchor)
    A1 = C0 + qb1_skill_diff
    A2 = C0 + identity set (qb1_continuity_diff, qb1_change_diff,
         qb1_primary_out_diff)
    A3 = C0 + all 4 Tier-5 columns

Each arm's baseline for the verdict = C0's within-run re-trained 12-pool
(the arm's own WITHOUT is C0 by construction; run_walk_forward re-fits the
production-config baseline inside every run, so no cross-pull drift is
possible). FEATURE_COLUMNS / the served pool / production config are NOT
touched by this harness - candidates are composed-but-unregistered.

CONDITIONAL TABLE (mandatory for every arm): ll/auc/ece cut on QB-change
games (either side's expected starter != that side's prior-game actual
starter - ``qb1_change_diff != 0``), pooled AND sealed, plus the
stable-games subset for contrast. Rows come from each arm's own per-game
Platt-calibrated predictions (run_walk_forward._history_df), so every cut
compares the same calibration map semantics across arms. The conditional
ECE is computed on that deployed-style map and is a DIAGNOSTIC surface -
the six-condition marginal verdicts stay exactly the record's numbers.

WORTH-HAVING BAR on any within-tolerance survivor: margins must not be
razor-thin (anything within ~1/3 of its TOL on a pooled leg is flagged) and
pooled corroboration is required (no single-window sealed shimmer). If all
arms reject on marginal but the conditional table shows a directional
signal, the record says DATA-LEVER (more games needed), never a forced
adoption. NO adoption is wired by this harness.

Usage (network + nflreadpy needed for the pull; depth charts through 2024
are weekly, 2025+ are dated rolling snapshots - see nfl_features):
    python3 run_tier5_qb_ablation.py                          # all arms
    python3 run_tier5_qb_ablation.py --arms C0,A1 --cache /tmp/t5.json
    python3 run_tier5_qb_ablation.py --assemble-only --cache /tmp/t5.json
Artifact: data_delivery/nfl_tier5_qb_ablation_<frame-sha>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from run_feature_winpct_ablation import DEPLOYED_12
from run_tier1_ablation import _frame_sha256, _member_metrics

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"

# The served 12-pool base (FEATURE_COLUMNS minus is_home; market-free).
BASE_12 = list(DEPLOYED_12)

TIER5 = [
    "qb1_skill_diff",
    "qb1_continuity_diff",
    "qb1_change_diff",
    "qb1_primary_out_diff",
]

# Conditional decision-surface split: a game is a QB-change game when either
# side's expected starter != that side's prior-game actual starter.
def qb_change_mask(df: pd.DataFrame) -> pd.Series:
    return df["qb1_change_diff"].abs() > 0.5


def load_features(features_csv: str | None) -> pd.DataFrame:
    """Feature frame: a provided CSV (must carry the 4 Tier-5 cols), else the
    nflreadpy pull + build (schedule 2018-2025 + pbp incl. passer_id + weekly
    depth charts 2019-2024 + rolling snapshots 2025) exactly as the harness
    arms consume it."""
    if features_csv and Path(features_csv).exists():
        feats = pd.read_csv(features_csv)
        feats["gameday"] = pd.to_datetime(feats["gameday"])
    else:
        import nflreadpy
        from nfl_features import (DEFAULT_SEASONS, TIER1_NEEDS, build_features,
                                  compose_tier5_qb_features)
        from nfl_moneyline import DECIDED_FRAME
        seasons = DEFAULT_SEASONS
        sched = nflreadpy.load_schedules(seasons).to_pandas()
        pbp_all = nflreadpy.load_pbp(seasons)
        keep = [c for c in (("game_id", "posteam", "yards_gained", "epa",
                             "qb_epa", "game_seconds_remaining", "passer_id")
                            + TIER1_NEEDS) if c in pbp_all.columns]
        pbp = pbp_all.select(keep).to_pandas()
        decided = pd.read_csv(DECIDED_FRAME)
        decided = decided[decided["season"].isin(seasons)]
        feats = build_features(decided, sched, pbp)
        depth_weekly = nflreadpy.load_depth_charts(
            seasons=[2019, 2020, 2021, 2022, 2023, 2024]).to_pandas()
        depth_snaps = nflreadpy.load_depth_charts(seasons=[2025]).to_pandas()
        feats = compose_tier5_qb_features(feats, sched, pbp,
                                          depth_weekly, depth_snaps)
    if "home_win" not in feats.columns:
        feats["home_win"] = (feats["home_score"] > feats["away_score"]).astype(int)
    return feats


def _only(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Columns present in the frame - never silently all-NaN inputs."""
    return [c for c in cols if c in df.columns]


def build_arms(feats: pd.DataFrame) -> dict[str, list[str]]:
    """C0 / A1 / A2 / A3 column lists (present-in-frame only)."""
    base = _only(feats, BASE_12)
    t5 = _only(feats, TIER5)
    return {
        "C0": base,
        "A1": base + [c for c in t5 if c == "qb1_skill_diff"],
        "A2": base + [c for c in t5 if c != "qb1_skill_diff"],
        "A3": base + t5,
    }


def _cache_key(cols: list[str]) -> str:
    c = sorted(set(cols))
    h = hashlib.sha1(("plain|" + json.dumps(c)).encode()).hexdigest()[:12]
    return f"plain|{h}"


# ---------------------------------------------------------------------------
# Conditional + marginal metric helpers
# ---------------------------------------------------------------------------
def _cond_metrics(hist: pd.DataFrame, ycol: str, pcol: str,
                  change_flags: pd.Series) -> dict:
    """{cut: {n, logloss, auc, ece}} over all / QB-change / stable rows.

    ``hist`` is one window's rows (pooled or sealed) from an arm's
    _history_df (per-game Platt-calibrated probs); ``change_flags`` is the
    per-game qb1_change_diff Series indexed by game_id."""
    from nfl_moneyline import auc, ece, logloss
    out: dict = {}
    rows = hist.copy()
    if rows.empty:
        return {"all": {"n": 0}, "qb_change": {"n": 0}, "stable": {"n": 0}}
    rows = rows.merge(change_flags.rename("_chg").reset_index(),
                      left_on="game_id", right_on="game_id", how="left")
    cuts = {"all": rows, "qb_change": rows[rows["_chg"].abs() > 0.5],
            "stable": rows[rows["_chg"].abs() <= 0.5]}
    for name, sub in cuts.items():
        if len(sub) == 0:
            out[name] = {"n": 0}
            continue
        y = sub[ycol].to_numpy(dtype=float)
        p = sub[pcol].to_numpy(dtype=float)
        out[name] = {
            "n": int(len(sub)),
            "logloss": round(float(logloss(y, p)), 4),
            "auc": round(float(auc(y, p)), 4) if auc(y, p) is not None else None,
            "ece": round(float(ece(y, p)), 4),
        }
    return out


def _hist_windows(res: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled-OOF and sealed-2025 per-game rows from an arm's history."""
    h = res.get("_history_df")
    if h is None or len(h) == 0:
        return pd.DataFrame(), pd.DataFrame()
    h = h.copy()
    h["_y"] = (h["home_score"] > h["away_score"]).astype(float)
    pooled = h[h["season"].between(2021, 2024)].reset_index(drop=True)
    sealed = h[h["season"] == 2025].reset_index(drop=True)
    return pooled, sealed


def _m(rec: dict) -> dict:
    """{logloss, auc, ece} from a within-run block (rounded for the record)."""
    out = {}
    for k in ("logloss", "auc", "ece"):
        v = rec.get(k)
        out[k] = v if v is None else round(float(v), 4)
    return out


def _worth_having(cand: dict, base: dict, cand_tag: str) -> dict:
    """Worth-having read for a within-tolerance survivor: pooled legs must not
    sit within ~1/3 of their TOL (razor-thin) and must not lean the wrong way.

    Returns {survivor, near_edge_pooled_legs, pooled_direction_ok,
    worth_having}. TOL directions: logloss/ece lower is better; auc higher is
    better. Only consulted when tolerance_verdict already said ADOPT."""
    from nfl_moneyline import ECE_TOL, TOL_AUC, TOL_LL
    deltas = {
        "pooled_ll": (float(cand["pooled"]["logloss"])
                      - float(base["pooled"]["logloss"])),  # <0 better
        "pooled_auc": (float(cand["pooled"]["auc"])
                       - float(base["pooled"]["auc"])),      # >0 better
        "pooled_ece": (float(cand["pooled"]["ece"])
                       - float(base["pooled"]["ece"])),      # <0 better
        "sealed_ll": (float(cand["sealed"]["logloss"])
                      - float(base["sealed"]["logloss"])),
        "sealed_auc": (float(cand["sealed"]["auc"])
                       - float(base["sealed"]["auc"])),
        "sealed_ece": (float(cand["sealed"]["ece"])
                       - float(base["sealed"]["ece"])),
    }
    tols = {"pooled_ll": TOL_LL, "pooled_auc": TOL_AUC, "pooled_ece": ECE_TOL,
            "sealed_ll": TOL_LL, "sealed_auc": TOL_AUC, "sealed_ece": ECE_TOL}
    better = {"pooled_ll": -1, "pooled_auc": 1, "pooled_ece": -1,
              "sealed_ll": -1, "sealed_auc": 1, "sealed_ece": -1}
    pooled_legs = ["pooled_ll", "pooled_auc", "pooled_ece"]
    near = []
    for leg in pooled_legs:
        d = deltas[leg]
        if abs(d) <= tols[leg] / 3.0:           # razor-thin by construction
            near.append(leg)
    # pooled direction: no pooled leg meaningfully worse than its tol/3
    pooled_ok = all(
        deltas[leg] * better[leg] >= -tols[leg] / 3.0 for leg in pooled_legs)
    return {
        "candidate": cand_tag,
        "deltas_vs_c0": {k: round(v, 4) for k, v in deltas.items()},
        "near_edge_pooled_legs": near,
        "pooled_direction_ok": bool(pooled_ok),
        "worth_having": bool(pooled_ok and len(near) == 0),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=None,
                    help="pre-built features CSV (skips the nflreadpy pull)")
    ap.add_argument("--arms", default=None,
                    help="comma list of arm keys (default: ALL of C0,A1,A2,A3)")
    ap.add_argument("--cache", default="/tmp/nfl_tier5_qb_cache.json",
                    help="JSON cache of per-arm walk-forward results "
                         "(default /tmp/nfl_tier5_qb_cache.json, never committed)")
    ap.add_argument("--assemble-only", action="store_true",
                    help="load the cache, print tables + write the record")
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON record")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    cache_path = Path(args.cache)

    if args.assemble_only:
        if not cache_path.exists():
            print(f"cache {cache_path} not found - run arms first",
                  file=sys.stderr)
            return 2
        return _assemble(cache_path, write_record=not args.no_record)

    feats = load_features(args.features)
    arms = build_arms(feats)
    frame_sha = _frame_sha256(feats)
    print(f"decided games: {len(feats)} | frame sha256: {frame_sha}",
          flush=True)

    # --- coverage + QB-change diagnostics (data facts, not a gate) -------
    t5 = _only(feats, TIER5)
    cov = {c: round(float(100 * feats[c].notna().mean()), 2) for c in t5}
    cov_season = {int(s): {c: round(float(100 * feats.loc[
        feats["season"] == s, c].notna().mean()), 2) for c in t5}
        for s in sorted(feats["season"].unique())}
    chg = qb_change_mask(feats)
    base_cols = [c for c in BASE_12 if c in feats.columns]
    corr = feats[base_cols + t5].corr()
    corr_diag = {}
    for c in t5:
        top = corr[c].drop(c).abs().sort_values(ascending=False).head(3)
        corr_diag[c] = [{"feature": k, "abs_r": round(float(v), 4)}
                        for k, v in top.items()]
    diag = {
        "decided_games": int(len(feats)),
        "coverage_pct": cov,
        "coverage_pct_by_season": cov_season,
        "qb_change_games": {
            "pooled_2021_2024": {"n": int(chg[feats["season"].between(
                2021, 2024)].sum()),
                "total": int(feats["season"].between(2021, 2024).sum())},
            "sealed_2025": {"n": int(chg[feats["season"] == 2025].sum()),
                "total": int((feats["season"] == 2025).sum())},
        },
        "corr_vs_12_pool_top3": corr_diag,
        "note": ("corr vs the 12-pool is a DIAGNOSTIC only - the corr-pair "
                 "admission gate is retired (2026-09-02); nothing is pruned "
                 "on |r|."),
    }
    change_flags = feats.set_index("game_id")["qb1_change_diff"]

    want = set((args.arms or "").split(",")) - {""}
    if not want:
        want = set(arms)

    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    if cache.get("frame_sha256") != frame_sha:
        cache = {"frame_sha256": frame_sha, "runs": {}}
    cache["diagnostics"] = diag

    from nfl_moneyline import run_walk_forward

    for key in sorted(want):
        if key not in arms:
            print(f"unknown arm {key!r} (skip)", file=sys.stderr)
            continue
        cols = arms[key]
        ck = _cache_key(cols)
        if ck in cache["runs"]:
            print(f"  [{key}] reusing cached run {ck}", flush=True)
            continue
        print(f"\n=== running arm {key}: {len(cols)} cols "
              f"({len(cols) - len(base_cols)} Tier-5) ===", flush=True)
        res = run_walk_forward(feats, model_features=cols)
        p_hist, s_hist = _hist_windows(res)
        run_blk = {
            "cols": cols,
            "fold_geometry": res.get("fold_geometry"),
            "pooled_model_platt": _m(res["pooled_preq_2021_2024"]["model_platt"]),
            "sealed_model_platt": _m(res["sealed_2025"]["model_platt"]),
            "members": _member_metrics(res, "members"),
            "members_sealed": _member_metrics(res, "members_sealed"),
            "conditional_pooled": _cond_metrics(
                p_hist, "_y", "home_win_prob_model_calibrated", change_flags),
            "conditional_sealed": _cond_metrics(
                s_hist, "_y", "home_win_prob_model_calibrated", change_flags),
            "history_n": {"pooled": int(len(p_hist)), "sealed": int(len(s_hist))},
        }
        cache["runs"][ck] = run_blk
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2))
        print(f"  cached {len(cache['runs'])} unique runs -> {cache_path}",
              flush=True)

    print(f"\ndone; {len(cache['runs'])} unique runs cached in {cache_path}\n"
          f"run with --assemble-only to write the records", flush=True)
    return 0


def _assemble(cache_path: Path, write_record: bool = True) -> int:
    from nfl_moneyline import tolerance_verdict

    cache = json.loads(cache_path.read_text())
    frame_sha = cache.get("frame_sha256", "?")
    diag = cache.get("diagnostics") or {}

    carrier = pd.DataFrame({c: [0.0] for c in set(BASE_12) | set(TIER5)})
    arms = build_arms(carrier)

    def key_of(key: str) -> str:
        return _cache_key(arms[key])

    def metrics(key: str) -> tuple[dict, dict]:
        r = cache["runs"][key_of(key)]
        return r["pooled_model_platt"], r["sealed_model_platt"]

    present = [k for k in ("C0", "A1", "A2", "A3")
               if key_of(k) in cache["runs"]]
    if "C0" not in present:
        print("C0 run missing - run the C0 arm first", file=sys.stderr)
        return 2

    verdicts: dict[str, dict] = {}
    for key in ("A1", "A2", "A3"):
        if key not in present:
            continue
        pc, sc = metrics(key)
        pb, sb = metrics("C0")
        v = tolerance_verdict(pooled_cand=pc, pooled_base=pb,
                              sealed_cand=sc, sealed_base=sb,
                              baseline_name="C0 (within-run served 12-pool)")
        v["worth_having"] = _worth_having(
            {"pooled": pc, "sealed": sc}, {"pooled": pb, "sealed": sb}, key)
        verdicts[key] = v

    print(f"\n=== Tier-5 QB-starter identity ablation (frame {frame_sha}) ===")
    print("coverage %: " + ", ".join(f"{k}={v}" for k, v in
                                     (diag.get("coverage_pct") or {}).items()))
    chg = diag.get("qb_change_games") or {}
    print("QB-change games: pooled %(pooled_2021_2024)s | sealed %(sealed_2025)s"
          % {"pooled_2021_2024": chg.get("pooled_2021_2024"),
             "sealed_2025": chg.get("sealed_2025")})

    print("\narm  pooled_ll pooled_auc pooled_ece  sealed_ll sealed_auc "
          "sealed_ece | cond pooled (all/chg/stable n) | cond sealed "
          "(all/chg/stable n)")
    for key in present:
        r = cache["runs"][key_of(key)]
        p, s = r["pooled_model_platt"], r["sealed_model_platt"]
        cp = r["conditional_pooled"]
        cs = r["conditional_sealed"]
        print(f"{key:4s} {str(p['logloss']):>8s} {str(p['auc']):>8s} "
              f"{str(p['ece']):>10s} {str(s['logloss']):>9s} "
              f"{str(s['auc']):>9s} {str(s['ece']):>9s} | "
              f"{cp['all'].get('n')}/{cp['qb_change'].get('n')}/"
              f"{cp['stable'].get('n')} | "
              f"{cs['all'].get('n')}/{cs['qb_change'].get('n')}/"
              f"{cs['stable'].get('n')}")

    print("\n=== conditional QB-change surface (ll/auc/ece on the cut) ===")
    for key in present:
        r = cache["runs"][key_of(key)]
        print(f"\n[{key}] pooled: " + " | ".join(
            f"{cut}: n={b.get('n')} ll={b.get('logloss')} "
            f"auc={b.get('auc')} ece={b.get('ece')}"
            for cut, b in r["conditional_pooled"].items()))
        print(f"[{key}] sealed: " + " | ".join(
            f"{cut}: n={b.get('n')} ll={b.get('logloss')} "
            f"auc={b.get('auc')} ece={b.get('ece')}"
            for cut, b in r["conditional_sealed"].items()))

    print("\n=== verdicts (tolerance_verdict: ll/auc/ece x pooled/sealed, "
          "each blocking; baseline = within-run C0) ===")
    for key in ("A1", "A2", "A3"):
        if key not in verdicts:
            continue
        v = verdicts[key]
        tag = "ADOPT" if v["adopt"] else "DON'T ADOPT"
        wh = v.get("worth_having") or {}
        wh_tag = ("WORTH HAVING" if wh.get("worth_having")
                  else ("within-tol but razor-thin pooled"
                        if v["adopt"] else "-"))
        print(f"{key:4s} vs C0 | {tag:10s} | {wh_tag}")
        for r_ in v["reasons"]:
            print(f"    - {r_}")

    if not write_record:
        return 0

    record = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "frame_sha256": frame_sha,
        "rule": ("tolerance_verdict (nfl_moneyline, THE one shared helper): "
                 "ll_ok = cand <= base + TOL_LL; auc_ok = cand >= base - "
                 "TOL_AUC; ece_ok = cand <= base + ECE_TOL - each on pooled "
                 "AND sealed, each blocking; adopt = all six. Baseline per "
                 "arm = the within-run C0 re-trained 12-pool. Decision "
                 "surface = the conditional QB-change table + worth-having "
                 "bar (marginal averages are dilution-heavy by design)."),
        "tol": {"ll": 0.012, "auc": 0.016, "ece": 0.01},
        "environment": "local run 2026-09-02 (pandas 2.3.3)",
        "diagnostics": diag,
        "arms": {k: {"features": cache["runs"][key_of(k)]["cols"],
                     "pooled_model_platt": cache["runs"][key_of(k)][
                         "pooled_model_platt"],
                     "sealed_model_platt": cache["runs"][key_of(k)][
                         "sealed_model_platt"],
                     "members": cache["runs"][key_of(k)].get("members"),
                     "members_sealed": cache["runs"][key_of(k)].get(
                         "members_sealed"),
                     "fold_geometry": cache["runs"][key_of(k)].get(
                         "fold_geometry"),
                     "conditional_pooled": cache["runs"][key_of(k)][
                         "conditional_pooled"],
                     "conditional_sealed": cache["runs"][key_of(k)][
                         "conditional_sealed"]}
                 for k in present},
        "verdicts": verdicts,
        "notes": [
            "First PLAYER-LEVEL expansion: expected-starter identity from "
            "pre-game published depth-chart QB1 (weekly files 2001-2024 keyed "
            "by season/week; 2025+ dated rolling snapshots keyed by the last "
            "state strictly before kickoff UTC). Actual starters enter ONLY "
            "as strictly-prior facts. pbp actual of the target game is never "
            "used for a feature value (PIT-tested).",
            "Conditional-table probabilities are each arm's per-game "
            "Platt-calibrated values from run_walk_forward._history_df "
            "(deployed-style map) - a consistent diagnostic surface across "
            "pooled/sealed and arms. The six-condition verdicts above use "
            "the record's own pooled (prequential) / sealed (within-run) "
            "blocks.",
            "Coverage < 100% rows are chart-data gaps (weekly chart lacks a "
            "QB1 cell for that team/week, e.g. some post-season weeks) - NaN, "
            "never fabricated; run_walk_forward drops only rows missing an "
            "arm column, so per-arm universes differ by <= ~2%.",
            "NO adoption wiring: nfl_features.FEATURE_COLUMNS and the served "
            "pool are untouched. If marginal says within-tolerance but the "
            "conditional table shows direction, the record says DATA-LEVER "
            "(more games needed), not a forced adoption.",
        ],
    }
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"nfl_tier5_qb_ablation_{frame_sha}.json"
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
