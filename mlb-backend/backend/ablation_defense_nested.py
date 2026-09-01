"""Nested contrasts + collinearity for the v3 defense ablation.

Reuses ablation_defense's exact machinery (same data prep via
run_margin_ablation.prepare_data, same condition walk-forward, same
folds/seed) to answer the report-level questions main() does not emit:

  NESTED CONTRASTS: C5 vs C3 (position-split vs aggregate), C6 vs C3
  (starter-conditioning vs aggregate), C4 vs C3 (trends vs levels),
  C7 vs each. Pairwise per-game logloss delta with DM + paired-t.
  COLLINEARITY: max |corr| of every defensive column against the
  existing elo / win_pct / bullpen / SP-ERA feature groups.

Writes data_delivery/ablation_defense_v3_<sha>_nested.json.
Experiment-only: touches no production code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import ablation_defense as ad  # noqa: E402
import run_margin_ablation as rma  # noqa: E402


def main() -> None:
    sha = rma.head_sha()
    (games, _t, hold_df, folds, _m, _hm, _r, _u) = rma.prepare_data(21)
    wide = ad._load_pbp_wide()
    pbp_lean = ad._load_pbp_lean()
    pbp = wide if wide is not None else pbp_lean
    f135 = ad.build_f1_f3_f5(pbp, games)
    f24 = ad.build_f2_f4(wide, games) if wide is not None else None
    print(f"commit={sha[:12]} games={len(games)} folds={len(folds)}", flush=True)

    # Run every condition walk-forward (identical folds/seed) and KEEP the
    # per-game loss series so pairwise contrasts get real DM/t statistics.
    conds = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    series: dict[str, dict[str, np.ndarray]] = {}
    fold_counts: dict[str, list] = {}
    # C0 has no defense cols -> walk_forward_condition returns None; build
    # the baseline explicitly (same as ablation_defense.main) so C0 gets
    # fold_counts for the subset-consistency comparison.
    import training
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    base_cols = [c for c in training.FEATURE_COLS if c in games.columns]
    y_base, p_base = [], {"lgbm": [], "logistic": []}
    fc0 = []
    for split in folds:
        train, val = split["train_games"], split["val_games"]
        if len(val) < 40 or len(train) < 200:
            fc0.append({"val_start": str(split.get("val_start", "")),
                        "n_val": len(val), "used": False, "reason": "fold_too_small"})
            continue
        y_tr = train["home_win"].values.astype(int)
        y_va = val["home_win"].values.astype(int)
        Xtr = train[base_cols].fillna(train[base_cols].median())
        Xva = val[base_cols].fillna(train[base_cols].median())
        lgbm = LGBMClassifier(random_state=ad.SEED, n_estimators=200,
                              learning_rate=0.05, num_leaves=15,
                              min_child_samples=30, verbose=-1)
        lgbm.fit(Xtr, y_tr)
        mu, sd = Xtr.mean(), Xtr.std()
        sd[sd == 0] = 1
        lr = LogisticRegression(max_iter=1000, random_state=ad.SEED)
        lr.fit((Xtr - mu) / sd, y_tr)
        assert len(y_va) == len(val), "subset violation in baseline fold"
        fc0.append({"val_start": str(split.get("val_start", "")),
                    "n_val": len(val), "used": True})
        y_base.extend(y_va.tolist())
        p_base["lgbm"].extend(lgbm.predict_proba(Xva)[:, 1].tolist())
        p_base["logistic"].extend(lr.predict_proba((Xva - mu) / sd)[:, 1].tolist())
    y_base = np.asarray(y_base)
    series["C0"] = {"y": y_base,
                     "lgbm": ad.logloss(y_base, np.asarray(p_base["lgbm"])),
                     "logistic": ad.logloss(y_base, np.asarray(p_base["logistic"]))}
    fold_counts["C0"] = fc0
    print(f"  running C0 (explicit baseline) ... n={len(y_base)}", flush=True)

    for cond in [c for c in conds if c != "C0"]:
        print(f"  running {cond} ...", flush=True)
        fams = ad.CONDITIONS[cond]
        res = ad.walk_forward_condition(fams, folds, f135, f24)
        if res is None:
            print(f"    None", flush=True)
            continue
        y = np.asarray(res["y"])
        series[cond] = {
            "y": y,
            "lgbm": ad.logloss(y, res["proxy_p"]["lgbm"]),
            "logistic": ad.logloss(y, res["proxy_p"]["logistic"]),
        }
        fold_counts[cond] = res["fold_counts"]
        print(f"    n={len(y)}", flush=True)

    # SUBSET-CONSISTENCY: all conditions must share identical folds.
    base = fold_counts["C0"]
    for cond, fc in fold_counts.items():
        if cond == "C0":
            continue
        assert len(fc) == len(base)
        for b, c in zip(base, fc):
            assert b["used"] == c["used"] and b["n_val"] == c["n_val"], (
                f"{cond}: subset differs from C0 at fold {b.get('val_start')}")
    print("subset-consistency: all conditions share C0's folds", flush=True)

    nested: dict[str, dict] = {}
    for a_name, b_name in (("C5", "C3"), ("C6", "C3"), ("C4", "C3"),
                           ("C7", "C1"), ("C7", "C2"), ("C7", "C3"),
                           ("C7", "C4"), ("C7", "C5"), ("C7", "C6")):
        if a_name not in series or b_name not in series:
            continue
        a, b = series[a_name], series[b_name]
        assert len(a["y"]) == len(b["y"])
        entry = {"n": int(len(a["y"])), "y_range": bool(a["y"].min() == b["y"].min() == 0.0)}
        for proxy in ("lgbm", "logistic"):
            la, lb = a[proxy], b[proxy]
            dm, dm_p = ad.diebold_mariano(lb, la)  # positive = A helps
            t, t_p = ad.paired_t(lb, la)
            entry[proxy] = {
                f"{a_name}_ll": round(float(la.mean()), 6),
                f"{b_name}_ll": round(float(lb.mean()), 6),
                f"delta_{a_name}-{b_name}": round(float(lb.mean() - la.mean()), 6),
                "dm": round(dm, 3) if np.isfinite(dm) else None,
                "dm_p": round(dm_p, 4) if np.isfinite(dm_p) else None,
                "t_p": round(t_p, 4) if np.isfinite(t_p) else None,
            }
        nested[f"{a_name}_vs_{b_name}"] = entry
        print(f"  {a_name} vs {b_name}: {entry}", flush=True)

    # Collinearity: |corr| of each defensive column vs the existing
    # trained feature groups (mirroring the report's requirement: how much
    # of each defense family is already proxied by elo / win_pct /
    # bullpen / SP-ERA features). Uses production FEATURE_COLS names.
    import training
    base_feats = [c for c in training.FEATURE_COLS if c in games.columns]
    groups = {
        "elo": [c for c in base_feats if "elo" in c],
        "win_pct": [c for c in base_feats if "win_pct" in c or "woba" in c],
        "bullpen": [c for c in base_feats if "bullpen" in c or "pen" in c],
        "sp_era": [c for c in base_feats if "sp_" in c or "starter" in c
                   or "era" in c or "fip" in c],
    }
    def_cols = sorted({c for f in ad.FAMILIES for c in ad.family_columns(f)})
    collin = {}
    for frame, tag in ((f135, "F1/F3/F5"), (f24, "F2/F4")):
        if frame is None:
            continue
        perf = games[["game_pk"] + base_feats].merge(
            frame.drop_duplicates("game_pk"), on="game_pk", how="left")
        for dc in def_cols:
            if dc not in perf.columns:
                continue
            row = {}
            for gname, cands in groups.items():
                cands = [c for c in cands if c in perf.columns]
                if not cands:
                    continue
                sub = perf[[dc] + cands].dropna()
                if len(sub) < 200 or sub[dc].nunique() < 10:
                    continue
                corrs = sub[[dc] + cands].corr()[dc].drop(labels=[dc])
                row[gname] = round(float(corrs.abs().max()), 4)
            if row:
                collin[dc] = {"from": tag, "max_abs_corr": row}
                print(f"  {dc} ({tag}): {row}", flush=True)

    out = Path(__file__).resolve().parent / ".." / "data_delivery" / \
        f"ablation_defense_v3_{sha[:12]}_nested.json"
    Path(out).resolve().parent.mkdir(parents=True, exist_ok=True)
    def _sanitize(o):
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_sanitize(v) for v in o]
        if isinstance(o, np.generic):
            return o.item()
        return o

    out.write_text(json.dumps(_sanitize({
        "commit": sha, "nested_contrasts": nested,
        "collinearity": collin,
        "cond_ll_series": {c: {p: series[c][p].tolist()
                               for p in ("lgbm", "logistic")}
                           for c in series},
    }), indent=2))
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()