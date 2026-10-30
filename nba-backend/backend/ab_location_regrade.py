"""Grade-only rerun: reads the two priced parquets from the crashed run."""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, "nba-backend/backend")
from ab_location_retrain import _grade_at_line, _paired  # noqa: E402

d = Path("nba-backend/run_diagnostics")
k0 = pd.read_parquet(d / "ab_location_retrain_K0_control_priced.parquet")
k1 = pd.read_parquet(d / "ab_location_retrain_K1_locbias_priced.parquet")
for f in (k0, k1):
    f["game_id"] = f["game_id"].astype(str)
    if "gameday" in f.columns and "gameday_x" not in f.columns:
        pass

market = pd.read_csv(d / "oof_with_market.csv",
                     usecols=["game_id", "gameday", "total_line"])
market["game_id"] = market["game_id"].astype(str)

# coalesce any gameday_x/gameday_y from the priced frames' own gameday
for f in (k0, k1):
    if "gameday" in f.columns:
        f.drop(columns=["gameday"], inplace=True)

ev0 = _grade_at_line(k0, market, "K0")
ev1 = _grade_at_line(k1, market, "K1")
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
out = d / "ab_location_retrain_paired_oof.csv"
m.to_csv(out, index=False)
print(f"\n  paired OOF written to {out}", flush=True)
