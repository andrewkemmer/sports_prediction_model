"""Target-isolation proof: disabling ONE target's lattice correction must
leave the OTHER target's priced columns untouched (separate code paths).

Derives each column's target ownership empirically by pricing the same
mini-frame under both maps vs each single-target-disabled map.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sport = sys.argv[1]
backend = Path(__file__).resolve().parents[1] / f"{sport}-backend" / "backend"
sys.path.insert(0, str(backend))
import distributions as dist  # noqa: E402

delivery = backend.parent / "data_delivery"
pattern = {"nhl": "nhl_run_engine_markets_20*.csv", "nba": "nba_run_engine_markets_20*.csv",
           "nfl": "nfl_run_engine_markets_20*.csv", "mlb": "run_engine_markets_20*.csv"}[sport]
date = sorted(delivery.glob(pattern))[-1].name.split("_")[-1].split(".")[0]
cfg = json.load(open(delivery / ("run_engine_markets_%s.meta.json" % date
                                 if sport == "mlb" else f"{sport}_run_engine_markets_{date}.meta.json")))
cal = (cfg.get("calibration", {}) or {}).get("calibrators") or {}
bundle = json.loads(json.dumps(cal))  # deep copy
if not bundle.get("maps", {}).get("lattice"):
    # Shared-sport metas (nhl/nba/nfl) disclose the lattice SUMMARY only
    # (magnitude + gate verdict, no delta vectors); full maps are not
    # stored. Isolation is then vacuous IFF nothing was ever admitted:
    # final specs carry no delta/share (checked here) and the write log
    # recorded zero per-origin admissions (checked by the caller).
    summ = cfg.get("lattice") or {}
    clean = all("delta" not in (summ.get(t) or {})
                and (summ.get(t) or {}).get("ot_home_share") is None
                for t in ("totals", "margin"))
    print(json.dumps({"sport": sport, "maps_stored": False,
                      "final_specs_clean": bool(clean),
                      "pass": bool(clean),
                      "note": "no-op sport: nothing admitted, nothing to isolate"}))
    raise SystemExit(0 if clean else 1)

markets = delivery / pattern.replace("20*", date)
frame = pd.read_csv(markets, float_precision="round_trip")
mini = frame[frame["kind"] == "oof"][[c for c in ("game_pk", "game_id", "game_date", "gameday",
                                                   "home_expected_runs", "away_expected_runs",
                                                   "mu_h", "mu_a", "home_score", "away_score")
                                      if c in frame.columns]].head(6).copy()


def price(maps):
    return dist._distribution_price(mini, maps, n_draws=500)


both = bundle["maps"]
totals_off = json.loads(json.dumps(both)); totals_off["lattice"]["totals"] = {}
margin_off = json.loads(json.dumps(both)); margin_off["lattice"]["margin"] = {}
p_both, p_toff, p_moff = price(both), price(totals_off), price(margin_off)


def changed(a, b):
    out = set()
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if not ((x == y) | (pd.isna(x) & pd.isna(y))).all():
            out.add(c)
    return out


totals_cols = changed(p_both, p_toff)    # columns that NEED the totals correction
margin_cols = changed(p_both, p_moff)    # columns that NEED the margin correction
overlap = margin_cols & totals_cols
result = {"sport": sport,
          "margin_only_columns": len(margin_cols),
          "totals_only_columns": len(totals_cols),
          "overlap_forbidden": sorted(overlap),
          "pass": len(overlap) == 0,
          "margin_sample": sorted(margin_cols)[:8],
          "totals_sample": sorted(totals_cols)[:8]}
print(json.dumps(result, indent=1))
raise SystemExit(0 if result["pass"] else 1)
