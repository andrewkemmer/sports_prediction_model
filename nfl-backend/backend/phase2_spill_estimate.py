"""Phase-2 spill estimator (background job).

For each team-game with Out donor(s) in exactly ONE skill position D:
  donor_tpg_D = sum of trailing _tpg of removed pool members (position D)
  for each surviving pool teammate (position A): actual game opp - trailing _tpg
  per-game share s_A = sum(delta_A) / donor_tpg_D
Report mean/SE of s_A over games by (D, A); remainder = vanish share.
Sanity arm: same delta on no-donor games (expect ~0).
"""
import sys
sys.path.insert(0, '.')
import logging
import numpy as np
import pandas as pd

import ingestion
import features as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("phase2_spill")

SEASONS = [2020, 2021, 2022, 2023, 2024]
SKILL = ("WR", "TE", "RB")

sched = ingestion.load_schedule(seasons=SEASONS)
sched = ingestion.eligible_games(sched)
reg = sched[sched["game_type"] == "REG"].copy()
log.info("REG games: %d", len(reg))
pbp = ingestion.load_pbp(seasons=SEASONS)
ps = ingestion.load_player_stats(seasons=[SEASONS[0] - 1] + SEASONS)
inj = ingestion.load_injuries_pit(sched, seasons=SEASONS)
w0 = inj[inj["availability_weight"].eq(0.0)].copy()
for c in ("game_id", "team", "player_id"):
    w0[c] = w0[c].astype("string").str.strip()
w0["team"] = w0["team"].str.upper().map(__import__("config").canonical_team)
log.info("w0 rows: %d", len(w0))

actual = F.epa_opportunity_table(pbp)
actual["game_id"] = actual["game_id"].astype(str)
actual["team"] = actual["team"].astype("string").str.strip().str.upper().map(
    __import__("config").canonical_team)
actual["player_id"] = actual["player_id"].astype("string").str.strip()

pos = ps[["game_id", "team", "player_id", "position"]].copy()
pos["game_id"] = pos["game_id"].astype(str)
pos["team"] = pos["team"].astype("string").str.strip()
pos["player_id"] = pos["player_id"].astype("string").str.strip()
pos["position"] = F._normalize_epa_lineup_positions(pos["position"])
pos = pos.dropna(subset=["player_id", "position"])
pos = pos.drop_duplicates(["game_id", "team", "player_id"], keep="last")

gdays = reg[["game_id", "gameday"]].copy()
gdays["game_id"] = gdays["game_id"].astype(str)
gdays["gameday"] = pd.to_datetime(gdays["gameday"], errors="coerce").dt.normalize()

hist = actual.merge(
    gdays.rename(columns={"gameday": "hg"}), on="game_id", how="inner")
hist = hist.merge(pos, on=["game_id", "team", "player_id"], how="inner")
hist = hist.sort_values(["team", "player_id", "hg", "game_id"])
hist["_tpg_trail"] = hist.groupby(["team", "player_id"])["opp"].transform(
    lambda s: s.shift(1).rolling(8, min_periods=1).mean())
hist["position"] = F._normalize_epa_lineup_positions(hist["position"])
pool = hist[hist["position"].isin(SKILL)].copy()
log.info("pool rows: %d", len(pool))

# latest trailing row per (team, player) strictly before each target game
pool["hg"] = pd.to_datetime(pool["hg"])
targets = reg[["game_id", "gameday", "home_team", "away_team"]].copy()
targets["game_id"] = targets["game_id"].astype(str)
targets["gameday"] = pd.to_datetime(targets["gameday"], errors="coerce").dt.normalize()
rows = []
for _, t in targets.iterrows():
    gd = t["gameday"]
    for side, team in (("h", t["home_team"]), ("a", t["away_team"])):
        team = str(team).upper()
        cand = pool[(pool["team"] == team) & (pool["hg"] < gd)]
        if cand.empty:
            continue
        if (gd - cand["hg"]).min().days > 300:
            continue
        latest = cand.sort_values(["player_id", "hg", "game_id"]).drop_duplicates(
            "player_id", keep="last")
        latest = latest[(gd - latest["hg"]).dt.days <= 300]
        out_ids = set(w0[(w0["game_id"] == t["game_id"])
                         & (w0["team"] == team)]["player_id"])
        latest["is_out"] = latest["player_id"].isin(out_ids)
        act = actual[(actual["game_id"] == t["game_id"])
                     & (actual["team"] == team)].set_index("player_id")["opp"]
        for _, r in latest.iterrows():
            rows.append({"game_id": t["game_id"], "team": team,
                         "player_id": r["player_id"], "pos": r["position"],
                         "is_out": bool(r["is_out"]),
                         "trail": float(r["_tpg_trail"]) if pd.notna(r["_tpg_trail"]) else np.nan,
                         "actual": float(act.get(r["player_id"], 0.0))})
panel = pd.DataFrame(rows).dropna(subset=["trail"])
panel["delta"] = panel["actual"] - panel["trail"]
log.info("panel rows: %d", len(panel))
panel.to_pickle("../run_diagnostics/phase2_panel.pkl")

out = []
for (gid, team), g in panel.groupby(["game_id", "team"]):
    don = g[g["is_out"] & g["pos"].isin(SKILL)]
    if don.empty:
        continue
    dpos = don["pos"].unique()
    if len(dpos) != 1:
        continue  # single-donor-position games only
    D = dpos[0]
    dtpg = float(don["trail"].sum())
    if not np.isfinite(dtpg) or dtpg <= 0:
        continue
    sur = g[~g["is_out"] & g["pos"].isin(SKILL)]
    rec = {"game_id": gid, "team": team, "D": D, "donor_tpg": dtpg,
           "n_donors": int(len(don))}
    for A in SKILL:
        rec[f"d_{A}"] = float(sur.loc[sur["pos"] == A, "delta"].sum())
        rec[f"s_{A}"] = rec[f"d_{A}"] / dtpg
    out.append(rec)
sg = pd.DataFrame(out)
log.info("single-donor games: %d", len(sg))
sg.to_pickle("../run_diagnostics/phase2_singledonor.pkl")

res = []
for D, gd in sg.groupby("D"):
    r = {"D": D, "n": len(gd),
         "donor_tpg_mean": round(float(gd["donor_tpg"].mean()), 2)}
    tot = 0.0
    for A in SKILL:
        m, se = float(gd[f"s_{A}"].mean()), float(gd[f"s_{A}"].sem())
        r[f"share_{A}"] = round(m, 3)
        r[f"se_{A}"] = round(se, 3)
        tot += m
    r["vanish"] = round(1.0 - tot, 3)
    res.append(r)
res_df = pd.DataFrame(res)
print(res_df.to_string(index=False))
res_df.to_csv("../run_diagnostics/phase2_spill_matrix.csv", index=False)

# sanity: no-donor games should show ~0 mean delta per position
nodon = panel.groupby(["game_id", "team"]).filter(lambda x: ~x["is_out"].any())
san = nodon[nodon["pos"].isin(SKILL)].groupby("pos")["delta"].agg(["mean", "sem", "count"])
print("no-donor sanity (expect ~0):")
print(san.round(3).to_string())
log.info("SAVED run_diagnostics/phase2_spill_matrix.csv")
