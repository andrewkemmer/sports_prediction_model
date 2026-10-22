"""One-shot: rebuild 2023-24 EPA aggs, calibrate edge->P(home win) per
variant on unaffected games, report AUC/logloss/Brier/ECE on affected games.
Saves nfl-backend/run_diagnostics/phase1_prob_eval.csv."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

import ingestion
import features as F

SEASONS = [2023, 2024]
pbp = ingestion.load_pbp(seasons=SEASONS)
ps = ingestion.load_player_stats(seasons=SEASONS)
games = ingestion.load_schedule(seasons=SEASONS)
print(f"game_types: {games['game_type'].value_counts().to_dict()}", flush=True)
games = games[games['game_type'] == 'REG'].copy()
print(f"REG games={len(games)}", flush=True)

obs = F.epa_opportunity_table(pbp)
days = games[["game_id", "gameday", "gametime"]].copy()
days["game_id"] = days["game_id"].astype(str)
days["gameday"] = pd.to_datetime(days["gameday"], errors="coerce").dt.normalize()
days["kickoff_utc"] = F._kickoff_utc(days)
obs["game_id"] = obs["game_id"].astype(str)
obs = obs.merge(days, on="game_id", how="inner", validate="many_to_one")
pos = ps[["game_id", "team", "player_id", "position"]].copy()
pos["game_id"] = pos["game_id"].astype(str)
pos["team"] = pos["team"].astype("string").str.strip()
pos["player_id"] = pos["player_id"].astype("string").str.strip()
pos["position"] = F._normalize_epa_lineup_positions(pos["position"])
pos = pos.dropna(subset=["player_id", "position"])
pos = pos.drop_duplicates(["game_id", "team", "player_id"], keep="last")
obs = obs.merge(pos, on=["game_id", "team", "player_id"],
                how="inner", validate="one_to_one")
obs = obs.dropna(subset=["gameday", "kickoff_utc"])
hist = F.epa_quality_ratings(obs)
inj = ingestion.load_injuries_pit(games, seasons=SEASONS)
g = games[["game_id", "gameday", "gametime", "home_team", "away_team"]].copy()
caps = F._phase1_tpg_caps(hist)
print("CAPS:", {k: round(v, 2) for k, v in caps.items()}, flush=True)

variants = {"none": F.epa_quality_team_agg(hist, g, None),
            "base": F.epa_quality_team_agg(hist, g, inj)}
for a in (0.5, 0.7, 0.9):
    al = {"QB": 1.0, "WR": a, "TE": a, "RB": a}
    variants[f"a{a:.1f}"] = F.epa_quality_team_agg(
        hist, g, inj, redistribute_phase1=True,
        phase1_caps=caps, phase1_alpha=al)
print("aggs done", flush=True)


def team_edge(agg):
    t = agg.copy()
    t["game_id"] = t["game_id"].astype(str)
    t["team"] = t["team"].astype(str)
    return t.groupby(["game_id", "team"])["epa_q"].sum()


E = {k: team_edge(v) for k, v in variants.items()}
sc = games.drop_duplicates("game_id").set_index("game_id")
rows = []
for gid in sc.index:
    gid = str(gid)
    try:
        h, a = str(sc.loc[gid, "home_team"]), str(sc.loc[gid, "away_team"])
        hs, aws = sc.loc[gid, "home_score"], sc.loc[gid, "away_score"]
        if pd.isna(hs) or pd.isna(aws):
            continue
        r = {"gid": gid, "y": int(hs > aws)}
        for k, S in E.items():
            r[k] = float(S.loc[(gid, h)] - S.loc[(gid, a)])
        rows.append(r)
    except KeyError:
        pass
d = pd.DataFrame(rows).dropna()
d["bite"] = (d["none"] - d["base"]).abs()
train = d[d["bite"] <= 0.5].copy()
test = d[d["bite"] > 0.5].copy()
print(f"games={len(d)} train(unaffected)={len(train)} "
      f"test(affected)={len(test)}", flush=True)


def ece(y, p, bins=10):
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    e, n = 0.0, len(y)
    for b in range(int(np.nanmax(q)) + 1):
        m = q == b
        if m.sum():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return e


out = []
for k in ["none", "base", "a0.5", "a0.7", "a0.9"]:
    clf = LogisticRegression().fit(train[[k]], train["y"])
    p = clf.predict_proba(test[[k]])[:, 1]
    y = test["y"].to_numpy(dtype=float)
    out.append({"variant": k,
                "n": len(y),
                "base_rate": round(float(y.mean()), 4),
                "AUC": round(float(roc_auc_score(y, p)), 4),
                "logloss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
                "brier": round(float(brier_score_loss(y, p)), 4),
                "ECE_10bin": round(float(ece(y, p)), 4)})
res = pd.DataFrame(out)
print(res.to_string(index=False), flush=True)
res.to_csv("../run_diagnostics/phase1_prob_eval.csv", index=False)
print("saved run_diagnostics/phase1_prob_eval.csv", flush=True)
