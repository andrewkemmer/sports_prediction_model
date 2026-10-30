"""Moneyline remediation: temperature scaling in logit space (saved OOF).

Served model prob is p_ensemble_calibrated (already per-origin Platt via
gated_calibrator in moneyline.py). Temperature T>1 cools overconfident
tails: q = sigmoid(logit(p)/T). Monotone -> AUC exactly preserved.

Market (mkt) is used ONLY for edge bucketing in this diagnostic, never
in the transform. Served gating uses |p-0.5| (market-free).

Outputs: baseline matrix, fixed T x gate scan, causal expanding T-pick
with gameday-blocked paired CI verdict.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

j = pd.read_csv('nba-backend/run_diagnostics/oof_with_market.csv')
j['gameday'] = pd.to_datetime(j['game_date'])
j = j.sort_values(['gameday', 'game_id']).reset_index(drop=True)
n = len(j)
p = np.clip(j['p_ensemble_calibrated'].to_numpy(float), 1e-6, 1 - 1e-6)
k = np.clip(j['mkt'].to_numpy(float), 1e-6, 1 - 1e-6)
y = j['home_win'].to_numpy(float)
clip = lambda q: np.clip(q, 1e-9, 1 - 1e-9)
llm = -(y * np.log(clip(p)) + (1 - y) * np.log(1 - clip(p)))
llk = -(y * np.log(clip(k)) + (1 - y) * np.log(1 - clip(k)))
ae = np.abs(p - k)
conf = np.abs(p - 0.5)

print(f"n={n} AUCm={roc_auc_score(y, p):.4f} AUCk={roc_auc_score(y, k):.4f}", flush=True)
print("baseline (model p_ensemble_calibrated vs market):", flush=True)
for lo, hi in [(0, .015), (.015, .03), (.03, .06), (.06, .10), (.10, 1)]:
    m = (ae > lo) & (ae <= hi)
    print(f"  {lo:.3f}-{hi:.3f}: n={m.sum():4d} LLm={llm[m].mean():.4f} "
          f"LLk={llk[m].mean():.4f} AUCm={roc_auc_score(y[m], p[m]):.3f}", flush=True)

lp = logit(p)
TS = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
GATES = [0.0, 0.03, 0.06, 0.10]
LL = {}
for g in GATES:
    gated = conf > g
    for T in TS:
        qs = np.where(gated, expit(lp / T), p)
        LL[(g, T)] = -(y * np.log(clip(qs)) + (1 - y) * np.log(1 - clip(qs)))

print("\ngain vs control LL (positive = scaled better):", flush=True)
print("gate\\T | " + " ".join(f"{T:>7.2f}" for T in TS), flush=True)
best = {}
for g in GATES:
    row = [(llm - LL[(g, T)]).mean() for T in TS]
    best[g] = TS[int(np.argmax(row))]
    print(f" {g:.2f}  | " + " ".join(f"{v:+.5f}" for v in row) +
          f"   best T={best[g]}", flush=True)

# ladder for the most promising (gate, T) candidates
for g, T in [(0.0, best[0.0]), (0.06, best[0.06])]:
    dd = llm - LL[(g, T)]
    print(f"\nladder gate={g} T={T} gain={dd.mean():+.5f}:", flush=True)
    for lo, hi in [(0, .015), (.015, .03), (.03, .06), (.06, .10), (.10, 1)]:
        m = (ae > lo) & (ae <= hi)
        print(f"  {lo:.3f}-{hi:.3f}: n={m.sum():4d} gain={dd[m].mean():+.5f}", flush=True)

# ---- causal expanding T-pick (gate fixed at 0.06, grid TS) ----
G = 0.06
gated = conf > G
LLg = np.column_stack([LL[(G, T)] for T in TS])
BURN = 200
cum = np.cumsum(LLg, axis=0)
runmean = cum / np.arange(1, n + 1)[:, None]
star = np.zeros(n, dtype=int)
star[BURN:] = np.argmin(runmean[:-BURN], axis=1)
ll_star = LLg[np.arange(n), star]
ll_star[:BURN] = llm[:BURN]
dd = ll_star - llm  # negative = scaled better
gd = j['gameday'].to_numpy()
days = np.array(sorted(set(gd)))
idx = {gg: np.where(gd == gg)[0] for gg in days}
rng = np.random.default_rng(7)
B = 2000
ms = np.empty(B)
for b in range(B):
    pick = rng.integers(0, len(days), len(days))
    samp = np.concatenate([idx[days[q]] for q in pick])
    ms[b] = dd[samp].mean()
lo, hi = np.percentile(ms, [2.5, 97.5])
used = star[BURN:]
print(f"\ncausal expanding pick (gate={G}, burn-in {BURN}), T* histogram:", flush=True)
for ti, T in enumerate(TS):
    print(f"  T={T:4.2f}: {(used == ti).sum():5d} games", flush=True)
print(f"causal paired delta (scaled - control, negative = scaled better), n={n}", flush=True)
print(f"  mean {dd.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]", flush=True)
print(f"  verdict: {'ADOPT' if hi < 0 else 'NOT distinguishable'}", flush=True)
edge3 = ae > 0.03
edge10 = ae > 0.10
for name, msk in (("edge>3%", edge3), ("edge<=3%", ~edge3),
                   ("|edge|>10%", edge10), ("agreement (|p-mkt|<=.03)", ~edge3)):
    v = dd[msk]
    print(f"--- {name}: n={msk.sum()} mean={v.mean():+.5f}", flush=True)
for loE, hiE in [(0, .015), (.015, .03), (.03, .06), (.06, .10), (.10, 1)]:
    msk = (ae > loE) & (ae <= hiE)
    print(f"  {loE:.3f}-{hiE:.3f}: n={msk.sum():4d} mean={dd[msk].mean():+.5f}", flush=True)
print(f"  AUC control={roc_auc_score(y, p):.4f} (scaled identical by monotonicity)", flush=True)
out = pd.DataFrame({"game_id": j['game_id'], "gameday": j['gameday'].astype(str),
                    "ll0": llm, "ll_star": ll_star,
                    "T_star": [TS[s] for s in star]})
out.to_csv('nba-backend/run_diagnostics/ab_moneyline_temp_paired_oof.csv', index=False)
print("  paired OOF written", flush=True)
