"""Fixed-scale + regime-gate scan for totals adoption (offline, on saved OOF).

Regime rule (served): if |pA_two_way - 0.5| <= 0.03 keep incumbent byte-identical;
else rescale implied sigma by S and cap |P-0.5| <= 0.12.
Scans S to find the knee where edge gains flatten before agreement damage grows.
Agreement slice is untouched by construction (gain exactly 0).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

j = pd.read_csv('nba-backend/run_diagnostics/ab_pace_variance_oof.csv')
j['gameday'] = pd.to_datetime(j['gameday'])
j = j.sort_values(['gameday', 'game_id']).reset_index(drop=True)
n = len(j)
fair = j['fair_total'].to_numpy(float)
U = j['U'].to_numpy(float)
pA = np.clip(j['p_A'].to_numpy(float), 1e-6, 1 - 1e-6)
y = j['y'].to_numpy(float)
d = U - fair
q = norm.ppf(np.clip(1 - pA, 1e-6, 1 - 1e-6))
sd = j['distribution_sd_total'].to_numpy(float)
sig = np.where(np.abs(d) < 1e-9, sd, d / np.where(np.abs(q) < 1e-9, np.nan, q))
sig = np.where(np.isfinite(sig) & (sig > 0), sig, sd)
ae = np.abs(pA - 0.5)
gate = ae > 0.03
llA = j['llA'].to_numpy(float)
llK = j['llK'].to_numpy(float)
print(f"n={n} gated={gate.sum()} ({gate.mean():.3f}) agreement={(~gate).sum()}", flush=True)
print("scale | full_gain | edge_gain | agree_gain | 10pct_gain | mean|dp|_gated", flush=True)
best = None
for S in [1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0]:
    pS = np.clip(1 - norm.cdf(d / np.clip(sig * S, 1e-6, None)), 1e-6, 1 - 1e-1 * 0 + 1 - 0.999999)
    pS = np.clip(1 - norm.cdf(d / np.clip(sig * S, 1e-6, None)), 1e-6, 1 - 1e-6)
    pS = np.clip(pS, 0.38, 0.62)
    pB = np.where(gate, pS, pA)
    clip = lambda p: np.clip(p, 1e-9, 1 - 1e-9)
    llB = -(y * np.log(clip(pB)) + (1 - y) * np.log(1 - clip(pB)))
    gain = llA - llB
    fg = gain.mean()
    eg = gain[gate].mean()
    ag = gain[~gate].mean()
    hi = ae > 0.10
    hg = gain[hi].mean()
    mdp = np.abs(pB - pA)[gate].mean()
    print(f" {S:.1f}  | {fg:+.4f}   | {eg:+.4f}   | {ag:+.4f}    | {hg:+.4f}   | {mdp:.4f}", flush=True)
