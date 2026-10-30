"""Global + pace variance arms for NBA totals (causal, post-hoc).

A   = incumbent p_model.
G   = causal global widen: per-row scale chosen by minimizing expanding
      prior-mean logloss over grid s in [1.00..1.40] (row transform itself
      is row-local: implied-sigma inversion, no leakage; choice uses
      strictly-prior rows only, min history 200).
GP  = G * causal pace-quintile relative tilt (variance-ratio, prior only).
GPC = GP capped to |P-0.50| <= 0.12 for served leans.

Fair total untouched. Market = 0.50.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

EV = 'nba-backend/run_diagnostics/ab_pace_variance_oof.csv'
OUT = 'nba-backend/run_diagnostics/ab_pace_global_oof.csv'

j = pd.read_csv(EV)
j['gameday'] = pd.to_datetime(j['gameday'])
j = j.sort_values(['gameday', 'game_id']).reset_index(drop=True)
n = len(j)
print(f"n={n}", flush=True)

fair = j['fair_total'].to_numpy(float)
sd = j['distribution_sd_total'].to_numpy(float)
U = j['U'].to_numpy(float)
pA = np.clip(j['p_A'].to_numpy(float), 1e-6, 1 - 1e-6)
y = j['y'].to_numpy(float)
pace = j['pace'].to_numpy(float)
d = U - fair
q = norm.ppf(np.clip(1 - pA, 1e-6, 1 - 1e-6))
sig_imp = np.where(np.abs(d) < 1e-9, sd, d / np.where(np.abs(q) < 1e-9, np.nan, q))
sig_imp = np.where(np.isfinite(sig_imp) & (sig_imp > 0), sig_imp, sd)

GRID = np.array([1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 1.60, 1.80, 2.00])
G = len(GRID)
# LL matrix: rows x grid (row-local transform, no cross-row info)
Pmat = np.empty((n, G))
for g_i, s in enumerate(GRID):
    Pmat[:, g_i] = np.clip(1 - norm.cdf(d / np.clip(sig_imp * s, 1e-6, None)), 1e-9, 1 - 1e-9)
LLmat = -(y[:, None] * np.log(Pmat) + (1 - y)[:, None] * np.log(1 - Pmat))
cum = np.cumsum(LLmat, axis=0)
cnt = np.arange(1, n + 1)[:, None]
prior_mean = np.vstack([np.full((1, G), np.inf), cum[:-1] / cnt[:-1]])
MIN_HIST = 200
choice = np.zeros(n, dtype=int)
for i in range(n):
    if i < MIN_HIST:
        choice[i] = 0  # s=1.00 warmup
    else:
        choice[i] = int(np.argmin(prior_mean[i]))
j['G_scale'] = GRID[choice]
j['p_G'] = Pmat[np.arange(n), choice]

# causal pace relative tilt (same as pace arm: sqrt(bin_mean/overall), prior only)
resid = j['total'].to_numpy(float) - fair
z2 = (resid / np.clip(sd, 1e-6, None)) ** 2
rel = np.ones(n)
for i in range(n):
    if i < MIN_HIST:
        continue
    pp, pz = pace[:i], z2[:i]
    qs = np.quantile(pp, [0.2, 0.4, 0.6, 0.8])
    b = int(np.searchsorted(qs, pace[i], side='right'))
    assign = np.searchsorted(qs, pp, side='right')
    num = np.full(5, np.nan)
    for qq in range(5):
        m = assign == qq
        if m.sum() >= 30:
            num[qq] = float(np.mean(pz[m]))
    overall = float(np.mean(pz))
    v = num[b] / overall if overall > 0 else np.nan
    rel[i] = float(np.clip(np.sqrt(v), 0.80, 1.40)) if np.isfinite(v) and v > 0 else 1.0
j['pace_rel'] = rel
sig_gp = sig_imp * j['G_scale'].to_numpy(float) * rel
j['p_GP'] = np.clip(1 - norm.cdf(d / np.clip(sig_gp, 1e-6, None)), 1e-6, 1 - 1e-6)
j['p_GPC'] = np.clip(j['p_GP'], 0.38, 0.62)

clip = lambda p: np.clip(p, 1e-9, 1 - 1e-9)
j['llA'] = -(y * np.log(clip(j['p_A'])) + (1 - y) * np.log(1 - clip(j['p_A'])))
for k in ['G', 'GP', 'GPC']:
    j[f'll{k}'] = -(y * np.log(clip(j[f'p_{k}'])) + (1 - y) * np.log(1 - clip(j[f'p_{k}'])))
j['llK'] = -(y * np.log(0.5) + (1 - y) * np.log(0.5))
j['aeA'] = (j['p_A'] - 0.5).abs()

def auc_of(s, col):
    return roc_auc_score(s['y'], s[col]) if s['y'].nunique() > 1 else float('nan')

print("=== A vs G vs GP vs GPC ===", flush=True)
for name, s in [('full', j), ('edge>3%', j[j.aeA > 0.03]), ('edge<=3%', j[j.aeA <= 0.03])]:
    row = f"{name}: n={len(s)}"
    for k in ['A', 'G', 'GP', 'GPC']:
        ll = s[f'll{k}'].mean() if k != 'A' else s['llA'].mean()
        row += f" ll{k}={ll:.4f}"
    for k in ['G', 'GP', 'GPC']:
        row += f" gain{k}={(s['llA'] - s[f'll{k}']).mean():+.4f}"
    row += f" aucA={auc_of(s,'p_A'):.3f} aucGPC={auc_of(s,'p_GPC'):.3f}"
    print(row, flush=True)

print("|edgeA| ladder gains (vs A):", flush=True)
for lo, hi in [(0, .015), (.015, .03), (.03, .06), (.06, .10), (.10, 1)]:
    s = j[(j['aeA'] > lo) & (j['aeA'] <= hi)]
    if len(s) == 0:
        continue
    print(f" ae {lo:.3f}-{hi:.3f}: n={len(s):4d} " + " ".join(f"gain{k}={(s['llA']-s[f'll{k}']).mean():+.4f}" for k in ['G', 'GP', 'GPC']), flush=True)

print("signed gains:", flush=True)
for name, s in [('over>3%', j[j.p_A - 0.5 > 0.03]), ('under>3%', j[j.p_A - 0.5 < -0.03])]:
    print(f" {name}: n={len(s)} " + " ".join(f"gain{k}={(s['llA']-s[f'll{k}']).mean():+.4f}" for k in ['G', 'GP', 'GPC']), flush=True)

print(f"chosen G_scale: mean={j['G_scale'].mean():.3f} last={j['G_scale'].iloc[-1]:.2f}", flush=True)
print("G_scale path (every 500th):", j['G_scale'].iloc[::500].tolist(), flush=True)

# gameday-blocked bootstrap for each arm's gain (full + edge)
j['_day'] = j['gameday'].dt.date
g = j.groupby('_day')
ids = np.array(list(g.groups.keys())); N = len(ids)
gn = g.size().to_numpy()
rng = np.random.default_rng(0); B = 500
for k in ['G', 'GP', 'GPC']:
    gg = g.apply(lambda t, _k=k: (t['llA'] - t[f'll{_k}']).sum(), include_groups=False).to_numpy()
    vals = np.empty(B)
    for b in range(B):
        ix = rng.integers(0, N, N)
        vals[b] = gg[ix].sum() / max(gn[ix].sum(), 1)
    sub = j[j.aeA > 0.03]
    print(f"gain{k} full: {(j['llA']-j[f'll{k}']).mean():+.4f} 95%CI [{np.percentile(vals,2.5):+.4f},{np.percentile(vals,97.5):+.4f}] edge>3%: {(sub['llA']-sub[f'll{k}']).mean():+.4f}", flush=True)

j.to_csv(OUT, index=False)
print(f"saved {OUT}", flush=True)
