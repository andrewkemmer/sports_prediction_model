"""Engine-refit harness, offline on saved OOF (causal, local).

Step 1 - compression decomposition: implied sigma (from shipped p at the
  market line) vs distribution_sd_total; raw-Gaussian p at the market line
  vs shipped p. Locates the 1.4-2x gap across Gaussian-width vs downstream
  (calibration/lattice) layers. All row-local, no leakage.
Step 2 - location shrinkage: causal expanding OLS of actual totals on
  fair_total (prior rows only) -> shrunk location per row -> p via the
  row-local Normal map (implied sigma preserved). Arm B.
Step 3 - earn the shrink: expanding grid search over scale S on prior rows
  (same cumsum trick as ab_pace_global), + cap 0.12. Arm C (GPC-successor).
Step 4 - pace tilt on top of C (prior variance-ratio, as in pace arm). Arm D.

Reference: GPC served values recomputed inline (S=2.0, gate, cap).
Gates identical to prior arms. Gameday-blocked bootstrap CIs, paired vs A.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

EV = 'nba-backend/run_diagnostics/ab_pace_variance_oof.csv'
OUT = 'nba-backend/run_diagnostics/ab_engine_refit_oof.csv'

j = pd.read_csv(EV)
j['gameday'] = pd.to_datetime(j['gameday'])
j = j.sort_values(['gameday', 'game_id']).reset_index(drop=True)
n = len(j)
print(f"n={n}", flush=True)

for c in ['fair_total', 'distribution_sd_total', 'total', 'U', 'p_A', 'y',
          'pace', 'llA', 'aeA']:
    assert c in j.columns, f"missing {c}"
mu = j['mu_total'].to_numpy(float) if 'mu_total' in j.columns else None
fair = j['fair_total'].to_numpy(float)
sd = j['distribution_sd_total'].to_numpy(float)
tot = j['total'].to_numpy(float)
U = j['U'].to_numpy(float)
pA = np.clip(j['p_A'].to_numpy(float), 1e-6, 1 - 1e-6)
y = j['y'].to_numpy(float)
pace = j['pace'].to_numpy(float)
d = U - fair

# ---------- Step 1: decomposition ----------
q = norm.ppf(np.clip(1 - pA, 1e-6, 1 - 1e-6))
sig_imp = np.where(np.abs(d) < 1e-9, sd, d / np.where(np.abs(q) < 1e-9, np.nan, q))
sig_imp = np.where(np.isfinite(sig_imp) & (sig_imp > 0), sig_imp, sd)
ratio = sig_imp / np.clip(sd, 1e-9, None)
print("sig_imp/sd_total: median=%.3f mean=%.3f p10=%.3f p90=%.3f" % (
    np.median(ratio), ratio.mean(), *np.percentile(ratio, [10, 90])), flush=True)
ae = np.abs(pA - 0.5)
for lo, hi in [(0, .03), (.03, .10), (.10, 1)]:
    m = (ae > lo) & (ae <= hi)
    print(f"  |edge| {lo:.2f}-{hi:.2f}: n={m.sum()} med_ratio={np.median(ratio[m]):.3f}", flush=True)
if mu is not None:
    print("location: mean|mu_total-fair|=%.2f corr=%.3f sd(mu)=%.2f sd(fair)=%.2f sd(actual)=%.2f" % (
        np.abs(mu - fair).mean(), np.corrcoef(mu, fair)[0, 1],
        mu.std(ddof=1), fair.std(ddof=1), tot.std(ddof=1)), flush=True)
    p_raw = np.clip(1 - norm.cdf((U - mu) / np.clip(sd, 1e-9, None)), 1e-9, 1 - 1e-9)
    ll_raw = -(y * np.log(p_raw) + (1 - y) * np.log(1 - p_raw))
    print("raw-Gaussian(mu,sd) LL=%.4f vs shipped LL=%.4f (diff=%+.4f; + means calibration/lattice help on average)" % (
        ll_raw.mean(), (j['llA']).mean(), ll_raw.mean() - j['llA'].mean()), flush=True)
    b_pool = np.cov(mu, tot, ddof=1)[0, 1] / np.var(mu, ddof=1)
    print("pooled OLS slope actual~mu_total: %.3f (1.0 = well-calibrated locations)" % b_pool, flush=True)
b_pool_fair = np.cov(fair, tot, ddof=1)[0, 1] / np.var(fair, ddof=1)
print("pooled OLS slope actual~fair_total: %.3f" % b_pool_fair, flush=True)

# ---------- Step 2: causal location shrinkage (expanding OLS) ----------
MIN_HIST = 200
x, yy = fair, tot
cx, cxx, cy, cxy = np.cumsum(x), np.cumsum(x * x), np.cumsum(yy), np.cumsum(x * yy)
cnt = np.arange(1, n + 1)
# prior-only moments: shift by one
sx, sxx, sy, sxy, cn = cx - x, cxx - x * x, cy - yy, cxy - x * yy, cnt - 1
den = cn * sxx - sx * sx
b = np.where(cn > 1, (cn * sxy - sx * sy) / np.where(den != 0, den, np.nan), np.nan)
a = np.where(cn > 1, (sy - b * sx) / np.where(cn > 0, cn, np.nan), np.nan)
b = np.where(np.isfinite(b), np.clip(b, 0.0, 1.5), np.nan)
fair_shr = a + b * x
use = np.isfinite(fair_shr) & (np.arange(n) >= MIN_HIST)
fair_shr = np.where(use, fair_shr, fair)
b_out = np.where(use, b, 1.0)
print("shrink slope path (every 500th):", np.round(b_out[::500], 3).tolist(), flush=True)
print("shrink: mean|fair_shr-fair|=%.2f" % np.abs(fair_shr - fair).mean(), flush=True)

# row-local Normal map at shrunk location, implied sigma preserved
dS = U - fair_shr
pB = np.clip(1 - norm.cdf(dS / np.clip(sig_imp, 1e-9, None)), 1e-6, 1 - 1e-6)

# ---------- Step 3: causal scale-earn on shrunk map + cap ----------
GRID = np.array([1.00, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0])
G = len(GRID)
Pmat = np.empty((n, G))
for gi, S in enumerate(GRID):
    Pp = np.clip(1 - norm.cdf(dS / np.clip(sig_imp * S, 1e-9, None)), 1e-9, 1 - 1e-9)
    Pmat[:, gi] = np.clip(Pp, 0.38, 0.62)
LLmat = -(y[:, None] * np.log(Pmat) + (1 - y)[:, None] * np.log(1 - Pmat))
cum = np.cumsum(LLmat, axis=0)
prior = np.vstack([np.full((1, G), np.inf), cum[:-1] / np.arange(1, n + 1)[:, None][:-1]])
choice = np.zeros(n, dtype=int)
for i in range(n):
    choice[i] = 0 if i < MIN_HIST else int(np.argmin(prior[i]))
pC = Pmat[np.arange(n), choice]
print("C scale path (every 500th):", GRID[choice[::500]].tolist(), flush=True)

# ---------- Step 4: pace tilt on top of C ----------
resid = tot - fair
z2 = (resid / np.clip(sd, 1e-9, None)) ** 2
rel = np.ones(n)
for i in range(n):
    if i < MIN_HIST:
        continue
    pp, pz = pace[:i], z2[:i]
    qs = np.quantile(pp, [0.2, 0.4, 0.6, 0.8])
    bb = int(np.searchsorted(qs, pace[i], side='right'))
    assign = np.searchsorted(qs, pp, side='right')
    num = np.full(5, np.nan)
    for qq in range(5):
        m = assign == qq
        if m.sum() >= 30:
            num[qq] = float(np.mean(pz[m]))
    overall = float(np.mean(pz))
    v = num[bb] / overall if overall > 0 else np.nan
    rel[i] = float(np.clip(np.sqrt(v), 0.80, 1.40)) if np.isfinite(v) and v > 0 else 1.0
sig_D = sig_imp * GRID[choice] * rel
pD = np.clip(np.clip(1 - norm.cdf(dS / np.clip(sig_D, 1e-9, None)), 1e-6, 1 - 1e-6), 0.38, 0.62)

# reference GPC (S=2.0, gate, cap) recomputed inline
pGPC = np.clip(1 - norm.cdf(d / np.clip(sig_imp * 2.0, 1e-9, None)), 1e-6, 1 - 1e-6)
pGPC = np.clip(pGPC, 0.38, 0.62)
gate = ae > 0.03
pGPC = np.where(gate, pGPC, pA)

j['p_B'] = pB
j['p_C'] = pC
j['p_D'] = pD
j['p_GPC'] = pGPC
clip = lambda p: np.clip(p, 1e-9, 1 - 1e-9)
for k in ['B', 'C', 'D', 'GPC']:
    j[f'll{k}'] = -(y * np.log(clip(j[f'p_{k}'])) + (1 - y) * np.log(1 - clip(j[f'p_{k}'])))
j['shrink_slope'] = b_out
j['G_scale'] = GRID[choice]

print("=== A vs B(shrink) vs C(shrink+earn+cap) vs D(+pace) vs GPC ===", flush=True)
for name, s in [('full', j), ('edge>3%', j[j.aeA > 0.03]), ('edge<=3%', j[j.aeA <= 0.03])]:
    row = f"{name}: n={len(s)} llA={s['llA'].mean():.4f}"
    for k in ['B', 'C', 'D', 'GPC']:
        row += f" ll{k}={s[f'll{k}'].mean():.4f} gain{k}={(s['llA'] - s[f'll{k}']).mean():+.4f}"
    print(row, flush=True)

print("|edgeA| ladder gains vs A:", flush=True)
for lo, hi in [(0, .015), (.015, .03), (.03, .06), (.06, .10), (.10, 1)]:
    s = j[(j['aeA'] > lo) & (j['aeA'] <= hi)]
    if len(s) == 0:
        continue
    print(f" ae {lo:.3f}-{hi:.3f}: n={len(s):4d} " + " ".join(
        f"gain{k}={(s['llA'] - s[f'll{k}']).mean():+.4f}" for k in ['B', 'C', 'D', 'GPC']), flush=True)

j['_day'] = j['gameday'].dt.date
g = j.groupby('_day')
ids = np.array(list(g.groups.keys()))
N = len(ids)
gn = g.size().to_numpy()
rng = np.random.default_rng(0)
B = 500
for k in ['B', 'C', 'D', 'GPC']:
    gg = g.apply(lambda t, _k=k: (t['llA'] - t[f'll{_k}']).sum(), include_groups=False).to_numpy()
    vals = np.empty(B)
    for bb in range(B):
        ix = rng.integers(0, N, N)
        vals[bb] = gg[ix].sum() / max(gn[ix].sum(), 1)
    sub = j[j.aeA > 0.03]
    print(f"gain{k} full: {(j['llA'] - j[f'll{k}']).mean():+.4f} 95%CI [{np.percentile(vals, 2.5):+.4f},{np.percentile(vals, 97.5):+.4f}] edge>3%: {(sub['llA'] - sub[f'll{k}']).mean():+.4f}", flush=True)

j.to_csv(OUT, index=False)
print(f"saved {OUT}", flush=True)
