"""Spreads remediation arm: causal global scale (GPC playbook) on saved OOF.

Mirrors ab_pace_scale_pick.py + served rescale_totals_legs_toward_half logic,
applied to the spread two-way leg at the offered market line:

  q_new = 1 - Phi((line - fair_spread) / (sigma_imp * S)),  gated + capped.

Two outputs:
  1. Fixed-scale scan table (offline knee-find, full OOF).
  2. Causal expanding-prior pick: each game graded with S* = argmin mean
     paired LL over strictly-prior games (burn-in 200). THIS is the verdict arm.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

j = pd.read_csv('nba-backend/run_diagnostics/edge_spread_eval.csv')
j['gameday'] = pd.to_datetime(j['gameday'])
j = j.sort_values(['gameday', 'game_id']).reset_index(drop=True)
j = j[~j['push_game'].astype(bool)].reset_index(drop=True)  # ex-push only

n = len(j)
q = np.clip(j['p_model'].to_numpy(float), 1e-6, 1 - 1e-6)
y = j['y'].to_numpy(float)
d = (j['home_line'] - j['fair_spread']).to_numpy(float)
sd = j['distribution_sd_margin'].to_numpy(float)
ae = np.abs(q - 0.5)
GATE, CAP = 0.03, 0.12
gated = ae > GATE

z = norm.ppf(np.clip(1 - q, 1e-6, 1 - 1e-6))
sig = np.where((np.abs(d) > 1e-9) & (np.abs(z) > 1e-9), d / z, np.nan)
sig = np.where(np.isfinite(sig) & (sig > 0), sig, sd)
sig = np.where(np.isfinite(sig) & (sig > 0), sig, 15.0)

SCALES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
clip = lambda p: np.clip(p, 1e-9, 1 - 1e-9)
ll0 = -(y * np.log(clip(q)) + (1 - y) * np.log(1 - clip(q)))
llk = np.full(n, -np.log(0.5))

# per-game LL under each fixed scale (gated rows rescaled, else control)
LL = np.empty((n, len(SCALES)))
for si, S in enumerate(SCALES):
    qn = q.copy()
    act = gated & np.isfinite(sig)
    qn[act] = 1 - norm.cdf(d[act] / np.clip(sig[act] * S, 1e-9, None))
    qn = np.clip(qn, 0.5 - CAP, 0.5 + CAP)
    qq = np.where(gated, qn, q)
    LL[:, si] = -(y * np.log(clip(qq)) + (1 - y) * np.log(1 - clip(qq)))

edge3 = ae > 0.03
edge10 = ae > 0.10
print(f"n={n} gated={gated.sum()} ({gated.mean():.3f})", flush=True)
print("scale | full_gain | edge3_gain | agree_gain | 10pct_gain | mean|dp|_gated", flush=True)
for si, S in enumerate(SCALES):
    gain = (ll0 - LL[:, si]).mean()
    g3 = (ll0[edge3] - LL[edge3, si]).mean()
    ga = (ll0[~gated] - LL[~gated, si]).mean()
    g10 = (ll0[edge10] - LL[edge10, si]).mean() if edge10.sum() else float('nan')
    mdp = np.abs(LL[gated, si] - ll0[gated]).mean() if gated.sum() else 0.0
    print(f" {S:4.2f} | {gain:+.5f} | {g3:+.5f} | {ga:+.5f} | {g10:+.5f} | {mdp:.5f} "
          f"(n10={edge10.sum()})", flush=True)

# ---- causal expanding-prior pick ----
BURN = 200
cum = np.cumsum(LL, axis=0)
cnt = np.arange(1, n + 1)[:, None]
runmean = cum / cnt
star = np.zeros(n, dtype=int)  # scale index used for game i (prior-only)
star[BURN:] = np.argmin(runmean[:-BURN], axis=1)
ll_star = LL[np.arange(n), star]
ll_star[:BURN] = ll0[:BURN]  # burn-in graded as control

print(f"\ncausal expanding pick (burn-in {BURN}), S* distribution:", flush=True)
used = star[BURN:]
for si, S in enumerate(SCALES):
    print(f"  S={S:4.2f}: {(used == si).sum():5d} games", flush=True)

dd = ll_star - ll0
gd = j['gameday'].to_numpy()
# gameday-blocked bootstrap CI on the paired delta
rng = np.random.default_rng(7)
days = np.array(sorted(set(gd)))
idx = {g: np.where(gd == g)[0] for g in days}
B = 2000
means = np.empty(B)
for b in range(B):
    pick = rng.integers(0, len(days), len(days))
    samp = np.concatenate([idx[days[k]] for k in pick])
    means[b] = dd[samp].mean()
lo, hi = np.percentile(means, [2.5, 97.5])
print(f"\ncausal paired delta (scaled - control, negative = scaled better), n={n}", flush=True)
print(f"  mean {dd.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]", flush=True)
print(f"  verdict: {'ADOPT' if hi < 0 else 'NOT distinguishable' if lo <= 0 else 'REJECT (harms)'}", flush=True)
for name, msk in (("edge>3%", edge3), ("edge<=3%", ~edge3),
                  ("|edge|>10%", edge10),
                  ("agreement (|q-0.5|<=.03)", ~gated)):
    v = dd[msk]
    print(f"--- {name}: n={msk.sum()} mean={v.mean():+.5f} "
          f"scaled-better {int((v < 0).sum())}/{msk.sum()}", flush=True)
for loE, hiE in [(0, .015), (.015, .03), (.03, .06), (.06, .10), (.10, 1)]:
    msk = (ae > loE) & (ae <= hiE)
    if not msk.sum():
        continue
    v = dd[msk]
    print(f"  {loE:.3f}-{hiE:.3f}: n={msk.sum():4d} mean={v.mean():+.5f}", flush=True)
from sklearn.metrics import roc_auc_score
try:
    q_star = np.where(gated, 1 - norm.cdf(
        d / np.clip(sig * np.array(SCALES)[star], 1e-9, None)), q)
    q_star = np.clip(q_star, 0.5 - CAP, 0.5 + CAP)
    q_star[:BURN] = q[:BURN]
    print(f"  AUC control={roc_auc_score(y, q):.4f} scaled={roc_auc_score(y, q_star):.4f}", flush=True)
except Exception as e:
    print(f"  AUC failed: {e}", flush=True)
out = pd.DataFrame({"game_id": j['game_id'], "gameday": j['gameday'].astype(str),
                    "ll0": ll0, "ll_star": ll_star, "S_star": [SCALES[s] for s in star]})
out.to_csv('nba-backend/run_diagnostics/ab_spread_scale_paired_oof.csv', index=False)
print("  paired OOF written", flush=True)
