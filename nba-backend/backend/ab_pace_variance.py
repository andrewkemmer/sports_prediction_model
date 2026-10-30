"""Pace-conditional variance scaling A/B for NBA totals (causal, post-hoc).

A = incumbent p_model (shipped grid, includes calibration/lattice).
B = pace-scaled: invert implied sigma from A, multiply by causal
    pace-quintile multiplier estimated on strictly-prior games only,
    recompute p_over via Normal CDF. Fair total untouched.

Pace proxy is strictly PIT: ewm_pace_home + ewm_pace_away from the
feature frame (same PIT contract as the model inputs).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

EV = 'nba-backend/run_diagnostics/edge_totals_eval.csv'
FEAT = 'nba-backend/run_diagnostics/ab_pool_gates_A_incumbent_features.parquet'
OUT = 'nba-backend/run_diagnostics/ab_pace_variance_oof.csv'

ev = pd.read_csv(EV)
print(f"eval rows={len(ev)}", flush=True)
# keep two-way graded only (same population as edge_totals.py)
ev = ev[np.isfinite(pd.to_numeric(ev['y'], errors='coerce'))].copy()
print(f"graded two-way n={len(ev)}", flush=True)

feat = pd.read_parquet(FEAT, columns=['game_id', 'gameday', 'ewm_pace_home', 'ewm_pace_away'])
feat['game_id'] = feat['game_id'].astype(str)
ev['game_id'] = ev['game_id'].astype(str)
feat['gameday'] = pd.to_datetime(feat['gameday'])
ev['gameday'] = pd.to_datetime(ev['gameday'])
j = ev.merge(feat, on=['game_id', 'gameday'], how='left')
print(f"pace join coverage: {j['ewm_pace_home'].notna().mean():.3f}", flush=True)
j['pace'] = j['ewm_pace_home'] + j['ewm_pace_away']
# fallback: median pace when missing (rare; keeps population intact, no market info)
med_pace = float(np.nanmedian(j['pace'].to_numpy(float)))
j['pace'] = j['pace'].fillna(med_pace)
j = j.sort_values(['gameday', 'game_id']).reset_index(drop=True)

fair = j['fair_total'].to_numpy(float)
sd = j['distribution_sd_total'].to_numpy(float)
U = j['U'].to_numpy(float)
pA = np.clip(j['p_model'].to_numpy(float), 1e-6, 1 - 1e-6)
y = j['y'].to_numpy(float)
pace = j['pace'].to_numpy(float)
resid = j['total'].to_numpy(float) - fair
# standardized sq residual under base sd (diagnostic for multiplier fit)
z2 = (resid / np.clip(sd, 1e-6, None)) ** 2

# --- causal expanding quintile multipliers ---
MIN_HIST = 200
mult = np.ones(len(j))
qcut_used = np.zeros(len(j), dtype=int)
for i in range(len(j)):
    if i < MIN_HIST:
        mult[i] = 1.0
        continue
    prior_pace = pace[:i]
    prior_z2 = z2[:i]
    # quintile cutoffs from prior only
    qs = np.quantile(prior_pace, [0.2, 0.4, 0.6, 0.8])
    b = int(np.searchsorted(qs, pace[i], side='right'))  # 0..4
    qcut_used[i] = b
    # prior mean z2 per quintile -> multiplier = sqrt(sd_q / sd_overall)
    # z2 mean == variance ratio vs base sd, so mult = sqrt(mean z2 in bin / overall mean z2)
    assign = np.searchsorted(qs, prior_pace, side='right')
    num = np.zeros(5); den = np.zeros(5)
    for q in range(5):
        m = assign == q
        if m.sum() >= 30:
            num[q] = np.mean(prior_z2[m])
        else:
            num[q] = np.nan
    overall = float(np.mean(prior_z2))
    if not np.isfinite(overall) or overall <= 0:
        mult[i] = 1.0
        continue
    v = num[b] / overall
    if not np.isfinite(v) or v <= 0:
        mult[i] = 1.0
    else:
        mult[i] = float(np.clip(np.sqrt(v), 0.80, 1.40))

j['pace_mult'] = mult
# --- invert implied sigma from A, rescale ---
# pA = 1 - Phi((U-fair)/sig_imp) -> sig_imp = (U-fair)/Phi^{-1}(1-pA)
d = U - fair
q = norm.ppf(np.clip(1 - pA, 1e-6, 1 - 1e-6))
sig_imp = np.where(np.abs(d) < 1e-9, sd, d / np.where(np.abs(q) < 1e-9, np.nan, q))
sig_imp = np.where(np.isfinite(sig_imp) & (sig_imp > 0), sig_imp, sd)
sigB = sig_imp * mult
pB = 1 - norm.cdf(d / np.clip(sigB, 1e-6, None))
j['p_A'] = pA
j['p_B'] = np.clip(pB, 1e-6, 1 - 1e-6)

clip = lambda p: np.clip(p, 1e-9, 1 - 1e-9)
j['llA'] = -(y * np.log(clip(j['p_A'])) + (1 - y) * np.log(1 - clip(j['p_A'])))
j['llB'] = -(y * np.log(clip(j['p_B'])) + (1 - y) * np.log(1 - clip(j['p_B'])))
j['llK'] = -(y * np.log(0.5) + (1 - y) * np.log(0.5))
j['dA'] = j['llA'] - j['llK']
j['dB'] = j['llB'] - j['llK']
j['gain'] = j['llA'] - j['llB']  # + = B wins
j['aeA'] = (j['p_A'] - 0.5).abs()

def auc_of(s, col):
    return roc_auc_score(s['y'], s[col]) if s['y'].nunique() > 1 else float('nan')

print("=== A vs B paired (market = 0.50) ===", flush=True)
for name, s in [('full', j), ('edge>3% (A edge)', j[j.aeA > 0.03]), ('edge<=3% (A edge)', j[j.aeA <= 0.03])]:
    print(f"{name}: n={len(s)} mean_dA={s['dA'].mean():+.4f} mean_dB={s['dB'].mean():+.4f} gain={s['gain'].mean():+.4f} P(B wins)={(s['gain']>0).mean():.3f} llA={s['llA'].mean():.4f} llB={s['llB'].mean():.4f} aucA={auc_of(s,'p_A'):.3f} aucB={auc_of(s,'p_B'):.3f}", flush=True)

print("|edgeA| ladder (gain + = B wins):", flush=True)
for lo, hi in [(0, .015), (.015, .03), (.03, .06), (.06, .10), (.10, 1)]:
    s = j[(j['aeA'] > lo) & (j['aeA'] <= hi)]
    if len(s) == 0:
        continue
    print(f" ae {lo:.3f}-{hi:.3f}: n={len(s):4d} gain={s['gain'].mean():+.4f} dA={s['dA'].mean():+.4f} dB={s['dB'].mean():+.4f} accA={(((s['p_A']>0.5)==(s['y']>0.5)).mean()):.3f}", flush=True)

print("signed (A edge dir):", flush=True)
for name, s in [('model-over>3%', j[j.p_A - 0.5 > 0.03]), ('model-under>3%', j[j.p_A - 0.5 < -0.03])]:
    print(f" {name}: n={len(s)} gain={s['gain'].mean():+.4f} dA={s['dA'].mean():+.4f} dB={s['dB'].mean():+.4f}", flush=True)

print("pace quintile multipliers (causal, last game):", flush=True)
print(f" mult range applied: [{mult.min():.3f},{mult.max():.3f}] mean={mult.mean():.4f}", flush=True)
# pooled (non-causal, diagnostic only) variance ratio by pace quintile
j['pace_q'] = pd.qcut(j['pace'], 5, labels=False, duplicates='drop')
print("pooled sd(resid) by pace quintile (diagnostic, NOT used for pricing):", flush=True)
for q, g in j.groupby('pace_q'):
    _z2 = ((g['total'] - g['fair_total']) / g['distribution_sd_total']).to_numpy(float) ** 2
    print(f"  Q{int(q)}: n={len(g)} pace~{g['pace'].mean():.2f} sd_resid={(g['total'] - g['fair_total']).std(ddof=1):.2f} mean_z2={float(np.mean(_z2)):.3f}", flush=True)

# gameday-blocked bootstrap on gain (full + edge slice)
j['_day'] = j['gameday'].dt.date
g = j.groupby('_day')
ids = np.array(list(g.groups.keys()))
gn = g.size().to_numpy()
gg = g['gain'].sum().to_numpy()
ge = g.apply(lambda t: (t['aeA'] > 0.03).sum(), include_groups=False).to_numpy()
geg = g.apply(lambda t: t.loc[t['aeA'] > 0.03, 'gain'].sum(), include_groups=False).to_numpy()
rng = np.random.default_rng(0); B = 500; N = len(ids)
def boot(vals_num, vals_den):
    out = np.empty(B)
    for b in range(B):
        ix = rng.integers(0, N, N)
        out[b] = vals_num[ix].sum() / max(vals_den[ix].sum(), 1)
    return out
gf = boot(gg, gn); ge_b = boot(geg, ge)
print(f"gain full: {j['gain'].mean():+.4f} 95%CI [{np.percentile(gf,2.5):+.4f},{np.percentile(gf,97.5):+.4f}]", flush=True)
sub = j[j.aeA > 0.03]
print(f"gain edge>3%: {sub['gain'].mean():+.4f} 95%CI [{np.percentile(ge_b,2.5):+.4f},{np.percentile(ge_b,97.5):+.4f}]", flush=True)

j.to_csv(OUT, index=False)
print(f"saved {OUT}", flush=True)
