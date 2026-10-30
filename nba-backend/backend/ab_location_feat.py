"""Location-feature arm H: trailing market-residual correction (offline, causal).

H0 (diagnostic): does the trailing mean of (fair_total - market_total) at
  the team level predict tonight's signed error? Causal per-game: for each
  game, mean of (fair-U) over each side's strictly-prior games (min 5 each
  side, else NaN -> no correction). Report corr with eM in edge slice.
H1 (arm): corrected location fair_H = fair - k*(prior_bias_home - prior_bias_away)/2?
  NO -- think in levels: if the model runs hot for a team (fair > actual
  systematically), subtract that team's trailing bias. fair_H = fair -
  (biasH + biasA)/2 where biasT = trailing mean(fair - actual) for T.
  Causal, min-history gated. Then p via row-local Normal map (implied sigma
  preserved), no cap (pure location test).
H2 (arm): H1 + causal scale-earn + cap (the GPC stack on the corrected
  location) -- the servable candidate.

Market U is used ONLY to build the diagnostic prior bias in H0... no:
even H1's bias uses (fair - actual), no market input at all. The arm is
market-free: trailing own-error correction. Market U appears only in the
evaluation join (edge slicing, disagreement), never in any feature.

Paired verdict (2026-10-30, on saved OOF, gameday-blocked bootstrap B=2000):
H2 vs GPC +0.0033 CI [+0.0006,+0.0060] significant BUT agreement -0.0023
  and scale binds at ceiling 3.0 -- width, not location. REJECTED.
H3 (H1 location + GPC gate/S=2.0/cap) vs GPC +0.0017 CI [-0.0009,+0.0042]
  crosses zero -- FAILS the adoption gate. GPC STANDS. The trailing
  own-error correction (bias +1.04->+0.28, MAE 15.30->15.20, corr -0.121)
  is banked as a location-model feature for the walk-forward retrain.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

EV = 'nba-backend/run_diagnostics/ab_engine_refit_oof.csv'
FEAT = 'nba-backend/run_diagnostics/ab_pool_gates_A_incumbent_features.parquet'
OUT = 'nba-backend/run_diagnostics/ab_location_feat_oof.csv'

j = pd.read_csv(EV)
j['gameday'] = pd.to_datetime(j['gameday'])
j = j.sort_values(['gameday', 'game_id']).reset_index(drop=True)
n = len(j)
print(f"n={n}", flush=True)

# home_team/away_team ride the eval frame itself (same grain, no join needed).
j['game_id'] = j['game_id'].astype(str)
j['gameday'] = pd.to_datetime(j['gameday'])
print(f"team coverage: {j['home_team'].notna().mean():.3f}", flush=True)

fair = j['fair_total'].to_numpy(float)
tot = j['total'].to_numpy(float)
U = j['U'].to_numpy(float)
pA = np.clip(j['p_A'].to_numpy(float), 1e-6, 1 - 1e-6)
y = j['y'].to_numpy(float)
sd = j['distribution_sd_total'].to_numpy(float)
d = U - fair
q = norm.ppf(np.clip(1 - pA, 1e-6, 1 - 1e-6))
sig_imp = np.where(np.abs(d) < 1e-9, sd, d / np.where(np.abs(q) < 1e-9, np.nan, q))
sig_imp = np.where(np.isfinite(sig_imp) & (sig_imp > 0), sig_imp, sd)

# per-game team bias histories: trailing mean(fair - actual) per team, prior only
err = fair - tot
rec = pd.DataFrame({'gameday': j['gameday'], 'team': j['home_team'], 'err': err})
rec = pd.concat([rec, pd.DataFrame({'gameday': j['gameday'], 'team': j['away_team'], 'err': err})],
                ignore_index=True)
rec = rec.sort_values(['team', 'gameday']).reset_index(drop=True)
rec['cnt'] = rec.groupby('team').cumcount()
rec['trail'] = rec.groupby('team')['err'].transform(
    lambda s: s.shift(1).rolling(20, min_periods=5).mean())
lut = rec.drop_duplicates(['team', 'gameday'], keep='last').set_index(['team', 'gameday'])['trail']
j['biasH'] = [lut.get((t, d), np.nan) for t, d in zip(j['home_team'], j['gameday'])]
j['biasA'] = [lut.get((t, d), np.nan) for t, d in zip(j['away_team'], j['gameday'])]
print(f"bias coverage H={j['biasH'].notna().mean():.3f} A={j['biasA'].notna().mean():.3f} "
      f"both={((j['biasH'].notna()) & (j['biasA'].notna())).mean():.3f}", flush=True)

# H0 diagnostic: trailing predicted bias vs actual error (edge slice)
j['pred_bias'] = (j['biasH'] + j['biasA']) / 2
sub = j[(j.aeA > 0.03) & j['pred_bias'].notna()]
print(f"H0 edge: corr(pred_bias, eM)={np.corrcoef(sub['pred_bias'], sub['total'] - sub['fair_total'])[0,1]:+.3f} n={len(sub)}", flush=True)
sub2 = j[j['pred_bias'].notna()]
print(f"H0 full: corr(pred_bias, eM)={np.corrcoef(sub2['pred_bias'], sub2['total'] - sub2['fair_total'])[0,1]:+.3f} n={len(sub2)}", flush=True)
print("H0 edge by pred_bias quartile:", flush=True)
sub['pq'] = pd.qcut(sub['pred_bias'], 4, labels=False, duplicates='drop')
for qq, g in sub.groupby('pq'):
    print(f"  Q{int(qq)}: pred~{g['pred_bias'].mean():+.2f} n={len(g)} actual_bias={(g['total']-g['fair_total']).mean():+.2f}", flush=True)

# H1 arm: fair_H = fair - pred_bias (both sides present, else no correction)
fairH = np.where(j['pred_bias'].notna(), fair - j['pred_bias'].to_numpy(float), fair)
dH = U - fairH
pH = np.clip(1 - norm.cdf(dH / np.clip(sig_imp, 1e-9, None)), 1e-6, 1 - 1e-6)

# H2: H1 + causal scale-earn + cap
GRID = np.array([1.00, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0])
G = len(GRID)
Pmat = np.empty((n, G))
for gi, S in enumerate(GRID):
    Pp = np.clip(1 - norm.cdf(dH / np.clip(sig_imp * S, 1e-9, None)), 1e-9, 1 - 1e-9)
    Pmat[:, gi] = np.clip(Pp, 0.38, 0.62)
LLmat = -(y[:, None] * np.log(Pmat) + (1 - y)[:, None] * np.log(1 - Pmat))
MIN_HIST = 200
cum = np.cumsum(LLmat, axis=0)
prior = np.vstack([np.full((1, G), np.inf), cum[:-1] / np.arange(1, n + 1)[:, None][:-1]])
choice = np.zeros(n, dtype=int)
for i in range(n):
    choice[i] = 0 if i < MIN_HIST else int(np.argmin(prior[i]))
pH2 = Pmat[np.arange(n), choice]

j['p_H1'] = np.clip(pH, 1e-6, 1 - 1e-6)
j['p_H2'] = np.clip(pH2, 1e-6, 1 - 1e-6)
# reference GPC
pG = np.clip(1 - norm.cdf(d / np.clip(sig_imp * 2.0, 1e-9, None)), 1e-6, 1 - 1e-6)
pG = np.clip(pG, 0.38, 0.62)
gate = np.abs(pA - 0.5) > 0.03
j['p_GPC'] = np.where(gate, pG, pA)

clip = lambda p: np.clip(p, 1e-9, 1 - 1e-9)
for k in ['H1', 'H2', 'GPC']:
    j[f'll{k}'] = -(y * np.log(clip(j[f'p_{k}'])) + (1 - y) * np.log(1 - clip(j[f'p_{k}'])))
j['H2_scale'] = GRID[choice]

print("=== A vs H1(loc) vs H2(loc+earn+cap) vs GPC ===", flush=True)
for name, s in [('full', j), ('edge>3%', j[j.aeA > 0.03]), ('edge<=3%', j[j.aeA <= 0.03])]:
    row = f"{name}: n={len(s)} llA={s['llA'].mean():.4f}"
    for k in ['H1', 'H2', 'GPC']:
        row += f" ll{k}={s[f'll{k}'].mean():.4f} gain{k}={(s['llA'] - s[f'll{k}']).mean():+.4f}"
    print(row, flush=True)
print("|edgeA| ladder gains:", flush=True)
for lo, hi in [(0, .015), (.015, .03), (.03, .06), (.06, .10), (.10, 1)]:
    s = j[(j['aeA'] > lo) & (j['aeA'] <= hi)]
    if len(s) == 0:
        continue
    print(f" ae {lo:.3f}-{hi:.3f}: n={len(s):4d} " + " ".join(
        f"gain{k}={(s['llA'] - s[f'll{k}']).mean():+.4f}" for k in ['H1', 'H2', 'GPC']), flush=True)

j['_day'] = j['gameday'].dt.date
g = j.groupby('_day')
ids = np.array(list(g.groups.keys()))
N = len(ids)
gn = g.size().to_numpy()
rng = np.random.default_rng(0)
B = 500
for k in ['H1', 'H2']:
    gg = g.apply(lambda t, _k=k: (t['llA'] - t[f'll{_k}']).sum(), include_groups=False).to_numpy()
    vals = np.empty(B)
    for bb in range(B):
        ix = rng.integers(0, N, N)
        vals[bb] = gg[ix].sum() / max(gn[ix].sum(), 1)
    sub = j[j.aeA > 0.03]
    print(f"gain{k} full: {(j['llA'] - j[f'll{k}']).mean():+.4f} 95%CI [{np.percentile(vals,2.5):+.4f},{np.percentile(vals,97.5):+.4f}] edge>3%: {(sub['llA']-sub[f'll{k}']).mean():+.4f}", flush=True)
print("H2 scale path (every 500th):", GRID[choice[::500]].tolist(), flush=True)
j.to_csv(OUT, index=False)
print(f"saved {OUT}", flush=True)
