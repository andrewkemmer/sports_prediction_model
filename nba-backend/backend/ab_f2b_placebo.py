"""Placebo checks for the F2b margin-shift: zeros / sign-flip / permutation.

If the ladder gains come from signal, sign-flip and permutation must go
flat-or-negative while the true shift stays negative. If they track the
true shift, the construction is an artifact.
"""
import numpy as np
import pandas as pd
from collections import deque
import warnings
warnings.filterwarnings('ignore')

s = pd.read_parquet('nba-backend/run_diagnostics/ab_location_retrain_score_oof.parquet')
s['game_id'] = s['game_id'].astype(str)
s['gameday'] = pd.to_datetime(s['gameday'])
s = s.sort_values(['gameday', 'game_id']).reset_index(drop=True)
n = len(s)
mu_m = (s['mu_h'] - s['mu_a']).to_numpy(float)
mar = (s['home_score'] - s['away_score']).to_numpy(float)
ht = s['home_team'].astype(str).to_numpy()
at = s['away_team'].astype(str).to_numpy()
err = mu_m - mar
bh = np.full(n, np.nan)
ba = np.full(n, np.nan)
mem = {}
for i in range(n):
    H, A = ht[i], at[i]
    mem.setdefault(H, deque())
    mem.setdefault(A, deque())
    if len(mem[H]) >= 5:
        bh[i] = float(np.mean(list(mem[H])[-20:]))
    if len(mem[A]) >= 5:
        ba[i] = float(np.mean(list(mem[A])[-20:]))
    if np.isfinite(err[i]):
        mem[H].append(err[i])
        mem[A].append(-err[i])

e2 = pd.read_csv('nba-backend/run_diagnostics/edge_spread_eval.csv')
e2['game_id'] = e2['game_id'].astype(str)
e2['gameday'] = pd.to_datetime(e2['gameday'])
e2 = e2.sort_values(['gameday', 'game_id']).reset_index(drop=True)
e2['bh'] = e2['game_id'].map(dict(zip(s['game_id'], bh)))
e2['ba'] = e2['game_id'].map(dict(zip(s['game_id'], ba)))
e2 = e2[e2['bh'].notna() & e2['ba'].notna()].reset_index(drop=True)
shift = np.clip(-0.5 * (e2['bh'].to_numpy(float) + e2['ba'].to_numpy(float)), -6, 6)
line = e2['home_line'].to_numpy(float)
mar2 = (e2['home_score'] - e2['away_score']).to_numpy(float)
keep = mar2 != line
e2 = e2[keep].reset_index(drop=True)
shift = shift[keep]
line = line[keep]
mar2 = mar2[keep]


def _sfx(L):
    if abs(L) % 1:
        return "m%d_5" % int(abs(L)) if L < 0 else "%d_5" % int(abs(L))
    return "m%d" % int(abs(L)) if L < 0 else "%d" % int(L)


grid = sorted([float("-%d" % i) for i in range(20, 0, -1)]
              + [float("%d" % i) for i in range(0, 21)]
              + [float("-%d.5" % i) for i in range(19, 0, -1)]
              + [float("%d.5" % i) for i in range(0, 20)])
HC = np.column_stack([e2['p_home_cover_%s' % _sfx(L)].to_numpy(float) for L in grid])
AC = np.column_stack([e2['p_away_cover_%s' % _sfx(L)].to_numpy(float) for L in grid])
TW = HC / np.clip(HC + AC, 1e-12, None)
y2 = (mar2 > line).astype(float)
q0 = np.clip(np.array([float(np.interp(x, grid, TW[r])) for r, x in enumerate(line)]), 1e-6, 1 - 1e-6)
ll0 = -(y2 * np.log(q0) + (1 - y2) * np.log(1 - q0))
ae = np.abs(q0 - 0.5)


def dd_of(sh):
    eff = np.clip(line - sh, -20.0, 20.0)
    qi = np.clip(np.array([float(np.interp(x, grid, TW[r])) for r, x in enumerate(eff)]), 1e-6, 1 - 1e-6)
    return (-(y2 * np.log(qi) + (1 - y2) * np.log(1 - qi))) - ll0


rng = np.random.default_rng(0)
cases = [('true shift', shift), ('sign-flipped', -shift),
         ('permuted', rng.permutation(shift)), ('zeros', np.zeros_like(shift))]
for name, sh in cases:
    dd = dd_of(sh)
    print("%-14s overall=%+.5f | 0-1.5%%=%+.5f (n%d) 10%%+=%+.5f (n%d)"
          % (name, dd.mean(), dd[ae <= .015].mean(), int((ae <= .015).sum()),
             dd[ae > .10].mean(), int((ae > .10).sum())), flush=True)
