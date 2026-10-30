"""Full-scope feature verdicts (offline, saved OOF, strictly causal).

F1  member dispersion -> per-game sigma multiplier (totals + spreads legs).
F2b trailing team margin-bias -> margin location shift (grid interpolation).
F2c trailing team prob-drift -> moneyline logit-space shift.
F2a on record: K1 location-bias retrain REJECTED (paired CI included 0).

All trailing features use strictly-prior games only (running deques per
team, trailing-20, min-5, both-sides-required, caps). Parameter choice
(k, cap) by causal expanding pick with burn-in; verdict = paired
gameday-blocked bootstrap CI on the picked arm.
"""
import numpy as np
import pandas as pd
from collections import deque
from scipy.special import expit, logit, ndtr, ndtri
from sklearn.metrics import roc_auc_score

WIN, MING = 20, 5
clip = lambda q: np.clip(q, 1e-9, 1 - 1e-9)

# ---------------- shared frame ----------------
s = pd.read_parquet('nba-backend/run_diagnostics/ab_location_retrain_score_oof.parquet')
s['game_id'] = s['game_id'].astype(str)
s['gameday'] = pd.to_datetime(s['gameday'])
s = s.sort_values(['gameday', 'game_id']).reset_index(drop=True)
n = len(s)
mu_t = (s['mu_h'] + s['mu_a']).to_numpy(float)
mu_m = (s['mu_h'] - s['mu_a']).to_numpy(float)
tot = s['total'].to_numpy(float)
mar = (s['home_score'] - s['away_score']).to_numpy(float)
ht = s['home_team'].astype(str).to_numpy()
at = s['away_team'].astype(str).to_numpy()
mem_t = np.column_stack([(s[f'mu_h_{m}'] + s[f'mu_a_{m}']).to_numpy(float)
                         for m in ('xgboost', 'lightgbm', 'elasticnet')])
mem_m = np.column_stack([(s[f'mu_h_{m}'] - s[f'mu_a_{m}']).to_numpy(float)
                         for m in ('xgboost', 'lightgbm', 'elasticnet')])
disp_t = mem_t.std(axis=1)
disp_m = mem_m.std(axis=1)

m = pd.read_csv('nba-backend/run_diagnostics/oof_with_market.csv',
                usecols=['game_id', 'p_ensemble_calibrated', 'home_win',
                         'total_line', 'home_line'])
m['game_id'] = m['game_id'].astype(str)
j = s.merge(m, on='game_id', how='left')
has_ml = j['p_ensemble_calibrated'].notna().to_numpy()
p_ml = np.clip(j['p_ensemble_calibrated'].fillna(0.5).to_numpy(float), 1e-6, 1 - 1e-6)
y_ml = j['home_win'].to_numpy(float)

# ---------------- causal trailing features ----------------
err_t = mu_t - tot                      # combined totals error (F2a/H convention)
err_m_h = mu_m - mar                    # margin error, home perspective
bias_t_h = np.full(n, np.nan)           # trailing totals bias per game (avg of sides)
bias_t_a = np.full(n, np.nan)
bias_m_h = np.full(n, np.nan)           # side-aware trailing margin bias
bias_m_a = np.full(n, np.nan)
drift_h = np.full(n, np.nan)            # trailing (p_team - y_team)
drift_a = np.full(n, np.nan)
mem = {t: {'et': deque(), 'em': deque(), 'ed': deque()} for t in set(ht) | set(at)}
pm = p_ml  # team-perspective prob computed inline below
for i in range(n):
    H, A = ht[i], at[i]
    for arr, key in ((bias_t_h, 'et'),):
        pass
    dh = mem[H]
    da = mem[A]
    if len(dh['et']) >= MING:
        bias_t_h[i] = float(np.mean(list(dh['et'])[-WIN:]))
    if len(da['et']) >= MING:
        bias_t_a[i] = float(np.mean(list(da['et'])[-WIN:]))
    if len(dh['em']) >= MING:
        bias_m_h[i] = float(np.mean(list(dh['em'])[-WIN:]))
    if len(da['em']) >= MING:
        bias_m_a[i] = float(np.mean(list(da['em'])[-WIN:]))
    if has_ml[i] and len(dh['ed']) >= MING:
        drift_h[i] = float(np.mean(list(dh['ed'])[-WIN:]))
    if has_ml[i] and len(da['ed']) >= MING:
        drift_a[i] = float(np.mean(list(da['ed'])[-WIN:]))
    # bank strictly-prior evidence AFTER reading
    if np.isfinite(err_t[i]):
        mem[H]['et'].append(err_t[i])
        mem[A]['et'].append(err_t[i])
    if np.isfinite(err_m_h[i]):
        mem[H]['em'].append(err_m_h[i])
        mem[A]['em'].append(-err_m_h[i])
    if has_ml[i] and np.isfinite(y_ml[i]):
        mem[H]['ed'].append(pm[i] - y_ml[i])
        mem[A]['ed'].append((1 - pm[i]) - (1 - y_ml[i]))

both_t = np.isfinite(bias_t_h) & np.isfinite(bias_t_a)
both_m = np.isfinite(bias_m_h) & np.isfinite(bias_m_a)
both_d = np.isfinite(drift_h) & np.isfinite(drift_a)
print(f"coverage: totals-bias {both_t.mean():.3f} margin-bias {both_m.mean():.3f} "
      f"drift {both_d.mean():.3f} (min-5 trailing-20, both-sides)", flush=True)


def blocked_ci(dd, gd, seed=7, B=2000):
    rng = np.random.default_rng(seed)
    days = np.array(sorted(set(gd)))
    idx = {g: np.where(gd == g)[0] for g in days}
    ms = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(days), len(days))
        samp = np.concatenate([idx[days[k]] for k in pick])
        ms[b] = dd[samp].mean()
    return dd.mean(), np.percentile(ms, [2.5, 97.5])


gd = j['gameday'].to_numpy()

# ---------------- F1: dispersion -> per-game sigma ----------------
# Totals leg at market U from pace-variance OOF (control two-way q + meta).
t = pd.read_csv('nba-backend/run_diagnostics/ab_pace_variance_oof.csv',
                usecols=['game_id', 'gameday', 'U', 'fair_total', 'p_A', 'y',
                         'distribution_sd_total'])
t['game_id'] = t['game_id'].astype(str)
t['gameday'] = pd.to_datetime(t['gameday'])
t = t.sort_values(['gameday', 'game_id']).reset_index(drop=True)
tj = t.merge(s[['game_id', 'gameday']].rename(columns={'gameday': 'gd2'}),
             on='game_id', how='left')
# dispersion aligned onto totals frame via game_id
dmap = dict(zip(s['game_id'], disp_t))
tj['disp'] = tj['game_id'].map(dmap).fillna(dmap and np.median(disp_t)).to_numpy(float)
tq = np.clip(tj['p_A'].to_numpy(float), 1e-6, 1 - 1e-6)
ty = tj['y'].to_numpy(float)
td = (tj['U'] - tj['fair_total']).to_numpy(float)
tsd = tj['distribution_sd_total'].to_numpy(float)
tz = ndtri(np.clip(1 - tq, 1e-6, 1 - 1e-6))
with np.errstate(divide='ignore', invalid='ignore'):
    tsig = np.where((np.abs(td) > 1e-9) & (np.abs(tz) > 1e-9), td / tz, np.nan)
tsig = np.where(np.isfinite(tsig) & (tsig > 0), tsig, tsd)
tsig = np.where(np.isfinite(tsig) & (tsig > 0), tsig, 15.0)
tdisp = tj['disp'].to_numpy(float)
tll0 = -(ty * np.log(clip(tq)) + (1 - ty) * np.log(1 - clip(tq)))
tgd = tj['gameday'].to_numpy()
KG = [0.0, 0.02, 0.05, 0.10, 0.20]
GATE_T, CAP = 0.03, 0.12
tg = np.abs(tq - 0.5) > GATE_T
TLL = np.empty((len(tj), len(KG)))
for ki, k in enumerate(KG):
    S = 1 + k * tdisp
    qn = tq.copy()
    act = tg & np.isfinite(tsig)
    qn[act] = 1 - ndtr(td[act] / np.clip(tsig[act] * S[act], 1e-9, None))
    qn = np.clip(qn, 0.5 - CAP, 0.5 + CAP)
    qq = np.where(tg, qn, tq)
    TLL[:, ki] = -(ty * np.log(clip(qq)) + (1 - ty) * np.log(1 - clip(qq)))
BURN = 200
cum = np.cumsum(TLL, axis=0)
runmean = cum / np.arange(1, len(tj) + 1)[:, None]
tstar = np.zeros(len(tj), dtype=int)
tstar[BURN:] = np.argmin(runmean[:-BURN], axis=1)
tll_s = TLL[np.arange(len(tj)), tstar]
tll_s[:BURN] = tll0[:BURN]
tdd = tll_s - tll0
mn, (lo, hi) = blocked_ci(tdd, tgd)
print(f"\nF1-totals causal k-pick: mean {mn:+.5f} CI [{lo:+.5f},{hi:+.5f}] "
      f"{'ADOPT' if hi < 0 else 'reject'}", flush=True)
print("  k* histogram:", {k: int((tstar[BURN:] == ki).sum()) for ki, k in enumerate(KG)}, flush=True)

# Spreads leg at market home_line from edge_spread_eval (control two-way).
e = pd.read_csv('nba-backend/run_diagnostics/edge_spread_eval.csv',
                usecols=['game_id', 'gameday', 'home_line', 'fair_spread', 'p_model',
                         'y', 'push_game', 'distribution_sd_margin'])
e['game_id'] = e['game_id'].astype(str)
e['gameday'] = pd.to_datetime(e['gameday'])
e = e[~e['push_game'].astype(bool)].reset_index(drop=True)
e = e.sort_values(['gameday', 'game_id']).reset_index(drop=True)
dmapm = dict(zip(s['game_id'], disp_m))
e['disp'] = e['game_id'].map(dmapm).to_numpy(float)
e = e[np.isfinite(e['disp'])].reset_index(drop=True)
eq = np.clip(e['p_model'].to_numpy(float), 1e-6, 1 - 1e-6)
ey = e['y'].to_numpy(float)
ed = (e['home_line'] - e['fair_spread']).to_numpy(float)
esd = e['distribution_sd_margin'].to_numpy(float)
ez = ndtri(np.clip(1 - eq, 1e-6, 1 - 1e-6))
with np.errstate(divide='ignore', invalid='ignore'):
    esig = np.where((np.abs(ed) > 1e-9) & (np.abs(ez) > 1e-9), ed / ez, np.nan)
esig = np.where(np.isfinite(esig) & (esig > 0), esig, esd)
esig = np.where(np.isfinite(esig) & (esig > 0), esig, 15.0)
edisp = e['disp'].to_numpy(float)
ell0 = -(ey * np.log(clip(eq)) + (1 - ey) * np.log(1 - clip(eq)))
eg = np.abs(eq - 0.5) > 0.06
GATE_S = 0.06
ELL = np.empty((len(e), len(KG)))
for ki, k in enumerate(KG):
    S = 1 + k * edisp
    qn = eq.copy()
    act = eg & np.isfinite(esig)
    qn[act] = 1 - ndtr(ed[act] / np.clip(esig[act] * S[act], 1e-9, None))
    qn = np.clip(qn, 0.5 - CAP, 0.5 + CAP)
    qq = np.where(eg, qn, eq)
    ELL[:, ki] = -(ey * np.log(clip(qq)) + (1 - ey) * np.log(1 - clip(qq)))
cum = np.cumsum(ELL, axis=0)
runmean = cum / np.arange(1, len(e) + 1)[:, None]
estar = np.zeros(len(e), dtype=int)
estar[BURN:] = np.argmin(runmean[:-BURN], axis=1)
ell_s = ELL[np.arange(len(e)), estar]
ell_s[:BURN] = ell0[:BURN]
edd = ell_s - ell0
mn, (lo, hi) = blocked_ci(edd, e['gameday'].to_numpy())
print(f"F1-spreads causal k-pick: mean {mn:+.5f} CI [{lo:+.5f},{hi:+.5f}] "
      f"{'ADOPT' if hi < 0 else 'reject'}", flush=True)
print("  k* histogram:", {k: int((estar[BURN:] == ki).sum()) for ki, k in enumerate(KG)}, flush=True)

# ---------------- F2b: margin location shift ----------------
e2 = pd.read_csv('nba-backend/run_diagnostics/edge_spread_eval.csv')
e2['game_id'] = e2['game_id'].astype(str)
e2['gameday'] = pd.to_datetime(e2['gameday'])
e2 = e2.sort_values(['gameday', 'game_id']).reset_index(drop=True)
bmap_h = dict(zip(s['game_id'], bias_m_h))
bmap_a = dict(zip(s['game_id'], bias_m_a))
e2['bh'] = e2['game_id'].map(bmap_h)
e2['ba'] = e2['game_id'].map(bmap_a)
okm = e2['bh'].notna() & e2['ba'].notna()
e2 = e2[okm].reset_index(drop=True)
shift = np.clip(-0.5 * (e2['bh'].to_numpy(float) + e2['ba'].to_numpy(float)), -6, 6)
line = e2['home_line'].to_numpy(float)
mar2 = (e2['home_score'] - e2['away_score']).to_numpy(float)
# ex-push grading at the OFFERED line (replicates the eval-frame contract)
keep = mar2 != line
print(f"F2b population: {keep.sum()}/{len(e2)} ex-push at offered line", flush=True)
e2 = e2[keep].reset_index(drop=True)
shift = shift[keep]
line = line[keep]
mar2 = mar2[keep]
eff = np.clip(line - shift, -20.0, 20.0)  # shifted evaluation line


def _sfx(L):
    return f"m{int(abs(L))}_5" if abs(L) % 1 else (f"m{int(abs(L))}" if L < 0 else f"{int(L)}")

grid = sorted([float(f"-{i}") for i in range(20, 0, -1)] + [float(i) for i in range(0, 21)]
              + [float(f"-{i}.5") for i in range(19, 0, -1)] + [float(f"{i}.5") for i in range(0, 20)])
HC = np.column_stack([e2[f'p_home_cover_{_sfx(L)}'].to_numpy(float) for L in grid])
AC = np.column_stack([e2[f'p_away_cover_{_sfx(L)}'].to_numpy(float) for L in grid])
TW = HC / np.clip(HC + AC, 1e-12, None)  # two-way cover at each grid line
# Consistent-baseline paired grading: control AND shifted legs come from the
# SAME grid interpolation (frame p_model legs may already carry served
# transforms, so mixing baselines is a confound -- see ab_f2b_placebo.py).
# Placebo-controlled: true shift must beat sign-flip and permutation.
q0 = np.empty(len(e2))
for r in range(len(e2)):
    q0[r] = float(np.interp(line[r], grid, TW[r]))
q0 = np.clip(q0, 1e-6, 1 - 1e-6)
y2 = (mar2 > line).astype(float)
ll20 = -(y2 * np.log(q0) + (1 - y2) * np.log(1 - q0))


def _dd_of(sh):
    ef = np.clip(line - sh, -20.0, 20.0)
    q = np.empty(len(e2))
    for r in range(len(e2)):
        q[r] = float(np.interp(ef[r], grid, TW[r]))
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return (-(y2 * np.log(q) + (1 - y2) * np.log(1 - q))) - ll20


_rng = np.random.default_rng(0)
_f2b = {}
for _name, _sh in (('true', shift), ('signflip', -shift),
                   ('permuted', _rng.permutation(shift))):
    _dd = _dd_of(_sh)
    _mn, (_lo, _hi) = blocked_ci(_dd, e2['gameday'].to_numpy().astype(str))
    _f2b[_name] = (_mn, _lo, _hi)
    print(f"F2b [{_name}]: mean {_mn:+.5f} CI [{_lo:+.5f},{_hi:+.5f}]", flush=True)
_tm, _tlo, _thi = _f2b['true']
_sm, _, _ = _f2b['signflip']
_pm, _, _ = _f2b['permuted']
_verdict = 'ADOPT' if (_thi < 0 and _tm < _sm and _tm < _pm) else 'reject'
print(f"F2b margin-shift placebo-controlled verdict: {_verdict} "
      f"(true must clear 0 AND beat both placebos)", flush=True)
qi_true = np.clip(np.array([float(np.interp(x, grid, TW[r]))
                             for r, x in enumerate(np.clip(line - shift, -20., 20.))]), 1e-6, 1 - 1e-6)
print(f"  AUC control={roc_auc_score(y2, q0):.4f} shifted={roc_auc_score(y2, qi_true):.4f}", flush=True)

# ---------------- F2c: moneyline team-drift logit shift ----------------
d3 = j[has_ml].reset_index(drop=True)
d3 = d3.sort_values(['gameday', 'game_id']).reset_index(drop=True)
pm3 = np.clip(d3['p_ensemble_calibrated'].to_numpy(float), 1e-6, 1 - 1e-6)
ym3 = d3['home_win'].to_numpy(float)
ht3 = d3['home_team'].astype(str).to_numpy()
at3 = d3['away_team'].astype(str).to_numpy()
nd = len(d3)
dh3 = np.full(nd, np.nan)
da3 = np.full(nd, np.nan)
dmem = {}
for i in range(nd):
    H, A = ht3[i], at3[i]
    dmem.setdefault(H, deque())
    dmem.setdefault(A, deque())
    if len(dmem[H]) >= MING:
        dh3[i] = float(np.mean(list(dmem[H])[-WIN:]))
    if len(dmem[A]) >= MING:
        da3[i] = float(np.mean(list(dmem[A])[-WIN:]))
    if np.isfinite(ym3[i]):
        dmem[H].append(pm3[i] - ym3[i])
        dmem[A].append((1 - pm3[i]) - (1 - ym3[i]))
both3 = np.isfinite(dh3) & np.isfinite(da3)
print(f"F2c coverage both-sides: {both3.mean():.3f}", flush=True)
CAP3 = 0.05
sh3 = np.zeros(nd)
sh3[both3] = np.clip(-0.5 * (dh3[both3] - da3[both3]), -CAP3, CAP3)
q3 = np.clip(pm3 + sh3, 0.02, 0.98)
ll30 = -(ym3 * np.log(clip(pm3)) + (1 - ym3) * np.log(1 - clip(pm3)))
ll3s = -(ym3 * np.log(clip(q3)) + (1 - ym3) * np.log(1 - clip(q3)))
dd3 = ll3s - ll30
mn, (lo, hi) = blocked_ci(dd3, d3['gameday'].to_numpy().astype(str))
print(f"F2c team-drift shift: mean {mn:+.5f} CI [{lo:+.5f},{hi:+.5f}] "
      f"{'ADOPT' if hi < 0 else 'reject'}", flush=True)
print(f"  AUC control={roc_auc_score(ym3, pm3):.4f} shifted={roc_auc_score(ym3, q3):.4f}", flush=True)
edge3 = np.abs(pm3 - 0.5)
for loE, hiE in [(0, .03), (.03, .06), (.06, .10), (.10, 1)]:
    msk = both3 & (edge3 > loE) & (edge3 <= hiE)
    if msk.sum():
        print(f"  drift-covered {loE:.2f}-{hiE:.2f}: n={msk.sum():4d} mean={dd3[msk].mean():+.5f}", flush=True)
