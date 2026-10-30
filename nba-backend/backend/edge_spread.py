import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

def lab(t):
    # threshold float -> grid label; t in 0.5 steps assumed
    if float(t).is_integer():
        v = int(t)
        return f"m{abs(v)}" if v < 0 else str(v)
    a = abs(t); whole = int(a); s = f"{whole}_5"
    return f"m{s}" if t < 0 else s

re = pd.read_csv('nba-backend/data_delivery/nba_run_engine_markets_20261029.csv')
re = re[(re['decided'] == True) & (re['frame_view'] == 'oof')].copy()
re['game_id'] = re['game_id'].astype(int)
print(f"RE oof rows={len(re)}", flush=True)
m = pd.read_csv('nba-backend/run_diagnostics/oof_with_market.csv',
                usecols=['game_id','gameday','home_line','home_score','away_score'])
m['game_id'] = m['game_id'].astype(int)
df = m.merge(re, on='game_id', how='inner', suffixes=('','_re'))
print(f"joined={len(df)}", flush=True)
df['margin'] = df['home_score'] - df['away_score']
df['T'] = -df['home_line']  # cover iff margin > T
# model raw cover prob via exact grid lookup
def get_p(r):
    c = 'p_home_cover_' + lab(r['T'])
    return r.get(c, np.nan)
df['p_raw'] = df.apply(get_p, axis=1)
oob = df['p_raw'].isna().sum()
print(f"out-of-grid (|T|>20.5): {oob}", flush=True)
df = df[np.isfinite(df['p_raw'])].copy()
# two-way for whole-number thresholds
whole = (df['T'] % 1 == 0)
push = np.zeros(len(df))
push[whole] = df[whole].apply(lambda r: r.get('p_push_' + lab(r['T']), np.nan), axis=1)
df['push'] = push
df['p_model'] = np.where(whole, df['p_raw']/(df['p_raw']+(1-df['p_raw']-df['push'])), df['p_raw'])
df['push_game'] = (df['margin'] == df['T'])
print(f"push games: {int(df['push_game'].sum())}", flush=True)
df['y'] = np.where(df['push_game'], np.nan, (df['margin'] > df['T']).astype(float))
ev = df[np.isfinite(df['y'])].copy()
print(f"eval n={len(ev)} (dropped pushes+oob)", flush=True)
ev['p_mkt'] = 0.5
ev['edge'] = ev['p_model'] - 0.5
ev['ae'] = ev['edge'].abs()
clip = lambda p: np.clip(p, 1e-9, 1-1e-9)
ev['llm'] = -(ev['y']*np.log(clip(ev['p_model']))+(1-ev['y'])*np.log(1-clip(ev['p_model'])))
ev['llk'] = -(ev['y']*np.log(0.5)+(1-ev['y'])*np.log(0.5))
ev['d'] = ev['llm'] - ev['llk']
ev['gameday'] = pd.to_datetime(ev['gameday'])
ev['line_dis'] = (ev['T'] - ev['fair_spread']).abs()

def auc_of(s, col):
    return roc_auc_score(s['y'], s[col]) if s['y'].nunique() > 1 else float('nan')

for name, s in [('full', ev), ('edge>3%', ev[ev.ae > 0.03]), ('edge<=3%', ev[ev.ae <= 0.03])]:
    acc = (((s['p_model'] > 0.5) == (s['y'] > 0.5))).mean()
    print(name + (": n=%d mean_d=%+.4f se=%.4f P(mkt wins)=%.3f llm=%.4f llk=%.4f auc_m=%.3f acc_m=%.3f cover_rate=%.3f mean_p=%.3f line_dis=%.2f" % (len(s), s['d'].mean(), s['d'].std()/np.sqrt(len(s)), (s['d'] > 0).mean(), s['llm'].mean(), s['llk'].mean(), auc_of(s, 'p_model'), acc, s['y'].mean(), s['p_model'].mean(), s['line_dis'].mean())), flush=True)

# gameday-blocked bootstrap for degradation
ev['_day'] = ev['gameday'].dt.date
g = ev.groupby('_day'); ids = np.array(list(g.groups.keys()))
dn = g.size().to_numpy(); dd = g['d'].sum().to_numpy()
de = g['ae'].apply(lambda s: (s > 0.03).sum()).to_numpy()
ded = g.apply(lambda t: t.loc[t['ae'] > 0.03, 'd'].sum(), include_groups=False).to_numpy()
rng = np.random.default_rng(0); B = 500; vals = np.empty(B); N = len(ids)
for b in range(B):
    ix = rng.integers(0, N, N)
    vals[b] = (ded[ix].sum()/de[ix].sum()) - (dd[ix].sum()/dn[ix].sum())
print(f"degradation(edge-full): {vals.mean():+.4f} 95%CI [{np.percentile(vals,2.5):+.4f},{np.percentile(vals,97.5):+.4f}]", flush=True)

print("bins on model p:", flush=True)
for lo in np.arange(0, 1, 0.05):
    hi = lo+0.05
    s = ev[(ev['p_model'] > lo) & (ev['p_model'] <= hi)] if lo > 0 else ev[(ev['p_model'] >= lo) & (ev['p_model'] <= hi)]
    if len(s) == 0: continue
    print(f" {lo:.2f}-{hi:.2f}: n={len(s):4d} cov={s['y'].mean():.3f} mp={s['p_model'].mean():.4f} "
          f"llm={s['llm'].mean():.4f} d={s['d'].mean():+.4f} auc={auc_of(s,'p_model'):.3f} ece={abs(s['p_model'].mean()-s['y'].mean()):.4f}", flush=True)

print("|edge| buckets:", flush=True)
for lo, hi in [(0, .015), (.015, .03), (.03, .06), (.06, .10), (.10, 1)]:
    s = ev[(ev['ae'] > lo) & (ev['ae'] <= hi)]
    if len(s) == 0: continue
    acc = (((s['p_model'] > 0.5) == (s['y'] > 0.5))).mean()
    print(" ae %.3f-%.3f: n=%4d mean_d=%+.4f llm=%.4f auc_m=%.3f acc_m=%.3f line_dis=%.2f" % (lo, hi, len(s), s['d'].mean(), s['llm'].mean(), auc_of(s, 'p_model'), acc, s['line_dis'].mean()), flush=True)

print("signed:", flush=True)
for name, s in [('model-home>3%', ev[ev.edge > 0.03]), ('model-away>3%', ev[ev.edge < -0.03])]:
    acc = (((s['p_model'] > 0.5) == (s['y'] > 0.5))).mean()
    print(" " + name + (": n=%d mean_d=%+.4f auc=%.3f acc=%.3f" % (len(s), s['d'].mean(), auc_of(s, 'p_model'), acc)), flush=True)

sub = ev[ev.ae > 0.03].copy()
sub['conf'] = sub['ae']
print("confidence bins in edge:", flush=True)
for lo, hi in [(0.03, .06), (.06, .10), (.10, .20), (.20, .5)]:
    s = sub[(sub['conf'] > lo) & (sub['conf'] <= hi)]
    if len(s) == 0: continue
    acc = (((s['p_model']>0.5)==(s['y']>0.5))).mean()
    print(" conf %.2f-%.2f: n=%4d mean_d=%+.4f acc=%.3f" % (lo, hi, len(s), s['d'].mean(), acc), flush=True)
print("season in edge:", flush=True)
for yr, g2 in sub.groupby(sub['gameday'].dt.year):
    print(f"  {yr}: n={len(g2)} mean_d={g2['d'].mean():+.4f} llm={g2['llm'].mean():.4f} auc={auc_of(g2,'p_model'):.3f}", flush=True)

# feature correlations with d in edge subset
DIFFS = ['elo_diff','win_pct_diff','rest_days_diff','back_to_back_diff','ewm_net_points_diff','ewm_off_rating_diff','ewm_def_rating_diff','ewm_pace_diff','ewm_efg_pct_diff','ewm_turnover_margin_diff','ewm_rebound_margin_diff','ewm_ast_per_game_diff','event_three_rate_diff','event_rim_rate_diff','event_live_tov_rate_diff','event_shot_distance_diff','event_possessions_diff','pl_dpm_c_diff','pl_dpm_f_diff','pl_dpm_g_diff','star_out_impact_diff','star_out_weight_share_diff']
feat = pd.read_parquet('nba-backend/run_diagnostics/ab_pool_gates_A_incumbent_features.parquet', columns=['game_id','gameday']+DIFFS)
feat['game_id'] = feat['game_id'].astype(str)
sub2 = sub.copy(); sub2['game_id'] = sub2['game_id'].astype(str)
feat['gameday'] = pd.to_datetime(feat['gameday']); sub2['gameday'] = pd.to_datetime(sub2['gameday'])
j = sub2.merge(feat, on=['game_id','gameday'], how='left')
print(f"feat join: {len(j)} rows", flush=True)
rows = []
for c in DIFFS:
    v = pd.to_numeric(j[c], errors='coerce').to_numpy(float)
    m2 = np.isfinite(v)
    if m2.sum() < 50: continue
    r = np.corrcoef(v[m2], j['d'].to_numpy()[m2])[0, 1]
    rows.append((c, r))
rows.sort(key=lambda t: -abs(t[1]))
print("feat corr with d (edge):", flush=True)
for c, r in rows: print(f"  {c:32s} r={r:+.3f}", flush=True)
ev.to_csv('nba-backend/run_diagnostics/edge_spread_eval.csv', index=False)
print("saved edge_spread_eval.csv", flush=True)
