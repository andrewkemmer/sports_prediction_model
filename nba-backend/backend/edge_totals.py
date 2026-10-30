import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

def lab(u):
    if float(u).is_integer():
        return str(int(u))
    return f"{int(u)}_5"

re = pd.read_csv('nba-backend/data_delivery/nba_run_engine_markets_20261029.csv')
re = re[(re['decided'] == True) & (re['frame_view'] == 'oof')].copy()
re['game_id'] = re['game_id'].astype(int)
print("RE oof rows=%d" % len(re), flush=True)
m = pd.read_csv('nba-backend/run_diagnostics/oof_with_market.csv',
                usecols=['game_id', 'gameday', 'total_line', 'home_score', 'away_score'])
m['game_id'] = m['game_id'].astype(int)
df = m.merge(re, on='game_id', how='inner', suffixes=('', '_re'))
print("joined=%d" % len(df), flush=True)
df['total'] = df['home_score'] + df['away_score']
df['U'] = df['total_line']
df['p_raw'] = df.apply(lambda r: r.get('p_over_' + lab(r['U'])), axis=1)
oob = int(df['p_raw'].isna().sum())
print("out-of-grid: %d" % oob, flush=True)
df = df[np.isfinite(df['p_raw'])].copy()
whole = (df['U'] % 1 == 0)
push = np.zeros(len(df))
pv = df[whole].apply(lambda r: r.get('p_push_total_' + lab(r['U']), np.nan), axis=1).to_numpy(float)
push[np.where(whole)[0]] = pv
df['push'] = push
df['p_model'] = np.where(whole, df['p_raw'] / (df['p_raw'] + (1 - df['p_raw'] - df['push'])), df['p_raw'])
df['push_game'] = (df['total'] == df['U'])
print("push games: %d" % int(df['push_game'].sum()), flush=True)
df['y'] = np.where(df['push_game'], np.nan, (df['total'] > df['U']).astype(float))
ev = df[np.isfinite(df['y'])].copy()
print("eval n=%d" % len(ev), flush=True)
ev['p_mkt'] = 0.5
ev['edge'] = ev['p_model'] - 0.5
ev['ae'] = ev['edge'].abs()
clip = lambda p: np.clip(p, 1e-9, 1 - 1e-9)
ev['llm'] = -(ev['y'] * np.log(clip(ev['p_model'])) + (1 - ev['y']) * np.log(1 - clip(ev['p_model'])))
ev['llk'] = -(ev['y'] * np.log(0.5) + (1 - ev['y']) * np.log(0.5))
ev['d'] = ev['llm'] - ev['llk']
ev['gameday'] = pd.to_datetime(ev['gameday'])
ev['line_dis'] = (ev['U'] - ev['fair_total']).abs()

def auc_of(s, col):
    return roc_auc_score(s['y'], s[col]) if s['y'].nunique() > 1 else float('nan')

def acc_of(s):
    return ((((s['p_model'] > 0.5) == (s['y'] > 0.5))).mean())

for name, s in [('full', ev), ('edge>3%', ev[ev.ae > 0.03]), ('edge<=3%', ev[ev.ae <= 0.03])]:
    print(name + (": n=%d mean_d=%+.4f se=%.4f P(mkt wins)=%.3f llm=%.4f llk=%.4f auc_m=%.3f acc_m=%.3f over_rate=%.3f mean_p=%.3f line_dis=%.2f" % (len(s), s['d'].mean(), s['d'].std() / np.sqrt(len(s)), (s['d'] > 0).mean(), s['llm'].mean(), s['llk'].mean(), auc_of(s, 'p_model'), acc_of(s), s['y'].mean(), s['p_model'].mean(), s['line_dis'].mean())), flush=True)

ev['_day'] = ev['gameday'].dt.date
g = ev.groupby('_day')
ids = np.array(list(g.groups.keys()))
dn = g.size().to_numpy()
dd = g['d'].sum().to_numpy()
de = g['ae'].apply(lambda s: (s > 0.03).sum()).to_numpy()
ded = g.apply(lambda t: t.loc[t['ae'] > 0.03, 'd'].sum(), include_groups=False).to_numpy()
rng = np.random.default_rng(0)
B = 500
vals = np.empty(B)
N = len(ids)
for b in range(B):
    ix = rng.integers(0, N, N)
    vals[b] = (ded[ix].sum() / de[ix].sum()) - (dd[ix].sum() / dn[ix].sum())
print("degradation(edge-full): %+.4f 95%%CI [%+.4f,%+.4f]" % (vals.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)), flush=True)

print("bins on model p:", flush=True)
for lo in np.arange(0, 1, 0.05):
    hi = lo + 0.05
    s = ev[(ev['p_model'] > lo) & (ev['p_model'] <= hi)] if lo > 0 else ev[(ev['p_model'] >= lo) & (ev['p_model'] <= hi)]
    if len(s) == 0:
        continue
    print(" %.2f-%.2f: n=%4d over=%.3f mp=%.4f llm=%.4f d=%+.4f auc=%.3f ece=%.4f" % (lo, hi, len(s), s['y'].mean(), s['p_model'].mean(), s['llm'].mean(), s['d'].mean(), auc_of(s, 'p_model'), abs(s['p_model'].mean() - s['y'].mean())), flush=True)

print("|edge| buckets:", flush=True)
for lo, hi in [(0, .015), (.015, .03), (.03, .06), (.06, .10), (.10, 1)]:
    s = ev[(ev['ae'] > lo) & (ev['ae'] <= hi)]
    if len(s) == 0:
        continue
    print(" ae %.3f-%.3f: n=%4d mean_d=%+.4f llm=%.4f auc_m=%.3f acc_m=%.3f line_dis=%.2f" % (lo, hi, len(s), s['d'].mean(), s['llm'].mean(), auc_of(s, 'p_model'), acc_of(s), s['line_dis'].mean()), flush=True)

print("signed:", flush=True)
for name, s in [('model-over>3%', ev[ev.edge > 0.03]), ('model-under>3%', ev[ev.edge < -0.03])]:
    print(" " + name + (": n=%d mean_d=%+.4f auc=%.3f acc=%.3f" % (len(s), s['d'].mean(), auc_of(s, 'p_model'), acc_of(s))), flush=True)

sub = ev[ev.ae > 0.03].copy()
print("confidence bins in edge:", flush=True)
for lo, hi in [(0.03, .06), (.06, .10), (.10, .20), (.20, .5)]:
    s = sub[(sub['ae'] > lo) & (sub['ae'] <= hi)]
    if len(s) == 0:
        continue
    print(" conf %.2f-%.2f: n=%4d mean_d=%+.4f acc=%.3f" % (lo, hi, len(s), s['d'].mean(), acc_of(s)), flush=True)
print("season in edge:", flush=True)
for yr, g2 in sub.groupby(sub['gameday'].dt.year):
    print("  %d: n=%d mean_d=%+.4f llm=%.4f auc=%.3f" % (yr, len(g2), g2['d'].mean(), g2['llm'].mean(), auc_of(g2, 'p_model')), flush=True)

DIFFS = ['elo_diff', 'win_pct_diff', 'rest_days_diff', 'back_to_back_diff', 'ewm_net_points_diff', 'ewm_off_rating_diff', 'ewm_def_rating_diff', 'ewm_pace_diff', 'ewm_efg_pct_diff', 'ewm_turnover_margin_diff', 'ewm_rebound_margin_diff', 'ewm_ast_per_game_diff', 'event_three_rate_diff', 'event_rim_rate_diff', 'event_live_tov_rate_diff', 'event_shot_distance_diff', 'event_possessions_diff', 'pl_dpm_c_diff', 'pl_dpm_f_diff', 'pl_dpm_g_diff', 'star_out_impact_diff', 'star_out_weight_share_diff']
feat = pd.read_parquet('nba-backend/run_diagnostics/ab_pool_gates_A_incumbent_features.parquet', columns=['game_id', 'gameday'] + DIFFS)
feat['game_id'] = feat['game_id'].astype(str)
sub2 = sub.copy()
sub2['game_id'] = sub2['game_id'].astype(str)
feat['gameday'] = pd.to_datetime(feat['gameday'])
sub2['gameday'] = pd.to_datetime(sub2['gameday'])
j = sub2.merge(feat, on=['game_id', 'gameday'], how='left')
print("feat join: %d rows" % len(j), flush=True)
rows = []
for c in DIFFS:
    v = pd.to_numeric(j[c], errors='coerce').to_numpy(float)
    m2 = np.isfinite(v)
    if m2.sum() < 50:
        continue
    r = np.corrcoef(v[m2], j['d'].to_numpy()[m2])[0, 1]
    rows.append((c, r))
rows.sort(key=lambda t: -abs(t[1]))
print("feat corr with d (edge):", flush=True)
for c, r in rows:
    print("  %-32s r=%+.3f" % (c, r), flush=True)
ev.to_csv('nba-backend/run_diagnostics/edge_totals_eval.csv', index=False)
print("saved edge_totals_eval.csv", flush=True)
