import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

feat = pd.read_parquet('nba-backend/run_diagnostics/ab_pool_gates_A_incumbent_features.parquet')
mkt = pd.read_csv('nba-backend/run_diagnostics/oof_with_market_v3.csv')
feat['game_id'] = feat['game_id'].astype(str)
mkt['game_id'] = mkt['game_id'].astype(str)
feat['gameday'] = pd.to_datetime(feat['gameday'])
mkt['gameday'] = pd.to_datetime(mkt['gameday'])
df = mkt.merge(feat, on=['game_id','gameday'], how='inner', suffixes=('','_f'))
print(f"joined={len(df)}")
y = df['home_win_x'].to_numpy(float)
pm = df['home_win_prob_model'].to_numpy(float)
pk = df['mkt_cons'].to_numpy(float)
def ll(p):
    p=np.clip(p,1e-9,1-1e-9); return -(y*np.log(p)+(1-y)*np.log(1-p))
llm, llk = ll(pm), ll(pk)
d = llm-llk
edge = pm-pk
ae = np.abs(edge)
df['d']=d; df['edge']=edge; df['ae']=ae; df['llm']=llm; df['llk']=llk
# 1. paired overall vs edge>3%
for name, mask in [('full', np.ones(len(df),bool)), ('edge>3%', ae>0.03), ('edge<=3%', ae<=0.03)]:
    s=df[mask]
    print(f"{name}: n={len(s)} mean_d={s['d'].mean():+.4f} se={s['d'].std()/np.sqrt(len(s)):.4f} frac_mkt_wins={(s['d']>0).mean():.3f} llm={s['llm'].mean():.4f} llk={s['llk'].mean():.4f} acc_m={((s['home_win_prob_model']>0.5)==(s['home_win_x']>0.5)).mean():.3f}")
# gameday-blocked bootstrap for degradation (edge minus full)
rng=np.random.default_rng(0)
days=df['gameday'].unique()
B=2000
vals=[]
for _ in range(B):
    samp_days=rng.choice(days, len(days), replace=True)
    s=pd.concat([df[df['gameday']==dd] for dd in samp_days])
    vals.append(s[s['ae']>0.03]['d'].mean()-s['d'].mean())
vals=np.array(vals)
print(f"degradation(edge-full) mean={vals.mean():+.4f} 95%CI [{np.percentile(vals,2.5):+.4f},{np.percentile(vals,97.5):+.4f}]")
# 2. edge buckets
print("\nedge buckets:")
for lo,hi in [(0,.015),(.015,.03),(.03,.06),(.06,.10),(.10,1)]:
    s=df[(df['ae']>lo)&(df['ae']<=hi)]
    if len(s)==0: continue
    print(f"ae {lo:.3f}-{hi:.3f}: n={len(s):4d} mean_d={s['d'].mean():+.4f} llm={s['llm'].mean():.4f} llk={s['llk'].mean():.4f} auc_m={roc_auc_score(s['home_win_x'],s['home_win_prob_model']):.3f} auc_k={roc_auc_score(s['home_win_x'],s['mkt_cons']):.3f} acc={((s['home_win_prob_model']>0.5)==(s['home_win_x']>0.5)).mean():.3f}")
# signed edge direction
print("\nsigned edge:")
for name, mask in [('model high (pm-pk>+3%)', df['edge']>0.03), ('market high (pm-pk<-3%)', df['edge']<-0.03)]:
    s=df[mask]
    print(f"{name}: n={len(s)} mean_d={s['d'].mean():+.4f} llm={s['llm'].mean():.4f} llk={s['llk'].mean():.4f} auc_m={roc_auc_score(s['home_win_x'],s['home_win_prob_model']):.3f} auc_k={roc_auc_score(s['home_win_x'],s['mkt_cons']):.3f}")
# 3. feature gaps: correlate features with d within edge subset
cands=[c for c in feat.columns if c.endswith('_diff') and not c.startswith('_')]
# drop leakage-ish: scores/margin/total
cands=[c for c in cands if c not in ('margin','total')]
sub=df[df['ae']>0.03].copy()
print(f"\nfeature corr with d (edge subset, n={len(sub)}):")
rows=[]
for c in cands:
    v=pd.to_numeric(sub[c],errors='coerce')
    m2=v.notna() & np.isfinite(v)
    if m2.sum()<100: continue
    r=np.corrcoef(v[m2], sub.loc[m2,'d'])[0,1]
    rows.append((c, r, v[m2].mean(), sub.loc[m2,'d'].mean()))
rows.sort(key=lambda t: -abs(t[1]))
for c,r,mu,md in rows[:25]:
    print(f"  {c:38s} r={r:+.3f} mean={mu:+.3f}")
print("... bottom 5:")
for c,r,mu,md in rows[-5:]:
    print(f"  {c:38s} r={r:+.3f} mean={mu:+.3f}")
# group means: high-|d| vs low
df['big_loss']=df['d']>0.2
print("\nfeature means: big model loss (d>0.2) vs rest:")
for c,_ ,_,_ in rows[:12]:
    v=pd.to_numeric(df[c],errors='coerce')
    print(f"  {c:38s} big={v[df['big_loss']].mean():+.3f} rest={v[~df['big_loss']].mean():+.3f}")
# model confidence vs correctness in edge subset
print("\nmodel confidence bins within edge subset:")
sub['conf']=np.abs(sub['home_win_prob_model']-0.5)
for lo,hi in [(0,.05),(.05,.10),(.10,.15),(.15,.25),(.25,.5)]:
    s=sub[(sub['conf']>lo)&(sub['conf']<=hi)]
    if len(s)==0: continue
    print(f"conf {lo:.2f}-{hi:.2f}: n={len(s):4d} mean_d={s['d'].mean():+.4f} acc_m={((s['home_win_prob_model']>0.5)==(s['home_win_x']>0.5)).mean():.3f} acc_k={((s['mkt_cons']>0.5)==(s['home_win_x']>0.5)).mean():.3f}")
# season trend
print("\nby season (edge subset):")
for s_,g in sub.groupby(sub['gameday'].dt.year):
    print(f"  {s_}: n={len(g)} mean_d={g['d'].mean():+.4f} llm={g['llm'].mean():.4f} llk={g['llk'].mean():.4f}")
df.to_csv('nba-backend/run_diagnostics/edge_degradation_joined.csv', index=False)
print("saved edge_degradation_joined.csv")
