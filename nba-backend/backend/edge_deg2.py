import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
DIFFS=['elo_diff','win_pct_diff','rest_days_diff','back_to_back_diff','ewm_net_points_diff','ewm_off_rating_diff','ewm_def_rating_diff','ewm_pace_diff','ewm_efg_pct_diff','ewm_turnover_margin_diff','ewm_rebound_margin_diff','ewm_ast_per_game_diff','event_three_rate_diff','event_rim_rate_diff','event_live_tov_rate_diff','event_shot_distance_diff','event_possessions_diff','pl_dpm_c_diff','pl_dpm_f_diff','pl_dpm_g_diff','star_out_impact_diff','star_out_weight_share_diff']
feat=pd.read_parquet('nba-backend/run_diagnostics/ab_pool_gates_A_incumbent_features.parquet', columns=['game_id','gameday']+DIFFS)
mkt=pd.read_csv('nba-backend/run_diagnostics/oof_with_market_v3.csv', usecols=['game_id','gameday','home_win_x','home_win_prob_model','mkt_cons'])
feat['game_id']=feat['game_id'].astype(str); mkt['game_id']=mkt['game_id'].astype(str)
feat['gameday']=pd.to_datetime(feat['gameday']); mkt['gameday']=pd.to_datetime(mkt['gameday'])
df=mkt.merge(feat, on=['game_id','gameday'], how='inner')
print(f"joined={len(df)}", flush=True)
y=df['home_win_x'].to_numpy(float); pm=df['home_win_prob_model'].to_numpy(float); pk=df['mkt_cons'].to_numpy(float)
clip=lambda p: np.clip(p,1e-9,1-1e-9)
llm=-(y*np.log(clip(pm))+(1-y)*np.log(1-clip(pm))); llk=-(y*np.log(clip(pk))+(1-y)*np.log(1-clip(pk)))
d=llm-llk; edge=pm-pk; ae=np.abs(edge)
df=df.assign(d=d,edge=edge,ae=ae,llm=llm,llk=llk)
for name,mask in [('full',np.ones(len(df),bool)),('edge>3%',ae>0.03),('edge<=3%',ae<=0.03)]:
    s=df[mask]
    print(f"{name}: n={len(s)} mean_d={s['d'].mean():+.4f} se={s['d'].std()/np.sqrt(len(s)):.4f} P(mkt wins)={(s['d']>0).mean():.3f} llm={s['llm'].mean():.4f} llk={s['llk'].mean():.4f}", flush=True)
# fast gameday-blocked bootstrap: per-day sums
df['_day']=df['gameday'].dt.date
g=df.groupby('_day')
day_ids=np.array(list(g.groups.keys()))
day_n=g.size().to_numpy()
day_dsum=g['d'].sum().to_numpy()
day_esum=(g['ae'].apply(lambda s:(s>0.03).sum())).to_numpy()
day_edsum=g.apply(lambda t: t.loc[t['ae']>0.03,'d'].sum(), include_groups=False).to_numpy()
rng=np.random.default_rng(0); B=500; vals=np.empty(B)
N=len(day_ids)
for b in range(B):
    idx=rng.integers(0,N,N)
    tot_n=day_n[idx].sum(); tot_d=day_dsum[idx].sum()
    e_n=day_esum[idx].sum(); e_d=day_edsum[idx].sum()
    vals[b]=(e_d/e_n)-(tot_d/tot_n)
print(f"degradation(edge-full): {vals.mean():+.4f} 95%CI [{np.percentile(vals,2.5):+.4f},{np.percentile(vals,97.5):+.4f}]", flush=True)
print("edge buckets:", flush=True)
for lo,hi in [(0,.015),(.015,.03),(.03,.06),(.06,.10),(.10,1)]:
    s=df[(df['ae']>lo)&(df['ae']<=hi)]
    if len(s)==0: continue
    print(f" ae {lo:.3f}-{hi:.3f}: n={len(s):4d} mean_d={s['d'].mean():+.4f} llm={s['llm'].mean():.4f} llk={s['llk'].mean():.4f} auc_m={roc_auc_score(s['home_win_x'],s['home_win_prob_model']):.3f} auc_k={roc_auc_score(s['home_win_x'],s['mkt_cons']):.3f}", flush=True)
print("signed:", flush=True)
for name,mask in [('model-high',df['edge']>0.03),('market-high',df['edge']<-0.03)]:
    s=df[mask]
    print(f" {name}: n={len(s)} mean_d={s['d'].mean():+.4f} llm={s['llm'].mean():.4f} llk={s['llk'].mean():.4f} auc_m={roc_auc_score(s['home_win_x'],s['home_win_prob_model']):.3f} auc_k={roc_auc_score(s['home_win_x'],s['mkt_cons']):.3f}", flush=True)
sub=df[df['ae']>0.03].copy()
print(f"feat corr with d in edge subset (n={len(sub)}):", flush=True)
rows=[]
for c in DIFFS:
    v=pd.to_numeric(sub[c],errors='coerce').to_numpy(float)
    m2=np.isfinite(v)
    if m2.sum()<100: continue
    r=np.corrcoef(v[m2],sub['d'].to_numpy()[m2])[0,1]
    rows.append((c,r,float(np.nanmean(v))))
rows.sort(key=lambda t:-abs(t[1]))
for c,r,mu in rows: print(f"  {c:32s} r={r:+.3f} mean={mu:+.3f}", flush=True)
sub['conf']=np.abs(sub['home_win_prob_model']-0.5)
print("conf bins in edge:", flush=True)
for lo,hi in [(0,.05),(.05,.10),(.10,.15),(.15,.5)]:
    s=sub[(sub['conf']>lo)&(sub['conf']<=hi)]
    if len(s)==0: continue
    print(f" conf {lo:.2f}-{hi:.2f}: n={len(s):4d} mean_d={s['d'].mean():+.4f} acc_m={(((s['home_win_prob_model']>0.5)==(s['home_win_x']>0.5)).mean()):.3f} acc_k={(((s['mkt_cons']>0.5)==(s['home_win_x']>0.5)).mean()):.3f}", flush=True)
print("season in edge:", flush=True)
for yr,g2 in sub.groupby(sub['gameday'].dt.year):
    print(f"  {yr}: n={len(g2)} mean_d={g2['d'].mean():+.4f} llm={g2['llm'].mean():.4f} llk={g2['llk'].mean():.4f}", flush=True)
# feature level shift: edge vs no-edge means, standardized
print("feature mean shift edge vs no-edge (std units):", flush=True)
e=df[df['ae']>0.03]; ne=df[df['ae']<=0.03]
for c,r,mu in rows:
    v1=pd.to_numeric(e[c],errors='coerce'); v0=pd.to_numeric(ne[c],errors='coerce')
    sd=pd.to_numeric(df[c],errors='coerce').std()
    print(f"  {c:32s} edge={v1.mean():+.3f} noedge={v0.mean():+.3f} shift={(v1.mean()-v0.mean())/(sd+1e-9):+.2f}sd", flush=True)
