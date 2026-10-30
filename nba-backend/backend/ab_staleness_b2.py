"""Staleness audit B (offline, fully local).

Per OOF game: hours from binding filing to tipoff + late-news flag, from the
PIT designation archive itself.

Filing clock: the archive's published_at is the report filing time; the
binding filing for a game is the LATEST published_at <= tipoff across that
game's rows. Filing age = tipoff - that max. Late-news = any filing for the
game within 24h pre-tipoff. Archive starts 2024-01-01, so games before that
are excluded from B (reported separately).

Tests (all on saved OOF, outcomes only grade):
1. |edge|, |eM|, signed eM by filing-age quartile.
2. Late-news vs not: edge rate, |eM|, signed eM, model bias by lean quintile.
3. Interaction: on late-news games, does eM point the same way as the
   market disagreement (U - fair)? A pure stale-model story says yes.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, 'nba-backend/backend')
import lineup_projection as proj

ev = pd.read_csv('nba-backend/run_diagnostics/ab_engine_refit_oof.csv',
                 usecols=['game_id', 'gameday', 'fair_total', 'total', 'U', 'p_A', 'aeA'])
ev['game_id'] = ev['game_id'].astype(str)
ev['gameday'] = pd.to_datetime(ev['gameday'])
ev['eM'] = ev['total'] - ev['fair_total']
ev['lean'] = ev['p_A'] - 0.5
ev['dis'] = ev['U'] - ev['fair_total']  # market-model disagreement in points

des = proj.load_designations(Path.home() / '.cache' / 'sports_prediction_model' / 'nba')
des['gameday'] = pd.to_datetime(des['gameday'])
des['published_at'] = pd.to_datetime(des['published_at'], errors='coerce')
des['tipoff_at'] = pd.to_datetime(des['tipoff_at'], errors='coerce')
arch_start = des['gameday'].min()
print(f"archive {arch_start.date()}..{des['gameday'].max().date()} rows={len(des)}", flush=True)
evc = ev[ev['gameday'] >= arch_start].copy()
print(f"OOF in-archive: {len(evc)}/{len(ev)}", flush=True)

# binding filing per game: max published_at over that date's rows
# (matchup-level; each date has its own filings; tipoff per game from tipoff_at)
pub = des.groupby('gameday')['published_at'].max()
tip = des.groupby('gameday')['tipoff_at'].max()
late_any = des.assign(late=des['published_at'] >= des['tipoff_at'] - pd.Timedelta(hours=24))
late = late_any.groupby('gameday')['late'].max()
nfil = des.groupby('gameday').size()

evc = evc.join(pub.rename('filed'), on='gameday')
evc = evc.join(tip.rename('tip'), on='gameday')
evc = evc.join(late.rename('late_news'), on='gameday')
evc = evc.join(nfil.rename('n_filings'), on='gameday')
evc['file_age_h'] = (evc['tip'] - evc['filed']).dt.total_seconds() / 3600
print("file_age_h desc:", evc['file_age_h'].describe().to_string(), flush=True)
print(f"late-news games: {evc['late_news'].mean():.3f}", flush=True)

print("--- 1. by filing-age quartile ---", flush=True)
evc['aq'] = pd.qcut(evc['file_age_h'], 4, labels=False, duplicates='drop')
for q, g in evc.groupby('aq'):
    print(f" Q{int(q)}: age~{g['file_age_h'].mean():.1f}h n={len(g)} |edge|={g['aeA'].mean():.4f} "
          f"|eM|={g['eM'].abs().mean():.2f} bias={g['eM'].mean():+.2f} edge_rate={(g['aeA'] > 0.03).mean():.3f}", flush=True)

print("--- 2. late-news vs not ---", flush=True)
for name, g in [('late', evc[evc.late_news]), ('not-late', evc[~evc.late_news.fillna(False)])]:
    print(f" {name}: n={len(g)} |edge|={g['aeA'].mean():.4f} |eM|={g['eM'].abs().mean():.2f} "
          f"bias={g['eM'].mean():+.2f} edge_rate={(g['aeA'] > 0.03).mean():.3f}", flush=True)
print("--- 2b. model bias by lean quintile x late-news ---", flush=True)
evc['lq'] = pd.qcut(evc['lean'], 5, labels=False, duplicates='drop')
for late, gg in [(True, evc[evc.late_news]), (False, evc[~evc.late_news.fillna(False)])]:
    print(f" late={late}:", flush=True)
    for q, g in gg.groupby('lq'):
        print(f"   LQ{int(q)}: lean~{g['lean'].mean():+.3f} n={len(g)} bias={g['eM'].mean():+.2f}", flush=True)

print("--- 3. interaction: sign(eM) vs sign(disagreement) on late-news edge games ---", flush=True)
sub = evc[(evc.aeA > 0.03) & (evc.late_news)].copy()
sub['same_sign'] = np.sign(sub['eM']) == np.sign(sub['dis'])
sub = sub[sub['dis'] != 0]
print(f" late edge games: n={len(sub)} P(eM same side as market move)={sub['same_sign'].mean():.3f} "
      f"(0.5 = no relation; stale-model story predicts HIGH)", flush=True)
for name, g in [('late edge', sub), ('not-late edge', evc[(evc.aeA > 0.03) & (~evc.late_news.fillna(False))])]:
    print(f" {name}: n={len(g)} bias={g['eM'].mean():+.2f} |eM|={g['eM'].abs().mean():.2f} "
          f"|dis|={g['dis'].abs().mean():.2f}", flush=True)

evc.to_csv('nba-backend/run_diagnostics/ab_staleness_oof.csv', index=False)
print("saved ab_staleness_oof.csv", flush=True)
