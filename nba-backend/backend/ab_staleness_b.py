"""Staleness audit B+C (offline, fully local, no new pulls).

B: designation staleness. For each OOF game: hours from binding filing
   (published_at of the last report strictly before tipoff, per team) to
   tipoff; + whether ANY filing for either team landed within 24h pre-tip
   (late-news window). Tests: does |edge| / |eM| grow with filing age, or
   cluster where news broke late?

C: overnight-move proxy. SI scraped file carries ONE line per game (1.91/1.91
   both sides, no open/close legs) -- so no open-vs-close move exists in this
   source. Fallback: |market total - fair_total| IS the disagreement; test
   whether |eM| grows with disagreement (already known: yes) AND whether the
   model's error is one-sided conditional on filing age (B) -- i.e. the
   interaction filing_age x disagreement. A pure staleness story predicts the
   model's error points the SAME way as the market move on late-news games.

All joins PIT-safe (filings strictly before tipoff; OOF outcomes only grade).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ev = pd.read_csv('nba-backend/run_diagnostics/ab_engine_refit_oof.csv',
                 usecols=['game_id', 'gameday', 'fair_total', 'total', 'U', 'p_A', 'aeA'])
ev['game_id'] = ev['game_id'].astype(str)
ev['gameday'] = ev['gameday'].astype(str)
ev['tip'] = pd.to_datetime(ev['gameday'])
ev['eM'] = ev['total'] - ev['fair_total']
ev['lean'] = ev['p_A'] - 0.5

import lineup_projection as proj
des = proj.load_designations(Path.home() / '.cache' / 'sports_prediction_model' / 'nba')
des['gameday'] = des['gameday'].astype(str)
des['published_at'] = pd.to_datetime(des['published_at'], errors='coerce')
des['tipoff_at'] = pd.to_datetime(des['tipoff_at'], errors='coerce')
# binding filing per (gameday, team): latest published_at strictly before tipoff
# published_at_tipoff column exists; use max published_at <= tipoff per team-game
j = des.merge(ev[['game_id', 'gameday']].drop_duplicates(), on='gameday', how='inner')
print(f"des rows on OOF dates: {len(j)}", flush=True)
# keep filings strictly before tipoff: join tipoff per game via tipoff_at
tip = j.groupby('gameday')['tipoff_at'].max().to_dict()
ev_tip = ev.copy()
ev_tip['tipoff_at'] = ev_tip['gameday'].map(tip)
jj = j.merge(ev_tip[['game_id', 'gameday', 'tipoff_at']].rename(
    columns={'tipoff_at': 'tip'}), on=['game_id', 'gameday'], how='left')
