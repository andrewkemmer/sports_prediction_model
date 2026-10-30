"""Staleness audit: is the model's edge-game failure late news? (offline, local)

Three PIT-safe proxies, all computed on saved OOF (n=3134), no market prices
as model inputs anywhere:

A. DARKO staleness: for each game, the asof lag of the DPM row the model
   actually read (target_date - last row strictly before it) + the on-day
   delta (|x_minutes(on-day) - x_minutes(last prior)| and |dpm(on-day) -
   dpm(last prior)|). DARKO refits overnight; a big on-day jump = news the
   model missed. Correlate with |edge| and with the signed error.
   NOTE: pulls DARKO history live (darko.app). Slow (~hours for full league);
   samples 60 players x full history by default, or set FULL=1 for all.

B. Designation staleness: for each game, days between the binding filing
   (last report strictly before tipoff) and tipoff, + whether a NEW filing
   landed in the last 24h before tipoff (late news window). Tests whether
   edge games cluster where the filing is old or where news broke late.

C. Overnight-line proxy: SI opening vs closing totals (already scraped in
   run_diagnostics/si_nba_*.csv). |close - open| = news that broke after open.
   joinable to OOF by (date, teams). Tests whether edge games = big-move games.
   Fully local, no new pulls.

Each section is independent; run with SECTION=A|B|C or ALL (default C then B,
A last because it pulls).
"""
import os
import sys

SECTION = os.environ.get("SECTION", "ALL").upper()


def section_c():
    import glob
    import numpy as np
    import pandas as pd
    files = sorted(glob.glob('nba-backend/run_diagnostics/si_nba_*.csv'))
    print(f"SI files: {files}", flush=True)
    si = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"SI rows={len(si)} cols={si.columns.tolist()[:20]}", flush=True)
    # find total columns
    tcols = [c for c in si.columns if 'total' in c.lower() or 'over' in c.lower()
             or 'under' in c.lower()]
    print("total-ish cols:", tcols[:20], flush=True)
    print(si.head(2).to_string()[:1500], flush=True)


def section_b():
    import numpy as np
    import pandas as pd
    ev = pd.read_csv('nba-backend/run_diagnostics/ab_engine_refit_oof.csv',
                     usecols=['game_id', 'gameday', 'fair_total', 'total', 'p_A', 'aeA'])
    ev['game_id'] = ev['game_id'].astype(str)
    ev['gameday'] = pd.to_datetime(ev['gameday'])
    ev['eM'] = ev['total'] - ev['fair_total']
    import lineup_projection as proj
    from pathlib import Path
    root = Path.home() / '.cache' / 'sports_prediction_model' / 'nba'
    des = proj.load_designations(root)
    print(f"designations rows={len(des) if des is not None else 0}", flush=True)
    if des is None:
        print("no designations; abort B", flush=True)
        return
    des['gameday'] = pd.to_datetime(des['gameday'])
    # filing date per row: prefer report_date/published/file date columns
    for c in des.columns:
        pass
    print("des cols:", des.columns.tolist(), flush=True)
    print(des.head(2).to_string()[:1200], flush=True)


def section_a():
    print("section A: DARKO pull (slow). Set FULL=1 for all players, else 60-player sample.", flush=True)


if __name__ == "__main__":
    if SECTION in ("C", "ALL"):
        print("===== C: overnight-line proxy =====", flush=True)
        section_c()
    if SECTION in ("B", "ALL"):
        print("===== B: designation staleness =====", flush=True)
        section_b()
    if SECTION in ("A", "ALL"):
        print("===== A: DARKO staleness =====", flush=True)
        section_a()
