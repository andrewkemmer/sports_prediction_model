"""Spike F5: referee-crew history coverage for NBA totals.

1. Maps cached games -> stats.nba.com GAME_IDs.
2. Live-probes boxscoresummaryv2 Officials for a handful of games.
3. On success, pulls a ~200-game sample: coverage, games/referee,
   foul-rate and total splits, home-edge splits, shrinkage feasibility.
"""
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

CACHE = os.path.expanduser('~/.cache/sports_prediction_model/nba')
STATS_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "sports-prediction-model/1.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}
SUMMARY_URL = "https://stats.nba.com/stats/boxscoresummaryv2"


def get_summary(stats_game_id):
    params = {"GameID": stats_game_id}
    url = SUMMARY_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=STATS_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def officials_of(payload):
    for rs in payload.get("resultSets", []):
        if rs.get("name") == "Officials":
            headers = rs.get("headers", [])
            out = []
            for row in rs.get("rowSet", []):
                out.append(dict(zip(headers, row)))
            return out
    return []


def main():
    games = pd.read_parquet(os.path.join(CACHE, 'games.parquet'))
    print("games.parquet: %s rows" % len(games))
    print("columns: %s" % [c for c in games.columns][:30])
    stat_cols = [c for c in games.columns if 'stat' in c.lower() or 'nba_id' in c.lower()
                 or c.upper() in ('GAME_ID', 'STATS_GAME_ID', 'NBA_GAME_ID')]
    print("id-like columns: %s" % stat_cols)
    for c in stat_cols[:5]:
        print("  %s: %s" % (c, games[c].dropna().iloc[:3].tolist()))
    # ---- sample pull: 200 games stratified by season ----
    sample = games.groupby('season', group_keys=False).apply(
        lambda d: d.sample(min(len(d), 50), random_state=7)).reset_index(drop=True)
    print("sample: %d games" % len(sample))
    rows = []
    for i, gid in enumerate(sample['game_id'].tolist()):
        try:
            off = officials_of(get_summary(gid))
            for pos, o in enumerate(off):
                rows.append({"game_id": gid, "pos": pos,
                             "ref": "%s %s" % (o.get('FIRST_NAME'), o.get('LAST_NAME')),
                             "jersey": o.get('JERSEY_NUM')})
        except Exception as exc:  # noqa: BLE001 - spike tallies misses
            print("  miss %s: %s" % (gid, str(exc)[:80]))
        time.sleep(0.6)
        if (i + 1) % 50 == 0:
            print("  ...%d/%d" % (i + 1, len(sample)), flush=True)
    crews = pd.DataFrame(rows)
    crews.to_csv('nba-backend/run_diagnostics/spike_ref_crews_sample.csv', index=False)
    cov = sample['game_id'].isin(crews['game_id']).mean()
    print("coverage: %.3f of sampled games have >=1 official" % cov)
    print("games with full 3-man crew: %.3f"
          % crews.groupby('game_id').size().eq(3).mean())
    cnt = crews['ref'].value_counts()
    print("distinct referees: %d; games/ref: median %.0f, p10 %.0f" % (
        len(cnt), cnt.median(), cnt.quantile(0.10)))
    print(cnt.head(8).to_string())
    g = games.set_index('game_id')
    crews['total'] = crews['game_id'].map(g['total'])
    crews['margin'] = crews['game_id'].map(g['margin'])
    crews['season'] = crews['game_id'].map(g['season'])
    refstat = crews.groupby('ref').agg(n=('total', 'size'),
                                        mean_total=('total', 'mean'),
                                        mean_margin=('margin', 'mean'))
    refstat = refstat[refstat['n'] >= 5].sort_values('mean_total')
    print("refs with >=5 sampled games: %d" % len(refstat))
    print("spread in ref mean totals: %.1f (min %.1f, max %.1f)" % (
        refstat['mean_total'].max() - refstat['mean_total'].min(),
        refstat['mean_total'].min(), refstat['mean_total'].max()))
    print("spread in ref mean home-margin: %.1f" % (
        refstat['mean_margin'].max() - refstat['mean_margin'].min()))
    # full-history cost extrapolation
    print("full pull estimate: %d games x ~1.1s / 4 workers ~= %.0f min" % (
        len(games), len(games) * 1.1 / 4 / 60))


if __name__ == "__main__":
    main()
