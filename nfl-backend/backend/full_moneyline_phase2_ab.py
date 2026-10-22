"""Two-arm moneyline OOF: adopted a=0.9 vs phase-2 spill (background job).

Paired design: identical sources, identical folds; only the epa_* columns
differ (phase-2 = cross-position spill matrix on top of the adopted a=0.9
same-position backbone). Reports evaluation.binary_metrics on p_ensemble
for all grading rows + the redistribution-affected slice, with paired
log-loss delta SE.
"""
import sys
sys.path.insert(0, '.')
import logging
import numpy as np
import pandas as pd

import ingestion
import features as feat_mod
import folds as folds_mod
import moneyline as ml_mod
import evaluation as eval_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("phase2_ab")

SEASONS = list(range(2016, 2026))

schedule = ingestion.load_schedule(seasons=SEASONS, use_cache=True)
schedule = ingestion.eligible_games(schedule)
schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce")
log.info("schedule rows: %d", len(schedule))
try:
    pit_weather = __import__("weather").cached_weather_for_games(schedule)
except Exception as exc:
    pit_weather = pd.DataFrame()
    log.warning("weather unavailable, stays NaN: %s", exc)
log.info("weather rows: %d", len(pit_weather))

pbp = ingestion.load_pbp(seasons=SEASONS, use_cache=True)
ps = ingestion.load_player_stats(seasons=[SEASONS[0] - 1] + SEASONS, use_cache=True)
ngs = ingestion.load_nextgen(seasons=[SEASONS[0] - 1] + SEASONS, use_cache=True)
snaps = ingestion.load_snap_counts(seasons=ingestion.snap_count_seasons(SEASONS), use_cache=True)
ftn = ingestion.load_ftn_charting(seasons=SEASONS, use_cache=True)
injuries = ingestion.load_injuries_pit(schedule, seasons=SEASONS, use_cache=True,
                                       refresh_upcoming=False)
weekly_injuries = ingestion.load_injuries_weekly(seasons=SEASONS, use_cache=True)
crosswalk = ingestion.load_player_id_crosswalk(use_cache=True)
weekly_rosters = ingestion.load_weekly_rosters(seasons=SEASONS, use_cache=True)
roster_unavailable = ingestion.roster_unavailable_table(weekly_rosters)

common = dict(pbp=pbp, ps=ps, ngs=ngs, snaps=snaps, ftn=ftn, weather=pit_weather,
              injuries=injuries, weekly_injuries=weekly_injuries, crosswalk=crosswalk,
              roster_unavailable=roster_unavailable)
log.info("building adopted a=0.9 features ...")
a09_df = feat_mod.build_game_features(schedule, **common)
log.info("a09 frame: %s", a09_df.shape)
log.info("building phase-2 spill features ...")
ph2_df = feat_mod.build_game_features(
    schedule, **common, spill_matrix=feat_mod.PHASE2_SPILL_ESTIMATE)
log.info("ph2 frame: %s", ph2_df.shape)

epa_cols = [c for c in a09_df.columns if c.startswith("epa_")
            and (c.endswith("_home") or c.endswith("_away") or c.endswith("_diff"))]
ao = a09_df[["game_id"] + epa_cols].copy()
po = ph2_df[["game_id"] + epa_cols].copy()
ao["game_id"] = ao["game_id"].astype(str)
po["game_id"] = po["game_id"].astype(str)
j = ao.merge(po, on="game_id", suffixes=("_a", "_p"))
dmx = np.abs(j[[c + "_p" for c in epa_cols]].to_numpy(float)
             - j[[c + "_a" for c in epa_cols]].to_numpy(float)).max(axis=1)
affected_ids = set(j.loc[dmx > 0.05, "game_id"])
log.info("affected games (|depa|>0.05): %d / %d", len(affected_ids), len(j))
pd.DataFrame({"game_id": sorted(affected_ids)}).to_csv(
    "../run_diagnostics/phase2_ab_affected.csv", index=False)

fold_list = folds_mod.make_folds(
    folds_mod.canonical_sort(a09_df, "gameday"), date_col="gameday")
log.info("folds: %d", len(fold_list))

out = []
for name, df in (("a09", a09_df), ("ph2", ph2_df)):
    log.info("OOF walk: %s ...", name)
    res = ml_mod.walk_forward_oof(df, date_col="gameday", fold_list=fold_list)
    oof = res["oof"]
    oof.to_pickle(f"../run_diagnostics/phase2_ab_oof_{name}.pkl")
    y = oof["home_win"].astype(float).to_numpy()
    p = pd.to_numeric(oof["p_ensemble"], errors="coerce").to_numpy(dtype=float)
    grade = (oof["grades_pooled"].astype(bool).to_numpy()
             if "grades_pooled" in oof else np.ones(len(oof), bool))
    aff = oof["game_id"].astype(str).isin(affected_ids).to_numpy()
    for slice_name, m in (("all_grading", grade),
                          ("affected_grading", grade & aff)):
        m = np.asarray(m, dtype=bool)
        met = eval_mod.binary_metrics(p[m], y[m])
        met.update({"arm": name, "slice": slice_name})
        out.append(met)
        log.info("%s %s: %s", name, slice_name, met)
    if name == "ph2":
        yo = oof["home_win"].astype(float).to_numpy()
        po = pd.to_numeric(oof["p_ensemble"], errors="coerce").to_numpy(dtype=float)
        ref = pd.read_pickle("../run_diagnostics/phase2_ab_oof_a09.pkl")
        yb = ref["home_win"].astype(float).to_numpy()
        pb = pd.to_numeric(ref["p_ensemble"], errors="coerce").to_numpy(dtype=float)
        gb = ref["grades_pooled"].astype(bool).to_numpy()
        assert (yb == yo).all() and (gb == grade).all()

        def _ll(yy, pp):
            pp = np.clip(pp, 1e-7, 1 - 1e-7)
            return -(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))

        d = _ll(yo[grade], po[grade]) - _ll(yb[gb], pb[gb])
        out.append({"arm": "delta(ph2-a09)", "slice": "all_grading",
                    "n": int(grade.sum()), "mean_dll": float(d.mean()),
                    "se_dll": float(d.std(ddof=1) / np.sqrt(len(d)))})
        log.info("paired dll mean=%.6f se=%.6f", float(d.mean()),
                 float(d.std(ddof=1) / np.sqrt(len(d))))

res_df = pd.DataFrame(out)
res_df.to_csv("../run_diagnostics/phase2_ab_moneyline.csv", index=False)
log.info("SAVED run_diagnostics/phase2_ab_moneyline.csv")
print(res_df.to_string(index=False))
