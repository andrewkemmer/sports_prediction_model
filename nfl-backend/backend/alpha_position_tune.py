"""Per-position alpha tune (background job).

Arms (one-at-a-time moves off adopted): adopted {QB:1.0,WR:0.9,TE:0.9,RB:0.9},
WR1.0, TE1.0, RB0.7, RB0.5. Each arm: full build + walk-forward OOF on
identical folds; binary_metrics on all grading rows + affected slice, paired
dLL vs adopted. Cheapest honest target (edge-logistic) pointed the wrong way
on phase-1, so every candidate pays the full-OOF price here.
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
log = logging.getLogger("alpha_tune")

ADOPTED = {"QB": 1.0, "WR": 0.9, "TE": 0.9, "RB": 0.9}
ARMS = {
    "adopted": dict(ADOPTED),
    "WR1.0": {"QB": 1.0, "WR": 1.0, "TE": 0.9, "RB": 0.9},
    "TE1.0": {"QB": 1.0, "WR": 0.9, "TE": 1.0, "RB": 0.9},
    "RB0.7": {"QB": 1.0, "WR": 0.9, "TE": 0.9, "RB": 0.7},
    "RB0.5": {"QB": 1.0, "WR": 0.9, "TE": 0.9, "RB": 0.5},
}
SEASONS = list(range(2016, 2026))

schedule = ingestion.load_schedule(seasons=SEASONS, use_cache=True)
schedule = ingestion.eligible_games(schedule)
schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce")
try:
    pit_weather = __import__("weather").cached_weather_for_games(schedule)
except Exception as exc:
    pit_weather = pd.DataFrame()
    log.warning("weather unavailable: %s", exc)
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

frames = {}
for name, alpha in ARMS.items():
    log.info("building %s %s ...", name, alpha)
    frames[name] = feat_mod.build_game_features(
        schedule, **common, redistribute_phase1=True,
        phase1_caps=None, phase1_alpha=dict(alpha))
    log.info("%s frame: %s", name, frames[name].shape)

fold_list = folds_mod.make_folds(
    folds_mod.canonical_sort(frames["adopted"], "gameday"), date_col="gameday")
log.info("folds: %d", len(fold_list))

ref_oof = None
out = []
for name, df in frames.items():
    log.info("OOF walk: %s ...", name)
    res = ml_mod.walk_forward_oof(df, date_col="gameday", fold_list=fold_list)
    oof = res["oof"]
    oof.to_pickle(f"../run_diagnostics/alpha_tune_oof_{name}.pkl")
    if name == "adopted":
        ref_oof = oof
    y = oof["home_win"].astype(float).to_numpy()
    p = pd.to_numeric(oof["p_ensemble"], errors="coerce").to_numpy(dtype=float)
    grade = (oof["grades_pooled"].astype(bool).to_numpy()
             if "grades_pooled" in oof else np.ones(len(oof), bool))
    for slice_name, m in (("all_grading", grade),):
        met = eval_mod.binary_metrics(p[m], y[m])
        met.update({"arm": name, "slice": slice_name})
        out.append(met)
        log.info("%s %s: %s", name, slice_name, met)
    if name != "adopted":
        yo = y
        po = p
        yb = ref_oof["home_win"].astype(float).to_numpy()
        pb = pd.to_numeric(ref_oof["p_ensemble"], errors="coerce").to_numpy(dtype=float)
        gb = ref_oof["grades_pooled"].astype(bool).to_numpy()
        assert (yb == yo).all() and (gb == grade).all()

        def _ll(yy, pp):
            pp = np.clip(pp, 1e-7, 1 - 1e-7)
            return -(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))

        d = _ll(yo[grade], po[grade]) - _ll(yb[gb], pb[gb])
        out.append({"arm": f"delta({name}-adopted)", "slice": "all_grading",
                    "n": int(grade.sum()), "mean_dll": float(d.mean()),
                    "se_dll": float(d.std(ddof=1) / np.sqrt(len(d)))})
        log.info("paired dll(%s) mean=%.6f se=%.6f", name, float(d.mean()),
                 float(d.std(ddof=1) / np.sqrt(len(d))))

res_df = pd.DataFrame(out)
res_df.to_csv("../run_diagnostics/alpha_tune_moneyline.csv", index=False)
log.info("SAVED run_diagnostics/alpha_tune_moneyline.csv")
print(res_df.to_string(index=False))
