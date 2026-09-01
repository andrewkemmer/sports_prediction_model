"""audit_pitcher_era_k9.py — READ-ONLY audit of the sp_era/sp_k9 feature math.

Verifies the findings from the ERA/K9 investigation against REAL Statcast
pitch data pulled directly from the Savant endpoint (the same source the
production pipeline ingests via ingestion.pull_statcast), NOT the reduced
pbp_chunks cache.

Checks
------
1. Event-frequency table of PA-ending events in the ingested window:
   force_out, fielders_choice, intent_walk, truncated_pa and every member
   of PA_END_EVENTS (features.PA_END_EVENTS, features.outs_on_pa map).
   Computes the outs-per-100-PA undercount of the current outs map.
2. Score-snapshot semantics: home_score/away_score are pre-pitch snapshots
   (the post_* twins are post-pitch). Rebuilds the features.py runs_on_pa
   (score-delta across consecutive PA-ending events) and compares the
   per-game run total against official finals (ESPN via predictions_history)
   — expecting a systematic shortfall equal to runs scored on each game's
   final PA (the pre-pitch snapshot drops them). Counts final-PA-at-risk
   games.
3. Coverage end-date: last game_date with pitcher stats in the features
   frame (game_level_features.csv) vs the pulled window vs the slate
   (todays_games_*.csv).
4. Duplicate/swap audit of the frame vs predictions_history: duplicate
   game_ids (doubleheader legs), total_runs mismatches, home/away swaps.

Usage
-----
    python audit_pitcher_era_k9.py --start 2026-08-01 --end 2026-08-24 \
        --out-json ../data_delivery/audit_pitcher_era_k9_20260831.json

Writes the JSON summary to --out-json (default data_delivery) and prints
tables to stdout. Makes NO changes to any production file.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DELIVERY = REPO_ROOT / "data_delivery"

# --- mirrors of features.py constants (read-only reproduction) --------
PA_END_EVENTS = [
    "single", "double", "triple", "home_run",
    "strikeout", "strikeout_double_play",
    "walk", "hit_by_pitch",
    "field_out", "field_error", "fielders_choice", "fielders_choice_out",
    "grounded_into_double_play", "double_play", "triple_play",
    "sac_fly", "sac_bunt", "sac_fly_double_play",
    "catcher_interf", "batter_interference",
    "force_out", "sacrifice_bunt_double_play",
]

# events -> outs credited in the CURRENT production outs_on_pa map
OUTS_MAP_CURRENT = {
    "field_out": 1,
    "strikeout": 1,
    "strikeout_double_play": 2,
    "grounded_into_double_play": 2,
    "double_play": 2,
    "triple_play": 3,
    "sac_fly": 1,
    "sac_bunt": 1,
    "fielders_choice_out": 1,
    "sac_fly_double_play": 2,
    "sacrifice_bunt_double_play": 2,
}
OUTS_MAP_CORRECTED = dict(OUTS_MAP_CURRENT)
OUTS_MAP_CORRECTED["force_out"] = 1          # 1 out, currently 0
OUTS_MAP_CORRECTED["fielders_choice"] = 1    # out on lead runner, currently 0
OUTS_MAP_CORRECTED["batter_interference"] = 1  # batter out (rare; PA_END_EVENTS member)
PA_END_EVENTS_CORRECTED = PA_END_EVENTS + ["intent_walk", "truncated_pa"]

KEEP_COLS = [
    "game_pk", "game_date", "inning", "inning_topbot", "at_bat_number",
    "pitch_number", "pitcher", "player_name", "events",
    "home_score", "away_score", "post_home_score", "post_away_score",
    "home_team", "away_team", "game_type",
]

SAVANT_BASE = (
    "https://baseballsavant.mlb.com/statcast_search/csv?all=true"
    "&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones="
    "&hfGT=R%7CPO%7CS%7C&hfC=&hfSea=2026%7C&hfSit="
    "&player_type=pitcher&hfOuts=&opponent_concept=&hfTeam=&home_away="
    "&hfRO=&hfFlag=&hfPull=&hfInfield=&hfInn="
    "&min_pitches=0&min_results=0&group_by=name"
    "&sort_col=pitches&player_event_sort=h_launch_speed&sort_order=desc"
    "&min_pas=0&type=details"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch_day(day: date) -> "pd.DataFrame | None":
    import pandas as pd

    d = day.strftime("%Y-%m-%d")
    url = f"{SAVANT_BASE}&game_date_gt={d}&game_date_lt={d}"
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=180, headers=HEADERS)
            if r.status_code != 200:
                time.sleep(5 * (attempt + 1))
                continue
            txt = r.content.decode("utf-8-sig", errors="replace")
            if "game_pk" not in txt.splitlines()[0] if txt else "":
                time.sleep(5 * (attempt + 1))
                continue
            df = pd.read_csv(io.StringIO(txt))
            if df is None or df.empty:
                time.sleep(5 * (attempt + 1))
                continue
            return df
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None


def pull_window(start: date, end: date, pause: float = 2.5) -> "pd.DataFrame":
    """Chunked daily pulls (mirrors ingestion.pull_statcast's rate habit)."""
    import pandas as pd

    frames = []
    day = start
    while day <= end:
        df = fetch_day(day)
        if df is not None and len(df):
            frames.append(df[KEEP_COLS])
            print(f"  {day}: {len(df):,} pitches", flush=True)
        else:
            print(f"  {day}: NO DATA", flush=True)
        time.sleep(pause)
        day += timedelta(days=1)
    if not frames:
        raise SystemExit("No Statcast data pulled — aborting audit.")
    out = pd.concat(frames, ignore_index=True)
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.date
    for c in ("home_score", "away_score", "post_home_score", "post_away_score"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["pitch_number"] = pd.to_numeric(out["pitch_number"], errors="coerce")
    return out


def pa_endings(pitches: "pd.DataFrame") -> "pd.DataFrame":
    """Reproduce features.py pa_boundary: last pitch of each PA, in order."""
    import pandas as pd

    sub = pitches[pitches["events"].notna()].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    sub = sub.drop_duplicates(
        ["game_pk", "at_bat_number"], keep="last")
    sub = sub.sort_values(["game_pk", "at_bat_number"])
    sub["tot_score"] = sub["home_score"].fillna(0) + sub["away_score"].fillna(0)
    sub["post_tot"] = (
        sub["post_home_score"].fillna(0) + sub["post_away_score"].fillna(0))
    prev = sub.groupby("game_pk")["tot_score"].shift(1)
    sub["runs_on_pa"] = sub["tot_score"] - prev
    sub["runs_on_pa"] = sub["runs_on_pa"].where(
        sub["runs_on_pa"] > 0, 0.0)  # GREATEST(delta, 0)
    # pre-pitch vs post-pitch snapshot check on the final pitch of each PA
    sub["post_minus_pre_final_pitch"] = sub["post_tot"] - sub["tot_score"]
    return sub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-08-01")
    ap.add_argument("--end", default="2026-08-24")
    ap.add_argument("--cache", default="",
                    help="Parquet cache path for the raw pull (reuse on reruns)")
    ap.add_argument("--out-json", default=str(
        DATA_DELIVERY / "audit_pitcher_era_k9_20260831.json"))
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_path = Path(args.out_json)
    cache_path = Path(args.cache) if args.cache else (
        Path(args.out_json).with_suffix(".pitches.parquet"))

    import pandas as pd
    if cache_path.exists():
        print(f"Loading cached pull → {cache_path}", flush=True)
        pitches = pd.read_parquet(cache_path)
        pitches["game_date"] = pd.to_datetime(pitches["game_date"]).dt.date
    else:
        print(f"Pulling real Statcast {start} → {end} (daily, Savant)...",
              flush=True)
        pitches = pull_window(start, end)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pitches.to_parquet(cache_path, index=False)
        print(f"Cached raw pull → {cache_path}", flush=True)

    print(f"{len(pitches):,} pitches across "
          f"{pitches['game_date'].nunique()} days\n", flush=True)

    report: dict = {"window": [str(start), str(end)],
                    "pitches_pulled": int(len(pitches))}

    # ── 1. Event-frequency + outs-map undercount ──────────────────────
    ev = pitches[pitches["events"].notna()]["events"].value_counts()
    tot_ev = int(ev.sum())
    rows = []
    for k in sorted(ev.index, key=lambda x: -ev[x]):
        rows.append({
            "event": k,
            "count": int(ev[k]),
            "pct_of_pa_endings": round(100 * ev[k] / tot_ev, 4),
            "in_pa_end_events": k in PA_END_EVENTS,
            "outs_current": OUTS_MAP_CURRENT.get(k, 0 if k in PA_END_EVENTS else None),
            "outs_corrected": OUTS_MAP_CORRECTED.get(k, 0 if k in PA_END_EVENTS else None),
        })
    def _count_weighted(mapping):
        return sum(ev[k] * mapping.get(k, 0) for k in ev.index)

    outs_credited_current = _count_weighted(OUTS_MAP_CURRENT)
    outs_lost_by_fix = _count_weighted(OUTS_MAP_CORRECTED) - outs_credited_current
    pdf = pd.DataFrame(rows)
    report["event_frequencies"] = rows
    report["outs_credited_per_100_pa_endings_current"] = round(
        100 * outs_credited_current / tot_ev, 4)
    report["outs_lost_per_100_pa_endings_current_vs_corrected"] = round(
        100 * outs_lost_by_fix / tot_ev, 4)
    report["relative_ip_deficit_pct"] = round(
        100 * outs_lost_by_fix / outs_credited_current, 3)
    report["approx_era_k9_inflation_pct"] = round(
        100 * outs_lost_by_fix / outs_credited_current, 3)

    print("=== 1. PA-ending event frequencies (real Savant window) ===")
    print(pdf.to_string(index=False))
    print(f"\nouts credited (current map): {outs_credited_current:,}  "
          f"({100*outs_credited_current/tot_ev:.2f} per 100 PA) | "
          f"outs lost by the outs-map fix: +{outs_lost_by_fix} "
          f"({100*outs_lost_by_fix/tot_ev:.2f} per 100 PA → "
          f"{100*outs_lost_by_fix/max(outs_credited_current,1):.2f}% relative IP "
          f"deficit → ~same % ERA/K9 inflation)\n")

    # ── 2. Score-snapshot semantics + run-delta vs official finals ────
    pa = pa_endings(pitches)
    pa = pa[pa["game_type"] == "R"]
    fin_pa = pa[pa["post_minus_pre_final_pitch"] != 0]
    snap_pct = float((pa["post_minus_pre_final_pitch"] != 0).mean())
    scoring_pa_pct = float((pa["runs_on_pa"] != 0).mean())
    # A pre-pitch snapshot shows post−pre == +runs EXACTLY on scoring PAs;
    # a post-pitch snapshot would show post == pre everywhere.
    consistent_pre = abs(snap_pct - scoring_pa_pct) < 0.02
    conclusion = ("home_score/away_score are PRE-pitch snapshots "
                  "(post_* twins carry the post-PA score; scoring PAs show "
                  "post − pre == runs, non-scoring PAs show 0)" if consistent_pre
                  else "snapshot semantics inconclusive/other")
    report["score_snapshot"] = {
        "pct_pa_endings_with_post_minus_pre_nonzero": round(100 * snap_pct, 3),
        "pct_pa_endings_with_runs_on_pa_nonzero": round(100 * scoring_pa_pct, 3),
        "consistent_with_pre_pitch_snapshot": bool(consistent_pre),
        "conclusion": conclusion,
    }
    print("=== 2. Score-snapshot semantics ===")
    print(f"final-pitch PA boundaries where post_score != pre_score: "
          f"{len(fin_pa)} / {len(pa)} ({100*snap_pct:.1f}%) vs scoring-PA "
          f"rate {100*scoring_pa_pct:.1f}% "
          f"→ {report['score_snapshot']['conclusion']}\n")

    # per-game delta totals (features.py semantics, pre-pitch)
    g = pa.groupby(["game_pk", "game_date", "home_team", "away_team"]).agg(
        delta_runs=("runs_on_pa", "sum"),
        final_pa_runs=("post_minus_pre_final_pitch",
                       lambda s: float(s.iloc[-1]) if len(s) else 0.0),
    ).reset_index()

    # official finals from the shipped prediction history (ESPN box scores)
    hist_path = DATA_DELIVERY / "predictions_history_20260831.csv"
    official = pd.DataFrame()
    if hist_path.exists():
        h = pd.read_csv(hist_path, usecols=["game_id", "home_score", "away_score"])
        h["final"] = h["home_score"].fillna(0) + h["away_score"].fillna(0)
        g["game_id"] = (g["game_date"].astype(str).str.replace("-", "")
                        + "_" + g["away_team"] + "@" + g["home_team"])
        official = g.merge(h, on="game_id", how="left")
    if not official.empty and official["final"].notna().any():
        o = official.dropna(subset=["final"]).copy()
        o["delta_home_plus_away"] = o["delta_runs"]
        o["final_mismatch"] = (o["delta_runs"] - o["final"]).round(6)
        o["mismatch_after_adding_final_pa"] = (
            (o["delta_runs"] + o["final_pa_runs"]) - o["final"]).round(6)
        exact1 = int((o["final_mismatch"].abs() < 0.5).sum())
        exact2 = int((o["mismatch_after_adding_final_pa"].abs() < 0.5).sum())
        final_pa_fixes = int(
            ((o["final_mismatch"].abs() >= 0.5)
             & (o["mismatch_after_adding_final_pa"].abs() < 0.5)).sum())
        report["run_delta_vs_official"] = {
            "games_compared": len(o),
            "delta_sum_exact_match": exact1,
            "exact_after_adding_final_pa_runs": exact2,
            "games_fixed_by_final_pa_runs": final_pa_fixes,
            "final_pa_runs_dropped_games": int((o["final_pa_runs"] != 0).sum()),
        }
        print("=== 2b. runs_on_pa (score-delta) vs official finals ===")
        print(f"games compared: {len(o)} | delta-sum == final: {exact1} "
              f"({100*exact1/len(o):.1f}%) | exact AFTER adding the dropped "
              f"final-PA runs: {exact2} ({100*exact2/len(o):.1f}%) | "
              f"games whose mismatch the final-PA fix resolves: "
              f"{final_pa_fixes}")
        print(f"games with nonzero final-PA scoring (runs never counted by "
              f"the pre-pitch delta): {report['run_delta_vs_official']['final_pa_runs_dropped_games']}\n")
    else:
        report["run_delta_vs_official"] = {"note": "predictions_history not available"}

    # ── 3. Coverage end-date ──────────────────────────────────────────
    frame_path = DATA_DELIVERY / "game_level_features.csv"
    cov: dict = {}
    if frame_path.exists():
        fr = pd.read_csv(frame_path, usecols=["game_date"])
        fr["game_date"] = pd.to_datetime(fr["game_date"])
        cov["features_frame_last_game_date"] = str(fr["game_date"].max().date())
    slate_paths = sorted(DATA_DELIVERY.glob("todays_games_2026*.csv"))
    if slate_paths:
        cov["latest_slate"] = slate_paths[-1].name
    cov["audit_pull_window"] = [str(start), str(end)]
    cov["last_pulled_game_date"] = str(max(pitches["game_date"]))
    report["coverage"] = cov
    print("=== 3. Coverage ===")
    for k, v in cov.items():
        print(f"  {k}: {v}")

    # ── 4. Duplicate/swap audit (frame vs prediction history) ─────────
    if frame_path.exists() and hist_path.exists():
        g2 = pd.read_csv(frame_path, usecols=["game_id", "home_team", "away_team",
                                              "home_score", "away_score",
                                              "total_runs"])
        dup_ids = int(g2["game_id"].duplicated().sum())
        h2 = pd.read_csv(hist_path, usecols=["game_id", "home_team", "away_team",
                                             "home_score", "away_score"])
        merg = (g2.drop_duplicates("game_id")
                  .merge(h2.drop_duplicates("game_id"), on="game_id",
                         how="inner", suffixes=("_f", "_h")))
        merg = merg.dropna(subset=["home_score_f", "home_score_h"])
        tot_mis = int((merg["home_score_f"] + merg["away_score_f"]
                       != merg["home_score_h"] + merg["away_score_h"]).sum())
        swaps = int(((merg["home_score_f"] == merg["away_score_h"])
                     & (merg["home_score_f"] != merg["home_score_h"])).sum())
        report["dup_swap_audit"] = {
            "frame_duplicate_game_ids": dup_ids,
            "prediction_history_duplicate_game_ids": int(h2["game_id"].duplicated().sum()),
            "games_total_mismatch_vs_espn": tot_mis,
            "games_home_away_swapped": swaps,
        }
        print("\n=== 4. Dup/swap audit (frame vs predictions_history) ===")
        for k, v in report["dup_swap_audit"].items():
            print(f"  {k}: {v}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nAudit JSON written → {out_path}")


if __name__ == "__main__":
    main()