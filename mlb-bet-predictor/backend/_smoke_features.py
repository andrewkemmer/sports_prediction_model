"""Smoke: run full DuckDB feature engineering + verify the 35-feature layout."""
import sys
sys.path.insert(0, "backend")

import numpy as np
import pandas as pd

N_GAMES = 40
TEAMS = ["NYY", "BOS", "SEA", "LAD"]
rng = np.random.RandomState(7)

rows = []
for g in range(N_GAMES):
    home = TEAMS[g % len(TEAMS)]
    away = TEAMS[(g + 1) % len(TEAMS)]
    if home == away:
        continue
    day = pd.Timestamp("2026-04-01") + pd.Timedelta(days=g // 2)
    pk = 900000 + g
    hs, as_ = int(rng.randint(0, 8)), int(rng.randint(0, 8))
    if hs == as_:
        as_ += 1
    # One PA per game per half-inning is enough to exercise every CTE
    for half, batter_team in (("Top", away), ("Bot", home)):
        rows.append(dict(
            game_pk=pk, game_date=day.strftime("%Y-%m-%d"), game_type="R",
            home_team=home, away_team=away,
            inning=1, inning_topbot=half, outs_when_up=0, balls=0, strikes=0,
            on_1b=np.nan, on_2b=np.nan, on_3b=np.nan,
            at_bat_number=len(rows), pitch_number=1,
            pitcher=float(100 + hash(half) % 50), batter=float(200 + g),
            p_throws="R", stand="L",
            pitch_type="FF", release_speed=94.0,
            description="hit_into_play", events="field_out",
            barrel=np.nan, hard_contact=np.nan,
            launch_speed=99.0, launch_angle=28.0,
            estimated_woba_using_speedangle=0.350,
            estimated_ba_using_speedangle=0.250,
            zone=5, home_score=hs, away_score=as_, spin_rate=2300.0,
            woba_value=0.0, babip_value=0.0, iso_value=0.0,
            delta_home_win_exp=0.0, delta_run_exp=-0.1,
            player_name=f"P{g}", hit_distance_sc=300.0,
            release_pos_x=0.0, release_pos_z=5.0,
            release_spin_rate=2300.0, release_extension=6.0,
            pfx_x=0.0, pfx_z=0.0,
        ))

df = pd.DataFrame(rows)
df.to_parquet("/tmp/smoke_pitches.parquet", index=False)

from features import build_features
game_df, pbp_df = build_features("/tmp/smoke_pitches.parquet", "/tmp/smoke_out")
print(f"games={len(game_df)} pitches={len(pbp_df)}")

# Give records so win-pct smoothing path activates
for side in ("home", "away"):
    game_df[f"{side}_wins"] = rng.randint(0, 30, len(game_df)).astype(float)
    game_df[f"{side}_losses"] = rng.randint(0, 30, len(game_df)).astype(float)
# Lineup hand splits + closer state present (as SQL now emits them)
out = add_diff = None
from features import add_diff_features
out = add_diff_features(game_df)

from training import FEATURE_COLS
missing = [c for c in FEATURE_COLS if c not in out.columns]
assert not missing, f"MISSING: {missing}"
print(f"all {len(FEATURE_COLS)} FEATURE_COLS present ✓")

# Spec-formula spot checks (float32 storage → compare with tolerance).
# Each formula is checked on a row where ITS inputs are observed — early
# rows have no trailing history (point-in-time NaNs), so their products
# are legitimately NaN.
def _first_observed(cols):
    sub = out.dropna(subset=cols)
    if not len(sub):
        return None  # fixture too small to populate this window
    return sub.iloc[0]

r_melt = _first_observed(["bullpen_pitches_diff", "bullpen_whip_diff"])
r_reg = _first_observed(["sp_fbvelo_diff", "sp_era_5g_diff"])
r_ace = _first_observed(["sp_k9_5g_diff", "sp_whiff_diff"])
def close(a, b):
    return np.isclose(
        pd.to_numeric(pd.Series(a), errors="coerce").astype(float),
        pd.to_numeric(pd.Series(b), errors="coerce").astype(float),
        rtol=1e-4, atol=1e-6, equal_nan=True,
    ).all()
assert close(r_reg["pitcher_regression_indicator"], float(r_reg["sp_fbvelo_diff"]) * float(r_reg["sp_era_5g_diff"]))
if r_ace is not None:
    assert close(r_ace["ace_efficiency_factor"], float(r_ace["sp_k9_5g_diff"]) * float(r_ace["sp_whiff_diff"]))
else:
    print("ace_efficiency_factor: no row with observed inputs — skipped")
if r_melt is not None:
    assert close(r_melt["bullpen_meltdown_risk"], float(r_melt["bullpen_pitches_diff"]) * float(r_melt["bullpen_whip_diff"]))
else:
    print("bullpen_meltdown_risk: no row with observed inputs — skipped")
print("spec-formula spot checks ✓")
assert close(out["lineup_depth_multiplier"], out["lineup_woba_mean_diff"] * out["lineup_woba_top3_diff"]).all()
# win_pct smoothing: zero games → exactly 0 diff
z = out.copy()
z["home_wins"] = z["home_losses"] = z["away_wins"] = z["away_losses"] = 0.0
z = add_diff_features(z)
assert (z["win_pct_diff"] == 0.0).all(), "smoothing must give exactly .500 − .500"
print("formula + smoothing checks ✓")
print("new raw columns sample:",
      [c for c in ("lineup_ops_vs_starter_hand_home", "time_zones_crossed_last_3d_home",
                   "closer_available_home", "bullpen_whip_3g_home", "sp_era_5g_home")
       if c in out.columns])
print("SMOKE OK")
