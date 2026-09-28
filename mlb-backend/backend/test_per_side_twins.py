"""Per-side twin contract: every served diff exposes its raw home/away halves.

The 2026-09-27 structural request: add the raw home and away metrics for
each served diff family to production, and give the three interaction
features their *_diff name plus per-side twins. Invariants pinned here:

  * every twin pair satisfies  home − away == diff  (the twins are the
    diff's own raw halves, served from the SAME strictly-prior source —
    no second derivation that can drift),
  * level twins share the diff's NULL pattern (same inputs in/out),
  * interaction twins are the within-side product of the interaction's
    own factors,
  * exp2 twins are the per-side scratch the diff was already computed
    from (previously dropped at the end of add_exp2_features),
  * twins join the serving universe but route TREE-ONLY (logistic keeps
    its diffs-only view; the run engine serves the FULL active moneyline
    list verbatim — the historical λ-view drop rule was removed
    2026-09-27),
  * the adopted RFE state is re-issued so apply_adopted_subset() binds
    the new width instead of silently falling back to the universe.

Run with: python mlb-backend/backend/test_per_side_twins.py
"""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

import features
import training
import run_engine as re_engine

# ── the 18 families from the request ────────────────────────────────────────
# (diff_or_level_base, home_twin, away_twin) for the 15 diff families,
# plus the 3 renamed interactions.
LEVEL_FAMILIES = {
    "rest_days_diff": ("rest_days_home", "rest_days_away"),
    "sp_era_5g_diff": ("sp_era_5g_home", "sp_era_5g_away"),
    "sp_fbvelo_diff": ("sp_fbvelo_3g_home", "sp_fbvelo_3g_away"),
    "lineup_woba_std_diff": ("lineup_woba_std_home", "lineup_woba_std_away"),
    "bullpen_pitches_diff": ("bullpen_pitches_3d_home", "bullpen_pitches_3d_away"),
    "team_hardhit_diff": ("team_hardhit_15g_home", "team_hardhit_15g_away"),
    "travel_fatigue_diff": ("time_zones_crossed_last_3d_home",
                            "time_zones_crossed_last_3d_away"),
}
RENAME_FAMILIES = {
    "pitcher_regression_indicator_diff": ("pitcher_regression_indicator_home",
                                          "pitcher_regression_indicator_away"),
    "lineup_depth_multiplier_diff": ("lineup_depth_multiplier_home",
                                     "lineup_depth_multiplier_away"),
    "ace_efficiency_factor_diff": ("ace_efficiency_factor_home",
                                   "ace_efficiency_factor_away"),
}
EXP2_FAMILIES = {
    f"exp2_{kind}_{cat}_diff": (f"exp2_{kind}_{cat}_home", f"exp2_{kind}_{cat}_away")
    for kind in ("centered_k", "cat_platoon_k_fastball")
    for cat in ("",)
}
EXP2_FAMILIES = {
    "exp2_centered_k_diff": ("exp2_centered_k_home", "exp2_centered_k_away"),
    "exp2_cat_platoon_k_fastball_diff": ("exp2_cat_platoon_k_fastball_home",
                                         "exp2_cat_platoon_k_fastball_away"),
    **{f"exp2_cat_k_{c}_diff": (f"exp2_cat_k_{c}_home", f"exp2_cat_k_{c}_away")
       for c in ("fastball", "breaking", "offspeed")},
    **{f"exp2_cat_xwoba_{c}_diff": (f"exp2_cat_xwoba_{c}_home", f"exp2_cat_xwoba_{c}_away")
       for c in ("fastball", "breaking", "offspeed")},
}
ALL_TWIN_DIFFS = {**LEVEL_FAMILIES, **RENAME_FAMILIES, **EXP2_FAMILIES}
assert len(ALL_TWIN_DIFFS) == 18, sorted(ALL_TWIN_DIFFS)

# Twin names that are raw LEVELS (pass through from the frame inputs, not
# computed by add_diff_features/add_exp2_features themselves).
PASSTHROUGH_TWINS = {
    "time_zones_crossed_last_3d_home", "time_zones_crossed_last_3d_away",
}


def _diff_frame(n: int = 4) -> pd.DataFrame:
    """Identity frame carrying every raw input add_diff_features reads for
    the 15 families, with jittered values so h−a and product invariants
    actually bind. Row 2 has a hole (rest_days_away) for NULL propagation."""
    rng = np.random.default_rng(7)

    def sides(base, lo, hi):
        return {f"{base}_home": np.round(rng.uniform(lo, hi, n), 4),
                f"{base}_away": np.round(rng.uniform(lo, hi, n), 4)}

    d: dict[str, object] = {}
    d.update(sides("rest_days", 1, 5))
    d.update(sides("sp_era_5g", 2.5, 6.0))
    d.update(sides("sp_fbvelo_3g", 90.0, 99.0))
    d.update(sides("lineup_woba_std", 0.03, 0.08))
    d.update(sides("bullpen_pitches_3d", 20.0, 80.0))
    d.update(sides("team_hardhit_15g", 0.35, 0.50))
    d.update(sides("time_zones_crossed_last_3d", 0, 2))
    d.update(sides("sp_k9_5g", 6.0, 12.0))
    d.update(sides("sp_whiff_3g", 0.20, 0.32))
    d.update(sides("lineup_woba_mean", 0.28, 0.36))
    d.update(sides("lineup_woba_top3", 0.30, 0.42))
    d.update({
        "home_elo": [1501.0, 1520.0, 1495.0, 1533.0],
        "away_elo": [1499.0, 1510.0, 1502.0, 1488.0],
        "home_win_pct": [0.55, 0.48, 0.60, 0.52],
        "away_win_pct": [0.45, 0.52, 0.40, 0.49],
        "home_team": ["NYY"] * n,
        "away_team": ["BOS"] * n,
    })
    df = pd.DataFrame(d)
    df.loc[2, "rest_days_away"] = np.nan
    return df


def _exp2_frame(n: int = 4) -> pd.DataFrame:
    """Frame carrying every add_exp2_features source column, jittered."""
    rng = np.random.default_rng(11)
    d: dict[str, object] = {
        "league_k_pct": np.round(rng.uniform(0.20, 0.24, n), 4),
        "league_k_pct_fb_vs_l": np.round(rng.uniform(0.22, 0.26, n), 4),
        "league_k_pct_fb_vs_r": np.round(rng.uniform(0.18, 0.22, n), 4),
    }
    for c in ("fastball", "breaking", "offspeed"):
        d[f"league_k_pct_cat_{c}"] = np.round(rng.uniform(0.16, 0.28, n), 4)
        d[f"league_xwoba_cat_{c}"] = np.round(rng.uniform(0.30, 0.36, n), 4)
    for s in ("home", "away"):
        d[f"sp_k9_{s}"] = np.round(rng.uniform(7.0, 12.0, n), 4)
        d[f"team_k_rate_30g_{s}"] = np.round(rng.uniform(0.19, 0.26, n), 4)
        d[f"opp_lefty_share_{s}"] = np.round(rng.uniform(0.3, 0.6, n), 4)
        d[f"sp_usage_cat_fastball_{s}"] = np.round(rng.uniform(0.45, 0.65, n), 4)
        for c in ("fastball", "breaking", "offspeed"):
            d[f"sp_usage_cat_{c}_{s}"] = np.round(rng.uniform(0.12, 0.55, n), 4)
            d[f"sp_k_pct_cat_{c}_{s}"] = np.round(rng.uniform(0.14, 0.34, n), 4)
            d[f"sp_xwoba_cat_{c}_{s}"] = np.round(rng.uniform(0.28, 0.40, n), 4)
            d[f"team_k_pct_cat_{c}_{s}"] = np.round(rng.uniform(0.16, 0.30, n), 4)
            d[f"team_xwoba_cat_{c}_{s}"] = np.round(rng.uniform(0.28, 0.38, n), 4)
        for h in ("l", "r"):
            d[f"sp_k_pct_fb_vs_{h}_{s}"] = np.round(rng.uniform(0.16, 0.30, n), 4)
            d[f"team_k_pct_fb_vs_{h}_{s}"] = np.round(rng.uniform(0.18, 0.28, n), 4)
    # one hole: away offspeed usage missing on row 1 → offspeed diff AND
    # both offspeed twins must go NULL on that row, nowhere else.
    df = pd.DataFrame(d)
    df.loc[1, "sp_usage_cat_offspeed_away"] = np.nan
    return df


def _same_vec(a: pd.Series, b: pd.Series) -> bool:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                       equal_nan=True)


# ── A. add_diff_features: renames, twins, invariants ────────────────────────

def test_diff_pass_renames_the_three_interactions():
    df = _diff_frame()
    out = features.add_diff_features(df)
    for new in RENAME_FAMILIES:
        assert new in out.columns, f"missing renamed diff {new}"
        assert out[new].notna().all(), f"renamed diff {new} has unexpected NULLs"
    for old in ("pitcher_regression_indicator", "lineup_depth_multiplier",
                "ace_efficiency_factor"):
        assert old not in out.columns, f"stale name {old} still created"


def test_diff_pass_creates_the_level_and_interaction_twins():
    df = _diff_frame()
    out = features.add_diff_features(df)
    for diff, (h, a) in {**LEVEL_FAMILIES, **RENAME_FAMILIES}.items():
        if diff in LEVEL_FAMILIES and h in PASSTHROUGH_TWINS:
            continue  # travel twins are frame inputs, not created here
        assert h in out.columns, f"missing twin {h} for {diff}"
        assert a in out.columns, f"missing twin {a} for {diff}"


def test_every_level_and_exp2_diff_satisfies_home_minus_away():
    """Level and exp2 twins ARE their diff's halves: home − away == diff.
    The three interaction diffs are products of diffs — their twins are the
    within-side representation (pinned separately below), so the identity
    deliberately does NOT hold for them."""
    df = _diff_frame()
    out = features.add_diff_features(df)
    for diff, (h, a) in LEVEL_FAMILIES.items():
        assert diff in out.columns, f"missing diff {diff}"
        assert _same_vec(out[h] - out[a], out[diff]), (
            f"{diff} != {h} − {a}")


def test_interaction_twins_are_the_within_side_product():
    df = _diff_frame()
    out = features.add_diff_features(df)
    expected = {
        "pitcher_regression_indicator_home": ("sp_fbvelo_3g_home", "sp_era_5g_home"),
        "pitcher_regression_indicator_away": ("sp_fbvelo_3g_away", "sp_era_5g_away"),
        "lineup_depth_multiplier_home": ("lineup_woba_mean_home", "lineup_woba_top3_home"),
        "lineup_depth_multiplier_away": ("lineup_woba_mean_away", "lineup_woba_top3_away"),
        "ace_efficiency_factor_home": ("sp_k9_5g_home", "sp_whiff_3g_home"),
        "ace_efficiency_factor_away": ("sp_k9_5g_away", "sp_whiff_3g_away"),
    }
    for twin, (f1, f2) in expected.items():
        assert _same_vec(out[f1] * out[f2], out[twin]), (
            f"{twin} != {f1} × {f2}")


def test_level_twins_carry_their_own_sides_null_mask():
    """A twin is NULL exactly where ITS side's observation is missing; the
    diff is NULL where EITHER side is (row 2 removes rest_days_away only).
    On real decided-frame data the sides ship in pairs, so the masks coincide;
    the fixture makes the per-side semantics explicit."""
    df = _diff_frame()  # row 2 has rest_days_away = NaN
    out = features.add_diff_features(df)
    assert out["rest_days_diff"].isna().tolist() == [False, False, True, False]
    assert out["rest_days_away"].isna().tolist() == [False, False, True, False]
    assert out["rest_days_home"].isna().tolist() == [False] * 4


# ── B. add_exp2_features: 24 columns, served scratch, invariants ────────────

def test_exp2_pass_creates_twenty_four_columns():
    df = _exp2_frame()
    out = features.add_exp2_features(df)
    created = set(out.columns) - set(df.columns)
    assert len(created) == 24, sorted(created)


def test_exp2_diffs_satisfy_home_minus_away():
    df = _exp2_frame()
    out = features.add_exp2_features(df)
    for diff, (h, a) in EXP2_FAMILIES.items():
        assert _same_vec(out[h] - out[a], out[diff]), f"{diff} != {h} − {a}"


def test_exp2_twins_carry_their_own_sides_null_mask():
    """Row 1 lacks the AWAY offspeed usage: the offspeed diffs go NULL there,
    the away twins follow their own side, the home twins stay populated."""
    df = _exp2_frame()
    out = features.add_exp2_features(df)
    mask = out["exp2_cat_k_offspeed_diff"].isna().tolist()
    assert mask == [False, True, False, False], mask
    for twin in ("exp2_cat_k_offspeed_away", "exp2_cat_xwoba_offspeed_away"):
        assert out[twin].isna().tolist() == mask, twin
    for twin in ("exp2_cat_k_offspeed_home", "exp2_cat_xwoba_offspeed_home"):
        assert out[twin].isna().tolist() == [False] * 4, twin


def test_exp2_no_scratch_columns_remain():
    df = _exp2_frame()
    out = features.add_exp2_features(df)
    assert not [c for c in out.columns if c.startswith("_exp2_side_")]


# ── C. universe, routing, adopted state ─────────────────────────────────────

def test_universe_carries_all_thirty_six_twins_and_renames():
    universe = training.MONEYLINE_FEATURE_COLS
    for diff, (h, a) in ALL_TWIN_DIFFS.items():
        assert diff in universe, f"diff {diff} not in serving universe"
        assert h in universe and a in universe, f"twins {h}/{a} not in universe"
    for old in ("pitcher_regression_indicator", "lineup_depth_multiplier",
                "ace_efficiency_factor"):
        assert old not in universe, f"stale name {old} still in universe"


def test_logistic_stays_diffs_only_twins_route_tree_only():
    logistic = set(training._logistic_feature_cols())
    for diff, (h, a) in ALL_TWIN_DIFFS.items():
        assert h not in logistic and a not in logistic, (
            f"twin {h}/{a} leaked into the logistic view")
        assert diff in logistic or diff == "travel_fatigue_diff", (
            f"diff {diff} missing from the logistic view")


def test_run_engine_lambda_view_serves_full_active_list():
    # Production contract: the run engine serves the ACTIVE moneyline list
    # verbatim — nothing is dropped by rule. (The historical derivation that
    # filtered *_diff composites out of the λ view was retired to monitor-only
    # by the 2026-08-30 restore and removed outright on 2026-09-27.)
    feats, dropped = re_engine._resolve_run_view()
    active = training.active_moneyline_feature_cols()
    assert feats == list(active), "run view must be the active list verbatim"
    assert dropped == [], "the no-drop contract must hold (dropped always empty)"


def test_run_engine_side_view_carries_every_served_feature():
    # build_side_frame's production branch must carry EVERY served moneyline
    # feature into the side models: each side's view holds its own side
    # columns plus the shared environment, and the UNION of the home+away
    # views must be exactly the active list — nothing dropped by rule. (The
    # historical derivation that filtered *_diff composites out of the λ view
    # was retired to monitor-only by the 2026-08-30 restore and removed
    # outright on 2026-09-27; the P1 projection column may only append.)
    games = pd.DataFrame({
        "game_pk": [1],
        "game_date": ["2026-09-20"],
        "home_team": ["NYY"], "away_team": ["BOS"],
        "home_score": [5], "away_score": [2], "home_win": [1.0],
        "elo_diff": [0.1],
    })
    _, home_cols = re_engine.build_side_frame(games, "home")
    _, away_cols = re_engine.build_side_frame(games, "away")
    active = set(training.active_moneyline_feature_cols())
    union = set(home_cols) | set(away_cols)
    missing = active - union
    assert not missing, (
        f"run side views dropped {len(missing)} served features: "
        f"{sorted(missing)[:8]}")
    # Side-agnostic matchup gaps are shared environment — present in BOTH
    # side views; per-side levels appear in their own side's view.
    for shared in ("win_pct_diff", "elo_diff", "bullpen_whip_3g_diff",
                   "bullpen_meltdown_risk",
                   "lineup_handedness_matchup_advantage"):
        assert shared in home_cols and shared in away_cols, shared
    for twin in ("pitcher_regression_indicator_home", "exp2_centered_k_home"):
        assert twin in home_cols, twin
    for twin in ("pitcher_regression_indicator_away", "exp2_centered_k_away"):
        assert twin in away_cols, twin


def test_adopted_state_binds_the_new_width():
    from feature_selection import apply_adopted_subset
    from config import DATA_DELIVERY_DIR
    import json
    state = json.loads(
        (DATA_DELIVERY_DIR / "mlb_feature_selection_state.json").read_text())
    cols = state["cols"]
    unknown = [c for c in cols if c not in training.KNOWN_FEATURE_COLS]
    assert not unknown, f"adopted state references non-pool columns: {unknown}"
    for diff, (h, a) in ALL_TWIN_DIFFS.items():
        assert diff in cols, f"adopted state missing {diff}"
        assert h in cols and a in cols, f"adopted state missing twins {h}/{a}"
    report = apply_adopted_subset()
    assert report.get("applied") is True, report
    assert len(training.active_moneyline_feature_cols()) == state["n_cols"]


# ── D. metadata ─────────────────────────────────────────────────────────────

def test_metadata_covers_the_full_serving_width_without_warnings():
    import feature_metadata
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=logging.WARNING)
    lg = logging.getLogger("feature_metadata")
    lg.addHandler(handler)
    try:
        meta, warnings = feature_metadata.build_features_metadata()
    finally:
        lg.removeHandler(handler)
    serving = training.active_moneyline_feature_cols()
    assert set(meta) == set(serving)
    assert not warnings, warnings
    loud = [r.getMessage() for r in records if r.levelno >= logging.WARNING]
    assert not loud, loud
    for name in ("pitcher_regression_indicator_diff",
                 "lineup_depth_multiplier_diff", "ace_efficiency_factor_diff",
                 "rest_days_home", "exp2_cat_k_offspeed_away"):
        entry = meta.get(name)
        assert entry is not None, f"no metadata row for {name}"
        assert entry.get("formula") not in (None, "", "—"), (
            f"placeholder metadata for {name}")


# ── runner ──────────────────────────────────────────────────────────────────

def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
