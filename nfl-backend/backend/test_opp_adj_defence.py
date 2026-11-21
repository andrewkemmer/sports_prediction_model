"""Tests for the defensive half of the opponent-adjustment pre-pass.

Before this existed, only offence was adjusted (against the defence faced),
so every defensive number in the pipeline was schedule-blind -- including the
``def_epa_play`` that the offensive adjustment used as its own strength.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import features

#: Offensive quality by team (EPA/play, higher is better).
QUALITY = {"STRONG": 0.15, "WEAK": -0.15, "DEF1": 0.0, "DEF2": 0.0}


def _pairs(week: int):
    """DEF1 and DEF2 alternate between the strong and the weak offence."""
    if week % 2 == 0:
        return [("STRONG", "DEF1"), ("WEAK", "DEF2")]
    return [("STRONG", "DEF2"), ("WEAK", "DEF1")]


def _ladder(weeks: int = 6, defence_quality=None) -> pd.DataFrame:
    defence_quality = defence_quality or {"DEF1": 0.10, "DEF2": 0.10}
    rows = []
    gid = 0
    for week in range(weeks):
        day = pd.Timestamp("2025-09-01") + pd.Timedelta(days=7 * week)
        for home, away in _pairs(week):
            gid += 1
            for team, opp, scored in ((home, away, 1.0), (away, home, 0.0)):
                rows.append({
                    "game_id": gid, "team": team, "opponent": opp,
                    "gameday": day,
                    "for": scored, "against": 1.0 - scored,
                    "epa_play": QUALITY[team],
                    "def_epa_play": defence_quality.get(team, 0.0),
                })
    frame = pd.DataFrame(rows)
    return frame.sort_values(["team", "gameday"], kind="mergesort").reset_index(drop=True)


def test_facing_a_strong_offence_lowers_adjusted_defensive_epa():
    ladder = _ladder()
    features._add_opp_adj_metrics(ladder)
    def1 = ladder[ladder["team"] == "DEF1"].sort_values("gameday")
    vs_strong = def1.iloc[2]      # week 2, DEF1 faces STRONG
    vs_weak = def1.iloc[3]        # week 3, DEF1 faces WEAK
    assert vs_strong["opponent"] == "STRONG"
    assert vs_weak["opponent"] == "WEAK"
    # Identical raw defensive performance, different opponent.
    assert np.isclose(vs_strong["def_epa_play"], vs_weak["def_epa_play"])
    assert vs_strong["def_epa_play_opp_adj"] < vs_weak["def_epa_play_opp_adj"]
    # Both sit on the same side of the raw value's meaning: allowing the same
    # EPA to a juggernaut reads better than allowing it to a weak offence.
    assert vs_strong["def_epa_play_opp_adj"] < vs_strong["def_epa_play"]
    assert vs_weak["def_epa_play_opp_adj"] > vs_weak["def_epa_play"]


def test_offensive_half_still_rewards_facing_a_good_defence():
    ladder = _ladder(defence_quality={"DEF1": -0.15, "DEF2": 0.15})
    features._add_opp_adj_metrics(ladder)
    strong = ladder[ladder["team"] == "STRONG"].sort_values("gameday")
    vs_good = strong.iloc[2]      # faces DEF1, a good defence
    vs_bad = strong.iloc[3]       # faces DEF2, a bad defence
    assert vs_good["opponent"] == "DEF1"
    assert vs_bad["opponent"] == "DEF2"
    assert vs_good["epa_play_opp_adj"] > vs_bad["epa_play_opp_adj"]


def test_no_prior_games_degrades_to_nan_not_a_fabricated_value():
    ladder = _ladder()
    features._add_opp_adj_metrics(ladder)
    def1 = ladder[ladder["team"] == "DEF1"].sort_values("gameday")
    assert np.isnan(def1.iloc[0]["def_epa_play_opp_adj"])
    assert np.isfinite(def1.iloc[1]["def_epa_play_opp_adj"])


def test_missing_offensive_source_degrades_to_nan():
    ladder = _ladder().drop(columns=["epa_play"])
    features._add_opp_adj_metrics(ladder)
    assert ladder["def_epa_play_opp_adj"].isna().all()


def test_both_directions_are_configured():
    import config
    assert config.PBP_OPP_ADJ_METRICS["epa_play"] == "def_epa_play"
    assert config.PBP_OPP_ADJ_METRICS["def_epa_play"] == "epa_play"
    assert "def_epa_play_opp_adj" in config.PBP_OPP_ADJ_TRAILING_SPECS


def test_candidate_manifest_documents_the_defensive_mirror():
    # manifest._build_candidate_manifest() runs at import and indexes a
    # per-metric doc table; an undocumented metric aborts the pipeline at
    # import, before Phase 1 (2026-11-21: def_epa_play_opp_adj).
    import manifest
    for col in ("pbp_def_epa_play_opp_adj_ewm_home",
                "pbp_def_epa_play_opp_adj_ewm_away",
                "pbp_def_epa_play_opp_adj_ewm_diff"):
        assert col in manifest.CANDIDATE_MANIFEST, col
