"""Tests for the pbp_defense retention window.

The family was marked NEVER DELETE citing two ablation harnesses
(ablation_defense.py, run_mlb_runline_defense_ablation.py) that were deleted
in ff372c3, so the exemption protected nothing while the daily run wrote
13.1 MB of pitch projection after it: 52 files / 333 MB on 2026-09-26, and
every Kaggle run clones the whole set before it starts.

These tests pin the traced facts so the exemption cannot come back on the
strength of a consumer that does not exist.

2026-10-03 update: the same audit that traced pbp_defense's dead consumers
traced pbp_chunks/ — its exemption cited build_pbp_defense, which never read
the directory, while its real producer had been deleted in ff372c3. The
38-file / 2.1 MB cache and the exemption both went; see
test_pbp_chunks_are_no_longer_protected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import retention_policy as rp  # noqa: E402

ANCHOR = "20260925"
# What master_pipeline hands classify_artifact: the blanket window is
# anchor-N days for N in 0..9, computed in the pipeline, not here.
RETENTION_DATES = {f"202609{d:02d}" for d in range(15, 26)}


def _v(rel: str) -> str:
    return rp.classify_artifact(rel, set(), RETENTION_DATES, set(), set(),
                                anchor_date=ANCHOR)


def test_old_pitch_projection_is_prunable():
    """The regression: outside the blanket window it must fall to stale, not
    protected. 52 files survived here and the family grew forever."""
    assert _v("mlb-backend/data_delivery/pbp_defense_20260831.parquet") == "stale"
    assert _v("mlb-backend/data_delivery/pbp_defense_20260831.meta.json") == "stale"


def test_recent_pitch_projection_is_kept():
    for d in ("20260922", "20260924", "20260925"):
        assert _v(f"mlb-backend/data_delivery/pbp_defense_{d}.parquet") == "current"
        assert _v(f"mlb-backend/data_delivery/pbp_defense_{d}.meta.json") == "current"


def test_this_runs_own_projection_is_never_pruned():
    assert rp.classify_artifact(
        "mlb-backend/data_delivery/pbp_defense_20260925.parquet",
        {"mlb-backend/data_delivery/pbp_defense_20260925.parquet"},
        RETENTION_DATES, set(), set(), anchor_date=ANCHOR) == "seen"


def test_backfill_anchor_keeps_newer_artifacts():
    """A run whose window ends in the past must not prune what came after."""
    assert rp.classify_artifact(
        "mlb-backend/data_delivery/pbp_defense_20261005.parquet",
        set(), RETENTION_DATES, set(), set(), anchor_date=ANCHOR) == "current"


def test_monitor_series_stays_protected():
    """The correction is scoped to pbp_defense. The producer folds ALL dated
    monitors into the rolling per-line series, so that exemption is real."""
    for d in ("20260826", "20260831", "20260701"):
        assert _v(f"mlb-backend/data_delivery/run_engine_monitor_{d}.json") \
            == "protected"


def test_pbp_chunks_are_no_longer_protected():
    """2026-10-03: the chunk cache lost its exemption AND its files. The old
    exemption cited build_pbp_defense as a cumulative consumer, but that
    builder reads --source pitches.parquet and never opens the directory,
    and the only producer (fetch_pbp_chunks.py) was deleted in ff372c3 —
    so the family could never be refreshed again. Re-adding a NEVER DELETE
    for a frozen cache with no producer is the same dead-citation failure
    this module exists to prevent."""
    rel = "mlb-backend/data_delivery/pbp_chunks/pbp_2026-09-01_2026-09-15.parquet"
    assert not rp.is_never_delete(rel), rel
    assert _v(rel) == "stale"


def test_no_prefix_match_keeps_pbp_defense_protected():
    """is_never_delete must be false for the family now, or the whole fix is
    inert — this is the assertion that would have caught the dead citation."""
    for rel in ("mlb-backend/data_delivery/pbp_defense_20260831.parquet",
                "mlb-backend/data_delivery/pbp_defense_20260925.meta.json"):
        assert not rp.is_never_delete(rel), rel


def test_il_stint_table_still_never_deleted():
    """Adjacent protection: the dateless IL table has no window to save it."""
    for rel in ("mlb-backend/data_delivery/il_stints.parquet",
                "mlb-backend/data_delivery/il_stints.meta.json"):
        assert rp.is_never_delete(rel), rel


def test_player_positions_map_survives_the_sweep():
    """The 2026-10-03 incident: the 06:36 sweep classified the dateless
    position map stale and git rm'd it, so the 11:08 run found neither
    copy and all 24 served pl_* columns shipped at 0% coverage —
    median-imputed constants the drift gate cannot see. The map is a
    master (the pl_<pos> pool key; only .adhoc/mlb_fetch_positions.py
    rebuilds it), so no window may ever delete it."""
    rel = "mlb-backend/data_delivery/player_positions.parquet"
    assert rp.is_never_delete(rel), rel
    assert _v(rel) == "protected"


def test_totals_history_store_survives_the_sweep():
    """Same trap, adjacent family: the frozen totals-history store is dateless
    and priced-once. A run whose store update throws skips the CSV rewrite,
    so the name reads stale and the sweep destroys rows that only re-seed
    from the 10-day markets window."""
    for name in ("run_engine_totals_history.csv",
                 "run_engine_totals_history.meta.json"):
        rel = f"mlb-backend/data_delivery/{name}"
        assert rp.is_never_delete(rel), rel
        assert _v(rel) == "protected"


def test_family_is_registered_and_not_allowlisted():
    fam = rp._family_for("mlb-backend/data_delivery/pbp_defense_20260925.parquet")
    assert fam is not None and fam.family == "pbp_defense"
    assert fam.allowlisted is False
    assert "newest-only" in fam.notes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
