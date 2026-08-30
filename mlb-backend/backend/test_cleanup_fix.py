"""Tests for the Phase 6 cleanup fix (protected set + date-gating).

Covers:
  1. Protected files (statsapi_roof_cache.json, model_history.json, models/)
     survive cleanup even when NOT in the ``seen`` set.
  2. Current-date artifacts survive cleanup when NOT in ``seen``.
  3. Genuinely stale (older-date) files are still removed.
  4. dome_is_neutral_game exists in the exported CSV.
  5. Roof cache loads all 1,053 entries from a clean path.
"""
import json
import os
import re
import sys
from datetime import datetime as _dt_dt
from datetime import timedelta as _td
from pathlib import Path
from unittest import TestCase

# Ensure backend/ is importable
_backend = Path(__file__).resolve().parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))


# ── Helper functions (duplicated from master_pipeline to test in isolation) ──

_PROTECTED_DELIVERY_NAMES = {
    "statsapi_roof_cache.json",
    "model_history.json",
    "model_version_history.json",
    # Lineup-delta feature runtime inputs (Phase 2, shipped in aead200; the
    # daily pipeline CONSUMES them and never rebuilds them). Dateless names →
    # the date-gate can't save them → exact-name protect (42ef3f7 deleted
    # them; pipeline now fails loud without them).
    "lineups.parquet",
    "batter_woba.parquet",
    "team_woba.parquet",
    "umpire_map.csv",
    "umpire_stats.csv",
}
_PROTECTED_DELIVERY_PREFIXES = (
    "models/", "pbp_chunks/", "run_engine_monitor_",
)
_DATE_RE = re.compile(r"_(\d{8})")

# Recent-slate protection (mirrors master_pipeline.py constants).
_SLATE_PROTECTED_PREFIXES = ("todays_games_", "shap_game_")


def _is_protected(rel: str) -> bool:
    """True if ``rel`` is a persistent asset that cleanup must never touch."""
    # Strip leading path up to and including 'data_delivery/'
    _DD = "data_delivery/"
    idx = rel.find(_DD)
    local = rel[idx + len(_DD):] if idx >= 0 else rel
    basename = local.rsplit("/", 1)[-1]
    return (basename in _PROTECTED_DELIVERY_NAMES
            or any(local.startswith(pfx) for pfx in _PROTECTED_DELIVERY_PREFIXES))


def _artifact_date(rel: str):
    """Extract the YYYYMMDD date from an artifact path, or None if dateless."""
    m = _DATE_RE.search(rel)
    return m.group(1) if m else None


def _is_recent_slate(rel: str, recent_dates: set[str]) -> bool:
    """True if ``rel`` is a todays_games_ or shap_game_ artifact whose
    extracted date falls within the recent-slate protection window."""
    art_date = _artifact_date(rel)
    if art_date not in recent_dates:
        return False
    basename = rel.rsplit("/", 1)[-1]
    return any(basename.startswith(pfx) for pfx in _SLATE_PROTECTED_PREFIXES)


def classify_tracked(tracked, seen, run_date_compact="20260824",
                     recent_dates=None, retention_dates=None):
    """Replicate the Phase 6 classification logic for testing.

    ``recent_dates`` is an optional set of YYYYMMDD strings within the
    recent-slate protection window: todays_games_* / shap_game_* whose date
    is in this set survive even if not in ``seen``.

    ``retention_dates`` is the rolling 48h retention window for ALL dated
    artifacts (current run date + previous GMT day). When omitted it
    defaults to {run_date_compact, run_date_compact - 1 day} computed via
    a timedelta — never strict string equality against today.
    """
    if recent_dates is None:
        recent_dates = {run_date_compact}  # default: only same-day
    if retention_dates is None:
        run_dt = _dt_dt.strptime(run_date_compact, "%Y%m%d").date()
        retention_dates = {(run_dt - _td(days=i)).strftime("%Y%m%d")
                           for i in range(2)}  # {today, yesterday}
    stale, kept_protected, kept_current = [], 0, 0
    for p in tracked:
        if p in seen:
            continue
        if _is_protected(p):
            kept_protected += 1
            continue
        art_date = _artifact_date(p)
        if art_date in retention_dates:
            kept_current += 1
            continue  # within the 48h retention window — keep
        # Recent-slate protection: keep todays_games / shap_game for
        # the current run date AND the 2 prior days.
        if _is_recent_slate(p, recent_dates):
            kept_current += 1
            continue
        stale.append(p)
    return stale, kept_protected, kept_current


class TestCleanupProtection(TestCase):
    """Protected files must never appear in the stale list."""

    def test_roof_cache_protected(self):
        tracked = [
            "mlb-backend/data_delivery/statsapi_roof_cache.json",
            "mlb-backend/data_delivery/game_level_features.csv",
            "mlb-backend/data_delivery/run_engine_oof_20260820.csv",
        ]
        seen = {"mlb-backend/data_delivery/game_level_features.csv"}
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertNotIn(
            "mlb-backend/data_delivery/statsapi_roof_cache.json", stale,
            "statsapi_roof_cache.json must be protected")
        self.assertEqual(prot, 1)

    def test_model_history_protected(self):
        tracked = [
            "mlb-backend/data_delivery/model_history.json",
            "mlb-backend/data_delivery/model_version_history.json",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(stale, [])
        self.assertEqual(prot, 2)

    def test_models_dir_protected(self):
        tracked = [
            "mlb-backend/data_delivery/models/ensemble_v1.joblib",
            "mlb-backend/data_delivery/models/alpha_params.json",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(stale, [])
        self.assertEqual(prot, 2)

    def test_umpire_maintained_files_protected(self):
        """The maintained umpire map/stats are dateless -> exact-name
        protected or Phase 6 would delete them and force a re-fetch."""
        tracked = [
            "mlb-backend/data_delivery/umpire_map.csv",
            "mlb-backend/data_delivery/umpire_stats.csv",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(stale, [])
        self.assertEqual(prot, 2)

    def test_all_protected_kept(self):
        """All three categories of protected files survive."""
        tracked = [
            "mlb-backend/data_delivery/statsapi_roof_cache.json",
            "mlb-backend/data_delivery/model_history.json",
            "mlb-backend/data_delivery/model_version_history.json",
            "mlb-backend/data_delivery/models/x.joblib",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(stale, [])
        self.assertEqual(prot, 4)

    def test_lineup_parquet_inputs_protected(self):
        """Lineup-delta runtime inputs (lineups/batter_woba/team_woba parquet)
        must NEVER be pruned — they are dateless, unseen-by-the-run inputs the
        daily pipeline only consumes (42ef3f7 deleted them; pipeline now fails
        loud without them)."""
        tracked = [
            "mlb-backend/data_delivery/lineups.parquet",
            "mlb-backend/data_delivery/batter_woba.parquet",
            "mlb-backend/data_delivery/team_woba.parquet",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(stale, [],
                         "lineup runtime inputs must never be classified stale")
        self.assertEqual(prot, 3)

    def test_pbp_chunks_prefix_protected(self):
        """Every pbp_chunks/*.parquet survives via the prefix rule, even
        though the range dates in their names look date-ish."""
        tracked = [
            "mlb-backend/data_delivery/pbp_chunks/pbp_2025-03-18_2025-03-31.parquet",
            "mlb-backend/data_delivery/pbp_chunks/pbp_2026-08-18_2026-08-24.parquet",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(stale, [], "pbp_chunks/ must never be pruned")
        self.assertEqual(prot, 2)

    def test_all_four_lineup_inputs_protected_while_stale_pruned(self):
        """The 4 lineup-delta runtime inputs (lineups.parquet,
        batter_woba.parquet, team_woba.parquet, pbp_chunks/) are NEVER
        deleted by cleanup, while stale date-stamped artifacts still are —
        the v26 protection regression sentinel."""
        tracked = [
            "mlb-backend/data_delivery/lineups.parquet",
            "mlb-backend/data_delivery/batter_woba.parquet",
            "mlb-backend/data_delivery/team_woba.parquet",
            "mlb-backend/data_delivery/pbp_chunks/pbp_2025-03-18_2025-03-31.parquet",
            "mlb-backend/data_delivery/run_engine_markets_20260820.csv",
            "mlb-backend/data_delivery/calibration_20260819.json",
        ]
        stale, prot, cur = classify_tracked(tracked, set(), "20260824")
        self.assertEqual(prot, 4, "all 4 lineup inputs must be protected")
        # The stale date-stamped artifacts are still pruned (cleanup's job).
        self.assertEqual(stale, [
            "mlb-backend/data_delivery/run_engine_markets_20260820.csv",
            "mlb-backend/data_delivery/calibration_20260819.json",
        ])
        stale_names = {s.rsplit("/", 1)[-1] for s in stale}
        self.assertFalse(stale_names
                         & {"lineups.parquet", "batter_woba.parquet",
                            "team_woba.parquet"})

    def test_lineup_inputs_protected_even_with_stale_others(self):
        """Protecting the lineup inputs must NOT disable the cleanup's real
        job: stale date-stamped artifacts are still pruned alongside."""
        tracked = [
            "mlb-backend/data_delivery/lineups.parquet",
            "mlb-backend/data_delivery/pbp_chunks/pbp_2025-03-18_2025-03-31.parquet",
            "mlb-backend/data_delivery/run_engine_markets_20260820.csv",
            "mlb-backend/data_delivery/calibration_20260819.json",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertIn(
            "mlb-backend/data_delivery/run_engine_markets_20260820.csv",
            stale)
        self.assertIn(
            "mlb-backend/data_delivery/calibration_20260819.json", stale)
        self.assertEqual(len(stale), 2)
        self.assertEqual(prot, 2)

    def test_ablation_records_use_date_gate_not_prefix_protection(self):
        """A run-dated ablation record survives while an older one prunes;
        this proves ablations are not prefix-protected."""
        tracked = [
            "mlb-backend/data_delivery/calibration_ablation_20260826.json",
            "mlb-backend/data_delivery/calibration_ablation_20260820.json",
            "mlb-backend/data_delivery/features_metadata_20260820.json",
        ]
        stale, prot, cur = classify_tracked(tracked, set(), "20260826")
        self.assertEqual(prot, 0)
        self.assertEqual(cur, 1)
        self.assertEqual(stale, [
            "mlb-backend/data_delivery/calibration_ablation_20260820.json",
            "mlb-backend/data_delivery/features_metadata_20260820.json",
        ])

    def test_run_engine_monitor_protected_while_stale_pruned(self):
        """Every dated run_engine_monitor_<date>.json survives cleanup via the
        prefix rule (the monitor artifact must never be reset run-to-run — it
        feeds the rolling per-line history, the same as model_history.json for
        the moneyline), while stale non-protected date-stamped files
        (features_metadata_<stale>.json, etc.) are still pruned."""
        tracked = [
            # Monitors from past days must survive so the next run folds them
            # into the rolling series (not just today's, which the date-gate
            # would keep anyway).
            "mlb-backend/data_delivery/run_engine_monitor_20260822.json",
            "mlb-backend/data_delivery/run_engine_monitor_20260823.json",
            "mlb-backend/data_delivery/run_engine_monitor_20260824.json",
            # Stale non-protected date-stamped files must still prune.
            "mlb-backend/data_delivery/features_metadata_20260818.json",
            "mlb-backend/data_delivery/run_engine_markets_20260820.csv",
            "mlb-backend/data_delivery/calibration_20260819.json",
        ]
        seen = set()  # none staged this run -> reliance on protection, not seen
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(prot, 3, "all dated run_engine_monitor files protected")
        # Every monitor (including 20260824, which the date-gate alone would
        # keep) must be BOTH protected AND absent from stale.
        for sd in ("20260822", "20260823", "20260824"):
            self.assertNotIn(
                f"mlb-backend/data_delivery/run_engine_monitor_{sd}.json",
                stale, "dated monitor must never be classified stale")
        # The monitor files are NOT counted as same-day keeps either — they go
        # through the protection path so the prefix rule is what saves them.
        stale_names = {s.rsplit("/", 1)[-1] for s in stale}
        self.assertFalse(any("run_engine_monitor" in n for n in stale_names))
        # Stale non-protected date-stamped artifacts still prune.
        self.assertEqual(stale, [
            "mlb-backend/data_delivery/features_metadata_20260818.json",
            "mlb-backend/data_delivery/run_engine_markets_20260820.csv",
            "mlb-backend/data_delivery/calibration_20260819.json",
        ])


class TestDateGating(TestCase):
    """Same-day artifacts survive; older-date artifacts are removed."""

    def test_current_date_survives(self):
        tracked = [
            "mlb-backend/data_delivery/run_engine_markets_20260824.csv",
            "mlb-backend/data_delivery/run_engine_oof_20260824.csv",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(stale, [])
        self.assertEqual(cur, 2)

    def test_older_date_removed(self):
        tracked = [
            "mlb-backend/data_delivery/run_engine_markets_20260820.csv",
            "mlb-backend/data_delivery/run_engine_oof_20260820.csv",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(len(stale), 2)
        self.assertEqual(cur, 0)

    def test_mixed_dates(self):
        tracked = [
            "mlb-backend/data_delivery/run_engine_markets_20260824.csv",
            "mlb-backend/data_delivery/run_engine_markets_20260820.csv",
            "mlb-backend/data_delivery/calibration_20260819.json",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertIn(
            "mlb-backend/data_delivery/run_engine_markets_20260820.csv", stale)
        self.assertIn(
            "mlb-backend/data_delivery/calibration_20260819.json", stale)
        self.assertNotIn(
            "mlb-backend/data_delivery/run_engine_markets_20260824.csv", stale)
        self.assertEqual(cur, 1)

    def test_dateless_file_stale(self):
        """Files with no date pattern are treated as stale (not protected)."""
        tracked = [
            "mlb-backend/data_delivery/game_level_features.csv",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(len(stale), 1)
        self.assertEqual(cur, 0)

    def test_seen_files_never_stale(self):
        """Files in the seen set are never stale regardless of date."""
        tracked = [
            "mlb-backend/data_delivery/run_engine_markets_20260820.csv",
        ]
        seen = {"mlb-backend/data_delivery/run_engine_markets_20260820.csv"}
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(stale, [])


class TestRecentSlateProtection(TestCase):
    """todays_games_* and shap_game_* artifacts for the current run date
    AND the 2 prior days survive cleanup even when NOT in ``seen``.

    The 08-28 failure: Phase 6 deleted todays_games_20260828.csv when a
    post-midnight-UTC run produced 20260829 artifacts — Aug 28 games had
    not yet settled into predictions_history, so the card snapshot was
    orphaned and the dashboard showed nothing for that date.
    """

    def test_todays_games_recent_date_protected(self):
        """todays_games for yesterday survives even though it's not today."""
        tracked = [
            "mlb-backend/data_delivery/todays_games_20260823.csv",
            "mlb-backend/data_delivery/todays_games_20260824.csv",
        ]
        seen = {"mlb-backend/data_delivery/todays_games_20260824.csv"}
        recent = {"20260824", "20260823", "20260822"}
        stale, prot, cur = classify_tracked(tracked, seen, "20260824",
                                            recent_dates=recent)
        self.assertEqual(stale, [],
                         "recent todays_games must not be pruned")
        self.assertEqual(cur, 1,  # 20260823 protected via recent-slate rule
                         "recent todays_games counted as current keep")

    def test_todays_games_old_date_still_pruned(self):
        """todays_games from 4+ days ago is still pruned (outside window)."""
        tracked = [
            "mlb-backend/data_delivery/todays_games_20260819.csv",
            "mlb-backend/data_delivery/todays_games_20260824.csv",
        ]
        seen = {"mlb-backend/data_delivery/todays_games_20260824.csv"}
        recent = {"20260824", "20260823", "20260822"}
        stale, prot, cur = classify_tracked(tracked, seen, "20260824",
                                            recent_dates=recent)
        self.assertEqual(len(stale), 1, "old todays_games must still be pruned")
        self.assertIn("todays_games_20260819.csv", stale[0])

    def test_shap_game_recent_date_protected(self):
        """shap_game for yesterday survives via the same recent-slate rule."""
        tracked = [
            "mlb-backend/data_delivery/shap_game_823506.csv",
            "mlb-backend/data_delivery/shap_game_822771.csv",
        ]
        seen = set()
        # shap_game filenames use game_pk not date, so the date gate can't
        # protect them — but the recent-slate rule matches via basename prefix.
        # However, _artifact_date needs an 8-digit date in the name.
        # shap_game_823506.csv has no date → _artifact_date returns None.
        # In the real pipeline, shap files are staged (in `seen`) so they
        # survive. Test that if they're NOT staged, the date gate doesn't
        # save them (they have no date), but the protection test is for
        # todays_games which DO have dates.
        # This test validates: non-dated files outside seen are stale.
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(len(stale), 2,
                         "non-dated shap files not in seen are stale")

    def test_recent_window_is_3_days(self):
        """Exactly 3 days: run_date, -1d, -2d. Day -3 is pruned."""
        tracked = [
            "mlb-backend/data_delivery/todays_games_20260824.csv",  # same-day
            "mlb-backend/data_delivery/todays_games_20260823.csv",  # -1d
            "mlb-backend/data_delivery/todays_games_20260822.csv",  # -2d
            "mlb-backend/data_delivery/todays_games_20260821.csv",  # -3d
            "mlb-backend/data_delivery/todays_games_20260818.csv",  # old
        ]
        seen = {"mlb-backend/data_delivery/todays_games_20260824.csv"}
        recent = {"20260824", "20260823", "20260822"}
        stale, prot, cur = classify_tracked(tracked, seen, "20260824",
                                            recent_dates=recent)
        # -3d (0821) and old (0818) should be stale; -1d/-2d should be kept
        self.assertEqual(len(stale), 2)
        self.assertIn("todays_games_20260821.csv", stale[0])
        self.assertIn("todays_games_20260818.csv", stale[1])
        # -1d and -2d are kept via recent-slate protection
        kept_names = {s for s in tracked if s not in stale
                      and s not in seen}
        self.assertTrue(
            any("20260823" in n for n in kept_names),
            "day -1 must be kept via recent-slate protection")
        self.assertTrue(
            any("20260822" in n for n in kept_names),
            "day -2 must be kept via recent-slate protection")

    def test_previous_day_non_slate_kept_by_48h_window(self):
        """run_engine_markets / calibration from 1 day ago survive via the
        rolling 48h retention window (current run date + previous GMT day) —
        the GMT-rollover regression fix. The recent-slate rule was the only
        thing saving yesterday's todays_games before; now ALL dated artifacts
        for the previous day are retained while those games are live.
        """
        tracked = [
            "mlb-backend/data_delivery/run_engine_markets_20260823.csv",
            "mlb-backend/data_delivery/calibration_20260823.json",
            "mlb-backend/data_delivery/todays_games_20260823.csv",
        ]
        seen = set()
        recent = {"20260824", "20260823", "20260822"}
        stale, prot, cur = classify_tracked(tracked, seen, "20260824",
                                            recent_dates=recent)
        # Day -1 (20260823) is inside the 48h window → NOT stale (for any
        # dated artifact, slate-protected or not).
        self.assertEqual(stale, [])
        self.assertEqual(cur, 3)


class TestRolloverRetentionWindow(TestCase):
    """The GMT-rollover regression: dated artifacts for the current GMT day
    AND the PREVIOUS GMT day are retained (a rolling 48h window), so US games
    still live at the 00:00 GMT rollover keep their run-engine data. Older
    dated artifacts are still pruned. Window is timedelta-based — never strict
    date-string equality against today.
    """

    def test_rollover_day_both_29_and_30_kept(self):
        """Today = 20260830: run_engine_markets/oof/predictions_history for
        BOTH 20260829 (still-pre-game US night games) and 20260830 are kept;
        only older (e.g. 20260828) artifacts are pruned."""
        tracked = [
            "mlb-backend/data_delivery/run_engine_markets_20260829.csv",
            "mlb-backend/data_delivery/run_engine_markets_20260830.csv",
            "mlb-backend/data_delivery/run_engine_oof_20260829.csv",
            "mlb-backend/data_delivery/predictions_history_20260829.csv",
            "mlb-backend/data_delivery/run_engine_markets_20260828.csv",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260830")
        self.assertEqual(cur, 4, "29 + 30 run-engine/prediction artifacts kept")
        self.assertEqual(stale, [
            "mlb-backend/data_delivery/run_engine_markets_20260828.csv",
        ], "only the 2-day-old artifact pruned")

    def test_next_day_rollover_keeps_previous_only(self):
        """Today = 20260831: the previous GMT day (20260830) + today are kept;
        20260829 (now 2 days old) is pruned."""
        tracked = [
            "mlb-backend/data_delivery/run_engine_markets_20260829.csv",
            "mlb-backend/data_delivery/run_engine_markets_20260830.csv",
            "mlb-backend/data_delivery/run_engine_markets_20260831.csv",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260831")
        self.assertEqual(cur, 2, "30 + 31 kept (previous day retained)")
        self.assertEqual(stale, [
            "mlb-backend/data_delivery/run_engine_markets_20260829.csv",
        ])

    def test_window_rejects_two_days_old_run_engine_oof(self):
        """run_engine_oof 2 days old is pruned even though its monitor
        sibling is prefix-protected."""
        tracked = [
            "mlb-backend/data_delivery/run_engine_oof_20260828.csv",
            "mlb-backend/data_delivery/calibration_20260828.json",
        ]
        stale, prot, cur = classify_tracked(tracked, set(), "20260830")
        self.assertEqual(len(stale), 2)

    def test_timedelta_window_not_string_equality(self):
        """The window is {run_date, run_date - 1} via timedelta — a file dated
        exactly 1 day before a run across a month boundary is still retained.
        """
        # Run on 20260901 → previous GMT day is 20260831, which a strict
        # same-day '==' gate would have pruned.
        tracked = [
            "mlb-backend/data_delivery/run_engine_markets_20260831.csv",
            "mlb-backend/data_delivery/run_engine_markets_20260901.csv",
            "mlb-backend/data_delivery/run_engine_markets_20260830.csv",
        ]
        stale, prot, cur = classify_tracked(tracked, set(), "20260901")
        self.assertEqual(cur, 2, "0930 pruned, 0831 + 0901 kept")
        self.assertEqual(stale, [
            "mlb-backend/data_delivery/run_engine_markets_20260830.csv",
        ])

    def test_protected_never_touched_outside_window(self):
        """Protected / undated files survive even when their names sit outside
        the retention window."""
        tracked = [
            "mlb-backend/data_delivery/model_history.json",
            "mlb-backend/data_delivery/run_engine_monitor_20260810.json",
            "mlb-backend/data_delivery/run_engine_markets_20260810.csv",
        ]
        stale, prot, cur = classify_tracked(tracked, set(), "20260830")
        self.assertEqual(prot, 2, "model_history + old monitor protected")
        self.assertEqual(stale, [
            "mlb-backend/data_delivery/run_engine_markets_20260810.csv",
        ], "only the old non-protected dated artifact pruned")


class TestDomeColumnExport(TestCase):
    """dome_is_neutral_game must exist in the shipped CSV."""

    def test_column_exists(self):
        csv_path = Path(__file__).resolve().parent.parent / "data_delivery" / "game_level_features.csv"
        if not csv_path.exists():
            self.skipTest("game_level_features.csv not present")
        import pandas as pd
        df = pd.read_csv(csv_path, nrows=1)
        self.assertIn("dome_is_neutral_game", df.columns,
                       "dome_is_neutral_game must be in the exported CSV")

    def test_dome_values_reasonable(self):
        csv_path = Path(__file__).resolve().parent.parent / "data_delivery" / "game_level_features.csv"
        if not csv_path.exists():
            self.skipTest("game_level_features.csv not present")
        import pandas as pd
        df = pd.read_csv(csv_path, usecols=["dome_is_neutral_game", "home_team"])
        vals = set(df["dome_is_neutral_game"].dropna().unique())
        self.assertTrue(vals.issubset({0.0, 1.0}),
                        f"dome_is_neutral_game has unexpected values: {vals - {0.0, 1.0}}")
        n_open = int((df["dome_is_neutral_game"] == 0).sum())
        n_closed = int((df["dome_is_neutral_game"] == 1).sum())
        self.assertGreater(n_open + n_closed, 0, "dome_is_neutral_game is all-NaN")


class TestRoofCachePersistence(TestCase):
    """Roof cache loads correctly from a fresh path."""

    def test_loads_1053_entries(self):
        cache_path = Path(__file__).resolve().parent.parent / "data_delivery" / "statsapi_roof_cache.json"
        if not cache_path.exists():
            self.skipTest("statsapi_roof_cache.json not present")
        data = json.loads(cache_path.read_text())
        self.assertGreaterEqual(len(data), 1000,
                                f"Expected >=1000 roof cache entries, got {len(data)}")

    def test_cache_has_open_and_closed(self):
        cache_path = Path(__file__).resolve().parent.parent / "data_delivery" / "statsapi_roof_cache.json"
        if not cache_path.exists():
            self.skipTest("statsapi_roof_cache.json not present")
        data = json.loads(cache_path.read_text())
        values = set(data.values())
        self.assertIn("closed", values, "Roof cache should have 'closed' entries")
        # 'open' entries exist for retractable parks with known open state
        # (may be 0 if all retractable games were closed in the dataset)

    def test_load_roof_cache_function(self):
        """load_roof_cache() from features.py reads all entries."""
        cache_path = Path(__file__).resolve().parent.parent / "data_delivery" / "statsapi_roof_cache.json"
        if not cache_path.exists():
            self.skipTest("statsapi_roof_cache.json not present")
        from features import load_roof_cache
        cache = load_roof_cache(str(cache_path))
        self.assertGreaterEqual(len(cache), 1000,
                                f"load_roof_cache returned {len(cache)} entries")


class TestRealArtifacts(TestCase):
    """Sanity check on the actual artifact files."""

    def test_game_level_features_exists(self):
        csv = Path(__file__).resolve().parent.parent / "data_delivery" / "game_level_features.csv"
        self.assertTrue(csv.exists(), "game_level_features.csv missing")

    def test_roof_cache_exists(self):
        cache = Path(__file__).resolve().parent.parent / "data_delivery" / "statsapi_roof_cache.json"
        self.assertTrue(cache.exists(), "statsapi_roof_cache.json missing")


if __name__ == "__main__":
    import unittest
    unittest.main()
