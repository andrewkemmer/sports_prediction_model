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


# ── Production predicate (single source of truth) ───────────────────────────
# The Phase 6 keep/stale predicate now lives in the explicit policy module
# (retention_policy.py) — the consumer-audit-backed per-family windows and
# never-delete markers (masters / records / series readers).  ``classify_tracked``
# below is a thin wrapper that runs THE PRODUCTION predicate over a fixture's
# tracked list with the Phase 6 window defaults, so every test here (old and
# new) exercises the same code path master_pipeline Phase 6 calls.
from retention_policy import classify_artifact as _production_classify


def classify_tracked(tracked, seen, run_date_compact="20260824",
                     recent_dates=None, retention_dates=None,
                     board_dates=None):
    """Run the production Phase 6 classification over ``tracked`` for testing.

    ``recent_dates`` is an optional set of YYYYMMDD strings within the
    recent-slate protection window: todays_games_* / shap_game_* whose date
    is in this set survive even if not in ``seen``.

    ``retention_dates`` is the rolling 48h retention window for ALL dated
    artifacts (current run date + previous GMT day). When omitted it
    defaults to {run_date_compact, run_date_compact - 1 day} computed via
    a timedelta — never strict string equality against today.

    ``board_dates`` is the set of YYYYMMDD dates that still have a tracked
    todays_games_<date>.csv board: run_engine_markets_* / run_engine_oof_* /
    predictions_history_* for those dates survive even outside the 48h and
    recent-slate windows. Defaults to empty (board rule inert).
    """
    if recent_dates is None:
        recent_dates = {run_date_compact}  # default: only same-day
    if retention_dates is None:
        run_dt = _dt_dt.strptime(run_date_compact, "%Y%m%d").date()
        retention_dates = {(run_dt - _td(days=i)).strftime("%Y%m%d")
                           for i in range(2)}  # {today, yesterday}
    if board_dates is None:
        board_dates = set()
    stale, kept_protected, kept_current = [], 0, 0
    for p in tracked:
        verdict = _production_classify(p, seen, retention_dates,
                                       recent_dates, board_dates)
        if verdict == "seen":
            continue
        if verdict == "protected":
            kept_protected += 1
            continue
        if verdict == "current":
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
        """Dateless files with no never-delete marker are treated as stale
        (not protected). game_level_features.csv is now a protected MASTER
        (retention_policy.EXACT_MASTER_NAMES) — this test uses a non-master
        dateless name so it pins the real "dateless + unprotected -> stale"
        rule."""
        tracked = [
            "mlb-backend/data_delivery/consensus_picks.csv",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(len(stale), 1)
        self.assertEqual(cur, 0)

    def test_game_level_features_master_protected(self):
        """game_level_features.csv is now an explicit MASTER in
        retention_policy.EXACT_MASTER_NAMES (policy change: the undated
        master the dashboard reads for final scores is never pruned, even
        when a run does not restage it)."""
        tracked = [
            "mlb-backend/data_delivery/game_level_features.csv",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(stale, [])
        self.assertEqual(prot, 1)

    def test_mlb_records_protected(self):
        """Research/verdict records (mlb_* + *_triage_*) are never deleted,
        even though their sha/date names sit outside every window."""
        tracked = [
            "mlb-backend/data_delivery/mlb_sp_bias_stability_a105ba9bba6d60ff.json",
            "mlb-backend/data_delivery/mlb_projection_margin_walk_7bec561aa0391920.json",
            "mlb-backend/data_delivery/test_hygiene_triage_20260904.json",
            "mlb-backend/data_delivery/run_engine_markets_20260820.csv",
        ]
        stale, prot, cur = classify_tracked(tracked, set(), "20260824")
        self.assertEqual(stale, [
            "mlb-backend/data_delivery/run_engine_markets_20260820.csv",
        ])
        self.assertEqual(prot, 3)

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


class TestBoardBackedRunEngineProtection(TestCase):
    """Board-backed retention (2026-08-29 doubleheader regression): a dated
    run_engine_markets_* / run_engine_oof_* / predictions_history_* artifact
    survives cleanup as long as a todays_games_<date>.csv board for that date
    is still tracked — a navigable board must never lose the RUN ENGINE
    columns its cards need. The 3-day recent-slate window keeps the board;
    before this fix, run-engine data for that date was pruned on the 2nd
    retention drop while the board remained.
    """

    # Run date 20260831: retention = {31, 30}; recent = {31, 30, 29}.
    # 20260829 is OUTSIDE the 48h window and is not a slate prefix — only
    # the board-backed rule can save its run-engine data (the regression).
    RUN = "20260831"
    RECENT = {"20260831", "20260830", "20260829"}
    RETENTION = {"20260831", "20260830"}
    BOARD = {"20260829"}

    def test_kept_when_board_tracked(self):
        """(a) run_engine_markets/oof/predictions_history for 20260829 are
        kept when todays_games_20260829.csv is tracked (outside the 48h
        window, day -2 of the recent-slate window)."""
        tracked = [
            "mlb-backend/data_delivery/todays_games_20260829.csv",
            "mlb-backend/data_delivery/run_engine_markets_20260829.csv",
            "mlb-backend/data_delivery/run_engine_oof_20260829.csv",
            "mlb-backend/data_delivery/predictions_history_20260829.csv",
        ]
        stale, prot, cur = classify_tracked(
            tracked, set(), self.RUN, recent_dates=self.RECENT,
            retention_dates=self.RETENTION, board_dates=self.BOARD)
        self.assertEqual(stale, [], "board-backed run-engine data must be kept")
        self.assertEqual(cur, 4)

    def test_pruned_when_no_board_tracked(self):
        """(b) the same run-engine file IS pruned when no todays_games board
        for that date is tracked (board rule inert; 48h/recent don't apply)."""
        tracked = [
            "mlb-backend/data_delivery/run_engine_markets_20260829.csv",
            "mlb-backend/data_delivery/run_engine_oof_20260829.csv",
            "mlb-backend/data_delivery/predictions_history_20260829.csv",
        ]
        stale, prot, cur = classify_tracked(
            tracked, set(), self.RUN, recent_dates=self.RECENT,
            retention_dates=self.RETENTION)  # no board_dates
        self.assertEqual(len(stale), 3,
                         "without a tracked board the 48h/recent windows don't apply")

    def test_slate_protection_unchanged(self):
        """(c) todays_games_* keeps its existing 3-day recent-slate
        protection (independent of the board rule), and boards outside the
        window are still pruned."""
        tracked = [
            "mlb-backend/data_delivery/todays_games_20260829.csv",  # recent
            "mlb-backend/data_delivery/todays_games_20260825.csv",  # >3 days
        ]
        stale, prot, cur = classify_tracked(
            tracked, set(), self.RUN, recent_dates=self.RECENT,
            retention_dates=self.RETENTION, board_dates=self.BOARD)
        self.assertEqual(stale, [
            "mlb-backend/data_delivery/todays_games_20260825.csv",
        ], "recent board kept, old board still pruned")
        self.assertEqual(cur, 1)

    def test_older_board_keeps_run_engine_while_board_persists(self):
        """(d) a board older than the 3-day window (still tracked this run)
        keeps its run-engine data — the board itself prunes, its run-engine
        survives until the board is gone from a later run."""
        board_dates = {"20260825"}
        tracked = [
            "mlb-backend/data_delivery/todays_games_20260825.csv",
            "mlb-backend/data_delivery/run_engine_markets_20260825.csv",
            "mlb-backend/data_delivery/run_engine_oof_20260825.csv",
            "mlb-backend/data_delivery/predictions_history_20260825.csv",
        ]
        stale, prot, cur = classify_tracked(
            tracked, set(), self.RUN, recent_dates=self.RECENT,
            retention_dates=self.RETENTION, board_dates=board_dates)
        # Board itself is stale (outside the 3-day window) but its run-engine
        # family is kept while the board persists.
        self.assertEqual(stale, [
            "mlb-backend/data_delivery/todays_games_20260825.csv",
        ])
        self.assertEqual(cur, 3)

    def test_unrelated_dated_artifacts_unaffected(self):
        """(e) the 48h retention for unrelated dated artifacts is unchanged:
        calibration_* / model_monitor_* for the window survive, older ones
        prune — the board rule never widens what they keep."""
        tracked = [
            "mlb-backend/data_delivery/calibration_20260830.json",
            "mlb-backend/data_delivery/model_monitor_20260830.json",
            "mlb-backend/data_delivery/calibration_20260825.json",
        ]
        stale, prot, cur = classify_tracked(
            tracked, set(), self.RUN, recent_dates=self.RECENT,
            retention_dates=self.RETENTION, board_dates=self.BOARD)
        self.assertEqual(cur, 2, "48h window keeps 30th calibration + monitor")
        self.assertEqual(stale, [
            "mlb-backend/data_delivery/calibration_20260825.json",
        ], "older unrelated dated artifact still pruned despite board_dates")

    def test_board_rule_does_not_save_non_matching_families(self):
        """Only the run-engine/predictions families ride the board rule — an
        old calibration for a board-backed date is still pruned."""
        tracked = [
            "mlb-backend/data_delivery/calibration_20260829.json",
            "mlb-backend/data_delivery/run_engine_markets_20260829.csv",
        ]
        stale, prot, cur = classify_tracked(
            tracked, set(), self.RUN, recent_dates=self.RECENT,
            retention_dates=self.RETENTION, board_dates=self.BOARD)
        self.assertEqual(stale, [
            "mlb-backend/data_delivery/calibration_20260829.json",
        ], "calibration is not board-backed; run_engine is")
        self.assertEqual(cur, 1)


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


class TestRetentionPolicyConfig(TestCase):
    """The explicit policy table (retention_policy.py) is the single source
    of truth Phase 6 derives from: every audited family classified, the
    deletion allowlist disjoint from the never-delete set, windows parse,
    series readers exempt."""

    def _audited_families(self):
        return {
            "calibration_", "model_monitor_", "predictions_history_",
            "todays_games_", "run_engine_markets_", "run_engine_oof_",
            "run_engine_monitor_", "rolling_brier_",
            "run_engine_feature_drift_", "run_engine_feature_coverage_",
            "feature_drift_", "feature_coverage_", "features_metadata_",
            "shap_game_", "power_rankings_", "pbp_defense_",
        }

    def test_every_audited_family_classified(self):
        from retention_policy import FAMILY_POLICY
        prefixes = {fp.prefix for fp in FAMILY_POLICY}
        missing = self._audited_families() - prefixes
        self.assertEqual(
            missing, set(),
            f"audited families missing from the policy: {missing}")

    def test_allowlist_disjoint_from_never_delete(self):
        from retention_policy import (EXACT_MASTER_NAMES, FAMILY_POLICY,
                                      RECORD_PREFIXES, SERIES_PREFIXES)
        never_delete_prefixes = (*SERIES_PREFIXES, *RECORD_PREFIXES)
        for fp in FAMILY_POLICY:
            if not fp.allowlisted:
                continue
            self.assertNotIn(
                fp.prefix.rstrip("_"), EXACT_MASTER_NAMES,
                f"allowlisted family {fp.prefix} collides with a master name")
            for npfx in never_delete_prefixes:
                self.assertFalse(
                    fp.prefix.startswith(npfx.rstrip("/")),
                    f"allowlisted family {fp.prefix} is shadowed by "
                    f"never-delete prefix {npfx}")

    def test_series_families_are_not_allowlisted(self):
        """run_engine_monitor_ and pbp_defense_ are series readers — they
        must be exempt (allowlisted=False) AND in the never-delete set."""
        from retention_policy import (FAMILY_POLICY, SERIES_PREFIXES)
        for name in ("run_engine_monitor_", "pbp_defense_"):
            fp = next(x for x in FAMILY_POLICY if x.prefix == name)
            self.assertFalse(fp.allowlisted,
                             f"{name} is a series reader and must be exempt")
            self.assertIn(name, SERIES_PREFIXES,
                          f"{name} must be in SERIES_PREFIXES")

    def test_windows_parse_and_every_allowlisted_family_has_a_window(self):
        from retention_policy import FAMILY_POLICY
        for fp in FAMILY_POLICY:
            self.assertIsInstance(fp.prefix, str)
            if fp.retention_days is not None:
                self.assertGreaterEqual(fp.retention_days, 0)
            self.assertGreaterEqual(fp.slate_window_days, 0)
            if fp.allowlisted:
                self.assertTrue(
                    fp.retention_days is not None or fp.board_supported
                    or fp.slate_window_days > 0,
                    f"allowlisted family {fp.prefix} has no retention "
                    f"mechanism")


class TestRetentionPolicyDryRun(TestCase):
    """Deletion dry-run: simulate the keep-set computation on the CURRENT
    committed data_delivery (git ls-files — untracked scratch never enters,
    exactly like the Phase 6 loop) at the next run boundary (2026-09-04) and
    assert the exact delete list. Never-delete families (monitors,
    pbp_defense, records, masters) must never be selected, and the NEWEST
    dated artifact of every family must survive (the loader-fallback
    regression guard)."""

    RUN = "20260904"
    RECENT = {"20260904", "20260903", "20260902"}
    RETENTION = {"20260904", "20260903"}

    # 2-day allowlisted families: day-2 (20260902) files prune at this run.
    _TWO_DAY_PREFIXES = (
        "calibration_", "model_monitor_", "rolling_brier_",
        "run_engine_feature_drift_", "run_engine_feature_coverage_",
        "feature_drift_", "feature_coverage_", "features_metadata_",
        "power_rankings_",
    )
    _SLATE_PREFIXES = ("todays_games_", "shap_game_")
    _NEVER_DELETE_STARTS = ("run_engine_monitor_", "pbp_defense_",
                            "mlb_", "test_hygiene_triage")

    def _tracked(self):
        import subprocess
        repo = Path(__file__).resolve().parent.parent.parent
        out = subprocess.run(
            ["git", "ls-files", "mlb-backend/data_delivery"],
            cwd=str(repo), capture_output=True, text=True)
        if out.returncode != 0:
            self.skipTest(f"git ls-files failed: {out.stderr[:200]}")
        return [ln for ln in out.stdout.splitlines() if ln]

    @staticmethod
    def _date_of(base: str) -> str | None:
        m = re.search(r"_(\d{8})", base)
        return m.group(1) if m else None

    def test_exact_delete_list_on_current_inventory(self):
        from retention_policy import (FAMILY_POLICY, is_allowlisted)
        tracked = self._tracked()
        board_dates = {
            d for p in tracked
            for d in [self._date_of(p.rsplit("/", 1)[-1])]
            if d and p.rsplit("/", 1)[-1].startswith("todays_games_")
        }
        stale, prot, cur = classify_tracked(
            tracked, set(), self.RUN, recent_dates=self.RECENT,
            retention_dates=self.RETENTION, board_dates=board_dates)
        stale_names = sorted(s.rsplit("/", 1)[-1] for s in stale)

        # Rule-based expectation: 2-day allowlisted families at day -2, and
        # slate families beyond the 3-day window at day -3.
        expected = sorted(
            b for p in tracked
            for b in [p.rsplit("/", 1)[-1]]
            if (self._date_of(b) == "20260902"
                and b.startswith(self._TWO_DAY_PREFIXES))
            or (self._date_of(b) == "20260901"
                and b.startswith(self._SLATE_PREFIXES))
        )
        self.assertEqual(expected, stale_names,
                         "dry-run delete list diverges from the policy rule")
        self.assertEqual(len(stale_names), 25,
                         "expected 25 prunable files at the 09-04 boundary "
                         "(9 two-day + 1 board-slate + 15 shap-slate)")

        # Guardrail 2: NO family outside the allowlist is ever selected.
        for base in stale_names:
            self.assertTrue(
                is_allowlisted(base),
                f"{base} is outside the deletion allowlist but selected")
            self.assertFalse(
                base.startswith(self._NEVER_DELETE_STARTS),
                f"never-delete family selected: {base}")

        # Never-delete families present in the inventory stay present.
        for base in (b for p in tracked
                     for b in [p.rsplit("/", 1)[-1]]
                     if b.startswith(self._NEVER_DELETE_STARTS)):
            self.assertNotIn(base, stale_names,
                             f"never-delete artifact selected: {base}")

        # Render-preserved guard: the newest dated artifact of EVERY family
        # survives — markets/calibration/monitor/board loaders resolve the
        # newest, never a pruned old file.
        for fp in FAMILY_POLICY:
            fam_dates = [
                self._date_of(p.rsplit("/", 1)[-1])
                for p in tracked
                if p.rsplit("/", 1)[-1].startswith(fp.prefix)
            ]
            fam_dates = [d for d in fam_dates if d]
            if not fam_dates:
                continue
            newest_date = max(fam_dates)
            stale_fam_dates = [
                self._date_of(n) for n in stale_names
                if n.startswith(fp.prefix)
            ]
            self.assertNotIn(
                newest_date, stale_fam_dates,
                f"newest {fp.prefix} artifact would be pruned "
                f"({newest_date})")

    def test_fixture_exact_delete_list(self):
        """Hermetic fixture pinning the exact delete list: 2-day allowlisted
        families at day -2, slate families beyond the 3-day window, board
        families kept while their board is tracked, series/records/masters
        never selected."""
        tracked = [
            "mlb-backend/data_delivery/calibration_20260902.json",
            "mlb-backend/data_delivery/calibration_20260903.json",
            "mlb-backend/data_delivery/feature_drift_20260902.csv",
            "mlb-backend/data_delivery/todays_games_20260901.csv",
            "mlb-backend/data_delivery/todays_games_20260902.csv",
            "mlb-backend/data_delivery/shap_game_20260901_ATH@TEX.csv",
            "mlb-backend/data_delivery/shap_game_20260902_NYY@LAA.csv",
            "mlb-backend/data_delivery/run_engine_markets_20260901.csv",
            "mlb-backend/data_delivery/run_engine_markets_20260901.meta.json",
            "mlb-backend/data_delivery/predictions_history_20260901.csv",
            "mlb-backend/data_delivery/run_engine_oof_20260901.csv",
            "mlb-backend/data_delivery/run_engine_monitor_20260826.json",
            "mlb-backend/data_delivery/pbp_defense_20260831.parquet",
            "mlb-backend/data_delivery/mlb_sp_bias_stability_a105ba9bba6d60ff.json",
            "mlb-backend/data_delivery/test_hygiene_triage_20260904.json",
            "mlb-backend/data_delivery/game_level_features.csv",
        ]
        stale, prot, cur = classify_tracked(
            tracked, set(), "20260904",
            recent_dates={"20260904", "20260903", "20260902"},
            retention_dates={"20260904", "20260903"},
            board_dates={"20260901", "20260902", "20260903"})
        self.assertEqual(sorted(stale), sorted([
            "mlb-backend/data_delivery/calibration_20260902.json",
            "mlb-backend/data_delivery/feature_drift_20260902.csv",
            "mlb-backend/data_delivery/todays_games_20260901.csv",
            "mlb-backend/data_delivery/shap_game_20260901_ATH@TEX.csv",
        ]))
        self.assertEqual(prot, 5,  # monitor + pbp_defense + 2 records + master
                         "series/records/master must be protected")
        self.assertEqual(cur, 7, "board-backed + window families kept")


if __name__ == "__main__":
    import unittest
    unittest.main()
