"""NFL master_pipeline stale-cleanup predicates (``master_pipeline.py``) —
focused pure-function tests for the board-backed retention rule.

Covered:
- The canonical decided frame (``nfl_game_level_features.csv``, a dateless
  name) is exact-name protected — never pruned, no matter the board dates or
  retention window.
- A dated ``nfl_moneyline_v1_<d>.json`` / ``nfl_feature_v1_<d>.json`` record
  is KEPT when ``<d>`` still renders a board (appears as a distinct game_date
  in the moneyline games[]), even outside the retention window.
- It is PRUNED when ``<d>`` has no board (the reverse direction is allowed).
- Unrelated dated files (monitor/calibration-like) are unaffected by the
  board-backed rule — they stay on the plain retention window.
- ``board_dates_from_records`` unions game_dates across ALL moneyline records
  (a blocked run's record has no games[] and must not drop protection).
- Staged files always win (this run's regenerated artifacts are never stale).

No network — pure classification, temp-dir JSON reads, and one real LOCAL
temp git repo for the post-push summary regression.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from master_pipeline import (  # noqa: E402
    _artifact_date,
    _board_backed_keep,
    _file_sha256,
    _is_protected_name,
    _latest_dated_artifacts,
    _post_push_summary,
    _prune_stale,
    _snapshot_delivery,
    board_dates_from_records,
    classify_stale,
    parse_args,
)

DD = "nfl-backend/data_delivery"
EMPTY: set[str] = set()


class TestCanonicalFrameProtected(unittest.TestCase):
    def test_canonical_frame_never_pruned(self):
        """The dateless canonical decided frame must survive cleanup no matter
        what — empty board dates and an empty retention window included."""
        rel = f"{DD}/nfl_game_level_features.csv"
        self.assertEqual(classify_stale(rel, EMPTY, EMPTY, EMPTY), "protected")
        self.assertTrue(_is_protected_name(rel))

    def test_staged_canonical_frame_still_wins(self):
        """'staged files win' ordering: a regenerated file is never stale even
        when its name is also protected (the ordering intact)."""
        rel = f"{DD}/nfl_game_level_features.csv"
        self.assertEqual(classify_stale(rel, {rel}, EMPTY, EMPTY), "staged")


class TestBoardBackedRecords(unittest.TestCase):
    def test_record_kept_when_date_still_renders_board(self):
        """nfl_moneyline_v1_<d> and nfl_feature_v1_<d> are kept when <d> is a
        distinct game_date in the moneyline games[] — even with an empty
        retention window (this is the MLB regression fix: a board date never
        loses the record it renders)."""
        board = {"20260909", "20260910"}
        for name in ("nfl_moneyline_v1_20260909.json", "nfl_feature_v1_20260909.json"):
            rel = f"{DD}/{name}"
            self.assertEqual(classify_stale(rel, EMPTY, board, EMPTY), "keep",
                             name)
            self.assertTrue(_board_backed_keep(rel, board))

    def test_calibration_and_history_families_board_backed(self):
        """The MLB-equivalent Part-A siblings (nfl_calibration_* /
        nfl_predictions_history_*) are board-backed exactly like the moneyline
        / feature records — a board date never loses the curve/history that
        renders it."""
        board = {"20260909"}
        for name in ("nfl_calibration_20260909.json",
                     "nfl_predictions_history_20260909.csv"):
            rel = f"{DD}/{name}"
            self.assertEqual(classify_stale(rel, EMPTY, board, EMPTY), "keep")
            self.assertTrue(_board_backed_keep(rel, board))
        # ... and pruned when their date has no board
        rel = f"{DD}/nfl_calibration_20260830.json"
        self.assertEqual(classify_stale(rel, EMPTY, board, EMPTY), "stale")
        self.assertFalse(_board_backed_keep(rel, board))

    def test_record_pruned_when_date_has_no_board(self):
        """A record whose filename date has no board is stale (the reverse
        direction is explicitly allowed), unless the retention window keeps it."""
        rel = f"{DD}/nfl_moneyline_v1_20260830.json"
        board = {"20260909"}          # board lives on 09-09, not 08-30
        retention = {"20260901"}      # 08-30 outside the window
        self.assertEqual(classify_stale(rel, EMPTY, board, retention), "stale")
        self.assertFalse(_board_backed_keep(rel, board))

    def test_record_kept_by_retention_when_no_board(self):
        """Without a board date the plain retention window still saves a
        recent record — the board-backed rule is ADDITIVE, not replacing."""
        rel = f"{DD}/nfl_moneyline_v1_20260830.json"
        self.assertEqual(classify_stale(rel, EMPTY, {"20260909"},
                                        {"20260830"}), "keep")


class TestUnrelatedDatedFiles(unittest.TestCase):
    def test_monitor_like_files_unaffected_by_board_rule(self):
        """A dated monitor/calibration-like file is never saved by the
        board-backed rule — it stays on the plain retention window."""
        rel = f"{DD}/nfl_model_monitor_20260909.json"
        board = {"20260909"}                     # board exists, but rule is
        self.assertFalse(_board_backed_keep(rel, board))  # scoped to records
        self.assertEqual(classify_stale(rel, EMPTY, board, EMPTY), "stale")
        # ... and the retention window still applies to it normally
        self.assertEqual(classify_stale(rel, EMPTY, board, {"20260909"}), "keep")

    def test_other_dated_artifact_within_retention(self):
        rel = f"{DD}/nfl_pbp_chunks_20260830.parquet"
        self.assertEqual(classify_stale(rel, EMPTY, {"20260909"},
                                        {"20260830"}), "keep")
        self.assertEqual(classify_stale(rel, EMPTY, {"20260909"},
                                        {"20260901"}), "stale")


class TestBoardDateSet(unittest.TestCase):
    def test_union_across_all_records(self):
        """S = distinct game_dates across EVERY moneyline record's games[] —
        a blocked (no games[]) record must not drop protection for the still-
        live boards its predecessors wrote."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "nfl_moneyline_v1_20260830.json").write_text(json.dumps({
                "verdict": {"adopt": True},
                "games": [{"game_date": "2026-09-09"}, {"game_date": "2026-09-10"}],
            }), encoding="utf-8")
            # a later blocked run: no games[] -> contributes nothing
            (d / "nfl_moneyline_v1_20260901.json").write_text(json.dumps({
                "verdict": {"adopt": False},
                "predictions": {"status": "blocked (not adopted)"},
            }), encoding="utf-8")
            self.assertEqual(board_dates_from_records(d),
                             {"20260909", "20260910"})

    def test_empty_dir_yields_empty_board(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(board_dates_from_records(Path(td)), set())

    def test_malformed_record_never_crashes(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "nfl_moneyline_v1_20260830.json").write_text(
                "{ not json", encoding="utf-8")
            self.assertEqual(board_dates_from_records(d), set())


class TestArtifactDate(unittest.TestCase):
    def test_date_extraction(self):
        self.assertEqual(_artifact_date(f"{DD}/nfl_moneyline_v1_20260830.json"),
                         "20260830")
        self.assertEqual(_artifact_date(f"{DD}/nfl_feature_v1_20260909.json"),
                         "20260909")
        self.assertIsNone(_artifact_date(f"{DD}/nfl_game_level_features.csv"))
        self.assertIsNone(_artifact_date(f"{DD}/models/bundle.joblib"))


class TestSeasonWindow(unittest.TestCase):
    """parse_args -> .window: the data/feature season window from
    --start-season / --end-season or NFL_START_SEASON / NFL_END_SEASON.
    Default (no overrides) is None -> each module keeps its full range."""

    def _window(self, argv, env=None):
        import os
        saved = os.environ.copy()
        env = env or {}
        for k, v in env.items():
            os.environ[k] = v
        try:
            return parse_args(argv).window
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def test_default_is_none(self):
        """No --start/--end-season -> window is None (unchanged default)."""
        self.assertIsNone(self._window([]))
        self.assertIsNone(self._window(["--slate-season", "2026"]))

    def test_both_cli(self):
        self.assertEqual(self._window(["--start-season", "2021",
                                       "--end-season", "2023"]),
                         [2021, 2022, 2023])

    def test_only_start_closes_to_last(self):
        self.assertEqual(self._window(["--start-season", "2022"]),
                         [2022, 2023, 2024, 2025])

    def test_only_end_closes_to_first(self):
        self.assertEqual(self._window(["--end-season", "2022"]),
                         [2019, 2020, 2021, 2022])

    def test_env_vars(self):
        self.assertEqual(self._window([], {"NFL_START_SEASON": "2019",
                                           "NFL_END_SEASON": "2020"}),
                         [2019, 2020])

    def test_env_takes_back_seat_to_cli(self):
        self.assertEqual(self._window(["--start-season", "2022",
                                       "--end-season", "2023"],
                                      {"NFL_START_SEASON": "2019",
                                       "NFL_END_SEASON": "2020"}),
                         [2022, 2023])

    def test_invalid_order_raises(self):
        with self.assertRaises(ValueError):
            parse_args(["--start-season", "2024", "--end-season", "2022"])


class TestContentDiffStaging(unittest.TestCase):
    """Phase 4 decides stale-vs-changed by CONTENT hash, not mtime (a fresh
    clone stamps files with the clone time, so mtime would silently skip
    re-generated artifacts on same-date re-runs)."""

    def test_changed_content_differs_from_committed(self):
        with tempfile.TemporaryDirectory() as td:
            committed = Path(td) / "nfl_moneyline_v1_20260831.json"
            committed.write_text('{"games":[],"verdict":{"adopt":true}}')
            regenerated = Path(td) / "nfl_moneyline_v1_20260831_REGENERATED.json"
            regenerated.write_text('{"games":["272 games"],"verdict":{"adopt":true}}')
            self.assertNotEqual(_file_sha256(committed), _file_sha256(regenerated))

    def test_snapshot_maps_rel_to_sha(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "a.json").write_text('{"x":1}')
            (d / "b.csv").write_text("a,b\n1,2")
            snap = _snapshot_delivery(d)
            self.assertEqual(set(snap), {"a.json", "b.csv"})
            self.assertNotEqual(snap["a.json"], snap["b.csv"])

    def test_unchanged_matches_committed(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "f.json"
            p.write_text('{"games":[]}')
            snap = _snapshot_delivery(Path(td))
            self.assertEqual(_file_sha256(p), snap["f.json"])  # identical -> stale

class TestPruneNeverDeletesTracked(unittest.TestCase):
    """2026-09-01 regression: Phase 5 gc'd the COMMITTED win_pct_diff evidence
    record (git rm + commit + push, 88d6c8f). _prune_stale must skip ANY
    tracked file with a loud-warning hit and delete only stale UNTRACKED
    files, returning the list of actually-deleted paths for the record."""

    def _mkfile(self, d: Path, name: str, content: str = "{}") -> Path:
        p = d / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_tracked_stale_skipped_untracked_stale_removed(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            # Committed files: the evidence record (dateless -> would be
            # stale) and the canonical frame (protected).
            evidence = self._mkfile(
                d, "nfl_feature_winpct_ablation_e4aee120a4b8.json",
                '{"verdict": "KEEP"}')
            frame = self._mkfile(d, "nfl_game_level_features.csv", "a,b\n1,2")
            # Untracked strays: a convention-uncommitted ablation record and
            # an old pbp chunk outside the retention window.
            stray = self._mkfile(d, "nfl_tier3_ablation_e4aee120a4b8.json",
                                 '{"x": 1}')
            old = self._mkfile(d, "nfl_pbp_chunks_20260830.parquet", "pq")
            tracked = {f"{DD}/nfl_feature_winpct_ablation_e4aee120a4b8.json",
                       f"{DD}/nfl_game_level_features.csv"}
            out = _prune_stale(d, EMPTY, EMPTY, {"20260901"}, tracked)
            # committed files untouched ...
            self.assertTrue(evidence.exists(), "committed record was deleted!")
            self.assertTrue(frame.exists())
            # ... untracked stale files removed and reported
            self.assertFalse(stray.exists())
            self.assertFalse(old.exists())
            self.assertEqual(
                out["deleted"],
                [f"{DD}/nfl_pbp_chunks_20260830.parquet",
                 f"{DD}/nfl_tier3_ablation_e4aee120a4b8.json"])
            # the guard hit is surfaced for the LOUD warning
            self.assertEqual(
                out["stale_tracked"],
                [f"{DD}/nfl_feature_winpct_ablation_e4aee120a4b8.json"])
            self.assertEqual(out["kept_protected"], 1)  # canonical frame

    def test_staged_never_deleted_even_if_untracked(self):
        """This run's staged files always win (classify_stale ordering)."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            staged_file = self._mkfile(d, "nfl_moneyline_v1_20260901.json",
                                       '{"games": []}')
            self._mkfile(d, "nfl_model_monitor_20260801.json", "{}")
            staged = {f"{DD}/nfl_moneyline_v1_20260901.json"}
            out = _prune_stale(d, staged, EMPTY, {"20260901"}, set())
            self.assertTrue(staged_file.exists())
            self.assertEqual(out["deleted"],
                             [f"{DD}/nfl_model_monitor_20260801.json"])

    def test_untracked_within_retention_kept(self):
        """Untracked but within the retention window: kept, not deleted."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._mkfile(d, "nfl_model_monitor_20260831.json", "{}")
            self._mkfile(d, "nfl_moneyline_v1_20260909.json", "{}")
            board = {"20260909"}   # moneyline record date still renders a board
            out = _prune_stale(d, EMPTY, board, {"20260831"}, set())
            self.assertTrue((d / "nfl_model_monitor_20260831.json").exists())
            self.assertTrue((d / "nfl_moneyline_v1_20260909.json").exists())
            self.assertEqual(out["deleted"], [])


class TestLatestDatedArtifacts(unittest.TestCase):
    def test_newest_three_with_prefix_and_date_filter(self):
        prefix = f"{DD}/"
        lines = [
            f"{prefix}nfl_game_level_features.csv",      # undated -> filtered
            f"{prefix}nfl_moneyline_v1_20260830.json",
            f"{prefix}nfl_moneyline_v1_20260831.json",
            f"{prefix}nfl_moneyline_v1_20260901.json",
            f"{prefix}nfl_power_rankings_20260901.csv",
            "mlb-backend/data_delivery/x_20260901.json",  # wrong prefix
        ]
        self.assertEqual(
            _latest_dated_artifacts(lines, prefix),
            [f"{prefix}nfl_moneyline_v1_20260831.json",
             f"{prefix}nfl_moneyline_v1_20260901.json",
             f"{prefix}nfl_power_rankings_20260901.csv"])

    def test_no_date_matches_returns_empty(self):
        self.assertEqual(
            _latest_dated_artifacts([f"{DD}/nfl_game_level_features.csv"],
                                    f"{DD}/"),
            [])


class TestPostPushSummaryState(unittest.TestCase):
    """The post-run summary must reflect the PUSHED state. Phase 4 pushes
    from a /tmp sync clone, so the working checkout's HEAD / origin ref stay
    pre-push (2026-09-01 regression: "Repo HEAD after run: 81aea53" while
    origin had advanced). _post_push_summary reads the passed repo's HEAD —
    here a real local git repo, where a second "push" commit must be what
    the summary reports."""

    def _init_repo(self, td: Path):
        import git
        repo = git.Repo.init(str(td / "repo"))
        with repo.config_writer() as cw:
            cw.set_value("user", "name", "Test")
            cw.set_value("user", "email", "test@example.com")
        return repo

    def _commit_artifact(self, repo, name: str, msg: str) -> None:
        d = Path(repo.working_tree_dir) / "nfl-backend" / "data_delivery"
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text('{"games": []}', encoding="utf-8")
        repo.git.add(f"nfl-backend/data_delivery/{name}")
        repo.index.commit(msg)

    def test_summary_reflects_latest_pushed_commit(self):
        with tempfile.TemporaryDirectory() as td:
            repo = self._init_repo(Path(td))
            try:
                self._commit_artifact(repo, "nfl_moneyline_v1_20260831.json",
                                      "run A")
                s1 = _post_push_summary(repo, f"{DD}/")
                self.assertIn("run A", s1["head"])
                # next push: a new commit lands -> summary must follow it
                self._commit_artifact(repo, "nfl_moneyline_v1_20260901.json",
                                      "run B")
                s2 = _post_push_summary(repo, f"{DD}/")
                self.assertIn("run B", s2["head"])
                self.assertNotIn("run B", s1["head"])
                self.assertIn(f"{DD}/nfl_moneyline_v1_20260901.json",
                              s2["latest_dated"])
                self.assertNotIn(f"{DD}/nfl_moneyline_v1_20260901.json",
                                 s1["latest_dated"])
            finally:
                repo.close()  # release file locks so the temp dir can go


class TestIncumbentBundleAvailability(unittest.TestCase):
    """The within-run incumbent gate (nfl_moneyline.adopt_decision) loads
    data_delivery/models/ensemble_latest.joblib as its baseline. For that to
    work on a fresh Kaggle clone — where the gate must run in within-run
    incumbent mode immediately, not advisory-once — the bundle must be (b)
    committed/pushed like MLB's, i.e. NOT gitignored, and (c) protected from
    Phase-5 stale cleanup."""

    def test_bundle_not_gitignored(self):
        import subprocess
        root = Path(__file__).resolve().parents[2]   # repo root
        rel = "nfl-backend/data_delivery/models/ensemble_latest.joblib"
        r = subprocess.run(["git", "check-ignore", "-q", rel],
                           cwd=root, capture_output=True)
        self.assertNotEqual(
            r.returncode, 0,
            f"{rel} must NOT be gitignored — the within-run incumbent gate "
            f"needs it on the remote (MLB mechanism: tracked beats gitignore)")

    def test_bundle_protected_from_stale_cleanup(self):
        from master_pipeline import _PROTECTED_DELIVERY_PREFIXES
        self.assertIn("models/", _PROTECTED_DELIVERY_PREFIXES)
        self.assertTrue(
            _is_protected_name(
                "nfl-backend/data_delivery/models/ensemble_latest.joblib"))
        # unrelated dated artifacts stay unprotected (plain retention)
        self.assertFalse(
            _is_protected_name(
                "nfl-backend/data_delivery/nfl_moneyline_v1_20260830.json"))


class TestRunEngineAndResearchProtection(unittest.TestCase):
    """Daily-emission wiring (2026-09-04): the run-engine dated families and
    the pinned research records a daily run depends on are PREFIX-PROTECTED
    — never swept, not even as stale-untracked copies (the tracked-file
    guard already protects committed copies; prefix-protection extends the
    same guarantee to untracked ones, so a sweep can never remove the last
    good dated store or a research record). Plain monitor-like dated files
    stay on the retention window (rule scoping intact)."""
    FAMILIES = [
        "nfl_run_engine_markets_20260830.csv",
        "nfl_run_engine_markets_20260830.meta.json",
        "nfl_run_engine_monitor_20260830.json",
        "nfl_slate_serve_20260830.json",
        "nfl_era_3e8c8a510f04.json",
        "nfl_market_3e8c8a510f04.json",
        "nfl_adoption_decision_3e8c8a510f04.json",
        # Run-engine drift/coverage emitters (diagnostics wiring, 2026-09-04)
        "run_engine_feature_drift_20260830.csv",
        "run_engine_feature_coverage_20260830.csv",
        # Dateless decision/diagnostic records (cb4036f, 2026-09-05): the
        # date-gate can never save them (no _YYYYMMDD), so like the other
        # nfl_* record families they get targeted prefix protection.
        "nfl_run_engine_diagnostics_v2_3e8c8a510f04.json",
        "nfl_markets_fit_panel_parity_3e8c8a510f04.json",
        # Binary calibration decision record (2026-09-05)
        "nfl_binary_calibration_3e8c8a510f04.json",
    ]

    def test_run_engine_and_research_families_never_stale(self):
        """Stale-dated, no board, empty retention — still protected."""
        for name in self.FAMILIES:
            rel = f"{DD}/{name}"
            self.assertEqual(classify_stale(rel, EMPTY, EMPTY, EMPTY),
                             "protected", name)
            self.assertTrue(_is_protected_name(rel), name)

    def test_untracked_sweep_keeps_families_and_removes_plain_stale(self):
        """A real sweep keeps the (untracked, stale-dated) run-engine /
        research files and still removes a plain stale monitor file — the
        protected set is additive, not a blanket no-delete."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for name in self.FAMILIES:
                (d / name).write_text("{}", encoding="utf-8")
            plain = d / "nfl_model_monitor_20260801.json"
            plain.write_text("{}", encoding="utf-8")
            out = _prune_stale(d, EMPTY, EMPTY, {"20260901"}, set())
            for name in self.FAMILIES:
                self.assertTrue((d / name).exists(), f"{name} was swept!")
            self.assertFalse(plain.exists())
            self.assertEqual(
                out["deleted"],
                [f"{DD}/nfl_model_monitor_20260801.json"])

    def test_diagnostics_v2_records_never_stale_and_scope_exact(self):
        """The cb4036f decision records classify PROTECTED (not would-be-
        stale on the tracked guard) — and the protection is TARGETED: the
        two record prefixes are the ONLY additions, no broad ``nfl_``
        prefix (the dated moneyline/feature families must keep riding the
        board-backed date-gate)."""
        from master_pipeline import _PROTECTED_DELIVERY_PREFIXES as P
        self.assertIn("nfl_run_engine_diagnostics_", P)
        self.assertIn("nfl_markets_fit_panel_parity_", P)
        self.assertNotIn("nfl_", P)
        for name in ("nfl_run_engine_diagnostics_v2_3e8c8a510f04.json",
                     "nfl_markets_fit_panel_parity_3e8c8a510f04.json"):
            rel = f"{DD}/{name}"
            self.assertTrue(_is_protected_name(rel), name)
            self.assertEqual(classify_stale(rel, EMPTY, EMPTY, EMPTY),
                             "protected", name)
        # Dated moneyline/feature families are untouched by the addition.
        self.assertFalse(
            _is_protected_name(f"{DD}/nfl_moneyline_v1_20260830.json"))
        self.assertFalse(
            _is_protected_name(f"{DD}/nfl_feature_v1_20260830.json"))

    def test_binary_calibration_record_prefix_scope_exact(self):
        """The binary-calibration decision record (2026-09-05) is prefix-
        protected like its nfl_* record siblings — never stale, and the
        protection stays TARGETED (no broad ``nfl_``; the dated
        moneyline/feature families keep the board-backed date-gate)."""
        from master_pipeline import _PROTECTED_DELIVERY_PREFIXES as P
        self.assertIn("nfl_binary_calibration_", P)
        self.assertNotIn("nfl_", P)
        rel = f"{DD}/nfl_binary_calibration_3e8c8a510f04.json"
        self.assertTrue(_is_protected_name(rel))
        self.assertEqual(classify_stale(rel, EMPTY, EMPTY, EMPTY), "protected")
        # Dated moneyline/feature families are untouched by the addition.
        self.assertFalse(
            _is_protected_name(f"{DD}/nfl_moneyline_v1_20260830.json"))
        self.assertFalse(
            _is_protected_name(f"{DD}/nfl_feature_v1_20260830.json"))

    def test_moneyline_records_still_board_backed_not_prefix_protected(self):
        """Rule scope intact: moneyline/feature records keep the board-backed
        retention rule (not prefix protection)."""
        rel = f"{DD}/nfl_moneyline_v1_20260830.json"
        self.assertFalse(_is_protected_name(rel))
        self.assertEqual(classify_stale(rel, EMPTY, {"20260830"}, EMPTY),
                         "keep")   # its date still renders a board


class TestDailyEmissionWiring(unittest.TestCase):
    """The daily pipeline phase and the standalone/backfill CLI share ONE
    emission core (``run_nfl_markets_backfill.run_daily_markets``) — no
    forked schema (spec Step 1). Source-level pins, like the slate tests."""
    BACKEND = Path(__file__).resolve().parent

    def _source(self, name: str) -> str:
        return (self.BACKEND / name).read_text(encoding="utf-8")

    def test_master_pipeline_phase3b_calls_the_shared_core(self):
        src = self._source("master_pipeline.py")
        # phase3 ends by invoking the Phase-3b sub-phase...
        self.assertIn("# Phase 3b — run-engine markets emission", src)
        self.assertIn("_phase3b(args)", src)
        # ...which imports and calls run_nfl_markets_backfill's shared core.
        tail = src[src.index("def _phase3b"):]
        self.assertIn(
            "from run_nfl_markets_backfill import run_daily_markets", tail)
        self.assertIn("res = run_daily_markets(out_dir=out_dir)", tail)

    def test_backfill_cli_delegates_to_the_same_core(self):
        src = self._source("run_nfl_markets_backfill.py")
        self.assertIn("def run_daily_markets(out_dir: Path | None = None,",
                      src)
        cli = src[src.index("def main("):]
        self.assertIn("run_daily_markets(out_dir=args.out_dir, "
                      "no_record=args.no_record)", cli)

    def test_phase_order_moneyline_then_emission_then_sync(self):
        """Insertion point: after the moneyline phase, before sync/cleanup."""
        src = self._source("master_pipeline.py")
        i_phase3 = src.index('_banner("PHASE 3",')
        i_phase3b_call = src.index("_phase3b(args)")
        i_phase4 = src.index("def phase4(")
        i_phase5 = src.index("def phase5(")
        self.assertLess(i_phase3, i_phase3b_call)
        self.assertLess(i_phase3b_call, i_phase4)
        self.assertLess(i_phase4, i_phase5)


class TestForceAddIgnoredDatedCsv(unittest.TestCase):
    """Judgment call 3: the dated markets CSV is gitignored
    (nfl-backend/.gitignore: ``data_delivery/*.csv``). Phase 4 must commit a
    BRAND-NEW dated .csv (never committed before its first push) with
    ``git add -f`` semantics — gitpython's ``index.add(..., force=True)``.
    Pin the explicit call in source AND the behavior in a scratch repo so a
    future gitpython change cannot silently skip the dated store."""

    def test_phase4_add_is_explicit_force(self):
        src = (Path(__file__).resolve().parent / "master_pipeline.py")\
            .read_text(encoding="utf-8")
        self.assertIn("repo.index.add(staged, force=True)", src)

    def test_new_ignored_csv_committed_by_force_add(self):
        import git
        with tempfile.TemporaryDirectory() as td:
            repo = git.Repo.init(str(Path(td) / "repo"))
            with repo.config_writer() as cw:
                cw.set_value("user", "name", "Test")
                cw.set_value("user", "email", "test@example.com")
            try:
                root = Path(repo.working_tree_dir)
                (root / ".gitignore").write_text(
                    "data_delivery/*.csv\n", encoding="utf-8")
                repo.index.add([".gitignore"])
                repo.index.commit("init ignore")
                dd = root / "data_delivery"
                dd.mkdir()
                (dd / "nfl_run_engine_markets_20260904.csv").write_text(
                    "a,b\n1,2\n", encoding="utf-8")
                rel = "data_delivery/nfl_run_engine_markets_20260904.csv"
                repo.index.add([rel], force=True)  # phase-4 call pattern
                repo.index.commit("stage dated csv")
                self.assertIn(rel,
                              set(repo.git.ls_files().splitlines()))
            finally:
                repo.close()  # release file locks so the temp dir can go


if __name__ == "__main__":
    unittest.main()
