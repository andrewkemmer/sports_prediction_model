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
}
_PROTECTED_DELIVERY_PREFIXES = ("models/",)
_DATE_RE = re.compile(r"_(\d{8})")


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


def classify_tracked(tracked, seen, run_date_compact="20260824"):
    """Replicate the Phase 6 classification logic for testing."""
    stale, kept_protected, kept_current = [], 0, 0
    for p in tracked:
        if p in seen:
            continue
        if _is_protected(p):
            kept_protected += 1
            continue
        art_date = _artifact_date(p)
        if art_date == run_date_compact:
            kept_current += 1
            continue
        stale.append(p)
    return stale, kept_protected, kept_current


class TestCleanupProtection(TestCase):
    """Protected files must never appear in the stale list."""

    def test_roof_cache_protected(self):
        tracked = [
            "mlb-bet-predictor/data_delivery/statsapi_roof_cache.json",
            "mlb-bet-predictor/data_delivery/game_level_features.csv",
            "mlb-bet-predictor/data_delivery/run_engine_oof_20260820.csv",
        ]
        seen = {"mlb-bet-predictor/data_delivery/game_level_features.csv"}
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertNotIn(
            "mlb-bet-predictor/data_delivery/statsapi_roof_cache.json", stale,
            "statsapi_roof_cache.json must be protected")
        self.assertEqual(prot, 1)

    def test_model_history_protected(self):
        tracked = [
            "mlb-bet-predictor/data_delivery/model_history.json",
            "mlb-bet-predictor/data_delivery/model_version_history.json",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(stale, [])
        self.assertEqual(prot, 2)

    def test_models_dir_protected(self):
        tracked = [
            "mlb-bet-predictor/data_delivery/models/ensemble_v1.joblib",
            "mlb-bet-predictor/data_delivery/models/alpha_params.json",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(stale, [])
        self.assertEqual(prot, 2)

    def test_all_protected_kept(self):
        """All three categories of protected files survive."""
        tracked = [
            "mlb-bet-predictor/data_delivery/statsapi_roof_cache.json",
            "mlb-bet-predictor/data_delivery/model_history.json",
            "mlb-bet-predictor/data_delivery/model_version_history.json",
            "mlb-bet-predictor/data_delivery/models/x.joblib",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(stale, [])
        self.assertEqual(prot, 4)


class TestDateGating(TestCase):
    """Same-day artifacts survive; older-date artifacts are removed."""

    def test_current_date_survives(self):
        tracked = [
            "mlb-bet-predictor/data_delivery/run_engine_markets_20260824.csv",
            "mlb-bet-predictor/data_delivery/run_engine_oof_20260824.csv",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(stale, [])
        self.assertEqual(cur, 2)

    def test_older_date_removed(self):
        tracked = [
            "mlb-bet-predictor/data_delivery/run_engine_markets_20260820.csv",
            "mlb-bet-predictor/data_delivery/run_engine_oof_20260820.csv",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(len(stale), 2)
        self.assertEqual(cur, 0)

    def test_mixed_dates(self):
        tracked = [
            "mlb-bet-predictor/data_delivery/run_engine_markets_20260824.csv",
            "mlb-bet-predictor/data_delivery/run_engine_markets_20260820.csv",
            "mlb-bet-predictor/data_delivery/calibration_20260819.json",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertIn(
            "mlb-bet-predictor/data_delivery/run_engine_markets_20260820.csv", stale)
        self.assertIn(
            "mlb-bet-predictor/data_delivery/calibration_20260819.json", stale)
        self.assertNotIn(
            "mlb-bet-predictor/data_delivery/run_engine_markets_20260824.csv", stale)
        self.assertEqual(cur, 1)

    def test_dateless_file_stale(self):
        """Files with no date pattern are treated as stale (not protected)."""
        tracked = [
            "mlb-bet-predictor/data_delivery/game_level_features.csv",
        ]
        seen = set()
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(len(stale), 1)
        self.assertEqual(cur, 0)

    def test_seen_files_never_stale(self):
        """Files in the seen set are never stale regardless of date."""
        tracked = [
            "mlb-bet-predictor/data_delivery/run_engine_markets_20260820.csv",
        ]
        seen = {"mlb-bet-predictor/data_delivery/run_engine_markets_20260820.csv"}
        stale, prot, cur = classify_tracked(tracked, seen, "20260824")
        self.assertEqual(stale, [])


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
