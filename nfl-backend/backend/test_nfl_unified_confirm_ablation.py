"""Tests for run_nfl_unified_confirm_ablation: cache-key stability, arm
geometry, and verdict-pair wiring.

Everything the harness orchestrates (run_walk_forward, tolerance_verdict,
the raw/xgb-reg runners) is tested by its origin harnesses; these tests pin
the ORCHESTRATION invariants that would otherwise fail silently:
  - _cache_key must be content-stable across processes (never Python's
    randomized builtin hash) and sensitive to column-set changes.
  - build_arms must produce the documented geometry on a carrier frame.
  - _assemble's verdict pairs must reference defined arms and every pair
    must land in the record (a wiring typo currently prints SKIPPED
    instead of failing — this pins it).
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import run_nfl_unified_confirm_ablation as hac  # noqa: E402
from run_nfl_unified_confirm_ablation import (_assemble, _cache_key,
                                              build_arms)  # noqa: E402

VERDICT_PAIRS = [
    ("WITHOUT_YPP", "WITH_12"),
    ("WITHOUT_BOTH", "WITH_QBEPA"),
    ("RAW_ADDED", "C0"),
    ("ROSTER_13", "WITHOUT_13"),
    ("ROSTER", "WITH_12"),
    ("QB", "WITH_12"),
    ("C1", "C0_REG"),
    ("T1_WITH", "T1_WITHOUT"),
    ("T1_WITH_ADMITTED", "T1_WITHOUT"),
    ("T1_WITH_SUBSET", "T1_WITHOUT"),
    ("T1_TIER1_ONLY", "T1_WITHOUT"),
]


def _all_cols() -> set[str]:
    """Union of every candidate column across all arms. With all candidates
    present, `_only` never filters, so the geometry is deterministic."""
    from nfl_features import TIER3_ROSTER_FEATURES
    from nfl_raw_columns import RAW_PER_SIDE_COLS
    from nfl_tier4 import TIER4_OPPADJ_FEATURES, TIER4_QB_FEATURES
    from run_feature_winpct_ablation import DEPLOYED_12
    from run_tier1_ablation import (TIER1_ADMITTED, TIER1_FEATURES,
                                    TIER1_SUBSET, WITHOUT_FEATURES)
    from run_tier2_ablation import VENUE_3_FEATURES

    cols = (set(DEPLOYED_12) | set(RAW_PER_SIDE_COLS)
            | set(TIER3_ROSTER_FEATURES) | set(TIER4_OPPADJ_FEATURES)
            | set(TIER4_QB_FEATURES) | set(TIER1_FEATURES)
            | set(TIER1_ADMITTED) | set(TIER1_SUBSET)
            | set(WITHOUT_FEATURES) | set(VENUE_3_FEATURES))
    return cols


def _carrier() -> pd.DataFrame:
    return pd.DataFrame({c: [0.0] for c in sorted(_all_cols())})


def _arms():
    return build_arms(_carrier())


class TestCacheKey:
    def test_content_stable_across_processes(self):
        cols = ["a", "b", "c"]
        k1 = _cache_key("plain", cols)
        assert k1 == _cache_key("plain", cols)
        sub_run = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'nfl-backend/backend'); "
             "from run_nfl_unified_confirm_ablation import _cache_key; "
             f"print(_cache_key('plain', {cols!r}))"],
            cwd=ROOT, capture_output=True, text=True, check=True)
        assert sub_run.stdout.strip() == k1, (
            "cache key differs in a fresh interpreter — Python's "
            "randomized builtin hash must never be used")

    def test_sensitive_to_runner_and_cols(self):
        base = ["a", "b", "c"]
        assert _cache_key("plain", base) != _cache_key("masked", base)
        assert _cache_key("plain", base) != _cache_key("plain", ["a", "b"])
        # order-insensitive: same set, different order -> same key
        assert _cache_key("plain", ["c", "a", "b"]) == _cache_key("plain", base)

    def test_logistic_cols_differentiates(self):
        cols = ["a", "b", "c"]
        assert (_cache_key("masked", cols, logistic_cols=cols) !=
                _cache_key("masked", cols, logistic_cols=["a", "b"]))


class TestArmGeometry:
    """Pin the documented column counts from the harness docstring:
    WITH_12=12, RAW_ADDED=26, ROSTER=14, ROSTER_13=15, QB=13, C1=15,
    T1_WITH=19, T1_WITH_ADMITTED=17, T1_WITH_SUBSET=13, T1_TIER1_ONLY=7,
    base pools 12 (served) / 13 (hist. tier-3) / 10 (tier-1 without)."""

    @pytest.fixture(scope="class")
    def arms(self):
        return _arms()

    def test_documented_geometries(self, arms):
        sizes = {k: len(a["cols"]) for k, a in arms.items()}
        assert sizes["WITH_12"] == 12
        assert sizes["WITHOUT_YPP"] == 11
        assert sizes["WITH_QBEPA"] == 13
        assert sizes["WITHOUT_BOTH"] == 11          # 13 minus ypp/qbepa twins
        assert sizes["C0"] == 12 and sizes["RAW_ADDED"] == 26
        assert sizes["WITHOUT_13"] == 13
        assert sizes["ROSTER_13"] == 15 and sizes["ROSTER"] == 14
        assert sizes["QB"] == 13                    # base12 + 1 conditional
        assert sizes["C0_REG"] == 12 and sizes["C1"] == 15
        assert sizes["T1_WITHOUT"] == 10
        assert sizes["T1_WITH"] == 19
        assert sizes["T1_WITH_ADMITTED"] == 17
        assert sizes["T1_WITH_SUBSET"] == 13
        assert sizes["T1_TIER1_ONLY"] == 7

    def test_without_both_is_qbepa_minus_twins(self, arms):
        dropped = (set(arms["WITH_QBEPA"]["cols"])
                   - set(arms["WITHOUT_BOTH"]["cols"]))
        assert dropped == {"ewm_ypp_diff", "ewm_qb_epa_play_diff"}

    def test_shared_column_sets_collapse_to_one_run(self, arms):
        """WITHOUT_YPP and WITHOUT_BOTH share one measurement (identical
        11-col arms) — the record's verdicts differ only via baseline."""
        k1 = _cache_key("plain", arms["WITHOUT_YPP"]["cols"])
        k2 = _cache_key("plain", arms["WITHOUT_BOTH"]["cols"])
        assert k1 == k2
        # WITH_12 == C0 share the 12-pool columns; runners differ
        assert _cache_key("plain", arms["WITH_12"]["cols"]) == \
            _cache_key("plain", arms["C0"]["cols"])
        assert _cache_key("reg", arms["C0_REG"]["cols"]) != \
            _cache_key("plain", arms["WITH_12"]["cols"])

    def test_every_baseline_is_a_defined_arm(self, arms):
        for key, a in arms.items():
            if a["baseline"] is not None:
                assert a["baseline"] in arms, (
                    f"{key} baseline {a['baseline']!r} is not an arm")


class TestPairWiring:
    def test_all_pairs_reference_defined_arms(self):
        arms = _arms()
        for cand, base in VERDICT_PAIRS:
            assert cand in arms and base in arms
            assert cand != base

    def test_record_verdicts_cover_every_pair(self, tmp_path):
        """An assembled record's verdicts must cover every pair — guards
        against adding an arm without wiring its verdict."""
        arms = _arms()
        cache = {"frame_sha256": "test", "runs": {}}
        for key, a in arms.items():
            ck = _cache_key(a["runner"], a["cols"], a["logistic_cols"])
            cache["runs"][ck] = {
                "runner": a["runner"], "cols": a["cols"],
                "logistic_cols": a["logistic_cols"],
                "pooled_model_platt": {"logloss": 0.6, "auc": 0.7,
                                       "ece": 0.03},
                "sealed_model_platt": {"logloss": 0.6, "auc": 0.7,
                                       "ece": 0.05},
                "members": {}, "members_sealed": {},
            }
        cache_path = tmp_path / "c.json"
        cache_path.write_text(json.dumps(cache, indent=2))

        orig_dir = hac.DATA_DELIVERY_DIR
        out_dir = (tmp_path / "out")
        out_dir.mkdir(parents=True, exist_ok=True)
        hac.DATA_DELIVERY_DIR = out_dir
        try:
            assert hac._assemble(cache_path, write_record=True) == 0
        finally:
            hac.DATA_DELIVERY_DIR = orig_dir

        rec = json.loads(
            (out_dir / f"nfl_unified_confirm_test.json").read_text())
        assert set(rec["verdicts"]) == {c for c, _ in VERDICT_PAIRS}


def test_every_arm_participates_in_a_pair():
    """Every arm is either a verdict candidate or a baseline — an arm not
    wired into any pair would silently never get a verdict."""
    participants = {c for c, _ in VERDICT_PAIRS} | {b for _, b in VERDICT_PAIRS}
    assert participants == set(_arms())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))