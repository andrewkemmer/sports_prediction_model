"""Canonical decided-frame accessor (frames.py) — the structural fix for the
four-strike fold-desync bug class (4,472-vs-4,466; 2-row slate gap; post-slate
coverage windows; unfiltered OOF persist).

Property tests on the real 6,953-game frame + synthetic edge cases:
  - every consumer's decided frame is identical (multiset AND order);
  - slate rows NEVER enter the decided frame (null-pk and ESPN-id shapes);
  - duplicate game_pks resolve deterministically (latest game_date wins);
  - fold signatures are stable for the same frame and sensitive to real
    divergence (dropped row); fold GEOMETRY is order-robust while the full
    signature pins row order (LGBM margin builds are order-sensitive).
"""
import unittest
from pathlib import Path

import pandas as pd

from frames import (fold_geometry_signature, fold_signature,
                    get_decided_frame, require_matching_signatures)
from training import walk_forward_splits

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data_delivery" / "game_level_features.csv"


def _real_frame() -> pd.DataFrame:
    return pd.read_csv(FEATURES)


def _slate_rows() -> pd.DataFrame:
    """The 08-27 slate shape that caused the 2-row desync: results filled
    post-merge, no StatsAPI identity (null game_pk)."""
    return pd.DataFrame({
        "game_pk": [None, None],
        "game_id": ["20260827_colwsh", "20260827_balstl"],
        "game_date": ["2026-08-27", "2026-08-27"],
        "home_team": ["WSH", "STL"],
        "away_team": ["COL", "BAL"],
        "home_win": [1.0, 1.0],
        "home_score": [7.0, 7.0],
        "away_score": [1.0, 5.0],
    })


class TestCanonicalDecidedFrame(unittest.TestCase):
    def test_accessor_is_row_for_row_noop_on_real_frame(self):
        """Zero-drift contract: on the canonical training frame the accessor
        returns EXACTLY the rows training consumed, in the same order (a
        game_pk tie-break sort would permute 6,892/6,953 rows and re-seed
        LGBM's order-sensitive subsampling — never do that)."""
        f = _real_frame()
        decided = get_decided_frame(f)
        expected = f[f["home_win"].notna()].reset_index(drop=True)
        self.assertEqual(len(decided), len(expected))
        self.assertEqual(decided["game_pk"].tolist(),
                         expected["game_pk"].tolist())
        # full-row identity, not just keys
        pd.testing.assert_frame_equal(decided, expected)

    def test_slate_rows_never_enter_decided_frame(self):
        """Slate-invariance: adding the 2 slate rows (results filled, null
        game_pk) to the raw frame NEVER changes the decided frame — the
        0d18eaf desync is structurally impossible now. (dtype check off: the
        concat widens game_pk to object; the accessor passes dtypes through.)"""
        f = _real_frame()
        clean = get_decided_frame(f)
        with_slate = get_decided_frame(pd.concat([f, _slate_rows()],
                                                 ignore_index=True))
        self.assertEqual(len(with_slate), len(clean))
        self.assertEqual(with_slate["game_pk"].tolist(),
                         clean["game_pk"].tolist())
        pd.testing.assert_frame_equal(clean, with_slate, check_dtype=False)

    def test_non_numeric_slate_pks_excluded(self):
        """Slate rows with a non-parseable game_pk (e.g. composite id
        strings) are excluded even if home_win is filled — a defensive edge
        case against future slate shapes."""
        df = pd.DataFrame({
            "game_pk": [1, "NYYvBOS-20260827", 3],
            "game_date": ["2026-08-25"] * 3,
            "home_win": [1.0, 1.0, 0.0],
        })
        out = get_decided_frame(df)
        self.assertEqual(out["game_pk"].tolist(), [1, 3])

    def test_duplicate_game_pk_resolves_deterministically(self):
        """Policy: one row per game_pk, LATEST game_date wins (stable sort —
        identical (pk, date) ties resolve by input order)."""
        df = pd.DataFrame({
            "game_pk": [1, 2, 1, 1],
            "game_date": ["2026-08-25", "2026-08-25", "2026-08-26", "2026-08-20"],
            "home_win": [1.0, 0.0, 0.0, 1.0],
        })
        out = get_decided_frame(df)
        self.assertEqual(len(out), 2)  # pk1 (deduped) + pk2
        kept = out[out["game_pk"] == 1]
        self.assertEqual(len(kept), 1)
        self.assertEqual(str(kept["game_date"].iloc[0])[:10], "2026-08-26")
        # determinism: same input -> same output, twice
        pd.testing.assert_frame_equal(out, get_decided_frame(df))

    def test_accessor_order_is_stable_and_chronological(self):
        df = pd.DataFrame({
            "game_pk": [3, 1, 2],
            "game_date": ["2026-08-27", "2026-08-26", "2026-08-27"],
            "home_win": [1.0, 1.0, 0.0],
        })
        out = get_decided_frame(df)
        self.assertEqual(out["game_pk"].tolist(), [1, 3, 2])  # date order,
        # within-day input order preserved (3 before 2)

    def test_decidedness_and_identity_filters_compose(self):
        """Pregame rows (home_win null) AND identity-less rows are both
        excluded — post-game-only + identity-required, in one frame."""
        df = pd.DataFrame({
            "game_pk": [1, None, 3, 4],
            "game_date": ["2026-08-25"] * 4,
            "home_win": [1.0, 1.0, None, 0.0],
        })
        out = get_decided_frame(df)
        self.assertEqual(out["game_pk"].tolist(), [1, 4])


class TestFoldSignature(unittest.TestCase):
    def test_consumer_parity_on_real_frame(self):
        """Every consumer's construction path lands on the SAME signature:
        training-style, run-engine-style, and accessor output (all preserve
        the frame's chronological row order)."""
        f = _real_frame()
        sig_accessor = fold_signature(get_decided_frame(f))
        sig_training = fold_signature(
            f[f["home_win"].notna()].reset_index(drop=True))
        sig_re = fold_signature(
            f[f["home_win"].notna()].reset_index(drop=True))
        self.assertEqual(sig_accessor, sig_training)
        self.assertEqual(sig_accessor, sig_re)

    def test_geometry_is_order_robust_sequence_is_not(self):
        """The drift margin build historically consumed a quicksort-sorted
        view of the same rows (equal fold GEOMETRY, different within-day
        order). The geometry signature must agree across orders; the full
        signature must NOT (order feeds LGBM subsampling — max |Δmargin|
        ≈ 1.49 measured between the two orders on the real frame)."""
        f = _real_frame()
        acc = get_decided_frame(f)
        qs = f[f["home_win"].notna()].sort_values("game_date")
        self.assertEqual(fold_geometry_signature(acc),
                         fold_geometry_signature(qs))
        self.assertNotEqual(fold_signature(acc), fold_signature(qs))

    def test_slate_invariance_extends_to_signature(self):
        f = _real_frame()
        self.assertEqual(
            fold_signature(get_decided_frame(f)),
            fold_signature(get_decided_frame(
                pd.concat([f, _slate_rows()], ignore_index=True))))

    def test_signature_is_sensitive_to_real_divergence(self):
        f = _real_frame()
        base = fold_signature(get_decided_frame(f))
        # a dropped row changes the multiset -> different signature
        dropped = get_decided_frame(f.iloc[1:].reset_index(drop=True))
        self.assertNotEqual(base, fold_signature(dropped))
        self.assertNotEqual(fold_geometry_signature(get_decided_frame(f)),
                            fold_geometry_signature(dropped))
        # identical frame -> identical signature (repeatable)
        self.assertEqual(base, fold_signature(get_decided_frame(f)))

    def test_geometry_signature_folds_match_walk_forward_splits(self):
        f = _real_frame()
        decided = get_decided_frame(f)
        splits = walk_forward_splits(decided, retrain_cadence_days=7)
        self.assertEqual(fold_geometry_signature(splits=splits),
                         fold_geometry_signature(decided))
        # full signature repeatable
        self.assertEqual(fold_signature(decided), fold_signature(decided))
        # fold count sanity on the expansion frame
        self.assertEqual(len(splits), 81)

    def test_require_matching_signatures_names_both_sides(self):
        with self.assertRaises(AssertionError) as cm:
            require_matching_signatures("drift", "aaa", "training", "bbb")
        self.assertIn("drift=aaa", str(cm.exception))
        self.assertIn("training=bbb", str(cm.exception))
        # None side -> skipped, never raises
        require_matching_signatures("drift", None, "training", "bbb")


if __name__ == "__main__":
    unittest.main()
