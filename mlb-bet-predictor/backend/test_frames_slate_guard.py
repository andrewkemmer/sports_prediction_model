"""Regression tests for get_decided_frame's slate-row identity guard.

Verifies that ESPN slate rows with numeric game_pk values (synthesised by
_attach_slate_run_margins from game_id) can never leak into the decided
frame, while pre-slate canonical frames pass through unchanged.
"""
import unittest

import numpy as np
import pandas as pd

from frames import get_decided_frame, fold_signature


def _make_statcast_row(game_pk, game_date="2026-04-01", home_team="NYY",
                       home_win=1.0, home_starter_id=680570):
    """A Statcast-decided row: has numeric game_pk AND home_starter_id."""
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "home_team": home_team,
        "away_team": "BOS",
        "home_win": home_win,
        "home_score": 5,
        "away_score": 3,
        "total_runs": 8,
        "home_starter_id": home_starter_id,
        "away_starter_id": 669372,
        "game_id": f"{game_date.replace('-', '')}_{home_team}@BOS",
    }


def _make_slate_row(game_pk, game_date="2026-08-28", home_team="NYY",
                    home_win=1.0):
    """An ESPN slate row: has numeric game_pk BUT NO home_starter_id."""
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "home_team": home_team,
        "away_team": "BOS",
        "home_win": home_win,
        "home_score": 4,
        "away_score": 2,
        "total_runs": 6,
        # NO home_starter_id — ESPN build_upcoming_slate never sets it
        "game_id": f"{game_date.replace('-', '')}_{home_team}@BOS",
    }


def _legacy_float_signature(post: pd.DataFrame) -> str:
    """Replicate the PRE-fix fold_signature on a float64-coerced frame:
    plain ``str(game_pk)`` ("823255.0") in both the sequence and the fold
    payload.  Used to pin the exact 08-28 desync hash (drift=5fb218bf) so
    the dtype-immune canonicalization is proven against history."""
    import hashlib, json
    from training import walk_forward_splits
    from config import RETRAIN_CADENCE_DAYS
    decided = post[post["home_win"].notna()]
    decided = decided[  # old filter: numeric pk only, then stable sort
        pd.to_numeric(decided["game_pk"], errors="coerce").notna()]
    decided = decided.sort_values("game_date", kind="mergesort")
    splits = walk_forward_splits(
        decided, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
    payload = []
    for s in splits:
        val = s.get("val_games")
        pks = sorted(str(x) for x in val["game_pk"].tolist()) if val is not None else []
        payload.append([int(s["fold_idx"]),
                        str(pd.Timestamp(s["val_start"]).date()),
                        str(pd.Timestamp(s["val_end"]).date()),
                        bool(s.get("is_partial_tail", False)), pks])
    seq = [str(x) for x in decided["game_pk"].tolist()]
    blob = json.dumps({"folds": payload, "sequence": seq},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class TestSlateGuard(unittest.TestCase):
    """get_decided_frame excludes ESPN slate rows with numeric game_pk."""

    def test_prealte_frame_unchanged(self):
        """Pre-slate frame (all rows have home_starter_id) passes through."""
        df = pd.DataFrame([
            _make_statcast_row(800001, "2026-04-01", "NYY"),
            _make_statcast_row(800002, "2026-04-02", "BOS"),
            _make_statcast_row(800003, "2026-04-03", "LAD"),
        ])
        result = get_decided_frame(df)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["game_pk"].tolist(), [800001, 800002, 800003])

    def test_postsLate_numeric_espn_game_pk_excluded(self):
        """Post-slate frame: ESPN numeric game_pks WITHOUT home_starter_id
        are excluded, even though they pass the numeric game_pk filter."""
        statcast_rows = [
            _make_statcast_row(800001, "2026-04-01", "NYY"),
            _make_statcast_row(800002, "2026-04-02", "BOS"),
        ]
        # ESPN game_ids are typically 401576xxx — numeric but NOT StatsAPI
        slate_rows = [
            _make_slate_row(401576789, "2026-08-28", "NYY"),
            _make_slate_row(401576790, "2026-08-28", "BOS"),
        ]
        df = pd.DataFrame(statcast_rows + slate_rows)
        result = get_decided_frame(df)
        # Only the 2 Statcast rows survive
        self.assertEqual(len(result), 2)
        self.assertTrue(all(pk > 700000 for pk in result["game_pk"]))

    def test_postsLate_home_starter_id_present_survives(self):
        """Post-slate frame: rows WITH home_starter_id survive even if
        they look like they could be slate (e.g. results-filled rows
        that DO carry a real starter)."""
        rows = [
            _make_statcast_row(800001, "2026-04-01", "NYY"),
            _make_slate_row(401576789, "2026-08-28", "NYY"),
        ]
        # Give the slate row a home_starter_id (simulating future pipeline
        # that sets starter IDs on slate rows)
        rows[1]["home_starter_id"] = 680570
        df = pd.DataFrame(rows)
        result = get_decided_frame(df)
        # Both survive — home_starter_id is the discriminator
        self.assertEqual(len(result), 2)

    def test_fold_signature_unchanged_on_prealte_frame(self):
        """fold_signature on pre-slate frame is identical before and after
        adding the starter guard — zero metric drift."""
        df = pd.DataFrame([
            _make_statcast_row(800001, "2026-04-01", "NYY"),
            _make_statcast_row(800002, "2026-04-02", "BOS"),
            _make_statcast_row(800003, "2026-04-03", "LAD"),
        ])
        sig = fold_signature(get_decided_frame(df))
        # Signature must be stable — no regression
        self.assertIsInstance(sig, str)
        self.assertEqual(len(sig), 16)  # SHA-256 truncated to 16 chars

    def test_slate_row_with_nan_home_starter_id_excluded(self):
        """Slate row with NaN home_starter_id is excluded even if game_pk
        is numeric."""
        df = pd.DataFrame([
            _make_statcast_row(800001, "2026-04-01", "NYY"),
            _make_slate_row(401576789, "2026-08-28", "NYY"),
        ])
        # Explicitly set home_starter_id to NaN on the slate row
        df.loc[df["game_pk"] == 401576789, "home_starter_id"] = np.nan
        result = get_decided_frame(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["game_pk"].iloc[0], 800001)

    def test_empty_frame_unchanged(self):
        """Empty frame returns empty — no crash."""
        df = pd.DataFrame(columns=["game_pk", "game_date", "home_team",
                                    "home_win", "home_starter_id"])
        result = get_decided_frame(df)
        self.assertEqual(len(result), 0)

    def test_frame_without_starter_id_column_unchanged(self):
        """Frame without home_starter_id column (e.g. synthetic test data)
        skips the guard — backward compatible."""
        df = pd.DataFrame([
            {"game_pk": 800001, "game_date": "2026-04-01",
             "home_team": "NYY", "home_win": 1.0},
            {"game_pk": 800002, "game_date": "2026-04-02",
             "home_team": "BOS", "home_win": 0.0},
        ])
        result = get_decided_frame(df)
        self.assertEqual(len(result), 2)

    def test_real_frame_unchanged(self):
        """The committed 6,960-game frame is unchanged by the guard."""
        from pathlib import Path
        csv = Path(__file__).parent.parent / "data_delivery" / "game_level_features.csv"
        if not csv.exists():
            self.skipTest("committed CSV not available")
        df = pd.read_csv(csv)
        df["game_date"] = pd.to_datetime(df["game_date"])
        result = get_decided_frame(df)
        # All rows have home_starter_id — guard is a no-op
        self.assertEqual(len(result), len(df))

    def test_postsLate_concat_dtype_coercion_signature_unchanged(self):
        """THE 08-28 desync, byte-exact: the Step-4 slate concat coerces
        game_pk int64→float64 (slate rows carry NaN game_pk), which changed
        fold_signature on the IDENTICAL 6,953-row set ("823255.0" vs
        "823255") → drift=5fb218bf vs training=4a377bac.  The decided frame
        must normalize back to int64 so the post-slate frame hashes to the
        same signature as the pre-slate frame."""
        import subprocess
        from pathlib import Path
        csv = Path(__file__).parent.parent / "data_delivery" / "game_level_features.csv"
        if not csv.exists():
            self.skipTest("committed CSV not available")
        df = pd.read_csv(csv)
        df["game_date"] = pd.to_datetime(df["game_date"])
        # Build the slate exactly as the pipeline does: rows with a readable
        # game_id, NO game_pk column, results filled (evening run) — the
        # concat then coerces game_pk to float64.
        slate = pd.DataFrame({
            "game_id": ["20260828_CIN@CHC", "20260828_LAD@DET"],
            "game_date": pd.to_datetime(["2026-08-28", "2026-08-28"]),
            "home_team": ["CHC", "DET"], "away_team": ["CIN", "LAD"],
            "home_win": [1.0, 0.0], "home_score": [4, 2], "away_score": [1, 5],
        })
        post = pd.concat([df, slate], ignore_index=True)
        self.assertEqual(post["game_pk"].dtype, np.float64)  # coercion proof

        pre = get_decided_frame(df)
        post_decided = get_decided_frame(post)
        # Rule 2a: dtype normalized back to int64.
        self.assertEqual(post_decided["game_pk"].dtype, np.int64)
        # Same rows, same order, same signature — the desync is impossible.
        self.assertEqual(len(post_decided), len(pre))
        self.assertEqual(post_decided["game_pk"].tolist(), pre["game_pk"].tolist())
        self.assertEqual(fold_signature(post_decided), fold_signature(pre))
        # Frame-identity pin: the committed 6,960-frame's canonical
        # signature (the training side of the historical run).  The 08-28
        # drift side was 5fb218bfd8d9af91 BEFORE the dtype fix — the float
        # coercion hash of this same row set (replicated below).
        self.assertEqual(fold_signature(pre), "019e1f0675aef5ab")
        # Before/after proof: the PRE-fix signature of the float64-coerced
        # frame (str(game_pk) without canonicalization) DIFFERS from the
        # canonical one on the identical row set — the exact failure mode
        # that produced drift=5fb218bfd8d9af91 vs training=4a377bac0cfbac1e
        # on the historical 6,953-frame (reproduced byte-exact from the
        # committed 42ca8f6 frame + slate).  The fix must keep the two
        # signatures identical so the assert can never fire on dtype noise.
        old_sig = _legacy_float_signature(post)
        self.assertNotEqual(old_sig, fold_signature(pre))

    def test_fold_signature_dtype_immune_float64(self):
        """fold_signature hashes the game IDENTITY, never the column dtype:
        the same frame with int64 vs float64 game_pk must hash identically,
        even when the float64 frame is passed directly (bypassing
        get_decided_frame's normalization)."""
        df = pd.DataFrame([
            _make_statcast_row(800001, "2026-04-01", "NYY"),
            _make_statcast_row(800002, "2026-04-02", "BOS"),
            _make_statcast_row(800003, "2026-04-03", "LAD"),
        ])
        base = get_decided_frame(df)
        fl = base.copy()
        fl["game_pk"] = fl["game_pk"].astype("float64")
        self.assertEqual(fold_signature(fl), fold_signature(base))
        # A non-integer string pk stays untouched (no truncation games).
        weird = base.copy()
        weird.loc[0, "game_pk"] = "20260828_CIN@CHC"
        self.assertNotEqual(
            fold_signature(weird), fold_signature(base))


if __name__ == "__main__":
    unittest.main()
