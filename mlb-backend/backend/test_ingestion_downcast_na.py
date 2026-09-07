"""Regression test for _downcast NA-safety (ingestion._downcast).

Found during the local production-path validation run: with pandas >= 3,
select_dtypes(include=["int64"]) can surface Arrow-backed / nullable int
columns whose max()/min() return pd.NA even when isna().any() is False,
crashing the comparison with "boolean value of NA is ambiguous". The fix
skips such columns instead of crashing.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingestion import _downcast


class TestDowncastNaSafety(unittest.TestCase):
    def test_nullable_int_column_with_na_max_does_not_crash(self):
        df = pd.DataFrame({
            "game_pk": pd.array([1, 2, None], dtype="int64[pyarrow]"),
            "pitch_number": pd.array([1, 2, 3], dtype="int64[pyarrow]"),
            "release_speed": pd.Series([94.5, 95.1, 93.8], dtype="float64"),
            "pitch_type": ["FF", "SL", "CH"],
        })
        out = _downcast(df.copy())
        # Original values preserved; no exception raised.
        self.assertEqual(out["pitch_number"].tolist(), [1, 2, 3])
        self.assertTrue(pd.isna(out["game_pk"]).tolist() == [False, False, True])
        # Ordinary downcast behavior still applies to plain float64.
        self.assertEqual(str(out["release_speed"].dtype), "float32")

    def test_plain_int64_downcast_unaffected(self):
        df = pd.DataFrame({"n": pd.Series([1, 2, 3], dtype="int64"),
                           "release_speed": [94.0, 95.0, 96.0]})
        out = _downcast(df.copy())
        self.assertEqual(str(out["n"].dtype), "int16")
        self.assertEqual(str(out["release_speed"].dtype), "float32")


if __name__ == "__main__":
    unittest.main()
