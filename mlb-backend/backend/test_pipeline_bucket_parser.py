from __future__ import annotations

import unittest

import pandas as pd

# Env-only dependency: the tests below need pytest.MonkeyPatch / pytest.raises.
# When pytest is absent (this Windows desktop box), report a readable module
# SKIP instead of an ImportError-class failure so discovery stays green.
try:
    import pytest
except ImportError:  # pragma: no cover - env without pytest
    raise unittest.SkipTest("pytest not installed (environment-only "
                            "dependency)")

import pipeline  # noqa: E402


def _oof() -> pd.DataFrame:
    return pd.DataFrame({
        "game_date": pd.to_datetime(["2026-08-01", "2026-08-01"]),
        "home_win": [1.0, 0.0],
        "home_win_prob_model": [0.60, 0.55],
        "home_win_prob_model_calibrated": [0.59, 0.54],
    })


def _run(bucket: str) -> list[dict]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pipeline, "calibration_buckets", lambda *_args, **_kwargs: [{"bucket": bucket}])
        return pipeline._daily_calibration_rows(_oof())


def test_daily_calibration_rows_accepts_en_dash_percent_bucket():
    rows = _run("50–60%")
    assert len(rows) == 1
    assert rows[0]["date"] == "20260801"


def test_daily_calibration_rows_accepts_hyphen_bucket():
    rows = _run("50-60")
    assert len(rows) == 1
    assert rows[0]["date"] == "20260801"


def test_daily_calibration_rows_descriptive_error_for_invalid_bucket():
    with pytest.raises(ValueError, match="Could not parse calibration bucket label"):
        _run("not-a-range")
