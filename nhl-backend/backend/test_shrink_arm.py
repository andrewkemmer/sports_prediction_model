"""Player-rating shrinkage A/B arm (NHL_SHRINK_ARM, 2026-10-02).

Both arms of the owner-declared rating shrinkage schedule, pinned at the
``shrink_rate()`` level — the exact function ``build_player_ratings()``
applies to every rolling row — so a refactor cannot silently change the
math. Carried over from MLB's batter-rating arm
(``mlb-backend/backend/test_shrink_arm.py``).

  bayesian (SHIPPED default)  w = n / (n + k) — prior never washes
      out (50% own weight at n = k).
  ramp (gated 2026-10-02 — NOT adopted; selectable via
      NHL_SHRINK_ARM=ramp)  w = min(n / k, 1) — full own weight at
      the 20%-of-a-season opportunity (k seconds), raw thereafter.

Unlike MLB's flat ``BATTER_SHRINK_K = 120``, NHL's k is data-derived
(``SHRINK_FRACTION_OF_SEASON * mean prior-season ice time`` per
position/situation), so these tests fix k explicitly and pin both
schedules against it.

Invariants covered:
  * the default arm is bayesian — the 2026-10-02 A/B returned a wash
    (holdout delta +0.00044, inside the ±0.001 bar) and the owner
    declined adoption for now; ramp stays one env var away
    (NHL_SHRINK_ARM=ramp), and any env value is one of the two arms;
  * an unknown arm raises instead of silently shipping the default;
  * bayesian evaluates to the shipped literal formula;
  * ramp keeps exactly w = n/k own weight below k and the raw rate at
    or above k (league prior fully washed out);
  * ramp's crossing at k wobbles by only (mu_ice - raw)/k — far below
    any visible step;
  * a player exactly at the league prior is untouched by either arm;
  * zero ice with a valid k shrinks fully to the prior under both arms;
  * non-finite input and a zero denominator stay NaN under both arms;
  * the documented mid-window divergence: at n = k/2 the ramp trusts
    the player's own data 50% vs bayesian's 33%.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import player_ratings as pr  # noqa: E402

MU60 = 0.5                 # league prior, xG per 60 minutes
K = 2_000.0                # seconds — test fixture for the data-derived k
MU = MU60 / pr.SECONDS_PER_HOUR   # league prior, xG per second


def _w(value: float, raw: float) -> float:
    """Recover own weight from value = w * raw + (1 - w) * mu."""
    return (value - MU) / (raw - MU)


def test_env_arm_is_valid_and_defaults_to_bayesian():
    # env unset -> module default must be the SHIPPED arm; an explicit
    # NHL_SHRINK_ARM in the environment must still be one of the two arms.
    assert pr.NHL_SHRINK_ARM in ("bayesian", "ramp")
    if "NHL_SHRINK_ARM" not in __import__("os").environ:
        assert pr.NHL_SHRINK_ARM == "bayesian"


def test_unknown_arm_raises_instead_of_silently_defaulting():
    with pytest.raises(ValueError, match="NHL_SHRINK_ARM"):
        pr.shrink_rate(1.0, 1_000.0, MU60, K, arm="linear")
    with pytest.raises(ValueError, match="NHL_SHRINK_ARM"):
        pr.shrink_rate(1.0, 1_000.0, MU60, K, arm="")


def test_build_player_ratings_rejects_an_unknown_arm_loudly():
    with pytest.raises(ValueError, match="NHL_SHRINK_ARM"):
        pr.build_player_ratings(pd.DataFrame(), shrink_arm="linear")


def test_build_records_the_arm_in_the_audit():
    _, audit = pr.build_player_ratings(pd.DataFrame(), id_col="player_id")
    assert audit["shrink_arm"] == pr.NHL_SHRINK_ARM


def test_bayesian_matches_shipped_formula():
    # thin sample leans hard on the prior
    got = pr.shrink_rate(2.0, 500.0, MU60, K, arm="bayesian")
    assert got == pytest.approx((2.0 + MU * K) / (500.0 + K))
    # at n = k the own weight is exactly 1/2 — the prior never washes out
    raw = 0.0002
    got = pr.shrink_rate(raw * K, K, MU60, K, arm="bayesian")
    assert _w(got, raw) == pytest.approx(0.5, abs=1e-12)


def test_ramp_weight_schedule_is_linear_then_raw():
    r_bar = 0.0002  # xG per second; mu = 0 isolates the weight schedule
    # below k: own weight exactly n/k
    for n in (1.0, 500.0, 1_000.0, 1_999.0):
        got = pr.shrink_rate(r_bar * n, n, 0.0, K, arm="ramp")
        assert got == pytest.approx(r_bar * n / K), f"n={n}"
    # at and above k: raw own rate, prior fully washed out
    for n in (2_000.0, 2_500.0):
        got = pr.shrink_rate(r_bar * n, n, 0.0, K, arm="ramp")
        assert got == pytest.approx(r_bar), f"n={n}"


def test_ramp_no_wobble_crossing_threshold():
    r_bar = 0.0002
    below = pr.shrink_rate(r_bar * (K - 1), K - 1, MU60, K, arm="ramp")
    at = pr.shrink_rate(r_bar * K, K, MU60, K, arm="ramp")
    # continuous: the 1-second step is (mu - raw)/k per second, i.e.
    # ~1e-4 xG/60 — nowhere near a tenth of a hundredth of a unit.
    assert abs(at - below) * pr.SECONDS_PER_HOUR < 0.001
    # and past k the league prior exerts ZERO pull on the ramp
    hot = pr.shrink_rate(0.0004 * 2_500.0, 2_500.0, MU60, K, arm="ramp")
    assert hot == pytest.approx(0.0004)


def test_league_average_player_untouched_in_both_arms():
    for arm in ("bayesian", "ramp"):
        for n in (500.0, 2_000.0, 3_000.0):
            got = pr.shrink_rate(MU * n, n, MU60, K, arm=arm)
            assert got == pytest.approx(MU), f"{arm} n={n}"


def test_zero_evidence_is_fully_the_prior_in_both_arms():
    for arm in ("bayesian", "ramp"):
        assert pr.shrink_rate(0.0, 0.0, MU60, K, arm=arm) == pytest.approx(MU)


def test_nan_gates_are_identical_under_both_arms():
    bad = [
        (float("nan"), 100.0, MU60, K),
        (1.0, float("nan"), MU60, K),
        (1.0, 100.0, float("nan"), K),
        (1.0, 100.0, MU60, float("nan")),
        (0.0, 0.0, MU60, 0.0),   # zero denominator
    ]
    for arm in ("bayesian", "ramp"):
        for args in bad:
            assert np.isnan(pr.shrink_rate(*args, arm=arm)), f"{arm} {args}"


def test_documented_midwindow_divergence():
    # at n = k/2 with a hot raw rate the ramp keeps 50% own weight, bayes 33%
    r_bar = 0.0004  # hot: 2x prior
    n = K / 2
    w_ramp = _w(pr.shrink_rate(r_bar * n, n, MU60, K, arm="ramp"), r_bar)
    w_bayes = _w(pr.shrink_rate(r_bar * n, n, MU60, K, arm="bayesian"), r_bar)
    assert w_ramp == pytest.approx(0.5)
    assert w_bayes == pytest.approx(1 / 3, abs=1e-9)
    assert w_ramp > w_bayes
