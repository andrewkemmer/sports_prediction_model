"""Blend-objective policy pin (2026-10-14 four-sport moneyline audit).

The binary moneyline's blend weights are earned by POOLED OOF log loss
(``compute_adaptive_weights``, simplex SLSQP in logit space). NBA and NHL
``compute_adaptive_weights`` hard-refuse any configured metric other than
logloss; MLB retains the historical ``ADAPTIVE_WEIGHT_METRIC == "auc"``
experiment branch, so the shipped value is pinned here instead: a config
flip must fail this test before it can silently re-aim the production
blend optimizer away from the stated objective.
"""
from __future__ import annotations

import numpy as np

try:
    from backend import config, training
except ImportError:  # pragma: no cover - direct script/import fallback
    import config
    import training


def test_shipped_config_pins_logloss():
    assert config.ADAPTIVE_WEIGHT_METRIC == "logloss"
    # training binds the metric at import time; both must agree.
    assert training.ADAPTIVE_WEIGHT_METRIC == "logloss"


def test_weights_minimize_pooled_logloss_on_the_simplex():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    strong = [0.2, 0.8, 0.3, 0.7, 0.25, 0.75, 0.35, 0.65]
    weak = [0.5] * 8  # uninformative constant member
    w = training.compute_adaptive_weights(
        {"elasticnet": strong, "lightgbm": weak}, y)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    # The member with the better pooled OOF log loss must dominate.
    assert w["elasticnet"] > w["lightgbm"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(0 if not any(
        fn() is False for fn in (
            test_shipped_config_pins_logloss,
            test_weights_minimize_pooled_logloss_on_the_simplex,
        )) else 1)
