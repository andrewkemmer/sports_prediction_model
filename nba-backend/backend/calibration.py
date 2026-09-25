"""NBA calibration compatibility facade."""
from __future__ import annotations
try:
    from backend.moneyline import (  # noqa: F401
        FAVORED_CALIBRATOR_METHOD, FAVORED_PROBABILITY_FLOOR, apply_platt,
        fit_platt, get_calibration_mode, moneyline_apply, moneyline_fit,
        set_calibration_mode,
    )
except ImportError:
    from moneyline import (  # noqa: F401
        FAVORED_CALIBRATOR_METHOD, FAVORED_PROBABILITY_FLOOR, apply_platt,
        fit_platt, get_calibration_mode, moneyline_apply, moneyline_fit,
        set_calibration_mode,
    )
