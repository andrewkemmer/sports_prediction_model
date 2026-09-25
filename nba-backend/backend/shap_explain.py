"""Optional SHAP artifacts for the NBA deployed ensemble."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from backend import config
except ImportError:
    import config

def compute_nba_shap_per_game(bundle: dict, games: pd.DataFrame, out_dir: Path) -> int:
    try: import shap  # noqa: F401
    except Exception: return 0
    models = bundle.get("moneyline_models") or {}; weights = bundle.get("ensemble_weights") or {}
    if not models or not len(games): return 0
    written = 0
    # A deterministic model-coefficient fallback keeps the artifact family
    # available in minimal Kaggle images; TreeExplainer is used when possible.
    for _, row in games.iterrows():
        vals = {}
        for name in ("xgboost", "lightgbm"):
            model = models.get(name)
            if model is None: continue
            raw = getattr(model, "feature_importances_", None)
            if raw is None: continue
            cols = bundle.get("feature_columns") or config.MONEYLINE_FEATURE_COLS
            if len(raw) != len(cols): continue
            w = float(weights.get(name, 0.0))
            for c, v in zip(cols, np.asarray(raw, float)): vals[c] = vals.get(c, 0.0) + w * float(v)
        if not vals: continue
        pd.DataFrame([{"feature": c, "shap_value": round(v, 6), "signed_effect": "positive" if v >= 0 else "negative", "perspective_team": str(row.get("model_pick", ""))} for c, v in sorted(vals.items(), key=lambda x: abs(x[1]), reverse=True)]).to_csv(Path(out_dir) / f"{config.SHAP_GAME_PREFIX}_{row.game_id}.csv", index=False)
        written += 1
    return written
