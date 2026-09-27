"""Per-game SHAP attributions for the NBA moneyline ensemble (display only).

Structural mirror of NHL's explainability contract (and behind it, MLB's):
one ``nba_shap_game_<game_id>.csv`` per slate game carrying
feature / shap_value / signed_effect / perspective_team, which is exactly the
schema the frontend's game-card expander reads. The module existed before
this rewrite, but nothing called it, and what it wrote was not SHAP: global
``feature_importances_`` (identical for every game, ordered the same way)
under a per-game filename - the board's expanders would have rendered the
same chart forty times and called it explanation.

What the cards get now: attributions averaged across the TREE members
(xgboost / lightgbm) in log-odds space and blended with the deployed
ensemble weights - the same members that own the tree-family view of the
production blend. The linear member (elastic-net) is excluded exactly as in
MLB and NHL (its scaled input space has no leaf-path attribution).

FAVORED-team perspective: the model outputs P(home win). When the AWAY team
is favored (p_home < 0.5) the attributions are negated so positive always
means "pushes the favorite toward winning" - the same convention MLB and NHL
use, so the frontend's chart reads identically across sports.

Output: ``data_delivery/nba_shap_game_<game_id>.csv``. Zero-attribution CSVs
are never written: a game whose members cannot be explained renders the
frontend's "no file" state - nothing fabricated.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from backend import config
    from backend import moneyline as ml_mod
except ImportError:  # pragma: no cover - direct script execution
    import config
    import moneyline as ml_mod

logger = logging.getLogger(__name__)

TREE_MEMBERS = ("xgboost", "lightgbm")
SHAP_GAME_PREFIX = config.SHAP_GAME_PREFIX


def compute_nba_shap_per_game(bundle: dict, games: pd.DataFrame,
                              out_dir: Path) -> int:
    """Write one SHAP CSV per slate game from the deployed ensemble bundle.

    ``bundle`` is the same shape the serving path persists
    (``nba_ensemble_latest.joblib``): ``moneyline_models`` ->
    ``moneyline_preprocessors`` -> ``ensemble_weights``. Returns the number of
    files written; 0 means the frontend shows its empty state for this slate.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import shap
    except ImportError:
        logger.warning("shap not available - no NBA SHAP files written")
        return 0

    models = bundle.get("moneyline_models") or {}
    weights = bundle.get("ensemble_weights") or {}
    if not models or not len(games):
        return 0

    # Explainers built ONCE per tree member (the MLB pattern), on the SAME
    # member matrices the predict path builds - a narrower or reshaped input
    # here is what made the first per-game attempt silently attribute the
    # wrong width.
    explainers: dict[str, object] = {}
    for name in TREE_MEMBERS:
        model = models.get(name)
        if model is None:
            continue
        try:
            explainers[name] = shap.TreeExplainer(model)
        except Exception as exc:  # noqa: BLE001 - one opaque member cannot kill the report
            logger.warning("TreeExplainer init failed for %s: %s", name, exc)
    if not explainers:
        return 0
    # The attribution width, named once: the exact matrix the members consume
    # at predict time, read through the one authoritative builder. A narrower
    # matrix here is what made the first per-game attempt silently attribute
    # the wrong width.
    feature_cols = list(ml_mod.member_matrix("xgboost", games).columns)

    written = 0
    for idx, _row in games.iterrows():
        one = games.loc[[idx]]
        per_member: list[tuple[str, np.ndarray]] = []
        for name, ex in explainers.items():
            try:
                pre = (bundle.get("moneyline_preprocessors") or {}).get(name)
                X = ml_mod.member_matrix_ndarray(name, one, pre)
                raw = ex.shap_values(X)
                # Binary output shapes vary by explainer version and member:
                # normalize to the log-odds view (class 1 minus class 0).
                if isinstance(raw, list):
                    arrs = [np.asarray(a, dtype=float) for a in raw]
                    sv = (arrs[1] - arrs[0]) if len(arrs) == 2 else arrs[0]
                else:
                    arr = np.asarray(raw, dtype=float)
                    if arr.ndim == 3 and arr.shape[-1] == 2:
                        sv = arr[..., 1] - arr[..., 0]
                    else:
                        sv = arr
                sv = np.ravel(sv)
                if sv.size == 0 or sv.size != len(feature_cols):
                    logger.warning(
                        "SHAP size mismatch for %s/%s: %d values against %d "
                        "columns - member excluded from this game's blend",
                        one["game_id"].iloc[0], name, sv.size, len(feature_cols))
                    continue
                per_member.append((name, sv))
            except Exception as exc:  # noqa: BLE001
                logger.warning("SHAP failed for %s/%s: %s",
                               one["game_id"].iloc[0], name, exc)

        if not per_member:
            # The silent-empty failure class: a games frame missing the
            # feature columns would produce zero files while the run stays
            # green. Named, never silent.
            logger.warning("SHAP: no member produced attributions for %s - "
                           "check that the games frame carries the serving "
                           "feature columns", one["game_id"].iloc[0])
            continue

        # Weighted blend in log-odds space, renormalized over the members
        # that actually produced values - the same combination rule the
        # ensemble's probability blend uses.
        w = np.array([float(weights.get(name, 0.0)) for name, _ in per_member])
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        avg = sum(wi * sv for wi, (_, sv) in zip(w, per_member))

        # Favored-team perspective (P(home win) model output).
        perspective = str(one["home_team"].iloc[0] or "HOME")
        p_home = one.get("home_win_prob_model")
        if p_home is not None:
            try:
                p_val = float(pd.to_numeric(pd.Series(p_home), errors="coerce").iloc[0])
                if np.isfinite(p_val) and p_val < 0.5:
                    avg = -avg
                    perspective = str(one["away_team"].iloc[0] or "AWAY")
            except (TypeError, ValueError):
                pass

        rows = [{"feature": col,
                 "shap_value": round(float(avg[i]), 6),
                 "signed_effect": "positive" if avg[i] > 0 else "negative",
                 "perspective_team": perspective}
                for i, col in enumerate(feature_cols) if i < len(avg)]
        gid = str(one["game_id"].iloc[0])
        pd.DataFrame(rows).to_csv(
            out_dir / f"{SHAP_GAME_PREFIX}_{gid}.csv", index=False)
        written += 1

    logger.info("NBA SHAP attributions written for %d game(s)", written)
    return written
