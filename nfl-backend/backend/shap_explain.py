"""Per-game SHAP attributions for the NFL moneyline ensemble (display only).

Mirrors the MLB explainability contract (``shap_game_<game_id>.csv`` →
feature / shap_value / signed_effect / perspective_team) with the NFL
ensemble's semantics: attributions are averaged across the TREE members
(xgboost / lightgbm / randomforest) in log-odds space and blended with the
deployed adaptive weights — the same members that own the tree-family view
of the production blend. Linear/MLP members are excluded exactly like MLB
(their scaled input space has no leaf-path attribution).

FAVORED-team perspective: the model outputs P(home win). When the AWAY team
is favored (p_home < 0.5) the attributions are negated so positive always
means "pushes the favorite toward winning" — the same convention MLB uses
and the calibration page's favored-side view.

Output: data_delivery/nfl_shap_game_<game_id>.csv (one file per slate game).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from backend import config
    from backend import features as feat_mod
    from backend import moneyline as ml_mod
except ImportError:  # pragma: no cover - direct script execution
    import config
    import features as feat_mod
    import moneyline as ml_mod

logger = logging.getLogger(__name__)

TREE_MEMBERS = ("xgboost", "lightgbm", "randomforest")
SHAP_GAME_PREFIX = config.SHAP_GAME_PREFIX


def compute_nfl_shap_per_game(bundle: dict, games: pd.DataFrame,
                              out_dir: Path) -> int:
    """Write one SHAP CSV per slate game from the deployed ensemble bundle.

    ``bundle`` is the persisted ``nfl_ensemble_latest.joblib`` dict (the
    SAME object the serving path loads — never a refit). Returns the number
    of files written. Zero-attribution CSVs are never written: a game whose
    members cannot be explained renders the frontend's 'no file' state
    (nothing fabricated).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import shap
    except ImportError:
        logger.warning("shap not available — no NFL SHAP files written")
        return 0

    models = bundle.get("moneyline_models") or {}
    weights = bundle.get("ensemble_weights") or {}
    if not models or not len(games):
        return 0

    # Explainers built ONCE per tree member (the MLB pattern). Feature
    # columns come from the full games frame via the shared member_matrix
    # builder — the exact matrix the members consume at predict time.
    explainers: dict[str, object] = {}
    for name in TREE_MEMBERS:
        model = models.get(name)
        if model is None:
            continue
        try:
            ex = shap.TreeExplainer(model)
            explainers[name] = ex
        except Exception as exc:  # noqa: BLE001
            logger.warning("TreeExplainer init failed for %s: %s", name, exc)
    if not explainers:
        return 0
    feature_cols = list(ml_mod.member_matrix("xgboost", games).columns)

    written = 0
    for idx, row in games.iterrows():
        one = games.loc[[idx]]
        per_member = []
        for name, ex in explainers.items():
            pre = (bundle.get("moneyline_preprocessors") or {}).get(name)
            try:
                X = ml_mod.member_matrix_ndarray(name, one, pre)
                raw = ex.shap_values(X)
                # Binary output shapes vary by explainer version/member:
                # a single (1, n) array, a list of two (1, n) arrays, or an
                # (n_samples, n, 2)-style stacked array. Normalize to the
                # log-odds view: class-1 minus class-0 when both are given.
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
                        "SHAP size mismatch for %s/%s: %d values, expected %d "
                        "— excluding member from attributions.",
                        row.get("game_id"), name, sv.size, len(feature_cols))
                    continue
                per_member.append((name, sv))
            except Exception as exc:  # noqa: BLE001
                logger.warning("SHAP failed for %s/%s: %s",
                               row.get("game_id"), name, exc)

        if not per_member:
            continue

        # Weighted blend in log-odds space: the tree members' attributions
        # are summed per member (log-odds additive), then combined with the
        # SAME adaptive weights the ensemble blend uses (renormalized over
        # the members that produced values — never over all five).
        w = np.array([float(weights.get(n, 0.0)) for n, _ in per_member])
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        avg = sum(wi * sv for wi, (_, sv) in zip(w, per_member))

        # Favored-team perspective (P(home win) model output).
        p_home = row.get("home_win_prob_model")
        perspective = str(row.get("home_team", "") or "HOME")
        try:
            if pd.notna(p_home) and float(p_home) < 0.5:
                avg = -avg
                perspective = str(row.get("away_team", "") or "AWAY")
        except (TypeError, ValueError):
            pass

        shap_values = {c: round(float(avg[i]), 6)
                       for i, c in enumerate(feature_cols)
                       if i < len(avg)}
        rows = [{"feature": feat,
                 "shap_value": val,
                 "signed_effect": "positive" if val > 0 else "negative",
                 "perspective_team": perspective}
                for feat, val in sorted(shap_values.items(),
                                        key=lambda kv: abs(kv[1]),
                                        reverse=True)]
        gid = str(row["game_id"])
        pd.DataFrame(rows).to_csv(
            out_dir / f"{SHAP_GAME_PREFIX}_{gid}.csv", index=False)
        written += 1

    logger.info("NFL SHAP attributions written for %d games", written)
    return written
