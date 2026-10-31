"""Per-game SHAP attributions for the NBA moneyline ensemble (display only).

Structural mirror of NHL's explainability contract (and behind it, MLB's):
one ``shap_game_<YYYYMMDD>_<AWAY>@<HOME>.csv`` per slate game carrying
feature / shap_value / signed_effect / perspective_team, which is exactly the
schema the frontend's game-card expander reads. The filename convention is
MLB's: the date and matchup ARE the key (MLB's game_id is that string), so a
card's expander resolves the same way in every sport.

What the cards get now: per-member log-odds attributions blended with the
deployed ensemble weights renormalized over the members that actually
produced values. Tree members (xgboost / lightgbm) contribute TreeExplainer
values; the linear member (elastic-net) contributes its exact margin
decomposition ``coef * z`` where ``z`` is the member's own standardized
input (train-fold means/stds via its fitted preprocessor, median-imputed) -
so the contributions are mean-centered against the training distribution and
sum to ``logit - intercept``. A weight-zero member is excluded (it does not
move the served prediction); a member with no values is excluded and named.

FAVORED-team perspective: the model outputs P(home win). When the AWAY team
is favored (p_home < 0.5) the attributions are negated so positive always
means "pushes the favorite toward winning" - the same convention MLB and NHL
use, so the frontend's chart reads identically across sports.

Output: ``data_delivery/shap_game_<YYYYMMDD>_<AWAY>@<HOME>.csv``.
Zero-attribution CSVs are never written: a game whose members cannot be
explained renders the frontend's "no file" state - nothing fabricated. And a
game with no explainable member is a WARNING, never a silent zero (2026-10-30:
elastic-net earned 100% of the blend while only tree members had explainers,
so every slate game was silently skipped and the cards showed nothing).
"""
from __future__ import annotations

import logging
import warnings
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


def shap_game_key(row: pd.Series) -> str:
    """MLB-style card key: ``YYYYMMDD_<AWAY>@<HOME>``.

    MLB's game_id IS this string, so its SHAP files resolve by construction.
    NBA game_ids are ESPN numerics, so the key is derived from the row's own
    game_date/away_team/home_team - the same triple the card already shows.
    """
    try:
        day = pd.Timestamp(row.get("game_date")).strftime("%Y%m%d")
    except (ValueError, TypeError, AttributeError):
        day = str(row.get("game_date", "") or "")[:10].replace("-", "")
    away = str(row.get("away_team", "") or "").strip()
    home = str(row.get("home_team", "") or "").strip()
    return f"{day}_{away}@{home}"


def _linear_margin_contrib(name: str, model, pre, X_df: pd.DataFrame,
                           gid: str) -> dict[str, float] | None:
    """Exact log-odds attribution for a linear (elastic-net) member.

    ``coef * z`` on the member's own standardized input sums to
    ``logit - intercept`` and is mean-centered against the training fold by
    construction (the preprocessor centers on train means). Returns None
    with a warning when shapes disagree - never a fabricated vector.
    """
    try:
        coef = np.asarray(getattr(model, "coef_", None), dtype=float)
    except (TypeError, ValueError):
        logger.warning("SHAP linear: %s/%s has no usable coef_ - member "
                       "excluded from this game's blend", gid, name)
        return None
    if coef.ndim == 2:
        if coef.shape[0] != 1:
            logger.warning("SHAP linear: %s/%s coef_ has %d rows, expected 1 "
                           "for binary logistic - member excluded",
                           gid, name, coef.shape[0])
            return None
        coef = coef[0]
    try:
        z = np.asarray(ml_mod.member_fit_input(name, X_df, pre), dtype=float)
    except Exception as exc:  # noqa: BLE001 - one opaque member cannot kill the report
        logger.warning("SHAP linear: %s/%s input build failed (%s) - member "
                       "excluded", gid, name, exc)
        return None
    if z.ndim != 2 or z.shape[0] != 1 or coef.shape[0] != z.shape[1]:
        logger.warning("SHAP linear: %s/%s size mismatch: %d coef(s) against "
                       "%s input - member excluded from this game's blend",
                       gid, name, coef.shape[0], z.shape)
        return None
    cols = list(X_df.columns)
    return {col: float(coef[i] * z[0, i]) for i, col in enumerate(cols)}


def _tree_margin_contrib(name: str, explainer, X_df: pd.DataFrame,
                         gid: str) -> dict[str, float] | None:
    """TreeExplainer log-odds attribution for one tree member, as a dict."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*LightGBM binary classifier with "
                        "TreeExplainer.*")
            raw = explainer.shap_values(X_df)
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
        cols = list(X_df.columns)
        if sv.size == 0 or sv.size != len(cols):
            logger.warning(
                "SHAP size mismatch for %s/%s: %d values against %d "
                "columns - member excluded from this game's blend",
                gid, name, sv.size, len(cols))
            return None
        return {col: float(sv[i]) for i, col in enumerate(cols)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP failed for %s/%s: %s", gid, name, exc)
        return None


def compute_nba_shap_per_game(bundle: dict, games: pd.DataFrame,
                              out_dir: Path) -> list[str]:
    """Write one SHAP CSV per slate game from the deployed ensemble bundle.

    ``bundle`` is the same shape the serving path persists
    (``nba_ensemble_latest.joblib``): ``moneyline_models`` ->
    ``moneyline_preprocessors`` -> ``ensemble_weights``. Returns the list of
    written file keys (``YYYYMMDD_AWAY@HOME``); an empty list means the
    frontend shows its empty state for this slate - and the reason is now
    always in the log as a WARNING, never a bare ``0``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import shap
    except ImportError:
        logger.warning("shap not available - no NBA SHAP files written")
        return []

    models = bundle.get("moneyline_models") or {}
    weights = bundle.get("ensemble_weights") or {}
    pre_map = bundle.get("moneyline_preprocessors") or {}
    if not models or not len(games):
        logger.warning("SHAP: no models (%d) or no games (%d) - no NBA SHAP "
                       "files written", len(models), len(games))
        return []

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

    linear_names = [n for n in getattr(ml_mod, "LINEAR_MEMBERS", set())
                    if models.get(n) is not None]

    written: list[str] = []
    failed = 0
    for idx, _row in games.iterrows():
        one = games.loc[[idx]]
        gid = str(one["game_id"].iloc[0])
        key = shap_game_key(one.iloc[0])
        per_member: list[tuple[str, float, dict[str, float]]] = []
        for name, ex in explainers.items():
            w = float(weights.get(name, 0.0))
            if w <= 0:
                continue  # weight-zero members do not move the prediction
            X_df = ml_mod.member_matrix(name, one)
            contrib = _tree_margin_contrib(name, ex, X_df, gid)
            if contrib is not None:
                per_member.append((name, w, contrib))
        for name in linear_names:
            w = float(weights.get(name, 0.0))
            if w <= 0:
                continue
            X_df = ml_mod.member_matrix(name, one)
            contrib = _linear_margin_contrib(
                name, models[name], pre_map.get(name), X_df, gid)
            if contrib is not None:
                per_member.append((name, w, contrib))

        if not per_member:
            # LOUD, not silent: this is the 2026-10-30 failure class (the
            # 100%-weighted member had no explainer, every game skipped, the
            # run stayed green and all cards showed "No SHAP file found").
            logger.warning(
                "SHAP: no member produced attributions for %s (%s) - "
                "weights %s, explainable members %s. Card will show the "
                "empty state; check that the serving weights and the "
                "explainer coverage agree.",
                gid, key,
                {n: round(float(weights.get(n, 0.0)), 4) for n in models},
                sorted(set(explainers) | set(linear_names)))
            failed += 1
            continue

        # Weighted blend in log-odds space, renormalized over the members
        # that actually produced values - the same combination rule the
        # ensemble's probability blend uses. Members have different feature
        # sets (linear_view drops raw per-side cols), so the blend runs over
        # the union with missing-as-zero, never a silently narrowed width.
        wsum = sum(w for _, w, _ in per_member)
        cols: list[str] = []
        for _, _, contrib in per_member:
            for col in contrib:
                if col not in cols:
                    cols.append(col)
        avg = {col: sum(w * contrib.get(col, 0.0)
                        for _, w, contrib in per_member) / wsum
               for col in cols}

        # Favored-team perspective (P(home win) model output).
        perspective = str(one["home_team"].iloc[0] or "HOME")
        p_home = one.get("home_win_prob_model")
        if p_home is not None:
            try:
                p_val = float(pd.to_numeric(pd.Series(p_home), errors="coerce").iloc[0])
                if np.isfinite(p_val) and p_val < 0.5:
                    avg = {col: -val for col, val in avg.items()}
                    perspective = str(one["away_team"].iloc[0] or "AWAY")
            except (TypeError, ValueError):
                pass

        rows = [{"feature": col,
                 "shap_value": round(float(avg[col]), 6),
                 "signed_effect": "positive" if avg[col] > 0 else "negative",
                 "perspective_team": perspective}
                for col in cols]
        pd.DataFrame(rows).to_csv(
            out_dir / f"{SHAP_GAME_PREFIX}_{key}.csv", index=False)
        written.append(key)

    logger.info("NBA SHAP attributions written for %d game(s)%s",
                len(written),
                f" ({failed} failed)" if failed else "")
    if failed and not written:
        logger.warning("SHAP: NO slate game produced attributions - every "
                       "card will show the empty state")
    return written
