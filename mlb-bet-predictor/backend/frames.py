"""Canonical frame accessors — the single source of truth for "the decided
frame".

This module exists because the decided-frame bug class bit FOUR times:

1. the original 4,472-vs-4,466 fold desync;
2. the 2-row slate gap (COL@WSH / BAL@STL results filled post-merge shifted
   fold boundaries; fixed by the ``_pre_slate_games`` snapshot in 0d18eaf);
3. the run-engine coverage CSV built on post-slate windows (288/96 vs the
   canonical 282/94; fixed in b74ef73);
4. the OOF persist boundary writing unfiltered frames (identity-less slate
   rows shipped in run_engine_oof_20260827.csv; fixed in b74ef73).

Root pattern: every consumer reconstructed ``games[games["home_win"].notna()]``
independently, and ANY row-set difference (slate rows, identity-less rows,
duplicate game_pks) shifted fold boundaries. ``get_decided_frame`` encodes the
canonical rules ONCE so divergence is structurally impossible, and
``fold_signature`` lets call sites assert agreement loudly.

Canonical rules (in order):

1. **Post-game only**: ``home_win`` must be non-null. Ties/postponements and
   pregame rows are excluded.
2. **Identity required**: ``game_pk`` must parse as a number. This is what
   makes "a decided frame is never the slate frame" true BY CONSTRUCTION:
   slate rows that get results filled in post-merge enter the pipeline frame
   with no StatsAPI identity (the 08-27 slate rows carried null game_pk), and
   ESPN-id slate rows carry a non-numeric game_pk. Either way they are not
   joinable to margins/markets and must never reach a fold.
3. **Deterministic dedup**: one row per numeric game_pk — the LATEST
   game_date wins (stable mergesort, so identical (game_pk, game_date) pairs
   resolve by input order — deterministic for a given frame). No-op on the
   real frame (verified: 0 duplicate game_pks in game_level_features.csv),
   but it pins the policy so a future duplicate cannot silently split a
   game across folds.
4. **Order preservation**: a STABLE sort by game_date only (mergesort), which
   is a no-op on the pipeline's chronologically-built frame. We deliberately
   do NOT add a game_pk tie-break: measured on the real 6,953-frame, a
   (game_date, game_pk) sort would permute 6,892 of 6,953 rows, and the
   margin build's LightGBM subsampling is row-order-sensitive (measured max
   |Δmargin| ≈ 1.49 between the CSV-order and quicksort-order builds) — so
   re-permuting within-day rows would silently change trained models. The
   frame's existing chronological order IS part of the canonical contract;
   ``fold_signature`` asserts it across call sites.

``fold_signature`` hashes the frame's game_pk sequence plus the walk-forward
fold geometry (boundaries + per-fold game_pk multisets) so two call sites can
prove they consumed the same decided frame — same discipline as the
_attach_oof_run_margins desync guard, but structural instead of after-the-fact.
"""
import hashlib
import json
from typing import Any, Optional

import pandas as pd

from config import RETRAIN_CADENCE_DAYS

__all__ = ["get_decided_frame", "fold_signature",
           "fold_geometry_signature", "require_matching_signatures"]


def _numeric_game_pk(games: pd.DataFrame) -> Optional[pd.Series]:
    """Numeric view of game_pk (NaN where the identity is absent/non-numeric),
    or None when the frame carries no game_pk column at all (synthetic test
    frames without run-engine inputs)."""
    if "game_pk" not in games.columns:
        return None
    return pd.to_numeric(games["game_pk"], errors="coerce")


def get_decided_frame(games: pd.DataFrame) -> pd.DataFrame:
    """The ONE canonical decided frame every consumer must use.

    Encodes: post-game rows only, real (numeric) game_pk identity required —
    slate/pregame rows excluded by construction — deterministic dedup by
    game_pk (latest game_date wins), stable chronological order preserving
    the pipeline's within-day row order. See the module docstring for the
    four-strike bug-class history and the order-preservation rationale.
    """
    if games.empty:
        return games.copy()
    if "home_win" not in games.columns:
        raise ValueError("get_decided_frame: frame must carry 'home_win'")

    out = games[games["home_win"].notna()].copy()
    pk = _numeric_game_pk(out)
    if pk is not None:
        # Rule 2 — identity: slate/pregame/identity-less rows never fold.
        out = out[pk.notna()].copy()
        pk = _numeric_game_pk(out)
        # Rule 3 — deterministic dedup: latest game_date wins per game_pk.
        if out["game_pk"].duplicated().any():
            order = pd.to_datetime(out["game_date"], errors="coerce")
            out = (out.assign(_order=order)
                      .sort_values("_order", kind="mergesort")
                      .drop_duplicates(subset="game_pk", keep="last")
                      .drop(columns="_order"))
    # Rule 4 — stable chronological order: date-sorted without permuting
    # within-day rows (mergesort). No-op on the pipeline's sorted frame.
    out = out.sort_values("game_date", kind="mergesort")
    return out.reset_index(drop=True)


def _fold_payload(splits: list) -> list:
    """Order-robust fold geometry: boundaries + per-fold game_pk multisets
    (sorted — fold membership is a multiset; within-fold row order is a
    margin-build concern, not a membership concern)."""
    payload: list[list[Any]] = []
    for s in splits:
        val = s.get("val_games")
        pks = sorted(str(x) for x in val["game_pk"].tolist()) if val is not None else []
        payload.append([
            int(s["fold_idx"]),
            str(pd.Timestamp(s["val_start"]).date()),
            str(pd.Timestamp(s["val_end"]).date()),
            bool(s.get("is_partial_tail", False)),
            pks,
        ])
    return payload


def _splits_for(decided: pd.DataFrame,
                retrain_cadence_days: int) -> list:
    from training import walk_forward_splits
    return walk_forward_splits(decided,
                               retrain_cadence_days=retrain_cadence_days)


def fold_signature(decided: pd.DataFrame,
                   retrain_cadence_days: int = RETRAIN_CADENCE_DAYS) -> str:
    """Full identity of a decided frame: game_pk SEQUENCE (order-sensitive —
    the pipeline's within-day row order feeds LGBM's order-sensitive
    subsampling, so two frames with equal multisets but different orders are
    NOT interchangeable) + the walk-forward fold geometry.

    Call sites that consume the accessor share this signature; a mismatch
    means the two consumers built folds on different row sets. Lazy training
    import keeps frames.py importable from anywhere (training.py imports
    frames at module load).
    """
    splits = _splits_for(decided, retrain_cadence_days)
    seq = ([str(x) for x in decided["game_pk"].tolist()]
           if "game_pk" in decided.columns else [])
    blob = json.dumps({"folds": _fold_payload(splits), "sequence": seq},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def fold_geometry_signature(decided: Optional[pd.DataFrame] = None,
                            splits: Optional[list] = None,
                            retrain_cadence_days: int = RETRAIN_CADENCE_DAYS
                            ) -> str:
    """Order-ROBUST fold-geometry hash: boundaries + per-fold game_pk
    multisets only. Two decided frames with the same fold membership agree
    even if their within-day row order differs (e.g. the drift margin
    build's historical quicksort order vs the training frame's CSV order —
    equal geometry, documented margin-value sensitivity)."""
    if splits is None:
        if decided is None:
            raise ValueError("fold_geometry_signature: pass decided or splits")
        splits = _splits_for(decided, retrain_cadence_days)
    blob = json.dumps({"folds": _fold_payload(splits)},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def require_matching_signatures(left_name: str, left_sig: Optional[str],
                                right_name: str, right_sig: Optional[str]) -> None:
    """Fail loudly naming BOTH sides when two consumers' fold signatures
    disagree. A None signature (consumer never ran in-process) is skipped
    with a log — there is nothing to compare against."""
    import logging
    log = logging.getLogger("frames")
    if left_sig is None or right_sig is None:
        log.info("fold signature: %s=%s %s=%s — side absent, nothing to compare",
                 left_name, left_sig, right_name, right_sig)
        return
    if left_sig != right_sig:
        raise AssertionError(
            f"decided-frame desync: fold signature mismatch — "
            f"{left_name}={left_sig} vs {right_name}={right_sig}. The two "
            f"consumers built folds on different row sets; every consumer "
            f"must derive its frame via frames.get_decided_frame(games).")
