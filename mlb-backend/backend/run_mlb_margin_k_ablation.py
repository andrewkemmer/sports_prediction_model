"""Moneyline margin-k ablation — C0 raw run_margin_diff vs C1 k-expanded.

Questions (2026-09 margin-k task): does applying the run engine's C2 edge
expansion (k) to the moneyline's ``run_margin_diff`` feature — fit
FOLD-LOCALLY within the moneyline's own fold discipline — improve the binary
moneyline vs the current raw edge?

Verified trace (read-only, 2001030): ``run_margin_diff = λ_home − λ_away``
at build_oof_margin.py:116 (OOF fold-attach) and :152 (refit_run_margins),
built via training._attach_oof_run_margins → build_oof_margin.oof_run_margins,
per-side λs from run_engine._fit_side_model. The run engine's C2 daily k
(fit once per run on the run engine's OOF, run_engine_k_edge.py) is NOT
applied there and MUST NOT be used here — the moneyline's fold-local OOF
discipline is the whole point.

Arms (identical folds/seed as the moneyline walk-forward, RETRAIN_CADENCE_DAYS
= 7, MIN_VAL_FOLD_GAMES = 40, same 5-member ensemble + adaptive blend, same
sealed 21-day holdout gate):
  * C0: production — raw run_margin_diff (unchanged).
  * C1: k-expanded run_margin_diff — for EACH moneyline fold f, fit k on
    that fold's strictly-prior training portion ONLY (rows with
    fold_idx < f, additionally date-guarded to be strictly before f's val
    window), then apply it to the margin attached to fold f's validation
    rows. NO use of the daily-run k, no cross-split fitted constants;
    sealed 2025 never participates in any k fit.

k math (same as the run engine's C2, run_engine_k_edge.py):
    m = (λ_H + λ_A)/2,  λ'_H = m + k(λ_H − m),  λ'_A = m − k(λ_A − m)
    ⇒ λ'_H + λ'_A = λ_H + λ_A (sum preserved) and
      λ'_H − λ'_A = k·(λ_H − λ_A)  (the midpoint cancels in the DIFFERENCE)
    Because run_margin_diff only carries the difference, the k-expanded
    margin is EXACTLY k × raw_margin — the harness implements it that way,
    and tests assert the equivalence with the two-sided expansion.
    k_f = OLS slope of ACTUAL margin on the λ edge over the strictly-prior
    rows (np.polyfit(edge, margin, 1)[0]) — the run engine's fit_k_edge
    definition. Guards mirror fit_k_edge: n ≥ 100 prior rows and edge
    sd ≥ 1e-9, else k_f = 1.0 (identity, warm-up folds).

Holdout (sealed) leg: production convention — fit-only refit at the median
fold round count (refit_run_margins, raw λs). For C1 the holdout margin is
scaled by k_last (the final executed fold's k — fitted strictly prior to the
holdout window by walk-forward construction), the natural per-run analogue
of the daily k.

Harness mechanics (mirrors run_margin_ablation.py exactly):
  * prepare_data → build_oof_margin.oof_run_margins on the moneyline folds,
    cached under /tmp/margin_oof_cache_<hash>.parquet (same key as
    run_margin_ablation, so the derivation is shared).
  * run_variant → per-fold training.train_moneyline_ensemble(train, val)
    (5 members), adaptive OOF-earned weights re-earned per variant
    (_LAST_ADAPTIVE_WEIGHTS cleared), prequential Platt on prior folds'
    pairs only, clip 1e-7. Sealed holdout: fit-only refit at the end +
    Platt fit on the FULL pooled OOF (strictly pre-holdout data).
  * Checkpoint-per-fold resume: an interrupted run resumes without
    recomputing finished folds (<out>.partial.json, deleted on completion).
  * FEATURE_COLS is NEVER assigned (both arms carry the identical 65-col
    layout — the only difference is the margin VALUES), so the served pool
    is untouched by construction.

Gate (task rule): C1 is the winner only if it beats C0 on the sealed
holdout logloss AND AUC with ECE-cal not degraded AND pooled OOF
corroboration. Production baselines (constant home edge, elo logistic) are
computed on the sealed holdout for the verdict rule.

Record: data_delivery/mlb_margin_k_ablation_<frame>.json (frame = data
hash), written after EACH arm. COMMITS NOTHING.

Usage:
    python run_mlb_margin_k_ablation.py              # both arms
    python run_mlb_margin_k_ablation.py --arms C0    # resume-friendly
    python run_mlb_margin_k_ablation.py --arms C0,C1
    python run_mlb_margin_k_ablation.py --limit-folds 4   # smoke / tests
    python run_mlb_margin_k_ablation.py --smoke      # 3 folds, /tmp, no gate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import training  # noqa: E402
from build_oof_margin import (  # noqa: E402
    MARGIN_COL,
    oof_run_margins,
    refit_run_margins,
)
from calibration import apply_platt, fit_platt, MIN_OOF_FOR_FIT  # noqa: E402
from config import (  # noqa: E402
    DATA_DELIVERY_DIR,
    MIN_VAL_FOLD_GAMES,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
)
from features import add_lineup_delta_features  # noqa: E402

EPS = 1e-7

# Run-engine fit_k_edge guards (mirror run_engine_k_edge.fit_k_edge).
K_MIN_PRIOR_ROWS = 100
K_MIN_EDGE_SD = 1e-9

# Reference point for the report only: the run engine's daily per-run k on
# its own OOF (challenger C2, ~1.49). NEVER used by this harness.
DAILY_K_REFERENCE = 1.49


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def head_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=_BACKEND_DIR.parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# k machinery (fold-local)
# ---------------------------------------------------------------------------
def fit_fold_k(prior_edge: np.ndarray, prior_margin: np.ndarray) -> float:
    """OLS slope of ACTUAL margin on the λ edge over strictly-prior rows.

    Guards mirror the run engine's fit_k_edge (n ≥ 100, edge sd ≥ 1e-9);
    otherwise k = 1.0 (identity) so early warm-up folds degrade to raw.
    """
    d = np.asarray(prior_edge, dtype=float)
    m = np.asarray(prior_margin, dtype=float)
    if len(d) < K_MIN_PRIOR_ROWS or np.std(d) < K_MIN_EDGE_SD:
        return 1.0
    return float(np.polyfit(d, m, 1)[0])


def per_fold_k(margins: pd.DataFrame, games: pd.DataFrame
               ) -> dict[int, float]:
    """Fit k per fold on the STRICTLY-PRIOR OOF rows only.

    For fold f: prior = OOF rows from executed folds < f whose game date is
    strictly before f's val window (the fold_idx < f condition alone is
    sufficient under walk-forward, the date guard is belt-and-braces). No
    row of fold f (or later) ever enters the fit; sealed rows do not exist
    in ``margins`` at all.

    Returns {fold_idx: k}. Deterministic (identical input → identical k).
    """
    m = margins.copy()
    # Rows without a fold (sealed/post-holdout games carry no OOF λ) can
    # never participate in a k fit by construction.
    m = m[m["fold_idx"].notna()].copy()
    if "game_date" not in m.columns:
        m = m.merge(games[["game_pk", "game_date"]].drop_duplicates("game_pk"),
                    on="game_pk", how="left")
    m["game_date"] = pd.to_datetime(m["game_date"])
    m["edge"] = m["lam_home"].to_numpy(float) - m["lam_away"].to_numpy(float)
    m["actual_margin"] = (
        m["home_score"].to_numpy(float) - m["away_score"].to_numpy(float)
        if "home_score" in m.columns and "away_score" in m.columns
        else np.nan)

    out: dict[int, float] = {}
    folds_sorted = sorted(int(f) for f in m["fold_idx"].unique())
    for f in folds_sorted:
        prior = m[m["fold_idx"] < f]
        va = m[m["fold_idx"] == f]
        if len(prior) == 0 or len(va) == 0 or va["game_date"].isna().any():
            out[f] = 1.0
            continue
        va_start = pd.Timestamp(va["game_date"].min())
        prior = prior[pd.to_datetime(prior["game_date"]) < va_start]
        prior = prior.dropna(subset=["edge", "actual_margin"])
        out[f] = fit_fold_k(prior["edge"].to_numpy(float),
                            prior["actual_margin"].to_numpy(float))
    return out


def k_expanded_margins(margins: pd.DataFrame, k_by_fold: dict[int, float],
                       ) -> pd.DataFrame:
    """C1 margin column: per-fold k × raw margin.

    Mathematically identical to the full two-sided λ expansion's DIFFERENCE
    (the C2 midpoint m cancels; the level/sum is preserved by construction —
    the margin only carries the difference). Tests assert the equivalence.
    """
    out = margins[["game_pk", "fold_idx"]].copy()
    out["k"] = margins["fold_idx"].map(k_by_fold).astype(float)
    out[MARGIN_COL] = np.round(
        out["k"].to_numpy(float) * margins[MARGIN_COL].to_numpy(float), 5)
    return out


# ---------------------------------------------------------------------------
# Data preparation (shared with run_margin_ablation's cache)
# ---------------------------------------------------------------------------
def prepare_data(holdout_days: int, limit_folds: int = 0):
    """Load + enrich, split holdout, build the leakage-free margin table on
    the moneyline folds (run engine READ-ONLY) — cached under
    /tmp/margin_oof_cache_<hash>.parquet with the SAME key as
    run_margin_ablation.prepare_data."""
    data_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    games = pd.read_csv(data_path)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_win"]).reset_index(drop=True)
    games = add_lineup_delta_features(games)

    cutoff = games["game_date"].max() - pd.Timedelta(days=holdout_days - 1)
    tune_df = games[games["game_date"] < cutoff].reset_index(drop=True)
    hold_df = games[games["game_date"] >= cutoff].reset_index(drop=True)
    all_splits = training.walk_forward_splits(
        tune_df, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
    folds = [s for s in all_splits if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]
    if limit_folds:
        folds = folds[:limit_folds]

    h = hashlib.sha256()
    h.update(sha256_file(data_path).encode())
    h.update(json.dumps([str(s["val_start"]) for s in folds]).encode())
    cache = (Path(tempfile.gettempdir())
             / f"margin_oof_cache_{h.hexdigest()[:16]}.parquet")

    if cache.exists():
        margins = pd.read_parquet(cache)
        meta = json.loads(cache.with_suffix(".meta.json").read_text())
        uncov = int(meta.pop("n_uncovered"))
        rounds = {k: int(v) for k, v in meta.items()}
    else:
        print("building OOF margins on the moneyline folds "
              "(run engine READ-ONLY) ...", flush=True)
        margins, _rounds_raw, uncov = oof_run_margins(tune_df, folds)
        rounds = {k: int(v) for k, v in _rounds_raw.items()}
        # k fitting needs the actual margins of the OOF rows — attach scores.
        margins = margins.merge(
            tune_df[["game_pk", "home_score", "away_score", "game_date"]]
            .drop_duplicates("game_pk"),
            on="game_pk", how="left")
        margins.to_parquet(cache)
        meta = {**{k: str(v) for k, v in rounds.items()},
                "n_uncovered": int(uncov)}
        cache.with_suffix(".meta.json").write_text(json.dumps(meta))

    hold_margins = refit_run_margins(tune_df, hold_df, rounds)

    tune_enriched = attach(tune_df, margins)
    enriched_splits = [
        s for s in training.walk_forward_splits(
            tune_enriched, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
        if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]
    if limit_folds:
        enriched_splits = enriched_splits[:limit_folds]
    assert len(enriched_splits) == len(folds) and all(
        a["fold_idx"] == b["fold_idx"]
        and pd.Timestamp(a["val_start"]) == pd.Timestamp(b["val_start"])
        and a["val_games"]["game_pk"].tolist()
        == b["val_games"]["game_pk"].tolist()
        for a, b in zip(folds, enriched_splits)), \
        "enriched-frame folds desynced from margin-build folds"
    return (games, tune_enriched, hold_df, enriched_splits, margins,
            hold_margins, rounds, uncov, h.hexdigest()[:16])


def attach(df: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    """Left-join margin by game_pk; uncovered rows stay NaN."""
    df = df.drop(columns=[MARGIN_COL] if MARGIN_COL in df.columns else []).copy()
    return df.merge(margins[["game_pk", MARGIN_COL]], on="game_pk", how="left")


def build_c1(tune_enriched: pd.DataFrame, hold_enriched: pd.DataFrame,
             margins: pd.DataFrame, k_by_fold: dict[int, float],
             folds: list[dict]):
    """C1 frames + folds: margin column = per-fold k × raw margin.

    Holdout margin is scaled by k_last (the final executed fold's k —
    strictly prior to the holdout window), the per-run analogue of the daily
    k. Base (non-margin) columns are byte-identical to C0's.
    """
    k_last = float(k_by_fold[max(k_by_fold)]) if k_by_fold else 1.0
    m1 = k_expanded_margins(margins, k_by_fold)

    tune_c1 = attach(tune_enriched, m1[["game_pk", MARGIN_COL]])
    c1_splits = [
        s for s in training.walk_forward_splits(
            tune_c1, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
        if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]
    # Mirror prepare_data's limit_folds truncation so geometry matches.
    c1_splits = c1_splits[:len(folds)]
    assert len(c1_splits) == len(folds) and all(
        a["fold_idx"] == b["fold_idx"]
        and pd.Timestamp(a["val_start"]) == pd.Timestamp(b["val_start"])
        and a["val_games"]["game_pk"].tolist()
        == b["val_games"]["game_pk"].tolist()
        for a, b in zip(folds, c1_splits)), \
        "C1 folds desynced from C0 folds"

    hold_c1 = hold_enriched.drop(
        columns=[MARGIN_COL] if MARGIN_COL in hold_enriched.columns else [])
    hold_c1 = hold_c1.copy()
    hold_c1[MARGIN_COL] = scale_hold_margin(hold_enriched, k_last)
    return tune_c1, hold_c1, c1_splits


def scale_hold_margin(hold_enriched: pd.DataFrame, k_last: float) -> pd.Series:
    """C1 sealed-holdout margin: raw refit margin scaled by the final
    executed fold's k (fitted strictly prior to the holdout window)."""
    hm = hold_enriched[MARGIN_COL].to_numpy(float)
    return pd.Series(np.round(k_last * hm, 5), index=hold_enriched.index)


# ---------------------------------------------------------------------------
# Full harness arm (mirrors run_margin_ablation.run_variant)
# ---------------------------------------------------------------------------
def run_variant(folds: list[dict], tune_df: pd.DataFrame, hold_df: pd.DataFrame,
                partial_path: Path | None = None) -> dict:
    """Full 5-member ensemble + adaptive blend over ``folds`` (carrying the
    arm's margin column), sealed-21d holdout, checkpoint-per-fold resume."""
    training._LAST_ADAPTIVE_WEIGHTS.clear()  # re-earn weights per arm

    done: dict[int, dict] = {}
    if partial_path and partial_path.exists():
        try:
            for rec in json.loads(partial_path.read_text()).get("folds", []):
                done[rec["fold_idx"]] = rec
        except Exception:
            done = {}

    def _flush(state: dict) -> None:
        if partial_path:
            partial_path.write_text(json.dumps(state))

    state: dict = {"folds": [done[i] for i in sorted(done)]}
    oof_y: list[float] = []
    oof_blend: list[float] = []
    oof_members: dict[str, list[float]] = {}
    oof_blend_cal: list[float] = []
    oof_members_cal: dict[str, list[float]] = {}

    def _accumulate(rec: dict) -> None:
        oof_y.extend(rec["y"])
        oof_blend.extend(rec["blend"])
        oof_blend_cal.extend(rec["blend_cal"])
        for name, plist in rec["members"].items():
            oof_members.setdefault(name, []).extend(plist)
        for name, plist in rec["members_cal"].items():
            oof_members_cal.setdefault(name, []).extend(plist)

    for rec in state["folds"]:
        _accumulate(rec)

    executed = len(done)
    for split in folds:
        fi = split["fold_idx"]
        if fi in done:
            continue
        train, val = split["train_games"], split["val_games"]
        try:
            models, _ = training.train_moneyline_ensemble(train, val)
        except Exception as e:
            print(f"  fold {fi} failed: {e}", flush=True)
            continue
        blend, member_probs, _wts = training.ensemble_predict(models, val)
        y_val = val["home_win"].values.astype(float)
        fold_cal = None
        if len(oof_y) >= MIN_OOF_FOR_FIT:
            fold_cal = fit_platt(np.asarray(oof_y), np.asarray(oof_blend))
        pa = {n: np.asarray(p, dtype=float) for n, p in member_probs.items()}
        rec = {
            "fold_idx": int(fi),
            "y": y_val.tolist(),
            "blend": np.asarray(blend, dtype=float).tolist(),
            "blend_cal": np.asarray(
                apply_platt(np.asarray(blend, dtype=float), fold_cal),
                dtype=float).tolist(),
            "members": {n: p.tolist() for n, p in pa.items()},
            "members_cal": {
                n: np.asarray(apply_platt(p, fold_cal), dtype=float).tolist()
                for n, p in pa.items()},
        }
        _accumulate(rec)
        done[fi] = rec
        state["folds"] = [done[i] for i in sorted(done)]
        _flush(state)
        executed += 1
        print(f"  fold {fi}: n_val={len(y_val)} blend_ll="
              f"{training.compute_metrics(np.array(oof_y), np.array(oof_blend))['logloss']:.4f}",
              flush=True)

    y_all = np.asarray(oof_y, dtype=float)
    pooled: dict[str, dict] = {}
    pooled["blend"] = training.compute_metrics(
        y_all, np.asarray(oof_blend, dtype=float))
    pooled["blend_calibrated"] = training.compute_metrics(
        y_all, np.asarray(oof_blend_cal, dtype=float))
    for name, plist in oof_members.items():
        pooled[name] = training.compute_metrics(
            y_all, np.asarray(plist, dtype=float))
        pooled[f"{name}_calibrated"] = training.compute_metrics(
            y_all, np.asarray(oof_members_cal.get(name, []), dtype=float))

    models, _ = training.train_moneyline_ensemble(tune_df)
    blend_hold, member_hold, _wts = training.ensemble_predict(models, hold_df)
    y_hold = hold_df["home_win"].values.astype(float)
    full_cal = fit_platt(y_all, np.asarray(oof_blend, dtype=float))
    holdout: dict[str, dict] = {
        "blend": training.compute_metrics(y_hold, np.asarray(blend_hold)),
        "blend_calibrated": training.compute_metrics(
            y_hold, np.asarray(apply_platt(
                np.asarray(blend_hold, dtype=float), full_cal))),
    }
    for name, p in member_hold.items():
        holdout[name] = training.compute_metrics(
            y_hold, np.asarray(p, dtype=float))

    if partial_path and partial_path.exists():
        partial_path.unlink()
    return {"folds_executed": executed, "pooled": pooled, "holdout": holdout}


def holdout_baselines(tune_df: pd.DataFrame, hold_df: pd.DataFrame) -> dict:
    """Constant home edge + elo logistic on the sealed holdout (fit on
    strictly-prior data only) — the production verdict-rule baselines."""
    y_hold = hold_df["home_win"].values.astype(float)
    home_rate = float(tune_df["home_win"].mean())
    const = np.full(len(y_hold), home_rate, dtype=float)
    const_metrics = training.compute_metrics(y_hold, const)
    elo = None
    if "elo_diff" in tune_df.columns and "elo_diff" in hold_df.columns:
        from sklearn.linear_model import LogisticRegression
        Xtr = tune_df[["elo_diff"]].to_numpy(float)
        Xho = hold_df[["elo_diff"]].to_numpy(float)
        Xtr = np.nan_to_num(Xtr, nan=np.nanmedian(Xtr))
        Xho = np.nan_to_num(Xho, nan=np.nanmedian(Xtr))
        elo_m = LogisticRegression(max_iter=2000).fit(
            Xtr, tune_df["home_win"].to_numpy(float))
        elo = elo_m.predict_proba(Xho)[:, 1]
        elo = np.clip(elo, EPS, 1 - EPS)
        elo_metrics = training.compute_metrics(y_hold, elo)
    else:
        elo_metrics = None
    return {"constant_home_edge": const_metrics, "elo_logistic": elo_metrics}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", type=str, default="C0,C1")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--holdout-days", type=int, default=21)
    ap.add_argument("--limit-folds", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="3 folds, output to /tmp, gate skipped")
    args = ap.parse_args()
    if args.smoke:
        args.limit_folds = min(args.limit_folds or 3, 3)
        args.out = Path("/tmp/mlb_margin_k_ablation_smoke.json")
        args.arms = "C0,C1"

    sha = head_sha()
    (games, tune_enriched, hold_df, folds, margins, hold_margins,
     rounds, uncov, frame_hash) = prepare_data(args.holdout_days,
                                               args.limit_folds)

    # Fold-local k (strictly-prior training portion only). k stats vs the
    # run engine's daily k (~1.49) are reported, never used.
    k_by_fold = per_fold_k(margins, tune_enriched)
    ks = np.array([v for v in k_by_fold.values()], dtype=float)
    k_stats = {"min": float(ks.min()), "max": float(ks.max()),
               "mean": float(ks.mean()), "sd": float(ks.std()),
               "n_folds_fit": int((ks != 1.0).sum()),
               "identity_warmup_folds": int((ks == 1.0).sum()),
               "fit_rule": "strictly-prior OOF rows (fold_idx<f, date-guarded) "
                           "per moneyline fold; n>=100, sd>=1e-9 else 1.0",
               "daily_run_k_reference": DAILY_K_REFERENCE}

    covered = float(tune_enriched[MARGIN_COL].notna().mean())
    hold_enriched = attach(hold_df, hold_margins)
    tune_c1, hold_c1, c1_splits = build_c1(
        tune_enriched, hold_enriched, margins, k_by_fold, folds)

    print(f"commit={sha[:12]} frame={frame_hash} games={len(games)} "
          f"tuning={len(tune_enriched)} holdout={len(hold_df)} "
          f"folds={len(folds)} seed={RANDOM_SEED} clip={EPS}")
    print(f"margin coverage: tuning {covered:.1%} | "
          f"uncovered_tune_games={uncov}")
    print(f"median fold rounds: home={rounds.get('home')} "
          f"away={rounds.get('away')}")
    print(f"per-fold k: min={k_stats['min']:.3f} max={k_stats['max']:.3f} "
          f"mean={k_stats['mean']:.3f} sd={k_stats['sd']:.3f} "
          f"| daily-run k ref ~{DAILY_K_REFERENCE} (never used)")

    target = pd.Timestamp.now().date().isoformat().replace("-", "")
    out = args.out or (DATA_DELIVERY_DIR /
                       f"mlb_margin_k_ablation_{frame_hash}.json")
    if out.exists():
        record = json.loads(out.read_text())
    else:
        record = {"schema": "mlb-margin-k-ablation/v1",
                  "commit_sha": sha,
                  "frame": frame_hash,
                  "data_sha256": sha256_file(
                      DATA_DELIVERY_DIR / "game_level_features.csv"),
                  "holdout_days": args.holdout_days,
                  "folds_executed": len(folds),
                  "margin_col": MARGIN_COL,
                  "margin_source": "run_engine per-side Poisson, OOF on the "
                                  "moneyline's own folds (read-only)",
                  "k_edge_c2": "run_engine_k_edge.py daily k NOT used; "
                               "fold-local k per moneyline fold",
                  "margin_coverage_tuning": round(covered, 4),
                  "uncovered_tune_games": int(uncov),
                  "median_fit_rounds": {k: int(v) for k, v in rounds.items()},
                  "k_stats": k_stats,
                  "k_by_fold": {int(f): round(v, 4)
                                for f, v in k_by_fold.items()},
                  "clip_eps": EPS, "seed": int(RANDOM_SEED),
                  "arms": {}}
        out.write_text(json.dumps(record, indent=2) + "\n")

    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    for name in want:
        if name in record["arms"]:
            print(f"  arm {name} already recorded — skipping")
            continue
        if name == "C0":
            tune, hold, arm_folds = tune_enriched, hold_enriched, folds
        elif name == "C1":
            tune, hold, arm_folds = tune_c1, hold_c1, c1_splits
        else:
            raise SystemExit(f"unknown arm {name!r}")
        print(f"  {name}: running ({len(arm_folds)} folds, "
              f"k_expanded={name == 'C1'}) ...", flush=True)
        r = run_variant(arm_folds, tune, hold,
                        partial_path=Path(str(out) + f".{name}.partial.json"))
        record["arms"][name] = r
        out.write_text(json.dumps(record, indent=2) + "\n")
        b, h = r["pooled"]["blend"], r["holdout"]["blend"]
        print(f"    pooled blend {b['logloss']:.4f}/{b['auc']:.4f} "
              f"brier {b['brier']:.4f} ece {b['ece']:.4f} | "
              f"holdout {h['logloss']:.4f}/{h['auc']:.4f} "
              f"ece {h['ece']:.4f}", flush=True)

    baselines = holdout_baselines(tune_enriched, hold_enriched)

    if (not args.smoke and "C0" in record["arms"] and "C1" in record["arms"]):
        c0, c1 = record["arms"]["C0"], record["arms"]["C1"]
        h0, h1 = c0["holdout"], c1["holdout"]
        b0, b1 = h0["blend"], h1["blend"]
        cal0 = h0.get("blend_calibrated", b0)
        cal1 = h1.get("blend_calibrated", b1)
        p0, p1 = c0["pooled"]["blend"], c1["pooled"]["blend"]
        hold_gain = (b1["logloss"] < b0["logloss"] and b1["auc"] > b0["auc"])
        cal_ok = cal1["ece"] <= cal0["ece"]
        pooled_ok = (p1["logloss"] <= p0["logloss"] and p1["auc"] >= p0["auc"])
        beats_baselines = (
            baselines.get("constant_home_edge")
            and baselines.get("elo_logistic")
            and b1["logloss"] < baselines["constant_home_edge"]["logloss"]
            and b1["auc"] > baselines["constant_home_edge"]["auc"]
            and b1["logloss"] < baselines["elo_logistic"]["logloss"]
            and b1["auc"] > baselines["elo_logistic"]["auc"])
        verdict = ("WINS" if (hold_gain and cal_ok and pooled_ok)
                   else "NULL (no adoption)")
        record["gate"] = {
            "verdict": verdict,
            "rule": "C1 wins only on sealed logloss AND AUC, ECE-cal not "
                    "degraded, pooled corroboration; baselines: const + elo",
            "holdout_c0": b0, "holdout_c1": b1,
            "holdout_c0_cal": cal0, "holdout_c1_cal": cal1,
            "pooled_c0": p0, "pooled_c1": p1,
            "baselines": {k: v for k, v in baselines.items() if v},
            "delta_holdout_logloss": round(
                b1["logloss"] - b0["logloss"], 5),
            "delta_holdout_auc": round(b1["auc"] - b0["auc"], 4),
            "delta_holdout_ece_cal": round(
                cal1["ece"] - cal0["ece"], 4),
            "delta_pooled_logloss": round(
                p1["logloss"] - p0["logloss"], 5),
            "delta_pooled_auc": round(p1["auc"] - p0["auc"], 4),
            "beats_baselines": bool(beats_baselines),
            "holdout_gain": bool(hold_gain),
            "holdout_cal_ok": bool(cal_ok),
            "pooled_ok": bool(pooled_ok),
        }
        record["verdict"] = verdict
        out.write_text(json.dumps(record, indent=2) + "\n")
        print("\n================= GATE (C1 vs C0) =================")
        print(f"sealed holdout: Δlogloss={record['gate']['delta_holdout_logloss']:+.5f} "
              f"ΔAUC={record['gate']['delta_holdout_auc']:+.4f} "
              f"ΔECE-cal={record['gate']['delta_holdout_ece_cal']:+.4f}")
        print(f"  holdout logloss AND AUC better : {hold_gain}")
        print(f"  holdout ECE-cal not degraded   : {cal_ok}")
        print(f"  pooled OOF not lost            : {pooled_ok}")
        print(f"  beats const+elo baselines      : {bool(beats_baselines)}")
        print(f"→ {verdict}")


if __name__ == "__main__":
    main()