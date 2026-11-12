"""Production pricer for the frozen NBA NB challenger (w_prod v1).

Replaces the split-skew/lattice distribution family for NBA pricing. The
engine has four stages, matching the frozen research build exactly:

  step 1  per-side score means (mu_h, mu_a) supplied by the caller — the
          distribution family's own inputs, unchanged;
  step 2  total residual  K_R = (T - (mu_h+mu_a)) + C_R, LGBM NB2 with the
          A-frame matrix plus mu_h/mu_a, dispersion r_R;
  step 3  margin residual K_E = (M - (mu_h-mu_a)) + C_E, same recipe, r_E;
  step 4  50k Monte Carlo draws per game, structural OT leg rule, then the
          frozen two-layer count-shrunk margin weights (w_prod) applied as
          w_norm. Calibration layer is identity (platt_v1 disabled).

Weights are NOT an artifact. Every run refits them in this process as
cumulative per-slate-date tables over strictly prior settled games
(walk-forward OOF history), applies them point-in-time pre-game, and then
discards them. Nothing here reads a stored weight table, and no weight table
is ever written to disk or committed to the repository. Calibration is
identity (the Platt layer is disabled by verdict).

Model id: ``nba_nb_hier_wprod_v1``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("nba_hier_mc")

MODEL_ID = "nba_nb_hier_wprod_v1"
C_R = 100.0
C_E = 60.0
N_MC = 50_000
MC_SEED = 20261111
GRID = np.arange(-110, 111)
TMIN, TMAX = 0, 400
OT_DEFAULTS = {"d_m1": 0.15, "d_p1": 0.28, "w_lo": 2, "w_hi": 8}
BUNDLE_NAME = "nba_hier_mc_latest.joblib"


# ------------------------------------------------- methodology (not data)
# k_trend / k_spike were selected by the expanding-window OOF verdict on the
# exact-margin grid. These are methodology constants: the per-game weights
# themselves are always computed fresh from OOS history inside this process
# (build_weight_tables) and are NEVER serialized to disk, committed, or
# shipped. There is no stored weight artifact anywhere in this repository.
K_TR = 150.0
K_SP = 200.0
#: Outer-tail taper: exact empirical weights to |m| = 25, blended to 1.0 by
#: |m| = 30, locked at 1.0 beyond. Counts below SPARSE_N are locked too.
TAIL_TAPER_START = 25
TAIL_TAPER_STOP = 30
SPARSE_N = 10
IDENTITY_CALIBRATION = {"mode": "identity", "a": 0.0, "b": 1.0}


def _two_layer(nm, pa, pb, k_sp):
    """Count-shrunk frequency-ratio reweighting (two layers, vendored copy).

    Vendored here so production never imports untracked experiment scripts:
    w = w_trend * w_spike, each layer 1 + n/(n+k)*(pa/pb - 1) with a 5-bin
    local trend (k_trend) and a single-integer spike (k_spike); w(0) = 0.
    """
    L = len(nm)
    r = np.ones(L)
    nz = pb > 0
    r[nz] = pa[nz] / np.maximum(pb[nz], 1e-12)
    r_smooth = np.ones(L)
    n_smooth = np.zeros(L)
    for i in range(L):
        if i == 110:
            continue
        sa = sb = sn = 0.0
        for d in (-2, -1, 0, 1, 2):
            j = i + d
            if j < 0 or j >= L or j == 110:
                continue
            sa += pa[j]
            sb += pb[j]
            sn += nm[j]
        r_smooth[i] = sa / sb if sb > 0 else 1.0
        n_smooth[i] = sn
    w_tr = 1.0 + (n_smooth / (n_smooth + K_TR)) * (r_smooth - 1.0)
    p_tr = pb * w_tr
    rs2 = np.ones(L)
    nz2 = p_tr > 0
    rs2[nz2] = pa[nz2] / np.maximum(p_tr[nz2], 1e-12)
    w_sp = 1.0 + (nm / (nm + k_sp)) * (rs2 - 1.0)
    w = w_tr * w_sp
    both0 = (nm == 0) & (n_smooth == 0)
    w[both0] = 1.0
    w[110] = 0.0
    w[~np.isfinite(w)] = 1.0
    return np.clip(w, 0, None), w_tr, w_sp


def clamp_outer_tails(w: np.ndarray, nm: np.ndarray) -> np.ndarray:
    """Strict outer-tail taper: blowout margins are governed by the 4-NB tails.

    The empirical layer is a count-shrunk frequency ratio, so a handful of
    historical games at an extreme margin can multiply a bin several times
    over (a 4.74x spike at +/-55 came from two games). Inside |m| <= 25 the
    two-layer weights are kept exactly as fit; from 25 to 30 they are blended
    back to 1 by alpha = (30 - |m|) / 5, and beyond 30 every bin is locked at
    1.000. Bins with fewer than SPARSE_N observations are locked the same way,
    so thin cells cannot invent a spike anywhere in the support.
    """
    w = np.array(w, dtype=float, copy=True)
    nm = np.asarray(nm, dtype=float)
    grid = np.arange(len(w)) - 110
    am = np.abs(grid)
    alpha = np.clip((TAIL_TAPER_STOP - am) / (TAIL_TAPER_STOP - TAIL_TAPER_START),
                    0.0, 1.0)
    w = 1.0 + alpha * (w - 1.0)
    w[(am > TAIL_TAPER_STOP) | (nm < SPARSE_N)] = 1.0
    w[110] = 0.0
    return np.clip(w, 0, None)


def _ot_rule():
    import config
    rule = dict(OT_DEFAULTS)
    p = Path(config.RUN_DIAGNOSTICS_DIR) / "ot_rule.json"
    if p.exists():
        try:
            rule.update(json.loads(p.read_text()))
        except Exception:  # noqa: BLE001 - defaults are the contract
            logger.warning("ot_rule.json unreadable; using defaults")
    return rule


# ------------------------------------------------------------------- models
def _features(frame: pd.DataFrame, mu_h: np.ndarray, mu_a: np.ndarray):
    import moneyline as ml_mod
    X = ml_mod.member_fit_input(
        "lightgbm", ml_mod.member_matrix("lightgbm", frame), None)
    if not hasattr(X, "columns"):
        raise ValueError("LGBM member matrix is not a frame")
    X = X.copy()
    X["mu_h"] = mu_h
    X["mu_a"] = mu_a
    return X


def fit_step1(frame: pd.DataFrame, progress_every: int = 0,
              r_override: dict | None = None):
    """Step 1: per-side NB score models (the frozen challenger's baseline arm)."""
    import config
    from distributions import (NegativeBinomialObjective, NB_RAW_CLIP,
                               estimate_nb_dispersion)
    from lightgbm import LGBMRegressor

    h = pd.to_numeric(frame.home_score, errors="coerce").to_numpy(float)
    a = pd.to_numeric(frame.away_score, errors="coerce").to_numpy(float)
    ok = np.isfinite(h) & np.isfinite(a)
    X = _features(frame.loc[ok], np.full(ok.sum(), np.nan),
                  np.full(ok.sum(), np.nan))
    X = X.drop(columns=[c for c in ("mu_h", "mu_a") if c in X.columns])
    cat = [c for c in config.TREE_CATEGORICAL_COLS if c in X.columns]
    out = {}
    for side, y in (("home", h[ok]), ("away", a[ok])):
        r = float((r_override or {}).get(f"r_{side[0]}")
                  or estimate_nb_dispersion(y))
        init = float(np.log(max(y.mean(), 1e-6)))
        model = LGBMRegressor(**{**config.LIGHTGBM_REG_PARAMS,
                                 "objective": NegativeBinomialObjective(r)})
        fit_kw = {"init_score": np.full(len(X), init)}
        if cat:
            fit_kw["categorical_feature"] = cat
        model.fit(X, y, **fit_kw)
        out[side] = {"model": model, "r": float(r), "init": init,
                     "NB_RAW_CLIP": NB_RAW_CLIP}
        if progress_every:
            logger.info("step1 %s fitted on %d rows (r=%.1f)", side, len(y), r)
    return {"step1": out, "columns": list(X.columns), "n_fit": int(ok.sum())}


def predict_step1(frame: pd.DataFrame, bundle: dict):
    X = _features(frame, np.zeros(len(frame)), np.zeros(len(frame)))
    X = X.drop(columns=[c for c in ("mu_h", "mu_a") if c in X.columns])
    X = X.reindex(columns=bundle["step1_columns"])
    mus = {}
    for side in ("home", "away"):
        s = bundle["step1"][side]
        raw = np.asarray(s["model"].predict(X), dtype=float)
        mus[side] = np.exp(np.clip(raw + s["init"], *s["NB_RAW_CLIP"]))
    return mus["home"], mus["away"]


def frozen_dispersions():
    """Dispersions from the validated OOF run (structural, not refit).

    The frozen weights and the validated spread assume the OOF-fitted
    dispersions; refitting them in-sample under-disperses the price.
    """
    import config
    p = Path(config.RUN_DIAGNOSTICS_DIR) / "ab_hier_mc_oof.parquet"
    if not p.exists():
        return None
    oof = pd.read_parquet(p, columns=["r_R", "r_E", "r_h", "r_a"])
    return {c: float(oof[c].median()) for c in ("r_R", "r_E", "r_h", "r_a")}


def fit_models(frame: pd.DataFrame, mu_h, mu_a, progress_every: int = 0,
               r_override: dict | None = None):
    """Fit the step-2/step-3 residual models on settled rows."""
    import config
    from distributions import (NegativeBinomialObjective, NB_RAW_CLIP,
                               estimate_nb_dispersion)
    from lightgbm import LGBMRegressor

    h = pd.to_numeric(frame.home_score, errors="coerce").to_numpy(float)
    a = pd.to_numeric(frame.away_score, errors="coerce").to_numpy(float)
    muh = np.asarray(mu_h, float)
    mua = np.asarray(mu_a, float)
    ok = np.isfinite(h) & np.isfinite(a) & np.isfinite(muh) & np.isfinite(mua)
    X = _features(frame.loc[ok], muh[ok], mua[ok])
    cat = [c for c in config.TREE_CATEGORICAL_COLS if c in X.columns]
    out = {}
    for tag, kind, c_init in (("KR", "total", C_R), ("KE", "margin", C_E)):
        y = ((h[ok] + a[ok]) - (muh[ok] + mua[ok]) + C_R if kind == "total"
             else (h[ok] - a[ok]) - (muh[ok] - mua[ok]) + C_E)
        if not np.all(y > 0):
            raise ValueError(f"{tag}: shifted residual must be strictly positive")
        r = float((r_override or {}).get(tag) or estimate_nb_dispersion(y))
        model = LGBMRegressor(**{**config.LIGHTGBM_REG_PARAMS,
                                 "objective": NegativeBinomialObjective(r)})
        fit_kw = {"init_score": np.full(len(X), np.log(c_init))}
        if cat:
            fit_kw["categorical_feature"] = cat
        model.fit(X, y, **fit_kw)
        out[tag] = {"model": model, "r": float(r), "c_init": float(c_init),
                    "kind": kind, "columns": list(X.columns),
                    "n_fit": int(len(y)), "NB_RAW_CLIP": NB_RAW_CLIP}
        if progress_every:
            logger.info("fitted %s on %d rows (r=%.1f)", tag, len(y), r)
    return {"models": out, "model_id": MODEL_ID,
            "feature_columns": list(X.columns), "fit_n": int(len(X))}


def predict_residuals(frame: pd.DataFrame, mu_h, mu_a, bundle: dict):
    rows = {"mu_KR": np.full(len(frame), np.nan), "mu_KE": np.full(len(frame), np.nan)}
    X = _features(frame, np.asarray(mu_h, float), np.asarray(mu_a, float))
    for tag, key in (("KR", "mu_KR"), ("KE", "mu_KE")):
        spec = bundle["models"][tag]
        Xi = X.reindex(columns=spec["columns"])
        raw = np.asarray(spec["model"].predict(Xi), dtype=float)
        rows[key] = np.exp(np.clip(raw + np.log(spec["c_init"]),
                                  *spec["NB_RAW_CLIP"]))
    return (rows["mu_KR"], rows["mu_KE"],
            float(bundle["models"]["KR"]["r"]),
            float(bundle["models"]["KE"]["r"]))


def _fit_core(frame: pd.DataFrame, progress_every: int = 0):
    """Fit the three model stages only (no weight tables).

    Split out so the walk-forward history can refit models per block without
    recursively rebuilding the tables inside every block fit.
    """
    disp = frozen_dispersions() or {}
    st1 = fit_step1(frame, progress_every=progress_every, r_override=disp)
    mu_h, mu_a = predict_step1(frame, {**st1, "step1_columns": st1["columns"]})
    res = fit_models(frame, mu_h, mu_a, progress_every=progress_every,
                     r_override={"KR": disp.get("r_R"), "KE": disp.get("r_E")})
    return {"step1": st1["step1"], "step1_columns": st1["columns"],
            "models": res["models"], "feature_columns": res["feature_columns"],
            "fit_n": res["fit_n"]}


def fit_bundle(frame: pd.DataFrame, progress_every: int = 0):
    """Fit all three model stages plus the cumulative weight tables."""
    core = _fit_core(frame, progress_every=progress_every)
    res = {"models": core["models"], "feature_columns": core["feature_columns"],
           "fit_n": core["fit_n"]}
    partial = {"step1": core["step1"], "step1_columns": core["step1_columns"],
               "models": core["models"]}
    tables = build_weight_tables(frame, partial, progress_every=progress_every)
    if tables.get("final") is not None:
        _TABLES_CACHE[int(res["fit_n"])] = tables
    return {"model_id": MODEL_ID,
            "step1": core["step1"], "step1_columns": core["step1_columns"],
            "models": res["models"], "feature_columns": res["feature_columns"],
            "fit_n": res["fit_n"], "trained_utc": pd.Timestamp.utcnow().isoformat(),
            # Runtime-only: rebuilt point-in-time every run, never serialized.
            "weight_tables": tables,
            "calibration": dict(IDENTITY_CALIBRATION),
            "weight_artifact_policy": "runtime_only_never_serialized_never_committed",
            "k_trend": K_TR, "k_spike": K_SP}


def _draw_game(mu_h, mu_a, mu_kr, mu_ke, r_r, r_e, rng, rule, n_draws):
    """One game's 50k (home, away) draws after the structural OT rule."""
    def draw(k):
        rs = _nb_draw(rng, mu_kr, r_r, k).astype(float) - C_R
        es = _nb_draw(rng, mu_ke, r_e, k).astype(float) - C_E
        hh = np.clip(np.rint(mu_h + 0.5 * rs + 0.5 * es), 0, None).astype(np.int32)
        aa = np.clip(np.rint(mu_a + 0.5 * rs - 0.5 * es), 0, None).astype(np.int32)
        return hh, aa

    hh, aa = draw(n_draws)
    return _apply_ot(hh, aa, rng, rule, lambda k: draw(k))


_HIST_CACHE: dict = {}
# Parallel to _HIST_CACHE (same key): per-game strictly-prior
# mus from the same block refits, for point-in-time OOF pricing.
_MUS_CACHE: dict = {}


MIN_HISTORY_ROWS = 1200
HISTORY_BLOCKS = 6


def _walk_key(frame: pd.DataFrame, order, dates):
    """Identity of a walk: size, span, and BOTH end game ids.

    Size + dates alone collide across fixtures (two frames of the same
    length ending the same day but holding different games); a stale cache
    hit would then hand one frame's strictly-prior mus to another.
    """
    def _gid(pos):
        if "game_id" not in frame.columns:
            return ""
        return str(frame["game_id"].iloc[int(pos)])

    return (len(frame), len(order),
            str(pd.Timestamp(dates.iloc[order[-1]]).date()),
            _gid(order[0]), _gid(order[-1]))


def _walkforward_history(frame: pd.DataFrame, ym, dates, rule,
                        progress_every: int = 0):
    """Out-of-sample mus for the settled history, built inside this process.

    The weight tables are frequency ratios between actual margins and the
    simulated margins of the same games, so those simulations must come from
    models that never saw the games they score. The settled frame is cut into
    contiguous date blocks; each block is predicted by a bundle fitted ONLY on
    the blocks before it. Nothing is read from disk: a fresh clone with
    re-pulled data builds byte-identical tables, and no OOF file has to exist
    in the repository for the weights to be leakage-free.

    Returns ``(hist, ym, dates, source)`` restricted to the scored rows, or
    ``None`` when there is too little settled history to score anything.
    """
    if "gameday" not in frame.columns:
        logger.warning("weight tables: no gameday column; pricing unweighted")
        return None
    settled = np.flatnonzero(np.isfinite(ym) & dates.notna().to_numpy())
    if len(settled) < MIN_HISTORY_ROWS:
        logger.warning("weight tables: %d settled rows (< %d); pricing "
                       "unweighted", len(settled), MIN_HISTORY_ROWS)
        return None
    order = settled[np.argsort(dates.to_numpy()[settled], kind="stable")]
    key = _walk_key(frame, order, dates)
    cached = _HIST_CACHE.get(key)
    if cached is not None:
        logger.info("weight tables: reusing the in-process walk-forward "
                    "history (%d games)", len(cached[0]))
        return cached

    # Work on a chronologically ordered copy: block slices must be contiguous
    # and every order-dependent feature (rolling / EWM) must see history in
    # time order, or the replay silently degrades to NaNs.
    ordered = frame.iloc[order].reset_index(drop=True)
    seed = max(600, len(ordered) // 10)
    edges = np.linspace(seed, len(ordered), HISTORY_BLOCKS + 1).astype(int)
    min_train = 500
    mus = np.full((len(ordered) - seed, 6), np.nan)
    for i in range(len(edges) - 1):
        tr = ordered.iloc[:edges[i]]
        va = ordered.iloc[edges[i]:edges[i + 1]]
        if len(va) == 0 or len(tr) < min_train:
            continue
        fb = _fit_core(tr)
        mu_h, mu_a = predict_step1(va, fb)
        mu_kr, mu_ke, r_r, r_e = predict_residuals(va, mu_h, mu_a, fb)
        # predict_residuals returns per-fold dispersion constants as scalars;
        # broadcast them to the block so the column stack stays rectangular.
        rr = np.broadcast_to(np.asarray(r_r, float), (len(va),))
        re_b = np.broadcast_to(np.asarray(r_e, float), (len(va),))
        mus[edges[i] - seed:edges[i + 1] - seed] = np.column_stack(
            [mu_h, mu_a, mu_kr, mu_ke, rr, re_b])
        if progress_every:
            logger.info("weight tables: walk-forward block %d/%d (%d games)",
                        i + 1, len(edges) - 1, len(va))

    ok = np.isfinite(mus).all(axis=1)
    if not ok.any():
        logger.warning("weight tables: walk-forward replay produced no out-of-"
                       "sample mus; pricing unweighted")
        return None
    hist = np.zeros((int(ok.sum()), len(GRID)))
    for j, i in enumerate(np.flatnonzero(ok)):
        rng = np.random.default_rng(MC_SEED + j)
        hs, aa = _draw_game(*[float(v) for v in mus[i]], rng, rule, N_MC)
        hist[j] = np.bincount(np.clip(hs - aa, -110, 110) + 110,
                              minlength=len(GRID)) / N_MC
    out = (hist, ym[order[seed:][ok]], dates.iloc[order[seed:][ok]],
           "walkforward_inprocess_blocks")
    # Per-game strictly-prior mus for point-in-time OOF pricing: row j was
    # scored by a bundle fitted only on blocks before its own. Keyed by
    # game_id so the production walk can join them onto its origins.
    # _MUS_CACHE shares _HIST_CACHE's key: same walk, same coverage.
    try:
        _gids = ordered["game_id"].astype(str).to_numpy()[seed:][ok]
        _MUS_CACHE[key] = pd.DataFrame(
            {"game_id": _gids, "mu_h": mus[ok, 0], "mu_a": mus[ok, 1],
             "mu_KR": mus[ok, 2], "mu_KE": mus[ok, 3],
             "r_R": mus[ok, 4], "r_E": mus[ok, 5]})
    except (KeyError, IndexError, ValueError) as exc:
        # Never a silent bundle fallback: without keys the store is empty,
        # so settled pricing refuses instead of leaking.
        logger.warning("weight tables: PIT mus could not be keyed (%s); "
                       "settled pricing will refuse this walk", exc)
        _MUS_CACHE[key] = None
    logger.info("weight tables: walk-forward history replayed in process "
                "(%d games, %d blocks)", int(ok.sum()), len(edges) - 1)
    _HIST_CACHE[key] = out
    return out


# Prepared PIT mus for the run's settled frame (see prepare_pit_mus):
# the production walk prices origins out of this frame, so per-origin
# refits would both explode runtime and re-split history. One walk, one
# frame, one store per process -- the same convention as _TABLES_CACHE.
_PREPARED_PIT_MUS: pd.DataFrame | None = None


def prepare_pit_mus(frame: pd.DataFrame, progress_every: int = 0):
    """Run the block-refit walk over a settled frame; cache per-game PIT mus.

    Returns one row per scored game (game_id, mu_h, mu_a, mu_KR, mu_KE,
    r_R, r_E). Games without sufficient prior history (pre-seed rows) are
    absent,
    never infilled: callers must exclude or refuse them, never price them
    with full-frame mus. Idempotent in-process under the walk's own key.
    """
    global _PREPARED_PIT_MUS
    rule = _ot_rule()
    if "gameday" in frame.columns:
        dates = pd.to_datetime(frame["gameday"]).dt.normalize()
    else:
        dates = pd.Series(pd.NaT, index=frame.index)
    if {"home_score", "away_score"}.issubset(frame.columns):
        ym = (pd.to_numeric(frame["home_score"], errors="coerce").to_numpy(float)
              - pd.to_numeric(frame["away_score"], errors="coerce").to_numpy(float))
    else:
        _PREPARED_PIT_MUS = None
        return None
    out = _walkforward_history(frame, ym, dates, rule, progress_every)
    if out is None:
        _PREPARED_PIT_MUS = None
        return None
    settled = np.flatnonzero(np.isfinite(ym) & dates.notna().to_numpy())
    order = settled[np.argsort(dates.to_numpy()[settled], kind="stable")]
    key = _walk_key(frame, order, dates)
    _PREPARED_PIT_MUS = _MUS_CACHE.get(key)
    return _PREPARED_PIT_MUS


def lookup_pit_mus(game_ids) -> pd.DataFrame | None:
    """Join prepared PIT mus onto game ids (NaN rows where uncovered).

    Pure lookup: never fits. Returns None when no walk was prepared -- and
    also when NONE of the requested ids belong to the prepared walk, which
    means the store came from a different frame (another run, a fixture, a
    recompute against a bundle). A prepared walk never misses an origin
    wholesale: the production prologue drops pre-coverage rows before the
    block split, so every origin it hands over overlaps the store. Stale
    stores therefore abstain instead of raising a false inconsistency.
    """
    if _PREPARED_PIT_MUS is None:
        return None
    ids = [str(g) for g in game_ids]
    if not ids:
        return None
    if not _PREPARED_PIT_MUS["game_id"].isin(ids).any():
        return None
    want = pd.DataFrame({"game_id": ids})
    return want.merge(_PREPARED_PIT_MUS, on="game_id", how="left")


def _history_for_tables(frame: pd.DataFrame, bundle: dict, ym, dates, rule,
                        progress_every: int = 0):
    """Leakage-free margin history for the cumulative tables (see above)."""
    return _walkforward_history(frame, ym, dates, rule, progress_every)


def build_weight_tables(frame: pd.DataFrame, bundle: dict,
                        progress_every: int = 0):
    """Cumulative date-ordered weight tables (one per settled slate date).

    Production must not price a game with weights that have seen that game.
    For every date d the table is fit ONLY on games before d (cumulative OOS
    history), from their simulated (unweighted) margin histograms and actual
    margins. Rows dated before the first usable table, and rows with no date,
    price unweighted (w = 1) - there is no stored table to fall back to.
    Serving dates use ``final`` (all scored rows).
    """
    k_sp = K_SP
    rule = _ot_rule()
    n = len(frame)
    dates = (pd.to_datetime(frame["gameday"]).dt.normalize()
             if "gameday" in frame.columns else pd.Series(pd.NaT, index=frame.index))
    if "margin_actual" in frame.columns:
        ym = pd.to_numeric(frame["margin_actual"], errors="coerce").to_numpy(float)
    elif {"home_score", "away_score"}.issubset(frame.columns):
        ym = (pd.to_numeric(frame["home_score"], errors="coerce").to_numpy(float)
              - pd.to_numeric(frame["away_score"], errors="coerce").to_numpy(float))
    else:
        # No realised margins on this frame (e.g. a live slate): nothing to
        # fit against, so pricing runs unweighted.
        ym = np.full(len(frame), np.nan)
    # Simulated histograms must come from models that never saw the games
    # being priced, so the walk-forward OOF store is preferred: its mus are
    # fold-trained. Only when that store is unavailable does the fit fall
    # back to this run's in-sample models (logged, and NOT leakage-free).
    hist_src = _history_for_tables(frame, bundle, ym, dates, rule, progress_every)
    if hist_src is None:
        return {"cutoffs": [], "tables": [], "final": None,
                "k_trend": K_TR, "k_spike": K_SP, "n_dates": 0,
                "n_table_dates": 0,
                "history_source": "identity_insufficient_history",
                "serialized": False}
    hist, ym, dates, hist_source = hist_src
    n = len(hist)

    def _fit(mask):
        nm = np.bincount(np.clip(np.rint(ym[mask]), -110, 110).astype(int) + 110,
                         minlength=len(GRID)).astype(float)
        if nm.sum() <= 0:
            return None
        pa = nm / nm.sum()
        pb = hist[mask].mean(axis=0)
        pb = pb / pb.sum()
        w, _, _ = _two_layer(nm, pa, pb, k_sp)
        return clamp_outer_tails(w, nm)

    usable_dates = pd.unique(dates.dropna())
    usable_dates = np.sort(np.asarray(usable_dates))
    cutoffs, tables = [], []
    for d in usable_dates:
        prior = np.flatnonzero((dates < d).to_numpy()
                               & np.isfinite(ym))
        if len(prior) < 500:
            continue
        cutoffs.append(str(pd.Timestamp(d).date()))
        tables.append(_fit(prior))
    final = _fit(np.asarray(np.isfinite(ym)) & dates.notna().to_numpy())
    return {"cutoffs": cutoffs, "tables": tables, "final": final,
            "k_trend": K_TR, "k_spike": k_sp,
            "tail_taper": {"start": TAIL_TAPER_START, "stop": TAIL_TAPER_STOP,
                           "sparse_n": SPARSE_N},
            "n_dates": int(len(usable_dates)),
            "n_table_dates": int(len(cutoffs)),
            "history_source": hist_source,
            "serialized": False}


def _row_weights(frame: pd.DataFrame, bundle: dict):
    """Per-row weights: the table fit strictly before that row's date.

    Rows with no applicable table price UNWEIGHTED (w = 1): there is no
    stored table anywhere to fall back to, by design.
    """
    tables = bundle.get("weight_tables") or {}
    cutoffs = tables.get("cutoffs") or []
    per = tables.get("tables") or []
    final = tables.get("final")
    if final is None or not cutoffs:
        return np.ones((len(frame), len(GRID)))
    cut = pd.to_datetime(pd.Series(cutoffs))
    if "gameday" in frame.columns:
        dates = pd.to_datetime(frame["gameday"]).dt.normalize()
    else:
        dates = pd.Series(pd.NaT, index=frame.index)
    out = np.empty((len(frame), len(GRID)))
    cache = {}
    for i, d in enumerate(dates):
        if pd.isna(d):
            out[i] = np.asarray(final)
            continue
        # last cutoff strictly before this game's date
        idx = int(np.searchsorted(cut.values, np.datetime64(d), side="left")) - 1
        if idx < 0:
            out[i] = np.ones(len(GRID))
            continue
        if idx not in cache:
            tbl = per[idx]
            cache[idx] = np.ones(len(GRID)) if tbl is None else np.asarray(tbl)
        out[i] = cache[idx]
    return out


# ---------------------------------------------------------------- simulation
def _nb_draw(rng, mu: float, r: float, n: int) -> np.ndarray:
    mu = max(float(mu), 1e-6)
    r = max(float(r), 1e-6)
    lam = rng.gamma(shape=r, scale=mu / r, size=n)
    return rng.poisson(lam).astype(np.int32)


def _apply_ot(hs, aws, rng, rule, resample):
    dm, dp = float(rule["d_m1"]), float(rule["d_p1"])
    lo, hi = int(rule["w_lo"]), int(rule["w_hi"])
    hs, aws = hs.copy(), aws.copy()
    m = hs - aws
    u = rng.random(m.shape)
    drop = (m == 0) | ((m == 1) & (u < dp)) | ((m == -1) & (u < dm))
    it = 0
    while bool(drop.any()) and it < 40:
        idx = np.flatnonzero(drop)
        nh, na = resample(len(idx))
        keep = (np.abs(nh - na) >= lo) & (np.abs(nh - na) <= hi)
        hs[idx[keep]] = nh[keep]
        aws[idx[keep]] = na[keep]
        m = hs - aws
        u = rng.random(m.shape)
        drop = np.zeros(m.shape, bool)
        drop[idx[~keep]] = True
        it += 1
    return hs, aws


def price_frame_cdfs(frame: pd.DataFrame, bundle: dict, n_draws: int = N_MC,
                     seed: int = MC_SEED, meta_out=None):
    """Price a frame with the same PIT split the production path uses.

    Settled rows take the prepared block-refit mus (and RAISE when the walk
    never covered them); unsettled rows take the bundle. There is no
    full-frame shortcut on this entry point either.
    """
    if {"home_score", "away_score"}.issubset(frame.columns):
        hs = pd.to_numeric(frame["home_score"], errors="coerce").to_numpy(float)
        as_ = pd.to_numeric(frame["away_score"], errors="coerce").to_numpy(float)
        settled = np.isfinite(hs) & np.isfinite(as_)
    else:
        settled = np.zeros(len(frame), dtype=bool)
    mu_h = np.full(len(frame), np.nan)
    mu_a = np.full(len(frame), np.nan)
    pit = None
    if settled.any():
        ids = (frame["game_id"].astype(str).to_numpy() if "game_id" in frame
               else np.arange(len(frame)).astype(str))
        pit = lookup_pit_mus(ids[settled])
        if pit is None or pit[["mu_h", "mu_a", "mu_KR", "mu_KE", "r_R",
                               "r_E"]].isna().any().any():
            raise RuntimeError(
                "price_frame_cdfs: settled rows without PIT mus; call "
                "prepare_pit_mus() on the settled frame first (full-frame "
                "trees have already seen these finals)")
        mu_h[settled] = pit["mu_h"].to_numpy(float)
        mu_a[settled] = pit["mu_a"].to_numpy(float)
        pit = pit.reset_index(drop=True)
    if (~settled).any():
        bh, ba = predict_step1(frame.iloc[np.flatnonzero(~settled)], bundle)
        mu_h[~settled] = np.asarray(bh, dtype=float)
        mu_a[~settled] = np.asarray(ba, dtype=float)
    return price_cdfs(frame, mu_h, mu_a, bundle, n_draws=n_draws, seed=seed,
                      meta_out=meta_out, pit=pit)


def price_cdfs(frame: pd.DataFrame, mu_h, mu_a, bundle: dict,
               n_draws: int = N_MC, seed: int = MC_SEED, meta_out=None,
               pit=None):
    """Weighted empirical CDFs for margin and total over integer supports.

    ``pit`` is the block-refit pricing inputs for settled rows (game_id,
    mu_h, mu_a, mu_KR, mu_KE, r_R, r_E): when given, BOTH the residual
    means and the NB dispersions come from that refit -- the bundle's
    full-frame step-2/3 trees never touch a settled game. Slate pricing
    passes pit=None and uses the bundle throughout (all history is prior).
    """
    rule = _ot_rule()
    from gaussian_family import MC_SE_TARGET
    # Same precision guard as the champion's inverse-CDF sampler: never price
    # below its standard-error target, and report the count actually used.
    used = max(int(n_draws), int(np.ceil(0.25 / MC_SE_TARGET ** 2)))
    if pit is not None:
        _need = ("mu_h", "mu_a", "mu_KR", "mu_KE", "r_R", "r_E")
        if len(pit) != len(frame) or pit[list(_need)].isna().any().any():
            raise ValueError("PIT mus must cover every priced row exactly; "
                             "mixed settled/unsettled frames are not priced "
                             "through one call")
        mu_kr = pd.to_numeric(pit["mu_KR"], errors="coerce").to_numpy(float)
        mu_ke = pd.to_numeric(pit["mu_KE"], errors="coerce").to_numpy(float)
        # Dispersions ride the same block refit: the block that predicted
        # these games fitted its own NB dispersion. Never the bundle's
        # full-frame values, never a frozen constant from a leaky run.
        r_r = pd.to_numeric(pit["r_R"], errors="coerce").to_numpy(float)
        r_e = pd.to_numeric(pit["r_E"], errors="coerce").to_numpy(float)
        mus_source = "walkforward_pit_blocks"
    else:
        mu_kr, mu_ke, r_r, r_e = predict_residuals(frame, mu_h, mu_a, bundle)
        mus_source = "full_frame_bundle"
    n = len(frame)
    cM = np.zeros((n, len(GRID)))
    cT = np.zeros((n, TMAX - TMIN + 1))
    ids = (frame["game_id"].astype(str).to_numpy() if "game_id" in frame
           else np.arange(n).astype(str))
    # Cumulative walk-forward tables: a fold's games are priced with weights
    # fit strictly on prior folds; serving rows use the final (all-settled)
    # table. Bundles without tables fall back to the frozen artifact.
    W_tab = _row_weights(frame, bundle)
    for i in range(n):
        digest = hashlib.blake2b(f"nba|hier|{seed}|{ids[i]}".encode(),
                                 digest_size=16).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))

        # Dispersions are per-game under PIT pricing (each row carries the
        # block that predicted it), scalars under bundle pricing.
        _rr = float(np.ravel(r_r)[i]) if np.ndim(r_r) else float(r_r)
        _re = float(np.ravel(r_e)[i]) if np.ndim(r_e) else float(r_e)
        hh, aa = _draw_game(float(mu_h[i]), float(mu_a[i]), float(mu_kr[i]),
                            float(mu_ke[i]), _rr, _re, rng, rule, used)
        m = np.clip(hh - aa, -110, 110) + 110
        t = np.clip(hh + aa, TMIN, TMAX)
        wd = W_tab[i][m]
        hm = np.bincount(m, weights=wd, minlength=len(GRID)).astype(float)
        ht = np.bincount(t, weights=wd, minlength=TMAX - TMIN + 1).astype(float)
        if hm.sum() <= 0 or ht.sum() <= 0:
            raise ValueError(f"game {ids[i]}: degenerate weighted histogram")
        # Clip: cumulative sums can land a hair above 1.0 in float, and the
        # family CRPS check rejects any probability outside [0, 1].
        cM[i] = np.minimum(np.cumsum(hm) / hm.sum(), 1.0)
        cT[i] = np.minimum(np.cumsum(ht) / ht.sum(), 1.0)
    if meta_out is not None:
        meta_out.update({"method": "hier_nb_weighted_mc", "model_id": MODEL_ID,
                         "n_draws": int(used),
                         "requested_draws": int(n_draws), "seed": int(seed),
                         "mc_se_totals_max": float(0.5 / np.sqrt(used)),
                         "reason": ("precision_guard" if used != int(n_draws)
                                    else "requested"),
                         "calibration": "identity",
                         "k_trend": K_TR,
                         "k_spike": K_SP,
                         "weight_scope": ("cumulative_walkforward"
                                          if (bundle.get("weight_tables") or {}).get("final")
                                          is not None else "identity_fallback_unweighted"),
                         "weight_history": (bundle.get("weight_tables") or {}).get("history_source"),
                         "weight_policy": "per-date tables fit on strictly prior "
                                           "settled games; never serialized, never committed",
                         "mus_source": mus_source})
    return {"margin": (GRID.astype(float), cM),
            "total": (np.arange(TMIN, TMAX + 1).astype(float), cT)}


# --------------------------------------------- champion market pricing
CHAMPION_METHOD = "hier_nb_mc"
CHAMPION_OBJECTIVE = "negative_binomial_mc"


def _champion_grid_key(prefix, value):
    import distributions as dist_mod
    return dist_mod._grid_key(prefix, value)


def _champion_cdf_at(support, cdf, threshold):
    sup = np.asarray(support, float)
    idx = np.searchsorted(sup, np.asarray(threshold, float), side="right") - 1
    idx = np.clip(idx, -1, len(sup) - 1)
    out = np.zeros(len(cdf), float)
    ok = idx >= 0
    rows = np.flatnonzero(ok)
    out[rows] = np.asarray(cdf)[rows, idx[ok]]
    return out


def champion_price_columns(frame, mu_h, mu_a, cdfs):
    sup_m, cM = (np.asarray(cdfs["margin"][0], float),
                 np.asarray(cdfs["margin"][1], float))
    sup_t, cT = (np.asarray(cdfs["total"][0], float),
                 np.asarray(cdfs["total"][1], float))
    import config
    n = len(frame)
    mu_h = np.asarray(mu_h, float)
    mu_a = np.asarray(mu_a, float)
    updates = {"mu_margin": mu_h - mu_a, "mu_total": mu_h + mu_a}
    for kind, sup, cdf in (("margin", sup_m, cM), ("total", sup_t, cT)):
        mass = np.diff(np.column_stack([np.zeros(n), cdf]), axis=1)
        mean = mass @ sup
        updates[f"distribution_mean_{kind}"] = mean
        updates[f"distribution_sd_{kind}"] = np.sqrt(
            np.maximum(mass @ (sup ** 2) - mean ** 2, 0.))
        base_lines = (list(config.TOTAL_GRID) if kind == "total"
                      else list(config.SPREAD_GRID))
        extra = [float(x) + .5 for x in base_lines]
        if kind == "margin":
            extra += [float(x) for x in config.HALF_STOP_LINES]
        half = sup[:-1] + .5
        j = np.argmin(np.abs(1. - cdf[:, :-1] - .5), axis=1)
        fair = half[j]
        lines = sorted(set([*base_lines, *extra, *fair.tolist()]))
        prefix = "p_over" if kind == "total" else "p_home_cover"
        pp = "p_push_total" if kind == "total" else "p_push"
        under = "p_under" if kind == "total" else "p_away_cover"
        for line in lines:
            upper = _champion_cdf_at(sup, cdf, np.full(n, line))
            lower = _champion_cdf_at(sup, cdf, np.full(n, np.ceil(line) - 1.))
            updates[_champion_grid_key(prefix, line)] = 1. - upper
            updates[_champion_grid_key(pp, line)] = np.maximum(upper - lower, 0.)
            updates[_champion_grid_key(under, line)] = lower
        if kind == "margin" and hasattr(config, "MARGIN_PUSH_EXTENSION"):
            for k in config.MARGIN_PUSH_EXTENSION:
                upper = _champion_cdf_at(sup, cdf, np.full(n, float(k)))
                lower = _champion_cdf_at(sup, cdf, np.full(n, np.ceil(float(k)) - 1.))
                updates.setdefault(_champion_grid_key("p_push", float(k)),
                                   np.maximum(upper - lower, 0.))
        if kind == "margin":
            p_le0 = _champion_cdf_at(sup, cdf, np.zeros(n))
            p_lem1 = _champion_cdf_at(sup, cdf, np.full(n, -1.))
            updates["p_home_win_derived"] = 1. - p_le0
            updates["p_tie"] = np.maximum(p_le0 - p_lem1, 0.)
            updates["p_away_win_derived"] = p_lem1
            updates["derived_ml"] = updates["p_home_win_derived"]
        updates["fair_total" if kind == "total" else "fair_spread"] = fair
        p_fair = ("p_over_fair" if kind == "total" else "p_cover_fair")
        updates[p_fair] = 1. - cdf[np.arange(n), j]
    out = pd.DataFrame(updates, index=frame.index)
    if {"home_score", "away_score"}.issubset(frame.columns):
        margin = (pd.to_numeric(frame.home_score, errors="coerce")
                  - pd.to_numeric(frame.away_score, errors="coerce"))
        total = (pd.to_numeric(frame.home_score, errors="coerce")
                 + pd.to_numeric(frame.away_score, errors="coerce"))
        decided = margin.notna() & total.notna()
        for col, values in {"y_cover_fair": margin > out.fair_spread,
                            "y_push_spread_fair": margin == out.fair_spread,
                            "y_over_fair": total > out.fair_total,
                            "y_under_fair": total < out.fair_total,
                            "y_push_fair": total == out.fair_total}.items():
            out[col] = np.where(decided, values.astype(float), np.nan)
    out.attrs["distribution_method"] = CHAMPION_METHOD
    out.attrs["distribution_objective"] = CHAMPION_OBJECTIVE
    return out


def _champion_reporter(priced, cdfs_by_gid):
    import distributions as dist_mod
    from sklearn.metrics import roc_auc_score
    frame = priced.reset_index(drop=True)
    gids = frame.game_id.astype(str).to_numpy()
    n = len(frame)
    values = {"total": [], "margin": []}
    for kind in ("total", "margin"):
        if kind == "total":
            y = (pd.to_numeric(frame.home_score, errors="coerce")
                 + pd.to_numeric(frame.away_score, errors="coerce")).to_numpy(float)
        else:
            y = (pd.to_numeric(frame.home_score, errors="coerce")
                 - pd.to_numeric(frame.away_score, errors="coerce")).to_numpy(float)
        loss = np.full(n, np.nan)
        for i, gid in enumerate(gids):
            c = cdfs_by_gid.get(str(gid))
            if c is None or not np.isfinite(y[i]):
                continue
            sup, cdf = (np.asarray(c[kind][0], float), np.asarray(c[kind][1], float))
            loss[i] = float(dist_mod.lattice_crps(
                np.asarray([cdf]), sup, np.asarray([y[i]]))[0])
        values[kind] = loss[np.isfinite(loss)]
    result = {"method": CHAMPION_METHOD, "roi_verified": False}
    for kind, key, line_key, name in (("total", "p_over_fair", "fair_total", "totals"),
                                      ("margin", "p_cover_fair", "fair_spread", "run_line")):
        p = pd.to_numeric(frame[key], errors="coerce").to_numpy(float)
        if kind == "total":
            y = (pd.to_numeric(frame.home_score, errors="coerce")
                 + pd.to_numeric(frame.away_score, errors="coerce")).to_numpy(float)
        else:
            y = (pd.to_numeric(frame.home_score, errors="coerce")
                 - pd.to_numeric(frame.away_score, errors="coerce")).to_numpy(float)
        line = pd.to_numeric(frame[line_key], errors="coerce").to_numpy(float)
        ok = np.isfinite(p) & np.isfinite(y) & np.isfinite(line)
        p, y, line = p[ok], y[ok], line[ok]
        event = y > line
        clipped = np.clip(p, 1e-7, 1 - 1e-7)
        ll = float(-np.mean(event * np.log(clipped)
                            + (~event) * np.log1p(-clipped))) if len(p) else float("nan")
        bins = np.minimum((p * 10).astype(int), 9)
        reliability = []
        error = 0.
        for b in np.unique(bins):
            m = bins == b
            gap = float(p[m].mean() - event[m].mean())
            error += float(m.mean()) * abs(gap)
            reliability.append({"n": int(m.sum()), "predicted": float(p[m].mean()),
                                "actual": float(event[m].mean()), "gap": gap})
        vals = values[kind]
        result[name] = {"n": int(len(p)), "n_pushes": 0,
                        "crps": float(np.mean(vals)) if len(vals) else float("nan"),
                        "auc": (float(roc_auc_score(event, p))
                                if len(np.unique(event)) == 2 else None),
                        "logloss": ll, "logscore": ll,
                        "brier": float(np.mean((p - event) ** 2)) if len(p) else float("nan"),
                        "predicted_mean": float(p.mean()) if len(p) else float("nan"),
                        "actual_win_rate": float(event.mean()) if len(p) else float("nan"),
                        "ece_cover": error, "reliability": reliability,
                        "calibration_in_the_large": (float(p.mean() - event.mean())
                                                     if len(p) else float("nan")),
                        "threshold_policy": "closest_half_point_to_model_50_percent"}
    prob = pd.to_numeric(frame.p_home_win_derived, errors="coerce").to_numpy(float)
    tie = pd.to_numeric(frame.p_tie, errors="coerce").fillna(0.).to_numpy(float)
    mg = (pd.to_numeric(frame.home_score, errors="coerce")
          - pd.to_numeric(frame.away_score, errors="coerce")).to_numpy(float)
    elig = np.isfinite(prob) & np.isfinite(mg) & (mg != 0)
    cond = np.clip(prob[elig] / np.maximum(1. - tie[elig], 1e-12), 1e-7, 1 - 1e-7)
    result["derived_moneyline"] = {
        "n": int(elig.sum()),
        "logloss": float(-np.mean((mg[elig] > 0) * np.log(cond)
                                    + (mg[elig] <= 0) * np.log1p(-cond))) if elig.any() else float("nan"),
        "brier": float(np.mean((cond - (mg[elig] > 0)) ** 2)) if elig.any() else float("nan"),
    }
    return result


def _merge_champion_prices(base_oof, prices):
    """Merge the champion's priced columns over a base OOF frame.

    The champion owns EVERY price column it produces. Dropping the base's
    copies first is the difference between "price the artifact with the
    champion" and "keep whatever marketize priced": the base arrives with
    p_home_win_derived, p_tie, p_away_win_derived, derived_ml and the
    distribution moments already stamped by the split-skew walk, so a
    ``c not in base.columns`` filter would keep all of them and the champion
    would write only game_id -- leaving derived_ml (and its moneyline) on the
    old engine under a champion label.
    """
    base = base_oof.copy()
    base["game_id"] = base.game_id.astype(str)
    owned = [c for c in prices.columns if c in base.columns and c != "game_id"]
    if owned:
        base = base.drop(columns=owned)
    keep = [c for c in prices.columns if c not in base.columns or c == "game_id"]
    return base.merge(prices[keep], on="game_id", how="left",
                      suffixes=("", "_champion"))


def champion_oof_markets(game_df, fold_list, base_oof, n_draws=None,
                         seed=None, progress_every=0):
    import distributions as dist_mod
    import folds as folds_mod
    n_draws = int(n_draws or dist_mod.MC_DRAWS)
    seed = int(seed or dist_mod.MC_SEED)
    df = folds_mod.canonical_sort(game_df)
    settled_mask = (pd.to_numeric(df.home_score, errors="coerce").notna()
                    & pd.to_numeric(df.away_score, errors="coerce").notna()).to_numpy()
    tables = build_weight_tables(df[settled_mask].reset_index(drop=True), {})
    price_parts = []
    cdf_store = {}
    origins = []
    for fold in fold_list:
        train = df.loc[fold.train_idx]
        val = df.loc[fold.val_idx]
        val_settled = val[pd.to_numeric(val.home_score, errors="coerce").notna()
                          & pd.to_numeric(val.away_score, errors="coerce").notna()]
        if not len(val_settled):
            continue
        core = _fit_core(train)
        mu_h, mu_a = predict_step1(val_settled, core)
        if not (np.isfinite(mu_h).all() and np.isfinite(mu_a).all()):
            raise ValueError(f"champion OOF: non-finite mus in fold {fold.fold_id}")
        bundle_o = {"models": core["models"], "weight_tables": tables}
        cdfs = price_cdfs(val_settled, mu_h, mu_a, bundle_o,
                           n_draws=n_draws, seed=seed)
        cols = champion_price_columns(val_settled, mu_h, mu_a, cdfs)
        cols["game_id"] = val_settled.game_id.astype(str).to_numpy()
        cols["mu_h_champion"] = np.asarray(mu_h, float)
        cols["mu_a_champion"] = np.asarray(mu_a, float)
        price_parts.append(cols)
        sup_m, cM = np.asarray(cdfs["margin"][0]), np.asarray(cdfs["margin"][1])
        sup_t, cT = np.asarray(cdfs["total"][0]), np.asarray(cdfs["total"][1])
        for k, gid in enumerate(val_settled.game_id.astype(str).to_numpy()):
            cdf_store[str(gid)] = {"margin": (sup_m, cM[k]), "total": (sup_t, cT[k])}
        origins.append({"fold_id": fold.fold_id, "n": int(len(val_settled)),
                        "fit_n": int(core.get("fit_n", 0))})
        if progress_every:
            logger.info("champion OOF: priced fold %s (%d games)",
                        fold.fold_id, len(val_settled))
    prices = pd.concat(price_parts, ignore_index=True) if price_parts else pd.DataFrame()
    out = _merge_champion_prices(base_oof, prices)
    for c in ("mu_h", "mu_a"):
        cc = f"{c}_champion"
        if cc in out.columns:
            champ = pd.to_numeric(out[cc], errors="coerce")
            base_mu = (pd.to_numeric(out[c], errors="coerce")
                       if c in out.columns else pd.Series(np.nan, index=out.index))
            out[c] = champ.fillna(base_mu)
            out["pred_home" if c == "mu_h" else "pred_away"] = out[c]
            out = out.drop(columns=[cc])
    # ``_marketize`` stamps derived_ml from p_home_win_derived BEFORE the
    # deferred pricing runs, so the base frame carries all-NaN here and the
    # merge keeps it (a base column is never overwritten). The champion's
    # derived moneyline is the margin CDF at -1/0, so re-stamp it: the
    # monitor's Derived ML card and per-line metric both read this column.
    out["derived_ml"] = out["p_home_win_derived"]
    # The champion owns the served moneyline too. The binary ensemble stays
    # in p_ensemble/p_ensemble_calibrated as the graded benchmark, but the
    # headline p_home_win/p_away_win ride the champion's margin CDF -- ties
    # excluded by construction, priced explicitly in p_away_win_derived.
    out["p_home_win"] = pd.to_numeric(out["p_home_win_derived"],
                                      errors="coerce")
    out["p_away_win"] = pd.to_numeric(out["p_away_win_derived"],
                                      errors="coerce")
    priced_ids = set(prices.game_id) if len(prices) else set()
    missing = out.game_id[~out.game_id.isin(priced_ids)]
    if len(missing):
        # Champion-only (2026-11-12 guardrail): a row with no champion price
        # would ship the ensemble's mu under a champion label, so it is an
        # error rather than a warning.
        raise ValueError(
            "champion OOF: %d of %d base rows have no champion price "
            "(first: %s); refusing to ship a mixed-engine artifact"
            % (len(missing), len(out), ", ".join(missing.astype(str)[:5])))
    metrics = _champion_reporter(out, cdf_store)
    bundle = {"method": CHAMPION_METHOD, "model_id": MODEL_ID,
              "monte_carlo": {"requested_draws": int(n_draws), "seed": int(seed),
                                "model_id": MODEL_ID, "method": "hier_nb_weighted_mc"},
              "fit_n": int(settled_mask.sum()),
              "fit_end": str(pd.to_datetime(df.gameday).max().date()),
              "origins": origins, "oof_metrics": metrics,
              "weight_tables_summary": {
                  "n_table_dates": (tables or {}).get("n_table_dates", 0),
                  "history_source": (tables or {}).get("history_source"),
                  "k_trend": K_TR, "k_spike": K_SP},
              "lattice_summary": {}, "maps": {}}
    out.attrs["distribution_metrics"] = metrics
    out.attrs["challenger_bundle"] = bundle
    out.attrs["distribution_method"] = CHAMPION_METHOD
    out.attrs["distribution_objective"] = CHAMPION_OBJECTIVE
    out.attrs["mc_meta"] = bundle["monte_carlo"]
    return out, bundle


def champion_slate_markets(slate_base, serving_bundle, n_draws=None, seed=None):
    import distributions as dist_mod
    n_draws = int(n_draws or dist_mod.MC_DRAWS)
    seed = int(seed or dist_mod.MC_SEED)
    frame = slate_base.copy().reset_index(drop=True)
    mu_h, mu_a = predict_step1(frame, serving_bundle)
    if not (np.isfinite(mu_h).all() and np.isfinite(mu_a).all()):
        raise ValueError("champion slate: non-finite mus")
    bundle = ensure_tables(frame, dict(serving_bundle))
    cdfs = price_cdfs(frame, mu_h, mu_a, bundle, n_draws=n_draws, seed=seed)
    cols = champion_price_columns(frame, mu_h, mu_a, cdfs)
    cols["mu_h"] = np.asarray(mu_h, float)
    cols["mu_a"] = np.asarray(mu_a, float)
    cols["pred_home"] = np.asarray(mu_h, float)
    cols["pred_away"] = np.asarray(mu_a, float)
    add = cols.drop(columns=["game_id"], errors="ignore")
    out = frame.copy().drop(columns=[c for c in add.columns
                                     if c in frame.columns])
    out = pd.concat([out.reset_index(drop=True),
                     add.reset_index(drop=True)], axis=1)
    # Served moneyline rides the champion (see champion_oof_markets): the
    # ensemble columns the base frame carries stay as the benchmark.
    out["p_home_win"] = pd.to_numeric(out["p_home_win_derived"],
                                      errors="coerce")
    out["p_away_win"] = pd.to_numeric(out["p_away_win_derived"],
                                      errors="coerce")
    out.attrs["distribution_method"] = CHAMPION_METHOD
    out.attrs["distribution_objective"] = CHAMPION_OBJECTIVE
    return out


# ------------------------------------------------------------------ bundles
def save_bundle(bundle: dict, path: str | Path | None = None) -> Path:
    """Persist models only. Weight tables can never be written to disk.

    The saved bundle is refit inputs (step-1/2/3 models). The cumulative
    weight tables are runtime-only: ensure_tables() rebuilds them in memory
    at serving time, so no serialized table can ever be committed.
    """
    import joblib
    import config
    slim = {k: v for k, v in bundle.items() if k != "weight_tables"}
    slim["weight_tables"] = None
    p = Path(path) if path else Path(config.MODELS_DIR) / BUNDLE_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(slim, p)
    return p


def ensure_tables(frame: pd.DataFrame, bundle: dict,
                  progress_every: int = 0) -> dict:
    """Attach cumulative weight tables to a bundle in memory (never saved).

    Serving path: load the model-only bundle, rebuild this run's per-date
    tables from OOS history here, and price. Point-in-time pricing without
    any persisted table.
    """
    if (bundle.get("weight_tables") or {}).get("final") is not None:
        return bundle
    key = int(bundle.get("fit_n") or 0)
    hit = _TABLES_CACHE.get(key)
    if hit is not None:
        # The pipeline run already replayed this run's history at fit time;
        # serving must reuse those tables instead of replaying again.
        bundle["weight_tables"] = hit
        return bundle
    if len(frame) < MIN_HISTORY_ROWS:
        # Per-slate pricing calls carry a handful of rows: there is no history
        # to fit against here, so answer with the identity tables immediately
        # instead of paying (and logging) a doomed history attempt. The run's
        # fit step already registered the real tables under this bundle.
        # Return a COPY: load_bundle() hands out a cached dict, and writing
        # identity tables into it would make the unweighted state sticky for
        # every later call in the process, including after the run's fit step
        # registered real tables.
        return {**bundle, "weight_tables": {
            "cutoffs": [], "tables": [], "final": None,
            "k_trend": K_TR, "k_spike": K_SP, "n_dates": 0,
            "n_table_dates": 0,
            "history_source": "identity_insufficient_history",
            "serialized": False}}
    partial = {"step1": bundle["step1"],
               "step1_columns": bundle["step1_columns"],
               "models": bundle["models"]}
    bundle["weight_tables"] = build_weight_tables(
        frame, partial, progress_every=progress_every)
    if bundle["weight_tables"].get("final") is not None:
        _TABLES_CACHE[key] = bundle["weight_tables"]
    return bundle


_TABLES_CACHE: dict = {}
_BUNDLE_CACHE: dict = {}


def load_bundle(path: str | Path | None = None,
                refresh: bool = False) -> dict | None:
    """Load the model-only bundle, cached per process by file mtime.

    The cache matters because ensure_tables() attaches the rebuilt tables to
    the returned dict in memory: repeat loads in one process reuse them
    instead of refitting per pricing call.
    """
    import joblib
    import config
    p = Path(path) if path else Path(config.MODELS_DIR) / BUNDLE_NAME
    if not p.exists():
        return None
    key = str(p.resolve())
    stamp = p.stat().st_mtime_ns
    hit = None if refresh else _BUNDLE_CACHE.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    bundle = joblib.load(p)
    _BUNDLE_CACHE[key] = (stamp, bundle)
    return bundle
