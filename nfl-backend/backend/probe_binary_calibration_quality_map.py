"""Probe: binary top-band over-confidence — quality-stratum LOCAL
recalibration (READ-ONLY; no production/model change).

Background (records e3aeece / fc4f4bd / c1a7c12): the global Platt twin
(a=1.276, b=0.122) over-stretches the raw ~0.66-0.72 region into the
70-80% band (pred 0.7485 vs actual 0.6754, n=191, ECE 0.0731) while the
raw axis is honest on those games (0.6815 vs 0.6754). Global map swaps
were refused twice (DO_NOT_REFIT / KEEP_PLATT) because pooled logloss
regressed. This probe tests a LOCAL correction: a monotone map applied
ONLY in the stratum (raw > h, h pinned from data), with the global
Platt map untouched outside.

Pre-registered mechanism:
  stratum   = raw-axis confidence > h; h scanned 0.66..0.74 step 0.01
              (prior ~0.68). Quality-extreme overlap (audit b8: top
              quartile |elo_diff|) reported as the stratum
              characterization.
  families  = L  local logistic anchored at (h, h): logit(s) =
                  logit(h) + a*(logit(p) - logit(h)), slope-only fit
                  on in-stratum strictly-prior rows (<=3 params,
                  low fold variance; monotone by construction, a>=0).
              P  local pchip: monotone cubic Hermite through
                  isotonic-smoothed knots over the in-stratum
                  strictly-prior rows, hard-clipped to [0,1].
              I  identity in-stratum (the degenerate monotone map —
                  reference row; the e3aeece "serve-time convention").
              isotonic excluded per the fc4f4bd logloss finding.
  outside   = the global Platt map (nested per-fold fit for the
              pre-holdout scan; the deployed a/b fit for the sealed
              protocol) — untouched by the local correction.
  serve     = raw <= h -> Platt(raw); raw > h -> local_map(raw).

Outputs:
  universe            shared 1,376 decided OOF + production pool
                      (1,107 pooled pre-holdout incl. playoff weeks;
                      285 sealed 2025)
  r0_gate             deployed global Platt reproduces the published
                      map (a/b to 3e-5) and sealed ll 0.6249 /
                      ECE 0.0745 bit-consistently
  stratum_char        in-stratum overlap with the audit quality-
                      extreme stratum (top-quartile |elo_diff|), mean
                      |binary - derived| in/out of stratum
  nested_scan         h x family surface on pooled pre-holdout (88
                      fold-weeks, strictly-earlier fits): pooled
                      cal-logloss, overall ECE, 70-80 band ECE on the
                      NEW served axis
  selection           h* / family* rule: among cells with nested
                      70-80 band ECE <= 0.03 pick min pooled ll; else
                      report surface and pick min band ECE
  fold_stability      per-fold local-map output spread at fixed raw
                      inputs 0.70/0.75/0.85 (forbidden >1.0 mode)
  deployed            chosen map fit on ALL pooled rows -> pooled
                      in-sample + sealed + S1/S2 leg table
  gate_legs           sealed S2 70-80 band ECE <= 0.03; pooled
                      cal-logloss within +/-0.001 vs served Platt;
                      sealed adjacent bands no regression > +0.01;
                      AUC-flat <= 0.001; no-bleed [0,1] + monotone;
                      worth-having; determinism (byte-identical
                      double run)
  verdict             GATE_PASS (-> serve-time local map adoption
                      commit) / GATE_FAIL (record + follow-ons)
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DD = Path(__file__).resolve().parent.parent / "data_delivery"
DATE = "20260904"

BANDS = ((0.60, 0.70), (0.70, 0.80), (0.80, 1.01))
HS = [round(0.66 + 0.01 * i, 2) for i in range(9)]  # 0.66..0.74
PIN_A, PIN_B = 1.276336, 0.121988
PIN_SEALED_LL, PIN_SEALED_ECE = 0.6249, 0.0745
IN_STRAT_MIN_ROWS = 10  # local-map fallback -> identity below this


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _band_ece_overall(y: np.ndarray, p: np.ndarray) -> float:
    """Weighted ECE over equal-width deciles (nfl_moneyline.ece)."""
    edges = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    n = len(p)
    total = 0.0
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        total += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(total)


def band_ece(y: np.ndarray, p: np.ndarray, lo: float, hi: float) -> float:
    m = (p >= lo) & (p < hi)
    if m.sum() == 0:
        return float("nan")
    return float(abs(p[m].mean() - y[m].mean()))


def _week_start(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return d - pd.to_timedelta(d.dt.weekday, unit="D")


def _fit_platt(p: np.ndarray, y: np.ndarray):
    """Global logistic on logit(raw) — identical to nfl_moneyline.platt_fit."""
    y = np.asarray(y, dtype=int)
    if len(y) < 10 or len(np.unique(y)) < 2:
        return None
    from sklearn.linear_model import LogisticRegression
    x = _logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=1e6)
    lr.fit(x, y)
    return lr


def _apply_platt(mapper, p: np.ndarray) -> np.ndarray:
    if mapper is None:
        return np.asarray(p, dtype=float).copy()
    return mapper.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def _fit_local_logistic(p: np.ndarray, y: np.ndarray, h: float,
                        anchor: float):
    """Slope-only logistic anchored at (h, anchor):
    logit(s) = logit(anchor) + a*(logit(p) - logit(h)). 1 free param, a>=0.
    anchor = Platt(h) keeps the served axis CONTINUOUS at h."""
    y = np.asarray(y, dtype=int)
    if len(y) < IN_STRAT_MIN_ROWS or len(np.unique(y)) < 2:
        return None
    from sklearn.linear_model import LogisticRegression
    x = (_logit(p) - _logit(np.array([h]))[0]).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, fit_intercept=False)
    lr.fit(x, y)
    return max(0.0, float(lr.coef_[0][0]))


def _apply_local_logistic(a: float | None, p: np.ndarray, h: float,
                          anchor: float) -> np.ndarray:
    if a is None:
        return np.full(len(p), float(anchor))
    return _logistic(_logit(np.full(len(p), anchor))
                     + a * (_logit(p) - _logit(np.full(len(p), h))))


def _fit_local_pchip(p: np.ndarray, y: np.ndarray, h: float,
                     anchor: float):
    """Monotone pchip with the seam point (h, anchor) as the FIRST knot
    (continuity with the global Platt map), then isotonic-smoothed knots;
    outputs hard-clipped to [0,1] (the fc4f4bd >1.0 mode is forbidden)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return None
    from scipy.interpolate import PchipInterpolator
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)
    xs = np.quantile(p, np.linspace(0.05, 0.95, 5))
    xs = np.unique(np.clip(xs, p.min(), p.max()))
    if len(xs) < 2:
        return None
    ys = np.clip(iso.predict(xs), 0.0, 1.0)
    xs = np.concatenate([[h], xs])
    ys = np.concatenate([[float(anchor)], ys])
    # keep the seam as the leftmost knot; guard equal-x duplicates
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    dupe = np.concatenate([[False], np.diff(xs) == 0])
    xs, ys = xs[~dupe], ys[~dupe]
    if len(xs) < 2:
        return None
    return PchipInterpolator(xs, ys)


def _apply_local_pchip(mapper, p: np.ndarray) -> np.ndarray:
    if mapper is None:
        return np.asarray(p, dtype=float).copy()
    return np.clip(mapper(np.asarray(p, dtype=float)), 0.0, 1.0)


def _fit_family(name: str, p: np.ndarray, y: np.ndarray, h: float,
                anchor: float):
    if name == "L_local_logistic":
        return _fit_local_logistic(p, y, h, anchor)
    if name == "P_local_pchip":
        return _fit_local_pchip(p, y, h, anchor)
    return None  # I_identity


def _apply_family(name: str, mapper, p: np.ndarray, h: float,
                  anchor: float) -> np.ndarray:
    if name == "L_local_logistic":
        return _apply_local_logistic(mapper, p, h, anchor)
    if name == "P_local_pchip":
        return _apply_local_pchip(mapper, p)
    return np.asarray(p, dtype=float).copy()


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def main() -> int:
    mk = pd.read_csv(DD / f"nfl_run_engine_markets_{DATE}.csv")
    oof = mk[mk["kind"] == "oof"][["game_id", "p_home_win_derived"]].copy()
    hist = pd.read_csv(DD / f"nfl_predictions_history_{DATE}.csv")

    df = oof.merge(
        hist[["game_id", "season", "week", "game_date", "home_score",
              "away_score", "home_win_prob_model",
              "home_win_prob_model_calibrated"]],
        on="game_id", how="left")
    df["raw"] = df["home_win_prob_model"]
    df["platt"] = df["home_win_prob_model_calibrated"]
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(float)
    df["week_start"] = _week_start(df["game_date"])

    hdf = hist.copy()
    hdf["raw"] = hdf["home_win_prob_model"]
    hdf["platt"] = hdf["home_win_prob_model_calibrated"]
    hdf["home_win"] = (hdf["home_score"] > hdf["away_score"]).astype(float)
    hdf["week_start"] = _week_start(hdf["game_date"])

    out: dict = {}
    out["universe"] = {
        "oof_rows": int(len(df)),
        "joined_with_binary": int(df["raw"].notna().sum()),
        "by_season": {int(k): int(v) for k, v in
                      df.groupby("season").size().items()},
    }

    # ---- R0 bit-consistency: deployed global Platt reproduction ----------
    pool = hdf[hdf["season"] <= 2024].sort_values("week_start").reset_index(drop=True)
    seal = hdf[hdf["season"] == 2025].sort_values("game_date").reset_index(drop=True)
    y_pool = pool["home_win"].to_numpy(float)
    raw_pool = pool["raw"].to_numpy(float)
    y_seal = seal["home_win"].to_numpy(float)
    raw_seal = seal["raw"].to_numpy(float)
    ma = _fit_platt(raw_pool, y_pool)
    a_fit, b_fit = float(ma.coef_[0][0]), float(ma.intercept_[0])
    cal_seal = _apply_platt(ma, raw_seal)
    sealed_ll = _ll(y_seal, cal_seal)
    sealed_ece = _band_ece_overall(y_seal, cal_seal)
    out["r0_gate"] = {
        "a_fit": round(a_fit, 6), "b_fit": round(b_fit, 6),
        "pin_a_b": {"a": PIN_A, "b": PIN_B},
        "a_err": abs(a_fit - PIN_A), "b_err": abs(b_fit - PIN_B),
        "sealed_ll": round(sealed_ll, 4), "pin_sealed_ll": PIN_SEALED_LL,
        "sealed_ece": round(sealed_ece, 4), "pin_sealed_ece": PIN_SEALED_ECE,
        "pass": (abs(a_fit - PIN_A) < 3e-5 and abs(b_fit - PIN_B) < 3e-5
                 and abs(sealed_ll - PIN_SEALED_LL) < 0.0005
                 and abs(sealed_ece - PIN_SEALED_ECE) < 0.0005),
    }

    # ---- stratum characterization (audit quality-extreme overlap) ---------
    # Primary stratum = raw > h (CSV-only, always computed). The audit
    # quality-extreme overlap (top-quartile |elo_diff|, audit b8) needs the
    # canonical decided feature frame — best-effort: degrades to a note when
    # the read-only cache is unreachable from the current drive/cwd.
    strat_char: dict = {}
    for h in (0.68, 0.70, 0.72):
        m = (df["raw"] > h).to_numpy()
        strat_char[f"raw_gt_{h:.2f}"] = {
            "n": int(m.sum()),
            "mean_abs_binary_derived_gap": round(float(
                (df["p_home_win_derived"] - df["platt"]).abs().to_numpy()[m].mean()), 4)
            if m.sum() else None,
        }
    try:
        from run_nfl_margin_ablation import load_features
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            feats = load_features(None)[["game_id", "elo_diff"]]
        fc = df.merge(feats, on="game_id", how="left")
        elo_abs = fc["elo_diff"].abs()
        top_q = fc["elo_diff"].abs() >= elo_abs.quantile(0.75)
        for h in (0.68, 0.70, 0.72):
            m = (fc["raw"] > h).to_numpy()
            strat_char[f"raw_gt_{h:.2f}"]["share_quality_extreme"] = round(
                float(top_q.to_numpy()[m].mean()), 3) if m.sum() else None
        strat_char["quality_extreme_n"] = int(top_q.sum())
        strat_char["feature_frame"] = ("canonical decided feature frame "
                                        "(load_features; read-only cache)")
    except Exception as exc:  # pragma: no cover - cache-dependent block
        strat_char["feature_frame"] = f"unavailable: {exc!r}"
        strat_char["quality_extreme_n"] = None
        for k in ("raw_gt_0.68", "raw_gt_0.70", "raw_gt_0.72"):
            strat_char[k]["share_quality_extreme"] = None
    strat_char["note"] = ("quality-extreme = top-quartile |elo_diff| on "
                          "the shared 1,376 universe (audit b8: the binary "
                          "over-recovers exactly where quality level is "
                          "extreme); overlap = share of in-stratum games in "
                          "that set (null when the feature cache is "
                          "unreachable)")
    out["stratum_char"] = strat_char

    # ---- nested scan: h x family on pooled pre-holdout (88 fold-weeks) ---
    weeks = pd.unique(pool["week_start"])
    week_idx = {w: i for i, w in enumerate(weeks)}
    n_folds = int(len(weeks))
    # precompute the nested Platt serving + per-fold mappers once (h-independent)
    platt_nested = np.zeros(len(pool))
    fold_mappers: list = []
    for w in weeks:
        prior = pool["week_start"] < w
        f_mask = (pool["week_start"] == w).to_numpy()
        mapper = _fit_platt(raw_pool[prior.to_numpy()], y_pool[prior.to_numpy()]) \
            if prior.sum() >= 10 else None
        fold_mappers.append(mapper)
        platt_nested[f_mask] = _apply_platt(mapper, raw_pool[f_mask])

    def _anchor_at(w: object, h: float) -> float:
        """Per-fold Platt value at the seam (continuity anchor)."""
        mapper = fold_mappers[week_idx[w]]
        if mapper is None:
            return float(h)
        return float(_apply_platt(mapper, np.array([h]))[0])

    families = ("L_local_logistic", "P_local_pchip", "I_identity")
    surface: dict[str, dict] = {}
    local_params: dict[tuple[float, str], np.ndarray] = {}
    for h in HS:
        for fam in families:
            served = platt_nested.copy()
            params = np.full(len(pool), np.nan)
            for w in weeks:
                prior = (pool["week_start"] < w).to_numpy()
                f_mask = (pool["week_start"] == w).to_numpy()
                in_strat = prior & (raw_pool > h)
                anchor = _anchor_at(w, h)
                if in_strat.sum() >= IN_STRAT_MIN_ROWS:
                    mapper = _fit_family(fam, raw_pool[in_strat],
                                         y_pool[in_strat], h, anchor)
                else:
                    mapper = None
                # capture per-fold slope for fold-stability (logistic only)
                if fam == "L_local_logistic":
                    params[f_mask] = mapper if mapper is not None else 1.0
                served[f_mask & (raw_pool > h)] = _apply_family(
                    fam, mapper, raw_pool[f_mask & (raw_pool > h)],
                    h, anchor)
            local_params[(h, fam)] = params
            bands = {f"{lo:.0%}-{hi:.0%}": (
                round(band_ece(y_pool, served, lo, hi), 4) if
                ((served >= lo) & (served < hi)).sum() else None)
                for lo, hi in BANDS}
            surface[f"h{h:.2f}_{fam}"] = {
                "h": h, "family": fam,
                "nested_pooled_logloss": round(_ll(y_pool, served), 4),
                "nested_pooled_ece": round(_band_ece_overall(y_pool, served), 4),
                "bands": bands,
            }
    out["nested_scan"] = {
        "pooled_pre_holdout_rows": int(len(pool)),
        "fold_weeks": n_folds,
        "hs": HS,
        "families": list(families),
        "protocol": ("nested per-fold strictly-earlier fits (the production "
                     "nested-Platt geometry); the global Platt part is "
                     "h-independent and computed once per fold; local maps "
                     "fit on in-stratum (raw>h) strictly-prior rows only; "
                     "identity fallback below IN_STRAT_MIN_ROWS"),
        "surface": surface,
    }

    # ---- selection: min nested pooled ll among target-band <= 0.03 --------
    # Eligible families = the pre-registered continuous monotone maps
    # (L/P); I_identity is a REFERENCE row — it cannot be the served axis
    # because the identity-in-stratum seam is discontinuous vs Platt
    # (breaks the global-monotonicity / AUC-flat contract by construction).
    sel_rows = []
    for key, row in surface.items():
        if row["family"] == "I_identity":
            continue
        b = row["bands"]["70%-80%"]
        if b is not None and b <= 0.03:
            sel_rows.append(row)
    if sel_rows:
        sel_rows.sort(key=lambda r: (r["nested_pooled_logloss"], r["bands"]["70%-80%"]))
        best = sel_rows[0]
        sel_rule = ("among eligible (L/P) cells with nested 70-80 band "
                    "ECE <= 0.03, pick min pooled logloss (tie-break: "
                    "band ECE); I_identity excluded as the reference row "
                    "(discontinuous seam)")
    else:
        cands = [r for r in surface.values() if r["family"] != "I_identity"]
        sel_rows = sorted(cands, key=lambda r: r["bands"]["70%-80%"])
        best = sel_rows[0]
        sel_rule = ("NO eligible cell reaches nested 70-80 band ECE <= 0.03 "
                    "— surface reported; selection = min band ECE "
                    "(informational)")
    h_star, fam_star = best["h"], best["family"]
    out["selection"] = {"rule": sel_rule, "h_star": h_star,
                        "family_star": fam_star, "best_row": best}

    # ---- fold stability at fixed raw inputs (forbidden >1.0 mode) ---------
    fx = np.array([0.70, 0.75, 0.85])
    fs: dict = {}
    for fam in ("L_local_logistic", "P_local_pchip"):
        rows = {}
        for x in fx:
            outs = []
            for w in weeks:
                prior = (pool["week_start"] < w).to_numpy()
                in_strat = prior & (raw_pool > h_star)
                anchor = _anchor_at(w, h_star)
                if in_strat.sum() >= IN_STRAT_MIN_ROWS:
                    mapper = _fit_family(fam, raw_pool[in_strat],
                                         y_pool[in_strat], h_star, anchor)
                else:
                    mapper = None
                outs.append(float(_apply_family(
                    fam, mapper, np.array([x]), h_star, anchor)[0]))
            outs = np.array(outs)
            rows[f"raw_{x:.2f}"] = {
                "mean": round(float(outs.mean()), 4),
                "min": round(float(outs.min()), 4),
                "max": round(float(outs.max()), 4),
                "std": round(float(outs.std(ddof=1)), 4) if len(outs) > 1 else 0.0,
            }
        rows["global_max_output"] = round(float(max(
            r["max"] for r in rows.values())), 4)
        rows["forbidden_gt_1_0"] = rows["global_max_output"] > 1.0 + 1e-12
        fs[fam] = rows
    out["fold_stability"] = {
        "h": h_star, "inputs": [float(x) for x in fx],
        "families": fs,
        "note": ("per-fold map output spread at a CONSTANT raw input across "
                 "the 88 fold-weeks; >1.0 outputs = the fc4f4bd pchip "
                 "extrapolation failure mode, forbidden here"),
    }

    # ---- deployed protocol: chosen map on ALL pooled rows -> sealed -------
    ma_deploy = _fit_platt(raw_pool, y_pool)
    anchor_d = float(_apply_platt(ma_deploy, np.array([h_star]))[0])
    if fam_star in ("L_local_logistic", "P_local_pchip"):
        in_strat_all = raw_pool > h_star
        mapper_d = _fit_family(fam_star, raw_pool[in_strat_all],
                               y_pool[in_strat_all], h_star, anchor_d)
    else:
        mapper_d = None

    def _served(raw: np.ndarray, deploy: bool = True) -> np.ndarray:
        base = _apply_platt(ma_deploy, raw) if deploy else platt_nested
        m = raw > h_star
        if m.sum():
            base = base.copy()
            base[m] = _apply_family(fam_star, mapper_d, raw[m], h_star,
                                    anchor_d)
        return base

    served_pool = _served(raw_pool)
    served_seal = _served(raw_seal)
    served_platt_seal = _apply_platt(ma_deploy, raw_seal)

    def _rows(y: np.ndarray, p: np.ndarray) -> dict:
        return {
            "n": int(len(y)),
            "logloss": round(_ll(y, p), 4),
            "ece": round(_band_ece_overall(y, p), 4),
            "auc": round(_auc(y, p), 4),
            "bands": {f"{lo:.0%}-{hi:.0%}": (
                round(band_ece(y, p, lo, hi), 4) if
                ((p >= lo) & (p < hi)).sum() else None)
                for lo, hi in BANDS},
        }

    half = len(seal) // 2
    s1, s2 = seal.iloc[:half], seal.iloc[half:]
    deploy = {
        "h_star": h_star, "family_star": fam_star,
        "global_platt_a_b": {"a": round(a_fit, 6), "b": round(b_fit, 6)},
        "pooled_in_sample": {
            "platt": _rows(y_pool, _apply_platt(ma_deploy, raw_pool)),
            "served": _rows(y_pool, served_pool)},
        "sealed": {
            "platt": _rows(y_seal, served_platt_seal),
            "served": _rows(y_seal, served_seal)},
        "sealed_s1": {
            "platt": _rows(s1["home_win"].to_numpy(float),
                           _apply_platt(ma_deploy, s1["raw"].to_numpy(float))),
            "served": _rows(s1["home_win"].to_numpy(float),
                            _served(s1["raw"].to_numpy(float)))},
        "sealed_s2": {
            "platt": _rows(s2["home_win"].to_numpy(float),
                           _apply_platt(ma_deploy, s2["raw"].to_numpy(float))),
            "served": _rows(s2["home_win"].to_numpy(float),
                            _served(s2["raw"].to_numpy(float)))},
    }
    out["deployed"] = deploy

    # ---- gate legs ---------------------------------------------------------
    g = {}
    s2_y = s2["home_win"].to_numpy(float)
    s2_served = _served(s2["raw"].to_numpy(float))
    s2_platt = _apply_platt(ma_deploy, s2["raw"].to_numpy(float))
    g["leg1_sealed_s2_70_80_band_ece_le_0_03"] = {
        "served": round(band_ece(s2_y, s2_served, 0.70, 0.80), 4),
        "platt": round(band_ece(s2_y, s2_platt, 0.70, 0.80), 4),
        "bar": "<= 0.03", "pass": band_ece(s2_y, s2_served, 0.70, 0.80) <= 0.03}

    # nested pooled logloss: the served family vs the nested Platt reference
    # (platt_nested = per-fold strictly-earlier Platt serving, computed once).
    platt_nested_ll = _ll(y_pool, platt_nested)
    served_nested_ll = surface[f"h{h_star:.2f}_{fam_star}"]["nested_pooled_logloss"]
    g["leg2_pooled_cal_logloss_within_0_001"] = {
        "nested_platt_ref": round(platt_nested_ll, 4),
        "nested_served": served_nested_ll,
        "delta": round(served_nested_ll - platt_nested_ll, 4),
        "deployed_in_sample_platt": deploy["pooled_in_sample"]["platt"]["logloss"],
        "deployed_in_sample_served": deploy["pooled_in_sample"]["served"]["logloss"],
        "deployed_in_sample_delta": round(
            deploy["pooled_in_sample"]["served"]["logloss"]
            - deploy["pooled_in_sample"]["platt"]["logloss"], 4),
        "bar": "within +/- 0.001 (nested primary)",
        "pass": abs(served_nested_ll - platt_nested_ll) <= 0.001}

    adj = {}
    for lo, hi, lab in ((0.60, 0.70, "60_70"), (0.80, 1.01, "80_plus")):
        d = round(band_ece(y_seal, served_seal, lo, hi)
                  - band_ece(y_seal, served_platt_seal, lo, hi), 4)
        adj[lab] = {"delta_served_minus_platt": d, "n": int(
            ((served_seal >= lo) & (served_seal < hi)).sum()),
            "bar": "no regression beyond +0.01"}
    g["leg3_sealed_adjacent_bands_no_regression"] = {
        **adj, "pass": all(v["delta_served_minus_platt"] <= 0.01 for v in adj.values())}

    auc_raw_pool = _auc(y_pool, raw_pool)
    auc_raw_seal = _auc(y_seal, raw_seal)
    auc_served_pool = _auc(y_pool, served_pool)
    auc_served_seal = _auc(y_seal, served_seal)
    g["leg4_auc_flat_within_0_001"] = {
        "pooled_raw": round(auc_raw_pool, 6), "pooled_served": round(auc_served_pool, 6),
        "sealed_raw": round(auc_raw_seal, 6), "sealed_served": round(auc_served_seal, 6),
        "max_abs_delta": round(max(abs(auc_raw_pool - auc_served_pool),
                                   abs(auc_raw_seal - auc_served_seal)), 6),
        "bar": "<= 0.001 (rank-invariant contract)",
        "pass": max(abs(auc_raw_pool - auc_served_pool),
                    abs(auc_raw_seal - auc_served_seal)) <= 0.001}

    in_strat = raw_seal > h_star
    mean_abs_change = float(np.abs(served_seal[in_strat]
                                   - served_platt_seal[in_strat]).mean()) if in_strat.sum() else 0.0
    band_gain = (band_ece(y_seal, served_platt_seal, 0.70, 0.80)
                 - band_ece(y_seal, served_seal, 0.70, 0.80))
    # noise ~ SE of the band actual rate at n = in-band sealed n
    n_band = int(((served_seal >= 0.70) & (served_seal < 0.80)).sum())
    band_noise = float(np.sqrt(max(0.0, served_seal.mean() * (1 - served_seal.mean())
                                   / max(n_band, 1)))) if n_band else float("nan")
    g["leg5_worth_having"] = {
        "sealed_band_gain_platt_minus_served": round(band_gain, 4),
        "band_noise_se": round(band_noise, 4) if n_band else None,
        "mean_abs_change_in_stratum": round(mean_abs_change, 4),
        "n_in_stratum_sealed": int(in_strat.sum()),
        "bar": "band gain > noise/3 AND mean |change| > 0.005",
        "pass": bool(n_band and band_gain > band_noise / 3.0
                     and mean_abs_change > 0.005)}

    served_min, served_max = float(served_seal.min()), float(served_seal.max())
    # monotone check on the local segment (raw>h_star, deployed map)
    mseg = raw_seal > h_star
    if mseg.sum() > 1:
        _o = np.argsort(raw_seal[mseg])
        mono = bool(np.all(np.diff(served_seal[mseg][_o]) >= -1e-12))
    else:
        mono = True
    # seam jump: served just above h vs the global Platt at h (0.0 for the
    # continuous L/P families; negative for I_identity = the discontinuity)
    seam_served = (float(_apply_family(fam_star, mapper_d, np.array([h_star]),
                                       h_star, anchor_d)[0])
                   if mapper_d is not None else float(h_star))
    jump = seam_served - float(_apply_platt(ma_deploy, np.array([h_star]))[0])
    g["leg6_no_bleed_monotone"] = {
        "served_min": round(served_min, 6), "served_max": round(served_max, 6),
        "in_range_0_1": bool(served_min >= 0.0 and served_max <= 1.0),
        "local_segment_monotone": mono,
        "seam_jump_served_minus_platt_at_h": round(jump, 4),
        "pass": bool(served_min >= 0.0 and served_max <= 1.0 and mono
                     and abs(jump) <= 1e-6)}

    out["gate_legs"] = g
    legs_ok = all(g[k]["pass"] for k in ("leg1_sealed_s2_70_80_band_ece_le_0_03",
                                         "leg2_pooled_cal_logloss_within_0_001",
                                         "leg3_sealed_adjacent_bands_no_regression",
                                         "leg4_auc_flat_within_0_001",
                                         "leg5_worth_having",
                                         "leg6_no_bleed_monotone"))
    out["verdict"] = {
        "result": "GATE_PASS" if legs_ok else "GATE_FAIL",
        "selection": {"h_star": h_star, "family_star": fam_star},
        "note": ("GATE_PASS -> serve-time local map adoption commit wiring "
                 "the chosen family/threshold from the recorded fit; "
                 "GATE_FAIL -> record + follow-ons, production untouched"),
    }

    blob = json.dumps(out, indent=2, default=str)
    print(blob)
    sys.stderr.write(f"sha256: {hashlib.sha256(blob.encode()).hexdigest()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())