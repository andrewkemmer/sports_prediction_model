"""Production monitoring / diagnostics.

Emits the ``nfl_model_monitor_<date>.json`` record the shared frontend
monitor page renders (MLB-shaped schema): feature drift (PSI), coverage,
ensemble member diagnostics, rolling OOF Brier, and version history.

All statistics derive from OOF predictions and point-in-time features —
never from full-history target relationships used for selection.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

try:
    from backend import config
except ImportError:
    import config

logger = logging.getLogger(__name__)

PSI_WARN = 0.10
PSI_ALERT = 0.25


def feature_status(psi: float) -> str:
    """The module's PSI status rule (the same thresholds feature_drift
    applies): ALERT >= 0.25, WARN >= 0.10, else OK. Non-finite PSI -> OK
    (the drift row is absent, not alarming)."""
    if not np.isfinite(psi):
        return "OK"
    if psi >= PSI_ALERT:
        return "ALERT"
    if psi >= PSI_WARN:
        return "WARN"
    return "OK"


def _psi(current: np.ndarray, baseline: np.ndarray, n_bins: int = 10) -> float:
    """Population stability index between the current-window and baseline
    distributions of a feature (quantile-binned on the baseline)."""
    c = pd.Series(current).dropna()
    b = pd.Series(baseline).dropna()
    if len(c) < 10 or len(b) < 10:
        return np.nan
    try:
        qs = np.unique(np.quantile(b, np.linspace(0, 1, n_bins + 1)))
    except Exception:
        return np.nan
    if len(qs) < 2:
        return np.nan
    qb = np.clip(np.searchsorted(qs, b, side="right") - 1, 0, len(qs) - 2)
    qc = np.clip(np.searchsorted(qs, c, side="right") - 1, 0, len(qs) - 2)
    pb = np.bincount(qb, minlength=len(qs) - 1) / len(b)
    pc = np.bincount(qc, minlength=len(qs) - 1) / len(c)
    pb, pc = np.clip(pb, 1e-6, None), np.clip(pc, 1e-6, None)
    return float(np.sum((pc - pb) * np.log(pc / pb)))


def feature_drift(full_df: pd.DataFrame, recent_df: pd.DataFrame,
                  weights: dict[str, float] | None = None) -> list[dict]:
    """PSI per served feature: recent slate window vs full-history baseline.

    Rows carry the MLB-shaped fields the shared monitor page renders:
    ``status`` (OK/WARN/ALERT from the same PSI thresholds MLB uses),
    ``weight_pct`` (the ensemble's blend-weighted feature importance, when
    member importances are available), and ``n_baseline`` / ``n_current``
    sample sizes behind each comparison. NFL's own PSI values and windows —
    nothing copied from MLB.
    """
    wmap = weights or {}
    rows = []
    for f in config.FEATURE_COLUMNS:
        if f not in full_df.columns:
            continue
        psi = _psi(recent_df[f].to_numpy(float), full_df[f].to_numpy(float))
        mean_cur = (float(np.nanmean(recent_df[f])) if len(recent_df) else np.nan)
        mean_base = float(np.nanmean(full_df[f]))
        mean_shift = mean_cur - mean_base
        se_cur = (float(np.nanstd(recent_df[f]) / np.sqrt(len(recent_df)))
                  if len(recent_df) > 1 else np.nan)
        se_base = (float(np.nanstd(full_df[f]) / np.sqrt(len(full_df)))
                   if len(full_df) > 1 else np.nan)
        shift_se = float(np.hypot(se_cur, se_base)) if np.isfinite(se_cur) \
            and np.isfinite(se_base) else np.nan
        rows.append({
            "feature": f,
            "current_mean": mean_cur,
            "baseline_mean": mean_base,
            "psi": psi,
            "psi_adjusted": psi,
            "status": ("ALERT" if (np.isfinite(psi) and psi >= PSI_ALERT)
                       else "WARN" if (np.isfinite(psi) and psi >= PSI_WARN)
                       else "OK"),
            "weight_pct": (round(100.0 * float(wmap.get(f, 0.0)), 2)
                           if wmap.get(f) else None),
            "n_baseline": int(full_df[f].notna().sum()),
            "n_current": int(recent_df[f].notna().sum()) if len(recent_df) else 0,
        })
    return rows


def coverage(full_df: pd.DataFrame) -> list[dict]:
    """Per-feature measured/non-null coverage over the decided pool.

    MLB-shaped fields: ``status`` (STARVED <25% measured / LOW_COVERAGE
    <80% / OK — the same thresholds the shared page documents) and
    ``n_default_zero``. The NFL engine does not default-fill features (NaN
    routes to imputation at fit time), so every present non-null value is a
    real measurement: pct_measured == pct_nonnull and n_default_zero is 0.
    """
    rows = []
    for f in config.FEATURE_COLUMNS:
        if f not in full_df.columns:
            rows.append({"feature": f, "window": "decided pool",
                         "n_games": len(full_df), "pct_measured": 0.0,
                         "pct_nonnull": 0.0, "n_default_zero": 0,
                         "status": "STARVED"})
            continue
        v = pd.to_numeric(full_df[f], errors="coerce")
        pct = round(100.0 * float(v.notna().mean()), 2)
        rows.append({
            "feature": f, "window": "decided pool", "n_games": int(len(full_df)),
            "pct_measured": pct,
            "pct_nonnull": pct,
            "n_default_zero": 0,
            "status": ("STARVED" if pct < 25.0
                       else "LOW_COVERAGE" if pct < 80.0 else "OK"),
        })
    return rows


def ensemble_table(oof: pd.DataFrame, weights: dict[str, float],
                   cal_p: np.ndarray | None = None) -> list[dict]:
    """Per-member OOF diagnostics + earned adaptive weights."""
    y = oof["home_win"].to_numpy(float)
    rows = []
    n = len(oof)
    for name in config.ENSEMBLE_MEMBERS:
        col = f"p_{name}"
        if col not in oof.columns:
            continue
        from evaluation import binary_metrics  # local import avoids cycles
        m = binary_metrics(oof[col].to_numpy(float), y)
        rows.append({
            "name": name, "weight": round(float(weights.get(name, 0.0)), 4),
            "auc": m.get("auc"), "brier": m.get("brier"),
            "logloss": m.get("logloss"), "n_eval": m.get("n"),
        })
    return rows


def rolling_brier(oof: pd.DataFrame, p_col: str = "p_ensemble_calibrated",
                  window_days: int = 30) -> list[dict]:
    """Per-game rolling Brier over the OOF timeline (MLB-shaped rows).

    Each day carries its decided-game count in ``games`` (the field the
    shared Rolling Brier section's sparse-day caption reads).
    """
    if p_col not in oof.columns:
        return []
    df = oof.dropna(subset=[p_col]).copy()
    df["gameday"] = pd.to_datetime(df["gameday"])
    df = df.sort_values("gameday")
    df["brier"] = (df[p_col] - df["home_win"]) ** 2
    out = []
    for day, grp in df.groupby(df["gameday"].dt.date):
        out.append({"date": str(day), "brier": float(grp["brier"].mean()),
                    "games": int(len(grp))})
    return out


def _dump_json(path, record: dict) -> None:
    """JSON-strict dump: NaN/Inf -> None (serving parity with serving.py)."""
    def _safe(o):
        if isinstance(o, dict):
            return {k: _safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_safe(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    path.write_text(json.dumps(_safe(record), indent=1, allow_nan=False,
                               default=str))


# ---------------------------------------------------------------------------
# Run-Line & Totals Monitor (run-engine counterpart of the moneyline
# monitor artifact — MLB ``run_engine_monitor_*.json`` shape, NFL data)
# ---------------------------------------------------------------------------

def _grid_col(base: str, x: float) -> str:
    """Artifact grid column tag (mirror of the frontend's ``_col``):
    '-' -> 'm', '.' -> '_', and an integral line drops its trailing
    '.0'. ``p_push_m3`` = P(margin == -3); ``p_push_42`` = P(total == 42)."""
    s = str(float(x))
    if s.endswith(".0"):
        s = s[:-2]
    return f"{base}_{s.replace('-', 'm').replace('.', '_')}"


def _reliability_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Decile reliability ECE over finite (y, p) pairs (frontend mirror)."""
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 20:
        return float("nan")
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0 + 1e-12
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    errs = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        errs.append(abs(float(p[m].mean()) - float(y[m].mean())))
    return float(np.mean(errs)) if errs else float("nan")


def _rank_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    """Rank-based AUC (Mann-Whitney U); None for a single-class pool."""
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype=float)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def _markets_card_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """MLB winner-card metric shape: actual_win_rate == win_rate == the
    empirical pick win rate (push-excluded), predicted_mean = pooled picked
    -side probability mean, plus AUC / ECE-cal / Brier / Logloss."""
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], np.clip(p[ok], 1e-6, 1 - 1e-6)
    out: dict = {"n": int(len(y)), "actual_win_rate": None, "win_rate": None,
                 "predicted_mean": None, "auc": None, "ece_raw": None,
                 "ece_calibrated": None, "brier": None, "logloss": None}
    if len(y) == 0:
        return out
    rate = round(float(y.mean()), 4)
    out["actual_win_rate"] = rate
    out["win_rate"] = rate
    out["predicted_mean"] = round(float(p.mean()), 4)
    out["brier"] = round(float(((p - y) ** 2).mean()), 4)
    # The run engine ships no separate calibration map — the raw reliability
    # ECE IS the calibrated figure (honest, same number the frontend shows).
    ece = _reliability_ece(y, p)
    out["ece_raw"] = round(ece, 4)
    out["ece_calibrated"] = round(ece, 4)
    out["logloss"] = _markets_logloss(y, p)
    auc = _rank_auc(y, p)
    out["auc"] = round(auc, 4) if auc is not None else None
    return out


def _markets_logloss(y: np.ndarray, p: np.ndarray) -> float | None:
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], np.clip(p[ok], 1e-6, 1 - 1e-6)
    if len(y) < 2:
        return None
    return round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4)


def markets_winner_cards(oof_rows: pd.DataFrame) -> dict[str, dict]:
    """The three binary winner cards computed from the decided OOF market
    rows — the same pick basis the frontend recomputes for rendering
    (nfl_market_diagnostics.winner_cards): 2-way no-push rescale, whole
    -number pushes excluded, PICK-SIDE framing for derived_ml (every
    metric on the picked side). The artifact ships the pooled card so the
    monitor history and downstream consumers see the exact values the
    frontend recomputes — one source of truth, never two.

    Cards: over_under (P(over) at each game's own fair total), run_line
    (favorite cover at its derived magnitude), derived_ml (the run-engine
    model's own moneyline).
    """
    def _pairs(kind: str, df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        for _, r in df.iterrows():
            if kind == "over_under":
                po = r.get("p_over_fair")
                U = r.get("fair_total")
                tot = r.get("total")
                try:
                    po, U, tot = float(po), float(U), float(tot)
                except (TypeError, ValueError):
                    continue
                if not (np.isfinite(U) and np.isfinite(tot)):
                    continue
                u_int = int(np.clip(round(U), config.TOTAL_GRID[0],
                                    config.TOTAL_GRID[-1]))
                # Pre-grid artifacts ship p_over_fair as all-null: fall
                # back to the SAME integer grid column (p_over_<U>) —
                # fair_total IS the integer median, so p_over_<U> IS the
                # fair-line over leg exactly (the frontend mirror does
                # the same, one source of truth).
                if not np.isfinite(po):
                    po = r.get(_grid_col("p_over", u_int))
                    try:
                        po = float(po)
                    except (TypeError, ValueError):
                        continue
                if not np.isfinite(po):
                    continue
                pp = r.get(_grid_col("p_push", u_int))
                try:
                    pp = float(pp)
                except (TypeError, ValueError):
                    pp = float("nan")
                pu = 1.0 - po - (pp if np.isfinite(pp) else 0.0)
                if po + pu <= 0:
                    continue
                if tot == U:          # whole-line push — 2-way excluded
                    continue
                rows.append({"p": po / (po + pu), "y": float(tot > U)})
            elif kind == "run_line":
                fs, margin, hw = (r.get("fair_spread"), r.get("margin"),
                                  r.get("derived_ml"))
                try:
                    fs, margin, hw = float(fs), float(margin), float(hw)
                except (TypeError, ValueError):
                    continue
                if not all(np.isfinite(v) for v in (fs, margin, hw)):
                    continue
                home_fav = hw >= 0.5
                m = int(max(abs(fs), 0.5))
                if home_fav:
                    cov = r.get(_grid_col("p_home_cover", m))
                    push_p = r.get(_grid_col("p_push", m))
                else:
                    ph = r.get(_grid_col("p_home_cover", -m))
                    push_p = r.get(_grid_col("p_push", -m))
                try:
                    cov, push_p = float(cov), float(push_p)
                    if not home_fav:
                        cov = 1.0 - cov - push_p   # away-favored cover leg
                except (TypeError, ValueError):
                    continue
                dog = 1.0 - cov - push_p
                if not (cov + dog) > 0:
                    continue
                if (margin == m and home_fav) or (margin == -m
                                                  and not home_fav):
                    continue
                covered = (margin > m) if home_fav else (margin < -m)
                rows.append({"p": cov / (cov + dog), "y": float(covered)})
            else:  # derived_ml — PICK-SIDE framing
                ml, margin = r.get("derived_ml"), r.get("margin")
                try:
                    ml, margin = float(ml), float(margin)
                except (TypeError, ValueError):
                    continue
                if not (np.isfinite(ml) and np.isfinite(margin)):
                    continue
                if ml >= 0.5:
                    rows.append({"p": ml, "y": float(margin > 0)})
                else:
                    rows.append({"p": 1.0 - ml, "y": float(margin < 0)})
        return pd.DataFrame(rows)

    def _build(kind: str, df: pd.DataFrame) -> dict:
        view = _pairs(kind, df)
        if not len(view):
            return {}
        card = _markets_card_metrics(view["y"].to_numpy(float),
                                     view["p"].to_numpy(float))
        hold: dict = {}
        if "frame_view" in df.columns and (df["frame_view"] == "sealed").any():
            hv = _pairs(kind, df.loc[df["frame_view"] == "sealed"])
            if len(hv):
                hold = _markets_card_metrics(hv["y"].to_numpy(float),
                                             hv["p"].to_numpy(float))
        card["holdout"] = hold
        card["source"] = "oof_decided_store"
        return card

    if oof_rows is None or not len(oof_rows):
        return {}
    return {"over_under": _build("over_under", oof_rows),
            "run_line": _build("run_line", oof_rows),
            "derived_ml": _build("derived_ml", oof_rows)}


def _run_engine_fit_block(oof_market_rows: pd.DataFrame) -> dict:
    """Run-engine fit diagnostics computed from the artifact's OWN rows —
    the data behind the fit panel's tail/variance captions (MLB ships the
    same anatomy from its NB sampler; the NFL engine is the pinned 76×76
    joint, so the modeled legs come from the artifact's grid columns).
    Row-derived only: nothing here is a constant."""
    out: dict = {}
    df = oof_market_rows
    if df is None or not len(df):
        return out
    total = pd.to_numeric(df.get("total"), errors="coerce")
    margin = pd.to_numeric(df.get("margin"), errors="coerce")

    def _col_mean(base: str, line: float) -> float:
        col = _grid_col(base, line)
        if col not in df.columns:
            return float("nan")
        return float(pd.to_numeric(df[col], errors="coerce").mean())

    # Total tail (the totals-law check the Distribution tab callouts use):
    # modeled = pooled grid legs, observed = decided scores.
    if total.notna().any():
        mod_ge = _col_mean("p_over", 59.0)          # P(total >= 60)
        mod_le = 1.0 - _col_mean("p_over", 35.0)    # P(total <= 35)
        obs_ge = float((total >= 60).mean())
        obs_le = float((total <= 35).mean())
        if all(np.isfinite(v) for v in (mod_ge, mod_le, obs_ge, obs_le)):
            out["total_tail"] = {
                "k_ge": 60, "obs_ge": round(obs_ge, 4),
                "mod_ge": round(mod_ge, 4),
                "k_le": 35, "obs_le": round(obs_le, 4),
                "mod_le": round(mod_le, 4),
            }
    # Margin tail: home-win / push band observed vs the joint's pooled legs.
    if margin.notna().any():
        ml_col = "derived_ml"
        ml_mean = (float(pd.to_numeric(df[ml_col], errors="coerce").mean())
                   if ml_col in df.columns else float("nan"))
        push_mean = _col_mean("p_push", 0.0)        # P(margin == 0)
        obs_home = float((margin > 0).mean())
        obs_push = float((margin == 0).mean())
        if all(np.isfinite(v) for v in (ml_mean, push_mean,
                                        obs_home, obs_push)):
            out["margin_tail"] = {
                "obs_home_win": round(obs_home, 4),
                "mod_home_win": round(ml_mean, 4),
                "obs_push": round(obs_push, 4),
                "mod_push": round(push_mean, 4),
            }
        # Per-side residual dispersion (the variance-check "obs" legs —
        # the pinned era sigmas are score-RESIDUAL SDs around mu, so the
        # observed legs are std(score − mu) over the decided rows; the
        # "implied" legs are the pinned sigma0 values the frontend
        # renders — never duplicated here).
        mu_h = pd.to_numeric(df.get("mu_h"), errors="coerce")
        mu_a = pd.to_numeric(df.get("mu_a"), errors="coerce")
        hs = pd.to_numeric(df.get("home_score"), errors="coerce")
        as_ = pd.to_numeric(df.get("away_score"), errors="coerce")
        res_h = hs - mu_h
        res_a = as_ - mu_a
        ok_h = res_h.notna()
        ok_a = res_a.notna()
        out["variance_obs"] = {
            "home": (round(float(res_h[ok_h].std(ddof=1)), 2)
                     if ok_h.sum() > 1 else None),
            "away": (round(float(res_a[ok_a].std(ddof=1)), 2)
                     if ok_a.sum() > 1 else None),
        }
    return out


def _run_engine_line_pairs(df: pd.DataFrame, line_kind: str,
                           line: float) -> tuple[np.ndarray, np.ndarray]:
    """(p, y) for one market line, 2-way push-excluded — the model-card
    per-line scoring basis: over_<U> (total > U, pushes on total == U),
    home_cover_<L> (margin > L, pushes on margin == L), derived_ml (2-way
    re-scaled, ties dead mass). NaN legs drop the row."""
    ps: list[float] = []
    ys: list[float] = []
    for _, r in df.iterrows():
        if line_kind == "over":
            tot = r.get("total")
            po = r.get(_grid_col("p_over", line))
            pp = r.get(_grid_col("p_push", line))
            try:
                tot, po, pp = float(tot), float(po), float(pp)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(tot) and np.isfinite(po)):
                continue
            pu = (1.0 - po - pp) if np.isfinite(pp) else (1.0 - po)
            if po + pu <= 0:
                continue
            if tot == line:
                continue
            ps.append(po / (po + pu))
            ys.append(float(tot > line))
        elif line_kind == "spread":
            margin = r.get("margin")
            ph = r.get(_grid_col("p_home_cover", line))
            pp = r.get(_grid_col("p_push", line))
            try:
                margin, ph, pp = float(margin), float(ph), float(pp)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(margin) and np.isfinite(ph)):
                continue
            pa = (1.0 - ph - pp) if np.isfinite(pp) else (1.0 - ph)
            if ph + pa <= 0:
                continue
            if margin == line:
                continue
            ps.append(ph / (ph + pa))
            ys.append(float(margin > line))
        else:  # derived moneyline — ties excluded, 2-way re-scaled
            margin = r.get("margin")
            ml = r.get("derived_ml")
            try:
                margin, ml = float(margin), float(ml)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(margin) and np.isfinite(ml)) or margin == 0:
                continue
            ps.append(ml / (ml + (1.0 - ml)))
            ys.append(float(margin > 0))
    return np.asarray(ps, dtype=float), np.asarray(ys, dtype=float)


def _run_engine_market_metrics(oof_market_rows: pd.DataFrame) -> dict:
    """Per-line OOF metrics for the Run-Engine Model card (MLB
    ``market_metrics`` shape): ECE/Brier/Logloss per canonical line over
    the artifact's own rows. The run engine ships no calibration map, so
    ECE-raw IS ECE-cal (the honest figure the frontend shows)."""
    df = oof_market_rows
    if df is None or not len(df):
        return {}
    out: dict = {}

    def _add(key: str, y: np.ndarray, p: np.ndarray) -> None:
        if not len(y):
            return
        m = _markets_card_metrics(y, p)
        out[key] = {
            "engine_ece_raw": m.get("ece_raw"),
            "engine_ece_calibrated": m.get("ece_calibrated"),
            "engine_brier": m.get("brier"),
            "engine_logloss": m.get("logloss"),
            "n": m.get("n"),
        }

    for u in config.RUN_ENGINE_FIXED_TOTALS:
        y, p = _run_engine_line_pairs(df, "over", float(u))
        _add(f"over_{u}", y, p)
    for l in config.RUN_ENGINE_CANONICAL_SPREADS:
        y, p = _run_engine_line_pairs(df, "spread", float(l))
        _add(f"home_cover_{str(l).replace('.', '_')}", y, p)
    y, p = _run_engine_line_pairs(df, "ml", 0.0)
    _add("derived_moneyline", y, p)
    return out


def write_markets_monitor_json(path, run_date: str,
                               oof_market_rows: pd.DataFrame,
                               config_meta: dict | None = None,
                               ml_reference: dict | None = None) -> dict:
    """``Run-Line & Totals Monitor`` artifact (MLB run-engine-monitor/v2
    shape, NFL data): the winner cards computed from THIS run's decided OOF
    store, the per-line ``market_metrics`` + row-derived ``fit`` block for
    the model card / fit panel, the markets_persisted flags, and a
    ``slate_history`` point per card dated today (the frontend folds the
    dated monitors' accumulating histories into the rolling table).
    ``ml_reference`` is the shared moneyline ensemble's pooled win rate
    (the derived-ML card's comparison anchor) — passed in by the caller
    that owns the moneyline OOF store. Nothing fabricated: cards from the
    artifact's own rows, empty when the store is empty."""
    iso_date = (f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:8]}"
                if len(str(run_date)) == 8 and str(run_date).isdigit()
                else str(run_date))
    cards = markets_winner_cards(oof_market_rows)
    slate_history: list[dict] = []
    for card_key in ("over_under", "run_line", "derived_ml"):
        c = cards.get(card_key) or {}
        if not c.get("n"):
            continue
        slate_history.append({
            "card": card_key,
            "date": iso_date,
            "ece_calibrated": c.get("ece_calibrated"),
            "brier": c.get("brier"),
            "logloss": c.get("logloss"),
            "predicted_mean": c.get("predicted_mean"),
            "n": c.get("n"),
        })
    if ml_reference and cards.get("derived_ml"):
        cards["derived_ml"]["ml_reference"] = dict(ml_reference)
    record = {
        "schema": "run-engine-monitor/v2",
        "date": iso_date,
        "markets_persisted": True,
        "markets_persist_error": None,
        "winner_cards": cards,
        "slate_history": slate_history,
        "fit": _run_engine_fit_block(oof_market_rows),
        "market_metrics": _run_engine_market_metrics(oof_market_rows),
        "config": config_meta or {},
    }
    _dump_json(path, record)
    return record


def write_monitor_json(path, run_date: str, drift: list[dict],
                       cov: list[dict], ensemble: list[dict],
                       rb: list[dict], baseline: float,
                       config_meta: dict, fold_info: dict,
                       metrics: dict | None = None,
                       platt: dict | None = None) -> dict:
    """MLB-shaped monitor artifact (frontend presentation contract).

    All rendering fields the shared monitor page reads are present:
    ISO retrain dates (+ same-day notes), the dense ``rolling_brier_meta``
    (window_days / min_games_per_day / excluded_sparse_days /
    calibrator_is_identity / map_scope_note), and a version-history row with
    pooled AUC / logloss / calibrated ECE + the deployed Platt map. All
    values are the NFL pipeline's own outputs. The *_note fields are None
    (MLB's emitter ships no notes) so the shared page renders the identical
    fallback presentation for both sports.
    """
    iso_date = f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:8]}" \
        if len(str(run_date)) == 8 and str(run_date).isdigit() else str(run_date)
    next_date = iso_date  # retrains every run — next run is tonight's run
    baseline_label = "Constant home-edge" if np.isfinite(baseline) else "n/a"
    m = metrics or {}
    cal = (platt if isinstance(platt, dict) and platt.get("a") is not None
           and platt.get("b") is not None else None)
    version_row: dict = {
        "version": run_date, "date": iso_date,
        "weights": {r["name"]: r["weight"] for r in ensemble},
        "auc": m.get("auc") or (ensemble[0].get("auc") if ensemble else None),
        "logloss": m.get("logloss"),
        "ece_calibrated": m.get("ece_calibrated") or m.get("ece"),
        "note": "rebuild run",
    }
    if cal:
        version_row["calibration"] = {"a": cal["a"], "b": cal["b"]}
    record = {
        "last_retrained": iso_date,
        # MLB's emitter ships no *_note fields — the shared frontend falls
        # back to its own presentation ("Model healthy — today" / "tonight"),
        # so the NFL artifact presents the same empty-note contract instead
        # of overriding the rendered KPI subtitles with NFL-specific copy.
        "last_retrained_note": None,
        "next_retrain": next_date,
        "next_retrain_note": None,
        # MLB's artifact has no upset_note -> the banner renders the shared
        # 'No note available.' empty state. Match it (the NFL upset-rate
        # context lives in the artifact's fold/metrics blocks, not here).
        "upset_note": None,
        "feature_drift": drift,
        "features_metadata": {r["feature"]: {"definition": "see backend/manifest.py",
                                             "source": "nflverse / stadiums table"}
                              for r in cov},
        "feature_coverage": cov,
        "ensemble": ensemble,
        "rolling_brier": rb,
        "brier_baseline": baseline,
        "brier_baseline_label": baseline_label,
        "rolling_brier_meta": {
            "window_days": 30,
            "min_games_per_day": 1,
            "excluded_sparse_days": 0,
            "calibrator_is_identity": False,
            "map_scope_note": ("Points use the deployed Platt map (fit on all "
                               "OOF games)."),
        },
        "version_history": [version_row],
        "fold_geometry": fold_info,
        "config": config_meta,
    }
    _dump_json(path, record)
    return record
