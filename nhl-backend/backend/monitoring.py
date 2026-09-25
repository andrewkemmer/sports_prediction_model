"""Production monitoring / diagnostics.

Structural mirror of the NFL monitoring.py. Emits the
``nhl_model_monitor_<date>.json`` record the shared frontend monitor page
renders (MLB-shaped schema): feature drift (PSI), coverage, ensemble member
diagnostics, rolling OOF Brier, and version history — plus the run-engine
counterpart ``nhl_run_engine_monitor_<date>.json`` (winner cards, fit block,
per-line market metrics, slate history).

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
    from backend import features as feat_mod
except ImportError:
    import config
    import features as feat_mod

logger = logging.getLogger(__name__)

PSI_WARN = 0.10
PSI_ALERT = 0.25


def feature_status(psi: float) -> str:
    """The module's PSI status rule (the same thresholds feature_drift
    applies): ALERT >= 0.25, WARN >= 0.10, else OK."""
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


def feature_importance_weights(models: dict,
                               member_weights: dict[str, float],
                               feature_frame: pd.DataFrame | None = None) -> dict[str, float]:
    """Return blend-weighted importance for each served feature."""
    importance = {f: 0.0 for f in config.active_moneyline_feature_cols()}
    for name, entry in (models or {}).items():
        model = entry.get("model") if isinstance(entry, dict) else entry
        if model is None:
            continue
        raw = getattr(model, "feature_importances_", None)
        if raw is not None:
            values = np.asarray(raw, dtype=float)
            columns = feat_mod.tree_view(feature_frame if feature_frame is not None
                                         else pd.DataFrame()).columns.tolist()
        else:
            coef = getattr(model, "coef_", None)
            if coef is None:
                continue
            values = np.abs(np.asarray(coef, dtype=float).reshape(-1))
            columns = feat_mod.linear_feature_columns()
        if len(values) != len(columns):
            continue
        total = float(np.nansum(values))
        if not np.isfinite(total) or total <= 0:
            continue
        model_weight = float(member_weights.get(name, 0.0))
        for column, value in zip(columns, values):
            if column in importance and np.isfinite(value):
                importance[column] += model_weight * float(value) / total
    total = sum(importance.values())
    if total <= 0:
        return {f: 0.0 for f in config.active_moneyline_feature_cols()}
    return {f: float(v / total) for f, v in importance.items()}


def feature_drift(full_df: pd.DataFrame, recent_df: pd.DataFrame,
                  weights: dict[str, float] | None = None) -> list[dict]:
    """PSI per served feature: recent slate window vs full-history baseline."""
    wmap = weights or {}
    has_weight_map = weights is not None
    rows = []
    for f in config.active_moneyline_feature_cols():
        if f not in full_df.columns:
            continue
        psi = _psi(recent_df[f].to_numpy(float), full_df[f].to_numpy(float))
        mean_cur = (float(np.nanmean(recent_df[f])) if len(recent_df) else np.nan)
        mean_base = float(np.nanmean(full_df[f]))
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
                           if has_weight_map else None),
            "n_baseline": int(full_df[f].notna().sum()),
            "n_current": int(recent_df[f].notna().sum()) if len(recent_df) else 0,
        })
    return rows


def write_run_engine_feature_artifacts(out_dir, date_c: str,
                                       full_df: pd.DataFrame,
                                       recent_df: pd.DataFrame,
                                       weights: dict[str, float] | None = None,
                                       slate_df: pd.DataFrame | None = None
                                       ) -> tuple[str, str]:
    """Emit MLB-shaped run-engine drift/coverage CSVs for the NHL page."""
    drift = feature_drift(full_df, recent_df, weights=weights)
    cov = coverage(full_df, slate_df=slate_df)
    drift_path = out_dir / f"run_engine_feature_drift_{date_c}.csv"
    cov_path = out_dir / f"run_engine_feature_coverage_{date_c}.csv"
    pd.DataFrame(drift).to_csv(drift_path, index=False)
    pd.DataFrame(cov).to_csv(cov_path, index=False)
    return drift_path.name, cov_path.name


def _warmup_mask(df: pd.DataFrame) -> pd.Series | None:
    """True where an observation was IMPOSSIBLE: a team's own first game.

    A trailing feature is null on a team's debut because there is no prior
    history anywhere — that is the designed warm-up, not a defect. Knowing
    which rows those are is what lets the coverage report tell a cold start
    apart from a feature that silently stopped being produced. Returns None
    when the frame cannot support the classification (no team columns), in
    which case callers fall back to the plain threshold rule.
    """
    cols = set(df.columns)
    if not {"home_team", "away_team", "gameday"} <= cols:
        return None
    order = df.copy()
    order["gameday"] = pd.to_datetime(order["gameday"], errors="coerce")
    order = order.sort_values(["gameday"], kind="stable")
    seen: set[str] = set()
    cold = []
    for r in order.itertuples(index=False):
        h, a = str(r.home_team), str(r.away_team)
        cold.append(h not in seen or a not in seen)
        seen.add(h)
        seen.add(a)
    mask = pd.Series(cold, index=order.index)
    return mask.reindex(df.index).fillna(False).astype(bool)


def _coverage_row(f: str, df: pd.DataFrame, window: str,
                  warmup: pd.Series | None) -> dict:
    n_games = int(len(df))
    if f not in df.columns:
        # The contract names a feature the frame never produced. That is a
        # wiring defect, and it must not be reported as merely unmeasured.
        return {
            "feature": f, "window": window, "n_games": n_games,
            "pct_measured": 0.0, "pct_nonnull": 0.0, "n_default_zero": 0,
            "status": "STARVED", "n_measured": 0, "n_null": n_games,
            "n_cold_null": 0, "n_warm_null": n_games,
            "pct_measured_eligible": 0.0,
            "cause": "absent_column",
        }
    v = pd.to_numeric(df[f], errors="coerce")
    null = v.isna()
    n_null = int(null.sum())
    pct = round(100.0 * float((~null).mean()) if n_games else 0.0, 2)
    if warmup is not None and n_null:
        warm = ~warmup.reindex(df.index).fillna(False).astype(bool)
        n_warm_null = int((null & warm).sum())
        n_cold_null = n_null - n_warm_null
    else:
        n_warm_null, n_cold_null = n_null, 0
    n_eligible = max(n_games - n_cold_null, 0)
    pct_eligible = round(100.0 * (n_eligible - n_warm_null) / n_eligible, 2) \
        if n_eligible else 0.0
    if n_games and not n_eligible:
        # Nothing was measurable: either the whole frame is warm-up, or the
        # feature is null on every game it could have been measured on.
        status = "STARVED" if pct == 0.0 else "OK"
    elif n_warm_null:
        status = "STARVED" if pct == 0.0 else "LOW_COVERAGE"
    else:
        status = "OK"
    return {
        "feature": f, "window": window, "n_games": n_games,
        "pct_measured": pct, "pct_nonnull": pct, "n_default_zero": 0,
        "status": status,
        "n_measured": int(n_games - n_null), "n_null": n_null,
        "n_cold_null": n_cold_null, "n_warm_null": n_warm_null,
        "pct_measured_eligible": pct_eligible,
        "cause": ("defect" if n_warm_null
                  else "cold_start" if n_cold_null else "complete"),
    }


def coverage(full_df: pd.DataFrame,
             slate_df: pd.DataFrame | None = None) -> list[dict]:
    """Per-feature coverage over the decided pool AND the serving slate.

    Two windows, one row each, because they answer different questions and
    only measuring one of them hid a total outage: the goalie family read
    96-98% on the decided pool while EVERY published prediction carried a
    null, since the decided builder can resolve an expected starter from the
    game's own boxscore and the slate builder cannot. A report that only
    looks at the decided pool is structurally blind to the worst case, so
    the slate the pipeline actually ships is measured too.

    Within a window, nulls are split into cold-start (a team's first game —
    no prior history exists, by design) and warm (a real defect). Only warm
    nulls drive the status, so the panel's starved/low counters mean
    "something is broken" rather than "the season started".
    """
    warmup = _warmup_mask(full_df)
    rows = [_coverage_row(f, full_df, "decided pool", warmup)
            for f in config.active_moneyline_feature_cols()]
    if slate_df is not None and len(slate_df):
        slate_warmup = pd.Series(False, index=slate_df.index)
        rows.extend(_coverage_row(f, slate_df, "serving slate", slate_warmup)
                    for f in config.active_moneyline_feature_cols())
    return rows


def ensemble_table(oof: pd.DataFrame, weights: dict[str, float],
                   cal_p: np.ndarray | None = None) -> list[dict]:
    """Per-member OOF diagnostics + earned adaptive weights."""
    y = oof["home_win"].to_numpy(float)
    rows = []
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
    """Per-game rolling Brier over the OOF timeline (MLB-shaped rows)."""
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
# monitor artifact — MLB ``run_engine_monitor_*.json`` shape, NHL data)
# ---------------------------------------------------------------------------

def _grid_col(base: str, x: float) -> str:
    """Artifact grid column tag: '-' -> 'm', '.' -> '_', integral line drops
    its trailing '.0'. ``p_push_total_5`` = P(total == 5); the tag is shared
    by the totals namespace (``p_push_total_{U}``) and the spread namespace
    (``p_push_{L}``)."""
    s = str(float(x))
    if s.endswith(".0"):
        s = s[:-2]
    return f"{base}_{s.replace('-', 'm').replace('.', '_')}"
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
    """MLB winner-card metric shape (push-excluded 2-way populations)."""
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
    rows — the same pick basis the frontend recomputes for rendering:
    2-way no-push rescale, whole-number pushes excluded, PICK-SIDE framing
    for derived_ml. Cards: over_under (P(over) at each game's own fair
    total), run_line (favorite cover at its derived magnitude), derived_ml
    (the run-engine model's own moneyline)."""
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
                if not np.isfinite(po):
                    po = r.get(_grid_col("p_over", u_int))
                    try:
                        po = float(po)
                    except (TypeError, ValueError):
                        continue
                if not np.isfinite(po):
                    continue
                pp = r.get(_grid_col("p_push_total", u_int))
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
                else:
                    # The artifact stores home-cover probabilities on one
                    # orientation. For an away favorite, read the negative
                    # spread line, then transform its complement below.
                    cov = r.get(_grid_col("p_home_cover", -m))
                push_p = r.get(_grid_col("p_push", m if home_fav else -m))
                try:
                    cov, push_p = float(cov), float(push_p)
                except (TypeError, ValueError):
                    continue
                if not all(np.isfinite(v) for v in (cov, push_p)):
                    continue
                if not home_fav:
                    cov = 1.0 - cov - push_p   # away-favored cover leg
                if not np.isfinite(cov) or cov < 0 or cov > 1 or push_p < 0:
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
    """Run-engine fit diagnostics computed from the artifact's OWN rows."""
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

    # Total tail: NHL-sized cutpoints (P(total >= 9) / P(total <= 5)).
    if total.notna().any():
        mod_ge = _col_mean("p_over", 8.0)           # P(total >= 9)
        mod_le = 1.0 - _col_mean("p_over", 4.0)     # P(total <= 4)
        obs_ge = float((total >= 9).mean())
        obs_le = float((total <= 4).mean())
        if all(np.isfinite(v) for v in (mod_ge, mod_le, obs_ge, obs_le)):
            out["total_tail"] = {
                "k_ge": 9, "obs_ge": round(obs_ge, 4),
                "mod_ge": round(mod_ge, 4),
                "k_le": 4, "obs_le": round(obs_le, 4),
                "mod_le": round(mod_le, 4),
            }
    # Margin tail: home-win / tie band observed vs the joint's pooled legs.
    if margin.notna().any():
        ml_col = "derived_ml"
        ml_mean = (float(pd.to_numeric(df[ml_col], errors="coerce").mean())
                   if ml_col in df.columns else float("nan"))
        push_mean = _col_mean("p_push", 0.0) if "p_push_0" in df.columns \
            else _col_mean("p_push_total", 0.0)      # P(margin == 0)
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
    """(p, y) for one market line, 2-way push-excluded."""
    ps: list[float] = []
    ys: list[float] = []
    for _, r in df.iterrows():
        if line_kind == "over":
            tot = r.get("total")
            po = r.get(_grid_col("p_over", line))
            pp = r.get(_grid_col("p_push_total", line))
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
    """Per-line OOF metrics for the Run-Engine Model card."""
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
    shape, NHL data)."""
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
    """MLB-shaped monitor artifact (frontend presentation contract)."""
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
        "last_retrained_note": None,
        "next_retrain": next_date,
        "next_retrain_note": None,
        "upset_note": None,
        "feature_drift": drift,
        "features_metadata": {r["feature"]: {"definition": "see backend/manifest.py",
                                             "source": "official NHL API"}
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
