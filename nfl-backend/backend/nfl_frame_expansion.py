"""NFL W2016 frame-expansion measurement — per-side + joint chain (record-only).

Measures what the 2016-2025 decided frame changes for the per-side mean
regressors and the joint layer, WITHOUT touching any engine/runner:

- Step 0 (code-read findings, recorded by the runner): ELO prior 1500 / K=32 /
  scale 400, NO season-boundary reset or regression-to-mean (the rating dict
  persists across all seasons; an unseen team initializes at 1500), NO
  home-field bonus in the ELO update (expected win is symmetric in ratings;
  HFA enters only through the is_home anchor / other features), strictly
  chronological iteration (gameday, game_id, is_home) with ``elo_entering``
  read from strictly-prior games only. EWM: halflife = 2 GAMES per team
  (per-team ``ewm(...).shift(1)`` over that team's own games, same shift(1)
  discipline as the windowed ladder, leak-gated by team_stats_ladder's strict
  monotonicity assertion). Frame-start interpretation: with no reversion, the
  ratings entering 2019 in the W2016 build carry 2016-18 history, and the
  Step-1.5 by-season delta trend measures whether that offset persists or
  decays through the scored window.

- Step 1 (runner): canonical decided frame 2016-2025 (nfl_game_frame rules),
  written to a scratch dir; the runner never mutates the canonical
  data_delivery frame.

- Step 1.5 (this module): feature delta diagnostics on IDENTICAL rows between
  the W2016 build (schedule+pbp 2015-2025, warmup 2015) and the W2019 build
  (schedule 2018-2025 + pbp 2019-2025 — the production pull ranges), plus the
  outcome regression of home_win / home_margin on the elo delta and the
  ewm/rolling/static sanity classification. Decision rule (recorded, NOT a
  gate): negligible => no ELO reset warranted, W2016 reads as a volume test;
  material+signal => the 2016 frame already captures the better strength
  estimate, a reset would lose it; material+noise => frame-start arbitrariness
  is real, spec a season-reset ELO probe.

- Steps 2/3 (runner): by-season away-bias table from the W2016 residual
  artifact + sealed refill, and the W2019 comparison tables read from the
  committed W2019 records.

Harness-geometry note (measured, recorded): the per-side/joint runners build
their folds from ``feats[season in TRAIN_SEASONS]`` (2019-2024), so expanding
the decided frame to 2016 does NOT add training rows through these runners —
the W2016 measurement via the unchanged runners isolates the FEATURE-level
frame-start effect (ELO priors from 2016-18 entering the 2019-24 rows). The
"~800 added training rows" component is only exercised by the window-gate
harness geometry (run_nfl_window_gate builds folds over the candidate's full
train window). This is recorded prominently; it does not bind the per-side
verdicts below.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Season windows ────────────────────────────────────────────────────────────
# W2016 configuration (mirrors run_nfl_window_gate: warmup B-1 + core B..2025).
W2016_WARMUP = 2015
W2016_CORE_START = 2016
W2016_SEASONS = list(range(W2016_WARMUP, 2026))          # 2015..2025
# W2019 (production) configuration — the exact pull ranges load_features uses.
W2019_SEASONS = [2018] + list(range(2019, 2026))         # 2018..2025
SCORED_SEASONS = [2021, 2022, 2023, 2024]                # pooled-OOF window
SEALED_SEASON = 2025

ELO_K = 32.0
ELO_PRIOR = 1500.0

# Decision-rule thresholds (operationalization, recorded in the record):
#   material   — p95(|Δ elo_diff|) on pooled 2021-24 rows >= this many ELO
#                points (one game swing is +-K/2 = +-16 with K=32; 20 = 1.25x).
#   signal     — |t| on the home_win ~ Δelo regression >= this (2-sided 95%
#                at n=1,091 is ~1.96).
MATERIAL_P95_DELTA = 20.0
SIGNAL_T = 2.0

# Sanity bars: EWM deltas on scored rows must be <= this (halflife-2 decay
# over 60-100 prior games => numerically zero); rolling/static = exact 0.
EWM_MAX_ABS_DELTA = 1e-3


# ---------------------------------------------------------------------------
# Pure primitives (offline-testable)
# ---------------------------------------------------------------------------
def decided_rows_from_schedule(sched: pd.DataFrame) -> pd.DataFrame:
    """Decided rows of a raw schedule (post-game only) — mirror of
    nfl_features._decided_rows used for the ELO/EWM ladder timeline."""
    out = sched.copy()
    if out.empty:
        return out
    return out[out[["away_score", "home_score", "result"]].notna().all(axis=1)]


def ladder_elo(sched: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-(game_id, team) elo_entering ladder over the schedule's decided
    rows, plus the per-team final ratings. Pure — uses only the exported
    nfl_features.compute_elo / team_events."""
    from nfl_features import compute_elo, team_events
    full = decided_rows_from_schedule(sched)
    ev = compute_elo(team_events(full))
    lut = {(r["game_id"], str(r["team"])): float(r["elo_entering"])
           for _, r in ev.iterrows()}
    ev2 = ev.sort_values(["team", "gameday", "game_id"])
    final = dict(ev2.drop_duplicates("team", keep="last")
                 .set_index("team")["elo_entering"])
    return ev, lut


def elo_by_side(ev: pd.DataFrame) -> pd.DataFrame:
    """(game_id, home_elo, away_elo) from the long ladder frame."""
    h = ev[ev["is_home"]][["game_id", "elo_entering"]].rename(
        columns={"elo_entering": "home_elo"})
    a = ev[~ev["is_home"]][["game_id", "elo_entering"]].rename(
        columns={"elo_entering": "away_elo"})
    out = h.merge(a, on="game_id", how="outer")
    out["elo_diff_ladder"] = out["home_elo"] - out["away_elo"]
    return out


def elo_delta_stats(df16: pd.DataFrame, df19: pd.DataFrame,
                    shared_ids: np.ndarray) -> list[dict]:
    """Per-season |Δ| / signed-Δ stats for elo_diff, home_elo, away_elo on the
    identical shared rows (2021-24 by default). Δ = W2016 value - W2019 value."""
    a = df16.set_index("game_id")
    b = df19.set_index("game_id")
    ids = [g for g in shared_ids if g in a.index and g in b.index]
    rows: list[dict] = []
    for season in sorted({int(a.loc[g, "season"]) for g in ids}):
        sel = [g for g in ids if int(a.loc[g, "season"]) == season]
        rec = {"season": season, "n": len(sel)}
        for col in ("elo_diff", "home_elo", "away_elo"):
            d = (a.loc[sel, col].astype(float).to_numpy()
                 - b.loc[sel, col].astype(float).to_numpy())
            rec[col] = {
                "mean_abs_delta": round(float(np.abs(d).mean()), 3),
                "p95_abs_delta": round(float(np.quantile(np.abs(d), 0.95)), 3),
                "max_abs_delta": round(float(np.abs(d).max()), 3),
                "mean_signed_delta": round(float(d.mean()), 3),
            }
        rows.append(rec)
    return rows


def ols(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Simple OLS y ~ x: beta, SE, 95% CI (t via scipy), r^2, n. NaN pairs
    dropped. Pure and deterministic."""
    from scipy import stats
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3 or np.allclose(x, x[0]):
        return {"beta": None, "se": None, "ci_lo": None, "ci_hi": None,
                "r2": None, "n": int(n)}
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    sxy = float(((x - xm) * (y - ym)).sum())
    beta = sxy / sxx
    alpha = ym - beta * xm
    resid = y - (alpha + beta * x)
    sigma2 = float((resid ** 2).sum()) / (n - 2)
    se = float(np.sqrt(sigma2 / sxx))
    t_crit = float(stats.t.ppf(0.975, max(n - 2, 1)))
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    return {"beta": round(beta, 5), "se": round(se, 5),
            "ci_lo": round(beta - t_crit * se, 5),
            "ci_hi": round(beta + t_crit * se, 5),
            "t": round(beta / se, 3) if se > 0 else None,
            "r2": round(r2, 6), "n": int(n)}


def outcome_regressions(shared: pd.DataFrame) -> dict[str, Any]:
    """Regress home_win and home_margin on (elo_2016frame - elo_2019frame) for
    2021-24 rows. Zero beta => residue is noise on these rows; nonzero =>
    2016-18 history carries real team-quality signal."""
    d = shared.dropna(subset=["elo_diff_2016", "elo_diff_2019"])
    delta = (d["elo_diff_2016"] - d["elo_diff_2019"]).to_numpy(float)
    y_win = d["home_win"].to_numpy(float)
    y_margin = (d["home_score"] - d["away_score"]).to_numpy(float)
    return {
        "home_win": ols(delta, y_win),
        "home_margin": ols(delta, y_margin),
        "note": ("beta = expected change in the outcome per 1 ELO point of "
                 "frame-start delta (W2016 minus W2019); rows = pooled "
                 "2021-24 shared games"),
    }


def feature_sanity(shared: pd.DataFrame,
                   ewm_cols: list[str], rolling_cols: list[str],
                   static_cols: list[str]) -> dict[str, Any]:
    """Max |Δ| on identical rows per feature class. EWM must be <= 1e-3
    (halflife-2 decay => numerically zero over 60-100 prior games — EWM is
    NOT a warm-up feature); rolling + static must be exactly 0."""
    out: dict[str, Any] = {}
    for label, cols in (("ewm", ewm_cols), ("rolling", rolling_cols),
                        ("static", static_cols)):
        present = [c for c in cols
                   if c + "_2016" in shared.columns
                   and c + "_2019" in shared.columns]
        per = {}
        for c in present:
            d = (shared[c + "_2016"] - shared[c + "_2019"]).abs()
            per[c] = round(float(d.max()), 12)
        out[label] = {"max_abs_delta_per_feature": per,
                      "max_abs_delta": round(
                          max(per.values(), default=0.0), 12)}
    out["ewm_bar"] = EWM_MAX_ABS_DELTA
    out["ewm_ok"] = out["ewm"]["max_abs_delta"] <= EWM_MAX_ABS_DELTA
    out["rolling_static_exact_zero"] = (
        out["rolling"]["max_abs_delta"] == 0.0
        and out["static"]["max_abs_delta"] == 0.0)
    return out


def decision_rule(material: bool, signal: bool) -> dict[str, str]:
    """The recorded (NOT gating) frame-start decision rule."""
    if not material:
        verdict = "negligible"
        reason = ("frame-start delta is within noise → no ELO reset warranted; "
                  "W2016 reads as a VOLUME test (the added pre-window training "
                  "rows, not the priors, would be the intervention)")
    elif signal:
        verdict = "material_plus_signal"
        reason = ("the 2016 frame already captures the better strength "
                  "estimate (2016-18 history carries outcome signal on the "
                  "scored window) → a season-reset ELO would LOSE it")
    else:
        verdict = "material_plus_noise"
        reason = ("frame-start arbitrariness is real (material delta, no "
                  "outcome signal) → spec a season-reset ELO probe as a "
                  "follow-on")
    return {"verdict": verdict, "reason": reason}


def away_bias_by_season(oof_art: pd.DataFrame,
                        sealed_preds: pd.DataFrame) -> list[dict]:
    """Per-season away (and home) mean OOF residual (actual - pred), pooled
    2021-24 from the artifact + sealed 2025 from the fit-only refill."""
    rows: list[dict] = []
    for season in SCORED_SEASONS:
        sub = oof_art[oof_art["season"] == season]
        rows.append({
            "season": season, "n": int(len(sub)),
            "away_bias": round(float(sub["resid_away"].mean()), 3),
            "home_bias": round(float(sub["resid_home"].mean()), 3),
        })
    s = sealed_preds.dropna(subset=["pred_away"])
    rows.append({
        "season": SEALED_SEASON, "n": int(len(s)),
        "away_bias": round(float(s["resid_away"].mean()), 3)
        if "resid_away" in s.columns else round(
            float((s["away_score"] - s["pred_away"]).mean()), 3),
        "home_bias": round(float(s["resid_home"].mean()), 3)
        if "resid_home" in s.columns else round(
            float((s["home_score"] - s["pred_home"]).mean()), 3),
    })
    return rows


def frame_sha256_of(df: pd.DataFrame) -> str:
    """Content hash (row-sorted) — the tier-1 convention."""
    import hashlib
    h = hashlib.sha256()
    sorted_df = df.sort_values("game_id").reset_index(drop=True)
    h.update(sorted_df.to_csv(index=False).encode("utf-8"))
    return h.hexdigest()[:12]