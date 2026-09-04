"""NFL run-engine feature drift + coverage emitters — MLB explainability
mirror for the run-engine feature view.

Mirrors ``mlb-backend/backend/explainability.py``'s run-engine PSI /
coverage machinery (``compute_run_engine_feature_drift`` /
``compute_run_engine_feature_coverage``) for the NFL run-engine's OWN
feature set: the 12-pool market-free per-side view (``SIDE_FEATURES`` =
``FEATURE_COLUMNS`` minus the ``is_home`` moneyline anchor) — the same
columns the slate/pricing chain actually prices with. Emits, per daily
run:

    data_delivery/run_engine_feature_drift_YYYYMMDD.csv
    data_delivery/run_engine_feature_coverage_YYYYMMDD.csv

column-for-column with MLB:

  drift   feature, current_mean, baseline_mean, psi, psi_adjusted,
          noise_floor, mean_shift, shift_se, location_shift, status,
          weight_pct, n_baseline, n_current
          (PSI quantile bins + add-one-half smoothing; status on
          NOISE-ADJUSTED PSI + a location gate: OK / WARN / ALERT /
          INSUFFICIENT)
  coverage feature, window, n_games, n_nonnull, pct_nonnull, n_measured,
          pct_measured, n_default_zero, status
          (OK / LOW_COVERAGE / STARVED by measured share)

Windows (MLB mirror, sport-adjusted): drift compares an ADJACENT current
vs baseline window over DECIDED games ONLY (never slate rows — pre-game
rows carry forward-looking PIT state and would distort PSI; the NFL
decided feature frame is decided-only by construction). MLB anchors the
current window at ``target_date - 7 days`` (a continuous daily season);
NFL is seasonal with a long off-season, so the anchor is the frame's
NEWEST DECIDED GAMEDAY minus a 28-day lookback (the normal NFL regular-
season cadence is a 32-day month of 7-9 games per team, so 28 days gives
a 4-week month of decided games). In-season this is ~identical to the
run-date rule (the newest decided games are the prior weekend); off-
season it still compares the latest decided period against its prior
instead of emitting an empty table. baseline = strictly-prior decided
tail(max(3 * len(current), 250)), chronological. This divergence (anchor
on newest decided gameday, 28-day window) is documented in the wiring
record so the NHL->MLB lattice stays auditable.

``weight_pct`` is always None (the run engine has no per-model blend
weight artifact — MLB passes ``model_weights=None`` for its run-engine
view too); the frontend renders a MODEL WEIGHT column only when a shared
feature-drift weight map is available.

Deterministic (no RNG): identical decided frame -> byte-identical CSVs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nfl_per_side_engine import SIDE_FEATURES  # noqa: E402

logger = logging.getLogger(__name__)

# MLB's PSI thresholds (config.PSI_WARN_THRESHOLD / PSI_ALERT_THRESHOLD) —
# the standard PSI ladder, mirrored exactly so the boards bucket identically.
PSI_WARN_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25

# The current drift window length over decided games. MLB uses 7 days (a
# daily season); NFL's cadence is weekly -> 28 days = a 4-week month of
# decided games (see module docstring).
DRIFT_CURRENT_WINDOW_DAYS = 28

# Baseline window rule (MLB mirror): strictly-prior tail of at least 3x
# the current window, with a 250-game floor.
BASELINE_MIN_GAMES = 250

DATA_DELIVERY = Path(__file__).resolve().parent.parent / "data_delivery"


def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index between two distributions — the exact
    MLB algorithm (explainability.compute_psi): quantile bin edges on the
    COMBINED sample, deduplicated (dense/discrete features don't produce
    zero-width bins), right edge nudged, add-one-half smoothing so empty
    bins stay bounded. 0 == identical distributions.
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    combined = np.concatenate([baseline, current])
    min_val, max_val = combined.min(), combined.max()

    if min_val == max_val:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.unique(np.quantile(combined, quantiles))
    if len(bin_edges) < 2:
        return 0.0
    bin_edges[-1] = max_val + 1e-10  # include the right edge

    baseline_counts = np.histogram(baseline, bins=bin_edges)[0].astype(float)
    current_counts = np.histogram(current, bins=bin_edges)[0].astype(float)

    k = len(bin_edges) - 1
    baseline_pct = (baseline_counts + 0.5) / (baseline_counts.sum() + 0.5 * k)
    current_pct = (current_counts + 0.5) / (current_counts.sum() + 0.5 * k)

    psi = float(np.sum((current_pct - baseline_pct)
                       * np.log(current_pct / baseline_pct)))
    return round(max(psi, 0.0), 6)


def psi_status(psi_value: float) -> str:
    """Map a (noise-adjusted) PSI value to OK / WARN / ALERT (MLB ladder)."""
    if psi_value >= PSI_ALERT_THRESHOLD:
        return "ALERT"
    elif psi_value >= PSI_WARN_THRESHOLD:
        return "WARN"
    return "OK"


def psi_noise_floor(n_baseline: int, n_current: int, n_bins: int = 10) -> float:
    """Expected PSI from sampling noise alone (two same-distribution
    samples): E[PSI] ≈ (k−1)/2 · (1/n_base + 1/n_cur) — the MLB formula.
    At NFL month-vs-prior window sizes (~8-260 vs ~250) this is ≈0.01-0.5;
    statuses are assigned on the NOISE-ADJUSTED PSI so identical
    distributions don't perpetually page."""
    if n_baseline <= 0 or n_current <= 0:
        return 0.0
    return (n_bins - 1) / 2.0 * (1.0 / n_baseline + 1.0 / n_current)


def run_engine_feature_cols() -> list[str]:
    """The NFL run engine's OWN feature view: the 12-pool market-free
    per-side features (SIDE_FEATURES = FEATURE_COLUMNS minus the is_home
    moneyline anchor) — the same columns the slate/pricing chain prices.
    The moneyline anchor is excluded exactly like MLB's run-engine view
    excludes its moneyline-only/derived columns."""
    logger.info("Run-engine drift view: %d/13 pool features (12-pool "
                "market-free, is_home anchor excluded)", len(SIDE_FEATURES))
    return list(SIDE_FEATURES)


def build_drift_windows(
    decided: pd.DataFrame,
    anchor_dt: Optional[pd.Timestamp] = None,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], dict]:
    """Adjacent current/baseline windows over DECIDED games, MLB-shaped.

    ``decided`` must be a decided-only feature frame (never slate/pregame
    rows) with a ``gameday`` column. Sorted chronologically; cutoff =
    anchor - DRIFT_CURRENT_WINDOW_DAYS; current = games >= cutoff;
    baseline = strictly-prior tail(max(3 * len(current),
    BASELINE_MIN_GAMES)). ``anchor_dt`` defaults to the frame's NEWEST
    decided gameday (the sport-adjusted rule — see the module docstring).
    Returns (baseline, current, meta); when no current games exist returns
    (None, None, meta) so the caller emits nothing rather than an empty
    table.
    """
    df = decided.copy()
    gd = pd.to_datetime(df["gameday"], errors="coerce")
    df = df[gd.notna()].sort_values("gameday").reset_index(drop=True)
    if df.empty:
        return None, None, {"anchor": None, "current_n": 0, "baseline_n": 0}

    gd = pd.to_datetime(df["gameday"], errors="coerce")
    anchor = anchor_dt if anchor_dt is not None else gd.max()
    cutoff = pd.Timestamp(anchor) - pd.Timedelta(days=DRIFT_CURRENT_WINDOW_DAYS)

    # current = [cutoff, anchor] — an explicit anchor in the middle of the
    # frame (e.g. a run date) must not sweep rows AFTER it (as-of semantics);
    # with the default anchor == newest decided gameday this is identical.
    current = df[(gd >= cutoff) & (gd <= anchor)]
    prior = df[gd < cutoff]
    baseline = (prior.tail(max(3 * len(current), BASELINE_MIN_GAMES))
                if not prior.empty else prior)

    if current.empty:
        return None, None, {
            "anchor": str(pd.Timestamp(anchor).date()),
            "cutoff": str(cutoff.date()),
            "current_n": 0, "baseline_n": int(len(baseline)),
            "max_decided_day": str(gd.max().date()),
        }

    # Leakage guard: strictly-prior baseline (mirror of the fold walk's
    # train < val invariant).
    if baseline.empty or baseline["gameday"].max() >= current["gameday"].min():
        return None, None, {
            "anchor": str(pd.Timestamp(anchor).date()),
            "cutoff": str(cutoff.date()),
            "current_n": int(len(current)), "baseline_n": int(len(baseline)),
            "error": "baseline not strictly prior to current",
        }

    meta = {
        "anchor": str(pd.Timestamp(anchor).date()),
        "cutoff": str(cutoff.date()),
        "current_n": int(len(current)),
        "baseline_n": int(len(baseline)),
        "current_min_day": str(current["gameday"].min().date()),
        "current_max_day": str(current["gameday"].max().date()),
        "baseline_max_day": str(baseline["gameday"].max().date()),
        "max_decided_day": str(gd.max().date()),
    }
    return baseline, current, meta


def compute_feature_drift(
    baseline_games: pd.DataFrame,
    current_games: pd.DataFrame,
    target_date_str: str,
    feature_cols: Optional[list[str]] = None,
    out_dir: Optional[Path] = None,
    out_name: Optional[str] = None,
) -> pd.DataFrame:
    """PSI per numeric feature over the adjacent windows -> drift CSV
    (full MLB column schema). ``feature_cols`` defaults to the run-engine
    view; ``out_dir`` defaults to data_delivery. Deterministic."""
    emit_dir = out_dir if out_dir is not None else DATA_DELIVERY
    emit_dir.mkdir(parents=True, exist_ok=True)
    cols = list(feature_cols) if feature_cols is not None \
        else run_engine_feature_cols()

    drift_rows = []
    for col in cols:
        if col not in baseline_games.columns or col not in current_games.columns:
            continue

        baseline_vals = pd.to_numeric(baseline_games[col],
                                      errors="coerce").dropna().values
        current_vals = pd.to_numeric(current_games[col],
                                     errors="coerce").dropna().values
        n_b, n_c = len(baseline_vals), len(current_vals)

        if n_b == 0 or n_c == 0:
            drift_rows.append({
                "feature": col,
                "current_mean": (round(float(current_vals.mean()), 4)
                                 if n_c else 0.0),
                "baseline_mean": (round(float(baseline_vals.mean()), 4)
                                  if n_b else 0.0),
                "psi": 0.0, "psi_adjusted": 0.0, "noise_floor": 0.0,
                "mean_shift": 0.0, "shift_se": 0.0, "location_shift": False,
                "status": "INSUFFICIENT", "weight_pct": None,
                "n_baseline": int(n_b), "n_current": int(n_c),
            })
            continue

        psi = compute_psi(baseline_vals, current_vals)
        noise = psi_noise_floor(n_b, n_c)
        psi_adjusted = max(psi - noise, 0.0)

        # MLB location gate: only escalate past OK when the mean ALSO moved
        # beyond sampling noise (clustering factor 1.5 — games in a window
        # share teams).
        if n_b + n_c > 2:
            pooled_sd = np.sqrt(
                ((n_b - 1) * baseline_vals.var(ddof=1)
                 + (n_c - 1) * current_vals.var(ddof=1))
                / (n_b + n_c - 2))
        else:
            pooled_sd = 0.0
        mean_shift = float(current_vals.mean() - baseline_vals.mean())
        if pooled_sd > 0:
            shift_se = float(pooled_sd * np.sqrt(1.0 / n_b + 1.0 / n_c) * 1.5)
            location_shift = abs(mean_shift) > 2.0 * shift_se
        else:
            shift_se = 0.0
            location_shift = psi_adjusted > 0

        # MLB small-window guard: PSI is statistically meaningless below
        # these sample sizes — report INSUFFICIENT, never page.
        if n_b < 100 or n_c < 30:
            status = "INSUFFICIENT"
        else:
            status = psi_status(psi_adjusted) if location_shift else "OK"

        drift_rows.append({
            "feature": col,
            "current_mean": round(float(current_vals.mean()), 4),
            "baseline_mean": round(float(baseline_vals.mean()), 4),
            "psi": psi,
            "psi_adjusted": round(psi_adjusted, 6),
            "noise_floor": round(noise, 6),
            "mean_shift": round(mean_shift, 6),
            "shift_se": round(shift_se, 6),
            "location_shift": bool(location_shift),
            "status": status,
            "weight_pct": None,
            "n_baseline": int(n_b),
            "n_current": int(n_c),
        })

    df = pd.DataFrame(drift_rows)
    out_path = emit_dir / (out_name or f"run_engine_feature_drift_"
                           f"{target_date_str}.csv")
    df.to_csv(out_path, index=False)

    n_warns = int((df["status"] == "WARN").sum())
    n_alerts = int((df["status"] == "ALERT").sum())
    logger.info(
        "Feature drift: %d features, %d warnings, %d alerts "
        "(statuses on noise-adjusted PSI; mean noise floor %.3f)",
        len(df), n_warns, n_alerts,
        float(df["noise_floor"].mean()) if len(df) else float("nan"))
    return df


def compute_feature_coverage(
    baseline_games: pd.DataFrame,
    current_games: pd.DataFrame,
    target_date_str: str,
    feature_cols: Optional[list[str]] = None,
    out_dir: Optional[Path] = None,
    out_name: Optional[str] = None,
) -> pd.DataFrame:
    """Per-feature non-null/measured coverage per drift window -> coverage
    CSV (full MLB column schema). NFL has no default-signature feature
    (MLB's weather dome default-0 special case), so n_measured ==
    n_nonnull and n_default_zero == 0 — the columns stay for schema
    parity. Deterministic."""
    emit_dir = out_dir if out_dir is not None else DATA_DELIVERY
    emit_dir.mkdir(parents=True, exist_ok=True)
    cols = list(feature_cols) if feature_cols is not None \
        else run_engine_feature_cols()

    def _window_rows(games: pd.DataFrame, window: str) -> list[dict]:
        rows: list[dict] = []
        if games is None or games.empty:
            return rows
        for col in cols:
            if col not in games.columns:
                continue
            vals = pd.to_numeric(games[col], errors="coerce")
            n_total = int(len(vals))
            n_nonnull = int(vals.notna().sum())
            n_measured = n_nonnull  # no default-signature features on NFL
            pct_nonnull = (round(100.0 * n_nonnull / n_total, 1)
                           if n_total else 0.0)
            pct_measured = (round(100.0 * n_measured / n_total, 1)
                            if n_total else 0.0)
            status = "OK" if pct_measured >= 80.0 else (
                "LOW_COVERAGE" if pct_measured >= 25.0 else "STARVED")
            rows.append({
                "feature": col,
                "window": window,
                "n_games": n_total,
                "n_nonnull": n_nonnull,
                "pct_nonnull": pct_nonnull,
                "n_measured": n_measured,
                "pct_measured": pct_measured,
                "n_default_zero": 0,
                "status": status,
            })
        return rows

    cov_rows = _window_rows(current_games, "current")
    cov_rows += _window_rows(baseline_games, "baseline")
    df = pd.DataFrame(cov_rows)
    out_path = emit_dir / (out_name or f"run_engine_feature_coverage_"
                           f"{target_date_str}.csv")
    df.to_csv(out_path, index=False)

    starved = df[df["status"] != "OK"]
    if not starved.empty:
        worst = starved.sort_values("pct_measured").head(5)
        detail = "; ".join(
            f"{r.feature}/{r.window}={r.pct_measured:.0f}% measured"
            for r in worst.itertuples())
        logger.warning("Feature coverage gaps: %s", detail)
    else:
        logger.info("Feature coverage: all %d feature-window pairs OK",
                    len(df))
    return df


def compute_run_engine_feature_drift(
    baseline_games: pd.DataFrame,
    current_games: pd.DataFrame,
    target_date_str: str,
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """PSI over the NFL run engine's OWN 12-pool view on the adjacent
    decided windows, column-for-column with MLB. Writes
    data_delivery/run_engine_feature_drift_YYYYMMDD.csv (or ``out_dir``)."""
    return compute_feature_drift(
        baseline_games, current_games, target_date_str,
        feature_cols=run_engine_feature_cols(), out_dir=out_dir,
        out_name=f"run_engine_feature_drift_{target_date_str}.csv")


def compute_run_engine_feature_coverage(
    baseline_games: pd.DataFrame,
    current_games: pd.DataFrame,
    target_date_str: str,
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Coverage over the NFL run engine's OWN 12-pool view on the same
    windows, column-for-column with MLB. Writes
    data_delivery/run_engine_feature_coverage_YYYYMMDD.csv (or
    ``out_dir``)."""
    return compute_feature_coverage(
        baseline_games, current_games, target_date_str,
        feature_cols=run_engine_feature_cols(), out_dir=out_dir,
        out_name=f"run_engine_feature_coverage_{target_date_str}.csv")