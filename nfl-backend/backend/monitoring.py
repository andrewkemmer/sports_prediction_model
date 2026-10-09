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
    from backend import features as feat_mod
    from backend import manifest
except ImportError:
    import config
    import features as feat_mod
    import manifest

logger = logging.getLogger(__name__)

PSI_WARN = 0.10
PSI_ALERT = 0.25
# A window too small to judge drift at all (MLB parity, explainability.py).
# Every served feature currently drifts on a 60-game recent window against a
# full-history baseline, and 10 quantile bins cannot be estimated from 39-60
# rows: measured against a NULL (current drawn from the SAME distribution as
# baseline), that window raises a false ALERT on 52% of features and a false
# WARN on 95%. Those rows are reported INSUFFICIENT rather than as a verdict,
# so a real distribution change still pages and a sampling artefact never does.
PSI_MIN_CURRENT = 30
PSI_MIN_BASELINE = 100
# Games inside one window share teams, so the naive standard error of a mean
# understates the true spread. MLB inflates by this clustering factor for the
# same reason; the NFL recent window is the same shape of data.
PSI_LOCATION_CLUSTER_FACTOR = 1.5
# Quantile bins per PSI comparison. Named once so the null is measured on the
# SAME binning the observation is scored on -- a null measured at a different
# k would not be the null the reported number is drawn from.
PSI_BINS = 10
# How many pseudo-windows the sampling null is measured from. 200 puts the
# standard error of the null's mean near 7% of its own size -- comfortably
# inside the 0.10/0.25 status grid -- for 48 x 200 bin counts per run, about
# 0.2s in total.
PSI_NULL_DRAWS = 200
# Seeded from a constant, not the clock: the artifact is a production record,
# so the same baseline must regenerate the same verdicts on every run.
PSI_NULL_SEED = 20260927

# Rolling-Brier series shape (parity with MLB's explainability module, which
# owns the shared monitor page's caption). The caption states a TRAILING-WINDOW
# mean over ``window_days`` with a ``min_games_per_day`` floor, so both numbers
# must be real module constants the series and the artifact meta both read --
# not literals retyped at the call site.
ROLLING_BRIER_WINDOW_DAYS = 30
# MLB uses 5. An NFL OOF timeline routinely carries ONE decided game on a date
# (Tuesday specials, early-season slates), so a 5-game floor would exclude most
# of the series and leave the page nearly empty. 1 keeps every date while the
# windowing -- not the floor -- is what makes the caption true.
ROLLING_BRIER_MIN_GAMES_PER_DAY = 1


def psi_noise_floor(n_baseline: int, n_current: int,
                    n_bins: int = PSI_BINS) -> float:
    """Analytic PSI that two SAME-distribution samples produce at these sizes.

    Retained as the documented FALLBACK for a baseline too small to measure a
    null from, and as the record of what the closed form claims. It is a lower
    bound, not an estimate. Scored against two independent samples of the same
    law with this module's own quantile binning, the real null reads 2.0x the
    formula at n=2000, 2.4x at n=60, 5.0x at n=30 and 7.5x at n=20 -- so the
    closer a window gets to unjudgeable, the more it understates. The reason
    is the same at every size: a bin proportion estimated from n rows carries
    a relative error of about sqrt((1-p)/n), which at n=60 is 39%, far outside
    the range where the log term is linear, and the neglected remainder is
    what dominates. MLB's ``explainability.psi_noise_floor`` carries the same
    closed form; it is left alone there and not copied forward as truth here.
    ``psi_sampling_null`` below measures the quantity instead of guessing it.
    """
    if n_baseline <= 0 or n_current <= 0:
        return 0.0
    return (n_bins - 1) / 2.0 * (1.0 / n_baseline + 1.0 / n_current)


def _bin_edges(baseline: np.ndarray, n_bins: int) -> np.ndarray | None:
    """The quantile bin edges ``_psi`` bins on, or None when they degenerate
    (fewer than two distinct edges means the feature is constant here and
    every sample scores identity)."""
    try:
        qs = np.unique(np.quantile(baseline, np.linspace(0, 1, n_bins + 1)))
    except Exception:
        return None
    return qs if len(qs) >= 2 else None


def _binned_baseline(baseline: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Share of the baseline falling in each bin defined by ``edges``."""
    n_bins = len(edges) - 1
    qb = np.clip(np.searchsorted(edges, baseline, side="right") - 1, 0, n_bins - 1)
    return np.bincount(qb, minlength=n_bins) / baseline.size


def _psi_from_binned(current: np.ndarray, pb: np.ndarray,
                     edges: np.ndarray) -> float:
    """PSI of ``current`` against fixed baseline bin shares ``pb``.

    Split out of ``_psi`` so the sampling null can score hundreds of
    pseudo-windows against ONE precomputed binning instead of re-deriving the
    quantiles for every draw. NumPy rather than pandas on this path: the null
    calls it tens of thousands of times per run and a Series per draw cost
    more than the whole rest of the drift phase.
    """
    c = np.asarray(current, dtype=float)
    c = c[np.isfinite(c)]
    n_bins = len(pb)
    if c.size < 10:
        return np.nan
    qc = np.clip(np.searchsorted(edges, c, side="right") - 1, 0, n_bins - 1)
    pc = np.bincount(qc, minlength=n_bins) / c.size
    pb, pc = np.clip(pb, 1e-6, None), np.clip(pc, 1e-6, None)
    return float(np.sum((pc - pb) * np.log(pc / pb)))


def psi_sampling_null(baseline: np.ndarray, n_current: int,
                      n_bins: int = PSI_BINS,
                      draws: int = PSI_NULL_DRAWS,
                      seed: int = PSI_NULL_SEED) -> dict:
    """MEASURED PSI that sampling noise alone produces at these two sizes.

    Draws ``draws`` pseudo-windows of ``n_current`` rows from the baseline's
    own distribution and scores each against the baseline with the same
    binning ``_psi`` uses. The recent window is never consulted, so the result
    is the null distribution every real change has to clear: a feature whose
    observed PSI sits inside it has not moved, whatever the raw number says.

    Reports the mean and the MEDIAN and deliberately not an upper quantile. At
    n_current=60 a ten-bin PSI leaves ~6 rows per bin, so roughly one draw in
    forty lands a bin at zero and contributes a log term near 11 -- the null
    has skew above 5, and its 95th percentile sits exactly on that mass. Two
    baselines of the same law and the same size therefore published a p95 of
    0.33 and of 1.22 from identical code, which is a property of which side of
    the cliff the baseline falls on, not of the feature. The mean and median
    both hold a coefficient of variation near 0.03-0.06 across baselines, so
    those are what the artifact carries.

    Seeded from a module constant, not the clock, because the emitted artifact
    is a production record: the same baseline must regenerate the same null and
    therefore the same verdicts. ``measured=False`` marks the fallback path, so
    a reader can tell a measured floor from the analytic one.
    """
    b = np.asarray(pd.Series(baseline).dropna(), dtype=float)
    b = b[np.isfinite(b)]
    n_current = int(n_current)
    fallback = {"mean": psi_noise_floor(b.size, n_current, n_bins),
                "median": float("nan"), "draws": 0, "measured": False}
    if b.size < 10 or n_current < 10 or n_current > b.size:
        return fallback
    edges = _bin_edges(b, n_bins)
    if edges is None:
        return fallback
    pb = _binned_baseline(b, edges)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(draws)):
        value = _psi_from_binned(b[rng.integers(0, b.size, n_current)],
                                 pb, edges)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return fallback
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "draws": int(arr.size), "measured": True}


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


def _psi(current: np.ndarray, baseline: np.ndarray,
         n_bins: int = PSI_BINS) -> float:
    """Population stability index between the current-window and baseline
    distributions of a feature (quantile-binned on the baseline)."""
    b = np.asarray(pd.Series(baseline).dropna(), dtype=float)
    b = b[np.isfinite(b)]
    if b.size < 10 or np.isfinite(np.asarray(current, dtype=float)).sum() < 10:
        return np.nan
    edges = _bin_edges(b, n_bins)
    if edges is None:
        return np.nan
    return _psi_from_binned(current, _binned_baseline(b, edges), edges)


def feature_importance_weights(models: dict,
                               member_weights: dict[str, float],
                               feature_frame: pd.DataFrame | None = None) -> dict[str, float]:
    """Return blend-weighted importance for each served feature.

    Tree members expose feature_importances_ on the named tree-view
    columns; the standardized elastic-net member exposes coef_ on the
    linear-view columns. Each member is normalized before its causal ensemble
    weight is applied, then the served-feature totals are normalized again.
    This is monitoring metadata only and never changes model fitting or
    prediction.
    """
    importance = {f: 0.0 for f in config.active_moneyline_feature_cols()}
    for name, entry in (models or {}).items():
        model = entry.get("model") if isinstance(entry, dict) else entry
        if model is None:
            continue
        raw = getattr(model, "feature_importances_", None)
        if raw is not None:
            values = np.asarray(raw, dtype=float)
            # Recreate the exact named tree-view columns used by the final
            # moneyline fit so importances stay aligned with the model.
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


def run_line_feature_weights(score_regressor,
                             feature_frame: pd.DataFrame | None = None
                             ) -> dict[str, float]:
    """The RUN LINE (distribution) model's own per-feature weights: pooled,
    n-weighted LightGBM split GAIN across the two per-side Poisson fits
    (home-λ + away-λ), normalized to sum to 1.0 over the served features.

    MLB parity (mlb ``distributions.run_line_feature_weights``): the Totals &
    Run Lines drift table's MODEL WEIGHT column reports the run line model
    itself — never the binary moneyline blend's weights, which is what the
    run-engine drift CSV used to carry. The NB layer itself has no per-feature
    parameters (it shapes dispersion/MC only), so feature usage lives entirely
    in the two per-side Poisson fits; their pooled, n-weighted split GAIN is
    the run line's honest importance.

    READ-ONLY over the SHIPPED fits (the ``ScoreRegressor`` the pipeline just
    persisted): no refit, no second training — the weights describe exactly
    the distribution model that priced this run's artifact. The team-ID
    categorical columns are the trees' encoding, not served features: their
    gain is dropped and the rest renormalized (weight_pct sums to 100 across
    the served list). Diagnostic only — a failure returns {} so the caller
    passes None and the frontend omits the column rather than rendering
    moneyline weights or zeros.
    """
    if score_regressor is None:
        return {}
    try:
        n_rows = float(len(feature_frame)) if feature_frame is not None \
            and len(feature_frame) else 1.0
        gain: dict[str, float] = {}
        for side in ("home", "away"):
            model = getattr(score_regressor, f"{side}_model", None)
            if model is None or not hasattr(model, "booster_"):
                return {}
            g = np.asarray(model.booster_.feature_importance(
                importance_type="gain"), dtype=float)
            names = list(model.booster_.feature_name())
            for f, v in zip(names, g):
                if np.isfinite(v):
                    gain[f] = gain.get(f, 0.0) + float(v) * n_rows
        total = sum(gain.values())
        if not np.isfinite(total) or total <= 0:
            return {}
        weight = {k: v / total for k, v in gain.items()}
        id_share = sum(weight.pop(c, 0.0)
                       for c in config.TREE_CATEGORICAL_COLS)
        if id_share > 0 and weight:
            kept = sum(weight.values())
            if kept > 0:
                weight = {k: v / kept for k, v in weight.items()}
        return weight
    except Exception as exc:  # noqa: BLE001 — diagnostic only, never block a run
        logger.warning("run_line_feature_weights: unavailable (%s)", exc)
        return {}


def drift_windows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice the drift comparison's two windows from one canonical frame.

    The structural twin of MLB's pipeline slice (mlb-backend pipeline.py:
    ``baseline = prior.tail(max(3 * len(current), 250))``) and NHL's
    monitoring.drift_windows (2026-09-27): "current" is the trailing
    :attr:`config.DRIFT_CURRENT_GAMES` decided games and "baseline" is the
    tail of the history that immediately precedes them — ``max(3x the
    current window, :attr:`config.DRIFT_BASELINE_MIN_GAMES`) games`` — never
    the full pool. A baseline drawn from the same recent era as the current
    window makes a PSI row answer "did the recent game change?"; the
    full-history baseline mixed whole seasons in, which flagged season- and
    era-boundary effects (the 2026-09-27 pace_plays_min_away ALERT lit on a
    slow multi-year league-wide pace decline, not a recent change).

    MONITORING ONLY: nothing in the fit or serve path reads these windows —
    training is expanding walk-forward over the full pool regardless.

    Season-phase matching (2026-09-29): the baseline pool is further
    restricted to the current window's calendar-month phase across prior
    seasons, so an early-September window is judged against prior
    Septembers instead of the prior season's December/January tail — a
    cumulative-through-season feature (inj_ol_out_home) otherwise produces a
    real 3-sigma location shift that is season-phase, not drift. Falls back
    to the plain preceding-era tail when same-phase prior seasons are too
    thin to fill the minimum baseline.

    Returns ``(baseline, current)`` in the same order callers pass them to
    :func:`feature_drift`. Falls back to the whole-pool tail when the frame
    is too small to slice both windows disjointly.
    """
    n_cur = min(int(config.DRIFT_CURRENT_GAMES), max(len(df) // 2, 1))
    current = df.tail(n_cur)
    prior = df.head(len(df) - n_cur)
    n_base = min(max(3 * n_cur, int(config.DRIFT_BASELINE_MIN_GAMES)),
                 len(prior))
    # Season-phase matching (2026-09-29 v9.6): a trailing-tail baseline in an
    # early-September window is the PRIOR season's December/January stretch.
    # Cumulative-through-season features (injury designations accumulate,
    # inj_ol_out_home 2.11 in Jan vs 1.57 in Sep) then produce a real 3-sigma
    # location shift that is a season-phase artifact, not drift — exactly the
    # season-boundary class this window geometry was built to avoid but only
    # fixed for era, not phase. The fix: draw the baseline from the same
    # CALENDAR-MONTH phase as the current window, STRICTLY PRIOR SEASONS only
    # (the current season's own earlier games are excluded — training camps
    # and injuries ramp within a season too, so Sep-vs-Sep inside one season
    # has the same artifact). Falls back to the plain preceding-era tail when
    # same-phase prior seasons are too thin to fill the minimum baseline.
    _gd_cur = pd.to_datetime(current["gameday"], errors="coerce")
    _gd_pri = pd.to_datetime(prior["gameday"], errors="coerce")
    if len(prior) and _gd_cur.notna().any():
        _m = int(_gd_cur.dt.month.mode().iloc[0])
        _cur_years = set(_gd_cur.dt.year.dropna().astype(int))
        _pool = prior[_gd_pri.dt.month.eq(_m)
                      & ~_gd_pri.dt.year.isin(_cur_years)]
        if len(_pool) >= n_base:
            baseline = _pool.tail(n_base)
        else:
            baseline = prior.tail(n_base)
    else:
        baseline = prior.tail(n_base)
    return baseline, current


def _phase_matched_baseline(bgames: pd.DataFrame, cgames: pd.DataFrame,
                            col: str, months_back: tuple[int, ...]) -> list:
    """Prior-season same-calendar-phase values for one feature.

    MLB ``explainability._phase_matched_baseline`` port (2026-09-30
    season-seam guard; NHL b698f90b). Current window's span = [min
    gameday, max gameday] of ``cgames`` padded ±7 calendar days (a one-week
    phase tolerance: the exact-span window can dip under the judge floor
    and flip borderline seasonal features back to ALERT). For each k in
    ``months_back`` (negative ints), take bgames rows whose gameday falls
    in the padded same-phase window shifted k years — the season seam
    moves WITH the calendar instead of across it. Only rows already
    present in the frame qualify (no new data is fetched). Missing or
    unparsable gameday columns yield [] — callers keep the plain baseline
    verdict.
    """
    if "gameday" not in bgames.columns or "gameday" not in cgames.columns:
        return []
    cd = pd.to_datetime(cgames["gameday"], errors="coerce")
    if cd.notna().sum() == 0:
        return []
    lo, hi = cd.min(), cd.max()
    bd = pd.to_datetime(bgames["gameday"], errors="coerce")
    out: list = []
    for k in months_back:
        lo2 = lo + pd.DateOffset(years=k) - pd.Timedelta(days=7)
        hi2 = hi + pd.DateOffset(years=k) + pd.Timedelta(days=7)
        win = bgames.loc[(bd >= lo2) & (bd <= hi2), col].dropna()
        out.extend(win.tolist())
    return out


def feature_drift(full_df: pd.DataFrame, recent_df: pd.DataFrame,
                  weights: dict[str, float] | None = None,
                  phase_frame: pd.DataFrame | None = None,
                  view: str = "moneyline") -> list[dict]:
    """PSI per served feature: current window vs its preceding-era baseline.

    The frames come from :func:`drift_windows` (MLB's trailing-tail geometry:
    the baseline is the era immediately before the current window, not the
    full pool — a full-history baseline mixes whole seasons in and flags
    season-boundary effects every early-season run).

    Rows carry the MLB-shaped fields the shared monitor page renders:
    ``status`` (OK/WARN/ALERT from the same PSI thresholds MLB uses),
    ``weight_pct`` (the ensemble's blend-weighted feature importance, when
    member importances are available), and ``n_baseline`` / ``n_current``
    sample sizes behind each comparison. NFL's own PSI values and windows —
    nothing copied from MLB.

    A 60-game window cannot be scored on raw PSI: the null distribution the
    report divides by is measured per feature by ``psi_sampling_null`` and
    subtracted, so ``psi``/``psi_adjusted`` is the excess over what identical
    populations produce, and ``psi_raw`` is the unadjusted figure kept for
    transparency. Status is then gated on a location shift before that excess
    is even read. ``psi_null_median`` is carried alongside so a reader can see
    the scale of the noise being removed rather than trusting a bare verdict.

    ``phase_frame`` (season-seam guard, MLB 2026-09-30 / NHL b698f90b
    parity): the frame the prior-season phase windows are pulled from. MUST
    be the full decided pool — the trailing baseline only covers the recent
    era and holds no prior-season rows, so without it the same-calendar-
    phase re-check can never fire. A location shift that survives the
    baseline is re-measured against the same calendar phase of prior years;
    a clean re-check relabels the row OK-SEASONAL (an every-year season-
    start seam — rest_days carrying the off-season gap, pace resetting with
    the schedule — is not a regime break). Callers that omit it get the
    old baseline-only behavior.

    ``view`` labels which monitoring surface produced these rows (MLB's
    2026-10-05 log review): the moneyline monitor and the run-engine CSV
    writer run this same function on the same windows, and the summary
    line emitted here — ``Feature drift [view]: ...`` — is how a run-log
    reader tells them apart and can see that drift happened at all (the
    2026-10-08 NHL log review read a fully clean log while the drift CSV
    beside it carried six alerts).
    """
    wmap = weights or {}
    has_weight_map = weights is not None
    rows = []
    for f in config.active_moneyline_feature_cols():
        if f not in full_df.columns:
            continue
        psi = _psi(recent_df[f].to_numpy(float), full_df[f].to_numpy(float))
        # A feature that is entirely NaN in this window has no mean, and
        # np.nanmean of an empty slice emits "RuntimeWarning: Mean of empty
        # slice" once per such feature. That is not an anomaly worth a
        # traceback in the run log -- the coverage table already reports
        # STARVED for it -- so the empty case is handled explicitly.
        _cur_vals = recent_df[f].dropna().to_numpy(float)
        _base_vals = full_df[f].dropna().to_numpy(float)
        mean_cur = float(_cur_vals.mean()) if _cur_vals.size else np.nan
        mean_base = float(_base_vals.mean()) if _base_vals.size else np.nan
        # Same rule for the standard error: a window holding a single
        # observation has no sample variance, and np.nanstd of one value emits
        # "Degrees of freedom <= 0 for slice". A NaN SE is the honest answer and
        # is what the shift_se guard below already expects.
        def _se(vals: np.ndarray) -> float:
            if vals.size < 2:
                return np.nan
            return float(vals.std(ddof=1) / np.sqrt(vals.size))

        se_cur = _se(_cur_vals)
        se_base = _se(_base_vals)
        shift_se = float(np.hypot(se_cur, se_base)) if np.isfinite(se_cur) \
            and np.isfinite(se_base) else np.nan

        n_base_n, n_cur_n = int(_base_vals.size), int(_cur_vals.size)
        # The null, MEASURED on this feature's own distribution at these exact
        # two sizes. The closed form this replaces understates it by 2.4x at
        # n=60, so it credited the 2026-09-27 report with less noise than the
        # window actually carries and paged on the difference. The location
        # gate below is a second, independent guard: PSI responds to any
        # distributional change, including pure binning wiggle on a quantized
        # feature (win_pct, temp_f, is_turf_home have many repeated values, so
        # a quantile edge landing inside a tie cluster moves whole games
        # between bins while nothing changed). Requiring the MEAN to move too
        # is what separates a real shift from that.
        null = psi_sampling_null(_base_vals, n_cur_n)
        noise = float(null["mean"])
        psi_adjusted = max(psi - noise, 0.0) if np.isfinite(psi) else np.nan
        if n_base_n + n_cur_n > 2:
            pooled_sd = float(np.sqrt(
                ((n_base_n - 1) * _base_vals.var(ddof=1)
                 + (n_cur_n - 1) * _cur_vals.var(ddof=1))
                / (n_base_n + n_cur_n - 2)))
        else:
            pooled_sd = 0.0
        mean_shift = (mean_cur - mean_base
                      if np.isfinite(mean_cur) and np.isfinite(mean_base)
                      else np.nan)
        if pooled_sd > 0 and np.isfinite(mean_shift):
            loc_se = float(pooled_sd * np.sqrt(1.0 / n_base_n + 1.0 / n_cur_n)
                           * PSI_LOCATION_CLUSTER_FACTOR)
            location_shift = bool(abs(mean_shift) > 2.0 * loc_se)
        else:
            loc_se = 0.0
            location_shift = bool(np.isfinite(psi_adjusted)
                                  and psi_adjusted > 0)

        # Status in precedence order: too small to judge, then the location
        # gate, then the noise-adjusted PSI. Raw PSI is reported but never
        # gates, because between two same-distribution samples of this size it
        # sits near the WARN threshold all by itself.
        structural_reason = None
        if n_base_n < PSI_MIN_BASELINE or n_cur_n < PSI_MIN_CURRENT:
            status = "INSUFFICIENT"
        elif not np.isfinite(psi):
            # A degenerate baseline (zero variance across the whole window)
            # yields no quantile edges and therefore no PSI at all. Two very
            # different causes, and the row must say which one it is:
            #
            # * STRUCTURAL -- the current window is constant at the SAME
            #   value. is_home is 1.0 by construction; is_snow and
            #   travel_miles_home are 0.0 in both the season-phase-matched
            #   September baseline and the early-October current window.
            #   Nothing can have drifted between two identical constants, so
            #   the honest report is the stable fact with its reason -- not
            #   a "cannot judge" that every early-season run re-emits as a
            #   WARNING (the three permanent INSUFFICIENT rows in the
            #   2026-09-29..10-01 logs, whose 'null psi' format a reader
            #   had to audit by hand to learn the means were equal).
            #
            # * INSUFFICIENT -- the current window MOVED (a different
            #   constant, or any variance). That is the corruption class the
            #   guard exists for: a venue bug pricing every prior game at
            #   the nominal home stadium (2026-09-29) while real games show
            #   travel. PSI cannot score it, so the verdict stays "cannot
            #   judge" -- never OK, never STRUCTURAL.
            base_const = (bool(n_base_n)
                          and bool(np.all(_base_vals == _base_vals[0])))
            cur_const = (bool(n_cur_n)
                         and bool(np.all(_cur_vals == _cur_vals[0])))
            if (base_const and cur_const
                    and float(_base_vals[0]) == float(_cur_vals[0])):
                status = "STRUCTURAL"
                structural_reason = (
                    f"constant {_base_vals[0]:g} in both windows "
                    "(cannot drift)")
            else:
                status = "INSUFFICIENT"
        elif not location_shift:
            status = "OK"
        else:
            status = feature_status(psi_adjusted)

        # Season-seam guard (MLB 2026-09-30 / NHL b698f90b): the trailing
        # baseline sits at the end of the previous season, so the first
        # window of a new season regularly flags REGULAR seasonal movement
        # (rest_days carrying the off-season gap, pace resetting with the
        # schedule) as drift. When a location shift survives the trailing
        # baseline, re-check the same mean shift against the SAME calendar
        # phase of prior years. A clean re-check means the shift is a
        # season seam, not a regime break — verdict only: PSI and every
        # evidence field in the row below stay exactly as computed.
        if (status in ("WARN", "ALERT")
                and n_base_n >= PSI_MIN_BASELINE
                and n_cur_n >= PSI_MIN_CURRENT):
            phase_vals = _phase_matched_baseline(
                phase_frame if phase_frame is not None else full_df,
                recent_df, f, tuple(config.DRIFT_PHASE_EXTENSION_MONTHS))
            if len(phase_vals) >= 100:
                pv = np.asarray(phase_vals, dtype=float)
                pooled2 = float(np.sqrt(
                    ((len(pv) - 1) * pv.var(ddof=1)
                     + (n_cur_n - 1) * _cur_vals.var(ddof=1))
                    / (len(pv) + n_cur_n - 2))) if len(pv) + n_cur_n > 2 else 0.0
                shift2 = float(_cur_vals.mean() - pv.mean())
                se2 = (pooled2 * np.sqrt(1.0 / len(pv) + 1.0 / n_cur_n)
                       * PSI_LOCATION_CLUSTER_FACTOR
                       if pooled2 > 0 else 0.0)
                if se2 > 0 and abs(shift2) <= 2.0 * se2:
                    status = "OK-SEASONAL"

        rows.append({
            "feature": f,
            "current_mean": mean_cur,
            "baseline_mean": mean_base,
            # `psi` is the value the STATUS was assigned from, because that is
            # the column the shared monitor page renders beside the status
            # pill. Shipping the raw figure there printed 1.363 next to OK
            # while a 0.386 sat next to ALERT, so the reader could not tell
            # which number decided anything. `psi_raw` keeps the unadjusted
            # figure and `psi_adjusted` keeps the name the NFL diagnostics page
            # prefers, so both consumers read the judged value.
            "psi": psi_adjusted,
            "psi_raw": psi,
            "psi_adjusted": psi_adjusted,
            "noise_floor": noise,
            "psi_null_median": null["median"],
            "psi_null_draws": null["draws"],
            "mean_shift": mean_shift,
            "shift_se": shift_se,
            "location_shift": location_shift,
            "status": status,
            "structural_reason": structural_reason,
            "weight_pct": (round(100.0 * float(wmap.get(f, 0.0)), 2)
                           if has_weight_map else None),
            "n_baseline": int(full_df[f].notna().sum()),
            "n_current": int(recent_df[f].notna().sum()) if len(recent_df) else 0,
        })

    if rows:
        # MLB's view-labeled summary line (explainability
        # compute_feature_drift; NHL b698f90b): without an emitter the run
        # log never says drift happened — the 2026-10-08 review read an
        # all-PASS log while the drift CSV beside it carried six alerts.
        # The label is how a log reader tells the moneyline monitor apart
        # from the run-engine CSV writer, which runs this same function on
        # the same windows.
        logger.info(
            "Feature drift [%s]: %d features, %d warnings, %d alerts, %d "
            "seasonal (statuses on noise-adjusted PSI; "
            "mean noise floor %.3f)",
            view, len(rows),
            sum(r["status"] == "WARN" for r in rows),
            sum(r["status"] == "ALERT" for r in rows),
            sum(r["status"] == "OK-SEASONAL" for r in rows),
            float(np.mean([r["noise_floor"] for r in rows])),
        )
    return rows


def write_run_engine_feature_artifacts(out_dir, date_c: str,
                                       full_df: pd.DataFrame,
                                       recent_df: pd.DataFrame,
                                       weights: dict[str, float] | None = None,
                                       phase_frame: pd.DataFrame | None = None
                                       ) -> tuple[str, str]:
    """Emit MLB-shaped run-engine drift/coverage CSVs for the NFL page.

    The drift CSV and the coverage CSV describe the SAME two frames — one
    call receives them once (the drift step's baseline/current slice), so
    the tables beside each other on the monitor page cannot answer different
    windows (MLB's 08-28 incident guard). The run engine intentionally
    resolves the same contract (config.MONEYLINE_FEATURE_COLS via
    features.tree_view) as binary moneyline; this is monitoring output only
    and does not create a second training feature contract.
    """
    drift = feature_drift(full_df, recent_df, weights=weights,
                          phase_frame=phase_frame, view="run-engine")
    cov = coverage(full_df, current_df=recent_df)
    drift_path = out_dir / f"run_engine_feature_drift_{date_c}.csv"
    cov_path = out_dir / f"run_engine_feature_coverage_{date_c}.csv"
    pd.DataFrame(drift).to_csv(drift_path, index=False)
    pd.DataFrame(cov).to_csv(cov_path, index=False)
    return drift_path.name, cov_path.name


def coverage(full_df: pd.DataFrame,
             current_df: pd.DataFrame | None = None) -> list[dict]:
    """Per-feature measured/non-null coverage over the drift windows.

    MLB-shaped fields: ``status`` (STARVED <25% measured / LOW_COVERAGE
    <80% / OK — the same thresholds the shared page documents) and
    ``n_default_zero``. The NFL engine does not default-fill features (NaN
    routes to imputation at fit time), so every present non-null value is a
    real measurement: pct_measured == pct_nonnull and n_default_zero is 0.

    The windows are the SAME two frames the drift table compares (the
    guaranteed shared-frames property MLB enforced after its 08-28 incident,
    when the coverage CSV answered a different window than the drift CSV
    beside it): pass ``current_df`` (the drift step's trailing window) to
    emit both drift windows — ``full_df`` is then labeled ``baseline`` and
    ``current_df`` ``current``; without it, ``full_df`` keeps the legacy
    ``decided pool`` label. The engine never default-fills, so the report
    also cannot mistake its own windows: the weather quartet's 71% is the
    all-time share of roofed/dome games (a real measurement absent by
    nature, not a broken fetcher) and stays visible for exactly that
    structural read.

    STRUCTURAL status (2026-09-29): a feature whose manifest
    ``missing_value_policy`` DECLARES the absent slice (indoor/closed
    weather games, season openers for rest, the week-1 player-rating cold
    start for the EPA lineup family) cannot "starve" at its own by-design
    rate. When its measured share is STABLE across the two windows
    (|baseline − current| <= 15 pts), the row reports STRUCTURAL with the
    reason — calm, visible, and answerable — while an UNSTABLE drop still
    escalates to LOW_COVERAGE/STARVED (the 2026 weather-truncation class:
    a fetcher can die and PSI rows keep showing plausible zeros, so the
    raw rate itself must keep its alarm). Features with no declared policy
    keep the raw thresholds at every rate.
    """
    _STRUCTURAL_MARKERS = (
        "indoor/closed",                        # weather quartet: no outdoor
                                                # observation exists by nature
        "first game of the season",             # rest family: opener is
                                                # definitionally undefined
        "no projected player at this position "
        "has a prior rating",                   # EPA family: week-1 rating
                                                # cold start
    )

    def _structural_reason(f: str) -> str | None:
        pol = manifest.FEATURE_MANIFEST.get(f, {}).get(
            "missing_value_policy", "")
        for marker in _STRUCTURAL_MARKERS:
            if marker in pol:
                return marker
        return None

    def _row(f: str, frame: pd.DataFrame, window: str) -> dict:
        if f not in frame.columns:
            return {"feature": f, "window": window, "n_games": len(frame),
                    "pct_measured": 0.0, "pct_nonnull": 0.0,
                    "n_default_zero": 0, "status": "STARVED"}
        v = pd.to_numeric(frame[f], errors="coerce")
        pct = round(100.0 * float(v.notna().mean()), 2)
        return {
            "feature": f, "window": window, "n_games": int(len(frame)),
            "pct_measured": pct,
            "pct_nonnull": pct,
            "n_default_zero": 0,
            "status": ("STARVED" if pct < 25.0
                       else "LOW_COVERAGE" if pct < 80.0 else "OK"),
        }

    def _classified_row(f: str, base_row: dict, pct_base: float,
                        pct_cur: float | None) -> dict:
        """Apply the STRUCTURAL override to one raw row when the documented
        policy declares the absent slice AND the rate is window-stable."""
        if pct_cur is None:
            return base_row
        reason = _structural_reason(f)
        if reason is None:
            return base_row
        if abs(pct_base - pct_cur) > 15.0:
            return base_row
        out = dict(base_row)
        out["status"] = "STRUCTURAL"
        out["structural_reason"] = reason
        return out

    rows = []
    for f in config.active_moneyline_feature_cols():
        if current_df is not None and len(current_df):
            base = _row(f, full_df, "baseline")
            cur = _row(f, current_df, "current")
            rows.append(_classified_row(f, base, base["pct_measured"],
                                        cur["pct_measured"]))
            rows.append(_classified_row(f, cur, base["pct_measured"],
                                        cur["pct_measured"]))
        else:
            rows.append(_row(f, full_df, "decided pool"))
    return rows


def ensemble_table(oof: pd.DataFrame, weights: dict[str, float],
                   cal_p: np.ndarray | None = None) -> list[dict]:
    """Per-member OOF diagnostics + earned adaptive weights.

    Scored on the GRADING population — the same rows as the causal
    rolling blend — so blend-vs-strongest-member reads apples-to-apples.
    Weights shown are final future-serving weights, not per-row weights
    (2026-10-08 causal-evaluation parity, NHL 42027979). Frames without a
    grades_pooled column fall back to the full frame.
    """
    grade = (oof["grades_pooled"].astype(bool).to_numpy()
             if "grades_pooled" in oof.columns
             else np.ones(len(oof), dtype=bool))
    y = oof["home_win"].to_numpy(float)[grade]
    rows = []
    for name in config.ENSEMBLE_MEMBERS:
        col = f"p_{name}"
        if col not in oof.columns:
            continue
        from evaluation import binary_metrics  # local import avoids cycles
        m = binary_metrics(oof[col].to_numpy(float)[grade], y)
        rows.append({
            "name": name, "weight": round(float(weights.get(name, 0.0)), 4),
            "auc": m.get("auc"), "brier": m.get("brier"),
            "logloss": m.get("logloss"), "n_eval": m.get("n"),
        })
    return rows


def rolling_brier(oof: pd.DataFrame, p_col: str = "p_ensemble_calibrated",
                  window_days: int = ROLLING_BRIER_WINDOW_DAYS,
                  min_games_per_day: int = ROLLING_BRIER_MIN_GAMES_PER_DAY
                  ) -> dict:
    """Rolling trailing-window Brier series from walk-forward OOF history.

    MLB-shaped (parity with ``explainability.compute_rolling_brier``): a dict
    carrying the ``series`` plus the scalars a caller would otherwise have to
    dig out of the list itself -- ``history_mean_brier``, ``n_games_total``,
    ``n_points``, ``excluded_sparse_days``, ``calibrator_is_identity``,
    ``map_scope_note``. Returning the record instead of a bare list is the
    point: the shared monitor page's caption promises a *trailing-window*
    mean ("mean Brier over the trailing 30 days"), and the previous
    per-day-only list could not supply one, so the pipeline reached into the
    rows to invent a headline and formatted a list into a ``%.4f`` slot.

    Each point is the mean per-game Brier over ALL games in the trailing
    ``window_days`` calendar days ending at that date -- a game-count-free
    calendar window, so off-days between game dates contribute nothing rather
    than breaking or NaN-ing the series. Days with fewer than
    ``min_games_per_day`` decided games are excluded and COUNTED in
    ``excluded_sparse_days``, never silently averaged in.

    NFL keeps ``min_games_per_day = 1`` where MLB uses 5: an NFL OOF timeline
    routinely carries a single decided game on a date (Tuesday specials,
    early-season slates), and a 5-game floor would exclude most of the
    series. The windowing, not the floor, is what makes the caption true.

    This function logs its own summary -- all scalars, no series -- so no
    caller ever has to format one.
    """
    result: dict = {
        "window_days": int(window_days),
        "min_games_per_day": int(min_games_per_day),
        "source_column": p_col,
        "calibrator_is_identity": False,
        "map_scope_note": ("Points use the deployed Platt map (fit on all "
                           "OOF games) and are not directly comparable to "
                           "prequential-calibrated metrics."),
        "n_points": 0,
        "n_games_total": 0,
        "excluded_sparse_days": 0,
        "history_mean_brier": None,
        "series": [],
    }
    if p_col not in oof.columns:
        logger.warning("Rolling Brier: OOF store has no %s column — series "
                       "empty (dashboard shows the empty state)", p_col)
        return result
    df = oof.dropna(subset=[p_col]).copy()
    # Naive midnight dates: the window below is a CALENDAR window, so any
    # tz-awareness or clock time on the store must not enter the comparison.
    _gd = pd.to_datetime(df["gameday"], errors="coerce")
    if getattr(_gd.dt, "tz", None) is not None:
        _gd = _gd.dt.tz_convert(None)
    df["gameday"] = _gd.dt.normalize()
    df = df.dropna(subset=["gameday"]).sort_values("gameday")
    df["brier"] = (df[p_col] - df["home_win"]) ** 2
    if df.empty:
        logger.warning("Rolling Brier: no decided games with finite %s — "
                       "series empty", p_col)
        return result

    daily = df.groupby(df["gameday"].dt.date)["brier"].agg(["mean", "size"])
    qualifying = daily[daily["size"] >= min_games_per_day]
    result["excluded_sparse_days"] = int((daily["size"] < min_games_per_day).sum())
    # Exclusion is consistent everywhere: a sparse day's games never reach a
    # series point's trailing-window mean either.
    df_q = df[df["gameday"].dt.date.isin(qualifying.index)]
    result["n_games_total"] = int(len(df))
    # Game-weighted over EVERY OOF game, so it is comparable to the constant
    # baseline the monitor page draws as a dashed rule.
    result["history_mean_brier"] = round(float(df["brier"].mean()), 6)

    span = pd.Timedelta(days=window_days - 1)
    series: list[dict] = []
    for day in qualifying.sort_index().index:
        day_ts = pd.Timestamp(day)
        window_games = df_q[(df_q["gameday"] >= day_ts - span)
                            & (df_q["gameday"] <= day_ts)]
        if window_games.empty:  # defensive; qualifying is a subset of df
            continue
        series.append({
            "date": str(day),
            "brier": round(float(window_games["brier"].mean()), 6),
            "games": int(len(window_games)),
        })
    result["n_points"] = len(series)
    result["series"] = series

    if series:
        # ASCII only: this line goes through logging, and a non-UTF-8 stream
        # (a Windows console codepage) turns a typographic arrow into a
        # UnicodeEncodeError and the "--- Logging error ---" traceback this
        # whole line exists to prevent.
        logger.info(
            "Rolling Brier: %d points (%s -> %s), %d games, %d sparse days "
            "excluded (<%d games/day), mean %.4f",
            len(series), series[0]["date"], series[-1]["date"],
            result["n_games_total"], result["excluded_sparse_days"],
            min_games_per_day, result["history_mean_brier"],
        )
    else:
        logger.warning(
            "Rolling Brier: %d decided-game days but none reached the "
            "%d-game minimum — series empty",
            len(daily), min_games_per_day,
        )
    return result


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
    # Return order is (y, p) — outcomes first, model probabilities second —
    # the contract every _run_engine_market_metrics call site unpacks. The
    # 2026-09-24 artifact bug: this returned (p, y), so the metrics computed
    # logloss/ECE with labels and predictions swapped (Brier survived — it
    # is symmetric — while logloss exploded to ~5-7 and ECE sat near 0.5).
    return np.asarray(ys, dtype=float), np.asarray(ps, dtype=float)


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


# ---------------------------------------------------------------------------
# Model Monitor report contract (MLB parity)
# ---------------------------------------------------------------------------
# The shared Model Monitor page renders the SAME tables for every sport, so
# the monitor artifact's report blocks must carry MLB's row structure and
# field SEMANTICS verbatim (mlb-backend/backend/explainability.py builds
# these rows; that file is the reference):
#
#   * feature_drift -- MLB's 13-key row. ``psi`` is the RAW PSI and
#     ``psi_adjusted`` is the judged excess (max(psi - noise_floor, 0)); the
#     NFL builder's own fields use ``psi`` for the judged value and park the
#     raw figure in ``psi_raw``, which made the report's PSI column mean
#     something different from MLB's beside the same caption. The
#     null-draw diagnostics (psi_raw / psi_null_median / psi_null_draws)
#     stay OUT of the report block (the run-engine drift CSV keeps them) --
#     MLB ships no such keys and the page renders no such columns.
#   * feature_coverage -- MLB's 9-key row, including ``n_nonnull`` /
#     ``n_measured`` counts (the engine never default-fills, so the two are
#     equal and n_default_zero is 0).
#   * version_history -- MLB's model_version_history.json snapshot row
#     (version vYYYY.MM.DD, ISO date, roster weights at 4 decimals, the
#     pooled metric keys, the deployed map's {a, b, n, method, floor}),
#     accumulated across runs into MLB's rolling window exactly like
#     training.update_model_version_history (merge by version, last
#     VERSION_HISTORY_CAP rows, newest run's row wins).
#
# ``structural_reason`` rides along ONLY when truthy -- it is the shared
# page's documented STRUCTURAL overlay (a stable constant / declared
# missing-value policy with its reason), which MLB's emitters simply never
# have cause to fill. Nothing here changes any model output: this is the
# monitoring report's shape only.

VERSION_HISTORY_CAP = 20  # MLB training.VERSION_HISTORY_CAP parity

_DRIFT_REPORT_KEYS = (
    "feature", "current_mean", "baseline_mean", "psi", "psi_adjusted",
    "noise_floor", "mean_shift", "shift_se", "location_shift", "status",
    "weight_pct", "n_baseline", "n_current",
)


def _mlb_drift_report_row(r: dict) -> dict:
    """One Feature Drift report row in MLB's exact structure."""
    out = {k: r.get(k) for k in _DRIFT_REPORT_KEYS}
    # psi <- the RAW PSI (MLB semantics); the NFL builder ships the judged
    # value under "psi" and the raw figure under "psi_raw".
    out["psi"] = r.get("psi_raw", r.get("psi"))
    if r.get("structural_reason"):
        out["structural_reason"] = r.get("structural_reason")
    return out


def _mlb_coverage_report_row(r: dict) -> dict:
    """One Feature Coverage report row in MLB's exact structure.

    ``n_nonnull`` / ``n_measured`` are exact when the builder shipped them
    and recovered from the published percentages otherwise (the percentages
    are round(100 * n / n_games, 2), so the reverse map is exact at these
    window sizes). The NFL engine never default-fills -- a non-null value is
    always a measurement -- so the two counts are equal by construction.
    """
    n_games = int(r.get("n_games") or 0)

    def _count(pct_key: str, n_key: str) -> int:
        if isinstance(r.get(n_key), (int, float)):
            return int(r[n_key])
        try:
            return int(round(float(r.get(pct_key)) * n_games / 100.0))
        except (TypeError, ValueError):
            return 0

    out = {
        "feature": r.get("feature"),
        "window": r.get("window"),
        "n_games": n_games,
        "n_nonnull": _count("pct_nonnull", "n_nonnull"),
        "pct_nonnull": r.get("pct_nonnull"),
        "n_measured": _count("pct_measured", "n_measured"),
        "pct_measured": r.get("pct_measured"),
        "n_default_zero": r.get("n_default_zero", 0),
        "status": r.get("status", "OK"),
    }
    if r.get("structural_reason"):
        out["structural_reason"] = r.get("structural_reason")
    return out


def _version_history_row(iso_date: str, ensemble: list[dict],
                         m: dict, cal: dict | None) -> dict:
    """One Model Version History snapshot row in MLB's exact structure
    (``training.update_model_version_history``'s row contract): the version
    stamp, the ISO date, the roster weights at 4 decimals, the pooled metric
    keys MLB ships, and the deployed map's {a, b, n, method, floor}. Values
    missing from this run stay ABSENT -- never fabricated (MLB writes no
    partial snapshots; the shared page renders an absent cell as '—')."""
    row: dict = {
        "version": "v" + iso_date.replace("-", "."),
        "date": iso_date,
        "weights": {str(r.get("name")): round(float(r.get("weight") or 0.0), 4)
                    for r in ensemble
                    if isinstance(r, dict) and r.get("name") is not None},
    }
    for k in ("auc", "brier", "logloss", "ece",
              "brier_calibrated", "logloss_calibrated", "ece_calibrated"):
        if isinstance(m.get(k), (int, float)):
            row[k] = m[k]
    if (isinstance(cal, dict) and cal.get("a") is not None
            and cal.get("b") is not None):
        row["calibration"] = {
            "a": cal["a"], "b": cal["b"],
            **({"n": int(cal["n"])}
               if isinstance(cal.get("n"), (int, float)) else {}),
            **({"method": str(cal["method"])} if cal.get("method") else {}),
            **({"floor": float(cal["floor"])}
               if cal.get("floor") is not None else {}),
        }
    return row


def _rolling_version_history(path, current_row: dict) -> list[dict]:
    """The rolling Model Version History -- MLB's model_version_history.json
    semantics with the dated monitor family as the store (NFL ships no
    separate master file).

    Every dated ``nfl_model_monitor_*.json`` beside ``path`` carries the rows
    it knew, so folding them preserves every record that ever shipped: merge
    by ``version`` (newest artifact wins, THIS run's row always wins), never
    pull a record dated after this run (a fold over a mixed-age dir must not
    leak a later run into an earlier report), order oldest-first, keep the
    last VERSION_HISTORY_CAP rows. Records survive retention because each
    new artifact embeds the rows this fold recovered."""
    prefix = str(config.MODEL_MONITOR_JSON).split("{")[0]
    cur_date = str(current_row.get("date") or "")

    def _key(row: dict):
        return str(row.get("version") or
                   (row.get("date"), str(row.get("weights"))))

    rows: dict = {}
    try:
        files = sorted(path.parent.glob(f"{prefix}*.json"),
                       key=lambda p: p.name, reverse=True)[:60]
    except OSError:  # pragma: no cover - unreadable artifact dir
        files = []
    for f in files:
        if f.name == path.name:
            continue  # the file being written now; its row comes from the caller
        try:
            prior = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for row in (prior.get("version_history") or []):
            if not isinstance(row, dict):
                continue
            rd = str(row.get("date") or "")
            if cur_date and (not rd or rd > cur_date):
                continue  # no date / future date: not this report's history
            rows.setdefault(_key(row), row)
    rows[_key(current_row)] = current_row  # this run's row wins its version
    ordered = sorted(rows.values(),
                     key=lambda r: (str(r.get("date") or ""),
                                    str(r.get("version") or "")))
    return ordered[-VERSION_HISTORY_CAP:]


def write_monitor_json(path, run_date: str, drift: list[dict],
                       cov: list[dict], ensemble: list[dict],
                       rb: dict, baseline: float,
                       config_meta: dict, fold_info: dict,
                       metrics: dict | None = None,
                       platt: dict | None = None) -> dict:
    """MLB-shaped monitor artifact (frontend presentation contract).

    All rendering fields the shared monitor page reads are present:
    ISO retrain dates (+ same-day notes), the dense ``rolling_brier_meta``
    (window_days / min_games_per_day / excluded_sparse_days /
    calibrator_is_identity / map_scope_note), the MLB-identical report
    blocks (:func:`_mlb_drift_report_row` / :func:`_mlb_coverage_report_row`)
    and the rolling version history (:func:`_version_history_row` /
    :func:`_rolling_version_history`) with pooled metrics + the deployed
    Platt map. All values are the NFL pipeline's own outputs. The *_note
    fields are None (MLB's emitter ships no notes) so the shared page
    renders the identical fallback presentation for both sports.

    ``rb`` is the ``rolling_brier`` RECORD, and ``rolling_brier_meta`` is
    populated from it. The meta block used to be hardcoded 30/1/0 -- numbers
    for machinery that did not exist -- so the page captioned the series with
    a trailing-window rule the NFL series never applied. A record-shaped
    input makes the caption and the data the same fact.
    """
    # Tolerate a bare list from an older caller rather than crashing the
    # artifact: an empty record is the honest rendering for "no series".
    if not isinstance(rb, dict):
        rb = {"series": list(rb or []), "window_days": ROLLING_BRIER_WINDOW_DAYS,
              "min_games_per_day": ROLLING_BRIER_MIN_GAMES_PER_DAY,
              "excluded_sparse_days": 0, "calibrator_is_identity": False,
              "map_scope_note": None}
    iso_date = f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:8]}" \
        if len(str(run_date)) == 8 and str(run_date).isdigit() else str(run_date)
    next_date = iso_date  # retrains every run — next run is tonight's run
    baseline_label = "Constant home-edge" if np.isfinite(baseline) else "n/a"
    m = metrics or {}
    cal = (platt if isinstance(platt, dict) and platt.get("a") is not None
           and platt.get("b") is not None else None)
    # Is the DEPLOYED map actually a no-op? The page captions the Brier series
    # from this flag ("calibrated probabilities" vs "no calibration map
    # deployed"), so it must be measured from the map that actually ships --
    # not asserted. Under CALIBRATION_MODE=identity the fit returns None and
    # the hardcoded "not identity" this replaced stated the opposite of the
    # truth. moneyline.is_identity is the MLB-parity predicate for exactly
    # this; local import to keep the module graph acyclic.
    try:
        from moneyline import is_identity as _is_identity
        calibrator_is_identity = bool(_is_identity(cal))
    except Exception:  # pragma: no cover - metadata only
        calibrator_is_identity = cal is None
    # The monitor's feature tooltips come from the manifest, which documents
    # every served feature (definition / source / lookback / PIT rule). This
    # block used to emit "see backend/manifest.py" for all of them -- a
    # placeholder pointing at the data that was already in this repo and one
    # import away.
    try:
        from manifest import feature_tooltips
        _tool_names = [r["feature"] for r in cov
                       if isinstance(r, dict) and r.get("feature")]
        features_meta = feature_tooltips(_tool_names)
    except Exception:  # pragma: no cover - metadata only
        features_meta = {}
    # One snapshot row for THIS run in MLB's schema, accumulated with the
    # rows the dated family already carries (MLB's rolling-20 presentation).
    version_row = _version_history_row(iso_date, ensemble or [], m, cal)
    record = {
        "date": run_date,
        "version": version_row["version"],
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
        "feature_drift": [_mlb_drift_report_row(r) for r in drift
                          if isinstance(r, dict)],
        "features_metadata": features_meta,
        "feature_coverage": [_mlb_coverage_report_row(r) for r in cov
                             if isinstance(r, dict)],
        "ensemble": ensemble,
        "rolling_brier": rb.get("series", []),
        "brier_baseline": baseline,
        "brier_baseline_label": baseline_label,
        "rolling_brier_meta": {
            "window_days": rb.get("window_days", ROLLING_BRIER_WINDOW_DAYS),
            "min_games_per_day": rb.get("min_games_per_day",
                                        ROLLING_BRIER_MIN_GAMES_PER_DAY),
            "excluded_sparse_days": int(rb.get("excluded_sparse_days", 0) or 0),
            "calibrator_is_identity": calibrator_is_identity,
            "map_scope_note": (rb.get("map_scope_note")
                               or "Points use the serving Platt map (fit "
                                  "through the nested prior-evidence gate)."),
        },
        # Headline block = the serving artifact's OWN pooled OOF scores (the
        # same numbers the Calibration KPI cards show — the caller passes the
        # calibration artifact's metrics block verbatim; the CAUSAL rolling
        # blend since the 2026-10-08 evaluation remediation). The shared
        # Model Monitor page's TOTAL row reads it for the
        # blend-vs-strongest-member comparison; MLB/NBA emit this key and
        # NHL/NFL omitted it, so that row rendered em-dashes against
        # perfectly good member rows.
        "metrics": m,
        "version_history": _rolling_version_history(path, version_row),
        "fold_geometry": fold_info,
        "config": config_meta,
    }
    _dump_json(path, record)
    return record
