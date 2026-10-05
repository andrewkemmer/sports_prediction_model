"""NBA production monitoring and drift/coverage artifacts."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from backend import config
except ImportError:
    import config

logger = logging.getLogger(__name__)

PSI_WARN, PSI_ALERT = 0.10, 0.25
#: A window too small to judge drift must say so rather than page.  MLB's
#: thresholds: a baseline under 100 rows or a current window under 30 rows
#: makes PSI an exercise in reading tea leaves, so the status is INSUFFICIENT
#: and the PSI is informational only.
INSUFFICIENT_BASELINE, INSUFFICIENT_CURRENT = 100, 30


def psi_status(psi: float | None) -> str:
    if psi is None or not np.isfinite(psi):
        return "OK"
    return "ALERT" if psi >= PSI_ALERT else "WARN" if psi >= PSI_WARN else "OK"


def psi_noise_floor(n_baseline: int, n_current: int, n_bins: int = 10) -> float:
    """Expected PSI from sampling noise alone when both samples come from the
    SAME distribution - ``(k-1)/2 * (1/n_base + 1/n_cur)``.

    Two same-distribution samples of these window sizes already average a raw
    PSI well into WARN territory, so statuses must be assigned on the
    NOISE-ADJUSTED value or identical distributions page constantly.  MLB
    gates its drift statuses on exactly this floor; the NBA monitor's absence
    of it is why a 60-row tail against a full season paged eleven features
    whose means had not moved at all.
    """
    if n_baseline <= 0 or n_current <= 0:
        return 0.0
    return (n_bins - 1) / 2.0 * (1.0 / n_baseline + 1.0 / n_current)


def _psi(current, baseline, bins: int = 10) -> float:
    """PSI over quantile bins of the COMBINED sample, additively smoothed.

    Three defects this replaces, two of them measured in a delivered drift
    CSV and one inherited from the reference implementation itself:

    * Binning on the baseline alone makes a baseline-constant feature
      degenerate to one bin, and one bin answers "psi 0.000, OK" no matter
      how far the current window has moved.  ``is_playoffs`` went from a
      7% playoff share to a 100% playoff window and the table called it OK.
      Binning the COMBINED sample keeps at least two bins whenever either
      side varies - for continuous features.
    * A two-valued feature defeats quantile binning entirely: the combined
      quantiles of 93% zeros and 7% ones collapse to the edges [0, 1],
      which is ONE bin, which absorbs 0 and 1 alike.  The delivered table's
      "is_playoffs: 1.0 vs 0.0725, PSI 0.000, OK" survives a combined-sample
      rewrite unchanged - MLB's ``compute_psi`` has the same blind spot,
      because both of its samples land in the single collapsed bin.  Any
      feature with few distinct observed values is measured here the way
      categorical drift is measured everywhere else: per-VALUE frequency
      PSI, which scores a 7% -> 100% regime flip as exactly what it is.
    * Empty bins were floored at 1e-6, which multiplies any absent bin into
      an enormous log term.  Add-one-half smoothing (MLB's ``compute_psi``)
      keeps empty bins bounded and each term ``(c - b) * ln(c / b)`` >= 0.
    """
    c, b = pd.Series(current).dropna(), pd.Series(baseline).dropna()
    if len(c) < 10 or len(b) < 10:
        return np.nan
    combined = np.concatenate([b.to_numpy(float), c.to_numpy(float)])
    levels = np.unique(combined)
    if len(levels) <= 10:
        # Discrete-frequency PSI over the observed value set.
        bi = np.searchsorted(levels, b.to_numpy(float))
        ci = np.searchsorted(levels, c.to_numpy(float))
        k = len(levels)
        pb = (np.bincount(bi, minlength=k) + 0.5) / (len(b) + 0.5 * k)
        pc = (np.bincount(ci, minlength=k) + 0.5) / (len(c) + 0.5 * k)
        return float(max(np.sum((pc - pb) * np.log(pc / pb)), 0.0))
    edges = np.unique(np.quantile(combined, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[-1] = max(edges[-1], combined.max() + 1e-10)
    bc = np.histogram(b, bins=edges)[0].astype(float)
    cc = np.histogram(c, bins=edges)[0].astype(float)
    k = len(edges) - 1
    pb = (bc + 0.5) / (bc.sum() + 0.5 * k)
    pc = (cc + 0.5) / (cc.sum() + 0.5 * k)
    return float(max(np.sum((pc - pb) * np.log(pc / pb)), 0.0))


def drift_windows(decided: pd.DataFrame, days: int = 7,
                  min_baseline: int = 250, min_current: int = 30,
                  max_days: int = 45) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two windows drift is measured on: a recent tail and its prior.

    Mirrors MLB's slice: ``current`` is the last ``days`` days of decided
    games, ``baseline`` the prior games, at least ``min_baseline`` rows or
    three times the current window, whichever is larger - never the whole
    history.  Comparing a playoff-heavy 60-row tail against a full season is
    the comparison that paged ``elo_diff`` (PSI 0.545) and ``win_pct_diff``
    (0.564) as ALERT while their means sat within noise; a like-for-like
    prior window is the comparison that makes PSI a measurement instead of a
    survivorship artefact.

    One NBA-specific widening: MLB's 7-day window holds ~45 games whenever
    baseball is being played, but an NBA season's last 7 days can be a
    three-game Finals tail, and a 3-game current window would report
    INSUFFICIENT for every feature - honest, and useless.  The window grows
    back a day at a time (never past ``max_days``) until it holds
    ``min_current`` decided games; if the calendar cannot supply that, the
    window falls back to the last 60 decided games.  Both fallbacks are
    logged, never silent.
    """
    frame = decided
    gd = pd.to_datetime(frame.get("gameday"), errors="coerce")
    if gd.notna().any() and len(frame):
        cutoff = gd.max() - pd.Timedelta(days=days)
        current = frame[gd >= cutoff]
        prior = frame[gd < cutoff]
        grown = 0
        while len(current) < min_current and grown < max_days - days:
            grown += 1
            cutoff = gd.max() - pd.Timedelta(days=days + grown)
            current = frame[gd >= cutoff]
            prior = frame[gd < cutoff]
        if grown:
            logger.info("drift window: only %d decided game(s) in the last "
                        "%d day(s); widened to %d day(s) for a current "
                        "window of %d", len(frame[gd >= gd.max()
                                                  - pd.Timedelta(days=days)]),
                        days, days + grown, len(current))
    else:  # no usable gameday: degrade to a positional tail, logged
        current = frame.tail(min(60, len(frame)))
        prior = frame.iloc[:max(len(frame) - len(current), 0)]
    if not len(current) or len(current) < min_current:  # sparse tail
        logger.warning("drift window: %d decided game(s) available against a "
                       "target of %d; falling back to the last 60 decided "
                       "games", len(current), min_current)
        current = frame.tail(min(60, len(frame)))
        prior = frame.iloc[:max(len(frame) - len(current), 0)]
    baseline = prior.tail(max(3 * len(current), min_baseline)) \
        if len(prior) else prior
    baseline = _composition_matched_baseline(baseline, prior, current,
                                             min_baseline)
    return baseline, current


#: The covariate whose composition shift between windows is KNOWN seasonal
#: structure, not drift: the NBA calendar concentrates postseason games in a
#: six-week tail, so a Finals current window is 100% playoff games against a
#: prior that is mostly regular season. Every feature that moves with the
#: postseason (playoff-team Elo, pace, assists) then reads as ALERT even
#: though nothing about the model's world changed - the same three pages
#: three Finals windows running.
SEASONALITY_COVARIATE = "is_playoffs"


def _composition_matched_baseline(baseline: pd.DataFrame, prior: pd.DataFrame,
                                  current: pd.DataFrame,
                                  min_baseline: int) -> pd.DataFrame:
    """Restrict the baseline to the current window's seasonal composition.

    A drift comparison answers "has the world the model sees changed?". When
    the current window is DOMINATED by one state of a known seasonal
    covariate (a Finals tail: every game a playoff game) and the prior slice
    is mostly the other state (the regular season), the raw comparison
    measures the calendar, not drift - is_playoffs itself pages at PSI 4.2,
    and every covariate-correlated feature pages with it. The like-for-like
    comparison restricts the baseline to prior rows in the SAME state
    (this season's earlier playoff rounds, then the prior postseason),
    newest first, keeping the time-adjacency the window pair is built on.

    Floors and honesty: the matched baseline must still hold at least
    ``3 x len(current)`` rows (the same like-for-like floor the window pair
    uses) or the restriction is refused and the mixed baseline stays -
    with a loud log, so an under-supplied match degrades to paging rather
    than to silence. A mixed current window (the regular season's own
    composition) is left untouched: matching is for the tail seasons, not
    a new filter on every window.
    """
    covariate = SEASONALITY_COVARIATE
    if (not len(current) or not len(prior)
            or covariate not in current.columns
            or covariate not in prior.columns):
        return baseline
    cur_share = pd.to_numeric(current[covariate], errors="coerce").mean()
    base_share = pd.to_numeric(baseline[covariate], errors="coerce").mean()
    if not (cur_share == cur_share and base_share == base_share):
        return baseline
    if not (cur_share >= 0.5 and base_share < cur_share - 0.2):
        return baseline
    state = cur_share >= 0.5
    matched = prior[pd.to_numeric(prior[covariate], errors="coerce")
                    .ge(0.5) == state]
    # The floor clears INSUFFICIENT_BASELINE too: a matched baseline below
    # the judgable-gate constant would trade pages for INSUFFICIENT rows -
    # quiet, but dishonestly ("cannot judge" instead of "compared like
    # for like").
    needed = max(3 * len(current), INSUFFICIENT_BASELINE)
    if len(matched) < needed:
        logger.warning("drift window: current window is %.0f%% %s against a "
                       "%.0f%% prior, but only %d matched prior row(s) exist "
                       "(floor %d); keeping the mixed baseline - statuses on "
                       "covariate-correlated features will page as seasonal "
                       "structure",
                       100 * cur_share, covariate, 100 * base_share,
                       len(matched), needed)
        return baseline
    matched_baseline = matched.tail(
        max(needed, min(min_baseline, len(matched))))
    logger.info("drift window: current window is %.0f%% %s against a %.0f%% "
                "prior; baseline composition-matched to %d %s row(s) so the "
                "comparison measures drift, not the calendar",
                100 * cur_share, covariate, 100 * base_share,
                len(matched_baseline), covariate)
    return matched_baseline


#: Content key of the last "feature drift" headline already logged at INFO
#: (2026-10-05 run-log review: the delivered log carried the SAME line twice,
#: 217 ms apart — the feature-report CSV pass and the model-monitor JSON pass
#: both call feature_drift with the identical window pair and blend
#: importances, so the repeats were byte-identical and tellable apart by no
#: one). Keyed on the RENDERED CONTENT, not a call counter: a window that
#: moves between calls logs again as a distinct line. See _log_feature_drift.
_LAST_LOGGED_DRIFT: tuple | None = None


def _log_feature_drift(n_features: int, n_warns: int, n_alerts: int,
                       detail: str) -> None:
    """Log the drift headline once per DISTINCT verdict, not per call.

    ``write_run_engine_feature_artifacts`` (feature report phase) and the
    monitor-JSON pass in master_pipeline compute the SAME drift table — same
    ``drift_windows`` pair, same blend importances — so their two calls
    rendered an identical line in the 2026-10-05 run. The first verdict of a
    given content logs at INFO; an identical repeat drops to DEBUG; a
    verdict that changed logs at INFO again.
    """
    global _LAST_LOGGED_DRIFT
    key = (int(n_features), int(n_warns), int(n_alerts), detail)
    if key == _LAST_LOGGED_DRIFT:
        logger.debug("feature drift unchanged: %d features, %d warning(s), "
                     "%d alert(s)", n_features, n_warns, n_alerts)
        return
    logger.info("feature drift: %d features, %d warning(s), %d alert(s); "
                "statuses on noise-adjusted PSI with a location gate%s",
                n_features, n_warns, n_alerts, detail)
    _LAST_LOGGED_DRIFT = key


def feature_drift(baseline_games: pd.DataFrame, current_games: pd.DataFrame,
                  weights: dict[str, float] | None = None) -> list[dict]:
    """Per-feature drift status, structured like MLB's ``compute_feature_drift``.

    Statuses are assigned on the NOISE-ADJUSTED PSI and only escalate above OK
    when the mean ALSO moved beyond its sampling error (the location gate):
    PSI responds to any distributional change, including binning wiggle on
    quantized features, and a status that fires without a location shift is
    noise wearing a costume.  Windows too small to judge (see
    ``INSUFFICIENT_BASELINE``) report INSUFFICIENT and never page.
    """
    out = []
    for feature in config.active_moneyline_feature_cols():
        if feature not in baseline_games.columns:
            continue
        # Observations only. A carried value repeats the team's previous
        # profile, so letting it into the PSI or the location gate compares a
        # stale snapshot against fresh ones and invents drift (or hides it);
        # the 2026-09-29 13:29 vs 15:38 comparison nearly ran on exactly
        # that. Where the builder stamps no provenance, dropna() is the
        # whole story and nothing changes.
        baseline = _measured_values(baseline_games, feature).dropna()
        current = (_measured_values(current_games, feature).dropna()
                   if feature in current_games.columns else pd.Series(dtype=float))
        n_b, n_c = len(baseline), len(current)
        if n_b == 0 or n_c == 0:
            out.append({"feature": feature,
                        "current_mean": round(float(current.mean()), 4) if n_c else 0.0,
                        "baseline_mean": round(float(baseline.mean()), 4) if n_b else 0.0,
                        "psi": 0.0, "psi_adjusted": 0.0, "noise_floor": 0.0,
                        "mean_shift": 0.0, "shift_se": 0.0,
                        "location_shift": False, "status": "INSUFFICIENT",
                        "weight_pct": (weights or {}).get(feature) if weights else None,
                        "n_baseline": int(n_b), "n_current": int(n_c)})
            continue
        psi = _psi(current, baseline)
        noise = psi_noise_floor(n_b, n_c)
        psi_adjusted = max(psi - noise, 0.0)
        mean_shift = float(current.mean() - baseline.mean())
        if n_b + n_c > 2:
            pooled_sd = np.sqrt(((n_b - 1) * baseline.var(ddof=1)
                                 + (n_c - 1) * current.var(ddof=1))
                                / (n_b + n_c - 2))
        else:
            pooled_sd = 0.0
        if pooled_sd > 0:
            # Games in a short window share teams (~7 starts each in MLB's
            # derivation), so the naive SE understates true variance; the
            # same clustering inflation applies to a week of NBA games.
            shift_se = float(pooled_sd * np.sqrt(1.0 / n_b + 1.0 / n_c) * 1.5)
            location_shift = abs(mean_shift) > 2.0 * shift_se
        else:
            shift_se = 0.0
            location_shift = psi_adjusted > 0
        if n_b < INSUFFICIENT_BASELINE or n_c < INSUFFICIENT_CURRENT:
            status = "INSUFFICIENT"
        else:
            status = psi_status(psi_adjusted) if location_shift else "OK"
        out.append({"feature": feature,
                    "current_mean": round(float(current.mean()), 4),
                    "baseline_mean": round(float(baseline.mean()), 4),
                    "psi": round(psi, 6),
                    "psi_adjusted": round(psi_adjusted, 6),
                    "noise_floor": round(noise, 6),
                    "mean_shift": round(mean_shift, 6),
                    "shift_se": round(shift_se, 6),
                    "location_shift": bool(location_shift),
                    "status": status,
                    # Weights arrive as blend-weighted percentages (0-100,
                    # mirroring MLB's helper contract), so they pass through
                    # as-is; the previous double-scaling published 4191.85
                    # where 41.9 belonged.
                    "weight_pct": (weights or {}).get(feature) if weights else None,
                    "n_baseline": int(n_b), "n_current": int(n_c)})
    n_warns = sum(r["status"] == "WARN" for r in out)
    n_alerts = sum(r["status"] == "ALERT" for r in out)
    # A bare count is not actionable: "8 alert(s)" names nothing, and
    # the drift table is a file nobody opens mid-run. The features
    # behind each non-OK status ride on the log line itself (bounded,
    # so a frame-wide alert cannot write an unbounded line).
    _alerted = [r["feature"] for r in out if r["status"] == "ALERT"]
    _warned = [r["feature"] for r in out if r["status"] == "WARN"]
    _named = []
    for _label, _names in (("alerts", _alerted), ("warnings", _warned)):
        if not _names:
            continue
        _shown = ", ".join(_names[:25])
        if len(_names) > 25:
            _shown += f" (+{len(_names) - 25} more)"
        _named.append(f"{_label}: {_shown}")
    _detail = f" [{'; '.join(_named)}]" if _named else ""
    _log_feature_drift(len(out), n_warns, n_alerts, _detail)
    return out


#: Features whose builder writes 0.0 where it has no observation, so an
#: exact zero in these columns is AMBIGUOUS between "measured 0" and "no
#: data, defaulted". The coverage table's n_default_zero uses this set to
#: keep its count honest instead of flagging every tie game's net_points.
DEFAULT_ZERO_FEATURES = frozenset({
    "back_to_back_diff", "back_to_back",
    "back_to_back_home", "back_to_back_away",
})


def _measured_values(frame: pd.DataFrame, feature: str) -> pd.Series:
    """The feature's values restricted to genuine observations.

    The contract carries ``_measured_<feature>`` 0/1 provenance for features
    whose builder forward-fills (the play-by-play family). A carried value
    repeats the team's PREVIOUS profile, so counting it as an observation
    let a cache hole pose as coverage: the 13:29 run on 2026-09-29
    published 99.6% frame-wide event coverage that was substantially frozen
    constants. Where no flag exists, every non-null value is an observation
    and the frame passes through unchanged.
    """
    values = pd.to_numeric(frame[feature], errors="coerce")
    flag_col = f"_measured_{feature}"
    if flag_col in frame.columns:
        flag = pd.to_numeric(frame[flag_col], errors="coerce")
        values = values.where(flag > 0)
    return values


def coverage(baseline_games: pd.DataFrame,
             current_games: pd.DataFrame | None = None) -> list[dict]:
    """Per-feature non-null share, per drift window - MLB's dual-window shape.

    This is the visual backstop for the empty-pl_epm incident: a feature that
    failed to build shows plausible means in no table at all, but its coverage
    row says 0% measured in plain numbers.  Both windows are reported, so a
    feature that starved only recently cannot hide behind a healthy baseline.
    """
    rows = []
    windows = [("current", current_games) if current_games is not None
               else ("decided pool", baseline_games)]
    if current_games is not None:
        windows.append(("baseline", baseline_games))
    for window, frame in windows:
        for feature in config.active_moneyline_feature_cols():
            values = (pd.to_numeric(frame[feature], errors="coerce")
                      if feature in frame else pd.Series(dtype=float))
            # measured vs carried: where the builder stamps forward-fill
            # provenance, non-null splits into genuine observations (the
            # basis for every status above) and CARRIES of the team's last
            # profile. The split is what stops a hole in the play-by-play
            # sweep from publishing as full coverage. pct_measured is the
            # observation share; pct_nonnull keeps the old non-null number
            # so a run with carries shows the gap between the two.
            n_measured = 0
            n_carried = 0
            measured_values = values
            if len(values):
                measured_values = _measured_values(frame, feature)
                n_measured = int(measured_values.notna().sum())
                n_carried = int(values.notna().sum()) - n_measured
            pct = round(100 * float(measured_values.notna().mean()), 2) if len(values) else 0.0
            # n_default_zero counts values that arrive DEFAULT-FILLED rather
            # than observed - the back_to_back case, where a team with no
            # prior game in the window has no rest days to compare and the
            # builder writes 0. A real zero observation (a tie game's
            # net_points) is a measurement, not a default, and the old
            # (values == 0) count conflated the two. The frame does not carry
            # a per-value provenance flag, so the count is the defensible
            # proxy: features whose builder NEVER default-fills report 0 and
            # the column stays informative for the ones that do.
            n_default = 0
            if feature in DEFAULT_ZERO_FEATURES and len(values):
                n_default = int((values == 0).sum())
            rows.append({"feature": feature, "window": window,
                         "n_games": int(len(frame)),
                         "n_nonnull": int(values.notna().sum()) if len(values) else 0,
                         "pct_measured": pct, "pct_nonnull": pct,
                         "n_measured": n_measured, "n_carried": n_carried,
                         "n_default_zero": n_default,
                         "status": "STARVED" if pct < 25
                                   else "LOW_COVERAGE" if pct < 80 else "OK"})
    return rows


def write_run_engine_feature_artifacts(out_dir, date_c: str,
                                       baseline_games: pd.DataFrame,
                                       current_games: pd.DataFrame,
                                       weights=None) -> tuple[str, str]:
    out_dir = Path(out_dir)
    drift_name = f"{config.RUN_ENGINE_FEATURE_DRIFT_PREFIX}{date_c}.csv"
    coverage_name = f"{config.RUN_ENGINE_FEATURE_COVERAGE_PREFIX}{date_c}.csv"
    pd.DataFrame(feature_drift(baseline_games, current_games, weights)).to_csv(
        out_dir / drift_name, index=False)
    pd.DataFrame(coverage(baseline_games, current_games)).to_csv(
        out_dir / coverage_name, index=False)
    return drift_name, coverage_name


def ensemble_table(oof: pd.DataFrame, weights: dict[str, float]) -> list[dict]:
    try:
        from backend.evaluation import binary_metrics
    except ImportError:
        from evaluation import binary_metrics
    y = oof.home_win.to_numpy(float)
    rows = []
    for name in config.ENSEMBLE_MEMBERS:
        col = f"p_{name}"
        if col not in oof:
            continue
        metrics = binary_metrics(oof[col].to_numpy(float), y)
        rows.append({
            "name": name,
            "weight": round(float(weights.get(name, 0)), 4),
            "n_eval": int(metrics.get("n", 0) or 0),
            **{key: metrics.get(key) for key in ("auc", "brier", "logloss")},
        })
    return rows


def brier_headline(rolling: list[dict]) -> dict:
    """Summarize a rolling Brier series for the run log without lying.

    The log used to print only the LAST DAY's value as "rolling Brier
    <x> over N day(s)". On an NBA calendar the last day is often a single
    game - the 2026-09-29 runs printed 0.2713 and 0.2589, both of them one
    game - while the same-window games-weighted mean moved the other way
    (0.2098 -> 0.2091, improving). Reading the headline as an aggregate
    invited a false regression alarm on a one-game tail.
    """
    if not rolling:
        return {"summary": "n/a", "weighted": None, "last_day": None}
    n_games = sum(int(r.get("games", 0)) for r in rolling)
    total = sum(float(r["brier"]) * int(r.get("games", 0)) for r in rolling)
    weighted = total / n_games if n_games else None
    last = rolling[-1]
    return {
        "summary": (
            f"{weighted:.4f} games-weighted over {len(rolling)} day(s) "
            f"({n_games} game(s)); last day {last['date']} "
            f"{last['brier']:.4f} over {last.get('games', '?')} game(s)"
        ),
        "weighted": None if weighted is None else round(weighted, 6),
        "last_day": {"date": str(last["date"]),
                     "brier": float(last["brier"]),
                     "games": int(last.get("games", 0))},
    }


def rolling_brier(oof: pd.DataFrame, p_col="p_ensemble_calibrated",
                  window_days: int = 30) -> list[dict]:
    if oof is None or p_col not in oof or not len(oof):
        return []
    frame = oof.dropna(subset=[p_col, "home_win"]).copy()
    frame["gameday"] = pd.to_datetime(frame.gameday, errors="coerce")
    frame["brier"] = (pd.to_numeric(frame[p_col], errors="coerce") - frame.home_win) ** 2
    rows = []
    for day, group in frame.dropna(subset=["gameday"]).sort_values("gameday").groupby(frame.gameday.dt.date):
        rows.append({"date": str(day), "brier": float(group.brier.mean()), "games": int(len(group))})
    return rows


def _json_safe(value):
    """Convert pandas/numpy values and non-finite floats to strict JSON."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _dump(path, record: dict) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(record), indent=1, allow_nan=False,
                               default=str))
    return record


def _feature_importance_block(decomp) -> dict:
    """The MODEL WEIGHT decomposition, with its reading attached.

    The block itself is data - ``model_weight`` (the published
    column), ``member_shares`` (the served blend) and
    ``member_profiles`` (each member's own importances). The note
    is the one-line arithmetic a dashboard consumer needs: the
    column is a blend-weighted average, so a concentrated member
    share means the column is that member's own profile, not the
    ensemble's - the difference between "the model is 79% Elo"
    and "the elastic net holds 83% of the blend, so 79% of the
    column is that member's opinion rather than the ensemble's".
    """
    if not decomp:
        return {}
    block = dict(decomp)
    block.setdefault("reading", (
        "model_weight[f] = sum(member_shares[m] * "
        "member_profiles[m][f]) over members m; all percentages. "
        "A concentrated member share means the MODEL WEIGHT column "
        "is that member's own importance profile"))
    return block


def write_monitor_json(path, date_c: str, drift, cov, members, rolling,
                       baseline, config_meta=None, fold_info=None,
                       metrics=None, platt=None,
                       feature_importance=None) -> dict:
    drift = list(drift or [])
    cov = list(cov or [])
    # INSUFFICIENT is a window-size statement, not a problem statement - it
    # must not page any more than OK does.
    feature_alerts = [row for row in drift if row.get("status") in {"WARN", "ALERT"}]
    coverage_alerts = [row for row in cov if row.get("status") in {"LOW_COVERAGE", "STARVED"}]
    # Shared frontend monitor cards use these explicit fields.
    retrain = str(date_c)
    next_retrain = (
        pd.Timestamp(retrain) + pd.Timedelta(days=config.RETRAIN_CADENCE_DAYS)
    ).strftime("%Y%m%d")
    record = {
        "created_utc": pd.Timestamp.utcnow().isoformat(), "date": date_c,
        "config": config_meta or {},
        "last_retrained": retrain,
        "last_retrained_note": "Fresh NBA model trained this run",
        "next_retrain": next_retrain,
        "next_retrain_note": (
            f"next expected run in {config.RETRAIN_CADENCE_DAYS} day(s)"
        ),
        "upset_note": "NBA upset rate is computed from settled walk-forward history.",
        "alerts": {"retrain": False, "feature_drift": feature_alerts, "coverage": coverage_alerts},
        "baseline_brier": baseline, "brier_baseline": baseline,
        "brier_baseline_label": "Constant home-edge",
        "metrics": metrics or {}, "calibration": platt or {},
        "ensemble": members or [], "rolling_brier": rolling or [],
        "feature_drift": drift, "coverage": cov,
        "feature_coverage": cov, "folds": fold_info or {},
        # The MODEL WEIGHT column taken apart: member blend shares
        # and each member's own importance profile, so the column's
        # headline number (a concentrated blend IS one member's
        # profile - elo_diff at 78.627% on 2026-10-01) can be
        # decomposed by hand instead of rerunning the pipeline.
        "feature_importance": _feature_importance_block(
            feature_importance),
        "version_history": [], "features_metadata": {},
    }
    return _dump(path, record)


def write_run_engine_monitor(path, date_c: str, metrics=None, markets=None,
                             calibration=None, config_meta=None,
                             slate_history=None, dispersion=None) -> dict:
    # The fit block must report the ENGINE's constants, not a number a
    # refactor forgot: the draw count here used to be hardcoded at 4000 while
    # the derivation itself moved to the MLB pair (10k default, 50k SE-guard
    # tail), so the monitor described a resolution the run never used.
    try:
        from backend import distributions as _dist_mod
    except ImportError:  # pragma: no cover - direct script/import fallback
        import distributions as _dist_mod
    record = {
        "created_utc": pd.Timestamp.utcnow().isoformat(), "date": date_c,
        "config": config_meta or {}, "winner_cards": metrics or {},
        "market_metrics": metrics or {}, "calibration_cards": calibration or {},
        "slate_history": slate_history or [],
        "fit": {"distribution": "negative_binomial",
                "mc_draws": _dist_mod.MC_DRAWS,
                "mc_draws_tail": _dist_mod.MC_DRAWS_TAIL,
                "mc_se_target": _dist_mod.MC_SE_TARGET,
                "seed": _dist_mod.MC_SEED,
                "spread_grid": [min(config.SPREAD_GRID), max(config.SPREAD_GRID)],
                "total_grid": [min(config.TOTAL_GRID), max(config.TOTAL_GRID)],
                "half_stops": list(config.HALF_STOP_LINES)},
    }
    # Sealed-holdout transparency (NHL 62d00fd / MLB v3 parity): the
    # monitor carries the gate scope so the evaluation numbers ship with
    # the record of what the alpha layer was allowed to fit on.
    holdout = (dispersion or {}).get("holdout") or {}
    if holdout:
        record["fit"]["holdout"] = {
            **holdout,
            "holdout_days": _dist_mod.HOLDOUT_DAYS,
        }
    return _dump(path, record)
