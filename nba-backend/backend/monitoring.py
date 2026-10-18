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
try:
    from backend import manifest as manifest_mod
except ImportError:  # pragma: no cover - top-level module execution
    try:
        import manifest as manifest_mod
    except ImportError:  # metadata is optional at import time
        manifest_mod = None

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
                  max_days: int = 45,
                  as_of: Any | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    _log_stale_window(gd, current, as_of)
    return baseline, current


#: Beyond this age the "current" drift window is FROZEN, not current: no
#: decided game has landed for two weeks, so every status describes the
#: previous season's tail — not the slate being served.  The in-season
#: calendar never reaches it (the All-Star break is ~1 week; the longest
#: gap between decided NBA games is days), so the line fires only when the
#: off-season — or an ingest outage — has stopped the clock.  2026-10-11
#: run-log review: the delivered log printed the same 10 playoff-tail
#: ALERTs it had printed for weeks with nothing saying the window's newest
#: row was 2026-06-13; a reader reasonably took "10 alerts" as a statement
#: about the October slate it did not measure.
DRIFT_STALE_DAYS = 14


def _log_stale_window(gd: pd.Series, current: pd.DataFrame,
                      as_of: Any | None) -> None:
    """Disclose a drift window whose newest decided game is old.

    Disclosure only — statuses, windows and thresholds are untouched.  The
    warning states the window's end, its age, and the one fact a reader
    needs: recurring alerts on a stale window repeat the same frozen
    comparison and are not new signal (off-season roster and trade movement
    is UNMEASURED until play resumes).
    """
    if not gd.notna().any() or not len(current):
        return
    newest = gd.max()
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
    age = (as_of_ts.normalize() - newest.normalize()).days
    if age <= DRIFT_STALE_DAYS:
        return
    logger.warning(
        "drift window: newest decided game is %s — %d day(s) before this "
        "run (staleness floor %d); statuses measure that frozen %d-row "
        "tail against its baseline, NOT the slate being served. Off-season "
        "roster and trade movement is UNMEASURED until play resumes, and "
        "recurring alerts repeat the same frozen comparison — they are not "
        "new signal",
        newest.date(), age, DRIFT_STALE_DAYS, len(current))


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


def _log_feature_drift(n_features: int, n_evaluated: int, n_insufficient: int,
                       n_warns: int, n_alerts: int, detail: str) -> None:
    """Log the drift headline once per DISTINCT verdict, not per call.

    ``write_run_engine_feature_artifacts`` (feature report phase) and the
    monitor-JSON pass in master_pipeline compute the SAME drift table — same
    ``drift_windows`` pair, same blend importances — so their two calls
    rendered an identical line in the 2026-10-05 run. The first verdict of a
    given content logs at INFO; an identical repeat drops to DEBUG; a
    verdict that changed logs at INFO again.

    2026-10-15 log review (MLB T15 / NHL 2026-10-11 parity): the headline
    advertised only warnings/alerts, so INSUFFICIENT rows were invisible —
    a future small-window run would read "71 features, 0 warning(s), 0
    alert(s)" while judging nothing. The counts now always sum to the
    CSV, and the all-unevaluated case pages UNMEASURED instead of reading
    healthy. INSUFFICIENT alone still never warns (a window-size
    statement, not a defect).
    """
    global _LAST_LOGGED_DRIFT
    key = (int(n_features), int(n_evaluated), int(n_insufficient),
           int(n_warns), int(n_alerts), detail)
    if key == _LAST_LOGGED_DRIFT:
        logger.debug("feature drift unchanged: %d features, %d evaluated, "
                     "%d INSUFFICIENT, %d warning(s), %d alert(s)",
                     n_features, n_evaluated, n_insufficient,
                     n_warns, n_alerts)
        return
    logger.info("feature drift: %d features, %d evaluated, %d INSUFFICIENT, "
                "%d warning(s), %d alert(s); statuses on noise-adjusted "
                "PSI with a location gate%s",
                n_features, n_evaluated, n_insufficient,
                n_warns, n_alerts, detail)
    _LAST_LOGGED_DRIFT = key
    if n_features and n_evaluated == 0:
        logger.warning("feature drift: 0 of %d features evaluable this run "
                       "(every row INSUFFICIENT — current/baseline windows "
                       "under the PSI sample floor) — drift is UNMEASURED, "
                       "not clean", n_features)


def feature_drift(baseline_games: pd.DataFrame, current_games: pd.DataFrame,
                  weights: dict[str, float] | None = None,
                  feature_cols: list[str] | None = None) -> list[dict]:
    """Per-feature drift status, structured like MLB's ``compute_feature_drift``.

    Statuses are assigned on the NOISE-ADJUSTED PSI and only escalate above OK
    when the mean ALSO moved beyond its sampling error (the location gate):
    PSI responds to any distributional change, including binning wiggle on
    quantized features, and a status that fires without a location shift is
    noise wearing a costume.  Windows too small to judge (see
    ``INSUFFICIENT_BASELINE``) report INSUFFICIENT and never page.

    ``feature_cols`` (2026-10-08 totals-page parity): the feature view to
    report. The run-engine artifacts pass the DISTRIBUTION model's input
    view so the Totals & Run Lines dashboard measures the model it prices
    with; the moneyline monitor pass omits it and keeps the binary model's
    contract — that dashboard is untouched.
    """
    out = []
    cols = (list(feature_cols) if feature_cols is not None
            else config.active_moneyline_feature_cols())
    for feature in cols:
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
    _n_insufficient = sum(r["status"] == "INSUFFICIENT" for r in out)
    _log_feature_drift(len(out), len(out) - _n_insufficient, _n_insufficient,
                        n_warns, n_alerts, _detail)
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
             current_games: pd.DataFrame | None = None,
             feature_cols: list[str] | None = None) -> list[dict]:
    """Per-feature non-null share, per drift window - MLB's dual-window shape.

    This is the visual backstop for the empty-pl_epm incident: a feature that
    failed to build shows plausible means in no table at all, but its coverage
    row says 0% measured in plain numbers.  Both windows are reported, so a
    feature that starved only recently cannot hide behind a healthy baseline.

    ``feature_cols`` (2026-10-08): the feature view to report — the run-engine
    coverage artifact passes the DISTRIBUTION model's input view (Totals &
    Run Lines dashboard); the moneyline monitor pass omits it and keeps the
    binary model's contract.
    """
    rows = []
    cols = (list(feature_cols) if feature_cols is not None
            else config.active_moneyline_feature_cols())
    windows = [("current", current_games) if current_games is not None
               else ("decided pool", baseline_games)]
    if current_games is not None:
        windows.append(("baseline", baseline_games))
    for window, frame in windows:
        for feature in cols:
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
            # ELIGIBILITY MASK (2026-10-08 coverage audit). pl_rapm_* goes
            # NaN when the projected pool has no builder-eligible member for
            # that team-game - the prior-minutes floor, the 30-day recency
            # gate, or a feed position listing with no centre for that club.
            # The projection stamps that refusal per row in
            # ``_eligible_<feature>`` (lineup_projection), so this is a
            # measured mask, not a policy keyword: a null with a refusal
            # behind it is source-supported missingness and must not page;
            # a null with flag 1 or NO flag (team-game the projection never
            # ran for) is unexplained and keeps the raw threshold. The row
            # still reports the true pct - the mask changes what the status
            # MEANS, never the numbers. STRUCTURAL only when every missing
            # value is explained, so a partially-explained family keeps its
            # real alarm.
            n_refused = 0
            structural_reason = None
            flag_col = f"_eligible_{feature}"
            if flag_col in frame.columns and len(values):
                flag = pd.to_numeric(frame[flag_col], errors="coerce")
                missing = values.isna()
                n_refused = int((missing & (flag == 0)).sum())
                if n_refused and n_refused == int(missing.sum()):
                    structural_reason = (
                        f"eligible-pool refusal on {n_refused} row(s): no "
                        "builder-eligible member (minutes floor / 30-day "
                        f"recency / position listing), mask {flag_col}; "
                        "source-supported, not a build hole")
            status = ("STARVED" if pct < 25
                      else "LOW_COVERAGE" if pct < 80 else "OK")
            if structural_reason and status != "OK":
                status = "STRUCTURAL"
            if status != "STRUCTURAL":
                # The reason exists only to explain a STRUCTURAL status: a
                # healthy family with a few refused rows keeps its OK and
                # reports the count, never a stray explanation beside a
                # pass (the monitor renders a reason as THE finding).
                structural_reason = None
            row = {"feature": feature, "window": window,
                   "n_games": int(len(frame)),
                   "n_nonnull": int(values.notna().sum()) if len(values) else 0,
                   "pct_measured": pct,
                   "pct_nonnull": (round(100 * float(values.notna().mean()), 2)
                                   if len(values) else 0.0),
                   "n_measured": n_measured, "n_carried": n_carried,
                   "n_default_zero": n_default,
                   "n_pool_refused": n_refused,
                   "status": status}
            if structural_reason:
                row["structural_reason"] = structural_reason
            rows.append(row)
    return rows


def write_run_engine_feature_artifacts(out_dir, date_c: str,
                                       baseline_games: pd.DataFrame,
                                       current_games: pd.DataFrame,
                                       weights=None,
                                       feature_cols=None) -> tuple[str, str]:
    """Emit the Totals & Run Lines dashboard's drift/coverage CSVs.

    ``weights`` and ``feature_cols`` are the DISTRIBUTION model's own per-
    feature importances and input view (2026-10-08): the run-engine panels
    report the model that prices totals/run lines, never the binary
    moneyline blend. Omitted → the historical defaults (binary contract,
    no weights) so a degraded run emits an honest artifact instead of
    borrowing another model's numbers.
    """
    out_dir = Path(out_dir)
    drift_name = f"{config.RUN_ENGINE_FEATURE_DRIFT_PREFIX}{date_c}.csv"
    coverage_name = f"{config.RUN_ENGINE_FEATURE_COVERAGE_PREFIX}{date_c}.csv"
    pd.DataFrame(feature_drift(baseline_games, current_games, weights,
                               feature_cols=feature_cols)).to_csv(
        out_dir / drift_name, index=False)
    pd.DataFrame(coverage(baseline_games, current_games,
                          feature_cols=feature_cols)).to_csv(
        out_dir / coverage_name, index=False)
    return drift_name, coverage_name


def ensemble_table(oof: pd.DataFrame, weights: dict[str, float]) -> list[dict]:
    """Per-member OOF diagnostics on the GRADING population.

    Members and the published (deployed) blend must be scored on the SAME
    rows — regular-season rows of non-provisional folds — so the Model
    Ensemble table's blend-vs-strongest-member comparison is apples-to-
    apples (2026-10-05 alignment; MLB parity). Frames without a
    grades_pooled column fall back to the full frame.
    """
    try:
        from backend.evaluation import binary_metrics
    except ImportError:
        from evaluation import binary_metrics
    grade = (oof["grades_pooled"].astype(bool).to_numpy()
             if "grades_pooled" in oof.columns
             else np.ones(len(oof), dtype=bool))
    y = oof.home_win.to_numpy(float)[grade]
    rows = []
    for name in config.ENSEMBLE_MEMBERS:
        col = f"p_{name}"
        if col not in oof:
            continue
        metrics = binary_metrics(oof[col].to_numpy(float)[grade], y)
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


def _iso_date(date_c: str) -> str:
    """'20261006' -> '2026-10-06'; anything else passes through."""
    s = str(date_c)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 and s.isdigit() else s


def _coverage_row_mlb(row: dict) -> dict:
    """Project a coverage row onto MLB's exact report schema.

    MLB's ``feature_coverage`` rows carry the nine fields
    feature/window/n_games/n_nonnull/pct_nonnull/n_measured/pct_measured/
    n_default_zero/status. The NBA builder's extra provenance columns stay
    on the run-engine CSV; the monitor block ships MLB's shape so the two
    sports' reports are structurally identical. ``pct_nonnull`` is
    recomputed as the true non-null share (the builder's measured-only
    figure cannot exceed itself when carries exist).
    """
    n_games = int(row.get("n_games") or 0)
    n_nonnull = int(row.get("n_nonnull") or 0)
    out = {
        "feature": row.get("feature"),
        "window": row.get("window"),
        "n_games": n_games,
        "n_nonnull": n_nonnull,
        "pct_nonnull": (round(100.0 * n_nonnull / n_games, 2)
                        if n_games else 0.0),
        "n_measured": int(row.get("n_measured") or 0),
        "pct_measured": row.get("pct_measured"),
        "n_default_zero": int(row.get("n_default_zero") or 0),
        "status": row.get("status"),
    }
    if row.get("structural_reason"):
        out["structural_reason"] = row["structural_reason"]
    return out


VERSION_HISTORY_CAP = 20  # MLB training.VERSION_HISTORY_CAP parity


def _version_history_row(date_c: str, members, raw: dict, cal: dict,
                         platt: dict | None) -> dict:
    """One retrain snapshot — MLB ``model_version_history`` row schema
    (NFL monitoring parity): version stamp vYYYY.MM.DD, ISO date, roster
    weights at 4 decimals, the pooled metric keys (plain = raw walk-forward
    scores; ``*_calibrated`` = the prequential per-fold layer), and the
    deployed map's {a, b, n, method, floor}. Values missing from this run
    stay ABSENT — never fabricated (the shared page renders an absent cell
    as '—').
    """
    raw = raw or {}
    cal = cal or {}
    row = {
        "version": f"v{_iso_date(date_c)}".replace("-", "."),
        "date": _iso_date(date_c),
        "weights": {str(m.get("name")): round(float(m.get("weight") or 0.0), 4)
                    for m in (members or [])
                    if isinstance(m, dict) and m.get("name") is not None},
    }
    # Plain keys are the raw pooled scores; the *_calibrated twins are the
    # prequential layer's scores (cal_metrics carries them under plain keys).
    for key in ("auc", "brier", "logloss", "ece"):
        if isinstance(raw.get(key), (int, float)):
            row[key] = raw[key]
    for key, src in (("brier_calibrated", "brier"),
                     ("logloss_calibrated", "logloss"),
                     ("ece_calibrated", "ece")):
        if isinstance(cal.get(src), (int, float)):
            row[key] = cal[src]
    if (isinstance(platt, dict) and platt.get("a") is not None
            and platt.get("b") is not None):
        row["calibration"] = {
            "a": platt.get("a"), "b": platt.get("b"),
            **({"n": int(platt["n"])}
               if isinstance(platt.get("n"), (int, float)) else {}),
            **({"method": str(platt["method"])}
               if platt.get("method") else {}),
            **({"floor": float(platt["floor"])}
               if platt.get("floor") is not None else {}),
        }
    return row


def _rolling_version_history(path, current_row: dict) -> list[dict]:
    """The rolling Model Version History — MLB's model_version_history.json
    semantics with the dated monitor family as the store (NFL parity: NBA
    ships no separate master file).

    Every dated ``nba_model_monitor_*.json`` beside ``path`` carries the rows
    it knew, so folding them preserves every record that ever shipped: merge
    by ``version`` (newest artifact wins, THIS run's row always wins), never
    pull a record dated after this run (a fold over a mixed-age dir must not
    leak a later run into an earlier report), order oldest-first, keep the
    last VERSION_HISTORY_CAP rows. Records survive retention because each
    new artifact embeds the rows this fold recovered.
    """
    prefix = str(config.MODEL_MONITOR_JSON).split("{")[0]
    cur_date = str(current_row.get("date") or "")

    def _key(row: dict):
        return str(row.get("version") or
                   (row.get("date"), str(row.get("weights"))))

    rows: dict = {}
    try:
        files = sorted(Path(path).parent.glob(f"{prefix}*.json"),
                       key=lambda p: p.name, reverse=True)[:60]
    except OSError:  # pragma: no cover - unreadable artifact dir
        files = []
    for f in files:
        if f.name == Path(path).name:
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


def write_monitor_json(path, date_c: str, drift, cov, members, rolling,
                       baseline, config_meta=None, fold_info=None,
                       metrics=None, platt=None,
                       feature_importance=None, raw_metrics=None) -> dict:
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
    # MLB-shaped blocks the shared Model Monitor page renders verbatim:
    # the coverage report (nine-field rows), the feature documentation
    # (labels + hover tooltips), the rolling-Brier caption context, and the
    # per-retrain version history. All four used to ship empty or absent,
    # so the NBA dashboard rendered "(no detailed metadata)" on every drift
    # row and "No version history yet" where MLB shows its retrain table.
    features_meta: dict = {}
    try:
        _names = sorted({str(r.get("feature")) for r in drift + cov
                         if isinstance(r, dict) and r.get("feature")})
        if manifest_mod is not None and _names:
            features_meta, meta_warnings = manifest_mod.build_features_metadata(
                _names)
            for msg in meta_warnings:
                logger.warning(msg)
    except Exception:  # metadata is presentational - never fatal
        logger.warning("feature metadata build skipped (non-fatal)",
                       exc_info=True)
        features_meta = {}
    cal_metrics = metrics or {}
    raw = raw_metrics or {}
    # One snapshot row for THIS run in MLB's schema, accumulated with the
    # rows the dated family already carries (MLB's rolling-20 presentation).
    version_row = _version_history_row(date_c, members, raw, cal_metrics,
                                       platt)
    version_history = _rolling_version_history(path, version_row)
    record = {
        "created_utc": pd.Timestamp.utcnow().isoformat(), "date": date_c,
        "version": version_row["version"],
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
        # Headline block = MLB's shape: the deployed blend's OWN pooled OOF
        # scores (raw keys) plus the prequential calibrated twins, so the
        # ensemble TOTAL row reads the same numbers the Calibration KPI
        # cards show.
        "metrics": {
            "auc": raw.get("auc"), "brier": raw.get("brier"),
            "logloss": raw.get("logloss"), "ece": raw.get("ece"),
            "brier_calibrated": cal_metrics.get("brier"),
            "logloss_calibrated": cal_metrics.get("logloss"),
            "ece_calibrated": cal_metrics.get("ece"),
            "calibrator_gated_out": False,
        },
        "calibration": platt or {},
        "ensemble": members or [], "rolling_brier": rolling or [],
        "feature_drift": drift, "coverage": cov,
        "feature_coverage": [_coverage_row_mlb(r) for r in cov],
        "features_metadata": features_meta,
        "rolling_brier_meta": {
            "window_days": 30,
            "min_games_per_day": 1,
            "excluded_sparse_days": 0,
            "calibrator_is_identity": False,
            # The series is p_ensemble_calibrated: the PREQUENTIAL per-fold
            # layer (fold k's map fitted on folds < k's published pairs) —
            # the same strictly prior series the OOF metrics score. The
            # pooled all-OOF Platt map is the serving layer and never prices
            # this series (NHL parity note).
            "map_scope_note": ("Points use the prequential per-fold "
                               "calibration layer (fit on prior OOF folds "
                               "only); the pooled all-OOF Platt map is the "
                               "serving layer and is not in this series."),
        },
        "folds": fold_info or {},
        # The MODEL WEIGHT column taken apart: member blend shares
        # and each member's own importance profile, so the column's
        # headline number (a concentrated blend IS one member's
        # profile - elo_diff at 78.627% on 2026-10-01) can be
        # decomposed by hand instead of rerunning the pipeline.
        "feature_importance": _feature_importance_block(
            feature_importance),
        "version_history": version_history,
    }
    return _dump(path, record)


# ---------------------------------------------------------------------------
# Run-Line & Totals Monitor (run-engine counterpart of the moneyline
# monitor artifact — MLB ``run_engine_monitor_*.json`` shape, NBA data)
# ---------------------------------------------------------------------------
# The 2026-10-11 structural repair: this artifact carries the distribution
# distribution scorer's ``{run_line, totals}`` summary under BOTH
# ``winner_cards`` and ``market_metrics``, so the Run-Engine Model card
# rendered two mislabelled rows instead of MLB's per-market-line table, the
# winner cards had no over_under/derived_ml sections at all, and
# ``slate_history`` was never written — the page's Rolling ECE-Calibrated
# History stayed on "first build starts empty" forever. Everything below is
# the NHL/NFL port of MLB's card anatomy on NBA's integer grids.


def _grid_col(base: str, x: float) -> str:
    """Artifact grid column tag: '-' -> 'm', '.' -> '_', integral line drops
    its trailing '.0' — ``p_push_total_225`` / ``p_home_cover_m2``."""
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
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def _markets_logloss(y: np.ndarray, p: np.ndarray) -> float | None:
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], np.clip(p[ok], 1e-6, 1 - 1e-6)
    if len(y) < 2:
        return None
    return round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4)


def _markets_card_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """MLB winner-card metric shape (push-excluded 2-way populations).

    Argument order is (y, p) — outcomes first — the contract every call
    site unpacks (the NHL 2026-10-11 / NFL 2026-09-24 pair-order fix: a
    swapped call keeps Brier sane and sends logloss to ~5-7 nats).
    """
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


def _grid_values(df: pd.DataFrame, base: str, grid) -> np.ndarray | None:
    """(n, len(grid)) numeric matrix of ``base_<line>`` columns, or None when
    the artifact carries no complete grid for that namespace."""
    cols = [_grid_col(base, float(x)) for x in grid]
    if any(c not in df.columns for c in cols):
        return None
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _two_way(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2-way re-normalized P(side a) — push mass folded out of both sides."""
    denom = a + b
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(denom) & (denom > 0)
    out = np.full(a.shape, np.nan)
    out[ok] = a[ok] / denom[ok]
    return out


def _fair_line(p_side: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Per-game FAIR line — grid argmin of |2-way P - 0.5|, ties to the
    LOWER line (the page's ``fair_total_lines`` / ``fair_spread_thresholds``
    semantics, so artifact and page price the same line). NaN when no grid
    line can be priced — never fabricated."""
    delta = np.abs(p_side - 0.5)
    delta = np.where(np.isfinite(delta), delta, np.inf)
    best = np.full(p_side.shape[0], np.nan)
    best_delta = np.full(p_side.shape[0], np.inf)
    for i in range(len(grid)):
        take = delta[:, i] < best_delta - 1e-12
        best_delta[take] = delta[take, i]
        best[take] = float(grid[i])
    best[~np.isfinite(best_delta)] = np.nan
    return best


def _column_at(mat: np.ndarray, grid: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Per-row lookup of ``mat``'s column at each row's own grid line."""
    out = np.full(len(line), np.nan)
    ok = np.isfinite(line)
    pos = np.searchsorted(grid, line[ok])
    valid = (pos >= 0) & (pos < len(grid))
    grid_pos = np.clip(pos, 0, len(grid) - 1)
    valid &= grid[grid_pos] == line[ok]
    rows = np.flatnonzero(ok)[valid]
    out[rows] = mat[rows, grid_pos[valid]]
    return out


def _pick_side(p_side: np.ndarray, y_side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Favored-side framing (MLB ``compute_winner_cards``): every game
    contributes the probability of the side the model picks and 1 when that
    side won — so ``actual_win_rate`` IS ``win_rate`` (pick accuracy) and
    ``predicted_mean`` is the mean pick probability, exactly as MLB / NHL /
    NFL ship them."""
    pick_home = np.where(np.isfinite(p_side), p_side >= 0.5, True)
    p_pick = np.where(pick_home, p_side, 1.0 - p_side)
    y_pick = np.where(pick_home, y_side, 1.0 - y_side)
    return y_pick, p_pick


def _attach_full_pool_auc(card: dict, y: np.ndarray, p: np.ndarray,
                          sealed_ok: np.ndarray) -> None:
    """Re-point a winner card's AUC at the FULL graded population.

    2026-10-13: the run_line / derived_ml cards grade only home-lean rows
    (P > 50%) so their calibration metrics describe the population a bet
    would actually be placed on. But AUC is a RANKING metric, and gating
    it to P > 0.5 range-restricts the score to (0.5, 1] — it throws away
    every pair that spans the 50/50 line, which is exactly where the
    discrimination lives. On the 2026-10-13 NBA artifact that collapsed
    derived-ML AUC from **0.7304** (full non-tie pool) to **0.6435**
    (lean-gated) — and 0.6435 is also what the old PICK-SIDE framing
    scored, which is why it read as "still using the pick side".

    The underlying column is genuinely home-oriented (mean P(home win)
    0.606 when home won vs 0.475 when home lost); only the reporting
    population was wrong. So AUC is recomputed here over every finite,
    non-dead row, and ``auc_n`` states that population so the card never
    implies the two metrics share one. The pooled and sealed-holdout
    figures are both re-pointed. No-op when the pool is empty or
    single-class.
    """
    if not card:
        return
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return
    y, p = y[ok], p[ok]
    sealed_ok = sealed_ok[ok]
    pooled = _markets_card_metrics(y, p)
    card["auc"] = pooled.get("auc")
    card["auc_n"] = pooled.get("n")
    hold = sealed_ok
    if hold.any():
        hold_card = _markets_card_metrics(y[hold], p[hold])
        card.setdefault("holdout", {})
        if card["holdout"]:
            card["holdout"]["auc"] = hold_card.get("auc")
            card["holdout"]["auc_n"] = hold_card.get("n")




def _sealed_holdout_crps(df: pd.DataFrame, outcome_col: str,
                         prob_base: str, grid) -> float | None:
    """Mean grid CRPS over sealed-holdout rows (the "Holdout CRPS" cell).

    CRPS = step * mean_rows(sum_lines((P(Y > line) - 1{y > line})^2)) over
    the sport's integer unit grid (raw cover/over legs — no 2-way rescale;
    push mass belongs to the CDF). Population: ALL sealed rows with a
    finite outcome and complete grid legs — deliberately NOT lean-gated
    (gating a distribution metric to P > 50% would range-restrict it, the
    same reason AUC is re-pointed at the full pool). Returns None when the
    grid is absent or no sealed row qualifies. Tails beyond the grid are
    truncated (documented, identical every run — comparable across runs).
    """
    import pandas as pd  # noqa: F811 (module already imports numpy/pandas)
    grid = np.asarray(grid, dtype=float)
    if grid.size < 2:
        return None
    step = float(grid[1] - grid[0])
    n = len(df)
    y = pd.to_numeric(df.get(outcome_col), errors="coerce").to_numpy(float)         if df.get(outcome_col) is not None else np.full(n, np.nan)
    P = _grid_values(df, prob_base, grid)
    if P is None:
        return None
    sealed = (df.get("frame_view") == "sealed").to_numpy()         if "frame_view" in df.columns else np.zeros(n, bool)
    ok = sealed & np.isfinite(y) & np.isfinite(P).all(axis=1)
    if not ok.any():
        return None
    gap = P[ok] - (y[ok][:, None] > grid[None, :]).astype(float)
    return round(float(step * (gap ** 2).sum(axis=1).mean()), 4)


def markets_winner_cards(oof_rows: pd.DataFrame) -> dict[str, dict]:
    """The three binary winner cards from the decided OOF market rows —
    MLB ``compute_winner_cards`` field shape on the NBA grids:

      over_under   pick Over at the game's own fair total (grid argmin of
                   |2-way P(over) - 0.5|) else Under; total == line pushes
                   excluded (pick-side framing — the standing pick flips
                   with the lean);
      run_line     HOME cover at the game's own fair spread threshold (the
                   0.5-ladder step closest to the 50/50 anchor), graded
                   only on home-lean rows (P > 50%); margin == T pushes and
                   no-lean rows excluded (HOME-side framing, 2026-10-12);
      derived_ml   the run engine's own HOME win probability at the +-0.5
                   dead cut (ties excluded), again home-lean rows only —
                   distinct from the moneyline ensemble.

    The home-side cards share the binary moneyline KPI structure: one row
    per graded game at its home probability, W/(W+L) as the win rate, and
    Actual/Predicted over the SAME home-lean population.
    """
    df = oof_rows
    if df is None or not len(df):
        return {}
    n = len(df)
    total = pd.to_numeric(df.get("total"), errors="coerce").to_numpy(float) \
        if df.get("total") is not None else np.full(n, np.nan)
    margin = pd.to_numeric(df.get("margin"), errors="coerce").to_numpy(float) \
        if df.get("margin") is not None else np.full(n, np.nan)
    sealed = (df.get("frame_view") == "sealed").to_numpy() \
        if "frame_view" in df.columns else np.zeros(n, bool)

    def _build(y: np.ndarray, p: np.ndarray,
               sealed_ok: np.ndarray,
               auc_pool: tuple[np.ndarray, np.ndarray, np.ndarray]
               | None = None) -> dict:
        """Card metrics over the pooled pool + the sealed-holdout twin
        (``sealed_ok`` is the caller's row mask re-indexed onto y/p).

        ``auc_pool`` is an optional (y, p, sealed) triple over the FULL
        graded population when the card itself is graded on a gated subset.
        """
        keep = np.isfinite(y) & np.isfinite(p)
        if not keep.any():
            return {}
        card = _markets_card_metrics(y[keep], p[keep])
        hold = keep & sealed_ok
        card["holdout"] = (_markets_card_metrics(y[hold], p[hold])
                           if hold.any() else {})
        card["source"] = "oof_decided_store"
        if auc_pool is not None:
            _attach_full_pool_auc(card, *auc_pool)
        return card

    cards: dict[str, dict] = {}

    # --- over/under at each game's own fair total -------------------------
    tg = np.asarray(config.TOTAL_GRID, dtype=float)
    over = _grid_values(df, "p_over", tg)
    under = _grid_values(df, "p_under", tg)
    if over is not None and under is not None:
        ft = _fair_line(_two_way(over, under), tg)
        po = _column_at(over, tg, ft)
        pu = _column_at(under, tg, ft)
        p2 = _two_way(po, pu)
        ok = np.isfinite(p2) & np.isfinite(total) & np.isfinite(ft) \
            & (total != ft)
        y_side = (total > ft).astype(float)
        y, p = _pick_side(p2, y_side)
        cards["over_under"] = _build(y[ok], p[ok], sealed[ok])

    # --- run line: HOME cover at its own fair spread threshold -----------
    # HOME-SIDE framing (2026-10-12): the line is the fair threshold on the
    # 0.5 ladder (grid argmin |2-way P - 0.5|, home orientation — the step
    # closest to a coin flip), the probability is the HOME cover with the
    # push band folded out, and only home-lean rows (P > 50%) are graded:
    # no-lean rows and whole-line pushes are excluded — the binary moneyline
    # KPI population with the home team as the standing pick.
    sg = np.asarray(config.SPREAD_GRID, dtype=float)
    hc = _grid_values(df, "p_home_cover", sg)
    ac = _grid_values(df, "p_away_cover", sg)
    if hc is not None and ac is not None:
        ft = _fair_line(_two_way(hc, ac), sg)
        ph = _column_at(hc, sg, ft)
        pa = _column_at(ac, sg, ft)
        p2 = _two_way(ph, pa)
        graded = np.isfinite(p2) & np.isfinite(margin) & np.isfinite(ft) \
            & (margin != ft)
        lean = graded & (p2 > 0.5)
        y_home = (margin > ft).astype(float)
        cards["run_line"] = _build(y_home[lean], p2[lean], sealed[lean],
                                   auc_pool=(y_home[graded], p2[graded],
                                             sealed[graded]))

    # --- derived moneyline (HOME-side at the +-0.5 dead cut) -------------
    # The run engine's own home win probability: ties (margin == 0) are
    # dead mass and rows without a home lean are excluded from the card's
    # calibration metrics — only the graded population changes vs the old
    # pick-side card; the fields and their meanings are unchanged. AUC is
    # scored over the FULL non-tie pool (see ``_attach_full_pool_auc``).
    ml = pd.to_numeric(df.get("derived_ml"), errors="coerce").to_numpy(float) \
        if df.get("derived_ml") is not None else np.full(n, np.nan)
    graded = np.isfinite(ml) & np.isfinite(margin) & (margin != 0)
    lean = graded & (ml > 0.5)
    y_home = (margin > 0).astype(float)
    cards["derived_ml"] = _build(y_home[lean], ml[lean], sealed[lean],
                                 auc_pool=(y_home[graded], ml[graded],
                                           sealed[graded]))

    # Holdout CRPS (replaces the degenerate Holdout ECE-cal cell): totals
    # CRPS on the O/U card, margin CRPS on the run-line + derived-ML cards.
    _t = _sealed_holdout_crps(oof_rows, "total", "p_over", config.TOTAL_GRID)
    _m = _sealed_holdout_crps(oof_rows, "margin", "p_home_cover", config.SPREAD_GRID)
    if _t is not None and isinstance(cards.get("over_under", {}).get("holdout"), dict):
        cards["over_under"]["holdout"]["crps"] = _t
    for _k in ("run_line", "derived_ml"):
        if _m is not None and isinstance(cards.get(_k, {}).get("holdout"), dict):
            cards[_k]["holdout"]["crps"] = _m
    return cards


def _run_engine_line_pairs(df: pd.DataFrame, line_kind: str,
                           line: float) -> tuple[np.ndarray, np.ndarray]:
    """(y, p) for one market line, 2-way push-excluded — the model-card
    per-line scoring basis: ``over_<U>`` (total > U, pushes on total == U),
    ``home_cover_<L>`` (margin > L, pushes on margin == L),
    ``derived_moneyline`` (home-side, ties dead mass). Returns outcomes
    first — the (y, p) contract the call sites unpack."""
    n = len(df)
    total = pd.to_numeric(df.get("total"), errors="coerce").to_numpy(float) \
        if df.get("total") is not None else np.full(n, np.nan)
    margin = pd.to_numeric(df.get("margin"), errors="coerce").to_numpy(float) \
        if df.get("margin") is not None else np.full(n, np.nan)

    def _col(base: str) -> np.ndarray:
        c = _grid_col(base, line)
        if c not in df.columns:
            return np.full(n, np.nan)
        return pd.to_numeric(df[c], errors="coerce").to_numpy(float)

    if line_kind == "over":
        po, pp = _col("p_over"), _col("p_push_total")
        pu = 1.0 - po - np.where(np.isfinite(pp), pp, 0.0)
        ok = np.isfinite(total) & np.isfinite(po) & (total != line) \
            & ((po + pu) > 0)
        y = (total > line).astype(float)
        p = po / (po + pu)
    elif line_kind == "spread":
        ph, pp = _col("p_home_cover"), _col("p_push")
        pa = 1.0 - ph - np.where(np.isfinite(pp), pp, 0.0)
        ok = np.isfinite(margin) & np.isfinite(ph) & (margin != line) \
            & ((ph + pa) > 0)
        y = (margin > line).astype(float)
        p = ph / (ph + pa)
    else:  # derived moneyline — home side, ties excluded
        ml = pd.to_numeric(df.get("derived_ml"), errors="coerce").to_numpy(float) \
            if df.get("derived_ml") is not None else np.full(n, np.nan)
        ok = np.isfinite(ml) & np.isfinite(margin) & (margin != 0)
        y = (margin > 0).astype(float)
        p = ml / (ml + (1.0 - ml))
    y, p = y[ok], p[ok]
    good = np.isfinite(p)
    return y[good], p[good]


def _run_engine_market_metrics(oof_market_rows: pd.DataFrame) -> dict:
    """Per-line OOF metrics for the Run-Engine Model card (MLB
    ``market_metrics`` shape): ECE/Brier/Logloss per canonical line over the
    artifact's own rows. The run engine ships no calibration map, so
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


def write_run_engine_monitor(path, date_c: str, markets=None,
                             calibration=None, config_meta=None,
                             ml_reference=None,
                             markets_persisted: bool = True,
                             markets_persist_error=None) -> dict:
    """``Run-Line & Totals Monitor`` artifact (MLB run-engine-monitor/v2
    shape, NBA data): the three winner cards computed from THIS run's
    decided OOF store, the per-line ``market_metrics`` + row-derived ``fit``
    block for the model card / fit panel, the markets_persisted flags, and a
    ``slate_history`` point per card dated today (the frontend folds the
    dated monitors' accumulating histories into the rolling table).

    ``markets`` is the persisted OOF market frame (kind='oof' rows); empty
    frames yield empty cards — nothing fabricated. ``ml_reference`` is the
    shared moneyline ensemble's pooled win rate — the derived-ML card's
    comparison anchor, passed in by the caller that owns the moneyline OOF
    store.

    ``calibration`` carries split-skew variance parameters, causal CRPS maps,
    sealed evaluation scope, and the actual Monte Carlo draw resolution.
    """
    iso_date = (f"{date_c[:4]}-{date_c[4:6]}-{date_c[6:8]}"
                if len(str(date_c)) == 8 and str(date_c).isdigit()
                else str(date_c))
    rows = markets if isinstance(markets, pd.DataFrame) else pd.DataFrame()
    if "kind" in rows.columns:
        rows = rows[rows["kind"].eq("oof")].copy()

    cards = markets_winner_cards(rows)
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
        "markets_persisted": bool(markets_persisted),
        "markets_persist_error": markets_persist_error,
        "winner_cards": cards,
        "slate_history": slate_history,
        "fit": {"distribution": "causal_split_skew_crps_mc",
                "family": "two_sided_asymmetric_gaussian_split_skew",
                "priced_by": "inverse_calibrated_cdf_monte_carlo",
                "variance": (calibration or {}).get("maps", {}).get("variance", {}),
                # Integer-lattice correction disclosure (totals + margin fit
                # independently; magnitude and gate verdict, no delta vectors).
                "lattice": (calibration or {}).get("lattice_summary", {}),
                "monte_carlo": (calibration or {}).get("monte_carlo", {}),
                "spread_grid": [min(config.SPREAD_GRID), max(config.SPREAD_GRID)],
                "total_grid": [min(config.TOTAL_GRID), max(config.TOTAL_GRID)],
                "half_stops": list(config.HALF_STOP_LINES)},
        "market_metrics": _run_engine_market_metrics(rows),
        "config": config_meta or {},
    }
    # Sealed-holdout per-line metrics (MLB v3 / NHL parity): the last
    # HOLDOUT_DAYS of OOF rows — stamped frame_view='sealed' — scored at the
    # canonical lines against the same grids. The sealed window never fit
    # the variance layer nor the causal CDF calibrators, so this is the engine's
    # honest recent-form evaluation, nested per line under each metric's own
    # "holdout" key exactly like MLB's card shape.
    hold_rows = None
    if len(rows) and "frame_view" in rows.columns \
            and (rows["frame_view"] == "sealed").any():
        hold_rows = rows.loc[rows["frame_view"] == "sealed"]
    if hold_rows is not None:
        for u in config.RUN_ENGINE_FIXED_TOTALS:
            y, p = _run_engine_line_pairs(hold_rows, "over", float(u))
            if len(y):
                entry = record["market_metrics"].get(f"over_{u}") or {}
                entry["holdout"] = _markets_card_metrics(y, p)
                record["market_metrics"][f"over_{u}"] = entry
        for l in config.RUN_ENGINE_CANONICAL_SPREADS:
            y, p = _run_engine_line_pairs(hold_rows, "spread", float(l))
            if len(y):
                key = f"home_cover_{str(l).replace('.', '_')}"
                entry = record["market_metrics"].get(key) or {}
                entry["holdout"] = _markets_card_metrics(y, p)
                record["market_metrics"][key] = entry
        y, p = _run_engine_line_pairs(hold_rows, "ml", 0.0)
        if len(y):
            entry = record["market_metrics"].get("derived_moneyline") or {}
            entry["holdout"] = _markets_card_metrics(y, p)
            record["market_metrics"]["derived_moneyline"] = entry
        record["holdout_gate"] = {
            "n_holdout": int(len(hold_rows)),
            "note": ("recent OOF tail sealed from variance fitting, "
                     "CDF calibration, and admission evidence"),
        }
    holdout = (calibration or {}).get("holdout") or {}
    if holdout:
        record["fit"]["holdout"] = holdout
    return _dump(path, record)
