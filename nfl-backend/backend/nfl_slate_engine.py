"""NFL slate-serve pricing module — per-game market probabilities for
SCHEDULED (undecided) board games + the run-line/totals market columns a
future dashboard consumes.

Research chain consumed (ALL pinned in committed records, never re-derived):
  era chain    — nfl_era_3e8c8a510f04.json: adopted arm E2 = per-side mu on
                 era-centered targets with ``ewm_2w`` centers; median rounds
                 home 20 / away 23; DN const-sigma 9.663 home / 9.0789 away,
                 rho 0.0076, tie calibrated 0.275%.
  market layer — nfl_market_3e8c8a510f04.json (13cb7ce): totals
                 ADOPT_SHRINK_TO_LINE with median-of-fold (c,d) =
                 (-0.3599, 0.3472) — the ONLY permitted market params.
  adoption      — nfl_adoption_decision_3e8c8a510f04.json: spread side
                 ADOPT_SHRINK_TO_LINE (median (c,d) = (0.446165, 0.307486)),
                 one feed decision governs BOTH sides.

What this module does at slate time (deterministic, no RNG):
  1. Fit-only refit of per-side mu_H/mu_A on ALL decided 2019-2025
     era-centered games at the era record's median rounds (20/23), same
     12-pool view, seeded deterministic (the era module's
     ``refit_centered_per_side`` — no engine edits).
  2. Era centers for each 2026 board row computed strictly-prior to that
     game's gameday from decided games available at run date (same-day
     excluded, leak-safe by the era module's invariant). This module carries
     its own day-recursion mirror (``board_era_centers``) because the board
     rows are UNDECIDED (no scores) and ``compute_centers`` requires a score
     per row — the mirror is pinned byte-identical to ``compute_centers`` on
     decided rows by tests, and skips undecided contributions (which can
     only sit at the timeline tail). The 56893d3 bias-calibration transform
     is NOT chained (pooled params ruled not wire-ready; era centers are the
     adopted mean correction).
  3. Joint params PINNED, never refit at slate time: DN, const sigma 9.663
     home / 9.0789 away, rho 0.0076, tie rate 0.275%. Refitting sigma on
     all-decided residuals would be in-sample for the refit mu models and
     would understate dispersion (the hot-totals defect). Conservative
     pinned dispersion is correct for a market product. A test asserts the
     slate builder uses exactly these constants.
  4. Per-game joint PMFs (76x76, mass conservation, tie calibrated) for
     every 2026 board row; 100% board coverage asserted.
  5. Market treatments per the feed decision: own-line quoting is the
     default mode (no known-vintage feed exists yet); shrink columns are
     computed and present but flagged ``shrink_applied=false`` until a feed
     with known vintage is wired. This is the conditional adoption, not a
     regression.

Outputs per board game (the honest-ECE market record contract):
  per-side score means (mu_H / mu_A), fair spread (median of margin PMF),
  fair total (median of total PMF), the offered line (nflreadpy schedule)
  WITH cover probabilities at that line, P(cover ±L) over the spread line
  grid, P(over/under/push at U) for integer totals, P(push at L) for
  integer spreads, raw P(±0.5 cover) per side AND derived ML
  P(H>A)/(1-P_tie) per side (the NFL-specific pair a future ±0.5 toggle
  needs), treatment-mode flags. Fair lines and offered lines are both
  present, never conflated.

Retention decision (recorded for the stale-artifact cleanup): the dated
markets/monitor artifacts are TRACKED-AND-ACCUMULATING like MLB's
(``run_engine_markets_*`` / ``run_engine_monitor_*``); they are committed
with this change and protected by the repo's tracked-file guard — the NFL
cleanup may never delete a committed file. Nothing here is wired into
master_pipeline; moneyline FEATURE_COLUMNS / served 12-pool / daily
pipeline untouched.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nfl_era_features import (CENTER_AWAY, CENTER_COLS, CENTER_HOME,
                              EWM_HALFLIFE_DAYS, NEUTRAL_CENTER,
                              compute_centers, refit_centered_per_side)
from nfl_joint_engine import (GRID_MAX, build_joint_pmfs, cover_prob,
                              margin_pmf_from_joint, over_prob,
                              total_pmf_from_joint)

# ── Pinned research constants (source records cited above) ──────────────────

# E2 adopted arm config (era record: spec ewm_2w, median rounds 20/23).
ERA_SPEC = "ewm_2w"
MEDIAN_ROUNDS = {"home": 20, "away": 23}

# Pinned joint params — DN, constant sigma, global rho, calibrated tie rate.
# NEVER refit at slate time (in-sample sigma on all-decided residuals would
# understate dispersion and re-create the hot totals).
PINNED_SIGMA_HOME = 9.663
PINNED_SIGMA_AWAY = 9.0789
PINNED_RHO = 0.0076
PINNED_P_TIE = 0.00275          # 3/1091 pooled-OOF empirical final-tie rate

# Market params (median-of-fold from the records) — the ONLY permitted
# shrink coefficients; no pooled re-fit at slate time.
TOTALS_CD = (-0.3599, 0.3472)   # totals: line + c + d*(mu_T - line)
SPREAD_CD = (0.446165, 0.307486)  # spread (adoption record): margin line
FEED_PRESENT = False            # no known-vintage feed yet -> own-line mode

# Offered-line + grid quoting constants.
SPREAD_INT_LINES = list(range(-14, 15))       # integer lines (P(margin > L))
TOTAL_INT_LINES = list(range(24, 67))         # integer totals (P(total > U))


def _fname(x: float) -> str:
    """Line-grid column tag: '-' -> 'm' and '.' -> '_' (MLB-style, so a
    dashboard joins columns by name without Python minus-sign issues); an
    integral line drops its trailing '.0' (integer grids)."""
    s = str(x)
    if s.endswith(".0"):
        s = s[:-2]
    return s.replace("-", "m").replace(".", "_")


def pinned_joint_params() -> dict[str, Any]:
    """The pinned joint params dict (identical shape to
    ``nfl_joint_engine.fit_joint_params`` output — fit_on pooled_oof so
    ``build_joint_pmfs``'s sealed leak guard accepts it). Tests pin the
    exact constants."""
    return {
        "family": "dn",
        "sigma_h": {"spec": "const", "sigma0": PINNED_SIGMA_HOME, "q": 0.0},
        "sigma_a": {"spec": "const", "sigma0": PINNED_SIGMA_AWAY, "q": 0.0},
        "rho": PINNED_RHO,
        "fit_on": "pooled_oof",
        "grid_max": GRID_MAX,
        "_pinned": True,
    }


# ── Board era centers (mirror of the era module's day recursion) ────────────

def board_era_centers(decided: pd.DataFrame, board: pd.DataFrame,
                      spec: str = ERA_SPEC) -> pd.DataFrame:
    """Leak-safe ewm centers for UNDECIDED board rows.

    Mirror of ``nfl_era_features._ewm_centers`` day-recursion for the spec's
    halflife: rows of ``decided`` (must carry home_score/away_score) and
    ``board`` (undecided — no scores) are interleaved by gameday; every row's
    center is the decayed weighted mean of strictly-prior DECIDED scores
    (same-day excluded). Board rows contribute nothing to the state (they are
    undecided) — safe because board rows postdate every decided game in this
    product's horizon. The recursion is pinned byte-identical to
    ``compute_centers`` on decided-only frames by tests (same decay, same
    day-level ordering, same NEUTRAL_CENTER fill, same 4-decimal rounding).

    Returns [game_id, era_center_home, era_center_away] for board rows.
    """
    if spec not in EWM_HALFLIFE_DAYS:
        raise ValueError(f"board_era_centers: spec {spec!r} not an EWM spec "
                         f"(expected one of {sorted(EWM_HALFLIFE_DAYS)})")
    hw_days = float(EWM_HALFLIFE_DAYS[spec])
    for name, df in (("decided", decided), ("board", board)):
        need = {"game_id", "gameday"}
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"board_era_centers: {name} missing {missing}")
    if "home_score" not in decided.columns or "away_score" not in decided.columns:
        raise ValueError("board_era_centers: decided needs home/away scores")

    d = decided.copy()
    b = board.copy()
    d["gameday"] = pd.to_datetime(d["gameday"], errors="coerce")
    b["gameday"] = pd.to_datetime(b["gameday"], errors="coerce")
    if d["gameday"].isna().any() or b["gameday"].isna().any():
        raise ValueError("board_era_centers: NaN gameday present")
    d["_kind"] = "decided"
    b["_kind"] = "board"
    for c in ("home_score", "away_score"):
        b[c] = np.nan
    all_rows = (pd.concat([d, b], ignore_index=True)
                .sort_values(["gameday", "game_id"]).reset_index(drop=True))
    hs = all_rows["home_score"].to_numpy(float)
    as_ = all_rows["away_score"].to_numpy(float)
    kind = all_rows["_kind"].to_numpy()
    day = all_rows["gameday"].dt.normalize()
    ords = (day - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    uniq, idx = np.unique(ords, return_inverse=True)
    out_h = np.full(len(all_rows), np.nan)
    out_a = np.full(len(all_rows), np.nan)
    dec = 0.5 ** (1.0 / hw_days)
    for s_arr, out in ((hs, out_h), (as_, out_a)):
        S = W = 0.0
        prev_o = None
        for i, o in enumerate(uniq):
            m = idx == i          # idx holds positions into uniq (not ords)
            out[m] = S / W if W > 0 else np.nan
            if prev_o is not None:
                diff = int(o - prev_o)
                S *= dec ** diff
                W *= dec ** diff
            day_kind = kind[m]
            contrib = np.isfinite(s_arr[m]) & (day_kind == "decided")
            if contrib.any():
                S += float(s_arr[m][contrib].sum())
                W += float(contrib.sum())
            prev_o = o

    out_h = np.where(np.isfinite(out_h), out_h, NEUTRAL_CENTER)
    out_a = np.where(np.isfinite(out_a), out_a, NEUTRAL_CENTER)
    out = pd.DataFrame({
        "game_id": all_rows["game_id"].values,
        CENTER_HOME: np.round(out_h, 4),
        CENTER_AWAY: np.round(out_a, 4),
    })
    board_ids = set(b["game_id"])
    out = out[out["game_id"].isin(board_ids)].reset_index(drop=True)
    if len(out) != len(b):
        raise RuntimeError("board_era_centers: board row coverage loss")
    return out


# ── Slate pricing ────────────────────────────────────────────────────────────

def price_board(refit: pd.DataFrame, params: dict[str, Any], p_tie: float,
                lines: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build per-board-game market rows from refit per-side means.

    Args:
        refit: board rows with game_id + pred_home/pred_away (from
            ``refit_centered_per_side``), in any order.
        params: pinned joint params (never fitted at slate time).
        p_tie: pinned tie rate.
        lines: optional offered-lines frame [game_id, spread_line,
            total_line] — when absent/NaN the offer columns are NaN and the
            fair lines (model) are still quoted (never conflated).

    Returns one row per board game with:
      identity: game_id, mu_H/mu_A means, margin/total fair lines (median of
                the PMFs), offered spread/total when available + cover/over
                at the offer, P(push) at integer offers,
      grid columns: p_home_cover_<±L> and p_push_<L> over SPREAD_INT_LINES,
                    p_over_<U> / p_under_<U> / p_push_<U> over
                    TOTAL_INT_LINES (all OWN-LINE quotes — the quoted mode),
      raw ±0.5 pair at the offered line (p_cover_minus_half / p_cover_plus_
      half per side) + derived ML P(H>A)/(1-P_tie) per side,
      shrink columns (additive, flagged): shrunk fair spread/total + cover/
      over/derived ML under BOTH records' median-of-fold (c,d), plus the
      treatment flags.
    """
    need = {"game_id", "pred_home", "pred_away"}
    missing = [c for c in need if c not in refit.columns]
    if missing:
        raise ValueError(f"price_board: refit missing {missing}")
    r = refit.copy()
    pmfs, summ = build_joint_pmfs(r[["game_id", "pred_home", "pred_away"]],
                                  params, p_tie)
    derived = summ["derived"].copy()
    marg = [margin_pmf_from_joint(J) for J in pmfs]
    tot = [total_pmf_from_joint(J) for J in pmfs]

    rows = derived.copy()
    rows["mu_h"] = np.round(r["pred_home"].to_numpy(float), 4)
    rows["mu_a"] = np.round(r["pred_away"].to_numpy(float), 4)
    rows["mu_margin"] = np.round(rows["mu_h"] - rows["mu_a"], 4)
    rows["mu_total"] = np.round(rows["mu_h"] + rows["mu_a"], 4)

    # Fair lines = discrete medians of the PMFs (model, no market blending).
    fair_spread = []
    fair_total = []
    for m_, t_ in zip(marg, tot):
        cdf_m = np.cumsum(m_)
        n = (len(m_) + 1) // 2
        fair_spread.append(int(np.searchsorted(cdf_m, 0.5)) - (n - 1))
        cdf_t = np.cumsum(t_)
        fair_total.append(int(np.searchsorted(cdf_t, 0.5)))
    rows["fair_spread"] = fair_spread
    rows["fair_total"] = fair_total

    if lines is not None and len(lines):
        rows = rows.merge(lines, on="game_id", how="left")
    else:
        rows["spread_line"] = np.nan
        rows["total_line"] = np.nan
    rows["has_offer"] = (rows["spread_line"].notna()
                         | rows["total_line"].notna()).astype(int)

    # ---- offer-level quotes (only when the offered line exists) ----
    p_cover_off = []
    p_push_off = []
    p_over_off = []
    p_under_off = []
    p_push_total_off = []
    p_cov_mh = []          # raw P(margin > L - 0.5)  (home, -0.5 leg)
    p_cov_ph = []          # raw P(margin > L + 0.5)  (home, +0.5 leg)
    p_cov_ma = []          # away mirror of the -0.5 leg (P(margin < L-0.5))
    p_cov_pa = []          # away mirror of the +0.5 leg
    for m_, t_, L0, U0 in zip(marg, tot,
                              rows["spread_line"].to_numpy(float),
                              rows["total_line"].to_numpy(float)):
        if np.isfinite(L0):
            p_cover_off.append(round(cover_prob(m_, L0), 6))
            p_push_off.append(round(m_[_margin_index(m_, L0)], 6)
                              if float(L0).is_integer() else np.nan)
            p_cov_ph.append(round(cover_prob(m_, L0 + 0.5), 6))
            p_cov_mh.append(round(cover_prob(m_, L0 - 0.5), 6))
            p_cov_pa.append(round(1.0 - p_cov_ph[-1], 6))
            p_cov_ma.append(round(1.0 - p_cov_mh[-1], 6))
        else:
            p_cover_off.append(np.nan)
            p_push_off.append(np.nan)
            p_cov_ph.append(np.nan)
            p_cov_mh.append(np.nan)
            p_cov_pa.append(np.nan)
            p_cov_ma.append(np.nan)
        if np.isfinite(U0):
            p_over_off.append(round(over_prob(t_, U0), 6))
            p_under_off.append(round(1.0 - over_prob(t_, U0)
                                     - _push_prob_total(t_, U0), 6))
            p_push_total_off.append(
                round(_push_prob_total(t_, U0), 6)
                if float(U0).is_integer() else np.nan)
        else:
            p_over_off.append(np.nan)
            p_under_off.append(np.nan)
            p_push_total_off.append(np.nan)
    rows["p_cover_offered"] = p_cover_off
    rows["p_push_offered"] = p_push_off
    rows["p_over_offered"] = p_over_off
    rows["p_under_offered"] = p_under_off
    rows["p_push_total_offered"] = p_push_total_off
    rows["p_home_cover_minus_half"] = p_cov_mh
    rows["p_home_cover_plus_half"] = p_cov_ph
    rows["p_away_cover_minus_half"] = p_cov_ma
    rows["p_away_cover_plus_half"] = p_cov_pa

    # ---- grid columns (own-line quotes; built as one frame to avoid
    # pandas fragmentation warnings on ~200 column inserts) ----
    grid: dict[str, list[float]] = {}
    for L in SPREAD_INT_LINES:
        grid[f"p_home_cover_{_fname(float(L))}"] = [
            round(cover_prob(m_, float(L)), 6) for m_ in marg]
        grid[f"p_push_{_fname(float(L))}"] = [
            round(m_[_margin_index(m_, float(L))], 6) for m_ in marg]
    for U in TOTAL_INT_LINES:
        grid[f"p_over_{_fname(float(U))}"] = [
            round(over_prob(t_, float(U)), 6) for t_ in tot]
        grid[f"p_under_{_fname(float(U))}"] = [
            round(1.0 - over_prob(t_, float(U))
                  - _push_prob_total(t_, float(U)), 6) for t_ in tot]
        grid[f"p_push_{_fname(float(U))}"] = [
            round(_push_prob_total(t_, float(U)), 6) for t_ in tot]
    rows = pd.concat([rows, pd.DataFrame(grid)], axis=1)

    # ---- derived ML per side (raw pair; the ±0.5-toggle product pair) ----
    rows["p_home_win_derived"] = rows["derived_ml"].round(6)
    rows["p_away_win_derived"] = (1.0 - rows["derived_ml"]).round(6)

    # ---- shrink columns (computed, additive, flagged not applied) ----
    shr = _shrink_arms(r[["game_id", "pred_home", "pred_away"]],
                       rows[["game_id", "spread_line", "total_line"]],
                       params, p_tie)
    rows = rows.merge(shr, on="game_id", how="left")
    rows["shrink_applied"] = 0
    rows["treatment_mode"] = ("own-line (no known-vintage feed); shrink "
                              "computed + flagged, never silently applied")
    return rows


def _margin_index(m_: np.ndarray, L: float) -> int:
    """Margin PMF index of a specific integer margin value L."""
    n = (len(m_) + 1) // 2
    return n - 1 + int(L)


def _push_prob_total(t_: np.ndarray, U: float) -> float:
    if not float(U).is_integer():
        return 0.0
    k = int(U)
    if k < 0 or k >= len(t_):
        return 0.0
    return float(t_[k])


def _shrink_arms(means: pd.DataFrame, offers: pd.DataFrame,
                 params: dict[str, Any], p_tie: float) -> pd.DataFrame:
    """Median-of-fold shrink arms (records' (c,d)) — margin-center (spread)
    and total-mean (totals) shifts composed, each delta applied delta/2."""
    mu_t = (means["pred_home"].to_numpy(float)
            + means["pred_away"].to_numpy(float))
    mu_m = (means["pred_home"].to_numpy(float)
            - means["pred_away"].to_numpy(float))
    L0 = offers["spread_line"].to_numpy(float)
    U0 = offers["total_line"].to_numpy(float)

    rows = means.copy()
    home = means["pred_home"].to_numpy(float).copy()
    away = means["pred_away"].to_numpy(float).copy()

    ct, dt = TOTALS_CD
    cs, ds = SPREAD_CD
    has_t = np.isfinite(U0)
    has_s = np.isfinite(L0)
    delta_t = np.where(has_t, U0 + ct + dt * (mu_t - U0) - mu_t, 0.0)
    delta_s = np.where(has_s, L0 + cs + ds * (mu_m - L0) - mu_m, 0.0)
    home = home + delta_t / 2.0 + delta_s / 2.0
    away = away + delta_t / 2.0 - delta_s / 2.0

    frame = pd.DataFrame({
        "game_id": means["game_id"].values,
        "pred_home": np.round(home, 4),
        "pred_away": np.round(away, 4),
    })
    pmfs, summ = build_joint_pmfs(frame, params, p_tie)
    derived = summ["derived"].copy()
    marg = [margin_pmf_from_joint(J) for J in pmfs]
    tot = [total_pmf_from_joint(J) for J in pmfs]
    out = pd.DataFrame({"game_id": means["game_id"].values,
                        "derived_ml_shrunk": derived["derived_ml"]})
    fair_s, fair_t = [], []
    for m_, t_ in zip(marg, tot):
        n = (len(m_) + 1) // 2
        fair_s.append(int(np.searchsorted(np.cumsum(m_), 0.5)) - (n - 1))
        fair_t.append(int(np.searchsorted(np.cumsum(t_), 0.5)))
    out["fair_spread_shrunk"] = fair_s
    out["fair_total_shrunk"] = fair_t
    pcov = [cover_prob(m_, float(L)) if np.isfinite(L) else np.nan
            for m_, L in zip(marg, L0)]
    pover = [over_prob(t_, float(U)) if np.isfinite(U) else np.nan
             for t_, U in zip(tot, U0)]
    out["p_cover_shrunk"] = [round(v, 6) if np.isfinite(v) else np.nan
                             for v in pcov]
    out["p_over_shrunk"] = [round(v, 6) if np.isfinite(v) else np.nan
                            for v in pover]
    return out
