"""Page 3 — Model Calibration.

Header, summary (today's record + upsets), KPI cards (AUC-ROC, Brier,
Log-Loss, Cal. Error), a merged confidence-vs-accuracy + calibration curve
(count bars + actual rate vs the Platt map vs the perfect-calibration
diagonal), and the reliability table with color-coded GAP values.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import inspect
import numpy as np
import pandas as pd
import streamlit as st

import moneyline_calibration as mlc
import utils

utils.inject_css()


def _render_nfl_calibration() -> None:
    """NFL Calibration page (step 3), sourced from the moneyline v1 record.

    Renders the aggregate calibration story the record carries — artifact
    header, ADOPT verdict banner, sealed KPIs, Sealed-vs-Pooled baselines,
    member performance + adaptive blend weights, and the post-hoc
    recalibration note — and GATES the per-1% favored-team curve, the
    reliability diagram, and the per-game history table behind per-game OOF
    data the shipped record does not yet carry (honest info lines when
    absent). Reuses the MLB per-1% curve builders verbatim.
    """
    rec = utils.load_nfl_moneyline_record()
    if not rec:
        st.warning("No NFL moneyline record found — install a "
                   "nfl_moneyline_v1_*.json under nfl-backend/data_delivery "
                   "and retry.")
        return

    pool = rec.get("pooled_preq_2021_2024") or {}
    sealed = rec.get("sealed_2025") or {}
    verdict = rec.get("verdict") or {}
    slate = rec.get("slate") or {}
    members = rec.get("members") or {}
    adaptive = rec.get("adaptive_weights") or {}
    n_games = pool.get("n") or 0

    def _ts_label(raw: str) -> str:
        """created_utc ISO → 'Month D, YYYY HH:MM ET'."""
        try:
            ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            et = ts.astimezone(ZoneInfo("America/New_York"))
            return f"{et.strftime('%B')} {et.day}, {et.year} {et.strftime('%H:%M')} ET"
        except (ValueError, TypeError):
            return str(raw) or "—"

    def _f(v, nd: int = 4) -> str:
        """Float formatting that never raises or prints None."""
        if v is None:
            return "—"
        try:
            return f"{float(v):.{nd}f}"
        except (TypeError, ValueError):
            return "—"

    def _kpi(label: str, value: str, color: str, cap: str) -> str:
        return (f'<div class="fb-kpi"><div class="label">{label}</div>'
                f'<div class="value" style="color:{color};">{value}</div>'
                f'<div class="cap">{cap}</div></div>')

    # -----------------------------------------------------------------
    # (1) Header — artifact date + pooled OOF n
    # -----------------------------------------------------------------
    st.markdown(
        "<div style='font-size:1.7rem;font-weight:800;color:#E2E8F0;'>Model Calibration Dashboard</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="display:inline-flex;align-items:center;gap:6px;margin:6px 0 2px;color:#94A3B8;
                    border:1px solid #1E293B;border-radius:999px;padding:3px 12px;font-size:0.85rem;">
          🏈 NFL · Record {_ts_label(rec.get('created_utc', ''))} · n = {n_games:,} pooled OOF games
        </div>
        <div style="color:#94A3B8;font-size:0.9rem;margin-top:4px;">
          Calibration of the NFL moneyline ensemble on a sealed 2025 holdout and pooled 2021–2024 prequential OOF
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------
    # (2) Verdict banner — ADOPT (green) vs NOT ADOPTED
    # -----------------------------------------------------------------
    adopt = bool(verdict.get("adopt"))
    _bg = "rgba(16,185,129,.12)" if adopt else "rgba(250,204,21,.10)"
    _bdr = "rgba(16,185,129,.45)" if adopt else "rgba(250,204,21,.5)"
    _fg = "#34D399" if adopt else "#FDE047"
    _badge = "✅ ADOPT" if adopt else "⚠ NOT ADOPTED"
    _season = slate.get("season") or "—"
    _week = slate.get("week") or "—"
    _smodels = slate.get("model") or "—"
    _reasons = " ".join(str(r) for r in (verdict.get("reasons") or []))
    if _reasons:
        _gate_note = (f'<div style="color:#64748B;font-size:0.82rem;margin-top:6px;">'
                      f'{_reasons}</div>')
    else:
        _state = "all gate conditions satisfied" if adopt else "not approved"
        _gate_note = (f'<div style="color:#64748B;font-size:0.82rem;margin-top:6px;">'
                      f'Gate: sealed beats elo · sealed beats constant · '
                      f'ECE&nbsp;≤&nbsp;0.08 · pooled/sealed inversion — {_state}</div>')
    st.markdown(
        f"""
        <div class="fb-box" style="margin:14px 0;padding:12px 18px;">
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;color:#E2E8F0;">
            <span style="background:{_bg};border:1px solid {_bdr};color:{_fg};border-radius:999px;
                         padding:4px 14px;font-weight:800;">{_badge}</span>
            <span style="font-weight:800;">moneyline ensemble</span>
            <span style="color:#94A3B8;font-size:0.9rem;">
              slate {_season} · week {_week} · {int(slate.get('n_games') or 0)} games · {_smodels}
            </span>
          </div>
          {_gate_note}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------
    # (3) KPI cards — sealed_2025.model_platt (logloss / auc / ece)
    # -----------------------------------------------------------------
    sp = sealed.get("model_platt") or {}
    kpi_specs = [
        ("LOG-LOSS", _f(sp.get("logloss")), "#FBBF24",
         "Sealed-2025 Platt · lower better"),
        ("AUC-ROC", _f(sp.get("auc")), utils.BLUE,
         "Sealed-2025 discrimination"),
        ("CAL. ERROR (ECE)", _f(sp.get("ece")), "#F472B6",
         "Sealed-2025 · after Platt"),
    ]
    kcols = st.columns(3)
    for col, (label, value, color, cap) in zip(kcols, kpi_specs):
        with col:
            st.markdown(_kpi(label, value, color, cap), unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # (4) Sealed-2025 vs Pooled-2021-2024 baseline table
    # -----------------------------------------------------------------
    st.markdown("### Sealed-2025 vs Pooled-2021-2024")
    baseline_rows = [
        ("constant_home_edge", "Constant home edge"),
        ("elo_logistic", "Elo-logistic baseline"),
        ("model_raw", "Ensemble (raw blend)"),
        ("model_platt", "Ensemble (Platt calibrated)"),
    ]

    def _arm(dic: dict, key: str):
        sub = dic.get(key)
        return {"logloss": (None if not sub else sub.get("logloss")),
                "auc": (None if not sub else sub.get("auc")),
                "ece": (None if not sub else sub.get("ece"))}

    rows_html = []
    for key, label in baseline_rows:
        s = _arm(sealed, key)
        p = _arm(pool, key)
        rows_html.append(
            f"<tr><td>{label}</td>"
            f"<td>{_f(s['logloss'])}</td><td>{_f(s['auc'])}</td><td>{_f(s['ece'])}</td>"
            f"<td>{_f(p['logloss'])}</td><td>{_f(p['auc'])}</td><td>{_f(p['ece'])}</td></tr>"
        )
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <table class="fb-table">
            <thead><tr><th>SIGNAL</th>
              <th class="col-grp" colspan="3">SEALED 2025 (n={int(sealed.get('n') or 0)})</th>
              <th class="col-grp" colspan="3">POOLED 2021–2024 (n={int(n_games)})</th></tr>
              <tr><th></th><th>LL</th><th>AUC</th><th>ECE</th><th>LL</th><th>AUC</th><th>ECE</th></tr></thead>
            <tbody>{''.join(rows_html)}</tbody>
          </table>
        </div>
        <div style="color:#64748B;font-size:0.78rem;margin-top:6px;">
          model_raw = the raw ensemble blend; model_platt = the deployed Platt-calibrated map. Constant home edge carries no ECE (single rate). ECE = expected calibration error over the arm's probability bins.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------
    # (5) Member performance + adaptive blend weights
    # -----------------------------------------------------------------
    st.markdown("### Ensemble Members — Weighted Blend")
    member_order = ["xgboost", "lightgbm", "logistic", "randomforest", "mlp"]
    avail = [m for m in member_order if m in members]
    m_rows = []
    for m in avail:
        md = members[m] or {}
        base_w = md.get("weight")
        adapt_w = adaptive.get(m)
        m_rows.append(
            f"<tr><td>{m}</td>"
            f"<td>{_f(base_w, 4)}</td><td>{_f(adapt_w, 4)}</td>"
            f"<td>{_f(md.get('logloss'))}</td><td>{_f(md.get('auc'))}</td>"
            f"<td>{_f(md.get('ece'))}</td><td>{_f(md.get('brier'))}</td></tr>"
        )
    _aw_sum = sum(float(v) for v in adaptive.values()
                  if isinstance(v, (int, float)))
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <table class="fb-table">
            <thead><tr><th>MODEL</th><th>BASE WEIGHT</th><th>ADAPTIVE WGT</th>
              <th>LOGLOSS</th><th>AUC</th><th>ECE</th><th>BRIER</th></tr></thead>
            <tbody>{''.join(m_rows)}</tbody>
          </table>
        </div>
        <div style="color:#64748B;font-size:0.78rem;margin-top:6px;">
          Base weight = prior blend weight; Adaptive wgt = pooled-OOF AUC-derived re-blend weights
          (sum = {_aw_sum:.4f}) that drive the deployed bundle. Member metrics are pooled prequential OOF (2019–2024, warm-up excluded).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------
    # (6) Post-hoc recalibration note
    # -----------------------------------------------------------------
    _ece_cal = _f(sp.get("ece"))
    st.markdown(
        f"""
        <div class="fb-box" style="margin:12px 0;padding:12px 18px;">
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;color:#E2E8F0;">
            <span style="font-weight:700;">Post-Hoc Recalibration:</span>
            <span style="background:rgba(59,130,246,.18);color:#60A5FA;border-radius:999px;padding:2px 12px;font-size:0.82rem;font-weight:700;">Platt scaling</span>
            <span style="color:#94A3B8;font-size:0.9rem;">fitted on pooled pre-holdout OOF predictions only — never leaked into the sealed 2025 fit</span>
            <span style="background:rgba(16,185,129,.15);color:#34D399;border-radius:999px;padding:2px 12px;font-size:0.82rem;font-weight:700;">
              calibrated ECE = {_ece_cal}
            </span>
          </div>
          <div style="color:#64748B;font-size:0.8rem;margin-top:6px;">
            The published (deployed) probability is the Platt map applied to the adaptive-weighted blend.
            The calibrated ECE to read is therefore <b>sealed_2025.model_platt.ece = {_ece_cal}</b>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =================================================================
    # CONDITIONAL SECTIONS — require per-game OOF / binned data the v1
    # schedule-only record does not yet carry. Each gates on its input.
    # =================================================================
    hist_curve = utils.load_nfl_prediction_history()
    pts = mlc.favored_calibration_pts(hist_curve)

    # --- (a) per-1% favored-team calibration curve ---
    st.markdown("### Calibration Curve — Favored Team")
    if pts.empty:
        st.info("Per-1% favored-team calibration curve ships when the backend emits "
                "per-game OOF history (predicted probability + outcome pairs) into "
                "the NFL moneyline record. The shipped v1 record carries only "
                "aggregate pooled/sealed metrics, so no curve points are available yet.")
    else:
        pts_cal = pd.DataFrame()  # no Platt a/b in the record → no green map overlay
        built = mlc.chart_favored_calibration(pts, pts_cal)
        utils.show_chart(built["chart"])
        st.caption(
            f"Favored-team view (n={len(pts)} 1% bins) · each OOF prediction taken from "
            "the favored side (≥ 50%) and binned to the nearest 1% · blue = actual "
            "win rate, count bars = games per bin"
        )

    # --- (b) reliability diagram ---
    st.markdown("### Reliability Diagram — Binned Data")
    buckets = rec.get("calibration_buckets") or rec.get("reliability_buckets") or []
    if not buckets:
        st.info("Reliability diagram ships when the record emits binned prequential "
                "buckets (per-bin mean predicted vs mean actual). The v1 record "
                "carries aggregate metrics only.")
    else:
        b_rows = []
        for b in buckets:
            if not isinstance(b, dict):
                continue
            mp = _f(b.get("mean_predicted"), 3)
            ma = _f(b.get("mean_actual"), 3)
            try:
                gap = float(b.get("gap"))
                gap_txt = f"{gap:+.3f}"
                gap_color = utils.PRIMARY if gap > 0 else utils.RED
            except (TypeError, ValueError, AttributeError):
                gap_txt, gap_color = "—", utils.SLATE
            b_rows.append(
                f"<tr><td>{b.get('bucket', '')}</td><td>{mp}</td><td>{ma}</td>"
                f"<td>{int(b.get('count') or 0)}</td>"
                f"<td style='color:{gap_color};font-weight:700;'>{gap_txt}</td></tr>"
            )
        st.markdown(
            f"""
            <div class="fb-box" style="padding:6px 8px;">
              <table class="fb-table">
                <thead><tr><th>BUCKET</th><th>MEAN PREDICTED</th><th>MEAN ACTUAL</th>
                <th>COUNT</th><th>GAP</th></tr></thead>
                <tbody>{''.join(b_rows)}</tbody>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- (c) per-game prediction-history table ---
    st.markdown("### Prediction History — Every Game")
    if hist_curve is None or hist_curve.empty or "home_win_prob_model" not in hist_curve.columns:
        st.info("No per-game prediction history yet — generated when the backend writes "
                "per-game OOF outcomes into the moneyline record (the v1 record is a "
                "schedule-only slate).")
    else:
        h = hist_curve.sort_values("game_date", ascending=False)
        rows = []
        for _, r in h.iterrows():
            ok = bool(r.get("correct"))
            ph = pd.to_numeric(pd.Series([r.get("home_win_prob_model")]),
                               errors="coerce").iloc[0]
            pick_prob = ph if r.get("model_pick") == r.get("home_team") else (
                1 - ph if pd.notna(ph) else None)
            res = (f"<td style='color:{utils.PRIMARY};font-weight:700;'>✓</td>"
                   if ok else f"<td style='color:{utils.RED};font-weight:700;'>✗</td>")
            rows.append(
                f"<tr><td>{r.get('game_date','')}</td>"
                f"<td>{r.get('away_team','')} @ {r.get('home_team','')}</td>"
                f"<td>{r.get('model_pick','')} ({pick_prob:.0%})</td>"
                f"<td>{r.get('actual_winner','')}</td>{res}</tr>"
            )
        st.markdown(
            f"""
            <div class="fb-box" style="padding:6px 8px;">
              <table class="fb-table">
                <thead><tr><th>DATE</th><th>MATCHUP</th><th>MODEL PICK</th>
                <th>WINNER</th><th>RESULT</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _nfl_calibration_page() -> None:
    """Dispatch helper: render the NFL page, then halt so the MLB block below
    does not execute for the NFL sport."""
    _render_nfl_calibration()
    st.stop()


if utils.get_sport() == "nfl":
    _nfl_calibration_page()

dates = utils.available_dates(**utils.get_source_config())
# Always show the most recent run (like Power Rankings / Model Monitor):
# ignore the date picked on Today's Games so the tab never drills into a
# past day's small per-day slice.
date_str = dates[0] if dates else "20260809"
if "use_daily" in inspect.signature(utils.load_calibration).parameters:
    cal = utils.load_calibration(date_str, use_daily=False)
else:
    # Deployed utils.py may predate the use_daily param (stale snapshot):
    # fall back to the plain call — date pinning alone still yields the
    # latest pooled view for current artifacts.
    cal = utils.load_calibration(date_str)
if not cal:
    st.warning(f"No calibration artifacts found for {date_str} or any recent date.")
    st.stop()

artifact_date = cal.get("_artifact_date", date_str)
n_games = cal.get("n_games", 0)
kpis = cal.get("kpis", {})
curve = cal.get("calibration_curve", [])
record = cal.get("today_record", {})
upsets = cal.get("upsets", [])


def _trained_label(raw: str) -> str:
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        et = ts.astimezone(ZoneInfo("America/New_York"))
        return f"{et.strftime('%B')} {et.day}, {et.year} {et.strftime('%H:%M')} ET"
    except (ValueError, TypeError):
        return raw or "—"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("<div style='font-size:1.7rem;font-weight:800;color:#E2E8F0;'>Model Calibration Dashboard</div>",
            unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="display:inline-flex;align-items:center;gap:6px;margin:6px 0 2px;color:#94A3B8;
                border:1px solid #1E293B;border-radius:999px;padding:3px 12px;font-size:0.85rem;">
      As of {utils.format_date_long(artifact_date)} · n = {n_games:,} games · Trained {_trained_label(cal.get('trained_at', ''))}
    </div>
    {f'<div style="color:#64748B;font-size:0.82rem;margin-top:2px;">ℹ No artifact for {utils.format_date_long(date_str)} — showing latest snapshot ({utils.format_date_long(artifact_date)})</div>' if artifact_date != date_str else ''}
    <div style="color:#94A3B8;font-size:0.9rem;margin-top:4px;">
      Assessing prediction reliability and accuracy across probability buckets
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Summary card
# ---------------------------------------------------------------------------
wins, losses = record.get("wins", 0), record.get("losses", 0)
completed = record.get("completed", wins + losses)
acc = (wins / completed * 100) if completed else 0.0
upset_text = " · ".join(f"{u['team']} {u['prob']:.0%} upset" for u in upsets) or "No upsets today"
st.markdown(
    f"""
    <div class="fb-box" style="margin:14px 0;padding:14px 18px;">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;color:#E2E8F0;">
        <span style="font-weight:700;">Today's Record:</span>
        <span style="background:rgba(16,185,129,.18);color:#34D399;border-radius:999px;padding:2px 12px;font-weight:800;">✓ {wins}-{losses}</span>
        <span style="color:#94A3B8;font-size:0.9rem;">{completed} completed games · {wins} correct picks ({acc:.1f}%) · {len(upsets)} upsets</span>
      </div>
      <div style="margin-top:8px;">
        <span style="background:rgba(245,158,11,.18);color:#FBBF24;border-radius:999px;padding:2px 12px;font-size:0.82rem;font-weight:700;">⚡ {upset_text}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
kpi_specs = [
    ("AUC-ROC", kpis.get("auc_roc", "—"), utils.BLUE, "Discrimination"),
    ("BRIER SCORE", _brier_disp := (
        f"{kpis['brier_score']} → {kpis['brier_calibrated']}"
        if kpis.get("brier_calibrated") is not None else kpis.get("brier_score", "—")
     ), utils.PRIMARY, "Lower is better" + (" · after calibration" if kpis.get("brier_calibrated") is not None else "")),
    ("LOG-LOSS", _ll_disp := (
        f"{kpis['log_loss']} → {kpis['log_loss_calibrated']}"
        if kpis.get("log_loss_calibrated") is not None else kpis.get("log_loss", "—")
     ), "#FBBF24", "Penalizes confidence" + (" · after calibration" if kpis.get("log_loss_calibrated") is not None else "")),
    ("CAL. ERROR", _ece_disp := (
        f"{kpis['cal_error']} → {kpis['cal_error_calibrated']}"
        if kpis.get("cal_error_calibrated") is not None else kpis.get("cal_error", "—")
     ), "#F472B6", "ECE raw → calibrated" if kpis.get("cal_error_calibrated") is not None else "ECE metric"),
]
kcols = st.columns(4)
for col, (label, value, color, cap) in zip(kcols, kpi_specs):
    with col:
        st.markdown(
            f'<div class="fb-kpi"><div class="label">{label}</div>'
            f'<div class="value" style="color:{color};">{value}</div>'
            f'<div class="cap">{cap}</div></div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Post-hoc recalibration banner (raw vs calibrated)
# ---------------------------------------------------------------------------
cal_sec = cal.get("calibration") or {}
if cal_sec.get("method") == "platt":
    _mr = cal_sec.get("metrics_raw") or {}
    _mc = cal_sec.get("metrics_calibrated") or {}
    _params = cal_sec.get("params") or {}
    _ece_raw = _mr.get("ece")
    _ece_cal = _mc.get("ece")
    if _ece_raw is not None and _ece_cal is not None:
        _delta = (_ece_raw - _ece_cal) * 100
        _arrow = "🟢" if _delta >= 0 else "🔴"
        st.markdown(
            f"""
            <div class="fb-box" style="margin:12px 0;padding:12px 18px;">
              <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;color:#E2E8F0;">
                <span style="font-weight:700;">Post-Hoc Recalibration:</span>
                <span style="background:rgba(59,130,246,.18);color:#60A5FA;border-radius:999px;padding:2px 12px;font-size:0.82rem;font-weight:700;">
                  Platt scaling · a={_params.get('a', '—')}, b={_params.get('b', '—')}
                </span>
                <span style="color:#94A3B8;font-size:0.9rem;">
                  fitted on {int(_params.get('n', 0) or 0):,} out-of-sample games
                </span>
                <span style="background:rgba(16,185,129,.15);color:#34D399;border-radius:999px;padding:2px 12px;font-size:0.82rem;font-weight:700;">
                  {_arrow} ECE {_ece_raw:.4f} → {_ece_cal:.4f} ({_delta:+.2f} pts)
                </span>
              </div>
              <div style="color:#64748B;font-size:0.8rem;margin-top:6px;">
                Published probabilities are corrected after blending: p<sub>cal</sub> = σ(a·logit(p) + b).
                Fitted only on out-of-fold predictions — each evaluation fold is scored by a map trained strictly on prior folds.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------
st.markdown("### Calibration Curve — Favored Team")

# Per-1%-probability calibration, built from the game-level prediction
# history: each OOF prediction is taken from the FAVORED team's side
# (probability >= 50%), binned to the nearest 1%; each 1% slice yields one
# calibration point (win_rate) AND one count bar (n) from the same frame.
hist_curve = utils.load_prediction_history(date_str)
pts = mlc.favored_calibration_pts(hist_curve)

# Green curve on the SAME RAW AXIS: the DEPLOYED Platt calibration map
# σ(a·logit(p)+b) evaluated at every raw favored probability. Because it
# is a single fitted monotone function (a > 0), the green line is strictly
# monotone by construction — unlike binned averages of the stored per-fold
# calibrated values, which mix calibrators fitted at different times.
# Vertical gap between blue (actual win rate) and green (what the
# calibrated model predicts) at a given raw x = the correction applied.
pts_cal = pd.DataFrame()
_params = cal_sec.get("params") or {}
try:
    _a = float(_params.get("a"))
    _b = float(_params.get("b"))
    _xs = np.arange(0.50, 1.0, 0.005)   # logit(p) undefined at p = 1.0
    _z = _a * np.log(_xs / (1.0 - _xs)) + _b
    _sigma = 1.0 / (1.0 + np.exp(-_z))
    # Favored-side convention mirrors the pipeline: max(p_cal, 1 - p_cal).
    pts_cal = pd.DataFrame({
        "prob": _xs,
        "cal_mean": np.maximum(_sigma, 1.0 - _sigma),
        "n": 0,
    })
    # Per-1%-bin game counts from history, so hover shows sample size.
    if hist_curve is not None and not hist_curve.empty \
            and "home_win_prob_model" in hist_curve.columns:
        _p0 = pd.to_numeric(hist_curve["home_win_prob_model"], errors="coerce").dropna()
        _raw0 = np.maximum(_p0.values, 1.0 - _p0.values)
        _cnt = pd.Series(np.round(_raw0 * 100).astype(int)).value_counts()
        pts_cal["n"] = pts_cal["prob"].map(
            lambda x: int(_cnt.get(int(round(x * 100)), 0))
        )
except (TypeError, ValueError, ZeroDivisionError):
    pts_cal = pd.DataFrame()

# Bucketed curve from the artifact (also feeds the reliability table below).
curve_df = pd.DataFrame(curve) if curve else pd.DataFrame()

if pts.empty:
    # Fallback: 10-point bucket curve from the calibration artifact
    if curve_df.empty:
        st.info("No calibration curve data available.")
    else:
        pts = curve_df.rename(columns={
            "mean_predicted": "prob", "mean_actual": "win_rate", "count": "n",
        })[["prob", "win_rate", "n"]]

if not pts.empty:
    # Merged confidence-vs-accuracy + calibration curve: count bars (LEFT
    # 'Games' axis) + blue actual-rate curve / green Platt map (RIGHT '%'
    # axis, independent) + gray dashed perfect-calibration diagonal. Bars and
    # the blue curve come from the SAME filled 1% bins, so bar height = games
    # in that confidence bucket and the curve = their accuracy — one chart, no
    # information lost from the former standalone 'Prediction Confidence &
    # Accuracy' section.
    built = mlc.chart_favored_calibration(pts, pts_cal)
    legend_extra = ""
    if not pts_cal.empty:
        legend_extra = (" · Green dashed: deployed Platt calibration map "
                        "(vertical gap = correction applied at that raw probability)")
    utils.show_chart(built["chart"])
    st.caption(
        f"Model (n={n_games:,}) · Count bars (left 'Games' axis): games per "
        f"1% predicted-probability bin — the bars are the confidence-vs-"
        f"accuracy view, bar height = how many games the model priced in that "
        f"confidence band and the blue curve = how often those games won · "
        f"Blue: actual win rate at each raw probability · "
        f"Green: calibrated probability σ(a·logit(p)+b) at each raw probability · "
        f"Perfect Calibration (dashed diagonal)"
        f"{legend_extra} · each game counted once from the favored side; "
        "blue curve binned to the nearest 1% — hover for games per point"
    )

# ---------------------------------------------------------------------------
# Reliability table
# ---------------------------------------------------------------------------
st.markdown("### Reliability Diagram — Binned Data")
# Prequential calibrated buckets (each point corrected by a map fitted on
# strictly PRIOR folds) shown alongside the raw view, so overconfidence can
# be judged at BOTH stages of the deployed chain.
_cal_buckets = {
    b.get("bucket"): b
    for b in ((cal.get("calibration") or {}).get("calibration_buckets_calibrated") or [])
}
if curve_df.empty:
    st.info("No reliability data available.")
else:
    rows = []
    for _, r in curve_df.iterrows():
        gap = r["gap"]
        gap_color = utils.PRIMARY if gap > 0 else utils.RED
        gap_txt = f"{gap:+.3f}"
        _cb = _cal_buckets.get(r["bucket"])
        cal_cell = (
            f"<td style='color:#34D399;'>{_cb['mean_predicted']:.3f}</td>"
            if _cb else "<td style='color:#475569;'>—</td>"
        )
        rows.append(
            f"<tr><td>{r['bucket']}</td><td>{r['mean_predicted']:.3f}</td>"
            f"{cal_cell}"
            f"<td>{r['mean_actual']:.3f}</td><td>{int(r['count'])}</td>"
            f"<td style='color:{gap_color};font-weight:700;'>{gap_txt}</td></tr>"
        )
    # TOTAL row: overall win rate across ALL predictions, count-weighted
    n_tot = int(curve_df["count"].sum())
    if n_tot > 0:
        mp_tot = float((curve_df["mean_predicted"] * curve_df["count"]).sum() / n_tot)
        ma_tot = float((curve_df["mean_actual"] * curve_df["count"]).sum() / n_tot)
        gap_tot = mp_tot - ma_tot
        tot_color = utils.PRIMARY if gap_tot > 0 else utils.RED
        rows.append(
            f"<tr style='border-top:2px solid #334155;font-weight:700;'><td>TOTAL</td>"
            f"<td>{mp_tot:.3f}</td><td style='color:#64748B;'>—</td>"
            f"<td>{ma_tot:.3f}</td><td>{n_tot}</td>"
            f"<td style='color:{tot_color};font-weight:700;'>{gap_tot:+.3f}</td></tr>"
        )
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <table class="fb-table">
            <thead><tr><th>BUCKET</th><th>MEAN PREDICTED (RAW)</th><th>CALIBRATED</th>
            <th>MEAN ACTUAL</th><th>COUNT</th><th>GAP (RAW)</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        <div style="color:#64748B;font-size:0.78rem;margin-top:6px;">
          Favored-team view: every game counted once at its pick probability (≥ 50%). GAP = mean predicted − mean actual. Green: overconfident (positive). Red: underconfident (negative).
          CALIBRATED = prequential Platt-corrected prediction per bucket — each game corrected by a map fitted only on prior games, the same convention as deployment.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Game-level history: every walk-forward prediction vs its actual result
# ---------------------------------------------------------------------------
st.markdown("### Prediction History — Every Game")
hist = utils.load_prediction_history(date_str)
if hist is None or hist.empty or "home_win_prob_model" not in hist.columns:
    st.info("No per-game prediction history available yet (generated on the next pipeline run).")
else:
    h = hist.copy()
    h["_date"] = pd.to_datetime(h["game_date"], errors="coerce")
    lo, hi = h["_date"].min().date(), h["_date"].max().date()

    fc1, fc2, _ = st.columns([1, 1, 2])
    start_d = fc1.date_input("Start date", value=lo, min_value=lo, max_value=hi)
    end_d = fc2.date_input("End date", value=hi, min_value=lo, max_value=hi)
    if start_d > end_d:
        start_d, end_d = end_d, start_d

    in_range = h[(h["_date"].dt.date >= start_d) & (h["_date"].dt.date <= end_d)]
    view = in_range.sort_values("_date", ascending=False)
    n_rng = len(view)
    if n_rng == 0:
        st.info("No games in the selected date range.")
    else:
        acc_rng = float(pd.to_numeric(view["correct"], errors="coerce").mean() * 100)
        # Display probabilities through the DEPLOYED Platt map σ(a·logit(p)+b)
        # so MODEL PICK % matches Today's Games and the green calibration
        # curve exactly. Picks are unchanged: the map is monotone increasing,
        # so argmax(raw) == argmax(calibrated).
        _cal_sec_h = cal.get("calibration") or {}
        _p_disp = pd.to_numeric(view["home_win_prob_model"], errors="coerce")
        try:
            if _cal_sec_h.get("method") == "platt":
                _ah = float((_cal_sec_h.get("params") or {}).get("a"))
                _bh = float((_cal_sec_h.get("params") or {}).get("b"))
                _pc = _p_disp.clip(1e-6, 1 - 1e-6)
                _z = _ah * np.log(_pc / (1 - _pc)) + _bh
                _p_disp = 1.0 / (1.0 + np.exp(-_z))
                _cal_note = " · probabilities are post-calibration σ(a·logit(p)+b)"
            else:
                _cal_note = ""
        except (TypeError, ValueError):
            _cal_note = ""
        st.caption(
            f"{n_rng:,} games · {acc_rng:.1f}% picks correct · most recent first — "
            "scroll for older results" + _cal_note
        )
        rows = []
        for _, r in view.iterrows():
            ok = pd.to_numeric(pd.Series([r.get("correct")]), errors="coerce").iloc[0]
            if pd.isna(ok):
                res = "<td>—</td>"
            elif bool(ok):
                res = f"<td style='color:{utils.PRIMARY};font-weight:700;'>✓</td>"
            else:
                res = f"<td style='color:{utils.RED};font-weight:700;'>✗</td>"
            prob = _p_disp.loc[r.name] if r.name in _p_disp.index else r.get("home_win_prob_model")
            pick_prob = prob if str(r.get("model_pick")) == str(r.get("home_team")) else 1 - prob
            score = "—"
            hs, asc = r.get("home_score"), r.get("away_score")
            if pd.notna(hs) and pd.notna(asc):
                score = f"{int(asc)}–{int(hs)}"
            rows.append(
                f"<tr><td>{r['_date'].strftime('%b %d, %Y')}</td>"
                f"<td>{r.get('away_team','')} @ {r.get('home_team','')}</td>"
                f"<td>{score}</td>"
                f"<td>{r.get('model_pick','')} ({pick_prob:.0%})</td>"
                f"<td>{r.get('actual_winner','')}</td>{res}</tr>"
            )
        st.markdown(
            f"""
            <div class="fb-box" style="padding:6px 8px;">
              <div style="max-height:480px;overflow-y:auto;">
                <table class="fb-table">
                  <thead><tr><th>DATE</th><th>MATCHUP</th><th>SCORE (A–H)</th>
                  <th>MODEL PICK</th><th>WINNER</th><th>RESULT</th></tr></thead>
                  <tbody>{''.join(rows)}</tbody>
                </table>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
