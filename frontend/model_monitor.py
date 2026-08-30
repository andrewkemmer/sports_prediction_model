"""Page 4 — Model & Data Drift Monitor.

Health cards (last retrain / next retrain / drift alerts), the upset
monitoring callout, the feature-drift (PSI) matrix with status pills, the
rolling 30-day Brier timeline vs a fixed baseline, and the model version
history table.
"""

from __future__ import annotations

import html
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

import utils

utils.inject_css()

if utils.get_sport() == "nfl":
    st.info("🏈 NFL Model & Data Drift Monitor arrives with step 3 — the "
            "FEATURE v1 admission/coverage record is a candidate for a "
            "sport-specific monitor, shipping with the full conditional UI.")
    st.stop()

dates = utils.available_dates(**utils.get_source_config())
date_str = st.session_state.get("selected_date", dates[0] if dates else "20260809")
mon = utils.load_model_monitor(date_str)
if not mon:
    st.warning(f"No model monitor artifacts found for {date_str}.")
    st.stop()


def _fmt_date(raw: str) -> str:
    try:
        d = date.fromisoformat(str(raw)[:10])
        return f"{d.strftime('%b')} {d.day}, {d.year}"
    except (ValueError, TypeError):
        return str(raw) or "—"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("<div style='font-size:1.7rem;font-weight:800;color:#E2E8F0;'>Model & Data Drift Monitor</div>",
            unsafe_allow_html=True)
st.markdown("<div style='color:#94A3B8;margin:2px 0 14px;'>Tracking model health, feature drift, and performance over time</div>",
            unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Top alert boxes
# ---------------------------------------------------------------------------
last_retrained = mon.get("last_retrained", "")
next_retrain = mon.get("next_retrain", "")
# Backend persists drift rows under "feature_drift"; older artifacts used
# "drift_alerts". Derive from whichever exists.
drift_rows_all = mon.get("feature_drift", []) or []
drift_alerts = mon.get("drift_alerts") or [
    r for r in drift_rows_all if r.get("status") in ("WARN", "ALERT")
]

try:
    days_since = (utils.parse_date(date_str) - date.fromisoformat(last_retrained[:10])).days
except (ValueError, TypeError):
    days_since = 0
try:
    days_until = (date.fromisoformat(next_retrain[:10]) - utils.parse_date(date_str)).days
except (ValueError, TypeError):
    days_until = 0

next_note = mon.get("next_retrain_note", "")
if "tonight" not in next_note and days_until <= 1:
    next_note = f"{next_note} — tonight" if next_note else "tonight"
last_note = mon.get("last_retrained_note", "Model healthy")
if "ago" not in last_note:
    last_note = f"{last_note} — {days_since} days ago"

n_warn = sum(1 for a in drift_alerts if a.get("status") == "WARN")
n_alert = sum(1 for a in drift_alerts if a.get("status") == "ALERT")
drift_value = "No drift" if not drift_alerts else (
    f"{n_warn + n_alert} Alert" if n_alert else f"{n_warn} Warning"
)
drift_sub = "—" if not drift_alerts else (
    f"{drift_alerts[0].get('feature', '')} — elevated PSI"
)

boxes = [
    ("LAST RETRAIN", _fmt_date(last_retrained), last_note, utils.PRIMARY),
    ("NEXT RETRAIN", _fmt_date(next_retrain), next_note, utils.BLUE),
    ("DRIFT ALERTS", drift_value, drift_sub, utils.AMBER),
]
bcols = st.columns(3)
for col, (label, value, sub, dot) in zip(bcols, boxes):
    with col:
        st.markdown(
            f"""
            <div class="fb-box" style="height:100%;">
              <div style="color:#94A3B8;font-size:0.72rem;font-weight:700;letter-spacing:1px;">
                <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{dot};margin-right:6px;"></span>{label}
              </div>
              <div style="font-size:1.35rem;font-weight:800;color:#E2E8F0;margin:4px 0 2px;">{value}</div>
              <div style="color:#94A3B8;font-size:0.82rem;">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Upset monitoring note
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="border:1px solid rgba(245,158,11,.55);background:rgba(245,158,11,.06);border-radius:12px;
                padding:12px 16px;margin:14px 0;">
      <div style="color:#FBBF24;font-weight:800;">Upset Monitoring Note — {utils.format_date_short(date_str)}</div>
      <div style="color:#E2E8F0;font-size:0.92rem;margin-top:4px;">{mon.get('upset_note', 'No note available.')}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Feature drift matrix
# ---------------------------------------------------------------------------
st.markdown("### Feature Drift Analysis (PSI Scores)")
drift = mon.get("feature_drift", [])
features_metadata = mon.get("features_metadata", {}) or {}
if drift:
    has_weights = any(r.get("weight_pct") is not None for r in drift)
    weight_header = "<th>MODEL WEIGHT</th>" if has_weights else ""
    rows = []
    for r in drift:
        psi = r.get("psi", 0.0)
        status = r.get("status", "OK")
        psi_color = utils.AMBER if status == "WARN" else (
            utils.RED if status == "ALERT" else utils.TEXT
        )
        pill_cls = {"OK": "ok", "WARN": "warn", "ALERT": "alert",
                    "INSUFFICIENT": "ok"}.get(status, "ok")
        n_base, n_cur = r.get("n_baseline"), r.get("n_current")
        samples = f" ({n_base}/{n_cur})" if n_base is not None and n_cur is not None else ""
        label = utils.describe_feature(r.get("feature", "")) or r.get("feature", "")
        # Hover tooltip from the backend-generated features_metadata artifact
        # (definition/formula/source/window/units/direction/members). Row
        # content unchanged — the tooltip is additive; unknown features fall
        # back to the existing blurb plus a 'no detailed metadata' note.
        feat_meta = features_metadata.get(r.get("feature", ""))
        if feat_meta:
            tip = html.escape(feat_meta.get("tooltip", ""), quote=False)
        else:
            tip = html.escape(label + "\n(no detailed metadata)", quote=False)
        feature_cell = (
            f"<span title='{tip}' style='cursor:help;'>{r.get('feature','')}</span>"
            f"<div style='color:#94A3B8;font-size:0.72rem;font-weight:400;margin-top:1px;'>{label}</div>"
        )
        weight_cell = f"<td>{utils.feature_weight_pct(r)}</td>" if has_weights else ""
        rows.append(
            f"<tr>"
            f"<td style='color:#E2E8F0;'>{feature_cell}</td>"
            f"<td>{r.get('current_mean', '—')}</td>"
            f"<td>{r.get('baseline_mean', '—')}</td>"
            f"<td style='color:{psi_color};font-weight:700;'>{psi:.3f}</td>"
            f"{weight_cell}"
            f"<td><span class='fb-status-pill {pill_cls}'>{status}</span>"
            f"<span style='color:#64748B;font-size:0.72rem;margin-left:5px;'>{samples}</span></td></tr>"
        )
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <table class="fb-table">
            <thead><tr><th>FEATURE</th><th>CURRENT MEAN</th><th>BASELINE MEAN</th><th>PSI</th>
            {weight_header}<th>STATUS</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        <div style="color:#64748B;font-size:0.78rem;margin-top:6px;">
          MODEL WEIGHT = share of the final blended ensemble riding on this feature
          (blend-weighted importances across members; sums to 100%).
          Status shows the sample sizes behind each comparison as baseline/current.
          INSUFFICIENT = window too small to judge drift; PSI is informational only.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("No drift data available.")

# ---------------------------------------------------------------------------
# Feature coverage panel (measured vs default-filled, per window)
# ---------------------------------------------------------------------------
st.markdown("### Feature Coverage (non-null / measured)")
coverage = mon.get("feature_coverage", []) or []
if coverage:
    # Worst first: lowest measured share at top. This is the visual backstop
    # for silent data starvation — a fetcher can die for a whole season while
    # PSI rows keep showing plausible zeros (the 2026 weather truncation did
    # exactly that). MEASURED excludes default-filled values (e.g. the dome
    # branch's exact-0 wind) so legitimate zeros cannot mask absence.
    cov_sorted = sorted(
        coverage,
        key=lambda r: (r.get("pct_measured", 0.0), r.get("feature", "")),
    )
    n_starved = sum(1 for r in coverage if r.get("status") == "STARVED")
    n_low = sum(1 for r in coverage if r.get("status") == "LOW_COVERAGE")
    sub = (
        f"<span style='color:{utils.RED};font-weight:700;'>{n_starved} starved</span>"
        f" · <span style='color:{utils.AMBER};font-weight:700;'>{n_low} low</span>"
        if (n_starved or n_low) else
        "<span style='color:#4ADE80;font-weight:700;'>all windows healthy</span>"
    )
    st.markdown(
        f"<div style='color:#94A3B8;font-size:0.8rem;margin:-6px 0 10px;'>"
        f"Share of games in each drift window with a real observation per feature — {sub}</div>",
        unsafe_allow_html=True)
    show_starved_only = n_starved + n_low > 0
    cov_rows = []
    shown = 0
    for r in cov_sorted:
        status = r.get("status", "OK")
        if show_starved_only and status == "OK" and shown >= 12:
            continue  # keep the table readable; healthy tail is summarized below
        pct_m = float(r.get("pct_measured", 0.0))
        pct_n = float(r.get("pct_nonnull", 0.0))
        n_def = int(r.get("n_default_zero", 0) or 0)
        color = utils.RED if status == "STARVED" else (
            utils.AMBER if status == "LOW_COVERAGE" else utils.TEXT)
        pill_cls = {"OK": "ok", "LOW_COVERAGE": "warn", "STARVED": "alert"}.get(status, "ok")
        default_cell = (
            f"<div style='color:#94A3B8;font-size:0.72rem;font-weight:400;margin-top:1px;'>"
            f"{n_def} default-zero</div>" if n_def else "")
        cov_rows.append(
            f"<tr>"
            f"<td style='color:#E2E8F0;'>{r.get('feature','')}</td>"
            f"<td>{r.get('window','')}</td>"
            f"<td>{r.get('n_games','—')}</td>"
            f"<td style='color:{color};font-weight:700;'>{pct_m:.0f}%</td>"
            f"<td>{pct_n:.0f}%{default_cell}</td>"
            f"<td><span class='fb-status-pill {pill_cls}'>{status}</span></td></tr>"
        )
        shown += 1
    n_hidden = len(coverage) - shown
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <table class="fb-table">
            <thead><tr><th>FEATURE</th><th>WINDOW</th><th>GAMES</th>
            <th>% MEASURED</th><th>% NON-NULL</th><th>STATUS</th></tr></thead>
            <tbody>{''.join(cov_rows)}</tbody>
          </table>
        </div>
        <div style="color:#64748B;font-size:0.78rem;margin-top:6px;">
          % MEASURED = real observations only (default-filled values excluded);
          % NON-NULL includes them. STARVED &lt;25% measured, LOW_COVERAGE &lt;80%.
          {f"{n_hidden} healthy feature-window pairs hidden." if n_hidden > 0 else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("No feature coverage data available (older run artifact).")

# ---------------------------------------------------------------------------
# Model ensemble composition
# ---------------------------------------------------------------------------
st.markdown("### Model Ensemble")
ensemble = mon.get("ensemble") or []


def _xgb_desc() -> str:
    """Derive XGBoost card text from the deployed config so it cannot drift."""
    try:
        import sys
        from pathlib import Path
        _backend = Path(__file__).resolve().parents[1] / "backend"
        if str(_backend) not in sys.path:
            sys.path.insert(0, str(_backend))
        from config import XGBOOST_PARAMS  # type: ignore[import-untyped]
        p = XGBOOST_PARAMS
    except Exception:
        return "XGBoost — config unavailable."
    learn = p.get("learning_rate", "?")
    depth = p.get("max_depth", "?")
    return (
        f"Gradient-boosted decision trees (max depth {depth}, lr {learn}, "
        "early-stopped on each fold's validation window, logloss eval). "
        "Train-median imputation replaces the old native-NaN routing — "
        "the Optuna winner picked it over raw NaN splitting."
    )


def _lgbm_desc() -> str:
    """Derive LightGBM card text from the deployed config."""
    try:
        import sys
        from pathlib import Path
        _backend = Path(__file__).resolve().parents[1] / "backend"
        if str(_backend) not in sys.path:
            sys.path.insert(0, str(_backend))
        from config import LIGHTGBM_PARAMS  # type: ignore[import-untyped]
        p = LIGHTGBM_PARAMS
    except Exception:
        return "LightGBM — config unavailable."
    learn = p.get("learning_rate", "?")
    depth = p.get("max_depth", "?")
    return (
        f"Leaf-wise histogram gradient boosting (max depth {depth}, lr {learn}, "
        f"{p.get('n_estimators', '?')} rounds, logloss eval). Grows deeper "
        "loss-guided trees than XGBoost at the same budget; routes missing "
        "values natively."
    )


ENSEMBLE_DESCRIPTIONS = {
    "xgboost": _xgb_desc(),
    "lightgbm": _lgbm_desc(),
    "logistic": (
        "L2-regularized linear model over standardized features (train-median "
        "imputation). A high-bias anchor that keeps the blend calibrated when tree "
        "members overfit thin early-season folds; also the most interpretable member."
    ),
    "randomforest": (
        "Bagged decision trees (300 estimators, deep-minimum leaves) — averaging "
        "instead of boosting, so its errors are decorrelated from XGBoost/LightGBM. "
        "Robust to noisy features; uses train-median imputation."
    ),
    "mlp": (
        "Small neural network (32×16, L2 penalty, early stopping). A low-capacity "
        "function approximator that can pick up smooth nonlinearities trees split "
        "around; earns ensemble weight only when it beats the other members OOF."
    ),
}

if ensemble:
    ens_rows = []
    total_weight = 0.0
    for m in sorted(ensemble, key=lambda x: -x.get("weight", 0.0)):
        name = m.get("name", "?")
        w = float(m.get("weight", 0.0))
        total_weight += w
        desc = ENSEMBLE_DESCRIPTIONS.get(
            name,
            f"Candidate \u201c{name}\u201d registered in the walk-forward roster."
            if w == 0.0 else f"Deployed candidate \u201c{name}\u201d.",
        )
        auc, brier, ll = m.get("auc"), m.get("brier"), m.get("logloss")
        n_eval = m.get("n_eval") or 0
        metric_txt = lambda v: (f"{v:.4f}" if isinstance(v, (int, float)) else "—")  # noqa: E731
        eval_note = f"<div style='color:#64748B;font-size:0.72rem;'>{n_eval} OOF games</div>" if n_eval else \
                    "<div style='color:#64748B;font-size:0.72rem;'>no OOF predictions</div>"
        weight_badge = (
            f"<span style='display:inline-block;min-width:52px;text-align:center;padding:2px 8px;border-radius:8px;"
            f"background:{'rgba(16,185,129,.15);color:#34D399' if w > 0 else 'rgba(100,116,139,.15);color:#94A3B8'};"
            f"font-weight:800;font-size:0.82rem;'>{w * 100:.0f}%</span>"
        )
        ens_rows.append(
            f"<tr>"
            f"<td style='color:#E2E8F0;font-weight:700;text-transform:capitalize;'>{name.replace('_', ' ')}</td>"
            f"<td style='max-width:420px;color:#94A3B8;'>{desc}{eval_note}</td>"
            f"<td>{weight_badge}</td>"
            f"<td style='color:#E2E8F0;font-weight:700;'>{metric_txt(auc)}</td>"
            f"<td>{metric_txt(brier)}</td>"
            f"<td>{metric_txt(ll)}</td>"
            f"</tr>"
        )
    total_ok = abs(total_weight - 1.0) < 0.001
    total_color = "#34D399" if total_ok else utils.AMBER
    ens_rows.append(
        f"<tr>"
        f"<td style='font-weight:800;color:#E2E8F0;'>TOTAL (blended ensemble)</td>"
        f"<td style='color:#64748B;'>Probability blend of deployed members; weights renormalize when a candidate fails to train.</td>"
        f"<td style='font-weight:800;color:{total_color};'>{total_weight * 100:.0f}%</td>"
        f"<td colspan='3'></td>"
        f"</tr>"
    )
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <table class="fb-table">
            <thead><tr><th>MODEL</th><th>DESCRIPTION</th><th>WEIGHT</th><th>AUC</th><th>BRIER</th><th>LOG LOSS</th></tr></thead>
            <tbody>{''.join(ens_rows)}</tbody>
          </table>
        </div>
        <div style="color:#64748B;font-size:0.78rem;margin-top:6px;">
          Every candidate from the walk-forward roster is listed — including zero-weight
          candidates. AUC/Brier/Log-Loss are each model's own pooled out-of-fold scores;
          the blended ensemble's headline metrics are shown in the KPI cards above.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info(
        "Ensemble composition appears after the next pipeline run "
        "(artifact predates the per-model reporting)."
    )

# ---------------------------------------------------------------------------
# Rolling Brier score timeline
# ---------------------------------------------------------------------------
st.markdown("### Rolling Brier Score (Last 30 Days)")
brier = mon.get("rolling_brier", []) or []
rb_meta = mon.get("rolling_brier_meta", {}) or {}
baseline = mon.get("brier_baseline")
baseline_label = mon.get("brier_baseline_label", "Baseline")
if brier:
    bdf = pd.DataFrame(brier)
    bdf["date"] = pd.to_datetime(bdf["date"])
    line = alt.Chart(bdf).mark_line(point=alt.OverlayMarkDef(filled=True, size=40), color="#F97316", strokeWidth=2.2).encode(
        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %d")),
        y=alt.Y("brier:Q", title="Brier Score", scale=alt.Scale(zero=False, domain=[
            max(0.10, float(bdf["brier"].min()) - 0.02),
            min(0.40, float(bdf["brier"].max()) + 0.02),
        ]) if len(bdf) > 1 else alt.Scale(zero=False)),
    )
    layers = []
    if baseline is not None:
        base_df = pd.DataFrame({"x": [bdf["date"].min(), bdf["date"].max()], "y": [baseline, baseline]})
        layers.append(alt.Chart(base_df).mark_line(color="#64748B", strokeDash=[5, 5], strokeWidth=1.5).encode(
            x="x:T", y=alt.Y("y:Q", scale=alt.Scale(zero=False)),
        ))
    layers.append(line)
    utils.show_chart(alt.layer(*layers).properties(height=300))
    window_days = rb_meta.get("window_days", 30)
    min_games = rb_meta.get("min_games_per_day", "—")
    sparse_note = (
        f" · {rb_meta['excluded_sparse_days']} sparse days excluded"
        if isinstance(rb_meta.get("excluded_sparse_days"), int) and rb_meta["excluded_sparse_days"]
        else ""
    )
    calib_note = " · calibrated probabilities" if rb_meta.get("calibrator_is_identity") is False else " · probabilities (no calibration map deployed)"
    st.caption(
        f"Orange: mean Brier over the trailing {window_days} days of decided games "
        f"(≥{min_games} games/day{sparse_note}){calib_note}."
        + (f" Dashed: {baseline_label} = {baseline:.4f}." if baseline is not None else "")
    )
    map_note = rb_meta.get("map_scope_note")
    if map_note:
        st.caption(f"⚠️ {map_note}")
else:
    st.info(
        "No rolling Brier data available yet — the series appears once a run "
        "ships decided walk-forward predictions in predictions_history."
    )

# ---------------------------------------------------------------------------
# Model version history
# ---------------------------------------------------------------------------
st.markdown("### Model Version History")
history = mon.get("version_history", []) or []
if history:
    _W_ABBR = {
        "xgboost": "xgb", "lightgbm": "lgb", "logistic": "log",
        "randomforest": "rf", "mlp": "mlp",
    }

    def _weights_str(row: dict) -> str:
        w = row.get("weights") or {}
        if not isinstance(w, dict) or not w:
            return "—"
        parts = sorted(w.items(), key=lambda kv: -float(kv[1]))
        return " / ".join(
            f"{_W_ABBR.get(name, name)} {100 * float(v):.0f}%"
            for name, v in parts if float(v) > 0
        ) or "—"

    def _fmt(row: dict, key: str, digits: int = 4) -> str:
        v = row.get(key)
        return f"{float(v):.{digits}f}" if isinstance(v, (int, float)) else "—"

    rows = []
    for row in history:
        cal = row.get("calibration") or {}
        cal_str = (
            f"a={float(cal['a']):.3f}, b={float(cal['b']):.3f}"
            if isinstance(cal, dict) and "a" in cal else "identity/none"
        )
        rows.append({
            "VERSION": row.get("version", "—"),
            "DATE": _fmt_date(str(row.get("date", ""))),
            "WEIGHTS": _weights_str(row),
            "AUC": _fmt(row, "auc"),
            "LOGLOSS": _fmt(row, "logloss"),
            "CAL. ECE": _fmt(row, "ece_calibrated"),
            "CAL. MAP": cal_str,
        })
    st.table(pd.DataFrame(rows).reset_index(drop=True))
    st.caption(
        "One row per retrain — OOF-earned ensemble weights, pooled walk-forward "
        "metrics (raw AUC/logloss; CAL. ECE is prequential-calibrated), and the "
        "deployed Platt map. Legacy rows show AUC/Brier only."
    )
else:
    st.info("No version history yet — it appears after the first run with version snapshots.")
