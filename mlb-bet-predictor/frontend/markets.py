"""Page 5 — Today's Totals & Run Lines (run engine, Phase 3).

Temporary UI simplification: this page renders ONLY the six diagnostics
charts (Distribution, Relativized, Pooled lines, Money line (rounded),
Totals picks, Run-line picks) from
frontend/market_diagnostics.py — pure functions
over run_engine_markets_<date>.csv; the render layer only draws what they
produce. The per-game slate board, total/run-line selectors, calibration
table, rolling totals-Brier, tail fit-check, holdout verdict, and the
derived-ML cross-reference are UN-RENDERED for now — nothing is deleted from
the backend or the artifact, and the charts' empty states and loud warnings
still fire exactly as before.

Missing/stale artifacts produce loud warnings and graceful empty states —
never fake data.
"""

from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st

import utils

utils.inject_css()

dates = utils.available_dates(**utils.get_source_config())
date_str = st.session_state.get("selected_date", dates[0] if dates else "20260809")

st.markdown(
    "<div style='font-size:1.7rem;font-weight:800;color:#E2E8F0;'>"
    "Today's Totals &amp; Run Lines</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='color:#94A3B8;margin:2px 0 14px;'>"
    "Run-engine model diagnostics from NB(λ, α(λ)) Monte Carlo over the "
    "line grid — per-game projections and full probabilities live in "
    "run_engine_markets_*.csv.</div>",
    unsafe_allow_html=True,
)


def _load_markets(ds):
    """Fetch the run-engine markets CSV.  relpath is a bare filename —
    _fetch_bytes prepends the repo subdir + data_delivery/ internally."""
    import logging
    _log = logging.getLogger("markets")
    fname = f"run_engine_markets_{ds}.csv"
    cfg = utils.get_source_config()
    try:
        raw, src = utils._fetch_bytes(fname, **cfg)
    except Exception as exc:
        # Build the attempted URL for actionable diagnostics
        url = utils._raw_url(fname, **cfg)
        _log.error("Markets fetch exception for %s (%s): %s", fname, url, exc)
        st.warning(f"Fetch error for run_engine_markets_{ds}.csv – see log.")
        return None
    if raw is None:
        url = utils._raw_url(fname, **cfg)
        _log.warning("Markets artifact not found: %s (URL: %s, source: %s)",
                     fname, url, src)
        return None
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        _log.error("Markets CSV parse failed for %s: %s", fname, exc)
        return None


markets = _load_markets(date_str)
if markets is None or not len(markets):
    url = utils._raw_url(f"run_engine_markets_{date_str}.csv",
                         **utils.get_source_config())
    st.warning(
        f"No run-engine markets artifact for {date_str}. "
        f"Attempted URL: `{url}`. "
        "The panel fills after the next pipeline run ships "
        "run_engine_markets_*.csv."
    )
    markets = pd.DataFrame()
elif "kind" not in markets.columns:
    st.warning(
        "Markets artifact predates Phase 3 (no line grid / slate rows). "
        "Waiting for the next pipeline run."
    )

# ---------------------------------------------------------------------------
# Diagnostics — six charts, computed by frontend/market_diagnostics.py
# (pure functions over the artifact; the render layer only draws).
# ---------------------------------------------------------------------------
import market_diagnostics as diag  # noqa: E402  (page module, imported late)

decided = diag.decided_rows(markets)
st.markdown("### Diagnostics")
if decided.empty:
    st.warning(
        "No decided OOF rows in run_engine_markets for this date — "
        "diagnostics need outcomes. They appear after a run that ships "
        "decided games; nothing is fabricated in the meantime."
    )
else:
    _tabs = st.tabs([
        "Distribution", "Relativized", "Pooled lines",
        "Money line (rounded)", "Totals picks", "Run-line picks",
    ])

    with _tabs[0]:   # 1 — totals distribution fit-check
        dist = diag.total_distribution(decided)
        if dist["warning"]:
            st.warning(dist["warning"])
        else:
            utils.show_chart(diag.chart_distribution(dist))
            c = dist["callouts"]
            st.caption(
                f"Observed bars vs modeled mean NB(λ, α(λ)) convolution over "
                f"{dist['n_games']:,} decided games · P(total≤1): observed "
                f"{c['P(total<=1)']['observed']:.3f} / modeled "
                f"{c['P(total<=1)']['modeled']:.3f} · P(total≥10): observed "
                f"{c['P(total>=10)']['observed']:.3f} / modeled "
                f"{c['P(total>=10)']['modeled']:.3f}. Per-TEAM tail checks are "
                "the fit-check table above — this is the totals law."
            )

    with _tabs[1]:   # 2 — relativized-offset calibration (primary curve)
        pairs = diag.relativized_pairs(decided)
        curve = diag.calibration_curve(pairs)
        if curve["warning"]:
            st.warning(curve["warning"])
        else:
            utils.show_chart(diag.chart_calibration(
                curve, "Relativized offsets −2.0 … +2.0"))
            xs = [b["mean_pred"] for b in curve["bins"]]
            st.caption(
                f"Each game priced at ITS own expected-total ± offset "
                f"({', '.join(f'{o:+.1f}' for o in diag.OFFSET_EDGES)}); lines "
                f"priced by monotone logit interpolation of the grid (documented "
                f"in market_diagnostics). {curve['n_pairs']:,} pairs, "
                f"{len(xs)} valid bins (≥30 each, {curve['n_dropped_bins']} "
                f"dropped) · predicted range {min(xs):.2f}–{max(xs):.2f} — the "
                "spread is the point; the dashed diagonal is perfect calibration."
            )
            st.warning(
                "**Known limitation — deep-over region:** At lines well below "
                "expected total (offset ≤ −2.0), p_over is high but the model "
                "is overconfident. Measured gap (3.5c full-coverage re-check, "
                "2026-08-24 OOF): prediction ≈ 0.66 vs actual ≈ 0.60 "
                "(≈ 0.06 shortfall, n ≈ 4,156). This is "
                "weather-independent — the gap is identical WITHOUT the "
                "env-level features (+0.058 vs +0.054), reflecting a "
                "tail-pricing limitation inherent to the mean-model's tight "
                "λ clustering interacting with extreme offsets. Deep-over "
                "lines carry lower confidence."
            )

    with _tabs[2]:   # 3 — pooled fixed lines
        fpairs = diag.fixed_line_pairs(decided, (7.5, 8.5, 9.5, 10.5))
        fcurve = diag.calibration_curve(fpairs)
        if fcurve["warning"]:
            st.warning(fcurve["warning"])
        else:
            utils.show_chart(diag.chart_calibration(
                fcurve, "Games pooled across 7.5 / 8.5 / 9.5 / 10.5"))
            st.caption(
                f"{fcurve['n_pairs']:,} (game × line) pairs; every game counts "
                "once per line — near-line degeneracy is diluted by the off-"
                "line games pooled in."
            )

    with _tabs[3]:   # 4 — per-game rounded-total money line, pooled
        mrt = diag.rounded_total_pairs(decided)
        mrtc = diag.calibration_curve(mrt)
        if mrtc["warning"]:
            st.warning(mrtc["warning"])
        else:
            utils.show_chart(diag.chart_calibration(
                mrtc, "Per-game rounded total"))
            xs = [b["mean_pred"] for b in mrtc["bins"]]
            _ps = diag.push_stats(decided)
            st.caption(
                f"Each game priced at ITS OWN rounded total — nearest 0.5 of "
                f"λ_home + λ_away (round half up; lines outside the shipped "
                f"6.5–12.5 grid clamp to the edge) — then pooled. "
                f"{_ps['n_games']:,} games · {_ps['n_pushes']:,} pushes "
                f"excluded ({_ps['push_rate']:.1%}) · predicted range "
                f"{min(xs):.2f}–{max(xs):.2f}. Pushes are UNDER-favored "
                "games landing exactly on the line (rounded line at/above "
                "the expected total → under favored) — excluded from the "
                "curve because they are neither wins nor losses. Every game "
                "sits at its own line, so probabilities concentrate near "
                "the money line by construction — the wide calibration "
                "spread lives in the Relativized tab."
            )

    with _tabs[4]:   # 5 — totals picks at each game's rounded line
        tpicks = diag.totals_pick_table(decided)
        if tpicks["warning"] or not tpicks["buckets"]:
            st.warning(tpicks.get("warning")
                       or "No totals picks could be formed.")
        else:
            built = diag.chart_pick_buckets(
                tpicks, "Totals picks (per-game rounded line)",
                total_line=True, acc_y_max=75.0)
            utils.show_chart(built["chart"])
            st.table(built["table"])
            st.caption(
                f"Pick rule: {tpicks['pick_rule']} · {tpicks['n_games']:,} "
                f"decided games · {tpicks['n_pushes']:,} pushes excluded "
                f"({tpicks['push_rate']:.1%}) · pooled win rate: "
                f"{tpicks['win_rate']:.1%}. Pushes are UNDER-favored games "
                "landing exactly on the line (rounded line at/above the "
                "expected total → under favored) and were "
                "previously scored as wins — excluding them LOWERS the honest "
                "win rate vs the inflated one (2026-08-24 artifact: 56.1% → 54.1%, "
                "≈2,420 wins/4,314 → ≈2,200 wins/4,066). Every game is priced "
                "at its own rounded total, so high-confidence buckets are "
                "small. Hit rate is NOT calibration — it is binary pick "
                "accuracy per favored-side confidence bucket."
            )

    with _tabs[5]:   # 6 — run-line pick accuracy buckets
        rpicks = diag.runline_pick_table(decided)
        if rpicks["warning"] or not rpicks["buckets"]:
            st.warning(rpicks.get("warning")
                       or "No run-line picks could be formed.")
        else:
            built = diag.chart_pick_buckets(rpicks, "Run-line picks at −1.5")
            utils.show_chart(built["chart"])
            st.table(built["table"])
            st.caption(
                f"Pick rule: {rpicks['pick_rule']} · {rpicks['n_games']:,} "
                "decided games · hit rate is NOT calibration."
            )


# ---------------------------------------------------------------------------
# Prediction history — every game (game totals & run lines)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _team_map() -> dict:
    """Dual-keyed team map from game_level_features.csv.

    The markets artifact carries game keys + scores but no team names;
    the matchups come from the shipped game-level features, which carry
    BOTH StatsAPI game_pk and ESPN game_id — so a markets key resolves
    either way (build_team_map keys by int(game_pk) AND str(game_id),
    the 145d841 slate-key discipline). Empty dict when the artifact is
    unavailable (the caller warns loudly and skips the tables — never
    fabricated)."""
    cfg = utils.get_source_config()
    raw, _ = utils._fetch_bytes("game_level_features.csv", **cfg)
    if raw is None:
        return {}
    try:
        gl = pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in (
            "game_pk", "game_id", "home_team", "away_team"))
        return diag.build_team_map(gl)
    except Exception:
        return {}


def _market_prob_note(date_str: str) -> str:
    """Honest calibration label for the shipped grid probabilities.

    The markets artifact ships RAW grid columns; the meta JSON records
    prequentially CALIBRATED metrics (engine_*_calibrated) but does not
    ship per-row Platt maps (and the CSV rows carry no fold id to apply
    one), so the displayed probabilities are raw. If a future meta ships
    a usable Platt section, this notes post-calibration instead — it
    never claims calibration that is not there."""
    cfg = utils.get_source_config()
    raw, _ = utils._fetch_bytes(f"run_engine_markets_{date_str}.meta.json",
                                **cfg)
    if raw is not None:
        try:
            meta = json.loads(raw)
        except Exception:
            meta = {}
        for k, v in (meta or {}).items():
            if "calibr" in str(k).lower() and isinstance(v, dict) \
                    and "a" in v and "b" in v:
                return (" · probabilities are post-calibration "
                        "σ(a·logit(p)+b)")
    return (" · probabilities are RAW (no calibration map shipped in "
            "the run-engine artifact)")


def _result_cell(correct) -> str:
    """✓/✗ RESULT cell, '—' for pushes (neither wins nor loses)."""
    if pd.isna(correct):
        return "<td>—</td>"
    if bool(correct):
        return f"<td style='color:{utils.PRIMARY};font-weight:700;'>✓</td>"
    return f"<td style='color:{utils.RED};font-weight:700;'>✗</td>"


def _score_cell(row) -> str:
    if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
        return f"{int(row['away_score'])}–{int(row['home_score'])}"
    return "—"


def _date_str(raw) -> str:
    d = pd.to_datetime(raw, errors="coerce")
    return d.strftime("%b %d, %Y") if pd.notna(d) else "—"


def _render_totals_history(tot: pd.DataFrame, teams: dict,
                           start_d, end_d, cal_note: str) -> None:
    """Game-totals prediction history: DATE | MATCHUP | SCORE (A–H) |
    LINE (O/U X.5) | MODEL PICK (Over/Under X.5 (p%)) | WINNER | RESULT.
    Most recent first in a fixed-height scroll container; header win rate
    recomputes from the filtered rows; pushes excluded from the win rate
    and counted in the caption (same semantics as the diagnostics)."""
    view = diag.filter_history_frame(tot, start_d, end_d)
    view = view.sort_values("game_date", ascending=False)
    if not len(view):
        st.markdown("#### Game Totals — Prediction History")
        st.info("No games in the selected date range.")
        return
    stats = diag.history_win_rate(view)
    rate = stats["win_rate"]
    rate_txt = f"{rate * 100:.1f}% picks correct" if rate is not None \
        else "no priced games"
    push_txt = (
        f" · {stats['n_pushes']:,} push(es) excluded — total == whole-"
        "number line, neither wins nor loses") if stats["n_pushes"] else ""
    st.markdown("#### Game Totals — Prediction History")
    st.caption(
        f"{stats['n_games']:,} games · {rate_txt} · most recent first — "
        f"scroll for older results{push_txt}{cal_note}"
    )
    rows = []
    for _, r in view.iterrows():
        away, home = diag.resolve_matchup_teams(teams, r["game_pk"])
        rows.append(
            f"<tr><td>{_date_str(r['game_date'])}</td>"
            f"<td>{away} @ {home}</td>"
            f"<td>{_score_cell(r)}</td>"
            f"<td>O/U {r['line']:.1f}</td>"
            f"<td>{r['pick']} {r['line']:.1f} ({r['pick_prob']:.0%})</td>"
            f"<td>{r['winner']}</td>{_result_cell(r['correct'])}</tr>"
        )
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <div style="max-height:480px;overflow-y:auto;">
            <table class="fb-table">
              <thead><tr><th>DATE</th><th>MATCHUP</th><th>SCORE (A–H)</th>
              <th>LINE</th><th>MODEL PICK</th><th>WINNER</th>
              <th>RESULT</th></tr></thead>
              <tbody>{''.join(rows)}</tbody>
            </table>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_runline_history(rl: pd.DataFrame, teams: dict,
                            start_d, end_d, cal_note: str) -> None:
    """Run-line (−1.5/+1.5) prediction history: DATE | MATCHUP |
    SCORE (A–H) | LINE (RL −1.5/+1.5) | MODEL PICK (TEAM ±1.5 (p%)) |
    WINNER | RESULT. Half-run lines never push, so every decided game
    counts toward the win rate (noted in the caption)."""
    view = diag.filter_history_frame(rl, start_d, end_d)
    view = view.sort_values("game_date", ascending=False)
    if not len(view):
        st.markdown("#### Run Lines — Prediction History")
        st.info("No games in the selected date range.")
        return
    stats = diag.history_win_rate(view)
    rate = stats["win_rate"]
    rate_txt = f"{rate * 100:.1f}% picks correct" if rate is not None \
        else "no priced games"
    st.markdown("#### Run Lines — Prediction History")
    st.caption(
        f"{stats['n_games']:,} games · {rate_txt} · most recent first — "
        f"scroll for older results · half-run lines never push, so every "
        f"decided game counts{cal_note}"
    )
    rows = []
    for _, r in view.iterrows():
        away, home = diag.resolve_matchup_teams(teams, r["game_pk"])
        if r["pick"] == "home":
            pick_txt = f"{home} −1.5 ({r['pick_prob']:.0%})"
            winner_txt = home
        else:
            pick_txt = f"{away} +1.5 ({r['pick_prob']:.0%})"
            winner_txt = away
        rows.append(
            f"<tr><td>{_date_str(r['game_date'])}</td>"
            f"<td>{away} @ {home}</td>"
            f"<td>{_score_cell(r)}</td>"
            f"<td>RL −1.5/+1.5</td>"
            f"<td>{pick_txt}</td>"
            f"<td>{winner_txt}</td>{_result_cell(r['correct'])}</tr>"
        )
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <div style="max-height:480px;overflow-y:auto;">
            <table class="fb-table">
              <thead><tr><th>DATE</th><th>MATCHUP</th><th>SCORE (A–H)</th>
              <th>LINE</th><th>MODEL PICK</th><th>WINNER</th>
              <th>RESULT</th></tr></thead>
              <tbody>{''.join(rows)}</tbody>
            </table>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown("### Prediction History — Totals & Run Lines")
if decided.empty:
    st.info(
        "No decided OOF rows in the run-engine markets artifact — the "
        "totals/run-line history fills after a run that ships decided "
        "games. Nothing is fabricated in the meantime."
    )
else:
    tot = diag.totals_history_frame(decided)
    rl = diag.runline_history_frame(decided)
    teams = _team_map()
    if not teams:
        st.warning(
            "Team-name artifact (game_level_features.csv) is unavailable "
            "— the history tables need matchups and are skipped. Nothing "
            "is fabricated."
        )
    else:
        _dts = pd.to_datetime(decided["game_date"], errors="coerce").dropna()
        lo, hi = _dts.min().date(), _dts.max().date()
        fc1, fc2, _ = st.columns([1, 1, 2])
        start_d = fc1.date_input("History start date", value=lo,
                                 min_value=lo, max_value=hi)
        end_d = fc2.date_input("History end date", value=hi,
                               min_value=lo, max_value=hi)
        if start_d > end_d:
            start_d, end_d = end_d, start_d
        _cal_note = _market_prob_note(date_str)
        _render_totals_history(tot, teams, start_d, end_d, _cal_note)
        _render_runline_history(rl, teams, start_d, end_d, _cal_note)





# ---------------------------------------------------------------------------
# Run-Line & Totals Monitor — per-line calibration cards, fit panel,
# rolling history from run_engine_monitor_YYYYMMDD.json
# ---------------------------------------------------------------------------
# The moneyline monitor lives in model_monitor_*.json (model_monitor.py).
# This is its run-engine counterpart: per-reference-line calibration stats,
# distributional fit diagnostics (alpha, chi-squared, tail checks), and a
# rolling per-line history.  The file is protected from Phase-6 cleanup by
# the ``run_engine_monitor_`` prefix in _PROTECTED_DELIVERY_PREFIXES.

import datetime as _dt  # noqa: E402 — page-level, after existing imports

_WINNER_CARDS = (
    ("over_under", "Over/Under",
     "Pick Over if P(over the game's line) > 50%, else Under"),
    ("run_line", "Run Line ±1.5",
     "Pick Home −1.5 if P(home cover −1.5) > 50%, else Away +1.5"),
    ("derived_ml", "Derived ML (run-line model moneyline)",
     "Pick the side with P > 50% — home if P(home win) > 50%, else away "
     "(NB Monte-Carlo derived ML; distinct from the moneyline ensemble)"),
)


def _load_run_engine_monitor(ds: str) -> dict | None:
    """Load run_engine_monitor_YYYYMMDD.json from data_delivery."""
    import logging as _lg
    _log = _lg.getLogger("markets")
    fname = f"run_engine_monitor_{ds}.json"
    cfg = utils.get_source_config()
    try:
        raw, src = utils._fetch_bytes(fname, **cfg)
    except Exception as exc:
        url = utils._raw_url(fname, **cfg)
        _log.error("Monitor fetch exception for %s (%s): %s", fname, url, exc)
        st.warning(f"Fetch error for {fname} -- see log.")
        return None
    if raw is None:
        url = utils._raw_url(fname, **cfg)
        _log.warning("Monitor artifact not found: %s (URL: %s, source: %s)",
                     fname, url, src)
        return None
    try:
        return json.loads(raw)
    except Exception as exc:
        _log.error("Monitor JSON parse failed for %s: %s", fname, exc)
        return None


def _fmt(v, nd=3, pct=False) -> str:
    """Format a numeric value for display; None/dash for missing."""
    if v is None:
        return "--"
    if pct:
        return f"{v * 100:.1f}%"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _ec_indicator(raw: float | None, cal: float | None) -> str:
    """Color-coded ECE badge: green < 0.01, yellow < 0.02, red >= 0.02."""
    val = cal if cal is not None else raw
    if val is None:
        return "⚪"
    if val < 0.01:
        return "🟢"
    if val < 0.02:
        return "🟡"
    return "🔴"


def _render_winner_cards(winner_cards: dict) -> None:
    """Render the three binary winner cards (pick framing) in a row."""
    items = [(k, label, rule) for k, label, rule in _WINNER_CARDS
             if k in winner_cards]
    if not items:
        st.info("No winner-card data in this monitor artifact.")
        return

    cols = st.columns(len(items))
    for i, (key, label, rule) in enumerate(items):
        card = winner_cards[key]
        h = card.get("holdout") or {}
        badge = _ec_indicator(
            card.get("ece_raw"), card.get("ece_calibrated"))
        with cols[i]:
            st.markdown(f"**{label}** {badge}")
            st.caption(rule)
            # One compact actual-vs-predicted line (NOT separate columns).
            awr = card.get("actual_win_rate")
            pm = card.get("predicted_mean")
            if awr is not None and pm is not None:
                compact = (f"Actual {awr * 100:.1f}% · "
                           f"Predicted {pm * 100:.1f}%")
            else:
                compact = "Actual -- · Predicted --"
            st.markdown(f"**{compact}**")
            c1, c2 = st.columns(2)
            c1.metric("Win rate", _fmt(card.get("win_rate"), pct=True))
            c2.metric("AUC", _fmt(card.get("auc"), 4))
            c1, c2 = st.columns(2)
            c1.metric("Pooled ECE-cal", _fmt(card.get("ece_calibrated")))
            c2.metric("Holdout ECE-cal",
                      _fmt(h.get("ece_calibrated")) if h else "--")
            c1, c2 = st.columns(2)
            c1.metric("Pooled Brier", _fmt(card.get("brier")))
            c2.metric("Pooled Logloss", _fmt(card.get("logloss"), 4))
            st.caption(f"n = {card.get('n', '--'):,} pooled"
                       + (f" / {h.get('n', 0):,} holdout" if h else ""))
            ref = card.get("ml_reference")
            if ref and key == "derived_ml":
                rwr = ref.get("win_rate")
                st.caption(
                    f"Moneyline ensemble reference (ml_win_prob): "
                    f"{_fmt(rwr, pct=True) if rwr is not None else '--'} "
                    f"win rate (n={ref.get('n', '--'):,}) — the run-line "
                    "model's NB moneyline underweights the home edge")


def _render_fit_panel(fit: dict) -> None:
    """Render distributional fit diagnostics (alpha, chi2/df, tail checks)."""
    if not fit:
        st.info("No fit diagnostics in the monitor artifact.")
        return

    st.markdown("#### Distributional Fit (NB Monte Carlo)")

    # Alpha per side
    a_h = fit.get("alpha_home") or {}
    a_a = fit.get("alpha_away") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("alpha_home", _fmt(a_h.get("alpha_hat"), 4))
    c2.metric("chi2/df_home", _fmt(fit.get("chi2_per_df_home"), 2))
    c3.metric("alpha_away", _fmt(a_a.get("alpha_hat"), 4))
    c4.metric("chi2/df_away", _fmt(fit.get("chi2_per_df_away"), 2))

    # Tail checks from fit tables
    for side_label, side_key in [("Home", "home"), ("Away", "away")]:
        tbl = (fit.get("fit_tables") or {}).get(side_key)
        if not tbl:
            continue
        tail_rows = [r for r in tbl if r.get("k") in ("<={1}", ">={10}")]
        if tail_rows:
            parts = []
            for r in tail_rows:
                k_label = str(r.get("k", "")).replace("<={1}", "k<=1").replace(
                    ">={10}", "k>=10")
                parts.append(
                    f"{k_label}: obs={r.get('observed', '--'):.3f} "
                    f"mod={r.get('modeled', '--'):.3f}")
            st.caption(f"**{side_label}** tail: " + " | ".join(parts))

    # Variance check
    vc = fit.get("variance_check") or {}
    if vc:
        st.caption(
            f"Variance check: home_raw={vc.get('home_raw_var', '--')}, "
            f"home_NB={vc.get('home_nb_var', '--')}, "
            f"away_raw={vc.get('away_raw_var', '--')}, "
            f"away_NB={vc.get('away_nb_var', '--')}")

    # Monte Carlo metadata
    mc = fit.get("mc_meta") or {}
    if mc:
        st.caption(
            f"Monte Carlo: {mc.get('n_samples', '--'):,} samples, "
            f"seed={mc.get('seed', '--')}")


def _render_rolling_history(rolling: dict) -> None:
    """Render per-line rolling ECE-calibrated history as a compact table."""
    if not rolling:
        st.info("No rolling history yet (first build starts empty).")
        return

    has_data = any(len(v) > 0 for v in rolling.values())
    if not has_data:
        st.info("Rolling history is empty (will populate on subsequent runs).")
        return

    st.markdown("#### Rolling ECE-Calibrated History (per winner card)")
    rows = []
    for key, label, _rule in _WINNER_CARDS:
        series = rolling.get(key) or []
        if not series:
            continue
        # Show last 10 points
        recent = series[-10:]
        for pt in recent:
            rows.append({
                "Line": label,
                "Date": pt.get("date", "--"),
                "ECE-cal": _fmt(pt.get("ece_calibrated")),
                "Brier": _fmt(pt.get("brier")),
                "Logloss": _fmt(pt.get("logloss"), 4),
                "Pred mean": _fmt(pt.get("predicted_mean")),
                "n": pt.get("n", 0),
            })
    if rows:
        rdf = pd.DataFrame(rows)
        st.dataframe(rdf, use_container_width=True, hide_index=True)
    else:
        st.info("No rolling history points yet.")


# ── Monitor section (appended after prediction history) ──────────────────
st.markdown("---")
st.markdown("### Run-Line & Totals Monitor")
st.caption(
    "Run-engine winner cards (over/under, run line, derived ML) + "
    "distributional fit + rolling history from run_engine_monitor_*.json.  "
    "**Honesty note:** run-engine "
    "ECE-calibrated is typically ~0.014-0.018 pooled / ~0.05 holdout -- "
    "weaker than the moneyline monitor's.  This is the actual calibration "
    "quality; no styling hides it.")

monitor = _load_run_engine_monitor(date_str)
if monitor is None:
    url = utils._raw_url(
        f"run_engine_monitor_{date_str}.json", **utils.get_source_config())
    st.warning(
        f"No run-engine monitor artifact for {date_str}. "
        f"Attempted URL: `{url}`. "
        "The monitor fills after the next pipeline run ships "
        "run_engine_monitor_*.csv.")
else:
    # Markets-persisted flag
    persisted = monitor.get("markets_persisted", True)
    persist_err = monitor.get("markets_persist_error")
    if not persisted:
        st.error(
            f"**Markets CSV was NOT persisted for {date_str}**: "
            f"{persist_err or 'unknown error'}.  "
            "The monitor data below is from the in-memory summary; "
            "downstream consumers should not trust the markets artifact.")
    elif persist_err:
        st.warning(
            f"Markets persist warning: {persist_err} (but CSV was written).")

    # Winner cards (v2 schema: three binary pick-framed cards)
    winner_cards = monitor.get("winner_cards") or {}
    if winner_cards:
        _render_winner_cards(winner_cards)
    else:
        st.info("No winner-card data in this monitor artifact (v1 monitor "
                "files predate the winner-card schema).")

    # Fit panel
    fit = monitor.get("fit") or {}
    if fit:
        with st.expander("Distributional Fit Diagnostics", expanded=False):
            _render_fit_panel(fit)

    # Rolling history
    rolling = monitor.get("rolling") or {}
    with st.expander("Rolling History (last 10 points per card)",
                     expanded=False):
        _render_rolling_history(rolling)
