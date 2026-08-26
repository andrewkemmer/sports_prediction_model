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


