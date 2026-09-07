"""NFL Today's Games — STRUCTURAL 1:1 MIRROR of the MLB Today's Games page
(the MLB board inside ``frontend/todays_games.py``), with NFL artifacts
substituted. Built as a page module exactly like the verified
``nfl_markets_page.py`` mirror of ``markets.py``; ``todays_games.py``'s NFL
branch delegates here (one dispatch line) so MLB behavior is untouched.

ANATOMY CHECKLIST (MLB render order — mirrored exactly; documented swaps
marked ⟐):

  1. Board title + artifact caption (same sentence shape; ⟐ NFL moneyline
     v1 tag instead of the MLB snapshot date).
  2. Date reset / first visit → nearest valid date to today (ET) — SHARED
     helpers (utils.nearest_valid_date, same session-state keys).
  3. Missing-date fallback → the SHARED `_render_nearest_valid_fallback`
     (imported — identical behavior, no duplicate logic).
  4. Board load: ⟐ ``utils.load_nfl_moneyline()`` filtered to the selected
     game date (MLB loads todays_games_<date>.csv + calibration + the
     run-engine slate map).
  5. Header strip: `· N of M games shown` + evening-games pill +
     `✓ W-L Today · acc% accuracy` badge — SAME markup, fed by ⟐
     ``utils.load_calibration(date, sport='nfl')`` (nfl_calibration_*.json;
     its record shape has no today_record yet → honest zeros, identical
     markup); evening count computed from the start times with the shared
     ``utils._is_evening_start`` convention.
  6. ``_render_date_nav`` — the SHARED date navigation (arrows + calendar
     + mobile rail), imported, not duplicated.
  7. Filter pills `All Games (n) / Final (n) / Live (n)` — same
     ``st.pills``/``segmented_control`` fallback pair, same session key
     shape (namespaced ``nfl_game_filter``), same counts math.
  8. ``st.divider()`` → two-per-row card loop — same columns(2), same
     iteration, same per-card flow: ⟐ run-engine O/U + spread selectors
     (the existing ``_nfl_run_engine_selectors`` seam) → card HTML →
     the SAME per-card SHAP expander as MLB (``_nfl_shap_expander``,
     imported from the shared board module; the backend emits
     ``nfl_shap_game_<game_id>.csv`` attributions from the deployed
     ensemble's tree members).
  9. Card (⟐ ``_card_html`` mirror): top badge strip (☀/🌙 + LIVE/PRE-GAME/
     FINAL + ✓/X pills) → scoreboard (winner bars; PRE-GAME renders the
     0-0 display exactly like MLB) → team rows (records, PICK badge,
     trophy, prob bars) → `fb-pregame` line → ⟐ QB-matchup twin boxes
     (the ``fb-pitchers`` grid, MLB's TWO-stat single-line format:
     Rating · TD/g — the ERA/K-9 analogs) → ``fb-venue`` (📍 stadium ·
     kickoff ET) → RUN ENGINE strip (the existing
     ``nfl_slate_view.runengine_html``: Proj / O/U / RL — no separate ML
     span, matching MLB's strip anatomy) → banner.
  10. Point-in-time caption — SAME wording.

Board title / artifact caption: the MLB board renders neither (the header
strip IS the page header), so the NFL mirror renders neither either —
the extra 'NFL moneyline board' line and the ⟐ title from earlier
iterations were removed for byte-parity with the MLB element tree.

Empty / missing states: no board → the shared notice; no games on the
selected date → the shared nearest-valid fallback; missing QB record →
each QB box renders '—' quietly (never fabricated); missing run-engine
slate row → the existing quiet 'unavailable' strip.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import nfl_slate_view as nfl_sv
import utils

# The shared render helpers the MLB board uses (imported — never duplicated).
from todays_games import (  # noqa: E402
    _match_slate_row,
    _nfl_card_html,
    _nfl_run_engine_selectors,
    _nfl_shap_expander,
    _render_date_nav,
    _render_nearest_valid_fallback,
)


# ---------------------------------------------------------------------------
# ⟐ QB-matchup card blocks (the pitcher-card substitute)
# ---------------------------------------------------------------------------

def _fmt(v, fmt: str) -> str:
    """Numeric render or '—' (the artifact never fabricates)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return "—"


def _qb_box(name: str, rating, td_game) -> str:
    """One side of the QB matchup — ``_pitcher_box`` byte-format: bold name
    line + ONE stats line with the same ' · ' separator (MLB shows exactly
    two stats: ERA · K/9; the QB pair is the passer rating and TD/game —
    rating is the ERA analog, TD/g the K/9 analog). The extra per-attempt
    columns stay in the artifact; the box renders the two-stat line so the
    card geometry (single pstats line, same height) matches MLB exactly."""
    name_html = f'<div class="pname">{name}</div>' if name else ""
    stats = (
        f"Rating {_fmt(rating, '.1f')} · TD/g {_fmt(td_game, '.1f')}"
    )
    return (f'<div class="fb-pitcher">{name_html}'
            f'<div class="pstats">{stats}</div></div>')


def _qb_matchup_html(qb_row) -> str:
    """The `fb-pitchers` twin grid with QB boxes. A missing/unresolvable QB
    renders the quiet '—' box (same honesty rule as MLB's TBD pitcher)."""
    if qb_row is None:
        qb_row = {}
    home = _qb_box(
        str(qb_row.get("qb_home_name", "") or ""),
        qb_row.get("qb_home_rating"), qb_row.get("qb_home_td_per_game"))
    away = _qb_box(
        str(qb_row.get("qb_away_name", "") or ""),
        qb_row.get("qb_away_rating"), qb_row.get("qb_away_td_per_game"))
    return f'<div class="fb-pitchers">{away}{home}</div>'


# ---------------------------------------------------------------------------
# ⟐ NFL card — the MLB _card_html mirror (same section order / classes)
# ---------------------------------------------------------------------------

def _nfl_mirror_card_html(g: pd.Series, qb_row=None, re_html: str = "") -> str:
    """MLB card chrome over the NFL row (the existing compact ``_nfl_card_html``
    stays for callers that need it; this is the structural 1:1 render)."""
    home_team = str(g.get("home_team", "") or "")
    away_team = str(g.get("away_team", "") or "")
    home_name = g.get("home_team_name", "") or home_team
    away_name = g.get("away_team_name", "") or away_team
    status = str(g.get("game_status", "") or "Scheduled")
    is_final = status == "Final"
    is_live = status == "Live"
    is_scheduled = status == "Scheduled"

    hs, as_ = g.get("home_score"), g.get("away_score")
    h_score_n = None if pd.isna(hs) else int(hs)
    a_score_n = None if pd.isna(as_) else int(as_)
    # PRE-GAME renders the 0-0 scoreboard exactly like MLB's card (the
    # moneyline artifact carries null scores pre-game; the presentation is
    # the shared 0-0 display — the banner still says locked at kickoff).
    h_disp = (0 if (is_scheduled and h_score_n is None) else
              ("" if h_score_n is None else h_score_n))
    a_disp = (0 if (is_scheduled and a_score_n is None) else
              ("" if a_score_n is None else a_score_n))

    ph = g.get("home_win_prob_model")
    pa = g.get("away_win_prob_model")
    p_home = float(ph) if ph is not None and not pd.isna(ph) else None
    p_away = float(pa) if pa is not None and not pd.isna(pa) else None
    if p_home is None and p_away is not None:
        p_home = 1.0 - p_away
    if p_away is None and p_home is not None:
        p_away = 1.0 - p_home
    pick = str(g.get("model_pick", "") or "")
    is_coin_flip = (not pick) or (
        p_home is not None and abs(p_home - 0.5) < 0.02)

    winner = ""
    if h_score_n is not None and a_score_n is not None and h_score_n != a_score_n:
        winner = home_team if h_score_n > a_score_n else away_team
    correct = bool(g.get("model_correct", False)) if is_final else False

    # --- top badge strip (MLB pills; final games show ✓/X like MLB) ---
    day_tag = "🌙 Night Game"
    start_iso = str(g.get("start_time_utc", "") or "")
    try:
        from todays_games import _is_evening_start_row  # type: ignore
    except Exception:
        _is_evening_start_row = None
    if start_iso:
        try:
            import utils as _u
            if not _u._is_evening_start(start_iso):
                day_tag = "☀ Day Game"
        except Exception:
            pass
    center_pill, right_pills = "", ""
    if is_coin_flip and is_final:
        center_pill = '<span class="fb-pill coinflip">🪙 COIN FLIP</span>'
    if is_live:
        right_pills = '<span class="fb-pill live">● LIVE</span>'
    elif is_scheduled:
        right_pills = '<span class="fb-pill final">PRE-GAME</span>'
    elif is_final:
        correct_pill = '' if is_coin_flip else (
            '<span class="fb-pill correct">✓ CORRECT PICK</span>' if correct
            else '<span class="fb-pill miss">X MISS</span>')
        right_pills = correct_pill + '<span class="fb-pill final">FINAL</span>'
    top = (
        f'<span class="fb-tag">{day_tag}</span>{center_pill}'
        f'<span class="spacer"></span>{right_pills}'
    )

    # --- scoreboard (winner bars; scheduled shows kickoff like MLB) ---
    if is_scheduled:
        mid = utils.start_time_et(start_iso) or "PREGAME"
    else:
        mid = "F" if is_final else "LIVE"
    from todays_games import _score_side  # noqa: E402  (shared renderer)
    score = (
        f'<div class="fb-score">'
        f'{_score_side(a_disp, away_team, is_winner=(winner == away_team))}'
        f'<span class="mid">{mid}</span>'
        f'{_score_side(h_disp, home_team, is_winner=(winner == home_team))}'
        f'</div>'
    )

    # --- team rows + probability bars (MLB geometry; no fabricated accents) ---
    def _row(name, team, rec, p, picked, is_home):
        pct_txt = "—" if p is None else f"{p:.0%}"
        width = 0 if p is None else int(round(p * 100))
        color = utils.PRIMARY if (p is not None and p >= 0.5) else (
            "#38BDF8" if p is not None else "#334155")
        pick_badge = '<span class="fb-pill pick">PICK</span>' if picked else ""
        home_tag = ('<span class="fb-tag" style="font-size:0.7rem;">HOME</span>'
                    if is_home else "")
        return (
            f'<div class="fb-team">'
            f'<span class="fb-accent" style="background:{color};"></span>'
            f'<div><span class="name">{name}</span> <span class="sub">{team} '
            f'{rec or ""}</span> {home_tag} {pick_badge}</div>'
            f'<span class="pct" style="color:{color};">{pct_txt}</span></div>'
            f'<div class="fb-bar"><div class="fill" '
            f'style="width:{width}%;background:{color};"></div></div>'
        )

    home_row = _row(home_name, home_team, g.get("home_record", ""),
                    p_home, picked=(pick == home_team), is_home=True)
    away_row = _row(away_name, away_team, g.get("away_record", ""),
                    p_away, picked=(pick == away_team), is_home=False)

    pregame = (f'<div class="fb-pregame">Pre-game: {away_team} '
               f'{_fmt(p_away, ".0%")} vs {home_team} '
               f'{_fmt(p_home, ".0%")}</div>')

    qb_block = _qb_matchup_html(qb_row)
    start_et = utils.start_time_et(start_iso)
    venue = (f'<div class="fb-venue">📍 {g.get("venue", "") or "—"}'
             f'{f" · {start_et}" if start_et else ""}</div>')
    banner = ''
    if is_scheduled:
        first = utils.start_time_et(start_iso)
        suffix = f" — {first}" if first else ""
        banner = (f'<div class="fb-banner blue">⏳ Pre-game{suffix} · '
                  f'prediction locked at kickoff</div>')
    elif is_final:
        if is_coin_flip:
            banner = (f'<div class="fb-banner amber">🪙 {winner} Won — '
                      f'Coin Flip Game (50/50)</div>')
        elif correct:
            banner = (f'<div class="fb-banner green">✓ {winner} Won — '
                      f'Model Correct</div>')
        else:
            banner = (f'<div class="fb-banner red">X {winner} Won — '
                      f'Model picked {pick}</div>')

    return (
        f'<div class="fb-card"><div class="fb-top">{top}</div>{score}'
        f'{home_row}{away_row}{pregame}{qb_block}{venue}{re_html}{banner}</div>'
    )


# ---------------------------------------------------------------------------
# Board (the MLB main() mirror)
# ---------------------------------------------------------------------------

def _evening_count(day: pd.DataFrame) -> int:
    n = 0
    for _, g in day.iterrows():
        try:
            if utils._is_evening_start(str(g.get("start_time_utc", "") or "")):
                n += 1
        except Exception:
            continue
    return n


def run() -> None:
    """Render the NFL Today's Games board — the MLB page mirror."""
    # CSS is already injected by Home.py and the shared board module
    # (todays_games.py imports run it at module level) — exactly the two
    # <style> blocks the MLB render emits. Calling it again here would add
    # a THIRD style block the MLB page doesn't have.

    # 1-2. Date reset (same session-state contract as MLB; MLB renders no
    # page title — the header strip is the page header)
    valid = list(utils.valid_dates("nfl"))
    valid_set = set(valid)
    if not valid:
        st.markdown(
            "<div style='font-size:1.7rem;font-weight:800;color:#E2E8F0;'>"
            "🏈 NFL — Moneyline</div>", unsafe_allow_html=True)
        st.info("No NFL per-game moneyline rows available.")
        return
    if (st.session_state.get("_nav_sport") != "nfl"
            or "selected_date" not in st.session_state):
        st.session_state["selected_date"] = (
            utils.nearest_valid_date(valid) or valid[0])
        st.session_state["_nav_sport"] = "nfl"
    date_str = st.session_state["selected_date"]
    if date_str not in valid_set:
        _render_nearest_valid_fallback(valid, date_str)
        st.stop()

    # 4. Board load (NFL artifact family)
    try:
        frame = utils.load_nfl_moneyline()
    except Exception:
        frame = pd.DataFrame()
    if frame is None or frame.empty:
        st.info("No NFL per-game moneyline rows available.")
        return
    frame = frame.dropna(subset=["home_team", "away_team"])
    day = frame[frame["game_date"].astype(str).str.replace("-", "") == date_str]
    if day.empty:
        _render_nearest_valid_fallback(valid, date_str)
        st.stop()

    # 5. Header strip — SAME markup as MLB (fed by nfl_calibration_*.json)
    cal = utils.load_calibration(date_str, sport="nfl") or {}
    record = cal.get("today_record", {}) or {}
    wins, losses = record.get("wins", 0), record.get("losses", 0)
    completed = record.get("completed", wins + losses)
    acc = (wins / completed * 100) if completed else 0.0
    league_total = cal.get("league_total", len(day))
    evening_league = cal.get("evening_games_league", _evening_count(day))
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:2px;">
          <div style="color:#94A3B8;font-size:0.9rem;">· {len(day)} of {league_total} games shown</div>
          <span style="background:rgba(59,130,246,.18);color:#93C5FD;border-radius:999px;padding:2px 10px;font-size:0.78rem;font-weight:700;">
            {evening_league} evening games begin 7 PM ET+
          </span>
          <span style="margin-left:auto;background:rgba(16,185,129,.18);color:#34D399;border-radius:999px;padding:2px 10px;font-size:0.8rem;font-weight:700;">
            ✓ {wins}-{losses} Today · {acc:.1f}% accuracy
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 6. Shared date nav (arrows + calendar + mobile rail)
    _render_date_nav(valid, date_str)

    fdate = utils.latest_artifact_date("nfl", "moneyline_json")
    tag = f"v1_{fdate}" if fdate else "—"
    # (the artifact tag stays resolved for the empty-state message below; the
    # MLB board renders no artifact caption line, so neither does NFL)

    # 7. Filter pills — same widget pair + counts math as MLB
    counts = {
        "All Games": len(day),
        "Final": int((day["game_status"] == "Final").sum()),
        "Live": int((day["game_status"] == "Live").sum()),
    }
    options = ["All Games", "Final", "Live"]
    fmt = {o: f"{o} ({counts[o]})" for o in options}
    if "nfl_game_filter" not in st.session_state:
        st.session_state["nfl_game_filter"] = "All Games"
    if hasattr(st, "pills"):
        selected = st.pills(
            "Filter", options, format_func=lambda o: fmt[o],
            key="nfl_game_filter", selection_mode="single",
            label_visibility="collapsed")
    else:
        selected = st.segmented_control(
            "Filter", options, format_func=lambda o: fmt[o],
            key="nfl_game_filter", label_visibility="collapsed")
    filtered = day
    if selected == "Final":
        filtered = day[day["game_status"] == "Final"]
    elif selected == "Live":
        filtered = day[day["game_status"] == "Live"]

    # ⟐ per-date enrichment: run-engine slate + QB matchup (NFL families)
    try:
        slate, _sdate = utils.load_nfl_run_engine_markets("nfl")
    except Exception:
        slate = pd.DataFrame()
    try:
        qb = utils.load_nfl_qb_matchup("nfl")
    except Exception:
        qb = pd.DataFrame()

    st.divider()

    # 8. Two-per-row card loop — same geometry as MLB
    for i in range(0, len(filtered), 2):
        cols = st.columns(2)
        for col, (_, g) in zip(cols, filtered.iloc[i:i + 2].iterrows()):
            with col:
                srow = _match_slate_row(g, slate)
                sel = _nfl_run_engine_selectors(g, srow) if srow is not None else None
                kw = {}
                if sel is not None:
                    kw = {"total_line": sel[0], "home_spread": sel[1],
                          "half_stop": sel[2]}
                re_html = ""
                if srow is not None:
                    try:
                        re_html = nfl_sv.runengine_html(
                            srow, str(g.get("home_team", "")),
                            str(g.get("away_team", "")), **kw)
                    except Exception:
                        re_html = ('<div class="fb-runengine">'
                                   '<span class="re-label">RUN ENGINE</span>'
                                   '<span class="re-na">n/a</span></div>')
                qb_row = None
                if qb is not None and len(qb):
                    gid = str(g.get("game_id", "") or "")
                    hit = qb[qb["game_id"].astype(str) == gid]
                    if len(hit):
                        qb_row = hit.iloc[0].to_dict()
                st.markdown(_nfl_mirror_card_html(g, qb_row, re_html),
                            unsafe_allow_html=True)
                # SAME per-card expander as MLB (📈 SHAP Features) — the
                # backend now emits nfl_shap_game_<game_id>.csv attributions
                # from the deployed ensemble, so the identical chart renders.
                _nfl_shap_expander(g)

    # 10. Same point-in-time caption
    st.caption("Model outputs are point-in-time — only data available before each "
               "game's scheduled start was used. See README for methodology.")
