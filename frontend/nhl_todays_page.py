"""NHL Today's Games — STRUCTURAL 1:1 MIRROR of the MLB Today's Games page
(the MLB board inside ``frontend/todays_games.py``), with NHL artifacts
substituted. Built as a page module exactly like the verified
``nfl_todays_page.py`` mirror; ``todays_games.py``'s NHL branch delegates
here (one dispatch line) so MLB behavior is untouched.

ANATOMY CHECKLIST (MLB render order — mirrored exactly; documented swaps
marked ⟐):

  1. Board title + artifact caption (same sentence shape; ⟐ NHL moneyline
     v1 tag instead of the MLB snapshot date).
  2. Date reset / first visit → nearest valid date to today (ET) — SHARED
     helpers (utils.nearest_valid_date, same session-state keys).
  3. Missing-date fallback → the SHARED `_render_nearest_valid_fallback`
     (imported — identical behavior, no duplicate logic).
  4. Board load: ⟐ ``utils.load_nhl_moneyline()`` filtered to the selected
     game date; past dates rebuild from the frozen production card store
     (the v1-history path — the moneyline JSON is current-slate-only).
  5. Header strip: `· N of M games shown` + evening-games pill +
     `✓ W-L Today · acc% accuracy` badge — SAME markup, fed by ⟐
     ``utils.load_calibration(date, sport='nhl')`` (nhl_calibration_*.json;
     its record shape has no today_record yet → honest zeros, identical
     markup); evening count computed from the start times with the shared
     ``utils._is_evening_start`` convention.
  6. ``_render_date_nav`` — the SHARED date navigation (arrows + calendar
     + mobile rail), imported, not duplicated.
  7. Filter pills `All Games (n) / Final (n) / Live (n)` — same
     ``st.pills``/``segmented_control`` fallback pair, same session key
     shape (namespaced ``nhl_game_filter``), same counts math.
  8. ``st.divider()`` → two-per-row card loop — same columns(2), same
     iteration, same per-card flow: ⟐ run-engine O/U + spread selectors
     (the NHL-sized ``_nhl_run_engine_selectors`` seam — totals 4..12.5,
     spreads ±0.5..±8.0, ``nhl_*`` widget keys) → card HTML → the SAME
     per-card SHAP expander as MLB (``_nhl_shap_expander``; the backend
     emits ``nhl_shap_game_<game_id>.csv`` attributions from the deployed
     ensemble's tree members).
  9. Card (⟐ ``_nhl_mirror_card_html``): top badge strip (☀/🌙 + LIVE/
     PRE-GAME/FINAL + ✓/X pills) → scoreboard (winner bars; PRE-GAME
     renders the 0-0 display exactly like MLB) → team rows (records, PICK
     badge, trophy, prob bars) → `fb-pregame` line → ⟐ GOALIE-matchup twin
     boxes (the ``fb-pitchers`` grid, MLB's TWO-stat single-line format:
     SV% · GAA — the ERA/K-9 analogs) → ``fb-venue`` (📍 arena · puck drop
     ET) → RUN ENGINE strip (``nhl_slate_view.runengine_html``: Proj / O/U
     / RL — no separate ML span, matching MLB's strip anatomy) → banner
     (⟐ puck-drop wording).
  10. Point-in-time caption — SAME wording.

Empty / missing states: no board → the shared notice; no games on the
selected date → the shared nearest-valid fallback; missing goalie record →
each goalie box renders '—' quietly (never fabricated); missing run-engine
slate row → the existing quiet 'n/a' strip.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

import nhl_slate_view as nhl_sv
import utils

# The shared render helpers the MLB board uses (imported — never duplicated).
from todays_games import (  # noqa: E402
    _card_fav_home,
    _match_slate_row,
    _render_date_nav,
    _render_nearest_valid_fallback,
    _score_side,
)


# ---------------------------------------------------------------------------
# ⟐ Goalie-matchup card blocks (the pitcher-card substitute)
# ---------------------------------------------------------------------------

def _fmt(v, fmt: str) -> str:
    """Numeric render or '—' (the artifact never fabricates)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return "—"


def _goalie_box(name: str, sv_pct, gaa) -> str:
    """One side of the goalie matchup — ``_pitcher_box`` byte-format: bold
    name line + ONE stats line with the same ' · ' separator (MLB shows
    exactly two stats: ERA · K/9; the goalie pair is SV% and GAA — SV% is
    the ERA analog, GAA the K/9 analog of the card's two-stat line). The
    extra ``starts`` column stays in the artifact; the box renders the
    two-stat line so the card geometry (single pstats line, same height)
    matches MLB exactly."""
    name_html = f'<div class="pname">{name}</div>' if name else ""
    stats = (
        f"SV% {_fmt(sv_pct, '.3f')} · GAA {_fmt(gaa, '.2f')}"
    )
    return (f'<div class="fb-pitcher">{name_html}'
            f'<div class="pstats">{stats}</div></div>')


def _goalie_matchup_html(goalie_row) -> str:
    """The `fb-pitchers` twin grid with goalie boxes. A missing/unresolvable
    starter renders the quiet '—' box (same honesty rule as MLB's TBD
    pitcher)."""
    if goalie_row is None:
        goalie_row = {}
    home = _goalie_box(
        str(goalie_row.get("g_home_name", "") or ""),
        goalie_row.get("g_home_sv_pct"), goalie_row.get("g_home_gaa"))
    away = _goalie_box(
        str(goalie_row.get("g_away_name", "") or ""),
        goalie_row.get("g_away_sv_pct"), goalie_row.get("g_away_gaa"))
    return f'<div class="fb-pitchers">{away}{home}</div>'


def _start_time_et(value) -> str:
    """Render an ISO start stamp as Eastern wall-clock time.

    NHL API ``start_time_utc`` is a genuine UTC instant (unlike NFL's
    already-ET gametime string), so this CONVERTS — the same convention
    ``utils._is_evening_start`` uses for the evening pill."""
    try:
        ts = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=ZoneInfo("UTC"))
        et = ts.astimezone(ZoneInfo("America/New_York"))
        return f"{et.hour % 12 or 12}:{et:%M} {et:%p} ET"
    except (ValueError, TypeError):
        return ""


def _widget_key(r) -> str:
    """Stable per-game widget key: game_id when present, else a composite."""
    gid = str(r.get("game_id", "") or "")
    if gid:
        return gid
    d = str(r.get("game_date", "") or "")[:10].replace("-", "")
    return f"{d}_{r.get('away_team', '')}@{r.get('home_team', '')}"


def _nhl_run_engine_selectors(r, srow):
    """Per-card O/U + run-line selectors for a game with a slate row.

    Mirror of the NFL card flow (``todays_games._nfl_run_engine_selectors``)
    with NHL-SIZED ladders: totals 4.0..12.5 and spreads ±0.5..±8.0 (the
    NFL 24..66.5 / ±0.5..±14.0 ladders would price a hockey game off a
    basketball-shaped grid). The strip prices the SELECTED grid line
    (defaults: the model's fair total, and the fair home spread), and the
    run-line dropdown includes the ±0.5 stop (favorite-anchored raw cover +
    derived-ML pair — they diverge by the regulation-tie rate on NHL).
    Returns (total_line, home_spread, half_stop) or None when the slate row
    lacks the fair-line columns. Keys are namespaced nhl_* so they never
    collide with the MLB/NFL card widgets. Model probabilities only — no
    offered lines.
    """
    if srow is None:
        return None
    fair_total = nhl_sv._f(srow, "fair_total")
    fair_spread = nhl_sv._f(srow, "fair_spread")
    if fair_total is None or fair_spread is None:
        return None
    gid = _widget_key(r)
    # NHL artifact carries integer-support distributions; half-point lines
    # are priced from the adjacent integer threshold in nhl_slate_view.
    totals = [float(v) / 2.0 for v in range(8, 26)]  # 4.0 … 12.5
    fair_total_value = float(fair_total)
    if fair_total_value not in totals:
        totals.append(fair_total_value)
        totals.sort()
    fair_home = -float(round(fair_spread))
    spread_options = [float(v) / 2.0 for v in range(1, 17)]  # ±0.5 … ±8.0
    fair_magnitude = float(abs(fair_home))
    if fair_magnitude not in spread_options:
        spread_options.append(fair_magnitude)
        spread_options.sort()
    c_ou, c_rl = st.columns([1.35, 1], gap="small")
    with c_ou:
        total_line = st.selectbox(
            "O/U line", totals, index=totals.index(fair_total_value),
            format_func=lambda u: (f"{u:.1f}" if float(u) % 1 else f"{int(u)}"),
            key=f"nhl_ou_{gid}", label_visibility="collapsed",
            help=("Totals line to price this game at — defaults to the "
                  "model's fair total; 0.5-goal increments are priced from "
                  "the NHL score distribution."))
    with c_rl:
        picked = st.selectbox(
            "Run line", spread_options,
            index=spread_options.index(fair_magnitude),
            format_func=lambda v: f"±{v:.1f}",
            key=f"nhl_rl_{gid}", label_visibility="collapsed",
            help=("Run-line pair to price this game at — defaults to the "
                  "fair home spread. 0.5-goal increments are available. At "
                  "±0.5, the run-ML pair excludes regulation ties and is "
                  "separate from the binary moneyline at the top of the "
                  "card."))
    half_stop = abs(float(picked) - 0.5) < 1e-9
    if half_stop:
        home_spread = None
    else:
        # The selector is a combined ± magnitude; orient the pair so the
        # moneyline favorite is the negative side, matching MLB's card.
        home_spread = -float(picked) if _card_fav_home(r) else float(picked)
    return float(total_line), home_spread, half_stop


# ---------------------------------------------------------------------------
# ⟐ NHL card — the MLB _card_html mirror (same section order / classes)
# ---------------------------------------------------------------------------

def _nhl_mirror_card_html(g: pd.Series, goalie_row=None, re_html: str = "") -> str:
    """MLB card chrome over the NHL row (the structural 1:1 render; the
    goalie twin boxes substitute the QB boxes)."""
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
    # the shared 0-0 display — the banner still says locked at puck drop).
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
    if start_iso:
        try:
            if not utils._is_evening_start(start_iso):
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

    # --- scoreboard (winner bars; scheduled shows puck drop like MLB) ---
    if is_scheduled:
        mid = _start_time_et(start_iso) or "PREGAME"
    else:
        mid = "F" if is_final else "LIVE"
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

    goalie_block = _goalie_matchup_html(goalie_row)
    start_et = _start_time_et(start_iso)
    venue = (f'<div class="fb-venue">📍 {g.get("venue", "") or "—"}'
             f'{f" · {start_et}" if start_et else ""}</div>')
    banner = ''
    if is_scheduled:
        first = _start_time_et(start_iso)
        suffix = f" — {first}" if first else ""
        banner = (f'<div class="fb-banner blue">⏳ Pre-game{suffix} · '
                  f'prediction locked at puck drop</div>')
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
        f'{home_row}{away_row}{pregame}{goalie_block}{venue}{re_html}{banner}</div>'
    )


def _nhl_shap_expander(r) -> None:
    """NHL SHAP accordion — the exact MLB per-card accordion (same expander
    title, chart, empty state and caption) keyed by the NHL game id. The
    backend emits per-game attributions from the deployed ensemble's tree
    members (nhl_shap_game_<game_id>.csv), so the shared loader resolves the
    real file and the identical MLB chart renders."""
    gid = _widget_key(r)
    date_str = str(r.get("game_date", "") or "").replace("-", "")
    with st.expander(f"📈 SHAP Features — {gid}", expanded=False):
        shap_df = utils.load_shap(gid, date_str, sport="nhl")
        if shap_df.empty:
            st.caption("No SHAP file found for this game.")
            return
        chart = utils.shap_chart(shap_df)
        if chart is not None:
            utils.show_chart(chart)
        # SHAP is aligned to the FAVORED team (backend negates values when
        # the away team is favored) — same caption semantics as MLB, with
        # the NHL ensemble's member list.
        persp = ""
        if "perspective_team" in shap_df.columns:
            pt = shap_df["perspective_team"].dropna().astype(str)
            pt = pt[pt.str.strip() != ""].iloc[0] if not pt.empty else ""
            if pt and pt not in ("HOME", "AWAY"):
                persp = f" · Viewing from {pt}'s perspective"
        st.caption("Positive values increase the favored team's win probability; "
                   "negative decrease it. Averaged across the XGBoost / LightGBM / "
                   f"elastic-net ensemble.{persp}")


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
    """Render the NHL current slate or an explicitly selected archive date."""
    # CSS is already injected by Home.py and the shared board module
    # (todays_games.py imports run it at module level) — exactly the two
    # <style> blocks the MLB render emits. Calling it again here would add
    # a THIRD style block the MLB page doesn't have.

    # Resolve the live/future board first.  It is the only source allowed to
    # choose the initial date; history can extend the explicit archive rail
    # but can never substitute for a missing current slate.
    try:
        current = utils.load_nhl_moneyline("nhl")
    except Exception:
        current = pd.DataFrame()
    if current is not None and not current.empty:
        current = current.dropna(subset=["home_team", "away_team"])
    current_dates = set(utils._distinct_game_dates(current))
    current_date = (max(current_dates) if current_dates else None)

    try:
        current_record = utils.load_nhl_moneyline_record("nhl")
    except Exception:
        current_record = {}
    current_slate_date = None
    if isinstance(current_record, dict):
        try:
            current_slate_date = datetime.strptime(
                str(current_record.get("slate_date", "")), "%Y-%m-%d"
            ).strftime("%Y%m%d")
        except (TypeError, ValueError):
            current_slate_date = None
    # The record's explicit slate_date is authoritative.  The per-game dates
    # remain a consistency fallback for a valid but unusual persisted frame.
    if current_slate_date:
        current_date = current_slate_date

    valid = list(utils.valid_dates("nhl"))
    valid_set = set(valid)
    if current_date and current_date not in valid_set:
        valid_set.add(current_date)
        valid.append(current_date)
        valid.sort(reverse=True)

    today_et = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")
    first_visit = (st.session_state.get("_nav_sport") != "nhl"
                   or "selected_date" not in st.session_state)
    if first_visit:
        # A valid future preview (for example a Sep 29 slate published Sep 24)
        # is the current board.  Only when none exists do we honestly target
        # today; nearest/stale-date substitution is forbidden.
        st.session_state["selected_date"] = current_date or today_et
        st.session_state["_nav_sport"] = "nhl"
    date_str = st.session_state["selected_date"]

    if current_date is None and date_str == today_et:
        st.markdown(
            "<div style='font-size:1.7rem;font-weight:800;color:#E2E8F0;'>"
            "🏒 NHL — Moneyline</div>", unsafe_allow_html=True)
        st.info(
            "No confirmed current NHL slate artifact is available. "
            "Historical predictions are not shown as a live board; choose an "
            "archive date below when one is available."
        )
        archive_dates = [d for d in valid if d < today_et]
        if archive_dates:
            # The empty current state has no valid current key.  Seed the
            # first explicitly historical choice; every later selection is an
            # intentional archive action through the shared date controls.
            # ``date_str`` must track the seeded value or the board below
            # would keep testing TODAY and dead-end on its own warning.
            st.session_state["selected_date"] = archive_dates[0]
            date_str = archive_dates[0]
        else:
            st.session_state["selected_date"] = today_et
            if valid:
                _render_date_nav(valid, date_str)
            return

    if date_str not in valid_set:
        st.warning(
            f"No archived NHL card is available for "
            f"{utils.format_date_long(date_str)} ({date_str})."
        )
        if valid:
            _render_date_nav(valid, date_str)
        return

    history_view = date_str not in current_dates
    if not history_view:
        day = current[current["game_date"].astype(str)
                         .str.replace("-", "") == date_str]
    else:
        # Explicit historical selection only.  The shared loader is frozen
        # production-card-store first; OOF history is a legacy/preseed fallback
        # and is never consulted for a current or future date.
        try:
            day = utils.load_history_games_v1(date_str, "nhl")
        except Exception:
            day = pd.DataFrame()
    if day is None or day.empty:
        st.info(
            f"No archived NHL card is available for "
            f"{utils.format_date_long(date_str)} ({date_str})."
        )
        _render_date_nav(valid, date_str)
        return

    # 5. Header strip — SAME markup as MLB (fed by nhl_calibration_*.json)
    cal = utils.load_calibration(date_str, sport="nhl") or {}
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

    fdate = utils.latest_artifact_date("nhl", "moneyline_json")
    tag = f"v1_{fdate}" if fdate else "—"
    # (the artifact tag stays resolved for the empty-state message below; the
    # MLB board renders no artifact caption line, so neither does NHL)

    # 7. Filter pills — same widget pair + counts math as MLB
    counts = {
        "All Games": len(day),
        "Final": int((day["game_status"] == "Final").sum()),
        "Live": int((day["game_status"] == "Live").sum()),
    }
    options = ["All Games", "Final", "Live"]
    fmt = {o: f"{o} ({counts[o]})" for o in options}
    if "nhl_game_filter" not in st.session_state:
        st.session_state["nhl_game_filter"] = "All Games"
    if hasattr(st, "pills"):
        selected = st.pills(
            "Filter", options, format_func=lambda o: fmt[o],
            key="nhl_game_filter", selection_mode="single",
            label_visibility="collapsed")
    else:
        selected = st.segmented_control(
            "Filter", options, format_func=lambda o: fmt[o],
            key="nhl_game_filter", label_visibility="collapsed")
    filtered = day
    if selected == "Final":
        filtered = day[day["game_status"] == "Final"]
    elif selected == "Live":
        filtered = day[day["game_status"] == "Live"]

    # Current-slate enrichment only.  Frozen archive cards never inherit a
    # later run's market/goalie/SHAP artifacts.
    slate = pd.DataFrame()
    goalies = pd.DataFrame()
    if not history_view:
        try:
            slate, _sdate = utils.load_nhl_run_engine_markets("nhl")
        except Exception:
            slate = pd.DataFrame()
        try:
            goalies = utils.load_nhl_goalie_matchup("nhl")
        except Exception:
            goalies = pd.DataFrame()
    else:
        st.info(
            "🗂 Archive view — this historical NHL card preserves the production "
            "prediction as first published. Current-slate market, goalie, and "
            "SHAP enrichment is intentionally unavailable for archive dates."
        )

    st.divider()

    # 8. Two-per-row card loop — same geometry as MLB
    for i in range(0, len(filtered), 2):
        cols = st.columns(2)
        for col, (_, g) in zip(cols, filtered.iloc[i:i + 2].iterrows()):
            with col:
                srow = _match_slate_row(g, slate)
                sel = _nhl_run_engine_selectors(g, srow) if srow is not None else None
                kw = {}
                if sel is not None:
                    kw = {"total_line": sel[0], "home_spread": sel[1],
                          "half_stop": sel[2]}
                re_html = ""
                if srow is not None:
                    try:
                        re_html = nhl_sv.runengine_html(
                            srow, str(g.get("home_team", "")),
                            str(g.get("away_team", "")), **kw)
                    except Exception:
                        re_html = ('<div class="fb-runengine">'
                                   '<span class="re-label">RUN ENGINE</span>'
                                   '<span class="re-na">n/a</span></div>')
                goalie_row = None
                if goalies is not None and len(goalies):
                    gid = str(g.get("game_id", "") or "")
                    hit = goalies[goalies["game_id"].astype(str) == gid]
                    if len(hit):
                        goalie_row = hit.iloc[0].to_dict()
                st.markdown(_nhl_mirror_card_html(g, goalie_row, re_html),
                            unsafe_allow_html=True)
                # SAME current-card expander as MLB (📈 SHAP Features).  Frozen
                # archive cards intentionally omit later-run SHAP attribution.
                if not history_view:
                    _nhl_shap_expander(g)

    # 10. Same point-in-time caption
    st.caption("Model outputs are point-in-time — only data available before each "
               "game's scheduled start was used. See README for methodology.")
