"""NBA Today's Games — structural mirror of the MLB/NFL/NHL card board."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

import nba_slate_view as nba_sv
import utils
from todays_games import (_card_fav_home, _match_slate_row, _render_date_nav,
                           _render_nearest_valid_fallback, _score_side)


def _fmt(value, fmt: str = ".1f") -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return "—"


def _start_time_et(value) -> str:
    try:
        stamp = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=ZoneInfo("UTC"))
        stamp = stamp.astimezone(ZoneInfo("America/New_York"))
        return f"{stamp.hour % 12 or 12}:{stamp:%M} {stamp:%p} ET"
    except (TypeError, ValueError):
        return ""


def _widget_key(row) -> str:
    gid = str(row.get("game_id", "") or "")
    if gid:
        return gid
    date = str(row.get("game_date", "") or "")[:10].replace("-", "")
    return f"{date}_{row.get('away_team', '')}@{row.get('home_team', '')}"


def _player_box(name: str, ppg, apg) -> str:
    """One player-leader box, matching MLB's two-stat pitcher geometry."""
    name_html = f'<div class="pname">{name}</div>' if name else ""
    return (f'<div class="fb-pitcher">{name_html}<div class="pstats">'
            f'PPG {_fmt(ppg)} · APG {_fmt(apg)}</div></div>')


def _player_matchup_html(player_row) -> str:
    player_row = player_row or {}
    home = _player_box(str(player_row.get("p_home_name", "") or ""),
                       player_row.get("p_home_ppg"), player_row.get("p_home_apg"))
    away = _player_box(str(player_row.get("p_away_name", "") or ""),
                       player_row.get("p_away_ppg"), player_row.get("p_away_apg"))
    return f'<div class="fb-pitchers">{away}{home}</div>'


def _nba_run_engine_selectors(row, slate_row):
    if slate_row is None:
        return None
    fair_total = nba_sv._f(slate_row, "fair_total")
    fair_spread = nba_sv._f(slate_row, "fair_spread")
    if fair_total is None or fair_spread is None:
        return None
    gid = _widget_key(row)
    totals = [float(x) for x in range(180, 281)]
    if fair_total not in totals:
        totals.append(float(fair_total)); totals.sort()
    fair_magnitude = min(20.0, max(0.5, abs(float(fair_spread))))
    spreads = [x / 2 for x in range(1, 41)]
    if fair_magnitude not in spreads:
        spreads.append(fair_magnitude); spreads.sort()
    c_ou, c_spread = st.columns([1.35, 1], gap="small")
    with c_ou:
        total = st.selectbox("O/U line", totals, index=totals.index(float(fair_total)),
                            format_func=lambda x: f"{x:.0f}", key=f"nba_ou_{gid}",
                            label_visibility="collapsed")
    with c_spread:
        magnitude = st.selectbox("Point spread", spreads,
                                 index=spreads.index(fair_magnitude),
                                 format_func=lambda x: f"±{x:.1f}", key=f"nba_spread_{gid}",
                                 label_visibility="collapsed")
    half_stop = abs(float(magnitude) - 0.5) < 1e-9
    home_spread = None if half_stop else (-float(magnitude) if _card_fav_home(row)
                                          else float(magnitude))
    return float(total), home_spread, half_stop


def _nba_mirror_card_html(game: pd.Series, player_row=None, run_html: str = "") -> str:
    home = str(game.get("home_team", "") or "")
    away = str(game.get("away_team", "") or "")
    home_name = game.get("home_team_name", "") or home
    away_name = game.get("away_team_name", "") or away
    status = str(game.get("game_status", "") or "Scheduled")
    final, live, scheduled = status == "Final", status == "Live", status == "Scheduled"
    hs, as_ = game.get("home_score"), game.get("away_score")
    hn = None if pd.isna(hs) else int(hs)
    an = None if pd.isna(as_) else int(as_)
    hd = 0 if scheduled and hn is None else ("" if hn is None else hn)
    ad = 0 if scheduled and an is None else ("" if an is None else an)
    ph, pa = game.get("home_win_prob_model"), game.get("away_win_prob_model")
    ph = None if ph is None or pd.isna(ph) else float(ph)
    pa = None if pa is None or pd.isna(pa) else float(pa)
    if ph is None and pa is not None: ph = 1 - pa
    if pa is None and ph is not None: pa = 1 - ph
    pick = str(game.get("model_pick", "") or "")
    coin = not pick or (ph is not None and abs(ph - .5) < .02)
    winner = ""
    if hn is not None and an is not None and hn != an:
        winner = home if hn > an else away
    correct = bool(game.get("model_correct", False)) if final else False
    if final and not bool(game.get("model_correct", False)):
        correct = bool(pick and winner and pick == winner)
    day_tag = "🌙 Night Game"
    start = str(game.get("start_time_utc", "") or "")
    if start:
        try:
            if not utils._is_evening_start(start): day_tag = "☀ Day Game"
        except Exception:
            pass
    center = '<span class="fb-pill coinflip">🪙 COIN FLIP</span>' if coin and final else ""
    if live: right = '<span class="fb-pill live">● LIVE</span>'
    elif scheduled: right = '<span class="fb-pill final">PRE-GAME</span>'
    elif final:
        pill = ('<span class="fb-pill correct">✓ CORRECT PICK</span>' if correct
                else '<span class="fb-pill miss">X MISS</span>') if not coin else ""
        right = pill + '<span class="fb-pill final">FINAL</span>'
    else: right = ""
    top = f'<span class="fb-tag">{day_tag}</span>{center}<span class="spacer"></span>{right}'
    mid = (_start_time_et(start) or "PREGAME") if scheduled else ("F" if final else "LIVE")
    score = (f'<div class="fb-score">{_score_side(ad, away, winner == away)}'
             f'<span class="mid">{mid}</span>{_score_side(hd, home, winner == home)}</div>')

    def team_row(name, team, record, probability, selected, is_home):
        pct = "—" if probability is None else f"{probability:.0%}"
        width = 0 if probability is None else int(round(probability * 100))
        color = utils.PRIMARY if probability is not None and probability >= .5 else (
            "#38BDF8" if probability is not None else "#334155")
        pick_badge = '<span class="fb-pill pick">PICK</span>' if selected else ""
        home_tag = '<span class="fb-tag" style="font-size:0.7rem;">HOME</span>' if is_home else ""
        return (f'<div class="fb-team"><span class="fb-accent" style="background:{color};"></span>'
                f'<div><span class="name">{name}</span> <span class="sub">{team} {record or ""}</span> '
                f'{home_tag} {pick_badge}</div><span class="pct" style="color:{color};">{pct}</span></div>'
                f'<div class="fb-bar"><div class="fill" style="width:{width}%;background:{color};"></div></div>')

    home_row = team_row(home_name, home, game.get("home_record", ""), ph,
                        pick == home, True)
    away_row = team_row(away_name, away, game.get("away_record", ""), pa,
                        pick == away, False)
    pregame = (f'<div class="fb-pregame">Pre-game: {away} {_fmt(pa, ".0%")} vs '
               f'{home} {_fmt(ph, ".0%")}</div>')
    player_block = _player_matchup_html(player_row)
    venue_time = _start_time_et(start)
    venue = (f'<div class="fb-venue">📍 {game.get("venue", "") or "—"}'
             f'{f" · {venue_time}" if venue_time else ""}</div>')
    if scheduled:
        banner = f'<div class="fb-banner blue">⏳ Pre-game{(" — " + venue_time) if venue_time else ""} · prediction locked at tipoff</div>'
    elif live:
        banner = '<div class="fb-banner blue">● LIVE — game in progress</div>'
    elif coin:
        banner = f'<div class="fb-banner amber">🪙 {winner} Won — Coin Flip Game (50/50)</div>'
    elif correct:
        banner = f'<div class="fb-banner green">✓ {winner} Won — Model Correct</div>'
    else:
        banner = f'<div class="fb-banner red">X {winner} Won — Model picked {pick}</div>'
    return (f'<div class="fb-card"><div class="fb-top">{top}</div>{score}{home_row}{away_row}'
            f'{pregame}{player_block}{venue}{run_html}{banner}</div>')


def _nba_shap_expander(row) -> None:
    gid = _widget_key(row)
    date = str(row.get("game_date", "") or "").replace("-", "")
    with st.expander(f"📈 SHAP Features — {gid}", expanded=False):
        frame = utils.load_shap(gid, date, sport="nba")
        if frame.empty:
            st.caption("No SHAP file found for this game.")
            return
        chart = utils.shap_chart(frame)
        if chart is not None:
            utils.show_chart(chart)
        st.caption("Positive values increase the favored team's win probability; negative values decrease it.")


def _evening_count(frame: pd.DataFrame) -> int:
    count = 0
    for _, row in frame.iterrows():
        try:
            count += int(utils._is_evening_start(str(row.get("start_time_utc", "") or "")))
        except Exception:
            pass
    return count


def run() -> None:
    valid = list(utils.valid_dates("nba"))
    if not valid:
        st.markdown("<div style='font-size:1.7rem;font-weight:800;color:#E2E8F0;'>🏀 NBA — Moneyline</div>", unsafe_allow_html=True)
        st.info("No NBA per-game moneyline rows available.")
        return
    if st.session_state.get("_nav_sport") != "nba" or "selected_date" not in st.session_state:
        st.session_state["selected_date"] = utils.nearest_valid_date(valid) or valid[0]
        st.session_state["_nav_sport"] = "nba"
    date = st.session_state["selected_date"]
    if date not in set(valid):
        _render_nearest_valid_fallback(valid, date)
        st.stop()
    try:
        frame = utils.load_nba_moneyline("nba")
    except Exception:
        frame = pd.DataFrame()
    frame = frame.dropna(subset=["home_team", "away_team"]) if frame is not None else pd.DataFrame()
    day = frame[frame.game_date.astype(str).str.replace("-", "") == date] if len(frame) else pd.DataFrame()
    history = False
    if day.empty:
        try:
            day = utils.load_nba_history_games(date)
        except Exception:
            day = pd.DataFrame()
        history = not day.empty
    if day.empty:
        _render_nearest_valid_fallback(valid, date)
        st.stop()
    cal = utils.load_calibration(date, sport="nba") or {}
    record = cal.get("today_record", {}) or {}
    wins, losses = record.get("wins", 0), record.get("losses", 0)
    completed = record.get("completed", wins + losses)
    accuracy = wins / completed * 100 if completed else 0
    st.markdown(f"<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:2px;'><div style='color:#94A3B8;font-size:0.9rem;'>· {len(day)} of {cal.get('league_total', len(day))} games shown</div><span style='background:rgba(59,130,246,.18);color:#93C5FD;border-radius:999px;padding:2px 10px;font-size:0.78rem;font-weight:700;'>{cal.get('evening_games_league', _evening_count(day))} evening games begin 7 PM ET+</span><span style='margin-left:auto;background:rgba(16,185,129,.18);color:#34D399;border-radius:999px;padding:2px 10px;font-size:0.8rem;font-weight:700;'>✓ {wins}-{losses} Today · {accuracy:.1f}% accuracy</span></div>", unsafe_allow_html=True)
    _render_date_nav(valid, date)
    counts = {"All Games": len(day), "Final": int((day.game_status == "Final").sum()), "Live": int((day.game_status == "Live").sum())}
    options = ["All Games", "Final", "Live"]
    if "nba_game_filter" not in st.session_state: st.session_state["nba_game_filter"] = "All Games"
    if hasattr(st, "pills"):
        selected = st.pills("Filter", options, format_func=lambda x: f"{x} ({counts[x]})", key="nba_game_filter", selection_mode="single", label_visibility="collapsed")
    else:
        selected = st.segmented_control("Filter", options, format_func=lambda x: f"{x} ({counts[x]})", key="nba_game_filter", label_visibility="collapsed")
    filtered = day[day.game_status == selected] if selected in {"Final", "Live"} else day
    try: slate, _ = utils.load_nba_run_engine_markets("nba")
    except Exception: slate = pd.DataFrame()
    try: players = utils.load_nba_player_matchup("nba")
    except Exception: players = pd.DataFrame()
    if history:
        st.info("🗂 Archive view — this historical NBA card serves the production prediction as first published; current-slate market and SHAP enrichment is unavailable for this retained date.")
    st.divider()
    for start in range(0, len(filtered), 2):
        columns = st.columns(2)
        for column, (_, game) in zip(columns, filtered.iloc[start:start + 2].iterrows()):
            with column:
                slate_row = _match_slate_row(game, slate)
                selection = _nba_run_engine_selectors(game, slate_row)
                run_html = ""
                if slate_row is not None:
                    kwargs = ({"total_line": selection[0], "home_spread": selection[1], "half_stop": selection[2]}
                              if selection else {})
                    try: run_html = nba_sv.runengine_html(slate_row, str(game.get("home_team", "")), str(game.get("away_team", "")), **kwargs)
                    except Exception: run_html = '<div class="fb-runengine"><span class="re-label">RUN ENGINE</span><span class="re-na">n/a</span></div>'
                player_row = None
                if players is not None and len(players):
                    hit = players[players.game_id.astype(str) == str(game.get("game_id", ""))]
                    if len(hit): player_row = hit.iloc[0].to_dict()
                st.markdown(_nba_mirror_card_html(game, player_row, run_html), unsafe_allow_html=True)
                _nba_shap_expander(game)
    st.caption("Model outputs are point-in-time — only data available before each game's scheduled tipoff was used. See README for methodology.")


if __name__ == "__main__":
    run()
