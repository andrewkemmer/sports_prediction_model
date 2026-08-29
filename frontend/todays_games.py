"""Page 1 — Today's Games.

Renders the daily game board: date ribbon with prev/next navigation, filter
pills (All Games / Final / Live), the accuracy badge, and two-column game
cards matching the reference dashboard (badge strips, probability bars,
pitcher panels, ML/edge bar, collapsible SHAP features, outcome banner).

Each card is enriched with run-engine projections from
run_engine_markets_<date>.csv (slate rows joined by game_id == game_pk):
projected team runs, the 8.5 O/U probability split, and the −1.5/+1.5 run
line — quiet 'n/a' when the line grid is missing, never fabricated.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import utils

utils.inject_css()


# ===========================================================================
# Card builders
# ===========================================================================

def _accent(team: str, winner: str, coin: bool, pick: str, live: bool) -> str:
    if coin and not live:
        return utils.BLUE
    if winner == team:
        return utils.PRIMARY
    if live and pick == team:
        return utils.BLUE
    return utils.RED


def _pct_color(team: str, winner: str, pick: str, coin: bool) -> str:
    if coin:
        return utils.BLUE
    if winner == team:
        return utils.PRIMARY
    if pick == team:
        return "#F87171"
    return "#94A3B8"


def _bar_color(team: str, winner: str, pick: str, coin: bool) -> str:
    if coin:
        return utils.BLUE
    if winner == team:
        return utils.PRIMARY
    if pick == team:
        return "#EF4444"
    return "#334155"


def _team_row(name, team, rec, prob, accent, pct_color, bar_color, picked, trophy, is_home) -> str:
    pick_badge = '<span class="fb-pill pick">PICK</span>' if picked else ""
    trophy_html = " 🏆" if trophy else ""
    home_tag = '<span class="fb-tag" style="font-size:0.7rem;">HOME</span>' if is_home else ""
    return (
        f'<div class="fb-team">'
        f'<span class="fb-accent" style="background:{accent};"></span>'
        f'<div><span class="name">{name}</span> <span class="sub">{team} {rec}</span> '
        f'{home_tag} {pick_badge}{trophy_html}</div>'
        f'<span class="pct" style="color:{pct_color};">{prob:.0%}</span></div>'
        f'<div class="fb-bar"><div class="fill" style="width:{prob:.0%};background:{bar_color};"></div></div>'
    )


def _banner_html(status, is_final, is_live, winner, pick, correct, coin, upset, home_team, away_team, g) -> str:
    winner_name = g.get("home_team_name", home_team) if winner == home_team else (
        g.get("away_team_name", away_team) if winner == away_team else "")
    if status == "Scheduled":
        first_pitch = utils.start_time_et(g.get("start_time_utc", ""))
        suffix = f" — {first_pitch}" if first_pitch else ""
        return f'<div class="fb-banner blue">⏳ Pre-game{suffix} · prediction locked at first pitch</div>'
    if is_live:
        leader = winner or "—"
        suffix = f" · {leader} leading" if leader != "—" else ""
        return (f'<div class="fb-banner blue">● LIVE — {g.get("final_inning", "In progress")}{suffix}</div>')
    if coin:
        return f'<div class="fb-banner amber">🪙 {winner_name} Won — Coin Flip Game (50/50)</div>'
    if correct:
        return f'<div class="fb-banner green">✓ {winner_name} Won — Model Correct</div>'
    if upset:
        return f'<div class="fb-banner red">X {winner_name} Won — Upset! Model picked {pick}</div>'
    return f'<div class="fb-banner red">X {winner_name} Won — Model picked {pick}</div>'


def _val(row: pd.Series, *keys: str, default: str = "—") -> str:
    """First non-empty value across column-name aliases (artifact schema
    changed spellings over time — try each)."""
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in ("nan", "none", "tbd"):
            return s
    return default


def _pitcher_box(name: str, era: str, k9: str) -> str:
    name_html = f'<div class="pname">{name}</div>' if name else ""
    return (f'<div class="fb-pitcher">{name_html}'
            f'<div class="pstats">ERA {era} · K/9 {k9}</div></div>')


def _runengine_html(bits, home_team: str, away_team: str) -> str:
    """Run-engine strip on the game card — projections, 8.5 O/U, run line.

    bits comes from market_diagnostics.run_engine_card_bits; None means no
    slate row for this game (strip omitted), has_grid=False renders a quiet
    'n/a'. The O/U split is priced at the game's OWN rounded total
    (bits["total_line"], nearest 0.5 of λ_home + λ_away) and notes when the
    line was clamped to the shipped grid. Away +1.5 is the exact complement
    of home −1.5 (the artifact ships home-cover columns only) — labeled as
    such.
    """
    if bits is None:
        return ""
    if not bits.get("has_grid"):
        return ('<div class="fb-runengine"><span class="re-label">'
                'RUN ENGINE</span><span class="re-na">n/a</span></div>')
    ou_label = (f'O/U {bits["total_line"]:.1f}'
                + (" (clamped)" if bits.get("clamped") else ""))
    return (
        f'<div class="fb-runengine">'
        f'<span class="re-label">RUN ENGINE</span>'
        f'<span>Proj: {away_team} {bits["proj_away"]:.1f} – '
        f'{home_team} {bits["proj_home"]:.1f}</span>'
        f'<span>{ou_label}: Over {bits["p_over"]:.0%}'
        + (f' / Push {(bits.get("p_push") or 0):.0%}'
           if (bits.get("p_push") or 0) > 0.005 else '')
        + f' / Under {bits["p_under"]:.0%}</span>'
        f'<span>RL: {home_team} −1.5 {bits["p_home_cover"]:.0%} · '
        f'{away_team} +1.5 {bits["p_away_cover"]:.0%} (complement)</span>'
        f'</div>'
    )


def _score_side(num, abbr: str, is_winner: bool) -> str:
    """Score column: green vertical bar to the left of the winning score
    (matches the reference dashboard), white number, abbreviation below."""
    bar = f'<span class="win-bar" style="background:{utils.PRIMARY};"></span>' if is_winner else ""
    return (f'<div class="side"><div class="num-wrap">{bar}'
            f'<span class="num">{num}</span></div>'
            f'<div class="abbr">{abbr}</div></div>')


def _card_html(g: pd.Series, re_bits=None) -> str:
    home_team, away_team = g["home_team"], g["away_team"]
    home_name = g.get("home_team_name", "") or home_team
    away_name = g.get("away_team_name", "") or away_team
    status = g["game_status"]
    is_final = status == "Final"
    is_live = status == "Live"
    is_scheduled = status == "Scheduled"

    hs = g.get("home_score")
    as_ = g.get("away_score")
    h_score_n = None if pd.isna(hs) else int(hs)
    a_score_n = None if pd.isna(as_) else int(as_)
    h_disp = "" if h_score_n is None else h_score_n
    a_disp = "" if a_score_n is None else a_score_n

    p_home, p_away = float(g["home_win_prob_model"]), float(g["away_win_prob_model"])
    pick = g.get("model_pick", "") or ""
    is_coin_flip = bool(g.get("is_coin_flip", False)) or pick == ""
    is_upset = bool(g.get("is_upset", False))

    winner = ""
    if h_score_n is not None and a_score_n is not None and h_score_n != a_score_n:
        winner = home_team if h_score_n > a_score_n else away_team
    correct = bool(g.get("model_correct", False)) if is_final else False

    # --- top badge strip ---
    day_tag = "☀ Day Game" if g.get("day_game") else "🌙 Night Game"
    center_pill, right_pills = "", ""
    if is_coin_flip and is_final:
        center_pill = '<span class="fb-pill coinflip">🪙 COIN FLIP</span>'
    elif is_upset and is_final:
        center_pill = '<span class="fb-pill upset">⚡ UPSET</span>'
    if is_live:
        right_pills = '<span class="fb-pill live">● LIVE</span>'
    elif is_scheduled:
        right_pills = '<span class="fb-pill final">PRE-GAME</span>'
    elif is_final:
        correct_pill = '' if is_coin_flip else (
            '<span class="fb-pill correct">✓ CORRECT PICK</span>' if correct else '<span class="fb-pill miss">X MISS</span>'
        )
        inning = g.get("final_inning", "F")
        if inning not in ("", "F"):
            final_pill = f'<span class="fb-pill final">FINAL ({inning})</span>'
        else:
            final_pill = '<span class="fb-pill final">FINAL</span>'
        right_pills = correct_pill + final_pill

    top = (
        f'<span class="fb-tag">{day_tag}</span>{center_pill}'
        f'<span class="spacer"></span>{right_pills}'
    )

    # --- scoreboard ---
    if is_scheduled:
        mid = utils.start_time_et(g.get("start_time_utc", "")) or "PREGAME"
    else:
        mid = "F" if is_final else g.get("final_inning", "LIVE")
        if is_final and g.get("final_inning", "F") not in ("", "F"):
            mid = f"F/{g['final_inning'].lstrip('F/')}"
        if is_live:
            raw = g.get("final_inning", "LIVE").lstrip("L")
            mid = f"L{raw}" if raw else "LIVE"
    score = (
        f'<div class="fb-score">'
        f'{_score_side(a_disp, away_team, is_winner=(winner == away_team))}'
        f'<span class="mid">{mid}</span>'
        f'{_score_side(h_disp, home_team, is_winner=(winner == home_team))}'
        f'</div>'
    )

    # --- team rows + probability bars ---
    home_row = _team_row(
        home_name, home_team, g.get("home_record", ""), p_home,
        _accent(home_team, winner, is_coin_flip, pick, is_live),
        _pct_color(home_team, winner, pick, is_coin_flip),
        _bar_color(home_team, winner, pick, is_coin_flip),
        picked=(pick == home_team), trophy=(winner == home_team and is_final), is_home=True,
    )
    away_row = _team_row(
        away_name, away_team, g.get("away_record", ""), p_away,
        _accent(away_team, winner, is_coin_flip, pick, is_live),
        _pct_color(away_team, winner, pick, is_coin_flip),
        _bar_color(away_team, winner, pick, is_coin_flip),
        picked=(pick == away_team), trophy=(winner == away_team and is_final), is_home=False,
    )

    pregame = f'<div class="fb-pregame">Pre-game: {home_team} {p_home:.0%} vs {away_team} {p_away:.0%}</div>'

    # --- pitchers / venue / odds ---
    pitchers = (
        f'<div class="fb-pitchers">'
        f'{_pitcher_box(_val(g, "starting_pitcher_home", "sp_name_home"), _val(g, "sp_home_era", "sp_era_home"), _val(g, "sp_home_k9", "sp_k9_home"))}'
        f'{_pitcher_box(_val(g, "starting_pitcher_away", "sp_name_away"), _val(g, "sp_away_era", "sp_era_away"), _val(g, "sp_away_k9", "sp_k9_away"))}'
        f'</div>'
    )
    start_et = utils.start_time_et(g.get("start_time_utc", ""))
    venue = f'<div class="fb-venue">📍 {g.get("venue", "")}{f" · {start_et}" if start_et else ""}</div>'

    edge = g.get("edge_home", 0.0) if pick == home_team else (
        g.get("edge_away", 0.0) if pick == away_team else 0.0
    )
    odds = (
        f'<div class="fb-odds"><span>ML: {home_team} {utils.american(g.get("moneyline_home"))}'
        f'&nbsp;&nbsp;{away_team} {utils.american(g.get("moneyline_away"))}</span>'
        f'<span class="edge" style="color:{utils.edge_color(edge)};">Edge: {utils.edge_str(edge)}</span></div>'
    )

    banner = _banner_html(status, is_final, is_live, winner, pick, correct, is_coin_flip,
                          is_upset, home_team, away_team, g)

    runengine = _runengine_html(re_bits, home_team, away_team)

    return (
        f'<div class="fb-card"><div class="fb-top">{top}</div>{score}{home_row}{away_row}'
        f'{pregame}{pitchers}{venue}{odds}{runengine}{banner}</div>'
    )


def _shap_expander(g: pd.Series, date_str: str) -> None:
    gid = g["game_id"]
    with st.expander(f"📈 SHAP Features — {gid}", expanded=False):
        shap_df = utils.load_shap(gid, date_str)
        if shap_df.empty:
            st.caption("No SHAP file found for this game.")
            return
        chart = utils.shap_chart(shap_df)
        if chart is not None:
            utils.show_chart(chart)
        # SHAP is aligned to the FAVORED team (backend negates values when the
        # away team is favored), so positive = pushes the favorite toward winning.
        persp = ""
        if "perspective_team" in shap_df.columns:
            pt = shap_df["perspective_team"].dropna().astype(str)
            pt = pt[pt.str.strip() != ""].iloc[0] if not pt.empty else ""
            if pt and pt not in ("HOME", "AWAY"):
                persp = f" · Viewing from {pt}'s perspective"
        st.caption("Positive values increase the favored team's win probability; "
                   "negative decrease it. Averaged across the XGBoost / LightGBM / "
                   f"Logistic Regression ensemble.{persp}")


# ===========================================================================
# Page
# ===========================================================================

def _et_today_compact() -> str:
    """Today's date in America/New_York as YYYYMMDD, used so the frontend
    defaults to the user's real calendar date rather than UTC (which would
    show tomorrow's slate after 8 PM ET / midnight UTC)."""
    from zoneinfo import ZoneInfo
    from datetime import datetime
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")


def main() -> None:
    dates = utils.available_dates(**utils.get_source_config())
    if not dates:
        dates = ["20260809"]
    if "selected_date" not in st.session_state:
        # Default to today (ET) if an artifact exists for it, else latest.
        et_today = _et_today_compact()
        st.session_state["selected_date"] = (
            et_today if et_today in dates else dates[0])
    date_str = st.session_state["selected_date"]

    games = utils.load_todays_games(date_str)
    cal = utils.load_calibration(date_str)

    # Run-engine slate rows keyed by game_pk (ESPN game_id pre-game — the
    # 145d841 convention); joined to cards by game_id. Empty frame when the
    # artifact is missing or predates Phase 3 → cards just omit the strip.
    import market_diagnostics as diag  # noqa: E402  (pure, import-safe)
    _markets = utils.load_run_engine_markets(date_str)
    slate_map = {}
    if len(_markets) and "kind" in _markets.columns:
        _sl = _markets[_markets["kind"] == "slate"]
        if len(_sl):
            slate_map = {str(pk): rec
                         for pk, rec in zip(_sl["game_pk"],
                                            _sl.to_dict("records"))}

    history_view = False
    if games.empty:
        # No full card snapshot for this date — rebuild a simplified board
        # from the walk-forward prediction history, which covers every game
        # the model has ever predicted.
        games = utils.load_history_games(date_str)
        if not games.empty:
            history_view = True

    if games.empty:
        st.warning(f"No game artifacts found for {date_str}. Run the Colab pipeline "
                   "or configure your GitHub repo in the sidebar.")
        st.stop()

    # --- header: date + accuracy badge + evening note ---
    record = cal.get("today_record", {})
    wins, losses = record.get("wins", 0), record.get("losses", 0)
    completed = record.get("completed", wins + losses)
    acc = (wins / completed * 100) if completed else 0.0
    league_total = cal.get("league_total", len(games))
    evening_league = cal.get(
        "evening_games_league",
        int(games["evening_game"].sum()) if "evening_game" in games.columns else 0,
    )

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:2px;">
          <div style="font-size:1.35rem;font-weight:800;color:#E2E8F0;">{utils.format_date_long(date_str)}</div>
          <div style="color:#94A3B8;font-size:0.9rem;">· {len(games)} of {league_total} games shown</div>
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

    utils.arrow_nav(dates)

    if history_view:
        st.info(
            "🗂 Archive view — the full snapshot for this date was never pushed "
            "(or has been pruned), so cards are rebuilt from prediction history: "
            "scores, picks and results only (no pitchers, odds or SHAP)."
        )

    # --- filter pills ---
    counts = {
        "All Games": len(games),
        "Final": int((games["game_status"] == "Final").sum()),
        "Live": int((games["game_status"] == "Live").sum()),
    }
    options = ["All Games", "Final", "Live"]
    fmt = {o: f"{o} ({counts[o]})" for o in options}
    if "game_filter" not in st.session_state:
        st.session_state["game_filter"] = "All Games"
    if hasattr(st, "pills"):
        selected = st.pills(
            "Filter", options, format_func=lambda o: fmt[o], key="game_filter",
            selection_mode="single", label_visibility="collapsed",
        )
    else:
        selected = st.segmented_control(
            "Filter", options, format_func=lambda o: fmt[o], key="game_filter",
            label_visibility="collapsed",
        )

    filtered = games
    if selected == "Final":
        filtered = games[games["game_status"] == "Final"]
    elif selected == "Live":
        filtered = games[games["game_status"] == "Live"]

    st.divider()

    # --- game cards (two per row) + SHAP accordions ---
    for i in range(0, len(filtered), 2):
        cols = st.columns(2)
        for col, (_, g) in zip(cols, filtered.iloc[i : i + 2].iterrows()):
            with col:
                re_bits = diag.run_engine_card_bits(
                    str(g.get("game_id", "")), slate_map)
                st.markdown(_card_html(g, re_bits), unsafe_allow_html=True)
                _shap_expander(g, date_str)

    st.caption("Model outputs are point-in-time — only data available before each "
               "game's scheduled start was used. See README for methodology.")


if __name__ == "__main__":
    main()
