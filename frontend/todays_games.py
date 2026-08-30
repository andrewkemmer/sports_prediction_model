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

# Bound at MODULE level (not inside main) so the resolve_* selector helpers
# below can never NameError. The inline import inside main() was the
# deployed crash: run 1 writes ou_line_<game_pk>/rl_line_<game_pk> into
# session_state, and run 2's resolve_totals_line/resolve_rl_line reached
# `diag` with no such global in scope. market_diagnostics is pure
# (stdlib + altair/numpy/pandas), so importing it here is import-safe.
import market_diagnostics as diag  # noqa: E402

# Grids for the per-card line selectors, resolved ONCE at module top with a
# safe fallback: if the attribute is ever renamed/missing upstream, degrade
# to the known defaults (totals 6.5-12.5 by 0.5; run lines 1.0-4.0) instead
# of crashing the page. The resolve_* helpers read these constants.
TOTAL_GRID = getattr(diag, "TOTAL_GRID",
                     [round(6.5 + 0.5 * i, 1) for i in range(13)])
RUN_LINE_GRID_FULL = getattr(diag, "RUN_LINE_GRID_FULL",
                             [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

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
    """Run-engine strip on the game card — projections, O/U, run line.

    bits comes from market_diagnostics.run_engine_card_bits; None means no
    slate row for this game (strip omitted), has_grid=False renders a quiet
    'n/a'. The O/U split is priced at the game's OWN rounded total
    (bits["total_line"], nearest 0.5 of λ_home + λ_away) unless a line was
    selected for the card (bits["line_selected"] set — the per-card
    selector / future market-lines mode), in which case the O/U reflects
    that line and the label says "line selected". Notes when the line was
    clamped to the shipped grid. Away +1.5 is the exact complement of home
    −1.5 (the artifact ships home-cover columns only) — labeled as such.
    """
    if bits is None:
        # No slate row for this game (run-engine artifacts missing for its
        # date — e.g. pruned at the GMT rollover). Never crash or break the
        # card layout: keep the block's container and show a muted notice
        # instead of silently omitting the RUN ENGINE strip.
        return ('<div class="fb-runengine"><span class="re-label">'
                'RUN ENGINE</span><span class="re-na">Run Engine data '
                'currently unavailable</span></div>')
    if not bits.get("has_grid"):
        return ('<div class="fb-runengine"><span class="re-label">'
                'RUN ENGINE</span><span class="re-na">n/a</span></div>')
    selected = bits.get("line_selected")
    ou_label = (f'O/U {bits["total_line"]:.1f}'
                + (" (clamped)" if bits.get("clamped") else "")
                + (f' (line selected: {selected:.1f})' if selected is not None
                   else ""))
    # The card shows the 2-WAY re-scaled split: Over + Under sum to 100%
    # (the push was folded proportionately by run_engine_card_bits, since a
    # push refunds the bet — whole-number lines price that way). p_push is
    # still read internally and exposed as an optional small annotation when
    # it is non-trivial (half-lines have p_push = 0 and show no annotation).
    push_note = ""
    if (bits.get("p_push") or 0) > 0.005:
        push_note = f' <span class="re-na">({bits["p_push"]:.0%} push)</span>'
    return (
        f'<div class="fb-runengine">'
        f'<span class="re-label">RUN ENGINE</span>'
        f'<span>Proj: {away_team} {bits["proj_away"]:.1f} – '
        f'{home_team} {bits["proj_home"]:.1f}</span>'
        f'<span>{ou_label}: Over {bits["p_over"]:.0%} / '
        f'Under {bits["p_under"]:.0%}{push_note}</span>'
        f'{_rl_html(bits, home_team, away_team)}'
        f'</div>'
    )


def _rl_html(bits, home_team: str, away_team: str) -> str:
    """Run-line span — the selected line (default ±1.5) with the RE-SCALED
    2-way cover split (push folded proportionately so home + away = 100%).
    Alternate lines not yet verified on the current artifact render as
    'unverified' (never fabricated)."""
    rl_line = bits.get("rl_line")
    default = bits.get("rl_line_default", 1.5)
    selected_note = (f" (line selected: −{rl_line:.1f})" if rl_line else "")
    if rl_line is None:
        # No per-card selection made — legacy ±1.5 pair.
        if bits.get("p_home_cover") is None:
            return f'<span>RL: n/a</span>'
        return (f'<span>RL: {home_team} −1.5 '
                f'{bits["p_home_cover"]:.0%} · '
                f'{away_team} +1.5 {bits["p_away_cover"]:.0%} '
                f'(complement)</span>')
    if bits.get("rl_unverified"):
        return (f'<span>RL: −{rl_line:.1f} '
                f'(line selected{selected_note}) — unverified on this '
                f'artifact</span>')
    if bits.get("rl_home") is None or bits.get("rl_away") is None:
        return f'<span>RL: n/a</span>'
    push_note = (f' ({bits["rl_push"]:.0%} push)'
                 if (bits.get("rl_push") or 0) > 0.005 else "")
    return (f'<span>RL: {home_team} −{rl_line:.1f} '
            f'{bits["rl_home"]:.0%} · '
            f'{away_team} +{rl_line:.1f} {bits["rl_away"]:.0%}{push_note}'
            f' (line selected{selected_note})</span>')


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

def _rl_verified_lines() -> set[float]:
    """Run lines the calibration gate has passed, from the committed
    run_line_calibration_*.json record (verdict 'calibrated'). Lines absent
    from the record (or with any other verdict) are NOT offered as
    verified — the card renders them 'unverified'. Read-only; the gate is
    versioned evidence, and a future record supersedes this one."""
    try:
        record = utils.load_rl_calibration()
        lines = record.get("lines", [])
        return {r["line"] for r in lines if r.get("verdict") == "calibrated"}
    except Exception:
        return set()


def resolve_rl_line(game_id, default_line: float = 1.5) -> float:
    """Resolve the per-card run-line selection, keyed by game_pk in
    session_state (mirrors resolve_totals_line; never bleeds between
    cards). The selector's value lives at ``rl_line_<game_pk>``; any
    invalid / out-of-grid value falls back to the default (1.5). A named
    helper so a future global market-lines mode can bulk-set the same
    keys from an odds feed without touching card rendering."""
    key = f"rl_line_{game_id}"
    if key not in st.session_state:
        st.session_state[key] = default_line
        return default_line
    try:
        val = round(float(st.session_state[key]), 1)
    except (TypeError, ValueError):
        return default_line
    if val not in RUN_LINE_GRID_FULL:
        return default_line
    return val


def resolve_totals_line(game_id, default_line: float) -> float:
    """Resolve the per-card O/U line selection, keyed by game_pk in
    session_state (persists across reruns; never bleeds between cards).

    ``default_line`` is the card's model-assigned line (its own rounded
    total). The selector's value lives at ``ou_line_<game_pk>``; any
    invalid / out-of-grid value (None, non-numeric, off the 6.5–12.5 grid)
    falls back to the model line — the card can never price a line the
    artifact doesn't carry. Structured as a helper so a future global
    'market lines' mode can bulk-set these keys from an odds feed without
    touching card rendering.
    """
    key = f"ou_line_{game_id}"
    if key not in st.session_state:
        st.session_state[key] = default_line
        return default_line
    try:
        val = round(float(st.session_state[key]), 1)
    except (TypeError, ValueError):
        return default_line
    if val not in TOTAL_GRID:
        return default_line
    return val


def _run_engine_dates() -> list[str]:
    """Available ``run_engine_markets_*.csv`` dates (YYYYMMDD), newest first.

    Enumerates the data_delivery contents API (same shape as
    ``utils.available_dates`` but for the run-engine artifact family) plus the
    local fallback. The ``*_rl`` bridge copy is excluded (not a date). Returns
    [] when nothing is reachable -- the resolver then degrades to 'unavailable'.
    """
    import requests
    cfg = utils.get_source_config()
    owner, repo, branch = cfg["owner"], cfg["repo"], cfg["branch"]
    dates: set[str] = set()
    if owner and repo:
        try:
            api = (f"https://api.github.com/repos/{owner}/{repo}/contents"
                   f"/{utils.REPO_SUBDIR}/data_delivery")
            resp = requests.get(api, timeout=15)
            if resp.ok:
                for item in resp.json():
                    name = item.get("name", "")
                    if name.startswith("run_engine_markets_") and name.endswith(".csv"):
                        rem = name[len("run_engine_markets_"):-len(".csv")]
                        if len(rem) == 8 and rem.isdigit():
                            dates.add(rem)
        except requests.RequestException:
            pass
    for p in utils.LOCAL_DATA_DIR.glob("run_engine_markets_*.csv"):
        rem = p.name[len("run_engine_markets_"):-len(".csv")]
        if len(rem) == 8 and rem.isdigit():
            dates.add(rem)
    return sorted(dates, reverse=True)


def _et_today_compact() -> str:
    """Today's date in America/New_York as YYYYMMDD, used so the frontend
    defaults to the user's real calendar date rather than UTC (which would
    show tomorrow's slate after 8 PM ET / midnight UTC)."""
    from zoneinfo import ZoneInfo
    from datetime import datetime
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")


def _nfl_card_html(r) -> str:
    """Compact NFL moneyline card (teams + win prob) from the adapted frame.

    Honors the shared card look (fb-card / fb-team / fb-bar) but omits the
    MLB-only sections (pitchers, run engine, ML odds). Win prob renders as
    '—' when the artifact carries none (per-game predictions ship with step
    3)."""
    home = str(r.get("home_team", "") or "")
    away = str(r.get("away_team", "") or "")
    home_name = str(r.get("home_team_name", "") or "") or home
    away_name = str(r.get("away_team_name", "") or "") or away
    ph = r.get("home_win_prob_model")
    pa = None if ph is None else 1.0 - float(ph)

    def _row(name, team, p):
        if p is None:
            pct = "—"
            width = 0
            color = "#334155"
        else:
            pct = f"{p:.0%}"
            width = int(round(p * 100))
            color = utils.PRIMARY if p >= 0.5 else "#38BDF8"
        return (
            f'<div class="fb-team"><span class="fb-accent" style="background:{color};"></span>'
            f'<div><span class="name">{name}</span> <span class="sub">{team}</span></div>'
            f'<span class="pct" style="color:{color};">{pct}</span></div>'
            f'<div class="fb-bar"><div class="fill" style="width:{width}%;background:{color};"></div></div>'
        )

    start = utils.start_time_et(str(r.get("start_time_utc", "") or ""))
    meta = str(r.get("game_date", "") or "")
    meta += f" · {start}" if start else ""
    meta_html = (f'<div class="fb-venue">📅 {meta}</div>' if meta
                 else '<div class="fb-venue">&nbsp;</div>')
    return (
        f'<div class="fb-card"><div class="fb-top">'
        f'<span class="fb-tag">NFL MONEYLINE</span><span class="spacer"></span></div>'
        f'{_row(home_name, home, ph)}{_row(away_name, away, pa)}{meta_html}</div>'
    )


def _render_nfl_cards(frame) -> None:
    """Render the compact NFL moneyline cards (two per row) for a frame."""
    for i in range(0, len(frame), 2):
        cols = st.columns(2)
        for col, (_, r) in zip(cols, frame.iloc[i:i + 2].iterrows()):
            with col:
                st.markdown(_nfl_card_html(r), unsafe_allow_html=True)
    st.caption("NFL moneyline probabilities are point-in-time model outputs.")


def _render_nfl_board() -> None:
    """Moneyline-first NFL Today's Games board (minimal render). MLB-only
    sections (pitchers, run engine, line selectors) are absent; if the
    shipped moneyline v1 artifact carries no per-game rows yet, show a clean
    notice instead of crashing."""
    st.markdown("<div style='font-size:1.7rem;font-weight:800;color:#E2E8F0;'>🏈 NFL — Moneyline</div>",
                unsafe_allow_html=True)
    fdate = utils.latest_artifact_date("nfl", "moneyline_json")
    tag = f"v1_{fdate}" if fdate else "—"
    st.markdown(
        f"<div style='color:#94A3B8;margin:2px 0 14px;'>NFL moneyline-first board · "
        f"artifact {tag} · per-game cards ship with step 3 (Calibration / "
        f"Model Monitor / Power Rankings show notices meanwhile).</div>",
        unsafe_allow_html=True,
    )
    frame = utils.load_nfl_moneyline()
    frame = frame.dropna(subset=["home_team", "away_team"]) if not frame.empty else frame
    if frame.empty:
        st.info(
            "No per-game NFL moneyline predictions are in the shipped "
            "nfl_moneyline_v1 artifact yet (it is aggregate-only). Moneyline "
            "cards will render here once the NFL pipeline ships per-game "
            "win probabilities (step 3)."
        )
        return
    _render_nfl_cards(frame)


def _render_nearest_valid_fallback(valid, current: str) -> None:
    """Missing-date fallback: never crash on a date with no board. Offer a
    one-click jump to the nearest valid game date; if there are none, this
    sport simply has no game-date artifacts yet."""
    st.warning(f"No game board exists for {utils.format_date_long(current)} "
               f"on this sport ({current}).")
    if valid:
        target = utils.nearest_valid_date(valid, current)
        if target and st.button(
                f"Jump to nearest game date — {utils.format_date_long(target)}"):
            st.session_state["selected_date"] = target
            st.session_state.pop("open_calendar", None)
            st.rerun()
    else:
        st.info("No per-sport game-date artifacts are available yet "
                "(run the pipeline / push artifacts).")


def _extract_calendar_date(cal) -> str | None:
    """Pull a picked date (YYYYMMDD) out of the streamlit-calendar return,
    tolerating the several return shapes across versions. None when nothing
    was (re)selected, so a plain calendar render never changes the date."""
    if not isinstance(cal, dict):
        return None

    def norm(v):
        if not v:
            return None
        s = str(v).strip()
        # The 'select' callback returns a full ISO timestamp
        # ('2026-08-29T00:00:00.000Z'); collapse it to the YYYY-MM-DD part.
        if len(s) >= 10 and s[4:5] == "-" and s[7:8] == "-":
            s = s[:10]
        if len(s) == 10 and s[4:5] == "-" and s.replace("-", "").isdigit():
            return s.replace("-", "")
        if len(s) == 8 and s.isdigit():
            return s
        return None

    sel = cal.get("select")
    if isinstance(sel, dict):
        hit = norm(sel.get("start") or sel.get("date") or sel.get("dateStr"))
        if hit:
            return hit
    else:
        hit = norm(sel)
        if hit:
            return hit
    for k in ("selectedDate", "dateClick", "eventClick"):
        v = cal.get(k)
        hit = norm(v if not isinstance(v, dict)
                   else (v.get("date") or v.get("dateStr") or v.get("start")))
        if hit:
            return hit
    for k, v in cal.items():
        hit = norm(v if not isinstance(v, dict)
                   else (v.get("start") or v.get("date") or v.get("dateStr")))
        if hit:
            return hit
    return None


# Tokens that tell phone/tablet-sized browser apart from a desktop UA, used
# only to relax the calendar's fixed outer height on narrow screens (where an
# inner scrollbar can swallow touch taps before they register). Best-effort —
# a missing/unknown UA just keeps the desktop height, never a degraded UI.
_MOBILE_UA_TOKENS = ("android", "iphone", "ipad", "mobile")


def _is_mobile_viewport() -> bool:
    """True when the request looks phone/tablet-sized.

    streamlit-calendar sets a fixed 440px ``height`` in its options dict; on a
    narrow phone that container can grow an inner scrollbar that intercepts
    the tap on a day cell before FullCalendar registers it. On small screens
    we hand FullCalendar ``"auto"`` instead, so the calendar body expands to
    fit (no inner scrollbar) and taps reach the highlighted cells. Desktop
    keeps the fixed 440px, so its layout is unchanged. Read from the request
    User-Agent where hosting exposes headers (Streamlit Community Cloud);
    never raises when headers are absent (tests / local reshapes).
    """
    try:
        headers = st.context.headers
    except Exception:
        return False
    ua = ""
    for key in ("User-Agent", "user-agent", "USER_AGENT"):
        val = headers.get(key)
        if isinstance(val, str):
            ua = val.strip().lower()
            if ua:
                break
    if not ua:
        return False
    return any(tok in ua for tok in _MOBILE_UA_TOKENS)


def _single_calendar_navigation(cal, valid) -> str | None:
    """Collapse a streamlit-calendar return to a single date to act on.

    Both ``dateClick`` and ``select`` are now subscribed: FullCalendar fires
    ``select`` on a desktop click but only ``dateClick`` on a touch tap (the
    Android bug), and on desktop a click can emit BOTH keys for the same
    interaction. Reducing through the existing ``_extract_calendar_date``
    (which prefers ``select.start``) folds the payload to one date. Returns
    that date for ANY tap on a valid/highlighted date — INCLUDING a re-tap of
    the currently-shown date, which the caller turns into a calendar close by
    popping ``open_calendar`` and rerunning (setting ``selected_date`` to the
    same value is a harmless no-op). Returns None for a plain open/close
    (empty payload), a blank/out-of-range cell, or invalid input, so those
    never trigger navigation.
    """
    chosen = _extract_calendar_date(cal)
    if chosen and chosen in set(valid):
        return chosen
    return None


def _render_calendar_picker(valid, current: str) -> None:
    """FullCalendar (streamlit-calendar) month picker: only the sport's valid
    dates are highlighted (background events) and selectable; the season is
    bounded by validRange so the calendar never spans months of no games.

    Subscribes BOTH ``dateClick`` and ``select`` callbacks: on a desktop a
    mouse click emits ``select`` (and often ``dateClick`` too), but on a touch
    device (Android/Chrome) a single tap only emits ``dateClick``. Routing the
    returned payload through ``_single_calendar_navigation`` keeps it to one
    navigation per distinct chosen date regardless of which callback(s)
    fire."""
    try:
        import streamlit_calendar as sl_cal
    except Exception:
        st.caption("Calendar component unavailable.")
        return

    def iso(d):
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    events = [{"title": " ", "start": iso(d), "end": iso(d), "allDay": True,
               "display": "background", "backgroundColor": "#10B981",
               "textColor": "#022C22"} for d in valid]
    from datetime import datetime, timedelta

    def _day_after(yyyymmdd):
        return (datetime.strptime(yyyymmdd, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")

    # FullCalendar's validRange.end is EXCLUSIVE by default, so set it to the
    # day AFTER the latest valid date — otherwise the last valid date is cut.
    vmin, vmax = iso(min(valid)), iso(_day_after(max(valid)))
    options = {
        "initialView": "dayGridMonth",
        "headerToolbar": {"left": "prev,next today",
                           "center": "title", "right": ""},
        # Fixed 440px on desktop; "auto" on phone/tablet-sized viewports so
        # the calendar body expands to content and never hosts an inner
        # scrollbar that swallows touch taps.
        "height": "auto" if _is_mobile_viewport() else 440,
        "validRange": {"start": vmin, "end": vmax},
        "selectable": True,
        "selectMirror": True,
        "navLinks": True,
        "eventDisplay": "background",
        "initialDate": iso(current),
    }
    try:
        cal = sl_cal.calendar(events=events, options=options,
                              callbacks=["select", "dateClick"],
                              key="todays_cal")
    except Exception:
        st.caption("Could not render the calendar.")
        return
    chosen = _single_calendar_navigation(cal, valid)
    if chosen:
        st.session_state["selected_date"] = chosen
        st.session_state.pop("open_calendar", None)
        st.rerun()


def _render_date_nav(valid, current: str) -> None:
    """Prev/next step ONLY through the sport's valid dates (skipping gaps),
    clamp + hint at the ends, and a clickable date field that opens the
    calendar (valid dates highlighted). Never lands on a zero-card date."""
    v = sorted(valid)
    idx = v.index(current) if current in v else -1
    prev = v[idx - 1] if idx > 0 else None
    nxt = v[idx + 1] if 0 <= idx < len(v) - 1 else None
    open_cal = st.session_state.get("open_calendar", False)
    field = "🗓 " + utils.format_date_long(current) + (" ▾" if not open_cal else " ▴")
    c1, c2, c3 = st.columns([1, 3, 1], gap="medium")
    with c1:
        prev_help = (f"Previous games: {utils.format_date_long(prev)}" if prev
                     else "Earliest date with games (previous disabled)")
        if st.button("◀", key="prev_day", use_container_width=True,
                     disabled=(prev is None), help=prev_help):
            st.session_state["selected_date"] = prev
            st.session_state.pop("open_calendar", None)
            st.rerun()
    with c2:
        if st.button(field, key="date_field", use_container_width=True,
                     help="Open the calendar — highlighted dates have game boards"):
            st.session_state["open_calendar"] = not open_cal
            st.rerun()
        if open_cal:
            _render_calendar_picker(valid, current)
            st.caption("Only highlighted dates have game boards for this sport; "
                       "◀/▶ step through valid dates only.")
    with c3:
        nxt_help = (f"Next games: {utils.format_date_long(nxt)}" if nxt
                    else "Latest date with games (next disabled)")
        if st.button("▶", key="next_day", use_container_width=True,
                     disabled=(nxt is None), help=nxt_help):
            st.session_state["selected_date"] = nxt
            st.session_state.pop("open_calendar", None)
            st.rerun()


def _render_nfl_day_board(frame, date_str: str) -> None:
    """Per-date NFL moneyline board for the selected valid game date."""
    st.markdown("<div style='font-size:1.7rem;font-weight:800;color:#E2E8F0;'>🏈 NFL — Moneyline</div>",
                unsafe_allow_html=True)
    fdate = utils.latest_artifact_date("nfl", "moneyline_json")
    tag = f"v1_{fdate}" if fdate else "—"
    st.markdown(
        f"<div style='color:#94A3B8;margin:2px 0 14px;'>NFL moneyline board · "
        f"{utils.format_date_long(date_str)} · artifact {tag}</div>",
        unsafe_allow_html=True,
    )
    _render_nfl_cards(frame)


def _run_nfl_main(valid, valid_set) -> None:
    """NFL Today's Games: moneyline board filtered to the selected valid game
    date. When the shipped record is aggregate-only (no per-game rows), a
    clean notice renders and there is nothing to navigate."""
    if not valid:
        _render_nfl_board()
        return
    if (st.session_state.get("_nav_sport") != "nfl"
            or "selected_date" not in st.session_state):
        st.session_state["selected_date"] = (utils.nearest_valid_date(valid) or valid[0])
        st.session_state["_nav_sport"] = "nfl"
    date_str = st.session_state["selected_date"]
    if date_str not in valid_set:
        _render_nearest_valid_fallback(valid, date_str)
        return
    _render_date_nav(valid, date_str)
    try:
        frame = utils.load_nfl_moneyline()
    except Exception:
        frame = pd.DataFrame()
    if frame is None or frame.empty:
        st.info("No NFL per-game moneyline rows available.")
        return
    frame = frame.dropna(subset=["home_team", "away_team"]) if not frame.empty else frame
    day = frame[frame["game_date"].astype(str).str.replace("-", "") == date_str]
    if day.empty:
        _render_nearest_valid_fallback(valid, date_str)
        return
    _render_nfl_day_board(day, date_str)


def main() -> None:
    """Sport-aware Today's Games: valid-date nav + calendar for the active
    sport (MLB board, NFL moneyline day board), with a graceful missing-date
    fallback. Touches date navigation only — card rendering/artifacts are
    passed through the existing loaders unchanged."""
    sport = utils.get_sport()
    valid = list(utils.valid_dates(sport))
    valid_set = set(valid)

    if sport == "nfl":
        _run_nfl_main(valid, valid_set)
        return

    # Sport reset / first visit -> nearest valid date to today (ET).
    if (st.session_state.get("_nav_sport") != "mlb"
            or "selected_date" not in st.session_state):
        st.session_state["selected_date"] = (utils.nearest_valid_date(valid) or "20260809")
        st.session_state["_nav_sport"] = "mlb"
    date_str = st.session_state["selected_date"]

    if date_str not in valid_set:
        _render_nearest_valid_fallback(valid, date_str)
        st.stop()

    games = utils.load_todays_games(date_str)
    cal = utils.load_calibration(date_str)

    # Run-engine slate rows resolved across the available dated
    # run_engine_markets_*.csv artifacts by game_pk (ESPN game_id pre-game --
    # the 145d841 convention), newest-first, instead of keying the lookup to
    # the game's exact date file. A game priced by a later run (artifact date
    # > game date) or a GMT-rollover evening game whose id carries the next
    # day's prefix still resolves here. An unresolvable id is simply absent
    # from the map -> the card renders the quiet 'unavailable' fallback.
    _frames = {}
    for _d in [date_str] + [d for d in _run_engine_dates() if d >= date_str]:
        if _d not in _frames:
            _frames[_d] = utils.load_run_engine_markets(_d)
    _gids = [str(g.get("game_id", "")) for _, g in games.iterrows()]
    slate_map = diag.resolve_slate_across_artifacts(_frames, _gids)

    history_view = False
    if games.empty:
        # No full card snapshot for this date — rebuild a simplified board
        # from the walk-forward prediction history, which covers every game
        # the model has ever predicted.
        games = utils.load_history_games(date_str)
        if not games.empty:
            history_view = True

    if games.empty:
        _render_nearest_valid_fallback(valid, date_str)
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

    _render_date_nav(valid, date_str)

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
                gid = str(g.get("game_id", ""))
                # Model line first (assigned), then the per-card selectors.
                model_bits = diag.run_engine_card_bits(gid, slate_map)
                if model_bits and model_bits.get("has_grid"):
                    model_line = model_bits["total_line"]
                    sel_line = resolve_totals_line(gid, model_line)
                    sel_rl = resolve_rl_line(gid, 1.5)
                    c_ou, c_rl = st.columns([1.6, 1], gap="small")
                    with c_ou:
                        st.selectbox(
                            "O/U line", diag.TOTAL_GRID,
                            index=diag.TOTAL_GRID.index(sel_line),
                            key=f"ou_line_{gid}",
                            label_visibility="collapsed",
                            help=("Totals line to price this game at — "
                                  "defaults to the model's assigned line; "
                                  "pick a sportsbook line to see the "
                                  "model's probability there."))
                    with c_rl:
                        _verified = _rl_verified_lines()
                        st.selectbox(
                            "Run line", diag.RUN_LINE_GRID_FULL,
                            index=diag.RUN_LINE_GRID_FULL.index(sel_rl),
                            format_func=lambda v: (
                                f"−{v:.1f}"
                                + ("" if v in _verified else " (unverified)")),
                            key=f"rl_line_{gid}",
                            label_visibility="collapsed",
                            help=("Run line to price this game at — "
                                  "defaults to ±1.5; alternates are gated "
                                  "on the committed calibration record "
                                  "(unverified lines render as such)."))
                    re_bits = diag.run_engine_card_bits(
                        gid, slate_map, line=sel_line, rl_line=sel_rl)
                else:
                    re_bits = model_bits
                st.markdown(_card_html(g, re_bits), unsafe_allow_html=True)
                _shap_expander(g, date_str)

    st.caption("Model outputs are point-in-time — only data available before each "
               "game's scheduled start was used. See README for methodology.")


if __name__ == "__main__":
    main()
