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
  5. Header strip: `· N of M games shown` + evening-games pill + the
     MLB THREE-STATE badge — `✓ W-L Today · acc% accuracy` only when THIS
     board carries decided games, otherwise `No results yet — pre-game
     slate` / `No games on this date` (never a fabricated `✓ 0-0 · 0.0%`);
     fed by ⟐ ``utils.load_calibration(date, sport='nfl')``
     (nfl_calibration_*.json) with the decided count resolved by
     ``_decided_mask`` (``home_win`` when the artifact grades it, else a
     Final row's score pair — the store ships NaN ``home_win``);
     evening count computed from the start times with the NFL serving
     convention (``_is_evening_kickoff`` — the stamp's hour IS ET).
  6. ``_render_date_nav`` — the SHARED date navigation (arrows + calendar
     + mobile rail), imported, not duplicated.
  6b. NO archive banner between the date nav and the filter pills — the
     MLB dashboard's rendered flow (header strip → date nav → filter
     pills) carries no page-level archive notice, so the mirror renders
     none either; the frozen-store contract lives in code, not a banner.
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
     ⟐ ENRICHMENT SPLIT: run-engine markets and SHAP are CURRENT-SLATE
     ONLY — the slate loads when ``not history_view`` and an archive card
     renders MLB's quiet 'Run Engine data currently unavailable' strip
     (no later run's OOF re-price on a frozen card) with no SHAP
     accordion — the same guard MLB applies through
     ``_slate_map_for_view``. The QB matchup is DATED (MLB's ``sp_*``
     structure): ``history_view`` loads THIS date's
     ``nfl_qb_matchup_<date>.json``, so the twin boxes show that slate's
     starters and only a missing file renders '—' quietly.
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
iterations were removed for byte-parity with the MLB element tree.Empty / missing states: no board → the shared notice; a date outside the
valid set, or a valid date whose own snapshot/store fetch fails → MLB's
backward recovery walk first (newest renderable date inside the rolling
10-day window, under a labelled notice — a retention-pruned candidate is
refused, so a stale date never renders another date's board), then the
shared nearest-valid fallback; missing QB record → each QB box renders
'—' quietly (never fabricated); missing run-engine slate row → MLB's
quiet 'unavailable' strip.
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import streamlit as st

import nfl_slate_view as nfl_sv
import utils

# The shared render helpers the MLB board uses (imported — never duplicated).
from todays_games import (  # noqa: E402
    _match_slate_row,
    _nfl_card_html,
    _nfl_run_engine_selectors,
    _nfl_start_time_et,
    _nfl_shap_expander,
    _recovery_dates,
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

def _is_evening_kickoff(iso) -> bool:
    """True when the kickoff stamp reads 7 PM ET or later — under the NFL
    SERVING CONVENTION.

    The NFL artifacts keep the ET wall-clock inside an ISO-shaped field
    (nflverse gametime is already ET — see ``todays_games._nfl_start_time_
    et``: "Do not apply a second UTC-to-ET conversion"). ``utils._is_
    evening_start`` parses stamps as true UTC instants (the MLB/true-UTC
    convention), which shifts these stamps 4–5 hours and would label every
    8:20 PM ET kickoff a '☀ Day Game'. The hour is read AS ET here,
    exactly like the time renderer. Absent/unparseable → False (the card
    keeps its quiet default, never a fabricated tag).
    """
    try:
        dt = datetime.strptime(str(iso or "")[:16].replace("T", " "),
                               "%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return False
    return dt.hour >= 19


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
    # Default is Night; a present kickoff demotes it to Day when the stamp
    # is NOT ≥ 7 PM ET under the NFL serving convention (see
    # _is_evening_kickoff — the ET wall-clock lives in the ISO-shaped
    # field and must not be UTC-shifted).
    day_tag = "🌙 Night Game"
    start_iso = str(g.get("start_time_utc", "") or "")
    if start_iso and not _is_evening_kickoff(start_iso):
        day_tag = "☀ Day Game"
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
        mid = _nfl_start_time_et(start_iso) or "PREGAME"
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
    start_et = _nfl_start_time_et(start_iso)
    venue = (f'<div class="fb-venue">📍 {g.get("venue", "") or "—"}'
             f'{f" · {start_et}" if start_et else ""}</div>')
    banner = ''
    if is_scheduled:
        first = _nfl_start_time_et(start_iso)
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
            if _is_evening_kickoff(str(g.get("start_time_utc", "") or "")):
                n += 1
        except Exception:
            continue
    return n


# MLB's quiet run-engine fallbacks (todays_games._runengine_html) — the
# SAME markup an MLB card renders when no slate row resolves, so an archive
# card keeps the block instead of silently dropping it.
_RE_UNAVAILABLE = ('<div class="fb-runengine"><span class="re-label">'
                   'RUN ENGINE</span><span class="re-na">Run Engine data '
                   'currently unavailable</span></div>')
_RE_NA = ('<div class="fb-runengine"><span class="re-label">RUN ENGINE'
          '</span><span class="re-na">n/a</span></div>')


def _fill_display_fields(day: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """Archive-card display parity with MLB — venue + kickoff, nothing else.

    The frozen first-publication store carries prices, scores, grades and
    status but no display metadata, so archive cards rendered MLB's card
    anatomy empty: ``📍 —`` with no kickoff time, and the top strip
    defaulting every archive game to 🌙 Night Game (that tag derives from
    the start stamp). The dated ``nfl_board_<date>.csv`` snapshot carries
    exactly those display facts for the same game ids.

    DISPLAY-ONLY by contract: only ``venue`` and ``start_time_utc`` are
    ever copied, and only where the day has none. The board file is
    re-published after kickoff, so its probabilities are NEVER
    authoritative for a frozen card — the store is (the verified
    first-publication price; the board's later re-price must not reach
    the card). Missing snapshot / failed fetch / id mismatch → the day
    passes through unchanged (quiet '—', never fabricated names or
    times).
    """
    if day is None or not len(day) or "game_id" not in day.columns:
        return day
    try:
        raw, _src = utils._fetch_bytes(
            f"nfl_board_{date_str}.csv", sport="nfl",
            **utils.get_source_config())
        board = pd.read_csv(io.BytesIO(raw)) if raw is not None else None
    except Exception:
        board = None
    if board is None or not len(board) or "game_id" not in board.columns:
        return day
    out = day.copy()
    ids = out["game_id"].astype(str)
    src = (board.dropna(subset=["game_id"])
           .drop_duplicates("game_id").set_index("game_id"))
    src.index = src.index.astype(str)
    for col in ("venue", "start_time_utc"):
        cur = (out[col] if col in out.columns
               else pd.Series("", index=out.index)).astype(object)
        blank = cur.isna() | (cur.astype(str).str.strip() == "")
        if col not in src.columns:
            out[col] = cur.mask(blank, "")
            continue
        fill = ids.map(src[col])
        take = blank & fill.notna() & (fill.astype(str).str.strip() != "")
        out[col] = cur.mask(take, fill)
    return out


def _load_day(date_str: str) -> tuple[pd.DataFrame, bool]:
    """Resolve one NFL board date — MLB's ``load_todays_games`` →
    ``load_history_games`` two-step, over the NFL artifact families.

    Returns ``(frame, history_view)``: the current-slate moneyline record
    filtered to ``date_str`` when it covers that date (production view),
    otherwise the frozen first-publication card store (archive view — the
    production-as-published prediction, never an OOF re-price; the OOF
    history CSV only seeds dates the store has never seen).

    The store carries no display metadata, so an archive day is enriched
    display-only (venue + kickoff stamp) from the dated board snapshot —
    see ``_fill_display_fields``.
    """
    try:
        frame = utils.load_nfl_moneyline()
    except Exception:
        frame = pd.DataFrame()
    if frame is None or frame.empty:
        frame = pd.DataFrame()
    else:
        frame = frame.dropna(subset=["home_team", "away_team"])
    if len(frame):
        day = frame[frame["game_date"].astype(str).str.replace("-", "") == date_str]
        if not day.empty:
            return day, False
    try:
        day = utils.load_nfl_history_games(date_str)
    except Exception:
        day = pd.DataFrame()
    if day is None or day.empty:
        return pd.DataFrame(), False
    return _fill_display_fields(day, date_str), True


def _decided_mask(day: pd.DataFrame) -> pd.Series:
    """MLB's decided-game test over an NFL frame (the accuracy badge's gate).

    MLB's board CSV grades a ``home_win`` per game; the NFL families
    populate it inconsistently — the frozen card store ships it NaN — so
    fall back to a Final row's score pair. Same honesty rule (the badge may
    only claim results THIS board carries), artifact-tolerant test:
    pre-game and live rows never count.
    """
    if day is None or not len(day):
        return pd.Series(dtype="object").notna()
    if "home_win" in day.columns:
        win = pd.to_numeric(day["home_win"], errors="coerce")
        if win.notna().any():
            return win.notna()
    if {"home_score", "away_score"}.issubset(day.columns):
        hs = pd.to_numeric(day["home_score"], errors="coerce")
        as_ = pd.to_numeric(day["away_score"], errors="coerce")
        final = (day["game_status"].astype(str) == "Final"
                 if "game_status" in day.columns
                 else pd.Series(True, index=day.index))
        return (hs.notna() & as_.notna()).where(final, False)
    return pd.Series(False, index=day.index)


def _recovered_day(date_str: str, cand: str, valid_set: set[str]):
    """A renderable ``(frame, history_view)`` for ``cand``, or None — the
    MLB ``_recovered_board`` shape with the NFL resolution order.

    Candidates outside the valid set never render: the valid set IS the
    rolling 10-day window (dated board family ∪ current slate ∪ retained
    history), so a retention-pruned or never-boarded date is refused
    outright and a stale date can never render another date's board.
    """
    if cand == date_str or cand not in valid_set:
        return None
    day, history_view = _load_day(cand)
    if day is None or day.empty:
        return None
    return day, history_view


def _walk_recovery(date_str: str, valid: list[str], valid_set: set[str]) -> bool:
    """MLB's backward recovery walk (its ``_recovery_dates`` loop), NFL
    loaders: when the requested date is unreachable, render the newest
    renderable recent board UNDER A NOTICE instead of dead-ending at the
    substitute-date fallback. Each candidate is probed through the same
    ``_load_day`` resolution, so a lagging fetch heals on a rerun without
    user action. False when nothing renders (caller falls back).
    """
    for cand in _recovery_dates(date_str):
        got = _recovered_day(date_str, cand, valid_set)
        if got is None:
            continue
        day, history_view = got
        st.info(
            f"🗂 Recovery view — no game board was reachable for "
            f"{utils.format_date_long(date_str)} right now, so the most "
            f"recent available board ({utils.format_date_long(cand)}) is "
            "shown. Re-select the original date once its snapshot lands "
            "(usually within minutes of the next push)."
        )
        _render_board(day, cand, valid, history_view)
        return True
    return False


def run() -> None:
    """Render the NFL Today's Games board — the MLB page mirror.

    Anatomy mirrors MLB's two functions: this is ``main()`` (date
    resolution + the recovery walk) and ``_render_board()`` below is the
    shared renderer, so a recovered board and a normal one are rendered by
    the same code.
    """
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
        if _walk_recovery(date_str, valid, valid_set):
            return
        _render_nearest_valid_fallback(valid, date_str)
        st.stop()

    # 4. Board load — MLB's snapshot-then-history step over the NFL
    # artifact families (current slate → frozen first-publication store).
    day, history_view = _load_day(date_str)
    if day.empty:
        # A union-listed date whose own fetch failed: walk backward through
        # recent dates (inside the retention window) before the honest empty
        # state — MLB's structure, NFL loaders.
        if _walk_recovery(date_str, valid, valid_set):
            return
        _render_nearest_valid_fallback(valid, date_str)
        st.stop()

    _render_board(day, date_str, valid, history_view)


def _render_board(day: pd.DataFrame, date_str: str, valid,
                  history_view: bool = False) -> None:
    """Shared NFL board renderer — MLB's ``_render_board`` anatomy:
    accuracy header strip → date nav → filter pills → two-per-row cards
    → point-in-time caption (no archive banner — MLB's rendered flow
    carries none).

    Called by BOTH entry points (``run()`` and the recovery walk), so a
    recovered board is byte-identical to a normally-loaded one apart from
    the recovery notice. ⟐ Run-engine markets + SHAP stay CURRENT-SLATE
    ONLY: an archive view renders empty frames, so a frozen card can
    never inherit a later run's OOF re-price. The QB matchup is DATED
    (MLB's ``sp_*`` card-field structure): ``history_view`` loads THIS
    date's ``nfl_qb_matchup_<date>.json`` — never the newest file — so
    a frozen card shows its own slate's starters or the quiet '—' boxes.
    """
    slate = pd.DataFrame()
    qb = pd.DataFrame()
    if not history_view:
        try:
            slate, _sdate = utils.load_nfl_run_engine_markets("nfl")
        except Exception:
            slate = pd.DataFrame()
        try:
            qb = utils.load_nfl_qb_matchup("nfl")
        except Exception:
            qb = pd.DataFrame()
    else:
        # MLB history parity: the DATED QB record for THIS date (the twin
        # of MLB's dated sp_* card fields — a pitcher box always renders
        # from the frame). Newest-only here would leak another date's
        # starters onto a frozen card; a missing file degrades to the
        # quiet '—' boxes, never fabricated names.
        try:
            qb = utils.load_nfl_qb_matchup("nfl", date_str=date_str)
        except Exception:
            qb = pd.DataFrame()

    # 5. Header strip — SAME markup as MLB (fed by nfl_calibration_*.json),
    # including MLB's THREE-STATE badge: the accuracy pill is a TODAY
    # claim, so it renders only when THIS board carries decided games (a
    # pre-game slate never claims "✓ 0-0 · 0.0% accuracy").
    cal = utils.load_calibration(date_str, sport="nfl") or {}
    record = cal.get("today_record", {}) or {}
    wins, losses = record.get("wins", 0), record.get("losses", 0)
    completed = record.get("completed", wins + losses)
    acc = (wins / completed * 100) if completed else 0.0
    league_total = cal.get("league_total", len(day))
    evening_league = cal.get("evening_games_league", _evening_count(day))
    _n_decided = int(_decided_mask(day).sum())
    if _n_decided:
        _badge = f"✓ {wins}-{losses} Today · {acc:.1f}% accuracy"
        _badge_bg, _badge_fg = "rgba(16,185,129,.18)", "#34D399"
    elif len(day):
        _badge = "No results yet — pre-game slate"
        _badge_bg, _badge_fg = "rgba(59,130,246,.18)", "#93C5FD"
    else:
        _badge = "No games on this date"
        _badge_bg, _badge_fg = "rgba(148,163,184,.18)", "#94A3B8"
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:2px;">
          <div style="color:#94A3B8;font-size:0.9rem;">· {len(day)} of {league_total} games shown</div>
          <span style="background:rgba(59,130,246,.18);color:#93C5FD;border-radius:999px;padding:2px 10px;font-size:0.78rem;font-weight:700;">
            {evening_league} evening games begin 7 PM ET+
          </span>
          <span style="margin-left:auto;background:{_badge_bg};color:{_badge_fg};border-radius:999px;padding:2px 10px;font-size:0.8rem;font-weight:700;">
            {_badge}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 6. Shared date nav (arrows + calendar + mobile rail)
    _render_date_nav(valid, date_str)

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
                # MLB keeps the block whenever no slate row resolves (its
                # _runengine_html(None) → quiet 'unavailable'); an archive
                # card therefore renders the same muted strip instead of
                # silently dropping the run-engine area.
                re_html = _RE_UNAVAILABLE
                if srow is not None:
                    try:
                        re_html = nfl_sv.runengine_html(
                            srow, str(g.get("home_team", "")),
                            str(g.get("away_team", "")), **kw)
                    except Exception:
                        re_html = _RE_NA
                qb_row = None
                if qb is not None and len(qb):
                    gid = str(g.get("game_id", "") or "")
                    hit = qb[qb["game_id"].astype(str) == gid]
                    if len(hit):
                        qb_row = hit.iloc[0].to_dict()
                st.markdown(_nfl_mirror_card_html(g, qb_row, re_html),
                            unsafe_allow_html=True)
                # SAME per-card expander as MLB (📈 SHAP Features) — the
                # backend emits nfl_shap_game_<game_id>.csv attributions
                # from the deployed ensemble, so the identical chart renders.
                # A frozen ARCHIVE card never shows them: the family is
                # game-keyed (not date-keyed) and the backend re-publishes
                # it on every run, so it WOULD resolve to a later run's OOF
                # attributions (current-slate-only; NHL mirror parity).
                if not history_view:
                    _nfl_shap_expander(g)

    # 10. Same point-in-time caption
    st.caption("Model outputs are point-in-time — only data available before each "
               "game's scheduled start was used. See README for methodology.")
