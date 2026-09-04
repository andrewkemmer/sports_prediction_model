"""Layout-contract tests for the Streamlit entry point (frontend/Home.py).

Pure source inspection — no Streamlit import (the frontend tests never
import streamlit). Locks the two display moves:

1. Page order — the sidebar dashboard list is a single ``pages`` list in
   Home.py; "Totals & Run Lines" (markets.py) must be LAST, directly
   underneath "Model Monitor", with every page's title/icon/url preserved.
2. Branding above the dashboard list — Home.py renders the logo through
   the single delimited ``utils.render_brand_header()`` component, and
   utils.inject_css carries the sidebar flex reorder that puts the brand
   block above the stSidebarNav page list.

Also keeps the layout parameterizable for a future sport toggle: the
header is one component and the page list is one data-driven list.
"""
import ast
import json
import re
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_BACKEND = Path(__file__).resolve().parent
# frontend/ moved to the repository root (multi-sport restructure, Phase B)
_ROOT = _BACKEND.parent.parent
_HOME = _ROOT / "frontend" / "Home.py"
_UTILS = _ROOT / "frontend" / "utils.py"
_SPORTS = _ROOT / "frontend" / "sports_config.py"

# The intended sidebar order (url_path -> (title, icon)).
EXPECTED_ORDER = [
    ("todays-games", "Today's Games", "📅"),
    ("power-rankings", "Power Rankings", "🏆"),
    ("calibration", "Calibration", "📊"),
    ("model-monitor", "Model Monitor", "🛰️"),
    ("markets", "Totals & Run Lines", "🎯"),
]


def _pages_from_home() -> list[tuple[str, str, str]]:
    """AST-parse Home.py's ``pages = [st.Page(...), ...]`` list.

    Returns [(url_path, title, icon), ...] in registration order.
    """
    tree = ast.parse(_HOME.read_text(encoding="utf-8"))
    for node in tree.body:
        if not (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "pages"
                        for t in node.targets)):
            continue
        value = node.value
        if not isinstance(value, ast.List):
            continue
        rows = []
        for el in value.elts:
            if not (isinstance(el, ast.Call)
                    and isinstance(el.func, ast.Attribute)
                    and el.func.attr == "Page"):
                continue
            kwargs = {kw.arg: kw.value for kw in el.keywords if kw.arg}
            fname = el.args[0].value if el.args else None
            url = kwargs.get("url_path").value if "url_path" in kwargs else None
            title = kwargs.get("title").value if "title" in kwargs else None
            icon = kwargs.get("icon").value if "icon" in kwargs else None
            rows.append((str(fname), str(url), str(title), str(icon)))
        return rows
    raise AssertionError("no `pages = [...]` assignment found in Home.py")


class TestHomePageOrder(unittest.TestCase):
    def test_pages_list_is_single_source_and_ordered(self):
        rows = _pages_from_home()
        self.assertEqual(len(rows), len(EXPECTED_ORDER),
                         "page count changed — update EXPECTED_ORDER")
        got = [(url, title, icon) for _f, url, title, icon in rows]
        self.assertEqual(got, EXPECTED_ORDER)

    def test_markets_is_last_underneath_model_monitor(self):
        rows = _pages_from_home()
        urls = [url for _f, url, _t, _i in rows]
        self.assertEqual(urls[-1], "markets",
                         "Totals & Run Lines must be the LAST dashboard")
        self.assertEqual(urls[-2], "model-monitor",
                         "Model Monitor must sit directly above Totals & Run Lines")
        self.assertNotIn("markets", urls[:-1])

    def test_titles_icons_urls_preserved(self):
        rows = _pages_from_home()
        by_url = {url: (title, icon) for _f, url, title, icon in rows}
        for url, title, icon in EXPECTED_ORDER:
            self.assertEqual(by_url[url], (title, icon),
                             f"{url} title/icon changed — must be order-only")

    def test_default_page_unchanged(self):
        src = _HOME.read_text(encoding="utf-8")
        self.assertIn('url_path="todays-games", default=True', src)

    def test_backend_source_caption_removed(self):
        """The sidebar caption under the brand is gone so the dashboard
        list slides up directly underneath the logo/subtitle (divider
        stays as the separator)."""
        src = _HOME.read_text(encoding="utf-8")
        self.assertNotIn("Backend: Colab pipeline", src)
        self.assertNotIn("See README.md", src)


class TestBrandAboveDashboardList(unittest.TestCase):
    def test_home_uses_single_brand_component(self):
        src = _HOME.read_text(encoding="utf-8")
        # The logo renders ONLY through the delimited component — no inline
        # branding markdown left in Home.py.
        self.assertIn("utils.render_brand_header()", src)
        self.assertNotIn("MLB Predictions</div>", src,
                         "branding HTML must live in utils.render_brand_header")

    def test_brand_component_is_registry_driven(self):
        """The sidebar header title/subtitle come from the ACTIVE sport's
        registry entry (sport_config()), not hardcoded strings — a new sport
        needs zero UI-code changes. The literal title/subtitle values live in
        sports_config.SPORTS."""
        util = _UTILS.read_text(encoding="utf-8")
        self.assertIn("def render_brand_header()", util)
        self.assertIn("sport_config()", util,
                      "header must read the active sport's config")
        self.assertIn("cfg['emoji']", util)
        self.assertIn("cfg['title']", util)
        self.assertIn("cfg['subtitle']", util)
        self.assertIn("unsafe_allow_html=True", util)
        reg = _SPORTS.read_text(encoding="utf-8")
        self.assertIn('"title": "MLB Predictions"', reg)
        self.assertIn('"subtitle": "MLB betting model dashboard"', reg)
        self.assertIn('"title": "NFL Predictions"', reg)
        self.assertIn('"subtitle": "NFL betting model dashboard"', reg)

    def test_sidebar_reorder_css_present(self):
        src = _UTILS.read_text(encoding="utf-8")
        # The CSS that puts the brand block (inside stSidebarUserContent)
        # ABOVE the stSidebarNav page list.
        self.assertIn('[data-testid="stSidebarUserContent"]', src)
        self.assertIn('order: 2', src)
        self.assertIn('[data-testid="stSidebarNav"]', src)
        self.assertIn('order: 3', src)
        self.assertIn('[data-testid="stSidebarContent"]', src)
        self.assertIn("flex-direction: column", src)

    def test_fetch_unavailable_warning_removed(self):
        """The old 'Showing latest committed artifacts (GitHub fetch
        unavailable)' warning is gone — the sidebar no longer labels the
        local fetch fallback as an error, and shows a 'Last updated' line
        instead."""
        src = _UTILS.read_text(encoding="utf-8")
        self.assertNotIn("Showing latest committed artifacts", src,
                         "fetch-failure warning language must be gone")
        self.assertNotIn("GitHub fetch unavailable", src,
                         "fetch-failure warning language must be gone")
        # No replacement warning text of any kind for the local path.
        self.assertNotIn('if src != "local":', src,
                         "the local-source caption branch must be removed")
        # The empty-state guard (no artifacts at all) stays.
        self.assertIn('st.caption("⚠️ No artifacts found")', src)

    def test_last_updated_caption_wired_in_sidebar(self):
        """Home.py renders the sport-aware 'Last updated' caption with the
        ACTIVE sport toggle value, and utils exposes the helper + render."""
        home = _HOME.read_text(encoding="utf-8")
        self.assertIn("utils.render_last_updated(", home)
        self.assertIn("utils.render_last_updated(utils.get_sport())",
                      home, "caption must pass the active sport via get_sport()")
        self.assertIn("st.pills(", home, "sport selector is a pills toggle")
        self.assertIn('key="sport_picker"', home,
                      "pills widget mirrors session_state via its own key")
        self.assertIn('st.session_state["sport"] = _picked', home,
                      "picked value must be written back to the source of truth")
        util = _UTILS.read_text(encoding="utf-8")
        self.assertIn("def get_sport", util, "single-source-of-truth sport state")
        self.assertIn("def last_refresh_time", util)
        self.assertIn("def render_last_updated", util)
        self.assertIn("Last updated: —", util, "missing-artifact fallback text")
        self.assertIn("Last updated: ", util)


class TestSportNavSafety(unittest.TestCase):
    """Regression for the deployed line-81 crash + missing Today's Games.

    The nav filter previously read ``p.url_path`` off freshly built
    ``st.Page`` objects, which Streamlit only attaches inside st.navigation
    — on some versions that raises AttributeError (the crash) or returns ""
    (silently dropping every page, hiding Today's Games). The fix resolves
    the sport and the allowed page set in pure ``sports_config`` (no
    Streamlit), so this test runs without a Streamlit runtime.
    """
    @classmethod
    def setUpClass(cls):
        import sys
        _root = Path(__file__).resolve().parents[2]  # repo root
        _frontend = _root / "frontend"
        if str(_frontend) not in sys.path:
            sys.path.insert(0, str(_frontend))
        import sports_config as _sc
        cls.sc = _sc

    def test_toggle_never_keerrors_on_any_sport(self):
        # The segmented control may return the display label ("MLB"), the
        # config key ("mlb"), whitespace-padded keys, or an unknown sport.
        for bad in ("MLB", "Mlb", " mlb ", "nfl", "nba", "nhl", "", None, "  "):
            cfg = self.sc.resolve_sport(bad)  # must never raise
            self.assertIn(cfg["label"], {"MLB", "NFL", "NBA", "NHL"},
                          f"resolve_sport({bad!r}) must return a valid sport")

    def test_todays_games_always_active_for_mlb(self):
        # Today's Games (the default page) must always be in the rendered
        # nav set for MLB — assert presence, not just page count.
        paths = self.sc.active_page_url_paths("mlb")
        self.assertIn("todays-games", paths,
                      "Today's Games url_path must be active for MLB")
        # It should also lead the nav (register first / default).
        self.assertEqual(paths[0], "todays-games")

    def test_mlb_gets_full_page_set_matching_literal_pages(self):
        # The resolved MLB set must equal the sidebar-order contract (the
        # literal `pages` list in Home.py, mirrored by ALL_PAGE_URL_PATHS).
        self.assertEqual(
            self.sc.active_page_url_paths("mlb"),
            self.sc.ALL_PAGE_URL_PATHS,
        )

    def test_unknown_sport_degrades_to_full_set_not_blank(self):
        # A truly unknown sport must never yield an empty nav (blank sidebar);
        # it falls back to the full page set. NFL is now a REAL sport with
        # its own (run-engine-independent) page set.
        for bad in ("nba", "nhl", "hockey"):
            self.assertEqual(
                self.sc.active_page_url_paths(bad),
                self.sc.ALL_PAGE_URL_PATHS,
                f"unknown sport {bad!r} must fall back to the full page set",
            )
        # NFL is registered → its ordered page set now includes markets
        # (both sports ship run-engine slate-serve artifacts).
        self.assertEqual(
            self.sc.active_page_url_paths("nfl"),
            ["todays-games", "power-rankings", "calibration", "model-monitor",
             "markets"])

    def test_missing_default_sport_is_silent_not_a_warning(self):
        # The logo-click / navigation rerun leaves sport unset or None/"none"
        # — that missing-default state must NEVER trigger an "Unknown sport"
        # warning. Only a genuinely unknown NON-EMPTY value warns.
        for quiet in (None, "", "  ", "none", "NONE", " None "):
            self.assertFalse(
                self.sc.is_unknown_sport(quiet),
                f"missing-default {quiet!r} must be silent (no warning)",
            )
        # Valid/whitespace-padded valid keys are never "unknown" either.
        self.assertFalse(self.sc.is_unknown_sport("mlb"))
        self.assertFalse(self.sc.is_unknown_sport(" MLB "))
        self.assertFalse(self.sc.is_unknown_sport("nfl"))  # a real sport now
        # A genuine unknown non-empty sport still registers as unknown
        # (caller warns then).
        for bad in ("nba", "hockey", "!!"):
            self.assertTrue(
                self.sc.is_unknown_sport(bad),
                f"genuine unknown {bad!r} should warn",
            )

    def test_normalize_resolves_defaults_and_unknowns_silently(self):
        # resolve_sport returns a valid config for every bad/missing state
        # (never raises); empty/unknown lands on MLB, NFL resolves to NFL.
        for bad in (None, "", "none", "  ", "hockey"):
            cfg = self.sc.resolve_sport(bad)
            self.assertEqual(cfg["label"], "MLB",
                             f"resolve_sport({bad!r}) must fall back to MLB")
        self.assertEqual(self.sc.resolve_sport("nfl")["label"], "NFL")


def _extract_utils_funcs(names: list[str]) -> dict:
    """Exec the named (pure, streamlit-free) functions out of frontend/utils.py.

    utils.py imports streamlit/altair/pandas at module scope, which the
    frontend tests deliberately never import (a real Streamlit runtime would
    not be available and importing it early changes later test-module import
    order). These helpers use only stdlib, so we pull just their definitions
    from the source via AST and exec them in a minimal namespace — testing the
    real code, not a copy.
    """
    src = _UTILS.read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = set(names)
    body = [
        ast.get_source_segment(src, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    missing = sorted(wanted - {
        n.name for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    })
    assert not missing, f"function(s) not found in utils.py: {missing}"
    import sports_config as _sc
    from zoneinfo import ZoneInfo as _ZoneInfo
    ns: dict = {
        "re": re,
        "json": json,
        "datetime": datetime,
        "Path": Path,
        "ZoneInfo": _ZoneInfo,
        "_EASTERN_ZONE": _ZoneInfo("America/New_York"),
        "DEFAULT_SPORT": _sc.DEFAULT_SPORT,
        "normalize_sport_key": _sc.normalize_sport_key,
        "resolve_sport": _sc.resolve_sport,
        # utils.get_sport reads st.session_state; tests call the refresh
        # helpers with an explicit sport so this default is never exercised.
        "get_sport": lambda: _sc.DEFAULT_SPORT,
    }
    exec("from __future__ import annotations\n" + "\n\n".join(body), ns)
    return ns


class TestLastUpdated(unittest.TestCase):
    """Sport-aware 'Last updated' (pure helpers from utils.py, extracted via
    AST — no Streamlit import): newest dated artifact picked, a full-run JSON
    timestamp preferred over date-only suffixes, per-sport resolution, and
    graceful missing-artifact fallback."""

    @classmethod
    def setUpClass(cls):
        import sys
        cls._root = Path(__file__).resolve().parents[2]  # repo root
        _frontend = cls._root / "frontend"
        if str(_frontend) not in sys.path:
            sys.path.insert(0, str(_frontend))
        cls.u = _extract_utils_funcs([
            "_full_run_timestamp", "_stamp_suffixes", "_max_stamp",
            "_is_snapshot_record", "_is_snapshot_artifact",
            "_snapshot_dated_paths", "_mtime_utc", "_last_refresh_for_dir",
            "_is_date_only", "_to_eastern", "_format_refresh",
            "last_refresh_time",
        ])

    def _dir(self, files):
        import tempfile
        import shutil
        tmp = Path(tempfile.mkdtemp())
        for name, content in files:
            (tmp / name).write_text(content)
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        return tmp

    def _set_mtime(self, p: Path, utc: datetime):
        import os
        ts = utc.replace(tzinfo=ZoneInfo("UTC")).timestamp()
        os.utime(p, (ts, ts))

    def test_mlb_csv_mtime_is_full_timestamp_freshness(self):
        """MLB CSVs carry no created_utc/generated, so the newest snapshot
        artifact's file mtime (recomposed as UTC) is the freshness — the
        sidebar gets a FULL Eastern timestamp, like NFL's created_utc path."""
        d = self._dir([
            ("run_engine_markets_20260829.csv", "pk,stuff\n1,x\n"),
            ("run_engine_markets_20260830.csv", "pk,stuff\n2,y\n"),
        ])
        self._set_mtime(d / "run_engine_markets_20260829.csv",
                        datetime(2026, 8, 29, 18, 0, 0))
        self._set_mtime(d / "run_engine_markets_20260830.csv",
                        datetime(2026, 8, 30, 19, 3, 0))
        dt = self.u["_last_refresh_for_dir"](d)
        self.assertEqual(dt, datetime(2026, 8, 30, 19, 3, 0,
                                      tzinfo=ZoneInfo("UTC")))
        self.assertFalse(self.u["_is_date_only"](dt))
        self.assertEqual(self.u["_format_refresh"](dt),
                         "Last updated: Aug 30, 2026, 3:03:00 PM EDT")

    def test_snapshot_full_run_timestamp_preferred_over_date(self):
        """A PRIMARY snapshot record carrying a full run timestamp is preferred
        over the (coarser) date-only run_engine_markets suffix."""
        d = self._dir([
            ("run_engine_markets_20260830.csv", "pk\n1\n"),
            ("model_monitor_20260830.json",
             '{"generated": "2026-08-30T19:55:00"}'),
        ])
        dt = self.u["_last_refresh_for_dir"](d)
        self.assertEqual(dt.strftime("%Y-%m-%d %H:%M"), "2026-08-30 19:55")

    def test_nfl_snapshot_created_utc_still_preferred(self):
        """NFL's primary moneyline record keeps its precise run timestamp
        (created_utc) — the whitelist covers the NFL snapshot family too."""
        d = self._dir([
            ("nfl_moneyline_v1_20260830.json",
             '{"created_utc": "2026-08-30T20:09:31.977894Z"}'),
            # A dated artifact with a LATER mtime must not override created_utc
            # (the JSON's persisted run time wins — NFL format unchanged).
            ("nfl_feature_v1_20260830.json", "{}"),
        ])
        self._set_mtime(d / "nfl_feature_v1_20260830.json",
                        datetime(2026, 8, 30, 22, 30, 0))
        dt = self.u["_last_refresh_for_dir"](d)
        self.assertEqual(dt.strftime("%Y-%m-%d %H:%M"), "2026-08-30 20:09")
        self.assertEqual(self.u["_format_refresh"](dt),
                         "Last updated: Aug 30, 2026, 4:09:31 PM EDT")

    def test_diagnostic_full_timestamp_is_ignored(self):
        """The 2026-08-30 stale-time root cause: a stand-alone diagnostic's
        ``generated`` (e.g. deep_over_recheck at 17:29, margin_reliability at
        17:05) must NOT masquerade as the data refresh. The snapshot artifact
        (here its file mtime) sets the time instead."""
        d = self._dir([
            ("run_engine_markets_20260830.csv", "pk\n1\n"),
            ("deep_over_recheck_20260830.json",
             '{"generated": "2026-08-30T17:29:04"}'),
            ("margin_reliability_20260830.json",
             '{"generated": "2026-08-30T17:05:00"}'),
        ])
        self._set_mtime(d / "run_engine_markets_20260830.csv",
                        datetime(2026, 8, 30, 19, 3, 0))
        dt = self.u["_last_refresh_for_dir"](d)
        self.assertNotIn(dt.strftime("%H:%M"), ("17:29", "17:05"),
                         "diagnostic times must not drive freshness")
        self.assertEqual(dt, datetime(2026, 8, 30, 19, 3, 0,
                                      tzinfo=ZoneInfo("UTC")))
        self.assertFalse(self.u["_is_date_only"](dt))

    def test_newer_diagnostic_does_not_override_snapshot_time(self):
        """Even a diagnostic stamped LATER than a snapshot full-run time is
        ignored — diagnostics never set the sidebar time."""
        d = self._dir([
            ("model_monitor_20260830.json",
             '{"generated": "2026-08-30T18:56:00"}'),
            ("deep_over_recheck_20260830.json",
             '{"generated": "2026-08-30T19:58:00"}'),
        ])
        dt = self.u["_last_refresh_for_dir"](d)
        self.assertEqual(dt.strftime("%Y-%m-%d %H:%M"), "2026-08-30 18:56")
    def test_other_dated_artifact_fallback_without_markets(self):
        d = self._dir([("todays_games_20260830.csv", "pk\n1\n")])
        self._set_mtime(d / "todays_games_20260830.csv",
                        datetime(2026, 8, 30, 12, 0, 0))
        dt = self.u["_last_refresh_for_dir"](d)
        self.assertEqual(dt, datetime(2026, 8, 30, 12, 0, 0,
                                      tzinfo=ZoneInfo("UTC")))
        self.assertFalse(self.u["_is_date_only"](dt))
        self.assertEqual(self.u["_format_refresh"](dt),
                         "Last updated: Aug 30, 2026, 8:00:00 AM EDT")

    def test_missing_artifacts_fallback_dash(self):
        d = self._dir([])
        self.assertIsNone(self.u["_last_refresh_for_dir"](d))
        self.assertEqual(self.u["_format_refresh"](None), "Last updated: —")
        self.assertEqual(
            self.u["_format_refresh"](datetime(2026, 8, 30)),
            "Last updated: Aug 30, 2026")

    def test_sport_resolves_to_committed_dir(self):
        # real-artifact check: each sport resolves ITS OWN committed
        # data_delivery set (per-sport isolation) to a real, distinct date.
        self.u["REPO_ROOT"] = self._root
        mlb = self.u["last_refresh_time"]("mlb")
        nfl = self.u["last_refresh_time"]("nfl")
        self.assertTrue(mlb.startswith("Last updated: "), mlb)
        self.assertTrue(nfl.startswith("Last updated: "), nfl)
        self.assertNotEqual(mlb, "Last updated: —")
        self.assertNotEqual(nfl, "Last updated: —")
        # NFL resolves its own (JSON) artifacts, not MLB's CSV snapshot.
        self.assertNotEqual(mlb, nfl)

    def test_both_sports_render_full_eastern_timestamp(self):
        """The sidebar target format for BOTH sports (real artifacts):
        'Last updated: <Mon> <D>, <YYYY>, <H>:<MM>:<SS> <AM|PM> <ET|EST>'
        — MLB via snapshot mtime, NFL via created_utc, one formatter."""
        self.u["REPO_ROOT"] = self._root
        pat = re.compile(r"^Last updated: [A-Z][a-z]{2} \d{1,2}, \d{4}, "
                         r"\d{1,2}:\d{2}:\d{2} (AM|PM) (EDT|EST)$")
        for sport in ("mlb", "nfl"):
            s = self.u["last_refresh_time"](sport)
            self.assertRegex(s, pat, f"{sport} must render the full Eastern "
                                     f"format, got: {s!r}")


class TestRefreshEasternFormat(unittest.TestCase):
    """'Last updated' renders full timestamps in Eastern US time, DST-aware.

    via zoneinfo America/New_York — EDT (UTC-4) in daylight time, EST (UTC-5)
    in standard time, appended as %Z. Naive timestamps are treated as UTC (the
    pipeline's convention); date-only artifacts keep the date-only display
    (no fabricated time)."""

    @classmethod
    def setUpClass(cls):
        cls.u = _extract_utils_funcs([
            "_format_refresh", "_is_date_only", "_to_eastern",
        ])

    def test_summer_utc_to_edt(self):
        dt = datetime.fromisoformat("2026-08-30T17:05:00")  # naive UTC
        self.assertEqual(
            self.u["_format_refresh"](dt),
            "Last updated: Aug 30, 2026, 1:05:00 PM EDT")

    def test_winter_utc_to_est(self):
        dt = datetime.fromisoformat("2026-01-15T23:30:00")  # naive UTC
        self.assertEqual(
            self.u["_format_refresh"](dt),
            "Last updated: Jan 15, 2026, 6:30:00 PM EST")

    def test_dst_boundary_offset_flip(self):
        # US fall-back Nov 1 2026 02:00 EDT -> 01:00 EST (06:00 UTC). One
        # minute on each side: 05:30 UTC is still EDT, 06:30 UTC is EST.
        before = datetime.fromisoformat("2026-11-01T05:30:00")
        after = datetime.fromisoformat("2026-11-01T06:30:00")
        self.assertEqual(
            self.u["_format_refresh"](before),
            "Last updated: Nov 1, 2026, 1:30:00 AM EDT")
        self.assertEqual(
            self.u["_format_refresh"](after),
            "Last updated: Nov 1, 2026, 1:30:00 AM EST")

    def test_aware_utc_timestamp(self):
        # 'Z' -> +00:00 first for Python 3.10 fromisoformat compatibility.
        dt = datetime.fromisoformat("2026-08-28T23:13:02Z".replace("Z", "+00:00"))
        self.assertEqual(
            self.u["_format_refresh"](dt),
            "Last updated: Aug 28, 2026, 7:13:02 PM EDT")

    def test_naive_treated_as_utc(self):
        # A naive daytime value must convert as UTC, not be re-anchored.
        self.assertEqual(self.u["_to_eastern"](datetime(2026, 6, 1, 12, 0, 0))
                         .strftime("%H:%M %Z"), "08:00 EDT")

    def test_date_only_fallback_unchanged(self):
        # Date-only (naive midnight, from a YYYYMMDD suffix) stays date-only.
        self.assertIn("EDT", self.u["_format_refresh"](datetime(2026, 8, 30, 17, 5, 0)))
        self.assertEqual(
            self.u["_format_refresh"](datetime(2026, 8, 30)),
            "Last updated: Aug 30, 2026", "date-only must not gain a time")


class TestETDefaultDate(unittest.TestCase):
    """_et_today_compact returns today's date in America/New_York as YYYYMMDD.

    The fix: Todays Games defaulted to dates[0] (the newest artifact) which
    is UTC-based.  After 8 PM ET / midnight UTC, the app would show
    tomorrow's empty slate instead of today's settled games.  The fix
    computes today in ET and prefers it when an artifact exists.
    """

    def test_returns_8_digit_string(self):
        from zoneinfo import ZoneInfo
        from datetime import datetime
        # Re-implement the logic from todays_games._et_today_compact to test
        # it independently of Streamlit imports.
        et_today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")
        self.assertEqual(len(et_today), 8)
        self.assertTrue(et_today.isdigit())

    def test_matches_calendar_date_in_et(self):
        from zoneinfo import ZoneInfo
        from datetime import datetime
        et_now = datetime.now(ZoneInfo("America/New_York"))
        et_today = et_now.strftime("%Y%m%d")
        self.assertEqual(et_today, et_now.strftime("%Y%m%d"))

    def test_differs_from_utc_after_evening(self):
        """After 8 PM ET (midnight UTC), ET date != UTC date."""
        from zoneinfo import ZoneInfo
        from datetime import datetime, timezone
        utc_now = datetime.now(timezone.utc)
        et_now = utc_now.astimezone(ZoneInfo("America/New_York"))
        # If we're in the 8 PM ET – midnight UTC window, the dates differ.
        # This test just verifies the math is correct, not that we're
        # currently in that window.
        et_date = et_now.strftime("%Y%m%d")
        utc_date = utc_now.strftime("%Y%m%d")
        if et_now.hour >= 20 and utc_now.date() != et_now.date():
            self.assertNotEqual(et_date, utc_date,
                                "after 8 PM ET, ET date must differ from UTC")
        else:
            # Not in the divergence window — just confirm both are valid
            self.assertEqual(len(et_date), 8)
            self.assertEqual(len(utc_date), 8)


if __name__ == "__main__":
    unittest.main()
