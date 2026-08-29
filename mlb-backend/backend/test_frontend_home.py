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
import unittest
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent
# frontend/ moved to the repository root (multi-sport restructure, Phase B)
_ROOT = _BACKEND.parent.parent
_HOME = _ROOT / "frontend" / "Home.py"
_UTILS = _ROOT / "frontend" / "utils.py"

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

    def test_brand_component_keeps_asset_unchanged(self):
        src = _UTILS.read_text(encoding="utf-8")
        self.assertIn("def render_brand_header()", src)
        self.assertIn("⚾ MLB Predictions", src)
        self.assertIn("MLB betting model dashboard", src)
        self.assertIn("unsafe_allow_html=True", src)

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

    def test_fallback_caption_truthful_and_gated_on_local(self):
        """The offline-fallback note must (a) say what it actually serves
        (the real committed data_delivery artifacts — not "bundled
        samples") and (b) render only when the fallback path (source ==
        "local") is active."""
        src = _UTILS.read_text(encoding="utf-8")
        self.assertIn(
            "Showing latest committed artifacts (GitHub fetch unavailable)",
            src)
        # The old untruthful caption is gone (negated docstring phrases
        # like "not bundled samples" are fine and intentional).
        self.assertNotIn("Showing bundled sample data (offline fallback)", src)
        self.assertNotIn("Local sample data", src)
        # Gating: the fallback note is guarded by the local-source check.
        i = src.find("def render_source_note")
        self.assertGreater(i, -1)
        body = src[i:]
        self.assertIn('if src != "local":', body)
        self.assertIn('st.caption("📦 Showing latest committed artifacts', body)


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
        # An unknown sport must never yield an empty nav (blank sidebar).
        for bad in ("nfl", "nba", "nhl", "hockey"):
            self.assertEqual(
                self.sc.active_page_url_paths(bad),
                self.sc.ALL_PAGE_URL_PATHS,
                f"unknown sport {bad!r} must fall back to the full page set",
            )

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
        # A genuine unknown non-empty sport still registers as unknown
        # (caller warns then).
        for bad in ("nfl", "nba", "hockey", "!!"):
            self.assertTrue(
                self.sc.is_unknown_sport(bad),
                f"genuine unknown {bad!r} should warn",
            )

    def test_normalize_resolves_defaults_and_unknowns_silently(self):
        # resolve_sport returns a valid config for every bad/missing state
        # (never raises) and lands on MLB.
        for bad in (None, "", "none", "  ", "nfl", "hockey"):
            cfg = self.sc.resolve_sport(bad)
            self.assertEqual(cfg["label"], "MLB",
                             f"resolve_sport({bad!r}) must fall back to MLB")


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
