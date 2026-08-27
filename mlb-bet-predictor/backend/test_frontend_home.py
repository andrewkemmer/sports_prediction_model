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
_ROOT = _BACKEND.parent
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


if __name__ == "__main__":
    unittest.main()
