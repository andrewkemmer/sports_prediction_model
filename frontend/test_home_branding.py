"""Sport-aware browser-tab branding (2026-10-11 request).

The browser tab used to read "MLB Predictions · Streamlit" with the
baseball icon no matter which sport the sidebar toggle selected (the
page config was hardcoded at the top of Home.py).  Home.py now derives
``page_title`` / ``page_icon`` from the SAME registry entry the sidebar
brand header and pills use (``sports_config.page_branding``), so the tab
follows the toggle: NFL → "NFL Predictions" + football, NHL, NBA and MLB
likewise.  The " · Streamlit" suffix Streamlit appends client-side is
stripped by a same-origin component in Home.py (no supported option
suppresses it in Streamlit 1.61).

Pins:
  * ``page_branding`` returns exactly the ``set_page_config`` kwargs, with
    the registry's own title/emoji, for every sport;
  * an unknown sport key falls back to the default sport's branding — the
    same fallback the sidebar header uses;
  * the registry entries themselves carry both keys (a new sport without
    them fails here, not at page config time).
"""
from __future__ import annotations

import sys
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import sports_config  # noqa: E402


def test_page_branding_follows_every_sport():
    for key, cfg in sports_config.SPORTS.items():
        brand = sports_config.page_branding(key)
        assert set(brand) == {"page_title", "page_icon"}, brand
        assert brand["page_title"] == cfg["title"], (key, brand)
        assert brand["page_icon"] == cfg["emoji"], (key, brand)


def test_page_branding_pins_the_four_sports():
    """The exact tab titles the user asked for, one per toggle."""
    assert sports_config.page_branding("mlb") == {
        "page_title": "MLB Predictions", "page_icon": "⚾"}
    assert sports_config.page_branding("nfl") == {
        "page_title": "NFL Predictions", "page_icon": "\U0001f3c8"}
    assert sports_config.page_branding("nhl") == {
        "page_title": "NHL Predictions", "page_icon": "\U0001f3d2"}
    assert sports_config.page_branding("nba") == {
        "page_title": "NBA Predictions", "page_icon": "\U0001f3c0"}


def test_unknown_sport_falls_back_to_the_default_brand():
    """Same fallback contract as resolve_sport / the sidebar header."""
    assert (sports_config.page_branding("not-a-sport")
            == sports_config.page_branding(sports_config.DEFAULT_SPORT))
    assert (sports_config.page_branding(None)
            == sports_config.page_branding(sports_config.DEFAULT_SPORT))


def test_every_registry_entry_carries_branding():
    """A new sport must ship title+emoji (else set_page_config fails at
    runtime, far from the registry edit that caused it)."""
    for key, cfg in sports_config.SPORTS.items():
        assert cfg.get("title", "").endswith(" Predictions"), (key, cfg)
        assert cfg.get("emoji"), key
