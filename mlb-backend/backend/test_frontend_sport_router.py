"""NFL integration steps 1-2 — sport state & routing, artifact resolver, NFL adapter.

Frontend-only, pure-Python (no Streamlit module import, matching
test_frontend_home.py). Locks:

1. Sport registry: NFL is registered (repo_subdir nfl-backend, no run engine,
   markets page excluded) with per-sport artifact patterns.
2. Sport state: ``utils.get_sport()`` is the single source of truth, reads
   ``st.session_state["sport"]``, normalizes, defaults to MLB.
3. Artifact resolver: ``resolve_sport_artifact`` / ``latest_artifact_date``
   resolve a sport+family to the newest committed artifact in that sport's
   own ``data_delivery`` dir (per-sport isolation), reused by
   ``last_refresh_time``.
4. NFL adapter: ``nfl_moneyline_to_frame`` maps the v1 moneyline JSON to the
   shared card DataFrame contract (MLB-only fields null/absent), returns an
   empty-with-schema frame for the current aggregate-only artifact.
5. MLB default path stays byte-identical (``load_todays_games`` dispatches on
   sport; sport=None/``mlb`` is the unchanged MLB branch).
6. AppTest smokes: both MLB and NFL render Home.py (and the NFL calibration
   notice) with 0 exceptions.

utils.py imports streamlit/altair at module scope, so the pure helpers are
extracted from its source via AST and executed in a minimal namespace — the
real code, not a copy (mirrors test_frontend_home's approach).
"""
import ast
import io
import json
import re
import subprocess
import sys
import unittest
from datetime import datetime
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent
_ROOT = _BACKEND.parent.parent
_FRONTEND = _ROOT / "frontend"
_MARKETS = _ROOT / "mlb-backend" / "data_delivery"
if str(_FRONTEND) not in sys.path:
    sys.path.insert(0, str(_FRONTEND))
_UTILS = _FRONTEND / "utils.py"
_TODAYS = _FRONTEND / "todays_games.py"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sports_config  # noqa: E402  (streamlit-free registry)


def _extract_todays(names):
    """Exec named top-level functions out of todays_games.py (pure parse
    helpers only — no Streamlit/graph exec)."""
    src = _TODAYS.read_text(encoding="utf-8")
    tree = ast.parse(src)
    segments = [
        ast.get_source_segment(src, node) for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    ns = {"pd": pd}
    exec("from __future__ import annotations\n" + "\n\n".join(segments), ns, ns)
    return ns


# ---------------------------------------------------------------------------
# AST extraction of pure utils helpers (no Streamlit import)
# ---------------------------------------------------------------------------

def _extract_utils(names):
    """Exec named top-level functions/assignments out of utils.py into an
    isolated namespace with the stdlib/pandas/sports-config deps they need.
    ``st`` is stubbed so ``get_sport`` reads a controllable session_state."""
    src = _UTILS.read_text(encoding="utf-8")
    tree = ast.parse(src)
    segments = []
    for node in tree.body:
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in names):
            segments.append(ast.get_source_segment(src, node))
        elif (isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id in names
                      for t in node.targets)):
            segments.append(ast.get_source_segment(src, node))
    ns = {
        "re": re, "json": json, "io": io, "pd": pd, "Path": Path,
        "np": np, "datetime": datetime,
        "REPO_ROOT": _ROOT,
        "DEFAULT_SPORT": sports_config.DEFAULT_SPORT,
        "SPORTS": sports_config.SPORTS,
        "normalize_sport_key": sports_config.normalize_sport_key,
        "resolve_sport": sports_config.resolve_sport,
        "artifact_patterns": sports_config.artifact_patterns,
    }
    ns["st"] = type("_St", (), {})()  # replaced below with a session stub
    exec("from __future__ import annotations\n" + "\n\n".join(segments),
         ns, ns)
    return ns


class _Session:
    """dict-like stand-in for streamlit session_state (get_sport reads it)."""

    def __init__(self, **kw):
        self._d = dict(kw)

    def __getitem__(self, k):
        return self._d[k]

    def __setitem__(self, k, v):
        self._d[k] = v

    def get(self, k, default=None):
        return self._d.get(k, default)


def _utils_with_sport(state: dict):
    """Extracted utils namespace with ``st.session_state`` bound to `state`."""
    u = _extract_utils([
        "get_sport", "sport_config",
        "_stamp_suffixes", "_max_stamp", "_latest_artifact_path",
        "resolve_sport_artifact", "latest_artifact_date",
        "NFL_CARD_COLUMNS", "_nl", "nfl_moneyline_to_frame",
        "load_nfl_moneyline", "load_todays_games",
    ])
    st = type("_St", (), {"session_state": _Session(**state)})()
    u["st"] = st
    # load_todays_games MLB path needs these; wire real ones via fakes at use.
    return u


# ---------------------------------------------------------------------------
# 1b. Per-sport valid game dates (Today's Games nav)
# ---------------------------------------------------------------------------

class TestValidDates(unittest.TestCase):
    """Per-sport valid game-date derivation (pure helpers, no Streamlit/net)."""

    @classmethod
    def setUpClass(cls):
        cls.u = _extract_utils([
            "_distinct_game_dates", "_valid_dates_impl", "nearest_valid_date",
        ])

    def test_nfl_valid_dates_are_distinct_game_date_from_games_array(self):
        fr = pd.DataFrame({"game_date": ["2026-09-09", "2026-09-13", "2026-09-09"]})
        self.assertEqual(self.u["_distinct_game_dates"](fr),
                         ["20260913", "20260909"])  # dedup, newest-first

    def test_nfl_empty_or_aggregate_only_yields_empty(self):
        self.assertEqual(self.u["_distinct_game_dates"](pd.DataFrame()), [])
        self.assertEqual(
            self.u["_distinct_game_dates"](pd.DataFrame({"home_team": ["A"]})), [])
        # aggregate-only artifact: moneyline frame carries game_date as empty
        self.assertEqual(
            self.u["_distinct_game_dates"](pd.DataFrame({"game_date": []})), [])

    def test_mlb_valid_dates_from_todays_games_artifact_listing(self):
        import shutil
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        (tmp / "todays_games_20260829.csv").write_text("x")
        (tmp / "todays_games_20260830.csv").write_text("x")
        (tmp / "power_rankings_20260830.csv").write_text("x")  # NOT a board
        got = self.u["_valid_dates_impl"](
            "mlb", ("20260829", "20260828"), tmp, pd.DataFrame())
        self.assertEqual(got, ["20260830", "20260829", "20260828"])

    def test_mlb_valid_dates_merge_history_dates(self):
        """MLB valid dates = todays_games snapshots UNION walk-forward history
        (calibration daily / prediction-history game dates) so season-long date
        navigation is preserved, not just the two committed snapshot days."""
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        got = self.u["_valid_dates_impl"](
            "mlb", ("20260830", "20260829"), tmp, pd.DataFrame(),
            history_dates=("20260425", "20260828", "20260829"))
        self.assertEqual(got, ["20260830", "20260829", "20260828", "20260425"])

    def test_mlb_missing_artifacts_yields_empty(self):
        import tempfile
        missing = Path(tempfile.mkdtemp()) / "nonexistent"
        self.assertEqual(
            self.u["_valid_dates_impl"]("mlb", (), missing, pd.DataFrame()), [])

    def test_mlb_and_nfl_date_sets_are_independent(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        # NFL derives dates from game_date, never from the MLB filename set.
        nfl = self.u["_valid_dates_impl"](
            "nfl", ("20260830",), tmp,
            pd.DataFrame({"game_date": ["2026-09-09"]}))
        mlb = self.u["_valid_dates_impl"](
            "mlb", ("20260830",), tmp, pd.DataFrame())
        self.assertEqual(nfl, ["20260909"])
        self.assertEqual(mlb, ["20260830"])

    def test_nearest_valid_date(self):
        nv = self.u["nearest_valid_date"]
        self.assertIsNone(nv([], "20260830"))
        self.assertEqual(nv(["20260829", "20260826"], "20260827"), "20260826")
        self.assertEqual(nv(["20260913", "20260909"], "20260830"), "20260909")
        self.assertEqual(nv(["20260830"], "20260830"), "20260830")

    def test_todays_games_nav_uses_valid_dates_not_available_dates(self):
        """The page steps/calendars off utils.valid_dates (per-sport), not the
        wider available_dates; the old date-ribbon arrow_nav is gone."""
        src = (_ROOT / "frontend" / "todays_games.py").read_text(encoding="utf-8")
        self.assertIn("utils.valid_dates(", src)
        self.assertIn("streamlit_calendar as sl_cal", src)
        self.assertIn("callbacks=[\"select\"]", src)
        self.assertNotIn("utils.arrow_nav(", src)


class TestCalendarExtraction(unittest.TestCase):
    """Parsing the streamlit-calendar click value into a navigable YYYYMMDD.
    Regression: the 'select' callback returns a full ISO timestamp
    ('2026-08-29T00:00:00.000Z'); norm() must collapse it to the date part or
    a calendar click never navigates."""

    @classmethod
    def setUpClass(cls):
        cls._mod = _extract_todays(["_extract_calendar_date"])

    def _extract(self, cal):
        return self._mod["_extract_calendar_date"](cal)

    def test_full_iso_select_payload(self):
        cal = {"callback": "select",
               "select": {"allDay": True,
                          "start": "2026-08-29T00:00:00.000Z",
                          "end": "2026-08-30T00:00:00.000Z"}}
        self.assertEqual(self._extract(cal), "20260829")

    def test_plain_date_and_other_shapes(self):
        self.assertEqual(self._extract({"select": {"start": "2026-08-30"}}), "20260830")
        self.assertEqual(self._extract({"selectedDate": "2026-09-13"}), "20260913")
        self.assertIsNone(self._extract({}))
        self.assertIsNone(self._extract(None))

    def test_invalid_payload_returns_none(self):
        self.assertIsNone(self._extract({"select": {"start": "not-a-date"}}))
        self.assertIsNone(self._extract({"select": {}}))


# ---------------------------------------------------------------------------
# 1. Sport registry & routing
# ---------------------------------------------------------------------------

class TestSportRegistryAndRouting(unittest.TestCase):
    def test_nfl_registered_with_no_run_engine(self):
        cfg = sports_config.resolve_sport("nfl")
        self.assertEqual(cfg["repo_subdir"], "nfl-backend")
        self.assertFalse(cfg["has_run_engine"])

    def test_nfl_excludes_markets_page(self):
        self.assertEqual(
            sports_config.active_page_url_paths("nfl"),
            ["todays-games", "power-rankings", "calibration", "model-monitor"])
        self.assertEqual(
            sports_config.active_page_url_paths("mlb"),
            ["todays-games", "power-rankings", "calibration", "model-monitor",
             "markets"])

    def test_artifact_patterns_are_explicit(self):
        self.assertEqual(
            sports_config.artifact_patterns("nfl"),
            {"moneyline_json": "nfl_moneyline_v1_*.json",
             "feature_json": "nfl_feature_v1_*.json"})
        self.assertIn("todays_games_csv",
                      sports_config.artifact_patterns("mlb"))

    def test_every_sport_carries_title_and_subtitle(self):
        # Registry-driven sidebar header: title/subtitle are per-sport so a
        # new sport needs zero UI-code changes.
        for key, cfg in sports_config.SPORTS.items():
            self.assertIsInstance(cfg.get("title"), str)
            self.assertTrue(cfg["title"].strip())
            self.assertIsInstance(cfg.get("subtitle"), str)
            self.assertTrue(cfg["subtitle"].strip())
        self.assertEqual(sports_config.resolve_sport("mlb")["title"],
                         "MLB Predictions")
        self.assertEqual(sports_config.resolve_sport("nfl")["title"],
                         "NFL Predictions")

    def test_normalize_defaults_and_unknowns(self):
        self.assertEqual(sports_config.normalize_sport_key("NFL"), "nfl")
        self.assertEqual(sports_config.normalize_sport_key(" mlb "), "mlb")
        # unknown non-empty falls back to the default (mlb)
        self.assertEqual(sports_config.normalize_sport_key("hockey"), "mlb")


# ---------------------------------------------------------------------------
# 2. Sport state (single source of truth)
# ---------------------------------------------------------------------------

class TestSportState(unittest.TestCase):
    def test_defaults_to_mlb_when_unset(self):
        u = _utils_with_sport({})
        self.assertEqual(u["get_sport"](), "mlb")

    def test_reads_session_state_selection(self):
        u = _utils_with_sport({"sport": "nfl"})
        self.assertEqual(u["get_sport"](), "nfl")
        u = _utils_with_sport({"sport": " NFL "})
        self.assertEqual(u["get_sport"](), "nfl")

    def test_unknown_value_falls_back(self):
        u = _utils_with_sport({"sport": "basketball"})
        self.assertEqual(u["get_sport"](), "mlb")

    def test_sport_config_follows_state(self):
        u = _utils_with_sport({"sport": "nfl"})
        self.assertEqual(u["sport_config"]()["repo_subdir"], "nfl-backend")


# ---------------------------------------------------------------------------
# 3. Artifact resolver (per-sport isolation)
# ---------------------------------------------------------------------------

class TestArtifactResolver(unittest.TestCase):
    def test_nfl_moneyline_resolves_real_artifact(self):
        u = _utils_with_sport({})
        # Per-game moneyline slate (step 3) is the newest moneyline artifact.
        self.assertEqual(u["latest_artifact_date"]("nfl", "moneyline_json"),
                         "20260830")

    def test_nfl_feature_resolves(self):
        u = _utils_with_sport({})
        path = u["resolve_sport_artifact"]("nfl", "feature_json")
        self.assertIsNotNone(path)
        self.assertIn("nfl-backend", str(path))

    def test_mlb_todays_games_resolves(self):
        u = _utils_with_sport({})
        d = u["latest_artifact_date"]("mlb", "todays_games_csv")
        self.assertIsNotNone(d)
        self.assertEqual(len(d), 8)

    def test_family_absent_returns_none(self):
        u = _utils_with_sport({})
        # NFL has no todays_games CSV family → None (no cross-sport bleed)
        self.assertIsNone(u["latest_artifact_date"]("nfl", "todays_games_csv"))
        self.assertIsNone(u["latest_artifact_date"]("mlb", "moneyline_json"))

    def test_per_sport_isolation(self):
        u = _utils_with_sport({})
        nfl = u["resolve_sport_artifact"]("nfl", "moneyline_json")
        self.assertIsNotNone(nfl)
        self.assertIn("nfl-backend", str(nfl))


# ---------------------------------------------------------------------------
# 4. NFL adapter — schema pin + aggregate-only path
# ---------------------------------------------------------------------------

class TestNflAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.u = _utils_with_sport({})
        cls.real = json.loads(
            (_ROOT / "nfl-backend" / "data_delivery" /
             "nfl_moneyline_v1_20260829.json").read_text())

    def test_schema_pinned_on_empty_record(self):
        """The current v1 moneyline JSON is aggregate-only → the adapter emits
        an EMPTY frame WITH the exact card column schema (never KeyErrors)."""
        f = self.u["nfl_moneyline_to_frame"](self.real)
        self.assertTrue(f.empty)
        self.assertEqual(list(f.columns), self.u["NFL_CARD_COLUMNS"])

    def test_per_game_fixture_maps_to_card_contract(self):
        data = {"games": [
            {"game_id": "2025_01_GB_CHI", "home_team": "CHI", "away_team": "GB",
             "home_win_prob": 0.42, "gameday": "2025-09-04",
             "home_score": 24, "away_score": 20},
        ]}
        f = self.u["nfl_moneyline_to_frame"](data)
        row = f.iloc[0]
        self.assertEqual(row["home_win_prob_model"], 0.42)
        self.assertAlmostEqual(row["away_win_prob_model"], 0.58)
        self.assertEqual(row["game_status"], "Final")
        self.assertEqual(row["model_pick"], "GB")          # away favored
        self.assertEqual(row["start_time_utc"], "2025-09-04T00:00:00Z")
        # MLB-only fields absent → null/"" (never fabricated)
        self.assertEqual(row["venue"], "")
        self.assertIsNone(row["edge_home"])

    def test_load_nfl_moneyline_returns_empty_schema(self):
        f = self.u["load_nfl_moneyline"]("nfl")
        self.assertEqual(list(f.columns), self.u["NFL_CARD_COLUMNS"])


# ---------------------------------------------------------------------------
# 5. MLB default path byte-identical (sport dispatch)
# ---------------------------------------------------------------------------

class TestMlbByteIdenticalDefaultPath(unittest.TestCase):
    def _dispatch(self, sport):
        u = _utils_with_sport({"sport": sport} if sport else {})
        fake_csv = b"game_id,home_team,away_team,home_win_prob_model\n1,H,A,0.6\n"
        calls = {}

        def _fetch_bytes(relpath, **kw):
            calls["relpath"] = relpath
            return fake_csv, "local"

        def _pick_date(d):
            return d

        def _normalize(df):
            return df  # identity — label the path is exercised

        u["_fetch_bytes"] = _fetch_bytes
        u["_pick_date"] = _pick_date
        u["normalize_games"] = _normalize
        u["get_source_config"] = lambda: {"owner": "", "repo": "", "branch": "main"}
        df = u["load_todays_games"]("20260830", sport=("mlb" if sport else None))
        return df, calls

    def test_default_and_mlb_are_byte_identical(self):
        df_def, calls_def = self._dispatch(None)
        df_mlb, calls_mlb = self._dispatch("mlb")
        pd.testing.assert_frame_equal(df_def, df_mlb)
        self.assertEqual(calls_def["relpath"], calls_mlb["relpath"])
        self.assertTrue(calls_mlb["relpath"].startswith("todays_games_"))

    def test_nfl_dispatch_never_touches_mlb_csv(self):
        u = _utils_with_sport({"sport": "nfl"})
        calls = {}

        def _fetch_bytes(relpath, **kw):
            calls["relpath"] = relpath
            raise AssertionError("NFL path must not fetch MLB todays_games CSV")

        u["_fetch_bytes"] = _fetch_bytes
        u["_pick_date"] = lambda d: d
        u["normalize_games"] = lambda df: df
        df = u["load_todays_games"]("20260829", sport="nfl")
        self.assertEqual(list(df.columns), u["NFL_CARD_COLUMNS"])
        self.assertNotIn("relpath", calls)


# ---------------------------------------------------------------------------
# 6. AppTest smokes — MLB + NFL with 0 exceptions
# ---------------------------------------------------------------------------

class TestNflAppTestSmoke(unittest.TestCase):
    def _run(self, script):
        res = subprocess.run([sys.executable, "-c", script],
                             capture_output=True, text=True, timeout=180)
        self.assertEqual(
            res.returncode, 0,
            f"AppTest subprocess failed:\nSTDOUT:\n{res.stdout}\n"
            f"STDERR:\n{res.stderr[-2000:]}")
        return res.stdout

    def test_mlb_home_renders_no_exception(self):
        script = (
            "import sys; sys.path.insert(0, %r);\n"
            "from streamlit.testing.v1 import AppTest;\n"
            "at = AppTest.from_file(%r, default_timeout=60);\n"
            "at.run();\n"
            "assert not at.exception, at.exception;\n"
            "print('MLB_OK')\n"
        ) % (str(_FRONTEND), str(_FRONTEND / "Home.py"))
        self.assertIn("MLB_OK", self._run(script))

    def test_nfl_home_renders_moneyline_board_no_exception(self):
        script = (
            "import sys; sys.path.insert(0, %r);\n"
            "from streamlit.testing.v1 import AppTest;\n"
            "at = AppTest.from_file(%r, default_timeout=60);\n"
            "at.session_state['sport'] = 'nfl';\n"
            "at.run();\n"
            "assert not at.exception, at.exception;\n"
            "all_text = ' '.join(getattr(m,'value','') for m in at.markdown);\n"
            "assert 'NFL \\u2014 Moneyline' in all_text, all_text[:500];\n"
            "print('NFL_OK')\n"
        ) % (str(_FRONTEND), str(_FRONTEND / "Home.py"))
        self.assertIn("NFL_OK", self._run(script))

    def test_nfl_calibration_shows_notice(self):
        script = (
            "import sys; sys.path.insert(0, %r);\n"
            "from streamlit.testing.v1 import AppTest;\n"
            "at = AppTest.from_file(%r, default_timeout=60);\n"
            "at.session_state['sport'] = 'nfl';\n"
            "at.run();\n"
            "assert not at.exception, at.exception;\n"
            "notices = ' '.join(i.value for i in at.info);\n"
            "assert 'NFL Calibration' in notices, notices[:500];\n"
            "print('NFL_CAL_OK')\n"
        ) % (str(_FRONTEND), str(_FRONTEND / "model_calibration.py"))
        self.assertIn("NFL_CAL_OK", self._run(script))

    def test_pills_select_nfl_updates_sport_and_header(self):
        """Clicking the NFL pill writes session_state['sport'] = 'nfl' and the
        registry-driven sidebar header switches to 'NFL Predictions'."""
        script = (
            "import sys; sys.path.insert(0, %r);\n"
            "from streamlit.testing.v1 import AppTest;\n"
            "at = AppTest.from_file(%r, default_timeout=60);\n"
            "at.run();\n"
            "assert not at.exception, at.exception;\n"
            "assert len(at.sidebar.pills) == 1, 'sport picker pills not rendered';\n"
            "assert at.session_state['sport'] == 'mlb';\n"
            "at.sidebar.pills[0].set_value('nfl');\n"
            "at.run();\n"
            "assert not at.exception, at.exception;\n"
            "assert at.session_state['sport'] == 'nfl', at.session_state;\n"
            "body = ' '.join(getattr(m,'value','') for m in at.markdown);\n"
            "assert 'NFL Predictions' in body, body[:500];\n"
            "assert 'NFL betting model dashboard' in body, body[:500];\n"
            "print('PILL_OK')\n"
        ) % (str(_FRONTEND), str(_FRONTEND / "Home.py"))
        self.assertIn("PILL_OK", self._run(script))

    def test_unknown_sport_falls_back_to_first_registry_key(self):
        """A stale/unknown session_state sport id is normalized to the first
        registry key (mlb) before the picker renders — never a crash or an
        empty nav."""
        script = (
            "import sys; sys.path.insert(0, %r);\n"
            "from streamlit.testing.v1 import AppTest;\n"
            "at = AppTest.from_file(%r, default_timeout=60);\n"
            "at.session_state['sport'] = 'hockey';\n"
            "at.run();\n"
            "assert not at.exception, at.exception;\n"
            "assert at.session_state['sport'] == 'mlb', at.session_state;\n"
            "body = ' '.join(getattr(m,'value','') for m in at.markdown);\n"
            "assert 'MLB Predictions' in body, body[:500];\n"
            "print('FALLBACK_OK')\n"
        ) % (str(_FRONTEND), str(_FRONTEND / "Home.py"))
        self.assertIn("FALLBACK_OK", self._run(script))


    def test_mlb_invalid_selected_date_falls_back_gracefully(self):
        """An invalid/stale selected_date (no board) never crashes — it shows
        the graceful fallback with a jump action instead of a zero-card page."""
        script = (
            "import sys; sys.path.insert(0, %r);\n"
            "from streamlit.testing.v1 import AppTest;\n"
            "at = AppTest.from_file(%r, default_timeout=60);\n"
            "at.session_state['_nav_sport'] = 'mlb';\n"
            "at.session_state['selected_date'] = '20000101';\n"
            "at.run();\n"
            "assert not at.exception, at.exception;\n"
            "warns = ' '.join(w.value for w in at.warning);\n"
            "assert 'No game board exists' in warns, warns[:400];\n"
            "print('MLB_FALLBACK_OK')\n"
        ) % (str(_FRONTEND), str(_FRONTEND / "Home.py"))
        self.assertIn("MLB_FALLBACK_OK", self._run(script))

    def test_mlb_valid_date_renders_board(self):
        script = (
            "import sys; sys.path.insert(0, %r);\n"
            "from streamlit.testing.v1 import AppTest;\n"
            "at = AppTest.from_file(%r, default_timeout=60);\n"
            "at.session_state['_nav_sport'] = 'mlb';\n"
            "at.session_state['selected_date'] = '20260830';\n"
            "at.run();\n"
            "assert not at.exception, at.exception;\n"
            "all_text = ' '.join(getattr(m,'value','') for m in at.markdown);\n"
            "assert 'games shown' in all_text, all_text[:300];\n"
            "print('MLB_BOARD_OK')\n"
        ) % (str(_FRONTEND), str(_FRONTEND / "Home.py"))
        self.assertIn("MLB_BOARD_OK", self._run(script))

    def test_mlb_aug29_board_renders_without_duplicate_keys(self):
        """Regression: the Aug 29 board crashed with StreamlitDuplicateElementKey
        because normalize_games' scores merge exploded a duplicated game_id into
        two rows (identical keyed selectboxes). It must render fully."""
        script = (
            "import sys; sys.path.insert(0, %r);\n"
            "import pandas as pd;\n"
            "import utils;\n"
            "g = utils.normalize_games(pd.read_csv(%r));\n"
            "ids = list(g['game_id']);\n"
            "assert len(ids) == len(set(ids)), 'duplicate game_id reached board';\n"
            "assert len(ids) > 1;\n"
            "print('AUG29_FRAME_OK', len(ids))\n"
        ) % (str(_FRONTEND),
             str(_ROOT / "mlb-backend/data_delivery/todays_games_20260829.csv"))
        self.assertIn("AUG29_FRAME_OK", self._run(script))

    def test_mlb_calendar_opens_no_exception(self):
        """Opening the FullCalendar picker with the sport's valid dates
        highlighted renders without crashing (streamlit-calendar loaded)."""
        script = (
            "import sys; sys.path.insert(0, %r);\n"
            "from streamlit.testing.v1 import AppTest;\n"
            "at = AppTest.from_file(%r, default_timeout=60);\n"
            "at.session_state['_nav_sport'] = 'mlb';\n"
            "at.session_state['open_calendar'] = True;\n"
            "at.session_state['selected_date'] = '20260830';\n"
            "at.run();\n"
            "assert not at.exception, at.exception;\n"
            "print('MLB_CAL_OK')\n"
        ) % (str(_FRONTEND), str(_FRONTEND / "Home.py"))
        self.assertIn("MLB_CAL_OK", self._run(script))

    def test_nfl_valid_date_renders_moneyline_day_board(self):
        """NFL dates come from the moneyline games[] array; a real per-game
        date renders the filtered moneyline cards (no aggregate-only notice)."""
        script = (
            "import sys; sys.path.insert(0, %r);\n"
            "from streamlit.testing.v1 import AppTest;\n"
            "at = AppTest.from_file(%r, default_timeout=60);\n"
            "at.session_state['sport'] = 'nfl';\n"
            "at.session_state['selected_date'] = '20260913';\n"
            "at.session_state['_nav_sport'] = 'nfl';\n"
            "at.run();\n"
            "assert not at.exception, at.exception;\n"
            "all_text = ' '.join(getattr(m,'value','') for m in at.markdown);\n"
            "assert 'NFL MONEYLINE' in all_text, all_text[:500];\n"
            "assert at.session_state['selected_date'] == '20260913';\n"
            "print('NFL_DATE_BOARD_OK')\n"
        ) % (str(_FRONTEND), str(_FRONTEND / "Home.py"))
        self.assertIn("NFL_DATE_BOARD_OK", self._run(script))


if __name__ == "__main__":
    unittest.main()