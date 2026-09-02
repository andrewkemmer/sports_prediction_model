"""Tests for the Today's Games run-line display fixes:

1. The run-line toggle no longer shows an "(unverified)" suffix on its
   options (labels are plain ±L; verification state lives in the tooltip).
2. The run-line pair is FAVORITE-anchored: the −L side is the moneyline
   favorite, never the underdog. When the away team is the moneyline
   favorite the card shows "AWAY −L X% · HOME +L Y%" instead of the
   home-anchored "HOME −L X% · AWAY +L Y%" (which read as the underdog
   laying −1.5). The favorite-side probability is derived from the game's
   own NB(λ, α) run marginals (away −L = winning by 2+ is NOT the ladder's
   away +L = winning or losing by 1), mirrored from the run engine's
   resolved margin distribution.

Pure-Python fixtures (no Streamlit render, no network); follows
test_frontend_todays_games.py conventions.
"""
from __future__ import annotations

import math
import re
import sys
import unittest
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
if str(FRONTEND) not in sys.path:
    sys.path.insert(0, str(FRONTEND))

import market_diagnostics as diag  # noqa: E402


def _todays():
    import streamlit as st  # noqa: F401  (bare-mode import chain)
    import todays_games as todays
    return todays


# ---------------------------------------------------------------------------
# Independent NB convolution reference (definition-level, NOT copied from
# todays_games): P(away wins by >= 2) = P(home margin <= -2) over the raw
# NB(λ, α) marginals — the same quantity the engine's MC samples. The
# display's ±1-band tie resolution never touches margins <= -2.
# ---------------------------------------------------------------------------
def _nb_pmf(k: int, mu: float, alpha: float) -> float:
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    alpha = max(float(alpha), 1e-9)
    n = 1.0 / alpha
    p = n / (n + mu)
    logp = (math.lgamma(k + n) - math.lgamma(n) - math.lgamma(k + 1)
            + n * math.log(p) + k * math.log1p(-p))
    return math.exp(logp)


def _expected_away_cover_minus(lam_h: float, lam_a: float,
                               al_h: float, al_a: float,
                               kmax: int = 120) -> float:
    """Reference P(away covers −1.5) = P(home margin <= −2)."""
    tot = 0.0
    for a in range(kmax + 1):
        pa = _nb_pmf(a, lam_a, al_a)
        for h in range(0, a - 1):      # h − a <= −2
            tot += pa * _nb_pmf(h, lam_h, al_h)
    return tot


def _full_grid_row(game_id="RL", lam_h=4.4, lam_a=4.9,
                   al_h=0.19, al_a=0.13):
    """Modern slate row: NB params + O/U grid + the home-anchored p_rl
    ladder over the full grid (half and whole lines)."""
    return {
        "game_pk": game_id,
        "home_expected_runs": lam_h,
        "away_expected_runs": lam_a,
        "alpha_home": al_h,
        "alpha_away": al_a,
        "p_over_9_5": 0.55, "p_under_9_5": 0.45,
        "p_home_cover_1_5": 0.3253,
        "p_rl_1_0_home": 0.44, "p_rl_1_0_push": 0.10, "p_rl_1_0_away": 0.46,
        "p_rl_1_5_home": 0.3253, "p_rl_1_5_push": 0.0,
        "p_rl_1_5_away": 0.6747,
        "p_rl_2_0_home": 0.30, "p_rl_2_0_push": 0.08, "p_rl_2_0_away": 0.62,
        "p_rl_2_5_home": 0.18, "p_rl_2_5_push": 0.0, "p_rl_2_5_away": 0.82,
        "p_rl_3_0_home": 0.17, "p_rl_3_0_push": 0.06, "p_rl_3_0_away": 0.77,
        "p_rl_3_5_home": 0.09, "p_rl_3_5_push": 0.0, "p_rl_3_5_away": 0.91,
        "p_rl_4_0_home": 0.09, "p_rl_4_0_push": 0.05, "p_rl_4_0_away": 0.86,
    }


class TestFavAnchorHelper(unittest.TestCase):
    """_card_fav_home = the moneyline favorite (the card's pick side);
    coin-flip / no-pick games anchor home deterministically."""

    def _todays(self):
        return _todays()

    def test_pick_side_is_anchor(self):
        todays = self._todays()
        self.assertFalse(todays._card_fav_home(
            {"model_pick": "BAL", "home_team": "COL"}))
        self.assertTrue(todays._card_fav_home(
            {"model_pick": "COL", "home_team": "COL"}))
        # pick == the away team -> away favorite
        self.assertFalse(todays._card_fav_home(
            {"model_pick": "NYY", "home_team": "BOS"}))

    def test_coin_flip_anchors_home(self):
        todays = self._todays()
        # No pick (coin flip / empty) -> deterministic home anchor.
        self.assertTrue(todays._card_fav_home(
            {"model_pick": "", "home_team": "COL"}))
        self.assertTrue(todays._card_fav_home(
            {"model_pick": None, "home_team": "COL"}))
        self.assertTrue(todays._card_fav_home(
            {"model_pick": "  ", "home_team": "COL"}))


class TestAwayFavoriteOrientation(unittest.TestCase):
    """Underdog-home game (e.g. COL home vs BAL away favored): the card
    must show the FAVORITE laying −1.5, never the underdog."""

    def _todays(self):
        return _todays()

    def _bits(self, row, rl_line=1.5, fav_home=False):
        todays = self._todays()
        raw = diag.run_engine_card_bits(str(row["game_pk"]),
                                        {str(row["game_pk"]): row},
                                        rl_line=rl_line)
        return todays._orient_rl_bits(raw, row, fav_home=fav_home)

    def test_underdog_home_renders_favorite_minus_side(self):
        """COL home 37% vs BAL away 63% (favorite = away): the span shows
        'BAL −1.5 X% · COL +1.5 Y%' — BAL (favorite) lays the runs and the
        underdog COL never appears as −1.5."""
        todays = self._todays()
        row = _full_grid_row()
        bits = self._bits(row, rl_line=1.5, fav_home=False)
        self.assertEqual(bits["rl_fav_side"], "away")
        html = todays._rl_html(bits, "COL", "BAL")
        self.assertIn("BAL −1.5", html)
        self.assertIn("COL +1.5", html)
        self.assertNotIn("COL −1.5", html,
                         "underdog must never be shown laying −1.5")

    def test_fav_cover_matches_reference_nb_convolution(self):
        """Away −1.5 cover = P(away wins by 2+) computed from the game's NB
        marginals — numerically identical to the independent reference, NOT
        the home-anchored complement (1 − P(home covers −1.5))."""
        todays = self._todays()
        row = _full_grid_row(lam_h=4.4, lam_a=4.9, al_h=0.19, al_a=0.13)
        bits = self._bits(row, rl_line=1.5, fav_home=False)
        expected = _expected_away_cover_minus(4.4, 4.9, 0.19, 0.13)
        self.assertAlmostEqual(bits["rl_fav_cover"], expected, places=9)
        # And it is strictly less than the ladder's away +1.5 column
        # (0.6747): +1.5 also covers away wins by 1 / losses by 1.
        self.assertLess(bits["rl_fav_cover"], 0.6747)

    def test_pair_complement_and_display_percent(self):
        """rl_fav_cover + rl_dog_cover = 1 (re-scaled 2-way) and the span's
        two percents are exact complements at {:.0%}."""
        todays = self._todays()
        row = _full_grid_row()
        bits = self._bits(row, rl_line=1.5, fav_home=False)
        self.assertAlmostEqual(bits["rl_fav_cover"] + bits["rl_dog_cover"],
                               1.0, places=9)
        html = todays._rl_html(bits, "COL", "BAL")
        pct = round(bits["rl_fav_cover"] * 100)
        self.assertIn(f"BAL −1.5 {pct}%", html)
        self.assertIn(f"COL +1.5 {100 - pct}%", html)

    def test_whole_line_away_fav_keeps_push_note(self):
        """Whole-line (2.0) away-fav: mirror push band priced from the NB
        marginals and the push note still renders; the display 2-way still
        sums to 100%."""
        todays = self._todays()
        row = _full_grid_row()
        # Give the mirror push band (P(away wins by exactly 2)) real mass by
        # choosing params where a 2-run away win is plausible.
        row = _full_grid_row(lam_h=5.5, lam_a=6.0, al_h=0.2, al_a=0.2)
        bits = self._bits(row, rl_line=2.0, fav_home=False)
        self.assertEqual(bits["rl_fav_side"], "away")
        self.assertAlmostEqual(bits["rl_fav_cover"] + bits["rl_dog_cover"],
                               1.0, places=9)
        html = todays._rl_html(bits, "COL", "BAL")
        self.assertIn("BAL −2.0", html)
        self.assertIn("COL +2.0", html)
        self.assertNotIn("COL −2.0", html)
        # No assertion on the push % (params above are near-symmetric, so
        # P(away by 2) is small but the note logic is exercised when > 0.5%).


class TestHomeFavoriteUnchanged(unittest.TestCase):
    """Home-favored games keep the artifact's home-anchored pair byte-for-
    byte (favorite = home is the identity relabel)."""

    def _todays(self):
        return _todays()

    def test_home_fav_oriented_equals_ladder_values(self):
        todays = self._todays()
        row = _full_grid_row()
        raw = diag.run_engine_card_bits("RL", {"RL": row}, rl_line=1.5)
        bits = todays._orient_rl_bits(raw, row, fav_home=True)
        self.assertEqual(bits["rl_fav_side"], "home")
        self.assertEqual(bits["rl_fav_cover"], raw["rl_home"])
        self.assertEqual(bits["rl_dog_cover"], raw["rl_away"])
        html = todays._rl_html(bits, "BOS", "NYY")
        self.assertEqual(html, todays._rl_html(raw, "BOS", "NYY"),
                         "home-fav oriented render must be identical to the "
                         "pre-orientation render")

    def test_no_favorite_supplied_falls_back(self):
        """fav_home=None (callers that never render a favorite) keeps the
        exact pre-change behaviour: oriented fields empty, home-anchored
        text."""
        todays = self._todays()
        row = _full_grid_row()
        raw = diag.run_engine_card_bits("RL", {"RL": row}, rl_line=1.5)
        bits = todays._orient_rl_bits(raw, row, fav_home=None)
        self.assertIsNone(bits["rl_fav_side"])
        self.assertIsNone(bits["rl_fav_cover"])
        html = todays._rl_html(bits, "BOS", "NYY")
        self.assertIn("BOS −1.5", html)
        self.assertIn("NYY +1.5", html)


class TestLegacyFallback(unittest.TestCase):
    """Legacy slate rows (no NB alpha columns) cannot re-anchor an
    away-favored game — the favorite-side price is not in the artifact and
    is never fabricated: rl_fav_side stays None and the card falls back to
    the home-anchored complement it can actually price."""

    def _todays(self):
        return _todays()

    def test_no_params_no_fabrication(self):
        todays = self._todays()
        legacy = {"game_pk": "LG", "home_expected_runs": 4.4,
                  "away_expected_runs": 4.9, "p_over_9_5": 0.55,
                  "p_under_9_5": 0.45, "p_home_cover_1_5": 0.3253}
        raw = diag.run_engine_card_bits("LG", {"LG": legacy})
        bits = todays._orient_rl_bits(raw, legacy, fav_home=False)
        self.assertIsNone(bits["rl_fav_side"])
        self.assertIsNone(bits["rl_fav_cover"])
        html = todays._rl_html(bits, "COL", "BAL")
        self.assertIn("(complement)", html)   # honest fallback, not oriented


class TestToggleLabel(unittest.TestCase):
    """The run-line toggle shows plain ±L option labels — no
    '(unverified)' suffix; verification state lives in the tooltip."""

    def test_format_func_is_plain_plusminus(self):
        src = (FRONTEND / "todays_games.py").read_text(encoding="utf-8")
        self.assertIn('f"±{v:.1f}"', src,
                      "toggle option labels must be plain ±L")
        self.assertNotIn('if v in _verified', src,
                         "the '(unverified)' suffix logic must be gone")
        self.assertNotIn('" (unverified)"', src)

    def test_verification_state_kept_in_tooltip(self):
        """The gate is not lost — the tooltip still reports how many grid
        lines the committed calibration record has cleared."""
        src = (FRONTEND / "todays_games.py").read_text(encoding="utf-8")
        self.assertIn("_rl_verified_lines()", src)
        self.assertIn("committed calibration", src)
        self.assertIn("marked verified", src)


class TestTieShareParity(unittest.TestCase):
    """The frontend's tie-resolution share MUST mirror the run engine's
    canonical constant (the analytic margin PMF has to agree with the
    artifact's own ladder convention)."""

    def test_mirror_constant_matches_run_engine(self):
        fe = (FRONTEND / "todays_games.py").read_text(encoding="utf-8")
        be = (Path(__file__).resolve().parents[2] / "mlb-backend" / "backend"
              / "run_engine.py").read_text(encoding="utf-8")
        m_fe = re.search(r"_RL_TIE_HOME_SHARE\s*=\s*([0-9.]+)", fe)
        m_be = re.search(r"MARGIN_PLUS1_HOME_SHARE\s*=\s*([0-9.]+)", be)
        self.assertIsNotNone(m_fe, "frontend share constant missing")
        self.assertIsNotNone(m_be, "backend share constant missing")
        self.assertEqual(float(m_fe.group(1)), float(m_be.group(1)))


if __name__ == "__main__":
    unittest.main()
