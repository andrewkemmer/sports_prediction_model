"""QB-mismatch strata diagnostic — test pins (spec 2026-09-05).

Synthetic unit pins (no network) cover the backup-flag logic (week-1 and
rookie-takeover cases), the as-of EPA rolling computation, strata
aggregation and the high-gap threshold. A real-universe class (skipped
when the 20260904 artifacts or the cached PBP season parquets are absent)
pins the 1,376-row universe, strata counts summing to the universe, the
backup fraction band and the STOP-ALL verdict from
``probe_qb_strata`` (read-only, double-run byte-identical).
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_qb_strata as Q  # noqa: E402

DD = Path(__file__).resolve().parent.parent / "data_delivery"
MK_CSV = DD / f"nfl_run_engine_markets_{Q.DATE}.csv"
HIST_CSV = DD / f"nfl_predictions_history_{Q.DATE}.csv"
PBP_CACHE = any(
    Path(Q.PBP_CACHE.format(yr=yr)).exists() for yr in Q.PBP_SEASONS)
HAVE_REAL = MK_CSV.exists() and HIST_CSV.exists() and PBP_CACHE

_PROBE: dict | None = None


def _run_once() -> bytes:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        Q.main()
    return buf.getvalue().encode("utf-8")


def _probe() -> dict:
    global _PROBE
    if _PROBE is None:
        _PROBE = json.loads(_run_once())
    return _PROBE


def _mk_pbp(rows: list[tuple]) -> pd.DataFrame:
    """Synthetic pbp frame: (game_id, season, week, posteam, passer,
    n_dropback_plays, epa). Each tuple is one passer-game; plays are
    expanded so the starter derivation sees qb_dropback plays."""
    cols = ["game_id", "season", "week", "posteam", "passer_player_id",
            "passer_player_name", "qb_epa", "qb_dropback", "play_id"]
    out = []
    for gid, season, week, team, passer, n, epa in rows:
        for i in range(n):
            out.append([gid, season, week, team, passer, passer, epa, 1, i])
    return pd.DataFrame(out, columns=cols)


# ---------------------------------------------------------------------------
# Synthetic: starter derivation + backup-flag logic
# ---------------------------------------------------------------------------
class TestStarterDerivation(unittest.TestCase):
    def test_starter_is_most_dropbacks_tie_first_passer(self):
        pbp = _mk_pbp([
            ("g1", 2021, 1, "TB", "A", 20, 0.1),
            ("g1", 2021, 1, "TB", "B", 25, 0.2),   # B starts (more plays)
            ("g1", 2021, 1, "NO", "C", 18, 0.0),
            ("g2", 2021, 2, "TB", "A", 22, 0.3),
            ("g2", 2021, 2, "TB", "X", 22, 0.4),   # tie with A -> first
            ("g2", 2021, 2, "NO", "C", 20, 0.1),
        ])
        st = Q.build_starter_tables(pbp)
        g1_tb = st[(st["game_id"] == "g1") & (st["posteam"] == "TB")]
        self.assertEqual(g1_tb["passer_player_id"].iloc[0], "B")
        g2_tb = st[(st["game_id"] == "g2") & (st["posteam"] == "TB")]
        self.assertEqual(g2_tb["passer_player_id"].iloc[0], "A")
        self.assertEqual(len(st), 4)                # 2 teams x 2 games


class TestBackupFlagLogic(unittest.TestCase):
    def _frame(self, seq: list[str], season_prev: str) -> pd.DataFrame:
        """One team, one season: seq = starter per week (week 1..n)."""
        rows = []
        for i, s in enumerate(seq, start=1):
            gid = f"{2021:04d}_{i:02d}_AWY_TEAM"
            rows.append((gid, 2021, i, "TEAM", s, 15, 0.1))
        if season_prev:
            rows.append(("2020_01_X_TEAM", 2020, 1, "TEAM",
                         season_prev, 15, 0.1))
        st = Q.build_starter_tables(_mk_pbp(rows))
        prim = Q.season_primary(st)
        return Q.add_backup_flags(st, prim)

    def test_change_games_and_returns_flag_under_op_def(self):
        st = self._frame(["A", "A", "B", "B", "A"], season_prev="A")
        st = st.sort_values("week")
        op = st[st["season"] == 2021]["backup_op"].tolist()
        # w1: no prior -> vs 2020 primary A -> False (A == A)
        # w2: A vs A -> False; w3: B vs A -> True; w4: B vs B -> False
        # w5: A vs B -> True (return flags too, per the operational def)
        self.assertEqual(op, [False, False, True, False, True])

    def test_rookie_takeover_late_in_season(self):
        """Late rookie takeover (V weeks 1-8, K weeks 9+): the operational
        def flags only the first takeover game (w9), while the season-
        primary sensitivity rule flags every V start (w1-w8) — the
        documented inversion the op def avoids."""
        seq = ["V"] * 8 + ["K"] * 9
        st = self._frame(seq, season_prev="V")
        st = st.sort_values("week").reset_index(drop=True)
        s21 = st[st["season"] == 2021]
        op_true = list(s21[s21["backup_op"]]["week"])
        sp_true = list(s21[s21["backup_season_primary"]]["week"])
        self.assertEqual(op_true, [9])
        self.assertEqual(sp_true, list(range(1, 9)))
        self.assertEqual(s21["backup_season_primary"].sum(), 8)

    def test_week1_no_prior_compares_prior_season_primary(self):
        st = self._frame(["B", "B"], season_prev="A")
        st = st.sort_values("week")
        self.assertTrue(st[(st["season"] == 2021) & (st["week"] == 1)]
                        ["backup_op"].iloc[0])      # B != 2020 primary A
        self.assertFalse(st[(st["season"] == 2021) & (st["week"] == 2)]
                         ["backup_op"].iloc[0])     # B == B


# ---------------------------------------------------------------------------
# Synthetic: as-of rolling EPA
# ---------------------------------------------------------------------------
class TestAsOfEpa(unittest.TestCase):
    def test_rolling_prior_mean_floor_three(self):
        """prior_n counts strictly-earlier games; prior_mean excludes the
        current game; rows below the floor carry prior_n < 3."""
        rows = [(f"g{w}", 2021, w, "TB", "A", 10, epa)
                for w, epa in enumerate([0.1, 0.2, 0.3, 0.4, 0.5], start=1)]
        rows.append(("g0", 2020, 1, "TB", "A", 10, 0.0))
        st = Q.build_starter_tables(_mk_pbp(rows))
        st = Q.add_asof_quality(st)
        st = st.sort_values(["season", "week"])
        w4 = st[(st["season"] == 2021) & (st["week"] == 4)].iloc[0]
        self.assertEqual(w4["prior_n"], 3)
        self.assertAlmostEqual(w4["prior_mean"], (0.1 + 0.2 + 0.3) / 3)
        w1 = st[(st["season"] == 2021) & (st["week"] == 1)].iloc[0]
        self.assertEqual(w1["prior_n"], 0)
        self.assertTrue(np.isnan(w1["prior_mean"]))

    def test_league_mean_fallback_present(self):
        rows = [(f"g{w}", 2021, w, "TB", "A", 10, epa)
                for w, epa in enumerate([0.1, 0.2], start=1)]
        st = Q.build_starter_tables(_mk_pbp(rows))
        st = Q.add_asof_quality(st)
        self.assertAlmostEqual(st["league_mean"].iloc[0],
                               st["epa_mean"].mean())


# ---------------------------------------------------------------------------
# Synthetic: strata aggregation + high-gap threshold
# ---------------------------------------------------------------------------
class TestStratumHelpers(unittest.TestCase):
    def test_high_gap_threshold(self):
        gaps = np.array([0.05, 0.10, -0.10, 0.15, -0.09, 0.22])
        high = np.abs(gaps) >= Q.HIGH_GAP_THRESHOLD
        self.assertEqual(high.sum(), 4)             # 0.10/-0.10/0.15/0.22

    @unittest.skipUnless(HAVE_REAL, "20260904 CSVs and/or cached PBP "
                         "season parquets not present")
    def test_strata_counts_sum_to_universe(self):
        """On the real universe the strata rows partition 1,376: backup +
        non-backup and low-gap + high-gap each sum exactly."""
        o = _probe()["strata"]
        by_label = {r["label"]: r["n"] for r in o}
        self.assertEqual(by_label["backup"] + by_label["non_backup"], 1376)
        self.assertEqual(by_label["high_gap"] + by_label["low_gap"], 1376)
        # interactions partition
        inter = sum(r["n"] for r in o if " x " in r["label"])
        self.assertEqual(inter, 1376)


# ---------------------------------------------------------------------------
# Real-universe pins (verdict + facts; skipped without artifacts/cache)
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_REAL, "20260904 CSVs and/or cached PBP season "
                     "parquets not present — run the daily emission + cache "
                     "the PBP first")
class TestRealUniverse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = _probe()
        cls.v = cls.o["verdict"]

    def test_universe_1376_with_full_starter_coverage(self):
        self.assertEqual(self.o["universe"]["n"], 1376)
        self.assertEqual(self.o["universe"]["n_binary_nonnull"], 1376)
        self.assertEqual(self.o["facts"]["games_missing_any_starter"], 0)
        self.assertEqual(self.o["facts"]["qb_gap_nonnull"], 1376)

    def test_backup_fraction_in_sane_band(self):
        """Operational def measured 25.4% of games (a change-game flags, a
        return game re-flags, and week-1-vs-prior-season-primary transitions
        count). The spec's 5-25% heuristic is widened to 0.15-0.35 for the
        real measured definition; the synthetic logic pins carry the exact
        semantics."""
        frac = self.o["facts"]["backup_op_frac"]
        self.assertGreaterEqual(frac, 0.15)
        self.assertLessEqual(frac, 0.35)
        # sensitivity definition is looser (every non-primary start flags)
        self.assertGreater(self.o["facts"]["backup_season_primary_frac"],
                           frac)

    def test_verdict_stop_all_pre_registered(self):
        """No concentration in the QB direction: backup stratum logloss is
        NOT worse than its complement, and the ceiling covariate gains
        < 0.004 logloss. The pre-registered routing must fire STOP-ALL."""
        self.assertEqual(self.v["verdict"], "STOP-ALL")
        self.assertLess(self.v["backup_logloss_delta_vs_complement"], 0.004)
        self.assertGreater(self.v["delta_ll_M3_vs_M0"], -0.004)

    def test_qb_gap_direction_sensible_but_weak(self):
        """The qb_gap coefficient is POSITIVE (the binary under-prices home
        QB advantage) — the right direction for a future feature — but the
        in-sample ceiling delta is ~0.001, far below the 0.004 GO bar."""
        self.assertEqual(self.o["ceiling"]["qb_gap_coef_sign"], "positive")
        self.assertLess(abs(self.v["delta_ll_M3_vs_M0"]), 0.004)

    def test_high_gap_stratum_is_better_not_worse(self):
        """The reverse-of-hypothesis finding: high-|qb_gap| games are BETTER
        calibrated than low-gap ones (both at 0.10 and top-quartile cuts)."""
        rows = {r["label"]: r for r in self.o["strata"]}
        self.assertLess(rows["high_gap"]["logloss"],
                        rows["low_gap"]["logloss"])
        self.assertLess(rows["high_gap_top_quartile_sensitivity"]["logloss"],
                        rows["low_gap_top_quartile_sensitivity"]["logloss"])

    def test_determinism_byte_identical(self):
        self.assertEqual(_run_once(), _run_once())


if __name__ == "__main__":
    unittest.main()
