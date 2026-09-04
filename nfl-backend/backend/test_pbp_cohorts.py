"""PBP player-cohort feature diagnostic — test pins (spec 2026-09-05).

Synthetic unit pins (no network) cover the as-of ewm chain (floor,
prior-season fallback, league-mean carry), the game-level pairing
construction (M_h / M_a / mismatch), the weekly-stat builders for the
PBP families, the redundancy R2 helper, the in-sample ceiling models and
the pre-registered verdict routing (GO / RE_TEST_CANDIDATE / STOP).  A
real-universe class (skipped when the 20260904 artifacts or the cached
PBP / PFR / features parquets are absent) pins the 1,376-row universe,
the per-family verdicts from ``probe_pbp_cohorts`` (read-only, double-run
byte-identical) and the no-redundancy finding.
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
import probe_pbp_cohorts as P  # noqa: E402
from probe_pbp_cohorts import (  # noqa: E402
    FAMILIES, add_asof, ceiling_models, pair_covariates,
    pre_registered_verdict, redundancy_r2, strata_rows,
    weekly_stats_pbp, weekly_stats_pressure,
)

DD = Path(__file__).resolve().parent.parent / "data_delivery"
MK_CSV = DD / f"nfl_run_engine_markets_{P.DATE}.csv"
HIST_CSV = DD / f"nfl_predictions_history_{P.DATE}.csv"

try:
    import hashlib
    _decided = pd.read_csv(
        DD / "nfl_game_level_features.csv")  # canonical decided frame
    _h = hashlib.sha256()
    _h.update(_decided.sort_values("game_id").reset_index(drop=True)
              .to_csv(index=False).encode("utf-8"))
    _sha = _h.hexdigest()[:12]
    FEAT_CACHE = Path(f"{P.FEATURES_DIR}/nfl_features_{_sha}.parquet")
except Exception:  # noqa: BLE001
    FEAT_CACHE = Path("/nonexistent")

HAVE_CACHE = (MK_CSV.exists() and HIST_CSV.exists()
              and any(Path(f"{P.PBP_DIR}/nfl_pbp_{yr}.parquet").exists()
                      for yr in P.PBP_SEASONS)
              and any(Path(f"{P.PFR_DIR}/nfl_pfr_advstats_{yr}.parquet").exists()
                      for yr in P.PFR_SEASONS)
              and FEAT_CACHE.exists())
HAVE_REAL = HAVE_CACHE

_PROBE: dict | None = None


def _run_once() -> bytes:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        P.main()
    return buf.getvalue().encode("utf-8")


def _probe() -> dict:
    global _PROBE
    if _PROBE is None:
        _PROBE = json.loads(_run_once())
    return _PROBE


def _mk_pbp() -> pd.DataFrame:
    """Tiny pbp frame: one game H@A (both teams) covering pass EPA, run
    EPA, big plays, a red-zone drive, and 3rd-down plays."""
    cols = ["game_id", "season", "week", "posteam", "defteam", "play_type",
            "epa", "air_yards", "yards_gained", "yardline_100", "down",
            "touchdown", "two_point_attempt", "fixed_drive"]
    # (posteam, defteam, type, epa, air, yds, yl100, down, td, twopt, drive)
    plays = [
        # A on offense vs B: two passes (EPA .4/.2), one big run 14 yds,
        # one 3rd-down pass, one RZ-drive TD
        ("A", "B", "pass", 0.4, 25, 30, 40, 1, 0, 0, 1),
        ("A", "B", "pass", 0.2, 5, 6, 45, 3, 0, 0, 1),
        ("A", "B", "run", 0.5, np.nan, 14, 15, 1, 1, 0, 1),
        #   ^ big + TD drive
        # B on offense vs A
        ("B", "A", "pass", -0.3, 3, 4, 60, 1, 0, 0, 2),
        ("B", "A", "run", -0.1, np.nan, 3, 70, 3, 0, 0, 2),
        ("B", "A", "run", -0.05, np.nan, 2, 10, 1, 0, 0, 2),  # RZ no TD
    ]
    rows = [["g1", 2021, 1] + list(p) for p in plays]
    return pd.DataFrame(rows, columns=cols)


def _mk_pfr() -> pd.DataFrame:
    cols = ["game_id", "season", "week", "team", "opponent",
            "times_pressured", "times_pressured_pct"]
    return pd.DataFrame([
        ["g1", 2021, 1, "A", "B", 8.0, 0.200],   # A OL: 8/40 dropbacks
        ["g1", 2021, 1, "B", "A", 3.0, 0.100],   # B OL: 3/30 dropbacks
    ], columns=cols)


# ---------------------------------------------------------------------------
# Weekly-stat builders (synthetic pbp / pfr)
# ---------------------------------------------------------------------------
class TestWeeklyBuilders(unittest.TestCase):
    def test_pass_epa_off_and_def(self):
        wk = weekly_stats_pbp(_mk_pbp())["F1_pass_epa"]
        a = wk[wk["team"] == "A"].iloc[0]
        self.assertAlmostEqual(a["off_val"], (0.4 + 0.2) / 2)   # vs B
        self.assertAlmostEqual(a["def_val"], -0.3)              # vs B pass
        b = wk[wk["team"] == "B"].iloc[0]
        self.assertAlmostEqual(b["off_val"], -0.3)
        self.assertAlmostEqual(b["def_val"], (0.4 + 0.2) / 2)

    def test_rush_epa_and_third_down(self):
        wk = weekly_stats_pbp(_mk_pbp())
        r = wk["F2_rush_epa"]
        self.assertAlmostEqual(r[r["team"] == "A"]["off_val"].iloc[0], 0.5)
        self.assertAlmostEqual(r[r["team"] == "B"]["def_val"].iloc[0], 0.5)
        t3 = wk["F6_third_down"]
        # A: 3rd-down pass epa 0.2 ; B: 3rd-down run epa -0.1
        self.assertAlmostEqual(t3[t3["team"] == "A"]["off_val"].iloc[0], 0.2)
        self.assertAlmostEqual(t3[t3["team"] == "B"]["off_val"].iloc[0], -0.1)

    def test_big_play_share(self):
        wk = weekly_stats_pbp(_mk_pbp())["F4_big_play"]
        a = wk[wk["team"] == "A"].iloc[0]
        # A scrimmage: pass(air 25 big), pass(air 5 no), run(14 big) -> 2/3
        self.assertAlmostEqual(a["off_val"], 2 / 3)
        # A's defense faced B: pass(3 yds no), run(3 no) -> 0/2
        self.assertAlmostEqual(a["def_val"], 0.0)
        b = wk[wk["team"] == "B"].iloc[0]
        self.assertAlmostEqual(b["off_val"], 0.0)
        self.assertAlmostEqual(b["def_val"], 2 / 3)

    def test_redzone_td_rate(self):
        wk = weekly_stats_pbp(_mk_pbp())["F5_redzone_td"]
        a = wk[wk["team"] == "A"].iloc[0]
        # A drive 1 enters RZ (yl 15 <= 20) and scores -> 1.0
        self.assertAlmostEqual(a["off_val"], 1.0)
        self.assertAlmostEqual(a["def_val"], 0.0)   # B's only drive: no RZ

    def test_pressure_starter_row_and_def_mirror(self):
        f3 = weekly_stats_pressure(_mk_pfr())
        a = f3[f3["team"] == "A"].iloc[0]
        self.assertAlmostEqual(a["off_val"], 0.2)     # A OL allowed
        self.assertAlmostEqual(a["def_val"], 0.1)     # A DL = B OL rate
        b = f3[f3["team"] == "B"].iloc[0]
        self.assertAlmostEqual(b["off_val"], 0.1)
        self.assertAlmostEqual(b["def_val"], 0.2)


# ---------------------------------------------------------------------------
# As-of chain: floor, prior-season fallback, league mean
# ---------------------------------------------------------------------------
class TestAsOfChain(unittest.TestCase):
    def _weekly(self) -> pd.DataFrame:
        """Team T: 2020 (5 weeks, all values) + 2021 weeks 1..5.
        Team U: 2021 weeks 1..3 only (no prior season -> league mean)."""
        rows = []
        for wk, v in [(1, 0.0), (2, 0.2), (3, 0.1), (4, 0.3), (5, 0.4)]:
            rows.append(["2020_%02d" % wk, 2020, wk, "T", v, -v])
        for wk, v in [(1, 0.5), (2, 0.6), (3, 0.55), (4, 0.7), (5, 0.8)]:
            rows.append(["2021_%02d" % wk, 2021, wk, "T", v, -v])
        for wk, v in [(1, 0.9), (2, 0.85), (3, 0.95)]:
            rows.append(["2021u_%02d" % wk, 2021, wk, "U", v, -v])
        return pd.DataFrame(rows, columns=P.SIDE_COLS)

    def test_prior_ewm_uses_strictly_prior_weeks(self):
        w = add_asof(self._weekly())
        t = w[w["team"] == "T"].sort_values(["season", "week"])
        w4 = t[(t["season"] == 2021) & (t["week"] == 4)].iloc[0]
        self.assertEqual(w4["off_val_prior_n"], 3)
        # 3 priors < floor 4 -> prior-season (2020) weekly mean fallback
        self.assertAlmostEqual(w4["off_asof"], (0.0 + 0.2 + 0.1 + 0.3 + 0.4)
                               / 5)
        w5 = t[(t["season"] == 2021) & (t["week"] == 5)].iloc[0]
        self.assertEqual(w5["off_val_prior_n"], 4)
        # ewm(hl=2) over strictly-prior [0.5, 0.6, 0.55, 0.7]
        s = pd.Series([0.5, 0.6, 0.55, 0.7])
        exp = s.ewm(halflife=2, adjust=False).mean().iloc[-1]
        self.assertAlmostEqual(w5["off_asof"], float(exp))

    def test_league_mean_carry_when_no_prior_season(self):
        w = add_asof(self._weekly())
        u = w[w["team"] == "U"].sort_values("week")
        u1 = u[u["week"] == 1].iloc[0]
        self.assertEqual(u1["off_val_prior_n"], 0)
        league = float(w["off_val"].mean())
        self.assertAlmostEqual(u1["off_asof"], league)
        u3 = u[u["week"] == 3].iloc[0]
        self.assertEqual(u3["off_val_prior_n"], 2)   # < 4 -> not ewm yet
        self.assertAlmostEqual(u3["off_asof"], league)


# ---------------------------------------------------------------------------
# Pairing + strata + redundancy + ceiling + verdict routing
# ---------------------------------------------------------------------------
class TestPairingAndStrata(unittest.TestCase):
    def test_pairing_construction(self):
        wk = pd.DataFrame([
            ["g1", "H", 0.10, -0.20],   # H off 0.10, H def -0.20
            ["g1", "A", -0.05, 0.30],   # A off -0.05, A def 0.30
        ], columns=["game_id", "team", "off_asof", "def_asof"])
        uni = pd.DataFrame([["g1", "H", "A"]],
                           columns=["game_id", "home_team", "away_team"])
        cov = pair_covariates(wk, uni)
        self.assertAlmostEqual(cov["M_h"].iloc[0], 0.10 - 0.30)   # H off-A def
        self.assertAlmostEqual(cov["M_a"].iloc[0], -0.05 - (-0.20))
        self.assertAlmostEqual(cov["mismatch"].iloc[0],
                               max(abs(-0.20), abs(0.15)))

    def test_strata_partition(self):
        rng = np.random.default_rng(0)
        n = 100
        cov = pd.DataFrame({
            "binary": rng.uniform(0.3, 0.7, n),
            "derived": rng.uniform(0.3, 0.7, n),
            "home_win": rng.integers(0, 2, n).astype(float),
            "mismatch": rng.uniform(0, 1, n),
        })
        rows = strata_rows(cov)
        by = {r["label"]: r for r in rows}
        self.assertEqual(by["top_quartile_mismatch"]["n"]
                         + by["rest"]["n"], n)
        self.assertGreater(by["top_quartile_mismatch"]["mean_mismatch"],
                           by["rest"]["mean_mismatch"])
        for r in rows:
            self.assertIn("logloss", r)
            self.assertIn("ece", r)

    def test_redundancy_r2_high_and_low(self):
        rng = np.random.default_rng(1)
        n = 60
        feats = pd.DataFrame({"game_id": range(n),
                              "elo_diff": rng.normal(size=n),
                              "ewm_ypp_diff": rng.normal(size=n),
                              "ewm_net_pts_diff": rng.normal(size=n)})
        lo = pd.DataFrame({"game_id": range(n), "M_h": rng.normal(size=n),
                           "M_a": rng.normal(size=n)})
        hi = pd.DataFrame({"game_id": range(n),
                           "M_h": feats["elo_diff"] * 2.0 + 1.0,
                           "M_a": feats["elo_diff"] * -1.5})
        r_hi = redundancy_r2(hi, feats)
        r_lo = redundancy_r2(lo, feats)
        self.assertGreater(r_hi["r2_M_h"], 0.95)
        self.assertLess(r_lo["r2_M_h"], 0.5)

    def test_ceiling_models_pick_up_signal(self):
        rng = np.random.default_rng(2)
        n = 400
        M_h = rng.normal(size=n)
        p = 1 / (1 + np.exp(-M_h * 1.2))
        y = rng.binomial(1, p, n)
        cov = pd.DataFrame({"binary": rng.uniform(0.3, 0.7, n),
                            "home_win": y.astype(float),
                            "M_h": M_h, "M_a": rng.normal(size=n)})
        c = ceiling_models(cov)
        self.assertLess(c["delta_ll_vs_M0"], -0.01)
        self.assertEqual(c["n"], n)

    def test_verdict_routing(self):
        def _red(r2: float) -> dict:
            return {"r2_M_h": r2, "r2_M_a": r2 - 0.1}

        def _strata(gap: float) -> list[dict]:
            return [{"label": "top_quartile_mismatch", "n": 100,
                     "logloss": 0.60 + gap},
                    {"label": "rest", "n": 300, "logloss": 0.60}]

        def _ceil(delta: float) -> dict:
            return {"delta_ll_vs_M0": delta, "r2_of_y_minus_p": 0.0}

        go = pre_registered_verdict(_red(0.2), _strata(0.01), _ceil(-0.005),
                                    "F_x")
        self.assertEqual(go["verdict"], "GO")
        retest = pre_registered_verdict(_red(0.2), _strata(0.003),
                                        _ceil(-0.0005), "F_x")
        self.assertEqual(retest["verdict"], "RE_TEST_CANDIDATE")
        stop1 = pre_registered_verdict(_red(0.99), _strata(0.01),
                                       _ceil(-0.005), "F_x")
        self.assertEqual(stop1["verdict"], "STOP")       # redundant
        self.assertTrue(stop1["redundant"])
        stop2 = pre_registered_verdict(_red(0.2), _strata(0.001),
                                       _ceil(-0.0001), "F_x")
        self.assertEqual(stop2["verdict"], "STOP")       # ceiling weak


# ---------------------------------------------------------------------------
# Real-universe pins (skipped without artifacts/cache)
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_REAL, "20260904 CSVs and/or cached PBP / PFR / "
                     "features parquets not present — run the daily emission "
                     "and cache PBP + PFR advstats + the feature frame first")
class TestRealUniverse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = _probe()
        cls.fams = cls.o["families"]

    def test_universe_1376_with_full_binary_coverage(self):
        self.assertEqual(self.o["universe"]["n"], 1376)
        self.assertEqual(self.o["universe"]["n_binary_nonnull"], 1376)

    def test_all_six_families_full_coverage(self):
        self.assertEqual(len(self.fams), 6)
        for fam in FAMILIES:
            self.assertIn(fam, self.fams)
            self.assertEqual(self.fams[fam]["coverage"]["n_paired"], 1376)
            self.assertGreaterEqual(
                self.fams[fam]["coverage"]["pct_of_universe"], 99.9, fam)

    def test_no_redundant_family(self):
        """All six pairing families survive the redundancy gate: max R2 of
        M_h/M_a on elo_diff + ewm_ypp_diff + ewm_net_pts_diff is well under
        0.95 — none is a re-expression of the served inputs."""
        for fam in FAMILIES:
            r = self.fams[fam]["verdict"]["redundancy_r2_max"]
            self.assertIsNotNone(r, fam)
            self.assertLessEqual(r, 0.95, fam)

    def test_pre_registered_verdicts(self):
        """Record-pinned routing: no family GO; F3 (pressure) is the only
        RE_TEST_CANDIDATE (strata gap in [0.002, 0.008)); the rest STOP."""
        v = {fam: self.fams[fam]["verdict"]["verdict"] for fam in FAMILIES}
        self.assertNotIn("GO", v.values())
        self.assertEqual(v["F3_pressure"], "RE_TEST_CANDIDATE")
        for fam, vv in v.items():
            if fam != "F3_pressure":
                self.assertEqual(vv, "STOP", fam)

    def test_ceiling_deltas_all_below_go_bar(self):
        """No family's in-sample ceiling delta reaches the -0.004 GO bar
        (all > -0.004); the ceiling is an UPPER BOUND labeled as such."""
        for fam in FAMILIES:
            d = self.fams[fam]["ceiling"]["delta_ll_vs_M0"]
            self.assertGreater(d, P.CEILING_GO, fam)
            self.assertIn("UPPER BOUND",
                          self.fams[fam]["ceiling"]["label"].upper())

    def test_determinism_byte_identical(self):
        self.assertEqual(_run_once(), _run_once())


if __name__ == "__main__":
    unittest.main()
