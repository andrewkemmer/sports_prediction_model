"""Tests for the worth-having review of RAW_ADDED and QB arms.

Pins the worth-having computation (TOL/3 razor-thin check, direction
agreement, pooled corroboration) and the record's machine-verified
consistency against the reauth audit's clean surfaces.  No harness arms
are added — both arms fail worth-having, so the served pool is unchanged.
"""
import json
from pathlib import Path

import pytest

RECORD_PATH = (Path(__file__).resolve().parent.parent
               / "data_delivery"
               / "nfl_worth_having_raw_qb_689c93da35b5.json")
AUDIT_PATH = (Path(__file__).resolve().parent.parent
              / "data_delivery"
              / "nfl_blend_reauth_audit_689c93da35b5.json")

TOL = {"ll": 0.012, "auc": 0.016, "ece": 0.01}
THIRD = {k: v / 3.0 for k, v in TOL.items()}

# Direction: ll/ece lower is better, auc higher is better
_BETTER = {"pooled_ll": -1, "pooled_auc": 1, "pooled_ece": -1}


def _load_record():
    return json.loads(RECORD_PATH.read_text())


def _load_audit():
    return json.loads(AUDIT_PATH.read_text())


def _audit_surfaces(audit, arm_key):
    """Pull clean pooled/sealed surfaces for an arm from the reauth audit."""
    runs = audit["runs"]
    # Map arm -> cache key (content-identity from the audit)
    _MAP = {
        "C0": "plain|cdb83c684ea4",
        "RAW_ADDED": "masked|7dbf8b0d4e8d",
        "QB": "plain|fdf3840ea72b",
    }
    r = runs[_MAP[arm_key]]
    return r["pooled_model_platt"], r["sealed_model_platt"]


def _worth_having_check(deltas):
    """Recompute the worth-having verdict from deltas (machine-verified)."""
    near_pooled = [leg for leg in ("pooled_ll", "pooled_auc", "pooled_ece")
                   if abs(deltas[leg]) <= THIRD[leg.replace("pooled_", "")]
                   or (leg == "pooled_ece"
                       and abs(deltas[leg]) <= THIRD["ece"])]
    near_sealed = [leg for leg in ("sealed_ll", "sealed_auc", "sealed_ece")
                   if abs(deltas[leg]) <= THIRD[leg.replace("sealed_", "")
                                                if "ece" not in leg
                                                else "ece"]]

    dir_agree = {}
    for base, seal in [("pooled_ll", "sealed_ll"),
                       ("pooled_auc", "sealed_auc"),
                       ("pooled_ece", "sealed_ece")]:
        pd, sd = deltas[base], deltas[seal]
        dir_agree[base.replace("pooled_", "")] = (
            (pd >= 0 and sd >= 0) or (pd < 0 and sd < 0))

    pooled_dir_ok = all(
        deltas[leg] * _BETTER[leg] >= -THIRD[leg.replace("pooled_", "")
                                             if "ece" not in leg
                                             else "ece"]
        for leg in ("pooled_ll", "pooled_auc", "pooled_ece"))

    return {
        "near_edge_pooled": near_pooled,
        "near_edge_sealed": near_sealed,
        "direction_agree": dir_agree,
        "pooled_dir_ok": pooled_dir_ok,
        "worth_having": (len(near_pooled) == 0 and len(near_sealed) == 0
                         and all(dir_agree.values()) and pooled_dir_ok),
    }


class TestRecordExists:
    def test_record_file_exists(self):
        assert RECORD_PATH.exists(), f"record not found: {RECORD_PATH}"

    def test_audit_file_exists(self):
        assert AUDIT_PATH.exists(), f"audit not found: {AUDIT_PATH}"

    def test_record_frame_matches_audit(self):
        rec = _load_record()
        audit = _load_audit()
        assert rec["frame_sha256"] == audit["frame_sha256"]


class TestDeltasMachineVerified:
    """Every delta in the record must match the audit's clean surfaces."""

    def test_raw_added_deltas(self):
        rec = _load_record()
        audit = _load_audit()
        cand_p, cand_s = _audit_surfaces(audit, "RAW_ADDED")
        base_p, base_s = _audit_surfaces(audit, "C0")
        rec_d = rec["arms"]["RAW_ADDED"]["deltas_vs_c0"]
        for leg, (c, b) in {
            "pooled_ll": (cand_p["logloss"], base_p["logloss"]),
            "pooled_auc": (cand_p["auc"], base_p["auc"]),
            "pooled_ece": (cand_p["ece"], base_p["ece"]),
            "sealed_ll": (cand_s["logloss"], base_s["logloss"]),
            "sealed_auc": (cand_s["auc"], base_s["auc"]),
            "sealed_ece": (cand_s["ece"], base_s["ece"]),
        }.items():
            expected = round(c - b, 6)
            assert rec_d[leg] == expected, (
                f"RAW_ADDED {leg}: record {rec_d[leg]} != computed {expected}")

    def test_qb_deltas(self):
        rec = _load_record()
        audit = _load_audit()
        cand_p, cand_s = _audit_surfaces(audit, "QB")
        base_p, base_s = _audit_surfaces(audit, "C0")
        rec_d = rec["arms"]["QB"]["deltas_vs_c0"]
        for leg, (c, b) in {
            "pooled_ll": (cand_p["logloss"], base_p["logloss"]),
            "pooled_auc": (cand_p["auc"], base_p["auc"]),
            "pooled_ece": (cand_p["ece"], base_p["ece"]),
            "sealed_ll": (cand_s["logloss"], base_s["logloss"]),
            "sealed_auc": (cand_s["auc"], base_s["auc"]),
            "sealed_ece": (cand_s["ece"], base_s["ece"]),
        }.items():
            expected = round(c - b, 6)
            assert rec_d[leg] == expected, (
                f"QB {leg}: record {rec_d[leg]} != computed {expected}")


class TestWorthHavingVerdicts:
    """Both arms must fail the worth-having bar."""

    def test_raw_added_not_worth_having(self):
        rec = _load_record()
        wh = rec["arms"]["RAW_ADDED"]["worth_having_check"]
        assert wh["worth_having"] is False

    def test_qb_not_worth_having(self):
        rec = _load_record()
        wh = rec["arms"]["QB"]["worth_having_check"]
        assert wh["worth_having"] is False

    def test_raw_added_pooled_razor_thin(self):
        rec = _load_record()
        wh = rec["arms"]["RAW_ADDED"]["worth_having_check"]
        assert "pooled_ll" in wh["near_edge_pooled_legs"]
        assert "pooled_auc" in wh["near_edge_pooled_legs"]

    def test_qb_all_pooled_razor_thin(self):
        rec = _load_record()
        wh = rec["arms"]["QB"]["worth_having_check"]
        assert "pooled_ll" in wh["near_edge_pooled_legs"]
        assert "pooled_auc" in wh["near_edge_pooled_legs"]
        assert "pooled_ece" in wh["near_edge_pooled_legs"]

    def test_ece_direction_disagrees_both_arms(self):
        """ECE must disagree between pooled (improves) and sealed (degrades)."""
        rec = _load_record()
        for arm in ("RAW_ADDED", "QB"):
            wh = rec["arms"][arm]["worth_having_check"]
            assert wh["direction_agreement"]["ece"] is False, (
                f"{arm} ECE direction should disagree")

    def test_machine_verified_against_recomputation(self):
        """Recompute from raw deltas and verify the record's verdict."""
        rec = _load_record()
        for arm in ("RAW_ADDED", "QB"):
            deltas = rec["arms"][arm]["deltas_vs_c0"]
            mv = _worth_having_check(deltas)
            rec_wh = rec["arms"][arm]["worth_having_check"]
            assert mv["worth_having"] == rec_wh["worth_having"], (
                f"{arm}: machine-verified {mv['worth_having']} != "
                f"record {rec_wh['worth_having']}")
            assert set(mv["near_edge_pooled"]) == set(
                rec_wh["near_edge_pooled_legs"]), (
                    f"{arm}: near-edge pooled legs differ")


class TestArmConstruction:
    """Pin arm geometry: feature counts and added-column sets."""

    def test_raw_added_26_cols(self):
        rec = _load_record()
        assert rec["arms"]["RAW_ADDED"]["feature_count"] == 26

    def test_qb_13_cols(self):
        rec = _load_record()
        assert rec["arms"]["QB"]["feature_count"] == 13

    def test_raw_added_adds_14(self):
        rec = _load_record()
        assert len(rec["arms"]["RAW_ADDED"]["features_added"]) == 14

    def test_qb_adds_1(self):
        rec = _load_record()
        assert rec["arms"]["QB"]["features_added"] == [
            "ewm_qb_epa_starter_diff"]

    def test_served_pool_unchanged(self):
        rec = _load_record()
        assert rec["decision"]["final_pool"] == [
            "elo_diff", "win_pct_diff", "rest_days_diff", "is_dome_home",
            "ewm_net_pts_diff", "ewm_ypp_diff", "pace_plays_min_diff",
            "rest_short_diff", "div_game", "travel_miles_diff",
            "altitude_home", "prime_time"]


class TestNoAdoption:
    """No adoption is wired by this commit."""

    def test_no_adopted_arms(self):
        rec = _load_record()
        assert rec["decision"]["adopted_arms"] == []

    def test_no_change_verdict(self):
        rec = _load_record()
        assert "NO-CHANGE" in rec["decision"]["verdict"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
