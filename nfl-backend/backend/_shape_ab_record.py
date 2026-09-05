"""One-off builder: nfl_run_engine_diagnostics_shape_ab record from the
probe outputs (run 1 + run 2 determinism evidence). Not part of the suite."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
DD = BACKEND.parent / "data_delivery"


def _load_probe(path: str) -> dict:
    raw = Path(path).read_text(encoding="utf-8")
    i = raw.find("{")
    if i < 0:
        raise RuntimeError(f"{path}: no JSON object found")
    return json.loads(raw[i:])


def _raw_json_part(path: str) -> tuple[str, str]:
    raw = Path(path).read_text(encoding="utf-8")
    i = raw.find("{")
    if i < 0:
        raise RuntimeError(f"{path}: no JSON object found")
    return raw[i:], raw[:i]


def main() -> int:
    out1_path = sys.argv[1]
    out2_path = sys.argv[2] if len(sys.argv) > 2 else None
    p1 = _load_probe(out1_path)
    det = {"run1_sha256": p1["determinism"]["sha256"]}
    if out2_path:
        p2 = _load_probe(out2_path)
        raw1, banner1 = _raw_json_part(out1_path)
        raw2, banner2 = _raw_json_part(out2_path)
        det["run2_sha256"] = p2["determinism"]["sha256"]
        det["raw_byte_identical"] = raw1 == raw2
        det["sha256_identical"] = (
            p1["determinism"]["sha256"] == p2["determinism"]["sha256"])
        det["banner_bytes_stripped_run1"] = len(banner1)
        det["banner_bytes_stripped_run2"] = len(banner2)
        det["note"] = ("raw JSON-part byte equality across two full probe "
                       "invocations (32 exact joint passes over 1,376 games "
                       "each); no timing fields in the output (the probe's "
                       "elapsed_s was removed for determinism)")
    else:
        det["run2_sha256"] = None
        det["raw_byte_identical"] = None
        det["sha256_identical"] = None

    rec = {
        "record": "nfl_run_engine_diagnostics_shape_ab",
        "target": ("run-engine under-recovery Variant A/B — find a shape fix "
                   "(sigma stretch / center-gap stretch / additive "
                   "quality-uncertainty variance) that restores the recovered "
                   "quality-margin spread (audit c1a7c12: 59-67%) to ~100% "
                   "without degrading totals or derived-ML calibration"),
        "frame_sha256": "3e8c8a510f04",
        "wired": "2026-09-04",
        "scope": ("READ-ONLY A/B against the pinned 1,376-row decided OOF "
                  "store — no production/model change unless GATE_PASS "
                  "(separate adoption commit). Evidence = decided OOF rows "
                  "only. Harness: nfl-backend/backend/"
                  "probe_run_engine_shape_ab.py (exact 76x76 joint "
                  "machinery from the slate emitter; deterministic)."),
        "mechanism_mapping": {
            "recovered_spread_metric": ("audit methodology (MLB 39c865e "
                "mirror): mean margin per sextile of each quality feature; "
                "recovery % = predicted sextile-5-minus-0 spread / actual "
                "spread, averaged over elo_diff / ewm_net_pts_diff / "
                "win_pct_diff / ewm_ypp_diff"),
            "knobs": {
                "per_side_means": ("mu_h / mu_a from the era walk (E2 "
                    "ewm_2w centers + 12-pool diff LGBM, median rounds "
                    "20/23) — carried unrounded in the OOF artifact"),
                "joint": ("pinned DN const-sigma joint: sigma_h 9.663 / "
                    "sigma_a 9.0789 / rho 0.0076 / tie 0.275% -> margin "
                    "sigma 13.2086, totals sigma 13.3092"),
                "derived_ml": "P(H>A)/(1-P_tie) from the calibrated 76x76 joint",
                "totals": ("P(total > line) from the total PMF of the SAME "
                    "joint — any per-side sigma change moves totals too "
                    "(the gate's cross-check)"),
            },
            "variant_definitions": {
                "V1": ("per-side const sigma x k (k in 1.00..1.50 step "
                    "0.05) — scales BOTH margin and totals sigma (honest DN "
                    "stretch); mu pair untouched"),
                "V2": ("gap scale k around the pair mean: mu'_h = m + "
                    "k(mu_h - m), mu'_a = m + k(mu_a - m) — total mean "
                    "preserved exactly, joint params pinned; k in "
                    "1.00..1.80 step 0.05"),
                "V3": ("global additive quality-uncertainty variance: "
                    "sigma0' = sqrt(sigma0^2 + c*mean(mu_gap^2)), c in "
                    "{0.1, 0.5, 1.0} — the pinned const-const joint cannot "
                    "carry per-game sigma (documented limitation); "
                    "last-resort arm"),
            },
        },
        "r0_gate": p1["r0_gate"],
        "universe": p1["universe"],
        "variants": p1["variants"],
        "gate": p1["gate"],
        "verdict": {
            "result": "GATE_FAIL — no variant clears the pre-registered "
                      "leg set; production shape unchanged",
            "leg_summary": {
                "leg1_recovery_90_110": ("only V2 moves it (63.5 -> 114.5% "
                    "across k=1.0..1.8; band crossed at k ~1.42-1.72); V1 "
                    "and V3 leave recovery at 63.5% — sigma shape does not "
                    "move the mu response"),
                "leg2_derived_ml_plusminus_0_002": ("binding for every "
                    "variant: V2 at the recovery crossing pays +0.008..+0.02 "
                    "pooled ll and +0.015..+0.04 sealed ll; V1 pays growing "
                    "pooled ll + ECE from k>=1.05"),
                "leg3_totals_ece_not_worse": ("V2 holds totals ECE "
                    "(0.087->0.0847 pooled, 0.1547->0.1459 sealed); V1/V3 "
                    "improve totals ECE by probability compression, which is "
                    "not a fix (it fails leg2 harder)"),
                "leg4_determinism": "see determinism block (raw byte-identical double run)",
            },
            "finding": ("the sigma-compression is not a shape parameter: "
                "the mu RESPONSE is compressed (V2 restores recovery by "
                "stretching the gap) but every stretch that reaches the "
                "90-110% band pays a derived-ML logloss cost far beyond the "
                "+/-0.002 leg — the same pooled-vs-sealed tension that "
                "blocked the global calibration swaps (e3aeece, fc4f4bd). "
                "Totals calibration is NOT the constraint (V2 holds it); "
                "derived-ML sharpness is."),
            "follow_ons": [
                "the remaining lever is the NFL-P1 opposing-quality LEVEL "
                "input arm (audit recommendation c1a7c12) — a feature-level "
                "response fix, not a post-hoc shape transform",
                "season-start prior/anchoring fix (audit complementary arm) "
                "remains cheap and orthogonal",
                "revisit when 2026 decided rows exist (~272/yr): the "
                "diagnostic gets stronger and the sealed band legs are the "
                "least certain",
            ],
        },
        "determinism": det,
        "judgment_calls": [
            "recovered-spread gate uses the mean over the 4 audit features "
            "(per-feature ratios reported in the table)",
            "V3 is defined with a GLOBAL additive variance (the pinned "
            "const-const joint schema cannot carry per-game sigma); it is "
            "reported for completeness and its verdict is unchanged by the "
            "limitation (it never approaches the band)",
            "leg3 compares against the MEASURED V0 totals ECE (0.087/0.1547), "
            "not the rounded record pins",
            "V0 machinery fidelity: max-abs-diff 5e-7 between the rebuilt "
            "joint derived_ml and the artifact column — the variant passes "
            "are exact, no analytic shortcut",
        ],
        "evidence_sources": [
            "nfl_run_engine_markets_20260904.csv kind==oof (1,376 rows: mu "
            "pair, derived_ml, p_over/p_cover_offered, y_*, lines, actuals, "
            "frame_view)",
            "canonical decided feature frame via load_features (recovery "
            "features elo_diff / ewm_net_pts_diff / win_pct_diff / "
            "ewm_ypp_diff; read-only cache)",
            "nfl_joint_engine.build_joint_pmfs + nfl_slate_engine pinned "
            "params (exact 76x76 calibrated DN joints)",
            "nfl_market_engine.totals_calibration / covers_calibration "
            "(the record's ECE conventions)",
            "nfl_margin_audit_3e8c8a510f04.json (recovery methodology + "
            "sextile pins)",
        ],
        "guardrails_met": {
            "1_no_hardcoded_stretch": ("no production change made — "
                "GATE_FAIL; any future adoption parameter must come from a "
                "recorded fit, never a hardcoded factor"),
            "2_totals_in_gate": ("totals ECE scored at the offered lines on "
                "every variant pass, pooled + sealed (leg3)"),
            "3_read_only_until_gate": ("probe/tests/record only; emission "
                "core untouched"),
            "4_determinism": "byte-identical double emit (see determinism)",
        },
    }
    out = DD / "nfl_run_engine_diagnostics_shape_ab_3e8c8a510f04.json"
    out.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    print("sha256:", hashlib.sha256(
        json.dumps(rec, indent=2, default=str).encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())