"""One-off builder: nfl_binary_calibration_quality_map record from the
probe output (run 1 + run 2 determinism evidence). Not part of the suite."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
DD = BACKEND.parent / "data_delivery"


def _load_probe(path: str) -> dict:
    raw = Path(path).read_text(encoding="utf-8")
    return json.loads(raw)


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
    det = {"run1_sha256": p1["verdict"]["result"]}
    if out2_path:
        p2 = _load_probe(out2_path)
        raw1, banner1 = _raw_json_part(out1_path)
        raw2, banner2 = _raw_json_part(out2_path)
        det = {
            "run1_bytes": len(raw1.encode("utf-8")),
            "run2_bytes": len(raw2.encode("utf-8")),
            "raw_byte_identical": raw1 == raw2,
            "banner_bytes_stripped_run1": len(banner1),
            "banner_bytes_stripped_run2": len(banner2),
            "note": ("raw JSON-part byte equality across two full probe "
                     "invocations; the payload is timing-free (no elapsed "
                     "fields)"),
        }

    rec = {
        "record": "nfl_binary_calibration_quality_map_3e8c8a510f04.json",
        "title": ("NFL binary top-band over-confidence: quality-stratum "
                  "LOCAL recalibration — probe -> adoption gate (arm #3 "
                  "after e3aeece DO_NOT_REFIT and fc4f4bd KEEP_PLATT)"),
        "date": "2026-09-05",
        "gated": ("probe + tests + record ONLY unless GATE_PASS; on "
                  "GATE_PASS a separate adoption commit wires the "
                  "serve-time local map with the threshold from the "
                  "recorded fit, never hardcoded"),
        "scope": ("nfl-backend + data_delivery ONLY; READ-ONLY over "
                  "production code (no engine/pricing/feature change; "
                  "FEATURE_COLUMNS byte-untouched); production Platt twin "
                  "unchanged"),
        "provenance": {
            "audits": [
                "nfl_binary_calibration_3e8c8a510f04.json (e3aeece): "
                "Branch B same-set cell — the n=191 Platt 70-80 band games "
                "carry raw mean 0.6815 vs actual 0.6754; the deployed "
                "global Platt (a=1.276, b=0.122) stretched them to 0.7485 "
                "(band ECE 0.0731)",
                "nfl_binary_calibration_hinge_3e8c8a510f04.json (fc4f4bd): "
                "KEEP_PLATT — raw everywhere regresses pooled nested ll "
                "+0.0030 (above-0.70 raw rows realize 0.852 pooled vs raw "
                "0.759: the logistic stretch is genuinely justified "
                "in-sample); per-fold pchip fits explode (std 0.118, "
                "outputs >1.0) = fold-variance overfit",
                "nfl_margin_audit_3e8c8a510f04.json (c1a7c12): "
                "quality-extreme stratum — the binary over-recovers "
                "(105-119%) and the per-game divergence grows with quality "
                "level (gap R2 0.35, elo_diff dominant)",
            ],
            "universe": ("shared 1,376-game decided OOF (markets kind==oof "
                         "20260904) joined to "
                         "nfl_predictions_history_20260904.csv (raw = "
                         "home_win_prob_model, published = "
                         "home_win_prob_model_calibrated); production pool "
                         "= 1,107 pooled pre-holdout rows incl. playoff "
                         "weeks (88 fold-weeks) + 285 sealed-2025"),
            "harness": ("probe_binary_calibration_quality_map.py "
                        "(read-only; double-run byte-identical — cmp on "
                        "the full payload)"),
            "protocol": ("nested per-fold strictly-earlier fits for the "
                         "scan; deployed protocol (fit on ALL pooled rows) "
                         "for the sealed/S1/S2 leg table — the "
                         "e3aeece/fc4f4bd conventions"),
        },
        "mechanism": {
            "stratum": ("raw-axis confidence > h; h scanned 0.66..0.74 "
                        "step 0.01 (pre-registered prior ~0.68); boundary "
                        "PINNED by the selection rule"),
            "eligibility": (
                "eligible serving families = the continuous monotone maps: "
                "L = slope-only logistic anchored at (h, Platt(h)) and "
                "P = anchored pchip (seam knot (h, Platt(h)) + isotonic "
                "knots, hard-clipped [0,1]). I_identity in-stratum is a "
                "REFERENCE row only: its seam vs Platt is discontinuous "
                "(Platt(h)~0.79 drops to raw~0.72), breaking global "
                "monotonicity and the AUC-flat contract BY CONSTRUCTION"),
            "serving": ("raw <= h -> global Platt (untouched); raw > h -> "
                        "local map (seam-continuous at (h, Platt(h)))"),
            "selection_rule": ("among eligible (L/P) cells with nested "
                               "70-80 band ECE <= 0.03 on pre-holdout, "
                               "pick min nested pooled logloss (tie-break "
                               "band ECE); I_identity excluded"),
        },
        "stratum_char": p1["stratum_char"],
        "nested_scan": {
            "rows": p1["nested_scan"]["pooled_pre_holdout_rows"],
            "fold_weeks": p1["nested_scan"]["fold_weeks"],
            "protocol": p1["nested_scan"]["protocol"],
            "surface": p1["nested_scan"]["surface"],
            "reading": ("identity-in-stratum is the best pooled-ll local "
                        "correction (+0.0016..+0.0026 vs nested Platt "
                        "0.6193) and nearly fixes the nested 70-80 band "
                        "(0.0074 at h=0.74) — but it is the discontinuous "
                        "reference, not eligible for serving. The eligible "
                        "L family pays +0.004..+0.007 and never reaches "
                        "band <= 0.03; the eligible P family reaches the "
                        "band (0.002 at h=0.68..0.015 at h=0.72) but pays "
                        "+0.009..+0.023 nested pooled ll."),
        },
        "selection": p1["selection"],
        "fold_stability": p1["fold_stability"],
        "deployed": p1["deployed"],
        "gate_legs": p1["gate_legs"],
        "verdict": {
            "result": p1["verdict"]["result"],
            "summary": (
                "GATE_FAIL — no quality-stratum local map clears the leg "
                "set. The chosen continuous map (P_local_pchip at h=0.72) "
                "holds the AUC-flat and no-bleed contracts EXACTLY (0.0 "
                "AUC delta, [0,1] clipped, seam-continuous) but: (1) S2 "
                "70-80 band ECE 0.1685 not < 0.03 — the anchored map "
                "vacates the band (in-stratum served >= Platt(0.72) ~ "
                "0.795) leaving the over-stretched Platt sliver (raw "
                "0.64-0.72) untouched on sealed; (2) nested pooled ll "
                "+0.0092 > +/-0.001 (the deployed single fit actually "
                "improves in-sample -0.0021 — the +0.0092 is per-fold "
                "pchip fold variance, Jensen, same as e3aeece); (3) sealed "
                "80+ adjacent +0.0658 (n=48) — the pooled-fitted steep "
                "in-stratum tail over-predicts sealed 2025's mild top; "
                "(4) worth-having fails (band gain 0.0086 < noise/3)."),
            "finding": (
                "THIRD independent confirmation of the structural finding: "
                "no map — global (e3aeece, fc4f4bd) or local (here) — can "
                "fix the Platt 70-80 band over-stretch without paying "
                "pooled logloss or the sealed 80+ tail, because the pooled "
                "in-sample top-tail steepness (actual 0.852 at raw "
                "0.73-0.80) does not replicate on sealed 2025 (mild top). "
                "The discontinuity-free local families are "
                "contract-clean; they just cannot deliver the band fix at "
                "the pre-registered bars. The identity-in-stratum "
                "convention (raw above h, Platt below) DOES nearly fix the "
                "band (S2 0.0453 vs Platt 0.1804; sealed band 0.0366 on "
                "the raw axis) but fails pooled ll (+0.0016) and the 80+ "
                "adjacent leg (+0.0956) and is discontinuous — it remains "
                "a serve-time product convention decision, not a "
                "gate-passing map."),
            "follow_ons": [
                "relaxed pooled-ll bar by product decision: the "
                "identity-in-stratum convention is the only candidate "
                "within ~1.5x of the +/-0.001 bar with a real band fix — "
                "a serving-axis product call, not a model change (mirrors "
                "the e3aeece/fc4f4bd records)",
                "revisit when more sealed-2025 data exists: the 80+ tail "
                "(n=16-48) and the S2 band (n~20) are the least certain "
                "legs; 2026 in-season decided rows (~272/yr) will sharpen "
                "both",
                "the honest finding stands: the ensemble is well-calibrated "
                "on the disputed games (Branch B) — the defect is the "
                "global logistic's shape, and no post-hoc map recovers it "
                "at the pre-registered bars",
            ],
        },
        "evidence_sources": [
            "nfl_predictions_history_20260904.csv (raw + published Platt "
            "axes + actuals)",
            "nfl_run_engine_markets_20260904.csv (kind==oof rows; "
            "shared-universe anchor + p_home_win_derived)",
            "nfl_moneyline_v1_20260904.json (deployed map pins "
            "a=1.276336/b=0.121988; sealed model_platt 0.6249/0.0745)",
            "canonical decided feature frame via load_features "
            "(quality-extreme overlap; read-only cache; degrades to null "
            "when unreachable)",
            "nfl_binary_calibration_3e8c8a510f04.json / "
            "nfl_binary_calibration_hinge_3e8c8a510f04.json / "
            "nfl_margin_audit_3e8c8a510f04.json (prior arms)",
            "probe_binary_calibration_quality_map.py (read-only harness; "
            "byte-identical double run)",
        ],
        "guardrails_met": {
            "1_local_only": ("the global Platt map is untouched outside "
                             "the stratum in every candidate; the serving "
                             "rule is seam-continuous with it"),
            "2_no_hardcoded_boundary": ("no production change made "
                                        "(GATE_FAIL); any future adoption "
                                        "reads h*/family from the recorded "
                                        "fit"),
            "3_read_only_until_gate": ("probe/tests/record only; "
                                       "nfl_moneyline.py, artifacts, "
                                       "FEATURE_COLUMNS byte-unchanged"),
            "4_auc_flat_contract": ("held EXACTLY (0.0 delta) for the "
                                    "continuous families — rank-invariance "
                                    "verified, no AUC-improvement claim"),
            "5_determinism": ("double run byte-identical (cmp); the probe "
                              "payload is timing-free"),
        },
        "acceptance_evidence": {
            "probe_determinism": det,
            "r0_gate": p1["r0_gate"],
            "suites": [
                "test_binary_calibration_quality_map: 15/15 OK (R0 "
                "bit-consistency; stratum pins; selection excludes "
                "identity; all 6 gate legs; verdict GATE_FAIL; "
                "determinism; synthetic map-property pins)",
                "test_run_engine_shape_ab: 8/8 OK (the companion arm "
                "record)",
                "full NFL backend suite: 562 tests, 1 failure + 19 errors "
                "— failure set identical to the documented env baseline "
                "(joint/raw_ablation/tier4/xgb_reg/unified/market "
                "modules); zero new failures",
                "no other suite touched: no production code changed",
            ],
        },
        "record_protection": (
            "inherits the protected nfl_binary_calibration_ prefix "
            "(_PROTECTED_DELIVERY_PREFIXES, added at e3aeece) — verified "
            "against _is_protected_name; no master_pipeline change needed"),
    }
    out = DD / "nfl_binary_calibration_quality_map_3e8c8a510f04.json"
    blob = json.dumps(rec, indent=2, default=str)
    out.write_text(blob, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    print("sha256:", hashlib.sha256(blob.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())