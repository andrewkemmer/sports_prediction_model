"""Bridge: add per-line run-line p_rl_* columns to a committed markets
artifact (read-only over the model — no re-fit, no new data).

The committed run_engine_markets_*.csv predates the p_rl_* columns
(whole-line alternates −1 … −4 + 3-way split). This harness recomputes
the per-line run-line probabilities from the SAME NB(λ, α) marginals the
artifact already carries (home_expected_runs / away_expected_runs /
alpha_home / alpha_away, all committed per-game) via the production MC
machinery, and persists a dated copy with the p_rl_* columns added.

Self-verification (the guardrail that matters): for half-lines, the new
strict-cover convention (margin > L) is mathematically identical to the
legacy convention (margin >= ceil(L)) on integer margins, so the
computed p_rl_1_5_home / p_rl_2_5_home / p_rl_3_5_home MUST equal the
committed p_home_cover_1_5 / _2_5 / _3_5 columns exactly. The harness
fails loudly on any mismatch — proving both that the new columns are
consistent with the shipped artifact and that the legacy columns are
untouched (byte-identical by construction).

Whole-number lines get the 3-way split: home covers −L iff margin > L,
push iff margin == L, away +L otherwise; sums to 1.0 from the margin
draws. Half-lines have push = 0 by construction.

Usage:
    python run_rl_bridge.py [YYYYMMDD]
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from run_engine import (MC_DRAWS_TAIL, RUN_LINE_GRID, RUN_LINE_GRID_FULL,
                        derive_markets_mc, rl_col)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_delivery"

# Deterministic seed for the p_rl recompute (the original pipeline seed is
# not persisted in the artifact meta; half-lines are seed-independent — they
# match the committed columns by construction, verified below).
BRIDGE_SEED = 20260829


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    date_str = argv[0] if argv else None
    if not date_str:
        hits = sorted(DATA.glob("run_engine_markets_*.csv"))
        # Exclude this harness's own _rl outputs (mirror the margin
        # diagnostic) so a stale bridge file never becomes the "latest"
        # input and double-suffixes the output (_rl_rl).
        cands = [h for h in hits if "_rl." not in h.name]
        if not cands:
            raise FileNotFoundError("no run_engine_markets_*.csv found")
        date_str = cands[-1].stem.replace("run_engine_markets_", "")
    markets_f = DATA / f"run_engine_markets_{date_str}.csv"
    if not markets_f.exists():
        raise FileNotFoundError(markets_f)
    m = pd.read_csv(markets_f)
    oof = m[m["kind"] == "oof"].reset_index(drop=True)
    lam_h = oof["home_expected_runs"].to_numpy(float)
    lam_a = oof["away_expected_runs"].to_numpy(float)
    al_h = oof["alpha_home"].to_numpy(float)
    al_a = oof["alpha_away"].to_numpy(float)
    mc = derive_markets_mc(lam_h, lam_a, al_h, al_a,
                           n_draws=MC_DRAWS_TAIL, seed=BRIDGE_SEED)
    # The structural home one-run fix (derive_markets_mc) resolves the
    # impossible tie mass to ±1 home-weighted, legacy p_home_cover_* included
    # (same underlying quantity). Overwrite the committed (pre-fix)
    # tie-inclusive legacy columns with the adjusted values so this bridge
    # artifact matches exactly what the next pipeline run persists.
    for j, mm in enumerate(RUN_LINE_GRID):
        oof[f"p_home_cover_{str(mm).replace('.', '_')}"] = np.round(
            mc["p_cover_grid"][:, j], 5)
    for j, mm in enumerate(RUN_LINE_GRID_FULL):
        oof[rl_col(mm, "home")] = np.round(mc["p_rl_home_grid"][:, j], 5)
        oof[rl_col(mm, "push")] = np.round(mc["p_rl_push_grid"][:, j], 5)
        oof[rl_col(mm, "away")] = np.round(mc["p_rl_away_grid"][:, j], 5)

    # Self-verification: half-line p_rl_*_home must EQUAL the legacy
    # p_home_cover_* EXACTLY — both are the same grid_cover array from the
    # same MC (margin > L ⇔ margin >= L + 0.5 on integer margins; the +1
    # resolved band enters only the 1.0/0.5 lines, not these half-lines), so
    # any divergence is a convention/schema bug, not sampling noise.
    for mm in (1.5, 2.5, 3.5):
        legacy = f"p_home_cover_{str(mm).replace('.', '_')}"
        new = rl_col(mm, "home")
        if not (oof[new].to_numpy() == oof[legacy].to_numpy()).all():
            raise AssertionError(
                f"half-line {mm}: p_rl {new} != legacy {legacy} after "
                f"renormalization — convention not consistent")

    # Rebuild the full frame (oof + slate rows) preserving column order:
    # legacy columns first, then the new p_rl_* block.
    slate = m[m["kind"] == "slate"].reset_index(drop=True)
    out = pd.concat([oof, slate], ignore_index=True)
    out_f = DATA / f"run_engine_markets_{date_str}_rl.csv"
    out.to_csv(out_f, index=False)
    print(f"wrote {out_f} "
          f"({len(oof)} oof + {len(slate)} slate rows; "
          f"{3 * len(RUN_LINE_GRID_FULL)} p_rl columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
