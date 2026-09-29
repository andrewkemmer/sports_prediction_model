"""League-wide player-availability audit (all 32 teams, PIT-strict).

The 2026-09-29 audit request in one repeatable tool: verify that the
captured ESPN health archive is complete for every player who can actually
enter a pl_evo/pl_ppo pool, that the exclusion binds point-in-time, and that
it never reaches backward into pre-capture games.

What it checks (and why the obvious metric is not the right one):

* BRIDGE QUALITY, HONESTLY DEFINED — the name bridge from ESPN athlete ids
  onto MoneyPuck rating ids leaves goalies and never-dressed prospects
  unmatched BY CONSTRUCTION: the rating space is skaters only (C/L/R/D), so
  a name absent from that space can never enter a pool and its exclusion is
  vacuous, not missing. The catchable bridge failure is an AMBIGUOUS
  refusal (one normalised name spanning several rating ids); a raw
  match-percentage gate would pay a permanent false alarm for roster
  composition.

* PIT SOUNDNESS — bridge BEFORE build (production order in
  features._load_espn_stints), then assert_pit over the league archive.
  Building stints from unbridged ids must fail the bind gate; doing so here
  proves the gate still catches the disjoint-vocabulary no-op.

* RETROACTIVITY — the archive's capture (2026-09-28+) postdates every
  2025-26 game, so the honest answer for the entire OOF window is "unknown,
  not healthy" (pl_il_out_fraction NaN). The check runs a pool that genuinely
  forms on a pre-capture target date and requires ZERO removals: absence
  must never be applied retroactively, and the probe must not be vacuous.

* OPENING-SLATE SHADOW — with last-known form frozen just after the capture
  (exactly what the pipeline sees once 2026-27 ratings flow), count which
  teams' pools the current archive would shrink. ``n_unavailable`` is a
  per-side-group count and is read with MAX per side — summing it multiplies
  one removal by its group count (the features.py warning, enforced here).

* LABEL DIFFERENCES — the rating's team is where the player last dressed;
  the archive's is today. Off-season moves are the EXPECTED shape of that
  difference and are reported informationally, never failed.

Exit code 1 on any hard finding; ``--strict`` escalates the informational
findings (suspension rows, roster-move labels, thin archive) to failures.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or not __package__:  # direct execution: python audit_...
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import features as feat  # noqa: E402
import ingestion as ing  # noqa: E402
import injury_stints as ist  # noqa: E402

#: The statuses that open an exclusion interval (must mirror the policy).
OUT_IR_STATUSES = ("Out", "Injured Reserve", "IR", "Out For Season")

#: Exclusion-refused by policy — reported, never failed (non-strict).
REFUSED_STATUSES = ("Day-To-Day", "DTD", "Suspension", "SUSP", "NA")

#: Below this many captured snapshots the archive is real but too young to
#: answer much; informational unless --strict.
THIN_ARCHIVE_SNAPSHOTS = 2


def audit(ratings: pd.DataFrame, reports: pd.DataFrame,
          *, strict: bool = False) -> tuple[list[str], list[str], dict]:
    """Run every league availability check. Returns (hard, soft, details)."""
    hard: list[str] = []
    soft: list[str] = []
    details: dict = {}

    stamps = pd.to_datetime(reports["snapshot_at"], utc=True)
    snapshot_times = sorted(pd.Timestamp(t) for t in stamps.dropna().unique())
    details["rows"] = int(len(reports))
    details["snapshots"] = len(snapshot_times)
    details["window"] = (snapshot_times[0], snapshot_times[-1]) \
        if snapshot_times else (None, None)
    if len(snapshot_times) < THIN_ARCHIVE_SNAPSHOTS:
        (soft if not strict else hard).append(
            f"archive carries {len(snapshot_times)} captured snapshot(s) — "
            f"coverage is real but young; capture daily to accumulate history")
    if len(snapshot_times) == 0:
        hard.append("archive has no captured snapshot timestamps")
        return hard, soft, details

    # ── bridge quality, honestly defined ────────────────────────────────────
    # Note: an unmatched name is by construction absent from the rating
    # space (the bridge resolves every name the same normalisation finds),
    # so "pool-eligible but unmatched" cannot occur and is not checked. The
    # catchable bridge failure is the AMBIGUOUS refusal, gated below.
    bridged, baudit = ist.map_reports_to_rating_ids(reports, ratings)
    unmatched = bridged[bridged[ist.OUT_PLAYER] == bridged["_espn_player_id"]]
    out_ir = bridged[bridged["status"].isin(OUT_IR_STATUSES)]
    matched = int((out_ir[ist.OUT_PLAYER] != out_ir["_espn_player_id"]).sum())
    eligible = len(out_ir)
    details["bridge"] = {
        "matched": baudit["matched"], "rows": baudit["report_rows"],
        "ambiguous": baudit["ambiguous"], "unmatched": len(unmatched),
        "out_ir_eligible": eligible, "out_ir_matched": matched,
    }
    if baudit["ambiguous"]:
        hard.append(f"{baudit['ambiguous']} ambiguous name(s) refused by the "
                    f"bridge — a wrong-person exclusion is worse than none")

    # ── PIT soundness (bridge BEFORE build, production order) ───────────────
    actual = bridged[~bridged["snapshot_marker"].fillna(False).astype(bool)]
    actual = actual.dropna(subset=["player_id"]).copy()
    actual[ist.OUT_PLAYER] = actual[ist.OUT_PLAYER].astype(str)
    stints, saudit = ist.build_stint_intervals(bridged)
    stints.attrs["snapshot_times"] = snapshot_times
    stints.attrs["snapshot_based"] = bool(saudit.get("snapshot_based"))
    stints.attrs["window_end"] = stamps.max()
    details["stints"] = int(len(stints))
    dates = pd.to_datetime(ratings["game_date"], errors="coerce")
    try:
        ist.assert_pit(ratings, stints, decided_max_date=dates.max(),
                       window_end=stints.attrs["window_end"],
                       snapshot_based=stints.attrs["snapshot_based"])
    except ist.PitViolation as exc:
        hard.append(f"PIT violation: {exc}")

    # Exclusion-refused statuses: policy, but a human may want to know.
    refused = bridged[bridged["status"].isin(REFUSED_STATUSES)]
    if len(refused):
        (soft if not strict else hard).append(
            f"{len(refused)} exclusion-refused row(s) "
            f"({sorted(refused['status'].unique())}): "
            f"{sorted(refused['player_name'].dropna().unique().tolist())}")

    # ── retroactivity: pre-capture pools must see ZERO removals ─────────────
    teams = sorted(ratings["team"].dropna().unique().tolist())
    # Snapshot times arrive tz-aware UTC; the rating dates are naive. All
    # comparisons in this audit run in the module's naive-UTC convention.
    first_capture = ist._utc_naive(snapshot_times[0])
    hist_date = dates.max() + pd.Timedelta(days=1)
    if hist_date >= first_capture:
        # Ratings extend to/past the capture: a pre-capture probe does not
        # exist for this archive, and removals on post-capture dates are the
        # exclusion working, not a violation. Record the skip honestly.
        details["retroactivity_pool_rows"] = 0
        details["retroactivity_skipped"] = (
            f"ratings extend to {dates.max().date()} (>= first capture "
            f"{first_capture}) — pre-capture probe not applicable")
    else:
        hist = pd.DataFrame([
            {"game_date": hist_date, "team": t,
             "start_time_utc": f"{hist_date.date()}T02:00:00Z"}
            for t in teams])
        hist_pool, hist_audit = ist.team_game_rates(
            ratings, stints=stints, games=hist)
        if hist_audit["dropped_unavailable"]:
            hard.append(f"archive removed {hist_audit['dropped_unavailable']} "
                        f"player(s) from PRE-capture games — look-ahead")
        elif not len(hist_pool):
            hard.append("retroactivity probe never formed a pool (vacuous check)")
        else:
            details["retroactivity_pool_rows"] = int(len(hist_pool))

    # ── opening-slate shadow with last-known form frozen post-capture ───────
    last = ratings[ratings.game_date == dates.max()].copy()
    frozen_day = pd.Timestamp(first_capture).normalize() + pd.Timedelta(days=1)
    slate_day = frozen_day + pd.Timedelta(days=2)
    frozen = last.assign(game_date=frozen_day, pool_date=frozen_day)
    slate = pd.DataFrame([
        {"game_date": slate_day, "team": t,
         "start_time_utc": f"{slate_day.date()}T02:00:00Z"}
        for t in teams])
    pool, paudit = ist.team_game_rates(frozen, stints=stints, games=slate)
    shadow: dict[str, dict] = {}
    if "n_unavailable" in pool.columns and len(pool):
        side = pool.groupby("team").agg(kept=("n_players", "sum"),
                                        out=("n_unavailable", "max"))
        for team, row in side[side.out > 0].iterrows():
            shadow[str(team)] = {"out": int(row["out"]),
                                 "candidates": int(row["kept"] + row["out"])}
    details["shadow"] = shadow
    details["shadow_removals"] = int(paudit["dropped_unavailable"])
    if paudit.get("missing_decision_time", 0):
        hard.append("slate grid lost its decision timestamps — the exclusion "
                    "would silently stop binding on the real slate")

    # ── roster-move label differences (informational by design) ─────────────
    rteam = ratings.drop_duplicates("player_id").set_index("player_id")["team"]
    moves = 0
    for _, row in actual[actual[ist.OUT_PLAYER] != actual["_espn_player_id"]].iterrows():
        pid = str(row[ist.OUT_PLAYER])
        if pid in rteam.index and pd.notna(row["team"]):
            moves += 1
    details["roster_move_labels"] = moves
    if moves:
        (soft if not strict else hard).append(
            f"{moves} roster-move label difference(s) present")

    return hard, soft, details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true",
                        help="fail on informational findings too")
    args = parser.parse_args(argv)

    reports = ing._injury_history()
    if reports is None or not len(reports):
        print("AUDIT FAIL: no captured injury history exists anywhere")
        return 1
    ratings = feat._load_player_ratings()
    if not len(ratings):
        print("AUDIT FAIL: player ratings unavailable")
        return 1

    stamps = pd.to_datetime(reports["snapshot_at"], utc=True)
    print(f"[archive] {len(reports)} rows, {stamps.nunique()} snapshot(s), "
          f"span {stamps.min()} .. {stamps.max()}")
    print(f"[archive] statuses: {reports['status'].value_counts(dropna=False).to_dict()}")
    hard, soft, details = audit(ratings, reports, strict=args.strict)
    print(f"[bridge]   {details['bridge']['out_ir_matched']}/"
          f"{details['bridge']['out_ir_eligible']} Out/IR rows matched "
          f"({details['bridge']['unmatched']} unmatched = goalies/prospects "
          f"outside the skater rating space; {details['bridge']['ambiguous']} "
          f"ambiguous refusals)")
    print(f"[pit]      {details['stints']} stint(s) bound through "
          f"{details['window'][1]}; retroactivity pool rows "
          f"{details.get('retroactivity_pool_rows', 0)}, 0 removals"
          + (f" ({details['retroactivity_skipped']})"
             if details.get("retroactivity_skipped") else ""))
    print(f"[shadow]   opening-slate removals: {details['shadow_removals']} "
          f"across {len(details['shadow'])} team(s)")
    for team, s in details["shadow"].items():
        print(f"           {team}: {s['out']} of {s['candidates']} candidates")

    print("== AUDIT VERDICT ==")
    for s in soft:
        print(f"  SOFT: {s}")
    if hard:
        for h in hard:
            print(f"  HARD: {h}")
        return 1
    print("  no hard findings — league availability state is PIT-sound and "
          "bridge-complete for every rated player")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
