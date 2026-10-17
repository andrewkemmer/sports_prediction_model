"""Standalone split-skew Gaussian mechanics, mirrored in each Kaggle backend.

The two-piece normal has a continuous density at its mode, distinct left
and right scales, and mean mode + sqrt(2/pi)*(right-left). We center that
mean on the score forecast before truncating totals at the zero lattice.
Sport parameters are fixed cold-start priors; fitted parameters use prior
OOF outcomes only. Calibration is a single increasing map for all lines.

LOW-PASS ARTIFACT / LATTICE CORRECTION (2026-10-14): any smooth continuous
distribution integrated over integer bins acts as a low-pass filter — it
systematically underestimates peaks and overestimates troughs at integer
score totals and integer margins, leaving a sawtooth/"comb" residual
r(k) = P_obs(k) - P_model(k) against observed OOF finals in every sport.
Neither the split-skew family nor the monotone calibration map can express
that alternation (both are smooth in the CDF). ``lattice_correct`` applies a
fitted discrete integer-lattice correction AFTER calibration and BEFORE
Monte Carlo sampling, on the two-sided combined distribution, for BOTH
targets — totals (home + away) and margin (home - away) — fit
independently per sport/target because the lattice structures differ (see
``LATTICE_CONFIG``). Corrections are causal (fit from strictly-prior OOF
finals only), zero-sum, clamped and renormalized, and a missing/all-zero
correction is a byte-identical passthrough.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.special import expit, logit, ndtr
from scipy.optimize import minimize

FAMILY = "two_sided_asymmetric_gaussian_split_skew"
METHOD = "causal_split_skew_crps_mc"
MC_DRAWS = 50_000
MC_SEED = 42
MC_SE_TARGET = .005
MIN_HISTORY = 200
# Reference total, initial side residual variance/mean, maximum support.
SPORT_PRIORS = {"nba": (220., 1.3, 600), "nfl": (44., 5., 250),
                "nhl": (6., 1., 100), "mlb": (9., 2., 160)}


def defaults(sport):
    _, variance, _ = SPORT_PRIORS[sport]
    return {"home_variance": variance, "away_variance": variance,
            "correlation": 0., "total_skew": 0., "margin_skew": 0.,
            # NHL odd/even parity (see parity_correct / lattice_correct): a
            # game tied after regulation has an EVEN regulation total and is
            # ALWAYS settled in OT/SO with exactly one extra goal, landing on
            # an ODD final. Odd finals run ~65% of NHL games, so alpha =
            # P(OT/SO | even regulation total) = 2*odd_share - 1 ~ 0.30
            # cold-start prior; fit_variance refits it from prior OOF finals.
            # Other sports keep 0.0 (MLB extras and NBA OT do not preserve
            # the parity flip).
            "total_parity_alpha": .30 if sport == "nhl" else 0.}


def design(h, a, sport):
    reference = SPORT_PRIORS[sport][0]
    return np.column_stack([np.ones(len(h)),
                            np.clip((h + a) / reference - 1., -1., 2.),
                            np.clip(np.abs(h - a) / reference, 0., 1.)])


def support_for(kind, sport):
    if kind not in ("total", "margin"):
        raise ValueError("distribution kind must be total or margin")
    high = SPORT_PRIORS[sport][2]
    return np.arange(0, high + 1) if kind == "total" else np.arange(-high, high + 1)


def fit_variance(h, a, outcomes, sport):
    parameters = defaults(sport)
    if len(h) < MIN_HISTORY:
        return parameters
    home = .5 * (outcomes["total"] + outcomes["margin"])
    away = .5 * (outcomes["total"] - outcomes["margin"])
    eh, ea = home - h, away - a
    # Center covariance: common pace is not confused with side bias.
    ch, ca = eh - eh.mean(), ea - ea.mean()
    vh = np.clip(np.sum(ch ** 2) / np.sum(h), .05, 30.)
    va = np.clip(np.sum(ca ** 2) / np.sum(a), .05, 30.)
    denom = np.sqrt(np.sum(ch ** 2) * np.sum(ca ** 2))
    rho = np.clip(np.sum(ch * ca) / denom, -.9, .9) if denom > 1e-12 else 0.
    parameters.update(home_variance=float(vh), away_variance=float(va),
                      correlation=float(rho))
    for kind, residual in (("total", eh + ea), ("margin", eh - ea)):
        residual = residual - residual.mean()
        left, right = residual[residual < 0], residual[residual >= 0]
        if len(left) >= 20 and len(right) >= 20:
            # Half log scale ratio: left=scale*exp(-skew), right=scale*exp(skew).
            l = np.sqrt(np.mean(left ** 2))
            r = np.sqrt(np.mean(right ** 2))
            parameters[f"{kind}_skew"] = float(np.clip(.5 * np.log(max(r, 1e-9) / max(l, 1e-9)), -.7, .7))
    if sport == "nhl" and outcomes.get("total") is not None:
        # Causal odd/even parity refit from PRIOR OOF finals only (the caller
        # passes strictly-prior outcomes). Odd share = 0.5 + P(regulation
        # tie), and every tied (even) regulation total flips to an ODD final,
        # so alpha = 2*odd_share - 1 recovers P(OT/SO | even regulation
        # total). Clip to [0, .6]: a parity-flat history disables the
        # correction instead of inverting it.
        total = np.round(np.asarray(outcomes["total"], float))
        if len(total):
            parameters["total_parity_alpha"] = float(
                np.clip(2. * ((total % 2.) == 1.).mean() - 1., 0., .6))
    return parameters


def parity_correct(cdf, support, alpha):
    """NHL odd/even parity projection of a calibrated TOTAL CDF.

    The smooth split-skew family plus its monotone calibration map cannot
    represent the alternation that NHL overtime rules impose on final totals:
    tied (even) regulation totals resolve in OT/SO to odd finals, so odd
    totals carry extra mass and even totals less. The CRPS-fit smooth
    projection absorbs exactly the even/odd DC half of that alternation
    (measured 2026-10-14 on the decided OOF store: per-line CDF gaps alternate
    sign with implied alpha 0.25..0.45 against the structural 0.36), leaving a
    residual +/- (alpha/2)*pmf_j at odd/even line j. This applies that residual
    AFTER calibration: odd lines rise by (alpha/2)*pmf, even lines fall by the
    same amount, where alpha = P(OT/SO | even regulation total).

    Totals only (margin lattices keep the no-tie rule and integer spreads do
    not flip parity). alpha <= 0 is a no-op, which leaves non-NHL sports and
    pre-parity stored bundles byte-unchanged. The alternating shift can
    non-monotone-sliver only where a pmf collapses against its neighbour; a
    running maximum flattens those to zero push mass so the result is always
    a valid CDF in [0, 1].

    Since 2026-10-14 this is the STRUCTURAL half of the NHL totals lattice
    correction: ``lattice_correct(..., target="totals")`` re-applies exactly
    this math (same alpha, same CDF-domain shift) at the totals choke point
    and adds the generic fitted residual on top. The function is kept for
    direct callers and must stay byte-stable.
    """
    alpha = float(alpha)
    if alpha <= 0. or len(support) < 3 or support[0] < 0:
        return cdf
    alpha = min(alpha, .6)
    mass = np.diff(np.column_stack([np.zeros(len(cdf)), cdf]), axis=1)
    sign = np.where(np.asarray(support) % 2 == 1, 1., -1.)
    corrected = np.clip(cdf + .5 * alpha * mass * sign[None, :], 0., 1.)
    corrected = np.maximum.accumulate(corrected, axis=1)
    # A saturated CDF point stays saturated: exact-1 mass cannot be lowered
    # (its pmf is absorbed by the support boundary), and the Monte Carlo
    # inverse transform must keep u <= cdf[-1] for every draw.
    corrected[cdf >= 1.] = 1.
    return corrected


# ---------------------------------------------------------------------------
# Integer-lattice (sawtooth / "comb") correction — sport-aware, BOTH targets
# ---------------------------------------------------------------------------
# WHY (2026-10-14): a smooth continuous density integrated over integer bins
# is a LOW-PASS filter. It underestimates peaks and overestimates troughs of
# the discrete score lattice, leaving a sawtooth residual
# r(k) = P_obs(k) - P_model(k) that alternates sign across integer totals and
# integer margins in every sport. lattice_correct() applies a fitted discrete
# correction on the calibrated two-sided CDF, per target, before sampling.
#
# Totals and margin are fit INDEPENDENTLY (separate residuals, separate
# coefficients, keyed by sport+target) because their lattice structures
# differ: the margin lattice carries the no-tie rule (P(M=0)=0 for
# NHL/MLB/NBA, redistributed to +/-1; NFL keeps valid tie mass) and the NHL
# totals lattice carries the OT/SO parity flip, while the generic residual
# captures whatever lattice structure remains (e.g. MLB's empirical
# walk-off-by-one tilt at M=+1 — deliberately NOT hard-coded).
LATTICE_TARGETS = ("totals", "margin")
# Localized per-bin shrinkage basis (2026-10-16; retires the global Fourier
# basis — see lattice_local_deltas). The prior signal scale is a fraction
# of the LOCAL model mass, so peak bins keep signal while empty tail bins
# shrink to zero; per-bin binomial SEs supply the noise floor.
LATTICE_PRIOR_FRAC = .1         # prior signal scale = frac * local model mass
LATTICE_DELTA_CAP = .03         # |delta(k)| clamp: a lattice fix, not a refit
# Structural spike headroom (2026-10-16, NFL key-number alignment): the
# preregistered key-number spikes are individually held-out gated (one-sided
# structural bar per bin), and their rule-derived residuals exceed the
# generic fix bound — the NFL margin +/-3 baselines sit +0.049/+0.044 under
# the observed frequencies (20261016 evidence, n=2574), where the shipped
# composed vector pinned at the .03 cap and still shipped post-lattice z
# 5.3/6.2 leftovers (run_diagnostics/nfl_keynumber_align_probe.log). The
# STRUCTURAL spike component therefore carries its own clamp; every generic
# component keeps LATTICE_DELTA_CAP, and the composed vector keeps it
# everywhere except at admitted structural spike bins (per-bin caps at the
# NFL compose site). No cap relief exists without a preregistered, individually
# gated spike at that bin.
LATTICE_SPIKE_CAP = .05
LATTICE_DELTA_EPS = 1e-9        # coefficients below this serialize as exact 0
LATTICE_GATE_Z = 3.0            # generic bar: held-out, two-sided, + MAE + collateral
LATTICE_STRUCTURAL_Z = 2.5      # preregistered one-sided bar (rule-derived
                                # mechanisms only — never a generic fallback)
LATTICE_TEST_FRACTION = .3      # time-ordered held-out tail of the evidence
LATTICE_MIN_TEST = 100          # held-out rows below this: never admit delta
LATTICE_ROUND = 6               # decimal rounding stored deltas reprice with
LATTICE_SHARE_BLOCKS = 4        # time blocks for the hierarchical share pool
LATTICE_SHARE_TAU = .05         # between-block SD prior for share pooling
LATTICE_SPIKE_SECONDARY_SHRINK = 4.  # extra shrinkage on secondary key numbers
# Key-number spike relief knob (2026-10-16) — shipped at 1.0 (no relief) BY
# MEASUREMENT: the mass-proportional prior (tau = PRIOR_FRAC * local mass)
# shrinks the NFL +/-3 spikes to ~36% of their train residuals (amplitude
# 0.017/0.014 vs 0.049/0.044), but the causal-walk ladder
# (run_diagnostics/nfl_keynumber_relief_ladder.json, sections (a)+(c))
# measured that DEEPER spike proposals fail their own held-out gate: at
# boost 2 the -3 mirror spike drops to z 2.476 < 2.5 and REJECTS (post-lattice
# -3 leftover z 5.9 vs 3.9 at boost 1), and boost 3+ additionally loses the
# +7/14 admissions — the squared-loss gate penalizes the deep proposal's
# amp^2 term faster than its linear hit gain. The spike amplitude stays at
# the shrink-limited level; the +/-3 alignment ships through the REGISTERED
# generic relief on spike-adjusted pmfs (LATTICE_RELIEF, (nfl, margin))
# where the second layer's own gate has headroom. The knob is kept so a
# future evidence regime can re-ladder it; the per-spike z in the
# spike_report is computed at whatever boost is configured, so an
# over-deep proposal fails its own gate instead of shipping.
LATTICE_SPIKE_RELIEF_BOOST = 1.
# Registered high-mass shrinkage relief (2026-10-16): the mass-proportional
# prior (tau = PRIOR_FRAC * w) over-shrinks high-mass bins (measured: NBA
# margin keeps ~1/3 of the +/-1 residual). Registered (sport, target) pairs
# boost tau by ``boost`` where local model mass sits at/above ``quantile``
# of the fitted support; every other target keeps exact current behavior.
# Gates, collateral, cap and zero-sum are untouched — relief only changes
# the proposed amplitude. Gated in production (see nba_shrinkage_experiment).
# The quantile is taken over MASS-BEARING bins only (2026-10-16 re-review):
# union lattices pad the support with hundreds of exact-zero bins, so a
# plain quantile collapsed to 0 and "targeted" relief silently applied to
# every bin (boost 3 was byte-for-byte global frac 0.3 on the 20261016
# evidence — see run_diagnostics/nba_pm1_totals_relief_probe2.json
# quantile_diag).
LATTICE_RELIEF_BOOST = 3.      # tau multiplier on mass bins >= the quantile
                                # (2026-10-16 walk ladder: boost 5 measured a
                                # REGRESSION at the dashboard band — per-origin
                                # admissions 49 -> 38 as the boosted fit's train-only
                                # +3 overshoot trips the collateral guard at origins
                                # 37..42, shipped +/-1 z -2.32/-2.46 vs -2.23/-2.09
                                # at boost 3; boost 8 no better). Boost 3 is the
                                # deepest amplitude whose admissions hold; with the
                                # no_tie mass transfer owning the +/-1 artifact the
                                # residual layer no longer needs depth. Evidence:
                                # run_diagnostics/nba_walk_boost_eval*.log)
LATTICE_RELIEF_QUANTILE = .75  # mass-bearing bins at/above this quantile get relief
LATTICE_RELIEF = {("nba", "margin"): {"boost": LATTICE_RELIEF_BOOST,
                                      "quantile": LATTICE_RELIEF_QUANTILE},
                  # NFL margin (2026-10-16, key-number alignment): the
                  # registered relief is the layer that actually closes the
                  # +/-3 underfit — the spike layer alone cannot (its deeper
                  # proposals fail their own held-out gate, see
                  # LATTICE_SPIKE_RELIEF_BOOST), so the SECOND layer — the
                  # generic residual on spike-adjusted pmfs — takes the
                  # leftover with the same mass-targeted relief the NBA
                  # +/-1 remediation used. Causal-walk ladder
                  # (run_diagnostics/nfl_keynumber_relief_ladder.json,
                  # section margin_relief_ladder): boost 3 keeps the generic
                  # gate at z 5.86 (bar 3.0), collateral clean (max excess
                  # 0.0), pooled MAE 0.00072 -> 0.00027, and brings BOTH
                  # key numbers inside the 95% band on the held-out tail
                  # (-3: z 3.91 -> 1.77; +3: 2.39 -> 0.07) while the M=0
                  # tie overprice (-2.82) also lands in band (-1.06);
                  # boost 5 trades a deeper +14 overshoot (z -3.74) and a
                  # cap-binding dmax for a marginal MAE gain, boost 8 is a
                  # measured COLLATERAL VETO (max excess +0.00012). Totals
                  # relief was A/B'd on the same walk and REJECTED
                  # (section totals_ladder): deepening the totals teeth
                  # fixes 51 but pushes bin 40 from z -2.17 to -3.5 and
                  # adds a new tooth at 66 — the band gets worse, so
                  # ("nfl", "totals") is deliberately unregistered.
                  ("nfl", "margin"): {"boost": 3.,
                                     "quantile": LATTICE_RELIEF_QUANTILE}}
# Totals relief was tested and REJECTED on the same evidence (2026-10-16):
# every deeper totals variant — global frac 0.2/0.3/0.5 and mass-targeted
# boost 2/3/5 — fails the held-out bar (gate z 2.37..2.76 < 3.0), so
# registering ("nba", "totals") would only REVOKE the admitted shallow fit
# (gate 3.077). Remaining totals teeth sit within band multiplicity (3 of
# 101 plotted bins outside 95% vs ~5 expected). The >=270 tail deficit
# (honest authority P(total>=270): obs 0.0242 vs modeled 0.0162, z +3.74)
# was A/B-tested on the variance/skew layer and REJECTED there too
# (2026-10-16, doc A7): skew boost +0.03..+0.25 moves the tail
# monotonically with zero collateral but the CRPS map absorbs it - even
# ~10x the fitted skew lands z +2.01, still outside band; symmetric scale
# x1.10 has margin collateral and opens a new tooth. It is a family-shape
# limit (Gaussian right tail decays too fast), fixable only by a
# heavier-tailed component under separate preregistration. The
# mass-proportional prior moves no tail mass by design. No totals entry
# here is deliberate.

# Pre-registered structural mechanisms (2026-10-16). A structural parameter
# is rule-derived with a DIRECTIONAL prior declared here BEFORE any fitting:
# the code enforces the direction (one-sided test) and the structural bar
# (LATTICE_STRUCTURAL_Z); the generic bar (LATTICE_GATE_Z, two-sided) still
# applies first wherever both paths exist. This table is the disclosure
# record — a mechanism not listed here cannot take the structural bar.
#   mlb margin walkoff_split: extras walk-off tilt — the home team wins a
#     larger share of +/-1 finals than the equal split (Manfred runner +
#     home bats last), shifting tie-mass-scaled (q-1/2)*m0 to M=+1.
#   nfl margin spikes: scoring rules pile margin mass at 3 (field goal) and
#     7 (touchdown + XP); secondary mass at 1, 4, 6, 10, 14, 17 — each with
#     its away-side mirror (-3/-7 primary; -1/-4/-6/-10/-14/-17 secondary),
#     individually gated: scoring symmetry is rule-derived, so mirrors take
#     the same one-sided structural bar.
#   nhl margin empty_net_2_to_3: trailing teams pull the goalie, so 2-goal
#     leads finish as 3-goal finals more often than a smooth density allows
#     (excess mass at |M|=3 on both signs).
LATTICE_PREREGISTERED = {
    ("mlb", "margin", "walkoff_split"): {
        "mechanism": ("extras walk-off tilt: the home team wins a larger share "
                        "of +/-1 finals than the equal split"),
        "direction": "q > 1/2", "test": "one-sided",
        "generic_bar": LATTICE_GATE_Z, "structural_bar": LATTICE_STRUCTURAL_Z},
    ("nfl", "margin", "spike_3"): {
        "mechanism": "field-goal key number: excess margin mass at exactly 3",
        "direction": "amplitude > 0", "test": "one-sided",
        "structural_bar": LATTICE_STRUCTURAL_Z},
    ("nfl", "margin", "spike_7"): {
        "mechanism": "touchdown+XP key number: excess margin mass at exactly 7",
        "direction": "amplitude > 0", "test": "one-sided",
        "structural_bar": LATTICE_STRUCTURAL_Z},
    ("nfl", "margin", "spike_secondary"): {
        "mechanism": ("secondary scoring key numbers (1, 4, 6, 10, 14, 17) "
                        "with stronger shrinkage, individually gated"),
        "direction": "amplitude > 0", "test": "one-sided",
        "structural_bar": LATTICE_STRUCTURAL_Z},
    ("nfl", "margin", "spike_m3"): {
        "mechanism": ("field-goal key number, away side: excess margin mass "
                        "at exactly -3 (scoring symmetry with spike_3)"),
        "direction": "amplitude > 0", "test": "one-sided",
        "structural_bar": LATTICE_STRUCTURAL_Z},
    ("nfl", "margin", "spike_m7"): {
        "mechanism": ("touchdown+XP key number, away side: excess margin "
                        "mass at exactly -7 (scoring symmetry with spike_7)"),
        "direction": "amplitude > 0", "test": "one-sided",
        "structural_bar": LATTICE_STRUCTURAL_Z},
    ("nfl", "margin", "spike_secondary_mirror"): {
        "mechanism": ("secondary scoring key numbers, away side "
                        "(-1, -4, -6, -10, -14, -17) with stronger shrinkage, "
                        "individually gated"),
        "direction": "amplitude > 0", "test": "one-sided",
        "structural_bar": LATTICE_STRUCTURAL_Z},
    ("nhl", "margin", "empty_net_2_to_3"): {
        "mechanism": ("empty-net goal: 2-goal leads finish as 3-goal finals; "
                        "excess margin mass at |M|=3"),
        "direction": "train residual at +3 and -3 both positive",
        "test": "one-sided",
        "generic_bar": LATTICE_GATE_Z, "structural_bar": LATTICE_STRUCTURAL_Z},
}


def lattice_preregistered(sport, target):
    """Pre-registered structural mechanisms for a sport/target pair ({} when
    none) — the disclosure record the structural bar enforces."""
    return {name: spec for (s, t, name), spec in LATTICE_PREREGISTERED.items()
            if s == sport and t == target}

# Per-sport, per-target lattice configuration. "structural" names the
# deterministic lattice rule the target carries BEFORE the generic residual:
#   nhl totals     -> "ot_parity" (parity_correct alpha from variance params)
#   nhl margin     -> "ot_no_tie" (P(M=0)=0; mass split +/-1 by causal OT
#                                  home-win estimate, see fit_ot_home_share)
#   mlb margin       -> "no_tie"    (P(M=0)=0; the +/-1 split weight is the
#                                  gated one-goal home-share estimate shared
#                                  with NHL — MLB's walk-off-by-one tilt is
#                                  fitted, NOT hard-coded; whatever remains
#                                  is left to the generic residual, see the
#                                  M=+1 diagnostic in the fit record)
#   nba margin       -> "no_tie"    (P(M=0)=0; the smooth center-bin mass
#                                  is a continuous-representation ARTIFACT —
#                                  the NBA has no tie process — so it is
#                                  redistributed PROPORTIONAL TO MODEL MASS
#                                  over legal bins (no_tie's nba branch),
#                                  NOT split onto +/-1. Direction declared
#                                  2026-10-16 before fitting: +/-1 residual
#                                  decreases; measured shipped band state
#                                  in run_diagnostics/nba_pm1_totals_
#                                  relief_probe2.json. Nearest-neighbor
#                                  splitting was measured inflating both
#                                  one-point bins by +0.0123 (center mass
#                                  0.0246/2) beyond their full smooth mass,
#                                  a systematic z -2.2/-2.1 over-confidence
#                                  the residual lattice could not remove
#                                  (amplitude and admissions both capped).)
#   nfl margin     -> None        (ties are possible; P(M=0) stays valid)
# "forbidden_bins" pins the lattice bins the structural rule owns: the fitted
# residual may never put mass back there.
LATTICE_CONFIG = {
    "nhl": {"totals": {"structural": "ot_parity", "forbidden_bins": ()},
            "margin": {"structural": "ot_no_tie", "forbidden_bins": (0,)}},
    "mlb": {"totals": {"structural": None, "forbidden_bins": ()},
            "margin": {"structural": "no_tie", "forbidden_bins": (0,)}},
    "nba": {"totals": {"structural": None, "forbidden_bins": ()},
            "margin": {"structural": "no_tie", "forbidden_bins": (0,)}},
    "nfl": {"totals": {"structural": None, "forbidden_bins": ()},
            "margin": {"structural": None, "forbidden_bins": ()}},
}


def lattice_config(sport, target):
    """The lattice config for a sport/target pair, or None when the pair is
    not configured — an unconfigured pair is a byte-identical passthrough."""
    if target not in LATTICE_TARGETS:
        raise ValueError(f"lattice target must be one of {LATTICE_TARGETS}")
    return LATTICE_CONFIG.get(sport, {}).get(target)


def _lattice_delta(correction):
    """The correction's residual vector as a 1-D float array, or None when
    absent, malformed, non-finite or all-zero (all of which mean no-op)."""
    if not correction:
        return None
    delta = correction.get("delta")
    if delta is None:
        return None
    delta = np.asarray(delta, float)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        return None
    return delta if np.any(delta != 0.) else None


def _realign_lattice_delta(delta, fit_bounds, support):
    """Map a fitted per-bin delta onto the pricing support by integer bin value.

    Dynamic Gaussian supports expand per-block with fitted skew/scale, so a
    delta fitted on the evidence-union lattice can be one bin narrower than
    the block being priced (measured 2026-10: NFL totals union 0..250 vs a
    later block's 0..251). Bins the fit saw keep their fitted value; bins
    outside the fitted lattice get exact 0 (the shrinkage-consistent choice
    — the fit's own prior scale is mass-proportional, so unseen tail bins
    would shrink to ~0 anyway). Returns None when the fitted lattice is
    unknown (no ``support`` bounds stored — pre-realignment bundles carry
    no delta, so this never fires for them) or not a contiguous unit
    integer lattice, leaving the caller to raise as before rather than
    guess. An all-zero mapping is likewise None (passthrough, not a fit).
    """
    if fit_bounds is None or len(fit_bounds) != 2:
        return None
    lo, hi = int(fit_bounds[0]), int(fit_bounds[1])
    fit = np.asarray(delta, float)
    support = np.asarray(support, float)
    if hi - lo + 1 != len(fit):
        return None
    if (not np.all(np.isfinite(support)) or not len(support)
            or not np.all(support == np.round(support))):
        return None
    out = np.zeros(len(support))
    idx = np.round(support).astype(int) - lo
    ok = (idx >= 0) & (idx < len(fit))
    out[ok] = fit[idx[ok]]
    return out if np.any(out != 0.) else None


def lattice_correct(cdf, support, correction, sport, target):
    """Apply the integer-lattice correction to a calibrated CDF.

    ``correction`` is the fitted per-target spec (see
    ``fit_lattice_correction``): optional ``alpha`` (NHL totals OT/SO
    parity — the structural term, applied with ``parity_correct``'s exact
    CDF-domain math) and optional ``delta`` (the generic residual vector
    over ``support``). Order: structural parity, then the residual in the
    PMF domain. Guards, in order: clamp negative probabilities to zero,
    re-zero the structural forbidden bins (e.g. margin 0 for no-tie
    sports), renormalize every row to exactly 1, rebuild a monotone CDF by
    cumulative sum, and re-pin the exact-1 tail so the Monte Carlo inverse
    transform keeps u <= cdf[-1] for every draw.

    A missing, all-zero or unconfigured correction returns the INPUT OBJECT
    byte-identical — stored bundles that predate lattice fitting, disabled
    targets and gate-fitted zeros price exactly as before.
    """
    config = lattice_config(sport, target)
    if config is None:
        return cdf
    alpha = float((correction or {}).get("alpha", 0.) or 0.) if target == "totals" and sport == "nhl" else 0.
    delta = _lattice_delta(correction)
    if delta is not None and len(delta) != len(support):
        # Support-width drift (see _realign_lattice_delta): realign by bin
        # value instead of refusing to price. Lengths that already match
        # skip this branch, so all previously pricing bundles are untouched.
        # An admitted delta that cannot be placed (unknown fitted lattice)
        # still raises — silently dropping a fitted correction would be
        # worse than refusing to price it.
        delta = _realign_lattice_delta(delta, (correction or {}).get("support"),
                                       support)
        if delta is None:
            raise ValueError("lattice delta must align with the distribution support")
    if alpha <= 0. and delta is None:
        return cdf
    if delta is not None and len(delta) != len(support):
        raise ValueError("lattice delta must align with the distribution support")
    values = np.asarray(cdf, float)
    if alpha > 0.:
        # NHL OT/SO parity, structural half of the totals lattice correction.
        values = parity_correct(values, support, alpha)
    if delta is None:
        # Structural-only corrections keep parity_correct's exact output.
        return values
    mass = np.diff(np.column_stack([np.zeros(len(values)), values]), axis=1)
    mass = np.clip(mass + delta[None, :], 0., None)
    for banned in config.get("forbidden_bins", ()):
        j = int(np.searchsorted(support, banned))
        if 0 <= j < mass.shape[1]:
            mass[:, j] = 0.
    total = mass.sum(axis=1)
    collapsed = ~np.isfinite(total) | (total <= 0.)
    if np.any(collapsed):
        # A row whose corrected mass collapsed cannot be renormalized; keep
        # its input pmf rather than fabricate a distribution.
        mass[collapsed] = np.diff(
            np.column_stack([np.zeros(len(values)), values]), axis=1)[collapsed]
        total = mass.sum(axis=1)
    mass = mass / total[:, None]
    out = np.cumsum(mass, axis=1)
    out[:, -1] = 1.                       # exact-tail guard
    # Cumulative sums of nonnegative mass are monotone by construction; the
    # clip is the defensive CDF-range guard the CRPS/sampler contract needs.
    return np.clip(out, 0., 1.)


def lattice_local_deltas(weight, resid, se, shrink_mult=1., relief_boost=1.,
                         relief_quantile=None, prior_mult=1.):
    """Localized per-bin SE-aware shrunk residual (2026-10-16; the retired
    global Fourier basis's replacement — one generic basis, one code path,
    parameterized only by the sport/target's own weight and noise floor).

    WHY LOCALIZED: the admitted mechanisms are single-bin phenomena (MLB
    walk-off tilt at M=+1, NFL key-number spikes at 3/7, NHL odd/even
    teeth). A global smooth basis cannot move mass at one bin without
    collateral damage elsewhere — the measured failure mode (NFL margin
    z=2.27 with WORSE pooled MAE). Per-bin fitting with binomial-SE
    shrinkage admits only bins whose residual exceeds their own noise
    floor; the per-bin collateral guard then has teeth.

    delta_k = r_k * tau_k^2 / (tau_k^2 + SE_k^2) with prior scale
    tau_k = LATTICE_PRIOR_FRAC * w_k (mass-proportional: the low-pass
    artifact moves mass in proportion to local model mass — parity_correct's
    alpha*pmf is exactly this with one parity tone). Peak bins keep ~all
    signal; empty tail bins shrink to exact 0. ``shrink_mult`` > 1
    strengthens shrinkage (secondary key numbers). Bins with no model
    mass stay exact 0. Registered (sport, target) relief (``relief_boost``
    > 1 with ``relief_quantile`` set) multiplies tau where local model
    mass sits at/above that quantile OF THE MASS-BEARING BINS — relief
    exactly where the prior over-shrinks; ``None`` keeps current behavior.
    ``prior_mult`` scales the prior signal scale itself for callers whose
    bins carry a RULE-DERIVED structural excess (the preregistered
    key-number spikes — see LATTICE_SPIKE_RELIEF_BOOST); gates, SE floors
    and the generic path are untouched.
    """
    weight = np.clip(np.asarray(weight, float), 0., None)
    resid = np.asarray(resid, float)
    se = np.asarray(se, float)
    tau = (float(prior_mult) * LATTICE_PRIOR_FRAC * weight
           / max(float(shrink_mult), 1e-12))
    if relief_quantile is not None and len(weight):
        # Quantile over MASS-BEARING bins: union lattices pad the support
        # with exact-zero bins, and a plain quantile over them collapses to
        # 0 — the "targeted" boost silently becomes a global one. Bins with
        # no model mass keep tau = 0 under either rule.
        positive = weight[weight > 0.]
        q = float(np.quantile(positive if len(positive) else weight,
                              relief_quantile))
        tau = np.where(weight >= q, tau * float(relief_boost), tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        shrink = np.where(tau > 0., tau * tau / (tau * tau + se * se), 0.)
    return resid * shrink


def _zero_sum_mass_proportional(delta, weight):
    """Zero-sum a delta by removing its total proportionally to model mass.

    Uniform mean-subtraction smears a flat offset onto near-zero-mass tail
    bins, where ANY offset exceeds their tiny 1-sigma floor — self-inflicted
    collateral (measured: a z=17 comb vetoed by 1.6e-3 of mean-subtraction
    at bins 0/40). Proportional removal puts the sink where the mass is,
    the same principle as the spike sink and parity_correct's alpha*pmf.
    """
    delta = np.asarray(delta, float).copy()
    w = np.clip(np.asarray(weight, float), 0., None)
    if w.sum() > 0.:
        delta -= delta.sum() * (w / w.sum())
    else:
        delta -= delta.mean()
    return delta


def fit_ot_home_share(outcomes, pmfs=None, support=None, min_decisions=40):
    """P(home wins | one-goal final) from PRIOR OOF finals — the no-tie
    margin lattice's +/-1 split weight — admitted only on held-out evidence.

    Decided finals never tie (NHL/MLB/NBA), so every extra-time game lands
    on margin +/-1 and one-goal finals are the closest observable proxy
    for the extra-time decision (NHL OT/SO, MLB walk-off extras;
    regulation one-goal finals mix in, which is why the estimate is clipped
    to [.35, .60]). The estimate is then VERIFIED: the +/-1 mass shift it
    implies is scored on the time-ordered held-out tail of the same prior
    evidence with the same paired squared-probability loss and gate the
    generic delta uses (LATTICE_GATE_Z). Home advantage is already carried
    by the score means, so a raw one-goal home share can double-count it —
    when the held-out tail does not confirm the shift (or no evidence is
    supplied), this returns None and the margin lattice keeps the exact
    equal-split no-op, so a cold-start or unconvinced walk prices
    byte-identically to before.
    """
    if outcomes is None or not len(outcomes):
        return None
    margin = np.round(np.asarray(outcomes, float))
    one = margin[np.abs(margin) == 1.]
    if len(one) < int(min_decisions):
        return None
    share = float(np.clip((one > 0.).mean(), .35, .60))
    pmfs = np.empty((0, 0)) if pmfs is None else np.asarray(pmfs, float)
    if support is None or pmfs.ndim != 2 or pmfs.shape[0] != len(margin)             or pmfs.shape[1] != len(np.asarray(support)):
        return None
    support = np.asarray(support)
    j = int(np.searchsorted(support, 1))
    k = int(np.searchsorted(support, -1))
    if not (0 <= j < len(support) and 0 <= k < len(support)):
        return None
    s = share - .5
    if s == 0.:
        return None
    # TIE-MASS scaling (2026-10-15): moving the split weight from 1/2 to q
    # moves (q-1/2) of each row's PRE-split tie mass, not (q-1/2) of its
    # whole probability — scoring the unscaled shift lets the ||delta||^2
    # term veto every share (measured on MLB: z=-4.5 against a stable
    # +0.033/-0.024 +/-1 residual). The evidence pmfs are equal-split, so
    # pre(+1)+pre(-1) is unrecoverable exactly; under the family's own
    # smoothness assumption pre is near-flat across {-1,0,+1} and the
    # per-row tie mass is m0 = (pmf(+1)+pmf(-1))/2 (the pmf contrast
    # pmf(+1)-pmf(-1) is exact — the tie mass cancels). The gate stays
    # honest: it scores the scaled shift on held-out rows.
    tie = .5 * (pmfs[:, j] + pmfs[:, k])
    n_test = int(LATTICE_TEST_FRACTION * len(margin))
    if n_test < LATTICE_MIN_TEST:
        return None
    hold = np.arange(len(margin) - n_test, len(margin))
    inside = (margin >= support[0]) & (margin <= support[-1])
    hold = hold[inside[hold]]
    if len(hold) < LATTICE_MIN_TEST:
        return None
    horn = np.zeros(len(support))
    horn[j], horn[k] = 1., -1.
    amt = s * tie[hold]
    d = (-2. * amt * horn[np.searchsorted(support, margin[hold])]
         + 2. * amt * (pmfs[hold, j] - pmfs[hold, k])
         + 2. * amt * amt)
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
    z = float(-d.mean() / se) if se > 0. else 0.
    return share if (d.mean() < 0. and z >= LATTICE_GATE_Z) else None


def _share_gate(outcomes, pmfs, support, share, bar):
    """Held-out paired gate for a +/-1 split share (shared by the legacy and
    pooled estimators): the tie-mass-scaled shift scored with the same
    squared-probability loss. Returns (admitted, z, n_test)."""
    margin = np.round(np.asarray(outcomes, float))
    support = np.asarray(support)
    j = int(np.searchsorted(support, 1))
    k = int(np.searchsorted(support, -1))
    tie = .5 * (np.asarray(pmfs, float)[:, j] + np.asarray(pmfs, float)[:, k])
    n_test = int(LATTICE_TEST_FRACTION * len(margin))
    hold = np.arange(len(margin) - n_test, len(margin))
    inside = (margin >= support[0]) & (margin <= support[-1])
    hold = hold[inside[hold]]
    s = float(share) - .5
    horn = np.zeros(len(support))
    horn[j], horn[k] = 1., -1.
    amt = s * tie[hold]
    d = (-2. * amt * horn[np.searchsorted(support, margin[hold])]
         + 2. * amt * (np.asarray(pmfs, float)[hold, j]
                        - np.asarray(pmfs, float)[hold, k])
         + 2. * amt * amt)
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
    z = float(-d.mean() / se) if se > 0. else 0.
    return bool(d.mean() < 0. and z >= bar), round(z, 3), int(len(hold))


def fit_pooled_home_share(outcomes, pmfs=None, support=None, min_decisions=40,
                          bar=LATTICE_GATE_Z, structural=False):
    """Hierarchically pooled P(home wins | one-goal final) — the strengthened
    +/-1 split-weight estimator (2026-10-16).

    The legacy estimator's raw pooled share is efficient but its single
    number hides across-time instability; partial pooling across
    time-blocks shrinks erratic blocks toward the grand mean and yields a
    lower-variance estimate on the SAME evidence. Train (pre-holdout)
    evidence is split into LATTICE_SHARE_BLOCKS contiguous blocks; per-block
    shares combine by precision weights w_b = 1/(sigma_b^2 + TAU^2) with
    between-block prior SD LATTICE_SHARE_TAU (random-effects pooling toward
    the grand mean — never toward 1/2 directly, so a real tilt survives).
    Under two valid blocks the estimate falls back to the raw pooled train
    share (reported, not hidden). The pooled share is then gated on the
    held-out tail with the same tie-mass-scaled paired scoring: ``bar`` is
    the generic LATTICE_GATE_Z by default; ``structural=True`` applies the
    pre-registered one-sided structural bar (direction q > 1/2 enforced)
    for rule-derived mechanisms only (see LATTICE_PREREGISTERED).

    Returns (share | None, report) — the report carries the per-block
    shares, the pooled value, z and the bar applied, so the disclosure
    shows exactly what was admitted and how.
    """
    report: dict = {"estimator": "hierarchical_pooled", "structural": bool(structural),
                     "bar": float(bar)}
    if outcomes is None or not len(outcomes):
        report["reason"] = "no_evidence"
        return None, report
    margin = np.round(np.asarray(outcomes, float))
    one = margin[np.abs(margin) == 1.]
    if len(one) < int(min_decisions):
        report["reason"] = "insufficient_history"
        return None, report
    pmfs = np.empty((0, 0)) if pmfs is None else np.asarray(pmfs, float)
    if support is None or pmfs.ndim != 2 or pmfs.shape[0] != len(margin) \
            or pmfs.shape[1] != len(np.asarray(support)):
        report["reason"] = "misaligned_evidence"
        return None, report
    support = np.asarray(support)
    j = int(np.searchsorted(support, 1))
    k = int(np.searchsorted(support, -1))
    if not (0 <= j < len(support) and 0 <= k < len(support)):
        report["reason"] = "support_lacks_pm1"
        return None, report
    n_test = int(LATTICE_TEST_FRACTION * len(margin))
    if n_test < LATTICE_MIN_TEST:
        report["reason"] = "insufficient_held_out_evidence"
        return None, report
    n_train = len(margin) - n_test
    edges = np.linspace(0, n_train, LATTICE_SHARE_BLOCKS + 1).astype(int)
    block_q, block_n = [], []
    for b in range(LATTICE_SHARE_BLOCKS):
        idx = np.arange(edges[b], edges[b + 1])
        blk = margin[idx][np.abs(margin[idx]) == 1.]
        if len(blk) >= 10:
            block_q.append(float((blk > 0.).mean()))
            block_n.append(int(len(blk)))
    grand = float((margin[:n_train][np.abs(margin[:n_train]) == 1.] > 0.).mean())
    if len(block_q) >= 2:
        var = np.array([grand * (1. - grand) / n for n in block_n])
        w = 1. / (var + LATTICE_SHARE_TAU ** 2)
        pooled = float(np.sum(w * np.asarray(block_q)) / np.sum(w))
        report["blocks"] = [round(v, 4) for v in block_q]
        report["block_n"] = block_n
    else:
        pooled = grand
        report["blocks"] = [round(v, 4) for v in block_q]
        report["block_n"] = block_n
        report["fallback"] = "raw_pooled_train_share"
    share = float(np.clip(pooled, .35, .60))
    report["share"] = round(share, 4)
    if share == .5:
        report["reason"] = "share_is_exact_half"
        return None, report
    if structural and not share > .5:
        # Pre-registered direction enforced BEFORE the test: a structural
        # walk-off claim with q <= 1/2 is a no-op, never a two-sided retry.
        report["reason"] = "structural_direction_failed"
        return None, report
    admitted, z, n_hold = _share_gate(margin, pmfs, support, share, bar)
    report["z"] = z
    report["n_test"] = n_hold
    if n_hold < LATTICE_MIN_TEST:
        report["reason"] = "insufficient_held_out_evidence"
        return None, report
    if not admitted:
        report["reason"] = "held_out_not_significant"
        return None, report
    report["significant"] = True
    return share, report


def fit_margin_spikes(pmfs, outcomes, support, sport,
                      primary=(3, 7), secondary=(1, 4, 6, 10, 14, 17),
                      bar=LATTICE_STRUCTURAL_Z, min_history=MIN_HISTORY):
    """Pre-registered key-number spike parameters for a margin lattice.

    Rule-derived mass piles (NFL: 3 = field goal, 7 = touchdown + XP, plus
    the symmetric away-side mirrors the NFL caller passes) are
    single-bin excesses the generic residual would have to re-discover;
    fitting them as structural parameters with the mass-proportional
    zero-sum removal (+a at the spike bin, mass removed proportionally to
    model mass from non-candidate bins) keeps the correction honest and
    localized. Each candidate is gated INDIVIDUALLY, one-sided
    (amplitude > 0, z >= bar — the pre-registered structural bar; see
    LATTICE_PREREGISTERED), with secondary key numbers under stronger
    shrinkage (LATTICE_SPIKE_SECONDARY_SHRINK). Unadmitted candidates
    contribute exact 0 — never a partial spike.

    Returns (delta | None, report): the combined spike delta (zero-sum,
    clamped at the STRUCTURAL LATTICE_SPIKE_CAP, rounded) or None when no
    spike admits, plus the per-spike verdict table for disclosure.
    """
    report: dict = {"estimator": "key_number_spikes", "sport": sport,
                     "bar": float(bar), "spikes": []}
    support = np.asarray(support)
    pmfs = np.empty((0, len(support))) if pmfs is None else np.asarray(pmfs, float)
    outcomes = np.empty(0) if outcomes is None else np.asarray(outcomes, float)
    if pmfs.ndim != 2 or pmfs.shape[1] != len(support) or len(pmfs) != len(outcomes):
        raise ValueError("spike evidence requires aligned pmfs and outcomes")
    inside = ((outcomes >= support[0]) & (outcomes <= support[-1])
              & np.isfinite(outcomes))
    if inside.sum() < int(min_history):
        report["reason"] = "insufficient_history"
        return None, report
    n_test = int(LATTICE_TEST_FRACTION * len(outcomes))
    if n_test < LATTICE_MIN_TEST or len(outcomes) - n_test < int(min_history):
        report["reason"] = "insufficient_held_out_evidence"
        return None, report
    train = np.flatnonzero(inside[:len(outcomes) - n_test])
    hold = np.flatnonzero(inside[len(outcomes) - n_test:])
    if len(train) < int(min_history) or len(hold) < LATTICE_MIN_TEST:
        report["reason"] = "insufficient_held_out_evidence"
        return None, report
    candidates = [(int(v), False) for v in primary] + [(int(v), True) for v in secondary]
    candidates = [(v, sec) for v, sec in candidates if support[0] <= v <= support[-1]]
    report["relief_boost"] = float(LATTICE_SPIKE_RELIEF_BOOST)
    n_train = max(int(len(train)), 1)
    pbar = np.clip(pmfs[train].mean(axis=0), 0., 1.)
    se = np.sqrt(pbar * (1. - pbar) / n_train)
    obs = np.bincount(np.searchsorted(support, outcomes[train]),
                      minlength=len(support)).astype(float)
    obs /= max(int(len(train)), 1)
    resid = obs - pmfs[train].mean(axis=0)
    sink = np.array([i for i in range(len(support))
                     if int(support[i]) not in [v for v, _ in candidates]])
    sink_mass = float(pbar[sink].sum()) if len(sink) else 0.
    if sink_mass <= 0.:
        report["reason"] = "no_sink_mass"
        return None, report
    indices_hold = np.searchsorted(support, outcomes[hold])
    spikes = []
    for value, secondary_flag in candidates:
        i = int(np.searchsorted(support, value))
        mult = LATTICE_SPIKE_SECONDARY_SHRINK if secondary_flag else 1.
        # Registered key-number relief: the structural spike's prior scale
        # is its own evidence (LATTICE_SPIKE_RELIEF_BOOST), not a fraction
        # of local model mass — the gate below still decides, at the
        # boosted amplitude.
        amp = float(lattice_local_deltas(pbar[i:i + 1], resid[i:i + 1],
                                         se[i:i + 1], shrink_mult=mult,
                                         prior_mult=LATTICE_SPIKE_RELIEF_BOOST)[0])
        row = {"bin": value, "secondary": bool(secondary_flag),
               "amplitude": round(amp, 6)}
        if not amp > 0.:
            # Pre-registered direction: key numbers carry EXCESS mass. A
            # non-positive amplitude is a no-op, never a two-sided retry.
            row["admitted"] = False
            row["reason"] = "amplitude_not_positive"
            report["spikes"].append(row)
            continue
        trial = np.zeros(len(support))
        trial[i] = amp
        trial[sink] = -amp * pbar[sink] / sink_mass
        d = (-2. * trial[indices_hold] + 2. * (pmfs[hold] @ trial)
             + float(trial @ trial))
        tse = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
        z = float(-d.mean() / tse) if tse > 0. else 0.
        row["z"] = round(z, 3)
        if d.mean() < 0. and z >= bar:
            row["admitted"] = True
            spikes.append((i, amp))
        else:
            row["admitted"] = False
            row["reason"] = "held_out_not_significant"
        report["spikes"].append(row)
    if not spikes:
        report["reason"] = "no_spike_admitted"
        return None, report
    delta = np.zeros(len(support))
    for i, amp in spikes:
        delta[i] += amp
        delta[sink] -= amp * pbar[sink] / sink_mass
    delta = _zero_sum_mass_proportional(delta, pbar)
    # Structural clamp: the spike component alone may reach
    # LATTICE_SPIKE_CAP (each bin already held-out gated above); the
    # generic component keeps LATTICE_DELTA_CAP and the composed vector
    # is capped per bin at the compose site.
    delta = np.clip(delta, -LATTICE_SPIKE_CAP, LATTICE_SPIKE_CAP)
    delta[np.abs(delta) < LATTICE_DELTA_EPS] = 0.
    delta = np.round(delta, LATTICE_ROUND)
    report["significant"] = True
    report["cap"] = float(LATTICE_SPIKE_CAP)
    report["delta_l1"] = round(float(np.abs(delta).sum()), 6)
    return [float(v) for v in delta], report


def lattice_evidence(entries):
    """Align per-block BASE pmfs onto their union support for lattice fitting.

    ``entries`` are (pmf, outcomes, support) triples the causal walk
    accumulated from already-priced prior blocks (each block's raw support
    expands with its own means, so widths differ); returns the stacked
    {"pmfs", "outcomes", "support"} evidence dict, zero-padding the low and
    high tails onto the union lattice, or None when empty.
    """
    if not entries:
        return None
    low = min(int(e[2][0]) for e in entries)
    high = max(int(e[2][-1]) for e in entries)
    support = np.arange(low, high + 1)
    pmfs, outcomes = [], []
    for pmf, y, sup in entries:
        wide = np.zeros((len(pmf), len(support)))
        j0 = int(sup[0]) - low
        wide[:, j0:j0 + len(sup)] = pmf
        pmfs.append(wide)
        outcomes.append(np.asarray(y, float))
    return {"pmfs": np.vstack(pmfs), "outcomes": np.concatenate(outcomes),
            "support": support}


def _structural_lattice_fallback(sport, target, support, r_train, d, z, out):
    """One-sided structural-bar fallback for a preregistered generic residual.

    The generic bar (LATTICE_GATE_Z, two-sided) applies first; only when it
    rejects does a mechanism listed in LATTICE_PREREGISTERED get its
    directional prior checked on TRAIN evidence (never the held-out tail)
    and the one-sided structural bar (LATTICE_STRUCTURAL_Z). Returns the
    disclosure update (structural/bar/mechanism) or None. A mechanism name
    with no coded direction check never admits — the table alone is not
    enough, the code must enforce the direction.
    """
    specs = [(name, spec) for (s, t, name), spec in LATTICE_PREREGISTERED.items()
             if s == sport and t == target and "structural_bar" in spec
             and "generic_bar" in spec]
    if not specs:
        return None
    if not (float(np.asarray(d).mean()) < 0. and z >= LATTICE_STRUCTURAL_Z
            and out.get("residual_mae_after", float("inf"))
            < out.get("residual_mae_before", float("-inf"))
            and bool(out.get("collateral_ok"))):
        return None
    support = np.asarray(support)
    r_train = np.asarray(r_train, float)
    for name, spec in specs:
        if (sport, target, name) == ("nhl", "margin", "empty_net_2_to_3"):
            j3 = int(np.searchsorted(support, 3))
            k3 = int(np.searchsorted(support, -3))
            if not (0 <= j3 < len(support) and 0 <= k3 < len(support)
                    and int(support[j3]) == 3 and int(support[k3]) == -3):
                continue
            if not (r_train[j3] > 0. and r_train[k3] > 0.):
                continue
            return {"structural": True, "bar": float(spec["structural_bar"]),
                    "mechanism": name}
    return None


def fit_lattice_correction(pmfs, outcomes, support, sport, target,
                           min_history=MIN_HISTORY):
    """Fit delta(k) = P_obs(k) - P_model(k) for one target from prior OOF.

    ``pmfs`` are the BASE model (post-calibration, post-structural) PMFs of
    strictly-prior OOF rows and ``outcomes`` their finals — both supplied by
    the causal walk for rows already priced, never the row being priced. The
    pooled residual is projected on the low-order Fourier basis under a ridge
    penalty, forced zero-sum (a lattice fix moves mass, it never creates
    it), clamped to +-LATTICE_DELTA_CAP and rounded so stored bundles
    reprice byte-identically. The delta is admitted only when a time-ordered
    held-out tail of the SAME prior evidence shows a significant paired
    improvement (the NFL's "only if statistically significant" rule, applied
    as one uniform policy); otherwise the returned correction carries
    diagnostics but no delta and prices as a no-op.
    """
    out: dict = {"target": target, "sport": sport, "n_fit": 0, "n_test": 0,
                 "significant": False}
    if lattice_config(sport, target) is None:
        out["reason"] = "target_not_configured"
        return out
    support = np.asarray(support)
    pmfs = np.empty((0, len(support))) if pmfs is None else np.asarray(pmfs, float)
    outcomes = np.empty(0) if outcomes is None else np.asarray(outcomes, float)
    if pmfs.ndim != 2 or pmfs.shape[1] != len(support) or len(pmfs) != len(outcomes):
        raise ValueError("lattice evidence requires aligned pmfs and outcomes")
    inside = ((outcomes >= support[0]) & (outcomes <= support[-1])
              & np.isfinite(outcomes))
    if inside.sum() < int(min_history):
        out["reason"] = "insufficient_history"
        return out
    n_test = int(LATTICE_TEST_FRACTION * len(outcomes))
    if n_test < LATTICE_MIN_TEST or len(outcomes) - n_test < int(min_history):
        out["reason"] = "insufficient_held_out_evidence"
        return out
    train = np.flatnonzero(inside[:len(outcomes) - n_test])
    hold = np.flatnonzero(inside[len(outcomes) - n_test:])
    if len(train) < int(min_history) or len(hold) < LATTICE_MIN_TEST:
        out["reason"] = "insufficient_held_out_evidence"
        return out
    indices = np.searchsorted(support, outcomes)
    # LOCALIZED SE-aware design (2026-10-16): per-bin shrunk residual — the
    # mass-proportional principle is preserved (prior scale = a fraction of
    # local model mass, so empty tail bins get exact 0 instead of wasted
    # amplitude), but each bin's amplitude is decided by ITS OWN residual
    # against ITS OWN binomial noise floor. A global smooth basis moves mass
    # at every bin to fix one (measured 2026-10-15: NFL margin z=2.27 with
    # pooled MAE WORSE after the fit — a localization failure).
    weight = np.clip(pmfs[inside].mean(axis=0), 0., None)

    def residual(index):
        obs = np.bincount(indices[index], minlength=len(support)).astype(float)
        obs /= max(len(index), 1)
        return obs - pmfs[index].mean(axis=0)

    r_all = residual(np.flatnonzero(inside))
    r_train = residual(train)
    n_train = max(int(len(train)), 1)
    pbar_train = np.clip(pmfs[train].mean(axis=0), 0., 1.)
    se_train = np.sqrt(pbar_train * (1. - pbar_train) / n_train)
    rel = LATTICE_RELIEF.get((sport, target), {})
    delta = lattice_local_deltas(weight, r_train, se_train,
                                 relief_boost=rel.get("boost", 1.),
                                 relief_quantile=rel.get("quantile"))
    delta = _zero_sum_mass_proportional(delta, weight)  # mass moves, none is made
    for banned in lattice_config(sport, target).get("forbidden_bins", ()):
        j = int(np.searchsorted(support, banned))
        if 0 <= j < len(delta):
            delta[j] = 0.
    delta = np.clip(delta, -LATTICE_DELTA_CAP, LATTICE_DELTA_CAP)
    delta[np.abs(delta) < LATTICE_DELTA_EPS] = 0.
    delta = np.round(delta, LATTICE_ROUND)
    out.update(n_fit=int(len(train)), n_test=int(len(hold)),
               basis=("localized_per_bin_shrinkage+relief"
                      if rel else "localized_per_bin_shrinkage"),
               residual_mae_before=float(np.abs(r_all).mean()),
               residual_mae_after=float(np.abs(r_all - delta).mean()),
               delta_l1=float(np.abs(delta).sum()),
               delta_max=float(np.abs(delta).max()))
    # Per-bin collateral guard: no single bin's |residual| may worsen by
    # more than its own 1-sigma binomial noise floor (pooled model mass
    # over all inside rows). Pooled MAE can improve while one bin takes
    # real damage — the guard vetoes that trade.
    n_all = max(int(inside.sum()), 1)
    pmod_all = np.clip(pmfs[inside].mean(axis=0), 0., 1.)
    se_pool = np.sqrt(pmod_all * (1. - pmod_all) / n_all)
    excess = _collateral_excess(r_all, delta, se_pool)
    out["collateral_max_excess"] = round(float(excess.max()), 6)
    out["collateral_ok"] = bool((excess <= 0.).all())
    if not np.any(delta):
        out["reason"] = "delta_collapsed_to_zero"
        return out
    # Held-out paired significance on the per-row squared-probability loss
    # sum_k (1{y=k} - pmf_k - delta_k)^2 - sum_k (1{y=k} - pmf_k)^2
    #   = -2*delta_y + 2*sum_k pmf_k*delta_k + sum_k delta_k^2 .
    d = (-2. * delta[indices[hold]] + 2. * (pmfs[hold] @ delta)
         + float(delta @ delta))
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
    z = float(-d.mean() / se) if se > 0. else 0.
    out["gate_z"] = round(z, 3)
    if not out["collateral_ok"]:
        out["reason"] = "collateral_violation"
        return out
    if not (d.mean() < 0. and z >= LATTICE_GATE_Z
            and out["residual_mae_after"] < out["residual_mae_before"]):
        structural = _structural_lattice_fallback(sport, target, support,
                                                  r_train, d, z, out)
        if structural is None:
            out["reason"] = "held_out_not_significant"
            return out
        out.update(structural)
    out["significant"] = True
    out["delta"] = [float(v) for v in delta]
    # Fitted-lattice bounds for support-drift realignment at pricing time
    # (see _realign_lattice_delta): the delta's bins are support[0]..support[-1]
    # unit-spaced. Disclosure-safe (two ints) and round-trips through JSON.
    out["support"] = [int(support[0]), int(support[-1])]
    return out


def lattice_diagnostics(support, pmfs, outcomes, correction):
    """The MLB walk-off (+1) margin diagnostic, kept generic.

    Returns the pooled residual at the diagnostic bins before/after the
    fitted delta so a reviewer can verify, per sport/target, what the
    correction moved (e.g. MLB margin M=+1 home walk-off wins in extras
    showing an elevated positive residual that the fitted delta absorbs).
    """
    support = np.asarray(support)
    pmfs = np.empty((0, len(support))) if pmfs is None else np.asarray(pmfs, float)
    outcomes = np.asarray(outcomes, float)
    inside = (outcomes >= support[0]) & (outcomes <= support[-1])
    obs = np.bincount(np.searchsorted(support, outcomes[inside]),
                      minlength=len(support)).astype(float)
    obs /= max(int(inside.sum()), 1)
    before = obs - (pmfs[inside].mean(axis=0) if len(pmfs) else 0.)
    delta = _lattice_delta(correction)
    after = before - (delta if delta is not None else 0.)
    return {int(k): {"observed": float(obs[int(np.searchsorted(support, k))]),
                     "residual_before": float(before[int(np.searchsorted(support, k))]),
                     "residual_after": float(after[int(np.searchsorted(support, k))])}
            for k in (-1, 0, 1) if support[0] <= k <= support[-1]}


def _collateral_excess(resid_pooled, delta, se_pool):
    """Per-bin collateral excess over the 1-sigma noise floor.

    excess_k = (|r_k - delta_k| - |r_k|) - SE_k: how much more the fitted
    delta worsens bin k's absolute residual than binomial sampling noise
    alone explains. Non-positive everywhere means the fit damages no bin
    beyond its own noise floor. Exposed for unit testing; the generic gate
    vetoes on any positive excess (reason ``collateral_violation``).
    """
    r = np.asarray(resid_pooled, float)
    d = np.asarray(delta, float)
    se = np.asarray(se_pool, float)
    return (np.abs(r - d) - np.abs(r)) - se


def compose_lattice_deltas(deltas, weight=None, caps=None):
    """Compose admitted delta components into one stored vector.

    Used where a structural component (NFL key-number spikes) and the
    generic residual are admitted separately and price as one delta: sum,
    mass-proportional zero-sum, clamp, eps-zero, round — the same post
    pipeline each fit applies alone, so a composed delta obeys the same
    contract. Components stay disclosed in their own report keys.

    ``caps`` optionally supplies a PER-BIN clamp instead of the uniform
    LATTICE_DELTA_CAP (shape must match the composed vector): the NFL
    margin compose site grants LATTICE_SPIKE_CAP only at bins where a
    preregistered, individually held-out-gated spike admitted — every
    other bin keeps the generic fix bound. ``None`` keeps the exact
    pre-existing uniform behavior.
    """
    if any(part is None for part in deltas):
        raise ValueError("compose_lattice_deltas requires admitted (non-None) components")
    total = np.zeros_like(np.asarray(deltas[0], float))
    for part in deltas:
        total = total + np.asarray(part, float)
    w = (np.clip(np.asarray(weight, float), 0., None) if weight is not None
         else np.ones_like(total))
    total = _zero_sum_mass_proportional(total, w)
    if caps is None:
        total = np.clip(total, -LATTICE_DELTA_CAP, LATTICE_DELTA_CAP)
    else:
        caps = np.asarray(caps, float)
        if caps.shape != total.shape or not np.isfinite(caps).all():
            raise ValueError("compose per-bin caps must be a finite vector "
                             "matching the composed delta")
        total = np.clip(total, -caps, caps)
    total[np.abs(total) < LATTICE_DELTA_EPS] = 0.
    return [float(v) for v in np.round(total, LATTICE_ROUND)]


def lattice_summary(correction):
    """JSON-safe disclosure of a fitted correction without the delta vector:
    monitor/meta fit panels disclose magnitude and the gate verdict, not the
    full support-length residual."""
    if not correction:
        return {}
    return {k: v for k, v in correction.items() if k != "delta"}


def split_cdf(x, mean, sd, skew):
    """Mean-centered two-piece Gaussian CDF (skew=0 is Gaussian)."""
    left, right = sd * np.exp(-skew), sd * np.exp(skew)
    # Normalize the asymmetric widths to preserve the requested variance.
    variance = left * right + (1. - 2. / np.pi) * (right - left) ** 2
    correction = sd / np.sqrt(variance)
    left, right = left * correction, right * correction
    mode = mean - np.sqrt(2. / np.pi) * (right - left)
    zleft, zright = (x - mode) / left, (x - mode) / right
    return np.where(x <= mode, 2. * left / (left + right) * ndtr(zleft),
                    1. - 2. * right / (left + right) * ndtr(-zright))


def raw_cdf(h, a, kind, sport, parameters=None):
    p = defaults(sport) if parameters is None else parameters
    vh, va = float(p["home_variance"]) * h, float(p["away_variance"]) * a
    rho = float(p["correlation"])
    if np.any(vh <= 0) or np.any(va <= 0) or not -.99 < rho < .99:
        raise ValueError("invalid Gaussian variance/correlation parameters")
    cross = 2. * rho * np.sqrt(vh * va)
    mean = h + a if kind == "total" else h - a
    sd = np.sqrt(np.maximum(vh + va + (cross if kind == "total" else -cross), 1e-9))
    skew = float(p[f"{kind}_skew"])
    # Expand from predictions/parameters only, never from scored outcomes.
    # This preserves refusal checks while accommodating high-variance refits.
    high = max(SPORT_PRIORS[sport][2], int(np.ceil(np.max(np.abs(mean) + 12. * sd * np.exp(abs(skew))))) if len(mean) else 0)
    if high > 10000:
        raise ValueError("Gaussian integration support exceeds resource limit")
    support = np.arange(0, high + 1) if kind == "total" else np.arange(-high, high + 1)
    cdf = split_cdf(support[None, :] + .5, mean[:, None], sd[:, None], skew)
    if kind == "total":
        lower = split_cdf(-.5, mean, sd, skew)
        if np.any(1. - lower < 1e-12):
            raise ValueError("Gaussian total has no nonnegative score mass")
        cdf = (cdf - lower[:, None]) / (1. - lower[:, None])
    else:
        lower = split_cdf(support[0] - .5, mean, sd, skew)
        if np.any(lower > 1e-8):
            raise ValueError("Gaussian lower tail exceeds integration support")
    if np.any(cdf[:, -1] < 1. - 1e-8):
        raise ValueError("Gaussian upper tail exceeds integration support")
    return support, np.clip(cdf, 0., 1.)


def transform(cdf, x, beta):
    beta = np.asarray(beta, float)
    if beta.shape != (9,):
        raise ValueError("split-skew calibration requires nine coefficients")
    z = logit(np.clip(cdf, 1e-12, 1 - 1e-12))
    # Separate positive left/right logit slopes are continuous and increasing
    # at zero. Their ratio adds conditional skew without crossing lines.
    slope = np.exp(np.where(z < 0, (x @ beta[3:6])[:, None], (x @ beta[6:9])[:, None]))
    out = expit(z * slope + (x @ beta[:3])[:, None])
    out[cdf <= 0.] = 0.
    out[cdf >= 1.] = 1.
    return out


def no_tie(cdf, support, sport, home_share=None):
    """Push the model's tie mass off the zero margin (no-tie sports only).

    P(M=0)=0 for the NHL/MLB/NBA margin lattices — those games are always
    decided. The tie mass is redistributed to -1 (share 1-q) and +1 (share
    q). q=.5 — the default and the EXACT pre-lattice formula — splits it
    evenly; the NHL margin lattice instead supplies a causal OT/SO home-win
    estimate q (see fit_ot_home_share) so the redistribution follows the OT
    win probabilities estimated from prior OOF. NFL margins keep valid tie
    mass (ties are possible), so this remains a no-op there.

    NBA uses a DIFFERENT, rule-derived transfer (2026-10-16, declared in
    LATTICE_CONFIG before fitting): the NBA has no tie process at all, so
    the center-bin mass of the smooth continuous density (which cannot
    express P(M=0)=0 — measured 0.0246 of model mass on the 20261016
    artifact) is a BINNING ARTIFACT, not evidence about one-point finals.
    Dumping it onto +/-1 (the nearest-neighbor split) inflates both
    one-point bins by +0.0123 each on top of their own already-full smooth
    mass — the systematic +/-1 over-confidence the lattice then could not
    remove (shrinkage and collateral admissions are both capped; shipped
    z -2.23/-2.09 against the dashboard's own 95% band). The artifact mass
    is instead redistributed PROPORTIONAL TO MODEL MASS over the legal
    support — the same neutral, mass-conserving rule
    _zero_sum_mass_proportional uses for its sink: the mass goes where the
    model already puts mass, which is also where the observed deficit sits
    (the +/-3..15 shoulders under-predict by ~0.002/bin while +/-1 sits
    over-predicted). Direction declared before fitting: +/-1 residual
    DECREASES. NHL/MLB keep their fitted/nearest-neighbor split unchanged
    (their measured +/-1 evidence supports it or lacks direction).
    """
    if sport == "nfl" or support[0] >= 0:
        return cdf
    out = cdf.copy()
    j = int(np.searchsorted(support, 0))
    if sport == "nba":
        base = np.diff(np.column_stack([np.zeros(len(out)), out]), axis=1)
        tie = base[:, j]                       # P(M=0) the rule must clear
        w = base.copy()
        w[:, j] = 0.                           # legal bins only
        wsum = w.sum(axis=1, keepdims=True)
        legal = wsum > 0.
        share = np.divide(w, np.where(legal, wsum, 1.),
                          out=np.full_like(w, 1. / max(len(support) - 1, 1)),
                          where=legal)         # degenerate rows: uniform
        pmf = w + tie[:, None] * share         # sums to 1 (tie re-added)
        # clip: cumsum can land 1 ulp past 1 and crps validates range
        return np.clip(np.cumsum(pmf, axis=1), 0., 1.)
    share = .5 if home_share is None else float(np.clip(home_share, 0., 1.))
    if share == .5:
        mid = .5 * (out[:, j - 1] + out[:, j])
        out[:, j - 1] = mid
        out[:, j] = mid
    else:
        lifted = out[:, j - 1] + (1. - share) * (out[:, j] - out[:, j - 1])
        out[:, j - 1] = lifted
        out[:, j] = lifted
    return out


def crps(cdf, support, y):
    y, support, cdf = np.asarray(y, float), np.asarray(support, float), np.asarray(cdf, float)
    if cdf.shape != (len(y), len(support)) or not np.isfinite(y).all():
        raise ValueError("CRPS requires aligned finite outcomes and CDFs")
    if not np.isfinite(cdf).all() or np.any(np.diff(cdf, axis=1) < -1e-9):
        raise ValueError("CRPS requires a finite nondecreasing CDF")
    if np.any(cdf < 0.) or np.any(cdf > 1.):
        raise ValueError("CDF probabilities must be in [0, 1]")
    return (np.square(cdf - (y[:, None] <= support[None, :])).sum(axis=1)
            + np.maximum(support[0] - y, 0.) + np.maximum(y - support[-1] - 1., 0.))


def fit_map(h, a, y, kind, sport, parameters):
    if len(y) < MIN_HISTORY:
        return [0.] * 9
    support, raw = raw_cdf(h, a, kind, sport, parameters)
    x = design(h, a, sport)
    logits = logit(np.clip(raw, 1e-12, 1 - 1e-12))
    target = np.asarray(y)[:, None] <= support[None, :]
    def objective(beta):
        left = logits < 0
        slope = np.exp(np.where(left, (x @ beta[3:6])[:, None], (x @ beta[6:9])[:, None]))
        cdf = expit(logits * slope + (x @ beta[:3])[:, None])
        cdf[raw <= 0.] = 0.; cdf[raw >= 1.] = 1.
        derivative = cdf * (1. - cdf)
        base_cdf = np.diff(np.column_stack([np.zeros(len(cdf)), cdf]), axis=1)
        adjusted = no_tie(cdf, support, sport) if kind == "margin" else cdf
        error = adjusted - target
        loss = np.square(error).sum(axis=1).mean()
        if kind == "margin" and sport == "nba":
            # Exact adjoint of no_tie's mass-proportional transfer (the
            # equal-split two-bin averaging below is the WRONG derivative
            # for it and drove L-BFGS to ABNORMAL termination). The nba
            # transfer simplifies to A_m = (c_m - 1{j<=m}*t)/(1 - t) with
            # t = b_j the center mass and c_K == 1, so dL/dc_q = dA_q/(1-t)
            # for q outside {j-1, j}, and the t-path (every A_m, m >= j,
            # carries -t in the numerator, m < j the denominator) lands on
            # c_j / c_{j-1} through t = c_j - c_{j-1}. A_last == 1 by
            # construction carries no gradient (clipped/cumsum identity).
            j = int(np.searchsorted(support, 0))
            dA = 2. * error / len(y)
            dA[:, -1] = 0.
            t = base_cdf[:, j]
            W = np.maximum(1. - t, 1e-12)
            gc = dA / W[:, None]
            dLdt = ((dA[:, j:] * (cdf[:, j:] - 1.)
                     / (W[:, None] ** 2)).sum(axis=1)
                    + (dA[:, :j] * adjusted[:, :j]
                       / W[:, None]).sum(axis=1))
            gc[:, j] += dLdt
            gc[:, j - 1] -= dLdt
            gc[:, -1] = 0.
            g = gc * derivative
        else:
            if kind == "margin" and sport != "nfl":
                j = int(np.searchsorted(support, 0))
                error[:, j - 1] = error[:, j] = .5 * (error[:, j - 1] + error[:, j])
            g = 2. * error * derivative / len(y)
        width = g * logits * slope
        gradient = np.r_[x.T @ g.sum(axis=1),
                         x.T @ np.where(left, width, 0.).sum(axis=1),
                         x.T @ np.where(left, 0., width).sum(axis=1)]
        return loss + .01 * np.dot(beta, beta), gradient + .02 * beta
    result = minimize(objective, np.zeros(9), jac=True, method="L-BFGS-B",
                      bounds=[(-2., 2.)] * 3 + [(-1.5, 1.5)] * 6,
                      options={"maxiter": 200, "ftol": 1e-8})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"split-skew CRPS calibration failed: {result.message}")
    return result.x.tolist()


def sample_cdf(cdf, support, identities, kind, sport, n_draws=MC_DRAWS, seed=MC_SEED):
    """IID inverse-CDF Monte Carlo with a stable per-game stream.

    Process one game at a time: O(draws + support) memory, not games*draws.
    Identity-keyed streams preserve forecasts under shuffling and batching.
    Histogramming enforces mass conservation and cross-line monotonicity.
    """
    if not isinstance(n_draws, (int, np.integer)) or n_draws < 100:
        raise ValueError("Monte Carlo requires an integer draw count >= 100")
    used = max(int(n_draws), int(np.ceil(.25 / MC_SE_TARGET ** 2)))
    empirical = np.empty_like(cdf)
    for i, identity in enumerate(identities):
        digest = hashlib.blake2b(f"{sport}|{kind}|{seed}|{identity}".encode(), digest_size=16).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        draws = np.searchsorted(cdf[i], rng.random(used), side="right")
        if np.any(draws >= len(support)):
            raise ValueError("Monte Carlo draw exceeds finite support")
        empirical[i] = np.cumsum(np.bincount(draws, minlength=len(support))) / used
    return empirical, {"method": "inverse_calibrated_cdf_monte_carlo", "family": FAMILY,
                       "n_draws": used, "requested_draws": int(n_draws), "seed": int(seed),
                       "mc_se_totals_max": float(.5 / np.sqrt(used)),
                       "reason": "precision_guard" if used != n_draws else "requested",
                       "memory_policy": "one_game_at_a_time"}
