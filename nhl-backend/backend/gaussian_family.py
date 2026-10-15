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
LATTICE_FOURIER_ORDER = 2       # low-order Fourier modes (a1, a2, ... )
LATTICE_RIDGE = .02             # ridge shrinkage on the Fourier coefficients
LATTICE_DELTA_CAP = .03         # |delta(k)| clamp: a lattice fix, not a refit
LATTICE_DELTA_EPS = 1e-9        # coefficients below this serialize as exact 0
LATTICE_GATE_Z = 3.0            # held-out significance required to admit delta
LATTICE_TEST_FRACTION = .3      # time-ordered held-out tail of the evidence
LATTICE_MIN_TEST = 100          # held-out rows below this: never admit delta
LATTICE_ROUND = 6               # decimal rounding stored deltas reprice with

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
#   nba margin       -> "no_tie"    (P(M=0)=0; exact equal split — no
#                                  one-goal asymmetry is fitted)
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


def lattice_basis(support):
    """Low-order Fourier basis on the integer lattice.

    On the integers the prompt's a_m*cos(m*pi*k) collapses to {1, (-1)^k},
    so the smooth modes live in the scaled coordinate
    u = (k - center)/half in [-1, 1] and the parity tone (-1)^k carries the
    odd/even alternation explicitly (the NHL totals comb).
    """
    k = np.asarray(support, float)
    if len(k) < 3:
        raise ValueError("lattice basis requires at least three support points")
    center = .5 * (k[0] + k[-1])
    half = max(.5 * (k[-1] - k[0]), 1.)
    u = (k - center) / half
    columns = [np.where(np.asarray(support) % 2 == 1, 1., -1.)]
    for m in range(1, LATTICE_FOURIER_ORDER + 1):
        columns.append(np.cos(m * np.pi * u))
        columns.append(np.sin(m * np.pi * u))
    return np.column_stack(columns)


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
    basis = lattice_basis(support)
    # MASS-PROPORTIONAL design: delta(k) = pmf_bar(k) * Fourier(k) — the
    # low-pass artifact moves mass in proportion to the local model mass
    # (parity_correct's alpha*pmf is exactly this with one parity tone), and
    # an additive-only basis would waste its amplitude on empty tail bins
    # where the model already has no mass (measured on the MLB totals
    # lattice: additive fits worsened the pooled MAE while looking
    # significant on the pmf-weighted held-out loss).
    weight = np.clip(pmfs[inside].mean(axis=0), 0., None)
    design = weight[:, None] * basis

    def residual(index):
        obs = np.bincount(indices[index], minlength=len(support)).astype(float)
        obs /= max(len(index), 1)
        return obs - pmfs[index].mean(axis=0)

    r_all = residual(np.flatnonzero(inside))
    r_train = residual(train)
    # Bin-space ridge: the regression is over the K support bins (pooled
    # residual), NOT over the n_train rows — dividing by len(train) would
    # shrink the Gram ~1e-5 and let the ridge swallow every coefficient
    # (measured 2026-10-15: MLB margin's +0.033/-0.024 +/-1 residual fit to
    # a 1.5e-05-l1 no-op). Unnormalized, diag(Gram) ~ sum pmf^2 ~ 1e-2..1e-1
    # so LATTICE_RIDGE = .02 is genuine shrinkage, not a veto.
    gram = design.T @ design
    coef = np.linalg.solve(gram + LATTICE_RIDGE * np.eye(design.shape[1]),
                           design.T @ r_train)
    delta = weight * (basis @ coef)
    delta = delta - delta.mean()               # zero-sum: mass moves, none is made
    for banned in lattice_config(sport, target).get("forbidden_bins", ()):
        j = int(np.searchsorted(support, banned))
        if 0 <= j < len(delta):
            delta[j] = 0.
    delta = np.clip(delta, -LATTICE_DELTA_CAP, LATTICE_DELTA_CAP)
    delta[np.abs(delta) < LATTICE_DELTA_EPS] = 0.
    delta = np.round(delta, LATTICE_ROUND)
    out.update(n_fit=int(len(train)), n_test=int(len(hold)),
               residual_mae_before=float(np.abs(r_all).mean()),
               residual_mae_after=float(np.abs(r_all - delta).mean()),
               delta_l1=float(np.abs(delta).sum()),
               delta_max=float(np.abs(delta).max()))
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
    if not (d.mean() < 0. and z >= LATTICE_GATE_Z
            and out["residual_mae_after"] < out["residual_mae_before"]):
        out["reason"] = "held_out_not_significant"
        return out
    out["significant"] = True
    out["delta"] = [float(v) for v in delta]
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
    """
    if sport == "nfl" or support[0] >= 0:
        return cdf
    out = cdf.copy()
    j = int(np.searchsorted(support, 0))
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
        adjusted = no_tie(cdf, support, sport) if kind == "margin" else cdf
        error = adjusted - target
        loss = np.square(error).sum(axis=1).mean()
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
