"""Stint-level offense/defense RAPM (the professional-grade rating).
The game-level ridge in :mod:`player_rapm` identifies a player's beta only
from game-to-game variation in his minutes: one row per game, teammates'
columns nearly collinear, the ridge doing most of the work. This module is
the upgrade that critique demands, built on the stint table
(``run_diagnostics/nba_stint_build.py``): every maximal segment with
unchanged 5-man units contributes TWO rows - one per side's attack - so a
player's coefficients are identified by who shared the floor, which is what
makes the rating RAPM rather than a game-level proxy.
Design (all rates per 48 stint-minutes)::
    y_home = home_pts / minutes * 48
           = sum(home_5 OFF) - sum(away_5 DEF) + home_attack_intercept
    y_away = away_pts / minutes * 48
           = sum(away_5 OFF) - sum(home_5 DEF) + away_attack_intercept
``beta_off > 0`` is good offense (points scored for), ``beta_def < 0`` is
good defense (points denied: the column prices points the opponent
scored while the defender was on court). Net impact is ``off - def``.
Rows are weighted by stint minutes (a rate over 8 minutes carries a
quarter the variance of a rate over 2), folded into ``x`` and ``y`` as
``sqrt(minutes)``. Two UNRIDGED intercepts absorb each side's mean
scoring rate; pace is NOT normalized (per-48 rates, not per-100
possessions - the residual pace confounding is documented, not solved),
garbage time is NOT filtered, and overtime needs no special case.
Point-in-time discipline mirrors :mod:`player_rapm`: one incremental
normal-equation design per season, ``advance`` folds stints with gameday
STRICTLY BEFORE the target, and seasons never mix.
The prior is a SINGLE ridge toward an informative mean, not double
shrinkage. The game-level rating shrinks twice (ridge toward zero, then
toward the position mean of those same betas, with a weak k). Here the
ridge centers on ``m`` directly - ``(A + lambda*I) beta = b + lambda*m`` - where ``m``
is the decayed carryover from the previous completed season's end
ratings blended with that season's position means::
    m = carry_w * prev_beta + (1 - carry_w) * prev_position_mean
A rookie with no previous season shrinks toward the position mean; a
player with no position is not rated at all. The previous season is
complete before the target season starts, so the carryover is
point-in-time by construction.
Uncertainty comes out of the same Cholesky factor as the betas:
``Cov = sigma^2 (A + lambda*I)^-1`` with ``sigma^2 = RSS / (N - p)`` from the weighted
residuals, so a +1.5 from 20 games and a +1.5 from 70 games no longer
look identical.
``lambda`` is chosen by walk-forward cross-validation
(``run_diagnostics/nba_stint_lambda.py``), not by stars-at-the-top
inspection: the tuning script, the grid and the winning value are on
record in ``config.STINT_RAPM_LAMBDA``.
"""
from __future__ import annotations
import logging
import math
import numpy as np
import pandas as pd
try:
    from backend import config
except ImportError:
    import config
try:
    from backend.player_rapm import (
        _player_id_str,
        _season_evidence_index,
        _season_key,
        _season_of,
    )
except ImportError:
    from player_rapm import (
        _player_id_str,
        _season_evidence_index,
        _season_key,
        _season_of,
    )
logger = logging.getLogger(__name__)
#: Stint durations that can enter the fit, in minutes. Zero-duration rows
#: Stint durations that can enter the fit, in minutes. Anything below 9
#: seconds is dropped, not down-weighted: simultaneous subs at one clock
#: manufacture 0-5 second "stints", and a single basket inside one reads
#: as a scoring rate in the thousands per 48 (the top 1% of raw stint
#: rates exceed 1200/48). 9 seconds keeps ~94% of rows while removing the
#: tail that sets the residual variance for everyone. Anything above a
#: full period is a corrupt clock.
STINT_MIN_MINUTES = 0.15
STINT_MAX_MINUTES = 12.05
def stint_rows(stints: pd.DataFrame) -> pd.DataFrame:
    """Two weighted equations per stint: one per side's attack.
    Returns columns ``gameday, season, off_cols (list), def_cols (list),
    y, w`` where ``off_cols`` are the five attackers' player ids,
    ``def_cols`` the five defenders', ``y`` the attack's per-48 scoring
    rate, and ``w`` the WLS weight (stint minutes). ``side`` marks the
    attacking side (1 = home attacking, 0 = away attacking) for the two
    unridged intercepts.
    """
    cols = {"gameday", "home_5", "away_5", "minutes", "home_pts", "away_pts"}
    if stints is None or not len(stints) or not cols <= set(stints.columns):
        return pd.DataFrame(columns=["gameday", "season", "off_cols",
                                     "def_cols", "y", "w", "side"])
    work = stints.copy()
    work["minutes"] = pd.to_numeric(work.minutes, errors="coerce")
    work = work[work.minutes.gt(STINT_MIN_MINUTES)
                & work.minutes.le(STINT_MAX_MINUTES)].copy()
    if work.empty:
        return pd.DataFrame(columns=["gameday", "season", "off_cols",
                                     "def_cols", "y", "w", "side"])
    work["gameday"] = pd.to_datetime(work.gameday, errors="coerce")
    work = work[work.gameday.notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["gameday", "season", "off_cols",
                                     "def_cols", "y", "w", "side"])
    work["season"] = [_season_of(d) for d in work.gameday]
    home_pts = pd.to_numeric(work.home_pts, errors="coerce").fillna(0.0)
    away_pts = pd.to_numeric(work.away_pts, errors="coerce").fillna(0.0)
    rate = 48.0 / work.minutes
    def _ids(cell):
        if isinstance(cell, (list, tuple, np.ndarray)):
            return [str(q) for q in cell]
        return [str(q) for q in str(cell).split(",") if str(q).strip()]
    rows = []
    for day, season, h5, a5, w, yh, ya in zip(
            work.gameday, work.season, work.home_5, work.away_5,
            work.minutes, home_pts * rate, away_pts * rate):
        h5, a5 = _ids(h5), _ids(a5)
        if len(h5) != 5 or len(a5) != 5:
            continue
        rows.append((day, season, h5, a5, float(yh), float(w), 1))
        rows.append((day, season, a5, h5, float(ya), float(w), 0))
    return pd.DataFrame(rows, columns=["gameday", "season", "off_cols",
                                       "def_cols", "y", "w", "side"])
class _StintDesign:
    """Running normal equations of the stint O/D fit for ONE season.
    Columns are ``[off_0..off_{P-1}, def_0..def_{P-1}, ih, ia]``: two per
    player plus the two unridged side intercepts. ``advance`` folds rows
    with gameday STRICTLY BEFORE the target; ``set_prior`` installs the
    ridge center ``m`` (carryover); ``solve`` returns betas and standard
    errors from one Cholesky factorisation.
    """
    def __init__(self, rows: pd.DataFrame,
                 lam: float = config.STINT_RAPM_LAMBDA):
        self.lam = float(lam)
        self.players: list = []
        self.pidx: dict = {}
        self.n = 0
        self.A = np.zeros((2, 2))
        self.b = np.zeros(2)
        self.yy = 0.0
        self.nrows = 0
        self.games = 0
        self._days: list = []
        self._ptr = 0
        self._cache = None
        self.blocks: dict = {}
        self._prior_m = None
        if rows is None or not len(rows):
            return
        work = rows.dropna(subset=["y", "w"]).copy()
        if work.empty:
            return
        pids = set()
        for cell in list(work.off_cols) + list(work.def_cols):
            pids.update(cell if isinstance(cell, list) else [cell])
        self.players = sorted(pids)
        self.pidx = {p: i for i, p in enumerate(self.players)}
        self.n = len(self.players)
        dim = 2 * self.n + 2
        self.A = np.zeros((dim, dim))
        self.b = np.zeros(dim)
        blocks: dict = {}
        for day, chunk in work.groupby("gameday", sort=True):
            day_blocks = []
            for r in chunk.itertuples():
                try:
                    off = [self.pidx[q] for q in r.off_cols]
                    dfn = [self.pidx[q] + self.n for q in r.def_cols]
                except KeyError:
                    continue
                if len(off) != 5 or len(dfn) != 5:
                    continue
                w = float(r.w)
                if not math.isfinite(w) or w <= 0:
                    continue
                share = w / 48.0 * math.sqrt(w)
                # Unridged side intercepts live in the final two columns.
                ih = 2 * self.n
                ia = 2 * self.n + 1
                if int(r.side) == 1:
                    cols = np.array(off + dfn + [ih], dtype=int)
                    vals = np.array([share] * 10 + [math.sqrt(w)],
                                    dtype=float)
                else:
                    cols = np.array(off + dfn + [ia], dtype=int)
                    vals = np.array([share] * 10 + [math.sqrt(w)],
                                    dtype=float)
                day_blocks.append((cols, vals, float(r.y) * math.sqrt(w)))
            if day_blocks:
                blocks[pd.Timestamp(day)] = day_blocks
        self.blocks = blocks
        self._days = sorted(blocks)
    def advance(self, target=None) -> None:
        """Fold in every stint row STRICTLY BEFORE ``target``."""
        if not self._days:
            return
        if target is None:
            target = self._days[-1] + pd.Timedelta(days=1)
        else:
            target = pd.Timestamp(target)
        while self._ptr < len(self._days) and self._days[self._ptr] < target:
            for cols, vals, y in self.blocks[self._days[self._ptr]]:
                self.A[np.ix_(cols, cols)] += np.outer(vals, vals)
                self.b[cols] += vals * y
                self.yy += y * y
                self.nrows += 1
            self._ptr += 1
            self.games += 1
    def set_prior(self, m: np.ndarray | None) -> None:
        """Install the ridge center (2n vector, off then def)."""
        self._prior_m = None if m is None else np.asarray(m, dtype=float)
        self._cache = None
    def _rhs(self) -> np.ndarray:
        rhs = self.b.copy()
        if self._prior_m is not None:
            rhs[:2 * self.n] += self.lam * self._prior_m
        return rhs
    def solve(self):
        """``(off, def, off_se, def_se, net, net_se, intercept_home,
        intercept_away)`` as Series keyed by player_id, or None.
        Standard errors come from ``sigma^2 (A + lambda*I)^-1`` with
        ``sigma^2 = RSS / (N - p)`` over the weighted rows; the inverse runs
        through the same Cholesky factor as the betas.
        """
        if self.nrows == 0 or self.n == 0:
            return None
        if self._cache is not None and self._cache[0] == self.nrows:
            return self._cache[1]
        n, dim = self.n, 2 * self.n + 2
        M = self.A.copy()
        M[np.arange(2 * n), np.arange(2 * n)] += self.lam
        rhs = self._rhs()
        try:
            from scipy.linalg import cho_solve, cholesky
            chol = cholesky(M, lower=True)
            full = cho_solve((chol, True), rhs)
            inv = cho_solve((chol, True), np.eye(dim))
        except Exception:  # singular by pathology, not by design
            full, *_ = np.linalg.lstsq(M, rhs, rcond=None)
            try:
                inv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(M)
        off = full[:n]
        dfn = full[n:2 * n]
        resid = (self.yy - 2.0 * float(full @ rhs)
                 + float(full @ (M @ full)))
        p = 2 * n + 2
        sigma2 = max(resid, 0.0) / max(self.nrows - p, 1)
        idx = np.arange(n)
        off_var = np.maximum(np.diag(inv)[idx], 0.0) * sigma2
        def_var = np.maximum(np.diag(inv)[n + idx], 0.0) * sigma2
        cross = inv[idx, n + idx] * sigma2
        net_var = np.maximum(off_var + def_var - 2.0 * cross, 0.0)
        out = {
            "off": pd.Series(off, index=pd.Index(self.players, dtype="object")),
            "def": pd.Series(dfn, index=pd.Index(self.players, dtype="object")),
            "off_se": pd.Series(np.sqrt(off_var),
                                index=pd.Index(self.players, dtype="object")),
            "def_se": pd.Series(np.sqrt(def_var),
                                index=pd.Index(self.players, dtype="object")),
            "net": pd.Series(off - dfn,
                             index=pd.Index(self.players, dtype="object")),
            "net_se": pd.Series(np.sqrt(net_var),
                                index=pd.Index(self.players, dtype="object")),
            "ih": float(full[2 * n]),
            "ia": float(full[2 * n + 1]),
        }
        self._cache = (self.nrows, out)
        return out
def _carryover_prior(designs: dict, rows: pd.DataFrame, season: str,
                     carry_w: float = config.STINT_RAPM_CARRY_WEIGHT,
                     pos_of: dict | None = None) -> tuple[np.ndarray, dict]:
    """Ridge center for a target season from the previous completed one.
    Solves the previous season's design at its end (all its rows predate
    any target in ``season``, so this is point-in-time), then blends each
    player's end betas with his position's mean end betas. A player with
    no previous-season evidence, or no position, centers on the position
    mean (or zero when the position has no evidence either).
    """
    key = _season_key(season)
    earlier = [label for label in designs
               if _season_key(label) < key]
    prev = max(earlier, key=_season_key) if earlier else None
    design = designs[prev] if prev is not None else None
    # The vector is indexed in the PREVIOUS season's own player order, so
    # entries line up with that solve's betas; the caller maps by player
    # id onto the target season's universe. A union order here would
    # silently misalign every entry whenever the two universes differ.
    players = list(design.players) if design is not None else []
    n = len(players)
    zero = (np.zeros(2 * n),
            {"off": {}, "def": {}, "pos_off": {}, "pos_def": {}})
    if design is None or n == 0:
        return zero
    design.advance(None)
    solved = design.solve()
    if solved is None:
        return zero
    pos_of = pos_of or {}
    off_by_pos: dict = {}
    def_by_pos: dict = {}
    for pid, pos in pos_of.items():
        if pid in solved["off"].index and pos:
            off_by_pos.setdefault(pos, []).append(float(solved["off"][pid]))
            def_by_pos.setdefault(pos, []).append(float(solved["def"][pid]))
    pos_off = {p: float(np.mean(v)) for p, v in off_by_pos.items() if v}
    pos_def = {p: float(np.mean(v)) for p, v in def_by_pos.items() if v}
    m = np.zeros(2 * n)
    info = {"off": {}, "def": {}, "pos_off": pos_off, "pos_def": pos_def,
            "from_season": prev, "players": list(players)}
    for i, pid in enumerate(players):
        pos = pos_of.get(pid, "")
        po = pos_off.get(pos, 0.0)
        pd_ = pos_def.get(pos, 0.0)
        if pid in solved["off"].index:
            m[i] = carry_w * float(solved["off"][pid]) + (1 - carry_w) * po
            m[n + i] = carry_w * float(solved["def"][pid]) + (1 - carry_w) * pd_
            info["off"][pid] = float(solved["off"][pid])
            info["def"][pid] = float(solved["def"][pid])
        else:
            m[i], m[n + i] = po, pd_
    return m, info
def build_stint_rapm(base: pd.DataFrame,
                     stints: pd.DataFrame,
                     target_dates=None,
                     lam: float = config.STINT_RAPM_LAMBDA,
                     ) -> pd.DataFrame:
    """Stint O/D ratings aligned to the game-level rating frame.
    ``base`` is the output of :func:`player_rapm.build_player_rapm`
    (membership, position, team, prior minutes - all strictly prior);
    this replaces its fitted columns with the stint fit's offense,
    defense, net and standard errors. Players with no stint evidence in
    the window keep NaN raw values (never a fabricated zero); the ridge
    center they shrink toward is carried in ``prior_off``/``prior_def``.
    ``rapm_raw``/``rapm_shrunk`` are set to the stint NET for downstream
    compatibility: the single ridge layer already shrinks toward the
    informative prior, so there is no second pass and the two columns
    coincide. ``lg_rapm`` carries the prior-mean net actually used.
    """
    out_cols = ["target_date", "player_id", "position", "positions", "team",
                "prior_eff", "prior_minutes", "prior_minutes_per_game",
                "prior_games", "lg_rapm", "k_eff", "rapm_raw", "rapm_shrunk",
                "off_raw", "def_raw", "net_raw", "off_se", "def_se",
                "net_se", "prior_off", "prior_def",
                "days_since_appearance", "roster_source", "roster_observed_at"]
    empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in out_cols})
    if base is None or not len(base) or stints is None or not len(stints):
        return empty
    rows = stint_rows(stints)
    if rows.empty:
        return empty
    if target_dates is None:
        dates = pd.Series(sorted(pd.to_datetime(
            base.target_date).dropna().unique()))
    else:
        dates = pd.Series(pd.to_datetime(pd.Series(target_dates),
                                         errors="coerce").dropna().unique())
    dates = dates.sort_values().reset_index(drop=True)
    if dates.empty:
        return empty
    designs: dict = {}
    for season in rows.season.dropna().unique():
        chunk = rows[rows.season == season]
        designs[str(season)] = _StintDesign(chunk, lam=lam)
    priors: dict = {}
    pos_of: dict = {}
    for pid, pos in zip(_player_id_str(base.player_id).values,
                        base.position.astype(str)):
        if pos and pos != "nan":
            pos_of.setdefault(str(pid), pos)
    frames = []
    for target in dates:
        season = _season_of(target)
        design = designs.get(season)
        if design is None:
            continue
        if season not in priors:
            m_full, info = _carryover_prior(
                designs, rows, season, pos_of=pos_of)
            # Map the carryover season's player order onto this design's.
            # info["players"] IS m_full's order (the previous season's own
            # list): mapping through any other order silently misaligns.
            pmap = {p: i for i, p in enumerate(info.get("players", []))}
            m = np.zeros(2 * design.n)
            for i, pid in enumerate(design.players):
                j = pmap.get(pid)
                if j is not None and j < len(m_full) // 2:
                    m[i] = m_full[j]
                    m[design.n + i] = m_full[len(m_full) // 2 + j]
                else:
                    pos = pos_of.get(pid, "")
                    m[i] = info["pos_off"].get(pos, 0.0)
                    m[design.n + i] = info["pos_def"].get(pos, 0.0)
            priors[season] = (m, info)
        m, info = priors[season]
        if len(m) != 2 * design.n:
            m = np.zeros(2 * design.n)
        design.set_prior(m)
        design.advance(target)
        solved = design.solve()
        day = base[pd.to_datetime(base.target_date) == pd.Timestamp(target)]
        if day.empty:
            continue
        snap = day.copy()
        snap["player_id"] = _player_id_str(snap.player_id).values
        for col in ("off_raw", "def_raw", "net_raw", "off_se", "def_se",
                    "net_se", "prior_off", "prior_def"):
            snap[col] = np.nan
        if solved is not None:
            snap["off_raw"] = snap.player_id.map(solved["off"])
            snap["def_raw"] = snap.player_id.map(solved["def"])
            snap["net_raw"] = snap.player_id.map(solved["net"])
            snap["off_se"] = snap.player_id.map(solved["off_se"])
            snap["def_se"] = snap.player_id.map(solved["def_se"])
            snap["net_se"] = snap.player_id.map(solved["net_se"])
        # No stint evidence in the window is "not in the fit", exactly
        # like the game-level rating's zero-column rule.
        evidenced = set(design.players[i] for i in range(design.n)
                        if design.A[i, i] > 0 or design.A[design.n + i,
                                                          design.n + i] > 0)
        has = snap.player_id.isin(evidenced)
        for col in ("off_raw", "def_raw", "net_raw", "off_se", "def_se",
                    "net_se"):
            snap[col] = snap[col].where(has)
        pmap_ids = {pid: i for i, pid in enumerate(design.players)}
        snap["prior_off"] = snap.player_id.map(
            lambda pid: float(m[pmap_ids[pid]]) if pid in pmap_ids else np.nan)
        snap["prior_def"] = snap.player_id.map(
            lambda pid: float(m[design.n + pmap_ids[pid]])
            if pid in pmap_ids else np.nan)
        snap["rapm_raw"] = snap.net_raw
        snap["rapm_shrunk"] = snap.net_raw
        snap["lg_rapm"] = snap.prior_off - snap.prior_def
        snap["k_eff"] = np.nan
        frames.append(snap)
    if not frames:
        return empty
    out = pd.concat(frames, ignore_index=True)
    for col in out_cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[out_cols].sort_values(
        ["target_date", "player_id"]).reset_index(drop=True)
