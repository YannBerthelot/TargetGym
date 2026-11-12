"""Bump tests, step-response fits and their validity, shared by the protocols.

An expert is tuned on the readings it will have, so its tests are single runs
(docs/proposals/expert-protocols-noise.md, N1 to N8): ``Plant.bump`` holds,
records a baseline and steps, and ``single_run`` reads the noise band, sizes the
step and holds the delay it reads in the fit. ``noise_tune`` then sizes the
filter and the gain on C6 and C7 of docs/proposals/expert-protocols-process.md.

The oracle horizon protocol (docs/proposals/oracle-horizon-protocol.md, H7 to
H9 and, from Amendment 1, H14 to H19) measures a simulator instead, and keeps
its twin run:

- ``Plant``: the shipped expert runs one episode from each training seed's
  reset, then the moves are held at the mean of its last commands (amended
  H8); one MV is stepped and every CV read against an unstepped run on the
  same noise, up to the first trip of either run (H9);
- both structures, FOPDT and integrating with lag, are fitted to every
  response by output-error least squares on the recorded input (H11), and
  judged by V1 to V4 (H14), with the noise and the fit measure of
  Amendment 2 (N1, N2);
- ``identify``: failed tests are retested at most twice (H15, H18); each
  pair-seed ends valid, insignificant (H16) or invalid, under the class its CV
  takes by majority (H17, H19).

Not hashed into any fingerprint (``provenance``): it measures, and what it
measures is written to data files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.signal import fftconvolve

__all__ = ["MODELS", "TRAIN_SEEDS", "Loop", "Plant", "Run", "class_of", "evidence"]
__all__ += ["FIT_QUIET", "fit", "identify", "insignificant", "judge", "nrmse_fit"]
__all__ += ["quiet_fits"]
__all__ += ["DEAD_BAND", "Bump", "Instrument", "Read", "band_exit", "noise_band"]
__all__ += ["noise_tune", "pick_fit", "rga", "simc", "single_run", "sized_step"]

#: The training seeds every protocol identifies on; 200-209 are held out.
TRAIN_SEEDS = tuple(range(100, 110))
MODELS = ("fopdt", "integrating_lag")
SNR_MIN, FIT_MIN = 10.0, 80.0  # V1 in noise bands (D8), V4 in percent (D10)
#: What a fit failing V1 alone needs to count toward P (Amendment 4), on V4's
#: measure, ``nrmse_fit`` (N2).
FIT_QUIET = 95.0
RETESTS, GROWTH = 2, 4  # D11: two retests, step and window up to 4 times
_LAG = {"fopdt": "tau", "integrating_lag": "tau2"}
# N1 to N7: 5 bands, a filter under 10 % of Ti, a spread of 1.5 (C5) and a dead
# band of 1 % of span are read. Ours, fixed before any run: 60 samples of
# baseline, the window the noise was first measured on, and at least 8 updates
# for a line through held values; 3 updates of persistence; 3 passes.
BASELINE, MIN_UPDATES, PERSIST, PASSES = 60, 8, 3, 3
BANDS, FILTER_CAP, SPREAD, DEAD_BAND = 5.0, 0.1, 1.5, 0.01


# ---------------------------------------------------------------------------
# Fits (H11) and their checks (H14)
# ---------------------------------------------------------------------------


def _unit_step(model, q, t):
    k, tau, theta = q
    x = np.maximum(t - theta, 0.0)
    if model == "fopdt":
        return k * (1 - np.exp(-x / tau))
    return k * (x - tau * (1 - np.exp(-x / tau)))


def _simulate(model, Q, dt, U, first=None):
    """The output at dt, 2 dt, ... of one ``model`` per column of ``U``, each
    input held over its step: a sum of unit steps at the input's changes.
    ``first`` is the first sample's time, if not ``dt``."""
    lags = (0.0 if first is None else first - dt) + dt * np.arange(1, len(U) + 1)
    moves = np.diff(U, axis=0, prepend=0.0)
    out = [
        fftconvolve(moves[:, i], _unit_step(model, q, lags))[: len(U)]
        for i, q in enumerate(Q)
    ]
    return np.sum(out, axis=0)


def nrmse_fit(y, yhat) -> float:
    """The fit in percent, 100 (1 - |y - yhat| / |y - y_pre|), with y_pre = 0
    the response's pre-step value (N2, ours: MWC's mean makes a pure gain
    0 / 0)."""
    spread = np.linalg.norm(y)
    return float(100 * (1 - np.linalg.norm(y - yhat) / spread)) if spread else -math.inf


def _guesses(model, t, dy, ref):
    moved = np.flatnonzero(np.abs(dy) > 0.02 * np.max(np.abs(dy)))
    th = float(t[moved[0] - 1]) if moved.size and moved[0] > 0 else 0.0
    if model == "fopdt":
        reached = np.flatnonzero(np.abs(dy) >= 0.632 * abs(dy[-1]))
        tau = max(float(t[reached[0]]) - th if reached.size else t[-1], t[0])
        return [(dy[-1] / ref, tau, th), (dy[-1] / ref, tau, 0.0)]
    h = len(t) // 2
    kp = (dy[-1] - dy[h]) / (t[-1] - t[h]) / ref
    lag = min(max(t[-1] - dy[-1] / (kp * ref), 0.0), t[-1]) if kp else 0.0
    return [(kp, max(lag / 2, t[0]), lag / 2), (kp, t[0], th)]


def fit(model: str, dt: float, dy, U, theta=None, first=None) -> dict:
    """Output-error least squares of ``model`` (one per column of ``U``, the
    recorded input deviations, the pair's own MV first) to the response
    ``dy``, with V1 to V4 of H14 and the 3 sigma half-width of the gain.
    sigma is the spread of the residual over the fit window (N1). ``theta``
    holds the pair's delay at that value (noise protocol, N4), and ``first`` is
    the first sample's time, if not ``dt``."""
    dy = np.asarray(dy, float)
    U = np.asarray(U, float).reshape(len(dy), -1)
    t = (0.0 if first is None else first - dt) + dt * np.arange(1, len(dy) + 1)
    n = U.shape[1]
    ref = U[np.argmax(np.abs(U[:, 0])), 0] or 1.0
    tau_hi = np.inf if model == "fopdt" else t[-1]
    lo, hi = [-np.inf, 1e-9, 0.0] * n, [np.inf, tau_hi, t[-1]] * n
    if theta is not None:
        lo[2], hi[2] = theta, theta + 1e-9 * t[-1]
    best = None
    for g in _guesses(model, t, dy, ref):
        x0 = np.clip(list(g) + [0.0, g[1], 0.0] * (n - 1), lo, hi)
        r = least_squares(
            lambda q: _simulate(model, q.reshape(n, 3), dt, U, first) - dy,
            x0,
            bounds=(lo, hi),
            x_scale="jac",
        )
        best = r if best is None or r.cost < best.cost else best
    Q, sse = best.x.reshape(n, 3), 2 * best.cost
    jac = best.jac if theta is None else np.delete(best.jac, 2, axis=1)
    cov = np.linalg.pinv(jac.T @ jac) * sse / max(len(dy) - jac.shape[1], 1)
    lag = _LAG[model]
    out = {"model": model, "k": Q[0, 0], lag: Q[0, 1], "theta": Q[0, 2]}
    if n > 1:
        out["others"] = [dict(zip(("k", lag, "theta"), map(float, q))) for q in Q[1:]]
    at_bound = np.isclose(Q[:, 2], t[-1], rtol=1e-6) | (
        np.isclose(Q[:, 1], tau_hi, rtol=1e-6)
    )
    score, sigma = nrmse_fit(dy, dy + best.fun), float(np.std(best.fun))
    peak = float(np.max(np.abs(dy)))
    snr = peak / (6 * sigma) if sigma > 0 else math.inf
    return {
        **{k: (v if isinstance(v, (str, list)) else float(v)) for k, v in out.items()},
        "sse": float(sse),
        "fit": score,
        "sigma": sigma,
        "snr": snr,
        "V1": snr >= SNR_MIN,
        "k_3sigma": float(3 * math.sqrt(max(cov[0, 0], 0.0))),
        "V2": bool(np.all(5 * Q[:, 1] + Q[:, 2] <= t[-1])),
        "V3": not bool(np.any(at_bound)),
        "V4": score >= FIT_MIN,
    }


def judge(dy, U, dt: float) -> dict:
    """One CV's response in one test, with both fits."""
    dy = np.asarray(dy, float)
    peak = float(np.max(np.abs(dy))) if len(dy) else 0.0
    out = {"samples": len(dy), "zero": len(dy) >= 4 and peak == 0.0}
    usable = len(dy) >= 4 and peak > 0
    out["fits"] = {m: fit(m, dt, dy, U) for m in MODELS} if usable else {}
    return out


def _fails(a, model) -> set:
    """The checks of H14 that ``model``'s fit of response ``a`` fails."""
    if model not in a["fits"]:
        return {"zero" if a["zero"] else "short"}
    return {v for v in ("V1", "V2", "V3", "V4") if not a["fits"][model][v]}


def passes(a, model=None) -> bool:
    """H14 for one structure, or for either when ``model`` is None."""
    return any(not _fails(a, m) for m in ([model] if model else MODELS))


def insignificant(a) -> bool:
    """H16: identically zero, inside one noise band, or a gain whose 3 sigma
    interval holds zero (on the closer of the two fits)."""
    if not a["fits"]:
        return a["zero"]
    f = min(a["fits"].values(), key=lambda f: f["sse"])
    return f["snr"] < 1 or abs(f["k"]) <= f["k_3sigma"]


def evidence(a):
    """H17 step 1: the structure one pair-seed's response supports, or None."""
    if not a["fits"]:
        return None
    f, g = (a["fits"][m] for m in MODELS)
    ok_f, ok_g = (x["V2"] and x["V3"] and x["V4"] for x in (f, g))
    if ok_f and ok_g:
        return MODELS[0] if f["sse"] <= g["sse"] else MODELS[1]
    if ok_f:
        return MODELS[0]
    return MODELS[1] if ok_g and not f["V2"] else None


def class_of(analyses):
    """H17 step 2: a CV's class, the majority of its seed evidence; on a tie,
    the structure with the smaller sum of (100 - fit). ``(class, votes)``."""
    said = [(a, evidence(a)) for a in analyses]
    said = [(a, e) for a, e in said if e]
    votes = {m: sum(e == m for _, e in said) for m in MODELS}
    if not said:
        return None, votes
    if votes[MODELS[0]] != votes[MODELS[1]]:
        return max(MODELS, key=votes.get), votes
    miss = {m: sum(100 - a["fits"][m]["fit"] for a, _ in said) for m in MODELS}
    return min(MODELS, key=miss.get), votes


# ---------------------------------------------------------------------------
# Single-run tests (expert-protocols-noise.md, N1 to N8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instrument:
    """What a reading's datasheet says, times in ``delta_t``'s units: the plant
    step ``dt``, ``every`` steps between updates, the ``dead_time`` of what it
    reports, whether it is ``sampled`` (holding its value between updates, not
    only its noise) and its ``rounding``. A sensor's lag has no term: it is
    dynamics a step response shows, and the fit takes it into the model's lag.
    """

    dt: float
    every: int = 1
    dead_time: float = 0.0
    sampled: bool = False
    rounding: float = 0.0

    @classmethod
    def of(cls, env, params, name):
        """The instrument on field or channel ``name``, or a plain reading."""
        dt = float(params.delta_t)
        for r in getattr(env, "readings", ()):
            if name in (getattr(r, "field", None), getattr(r, "channel", None)):
                every = max(round(r.interval / dt), 1)
                return cls(dt, every, r.dead_time, r.sampled, float(r.step))
        return cls(dt)

    @property
    def h(self) -> float:
        """The time between updates, the grid a test is read on (N8)."""
        return self.every * self.dt

    @property
    def half_sample(self) -> float:
        """C4's h/2. A sampled reading's cycle replaces the step, which with a
        dead time of one cycle makes the 1.5 cycles of analyser practice."""
        return (self.h if self.sampled else self.dt) / 2

    @property
    def baseline(self) -> int:
        """N1's hold, in steps."""
        return max(BASELINE, MIN_UPDATES * self.every)

    def delay(self, exit_index) -> float:
        """N4's delay from the first update outside the band, which brackets it
        within one update: the middle of what the grid can tell, and never
        under the dead time the datasheet gives. No exit reads as that dead
        time. On a fast reading this is the steps before the reading moved."""
        if exit_index is None:
            return self.dead_time
        return max(exit_index * self.h - (self.h - self.dt) / 2, self.dead_time)

    def gate(self, band: float, delay: float) -> dict:
        """N8 for an analyser loop: act on an update only past ``band``, and
        after a move sit out the updates that cannot show it yet, ``wait``
        counting from the update acted on to the first that can."""
        wait = math.ceil((delay + self.dt) / self.h - 1e-9)
        return {"every": self.every, "band": band, "wait": wait}


@dataclass
class Bump:
    """One single run: ``base`` the reading once per update before the step,
    ``dy`` after it less the baseline's mean, ``U`` the (m, 1) input deviation
    over the same updates, and ``step`` the size. ``trip`` is where one cut it,
    and ``clock`` the plant's step count when the step landed."""

    base: np.ndarray
    dy: np.ndarray
    U: np.ndarray
    step: float
    trip: tuple | None = None
    clock: int = 0


@dataclass
class Read:
    """N1 to N4 on one loop: a fit per test, the band, the last step and the
    response in bands, which a test too small to read ships with."""

    fits: list
    nb: float
    step: float
    ratio: float


def noise_band(baselines, rounding: float = 0.0) -> float:
    """N2: 6 sd of the reading about a line through each baseline, pooled over
    them, since an analyser gives each only a few values. At least ``rounding``."""
    sse, dof = 0.0, 0
    for y in (np.asarray(b, float) for b in baselines):
        if len(y) > 2:
            t = np.arange(len(y), dtype=float)
            sse += float(np.sum((y - np.polyval(np.polyfit(t, y, 1), t)) ** 2))
            dof += len(y) - 2
    return max(6 * math.sqrt(sse / max(dof, 1)), rounding)


def band_exit(dy, nb: float, persist: int = PERSIST):
    """N4: the index of the first of ``persist`` readings in a row outside the
    band on one side, or None."""
    side = np.sign(dy) * (np.abs(dy) > nb / 2)
    for i in range(len(side) - persist + 1):
        if side[i] and np.all(side[i : i + persist] == side[i]):
            return i
    return None


def sized_step(du: float, k: float, nb: float, room: float, record=None):
    """N3: ``(step, ratio)``. The larger of ``du`` and what moves the reading 5
    bands within ``room``, and the bands it moves. A slope model's ``k`` is a
    rate, so it reads over its ``record`` length."""
    reach = abs(k) * (record or 1.0)
    want = BANDS * nb / reach if reach else math.inf
    step = max(abs(du), min(want, room))
    return math.copysign(step, du), step * reach / nb if nb else math.inf


def _read(bump, nb, instrument, model, delay=None):
    """One test's model, its delay read at the band exit and held (N4). The
    ramp of an integrating loop spends ``nb / (2 |k' du|)`` inside the band.
    ``delay`` holds a datasheet's figure instead, where no step can clear the
    band."""
    if len(bump.dy) < 4:
        return None
    h, first = instrument.h, instrument.dt
    if delay is not None:
        return fit(model, h, bump.dy, bump.U, delay, first)
    exit_index = band_exit(bump.dy, nb)
    delay = instrument.delay(exit_index)
    if model != "fopdt" and exit_index is not None:
        slope = abs(fit(model, h, bump.dy, bump.U, first=first)["k"] * bump.step)
        late = nb / (2 * slope) if slope else 0.0
        delay = max(delay - late, instrument.dead_time)
    return fit(model, h, bump.dy, bump.U, delay, first)


def single_run(run, du, room, instrument, model="fopdt", delay=None) -> Read:
    """N1 to N4 on one loop: ``run(step)`` gives a :class:`Bump` per seed. The
    band sizes the step, rerun once if that is larger. ``room`` is what the
    actuator allows; a limit that blocks the step is not an error. ``delay`` is
    held in the fit in place of the band's read (``_read``)."""

    def fits_at(step):
        bumps = run(step)
        nb = noise_band([b.base for b in bumps], instrument.rounding)
        fits = [f for b in bumps if (f := _read(b, nb, instrument, model, delay))]
        record = None
        if model != "fopdt":
            record = instrument.h * max(len(b.dy) for b in bumps)
        return fits, nb, record

    fits, nb, record = fits_at(du)
    k = float(np.median([abs(f["k"]) for f in fits]))
    step, ratio = sized_step(du, k, nb, room, record)
    if abs(step) > abs(du):
        fits, nb, record = fits_at(step)
        k = float(np.median([abs(f["k"]) for f in fits]))
        ratio = sized_step(step, k, nb, abs(step), record)[1]
    return Read(fits, nb, step, ratio)


def pick_fit(fits):
    """C5 as decided for noise: ``(fit, spread, explained)``. The median model,
    unless the spread passes the 1.5 that costs little by more than the
    measured noise stretches any spread (ours), the ratio of the median gain
    plus and minus its 3 sigma. Then the highest gain, with its own model."""
    gains = np.abs([f["k"] for f in fits])
    spread = float(gains.max() / gains.min()) if gains.min() else math.inf
    mid, half = np.median(gains), np.median([f["k_3sigma"] for f in fits])
    explained = float((mid + half) / (mid - half)) if mid > half else math.inf
    if spread > SPREAD * explained:
        return fits[int(np.argmax(gains))], spread, explained
    keys = "k", _LAG[fits[0]["model"]], "theta"
    median = {key: float(np.median([f[key] for f in fits])) for key in keys}
    return {**fits[0], **median}, spread, explained


def simc(k: float, tau1, theta: float, tau_c: float):
    """C7, ``(Kc, tau_I)``. ``tau1`` is None on an integrating loop, whose
    ``k`` is the slope k'."""
    if tau1 is None:
        return 1 / (k * (tau_c + theta)), 4 * (tau_c + theta)
    tau1 = tau1 + theta / 3 if tau1 < theta else tau1
    return tau1 / (k * (tau_c + theta)), min(tau1, 4 * (tau_c + theta))


def rga(G) -> np.ndarray:
    """Bristol's relative gain array of a square gain matrix, ``G * inv(G)^T``:
    each element is a pairing's gain with the other loops open over its gain
    with them closed, so rows and columns sum to 1 and a pairing wants it near 1
    and never negative."""
    G = np.asarray(G, float)
    return G * np.linalg.inv(G).T


def _attenuation(tf: float, h: float) -> float:
    """What a first-order filter of ``tf`` leaves of white noise read every
    ``h``: sqrt(a / (2 - a)), a = 1 - exp(-h / tf)."""
    a = -math.expm1(-h / tf) if tf else 1.0
    return math.sqrt(a / (2 - a))


def _filter_time(kc: float, nb: float, h: float, db: float) -> float:
    """N5 solved for the filter: ``|Kc| NB sqrt(a / (2 - a)) = DB``."""
    r = db / (abs(kc) * nb)
    return 0.0 if r >= 1 else -h / math.log1p(-2 * r**2 / (1 + r**2))


def noise_tune(k, tau1, theta, nb, h, db=None, kc_min=0.0, rule=None) -> dict:
    """N5 to N7 on C6 and C7. ``theta`` has C4's half sample, ``h`` is the
    reading's update time. ``db`` is the dead band in the output's units,
    ``DEAD_BAND`` of its span, multiplied through any inner loop's gain; None on
    an actuator with no sourced figure, which N5 and N7 leave alone. ``rule(theta,
    tau_c) -> (Kc, Ti)`` replaces C7, for a loop on another tuning rule."""
    law = rule or (lambda th, tc: simc(k, tau1, th, tc))
    kc, ti = law(theta, theta)
    tf, cap, tau_c = 0.0, FILTER_CAP * ti, theta
    if db is not None and nb > 0:
        for _ in range(PASSES):
            tf = min(_filter_time(kc, nb, h, db), cap)
            kc, ti = law(theta + tf, theta + tf)
        tau_c = theta + tf
        most = db / (nb * _attenuation(tf, h))
        if abs(kc) > most:
            lowered = max(most, min(kc_min, abs(kc)))
            if rule is None:
                tau_c = 2 * (theta + tf) * abs(kc) / lowered - (theta + tf)
            else:
                tau_c = brentq(
                    lambda tc: abs(law(theta + tf, tc)[0]) - lowered,
                    theta + tf,
                    (theta + tf) * (1.0 + 2.0 * abs(kc) / lowered) ** 2,
                )
            kc, ti = law(theta + tf, tau_c)
    return {"Kc": kc, "Ti": ti, "Tf": tf, "theta": theta + tf, "tau_c": tau_c}


# ---------------------------------------------------------------------------
# Tests and retests (H8, H9, H15, H18), and the labels (H16, H17, H19)
# ---------------------------------------------------------------------------


class TrippedDuringTest(RuntimeError):
    """A test run tripped the plant. Under the continuing default the plant
    restarts inside the same trajectory (``base.failure_kernel``), so what
    follows is the restart, not the response: the run is refused, never fitted."""


@dataclass(frozen=True)
class Loop:
    """A P-only loop of H18: MV ``mv`` holds CV ``cv`` with gain ``kc``,
    closed ``n_pre`` steps before the step."""

    cv: int
    mv: int
    kc: float
    n_pre: int


@dataclass
class Run:
    """One test: MV ``j`` stepped by ``step`` for ``window`` steps. ``dy`` and
    ``base`` are (m, n_cv), ``U`` the (m, n_mv) MV deviations, ``y0`` the CVs
    at the step, ``trips`` ``{run: (index, cv)}`` for each run that tripped."""

    step: float
    window: int
    loops: tuple
    dy: np.ndarray
    base: np.ndarray
    U: np.ndarray
    y0: np.ndarray
    trips: dict

    @property
    def cut(self) -> bool:
        return bool(self.trips)


def _trial(plant, s, j, step, window, loops):
    run = plant.test(s, j, step, window, loops)
    moving = [j] + [m for m in range(plant.n_mv) if m != j and np.any(run.U[:, m])]
    A = [judge(run.dy[:, i], run.U[:, moving], plant.dt) for i in range(plant.n_cv)]
    return run, A


def _closer(a):
    """A failing pair's failed checks: those of its closer structure (fewer
    failures, then the lower squared error). Ours."""
    if not a["fits"]:
        return _fails(a, MODELS[0])
    return min((_fails(a, m) for m in MODELS), key=len)


def _snr(a) -> float:
    """A response's SNR on its closer fit, 0 when it has none."""
    fits = a["fits"].values()
    return min(fits, key=lambda f: f["sse"])["snr"] if fits else 0.0


def _step_cap(plant, s, j, run, A, size):
    """10.4: the MV's range, SEB eq. 20-75 on every CV with a trip limit, and
    four times the H7 step."""
    lo, hi = plant.mv_range
    cap = min(
        (hi - plant.u0[s][j]) if run.step > 0 else (plant.u0[s][j] - lo),
        GROWTH * abs(size),
    )
    for i, limits in enumerate(plant.limits):
        bounds = [b for b in limits if b is not None]
        if not bounds or not A[i]["fits"]:
            continue
        f = min(A[i]["fits"].values(), key=lambda f: f["sse"])
        gain = abs(f["k"]) * (run.window * plant.dt if f["model"] != "fopdt" else 1.0)
        room = min(abs(run.y0[i] - b) for b in bounds) - 3 * f["sigma"]
        cap = min(cap, max(room, 0.0) / gain) if gain else cap
    return cap


def _loops(plant, s, j, run, A, failing, trials):
    """H18: a P-only loop on each CV that tripped, its gain by COOPER D.2 or
    eq. 5.4 from the better fit (higher V4) of that loop's MV-CV pair."""
    loops, used = list(run.loops), {lp.mv for lp in run.loops}
    for c in {cv for _, cv in run.trips.values() if cv is not None}:
        if any(lp.cv == c for lp in loops):
            continue
        if c in failing:
            mv = j
        elif c in plant.expert_mv:
            mv = plant.expert_mv[c]
        else:
            mv = max(
                {m for m, _ in trials},
                key=lambda m: max(_snr(a[c]) for _, a in trials[m, s]),
            )
        a = A[c] if mv == j else trials[mv, s][-1][1][c]
        if mv in used or not a["fits"]:
            continue  # no model: the pair-seed stays invalid
        f = max(a["fits"].values(), key=lambda f: f["fit"])
        theta_p = f["theta"] + plant.dt / 2 + f.get("tau2", 0.0) / 2
        if f["model"] == "fopdt":
            kc = (0.2 / f["k"]) * (f["tau"] / theta_p) ** 1.22 if f["k"] else 0.0
        else:
            kc = 1 / (f["k"] * theta_p) if f["k"] else 0.0
        if kc:
            n_pre = math.ceil(5 * math.sqrt(10) * theta_p / plant.dt - 1e-9)
            loops.append(Loop(c, mv, float(kc), n_pre))
            used.add(mv)
    return tuple(loops)


def _remedies(plant, s, j, run, A, size, failing, trials):
    """H15: the next (step, window, loops) from every failing pair's checks."""
    fails = set().union(*(_closer(A[i]) for i in failing))
    step, window, loops = run.step, run.window, run.loops
    if "V1" in fails:
        grown = min(2 * abs(step), _step_cap(plant, s, j, run, A, size))
        step = math.copysign(max(grown, abs(step)), step)
    if fails & {"V2", "V3"} and not run.cut:
        window = min(2 * window, GROWTH * plant.n_ep)
    if fails and run.cut:
        loops = _loops(plant, s, j, run, A, failing, trials)
    return step, window, loops


def identify(plant, sizes, mvs=None):
    """H15 to H19 on every seed of ``plant``, every MV or those in ``mvs``:
    ``(pairs, classes)``, pairs keyed ``(mv, cv, seed index)`` with their tests,
    kept test, label, fit and failed checks; classes keyed by CV with the vote."""
    trials = {}
    for j in range(plant.n_mv) if mvs is None else mvs:
        for s in range(plant.n_seeds):
            step = plant.direction(plant.u0[s][j], sizes[j])
            trials[j, s] = [_trial(plant, s, j, step, plant.n_ep, ())]
    for (j, s), tried in trials.items():
        for _ in range(RETESTS):
            run, A = tried[-1]
            failing = [
                i
                for i in range(plant.n_cv)
                if not A[i]["zero"] and not any(passes(a[i]) for _, a in tried)
            ]
            if not failing:
                break
            nxt = _remedies(plant, s, j, run, A, sizes[j], failing, trials)
            if nxt == (run.step, run.window, run.loops):
                break
            tried.append(_trial(plant, s, j, *nxt))
    pairs = {}
    for (j, s), tried in trials.items():
        for i in range(plant.n_cv):
            seen = [a[i] for _, a in tried]
            # A test with no samples says nothing, so it is not the last test
            # (horizon protocol, H23).
            last = max((n for n, a in enumerate(seen) if a["samples"]), default=0)
            kept = next((n for n, a in enumerate(seen) if passes(a)), last)
            gone = not passes(seen[kept]) and insignificant(seen[last])
            pairs[j, i, s] = {
                "tests": [(run, a[i]) for run, a in tried],
                "kept": kept,
                "label": "insignificant" if gone else None,
            }
    classes = {}
    for i in range(plant.n_cv):
        mine = [p for (_, c, _), p in pairs.items() if c == i and p["label"] is None]
        cls, votes = class_of([p["tests"][p["kept"]][1] for p in mine])
        classes[i] = {"class": cls, "votes": votes}
        for p in mine:
            a = p["tests"][p["kept"]][1]
            bad = sorted(_fails(a, cls)) if cls else ["no class"]
            p.update(label="invalid" if bad else "valid", failed=bad)
            p["fit"] = a["fits"].get(cls) if cls else None
    return pairs, classes


def quiet_fits(pairs, classes) -> dict:
    """The fits that fail V1 alone at a fit of at least ``FIT_QUIET`` (horizon
    protocol, Amendment 4), keyed ``(mv, cv, seed index, test)``. A gain needs a
    settled fit, not a quiet one, as in H21, so these count toward P, though
    they set no T_c. Every test of a pair is read, under its CV's class: a retest
    can keep a fast valid fit over a slow quiet one. As in H21, an insignificant
    pair has none."""
    out = {}
    for (j, i, s), pair in pairs.items():
        cls = classes[i]["class"]
        if cls is None or pair["label"] == "insignificant":
            continue
        for n, (_, a) in enumerate(pair["tests"]):
            if _fails(a, cls) == {"V1"} and a["fits"][cls]["fit"] >= FIT_QUIET:
                out[j, i, s, n] = a["fits"][cls]
    return out


def away_from_nearer_limit(u: float, size: float) -> float:
    """C2's direction for a raw MV in [-1, 1]: down from the upper half."""
    return -size if u > 0 else size


class Plant:
    """A task at steady operation on each training seed (amended H8), ready to
    be bump-tested one MV at a time.

    ``cvs`` are ``f(obs, state, params)`` and ``limits`` their trip limits
    ``(lo, hi)``, either may be None. The expert settles one episode; the
    moves then hold ``start(actions, memories)``, by default the mean of its
    last ``window`` commands. ``controller`` settles it in the expert's place,
    since a recipe's test must not start from the gains it replaces.
    ``actuate(u, y, memory) -> (action, memory)`` maps the test's MVs to the
    env's action (by default the raw action), and ``applied(u, action)`` is
    what a test records as its input (by default ``u``), for a loop closed
    inside ``actuate`` whose MV is not all that moves the plant. ``reads`` are
    further ``f(obs, state, params)``, appended to the ``y`` that ``actuate`` and
    ``settled`` get, for a closed loop that reads what no CV does.
    ``settled(u0, y)`` replaces the held moves once the plant has settled, for a
    test MV that is not the expert's command, and ``angles`` lists the CVs that
    wrap, which are unwrapped before a response is read. With ``paired`` a bump
    is read against the same seed's unstepped run, whose noise and disturbances
    are the stepped run's, instead of against its baseline's mean: a slow walk or
    drift the two share then leaves the response. The baseline still sets the
    noise band, which a real reading has.
    """

    def __init__(
        self,
        spec,
        params,
        cvs,
        limits,
        *,
        window=20,
        seeds=TRAIN_SEEDS,
        start=None,
        actuate=None,
        mv_range=(-1.0, 1.0),
        expert_mv=None,
        direction=away_from_nearer_limit,
        controller=None,
        applied=None,
        reads=(),
        settled=None,
        angles=(),
        paired=False,
    ):
        self.paired = paired
        env = spec.make_env()
        self.n_ep = int(params.max_steps_in_episode)
        self.dt = getattr(env, "control_period", 1) * float(params.delta_t)
        # Long enough for any retest; continuing off, so a trip leaves the
        # out-of-limit state to read which CV tripped.
        params = params.replace(max_steps_in_episode=1000 * self.n_ep)
        params = (
            params.replace(continuing=False)
            if hasattr(params, "continuing")
            else params
        )
        ctrl = controller or spec.make_pid_controller()
        self.n_mv, self.n_cv, self.n_seeds = (
            env.action_space(params).shape[0],
            len(cvs),
            len(seeds),
        )
        self.limits, self.mv_range, self.direction = list(limits), mv_range, direction
        self.expert_mv, self._actuate = expert_mv or {}, actuate or (
            lambda u, y, m: (u, m)
        )
        self._applied = applied or (lambda u, a: u)
        self._keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
        n = self.n_ep

        def expert(key):
            def body(carry, t):
                obs, state, mem = carry
                a, mem = ctrl.step(ctrl.params, mem, obs, jnp)
                a = jnp.reshape(a, (self.n_mv,))
                obs, state, _, _, info = env.step_env(
                    jax.random.fold_in(key, t), state, a, params
                )
                return (obs, state, mem), (a, mem, info["tripped"])

            return jax.lax.scan(
                body, (*env.reset_env(key, params), ctrl.init_state), jnp.arange(n)
            )

        self._angles = tuple(angles)

        def read(obs, state):
            return jnp.stack(
                [jnp.reshape(f(obs, state, params), ()) for f in (*cvs, *reads)]
            )

        def step(key, state, a):
            obs, state, _, _, info = env.step_env(key, state, a, params)
            return state, read(obs, state), info["tripped"]

        (obs, state, _), (acts, mems, trips) = jax.jit(jax.vmap(expert))(self._keys)
        # The tests start from this hold: one that tripped holds no operating
        # point, and its average reads the excursion as one.
        tripped = np.flatnonzero(np.asarray(trips).any(axis=1))
        if tripped.size:
            raise TrippedDuringTest(
                f"{spec.name}: the settling hold tripped on seeds "
                f"{[int(seeds[i]) for i in tripped]}"
            )
        take = start or (lambda a, m: a[:, -window:].mean(axis=1))
        self.u0 = np.asarray(take(np.asarray(acts), mems), float).reshape(
            len(seeds), -1
        )
        self._y = np.asarray(jax.vmap(read)(obs, state), float)
        if settled is not None:
            self.u0 = np.asarray(settled(self.u0, self._y), float)
        self._states = [
            jax.tree_util.tree_map(lambda x, s=s: x[s], state)
            for s in range(len(seeds))
        ]
        self._step, self._base = jax.jit(step), {}

    def _which(self, y):
        for i, (lo, hi) in enumerate(self.limits):
            if (lo is not None and y[i] <= lo) or (hi is not None and y[i] >= hi):
                return i
        return None

    def _run(self, s, j, step, window, loops, n_pre=None):
        if n_pre is None:
            n_pre = max((lp.n_pre for lp in loops), default=0)
        fold = jax.vmap(lambda t: jax.random.fold_in(self._keys[s], self.n_ep + t))
        keys = np.asarray(fold(jnp.arange(n_pre + window)))
        state, y, u = self._states[s], self._y[s].copy(), self.u0[s].copy()
        sp, mem, Y, U, y_at, trip = (
            {lp.cv: y[lp.cv] for lp in loops},
            None,
            [],
            [],
            y,
            None,
        )
        Y0, U0 = [], []
        for t in range(n_pre + window):
            if t == n_pre:
                y_at = y.copy()
                held = next((lp for lp in loops if lp.mv == j), None)
                if j is not None and held:
                    sp[held.cv] += step / held.kc  # its first move is the step
                elif j is not None:
                    u[j] += step
            for lp in loops:
                u[lp.mv] = self.u0[s][lp.mv] + lp.kc * (sp[lp.cv] - y[lp.cv])
            u = np.clip(u, *self.mv_range)
            a, mem = self._actuate(u, y, mem)
            state, y, tripped = self._step(keys[t], state, jnp.asarray(a, jnp.float32))
            y = np.asarray(y, float)
            if bool(tripped):
                trip = (max(t - n_pre, 0), self._which(y))
                break
            (Y if t >= n_pre else Y0).append(y[: self.n_cv])
            (U if t >= n_pre else U0).append(np.array(self._applied(u, a), float))
        shape = (-1, self.n_cv), (-1, self.n_mv)
        pre = np.reshape(Y0, shape[0]), np.reshape(U0, shape[1])
        Y = np.reshape(Y, shape[0])
        for i in self._angles:
            Y[:, i] = np.unwrap(Y[:, i])
        return Y, np.reshape(U, shape[1]), trip, y_at[: self.n_cv], pre

    def test(self, s, j, step, window, loops) -> Run:
        """MV ``j`` stepped by ``step`` on seed index ``s``, against the
        unstepped run with the same ``loops``, cut at the first trip."""
        if (s, window, loops) not in self._base:
            self._base[s, window, loops] = self._run(s, None, 0.0, window, loops)
        Yb, Ub, tb, y0, _ = self._base[s, window, loops]
        Ys, Us, ts, *_ = self._run(s, j, step, window, loops)
        m = min(len(Yb), len(Ys))
        trips = {k: v for k, v in (("base", tb), ("step", ts)) if v is not None}
        return Run(
            step, window, loops, Ys[:m] - Yb[:m], Yb[:m], Us[:m] - Ub[:m], y0, trips
        )

    def room(self, j) -> float:
        """N3's limit: the longest step every seed can take away from its
        nearer limit."""
        lo, hi = self.mv_range
        return float(np.min(np.maximum(hi - self.u0[:, j], self.u0[:, j] - lo)))

    def bump(self, s, j, cv, size, window, instrument, loops=(), at=None) -> Bump:
        """N1 on seed index ``s``: one run holds for an episode, as long as the
        expert had to settle, since a stale reading can leave it cycling (ours).
        It then records ``instrument``'s baseline, steps MV ``j`` by ``size``
        away from its nearer limit, and reads CV ``cv`` for ``window`` steps,
        once per update. With ``loops`` the test is closed: they hold from the
        start, and a step on a loop's own MV moves its setpoint (N1's loop
        nobody holds).

        The step lands one step before an update, so the first reading after
        it is the first that can show it. ``fit`` takes ``first=instrument.dt``
        on this grid, which a fast reading does not need. ``at=(period, phase)``
        is for a plant with a cycle of its own: the hold runs on until its clock
        reads ``phase`` modulo ``period``, and the response is read against the
        same phase of the cycle before, as an engineer lays one trace on another,
        so the cycle's own swing stays out of it. ``window`` must fit in a cycle."""
        every, now = instrument.every, int(self._states[s].time)
        hold = self.n_ep + (-(now + self.n_ep + 1) % every)
        base_steps = math.ceil(instrument.baseline / every) * every
        if at is not None:
            hold += (at[1] - (now + hold + base_steps)) % at[0]
        n_pre = hold + base_steps
        step = self.direction(self.u0[s][j], size)
        Y, U, trip, _, (Y0, U0) = self._run(s, j, step, window, loops, n_pre)
        base = Y0[hold::every, cv]
        before = np.mean(base) if at is None else Y0[-at[0] :, cv][: len(Y)]
        u_before = np.mean(U0[hold::every, j])
        if self.paired:
            Yr, Ur, _, _, _ = self._run(s, j, 0.0, window, loops, n_pre)
            n = min(len(Y), len(Yr))
            Y, U = Y[:n] - Yr[:n], U[:n] - Ur[:n]
            before, u_before = 0.0, 0.0
        return Bump(
            base,
            Y[::every, cv] - before,
            U[::every, [j]] - u_before,
            step,
            trip,
            now + n_pre,
        )
