"""Bump tests, step-response fits and their validity, shared by the protocols.

The expert protocols (docs/proposals/expert-protocols-process.md, C1 and C2)
and the oracle horizon protocol (docs/proposals/oracle-horizon-protocol.md,
H7 to H9 and, from Amendment 1, H14 to H19) identify a plant the same way:

- ``Plant``: the shipped expert runs one episode from each training seed's
  reset, then the moves are held at the mean of its last commands (amended
  H8); one MV is stepped and every CV read against an unstepped run on the
  same noise, up to the first trip of either run (H9);
- both structures, FOPDT and integrating with lag, are fitted to every
  response by output-error least squares on the recorded input (H11), and
  judged by V1 to V4 (H14);
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
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

__all__ = ["MODELS", "TRAIN_SEEDS", "Loop", "Plant", "Run", "class_of", "evidence"]
__all__ += ["fit", "identify", "insignificant", "judge", "noise_sigma", "nrmse_fit"]

#: The training seeds every protocol identifies on; 200-209 are held out.
TRAIN_SEEDS = tuple(range(100, 110))
MODELS = ("fopdt", "integrating_lag")
SNR_MIN, FIT_MIN = 10.0, 80.0  # V1 in noise bands (D8), V4 in percent (D10)
RETESTS, GROWTH = 2, 4  # D11: two retests, step and window up to 4 times
_LAG = {"fopdt": "tau", "integrating_lag": "tau2"}


# ---------------------------------------------------------------------------
# Fits (H11) and their checks (H14)
# ---------------------------------------------------------------------------


def _unit_step(model, q, t):
    k, tau, theta = q
    x = np.maximum(t - theta, 0.0)
    if model == "fopdt":
        return k * (1 - np.exp(-x / tau))
    return k * (x - tau * (1 - np.exp(-x / tau)))


def _simulate(model, Q, dt, U):
    """The output at dt, 2 dt, ... of one ``model`` per column of ``U``, each
    input held over its step: a sum of unit steps at the input's changes."""
    lags = dt * np.arange(1, len(U) + 1)
    moves = np.diff(U, axis=0, prepend=0.0)
    out = [
        fftconvolve(moves[:, i], _unit_step(model, q, lags))[: len(U)]
        for i, q in enumerate(Q)
    ]
    return np.sum(out, axis=0)


def nrmse_fit(y, yhat) -> float:
    """The NRMSE fit in percent, 100 (1 - |y - yhat| / |y - mean y|) (MWC)."""
    spread = np.linalg.norm(y - np.mean(y))
    return float(100 * (1 - np.linalg.norm(y - yhat) / spread)) if spread else -math.inf


def noise_sigma(y) -> float:
    """sigma of a CV in the unstepped run, about its least-squares straight
    line (D9: a held MV leaves a drift on an integrating CV, not noise)."""
    t = np.arange(len(y), dtype=float)
    if len(y) < 3:
        return 0.0
    return float(np.std(y - np.polyval(np.polyfit(t, y, 1), t)))


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


def fit(model: str, dt: float, dy, U) -> dict:
    """Output-error least squares of ``model`` (one per column of ``U``, the
    recorded input deviations, the pair's own MV first) to the response
    ``dy``, with V2 to V4 of H14 and the 3 sigma half-width of the gain."""
    dy = np.asarray(dy, float)
    U = np.asarray(U, float).reshape(len(dy), -1)
    t, n = dt * np.arange(1, len(dy) + 1), U.shape[1]
    ref = U[np.argmax(np.abs(U[:, 0])), 0] or 1.0
    tau_hi = np.inf if model == "fopdt" else t[-1]
    lo, hi = [-np.inf, 1e-9, 0.0] * n, [np.inf, tau_hi, t[-1]] * n
    best = None
    for g in _guesses(model, t, dy, ref):
        x0 = np.clip(list(g) + [0.0, g[1], 0.0] * (n - 1), lo, hi)
        r = least_squares(
            lambda q: _simulate(model, q.reshape(n, 3), dt, U) - dy,
            x0,
            bounds=(lo, hi),
            x_scale="jac",
        )
        best = r if best is None or r.cost < best.cost else best
    Q, sse = best.x.reshape(n, 3), 2 * best.cost
    cov = np.linalg.pinv(best.jac.T @ best.jac) * sse / max(len(dy) - 3 * n, 1)
    lag = _LAG[model]
    out = {"model": model, "k": Q[0, 0], lag: Q[0, 1], "theta": Q[0, 2]}
    if n > 1:
        out["others"] = [dict(zip(("k", lag, "theta"), map(float, q))) for q in Q[1:]]
    at_bound = np.isclose(Q[:, 2], t[-1], rtol=1e-6) | (
        np.isclose(Q[:, 1], tau_hi, rtol=1e-6)
    )
    score = nrmse_fit(dy, dy + best.fun)
    return {
        **{k: (v if isinstance(v, (str, list)) else float(v)) for k, v in out.items()},
        "sse": float(sse),
        "fit": score,
        "k_3sigma": float(3 * math.sqrt(max(cov[0, 0], 0.0))),
        "V2": bool(np.all(5 * Q[:, 1] + Q[:, 2] <= t[-1])),
        "V3": not bool(np.any(at_bound)),
        "V4": score >= FIT_MIN,
    }


def judge(dy, base, U, dt: float) -> dict:
    """One CV's response in one test: sigma, SNR and V1, and both fits."""
    dy, sigma = np.asarray(dy, float), noise_sigma(np.asarray(base, float))
    peak = float(np.max(np.abs(dy))) if len(dy) else 0.0
    snr = peak / (6 * sigma) if sigma > 0 else (math.inf if peak > 0 else 0.0)
    out = {"samples": len(dy), "sigma": sigma, "snr": snr, "V1": snr >= SNR_MIN}
    out["zero"] = len(dy) >= 4 and peak == 0.0
    usable = len(dy) >= 4 and peak > 0
    out["fits"] = {m: fit(m, dt, dy, U) for m in MODELS} if usable else {}
    return out


def _fails(a, model) -> set:
    """The checks of H14 that ``model``'s fit of response ``a`` fails."""
    if model not in a["fits"]:
        return {"zero" if a["zero"] else "short"}
    f = a["fits"][model]
    return {v for v in ("V2", "V3", "V4") if not f[v]} | (set() if a["V1"] else {"V1"})


def passes(a, model=None) -> bool:
    """H14 for one structure, or for either when ``model`` is None."""
    return any(not _fails(a, m) for m in ([model] if model else MODELS))


def insignificant(a) -> bool:
    """H16: identically zero, inside one noise band, or a gain whose 3 sigma
    interval holds zero (on the closer of the two fits)."""
    if not a["fits"]:
        return a["zero"]
    f = min(a["fits"].values(), key=lambda f: f["sse"])
    return a["snr"] < 1 or abs(f["k"]) <= f["k_3sigma"]


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
# Tests and retests (H8, H9, H15, H18), and the labels (H16, H17, H19)
# ---------------------------------------------------------------------------


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
    A = [
        judge(run.dy[:, i], run.base[:, i], run.U[:, moving], plant.dt)
        for i in range(plant.n_cv)
    ]
    return run, A


def _closer(a):
    """A failing pair's failed checks: those of its closer structure (fewer
    failures, then the lower squared error). Ours."""
    if not a["fits"]:
        return _fails(a, MODELS[0])
    return min((_fails(a, m) for m in MODELS), key=len)


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
        room = min(abs(run.y0[i] - b) for b in bounds) - 3 * A[i]["sigma"]
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
                range(plant.n_mv),
                key=lambda m: max(a[c]["snr"] for _, a in trials[m, s]),
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


def identify(plant, sizes):
    """H15 to H19 on every MV and seed of ``plant``: ``(pairs, classes)``,
    pairs keyed ``(mv, cv, seed index)`` with their tests, kept test, label,
    fit and failed checks; classes keyed by CV with the vote."""
    trials = {}
    for j in range(plant.n_mv):
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
            kept = next((n for n, a in enumerate(seen) if passes(a)), len(seen) - 1)
            gone = not passes(seen[kept]) and insignificant(seen[-1])
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


def away_from_nearer_limit(u: float, size: float) -> float:
    """C2's direction for a raw MV in [-1, 1]: down from the upper half."""
    return -size if u > 0 else size


class Plant:
    """A task at steady operation on each training seed (amended H8), ready to
    be bump-tested one MV at a time.

    ``cvs`` are ``f(obs, state, params)`` and ``limits`` their trip limits
    ``(lo, hi)``, either may be None. The expert settles one episode; the
    moves then hold ``start(actions, memories)``, by default the mean of its
    last ``window`` commands. ``actuate(u, y, memory) -> (action, memory)``
    maps the test's MVs to the env's action (by default the raw action).
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
    ):
        self.n_ep, self.dt = int(params.max_steps_in_episode), float(params.delta_t)
        # Long enough for any retest; continuing off, so a trip leaves the
        # out-of-limit state to read which CV tripped.
        params = params.replace(max_steps_in_episode=1000 * self.n_ep)
        params = (
            params.replace(continuing=False)
            if hasattr(params, "continuing")
            else params
        )
        env, ctrl = spec.make_env(), spec.make_pid_controller()
        self.n_mv, self.n_cv, self.n_seeds = (
            env.action_space(params).shape[0],
            len(cvs),
            len(seeds),
        )
        self.limits, self.mv_range, self.direction = list(limits), mv_range, direction
        self.expert_mv, self._actuate = expert_mv or {}, actuate or (
            lambda u, y, m: (u, m)
        )
        self._keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
        n = self.n_ep

        def expert(key):
            def body(carry, t):
                obs, state, mem = carry
                a, mem = ctrl.step(ctrl.params, mem, obs, jnp)
                a = jnp.reshape(a, (self.n_mv,))
                obs, state, *_ = env.step_env(
                    jax.random.fold_in(key, t), state, a, params
                )
                return (obs, state, mem), (a, mem)

            return jax.lax.scan(
                body, (*env.reset_env(key, params), ctrl.init_state), jnp.arange(n)
            )

        def read(obs, state):
            return jnp.stack([jnp.reshape(f(obs, state, params), ()) for f in cvs])

        def step(key, state, a):
            obs, state, _, _, info = env.step_env(key, state, a, params)
            return state, read(obs, state), info["tripped"]

        (obs, state, _), (acts, mems) = jax.jit(jax.vmap(expert))(self._keys)
        take = start or (lambda a, m: a[:, -window:].mean(axis=1))
        self.u0 = np.asarray(take(np.asarray(acts), mems), float).reshape(
            len(seeds), -1
        )
        self._y = np.asarray(jax.vmap(read)(obs, state), float)
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

    def _run(self, s, j, step, window, loops):
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
            if t >= n_pre:
                Y.append(y)
                U.append(u.copy())
        shape = (-1, self.n_cv), (-1, self.n_mv)
        return np.reshape(Y, shape[0]), np.reshape(U, shape[1]), trip, y_at

    def test(self, s, j, step, window, loops) -> Run:
        """MV ``j`` stepped by ``step`` on seed index ``s``, against the
        unstepped run with the same ``loops``, cut at the first trip."""
        if (s, window, loops) not in self._base:
            self._base[s, window, loops] = self._run(s, None, 0.0, window, loops)
        Yb, Ub, tb, y0 = self._base[s, window, loops]
        Ys, Us, ts, _ = self._run(s, j, step, window, loops)
        m = min(len(Yb), len(Ys))
        trips = {k: v for k, v in (("base", tb), ("step", ts)) if v is not None}
        return Run(
            step, window, loops, Ys[:m] - Yb[:m], Yb[:m], Us[:m] - Ub[:m], y0, trips
        )
