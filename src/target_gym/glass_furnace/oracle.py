"""The glass furnace's oracle (the MPC slot): the valve a planner with foresight fires.

The oracle is the problem definition solved (charter J3): the true plant, its true state
and the episode's future, here every walk innovation and event the episode's keys draw,
planned on the task's own reward through the env's own ``step_env`` and
``compute_reward_terms``, so no cost is written twice.

The reward is the fuel bill plus the step's charged pull at the lost glass price when the
canal is under its target or the fining share under its pass mark. That second term is a
step: losing a step costs 416.88 EUR, nine times the step's whole fuel bill. The planner
therefore states it as what it is at any plan worth having, a limit on every step,
fining >= pass and canal >= target, and minimises the fuel bill under them. It also
asks no more than the crown override's cap: a valve asked over the cap gets the cap,
and asking that cap's path itself gets the same valve, the same override integral and
so the same episode, so on the env's step this loses nothing, and it keeps the
override's min out of the problem's derivatives. A plan that
meets them pays exactly the reward's cost, and it is a local minimiser of the reward
itself: a small move either keeps every step and costs more by the KKT conditions, or
loses a step and pays its 416.88 EUR jump for a saving that vanishes with the move. The
limits are planned ``FINING_BACKOFF`` and ``CANAL_BACKOFF`` inside the marks, so a plan
still passes them when the runner replays it step by step.

The furnace runs on after the episode: the tasks are continuing, ended by a time limit
that no agent sees (docs/rl-protocol.md). So the plan does not end at the cut. Its last
knot is held for ``TAIL_HOURS`` more on the walks' mean path, under the same limits, and
the tail's fuel is charged in the plan's objective, though not in the cost the plan
reports, which is the episode's. Without a tail the plan closes the valve on its last
knot, whose fuel first moves the canal's fining 5.4 h later, and spends the fining the
furnace needs after the cut. With a tail whose fuel is free, the plan drops the knot
before the last and buys its fining back with that fuel (about 0.6 point of the room,
measured 2026-11-09).

The planner runs in the env's float32: under x64, ``jax.random.normal`` draws other walk
innovations from the same keys, so a float64 plan would be planned on another episode.

The move is the valve, held over knots of ``knot_hours`` (a control parametrisation;
the env's step is 120 s, so the plan is the best one held over its knots). The solver
is sequential quadratic programming (scipy's SLSQP) on exact Jacobians by forward
differentiation through the episode. Convergence is shown by a certificate the
solver does not grade itself: the KKT stationarity of the plan, from non-negative
multipliers on its active limits and bounds (``kkt``), and its worst violation.

On the registered plant it does not converge (measured 2026-11-09 with the tail, seeds
100 to 104, ``data/oracle_ceilings/glass_furnace.json``): none of 50 starts did. SLSQP
stops at its iteration cap or on a failed line search, with the KKT certificate near 1.
The plant is not smooth in the valve: the fining band has two ends
(``physical.fining_done``), and the refiner's cooling loop clamps at zero, so the
limits' Jacobians jump. The iterates drop the valve for a knot every 12 to 20 h. A
solve's plan is then its cheapest iterate that loses no glass, or its last one raised until it loses none,
whichever costs less. It is replayed and its replayed cost reported, an upper bound on
the ceiling, so the room from it is a lower bound. On three scenario-seeds (pull_down 100
and 102, pull_up 102) it costs more than the reference ladder's kept rung.

With an episode's ``seed`` (``benchmark.run_policy``'s keys for it), ``step`` plays the
plan made with full foresight. Without one it plans on the walks' mean path from the
true state (the innovations unseen, the events known, since the state holds their
times) and plans again every ``replan_hours``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.glass_furnace import env_jax as E
from target_gym.glass_furnace import physical as P

#: The suite was solved on 4 h knots: 2 h knots cost twice and ran no better.
KNOT_HOURS = 4.0
#: Inside the pass mark (fining share) and the canal target (K): the planner's scan and
#: the runner's step-by-step replay round apart in float32, by up to 7e-5 of share and
#: 0.028 K of canal over an episode (measured 2026-11-10 on four kept plans, where a
#: 1e-3 K backoff let the runner lose 1 to 4 steps), so about three and two times that
#: (ours).
FINING_BACKOFF, CANAL_BACKOFF = 2e-4, 0.05
#: Limit rows' scale for the solver: fining shares to tenths, the canal in K, the
#: valve's headroom under the override in hundredths of a kg/s.
FINING_SCALE, CANAL_SCALE, HEADROOM_SCALE = 10.0, 0.1, 100.0
#: The objective's scale for the solver: SLSQP's first quasi-Newton step is the
#: scaled gradient itself, about 1 kg/s per knot unscaled, which slams every knot to a
#: bound; a hundredth makes it about 0.01 kg/s.
OBJECTIVE_SCALE = 0.01
#: A limit or bound within this of its edge counts as active in the certificate.
ACTIVE = 1e-6
#: SLSQP's cap, and its stop on the change in the scaled objective, 1e-6 EUR per step,
#: about float32's resolution of a mean near 46 EUR; convergence is judged by the KKT
#: certificate, not by this.
MAX_ITER, FTOL = 400, 1e-8
#: Stationarity a converged float32 plan reaches (ours, float32 Jacobians).
KKT_TOL = 1e-3
#: Hours the last knot is held past the episode, under the limits, its fuel charged: the
#: fining's dead time (11.8 h, scripts/ladder_rows/glass_furnace.py) plus a knot, with a
#: margin (ours). Without a tail, the plans' end effect spans the last 24 h (measured
#: 2026-11-09). Industrial MPC predicts past its last move until the moves' effects have
#: settled (Qin and Badgwell 2003, p. 747); decision D4 lines out at the horizon's end.
TAIL_HOURS = 24.0


class Plan(NamedTuple):
    valve: np.ndarray  # kg/s asked, one per step
    knots: np.ndarray  # kg/s, one per knot
    cost: float  # EUR per step, the reward's mean cost on the planned rollout
    running: float  # EUR per step, its fuel bill
    tail_running: float  # EUR per tail step, the tail's fuel bill as planned
    lost_steps: int  # steps the planned rollout loses
    status: int  # SLSQP's exit mode, 0 converged
    message: str
    iterations: int
    capped: bool  # the iteration cap was hit
    kkt: float  # stationarity, relative to the gradient
    violation: float  # worst limit violation, in the scaled rows
    active: int  # active limit rows
    how: str  # "final": SLSQP's last iterate; else how a plan losing no glass was had


def rollout(env, params, state, keys, valve):
    """The env's episode from ``state`` under ``valve`` (kg/s per step), on the
    episode's step ``keys``: per step ``(cost, running, fining, canal, headroom)``, EUR,
    the reward's own readings, and kg/s between the override's cap and the valve."""
    return _scan(env, params, state, keys, valve)[1]


def _scan(env, params, state, keys, valve):
    """``rollout``'s readings and the state it ends in."""

    def f(s, x):
        k, u = x
        hot = P.sensors(s.plant, params.grid, params.plant)["hot"]
        cap = s.limit + params.override_gain * (
            params.crown_high - params.override_margin - hot
        )  # the override's cap on this step (env_jax.override)
        _, s1, r, _, _ = env.step_env(k, s, E.to_action(u, params), params)
        read = P.sensors(s1.plant, params.grid, params.plant)
        run = E.compute_reward_terms(s1, params)["running"]
        headroom = jnp.maximum(cap, params.fuel_min) - u
        return s1, (-r, run, read["fining"], read["canal"], headroom)

    return jax.lax.scan(f, state, (keys, valve))


def episode_keys(seed: int, n_steps: int):
    """``(reset key, step keys (n_steps, 2))`` that ``benchmark.run_policy`` runs
    seed ``seed`` on."""
    from target_gym.benchmark import episode_keys as keys

    reset, steps = keys(jax.random.PRNGKey(seed), 1, n_steps)
    return reset[0], steps[:, 0]


def kkt(grad, rows, jac, lo_gap, hi_gap):
    """Relative stationarity ``|grad - J_a' lam - mu_lo + mu_hi|_inf / |grad|_inf``
    with ``lam, mu >= 0`` on the active rows and bounds, by non-negative least
    squares, and the number of active rows."""
    from scipy.optimize import nnls

    a = rows <= ACTIVE
    n = grad.size
    cols = [jac[a].T, np.eye(n)[:, lo_gap <= ACTIVE], -np.eye(n)[:, hi_gap <= ACTIVE]]
    M = np.concatenate(cols, axis=1)
    if M.shape[1] == 0:
        res = grad
    else:
        x, _ = nnls(M, grad, maxiter=50 * M.shape[1])
        res = grad - M @ x
    scale = max(float(np.max(np.abs(grad))), 1e-12)
    return float(np.max(np.abs(res)) / scale), int(a.sum())


def solve(
    env, params, state, keys, knot_steps: int, start=None, max_iter=MAX_ITER, log=None,
    tail_hours=TAIL_HOURS, charge_tail=True,
):  # fmt: skip
    """The plan from ``state`` over the steps ``keys`` cover, one valve per
    ``knot_steps``, its last knot held ``tail_hours`` past them on the walks' mean path,
    under the limits, the tail's fuel charged in the objective unless ``charge_tail`` is
    False; ``start`` the knots to start from (the state's valve if None);
    ``log(iteration, knots, objective, worst row)`` called after each iteration if
    given. The objective is the fuel bill over the episode and the tail, per episode
    step; the plan's ``cost`` is the episode's alone.
    """
    from scipy.optimize import minimize

    n = int(keys.shape[0])
    k = -(-n // knot_steps)
    p, st = params, state
    lo, hi = float(p.fuel_min), float(p.fuel_max)
    n_tail = int(round(tail_hours * 3600.0 / float(p.delta_t)))
    tail_p = p.replace(walk_sd=(0.0,) * len(E.WALKS))
    tail_keys = jax.random.split(jax.random.PRNGKey(0), n_tail)

    def valve(z):
        return jnp.repeat(z, knot_steps)[:n]

    def outs(z):
        end, (cost, run, fining, canal, headroom) = _scan(env, p, st, keys, valve(z))
        run_t = jnp.zeros(1)
        if n_tail:
            _, (_, run_t, f_t, c_t, h_t) = _scan(
                env, tail_p, end, tail_keys, jnp.full(n_tail, z[-1])
            )
            fining, canal, headroom = (jnp.concatenate(v) for v in
                                       ((fining, f_t), (canal, c_t), (headroom, h_t)))  # fmt: skip
        rows = jnp.concatenate([
            FINING_SCALE * (fining - p.fining_pass - FINING_BACKOFF),
            CANAL_SCALE * (canal - p.canal_target - CANAL_BACKOFF),
            HEADROOM_SCALE * headroom,
        ])  # fmt: skip
        lost = (fining[:n] < p.fining_pass) | (canal[:n] < p.canal_target)
        spent = jnp.sum(run) + (jnp.sum(run_t) if charge_tail else 0.0)
        return (
            spent / n,
            rows,
            jnp.mean(cost),
            jnp.sum(lost),
            jnp.mean(run),
            jnp.mean(run_t),
        )

    value = jax.jit(outs)
    jac = jax.jit(jax.jacfwd(lambda z: outs(z)[:2]))
    cache: dict = {}

    def at(z, deriv):
        key = (z.tobytes(), deriv)
        if key not in cache:
            zz = jnp.asarray(z)
            cache[key] = [np.asarray(v, float) for v in
                          (jac(zz) if deriv else value(zz)[:2])]  # fmt: skip
        return cache[key]

    z0 = np.full(k, float(state.fuel)) if start is None else np.asarray(start)
    count = [0]

    best = {}

    def keeps(rows):  # no step of the episode or the tail loses glass
        return bool(np.all(rows[: 2 * (n + n_tail)] >= 0.0))

    def progress(z):
        count[0] += 1
        f, rows = at(z, False)
        if keeps(rows) and f < best.get("f", np.inf):
            best.update(f=f, z=np.clip(z, lo, hi))
        if log is not None:
            log(count[0], z, float(f), float(rows.min()))

    res = minimize(
        lambda z: OBJECTIVE_SCALE * at(z, False)[0], np.clip(z0, lo, hi),
        method="SLSQP", jac=lambda z: OBJECTIVE_SCALE * at(z, True)[0],
        bounds=[(lo, hi)] * k,
        constraints=[{"type": "ineq", "fun": lambda z: at(z, False)[1],
                      "jac": lambda z: at(z, True)[1]}],
        options={"maxiter": max_iter, "ftol": FTOL}, callback=progress,
    )  # fmt: skip
    z, how = np.clip(res.x, lo, hi), "final"
    if not keeps(at(z, False)[1]):
        # The last iterate loses glass. Repair it: the least raise of every knot that
        # loses none, by bisection, against the best iterate that lost none.
        def raised(d):
            return np.clip(z + d, lo, hi)

        a, b = 0.0, hi - lo
        if keeps(at(raised(b), False)[1]):
            for _ in range(30):
                m = 0.5 * (a + b)
                a, b = (a, m) if keeps(at(raised(m), False)[1]) else (m, b)
            if at(raised(b), False)[0] < best.get("f", np.inf):
                best.update(f=at(raised(b), False)[0], z=raised(b))
                how = f"final iterate raised {b:.2e} kg/s"
            else:
                how = "best iterate that lost no glass"
        elif "z" in best:
            how = "best iterate that lost no glass"
        z = best.get("z", z)
    _, rows, cost, lost, run, run_t = (
        np.asarray(v, float) for v in value(jnp.asarray(z))
    )
    g, J = at(z, True)
    stat, active = kkt(g, rows, J, z - lo, hi - z)
    return Plan(
        valve=np.asarray(valve(jnp.asarray(z))), knots=z, cost=float(cost),
        running=float(run), tail_running=float(run_t), lost_steps=int(lost), status=int(res.status),
        message=str(res.message), iterations=int(res.nit),
        capped=int(res.nit) >= max_iter, kkt=stat,
        violation=float(max(0.0, -rows.min())), active=active, how=how,
    )  # fmt: skip


class GlassOracle:
    """The registry's oracle: ``step(obs, state)`` reads the true state."""

    oracle = True

    def __init__(
        self, env, params, seed=None, knot_hours=KNOT_HOURS, replan_hours=24.0
    ):
        self.env, self.params, self.seed = env, params, seed
        hour = 3600.0 / float(params.delta_t)
        self.knot = max(1, int(round(knot_hours * hour)))
        self.replan = max(1, int(round(replan_hours * hour)))
        self.plans: list[Plan] = []
        self.reset()

    def reset(self):
        self._valve, self._t0 = None, 0

    def _plan(self, state, t: int):
        n = int(self.params.max_steps_in_episode)
        if self.seed is not None:
            p, keys = self.params, episode_keys(self.seed, n)[1][t:]
        else:  # the walks' mean path: no innovation, the events where the state has them
            p = self.params.replace(walk_sd=(0.0,) * len(E.WALKS))
            keys = jax.random.split(jax.random.PRNGKey(0), n - t)
        plan = solve(self.env, p, state, keys, self.knot)
        self.plans.append(plan)
        self._valve, self._t0 = plan.valve, t

    def step(self, _obs, state):
        t = int(state.time)
        stale = self._valve is None or t < self._t0 or t - self._t0 >= len(self._valve)
        if stale or (self.seed is None and t - self._t0 >= self.replan):
            self._plan(state, t)
        u = self._valve[t - self._t0]
        return float(np.clip(np.asarray(E.to_action(u, self.params))[0], -1.0, 1.0))


def make_glass_furnace_mpc(env, params, seed=None, **kwargs):
    """The oracle for one episode: with ``seed``, full foresight of it."""
    return GlassOracle(env, params, seed=seed, **kwargs)
