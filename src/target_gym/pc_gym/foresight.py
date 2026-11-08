"""The process family's ceiling (charter Words "Ceiling", J3): one episode's
whole plan, solved with the true model, the true state and the future.

The problem is the task's own: the cost is ``env.step_env``'s reward, summed
over the episode, on the seed's own reset and step keys
(``benchmark.episode_keys``), so the plan knows every disturbance the episode
will meet, its events, schedules and noise draws, and nothing is restated by
hand. The planned cost (``plan_cost``) is the env's, in its own float32; a
plan is replayed through ``benchmark.run_policy`` (``PlanPolicy``), the loop
every reference runs in, and only that replayed cost is reported.

The solver is L-BFGS-B on the raw moves in their box [-1, 1], through JAX's
gradient of the episode's cost, in the env's own float32: the plan the
solver sees is the plan the env replays, also on an unstable plant, where a
float64 plan drifts from its float32 replay. The reward's zone and trip terms
are steps, whose gradient is zero, so each task's ``ceiling_margins(state,
params)`` names the quantities those steps switch on (the permit's pH, the
products' purities, the header zone, the trip variable), in units of a
resolution, each backed off a little, with a price; a margin below 0 is
charged ``price x rho x margin^2`` per step while the plan is solved, in place
of the step terms (``CEILING_TERMS`` names the terms kept), with rho raised a
hundredfold until the plan meets every margin. On a plan that meets them the
step terms and the added charge are both 0, so there the solver's objective
is the env's cost exactly. The plan that applies is the stage's the env
charges least, never above the start's.
"""

from __future__ import annotations

import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from target_gym.benchmark import episode_keys, episodes_of, run_policy, task_info

#: rho's stages, on the task's price.
RHO = (1.0, 100.0, 1e4)


def _episode(env, params, seed: int):
    """The seed's reset state and step keys, as ``run_policy`` draws them."""
    T = int(params.max_steps_in_episode)
    reset_keys, step_keys = episode_keys(jax.random.PRNGKey(seed), 1, T)
    _, s0 = env.reset_env(reset_keys[0], params)
    return s0, step_keys[:, 0]


def _rollout(env, params, s0, keys, plan, margins: Callable | None):
    """Per step: the env's cost, its trip flag and the margins (or 0)."""

    def f(s, ka):
        k, a = ka
        _, s1, r, _, info = env.step_env(k, s, a, params)
        s1 = jax.tree_util.tree_map(lambda x, y: x.astype(y.dtype), s1, s)
        m = jnp.zeros(1) if margins is None else margins(s1, params)[0]
        return s1, (-r, info["tripped"], m)

    return jax.lax.scan(f, s0, (keys, plan))[1]


class _Problem:
    """The jitted pieces of one task's planning problem, compiled once per
    plan length: the env's summary of a plan (mean cost, trip, smallest
    margin) and the solver's objective with its gradient, both from a given
    state on given keys."""

    def __init__(self, env, params, margins: Callable | None, terms: tuple = ()):
        # Planned on the proposal: a trip ends the episode there instead of
        # restarting, so the margins see the state that tripped. A plan with
        # no trip is charged the same either way.
        planning = params.replace(continuing=False)

        def summary(z, s0, keys):
            cost, trip, m = _rollout(env, params, s0, keys, z, margins)
            return jnp.mean(cost), jnp.any(trip), jnp.min(m)

        def objective(z, weight, s0, keys):
            def f(s, ka):
                k, a = ka
                _, s1, _, _, _ = env.step_env(k, s, a, planning)
                s1 = jax.tree_util.tree_map(lambda x, y: x.astype(y.dtype), s1, s)
                named = env.reward_terms(s1, planning)
                return s1, (sum(named[t] for t in terms), margins(s1, planning)[0])

            cost, m = jax.lax.scan(f, s0, (keys, z))[1]
            price = margins(s0, params)[1]
            return jnp.sum(cost) + weight * price * jnp.sum(jnp.minimum(m, 0.0) ** 2)

        def advance(s0, keys, z):
            def f(s, ka):
                k, a = ka
                _, s1, _, _, _ = env.step_env(k, s, a, params)
                return (
                    jax.tree_util.tree_map(lambda x, y: x.astype(y.dtype), s1, s),
                    None,
                )

            return jax.lax.scan(f, s0, (keys, z))[0]

        self.summary = jax.jit(summary)
        self.grad = None if margins is None else jax.jit(jax.value_and_grad(objective))
        self.advance = jax.jit(advance)


def plan_cost(env, params, seed: int, plan) -> float:
    """The mean cost per step the env charges ``plan`` on ``seed``: the
    oracle's planned cost."""
    s0, keys = _episode(env, params, seed)
    cost = _Problem(env, params, None).summary(jnp.asarray(plan, jnp.float32), s0, keys)
    return float(cost[0])


def _window(problem, s0, keys, init, maxiter, rho):
    """L-BFGS-B by rho's stages on one stretch of plan from ``s0``."""
    init = np.clip(np.asarray(init, float), -1.0, 1.0)
    shape = init.shape

    def summary(z):
        cost, trip, m = problem.summary(jnp.asarray(z, jnp.float32), s0, keys)
        return float(cost), bool(trip), float(m)

    start_cost, start_trip, start_m = summary(init)
    scale = max(abs(start_cost) * shape[0], 1.0)
    stages, z = [], init
    for weight in rho:

        def fun(flat, weight=weight):
            z32 = jnp.asarray(flat.reshape(shape), jnp.float32)
            v, g = problem.grad(z32, weight, s0, keys)
            return float(v) / scale, np.asarray(g, float).ravel() / scale

        res = minimize(
            fun, z.ravel(), jac=True, method="L-BFGS-B",
            bounds=[(-1.0, 1.0)] * z.size,
            options=dict(maxiter=maxiter, maxfun=4 * maxiter, ftol=1e-10,
                         gtol=1e-8, maxcor=30),
        )  # fmt: skip
        z = res.x.reshape(shape)
        _, g = fun(res.x)
        pg = np.max(np.abs(np.clip(res.x - g, -1.0, 1.0) - res.x))
        cost, trip, m = summary(z)
        stages.append(
            dict(
                plan=z, rho=weight, status=int(res.status),
                message=str(res.message), projected_gradient=float(pg),
                iterations=int(res.nit), capped=int(res.nit) >= maxiter,
                planned_cost=cost, min_margin=m, tripped=trip,
            )
        )  # fmt: skip
        if m >= 0.0 and not trip:
            break
    untripped = [s for s in stages if not s["tripped"]] or stages
    best = min(untripped, key=lambda s: s["planned_cost"])
    cert = {k: v for k, v in best.items() if k != "plan"}
    plan = best["plan"]
    cert["fell_back"] = (cert["tripped"] and not start_trip) or (
        cert["tripped"] == start_trip and cert["planned_cost"] > start_cost
    )
    if cert["fell_back"]:  # never above the start
        plan = init
        cert.update(planned_cost=start_cost, min_margin=start_m, tripped=start_trip)
    cert.update(
        start_cost=start_cost,
        total_iterations=sum(s["iterations"] for s in stages),
        stages=len(stages),
    )
    return plan, cert


def solve(
    env,
    params,
    seed: int,
    margins: Callable,
    terms: tuple,
    init,
    maxiter: int = 2000,
    rho=RHO,
    window: tuple[int, int] | None = None,
):
    """The ceiling's plan for ``seed`` from the plan ``init`` ``(T, nu)``:
    ``(plan, certificate)``. The objective is the env's reward terms named in
    ``terms`` (``env.reward_terms``), the smooth ones, plus the margins'
    charge; the step terms left out are 0 on a plan that meets every margin.

    Each stage of rho runs L-BFGS-B from the last stage's plan; the stages stop
    once a plan meets every margin with no trip. The plan that applies is the
    stage's with the lowest cost the env charges (no trip), never above the
    start's. Its certificate: L-BFGS-B's ``status`` (0 converged, 1 the
    iteration cap, 2 a line search that found no decrease, at float32's
    resolution of the objective) and ``message``, the ``projected_gradient``'s
    largest entry on the objective over the start's total cost, ``iterations``
    and ``capped``; the plan's ``planned_cost`` (the env's mean per step), its
    smallest margin ``min_margin`` and whether it ``tripped``; and
    ``fell_back`` when no stage beat the start.

    ``window = (H, C)`` solves the episode as overlapping stretches instead:
    H steps from the state the plan so far reaches, keeping the first C, so
    no stretch is planned open loop over more than H steps. For an unstable
    plant, whose float32 rounding grows along an open-loop plan, so that a
    whole-episode plan's cost is noise to its gradient. Each stretch still
    sees the future over its H steps; its certificate is the worst stretch's
    (status, projected gradient, cap), with the count of stretches."""
    t0 = time.perf_counter()
    s0, keys = _episode(env, params, seed)
    problem = _Problem(env, params, margins, terms)
    init = np.clip(np.asarray(init, float), -1.0, 1.0)
    if window is None:
        plan, cert = _window(problem, s0, keys, init, maxiter, rho)
        cert["seconds"] = time.perf_counter() - t0
        return plan, cert
    H, C = window
    T = init.shape[0]
    plan, state, certs = init.copy(), s0, []
    for k in range(0, T, C):
        end = min(k + H, T)
        part, cert = _window(problem, state, keys[k:end], plan[k:end], maxiter, rho)
        plan[k:end] = part
        certs.append(cert)
        keep = min(C, end - k)
        state = problem.advance(
            state, keys[k : k + keep], jnp.asarray(part[:keep], jnp.float32)
        )
    whole = problem.summary(jnp.asarray(plan, jnp.float32), s0, keys)
    cost, trip, m = float(whole[0]), bool(whole[1]), float(whole[2])
    start = problem.summary(jnp.asarray(init, jnp.float32), s0, keys)
    start_cost, start_trip = float(start[0]), bool(start[1])
    fell_back = (trip and not start_trip) or (trip == start_trip and cost > start_cost)
    if fell_back:
        plan = init
        cost, trip, m = start_cost, start_trip, float(start[2])
    worst = max(certs, key=lambda c: c["projected_gradient"])
    cert = {
        "status": max(c["status"] for c in certs),
        "statuses": sorted({c["status"] for c in certs}),
        "message": worst["message"],
        "projected_gradient": worst["projected_gradient"],
        "iterations": max(c["iterations"] for c in certs),
        "capped": any(c["capped"] for c in certs),
        "capped_windows": sum(c["capped"] for c in certs),
        "rho": max(c["rho"] for c in certs),
        "planned_cost": cost, "min_margin": m, "tripped": trip,
        "fell_back": fell_back, "start_cost": start_cost,
        "total_iterations": sum(c["total_iterations"] for c in certs),
        "stages": max(c["stages"] for c in certs),
        "windows": len(certs), "window": [H, C],
        "seconds": time.perf_counter() - t0,
    }  # fmt: skip
    return plan, cert


class PlanPolicy:
    """Replays fixed plans ``(n, T, nu)``, one per episode, open loop."""

    name = "ceiling"
    oracle = False

    def __init__(self, plans):
        self.plans = np.asarray(plans, float)

    def reset(self, key, obs, task):
        return None

    def act(self, carry, obs, t):
        return self.plans[:, t], carry


def tracking(env, params, seed: int, plan, feedback) -> dict:
    """What ``TrackingPolicy`` needs to hold one plan's path: along the
    planned path, the states ``x`` (``get`` of each state before its step),
    and the first-order effect of the state and of the move on the pinned
    output after the step, ``dy_dx`` and ``dy_du`` (one move).

    ``feedback = (get, put, i)``: ``get(state)`` the plant's state vector,
    ``put(state, x)`` the state with that vector, ``i`` the pinned entry."""
    get, put, i = feedback
    s0, keys = _episode(env, params, seed)
    plan = jnp.asarray(plan, jnp.float32)

    def f(s, ka):
        k, a = ka
        _, s1, _, _, _ = env.step_env(k, s, a, params)
        return jax.tree_util.tree_map(lambda x, y: x.astype(y.dtype), s1, s), s

    states = jax.lax.scan(f, s0, (keys, plan))[1]

    def pinned(x, u, s, k):
        return get(env.step_env(k, put(s, x), u, params)[1])[i]

    xs = jax.vmap(get)(states)
    dx, du = jax.jit(jax.vmap(jax.grad(pinned, argnums=(0, 1))))(xs, plan, states, keys)
    return {"x": np.asarray(xs), "dy_dx": np.asarray(dx), "dy_du": np.asarray(du)}


class TrackingPolicy:
    """Plans held on their path by the true state (an oracle row): the planned
    move plus the one that cancels, to first order, the pinned output's
    deviation after the step, ``-dy_dx . (x - x_plan) / dy_du``.

    On an open-loop unstable plant a plan's path is not reproducible: the
    benchmark's loop batches the episodes, whose float32 rounding differs in
    the last bit from the planner's, and the plant grows it. This moves only
    by that rounding, so the plan's cost is what the env charges."""

    name = "ceiling"
    oracle = True

    def __init__(self, plans, paths, get):
        self.plans = np.asarray(plans, float)  # (n, T, 1)
        self.x = np.stack([p["x"] for p in paths])  # (n, T, nx)
        self.dx = np.stack([p["dy_dx"] for p in paths])
        du = np.stack([p["dy_du"] for p in paths])[..., 0]  # (n, T)
        self.du = np.where(np.abs(du) > 1e-12, du, np.inf)
        self.get = jax.jit(jax.vmap(get))

    def reset(self, key, obs, task):
        return None

    def act(self, carry, obs, t, state=None):
        x = np.asarray(self.get(state), float)
        dy = np.sum(self.dx[:, t] * (x - self.x[:, t]), axis=-1)
        u = self.plans[:, t, 0] - dy / self.du[:, t]
        return np.clip(u, -1.0, 1.0)[:, None], carry


def replay(name: str, env, params, seeds, plans, policy=None) -> dict:
    """``plans`` on ``seeds`` through ``run_policy`` (open loop, or by
    ``policy``): each seed's mean cost per step and whether it tripped, as
    the references are scored."""
    rollout = run_policy(
        policy or PlanPolicy(plans), env, params,
        task_info(name, env, params, len(seeds)), seeds=list(seeds),
    )  # fmt: skip
    eps = episodes_of(rollout)
    return {
        "costs": [float(e.cost.mean()) for e in eps],
        "trips": [bool(np.any(e.failed)) for e in eps],
    }
