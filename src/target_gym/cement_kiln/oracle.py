"""The oracle (the MPC slot) for the cement kiln task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.4), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's own step, ``advance``, with the raw moves' clips removed
  and the moves bounded instead (T1), run in float64, which keeps the
  Arrhenius terms' derivatives finite; the raw-meal feed on its noise-free
  mean path, updated before the step as the env does;
- objective: the free-lime tracking term as (e / e_floor)^2, exact since
  e_tol = 0 and p = 2, and the running cost max(fuel - c_hold, 0) through
  epigraph rows on every visited state (T2);
- limits, elastic at the trip cost per unit (T3) and backed off by the
  noise's reach (K11, in the widths file): each cell below T_bz_max; the
  burning zone above T_bz_min, its max over the cells a conservative soft max
  (K2, K3; K5's width, in the widths file); and the discharge's free lime
  below lime_max;
- dropped guards (T4): the speed, gas-march and temperature guards, and the
  clip of the two fractions after the step: ``calcination_step`` keeps alpha a
  fraction by construction, and lime stays inside [0, 1] below the trip.

The trip's reset, which ``step_env`` evaluates on every step (the kiln's
4000-step profile march), drops out of a feasible plan with the trip. The
cold start holds the fuel and the speed where they are.
"""

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from target_gym.cement_kiln.env import FEED_OU_THETA, advance
from target_gym.oracles.controller import Oracle, TaskModel, raw_of
from target_gym.oracles.shooting import Hinge, Limit
from target_gym.reward import trip_cost


def soft_max(values, width):
    """K2's max over a set, shifted by K3 so that it never overstates: the
    exact max at ``width`` = 0, with the width guarded (JAX FAQ)."""
    w = jnp.where(width > 0.0, width, 1.0)
    smooth = w * (logsumexp(values / w) - jnp.log(values.size))
    return jnp.where(width > 0.0, smooth, jnp.max(values))


def kiln_model(env, p) -> TaskModel:
    assert float(p.e_tol) == 0.0 and float(p.tracking_exponent) == 2.0
    n = int(p.n_zones)
    decay = 1.0 - FEED_OU_THETA * p.delta_t  # the deviation's mean per step

    def step(d, x, u, t):
        target, deviation = d["task"]
        fuel = p.fuel_min + 0.5 * (u[0] + 1.0) * (p.fuel_max - p.fuel_min)  # T1
        rpm = p.rpm_min + 0.5 * (u[1] + 1.0) * (p.rpm_max - p.rpm_min)
        meal = p.raw_meal_nominal + deviation * decay ** (t + 1)
        x1 = advance(x, (fuel, rpm), meal, p, env.integration_method)
        lime, T_solid = x1[-1], x1[:n]
        cost = ((lime - target) / p.e_floor) ** 2
        y = jnp.concatenate([
            jnp.stack([p.running_weight * (fuel - p.c_hold) / p.c_hold]), T_solid,
            jnp.stack([soft_max(T_solid, d["widths"][0]), lime]),
        ])  # fmt: skip
        return x1, cost, y, lime[None], ()

    price = float(trip_cost(p))
    return TaskModel(
        nx=4 * n, nu=2, step=step,
        state_x=lambda s: jnp.concatenate([s.T_solid, s.T_wall, s.alpha, s.lime]),
        data=lambda s: (s.target_lime, s.raw_meal - p.raw_meal_nominal),
        hold=lambda s: jnp.stack([
            raw_of(s.fuel, p.fuel_min, p.fuel_max),
            raw_of(s.rpm, p.rpm_min, p.rpm_max),
        ]),
        tracked=lambda s: s.lime[-1:],
        resolution=(float(p.precision_floor),),
        hinges=(Hinge(weight=1.0),),
        limits=(Limit(hi=p.T_bz_max, price=price),) * n
        + (Limit(lo=p.T_bz_min, price=price), Limit(hi=p.lime_max, price=price)),
    )  # fmt: skip


def make_cement_kiln_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``kiln_model``."""
    return Oracle(env, params, "cement_kiln", kiln_model)
