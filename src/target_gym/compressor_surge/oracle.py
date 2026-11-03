"""The oracle (the MPC slot) for the compressor, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.4, and decision S10), run by
``target_gym.oracles.controller.Oracle``:

- model: the env's step restated (``compressor_step``) with its own
  functions: the commands' clips removed and the moves bounded (T1), the
  drive and valve rate limits of each substep smoothed by softplus at their
  K5 widths (T5, ``data/oracle_widths/compressor_surge.json``), the
  consumers' deviation on its noise-free mean path, a^t times its measured
  value;
- objective: the header pressure's tracking term as (e / e_floor)^2, exact
  since e_tol = 0 and p = 2, and the running cost max(q - c_hold, 0) through
  epigraph rows on every visited state (T2);
- limit: the env's trip variable, the soft minimum of Phi over the step, at
  least 2W on every visited state, elastic at the trip cost per unit (T3);
- dropped guard: the characteristic's switch to its tangent past
  ``phi_join`` (T4, K9); every plan's substep Phi is checked against it.

The cold start holds the drive setpoint and the valve where they are.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.compressor_surge.env import (
    KPA,
    N_SETPOINT_BLOCKS,
    _substeps,
    compute_velocity,
    recycle_flow,
    suction_density,
)
from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel, raw_of, softplus_clip
from target_gym.oracles.shooting import Hinge, Limit
from target_gym.reward import trip_cost


def compressor_step(env, p, x, u, t, task, widths, deltas=None):
    """One env step from ``x`` = (m_c, dp, N, N_ramp, x) under the raw move
    ``u``: ``(x_next, phi_min, the substep Phis, the rate limits' arguments)``.
    ``widths`` are the drive's and the valve's (0.0 is the exact clip), and
    ``deltas[i, j]`` is added to site i's output at substep j (K5's probe)."""
    setpoints, demand, clock, drift = task
    n = _substeps(env.integration_method)
    h, rho = p.delta_t / n, suction_density(p)
    N_cmd = p.N_min + 0.5 * (u[0] + 1.0) * (p.N_max - p.N_min)  # T1
    x_cmd = 0.5 * (u[1] + 1.0)
    dev = drift * jnp.exp(-p.demand_theta * p.delta_t) ** t
    deltas = jnp.zeros((2, n)) if deltas is None else deltas

    def substep(carry, j):
        m_c, dp, N, N_ramp, xv = carry
        args = jnp.stack([N_cmd - N_ramp, x_cmd - xv])
        r_N, lo_x, hi_x = (
            p.drive_rate * h,
            p.valve_close_rate * h,
            p.valve_open_rate * h,
        )
        N_ramp = N_ramp + softplus_clip(args[0], -r_N, r_N, widths[0]) + deltas[0, j]
        xv = xv + softplus_clip(args[1], -lo_x, hi_x, widths[1]) + deltas[1, j]
        velocity = partial(
            compute_velocity, action=None, N_ramp=N_ramp, x=xv, dev=dev,
            demand_levels=demand, params=p,
        )  # fmt: skip
        new, _ = integrate_dynamics(
            positions=jnp.stack([m_c, dp, N, (clock + t) * p.delta_t + j * h]),
            delta_t=h, method="rk4_1", compute_velocity=velocity,
        )  # fmt: skip
        phi = new[0] / (rho * p.A_c * p.U_r * new[2])
        return (new[0], new[1], new[2], N_ramp, xv), (phi, args)

    out, (phis, args) = jax.lax.scan(substep, tuple(x), jnp.arange(n))
    phi_min = -p.soft_min_temperature * jax.nn.logsumexp(-phis / p.soft_min_temperature)
    return jnp.stack(out), phi_min, phis, args


def compressor_model(env, p) -> TaskModel:
    assert float(p.e_tol) == 0.0 and float(p.tracking_exponent) == 2.0

    def step(d, x, u, t):
        w = d["widths"]  # in name order: drive_rate, valve_rate
        x1, phi_min, phis, _ = compressor_step(env, p, x, u, t, d["task"], (w[0], w[1]))
        setpoints, _, clock, _ = d["task"]
        block = jnp.minimum(
            (clock + t + 1) // p.setpoint_block_steps, N_SETPOINT_BLOCKS - 1
        )
        dp = x1[1]
        cost = ((setpoints[block] - dp) / KPA / p.e_floor) ** 2
        power = recycle_flow(dp, x1[4], p) * dp / suction_density(p)
        y = jnp.stack([p.running_weight * (power - p.c_hold) / p.c_hold, phi_min])
        return x1, cost, y, (dp / KPA)[None], p.phi_join - phis

    return TaskModel(
        nx=5, nu=2, step=step,
        state_x=lambda s: jnp.stack([s.m_c, s.dp, s.N, s.N_ramp, s.x]),
        data=lambda s: (s.setpoint_levels, s.demand_levels, s.block_clock, s.demand_dev),
        hold=lambda s: jnp.stack([raw_of(s.N_ramp, p.N_min, p.N_max), raw_of(s.x, 0.0, 1.0)]),
        tracked=lambda s: (s.dp / KPA)[None],
        resolution=(float(p.precision_floor),),
        hinges=(Hinge(weight=1.0),),
        limits=(Limit(lo=2.0 * p.W, price=float(trip_cost(p))),),
    )  # fmt: skip


def make_compressor_surge_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``compressor_model``."""
    return Oracle(env, params, "compressor_surge", compressor_model)
