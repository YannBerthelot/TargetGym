"""The oracle (the MPC slot) for the glass furnace task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.3), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's step restated (``glass_step``) with its own functions,
  all four regenerator nodes, the fuel pipeline and the reversal cycle
  included: the raw move's clip removed and the move bounded instead (T1);
  the batch blanket's guards on what the right-hand side and the flame read
  smoothed by one softplus at the K5 width (T5,
  ``data/oracle_widths/glass_furnace.json``); the pull disturbance on its
  noise-free mean path, updated before the step as the env does;
- objective: the crown's tracking term as (e / e_floor)^2, exact since
  e_tol = 0 and p = 2, and the running cost max(fuel - c_hold, 0) through
  epigraph rows on every visited state (T2);
- limits: the crown, melt and working-end bands on every visited state,
  elastic at the trip cost per unit (T3), backed off by the planner's error
  on them (K11, in the widths file);
- dropped guards (T4): the flame's clip; the heat to the batch, which the
  bands keep positive; and the stored blanket's floor at 0 (env.py:806),
  checked on every plan: a softplus there would add its gap to the state
  every step, where in the right-hand side it costs it once.

The cold start holds the last fuel command.
"""

import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.glass_furnace.env import (
    M_PULL_AR_RHO,
    N_REGEN_NODES,
    compute_velocity,
    firing_fraction,
    get_target_from_schedule,
    regenerator_diagnostics,
    reversal_phase,
    solve_T_gas,
)
from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel, raw_of, softplus_clip
from target_gym.oracles.shooting import Hinge, Limit
from target_gym.reward import trip_cost

#: x = (T_crown, T_melt, T_work, m_batch, T_rA, T_rB, T_gas, fuel_pipeline).
BATCH, GAS, PIPE = 3, 4 + 2 * N_REGEN_NODES, 5 + 2 * N_REGEN_NODES


def velocity(position, width, delta, p, **plant):
    """``compute_velocity`` on the blanket its guards leave, smoothed (T5):
    a mass inside (0, m_batch_full) passes them exactly."""
    blanket = softplus_clip(position[BATCH], 0.0, p.m_batch_full, width) + delta
    return compute_velocity(position.at[BATCH].set(blanket), params=p, **plant)


def glass_step(env, p, x, u, t, task, width, delta=0.0):
    """One env step from ``x`` under the raw move ``u``: ``(x_next, the fuel
    burnt)``. ``width`` is the blanket's (0.0 is the exact clip) and ``delta``
    is added to its output at every evaluation in the step (K5's probe)."""
    _, start, disturbance = task
    now = start + t
    command = p.fuel_min + 0.5 * (u[0] + 1.0) * (p.fuel_max - p.fuel_min)  # T1
    fuel = x[PIPE] * firing_fraction(now, p)
    pull = jnp.maximum(p.m_pull + disturbance * M_PULL_AR_RHO ** (t + 1), 0.1)
    phase = reversal_phase(now, p)
    plant = x[:GAS]
    cover = softplus_clip(x[BATCH], 0.0, p.m_batch_full, width) + delta
    melt_open = 1.0 - p.batch_shield * cover / p.m_batch_full
    T_air, _ = regenerator_diagnostics(plant, x[GAS], fuel, phase, p)
    T_gas = solve_T_gas(x[GAS], x[0], x[1], x[2], T_air, fuel, melt_open, p)
    new, _ = integrate_dynamics(
        positions=plant, delta_t=p.delta_t, method=env.integration_method,
        compute_velocity=partial(
            velocity, width=width, delta=delta, p=p, action=fuel, m_pull=pull,
            phase=phase, T_gas=T_gas, position_time=now,
        ),
    )  # fmt: skip
    pipeline = jnp.stack([x[PIPE + 1], command])
    return jnp.concatenate([new, T_gas[None], pipeline]), fuel


def glass_model(env, p) -> TaskModel:
    assert float(p.e_tol) == 0.0 and float(p.tracking_exponent) == 2.0

    def step(d, x, u, t):
        task = d["task"]
        x1, fuel = glass_step(env, p, x, u, t, task, d["widths"][0])
        target = get_target_from_schedule(task[0], task[1] + t + 1, p)
        cost = ((target - x1[0]) / p.e_floor) ** 2
        y = jnp.stack([
            p.running_weight * (fuel - p.c_hold) / p.c_hold, x1[0], x1[1], x1[2],
        ])  # fmt: skip
        return x1, cost, y, x1[:1], (x1[BATCH],)

    price = float(trip_cost(p))
    glass = Limit(p.T_glass_min, p.T_glass_max, price)
    return TaskModel(
        nx=PIPE + 2, nu=1, step=step,
        state_x=lambda s: jnp.concatenate([
            jnp.stack([s.T_crown, s.T_melt, s.T_work, s.m_batch]),
            s.T_rA, s.T_rB, s.T_gas[None], s.fuel_pipeline,
        ]),
        data=lambda s: (s.target_schedule, s.time, s.m_pull_disturbance),
        hold=lambda s: raw_of(s.fuel_pipeline[-1], p.fuel_min, p.fuel_max)[None],
        tracked=lambda s: s.T_crown[None],
        resolution=(float(p.precision_floor),),
        hinges=(Hinge(weight=1.0),),
        limits=(Limit(p.T_crown_min, p.T_crown_max, price), glass, glass),
    )  # fmt: skip


def make_glass_furnace_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``glass_model``."""
    return Oracle(env, params, "glass_furnace", glass_model)
