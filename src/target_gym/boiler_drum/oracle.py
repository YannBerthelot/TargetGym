"""The oracle (the MPC slot) for the boiler drum task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.4), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's own ``compute_velocity`` and integrator behind the fuel's
  and the feedwater's rate limits and lags: each move is an actuator's rate, a
  fraction of its limit, sent as the command that makes it (T1rr,
  ``oracles.rates``), with that command inside the box (T1); the turbine valve on
  its noise-free mean path, updated before the step as the env does, so the
  steam it passes follows the planned pressure;
- objective: the level's and the pressure's tracking terms as (e / e_floor)^2,
  exact since e_tol = 0 and p = 2, and the running cost, linear in the firing (T2);
- limits: the level and the pressure, abs(level) < level_trip and
  pressure_min < pressure < pressure_max on every visited state, elastic at
  the trip cost per unit (T3), backed off by the noise's reach (K11, in the
  widths file);
- dropped guards (T4): the void, circulation, pressure, enthalpy, determinant,
  volume and mass guards, which the limits keep off their edges, and the
  condensation flow's max(Q - q_dc subcool, 0), whose reach is not known, so
  every plan's argument is checked against it (K9).

The cold start holds both actuators where they stand.
"""

import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.boiler_drum.env import (
    STEAM_OU_THETA,
    circulation_flow,
    compute_velocity,
    drum_level,
    void_fraction,
    water_enthalpy,
)
from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel
from target_gym.oracles.rates import RateLimit
from target_gym.oracles.shooting import Limit
from target_gym.reward import trip_cost


def boiling_margin(x, Q, q_feed, p):
    """The argument of the condensation flow's max, env.py:399, at ``x``."""
    alpha = void_fraction(x[2], x[0], p)
    q_dc = circulation_flow(alpha, x[0], p)
    subcool = (1.0 - p.f_cd) * q_feed / q_dc * (
        water_enthalpy(x[0], p) - p.h_feedwater
    )  # fmt: skip
    return Q - q_dc * subcool


def boiler_model(env, p) -> TaskModel:
    assert float(p.e_tol) == 0.0 and float(p.tracking_exponent) == 2.0
    decay = 1.0 - STEAM_OU_THETA * p.delta_t  # the deviation's mean per step
    rate = RateLimit.of(env, p)

    def step(d, x, u, t):
        target, level_ref, deviation = d["task"]
        held = rate.reached(x[4:], u)
        Q, q_feed = (
            0.5 * (held[0] + 1.0) * p.Q_max, 0.5 * (held[1] + 1.0) * p.q_feed_max
        )  # fmt: skip
        q_demand = p.q_steam_nominal + deviation * decay ** (t + 1)
        x1, _ = integrate_dynamics(
            positions=x[:4], delta_t=p.delta_t, method=env.integration_method,
            compute_velocity=partial(
                compute_velocity, action=(Q, q_feed), q_demand=q_demand, params=p
            ),
        )  # fmt: skip
        level = drum_level(x1[1], x1[2], x1[3], level_ref, x1[0], p)
        e_p = x1[0] - target
        cost = (level / p.e_floor_level) ** 2 + (e_p / p.e_floor_pressure) ** 2
        cost = cost + p.running_weight * Q / p.c_hold
        y = jnp.stack([level, x1[0]])
        guard = boiling_margin(x1, Q, q_feed, p) / p.Q_max
        return jnp.concatenate([x1, held]), cost, y, jnp.stack([level, x1[0]]), (guard,)

    price = float(trip_cost(p))
    return TaskModel(
        nx=6, nu=2, step=step,
        state_x=lambda s: jnp.concatenate([
            jnp.stack([s.pressure, s.V_wt, s.m_sr, s.m_sd]), s.actuator_held,
        ]),
        data=lambda s: (s.target_pressure, s.level_ref, s.q_demand - p.q_steam_nominal),
        hold=lambda s: jnp.zeros(2, s.actuator_held.dtype),
        action=lambda s, u: rate.command(s.actuator_held, u),
        move_rows=lambda x, u, steps: rate.rows(x[4:], u, steps),
        tracked=lambda s: jnp.stack([s.level, s.pressure]),
        resolution=(
            float(p.level_precision_floor), float(p.pressure_precision_floor)
        ),
        limits=(
            Limit(-p.level_trip, p.level_trip, price),
            Limit(p.pressure_min, p.pressure_max, price),
        ),
    )  # fmt: skip


def make_boiler_drum_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``boiler_model``."""
    return Oracle(env, params, "boiler_drum", boiler_model)
