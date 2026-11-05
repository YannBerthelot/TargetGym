"""The oracle (the MPC slot) for the HVAC task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.4), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's own ``compute_velocity`` and integrator, with the raw
  move's clip removed and the move bounded instead (T1); the weather's
  deviation on its noise-free mean path, updated before the step as the env
  does; the clock, sun, occupancy and setpoint precomputed along the plan;
- objective: the gas bill, which is smooth, and the comfort hinge through
  epigraph rows on every visited state (T2): the air's excess over the band
  while occupied and its shortfall below it, the shortfall alone at setback,
  each squared, exact since at most one is positive;
- limit: the air temperature band, T_air_min < T_air < T_air_max on every
  visited state, elastic at the trip cost per unit (T3);
- no smoothed site (T5) and no dropped guard (T4).

The reward is in euros, so the tie-break's budget is what half a resolution
costs in one step in euros (G10). The cold start holds the emitter where the
reset put it.
"""

import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.hvac.env import (
    WEATHER_OU_THETA,
    compute_velocity,
    internal_gain,
    is_occupied,
    outdoor_temperature,
    scheduled_setpoint,
    solar_gain,
    solve_air_and_surface,
    split_gains,
)
from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel, raw_of
from target_gym.oracles.shooting import Hinge, Limit
from target_gym.reward import trip_cost


def hvac_model(env, p) -> TaskModel:
    assert bool(p.setback_lower_bound_only) and float(p.tracking_exponent) == 2.0
    hours = p.delta_t / 3600.0
    decay = 1.0 - WEATHER_OU_THETA * p.delta_t  # the deviation's mean per step
    band = p.comfort_tolerance

    def step(d, x, u, t):
        start, occupied_setpoint, deviation = d["task"]
        Q = 0.5 * (u[0] + 1.0) * p.Q_heat_max  # T1
        dev = deviation * decay ** (t + 1)
        x1, _ = integrate_dynamics(
            positions=x, delta_t=p.delta_t, method=env.integration_method,
            compute_velocity=partial(
                compute_velocity, action=Q, time=start + t, weather_dev=dev, params=p
            ),
        )  # fmt: skip
        now = start + t + 1
        phi_ia, phi_st, _ = split_gains(internal_gain(now, p), solar_gain(now, p), p)
        T_air, _ = solve_air_and_surface(
            x1[0], outdoor_temperature(now, dev, p), x1[1], phi_st, phi_ia, p
        )
        e = T_air - scheduled_setpoint(now, occupied_setpoint, p)
        occupied = is_occupied(now, p)
        y = jnp.stack([
            jnp.where(occupied, e - band, -band),
            jnp.where(occupied, -e - band, -e),
            T_air,
        ])  # fmt: skip
        return x1, p.gas_price * x1[1] / 1000.0 * hours, y, T_air[None], ()

    weight = float(p.comfort_price * hours)
    return TaskModel(
        nx=2, nu=1, step=step,
        state_x=lambda s: jnp.stack([s.T_mass, s.Q_emitter]),
        data=lambda s: (s.time, s.setpoint_occupied, s.weather_dev),
        hold=lambda s: raw_of(s.Q_emitter, 0.0, p.Q_heat_max)[None],
        tracked=lambda s: s.T_air[None],
        resolution=(float(p.precision_floor),),
        hinges=(Hinge(weight=weight, power=2), Hinge(weight=weight, power=2)),
        limits=(Limit(p.T_air_min, p.T_air_max, float(trip_cost(p))),),
        tie_budget=weight * (0.5 * float(p.precision_floor)) ** 2,
    )  # fmt: skip


def make_hvac_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``hvac_model``."""
    return Oracle(env, params, "hvac", hvac_model)
