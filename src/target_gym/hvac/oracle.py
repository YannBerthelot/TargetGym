"""The oracle (the MPC slot) for the HVAC task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.4), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's own ``compute_velocity`` and integrator on the mass and
  slab core, the move a supply water temperature inside the valve's range
  (T1); the weather's deviation on its noise-free mean path, updated before
  the step as the env does; the clock, sun and gains along the plan;
- objective: the heat pump's electricity at the tariff, its slab heat
  smoothed at the heating-only edge, and the comfort hinges through epigraph
  rows on every occupied state (T2): the room below and above the band and the
  floor above its limit, each linear in kelvin-hours at the work-loss price;
- no limit trips (T3), no smoothed site (T5) and no dropped guard (T4).

Frost protection is not planned: the band keeps the room far above it. Not
yet run on this plant (rule 9's room is "not measured").
"""

import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.hvac.env import (
    WEATHER_OU_THETA,
    compute_velocity,
    electricity_price,
    floor_surface_temperature,
    heat_pump_cop,
    internal_gain,
    is_occupied,
    outdoor_temperature,
    slab_conductances,
    solar_gain,
    solve_air_and_surface,
    split_gains,
)
from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel, raw_of
from target_gym.oracles.shooting import Hinge


def hvac_model(env, p) -> TaskModel:
    hours = p.delta_t / 3600.0
    decay = 1.0 - WEATHER_OU_THETA * p.delta_t  # the deviation's mean per step
    H_water = slab_conductances(p)["H_water"]

    def step(d, x, u, t):
        start, deviation = d["task"]
        T_sup = p.T_supply_min + 0.5 * (u[0] + 1.0) * (p.T_supply_max - p.T_supply_min)
        dev = deviation * decay ** (t + 1)
        x1, _ = integrate_dynamics(
            positions=x, delta_t=p.delta_t, method=env.integration_method,
            compute_velocity=partial(
                compute_velocity, action=T_sup, time=start + t, weather_dev=dev,
                params=p,
            ),
        )  # fmt: skip
        now = start + t + 1
        phi_ia, phi_st, _ = split_gains(internal_gain(now, p), solar_gain(now, p), p)
        T_air, _ = solve_air_and_surface(
            x1[0], x1[1], outdoor_temperature(now, dev, p), phi_st, phi_ia, p
        )
        T_floor = floor_surface_temperature(x1[1], T_air, p)
        gap = T_sup - x1[1]
        heat = H_water * 0.5 * (gap + jnp.sqrt(gap * gap + 1e-4))
        occupied = is_occupied(now, p)
        y = jnp.where(
            occupied,
            jnp.stack(
                [p.band_low - T_air, T_air - p.band_high, T_floor - p.floor_surface_max]
            ),
            -1.0,
        )
        kwh = heat / heat_pump_cop(T_sup, p) / 1000.0 * hours
        return x1, electricity_price(start + t, p) * kwh, y, T_air[None], ()

    weight = float(p.comfort_price * hours)
    return TaskModel(
        nx=2, nu=1, step=step,
        state_x=lambda s: jnp.stack([s.T_mass, s.T_core]),
        data=lambda s: (s.time, s.weather_dev),
        hold=lambda s: raw_of(s.T_supply, p.T_supply_min, p.T_supply_max)[None],
        tracked=lambda s: s.T_air[None],
        resolution=(float(p.precision_floor),),
        hinges=(Hinge(weight=weight), Hinge(weight=weight), Hinge(weight=weight)),
        tie_budget=weight * 0.5 * float(p.precision_floor),
    )  # fmt: skip


def make_hvac_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``hvac_model``."""
    return Oracle(env, params, "hvac", hvac_model)
