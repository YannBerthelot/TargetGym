"""The oracle (the MPC slot) for the HVAC task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.4), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's ``compute_velocity`` and integrator on the mass and slab
  core (``velocity``), the move a supply water temperature inside the valve's
  range (T1); the weather's deviation, updated before the step as the env
  does: on its noise-free mean path in the registry's receding slot, and on
  the episode's own path for the ceiling (``ceiling_data``), which knows the
  weather ahead as a ceiling may; the clock, sun and gains along the plan;
- objective: the heat pump's electricity at the tariff and the comfort
  hinges through epigraph rows on every occupied state (T2): the room below
  and above the band and the floor above its limit, each linear in
  kelvin-hours at the work-loss price;
- smoothed site (T5): the slab heat's heating-only edge, ``max(T_sup -
  T_core, 0)`` in ``slab_heat``, which enters the dynamics and so cannot be an
  epigraph row: K2's softplus at the solve's width, the exact edge at width 0,
  where the planned cost of a plan is the env's
  (``tests/oracles/test_thermal_energy_oracles.py``). Water at the valve's
  lowest setting, which the env counts as off, is below the core, so it gives
  no heat either way;
- no limit trips (T3) and no dropped guard (T4).

Frost protection is not planned: the band keeps the room far above it.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.hvac.env import (
    WEATHER_OU_THETA,
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
    zone_conductances,
)
from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel, raw_of, softplus_clip
from target_gym.oracles.shooting import Hinge


def heat(T_supply, T_core, p, width):
    """``slab_heat`` with its heating-only edge smoothed at ``width`` (K)."""
    H_water = slab_conductances(p)["H_water"]
    return H_water * softplus_clip(T_supply - T_core, 0.0, jnp.inf, width)


def velocity(position, action, time, weather_dev, width, p):
    """``compute_velocity`` with the slab heat of ``heat``."""
    T_mass, T_core = position[0], position[1]
    c, s = zone_conductances(p), slab_conductances(p)
    T_out = outdoor_temperature(time, weather_dev, p)
    phi_ia, phi_st, phi_m = split_gains(internal_gain(time, p), solar_gain(time, p), p)
    T_air, T_surface = solve_air_and_surface(T_mass, T_core, T_out, phi_st, phi_ia, p)
    dT_mass = (
        c["H_tr_ms"] * (T_surface - T_mass) + c["H_tr_em"] * (T_out - T_mass) + phi_m
    ) / c["C_m"]
    dT_core = (heat(action, T_core, p, width) + s["H_slab"] * (T_air - T_core)) / s[
        "C_slab"
    ]
    return jnp.stack([dT_mass, dT_core]), None


def hvac_model(env, p) -> TaskModel:
    hours = p.delta_t / 3600.0
    decay = 1.0 - WEATHER_OU_THETA * p.delta_t  # the deviation's mean per step

    def step(d, x, u, t):
        start, deviation, path = d["task"]
        widths = jnp.asarray(d["widths"])
        width = widths[0] if widths.shape[0] else jnp.zeros((), x.dtype)
        T_sup = p.T_supply_min + 0.5 * (u[0] + 1.0) * (p.T_supply_max - p.T_supply_min)
        if path is None:
            dev = deviation * decay ** (t + 1)
        else:  # the episode's own weather, t + 1 steps after the solve's start
            dev = path[jnp.minimum(t, path.shape[0] - 1)]
        x1, _ = integrate_dynamics(
            positions=x, delta_t=p.delta_t, method=env.integration_method,
            compute_velocity=partial(
                velocity, action=T_sup, time=start + t, weather_dev=dev,
                width=width, p=p,
            ),
        )  # fmt: skip
        now = start + t + 1
        phi_ia, phi_st, _ = split_gains(internal_gain(now, p), solar_gain(now, p), p)
        T_air, _ = solve_air_and_surface(
            x1[0], x1[1], outdoor_temperature(now, dev, p), phi_st, phi_ia, p
        )
        T_floor = floor_surface_temperature(x1[1], T_air, p)
        occupied = is_occupied(now, p)
        y = jnp.where(
            occupied,
            jnp.stack(
                [p.band_low - T_air, T_air - p.band_high, T_floor - p.floor_surface_max]
            ),
            -1.0,
        )
        # The env scores the step's heat on the core it ends at.
        kwh = heat(T_sup, x1[1], p, width) / heat_pump_cop(T_sup, p) / 1000.0 * hours
        return x1, electricity_price(start + t, p) * kwh, y, T_air[None], ()

    weight = float(p.comfort_price * hours)
    return TaskModel(
        nx=2, nu=1, step=step,
        state_x=lambda s: jnp.stack([s.T_mass, s.T_core]),
        data=lambda s: (s.time, s.weather_dev, None),
        hold=lambda s: raw_of(s.T_supply, p.T_supply_min, p.T_supply_max)[None],
        tracked=lambda s: s.T_air[None],
        resolution=(float(p.precision_floor),),
        hinges=(Hinge(weight=weight), Hinge(weight=weight), Hinge(weight=weight)),
        tie_budget=weight * 0.5 * float(p.precision_floor),
    )  # fmt: skip


def ceiling_data(env, p, state, step_keys):
    """The data of a plan from ``state`` over the steps of ``step_keys``: the
    weather's deviation after every step, from the episode's step keys. The
    deviation does not depend on the moves, so it is read off a run of the env
    at any move (here the reset's supply)."""

    def advance(s, k):
        s1 = env.step_env(k, s, jnp.zeros(1, jnp.float32), p)[1]
        return s1, s1.weather_dev

    path = jax.lax.scan(advance, state, step_keys)[1]
    return state.time, state.weather_dev, path


def make_hvac_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``hvac_model``."""
    return Oracle(env, params, "hvac", hvac_model)
