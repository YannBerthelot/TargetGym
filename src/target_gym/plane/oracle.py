"""The oracle (the MPC slot) for the 2D aircraft, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.2), run by ``target_gym.oracles.controller.Oracle``:

- model: ``compute_next_state`` restated (``plane_step``), the plant's fixed
  inner loop at 4 Hz included, on the equations of ``oracle_model`` with its
  sites smoothed at the K5 widths (T5, ``data/oracle_widths``) and its guards
  dropped (T4); the raw moves' clips removed and the moves bounded (T1); the
  gust on its noise-free decay from the measured one;
- objective: the altitude error as (e / e_floor)^2, exact since e_tol = 0 and
  p = 2, and the speed band through epigraph rows on every visited state (T2);
- limits: the altitude band, the load factor at each inner sub-step, VMO and MMO
  on every visited state, elastic at the trip cost per unit (T3), backed off by
  the planner's error on them (K11, in the widths file).

The cold start holds the throttle and the attitude where they are.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel, raw_of
from target_gym.oracles.shooting import Hinge, Limit
from target_gym.plane import oracle_model as om
from target_gym.plane.dynamics import (
    INNER_STEPS,
    PITCH_DEMAND_MAX,
    STICK_TRAVEL,
    airspeed_scaler,
    compute_air_density_from_altitude,
    compute_thrust_output,
    protected_pitch_demand,
    scheduled_airspeed,
    total_wind_2d,
)
from target_gym.plane.env import commanded_altitude
from target_gym.reward import trip_cost

#: Limits in the order of ``plane_step``'s outputs after its hinge.
N_LOADS = INNER_STEPS


def accelerate(velocities, positions, thrust, stick, wind, mass, p, w, d):
    """``compute_acceleration`` on the planner's air, its guards dropped."""
    x_dot, z_dot, q = velocities
    _, z, theta = positions
    air_x, air_z = x_dot - wind[0], z_dot - wind[1]
    gamma = jnp.arctan2(air_z, air_x)
    alpha = theta - gamma
    speed = jnp.sqrt(air_x**2 + air_z**2)
    mach = speed / om.sound_speed(z, w, d)
    rho = compute_air_density_from_altitude(z)
    a = om.airframe(alpha, speed, mach, rho, q, stick, p, w, d)
    sin_g, cos_g = jnp.sin(gamma), jnp.cos(gamma)
    f_x = thrust * jnp.cos(theta) - a.drag * cos_g - a.lift * sin_g
    f_z = thrust * jnp.sin(theta) - a.drag * sin_g + a.lift * cos_g
    f_z = f_z - mass * p.gravity
    accel = jnp.stack([f_x / mass, f_z / mass, a.moment / p.I])
    return accel, (a.lift, mach, a.cl_peak, accel[2], alpha)


def plane_step(p, x, u, t, gust, method, widths, deltas=om.NO_DELTA):
    """One task step of the 2D aircraft: ``(x_next, outputs)``, ``x`` = (z, x_dot,
    z_dot, theta, q, power, stick, pitch integral, fuel). ``outputs`` hold the
    loads at the inner sub-steps, the airspeed the trips read, the Mach number and
    the guards' arguments."""
    z, x_dot, z_dot, theta, q, power, stick, integ, fuel = x
    dt = p.delta_t / INNER_STEPS
    decay = (1.0 - p.turbulence_theta * p.delta_t) ** (t + 1)
    gust_x, gust_z = gust[0] * decay, gust[1] * decay
    wind = total_wind_2d(z, gust_x, gust_z, p)
    mass = p.initial_mass - (p.initial_fuel_quantity - fuel)
    lever, demand = 0.5 * (u[0] + 1.0), u[1] * PITCH_DEMAND_MAX
    gains = p.inner_loop
    w = om.sites(widths)

    def inner(carry, _):
        velocities, positions, power, stick, integ, burned = carry
        vx, vz, rate = velocities
        height, attitude = positions[1], positions[2]
        air_speed = jnp.sqrt((vx - wind[0]) ** 2 + (vz - wind[1]) ** 2)
        rho = compute_air_density_from_altitude(height)
        eas = air_speed * jnp.sqrt(rho / p.air_density_at_sea_level)
        scaler = airspeed_scaler(eas, gains)
        # The plant's normal-law protection (``protected_pitch_demand``).
        held = protected_pitch_demand(
            demand, jnp.arctan2(vz - wind[1], vx - wind[0]), air_speed, height,
            mass, p,
        )  # fmt: skip
        surface, integ = om.attitude(
            gains.Kp_theta, gains.Ki_theta, gains.Kd_theta, held - attitude, rate,
            integ, scaler, dt, w.surface_clip, deltas.surface_clip,
        )  # fmt: skip
        stick = om.surface_step(
            surface * STICK_TRAVEL, stick, dt, p.stick_response_rate,
            p.surface_rate_limit, w.surface_rate, deltas.surface_rate,
        )  # fmt: skip
        power = om.engine(lever, power, dt, p)
        thrust = compute_thrust_output(
            power=power,
            thrust_output_at_sea_level=p.thrust_output_at_sea_level,
            rho=rho,
            M=air_speed / om.sound_speed(height, w, deltas),
        )
        velocities, positions, metrics = integrate_dynamics(
            velocities=velocities,
            positions=positions,
            delta_t=dt,
            compute_acceleration=partial(
                accelerate,
                thrust=thrust,
                stick=stick,
                wind=wind,
                mass=mass,
                p=p,
                w=w,
                d=deltas,
            ),
            method=method,
        )
        burned = burned + p.specific_fuel_consumption * 1e-3 * thrust * dt
        lift, mach, peak, pitch_accel, alpha = jax.tree_util.tree_map(
            lambda m: m[-1], metrics
        )
        load = lift / (mass * p.gravity)
        guards = jnp.stack([
            p.M_crit - mach,
            om.SECTION_CAP - peak,
            om.ANGULAR_CLIP - jnp.abs(pitch_accel),
            jnp.pi - jnp.abs(alpha),
            eas - gains.scaling_speed / gains.scale_max,
        ])  # fmt: skip
        return (velocities, positions, power, stick, integ, burned), (load, guards)

    start = (
        jnp.stack([x_dot, z_dot, q]), jnp.stack([jnp.zeros_like(z), z, theta]),
        power, stick, integ, jnp.zeros_like(z),
    )  # fmt: skip
    end, (loads, guards) = jax.lax.scan(inner, start, None, length=INNER_STEPS)
    (x_dot1, z_dot1, q1), (_, z1, theta1) = end[0], end[1]
    power1, stick1, integ1, burned = end[2:]
    fuel1 = fuel - burned
    x1 = jnp.stack([z1, x_dot1, z_dot1, theta1, q1, power1, stick1, integ1, fuel1])
    # The trips read the state reached, in the wind it stays in.
    wind1 = total_wind_2d(z1, gust_x, gust_z, p)
    air_speed = jnp.sqrt((x_dot1 - wind1[0]) ** 2 + (z_dot1 - wind1[1]) ** 2)
    rho = compute_air_density_from_altitude(z1)
    eas = air_speed * jnp.sqrt(rho / p.air_density_at_sea_level)
    mach = air_speed / om.sound_speed(z1, w, deltas)
    return x1, (loads, eas, mach, jnp.append(jnp.min(guards, axis=0), fuel1))


def state_x(s):
    return jnp.stack([
        s.z, s.x_dot, s.z_dot, s.theta, s.theta_dot, s.power, s.stick,
        s.pitch_integ, s.fuel,
    ])  # fmt: skip


def plane_model(env, p) -> TaskModel:
    if bool(p.speed_band_cost) or float(p.e_tol) != 0.0:
        # ``plane`` since Amendment 2 scores bands (plane/PHYSICS.md); this
        # model plans the earlier quadratic cost. Its room is "not measured"
        # (charter rule 9) until the oracle is rebuilt on the new score.
        raise NotImplementedError(
            "the 2D oracle plans the pre-Amendment-2 cost; "
            f"{env.registry_name}'s band score needs the oracle rebuilt"
        )
    assert float(p.tracking_exponent) == 2.0
    price = float(trip_cost(p))
    method = env.integration_method

    def step(d, x, u, t):
        base, start, gust = d["task"]
        x1, (loads, eas, mach, guards) = plane_step(
            p, x, u, t, gust, method, d["widths"]
        )
        target = commanded_altitude(base, start + t + 1, p)
        cost = ((target - x1[0]) / p.e_floor) ** 2
        speed = jnp.sqrt(x1[1] ** 2 + x1[2] ** 2)
        band = speed - scheduled_airspeed(x1[0], p.target_cas, p.target_mach)
        y = jnp.concatenate([band[None], x1[:1], loads, eas[None], mach[None]])
        return x1, cost, y, x1[:1], guards

    def hold(s):
        lever = (s.power - p.idle_power) / (1.0 - p.idle_power)
        return jnp.stack([
            raw_of(lever, 0.0, 1.0),
            raw_of(s.theta, -PITCH_DEMAND_MAX, PITCH_DEMAND_MAX),
        ])  # fmt: skip

    return TaskModel(
        nx=9, nu=2, step=step, state_x=state_x, hold=hold,
        data=lambda s: (
            s.base_target_altitude, s.time, jnp.stack([s.gust_x, s.gust_z])
        ),
        tracked=lambda s: jnp.stack([s.z]),
        resolution=(float(p.precision_floor),),
        hinges=(
            Hinge(float(p.running_weight / p.c_hold), True, float(p.c_hold), 1),
        ),
        limits=(
            Limit(float(p.min_alt), float(p.max_alt), price),
            *(
                Limit(float(p.load_factor_min), float(p.load_factor_max), price)
                for _ in range(N_LOADS)
            ),
            Limit(hi=float(p.vmo_eas), price=price),
            Limit(hi=float(p.mmo), price=price),
        ),
    )  # fmt: skip


def make_plane_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``plane_model``."""
    return Oracle(env, params, env.registry_name, plane_model)
